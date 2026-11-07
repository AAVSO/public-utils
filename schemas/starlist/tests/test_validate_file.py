import json
import os
import shutil
import subprocess
import warnings
from pathlib import Path

import pytest
from pydantic import ValidationError

from aavso_starlist_schema import (
    _MIGRATIONS,
    SCHEMA_VERSION,
    NewerSchemaVersionError,
    SchemaMigrationWarning,
    StarList,
    StarListSet,
    UnsupportedSchemaVersionError,
    _generation,
    _report_validation,
    cli,
    validate_file,
)

# One directory per migration step, named <old generation>_to_<new generation>,
# holding a before.json / after.json pair; before.json is a real file of the
# older generation.
MIGRATION_DATA = Path(__file__).parent / "data" / "migrations"


def _older_file(generation):
    """
    Locate the sample file written against an older generation.

    Parameters
    ----------
    generation : str
        A generation with a migration, a key of ``_MIGRATIONS``.

    Returns
    -------
    pathlib.Path
        ``tests/data/migrations/<generation>_to_<target>/before.json``.
    """
    target_generation = _generation(_MIGRATIONS[generation].target)
    return MIGRATION_DATA / f"{generation}_to_{target_generation}" / "before.json"


def _write(directory, name, data):
    """
    Write raw starlist-set data to a JSON file.

    Parameters
    ----------
    directory : pathlib.Path
        Where to write; a ``tmp_path``.
    name : str
        The file name, suffix included.
    data : dict
        The data to dump as JSON.

    Returns
    -------
    pathlib.Path
        The file written.
    """
    path = directory / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _current_data():
    """
    Build raw data for a star list set at the current version.

    Returns
    -------
    dict
        One star list made of the models' example values, as ``json.loads``
        would give it. (``StarListSet.from_examples()`` has no star lists.)
    """
    star_list = StarList.from_examples().model_dump(mode="json")
    return {"schema_version": SCHEMA_VERSION, "star_lists": [star_list]}


@pytest.fixture
def current_file(tmp_path):
    """A file valid for the current schema version."""
    return _write(tmp_path, "current.json", _current_data())


@pytest.fixture
def invalid_file(tmp_path):
    """A current-version file whose star list lacks its observer."""
    data = _current_data()
    del data["star_lists"][0]["observer"]
    return _write(tmp_path, "invalid.json", data)


@pytest.fixture
def not_json_file(tmp_path):
    """A file that is not JSON at all."""
    path = tmp_path / "notes.json"
    path.write_text("this is not JSON", encoding="utf-8")
    return path


@pytest.fixture
def older_release_of_current_generation():
    """
    An older released version of ``SCHEMA_VERSION``'s generation.

    Returns
    -------
    str
        ``SCHEMA_VERSION`` with its last component set to 0, which must be a
        released version; the test is skipped while ``SCHEMA_VERSION`` is the
        first release of its generation.
    """
    *generation, patch = SCHEMA_VERSION.split(".")
    if patch == "0":
        pytest.skip(f"{SCHEMA_VERSION} is the first release of its generation")
    return ".".join([*generation, "0"])


@pytest.fixture
def older_release_file(tmp_path, older_release_of_current_generation):
    """A current-generation file stating an older release of the generation."""
    data = _current_data()
    data["schema_version"] = older_release_of_current_generation
    return _write(tmp_path, "older_release.json", data)


def test_validate_file_returns_a_current_model(current_file):
    # A current-version file reads without a warning (pytest treats an
    # unexpected SchemaMigrationWarning as an error) and states the current
    # version. Both a Path and a str name the file.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        from_path = validate_file(current_file)
        from_str = validate_file(str(current_file))

    assert isinstance(from_path, StarListSet)
    assert from_path.schema_version == SCHEMA_VERSION
    assert from_path == from_str
    assert from_path == StarListSet.model_validate_json(current_file.read_text())


@pytest.mark.parametrize("generation", _MIGRATIONS)
def test_validate_file_upgrades_an_older_file_with_a_warning(generation):
    # By default a file of an older generation is upgraded, as StarListSet
    # does it, with the usual warning.
    path = _older_file(generation)
    with pytest.warns(SchemaMigrationWarning):
        result = validate_file(path)

    assert isinstance(result, StarListSet)
    assert result.schema_version == SCHEMA_VERSION
    with pytest.warns(SchemaMigrationWarning):
        assert result == StarListSet.model_validate_json(path.read_text())


@pytest.mark.parametrize("generation", _MIGRATIONS)
def test_validate_file_as_written_returns_the_frozen_model(generation):
    # With as_written the file is checked against its own generation's frozen
    # model and not upgraded, so nothing warns and the file's own version is
    # kept.
    path = _older_file(generation)
    stated = json.loads(path.read_text())["schema_version"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = validate_file(path, as_written=True)

    assert isinstance(result, _MIGRATIONS[generation].model)
    assert not isinstance(result, StarListSet)
    assert result.schema_version == stated


def test_validate_file_as_written_on_a_current_file(current_file):
    # A current file is a StarListSet either way.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = validate_file(current_file, as_written=True)

    assert isinstance(result, StarListSet)
    assert result.schema_version == SCHEMA_VERSION


def test_validate_file_as_written_reads_an_older_release_as_current(older_release_file):
    # Within a generation there is nothing to migrate: a file stating an
    # older release of the current generation is a StarListSet, which states
    # the current version, and nothing warns.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = validate_file(older_release_file, as_written=True)

    assert isinstance(result, StarListSet)
    assert result.schema_version == SCHEMA_VERSION


@pytest.mark.parametrize("text", ["[]", "null"])
def test_non_object_json_raises_validation_error(tmp_path, text):
    # JSON that is not an object has no schema_version to read; both modes
    # give pydantic's own error rather than a crash in the version lookup.
    path = tmp_path / "not_an_object.json"
    path.write_text(text, encoding="utf-8")

    for as_written in (False, True):
        with pytest.raises(ValidationError) as info:
            validate_file(path, as_written=as_written)
        assert info.value.errors()[0]["type"] == "model_type"


def test_invalid_file_raises_validation_error_naming_the_field(invalid_file):
    # A file that does not match its schema is the usual pydantic error, with
    # the location of the problem, in both modes.
    with pytest.raises(ValidationError) as info:
        validate_file(invalid_file)
    assert "observer" in str(info.value)
    assert info.value.errors()[0]["loc"] == ("star_lists", 0, "observer")

    with pytest.raises(ValidationError):
        validate_file(invalid_file, as_written=True)


def test_not_json_raises_json_decode_error(not_json_file):
    # A file that is not JSON fails in the standard library, before the schema
    # is looked at.
    with pytest.raises(json.JSONDecodeError):
        validate_file(not_json_file)
    with pytest.raises(json.JSONDecodeError):
        validate_file(not_json_file, as_written=True)


def test_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        validate_file(tmp_path / "missing.json")


def test_version_errors_propagate_unwrapped(tmp_path, next_generation_version):
    # Version problems come out as the SchemaVersionErrors StarListSet raises.
    newer = _write(tmp_path, "newer.json", {"schema_version": next_generation_version, "star_lists": []})
    unsupported = _write(tmp_path, "banana.json", {"schema_version": "banana", "star_lists": []})

    for as_written in (False, True):
        with pytest.raises(NewerSchemaVersionError):
            validate_file(newer, as_written=as_written)
        with pytest.raises(UnsupportedSchemaVersionError):
            validate_file(unsupported, as_written=as_written)


def test_validate_command_reports_a_valid_file(current_file, capsys):
    # The validate command prints one line to stdout, nothing to stderr,
    # and writes no file.
    _report_validation(str(current_file))

    captured = capsys.readouterr()
    assert captured.out == f"{current_file}: valid (schema version {SCHEMA_VERSION})\n"
    assert captured.err == ""
    assert list(current_file.parent.iterdir()) == [current_file]


def test_validate_command_does_not_replace_the_suffix(tmp_path, capsys):
    # The file to validate is used exactly as given, whatever its suffix; the
    # suffix replacement belongs to write-schema only.
    path = _write(tmp_path, "starlists.txt", _current_data())
    _report_validation(str(path))

    captured = capsys.readouterr()
    assert captured.out.startswith(f"{path}: valid")
    assert list(tmp_path.iterdir()) == [path]


def test_validate_command_invalid_file_exits_1(invalid_file, capsys):
    # An invalid file is reported on stderr, naming the field, with exit status 1.
    with pytest.raises(SystemExit) as info:
        _report_validation(str(invalid_file))

    assert info.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith(f"{invalid_file}: invalid\n")
    assert "observer" in captured.err


@pytest.mark.parametrize("generation", _MIGRATIONS)
def test_validate_command_reports_an_upgrade(generation, capsys):
    # An older file is valid, upgraded, and the migration warning is printed
    # rather than issued (pytest would turn it into an error).
    path = _older_file(generation)
    stated = json.loads(path.read_text())["schema_version"]
    _report_validation(str(path))

    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert lines[0] == f"{path}: valid (schema version {SCHEMA_VERSION})"
    assert len(lines) == 2
    assert f"schema_version was {stated!r}; upgraded to {SCHEMA_VERSION!r}" in lines[1]
    assert captured.err == ""


@pytest.mark.parametrize("generation", _MIGRATIONS)
def test_validate_command_as_written_reports_the_files_own_version(generation, capsys):
    # With as_written an older file reports the version it states, and there
    # is no upgrade to mention.
    path = _older_file(generation)
    stated = json.loads(path.read_text())["schema_version"]
    _report_validation(str(path), as_written=True)

    captured = capsys.readouterr()
    assert captured.out == f"{path}: valid (schema version {stated})\n"
    assert captured.err == ""


def test_validate_command_as_written_reports_an_older_release_as_current(
    older_release_file, capsys
):
    # Same as the Python API: an older release of the current generation is
    # reported as the current version, with no upgrade line and no warning.
    _report_validation(str(older_release_file), as_written=True)

    captured = capsys.readouterr()
    assert captured.out == f"{older_release_file}: valid (schema version {SCHEMA_VERSION})\n"
    assert captured.err == ""


@pytest.mark.parametrize("text", ["[]", "null"])
def test_validate_command_non_object_json_exits_1(tmp_path, capsys, text):
    path = tmp_path / "not_an_object.json"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(SystemExit) as info:
        _report_validation(str(path), as_written=True)

    assert info.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith(f"{path}: invalid\n")
    assert "model_type" in captured.err


def test_validate_command_not_json_exits_1(not_json_file, capsys):
    with pytest.raises(SystemExit) as info:
        _report_validation(str(not_json_file))

    assert info.value.code == 1
    captured = capsys.readouterr()
    assert captured.err.startswith(f"{not_json_file}: invalid\nnot valid JSON")


def test_validate_command_missing_file_exits_1(tmp_path, capsys):
    missing = tmp_path / "missing.json"
    with pytest.raises(SystemExit) as info:
        _report_validation(str(missing))

    assert info.value.code == 1
    captured = capsys.readouterr()
    assert captured.err == f"{missing}: invalid\nno such file\n"
    assert not missing.exists()


def test_validate_command_unreadable_file_exits_1(tmp_path, capsys):
    # A path that names a directory, or a file that cannot be read, is
    # reported like any other invalid input, not as a traceback.
    directory = tmp_path / "starlists.json"
    directory.mkdir()
    with pytest.raises(SystemExit) as info:
        _report_validation(str(directory))

    assert info.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith(f"{directory}: invalid\ncannot read the file: ")

    unreadable = _write(tmp_path, "unreadable.json", _current_data())
    unreadable.chmod(0)
    try:
        if os.access(unreadable, os.R_OK):
            pytest.skip("file permissions are not enforced here (root?)")
        with pytest.raises(SystemExit) as info:
            _report_validation(str(unreadable))
    finally:
        unreadable.chmod(0o600)

    assert info.value.code == 1
    assert capsys.readouterr().err.startswith(
        f"{unreadable}: invalid\ncannot read the file: "
    )


def test_validate_command_not_utf8_exits_1(tmp_path, capsys):
    # Bytes that are not UTF-8 never reach the JSON parser; the decoding
    # error is reported in its place.
    path = tmp_path / "utf16.json"
    path.write_bytes(b"\xff\xfe{}")

    with pytest.raises(SystemExit) as info:
        _report_validation(str(path))

    assert info.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith(f"{path}: invalid\nnot UTF-8 text: ")


@pytest.mark.parametrize(
    "version, expected",
    [
        ("banana", "unsupported schema version"),
        ("9999.0.0", "newer schema version"),
    ],
)
def test_validate_command_version_errors_exit_1(tmp_path, capsys, version, expected):
    path = _write(tmp_path, "versioned.json", {"schema_version": version, "star_lists": []})
    with pytest.raises(SystemExit) as info:
        _report_validation(str(path))

    assert info.value.code == 1
    captured = capsys.readouterr()
    assert captured.err.startswith(f"{path}: invalid\n{expected}")
    assert repr(version) in captured.err


def test_cli_validates_a_file(mocker, capsys):
    # The console-script entry point parses the validate command and
    # --as-written. The positional file name comes first: Fire binds a value
    # following a bool flag to that flag.
    path = _older_file(next(iter(_MIGRATIONS)))
    stated = json.loads(path.read_text())["schema_version"]

    mocker.patch("sys.argv", ["aavso-starlist-schema", "validate", str(path)])
    cli()
    assert capsys.readouterr().out.splitlines()[0] == (
        f"{path}: valid (schema version {SCHEMA_VERSION})"
    )

    mocker.patch(
        "sys.argv", ["aavso-starlist-schema", "validate", str(path), "--as-written"]
    )
    cli()
    assert capsys.readouterr().out == f"{path}: valid (schema version {stated})\n"


def test_cli_validates_a_file_whose_name_is_a_literal(tmp_path, monkeypatch, capsys):
    # Fire turns an argument that looks like a Python literal into one, so a
    # file named 123 arrives as an int; the command must still find the file.
    monkeypatch.chdir(tmp_path)
    _write(tmp_path, "123", _current_data())

    monkeypatch.setattr("sys.argv", ["aavso-starlist-schema", "validate", "123"])
    cli()

    captured = capsys.readouterr()
    assert captured.out == f"123: valid (schema version {SCHEMA_VERSION})\n"
    assert captured.err == ""


def test_cli_exit_status_for_an_invalid_file(invalid_file, mocker, capsys):
    # Fire lets the SystemExit through, so the shell sees status 1.
    mocker.patch("sys.argv", ["aavso-starlist-schema", "validate", str(invalid_file)])
    with pytest.raises(SystemExit) as info:
        cli()

    assert info.value.code == 1
    assert "observer" in capsys.readouterr().err


def test_console_script_validates_end_to_end(current_file, invalid_file):
    # The installed script, run as a subprocess: exit status and streams.
    # Without the script this is a skip locally, but a failure in CI, where
    # the project must be installed: a silent skip would hide a broken entry
    # point.
    if shutil.which("aavso-starlist-schema") is None:
        if os.environ.get("CI"):
            pytest.fail("console script not installed; CI must install the project")
        pytest.skip("console script not installed in this environment")

    valid = subprocess.run(
        ["aavso-starlist-schema", "validate", str(current_file)],
        capture_output=True,
        text=True,
    )
    assert valid.returncode == 0
    assert valid.stdout == f"{current_file}: valid (schema version {SCHEMA_VERSION})\n"
    assert valid.stderr == ""

    invalid = subprocess.run(
        ["aavso-starlist-schema", "validate", str(invalid_file)],
        capture_output=True,
        text=True,
    )
    assert invalid.returncode == 1
    assert invalid.stdout == ""
    assert invalid.stderr.startswith(f"{invalid_file}: invalid\n")
    assert "observer" in invalid.stderr
