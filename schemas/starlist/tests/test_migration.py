import copy
import json
import re
import warnings
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

import pytest
from pydantic import ValidationError

import _aavso_starlist_legacy
from aavso_starlist_schema import (
    _MIGRATIONS,
    SCHEMA_VERSION,
    MigrationResultError,
    NewerSchemaVersionError,
    SchemaMigrationWarning,
    StarListSet,
    UnsupportedSchemaVersionError,
    _generation,
    _Migration,
    _parse_semver,
    upgrade,
    validate_as_written,
)

# One directory per migration step, named <old generation>_to_<new generation>,
# holding a before.json / after.json pair.
MIGRATION_DATA = Path(__file__).parent / "data" / "migrations"

# "0.0.1" was the example value in the legacy schema, so files may carry it.
LEGACY_VERSIONS = [
    None,
    "0.0.1.dev451+gde1568f",
    "0.1.dev66+g6f57c4d02",
    "0.0.0",
    "0.0.1",
    "0.1.0",
]


def _example_data(**overrides):
    """
    Build raw, unvalidated data for a star list set at the current version.

    Parameters
    ----------
    **overrides
        Top-level keys to set on the result, for example a different
        ``schema_version``.

    Returns
    -------
    dict
        The models' example values dumped to JSON-compatible types, with
        ``overrides`` applied: what ``upgrade()`` sees after ``json.loads``.
    """
    data = StarListSet.from_examples().model_dump(mode="json")
    data.update(overrides)
    return data


def _legacy_data(version):
    """
    Build example data as a pre-versioning file would present it.

    Parameters
    ----------
    version : str or None
        The legacy ``schema_version`` string to stamp, or ``None`` to omit
        the field entirely, as the oldest files did.

    Returns
    -------
    dict
        Raw starlist-set data carrying the legacy version.
    """
    data = _example_data()
    if version is None:
        del data["schema_version"]
    else:
        data["schema_version"] = version
    return data


def _before_after_pair(generation):
    """
    Load the before/after sample files for the migration out of a generation.

    Parameters
    ----------
    generation : str
        The generation being left, a key of ``_MIGRATIONS``.

    Returns
    -------
    tuple of dict
        The parsed ``before.json`` and ``after.json`` from
        ``tests/data/migrations/<generation>_to_<target>/``.
    """
    target_generation = _generation(_MIGRATIONS[generation].target)
    directory = MIGRATION_DATA / f"{generation}_to_{target_generation}"
    return (
        json.loads((directory / "before.json").read_text()),
        json.loads((directory / "after.json").read_text()),
    )


@pytest.mark.parametrize("generation", _MIGRATIONS)
def test_each_migration_step_matches_its_before_after_pair(generation):
    # Every migration needs a before/after pair; a missing directory fails here.
    before, after = _before_after_pair(generation)
    step = _MIGRATIONS[generation]

    result = step.func(copy.deepcopy(before))
    result["schema_version"] = step.target

    assert result == after


@pytest.mark.parametrize("generation", _MIGRATIONS)
def test_before_and_after_files_are_valid_for_their_own_generation(generation):
    # A migration must be tested against data that really conforms to the
    # schemas on both sides of it: before.json to the frozen model of the
    # generation being left, after.json to the model of the generation reached.
    before, after = _before_after_pair(generation)

    assert isinstance(validate_as_written(before), _MIGRATIONS[generation].model)
    validate_as_written(after)


@pytest.mark.parametrize("generation", _MIGRATIONS)
def test_before_files_upgrade_to_a_valid_current_file(generation):
    # Old files must stay readable however many generations later.
    before, _ = _before_after_pair(generation)

    with pytest.warns(SchemaMigrationWarning):
        upgraded = upgrade(before)

    assert StarListSet.model_validate(upgraded).schema_version == SCHEMA_VERSION


@pytest.mark.parametrize("version", LEGACY_VERSIONS)
def test_legacy_versions_are_upgraded_with_warning(version):
    # Every legacy version string, and a missing version, is upgraded to the
    # current version with exactly one warning.
    with pytest.warns(SchemaMigrationWarning) as record:
        result = upgrade(_legacy_data(version))

    # Exactly one warning, naming both ends of the migration and what changed.
    assert len(record) == 1
    message = str(record[0].message)
    assert ("missing" if version is None else version) in message
    assert SCHEMA_VERSION in message
    assert _MIGRATIONS["legacy"].summary in message

    assert result["schema_version"] == SCHEMA_VERSION


def test_current_version_is_returned_as_an_equal_copy_without_warning():
    # Data already at the current version comes back silently as an equal
    # copy that shares nothing with the input.
    data = _example_data(schema_version=SCHEMA_VERSION)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = upgrade(data)

    assert result == data
    assert result is not data
    assert result["star_lists"] is not data["star_lists"]


def test_upgraded_file_is_stamped_with_a_later_patch_version(mocker):
    # After a patch bump the last migration target is older than SCHEMA_VERSION;
    # an upgraded file must still come out at the current version.
    major, minor, patch = SCHEMA_VERSION.split(".")
    patched = f"{major}.{minor}.{int(patch) + 1}"
    legacy = _legacy_data(None)  # built before patching: it goes through the model
    mocker.patch("aavso_starlist_schema.SCHEMA_VERSION", patched)

    with pytest.warns(SchemaMigrationWarning, match=re.escape(patched)):
        result = upgrade(legacy)

    assert result["schema_version"] == patched


def test_older_version_of_current_generation_is_restamped(mocker):
    # A file from an older version of the current generation needs no
    # migration, but comes out stating the current version, silently and
    # without the input being modified. Simulate a patch bump.
    major, minor, patch = SCHEMA_VERSION.split(".")
    patched = f"{major}.{minor}.{int(patch) + 1}"
    data = {"schema_version": SCHEMA_VERSION, "star_lists": []}
    mocker.patch("aavso_starlist_schema.SCHEMA_VERSION", patched)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = upgrade(data)

    assert result == {"schema_version": patched, "star_lists": []}
    assert data["schema_version"] == SCHEMA_VERSION
    assert result["star_lists"] is not data["star_lists"]


def test_model_restamps_older_version_of_current_generation(mocker):
    # The model's validator restamps an older version of the current
    # generation. Simulate a patch bump and call the validator with a
    # pass-through handler, since the field pattern is fixed at import.
    major, minor, patch = SCHEMA_VERSION.split(".")
    patched = f"{major}.{minor}.{int(patch) + 1}"
    mocker.patch("aavso_starlist_schema.SCHEMA_VERSION", patched)
    mocker.patch("aavso_starlist_schema._RELEASED_VERSIONS", (SCHEMA_VERSION, patched))

    result = StarListSet._read_older_versions(
        {"schema_version": SCHEMA_VERSION, "star_lists": []}, lambda data: data
    )

    assert result["schema_version"] == patched


def test_each_migration_step_receives_data_stamped_with_its_source_version(mocker):
    # In a multi-hop upgrade, each step must see data stamped with the version
    # the previous step produced, not the original file's version. Simulate a
    # legacy -> 0.2 -> 1 chain with a fake second step.
    seen = []

    def _record_version(data):
        """
        Record the version a fake migration step receives.

        Parameters
        ----------
        data : dict
            Raw starlist-set data.

        Returns
        -------
        dict
            ``data`` unchanged.
        """
        seen.append(data.get("schema_version"))
        return data

    mocker.patch.dict(
        _MIGRATIONS, {"0.2": _Migration(StarListSet, "1.0.0", _record_version, "fake")}
    )
    mocker.patch("aavso_starlist_schema.SCHEMA_VERSION", "1.0.0")

    with pytest.warns(SchemaMigrationWarning):
        result = upgrade({"star_lists": []})

    assert seen == [_MIGRATIONS["legacy"].target]
    assert result["schema_version"] == "1.0.0"


def test_gap_in_migration_chain_names_the_generation_without_a_migration(mocker):
    # A breaking bump whose migration out of the generation being left has not
    # been written yet must say so, not fail with a bare KeyError. Simulate a
    # bump to 1.0.0 with no migration out of 0.2: a legacy file still has its
    # first step, then has nowhere to go.
    mocker.patch("aavso_starlist_schema.SCHEMA_VERSION", "1.0.0")
    expected = "no migration out of schema generation '0.2'"

    with pytest.raises(UnsupportedSchemaVersionError, match=expected):
        upgrade({"star_lists": []})
    with pytest.raises(UnsupportedSchemaVersionError, match=expected):
        StarListSet.model_validate({"star_lists": []})


def test_model_applies_version_handling_to_any_mapping(next_generation_version):
    # A mapping that is not a dict gets the same version handling as a dict:
    # a legacy one is upgraded and a newer one is refused.
    with pytest.warns(SchemaMigrationWarning):
        upgraded = StarListSet.model_validate(MappingProxyType({"star_lists": []}))
    assert upgraded.schema_version == SCHEMA_VERSION

    newer = MappingProxyType(
        {"schema_version": next_generation_version, "star_lists": []}
    )
    with pytest.raises(NewerSchemaVersionError):
        StarListSet.model_validate(newer)


def test_newer_version_raises(next_generation_version):
    # Any version newer than the current one, even a patch bump in the same
    # generation, is an error telling the user to upgrade the package, not a
    # validation error: this reader cannot know what changed, and a newer
    # version of its own generation (a patch while 0.y.z, a minor from 1.0.0
    # on) may add fields it would silently drop.
    major, minor, patch = SCHEMA_VERSION.split(".")
    next_patch = f"{major}.{minor}.{int(patch) + 1}"
    for version in ["9.0.0", next_generation_version, next_patch]:
        with pytest.raises(NewerSchemaVersionError, match=re.escape(version)):
            upgrade(_example_data(schema_version=version))


@pytest.mark.parametrize(
    "version",
    [
        "banana",
        2,
        ["0.2.0"],
        "1.2",
        None,
        "0.3.0rc1",
        "0.3.0.dev1",
        # Leading zeros: not a second spelling of a version.
        "0.2.00",
        "00.2.0",
        "0.01.0",
    ],
)
def test_unsupported_versions_raise(version):
    # Non-strings, an explicit null, and strings that are neither X.Y.Z nor a
    # legacy package version are rejected as uninterpretable, not treated as
    # legacy. (None here is a null value; a missing field is legacy.)
    with pytest.raises(UnsupportedSchemaVersionError):
        upgrade(_example_data(schema_version=version))


def test_version_errors_are_not_value_errors():
    # Callers catching ValueError for validation problems must not swallow a
    # version problem.
    assert not issubclass(NewerSchemaVersionError, ValueError)
    assert not issubclass(UnsupportedSchemaVersionError, ValueError)


def test_upgrade_does_not_mutate_input():
    # upgrade() returns a new dict and leaves the caller's data untouched.
    data = _legacy_data(None)
    snapshot = copy.deepcopy(data)
    with pytest.warns(SchemaMigrationWarning):
        result = upgrade(data)
    assert data == snapshot
    assert result is not data


def test_migration_chain_reaches_current_generation():
    # Following the migrations from the oldest generation must end at the
    # generation of SCHEMA_VERSION, with no gaps. This fails when a breaking
    # bump lands without its migration.
    generation = "legacy"
    seen = set()
    while generation != _generation(SCHEMA_VERSION):
        assert generation not in seen, f"migration cycle at {generation!r}"
        seen.add(generation)
        assert generation in _MIGRATIONS, (
            f"no migration out of generation {generation!r}; every new generation "
            "needs a _MIGRATIONS entry and a before/after pair, see 'Changing "
            "the schema' in README.md"
        )
        target = _MIGRATIONS[generation].target
        assert _parse_semver(target) is not None
        generation = _generation(target)

    # No migration should start from the current generation or beyond.
    assert _generation(SCHEMA_VERSION) not in _MIGRATIONS
    assert seen == set(_MIGRATIONS)


def test_model_validate_json_upgrades_a_legacy_file():
    # A real legacy file loads through the standard pydantic entry point,
    # upgraded, with all of its star lists and star items intact.
    before, _ = _before_after_pair("legacy")
    with pytest.warns(SchemaMigrationWarning):
        result = StarListSet.model_validate_json(json.dumps(before))

    assert result.schema_version == SCHEMA_VERSION
    assert len(result.star_lists) == len(before["star_lists"])
    assert len(result.star_lists[0].staritems) == len(before["star_lists"][0]["staritems"])


def test_model_validate_json_current_round_trip():
    # A current-version file is a plain parse: no warning, equal object.
    original = StarListSet.from_examples()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert StarListSet.model_validate_json(original.model_dump_json()) == original


def test_model_validate_and_constructor_upgrade_legacy_data():
    # The upgrade runs for every way of creating a StarListSet, not just from
    # JSON text; data without a schema_version is legacy.
    with pytest.warns(SchemaMigrationWarning):
        assert StarListSet.model_validate({"star_lists": []}).schema_version == SCHEMA_VERSION
    with pytest.warns(SchemaMigrationWarning):
        assert StarListSet(star_lists=[]).schema_version == SCHEMA_VERSION


def test_model_rejects_newer_version_in_current_generation():
    # The model does not read a newer file of its own generation and ignore
    # the fields it does not know.
    major, minor, patch = SCHEMA_VERSION.split(".")
    with pytest.raises(NewerSchemaVersionError):
        StarListSet(schema_version=f"{major}.{minor}.{int(patch) + 1}", star_lists=[])


def test_model_raises_version_errors_unwrapped(next_generation_version):
    # Version problems surface from the model as SchemaVersionErrors, not
    # wrapped in a pydantic ValidationError.
    with pytest.raises(NewerSchemaVersionError):
        StarListSet.model_validate_json(
            json.dumps(_example_data(schema_version=next_generation_version))
        )
    with pytest.raises(UnsupportedSchemaVersionError):
        StarListSet(schema_version="banana", star_lists=[])


def test_validate_as_written_uses_the_model_of_the_files_own_generation():
    # A legacy file is checked against the frozen legacy model and a current
    # file against the live model; neither is upgraded, so nothing warns.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        legacy = validate_as_written(_legacy_data("0.1.dev66+g6f57c4d02"))
        current = validate_as_written(_example_data())

    assert isinstance(legacy, _aavso_starlist_legacy.StarListSet)
    assert legacy.schema_version == "0.1.dev66+g6f57c4d02"
    assert isinstance(current, StarListSet)


def test_version_that_was_never_released_raises(mocker):
    # A version at or below the current one that has no archive in the data folder
    # was never released, so no file can legitimately carry it. Simulate two
    # patch bumps where the first was skipped.
    major, minor, patch = (int(part) for part in SCHEMA_VERSION.split("."))
    skipped = f"{major}.{minor}.{patch + 1}"
    mocker.patch("aavso_starlist_schema.SCHEMA_VERSION", f"{major}.{minor}.{patch + 2}")

    with pytest.raises(UnsupportedSchemaVersionError, match="never released"):
        StarListSet.model_validate({"schema_version": skipped, "star_lists": []})

    # The archived version is still accepted by the version check.
    upgrade({"schema_version": SCHEMA_VERSION, "star_lists": []})


def test_file_invalid_as_written_is_a_validation_error():
    # A legacy file that does not match the legacy schema is the file's
    # problem: an ordinary ValidationError, raised before any upgrade.
    data = _legacy_data(None)
    data["star_lists"] = [{"observer": "a star list missing its other fields"}]

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValidationError):
            validate_as_written(data)
        with pytest.raises(ValidationError):
            StarListSet.model_validate(data)


def test_invalid_migration_result_is_a_distinct_error(mocker):
    # A file valid as written whose upgraded form is invalid means a migration
    # is wrong. That is reported as MigrationResultError, not as a
    # ValidationError blaming the file.
    def _broken_migration(data):
        """
        Stand in for a buggy migration by dropping a required field.

        Parameters
        ----------
        data : dict
            Raw starlist-set data.

        Returns
        -------
        dict
            ``data`` without its ``star_lists``.
        """
        del data["star_lists"]
        return data

    broken = _Migration(
        _aavso_starlist_legacy.StarListSet, SCHEMA_VERSION, _broken_migration, "broken"
    )
    mocker.patch.dict(_MIGRATIONS, {"legacy": broken})

    with pytest.warns(SchemaMigrationWarning):
        with pytest.raises(MigrationResultError) as excinfo:
            StarListSet.model_validate(_legacy_data(None))

    assert not isinstance(excinfo.value, ValueError)
    assert isinstance(excinfo.value.__cause__, ValidationError)


class _Mapping(Mapping):
    """
    A minimal mapping that is not a dict, as code calling the model might pass.

    Unlike ``MappingProxyType`` it can be deep-copied, so `upgrade` accepts it
    when called directly.

    Parameters
    ----------
    data : dict
        The items to expose.
    """

    def __init__(self, data):
        self._data = dict(data)

    def __getitem__(self, key):
        """
        Look up a key.

        Parameters
        ----------
        key : str
            The key.

        Returns
        -------
        object
            Its value.
        """
        return self._data[key]

    def __iter__(self):
        """
        Iterate over the keys.

        Returns
        -------
        iterator
            The keys.
        """
        return iter(self._data)

    def __len__(self):
        """
        Count the keys.

        Returns
        -------
        int
            The number of keys.
        """
        return len(self._data)


def _legacy_python_data(shape):
    """
    Build a legacy star list set as code might, in a shape JSON cannot give.

    Parameters
    ----------
    shape : str
        ``"instances"`` for frozen legacy ``StarList`` instances, ``"tuple"``
        for a tuple of star lists, ``"mapping"`` for star lists that are
        mappings but not dicts, or ``"enum"`` for an enum member as ``filter``.

    Returns
    -------
    dict
        The legacy sample file, with its star lists in that shape.
    """
    before, _ = _before_after_pair("legacy")
    star_lists = before["star_lists"]
    if shape == "instances":
        before["star_lists"] = [
            _aavso_starlist_legacy.StarList.model_validate(star_list)
            for star_list in star_lists
        ]
    elif shape == "tuple":
        before["star_lists"] = tuple(star_lists)
    elif shape == "mapping":
        before["star_lists"] = [_Mapping(star_list) for star_list in star_lists]
    elif shape == "enum":
        for star_list in star_lists:
            star_list["filter"] = _aavso_starlist_legacy.AAVSOFilters(star_list["filter"])
    return before


def _legacy_data_json():
    """
    Load the legacy sample file as plain JSON data.

    Returns
    -------
    dict
        The parsed ``before.json`` of the legacy migration.
    """
    before, _ = _before_after_pair("legacy")
    return before


def _assert_plain(value):
    """
    Check that a value is made only of the types JSON text parses into.

    Parameters
    ----------
    value : object
        The value to check, recursively.
    """
    if type(value) is dict:
        for key, item in value.items():
            assert type(key) is str, f"key {key!r} is a {type(key).__name__}"
            _assert_plain(item)
    elif type(value) is list:
        for item in value:
            _assert_plain(item)
    else:
        assert type(value) in (str, int, float, bool, type(None)), (
            f"{value!r} is a {type(value).__name__}"
        )


@pytest.mark.parametrize("shape", ["instances", "tuple", "mapping", "enum"])
def test_model_upgrades_legacy_data_built_in_python(shape):
    # Data the frozen model accepts in a shape JSON cannot give is converted to
    # plain data before migrating, so it upgrades like the same file read from
    # JSON instead of being rejected by the current model as a migration bug.
    with pytest.warns(SchemaMigrationWarning):
        expected = StarListSet.model_validate_json(json.dumps(_legacy_data_json()))
    with pytest.warns(SchemaMigrationWarning):
        validated = StarListSet.model_validate(_legacy_python_data(shape))
    with pytest.warns(SchemaMigrationWarning):
        constructed = StarListSet(**_legacy_python_data(shape))

    assert validated == expected
    assert constructed == expected


@pytest.mark.parametrize("shape", ["instances", "tuple", "mapping", "enum"])
def test_migration_reached_through_the_model_sees_only_plain_data(shape, mocker):
    # Through the model a migration only ever sees what JSON text would give,
    # whatever shape the caller built the data in.
    seen = []

    def _record_argument(data):
        """
        Record the argument a stand-in migration receives.

        Parameters
        ----------
        data : dict
            Raw starlist-set data.

        Returns
        -------
        dict
            ``data`` unchanged.
        """
        seen.append(copy.deepcopy(data))
        return data

    legacy = _MIGRATIONS["legacy"]
    mocker.patch.dict(_MIGRATIONS, {"legacy": legacy._replace(func=_record_argument)})

    with pytest.warns(SchemaMigrationWarning):
        StarListSet.model_validate(_legacy_python_data(shape))

    assert len(seen) == 1
    _assert_plain(seen[0])
    assert seen[0]["star_lists"] == _legacy_data_json()["star_lists"]


def test_key_the_frozen_model_ignores_reaches_the_migration(mocker):
    # The raw input is converted, not a dump of the validated frozen model, so
    # a key that model ignores is still there for a migration that uses it.
    seen = []

    def _record_argument(data):
        """
        Record the argument a stand-in migration receives.

        Parameters
        ----------
        data : dict
            Raw starlist-set data.

        Returns
        -------
        dict
            ``data`` unchanged.
        """
        seen.append(copy.deepcopy(data))
        return data

    legacy = _MIGRATIONS["legacy"]
    mocker.patch.dict(_MIGRATIONS, {"legacy": legacy._replace(func=_record_argument)})
    data = _legacy_python_data("instances")
    data["unknown_key"] = ("kept", MappingProxyType({"nested": 1}))
    data["star_lists"][0] = MappingProxyType(
        {**data["star_lists"][0].model_dump(), "unknown_in_star_list": "kept"}
    )

    with pytest.warns(SchemaMigrationWarning):
        StarListSet.model_validate(data)

    assert seen[0]["unknown_key"] == ["kept", {"nested": 1}]
    assert seen[0]["star_lists"][0]["unknown_in_star_list"] == "kept"


def test_upgrade_called_directly_passes_values_through_unconverted(mocker):
    # upgrade() itself is a pure dict transform: it copies the data but does
    # not convert what is inside, so a migration called that way may see
    # tuples, other mappings, model instances and enum members.
    seen = []

    def _record_argument(data):
        """
        Record the argument a stand-in migration receives.

        Parameters
        ----------
        data : dict
            Raw starlist-set data.

        Returns
        -------
        dict
            ``data`` unchanged.
        """
        seen.append(data)
        return data

    legacy = _MIGRATIONS["legacy"]
    mocker.patch.dict(_MIGRATIONS, {"legacy": legacy._replace(func=_record_argument)})
    star_list = _legacy_python_data("instances")["star_lists"][0]
    data = {
        "star_lists": (star_list, _Mapping({"a": 1})),
        "filter": _aavso_starlist_legacy.AAVSOFilters(star_list.filter),
    }

    with pytest.warns(SchemaMigrationWarning):
        upgrade(data)

    received = seen[0]
    assert type(received["star_lists"]) is tuple
    assert type(received["star_lists"][0]) is _aavso_starlist_legacy.StarList
    assert type(received["star_lists"][1]) is _Mapping
    assert type(received["filter"]) is _aavso_starlist_legacy.AAVSOFilters
