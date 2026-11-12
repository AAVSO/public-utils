import copy
import datetime
import decimal
import io
import json
import re
import warnings
from collections import deque
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import List  # noqa: UP035 -- the bare typing.List is tested

import numpy as np
import pytest
from _helpers import float_fields, int_fields
from pydantic import BaseModel, ConfigDict, ValidationError

import _aavso_starlist_legacy
import _aavso_starlist_v0_2
import _aavso_starlist_v0_3
import _aavso_starlist_v0_4
from aavso_starlist_schema import (
    _0_3_STAR_ITEM_FLOATS,
    _0_3_STAR_LIST_FLOATS,
    _0_4_MAX_PHOTOMETRY_SOFTWARE_ENTRY,
    _0_4_MAX_STARITEMS,
    _0_4_MAX_STRING_LENGTHS,
    _0_4_STAR_ITEM_FLOATS,
    _0_4_STAR_LIST_FLOATS,
    _0_4_STAR_LIST_INTS,
    _1_INT_MAXIMA,
    _1_MAX_PHOTOMETRY_SOFTWARE,
    _1_MAX_STAR_LISTS,
    _1_MAX_STRING_LENGTHS,
    _1_OBS_TIME,
    _1_OBSERVER,
    _MIGRATIONS,
    _OBS_TIME_PATTERN,
    SCHEMA_VERSION,
    AAVSOFilters,
    MigrationResultError,
    NewerSchemaVersionError,
    SchemaMigrationWarning,
    StarItem,
    StarList,
    StarListSet,
    UnsupportedSchemaVersionError,
    _1_is_iso_datetime,
    _generation,
    _Migration,
    _parse_semver,
    _prune_and_materialize,
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
    # A legacy file goes through every migration, so the warning names each.
    for step in _MIGRATIONS.values():
        assert step.summary in message

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


def test_gap_in_migration_chain_names_the_generation_without_a_migration(
    mocker, next_generation_version
):
    # A breaking bump whose migration out of the generation being left has not
    # been written yet must say so, not fail with a bare KeyError. Simulate a
    # bump to the next generation with no migration out of the current one: a
    # legacy file still has its steps up to the current generation, then has
    # nowhere to go.
    mocker.patch("aavso_starlist_schema.SCHEMA_VERSION", next_generation_version)
    expected = f"no migration out of schema generation {_generation(SCHEMA_VERSION)!r}"

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
    # This is the end-to-end check. The 0.2 to 0.3 migration also copes with
    # these shapes itself, so it is the next test, which records what each
    # migration receives, that pins the conversion.
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
    # Through the model every migration in the chain, not only the first, sees
    # only what JSON text would give, whatever shape the caller built the data
    # in. A legacy file passes through every registered migration.
    seen = []

    def _recording(func):
        """
        Wrap a migration so that it records the argument it receives.

        Parameters
        ----------
        func : callable
            The real migration function.

        Returns
        -------
        callable
            A function that records a deep copy of its argument, then
            returns what ``func`` returns for it.
        """

        def _record_argument(data):
            """
            Record the argument, then run the real migration on it.

            Parameters
            ----------
            data : dict
                Raw starlist-set data.

            Returns
            -------
            dict
                The result of the real migration.
            """
            seen.append(copy.deepcopy(data))
            return func(data)

        return _record_argument

    mocker.patch.dict(
        _MIGRATIONS,
        {
            generation: step._replace(func=_recording(step.func))
            for generation, step in _MIGRATIONS.items()
        },
    )

    with pytest.warns(SchemaMigrationWarning):
        StarListSet.model_validate(_legacy_python_data(shape))

    assert len(seen) == len(_MIGRATIONS)
    for argument in seen:
        _assert_plain(argument)
    assert seen[0]["star_lists"] == _legacy_data_json()["star_lists"]


def test_keys_the_frozen_model_ignores_do_not_reach_the_migration(mocker):
    # Keys the legacy model does not define are dropped before validation,
    # even those a later generation defines (bias, photometry_software, star
    # item fwhm): no schema validated their values, so no migration may use
    # them, and the upgraded file has the defaults a legacy file gets. Before
    # the fix they were pruned by the 0.3 field set and reached the
    # migrations unvalidated.
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
    star_list = data["star_lists"][0].model_dump()
    star_list["staritems"][0] = MappingProxyType({**star_list["staritems"][0], "fwhm": 2.5})
    data["star_lists"][0] = MappingProxyType(
        {**star_list, "bias": 5.0, "photometry_software": ("kept",)}
    )

    with pytest.warns(SchemaMigrationWarning):
        result = StarListSet.model_validate(data)

    received = seen[0]["star_lists"][0]
    assert "bias" not in received
    assert "photometry_software" not in received
    assert "fwhm" not in received["staritems"][0]
    assert result.star_lists[0].bias is None
    assert result.star_lists[0].photometry_software == ["unknown"]
    assert result.star_lists[0].staritems[0].fwhm is None


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
        # A shallow copy: the later 0.2 to 0.3 step replaces the star lists in
        # the dict it is passed, and the values inside are what is checked.
        seen.append(dict(data))
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


def test_file_from_before_absolute_focus_reads_it_as_none():
    # The 0.2.0 sample file predates absolute_focus (added in 0.2.1); it is
    # upgraded to the current version with absolute_focus None.
    _, written_at_0_2_0 = _before_after_pair("legacy")
    assert written_at_0_2_0["schema_version"] == "0.2.0"
    assert "absolute_focus" not in written_at_0_2_0["star_lists"][0]

    with pytest.warns(SchemaMigrationWarning):
        result = StarListSet.model_validate(written_at_0_2_0)

    assert result.schema_version == SCHEMA_VERSION
    assert result.star_lists[0].absolute_focus is None


def test_legacy_file_is_upgraded_with_absolute_focus_none():
    # A legacy file is migrated and restamped at the current version, and
    # reads the field added since as None.
    before, _ = _before_after_pair("legacy")

    with pytest.warns(SchemaMigrationWarning):
        result = StarListSet.model_validate(before)

    assert result.schema_version == SCHEMA_VERSION
    assert result.star_lists[0].absolute_focus is None


def test_file_from_before_star_item_fwhm_reads_it_as_none():
    # The 0.3.0 sample file predates the per-star fwhm (added in 0.3.1); it is
    # upgraded with a warning to the current version, with fwhm None on every
    # star item.
    _, written_at_0_3_0 = _before_after_pair("0.2")
    assert written_at_0_3_0["schema_version"] == "0.3.0"
    star_items = [
        item for star_list in written_at_0_3_0["star_lists"] for item in star_list["staritems"]
    ]
    assert star_items
    assert all("fwhm" not in item for item in star_items)

    with pytest.warns(SchemaMigrationWarning):
        result = StarListSet.model_validate(written_at_0_3_0)

    assert result.schema_version == SCHEMA_VERSION
    assert [item.fwhm for star_list in result.star_lists for item in star_list.staritems] == [
        None
    ] * len(star_items)


def test_legacy_file_is_upgraded_with_star_item_fwhm_none():
    # A legacy file is migrated through both generations and restamped at
    # the current version, and every star item reads the per-star fwhm added
    # since as None.
    before, _ = _before_after_pair("legacy")

    with pytest.warns(SchemaMigrationWarning):
        result = StarListSet.model_validate(before)

    assert result.schema_version == SCHEMA_VERSION
    star_items = [item for star_list in result.star_lists for item in star_list.staritems]
    assert star_items
    assert all(item.fwhm is None for item in star_items)


def test_current_file_with_star_item_fwhm_round_trips():
    # A file at the current version whose star items set fwhm is read without
    # a warning (pytest treats an unexpected SchemaMigrationWarning as an
    # error), keeps the values, and is written back unchanged.
    star_list = StarList.from_examples()
    star_list.staritems = [StarItem.from_examples()]
    data = _example_data(star_lists=[star_list.model_dump(mode="json")])
    assert data["star_lists"][0]["staritems"][0]["fwhm"] == 3.4

    result = StarListSet.model_validate_json(json.dumps(data))

    assert result.star_lists[0].staritems[0].fwhm == 3.4
    assert result.model_dump(mode="json") == data


@pytest.mark.parametrize("version", ["0.3.0", "0.3.1"])
def test_file_from_before_bias_reads_it_as_none(version):
    # A file stamped at a version of the 0.3 generation from before the
    # per-frame bias (added in 0.3.2) has no bias. The 0.3.0 sample file,
    # restamped at each such version, is upgraded with a warning to the
    # current version, with bias None on every star list.
    _, written_before_bias = _before_after_pair("0.2")
    assert written_before_bias["schema_version"] == "0.3.0"
    assert written_before_bias["star_lists"]
    assert all("bias" not in star_list for star_list in written_before_bias["star_lists"])
    written_before_bias["schema_version"] = version

    with pytest.warns(SchemaMigrationWarning):
        result = StarListSet.model_validate(written_before_bias)

    assert result.schema_version == SCHEMA_VERSION
    assert [star_list.bias for star_list in result.star_lists] == [None] * len(
        result.star_lists
    )


@pytest.mark.parametrize("generation", ["legacy", "0.2"])
def test_older_file_is_upgraded_with_bias_none(generation):
    # A legacy or 0.2 file is migrated to the current generation and restamped
    # at the current version, and every star list reads the bias added since
    # as None.
    before, _ = _before_after_pair(generation)

    with pytest.warns(SchemaMigrationWarning):
        result = StarListSet.model_validate(before)

    assert result.schema_version == SCHEMA_VERSION
    assert result.star_lists
    assert all(star_list.bias is None for star_list in result.star_lists)


@pytest.mark.parametrize("bias", [1000.0, None], ids=["number", "none"])
def test_current_file_with_bias_round_trips(bias):
    # A file at the current version that gives a bias is read without a
    # warning (pytest treats an unexpected SchemaMigrationWarning as an
    # error), keeps the value, and is written back unchanged; so is one that
    # gives null, which says the bias is not available.
    star_list = StarList.from_examples().model_dump(mode="json")
    star_list["bias"] = bias
    data = _example_data(star_lists=[star_list])

    result = StarListSet.model_validate_json(json.dumps(data))

    assert result.star_lists[0].bias == bias
    assert result.model_dump(mode="json") == data


def _written_at_0_2(version):
    """
    Load the sample file written at a version of the 0.2 generation.

    Parameters
    ----------
    version : {"0.2.0", "0.2.1"}
        The version the file states.

    Returns
    -------
    dict
        For 0.2.0, ``after.json`` of the legacy migration, which predates
        ``absolute_focus``; for 0.2.1, ``before.json`` of the 0.2 migration,
        which sets ``absolute_focus`` on one star list and not the other.
    """
    before, after = _before_after_pair("legacy" if version == "0.2.0" else "0.2")
    data = after if version == "0.2.0" else before
    assert data["schema_version"] == version
    return data


@pytest.mark.parametrize("version", ["0.2.0", "0.2.1"])
def test_0_2_file_is_validated_against_the_frozen_0_2_model(version):
    # validate_as_written checks a file from either 0.2 version against the
    # frozen 0.2 model, without upgrading it, so without a warning (pytest
    # treats an unexpected SchemaMigrationWarning as an error).
    result = validate_as_written(_written_at_0_2(version))

    assert isinstance(result, _aavso_starlist_v0_2.StarListSet)
    assert result.schema_version == version


@pytest.mark.parametrize("version", ["0.2.0", "0.2.1"])
def test_0_2_file_is_upgraded_with_photometry_software_unknown(version):
    # A file from either 0.2 version is upgraded with one warning, naming the
    # 0.2 migration but not the legacy one, and every star list gets
    # photometry_software ["unknown"].
    with pytest.warns(SchemaMigrationWarning) as record:
        result = StarListSet.model_validate(_written_at_0_2(version))

    assert len(record) == 1
    message = str(record[0].message)
    assert repr(version) in message
    assert _MIGRATIONS["0.2"].summary in message
    assert _MIGRATIONS["legacy"].summary not in message

    assert result.schema_version == SCHEMA_VERSION
    assert [star_list.photometry_software for star_list in result.star_lists] == [
        ["unknown"]
    ] * len(result.star_lists)


@pytest.mark.parametrize("value", [["bandaid 1.2.3"], [], "bandaid 1.2.3", None])
def test_0_2_file_with_photometry_software_already_gets_unknown(value):
    # The frozen 0.2 models ignore keys they do not define, so a 0.2 file may
    # carry a photometry_software key, valid for 0.3 or not. The 0.2 schema
    # never validated it, so it is dropped before the migrations and the 0.2
    # migration sets ["unknown"] like on any other 0.2 star list: a file
    # valid for 0.2 is always valid after migration.
    data = copy.deepcopy(_written_at_0_2("0.2.1"))
    data["star_lists"][0]["photometry_software"] = value

    with pytest.warns(SchemaMigrationWarning):
        result = StarListSet.model_validate(data)

    assert [star_list.photometry_software for star_list in result.star_lists] == [
        ["unknown"],
        ["unknown"],
    ]


def test_legacy_file_is_upgraded_through_every_migration():
    # A legacy file is migrated to 0.2, then to 0.3, 0.4 and on to the
    # current generation, and so gets photometry_software ["unknown"] too,
    # and through the 0.4 step its lowercase observer code is uppercased and
    # a number written as a string becomes a number. (That the warning names
    # every step is checked by test_legacy_versions_are_upgraded_with_warning.)
    before, _ = _before_after_pair("legacy")
    before["star_lists"][0]["observer"] = "mmu"
    before["star_lists"][0]["exposure"] = "30"

    with pytest.warns(SchemaMigrationWarning):
        result = StarListSet.model_validate(before)

    assert result.schema_version == SCHEMA_VERSION
    assert result.star_lists[0].photometry_software == ["unknown"]
    assert result.star_lists[0].observer == "MMU"
    assert result.star_lists[0].exposure == 30.0


@pytest.mark.parametrize(
    "data",
    [
        pytest.param({"schema_version": "0.2.1"}, id="no-star_lists"),
        pytest.param({"schema_version": "0.2.1", "star_lists": {"a": 1}}, id="mapping"),
        pytest.param(
            {"schema_version": "0.2.1", "star_lists": [1, None]},
            id="non-mapping-entries",
        ),
    ],
)
def test_0_2_migration_passes_unrecognised_shapes_through(data):
    # The 0.2 -> 0.3 migration only touches a list or tuple of star lists, and
    # within it only mapping or model entries. Anything else is returned
    # unchanged for validation to reject. These shapes reach the migration
    # only when it is called directly, since the frozen model refuses them.
    snapshot = copy.deepcopy(data)
    assert _MIGRATIONS["0.2"].func(data) == snapshot


def test_current_file_with_absolute_focus_round_trips():
    # A file at the current version that sets absolute_focus is read without
    # a warning (pytest treats an unexpected SchemaMigrationWarning as an
    # error), keeps the value, and is written back unchanged.
    data = _example_data(star_lists=[StarList.from_examples().model_dump(mode="json")])
    assert data["star_lists"][0]["absolute_focus"] == 1823.0

    result = StarListSet.model_validate_json(json.dumps(data))

    assert result.star_lists[0].absolute_focus == 1823.0
    assert result.model_dump(mode="json") == data


def _written_at_0_3(version="0.3.2"):
    """
    Load the sample file of the 0.3 generation, stamped at a version of it.

    Parameters
    ----------
    version : {"0.3.0", "0.3.1", "0.3.2"}, optional
        The version the file is to state. Default ``"0.3.2"``, the version
        ``before.json`` of the 0.3 migration states.

    Returns
    -------
    dict
        A deep copy of ``before.json`` of the 0.3 migration, which carries
        keys the 0.3 models do not define at every level and a ``stack`` of
        0, with ``schema_version`` set to ``version``. The frozen 0.3 model,
        a copy of 0.3.2, validates it at any of the three versions.
    """
    before, _ = _before_after_pair("0.3")
    data = copy.deepcopy(before)
    data["schema_version"] = version
    return data


@pytest.mark.parametrize("version", ["0.3.0", "0.3.1", "0.3.2"])
def test_0_3_file_is_validated_against_the_frozen_0_3_model(version):
    # validate_as_written checks a file from any 0.3 version against the
    # frozen 0.3 model, without upgrading it, so without a warning (pytest
    # treats an unexpected SchemaMigrationWarning as an error).
    result = validate_as_written(_written_at_0_3(version))

    assert isinstance(result, _aavso_starlist_v0_3.StarListSet)
    assert result.schema_version == version


@pytest.mark.parametrize("version", ["0.3.0", "0.3.1", "0.3.2"])
def test_0_3_file_is_upgraded_with_one_warning_naming_the_0_3_step(version):
    # A file from any 0.3 version is upgraded with one warning, naming the
    # 0.3 migration but not the earlier ones, to the current version.
    with pytest.warns(SchemaMigrationWarning) as record:
        result = StarListSet.model_validate(_written_at_0_3(version))

    assert len(record) == 1
    message = str(record[0].message)
    assert repr(version) in message
    assert _MIGRATIONS["0.3"].summary in message
    assert _MIGRATIONS["0.2"].summary not in message

    assert result.schema_version == SCHEMA_VERSION


def test_0_3_migration_drops_keys_the_0_3_models_do_not_define():
    # The 0.3 models ignored keys they did not define, and 0.4.0 forbids
    # them, so the migration drops them at every level: the set, each star
    # list and each star item. Nothing a 0.3 file meant is lost.
    before = _written_at_0_3()
    assert "pipeline" in before
    assert "instrument_notes" in before["star_lists"][0]
    assert "flag" in before["star_lists"][0]["staritems"][0]

    with pytest.warns(SchemaMigrationWarning):
        result = StarListSet.model_validate(before).model_dump(mode="json")
    migrated = _MIGRATIONS["0.3"].func(copy.deepcopy(before))

    for data in (result, migrated):
        assert "pipeline" not in data
        assert "instrument_notes" not in data["star_lists"][0]
        assert "flag" not in data["star_lists"][0]["staritems"][0]


@pytest.mark.parametrize(
    "value",
    [0, -1, "0", False, -(10**400), np.int64(0)],
    ids=["zero", "negative", "numeric-string", "false", "huge-negative", "numpy-zero"],
)
def test_stack_below_one_in_a_0_3_file_becomes_none(value):
    # A stack of 0 or less counts no images, so it becomes None, "not
    # applicable". The frozen 0.3 model accepts the numeric string, the bool,
    # an int too large for a float and a numpy integer built in Python, so
    # the migration reads them as numbers too.
    data = _written_at_0_3()
    data["star_lists"][0]["stack"] = value

    with pytest.warns(SchemaMigrationWarning):
        result = StarListSet.model_validate(data)

    assert result.star_lists[0].stack is None
    assert _MIGRATIONS["0.3"].func(copy.deepcopy(data))["star_lists"][0]["stack"] is None


@pytest.mark.parametrize(
    "value", [1, 3, None, 10**400], ids=["one", "three", "none", "huge"]
)
def test_stack_of_one_or_none_in_a_0_3_file_is_kept(value):
    # A stack of 1 or more, or None, is valid for 0.4.0 and comes through
    # the 0.3 migration unchanged, even an int too large for a float (which
    # the 0.4 to 1 migration then refuses, see
    # test_0_4_file_the_new_generation_cannot_hold_is_rejected).
    data = _written_at_0_3()
    data["star_lists"][0]["stack"] = value

    migrated = _MIGRATIONS["0.3"].func(copy.deepcopy(data))
    migrated["schema_version"] = _MIGRATIONS["0.3"].target

    assert migrated["star_lists"][0]["stack"] == value
    assert _aavso_starlist_v0_4.StarListSet.model_validate(migrated).star_lists[0].stack == value


def _set_star_item_x_to_infinity(data):
    """
    Make the x of the second star item of the first star list infinite.

    Parameters
    ----------
    data : dict
        Raw 0.3 starlist-set data, modified in place.
    """
    data["star_lists"][0]["staritems"][1]["x"] = float("inf")


def _set_site_elev_to_nan(data):
    """
    Make the site elevation of the second star list NaN.

    Parameters
    ----------
    data : dict
        Raw 0.3 starlist-set data, modified in place.
    """
    data["star_lists"][1]["site_elev"] = float("nan")


def _set_bkgd_count_to_the_string_infinity(data):
    """
    Make the background count of the first star item the string "Infinity".

    Parameters
    ----------
    data : dict
        Raw 0.3 starlist-set data, modified in place.
    """
    data["star_lists"][0]["staritems"][0]["bkgd_count"] = "Infinity"


def _set_site_elev_to_numpy_infinity(data):
    """
    Make the site elevation of the first star list a numpy float32 infinity.

    Parameters
    ----------
    data : dict
        Raw 0.3 starlist-set data, modified in place. Only data built in
        Python can hold a numpy value.
    """
    data["star_lists"][0]["site_elev"] = np.float32(np.inf)


def _set_software_entry_to_64_characters(data):
    """
    Make the only photometry_software entry of the first star list too long.

    Parameters
    ----------
    data : dict
        Raw 0.3 starlist-set data, modified in place.
    """
    data["star_lists"][0]["photometry_software"] = ["x" * 64]


def _set_100001_star_items(data):
    """
    Give the first star list one star item more than 0.4.0 allows.

    Parameters
    ----------
    data : dict
        Raw 0.3 starlist-set data, modified in place; the first star list
        gets 100001 copies of its first star item.
    """
    item = data["star_lists"][0]["staritems"][0]
    data["star_lists"][0]["staritems"] = [item] * 100_001


@pytest.mark.parametrize(
    "mutate, path, problem",
    [
        pytest.param(
            _set_star_item_x_to_infinity,
            "star_lists[0].staritems[1].x",
            "requires a finite number",
            id="inf-star-item",
        ),
        pytest.param(
            _set_site_elev_to_nan,
            "star_lists[1].site_elev",
            "requires a finite number",
            id="nan-star-list",
        ),
        pytest.param(
            _set_bkgd_count_to_the_string_infinity,
            "star_lists[0].staritems[0].bkgd_count",
            "requires a finite number",
            id="inf-as-string",
        ),
        pytest.param(
            _set_site_elev_to_numpy_infinity,
            "star_lists[0].site_elev",
            "requires a finite number",
            id="inf-as-numpy",
        ),
        pytest.param(
            _set_software_entry_to_64_characters,
            "star_lists[0].photometry_software[0]",
            "has 64 characters; schema version 0.4.0 allows at most 63",
            id="software-entry-too-long",
        ),
        pytest.param(
            _set_100001_star_items,
            "star_lists[0].staritems",
            "has 100001 entries; schema version 0.4.0 allows at most 100000",
            id="too-many-star-items",
        ),
    ],
)
def test_0_3_file_the_new_generation_cannot_hold_is_rejected(mutate, path, problem):
    # A 0.3 file may hold a value 0.4.0 has no form for: an infinite or NaN
    # float (also as a string, which the frozen 0.3 model accepts), a string
    # longer than 0.4.0 allows, or more star items than it allows.
    # The migration refuses it naming the field: through the model that is
    # the usual ValidationError, a problem with the file and not a
    # MigrationResultError; from upgrade() it is the ValueError itself. The
    # refusal comes before the warning, and pytest treats an unexpected
    # SchemaMigrationWarning as an error, so none is issued.
    data = _written_at_0_3()
    mutate(data)

    with pytest.raises(ValidationError) as excinfo:
        StarListSet.model_validate(data)
    error = excinfo.value.errors()[0]
    assert error["type"] == "value_error"
    assert error["loc"] == ()
    assert path in error["msg"]
    assert problem in error["msg"]
    assert "cannot be upgraded to it" in error["msg"]
    assert type(error["ctx"]["error"]) is ValueError

    with pytest.raises(ValueError, match=re.escape(path)) as excinfo:
        upgrade(data)
    assert type(excinfo.value) is ValueError

    # The file itself is valid for the schema it was written for.
    validate_as_written(data)


@pytest.mark.parametrize(
    "name, limit", _0_4_MAX_STRING_LENGTHS.items(), ids=list(_0_4_MAX_STRING_LENGTHS)
)
def test_0_3_string_over_its_0_4_limit_is_rejected(name, limit):
    # Every string field 0.4.0 caps is refused one character over its limit,
    # naming the field and the limit, through the model and from upgrade();
    # the file itself is still valid for 0.3.
    data = _written_at_0_3()
    data["star_lists"][0][name] = "x" * (limit + 1)
    path = f"star_lists[0].{name}"

    with pytest.raises(ValidationError) as excinfo:
        StarListSet.model_validate(data)
    error = excinfo.value.errors()[0]
    assert path in error["msg"]
    assert f"allows at most {limit}" in error["msg"]

    with pytest.raises(ValueError, match=re.escape(path)):
        upgrade(data)

    validate_as_written(data)


@pytest.mark.parametrize(
    "data",
    [
        pytest.param({"schema_version": "0.3.2"}, id="no-star_lists"),
        pytest.param({"schema_version": "0.3.2", "star_lists": {"a": 1}}, id="mapping"),
        pytest.param(
            {"schema_version": "0.3.2", "star_lists": [1, None]},
            id="non-mapping-entries",
        ),
        pytest.param(
            {"schema_version": "0.3.2", "star_lists": [{"staritems": "not a list"}]},
            id="staritems-not-a-list",
        ),
        pytest.param(
            {"schema_version": "0.3.2", "star_lists": [{"staritems": [1, None]}]},
            id="non-mapping-star-items",
        ),
    ],
)
def test_0_3_migration_passes_unrecognised_shapes_through(data):
    # The 0.3 -> 0.4 migration only touches a list or tuple of star lists or
    # star items, and within it only mapping or model entries. Anything else
    # is returned unchanged for validation to reject. These shapes reach the
    # migration only when it is called directly, since the frozen model
    # refuses them.
    snapshot = copy.deepcopy(data)
    assert _MIGRATIONS["0.3"].func(data) == snapshot


def test_0_3_migration_accepts_data_built_in_python():
    # upgrade() called directly passes values through unconverted, so the
    # migration must handle a tuple of frozen 0.3 StarList instances, whose
    # filter is a frozen 0.3 enum member; the instances, built by the frozen
    # model, have already lost the keys it ignores. The result is valid for
    # the current model, which reads that member as its own enum.
    data = _written_at_0_3()
    data["star_lists"] = tuple(
        _aavso_starlist_v0_3.StarList.model_validate(star_list)
        for star_list in data["star_lists"]
    )

    with pytest.warns(SchemaMigrationWarning):
        result = upgrade(data)

    star_list = StarListSet.model_validate(result).star_lists[0]
    assert star_list.stack is None
    assert star_list.filter is AAVSOFilters.TG


def test_0_3_migration_float_field_lists_match_the_frozen_models():
    # The migration names the 0.3 float fields as literals, since it is
    # frozen once released; they must be exactly the float fields of the
    # frozen 0.3 models, so none escapes the check for infinity and NaN.
    assert _0_3_STAR_ITEM_FLOATS == float_fields(_aavso_starlist_v0_3.StarItem)
    assert _0_3_STAR_LIST_FLOATS == float_fields(_aavso_starlist_v0_3.StarList)


def test_0_3_migration_limits_match_the_0_4_models():
    # The migration names the 0.4.0 string and star item limits as literals,
    # since it is frozen once released; they must be the limits of the frozen
    # 0.4 models.
    properties = _aavso_starlist_v0_4.StarList.model_json_schema()["properties"]

    assert _0_4_MAX_STRING_LENGTHS == {
        name: prop["maxLength"] for name, prop in properties.items() if "maxLength" in prop
    }
    assert (
        _0_4_MAX_PHOTOMETRY_SOFTWARE_ENTRY
        == properties["photometry_software"]["items"]["maxLength"]
    )
    assert _0_4_MAX_STARITEMS == properties["staritems"]["maxItems"]


def test_0_3_file_at_the_new_limits_is_migrated_to_0_4():
    # A 0.3 file with strings exactly as long as 0.4.0 allows and exactly
    # 100000 star items is migrated to a file the frozen 0.4 model accepts,
    # which keeps them. (Some of those strings are no longer valid in 1.0.0,
    # which the 0.4 to 1 migration refuses.)
    data = _written_at_0_3()
    star_list = data["star_lists"][0]
    for name, limit in _0_4_MAX_STRING_LENGTHS.items():
        star_list[name] = "x" * limit
    star_list["photometry_software"] = ["x" * _0_4_MAX_PHOTOMETRY_SOFTWARE_ENTRY]
    star_list["staritems"] = [star_list["staritems"][1]] * _0_4_MAX_STARITEMS

    migrated = _MIGRATIONS["0.3"].func(copy.deepcopy(data))
    migrated["schema_version"] = _MIGRATIONS["0.3"].target
    result = _aavso_starlist_v0_4.StarListSet.model_validate(migrated)

    upgraded = result.star_lists[0]
    for name, limit in _0_4_MAX_STRING_LENGTHS.items():
        assert getattr(upgraded, name) == "x" * limit
    assert upgraded.photometry_software == ["x" * _0_4_MAX_PHOTOMETRY_SOFTWARE_ENTRY]
    assert len(upgraded.staritems) == 100_000


def test_0_3_star_items_given_as_a_generator_are_all_upgraded():
    # A 0.3 dict built in Python may hold its star items as a generator,
    # which can be read only once; the model reads it once for both the
    # frozen 0.3 validation and the migration, so no star item is lost.
    data = _written_at_0_3()
    items = data["star_lists"][0]["staritems"][:2]
    data["star_lists"][0]["staritems"] = (item for item in items)

    with pytest.warns(SchemaMigrationWarning):
        result = StarListSet.model_validate(data)

    assert len(result.star_lists[0].staritems) == 2


def _written_at_0_4(version="0.4.1"):
    """
    Load the sample file of the 0.4 generation, stamped at a version of it.

    Parameters
    ----------
    version : {"0.4.0", "0.4.1"}, optional
        The version the file is to state. Default ``"0.4.1"``, the version
        ``before.json`` of the 0.4 migration states.

    Returns
    -------
    dict
        A deep copy of ``before.json`` of the 0.4 migration, which carries a
        lowercase observer code and a number written as a string, with
        ``schema_version`` set to ``version``. The frozen 0.4 model, a copy
        of 0.4.1, validates it at either version.
    """
    before, _ = _before_after_pair("0.4")
    data = copy.deepcopy(before)
    data["schema_version"] = version
    return data


@pytest.mark.parametrize("version", ["0.4.0", "0.4.1"])
def test_0_4_file_is_validated_against_the_frozen_0_4_model(version):
    # validate_as_written checks a file from any 0.4 version against the
    # frozen 0.4 model, without upgrading it, so without a warning.
    result = validate_as_written(_written_at_0_4(version))

    assert isinstance(result, _aavso_starlist_v0_4.StarListSet)
    assert result.schema_version == version


@pytest.mark.parametrize("version", ["0.4.0", "0.4.1"])
def test_0_4_file_is_upgraded_with_one_warning_naming_the_0_4_step(version):
    # A file from any 0.4 version is upgraded with one warning, naming the
    # 0.4 migration but not the earlier ones, to the current version.
    with pytest.warns(SchemaMigrationWarning) as record:
        result = StarListSet.model_validate(_written_at_0_4(version))

    assert len(record) == 1
    message = str(record[0].message)
    assert repr(version) in message
    assert _MIGRATIONS["0.4"].summary in message
    assert _MIGRATIONS["0.3"].summary not in message
    assert "dropped" not in message

    assert result.schema_version == SCHEMA_VERSION


@pytest.mark.parametrize("value", ["mmu", "Mmu", "abcdefg", "MMU"])
def test_observer_code_of_a_0_4_file_is_uppercased(value):
    # AAVSO issues observer codes in uppercase, so a 0.4 code of 2 to 7
    # letters in any case is the same code uppercased; one already in
    # uppercase is kept.
    data = _written_at_0_4()
    data["star_lists"][0]["observer"] = value

    with pytest.warns(SchemaMigrationWarning):
        result = StarListSet.model_validate(data)
    migrated = _MIGRATIONS["0.4"].func(copy.deepcopy(data))

    assert result.star_lists[0].observer == value.upper()
    assert migrated["star_lists"][0]["observer"] == value.upper()


@pytest.mark.parametrize(
    "level, name, value, expected",
    [
        pytest.param("star list", "bias", "1000", 1000.0, id="string-float"),
        pytest.param("star list", "egain", True, 1.0, id="bool-float"),
        pytest.param("star list", "width", "2048", 2048, id="string-int"),
        pytest.param("star list", "stack", True, 1, id="bool-int"),
        pytest.param("star item", "x", "12.5", 12.5, id="string-star-item"),
        pytest.param("star item", "bkgd_count", False, 0.0, id="bool-star-item"),
    ],
)
def test_number_given_as_a_string_or_bool_in_a_0_4_file_becomes_the_number(
    level, name, value, expected
):
    # The 0.4 models read a bool or a numeric string in a numeric field as a
    # number (pydantic's lax mode); 1.0.0 refuses both, so the migration
    # writes the number the 0.4 reader produced, of the field's own type.
    data = _written_at_0_4()
    star_list = data["star_lists"][0]
    record = star_list if level == "star list" else star_list["staritems"][0]
    record[name] = value

    with pytest.warns(SchemaMigrationWarning):
        result = StarListSet.model_validate(data)
    migrated = _MIGRATIONS["0.4"].func(copy.deepcopy(data))["star_lists"][0]

    upgraded = result.star_lists[0]
    if level == "star item":
        upgraded = upgraded.staritems[0]
        migrated = migrated["staritems"][0]
    assert getattr(upgraded, name) == expected
    assert migrated[name] == expected
    assert type(migrated[name]) is type(expected)


@pytest.mark.parametrize(
    "level, name, expected",
    [
        pytest.param("star list", "bias", 1.0, id="float"),
        pytest.param("star list", "width", 1, id="int"),
        pytest.param("star item", "x", 1.0, id="star-item"),
    ],
)
def test_numpy_bool_in_a_0_4_file_becomes_the_number(level, name, expected):
    # Data built in Python may hold a numpy bool, which the frozen 0.4 model
    # reads as a number and 1.0.0 refuses like a bool; the migration repairs
    # it, so the upgrade is not a MigrationResultError.
    data = _written_at_0_4()
    star_list = data["star_lists"][0]
    record = star_list if level == "star list" else star_list["staritems"][0]
    record[name] = np.bool_(True)

    with pytest.warns(SchemaMigrationWarning):
        result = StarListSet.model_validate(data)

    upgraded = result.star_lists[0]
    if level == "star item":
        upgraded = upgraded.staritems[0]
    assert getattr(upgraded, name) == expected
    assert type(getattr(upgraded, name)) is type(expected)


@pytest.mark.parametrize(
    "value",
    [True, np.bool_(True), "1", b"1", np.array("1"), np.array(True), np.array(b"1"),
     np.void(b"1")],
    ids=["bool", "numpy-bool", "str", "bytes", "str-array", "bool-array", "bytes-array",
         "void"],
)
def test_0_4_migration_repairs_every_kind_the_current_model_refuses(value):
    # Each kind of value _reject_bool_and_str refuses in a numeric field is
    # one the 0.4 to 1 migration repairs, in a float and an int field, so the
    # two lists cannot drift apart: the migrated data is valid for 1.0.0.
    data = _written_at_0_4()
    data["star_lists"][0]["bias"] = value
    data["star_lists"][0]["stack"] = value

    migrated = _MIGRATIONS["0.4"].func(copy.deepcopy(data))
    migrated["schema_version"] = SCHEMA_VERSION

    star_list = StarListSet.model_validate(migrated).star_lists[0]
    assert star_list.bias == 1.0
    assert star_list.stack == 1


@pytest.mark.parametrize(
    "name, value, expected",
    [
        pytest.param(name, value, expected, id=f"{name}-{value}")
        for name in _0_4_STAR_LIST_INTS
        for value, expected in [("5E+1", 50), ("0E+2", 0)]
        # 0 is valid for 0.4 only where the field allows it.
        if expected or name in ("adc_depth", "largest_usable_adu_value")
    ],
)
def test_decimal_with_an_exponent_in_a_0_4_int_field_becomes_the_int(name, value, expected):
    # The frozen 0.4 model reads a Decimal with an exponent in an int field
    # as the int, but plain data holds it as a string such as "5E+1", which
    # lax mode cannot read as an int; the migration reads it as a Decimal,
    # so the upgrade gives the int and not a MigrationResultError.
    data = _written_at_0_4()
    data["star_lists"][0][name] = decimal.Decimal(value)

    with pytest.warns(SchemaMigrationWarning):
        result = StarListSet.model_validate(data)

    assert getattr(result.star_lists[0], name) == expected
    assert type(getattr(result.star_lists[0], name)) is int


def test_decimal_with_an_exponent_in_a_0_4_float_field_becomes_the_float():
    # A Decimal with an exponent in a float field reaches the migration as a
    # string such as "1.5E+3", which lax mode reads as the float.
    data = _written_at_0_4()
    data["star_lists"][0]["bias"] = decimal.Decimal("1.5E+3")

    with pytest.warns(SchemaMigrationWarning):
        result = StarListSet.model_validate(data)

    assert result.star_lists[0].bias == 1500.0


# Numpy wrappers of a bool, a str or bytes, which lax mode reads as numbers,
# each with the number it reads as.
NUMPY_WRAPPERS = [
    pytest.param(np.array("5"), 5, id="str-array"),
    pytest.param(np.array(True), 1, id="bool-array"),
    pytest.param(np.array(b"5"), 5, id="bytes-array"),
    pytest.param(np.void(b"5"), 5, id="void"),
]


@pytest.mark.parametrize("value, expected", NUMPY_WRAPPERS)
@pytest.mark.parametrize("name", ["bias", "width"])
def test_numpy_wrapper_in_a_0_4_file_becomes_the_number(name, value, expected):
    # Data built in Python may hold a numpy wrapper of a bool, a str or
    # bytes in a numeric field, which the frozen 0.4 model reads as a number
    # and 1.0.0 refuses; the migration unwraps and repairs it.
    data = _written_at_0_4()
    data["star_lists"][0][name] = value

    with pytest.warns(SchemaMigrationWarning):
        result = StarListSet.model_validate(data)

    assert getattr(result.star_lists[0], name) == expected


def test_0_4_value_lax_mode_cannot_read_is_left_for_validation():
    # Only upgrade() called directly can pass a string the 0.4 reader would
    # not have read as a number; the migration leaves it as given, and the
    # current model then refuses it.
    data = _written_at_0_4()
    data["star_lists"][0]["bias"] = "high"
    data["star_lists"][0]["width"] = "wide"

    with pytest.warns(SchemaMigrationWarning):
        upgraded = upgrade(data)

    assert upgraded["star_lists"][0]["bias"] == "high"
    assert upgraded["star_lists"][0]["width"] == "wide"
    with pytest.raises(ValidationError):
        StarListSet.model_validate(upgraded)


def _set_one(path, value):
    """
    Make a function that sets one value of 0.4 sample data.

    Parameters
    ----------
    path : tuple of str or int
        The keys and indexes leading from the starlist set to the value, such
        as ``("star_lists", 0, "observer")``.
    value : object
        The value to set.

    Returns
    -------
    callable
        Takes starlist-set data and sets ``value`` at ``path`` in place.
    """
    def mutate(data):
        """
        Set the value in starlist-set data.

        Parameters
        ----------
        data : dict
            Raw starlist-set data, modified in place.
        """
        parent = data
        for key in path[:-1]:
            parent = parent[key]
        parent[path[-1]] = value

    return mutate


def _set_1001_star_lists(data):
    """
    Give 0.4 sample data one star list more than 1.0.0 allows.

    Parameters
    ----------
    data : dict
        Raw starlist-set data, modified in place.
    """
    data["star_lists"] = [data["star_lists"][1]] * (_1_MAX_STAR_LISTS + 1)


@pytest.mark.parametrize(
    "mutate, path, problem",
    [
        pytest.param(
            _set_one(("star_lists", 0, "observer"), "MM1"),
            "star_lists[0].observer",
            "requires an observer code of 2 to 7 letters",
            id="observer-digit",
        ),
        pytest.param(
            _set_one(("star_lists", 1, "observer"), "M"),
            "star_lists[1].observer",
            "requires an observer code of 2 to 7 letters",
            id="observer-one-letter",
        ),
        pytest.param(
            _set_one(("star_lists", 0, "observer"), "M U"),
            "star_lists[0].observer",
            "requires an observer code of 2 to 7 letters",
            id="observer-space",
        ),
        pytest.param(
            _set_one(("star_lists", 0, "obs_time"), "not a time"),
            "star_lists[0].obs_time",
            "requires obs_time of the form YYYY-MM-DDTHH:MM:SS[.ffffff][Z|±HH:MM]",
            id="obs-time-text",
        ),
        pytest.param(
            _set_one(("star_lists", 0, "obs_time"), "2021-06-15"),
            "star_lists[0].obs_time",
            "requires obs_time of the form YYYY-MM-DDTHH:MM:SS[.ffffff][Z|±HH:MM]",
            id="obs-time-date-only",
        ),
        pytest.param(
            _set_one(("star_lists", 0, "obs_time"), "2021-06-15t03:45:00"),
            "star_lists[0].obs_time",
            "requires obs_time of the form YYYY-MM-DDTHH:MM:SS[.ffffff][Z|±HH:MM]",
            id="obs-time-lowercase-t",
        ),
        pytest.param(
            _set_one(("star_lists", 0, "obs_time"), "2021-13-01T00:00:00"),
            "star_lists[0].obs_time",
            "requires obs_time of the form YYYY-MM-DDTHH:MM:SS[.ffffff][Z|±HH:MM]",
            id="obs-time-month-13",
        ),
        pytest.param(
            _set_one(("star_lists", 0, "obs_time"), "2021-02-30T00:00:00"),
            "star_lists[0].obs_time",
            "requires obs_time of the form YYYY-MM-DDTHH:MM:SS[.ffffff][Z|±HH:MM]",
            id="obs-time-february-30",
        ),
        pytest.param(
            _set_one(("star_lists", 0, "obs_time"), "2021-06-15T03:45:00+00:60"),
            "star_lists[0].obs_time",
            "requires obs_time of the form YYYY-MM-DDTHH:MM:SS[.ffffff][Z|±HH:MM]",
            id="obs-time-offset-minute-60",
        ),
        pytest.param(
            _set_one(("star_lists", 0, "tel_manufac"), "x" * 32),
            "star_lists[0].tel_manufac",
            "has 32 characters; schema version 1.0.0 allows at most 31",
            id="tel-manufac-too-long",
        ),
        pytest.param(
            _set_one(("star_lists", 1, "tel_model"), "x" * 63),
            "star_lists[1].tel_model",
            "has 63 characters; schema version 1.0.0 allows at most 31",
            id="tel-model-too-long",
        ),
        pytest.param(
            _set_one(("star_lists", 0, "width"), 32768),
            "star_lists[0].width",
            "is 32768; schema version 1.0.0 allows at most 32767",
            id="width-one-over",
        ),
        pytest.param(
            _set_one(("star_lists", 0, "height"), 32768),
            "star_lists[0].height",
            "is 32768; schema version 1.0.0 allows at most 32767",
            id="height-one-over",
        ),
        pytest.param(
            _set_one(("star_lists", 0, "stack"), 32768),
            "star_lists[0].stack",
            "is 32768; schema version 1.0.0 allows at most 32767",
            id="stack-one-over",
        ),
        pytest.param(
            _set_one(("star_lists", 0, "width"), 40000),
            "star_lists[0].width",
            "is 40000; schema version 1.0.0 allows at most 32767",
            id="width-too-large",
        ),
        pytest.param(
            _set_one(("star_lists", 0, "height"), "70000"),
            "star_lists[0].height",
            "is 70000; schema version 1.0.0 allows at most 32767",
            id="height-too-large-as-string",
        ),
        pytest.param(
            _set_one(("star_lists", 0, "stack"), 10**400),
            "star_lists[0].stack",
            "schema version 1.0.0 allows at most 32767",
            id="stack-huge",
        ),
        pytest.param(
            _set_one(("star_lists", 0, "adc_depth"), 65),
            "star_lists[0].adc_depth",
            "is 65; schema version 1.0.0 allows at most 64",
            id="adc-depth-too-large",
        ),
        pytest.param(
            _set_one(("star_lists", 0, "largest_usable_adu_value"), 2**31),
            "star_lists[0].largest_usable_adu_value",
            "is 2147483648; schema version 1.0.0 allows at most 2147483647",
            id="adu-value-too-large",
        ),
        pytest.param(
            _set_one(("star_lists", 0, "photometry_software"), ["bandaid 1.2.3"] * 101),
            "star_lists[0].photometry_software",
            "has 101 entries; schema version 1.0.0 allows at most 100",
            id="too-many-software-entries",
        ),
        pytest.param(
            _set_1001_star_lists,
            "star_lists",
            "has 1001 entries; schema version 1.0.0 allows at most 1000",
            id="too-many-star-lists",
        ),
    ],
)
def test_0_4_file_the_new_generation_cannot_hold_is_rejected(mutate, path, problem):
    # A 0.4 file may hold a value 1.0.0 has no form for: an observer code
    # that is not 2 to 7 letters, an obs_time not of the 1.0.0 form or not a
    # real date and time, a telescope string longer than 31 characters, an integer
    # over its new maximum (also as a string, which the frozen 0.4 model
    # accepts), or more star lists or software entries than 1.0.0 allows.
    # The migration refuses it naming the field: through the model that is
    # the usual ValidationError, not a MigrationResultError; from upgrade()
    # it is the ValueError itself. The refusal comes before the warning, so
    # none is issued.
    data = _written_at_0_4()
    mutate(data)

    with pytest.raises(ValidationError) as excinfo:
        StarListSet.model_validate(data)
    error = excinfo.value.errors()[0]
    assert error["type"] == "value_error"
    assert error["loc"] == ()
    assert f"{path} " in error["msg"]
    assert problem in error["msg"]
    assert error["msg"].endswith("cannot be upgraded to it")
    assert type(error["ctx"]["error"]) is ValueError

    with pytest.raises(ValueError, match=re.escape(path)) as excinfo:
        upgrade(data)
    assert type(excinfo.value) is ValueError

    # The file itself is valid for the schema it was written for.
    validate_as_written(data)


def test_0_4_migration_limits_match_the_current_models():
    # The migration names the 1.0.0 limits as literals, since it is frozen
    # once released; they must be the limits of the 1.0.0 models. Valid while
    # 1 is the current generation: once it is left, compare against the
    # frozen 1 models instead.
    set_properties = StarListSet.model_json_schema()["properties"]
    properties = StarList.model_json_schema()["properties"]

    for name, limit in _1_MAX_STRING_LENGTHS.items():
        assert properties[name]["maxLength"] == limit
    assert _1_INT_MAXIMA == {
        name: next(
            entry["maximum"]
            for entry in properties[name].get("anyOf", [properties[name]])
            if "maximum" in entry
        )
        for name in int_fields(StarList)
    }
    assert _1_MAX_PHOTOMETRY_SOFTWARE == properties["photometry_software"]["maxItems"]
    assert _1_MAX_STAR_LISTS == set_properties["star_lists"]["maxItems"]
    # The migration accepts the letters in any case and uppercases them.
    assert properties["observer"]["pattern"] == "^" + _1_OBSERVER.pattern.replace(
        "A-Za-z", "A-Z"
    ) + "$"


@pytest.mark.parametrize(
    "value, valid",
    [
        ("2021-06-15T03:45:00", True),
        ("2021-06-15T03:45:00Z", True),
        ("2021-06-15 03:45:00", True),
        ("2021-06-15T03:45:00.123456", True),
        ("2021-06-15T03:45:00+05:30", True),
        ("2021-06-15T03:45:00-07:00", True),
        ("2020-02-29T00:00:00", True),
        ("2021-06-15", False),
        ("2021-06-15T03", False),
        ("20210615T034500", False),
        ("2021-W24-2T03:45:00", False),
        ("2021-06-15t03:45:00", False),
        ("2021-06-15T03:45:00.1234567", False),
        ("2021-06-15T03:45:00+0530", False),
        ("2021-13-01T00:00:00", False),
        ("2021-06-15T24:00:00", False),
        ("not a time", False),
        ("2021-06-15T03:45:00+00:60", False),
        ("2021-06-15T03:45:00+00:99", False),
        ("2021-06-15T03:45:00+24:00", False),
        ("2021-06-15T03:45:60", False),
        ("2016-12-31T23:59:60Z", False),
        ("2021-02-30T00:00:00", False),
        ("2021-02-29T00:00:00", False),
        ("2021-00-10T00:00:00", False),
        ("2021-06-15T03:45:00z", False),
        ("2021-06-15x03:45:00", False),
        ("2021-06-15T03:45:00\n", False),
    ],
)
def test_0_4_migration_obs_time_rule_matches_the_current_model(value, valid):
    # The migration carries its own frozen copy of the 1.0.0 obs_time rule
    # (pattern and calendar check), since it is frozen once released; it must
    # agree with the live model, pattern and validator together. Valid while
    # 1 is the current generation: once it is left, compare against the
    # frozen 1 models instead.
    data = StarList.from_examples().model_dump()
    data["obs_time"] = value
    try:
        StarList.model_validate(data)
    except ValidationError:
        accepted = False
    else:
        accepted = True

    assert accepted is valid
    assert _1_is_iso_datetime(value) is valid


def test_obs_time_pattern_matches_the_frozen_migration_rule():
    # The live pattern and the migration's frozen copy are separate literals;
    # while 1 is the current generation they must be the same regex.
    assert _OBS_TIME_PATTERN == "^" + _1_OBS_TIME.pattern + "$"


def test_0_4_migration_number_field_lists_match_the_frozen_0_4_models():
    # The migration names the 0.4 float and int fields as literals, since it
    # is frozen once released; they must be exactly the numeric fields of the
    # frozen 0.4 models, so none escapes the repair of booleans and strings.
    assert _0_4_STAR_ITEM_FLOATS == float_fields(_aavso_starlist_v0_4.StarItem)
    assert _0_4_STAR_LIST_FLOATS == float_fields(_aavso_starlist_v0_4.StarList)
    assert _0_4_STAR_LIST_INTS == int_fields(_aavso_starlist_v0_4.StarList)
    assert int_fields(_aavso_starlist_v0_4.StarItem) == ()


def test_0_4_file_at_the_new_limits_is_upgraded():
    # A 0.4 file exactly at every 1.0.0 limit (telescope strings, integer
    # maxima, star lists and software entries) is upgraded, with the
    # warning, and keeps its values.
    data = _written_at_0_4()
    star_list = data["star_lists"][1]
    for name, limit in _1_MAX_STRING_LENGTHS.items():
        star_list[name] = "x" * limit
    for name, limit in _1_INT_MAXIMA.items():
        star_list[name] = limit
    star_list["photometry_software"] = ["bandaid 1.2.3"] * _1_MAX_PHOTOMETRY_SOFTWARE
    data["star_lists"] = [star_list] * _1_MAX_STAR_LISTS

    with pytest.warns(SchemaMigrationWarning):
        result = StarListSet.model_validate(data)

    assert len(result.star_lists) == _1_MAX_STAR_LISTS
    upgraded = result.star_lists[0]
    for name, limit in _1_MAX_STRING_LENGTHS.items():
        assert getattr(upgraded, name) == "x" * limit
    for name, limit in _1_INT_MAXIMA.items():
        assert getattr(upgraded, name) == limit
    assert len(upgraded.photometry_software) == _1_MAX_PHOTOMETRY_SOFTWARE


def test_0_4_migration_handles_tuples_and_other_mappings():
    # upgrade() called directly may pass star lists and star items as tuples
    # and records as mappings that are not dicts; the migration reads them
    # all, repairs what it repairs and returns lists of dicts.
    data = _written_at_0_4()
    star_list = data["star_lists"][0]
    star_list["staritems"] = tuple(
        MappingProxyType({**item, "x": "12.5"}) for item in star_list["staritems"]
    )
    data["star_lists"] = (_Mapping(star_list),)

    migrated = _MIGRATIONS["0.4"].func(data)

    [result] = migrated["star_lists"]
    assert isinstance(migrated["star_lists"], list)
    assert result["observer"] == "MMU"
    assert result["bias"] == 1000.0
    assert isinstance(result["staritems"], list)
    assert [item["x"] for item in result["staritems"]] == [12.5, 12.5]


@pytest.mark.parametrize(
    "data",
    [
        {"star_lists": "garbage"},
        {"star_lists": [5, None]},
        {"star_lists": [{"observer": 7, "obs_time": None, "staritems": "garbage"}]},
        {"star_lists": [{"staritems": [5]}]},
    ],
    ids=["star-lists-not-a-list", "star-list-not-a-mapping", "wrong-types", "star-item-int"],
)
def test_0_4_migration_passes_unrecognised_shapes_through(data):
    # The 0.4 -> 1 migration only touches what it recognises; anything else
    # is passed through for the current model to reject.
    assert _MIGRATIONS["0.4"].func(copy.deepcopy(data)) == data


# The generations older than the current one, each a key of _MIGRATIONS whose
# before.json sample (at 0.0.1.dev451+gde1568f, 0.2.1, 0.3.2 and 0.4.1) is used.
OLDER_GENERATIONS = ["legacy", "0.2", "0.3", "0.4"]

# The older generations whose frozen models ignore keys they do not define;
# from 0.4 on the models forbid them.
IGNORING_GENERATIONS = ["legacy", "0.2", "0.3"]


@pytest.mark.parametrize("generation", OLDER_GENERATIONS)
def test_older_file_with_a_datetime_obs_time_is_rejected(generation):
    # The data is validated against the frozen model as given, so a datetime
    # as obs_time, which no older schema allowed, is refused there as not a
    # string. Before the fix the model validated a converted copy, in which
    # the datetime had already become a string, and upgraded it silently.
    data, _ = _before_after_pair(generation)
    data["star_lists"][0]["obs_time"] = datetime.datetime(2021, 6, 15, 3, 45)

    with pytest.raises(ValidationError) as excinfo:
        StarListSet.model_validate(data)

    error = excinfo.value.errors()[0]
    assert error["type"] == "string_type"
    assert error["loc"] == ("star_lists", 0, "obs_time")


def _unreadable():
    """
    Build a generator that fails if anything reads it.

    Returns
    -------
    generator
        A generator whose first ``next`` raises ``AssertionError``.
    """
    raise AssertionError("a value that must not be read was read")
    yield  # makes this a generator; never reached


def _closed_file():
    """
    Build a closed text file, which raises ``ValueError`` if iterated.

    Returns
    -------
    io.StringIO
        A closed ``StringIO``.
    """
    closed = io.StringIO("notes")
    closed.close()
    return closed


@pytest.mark.parametrize("generation", IGNORING_GENERATIONS)
def test_values_under_undefined_keys_of_an_older_file_are_not_read(generation):
    # A key no frozen model defines is dropped at every level before anything
    # reads its value: the frozen model ignores it, so reading it could only
    # fail (a closed file) or never finish (an endless generator). Each value
    # here fails if read.
    data, _ = _before_after_pair(generation)
    data["pipeline"] = _unreadable()
    data["star_lists"][0]["instrument_notes"] = _closed_file()
    data["star_lists"][0]["staritems"][0]["flag"] = _unreadable()

    with pytest.warns(SchemaMigrationWarning):
        result = StarListSet.model_validate(data).model_dump(mode="json")

    assert "pipeline" not in result
    assert "instrument_notes" not in result["star_lists"][0]
    assert "flag" not in result["star_lists"][0]["staritems"][0]


# Keys the 0.3 models define but the frozen models of an earlier generation
# do not, each with the level it is on.
_DEFINED_FROM_0_3 = {
    "legacy": [("star list", "bias"), ("star list", "photometry_software"),
               ("star list", "absolute_focus"), ("star item", "fwhm")],
    "0.2": [("star list", "bias"), ("star list", "photometry_software"),
            ("star item", "fwhm")],
}


@pytest.mark.parametrize(
    "generation, level, key",
    [
        pytest.param(generation, level, key, id=f"{generation}-{key}")
        for generation, keys in _DEFINED_FROM_0_3.items()
        for level, key in keys
    ],
)
@pytest.mark.parametrize("make_value", [_unreadable, _closed_file], ids=["generator", "closed"])
def test_keys_a_later_generation_defines_are_not_read_from_an_older_file(
    generation, level, key, make_value
):
    # The pruning follows the frozen model of the file's own generation, not
    # the 0.3 models, so a key only a later generation defines is dropped
    # unread from a legacy or 0.2 file like any other unknown key, and the
    # upgraded file has that field's default. Before the fix an endless
    # generator there hung the read and a closed file failed it.
    data, _ = _before_after_pair(generation)
    star_list = data["star_lists"][0]
    record = star_list if level == "star list" else star_list["staritems"][0]
    record[key] = make_value()

    with pytest.warns(SchemaMigrationWarning):
        result = StarListSet.model_validate(data)

    upgraded = result.star_lists[0]
    if level == "star item":
        upgraded = upgraded.staritems[0]
    expected = ["unknown"] if key == "photometry_software" else None
    assert getattr(upgraded, key) == expected


@pytest.mark.parametrize("value", ["garbage", float("inf"), 5])
def test_legacy_bias_is_dropped_whatever_its_value(value):
    # No legacy schema defined bias, so its value was never validated and is
    # dropped before the migrations: a wrong value is not reported as a bug
    # in a migration (MigrationResultError), an infinite one is not refused
    # as unupgradeable, and a valid-looking one is not carried over.
    data, _ = _before_after_pair("legacy")
    data["star_lists"][0]["bias"] = value

    with pytest.warns(SchemaMigrationWarning):
        result = StarListSet.model_validate(data)

    assert result.star_lists[0].bias is None


@pytest.mark.parametrize("generation", OLDER_GENERATIONS)
@pytest.mark.parametrize("make_value", [_unreadable, _closed_file], ids=["generator", "closed"])
def test_wrong_value_under_a_scalar_field_is_reported_where_it_is(generation, make_value):
    # A value under a scalar field is validated as given, not read first, so
    # a closed file or a generator as observer is reported by pydantic as
    # not a string at its own location. Before the fix it was iterated: the
    # closed file failed as a value_error with an empty location, and an
    # endless generator never finished.
    data, _ = _before_after_pair(generation)
    data["star_lists"][0]["observer"] = make_value()

    with pytest.raises(ValidationError) as excinfo:
        StarListSet.model_validate(data)

    error = excinfo.value.errors()[0]
    assert error["type"] == "string_type"
    assert error["loc"] == ("star_lists", 0, "observer")


class _Inner(BaseModel):
    """
    A throwaway nested model for the pruning test, ignoring extra keys.
    """
    value: int


class _InnerForbidding(BaseModel):
    """
    A throwaway nested model for the pruning test, forbidding extra keys.
    """
    model_config = ConfigDict(extra="forbid")
    value: int


class _OuterForbidding(BaseModel):
    """
    A throwaway model that forbids extra keys, as a frozen 0.4 model would.
    """
    model_config = ConfigDict(extra="forbid")
    items: list[_InnerForbidding]


class _OuterIgnoring(BaseModel):
    """
    A throwaway model that ignores extra keys, as the older frozen models do.
    """
    items: list[_Inner] | None = None


def test_pruning_keeps_unknown_keys_for_a_model_that_forbids_them():
    # A frozen model that forbids extra keys (a future frozen 0.4 model) must
    # see them to reject them, so the pruning keeps them, unread, at every
    # level that forbids them, and validation then rejects them where they
    # are; a model that ignores them gets them dropped, unread. Each level
    # follows its own model's config, and a list of models given as a
    # generator is read into a list.
    unread_outer, unread_inner = _unreadable(), _unreadable()
    data = {"items": (item for item in [{"value": 1, "xi": unread_inner}]),
            "extra": unread_outer}
    kept = _prune_and_materialize(data, _OuterForbidding)
    assert list(kept) == ["items", "extra"]
    assert kept["extra"] is unread_outer
    assert kept["items"] == [{"value": 1, "xi": unread_inner}]
    assert kept["items"][0]["xi"] is unread_inner

    with pytest.raises(ValidationError) as excinfo:
        _OuterForbidding.model_validate(kept)
    errors = {error["loc"]: error["type"] for error in excinfo.value.errors()}
    assert errors == {("items", 0, "xi"): "extra_forbidden", ("extra",): "extra_forbidden"}

    data = {"items": (item for item in [{"value": 1, "extra": _unreadable()}]),
            "extra": _unreadable()}
    assert _prune_and_materialize(data, _OuterIgnoring) == {"items": [{"value": 1}]}


class _BareList(BaseModel):
    """
    A throwaway model with a list field annotated as bare ``typing.List``.
    """
    names: List  # noqa: UP006 -- the bare typing.List is what is tested


def test_pruning_reads_a_bare_typing_list_field():
    # A field annotated as bare typing.List has no item type; the pruning
    # reads a generator under it into a list like any other list field.
    # Before the fix finding its item type raised IndexError.
    kept = _prune_and_materialize({"names": (name for name in ["a", "b"])}, _BareList)
    assert kept == {"names": ["a", "b"]}


def _self_referential():
    """
    Build a list that holds itself as its second item.

    Returns
    -------
    list
        ``["a", <the list itself>]``.
    """
    looped = ["a"]
    looped.append(looped)
    return looped


@pytest.mark.parametrize(
    "make_item", [_unreadable, _closed_file, _self_referential],
    ids=["generator", "closed", "self-referential"],
)
def test_wrong_item_of_a_list_of_strings_is_reported_at_its_index(make_item):
    # A list of plain values such as photometry_software is read one level
    # only, so an item that is not a string (a generator, a closed file, the
    # list itself) is reported by pydantic as string_type at its index. Only
    # the 0.3 frozen model defines photometry_software; legacy and 0.2 files
    # have it dropped unread, which
    # test_keys_a_later_generation_defines_are_not_read_from_an_older_file
    # covers. Before the fix every item was walked all the way down: the
    # closed file failed as a value_error with an empty loc, the
    # self-referential list raised RecursionError, an endless generator hung.
    data, _ = _before_after_pair("0.3")
    item = make_item()
    data["star_lists"][0]["photometry_software"] = ["ok", item]
    if isinstance(item, list):
        data["star_lists"][0]["photometry_software"] = item

    with pytest.raises(ValidationError) as excinfo:
        StarListSet.model_validate(data)

    error = excinfo.value.errors()[0]
    assert error["type"] == "string_type"
    assert error["loc"] == ("star_lists", 0, "photometry_software", 1)


def test_changed_frozen_instance_of_the_files_generation_is_validated():
    # The frozen models do not revalidate their own instances, so a frozen
    # 0.3 StarList changed after it was built is read as its fields and
    # validated like a dict: an int as observer is reported as string_type
    # at the field. Before the fix the instance was accepted as given and
    # the bad value reached the current model, reported as a package bug
    # (MigrationResultError).
    data, _ = _before_after_pair("0.3")
    star_list = _aavso_starlist_v0_3.StarList.model_validate(data["star_lists"][0])
    star_list.observer = 12345
    data["star_lists"][0] = star_list

    with pytest.raises(ValidationError) as excinfo:
        StarListSet.model_validate(data)

    error = excinfo.value.errors()[0]
    assert error["type"] == "string_type"
    assert error["loc"] == ("star_lists", 0, "observer")


def test_unchanged_frozen_instance_of_the_files_generation_is_upgraded():
    # The other side of the revalidation fix: an unchanged frozen 0.3
    # StarList in a 0.3 dict is read as its fields, its frozen enum member
    # filter included, validated, and upgraded with the warning.
    data, _ = _before_after_pair("0.3")
    data["star_lists"][0] = _aavso_starlist_v0_3.StarList.model_validate(data["star_lists"][0])

    with pytest.warns(SchemaMigrationWarning):
        result = StarListSet.model_validate(data)

    assert result.star_lists[0].filter is AAVSOFilters.TG


@pytest.mark.parametrize("generation", OLDER_GENERATIONS)
def test_older_file_with_a_decimal_float_is_upgraded(generation):
    # The boundary of the datetime fix: a value the frozen model accepts for
    # a float field, such as a Decimal, is still validated and upgraded.
    data, _ = _before_after_pair(generation)
    data["star_lists"][0]["exposure"] = decimal.Decimal("30.5")

    with pytest.warns(SchemaMigrationWarning):
        result = StarListSet.model_validate(data)

    assert result.star_lists[0].exposure == 30.5


class _IterableFloat(float):
    """A float that is also iterable, as a scalar from some library might be."""

    def __iter__(self):
        """
        Iterate over nothing.

        Returns
        -------
        iterator
            An empty iterator.
        """
        return iter(())


def test_0_3_iterable_float_is_upgraded_as_a_float():
    # A float subclass that defines __iter__ is read as a float by the frozen
    # 0.3 model, so it must reach the migration as one: the upgrade gives
    # the same result as the plain float. Before the fix the conversion to
    # plain data turned it into [] and the result was a MigrationResultError.
    data = {"schema_version": "0.3.2", "star_lists": [_star_list_0_3(exposure=30.5)]}
    iterable = {
        "schema_version": "0.3.2",
        "star_lists": [_star_list_0_3(exposure=_IterableFloat(30.5))],
    }
    assert validate_as_written(iterable).star_lists[0].exposure == 30.5

    with pytest.warns(SchemaMigrationWarning):
        expected = StarListSet.model_validate(data)
    with pytest.warns(SchemaMigrationWarning):
        result = StarListSet.model_validate(iterable)

    assert result.star_lists[0].exposure == 30.5
    assert result == expected


def test_frozen_instance_upgrades_with_the_fields_a_dict_would_set():
    # A frozen 0.3 StarList built without bias, absolute_focus and stack
    # upgrades with the same fields set as the same data given as a dict:
    # the defaults it filled in are not read as values the file holds.
    # Before the fix the instance's upgraded star list had all three set.
    star_list = _star_list_0_3()
    for name in ("bias", "absolute_focus", "stack"):
        del star_list[name]
    instance = _aavso_starlist_v0_3.StarList.model_validate(star_list)

    with pytest.warns(SchemaMigrationWarning):
        from_dict = StarListSet.model_validate(
            {"schema_version": "0.3.2", "star_lists": [star_list]}
        )
    with pytest.warns(SchemaMigrationWarning):
        from_instance = StarListSet.model_validate(
            {"schema_version": "0.3.2", "star_lists": [instance]}
        )

    fields_set = from_instance.star_lists[0].model_fields_set
    assert fields_set == from_dict.star_lists[0].model_fields_set
    assert not fields_set & {"bias", "absolute_focus", "stack"}


# Containers a pydantic list field reads as a list but JSON text never gives.
LIST_LIKE_CONTAINERS = [
    pytest.param(deque, id="deque"),
    pytest.param(lambda items: np.array(items, dtype=object), id="ndarray"),
]


def _star_list_0_3(**overrides):
    """
    Build one star list of the 0.3 generation as code might.

    Parameters
    ----------
    **overrides
        Keys to set on the star list.

    Returns
    -------
    dict
        The current example `StarList` dumped to Python types, which the
        frozen 0.3 model also accepts, with ``overrides`` applied.
    """
    return {**StarList.from_examples().model_dump(), **overrides}


@pytest.mark.parametrize("container", LIST_LIKE_CONTAINERS)
def test_0_3_star_lists_in_a_list_like_container_are_migrated(container):
    # Star lists held in a deque or an object ndarray are read as a list
    # before migrating, so the 0.3 migration turns a stack of 0 into None.
    # Before the fix the migration skipped the container and the result,
    # still holding stack 0, was a MigrationResultError.
    data = {"schema_version": "0.3.2", "star_lists": container([_star_list_0_3(stack=0)])}
    validate_as_written(data)

    with pytest.warns(SchemaMigrationWarning):
        result = StarListSet.model_validate(data)

    assert result.star_lists[0].stack is None


@pytest.mark.parametrize("container", LIST_LIKE_CONTAINERS)
def test_0_3_star_items_in_a_list_like_container_are_migrated(container):
    # Star items held in a deque or an object ndarray are read as a list
    # before migrating, so the 0.3 migration drops a key the 0.3 model
    # ignored. Before the fix it skipped the container and the current
    # model refused the key as extra_forbidden.
    item = {**StarItem.from_examples().model_dump(), "old_key": 1}
    data = {
        "schema_version": "0.3.2",
        "star_lists": [_star_list_0_3(staritems=container([item]))],
    }
    validate_as_written(data)

    with pytest.warns(SchemaMigrationWarning):
        result = StarListSet.model_validate(data)

    star_items = result.star_lists[0].staritems
    assert len(star_items) == 1
    assert "old_key" not in star_items[0].model_dump()


def test_0_3_generator_of_star_items_inside_a_deque_is_upgraded():
    # A one-shot generator nested inside a list-like container is read once,
    # at whatever depth, so every star item reaches the upgraded file.
    item = StarItem.from_examples().model_dump()
    data = {
        "schema_version": "0.3.2",
        "star_lists": deque([_star_list_0_3(staritems=(i for i in [item, item, item]))]),
    }

    with pytest.warns(SchemaMigrationWarning):
        result = StarListSet.model_validate(data)

    assert len(result.star_lists[0].staritems) == 3


@pytest.mark.parametrize("generation", ["legacy", "0.2"])
def test_older_generation_refusal_does_not_claim_the_file_is_0_3(generation):
    # A legacy or 0.2 file goes through the 0.3 -> 0.4 step too, so its
    # refusal must not say the file is from 0.3: it names the field and says
    # this file cannot be upgraded, whatever generation it came from.
    data, _ = _before_after_pair(generation)
    data["star_lists"][0]["site_elev"] = float("inf")

    with pytest.raises(ValidationError) as excinfo:
        StarListSet.model_validate(data)

    message = excinfo.value.errors()[0]["msg"]
    assert "star_lists[0].site_elev is inf" in message
    assert "cannot be upgraded to it" in message
    assert "from 0.3" not in message


@pytest.mark.parametrize(
    "model, field_name",
    [
        (_aavso_starlist_v0_3.StarItem, "fwhm"),
        (_aavso_starlist_v0_3.StarList, "bias"),
    ],
    ids=["StarItem.fwhm", "StarList.bias"],
)
def test_frozen_0_3_fields_reject_infinity(model, field_name):
    # The frozen 0.3 fwhm and bias refuse infinity on their own, as 0.3.2
    # did; that per-field setting does not show in the JSON schema, so the
    # archive test cannot notice if it is dropped from the frozen module.
    star_list = _written_at_0_3()["star_lists"][0]
    record = star_list if model is _aavso_starlist_v0_3.StarList else star_list["staritems"][0]
    data = {**record, field_name: float("inf")}

    with pytest.raises(ValidationError, match="finite number") as excinfo:
        model.model_validate(data)

    assert excinfo.value.errors()[0]["loc"] == (field_name,)
