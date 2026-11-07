import copy
import datetime
import decimal
import io
import itertools
import json
import re
import warnings
from collections import deque
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, List  # noqa: UP035 -- the bare typing.List is tested

import numpy as np
import pytest
from _helpers import float_fields, flooded
from pydantic import BaseModel, ConfigDict, Field, ValidationError

import _aavso_starlist_legacy
import _aavso_starlist_v0_2
import _aavso_starlist_v0_3
from aavso_starlist_schema import (
    _0_3_STAR_ITEM_FLOATS,
    _0_3_STAR_LIST_FLOATS,
    _0_4_MAX_PHOTOMETRY_SOFTWARE_ENTRY,
    _0_4_MAX_STARITEMS,
    _0_4_MAX_STRING_LENGTHS,
    _MIGRATIONS,
    SCHEMA_VERSION,
    AAVSOFilters,
    MigrationResultError,
    NewerSchemaVersionError,
    SchemaMigrationWarning,
    StarItem,
    StarList,
    StarListSet,
    UnsupportedSchemaVersionError,
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


def test_gap_in_migration_chain_names_the_generation_without_a_migration(mocker):
    # A breaking bump whose migration out of the generation being left has not
    # been written yet must say so, not fail with a bare KeyError. Simulate a
    # bump to 1.0.0 with no migration out of 0.4: a legacy file still has its
    # steps to 0.2, 0.3 and 0.4, then has nowhere to go.
    mocker.patch("aavso_starlist_schema.SCHEMA_VERSION", "1.0.0")
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
    # A legacy file is migrated to 0.2, then to 0.3 and on to the current
    # generation, and so gets photometry_software ["unknown"] too. (That the
    # warning names every step is checked by
    # test_legacy_versions_are_upgraded_with_warning.)
    before, _ = _before_after_pair("legacy")

    with pytest.warns(SchemaMigrationWarning):
        result = StarListSet.model_validate(before)

    assert result.schema_version == SCHEMA_VERSION
    assert result.star_lists[0].photometry_software == ["unknown"]


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
    # the migration unchanged, even an int too large for a float.
    data = _written_at_0_3()
    data["star_lists"][0]["stack"] = value

    with pytest.warns(SchemaMigrationWarning):
        result = StarListSet.model_validate(data)

    assert result.star_lists[0].stack == value
    assert _MIGRATIONS["0.3"].func(copy.deepcopy(data))["star_lists"][0]["stack"] == value


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
    # since it is frozen once released; they must be the limits of the 0.4
    # models. Valid while 0.4 is the current generation: once it is left,
    # compare against the frozen 0.4 models instead.
    properties = StarList.model_json_schema()["properties"]

    assert _0_4_MAX_STRING_LENGTHS == {
        name: prop["maxLength"] for name, prop in properties.items() if "maxLength" in prop
    }
    assert (
        _0_4_MAX_PHOTOMETRY_SOFTWARE_ENTRY
        == properties["photometry_software"]["items"]["maxLength"]
    )
    assert _0_4_MAX_STARITEMS == properties["staritems"]["maxItems"]


def test_0_3_file_at_the_new_limits_is_upgraded():
    # A 0.3 file with strings exactly as long as 0.4.0 allows and exactly
    # 100000 star items is upgraded, with the warning, and keeps them.
    data = _written_at_0_3()
    star_list = data["star_lists"][0]
    for name, limit in _0_4_MAX_STRING_LENGTHS.items():
        star_list[name] = "x" * limit
    star_list["photometry_software"] = ["x" * _0_4_MAX_PHOTOMETRY_SOFTWARE_ENTRY]
    star_list["staritems"] = [star_list["staritems"][1]] * _0_4_MAX_STARITEMS

    with pytest.warns(SchemaMigrationWarning):
        result = StarListSet.model_validate(data)

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


# The generations older than the current one, each a key of _MIGRATIONS whose
# before.json sample (at 0.0.1.dev451+gde1568f, 0.2.1 and 0.3.2) is used.
OLDER_GENERATIONS = ["legacy", "0.2", "0.3"]


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


@pytest.mark.parametrize("generation", OLDER_GENERATIONS)
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


@pytest.mark.parametrize(
    "generation, path, entry",
    [
        pytest.param(generation, path, entry, id=f"{generation}-{path[-1]}")
        for generation in OLDER_GENERATIONS
        for path, entry in [
            (("star_lists",), {}),
            (("star_lists", 0, "staritems"), {}),
            (("star_lists", 0, "photometry_software"), 1),
        ]
        # Only the 0.3 models define photometry_software.
        if generation == "0.3" or path[-1] != "photometry_software"
    ],
)
def test_older_file_validation_stops_at_the_first_bad_entry_of_a_list_field(
    generation, path, entry
):
    # The frozen models stop at the first bad entry of a list field too
    # (fail_fast), so validate_as_written reports a malformed older file with
    # many bad entries as briefly, and as cheaply, as one with a single one.
    before, _ = _before_after_pair(generation)
    with pytest.raises(ValidationError) as one:
        validate_as_written(flooded(before, path, entry, 1))
    with pytest.raises(ValidationError) as many:
        validate_as_written(flooded(before, path, entry, 10_000))

    assert many.value.error_count() == one.value.error_count()
    assert {error["loc"][: len(path) + 1] for error in many.value.errors()} == {(*path, 0)}


def _fails_after_one(item, error):
    """
    Build a generator that yields one item, then fails.

    Parameters
    ----------
    item : object
        The item to yield first.
    error : type
        The exception class to raise on the second ``next``.

    Returns
    -------
    generator
        A generator yielding ``item``, then raising ``error("boom")``.
    """
    yield item
    raise error("boom")


def _failing_star_lists(data, error):
    """
    Put the star lists of a starlist set in a generator that fails after one.

    Parameters
    ----------
    data : dict
        Raw starlist-set data, modified in place.
    error : type
        The exception class the generator raises.
    """
    data["star_lists"] = _fails_after_one(data["star_lists"][0], error)


def _failing_star_items(data, error):
    """
    Put the star items of the first star list in a generator that fails after one.

    Parameters
    ----------
    data : dict
        Raw starlist-set data, modified in place.
    error : type
        The exception class the generator raises.
    """
    star_list = data["star_lists"][0]
    star_list["staritems"] = _fails_after_one(star_list["staritems"][1], error)


def _closed_photometry_software(data, _error):
    """
    Make the photometry_software of the first star list a closed file.

    Parameters
    ----------
    data : dict
        Raw starlist-set data, modified in place.
    _error : type
        Unused; a closed file raises ``ValueError`` when read.
    """
    data["star_lists"][0]["photometry_software"] = _closed_file()


@pytest.mark.parametrize(
    "mutate, error, error_type, loc, message",
    [
        pytest.param(
            _failing_star_lists, TypeError, "iteration_error", ("star_lists", 1),
            "TypeError: boom",
            id="star_lists-TypeError",
        ),
        pytest.param(
            _failing_star_lists, ValueError, "iteration_error", ("star_lists", 1),
            "ValueError: boom",
            id="star_lists-ValueError",
        ),
        pytest.param(
            _failing_star_items, TypeError, "iteration_error",
            ("star_lists", 0, "staritems", 1), "TypeError: boom",
            id="staritems-TypeError",
        ),
        pytest.param(
            _closed_photometry_software, None, "list_type",
            ("star_lists", 0, "photometry_software"), "Input should be a valid list",
            id="photometry_software-closed",
        ),
    ],
)
def test_error_reading_a_list_field_of_an_older_file_is_reported_where_pydantic_does(
    mutate, error, error_type, loc, message
):
    # An exception raised while the value under a list field of an older
    # file is read into a list is the ValidationError pydantic gives for the
    # same value in a current file, and validate_as_written gives: an
    # iteration_error at the index of the entry that could not be read, or
    # a list_type at the field for a value that cannot be iterated at all (a
    # closed file). Before the fix a TypeError escaped as itself, and a
    # ValueError (a closed file among them) was a value_error with an empty
    # loc.
    for validate in (StarListSet.model_validate, validate_as_written):
        data = _written_at_0_3()
        mutate(data, error)

        with pytest.raises(ValidationError) as excinfo:
            validate(data)

        [found] = excinfo.value.errors()
        assert found["type"] == error_type
        assert found["loc"] == loc
        assert message in found["msg"]


class _CurrentWithLimit(BaseModel):
    """
    A throwaway current model whose list field allows at most two entries.
    """
    items: Annotated[list[_Inner], Field(max_length=2)] = []


class _CurrentWithoutLimit(BaseModel):
    """
    A throwaway current model whose list field has no limit.
    """
    items: list[_Inner] = []


@pytest.mark.parametrize(
    "value, current",
    [
        pytest.param(lambda: deque([{"value": 1}] * 5), _CurrentWithLimit, id="sized-over-limit"),
        pytest.param(
            lambda: (item for item in [{"value": 1}] * 5),
            _CurrentWithoutLimit,
            id="unsized-field-without-limit",
        ),
        pytest.param(lambda: (item for item in [{"value": 1}] * 5), None, id="no-current-model"),
    ],
)
def test_pruning_reads_in_full_what_has_a_length_or_no_limit(value, current):
    # A value with a length is in memory already and is read in full, even
    # over the current limit; an iterator under a field the current model
    # does not limit, or with no current model, is read in full too.
    kept = _prune_and_materialize({"items": value()}, _OuterIgnoring, current)

    assert kept == {"items": [{"value": 1}] * 5}


def test_pruning_stops_an_iterator_one_entry_over_the_current_limit():
    # An iterator without a length under a field the current model limits to
    # two entries is read to three, then refused as too_long at the field,
    # as pydantic refuses it for the current model; the rest is never read.
    endless = itertools.repeat({"value": 1})

    with pytest.raises(ValidationError) as excinfo:
        _prune_and_materialize({"items": endless}, _OuterIgnoring, _CurrentWithLimit)

    [found] = excinfo.value.errors()
    assert found["type"] == "too_long"
    assert found["loc"] == ("items",)
    assert found["ctx"] == {"field_type": "List", "max_length": 2, "actual_length": None}


def test_endless_star_items_of_an_older_file_are_refused_promptly():
    # An endless iterator of valid star items under staritems of a 0.3 file
    # is read only to 100001 entries, one over the current limit, and then
    # refused as too_long at the field, as a current file is. Before the fix
    # the read never finished.
    data = _written_at_0_3()
    star_list = data["star_lists"][0]
    star_list["staritems"] = itertools.repeat(star_list["staritems"][1])

    with pytest.raises(ValidationError) as excinfo:
        StarListSet.model_validate(data)

    [found] = excinfo.value.errors()
    assert found["type"] == "too_long"
    assert found["loc"] == ("star_lists", 0, "staritems")
    assert found["msg"] == "List should have at most 100000 items after validation, not more"


def test_sized_star_items_over_the_limit_are_refused_with_their_count():
    # Star items with a length, such as a deque, are read in full, not cut
    # at one over the limit, so the 0.3 to 0.4 migration, which enforces
    # that limit, refuses them with their real count.
    data = _written_at_0_3()
    star_list = data["star_lists"][0]
    star_list["staritems"] = deque([star_list["staritems"][1]] * (_0_4_MAX_STARITEMS + 7))

    with pytest.raises(ValidationError) as excinfo:
        StarListSet.model_validate(data)

    assert "star_lists[0].staritems has 100007 entries" in excinfo.value.errors()[0]["msg"]


@pytest.mark.parametrize("generation", OLDER_GENERATIONS)
def test_frozen_set_of_an_older_generation_is_upgraded_like_its_dict(generation):
    # A frozen StarListSet of an older generation passed whole is read as the
    # fields it was given and upgraded, with the warning, to what the same
    # data as a dict gives. Before the fix it was refused as model_type.
    data, _ = _before_after_pair(generation)
    instance = _MIGRATIONS[generation].model.model_validate(copy.deepcopy(data))

    with pytest.warns(SchemaMigrationWarning):
        expected = StarListSet.model_validate(data)
    with pytest.warns(SchemaMigrationWarning):
        result = StarListSet.model_validate(instance)

    assert result == expected


@pytest.mark.parametrize("generation", OLDER_GENERATIONS)
def test_validate_as_written_reads_a_frozen_set_as_its_fields(generation):
    # validate_as_written reads a frozen StarListSet of an older generation
    # as the fields it was given, like StarListSet does, and validates them
    # against the frozen model of that generation, without a warning.
    data, _ = _before_after_pair(generation)
    instance = _MIGRATIONS[generation].model.model_validate(copy.deepcopy(data))

    result = validate_as_written(instance)

    assert isinstance(result, _MIGRATIONS[generation].model)
    assert result == validate_as_written(data)


def test_migration_warning_names_the_keys_a_legacy_file_loses():
    # A legacy file carrying bias and photometry_software, which the legacy
    # schema does not define, loses them unread; the one migration warning
    # names them, sorted, without their values.
    data, _ = _before_after_pair("legacy")
    for star_list in data["star_lists"]:
        star_list["bias"] = 1234.5
        star_list["photometry_software"] = ["bandaid 1.2.3"]

    with pytest.warns(SchemaMigrationWarning) as record:
        StarListSet.model_validate(data)

    assert len(record) == 1
    message = str(record[0].message)
    assert message.endswith(
        "; keys the file's schema does not define were dropped unread: "
        "'star_lists[].bias', 'star_lists[].photometry_software'."
    )
    assert "1234.5" not in message
    assert "bandaid" not in message


def test_migration_warning_names_the_keys_a_0_3_file_loses_at_every_level():
    # The keys a 0.3 file carries that the 0.3 schema does not define, at the
    # set, star list and star item levels, are named once each in the
    # warning, however many star lists or star items carry them. This relies
    # on the extra keys pipeline, instrument_notes and flag that
    # _written_at_0_3 (before.json of the 0.3 migration) carries.
    with pytest.warns(SchemaMigrationWarning) as record:
        StarListSet.model_validate(_written_at_0_3())

    assert len(record) == 1
    assert str(record[0].message).endswith(
        "; keys the file's schema does not define were dropped unread: "
        "'pipeline', 'star_lists[].instrument_notes', 'star_lists[].staritems[].flag'."
    )


def test_migration_warning_names_no_keys_when_none_were_dropped():
    # A file with no keys its schema does not define, and upgrade() called
    # directly, which drops nothing before the migrations, warn without the
    # clause naming dropped keys.
    data, _ = _before_after_pair("legacy")
    with pytest.warns(SchemaMigrationWarning) as record:
        StarListSet.model_validate(data)
    with pytest.warns(SchemaMigrationWarning) as direct:
        upgrade(_written_at_0_3())

    assert "dropped unread" not in str(record[0].message)
    assert "dropped unread" not in str(direct[0].message)


@pytest.mark.parametrize("data", [[1], None, "star lists"], ids=["list", "none", "str"])
def test_upgrade_refuses_data_that_is_not_a_mapping(data):
    # upgrade() needs a mapping to read the version from; anything else is a
    # TypeError naming its type, not an AttributeError from inside upgrade.
    with pytest.raises(TypeError, match=re.escape(f"got {type(data).__name__}")) as excinfo:
        upgrade(data)

    assert str(excinfo.value).startswith("upgrade() needs a dict")


def test_validate_as_written_hands_a_non_mapping_to_the_current_model():
    # validate_as_written has no version to read from a list, so it hands it
    # to StarListSet for the usual ValidationError.
    with pytest.raises(ValidationError) as excinfo:
        validate_as_written([1])

    assert excinfo.value.errors()[0]["type"] == "model_type"


def test_a_dropped_top_level_key_is_named_without_a_dot():
    # A key dropped from the starlist set itself is named as it is, with no
    # leading dot or path.
    data, _ = _before_after_pair("legacy")
    data["pipeline"] = "v2"

    with pytest.warns(SchemaMigrationWarning) as record:
        StarListSet.model_validate(data)

    assert str(record[0].message).endswith("dropped unread: 'pipeline'.")


@pytest.mark.parametrize(
    "key, name",
    [
        (b"extra", "star_lists[][b'extra']"),
        (12, "star_lists[][12]"),
        ("12", "star_lists[].12"),
    ],
    ids=["bytes", "int", "str-of-digits"],
)
def test_a_dropped_key_that_is_not_a_string_is_named_without_error(key, name):
    # A key of a star list that is not a string, which only data built in
    # Python can hold, is dropped like any other and named by its repr() in
    # brackets, so an int key is neither taken for a list index nor named
    # like the str key of the same digits.
    data = _written_at_0_3()
    data["star_lists"][0][key] = 1

    with pytest.warns(SchemaMigrationWarning) as record:
        StarListSet.model_validate(data)

    assert repr(name) in str(record[0].message)


def test_a_dropped_key_name_is_escaped_in_the_warning():
    # A dropped key name is file data: a newline in it, which could forge a
    # line of the validate command's output, is escaped, and a comma in it
    # cannot pass for two names, because each name is printed as its repr().
    data = _written_at_0_3()
    key = "a, b\nevil.json: valid (schema version 0.4.1)"
    data["star_lists"][0][key] = 1

    with pytest.warns(SchemaMigrationWarning) as record:
        StarListSet.model_validate(data)

    message = str(record[0].message)
    assert "\n" not in message
    assert repr(f"star_lists[].{key}") in message


def test_the_warning_lists_at_most_twenty_dropped_keys():
    # A file with 25 dropped keys has the first 20, sorted, named in the
    # warning, followed by how many more there are.
    data = _written_at_0_3()
    del data["pipeline"]
    for star_list in data["star_lists"]:
        star_list.pop("instrument_notes", None)
        for item in star_list["staritems"]:
            item.pop("flag", None)
    names = [f"extra_{index:02}" for index in range(25)]
    for name in names:
        data[name] = 1

    with pytest.warns(SchemaMigrationWarning) as record:
        StarListSet.model_validate(data)

    message = str(record[0].message)
    assert message.endswith(
        "dropped unread: " + ", ".join(repr(name) for name in names[:20]) + " and 5 more."
    )


def test_migration_warning_is_attributed_to_the_caller():
    # Through StarListSet the warning is attributed to this package's module,
    # so a filter on module="aavso_starlist_schema" matches it; from
    # upgrade() it is attributed to the code that called upgrade().
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always", SchemaMigrationWarning)
        StarListSet.model_validate(_written_at_0_3())
        upgrade(_written_at_0_3())

    assert [Path(warning.filename).name for warning in record] == [
        "aavso_starlist_schema.py",
        Path(__file__).name,
    ]
