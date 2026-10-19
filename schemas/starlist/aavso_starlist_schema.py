"""JSON-schema definition and generators for AAVSO smart-telescope starlists.

This single-file module defines the pydantic models that describe an AAVSO Smart
Telescope starlist (``StarItem``, ``StarList``, ``StarListSet``), helpers to emit
the schema as JSON or as a markdown table, and a Fire-based command-line entry
point.
"""

import copy
import json
import re
import warnings
from collections import defaultdict
from collections.abc import Callable, Mapping
from enum import StrEnum
from pathlib import Path
from typing import Annotated, NamedTuple

from astropy.table import Table
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from pydantic.alias_generators import to_snake
from pydantic_core import PydanticSerializationError, to_jsonable_python

import _aavso_starlist_legacy  # frozen models of the legacy generation

try:
    from _aavso_version import __version__  # top-level sibling, written by hatch-vcs
except ImportError:  # pragma: no cover - source tree without the generated file
    __version__ = "0.0.0"

# Version of the *schema* (the contract manufacturers write files against). This is
# deliberately independent of the package version above: bump it whenever the
# generated schema under aavso_starlist_schema_data/ changes. The archive test
# fails if the generated schema changes without bumping this constant.
SCHEMA_VERSION = "0.2.0"

# Every released schema version from 0.2.0 on, oldest first; add each new
# SCHEMA_VERSION here. Listed in the code, rather than read from the archive
# folders, so the versions this reader accepts never depend on the filesystem.
# The archive test fails if this differs from the archived versions.
_RELEASED_VERSIONS = ("0.2.0",)

__all__ = [
    "AAVSOFilters",
    "StarItem",
    "StarList",
    "StarListSet",
    "generate_starlist_schema",
    "generate_star_list_set_schema",
    "main",
    "cli",
    "DATA_DIR",
    "SCHEMA_VERSION",
    "upgrade",
    "validate_as_written",
    "SchemaVersionError",
    "NewerSchemaVersionError",
    "UnsupportedSchemaVersionError",
    "MigrationResultError",
    "SchemaMigrationWarning",
]

# Reference schema files (a shipped deliverable) live alongside this module, both
# in the source tree and in an installed wheel. The folder is named for the
# package because a wheel installs it at the top level of site-packages.
DATA_DIR = Path(__file__).parent / "aavso_starlist_schema_data"


class AAVSOFilters(StrEnum):
    """
    Photometric filters a smart telescope may report.

    Values are AAVSO filter names. The list is restricted to the filters
    such telescopes actually use so that typos and unknown codes are
    rejected by the schema.
    """
    TG = "TG"
    TR = "TR"
    TB = "TB"
    L3 = "L3"
    L4 = "L4"
    B = "B"
    V = "V"
    R = "R"
    I = "I"  # noqa: E741 - AAVSO filter name (Cousins I), not an ambiguous identifier
    SG = "SG"
    SR = "SR"
    SI = "SI"
    CR = "CR"
    CV = "CV"


class PrettyPrintMixin:
    """
    Mixin to render a pydantic model's fields as a markdown table.

    Expects every field to define ``title``, ``description``, ``examples``
    and a ``unit`` entry in ``json_schema_extra``.
    """
    @classmethod
    def markdown_table(cls):
        """
        Generate a markdown table of the model fields for the AAVSO starlist schema.
        """
        rows = ["| Title | JSON Field | Type | Unit | Description | Examples |"]
        rows.append("| --- | --- | --- | --- | --- | --- |")
        for name, field_info in cls.model_fields.items():
            type_name = getattr(field_info.annotation, "__name__", field_info.annotation)
            row = (
                f"| {field_info.title} | {name} | {type_name} "
                f"| {field_info.json_schema_extra['unit']} "
                f"| {field_info.description} | {field_info.examples[0]} |"
            )
            rows.append(row)

        return "\n".join(rows)


class GenerateInstanceFromExamplesMixin:
    """
    Mixin to generate an instance of the class from the examples in the schema.
    """
    @classmethod
    def from_examples(cls):
        """
        Generate an instance of the class from the examples in the schema.
        """
        example_data = {name: field_info.examples[0] for name, field_info in cls.model_fields.items()}
        return cls(**example_data)


class StarItem(BaseModel, PrettyPrintMixin, GenerateInstanceFromExamplesMixin):
    """
    Definition of individual entries in an AAVSO star list.
    """
    x: Annotated[
        float,
        Field(
            ge=0,
            title="X-coordinate",
            description="X-coordinate of the star center (in pixel coordinates)",
            json_schema_extra=dict(unit="pixel"),
            examples=[1206.78]
        )
    ]
    y: Annotated[
        float,
        Field(
            ge=0,
            title="Y-coordinate",
            description="Y-coordinate of the star center (in pixel coordinates)",
            json_schema_extra=dict(unit="pixel"),
            examples=[620.10]
        )
    ]
    ra: Annotated[
        float,
        Field(
            ge=0,
            lt=360,
            title="Right Ascension",
            description=(
                "Right Ascension of the star (in decimal degrees) at "
                "the epoch specified in the metadata"
            ),
            json_schema_extra=dict(unit="degree"),
            examples=[212.56789]
        )
    ]
    dec: Annotated[
        float,
        Field(
            ge=-90,
            le=90,
            title="Declination",
            description=(
                "Declination of the star (in decimal degrees) at "
                "the epoch specified in the metadata"
            ),
            json_schema_extra=dict(unit="degree"),
            examples=[-12.12345]
        )
    ]
    tot_count: Annotated[
        float,
        Field(
            ge=0,
            title="Star Count",
            description="Total integrated counts of the star, background-subtracted",
            json_schema_extra=dict(unit="adu"),
            examples=[156700.4]
        )
    ]
    count_err: Annotated[
        float,
        Field(
            ge=0,
            title="Count Error",
            description="Error in the total integrated counts of the star",
            json_schema_extra=dict(unit="adu"),
            examples=[15300.1]
        )
    ]
    bkgd_count: Annotated[
        float,
        Field(
            title="Background counts",
            description="Background count level in the vicinity of the star",
            json_schema_extra=dict(unit="adu / pixel"),
            examples=[1209.45]
        )
    ]
    peak_count: Annotated[
        float,
        Field(
            ge=0,
            title="Peak Counts",
            description="Peak counts of the star",
            json_schema_extra=dict(unit="adu"),
            examples=[31454.963]
        )
    ]


class StarList(BaseModel, PrettyPrintMixin, GenerateInstanceFromExamplesMixin):
    """
    Definition of the header section of an AAVSO star list schema.
    """
    obs_time: Annotated[
        str,
        Field(
            title="Observation Start Time",
            json_schema_extra=dict(
                unit=None,
                format="date-time",
                scale="UTC",
            ),
            description="UTC time at start of observation",
            examples=["2021-06-15T03:45:00"]
        )
    ]
    site_lat: Annotated[
        float,
        Field(
            ge=-90,
            le=90,
            title="Site Latitude",
            description="Latitude of the observing site",
            json_schema_extra=dict(unit="degree"),
            examples=[-41.56896]
        )
    ]
    site_lon: Annotated[
        float,
        Field(
            ge=-180,
            le=180,
            title="Site Longitude",
            description="Longitude of the observing site",
            json_schema_extra=dict(unit="degree"),
            examples=[-71.23841]
        )
    ]
    site_elev: Annotated[
        float,
        Field(
            title="Site Elevation",
            description="Observer's elevation above mean sea level",
            json_schema_extra=dict(unit="meter"),
            examples=[211.0]
        )
    ]
    observer: Annotated[
        str,
        Field(
            title="Observer Code",
            description="AAVSO code of the observer",
            json_schema_extra=dict(unit="none"),
            examples=["MMU"]
        )
    ]
    filter: Annotated[
        AAVSOFilters,
        Field(
            title="Filter",
            description="Filter used for the observation, from https://www.aavso.org/filters",
            json_schema_extra=dict(unit="none"),
            examples=[AAVSOFilters.TG]
        )
    ]
    block_filter: Annotated[
        str,
        Field(
            title="Blocking Filter",
            description="Name of blocking filter used on telescope",
            json_schema_extra=dict(unit="none"),
            examples=["UV+IR"]
        )
    ]
    exposure: Annotated[
        float,
        Field(
            ge=0,
            title="Exposure Time",
            description="Effective duration of exposure",
            json_schema_extra=dict(unit="second"),
            examples=[30.0]
        )
    ]
    tel_manufac: Annotated[
        str,
        Field(
            title="Telescope Manufacturer",
            description="Name of the telescope manufacturer",
            json_schema_extra=dict(unit="none"),
            examples=["Celestron"]
        )
    ]
    width : Annotated[
        int,
        Field(
            gt=0,
            title="Image Width",
            description="Width of the image in pixels",
            json_schema_extra=dict(unit="pixel"),
            examples=[2048]
        )
    ]
    height: Annotated[
        int,
        Field(
            gt=0,
            title="Image Height",
            description="Height of the image in pixels",
            json_schema_extra=dict(unit="pixel"),
            examples=[1024]
        )
    ]
    stack: Annotated[
        int | None,
        Field(
            title="Number of images",
            description=(
                "Number of images stacked to create this image. "
                "If not applicable, set to None."
            ),
            json_schema_extra=dict(unit="none"),
            examples=[3]
        )
    ] = None
    tel_model: Annotated[
        str,
        Field(
            title="Telescope Model",
            description="Model of the telescope",
            json_schema_extra=dict(unit="none"),
            examples=["Origin 1"]
        )
    ]
    tel_firmware: Annotated[
        str,
        Field(
            title="Telescope Firmware",
            description="Firmware version of the telescope",
            json_schema_extra=dict(unit="none"),
            examples=["20240817.01"]
        )
    ]
    adc_depth: Annotated[
        int,
        Field(
            ge=0,
            title="A/D Converter Bit Depth",
            description="Bit depth of the analog-to-digital converter",
            json_schema_extra=dict(unit="bit"),
            examples=[14]
        )
    ]
    largest_usable_adu_value: Annotated[
        int,
        Field(
            ge=0,
            title="Largest Usable ADU Value",
            description="Largest usable analog-to-digital unit value",
            json_schema_extra=dict(unit="adu"),
            examples=[41000]
        )
    ]
    egain: Annotated[
        float,
        Field(
            ge=0,
            title="System gain",
            description="Gain of the camera in e-/adu",
            json_schema_extra=dict(unit="e-/adu"),
            examples=[1.2]
        )
    ]
    fwhm: Annotated[
        float,
        Field(
            gt=0,
            title="FWHM",
            description="Typical full width at half maximum of the star image",
            json_schema_extra=dict(unit="pixel"),
            examples=[3.5]
        )
    ]
    refframe: Annotated[
        str,
        Field(
            title="Coordinate Reference Frame",
            description="Reference frame of the observation",
            json_schema_extra=dict(unit="none"),
            examples=["ICRS"]
        )
    ]
    staritems: Annotated[
        list[StarItem],
        Field(
            title="Star Items",
            description="List of stars detected in the image",
            json_schema_extra=dict(unit="none"),
            examples=[[]]
        )
    ]

    @classmethod
    def from_table(cls, table, metadata=None):
        """
        Create a StarList object from a `astropy.table.Table` object.

        Parameters
        ----------
        table : `astropy.table.Table`
            The input table containing one row per star item.

        metadata : dict, optional
            Additional metadata to ustoin the StarList object.
            If not provided, the metadata from the table will be used.
            If both are provided, the metadata from this argument will
            take precedence.

        Returns
        -------
        StarList
            An instance of the StarList class.
        """
        # First create the star items from the table rows.
        star_items = []
        missing_keys = set(StarItem.model_fields.keys()) - set(table.colnames)
        if missing_keys:
            raise ValueError(f"Missing columns in table: {', '.join(missing_keys)}")
        for row in table:
            star_items.append(
                StarItem(**{key: row[key] for key in StarItem.model_fields.keys()})
            )

        # Construct metadata, with any passed in to the metadata argument
        # taking precedence over metadata in the table.
        final_meta = table.meta.copy()
        if metadata:
            final_meta.update(metadata)

        final_meta["staritems"] = star_items

        if missing_keys := set(cls.model_fields.keys()) - set(final_meta.keys()):
            raise ValueError(f"Missing keys in metadata: {', '.join(missing_keys)}")

        # Create the StarList object
        return cls.model_validate(final_meta)

    def to_table(self):
        """
        Create an astropy table from a starlist.

        Returns
        -------

        `astropy.table.Table`
            A table in which the columns are the  individual star items
            properties, with one star item per row. The remaining information
            from the starlist is stored in the table metata.

        """
        table_columns = self.staritems[0].model_fields
        table_dict = defaultdict(list)
        for star in self.staritems:
            for col in table_columns:
                table_dict[col].append(getattr(star, col))

        table_meta = self.model_dump()
        table_meta.pop("staritems")

        return Table(table_dict, meta=table_meta)


class SchemaVersionError(Exception):
    """
    Base class for problems with the schema version of a starlist file.

    Notes
    -----
    Deliberately not a `ValueError`, so that callers who catch validation
    errors do not silently swallow a version problem.
    """


class NewerSchemaVersionError(SchemaVersionError):
    """The file was written against a newer schema than this reader supports."""


class UnsupportedSchemaVersionError(SchemaVersionError):
    """The schema version is null, unparseable or never released, or has no migration path."""


class MigrationResultError(SchemaVersionError):
    """
    A file valid for its own schema version was not valid after being upgraded.

    Notes
    -----
    This points at a bug in a migration, not at the file: the file passed
    validation against the schema it was written for. The underlying
    ``pydantic.ValidationError`` is available as ``__cause__``.
    """


class SchemaMigrationWarning(UserWarning):
    """A starlist file was written against an older schema and was upgraded."""


# No leading zeros, so each version has exactly one spelling.
_STRICT_SEMVER = re.compile(r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)")

# Pre-versioning files carry the hatch-vcs package version, e.g.
# 0.1.dev66+g6f57c4d02 or 0.0.1.dev451+gde1568f (possibly with a .dYYYYMMDD suffix).
_LEGACY_PACKAGE_VERSION = re.compile(r"0\.[01](\.\d+)?\.dev\d+(\+.*)?")


def _parse_semver(version):
    """
    Parse a strict ``X.Y.Z`` version string.

    Parameters
    ----------
    version : str
        The value found in a file's ``schema_version`` field.

    Returns
    -------
    tuple of int or None
        ``(major, minor, patch)``, or ``None`` if ``version`` has extra
        components, such as the package version ``0.1.dev66+g6f57c4d02``. The
        caller decides whether ``None`` means legacy or unsupported.
    """
    match = _STRICT_SEMVER.fullmatch(version)
    return tuple(int(part) for part in match.groups()) if match else None


def _generation(version):
    """
    Determine the generation a schema version belongs to.

    A generation is a set of versions that need no migration between them:
    ``"0.y"`` for ``0.y.z``, ``"x"`` for ``x.y.z`` with ``x >= 1``, and
    ``"legacy"`` for everything before 0.2.0, including a missing version.

    Parameters
    ----------
    version : str or None
        The ``schema_version`` value from a file; ``None`` if absent.

    Returns
    -------
    str
        The generation name.

    Raises
    ------
    UnsupportedSchemaVersionError
        If ``version`` is not a string, or is a string that is neither
        ``X.Y.Z`` nor a pre-versioning package version such as
        ``0.1.dev66+g6f57c4d02``.
    """
    if version is None:
        return "legacy"
    if not isinstance(version, str):
        raise UnsupportedSchemaVersionError(
            f"schema_version must be a string, got {type(version).__name__}: {version!r}"
        )
    parsed = _parse_semver(version)
    if parsed is None:
        if _LEGACY_PACKAGE_VERSION.fullmatch(version):
            return "legacy"
        raise UnsupportedSchemaVersionError(
            f"Cannot interpret schema_version {version!r}; expected X.Y.Z"
        )
    if parsed < (0, 2, 0):
        return "legacy"
    if parsed[0] == 0:
        return f"0.{parsed[1]}"
    return str(parsed[0])


class _Migration(NamedTuple):
    """
    One step of the migration chain, taking a file one generation forward.

    A migration function is called only by `upgrade`, on a deep copy of the
    input, so it may modify its argument in place and return it, or return a
    new dict. It need not set ``schema_version``: `upgrade` stamps the target
    version on the result. The first migration sees the version the file
    states (or none), each later one the target of the step before. Nothing
    is validated between migrations.

    The argument is always a ``dict``, but what is inside depends on how
    `upgrade` was reached:

    - From ``StarListSet.model_validate_json``, ``model_validate`` or the
      constructor, plain JSON data: dicts, lists, strings, numbers, booleans
      and ``None``. The input is first validated, as given, against the
      frozen ``model``, then converted by `_to_plain`: tuples become lists,
      other mappings dicts, instances of the frozen generation's own
      ``StarList`` or ``StarItem`` dicts of their fields, enum members their
      values. Keys the frozen model ignores are kept.
    - From `upgrade` called directly, anything at all: nothing has been
      validated or converted.

    Attributes
    ----------
    model : type
        The frozen ``StarListSet`` model of the generation being left, used to
        validate a file against the schema it was written for.
    target : str
        The schema version the migrated data conforms to.
    func : callable
        Takes the raw dict, a copy it may modify, and returns the migrated
        dict. Frozen once released.
    summary : str
        What changed, for the migration warning.
    """
    model: type[BaseModel]
    target: str
    func: Callable[[dict], dict]
    summary: str


def _migrate_legacy_to_0_2(data):
    """
    Migrate a pre-versioning (legacy) starlist set to schema version 0.2.0.

    Parameters
    ----------
    data : dict
        Raw starlist-set data from a legacy file.

    Returns
    -------
    dict
        ``data`` unchanged. The only difference at 0.2.0 is that
        ``schema_version`` became required, and `upgrade` stamps the target
        version. The legacy version string is deliberately not interpreted.
    """
    return data


# Keyed by the generation a file is in; the value takes it to the next generation.
# The next breaking change adds an entry here, keyed by the generation being left,
# with a frozen copy of that generation's models in its own _aavso_starlist_*.py.
_MIGRATIONS = {
    "legacy": _Migration(
        _aavso_starlist_legacy.StarListSet,
        "0.2.0",
        _migrate_legacy_to_0_2,
        "schema_version became required; no other field changed",
    ),
}


def _released_versions():
    """
    List the schema versions that have been released.

    Returns
    -------
    frozenset of tuple of int
        ``(major, minor, patch)`` of every version in ``_RELEASED_VERSIONS``,
        plus ``SCHEMA_VERSION`` itself.
    """
    return frozenset(
        _parse_semver(version) for version in (*_RELEASED_VERSIONS, SCHEMA_VERSION)
    )


def _version_pattern():
    """
    Build a regex matching the versions the current model accepts.

    Used as the ``pattern`` of the ``schema_version`` field, so the generated
    schema admits exactly the versions this reader takes without upgrading:
    the released versions of the current generation, up to `SCHEMA_VERSION`.

    Returns
    -------
    str
        Anchored regex listing each version, e.g. ``^(0\\.2\\.0|0\\.2\\.1)$``.
    """
    current_generation = _generation(SCHEMA_VERSION)
    versions = [
        ".".join(str(part) for part in version)
        for version in sorted(_released_versions())
        if _generation(".".join(str(part) for part in version)) == current_generation
    ]
    return "^(" + "|".join(re.escape(version) for version in versions) + ")$"


def _supported_generation(data):
    """
    Determine the generation of raw starlist-set data, if this reader handles it.

    Parameters
    ----------
    data : dict
        The starlist set as read from JSON, before any validation.

    Returns
    -------
    str
        The current generation, or an older one that has a migration.

    Raises
    ------
    NewerSchemaVersionError
        If the file was written against a newer schema than this reader.
    UnsupportedSchemaVersionError
        If the version is null, cannot be interpreted, is 0.2.0 or later but
        was never released, or has no migration path.
    """
    # Only a missing field means legacy; an explicit null was never valid.
    if "schema_version" in data and data["schema_version"] is None:
        raise UnsupportedSchemaVersionError("schema_version is null")

    original = data.get("schema_version")
    generation = _generation(original)
    supported = _parse_semver(SCHEMA_VERSION)
    parsed = None if generation == "legacy" else _parse_semver(original)
    # Any newer version is refused, even in the current generation: this reader
    # cannot know what changed, and a newer version of its own generation (a
    # patch while 0.y.z, a minor from 1.0.0 on) may add fields it would
    # silently drop.
    if parsed is not None and parsed > supported:
        raise NewerSchemaVersionError(
            f"File has schema_version {original!r}, newer than the "
            f"{SCHEMA_VERSION!r} this reader supports; upgrade the "
            "aavso-starlist-schema package."
        )
    # A version from 0.2.0 up to the current one must be one that was actually
    # released. (Anything earlier is legacy, and parsed is None.)
    if parsed is not None and parsed not in _released_versions():
        raise UnsupportedSchemaVersionError(
            f"File has schema_version {original!r}, which was never released; "
            f"see the archived versions under {DATA_DIR.name}/."
        )
    if generation == _generation(SCHEMA_VERSION) or generation in _MIGRATIONS:
        return generation

    raise UnsupportedSchemaVersionError(
        f"File has schema_version {original!r}, for which there is no "
        f"migration to the {SCHEMA_VERSION!r} this reader supports."
    )


def validate_as_written(data):
    """
    Validate raw starlist-set data against the schema of its own generation.

    Parameters
    ----------
    data : dict
        The starlist set as read from JSON.

    Returns
    -------
    pydantic.BaseModel
        The validated data: a `StarListSet` if the data is in the current
        generation, otherwise an instance of the frozen model of its own
        generation.

    Raises
    ------
    pydantic.ValidationError
        If the data does not match the schema of its own generation.
    NewerSchemaVersionError
        If the file was written against a newer schema than this reader.
    UnsupportedSchemaVersionError
        If the version is null, cannot be interpreted, is 0.2.0 or later but
        was never released, or has no migration path.

    Notes
    -----
    This answers "is this file valid for the schema it was written for?"
    without migrating it to a newer generation. A file from an older
    generation is checked against that generation's frozen model and returned
    as that model. A file in the current generation is checked against the
    current model and, like any `StarListSet`, states `SCHEMA_VERSION`, even
    if the file stated an older version of the generation.

    A legacy file without a ``schema_version`` comes back with the legacy
    model's frozen default in that field, a version the file never stated. To
    pass the result on, use ``model_dump(exclude_unset=True)``, or validate
    the raw data with `StarListSet` directly.
    """
    generation = _supported_generation(data)
    if generation in _MIGRATIONS:
        return _MIGRATIONS[generation].model.model_validate(data)
    return StarListSet.model_validate(data)


def _to_plain(value):
    """
    Convert data built in Python to the plain form JSON text would give.

    Parameters
    ----------
    value : object
        Raw starlist-set data, or any value inside it.

    Returns
    -------
    object
        A new structure of dicts, lists, strings, numbers, booleans and
        ``None``: every mapping becomes a dict, a tuple a list, a model
        instance a dict of its fields, an enum member its value and a
        datetime an ISO 8601 string. A value of a type pydantic cannot
        serialize, possible only under a key the models ignore, is passed on
        as given.
    """
    if isinstance(value, Mapping):
        return {key: _to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(item) for item in value]
    try:
        return to_jsonable_python(value)
    except PydanticSerializationError:
        return value


def upgrade(data):
    """
    Upgrade raw starlist-set data to the current schema version.

    The data is transformed as a plain dict and is not validated, neither
    before nor after; see `validate_as_written` and `StarListSet`. The
    migrations work on a deep copy of ``data``, so they may modify it in
    place; the values inside are passed to them as given, not converted to
    plain dicts and lists, as `StarListSet` does before calling this (see
    `_Migration`).

    Parameters
    ----------
    data : dict
        The starlist set as read from JSON, before any validation.

    Returns
    -------
    dict
        A deep copy of ``data`` stamped with `SCHEMA_VERSION`, upgraded first
        if it is from an older generation. The result shares nothing with the
        input, which is never modified.

    Raises
    ------
    NewerSchemaVersionError
        If the file was written against a newer schema than this reader.
    UnsupportedSchemaVersionError
        If the version is null, cannot be interpreted, is 0.2.0 or later but
        was never released, or has no migration path.

    Warns
    -----
    SchemaMigrationWarning
        Once, if the data had to be migrated.
    """
    original = data.get("schema_version")
    generation = _supported_generation(data)
    current_generation = _generation(SCHEMA_VERSION)

    # Always a copy, so the result never shares anything with the input.
    upgraded = copy.deepcopy(data)
    if generation == current_generation:
        # Needs no migration, only the stamp of the version it now conforms
        # to; that changes nothing unless the file stated an older version.
        upgraded["schema_version"] = SCHEMA_VERSION
        return upgraded

    summaries = []
    while generation != current_generation:
        if generation not in _MIGRATIONS:
            raise UnsupportedSchemaVersionError(
                f"There is no migration out of schema generation {generation!r} "
                f"toward {SCHEMA_VERSION!r}. This is a gap in the migration chain "
                "of aavso-starlist-schema, not a problem with the file."
            )
        step = _MIGRATIONS[generation]
        upgraded = step.func(upgraded)
        # Stamp each step's result, so the next step sees the version it
        # migrates from.
        upgraded["schema_version"] = step.target
        summaries.append(f"{step.target}: {step.summary}")
        generation = _generation(step.target)
    # Now in the current generation, so the data conforms to the current
    # version, which may be later than the last step's target.
    upgraded["schema_version"] = SCHEMA_VERSION

    was = "missing" if original is None else repr(original)
    warnings.warn(
        f"Starlist file schema_version was {was}; upgraded to "
        f"{upgraded['schema_version']!r} ({'; '.join(summaries)}).",
        SchemaMigrationWarning,
        stacklevel=2,
    )
    return upgraded


class StarListSet(BaseModel, PrettyPrintMixin, GenerateInstanceFromExamplesMixin):
    """
    Class to hold a list for which each entry is a star list.
    """
    # Make the generated JSON schema state its own version.
    model_config = ConfigDict(json_schema_extra={"version": SCHEMA_VERSION})

    # Put the version here because this is the file we expect manufacturers
    # to submit.
    schema_version: Annotated[
        str,
        Field(
            title="Starlist Schema Version",
            description=(
                "The version of this schema that the file was written "
                "against, assigned by AAVSO"
            ),
            json_schema_extra=dict(unit="none"),
            examples=[SCHEMA_VERSION],
            pattern=_version_pattern(),
        )
    ]
    star_lists: Annotated[
        list[StarList],
        Field(
            title="Star List Set",
            description="List of star lists",
            json_schema_extra=dict(unit="none"),
            examples=[[]]
        )
    ]

    @model_validator(mode="wrap")
    @classmethod
    def _read_older_versions(cls, data, handler):
        """
        Validate data from an older schema version as written, then upgrade it.

        This runs for ``model_validate``, ``model_validate_json`` and the
        constructor, so every way of creating a `StarListSet` reads old files.
        Data from an older generation goes through three steps:

        1. Validate it, as given, against the frozen model of its own
           generation.
        2. Convert it to plain JSON data with `_to_plain`, keeping keys the
           frozen model ignores, and upgrade it to the current version with
           `upgrade`.
        3. Validate the result against this model.

        Data already in the current generation only gets step 3, stamped with
        `SCHEMA_VERSION` if it states an older version of that generation.

        Parameters
        ----------
        data : object
            The raw input to validation; only a mapping is checked for its
            version.
        handler : callable
            Pydantic's validator for this model.

        Returns
        -------
        StarListSet
            The validated model, at the current schema version.

        Raises
        ------
        pydantic.ValidationError
            If the data is not valid for the schema version it states.
        MigrationResultError
            If older data was valid as written but the upgraded data is not
            valid; this is a bug in a migration.
        NewerSchemaVersionError, UnsupportedSchemaVersionError
            If the data's schema version is newer than, or cannot be migrated
            to, `SCHEMA_VERSION`. Like `MigrationResultError`, these are not
            wrapped in a ``pydantic.ValidationError``.

        Warns
        -----
        SchemaMigrationWarning
            If the data was written against an older schema and was upgraded.
        """
        if isinstance(data, Mapping):
            data = dict(data)
        if not isinstance(data, dict):
            return handler(data)
        generation = _supported_generation(data)
        if generation not in _MIGRATIONS:
            # Already in the current generation: restamp an older version so
            # the model always states the version it conforms to.
            return handler({**data, "schema_version": SCHEMA_VERSION})

        _MIGRATIONS[generation].model.model_validate(data)
        # Convert the raw input, not the validated model, so keys the frozen
        # model ignores still reach the migrations.
        upgraded = upgrade(_to_plain(data))
        try:
            return handler(upgraded)
        except ValidationError as error:
            raise MigrationResultError(
                f"A file valid for schema generation {generation!r} was not valid "
                f"after being upgraded to {SCHEMA_VERSION!r}. This is a bug in "
                f"aavso-starlist-schema, not in the file: {error}"
            ) from error


def generate_starlist_schema():
    """
    Generate the JSON schema for a single `StarList`.

    Returns
    -------
    str
        The schema as indented JSON text.
    """
    return json.dumps(StarList.model_json_schema(), indent=2)


def generate_star_list_set_schema():
    """
    Generate the JSON schema for a `StarListSet`.

    This is the schema of the file manufacturers submit, and what the
    reference files under ``aavso_starlist_schema_data/`` and the command line tool produce.

    Returns
    -------
    str
        The schema as indented JSON text.
    """
    return json.dumps(StarListSet.model_json_schema(), indent=2)


def _nice_name(name):
    """
    Turn a CamelCase class name into a title-case heading.

    Parameters
    ----------
    name : str
        A class name such as ``StarListSet``.

    Returns
    -------
    str
        Words separated by spaces, each capitalized: ``Star List Set``.
    """
    # Convert the name to snake case
    snake_name = to_snake(name)
    return snake_name.replace("_", " ").title()


def _generate_markdown():
    """
    Generate document with the container class, StarListSet, up at top,
    followed by the individual StarList StarItem classes.

    That reads a little better than the other way around.
    """
    return (
        "# " + _nice_name(StarListSet.__name__) + "\n\n" +
        StarListSet.markdown_table() + 3 * "\n\n" +
        "# " + _nice_name(StarList.__name__) + "\n\n" +
        StarList.markdown_table() + 3 * "\n\n" +
        "# " + _nice_name(StarItem.__name__) + "\n\n" +
        StarItem.markdown_table() +
        # Please please end with a single newline....many editors will add one
        # automatically, so it should be there.
        "\n"
    )


def main(filename, markdown=False):
    """
    Write the `StarListSet` schema to a file, as JSON or as a markdown table.

    Parameters
    ----------
    filename : str
        Output path. Its suffix is replaced with ``.json`` or ``.md``.
    markdown : bool, optional
        If ``True``, write the markdown table instead of the JSON schema.
        Default is ``False``.
    """
    extension = ".md" if markdown else ".json"
    # Make sure the path has the right suffix
    p = Path(filename).with_suffix(extension)

    if markdown:
        content = _generate_markdown()
    else:
        content = generate_star_list_set_schema()

    with p.open("w") as f:
        f.write(content)


def cli():
    """
    Run `main` as a command line program.

    This is the ``aavso-starlist-schema`` console script; argument parsing is
    handled by Fire, which is imported lazily so that importing the module
    does not require it.
    """
    import fire  # lazy import: keep `import aavso_starlist_schema` fire-free

    fire.Fire(main)


if __name__ == "__main__":
    cli()
