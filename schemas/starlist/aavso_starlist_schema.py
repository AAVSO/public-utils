"""JSON-schema definition and generators for AAVSO smart-telescope starlists.

This single-file module defines the pydantic models that describe an AAVSO Smart
Telescope starlist (``StarItem``, ``StarList``, ``StarListSet``), helpers to emit
the schema as JSON or as a markdown table, and a Fire-based command-line entry
point.
"""

import copy
import json
import math
import re
import sys
import warnings
from collections.abc import Callable, Iterable, Mapping, Sized
from enum import StrEnum
from pathlib import Path
from types import UnionType
from typing import Annotated, NamedTuple, Union, get_args, get_origin

import numpy as np
from astropy.table import MaskedColumn, Table
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from pydantic.alias_generators import to_snake
from pydantic_core import PydanticSerializationError, to_jsonable_python

import _aavso_starlist_legacy  # frozen models of the legacy generation
import _aavso_starlist_v0_2  # frozen models of the 0.2 generation
import _aavso_starlist_v0_3  # frozen models of the 0.3 generation

try:
    from _aavso_version import __version__  # top-level sibling, written by hatch-vcs
except ImportError:  # pragma: no cover - source tree without the generated file
    __version__ = "0.0.0"

# Version of the *schema* (the contract manufacturers write files against). This is
# deliberately independent of the package version above: bump it whenever the
# generated schema under aavso_starlist_schema_data/ changes. The archive test
# fails if the generated schema changes without bumping this constant.
SCHEMA_VERSION = "0.4.1"

# Every released schema version from 0.2.0 on, oldest first; add each new
# SCHEMA_VERSION here. Listed in the code, rather than read from the archive
# folders, so the versions this reader accepts never depend on the filesystem.
# The archive test fails if this differs from the archived versions.
_RELEASED_VERSIONS = ("0.2.0", "0.2.1", "0.3.0", "0.3.1", "0.3.2", "0.4.0", "0.4.1")

__all__ = [
    "AAVSOFilters",
    "StarItem",
    "StarList",
    "StarListSet",
    "generate_starlist_schema",
    "generate_star_list_set_schema",
    "write_schema",
    "cli",
    "DATA_DIR",
    "SCHEMA_VERSION",
    "upgrade",
    "validate_as_written",
    "validate_file",
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


def _type_name(annotation):
    """
    Render a field annotation the way it is written in the source.

    Parameters
    ----------
    annotation : type or typing special form
        The ``annotation`` of a pydantic ``FieldInfo``.

    Returns
    -------
    str
        ``Annotated[...]`` wrappers are dropped, a union is written with
        ``|``, ``NoneType`` is ``None``, and a generic names its parameters,
        so ``list[StarItem]`` is ``list[StarItem]`` and ``dict[key, value]``
        names both of its parameters. Any other type, such as an enum or a
        model class, is its ``__name__``, or its ``str()`` if it has none.
    """
    origin = get_origin(annotation)
    if origin is Annotated:
        return _type_name(get_args(annotation)[0])
    if origin is Union or origin is UnionType:
        return " | ".join(_type_name(arg) for arg in get_args(annotation))
    if origin is not None:
        params = ", ".join(_type_name(arg) for arg in get_args(annotation))
        return f"{_type_name(origin)}[{params}]"
    if annotation is type(None):
        return "None"
    return getattr(annotation, "__name__", str(annotation))


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
            # Escape the pipe in a union such as ``int | None`` so it does not
            # split the cell.
            type_name = _type_name(field_info.annotation).replace("|", r"\|")
            # The example is written as it would appear in a JSON file: enum
            # members as their value, None as null, strings in double quotes.
            example = json.dumps(to_jsonable_python(field_info.examples[0]), ensure_ascii=False)
            example = example.replace("|", r"\|")
            row = (
                f"| {field_info.title} | {name} | {type_name} "
                f"| {field_info.json_schema_extra['unit']} "
                f"| {field_info.description} | {example} |"
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


def _is_missing(value):
    """
    Tell whether a table entry means "no value".

    Parameters
    ----------
    value : object
        One entry of an `astropy.table.Table` column, as indexing a row gives it.

    Returns
    -------
    bool
        True if ``value`` is masked or a NaN, the two ways a table marks a
        missing number; False for anything else, including a string.
    """
    if value is np.ma.masked:
        return True
    try:
        return bool(np.isnan(value))
    except (TypeError, ValueError):
        # Not a number (a string, say), or an array cell with no single truth
        # value: either way it is a value, not a gap.
        return False


class StarItem(BaseModel, PrettyPrintMixin, GenerateInstanceFromExamplesMixin):
    """
    Definition of individual entries in an AAVSO star list.
    """
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")

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
            description="Background count level per pixel in the vicinity of the star",
            json_schema_extra=dict(unit="adu"),
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
    fwhm: Annotated[
        float | None,
        Field(
            gt=0,
            title="Star FWHM",
            description=(
                "Full width at half maximum of this star's image, as opposed "
                "to the typical value for the whole image in the star list's "
                "`fwhm`. None if not available."
            ),
            json_schema_extra=dict(unit="pixel"),
            examples=[3.4]
        )
    ] = None


class StarList(BaseModel, PrettyPrintMixin, GenerateInstanceFromExamplesMixin):
    """
    Definition of the header section of an AAVSO star list schema.
    """
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")

    obs_time: Annotated[
        str,
        Field(
            max_length=63,
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
            max_length=7,
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
            max_length=31,
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
            max_length=63,
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
            ge=1,
            title="Number of images",
            description=(
                "Number of images stacked to create this image, 1 or more. "
                "None if not applicable."
            ),
            json_schema_extra=dict(unit="none"),
            examples=[3]
        )
    ] = None
    tel_model: Annotated[
        str,
        Field(
            max_length=63,
            title="Telescope Model",
            description="Model of the telescope",
            json_schema_extra=dict(unit="none"),
            examples=["Origin 1"]
        )
    ]
    tel_firmware: Annotated[
        str,
        Field(
            max_length=31,
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
    bias: Annotated[
        float | None,
        Field(
            ge=0,
            title="Bias Level",
            description=(
                "Bias (offset) level of this frame in adu: the value "
                "`bkgd_count` would hold with no photon arrivals and no dark "
                "current, in the pixel values the counts in this star list "
                "were measured from. For a frame the vendor has "
                "dark-subtracted and then offset to keep pixel values "
                "positive, this is that fixed offset. For a raw frame with no "
                "dark, flat or bias correction, it is the mean level of a "
                "bias frame. It is 0 if the bias or offset has already been "
                "subtracted from the pixel values. For a star list measured on "
                "a stacked image it is the level in that image: the sum of the "
                "per-frame offsets for a summed stack, the per-frame offset "
                "for a mean or median stack. None if not available."
            ),
            json_schema_extra=dict(unit="adu"),
            examples=[1000.0]
        )
    ] = None
    fwhm: Annotated[
        float,
        Field(
            gt=0,
            title="Typical FWHM",
            description="Typical full width at half maximum of the stars in this image",
            json_schema_extra=dict(unit="pixel"),
            examples=[3.5]
        )
    ]
    refframe: Annotated[
        str,
        Field(
            max_length=15,
            title="Coordinate Reference Frame",
            description="Reference frame of the observation",
            json_schema_extra=dict(unit="none"),
            examples=["ICRS"]
        )
    ]
    photometry_software: Annotated[
        list[Annotated[str, Field(min_length=1, max_length=63)]],
        Field(
            fail_fast=True,
            min_length=1,
            title="Photometry Software",
            description=(
                "Software used to produce this photometry, one entry per "
                "package, each written as `<name> <version>`. Files upgraded "
                "from a schema version before 0.3.0 carry `[\"unknown\"]`."
            ),
            json_schema_extra=dict(unit="none"),
            examples=[["bandaid 1.2.3", "browser-photometry 4.5.6"]]
        )
    ]
    absolute_focus: Annotated[
        float | None,
        Field(
            title="Absolute Focus Position",
            description=(
                "Absolute focus position of the telescope, in the focuser's "
                "native units (typically motor steps), comparable only between "
                "images from the same telescope model. None if not available."
            ),
            json_schema_extra=dict(unit="none"),
            examples=[1823.0]
        )
    ] = None
    staritems: Annotated[
        list[StarItem],
        Field(
            fail_fast=True,
            max_length=100_000,
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
            The input table containing one row per star item. Columns of
            optional `StarItem` fields, such as ``fwhm``, may be omitted. When
            such a column is present it is read and validated like any other:
            a masked or NaN entry reads as None, and every other entry must
            be a valid value for the field. A NaN in a required column is not
            a valid value and is rejected. Of ``table.meta`` only the keys
            that are `StarList` fields are used; other keys, such as FITS
            header cards, are ignored.

        metadata : dict, optional
            Additional metadata to use in the StarList object.
            If not provided, the metadata from the table will be used.
            If both are provided, the metadata from this argument will
            take precedence. Optional fields may be omitted and take their
            default. It is passed through unfiltered, so a key that is not a
            `StarList` field is rejected by validation.

        Raises
        ------
        ValueError
            If the table lacks a required star item column, or the metadata
            lacks a required field.
        pydantic.ValidationError
            If a value is not valid for its field, or ``metadata`` holds a key
            that is not a field.

        Returns
        -------
        StarList
            An instance of the StarList class.
        """
        # First create the star items from the table rows. Only the required
        # star item fields must be columns; an optional one, such as the
        # per-star fwhm, is read when its column is present and a masked or
        # NaN entry in it (the table's ways of saying "no value") reads as
        # None, which is the only "no value" JSON can carry.
        required_columns = {
            name for name, field in StarItem.model_fields.items() if field.is_required()
        }
        if missing_columns := required_columns - set(table.colnames):
            raise ValueError(f"Missing columns in table: {', '.join(sorted(missing_columns))}")
        columns = [name for name in StarItem.model_fields if name in table.colnames]
        star_items = []
        for row in table:
            values = {}
            for key in columns:
                value = row[key]
                if key not in required_columns and _is_missing(value):
                    value = None
                values[key] = value
            star_items.append(StarItem(**values))

        # Construct metadata, with any passed in to the metadata argument
        # taking precedence over metadata in the table. Only the StarList
        # fields are taken from the table's metadata, which may carry
        # anything (FITS header cards, say); the metadata argument is the
        # caller's own and passes unfiltered, so a misspelled key is reported.
        final_meta = {key: value for key, value in table.meta.items() if key in cls.model_fields}
        if metadata:
            final_meta.update(metadata)

        final_meta["staritems"] = star_items

        # Only required fields must be present; a field with a default, such
        # as ``stack``, takes its default when it is missing.
        required = {name for name, field in cls.model_fields.items() if field.is_required()}
        if missing_keys := required - set(final_meta.keys()):
            raise ValueError(f"Missing keys in metadata: {', '.join(sorted(missing_keys))}")

        # Create the StarList object
        return cls.model_validate(final_meta)

    def to_table(self):
        """
        Create an astropy table from a starlist.

        Returns
        -------

        `astropy.table.Table`
            A table with one column per `StarItem` field, including the
            optional ones, and one star item per row; with no star items the
            columns are empty. The column of an optional field in which any
            star item has None is a float `astropy.table.MaskedColumn`, masked
            where the value is None, so that `from_table` reads it back as
            None. The remaining information from the starlist is stored in
            the table metadata as plain JSON types (an enum member becomes
            its string value), so the table can be written to ECSV.

        """
        table_dict = {col: [] for col in StarItem.model_fields}
        for star in self.staritems:
            for col in table_dict:
                table_dict[col].append(getattr(star, col))

        # An optional column with a None in it becomes a masked column, masked
        # where the value is None, so that from_table reads the None back. The
        # data under the mask is NaN rather than a number that could pass for
        # a measurement if the mask is ever dropped: from_table reads NaN back
        # as None, and np.asarray(column) shows the gap for what it is. This
        # assumes optional star item fields are numbers, as fwhm is.
        for col, values in table_dict.items():
            if not StarItem.model_fields[col].is_required() and any(
                value is None for value in values
            ):
                table_dict[col] = MaskedColumn(
                    [np.nan if value is None else value for value in values],
                    mask=[value is None for value in values],
                )

        table_meta = self.model_dump(mode="json")
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
    is validated between migrations. A migration raises `ValueError`, naming
    the field, for data the target generation cannot hold; through
    `StarListSet` pydantic reports that as a ``ValidationError``, so it reads
    as a problem with the file, which it is.

    The argument is always a ``dict``, but what is inside depends on how
    `upgrade` was reached:

    - From `StarListSet`, plain JSON data (dicts, lists, strings, numbers,
      booleans and ``None``) that has been validated against the frozen
      model of the file's own generation; see
      ``StarListSet._read_older_versions`` for how the input is prepared.
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


def _with_photometry_software(star_list):
    """
    Give one star list from a 0.2 file the 0.3.0 ``photometry_software`` field.

    Parameters
    ----------
    star_list : object
        One entry of ``star_lists``. Reached through the model it is always a
        dict of plain data (see `_Migration`); from `upgrade` called directly
        it may also be another mapping or a model instance, which are
        converted to a dict, or anything else.

    Returns
    -------
    dict or object
        A dict of the star list's fields with ``photometry_software`` set to
        ``["unknown"]``. The 0.2 schema does not define the key, so a value
        already there was never validated and is replaced: a file valid for
        0.2 is then always valid after migration. Anything that is neither a
        mapping nor a model is returned unchanged, for validation to reject.
    """
    if isinstance(star_list, BaseModel):
        star_list = star_list.model_dump()
    if not isinstance(star_list, Mapping):
        return star_list
    return {**star_list, "photometry_software": ["unknown"]}


def _migrate_0_2_to_0_3(data):
    """
    Migrate a 0.2 starlist set to schema version 0.3.0.

    Parameters
    ----------
    data : dict
        Raw starlist-set data from a 0.2 file. `upgrade` passes a copy, so it
        is modified in place.

    Returns
    -------
    dict
        ``data`` with ``photometry_software``, which became required in
        0.3.0, set to ``["unknown"]`` on every star list, and ``star_lists``
        a list. A tuple of star lists, which only `upgrade`
        called directly can pass, is handled too.
    """
    star_lists = data.get("star_lists")
    if isinstance(star_lists, (list, tuple)):
        data["star_lists"] = [
            _with_photometry_software(star_list) for star_list in star_lists
        ]
    return data


# What schema version 0.4.0 forbids that 0.3 allowed, as literals: this migration
# is frozen once released, so it must not read these from the live models.
_0_3_STAR_LIST_FLOATS = (
    "site_lat", "site_lon", "site_elev", "exposure", "egain", "bias", "fwhm", "absolute_focus",
)
_0_3_STAR_ITEM_FLOATS = (
    "x", "y", "ra", "dec", "tot_count", "count_err", "bkgd_count", "peak_count", "fwhm",
)
_0_4_CANNOT_UPGRADE = "schema version 0.4.0 {rule}, so this file cannot be upgraded to it"
_0_4_MAX_STRING_LENGTHS = {
    "obs_time": 63, "observer": 7, "block_filter": 31, "tel_manufac": 63, "tel_model": 63,
    "tel_firmware": 31, "refframe": 15,
}
_0_4_MAX_PHOTOMETRY_SOFTWARE_ENTRY = 63
_0_4_MAX_STARITEMS = 100_000


def _as_number(value):
    """
    Read a value the way the frozen 0.3 models read a number.

    Parameters
    ----------
    value : object
        A value from a 0.3 file, under a key of a numeric field.

    Returns
    -------
    float or None
        ``float(value)`` for anything ``float`` accepts: an int, a float, a
        bool, a numeric string (including ``"inf"``, ``"Infinity"`` and
        ``"nan"``, which the frozen 0.3 models accept) and numeric types such
        as numpy scalars or ``fractions.Fraction``, which only data built in
        Python can hold. An int too large for a float becomes infinity of its
        sign. None for anything else, such as a string that is not a number.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
    except OverflowError:
        return math.inf if value > 0 else -math.inf


def _keep_defined(record, model):
    """
    Keep only the keys of a record that a frozen 0.3 model defines.

    Parameters
    ----------
    record : object
        A starlist set, star list or star item. Reached through the model it
        is always a dict of plain data (see `_Migration`); from `upgrade`
        called directly it may also be another mapping or a model instance,
        which are converted to a dict, or anything else.
    model : type
        The frozen 0.3 model of ``record``: ``StarListSet``, ``StarList`` or
        ``StarItem`` of `_aavso_starlist_v0_3`.

    Returns
    -------
    dict or object
        A new dict of the keys of ``record`` that are fields of ``model``, in
        their original order. The 0.3 models ignored any other key, so
        dropping it loses nothing a 0.3 file meant. Anything that is neither
        a mapping nor a model is returned unchanged, for validation to
        reject.
    """
    if isinstance(record, BaseModel):
        record = record.model_dump()
    if not isinstance(record, Mapping):
        return record
    return {key: value for key, value in record.items() if key in model.model_fields}


def _refuse_non_finite(record, float_fields, path):
    """
    Refuse a record with an infinite or NaN float, which 0.4.0 cannot hold.

    Parameters
    ----------
    record : dict
        A star list or star item from a 0.3 file.
    float_fields : tuple of str
        The names of the float fields of ``record``'s 0.3 model.
    path : str
        Where ``record`` is in the file, such as ``star_lists[0]``, for the
        error message.

    Raises
    ------
    ValueError
        If one of ``float_fields`` holds infinity or NaN, as a number or as a
        numeric string; the message names ``path`` and the field.
    """
    for name in float_fields:
        if name in record:
            number = _as_number(record[name])
            if number is not None and not math.isfinite(number):
                raise ValueError(
                    f"{path}.{name} is {record[name]!r}; "
                    + _0_4_CANNOT_UPGRADE.format(rule="requires a finite number")
                )


def _refuse_overlong(star_list, path):
    """
    Refuse a star list with a string longer than 0.4.0 allows.

    Parameters
    ----------
    star_list : dict
        A star list from a 0.3 file.
    path : str
        Where ``star_list`` is in the file, such as ``star_lists[0]``, for the
        error message.

    Raises
    ------
    ValueError
        If one of the fields in `_0_4_MAX_STRING_LENGTHS`, or an entry of
        ``photometry_software``, is a string with more characters (code
        points, as pydantic counts them) than 0.4.0 allows; the message names
        ``path`` and the field.
    """
    for name, limit in _0_4_MAX_STRING_LENGTHS.items():
        value = star_list.get(name)
        if isinstance(value, str) and len(value) > limit:
            raise ValueError(
                f"{path}.{name} has {len(value)} characters; "
                + _0_4_CANNOT_UPGRADE.format(rule=f"allows at most {limit}")
            )
    software = star_list.get("photometry_software")
    if isinstance(software, (list, tuple)):
        for index, entry in enumerate(software):
            if isinstance(entry, str) and len(entry) > _0_4_MAX_PHOTOMETRY_SOFTWARE_ENTRY:
                raise ValueError(
                    f"{path}.photometry_software[{index}] has {len(entry)} characters; "
                    + _0_4_CANNOT_UPGRADE.format(
                        rule=f"allows at most {_0_4_MAX_PHOTOMETRY_SOFTWARE_ENTRY}"
                    )
                )


def _star_item_0_4(item, path):
    """
    Migrate one star item of a 0.3 file to schema version 0.4.0.

    Parameters
    ----------
    item : object
        One entry of a star list's ``staritems``; see `_keep_defined` for
        what it may be.
    path : str
        Where ``item`` is in the file, such as ``star_lists[0].staritems[2]``,
        for the error message.

    Returns
    -------
    dict or object
        ``item`` as a dict with only the keys the 0.3 ``StarItem`` defines,
        or ``item`` unchanged if it is neither a mapping nor a model.

    Raises
    ------
    ValueError
        If a float field of ``item`` holds infinity or NaN.
    """
    item = _keep_defined(item, _aavso_starlist_v0_3.StarItem)
    if isinstance(item, dict):
        _refuse_non_finite(item, _0_3_STAR_ITEM_FLOATS, path)
    return item


def _star_list_0_4(star_list, path):
    """
    Migrate one star list of a 0.3 file to schema version 0.4.0.

    Parameters
    ----------
    star_list : object
        One entry of ``star_lists``; see `_keep_defined` for what it may be.
    path : str
        Where ``star_list`` is in the file, such as ``star_lists[0]``, for the
        error message.

    Returns
    -------
    dict or object
        ``star_list`` as a dict with only the keys the 0.3 ``StarList``
        defines, a ``stack`` below 1 set to None, and each of its star items
        migrated by `_star_item_0_4` (``staritems`` becomes a list); or
        ``star_list`` unchanged if it is neither a mapping nor a model.

    Raises
    ------
    ValueError
        If a float field of the star list or of one of its star items holds
        infinity or NaN, a string field is longer than 0.4.0 allows, or the
        star list has more than 100000 star items.
    """
    star_list = _keep_defined(star_list, _aavso_starlist_v0_3.StarList)
    if not isinstance(star_list, dict):
        return star_list
    _refuse_non_finite(star_list, _0_3_STAR_LIST_FLOATS, path)
    _refuse_overlong(star_list, path)
    # A stack of 0 or less counts no images; "not applicable" is its honest
    # reading. The frozen 0.3 model also took "0" and False, so read numbers.
    if "stack" in star_list:
        number = _as_number(star_list["stack"])
        if number is not None and number < 1:
            star_list["stack"] = None
    items = star_list.get("staritems")
    if isinstance(items, (list, tuple)):
        if len(items) > _0_4_MAX_STARITEMS:
            raise ValueError(
                f"{path}.staritems has {len(items)} entries; "
                + _0_4_CANNOT_UPGRADE.format(rule=f"allows at most {_0_4_MAX_STARITEMS}")
            )
        star_list["staritems"] = [
            _star_item_0_4(item, f"{path}.staritems[{index}]") for index, item in enumerate(items)
        ]
    return star_list


def _migrate_0_3_to_0_4(data):
    """
    Migrate a 0.3 starlist set to schema version 0.4.0.

    Parameters
    ----------
    data : dict
        Raw starlist-set data from a 0.3 file. `upgrade` passes a copy.

    Returns
    -------
    dict
        ``data`` with the keys the 0.3 models do not define dropped at every
        level (the 0.3 reader ignored them; 0.4.0 forbids them), a ``stack``
        below 1 set to None on every star list, and ``star_lists`` a list. A
        tuple of star lists or star items, which only `upgrade` called
        directly can pass, is handled too.

    Raises
    ------
    ValueError
        If a float field holds infinity or NaN, a string field is longer
        than 0.4.0 allows, or a star list has more than 100000 star items.
        Through `StarListSet` pydantic reports it as a ``ValidationError``
        (type ``value_error``); from `upgrade` it is raised as is. Such a
        file is valid for 0.3 but has no honest 0.4 form.
    """
    data = _keep_defined(data, _aavso_starlist_v0_3.StarListSet)
    star_lists = data.get("star_lists")
    if isinstance(star_lists, (list, tuple)):
        data["star_lists"] = [
            _star_list_0_4(star_list, f"star_lists[{index}]")
            for index, star_list in enumerate(star_lists)
        ]
    return data


# Keyed by the generation a file is in; the value takes it to the next generation.
# Each breaking change adds an entry here, keyed by the generation being left,
# with a frozen copy of that generation's models in its own _aavso_starlist_*.py.
_MIGRATIONS = {
    "legacy": _Migration(
        _aavso_starlist_legacy.StarListSet,
        "0.2.0",
        _migrate_legacy_to_0_2,
        "schema_version became required; no other field changed",
    ),
    "0.2": _Migration(
        _aavso_starlist_v0_2.StarListSet,
        "0.3.0",
        _migrate_0_2_to_0_3,
        'photometry_software became required; every star list gets ["unknown"]',
    ),
    "0.3": _Migration(
        _aavso_starlist_v0_3.StarListSet,
        "0.4.0",
        _migrate_0_3_to_0_4,
        "keys the 0.3 schema does not define were dropped and a stack below 1 became null",
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


def _fields_of_frozen_set(data):
    """
    Read a frozen starlist set of an older generation as the fields it was given.

    Parameters
    ----------
    data : object
        The input to `validate_as_written` or `StarListSet`.

    Returns
    -------
    object
        If ``data`` is an instance of the frozen ``StarListSet`` of any
        generation in `_MIGRATIONS`, a dict of the fields it was given (see
        `_given_fields`), as `_prune_and_materialize` reads a nested frozen
        instance; its version is then read like that of any mapping.
        Otherwise ``data`` as given.
    """
    if isinstance(data, tuple(step.model for step in _MIGRATIONS.values())):
        return _given_fields(data)
    return data


def validate_as_written(data):
    """
    Validate raw starlist-set data against the schema of its own generation.

    Parameters
    ----------
    data : dict or frozen StarListSet
        The starlist set as read from JSON. An instance of the frozen
        ``StarListSet`` of an older generation is read as the fields it was
        given (see `_fields_of_frozen_set`) and validated like a dict.

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

    It does not apply anything a later version added: a 0.3.x file may hold
    infinity or NaN, strings of any length, more than 100000 star items or a
    ``stack`` of 0, and ``model_dump(mode="json")`` of the result emits them
    as they are. To hold every file to the current schema, read it with
    ``StarListSet.model_validate`` or `validate_file`, which upgrade it.

    A legacy file without a ``schema_version`` comes back with the legacy
    model's frozen default in that field, a version the file never stated. To
    pass the result on, use ``model_dump(exclude_unset=True)``, or validate
    the raw data with `StarListSet` directly.

    Data that is not a JSON object at all (a list, ``null``, a number) has no
    version to read, so it is handed to `StarListSet` for the same
    ``ValidationError`` that the default validation raises.
    """
    data = _fields_of_frozen_set(data)
    if not isinstance(data, Mapping):
        return StarListSet.model_validate(data)
    generation = _supported_generation(data)
    if generation in _MIGRATIONS:
        return _MIGRATIONS[generation].model.model_validate(data)
    return StarListSet.model_validate(data)


def validate_file(path, as_written=False):
    """
    Validate a starlist file against the schema.

    This is the file-level convenience for a user who does not want to deal
    with pydantic: it reads the file, parses it and validates it the way
    `StarListSet` does, or, with ``as_written``, the way `validate_as_written`
    does.

    Parameters
    ----------
    path : str or pathlib.Path
        The starlist file, JSON text encoded as UTF-8, with or without a
        byte order mark.
    as_written : bool, optional
        If ``True``, validate the file against the schema of the version it
        states, without upgrading it; see `validate_as_written`. That does
        not apply anything a later version added (a 0.3.x file may hold
        infinity or NaN, for one), so to hold every file to the current
        schema leave this ``False``. Default is ``False``: validate the file
        and upgrade it to the current schema version, as `StarListSet` does.

    Returns
    -------
    pydantic.BaseModel
        By default a `StarListSet` at the current schema version, upgraded if
        the file is from an older generation. With ``as_written=True``, what
        `validate_as_written` returns: a `StarListSet` if the file is in the
        current generation (a file from an older release of the current
        generation is read as the current version), otherwise an instance of
        the frozen model of its own generation, stating the file's own
        version.

    Raises
    ------
    FileNotFoundError
        If there is no file at ``path``.
    json.JSONDecodeError
        If the file is not valid JSON.
    pydantic.ValidationError
        If the file does not match the schema of its own generation.
    MigrationResultError
        If the file was valid for its own generation but not after being
        upgraded; a bug in a migration, not in the file. Never raised with
        ``as_written=True``.
    NewerSchemaVersionError
        If the file was written against a newer schema than this reader.
    UnsupportedSchemaVersionError
        If the version is null, cannot be interpreted, is 0.2.0 or later but
        was never released, or has no migration path.

    Warns
    -----
    SchemaMigrationWarning
        If the file was written against an older schema and was upgraded.
        Never with ``as_written=True``.
    """
    # utf-8-sig is UTF-8 that also drops a leading byte order mark, which
    # Windows editors commonly write; RFC 8259 lets a JSON parser ignore it.
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if as_written:
        return validate_as_written(data)
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
        ``None``: every mapping becomes a dict, every list a new list, and
        every other value is serialized by pydantic, so a model instance
        becomes a dict of its fields, an enum member its value and a
        datetime an ISO 8601 string. A value of a type pydantic cannot
        serialize is passed on as given.

    Notes
    -----
    Only a ``list`` is converted item by item, not every list-like value
    (see `_is_sequence_like`): by the time this runs, `_prune_and_materialize`
    has read the value under every list field into a list and the frozen
    model has validated the rest, so an iterable left anywhere else is a
    value pydantic read as a scalar, such as a float subclass that defines
    ``__iter__``, and must stay one.
    """
    if isinstance(value, Mapping):
        return {key: _to_plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_plain(item) for item in value]
    try:
        return to_jsonable_python(value)
    except PydanticSerializationError:
        return value


def _is_sequence_like(value):
    """
    Tell whether a pydantic list field would read a value as a list.

    Parameters
    ----------
    value : object
        Raw starlist-set data, or any value inside it.

    Returns
    -------
    bool
        True for an iterable that is not a string, bytes, mapping, model
        instance or zero-dimensional array.
    """
    if isinstance(value, (str, bytes, bytearray, Mapping, BaseModel)):
        return False
    if getattr(value, "ndim", 1) == 0:
        return False
    return isinstance(value, Iterable)


def _list_item_type(annotation):
    """
    Find the item type of a list field's annotation.

    Parameters
    ----------
    annotation : type or typing special form
        The ``annotation`` of a pydantic ``FieldInfo``.

    Returns
    -------
    type or typing special form or None
        The item type if ``annotation`` is a list, possibly wrapped in
        ``Annotated`` or in a union with ``None``; ``object`` for a bare
        ``list``. None if it is not a list, or is a union of several lists.
    """
    origin = get_origin(annotation)
    if origin is Annotated:
        return _list_item_type(get_args(annotation)[0])
    if origin is Union or origin is UnionType:
        found = [
            item for item in map(_list_item_type, get_args(annotation)) if item is not None
        ]
        return found[0] if len(found) == 1 else None
    if annotation is list:
        return object
    if origin is list:
        if not get_args(annotation):
            return object
        item = get_args(annotation)[0]
        while get_origin(item) is Annotated:
            item = get_args(item)[0]
        return item
    return None


def _given_fields(instance):
    """
    Read a model instance as the fields it was given, for validation.

    Parameters
    ----------
    instance : pydantic.BaseModel
        An instance of a frozen model, built in Python.

    Returns
    -------
    dict
        The fields set on ``instance``, defaults excluded, so it upgrades
        like the same data given as a dict. Serializer warnings are off: the
        frozen models do not revalidate their own instances, so one changed
        after it was built may hold a wrong value, which validation then
        reports where it is.
    """
    return instance.model_dump(warnings=False, exclude_unset=True)


def _nested_model(field):
    """
    Find the model a list-of-models field holds.

    Parameters
    ----------
    field : pydantic.fields.FieldInfo or None
        A field of a model, or None for a field the model does not define.

    Returns
    -------
    type or None
        The item type of ``field`` if it is a list of pydantic models; None
        otherwise.
    """
    item_type = None if field is None else _list_item_type(field.annotation)
    if isinstance(item_type, type) and issubclass(item_type, BaseModel):
        return item_type
    return None


def _max_length(field):
    """
    Find the largest number of entries a list field allows.

    Parameters
    ----------
    field : pydantic.fields.FieldInfo or None
        A field of a model, or None for a field the model does not define.

    Returns
    -------
    int or None
        The ``max_length`` of ``field``, or None if it has none or ``field``
        is None.
    """
    if field is None:
        return None
    return next(
        (
            constraint.max_length
            for constraint in field.metadata
            if getattr(constraint, "max_length", None) is not None
        ),
        None,
    )


def _dotted(path):
    """
    Name a place in a starlist set without its list indexes.

    Parameters
    ----------
    path : tuple of str or int
        Field names and list indexes leading from the starlist set to a
        record, such as ``("star_lists", 0, "staritems", 2)``.

    Returns
    -------
    str
        The names joined with dots, each index written as ``[]``, such as
        ``star_lists[].staritems[]``; ``""`` for the starlist set itself.
    """
    name = ""
    for part in path:
        if isinstance(part, int):
            name += "[]"
        else:
            name += f".{part}" if name else part
    return name


def _list_error(error_type, loc, value, ctx=None):
    """
    Build the validation error `_read_list` raises.

    Parameters
    ----------
    error_type : str
        The pydantic error type, such as ``iteration_error``.
    loc : tuple of str or int
        Where the error is in the starlist set.
    value : object
        The value under the list field.
    ctx : dict, optional
        The error's context, as pydantic gives it for ``error_type``.

    Returns
    -------
    pydantic.ValidationError
        One error of ``error_type`` at ``loc``, titled ``StarListSet``.
    """
    line = {"type": error_type, "loc": loc, "input": value}
    if ctx is not None:
        line["ctx"] = ctx
    return ValidationError.from_exception_data("StarListSet", [line])


def _key_name(path, key):
    """
    Name a key dropped from a record, for the migration warning.

    Parameters
    ----------
    path : tuple of str or int
        Where the record is in the starlist set (see `_dotted`).
    key : object
        The key, as the record holds it.

    Returns
    -------
    str
        ``<record>.<key>`` for a str key, such as ``star_lists[].bias``, or
        the key alone at the top level; ``<record>[<repr of key>]`` for any
        other key.
    """
    prefix = _dotted(path)
    if not isinstance(key, str):
        return f"{prefix}[{key!r}]"
    return f"{prefix}.{key}" if prefix else key


def _read_list(value, limit, path):
    """
    Read the list-like value under a list field of an older file into a list.

    Parameters
    ----------
    value : object
        A list-like value (see `_is_sequence_like`) under a list field.
    limit : int or None
        The largest number of entries the field may hold, or None for no
        limit. A value without a length, such as a generator, is read only
        up to ``limit + 1`` entries; a value with a length is read in full.
    path : tuple of str or int
        Where the field is in the starlist set, for the error.

    Returns
    -------
    list
        The entries of ``value``, each as given.

    Raises
    ------
    pydantic.ValidationError
        As pydantic reports the same value in a file of the current
        generation: ``list_type`` at ``path`` if ``value`` cannot be
        iterated at all; ``iteration_error`` at ``(*path, index)`` if
        reading an entry raises, which is its ``__cause__``; ``too_long``
        at ``path`` if a value without a length holds more than ``limit``
        entries.
    """
    try:
        iterator = iter(value)
    except Exception as error:
        raise _list_error("list_type", path, value) from error
    if isinstance(value, Sized):
        limit = None
    items = []
    while limit is None or len(items) <= limit:
        try:
            items.append(next(iterator))
        except StopIteration:
            return items
        except Exception as error:
            raise _list_error(
                "iteration_error",
                (*path, len(items)),
                value,
                {"error": f"{type(error).__name__}: {error}"},
            ) from error
    # The real length is unknown, as it is to pydantic, which reports this
    # too with no actual length ("not more").
    raise _list_error(
        "too_long",
        path,
        value,
        {"field_type": "List", "max_length": limit, "actual_length": None},
    )


def _prune_and_materialize(record, model, current=None, path=(), dropped=None):
    """
    Keep the keys a frozen model defines, reading one-shot iterators in lists.

    Parameters
    ----------
    record : object
        Raw starlist-set data from a file of a generation before the current
        one, or a star list or star item inside it.
    model : type
        The frozen model of ``record`` in the file's own generation.
    current : type, optional
        The current model of ``record``, whose list fields give the limit on
        the entries read. Default None: no limit.
    path : tuple of str or int, optional
        Where ``record`` is in the starlist set, for errors and dropped key
        names. Default ``()``, the starlist set itself.
    dropped : set of str, optional
        If given, the name of each key dropped (see `_key_name`) is added to
        it; never a value. Default None.

    Returns
    -------
    object
        An instance of ``model`` is first read as the fields it was given
        (see `_given_fields`); any other value that is not a mapping is
        returned as given, for validation to reject. A mapping becomes a
        dict of its keys in their original order, where a key ``model`` does
        not define is dropped without its value being read (unless ``model``
        forbids extra keys, when it is kept for validation to reject); a
        list-like value under a list of models is read into a list with each
        item passed through this function; a list-like value under any other
        list field is read into a list, its items as given; and every other
        value is kept as given.

        A dropped value is never read because nothing downstream may use a
        value no schema validated, and reading it could fail or never finish.
        A one-shot iterator under a list field is read once, here, so the
        frozen validation and the migrations see the same items; one without
        a length is read at most one entry past what ``current`` allows and
        then refused (see `_read_list`).

    Raises
    ------
    pydantic.ValidationError
        If a value under a list field cannot be read; see `_read_list`.
    """
    if isinstance(record, model):
        record = _given_fields(record)
    if not isinstance(record, Mapping):
        return record
    forbids_extra = model.model_config.get("extra") == "forbid"
    kept = {}
    for key, value in record.items():
        field = model.model_fields.get(key)
        if field is None:
            if forbids_extra:
                kept[key] = value
            elif dropped is not None:
                dropped.add(_key_name(path, key))
            continue
        if _list_item_type(field.annotation) is None or not _is_sequence_like(value):
            kept[key] = value
            continue
        current_field = None if current is None else current.model_fields.get(key)
        items = _read_list(value, _max_length(current_field), (*path, key))
        nested = _nested_model(field)
        if nested is None:
            kept[key] = items
        else:
            kept[key] = [
                _prune_and_materialize(
                    item, nested, _nested_model(current_field), (*path, key, index), dropped
                )
                for index, item in enumerate(items)
            ]
    return kept


# The most dropped key names the migration warning lists.
_MAX_DROPPED_NAMES = 20


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
    TypeError
        If ``data`` is a value that is not a mapping, such as a list: it has
        no version to read.
    NewerSchemaVersionError
        If the file was written against a newer schema than this reader.
    UnsupportedSchemaVersionError
        If the version is null, cannot be interpreted, is 0.2.0 or later but
        was never released, or has no migration path.
    ValueError
        If a migration refuses the data because the target generation cannot
        hold it; the message names the field.

    Warns
    -----
    SchemaMigrationWarning
        Once, if the data had to be migrated. Called directly, nothing is
        dropped before the migrations, so the warning names no dropped keys
        (see `_upgrade`).
    """
    return _upgrade(data, stacklevel=3)


def _upgrade(data, dropped=(), stacklevel=2):
    """
    Upgrade raw starlist-set data, naming in the warning the keys dropped first.

    This does the work of `upgrade`, which calls it with no dropped keys;
    `StarListSet` calls it with the keys `_prune_and_materialize` dropped
    from an older file before its frozen validation.

    Parameters
    ----------
    data : dict
        The starlist set as read from JSON, before any validation, or, from
        `StarListSet`, after pruning, validation and conversion to plain
        data.
    dropped : collection of str, optional
        The names of the keys dropped from ``data`` unread because the frozen
        model of its own generation does not define them, such as
        ``star_lists[].bias``. Default ``()``: none.
    stacklevel : int, optional
        Passed to `warnings.warn`: 2, the default, attributes the warning to
        the caller of this function (the `StarListSet` validator, in this
        module, so a filter on ``module="aavso_starlist_schema"`` matches);
        `upgrade` passes 3, attributing it to the caller of `upgrade`.

    Returns
    -------
    dict
        As `upgrade` returns.

    Raises
    ------
    TypeError, NewerSchemaVersionError, UnsupportedSchemaVersionError, ValueError
        As `upgrade` raises them.

    Warns
    -----
    SchemaMigrationWarning
        Once, if the data had to be migrated, naming the original version,
        the new version, what each migration changed and, if ``dropped`` is
        not empty, the dropped keys, sorted, each as its ``repr()``, at most
        `_MAX_DROPPED_NAMES` of them followed by how many more there are.
    """
    if not isinstance(data, Mapping):
        raise TypeError(
            "upgrade() needs a dict (a starlist set as read from JSON), "
            f"got {type(data).__name__}"
        )
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
    lost = ""
    if dropped:
        # The names are file data: repr() escapes a newline or a comma in one
        # (which could otherwise forge a line of output or a second name),
        # and the list is capped.
        names = sorted(dropped)
        lost = "; keys the file's schema does not define were dropped unread: " + ", ".join(
            repr(name) for name in names[:_MAX_DROPPED_NAMES]
        )
        if len(names) > _MAX_DROPPED_NAMES:
            lost += f" and {len(names) - _MAX_DROPPED_NAMES} more"
    warnings.warn(
        f"Starlist file schema_version was {was}; upgraded to "
        f"{upgraded['schema_version']!r} ({'; '.join(summaries)}){lost}.",
        SchemaMigrationWarning,
        stacklevel=stacklevel,
    )
    return upgraded


class StarListSet(BaseModel, PrettyPrintMixin, GenerateInstanceFromExamplesMixin):
    """
    Class to hold a list for which each entry is a star list.
    """
    # Make the generated JSON schema state its own version.
    model_config = ConfigDict(extra="forbid", json_schema_extra={"version": SCHEMA_VERSION})

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
            fail_fast=True,
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
        Validate data from an older schema version as given, then upgrade it.

        This runs for ``model_validate``, ``model_validate_json`` and the
        constructor, so every way of creating a `StarListSet` reads old files.
        Data from an older generation goes through three steps:

        1. Prepare it with `_prune_and_materialize`: drop the keys the frozen
           model of its own generation does not define, unread, and read
           list-like values under list fields into lists, once and bounded;
           then validate it, otherwise as given, against that frozen model,
           so every value the first migration sees has been validated by the
           schema of the file's own generation.
        2. Convert the data to plain JSON data with `_to_plain` and upgrade
           it to the current version with `_upgrade`, which names the
           dropped keys in its warning.
        3. Validate the result against this model.

        An instance of the frozen ``StarListSet`` of an older generation,
        passed whole, is first read as the fields it was given (see
        `_fields_of_frozen_set`) and then handled like a mapping.

        Data already in the current generation only gets step 3, stamped with
        `SCHEMA_VERSION` if it states an older version of that generation.

        Parameters
        ----------
        data : object
            The raw input to validation; only a mapping, or a frozen
            ``StarListSet`` of an older generation read as one, is checked
            for its version.
        handler : callable
            Pydantic's validator for this model.

        Returns
        -------
        StarListSet
            The validated model, at the current schema version.

        Raises
        ------
        pydantic.ValidationError
            If the data is not valid for the schema version it states; if
            the value under one of its list fields cannot be read (see
            `_read_list`); or if a migration refuses it because the current
            generation cannot hold a value in it (a `ValueError` from the
            migration, which pydantic wraps).
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
            If the data was written against an older schema and was upgraded;
            once, naming the keys dropped unread in step 1, if any.
        """
        data = _fields_of_frozen_set(data)
        if isinstance(data, Mapping):
            data = dict(data)
        if not isinstance(data, dict):
            return handler(data)
        generation = _supported_generation(data)
        if generation not in _MIGRATIONS:
            # Already in the current generation: restamp an older version so
            # the model always states the version it conforms to.
            return handler({**data, "schema_version": SCHEMA_VERSION})

        # Drop keys the file's own frozen model does not define unread (it
        # ignores them, and no migration may use a value no schema
        # validated), collecting their names for the warning, and read
        # one-shot iterators under list fields once, before both steps: a
        # generator read by the frozen model would reach the migrations
        # empty. Validate the data otherwise as given, so a datetime in a
        # string field is still refused. Upgrade the plain input, not the
        # validated model, so the upgraded data holds what the file holds,
        # not the model's defaults.
        frozen = _MIGRATIONS[generation].model
        dropped = set()
        data = _prune_and_materialize(data, frozen, cls, (), dropped)
        frozen.model_validate(data)
        upgraded = _upgrade(_to_plain(data), dropped, stacklevel=2)
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


def _report_validation(path, as_written=False):
    """
    Validate a starlist file and report the result on the console.

    This is the ``validate`` command of the ``aavso-starlist-schema`` console
    script. It only reads the file; nothing is written.

    Parameters
    ----------
    path : str
        The starlist file, as given on the command line.
    as_written : bool, optional
        If ``True``, validate the file against the schema of the version it
        states, without upgrading it (see `validate_file`). Only the limits
        of that version's generation apply, so an older file may pass that
        the current schema would refuse.

    Raises
    ------
    SystemExit
        With status 1 if the file is missing, cannot be read (a directory, a
        permission problem), is not UTF-8 text, is not JSON, is not valid for
        its schema version or cannot be upgraded. The reason is written to
        stderr first, so a shell script can rely on the exit status alone.

    Notes
    -----
    On success one line goes to stdout, ``<path>: valid (schema version
    <version>)``, where ``<version>`` is the one the validated model states.
    If the file was upgraded, the `SchemaMigrationWarning` message follows on
    a second line, instead of being issued as a warning.
    """
    # Fire evaluates a command-line value that looks like a Python literal
    # before passing it on, so a file named 123, True or None arrives as an
    # int, a bool or None; make it a str again. (A float-looking name such
    # as 2024.10 is already 2024.1 by then; nothing to do about that here.)
    path = str(path)
    try:
        with warnings.catch_warnings(record=True) as caught:
            # Record the migration warning even where a warnings filter would
            # turn it into an error, for instance under pytest.
            warnings.simplefilter("always", SchemaMigrationWarning)
            model = validate_file(path, as_written=as_written)
    except FileNotFoundError:
        problem = "no such file"
    except OSError as error:
        # A directory, no read permission, and so on.
        problem = f"cannot read the file: {error.strerror}"
    except UnicodeDecodeError as error:
        # A ValueError like JSONDecodeError, but a sibling of it, not a
        # subclass: the file never reached the JSON parser.
        problem = f"not UTF-8 text: {error}"
    except json.JSONDecodeError as error:
        problem = f"not valid JSON: {error}"
    except ValidationError as error:
        problem = str(error)
    except MigrationResultError as error:
        problem = f"upgrading the file gave an invalid result: {error}"
    except NewerSchemaVersionError as error:
        problem = f"newer schema version: {error}"
    except UnsupportedSchemaVersionError as error:
        problem = f"unsupported schema version: {error}"
    else:
        print(f"{path}: valid (schema version {model.schema_version})")
        for warning in caught:
            if issubclass(warning.category, SchemaMigrationWarning):
                print(warning.message)
            else:
                # Not ours to report; let it through as the warning it was.
                warnings.warn_explicit(
                    warning.message, warning.category, warning.filename, warning.lineno
                )
        return

    print(f"{path}: invalid\n{problem}", file=sys.stderr)
    raise SystemExit(1)


def write_schema(filename, markdown=False, force=False):
    """
    Write the `StarListSet` schema to a file.

    This is the ``write-schema`` command of the ``aavso-starlist-schema``
    console script. To check a starlist file use the ``validate`` command
    instead; this command never reads the file it is given.

    Parameters
    ----------
    filename : str
        Output path; its suffix is replaced with ``.json`` or ``.md``.
    markdown : bool, optional
        If ``True``, write the markdown table instead of the JSON schema.
        Default is ``False``.
    force : bool, optional
        If ``True``, overwrite the output file if it already exists. Default
        is ``False``: an existing file is left alone and the command fails.

    Raises
    ------
    SystemExit
        With status 1 if the output file exists (a dangling symlink included)
        and ``force`` is ``False``, or if the file cannot be written: a
        directory at the output path, a missing parent directory, no write
        permission. The reason is written to stderr.
    """
    extension = ".md" if markdown else ".json"
    # Fire evaluates a command-line value that looks like a Python literal
    # before passing it on, so a path named 2024 or True arrives as an int or
    # a bool; make it a str again. (A float-looking name such as 2024.10 is
    # already 2024.1 by then; nothing to do about that here.) Then make sure
    # the path has the right suffix.
    p = Path(str(filename)).with_suffix(extension)

    if markdown:
        content = _generate_markdown()
    else:
        content = generate_star_list_set_schema()

    # Most likely the user meant `validate`, or forgot that the path is the
    # output: never write over a file that is there unless told to. Exclusive
    # creation ("x") leaves that check to the OS, so nothing can appear
    # between a check and the write, and a dangling symlink, which exists()
    # reports as absent, is refused rather than written through.
    #
    # The committed reference files are UTF-8 with LF line endings, and the
    # pre-commit hook regenerates them on any platform: fix both so the
    # output is byte-identical everywhere.
    try:
        with p.open("w" if force else "x", encoding="utf-8", newline="\n") as f:
            f.write(content)
    except FileExistsError:
        problem = "exists; not overwriting it (use --force to replace it)"
    except OSError as error:
        # A directory at the path, a missing parent, no write permission.
        problem = f"cannot write the file: {error.strerror}"
    else:
        return

    print(f"{p}: {problem}", file=sys.stderr)
    raise SystemExit(1)


def cli():
    """
    Run the ``aavso-starlist-schema`` console script.

    The script has two commands, ``validate <file>`` and ``write-schema
    <path>``; the first only reads its file and the second only writes to
    its path, so neither can be mistaken for the other. Argument parsing is
    handled by Fire, which is imported lazily so that importing the module
    does not require it.
    """
    import fire  # lazy import: keep `import aavso_starlist_schema` fire-free

    fire.Fire({"validate": _report_validation, "write-schema": write_schema})


if __name__ == "__main__":
    cli()
