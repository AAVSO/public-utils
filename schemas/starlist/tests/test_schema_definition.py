import json
import re
from typing import Annotated, Optional

import numpy as np
import pytest
from astropy.table import MaskedColumn, Table
from pydantic import ValidationError

from aavso_starlist_schema import (
    DATA_DIR,
    SCHEMA_VERSION,
    AAVSOFilters,
    BayerGains,
    StarItem,
    StarList,
    StarListSet,
    _generate_markdown,
    _is_missing,
    _type_name,
    cli,
    generate_star_list_set_schema,
    generate_starlist_schema,
    main,
)


@pytest.mark.parametrize("klass", [StarItem, StarList, StarListSet])
def test_schema_has_all_require_properties(klass):
    # Every field of every model carries a title, description, examples and a unit
    # (scale, for obs_time), so the generated schema and markdown are complete.
    required_fields = [
        "title",
        "description",
        "examples",
    ]

    for field_name, field_info in klass.model_fields.items():
        for required_field in required_fields:
            assert len(getattr(field_info, required_field)) > 0

        if field_name != "obs_time":
            assert len(field_info.json_schema_extra["unit"]) > 0
        else:
            assert len(field_info.json_schema_extra["scale"]) > 0


@pytest.mark.parametrize("klass", [StarItem, StarList, StarListSet])
def test_example_values_are_valid(klass):
    # Check that the example values are valid
    # by creating an instance of the class with the example values
    klass.from_examples()


def test_starlist_markdown_table():
    # The committed markdown reference file is current with the models.
    mdown_file = DATA_DIR / "schema_definition.md"
    with open(mdown_file) as f:
        mdown_file_content = f.read()

    assert _generate_markdown() == mdown_file_content


def test_absolute_focus_is_optional():
    # absolute_focus may be left out of a star list, which then reads it as
    # None; it is not merely nullable but also absent from "required".
    data = StarList.from_examples().model_dump()
    del data["absolute_focus"]

    assert StarList.model_validate(data).absolute_focus is None
    assert "absolute_focus" not in StarList.model_json_schema()["required"]


def test_absolute_focus_is_kept_when_present():
    # A value given for absolute_focus survives a JSON round trip.
    star_list = StarList.from_examples()

    assert star_list.absolute_focus == 1823.0
    assert StarList.model_validate_json(star_list.model_dump_json()).absolute_focus == 1823.0


def test_star_item_fwhm_is_optional():
    # A star item's fwhm may be left out, which then reads it as None; it is
    # not merely nullable but also absent from "required".
    data = StarItem.from_examples().model_dump()
    del data["fwhm"]

    assert StarItem.model_validate(data).fwhm is None
    assert "fwhm" not in StarItem.model_json_schema()["required"]


def test_star_item_fwhm_is_kept_when_present():
    # A value given for a star item's fwhm survives a JSON round trip.
    star_item = StarItem.from_examples()

    assert star_item.fwhm == 3.4
    assert StarItem.model_validate_json(star_item.model_dump_json()).fwhm == 3.4


@pytest.mark.parametrize("value", [0, -1.5])
def test_star_item_fwhm_must_be_positive(value):
    # A width of zero or less is not a width; both are rejected.
    data = StarItem.from_examples().model_dump()
    data["fwhm"] = value

    with pytest.raises(ValidationError, match="greater than 0"):
        StarItem.model_validate(data)


@pytest.mark.parametrize("value", [float("inf"), float("nan")])
def test_star_item_fwhm_rejects_infinity_and_nan(value):
    # JSON has no infinity or NaN, so pydantic would write either as null,
    # which is a valid fwhm meaning "not available"; the value would be lost
    # with no error. Both are rejected instead.
    data = StarItem.from_examples().model_dump()
    data["fwhm"] = value

    with pytest.raises(ValidationError, match="finite number"):
        StarItem.model_validate(data)


def _example_egain_per_channel():
    """
    Give the per-channel example value of ``egain``.

    Returns
    -------
    dict of str to float
        The second example of `StarList.egain`, one gain per Bayer channel.
    """
    example = StarList.model_fields["egain"].examples[1]
    assert isinstance(example, dict)
    return example


def test_egain_scalar_round_trips():
    # A single system gain, the only form before 0.3.2, is still accepted and
    # survives a JSON round trip as a float.
    star_list = StarList.from_examples()

    assert star_list.egain == 1.2
    assert StarList.model_validate_json(star_list.model_dump_json()).egain == 1.2


def test_egain_per_channel_round_trips():
    # A gain per Bayer channel is accepted and reads as a BayerGains model
    # with a float per channel, written back as a JSON object keyed by the
    # channel names.
    data = StarList.from_examples().model_dump()
    data["egain"] = _example_egain_per_channel()

    star_list = StarList.model_validate(data)

    assert star_list.egain == BayerGains(R=1.1, G=1.2, B=1.3)
    assert (star_list.egain.R, star_list.egain.G, star_list.egain.B) == (1.1, 1.2, 1.3)
    assert json.loads(star_list.model_dump_json())["egain"] == {"R": 1.1, "G": 1.2, "B": 1.3}
    assert StarList.model_validate_json(star_list.model_dump_json()) == star_list


@pytest.mark.parametrize(
    "value",
    [
        pytest.param({"R": 1.1, "G": 1.2, "B": 1.3, "L": 1.0}, id="unknown-channel"),
        pytest.param({"r": 1.1, "g": 1.2, "b": 1.3}, id="lower-case-channels"),
        pytest.param({"R": 1.1, "G": 1.2}, id="missing-channel"),
        pytest.param({}, id="empty"),
        pytest.param({"R": -1.0, "G": 1.2, "B": 1.3}, id="negative-channel-gain"),
        pytest.param({"R": float("inf"), "G": 1.2, "B": 1.3}, id="infinite-channel-gain"),
        pytest.param({"R": float("nan"), "G": 1.2, "B": 1.3}, id="nan-channel-gain"),
        pytest.param({"R": "high", "G": 1.2, "B": 1.3}, id="non-numeric-channel-gain"),
        pytest.param(float("inf"), id="infinite-scalar"),
        pytest.param(float("nan"), id="nan-scalar"),
        pytest.param(-1.0, id="negative-scalar"),
        pytest.param("high", id="non-numeric-scalar"),
    ],
)
def test_egain_rejects_bad_values(value):
    # The per-channel form needs exactly the three channels R, G and B, in
    # that spelling, and neither form admits a negative gain or a non-numeric
    # one. An infinite or NaN gain is rejected too: JSON has neither, so
    # pydantic would write it as null, which does not read back.
    data = StarList.from_examples().model_dump()
    data["egain"] = value

    with pytest.raises(ValidationError):
        StarList.model_validate(data)


def test_egain_schema_offers_both_forms_and_requires_every_channel():
    # The generated schema says egain is a non-negative number or an object
    # with exactly the three channel keys, each a non-negative number.
    schema = StarList.model_json_schema()
    number, per_channel = schema["properties"]["egain"]["anyOf"]
    gains = schema["$defs"]["BayerGains"]

    assert number == {"type": "number", "minimum": 0}
    assert per_channel == {"$ref": "#/$defs/BayerGains"}
    assert gains["required"] == ["R", "G", "B"]
    assert gains["additionalProperties"] is False
    assert all(
        gains["properties"][channel]["type"] == "number"
        and gains["properties"][channel]["minimum"] == 0
        for channel in ("R", "G", "B")
    )


def test_egain_markdown_row_names_both_forms():
    # The markdown table writes egain's type as in the source, not as
    # pydantic's internal Union[Annotated[...]] spelling.
    row = next(
        line for line in StarList.markdown_table().splitlines() if "| egain |" in line
    )

    assert row.split(" | ")[2] == r"float \| BayerGains"


def test_markdown_document_has_a_bayer_gains_section():
    # The human-readable schema documents the per-channel form of egain in
    # its own section, between the star list and the star item, with one row
    # per channel, so a reader can find what the BayerGains type holds.
    headings = [line for line in _generate_markdown().splitlines() if line.startswith("# ")]
    section = _generate_markdown().split("# Bayer Gains")[1].split("# Star Item")[0]

    assert headings == ["# Star List Set", "# Star List", "# Bayer Gains", "# Star Item"]
    assert [line.split(" | ")[1] for line in section.splitlines() if line.startswith("| ")][2:] == [
        "R",
        "G",
        "B",
    ]


def test_photometry_software_is_required_in_a_current_file():
    # A file at the current version must give photometry_software for every
    # star list. It is not upgraded, so without a warning (pytest treats an
    # unexpected SchemaMigrationWarning as an error), and the field is not
    # filled in for it.
    star_list = StarList.from_examples().model_dump(mode="json")
    del star_list["photometry_software"]
    data = {"schema_version": SCHEMA_VERSION, "star_lists": [star_list]}

    assert "photometry_software" in StarList.model_json_schema()["required"]
    with pytest.raises(ValidationError, match="photometry_software"):
        StarListSet.model_validate(data)


@pytest.mark.parametrize("value", [[], [""], ["bandaid 1.2.3", ""]])
def test_photometry_software_rejects_empty_values(value):
    # photometry_software needs at least one entry, and no entry may be empty.
    data = StarList.from_examples().model_dump()
    data["photometry_software"] = value

    with pytest.raises(ValidationError, match="photometry_software"):
        StarList.model_validate(data)


def test_photometry_software_survives_a_table_round_trip():
    # photometry_software is carried in the table metadata, so a star list
    # turned into a table and back keeps it.
    star_list = StarList.from_examples()
    star_list.staritems = [StarItem.from_examples()]

    result = StarList.from_table(star_list.to_table())

    assert result.photometry_software == ["bandaid 1.2.3", "browser-photometry 4.5.6"]
    assert result == star_list


def _count_cells(row):
    """
    Count the cells in one row of a markdown table.

    Parameters
    ----------
    row : str
        A table row, beginning and ending with ``|``.

    Returns
    -------
    int
        The number of cells, counting only pipes not escaped as ``\\|``.
    """
    return len(re.findall(r"(?<!\\)\|", row)) - 1


@pytest.mark.parametrize("klass", [StarItem, StarList, StarListSet])
def test_markdown_table_rows_have_as_many_cells_as_the_header(klass):
    # A pipe written into a cell, such as the one in the type ``int | None``,
    # would add a cell and shift the rest of the row out of its columns.
    header, *rows = klass.markdown_table().splitlines()

    for row in rows:
        assert _count_cells(row) == _count_cells(header), row


def _markdown_row(klass, field_name):
    """
    Find the row of a field in a model's markdown table, split into cells.

    Parameters
    ----------
    klass : type
        A model with a ``markdown_table`` method.
    field_name : str
        The name of the field, as written in the JSON Field column.

    Returns
    -------
    dict
        The row's cells, stripped of surrounding spaces, keyed by the column
        headers. Pipes escaped as ``\\|`` stay inside their cell.
    """
    header, _, *rows = klass.markdown_table().splitlines()

    def split(row):
        """
        Split a markdown table row into its stripped cells.

        Parameters
        ----------
        row : str
            A table row, beginning and ending with ``|``.

        Returns
        -------
        list of str
            The cells, split only at pipes not escaped as ``\\|``.
        """
        return [cell.strip() for cell in re.split(r"(?<!\\)\|", row)[1:-1]]

    columns = split(header)
    for row in rows:
        cells = dict(zip(columns, split(row), strict=True))
        if cells["JSON Field"] == field_name:
            return cells
    raise AssertionError(f"No row for {field_name} in the {klass.__name__} table")


@pytest.mark.parametrize(
    "klass, field_name, type_name, example",
    [
        (
            StarList,
            "photometry_software",
            "list[str]",
            '["bandaid 1.2.3", "browser-photometry 4.5.6"]',
        ),
        (StarList, "absolute_focus", r"float \| None", "1823.0"),
        (StarList, "staritems", "list[StarItem]", "[]"),
        (StarList, "filter", "AAVSOFilters", '"TG"'),
        (StarList, "observer", "str", '"MMU"'),
        (StarListSet, "star_lists", "list[StarList]", "[]"),
    ],
)
def test_markdown_table_renders_type_and_example(klass, field_name, type_name, example):
    # The Type column names a type as the source writes it, with the
    # parameters of a generic and an escaped pipe in a union, and the Examples
    # column holds the example as JSON: strings in double quotes, a list in
    # JSON syntax and an enum member as its string value.
    row = _markdown_row(klass, field_name)

    assert row["Type"] == type_name
    assert row["Examples"] == example


@pytest.mark.parametrize(
    "annotation, expected",
    [
        (Annotated[float, "metadata"], "float"),
        (int | None, "int | None"),
        (Optional[int], "int | None"),  # noqa: UP045 - the typing.Union spelling is the point
        (type(None), "None"),
        (list[str], "list[str]"),
        (list[StarItem], "list[StarItem]"),
        (dict[AAVSOFilters, float], "dict[AAVSOFilters, float]"),
        (Annotated[list[float] | None, "metadata"], "list[float] | None"),
        (AAVSOFilters, "AAVSOFilters"),
    ],
)
def test_type_name(annotation, expected):
    # _type_name drops Annotated, joins a union (either spelling) with " | ",
    # writes NoneType as None, keeps the parameters of a generic, and names an
    # enum or class by its __name__.
    assert _type_name(annotation) == expected


def test_starlist_json():
    # The committed JSON reference file is current with the models.
    # Compared as text, so key order and whitespace must match too.
    json_file = DATA_DIR / "schema_definition.json"

    assert generate_star_list_set_schema() == json_file.read_text()


def test_schema_version_is_embedded():
    # The schema version is a hand-maintained constant, independent of the
    # package version, so generation is deterministic.
    schema = json.loads(generate_star_list_set_schema())
    assert schema["version"] == SCHEMA_VERSION
    assert "schema_version" in schema["required"]
    assert "default" not in schema["properties"]["schema_version"]


def test_schema_pattern_admits_only_released_versions_of_current_generation(
    next_generation_version,
):
    # The schema_version pattern in the generated schema admits only released
    # versions of the current generation, up to the current version, so other
    # tools validating a file against the schema reject what the model rejects.
    # (The model itself upgrades or raises a SchemaVersionError before the
    # pattern is reached; see test_migration.py.)
    schema = json.loads(generate_star_list_set_schema())
    pattern = schema["properties"]["schema_version"]["pattern"]
    major, minor, patch = SCHEMA_VERSION.split(".")

    assert re.search(pattern, SCHEMA_VERSION)
    for bad_version in [
        "banana",
        "0.0.1.dev451+gde1568f",  # legacy dev string
        "0.1.0",  # before versioning began
        f"{major}.{minor}.{int(patch) + 1}",  # newer, same generation
        next_generation_version,
    ]:
        assert not re.search(pattern, bad_version)


def test_make_star_list_from_table_of_items():
    # Make a table with a single star item and turn it into a star list
    table = Table(
        dict(
            x=[12.6],
            y=[23.4],
            ra=[123.6],
            dec=[43.4],
            tot_count=[100.0],
            count_err=[10.1],
            bkgd_count=[4.0],
            peak_count=[100.0],
        )
    )
    # There are two ways to provide the "metadata", i.e. the non-StarItems
    # that are required to create a StarList
    # 1. Provide a dictionary with the metadata

    # Here we generate the dictionary from the example StarList
    meta = StarList.from_examples().model_dump()

    # Get rid of the empty staritems
    del meta["staritems"]

    sl = StarList.from_table(table, meta)
    sl_dict = sl.model_dump()

    # Check the non-star items
    for key in meta:
        assert sl_dict[key] == meta[key]

    # Check a couple of the star item properties
    assert len(sl.staritems) == 1
    assert sl.staritems[0].x == 12.6
    assert sl.staritems[0].bkgd_count == 4.0

    # 2. Provide a Table that has the metadata

    table.meta = meta.copy()
    sl2 = StarList.from_table(table)

    assert sl2 == sl

    # 3. Provide both table metadata and explicit metadata argument.
    #    The explicit metadata should take precedence
    fake_observer = "Ima Fake"
    meta["observer"] = fake_observer
    sl3 = StarList.from_table(table, meta)
    assert sl3.observer == fake_observer

    # Finally, test a couple of cases where errors should be raised.

    # If the table is missing a required column we should get an error
    x_col = table["x"]
    del table["x"]
    with pytest.raises(ValueError, match="Missing columns in table: x"):
        StarList.from_table(table)

    # Put the column back for the next test
    table["x"] = x_col

    # If an item is missing from the required metadata we should get a
    # helpful error.
    missing_item = "exposure"
    del table.meta[missing_item]

    with pytest.raises(ValueError, match=f"Missing keys in metadata: {missing_item}"):
        StarList.from_table(table)


def _single_star_table():
    """
    Build a table of one star item whose metadata holds the example StarList.

    Returns
    -------
    astropy.table.Table
        One row with every `StarItem` column, and every `StarList` field
        except ``staritems`` in its ``meta``, at its example value.
    """
    star_item = StarItem.from_examples().model_dump()
    table = Table({key: [value] for key, value in star_item.items()})
    table.meta = StarList.from_examples().model_dump()
    del table.meta["staritems"]
    return table


@pytest.mark.parametrize(
    "field_name",
    [name for name, field in StarList.model_fields.items() if not field.is_required()],
)
def test_from_table_allows_missing_optional_metadata(field_name):
    # An optional field may be left out of the metadata; the star list then
    # gets the field's default instead of an error about a missing key.
    table = _single_star_table()
    del table.meta[field_name]

    star_list = StarList.from_table(table)

    assert getattr(star_list, field_name) == StarList.model_fields[field_name].default


@pytest.mark.parametrize(
    "field_name",
    [name for name, field in StarList.model_fields.items() if not field.is_required()],
)
def test_from_table_keeps_optional_metadata_that_is_present(field_name):
    # An optional field given in the metadata is kept, not replaced by its
    # default.
    table = _single_star_table()

    star_list = StarList.from_table(table)

    assert getattr(star_list, field_name) == StarList.model_fields[field_name].examples[0]


def _optional_star_item_columns():
    """
    Name the optional fields of `StarItem`.

    Returns
    -------
    list of str
        The fields of `StarItem` that are not required; ``fwhm`` at present.
    """
    return [name for name, field in StarItem.model_fields.items() if not field.is_required()]


@pytest.mark.parametrize("column", _optional_star_item_columns())
def test_from_table_allows_missing_optional_star_item_column(column):
    # A table without the column of an optional star item field is accepted,
    # and every star item reads that field as its default.
    table = _single_star_table()
    del table[column]

    star_list = StarList.from_table(table)

    assert len(star_list.staritems) == 1
    assert getattr(star_list.staritems[0], column) == StarItem.model_fields[column].default


@pytest.mark.parametrize("column", _optional_star_item_columns())
def test_from_table_keeps_optional_star_item_column_that_is_present(column):
    # The column of an optional star item field, when given, is read, not
    # replaced by its default.
    table = _single_star_table()

    star_list = StarList.from_table(table)

    assert getattr(star_list.staritems[0], column) == StarItem.model_fields[column].examples[0]


@pytest.mark.parametrize("missing", ["masked", "nan"])
def test_from_table_reads_missing_optional_star_item_value_as_none(missing):
    # A masked entry, or a NaN, in an optional column is the table's way of
    # saying there is no value; it reads as None rather than failing
    # validation (fwhm must be greater than zero, which NaN is not).
    table = _single_star_table()
    if missing == "masked":
        table["fwhm"] = MaskedColumn([3.4], mask=[True])
    else:
        table["fwhm"] = [np.nan]

    star_list = StarList.from_table(table)

    assert star_list.staritems[0].fwhm is None


def test_from_table_still_requires_the_required_star_item_columns():
    # Making optional columns optional did not loosen the required ones: a
    # table without required star item columns is rejected, naming them in
    # a fixed (sorted) order.
    table = _single_star_table()
    del table["peak_count"]
    del table["dec"]

    with pytest.raises(ValueError, match="Missing columns in table: dec, peak_count$"):
        StarList.from_table(table)


def test_is_missing_treats_a_string_as_a_value():
    # A string cannot be NaN; it is a value, not a gap, so a future optional
    # string column would not have its entries turned into None.
    assert not _is_missing("3.4")
    assert _is_missing(np.nan)
    assert _is_missing(np.ma.masked)


def test_to_table_masks_none_in_an_optional_star_item_column():
    # None in an optional star item field becomes a masked entry of a float
    # column, so that from_table reads it back as None; the values that are
    # present are kept.
    star_list = StarList.from_examples()
    with_fwhm = StarItem.from_examples()
    without_fwhm = StarItem.from_examples()
    without_fwhm.fwhm = None
    star_list.staritems = [with_fwhm, without_fwhm, with_fwhm]

    table = star_list.to_table()

    assert isinstance(table["fwhm"], MaskedColumn)
    assert table["fwhm"].dtype.kind == "f"
    assert list(table["fwhm"].mask) == [False, True, False]
    assert table["fwhm"][0] == 3.4
    assert not isinstance(table["x"], MaskedColumn)

    round_tripped = StarList.from_table(table)

    assert [star.fwhm for star in round_tripped.staritems] == [3.4, None, 3.4]
    assert round_tripped == star_list


def test_empty_star_list_round_trips_through_a_table():
    # A star list with no star items gives a table with every star item
    # column present but empty, which from_table accepts and reads back as a
    # star list with no star items.
    star_list = StarList.from_examples()
    star_list.staritems = []

    table = star_list.to_table()

    assert table.colnames == list(StarItem.model_fields)
    assert len(table) == 0
    assert StarList.from_table(table) == star_list


def test_to_table_keeps_optional_star_item_column_plain_when_no_none():
    # When every star item has a value for an optional field, its column is an
    # ordinary one, as before the field became optional.
    star_list = StarList.from_examples()
    star_list.staritems = [StarItem.from_examples(), StarItem.from_examples()]

    table = star_list.to_table()

    assert not isinstance(table["fwhm"], MaskedColumn)
    assert list(table["fwhm"]) == [3.4, 3.4]


def test_table_round_trip_keeps_egain_per_channel():
    # A per-channel egain goes into the table metadata as plain strings and
    # numbers, and from_table reads it back as the same star list.
    data = StarList.from_examples().model_dump()
    data["egain"] = _example_egain_per_channel()
    data["staritems"] = [StarItem.from_examples()]
    star_list = StarList.model_validate(data)

    table = star_list.to_table()

    assert table.meta["egain"] == {"R": 1.1, "G": 1.2, "B": 1.3}
    assert type(table.meta["egain"]) is dict
    assert StarList.from_table(table) == star_list


@pytest.mark.parametrize("egain", [1.2, {"R": 1.1, "G": 1.2, "B": 1.3}], ids=["scalar", "per-channel"])
def test_table_with_either_egain_is_written_to_ecsv_and_read_back(tmp_path, egain):
    # The table metadata holds plain JSON types only (the filter enum member
    # is a string), so the table can be written to ECSV, which could not be
    # done while the meta held enum members; reading the file back and
    # passing it to from_table gives the same star list, with either egain.
    data = StarList.from_examples().model_dump()
    data["egain"] = egain
    data["staritems"] = [StarItem.from_examples()]
    star_list = StarList.model_validate(data)
    path = tmp_path / "starlist.ecsv"

    star_list.to_table().write(path, format="ascii.ecsv")
    read_back = StarList.from_table(Table.read(path, format="ascii.ecsv"))

    assert read_back == star_list


def test_make_table_from_starlist():
    # Make sure we can make a table from a starlist

    # Make a StarItem and a StarList from the example values we provide
    # in the schema
    staritem = StarItem.from_examples()
    sl = StarList.from_examples()
    sl.staritems = [staritem]

    table = sl.to_table()

    # Check that the table has the same number of rows as the StarItems
    assert len(table) == len(sl.staritems)

    # Check that the non-star item stuff is in the table
    sl_dict = sl.model_dump()

    # We check star items separately
    del sl_dict["staritems"]

    assert table.meta == sl_dict

    # Check that the table has the right cols, and only those columns.
    assert set(table.colnames) == set(StarItem.model_fields.keys())

    # Check the values in the only row in this table
    star_item_dict = staritem.model_dump()

    for key in star_item_dict:
        assert table[key][0] == star_item_dict[key]


def test_generate_starlist_schema_is_valid_json():
    # The single-StarList schema generator should produce parseable JSON
    # whose top-level object describes the StarList model.
    schema = json.loads(generate_starlist_schema())
    assert schema["title"] == "StarList"
    assert "staritems" in schema["properties"]


def test_main_writes_json(tmp_path):
    # main() should write the StarListSet JSON schema, adding the .json suffix.
    out = tmp_path / "schema"
    main(str(out))

    json_path = out.with_suffix(".json")
    assert json_path.exists()
    written = json.loads(json_path.read_text())
    assert json.loads(generate_star_list_set_schema()) == written


def test_main_writes_markdown(tmp_path):
    # main(markdown=True) should write the markdown table, adding the .md suffix.
    out = tmp_path / "schema"
    main(str(out), markdown=True)

    md_path = out.with_suffix(".md")
    assert md_path.exists()
    assert md_path.read_text() == _generate_markdown()


@pytest.mark.parametrize("markdown_flag, suffix", [([], ".json"), (["--markdown"], ".md")])
def test_cli_writes_file(tmp_path, mocker, markdown_flag, suffix):
    # The console-script entry point parses argv and writes the requested format.
    # Fire binds a value following a bool flag to that flag, so the positional
    # filename comes first and --markdown is a trailing standalone flag.
    out = tmp_path / "from_cli"
    mocker.patch("sys.argv", ["aavso-starlist-schema", str(out), *markdown_flag])

    cli()

    expected = out.with_suffix(suffix)
    assert expected.exists()
    assert expected.read_text()
