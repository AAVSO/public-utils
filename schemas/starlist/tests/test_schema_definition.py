import decimal
import json
import re
from typing import Annotated, Optional

import numpy as np
import pytest
from _helpers import float_fields, flooded, int_fields
from astropy.table import MaskedColumn, Table
from pydantic import ValidationError

from aavso_starlist_schema import (
    _OBS_TIME_PATTERN,
    DATA_DIR,
    SCHEMA_VERSION,
    AAVSOFilters,
    StarItem,
    StarList,
    StarListSet,
    _generate_markdown,
    _is_missing,
    _type_name,
    cli,
    generate_star_list_set_schema,
    generate_starlist_schema,
    write_schema,
)

# The float fields of each model, as the next tests check them; the first test
# below keeps these lists complete.
STAR_ITEM_FLOAT_FIELDS = [
    "x", "y", "ra", "dec", "tot_count", "count_err", "bkgd_count", "peak_count", "fwhm",
]
STAR_LIST_FLOAT_FIELDS = [
    "site_lat", "site_lon", "site_elev", "exposure", "egain", "bias", "fwhm", "absolute_focus",
]

# The int fields of a star list (a star item has none), kept complete by
# test_int_fields_are_all_listed.
STAR_LIST_INT_FIELDS = ["width", "height", "stack", "adc_depth", "largest_usable_adu_value"]

# Every numeric field, as (model, field name), for the tests that a number is
# required.
NUMERIC_FIELDS = (
    [(StarItem, name) for name in STAR_ITEM_FLOAT_FIELDS]
    + [(StarList, name) for name in STAR_LIST_FLOAT_FIELDS + STAR_LIST_INT_FIELDS]
)

# The maximum length, in characters, of each capped string field of a star list.
STRING_LIMITS = {
    "obs_time": 63, "observer": 7, "block_filter": 31, "tel_manufac": 31, "tel_model": 31,
    "tel_firmware": 31, "refframe": 15,
}

# The maximum of each int field of a star list that has one.
INT_MAXIMA = {
    "width": 32767, "height": 32767, "stack": 32767, "adc_depth": 64,
    "largest_usable_adu_value": 2_147_483_647,
}


def _string_of_length(field_name, length):
    """
    Build a string of a given length that is otherwise valid for a field.

    Parameters
    ----------
    field_name : str
        A string field of `StarList`, a key of ``STRING_LIMITS``.
    length : int
        The number of characters wanted.

    Returns
    -------
    str
        For ``observer``, uppercase letters; for any other field, ``"x"``
        repeated (for ``obs_time`` that is not a date and time, but a string
        over the length limit is refused for its length first).
    """
    if field_name == "observer":
        return "X" * length
    return "x" * length


def _one_star_set_data():
    """
    Build raw data for a current star list set of one star list of one star item.

    Returns
    -------
    dict
        The example `StarListSet`, with one example `StarList` holding one
        example `StarItem`, dumped to JSON-compatible types.
    """
    star_list = StarList.from_examples()
    star_list.staritems = [StarItem.from_examples()]
    data = StarListSet.from_examples().model_dump(mode="json")
    data["star_lists"] = [star_list.model_dump(mode="json")]
    return data


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


@pytest.mark.parametrize(
    "klass, expected",
    [
        pytest.param(StarItem, STAR_ITEM_FLOAT_FIELDS, id="StarItem"),
        pytest.param(StarList, STAR_LIST_FLOAT_FIELDS, id="StarList"),
    ],
)
def test_float_fields_are_all_listed(klass, expected):
    # The lists of float fields the next tests check are every field whose
    # type is float, optional or not, so a new float field cannot escape them.
    assert list(float_fields(klass)) == expected


def test_int_fields_are_all_listed():
    # The int fields the tests below check are every field typed int, optional
    # or not; a star item has none.
    assert list(int_fields(StarList)) == STAR_LIST_INT_FIELDS
    assert int_fields(StarItem) == ()


@pytest.mark.parametrize(
    "value, kind",
    [
        (True, "a boolean"),
        (np.bool_(False), "a boolean"),
        ("1000", "a string"),
        (b"1", "bytes"),
        (np.array("5"), "a string"),
        (np.array(True), "a boolean"),
        (np.array(b"5"), "bytes"),
        (np.void(b"5"), "bytes"),
    ],
    ids=["bool", "numpy-bool", "str", "bytes", "str-array", "bool-array", "bytes-array",
         "void"],
)
@pytest.mark.parametrize("klass, field_name", NUMERIC_FIELDS)
def test_numeric_field_rejects_a_boolean_string_or_bytes(klass, field_name, value, kind):
    # JSON Schema's number and integer types refuse true and "1000", which
    # pydantic's lax mode would read as 1 and 1000; every numeric field refuses
    # them too, with an error of its own type located at the field.
    data = klass.from_examples().model_dump()
    data[field_name] = value

    with pytest.raises(ValidationError) as excinfo:
        klass.model_validate(data)

    error = excinfo.value.errors()[0]
    assert error["type"] == "number_type"
    assert error["msg"] == f"Input should be a number, not {kind}"
    assert error["loc"] == (field_name,)


@pytest.mark.parametrize(
    "value",
    [3, np.float64(3.0), np.int64(3), decimal.Decimal("3")],
    ids=["int", "numpy-float", "numpy-int", "decimal"],
)
@pytest.mark.parametrize(
    "klass, field_name",
    [(klass, name) for klass, name in NUMERIC_FIELDS if name not in STAR_LIST_INT_FIELDS],
)
def test_float_field_still_accepts_other_numbers(klass, field_name, value):
    # Refusing booleans and strings leaves the rest of lax mode alone: an int,
    # a numpy scalar or a Decimal in a float field is read as the float.
    data = klass.from_examples().model_dump()
    data[field_name] = value

    result = getattr(klass.model_validate(data), field_name)

    assert result == 3.0
    assert type(result) is float


@pytest.mark.parametrize(
    "value",
    [3.0, np.float64(3.0), np.int64(3), decimal.Decimal("3")],
    ids=["float", "numpy-float", "numpy-int", "decimal"],
)
@pytest.mark.parametrize("field_name", STAR_LIST_INT_FIELDS)
def test_int_field_still_accepts_other_whole_numbers(field_name, value):
    # JSON Schema's integer type accepts 3.0, so an int field reads it, and a
    # numpy scalar or a Decimal, as the int 3, as before 1.0.0.
    data = StarList.from_examples().model_dump()
    data[field_name] = value

    result = getattr(StarList.model_validate(data), field_name)

    assert result == 3
    assert type(result) is int


@pytest.mark.parametrize("klass, field_name", NUMERIC_FIELDS)
def test_numeric_field_schema_is_a_plain_number(klass, field_name):
    # The check for booleans and strings runs only in Python: the schema of a
    # numeric field is still a plain number or integer, with no trace of it.
    schema = klass.model_json_schema()["properties"][field_name]
    types = [schema] if "type" in schema else schema["anyOf"]

    assert {entry["type"] for entry in types} <= {"number", "integer", "null"}
    assert set(schema) <= {
        "type", "anyOf", "default", "title", "description", "examples", "unit",
        "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
    }


@pytest.mark.parametrize(
    "value", [float("inf"), float("-inf"), float("nan")], ids=["inf", "-inf", "nan"]
)
@pytest.mark.parametrize(
    "klass, field_name",
    [(StarItem, name) for name in STAR_ITEM_FLOAT_FIELDS]
    + [(StarList, name) for name in STAR_LIST_FLOAT_FIELDS],
)
def test_float_field_rejects_infinity_and_nan(klass, field_name, value):
    # JSON has no infinity or NaN, so pydantic would write either as null (a
    # valid value for an optional field, meaning "not available", and an
    # invalid file otherwise); every float field rejects them, naming itself.
    data = klass.from_examples().model_dump()
    data[field_name] = value

    with pytest.raises(ValidationError, match="finite number") as excinfo:
        klass.model_validate(data)

    assert excinfo.value.errors()[0]["loc"] == (field_name,)


@pytest.mark.parametrize("value", [float("inf"), float("nan")], ids=["inf", "nan"])
def test_json_text_with_infinity_or_nan_token_is_rejected(value):
    # JSON text with the non-standard Infinity or NaN tokens, which Python's
    # json module writes, is rejected in the two float fields that have no
    # other constraint, so nothing but the finiteness check stops them.
    data = _one_star_set_data()
    data["star_lists"][0]["site_elev"] = value
    data["star_lists"][0]["staritems"][0]["bkgd_count"] = value
    text = json.dumps(data)

    with pytest.raises(ValidationError, match="finite number") as excinfo:
        StarListSet.model_validate_json(text)

    errors = excinfo.value.errors()
    assert {tuple(error["loc"]) for error in errors} == {
        ("star_lists", 0, "site_elev"),
        ("star_lists", 0, "staritems", 0, "bkgd_count"),
    }
    assert all(error["type"] == "finite_number" for error in errors)


@pytest.mark.parametrize(
    "level, loc",
    [
        ("set", ("bogus",)),
        ("star_list", ("star_lists", 0, "bogus")),
        ("star_item", ("star_lists", 0, "staritems", 0, "bogus")),
    ],
    ids=["set", "star_list", "star_item"],
)
def test_unknown_key_is_rejected_at_each_level(level, loc):
    # A key the schema does not define is rejected at every level of the
    # file, and the error says where it is.
    data = _one_star_set_data()
    target = {
        "set": data,
        "star_list": data["star_lists"][0],
        "star_item": data["star_lists"][0]["staritems"][0],
    }[level]
    target["bogus"] = 1

    with pytest.raises(ValidationError) as excinfo:
        StarListSet.model_validate(data)

    error = excinfo.value.errors()[0]
    assert error["type"] == "extra_forbidden"
    assert error["loc"] == loc
    assert "bogus" in str(excinfo.value)
    assert "Extra inputs are not permitted" in str(excinfo.value)


def test_misspelled_optional_field_is_reported_not_dropped():
    # A misspelled optional field used to be ignored, so the field silently
    # read as None; now the misspelling is reported.
    data = StarList.from_examples().model_dump()
    data["absolute_focs"] = data.pop("absolute_focus")

    with pytest.raises(ValidationError, match="absolute_focs"):
        StarList.model_validate(data)


def test_misspelled_required_field_is_reported_with_the_missing_field():
    # A misspelled required field is reported twice: the correct name as
    # missing and the misspelling as an unknown key, so the reader sees both
    # halves of the mistake.
    data = _one_star_set_data()
    data["star_lists"][0]["observr"] = data["star_lists"][0].pop("observer")

    with pytest.raises(ValidationError) as excinfo:
        StarListSet.model_validate(data)

    assert {(error["type"], error["loc"]) for error in excinfo.value.errors()} == {
        ("missing", ("star_lists", 0, "observer")),
        ("extra_forbidden", ("star_lists", 0, "observr")),
    }


def test_schema_forbids_additional_properties():
    # The generated schema forbids keys it does not define on the star list
    # set, each star list and each star item.
    schema = json.loads(generate_star_list_set_schema())

    assert schema["additionalProperties"] is False
    assert schema["$defs"]["StarList"]["additionalProperties"] is False
    assert schema["$defs"]["StarItem"]["additionalProperties"] is False


@pytest.mark.parametrize("value", [0, -1])
def test_stack_rejects_values_below_one(value):
    # A stack counts the images in the image, so it is at least 1; 0 and
    # negative counts are rejected, and the error is located at the field.
    data = StarList.from_examples().model_dump()
    data["stack"] = value

    with pytest.raises(ValidationError, match="greater than or equal to 1") as excinfo:
        StarList.model_validate(data)

    error = excinfo.value.errors()[0]
    assert error["type"] == "greater_than_equal"
    assert error["loc"] == ("stack",)


@pytest.mark.parametrize("value", [1, None], ids=["one", "none"])
def test_stack_accepts_one_or_none(value):
    # 1, the boundary, is accepted and kept; so is None, "not applicable".
    data = StarList.from_examples().model_dump()
    data["stack"] = value

    assert StarList.model_validate(data).stack == value


def test_stack_schema_has_minimum_one_and_maximum():
    # The generated schema says stack is an integer of at least 1, and at
    # most its sanity ceiling, or null.
    schema = StarList.model_json_schema()

    assert schema["properties"]["stack"]["anyOf"] == [
        {"type": "integer", "minimum": 1, "maximum": 32767},
        {"type": "null"},
    ]


def test_bias_is_optional():
    # bias may be left out of a star list, which then reads it as None; it is
    # not merely nullable but also absent from "required".
    data = StarList.from_examples().model_dump()
    del data["bias"]

    assert StarList.model_validate(data).bias is None
    assert "bias" not in StarList.model_json_schema()["required"]


def test_bias_is_kept_when_present():
    # A value given for bias survives a JSON round trip.
    star_list = StarList.from_examples()

    assert star_list.bias == 1000.0
    assert StarList.model_validate_json(star_list.model_dump_json()).bias == 1000.0


def test_bias_accepts_null():
    # An explicit null is the same as leaving bias out: not available.
    data = StarList.from_examples().model_dump()
    data["bias"] = None

    assert StarList.model_validate(data).bias is None


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(0.0, id="zero"),
        pytest.param(1000.0, id="positive"),
    ],
)
def test_bias_accepts_non_negative_values(value):
    # A non-negative bias is accepted and kept. 0 is the boundary of ge=0 and
    # the value a frame whose bias or offset has already been subtracted gives.
    data = StarList.from_examples().model_dump()
    data["bias"] = value

    assert StarList.model_validate(data).bias == value


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(-1.0, id="negative"),
        pytest.param(float("inf"), id="infinite"),
        pytest.param(float("nan"), id="nan"),
        pytest.param("high", id="non-numeric"),
    ],
)
def test_bias_rejects_bad_values(value):
    # A negative bias, or a non-numeric one, is rejected. So are infinity and
    # NaN: JSON has neither, so pydantic would write either as null, which is
    # a valid bias meaning "not available"; the value would be lost with no
    # error.
    data = StarList.from_examples().model_dump()
    data["bias"] = value

    with pytest.raises(ValidationError, match="bias"):
        StarList.model_validate(data)


def test_bias_schema_is_an_optional_nullable_non_negative_number():
    # The generated schema says bias is a non-negative number or null, with
    # null as its default, and does not list it among the required properties.
    schema = StarList.model_json_schema()

    assert schema["properties"]["bias"]["anyOf"] == [
        {"type": "number", "minimum": 0},
        {"type": "null"},
    ]
    assert schema["properties"]["bias"]["default"] is None
    assert schema["properties"]["bias"]["unit"] == "adu"
    assert "bias" not in schema["required"]


def test_bias_markdown_row():
    # The markdown table writes bias with its type as in the source, its
    # unit and its example.
    row = next(line for line in StarList.markdown_table().splitlines() if "| bias |" in line)
    cells = row.split(" | ")

    assert cells[0] == "| Bias Level"
    assert cells[2] == r"float \| None"
    assert cells[3] == "adu"
    assert cells[5] == "1000.0 |"


@pytest.mark.parametrize(
    "klass, field_name",
    [
        pytest.param(klass, field_name, id=f"{klass.__name__}-{field_name}")
        for klass, field_name in [
            (StarItem, "tot_count"),
            (StarItem, "count_err"),
            (StarItem, "bkgd_count"),
            (StarItem, "peak_count"),
            (StarList, "bias"),
            (StarList, "largest_usable_adu_value"),
        ]
    ],
)
def test_levels_in_adu_share_one_unit(klass, field_name):
    # Every field that holds a count or a level in adu states the same unit,
    # plain "adu", so that bias and bkgd_count, which are compared, agree.
    # bkgd_count is per pixel; its description says so in words.
    assert klass.model_fields[field_name].json_schema_extra["unit"] == "adu"


def test_bkgd_count_markdown_row():
    # The markdown table writes bkgd_count with the unit adu and a
    # description that says the level is per pixel.
    row = _markdown_row(StarItem, "bkgd_count")

    assert row["Unit"] == "adu"
    assert row["Description"] == "Background count level per pixel in the vicinity of the star"


@pytest.mark.parametrize("bias", [1000.0, 0.0, None], ids=["number", "zero", "none"])
def test_bias_survives_a_table_round_trip(bias):
    # bias is carried in the table metadata, a number or None, so a star list
    # turned into a table and back in memory keeps it. The metadata holds
    # plain JSON types, so the filter enum member is stored as its string
    # value, which is what lets the table be written to ECSV.
    data = StarList.from_examples().model_dump()
    data["bias"] = bias
    data["staritems"] = [StarItem.from_examples()]
    star_list = StarList.model_validate(data)

    table = star_list.to_table()

    assert table.meta["bias"] == bias
    assert type(table.meta["filter"]) is str
    assert StarList.from_table(table) == star_list


@pytest.mark.parametrize("bias", [1000.0, None], ids=["number", "none"])
def test_table_with_either_bias_is_written_to_ecsv_and_read_back(tmp_path, bias):
    # A table whose metadata holds a bias, a number or None, is written to
    # ECSV, where None appears as null, and read back into the same star list.
    data = StarList.from_examples().model_dump()
    data["bias"] = bias
    data["staritems"] = [StarItem.from_examples()]
    star_list = StarList.model_validate(data)
    path = tmp_path / "starlist.ecsv"

    star_list.to_table().write(path, format="ascii.ecsv")
    read_back = StarList.from_table(Table.read(path, format="ascii.ecsv"))

    assert f"bias: {'null' if bias is None else bias}" in path.read_text()
    assert read_back.bias == bias
    assert read_back == star_list


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


# obs_time is left out: its pattern allows at most 32 characters, so its
# maxLength of 63 cannot be reached by a valid value.
@pytest.mark.parametrize(
    "field_name,limit",
    [(name, limit) for name, limit in STRING_LIMITS.items() if name != "obs_time"],
    ids=[name for name in STRING_LIMITS if name != "obs_time"],
)
def test_string_field_accepts_its_maximum_length(field_name, limit):
    # A string exactly as long as its field allows is accepted and kept.
    data = StarList.from_examples().model_dump()
    data[field_name] = _string_of_length(field_name, limit)

    assert getattr(StarList.model_validate(data), field_name) == data[field_name]


@pytest.mark.parametrize(
    "field_name,limit", list(STRING_LIMITS.items()), ids=list(STRING_LIMITS)
)
def test_string_field_rejects_one_character_over_its_maximum(field_name, limit):
    # One character over the limit is rejected, naming the field and the limit.
    data = StarList.from_examples().model_dump()
    data[field_name] = _string_of_length(field_name, limit + 1)

    with pytest.raises(ValidationError, match=f"at most {limit} characters") as excinfo:
        StarList.model_validate(data)

    assert excinfo.value.errors()[0]["loc"] == (field_name,)


@pytest.mark.parametrize(
    "field_name,limit", list(STRING_LIMITS.items()), ids=list(STRING_LIMITS)
)
def test_string_limits_are_in_the_schema(field_name, limit):
    # Each string limit appears in the generated schema as maxLength.
    schema = StarList.model_json_schema()

    assert schema["properties"][field_name]["maxLength"] == limit


def test_photometry_software_entry_accepts_its_maximum_length():
    # A photometry_software entry of exactly 63 characters is accepted and
    # kept, and the schema states the limit on the list's items.
    data = StarList.from_examples().model_dump()
    data["photometry_software"] = ["x" * 63]

    assert StarList.model_validate(data).photometry_software == ["x" * 63]
    schema = StarList.model_json_schema()
    assert schema["properties"]["photometry_software"]["items"]["maxLength"] == 63


def test_photometry_software_entry_rejects_one_character_over():
    # A photometry_software entry of 64 characters is rejected, and the
    # error points at the entry.
    data = StarList.from_examples().model_dump()
    data["photometry_software"] = ["x" * 64]

    with pytest.raises(ValidationError, match="at most 63") as excinfo:
        StarList.model_validate(data)

    assert excinfo.value.errors()[0]["loc"] == ("photometry_software", 0)


def test_staritems_accepts_the_maximum_number():
    # A star list of exactly 100000 star items, the cap, is accepted.
    data = StarList.from_examples().model_dump()
    data["staritems"] = [StarItem.from_examples().model_dump()] * 100_000

    assert len(StarList.model_validate(data).staritems) == 100_000


def test_staritems_rejects_one_over_the_maximum():
    # One star item over the cap is rejected, and the schema states the cap
    # as maxItems.
    data = StarList.from_examples().model_dump()
    data["staritems"] = [StarItem.from_examples().model_dump()] * 100_001

    with pytest.raises(ValidationError, match="at most 100000 items"):
        StarList.model_validate(data)
    schema = StarList.model_json_schema()
    assert schema["properties"]["staritems"]["maxItems"] == 100_000


@pytest.mark.parametrize("value", ["MMU", "AB", "ABCDEFG"])
def test_observer_accepts_a_code_of_2_to_7_uppercase_letters(value):
    # An observer code is 2 to 7 uppercase letters, as AAVSO issues it.
    data = StarList.from_examples().model_dump()
    data["observer"] = value

    assert StarList.model_validate(data).observer == value


@pytest.mark.parametrize(
    "value",
    ["mmu", "Mmu", "MM1", "M", "ABCDEFGH", "", "M U", "ÄBC"],
    ids=["lowercase", "mixed-case", "digit", "one-letter", "eight-letters", "empty", "space",
         "non-ascii"],
)
def test_observer_rejects_anything_but_2_to_7_uppercase_letters(value):
    # Lowercase, digits, other characters and a code too short or too long are
    # all refused, naming the field.
    data = StarList.from_examples().model_dump()
    data["observer"] = value

    with pytest.raises(ValidationError) as excinfo:
        StarList.model_validate(data)

    assert excinfo.value.errors()[0]["loc"] == ("observer",)


@pytest.mark.parametrize(
    "value",
    [
        "2021-06-15T03:45:00",
        "2021-06-15T03:45:00Z",
        "2021-06-15 03:45:00",
        "2021-06-15T03:45:00.123456",
        "2021-06-15T03:45:00+05:30",
        "2021-06-15T03:45:00-07:00",
        "2020-02-29T00:00:00",
    ],
    ids=["plain", "z", "space", "microseconds", "plus-offset", "minus-offset", "leap-day"],
)
def test_obs_time_accepts_the_1_0_0_form(value):
    # obs_time of the 1.0.0 form is accepted and kept as the file's text.
    data = StarList.from_examples().model_dump()
    data["obs_time"] = value

    assert StarList.model_validate(data).obs_time == value


@pytest.mark.parametrize(
    "value",
    [
        "2021-06-15",
        "2021-06-15T03",
        "20210615T034500",
        "2021-W24-2T03:45:00",
        "2021-06-15t03:45:00",
        "2021-06-15T03:45:00z",
        "2021-06-15T03:45:00.1234567",
        "2021-06-15T03:45:00+0530",
        "2021-06-15x03:45:00",
        "2021-13-01T00:00:00",
        "2021-06-15T24:00:00",
        "not a time",
        "2021-06-15T03:45:00+00:60",
        "2021-06-15T03:45:00+00:99",
        "2021-06-15T03:45:60",
        "2021-02-30T00:00:00",
        "2021-02-29T00:00:00",
        "2021-00-10T00:00:00",
    ],
    ids=["date-only", "hour-only", "basic", "week-date", "lowercase-t", "lowercase-z",
         "seven-digit-fraction", "offset-without-colon", "other-separator", "month-13",
         "hour-24", "text", "offset-minute-60", "offset-minute-99", "second-60",
         "february-30", "february-29-non-leap", "month-00"],
)
def test_obs_time_rejects_anything_but_the_1_0_0_form(value):
    # A string not of the form YYYY-MM-DDTHH:MM:SS[.ffffff][Z|+HH:MM] is
    # refused by the pattern, and one of that form that is not a real date
    # and time (month 13, hour 24) by the check that follows it; either way
    # the error is located at the field.
    data = StarList.from_examples().model_dump()
    data["obs_time"] = value

    with pytest.raises(ValidationError) as excinfo:
        StarList.model_validate(data)

    error = excinfo.value.errors()[0]
    assert error["loc"] == ("obs_time",)
    assert error["type"] in ("string_pattern_mismatch", "value_error")


def test_obs_time_schema_has_a_pattern_and_no_format():
    # The JSON schema states the same form the model checks, as a pattern,
    # and no format: "date-time" would mean RFC 3339, which requires an
    # offset and allows a lowercase t, unlike the model.
    schema = StarList.model_json_schema()["properties"]["obs_time"]

    assert schema["pattern"] == _OBS_TIME_PATTERN
    assert schema["maxLength"] == 63
    assert "format" not in schema
    assert schema["scale"] == "UTC"


@pytest.mark.parametrize("field_name, limit", INT_MAXIMA.items(), ids=list(INT_MAXIMA))
def test_int_field_accepts_its_maximum(field_name, limit):
    # An int field at its sanity ceiling is accepted and kept.
    data = StarList.from_examples().model_dump()
    data[field_name] = limit

    assert getattr(StarList.model_validate(data), field_name) == limit


@pytest.mark.parametrize("field_name, limit", INT_MAXIMA.items(), ids=list(INT_MAXIMA))
def test_int_field_rejects_one_over_its_maximum(field_name, limit):
    # One over the ceiling is refused, naming the field and the limit.
    data = StarList.from_examples().model_dump()
    data[field_name] = limit + 1

    with pytest.raises(ValidationError, match=f"less than or equal to {limit}") as excinfo:
        StarList.model_validate(data)

    assert excinfo.value.errors()[0]["loc"] == (field_name,)


def test_star_lists_accepts_the_maximum_number():
    # A starlist set of exactly 1000 star lists, the cap, is accepted.
    data = flooded(_one_star_set_data(), ("star_lists",), _one_star_set_data()["star_lists"][0], 1000)

    assert len(StarListSet.model_validate(data).star_lists) == 1000


def test_star_lists_rejects_one_over_the_maximum():
    # One star list over the cap is refused at the field.
    data = flooded(_one_star_set_data(), ("star_lists",), _one_star_set_data()["star_lists"][0], 1001)

    with pytest.raises(ValidationError, match="at most 1000 items") as excinfo:
        StarListSet.model_validate(data)

    assert excinfo.value.errors()[0]["loc"] == ("star_lists",)


def test_photometry_software_accepts_the_maximum_number():
    # A star list naming exactly 100 software packages, the cap, is accepted.
    data = StarList.from_examples().model_dump()
    data["photometry_software"] = ["bandaid 1.2.3"] * 100

    assert len(StarList.model_validate(data).photometry_software) == 100


def test_photometry_software_rejects_one_over_the_maximum():
    # One software entry over the cap is refused at the field.
    data = StarList.from_examples().model_dump()
    data["photometry_software"] = ["bandaid 1.2.3"] * 101

    with pytest.raises(ValidationError, match="at most 100 items") as excinfo:
        StarList.model_validate(data)

    assert excinfo.value.errors()[0]["loc"] == ("photometry_software",)


def test_reference_schema_states_the_1_0_0_limits():
    # The committed schema_definition.json, which other languages validate
    # against, carries each limit 1.0.0 added.
    schema = json.loads((DATA_DIR / "schema_definition.json").read_text())
    star_list = schema["$defs"]["StarList"]["properties"]

    assert schema["properties"]["star_lists"]["maxItems"] == 1000
    assert star_list["photometry_software"]["maxItems"] == 100
    assert star_list["observer"]["pattern"] == "^[A-Z]{2,7}$"
    assert star_list["tel_manufac"]["maxLength"] == 31
    assert star_list["tel_model"]["maxLength"] == 31
    assert star_list["obs_time"]["pattern"] == _OBS_TIME_PATTERN
    assert "format" not in star_list["obs_time"]
    assert star_list["obs_time"]["maxLength"] == 63
    for name, limit in INT_MAXIMA.items():
        types = star_list[name].get("anyOf", [star_list[name]])
        assert types[0]["maximum"] == limit, name


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
        (StarList, "width", "int", "2048"),
        (StarList, "stack", r"int \| None", "3"),
        (StarList, "bias", r"float \| None", "1000.0"),
        (StarItem, "x", "float", "1206.78"),
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
    fake_observer = "FAKE"
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


def test_from_table_ignores_unknown_table_metadata_keys():
    # A table's metadata may carry anything, such as FITS header cards;
    # from_table uses only the StarList fields from it and ignores the rest.
    expected = StarList.from_table(_single_star_table())
    table = _single_star_table()
    table.meta["EXPTIME"] = 30.0
    table.meta["COMMENT"] = "fits card"

    assert StarList.from_table(table) == expected


def test_from_table_rejects_unknown_explicit_metadata_keys():
    # The metadata argument is the caller's own, so it is not filtered: a
    # misspelled key there is reported rather than ignored.
    with pytest.raises(ValidationError, match="observr"):
        StarList.from_table(_single_star_table(), metadata={"observr": "MMU"})


def test_from_table_rejects_nan_in_a_required_star_item_column():
    # Only an optional column reads NaN as None; in a required column NaN is
    # a value, and every float field rejects it.
    table = _single_star_table()
    table["bkgd_count"] = [np.nan]

    with pytest.raises(ValidationError, match="finite number"):
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


def test_write_schema_writes_json(tmp_path):
    # write_schema() should write the StarListSet JSON schema, adding the
    # .json suffix.
    out = tmp_path / "schema"
    write_schema(str(out))

    json_path = out.with_suffix(".json")
    assert json_path.exists()
    written = json.loads(json_path.read_text())
    assert json.loads(generate_star_list_set_schema()) == written


def test_write_schema_writes_markdown(tmp_path):
    # write_schema(markdown=True) should write the markdown table, adding the
    # .md suffix.
    out = tmp_path / "schema"
    write_schema(str(out), markdown=True)

    md_path = out.with_suffix(".md")
    assert md_path.exists()
    assert md_path.read_text() == _generate_markdown()


@pytest.mark.parametrize("markdown", [False, True])
def test_write_schema_refuses_to_overwrite(tmp_path, markdown, capsys):
    # An existing file at the output path is left exactly as it was, the
    # refusal names the file on stderr and the exit status is 1. This is the
    # guard against `write-schema starlists.json` typed in place of `validate`.
    out = tmp_path / ("schema.md" if markdown else "schema.json")
    out.write_text("keep me")

    with pytest.raises(SystemExit) as info:
        write_schema(str(out), markdown=markdown)

    assert info.value.code == 1
    assert out.read_text() == "keep me"
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == f"{out}: exists; not overwriting it (use --force to replace it)\n"


def test_write_schema_guard_checks_the_path_after_the_suffix_is_replaced(tmp_path):
    # The guard applies to the file that would actually be written, not to
    # the path as given.
    out = tmp_path / "schema"
    out.with_suffix(".json").write_text("keep me")

    with pytest.raises(SystemExit):
        write_schema(str(out))
    assert out.with_suffix(".json").read_text() == "keep me"


def test_write_schema_force_overwrites(tmp_path):
    out = tmp_path / "schema.json"
    out.write_text("replace me")

    write_schema(str(out), force=True)

    assert json.loads(out.read_text()) == json.loads(generate_star_list_set_schema())


def test_write_schema_refuses_a_dangling_symlink(tmp_path, capsys):
    # A symlink to nothing is not "absent": writing would create the target
    # behind the user's back. It is refused like an existing file, and the
    # target stays missing.
    out = tmp_path / "schema.json"
    target = tmp_path / "nowhere"
    out.symlink_to(target)

    with pytest.raises(SystemExit) as info:
        write_schema(str(out))

    assert info.value.code == 1
    assert capsys.readouterr().err == (
        f"{out}: exists; not overwriting it (use --force to replace it)\n"
    )
    assert out.is_symlink()
    assert not target.exists()
    assert sorted(tmp_path.iterdir()) == [out]


def test_write_schema_directory_at_the_output_path_exits_1(tmp_path, capsys):
    # --force cannot replace a directory; the failure is reported, not a
    # traceback, and the directory is untouched.
    out = tmp_path / "schema.json"
    out.mkdir()

    with pytest.raises(SystemExit) as info:
        write_schema(str(out), force=True)

    assert info.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith(f"{out}: cannot write the file: ")
    assert out.is_dir()
    assert list(out.iterdir()) == []


def test_write_schema_missing_parent_directory_exits_1(tmp_path, capsys):
    # The output directory is not created; a missing one is reported.
    out = tmp_path / "missing" / "schema"

    with pytest.raises(SystemExit) as info:
        write_schema(str(out))

    assert info.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith(f"{out.with_suffix('.json')}: cannot write the file: ")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("markdown_flag, suffix", [([], ".json"), (["--markdown"], ".md")])
def test_cli_write_schema_writes_file(tmp_path, mocker, markdown_flag, suffix):
    # The console-script entry point parses the write-schema command and
    # writes the requested format. Fire binds a value following a bool flag
    # to that flag, so the positional path comes first and --markdown is a
    # trailing standalone flag.
    out = tmp_path / "from_cli"
    mocker.patch("sys.argv", ["aavso-starlist-schema", "write-schema", str(out), *markdown_flag])

    cli()

    expected = out.with_suffix(suffix)
    assert expected.exists()
    assert expected.read_text()


def test_cli_write_schema_path_that_is_a_literal(tmp_path, monkeypatch):
    # Fire turns an argument that looks like a Python literal into one, so a
    # path named 2024 arrives as an int; the file is still 2024.json.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["aavso-starlist-schema", "write-schema", "2024"])

    cli()

    assert list(tmp_path.iterdir()) == [tmp_path / "2024.json"]
    written = json.loads((tmp_path / "2024.json").read_text())
    assert written == json.loads(generate_star_list_set_schema())


def test_cli_write_schema_refuses_then_forces(tmp_path, mocker, capsys):
    # Without --force an existing file survives and the shell sees status 1;
    # with --force it is replaced.
    out = tmp_path / "from_cli.json"
    out.write_text("keep me")

    mocker.patch("sys.argv", ["aavso-starlist-schema", "write-schema", str(out)])
    with pytest.raises(SystemExit) as info:
        cli()
    assert info.value.code == 1
    assert out.read_text() == "keep me"
    assert "not overwriting" in capsys.readouterr().err

    mocker.patch("sys.argv", ["aavso-starlist-schema", "write-schema", str(out), "--force"])
    cli()
    assert json.loads(out.read_text()) == json.loads(generate_star_list_set_schema())


def test_cli_bare_path_is_not_a_command(tmp_path, mocker):
    # The old form, a path with no command, must not write anything: Fire
    # rejects an unknown command with status 2 and the file is untouched.
    path = tmp_path / "starlists.json"
    path.write_text("keep me")

    mocker.patch("sys.argv", ["aavso-starlist-schema", str(path)])
    with pytest.raises(SystemExit) as info:
        cli()

    assert info.value.code == 2
    assert path.read_text() == "keep me"
    assert list(tmp_path.iterdir()) == [path]


# Each list field of the current models, with the path to it in a star list
# set and an entry that is not valid for it.
BAD_LIST_ENTRIES = [
    pytest.param(("star_lists",), {}, id="star_lists"),
    pytest.param(("star_lists", 0, "staritems"), {}, id="staritems"),
    pytest.param(("star_lists", 0, "photometry_software"), 1, id="photometry_software"),
]


@pytest.mark.parametrize("path, entry", BAD_LIST_ENTRIES)
def test_validation_stops_at_the_first_bad_entry_of_a_list_field(path, entry):
    # A list field full of bad entries reports only its first one (fail_fast),
    # so a malformed file with many entries gives a short error, as cheaply
    # as one with a single bad entry.
    with pytest.raises(ValidationError) as one:
        StarListSet.model_validate(flooded(_one_star_set_data(), path, entry, 1))
    with pytest.raises(ValidationError) as many:
        StarListSet.model_validate(flooded(_one_star_set_data(), path, entry, 10_000))

    assert many.value.error_count() == one.value.error_count()
    assert {error["loc"][: len(path) + 1] for error in many.value.errors()} == {(*path, 0)}
