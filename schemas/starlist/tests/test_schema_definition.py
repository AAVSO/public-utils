import json

import pytest
from astropy.table import Table

from aavso_starlist_schema import (
    DATA_DIR,
    StarItem,
    StarList,
    StarListSet,
    __version__,
    _generate_markdown,
    cli,
    generate_star_list_set_schema,
    generate_starlist_schema,
    main,
)


@pytest.mark.parametrize("klass", [StarItem, StarList, StarListSet])
def test_schema_has_all_require_properties(klass):
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
    mdown_file = DATA_DIR / "schema_definition.md"
    with open(mdown_file) as f:
        mdown_file_content = f.read()

    assert _generate_markdown() == mdown_file_content


def test_starlist_json():
    json_file = DATA_DIR / "schema_definition.json"

    with open(json_file) as f:
        expected_content = json.load(f)

    current_schema = json.loads(generate_star_list_set_schema())
    if "dev" in __version__:
        # The schema contains the version number, which will be different
        # in a source install.
        # If we are in a source install, we want to ignore the version number.
        current_schema["properties"]["schema_version"]["default"] = expected_content["properties"]["schema_version"]["default"]

    assert current_schema == expected_content


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
def test_cli_writes_file(tmp_path, monkeypatch, markdown_flag, suffix):
    # The console-script entry point parses argv and writes the requested format.
    # Fire binds a value following a bool flag to that flag, so the positional
    # filename comes first and --markdown is a trailing standalone flag.
    out = tmp_path / "from_cli"
    monkeypatch.setattr("sys.argv", ["aavso-starlist-schema", str(out), *markdown_flag])

    cli()

    expected = out.with_suffix(suffix)
    assert expected.exists()
    assert expected.read_text()
