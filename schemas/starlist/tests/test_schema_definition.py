import pytest

from astropy.utils.data import get_pkg_data_filename

from st_pipeline.schema_definition import (
    StarListItem, StarList, generate_starlist_schema
)
from st_pipeline.schema_definition.schema_script import _generate_markdown


@pytest.mark.parametrize("klass", [StarListItem, StarList])
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


def test_starlist_markdown_table():
    mdown_file = get_pkg_data_filename(
        "data/schema_definition.md",
        package="st_pipeline.schema_definition"
    )
    with open(mdown_file) as f:
        mdown_file_content = f.read()

    assert _generate_markdown() == mdown_file_content


def test_starlist_json():
    json_file = get_pkg_data_filename(
        "data/schema_definition.json",
        package="st_pipeline.schema_definition"
    )
    with open(json_file) as f:
        json_file_content = f.read()

    assert generate_starlist_schema() == json_file_content
