import pytest

from st_pipeline.schema_definition import StarList, SchemaHeader


@pytest.mark.parametrize("klass", [StarList, SchemaHeader])
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
