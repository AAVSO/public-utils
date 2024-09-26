# JSON Schema generation for AAVSO Smart Telescope starlist

## Generate the schema

To generate the JSON Schema for the AAVSO Smart Telescope starlist do
this in Python:

```python
from st_pipeline.schema_definition import generate_starlist_schema

# Display the schema on the screen
print(generate_starlist_schema())

# Save the schema to a file
with open('starlist_schema.json', 'w') as f:
    f.write(generate_starlist_schema())
```

## Generate a markdown table with the schema information

To generate a markdown table with the schema information do this in Python:

```python
from st_pipeline.schema_definition import StarList, HeaderSchema

# Display the schema for individual starlist entries on the screen
print(StarList.markdown_table())

# Display the schema for the header on the screen
print(HeaderSchema.markdown_table())
```
