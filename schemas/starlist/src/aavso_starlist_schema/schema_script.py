from argparse import ArgumentParser
from pathlib import Path

from pydantic.alias_generators import to_snake

from st_pipeline.schema_definition.schema_generator import (
    StarItem,
    StarList,
    StarListSet,
    generate_star_list_set_schema,
)


def _nice_name(name):
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
    extension = ".md" if markdown else ".json"
    # Make sure the path has the right suffix
    p = Path(filename).with_suffix(extension)

    if markdown:
        content = _generate_markdown()
    else:
        content = generate_star_list_set_schema()

    with p.open("w") as f:
        f.write(content)


if __name__ == "__main__":
    parser = ArgumentParser(
        description=(
            "Generate schema as either JSON or markdown table. "
            "The default output is JSON"
        )
    )
    parser.add_argument(
        "-m", "--markdown",
        action="store_true",
        help="Generate a markdown table of the schema instead of JSON.",
    )
    parser.add_argument(
        "filename",
        type=str,
        action="store",
        default="aavso_star_schema",
        help=(
            "Filename to save the schema to. The extension will be added "
            "automatically based on the type of output selected."
        ),
    )
    args = parser.parse_args()
    main(args.filename, args.markdown)
