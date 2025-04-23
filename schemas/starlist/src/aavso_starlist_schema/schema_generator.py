import json
from collections import defaultdict
from typing import Annotated

from astropy.table import Table
from pydantic import BaseModel, Field

from .. import __version__
from .passband_names import AAVSOFilters

__all__ = ["StarItem", "StarList", "StarListSet", "generate_starlist_schema", "generate_star_list_set_schema"]


class PrettyPrintMixin:
    @classmethod
    def markdown_table(cls):
        """
        Generate a markdown table of the model fields for the AAVSO starlist schema.
        """
        rows = ["| Title | JSON Field | Type | Unit | Description | Examples |"]
        rows.append("| --- | --- | --- | --- | --- | --- |")
        for name, field_info in cls.model_fields.items():
            row = (
                f"| {field_info.title} | {name} | {field_info.annotation.__name__} "
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
    tot_flux: Annotated[
        float,
        Field(
            ge=0,
            title="Star Flux",
            description="Total integrated counts of the star, background-subtracted",
            json_schema_extra=dict(unit="adu"),
            examples=[156700.4]
        )
    ]
    flux_err: Annotated[
        float,
        Field(
            ge=0,
            title="Flux Error",
            description="Error in the total integrated counts of the star",
            json_schema_extra=dict(unit="adu"),
            examples=[15300.1]
        )
    ]
    bkgd_flux: Annotated[
        float,
        Field(
            ge=0,
            title="Background counts",
            description="Background count level in the vicinity of the star",
            json_schema_extra=dict(unit="adu / pixel"),
            examples=[1209.45]
        )
    ]
    peak_flux: Annotated[
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
    gain: Annotated[
        float,
        Field(
            ge=0,
            title="Gain",
            description="Gain of the camera",
            json_schema_extra=dict(unit="e-/adu"),
            examples=[1.2]
        )
    ]
    epoch: Annotated[
        str,
        Field(
            title="Reporting Epoch",
            description="Epoch of the observation",
            json_schema_extra=dict(unit="none"),
            examples=["J2000"]
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
            raise ValueError(f"Missing keys in metadata: {", ".join(missing_keys)}")

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

class StarListSet(BaseModel, PrettyPrintMixin, GenerateInstanceFromExamplesMixin):
    """
    Class to hold a list for which each entry is a star list.
    """
    # Put the version here because this is the file we expect manufacturers
    # to submit.
    schema_version: Annotated[
        str,
        Field(
            title="Starlist Schema Version",
            description="An AAVSO-assigned string that identifies the schema version",
            json_schema_extra=dict(unit="none"),
            examples=["0.0.1"],
            default=__version__,
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


def generate_starlist_schema():
    return json.dumps(StarList.model_json_schema(), indent=2)


def generate_star_list_set_schema():
    return json.dumps(StarListSet.model_json_schema(), indent=2)
