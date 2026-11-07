"""Frozen models for the legacy generation of the AAVSO starlist schema.

The legacy generation is everything before schema version 0.2.0: files with no
``schema_version``, or with a package version such as ``0.1.dev66+g6f57c4d02``
in that field. These models are a copy of the models as they were immediately
before 0.2.0, kept so that a legacy file can be validated against the schema it
was written for, separately from validating the result of upgrading it.

This module is frozen:

- Do not edit the models.
  ``aavso_starlist_schema_data/legacy/schema_definition.json`` records the
  schema they generate, and the tests fail if the two differ.
- Do not import anything from ``aavso_starlist_schema``. A frozen model that
  shares code with the live models changes whenever they do.

In the original, ``schema_version`` defaulted to the version of the installed
package, which differed in every build. Here the default is frozen at the value
in the last legacy schema file committed to the repository, so the schema these
models generate is identical to that file.
"""

from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, Field

__all__ = ["AAVSOFilters", "StarItem", "StarList", "StarListSet"]


# No docstring: it would become a schema description that the legacy schema
# did not have.
class AAVSOFilters(StrEnum):
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


class StarItem(BaseModel):
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


class StarList(BaseModel):
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


class StarListSet(BaseModel):
    """
    Class to hold a list for which each entry is a star list.
    """
    schema_version: Annotated[
        str,
        Field(
            title="Starlist Schema Version",
            description="An AAVSO-assigned string that identifies the schema version",
            json_schema_extra=dict(unit="none"),
            examples=["0.0.1"],
            default="0.1.dev63+g61de51202.d20260609",
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
