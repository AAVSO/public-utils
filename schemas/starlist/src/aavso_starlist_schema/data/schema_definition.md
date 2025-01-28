# Star List Set

| Title | JSON Field | Type | Unit | Description | Examples |
| --- | --- | --- | --- | --- | --- |
| Starlist Schema Version | schema_version | str | none | An AAVSO-assigned string that identifies the schema version | 0.0.1 |
| Star List Set | star_lists | List | none | List of star lists | Each item should be a StarList |





# Star List

| Title | JSON Field | Type | Unit | Description | Examples |
| --- | --- | --- | --- | --- | --- |
| Observation Start Time | obs_time | str | None | UTC time at start of observation | 2021-06-15T03:45:00 |
| Site Latitude | site_lat | float | degree | Latitude of the observing site | -41.56896 |
| Site Longitude | site_lon | float | degree | Longitude of the observing site | -71.23841 |
| Site Elevation | site_elev | float | meter | Observer's elevation above mean sea level | 211.0 |
| Observer Code | observer | str | none | AAVSO code of the observer | MMU |
| Filter | filter | AAVSOFilters | none | Filter used for the observation, from https://www.aavso.org/filters | TG |
| Blocking Filter | block_filter | str | none | Name of blocking filter used on telescope | UV+IR |
| Exposure Time | exposure | float | second | Effective duration of exposure | 30.0 |
| Telescope Manufacturer | tel_manufac | str | none | Name of the telescope manufacturer | Celestron |
| Telescope Model | tel_model | str | none | Model of the telescope | Origin 1 |
| Telescope Firmware | tel_firmware | str | none | Firmware version of the telescope | 20240817.01 |
| A/D Converter Bit Depth | adc_depth | int | bit | Bit depth of the analog-to-digital converter | 14 |
| Largest Usable ADU Value | largest_usable_adu_value | int | adu | Largest usable analog-to-digital unit value | 41000 |
| Gain | gain | float | e-/adu | Gain of the camera | 1.2 |
| Reporting Epoch | epoch | str | none | Epoch of the observation | J2000 |
| Coordinate Reference Frame | refframe | str | none | Reference frame of the observation | ICRS |
| Star Items | staritems | List | none | List of stars detected in the image | Each item should be a StarItem |





# Star Item

| Title | JSON Field | Type | Unit | Description | Examples |
| --- | --- | --- | --- | --- | --- |
| X-coordinate | x | float | pixel | X-coordinate of the star center (in pixel coordinates) | 1206.78 |
| Y-coordinate | y | float | pixel | Y-coordinate of the star center (in pixel coordinates) | 620.1 |
| Right Ascension | ra | float | degree | Right Ascension of the star (in decimal degrees) at the epoch specified in the metadata | 212.56789 |
| Declination | dec | float | degree | Declination of the star (in decimal degrees) at the epoch specified in the metadata | -12.12345 |
| Star Flux | tot_flux | float | adu | Total integrated counts of the star, background-subtracted | 156700.4 |
| Flux Error | flux_err | float | adu | Error in the total integrated counts of the star | 15300.1 |
| Background counts | bkgd_flux | float | adu / pixel | Background count level in the vicinity of the star | 1209.45 |
| Peak Counts | peak_flux | float | adu | Peak counts of the star | 31454.963 |
