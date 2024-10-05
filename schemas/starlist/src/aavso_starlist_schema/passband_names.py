from enum import StrEnum

import requests


PASSBAND_URL = "https://www.aavso.org/vsx/index.php?view=api.bands&format=json"
PASSBAND_VERSION = "2"


def get_passband_data():
    """
    Retrieve the passband data from the AAVSO API and return it as a dictionary.

    Note that only the passband data is returned, not the entire response.
    """
    response = requests.get(PASSBAND_URL)
    if response.status_code != 200:
        raise ValueError(
            f"Failed to retrieve passband data from {PASSBAND_URL}:\n"
            f"{response.text}"
        )
    content = response.json()
    if content["Bands"]["@version"] != PASSBAND_VERSION:
        raise ValueError(
            f"Unexpected passband data version. Expected {PASSBAND_VERSION}, "
            f"but got {content['Bands']['@version']}"
        )

    return content["Bands"]["Band"]


AAVSOFilters = StrEnum(
    "AAVSOFilters",
    {
        # This odd structure ensures that the enum value is the same as the
        # short filter name.
        band["ShortName"]: band["ShortName"]
        for band in get_passband_data()
    }
)
