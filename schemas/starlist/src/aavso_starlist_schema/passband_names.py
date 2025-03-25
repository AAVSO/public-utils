from enum import StrEnum

import requests

PASSBAND_URL = "https://www.aavso.org/vsx/index.php?view=api.bands&format=json"
PASSBAND_VERSION = "2"

# Copy/paste from stellarphot
class _AAVSOFilters(StrEnum):
    """
    The definitive list of AAVSO filters is at https://www.aavso.org/filters
    """

    U = "U"
    B = "B"
    V = "V"
    RJ = "RJ"
    Rc = "R"
    Ic = "I"
    IJ = "IJ"
    J = "J"
    H = "H"
    K = "K"
    TG = "TG"
    TB = "TB"
    TR = "TR"
    CV = "CV"
    CR = "CR"
    SZ = "SZ"
    SU = "SU"
    SG = "SG"
    SR = "SR"
    SI = "SI"
    STU = "STU"
    STV = "STV"
    STB = "STB"
    STY = "STY"
    STHBW = "STHBW"
    STHBN = "STHBN"
    MA = "MA"
    MB = "MB"
    MI = "MI"
    ZS = "ZS"
    Y = "Y"
    HA = "HA"
    HAC = "HAC"
    CBB = "CBB"
    O = "O"  # noqa: E741



def _fallback_filters():
    """
    Provide filter names in case VSX is down or the request times out.
    """
    return [{"ShortName": filter_name} for filter_name in _AAVSOFilters]



def get_passband_data():
    """
    Retrieve the passband data from the AAVSO API and return it as a dictionary.

    Note that only the passband data is returned, not the entire response.
    """
    try:
        response = requests.get(PASSBAND_URL, timeout=10)
    except requests.ReadTimeout:
        # Fall back to some hard-coded passband data if the request times out
        return _fallback_filters()

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
