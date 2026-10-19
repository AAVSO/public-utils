import pytest

from aavso_starlist_schema import SCHEMA_VERSION


@pytest.fixture
def next_generation_version():
    """
    Compute the first version of the generation after ``SCHEMA_VERSION``'s.

    Returns
    -------
    str
        The next minor while the schema is ``0.y.z``, and the next major from
        1.0.0 on, so the tests stay correct as ``SCHEMA_VERSION`` moves.
    """
    major, minor, _ = (int(part) for part in SCHEMA_VERSION.split("."))
    return f"0.{minor + 1}.0" if major == 0 else f"{major + 1}.0.0"
