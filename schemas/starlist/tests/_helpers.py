"""
Helpers shared by the test modules, which import them from here.

They live outside ``conftest.py`` because pytest advises against importing
from it; ``pythonpath = ["tests"]`` in ``pyproject.toml`` makes this module
importable under every import mode.
"""

from aavso_starlist_schema import _type_name


def float_fields(model):
    """
    Name the fields of a model whose type is float, optional or not.

    Parameters
    ----------
    model : type
        A pydantic model class.

    Returns
    -------
    tuple of str
        The names of the fields typed ``float`` or ``float | None``, in the
        order the model defines them.
    """
    return tuple(
        name
        for name, field in model.model_fields.items()
        if _type_name(field.annotation) in ("float", "float | None")
    )
