"""
Helpers shared by the test modules, which import them from here.

They live outside ``conftest.py`` because pytest advises against importing
from it; ``pythonpath = ["tests"]`` in ``pyproject.toml`` makes this module
importable under every import mode.
"""

import copy

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


def flooded(data, path, entry, count):
    """
    Copy starlist-set data with one list field holding many copies of an entry.

    Parameters
    ----------
    data : dict
        Raw starlist-set data, which is not modified.
    path : tuple of str or int
        The keys and indexes leading from ``data`` to the list field, such as
        ``("star_lists", 0, "staritems")``.
    entry : object
        The entry to repeat, typically one that is not valid for the field.
    count : int
        How many times to repeat it.

    Returns
    -------
    dict
        A deep copy of ``data`` with the value at ``path`` replaced by a list
        of ``count`` references to ``entry``.
    """
    data = copy.deepcopy(data)
    parent = data
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = [entry] * count
    return data
