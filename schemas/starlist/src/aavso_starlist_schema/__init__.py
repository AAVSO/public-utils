try:
    from ._version import __version__
except ImportError:
    __version__ = "0.0.0"

from .schema_generator import *
