try:
    from ._version import __version__
except ImportError:  # pragma: no cover - fallback for source tree without hatch-vcs _version.py
    __version__ = "0.0.0"

from .schema_generator import *
