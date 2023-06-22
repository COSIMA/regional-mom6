try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"

from .regional_mom6 import *  # noqa
