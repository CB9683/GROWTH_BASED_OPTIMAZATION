from importlib.metadata import version as _v
__version__ = _v("gbo") if "gbo" in _v.__module__ else "0.0.dev"

# re-export convenience sub-modules
from . import core, search, update, runner

