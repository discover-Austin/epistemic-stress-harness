"""
Deprecated compatibility shim.

v0.1 shipped a second, drifting copy of the harness in this file. The
implementation now lives only in the `harness` package; this module re-exports
it so old imports keep working.
"""

import warnings

warnings.warn(
    "epistemic_harness.py is deprecated; import from the `harness` package.",
    DeprecationWarning,
    stacklevel=2,
)

from harness import *  # noqa: E402,F401,F403
from harness import __all__  # noqa: E402,F401
