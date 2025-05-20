"""synth_env namespace package

This file re-exports the existing top-level modules in the *Environments* repository
under the ``synth_env`` namespace so that they can be imported as, e.g.,
``import synth_env.stateful.core``.

The repository predates this layout and keeps each sub-package (``stateful``,
``environment`` …) at the project root.  Rather than move every directory we
shim them into the desired namespace dynamically here.

This approach keeps backward-compatibility for callers expecting
``synth_env.<submodule>`` while leaving the on-disk structure unchanged.
"""

from importlib import import_module
import sys

# List every top-level package that belongs to the synth_env namespace.
_submodules = [
    "stateful",
    "environment",
    "tasks",
    "service",
    "examples",
    "reproducibility",
    "v0_observability",
]

_current = sys.modules[__name__]

for _name in _submodules:
    # Import the real top-level module.
    _mod = import_module(_name)
    # Expose it as synth_env.<name>
    sys.modules[f"{__name__}.{_name}"] = _mod
    setattr(_current, _name, _mod)

# Clean up the helper names from the namespace.
del _current, _mod, _name, _submodules, import_module, sys 