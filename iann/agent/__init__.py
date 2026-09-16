"""Agent-facing interface to IANN.

This subpackage is the machine-readable front door to the framework: one
implementation in :mod:`iann.agent.commands`, exposed through a CLI
(:mod:`iann.agent.cli`, installed as the ``iann`` command) and an MCP server
(:mod:`iann.agent.mcp_server`).

Every command returns a plain JSON-serialisable ``dict`` and raises on failure,
which is what the rest of the library does not do: ``Trainer.train()`` returns
``None`` and reports metrics only into a log file, and
``convert_model_for_lammps()`` returns ``None`` and prints when its input is
missing. Those are the gaps this package closes.

Nothing here imports torch at module level, so ``iann agent`` and
``iann doctor`` still work in an environment where the numerical stack is
broken -- which is precisely when a diagnostic is wanted.
"""

from .commands import (  # noqa: F401
    doctor,
    export_lammps,
    foundation_fetch,
    foundation_info,
    foundation_list,
    inspect_checkpoint,
    list_architectures,
    predict,
    train,
    training_status,
)

__all__ = [
    "doctor",
    "list_architectures",
    "foundation_list",
    "foundation_info",
    "foundation_fetch",
    "inspect_checkpoint",
    "predict",
    "train",
    "training_status",
    "export_lammps",
]
