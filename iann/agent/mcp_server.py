"""MCP server exposing IANN to any Model Context Protocol client.

An adapter, nothing more: every tool calls one function from
:mod:`iann.agent.commands` and returns its dict. The CLI calls the same
functions, so the two front ends cannot drift apart.

Run it::

    python -m iann.agent.mcp_server

Register it with Claude Code::

    claude mcp add iann -- /opt/anaconda3/envs/iann/bin/python -m iann.agent.mcp_server

Requires the optional dependency: ``pip install -e ".[agent]"``.

What is deliberately not exposed: unbounded training (every training tool takes
a mandatory step budget), job submission to SLURM or PBS, uploads to the
HuggingFace Hub, and anything that deletes files. An agent can spend a bounded
amount of local compute here and nothing else.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from . import commands

SERVER_NAME = "iann"

# One entry per tool: the name an agent sees, the function it calls, the
# description it reads, and its input schema. Descriptions carry the facts that
# stop an agent wasting time -- measured per-step costs, and which architectures
# need config the checkpoint does not contain.
TOOLS: List[Dict[str, Any]] = [
    {
        "name": "iann_doctor",
        "fn": lambda **kw: commands.doctor(),
        "description": (
            "Check whether this environment can run IANN. Reports each dependency "
            "separately and distinguishes a hard failure (asap3 cannot import) from "
            "a silent degradation (torch imports but warns that NumPy failed to "
            "initialise). Call this first if anything else fails confusingly."),
        "schema": {"type": "object", "properties": {}},
    },
    {
        "name": "iann_list_architectures",
        "fn": lambda **kw: commands.list_architectures(),
        "description": (
            "List the neural-network architectures the trainer accepts, whether each "
            "can be exported to LAMMPS, and which structural parameters each one "
            "needs that checkpoints do not record."),
        "schema": {"type": "object", "properties": {}},
    },
    {
        "name": "iann_list_foundation_models",
        "fn": lambda group="painn", **kw: commands.foundation_list(group),
        "description": (
            "List the released pretrained foundation models with their energy and "
            "force MAEs. Group 'painn' is the twelve released potentials, 'arch' the "
            "seven architecture-comparison runs, 'all' both. Match the exchange-"
            "correlation functional to your system: rpbe-* for surfaces and "
            "catalysis, pbe-* for bulk, r2scan-* for meta-GGA."),
        "schema": {
            "type": "object",
            "properties": {"group": {"type": "string",
                                     "enum": ["painn", "arch", "all"],
                                     "default": "painn"}},
        },
    },
    {
        "name": "iann_foundation_info",
        "fn": lambda name, **kw: commands.foundation_info(name),
        "description": (
            "Details of one foundation model, including whether it is already cached "
            "locally and where. Names are case-insensitive and accept the paper's "
            "spellings; an unknown name returns a did-you-mean list."),
        "schema": {
            "type": "object",
            "properties": {"name": {"type": "string",
                                    "description": "e.g. 'rpbe-all', 'pbe-mptrj'"}},
            "required": ["name"],
        },
    },
    {
        "name": "iann_inspect_checkpoint",
        "fn": lambda model_path, **kw: commands.inspect_checkpoint(model_path),
        "description": (
            "Report what a trained checkpoint records -- architecture, step, "
            "best validation loss, channel and layer counts -- and, importantly, "
            "which structural parameters it does NOT record. Call this before "
            "rebuilding or exporting a model: EquiformerV3 and UMA checkpoints omit "
            "values that must be supplied or load_state_dict fails with size "
            "mismatches."),
        "schema": {
            "type": "object",
            "properties": {"model_path": {"type": "string"}},
            "required": ["model_path"],
        },
    },
    {
        "name": "iann_predict",
        "fn": lambda model_path, structure, index=0, device=None, ensemble=None,
                     config=None, **kw: commands.predict(
                         model_path, structure, index=index, device=device,
                         ensemble=ensemble, config=config),
        "description": (
            "Single-point energy and forces for one structure, using any ASE-readable "
            "file. Cheap: 9 ms per call for PaiNN, up to 115 ms for EquiformerV2. "
            "Passing extra checkpoints in 'ensemble' also returns the variance across "
            "them, which is the signal to use for active learning."),
        "schema": {
            "type": "object",
            "properties": {
                "model_path": {"type": "string"},
                "structure": {"type": "string",
                              "description": "ASE-readable file, e.g. a .traj or .cif"},
                "index": {"type": "integer", "default": 0,
                          "description": "frame index within a trajectory"},
                "device": {"type": "string", "description": "'cpu' or 'cuda'"},
                "ensemble": {"type": "array", "items": {"type": "string"},
                             "description": "additional checkpoints for uncertainty"},
                "config": {"type": "object",
                           "description": "extra model parameters; see "
                                          "iann_inspect_checkpoint"},
            },
            "required": ["model_path", "structure"],
        },
    },
    {
        "name": "iann_train",
        "fn": lambda model, dataset, max_steps, output_dir=None, config=None,
                     **kw: commands.train(model, dataset, max_steps,
                                          output_dir=output_dir, config=config),
        "description": (
            "Train a model for a BOUNDED number of steps and return a manifest with "
            "the final metrics. 'max_steps' is mandatory: the trainer's own default is "
            "1,000,000 steps, so omitting a budget would start a run lasting days. "
            "Budget from the measured cost -- PaiNN about 1.6 s/step on CPU, NequIP "
            "about 4.1 s/step, EquiformerV2 slower. Use 10-50 steps to check a setup "
            "works; real training belongs on a cluster."),
        "schema": {
            "type": "object",
            "properties": {
                "model": {"type": "string",
                          "description": "painn, nequip, allegro, mace, equiformerv2, "
                                         "equiformerv3, uma, fastpot"},
                "dataset": {"type": "string",
                            "description": "ASE .traj or .db with energies and forces"},
                "max_steps": {"type": "integer", "minimum": 1,
                              "description": "REQUIRED step budget"},
                "output_dir": {"type": "string"},
                "config": {"type": "object",
                           "description": "extra trainer/model keys; see DEFAULT_CONFIG"},
            },
            "required": ["model", "dataset", "max_steps"],
        },
    },
    {
        "name": "iann_training_status",
        "fn": lambda output_dir, **kw: commands.training_status(output_dir),
        "description": (
            "Progress of a training run, from its log and its checkpoint. Safe to "
            "call while training is still running. Returns the latest step and "
            "metrics plus the best validation loss so far, and a state of running, "
            "early_stopped, max_steps_reached or nan_exit."),
        "schema": {
            "type": "object",
            "properties": {"output_dir": {"type": "string"}},
            "required": ["output_dir"],
        },
    },
    {
        "name": "iann_export_lammps",
        "fn": lambda model_path, model_type=None, output_path=None, config=None,
                     ensemble=None, **kw: commands.export_lammps(
                         model_path, model_type, output_path,
                         config=config, ensemble=ensemble),
        "description": (
            "Export a checkpoint to TorchScript for LAMMPS 'pair_style iann'. "
            "model_type is inferred from the checkpoint when omitted. FastPot cannot "
            "be exported, and UMA requires num_experts == 0. If this fails with a "
            "size mismatch, call iann_inspect_checkpoint and pass the missing "
            "structural parameters in 'config'."),
        "schema": {
            "type": "object",
            "properties": {
                "model_path": {"type": "string"},
                "model_type": {"type": "string"},
                "output_path": {"type": "string"},
                "config": {"type": "object"},
                "ensemble": {"type": "array", "items": {"type": "string"},
                             "description": "extra checkpoints for an ensemble export"},
            },
            "required": ["model_path"],
        },
    },
]


def _describe() -> List[Dict[str, Any]]:
    """The tool list without the callables, for tests and introspection."""
    return [{"name": t["name"], "description": t["description"],
             "inputSchema": t["schema"]} for t in TOOLS]


def call_tool(name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Invoke one tool by name. Errors are returned, not raised.

    An MCP client is better served by a structured error it can reason about
    than by a transport-level failure, so a failing command becomes
    ``{"error": ..., "message": ...}``.
    """
    for tool in TOOLS:
        if tool["name"] == name:
            try:
                return tool["fn"](**(arguments or {}))
            except Exception as exc:                   # noqa: BLE001 - tool boundary
                return {"error": type(exc).__name__, "message": str(exc)}
    return {"error": "UnknownTool", "message": f"No such tool: {name}",
            "available": [t["name"] for t in TOOLS]}


def _serve_v2(mcp_module, types, anyio) -> None:
    """Serve using the 2.x SDK, whose high-level server class is ``MCPServer``.

    ``MCPServer.add_tool`` derives each input schema from the registered
    function's signature, which would discard the schemas above -- their enums,
    defaults and per-parameter descriptions are the part an agent actually
    reads. So the two public methods the request handlers dispatch through are
    overridden instead, and :data:`TOOLS` stays the single source of truth.
    """
    MCPServer = mcp_module.server.MCPServer

    class IannServer(MCPServer):
        async def list_tools(self):
            return [types.Tool(name=t["name"], description=t["description"],
                               inputSchema=t["schema"]) for t in TOOLS]

        async def call_tool(self, name, arguments, context=None):
            # A worker thread: every command is synchronous and some are long
            # enough (training) to block the event loop.
            result = await anyio.to_thread.run_sync(
                lambda: call_tool(name, arguments or {}))
            return types.CallToolResult(
                content=[types.TextContent(
                    type="text", text=json.dumps(result, indent=2, default=str))],
                isError="error" in result)

    server = IannServer(name=SERVER_NAME)
    anyio.run(server.run_stdio_async)


def _serve_v1(types, anyio) -> None:
    """Serve using the 1.x SDK, where the low-level server carries decorators."""
    from mcp.server import Server
    from mcp.server.stdio import stdio_server

    server = Server(SERVER_NAME)

    @server.list_tools()
    async def list_tools():
        return [types.Tool(name=t["name"], description=t["description"],
                           inputSchema=t["schema"]) for t in TOOLS]

    @server.call_tool()
    async def handle(name: str, arguments: Dict[str, Any]):
        result = await anyio.to_thread.run_sync(lambda: call_tool(name, arguments))
        return [types.TextContent(type="text",
                                  text=json.dumps(result, indent=2, default=str))]

    async def _run():
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    anyio.run(_run)


def main() -> None:
    """Serve over stdio, on either generation of the MCP SDK.

    The 1.x and 2.x SDKs disagree about how a server is built: 2.x removed the
    ``@server.list_tools()`` and ``@server.call_tool()`` decorators from the
    low-level ``Server`` and moved the supported surface to ``MCPServer``. Since
    ``pip install "iann[agent]"`` resolves to whatever is current, both are
    supported rather than pinning users to the older one.
    """
    try:
        import mcp
        import mcp.server
        import mcp.types as types
    except ImportError as exc:                         # pragma: no cover - env dependent
        raise SystemExit(
            f"The MCP SDK is required to run the server ({exc}). Install it with:\n"
            "    pip install -e \".[agent]\"\n"
            "The `iann` CLI works without it."
        )

    import anyio

    if hasattr(mcp.server, "MCPServer"):
        _serve_v2(mcp, types, anyio)
    else:
        _serve_v1(types, anyio)


if __name__ == "__main__":
    main()
