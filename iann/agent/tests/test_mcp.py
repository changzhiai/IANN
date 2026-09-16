"""Checks for the MCP server: the tool table, and a real stdio round trip.

Run from the repository root::

    python iann/agent/tests/test_mcp.py

Two halves, and the first one matters more. The tool table and the
``call_tool`` dispatcher are checked **without** the MCP SDK, because that is
the part the CLI and the server share and the part that breaks when
``commands.py`` changes. The stdio round trip needs the optional SDK and is
skipped, not failed, when it is missing -- ``pip install -e ".[agent]"``.

Exit codes match the rest of the repo's scripted checks: 0 pass, 1 fail.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

from iann.agent import mcp_server                                   # noqa: E402

EXPECTED_TOOLS = [
    "iann_doctor",
    "iann_list_architectures",
    "iann_list_foundation_models",
    "iann_foundation_info",
    "iann_inspect_checkpoint",
    "iann_predict",
    "iann_train",
    "iann_training_status",
    "iann_export_lammps",
]

# Every tool that cannot write a file or spend real compute. These are the ones
# the round trip is allowed to call.
READ_ONLY = {
    "iann_doctor": {},
    "iann_list_architectures": {},
    "iann_list_foundation_models": {"group": "arch"},
    "iann_foundation_info": {"name": "rpbe-all"},
    "iann_inspect_checkpoint": {"model_path": "test/painn/output/model.pt"},
}

_failures = []
_skipped = []


def check(label, condition, detail=""):
    if condition:
        print(f"  ok   {label}")
    else:
        print(f"  FAIL {label}" + (f" -- {detail}" if detail else ""))
        _failures.append(label)
    return bool(condition)


# ------------------------------------------------------------ the tool table

def test_tool_table():
    print("\n[tool table] (no SDK required)")
    described = mcp_server._describe()
    names = [t["name"] for t in described]

    check("nine tools are exposed", len(described) == 9, f"got {len(described)}")
    check("names are exactly the documented set", names == EXPECTED_TOOLS,
          f"got {names}")

    for tool in described:
        label = tool["name"]
        schema = tool["inputSchema"]
        ok = (isinstance(tool["description"], str) and tool["description"]
              and schema.get("type") == "object"
              and isinstance(schema.get("properties"), dict))
        check(f"{label} has a description and an object schema", ok)
        # A schema must be serialisable, or the client never sees the tool.
        try:
            json.dumps(schema)
        except TypeError as exc:
            check(f"{label} schema is JSON-serialisable", False, str(exc))

    # The required arguments are the contract; assert them rather than trusting
    # that nobody edited the schema.
    required = {t["name"]: t["inputSchema"].get("required", []) for t in described}
    check("iann_train requires model, dataset and max_steps",
          sorted(required["iann_train"]) == ["dataset", "max_steps", "model"],
          str(required["iann_train"]))
    check("iann_predict requires model_path and structure",
          sorted(required["iann_predict"]) == ["model_path", "structure"],
          str(required["iann_predict"]))
    check("iann_export_lammps requires only model_path",
          required["iann_export_lammps"] == ["model_path"],
          str(required["iann_export_lammps"]))
    check("iann_doctor takes no required arguments",
          required["iann_doctor"] == [])

    # max_steps being mandatory is the whole bounded-compute guarantee, so the
    # description has to say so where an agent will read it.
    train_desc = dict((t["name"], t["description"]) for t in described)["iann_train"]
    check("iann_train's description states that max_steps is mandatory",
          "mandatory" in train_desc.lower())


def test_dispatcher():
    print("\n[dispatcher] errors are returned, not raised")

    result = mcp_server.call_tool("no_such_tool", {})
    check("an unknown tool returns a structured error",
          result.get("error") == "UnknownTool", str(result)[:120])
    check("an unknown tool lists what is available",
          sorted(result.get("available", [])) == sorted(EXPECTED_TOOLS))

    # A missing mandatory argument must come back as data, not as a traceback
    # through the transport.
    result = mcp_server.call_tool("iann_train", {"model": "painn"})
    check("a missing mandatory argument returns an error object",
          "error" in result and "message" in result, str(result)[:160])

    result = mcp_server.call_tool("iann_foundation_info", {"name": "nosuchmodel"})
    check("an unknown foundation model returns an error object",
          result.get("error") == "UnknownFoundationModelError", str(result)[:160])

    result = mcp_server.call_tool("iann_inspect_checkpoint",
                                  {"model_path": "/nonexistent/model.pt"})
    check("a missing checkpoint returns an error object",
          "error" in result, str(result)[:160])


# ----------------------------------------------------------- the stdio server

def test_stdio_round_trip():
    print("\n[stdio] a real client talking to a real server")
    try:
        import anyio
        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client
    except ImportError as exc:
        print(f"  skip  the MCP SDK is not importable ({exc})")
        print('        install it with: pip install -e ".[agent]"')
        _skipped.append("stdio round trip")
        return

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "iann.agent.mcp_server"],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": ROOT},
    )

    async def _exercise():
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                listed = await session.list_tools()
                names = [t.name for t in listed.tools]
                check("the server lists nine tools over stdio",
                      len(names) == 9, f"got {len(names)}")
                check("the served names match the documented set",
                      names == EXPECTED_TOOLS, f"got {names}")

                served = {t.name: t for t in listed.tools}
                for tool in mcp_server._describe():
                    wire = served.get(tool["name"])
                    if wire is None:
                        continue
                    # SDK 2.x renamed the attribute to `input_schema` and kept
                    # `inputSchema` only as an input alias, so read both.
                    wire_schema = getattr(wire, "input_schema", None)
                    if wire_schema is None:
                        wire_schema = getattr(wire, "inputSchema", None)
                    # The schema must survive the wire intact -- this is what
                    # broke silently between SDK generations.
                    check(f"{tool['name']} schema survives the wire",
                          wire_schema == tool["inputSchema"],
                          f"{wire_schema} != {tool['inputSchema']}")
                    check(f"{tool['name']} description survives the wire",
                          wire.description == tool["description"])

                for name, arguments in READ_ONLY.items():
                    response = await session.call_tool(name, arguments)
                    text = "".join(c.text for c in response.content
                                   if getattr(c, "type", None) == "text")
                    try:
                        payload = json.loads(text)
                    except json.JSONDecodeError as exc:
                        check(f"{name} returns parseable JSON", False,
                              f"{exc}: {text[:120]}")
                        continue
                    check(f"{name} returns parseable JSON without an error",
                          isinstance(payload, dict) and "error" not in payload,
                          str(payload)[:160])

                # And the refusal path, end to end over the transport.
                response = await session.call_tool("iann_train", {"model": "painn"})
                text = "".join(c.text for c in response.content
                               if getattr(c, "type", None) == "text")
                check("unbounded training is refused over the transport",
                      "error" in json.loads(text), text[:160])

    anyio.run(_exercise)


def test_server_without_sdk():
    """The server must explain itself rather than traceback when the SDK is absent."""
    print("\n[degradation] behaviour with no MCP SDK on the path")
    # A stub module named `mcp` that raises on import reproduces "not
    # installed" without uninstalling anything. It goes in a temp directory so
    # it cannot shadow the real SDK for anything else.
    stub = tempfile.mkdtemp(prefix="iann-nomcp-")
    with open(os.path.join(stub, "mcp.py"), "w") as fh:
        fh.write("raise ImportError(\"No module named 'mcp'\")\n")

    proc = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, sys.argv[1]);"
         "from iann.agent.mcp_server import main; main()", stub],
        cwd=ROOT, capture_output=True, text=True, timeout=120,
        env={**os.environ, "PYTHONPATH": ROOT},
    )
    shutil.rmtree(stub, ignore_errors=True)
    output = proc.stdout + proc.stderr
    check("it exits non-zero", proc.returncode != 0, f"rc={proc.returncode}")
    check("it names the optional extra to install",
          'pip install -e ".[agent]"' in output, output[-300:])
    check("it says the CLI still works",
          "CLI works without it" in output, output[-300:])
    check("it is not a bare traceback",
          "Traceback" not in output or "SystemExit" not in output)


def main():
    print("=" * 66)
    print("MCP server checks".center(66))
    print("=" * 66)

    test_tool_table()
    test_dispatcher()
    test_server_without_sdk()
    test_stdio_round_trip()

    print("\n" + "=" * 66)
    if _failures:
        print(f"FAILED: {len(_failures)} check(s)")
        for name in _failures:
            print(f"   - {name}")
        return 1
    if _skipped:
        print(f"PASSED (with {len(_skipped)} skipped: {', '.join(_skipped)})")
        return 0
    print("PASSED: all checks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
