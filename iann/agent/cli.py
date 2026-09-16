"""The ``iann`` command: a machine-readable front end to the framework.

Argument parsing and serialisation only -- every operation lives in
:mod:`iann.agent.commands`. Exit codes follow the convention already used by
``test/docs/build.sh``, the repo's other scripted check:

* ``0`` success
* ``1`` the operation failed (a JSON error object goes to stderr)
* ``2`` the command line was wrong

``--json`` prints the raw result for a caller to parse; without it, a compact
human summary is printed instead. Failures are never a bare traceback, and
never a zero exit with an empty result.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional

from . import commands

# Exit codes, named so the tests and the docs can refer to them.
EXIT_OK, EXIT_FAIL, EXIT_USAGE = 0, 1, 2


def _load_config(path: Optional[str]) -> Optional[Dict[str, Any]]:
    """Read a TOML or JSON config file of extra model/trainer keys."""
    if not path:
        return None
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No such config file: {path}")
    if path.endswith(".json"):
        with open(path) as fh:
            return json.load(fh)
    import toml
    return toml.load(path)


# ------------------------------------------------------------ human rendering

def _print_human(cmd: str, result: Dict[str, Any]) -> None:
    """A short summary for a person; --json is for everything else."""
    if cmd == "doctor":
        status = "OK" if result["ok"] else "PROBLEMS"
        print(f"environment : {status}")
        print(f"python      : {result['python']}  ({result['executable']})")
        for name, dep in result["dependencies"].items():
            mark = "ok  " if dep["ok"] else "FAIL"
            if dep["ok"] and dep["warnings"]:
                mark = "warn"
            extra = dep["error"] or (dep["warnings"][0] if dep["warnings"] else "")
            ver = dep["version"] or ""
            print(f"   {mark}  {name:16s} {ver:10s} {extra[:60]}")
        cuda = result["cuda"]
        print(f"cuda        : {cuda['available']} ({cuda['device_count']} device(s))")
        if result["hint"]:
            print(f"\n{result['hint']}")
        return

    if cmd == "models":
        for arch in result["architectures"]:
            flag = "lammps" if arch["lammps_exportable"] else "      "
            missing = (" needs: " + ", ".join(arch["unpersisted_config"])
                       if arch["unpersisted_config"] else "")
            print(f"   {arch['name']:14s} {flag}{missing}")
        print(f"\n{result['count']} architectures")
        return

    if cmd == "foundation-list":
        hdr = f"   {'name':16s} {'functional':11s} {'dataset':10s} {'E_MAE':>8s} {'F_MAE':>8s}"
        print(hdr)
        for row in result["models"]:
            print(f"   {row['name']:16s} {row['functional']:11s} {row['dataset']:10s} "
                  f"{row['energy_mae']:>8.4f} {row['forces_mae']:>8.4f}")
        print(f"\n{result['count']} models")
        return

    if cmd == "inspect":
        print(f"path          : {result['path']}")
        print(f"architecture  : {result['architecture']}  "
              f"(declared={result['model_type_declared']}, "
              f"inferred={result['model_type_inferred']})")
        print(f"tensors       : {result['n_parameter_tensors']}")
        for key, value in result["metadata"].items():
            print(f"   {key:18s} {value}")
        if result["missing_config"]:
            print("\nNOT stored in this checkpoint -- pass these when rebuilding "
                  "or exporting if they were non-default:")
            for key in result["missing_config"]:
                print(f"   {key}")
        return

    if cmd == "predict":
        print(f"structure   : {result['formula']} ({result['n_atoms']} atoms)")
        print(f"energy      : {result['energy']:.6f} eV "
              f"({result['energy_per_atom']:.6f} eV/atom)")
        print(f"max |force| : {result['max_force']:.6f} eV/A")
        if "ensemble" in result:
            print(f"ensemble    : {result['n_models']} models, "
                  f"{result['ensemble']}")
        return

    if cmd == "train":
        print(f"model       : {result['model']}")
        print(f"steps       : {result['progress']['step']} of {result['max_steps']} "
              f"({result['progress']['state']})")
        for key, value in result["final_metrics"].items():
            print(f"   {key:18s} {value}")
        print(f"checkpoint  : {result['model_path']}")
        print(f"manifest    : {result['manifest_path']}")
        return

    if cmd == "status":
        print(f"output_dir  : {result['output_dir']}")
        print(f"state       : {result['state']}")
        print(f"step        : {result['step']}")
        for key, value in result.get("metrics", {}).items():
            print(f"   {key:18s} {value}")
        if result.get("checkpoint"):
            print(f"best_val_loss: {result['checkpoint'].get('best_val_loss')}")
        return

    if cmd == "export":
        print(f"exported    : {result['output_path']}")
        print(f"size        : {result['size_bytes']} bytes")
        return

    if cmd == "agent-install":
        print(f"target      : {result['target']}")
        for action in result["actions"]:
            rel = os.path.relpath(action["path"], result["target"])
            print(f"   {action['action']:18s} {rel}")
        print(f"\n{result['written']} written, {result['unchanged']} unchanged, "
              f"{len(result['skipped_modified'])} skipped")
        if result["hint"]:
            print(f"\n{result['hint']}")
        return

    print(json.dumps(result, indent=2))


# ----------------------------------------------------------------- arg parsing

def build_parser() -> argparse.ArgumentParser:
    # --json is declared on a shared parent so that it is accepted both before
    # and after the subcommand. `iann models --json` is the order a caller
    # reaches for first, and argparse rejects it if the flag lives only on the
    # top-level parser.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true",
                        help="emit the raw result as JSON instead of a summary")

    parser = argparse.ArgumentParser(
        prog="iann", parents=[common],
        description="Machine-readable interface to the IANN framework. "
                    "Add --json to any subcommand for parseable output.")
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    sub.add_parser("doctor", parents=[common],
                   help="check whether this environment can run IANN")
    sub.add_parser("models", parents=[common],
                   help="list architectures and which reach LAMMPS")

    fnd = sub.add_parser("foundation", parents=[common],
                         help="released foundation models")
    fnd_sub = fnd.add_subparsers(dest="subcommand", metavar="<action>")
    p = fnd_sub.add_parser("list", parents=[common], help="list the released models")
    p.add_argument("--group", default="painn", choices=["painn", "arch", "all"])
    p = fnd_sub.add_parser("info", parents=[common], help="details for one model, including cache state")
    p.add_argument("name")
    p = fnd_sub.add_parser("fetch", parents=[common], help="resolve to a local path, downloading if needed")
    p.add_argument("name")
    p.add_argument("--local-files-only", action="store_true",
                   help="fail rather than reach the network")

    p = sub.add_parser("inspect", parents=[common], help="what a checkpoint records, and what it is missing")
    p.add_argument("model_path")

    p = sub.add_parser("predict", parents=[common], help="single-point energy and forces")
    p.add_argument("--model", required=True)
    p.add_argument("--structure", required=True)
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--device")
    p.add_argument("--ensemble", nargs="+", metavar="CKPT",
                   help="extra checkpoints; reports variance across the ensemble")
    p.add_argument("--config", help="TOML/JSON file of extra model parameters")

    p = sub.add_parser("train", parents=[common], help="train for a bounded number of steps")
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--max-steps", type=int, required=True, dest="max_steps",
                   help="REQUIRED. The trainer's own default is 1,000,000 steps, "
                        "so a budget must be stated explicitly. PaiNN costs "
                        "roughly 1.6 s/step on CPU, NequIP 4.1 s/step.")
    p.add_argument("--out", dest="output_dir")
    p.add_argument("--config", help="TOML/JSON file of extra trainer/model parameters")

    p = sub.add_parser("status", parents=[common], help="progress of a training run")
    p.add_argument("--output-dir", required=True, dest="output_dir")

    p = sub.add_parser("export", parents=[common], help="export a checkpoint to TorchScript for LAMMPS")
    p.add_argument("--model", required=True)
    p.add_argument("--type", dest="model_type",
                   help="architecture; inferred from the checkpoint when omitted")
    p.add_argument("--out", dest="output_path")
    p.add_argument("--ensemble", nargs="+", metavar="CKPT")
    p.add_argument("--config", help="TOML/JSON file of extra model parameters; "
                                    "see `iann inspect` for what a checkpoint lacks")

    ag = sub.add_parser("agent", parents=[common],
                        help="install the agent notes and Claude Code skills")
    ag_sub = ag.add_subparsers(dest="subcommand", metavar="<action>")
    p = ag_sub.add_parser("install", parents=[common],
                          help="write AGENTS.md, CLAUDE.md and .claude/skills/")
    p.add_argument("--force", action="store_true",
                   help="overwrite files that have been edited")
    p.add_argument("--target", default=".", help="repository root (default: .)")

    # The commands that are only containers for actions. `iann foundation` with
    # no action is a usage error, and printing its help must not exit 0 -- a
    # caller that checks the exit code would read the help text as success.
    parser.iann_group_parsers = {"foundation": fnd, "agent": ag}

    return parser


def _dispatch(args: argparse.Namespace) -> tuple:
    """Return ``(label, result)`` for the parsed arguments."""
    cmd = args.command

    if cmd == "doctor":
        return "doctor", commands.doctor()
    if cmd == "models":
        return "models", commands.list_architectures()

    if cmd == "foundation":
        if args.subcommand == "list":
            return "foundation-list", commands.foundation_list(args.group)
        if args.subcommand == "info":
            return "foundation-info", commands.foundation_info(args.name)
        if args.subcommand == "fetch":
            return "foundation-fetch", commands.foundation_fetch(
                args.name, local_files_only=args.local_files_only)
        raise SystemExit(EXIT_USAGE)

    if cmd == "inspect":
        return "inspect", commands.inspect_checkpoint(args.model_path)

    if cmd == "predict":
        return "predict", commands.predict(
            args.model, args.structure, index=args.index, device=args.device,
            ensemble=args.ensemble, config=_load_config(args.config))

    if cmd == "train":
        return "train", commands.train(
            args.model, args.dataset, args.max_steps,
            output_dir=args.output_dir, config=_load_config(args.config))

    if cmd == "status":
        return "status", commands.training_status(args.output_dir)

    if cmd == "export":
        return "export", commands.export_lammps(
            args.model, args.model_type, args.output_path,
            config=_load_config(args.config), ensemble=args.ensemble)

    if cmd == "agent":
        if args.subcommand == "install":
            from .install import install
            return "agent-install", install(args.target, force=args.force)
        raise SystemExit(EXIT_USAGE)

    raise SystemExit(EXIT_USAGE)


def main(argv=None) -> int:
    parser = build_parser()
    raw = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(argv)

    # Both `iann --json models` and `iann models --json` must work. Because the
    # flag is declared on a shared parent, the subparser re-applies its own
    # default and clears a value set before the subcommand, so the flag is read
    # from the raw arguments rather than trusted from the namespace.
    args.json = getattr(args, "json", False) or "--json" in raw
    if not args.command:
        parser.print_help()
        return EXIT_USAGE
    if getattr(args, "subcommand", "sentinel") is None:
        # A group command such as `iann foundation` with no action. Print the
        # group's own help, but return a usage code rather than argparse's 0.
        group = getattr(parser, "iann_group_parsers", {}).get(args.command)
        (group or parser).print_help()
        return EXIT_USAGE

    try:
        # Under --json, stdout must carry nothing but the JSON document. The
        # library underneath prints progress freely -- the converter announces
        # "Loading model from ..." and the trainer logs every step -- so that
        # output is redirected to stderr, where it stays visible without
        # corrupting what a caller parses.
        if args.json:
            import contextlib
            with contextlib.redirect_stdout(sys.stderr):
                label, result = _dispatch(args)
        else:
            label, result = _dispatch(args)
    except SystemExit:
        raise
    except Exception as exc:                           # noqa: BLE001 - the CLI boundary
        payload = {"error": type(exc).__name__, "message": str(exc)}
        if args.json:
            print(json.dumps(payload, indent=2), file=sys.stderr)
        else:
            print(f"{payload['error']}: {payload['message']}", file=sys.stderr)
        return EXIT_FAIL

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        _print_human(label, result)

    # `doctor` is a check, so its verdict decides the exit code.
    if label == "doctor" and not result["ok"]:
        return EXIT_FAIL
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
