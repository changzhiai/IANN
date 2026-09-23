"""Checks for the ``iann`` CLI: every subcommand, the failure paths, and one
real end-to-end loop.

Run from the repository root::

    python iann/agent/tests/test_cli.py
    python iann/agent/tests/test_cli.py --fast   # skip the bounded training run

The negative cases are the point of this file. A tool that an agent drives has
to fail loudly: the converter used to return ``None`` and print on a missing
checkpoint, which any caller would read as success. So each failure mode is
asserted to produce both a non-zero exit code and a JSON error object.

Fixtures are the ones already in the repository: ``test/Pt_ads.traj`` (tracked)
and ``test/painn/output/model.pt`` (produced by ``test/painn/train.py``; the
checks that need it are skipped when it is absent, since it is gitignored).

Exit codes: 0 pass, 1 fail.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

from iann.agent.cli import EXIT_FAIL, EXIT_OK, EXIT_USAGE          # noqa: E402

TRAJ = "test/Pt_ads.traj"
PAINN_CKPT = "test/painn/output/model.pt"

_failures = []
_skipped = []


def check(label, condition, detail=""):
    if condition:
        print(f"  ok   {label}")
    else:
        print(f"  FAIL {label}" + (f" -- {detail}" if detail else ""))
        _failures.append(label)
    return bool(condition)


def skip(label, why):
    print(f"  skip {label} -- {why}")
    _skipped.append(label)


def run(args, timeout=900):
    """Invoke the CLI as a subprocess so the exit code is the real one.

    ``-m iann.agent.cli`` rather than the ``iann`` console script: the script
    only exists after the package is (re)installed, and these checks must work
    in a fresh checkout.
    """
    proc = subprocess.run(
        [sys.executable, "-m", "iann.agent.cli"] + args,
        cwd=ROOT, capture_output=True, text=True, timeout=timeout,
        env={**os.environ, "PYTHONPATH": ROOT},
    )
    payload = None
    for stream in (proc.stdout, proc.stderr):
        payload = _extract_json(stream)
        if payload is not None:
            break
    return proc.returncode, payload, proc.stdout, proc.stderr


def _extract_json(stream):
    """Parse the JSON document in a stream, tolerating leading log lines.

    Success JSON goes to stdout and is the only thing there. Error JSON goes to
    stderr, which also carries whatever the library printed on its way to
    failing, so the document has to be found rather than assumed.
    """
    if not stream or not stream.strip():
        return None
    try:
        return json.loads(stream)
    except json.JSONDecodeError:
        pass
    lines = stream.splitlines()
    for start in range(len(lines)):
        if lines[start].startswith(("{", "[")):
            try:
                return json.loads("\n".join(lines[start:]))
            except json.JSONDecodeError:
                continue
    return None


def has_keys(payload, keys):
    if not isinstance(payload, dict):
        return False, f"not an object: {str(payload)[:80]}"
    missing = [k for k in keys if k not in payload]
    return (not missing), f"missing {missing}"


# ------------------------------------------------------------ read-only commands

def test_doctor():
    print("\n[doctor]")
    rc, payload, out, err = run(["doctor", "--json"])
    ok, detail = has_keys(payload, ["ok", "python", "executable", "platform",
                                    "dependencies", "cuda", "architectures",
                                    "broken", "degraded", "hint"])
    check("doctor --json carries the documented keys", ok, detail)
    if not isinstance(payload, dict):
        return
    # The verdict decides the exit code; that is what makes it usable in CI.
    expected = EXIT_OK if payload.get("ok") else EXIT_FAIL
    check("doctor's exit code follows its verdict", rc == expected,
          f"ok={payload.get('ok')} rc={rc}")
    check("doctor reports each dependency separately",
          isinstance(payload.get("dependencies"), dict)
          and len(payload["dependencies"]) >= 5,
          str(list((payload.get("dependencies") or {}).keys())))
    for name, dep in (payload.get("dependencies") or {}).items():
        sub_ok, sub_detail = has_keys(dep, ["ok", "version", "error", "warnings"])
        if not check(f"dependency {name} is fully described", sub_ok, sub_detail):
            break
    # asap3 is the one that silently breaks in the `base` env, so it must be
    # probed by name rather than inferred from an `import iann` succeeding.
    check("asap3 is probed by name", "asap3" in (payload.get("dependencies") or {}))

    rc, _, out, _ = run(["doctor"])
    check("doctor's human output is a summary, not raw JSON",
          "environment :" in out and not out.lstrip().startswith("{"), out[:120])


def test_models():
    print("\n[models]")
    rc, payload, _, _ = run(["models", "--json"])
    check("models exits 0", rc == EXIT_OK)
    ok, detail = has_keys(payload, ["architectures", "count"])
    check("models --json carries the documented keys", ok, detail)
    if not isinstance(payload, dict):
        return
    archs = payload["architectures"]
    check("the count matches the list", payload["count"] == len(archs))
    for arch in archs:
        sub_ok, detail = has_keys(arch, ["name", "lammps_exportable",
                                         "unpersisted_config"])
        if not check(f"architecture {arch.get('name')} is fully described",
                     sub_ok, detail):
            break
    names = [a["name"] for a in archs]
    check("painn is listed", "painn" in names, str(names))
    # FastPot is the one architecture with no export path; if that ever flips
    # silently, the export skill starts lying.
    fastpot = [a for a in archs if a["name"] == "fastpot"]
    if fastpot:
        check("fastpot is marked as not exportable",
              fastpot[0]["lammps_exportable"] is False)


def test_foundation():
    print("\n[foundation]")
    rc, payload, _, _ = run(["foundation", "list", "--json"])
    check("foundation list exits 0", rc == EXIT_OK)
    ok, detail = has_keys(payload, ["models", "count", "group"])
    check("foundation list --json carries the documented keys", ok, detail)
    if isinstance(payload, dict) and payload.get("models"):
        sub_ok, detail = has_keys(payload["models"][0],
                                  ["name", "functional", "dataset",
                                   "energy_mae", "forces_mae"])
        check("each catalog row carries the metrics needed to choose", sub_ok, detail)
        check("the painn group is the nine released potentials",
              payload["count"] == 9, f"got {payload['count']}")

    for group in ("arch", "all"):
        rc, payload, _, _ = run(["foundation", "list", "--group", group, "--json"])
        check(f"foundation list --group {group} exits 0", rc == EXIT_OK)
        check(f"group {group} returns rows",
              isinstance(payload, dict) and payload.get("count", 0) > 0)

    rc, payload, _, _ = run(["foundation", "info", "rpbe-oc20", "--json"])
    check("foundation info exits 0", rc == EXIT_OK)
    ok, detail = has_keys(payload, ["name", "functional", "dataset", "repo_id",
                                    "cached", "local_path"])
    check("foundation info --json carries the documented keys", ok, detail)

    # Names are accepted case-insensitively; agents will not match our casing.
    rc, payload, _, _ = run(["foundation", "info", "RPBE-OC20", "--json"])
    check("foundation info is case-insensitive", rc == EXIT_OK,
          f"rc={rc} {str(payload)[:120]}")


def test_inspect():
    print("\n[inspect]")
    if not os.path.exists(os.path.join(ROOT, PAINN_CKPT)):
        skip("inspect", f"{PAINN_CKPT} is gitignored; run test/painn/train.py")
        return
    rc, payload, _, _ = run(["inspect", PAINN_CKPT, "--json"])
    check("inspect exits 0", rc == EXIT_OK)
    ok, detail = has_keys(payload, ["path", "architecture", "model_type_declared",
                                    "model_type_inferred", "n_parameter_tensors",
                                    "metadata", "missing_config"])
    check("inspect --json carries the documented keys", ok, detail)
    if not isinstance(payload, dict):
        return
    check("inspect infers the architecture", payload["architecture"] == "painn",
          str(payload.get("architecture")))
    check("inspect counts parameter tensors",
          isinstance(payload["n_parameter_tensors"], int)
          and payload["n_parameter_tensors"] > 0)
    # Reporting the gap is the whole reason this subcommand exists.
    check("missing_config is a list", isinstance(payload["missing_config"], list))


def test_predict():
    print("\n[predict]")
    if not os.path.exists(os.path.join(ROOT, PAINN_CKPT)):
        skip("predict", f"{PAINN_CKPT} is gitignored; run test/painn/train.py")
        return
    rc, payload, _, _ = run(["predict", "--model", PAINN_CKPT,
                             "--structure", TRAJ, "--json"])
    check("predict exits 0", rc == EXIT_OK)
    ok, detail = has_keys(payload, ["energy", "energy_per_atom", "forces",
                                    "formula", "n_atoms", "max_force", "index"])
    check("predict --json carries the documented keys", ok, detail)
    if not isinstance(payload, dict):
        return
    check("the forces array has one vector per atom",
          len(payload["forces"]) == payload["n_atoms"],
          f"{len(payload['forces'])} vs {payload['n_atoms']}")
    check("each force is a 3-vector", all(len(f) == 3 for f in payload["forces"]))
    check("energy_per_atom is consistent with energy",
          abs(payload["energy"] / payload["n_atoms"]
              - payload["energy_per_atom"]) < 1e-6)

    # A different frame must give a different structure, not a cached first one.
    rc, other, _, _ = run(["predict", "--model", PAINN_CKPT, "--structure", TRAJ,
                           "--index", "10", "--json"])
    check("--index selects a different frame",
          rc == EXIT_OK and isinstance(other, dict) and other["index"] == 10,
          str(other)[:120] if other else "")

    # An ensemble of the same checkpoint twice must report zero spread.
    rc, ens, _, _ = run(["predict", "--model", PAINN_CKPT, "--structure", TRAJ,
                         "--ensemble", PAINN_CKPT, "--json"])
    if rc == EXIT_OK and isinstance(ens, dict):
        check("an ensemble reports how many models it used",
              ens.get("n_models") == 2, str(ens.get("n_models")))
        check("a self-ensemble has near-zero energy variance",
              abs(ens.get("ensemble", {}).get("energy_var", 1.0)) < 1e-6,
              str(ens.get("ensemble"))[:120])
    else:
        check("ensemble prediction exits 0", False, str(ens)[:160])


def test_json_flag_positions():
    print("\n[--json in both positions]")
    rc_after, after, _, _ = run(["models", "--json"])
    rc_before, before, _, _ = run(["--json", "models"])
    check("iann models --json is accepted", rc_after == EXIT_OK)
    check("iann --json models is accepted", rc_before == EXIT_OK)
    check("both positions produce the same JSON", after == before,
          "the subparser default overwrote the flag")


# ---------------------------------------------------------------- failure paths

def test_failures():
    print("\n[failure paths] loud, structured, non-zero")

    rc, payload, out, err = run(["export", "--model", "/nonexistent/model.pt",
                                 "--json"])
    check("export on a missing checkpoint exits 1", rc == EXIT_FAIL, f"rc={rc}")
    check("export on a missing checkpoint returns a JSON error",
          isinstance(payload, dict) and "error" in payload and "message" in payload,
          (err or out)[:160])
    check("the error goes to stderr", "error" in err, err[:120])

    # A missing required flag is argparse's job, and it must be usage, not
    # failure -- the two mean different things to a caller.
    rc, _, _, err = run(["train", "--model", "painn", "--dataset", TRAJ])
    check("train without --max-steps exits 2 (usage)", rc == EXIT_USAGE, f"rc={rc}")
    check("the usage error names --max-steps", "max-steps" in err, err[-200:])

    rc, payload, out, err = run(["foundation", "info", "nosuchmodel", "--json"])
    check("an unknown foundation model exits 1", rc == EXIT_FAIL, f"rc={rc}")
    check("it reports UnknownFoundationModelError",
          isinstance(payload, dict)
          and payload.get("error") == "UnknownFoundationModelError",
          str(payload)[:160])
    # With no close match, the message enumerates the catalog instead, so an
    # agent can recover from a bad guess in one step either way.
    check("it enumerates the available models",
          isinstance(payload, dict) and "Available" in payload.get("message", ""),
          str(payload)[:200] if payload else "")

    rc, payload, _, _ = run(["foundation", "info", "rpbe-al", "--json"])
    check("a near-miss name exits 1", rc == EXIT_FAIL, f"rc={rc}")
    check("a near-miss name offers a did-you-mean list",
          isinstance(payload, dict) and "Did you mean" in payload.get("message", ""),
          str(payload)[:200] if payload else "")

    rc, payload, _, err = run(["train", "--model", "painn", "--dataset", TRAJ,
                               "--max-steps", "0", "--json"])
    check("a non-positive step budget exits 1", rc == EXIT_FAIL, f"rc={rc}")
    check("it reports ValueError",
          isinstance(payload, dict) and payload.get("error") == "ValueError",
          str(payload)[:160])

    rc, _, out, err = run([])
    check("no subcommand exits 2 and prints help", rc == EXIT_USAGE
          and "usage: iann" in (out + err), f"rc={rc}")

    rc, _, out, err = run(["foundation"])
    check("a group command with no action exits 2", rc == EXIT_USAGE, f"rc={rc}")

    rc, payload, _, err = run(["inspect", "/nonexistent/model.pt", "--json"])
    check("inspect on a missing file exits 1", rc == EXIT_FAIL, f"rc={rc}")
    check("inspect's failure is structured",
          isinstance(payload, dict) and "error" in payload, err[:160])


# ------------------------------------------------------------------ round trip

def test_round_trip():
    """train -> status -> inspect -> predict -> export, asserting each artefact."""
    print("\n[round trip] a real bounded run")
    workdir = tempfile.mkdtemp(prefix="iann-agent-roundtrip-")
    try:
        started = time.time()
        rc, payload, out, err = run(["train", "--model", "painn",
                                     "--dataset", TRAJ, "--max-steps", "2",
                                     "--out", workdir, "--json"])
        elapsed = time.time() - started
        if not check("a 2-step run exits 0", rc == EXIT_OK,
                     (err or out)[-400:]):
            return
        ok, detail = has_keys(payload, ["model", "max_steps", "progress",
                                        "final_metrics", "model_path",
                                        "manifest_path"])
        check("train --json carries the documented keys", ok, detail)
        # The trainer logs every step to stdout; under --json it must not.
        check("train --json puts nothing but JSON on stdout",
              _extract_json(out) is not None and json.loads(out) == payload,
              out[:200])
        check("a bounded run is actually quick", elapsed < 600,
              f"{elapsed:.0f}s")
        if not isinstance(payload, dict):
            return

        check("the run stopped at the budget",
              payload["progress"].get("state") == "max_steps_reached",
              str(payload["progress"])[:160])
        check("the budget is echoed back", payload["max_steps"] == 2)

        # The manifest is what makes a run reproducible from the outside.
        manifest_path = payload["manifest_path"]
        check("run.json exists", os.path.exists(manifest_path), manifest_path)
        if os.path.exists(manifest_path):
            with open(manifest_path) as fh:
                manifest = json.load(fh)
            ok, detail = has_keys(manifest, ["model", "dataset", "max_steps",
                                             "config", "final_metrics",
                                             "model_path"])
            check("run.json carries the config and the metrics", ok, detail)
            check("run.json is self-consistent with the result",
                  manifest.get("max_steps") == payload["max_steps"])

        ckpt = payload["model_path"]
        check("the checkpoint was written", os.path.exists(ckpt), ckpt)

        rc, status, _, err = run(["status", "--output-dir", workdir, "--json"])
        check("status exits 0 on the finished run", rc == EXIT_OK, err[:160])
        ok, detail = has_keys(status, ["output_dir", "state", "step", "metrics",
                                       "log_path", "log_found"])
        check("status --json carries the documented keys", ok, detail)
        if isinstance(status, dict):
            check("status found the log", status.get("log_found") is True)
            check("status reports the terminal state",
                  status.get("state") == "max_steps_reached",
                  str(status.get("state")))
            check("status reports an integer step",
                  isinstance(status.get("step"), int), str(status.get("step")))

        if os.path.exists(ckpt):
            rc, insp, _, err = run(["inspect", ckpt, "--json"])
            check("inspect reads the fresh checkpoint", rc == EXIT_OK, err[:160])
            check("inspect identifies it as painn",
                  isinstance(insp, dict) and insp.get("architecture") == "painn",
                  str(insp)[:120] if insp else "")

            rc, pred, _, err = run(["predict", "--model", ckpt,
                                    "--structure", TRAJ, "--json"])
            check("predict works on the fresh checkpoint", rc == EXIT_OK, err[:160])
            check("predict returns a finite energy",
                  isinstance(pred, dict)
                  and isinstance(pred.get("energy"), float)
                  and pred["energy"] == pred["energy"],
                  str(pred)[:120] if pred else "")

            exported = os.path.join(workdir, "lammps_model.pt")
            rc, exp, out, err = run(["export", "--model", ckpt,
                                     "--out", exported, "--json"])
            check("export exits 0", rc == EXIT_OK, err[-300:])
            ok, detail = has_keys(exp, ["output_path", "size_bytes"])
            check("export --json carries the documented keys", ok, detail)
            check("the exported file exists and is non-empty",
                  os.path.exists(exported) and os.path.getsize(exported) > 0,
                  exported)
            # The converter prints "Loading model from ..." as it works. Under
            # --json that must not reach stdout, or the document a caller
            # parses is preceded by prose.
            check("export --json puts nothing but JSON on stdout",
                  _extract_json(out) is not None and json.loads(out) == exp,
                  out[:200])
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


# --------------------------------------------------------------- agent install

def test_agent_install():
    print("\n[agent install] idempotent, and refuses to clobber edits")
    target = tempfile.mkdtemp(prefix="iann-agent-install-")
    try:
        rc, first, _, err = run(["agent", "install", "--target", target, "--json"])
        if not check("the first install exits 0", rc == EXIT_OK, err[:200]):
            return
        ok, detail = has_keys(first, ["target", "actions", "written", "unchanged",
                                      "skipped_modified", "hint"])
        check("agent install --json carries the documented keys", ok, detail)
        if not isinstance(first, dict):
            return

        # The notes are tool-neutral but the filenames agents look for are not,
        # so one source is written out under both.
        agents_md = os.path.join(target, "AGENTS.md")
        claude_md = os.path.join(target, "CLAUDE.md")
        check("AGENTS.md is written to the repository root",
              os.path.exists(agents_md))
        check("CLAUDE.md is written to the repository root",
              os.path.exists(claude_md))
        if os.path.exists(agents_md) and os.path.exists(claude_md):
            with open(agents_md) as a, open(claude_md) as b:
                check("both names carry identical content", a.read() == b.read())
        skills_dir = os.path.join(target, ".claude", "skills")
        check(".claude/skills/ is created", os.path.isdir(skills_dir))
        installed = sorted(os.listdir(skills_dir)) if os.path.isdir(skills_dir) else []
        check("all five skills are installed", len(installed) == 5, str(installed))
        for name in installed:
            path = os.path.join(skills_dir, name, "SKILL.md")
            if not check(f"{name}/SKILL.md exists", os.path.exists(path)):
                break
            with open(path) as fh:
                head = fh.read(400)
            # Claude Code discovers skills by their YAML frontmatter; without it
            # the file is installed but invisible.
            if not check(f"{name} has name and description frontmatter",
                         head.startswith("---") and "name:" in head
                         and "description:" in head, head[:80]):
                break
        check("the first install reports files written", first["written"] > 0)

        rc, second, _, err = run(["agent", "install", "--target", target, "--json"])
        check("the second install exits 0", rc == EXIT_OK, err[:200])
        if isinstance(second, dict):
            check("the second install writes nothing",
                  second["written"] == 0
                  and second["unchanged"] == first["written"],
                  f"written={second['written']} unchanged={second['unchanged']}")
            check("the second install reports the same file set",
                  len(second["actions"]) == len(first["actions"]))

        # An edited file must survive a reinstall -- a user's notes are theirs.
        with open(claude_md, "a") as fh:
            fh.write("\n<!-- a local edit -->\n")
        rc, third, _, err = run(["agent", "install", "--target", target, "--json"])
        check("installing over an edited file exits 0", rc == EXIT_OK, err[:200])
        if isinstance(third, dict):
            check("the edited file is skipped, not overwritten",
                  any(claude_md in p for p in third["skipped_modified"]),
                  str(third["skipped_modified"]))
        with open(claude_md) as fh:
            check("the local edit is still there", "a local edit" in fh.read())

        rc, forced, _, err = run(["agent", "install", "--target", target,
                                  "--force", "--json"])
        check("--force exits 0", rc == EXIT_OK, err[:200])
        if isinstance(forced, dict):
            check("--force overwrites the edited file", forced["written"] > 0,
                  str(forced)[:200])
        with open(claude_md) as fh:
            check("the local edit is gone after --force",
                  "a local edit" not in fh.read())

        rc, _, out, _ = run(["agent", "install", "--target", target])
        check("agent install's human output is a summary, not raw JSON",
              "target" in out and not out.lstrip().startswith("{"), out[:120])
    finally:
        shutil.rmtree(target, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Check the iann CLI")
    parser.add_argument("--fast", action="store_true",
                        help="skip the bounded training round trip")
    args = parser.parse_args()

    print("=" * 66)
    print("iann CLI checks".center(66))
    print("=" * 66)

    test_doctor()
    test_models()
    test_foundation()
    test_inspect()
    test_predict()
    test_json_flag_positions()
    test_failures()
    test_agent_install()
    if args.fast:
        skip("round trip", "--fast")
    else:
        test_round_trip()

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
