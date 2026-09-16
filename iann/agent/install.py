"""Materialise the agent instruction files from the copies shipped in this package.

The repository notes in ``AGENTS.md`` are not specific to any one assistant, but
the *filenames* agents look for are fixed by their tools: ``AGENTS.md`` is the
cross-tool convention, while Claude Code reads only ``CLAUDE.md``. So the single
source at ``iann/agent/AGENTS.md`` is written out under both names, and skills --
a Claude Code feature, discovered at ``<repo>/.claude/skills/<name>/SKILL.md`` --
are copied from ``iann/agent/skills/``.

Keeping the sources inside the subpackage means the whole agent-facing surface is
one reviewable directory, and that the files travel with an installed wheel
rather than only existing in a checkout.

Idempotent: re-running reports ``unchanged`` for files whose content already
matches. A destination that has been edited by hand is left alone unless
``force=True``, so local tweaks are never silently discarded.
"""

from __future__ import annotations

import os
import shutil
from typing import Any, Dict, List

_HERE = os.path.dirname(os.path.abspath(__file__))
SKILLS_DIR = os.path.join(_HERE, "skills")
# A plain .md rather than a .tmpl: nothing is substituted into it, so a template
# extension only obscured what the file is.
NOTES_SOURCE = os.path.join(_HERE, "AGENTS.md")
# One source, two destination names, because the content is tool-neutral but the
# names agents look for are not.
NOTES_DESTINATIONS = ("AGENTS.md", "CLAUDE.md")


def _same(src: str, dst: str) -> bool:
    if not os.path.isfile(dst):
        return False
    with open(src, "rb") as a, open(dst, "rb") as b:
        return a.read() == b.read()


def _copy(src: str, dst: str, force: bool, actions: List[Dict[str, str]]) -> None:
    if _same(src, dst):
        actions.append({"path": dst, "action": "unchanged"})
        return
    if os.path.isfile(dst) and not force:
        actions.append({"path": dst, "action": "skipped-modified"})
        return
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    shutil.copyfile(src, dst)
    actions.append({"path": dst, "action": "written"})


def install(target: str = ".", *, force: bool = False) -> Dict[str, Any]:
    """Copy the agent notes and every skill into ``target``."""
    target = os.path.abspath(target)
    if not os.path.isdir(target):
        raise FileNotFoundError(f"No such directory: {target}")
    if not os.path.isdir(SKILLS_DIR):
        raise FileNotFoundError(
            f"Skill sources are missing from the installed package ({SKILLS_DIR}). "
            "If this is a wheel install, iann.agent.skills was not packaged.")

    actions: List[Dict[str, str]] = []

    if os.path.isfile(NOTES_SOURCE):
        for name in NOTES_DESTINATIONS:
            _copy(NOTES_SOURCE, os.path.join(target, name), force, actions)

    skills: List[str] = []
    for name in sorted(os.listdir(SKILLS_DIR)):
        src = os.path.join(SKILLS_DIR, name, "SKILL.md")
        if not os.path.isfile(src):
            continue
        skills.append(name)
        _copy(src, os.path.join(target, ".claude", "skills", name, "SKILL.md"),
              force, actions)

    skipped = [a["path"] for a in actions if a["action"] == "skipped-modified"]
    return {
        "target": target,
        "skills": skills,
        "actions": actions,
        "written": sum(1 for a in actions if a["action"] == "written"),
        "unchanged": sum(1 for a in actions if a["action"] == "unchanged"),
        "skipped_modified": skipped,
        "hint": ("Some destinations differ from the packaged copies and were left "
                 "alone; re-run with --force to overwrite them.") if skipped else None,
    }
