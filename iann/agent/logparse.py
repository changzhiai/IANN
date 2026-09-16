"""Parse a training run's ``output.log`` into structured progress.

The trainer reports progress only as formatted log lines, so this is the only
way to observe a run from outside the process. Two properties of that log shape
the parser:

* It is opened in append mode, so one file can hold several concatenated
  sessions. Everything here reports the **last** session's state.
* Zero-valued metrics are dropped before the line is formatted, so the set of
  ``key=value`` pairs varies between runs and between architectures. Metrics are
  therefore parsed as a mapping, never by position.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional

# 2026-05-02 01:41:46 [RANK0] [INFO ]  step=49, energy_mae=0.141, ...
_STEP_LINE = re.compile(r"^\s*(?P<ts>\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\s+\[RANK(?P<rank>\d+)\]"
                        r"\s+\[\w+\s*\]\s+(?P<body>step=.*)$")
_KV = re.compile(r"([A-Za-z_()][\w()]*(?:\s+\w+)*?)\s*=\s*(-?[\d.]+(?:e[-+]?\d+)?)")

# Terminal states, most specific first; the message text is the trainer's own.
_TERMINAL = [
    ("nan_exit", "NaN values detected in best_val_loss"),
    ("early_stopped", "Early stopping, training complete"),
    ("max_steps_reached", "reached, training complete"),
]


def parse_metrics(body: str) -> Dict[str, float]:
    """Pull ``key=value`` pairs out of one progress line's body.

    Keys such as ``sqrt(total_loss)`` and ``training time`` contain parentheses
    and spaces, so the pattern is deliberately permissive; values are always
    numeric. Units in the original line (``... time=0.026 min``) are dropped.

    Spaces in keys are replaced with underscores (``training time`` becomes
    ``training_time``) so a caller can address them as ordinary identifiers.
    Nothing else about the names is changed.
    """
    out: Dict[str, float] = {}
    for key, value in _KV.findall(body):
        try:
            out[key.strip().replace(" ", "_")] = float(value)
        except ValueError:
            continue
    return out


def parse_log(log_path: str) -> Dict[str, object]:
    """Summarise the most recent training session in ``log_path``."""
    if not os.path.isfile(log_path):
        return {"log_path": log_path, "log_found": False, "state": "no_log",
                "step": None, "metrics": {}, "sessions": 0}

    with open(log_path, "r", errors="replace") as fh:
        lines = fh.readlines()

    # Each session re-emits the configuration banner; count them so a caller
    # knows the file is cumulative rather than a single run.
    sessions = sum(1 for l in lines if "Configuration Settings" in l)

    last_body: Optional[str] = None
    last_ts: Optional[str] = None
    steps_seen: List[int] = []
    for line in lines:
        m = _STEP_LINE.match(line)
        if m:
            last_body, last_ts = m.group("body"), m.group("ts")
            sm = re.search(r"step=(\d+)", last_body)
            if sm:
                steps_seen.append(int(sm.group(1)))

    state = "running"
    tail = "".join(lines[-40:])
    for name, needle in _TERMINAL:
        if needle in tail:
            state = name
            break

    metrics = parse_metrics(last_body) if last_body else {}
    # `step` is reported as its own integer field rather than left among the
    # float metrics.
    metrics.pop("step", None)
    return {
        "log_path": log_path,
        "log_found": True,
        "state": state,
        "step": steps_seen[-1] if steps_seen else None,
        "last_log_time": last_ts,
        "metrics": metrics,
        "sessions": sessions,
        "steps_logged": len(steps_seen),
    }
