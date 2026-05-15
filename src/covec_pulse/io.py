"""JSONL utilities for reading probe output."""

from __future__ import annotations

import json
from pathlib import Path


def read_probe_jsonl(path: str | Path) -> list[dict]:
    """Read a pulse_probe.jsonl file into a list of records."""
    rows: list[dict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def summary(records: list[dict]) -> dict:
    """Compute basic summary statistics from probe records."""
    if not records:
        return {}
    first, last = records[0], records[-1]
    ratios = [r["hm_am_ratio"] for r in records]
    return {
        "n_records": len(records),
        "step_range": [first["step"], last["step"]],
        "hm_am_ratio_initial": ratios[0],
        "hm_am_ratio_final": ratios[-1],
        "hm_am_ratio_min": min(ratios),
        "hm_am_ratio_max": max(ratios),
        "hm_am_trend": "declining" if ratios[-1] < ratios[0] * 0.5 else "stable",
        "am_initial": first["am"],
        "am_final": last["am"],
        "n_params": first["n_params"],
    }
