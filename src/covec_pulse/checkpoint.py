"""Diagnose training health from a saved checkpoint — no retraining needed."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from covec_pulse.core import ProbeSnapshot, _EPS, _default_filter, _layer_id, LayerSnapshot

import time


def _find_optimizer_state(checkpoint_dir: Path) -> Path | None:
    for name in ("optimizer.pt", "optimizer.bin"):
        p = checkpoint_dir / name
        if p.exists():
            return p
    return None


def _find_model_state(checkpoint_dir: Path) -> Path | None:
    for name in ("pytorch_model.bin", "model.safetensors"):
        p = checkpoint_dir / name
        if p.exists():
            return p
    if (checkpoint_dir / "pytorch_model.bin.index.json").exists():
        return checkpoint_dir / "pytorch_model.bin.index.json"
    if (checkpoint_dir / "model.safetensors.index.json").exists():
        return checkpoint_dir / "model.safetensors.index.json"
    return None


def _build_param_id_to_name(checkpoint_dir: Path) -> dict[int, str] | None:
    """Map optimizer param indices to parameter names via model state dict key order."""
    model_path = _find_model_state(checkpoint_dir)
    if model_path is None:
        return None

    if model_path.name.endswith(".index.json"):
        index = json.loads(model_path.read_text(encoding="utf-8"))
        names = list(index.get("weight_map", {}).keys())
    elif model_path.name.endswith(".safetensors"):
        try:
            from safetensors import safe_open
            with safe_open(str(model_path), framework="pt") as f:
                names = list(f.keys())
        except ImportError:
            names = list(torch.load(model_path, map_location="cpu", weights_only=True).keys())
    else:
        names = list(torch.load(model_path, map_location="cpu", weights_only=True).keys())

    return {i: name for i, name in enumerate(names)}


def diagnose_checkpoint(
    checkpoint_dir: str | Path,
    param_filter: callable | None = None,
    track_per_layer: bool = True,
) -> ProbeSnapshot | None:
    """Analyze a HuggingFace Trainer checkpoint and return a health snapshot.

    The checkpoint directory should contain ``optimizer.pt`` and either
    ``pytorch_model.bin`` or ``model.safetensors``.

    No retraining needed — this reads the optimizer state as-is.
    """
    checkpoint_dir = Path(checkpoint_dir)
    filt = param_filter or _default_filter

    opt_path = _find_optimizer_state(checkpoint_dir)
    if opt_path is None:
        raise FileNotFoundError(f"No optimizer.pt found in {checkpoint_dir}")

    opt_state = torch.load(opt_path, map_location="cpu", weights_only=False)
    param_states = opt_state.get("state", {})
    if not param_states:
        raise ValueError("Optimizer state is empty")

    id_to_name = _build_param_id_to_name(checkpoint_dir)

    total_sum = 0.0
    total_inv_sum = 0.0
    total_params = 0
    total_tensors = 0
    layer_stats: dict[str, dict] = {}
    skipped = 0

    for param_id, state in param_states.items():
        if "exp_avg_sq" not in state:
            continue

        v_raw = state["exp_avg_sq"]
        name = id_to_name.get(int(param_id), f"param_{param_id}") if id_to_name else f"param_{param_id}"

        if v_raw.ndim < 2:
            continue
        if id_to_name and not filt(name, v_raw):
            skipped += 1
            continue

        v = v_raw.to(dtype=torch.float64).flatten()
        v_safe = torch.clamp(v, min=_EPS)

        s = v.sum().item()
        inv_s = torch.reciprocal(v_safe).sum().item()
        n = v.numel()

        total_sum += s
        total_inv_sum += inv_s
        total_params += n
        total_tensors += 1

        if track_per_layer:
            lid = _layer_id(name)
            if lid not in layer_stats:
                layer_stats[lid] = {"sum": 0.0, "inv_sum": 0.0, "count": 0}
            layer_stats[lid]["sum"] += s
            layer_stats[lid]["inv_sum"] += inv_s
            layer_stats[lid]["count"] += n

    if total_params == 0:
        if id_to_name is None:
            raise ValueError(
                "No model state found — cannot filter params by name. "
                "Make sure the checkpoint directory contains pytorch_model.bin or model.safetensors."
            )
        raise ValueError(f"No matching params found ({skipped} skipped by filter)")

    am = total_sum / total_params
    hm = total_params / total_inv_sum if total_inv_sum > 0 else 0.0
    ratio = hm / am if am > 0 else 0.0

    per_layer = None
    if track_per_layer and layer_stats:
        per_layer = {}
        for lid, st in sorted(layer_stats.items()):
            if st["count"] > 0 and st["inv_sum"] > 0:
                am_l = st["sum"] / st["count"]
                hm_l = st["count"] / st["inv_sum"]
                per_layer[lid] = LayerSnapshot(
                    am=am_l, hm=hm_l, hm_am=hm_l / am_l if am_l > 0 else 0.0, n=st["count"],
                )

    return ProbeSnapshot(
        step=0,
        timestamp=time.time(),
        am=am,
        hm=hm,
        hm_am_ratio=ratio,
        n_params=total_params,
        n_tensors=total_tensors,
        per_layer=per_layer,
    )
