"""Low-level probe function — works with any training loop."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass

import torch


_EPS = 1e-30
SCHEMA_VERSION = "1"


@dataclass(frozen=True)
class LayerSnapshot:
    am: float
    hm: float
    hm_am: float
    n: int


@dataclass(frozen=True)
class ProbeSnapshot:
    step: int
    timestamp: float
    am: float
    hm: float
    hm_am_ratio: float
    n_params: int
    n_tensors: int
    lr: float | None = None
    beta2: float | None = None
    epsilon: float | None = None
    weight_decay: float | None = None
    per_layer: dict[str, LayerSnapshot] | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["schema_version"] = SCHEMA_VERSION
        if d["per_layer"]:
            d["per_layer"] = {k: asdict(v) for k, v in (self.per_layer or {}).items()}
        else:
            d.pop("per_layer", None)
        for k in ("lr", "beta2", "epsilon", "weight_decay"):
            if d.get(k) is None:
                d.pop(k, None)
        return d


def _default_filter(name: str, param: torch.Tensor) -> bool:
    if param.ndim < 2:
        return False
    lo = name.lower()
    if "embed" in lo or "norm" in lo:
        return False
    return "attn" in lo or "mlp" in lo or "self_attn" in lo


def _layer_id(name: str) -> str:
    parts = name.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
            return f"layer_{parts[i + 1]}"
    return "other"


def probe_optimizer(
    optimizer: torch.optim.Optimizer,
    step: int = 0,
    param_filter: callable | None = None,
    track_per_layer: bool = False,
) -> ProbeSnapshot | None:
    """Read optimizer second moments and return a health snapshot.

    Works with any Adam-family optimizer that stores ``exp_avg_sq``.
    """
    filt = param_filter or _default_filter

    total_sum = 0.0
    total_inv_sum = 0.0
    total_params = 0
    total_tensors = 0
    layer_stats: dict[str, dict] = {}

    for group in optimizer.param_groups:
        for param in group["params"]:
            state = optimizer.state.get(param)
            if state is None or "exp_avg_sq" not in state:
                continue
            name = getattr(param, "_param_name", f"param_{id(param)}")
            if not filt(name, param):
                continue

            v = state["exp_avg_sq"].detach().to(dtype=torch.float64, device="cpu").flatten()
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
        return None

    am = total_sum / total_params
    hm = total_params / total_inv_sum if total_inv_sum > 0 else 0.0
    ratio = hm / am if am > 0 else 0.0

    opt_lr = opt_beta2 = opt_eps = opt_wd = None
    if optimizer.param_groups:
        g0 = optimizer.param_groups[0]
        opt_lr = g0.get("lr")
        betas = g0.get("betas")
        if betas and len(betas) >= 2:
            opt_beta2 = betas[1]
        opt_eps = g0.get("eps")
        opt_wd = g0.get("weight_decay")

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
        step=step,
        timestamp=time.time(),
        am=am,
        hm=hm,
        hm_am_ratio=ratio,
        n_params=total_params,
        n_tensors=total_tensors,
        lr=opt_lr,
        beta2=opt_beta2,
        epsilon=opt_eps,
        weight_decay=opt_wd,
        per_layer=per_layer,
    )
