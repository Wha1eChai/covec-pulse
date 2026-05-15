"""Basic tests for covec-pulse probe."""

from __future__ import annotations

import math

import torch

from covec_pulse.core import probe_optimizer, ProbeSnapshot


def _make_adam_with_state(param_sizes, v_values):
    """Create a minimal Adam optimizer with known exp_avg_sq values."""
    params = []
    for size, v_val in zip(param_sizes, v_values):
        p = torch.randn(size, requires_grad=True)
        p._param_name = "model.layers.0.self_attn.q_proj.weight"
        params.append(p)

    optimizer = torch.optim.Adam(params, lr=1e-4)
    optimizer.step()

    for param, v_val in zip(params, v_values):
        state = optimizer.state[param]
        state["exp_avg_sq"] = torch.full(param.shape, v_val, dtype=torch.float32)

    return optimizer


class TestProbeOptimizer:
    def test_returns_snapshot(self):
        opt = _make_adam_with_state([(4, 4)], [1e-6])
        snap = probe_optimizer(opt, step=10)
        assert snap is not None
        assert isinstance(snap, ProbeSnapshot)
        assert snap.step == 10
        assert snap.n_params == 16
        assert snap.n_tensors == 1

    def test_isotropic_ratio_is_one(self):
        opt = _make_adam_with_state([(8, 8)], [0.5])
        snap = probe_optimizer(opt, step=1)
        assert snap is not None
        assert abs(snap.hm_am_ratio - 1.0) < 1e-10

    def test_anisotropic_ratio_below_one(self):
        p = torch.randn(8, 8, requires_grad=True)
        p._param_name = "model.layers.0.self_attn.q_proj.weight"
        opt = torch.optim.Adam([p], lr=1e-4)
        opt.step()
        v = torch.ones(8, 8)
        v[0, 0] = 1e-10
        opt.state[p]["exp_avg_sq"] = v
        snap = probe_optimizer(opt, step=1)
        assert snap is not None
        assert snap.hm_am_ratio < 0.01

    def test_skips_1d_params(self):
        p1d = torch.randn(64, requires_grad=True)
        p1d._param_name = "model.layers.0.self_attn.bias"
        opt = torch.optim.Adam([p1d], lr=1e-4)
        opt.step()
        opt.state[p1d]["exp_avg_sq"] = torch.ones(64)
        snap = probe_optimizer(opt, step=1)
        assert snap is None

    def test_skips_embedding(self):
        p = torch.randn(32000, 128, requires_grad=True)
        p._param_name = "model.embed_tokens.weight"
        opt = torch.optim.Adam([p], lr=1e-4)
        opt.step()
        opt.state[p]["exp_avg_sq"] = torch.ones(32000, 128)
        snap = probe_optimizer(opt, step=1)
        assert snap is None

    def test_hyperparams_captured(self):
        opt = _make_adam_with_state([(4, 4)], [1e-6])
        snap = probe_optimizer(opt, step=1)
        assert snap is not None
        assert snap.lr == 1e-4
        assert snap.beta2 is not None
        assert snap.epsilon is not None

    def test_to_dict(self):
        opt = _make_adam_with_state([(4, 4)], [1e-6])
        snap = probe_optimizer(opt, step=1)
        d = snap.to_dict()
        assert isinstance(d, dict)
        assert "step" in d
        assert "am" in d
        assert "per_layer" not in d

    def test_per_layer(self):
        opt = _make_adam_with_state([(4, 4)], [1e-6])
        snap = probe_optimizer(opt, step=1, track_per_layer=True)
        assert snap is not None
        assert snap.per_layer is not None
        assert "layer_0" in snap.per_layer
