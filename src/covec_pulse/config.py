"""Configuration loading — env vars > .covec.yaml > defaults."""

from __future__ import annotations

import os
from pathlib import Path


def load_config() -> dict:
    """Load Pulse configuration with priority: env vars > .covec.yaml > defaults."""
    config = {
        "endpoint": None,
        "api_key": None,
        "log_every": 50,
        "scope_every": 100,
        "output_dir": "pulse_outputs",
        "track_per_layer": False,
    }

    yaml_path = Path(".covec.yaml")
    if yaml_path.exists():
        try:
            import yaml
            with open(yaml_path, "r", encoding="utf-8") as f:
                file_config = yaml.safe_load(f) or {}
            if isinstance(file_config, dict):
                for k in config:
                    if k in file_config:
                        config[k] = file_config[k]
        except ImportError:
            pass

    env_map = {
        "COVEC_ENDPOINT": "endpoint",
        "COVEC_API_KEY": "api_key",
        "COVEC_LOG_EVERY": "log_every",
        "COVEC_SCOPE_EVERY": "scope_every",
        "COVEC_OUTPUT_DIR": "output_dir",
    }
    for env_key, config_key in env_map.items():
        val = os.environ.get(env_key)
        if val is not None:
            if config_key in ("log_every", "scope_every"):
                config[config_key] = int(val)
            else:
                config[config_key] = val

    return config


def make_callback_from_config():
    """Create PulseCallback from auto-discovered configuration.

    Usage::

        from covec_pulse.config import make_callback_from_config
        trainer.add_callback(make_callback_from_config())
    """
    from covec_pulse.callback import PulseCallback
    cfg = load_config()
    return PulseCallback(
        log_every=cfg["log_every"],
        output_dir=cfg["output_dir"],
        track_per_layer=cfg["track_per_layer"],
        endpoint=cfg["endpoint"],
        api_key=cfg["api_key"],
        scope_every=cfg["scope_every"],
    )
