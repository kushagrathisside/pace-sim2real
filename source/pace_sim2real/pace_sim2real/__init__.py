# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

"""
PACE: Precise Adaptation through Continuous Evolution

Public Python API for the pace_sim2real package.

The optimizer and the rest of :mod:`pace_sim2real.core` do not need a simulator. The Isaac Lab configurations below
are imported on first access and require Isaac Lab.
"""

import importlib

# Optimizer
from .optim import CMAESOptimizer

_LAZY_ISAAC_EXPORTS = {
    "PaceSim2realEnvCfg": ".tasks.manager_based.pace.pace_sim2real_env_cfg",
    "PaceSim2realSceneCfg": ".tasks.manager_based.pace.pace_sim2real_env_cfg",
    "PaceCfg": ".tasks.manager_based.pace.pace_sim2real_env_cfg",
}


def __getattr__(name):
    if name in _LAZY_ISAAC_EXPORTS:
        value = getattr(importlib.import_module(_LAZY_ISAAC_EXPORTS[name], __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "PaceSim2realEnvCfg",
    "PaceSim2realSceneCfg",
    "PaceCfg",
    "CMAESOptimizer"
]
