# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

"""
Utility functions and actuator models for PACE.

The Isaac Lab actuator models are imported on first access and require Isaac Lab.
"""

import importlib

from .paths import project_root

_LAZY_ISAAC_EXPORTS = {
    "PaceDCMotorCfg": ".pace_actuator_cfg",
    "PaceDCMotor": ".pace_actuator",
}


def __getattr__(name):
    if name in _LAZY_ISAAC_EXPORTS:
        value = getattr(importlib.import_module(_LAZY_ISAAC_EXPORTS[name], __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "PaceDCMotorCfg",
    "PaceDCMotor",
    "project_root",
]
