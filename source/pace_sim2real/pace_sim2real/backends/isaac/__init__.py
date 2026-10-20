# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

"""Isaac Lab backend.

The adapter itself does not import Isaac Lab. The Isaac Lab environment and actuator configurations are re-exported
lazily from :mod:`pace_sim2real.tasks` and :mod:`pace_sim2real.utils` and require Isaac Lab when accessed.
"""

import importlib

from .adapter import IsaacLabBackend, apply_pace_parameters, write_actuator_parameters, write_joint_parameters

_LAZY_EXPORTS = {
    "PaceSim2realEnvCfg": "pace_sim2real.tasks.manager_based.pace.pace_sim2real_env_cfg",
    "PaceSim2realSceneCfg": "pace_sim2real.tasks.manager_based.pace.pace_sim2real_env_cfg",
    "PaceCfg": "pace_sim2real.tasks.manager_based.pace.pace_sim2real_env_cfg",
    "CMAESOptimizerCfg": "pace_sim2real.tasks.manager_based.pace.pace_sim2real_env_cfg",
    "PaceDCMotorCfg": "pace_sim2real.utils.pace_actuator_cfg",
    "PaceDCMotor": "pace_sim2real.utils.pace_actuator",
}


def __getattr__(name):
    if name in _LAZY_EXPORTS:
        return getattr(importlib.import_module(_LAZY_EXPORTS[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "IsaacLabBackend",
    "apply_pace_parameters",
    "write_actuator_parameters",
    "write_joint_parameters",
    *_LAZY_EXPORTS,
]
