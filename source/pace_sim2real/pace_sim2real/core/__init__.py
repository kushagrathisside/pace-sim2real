# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

"""Simulator-independent PACE core: parameter model, objective, CMA-ES optimizer, data contract and validation.

This package must not import any simulator (Isaac Lab, Isaac Sim, MuJoCo, ...). Simulators are accessed through
:class:`pace_sim2real.backends.base.SimulatorBackend`.
"""

from .data import TrajectoryData, linear_chirp, shape_excitation
from .fitting import fit
from .logger import NullWriter, create_writer
from .objective import step_error, trajectory_rmse, trajectory_score
from .optimizer import CMAESOptimizer
from .parameters import (
    JointParameters,
    ParameterLayout,
    extend_bounds_with_friction_ratio,
    normalized_to_physical,
    physical_to_normalized,
    sim_params_from_values,
)
from .validation import (
    ParameterError,
    RecoveryResult,
    format_report,
    generate_synthetic_data,
    max_normalized_error,
    recovery_report,
    synthetic_recovery,
    within_bounds,
)

__all__ = [
    "CMAESOptimizer",
    "JointParameters",
    "NullWriter",
    "ParameterError",
    "ParameterLayout",
    "RecoveryResult",
    "TrajectoryData",
    "create_writer",
    "extend_bounds_with_friction_ratio",
    "fit",
    "format_report",
    "generate_synthetic_data",
    "linear_chirp",
    "max_normalized_error",
    "normalized_to_physical",
    "physical_to_normalized",
    "recovery_report",
    "shape_excitation",
    "sim_params_from_values",
    "synthetic_recovery",
    "step_error",
    "trajectory_rmse",
    "trajectory_score",
    "within_bounds",
]
