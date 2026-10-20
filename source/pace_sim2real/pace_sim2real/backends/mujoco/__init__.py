# © 2026 Kushagra Srivastava
# Licensed under the Apache License 2.0

"""MuJoCo backend for PACE: CPU-only parameter identification and synthetic validation without Isaac Sim."""

try:
    import mujoco  # noqa: F401
except ImportError as error:
    raise ImportError("The MuJoCo backend requires the 'mujoco' package: pip install -e source/pace_sim2real[mujoco]") from error

from pace_sim2real.backends.actuator import DelayBuffer, PaceDCMotorModel, PaceDCMotorParams

from .adapter import MujocoBackend
from .model import joint_indices, load_spec, replicate, serial_chain_xml

__all__ = [
    "DelayBuffer",
    "MujocoBackend",
    "PaceDCMotorModel",
    "PaceDCMotorParams",
    "joint_indices",
    "load_spec",
    "replicate",
    "serial_chain_xml",
]
