# © 2026 Kushagra Srivastava
# Licensed under the Apache License 2.0

"""NumPy reference implementation of the PACE DC motor actuator, for backends without an actuator framework.

It reproduces ``PaceDCMotor`` of the Isaac Lab backend: a PD controller acting on encoder-frame positions
(``q - encoder_bias``), a delay of the commands or of the computed torque, and the DC motor torque-speed limits of
Isaac Lab's ``DCMotor``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass
class PaceDCMotorParams:
    """Fixed (not identified) actuator properties, scalars or one value per joint."""

    stiffness: float | Sequence[float] = 85.0
    """P gain [Nm/rad]."""
    damping: float | Sequence[float] = 0.6
    """D gain [Nm s/rad]."""
    effort_limit: float | Sequence[float] = np.inf
    """Continuous torque limit [Nm]."""
    saturation_effort: float | Sequence[float] = np.inf
    """Peak motor torque at zero velocity [Nm]."""
    velocity_limit: float | Sequence[float] = np.inf
    """No-load speed of the motor [rad/s]."""
    max_delay: int = 10
    """Largest delay that can be identified [simulation steps]."""
    delay_mode: str = "command"
    """"command" delays the targets, "effort" delays the computed torque (see ``PaceDCMotorCfg.delay_mode``)."""


class DelayBuffer:
    """Per-environment delay line with the semantics of Isaac Lab's ``DelayBuffer``.

    The output at step ``k`` is the input of step ``max(k - lag, 0)``: right after a reset the buffer is filled with
    the first input.
    """

    def __init__(self, max_delay: int, num_envs: int, num_joints: int):
        self.max_delay = max_delay
        self._buffer = np.zeros((max_delay + 1, num_envs, num_joints))
        self._env_ids = np.arange(num_envs)
        self._lags = np.zeros(num_envs, dtype=np.int64)
        self._pointer = 0
        self._num_pushes = 0

    def set_time_lag(self, lags) -> None:
        lags = np.broadcast_to(np.asarray(lags, dtype=np.int64), self._lags.shape)
        if np.any(lags < 0) or np.any(lags > self.max_delay):
            raise ValueError(f"Delays must lie in [0, {self.max_delay}] steps, got {lags.min()} to {lags.max()}.")
        self._lags = lags.copy()

    def reset(self) -> None:
        self._num_pushes = 0

    def compute(self, data: np.ndarray) -> np.ndarray:
        if self._num_pushes == 0:
            self._buffer[:] = data
            self._pointer = 0
        else:
            self._pointer = (self._pointer + 1) % self._buffer.shape[0]
            self._buffer[self._pointer] = data
        self._num_pushes += 1
        lags = np.minimum(self._lags, self._num_pushes - 1)
        return self._buffer[(self._pointer - lags) % self._buffer.shape[0], self._env_ids].copy()


class PaceDCMotorModel:
    """Batched PACE DC motor for ``num_envs`` robots with ``num_joints`` actuated joints each."""

    def __init__(self, params: PaceDCMotorParams, num_envs: int, num_joints: int):
        if params.delay_mode not in ("command", "effort"):
            raise ValueError(f"delay_mode must be 'command' or 'effort', but got '{params.delay_mode}'.")
        self.params = params
        self.num_envs = num_envs
        self.num_joints = num_joints
        shape = (num_joints,)
        self.stiffness = np.broadcast_to(np.asarray(params.stiffness, dtype=np.float64), shape).copy()
        self.damping = np.broadcast_to(np.asarray(params.damping, dtype=np.float64), shape).copy()
        self.effort_limit = np.broadcast_to(np.asarray(params.effort_limit, dtype=np.float64), shape).copy()
        self.saturation_effort = np.broadcast_to(np.asarray(params.saturation_effort, dtype=np.float64), shape).copy()
        self.velocity_limit = np.broadcast_to(np.asarray(params.velocity_limit, dtype=np.float64), shape).copy()
        self.encoder_bias = np.zeros((num_envs, num_joints))
        # one buffer for the torque ("effort") or one per command signal ("command")
        names = ("effort",) if params.delay_mode == "effort" else ("position", "velocity", "effort")
        self.delay_buffers = {name: DelayBuffer(params.max_delay, num_envs, num_joints) for name in names}
        self.computed_effort = np.zeros((num_envs, num_joints))
        self.applied_effort = np.zeros((num_envs, num_joints))

    def update_encoder_bias(self, encoder_bias: np.ndarray) -> None:
        self.encoder_bias = np.broadcast_to(np.asarray(encoder_bias, dtype=np.float64), (self.num_envs, self.num_joints)).copy()

    def update_time_lags(self, delay) -> None:
        for buffer in self.delay_buffers.values():
            buffer.set_time_lag(delay)

    def reset(self) -> None:
        for buffer in self.delay_buffers.values():
            buffer.reset()

    def compute(self, joint_pos_target: np.ndarray, joint_pos: np.ndarray, joint_vel: np.ndarray,
                joint_vel_target: np.ndarray | None = None, joint_effort_target: np.ndarray | None = None) -> np.ndarray:
        """Joint torques (num_envs, num_joints) from encoder-frame targets and the simulator-frame joint state."""
        joint_vel_target = np.zeros_like(joint_pos_target) if joint_vel_target is None else joint_vel_target
        joint_effort_target = np.zeros_like(joint_pos_target) if joint_effort_target is None else joint_effort_target
        if self.params.delay_mode == "command":
            joint_pos_target = self.delay_buffers["position"].compute(joint_pos_target)
            joint_vel_target = self.delay_buffers["velocity"].compute(joint_vel_target)
            joint_effort_target = self.delay_buffers["effort"].compute(joint_effort_target)
        # PD control on the joint position in the encoder frame, not the simulation frame
        error_pos = joint_pos_target - (joint_pos - self.encoder_bias)
        error_vel = joint_vel_target - joint_vel
        self.computed_effort = self.stiffness * error_pos + self.damping * error_vel + joint_effort_target
        self.applied_effort = self._clip_effort(self.computed_effort, joint_vel)
        if self.params.delay_mode == "effort":
            self.applied_effort = self.delay_buffers["effort"].compute(self.applied_effort)
        return self.applied_effort

    def _clip_effort(self, effort: np.ndarray, joint_vel: np.ndarray) -> np.ndarray:
        """Torque-speed limits of a DC motor, as in Isaac Lab's ``DCMotor``."""
        with np.errstate(invalid="ignore"):
            max_effort = self.saturation_effort * (1.0 - joint_vel / self.velocity_limit)
            min_effort = self.saturation_effort * (-1.0 - joint_vel / self.velocity_limit)
        max_effort = np.clip(np.nan_to_num(max_effort, nan=0.0), 0.0, self.effort_limit)
        min_effort = np.clip(np.nan_to_num(min_effort, nan=0.0), -self.effort_limit, 0.0)
        return np.clip(effort, min_effort, max_effort)
