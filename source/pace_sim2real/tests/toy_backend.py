# © 2026 Kushagra Srivastava
# Licensed under the Apache License 2.0

"""Simulator-free backend for testing the PACE core: independent pendulum joints integrated with PyTorch."""

from __future__ import annotations

import numpy as np
import torch

from pace_sim2real.backends.actuator import PaceDCMotorModel, PaceDCMotorParams
from pace_sim2real.backends.base import SimulatorBackend
from pace_sim2real.core.parameters import JointParameters


class ToyPendulumBackend(SimulatorBackend):
    """Every joint is a damped pendulum ``(I + armature) qdd = tau - damping qd - friction(qd) - m g l sin(q)``.

    Coulomb friction is smoothed with ``tanh`` and switches from static to dynamic above ``static_friction_velocity``.
    """

    def __init__(self, joint_names: list[str], num_envs: int, actuator: PaceDCMotorParams | None = None,
                 physics_dt: float = 0.0025, inertia: float = 0.02, gravity_torque: float = 1.0,
                 static_friction_velocity: float = 1e-2):
        self.joint_names = list(joint_names)
        self._num_envs = num_envs
        self._dt = physics_dt
        self.inertia = inertia
        self.gravity_torque = gravity_torque
        self.static_friction_velocity = static_friction_velocity
        shape = (num_envs, len(joint_names))
        self.q = torch.zeros(shape, dtype=torch.float64)
        self.qd = torch.zeros(shape, dtype=torch.float64)
        self.params: JointParameters | None = None
        self.actuator = PaceDCMotorModel(actuator or PaceDCMotorParams(stiffness=20.0, damping=0.5), num_envs, len(joint_names))
        self.num_resets = 0
        self.closed = False

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def device(self) -> str:
        return "cpu"

    @property
    def physics_dt(self) -> float:
        return self._dt

    def reset(self) -> None:
        self.q.zero_()
        self.qd.zero_()
        self.actuator.reset()
        self.num_resets += 1

    def set_joint_parameters(self, params: JointParameters) -> None:
        self.params = JointParameters(*(getattr(params, name).detach().clone().cpu() for name in JointParameters._fields()))
        self.actuator.update_encoder_bias(self.params.bias.double().numpy())
        self.actuator.update_time_lags(self.params.delay.numpy().astype(np.int64))
        self.actuator.reset()

    def set_joint_state(self, joint_pos: torch.Tensor, joint_vel: torch.Tensor) -> None:
        self.q[:] = joint_pos.double()
        self.qd[:] = joint_vel.double()

    def step(self, joint_pos_targets: torch.Tensor) -> None:
        p = self.params
        tau = torch.from_numpy(self.actuator.compute(joint_pos_targets.double().cpu().numpy(), self.q.numpy(), self.qd.numpy()))
        friction = torch.where(self.qd.abs() < self.static_friction_velocity, p.static_friction, p.dynamic_friction).double()
        tau = tau - p.damping.double() * self.qd - friction * torch.tanh(self.qd / 1e-3) - self.gravity_torque * torch.sin(self.q)
        self.qd += self._dt * tau / (self.inertia + p.armature.double())
        self.q += self._dt * self.qd

    def get_joint_positions(self) -> torch.Tensor:
        return self.q.float()

    def get_joint_velocities(self) -> torch.Tensor:
        return self.qd.float()

    def close(self) -> None:
        self.closed = True
