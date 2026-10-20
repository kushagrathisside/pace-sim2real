# © 2026 Kushagra Srivastava
# Licensed under the Apache License 2.0

"""MuJoCo backend for PACE (CPU, no Isaac Sim required).

Mapping of the PACE parameters onto MuJoCo:

==================  ==========================================================================================
PACE parameter      MuJoCo
==================  ==========================================================================================
armature            ``dof_armature``
viscous friction    ``dof_damping`` (passive force ``-damping * qvel``)
static friction     ``dof_frictionloss`` while ``|qvel| < static_friction_velocity``
dynamic friction    ``dof_frictionloss`` otherwise (MuJoCo itself has a single dry friction coefficient)
encoder bias        offset of the measured position fed to the PD controller (:class:`PaceDCMotorModel`)
delay               delay line on the commands or the torque (:class:`PaceDCMotorModel`)
==================  ==========================================================================================

The actuator torque is applied through ``qfrc_applied``. Actuators defined in the MJCF are disabled, as are contacts
(the robot is identified hanging with a fixed base, without self-collisions).
"""

from __future__ import annotations

import mujoco
import numpy as np
import torch

from pace_sim2real.backends.actuator import PaceDCMotorModel, PaceDCMotorParams
from pace_sim2real.backends.base import SimulatorBackend
from pace_sim2real.core.parameters import JointParameters

from .model import joint_indices, replicate


class MujocoBackend(SimulatorBackend):
    """PACE backend simulating ``num_envs`` copies of an MJCF robot in one MuJoCo model.

    Args:
        model: MJCF file path or XML string of a single robot.
        joint_order: Names of the identified joints (hinge or slide joints of the MJCF).
        num_envs: Number of robots simulated in parallel (the CMA-ES population size when fitting).
        actuator: PD gains, torque limits and delay settings of the PACE actuator.
        physics_dt: Simulation time step [s], defaults to the time step of the MJCF.
        fix_base: Weld floating bases to the world.
        disable_contacts: Disable all contacts (the copies of the robot overlap otherwise).
        static_friction_velocity: Joint speed [rad/s] below which the static instead of the dynamic friction acts.
        env_spacing: Distance between the copies [m], only relevant for visualization.
    """

    def __init__(self, model, joint_order: list[str], num_envs: int, actuator: PaceDCMotorParams | None = None,
                 physics_dt: float | None = None, fix_base: bool = True, disable_contacts: bool = True,
                 static_friction_velocity: float = 1e-2, env_spacing: float = 2.0):
        self.joint_names = list(joint_order)
        self._num_envs = num_envs
        self.model = replicate(model, num_envs, physics_dt=physics_dt, fix_base=fix_base, env_spacing=env_spacing)
        self.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_ACTUATION
        if disable_contacts:
            self.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
        self.data = mujoco.MjData(self.model)
        self._scratch_data = mujoco.MjData(self.model)
        self.qpos_ids, self.dof_ids = joint_indices(self.model, self.joint_names, num_envs)
        self.actuator = PaceDCMotorModel(actuator or PaceDCMotorParams(), num_envs, len(self.joint_names))
        self.static_friction_velocity = static_friction_velocity
        self._static_friction = self.model.dof_frictionloss[self.dof_ids].copy()
        self._dynamic_friction = self._static_friction.copy()

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def device(self) -> str:
        return "cpu"

    @property
    def physics_dt(self) -> float:
        return float(self.model.opt.timestep)

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.actuator.reset()
        mujoco.mj_forward(self.model, self.data)

    def set_joint_parameters(self, params: JointParameters) -> None:
        if params.num_envs != self.num_envs or params.num_joints != len(self.joint_names):
            raise ValueError(f"Expected parameters for {self.num_envs} environments and {len(self.joint_names)} joints, "
                             f"got {params.num_envs} and {params.num_joints}.")
        self.model.dof_armature[self.dof_ids] = _numpy(params.armature)
        self.model.dof_damping[self.dof_ids] = _numpy(params.damping)
        # the constraint solver scales limits and dry friction with the inertia at qpos0, which includes the armature
        mujoco.mj_setConst(self.model, self._scratch_data)
        self._static_friction = _numpy(params.static_friction)
        self._dynamic_friction = _numpy(params.dynamic_friction)
        self.model.dof_frictionloss[self.dof_ids] = self._static_friction
        self.actuator.update_encoder_bias(_numpy(params.bias))
        self.actuator.update_time_lags(_numpy(params.delay).astype(np.int64))
        self.actuator.reset()

    def set_joint_state(self, joint_pos: torch.Tensor, joint_vel: torch.Tensor) -> None:
        self.data.qpos[self.qpos_ids] = _numpy(joint_pos)
        self.data.qvel[self.dof_ids] = _numpy(joint_vel)
        mujoco.mj_forward(self.model, self.data)

    def step(self, joint_pos_targets: torch.Tensor) -> None:
        joint_pos = self.data.qpos[self.qpos_ids]
        joint_vel = self.data.qvel[self.dof_ids]
        sticking = np.abs(joint_vel) < self.static_friction_velocity
        self.model.dof_frictionloss[self.dof_ids] = np.where(sticking, self._static_friction, self._dynamic_friction)
        torque = self.actuator.compute(_numpy(joint_pos_targets), joint_pos, joint_vel)
        self.data.qfrc_applied[self.dof_ids] = torque
        mujoco.mj_step(self.model, self.data)

    def get_joint_positions(self) -> torch.Tensor:
        return torch.from_numpy(self.data.qpos[self.qpos_ids]).to(torch.float32)

    def get_joint_velocities(self) -> torch.Tensor:
        return torch.from_numpy(self.data.qvel[self.dof_ids]).to(torch.float32)


def _numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().double().numpy()
    return np.asarray(value, dtype=np.float64)
