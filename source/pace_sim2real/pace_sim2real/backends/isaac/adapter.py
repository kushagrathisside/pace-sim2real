# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

"""Isaac Lab backend.

The functions here only call methods of Isaac Lab objects that are passed in (environment, articulation, actuators),
so this module does not import Isaac Lab itself. The Isaac Lab environment must use :class:`PaceDCMotorCfg` actuators
and a joint position action over all joints, as :class:`PaceSim2realEnvCfg` does.
"""

from __future__ import annotations

import torch

from pace_sim2real.backends.base import SimulatorBackend
from pace_sim2real.core.parameters import JointParameters


def write_joint_parameters(articulation, joint_ids: torch.Tensor, params: JointParameters) -> None:
    """Write armature, viscous friction and static / dynamic friction of every environment into the simulator."""
    env_ids = torch.arange(len(params.armature))
    articulation.write_joint_armature_to_sim(params.armature, joint_ids=joint_ids, env_ids=env_ids)
    articulation.data.default_joint_armature[:, joint_ids] = params.armature
    articulation.write_joint_viscous_friction_coefficient_to_sim(params.damping, joint_ids=joint_ids, env_ids=env_ids)
    articulation.data.default_joint_viscous_friction_coeff[:, joint_ids] = params.damping
    # If we set static friction lower than dynamic friction, the sim complains. So we need to do this weird order.
    articulation.write_joint_dynamic_friction_coefficient_to_sim(0.0, joint_ids=joint_ids, env_ids=env_ids)
    articulation.write_joint_friction_coefficient_to_sim(params.static_friction, joint_ids=joint_ids, env_ids=env_ids)
    articulation.data.default_joint_friction_coeff[:, joint_ids] = params.static_friction
    articulation.write_joint_dynamic_friction_coefficient_to_sim(params.dynamic_friction, joint_ids=joint_ids, env_ids=env_ids)
    articulation.data.default_joint_dynamic_friction_coeff[:, joint_ids] = params.dynamic_friction


def write_actuator_parameters(articulation, joint_ids: torch.Tensor, params: JointParameters) -> None:
    """Write encoder bias and delay into the PACE actuators and reset their delay buffers."""
    env_ids = torch.arange(len(params.armature))
    for drive_type in articulation.actuators.keys():
        drive_indices = articulation.actuators[drive_type].joint_indices
        if isinstance(drive_indices, slice):
            all_idx = torch.arange(joint_ids.shape[0], device=joint_ids.device)
            drive_indices = all_idx[drive_indices]
        comparison_matrix = (joint_ids.unsqueeze(1) == drive_indices.unsqueeze(0))
        drive_joint_idx = torch.argmax(comparison_matrix.int(), dim=0)
        articulation.actuators[drive_type].update_encoder_bias(params.bias[:, drive_joint_idx])
        articulation.actuators[drive_type].update_time_lags(params.delay)
        articulation.actuators[drive_type].reset(env_ids)


def apply_pace_parameters(articulation, joint_ids: torch.Tensor, params: JointParameters,
                          initial_position: torch.Tensor) -> None:
    """Write the parameters and place the joints at rest at ``initial_position`` (encoder frame) plus the bias.

    This is the write sequence of the original ``CMAESOptimizer.update_simulator``.
    """
    write_joint_parameters(articulation, joint_ids, params)
    articulation.write_joint_position_to_sim(initial_position + params.bias, joint_ids=joint_ids)
    articulation.write_joint_velocity_to_sim(torch.zeros_like(initial_position), joint_ids=joint_ids)
    write_actuator_parameters(articulation, joint_ids, params)


class IsaacLabBackend(SimulatorBackend):
    """PACE backend wrapping an Isaac Lab ``ManagerBasedRLEnv`` (e.g. created with ``gym.make``).

    Args:
        env: The (possibly wrapped) Isaac Lab environment, one environment per CMA-ES candidate.
        joint_order: Names of the identified joints.
        articulation_name: Name of the robot in the interactive scene.
    """

    def __init__(self, env, joint_order: list[str], articulation_name: str = "robot"):
        self.env = env
        self._unwrapped = env.unwrapped
        self.articulation = self._unwrapped.scene[articulation_name]
        self.joint_names = list(joint_order)
        self.joint_ids = torch.tensor([self.articulation.joint_names.index(name) for name in joint_order], device=self._unwrapped.device)

    @property
    def num_envs(self) -> int:
        return self._unwrapped.num_envs

    @property
    def device(self):
        return self._unwrapped.device

    @property
    def physics_dt(self) -> float:
        return self._unwrapped.sim.cfg.dt

    def reset(self) -> None:
        self.env.reset()

    def set_joint_parameters(self, params: JointParameters) -> None:
        write_joint_parameters(self.articulation, self.joint_ids, params)
        write_actuator_parameters(self.articulation, self.joint_ids, params)

    def set_joint_state(self, joint_pos: torch.Tensor, joint_vel: torch.Tensor) -> None:
        self.articulation.write_joint_position_to_sim(joint_pos, joint_ids=self.joint_ids)
        self.articulation.write_joint_velocity_to_sim(joint_vel, joint_ids=self.joint_ids)

    def start_rollout(self, params: JointParameters, initial_joint_pos: torch.Tensor) -> None:
        self.reset()
        initial_joint_pos = initial_joint_pos.to(self.device)
        if initial_joint_pos.ndim == 1:
            initial_joint_pos = initial_joint_pos.unsqueeze(0).repeat(self.num_envs, 1)
        apply_pace_parameters(self.articulation, self.joint_ids, params, initial_joint_pos)

    def step(self, joint_pos_targets: torch.Tensor) -> None:
        # the action term commands absolute position targets of all joints in articulation order
        actions = torch.zeros(self.env.action_space.shape, device=self.device)
        actions[:, self.joint_ids] = joint_pos_targets
        self.env.step(actions)

    def get_joint_positions(self) -> torch.Tensor:
        return self.articulation.data.joint_pos[:, self.joint_ids]

    def get_joint_velocities(self) -> torch.Tensor:
        return self.articulation.data.joint_vel[:, self.joint_ids]

    def close(self) -> None:
        self.env.close()
