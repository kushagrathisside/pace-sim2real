# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

"""Contract between the PACE core and a simulator.

A backend simulates ``num_envs`` copies of the robot in parallel, one per CMA-ES candidate, and exposes only the
identified joints, in the order of ``joint_order``. Joint positions are exchanged in two frames:

* the simulator frame (true joint angle), returned by :meth:`SimulatorBackend.get_joint_positions`,
* the encoder frame (``true - bias``), used for measurements and position targets.

Every backend models the PACE actuator: a PD controller acting on encoder-frame positions, the encoder bias, an
actuator delay in simulation steps, and the joint armature, viscous friction and static / dynamic Coulomb friction.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import torch

from pace_sim2real.core.parameters import JointParameters


class SimulatorBackend(ABC):
    """Minimal simulator interface required by PACE."""

    joint_names: list[str]
    """Identified joints, in the order used by all tensors exchanged with the backend."""

    @property
    @abstractmethod
    def num_envs(self) -> int:
        """Number of robots simulated in parallel."""

    @property
    @abstractmethod
    def device(self) -> torch.device | str:
        """Device of the tensors returned by the backend."""

    @property
    @abstractmethod
    def physics_dt(self) -> float:
        """Duration of one :meth:`step` [s]. The actuator delay is counted in these steps."""

    @abstractmethod
    def reset(self) -> None:
        """Reset the simulation and clear actuator histories (delay buffers) of all environments."""

    @abstractmethod
    def set_joint_parameters(self, params: JointParameters) -> None:
        """Write armature, viscous friction, static / dynamic friction, encoder bias and delay of every environment."""

    @abstractmethod
    def set_joint_state(self, joint_pos: torch.Tensor, joint_vel: torch.Tensor) -> None:
        """Teleport the joints to positions (simulator frame) and velocities, both (num_envs, num_joints)."""

    @abstractmethod
    def step(self, joint_pos_targets: torch.Tensor) -> None:
        """Apply encoder-frame joint position targets (num_envs, num_joints) and advance by :attr:`physics_dt`."""

    @abstractmethod
    def get_joint_positions(self) -> torch.Tensor:
        """Current joint positions (num_envs, num_joints), simulator frame."""

    @abstractmethod
    def get_joint_velocities(self) -> torch.Tensor:
        """Current joint velocities (num_envs, num_joints)."""

    def close(self) -> None:
        """Release simulator resources."""

    def start_rollout(self, params: JointParameters, initial_joint_pos: torch.Tensor) -> None:
        """Prepare a rollout: reset, apply ``params`` and start at rest at ``initial_joint_pos``.

        Args:
            params: Parameters of every environment.
            initial_joint_pos: Initial joint positions in the encoder frame, (num_joints,) or (num_envs, num_joints).
                The joints are placed at ``initial_joint_pos + params.bias`` in the simulator frame.
        """
        self.reset()
        self.set_joint_parameters(params)
        joint_pos = initial_joint_pos.to(self.device).expand(self.num_envs, -1) + params.bias.to(self.device)
        self.set_joint_state(joint_pos, torch.zeros_like(joint_pos))

    def rollout(self, params: JointParameters, initial_joint_pos: torch.Tensor, joint_pos_targets: torch.Tensor,
                on_step: Callable[[int, torch.Tensor], bool | None] | None = None) -> torch.Tensor:
        """Simulate a target trajectory and record the joint positions.

        Positions are recorded *before* each target is applied, so sample ``k`` is the state in which target ``k``
        is commanded (as in the data collection).

        Args:
            params: Parameters of every environment.
            initial_joint_pos: Initial joint positions in the encoder frame, see :meth:`start_rollout`.
            joint_pos_targets: Encoder-frame targets (num_steps, num_joints) or (num_steps, num_envs, num_joints).
            on_step: Optional callback ``on_step(k, joint_pos)`` called with every recorded sample. Returning ``False``
                aborts the rollout.

        Returns:
            Joint positions (num_steps, num_envs, num_joints), simulator frame. Aborted rollouts are truncated.
        """
        self.start_rollout(params, initial_joint_pos)
        targets = joint_pos_targets.to(self.device)
        num_steps = targets.shape[0]
        samples = []
        with torch.inference_mode():
            for k in range(num_steps):
                joint_pos = self.get_joint_positions().clone()
                samples.append(joint_pos)
                if on_step is not None and on_step(k, joint_pos) is False:
                    break
                self.step(targets[k].expand(self.num_envs, -1))
        return torch.stack(samples) if samples else torch.zeros((0, self.num_envs, len(self.joint_names)), device=self.device)
