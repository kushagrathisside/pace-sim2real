# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

"""PACE objective: squared joint position tracking error between simulation and measurements.

Simulated joint positions live in the simulator (true joint) frame, whereas measured positions live in the encoder
frame, which is offset by the encoder bias. The residual is therefore ``q_sim - q_measured - bias``.
"""

from __future__ import annotations

import torch


def step_error(sim_dof_pos: torch.Tensor, real_dof_pos: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Squared position error of one time step, summed over joints.

    Args:
        sim_dof_pos: Simulated joint positions (num_envs, num_joints), simulator frame.
        real_dof_pos: Measured joint positions (num_envs, num_joints) or (num_joints,), encoder frame.
        bias: Encoder bias of every candidate (num_envs, num_joints).

    Returns:
        Error per candidate (num_envs,).
    """
    return torch.sum(torch.square(sim_dof_pos - real_dof_pos - bias), dim=-1)


def trajectory_score(sim_dof_pos: torch.Tensor, real_dof_pos: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """PACE score of whole trajectories: :func:`step_error` averaged over time.

    Args:
        sim_dof_pos: Simulated trajectories (num_steps, num_envs, num_joints), simulator frame.
        real_dof_pos: Measured trajectory (num_steps, num_joints), encoder frame.
        bias: Encoder bias of every candidate (num_envs, num_joints).

    Returns:
        Score per candidate (num_envs,), lower is better.
    """
    return step_error(sim_dof_pos, real_dof_pos.unsqueeze(1), bias.unsqueeze(0)).mean(dim=0)


def trajectory_rmse(sim_dof_pos: torch.Tensor, real_dof_pos: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Root mean square position error per candidate and joint (num_envs, num_joints), same inputs as :func:`trajectory_score`."""
    return torch.sqrt(torch.mean(torch.square(sim_dof_pos - real_dof_pos.unsqueeze(1) - bias.unsqueeze(0)), dim=0))
