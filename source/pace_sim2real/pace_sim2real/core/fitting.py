# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

"""Backend-independent PACE fitting loop."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import torch

from .data import TrajectoryData
from .optimizer import CMAESOptimizer

if TYPE_CHECKING:
    from pace_sim2real.backends.base import SimulatorBackend


def fit(backend: SimulatorBackend, optimizer: CMAESOptimizer, data: TrajectoryData | dict,
        should_continue: Callable[[], bool] | None = None, log_interval: int = 400) -> torch.Tensor | None:
    """Identify the PACE parameters by replaying the measured excitation in the simulator.

    Every CMA-ES iteration, each environment of ``backend`` simulates one candidate over the whole trajectory,
    starting at rest at the first measured position. The commanded targets ``des_dof_pos`` are replayed and the
    simulated positions are compared to ``dof_pos``.

    Args:
        backend: Simulator with ``optimizer.population_size`` environments.
        optimizer: Optimizer created for ``data``.
        data: Measured trajectory following the PACE data contract.
        should_continue: Polled every step, the fit stops early when it returns ``False`` (e.g. the simulation app
            was closed).
        log_interval: Print the progress every this many steps (0 disables it).

    Returns:
        Physical parameters of the final CMA-ES mean, or ``None`` if the fit was stopped early.
    """
    if not isinstance(data, TrajectoryData):
        data = TrajectoryData.from_dict(data)
    if optimizer.population_size != backend.num_envs:
        raise ValueError(f"The backend simulates {backend.num_envs} environments, but the population size is {optimizer.population_size}.")
    if data.num_joints != len(backend.joint_names):
        raise ValueError(f"The data has {data.num_joints} joints, but the backend identifies {len(backend.joint_names)}.")

    num_envs = backend.num_envs
    target_dof_pos = data.des_dof_pos.to(backend.device)
    measured_dof_pos = data.dof_pos.to(backend.device)
    initial_dof_pos = measured_dof_pos[0, :].unsqueeze(0).repeat(num_envs, 1)
    time_steps = data.num_steps
    end_time = data.time[-1].item()
    sim_dt = backend.physics_dt

    backend.start_rollout(optimizer.joint_parameters(), initial_dof_pos)

    counter = 0
    while should_continue is None or should_continue():
        # run everything in inference mode
        with torch.inference_mode():
            optimizer.tell(backend.get_joint_positions(), measured_dof_pos[counter, :].unsqueeze(0).repeat(num_envs, 1))
            backend.step(target_dof_pos[counter, :].unsqueeze(0).repeat(num_envs, 1))
            counter += 1
            if log_interval > 0 and counter % log_interval == 0:
                print(f"[INFO]: Step {counter * sim_dt:.1f} / {end_time:.1f} seconds ({counter / time_steps * 100:.1f} %)")
            if counter >= time_steps:
                print("[INFO]: Reached the end of the trajectory, exiting.")
                counter = 0
                optimizer.evolve()
                if optimizer.finished():
                    return optimizer.get_best_sim_params()
                backend.start_rollout(optimizer.joint_parameters(), initial_dof_pos)
    return None
