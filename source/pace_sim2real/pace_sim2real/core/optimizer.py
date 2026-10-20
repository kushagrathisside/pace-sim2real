# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

"""Simulator-independent CMA-ES optimizer for PACE.

The optimizer samples a population of parameter vectors, collects the tracking error of every candidate over a
trajectory (:meth:`CMAESOptimizer.tell`) and updates the search distribution once per trajectory
(:meth:`CMAESOptimizer.evolve`). Writing the candidates into a simulator is the job of a
:class:`~pace_sim2real.backends.base.SimulatorBackend`, which receives :meth:`CMAESOptimizer.joint_parameters`.
"""

from __future__ import annotations

import os
from datetime import datetime

import cmaes
import torch

from .data import TrajectoryData
from .logger import create_writer
from .objective import step_error
from .parameters import JointParameters, ParameterLayout, normalized_to_physical


class CMAESOptimizer:
    def __init__(self, bounds, population_size, log_dir, joint_order, max_iteration, data, device, epsilon=None, sigma=0.5, save_interval=10, save_optimization_process=False, separate_dynamic_friction=False,
                 seed=0, use_tensorboard=True):

        if isinstance(data, TrajectoryData):
            data = data.as_dict()
        self.joint_order = joint_order
        self.max_iteration = max_iteration
        self.epsilon = epsilon
        self.save_interval = save_interval
        self.device = device
        self.save_optimization_process = save_optimization_process
        self.separate_dynamic_friction = separate_dynamic_friction
        self._timer_start = datetime.now()  # timer for logging purposes

        num_joints = len(joint_order)
        self.layout = ParameterLayout(num_joints, separate_dynamic_friction)
        self.armature_idx = self.layout.armature
        self.damping_idx = self.layout.damping
        self.friction_idx = self.layout.friction
        self.bias_idx = self.layout.bias
        self.delay_idx = self.layout.delay
        self.friction_ratio_idx = self.layout.friction_ratio
        self.layout.validate_bounds(bounds)

        # create log_dir in YY_MM_DD_hh-mm-ss format
        folder_time = datetime.now().strftime("%y_%m_%d_%H-%M-%S")
        # create string with time and date
        log_dir = os.path.join(log_dir, folder_time)
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.writer = create_writer(log_dir, use_tensorboard)
        torch.save({"bounds": bounds,
                    "joint_order": joint_order,
                    "dof_pos": data["dof_pos"],
                    "des_dof_pos": data["des_dof_pos"],
                    "time": data["time"],
                    "separate_dynamic_friction": separate_dynamic_friction,
                    }, os.path.join(log_dir, "config.pt"))

        self.bounds = bounds

        bounds_normalized = torch.ones_like(bounds)
        bounds_normalized[:, 0] *= -1
        mean_normalized = torch.zeros_like(bounds[:, 0])

        self.optimizer = cmaes.CMA(mean=mean_normalized.cpu().numpy(), sigma=sigma, bounds=bounds_normalized.cpu().numpy(), seed=seed, population_size=population_size)

        self.scores_counter = 0
        self.iteration_counter = 0

        self.scores = torch.zeros(population_size, device=device)
        self.scores_buffer = torch.zeros((max_iteration, population_size), device=device)
        self.sim_dof_pos_buffer = torch.zeros((population_size, data["dof_pos"].shape[0], num_joints), device=device)

        self.params = torch.zeros((population_size, bounds.shape[0]), device=device)
        self.sim_params = torch.zeros_like(self.params)
        if save_optimization_process:
            self.sim_params_buffer = torch.zeros((max_iteration, population_size, bounds.shape[0]), device=device)

        self._reset_population()
        print("CMA-ES optimizer initialized.")
        print("Current iteration: ", self.iteration_counter)

    @property
    def population_size(self) -> int:
        return self.optimizer.population_size

    def ask(self):
        return self.optimizer.ask()

    def tell(self, sim_dof_pos, real_dof_pos):
        """Accumulate the tracking error of one time step.

        Args:
            sim_dof_pos: Simulated joint positions (population_size, num_joints), simulator frame.
            real_dof_pos: Measured joint positions, encoder frame, broadcastable to ``sim_dof_pos``.
        """
        self.scores += step_error(sim_dof_pos, real_dof_pos, self.sim_params[:, self.bias_idx])
        self.sim_dof_pos_buffer[:, self.scores_counter, :] = sim_dof_pos
        self.scores_counter += 1

    def evolve(self):
        self.scores /= self.scores_counter
        self.scores_buffer[self.iteration_counter, :] = self.scores
        if self.save_optimization_process:
            self.sim_params_buffer[self.iteration_counter, :, :] = self.sim_params
        solutions = []
        for i in range(self.optimizer.population_size):
            solutions.append((self.params[i].cpu().numpy(), self.scores[i].item()))
        self.optimizer.tell(solutions)
        if self.save_interval > 0 and self.iteration_counter % self.save_interval == 0:
            self.save_checkpoint(self._params_to_sim_params(torch.tensor(self.optimizer._mean, device=self.device)), self.iteration_counter)
        self._print_iteration()

        self._reset_population()

        self.scores = torch.zeros_like(self.scores)
        self.scores_counter = 0
        self.iteration_counter += 1
        print("CMA-ES optimizer iteration: ", self.iteration_counter)

    def finished(self):
        finished = self.max_iteration <= self.iteration_counter
        diff_score = (self.scores_buffer[self.iteration_counter - 1, :].max() - self.scores_buffer[self.iteration_counter - 1, :].min()) / self.scores_buffer[self.iteration_counter - 1, :].min()
        finished = finished or (self.epsilon is not None and diff_score < self.epsilon)
        if finished:
            print("CMA-ES optimization finished.")
            self.save_checkpoint(self._params_to_sim_params(torch.tensor(self.optimizer._mean, device=self.device)), self.iteration_counter - 1, finished=True)
        return finished

    def _reset_population(self):
        for i in range(self.optimizer.population_size):
            self.params[i, :] = torch.tensor(self.optimizer.ask(), device=self.device)
        self.sim_params = self._params_to_sim_params(self.params)

    def joint_parameters(self) -> JointParameters:
        """Physical parameters of the current population, to be written into the simulator by a backend."""
        return JointParameters.from_sim_params(self.sim_params, self.layout)

    def dynamic_friction(self):
        """Dynamic Coulomb friction per env and joint, equal to the static friction unless identified separately."""
        if self.separate_dynamic_friction:
            return self.sim_params[:, self.friction_idx] * self.sim_params[:, self.friction_ratio_idx]
        return self.sim_params[:, self.friction_idx]

    def _print_iteration(self):
        min_score = torch.min(self.scores)
        max_score = torch.max(self.scores)
        min_index = torch.argmin(self.scores)
        print("Max score: ", max_score.item())
        print("Min score: ", min_score.item(), " at index: ", min_index.item())
        print("Armature: ", self.sim_params[min_index, self.armature_idx].tolist())
        print("Viscous Friction: ", self.sim_params[min_index, self.damping_idx].tolist())
        if self.separate_dynamic_friction:
            print("Static Friction: ", self.sim_params[min_index, self.friction_idx].tolist())
            print("Dynamic Friction: ", self.dynamic_friction()[min_index].tolist())
        else:
            print("Static/Dynamic Friction: ", self.sim_params[min_index, self.friction_idx].tolist())
        print("Bias: ", self.sim_params[min_index, self.bias_idx].tolist())
        print("Delay: ", self.sim_params[min_index, self.delay_idx].tolist())
        print(f"Elapsed time: {(datetime.now() - self._timer_start).total_seconds():.1f} seconds")
        self._timer_start = datetime.now()
        self._log()

    def _params_to_sim_params(self, params):
        return normalized_to_physical(params, self.bounds)

    def get_best_sim_params(self):
        """Physical parameters of the current CMA-ES mean."""
        best_params = torch.tensor(self.optimizer._mean, device=self.bounds.device)
        return self._params_to_sim_params(best_params)

    def _log(self):
        min_score, min_score_index = torch.min(self.scores, dim=0)
        max_score, _ = torch.max(self.scores, dim=0)
        for i in range(len(self.joint_order)):
            self.writer.add_histogram("4_Bias/distribution_" + self.joint_order[i], self.sim_params[:, self.bias_idx][:, i], self.iteration_counter)
            self.writer.add_histogram("3_Static_Dynamic_Friction/distribution_" + self.joint_order[i], self.sim_params[:, self.friction_idx][:, i], self.iteration_counter)
            self.writer.add_histogram("2_Viscous_Friction/distribution_" + self.joint_order[i], self.sim_params[:, self.damping_idx][:, i], self.iteration_counter)
            self.writer.add_histogram("1_Armature/distribution_" + self.joint_order[i], self.sim_params[:, self.armature_idx][:, i], self.iteration_counter)

            self.writer.add_scalar("4_Bias/best_" + self.joint_order[i], self.sim_params[min_score_index, self.bias_idx][i].item(), self.iteration_counter)
            self.writer.add_scalar("3_Static_Dynamic_Friction/best_" + self.joint_order[i], self.sim_params[min_score_index, self.friction_idx][i].item(), self.iteration_counter)
            self.writer.add_scalar("2_Viscous_Friction/best_" + self.joint_order[i], self.sim_params[min_score_index, self.damping_idx][i].item(), self.iteration_counter)
            self.writer.add_scalar("1_Armature/best_" + self.joint_order[i], self.sim_params[min_score_index, self.armature_idx][i].item(), self.iteration_counter)
            if self.separate_dynamic_friction:
                self.writer.add_histogram("6_Dynamic_Friction_Ratio/distribution_" + self.joint_order[i], self.sim_params[:, self.friction_ratio_idx][:, i], self.iteration_counter)
                self.writer.add_scalar("6_Dynamic_Friction_Ratio/best_" + self.joint_order[i], self.sim_params[min_score_index, self.friction_ratio_idx][i].item(), self.iteration_counter)
                self.writer.add_scalar("6_Dynamic_Friction/best_" + self.joint_order[i], self.dynamic_friction()[min_score_index, i].item(), self.iteration_counter)
        self.writer.add_histogram("0_Delay/distribution", self.sim_params[:, self.delay_idx], self.iteration_counter)
        self.writer.add_scalar("0_Delay/best", self.sim_params[min_score_index, self.delay_idx].item(), self.iteration_counter)

        self.writer.add_scalar("0_Episode/score", min_score.item(), self.iteration_counter)
        self.writer.add_scalar("0_Episode/max_score", max_score.item(), self.iteration_counter)
        self.writer.add_scalar("0_Episode/diff_score", (max_score - min_score) / min_score, self.iteration_counter)

    def save_checkpoint(self, mean, iteration, finished=False):
        min_index = torch.argmin(self.scores_buffer[iteration, :])
        best_traj = self.sim_dof_pos_buffer[min_index].detach().clone()
        best_traj = best_traj.cpu()
        torch.save(best_traj, os.path.join(self.log_dir, "best_trajectory.pt"))
        torch.save(mean, os.path.join(self.log_dir, "mean_" + f"{iteration:03}" + ".pt"))
        if finished and self.save_optimization_process:
            torch.save({"params_buffer": self.sim_params_buffer,
                        "scores_buffer": self.scores_buffer, },
                       os.path.join(self.log_dir, "progress.pt"))

    def close(self):
        self.writer.close()
