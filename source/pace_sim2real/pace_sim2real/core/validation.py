# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

"""Synthetic validation: generate data with known parameters and measure how well PACE recovers them."""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import torch

from .data import TrajectoryData
from .fitting import fit
from .optimizer import CMAESOptimizer
from .parameters import JointParameters, ParameterLayout

if TYPE_CHECKING:
    from pace_sim2real.backends.base import SimulatorBackend


def generate_synthetic_data(backend: SimulatorBackend, true_sim_params: torch.Tensor, layout: ParameterLayout,
                            joint_pos_targets: torch.Tensor, time: torch.Tensor | None = None,
                            initial_joint_pos: torch.Tensor | None = None, noise_std: float = 0.0,
                            seed: int = 0) -> TrajectoryData:
    """Record a trajectory of the simulated robot with known ("ground-truth") parameters.

    Every environment of ``backend`` simulates the same parameters, the first one is recorded. The measurements are
    stored in the encoder frame (simulated position minus the true bias), like real encoder readings.

    Args:
        backend: Simulator to generate the data with.
        true_sim_params: Physical parameter vector (num_params,).
        layout: Layout of ``true_sim_params``.
        joint_pos_targets: Encoder-frame position targets (num_steps, num_joints).
        time: Time stamps (num_steps,), defaults to multiples of the backend time step.
        initial_joint_pos: Encoder-frame initial positions (num_joints,), defaults to the first target.
        noise_std: Standard deviation of white Gaussian noise added to the measured positions [rad].
        seed: Seed of the measurement noise.
    """
    num_steps = joint_pos_targets.shape[0]
    params = JointParameters.from_sim_params(true_sim_params.to(backend.device).unsqueeze(0).repeat(backend.num_envs, 1), layout)
    if initial_joint_pos is None:
        initial_joint_pos = joint_pos_targets[0]
    sim_dof_pos = backend.rollout(params, initial_joint_pos, joint_pos_targets)[:, 0, :].cpu()
    dof_pos = sim_dof_pos - params.bias[0].cpu()
    if noise_std > 0.0:
        generator = torch.Generator().manual_seed(seed)
        dof_pos = dof_pos + noise_std * torch.randn(dof_pos.shape, generator=generator, dtype=dof_pos.dtype)
    if time is None:
        time = torch.arange(num_steps, dtype=dof_pos.dtype) * backend.physics_dt
    return TrajectoryData(time=time.cpu(), dof_pos=dof_pos, des_dof_pos=joint_pos_targets.cpu().to(dof_pos.dtype),
                          joint_order=list(backend.joint_names),
                          metadata={"true_sim_params": true_sim_params.cpu(), "noise_std": noise_std})


@dataclass
class ParameterError:
    """Recovery error of one identified parameter."""

    quantity: str
    joint: str
    true: float
    estimated: float
    lower: float
    upper: float

    @property
    def abs_error(self) -> float:
        return abs(self.estimated - self.true)

    @property
    def normalized_error(self) -> float:
        """Absolute error relative to the width of the search interval."""
        width = self.upper - self.lower
        return self.abs_error / width if width > 0 else 0.0


def _named_slices(layout: ParameterLayout) -> list[tuple[str, slice | int]]:
    named = [("armature", layout.armature), ("damping", layout.damping), ("friction", layout.friction),
             ("bias", layout.bias), ("delay", layout.delay)]
    if layout.separate_dynamic_friction:
        named.append(("friction_ratio", layout.friction_ratio))
    return named


def recovery_report(true_sim_params: torch.Tensor, estimated_sim_params: torch.Tensor, bounds: torch.Tensor,
                    layout: ParameterLayout, joint_order: list[str]) -> list[ParameterError]:
    """Compare estimated to true physical parameters, one entry per scalar parameter.

    The delay is compared in whole simulation steps, as the simulators truncate it (see :class:`JointParameters`).
    """
    true_sim_params = true_sim_params.detach().cpu().double().clone()
    estimated_sim_params = estimated_sim_params.detach().cpu().double().clone()
    bounds = bounds.detach().cpu().double()
    true_sim_params[layout.delay] = torch.trunc(true_sim_params[layout.delay])
    estimated_sim_params[layout.delay] = torch.trunc(estimated_sim_params[layout.delay])
    report = []
    for quantity, index in _named_slices(layout):
        indices = [index] if isinstance(index, int) else list(range(index.start, index.stop))
        joints = ["all"] if isinstance(index, int) else joint_order
        for joint, i in zip(joints, indices):
            report.append(ParameterError(quantity, joint, true_sim_params[i].item(), estimated_sim_params[i].item(),
                                         bounds[i, 0].item(), bounds[i, 1].item()))
    if layout.separate_dynamic_friction:
        # dynamic friction = static friction * ratio, usually better observable than its factors
        friction = range(layout.friction.start, layout.friction.stop)
        ratio = range(layout.friction_ratio.start, layout.friction_ratio.stop)
        for joint, i, j in zip(joint_order, friction, ratio):
            report.append(ParameterError("dynamic_friction", joint,
                                         (true_sim_params[i] * true_sim_params[j]).item(),
                                         (estimated_sim_params[i] * estimated_sim_params[j]).item(),
                                         (bounds[i, 0] * bounds[j, 0]).item(), (bounds[i, 1] * bounds[j, 1]).item()))
    return report


def max_normalized_error(report: list[ParameterError], quantities: tuple[str, ...] | None = None) -> float:
    """Largest normalized error of the selected quantities (all if ``None``)."""
    errors = [entry.normalized_error for entry in report if quantities is None or entry.quantity in quantities]
    return max(errors) if errors else 0.0


def within_bounds(sim_params: torch.Tensor, bounds: torch.Tensor, tolerance: float = 1e-6) -> bool:
    """Whether all physical parameters lie inside ``bounds`` (up to ``tolerance``)."""
    return bool(torch.all(sim_params >= bounds[:, 0] - tolerance) and torch.all(sim_params <= bounds[:, 1] + tolerance))


@dataclass
class RecoveryResult:
    """Outcome of :func:`synthetic_recovery`."""

    true_sim_params: torch.Tensor
    estimated_sim_params: torch.Tensor
    report: list[ParameterError]
    data: TrajectoryData
    optimizer: CMAESOptimizer
    best_scores: torch.Tensor
    """Lowest score of every iteration (num_iterations,)."""


def synthetic_recovery(make_backend: Callable[[int], SimulatorBackend], true_sim_params: torch.Tensor,
                       bounds: torch.Tensor, layout: ParameterLayout, joint_pos_targets: torch.Tensor, log_dir: str,
                       population_size: int = 32, max_iteration: int = 50, sigma: float = 0.5,
                       epsilon: float | None = None, noise_std: float = 0.0, seed: int = 0,
                       time: torch.Tensor | None = None) -> RecoveryResult:
    """Generate synthetic data with ``true_sim_params`` and identify them again with CMA-ES.

    Args:
        make_backend: Creates a backend with the given number of environments.
        true_sim_params: Ground-truth physical parameter vector (num_params,).
        bounds: Search bounds (num_params, 2).
        layout: Parameter layout.
        joint_pos_targets: Encoder-frame excitation (num_steps, num_joints).
        log_dir: Directory of the optimizer logs and checkpoints.
        population_size: CMA-ES population size (= number of simulated environments).
        max_iteration: Number of CMA-ES iterations.
        sigma: Initial CMA-ES step size in the normalized space.
        epsilon: Optional early-stopping threshold on the relative score spread.
        noise_std: Measurement noise [rad].
        seed: Seed of the measurement noise and of CMA-ES.
        time: Optional time stamps of the excitation.
    """
    generator = make_backend(1)
    data = generate_synthetic_data(generator, true_sim_params, layout, joint_pos_targets, time=time,
                                   noise_std=noise_std, seed=seed)
    generator.close()
    backend = make_backend(population_size)
    optimizer = CMAESOptimizer(bounds, population_size, log_dir, list(backend.joint_names), max_iteration, data,
                               backend.device, epsilon=epsilon, sigma=sigma, save_interval=0,
                               separate_dynamic_friction=layout.separate_dynamic_friction, seed=seed,
                               use_tensorboard=False)
    with contextlib.redirect_stdout(io.StringIO()):
        estimated = fit(backend, optimizer, data, log_interval=0)
    optimizer.close()
    backend.close()
    report = recovery_report(true_sim_params, estimated, bounds, layout, list(backend.joint_names))
    best_scores = optimizer.scores_buffer[:optimizer.iteration_counter].min(dim=1).values.cpu()
    return RecoveryResult(true_sim_params, estimated.cpu(), report, data, optimizer, best_scores)


def format_report(report: list[ParameterError]) -> str:
    """Human readable table of a :func:`recovery_report`."""
    lines = [f"{'quantity':<18}{'joint':<12}{'true':>12}{'estimated':>12}{'abs err':>12}{'norm err':>10}"]
    for entry in report:
        lines.append(f"{entry.quantity:<18}{entry.joint:<12}{entry.true:>12.5f}{entry.estimated:>12.5f}"
                     f"{entry.abs_error:>12.5f}{entry.normalized_error:>10.3%}")
    return "\n".join(lines)
