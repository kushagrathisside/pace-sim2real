# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

"""Simulator-independent PACE parameter model.

A candidate is a flat parameter vector. For ``n`` identified joints the layout is::

    [0,      n)       armature                       [kg m^2]
    [n,     2n)       viscous friction ("damping")   [Nm s/rad]
    [2n,    3n)       static (Coulomb) friction      [Nm]
    [3n,    4n)       encoder bias                   [rad]
    4n                actuator delay                 [simulation steps]
    [4n+1, 5n+1)      dynamic / static friction ratio (only with ``separate_dynamic_friction``)

The optimizer samples in a normalized space ``[-1, 1]`` per parameter, which is mapped linearly onto the bounds.
Backends never see the flat vector, they receive :class:`JointParameters` with one tensor per physical quantity.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ParameterLayout:
    """Index layout of the flat PACE parameter vector."""

    num_joints: int
    separate_dynamic_friction: bool = False

    @property
    def armature(self) -> slice:
        return slice(0, self.num_joints)

    @property
    def damping(self) -> slice:
        return slice(self.num_joints, 2 * self.num_joints)

    @property
    def friction(self) -> slice:
        return slice(2 * self.num_joints, 3 * self.num_joints)

    @property
    def bias(self) -> slice:
        return slice(3 * self.num_joints, 4 * self.num_joints)

    @property
    def delay(self) -> int:
        return 4 * self.num_joints

    @property
    def friction_ratio(self) -> slice:
        # dynamic to static friction ratios come after all other parameters
        return slice(4 * self.num_joints + 1, 5 * self.num_joints + 1)

    @property
    def num_params(self) -> int:
        return 4 * self.num_joints + 1 + (self.num_joints if self.separate_dynamic_friction else 0)

    def validate_bounds(self, bounds: torch.Tensor) -> None:
        """Raise a ``ValueError`` if ``bounds`` (num_params x 2) does not fit this layout."""
        if bounds.ndim != 2 or bounds.shape[1] != 2:
            raise ValueError(f"Parameter bounds must have shape (num_params, 2), got {tuple(bounds.shape)}.")
        if bounds.shape[0] != self.num_params:
            raise ValueError(f"Expected {self.num_params} parameter bounds (separate_dynamic_friction="
                             f"{self.separate_dynamic_friction}), got {bounds.shape[0]}.")
        if torch.any(bounds[:, 0] > bounds[:, 1]):
            raise ValueError("Every parameter lower bound must not exceed its upper bound.")
        # dynamic friction = ratio * static friction must stay within [0, static friction]
        if self.separate_dynamic_friction:
            ratio_bounds = bounds[self.friction_ratio]
            if torch.any(ratio_bounds[:, 0] < 0.0) or torch.any(ratio_bounds[:, 1] > 1.0) or torch.any(ratio_bounds[:, 0] > ratio_bounds[:, 1]):
                raise ValueError(f"Dynamic to static friction ratio bounds must satisfy 0 <= lower <= upper <= 1, got {ratio_bounds.tolist()}.")


def extend_bounds_with_friction_ratio(bounds: torch.Tensor, num_joints: int,
                                      friction_ratio_bounds: tuple[float, float] = (0.0, 1.0)) -> torch.Tensor:
    """Append the dynamic to static friction ratio bounds (one per joint) to ``bounds``.

    Bounds that already contain the ratios are returned unchanged.
    """
    num_params = bounds.shape[0]
    num_before = 4 * num_joints + 1
    if num_params == num_before:
        ratio_bounds = torch.tensor(friction_ratio_bounds, dtype=bounds.dtype, device=bounds.device).repeat(num_joints, 1)
        return torch.cat([bounds, ratio_bounds], dim=0)
    if num_params != num_before + num_joints:
        raise ValueError(f"Expected {num_before} or {num_before + num_joints} bounds with separate dynamic friction, got {num_params}.")
    return bounds


def normalized_to_physical(params: torch.Tensor, bounds: torch.Tensor) -> torch.Tensor:
    """Map normalized parameters in ``[-1, 1]`` linearly onto ``[lower, upper]`` of ``bounds``."""
    sim_params = (params + 1.0) / 2.0  # change range from 0 to 1
    return bounds[:, 0] + sim_params * (bounds[:, 1] - bounds[:, 0])  # range from lower to upper bound


def physical_to_normalized(sim_params: torch.Tensor, bounds: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`normalized_to_physical`. Parameters with zero-width bounds map to 0."""
    width = bounds[:, 1] - bounds[:, 0]
    safe_width = torch.where(width > 0, width, torch.ones_like(width))
    return torch.where(width > 0, 2.0 * (sim_params - bounds[:, 0]) / safe_width - 1.0, torch.zeros_like(sim_params))


@dataclass
class JointParameters:
    """Physical joint and actuator parameters of a population, one row per environment / candidate.

    All joint tensors have shape (num_envs, num_joints) and follow the identified ``joint_order``.
    """

    armature: torch.Tensor
    damping: torch.Tensor
    """Viscous friction coefficient [Nm s/rad]."""
    static_friction: torch.Tensor
    dynamic_friction: torch.Tensor
    """Equal to :attr:`static_friction` unless static and dynamic friction are identified separately."""
    bias: torch.Tensor
    """Encoder bias [rad]: the controller and the measurements see ``q_true - bias``."""
    delay: torch.Tensor
    """Actuator delay per environment in simulation steps, shape (num_envs,), dtype int."""

    @classmethod
    def from_sim_params(cls, sim_params: torch.Tensor, layout: ParameterLayout) -> JointParameters:
        """Split physical parameter vectors (num_envs x num_params, or a single vector) into named quantities."""
        if sim_params.ndim == 1:
            sim_params = sim_params.unsqueeze(0)
        friction = sim_params[:, layout.friction]
        if layout.separate_dynamic_friction:
            dynamic_friction = friction * sim_params[:, layout.friction_ratio]
        else:
            dynamic_friction = friction
        return cls(
            armature=sim_params[:, layout.armature],
            damping=sim_params[:, layout.damping],
            static_friction=friction,
            dynamic_friction=dynamic_friction,
            bias=sim_params[:, layout.bias],
            delay=sim_params[:, layout.delay].to(torch.int),
        )

    @property
    def num_envs(self) -> int:
        return self.armature.shape[0]

    @property
    def num_joints(self) -> int:
        return self.armature.shape[1]

    def select(self, env_ids) -> JointParameters:
        """Return the parameters of a subset of environments."""
        return JointParameters(*(getattr(self, name)[env_ids] for name in self._fields()))

    def to(self, device) -> JointParameters:
        return JointParameters(*(getattr(self, name).to(device) for name in self._fields()))

    @staticmethod
    def _fields() -> tuple[str, ...]:
        return ("armature", "damping", "static_friction", "dynamic_friction", "bias", "delay")


def sim_params_from_values(layout: ParameterLayout, armature, damping, friction, bias, delay, friction_ratio=None,
                           dtype=torch.float32, device="cpu") -> torch.Tensor:
    """Assemble a single physical parameter vector from per-joint values (scalars or sequences).

    Convenient for defining ground-truth parameters in synthetic benchmarks.
    """
    n = layout.num_joints
    sim_params = torch.zeros(layout.num_params, dtype=dtype, device=device)
    sim_params[layout.armature] = torch.as_tensor(armature, dtype=dtype, device=device).expand(n)
    sim_params[layout.damping] = torch.as_tensor(damping, dtype=dtype, device=device).expand(n)
    sim_params[layout.friction] = torch.as_tensor(friction, dtype=dtype, device=device).expand(n)
    sim_params[layout.bias] = torch.as_tensor(bias, dtype=dtype, device=device).expand(n)
    sim_params[layout.delay] = float(delay)
    if layout.separate_dynamic_friction:
        if friction_ratio is None:
            raise ValueError("friction_ratio is required with separate dynamic friction.")
        sim_params[layout.friction_ratio] = torch.as_tensor(friction_ratio, dtype=dtype, device=device).expand(n)
    elif friction_ratio is not None:
        raise ValueError("friction_ratio given, but the layout does not identify dynamic friction separately.")
    return sim_params
