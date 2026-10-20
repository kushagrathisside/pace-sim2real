# © 2026 Kushagra Srivastava
# Licensed under the Apache License 2.0

"""PACE task configurations for the MuJoCo backend.

A configuration bundles the robot model, the fixed actuator properties, the search bounds, an excitation and, for
synthetic benchmarks, the ground-truth parameters. ``ROBOTS`` contains built-in synthetic robots that need no assets.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from pace_sim2real.backends.actuator import PaceDCMotorParams
from pace_sim2real.core.data import linear_chirp, shape_excitation
from pace_sim2real.core.parameters import ParameterLayout, extend_bounds_with_friction_ratio, sim_params_from_values

from .adapter import MujocoBackend
from .model import serial_chain_xml


@dataclass
class ExcitationCfg:
    """Chirp excitation ``target = (chirp + offset) * direction * scale`` per joint, as in the data collection."""

    duration: float = 20.0
    min_frequency: float = 0.1
    max_frequency: float = 10.0
    offset: list[float] | float = 0.0
    direction: list[float] | float = 1.0
    scale: list[float] | float = 0.5


@dataclass
class CMAESCfg:
    max_iteration: int = 200
    epsilon: float | None = 1e-2
    sigma: float = 0.5
    save_interval: int = 10
    save_optimization_process: bool = False
    population_size: int = 32


@dataclass
class MujocoPaceCfg:
    """PACE configuration of a robot simulated with MuJoCo."""

    robot_name: str
    model: str
    """MJCF file path or XML string of a single robot."""
    joint_order: list[str]
    bounds_params: torch.Tensor
    """Search bounds (4 * n + 1, 2) or (5 * n + 1, 2), see :mod:`pace_sim2real.core.parameters`."""
    actuator: PaceDCMotorParams = field(default_factory=PaceDCMotorParams)
    physics_dt: float = 0.0025
    separate_dynamic_friction: bool = False
    friction_ratio_bounds: tuple[float, float] = (0.0, 1.0)
    static_friction_velocity: float = 1e-2
    excitation: ExcitationCfg = field(default_factory=ExcitationCfg)
    cmaes: CMAESCfg = field(default_factory=CMAESCfg)
    true_sim_params: torch.Tensor | None = None
    """Ground-truth parameters used to generate synthetic data, if known."""

    def __post_init__(self):
        if self.separate_dynamic_friction:
            self.bounds_params = extend_bounds_with_friction_ratio(self.bounds_params, len(self.joint_order), self.friction_ratio_bounds)
        self.layout.validate_bounds(self.bounds_params)

    @property
    def layout(self) -> ParameterLayout:
        return ParameterLayout(len(self.joint_order), self.separate_dynamic_friction)

    def make_backend(self, num_envs: int) -> MujocoBackend:
        return MujocoBackend(self.model, self.joint_order, num_envs, actuator=self.actuator, physics_dt=self.physics_dt,
                             static_friction_velocity=self.static_friction_velocity)

    def excitation_targets(self, sample_rate: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Time stamps (num_steps,) and joint position targets (num_steps, num_joints) of the chirp excitation."""
        ex = self.excitation
        num_joints = len(self.joint_order)
        t, signal = linear_chirp(ex.duration, sample_rate or 1.0 / self.physics_dt, ex.min_frequency, ex.max_frequency)
        expand = lambda value: torch.as_tensor(value, dtype=torch.float32).expand(num_joints)  # noqa: E731
        return t, shape_excitation(signal, expand(ex.offset), expand(ex.direction), expand(ex.scale))


def pendulum_cfg() -> MujocoPaceCfg:
    """Single hinge joint, a 1 kg link of 25 cm, Coulomb friction (static = dynamic)."""
    joints = ["joint"]
    layout = ParameterLayout(1)
    return MujocoPaceCfg(
        robot_name="mujoco_pendulum",
        model=serial_chain_xml(joints),
        joint_order=joints,
        bounds_params=torch.tensor([[1e-5, 0.1], [0.0, 1.0], [0.0, 0.5], [-0.1, 0.1], [0.0, 8.0]]),
        actuator=PaceDCMotorParams(stiffness=20.0, damping=0.5, effort_limit=20.0, saturation_effort=30.0,
                                   velocity_limit=20.0, max_delay=10),
        excitation=ExcitationCfg(duration=5.0, min_frequency=0.1, max_frequency=5.0, scale=0.6),
        cmaes=CMAESCfg(max_iteration=60, population_size=16, epsilon=None),
        true_sim_params=sim_params_from_values(layout, armature=0.02, damping=0.3, friction=0.1, bias=0.03, delay=3),
    )


def leg_cfg() -> MujocoPaceCfg:
    """Three-joint leg (HAA about x, HFE and KFE about y) with separately identified static and dynamic friction."""
    joints = ["HAA", "HFE", "KFE"]
    layout = ParameterLayout(3, separate_dynamic_friction=True)
    n = len(joints)
    bounds = torch.zeros((4 * n + 1, 2))
    bounds[0:n] = torch.tensor([1e-5, 0.1])  # armature [kg m^2]
    bounds[n:2 * n] = torch.tensor([0.0, 1.0])  # viscous friction [Nm s/rad]
    bounds[2 * n:3 * n] = torch.tensor([0.0, 0.5])  # static friction [Nm]
    bounds[3 * n:4 * n] = torch.tensor([-0.1, 0.1])  # encoder bias [rad]
    bounds[4 * n] = torch.tensor([0.0, 8.0])  # delay [sim steps]
    return MujocoPaceCfg(
        robot_name="mujoco_leg",
        model=serial_chain_xml(joints, axes=["x", "y", "y"], link_length=0.25, link_mass=1.0, joint_range=2.5),
        joint_order=joints,
        bounds_params=bounds,
        separate_dynamic_friction=True,
        actuator=PaceDCMotorParams(stiffness=[40.0, 40.0, 30.0], damping=[1.0, 1.0, 0.8], effort_limit=40.0,
                                   saturation_effort=60.0, velocity_limit=20.0, max_delay=10),
        # frequencies up to 10 Hz are needed to tell the delay apart from friction
        excitation=ExcitationCfg(duration=10.0, min_frequency=0.1, max_frequency=10.0, offset=[0.0, 0.4, 0.8],
                                 direction=[1.0, 1.0, -1.0], scale=[0.25, 0.5, 0.6]),
        cmaes=CMAESCfg(max_iteration=150, population_size=32, epsilon=None),
        true_sim_params=sim_params_from_values(layout, armature=[0.01, 0.02, 0.015], damping=[0.2, 0.4, 0.3],
                                               friction=[0.08, 0.12, 0.1], bias=[0.02, -0.03, 0.05], delay=4,
                                               friction_ratio=[0.5, 0.7, 0.6]),
    )


ROBOTS: dict[str, Callable[[], MujocoPaceCfg]] = {
    "pendulum": pendulum_cfg,
    "leg": leg_cfg,
}
