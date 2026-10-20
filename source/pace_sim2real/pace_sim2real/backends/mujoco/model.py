# © 2026 Kushagra Srivastava
# Licensed under the Apache License 2.0

"""MuJoCo model handling: load an MJCF robot, replicate it once per environment and build synthetic test robots.

All environments live in a single ``MjModel`` (like the environments of an Isaac Lab scene), so one ``mj_step``
advances the whole CMA-ES population. Joint ``j`` of environment ``i`` is called ``env{i}/{j}``.
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np

_OPTION_FIELDS = ("ccd_iterations", "ccd_tolerance", "cone", "density", "disableactuator", "disableflags", "enableflags",
                  "gravity", "impratio", "integrator", "iterations", "jacobian", "ls_iterations", "ls_tolerance",
                  "magnetic", "noslip_iterations", "noslip_tolerance", "o_friction", "o_margin", "o_solimp", "o_solref",
                  "sdf_initpoints", "sdf_iterations", "sleep_tolerance", "solver", "timestep", "tolerance", "viscosity",
                  "wind")


def env_prefix(env_id: int) -> str:
    return f"env{env_id}/"


def load_spec(model: str | Path | mujoco.MjSpec) -> mujoco.MjSpec:
    """Load an MJCF model from a file path or an XML string."""
    if isinstance(model, mujoco.MjSpec):
        return model.copy()
    if isinstance(model, Path) or not str(model).lstrip().startswith("<"):
        path = Path(model).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"MJCF file {path} does not exist.")
        return mujoco.MjSpec.from_file(str(path))
    return mujoco.MjSpec.from_string(str(model))


def replicate(model: str | Path | mujoco.MjSpec, num_envs: int, physics_dt: float | None = None, fix_base: bool = True,
              env_spacing: float = 2.0) -> mujoco.MjModel:
    """Compile a model with ``num_envs`` independent copies of the robot.

    Args:
        model: MJCF file path, XML string or spec of a single robot.
        num_envs: Number of copies.
        physics_dt: Simulation time step [s], defaults to the time step of the model.
        fix_base: Remove free joints, i.e. weld floating bases to the world (PACE identifies hanging robots).
        env_spacing: Distance between neighboring copies on a square grid [m].
    """
    child = load_spec(model)
    if fix_base:
        for joint in list(child.joints):
            if joint.type == mujoco.mjtJoint.mjJNT_FREE:
                child.delete(joint)
    # keyframes describe the state of a single robot
    for key in list(child.keys):
        child.delete(key)
    if physics_dt is not None:
        child.option.timestep = physics_dt

    parent = mujoco.MjSpec()
    parent.modelname = f"{child.modelname or 'robot'}_x{num_envs}"
    # the parent keeps its own physics options on attach, so take them from the robot
    for name in _OPTION_FIELDS:
        if hasattr(parent.option, name):
            setattr(parent.option, name, getattr(child.option, name))
    num_cols = math.ceil(math.sqrt(num_envs))
    for env_id in range(num_envs):
        row, col = divmod(env_id, num_cols)
        frame = parent.worldbody.add_frame(pos=[col * env_spacing, row * env_spacing, 0.0])
        parent.attach(child.copy(), frame=frame, prefix=env_prefix(env_id))
    return parent.compile()


def joint_indices(model: mujoco.MjModel, joint_names: list[str], num_envs: int) -> tuple[np.ndarray, np.ndarray]:
    """Address of every identified joint in ``qpos`` and ``qvel``, both of shape (num_envs, num_joints)."""
    qpos_ids = np.zeros((num_envs, len(joint_names)), dtype=np.int64)
    dof_ids = np.zeros_like(qpos_ids)
    for env_id in range(num_envs):
        for i, name in enumerate(joint_names):
            full_name = env_prefix(env_id) + name
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, full_name)
            if joint_id < 0:
                raise KeyError(f"Joint '{name}' not found in the MuJoCo model.")
            if int(model.jnt_type[joint_id]) not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
                raise ValueError(f"Joint '{name}' must be a hinge or slide joint to be identified.")
            qpos_ids[env_id, i] = model.jnt_qposadr[joint_id]
            dof_ids[env_id, i] = model.jnt_dofadr[joint_id]
    return qpos_ids, dof_ids


def serial_chain_xml(joint_names: list[str], link_length: float = 0.25, link_mass: float = 1.0,
                     axes: list[str] | None = None, joint_range: float | None = None, base_height: float = 1.0,
                     timestep: float = 0.0025) -> str:
    """MJCF of a fixed-base serial chain hanging down from the world, one hinge joint per link.

    Useful as a synthetic robot with known structure: one joint is a pendulum, three joints with axes
    ``["x", "y", "y"]`` resemble a leg (HAA, HFE, KFE).

    Args:
        joint_names: Joint names from the base to the tip.
        link_length: Length of every link [m], the links point downwards.
        link_mass: Mass of every link [kg].
        axes: Rotation axis ("x", "y" or "z") per joint, defaults to "y" for all joints.
        joint_range: Symmetric joint limit [rad] (the model uses radians), ``None`` for unlimited joints.
        base_height: Height of the first joint above the ground [m].
        timestep: Simulation time step [s].
    """
    axes = axes or ["y"] * len(joint_names)
    if len(axes) != len(joint_names):
        raise ValueError("axes must have one entry per joint.")
    axis_vectors = {"x": "1 0 0", "y": "0 1 0", "z": "0 0 1"}
    limit = f'limited="true" range="{-joint_range} {joint_range}"' if joint_range is not None else 'limited="false"'
    bodies = ""
    closing = ""
    for i, (name, axis) in enumerate(zip(joint_names, axes)):
        pos = f"0 0 {base_height}" if i == 0 else f"0 0 {-link_length}"
        bodies += (f'<body name="link_{name}" pos="{pos}">'
                   f'<joint name="{name}" type="hinge" axis="{axis_vectors[axis]}" {limit}/>'
                   f'<geom type="capsule" fromto="0 0 0 0.02 0 {-link_length}" size="0.02" mass="{link_mass}"/>')
        closing += "</body>"
    return (f'<mujoco model="serial_chain_{len(joint_names)}">'
            '<compiler angle="radian"/>'
            f'<option timestep="{timestep}" integrator="implicitfast"/>'
            f"<worldbody>{bodies}{closing}</worldbody></mujoco>")
