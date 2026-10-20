# © 2026 Kushagra Srivastava
# Licensed under the Apache License 2.0

"""MuJoCo backend: model replication, parameter mapping and actuator semantics."""

import numpy as np
import pytest
import torch

mujoco = pytest.importorskip("mujoco")

from pace_sim2real.backends.mujoco import MujocoBackend, PaceDCMotorParams, replicate, serial_chain_xml  # noqa: E402
from pace_sim2real.backends.mujoco.robots import ROBOTS  # noqa: E402
from pace_sim2real.core import (  # noqa: E402
    JointParameters,
    ParameterLayout,
    generate_synthetic_data,
    linear_chirp,
    shape_excitation,
    sim_params_from_values,
    trajectory_score,
)

JOINTS = ["hip", "knee"]
LAYOUT = ParameterLayout(2, separate_dynamic_friction=True)
ACTUATOR = PaceDCMotorParams(stiffness=30.0, damping=0.8, max_delay=6)

FLOATING_XML = """
<mujoco model="floating">
  <compiler angle="radian"/>
  <worldbody>
    <body name="base" pos="0 0 1">
      <freejoint name="root"/>
      <geom type="box" size="0.1 0.1 0.1" mass="2"/>
      <body name="thigh">
        <joint name="hip" axis="0 1 0" range="-1 1"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.3" size="0.02" mass="1"/>
        <body name="shank" pos="0 0 -0.3">
          <joint name="knee" axis="0 1 0"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.3" size="0.02" mass="0.5"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator><position joint="hip" kp="100"/></actuator>
  <keyframe><key name="home" qpos="0 0 1 1 0 0 0 0.1 0.2"/></keyframe>
</mujoco>
"""


def params(num_envs, **values):
    defaults = dict(armature=[0.01, 0.02], damping=[0.1, 0.2], friction=[0.05, 0.08], bias=[0.02, -0.03], delay=2,
                    friction_ratio=[0.5, 0.8])
    defaults.update(values)
    sim_params = sim_params_from_values(LAYOUT, **defaults)
    return JointParameters.from_sim_params(sim_params.unsqueeze(0).repeat(num_envs, 1), LAYOUT)


def targets(num_steps=400):
    t, signal = linear_chirp(num_steps * 0.0025, 400.0, 0.5, 6.0)
    return shape_excitation(signal, [0.0, 0.3], [1.0, -1.0], [0.4, 0.5])


@pytest.fixture
def backend():
    return MujocoBackend(serial_chain_xml(JOINTS), JOINTS, num_envs=3, actuator=ACTUATOR)


def test_replicate_fixes_base_and_prefixes_joints():
    model = replicate(FLOATING_XML, num_envs=4, physics_dt=0.002)
    names = [model.joint(i).name for i in range(model.njnt)]
    assert names == [f"env{i}/{joint}" for i in range(4) for joint in ("hip", "knee")]
    assert model.nq == model.nv == 8  # the free joints are gone
    assert model.nkey == 0
    assert model.opt.timestep == pytest.approx(0.002)
    np.testing.assert_allclose(model.jnt_range[0], [-1.0, 1.0])


def test_replicate_keeps_floating_base_on_request():
    model = replicate(FLOATING_XML, num_envs=2, fix_base=False)
    assert model.nq == 2 * (7 + 2)


def test_backend_from_file_with_floating_base(tmp_path):
    path = tmp_path / "robot.xml"
    path.write_text(FLOATING_XML)
    backend = MujocoBackend(str(path), ["knee", "hip"], num_envs=2, actuator=ACTUATOR)
    assert backend.model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_ACTUATION
    assert backend.model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_CONTACT
    # joint order of the backend, not of the MJCF
    backend.reset()
    backend.set_joint_state(torch.tensor([[0.1, 0.2], [0.3, 0.4]]), torch.zeros(2, 2))
    assert backend.data.qpos.tolist() == pytest.approx([0.2, 0.1, 0.4, 0.3])
    torch.testing.assert_close(backend.get_joint_positions(), torch.tensor([[0.1, 0.2], [0.3, 0.4]]))


def test_unknown_joint_and_missing_file(tmp_path):
    with pytest.raises(KeyError, match="ankle"):
        MujocoBackend(serial_chain_xml(JOINTS), ["ankle"], num_envs=1)
    with pytest.raises(FileNotFoundError):
        MujocoBackend(str(tmp_path / "missing.xml"), JOINTS, num_envs=1)


def test_parameters_are_written_per_environment(backend):
    p = params(3)
    p.armature[1] = torch.tensor([0.05, 0.06])
    backend.set_joint_parameters(p)
    dofs = backend.dof_ids
    np.testing.assert_allclose(backend.model.dof_armature[dofs], p.armature.numpy(), rtol=1e-6)
    np.testing.assert_allclose(backend.model.dof_damping[dofs], p.damping.numpy(), rtol=1e-6)
    np.testing.assert_allclose(backend.model.dof_frictionloss[dofs], p.static_friction.numpy(), rtol=1e-6)
    np.testing.assert_allclose(backend.actuator.encoder_bias, p.bias.numpy(), rtol=1e-6)


def test_wrong_parameter_shape_and_delay(backend):
    with pytest.raises(ValueError, match="3 environments"):
        backend.set_joint_parameters(params(2))
    with pytest.raises(ValueError, match="Delays"):
        backend.set_joint_parameters(params(3, delay=7))


def test_static_and_dynamic_friction_switch(backend):
    backend.reset()
    backend.set_joint_parameters(params(3))
    p = params(3)
    backend.set_joint_state(torch.zeros(3, 2), torch.tensor([[0.0, 0.0], [5.0, 0.0], [0.0, -5.0]]))
    backend.step(torch.zeros(3, 2))
    friction = backend.model.dof_frictionloss[backend.dof_ids]
    expected = p.static_friction.numpy().copy()
    expected[1, 0] = p.dynamic_friction[1, 0]
    expected[2, 1] = p.dynamic_friction[2, 1]
    np.testing.assert_allclose(friction, expected, rtol=1e-6)


def test_environments_are_independent(backend):
    p = params(3)
    p.damping[2] = torch.tensor([0.9, 0.9])
    sim = backend.rollout(p, torch.zeros(2), targets())
    torch.testing.assert_close(sim[:, 0], sim[:, 1])
    assert (sim[:, 0] - sim[:, 2]).abs().max() > 1e-3


def test_rollouts_are_reproducible(backend):
    first = backend.rollout(params(3), torch.zeros(2), targets())
    second = backend.rollout(params(3), torch.zeros(2), targets())
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_command_delay_equals_shifted_targets(backend):
    """With delay_mode "command", a delay of d steps is the same as replaying target[max(k - d, 0)]."""
    delay = 3
    commands = targets()
    shifted = torch.cat([commands[:1].repeat(delay, 1), commands[:-delay]])
    delayed = backend.rollout(params(3, delay=delay), commands[0], commands)
    undelayed = backend.rollout(params(3, delay=0), commands[0], shifted)
    torch.testing.assert_close(delayed, undelayed)


def test_effort_delay_differs_from_command_delay():
    commands = targets()
    effort = MujocoBackend(serial_chain_xml(JOINTS), JOINTS, num_envs=1,
                           actuator=PaceDCMotorParams(stiffness=30.0, damping=0.8, max_delay=6, delay_mode="effort"))
    command = MujocoBackend(serial_chain_xml(JOINTS), JOINTS, num_envs=1, actuator=ACTUATOR)
    a = effort.rollout(params(1, delay=3), commands[0], commands)
    b = command.rollout(params(1, delay=3), commands[0], commands)
    assert (a - b).abs().max() > 1e-4


def test_encoder_bias_equals_shifted_targets(backend):
    """A bias b makes the controller track target + b in the simulator frame, starting at initial + b."""
    commands = targets()
    bias = torch.tensor([0.04, -0.05])
    biased = backend.rollout(params(3, bias=bias.tolist(), delay=0), commands[0], commands)
    unbiased = backend.rollout(params(3, bias=[0.0, 0.0], delay=0), commands[0] + bias, commands + bias)
    torch.testing.assert_close(biased, unbiased)


def test_synthetic_data_has_zero_score_for_true_parameters(backend):
    true = sim_params_from_values(LAYOUT, armature=[0.01, 0.02], damping=[0.1, 0.2], friction=[0.05, 0.08],
                                  bias=[0.02, -0.03], delay=2, friction_ratio=[0.5, 0.8])
    data = generate_synthetic_data(MujocoBackend(serial_chain_xml(JOINTS), JOINTS, 1, actuator=ACTUATOR), true, LAYOUT,
                                   targets())
    p = params(3)
    sim = backend.rollout(p, data.dof_pos[0], data.des_dof_pos)
    assert trajectory_score(sim, data.dof_pos, p.bias).max() < 1e-12
    # a wrong armature is visible in the score
    p.armature[1] *= 3.0
    sim = backend.rollout(p, data.dof_pos[0], data.des_dof_pos)
    assert trajectory_score(sim, data.dof_pos, p.bias)[1] > 1e-6


def test_dc_motor_limits_torque():
    backend = MujocoBackend(serial_chain_xml(["joint"]), ["joint"], num_envs=1,
                            actuator=PaceDCMotorParams(stiffness=1000.0, damping=0.0, effort_limit=2.0,
                                                       saturation_effort=3.0, velocity_limit=10.0, max_delay=0))
    backend.reset()
    single = ParameterLayout(1)
    backend.set_joint_parameters(JointParameters.from_sim_params(torch.zeros(1, single.num_params), single))
    backend.set_joint_state(torch.zeros(1, 1), torch.zeros(1, 1))
    backend.step(torch.tensor([[1.0]]))
    assert backend.data.qfrc_applied[backend.dof_ids[0, 0]] == pytest.approx(2.0)


@pytest.mark.parametrize("name", sorted(ROBOTS))
def test_builtin_robot_configs(name):
    cfg = ROBOTS[name]()
    assert cfg.bounds_params.shape == (cfg.layout.num_params, 2)
    assert cfg.true_sim_params.shape == (cfg.layout.num_params,)
    assert torch.all(cfg.true_sim_params >= cfg.bounds_params[:, 0]) and torch.all(cfg.true_sim_params <= cfg.bounds_params[:, 1])
    t, commands = cfg.excitation_targets()
    assert commands.shape == (t.shape[0], len(cfg.joint_order))
    backend = cfg.make_backend(2)
    assert backend.num_envs == 2 and backend.physics_dt == pytest.approx(cfg.physics_dt)
    # the joints are not stuck at their limits (MJCF angles are in radians)
    data = generate_synthetic_data(cfg.make_backend(1), cfg.true_sim_params, cfg.layout, commands[:800])
    assert (data.dof_pos.max(0).values - data.dof_pos.min(0).values).min() > 0.1


def test_other_environments_do_not_affect_a_rollout():
    """Replicas share one MjModel; a candidate must simulate the same no matter what the other candidates are."""
    commands = targets()
    alone = MujocoBackend(serial_chain_xml(JOINTS), JOINTS, num_envs=1, actuator=ACTUATOR)
    expected = alone.rollout(params(1), commands[0], commands)[:, 0]
    crowd = MujocoBackend(serial_chain_xml(JOINTS), JOINTS, num_envs=8, actuator=ACTUATOR)
    p = params(8)
    generator = torch.Generator().manual_seed(0)
    for name in ("armature", "damping", "static_friction", "bias"):
        getattr(p, name)[1:] = 0.1 * torch.rand(7, 2, generator=generator)
    p.dynamic_friction[1:] = 0.5 * p.static_friction[1:]
    p.delay[1:] = torch.randint(0, 6, (7,), generator=generator, dtype=torch.int32)
    torch.testing.assert_close(crowd.rollout(p, commands[0], commands)[:, 0], expected, rtol=0, atol=0)
