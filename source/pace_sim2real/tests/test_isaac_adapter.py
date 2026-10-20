# © 2026 Kushagra Srivastava
# Licensed under the Apache License 2.0

"""The Isaac Lab backend must drive Isaac Lab exactly like PACE v0.2.0 did.

Isaac Sim is not needed: a mock environment records every call into the articulation, the actuators and the
environment. The reference implementations below are the write sequence of ``CMAESOptimizer.update_simulator`` and
the loop of ``scripts/pace/fit.py`` from PACE v0.2.0.
"""

import contextlib
import io
import os
import types

import pytest
import torch

from pace_sim2real import CMAESOptimizer
from pace_sim2real.backends.isaac import IsaacLabBackend, apply_pace_parameters
from pace_sim2real.core import JointParameters, ParameterLayout, fit
from pace_sim2real.core.optimizer import CMAESOptimizer as CoreCMAESOptimizer

JOINT_ORDER = ["a", "b", "c"]


class CallLog(list):
    def record(self, name, *args, **kwargs):
        def copy(value):
            return value.detach().clone() if isinstance(value, torch.Tensor) else value
        self.append((name, tuple(copy(a) for a in args), {k: copy(v) for k, v in sorted(kwargs.items())}))


class MockActuator:
    def __init__(self, log, name, joint_indices):
        self.log, self.name, self.joint_indices = log, name, joint_indices

    def update_encoder_bias(self, bias):
        self.log.record(f"{self.name}.update_encoder_bias", bias)

    def update_time_lags(self, delay):
        self.log.record(f"{self.name}.update_time_lags", delay)

    def reset(self, env_ids):
        self.log.record(f"{self.name}.reset", env_ids)


class MockArticulation:
    # simulator joint order differs from the identified joint order on purpose
    joint_names = ["b", "x", "a", "c"]

    def __init__(self, log, num_envs):
        self.log = log
        self.data = types.SimpleNamespace(**{name: torch.zeros(num_envs, 4) for name in (
            "default_joint_armature", "default_joint_viscous_friction_coeff", "default_joint_friction_coeff",
            "default_joint_dynamic_friction_coeff", "joint_pos", "joint_vel")})
        # one actuator group with index tensors, one with a slice
        self.actuators = {"legs": MockActuator(log, "legs", torch.tensor([0, 2, 3])),
                          "arm": MockActuator(log, "arm", slice(1, 2))}
        self.armature = torch.zeros(num_envs, 4)
        self.damping = torch.zeros(num_envs, 4)

    def write_joint_armature_to_sim(self, value, joint_ids, env_ids):
        self.log.record("armature", value, joint_ids=joint_ids, env_ids=env_ids)
        self.armature[:, joint_ids] = value

    def write_joint_viscous_friction_coefficient_to_sim(self, value, joint_ids, env_ids):
        self.log.record("viscous_friction", value, joint_ids=joint_ids, env_ids=env_ids)
        self.damping[:, joint_ids] = value

    def write_joint_dynamic_friction_coefficient_to_sim(self, value, joint_ids, env_ids):
        self.log.record("dynamic_friction", value, joint_ids=joint_ids, env_ids=env_ids)

    def write_joint_friction_coefficient_to_sim(self, value, joint_ids, env_ids):
        self.log.record("friction", value, joint_ids=joint_ids, env_ids=env_ids)

    def write_joint_position_to_sim(self, value, joint_ids):
        self.log.record("joint_pos", value, joint_ids=joint_ids)
        self.data.joint_pos[:, joint_ids] = value

    def write_joint_velocity_to_sim(self, value, joint_ids):
        self.log.record("joint_vel", value, joint_ids=joint_ids)
        self.data.joint_vel[:, joint_ids] = value


class MockEnv:
    """Isaac Lab environment stand-in with deterministic first-order dynamics that depend on the parameters."""

    def __init__(self, num_envs, log=None):
        self.log = CallLog() if log is None else log
        self.num_envs = num_envs
        self.device = "cpu"
        self.articulation = MockArticulation(self.log, num_envs)
        self.scene = {"robot": self.articulation}
        self.sim = types.SimpleNamespace(cfg=types.SimpleNamespace(dt=0.0025))
        self.action_space = types.SimpleNamespace(shape=(num_envs, 4))

    @property
    def unwrapped(self):
        return self

    def reset(self):
        self.log.record("env.reset")

    def step(self, actions):
        self.log.record("env.step", actions)
        a = self.articulation
        gain = 0.3 / (1.0 + 5.0 * a.armature + a.damping)
        a.data.joint_pos += gain * (actions - a.data.joint_pos) + 0.01 * torch.sin(a.data.joint_pos)

    def close(self):
        self.log.record("env.close")


def reference_update_simulator(opt, articulation, joint_ids, initial_position):
    """``CMAESOptimizer.update_simulator`` of PACE v0.2.0."""
    env_ids = torch.arange(len(opt.sim_params[:, opt.armature_idx]))
    articulation.write_joint_armature_to_sim(opt.sim_params[:, opt.armature_idx], joint_ids=joint_ids, env_ids=env_ids)
    articulation.data.default_joint_armature[:, joint_ids] = opt.sim_params[:, opt.armature_idx]
    articulation.write_joint_viscous_friction_coefficient_to_sim(opt.sim_params[:, opt.damping_idx], joint_ids=joint_ids, env_ids=env_ids)
    articulation.data.default_joint_viscous_friction_coeff[:, joint_ids] = opt.sim_params[:, opt.damping_idx]
    articulation.write_joint_dynamic_friction_coefficient_to_sim(0.0, joint_ids=joint_ids, env_ids=env_ids)
    articulation.write_joint_friction_coefficient_to_sim(opt.sim_params[:, opt.friction_idx], joint_ids=joint_ids, env_ids=env_ids)
    articulation.data.default_joint_friction_coeff[:, joint_ids] = opt.sim_params[:, opt.friction_idx]
    articulation.write_joint_dynamic_friction_coefficient_to_sim(opt.dynamic_friction(), joint_ids=joint_ids, env_ids=env_ids)
    articulation.data.default_joint_dynamic_friction_coeff[:, joint_ids] = opt.dynamic_friction()
    articulation.write_joint_position_to_sim(initial_position + opt.sim_params[:, opt.bias_idx], joint_ids=joint_ids)
    articulation.write_joint_velocity_to_sim(torch.zeros_like(initial_position), joint_ids=joint_ids)
    for drive_type in articulation.actuators.keys():
        drive_indices = articulation.actuators[drive_type].joint_indices
        if isinstance(drive_indices, slice):
            all_idx = torch.arange(joint_ids.shape[0], device=joint_ids.device)
            drive_indices = all_idx[drive_indices]
        comparison_matrix = (joint_ids.unsqueeze(1) == drive_indices.unsqueeze(0))
        drive_joint_idx = torch.argmax(comparison_matrix.int(), dim=0)
        articulation.actuators[drive_type].update_encoder_bias(opt.sim_params[:, opt.bias_idx][:, drive_joint_idx])
        articulation.actuators[drive_type].update_time_lags(opt.sim_params[:, opt.delay_idx].to(torch.int))
        articulation.actuators[drive_type].reset(env_ids)


def reference_fit(env, opt, data, joint_order):
    """Main loop of ``scripts/pace/fit.py`` of PACE v0.2.0 (with ``simulation_app.is_running()`` always true)."""
    articulation = env.unwrapped.scene["robot"]
    sim_joint_ids = torch.tensor([articulation.joint_names.index(name) for name in joint_order], device=env.unwrapped.device)
    time_data = data["time"].to(env.unwrapped.device)
    target_dof_pos = data["des_dof_pos"].to(env.unwrapped.device)
    measured_dof_pos = data["dof_pos"].to(env.unwrapped.device)
    initial_dof_pos = measured_dof_pos[0, :].unsqueeze(0).repeat(env.unwrapped.num_envs, 1)
    time_steps = time_data.shape[0]
    sim_dt = env.unwrapped.sim.cfg.dt
    env.reset()
    reference_update_simulator(opt, articulation, sim_joint_ids, initial_dof_pos)
    counter = 0
    while True:
        with torch.inference_mode():
            opt.tell(env.unwrapped.scene["robot"].data.joint_pos[:, sim_joint_ids], measured_dof_pos[counter, :].unsqueeze(0).repeat(env.unwrapped.num_envs, 1))
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            actions[:, sim_joint_ids] = target_dof_pos[counter, :].unsqueeze(0).repeat(env.unwrapped.num_envs, 1)
            env.step(actions)
            counter += 1
            if counter % 400 == 0:
                print(f"[INFO]: Step {counter * sim_dt:.1f} / {time_data[-1]:.1f} seconds ({counter / time_steps * 100:.1f} %)")
            if counter >= time_steps:
                print("[INFO]: Reached the end of the trajectory, exiting.")
                counter = 0
                opt.evolve()
                if opt.finished():
                    break
                env.reset()
                reference_update_simulator(opt, env.unwrapped.scene["robot"], sim_joint_ids, initial_dof_pos)


def assert_same_calls(expected, actual):
    assert [call[0] for call in actual] == [call[0] for call in expected]
    for (name, args_e, kwargs_e), (_, args_a, kwargs_a) in zip(expected, actual):
        assert len(args_e) == len(args_a) and kwargs_e.keys() == kwargs_a.keys(), name
        for value_e, value_a in list(zip(args_e, args_a)) + [(kwargs_e[k], kwargs_a[k]) for k in kwargs_e]:
            if isinstance(value_e, torch.Tensor):
                assert isinstance(value_a, torch.Tensor) and value_a.dtype == value_e.dtype, name
                assert torch.equal(value_a, value_e), name
            else:
                assert value_a == value_e, name


@pytest.fixture
def problem():
    torch.manual_seed(0)
    num_steps, n = 25, len(JOINT_ORDER)
    data = {"time": torch.arange(num_steps) * 0.0025, "des_dof_pos": 0.3 * torch.randn(num_steps, n),
            "dof_pos": 0.3 * torch.randn(num_steps, n)}
    bounds = torch.tensor([[1e-5, 1.0]] * n + [[0.0, 7.0]] * n + [[0.0, 0.5]] * n + [[-0.1, 0.1]] * n + [[0.0, 10.0]]
                          + [[0.0, 1.0]] * n)
    return data, bounds


def make_optimizer(problem, log_dir, cls=CMAESOptimizer, max_iteration=4, population_size=6):
    data, bounds = problem
    with contextlib.redirect_stdout(io.StringIO()):
        return cls(bounds=bounds, population_size=population_size, log_dir=log_dir, joint_order=JOINT_ORDER,
                   max_iteration=max_iteration, data=data, device="cpu", epsilon=None, sigma=0.5, save_interval=2,
                   save_optimization_process=True, separate_dynamic_friction=True, use_tensorboard=False)


def test_legacy_optimizer_keeps_update_simulator(problem, log_dir):
    assert issubclass(CMAESOptimizer, CoreCMAESOptimizer)
    assert not hasattr(CoreCMAESOptimizer, "update_simulator")
    opt = make_optimizer(problem, log_dir)
    joint_ids = torch.tensor([2, 0, 3])
    initial = torch.randn(1, 3).repeat(6, 1)

    expected_env, actual_env = MockEnv(6), MockEnv(6)
    reference_update_simulator(opt, expected_env.articulation, joint_ids, initial)
    opt.update_simulator(actual_env.articulation, joint_ids, initial)
    assert_same_calls(expected_env.log, actual_env.log)
    for name in vars(expected_env.articulation.data):
        assert torch.equal(getattr(actual_env.articulation.data, name), getattr(expected_env.articulation.data, name))


def test_backend_start_rollout_matches_reference(problem, log_dir):
    opt = make_optimizer(problem, log_dir)
    initial = torch.randn(3)

    expected_env = MockEnv(6)
    expected_env.reset()
    reference_update_simulator(opt, expected_env.articulation, torch.tensor([2, 0, 3]), initial.unsqueeze(0).repeat(6, 1))

    actual_env = MockEnv(6)
    IsaacLabBackend(actual_env, JOINT_ORDER).start_rollout(opt.joint_parameters(), initial)
    assert_same_calls(expected_env.log, actual_env.log)


def test_backend_maps_joints_and_actions():
    env = MockEnv(2)
    backend = IsaacLabBackend(env, JOINT_ORDER)
    assert backend.num_envs == 2 and backend.device == "cpu" and backend.physics_dt == 0.0025
    assert backend.joint_ids.tolist() == [2, 0, 3]
    env.articulation.data.joint_pos[:] = torch.tensor([[10.0, 11.0, 12.0, 13.0]])
    assert backend.get_joint_positions().tolist() == [[12.0, 10.0, 13.0]] * 2

    backend.step(torch.tensor([[1.0, 2.0, 3.0]]).repeat(2, 1))
    name, (actions,), _ = env.log[-1]
    assert name == "env.step"
    assert actions.tolist() == [[2.0, 0.0, 1.0, 3.0]] * 2  # untouched joint "x" gets a zero target

    backend.close()
    assert env.log[-1][0] == "env.close"


def test_set_joint_parameters_and_state():
    env = MockEnv(2)
    backend = IsaacLabBackend(env, JOINT_ORDER)
    sim_params = torch.arange(16, dtype=torch.float32).repeat(2, 1) / 100.0
    params = JointParameters.from_sim_params(sim_params, ParameterLayout(3, separate_dynamic_friction=True))
    backend.set_joint_parameters(params)
    assert torch.equal(env.articulation.armature[:, [2, 0, 3]], params.armature)
    assert torch.equal(env.articulation.data.default_joint_dynamic_friction_coeff[:, [2, 0, 3]], params.dynamic_friction)
    assert [call[0] for call in env.log].count("legs.reset") == 1

    backend.set_joint_state(torch.ones(2, 3), 2 * torch.ones(2, 3))
    assert env.articulation.data.joint_pos[:, [2, 0, 3]].eq(1.0).all()
    assert env.articulation.data.joint_vel[:, [2, 0, 3]].eq(2.0).all()


def test_apply_pace_parameters_bias_offsets_initial_position():
    env = MockEnv(1)
    sim_params = torch.zeros(1, 13)
    sim_params[0, 9:12] = torch.tensor([0.01, -0.02, 0.03])  # encoder bias
    params = JointParameters.from_sim_params(sim_params, ParameterLayout(3))
    apply_pace_parameters(env.articulation, torch.tensor([2, 0, 3]), params, torch.tensor([[0.5, 0.5, 0.5]]))
    torch.testing.assert_close(env.articulation.data.joint_pos[0, [2, 0, 3]], torch.tensor([0.51, 0.48, 0.53]))


def test_fit_matches_reference_loop(problem, tmp_path):
    data, _ = problem
    expected_env = MockEnv(6)
    expected_opt = make_optimizer(problem, str(tmp_path / "expected"))
    with contextlib.redirect_stdout(io.StringIO()) as expected_out:
        reference_fit(expected_env, expected_opt, data, JOINT_ORDER)

    actual_env = MockEnv(6)
    actual_opt = make_optimizer(problem, str(tmp_path / "actual"))
    with contextlib.redirect_stdout(io.StringIO()) as actual_out:
        best = fit(IsaacLabBackend(actual_env, JOINT_ORDER), actual_opt, data)

    assert_same_calls(expected_env.log, actual_env.log)
    assert actual_out.getvalue() == expected_out.getvalue()
    assert torch.equal(actual_opt.scores_buffer, expected_opt.scores_buffer)
    assert torch.equal(actual_opt.sim_params_buffer, expected_opt.sim_params_buffer)
    for name in sorted(os.listdir(expected_opt.log_dir)):
        expected = torch.load(os.path.join(expected_opt.log_dir, name))
        actual = torch.load(os.path.join(actual_opt.log_dir, name))
        if isinstance(expected, dict):
            assert expected.keys() == actual.keys()
            for key in expected:
                assert torch.equal(torch.as_tensor(actual[key]), torch.as_tensor(expected[key])) if isinstance(expected[key], torch.Tensor) else actual[key] == expected[key]
        else:
            assert torch.equal(actual, expected), name
    torch.testing.assert_close(best, torch.load(os.path.join(actual_opt.log_dir, "mean_003.pt")))
