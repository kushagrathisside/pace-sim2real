# © 2026 Kushagra Srivastava
# Licensed under the Apache License 2.0

"""CMA-ES optimizer and fitting loop on the simulator-free toy backend."""

import contextlib
import io
import os

import pytest
import torch
from toy_backend import ToyPendulumBackend

from pace_sim2real.core import (
    CMAESOptimizer,
    ParameterLayout,
    TrajectoryData,
    fit,
    generate_synthetic_data,
    linear_chirp,
    max_normalized_error,
    recovery_report,
    shape_excitation,
    sim_params_from_values,
    synthetic_recovery,
    within_bounds,
)

JOINTS = ["j0", "j1"]
LAYOUT = ParameterLayout(len(JOINTS))
TRUE_PARAMS = sim_params_from_values(LAYOUT, armature=[0.01, 0.03], damping=[0.2, 0.1], friction=[0.05, 0.1],
                                     bias=[0.02, -0.01], delay=2)
BOUNDS = torch.tensor([[1e-5, 0.05]] * 2 + [[0.0, 0.5]] * 2 + [[0.0, 0.2]] * 2 + [[-0.05, 0.05]] * 2 + [[0.0, 5.0]])


def make_backend(num_envs):
    return ToyPendulumBackend(JOINTS, num_envs)


def excitation(duration=2.0):
    t, signal = linear_chirp(duration, 400.0, 0.2, 4.0)
    return t, shape_excitation(signal, [0.0, 0.2], [1.0, -1.0], [0.5, 0.4])


@pytest.fixture(scope="module")
def data():
    t, targets = excitation()
    return generate_synthetic_data(make_backend(1), TRUE_PARAMS, LAYOUT, targets, time=t)


def make_optimizer(data, log_dir, population_size=8, max_iteration=5, **kwargs):
    kwargs.setdefault("use_tensorboard", False)
    with contextlib.redirect_stdout(io.StringIO()):
        return CMAESOptimizer(BOUNDS, population_size, log_dir, JOINTS, max_iteration, data, "cpu", **kwargs)


def run_fit(backend, opt, data, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fit(backend, opt, data, **kwargs)


def test_synthetic_data_is_in_encoder_frame(data):
    assert data.joint_order == JOINTS
    assert data.num_steps == 800
    t, targets = excitation()
    torch.testing.assert_close(data.des_dof_pos, targets)
    # the robot starts at the first target (encoder frame), i.e. at target + bias in the simulator frame
    torch.testing.assert_close(data.dof_pos[0], targets[0])
    torch.testing.assert_close(data.metadata["true_sim_params"], TRUE_PARAMS)


def test_synthetic_data_noise_is_seeded():
    t, targets = excitation(0.5)
    clean = generate_synthetic_data(make_backend(1), TRUE_PARAMS, LAYOUT, targets, time=t)
    noisy_a = generate_synthetic_data(make_backend(1), TRUE_PARAMS, LAYOUT, targets, time=t, noise_std=1e-3, seed=3)
    noisy_b = generate_synthetic_data(make_backend(1), TRUE_PARAMS, LAYOUT, targets, time=t, noise_std=1e-3, seed=3)
    torch.testing.assert_close(noisy_a.dof_pos, noisy_b.dof_pos)
    assert 5e-4 < (noisy_a.dof_pos - clean.dof_pos).std() < 2e-3


def test_fit_writes_population_into_backend(data, log_dir):
    backend = make_backend(8)
    opt = make_optimizer(data, log_dir, max_iteration=2, save_optimization_process=True)
    run_fit(backend, opt, data)
    assert opt.iteration_counter == 2
    assert backend.num_resets == 2  # one rollout per iteration
    # the backend holds the population of the last iteration
    evaluated = opt.sim_params_buffer[1]
    torch.testing.assert_close(backend.params.armature, evaluated[:, LAYOUT.armature])
    torch.testing.assert_close(backend.params.bias, evaluated[:, LAYOUT.bias])
    assert torch.equal(backend.params.delay, evaluated[:, LAYOUT.delay].int())


def test_fit_matches_manual_rollout_scores(data, log_dir):
    """The score of every candidate equals the time-averaged tracking error of its rollout."""
    backend = make_backend(8)
    opt = make_optimizer(data, log_dir, max_iteration=1, save_optimization_process=True)
    params = opt.joint_parameters()
    run_fit(backend, opt, data)
    sim = make_backend(8).rollout(params, data.dof_pos[0], data.des_dof_pos)
    expected = ((sim - data.dof_pos.unsqueeze(1) - params.bias.unsqueeze(0)) ** 2).sum(-1).mean(0)
    torch.testing.assert_close(opt.scores_buffer[0], expected)


def test_synthetic_recovery_toy(tmp_path):
    t, targets = excitation()
    result = synthetic_recovery(make_backend, TRUE_PARAMS, BOUNDS, LAYOUT, targets, str(tmp_path), population_size=16,
                                max_iteration=60, time=t)
    assert result.best_scores[-1] < 1e-3 * result.best_scores[0]
    assert within_bounds(result.estimated_sim_params, BOUNDS)
    # the bias is only observable through the weak gravity torque of the toy model, see the MuJoCo benchmarks
    assert max_normalized_error(result.report, ("armature", "damping")) < 0.05
    assert max_normalized_error(result.report, ("delay",)) == 0.0


def test_fit_is_deterministic(data, tmp_path):
    def run(seed, name):
        opt = make_optimizer(data, str(tmp_path / name), max_iteration=3, seed=seed)
        best = run_fit(make_backend(8), opt, data)
        return opt.scores_buffer, best

    scores_a, best_a = run(0, "a")
    scores_b, best_b = run(0, "b")
    scores_c, _ = run(1, "c")
    assert torch.equal(scores_a, scores_b) and torch.equal(best_a, best_b)
    assert not torch.equal(scores_a, scores_c)


def test_checkpoints(data, log_dir):
    opt = make_optimizer(data, log_dir, max_iteration=4, save_interval=2, save_optimization_process=True)
    best = run_fit(make_backend(8), opt, data)
    files = sorted(os.listdir(opt.log_dir))
    assert files == ["best_trajectory.pt", "config.pt", "mean_000.pt", "mean_002.pt", "mean_003.pt", "progress.pt"]
    config = torch.load(os.path.join(opt.log_dir, "config.pt"))
    assert config["joint_order"] == JOINTS and config["separate_dynamic_friction"] is False
    torch.testing.assert_close(config["bounds"], BOUNDS)
    torch.testing.assert_close(config["dof_pos"], data.dof_pos)
    assert torch.load(os.path.join(opt.log_dir, "best_trajectory.pt")).shape == (data.num_steps, 2)
    torch.testing.assert_close(torch.load(os.path.join(opt.log_dir, "mean_003.pt")), best)
    progress = torch.load(os.path.join(opt.log_dir, "progress.pt"))
    assert progress["params_buffer"].shape == (4, 8, LAYOUT.num_params)
    assert within_bounds(progress["params_buffer"].reshape(-1, LAYOUT.num_params), BOUNDS)


def test_tensorboard_summaries(data, log_dir):
    pytest.importorskip("tensorboard")
    opt = make_optimizer(data, log_dir, max_iteration=1, use_tensorboard=True)
    run_fit(make_backend(8), opt, data)
    opt.close()
    assert any(name.startswith("events.out.tfevents") for name in os.listdir(opt.log_dir))


def test_early_stopping_with_epsilon(data, log_dir):
    opt = make_optimizer(data, log_dir, max_iteration=50, epsilon=1e6)
    run_fit(make_backend(8), opt, data)
    assert opt.iteration_counter == 1


def test_should_continue_aborts(data, log_dir):
    opt = make_optimizer(data, log_dir)
    calls = []

    def should_continue():
        calls.append(None)
        return len(calls) <= 10

    assert run_fit(make_backend(8), opt, data, should_continue=should_continue) is None
    assert opt.iteration_counter == 0 and opt.scores_counter == 10


def test_fit_validates_shapes(data, log_dir):
    opt = make_optimizer(data, log_dir)
    with pytest.raises(ValueError, match="population size"):
        fit(make_backend(4), opt, data)
    with pytest.raises(ValueError, match="joints"):
        fit(ToyPendulumBackend(["only"], 8), opt, data)


def test_fit_accepts_dict_data(data, log_dir):
    opt = make_optimizer(data.as_dict(), log_dir, max_iteration=1)
    assert run_fit(make_backend(8), opt, data.as_dict()) is not None


def test_optimizer_rejects_bad_bounds(data, log_dir):
    with pytest.raises(ValueError, match="Expected 11 parameter bounds"):
        make_optimizer(data, log_dir, separate_dynamic_friction=True)
    assert not os.path.exists(log_dir)  # nothing is written for an invalid configuration


def test_best_sim_params_follow_bounds_device(data, log_dir):
    opt = make_optimizer(data, log_dir)
    best = opt.get_best_sim_params()
    assert best.device == BOUNDS.device and best.shape == (LAYOUT.num_params,)
    # the initial CMA-ES mean is the center of the bounds
    torch.testing.assert_close(best.float(), BOUNDS.mean(dim=1))


def test_recovery_report_truncates_delay():
    estimated = TRUE_PARAMS.clone()
    estimated[LAYOUT.delay] = 2.9
    report = recovery_report(TRUE_PARAMS, estimated, BOUNDS, LAYOUT, JOINTS)
    assert [entry.quantity for entry in report].count("armature") == 2
    assert max_normalized_error(report) == 0.0


def test_trajectory_data_roundtrip_through_optimizer_config(data, log_dir):
    opt = make_optimizer(data, log_dir, max_iteration=1)
    config = torch.load(os.path.join(opt.log_dir, "config.pt"))
    restored = TrajectoryData.from_dict(config, joint_order=config["joint_order"])
    torch.testing.assert_close(restored.dof_pos, data.dof_pos)
