# © 2026 Kushagra Srivastava
# Licensed under the Apache License 2.0

import math
import sys

import pytest
import torch

from pace_sim2real.core.data import TrajectoryData, linear_chirp, shape_excitation
from pace_sim2real.core.logger import NullWriter, create_writer
from pace_sim2real.core.objective import step_error, trajectory_rmse, trajectory_score

DOF_KEYS = ("time", "dof_pos", "des_dof_pos")


def _randn(*shape, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=generator, dtype=torch.float64)


def _trajectory_dict(steps=5, joints=2, **extra):
    return {
        "time": torch.arange(steps, dtype=torch.float64) * 0.01,
        "dof_pos": _randn(steps, joints, seed=10),
        "des_dof_pos": _randn(steps, joints, seed=11),
        **extra,
    }


def test_step_error_is_the_sum_of_squared_residuals_over_joints():
    sim = _randn(2, 3, seed=0)
    real = _randn(2, 3, seed=1)
    bias = _randn(2, 3, seed=2)
    expected = torch.zeros(2, dtype=torch.float64)
    for n in range(2):
        for j in range(3):
            expected[n] += (sim[n, j] - real[n, j] - bias[n, j]) ** 2
    torch.testing.assert_close(step_error(sim, real, bias), expected)
    # a single measurement vector (num_joints,) is shared by all candidates
    real_vector = real[0]
    torch.testing.assert_close(step_error(sim, real_vector, bias), step_error(sim, real_vector.expand(2, 3), bias))


def test_step_error_vanishes_when_simulation_matches_measurement_plus_bias():
    real = _randn(4, 3, seed=3)
    bias = _randn(4, 3, seed=4)
    torch.testing.assert_close(step_error(real + bias, real, bias), torch.zeros(4, dtype=torch.float64))


def test_trajectory_score_is_the_mean_of_the_per_step_errors():
    steps, candidates, joints = 5, 3, 2
    sim = _randn(steps, candidates, joints, seed=5)
    real = _randn(steps, joints, seed=6)
    bias = _randn(candidates, joints, seed=7)
    total = torch.zeros(candidates, dtype=torch.float64)
    for t in range(steps):
        total += step_error(sim[t], real[t], bias)
    torch.testing.assert_close(trajectory_score(sim, real, bias), total / steps)


def test_trajectory_rmse_has_candidate_joint_shape_and_is_zero_for_a_perfect_match():
    real = _randn(6, 2, seed=8)
    bias = _randn(3, 2, seed=9)
    perfect = real.unsqueeze(1) + bias.unsqueeze(0)  # (T, N, J), equal to real + bias for every candidate
    rmse = trajectory_rmse(perfect, real, bias)
    assert rmse.shape == (3, 2)
    torch.testing.assert_close(rmse, torch.zeros(3, 2, dtype=torch.float64), atol=1e-12, rtol=0.0)
    # a constant unit offset in the simulation gives an RMSE of exactly one
    torch.testing.assert_close(trajectory_rmse(perfect + 1.0, real, bias), torch.ones(3, 2, dtype=torch.float64))


def test_from_dict_builds_the_trajectory_and_exposes_derived_properties():
    data = _trajectory_dict(steps=5, joints=2)
    traj = TrajectoryData.from_dict(data)
    assert traj.num_steps == 5
    assert traj.num_joints == 2
    torch.testing.assert_close(traj.time, data["time"])
    torch.testing.assert_close(traj.des_dof_pos, data["des_dof_pos"])
    torch.testing.assert_close(traj.initial_dof_pos, data["dof_pos"][0])
    assert traj.joint_order is None
    assert traj.metadata == {}


def test_from_dict_keeps_extra_keys_as_metadata_and_reads_joint_order():
    traj = TrajectoryData.from_dict(_trajectory_dict(joint_order=["hip", "knee"], sample_rate=100.0, source="bench"))
    assert traj.joint_order == ["hip", "knee"]
    assert traj.metadata == {"sample_rate": 100.0, "source": "bench"}


def test_from_dict_raises_key_error_when_a_required_key_is_missing():
    data = _trajectory_dict()
    del data["des_dof_pos"]
    with pytest.raises(KeyError, match="des_dof_pos"):
        TrajectoryData.from_dict(data)


def test_inconsistent_shapes_and_joint_order_raise_value_error():
    data = _trajectory_dict(steps=5, joints=2)
    data["des_dof_pos"] = torch.zeros(5, 3, dtype=torch.float64)
    with pytest.raises(ValueError, match="same shape"):
        TrajectoryData.from_dict(data)

    data = _trajectory_dict(steps=5, joints=2)
    data["time"] = torch.zeros(4, dtype=torch.float64)
    with pytest.raises(ValueError, match="time must have shape"):
        TrajectoryData.from_dict(data)

    with pytest.raises(ValueError, match="joint_order"):
        TrajectoryData.from_dict(_trajectory_dict(steps=5, joints=2), joint_order=["hip", "knee", "ankle"])


@pytest.mark.parametrize("key", DOF_KEYS)
def test_non_finite_values_raise_value_error(key):
    data = _trajectory_dict()
    data[key][1] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        TrajectoryData.from_dict(data)


def test_save_and_load_round_trip_preserves_tensors_joint_order_and_metadata(tmp_path):
    traj = TrajectoryData.from_dict(_trajectory_dict(joint_order=["hip", "knee"], sample_rate=100.0, note="bench"))
    path = tmp_path / "nested" / "trajectory.pt"
    traj.save(path)
    loaded = TrajectoryData.load(path)
    torch.testing.assert_close(loaded.time, traj.time)
    torch.testing.assert_close(loaded.dof_pos, traj.dof_pos)
    torch.testing.assert_close(loaded.des_dof_pos, traj.des_dof_pos)
    assert loaded.joint_order == ["hip", "knee"]
    assert loaded.metadata == {"sample_rate": 100.0, "note": "bench"}


def test_as_dict_contains_the_required_keys_joint_order_and_metadata():
    traj = TrajectoryData.from_dict(_trajectory_dict(joint_order=["a", "b"], note="x"))
    data = traj.as_dict()
    assert set(DOF_KEYS) <= set(data)
    assert data["joint_order"] == ["a", "b"]
    assert data["note"] == "x"


def test_to_cpu_returns_equal_trajectory_with_the_same_metadata():
    traj = TrajectoryData.from_dict(_trajectory_dict(joint_order=["a", "b"], note="x"))
    moved = traj.to("cpu")
    torch.testing.assert_close(moved.time, traj.time)
    torch.testing.assert_close(moved.dof_pos, traj.dof_pos)
    torch.testing.assert_close(moved.des_dof_pos, traj.des_dof_pos)
    assert moved.joint_order == ["a", "b"]
    assert moved.metadata == {"note": "x"}


def test_linear_chirp_has_expected_length_endpoints_and_unit_amplitude():
    t, signal = linear_chirp(duration=2.0, sample_rate=100, min_frequency=0.5, max_frequency=2.0)
    assert t.shape == (200,)
    assert signal.shape == (200,)
    assert t[0].item() == 0.0
    assert t[-1].item() == 2.0
    assert signal[0].item() == 0.0
    assert signal.abs().max().item() <= 1.0


def test_linear_chirp_matches_the_closed_form_phase():
    t, signal = linear_chirp(duration=2.0, sample_rate=100, min_frequency=0.5, max_frequency=2.0)
    t64 = t.double()
    phase = 2 * math.pi * (0.5 * t64 + (2.0 - 0.5) / (2 * 2.0) * t64 ** 2)
    torch.testing.assert_close(signal.double(), torch.sin(phase), atol=1e-5, rtol=0.0)


def test_linear_chirp_sample_count_is_not_truncated_by_float_rounding():
    t, signal = linear_chirp(duration=0.29, sample_rate=100, min_frequency=0.5, max_frequency=2.0)
    assert t.shape == (29,)
    assert signal.shape == (29,)


def test_shape_excitation_applies_offset_direction_and_scale_per_joint():
    signal = torch.linspace(-1.0, 1.0, steps=11, dtype=torch.float64)
    offset = [0.1, -0.2, 0.0]
    direction = [1.0, -1.0, 0.5]
    scale = [2.0, 3.0, 0.5]
    targets = shape_excitation(signal, offset, direction, scale)
    assert targets.shape == (11, 3)
    expected = ((signal[:, None] + torch.tensor(offset, dtype=torch.float64))
                * torch.tensor(direction, dtype=torch.float64) * torch.tensor(scale, dtype=torch.float64))
    torch.testing.assert_close(targets, expected)


def test_create_writer_without_tensorboard_returns_a_null_writer(log_dir):
    writer = create_writer(log_dir, use_tensorboard=False)
    assert isinstance(writer, NullWriter)
    assert writer.log_dir == log_dir
    # summaries are accepted and discarded
    writer.add_scalar("loss", 1.0, 0)
    writer.add_histogram("params", torch.zeros(3), 0)
    assert writer.close() is None


def test_create_writer_warns_and_falls_back_when_tensorboard_is_missing(log_dir, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch.utils.tensorboard", None)  # makes the import raise ImportError
    with pytest.warns(UserWarning, match="TensorBoard is not installed"):
        writer = create_writer(log_dir)
    assert isinstance(writer, NullWriter)
    assert writer.log_dir == log_dir
