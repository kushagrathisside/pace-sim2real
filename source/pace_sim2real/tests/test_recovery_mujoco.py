# © 2026 Kushagra Srivastava
# Licensed under the Apache License 2.0

"""Synthetic parameter recovery with the MuJoCo backend: PACE must find the parameters that generated the data."""

import pytest

pytest.importorskip("mujoco")

from pace_sim2real.backends.mujoco.robots import leg_cfg, pendulum_cfg  # noqa: E402
from pace_sim2real.core import format_report, max_normalized_error, synthetic_recovery, within_bounds  # noqa: E402

CONTINUOUS = ("armature", "damping", "friction", "bias")


def recover(cfg, tmp_path, population_size, max_iteration, noise_std=0.0, seed=0, bounds=None):
    t, targets = cfg.excitation_targets()
    bounds = cfg.bounds_params if bounds is None else bounds
    result = synthetic_recovery(cfg.make_backend, cfg.true_sim_params, bounds, cfg.layout, targets, str(tmp_path),
                                population_size=population_size, max_iteration=max_iteration, noise_std=noise_std,
                                seed=seed, time=t)
    print(format_report(result.report))
    assert within_bounds(result.estimated_sim_params, bounds)
    # the best score of single generations may fluctuate, but the search must improve overall
    assert result.best_scores.min() < 1e-2 * result.best_scores[0]
    return result


def test_pendulum_recovery(tmp_path):
    result = recover(pendulum_cfg(), tmp_path, population_size=16, max_iteration=60)
    assert result.best_scores[-1] < 1e-4 * result.best_scores[0]
    assert max_normalized_error(result.report, CONTINUOUS) < 0.05
    assert max_normalized_error(result.report, ("delay",)) == 0.0


def test_pendulum_recovery_with_noise(tmp_path):
    noise_std = 1e-3
    result = recover(pendulum_cfg(), tmp_path, population_size=16, max_iteration=60, noise_std=noise_std)
    # the score cannot drop below the noise floor (one joint: noise variance)
    assert result.best_scores[-1] == pytest.approx(noise_std ** 2, rel=0.5)
    assert max_normalized_error(result.report, CONTINUOUS) < 0.10
    assert max_normalized_error(result.report, ("delay",)) == 0.0


@pytest.mark.slow
def test_leg_recovery(tmp_path):
    """Three coupled joints with separately identified static and dynamic friction, excited up to 10 Hz."""
    result = recover(leg_cfg(), tmp_path, population_size=32, max_iteration=150)
    assert max_normalized_error(result.report, ("armature", "damping", "dynamic_friction", "bias")) < 0.05
    assert max_normalized_error(result.report, ("delay",)) == 0.0
