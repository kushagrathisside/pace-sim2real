# © 2026 Kushagra Srivastava
# Licensed under the Apache License 2.0

import numpy as np
import pytest
import torch

from pace_sim2real.core.parameters import (
    JointParameters,
    ParameterLayout,
    extend_bounds_with_friction_ratio,
    normalized_to_physical,
    physical_to_normalized,
    sim_params_from_values,
)

JOINT_FIELDS = ("armature", "damping", "static_friction", "dynamic_friction", "bias", "delay")
NORMALIZATION_BOUNDS = torch.tensor([[-1.0, 1.0], [0.0, 4.0], [2.0, 6.0], [-0.5, 0.0]], dtype=torch.float64)


def _generator(seed=0):
    return torch.Generator().manual_seed(seed)


def _bounds(num_rows, lower=-1.0, upper=1.0, dtype=torch.float64):
    return torch.tensor([[lower, upper]] * num_rows, dtype=dtype)


def _sim_params(layout, num_envs, seed=0):
    # uniform in [0.1, 1.1): every physical quantity is positive and the delays are not integers
    return torch.rand(num_envs, layout.num_params, generator=_generator(seed), dtype=torch.float64) + 0.1


def test_layout_slices_for_three_joints():
    layout = ParameterLayout(num_joints=3)
    assert layout.armature == slice(0, 3)
    assert layout.damping == slice(3, 6)
    assert layout.friction == slice(6, 9)
    assert layout.bias == slice(9, 12)
    assert layout.delay == 12
    assert layout.friction_ratio == slice(13, 16)
    assert layout.num_params == 13
    separate = ParameterLayout(num_joints=3, separate_dynamic_friction=True)
    assert separate.num_params == 16
    assert separate.friction_ratio == slice(13, 16)


def test_layout_entries_cover_the_flat_vector_exactly_once():
    for separate in (False, True):
        layout = ParameterLayout(num_joints=3, separate_dynamic_friction=separate)
        flat = np.arange(layout.num_params)
        pieces = [flat[layout.armature], flat[layout.damping], flat[layout.friction], flat[layout.bias], [layout.delay]]
        if separate:
            pieces.append(flat[layout.friction_ratio])
        np.testing.assert_array_equal(np.sort(np.concatenate(pieces)), flat)


def test_validate_bounds_accepts_matching_bounds():
    ParameterLayout(num_joints=3).validate_bounds(_bounds(13))
    ParameterLayout(num_joints=3, separate_dynamic_friction=True).validate_bounds(_bounds(16, 0.0, 1.0))


def test_validate_bounds_rejects_wrong_shape():
    layout = ParameterLayout(num_joints=3)
    with pytest.raises(ValueError, match="Expected 13 parameter bounds"):
        layout.validate_bounds(_bounds(12))
    with pytest.raises(ValueError, match="must have shape"):
        layout.validate_bounds(torch.zeros(13, 3))


def test_validate_bounds_rejects_lower_above_upper():
    bounds = _bounds(13)
    bounds[4] = torch.tensor([0.5, -0.5], dtype=torch.float64)
    with pytest.raises(ValueError):
        ParameterLayout(num_joints=3).validate_bounds(bounds)


def test_validate_bounds_rejects_friction_ratio_outside_unit_interval():
    layout = ParameterLayout(num_joints=3, separate_dynamic_friction=True)
    bounds = _bounds(16, 0.0, 1.0)
    bounds[13] = torch.tensor([-0.1, 1.5], dtype=torch.float64)
    with pytest.raises(ValueError, match="ratio"):
        layout.validate_bounds(bounds)


def test_validate_bounds_rejects_friction_ratio_lower_above_upper():
    layout = ParameterLayout(num_joints=3, separate_dynamic_friction=True)
    bounds = _bounds(16, 0.0, 1.0)
    bounds[15] = torch.tensor([0.8, 0.5], dtype=torch.float64)
    # the generic lower <= upper check rejects this row before the ratio-specific check runs
    with pytest.raises(ValueError):
        layout.validate_bounds(bounds)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_extend_appends_one_ratio_row_per_joint_and_keeps_dtype(dtype):
    bounds = _bounds(13, -2.0, 2.0, dtype=dtype)
    extended = extend_bounds_with_friction_ratio(bounds, 3, (0.2, 0.9))
    assert extended.shape == (16, 2)
    assert extended.dtype == dtype
    torch.testing.assert_close(extended[:13], bounds)
    torch.testing.assert_close(extended[13:], torch.tensor([[0.2, 0.9]] * 3, dtype=dtype))


def test_extend_returns_already_extended_bounds_unchanged():
    bounds = _bounds(16, 0.0, 1.0)
    torch.testing.assert_close(extend_bounds_with_friction_ratio(bounds, 3, (0.2, 0.9)), bounds)


def test_extend_rejects_any_other_row_count():
    for num_rows in (12, 14, 17):
        with pytest.raises(ValueError):
            extend_bounds_with_friction_ratio(_bounds(num_rows), 3, (0.2, 0.9))


def test_normalized_to_physical_maps_endpoints_and_midpoint():
    ones = torch.ones(4, dtype=torch.float64)
    torch.testing.assert_close(normalized_to_physical(-ones, NORMALIZATION_BOUNDS), NORMALIZATION_BOUNDS[:, 0])
    torch.testing.assert_close(normalized_to_physical(ones, NORMALIZATION_BOUNDS), NORMALIZATION_BOUNDS[:, 1])
    midpoint = normalized_to_physical(torch.zeros(4, dtype=torch.float64), NORMALIZATION_BOUNDS)
    torch.testing.assert_close(midpoint, NORMALIZATION_BOUNDS.mean(dim=1))


def test_normalized_to_physical_is_batched_over_a_population():
    params = torch.rand(5, 4, generator=_generator(), dtype=torch.float64) * 2.0 - 1.0
    physical = normalized_to_physical(params, NORMALIZATION_BOUNDS)
    assert physical.shape == (5, 4)
    for candidate in range(5):
        torch.testing.assert_close(physical[candidate], normalized_to_physical(params[candidate], NORMALIZATION_BOUNDS))
    expected = NORMALIZATION_BOUNDS[:, 0] + (params + 1.0) / 2.0 * (NORMALIZATION_BOUNDS[:, 1] - NORMALIZATION_BOUNDS[:, 0])
    torch.testing.assert_close(physical, expected)


def test_physical_to_normalized_inverts_normalized_to_physical():
    params = torch.rand(7, 4, generator=_generator(1), dtype=torch.float64) * 2.0 - 1.0
    recovered = physical_to_normalized(normalized_to_physical(params, NORMALIZATION_BOUNDS), NORMALIZATION_BOUNDS)
    torch.testing.assert_close(recovered, params, atol=1e-6, rtol=0.0)


def test_physical_to_normalized_maps_zero_width_bounds_to_zero_without_nan():
    bounds = torch.tensor([[-1.0, 1.0], [0.3, 0.3]], dtype=torch.float64)
    normalized = physical_to_normalized(torch.tensor([0.5, 0.3], dtype=torch.float64), bounds)
    assert torch.isfinite(normalized).all()
    torch.testing.assert_close(normalized, torch.tensor([0.5, 0.0], dtype=torch.float64))


def test_from_sim_params_splits_quantities_into_per_joint_tensors():
    layout = ParameterLayout(num_joints=3)
    sim = _sim_params(layout, 4)
    joints = JointParameters.from_sim_params(sim, layout)
    torch.testing.assert_close(joints.armature, sim[:, layout.armature])
    torch.testing.assert_close(joints.damping, sim[:, layout.damping])
    torch.testing.assert_close(joints.static_friction, sim[:, layout.friction])
    torch.testing.assert_close(joints.bias, sim[:, layout.bias])
    for name in ("armature", "damping", "static_friction", "dynamic_friction", "bias"):
        assert getattr(joints, name).shape == (4, 3), name
    assert joints.delay.shape == (4,)
    assert joints.delay.dtype == torch.int32
    assert (joints.num_envs, joints.num_joints) == (4, 3)


def test_delay_is_truncated_to_whole_simulation_steps():
    layout = ParameterLayout(num_joints=2)
    sim = torch.zeros(3, layout.num_params, dtype=torch.float64)
    sim[:, layout.delay] = torch.tensor([3.9, 0.2, 7.0], dtype=torch.float64)
    assert JointParameters.from_sim_params(sim, layout).delay.tolist() == [3, 0, 7]


def test_dynamic_friction_equals_static_friction_unless_identified_separately():
    plain = ParameterLayout(num_joints=3)
    joints = JointParameters.from_sim_params(_sim_params(plain, 2), plain)
    torch.testing.assert_close(joints.dynamic_friction, joints.static_friction)

    separate = ParameterLayout(num_joints=3, separate_dynamic_friction=True)
    sim = _sim_params(separate, 2)
    joints = JointParameters.from_sim_params(sim, separate)
    expected = sim[:, separate.friction] * sim[:, separate.friction_ratio]
    torch.testing.assert_close(joints.dynamic_friction, expected)


def test_single_parameter_vector_is_treated_as_one_environment():
    layout = ParameterLayout(num_joints=3)
    joints = JointParameters.from_sim_params(_sim_params(layout, 1)[0], layout)
    assert joints.num_envs == 1
    assert joints.armature.shape == (1, 3)
    assert joints.delay.shape == (1,)


def test_select_and_to_keep_the_values_of_the_selected_environments():
    layout = ParameterLayout(num_joints=3)
    joints = JointParameters.from_sim_params(_sim_params(layout, 4), layout)
    subset = joints.select([0, 2])
    assert subset.num_envs == 2
    for name in JOINT_FIELDS:
        assert torch.equal(getattr(subset, name), getattr(joints, name)[[0, 2]]), name
    moved = joints.to("cpu")
    for name in JOINT_FIELDS:
        assert torch.equal(getattr(moved, name), getattr(joints, name)), name


def test_sim_params_from_values_broadcasts_scalars_and_places_lists_per_joint():
    layout = ParameterLayout(num_joints=3)
    scalars = sim_params_from_values(layout, armature=0.1, damping=0.2, friction=0.3, bias=0.05, delay=4)
    assert scalars.shape == (13,)
    torch.testing.assert_close(scalars[layout.armature], torch.full((3,), 0.1))
    torch.testing.assert_close(scalars[layout.friction], torch.full((3,), 0.3))
    torch.testing.assert_close(scalars[layout.bias], torch.full((3,), 0.05))
    assert scalars[layout.delay].item() == 4.0

    lists = sim_params_from_values(layout, armature=[0.1, 0.2, 0.3], damping=[1.0, 2.0, 3.0],
                                   friction=[4.0, 5.0, 6.0], bias=[-0.1, 0.0, 0.1], delay=5)
    torch.testing.assert_close(lists[layout.armature], torch.tensor([0.1, 0.2, 0.3]))
    torch.testing.assert_close(lists[layout.damping], torch.tensor([1.0, 2.0, 3.0]))
    torch.testing.assert_close(lists[layout.friction], torch.tensor([4.0, 5.0, 6.0]))
    torch.testing.assert_close(lists[layout.bias], torch.tensor([-0.1, 0.0, 0.1]))
    assert lists[layout.delay].item() == 5.0


def test_friction_ratio_is_required_and_rejected_by_the_layout():
    separate = ParameterLayout(num_joints=3, separate_dynamic_friction=True)
    with pytest.raises(ValueError, match="friction_ratio is required"):
        sim_params_from_values(separate, 0.1, 0.2, 0.3, 0.0, 2)
    plain = ParameterLayout(num_joints=3)
    with pytest.raises(ValueError, match="friction_ratio given"):
        sim_params_from_values(plain, 0.1, 0.2, 0.3, 0.0, 2, friction_ratio=0.5)


def test_friction_ratio_fills_last_block_and_feeds_dynamic_friction():
    layout = ParameterLayout(num_joints=3, separate_dynamic_friction=True)
    sim = sim_params_from_values(layout, armature=0.1, damping=0.2, friction=[2.0, 4.0, 8.0], bias=0.0, delay=2,
                                 friction_ratio=[0.25, 0.5, 1.0])
    assert sim.shape == (16,)
    torch.testing.assert_close(sim[layout.friction_ratio], torch.tensor([0.25, 0.5, 1.0]))
    joints = JointParameters.from_sim_params(sim, layout)
    torch.testing.assert_close(joints.dynamic_friction, torch.tensor([[0.5, 2.0, 8.0]]))
