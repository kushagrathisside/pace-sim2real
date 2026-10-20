# © 2026 Kushagra Srivastava
# Licensed under the Apache License 2.0

import numpy as np
import pytest

from pace_sim2real.backends.actuator import DelayBuffer, PaceDCMotorModel, PaceDCMotorParams

# Isaac Lab DCMotor-style limits: 10 Nm at standstill, 8 Nm continuous, 5 rad/s no-load speed
DC_LIMITS = dict(stiffness=85.0, damping=0.6, effort_limit=8.0, saturation_effort=10.0, velocity_limit=5.0)


def _filled(value, shape):
    return np.full(shape, float(value))


def test_delay_buffer_gives_each_env_its_own_lag():
    buffer = DelayBuffer(max_delay=3, num_envs=2, num_joints=1)
    buffer.set_time_lag([0, 2])
    outputs = np.stack([buffer.compute(_filled(k, (2, 1))) for k in range(1, 6)])  # (steps, envs, joints)
    np.testing.assert_array_equal(outputs[:, 0, 0], [1, 2, 3, 4, 5])  # lag 0: the current input
    np.testing.assert_array_equal(outputs[:, 1, 0], [1, 1, 1, 2, 3])  # lag 2: the input of step max(k - 2, 0)


def test_delay_buffer_reset_refills_the_history_with_the_first_input():
    buffer = DelayBuffer(max_delay=3, num_envs=2, num_joints=1)
    buffer.set_time_lag([0, 2])
    for value in (1.0, 2.0, 3.0):
        buffer.compute(_filled(value, (2, 1)))
    buffer.reset()
    after_reset = [buffer.compute(_filled(value, (2, 1))) for value in (7.0, 8.0, 9.0, 10.0)]
    np.testing.assert_array_equal([out[0, 0] for out in after_reset], [7, 8, 9, 10])
    np.testing.assert_array_equal([out[1, 0] for out in after_reset], [7, 7, 7, 8])


@pytest.mark.parametrize("lag", [-1, 4])
def test_delay_buffer_rejects_lags_outside_zero_to_max_delay(lag):
    buffer = DelayBuffer(max_delay=3, num_envs=2, num_joints=1)
    with pytest.raises(ValueError, match="Delays must lie in"):
        buffer.set_time_lag([0, lag])


def test_delay_buffer_scalar_lag_applies_to_all_envs():
    buffer = DelayBuffer(max_delay=3, num_envs=2, num_joints=1)
    buffer.set_time_lag(1)
    outputs = np.stack([buffer.compute(_filled(k, (2, 1))) for k in range(1, 5)])
    np.testing.assert_array_equal(outputs[:, 0, 0], [1, 1, 2, 3])
    np.testing.assert_array_equal(outputs[:, 1, 0], [1, 1, 2, 3])


def test_torque_is_pd_law_on_encoder_frame_positions_with_default_limits():
    motor = PaceDCMotorModel(PaceDCMotorParams(stiffness=50.0, damping=2.0), num_envs=2, num_joints=3)
    target = np.array([[0.3, -0.1, 0.0], [1.0, 0.5, -0.5]])
    q = np.array([[0.1, 0.2, -0.3], [0.0, 0.4, 0.1]])
    qd = np.array([[0.5, -1.0, 2.0], [0.0, 3.0, -0.5]])
    torque = motor.compute(target, q, qd)
    np.testing.assert_allclose(torque, 50.0 * (target - q) + 2.0 * (0.0 - qd))


def test_per_joint_gains_are_broadcast_over_joints():
    params = PaceDCMotorParams(stiffness=[10.0, 20.0], damping=[1.0, 2.0])
    motor = PaceDCMotorModel(params, num_envs=1, num_joints=2)
    torque = motor.compute(np.array([[0.5, -0.25]]), np.zeros((1, 2)), np.ones((1, 2)))
    np.testing.assert_allclose(torque, [[10.0 * 0.5 - 1.0 * 1.0, 20.0 * -0.25 - 2.0 * 1.0]])


def test_encoder_bias_shifts_the_position_the_controller_sees():
    bias = np.array([[0.05, -0.1]])
    target = np.array([[0.3, -0.2]])
    motor = PaceDCMotorModel(PaceDCMotorParams(stiffness=85.0, damping=0.6), num_envs=1, num_joints=2)
    motor.update_encoder_bias(bias)
    # the controller reads q - bias, so a measurement q = target + bias is exactly on target
    torque = motor.compute(target, target + bias, np.zeros((1, 2)))
    np.testing.assert_allclose(torque, 0.0, atol=1e-12)
    # without the bias correction the same measurement is off target by the bias
    unbiased = PaceDCMotorModel(PaceDCMotorParams(stiffness=85.0, damping=0.6), num_envs=1, num_joints=2)
    np.testing.assert_allclose(unbiased.compute(target, target + bias, np.zeros((1, 2))), -85.0 * bias)


def test_command_delay_postpones_a_step_in_the_position_target_by_the_lag():
    motor = PaceDCMotorModel(PaceDCMotorParams(stiffness=10.0, damping=0.0), num_envs=1, num_joints=1)
    motor.update_time_lags(2)
    still = np.zeros((1, 1))
    torques = [motor.compute(np.full((1, 1), target), still, still)[0, 0] for target in (0.0, 1.0, 1.0, 1.0, 1.0)]
    # the step at call 1 reaches the torque two calls later, at call 3
    np.testing.assert_allclose(torques, [0.0, 0.0, 0.0, 10.0, 10.0])


def test_effort_delay_postpones_the_computed_torque_instead_of_the_target():
    params = PaceDCMotorParams(stiffness=10.0, damping=0.0, delay_mode="effort")
    motor = PaceDCMotorModel(params, num_envs=1, num_joints=1)
    motor.update_time_lags(2)
    still = np.zeros((1, 1))
    applied, computed = [], []
    for target in (0.0, 1.0, 1.0, 1.0):
        applied.append(motor.compute(np.full((1, 1), target), still, still)[0, 0])
        computed.append(motor.computed_effort[0, 0])
    np.testing.assert_allclose(computed, [0.0, 10.0, 10.0, 10.0])  # the PD torque reacts immediately
    np.testing.assert_allclose(applied, [0.0, 0.0, 0.0, 10.0])  # the applied torque is delayed by two calls


@pytest.mark.parametrize(("velocity", "max_torque", "min_torque"), [
    (0.0, 8.0, -8.0),  # standstill: the 10 Nm saturation is cut to the 8 Nm effort limit
    (2.5, 5.0, -8.0),  # half the no-load speed: the positive limit is 10 * (1 - 0.5) = 5 Nm
    (5.0, 0.0, -8.0),  # no-load speed: no positive torque is left
    (6.0, 0.0, -8.0),  # beyond no-load speed the positive limit is negative and clipped to zero
    (-2.5, 8.0, -5.0),  # the opposite direction mirrors the limits
])
def test_dc_motor_torque_speed_limits(velocity, max_torque, min_torque):
    motor = PaceDCMotorModel(PaceDCMotorParams(**DC_LIMITS), num_envs=1, num_joints=1)
    still, speed = np.zeros((1, 1)), np.full((1, 1), velocity)
    # a huge position error saturates the PD command in either direction
    assert motor.compute(np.full((1, 1), 1e3), still, speed)[0, 0] == pytest.approx(max_torque)
    assert motor.compute(np.full((1, 1), -1e3), still, speed)[0, 0] == pytest.approx(min_torque)


def test_dc_motor_passes_torque_inside_the_limits_unchanged():
    motor = PaceDCMotorModel(PaceDCMotorParams(**DC_LIMITS), num_envs=1, num_joints=1)
    torque = motor.compute(np.full((1, 1), 0.02), np.zeros((1, 1)), np.zeros((1, 1)))
    assert torque[0, 0] == pytest.approx(85.0 * 0.02)  # 1.7 Nm, well inside the 8 Nm limit


def test_invalid_delay_mode_raises_value_error():
    with pytest.raises(ValueError, match="delay_mode"):
        PaceDCMotorModel(PaceDCMotorParams(delay_mode="position"), num_envs=1, num_joints=1)
