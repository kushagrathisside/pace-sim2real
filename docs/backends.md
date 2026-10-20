# Simulator backends

PACE separates the simulator-independent identification core from the simulators that produce the robot dynamics. The
core fits the actuator and joint parameters with CMA-ES. A backend simulates the robot, one copy per CMA-ES candidate,
and the core compares the simulated trajectories with the measured ones.

Two backends are available:

* **Isaac Lab** (`IsaacLabBackend`) is the original setup and requires Isaac Sim and Isaac Lab.
* **MuJoCo** (`MujocoBackend`) is a prototype that runs on the CPU without Isaac Sim.

---

## Overview

```text
validation          synthetic data, recovery reports, benchmark scripts
     |
core                parameters, objective, CMA-ES, fit()
     |              (imports no simulator)
     |
backend interface   SimulatorBackend          (backends/base.py)
     |              NumPy reference actuator   (backends/actuator.py)
     |
     +--> Isaac Lab backend   IsaacLabBackend  (backends/isaac/)
     |
     +--> MuJoCo backend      MujocoBackend    (backends/mujoco/)
```

The core owns the parameter model and the fitting loop and reaches a simulator only through the `SimulatorBackend`
interface; the backends implement that interface and depend on the core, never the other way around. Nothing under
`pace_sim2real/core/` imports a simulator, and nothing under `core/`, `optim/` or `backends/` imports Isaac Lab (the
Isaac Lab code lives in `tasks/`, `utils/pace_actuator*.py` and the Omniverse UI example). The import-hygiene tests
check these rules.

---

## Core API

The core is available from `pace_sim2real.core`. `fit()` takes a backend instance, `synthetic_recovery()` a factory
that creates backends with a given number of environments (`cfg.make_backend`), so the core does not depend on any
particular simulator. The example identifies the parameters of the built-in MuJoCo `pendulum` robot from synthetic
data with known ground truth:

```python
from pace_sim2real.backends.mujoco.robots import ROBOTS
from pace_sim2real.core import format_report, synthetic_recovery

cfg = ROBOTS["pendulum"]()  # MujocoPaceCfg: model, joints, bounds, actuator and ground truth
t, targets = cfg.excitation_targets()  # time stamps and joint position targets of the chirp
result = synthetic_recovery(cfg.make_backend, cfg.true_sim_params, cfg.bounds_params, cfg.layout, targets,
                            "logs/tmp", population_size=16, max_iteration=60, time=t)
print(format_report(result.report))
```

`cfg.make_backend(num_envs)` returns a `MujocoBackend` that simulates `num_envs` copies of the robot in one MuJoCo model.

The main modules of the core are:

| Module          | Contents                                                                                  |
| --------------- | ----------------------------------------------------------------------------------------- |
| `parameters.py` | `ParameterLayout`, `JointParameters`, bounds helpers, normalization                        |
| `objective.py`  | `trajectory_score`, `trajectory_rmse`, `step_error`                                       |
| `optimizer.py`  | `CMAESOptimizer`                                                                          |
| `fitting.py`    | `fit(backend, optimizer, data)`, which runs CMA-ES against any `SimulatorBackend`          |
| `data.py`       | `TrajectoryData`, `linear_chirp`, `shape_excitation`                                      |
| `validation.py` | `generate_synthetic_data`, `synthetic_recovery`, `recovery_report`, `format_report`       |
| `logger.py`     | `create_writer`, `NullWriter`: optional TensorBoard logging                                |

---

## Backend interface

A backend implements `SimulatorBackend` (`pace_sim2real/backends/base.py`). It simulates `num_envs` robots in parallel,
one per CMA-ES candidate, and exposes only the identified joints, in the order of `joint_names`.

| Member                                                                | Kind     | Description                                                                                                     |
| --------------------------------------------------------------------- | -------- | --------------------------------------------------------------------------------------------------------------- |
| `joint_names`                                                         | attribute | Identified joints, in the order used by all tensors.                                                            |
| `num_envs`                                                            | property | Number of robots simulated in parallel.                                                                         |
| `device`                                                              | property | Device of the tensors returned by the backend.                                                                  |
| `physics_dt`                                                          | property | Duration of one `step()` in seconds. The actuator delay is counted in these steps.                              |
| `reset()`                                                             | abstract | Reset the simulation and clear the actuator histories (delay buffers).                                          |
| `set_joint_parameters(params)`                                        | abstract | Write armature, viscous friction, static and dynamic friction, encoder bias and delay of every environment.     |
| `set_joint_state(joint_pos, joint_vel)`                               | abstract | Teleport the joints to positions (simulator frame) and velocities, both `(num_envs, num_joints)`.               |
| `step(joint_pos_targets)`                                             | abstract | Apply encoder-frame position targets `(num_envs, num_joints)` and advance by `physics_dt`.                      |
| `get_joint_positions()`                                               | abstract | Current joint positions `(num_envs, num_joints)`, simulator frame.                                              |
| `get_joint_velocities()`                                              | abstract | Current joint velocities `(num_envs, num_joints)`.                                                              |
| `close()`                                                             | optional | Release simulator resources.                                                                                    |
| `start_rollout(params, initial_joint_pos)`                            | provided | Reset, apply `params` and start every environment at rest at `initial_joint_pos + bias` (simulator frame).      |
| `rollout(params, initial_joint_pos, joint_pos_targets, on_step=None)` | provided | Simulate a target trajectory and return the simulator-frame positions `(num_steps, num_envs, num_joints)`.      |

Positions returned by the backend are in the simulator frame (the true joint angle). Targets and measurements are in the
encoder frame, `true - bias`. `rollout` records each position before the matching target is applied. If the `on_step`
callback returns `False`, the rollout stops and the result is truncated.

---

## Isaac Lab backend

`IsaacLabBackend` (`pace_sim2real/backends/isaac/adapter.py`) implements the interface on top of Isaac Lab. Installation
is unchanged, see [Installation](installation.md).

### What changed

* `scripts/pace/fit.py` builds an `IsaacLabBackend` and calls `pace_sim2real.core.fit`. Its command line is unchanged.
* `CMAESOptimizer` is implemented in `pace_sim2real.core`. The simulator-independent class has no `update_simulator`
  method. It adds the optional arguments `seed=0` and `use_tensorboard=True` and the method `joint_parameters()`.
* `import pace_sim2real` no longer imports Isaac Lab. `PaceCfg`, `PaceSim2realEnvCfg` and `PaceSim2realSceneCfg` are
  loaded on first access, and so are `PaceDCMotorCfg` and `PaceDCMotor` from `pace_sim2real.utils`. Existing imports
  such as `from pace_sim2real import PaceCfg` still work inside Isaac Lab.

### What did not change

* `import pace_sim2real.tasks` still registers the Gym environments.
* The existing Isaac code (`tasks/`, `utils/pace_actuator*.py`) stays where it was.
* No new hard dependency is added for Isaac Lab users.

### Backward compatibility

`pace_sim2real.CMAESOptimizer` keeps `update_simulator(articulation, joint_ids, initial_position)`, so code that imports
it from the top-level package keeps working.

### Verification status

Isaac Sim was not run for this refactor. The development machine has 10 GiB RAM and 8 GB VRAM, which is below the
requirements of Isaac Sim.

The Isaac path is verified with mocks only. A differential test with a mocked Isaac Lab environment checks that the
refactored path issues the same Isaac Lab API calls, in the same order, with identical tensors, and produces
bit-identical scores and checkpoint files as v0.2.0. A run on a real Isaac Sim installation has not been done yet.

---

## MuJoCo backend

`MujocoBackend` (`pace_sim2real/backends/mujoco/adapter.py`) simulates the robots with MuJoCo on the CPU. It is a
prototype.

### Installation

Use Python 3.10 to 3.12 and run the following from the repository root:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e "source/pace_sim2real[mujoco,tensorboard,test]"
```

The `mujoco` extra requires `mujoco>=3.3.4` (the first release whose `MjSpec` has `delete`, needed to weld floating bases; `attach` exists since 3.3.0). The test suite passes with MuJoCo 3.3.4 and 3.16.0.

### Workflow

Run from the repository root:

```bash
python scripts/pace_mujoco/data_collection.py --robot pendulum
python scripts/pace_mujoco/fit.py --robot pendulum
python scripts/pace/plot_trajectory.py --robot_name mujoco_pendulum --plot_trajectory
python scripts/pace_mujoco/benchmark.py --quick
```

* `data_collection.py` simulates the chirp excitation and writes `data/<robot_name>/chirp_data.pt`. The format is the
  same as for the Isaac data collection, with the ground-truth parameters added. Options: `--robot {leg,pendulum}`,
  `--noise_std`, `--duration`, `--min_frequency`, `--max_frequency`, `--seed`, `--plot`.
* `fit.py` runs CMA-ES and writes the logs to `logs/pace/<robot_name>/<timestamp>/`, in the same format as for Isaac.
  When the ground truth is known, it prints a recovery report. Options: `--robot {leg,pendulum}`,
  `--population_size`, `--max_iteration`, `--sigma`, `--seed`, `--data_file`.
* `plot_trajectory.py` works on these logs unchanged.
* `benchmark.py` runs the synthetic recovery benchmarks described below. Options: `--quick`, `--cases`, `--seed`.

### Parameter mapping

| PACE parameter             | MuJoCo implementation                                                                 |
| -------------------------- | ------------------------------------------------------------------------------------- |
| armature                   | `dof_armature`                                                                        |
| viscous friction (damping) | `dof_damping`, passive force `-damping * qvel`                                        |
| static friction            | `dof_frictionloss` while `abs(qvel) < static_friction_velocity`                       |
| dynamic friction           | `dof_frictionloss` otherwise                                                          |
| encoder bias               | offset of the measured position fed to the PD controller of `PaceDCMotorModel`        |
| delay                      | command or effort delay of `PaceDCMotorModel`                                         |

* MuJoCo has a single dry-friction coefficient, so the static and dynamic friction switch is an approximation of the
  PhysX friction model. The default `static_friction_velocity` is 0.01 rad/s.
* Encoder bias and delay are handled by the NumPy PACE actuator (`PaceDCMotorModel` in `backends/actuator.py`), which
  computes a PD torque on encoder-frame positions with the command or effort delay and the DC-motor limits. The torque
  is applied through `qfrc_applied`.
* The actuators defined in the MJCF are disabled, and so are all contacts. Floating bases are welded to the world
  (`fix_base=True`).
* All environments are replicated into one `MjModel` with `MjSpec`, and their joint names are prefixed `env{i}/`. One
  `mj_step` advances the whole population.

### Built-in robots

`ROBOTS` in `pace_sim2real/backends/mujoco/robots.py` contains two synthetic robots that need no assets. Each entry is a function that returns a
`MujocoPaceCfg` that bundles the MJCF model, the joint order, the search bounds, the actuator settings, the excitation,
the CMA-ES settings and, for synthetic data, the ground truth (`true_sim_params`).

| Name      | `robot_name`    | Joints                                                                         | Excitation           | Population / iterations |
| --------- | --------------- | ------------------------------------------------------------------------------ | -------------------- | ----------------------- |
| `pendulum` | `mujoco_pendulum` | one hinge joint (`joint`), Coulomb friction (static = dynamic)               | 5 s chirp, 0.1 to 5 Hz | 16 / 60                 |
| `leg`     | `mujoco_leg`    | `HAA` (about x), `HFE` and `KFE` (about y), separate static and dynamic friction | 10 s chirp, 0.1 to 10 Hz | 32 / 150                |

### Performance

Measured on the development machine (Intel i7-14650HX, CPU only):

* pendulum recovery (population 16, 60 iterations, 2000 steps): about 10 to 15 s;
* leg recovery (population 32, 150 iterations, 4000 steps): about 2 minutes;
* 12-joint ANYmal-sized model: about 0.5 ms per step for 64 environments, which is about 4 s per CMA-ES iteration for a
  20 s chirp. With 256 environments, about 19 s per iteration.

Populations of thousands, as used with Isaac Lab on the GPU, are not practical on the CPU.

### Findings

These are observations of this backend on the synthetic benchmarks, not general results.

1. The integer delay makes the objective piecewise constant. With small populations and an excitation limited to 5 Hz,
   CMA-ES locked onto a wrong delay (3 instead of 4 steps) and compensated with friction. Exciting up to 10 Hz fixed this.
2. With separately identified friction, the dynamic friction (static friction times the ratio) is recovered well. The
   static friction and the ratio are weakly observable with chirp excitation, because the joints rarely stick. For this
   reason the benchmark checks the dynamic friction.
3. The encoder bias is observable only through position-dependent forces in the simulator frame, for example gravity.

### Limitations

* The backend is a prototype and has not been validated on a real robot.
* The static and dynamic friction switch is an approximation (see the parameter mapping).
* There is no port of the Isaac task, scene or RL framework.
* Only hinge and slide joints can be identified.
* Results are not expected to match Isaac Lab (PhysX) numerically for the same parameters, because the solvers and
  friction models differ. They are comparable, not identical.

---

## Validation and tests

### Synthetic validation

`generate_synthetic_data` records a trajectory of a backend with known ground-truth parameters, in the same data format
as real measurements. `synthetic_recovery` generates such data, identifies the parameters again with CMA-ES and returns a
`RecoveryResult` with a recovery report.

`scripts/pace_mujoco/benchmark.py` runs the following cases:

| Case             | Robot     | Population / iterations | Tolerance | Measurement noise         |
| ---------------- | --------- | ----------------------- | --------- | ------------------------- |
| `pendulum`       | `pendulum` | 16 / 60                 | 5%        | none                      |
| `pendulum_noise` | `pendulum` | 16 / 60                 | 10%       | 1e-3 rad                  |
| `leg`            | `leg`     | 32 / 150                | 5%        | none, skipped by `--quick` |

A case passes if all of the following hold:

* the normalized error of armature, damping, (dynamic) friction and bias is within the tolerance, which is relative to
  the width of the search interval;
* the delay is recovered exactly;
* the final best score is below 1% of the initial best score;
* the estimate lies within the search bounds.

The script exits with status 1 if any case fails.

### Tests

```bash
cd source/pace_sim2real
python -m pytest tests -q            # fast suite
python -m pytest tests -q --runslow  # also runs the multi-joint leg recovery
```

The fast suite covers:

* the parameter layout, the objective and the data contract;
* the actuator semantics;
* the optimizer and the fit loop on a pure PyTorch toy backend, which needs no simulator;
* import hygiene: the core imports no simulator, `core/`, `optim/` and `backends/` import no Isaac Lab, and the
  package imports without Isaac Lab installed;
* the Isaac adapter, checked against v0.2.0 with mocks;
* the MuJoCo backend, through exact identities: a command delay equals shifted targets, an encoder bias equals
  shifted targets and start positions, and a candidate simulates bit-identically regardless of the other candidates
  in the shared model.

---

## Adding a new backend

1. Subclass `SimulatorBackend` (`pace_sim2real/backends/base.py`) and implement the abstract methods. Simulate one robot
   per environment and expose only the identified joints, in the order of `joint_names`.
2. Map `JointParameters` to the simulator: armature, viscous friction, static and dynamic friction, encoder bias and
   delay.
3. If the simulator has no actuator model, reuse `pace_sim2real.backends.actuator.PaceDCMotorModel`, as the MuJoCo
   backend does.
4. Identify the parameters with `pace_sim2real.core.fit`, and run the synthetic checks with `synthetic_recovery`.
5. Add an import-hygiene rule to the test suite, so that the core stays free of the new simulator.
