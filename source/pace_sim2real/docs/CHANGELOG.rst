Changelog
---------

Unreleased
~~~~~~~~~~

Added
^^^^^

* Added the simulator-independent core package ``pace_sim2real.core`` with the parameter model, the objective, the
  CMA-ES optimizer, ``fit()``, the data contract and the validation utilities. The core imports no simulator.
* Added the simulator backend interface ``pace_sim2real.backends.base.SimulatorBackend``.
* Added the Isaac Lab backend adapter ``IsaacLabBackend`` in ``pace_sim2real.backends.isaac``.
* Added the MuJoCo backend ``MujocoBackend`` in ``pace_sim2real.backends.mujoco`` (prototype), with the built-in
  synthetic robots ``pendulum`` and ``leg``.
* Added the NumPy reference actuator ``PaceDCMotorModel`` in ``pace_sim2real.backends.actuator``.
* Added synthetic validation utilities (``synthetic_recovery``, recovery reports) and the MuJoCo scripts
  ``scripts/pace_mujoco/data_collection.py``, ``scripts/pace_mujoco/fit.py`` and ``scripts/pace_mujoco/benchmark.py``.
* Added a pytest suite in ``source/pace_sim2real/tests``.
* Added the ``core``, ``mujoco``, ``tensorboard`` and ``test`` extras to ``setup.py``.

Changed
^^^^^^^

* ``import pace_sim2real`` no longer imports Isaac Lab. ``PaceCfg``, ``PaceSim2realEnvCfg`` and
  ``PaceSim2realSceneCfg`` are loaded lazily on first access, and so are ``PaceDCMotorCfg`` and ``PaceDCMotor`` in
  ``pace_sim2real.utils``. As a consequence, ``import pace_sim2real`` alone no longer registers the Gym environments
  as a side effect; ``import pace_sim2real.tasks`` does, as all scripts and the documentation already do.
* ``scripts/pace/fit.py`` uses an ``IsaacLabBackend`` and the core ``fit()`` loop. The command line and the outputs
  are unchanged.
* ``CMAESOptimizer`` moved to ``pace_sim2real.core``. The old import path ``pace_sim2real.CMAESOptimizer`` keeps
  ``update_simulator``.
* The optimizer checks the bounds before it creates the log directory.
* TensorBoard is optional.

Fixed
^^^^^

* ``setup.py`` now installs all subpackages. Previously only the top-level package was listed, which only worked for
  editable installs.
* ``CMAESOptimizer.get_best_sim_params`` creates the mean on the device of the bounds.


0.2.0 (2026-10-13)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added ``delay_mode`` to :class:`PaceDCMotorCfg` to choose where the actuator delay acts. ``"command"`` delays the
  position/velocity targets and feed-forward torques, ``"effort"`` delays the computed torque.
* Added ``separate_dynamic_friction`` to :class:`PaceCfg` to identify static and dynamic friction separately. Dynamic
  friction is identified as a ratio of the static friction within ``friction_ratio_bounds`` (defaults to ``(0.0, 1.0)``),
  which appends one parameter per joint after the delay.
* Added a GitHub star history page to the documentation.

Changed
^^^^^^^

* The ANYmal example identifies static and dynamic friction separately (61 instead of 49 parameters), and the
  synthetic data uses a dynamic friction of half the static friction.
* Split the license file into license and copyright notice.
* Updated the citation to the published IJRR article.

Fixed
^^^^^

* **Breaking:** The actuator delay was applied to the computed torque, while the paper describes a delay of the
  commands. The delay now acts on the position/velocity targets and feed-forward torques (``delay_mode="command"``,
  the new default), which also keeps the PD loop stable for large delays. The results in the paper were produced with
  the previous implementation, which remains available via ``delay_mode="effort"``.
* Fixed ``plot_trajectory.py`` printing the last parameter instead of the delay.
* Removed the deprecated polyfill.io dependency from the documentation.


0.1.2 (2026-03-12)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Set both static and dynamic friction in the simulator to model Coulomb friction in Isaac Sim 5.0.
* Improved ``encoder_bias`` parsing of :class:`PaceDCMotor`.


0.1.1 (2026-01-30)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Disabled the default joint position offset of the actions to avoid spawn failures on robots where 0.0 is not a
  feasible joint position.
* Fixed the documentation deployment and added contributing guidelines.


0.1.0 (2025-09-30)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Created an initial template for building an extension or project based on Isaac Lab
