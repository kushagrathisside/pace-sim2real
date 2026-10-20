# © 2026 Kushagra Srivastava
# Licensed under the Apache License 2.0

"""Identify the PACE parameters with CMA-ES using the MuJoCo backend (CPU only, no Isaac Sim required).

The logs are written to ``logs/pace/<robot_name>`` like ``scripts/pace/fit.py``, so ``scripts/pace/plot_trajectory.py``
can plot the result.
"""

import argparse

from pace_sim2real.backends.mujoco.robots import ROBOTS
from pace_sim2real.core import CMAESOptimizer, TrajectoryData, fit, format_report, recovery_report
from pace_sim2real.utils import project_root

parser = argparse.ArgumentParser(description="PACE parameter identification with MuJoCo.")
parser.add_argument("--robot", type=str, default="pendulum", choices=sorted(ROBOTS), help="Robot configuration.")
parser.add_argument("--data_file", type=str, default=None, help="Data file, defaults to data/<robot_name>/chirp_data.pt.")
parser.add_argument("--population_size", type=int, default=None, help="CMA-ES population size (simulated robots).")
parser.add_argument("--max_iteration", type=int, default=None, help="Maximum number of CMA-ES iterations.")
parser.add_argument("--sigma", type=float, default=None, help="Initial CMA-ES step size in the normalized space.")
parser.add_argument("--seed", type=int, default=0, help="CMA-ES seed.")
args_cli = parser.parse_args()


def main():
    cfg = ROBOTS[args_cli.robot]()
    population_size = args_cli.population_size or cfg.cmaes.population_size
    max_iteration = args_cli.max_iteration or cfg.cmaes.max_iteration
    sigma = args_cli.sigma or cfg.cmaes.sigma

    data_file = args_cli.data_file or project_root() / "data" / cfg.robot_name / "chirp_data.pt"
    log_dir = project_root() / "logs" / "pace" / cfg.robot_name
    data = TrajectoryData.load(data_file, joint_order=cfg.joint_order)

    backend = cfg.make_backend(population_size)
    opt = CMAESOptimizer(
        bounds=cfg.bounds_params,
        population_size=population_size,
        log_dir=log_dir,
        joint_order=cfg.joint_order,
        max_iteration=max_iteration,
        data=data,
        device=backend.device,
        epsilon=cfg.cmaes.epsilon,
        sigma=sigma,
        save_interval=cfg.cmaes.save_interval,
        save_optimization_process=cfg.cmaes.save_optimization_process,
        separate_dynamic_friction=cfg.separate_dynamic_friction,
        seed=args_cli.seed,
    )
    best = fit(backend, opt, data, log_interval=0)
    opt.close()
    backend.close()

    print(f"[INFO]: Logs written to {opt.log_dir}")
    if "true_sim_params" in data.metadata:
        report = recovery_report(data.metadata["true_sim_params"], best, cfg.bounds_params, cfg.layout, cfg.joint_order)
        print("[INFO]: Recovery of the ground-truth parameters of the synthetic data:")
        print(format_report(report))


if __name__ == "__main__":
    main()
