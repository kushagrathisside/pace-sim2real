# © 2026 Kushagra Srivastava
# Licensed under the Apache License 2.0

"""Synthetic parameter recovery benchmarks with the MuJoCo backend.

Every case generates data with known parameters, identifies them with CMA-ES and checks the recovery error against a
tolerance (relative to the width of the search interval). Exits with status 1 if any case fails.
"""

import argparse
import sys
import tempfile
import time
from dataclasses import dataclass

from pace_sim2real.backends.mujoco.robots import ROBOTS
from pace_sim2real.core import format_report, max_normalized_error, synthetic_recovery, within_bounds


@dataclass
class Case:
    name: str
    robot: str
    population_size: int
    max_iteration: int
    tolerance: float
    """Largest allowed normalized error of armature, damping, (dynamic) friction and bias."""
    noise_std: float = 0.0
    quick: bool = True


CASES = [
    Case("pendulum", "pendulum", population_size=16, max_iteration=60, tolerance=0.05),
    Case("pendulum_noise", "pendulum", population_size=16, max_iteration=60, tolerance=0.10, noise_std=1e-3),
    Case("leg", "leg", population_size=32, max_iteration=150, tolerance=0.05, quick=False),
]

parser = argparse.ArgumentParser(description="Synthetic PACE parameter recovery benchmarks (MuJoCo).")
parser.add_argument("--quick", action="store_true", help="Only run the fast single-joint cases.")
parser.add_argument("--cases", nargs="*", default=None, help=f"Subset of cases: {[case.name for case in CASES]}.")
parser.add_argument("--seed", type=int, default=0, help="Seed of CMA-ES and the measurement noise.")
args_cli = parser.parse_args()


def run(case: Case, seed: int) -> bool:
    cfg = ROBOTS[case.robot]()
    bounds = cfg.bounds_params
    t, targets = cfg.excitation_targets()

    start = time.time()
    with tempfile.TemporaryDirectory() as log_dir:
        result = synthetic_recovery(cfg.make_backend, cfg.true_sim_params, bounds, cfg.layout, targets, log_dir,
                                    population_size=case.population_size, max_iteration=case.max_iteration,
                                    noise_std=case.noise_std, seed=seed, time=t)
    elapsed = time.time() - start

    # with separately identified friction, the dynamic friction is checked: the static friction only acts while a joint
    # sticks, which a chirp excitation hardly excites (it is reported, but not checked)
    friction = "dynamic_friction" if cfg.separate_dynamic_friction else "friction"
    continuous_error = max_normalized_error(result.report, ("armature", "damping", friction, "bias"))
    delay_error = max_normalized_error(result.report, ("delay",))
    checks = {
        f"continuous parameters within {case.tolerance:.0%}": continuous_error <= case.tolerance,
        "delay exact": delay_error == 0.0,
        "score decreased 100x": bool(result.best_scores[-1] < 1e-2 * result.best_scores[0]),
        "estimate within bounds": within_bounds(result.estimated_sim_params, bounds),
    }
    passed = all(checks.values())
    print(f"\n=== {case.name}: {'PASS' if passed else 'FAIL'} ({elapsed:.1f} s, population {case.population_size}, "
          f"{case.max_iteration} iterations, {targets.shape[0]} steps, noise {case.noise_std} rad)")
    print(format_report(result.report))
    print(f"best score: {result.best_scores[0]:.3e} -> {result.best_scores[-1]:.3e}")
    for check, ok in checks.items():
        print(f"  [{'x' if ok else ' '}] {check}")
    return passed


def main():
    cases = [case for case in CASES if (args_cli.cases is None or case.name in args_cli.cases)
             and (not args_cli.quick or case.quick)]
    results = {case.name: run(case, args_cli.seed) for case in cases}
    print("\nSummary: " + ", ".join(f"{name} {'PASS' if ok else 'FAIL'}" for name, ok in results.items()))
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
