# © 2026 Kushagra Srivastava
# Licensed under the Apache License 2.0

"""Generate synthetic chirp data with known parameters using the MuJoCo backend (no Isaac Sim required).

The data is stored in the same format as ``scripts/pace/data_collection.py`` and additionally contains the
ground-truth parameters, so ``fit.py`` can report how well they are recovered.
"""

import argparse

from pace_sim2real.backends.mujoco.robots import ROBOTS
from pace_sim2real.core import generate_synthetic_data
from pace_sim2real.utils import project_root

parser = argparse.ArgumentParser(description="Synthetic PACE data collection with MuJoCo.")
parser.add_argument("--robot", type=str, default="pendulum", choices=sorted(ROBOTS), help="Robot configuration.")
parser.add_argument("--min_frequency", type=float, default=None, help="Minimum frequency of the chirp signal in Hz.")
parser.add_argument("--max_frequency", type=float, default=None, help="Maximum frequency of the chirp signal in Hz.")
parser.add_argument("--duration", type=float, default=None, help="Duration of the chirp signal in seconds.")
parser.add_argument("--noise_std", type=float, default=0.0, help="Measurement noise standard deviation in rad.")
parser.add_argument("--seed", type=int, default=0, help="Seed of the measurement noise.")
parser.add_argument("--plot", action="store_true", help="Plot the recorded trajectories.")
args_cli = parser.parse_args()


def main():
    cfg = ROBOTS[args_cli.robot]()
    if args_cli.min_frequency is not None:
        cfg.excitation.min_frequency = args_cli.min_frequency
    if args_cli.max_frequency is not None:
        cfg.excitation.max_frequency = args_cli.max_frequency
    if args_cli.duration is not None:
        cfg.excitation.duration = args_cli.duration

    t, targets = cfg.excitation_targets()
    print(f"[INFO]: Simulating {args_cli.robot} for {cfg.excitation.duration:.1f} s ({targets.shape[0]} steps).")
    data = generate_synthetic_data(cfg.make_backend(1), cfg.true_sim_params, cfg.layout, targets, time=t,
                                   noise_std=args_cli.noise_std, seed=args_cli.seed)

    data_file = project_root() / "data" / cfg.robot_name / "chirp_data.pt"
    data.save(data_file)
    print(f"[INFO]: Saved {data_file}")

    if args_cli.plot:
        import matplotlib.pyplot as plt

        for i, name in enumerate(cfg.joint_order):
            plt.figure()
            plt.plot(data.time.numpy(), data.dof_pos[:, i].numpy(), label=f"{name} pos")
            plt.plot(data.time.numpy(), data.des_dof_pos[:, i].numpy(), label=f"{name} target", linestyle="dashed")
            plt.title(f"Joint {name} Trajectory")
            plt.xlabel("Time [s]")
            plt.ylabel("Joint position [rad]")
            plt.grid()
            plt.legend()
            plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
