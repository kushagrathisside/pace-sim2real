# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

"""Backward compatible optimizer API.

The optimizer lives in :mod:`pace_sim2real.core.optimizer`. This subclass keeps the Isaac Lab specific
:meth:`CMAESOptimizer.update_simulator` for existing scripts, new code should use a backend instead.
"""

from __future__ import annotations

from pace_sim2real.core.optimizer import CMAESOptimizer as _CoreCMAESOptimizer


class CMAESOptimizer(_CoreCMAESOptimizer):

    def update_simulator(self, articulation, joint_ids, initial_position):
        """Write the current population into an Isaac Lab articulation (see :class:`IsaacLabBackend`)."""
        from pace_sim2real.backends.isaac.adapter import apply_pace_parameters

        apply_pace_parameters(articulation, joint_ids, self.joint_parameters(), initial_position)
