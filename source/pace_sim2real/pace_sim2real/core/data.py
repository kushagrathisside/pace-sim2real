# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

"""Trajectory data contract and excitation signals.

PACE consumes a dictionary (stored with :func:`torch.save`) with the keys

* ``time``: (num_steps,) time stamps [s],
* ``dof_pos``: (num_steps, num_joints) measured joint positions in the encoder frame [rad],
* ``des_dof_pos``: (num_steps, num_joints) commanded joint position targets [rad],

where the joint dimension follows ``joint_order``. Further keys are kept as metadata. This is the format written by
the data collection scripts, independent of the simulator or robot the data comes from.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import torch

REQUIRED_KEYS = ("time", "dof_pos", "des_dof_pos")


@dataclass
class TrajectoryData:
    """A recorded excitation trajectory following the PACE data contract."""

    time: torch.Tensor
    dof_pos: torch.Tensor
    des_dof_pos: torch.Tensor
    joint_order: list[str] | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.dof_pos.ndim != 2:
            raise ValueError(f"dof_pos must have shape (num_steps, num_joints), got {tuple(self.dof_pos.shape)}.")
        if self.des_dof_pos.shape != self.dof_pos.shape:
            raise ValueError(f"des_dof_pos {tuple(self.des_dof_pos.shape)} and dof_pos {tuple(self.dof_pos.shape)} must have the same shape.")
        if self.time.ndim != 1 or self.time.shape[0] != self.dof_pos.shape[0]:
            raise ValueError(f"time must have shape ({self.dof_pos.shape[0]},), got {tuple(self.time.shape)}.")
        if self.joint_order is not None and len(self.joint_order) != self.num_joints:
            raise ValueError(f"joint_order has {len(self.joint_order)} joints, but the data has {self.num_joints}.")
        for key in REQUIRED_KEYS:
            if not torch.all(torch.isfinite(getattr(self, key))):
                raise ValueError(f"{key} contains non-finite values.")

    @classmethod
    def from_dict(cls, data: dict, joint_order: list[str] | None = None) -> TrajectoryData:
        missing = [key for key in REQUIRED_KEYS if key not in data]
        if missing:
            raise KeyError(f"Trajectory data is missing the keys {missing}, expected {list(REQUIRED_KEYS)}.")
        metadata = {key: value for key, value in data.items() if key not in REQUIRED_KEYS and key != "joint_order"}
        if joint_order is None:
            joint_order = data.get("joint_order")
        return cls(time=data["time"], dof_pos=data["dof_pos"], des_dof_pos=data["des_dof_pos"],
                   joint_order=list(joint_order) if joint_order is not None else None, metadata=metadata)

    @classmethod
    def load(cls, path: str | Path, joint_order: list[str] | None = None, map_location="cpu") -> TrajectoryData:
        return cls.from_dict(torch.load(path, map_location=map_location), joint_order=joint_order)

    def as_dict(self) -> dict:
        """Dictionary in the stored format (the format :class:`CMAESOptimizer` expects as ``data``)."""
        data = dict(self.metadata)
        data.update({"time": self.time, "dof_pos": self.dof_pos, "des_dof_pos": self.des_dof_pos})
        if self.joint_order is not None:
            data["joint_order"] = list(self.joint_order)
        return data

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({key: value.cpu() if isinstance(value, torch.Tensor) else value for key, value in self.as_dict().items()}, path)

    def to(self, device) -> TrajectoryData:
        return TrajectoryData(self.time.to(device), self.dof_pos.to(device), self.des_dof_pos.to(device),
                              self.joint_order, dict(self.metadata))

    @property
    def num_steps(self) -> int:
        return self.dof_pos.shape[0]

    @property
    def num_joints(self) -> int:
        return self.dof_pos.shape[1]

    @property
    def initial_dof_pos(self) -> torch.Tensor:
        """Measured joint positions of the first sample (num_joints,), encoder frame."""
        return self.dof_pos[0]


def linear_chirp(duration: float, sample_rate: float, min_frequency: float, max_frequency: float,
                 device="cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """Unit amplitude linear chirp sweeping from ``min_frequency`` to ``max_frequency`` [Hz].

    Returns:
        Time stamps and signal, both of shape (round(duration * sample_rate),).
    """
    num_steps = round(duration * sample_rate)  # round, as e.g. 0.29 * 100 == 28.999999999999996
    t = torch.linspace(0, duration, steps=num_steps, device=device)
    # Linear chirp: phase = 2*pi*(f0*t + (f1-f0)/(2*duration)*t^2)
    phase = 2 * math.pi * (min_frequency * t + ((max_frequency - min_frequency) / (2 * duration)) * t ** 2)
    return t, torch.sin(phase)


def shape_excitation(signal: torch.Tensor, offset, direction, scale) -> torch.Tensor:
    """Turn a unit signal (num_steps,) into joint targets (num_steps, num_joints).

    ``target = (signal + offset) * direction * scale`` with per-joint ``offset``, ``direction`` and ``scale``.
    """
    offset = torch.as_tensor(offset, dtype=signal.dtype, device=signal.device)
    direction = torch.as_tensor(direction, dtype=signal.dtype, device=signal.device)
    scale = torch.as_tensor(scale, dtype=signal.dtype, device=signal.device)
    return (signal.unsqueeze(-1) + offset.unsqueeze(0)) * direction.unsqueeze(0) * scale.unsqueeze(0)
