# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

"""Optimization logging. TensorBoard is optional, without it the optimizer still writes its checkpoints."""

from __future__ import annotations

import warnings


class NullWriter:
    """Drop-in replacement for the TensorBoard ``SummaryWriter`` that discards all summaries."""

    def __init__(self, log_dir: str):
        self.log_dir = log_dir

    def add_scalar(self, *args, **kwargs):
        pass

    def add_histogram(self, *args, **kwargs):
        pass

    def close(self):
        pass


def create_writer(log_dir: str, use_tensorboard: bool = True):
    """Return a TensorBoard ``SummaryWriter`` for ``log_dir``, or a :class:`NullWriter` if TensorBoard is unavailable."""
    if use_tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            warnings.warn("TensorBoard is not installed, optimization summaries are not logged (pip install tensorboard).",
                          stacklevel=2)
        else:
            return SummaryWriter(log_dir=log_dir)
    return NullWriter(log_dir)
