"""Common training utilities for apps.

Expose Trainer and TrainerConfig for application entrypoints to reuse.
"""

from .trainer import Trainer, TrainerConfig

__all__ = ["Trainer", "TrainerConfig"]
