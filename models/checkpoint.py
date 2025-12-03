"""Checkpoint utilities for model loading and saving."""

import logging

import torch


def load_checkpoint(ckpt_path: str, logger: logging.Logger) -> dict | None:
    """Load model checkpoint from file.

    Args:
        ckpt_path: Path to checkpoint file
        logger: Logger instance

    Returns:
        Dictionary containing checkpoint data, or None if loading fails
    """
    logger.info("Loading checkpoint from: %s", ckpt_path)
    try:
        checkpoint = torch.load(ckpt_path, weights_only=True)
        logger.info("Checkpoint keys: %s", list(checkpoint.keys()))
        return checkpoint
    except FileNotFoundError:
        logger.error("Checkpoint file not found: %s", ckpt_path)
        return None
    except Exception as e:
        logger.error("Failed to load checkpoint: %s", e)
        return None
