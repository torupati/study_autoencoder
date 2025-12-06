"""Utility functions for plotting training results."""

import logging
import os

import matplotlib.pyplot as plt


def plot_loss(
    train_loss_log: list[float],
    test_loss_log: list[float],
    output_dir: str,
    logger: logging.Logger,
    title: str = "Loss",
    filename: str = "loss.png",
) -> None:
    """Plot and save training/test loss curves.

    Args:
        train_loss_log: List of training losses
        test_loss_log: List of test losses
        output_dir: Directory to save the plot
        logger: Logger instance
        title: Title for the plot
        filename: Output filename for the plot
    """
    plt.figure(figsize=(10, 6))
    plt.suptitle(title)
    plt.plot(train_loss_log, label="train_loss", marker="o")
    plt.plot(test_loss_log, label="test_loss", marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()

    loss_plot_path = os.path.join(output_dir, filename)
    plt.savefig(loss_plot_path, dpi=100, bbox_inches="tight")
    logger.info(f"Saved loss plot to {loss_plot_path}")
    plt.close()
