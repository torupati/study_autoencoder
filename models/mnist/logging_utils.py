"""Logging utilities for model training."""

import logging


def setup_logging(log_file: str = "training.log") -> logging.Logger:
    """Setup logging to both console and file.

    Args:
        log_file: Path to log file

    Returns:
        Logger instance
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Clear existing handlers to avoid duplicates
    logger.handlers = []

    # Console handler (INFO level)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(module)s %(funcName)s %(message)s"
    )
    stream_handler.setFormatter(stream_formatter)

    # File handler (DEBUG level)
    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(module)s %(funcName)s %(message)s"
    )
    file_handler.setFormatter(file_formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger
