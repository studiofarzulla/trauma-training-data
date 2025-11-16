"""
Logging configuration for trauma models.

Provides consistent logging across all experiments.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "trauma_models",
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    verbose: bool = True
) -> logging.Logger:
    """
    Configure logger for trauma models.

    Args:
        name: Logger name
        level: Logging level (logging.DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for log output
        verbose: If True, also log to console

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers = []

    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    if verbose:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "trauma_models") -> logging.Logger:
    """
    Get existing logger or create new one.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)

    # If logger not yet configured, set up with defaults
    if not logger.handlers:
        setup_logger(name)

    return logger
