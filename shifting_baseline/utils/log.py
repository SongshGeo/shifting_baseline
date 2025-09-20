#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from omegaconf import DictConfig


def setup_logger(
    console_level: str = "WARNING",
    file_level: str = "DEBUG",
    file_path: Optional[str] = None,
    logger_name: str = "shifting_baseline",
) -> logging.Logger:
    """
    Set up logging configuration for the application.

    Args:
        console_level: Console output level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        file_level: File output level
        file_path: Log file path, defaults to logs directory with timestamp
        logger_name: Name of the logger to configure

    Returns:
        Configured logger instance
    """
    if file_path is None:
        # Create logs directory in project root and default filename
        # Find project root by looking for pyproject.toml or similar
        current_path = Path.cwd()
        project_root = current_path

        # Walk up the directory tree to find project root
        for parent in current_path.parents:
            if (
                (parent / "pyproject.toml").exists()
                or (parent / "setup.py").exists()
                or (parent / "README.md").exists()
            ):
                project_root = parent
                break

        logs_dir = project_root / "logs"
        logs_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d")
        file_path = str(logs_dir / f"{logger_name}_{timestamp}.log")
    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    simple_formatter = logging.Formatter(fmt="%(levelname)s | %(message)s")

    # Get or create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)  # Set root level to DEBUG

    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, console_level.upper()))
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(file_path, encoding="utf-8")
    file_handler.setLevel(getattr(logging, file_level.upper()))
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    return logger


def setup_logger_from_hydra(cfg: "DictConfig") -> logging.Logger:
    """
    Set up logging to work with Hydra's job_logging system.

    This function doesn't override Hydra's logging configuration,
    but ensures our custom loggers use the same configuration.

    Args:
        cfg: Hydra configuration object

    Returns:
        Configured logger instance
    """
    # Get the root logger to ensure it's configured by Hydra
    root_logger = logging.getLogger()

    # If Hydra hasn't configured logging yet, use our default setup
    if not root_logger.handlers:
        return setup_logger()

    # Hydra has already configured logging, just return our logger
    return get_logger("shifting_baseline")


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance. If no name provided, returns the main logger.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance
    """
    if name is None:
        name = "shifting_baseline"

    logger = logging.getLogger(name)

    # Check if our main logger has handlers
    main_logger = logging.getLogger("shifting_baseline")
    if not main_logger.handlers:
        # Check if root logger has handlers (Hydra configuration)
        root_logger = logging.getLogger()
        if not root_logger.handlers:
            # No Hydra configuration, use our default setup for notebook environment
            setup_logger()

    return logger


def adjust_log_level(
    console_level: Optional[str] = None, file_level: Optional[str] = None
) -> None:
    """
    Dynamically adjust log levels for existing handlers.
    Useful for notebook environments where you want to change logging without restarting.

    Args:
        console_level: New console log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        file_level: New file log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Get our main logger
    main_logger = logging.getLogger("shifting_baseline")

    # Ensure we have a logger setup first
    if not main_logger.handlers:
        setup_logger()
        main_logger = logging.getLogger("shifting_baseline")

    # Adjust handlers on our main logger
    for handler in main_logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(
            handler, logging.FileHandler
        ):
            # Console handler
            if console_level:
                handler.setLevel(getattr(logging, console_level.upper()))
        elif isinstance(handler, logging.FileHandler):
            # File handler
            if file_level:
                handler.setLevel(getattr(logging, file_level.upper()))

    # Also update the logger level to the minimum of all handlers
    levels = []
    if console_level:
        levels.append(getattr(logging, console_level.upper()))
    if file_level:
        levels.append(getattr(logging, file_level.upper()))

    if levels:
        min_level = min(levels)
        main_logger.setLevel(min_level)


if __name__ == "__main__":
    # Test the logging setup
    logger = setup_logger(console_level="DEBUG", file_level="DEBUG")
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    logger.critical("Critical message")
