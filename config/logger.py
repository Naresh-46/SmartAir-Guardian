"""
Logging configuration for SmartAir Guardian application.

Provides centralized logging setup for consistent logging across all modules.
Supports both file and console output with configurable log levels.
"""

import logging
import logging.handlers
from typing import Optional
from pathlib import Path

from config.settings import LOG_DIR, LOG_LEVEL, LOG_FORMAT, LOG_FILE


def setup_logger(
    name: str,
    log_file: Optional[Path] = None,
    level: str = LOG_LEVEL,
    fmt: str = LOG_FORMAT
) -> logging.Logger:
    """
    Configure and return a logger instance.
    
    Parameters
    ----------
    name : str
        The name of the logger (typically the module name).
    log_file : Path, optional
        Path to log file. If None, uses the default from settings.
    level : str
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    fmt : str
        Log message format string.
    
    Returns
    -------
    logging.Logger
        Configured logger instance.
    
    Examples
    --------
    >>> logger = setup_logger(__name__)
    >>> logger.info("Application started")
    """
    if log_file is None:
        log_file = LOG_FILE
    
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    # Prevent duplicate handlers
    if logger.hasHandlers():
        return logger
    
    # Formatter
    formatter = logging.Formatter(fmt)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler with rotation
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10_000_000,  # 10 MB
        backupCount=5
    )
    file_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.
    
    Parameters
    ----------
    name : str
        The module name (typically __name__).
    
    Returns
    -------
    logging.Logger
        Logger instance for the module.
    
    Examples
    --------
    >>> from config.logger import get_logger
    >>> logger = get_logger(__name__)
    """
    return setup_logger(name)
