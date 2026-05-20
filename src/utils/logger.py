"""
src/utils/logger.py
--------------------
Simple logging utility — writes to console and optionally to a file.
"""

import logging
import os


def get_logger(name: str = "ML_Project", log_file: str = None,
               level: int = logging.INFO) -> logging.Logger:
    """
    Create and return a configured logger.

    Parameters
    ----------
    name     : str   — logger name
    log_file : str   — optional path to save logs to file
    level    : int   — logging level (default: INFO)

    Returns
    -------
    logger : logging.Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler (optional)
    if log_file:
        os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else ".", exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


if __name__ == "__main__":
    log = get_logger("test_logger", log_file="logs/test.log")
    log.info("Logger initialised successfully.")
    log.warning("This is a warning.")
    log.error("This is an error message.")