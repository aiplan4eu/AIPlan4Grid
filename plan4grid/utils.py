import logging
import os
import shutil
from os.path import join as pjoin

import numpy as np

import plan4grid.config as cfg


def compute_size_array(array: np.ndarray) -> int:
    """Compute the size of an array.

    Args:
        array (np.ndarray): Array to compute the size of.

    Returns:
        int: Size of the array.
    """
    size = 1
    for i in array.shape:
        size *= i
    return size


def setup_logger(
    name: str,
    log_dir: str = cfg.LOG_DIR,
    level: int = logging.INFO,
) -> logging.Logger:
    """Setup a logger.

    Args:
        name (str): name of the logger.
        log_dir (str, optional): name of the log directory. Defaults to cfg.LOG_DIR.
        level (int, optional): level of the logger. Defaults to logging.INFO.

    Returns:
        logging.Logger: the logger.
    """
    os.makedirs(log_dir, exist_ok=True)
    filename = pjoin(log_dir, f"{name}{cfg.LOG_SUFFIX}")
    file_handler = logging.FileHandler(filename=filename)
    formatter = logging.Formatter("| %(levelname)-7s | %(asctime)s | %(message)s", datefmt="%I:%M")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    logger = logging.getLogger(name)
    logger.addHandler(file_handler)
    logger.setLevel(level)
    return logger


def strtobool(val: str) -> bool:
    """Convert a string representation of truth to true (1) or false (0).
    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    """
    val = val.lower()
    if val in ("y", "yes", "t", "true", "on", "1"):
        return True
    elif val in ("n", "no", "f", "false", "off", "0"):
        return False
    else:
        raise ValueError("invalid truth value %r" % (val,))


def clean_logs():
    """Remove recursively the log directory if it exists."""
    try:
        shutil.rmtree(cfg.LOG_DIR)
    except FileNotFoundError:
        pass
