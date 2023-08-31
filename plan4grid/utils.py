import logging
import sys
from os.path import join as pjoin

import numpy as np

import plan4grid.config as cfg


def verbose_print(verbose: bool) -> callable:
    """Return a function that prints only if verbose is True."""
    if verbose:

        def _vprint(*args, **kwargs):
            print(*args, **kwargs)

    else:
        _vprint = lambda *_, **__: None  # do-nothing function
    return _vprint


def compute_size_array(array: np.ndarray) -> int:
    """Compute the size of an numpy array."""
    size = 1
    for i in array.shape:
        size *= i
    return size


def setup_logger(name, log_dir, level=0):
    """To setup as many loggers as you want."""
    logging.basicConfig(
        filename=pjoin(log_dir, cfg.OUT_FILE),
        format="| %(levelname)-6s| %(asctime)s | %(message)s",
        datefmt="%I:%M",
        level=level,
        force=True,
    )

    sys.stderr.write = open(pjoin(log_dir, cfg.WARN_FILE), "w").write

    logger = logging.getLogger(name)

    return logger
