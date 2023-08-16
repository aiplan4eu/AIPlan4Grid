import logging
import sys
from os.path import join as pjoin

import numpy as np

import config as cfg


def verbose_print(verbose: bool):
    if verbose:

        def _vprint(*args, **kwargs):
            print(*args, **kwargs)

    else:
        _vprint = lambda *_, **__: None  # do-nothing function
    return _vprint


def compute_size_array(array: np.ndarray) -> int:
    size = 1
    for i in array.shape:
        size *= i
    return size


def setup_logger(name, log_dir, level=0):
    """To setup as many loggers as you want"""
    logging.basicConfig(
        filename=pjoin(log_dir, cfg.OUT_FILE),
        format="| %(levelname)s | %(asctime)s | %(message)s",
        datefmt="%I:%M",
        level=level,
        force=True,
    )

    sys.stderr.write = open(pjoin(log_dir, cfg.ERR_FILE), "w").write

    logger = logging.getLogger(name)

    return logger
