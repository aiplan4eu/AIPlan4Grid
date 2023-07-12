import numpy as np


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
