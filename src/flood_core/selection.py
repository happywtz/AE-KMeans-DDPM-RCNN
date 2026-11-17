"""Random candidate selection without validation-target-based screening."""

import numpy as np


def random_candidates(candidates, count=100, seed=42):
    """Return selected sequences and zero-based IDs in the supplied pool.

    This standalone NumPy selection helper is not a claim of bitwise identity
    with every archived pool or its generation/batching execution.
    """
    pool = np.asarray(candidates)
    if pool.ndim != 3 or pool.shape[1:] != (2, 48) or not np.isfinite(pool).all():
        raise ValueError("Finite candidate array of shape (n, 2, 48) required")
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)):
        raise ValueError("Selection count must be an integer")
    if not 1 <= count <= len(pool):
        raise ValueError("Selection count must be between 1 and pool size")
    ids = np.random.default_rng(seed).choice(len(pool), size=count, replace=False)
    return pool[ids].copy(), ids
