"""Training-side Q/P scaling, without loading observations or fitted values."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QPScaler:
    q_min: float
    q_range: float
    p_min: float
    p_range: float

    def __post_init__(self):
        values = (self.q_min, self.q_range, self.p_min, self.p_range)
        if not np.isfinite(values).all() or min(self.q_range, self.p_range) <= 0:
            raise ValueError("Finite parameters and positive channel ranges required")

    @classmethod
    def fit(cls, training_events):
        """Fit only permitted training events, shaped (events, 48, 2), Q then P."""
        x = np.asarray(training_events, dtype=np.float64)
        if x.ndim != 3 or x.shape[1:] != (48, 2) or not len(x):
            raise ValueError("Expected nonempty training events of shape (n, 48, 2)")
        if not np.isfinite(x).all():
            raise ValueError("Training observations must be finite")
        lo, hi = x.min(axis=(0, 1)), x.max(axis=(0, 1))
        return cls(float(lo[0]), float(hi[0] - lo[0]),
                   float(lo[1]), float(hi[1] - lo[1]))

    def transform(self, events):
        x = np.asarray(events, dtype=np.float64)
        if x.ndim < 2 or x.shape[-1] != 2 or not np.isfinite(x).all():
            raise ValueError("Finite values with final Q/P axis required")
        # Do not clip held-out values to the training range.
        return (x - [self.q_min, self.p_min]) / [self.q_range, self.p_range]

    def inverse(self, events):
        x = np.asarray(events, dtype=np.float64)
        if x.ndim < 2 or x.shape[-1] != 2 or not np.isfinite(x).all():
            raise ValueError("Finite values with final Q/P axis required")
        return x * [self.q_range, self.p_range] + [self.q_min, self.p_min]

    def inverse_q(self, discharge):
        return np.asarray(discharge) * self.q_range + self.q_min
