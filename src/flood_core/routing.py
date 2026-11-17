"""Forecast-time MLP extracted from the archived routing implementation."""

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def router_features(history_qp, future_p):
    """48 inputs: past 12 Q, past 12 P, then given future 24 P. No future Q."""
    h, p = np.asarray(history_qp), np.asarray(future_p)
    if h.ndim != 3 or h.shape[1:] != (2, 12) or not len(h):
        raise ValueError("history_qp must have shape (batch, 2, 12)")
    if p.shape != (len(h), 1, 24):
        raise ValueError("future_p must have shape (batch, 1, 24)")
    x = np.concatenate([h[:, 0], h[:, 1], p[:, 0]], axis=1).astype(float)
    if not np.isfinite(x).all():
        raise ValueError("Routing features must be finite")
    return x


def make_router(seed=42):
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", MLPClassifier(hidden_layer_sizes=(16,), activation="relu",
                                solver="lbfgs", alpha=0.01, max_iter=2000,
                                random_state=seed)),
    ])


def balance_for_mlp(x, y, rng):
    """Training-fold-only random oversampling; original selection order retained."""
    classes, counts = np.unique(y, return_counts=True)
    target = int(counts.max())
    selected = []
    for class_id, count in zip(classes, counts):
        indices = np.flatnonzero(y == class_id)
        selected.extend(indices.tolist())
        if count < target:
            selected.extend(rng.choice(indices, size=target-count, replace=True).tolist())
    selected_array = np.asarray(selected, dtype=int)
    rng.shuffle(selected_array)
    return x[selected_array], y[selected_array]


def fit_router(training_events, event_labels, seed=42):
    """Fit the archived small MLP on the 13 windows of each training event.

    Caller supplies training events only, in physical units, (n, 48, 2).
    Teacher labels must come from the same training-side AE clustering.
    """
    events = np.asarray(training_events, dtype=float)
    labels = np.asarray(event_labels)
    if events.ndim != 3 or events.shape[1:] != (48, 2) or not len(events):
        raise ValueError("Expected training events with shape (n, 48, 2)")
    if not np.isfinite(events).all() or labels.shape != (len(events),):
        raise ValueError("Finite events and one label per training event required")
    if np.unique(labels).size < 2:
        raise ValueError("At least two training classes required")
    features = []
    for event in events:
        for start in range(13):
            window = event[start:start+36]
            features.append(np.concatenate([window[:12, 0], window[:12, 1],
                                            window[12:, 1]]))
    x = np.asarray(features)
    y = np.repeat(labels, 13)
    x, y = balance_for_mlp(x, y, np.random.default_rng(seed+99999))
    model = make_router(seed)
    model.fit(x, y)
    return model
