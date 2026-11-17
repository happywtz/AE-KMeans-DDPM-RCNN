"""Absolute window scores on physical-scale, unmodified predictions."""

import numpy as np


def window_metrics(observed_q, predicted_q):
    """Return per-window NSE, RMSE, MAE and Pearson R (windows, 24).

    Undefined NSE or R is NaN, not replaced by an arbitrary finite score.
    Negative predictions and negative NSE remain in the evaluation.
    """
    obs, pred = np.asarray(observed_q, float), np.asarray(predicted_q, float)
    if obs.ndim != 2 or obs.shape[1] != 24 or pred.shape != obs.shape or not len(obs):
        raise ValueError("Matching nonempty arrays of shape (windows, 24) required")
    if not np.isfinite(obs).all() or not np.isfinite(pred).all():
        raise ValueError("Observed and predicted values must be finite")
    err = pred - obs
    oc = obs - obs.mean(axis=1, keepdims=True)
    pc = pred - pred.mean(axis=1, keepdims=True)
    denominator = np.sum(oc**2, axis=1)
    nse = np.full(len(obs), np.nan)
    valid = denominator > 0
    nse[valid] = 1 - np.sum(err**2, axis=1)[valid] / denominator[valid]
    rden = np.sqrt(denominator * np.sum(pc**2, axis=1))
    corr = np.full(len(obs), np.nan)
    valid_r = rden > 0
    corr[valid_r] = np.sum(oc*pc, axis=1)[valid_r] / rden[valid_r]
    return {"NSE": nse, "RMSE": np.sqrt(np.mean(err**2, axis=1)),
            "MAE": np.mean(np.abs(err), axis=1), "R": corr}


def mean_window_metrics(observed_q, predicted_q):
    """Arithmetic means of window scores; NaNs propagate and are not dropped."""
    return {key: float(np.mean(values))
            for key, values in window_metrics(observed_q, predicted_q).items()}
