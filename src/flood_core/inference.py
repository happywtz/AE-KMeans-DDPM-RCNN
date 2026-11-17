"""MLP class assignment followed by the matching class-specific RCNN."""

from dataclasses import dataclass

import numpy as np
import torch

from .preprocessing import QPScaler
from .routing import router_features


@dataclass
class ForecastExpert:
    model: torch.nn.Module
    scaler: QPScaler


def forecast(router, experts, history_qp, future_p):
    """Return physical-scale Q and route IDs, using no future observed Q.

    Inputs are physical-unit arrays (batch, 2, 12) and (batch, 1, 24).
    Supply trained experts and their own training-fitted scalers, keyed by
    the router's actual label values. No cross-fold label numbering is assumed.
    """
    h, p = np.asarray(history_qp, dtype=float), np.asarray(future_p, dtype=float)
    routes = np.asarray(router.predict(router_features(h, p)))
    if routes.shape != (len(h),):
        raise ValueError("Router must return one class label per window")
    missing = set(np.unique(routes)) - set(experts)
    if missing:
        raise ValueError(f"Missing RCNN experts for labels: {sorted(missing)}")
    prediction = np.empty((len(h), 24), dtype=float)
    for cls in np.unique(routes):
        ids = np.flatnonzero(routes == cls)
        expert = experts[cls]
        scaler = expert.scaler
        normalized_h = scaler.transform(h[ids].transpose(0, 2, 1)).transpose(0, 2, 1)
        normalized_p = (p[ids] - scaler.p_min) / scaler.p_range
        ref = next(expert.model.parameters())
        ht = torch.as_tensor(normalized_h.copy(), dtype=ref.dtype, device=ref.device)
        pt = torch.as_tensor(normalized_p, dtype=ref.dtype, device=ref.device)
        training = expert.model.training
        try:
            expert.model.eval()
            with torch.no_grad():
                out = expert.model(ht, pt, 12, 24).detach().cpu().numpy()
        finally:
            expert.model.train(training)
        if out.shape != (len(ids), 24) or not np.isfinite(out).all():
            raise ValueError("Expert must produce finite (batch, 24) discharge")
        prediction[ids] = scaler.inverse_q(out)
    return prediction, routes
