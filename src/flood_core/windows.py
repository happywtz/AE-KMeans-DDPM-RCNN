"""In-memory tensor interfaces; no file loading or fitted preprocessing."""

import torch


def forecast_inputs(history_qp: torch.Tensor, future_p: torch.Tensor):
    """Return RCNN inputs from 12 h of Q/P and 24 h of given rainfall."""
    if history_qp.ndim != 3 or history_qp.shape[1:] != (12, 2):
        raise ValueError("history_qp must have shape (batch, 12, 2), channels Q/P")
    if future_p.shape != (history_qp.shape[0], 24):
        raise ValueError("future_p must have shape (batch, 24)")
    return history_qp.transpose(1, 2).contiguous(), future_p.unsqueeze(1)


def training_windows(training_events: torch.Tensor):
    """Build 13 paired 12-to-24 h windows from each 48 h training event."""
    if training_events.ndim != 3 or training_events.shape[1:] != (48, 2):
        raise ValueError("training_events must have shape (events, 48, 2), channels Q/P")
    windows = training_events.transpose(1, 2).unfold(2, 36, 1)
    windows = windows.permute(0, 2, 1, 3).contiguous().view(-1, 2, 36)
    history = windows[:, :, :12].contiguous()
    future_p = windows[:, 1:2, 12:].contiguous()
    target_q = windows[:, 0, 12:].contiguous()
    return history, future_p, target_q
