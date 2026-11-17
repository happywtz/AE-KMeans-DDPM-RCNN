"""Small training entry points, separate from the archived experiment runners.

These helpers perform explicitly requested optimizer updates. They do not
choose paper checkpoints, select evaluation folds or reproduce the paper tables.
"""

import numpy as np
import torch

from .objectives import ae_loss, rcnn_loss
from .windows import training_windows


def optimizer_steps(model, optimizer, loss_function, updates):
    """Run updates with a zero-argument closure returning a scalar loss."""
    if isinstance(updates, bool) or not isinstance(updates, int) or updates < 1:
        raise ValueError("A positive integer number of updates is required")
    model.train()
    losses = []
    for _ in range(updates):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_function()
        if loss.ndim != 0 or not torch.isfinite(loss).item():
            raise ValueError("Expected a finite scalar training loss")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return losses


def _tensor_for(model, values):
    ref = next(model.parameters())
    x = torch.as_tensor(values, device=ref.device, dtype=ref.dtype)
    if not torch.isfinite(x).all().item():
        raise ValueError("Training inputs must be finite")
    return x


def train_autoencoder(model, optimizer, normalized_training_events, updates):
    """Full-batch AE updates; input shape (n, 2, 48), n >= 2 for BatchNorm."""
    x = _tensor_for(model, normalized_training_events)
    if x.ndim != 3 or x.shape[1:] != (2, 48) or len(x) < 2:
        raise ValueError("AE training requires at least two (2, 48) sequences")
    return optimizer_steps(model, optimizer, lambda: ae_loss(model(x)[0], x), updates)


def train_rcnn(model, optimizer, normalized_training_events, updates):
    """Full-window SmoothL1 updates; training events are (n, 48, 2)."""
    x = _tensor_for(model, normalized_training_events)
    if len(x) < 1:
        raise ValueError("Training events cannot be empty")
    history, rain, target = training_windows(x)
    return optimizer_steps(model, optimizer,
                           lambda: rcnn_loss(model(history, rain, 12, 24), target),
                           updates)


def train_diffusion(diffusion, optimizer, normalized_training_events,
                    updates, batch_size, seed=2024):
    """Uniform minibatch DDPM updates, using (n, 2, 48) training events.

    This helper has no historical gradient-accumulation or EMA wrapper. Specify
    those explicitly in a separate runner if required by a study protocol.
    """
    x = _tensor_for(diffusion, normalized_training_events)
    if x.ndim != 3 or x.shape[1:] != (2, 48) or len(x) < 1:
        raise ValueError("Expected training events of shape (n, 2, 48)")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    rng = np.random.default_rng(seed)

    def loss():
        ids = torch.as_tensor(rng.integers(0, len(x), size=batch_size), device=x.device)
        return diffusion(x[ids])

    return optimizer_steps(diffusion, optimizer, loss, updates)
