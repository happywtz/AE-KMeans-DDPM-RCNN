"""Loss functions separated from the original training scripts."""

import torch.nn.functional as F


def ae_loss(reconstruction, target):
    """Channel-weighted reconstruction MSE; channel order is Q, P."""
    return (
        0.05 * F.mse_loss(reconstruction[:, 0], target[:, 0])
        + 0.95 * F.mse_loss(reconstruction[:, 1], target[:, 1])
    )


def rcnn_loss(prediction_q, target_q):
    return F.smooth_l1_loss(prediction_q, target_q)
