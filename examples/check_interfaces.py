"""Artificial-input interface example; no study data, fitting or skill claims."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from flood_core import Autoencoder, RCNN
from flood_core.evaluation import mean_window_metrics
from flood_core.preprocessing import QPScaler
from flood_core.routing import router_features
from flood_core.selection import random_candidates
from flood_core.windows import training_windows


def main():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    artificial = rng.uniform(0.0, 1.0, (3, 48, 2))
    scaler = QPScaler.fit(artificial)
    values = torch.tensor(scaler.transform(artificial), dtype=torch.float32)
    ae, rcnn = Autoencoder().eval(), RCNN(256, 24).eval()
    history, rain, target = training_windows(values)
    with torch.no_grad():
        reconstructed, codes = ae(values.transpose(1, 2))
        prediction = rcnn(history, rain, 12, 24)
    features = router_features(history.numpy(), rain.numpy())
    selected, ids = random_candidates(rng.random((300, 2, 48)))
    print("Artificial inputs only; the randomly initialized models are not fitted.")
    print("AE reconstruction / latent:", tuple(reconstructed.shape), tuple(codes.shape))
    print("RCNN / MLP features:", tuple(prediction.shape), features.shape)
    print("Selected candidate shape / IDs:", selected.shape, ids.shape)
    # A constructed identity prediction verifies the scoring API only.
    physical_target = scaler.inverse_q(target.numpy())
    print("Identity-prediction check:", mean_window_metrics(physical_target, physical_target))


if __name__ == "__main__":
    main()
