"""CPU interface checks with artificial tensors only."""

from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from torch import nn

from flood_core import (
    Autoencoder, RCNN, Unet1D, GaussianDiffusion1D,
    fit_kmeans, euclidean_distance_matrix, sample_silhouettes,
)
from flood_core.windows import forecast_inputs, training_windows
from flood_core.objectives import ae_loss, rcnn_loss


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(7)
        random.seed(7)

    def test_autoencoder_shapes(self):
        model = Autoencoder().cpu().eval()
        values = torch.rand(2, 2, 48)
        with torch.no_grad():
            reconstruction, latent = model(values)
        self.assertEqual(tuple(reconstruction.shape), (2, 2, 48))
        self.assertEqual(tuple(latent.shape), (2, 5, 1))
        self.assertTrue(torch.isfinite(reconstruction).all().item())

    def test_rcnn_shapes_and_original_output(self):
        model = RCNN(256, 24).cpu().eval()
        history, future_p = forecast_inputs(torch.rand(2, 12, 2), torch.rand(2, 24))
        with torch.no_grad():
            prediction = model(history, future_p, 12, 24)
        self.assertEqual(tuple(prediction.shape), (2, 24))
        self.assertTrue(torch.isfinite(prediction).all().item())
        self.assertIsInstance(model.yubao[-1], nn.LeakyReLU)

    def test_training_window_alignment(self):
        events = torch.arange(2 * 48 * 2, dtype=torch.float32).reshape(2, 48, 2)
        history, future_p, target_q = training_windows(events)
        self.assertEqual(tuple(history.shape), (26, 2, 12))
        self.assertEqual(tuple(future_p.shape), (26, 1, 24))
        self.assertEqual(tuple(target_q.shape), (26, 24))
        torch.testing.assert_close(history[0], events[0, :12].T)
        torch.testing.assert_close(target_q[0], events[0, 12:36, 0])
        torch.testing.assert_close(target_q[12], events[0, 24:48, 0])
        torch.testing.assert_close(history[13], events[1, :12].T)
        torch.testing.assert_close(future_p[13, 0], events[1, 12:36, 1])

    def test_forecast_inputs_reject_wrong_shapes(self):
        with self.assertRaises(ValueError):
            forecast_inputs(torch.rand(2, 24, 2), torch.rand(2, 24))
        with self.assertRaises(ValueError):
            forecast_inputs(torch.rand(2, 12, 2), torch.rand(2, 12))
        with self.assertRaises(ValueError):
            training_windows(torch.rand(2, 36, 2))

    def test_losses(self):
        target = torch.zeros(2, 2, 48)
        reconstructed = torch.ones_like(target)
        reconstructed[:, 1] = 2.0
        self.assertAlmostEqual(ae_loss(reconstructed, target).item(), 3.85, places=5)
        self.assertAlmostEqual(rcnn_loss(torch.ones(2, 24), torch.zeros(2, 24)).item(), 0.5)

    def test_kmeans_and_silhouette(self):
        rng = np.random.default_rng(7)
        codes = np.concatenate([
            rng.normal(-2, 0.05, (6, 5)), rng.normal(2, 0.05, (6, 5))
        ])
        labels, centers, inertia, iterations, seed = fit_kmeans(codes, 2, 42, 4, 50)
        self.assertEqual(centers.shape, (2, 5))
        self.assertEqual(len(np.unique(labels)), 2)
        self.assertEqual(len(np.unique(labels[:6])), 1)
        self.assertEqual(len(np.unique(labels[6:])), 1)
        self.assertNotEqual(labels[0], labels[-1])
        self.assertGreaterEqual(inertia, 0)
        self.assertGreater(iterations, 0)
        scores = sample_silhouettes(euclidean_distance_matrix(codes), labels)
        self.assertGreater(scores.mean(), 0.9)
        repeated = fit_kmeans(codes, 2, 42, 4, 50)
        np.testing.assert_array_equal(labels, repeated[0])
        np.testing.assert_allclose(centers, repeated[1])

    def test_ddpm_loss_and_sampling(self):
        network = Unet1D(dim=8, dim_mults=(1, 2), channels=2).cpu().eval()
        diffusion = GaussianDiffusion1D(
            network, seq_length=48, timesteps=4, sampling_timesteps=4,
            objective="pred_noise",
        ).cpu().eval()
        with torch.no_grad():
            loss = diffusion(torch.rand(2, 2, 48))
            generated = diffusion.sample(batch_size=2)
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss).item())
        self.assertEqual(tuple(generated.shape), (2, 2, 48))
        self.assertTrue(torch.isfinite(generated).all().item())


if __name__ == "__main__":
    unittest.main()
