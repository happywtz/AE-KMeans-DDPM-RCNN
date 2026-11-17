"""Artificial-input checks for the expanded component interfaces."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from torch import nn

from flood_core.evaluation import mean_window_metrics, window_metrics
from flood_core.inference import ForecastExpert, forecast
from flood_core.preprocessing import QPScaler
from flood_core.routing import balance_for_mlp, make_router, router_features
from flood_core.selection import random_candidates
from flood_core.training import optimizer_steps


class ConstantExpert(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(float(value)))

    def forward(self, h, p, pre, num):
        return self.value.expand(len(h), 24)


class ToyRouter:
    def predict(self, x):
        return np.where(x[:, 0] < 1, 3, 7)


class ExtendedTests(unittest.TestCase):
    def test_scaler_round_trip_and_no_clipping(self):
        x = np.arange(3*48*2, dtype=float).reshape(3, 48, 2)
        scaler = QPScaler.fit(x)
        np.testing.assert_allclose(scaler.inverse(scaler.transform(x)), x)
        self.assertGreater(scaler.transform(x+1000).max(), 1)
        with self.assertRaises(ValueError):
            QPScaler.fit(np.ones((3, 48, 2)))

    def test_router_feature_order(self):
        h = np.arange(2*12).reshape(1, 2, 12)
        p = np.arange(24, 48).reshape(1, 1, 24)
        np.testing.assert_array_equal(router_features(h, p), np.arange(48)[None])
        with self.assertRaises(ValueError):
            router_features(h, np.zeros((1, 24)))

    def test_original_mlp_configuration(self):
        model = make_router(42).named_steps["model"]
        self.assertEqual(model.hidden_layer_sizes, (16,))
        self.assertEqual(model.solver, "lbfgs")
        self.assertEqual(model.alpha, .01)
        self.assertEqual(model.max_iter, 2000)

    def test_training_only_balance(self):
        x = np.arange(4*48).reshape(4, 48)
        y = np.array([3, 3, 3, 7])
        a, b = balance_for_mlp(x, y, np.random.default_rng(42))
        self.assertEqual(np.unique(b, return_counts=True)[1].tolist(), [3, 3])
        self.assertTrue(all(any(np.array_equal(row, source) for source in x) for row in a))

    def test_candidate_selection(self):
        pool = np.arange(300*2*48).reshape(300, 2, 48)
        values, ids = random_candidates(pool)
        self.assertEqual(len(np.unique(ids)), 100)
        np.testing.assert_array_equal(values, pool[ids])
        np.testing.assert_array_equal(ids, random_candidates(pool)[1])
        with self.assertRaises(ValueError):
            random_candidates(pool, 301)

    def test_metric_values(self):
        obs = np.arange(24, dtype=float)[None]
        scores = mean_window_metrics(obs, obs)
        self.assertAlmostEqual(scores["NSE"], 1)
        self.assertAlmostEqual(scores["R"], 1)
        self.assertEqual(scores["MAE"], 0)
        err = mean_window_metrics(obs, obs+2)
        self.assertAlmostEqual(err["RMSE"], 2)
        self.assertAlmostEqual(err["MAE"], 2)
        self.assertAlmostEqual(err["NSE"], 1-24*4/np.sum((obs-obs.mean())**2))
        constant = window_metrics(np.ones((1, 24)), np.zeros((1, 24)))
        self.assertTrue(np.isnan(constant["NSE"][0]))
        self.assertTrue(np.isnan(constant["R"][0]))

    def test_routed_forecast_order_and_signed_output(self):
        h, p = np.zeros((3, 2, 12)), np.zeros((3, 1, 24))
        h[1, 0, 0] = 2
        scaler = QPScaler(0, 10, 0, 2)
        first, second = ConstantExpert(-1), ConstantExpert(2)
        experts = {3: ForecastExpert(first, scaler), 7: ForecastExpert(second, scaler)}
        q, routes = forecast(ToyRouter(), experts, h, p)
        np.testing.assert_array_equal(routes, [3, 7, 3])
        np.testing.assert_allclose(q[:, 0], [-10, 20, -10])
        self.assertTrue(first.training)
        with self.assertRaises(ValueError):
            forecast(ToyRouter(), {3: experts[3]}, h, p)

    def test_invalid_training_budget_does_not_update(self):
        model = nn.Linear(1, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        before = model.weight.detach().clone()
        with self.assertRaises(ValueError):
            optimizer_steps(model, optimizer, lambda: model.weight.sum(), 0)
        torch.testing.assert_close(model.weight, before)


if __name__ == "__main__":
    unittest.main()
