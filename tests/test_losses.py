import unittest
from types import SimpleNamespace

import torch

from model.losses import concordance_correlation_loss, get_motion_RMSE
from utils import DatasetLossTracker


class ConcordanceCorrelationLossTest(unittest.TestCase):
    def setUp(self):
        self.target = torch.tensor(
            [[[0.1, -0.2, 0.3],
              [0.4, 0.0, 0.1],
              [0.8, 0.5, -0.2],
              [0.5, 0.2, 0.4],
              [0.2, -0.1, 0.0]]],
            dtype=torch.float64,
        )

    def test_matching_signal_has_near_zero_loss(self):
        loss = concordance_correlation_loss(self.target, self.target)
        self.assertLess(loss.item(), 1e-6)

    def test_flat_signal_is_penalized(self):
        prediction = self.target.mean(dim=-2, keepdim=True).expand_as(self.target)
        loss = concordance_correlation_loss(prediction, self.target)
        target_variance = self.target.var(dim=-2, correction=0).mean(dim=-1)
        expected_activity = target_variance / (target_variance + 0.1 ** 2)
        self.assertTrue(torch.allclose(loss, expected_activity.mean(), atol=1e-8))

    def test_under_and_over_scaled_signals_are_penalized(self):
        exact = concordance_correlation_loss(self.target, self.target)
        under = concordance_correlation_loss(0.3 * self.target, self.target)
        over = concordance_correlation_loss(1.5 * self.target, self.target)
        self.assertGreater(under.item(), exact.item())
        self.assertGreater(over.item(), exact.item())

    def test_constant_axes_are_excluded_from_axis_loss(self):
        varying_x = self.target.clone()
        varying_x[..., 1:] = 2.0
        prediction = varying_x.clone()
        prediction[..., 1:] = -100.0

        loss = concordance_correlation_loss(prediction, varying_x, tau=0.1)

        self.assertLess(loss.item(), 1e-6)

    def test_nearly_stationary_window_has_little_ccc_contribution(self):
        target = 2.0 + 1e-3 * self.target
        prediction = 2.0 - 1e-3 * self.target

        loss = concordance_correlation_loss(prediction, target, tau=0.1)

        self.assertLess(loss.item(), 1e-3)

    def test_loss_has_finite_gradients(self):
        prediction = (0.3 * self.target).clone().requires_grad_()
        loss = concordance_correlation_loss(prediction, self.target)
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())


class MotionEvaluationMetricsTest(unittest.TestCase):
    def test_rmse_does_not_allow_temporal_error_cancellation(self):
        error = torch.tensor(
            [[[0.5, 0.0, 0.0],
              [0.5, 0.0, 0.0],
              [-0.5, 0.0, 0.0],
              [-0.5, 0.0, 0.0]]]
        )
        result = get_motion_RMSE(
            {"net_vel": error},
            torch.zeros_like(error),
            SimpleNamespace(propcov=False),
        )

        self.assertAlmostEqual(result["loss"].item(), 0.5)
        self.assertAlmostEqual(result["dist"].item(), 0.5)

    def test_motion_statistics_accumulate_before_correlation_and_gain(self):
        first_target = torch.tensor([[[1.0, 2.0, 4.0], [2.0, 2.0, 3.0]]])
        second_target = torch.tensor([[[3.0, 2.0, 2.0], [4.0, 2.0, 1.0]]])
        first_prediction = 0.5 * first_target
        second_prediction = 0.5 * second_target

        total = DatasetLossTracker._update_motion_total(
            None, first_prediction, first_target
        )
        total = DatasetLossTracker._update_motion_total(
            total, second_prediction, second_target
        )
        rmse, correlation, gain = DatasetLossTracker._motion_metrics(total)

        expected_rmse = torch.sqrt(
            torch.cat((first_prediction - first_target,
                       second_prediction - second_target), dim=1).double()
            .square().sum(dim=-1).mean()
        )
        self.assertAlmostEqual(rmse, expected_rmse.item())
        self.assertEqual(correlation, [1.0, 0.0, 1.0])
        for axis_gain in gain:
            self.assertAlmostEqual(axis_gain, 0.5, places=7)


if __name__ == "__main__":
    unittest.main()
