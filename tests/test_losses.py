import unittest
from types import SimpleNamespace

import torch

from model.losses import (
    concordance_correlation_loss,
    get_motion_loss,
    get_motion_RMSE,
    normalized_squared_velocity_loss,
    velocity_gain_loss,
)
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


class VelocityGainLossTest(unittest.TestCase):
    def setUp(self):
        self.target = torch.tensor(
            [[[0.2, 0.4, -0.3],
              [0.3, 0.2, -0.1],
              [0.4, -0.2, 0.2],
              [0.5, -0.4, 0.4]]],
            dtype=torch.float64,
        )

    def test_matching_gain_has_zero_loss(self):
        loss = velocity_gain_loss(self.target, self.target)
        self.assertLess(loss.item(), 1e-12)

    def test_half_gain_has_quarter_loss(self):
        loss = velocity_gain_loss(0.5 * self.target, self.target)
        self.assertAlmostEqual(loss.item(), 0.25, places=7)

    def test_near_zero_target_axes_are_ignored(self):
        target = torch.zeros((1, 4, 3), dtype=torch.float64)
        target[..., 0] = 0.2
        target[..., 1] = 0.099
        prediction = target.clone()
        prediction[..., 1:] = 1000.0

        loss = velocity_gain_loss(prediction, target, min_rms=0.1)

        self.assertLess(loss.item(), 1e-12)

    def test_mask_is_applied_per_window_and_axis(self):
        target = torch.zeros((2, 4, 3), dtype=torch.float64)
        target[0, :, 0] = 0.2
        target[1, :, 1] = 0.05
        prediction = torch.zeros_like(target)
        prediction[0, :, 0] = 0.1
        prediction[1, :, 1] = 100.0

        loss = velocity_gain_loss(prediction, target, min_rms=0.1)

        self.assertAlmostEqual(loss.item(), 0.25, places=7)

    def test_each_valid_window_has_equal_weight(self):
        target = torch.zeros((2, 4, 3), dtype=torch.float64)
        target[0, :, 0] = 0.2
        target[1, :, :] = 0.2
        prediction = target.clone()
        prediction[0, :, 0] = 0.0

        loss = velocity_gain_loss(prediction, target, min_rms=0.1)

        # Window 0 has loss 1 from its only valid axis; window 1 has loss 0
        # across three valid axes. Each window receives one equal vote.
        self.assertAlmostEqual(loss.item(), 0.5, places=7)

    def test_constant_velocity_is_not_centered_out(self):
        target = torch.full((1, 4, 3), 0.2, dtype=torch.float64)
        prediction = torch.full_like(target, 0.1)

        loss = velocity_gain_loss(prediction, target)

        self.assertAlmostEqual(loss.item(), 0.25, places=7)

    def test_loss_has_finite_prediction_gradients(self):
        prediction = (0.5 * self.target).clone().requires_grad_()
        loss = velocity_gain_loss(prediction, self.target)
        loss.backward()

        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertLess((prediction.grad * self.target).sum().item(), 0.0)

    def test_all_masked_loss_remains_differentiable(self):
        target = torch.zeros((2, 4, 3), dtype=torch.float64)
        prediction = torch.ones_like(target, requires_grad=True)
        loss = velocity_gain_loss(prediction, target)
        loss.backward()

        self.assertEqual(loss.item(), 0.0)
        self.assertTrue(torch.equal(prediction.grad, torch.zeros_like(prediction)))

    def test_gain_is_added_as_a_weighted_auxiliary_loss(self):
        class Config(dict):
            __getattr__ = dict.__getitem__

        prediction = 0.5 * self.target
        config = Config(
            loss="normalized_squared_velocity",
            s0=0.25,
            ccc_weight=0.25,
            ccc_tau=0.1,
            gain_weight=0.05,
            gain_min_rms=0.1,
            propcov=False,
            weight=1.0,
        )
        result = get_motion_loss({"net_vel": prediction}, self.target, config)
        normalized, _ = normalized_squared_velocity_loss(
            prediction, self.target, s0=0.25
        )
        expected = (
            normalized
            + 0.25 * concordance_correlation_loss(prediction, self.target, tau=0.1)
            + 0.05 * velocity_gain_loss(prediction, self.target, min_rms=0.1)
        )

        self.assertTrue(torch.allclose(result["loss"], expected))


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
