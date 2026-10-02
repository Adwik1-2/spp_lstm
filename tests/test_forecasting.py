import unittest

import numpy as np
import pandas as pd

from forecasting import (
    apply_log_bias,
    build_sequences,
    data_fingerprint,
    display_currency,
    engineer_features,
    format_volume,
    normalize_ticker,
    per_step_log_error,
)


class ForecastingTests(unittest.TestCase):
    def test_sequence_target_is_aligned_with_last_window_row(self):
        features = np.arange(12, dtype=np.float32).reshape(6, 2)
        targets = np.arange(100, 106, dtype=np.float32)

        windows, aligned = build_sequences(features, targets, time_step=3)

        np.testing.assert_array_equal(windows[0], features[:3])
        np.testing.assert_array_equal(windows[-1], features[-3:])
        np.testing.assert_array_equal(aligned, [102, 103, 104, 105])

    def test_ticker_normalization_and_validation(self):
        self.assertEqual(normalize_ticker(" reliance.ns "), "RELIANCE.NS")
        self.assertEqual(normalize_ticker("^gspc"), "^GSPC")
        with self.assertRaises(ValueError):
            normalize_ticker("AAPL; DROP TABLE")

    def test_feature_engineering_is_finite_and_targets_next_return(self):
        index = pd.date_range("2025-01-01", periods=20, freq="B")
        frame = pd.DataFrame(
            {"Close": np.arange(100.0, 120.0), "Volume": np.arange(1_000, 1_020)},
            index=index,
        )
        features, clean = engineer_features(frame)

        expected = features["ret1"].shift(-1).loc[clean.index]
        np.testing.assert_allclose(clean["target_ret_next"], expected)
        self.assertTrue(np.isfinite(clean.to_numpy()).all())

    def test_data_fingerprint_tracks_training_changes(self):
        columns = ["ret1", "vol_change", "sma_10_dist", "volatility_10", "target_ret_next"]
        frame = pd.DataFrame(np.ones((3, 5)), columns=columns)
        changed = frame.copy()
        changed.loc[2, "ret1"] = 2
        self.assertNotEqual(data_fingerprint(frame), data_fingerprint(changed))

    def test_display_currency(self):
        self.assertEqual(display_currency("TCS.NS"), "₹")
        self.assertEqual(display_currency("AAPL"), "$")

    def test_volume_formatting(self):
        self.assertEqual(format_volume(1_250_000), "1.25M")
        self.assertEqual(format_volume(42_000), "42.00K")

    def test_feedback_uses_compounded_log_error(self):
        bias = per_step_log_error(predicted_price=100, actual_price=121, steps=2)
        self.assertAlmostEqual(bias, np.log(1.21) / 2)
        self.assertAlmostEqual(apply_log_bias(0.0, bias), 0.10, places=6)

    def test_corrected_return_is_safely_bounded(self):
        self.assertAlmostEqual(apply_log_bias(10.0, 0.0), 0.20)
        self.assertAlmostEqual(apply_log_bias(-0.99, 0.0), -0.20)


if __name__ == "__main__":
    unittest.main()
