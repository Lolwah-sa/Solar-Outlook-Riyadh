# Unit tests for the Riyadh forecasting project. Run all of them from the project folder with:
#     python -m unittest discover -s tests
# The first group checks the method itself (no future information, correct target, correct
# formulas). The second group recomputes every published number from the saved forecasts, so the
# dashboard and the report cannot show a value that the code did not actually produce.
"""Tests for data preparation, leakage safety, metrics and the consistency of the saved results."""
# json reads the saved audit; tempfile creates a throw-away CSV for the leakage test.
import json
import tempfile
import unittest
from pathlib import Path
# NumPy/pandas build expected values; scikit-learn recomputes metrics independently.
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, mean_absolute_error, r2_score

# The functions under test.
from src.pipeline import (CAPACITY_KW, RAW, clear_sky_persistence, daylight, drop_events, prepare,
                          score, solar_elevation)
from src.diagnostics import day_block_bootstrap

# Folder with the saved experiment results.
OUT = Path(__file__).resolve().parents[1] / 'artifacts'


class MethodTests(unittest.TestCase):
    """Checks of the forecasting method, independent of the saved results."""

    def test_sun_position_is_physically_correct_for_riyadh(self):
        # Riyadh is close to the Tropic of Cancer: near the June solstice the noon sun is almost
        # overhead (about 88 degrees), while at midnight it is far below the horizon.
        times = pd.DatetimeIndex(['2023-06-21 12:00', '2023-06-21 00:00', '2023-12-21 12:00'])
        noon_june, midnight, noon_december = solar_elevation(times)
        self.assertGreater(noon_june, 85)
        self.assertLess(midnight, -30)
        self.assertTrue(40 < noon_december < 45)

    def test_target_is_exactly_the_next_hour(self):
        # The value to predict at time t must be the PV power at t + 1 hour.
        d, _, _ = prepare()
        nxt = d.index + pd.Timedelta(hours=1)
        both = nxt.isin(d.index)
        np.testing.assert_allclose(d.target.to_numpy()[both], d.loc[nxt[both], 'power'].to_numpy())

    def test_features_never_use_future_rows(self):
        # Corrupt all data AFTER a cut-off time; features BEFORE that time must not change.
        raw = pd.read_csv(RAW).head(24 * 20)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'sample.csv'
            raw.to_csv(path, index=False)
            first, features, _ = prepare(path)
            changed = raw.copy()
            cut = len(changed) // 2
            numeric = ['ghi', 'ghi_clearsky', 'dni', 'dhi', 'temp_c', 'rel_humidity', 'wind_speed', 'pressure_kpa', 'actual_kw']
            changed.loc[cut:, numeric] = 999.0
            changed.to_csv(path, index=False)
            second, _, _ = prepare(path)
        cut_time = pd.Timestamp(raw.timestamp.iloc[cut])
        # Features at times before the cut-off are identical (the target of the last row may differ).
        early = first.index < cut_time
        # check_freq=False: only the index's frequency label may differ (dropped rows elsewhere), never the values.
        pd.testing.assert_frame_equal(first.loc[early, features], second.loc[second.index < cut_time, features], check_freq=False)

    def test_zero_output_under_strong_sun_is_treated_as_missing(self):
        # Gaps in the satellite record appear as 0 kW at midday; they must not become training targets.
        d, _, audit = prepare()
        self.assertGreater(audit['pv_zero_under_sun_set_missing'], 0)
        gap = pd.Timestamp(audit['pv_zero_under_sun_examples'][0])
        self.assertNotIn(gap, d.index)                                   # not used as an example
        self.assertNotIn(gap - pd.Timedelta(hours=1), d.index)           # not used as a target

    def test_accuracy_and_tolerance_use_capacity(self):
        # Errors of 0.5, 0.4 and 0.6 kW with a 5 kW capacity: two of three are within ±10% (0.5 kW).
        result = score([0, 1, 4], [0.5, 1.4, 3.4])
        self.assertAlmostEqual(result['within_10pct'], 200 / 3)
        rmse = np.sqrt((0.25 + 0.16 + 0.36) / 3)
        self.assertAlmostEqual(result['accuracy_percent'], 100 * (1 - rmse / CAPACITY_KW), places=6)

    def test_clear_sky_persistence_is_zero_after_sunset(self):
        # If the sun is below the horizon at the target hour, the smart baseline must forecast 0.
        x = pd.DataFrame({'power': [2.0], 'cs_now': [.3], 'cs_next': [0.0]})
        self.assertEqual(clear_sky_persistence(x)[0], 0.0)

    def test_bootstrap_is_reproducible_and_zero_for_identical_forecasts(self):
        index = pd.date_range('2023-01-01', periods=24 * 5, freq='h')
        frame = pd.DataFrame({'target': np.arange(len(index)) + 1.0, 'A': np.arange(len(index)) + 2.0,
                              'B': np.arange(len(index)) + 2.0}, index=index)
        first = day_block_bootstrap(frame, 'A', 'B', repetitions=100)
        self.assertEqual(first, day_block_bootstrap(frame, 'A', 'B', repetitions=100))
        self.assertEqual(first['low_percent'], 0)


class SavedResultTests(unittest.TestCase):
    """Recompute the published numbers from artifacts/ (fast, no retraining)."""

    @classmethod
    def setUpClass(cls):
        cls.audit = json.loads((OUT / 'audit.json').read_text())
        cls.pred = pd.read_csv(OUT / 'predictions.csv', index_col='timestamp', parse_dates=True)
        cls.metrics = pd.read_csv(OUT / 'metrics.csv')

    def test_test_year_is_2023_and_never_used_for_training(self):
        self.assertTrue((self.pred.index.year == 2023).all())
        validation = pd.read_csv(OUT / 'validation.csv')
        self.assertTrue((validation.validation_year < 2023).all())

    def test_published_metrics_match_saved_forecasts(self):
        day = self.pred.elev_next > 0
        for _, row in self.metrics.iterrows():
            mask = day if row.period == 'Daylight' else np.ones(len(self.pred), bool)
            y, p = self.pred.target[mask], self.pred[row.model][mask]
            # CSV rounding to 4 decimals changes errors by far less than 0.001 kW.
            self.assertAlmostEqual(row.MAE_kW, mean_absolute_error(y, p), delta=1e-3)
            self.assertAlmostEqual(row.R2, r2_score(y, p), delta=1e-3)
            self.assertAlmostEqual(row.within_10pct, 100 * np.mean(np.abs(p - y) <= .5), delta=.1)

    def test_warning_counts_match_the_frozen_drop_rule(self):
        winner = self.audit['selected_model']
        # The saved table contains power, cs_now, cs_next and elev_next, which the rule needs.
        x = self.pred
        truth = drop_events(x, x.target)
        alert = drop_events(x, x[winner])
        day = daylight(x)
        expected = confusion_matrix(truth[day], alert[day], labels=[False, True]).ravel().tolist()
        saved = next(w for w in self.audit['warnings'] if w['model'] == winner)
        self.assertEqual(expected, saved['confusion_matrix_TN_FP_FN_TP'])


if __name__ == '__main__':
    unittest.main()
