# Solar Outlook Riyadh — the main experiment.
# Run it from the project folder with:   python -m src.pipeline
# It reads the hourly Riyadh dataset, builds features that are known at forecast time, compares
# baselines and machine-learning models with year-by-year chronological validation, scores the
# selected model ONCE on the untouched 2023 test year, and saves small result files for the
# dashboard and the report.
"""Hourly 1-hour-ahead PV power forecasting for a 5 kWp system in Riyadh (2019-2023)."""
# Path builds file locations relative to this file, so the code runs on any computer.
from pathlib import Path
# hashlib fingerprints the input CSV so others can check they used exactly the same data.
import hashlib
# json writes the audit and summary files in a readable text format.
import json
# pickle is only used to MEASURE the size of each fitted model (memory evidence for the report).
import pickle
# time and tracemalloc measure run time and peak Python memory of every stage.
import time
import tracemalloc
# joblib saves the fitted selected model locally (it is not needed by the dashboard).
import joblib
# NumPy for fast vectorized maths; pandas for the timestamped table.
import numpy as np
import pandas as pd
# clone makes a fresh, unfitted copy of a model for every validation fold.
from sklearn.base import clone
# Candidate learners: linear (Ridge), bagged trees (random forest, extra trees) and boosted trees.
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
# A pipeline keeps scaling and the linear model together, so scaling is learned on training rows only.
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
# Standard regression and classification metrics.
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score, confusion_matrix,
                             precision_recall_fscore_support, average_precision_score)

# ---------------------------------------------------------------------------------------------
# 0. FIXED SETTINGS (decided before any model was scored)
# ---------------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/raw/saudi_solar_riyadh_2019_2023.csv'
OUT = ROOT / 'artifacts'
# Riyadh coordinates used by the download script, and the local time zone offset (UTC+3, no DST).
LAT, LON, TZ = 24.7136, 46.6753, 3
# Installed capacity of the simulated PV system (kWp), set in the PVGIS request.
CAPACITY_KW = 5.0
# Accuracy tolerance bands, as fractions of capacity. ±10% is the headline band.
TOLERANCES = [.05, .10, .20]
# Development years (used for training and validation) and the untouched test year.
DEV_YEARS = [2019, 2020, 2021, 2022]
TEST_YEAR = 2023
# A sudden weather-driven drop: next-hour power at least 20% below the clear-sky expectation
# AND at least 5% of capacity (0.25 kW) below it. Both thresholds are fixed in advance.
DROP_RELATIVE, DROP_ABSOLUTE_KW = .20, .05 * CAPACITY_KW


# ---------------------------------------------------------------------------------------------
# 1. SOLAR GEOMETRY (astronomy, known in advance for any future hour)
# ---------------------------------------------------------------------------------------------
def solar_elevation(times, lat=LAT, lon=LON, tz=TZ):
    """Sun elevation angle in degrees for local timestamps (NOAA general solar position formulas).

    The position of the sun is pure astronomy, so it is legitimately known for the NEXT hour
    at forecast time. Everything is vectorized: one NumPy expression for all 43,824 hours.
    """
    # Day of year (1..365) and decimal hour of every timestamp, as NumPy arrays.
    doy = times.dayofyear.to_numpy()
    hour = times.hour.to_numpy() + times.minute.to_numpy() / 60
    # Fractional year in radians.
    g = 2 * np.pi / 365 * (doy - 1 + (hour - 12) / 24)
    # Equation of time (minutes) and solar declination (radians).
    eqtime = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                       - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
            + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    # True solar time (minutes) -> hour angle (radians).
    true_solar = hour * 60 + eqtime + 4 * lon - 60 * tz
    hour_angle = np.radians(true_solar / 4 - 180)
    # Latitude in radians; the zenith-angle formula below gives the cosine of the sun's angle from vertical.
    phi = np.radians(lat)
    cos_zenith = np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.cos(hour_angle)
    # Elevation = 90° − zenith; clip guards against tiny rounding errors outside [-1, 1].
    return np.degrees(np.arcsin(np.clip(cos_zenith, -1, 1)))


def clear_sky_factor(elevation_deg):
    """A simple clear-sky shape: sine of the sun elevation, zero when the sun is below the horizon."""
    return np.maximum(np.sin(np.radians(elevation_deg)), 0)


# ---------------------------------------------------------------------------------------------
# 2. LOAD, AUDIT AND BUILD FEATURES
# ---------------------------------------------------------------------------------------------
def prepare(path=RAW):
    """Load the CSV, check its quality, and build a float32 feature table with the 1-hour-ahead target."""
    # Read the CSV, use the timestamp as the index and make sure rows are in time order.
    raw = pd.read_csv(path, parse_dates=['timestamp']).set_index('timestamp').sort_index()
    # Keep only local calendar years 2019-2023 (the UTC download spills 3 night hours into 2024).
    raw = raw.loc[str(DEV_YEARS[0]):str(TEST_YEAR)]
    # The numeric columns that are checked and used.
    numeric = ['ghi', 'ghi_clearsky', 'dni', 'dhi', 'temp_c', 'rel_humidity', 'wind_speed', 'pressure_kpa', 'actual_kw']
    # The audit records what was checked so the data processing can be described and verified.
    # The complete list of hours between the first and last timestamp, used to detect gaps.
    full_grid = pd.date_range(raw.index.min(), raw.index.max(), freq='h')
    audit = {
        # File name and SHA-256 fingerprint of the exact input file.
        'source_file': path.name,
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'rows': len(raw), 'start': str(raw.index.min()), 'end': str(raw.index.max()),
        # Duplicate and missing hours would break the meaning of 'one hour earlier'.
        'duplicate_timestamps': int(raw.index.duplicated().sum()),
        'missing_hours': int(len(full_grid.difference(raw.index))),
        # Missing, negative and out-of-range values per column.
        'missing_values': {c: int(raw[c].isna().sum()) for c in numeric},
        'negative_values': {c: int((raw[c] < 0).sum()) for c in numeric if c not in ('temp_c',)},
        'power_above_capacity': int((raw.actual_kw > CAPACITY_KW).sum()),
        # Measured irradiance should not exceed the clear-sky value by more than a small margin.
        'ghi_above_clearsky_plus_5pct': int((raw.ghi > 1.05 * raw.ghi_clearsky + 1).sum()),
        'pv_database': sorted(raw.pvgis_db.unique().tolist()),
        'capacity_kWp': CAPACITY_KW,
    }
    if audit['duplicate_timestamps'] or audit['missing_hours']:
        raise ValueError('The hourly grid must be complete and unique; review the download.')
    # Physical plausibility check: a working 5 kWp system cannot produce exactly 0 kW while the
    # sun gives more than 200 W/m². These hours are gaps in the PVGIS satellite record that were
    # stored as zero. They are marked as MISSING (not filled in), so no example uses them as a
    # feature or as a target, and they cannot appear as fake "drops".
    gaps = (raw.actual_kw == 0) & (raw.ghi > 200)
    # Record how many hours were affected and a few examples, for the report.
    audit['pv_zero_under_sun_set_missing'] = int(gaps.sum())
    audit['pv_zero_under_sun_examples'] = [str(t) for t in raw.index[gaps][:5]]
    # Copy before editing, then mark the gaps as missing (NaN).
    raw = raw.copy()
    raw.loc[gaps, 'actual_kw'] = np.nan
    # Reindex to the full hourly grid so that shift(1) always means exactly one hour earlier.
    d = raw[numeric].reindex(full_grid)
    d.index.name = 'timestamp'
    # `f` will hold one row per hour with every input feature.
    f = pd.DataFrame(index=d.index)
    # --- Current and past PV output (all known at forecast time t) ---
    f['power'] = d.actual_kw
    # Power 1, 2 and 3 hours ago (recent trend).
    for lag in [1, 2, 3]:
        f[f'power_lag_{lag}'] = d.actual_kw.shift(lag)
    # Change in power over the last hour (ramp up or down).
    f['power_change'] = d.actual_kw.diff()
    f['power_lag_24'] = d.actual_kw.shift(24)          # same hour yesterday
    # --- Current weather and irradiance (measured/estimated at time t) ---
    # Copy the current weather columns unchanged.
    for c in ['ghi', 'dni', 'dhi', 'ghi_clearsky', 'temp_c', 'rel_humidity', 'wind_speed', 'pressure_kpa']:
        f[c] = d[c]
    # Clear-sky index: how much of the clear-sky irradiance reaches the ground (low = dust/cloud).
    f['kt'] = (d.ghi / d.ghi_clearsky.where(d.ghi_clearsky > 20)).clip(0, 1.2).fillna(0)
    # Clear-sky index one hour ago, and the latest change in irradiance.
    f['kt_lag_1'] = f.kt.shift(1)
    f['ghi_change'] = d.ghi.diff()
    # --- Astronomy: sun elevation now and at the TARGET hour (deterministic, known in advance) ---
    # Sun elevation now and one hour later; vectorized over all hours at once.
    elev_now = solar_elevation(d.index)
    elev_next = solar_elevation(d.index + pd.Timedelta(hours=1))
    f['elev_now'] = elev_now
    f['elev_next'] = elev_next
    # Clear-sky shapes (0 at night, up to 1 with the sun overhead).
    f['cs_now'] = clear_sky_factor(elev_now)
    f['cs_next'] = clear_sky_factor(elev_next)
    # Power per unit of clear-sky shape: a "performance index" of the system right now.
    f['power_per_cs'] = (d.actual_kw / f.cs_now.where(f.cs_now > .05)).clip(0, 2 * CAPACITY_KW).fillna(0)
    # --- Calendar encoded as circles so 23:00 is close to 00:00 and 31 Dec to 1 Jan ---
    # Hour of day and day of year for the circular calendar features.
    hour = d.index.hour
    doy = d.index.dayofyear
    f['hour_sin'], f['hour_cos'] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    f['doy_sin'], f['doy_cos'] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    # Remember the feature names before the target column is added.
    features = list(f.columns)
    # --- Target: PV power exactly one hour after the issue time ---
    f['target'] = d.actual_kw.shift(-1)
    # Drop the few rows at the very start/end that lack a full history or a target.
    clean = f.dropna().astype('float32')
    # Record sizes: rows kept, number of features, and memory of the table as float32 vs float64.
    audit['usable_samples'] = len(clean)
    audit['feature_count'] = len(features)
    audit['feature_table_bytes'] = int(clean.memory_usage(deep=True).sum())
    audit['feature_table_bytes_if_float64'] = int(clean.astype('float64').memory_usage(deep=True).sum())
    return clean, features, audit


# ---------------------------------------------------------------------------------------------
# 3. BASELINES, CANDIDATE MODELS AND METRICS
# ---------------------------------------------------------------------------------------------
def clear_sky_persistence(x):
    """Smart baseline: keep today's 'cloudiness' but follow the sun.

    Forecast = current power × (clear-sky shape next hour / clear-sky shape now).
    Near sunrise the ratio is unstable, so we fall back to plain persistence there.
    """
    # Ratio of clear-sky shapes; 1.0 (plain persistence) when the sun is still very low.
    ratio = np.where(x.cs_now > .1, x.cs_next / np.maximum(x.cs_now, 1e-6), 1.0)
    # Zero after sunset; otherwise scale the current power by the ratio.
    forecast = np.where(x.cs_next <= 0, 0.0, x.power * ratio)
    # Never below 0 or above the 5 kW capacity.
    return np.clip(forecast, 0, CAPACITY_KW)


def candidate_models():
    """Declared before validation. None of these settings was tuned on the 2023 test year."""
    return {
        # Ridge: linear model with a small penalty; the scaler makes all inputs comparable.
        'Ridge regression': make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        # Random forest: 200 deep-ish trees on bootstrap samples, 60% of features tried per split; all CPU cores.
        'Random forest': RandomForestRegressor(n_estimators=200, max_depth=14, min_samples_leaf=5,
                                               max_features=.6, n_jobs=-1, random_state=42),
        # Extra trees: like a random forest but with random split thresholds (more variety, often smoother).
        'Extra trees': ExtraTreesRegressor(n_estimators=200, max_depth=16, min_samples_leaf=5,
                                           max_features=.6, n_jobs=-1, random_state=42),
        # Histogram gradient boosting: small trees added one by one, each fixing the previous errors.
        'Gradient boosting (31 leaves)': HistGradientBoostingRegressor(max_iter=400, learning_rate=.05, max_leaf_nodes=31,
                                                                       min_samples_leaf=20, l2_regularization=1.0, random_state=42),
        'Gradient boosting (63 leaves)': HistGradientBoostingRegressor(max_iter=600, learning_rate=.03, max_leaf_nodes=63,
                                                                       min_samples_leaf=30, l2_regularization=3.0, random_state=42),
    }


def predict(name, model, x):
    """Forecast with a baseline or a fitted model, always within the physical range 0..capacity."""
    # Plain persistence: next hour = current power.
    if name == 'Persistence':
        return x.power.to_numpy().astype(float)
    if name == 'Clear-sky persistence':
        return clear_sky_persistence(x)
    # Learned models: predict, then keep the result inside 0..capacity.
    forecast = np.clip(model.predict(x), 0, CAPACITY_KW)
    # Physical rule known in advance: the sun is below the horizon next hour -> no PV output.
    return np.where(x.elev_next.to_numpy() <= -1, 0.0, forecast)


def daylight(x):
    """Scoring mask: the sun is above the horizon at the target hour (night power is trivially zero)."""
    return (x.elev_next > 0).to_numpy()


def score(y, p):
    """All regression metrics. Accuracy = 1 − RMSE / capacity (capacity-normalised RMSE, NB/T 32011-2013)."""
    # Convert inputs to float arrays; err > 0 means over-prediction.
    y, p = np.asarray(y, float), np.asarray(p, float)
    err = p - y
    rmse = float(np.sqrt(np.mean(err ** 2)))
    # MAE and RMSE in kW, R², error as % of capacity, accuracy, average bias and sample count.
    out = {'MAE_kW': float(np.mean(np.abs(err))), 'RMSE_kW': rmse, 'R2': float(r2_score(y, p)),
           'nMAE_percent': float(100 * np.mean(np.abs(err)) / CAPACITY_KW),
           'accuracy_percent': float(100 * (1 - rmse / CAPACITY_KW)),
           'bias_kW': float(np.mean(err)), 'samples': int(len(y))}
    for t in TOLERANCES:
        # Share of forecasts whose absolute error is within ±t of capacity.
        out[f'within_{int(t * 100)}pct'] = float(100 * np.mean(np.abs(err) <= t * CAPACITY_KW))
    return out


def drop_events(x, next_power):
    """Weather-driven drop: next-hour power falls well below the clear-sky expectation (dust or cloud)."""
    # What clear-sky persistence expects for the next hour.
    expected = clear_sky_persistence(x)
    # How far below that expectation the (actual or forecast) next-hour power is.
    shortfall = expected - np.asarray(next_power, float)
    return (shortfall >= DROP_RELATIVE * expected) & (shortfall >= DROP_ABSOLUTE_KW) & daylight(x)


# ---------------------------------------------------------------------------------------------
# 4. THE EXPERIMENT
# ---------------------------------------------------------------------------------------------
def run():
    # Create the results folder; `stages` collects time and memory per stage.
    OUT.mkdir(exist_ok=True)
    stages = {}

    def measure(name, func):
        """Run one stage while recording its run time and peak traced Python memory."""
        # Start tracing Python memory allocations and the clock.
        tracemalloc.start()
        start = time.perf_counter()
        result = func()
        # Store elapsed seconds and the peak traced memory in MiB.
        stages[name] = {'seconds': round(time.perf_counter() - start, 3),
                        'peak_python_MiB': round(tracemalloc.get_traced_memory()[1] / 1024 ** 2, 2)}
        tracemalloc.stop()
        return result

    # Overall timer for the whole experiment.
    total_start = time.perf_counter()
    d, features, audit = measure('load_audit_features', prepare)
    year = d.index.year
    # Development = 2019-2022. The last development hour is purged because its target is in 2023.
    dev = d.loc[(year <= DEV_YEARS[-1]) & (d.index + pd.Timedelta(hours=1) < pd.Timestamp(f'{TEST_YEAR}-01-01'))]
    # The untouched test year.
    test = d.loc[year == TEST_YEAR]
    # Two baselines plus the five learned candidates.
    names = ['Persistence', 'Clear-sky persistence'] + list(candidate_models())

    # ---- 4a. Chronological validation: train on earlier years, validate on the next year ----
    def validate():
        # `rows` collects fold scores; `sizes` the saved size of each fitted model.
        rows, sizes = [], {}
        # Fold 1: train 2019 -> validate 2020; fold 2: 2019-2020 -> 2021; fold 3: 2019-2021 -> 2022.
        for k, val_year in enumerate(DEV_YEARS[1:], 1):
            fit = dev.loc[dev.index.year < val_year]
            # Purge the last fitting hour: its target belongs to the validation year.
            fit = fit.iloc[:-1]
            val = dev.loc[dev.index.year == val_year]
            # Safety check: no training target may be later than the first validation hour.
            assert fit.index.max() + pd.Timedelta(hours=1) < val.index.min()
            # Score daylight hours only.
            mask = daylight(val)
            for name in names:
                model = None
                # Time the fit of each model.
                t0 = time.perf_counter()
                if name in candidate_models():
                    model = clone(candidate_models()[name]).fit(fit[features], fit.target)
                    # Measure the model's size in bytes by serialising it in memory.
                    sizes[name] = len(pickle.dumps(model))
                seconds = time.perf_counter() - t0
                # Forecast the validation year and record all metrics.
                p = predict(name, model, val[features])
                rows.append({'model': name, 'fold': k, 'fit_years': f'{DEV_YEARS[0]}-{val_year - 1}',
                             'validation_year': val_year, 'fit_seconds': round(seconds, 2),
                             **score(val.target[mask], p[mask])})
        return pd.DataFrame(rows), sizes
    cv, model_bytes = measure('validation_3_folds', validate)
    cv.to_csv(OUT / 'validation.csv', index=False)
    # Selection rule (declared in advance): lowest mean daylight RMSE over the three folds,
    # which is the same as the highest mean forecast accuracy (1 − RMSE / capacity).
    # Only learned models compete; the baselines are references.
    learned = cv[cv.model.isin(candidate_models())]
    winner = learned.groupby('model').RMSE_kW.mean().idxmin()

    # ---- 4b. Refit every learned model on all development years; score 2023 once ----
    def final_fit():
        # Refit each learned model on all development years (2019-2022).
        fitted = {}
        for name, template in candidate_models().items():
            fitted[name] = clone(template).fit(dev[features], dev.target)
        return fitted
    fitted = measure('final_fit_2019_2022', final_fit)

    def evaluate():
        # Keep the inputs needed by the dashboard next to every model's forecast.
        predictions = test[['power', 'target', 'ghi', 'ghi_clearsky', 'kt', 'temp_c', 'elev_next', 'cs_now', 'cs_next']].copy()
        results = []
        # Forecast 2023 once with every model and score all hours and daylight hours.
        for name in names:
            p = predict(name, fitted.get(name), test[features])
            predictions[name] = p
            for period, mask in [('All hours', np.ones(len(test), bool)), ('Daylight', daylight(test))]:
                results.append({'model': name, 'period': period, **score(test.target[mask], p[mask])})
        return predictions, pd.DataFrame(results)
    predictions, metrics = measure('test_2023_evaluation', evaluate)
    # Express each model's MAE as % improvement over both baselines.
    for period in metrics.period.unique():
        rows = metrics.period == period
        for base in ['Persistence', 'Clear-sky persistence']:
            ref = metrics.loc[rows & (metrics.model == base), 'MAE_kW'].iloc[0]
            col = 'MAE_gain_vs_persistence' if base == 'Persistence' else 'MAE_gain_vs_clear_sky'
            metrics.loc[rows, col] = 100 * (1 - metrics.loc[rows, 'MAE_kW'] / ref)
    metrics.to_csv(OUT / 'metrics.csv', index=False)

    # ---- 4c. Sudden weather-driven drop warnings (classification view of the same forecasts) ----
    # Actual drops come from the real next-hour power; each model warns from its own forecast.
    day = daylight(test)
    truth = drop_events(test[features], test.target)
    predictions['actual_drop'] = truth
    expected = clear_sky_persistence(test[features])
    warnings = []
    # Precision, recall, F1 and confusion matrix for every model's warnings.
    for name in names:
        alert = drop_events(test[features], predictions[name])
        # The selected model's warnings are saved for the dashboard.
        if name == winner:
            predictions['warning'] = alert
        p_, r_, f_, _ = precision_recall_fscore_support(truth[day], alert[day], average='binary', zero_division=0)
        # A continuous warning strength (relative shortfall) for the ranking score.
        strength = (expected - predictions[name].to_numpy()) / np.maximum(expected, .05)
        warnings.append({'model': name, 'precision': float(p_), 'recall': float(r_), 'F1': float(f_),
                         'events': int(truth[day].sum()),
                         'confusion_matrix_TN_FP_FN_TP': confusion_matrix(truth[day], alert[day], labels=[False, True]).ravel().tolist(),
                         'average_precision': float(average_precision_score(truth[day], strength[day]))})
    # Save all 2023 forecasts (4 decimals keeps the file small, about 1.3 MB).
    predictions.index.name = 'timestamp'
    predictions.to_csv(OUT / 'predictions.csv', float_format='%.4f')

    # ---- 4d. Save the model and the audit (with all resource measurements) ----
    # The fitted model is saved locally (about 44 MB) and excluded from Git; the dashboard does not need it.
    joblib.dump({'model': fitted[winner], 'features': features, 'selected_model': winner}, OUT / 'model.joblib')
    # Add the decisions, split sizes, rules, warning scores and resource measurements to the audit.
    audit.update(selected_model=winner,
                 selection_rule='Lowest mean daylight RMSE (= highest mean accuracy) over 3 yearly chronological folds',
                 development_years=DEV_YEARS, test_year=TEST_YEAR,
                 development_samples=len(dev), test_samples=len(test), test_daylight_samples=int(day.sum()),
                 horizon='1 hour ahead', capacity_kW=CAPACITY_KW, tolerances=TOLERANCES,
                 drop_rule={'relative': DROP_RELATIVE, 'absolute_kW': DROP_ABSOLUTE_KW,
                            'expectation': 'clear-sky persistence of current power'},
                 warnings=warnings, features=features,
                 model_pickle_bytes=model_bytes, resources=stages,
                 runtime_seconds=round(time.perf_counter() - total_start, 2))
    (OUT / 'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    # Extra evidence (importance, seasons, bootstrap) is created by the diagnostics module.
    from src.diagnostics import build_diagnostics
    build_diagnostics(ROOT, fitted[winner], d, features, dev)
    # Print a short summary in the terminal.
    print('Selected:', winner)
    print(metrics[metrics.period == 'Daylight'][['model', 'MAE_kW', 'RMSE_kW', 'R2', 'accuracy_percent', 'within_10pct']].round(3).to_string(index=False))


# Run the experiment when the file is executed: python -m src.pipeline
if __name__ == '__main__':
    run()
