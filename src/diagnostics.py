# Extra evidence for the dashboard and the report: which inputs matter, how accuracy changes with
# the season and with dusty/cloudy days, and how uncertain the improvement over the baseline is.
# It is called automatically at the end of src/pipeline.py and never re-selects the model.
"""Diagnostics for the selected model: importance, seasons, sky condition and bootstrap uncertainty."""
# Path locates the artifacts folder; json writes the summary file.
from pathlib import Path
import json
# NumPy for the vectorized bootstrap; pandas for grouping by season, day and month.
import numpy as np
import pandas as pd
# clone + permutation_importance explain the model on data it was NOT trained on.
from sklearn.base import clone
from sklearn.inspection import permutation_importance

# Reuse the exact rules of the main experiment so numbers are consistent everywhere.
from src.pipeline import CAPACITY_KW, DEV_YEARS, candidate_models, daylight, score

SEASONS = {12: 'Winter (DJF)', 1: 'Winter (DJF)', 2: 'Winter (DJF)', 3: 'Spring (MAM)', 4: 'Spring (MAM)',
           5: 'Spring (MAM)', 6: 'Summer (JJA)', 7: 'Summer (JJA)', 8: 'Summer (JJA)', 9: 'Autumn (SON)',
           10: 'Autumn (SON)', 11: 'Autumn (SON)'}


def day_block_bootstrap(frame, model, baseline, repetitions=2000, seed=42):
    """95% interval for the MAE improvement over a baseline, resampling whole days (keeps within-day dependence)."""
    # Absolute error of the model (m) and of the baseline (b) for every hour.
    frame = frame.assign(m=(frame[model] - frame.target).abs(), b=(frame[baseline] - frame.target).abs())
    # One row per day: total absolute error of the model and of the baseline.
    daily = frame.groupby(frame.index.date)[['m', 'b']].sum()
    # Fixed seed -> the same interval every time the code runs.
    rng = np.random.default_rng(seed)
    # Each row of `draws` is one resampled year of days (with replacement), computed in one step.
    draws = rng.integers(0, len(daily), size=(repetitions, len(daily)))
    # Total errors of each resampled year, for model and baseline (no Python loop).
    m = daily.m.to_numpy()[draws].sum(axis=1)
    b = daily.b.to_numpy()[draws].sum(axis=1)
    # Percentage MAE improvement in each resampled year.
    gain = 100 * (1 - m / b)
    # The middle 95% of the 2,000 improvements is the interval.
    low, high = np.quantile(gain, [.025, .975])
    return {'low_percent': float(low), 'high_percent': float(high), 'days': int(len(daily)),
            'repetitions': repetitions, 'seed': seed, 'baseline': baseline}


# Main entry point, called at the end of src/pipeline.py. `final_model` is kept for API compatibility.
def build_diagnostics(root, final_model, d, features, dev):
    # Load the saved results of the experiment.
    out = Path(root) / 'artifacts'
    audit = json.loads((out / 'audit.json').read_text())
    winner = audit['selected_model']
    pred = pd.read_csv(out / 'predictions.csv', index_col='timestamp', parse_dates=True)
    # Daylight hours of 2023 only (the sun is up at the target hour).
    day = pred[pred.elev_next > 0].copy()
    # Compare the selected model with both baselines everywhere below.
    models = [c for c in ['Persistence', 'Clear-sky persistence', winner]]

    # --- 1. Seasonal accuracy on the 2023 test year (daylight hours) ---
    # Map each month to its meteorological season.
    day['season'] = day.index.month.map(SEASONS)
    seasons = []
    # Score each model inside each season.
    for season, g in day.groupby('season'):
        for m in models:
            seasons.append({'season': season, 'model': m, **score(g.target, g[m])})
    pd.DataFrame(seasons).to_csv(out / 'seasonal_metrics.csv', index=False)

    # --- 2. Clear versus dusty/cloudy days (daily mean clear-sky index of the irradiance) ---
    # Average clear-sky index per day: near 1 = clear, low = dusty or cloudy.
    daily_kt = day.groupby(day.index.date).kt.mean()
    sky = pd.cut(daily_kt, [-1, .7, .9, 2], labels=['Dusty/cloudy (kt < 0.7)', 'Partly hazy (0.7-0.9)', 'Clear (kt >= 0.9)'])
    # Give every hour the sky label of its day.
    day['sky'] = pd.Series(sky.to_numpy(), index=daily_kt.index).reindex(day.index.date).to_numpy()
    sky_rows = []
    # Score each model inside each sky-condition group.
    for label, g in day.groupby('sky', observed=True):
        for m in models:
            sky_rows.append({'sky': str(label), 'days': int(g.index.normalize().nunique()), 'model': m, **score(g.target, g[m])})
    pd.DataFrame(sky_rows).to_csv(out / 'sky_condition_metrics.csv', index=False)

    # --- 3. Monthly accuracy of the selected model (for the dashboard line chart) ---
    monthly = []
    # One accuracy value per month and model (12 x 3 rows).
    for month, g in day.groupby(day.index.month):
        for m in models:
            s = score(g.target, g[m])
            monthly.append({'month': int(month), 'model': m, 'accuracy_percent': s['accuracy_percent'],
                            'within_10pct': s['within_10pct'], 'MAE_kW': s['MAE_kW']})
    pd.DataFrame(monthly).to_csv(out / 'monthly_metrics.csv', index=False)

    # --- 4. Permutation importance on the LAST validation year (2022), never on the test year ---
    # Train on 2019-2021 and explain on 2022, so the explanation is not based on training data.
    fit = dev.loc[dev.index.year < DEV_YEARS[-1]].iloc[:-1]
    val = dev.loc[dev.index.year == DEV_YEARS[-1]]
    # 3,000 random daylight hours keep the computation fast.
    val = val.loc[daylight(val)].sample(n=min(3000, int(daylight(val).sum())), random_state=42)
    fold_model = clone(candidate_models()[winner]).fit(fit[features], fit.target)
    # Shuffle one feature at a time (5 times) and measure how much worse the MAE gets.
    perm = permutation_importance(fold_model, val[features], val.target, scoring='neg_mean_absolute_error',
                                  n_repeats=5, random_state=42, n_jobs=1)
    # Sort from most to least important and save.
    importance = pd.DataFrame({'feature': features, 'importance_kW': perm.importances_mean,
                               'std_kW': perm.importances_std}).sort_values('importance_kW', ascending=False)
    importance.to_csv(out / 'feature_importance.csv', index=False)

    # --- 5. Uncertainty of the improvement over both baselines (day-block bootstrap) ---
    # Collect the bootstrap intervals and day counts in one JSON file.
    summary = {
        'selected_model': winner,
        'gain_vs_persistence': day_block_bootstrap(day, winner, 'Persistence'),
        'gain_vs_clear_sky_persistence': day_block_bootstrap(day, winner, 'Clear-sky persistence'),
        'sky_day_counts': {str(k): int(v) for k, v in sky.value_counts().items()},
        'importance_method': 'Permutation importance (MAE increase, kW) on 3,000 daylight hours of validation year 2022; model fitted on 2019-2021',
        'capacity_kW': CAPACITY_KW,
    }
    # Write the summary for the dashboard and report.
    (out / 'diagnostics.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary
