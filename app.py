"""Solar Outlook Riyadh: interactive dashboard for the recorded 1-hour-ahead forecasting experiment.

Run with:  python -m streamlit run app.py
The page only READS the small result files written by `python -m src.pipeline`. It never retrains
a model, downloads data or controls real equipment, so it opens instantly and runs on free hosting.
"""
# json reads the experiment summaries; Path finds files relative to this script on any computer.
import json
from pathlib import Path
# pandas loads and summarises the result tables.
import pandas as pd
# Plotly draws interactive charts; Streamlit turns this script into a web page.
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'artifacts'
# One colour per meaning everywhere: gold = actual, green = model, grey = baselines, blue = sun, red = drops.
SUN, GREEN, GREY, SKY, RED = '#e09a10', '#2f7a5b', '#8a948e', '#3a7bd5', '#d9534f'
st.set_page_config(page_title='Solar Outlook Riyadh', page_icon='☀️', layout='wide')

# ---------------------------------------------------------------------------------------------
# 1. LOAD THE SAVED RESULTS
# ---------------------------------------------------------------------------------------------
required = ['audit.json', 'metrics.csv', 'predictions.csv', 'validation.csv']
missing = [name for name in required if not (OUT / name).exists()]
if missing:
    # A clear message instead of a crash, with the command that creates the files.
    st.error('Missing result files: ' + ', '.join(missing))
    st.info('Run  python -m src.pipeline  from the project folder first.')
    st.stop()


@st.cache_data
def load():
    """Read every result file once; Streamlit caches them so widget changes are instant."""
    audit = json.loads((OUT / 'audit.json').read_text())
    diagnostics = json.loads((OUT / 'diagnostics.json').read_text()) if (OUT / 'diagnostics.json').exists() else {}
    tables = {name: pd.read_csv(OUT / f'{name}.csv') for name in
              ['metrics', 'validation', 'seasonal_metrics', 'sky_condition_metrics', 'monthly_metrics', 'feature_importance']
              if (OUT / f'{name}.csv').exists()}
    # The 2023 forecasts, one row per issue hour.
    pred = pd.read_csv(OUT / 'predictions.csv', parse_dates=['timestamp'])
    # Each forecast is issued at `timestamp` for the following hour (the target time).
    pred['target_time'] = pred.timestamp + pd.Timedelta(hours=1)
    return audit, diagnostics, tables, pred


# Load once (cached) and unpack the tables used below.
audit, diagnostics, T, pred = load()
metrics, cv = T['metrics'], T['validation']
# The model chosen by validation, and the system capacity (5 kW) used for all accuracy figures.
winner = audit['selected_model']
C = audit['capacity_kW']

# ---------------------------------------------------------------------------------------------
# 2. SIDEBAR CONTROLS
# ---------------------------------------------------------------------------------------------
st.sidebar.markdown('### Explore the experiment')
# Every model in the results can be inspected; the validation winner is pre-selected.
names = metrics.model.unique().tolist()
model = st.sidebar.selectbox('Forecast model', names, index=names.index(winner))
# Daylight scores are the honest headline: night power is trivially zero.
period = st.sidebar.radio('Scoring period', ['Daylight', 'All hours'])
# Tolerance band for the 'within ±X%' accuracy card.
tolerance = st.sidebar.select_slider('Accuracy tolerance (± % of 5 kWp capacity)', options=[5, 10, 20], value=10)
# Only dates of the 2023 test year can be chosen.
dates = sorted(pred.target_time.dt.date.unique())
default_day = pd.Timestamp('2023-04-15').date()
chosen = st.sidebar.date_input('Target date (2023 test year)', value=default_day, min_value=dates[0], max_value=dates[-1])
# Light theme by default: it prints clearly in the report and on projectors.
dark = st.sidebar.toggle('Dark theme', value=False)
# Short reminders of the setup and links to the data sources.
st.sidebar.caption(f'Validation winner: {winner}')
st.sidebar.caption('Riyadh (24.71°N, 46.68°E) · 5 kWp PV · 1-hour horizon · 2019–2023')
st.sidebar.markdown('[NASA POWER](https://power.larc.nasa.gov) · [PVGIS](https://re.jrc.ec.europa.eu/pvg_tools/en/)')

# ---------------------------------------------------------------------------------------------
# 3. PAGE STYLE
# ---------------------------------------------------------------------------------------------
# Colours for the dark or light theme: background, cards, borders, text and secondary text.
bg, surface, border, text, muted = ('#121614', '#1b211e', '#2b332f', '#e8ece9', '#aebbb3') if dark else ('#f6f7f4', '#ffffff', '#e4e6df', '#1d2420', '#66706a')
# Custom CSS for the page, cards and header (double braces are literal braces in an f-string).
st.markdown(f'''<style>
 .stApp {{background:{bg};color:{text};}}
 .block-container {{max-width:1320px;padding-top:2rem;}}
 [data-testid="stSidebar"] {{background:{surface};}}
 [data-testid="stMetric"] {{background:{surface};border:1px solid {border};border-radius:14px;padding:14px;}}
 [data-testid="stMetricLabel"] {{color:{muted};min-height:2.6em;}}
 [data-testid="stMetricLabel"] p {{white-space:normal;}}
 [data-testid="stMetricValue"] {{color:{text};font-size:1.6rem;}}
 h1,h2,h3,p,label {{color:{text};}}
 .head {{display:flex;gap:14px;align-items:center;margin-bottom:10px;}}
 .mark {{background:{SUN};color:#fff;border-radius:12px;font-size:26px;padding:6px 12px;}}
 .head h1 {{font-size:26px;margin:0;}} .head p {{color:{muted};margin:4px 0 0;font-size:13px;}}
</style>''', unsafe_allow_html=True)
# Page header and a data note that the PV output is simulated.
st.markdown('''<div class="head"><div class="mark">☀</div><div><h1>Solar Outlook Riyadh</h1>
<p>CS316 · 1-hour-ahead PV power forecast · 5 kWp system · NASA POWER weather + PVGIS simulated PV output</p></div></div>''', unsafe_allow_html=True)
st.info('Data note: the PV output is **simulated** by PVGIS (EU JRC) from satellite irradiance for a 5 kWp system in Riyadh; '
        'the weather inputs come from NASA POWER. The model is trained on 2019–2022 and tested once on the untouched year 2023.')


def chart(fig, height=300):
    """Apply one consistent look (theme, colours, margins) to every Plotly chart."""
    # Match chart colours to the theme and use a horizontal legend above the plot.
    fig.update_layout(template='plotly_dark' if dark else 'plotly_white', height=height, paper_bgcolor=surface,
                      plot_bgcolor=surface, font_color=text, margin=dict(l=10, r=10, t=40, b=20),
                      legend=dict(orientation='h', y=1.15, x=0))
    fig.update_xaxes(gridcolor=border)
    fig.update_yaxes(gridcolor=border)
    # 'stretch' makes the chart fill the width of its card.
    st.plotly_chart(fig, width='stretch')


# ---------------------------------------------------------------------------------------------
# 4. HEADLINE CARDS
# ---------------------------------------------------------------------------------------------
# Scores of the chosen model and of the clear-sky baseline for the chosen period.
row = metrics[(metrics.model == model) & (metrics.period == period)].iloc[0]
base = metrics[(metrics.model == 'Clear-sky persistence') & (metrics.period == period)].iloc[0]
# The scored rows for the chosen period (daylight = sun above the horizon at the target hour).
# Rows used for the tolerance accuracy: daylight only, or all hours.
scored = pred[pred.elev_next > 0] if period == 'Daylight' else pred
# Share of forecasts within ±tolerance of capacity, for the model and the baseline.
within = 100 * ((scored[model] - scored.target).abs() <= tolerance / 100 * C).mean()
within_base = 100 * ((scored['Clear-sky persistence'] - scored.target).abs() <= tolerance / 100 * C).mean()
# Six headline cards: two accuracy measures, R², MAE, RMSE and the gain over the baseline.
cards = st.columns(6)
for col, label, value in zip(cards,
        ['Forecast accuracy (1 − nRMSE)', f'Within ±{tolerance}% of capacity', 'R²', 'MAE · kW', 'RMSE · kW', 'MAE gain vs clear-sky'],
        [f'{row.accuracy_percent:.1f}%', f'{within:.1f}%', f'{row.R2:.3f}', f'{row.MAE_kW:.3f}', f'{row.RMSE_kW:.3f}',
         f'{row.MAE_gain_vs_clear_sky:+.1f}%']):
    col.metric(label, value)
# Exact definitions under the cards, including the baseline for comparison.
st.caption(f'Forecast accuracy = 1 − RMSE / {C:.0f} kW capacity (capacity-normalised RMSE, NB/T 32011-2013); clear-sky persistence: '
           f'{base.accuracy_percent:.1f}%. Within ±{tolerance}% = share of {period.lower()} forecasts with |error| ≤ {tolerance / 100 * C:.2f} kW; '
           f'clear-sky persistence: {within_base:.1f}%. {int(row.samples):,} test hours in 2023.')

# Five tabs keep the page organised.
overview, evaluation, conditions, alerts, methods = st.tabs(
    ['Forecast & weather', 'Model evaluation', 'Seasons & sky conditions', 'Drop warnings', 'Methods & data audit'])

# ---------------------------------------------------------------------------------------------
# 5. TAB — FORECAST & WEATHER (one selected day)
# ---------------------------------------------------------------------------------------------
with overview:
    # All target hours of the chosen date.
    view = pred[pred.target_time.dt.date == chosen]
    left, right = st.columns([2.1, 1])
    with left, st.container(border=True):
        st.subheader('Actual vs. forecast PV power')
        # Line chart: actual output, chosen model, clear-sky baseline, and red markers for drops.
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=view.target_time, y=view.target, name='Actual (PVGIS)', line=dict(color=SUN, width=3)))
        fig.add_trace(go.Scatter(x=view.target_time, y=view[model], name=model, line=dict(color=GREEN, width=2, dash='dash')))
        fig.add_trace(go.Scatter(x=view.target_time, y=view['Clear-sky persistence'], name='Clear-sky persistence', line=dict(color=GREY, width=1, dash='dot')))
        drops = view[view.actual_drop]
        fig.add_trace(go.Scatter(x=drops.target_time, y=drops.target, mode='markers', name='Weather-driven drop', marker=dict(color=RED, size=9)))
        fig.update_layout(yaxis_title='kW', xaxis_title='Target hour (Riyadh time)')
        chart(fig, 340)
    with right, st.container(border=True):
        st.subheader('Selected day')
        # Hourly kW × 1 h = kWh, so the sum of hourly values is the daily energy.
        # Hourly kW summed over a day equals the daily energy in kWh.
        st.metric('Actual energy', f'{view.target.sum():.1f} kWh')
        st.metric('Forecast energy', f'{view[model].sum():.1f} kWh')
        st.metric('Daily mean clear-sky index', f"{view.loc[view.elev_next > 0, 'kt'].mean():.2f}")
        st.caption('Clear-sky index kt = GHI / clear-sky GHI. Values near 1 mean a clear sky; low values indicate dust or cloud.')
    c1, c2 = st.columns(2)
    with c1, st.container(border=True):
        st.subheader('Irradiance and temperature')
        # Irradiance on the left axis and temperature on a second (right) axis.
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=view.timestamp, y=view.ghi, name='GHI', line_color=SKY))
        fig.add_trace(go.Scatter(x=view.timestamp, y=view.ghi_clearsky, name='Clear-sky GHI', line=dict(color=SKY, dash='dot')))
        fig.add_trace(go.Scatter(x=view.timestamp, y=view.temp_c, name='Air temperature', yaxis='y2', line_color=RED))
        fig.update_layout(yaxis_title='W/m²', yaxis2=dict(title='°C', overlaying='y', side='right'))
        chart(fig)
    with c2, st.container(border=True):
        st.subheader('GHI vs. next-hour power (2023)')
        # A random sample of 3,000 hours keeps the scatter plot fast.
        sample = scored.sample(min(3000, len(scored)), random_state=1)
        chart(px.scatter(sample, x='ghi', y='target', color='kt', opacity=.6, color_continuous_scale='Viridis',
                         labels={'ghi': 'GHI at issue time (W/m²)', 'target': 'PV power 1 h later (kW)', 'kt': 'kt'}))

# ---------------------------------------------------------------------------------------------
# 6. TAB — MODEL EVALUATION
# ---------------------------------------------------------------------------------------------
with evaluation:
    st.info('The winner was chosen by the lowest mean daylight RMSE over three yearly folds (2020, 2021, 2022), before 2023 was scored.')
    c1, c2 = st.columns(2)
    with c1, st.container(border=True):
        st.markdown('#### Chronological validation (daylight RMSE)')
        # Grouped bars: RMSE of each model in each validation year.
        chart(px.bar(cv, x='model', y='RMSE_kW', color=cv.validation_year.astype(str), barmode='group',
                     labels={'RMSE_kW': 'RMSE (kW)', 'color': 'Validation year', 'model': ''}), 360)
    with c2, st.container(border=True):
        st.markdown(f'#### Accuracy within tolerance ({period.lower()})')
        # Accuracy at ±5/10/20% for every model, computed from the saved forecasts.
        rows = []
        for m in names:
            for t in [5, 10, 20]:
                rows.append({'Model': m, 'Tolerance': f'±{t}%', 'Accuracy (%)': 100 * ((scored[m] - scored.target).abs() <= t / 100 * C).mean()})
        chart(px.bar(pd.DataFrame(rows), x='Model', y='Accuracy (%)', color='Tolerance', barmode='group', text_auto='.0f'), 360)
    st.markdown('#### All models on the 2023 test year')
    # Full metrics table for the chosen period.
    st.dataframe(metrics[metrics.period == period].round(3), hide_index=True, width='stretch')
    if diagnostics:
        # Bootstrap interval of the improvement over clear-sky persistence.
        g = diagnostics['gain_vs_clear_sky_persistence']
        st.write(f"**Uncertainty:** 95% day-block bootstrap interval for the daylight MAE improvement of {winner} over clear-sky persistence: "
                 f"{g['low_percent']:.1f}% to {g['high_percent']:.1f}% ({g['days']} days, {g['repetitions']} resamples).")
    c1, c2 = st.columns(2)
    with c1, st.container(border=True):
        st.markdown('#### Residuals (forecast − actual)')
        # Histogram of errors: narrow and centred on 0 is best.
        chart(px.histogram(x=scored[model] - scored.target, nbins=60, color_discrete_sequence=[GREEN],
                           labels={'x': 'Error (kW)'}), 300)
    with c2, st.container(border=True):
        st.markdown('#### Permutation importance (validation year 2022)')
        # Ten most important inputs, smallest at the bottom of the bar chart.
        imp = T['feature_importance'].head(10).sort_values('importance_kW')
        chart(px.bar(imp, x='importance_kW', y='feature', orientation='h', color_discrete_sequence=[GREEN],
                     labels={'importance_kW': 'MAE increase when shuffled (kW)', 'feature': ''}), 300)

# ---------------------------------------------------------------------------------------------
# 7. TAB — SEASONS AND SKY CONDITIONS
# ---------------------------------------------------------------------------------------------
with conditions:
    c1, c2 = st.columns(2)
    with c1, st.container(border=True):
        st.markdown('#### Accuracy by season (daylight, 2023)')
        # Seasonal accuracy (1 − nRMSE) for the two baselines and the selected model.
        chart(px.bar(T['seasonal_metrics'], x='season', y='accuracy_percent', color='model', barmode='group',
                     color_discrete_sequence=[GREY, SKY, GREEN], labels={'accuracy_percent': 'Accuracy 1 − nRMSE (%)', 'season': ''}), 340)
    with c2, st.container(border=True):
        st.markdown('#### Within ±10% by sky condition (daylight, 2023)')
        # Share within ±10% on clear, hazy and dusty/cloudy days.
        chart(px.bar(T['sky_condition_metrics'], x='sky', y='within_10pct', color='model', barmode='group',
                     color_discrete_sequence=[GREY, SKY, GREEN], labels={'within_10pct': 'Within ±10% (%)', 'sky': ''}), 340)
    with st.container(border=True):
        st.markdown('#### Monthly accuracy of each model (daylight, 2023)')
        # Month-by-month share within ±10%.
        chart(px.line(T['monthly_metrics'], x='month', y='within_10pct', color='model', markers=True,
                      color_discrete_sequence=[GREY, SKY, GREEN], labels={'within_10pct': 'Within ±10% (%)'}), 320)
    st.caption('Sky condition uses the daily mean clear-sky index kt. Dusty or cloudy days (kt < 0.7) are the hardest: '
               'irradiance can change quickly and is not visible in the current-hour inputs.')

# ---------------------------------------------------------------------------------------------
# 8. TAB — WEATHER-DRIVEN DROP WARNINGS
# ---------------------------------------------------------------------------------------------
with alerts:
    # The frozen drop rule saved by the pipeline.
    rule = audit['drop_rule']
    st.write(f"A **weather-driven drop** happens when next-hour power falls at least **{rule['relative']:.0%}** and "
             f"**{rule['absolute_kW']:.2f} kW** below the clear-sky expectation (dust or cloud), during daylight. "
             'A model raises a warning when its own forecast meets the same rule. Thresholds were fixed before testing.')
    # Warning scores of every model; select the chosen model's row.
    w = pd.DataFrame(audit['warnings'])
    sel = w[w.model == model].iloc[0]
    # Confusion matrix order from scikit-learn: TN, FP, FN, TP.
    tn, fp, fn, tp = sel.confusion_matrix_TN_FP_FN_TP
    # Four cards with the warning scores of the chosen model.
    c = st.columns(4)
    c[0].metric('Precision', f'{sel.precision:.1%}')
    c[1].metric('Recall', f'{sel.recall:.1%}')
    c[2].metric('F1', f'{sel.F1:.3f}')
    c[3].metric('Detected / actual drops', f'{tp} / {tp + fn}')
    c1, c2 = st.columns([1, 1.5])
    with c1, st.container(border=True):
        # 2×2 heatmap of the confusion matrix with the counts written in each cell.
        chart(px.imshow([[tn, fp], [fn, tp]], x=['No warning', 'Warning'], y=['No drop', 'Drop'], text_auto=True,
                        color_continuous_scale='Greens', title='Actual drops vs. warnings'), 320)
    with c2, st.container(border=True):
        # Scores of every model's warnings for comparison.
        st.dataframe(w.drop(columns='confusion_matrix_TN_FP_FN_TP').round(3), hide_index=True, width='stretch')
        st.caption('Clear-sky persistence never warns because it IS the expectation. Average precision ranks hours by forecast shortfall.')

# ---------------------------------------------------------------------------------------------
# 9. TAB — METHODS, DATA AUDIT AND EFFICIENCY
# ---------------------------------------------------------------------------------------------
with methods:
    # One-line summary of each stage of the method.
    st.dataframe(pd.DataFrame([
        ('Data', 'NASA POWER hourly weather/irradiance + PVGIS-SARAH3 simulated output of a 5 kWp system, Riyadh, 2019–2023 (43,821 hours).'),
        ('Checks', f"Complete hourly grid, no duplicates, no missing or negative values, no power above capacity; "
                   f"{audit['pv_zero_under_sun_set_missing']} hours with 0 kW under strong sun (GHI > 200 W/m²) treated as missing satellite gaps, not filled in."),
        ('Features', f"{audit['feature_count']} inputs known at issue time: current and past power, irradiance, weather, clear-sky index, "
                     'sun elevation now and next hour (astronomy), calendar.'),
        ('Target', 'PV power exactly one hour after the issue time.'),
        ('Split', 'Development 2019–2022; untouched test year 2023. Boundary label purged.'),
        ('Selection', audit['selection_rule']),
        ('Baselines', 'Persistence (next = now) and clear-sky persistence (follows the sun at today\'s sky condition).'),
        ('Accuracy', 'Main: 1 − RMSE / capacity. Also share within ±5/10/20% of capacity, MAE, RMSE, R², bootstrap interval.'),
    ], columns=['Stage', 'What was done']), hide_index=True, width='stretch')
    c1, c2 = st.columns(2)
    with c1, st.container(border=True):
        st.markdown('#### Data audit')
        # Key data-quality numbers from the audit (expandable).
        st.json({k: audit[k] for k in ['rows', 'start', 'end', 'duplicate_timestamps', 'missing_hours', 'missing_values',
                                       'power_above_capacity', 'pv_zero_under_sun_set_missing', 'usable_samples', 'sha256']}, expanded=False)
    with c2, st.container(border=True):
        st.markdown('#### Run time, memory and model size')
        # Run time and peak Python memory of each pipeline stage.
        res = pd.DataFrame(audit['resources']).T.reset_index().rename(columns={'index': 'stage'})
        st.dataframe(res, hide_index=True, width='stretch')
        # Saved size of each fitted model in MiB (memory trade-off between models).
        sizes = pd.Series(audit['model_pickle_bytes']).div(1024 ** 2).round(2).rename('model size (MiB)')
        st.dataframe(sizes, width='stretch')
        st.caption(f"Feature table: {audit['feature_table_bytes'] / 1024 ** 2:.1f} MiB as float32 "
                   f"(would be {audit['feature_table_bytes_if_float64'] / 1024 ** 2:.1f} MiB as float64).")
    # Let the viewer download every 2023 forecast.
    st.download_button('Download 2023 forecasts', pred.to_csv(index=False), 'riyadh_forecasts_2023.csv')

# Footer: scope of this prototype.
st.caption('Academic prototype · simulated PV output · not for operating real equipment.')
