# Solar Outlook Riyadh: One Hour Ahead Photovoltaic Power Forecasting and Drop Warnings

**Lolwah Alsaadoun** (223410496) · **Almas Alrasheedi** (223410901) · **Jood Alghamdi** (223410448)

*CS316: Introduction to AI and Data Science — Term 261 · Instructor: Dr. Heyfa Ammar*\
*Department of Computer Science, Prince Sultan University, Riyadh, Saudi Arabia*

- **Code:** <https://github.com/Lolwah-sa/Solar-Outlook-Riyadh>
- **Dashboard:** <https://solar-outlook-riyadh.streamlit.app/>

Solar Outlook Riyadh forecasts the output of a 5 kWp photovoltaic (PV) system in **Riyadh, Saudi Arabia**
**one hour ahead**, and warns about **sudden weather-driven drops** caused by dust or cloud. It uses five
years of hourly data (2019–2023):
- weather and irradiance from **NASA POWER**;
- PV output simulated for Riyadh by the EU's **PVGIS** tool.

Baselines and machine-learning models are compared with **year-by-year chronological validation**, and the
selected model is tested once on the untouched year **2023**. The project supports **Saudi Vision 2030**
(about 50% renewable electricity by 2030) and **UN SDG 7 and SDG 13**: better short-term solar forecasts
help grid operators use more solar power with less fossil back-up.

> **Data note.** The PV output is *simulated* by PVGIS from satellite irradiance for a 5 kWp system; it is
> not metered output from a real plant. Real plants add inverter faults, soiling and outages, so real-world
> accuracy would be lower.

---

## Key results (test year 2023, daylight hours)

| Model | Forecast accuracy (1 − nRMSE) | Within ±10% of capacity | MAE (kW) | RMSE (kW) | R² |
|---|---:|---:|---:|---:|---:|
| Persistence (next = now) | 84.8% | 42.4% | 0.620 | 0.761 | 0.671 |
| Clear-sky persistence | 90.5% | 82.3% | 0.296 | 0.476 | 0.871 |
| Ridge regression | 93.4% | 90.6% | 0.202 | 0.328 | 0.939 |
| Random forest | 94.3% | 92.9% | 0.141 | 0.287 | 0.953 |
| **Extra trees (selected)** | **94.2%** | **93.1%** | **0.142** | **0.288** | **0.953** |
| Gradient boosting (31 leaves) | 94.0% | 92.6% | 0.152 | 0.298 | 0.950 |
| Gradient boosting (63 leaves) | 94.1% | 92.5% | 0.151 | 0.297 | 0.950 |

- **How accuracy is defined:**
  - **Forecast accuracy** = 1 − RMSE / 5 kW capacity. This is the capacity-normalised RMSE used by the
    PV forecasting standard NB/T 32011-2013.
  - **Within ±10%** = share of forecasts whose error is at most 0.5 kW.
- **Over all hours** (including night), the selected model reaches **95.9%** accuracy and **96.5%**
  within ±10%.
- **Improvement over the smart baseline:** daylight MAE is **51.9% lower than clear-sky persistence**
  (95% day-block bootstrap: 48.9–55.2%).
- **By season:** the model is best in summer (98.3% accuracy, 99.7% within ±10%) and hardest in winter
  and spring (about 92.7% accuracy, 89% within ±10%).
- **By sky condition:**
  - clear days: 98.8% within ±10%;
  - dusty/cloudy days (daily clear-sky index < 0.7): 69.8% within ±10%.
- **Weather-driven drop warnings:** 80.1% precision, 59.5% recall, F1 0.683. That's 301 of 506 drop hours
  detected, with 75 false alarms.
- **Close race:** extra trees won validation by a hair over random forest (mean RMSE 0.2271 vs
  0.2274 kW). Random forest is marginally better on 2023, but the choice was fixed before testing.

---

## Repository structure

```text
.
├── README.md                ← this file
├── app.py                   ← Streamlit dashboard (reads artifacts/ only; never retrains)
├── download_data.py         ← re-downloads the dataset from NASA POWER + PVGIS (optional)
├── requirements.txt         ← packages; requirements-lock.txt = versions used for the saved results
├── .streamlit/config.toml   ← light theme
├── data/raw/
│   └── saudi_solar_riyadh_2019_2023.csv   ← 43,824 hourly rows (included, 3.8 MB)
├── src/
│   ├── pipeline.py          ← audit, features, baselines, models, validation, test, warnings, resources
│   └── diagnostics.py       ← seasons, sky conditions, monthly scores, importance, bootstrap
├── tests/
│   └── test_pipeline.py     ← 10 tests: solar geometry, target, no leakage, data gaps, formulas, saved results
├── docs/
│   ├── dashboard.html       ← stand-alone HTML dashboard (open in any browser)
│   ├── figures/             ← report figures
│   └── Solar_Outlook_Riyadh_Report.docx / .pdf   ← IEEE report
└── artifacts/               ← saved results used by the dashboards and the report
```

Every Python file is commented to explain **what** each step does and **why**.

---

## How to run

**Requirements:** Python 3.10 or newer (the results were produced with Python 3.13).

### Windows (PowerShell, VS Code terminal)

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.pipeline              # 1. run the experiment (~2 minutes)
.\.venv\Scripts\python.exe -m unittest discover -s tests # 2. run the 10 tests
.\.venv\Scripts\python.exe -m streamlit run app.py       # 3. open http://localhost:8501
```

### macOS / Linux

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.pipeline
python -m unittest discover -s tests
streamlit run app.py
```

- **Dashboard only:** the saved results are already in `artifacts/`, so `streamlit run app.py` works
  straight after installing the requirements.
- **Without Python:** open `docs/dashboard.html` in a browser.
- **Re-downloading the data** is optional, because the CSV is included:
  - run `python download_data.py`;
  - it needs internet access to `power.larc.nasa.gov` and `re.jrc.ec.europa.eu`.

### Online dashboard (Streamlit Community Cloud)

The dashboard is live at <https://solar-outlook-riyadh.streamlit.app/> (Python 3.13). It is deployed from
the `main` branch of this repository and updates automatically on every push.

---

## Dataset

| Item | Value |
|---|---|
| Location | Riyadh, Saudi Arabia (24.7136°N, 46.6753°E), local time UTC+3 (no daylight saving) |
| Period / resolution | 1 Jan 2019 – 31 Dec 2023, hourly (43,821 hours after trimming to local calendar years; 43,575 usable examples) |
| Weather inputs (NASA POWER, CC BY 4.0) | GHI, clear-sky GHI, DNI, DHI (W/m²), air temperature, relative humidity, wind speed, pressure |
| Target (PVGIS-SARAH3, EU JRC) | Simulated AC output of a 5 kWp system with 14% losses and optimal tilt (kW) |
| Quality checks | 0 missing hours, 0 duplicates, 0 missing or negative values, 0 hours above 5 kW; **43 hours with 0 kW under strong sun (GHI > 200 W/m²) treated as missing** (gaps in the satellite record) |

## Method

1. **Audit.** The pipeline checks the hourly grid for completeness and duplicates, looks for missing,
   negative and out-of-range values, and records the file's SHA-256 hash. It also applies a physical check:
   0 kW while GHI > 200 W/m² is impossible for a working system. Those 43 hours are marked missing (not
   filled in), so they are never used as inputs or targets.
2. **Features.** All 26 features are known at forecast time *t*:
   - current power, power 1–3 hours ago and 24 hours ago, and the latest change;
   - irradiance and weather;
   - the clear-sky index *kt*;
   - the **sun's elevation now and at the target hour** (astronomy, computed with NOAA formulas);
   - hour and day of year as sine/cosine pairs.
3. **Target.** PV power exactly 1 hour after *t*.
4. **Baselines.**
   - *Persistence*: next hour = now.
   - *Clear-sky persistence*: today's sky condition, scaled by where the sun will be.
5. **Models.** Ridge regression, random forest, extra trees, and two histogram gradient boosting settings,
   all with fixed seeds.
6. **Selection.** Each model trains on earlier years and is validated on the next year (2020, 2021 and
   2022). The winner has the lowest mean daylight RMSE, which is the same as the highest accuracy.
   2023 is never used for this choice.
7. **Test.** The models are refitted on 2019–2022 and scored once on 2023. A physical rule sets the
   forecast to 0 when the sun is below the horizon at the target hour.
8. **Warnings.** A weather-driven drop means next-hour power falls ≥20% **and** ≥0.25 kW below the
   clear-sky expectation. A model warns when its own forecast meets the same rule.
9. **Diagnostics:**
   - seasonal, sky-condition and monthly scores;
   - permutation importance on validation year 2022;
   - a 2,000-resample day-block bootstrap.

## Efficiency and memory (details in the report's Appendix A)

- **Vectorized code:** the sun-position formula and all features are vectorized (no Python loops over
  the 43,821 hours).
- **float32 feature table:** **4.8 MiB** instead of 9.3 MiB as float64 (−48%).
- **Measured cost:** each stage's run time and peak Python memory are measured with `time.perf_counter`
  and `tracemalloc` (`artifacts/audit.json`). The full run takes about **96 s**, and validation peaks at
  **96 MiB**.
- **Parallel training:** the forests train on all CPU cores (`n_jobs=-1`).
- **Model size:** every model's saved size is recorded. Extra trees takes 44 MiB, while gradient
  boosting (31 leaves) takes 0.4 MiB for almost the same accuracy (94.0% vs 94.2%). That's a 109×
  smaller choice if memory matters.
- **Lean dashboard:** it reads only small CSV/JSON files (about 1.3 MB), never the raw data or the
  44 MiB model.

## Limitations

- **The target is simulated.** PVGIS output has no inverter faults, soiling or curtailment.
- **The sources differ.** Inputs (NASA POWER) and target (PVGIS-SARAH3) come from different irradiance
  sources, which is realistic but adds noise.
- **Hourly resolution:** fast changes within an hour are not visible.
- **Sudden changes are the hard part.** Accuracy drops on dusty or cloudy days, because the inputs can't
  see dust or cloud arriving. Sky images, satellite cloud motion or weather forecasts would help.
- **No measured impact.** No carbon, financial or grid benefit was measured.

## References

1. NASA Langley Research Center, POWER Project, hourly data API, <https://power.larc.nasa.gov> (CC BY 4.0).
2. European Commission JRC, PVGIS 5.3 (SARAH3), <https://re.jrc.ec.europa.eu/pvg_tools/en/>.
3. F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," *JMLR*, 12, 2011.
4. P. Geurts, D. Ernst, L. Wehenkel, "Extremely randomized trees," *Machine Learning*, 63, 2006.
5. National Energy Administration of China, *NB/T 32011-2013: Technical requirements of power forecasting
   system for PV power station*, 2013.
6. IEA, "Saudi National Renewable Energy Program (NREP)," <https://www.iea.org/policies/14658>.
7. United Nations, "The 17 Sustainable Development Goals," <https://sdgs.un.org/goals>.
