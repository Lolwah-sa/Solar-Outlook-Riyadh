"""
Download a Saudi solar dataset for the CS316 solar power forecasting project.

Sources (both free, no API key):
  1. NASA POWER hourly point API  -> weather + irradiance (GHI, clear-sky GHI, DNI, DHI, temp, humidity, wind, pressure)
     https://power.larc.nasa.gov  (CC BY 4.0, hourly from 2001)
  2. PVGIS (EU JRC) seriescalc    -> simulated hourly PV output (P, in W) for a system you define
     https://re.jrc.ec.europa.eu  (years 2005-2023)

Output: one CSV per city in ./data/raw/, hourly, Asia/Riyadh local time.
The project already includes data/raw/saudi_solar_riyadh_2019_2023.csv, so running this script is optional.

Usage:
  pip install requests pandas pvlib
  python get_saudi_solar_data.py                       # Riyadh, 2019-2023
  python get_saudi_solar_data.py --cities all          # all five cities
  python get_saudi_solar_data.py --start 2015 --end 2023 --peak-kw 5
"""
# argparse reads command-line options; os creates the output folder; time pauses between API calls.
import argparse
import os
import time

# pandas builds the hourly table; requests calls the NASA POWER web API.
import pandas as pd
import requests

# Latitude and longitude of each supported city (the project uses Riyadh).

CITIES = {
    "riyadh": (24.7136, 46.6753),
    "jeddah": (21.4858, 39.1925),
    "dammam": (26.4207, 50.0888),
    "abha":   (18.2164, 42.5053),
    "tabuk":  (28.3835, 36.5662),
}

# NASA POWER variable codes -> readable column names used by the project.
NASA_PARAMS = {
    "ALLSKY_SFC_SW_DWN":  "ghi",           # W/m2, all-sky global horizontal irradiance
    "CLRSKY_SFC_SW_DWN":  "ghi_clearsky",  # W/m2, clear-sky GHI
    "ALLSKY_SFC_SW_DNI":  "dni",           # W/m2
    "ALLSKY_SFC_SW_DIFF": "dhi",           # W/m2
    "T2M":   "temp_c",                     # deg C at 2 m
    "RH2M":  "rel_humidity",               # % at 2 m
    "WS10M": "wind_speed",                 # m/s at 10 m
    "PS":    "pressure_kpa",               # kPa
}
NASA_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"


def get_nasa_power(lat, lon, year):
    """One year of hourly NASA POWER data, requested in UTC and converted to Asia/Riyadh."""
    params = {
        "start": f"{year}0101", "end": f"{year}1231",
        "latitude": lat, "longitude": lon,
        "community": "re",
        "parameters": ",".join(NASA_PARAMS),
        "time-standard": "utc",
        "format": "json",
    }
    # One HTTP request per year keeps each response small.
    r = requests.get(NASA_URL, params=params, timeout=120)
    r.raise_for_status()
    p = r.json()["properties"]["parameter"]
    df = pd.DataFrame(p)
    # Keys look like 2023010112 (YYYYMMDDHH, UTC). Convert to Riyadh local time (UTC+3).
    df.index = (pd.to_datetime(df.index, format="%Y%m%d%H")
                .tz_localize("UTC").tz_convert("Asia/Riyadh").tz_localize(None))
    # NASA marks missing values as -999.
    df = df.rename(columns=NASA_PARAMS).replace(-999, pd.NA)
    return df


def get_pvgis_power(lat, lon, start, end, peak_kw, loss_pct):
    """Simulated hourly PV output from PVGIS, converted to Asia/Riyadh and floored to the hour."""
    # pvlib is only needed here, so it is imported inside the function.
    import pvlib

    last_err = None
    # Prefer the satellite database SARAH3; fall back to the ERA5 reanalysis if it is unavailable.
    for db in ("PVGIS-SARAH3", "PVGIS-ERA5"):
        try:
            result = pvlib.iotools.get_pvgis_hourly(
                lat, lon, start=start, end=end, raddatabase=db,
                pvcalculation=True, peakpower=peak_kw, loss=loss_pct,
                optimal_surface_tilt=True, optimalangles=True,
                components=False, timeout=120,
            )
            data = result[0]  # pvlib >=0.13 returns (data, meta); older returns (data, inputs, meta)
            # PVGIS timestamps are a few minutes past the hour (UTC); align them to whole local hours.
            data.index = data.index.tz_convert("Asia/Riyadh").tz_localize(None).floor("h")
            # P is the simulated AC output in watts; convert to kW.
            out = pd.DataFrame({"actual_kw": data["P"] / 1000.0})
            out["pvgis_db"] = db
            return out
        except Exception as e:  # try the next database
            last_err = e
    raise RuntimeError(f"PVGIS failed for both databases: {last_err}")


def main():
    """Download each requested city and save one CSV per city."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", default="riyadh", help="comma list or 'all'")
    ap.add_argument("--start", type=int, default=2019)
    ap.add_argument("--end", type=int, default=2023, help="PVGIS currently ends at 2023")
    ap.add_argument("--peak-kw", type=float, default=5.0, help="PV system size, kWp")
    ap.add_argument("--loss", type=float, default=14.0, help="system losses, %%")
    ap.add_argument("--no-pv", action="store_true", help="skip PVGIS (weather/irradiance only)")
    args = ap.parse_args()

    names = list(CITIES) if args.cities == "all" else [c.strip().lower() for c in args.cities.split(",")]
    os.makedirs("data/raw", exist_ok=True)

    for name in names:
        lat, lon = CITIES[name]
        print(f"[{name}] NASA POWER {args.start}-{args.end} ...")
        frames = []
        for y in range(args.start, args.end + 1):
            frames.append(get_nasa_power(lat, lon, y))
            time.sleep(1)  # be polite to the API
        # Join all years, then add the simulated PV output for the same hours.
        df = pd.concat(frames).sort_index()

        if not args.no_pv:
            print(f"[{name}] PVGIS PV simulation ...")
            pv = get_pvgis_power(lat, lon, args.start, args.end, args.peak_kw, args.loss)
            df = df.join(pv, how="left")

        df.index.name = "timestamp"
        df.insert(0, "city", name)
        path = f"data/raw/saudi_solar_{name}_{args.start}_{args.end}.csv"
        df.to_csv(path)
        print(f"[{name}] saved {len(df):,} rows -> {path}")


if __name__ == "__main__":
    main()
