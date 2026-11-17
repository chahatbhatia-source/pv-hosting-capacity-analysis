"""
real_data.py
------------
Real (measurement-based) load and PV profiles, as a drop-in
replacement for the synthetic curves in `profiles.py`.

Three sources are supported:

1. **SimBench** (default, works offline once `simbench` is installed)
   The SimBench project (Meinecke et al., *Energies* 13(12), 2020)
   publishes one full year (2016, 15-min resolution) of load and
   generation profiles for German distribution grids, derived from
   measured data. We use:
     * ``lv_semiurb4``  - aggregated semi-urban LV-grid demand, used for
                          the CIGRE *residential* loads (``Load R*``),
     * ``G0-A``         - standard commercial (BDEW G0) profile, used for
                          the CIGRE *commercial/industrial* loads
                          (``Load CI*``),
     * ``PV3``          - an MV-connected PV plant feed-in profile.

2. **PVGIS** (`fetch_pvgis_pv`) - hourly PV output for *any* location
   from the EU Joint Research Centre's PVGIS API (satellite irradiance
   + PV system model). Needs internet access to re.jrc.ec.europa.eu.
   Results are cached to ``data/`` as CSV.

3. **Your own CSV** (`load_csv_series`) - e.g. smart-meter data from a
   DSO open-data portal, or a PVGIS file downloaded by hand.

Normalization conventions (important for interpreting results):

* Load profiles are normalized to their **annual peak = 1.0**, because
  CIGRE load ratings (`p_mw`) are peak demands.
* PV profiles are expressed as **fraction of installed DC/AC
  nameplate** and are NOT re-normalized: real PV rarely exceeds
  ~65-80 % of nameplate at the grid connection point (inverter losses,
  temperature, orientation). Re-scaling to a peak of 1.0, as the
  synthetic curve does, overstates feed-in and therefore understates
  hosting capacity.

All functions return pandas objects on a timezone-naive DatetimeIndex.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SIMBENCH_CACHE = os.path.join(DATA_DIR, "simbench_2016_profiles.csv")

SIMBENCH_LOAD_GRID = "1-MV-urban--0-sw"     # has lv_semiurb4 + G0-A load profiles
SIMBENCH_PV_GRID = "1-MV-semiurb--0-sw"     # has the PV3 plant profile
SIMBENCH_RESIDENTIAL = "lv_semiurb4"
SIMBENCH_COMMERCIAL = "G0-A"
SIMBENCH_PV = "PV3"


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_simbench_profiles(use_cache: bool = True, cache_path: str = SIMBENCH_CACHE
                           ) -> pd.DataFrame:
    """Return a 15-min, full-year DataFrame with columns
    ``residential``, ``commercial`` (normalized to annual peak = 1) and
    ``pv`` (fraction of nameplate).

    The first call extracts the profiles from the `simbench` package and
    writes a small CSV cache to ``data/``; later calls (and users without
    `simbench` installed) read that cache.
    """
    if use_cache and os.path.exists(cache_path):
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        df.index.name = "time"
        if df.index.is_monotonic_increasing and df.index.is_unique:
            return df
        # stale cache from an older version -> rebuild below

    try:
        import simbench as sb
    except ImportError as e:  # pragma: no cover - depends on environment
        raise ImportError(
            "SimBench profiles are not cached and the `simbench` package is not "
            "installed. Run `pip install simbench` once to create the cache."
        ) from e

    load = sb.get_simbench_net(SIMBENCH_LOAD_GRID).profiles["load"]
    ren = sb.get_simbench_net(SIMBENCH_PV_GRID).profiles["renewables"]
    assert (load["time"].to_numpy() == ren["time"].to_numpy()).all()
    # SimBench time stamps are German local time WITH daylight-saving
    # shifts (27 Mar has 92 steps, 30 Oct has 100, 02:00-02:45 repeated).
    # The samples themselves are consecutive 15-min intervals, so we
    # re-label them on a regular CET (UTC+1, no DST) grid: monotonic,
    # 96 steps every day, and solar noon stays near 12:30 all year.
    first = pd.to_datetime(load["time"].iloc[0], format="%d.%m.%Y %H:%M")
    time = pd.date_range(first, periods=len(load), freq="15min")

    res = load[f"{SIMBENCH_RESIDENTIAL}_pload"].to_numpy(dtype=float)
    com = load[f"{SIMBENCH_COMMERCIAL}_pload"].to_numpy(dtype=float)
    pv = ren[SIMBENCH_PV].to_numpy(dtype=float)

    df = pd.DataFrame({
        "residential": res / res.max(),
        "commercial": com / com.max(),
        "pv": np.clip(pv, 0, 1),
    }, index=pd.DatetimeIndex(time, name="time"))

    if use_cache:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df.round(5).to_csv(cache_path)
    return df


def fetch_pvgis_pv(lat: float, lon: float, year: int = 2020, tilt: float = 35.0,
                   azimuth: float = 0.0, loss_pct: float = 14.0,
                   cache_dir: str = DATA_DIR) -> pd.Series:
    """Hourly PV output (fraction of 1 kWp nameplate) from PVGIS.

    ``azimuth`` follows the PVGIS convention: 0 = south, -90 = east,
    90 = west. Requires internet access; the result is cached as
    ``data/pvgis_<lat>_<lon>_<year>.csv`` so later runs are offline.
    """
    fname = os.path.join(cache_dir, f"pvgis_{lat:.3f}_{lon:.3f}_{year}.csv")
    if os.path.exists(fname):
        return load_csv_series(fname, value_col="pv")

    import json
    import urllib.parse
    import urllib.request

    params = dict(lat=lat, lon=lon, startyear=year, endyear=year, pvcalculation=1,
                  peakpower=1, loss=loss_pct, angle=tilt, aspect=azimuth,
                  outputformat="json")
    url = "https://re.jrc.ec.europa.eu/api/v5_2/seriescalc?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=60) as r:
        payload = json.load(r)

    rows = payload["outputs"]["hourly"]
    # PVGIS timestamps look like "20200101:0010" (UTC, minute = sample offset)
    idx = pd.to_datetime([row["time"] for row in rows], format="%Y%m%d:%H%M").floor("h")
    p_kw = np.array([row["P"] for row in rows], dtype=float) / 1000.0  # W per kWp -> p.u.
    s = pd.Series(np.clip(p_kw, 0, 1), index=pd.DatetimeIndex(idx, name="time"), name="pv")

    os.makedirs(cache_dir, exist_ok=True)
    s.to_frame().to_csv(fname)
    return s


def load_csv_series(path: str, value_col: str | None = None, time_col: str | None = None,
                    normalize: str | None = None, name: str | None = None) -> pd.Series:
    """Read a single time series from a CSV file.

    Parameters
    ----------
    value_col : column holding the values (default: first non-time column)
    time_col  : column holding timestamps (default: first column)
    normalize : ``"peak"`` to divide by the maximum (use for load data in
                kW/MW), ``None`` to keep values as-is (use for PV already
                in p.u. of nameplate).
    """
    df = pd.read_csv(path)
    time_col = time_col or df.columns[0]
    value_col = value_col or [c for c in df.columns if c != time_col][0]
    s = pd.Series(df[value_col].to_numpy(dtype=float),
                  index=pd.DatetimeIndex(pd.to_datetime(df[time_col]), name="time"),
                  name=name or value_col).sort_index()
    s = s[~s.index.duplicated()].interpolate(limit=4).fillna(0.0)
    if normalize == "peak":
        s = s / s.max()
    return s


def replace_pv(load_df: pd.DataFrame, pv: pd.Series) -> pd.DataFrame:
    """Swap the PV column of a SimBench-style frame for another PV series
    (e.g. PVGIS for your own location), matching on calendar month/day/
    time-of-day so that different years line up. Days missing from `pv`
    (e.g. 29 Feb) are filled from the nearest available day."""
    if pv.index[1] - pv.index[0] > pd.Timedelta("15min"):
        # hourly interval averages -> 15-min steps (forward fill keeps energy)
        last = pv.index[-1] + (pv.index[1] - pv.index[0]) - pd.Timedelta("15min")
        pv = pv.reindex(pd.date_range(pv.index[0], last, freq="15min")).ffill()
    key = lambda idx: idx.strftime("%m-%d %H:%M")  # noqa: E731
    lookup = pd.Series(pv.to_numpy(), index=key(pv.index))
    lookup = lookup[~lookup.index.duplicated()]
    out = load_df.copy()
    out["pv"] = lookup.reindex(key(out.index)).to_numpy()
    out["pv"] = out["pv"].ffill().bfill().clip(0, 1)
    return out


# --------------------------------------------------------------------------
# Preparation / day selection
# --------------------------------------------------------------------------
def resample(df: pd.DataFrame | pd.Series, steps_per_day: int) -> pd.DataFrame | pd.Series:
    """Resample to ``steps_per_day`` (24 = hourly, 96 = 15-min) using the
    interval mean, which preserves energy."""
    minutes = 24 * 60 // steps_per_day
    return df.resample(f"{minutes}min").mean().dropna(how="all")


def aggregate_load(df: pd.DataFrame, net=None) -> pd.Series:
    """Feeder-total normalized load, weighting residential/commercial by
    the CIGRE nominal ratings (or 50/50 if no network is given)."""
    if net is None:
        w_res = w_com = 0.5
    else:
        is_res = net.load.name.str.contains("Load R")
        p = net.load["p_mw_nominal"] if "p_mw_nominal" in net.load else net.load["p_mw"]
        w_res, w_com = p[is_res].sum(), p[~is_res].sum()
    tot = (w_res * df["residential"] + w_com * df["commercial"]) / (w_res + w_com)
    return tot.rename("load_norm")


def daily_stress(df: pd.DataFrame, net=None) -> pd.Series:
    """Per-day maximum of PV / feeder load - the reverse-power-flow stress
    indicator used to pick the worst-case hosting-capacity day."""
    ratio = df["pv"] / aggregate_load(df, net).clip(lower=1e-3)
    return ratio.groupby(ratio.index.date).max()


@dataclass
class DayProfiles:
    """One day of real data, in the shape `run_day` expects."""
    label: str
    date: pd.Timestamp
    load: pd.DataFrame   # columns residential / commercial
    pv: pd.Series

    @property
    def pv_energy_pu_h(self) -> float:
        step_h = (self.pv.index[1] - self.pv.index[0]).total_seconds() / 3600
        return float(self.pv.sum() * step_h)


def extract_day(df: pd.DataFrame, date, steps_per_day: int = 24, label: str = "") -> DayProfiles:
    date = pd.Timestamp(date).normalize()
    day = resample(df.loc[str(date.date())], steps_per_day)
    if len(day) != steps_per_day:
        raise ValueError(f"{date.date()}: expected {steps_per_day} steps, got {len(day)}")
    return DayProfiles(label or str(date.date()), date,
                       day[["residential", "commercial"]], day["pv"].rename("pv_norm"))


def representative_days(df: pd.DataFrame, net=None) -> dict[str, pd.Timestamp]:
    """Pick the study days used in the paper-style analysis:

    * ``worst_case``  - day with the highest PV/load ratio (hosting-capacity
                        design day),
    * ``summer_peak`` - day with the highest PV energy,
    * ``typical_summer`` / ``typical_winter`` - the day whose PV energy is
                        closest to the median of June-August / Dec-Feb.
    """
    pv_daily = df["pv"].groupby(df.index.date).sum()
    pv_daily.index = pd.to_datetime(pv_daily.index)
    stress = daily_stress(df, net)
    stress.index = pd.to_datetime(stress.index)

    def closest_to_median(months):
        s = pv_daily[pv_daily.index.month.isin(months)]
        return (s - s.median()).abs().idxmin()

    return {
        "worst_case": stress.idxmax(),
        "summer_peak": pv_daily.idxmax(),
        "typical_summer": closest_to_median([6, 7, 8]),
        "typical_winter": closest_to_median([12, 1, 2]),
    }


def monthly_typical_days(df: pd.DataFrame) -> dict[int, pd.Timestamp]:
    """For each month, the day whose PV energy is closest to that month's
    median - used to approximate annual curtailment from 12 simulated days."""
    pv_daily = df["pv"].groupby(df.index.date).sum()
    pv_daily.index = pd.to_datetime(pv_daily.index)
    out = {}
    for m in range(1, 13):
        s = pv_daily[pv_daily.index.month == m]
        out[m] = (s - s.median()).abs().idxmin()
    return out


def profile_statistics(df: pd.DataFrame, net=None) -> pd.Series:
    """Headline statistics of the dataset, handy for the paper's data section."""
    step_h = (df.index[1] - df.index[0]).total_seconds() / 3600
    load = aggregate_load(df, net)
    return pd.Series({
        "start": df.index[0], "end": df.index[-1], "n_steps": len(df),
        "resolution_min": step_h * 60,
        "pv_full_load_hours": df["pv"].sum() * step_h,
        "pv_capacity_factor_pct": 100 * df["pv"].mean(),
        "pv_peak_pu": df["pv"].max(),
        "load_factor_pct": 100 * load.mean() / load.max(),
        "load_min_over_peak": load.min() / load.max(),
    })


if __name__ == "__main__":
    data = load_simbench_profiles()
    print(profile_statistics(data).to_string())
    for k, d in representative_days(data).items():
        print(f"{k:15s} {d.date()}")
