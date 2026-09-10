"""
profiles.py
-----------
Generates synthetic but physically-plausible 24h profiles:

  * load_profile   : normalized residential demand curve (morning +
                      evening peaks, midday trough), scaled per-bus
                      by each load's nominal p_mw.
  * pv_profile      : bell-shaped clear-sky solar irradiance curve
                      (zero at night, peak at solar noon), with
                      optional cloud-transient noise.
  * wind_profile    : autocorrelated stochastic wind availability
                      curve (Ornstein-Uhlenbeck-like), independent of
                      time of day.

All profiles are returned as pandas Series indexed by a DatetimeIndex
spanning one day, normalized to [0, 1] so they can be multiplied by
any installed capacity (MW). Resolution defaults to hourly (24 steps)
for fast iteration; pass steps_per_day=96 for 15-minute detail.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STEPS_PER_DAY = 24  # hourly resolution by default (use 96 for 15-min detail)


def _time_index(steps_per_day: int) -> pd.DatetimeIndex:
    minutes = 24 * 60 // steps_per_day
    return pd.date_range("2026-06-21 00:00", periods=steps_per_day, freq=f"{minutes}min")


def load_profile(seed: int | None = 0, steps_per_day: int = STEPS_PER_DAY) -> pd.Series:
    """Normalized residential load curve, double-peaked (0-1 range)."""
    idx = _time_index(steps_per_day)
    t = np.linspace(0, 24, steps_per_day, endpoint=False)
    morning = 0.55 * np.exp(-((t - 8.0) ** 2) / (2 * 1.3 ** 2))
    evening = 1.00 * np.exp(-((t - 19.5) ** 2) / (2 * 1.8 ** 2))
    base = 0.28
    curve = base + morning + evening
    curve = curve / curve.max()
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 0.015, size=steps_per_day) if seed is not None else 0
    curve = np.clip(curve + noise, 0.05, 1.0)
    return pd.Series(curve, index=idx, name="load_norm")


def pv_profile(seed: int | None = 1, cloudiness: float = 0.08,
                steps_per_day: int = STEPS_PER_DAY) -> pd.Series:
    """Clear-sky-like PV output curve with mild cloud-transient noise (0-1)."""
    idx = _time_index(steps_per_day)
    t = np.linspace(0, 24, steps_per_day, endpoint=False)
    sunrise, sunset, solar_noon = 5.5, 20.5, 13.0
    daylight = (t > sunrise) & (t < sunset)
    curve = np.zeros(steps_per_day)
    day_t = t[daylight]
    half_span = (sunset - sunrise) / 2
    curve[daylight] = np.clip(
        np.cos((day_t - solar_noon) / half_span * (np.pi / 2)), 0, None
    ) ** 1.3
    if seed is not None:
        rng = np.random.default_rng(seed)
        clouds = rng.normal(1.0, cloudiness, size=steps_per_day)
        clouds = np.clip(clouds, 0.4, 1.05)
        curve = curve * clouds
    curve = np.clip(curve, 0, 1)
    return pd.Series(curve, index=idx, name="pv_norm")


def wind_profile(seed: int | None = 2, mean: float = 0.45, theta: float = 0.15,
                  sigma: float = 0.09, steps_per_day: int = STEPS_PER_DAY) -> pd.Series:
    """Stochastic, autocorrelated wind availability curve (0-1)."""
    idx = _time_index(steps_per_day)
    rng = np.random.default_rng(seed)
    x = np.zeros(steps_per_day)
    x[0] = mean
    for i in range(1, steps_per_day):
        x[i] = x[i - 1] + theta * (mean - x[i - 1]) + sigma * rng.normal()
    x = np.clip(x, 0, 1)
    return pd.Series(x, index=idx, name="wind_norm")


def build_profile_set(seed: int = 0, steps_per_day: int = STEPS_PER_DAY) -> dict[str, pd.Series]:
    return {
        "load": load_profile(seed=seed, steps_per_day=steps_per_day),
        "pv": pv_profile(seed=seed + 1, steps_per_day=steps_per_day),
        "wind": wind_profile(seed=seed + 2, steps_per_day=steps_per_day),
    }


if __name__ == "__main__":
    profiles = build_profile_set()
    for name, s in profiles.items():
        print(f"{name:6s}  min={s.min():.2f}  max={s.max():.2f}  mean={s.mean():.2f}")
