"""
tests/test_real_data.py
-----------------------
Checks the real-data pipeline: the cached SimBench year loads with the
expected shape and normalization, day extraction / resampling preserves
energy, representative days are sensible, external PV series can be
spliced in, and curtailed energy is counted with the correct time-step
length (regression test for a hard-coded 15-min factor).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import pytest

import network as netmod
import real_data
from timeseries_sim import load_multipliers, run_day, step_hours


@pytest.fixture(scope="module")
def year():
    return real_data.load_simbench_profiles()


@pytest.fixture
def net():
    n = netmod.build_network()
    netmod.attach_sgen_fleet(n, netmod.candidate_buses(n))
    return n


def test_simbench_year_shape_and_columns(year):
    assert list(year.columns) == ["residential", "commercial", "pv"]
    assert len(year) == 366 * 96  # 2016 is a leap year, 15-min steps
    assert year.index.is_monotonic_increasing
    assert not year.isna().any().any()


def test_loads_normalized_to_annual_peak_and_pv_in_pu(year):
    assert year["residential"].max() == pytest.approx(1.0, abs=1e-4)
    assert year["commercial"].max() == pytest.approx(1.0, abs=1e-4)
    # real PV never reaches nameplate at the grid connection point
    assert 0.4 < year["pv"].max() < 0.95
    assert year["pv"].min() >= 0.0


def test_pv_is_seasonal_and_zero_at_night(year):
    monthly = year["pv"].groupby(year.index.month).mean()
    assert monthly[6] > 3 * monthly[12]
    assert year["pv"][year.index.hour.isin([0, 1, 2, 23])].max() < 0.01


def test_extract_day_resampling_preserves_energy(year):
    date = "2016-06-21"
    hourly = real_data.extract_day(year, date, steps_per_day=24)
    fine = real_data.extract_day(year, date, steps_per_day=96)
    assert len(hourly.pv) == 24 and len(fine.pv) == 96
    assert hourly.pv_energy_pu_h == pytest.approx(fine.pv_energy_pu_h, rel=1e-6)
    assert step_hours(hourly.pv) == 1.0 and step_hours(fine.pv) == 0.25


def test_representative_days(year, net):
    days = real_data.representative_days(year, net)
    assert set(days) == {"worst_case", "summer_peak", "typical_summer", "typical_winter"}
    assert days["typical_winter"].month in (12, 1, 2)
    assert days["typical_summer"].month in (6, 7, 8)
    assert 3 <= days["worst_case"].month <= 9  # high-PV season
    assert len(real_data.monthly_typical_days(year)) == 12


def test_load_multipliers_map_residential_and_commercial(year, net):
    day = real_data.extract_day(year, "2016-03-01")
    mult = load_multipliers(net, day.load)
    is_res = net.load.name.str.contains("Load R")
    i_res, i_com = is_res[is_res].index[0], is_res[~is_res].index[0]
    assert np.allclose(mult[i_res], day.load["residential"].to_numpy())
    assert np.allclose(mult[i_com], day.load["commercial"].to_numpy())
    # a plain Series still applies the same curve to every load
    mult_s = load_multipliers(net, day.load["residential"])
    assert np.allclose(mult_s[i_com], day.load["residential"].to_numpy())


def test_replace_pv_aligns_by_calendar_day(year):
    idx = pd.date_range("2021-01-01", "2021-12-31 23:00", freq="h")
    fake = pd.Series(np.where(idx.month == 6, 0.5, 0.1), index=idx, name="pv")
    out = real_data.replace_pv(year, fake)
    assert len(out) == len(year)
    assert out.loc["2016-06-15", "pv"].eq(0.5).all()
    assert out.loc["2016-02-29", "pv"].notna().all()  # leap day filled


def test_load_csv_series_peak_normalization(tmp_path):
    p = tmp_path / "meter.csv"
    pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=4, freq="h"),
                  "kw": [2.0, 4.0, 8.0, 6.0]}).to_csv(p, index=False)
    s = real_data.load_csv_series(str(p), normalize="peak")
    assert s.max() == 1.0 and s.iloc[0] == 0.25


def test_curtailed_energy_uses_actual_step_length(net):
    """Regression: CurtailmentController used to multiply by 15/60 h even
    for hourly profiles, under-reporting curtailed MWh by 4x. Curtailed
    energy for the same day must not depend on the resolution chosen."""
    buses = netmod.candidate_buses(net)
    t24 = pd.date_range("2026-06-21", periods=24, freq="h")
    t96 = pd.date_range("2026-06-21", periods=96, freq="15min")
    load24 = pd.Series(0.3, index=t24)
    load96 = pd.Series(0.3, index=t96)
    hours = np.arange(96) / 4
    pv96 = pd.Series(np.where((hours >= 11) & (hours < 15), 0.9, 0.0), index=t96)
    pv24 = pv96.resample("h").mean()

    cap = {b: 2.5 for b in buses}  # far above hosting capacity -> curtailment
    _, c24 = run_day(net, buses, load24, pv24, cap, "/tmp/hc_test24",
                     use_qu_control=True, use_curtailment=True)
    _, c96 = run_day(net, buses, load96, pv96, cap, "/tmp/hc_test96",
                     use_qu_control=True, use_curtailment=True)
    assert c24.step_hours == 1.0 and c96.step_hours == 0.25
    assert c24.energy_curtailed_mwh > 0
    assert c24.energy_curtailed_mwh == pytest.approx(c96.energy_curtailed_mwh, rel=0.05)
