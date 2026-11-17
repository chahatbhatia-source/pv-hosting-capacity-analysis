"""
tests/test_annual.py
--------------------
Checks the full-year simulation machinery in annual.py on short windows
(so the suite stays fast): critical-step screening must reproduce a
fully controlled simulation, energy bookkeeping must be consistent, and
the capacity searches must behave monotonically.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import pytest

import annual
import real_data


@pytest.fixture(scope="module")
def year():
    return real_data.resample(real_data.load_simbench_profiles(), 24)


def test_hourly_year_has_8784_steps(year):
    assert len(year) == 366 * 24  # 2016 is a leap year
    assert year.index.freq is not None or year.index.to_series().diff().nunique() == 1


def test_screening_matches_full_control(year):
    """Critical-step screening must give the same curtailed energy and
    peak voltage as running the controllers on every step."""
    val = annual.validate_screening(year, 1.6, "2016-05-24", "2016-05-27")
    assert val.at["curtailed_mwh", "full_control"] > 0, "window must actually curtail"
    assert val.at["curtailed_mwh", "screened"] == pytest.approx(
        val.at["curtailed_mwh", "full_control"], rel=1e-6, abs=1e-6)
    assert val.at["vmax", "screened"] == pytest.approx(val.at["vmax", "full_control"], abs=1e-6)
    assert val.at["controlled_steps", "screened"] < val.at["controlled_steps", "full_control"]


def test_energy_bookkeeping(year):
    res = annual.simulate_year(year.loc["2016-05-24":"2016-05-26"], 1.6, "qu_curtail")
    s = annual.summarize_year(res)
    n_bus = res.attrs["n_buses"]
    expected_pot = year.loc["2016-05-24":"2016-05-26", "pv"].sum() * 1.6 * n_bus
    assert s["potential_mwh"] == pytest.approx(expected_pot, rel=1e-9)
    assert 0 < s["curtailed_mwh"] < s["potential_mwh"]
    assert (res["curtailed_mw"] >= 0).all()
    # no curtailment at night or on steps that were never controlled
    assert (res.loc[~res["controlled"], "curtailed_mw"] == 0).all()
    # control removes all over-voltage that existed without it
    assert s["overvoltage_hours_noctrl"] > 0 and s["overvoltage_hours"] == 0


def test_no_control_mode_has_no_curtailment(year):
    res = annual.simulate_year(year.loc["2016-05-26"], 1.6, "none")
    assert not res["controlled"].any()
    assert res["curtailed_mw"].sum() == 0
    assert (res["vmax"] == res["vmax_noctrl"]).all()


def test_budget_capacity_interpolation():
    summ = pd.DataFrame({"curtailed_pct": [0.0, 1.0, 4.0, 9.0]}, index=[1.0, 1.2, 1.4, 1.6])
    assert annual.budget_capacity(summ, 5.0) == pytest.approx(1.44)
    assert annual.budget_capacity(summ, 20.0) is None


def test_zero_violation_search_on_short_window(year):
    """On one high-PV week, Q(U) must allow at least as much PV as no control,
    and the found capacity must indeed be compliant."""
    week = year.loc["2016-05-22":"2016-05-28"]
    none = annual.zero_violation_capacity(week, "none", capacity_max_mw=2.0, tol_mw=0.05)
    qu = annual.zero_violation_capacity(week, "qu", capacity_max_mw=2.0, tol_mw=0.05)
    assert 0.3 < none.capacity_mw < 2.0
    assert qu.capacity_mw >= none.capacity_mw - 0.05
    assert annual.is_compliant(annual.simulate_year(week, none.capacity_mw, "none"))
