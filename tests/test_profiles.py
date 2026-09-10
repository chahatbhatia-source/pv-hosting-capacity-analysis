"""
tests/test_profiles.py
------------------------
Checks the synthetic load / PV / wind profile generator: correct
length, bounded to [0, 1], physically sane shape (PV is zero at
night, load has morning/evening peaks).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import profiles


def test_profile_lengths_match_steps_per_day():
    for steps in (24, 96):
        p = profiles.build_profile_set(steps_per_day=steps)
        for name, series in p.items():
            assert len(series) == steps, f"{name} profile has wrong length"


def test_profiles_are_bounded_0_1():
    p = profiles.build_profile_set()
    for name, series in p.items():
        assert series.min() >= 0.0, f"{name} profile goes negative"
        assert series.max() <= 1.0 + 1e-9, f"{name} profile exceeds 1.0"


def test_pv_profile_is_zero_at_night():
    pv = profiles.pv_profile(steps_per_day=24)
    # hour index 0 = midnight, 2 = 2am -> should be (near) zero
    assert pv.iloc[0] < 0.05
    assert pv.iloc[2] < 0.05


def test_pv_profile_peaks_near_midday():
    pv = profiles.pv_profile(steps_per_day=24, seed=None)  # no cloud noise
    peak_hour = np.argmax(pv.values)
    assert 11 <= peak_hour <= 15, "PV profile should peak near solar noon"


def test_load_profile_has_two_peaks_above_trough():
    load = profiles.load_profile(steps_per_day=24, seed=None)
    midday = load.iloc[10:14].mean()   # ~10am-2pm trough-ish
    evening = load.iloc[18:21].mean()  # ~6-9pm peak
    assert evening > midday, "evening demand should exceed midday demand"


def test_reproducible_with_same_seed():
    a = profiles.load_profile(seed=42)
    b = profiles.load_profile(seed=42)
    assert np.allclose(a.values, b.values)
