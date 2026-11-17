"""
tests/test_control.py
-----------------------
Tests the Q(U) droop characteristic in isolation (no power flow
needed for the pure-function checks) and confirms the curtailment
controller actually reduces active power under a forced violation.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandapower as pp

import network
from control import QUDroopController, CurtailmentController


def _dummy_qu():
    net = network.build_network()
    buses = network.candidate_buses(net)
    network.attach_sgen_fleet(net, buses, p_mw=0.1)
    sgen_idx = list(net.sgen.index)
    return QUDroopController(net, sgen_idx=sgen_idx)


def test_droop_factor_full_injection_below_v1():
    qu = _dummy_qu()
    assert qu._droop_factor(0.90) == 1.0
    assert qu._droop_factor(qu.v1) == 1.0


def test_droop_factor_full_absorption_above_v4():
    qu = _dummy_qu()
    assert qu._droop_factor(1.10) == -1.0
    assert qu._droop_factor(qu.v4) == -1.0


def test_droop_factor_zero_in_deadband():
    qu = _dummy_qu()
    mid_deadband = (qu.v2 + qu.v3) / 2
    assert qu._droop_factor(mid_deadband) == 0.0


def test_droop_factor_is_continuous_no_jump_at_v1():
    """Regression test: the droop curve must ramp, not jump, at the
    v1/v2 and v3/v4 boundaries. A discontinuity here previously caused
    the iterative time-series solver to oscillate forever at buses
    sitting near the boundary (see control.py docstring)."""
    qu = _dummy_qu()
    eps = 1e-4
    just_below = qu._droop_factor(qu.v1 - eps)
    just_above = qu._droop_factor(qu.v1 + eps)
    assert abs(just_below - just_above) < 0.01, "discontinuity at v1"

    just_below = qu._droop_factor(qu.v4 - eps)
    just_above = qu._droop_factor(qu.v4 + eps)
    assert abs(just_below - just_above) < 0.01, "discontinuity at v4"


def test_droop_factor_monotonic_decreasing():
    qu = _dummy_qu()
    vs = np.linspace(0.9, 1.1, 200)
    factors = [qu._droop_factor(v) for v in vs]
    diffs = np.diff(factors)
    assert np.all(diffs <= 1e-9), "droop factor must be monotonically non-increasing in v"


def test_curtailment_reduces_power_under_forced_overvoltage():
    net = network.build_network()
    buses = network.candidate_buses(net)
    network.attach_sgen_fleet(net, buses, p_mw=0.0)

    # push a large PV injection at one bus to force an overvoltage
    stressed_bus = buses[0]
    sgen_idx = net.sgen[net.sgen.bus == stressed_bus].index[0]
    net.sgen.at[sgen_idx, "p_mw"] = 4.0

    ctrl = CurtailmentController(net, sgen_idx=[sgen_idx], p_rated_mw=[4.0],
                                  v_min=0.95, v_max=1.05, step_pct=0.2)
    pp.runpp(net)
    p_before = net.sgen.at[sgen_idx, "p_mw"]

    for _ in range(10):
        if ctrl.is_converged(net):
            break
        ctrl.control_step(net)
        pp.runpp(net)

    p_after = net.sgen.at[sgen_idx, "p_mw"]
    assert p_after <= p_before, "curtailment must not increase active power"
    assert net.res_bus.vm_pu.max() <= 1.05 + 1e-6, "overvoltage should be resolved"


def test_qu_control_converges_on_heavily_loaded_feeder():
    """Regression: with a fixed under-relaxation factor of 0.5 the Q(U)
    iteration entered a period-2 limit cycle (ControllerNotConverged) at
    >= ~1.3 MW/bus, because the local voltage->Q loop gain reached ~3.
    Adaptive damping must make every case converge."""
    import pandas as pd
    from timeseries_sim import run_day

    net = network.build_network()
    buses = network.candidate_buses(net)
    network.attach_sgen_fleet(net, buses)
    t = pd.date_range("2026-06-21", periods=24, freq="h")
    pv = pd.Series(np.where((t.hour >= 11) & (t.hour < 15), 0.9, 0.0), index=t)
    load = pd.Series(0.3, index=t)
    for cap in (1.3, 2.5):
        out, _ = run_day(net, buses, load, pv, {b: cap for b in buses},
                         "/tmp/hc_qu_conv", use_qu_control=True, use_curtailment=True)
        assert len(out["res_bus.vm_pu"]) == 24
        assert out["res_bus.vm_pu"].max().max() <= 1.05 + 1e-6


def test_curtailment_ignores_undervoltage():
    """Regression: curtailing PV cannot fix an under-voltage (it makes it
    worse), so the controller must not react to v < v_min."""
    net = network.build_network()
    buses = network.candidate_buses(net)
    sgens = network.attach_sgen_fleet(net, buses, p_mw=0.1)
    # at nominal peak load the CIGRE feeder already sags to ~0.936 p.u.
    # while lines stay < 80 % loaded -> pure under-voltage case
    pp.runpp(net)
    assert net.res_bus.vm_pu.min() < 0.95
    assert net.res_bus.vm_pu.max() <= 1.05
    ctrl = CurtailmentController(net, sgen_idx=sgens, p_rated_mw=[0.1] * len(sgens),
                                 v_min=0.95, v_max=1.05)
    assert ctrl.is_converged(net), "under-voltage alone must not trigger curtailment"
