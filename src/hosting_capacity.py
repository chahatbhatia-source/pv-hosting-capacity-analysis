"""
hosting_capacity.py
--------------------
Two complementary hosting-capacity search algorithms:

1. `per_bus_snapshot_capacity`
   Classic fast screening approach (as used in the pandapower/CIGRE
   literature): for every candidate bus, bisection-search the maximum
   PV capacity that can be connected *at that bus alone* without a
   voltage or thermal violation, evaluated at the single worst-case
   operating snapshot (maximum PV output coincident with minimum
   demand). This produces a per-bus hosting-capacity map in seconds.

2. `global_time_series_capacity`
   A stricter, slower search: assume PV is installed at every
   candidate bus simultaneously (uniform capacity), and bisection
   search the largest per-bus capacity for which a full 24h
   time-series simulation has zero violations. Can optionally engage
   the Q(U) droop / curtailment controllers from `control.py`, so it
   directly quantifies how much extra hosting capacity active DER
   control is worth.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pandapower as pp

import network as netmod
from timeseries_sim import run_day, summarize_violations, load_multipliers, step_hours


def _feeder_load_curve(load_profile, net=None) -> np.ndarray:
    """Feeder-total normalized load (handles Series or res/com DataFrame)."""
    if isinstance(load_profile, pd.Series) or net is None:
        if isinstance(load_profile, pd.DataFrame):
            return load_profile.mean(axis=1).to_numpy()
        return load_profile.to_numpy()
    mult = load_multipliers(net, load_profile)
    p = net.load["p_mw_nominal"]
    return (mult * p).sum(axis=1).to_numpy() / p.sum()


def _worst_case_step(load_profile, pv_profile, net=None) -> int:
    """Index of the time step with the highest PV / lowest load ratio
    -- the classic reverse-power-flow stress point for hosting capacity.
    """
    ratio = pv_profile.to_numpy() / np.clip(_feeder_load_curve(load_profile, net), 1e-3, None)
    return int(np.argmax(ratio))


def _violates(net, v_min, v_max, line_max_pct) -> bool:
    # (v_min kept in the signature for API compatibility; at the worst-case
    # max-PV/min-load snapshot under-voltage is not caused by PV)
    if net.res_bus.vm_pu.isna().any():
        return True  # non-convergence counts as a violation
    if net.res_bus.vm_pu.max() > v_max:
        return True
    if len(net.res_line) and net.res_line.loading_percent.max() > line_max_pct:
        return True
    if len(net.res_trafo) and net.res_trafo.loading_percent.max() > line_max_pct:
        return True
    return False


def per_bus_snapshot_capacity(net, buses, load_profile, pv_profile,
                               v_min=0.95, v_max=1.05, line_max_pct=100.0,
                               capacity_max_mw=5.0, tol_mw=0.01) -> pd.Series:
    """Bisection-search the standalone hosting capacity of every bus.

    Returns a pandas Series indexed by bus id (MW).
    """
    w = _worst_case_step(load_profile, pv_profile, net)
    # per-load multiplier at the worst-case step (Series aligned to net.load)
    load_scale = load_multipliers(net, load_profile).iloc[w]
    pv_scale = pv_profile.to_numpy()[w]

    # Always scale from the frozen nominal ratings (see network.build_network),
    # never from net.load.p_mw/q_mvar directly -- those may already have been
    # overwritten by a prior time-series run reusing this same net object.
    base_load = net.load[["p_mw_nominal", "q_mvar_nominal"]].rename(
        columns={"p_mw_nominal": "p_mw", "q_mvar_nominal": "q_mvar"}
    )
    sgen_for_bus = {net.sgen.at[i, "bus"]: i for i in net.sgen.index}

    results = {}
    for b in buses:
        net.load["p_mw"] = base_load["p_mw"] * load_scale
        net.load["q_mvar"] = base_load["q_mvar"] * load_scale
        netmod.reset_sgens(net, 0.0)

        sgen_idx = sgen_for_bus[b]
        lo, hi = 0.0, capacity_max_mw
        # make sure hi is actually infeasible before bisecting
        net.sgen.at[sgen_idx, "p_mw"] = hi * pv_scale
        try:
            pp.runpp(net)
            hi_feasible = not _violates(net, v_min, v_max, line_max_pct)
        except Exception:
            hi_feasible = False
        if hi_feasible:
            results[b] = hi
            continue

        while hi - lo > tol_mw:
            mid = (lo + hi) / 2
            net.sgen.at[sgen_idx, "p_mw"] = mid * pv_scale
            try:
                pp.runpp(net)
                ok = not _violates(net, v_min, v_max, line_max_pct)
            except Exception:
                ok = False
            if ok:
                lo = mid
            else:
                hi = mid
        results[b] = lo

    net.load["p_mw"] = base_load["p_mw"]
    net.load["q_mvar"] = base_load["q_mvar"]
    netmod.reset_sgens(net, 0.0)
    return pd.Series(results, name="hosting_capacity_mw").sort_index()


def global_time_series_capacity(net, buses, load_profile, pv_profile, out_dir,
                                 v_min=0.95, v_max=1.05, line_max_pct=100.0,
                                 use_qu_control=False, use_curtailment=False,
                                 capacity_max_mw=1.2, tol_mw=0.02):
    """Bisection-search the largest *uniform* per-bus PV capacity for
    which a full 24h time-series run has zero violations.

    Returns (capacity_mw, last_output) where `last_output` is the
    OutputWriter result dict for the feasible capacity found.
    """
    lo, hi = 0.0, capacity_max_mw
    best_output = None

    def feasible(cap):
        pv_cap = {b: cap for b in buses}
        out, _ = run_day(net, buses, load_profile, pv_profile, pv_cap, out_dir,
                          use_qu_control=use_qu_control, use_curtailment=use_curtailment,
                          v_min=v_min, v_max=v_max, line_max_pct=line_max_pct)
        summ = summarize_violations(out, v_min, v_max, line_max_pct)
        # PV can only cause OVER-voltage and overloading; pre-existing
        # under-voltage at load peaks (no PV output) is not a hosting-
        # capacity limit and must not make the search return 0 MW.
        ok = summ["n_overvoltage"] == 0 and summ["n_line_overloads"] == 0
        return ok, out

    ok_hi, out_hi = feasible(hi)
    if ok_hi:
        return hi, out_hi

    while hi - lo > tol_mw:
        mid = (lo + hi) / 2
        ok, out = feasible(mid)
        if ok:
            lo = mid
            best_output = out
        else:
            hi = mid

    if best_output is None:
        # even the smallest step still violates -> re-run at lo for reporting
        _, best_output = feasible(max(lo, 0.001))
    return lo, best_output


def capacity_with_curtailment_budget(net, buses, load_profile, pv_profile, out_dir,
                                      v_min=0.95, v_max=1.05, line_max_pct=100.0,
                                      use_qu_control=True, max_curtailment_pct=5.0,
                                      capacity_max_mw=3.0, tol_mw=0.05):
    """Because active curtailment can force *any* nameplate capacity into
    compliance (it simply clips output until violations disappear), a
    zero-violation search against curtailed capacity is degenerate --
    it always says "yes, feasible". The economically meaningful question
    is instead: how much PV can be installed before the *curtailed
    energy fraction* exceeds an acceptable planning threshold (a common
    DSO convention, e.g. 3-5% of annual PV yield)?

    This bisects `capacity_mw` (installed per candidate bus) for the
    largest value with curtailed_energy / potential_energy * 100 <=
    max_curtailment_pct, with Q(U) droop control always engaged
    alongside curtailment (curtailment is the last resort, per
    control.py's controller ordering).
    """
    def curtailment_pct(cap):
        pv_cap = {b: cap for b in buses}
        try:
            out, curt_ctrl = run_day(net, buses, load_profile, pv_profile, pv_cap, out_dir,
                                      use_qu_control=use_qu_control, use_curtailment=True,
                                      v_min=v_min, v_max=v_max, line_max_pct=line_max_pct)
        except Exception:
            # non-convergence at extreme capacity -> treat as fully infeasible
            return 100.0, None, None, None
        potential_mwh = cap * len(buses) * pv_profile.sum() * step_hours(pv_profile)
        curtailed_mwh = curt_ctrl.energy_curtailed_mwh if curt_ctrl else 0.0
        pct = 100 * curtailed_mwh / potential_mwh if potential_mwh > 0 else 0.0
        return pct, out, curtailed_mwh, potential_mwh

    lo, hi = 0.0, capacity_max_mw
    result = None
    while hi - lo > tol_mw:
        mid = (lo + hi) / 2
        pct, out, curt_mwh, pot_mwh = curtailment_pct(mid)
        if pct <= max_curtailment_pct:
            lo = mid
            result = (mid, pct, out, curt_mwh, pot_mwh)
        else:
            hi = mid

    if result is None:
        result = (lo,) + curtailment_pct(max(lo, 0.05))
    return result  # (capacity_mw, curtailed_pct, output, curtailed_mwh, potential_mwh)


if __name__ == "__main__":
    import profiles

    net = netmod.build_network()
    buses = netmod.candidate_buses(net)
    netmod.attach_sgen_fleet(net, buses)
    prof = profiles.build_profile_set()

    hc = per_bus_snapshot_capacity(net, buses, prof["load"], prof["pv"])
    print("Per-bus PV hosting capacity (MW):")
    print(hc)
    print(f"\nTotal (naive sum, buses independent): {hc.sum():.2f} MW")
