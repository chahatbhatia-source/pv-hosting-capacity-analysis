"""
annual.py
---------
Full-year (8760 h, or 35 136 x 15 min) quasi-static time-series
simulation, made tractable with *critical time-step screening*.

Why screening?
    With the Q(U) and curtailment controllers active, every time step
    needs ~8 iterated power flows, so a controlled 8760-step year takes
    ~25 min per PV capacity. Most hours, though, are nowhere near a
    limit (night, winter, cloudy days).

How it works (``simulate_year``):
    1. Run the whole year WITHOUT control (one power flow per step, ~2
       min). This gives the uncontrolled voltages/loadings for every
       step.
    2. Mark *critical* steps: PV is producing and the uncontrolled
       maximum voltage exceeds ``screen_v`` (default 1.03 p.u., i.e. a
       0.02 p.u. margin below the 1.05 limit) or line loading exceeds
       ``screen_loading_pct`` (default 90 %).
    3. Re-run ONLY the critical steps with Q(U) (+ curtailment).

    This is exact for curtailment and violations as long as no
    non-critical step would violate a limit under control. That holds
    because (a) both controllers reset their state at every time step
    (``initialize_control``), so steps are independent, and (b) Q(U) only
    *lowers* voltages that are above 1.02 p.u.; it can raise voltages only
    by injecting Q at buses below 0.98 p.u., which cannot lift a feeder
    whose uncontrolled maximum is <= 1.03 p.u. above 1.05 p.u. The
    equivalence is checked against a full controlled simulation in
    ``validate_screening`` and in ``tests/test_annual.py``.

    The per-step results of non-critical steps are the uncontrolled ones
    (Q(U) would only nudge voltages inside the 0.98-1.02 band there).

Hosting-capacity searches over the full year (``zero_violation_capacity``)
use the same idea: the steps that violate at the upper end of the
search interval are a superset of the steps that violate at any lower
capacity, so the bisection only re-simulates those steps.
"""
from __future__ import annotations

import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import network as netmod
from timeseries_sim import run_day, step_hours

V_MAX = 1.05
V_MIN = 0.95
LINE_MAX = 100.0

CONTROL_MODES = ("none", "qu", "qu_curtail")


# --------------------------------------------------------------------------
# Single-capacity full-year simulation
# --------------------------------------------------------------------------
def _fresh_network():
    net = netmod.build_network()
    buses = netmod.candidate_buses(net)
    netmod.attach_sgen_fleet(net, buses)
    return net, buses


def _step_metrics(output) -> pd.DataFrame:
    vm = output["res_bus.vm_pu"]
    line = output["res_line.loading_percent"]
    trafo = output["res_trafo.loading_percent"]
    return pd.DataFrame({
        "vmax": vm.max(axis=1),
        "vmin": vm.min(axis=1),
        "line_max_pct": line.max(axis=1),
        "trafo_max_pct": trafo.max(axis=1),
        "pv_fed_mw": output["sgen.p_mw"].sum(axis=1),
        "q_mvar": output["sgen.q_mvar"].sum(axis=1),
    })


def simulate_year(year: pd.DataFrame, capacity_mw: float, control: str = "qu_curtail",
                  screen_v: float = 1.03, screen_loading_pct: float = 90.0,
                  steps: list[int] | None = None, full_control: bool = False,
                  out_dir: str | None = None) -> pd.DataFrame:
    """Simulate a whole year (or the positional ``steps`` subset of it).

    Parameters
    ----------
    year : DataFrame with columns residential / commercial / pv on a
           regular DatetimeIndex (hourly or 15-min).
    capacity_mw : PV installed at every candidate bus (uniform).
    control : "none", "qu" (Q(U) droop only) or "qu_curtail".
    full_control : if True, skip screening and run the controllers on
           every step (slow; used to validate the screening).

    Returns a per-step DataFrame indexed by timestamp with columns
    vmax, vmin, line_max_pct, trafo_max_pct, pv_fed_mw, q_mvar,
    vmax_noctrl, pv_potential_mw, curtailed_mw, controlled (bool),
    converged (bool).
    """
    if control not in CONTROL_MODES:
        raise ValueError(f"control must be one of {CONTROL_MODES}")
    net, buses = _fresh_network()
    load = year[["residential", "commercial"]]
    pv = year["pv"]
    cap = {b: capacity_mw for b in buses}
    tmp = out_dir or tempfile.mkdtemp(prefix="hc_year_")
    all_steps = list(range(len(year))) if steps is None else list(steps)

    base_out, _ = run_day(net, buses, load, pv, cap, os.path.join(tmp, "base"),
                          time_steps=all_steps, continue_on_divergence=True)
    res = _step_metrics(base_out).reindex(all_steps)
    res["vmax_noctrl"] = res["vmax"]
    res["controlled"] = False

    if control != "none":
        pv_on = pv.to_numpy()[all_steps] > 1e-6
        if full_control:
            critical = [s for s, on in zip(all_steps, pv_on) if on]
        else:
            near = ((res["vmax"] > screen_v) | (res["line_max_pct"] > screen_loading_pct)
                    | res["vmax"].isna()).to_numpy()
            critical = [s for s, on, n in zip(all_steps, pv_on, near) if on and n]
        if critical:
            ctrl_out, _ = run_day(net, buses, load, pv, cap, os.path.join(tmp, "ctrl"),
                                  use_qu_control=True,
                                  use_curtailment=(control == "qu_curtail"),
                                  time_steps=critical, continue_on_divergence=True)
            ctrl = _step_metrics(ctrl_out)
            for col in ctrl.columns:
                res.loc[critical, col] = ctrl[col].to_numpy()
            res.loc[critical, "controlled"] = True

    res["pv_potential_mw"] = pv.to_numpy()[all_steps] * capacity_mw * len(buses)
    res["curtailed_mw"] = (res["pv_potential_mw"] - res["pv_fed_mw"]).clip(lower=0)
    res.loc[res["curtailed_mw"] < 1e-9, "curtailed_mw"] = 0.0
    res["converged"] = res["vmax"].notna()
    res.index = year.index[all_steps]
    res.attrs.update(capacity_mw=capacity_mw, control=control,
                     step_hours=step_hours(pv), n_buses=len(buses))
    return res


def summarize_year(res: pd.DataFrame, v_max: float = V_MAX, v_min: float = V_MIN,
                   line_max: float = LINE_MAX) -> dict:
    """Annual KPIs from a ``simulate_year`` result."""
    h = res.attrs.get("step_hours", 1.0)
    pot = res["pv_potential_mwh"] if "pv_potential_mwh" in res else res["pv_potential_mw"] * h
    cur = res["curtailed_mw"] * h
    return {
        "capacity_mw": res.attrs.get("capacity_mw"),
        "control": res.attrs.get("control"),
        "potential_mwh": float(pot.sum()),
        "curtailed_mwh": float(cur.sum()),
        "curtailed_pct": float(100 * cur.sum() / pot.sum()) if pot.sum() > 0 else 0.0,
        "overvoltage_hours": float((res["vmax"] > v_max + 1e-9).sum() * h),
        "overvoltage_hours_noctrl": float((res["vmax_noctrl"] > v_max + 1e-9).sum() * h),
        "undervoltage_hours": float((res["vmin"] < v_min).sum() * h),
        "overload_hours": float((res["line_max_pct"] > line_max).sum() * h),
        "vmax": float(res["vmax"].max()),
        "vmax_noctrl": float(res["vmax_noctrl"].max()),
        "vmin": float(res["vmin"].min()),
        "max_line_loading_pct": float(res["line_max_pct"].max()),
        "hours_with_curtailment": float((res["curtailed_mw"] > 0).sum() * h),
        "controlled_steps": int(res["controlled"].sum()),
        "nonconverged_steps": int((~res["converged"]).sum()),
    }


def is_compliant(res: pd.DataFrame, v_max=V_MAX, line_max=LINE_MAX) -> bool:
    """No PV-caused violation (over-voltage / overload / divergence)."""
    return bool(res["converged"].all() and (res["vmax"] <= v_max + 1e-9).all()
                and (res["line_max_pct"] <= line_max + 1e-9).all())


# --------------------------------------------------------------------------
# Sweeps and capacity searches
# --------------------------------------------------------------------------
def _sweep_job(args):
    year, cap, control, screen_v = args
    return cap, simulate_year(year, cap, control, screen_v=screen_v)


def annual_sweep(year: pd.DataFrame, capacities, control: str = "qu_curtail",
                 workers: int = 1, screen_v: float = 1.03) -> dict[float, pd.DataFrame]:
    """``simulate_year`` for several capacities, in parallel processes."""
    jobs = [(year, float(c), control, screen_v) for c in capacities]
    if workers <= 1:
        return dict(_sweep_job(j) for j in jobs)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return dict(ex.map(_sweep_job, jobs))


@dataclass
class CapacitySearch:
    capacity_mw: float
    control: str
    evaluations: list = field(default_factory=list)  # (capacity, compliant)
    n_screened_steps: int = 0


def zero_violation_capacity(year: pd.DataFrame, control: str = "none",
                            capacity_max_mw: float = 3.0, tol_mw: float = 0.01,
                            screen_v: float = 1.03) -> CapacitySearch:
    """Largest uniform capacity with ZERO over-voltage / overload hours in
    the whole year, for control "none" or "qu".

    The full year is simulated once (uncontrolled) at ``capacity_max_mw``;
    only steps whose uncontrolled max voltage exceeds ``screen_v`` (or
    which overload / diverge) there can violate at any lower capacity,
    so the bisection re-simulates just those steps.
    """
    if control not in ("none", "qu"):
        raise ValueError("zero-violation search is defined for control 'none' or 'qu'")
    hi_res = simulate_year(year, capacity_max_mw, "none")
    cand_mask = ((hi_res["vmax"] > screen_v) | (hi_res["line_max_pct"] > 90)
                 | ~hi_res["converged"]).to_numpy()
    candidates = list(np.flatnonzero(cand_mask))
    search = CapacitySearch(0.0, control, n_screened_steps=len(candidates))
    if not candidates:
        search.capacity_mw = capacity_max_mw
        return search

    def ok(cap):
        r = simulate_year(year, cap, control, steps=candidates, screen_v=screen_v)
        good = is_compliant(r)
        search.evaluations.append((cap, good))
        return good

    lo, hi = 0.0, capacity_max_mw
    if ok(hi):
        search.capacity_mw = hi
        return search
    while hi - lo > tol_mw:
        mid = 0.5 * (lo + hi)
        if ok(mid):
            lo = mid
        else:
            hi = mid
    search.capacity_mw = lo
    return search


def budget_capacity(summary: pd.DataFrame, budget_pct: float) -> float | None:
    """Interpolate the capacity at which annual curtailment = budget from a
    sweep summary (index = capacity, column curtailed_pct). Returns None
    if the budget is never reached in the swept range."""
    s = summary.sort_index()
    caps = s.index.to_numpy(dtype=float)
    pct = s["curtailed_pct"].to_numpy()
    if pct.max() < budget_pct:
        return None
    above = int(np.argmax(pct >= budget_pct))
    if above == 0:
        return float(caps[0])
    return float(np.interp(budget_pct, pct[above - 1: above + 1], caps[above - 1: above + 1]))


def validate_screening(year: pd.DataFrame, capacity_mw: float, start: str, end: str,
                       screen_v: float = 1.03) -> pd.DataFrame:
    """Compare screened vs. fully-controlled simulation over [start, end]."""
    sub = year.loc[start:end]
    screened = simulate_year(sub, capacity_mw, "qu_curtail", screen_v=screen_v)
    full = simulate_year(sub, capacity_mw, "qu_curtail", full_control=True)
    a, b = summarize_year(screened), summarize_year(full)
    keys = ["curtailed_mwh", "curtailed_pct", "overvoltage_hours", "vmax", "controlled_steps"]
    return pd.DataFrame({"screened": [a[k] for k in keys], "full_control": [b[k] for k in keys]},
                        index=keys)
