"""
annual_study.py
---------------
Full-year (8760-hour) hosting-capacity study on real data. Complements
run_study.py, which works on single design / typical days.

Steps
  1. load one year of real profiles (SimBench by default) at hourly
     (or, with --fine, 15-min) resolution,
  2. validate the critical-step screening against a fully controlled
     simulation for one week (see annual.py),
  3. simulate the full year with Q(U) + curtailment for a sweep of
     uniform PV capacities (in parallel processes),
  4. interpolate the capacity at which ANNUAL curtailed energy equals the
     planning budget, and verify it with one more full-year run,
  5. bisection-search the zero-violation capacity over the full year
     without control and with Q(U) only,
  6. compare with the design-day and 12-typical-day results of
     run_study.py --data simbench (if present), save tables + figures.

Usage
    python src/annual_study.py                      # SimBench, hourly, ~15-25 min
    python src/annual_study.py --workers 4 --budget-pct 3
    python src/annual_study.py --data pvgis --lat 48.14 --lon 11.58
    python src/annual_study.py --fine               # 15-min (4x slower)

Outputs go to results/annual/.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd

import analysis
import annual
import real_data
from run_study import load_real_dataset

RESULTS_ROOT = os.path.join(os.path.dirname(__file__), "..", "results")
OUT_DIR = os.path.join(RESULTS_ROOT, "annual")
DEFAULT_CAPS = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]


def _zero_violation_job(args):
    year, control, cap_max, screen_v = args
    return control, annual.zero_violation_capacity(year, control, capacity_max_mw=cap_max,
                                                   tol_mw=0.01, screen_v=screen_v)


def _read_optional(path, **kw):
    return pd.read_csv(path, **kw) if os.path.exists(path) else None


def main():
    p = argparse.ArgumentParser(description="Full-year (8760 h) hosting-capacity study")
    p.add_argument("--data", choices=["simbench", "pvgis", "csv"], default="simbench")
    p.add_argument("--lat", type=float, default=52.52)
    p.add_argument("--lon", type=float, default=13.40)
    p.add_argument("--pv-year", type=int, default=2020)
    p.add_argument("--pv-csv")
    p.add_argument("--capacities", type=float, nargs="+", default=DEFAULT_CAPS,
                   help="uniform PV capacities per bus (MW) for the curtailment sweep")
    p.add_argument("--budget-pct", type=float, default=5.0)
    p.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    p.add_argument("--screen-v", type=float, default=1.03,
                   help="critical-step screening threshold on uncontrolled max voltage")
    p.add_argument("--fine", action="store_true", help="15-min resolution (35 136 steps)")
    p.add_argument("--skip-validation", action="store_true")
    args = p.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    steps_per_day = 96 if args.fine else 24

    print("=" * 70)
    print("FULL-YEAR HOSTING CAPACITY STUDY (critical time-step screening)")
    print("=" * 70)
    df = load_real_dataset(args, None)
    year = real_data.resample(df, steps_per_day)
    print(f"Data: {args.data}, {len(year)} steps "
          f"({year.index[0]} .. {year.index[-1]}), workers={args.workers}")

    # --- 1. screening validation -----------------------------------------
    if not args.skip_validation:
        print("\n[1] Validating screening vs. full control (1-7 May, 1.6 MW/bus)...")
        val = annual.validate_screening(year, 1.6, "2016-05-01", "2016-05-07",
                                        screen_v=args.screen_v)
        val.to_csv(os.path.join(OUT_DIR, "screening_validation.csv"))
        print(val.to_string(float_format=lambda v: f"{v:.4f}"))

    # --- 2. full-year sweep ---------------------------------------------
    print(f"\n[2] Full-year Q(U) + curtailment sweep over {args.capacities} MW/bus...")
    sweep = annual.annual_sweep(year, args.capacities, "qu_curtail",
                                workers=args.workers, screen_v=args.screen_v)
    summary = pd.DataFrame([annual.summarize_year(r) for r in sweep.values()]
                           ).set_index("capacity_mw").sort_index()
    summary.round(4).to_csv(os.path.join(OUT_DIR, "annual_sweep_summary.csv"))
    cols = ["curtailed_pct", "curtailed_mwh", "hours_with_curtailment",
            "overvoltage_hours_noctrl", "vmax_noctrl", "vmax", "max_line_loading_pct",
            "controlled_steps", "nonconverged_steps"]
    print(summary[cols].to_string(float_format=lambda v: f"{v:.3f}"))

    # --- 3. annual-budget capacity ---------------------------------------
    cap_budget = annual.budget_capacity(summary, args.budget_pct)
    verify = None
    if cap_budget is None:
        print(f"\n[3] Annual curtailment stays below {args.budget_pct}% up to "
              f"{max(args.capacities)} MW/bus - extend --capacities.")
    else:
        print(f"\n[3] Interpolated annual-budget capacity: {cap_budget:.3f} MW/bus; verifying...")
        verify = annual.simulate_year(year, cap_budget, "qu_curtail", screen_v=args.screen_v)
        v = annual.summarize_year(verify)
        print(f"    full-year run at {cap_budget:.3f} MW/bus -> {v['curtailed_pct']:.2f}% "
              f"curtailed ({v['curtailed_mwh']:.0f} of {v['potential_mwh']:.0f} MWh), "
              f"{v['hours_with_curtailment']:.0f} h with curtailment")
        monthly = pd.DataFrame({
            "potential_mwh": (verify["pv_potential_mw"]).groupby(verify.index.month).sum()
                             * verify.attrs["step_hours"],
            "curtailed_mwh": (verify["curtailed_mw"]).groupby(verify.index.month).sum()
                             * verify.attrs["step_hours"],
        })
        monthly["curtailed_pct"] = 100 * monthly["curtailed_mwh"] / monthly["potential_mwh"]
        monthly.round(3).to_csv(os.path.join(OUT_DIR, "monthly_at_budget_capacity.csv"))
        analysis.plot_curtailment_heatmap(verify, OUT_DIR)
        analysis.plot_monthly_energy(verify, OUT_DIR)

    # --- 4. zero-violation capacities over the full year ------------------
    print("\n[4] Zero-violation capacity over the full year (no control / Q(U))...")
    jobs = [(year, "none", 2.0, args.screen_v), (year, "qu", 2.0, args.screen_v)]
    with ProcessPoolExecutor(max_workers=min(2, args.workers)) as ex:
        zv = dict(ex.map(_zero_violation_job, jobs))
    for k, s in zv.items():
        print(f"    {k:5s}: {s.capacity_mw:.2f} MW/bus "
              f"({s.n_screened_steps} candidate steps, {len(s.evaluations)} evaluations)")

    # --- 5. comparison table + figures -----------------------------------
    n_bus = next(iter(sweep.values())).attrs["n_buses"]
    rows = {
        "no control, zero violations (full year)": zv["none"].capacity_mw,
        "Q(U) only, zero violations (full year)": zv["qu"].capacity_mw,
    }
    if cap_budget is not None:
        rows[f"Q(U) + curtailment, <= {args.budget_pct:g}% annual (full year)"] = cap_budget

    typical = _read_optional(os.path.join(RESULTS_ROOT, "real_data", "annual_curtailment.csv"),
                             index_col=0)
    design = _read_optional(os.path.join(RESULTS_ROOT, "real_data",
                                         "hosting_capacity_summary.csv"), index_col=0)
    if design is not None:
        for label, val in design["capacity_mw_per_bus"].items():
            rows[f"[run_study] {label}"] = val
    table = pd.DataFrame({"capacity_mw_per_bus": pd.Series(rows)})
    table["feeder_total_mw"] = table["capacity_mw_per_bus"] * n_bus
    table.round(3).to_csv(os.path.join(OUT_DIR, "hosting_capacity_comparison.csv"))
    print("\nHosting-capacity comparison:\n" + table.round(2).to_string())

    analysis.plot_annual_curtailment_curve(
        summary, OUT_DIR, args.budget_pct,
        typical_days=None if typical is None else typical["annual_curtailed_pct"],
        budget_cap=cap_budget)
    dur = {c: sweep[c] for c in sorted(sweep)[::2]}
    analysis.plot_voltage_duration_curves(dur, OUT_DIR)

    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Outputs: {os.path.abspath(OUT_DIR)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
