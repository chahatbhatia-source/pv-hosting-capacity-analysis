"""
run_study.py
------------
End-to-end entry point. Running this script:

  1. builds the CIGRE MV benchmark network,
  2. loads 24h load / PV profiles - synthetic, or a design day picked
     from a year of REAL data (see real_data.py),
  3. computes the per-bus standalone PV hosting capacity map,
  4. runs a 24h time-series simulation with NO control (baseline),
  5. runs the same scenario WITH Q(U) droop + curtailment control,
  6. bisection-searches the installed capacity that keeps curtailed
     energy under a planning budget on the design day,
  and, with real data only:
  7. repeats the with/without-control comparison on four
     representative days (worst case, summer peak, typical summer,
     typical winter),
  8. estimates ANNUAL curtailment for several capacities from one
     typical day per month, and the capacity that meets the budget on
     an annual basis (how DSOs actually define the budget).

Usage
-----
    python src/run_study.py                               # synthetic profiles
    python src/run_study.py --data simbench               # real data (SimBench 2016)
    python src/run_study.py --data pvgis --lat 52.52 --lon 13.40 --pv-year 2020
    python src/run_study.py --data csv --pv-csv data/my_pv.csv

Options
    --capacity-mw   PV capacity (MW) per candidate bus for the
                     baseline-vs-control comparison (default 0.7)
    --day            design day for real data: worst_case (default),
                     summer_peak, typical_summer, typical_winter or a
                     date such as 2016-06-21
    --fine           15-minute resolution (96 steps) instead of hourly
    --budget-pct     curtailment planning budget, % of potential yield
    --skip-annual    real data: skip steps 7-8 (faster)
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

import network as netmod
import profiles as profmod
import real_data
import analysis
from timeseries_sim import run_day, summarize_violations, step_hours
from hosting_capacity import (per_bus_snapshot_capacity, capacity_with_curtailment_budget,
                              global_time_series_capacity)

RESULTS_ROOT = os.path.join(os.path.dirname(__file__), "..", "results")
TMP = os.path.join(os.environ.get("TMPDIR", "/tmp"), "hc_runs")


def curtailment_pct(net, buses, load, pv, cap_mw, use_qu=True):
    """Run one controlled day; return (curtailed %, curtailed MWh, potential MWh).

    If the controllers fail to converge (only happens far beyond the
    hosting capacity), the day is counted conservatively as 100 %
    curtailed so it cannot silently drop out of an annual total.
    """
    potential = cap_mw * len(buses) * pv.sum() * step_hours(pv)
    try:
        _, ctrl = run_day(net, buses, load, pv, {b: cap_mw for b in buses},
                          os.path.join(TMP, "curt"), use_qu_control=use_qu,
                          use_curtailment=True)
    except Exception:
        return 100.0, potential, potential
    curtailed = ctrl.energy_curtailed_mwh if ctrl else 0.0
    return (100 * curtailed / potential if potential > 0 else 0.0), curtailed, potential


def load_real_dataset(args, net):
    """Return the full-year DataFrame (residential / commercial / pv)."""
    df = real_data.load_simbench_profiles()
    if args.data == "pvgis":
        pv = real_data.fetch_pvgis_pv(args.lat, args.lon, year=args.pv_year)
        df = real_data.replace_pv(df, pv)
    elif args.data == "csv":
        if not args.pv_csv:
            sys.exit("--data csv needs --pv-csv PATH (columns: time, pv in p.u. of nameplate)")
        df = real_data.replace_pv(df, real_data.load_csv_series(args.pv_csv, name="pv"))
    return df


def seasonal_comparison(net, buses, df, steps_per_day, cap_mw, out_dir):
    rows = {}
    days = real_data.representative_days(df, net)
    for label, date in days.items():
        d = real_data.extract_day(df, date, steps_per_day, label)
        out_b, _ = run_day(net, buses, d.load, d.pv, {b: cap_mw for b in buses},
                           os.path.join(TMP, "season_b"))
        sb_ = summarize_violations(out_b)
        out_c, _ = run_day(net, buses, d.load, d.pv, {b: cap_mw for b in buses},
                           os.path.join(TMP, "season_c"), use_qu_control=True,
                           use_curtailment=True)
        sc = summarize_violations(out_c)
        pct, _, _ = curtailment_pct(net, buses, d.load, d.pv, cap_mw)
        rows[label] = {
            "date": str(date.date()),
            "pv_kwh_per_kwp": d.pv_energy_pu_h,
            "vmax_no_ctrl": sb_["max_voltage_pu"],
            "overvoltage_steps_no_ctrl": int((out_b["res_bus.vm_pu"] > 1.05).sum().sum()),
            "vmax_ctrl": sc["max_voltage_pu"],
            "overvoltage_steps_ctrl": int((out_c["res_bus.vm_pu"] > 1.05).sum().sum()),
            "vmin": sc["min_voltage_pu"],
            "max_line_loading_pct": sc["max_line_loading_pct"],
            "curtailed_pct": pct,
        }
    table = pd.DataFrame(rows).T
    for c in table.columns:
        if c != "date":
            table[c] = pd.to_numeric(table[c])
    table.round(4).to_csv(os.path.join(out_dir, "seasonal_comparison.csv"))
    analysis.plot_seasonal_results(table, out_dir)
    return table


def annual_curtailment(net, buses, df, steps_per_day, capacities, out_dir, budget_pct):
    """Annual curtailment estimate from one typical day per month, each
    weighted by the number of days in that month."""
    month_days = real_data.monthly_typical_days(df)
    n_days = df["pv"].groupby(df.index.month).size() / 96  # 15-min source data
    rows = []
    for cap in capacities:
        for m, date in month_days.items():
            d = real_data.extract_day(df, date, steps_per_day)
            pct, cur, pot = curtailment_pct(net, buses, d.load, d.pv, cap)
            rows.append({"capacity_mw": cap, "month": m, "date": str(date.date()),
                         "curtailed_pct": pct, "curtailed_mwh_day": cur,
                         "potential_mwh_day": pot, "days_in_month": n_days[m]})
        print(f"  {cap:.2f} MW/bus done")
    monthly = pd.DataFrame(rows)
    monthly.round(5).to_csv(os.path.join(out_dir, "monthly_curtailment.csv"), index=False)
    analysis.plot_monthly_curtailment(monthly, out_dir, budget_pct)

    w = monthly.assign(cur=monthly.curtailed_mwh_day * monthly.days_in_month,
                       pot=monthly.potential_mwh_day * monthly.days_in_month)
    annual = w.groupby("capacity_mw")[["cur", "pot"]].sum()
    annual["annual_curtailed_pct"] = 100 * annual["cur"] / annual["pot"]
    annual = annual.rename(columns={"cur": "curtailed_mwh_yr", "pot": "potential_mwh_yr"})
    annual.round(3).to_csv(os.path.join(out_dir, "annual_curtailment.csv"))
    return monthly, annual


def interpolate_budget_capacity(annual: pd.DataFrame, budget_pct: float) -> float | None:
    caps = annual.index.to_numpy(dtype=float)
    pcts = annual["annual_curtailed_pct"].to_numpy()
    if pcts[-1] <= budget_pct:
        return None  # budget never reached in the scanned range
    return float(np.interp(budget_pct, pcts, caps))


def main():
    parser = argparse.ArgumentParser(description="Renewable hosting capacity & bottleneck study")
    parser.add_argument("--data", choices=["synthetic", "simbench", "pvgis", "csv"],
                        default="synthetic", help="profile source (default: synthetic)")
    parser.add_argument("--day", default="worst_case",
                        help="real-data design day: worst_case | summer_peak | "
                             "typical_summer | typical_winter | YYYY-MM-DD")
    parser.add_argument("--lat", type=float, default=52.52, help="PVGIS latitude")
    parser.add_argument("--lon", type=float, default=13.40, help="PVGIS longitude")
    parser.add_argument("--pv-year", type=int, default=2020, help="PVGIS year")
    parser.add_argument("--pv-csv", help="CSV with columns time,pv (p.u. of nameplate)")
    parser.add_argument("--capacity-mw", type=float, default=0.7,
                        help="PV capacity per bus (MW) for the baseline-vs-control comparison")
    parser.add_argument("--fine", action="store_true", help="use 15-min resolution (96 steps)")
    parser.add_argument("--budget-pct", type=float, default=5.0,
                        help="curtailment planning budget, %% of potential PV yield")
    parser.add_argument("--skip-annual", action="store_true",
                        help="real data: skip the seasonal / annual analysis")
    args = parser.parse_args()

    steps_per_day = 96 if args.fine else 24
    real = args.data != "synthetic"
    out_dir = os.path.join(RESULTS_ROOT, "real_data") if real else RESULTS_ROOT
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("RENEWABLE HOSTING CAPACITY & BOTTLENECK ANALYSIS")
    print("Network: CIGRE MV benchmark (residential + commercial feeder)")
    print(f"Profiles: {args.data}")
    print("=" * 70)

    net = netmod.build_network()
    buses = netmod.candidate_buses(net)
    netmod.attach_sgen_fleet(net, buses)
    print("\n" + netmod.network_summary(net))
    print(f"Candidate DER connection buses: {buses}")

    if real:
        df = load_real_dataset(args, net)
        print("\nDataset statistics:\n" + real_data.profile_statistics(df, net).to_string())
        analysis.plot_annual_overview(df, out_dir)
        rep = real_data.representative_days(df, net)
        date = rep.get(args.day) or pd.Timestamp(args.day)
        day = real_data.extract_day(df, date, steps_per_day, args.day)
        load, pv = day.load, day.pv
        print(f"Design day: {args.day} = {date.date()} "
              f"(PV yield {day.pv_energy_pu_h:.2f} kWh/kWp)")
        syn = profmod.build_profile_set(steps_per_day=steps_per_day)
        analysis.plot_profile_comparison(
            {k: real_data.extract_day(df, v, steps_per_day, k) for k, v in rep.items()},
            syn["load"], syn["pv"], out_dir)
        analysis.plot_pv_and_load_profiles(load, pv, out_dir,
                                           title=f"Design day {date.date()} (real data)")
    else:
        prof = profmod.build_profile_set(steps_per_day=steps_per_day)
        load, pv = prof["load"], prof["pv"]
        analysis.plot_pv_and_load_profiles(load, pv, out_dir)

    # --- 1. Per-bus standalone hosting capacity (fast screening) ---------
    print("\n[1] Per-bus standalone PV hosting capacity (worst-case snapshot)...")
    hc = per_bus_snapshot_capacity(net, buses, load, pv, capacity_max_mw=30.0)
    analysis.plot_hosting_capacity_map(hc, out_dir)
    hc.round(3).to_csv(os.path.join(out_dir, "hosting_capacity_by_bus.csv"))
    print(hc.to_string())

    # --- 2. Baseline 24h time series, no control --------------------------
    cap = {b: args.capacity_mw for b in buses}
    print(f"\n[2] 24h time-series @ {args.capacity_mw} MW/bus, NO control...")
    out_base, _ = run_day(net, buses, load, pv, cap, os.path.join(TMP, "base"))
    summ_base = summarize_violations(out_base)
    print(summ_base)
    analysis.plot_voltage_profile(out_base, out_dir, name="voltage_profile_no_control.png")
    analysis.plot_line_loading(out_base, out_dir, name="line_loading_no_control.png")

    # --- 3. Same scenario WITH Q(U) + curtailment --------------------------
    print(f"\n[3] 24h time-series @ {args.capacity_mw} MW/bus, Q(U) + curtailment...")
    out_ctrl, _ = run_day(net, buses, load, pv, cap, os.path.join(TMP, "ctrl"),
                          use_qu_control=True, use_curtailment=True)
    summ_ctrl = summarize_violations(out_ctrl)
    print(summ_ctrl)
    analysis.plot_voltage_profile(out_ctrl, out_dir, name="voltage_profile_with_control.png")
    analysis.plot_line_loading(out_ctrl, out_dir, name="line_loading_with_control.png")
    analysis.plot_before_after_comparison(summ_base, summ_ctrl, out_dir)

    # --- 3b. Zero-violation hosting capacity: none vs. Q(U) ------------------
    print("\n[3b] Zero-violation uniform hosting capacity on the design day...")
    hc_rows = {}
    for label, qu in [("no control", False), ("Q(U) droop only", True)]:
        c, _ = global_time_series_capacity(net, buses, load, pv, os.path.join(TMP, "hc"),
                                           use_qu_control=qu, capacity_max_mw=3.0,
                                           tol_mw=0.02)
        hc_rows[label] = c
        print(f"  {label:18s}: {c:.2f} MW/bus  ({c * len(buses):.1f} MW total)")

    # --- 4. Curtailment-budget hosting capacity (design day) --------------
    print(f"\n[4] Design-day curtailment vs. capacity, {args.budget_pct:.0f}% budget...")
    caps, pcts = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2], []
    for c in caps:
        pcts.append(curtailment_pct(net, buses, load, pv, c)[0])
        print(f"  {c:.1f} MW/bus -> {pcts[-1]:.2f}% curtailed")
    analysis.plot_curtailment_vs_capacity(caps, pcts, out_dir, budget_pct=args.budget_pct)

    best_cap, best_pct, _, _, _ = capacity_with_curtailment_budget(
        net, buses, load, pv, os.path.join(TMP, "budget"),
        use_qu_control=True, max_curtailment_pct=args.budget_pct,
        capacity_max_mw=1.5, tol_mw=0.05)
    print(f"\n>>> Design-day PV hosting capacity per bus: {best_cap:.2f} MW "
          f"(curtailing {best_pct:.2f}% of that day's potential yield)")
    hc_rows[f"Q(U) + curtailment, <= {args.budget_pct:.0f}% on design day"] = best_cap

    # --- 5/6. Real data: seasonal + annual view ---------------------------
    if real and not args.skip_annual:
        print(f"\n[5] Representative days @ {args.capacity_mw} MW/bus...")
        table = seasonal_comparison(net, buses, df, steps_per_day, args.capacity_mw, out_dir)
        print(table.to_string(float_format=lambda v: f"{v:.3f}"))

        print("\n[6] Annual curtailment estimate (12 typical days, month-weighted)...")
        annual_caps = [0.4, 0.7, 1.0, 1.3, 1.6, 2.0]
        _, annual = annual_curtailment(net, buses, df, steps_per_day, annual_caps,
                                       out_dir, args.budget_pct)
        print(annual.round(3).to_string())
        annual_cap = interpolate_budget_capacity(annual, args.budget_pct)
        if annual_cap is None:
            print(f"\n>>> Annual curtailment stays below {args.budget_pct:.0f}% up to "
                  f"{annual_caps[-1]} MW/bus")
        else:
            print(f"\n>>> Annual-budget PV hosting capacity per bus: {annual_cap:.2f} MW "
                  f"({annual_cap * len(buses):.1f} MW feeder total) at "
                  f"{args.budget_pct:.0f}% annual curtailment")
            hc_rows[f"Q(U) + curtailment, <= {args.budget_pct:.0f}% annual"] = annual_cap

    summary = pd.DataFrame({"capacity_mw_per_bus": pd.Series(hc_rows)})
    summary["feeder_total_mw"] = summary["capacity_mw_per_bus"] * len(buses)
    summary.round(3).to_csv(os.path.join(out_dir, "hosting_capacity_summary.csv"))
    print("\nHosting-capacity summary:\n" + summary.round(2).to_string())

    print(f"\nAll plots and tables saved to: {os.path.abspath(out_dir)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
