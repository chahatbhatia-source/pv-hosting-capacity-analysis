"""
run_study.py
------------
End-to-end entry point. Running this script:

  1. builds the CIGRE MV benchmark network,
  2. generates 24h load / PV profiles,
  3. computes the per-bus standalone PV hosting capacity map,
  4. runs a 24h time-series simulation with NO control (baseline),
  5. runs the same scenario WITH Q(U) droop + curtailment control,
  6. bisection-searches the installed capacity that keeps curtailed
     energy under a 5% planning budget,
  7. saves every plot into results/, and prints a text summary.

Usage
-----
    python src/run_study.py [--capacity-mw 0.35] [--fine]

    --capacity-mw   PV capacity (MW) installed at every candidate bus
                     for the baseline vs. controlled comparison
                     (default: 0.35 MW/bus -> a clearly stressed case)
    --fine           use 15-minute resolution (96 steps) instead of
                     the default hourly resolution (slower, finer)
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import network as netmod
import profiles as profmod
import analysis
from timeseries_sim import run_day, summarize_violations
from hosting_capacity import per_bus_snapshot_capacity, capacity_with_curtailment_budget

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def main():
    parser = argparse.ArgumentParser(description="Renewable hosting capacity & bottleneck study")
    parser.add_argument("--capacity-mw", type=float, default=0.35,
                         help="PV capacity per bus (MW) for the baseline-vs-control comparison")
    parser.add_argument("--fine", action="store_true", help="use 15-min resolution (96 steps)")
    parser.add_argument("--budget-pct", type=float, default=5.0,
                         help="curtailment planning budget, %% of potential PV yield")
    args = parser.parse_args()

    steps_per_day = 96 if args.fine else 24
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 70)
    print("RENEWABLE HOSTING CAPACITY & BOTTLENECK ANALYSIS")
    print("Network: CIGRE MV benchmark (11 kV residential feeder)")
    print("=" * 70)

    net = netmod.build_network()
    buses = netmod.candidate_buses(net)
    netmod.attach_sgen_fleet(net, buses)
    print("\n" + netmod.network_summary(net))
    print(f"Candidate DER connection buses: {buses}")

    prof = profmod.build_profile_set(steps_per_day=steps_per_day)
    analysis.plot_pv_and_load_profiles(prof["load"], prof["pv"], RESULTS_DIR)

    # --- 1. Per-bus standalone hosting capacity (fast screening) ---------
    print("\n[1/4] Per-bus standalone PV hosting capacity (worst-case snapshot)...")
    hc = per_bus_snapshot_capacity(net, buses, prof["load"], prof["pv"])
    analysis.plot_hosting_capacity_map(hc, RESULTS_DIR)
    print(hc.to_string())

    # --- 2. Baseline 24h time series, no control --------------------------
    print(f"\n[2/4] 24h time-series @ {args.capacity_mw} MW/bus, NO control...")
    out_base, _ = run_day(net, buses, prof["load"], prof["pv"],
                           {b: args.capacity_mw for b in buses}, "/tmp/hc_base_run",
                           use_qu_control=False, use_curtailment=False)
    summ_base = summarize_violations(out_base)
    print(summ_base)
    analysis.plot_voltage_profile(out_base, RESULTS_DIR, name="voltage_profile_no_control.png")
    analysis.plot_line_loading(out_base, RESULTS_DIR, name="line_loading_no_control.png")

    # --- 3. Same scenario WITH Q(U) + curtailment --------------------------
    print(f"\n[3/4] 24h time-series @ {args.capacity_mw} MW/bus, Q(U) + curtailment...")
    out_ctrl, curt_ctrl = run_day(net, buses, prof["load"], prof["pv"],
                                   {b: args.capacity_mw for b in buses}, "/tmp/hc_ctrl_run",
                                   use_qu_control=True, use_curtailment=True)
    summ_ctrl = summarize_violations(out_ctrl)
    print(summ_ctrl)
    analysis.plot_voltage_profile(out_ctrl, RESULTS_DIR, name="voltage_profile_with_control.png")
    analysis.plot_line_loading(out_ctrl, RESULTS_DIR, name="line_loading_with_control.png")
    analysis.plot_before_after_comparison(summ_base, summ_ctrl, RESULTS_DIR)

    # --- 4. Curtailment-budget hosting capacity sweep -----------------------
    print(f"\n[4/4] Searching max capacity for <= {args.budget_pct:.0f}% curtailment budget...")
    caps, pcts = [], []
    for c in [0.2, 0.4, 0.6, 0.8, 1.0]:
        try:
            from hosting_capacity import _worst_case_step  # noqa
            from control import CurtailmentController  # noqa
            out_c, ctrl_c = run_day(net, buses, prof["load"], prof["pv"],
                                     {b: c for b in buses}, "/tmp/hc_sweep",
                                     use_qu_control=True, use_curtailment=True)
            potential_mwh = c * len(buses) * prof["pv"].sum() * (
                (prof["pv"].index[1] - prof["pv"].index[0]).seconds / 3600)
            curtailed_mwh = ctrl_c.energy_curtailed_mwh if ctrl_c else 0.0
            pct = 100 * curtailed_mwh / potential_mwh if potential_mwh > 0 else 0.0
        except Exception as e:
            pct = 100.0
        caps.append(c)
        pcts.append(pct)
        print(f"  {c:.1f} MW/bus -> {pct:.2f}% curtailed")
    analysis.plot_curtailment_vs_capacity(caps, pcts, RESULTS_DIR, budget_pct=args.budget_pct)

    best_cap, best_pct, _, curt_mwh, pot_mwh = capacity_with_curtailment_budget(
        net, buses, prof["load"], prof["pv"], "/tmp/hc_budget_final",
        use_qu_control=True, max_curtailment_pct=args.budget_pct,
        capacity_max_mw=1.5, tol_mw=0.05)
    print(f"\n>>> Recommended PV hosting capacity per bus: {best_cap:.2f} MW "
          f"(curtailing {best_pct:.2f}% of potential yield, within the "
          f"{args.budget_pct:.0f}% budget)")

    print(f"\nAll plots saved to: {os.path.abspath(RESULTS_DIR)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
