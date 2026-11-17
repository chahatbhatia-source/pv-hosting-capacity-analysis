"""
analysis.py
-----------
Plotting and reporting helpers. Every function saves a PNG into the
given output directory and returns the path, so the CLI / notebook
can build a full results gallery in one pass.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.size": 10,
})


def _save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_voltage_profile(output, out_dir, v_min=0.95, v_max=1.05, name="voltage_profile.png"):
    vm = output["res_bus.vm_pu"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for col in vm.columns:
        ax.plot(vm.index, vm[col], linewidth=0.9, alpha=0.65)
    ax.axhline(v_min, color="crimson", linestyle="--", linewidth=1.2, label=f"v_min = {v_min}")
    ax.axhline(v_max, color="crimson", linestyle="--", linewidth=1.2, label=f"v_max = {v_max}")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Bus voltage (p.u.)")
    ax.set_title("24h bus voltage profiles")
    ax.legend(loc="upper right", fontsize=8)
    return _save(fig, out_dir, name)


def plot_line_loading(output, out_dir, line_max_pct=100.0, name="line_loading.png"):
    ll = output["res_line.loading_percent"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for col in ll.columns:
        ax.plot(ll.index, ll[col], linewidth=0.9, alpha=0.65)
    ax.axhline(line_max_pct, color="crimson", linestyle="--", linewidth=1.2,
               label=f"thermal limit = {line_max_pct:.0f}%")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Line loading (%)")
    ax.set_title("24h line loading profiles")
    ax.legend(loc="upper right", fontsize=8)
    return _save(fig, out_dir, name)


def plot_hosting_capacity_map(hc: pd.Series, out_dir, name="hosting_capacity_by_bus.png"):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(hc)))
    ax.bar([str(b) for b in hc.index], hc.values, color=colors)
    ax.set_xlabel("Bus")
    ax.set_ylabel("Standalone PV hosting capacity (MW)")
    ax.set_title("Per-bus PV hosting capacity (worst-case snapshot)")
    for i, v in enumerate(hc.values):
        ax.text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=8)
    return _save(fig, out_dir, name)


def plot_curtailment_vs_capacity(capacities, curtailment_pcts, out_dir,
                                  budget_pct=5.0, name="curtailment_vs_capacity.png"):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(capacities, curtailment_pcts, marker="o", linewidth=1.6, color="#2b7a78")
    ax.axhline(budget_pct, color="crimson", linestyle="--", linewidth=1.2,
               label=f"curtailment budget = {budget_pct:.0f}%")
    ax.set_xlabel("Installed PV capacity per bus (MW)")
    ax.set_ylabel("Curtailed energy (% of potential PV yield)")
    ax.set_title("Curtailed energy vs. installed PV capacity")
    ax.legend(loc="upper left", fontsize=8)
    return _save(fig, out_dir, name)


def plot_before_after_comparison(summary_before: dict, summary_after: dict, out_dir,
                                  name="before_after_control.png"):
    metrics = ["max_voltage_pu", "max_line_loading_pct", "n_voltage_violations", "n_line_overloads"]
    labels = ["Max voltage\n(p.u.)", "Max line\nloading (%)", "Voltage\nviolations (#)", "Line\noverloads (#)"]
    before = [summary_before[m] for m in metrics]
    after = [summary_after[m] for m in metrics]

    fig, axes = plt.subplots(1, 4, figsize=(11, 3.5))
    for ax, lbl, b, a in zip(axes, labels, before, after):
        ax.bar(["No control", "Q(U)+curtail"], [b, a], color=["#c94c4c", "#2b7a78"])
        ax.set_title(lbl, fontsize=9)
        for x, val in enumerate([b, a]):
            ax.text(x, val, f"{val:.2f}" if val < 10 else f"{val:.0f}",
                    ha="center", va="bottom", fontsize=8)
    fig.suptitle("Impact of Q(U) droop + curtailment control", y=1.05)
    fig.tight_layout()
    return _save(fig, out_dir, name)


def plot_pv_and_load_profiles(load_profile, pv_profile, out_dir, name="input_profiles.png",
                              title="24h load and PV generation profiles"):
    fig, ax = plt.subplots(figsize=(8, 4))
    if isinstance(load_profile, pd.DataFrame):
        for col in load_profile.columns:
            ax.plot(load_profile.index, load_profile[col].values,
                    label=f"Load - {col} (normalized)", linewidth=1.8)
    else:
        ax.plot(load_profile.index, load_profile.values, label="Load (normalized)", linewidth=1.8)
    ax.plot(pv_profile.index, pv_profile.values, label="PV (p.u. of nameplate)", linewidth=1.8)
    ax.set_xlabel("Time")
    ax.set_ylabel("Normalized output [0-1]")
    ax.set_title(title)
    ax.legend(fontsize=8)
    return _save(fig, out_dir, name)


# --------------------------------------------------------------------------
# Real-data figures
# --------------------------------------------------------------------------
def plot_annual_overview(df: pd.DataFrame, out_dir, name="real_data_annual_overview.png"):
    """Two-panel overview of a year of real data: daily PV energy and daily
    min/max of the normalized loads - the 'data' figure for a report."""
    daily_pv = df["pv"].resample("D").mean() * 24  # full-load hours per day
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    a1.bar(daily_pv.index, daily_pv.values, width=1.0, color="#e1a32a")
    a1.set_ylabel("PV yield (kWh/kWp/day)")
    a1.set_title("Real input data: one year (SimBench 2016, 15-min)")
    for col, color in [("residential", "#2b7a78"), ("commercial", "#6a4c93")]:
        if col in df:
            d = df[col].resample("D")
            a2.fill_between(daily_pv.index, d.min().values, d.max().values,
                            alpha=0.35, color=color, label=f"{col} (daily min-max)")
    a2.set_ylabel("Load (p.u. of annual peak)")
    a2.legend(fontsize=8, loc="upper right")
    return _save(fig, out_dir, name)


def plot_profile_comparison(days: dict, synthetic_load, synthetic_pv, out_dir,
                            name="synthetic_vs_real_profiles.png"):
    """Overlay the synthetic design day on the real representative days."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    hours = np.arange(len(synthetic_pv)) * 24 / len(synthetic_pv)
    a1.plot(hours, synthetic_pv.values, "k--", linewidth=2, label="synthetic")
    a2.plot(hours, synthetic_load.values, "k--", linewidth=2, label="synthetic")
    for label, d in days.items():
        h = np.arange(len(d.pv)) * 24 / len(d.pv)
        a1.plot(h, d.pv.values, linewidth=1.4, label=f"{label} ({d.date.date()})")
        a2.plot(h, d.load["residential"].values, linewidth=1.4, label=label)
    a1.set_title("PV (p.u. of nameplate)")
    a2.set_title("Residential load (p.u. of annual peak)")
    for a in (a1, a2):
        a.set_xlabel("Hour of day")
        a.set_xlim(0, 24)
    a1.legend(fontsize=7)
    return _save(fig, out_dir, name)


def plot_seasonal_results(table: pd.DataFrame, out_dir, name="seasonal_comparison.png"):
    """Bar charts of the per-day results table from run_study step 5."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    x = np.arange(len(table))
    axes[0].bar(x - 0.2, table["vmax_no_ctrl"], 0.4, label="no control", color="#c94c4c")
    axes[0].bar(x + 0.2, table["vmax_ctrl"], 0.4, label="Q(U)+curtail", color="#2b7a78")
    axes[0].axhline(1.05, color="crimson", linestyle="--", linewidth=1)
    lo = min(table["vmax_no_ctrl"].min(), table["vmax_ctrl"].min())
    axes[0].set_ylim(min(lo, 1.0) - 0.01, None)
    axes[0].set_title("Max bus voltage (p.u.)")
    axes[0].legend(fontsize=7)
    axes[1].bar(x, table["overvoltage_steps_no_ctrl"], color="#c94c4c")
    axes[1].set_title("Over-voltage bus-steps, no control")
    axes[2].bar(x, table["curtailed_pct"], color="#e1a32a")
    axes[2].set_title("Energy curtailed with control (%)")
    for a in axes:
        a.set_xticks(x)
        a.set_xticklabels(table.index, rotation=25, ha="right", fontsize=8)
    fig.tight_layout()
    return _save(fig, out_dir, name)


def plot_monthly_curtailment(monthly: pd.DataFrame, out_dir, budget_pct=5.0,
                             name="monthly_curtailment.png"):
    fig, ax = plt.subplots(figsize=(8, 4))
    for cap, grp in monthly.groupby("capacity_mw"):
        ax.plot(grp["month"], grp["curtailed_pct"], marker="o", label=f"{cap:.2f} MW/bus")
    ax.axhline(budget_pct, color="crimson", linestyle="--", linewidth=1,
               label=f"{budget_pct:.0f}% budget")
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("Month (typical day simulated)")
    ax.set_ylabel("Curtailed energy (% of potential)")
    ax.set_title("Seasonality of curtailment (real data)")
    ax.legend(fontsize=8)
    return _save(fig, out_dir, name)
