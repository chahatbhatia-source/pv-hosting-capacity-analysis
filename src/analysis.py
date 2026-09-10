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


def plot_pv_and_load_profiles(load_profile, pv_profile, out_dir, name="input_profiles.png"):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(load_profile.index, load_profile.values, label="Load (normalized)", linewidth=1.8)
    ax.plot(pv_profile.index, pv_profile.values, label="PV (normalized)", linewidth=1.8)
    ax.set_xlabel("Time")
    ax.set_ylabel("Normalized output [0-1]")
    ax.set_title("24h load and PV generation profiles")
    ax.legend()
    return _save(fig, out_dir, name)
