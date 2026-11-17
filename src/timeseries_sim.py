"""
timeseries_sim.py
------------------
Wires up pandapower's `timeseries` module: DataSource-driven
ConstControls for loads and DER, an OutputWriter that logs bus
voltages / line loadings / sgen output every step, and a thin
wrapper `run_day` that executes a full 24h quasi-static AC power
flow sweep, optionally with the Q(U) and curtailment controllers
from `control.py` active.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

import pandapower as pp
from pandapower.control import ConstControl
from pandapower.timeseries import DFData, OutputWriter
from pandapower.timeseries.run_time_series import run_timeseries

from control import QUDroopController, CurtailmentController


def _clear_controllers(net):
    net.controller = net.controller.iloc[0:0]


def step_hours(profile) -> float:
    """Duration of one time step (hours), from the profile's DatetimeIndex."""
    if isinstance(profile.index, pd.DatetimeIndex) and len(profile) > 1:
        return (profile.index[1] - profile.index[0]).total_seconds() / 3600
    return 24.0 / len(profile)


def load_multipliers(net, load_profile) -> pd.DataFrame:
    """Per-load normalized multiplier for every time step (steps x loads).

    `load_profile` is either
      * a Series - one curve applied to every load (synthetic case), or
      * a DataFrame with ``residential`` and ``commercial`` columns - CIGRE
        loads named ``Load R*`` follow the residential curve, all others
        (``Load CI*``) the commercial one (real-data case).
    """
    if isinstance(load_profile, pd.Series):
        return pd.DataFrame({i: load_profile.to_numpy() for i in net.load.index})
    is_res = net.load.name.str.contains("Load R")
    return pd.DataFrame({
        i: load_profile["residential" if is_res.at[i] else "commercial"].to_numpy()
        for i in net.load.index
    })


def setup_time_series(net, buses, load_profile, gen_profile, pv_capacity_mw,
                       use_qu_control=False, use_curtailment=False,
                       v_min=0.95, v_max=1.05, line_max_pct=100.0):
    """Attach ConstControls (and optionally Q(U)/curtailment controllers)
    that drive `net.load` and `net.sgen` from the given normalized
    profiles, scaled by each element's rated power / the installed
    PV capacity dict {bus: mw}.
    """
    _clear_controllers(net)

    n_steps = len(load_profile)

    # --- Loads: absolute profile per element = normalized curve x rated p/q ---
    # IMPORTANT: scale from the frozen p_mw_nominal/q_mvar_nominal columns
    # (set once in network.build_network), never from net.load.p_mw/q_mvar
    # directly -- those get overwritten by ConstControl during every
    # time-series run, so reading them here would compound scaling errors
    # on any second/third run that reuses the same net object.
    mult = load_multipliers(net, load_profile)
    load_abs = mult * net.load["p_mw_nominal"]
    loadq_abs = mult * net.load["q_mvar_nominal"]
    ConstControl(net, element="load", variable="p_mw", element_index=net.load.index,
                 data_source=DFData(load_abs), profile_name=list(net.load.index))
    ConstControl(net, element="load", variable="q_mvar", element_index=net.load.index,
                 data_source=DFData(loadq_abs), profile_name=list(net.load.index))

    # --- Sgens: each candidate bus's sgen output = pv_capacity_mw[bus] * gen curve ---
    sgen_for_bus = {net.sgen.at[idx, "bus"]: idx for idx in net.sgen.index}
    sgen_abs = pd.DataFrame({
        sgen_for_bus[b]: gen_profile.values * pv_capacity_mw.get(b, 0.0)
        for b in buses if b in sgen_for_bus
    })
    ConstControl(net, element="sgen", variable="p_mw",
                 element_index=list(sgen_abs.columns),
                 data_source=DFData(sgen_abs), profile_name=list(sgen_abs.columns))

    if use_qu_control:
        QUDroopController(net, sgen_idx=list(sgen_abs.columns), order=1, level=1)

    curt_ctrl = None
    if use_curtailment:
        p_rated = [pv_capacity_mw.get(net.sgen.at[i, "bus"], 0.0) for i in sgen_abs.columns]
        curt_ctrl = CurtailmentController(net, sgen_idx=list(sgen_abs.columns), p_rated_mw=p_rated,
                                           v_min=v_min, v_max=v_max, line_max_pct=line_max_pct,
                                           step_hours=step_hours(gen_profile),
                                           order=2, level=2)

    return n_steps, curt_ctrl


def build_output_writer(net, out_dir, n_steps):
    os.makedirs(out_dir, exist_ok=True)
    ow = OutputWriter(net, output_path=out_dir, output_file_type=".csv",
                       log_variables=[])
    ow.log_variable("res_bus", "vm_pu")
    ow.log_variable("res_line", "loading_percent")
    ow.log_variable("res_trafo", "loading_percent")
    ow.log_variable("sgen", "p_mw")
    ow.log_variable("sgen", "q_mvar")
    return ow


def run_day(net, buses, load_profile, gen_profile, pv_capacity_mw, out_dir,
            use_qu_control=False, use_curtailment=False,
            v_min=0.95, v_max=1.05, line_max_pct=100.0):
    """Run a full 24h (len(load_profile) steps) time-series AC power flow.

    Returns the OutputWriter's collected results dict-of-DataFrames.
    """
    n_steps, curt_ctrl = setup_time_series(
        net, buses, load_profile, gen_profile, pv_capacity_mw,
        use_qu_control=use_qu_control, use_curtailment=use_curtailment,
        v_min=v_min, v_max=v_max, line_max_pct=line_max_pct,
    )
    ow = build_output_writer(net, out_dir, n_steps)
    # the OutputWriter registers itself in net.output_writer on creation;
    # passing it again to run_timeseries is deprecated in pandapower >= 3
    run_timeseries(net, time_steps=range(n_steps), verbose=False, max_iter=100)
    return ow.output, curt_ctrl


def summarize_violations(output, v_min=0.95, v_max=1.05, line_max_pct=100.0):
    vm = output["res_bus.vm_pu"]
    line = output["res_line.loading_percent"]
    trafo = output["res_trafo.loading_percent"]
    return {
        "min_voltage_pu": float(vm.min().min()),
        "max_voltage_pu": float(vm.max().max()),
        "n_voltage_violations": int(((vm < v_min) | (vm > v_max)).sum().sum()),
        "max_line_loading_pct": float(line.max().max()) if not line.empty else 0.0,
        "n_line_overloads": int((line > line_max_pct).sum().sum()) if not line.empty else 0,
        "max_trafo_loading_pct": float(trafo.max().max()) if not trafo.empty else 0.0,
    }
