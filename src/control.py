"""
control.py
----------
Custom pandapower controllers implementing two grid-code-inspired
DER control strategies used to raise hosting capacity without
network reinforcement:

  * QUDroopController   : Q(U) reactive power droop, per VDE-AR-N
                           4110 / 4105-style characteristic. As bus
                           voltage rises above a deadband, the DER
                           absorbs reactive power (negative Q) to
                           pull the voltage back down; symmetric
                           injection below the deadband.

  * CurtailmentController : simple local active-power curtailment.
                             If Q(U) support alone is insufficient to
                             clear a voltage or thermal violation at
                             a bus, active power is ramped down in
                             fixed steps until the violation clears
                             or a minimum curtailment floor is hit.

Both controllers subclass pandapower's `basic_controller.Controller`
so they plug directly into `pandapower.timeseries.run_timeseries`
and are re-evaluated (and iterated to convergence) at every time
step, alongside the load/sgen `ConstControl` profile controllers.
"""
from __future__ import annotations

import numpy as np
from pandapower.control.basic_controller import Controller


class QUDroopController(Controller):
    """Q(U) droop control for one or more sgens sharing a bus set.

    Characteristic (per-unit voltage vs. reactive power, symmetric):

        v <= v1                       : Q = +Q_max   (full injection)
        v1 < v < v2 (deadband)        : Q = 0
        v2 <= v < v3                  : Q ramps 0 -> -Q_max
        v >= v3                       : Q = -Q_max   (full absorption)

    Q_max is expressed as a fraction of each sgen's apparent power
    capability, itself derived from its active power rating and an
    assumed inverter overrating (default cos(phi) = 0.95 -> ~33% Q
    headroom at rated P, matching common LV/MV grid-code inverters).
    """

    def __init__(self, net, sgen_idx, v1=0.96, v2=0.98, v3=1.02, v4=1.04,
                 q_capability=0.33, tol=3e-3, damping=0.5, in_service=True,
                 order=0, level=0, **kwargs):
        super().__init__(net, in_service=in_service, order=order, level=level,
                          initial_run=True, **kwargs)
        self.sgen_idx = list(sgen_idx)
        self.v1, self.v2, self.v3, self.v4 = v1, v2, v3, v4
        self.q_capability = q_capability
        self.tol = tol
        self.damping = damping  # under-relaxation to avoid limit-cycle oscillation
        self.applied = np.zeros(len(self.sgen_idx))

    def _droop_factor(self, v):
        """Continuous piecewise-linear droop factor in [-1, 1].

        v <= v1            : +1.0 (full injection, undervoltage support)
        v1 < v <= v2        : ramps +1.0 -> 0.0
        v2 < v <= v3        : 0.0  (deadband)
        v3 < v <= v4        : ramps 0.0 -> -1.0
        v > v4              : -1.0 (full absorption, overvoltage mitigation)
        """
        if v <= self.v1:
            return 1.0
        if v <= self.v2:
            return 1.0 - (v - self.v1) / (self.v2 - self.v1)
        if v <= self.v3:
            return 0.0
        if v <= self.v4:
            return -(v - self.v3) / (self.v4 - self.v3)
        return -1.0

    def initialize_control(self, net):
        self.applied[:] = 0.0

    def is_converged(self, net):
        if "res_bus" not in net or net.res_bus.empty:
            return False
        buses = net.sgen.loc[self.sgen_idx, "bus"].values
        v = net.res_bus.loc[buses, "vm_pu"].values
        target = np.array([self._droop_factor(vi) for vi in v])
        return np.all(np.abs(target - self.applied) < self.tol)

    def control_step(self, net):
        buses = net.sgen.loc[self.sgen_idx, "bus"].values
        v = net.res_bus.loc[buses, "vm_pu"].values
        p_mw = net.sgen.loc[self.sgen_idx, "p_mw"].values
        s_max = np.abs(p_mw) / 0.95  # inverter apparent power rating
        q_max = self.q_capability * s_max
        target = np.array([self._droop_factor(vi) for vi in v])
        # under-relaxed update: step only partway toward the new droop target
        # each iteration, which damps the voltage<->Q feedback loop and
        # prevents limit-cycle oscillation in meshed/short-line networks
        blended = self.applied + self.damping * (target - self.applied)
        q_new = blended * q_max
        net.sgen.loc[self.sgen_idx, "q_mvar"] = q_new
        self.applied = blended


class CurtailmentController(Controller):
    """Local active-power curtailment as a last-resort measure.

    Triggers only when, after Q(U) support has settled, a bus is
    still outside [v_min, v_max] or a line is above `line_max_pct`.
    Curtails the sgen(s) electrically closest to the worst violation
    by `step_pct` of currently available power per iteration, down to
    `p_floor_pct` of rated capacity.
    """

    def __init__(self, net, sgen_idx, p_rated_mw, v_min=0.95, v_max=1.05,
                 line_max_pct=100.0, step_pct=0.20, p_floor_pct=0.0,
                 tol=1e-5, order=1, level=1, in_service=True, **kwargs):
        super().__init__(net, in_service=in_service, order=order, level=level,
                          initial_run=True, **kwargs)
        self.sgen_idx = list(sgen_idx)
        self.p_rated = np.array(p_rated_mw, dtype=float)
        self.v_min, self.v_max = v_min, v_max
        self.line_max_pct = line_max_pct
        self.step_pct = step_pct
        self.p_floor_pct = p_floor_pct
        self.tol = tol
        self.curtail_frac = np.zeros(len(self.sgen_idx))
        self.energy_curtailed_mwh = 0.0

    def initialize_control(self, net):
        # curtail_frac resets each time step (fresh decision per snapshot);
        # energy_curtailed_mwh is a running total across the whole day and
        # must NOT be reset here.
        self.curtail_frac[:] = 0.0

    def _violations(self, net):
        v_viol = (net.res_bus.vm_pu.min() < self.v_min) or \
                 (net.res_bus.vm_pu.max() > self.v_max)
        line_viol = net.res_line.loading_percent.max() > self.line_max_pct \
            if len(net.res_line) else False
        return v_viol, line_viol

    def is_converged(self, net):
        if "res_bus" not in net or net.res_bus.empty:
            return False
        v_viol, line_viol = self._violations(net)
        floor_reached = np.all(self.curtail_frac >= (1 - self.p_floor_pct) - self.tol)
        return (not v_viol and not line_viol) or floor_reached

    def control_step(self, net):
        v_viol, line_viol = self._violations(net)
        if v_viol or line_viol:
            self.curtail_frac = np.minimum(
                self.curtail_frac + self.step_pct, 1 - self.p_floor_pct
            )
        p_before = net.sgen.loc[self.sgen_idx, "p_mw"].values.copy()
        p_new = self.p_rated * (1 - self.curtail_frac)
        net.sgen.loc[self.sgen_idx, "p_mw"] = np.minimum(p_before.clip(min=0), p_new) \
            if np.any(p_before > 0) else p_new
        # only ever curtail, never increase back above the profile-driven setpoint
        net.sgen.loc[self.sgen_idx, "p_mw"] = np.minimum(
            net.sgen.loc[self.sgen_idx, "p_mw"], p_before
        )
        self.energy_curtailed_mwh += float(
            np.sum(np.maximum(p_before - net.sgen.loc[self.sgen_idx, "p_mw"].values, 0))
        ) * (15 / 60)  # 15-min step -> MWh
