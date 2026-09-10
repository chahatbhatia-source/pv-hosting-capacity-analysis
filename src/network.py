"""
network.py
----------
Builds the benchmark distribution grid used throughout the study
(CIGRE Medium-Voltage benchmark network, residential feeder) and
provides helpers to attach scalable, controllable PV / wind
generators at candidate connection points.

The CIGRE MV network is a widely used European benchmark: an 11-bus,
20 kV radial/ring feeder supplied from a single HV/MV transformer,
originally published by CIGRE Task Force C6.04.02 and shipped
natively with pandapower.
"""
from __future__ import annotations

import pandapower as pp
import pandapower.networks as pn


def build_network(with_der: bool = False) -> pp.pandapowerNet:
    """Load the CIGRE MV benchmark network (residential branch).

    Parameters
    ----------
    with_der : bool
        If True, load the network with CIGRE's own predefined DER
        set already attached. We default to False because this study
        adds its own controllable, scalable PV/wind fleet instead.
    """
    net = pn.create_cigre_network_mv(with_der="pv_wind" if with_der else False)
    # Freeze the nominal (t=0) load ratings in dedicated columns. Time-series
    # runs repeatedly overwrite net.load.p_mw / q_mvar via ConstControl, so
    # anything that later needs "the load's rated power" (e.g. a second
    # simulation reusing this same net object) MUST read these columns --
    # never net.load.p_mw/q_mvar directly once a time series has run.
    net.load["p_mw_nominal"] = net.load["p_mw"]
    net.load["q_mvar_nominal"] = net.load["q_mvar"]
    return net


def candidate_buses(net: pp.pandapowerNet) -> list[int]:
    """Return the bus indices eligible for new DER connections.

    We exclude the slack/HV bus and any auxiliary buses so hosting
    capacity is only evaluated at real MV load buses (the points
    where a feed-in application would actually land).
    """
    load_buses = sorted(net.load.bus.unique().tolist())
    return load_buses


def attach_sgen_fleet(net: pp.pandapowerNet, buses: list[int], p_mw: float = 0.0,
                       sgen_type: str = "PV") -> list[int]:
    """Attach one controllable static generator (sgen) per candidate bus.

    Each sgen starts at p_mw (default 0, i.e. "not yet built") and its
    scaling factor / p_mw is what the hosting-capacity search and the
    time-series simulation will drive.

    Returns
    -------
    list[int]
        The pandapower element indices of the created sgens, in the
        same order as `buses`.
    """
    sgen_idx = []
    for b in buses:
        idx = pp.create_sgen(
            net, bus=b, p_mw=p_mw, q_mvar=0.0,
            name=f"{sgen_type}_{b}", type=sgen_type,
            controllable=True, in_service=True,
        )
        sgen_idx.append(idx)
    return sgen_idx


def reset_sgens(net: pp.pandapowerNet, p_mw: float = 0.0) -> None:
    """Reset every sgen in the network back to a given output level."""
    net.sgen["p_mw"] = p_mw
    net.sgen["q_mvar"] = 0.0


def network_summary(net: pp.pandapowerNet) -> str:
    lines = [
        f"Buses:        {len(net.bus)}",
        f"Lines:        {len(net.line)}",
        f"Loads:        {len(net.load)} "
        f"(total {net.load.p_mw.sum():.3f} MW / {net.load.q_mvar.sum():.3f} Mvar)",
        f"Transformers: {len(net.trafo)}",
        f"Ext. grids:   {len(net.ext_grid)}",
        f"Static gens:  {len(net.sgen)}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    net = build_network()
    print(network_summary(net))
    print("Candidate connection buses:", candidate_buses(net))
