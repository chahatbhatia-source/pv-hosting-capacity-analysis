"""
tests/test_network.py
----------------------
Basic sanity checks for the network builder: makes sure the benchmark
grid loads with the expected shape and that DER attachment /
reset behave correctly.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandapower as pp
import network


def test_build_network_has_expected_elements():
    net = network.build_network()
    assert len(net.bus) > 0
    assert len(net.line) > 0
    assert len(net.load) > 0
    assert len(net.ext_grid) == 1


def test_nominal_load_columns_are_frozen():
    net = network.build_network()
    assert "p_mw_nominal" in net.load.columns
    assert "q_mvar_nominal" in net.load.columns
    assert (net.load["p_mw_nominal"] == net.load["p_mw"]).all()
    # mutating the live column must NOT affect the frozen nominal column
    net.load["p_mw"] = 0.0
    assert (net.load["p_mw_nominal"] != net.load["p_mw"]).any()


def test_candidate_buses_are_load_buses():
    net = network.build_network()
    buses = network.candidate_buses(net)
    assert set(buses) == set(net.load.bus.unique().tolist())
    assert len(buses) > 0


def test_attach_sgen_fleet_creates_one_sgen_per_bus():
    net = network.build_network()
    buses = network.candidate_buses(net)
    sgen_idx = network.attach_sgen_fleet(net, buses, p_mw=0.0)
    assert len(sgen_idx) == len(buses)
    assert len(net.sgen) == len(buses)
    assert set(net.sgen.bus) == set(buses)


def test_reset_sgens_zeroes_output():
    net = network.build_network()
    buses = network.candidate_buses(net)
    network.attach_sgen_fleet(net, buses, p_mw=1.0)
    network.reset_sgens(net, p_mw=0.0)
    assert (net.sgen["p_mw"] == 0.0).all()
    assert (net.sgen["q_mvar"] == 0.0).all()


def test_network_solves_power_flow_with_no_der():
    net = network.build_network()
    buses = network.candidate_buses(net)
    network.attach_sgen_fleet(net, buses, p_mw=0.0)
    pp.runpp(net)
    assert net.converged
    assert net.res_bus.vm_pu.notna().all()
