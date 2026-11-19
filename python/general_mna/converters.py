"""Verified continuous-conduction state-space averaging examples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import sympy as sp

from ._native import build_mna, numeric_state_space
from .symbolic import descriptor_to_state_space, reorder_states, simplified_equal

D, L, C, R, V_g, s = sp.symbols("D L C R V_g s")
Q = 1 - D
STATE_ORDER = ["I(L1)", "V(out)"]


@dataclass(frozen=True)
class ConverterSpec:
    """The exact ON/OFF netlists and presentation metadata for a converter."""

    key: str
    title: str
    on_netlist: str
    off_netlist: str
    output_polarity: str


COMMON_OUTPUT = "C1 out 0 C\nR1 out 0 R"

SPECS = {
    "buck": ConverterSpec(
        key="buck",
        title="Buck converter",
        on_netlist=f"Vin vin 0 Vin\nL1 sw out L\n{COMMON_OUTPUT}\nVpath vin sw 0",
        off_netlist=f"Vin vin 0 Vin\nL1 sw out L\n{COMMON_OUTPUT}\nVpath 0 sw 0",
        output_polarity="positive",
    ),
    "boost": ConverterSpec(
        key="boost",
        title="Boost converter",
        on_netlist=f"Vin vin 0 Vin\nL1 vin sw L\n{COMMON_OUTPUT}\nVpath 0 sw 0",
        off_netlist=f"Vin vin 0 Vin\nL1 vin sw L\n{COMMON_OUTPUT}\nVpath out sw 0",
        output_polarity="positive",
    ),
    "buck_boost": ConverterSpec(
        key="buck_boost",
        title="Inverting buck-boost converter",
        on_netlist=f"Vin vin 0 Vin\nL1 x 0 L\n{COMMON_OUTPUT}\nVpath vin x 0",
        off_netlist=f"Vin vin 0 Vin\nL1 x 0 L\n{COMMON_OUTPUT}\nVpath out x 0",
        output_polarity="negative",
    ),
}


def _hand_model(key: str) -> dict[str, sp.Matrix | sp.Expr]:
    """Independent textbook derivation in the state order [i_L, v_o]."""
    loss = -1 / (R * C)
    if key == "buck":
        a_on = a_off = sp.Matrix([[0, -1 / L], [1 / C, loss]])
        bg_on = sp.Matrix([1 / L, 0])
        bg_off = sp.zeros(2, 1)
        a_avg = a_on
        bg = sp.Matrix([D / L, 0])
        x_dc = sp.Matrix([D * V_g / R, D * V_g])
        bd = sp.Matrix([V_g / L, 0])
        denominator = L * C * s**2 + L * s / R + 1
        h_vg = D / denominator
        h_vd = V_g / denominator
    elif key == "boost":
        a_on = sp.Matrix([[0, 0], [0, loss]])
        a_off = sp.Matrix([[0, -1 / L], [1 / C, loss]])
        bg_on = bg_off = sp.Matrix([1 / L, 0])
        a_avg = sp.Matrix([[0, -Q / L], [Q / C, loss]])
        bg = sp.Matrix([1 / L, 0])
        x_dc = sp.Matrix([V_g / (R * Q**2), V_g / Q])
        bd = sp.Matrix([V_g / (L * Q), -V_g / (R * C * Q**2)])
        denominator = L * C * s**2 + L * s / R + Q**2
        h_vg = Q / denominator
        h_vd = (V_g - L * V_g * s / (R * Q**2)) / denominator
    elif key == "buck_boost":
        a_on = sp.Matrix([[0, 0], [0, loss]])
        a_off = sp.Matrix([[0, 1 / L], [-1 / C, loss]])
        bg_on = sp.Matrix([1 / L, 0])
        bg_off = sp.zeros(2, 1)
        a_avg = sp.Matrix([[0, Q / L], [-Q / C, loss]])
        bg = sp.Matrix([D / L, 0])
        x_dc = sp.Matrix([D * V_g / (R * Q**2), -D * V_g / Q])
        bd = sp.Matrix([V_g / (L * Q), D * V_g / (R * C * Q**2)])
        denominator = L * C * s**2 + L * s / R + Q**2
        h_vg = -D * Q / denominator
        h_vd = (-V_g + L * D * V_g * s / (R * Q**2)) / denominator
    else:
        raise KeyError(key)

    return {
        "A_on": a_on,
        "A_off": a_off,
        "Bg_on": bg_on,
        "Bg_off": bg_off,
        "A": a_avg,
        "Bg": bg,
        "X_dc": x_dc,
        "Bd": bd,
        "H_vg": sp.factor(h_vg),
        "H_vd": sp.factor(h_vd),
    }


def _assert_matrix(label: str, calculated: sp.Matrix, expected: sp.Matrix) -> None:
    if not simplified_equal(calculated, expected):
        raise AssertionError(f"{label} mismatch\ncalculated={calculated}\nexpected={expected}")


def _assert_expression(label: str, calculated: sp.Expr, expected: sp.Expr) -> None:
    if sp.simplify(calculated - expected) != 0:
        raise AssertionError(f"{label} mismatch: calculated={calculated}, expected={expected}")


def _native_phase(netlist: str) -> dict[str, Any]:
    return reorder_states(descriptor_to_state_space(build_mna(netlist)), STATE_ORDER)


def _verify_native_numeric(netlist: str, symbolic: dict[str, Any]) -> None:
    values = {"L": 100e-6, "C": 220e-6, "R": 12.0}
    native = numeric_state_space(netlist, values)
    permutation = [native["states"].index(name) for name in STATE_ORDER]
    native_a = np.asarray(native["a"], dtype=float)[np.ix_(permutation, permutation)]
    native_b = np.asarray(native["b"], dtype=float)[permutation, :]
    substitutions = {L: values["L"], C: values["C"], R: values["R"]}
    sympy_a = np.asarray(symbolic["A"].subs(substitutions), dtype=float)
    sympy_b = np.asarray(symbolic["B"].subs(substitutions), dtype=float)
    if not np.allclose(native_a, sympy_a, rtol=1e-10, atol=1e-10):
        raise AssertionError("Rust and SymPy phase A matrices differ numerically")
    if not np.allclose(native_b, sympy_b, rtol=1e-10, atol=1e-10):
        raise AssertionError("Rust and SymPy phase B matrices differ numerically")


def analyze_converter(key: str) -> dict[str, Any]:
    """Calculate and independently verify one converter's SSA model."""
    spec = SPECS[key]
    on = _native_phase(spec.on_netlist)
    off = _native_phase(spec.off_netlist)
    if on["inputs"] != off["inputs"]:
        raise AssertionError("phase input vectors differ")
    vin_index = on["inputs"].index("Vin")

    a_on, a_off = on["A"], off["A"]
    b_on, b_off = on["B"], off["B"]
    bg_on = b_on[:, vin_index]
    bg_off = b_off[:, vin_index]
    a_avg = (D * a_on + Q * a_off).applyfunc(sp.factor)
    b_avg = (D * b_on + Q * b_off).applyfunc(sp.factor)
    bg = b_avg[:, vin_index]

    input_operating_point = sp.zeros(b_on.cols, 1)
    input_operating_point[vin_index] = V_g
    x_dc = (-a_avg.inv() * bg * V_g).applyfunc(sp.factor)
    bd = (
        (a_on - a_off) * x_dc + (b_on - b_off) * input_operating_point
    ).applyfunc(sp.factor)

    output = sp.Matrix([[0, 1]])
    resolvent = (s * sp.eye(2) - a_avg).inv()
    h_vg = sp.factor((output * resolvent * bg)[0])
    h_vd = sp.factor((output * resolvent * bd)[0])

    hand = _hand_model(key)
    checks = [
        ("ON topology A", a_on, hand["A_on"]),
        ("OFF topology A", a_off, hand["A_off"]),
        ("ON input vector", bg_on, hand["Bg_on"]),
        ("OFF input vector", bg_off, hand["Bg_off"]),
        ("averaged A", a_avg, hand["A"]),
        ("input-to-state vector", bg, hand["Bg"]),
        ("DC operating point", x_dc, hand["X_dc"]),
        ("control-to-state vector", bd, hand["Bd"]),
    ]
    for label, calculated, expected in checks:
        _assert_matrix(label, calculated, expected)
    _assert_expression("input-to-output transfer", h_vg, hand["H_vg"])
    _assert_expression("control-to-output transfer", h_vd, hand["H_vd"])
    _verify_native_numeric(spec.on_netlist, on)
    _verify_native_numeric(spec.off_netlist, off)

    return {
        "spec": spec,
        "states": STATE_ORDER,
        "inputs": on["inputs"],
        "A_on": a_on,
        "A_off": a_off,
        "Bg_on": bg_on,
        "Bg_off": bg_off,
        "A": a_avg,
        "Bg": bg,
        "X_dc": x_dc,
        "Bd": bd,
        "C_y": output,
        "H_vg": h_vg,
        "H_vd": h_vd,
        "hand": hand,
        "verification": [
            "Rust MNA phase matrices equal the hand-derived phase equations",
            "Rust numeric Schur reduction equals the SymPy descriptor reduction",
            "Averaged A, input vector, DC point, and duty vector equal hand derivation",
            "Input-to-output and control-to-output transfer functions simplify exactly",
        ],
    }


def summary_text(analysis: dict[str, Any]) -> str:
    """Readable notebook output for a verified analysis."""
    lines = [
        analysis["spec"].title,
        f"states: {analysis['states']}",
        f"A = {analysis['A']}",
        f"B_g = {analysis['Bg']}",
        f"B_d = {analysis['Bd']}",
        f"X_dc = {analysis['X_dc']}",
        f"G_vg(s) = {analysis['H_vg']}",
        f"G_vd(s) = {analysis['H_vd']}",
        "verification: PASS",
    ]
    return "\n".join(lines)


def analyze_all() -> dict[str, dict[str, Any]]:
    """Analyze all three converter families, raising on any discrepancy."""
    return {key: analyze_converter(key) for key in SPECS}
