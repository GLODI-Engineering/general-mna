"""SymPy conversion and descriptor reduction for binding DTOs."""

from __future__ import annotations

from typing import Any, Iterable

import sympy as sp


def expression(text: str) -> sp.Expr:
    """Parse the Rust expression display syntax as a SymPy expression."""
    return sp.sympify(text, locals={"sqrt": sp.sqrt})


def matrix(rows: Iterable[Iterable[str]]) -> sp.Matrix:
    """Convert nested string rows returned by the native binding."""
    return sp.Matrix([[expression(value) for value in row] for row in rows])


def descriptor_to_state_space(mna: dict[str, Any]) -> dict[str, Any]:
    """Symbolically eliminate algebraic variables from an MNA DTO.

    The input equation is ``A*x + K*dot(x) = B*u``. Dynamic coordinates are
    selected from nonzero rows of K, matching the Rust numeric reducer.
    """
    a = matrix(mna["a"])
    k = matrix(mna["k"])
    b = matrix(mna["b"])
    order = a.rows
    state_indices = [
        row
        for row in range(order)
        if any(sp.simplify(k[row, column]) != 0 for column in range(order))
    ]
    algebraic_indices = [index for index in range(order) if index not in state_indices]
    if not state_indices:
        raise ValueError("descriptor has no dynamic storage rows")

    a_ss = a.extract(state_indices, state_indices)
    k_ss = k.extract(state_indices, state_indices)
    b_s = b.extract(state_indices, range(b.cols))

    if algebraic_indices:
        a_sa = a.extract(state_indices, algebraic_indices)
        a_as = a.extract(algebraic_indices, state_indices)
        a_aa = a.extract(algebraic_indices, algebraic_indices)
        b_a = b.extract(algebraic_indices, range(b.cols))
        a_reduced = a_ss - a_sa * a_aa.inv() * a_as
        b_reduced = b_s - a_sa * a_aa.inv() * b_a
    else:
        a_reduced = a_ss
        b_reduced = b_s

    state_a = (-k_ss.inv() * a_reduced).applyfunc(sp.factor)
    state_b = (k_ss.inv() * b_reduced).applyfunc(sp.factor)
    return {
        "A": state_a,
        "B": state_b,
        "states": [mna["unknowns"][index] for index in state_indices],
        "inputs": list(mna["inputs"]),
        "mna_state_indices": state_indices,
    }


def reorder_states(model: dict[str, Any], desired: list[str]) -> dict[str, Any]:
    """Reorder an explicit state model and both matrix axes consistently."""
    if set(model["states"]) != set(desired):
        raise ValueError(f"states {model['states']} do not match requested order {desired}")
    permutation = [model["states"].index(name) for name in desired]
    return {
        **model,
        "A": model["A"].extract(permutation, permutation),
        "B": model["B"].extract(permutation, range(model["B"].cols)),
        "states": desired,
    }


def simplified_equal(left: sp.Matrix, right: sp.Matrix) -> bool:
    """Return true when two symbolic matrices are elementwise identical."""
    return left.shape == right.shape and all(
        sp.simplify(left[row, column] - right[row, column]) == 0
        for row in range(left.rows)
        for column in range(left.cols)
    )
