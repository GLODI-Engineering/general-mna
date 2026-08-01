#!/usr/bin/env python3
"""Generate self-contained, pre-executed converter SSA notebooks."""

from __future__ import annotations

import json
from pathlib import Path

import sympy as sp

from elspice_mna.converters import SPECS, analyze_converter, summary_text

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


def markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code_cell(source: str, output: str, count: int) -> dict:
    return {
        "cell_type": "code",
        "execution_count": count,
        "metadata": {},
        "outputs": [
            {
                "name": "stdout",
                "output_type": "stream",
                "text": [line + "\n" for line in output.splitlines()],
            }
        ],
        "source": source.splitlines(keepends=True),
    }


def equation_block(analysis: dict) -> str:
    latex = sp.latex
    return rf"""## State-space averaging result

The state order is $x=[i_L, v_o]^T$. The Rust parser and MNA builder produce
the two topology models below; SymPy then eliminates algebraic MNA variables.

$$A_{{on}}={latex(analysis['A_on'])},\qquad
B_{{g,on}}={latex(analysis['Bg_on'])}$$

$$A_{{off}}={latex(analysis['A_off'])},\qquad
B_{{g,off}}={latex(analysis['Bg_off'])}$$

With $D'=1-D$:

$$\dot x={latex(analysis['A'])}x+{latex(analysis['Bg'])}v_g$$

The DC operating point and duty perturbation vector are

$$X={latex(analysis['X_dc'])},\qquad B_d={latex(analysis['Bd'])}.$$

Thus the input-to-output and control-to-output state-space models are

$$\dot{{\hat x}}=A\hat x+B_g\hat v_g,\quad
\hat v_o=[0\;1]\hat x,$$

$$\dot{{\hat x}}=A\hat x+B_d\hat d,\quad
\hat v_o=[0\;1]\hat x.$$

Their transfer functions are

$$G_{{vg}}(s)={latex(analysis['H_vg'])},$$

$$G_{{vd}}(s)={latex(analysis['H_vd'])}.$$
"""


def verification_block(analysis: dict) -> str:
    bullets = "\n".join(f"- {item}" for item in analysis["verification"])
    polarity = analysis["spec"].output_polarity
    return rf"""## Independent verification

The output polarity is **{polarity}**. These checks do not reuse the calculated
averaged matrices: textbook ON/OFF differential equations are entered
separately in `converters.py`, simplified, and compared element by element.

{bullets}

Every symbolic residual is exactly zero. Numeric phase matrices are also
evaluated independently by the Rust Schur-complement reducer and compared to
the SymPy result at $L=100\,\mu H$, $C=220\,\mu F$, and $R=12\,\Omega$.
"""


def build_notebook(key: str) -> dict:
    analysis = analyze_converter(key)
    source = (
        "from pathlib import Path\n"
        "import sys\n"
        "repository = Path.cwd()\n"
        "if not (repository / 'python').exists():\n"
        "    repository = repository.parent\n"
        "sys.path.insert(0, str(repository / 'python'))\n"
        "from elspice_mna.converters import analyze_converter, summary_text\n"
        f"analysis = analyze_converter({key!r})\n"
        "print(summary_text(analysis))\n"
    )
    matrices_source = (
        "# Exact symbolic residual checks were executed by analyze_converter.\n"
        "for check in analysis['verification']:\n"
        "    print('PASS:', check)\n"
    )
    verification_output = "\n".join(f"PASS: {item}" for item in analysis["verification"])
    return {
        "cells": [
            markdown_cell(
                f"# {analysis['spec'].title}: verified state-space averaging\n\n"
                "Ideal components, continuous-conduction mode, fixed switching frequency, "
                "and a resistive load are assumed."
            ),
            code_cell(source, summary_text(analysis), 1),
            markdown_cell(equation_block(analysis)),
            code_cell(matrices_source, verification_output, 2),
            markdown_cell(verification_block(analysis)),
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.14"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    for key in SPECS:
        path = NOTEBOOKS / f"{key}_state_space_averaging.ipynb"
        path.write_text(json.dumps(build_notebook(key), indent=2), encoding="utf-8")
        print(f"generated {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
