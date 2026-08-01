import unittest

import sympy as sp

from elspice_mna import build_mna, numeric_state_space
from elspice_mna.converters import SPECS, analyze_converter


class BindingTests(unittest.TestCase):
    def test_build_mna_returns_symbolic_descriptor(self):
        model = build_mna("V1 in 0 1\nR1 in out R\nC1 out 0 C")
        self.assertEqual(model["unknowns"], ["V(in)", "V(out)", "I(V1)"])
        self.assertEqual(model["inputs"], ["V1"])
        self.assertEqual(model["k"][1][1], "C")

    def test_native_numeric_state_space(self):
        model = numeric_state_space(
            "V1 in 0 1\nR1 in out R\nC1 out 0 C",
            {"R": 1000.0, "C": 1e-6},
        )
        self.assertEqual(model["states"], ["V(out)"])
        self.assertAlmostEqual(model["a"][0][0], -1000.0)
        self.assertAlmostEqual(model["b"][0][0], 1000.0)


class ConverterVerificationTests(unittest.TestCase):
    def test_all_converter_models_match_independent_derivations(self):
        for key in SPECS:
            with self.subTest(converter=key):
                result = analyze_converter(key)
                self.assertEqual(sp.simplify(result["H_vg"] - result["hand"]["H_vg"]), 0)
                self.assertEqual(sp.simplify(result["H_vd"] - result["hand"]["H_vd"]), 0)


if __name__ == "__main__":
    unittest.main()

