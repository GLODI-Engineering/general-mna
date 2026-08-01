"""Python interface to the Rust MNA core and SymPy reduction helpers."""

from ._native import build_mna, numeric_state_space
from .symbolic import descriptor_to_state_space

__all__ = ["build_mna", "descriptor_to_state_space", "numeric_state_space"]

