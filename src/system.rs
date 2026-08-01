use std::collections::BTreeMap;

use crate::{Expression, Matrix};

/// A symbolic MNA descriptor system.
#[derive(Debug, Clone, PartialEq)]
pub struct MnaSystem {
    /// Memoryless matrix in `A*x + K*dot(x) = B*u`.
    pub a: Matrix<Expression>,
    /// Storage matrix in `A*x + K*dot(x) = B*u`.
    pub k: Matrix<Expression>,
    /// Independent-source incidence matrix.
    pub b: Matrix<Expression>,
    /// Expanded right-hand-side vector using the source values from the netlist.
    pub u: Vec<Expression>,
    /// MNA unknown names in matrix order.
    pub unknowns: Vec<String>,
    /// Independent source names in `B` column order.
    pub inputs: Vec<String>,
    /// Source values in `inputs` order.
    pub input_values: Vec<Expression>,
    /// Defaults collected from `.param` statements.
    pub parameter_defaults: BTreeMap<String, Expression>,
    /// Non-fatal build messages, primarily for deliberately ignored elements.
    pub warnings: Vec<String>,
}

impl MnaSystem {
    /// Number of MNA unknowns.
    pub fn order(&self) -> usize {
        self.unknowns.len()
    }

    /// Converts the system into strings suitable for a thin Python or WASM DTO.
    pub fn to_string_system(&self) -> StringMnaSystem {
        StringMnaSystem {
            rows: self.order(),
            inputs_count: self.inputs.len(),
            a: self.a.iter().map(ToString::to_string).collect(),
            k: self.k.iter().map(ToString::to_string).collect(),
            b: self.b.iter().map(ToString::to_string).collect(),
            u: self.u.iter().map(ToString::to_string).collect(),
            unknowns: self.unknowns.clone(),
            inputs: self.inputs.clone(),
            input_values: self.input_values.iter().map(ToString::to_string).collect(),
            parameter_defaults: self
                .parameter_defaults
                .iter()
                .map(|(name, value)| (name.clone(), value.to_string()))
                .collect(),
            warnings: self.warnings.clone(),
        }
    }
}

/// Serialization-neutral string representation for language bindings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StringMnaSystem {
    /// Number of rows and columns in `a` and `k`.
    pub rows: usize,
    /// Number of columns in `b`.
    pub inputs_count: usize,
    /// Row-major memoryless matrix.
    pub a: Vec<String>,
    /// Row-major storage matrix.
    pub k: Vec<String>,
    /// Row-major input-incidence matrix.
    pub b: Vec<String>,
    /// Expanded source vector.
    pub u: Vec<String>,
    /// Unknown names.
    pub unknowns: Vec<String>,
    /// Input names.
    pub inputs: Vec<String>,
    /// Netlist source values.
    pub input_values: Vec<String>,
    /// Netlist parameter defaults.
    pub parameter_defaults: BTreeMap<String, String>,
    /// Build warnings.
    pub warnings: Vec<String>,
}
