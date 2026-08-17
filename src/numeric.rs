use std::collections::BTreeMap;
use std::fmt;

use crate::{EvaluationError, Matrix, MnaSystem};

/// Numerically evaluated descriptor system.
#[derive(Debug, Clone, PartialEq)]
pub struct NumericMnaSystem {
    /// Memoryless matrix.
    pub a: Matrix<f64>,
    /// Storage matrix.
    pub k: Matrix<f64>,
    /// Independent-source incidence matrix.
    pub b: Matrix<f64>,
    /// Unknown names.
    pub unknowns: Vec<String>,
    /// Input names.
    pub inputs: Vec<String>,
    /// Source values in `inputs` order, evaluated the same way `a`/`k`/`b` are. Unlike
    /// [`crate::MnaSystem::u`] (baked once, symbolically, at build time from each netlist
    /// source's literal value), this is recomputed by every [`MnaSystem::evaluate`] call —
    /// which matters for a source whose value genuinely changes between evaluations, such as
    /// a piecewise-linear device's Norton-equivalent current (`{name}_Ioff`), rather than a
    /// netlist-declared `V`/`I` source whose value never changes after parsing.
    ///
    /// Evaluating an input's value is best-effort, not required: a source whose symbol has no
    /// entry in `values` (e.g. a duty ratio or `Vin` a caller only cares about symbolically,
    /// while evaluating unrelated structural matrices) gets `f64::NAN` here rather than
    /// failing the whole [`MnaSystem::evaluate`] call, matching this method's existing
    /// behavior for `a`/`k`/`b` requiring only the symbols the caller actually supplied. `u`
    /// below treats a `NAN` input as contributing `0.0`, not `NAN`, to every row.
    pub input_values: Vec<f64>,
    /// The expanded right-hand side `B * input_values`, in unknown order — the numeric twin
    /// of [`crate::MnaSystem::u`], computed fresh from `input_values` above rather than from
    /// the netlist's original (possibly stale) literal values. Rows depending only on inputs
    /// that evaluated successfully are exact; see `input_values` above for the unresolved case.
    pub u: Vec<f64>,
}

/// Explicit state-space system `dot(x) = A*x + B*u`.
#[derive(Debug, Clone, PartialEq)]
pub struct NumericStateSpace {
    /// Dynamics matrix.
    pub a: Matrix<f64>,
    /// Input matrix.
    pub b: Matrix<f64>,
    /// Dynamic variable names.
    pub states: Vec<String>,
    /// Independent input names.
    pub inputs: Vec<String>,
    /// Indices of the dynamic variables in the original MNA vector.
    pub mna_state_indices: Vec<usize>,
}

impl MnaSystem {
    /// Evaluates component expressions using explicit values plus numeric
    /// `.param` defaults from the netlist.
    pub fn evaluate(
        &self,
        values: &BTreeMap<String, f64>,
    ) -> Result<NumericMnaSystem, EvaluationError> {
        let mut environment = BTreeMap::new();
        for (name, expression) in &self.parameter_defaults {
            if let Ok(value) = expression.evaluate(values) {
                environment.insert(name.clone(), value);
            }
        }
        environment.extend(values.iter().map(|(name, value)| (name.clone(), *value)));

        let b = evaluate_matrix(&self.b, &environment)?;
        let input_values: Vec<f64> = self
            .input_values
            .iter()
            .map(|expression| expression.evaluate(&environment).unwrap_or(f64::NAN))
            .collect();
        let order = self.unknowns.len();
        let mut u = vec![0.0; order];
        for row in 0..order {
            for (column, input_value) in input_values.iter().enumerate() {
                if !input_value.is_nan() {
                    u[row] += b[(row, column)] * input_value;
                }
            }
        }

        Ok(NumericMnaSystem {
            a: evaluate_matrix(&self.a, &environment)?,
            k: evaluate_matrix(&self.k, &environment)?,
            b,
            unknowns: self.unknowns.clone(),
            inputs: self.inputs.clone(),
            input_values,
            u,
        })
    }
}

impl NumericMnaSystem {
    /// Eliminates algebraic variables with a Schur complement.
    ///
    /// Dynamic variables are the MNA indices whose row in `K` is nonzero.
    /// This matches capacitor-node voltages and inductor branch currents for
    /// the linear stamps produced by this crate.
    pub fn to_state_space(&self, tolerance: f64) -> Result<NumericStateSpace, StateSpaceError> {
        let order = self.unknowns.len();
        if self.a.rows() != order
            || self.a.cols() != order
            || self.k.rows() != order
            || self.k.cols() != order
            || self.b.rows() != order
        {
            return Err(StateSpaceError::InconsistentDimensions);
        }

        let state_indices: Vec<usize> = (0..order)
            .filter(|row| (0..order).any(|col| self.k[(*row, col)].abs() > tolerance))
            .collect();
        if state_indices.is_empty() {
            return Err(StateSpaceError::NoDynamicVariables);
        }
        let algebraic_indices: Vec<usize> = (0..order)
            .filter(|index| !state_indices.contains(index))
            .collect();

        let a_ss = extract(&self.a, &state_indices, &state_indices);
        let k_ss = extract(&self.k, &state_indices, &state_indices);
        let b_s = extract_rows(&self.b, &state_indices);

        let (a_reduced, b_reduced) = if algebraic_indices.is_empty() {
            (a_ss, b_s)
        } else {
            let a_sa = extract(&self.a, &state_indices, &algebraic_indices);
            let a_as = extract(&self.a, &algebraic_indices, &state_indices);
            let a_aa = extract(&self.a, &algebraic_indices, &algebraic_indices);
            let b_a = extract_rows(&self.b, &algebraic_indices);
            let solved_as = solve(&a_aa, &a_as, tolerance).map_err(|pivot| {
                StateSpaceError::singular_algebraic(&self.unknowns, &algebraic_indices, pivot)
            })?;
            let solved_b = solve(&a_aa, &b_a, tolerance).map_err(|pivot| {
                StateSpaceError::singular_algebraic(&self.unknowns, &algebraic_indices, pivot)
            })?;
            (
                subtract(&a_ss, &multiply(&a_sa, &solved_as)),
                subtract(&b_s, &multiply(&a_sa, &solved_b)),
            )
        };

        let negative_a = a_reduced.map(|value| -*value);
        let state_a = solve(&k_ss, &negative_a, tolerance).map_err(|pivot| {
            StateSpaceError::singular_storage(&self.unknowns, &state_indices, pivot)
        })?;
        let state_b = solve(&k_ss, &b_reduced, tolerance).map_err(|pivot| {
            StateSpaceError::singular_storage(&self.unknowns, &state_indices, pivot)
        })?;

        Ok(NumericStateSpace {
            a: state_a,
            b: state_b,
            states: state_indices
                .iter()
                .map(|index| self.unknowns[*index].clone())
                .collect(),
            inputs: self.inputs.clone(),
            mna_state_indices: state_indices,
        })
    }
}

fn evaluate_matrix(
    matrix: &Matrix<crate::Expression>,
    values: &BTreeMap<String, f64>,
) -> Result<Matrix<f64>, EvaluationError> {
    let data = matrix
        .iter()
        .map(|expression| expression.evaluate(values))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Matrix::from_vec(matrix.rows(), matrix.cols(), data)
        .expect("mapping a matrix preserves its shape"))
}

fn extract(matrix: &Matrix<f64>, rows: &[usize], cols: &[usize]) -> Matrix<f64> {
    let mut result = Matrix::filled(rows.len(), cols.len(), 0.0);
    for (target_row, source_row) in rows.iter().enumerate() {
        for (target_col, source_col) in cols.iter().enumerate() {
            result[(target_row, target_col)] = matrix[(*source_row, *source_col)];
        }
    }
    result
}

fn extract_rows(matrix: &Matrix<f64>, rows: &[usize]) -> Matrix<f64> {
    let columns: Vec<usize> = (0..matrix.cols()).collect();
    extract(matrix, rows, &columns)
}

fn multiply(lhs: &Matrix<f64>, rhs: &Matrix<f64>) -> Matrix<f64> {
    assert_eq!(lhs.cols(), rhs.rows());
    let mut result = Matrix::filled(lhs.rows(), rhs.cols(), 0.0);
    for row in 0..lhs.rows() {
        for col in 0..rhs.cols() {
            result[(row, col)] = (0..lhs.cols())
                .map(|inner| lhs[(row, inner)] * rhs[(inner, col)])
                .sum();
        }
    }
    result
}

fn subtract(lhs: &Matrix<f64>, rhs: &Matrix<f64>) -> Matrix<f64> {
    assert_eq!((lhs.rows(), lhs.cols()), (rhs.rows(), rhs.cols()));
    Matrix::from_vec(
        lhs.rows(),
        lhs.cols(),
        lhs.iter()
            .zip(rhs.iter())
            .map(|(left, right)| left - right)
            .collect(),
    )
    .expect("zipped equal matrices preserve shape")
}

/// Solves `coefficients * x = rhs` by Gauss-Jordan elimination with partial
/// pivoting. On failure, returns the index (within `coefficients`) of the
/// column that had no usable pivot, so callers can report which unknown the
/// singularity localizes to.
fn solve(
    coefficients: &Matrix<f64>,
    rhs: &Matrix<f64>,
    tolerance: f64,
) -> Result<Matrix<f64>, usize> {
    let n = coefficients.rows();
    if coefficients.cols() != n || rhs.rows() != n {
        return Err(0);
    }
    let rhs_cols = rhs.cols();
    let mut augmented = Matrix::filled(n, n + rhs_cols, 0.0);
    for row in 0..n {
        for col in 0..n {
            augmented[(row, col)] = coefficients[(row, col)];
        }
        for col in 0..rhs_cols {
            augmented[(row, n + col)] = rhs[(row, col)];
        }
    }

    for pivot_col in 0..n {
        let pivot_row = (pivot_col..n)
            .max_by(|left, right| {
                augmented[(*left, pivot_col)]
                    .abs()
                    .total_cmp(&augmented[(*right, pivot_col)].abs())
            })
            .ok_or(pivot_col)?;
        if augmented[(pivot_row, pivot_col)].abs() <= tolerance {
            return Err(pivot_col);
        }
        if pivot_row != pivot_col {
            for col in 0..n + rhs_cols {
                let temporary = augmented[(pivot_col, col)];
                augmented[(pivot_col, col)] = augmented[(pivot_row, col)];
                augmented[(pivot_row, col)] = temporary;
            }
        }

        let pivot = augmented[(pivot_col, pivot_col)];
        for col in pivot_col..n + rhs_cols {
            augmented[(pivot_col, col)] /= pivot;
        }
        for row in 0..n {
            if row == pivot_col {
                continue;
            }
            let factor = augmented[(row, pivot_col)];
            for col in pivot_col..n + rhs_cols {
                augmented[(row, col)] -= factor * augmented[(pivot_col, col)];
            }
        }
    }

    let mut result = Matrix::filled(n, rhs_cols, 0.0);
    for row in 0..n {
        for col in 0..rhs_cols {
            result[(row, col)] = augmented[(row, n + col)];
        }
    }
    Ok(result)
}

/// Error returned during descriptor-to-state-space reduction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StateSpaceError {
    /// Matrix and name dimensions do not agree.
    InconsistentDimensions,
    /// No row of the storage matrix is dynamic.
    NoDynamicVariables,
    /// Algebraic constraints cannot be uniquely eliminated.
    SingularAlgebraicBlock {
        /// The algebraic unknown whose equation had no usable pivot.
        unknown: String,
        /// All algebraic unknowns considered in the same elimination.
        block: Vec<String>,
    },
    /// The reduced storage (capacitor/inductor) block is singular: the
    /// circuit has fewer independent dynamic states than reactive elements.
    SingularStorageBlock {
        /// The state variable that turned out not to be independent.
        unknown: String,
        /// All capacitor-voltage/inductor-current states in the same block.
        block: Vec<String>,
    },
}

impl StateSpaceError {
    pub(crate) fn singular_algebraic(unknowns: &[String], indices: &[usize], pivot: usize) -> Self {
        Self::SingularAlgebraicBlock {
            unknown: unknowns[indices[pivot]].clone(),
            block: indices
                .iter()
                .map(|index| unknowns[*index].clone())
                .collect(),
        }
    }

    pub(crate) fn singular_storage(unknowns: &[String], indices: &[usize], pivot: usize) -> Self {
        Self::SingularStorageBlock {
            unknown: unknowns[indices[pivot]].clone(),
            block: indices
                .iter()
                .map(|index| unknowns[*index].clone())
                .collect(),
        }
    }
}

impl fmt::Display for StateSpaceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InconsistentDimensions => f.write_str("inconsistent MNA matrix dimensions"),
            Self::NoDynamicVariables => {
                f.write_str("the circuit has no capacitor or inductor state")
            }
            Self::SingularAlgebraicBlock { unknown, block } => write!(
                f,
                "no unique solution for {unknown}: the algebraic (non-storage) equations for {{{}}} are linearly dependent — check for a loop of ideal voltage sources/VCVS elements or a cut-set of ideal current sources/VCCS elements among them",
                block.join(", "),
            ),
            Self::SingularStorageBlock { unknown, block } => {
                let hint = if unknown.starts_with("V(") {
                    "a capacitor-only loop with no resistive path, so its voltage is not independent of the others — add a series resistor to break the loop"
                } else {
                    "an inductor-only cut-set with no resistive path, so its current is not independent of the others (or a redundant coupled-inductor `K` specification) — add a parallel resistor or check the coupling coefficients"
                };
                write!(
                    f,
                    "{unknown} is not an independent state variable among {{{}}}: likely {hint}",
                    block.join(", "),
                )
            }
        }
    }
}

impl std::error::Error for StateSpaceError {}
