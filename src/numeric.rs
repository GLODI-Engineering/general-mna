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

        Ok(NumericMnaSystem {
            a: evaluate_matrix(&self.a, &environment)?,
            k: evaluate_matrix(&self.k, &environment)?,
            b: evaluate_matrix(&self.b, &environment)?,
            unknowns: self.unknowns.clone(),
            inputs: self.inputs.clone(),
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
            let solved_as = solve(&a_aa, &a_as, tolerance)
                .map_err(|_| StateSpaceError::SingularAlgebraicBlock)?;
            let solved_b = solve(&a_aa, &b_a, tolerance)
                .map_err(|_| StateSpaceError::SingularAlgebraicBlock)?;
            (
                subtract(&a_ss, &multiply(&a_sa, &solved_as)),
                subtract(&b_s, &multiply(&a_sa, &solved_b)),
            )
        };

        let negative_a = a_reduced.map(|value| -*value);
        let state_a = solve(&k_ss, &negative_a, tolerance)
            .map_err(|_| StateSpaceError::SingularStorageBlock)?;
        let state_b = solve(&k_ss, &b_reduced, tolerance)
            .map_err(|_| StateSpaceError::SingularStorageBlock)?;

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

fn solve(coefficients: &Matrix<f64>, rhs: &Matrix<f64>, tolerance: f64) -> Result<Matrix<f64>, ()> {
    let n = coefficients.rows();
    if coefficients.cols() != n || rhs.rows() != n {
        return Err(());
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
            .ok_or(())?;
        if augmented[(pivot_row, pivot_col)].abs() <= tolerance {
            return Err(());
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StateSpaceError {
    /// Matrix and name dimensions do not agree.
    InconsistentDimensions,
    /// No row of the storage matrix is dynamic.
    NoDynamicVariables,
    /// Algebraic constraints cannot be uniquely eliminated.
    SingularAlgebraicBlock,
    /// The reduced storage matrix is singular.
    SingularStorageBlock,
}

impl fmt::Display for StateSpaceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InconsistentDimensions => f.write_str("inconsistent MNA matrix dimensions"),
            Self::NoDynamicVariables => {
                f.write_str("the circuit has no capacitor or inductor state")
            }
            Self::SingularAlgebraicBlock => f.write_str("the algebraic MNA block is singular"),
            Self::SingularStorageBlock => f.write_str("the reduced storage block is singular"),
        }
    }
}

impl std::error::Error for StateSpaceError {}
