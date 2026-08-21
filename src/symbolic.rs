use std::fmt;

use crate::numeric::StateSpaceError;
use crate::{Expression, Matrix, MnaSystem};

/// Explicit symbolic state-space system `dot(x) = A*x + B*u`.
///
/// Unlike [`crate::NumericStateSpace`], entries are kept as [`Expression`]
/// trees in terms of component/parameter symbols only (never evaluated to
/// `f64`, and never containing the Laplace variable `s`). This is the
/// symbolic twin of [`crate::NumericMnaSystem::to_state_space`]: the same
/// Schur-complement elimination of the algebraic MNA rows, over `Expression`
/// arithmetic instead of floating point.
#[derive(Debug, Clone, PartialEq)]
pub struct SymbolicStateSpace {
    /// Dynamics matrix, in terms of component symbols only.
    pub a: Matrix<Expression>,
    /// Input matrix, in terms of component symbols only.
    pub b: Matrix<Expression>,
    /// Dynamic variable names (capacitor voltages / inductor currents).
    pub states: Vec<String>,
    /// Independent input names.
    pub inputs: Vec<String>,
    /// Indices of the dynamic variables in the original MNA vector.
    pub mna_state_indices: Vec<usize>,
}

impl MnaSystem {
    /// Eliminates algebraic variables with a Schur complement, entirely
    /// symbolically: component values are never evaluated to numbers.
    ///
    /// Dynamic variables are the MNA indices whose row in `K` is nonzero,
    /// exactly as in the numeric reduction. Pivoting is *structural*: at each
    /// elimination step we pick the first candidate row whose entry is not
    /// the literal expression `0` (there is no notion of "numerically small"
    /// for a symbolic tree). This reliably catches rows that were never
    /// stamped at all (e.g. a disconnected node), but it cannot catch a
    /// combination that is only zero because of how *specific* component
    /// symbols happen to cancel (e.g. a capacitor-only loop) — `Expression`
    /// does not combine like symbolic terms. That case instead surfaces as
    /// [`crate::EvaluationError::DivisionByZero`] when a coefficient is later
    /// evaluated at concrete parameter values, which is an acceptable and
    /// correctly-reported place to find it.
    pub fn to_symbolic_state_space(&self) -> Result<SymbolicStateSpace, StateSpaceError> {
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
            .filter(|row| (0..order).any(|col| !self.k[(*row, col)].is_zero()))
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
            let solved_as = solve(&a_aa, &a_as).map_err(|pivot| {
                StateSpaceError::singular_algebraic(&self.unknowns, &algebraic_indices, pivot)
            })?;
            let solved_b = solve(&a_aa, &b_a).map_err(|pivot| {
                StateSpaceError::singular_algebraic(&self.unknowns, &algebraic_indices, pivot)
            })?;
            (
                subtract(&a_ss, &multiply(&a_sa, &solved_as)),
                subtract(&b_s, &multiply(&a_sa, &solved_b)),
            )
        };

        let negative_a = a_reduced.map(|value| -value.clone());
        let state_a = solve(&k_ss, &negative_a).map_err(|pivot| {
            StateSpaceError::singular_storage(&self.unknowns, &state_indices, pivot)
        })?;
        let state_b = solve(&k_ss, &b_reduced).map_err(|pivot| {
            StateSpaceError::singular_storage(&self.unknowns, &state_indices, pivot)
        })?;

        Ok(SymbolicStateSpace {
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

/// A rational transfer function `G(s) = numerator(s) / denominator(s)`
/// between one reduced state variable and one independent source.
///
/// Coefficients are `Expression` trees in terms of component symbols only;
/// `s` never appears inside them; it is the polynomial variable the
/// coefficient lists themselves are indexed against.
#[derive(Debug, Clone, PartialEq)]
pub struct SymbolicTransferFunction {
    /// Numerator coefficients, descending powers of `s`. Length equals the
    /// state count (one degree lower than the denominator, since a reduced
    /// state-space realization has no direct feedthrough term).
    pub numerator: Vec<Expression>,
    /// Denominator (characteristic polynomial `det(sI - A)`) coefficients,
    /// descending powers of `s`, leading coefficient `1`. Length is the state
    /// count plus one.
    pub denominator: Vec<Expression>,
    /// The chosen output state name.
    pub output: String,
    /// The chosen independent source name.
    pub input: String,
}

/// Error returned when deriving a transfer function from a
/// [`SymbolicStateSpace`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferFunctionError {
    /// The requested output state index is out of range.
    InvalidOutputIndex(usize),
    /// The requested input index is out of range.
    InvalidInputIndex(usize),
}

impl fmt::Display for TransferFunctionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidOutputIndex(index) => {
                write!(f, "state index {index} is out of range")
            }
            Self::InvalidInputIndex(index) => {
                write!(f, "input index {index} is out of range")
            }
        }
    }
}

impl std::error::Error for TransferFunctionError {}

impl SymbolicStateSpace {
    /// Derives `G(s) = C (sI - A)^-1 B` between `states[output_index]` and
    /// `inputs[input_index]` using the Faddeev-LeVerrier recursion
    /// ([`faddeev_leverrier`]), so it scales to any state count without the
    /// symbolic Gaussian elimination and pivoting a direct matrix inverse
    /// would need.
    pub fn transfer_function(
        &self,
        output_index: usize,
        input_index: usize,
    ) -> Result<SymbolicTransferFunction, TransferFunctionError> {
        let n = self.states.len();
        if output_index >= n {
            return Err(TransferFunctionError::InvalidOutputIndex(output_index));
        }
        if input_index >= self.inputs.len() {
            return Err(TransferFunctionError::InvalidInputIndex(input_index));
        }

        let (denominator, adjugate_terms) = faddeev_leverrier(&self.a);
        let numerator = adjugate_terms
            .iter()
            .map(|term| {
                (0..n).fold(Expression::zero(), |sum, column| {
                    let value = term[(output_index, column)].clone()
                        * self.b[(column, input_index)].clone();
                    if value.is_zero() {
                        sum
                    } else {
                        sum + value
                    }
                })
            })
            .collect();

        Ok(SymbolicTransferFunction {
            numerator,
            denominator,
            output: self.states[output_index].clone(),
            input: self.inputs[input_index].clone(),
        })
    }
}

/// Computes the characteristic polynomial and adjugate of `sI - a` for an
/// arbitrary `n x n` symbolic matrix, via the Faddeev-LeVerrier recursion.
///
/// This needs only matrix multiplication, addition, and division by the
/// small integers `1..=n` — never symbolic Gaussian elimination or pivoting —
/// so it scales to any state count, unlike expanding `det(sI - A)` by
/// cofactors (factorial blow-up) or eliminating symbolically (which needs a
/// pivoting strategy `Expression` cannot support in general).
///
/// Returns `(coefficients, adjugate_terms)`:
/// - `coefficients[0..=n]`: `det(sI - a) = s^n + coefficients[1] s^(n-1) +
///   ... + coefficients[n]`, descending powers of `s`, `coefficients[0] ==
///   1`.
/// - `adjugate_terms[0..n]`: `adj(sI - a) = adjugate_terms[0] s^(n-1) + ... +
///   adjugate_terms[n-1]`, descending powers of `s`.
pub fn faddeev_leverrier(a: &Matrix<Expression>) -> (Vec<Expression>, Vec<Matrix<Expression>>) {
    let n = a.rows();
    assert_eq!(a.cols(), n, "faddeev_leverrier requires a square matrix");

    let mut m = zeros(n);
    let mut c = Expression::one();
    let mut coefficients = vec![c.clone()];
    let mut adjugate_terms = Vec::with_capacity(n);

    for k in 1..=n {
        let scaled_identity = scalar_multiply(&identity(n), c);
        m = add(&multiply(a, &m), &scaled_identity);
        let trace_term = trace(&multiply(a, &m));
        c = trace_term * Expression::Constant(-1.0 / k as f64);
        coefficients.push(c.clone());
        adjugate_terms.push(m.clone());
    }

    (coefficients, adjugate_terms)
}

fn identity(n: usize) -> Matrix<Expression> {
    let mut result = Matrix::filled(n, n, Expression::zero());
    for index in 0..n {
        result[(index, index)] = Expression::one();
    }
    result
}

fn zeros(n: usize) -> Matrix<Expression> {
    Matrix::filled(n, n, Expression::zero())
}

fn scalar_multiply(matrix: &Matrix<Expression>, scalar: Expression) -> Matrix<Expression> {
    matrix.map(|value| value.clone() * scalar.clone())
}

fn trace(matrix: &Matrix<Expression>) -> Expression {
    assert_eq!(matrix.rows(), matrix.cols());
    (0..matrix.rows()).fold(Expression::zero(), |sum, index| {
        let value = matrix[(index, index)].clone();
        if value.is_zero() {
            sum
        } else {
            sum + value
        }
    })
}

fn extract(matrix: &Matrix<Expression>, rows: &[usize], cols: &[usize]) -> Matrix<Expression> {
    let mut result = Matrix::filled(rows.len(), cols.len(), Expression::zero());
    for (target_row, source_row) in rows.iter().enumerate() {
        for (target_col, source_col) in cols.iter().enumerate() {
            result[(target_row, target_col)] = matrix[(*source_row, *source_col)].clone();
        }
    }
    result
}

fn extract_rows(matrix: &Matrix<Expression>, rows: &[usize]) -> Matrix<Expression> {
    let columns: Vec<usize> = (0..matrix.cols()).collect();
    extract(matrix, rows, &columns)
}

fn multiply(lhs: &Matrix<Expression>, rhs: &Matrix<Expression>) -> Matrix<Expression> {
    assert_eq!(lhs.cols(), rhs.rows());
    let mut result = Matrix::filled(lhs.rows(), rhs.cols(), Expression::zero());
    for row in 0..lhs.rows() {
        for col in 0..rhs.cols() {
            let sum = (0..lhs.cols()).fold(Expression::zero(), |sum, inner| {
                let term = lhs[(row, inner)].clone() * rhs[(inner, col)].clone();
                if term.is_zero() {
                    sum
                } else {
                    sum + term
                }
            });
            result[(row, col)] = sum;
        }
    }
    result
}

fn add(lhs: &Matrix<Expression>, rhs: &Matrix<Expression>) -> Matrix<Expression> {
    assert_eq!((lhs.rows(), lhs.cols()), (rhs.rows(), rhs.cols()));
    Matrix::from_vec(
        lhs.rows(),
        lhs.cols(),
        lhs.iter()
            .zip(rhs.iter())
            .map(|(left, right)| left.clone() + right.clone())
            .collect(),
    )
    .expect("zipped equal matrices preserve shape")
}

fn subtract(lhs: &Matrix<Expression>, rhs: &Matrix<Expression>) -> Matrix<Expression> {
    assert_eq!((lhs.rows(), lhs.cols()), (rhs.rows(), rhs.cols()));
    Matrix::from_vec(
        lhs.rows(),
        lhs.cols(),
        lhs.iter()
            .zip(rhs.iter())
            .map(|(left, right)| left.clone() - right.clone())
            .collect(),
    )
    .expect("zipped equal matrices preserve shape")
}

/// Solves `coefficients * x = rhs` by Gauss-Jordan elimination, symbolically.
///
/// Pivoting is structural: the first row (from the current column downward)
/// whose entry is not the literal expression `0`. On failure, returns the
/// column index that had no such candidate.
fn solve(
    coefficients: &Matrix<Expression>,
    rhs: &Matrix<Expression>,
) -> Result<Matrix<Expression>, usize> {
    let n = coefficients.rows();
    if coefficients.cols() != n || rhs.rows() != n {
        return Err(0);
    }
    let rhs_cols = rhs.cols();
    let mut augmented = Matrix::filled(n, n + rhs_cols, Expression::zero());
    for row in 0..n {
        for col in 0..n {
            augmented[(row, col)] = coefficients[(row, col)].clone();
        }
        for col in 0..rhs_cols {
            augmented[(row, n + col)] = rhs[(row, col)].clone();
        }
    }

    for pivot_col in 0..n {
        let pivot_row = (pivot_col..n).find(|row| !augmented[(*row, pivot_col)].is_zero());
        let pivot_row = pivot_row.ok_or(pivot_col)?;

        if pivot_row != pivot_col {
            for col in 0..n + rhs_cols {
                let temporary = augmented[(pivot_col, col)].clone();
                augmented[(pivot_col, col)] = augmented[(pivot_row, col)].clone();
                augmented[(pivot_row, col)] = temporary;
            }
        }

        let inverse_pivot = augmented[(pivot_col, pivot_col)].clone().reciprocal();
        for col in pivot_col..n + rhs_cols {
            augmented[(pivot_col, col)] =
                augmented[(pivot_col, col)].clone() * inverse_pivot.clone();
        }
        for row in 0..n {
            if row == pivot_col {
                continue;
            }
            let factor = augmented[(row, pivot_col)].clone();
            if factor.is_zero() {
                continue;
            }
            for col in pivot_col..n + rhs_cols {
                let term = factor.clone() * augmented[(pivot_col, col)].clone();
                augmented[(row, col)] = augmented[(row, col)].clone() - term;
            }
        }
    }

    let mut result = Matrix::filled(n, rhs_cols, Expression::zero());
    for row in 0..n {
        for col in 0..rhs_cols {
            result[(row, col)] = augmented[(row, n + col)].clone();
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn solve_matches_hand_derived_answer_for_voltage_divider_block() {
        // a_aa = [[1/Rs, -1/Rs, 1], [-1/Rs, 1/Rs, 0], [1, 0, 0]]
        // rhs column = [0, 1, 0]^T
        // Hand solution: x1=0, x2=Rs, x3=1.
        //
        // Regression test: structural (exact-zero) pivoting alone can pick
        // row 1 at the second pivot column, whose entry is only zero after
        // the first elimination step combines two sub-expressions that look
        // different syntactically (e.g. `1/Rs` and a derived product that is
        // also `1/Rs` once simplified). Without `Expression`'s like-term
        // combination this silently produces a wrong (but finite) answer
        // instead of an error.
        let rs = Expression::symbol("Rs");
        let g = rs.reciprocal();

        let a_aa = Matrix::from_vec(
            3,
            3,
            vec![
                g.clone(),
                -g.clone(),
                Expression::one(),
                -g.clone(),
                g.clone(),
                Expression::zero(),
                Expression::one(),
                Expression::zero(),
                Expression::zero(),
            ],
        )
        .unwrap();
        let rhs = Matrix::from_vec(
            3,
            1,
            vec![Expression::zero(), Expression::one(), Expression::zero()],
        )
        .unwrap();

        let solved = solve(&a_aa, &rhs).unwrap();
        let values = BTreeMap::from([("Rs".to_string(), 2.5)]);
        let x1 = solved[(0, 0)].evaluate(&values).unwrap();
        let x2 = solved[(1, 0)].evaluate(&values).unwrap();
        let x3 = solved[(2, 0)].evaluate(&values).unwrap();
        assert!((x1 - 0.0).abs() < 1e-9, "x1={x1}");
        assert!((x2 - 2.5).abs() < 1e-9, "x2={x2}");
        assert!((x3 - 1.0).abs() < 1e-9, "x3={x3}");
    }
}
