use std::collections::BTreeMap;
use std::fmt;

use crate::{Expression, Matrix, MnaSystem};

/// One topology and its fraction of the switching period.
#[derive(Debug, Clone, Copy)]
pub struct WeightedPhase<'a> {
    /// Phase MNA system.
    pub system: &'a MnaSystem,
    /// Phase duration divided by switching period.
    pub weight: &'a Expression,
}

/// Produces the weighted descriptor model of converter phases.
///
/// For two phases, pass weights `D` and `1-D`. Resistive switch models keep
/// phase dimensions identical and make descriptor averaging straightforward.
pub fn average(phases: &[WeightedPhase<'_>]) -> Result<MnaSystem, AveragingError> {
    let first = phases.first().ok_or(AveragingError::NoPhases)?.system;
    for phase in phases.iter().skip(1) {
        ensure_compatible(first, phase.system)?;
    }

    let a = weighted_matrix(phases, |system| &system.a);
    let k = weighted_matrix(phases, |system| &system.k);
    let b = weighted_matrix(phases, |system| &system.b);
    let mut u = vec![Expression::zero(); first.order()];
    for phase in phases {
        for (row, value) in phase.system.u.iter().enumerate() {
            u[row] = u[row].clone() + phase.weight.clone() * value.clone();
        }
    }

    let mut parameter_defaults = BTreeMap::new();
    let mut transient_sources = BTreeMap::new();
    let mut warnings = Vec::new();
    for (phase_index, phase) in phases.iter().enumerate() {
        parameter_defaults.extend(phase.system.parameter_defaults.clone());
        transient_sources.extend(phase.system.transient_sources.clone());
        warnings.extend(
            phase
                .system
                .warnings
                .iter()
                .map(|warning| format!("phase {}: {warning}", phase_index + 1)),
        );
    }

    Ok(MnaSystem {
        a,
        k,
        b,
        u,
        unknowns: first.unknowns.clone(),
        inputs: first.inputs.clone(),
        input_values: first.input_values.clone(),
        transient_sources,
        // An averaged system describes a duty-weighted *steady-state* model, not one particular
        // run from one particular starting point, so a phase netlist's `ic=` values have no
        // meaning here and are deliberately dropped rather than carried through.
        initial_conditions: Vec::new(),
        parameter_defaults,
        warnings,
    })
}

/// Computes the steady-state duty-perturbation column for two phases.
///
/// For `A*x + K*dot(x) = B*u`, linearizing the two-phase weighted model at
/// `dot(X)=0` gives the additional right-hand-side column
///
/// ```text
/// ((B_on-B_off) U - (A_on-A_off) X) * d
/// ```
///
/// `state_operating_point` is the full MNA operating-point vector, and
/// `input_operating_point` follows the systems' input order.
pub fn small_signal_duty_input(
    on: &MnaSystem,
    off: &MnaSystem,
    state_operating_point: &[Expression],
    input_operating_point: &[Expression],
) -> Result<Vec<Expression>, AveragingError> {
    ensure_compatible(on, off)?;
    if state_operating_point.len() != on.order() {
        return Err(AveragingError::OperatingPointSize {
            expected: on.order(),
            actual: state_operating_point.len(),
            kind: "state",
        });
    }
    if input_operating_point.len() != on.inputs.len() {
        return Err(AveragingError::OperatingPointSize {
            expected: on.inputs.len(),
            actual: input_operating_point.len(),
            kind: "input",
        });
    }

    let mut result = vec![Expression::zero(); on.order()];
    for (row, result_row) in result.iter_mut().enumerate() {
        for (column, input) in input_operating_point.iter().enumerate() {
            let difference = on.b[(row, column)].clone() - off.b[(row, column)].clone();
            *result_row = result_row.clone() + difference * input.clone();
        }
        for (column, state) in state_operating_point.iter().enumerate() {
            let difference = on.a[(row, column)].clone() - off.a[(row, column)].clone();
            *result_row = result_row.clone() - difference * state.clone();
        }
    }
    Ok(result)
}

fn ensure_compatible(first: &MnaSystem, other: &MnaSystem) -> Result<(), AveragingError> {
    if first.unknowns != other.unknowns {
        return Err(AveragingError::UnknownsDiffer);
    }
    if first.inputs != other.inputs {
        return Err(AveragingError::InputsDiffer);
    }
    Ok(())
}

fn weighted_matrix<F>(phases: &[WeightedPhase<'_>], select: F) -> Matrix<Expression>
where
    F: Fn(&MnaSystem) -> &Matrix<Expression>,
{
    let template = select(phases[0].system);
    let mut result = Matrix::filled(template.rows(), template.cols(), Expression::zero());
    for phase in phases {
        let matrix = select(phase.system);
        for row in 0..matrix.rows() {
            for column in 0..matrix.cols() {
                result[(row, column)] = result[(row, column)].clone()
                    + phase.weight.clone() * matrix[(row, column)].clone();
            }
        }
    }
    result
}

/// Error returned by converter averaging helpers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AveragingError {
    /// No phase was supplied.
    NoPhases,
    /// Phase unknown vectors differ.
    UnknownsDiffer,
    /// Phase input vectors differ.
    InputsDiffer,
    /// An operating point has the wrong length.
    OperatingPointSize {
        /// Required length.
        expected: usize,
        /// Supplied length.
        actual: usize,
        /// `"state"` or `"input"`.
        kind: &'static str,
    },
}

impl fmt::Display for AveragingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoPhases => f.write_str("at least one converter phase is required"),
            Self::UnknownsDiffer => f.write_str("converter phases have different unknown vectors"),
            Self::InputsDiffer => f.write_str("converter phases have different input vectors"),
            Self::OperatingPointSize {
                expected,
                actual,
                kind,
            } => write!(
                f,
                "{kind} operating point needs {expected} entries, received {actual}"
            ),
        }
    }
}

impl std::error::Error for AveragingError {}
