use std::collections::BTreeMap;
use std::fmt;

use crate::{EvaluationError, InitialCondition, Matrix, MnaSystem};

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

/// Default residual tolerance for [`MnaSystem::initial_state`]'s consistency check, relative to
/// the magnitude of the constraint row being checked. Matches the magnitude the state-space
/// reduction in this module is normally called with; pass an explicit value for a circuit whose
/// element values span a very different range.
pub const DEFAULT_INITIAL_STATE_TOLERANCE: f64 = 1e-12;

/// Why a netlist's declared `ic=` values could not be turned into a starting state vector.
#[derive(Debug, Clone, PartialEq)]
pub enum InitialStateError {
    /// A component value, or an `ic=` value itself, could not be evaluated numerically.
    Evaluation(EvaluationError),
    /// Two or more `ic=` values on capacitors contradict each other, independently of the rest
    /// of the circuit: they form a loop whose declared voltages do not sum to zero, so no
    /// assignment of node voltages satisfies all of them at once.
    ConflictingConditions {
        /// The element whose condition closed the contradictory loop.
        element: String,
        /// The voltage the other conditions in the loop already imply across it, in the
        /// orientation the loop was walked in — which may be the reverse of the card's own
        /// node order, so compare it with `declared` rather than reading a polarity out of it.
        implied: f64,
        /// The voltage this element declares, in that same orientation.
        declared: f64,
    },
    /// An assignment violates one of the circuit's own algebraic constraints — every unknown in
    /// that constraint is fixed by an `ic=`, and the values they are fixed to do not satisfy it.
    ///
    /// The two textbook cases: an `ic` on a capacitor wired directly across an ideal voltage
    /// source (the source's branch equation already fixes that voltage), and two series
    /// inductors whose `ic` values declare different currents (the shared node's KCL row
    /// already fixes them equal).
    InconsistentWithCircuit {
        /// The unknown whose equation is violated. A node voltage `V(n)` names that node's KCL
        /// equation; a branch current `I(name)` names that device's own branch equation.
        constraint: String,
        /// The constraint's residual at the assigned state — how far from satisfied it is, in
        /// the equation's own units.
        residual: f64,
    },
}

impl fmt::Display for InitialStateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Evaluation(error) => {
                write!(f, "cannot evaluate the ic= initial state: {error}")
            }
            Self::ConflictingConditions {
                element,
                implied,
                declared,
            } => write!(
                f,
                "contradictory ic= values: '{element}' declares {declared} V across itself, but \
                 the other ic= conditions it forms a loop with already imply {implied} V in the \
                 same orientation"
            ),
            Self::InconsistentWithCircuit {
                constraint,
                residual,
            } => write!(
                f,
                "the declared ic= values do not satisfy the circuit's own equation for \
                 {constraint} (residual {residual}); every unknown in that equation is fixed by \
                 an ic=, so nothing is left free to absorb the difference — check for an ic on a \
                 capacitor across an ideal voltage source, or series inductors declaring \
                 different currents"
            ),
        }
    }
}

impl std::error::Error for InitialStateError {}

impl From<EvaluationError> for InitialStateError {
    fn from(error: EvaluationError) -> Self {
        Self::Evaluation(error)
    }
}

impl MnaSystem {
    /// Turns this netlist's declared `ic=` values into the starting state vector of a transient
    /// run, in `unknowns` order — the `x` at `t = 0`.
    ///
    /// Returns `Ok(None)` when the netlist declares no `ic=` at all, so a caller can keep its
    /// existing "start from rest" default untouched rather than having to special-case an
    /// all-zero vector.
    ///
    /// # It is an assignment, not a solve
    ///
    /// This is what "use initial conditions" means: the operating point is *skipped*. The
    /// declared states are written into `x`, every other unknown starts at rest, and the system
    /// that actually gets solved — `A`, `K`, `B`, `u` — is not touched at all. No element is
    /// swapped for a source, nothing is opened or shorted, and no auxiliary unknown is
    /// introduced, so the `unknowns` ordering (a public contract of this crate) cannot move.
    ///
    /// Concretely, the assigned unknowns are:
    ///
    /// - for an `ic`-bearing inductor, its own branch-current unknown `I(<name>)`, set to `ic`;
    /// - for an `ic`-bearing capacitor, the *difference* between its two node voltages, which
    ///   is not one unknown but two. Those capacitors are treated as a graph over nodes, one
    ///   edge per declared `ic`, and each connected component's potentials are propagated from
    ///   a root at `0`: ground when the component touches ground, otherwise the component's
    ///   lowest-indexed node. A series chain and a floating island therefore both come out
    ///   right, and the result does not depend on the order the cards appear in.
    ///
    /// Everything else — the other node voltages, every source's branch current, every
    /// `ic`-free capacitor's voltage — stays at `0`. That is the visible difference from a
    /// constrained operating-point solve, which would let the rest of the circuit move to
    /// accommodate the declared value and would hand back an `ic`-free capacitor pre-charged by
    /// a circuit that has not run yet.
    ///
    /// The vector this returns therefore need not satisfy the circuit's algebraic constraints,
    /// exactly like any hand-built `x_initial`. That is a consumer's problem to know about, not
    /// a defect: a first backward-Euler step re-imposes them.
    ///
    /// # What is reported rather than silently overridden
    ///
    /// - `ic=` values that contradict *each other* — a loop of `ic`-bearing capacitors whose
    ///   declared voltages do not sum to zero — are [`InitialStateError::ConflictingConditions`].
    /// - An assignment that violates one of the circuit's own algebraic equations, in which
    ///   every unknown is fixed by an `ic=`, is
    ///   [`InitialStateError::InconsistentWithCircuit`]. That covers an `ic` on a capacitor
    ///   directly across an ideal voltage source, and series `ic`-bearing inductors declaring
    ///   different currents. Only equations with no `K` row are checked, and only those all of
    ///   whose unknowns are assigned: any equation with a free unknown in it, or with storage
    ///   in it, is one the circuit can still satisfy and is not this method's business.
    ///
    /// # Sign conventions
    ///
    /// Stated in full on [`InitialCondition`], and worth repeating for the one that is
    /// routinely misremembered: an inductor's `ic` is the current flowing **from its first
    /// node to its second node, through the inductor**. `L1 a b 5e-6 ic=12` puts 12 A into `a`
    /// and out of `b`; `L1 b a 5e-6 ic=12` is the opposite physical current.
    ///
    /// `values` supplies any symbol the netlist left open, exactly as
    /// [`MnaSystem::evaluate`] takes it. `tolerance` is the relative residual threshold of the
    /// consistency check; see [`DEFAULT_INITIAL_STATE_TOLERANCE`].
    pub fn initial_state(
        &self,
        values: &BTreeMap<String, f64>,
        tolerance: f64,
    ) -> Result<Option<Vec<f64>>, InitialStateError> {
        if self.initial_conditions.is_empty() {
            return Ok(None);
        }

        let order = self.unknowns.len();
        let mut environment = BTreeMap::new();
        for (name, expression) in &self.parameter_defaults {
            if let Ok(value) = expression.evaluate(values) {
                environment.insert(name.clone(), value);
            }
        }
        environment.extend(values.iter().map(|(name, value)| (name.clone(), *value)));

        let mut x = vec![0.0; order];
        let mut assigned = vec![false; order];

        for condition in &self.initial_conditions {
            let InitialCondition::InductorCurrent { branch, value, .. } = condition else {
                continue;
            };
            x[*branch] = value.evaluate(&environment)?;
            assigned[*branch] = true;
        }

        assign_capacitor_potentials(
            &self.initial_conditions,
            &environment,
            tolerance,
            &mut x,
            &mut assigned,
        )?;

        let numeric = self.evaluate(values)?;
        check_assignment_against_circuit(&numeric, &x, &assigned, tolerance)?;

        Ok(Some(x))
    }
}

/// Propagates the node potentials implied by every `ic`-bearing capacitor.
///
/// Each condition is an edge `v(positive) - v(negative) = ic` over the node unknowns, with
/// ground as an extra vertex pinned at `0`. Every connected component is walked breadth-first
/// from a root held at `0` — ground when the component contains it, otherwise the component's
/// lowest-indexed node, since a floating island's absolute potential is not something the
/// netlist declared and any choice is as good as another. An edge that closes a loop with a
/// mismatching sum is a contradiction between the conditions themselves and is reported.
/// A vertex of the `ic`-bearing-capacitor graph: an index into `unknowns` for a node voltage, or
/// `None` for ground, which is not an unknown and is always at zero.
type PotentialNode = Option<usize>;

/// One edge of that graph, as stored on the vertex it leaves: the vertex it reaches, the voltage
/// `v(this) - v(that)` the condition declares *in that direction*, and the element that declared
/// it, for diagnostics. Each condition contributes two, one per direction.
type PotentialEdge<'a> = (PotentialNode, f64, &'a str);

fn assign_capacitor_potentials(
    conditions: &[InitialCondition],
    environment: &BTreeMap<String, f64>,
    tolerance: f64,
    x: &mut [f64],
    assigned: &mut [bool],
) -> Result<(), InitialStateError> {
    let mut edges: BTreeMap<PotentialNode, Vec<PotentialEdge<'_>>> = BTreeMap::new();
    for condition in conditions {
        let InitialCondition::CapacitorVoltage {
            element,
            positive,
            negative,
            value,
        } = condition
        else {
            continue;
        };
        let volts = value.evaluate(environment)?;
        edges
            .entry(*positive)
            .or_default()
            .push((*negative, volts, element.as_str()));
        edges
            .entry(*negative)
            .or_default()
            .push((*positive, -volts, element.as_str()));
    }

    let mut potential: BTreeMap<PotentialNode, f64> = BTreeMap::new();
    // `BTreeMap` iteration puts `None` (ground) first and then ascending node indices, which is
    // exactly the root order this method documents.
    let roots: Vec<PotentialNode> = edges.keys().copied().collect();
    for root in roots {
        if potential.contains_key(&root) {
            continue;
        }
        potential.insert(root, 0.0);
        let mut queue = std::collections::VecDeque::from([root]);
        while let Some(node) = queue.pop_front() {
            let here = potential[&node];
            for (neighbor, delta, element) in edges.get(&node).into_iter().flatten() {
                // `delta` is `v(node) - v(neighbor)` as stored above.
                let implied = here - delta;
                match potential.get(neighbor) {
                    Some(known) => {
                        let scale = known.abs().max(implied.abs()).max(1.0);
                        if (known - implied).abs() > tolerance * scale {
                            return Err(InitialStateError::ConflictingConditions {
                                element: (*element).to_string(),
                                implied: here - known,
                                declared: *delta,
                            });
                        }
                    }
                    None => {
                        potential.insert(*neighbor, implied);
                        queue.push_back(*neighbor);
                    }
                }
            }
        }
    }

    for (node, volts) in potential {
        if let Some(index) = node {
            x[index] = volts;
            assigned[index] = true;
        }
    }
    Ok(())
}

/// Reports an assignment that violates one of the circuit's own algebraic equations.
///
/// Only rows with an all-zero `K` row are checked: a row with storage in it describes a
/// derivative this method says nothing about, so its residual at `t = 0` is not a contradiction
/// — it is the current that starts flowing. And within those rows, only the ones every one of
/// whose unknowns is assigned: a row with a free unknown left in it is a row the circuit can
/// still satisfy on its own, and the first solved step is where that happens.
fn check_assignment_against_circuit(
    numeric: &crate::NumericMnaSystem,
    x: &[f64],
    assigned: &[bool],
    tolerance: f64,
) -> Result<(), InitialStateError> {
    let order = x.len();
    for row in 0..order {
        if (0..order).any(|col| numeric.k[(row, col)] != 0.0) {
            continue;
        }
        let contributing: Vec<usize> = (0..order)
            .filter(|col| numeric.a[(row, *col)] != 0.0)
            .collect();
        // A *state* is determined whether or not it carries an `ic`: `initial_state` starts
        // every unknown at zero and only overwrites the ones an `ic` names, so an `ic`-free
        // inductor current is pinned at zero, not free. Nothing re-solves it -- the integrator
        // takes that zero as the initial current -- so a row containing it is fully determined
        // and its residual is a real contradiction. Treating it as free is what let an
        // inductor cut-set with an `ic` on some of its branches and not others be accepted
        // silently, starting 134% away from the requested current.
        //
        // Dynamic variables are the indices whose row in `K` is nonzero, the same test
        // `schur_complement` uses.
        // Restricted to inductor *branch currents*, which are the states nothing re-solves. A
        // capacitor's node voltage is also a state started at zero, but it is shared with the
        // resistive network and the first step genuinely does re-impose it -- treating it as
        // determined rejects `V1 a 0 400` + an `ic`-free capacitor on `a` with residual -400,
        // which is the documented and correct behaviour, not a contradiction. Measured: that
        // over-reach rejected 12 of 40 of this project's own committed decks.
        //
        // An inductor branch current is `I(...)` with storage: a voltage source's branch
        // current is also `I(...)` but has an all-zero `K` row, and a capacitor node voltage
        // has storage but is `V(...)`.
        let determined = |col: usize| {
            assigned[col]
                || (numeric.unknowns[col].starts_with("I(")
                    && (0..order).any(|j| numeric.k[(col, j)] != 0.0))
        };
        if contributing.is_empty() || !contributing.iter().all(|col| determined(*col)) {
            continue;
        }
        let mut residual = -numeric.u[row];
        let mut scale = numeric.u[row].abs();
        for col in contributing {
            let term = numeric.a[(row, col)] * x[col];
            residual += term;
            scale = scale.max(term.abs());
        }
        if residual.abs() > tolerance * scale.max(1.0) {
            return Err(InitialStateError::InconsistentWithCircuit {
                constraint: numeric.unknowns[row].clone(),
                residual,
            });
        }
    }
    Ok(())
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
