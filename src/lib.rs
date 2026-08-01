//! Modified nodal analysis (MNA) for educational circuit tools.
//!
//! The crate consumes the dialect-neutral AST from `spice-core` and builds
//! the descriptor equation
//!
//! ```text
//! A x(t) + K dx(t)/dt = B u(t)
//! ```
//!
//! `A` is the memoryless matrix (often called `G`), `K` is the storage
//! matrix (often called `C`), and `B` maps independent sources into the
//! equation. Matrices contain [`Expression`] values so downstream Python,
//! SymPy, JavaScript, and PDF layers can preserve component names.

mod averaging;
mod builder;
mod expression;
mod matrix;
mod numeric;
mod system;

#[cfg(feature = "python")]
mod python;

pub use averaging::{average, small_signal_duty_input, AveragingError, WeightedPhase};
pub use builder::{BuildError, BuildOptions, MnaBuilder, SwitchState, UnsupportedElementPolicy};
pub use expression::{EvaluationError, Expression};
pub use matrix::Matrix;
pub use numeric::{NumericMnaSystem, NumericStateSpace, StateSpaceError};
pub use system::{MnaSystem, StringMnaSystem};
