//! Modified nodal analysis (MNA) for educational circuit tools.
//!
//! The crate consumes the dialect-neutral AST from `general-spice-core` and builds
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
//!
//! Every public item is documented — `cargo doc --open` is a complete
//! reference; reading the source should never be necessary just to call
//! this crate. Start with [`MnaBuilder`] for parsing a netlist into an
//! [`MnaSystem`], [`average`] for converter-phase averaging, and
//! [`block_graph`] for the signal-domain block vocabulary consumed by
//! `dae-runtime`.

#![warn(missing_docs)]

mod averaging;
pub mod block_graph;
mod builder;
mod expression;
mod hierarchy;
mod matrix;
mod numeric;
mod symbolic;
mod system;
mod system_builder;
mod transient_source;

#[cfg(feature = "python")]
mod python;

pub use averaging::{average, small_signal_duty_input, AveragingError, WeightedPhase};
pub use builder::{BuildError, BuildOptions, MnaBuilder, SwitchState, UnsupportedElementPolicy};
pub use expression::{EvaluationError, Expression};
pub use matrix::Matrix;
pub use numeric::{
    InitialStateError, NumericMnaSystem, NumericStateSpace, StateSpaceError,
    DEFAULT_INITIAL_STATE_TOLERANCE,
};
pub use symbolic::{
    faddeev_leverrier, SymbolicStateSpace, SymbolicTransferFunction, TransferFunctionError,
};
pub use system::{InitialCondition, MnaSystem, StringMnaSystem};
pub use system_builder::{build_system, parse_and_flatten, System};
pub use transient_source::{PwlPoints, TransientFunction};
