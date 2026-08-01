use std::collections::{BTreeMap, HashMap};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};
use spice_core::Dialect;

use crate::{BuildOptions, Expression, MnaBuilder, MnaSystem, SwitchState};

#[pyfunction]
#[pyo3(signature = (source, dialect="ngspice", document=false, switch_states=None, ron="Ron", roff="Roff"))]
fn build_mna(
    py: Python<'_>,
    source: &str,
    dialect: &str,
    document: bool,
    switch_states: Option<HashMap<String, bool>>,
    ron: &str,
    roff: &str,
) -> PyResult<Py<PyDict>> {
    let builder = configured_builder(dialect, switch_states, ron, roff)?;
    let system = if document {
        builder.build_document(source)
    } else {
        builder.build_fragment(source)
    }
    .map_err(value_error)?;
    system_to_dict(py, &system)
}

#[pyfunction]
#[pyo3(signature = (source, values, dialect="ngspice", document=false, switch_states=None, ron="Ron", roff="Roff", tolerance=1e-12))]
// Python callers benefit from keyword options matching `build_mna`; collecting
// them into an opaque config object would make the small binding harder to use.
#[allow(clippy::too_many_arguments)]
fn numeric_state_space(
    py: Python<'_>,
    source: &str,
    values: HashMap<String, f64>,
    dialect: &str,
    document: bool,
    switch_states: Option<HashMap<String, bool>>,
    ron: &str,
    roff: &str,
    tolerance: f64,
) -> PyResult<Py<PyDict>> {
    let builder = configured_builder(dialect, switch_states, ron, roff)?;
    let system = if document {
        builder.build_document(source)
    } else {
        builder.build_fragment(source)
    }
    .map_err(value_error)?;
    let values: BTreeMap<_, _> = values.into_iter().collect();
    let state_space = system
        .evaluate(&values)
        .map_err(value_error)?
        .to_state_space(tolerance)
        .map_err(value_error)?;

    let result = PyDict::new(py);
    result.set_item("a", matrix_rows(&state_space.a))?;
    result.set_item("b", matrix_rows(&state_space.b))?;
    result.set_item("states", state_space.states)?;
    result.set_item("inputs", state_space.inputs)?;
    result.set_item("mna_state_indices", state_space.mna_state_indices)?;
    Ok(result.unbind())
}

fn configured_builder(
    dialect: &str,
    switch_states: Option<HashMap<String, bool>>,
    ron: &str,
    roff: &str,
) -> PyResult<MnaBuilder> {
    let dialect = match dialect.to_ascii_lowercase().as_str() {
        "ngspice" => Dialect::Ngspice,
        "xyce" => Dialect::Xyce,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown SPICE dialect '{other}'"
            )))
        }
    };
    let mut options = BuildOptions {
        on_resistance: Expression::parse_scalar(ron).map_err(PyValueError::new_err)?,
        off_resistance: Expression::parse_scalar(roff).map_err(PyValueError::new_err)?,
        ..BuildOptions::default()
    };
    for (name, on) in switch_states.unwrap_or_default() {
        options.set_switch(
            name,
            if on {
                SwitchState::On
            } else {
                SwitchState::Off
            },
        );
    }
    Ok(MnaBuilder::with_options(dialect, options))
}

fn system_to_dict(py: Python<'_>, system: &MnaSystem) -> PyResult<Py<PyDict>> {
    let strings = system.to_string_system();
    let result = PyDict::new(py);
    result.set_item("a", string_rows(&strings.a, strings.rows, strings.rows))?;
    result.set_item("k", string_rows(&strings.k, strings.rows, strings.rows))?;
    result.set_item(
        "b",
        string_rows(&strings.b, strings.rows, strings.inputs_count),
    )?;
    result.set_item("u", strings.u)?;
    result.set_item("unknowns", strings.unknowns)?;
    result.set_item("inputs", strings.inputs)?;
    result.set_item("input_values", strings.input_values)?;
    result.set_item("parameter_defaults", strings.parameter_defaults)?;
    result.set_item("warnings", strings.warnings)?;
    Ok(result.unbind())
}

fn string_rows(values: &[String], rows: usize, cols: usize) -> Vec<Vec<String>> {
    (0..rows)
        .map(|row| values[row * cols..(row + 1) * cols].to_vec())
        .collect()
}

fn matrix_rows(matrix: &crate::Matrix<f64>) -> Vec<Vec<f64>> {
    (0..matrix.rows())
        .map(|row| {
            (0..matrix.cols())
                .map(|column| matrix[(row, column)])
                .collect()
        })
        .collect()
}

fn value_error(error: impl ToString) -> PyErr {
    PyValueError::new_err(error.to_string())
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(build_mna, module)?)?;
    module.add_function(wrap_pyfunction!(numeric_state_space, module)?)?;
    Ok(())
}
