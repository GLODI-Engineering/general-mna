//! Reading a measured waveform off disk: the loader behind `kind=filewave`.
//!
//! # Why this exists
//!
//! Nothing in this crate could read a data file. A user who measures a real waveform on a board
//! — with an oscilloscope, or printed over a serial link by the microcontroller driving it —
//! had no way to feed it into a simulation, so the two things worth doing with a capture were
//! both out of reach: **driving** the circuit with the measured input, and **grading** the
//! simulated output against the measured one (`kind=measure type=error`, whose `ref=` had
//! nothing but the simulation's own signals to point at).
//!
//! # What this module is, and is not
//!
//! It is a loader and a validator. It is **not** an interpolator: a column of `(t, v)` samples
//! evaluated over time is exactly [`crate::block_graph::BlockKind::Pwl`] (linear) or
//! [`crate::block_graph::BlockKind::Pwc`] (sample-and-hold) semantics, both of which are
//! already implemented and tested. So the file becomes breakpoints here, once, and the existing
//! evaluation answers every question about what the waveform is between samples.
//!
//! # The contract, and the three parts of it that are about real files
//!
//! A clean two-column CSV is the easy case and not the one that matters. These three decide
//! whether a capture off an actual instrument loads:
//!
//! 1. **The preamble is skipped by shape, not by vendor.** Every leading line whose first field
//!    is not a number is preamble. Tektronix, Keysight, Rigol and Siglent all write a different
//!    header — a sniffer that had to recognise each of them would be wrong about the fifth.
//! 2. **The column header becomes the output's name.** The last preamble line with the same
//!    field count as the data is the header, so a Rigol `X,CH1,CH2` yields outputs `CH1` and
//!    `CH2` -- and since those are top-level signal names, a measurement reads `ref=CH2`
//!    instead of `ref=c3`.
//! 3. **The time base is shifted to zero by default.** A scope export is centred on the
//!    trigger, so its time column starts negative — commonly at `-5 ms` for a 10 ms window. A
//!    capture that silently began before `t = 0` would put an entire comparison half a window
//!    out of phase, and the simulation would look wrong for a reason nowhere on screen. The
//!    shift is applied and [`WaveTable::applied_shift`] reports it, so the run can say so.
//!
//! # What is refused rather than guessed
//!
//! A **decimal comma**. `0,001;1,23` is a real European-locale export, and the same bytes with
//! a comma delimiter are four fields instead of two. There is no reading that is right for both
//! files, so this one is named and refused with what to do about it.
//!
//! A **non-finite sample**. `nan`, `inf` and `-inf` all parse, because Rust's own float parser
//! accepts them, and none of them is a thing a real signal does. A NaN value would become a
//! breakpoint and take the rest of the run with it; a NaN *timestamp* would pass the
//! monotonicity check below, since `time < previous` is false when either side is NaN, and leave
//! the breakpoint list with no total order for its binary search to rely on. Both are refused at
//! the field, naming the line and the column.
//!
//! Everything else that goes wrong — a ragged row, a non-numeric field, a time column that goes
//! backwards, a file with one column — names the file, the line and the problem.

use std::collections::BTreeSet;
use std::fmt;
use std::path::{Path, PathBuf};

use crate::expression::parse_spice_number;

/// Which column, by position or by the name its header gave it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ColumnRef {
    /// Zero-based position in the file.
    Index(usize),
    /// A name from the header line, compared case-insensitively.
    Name(String),
}

impl fmt::Display for ColumnRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ColumnRef::Index(index) => write!(f, "column {}", index + 1),
            ColumnRef::Name(name) => write!(f, "column '{name}'"),
        }
    }
}

/// What to do with the capture's own time base.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TimeShift {
    /// Subtract the first sample's time, so the capture starts at `t = 0`. The default, because
    /// a scope export is centred on its trigger and starts negative.
    Auto,
    /// Use the file's times as they are, negative ones included.
    None,
    /// Add this many seconds to every sample time.
    By(f64),
}

/// How a column's samples are read.
#[derive(Debug, Clone, PartialEq)]
pub struct LoadOptions {
    /// Which column holds time. Defaults to the first.
    pub time: ColumnRef,
    /// Which columns become outputs, in order. `None` means every column but the time column.
    pub columns: Option<Vec<ColumnRef>>,
    /// Output names, overriding the header's. Must match the column count when given.
    pub names: Option<Vec<String>>,
    /// Preamble lines to drop. `None` detects them by shape.
    pub skip: Option<usize>,
    /// Field separator. `None` detects it.
    pub delimiter: Option<char>,
    /// What to do with the capture's own time base — see [`TimeShift`].
    pub time_shift: TimeShift,
    /// Applied to every value as `scale * v + offset` — a ×10 probe, or a removed DC offset.
    pub scale: f64,
    /// The additive half of the same transform.
    pub offset: f64,
}

impl Default for LoadOptions {
    fn default() -> Self {
        LoadOptions {
            time: ColumnRef::Index(0),
            columns: None,
            names: None,
            skip: None,
            delimiter: None,
            time_shift: TimeShift::Auto,
            scale: 1.0,
            offset: 0.0,
        }
    }
}

/// One loaded capture: a named `(time, value)` series per output.
#[derive(Debug, Clone, PartialEq)]
pub struct WaveTable {
    /// Output names, in order. `names[0]` aliases the block's own name, following the
    /// `output_names` convention every other multi-output block kind uses.
    pub names: Vec<String>,
    /// Breakpoints per output, each already shifted, scaled and offset — ready to be a
    /// `BlockKind::Pwl`/`Pwc` `points` list with no further work.
    pub columns: Vec<Vec<(f64, f64)>>,
    /// The seconds added to every sample time, so a caller can report it. `0.0` when the file's
    /// own time base was used unchanged.
    pub applied_shift: f64,
    /// Samples read, before any of them became breakpoints — for a log line that says how big
    /// the capture was.
    pub samples: usize,
}

/// Why a capture could not be read. Every variant names the file, and every variant that can
/// name a line does.
#[derive(Debug, Clone, PartialEq)]
pub enum WaveFileError {
    /// The file could not be opened or read at all.
    Unreadable {
        /// The path as the netlist spelled it.
        path: PathBuf,
        /// The operating system's own account.
        reason: String,
    },
    /// No line in the file begins with a number.
    NoData {
        /// The file that holds no numbers.
        path: PathBuf,
    },
    /// A row has a different field count from the first data row.
    RaggedRow {
        /// The file.
        path: PathBuf,
        /// The offending line, counted as an editor counts them.
        line: usize,
        /// Fields in the first data row.
        expected: usize,
        /// Fields in this row.
        found: usize,
    },
    /// A field that has to be a number is not one.
    NotANumber {
        /// The file.
        path: PathBuf,
        /// The line it is on.
        line: usize,
        /// Its one-based column.
        column: usize,
        /// The field itself, so the message can quote it.
        field: String,
    },
    /// A field that parses as a number, but not as a finite one: `nan`, `inf`, `-inf`.
    NotFinite {
        /// The file.
        path: PathBuf,
        /// The line it is on.
        line: usize,
        /// Its one-based column.
        column: usize,
        /// The field itself, so the message can quote it.
        field: String,
    },
    /// A number written with a decimal comma — see the module doc.
    DecimalComma {
        /// The file.
        path: PathBuf,
        /// The line it is on.
        line: usize,
        /// The field itself.
        field: String,
    },
    /// Time went backwards between two rows.
    TimeNotMonotonic {
        /// The file.
        path: PathBuf,
        /// The line whose time is lower than the one before it.
        line: usize,
        /// The time on the previous row.
        previous: f64,
        /// The time on this one.
        found: f64,
    },
    /// Fewer than two columns, so there is a time column and nothing to read.
    OneColumn {
        /// The file.
        path: PathBuf,
    },
    /// A requested column is not in the file.
    NoSuchColumn {
        /// The file.
        path: PathBuf,
        /// What was asked for.
        column: ColumnRef,
        /// How many columns the file has.
        available: usize,
        /// The names it does have, where it has a header.
        names: Vec<String>,
    },
    /// `outputs=` named a different number of outputs than there are columns.
    NameCountMismatch {
        /// The file.
        path: PathBuf,
        /// Names given.
        names: usize,
        /// Columns selected.
        columns: usize,
    },
}

impl fmt::Display for WaveFileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WaveFileError::Unreadable { path, reason } => {
                write!(f, "cannot read '{}': {reason}", path.display())
            }
            WaveFileError::NoData { path } => write!(
                f,
                "'{}' has no data: no line in it begins with a number",
                path.display()
            ),
            WaveFileError::RaggedRow {
                path,
                line,
                expected,
                found,
            } => write!(
                f,
                "'{}' line {line}: {found} fields, but the first data row has {expected} -- \
                 every row must have the same number",
                path.display()
            ),
            WaveFileError::NotANumber {
                path,
                line,
                column,
                field,
            } => write!(
                f,
                "'{}' line {line} column {column}: '{field}' is not a number",
                path.display()
            ),
            WaveFileError::NotFinite {
                path,
                line,
                column,
                field,
            } => write!(
                f,
                "'{}' line {line} column {column}: '{field}' is a number but not a finite one \
                 -- a capture records a real signal, and no sample of one is nan or inf. An \
                 overrange or dropped sample has to be removed or replaced before the file can \
                 drive a simulation",
                path.display()
            ),
            WaveFileError::DecimalComma { path, line, field } => write!(
                f,
                "'{}' line {line}: '{field}' looks like a number written with a decimal comma. \
                 This file cannot be read either way -- with a comma separator those are two \
                 fields, without one they are a single number -- so re-export it with a \
                 decimal point",
                path.display()
            ),
            WaveFileError::TimeNotMonotonic {
                path,
                line,
                previous,
                found,
            } => write!(
                f,
                "'{}' line {line}: time went backwards, from {previous} to {found} -- a \
                 waveform needs a time column that never decreases (is the time column the one \
                 this file puts first?)",
                path.display()
            ),
            WaveFileError::OneColumn { path } => write!(
                f,
                "'{}' has one column: a waveform needs a time column and at least one value \
                 column",
                path.display()
            ),
            WaveFileError::NoSuchColumn {
                path,
                column,
                available,
                names,
            } => {
                write!(
                    f,
                    "'{}' has no {column}; it has {available}",
                    path.display()
                )?;
                if !names.is_empty() {
                    write!(f, ", named {}", names.join(", "))?;
                }
                Ok(())
            }
            WaveFileError::NameCountMismatch {
                path,
                names,
                columns,
            } => write!(
                f,
                "outputs= names {names} output(s) but {columns} column(s) of '{}' were \
                 selected",
                path.display()
            ),
        }
    }
}

impl std::error::Error for WaveFileError {}

/// Reads `path` under `options`.
///
/// A relative path resolves against the process's working directory, **not** the netlist's
/// directory — the same rule `kind=cscript`'s `lib=` and `kind=pyblock`'s `path=` already
/// follow, and already documented in the engine's own `cscript.md`. One rule for every file a
/// netlist names is worth more than a more convenient rule that only some of them obey.
pub fn load(path: &Path, options: &LoadOptions) -> Result<WaveTable, WaveFileError> {
    let text = std::fs::read_to_string(path).map_err(|error| WaveFileError::Unreadable {
        path: path.to_path_buf(),
        reason: error.to_string(),
    })?;
    load_str(&text, path, options)
}

/// [`load`], for text already in hand. `path` is used only in messages, so a caller can report
/// a name for data that never was a file.
pub fn load_str(
    text: &str,
    path: &Path,
    options: &LoadOptions,
) -> Result<WaveTable, WaveFileError> {
    let lines: Vec<(usize, &str)> = text
        .lines()
        .enumerate()
        .map(|(index, line)| (index + 1, line))
        .filter(|(_, line)| !is_blank_or_comment(line))
        .collect();

    // The delimiter has to be settled first: whether a line "starts with a number" cannot be
    // answered without knowing where its first field ends. `0,0` is one field of nonsense under
    // a whitespace split and two numbers under a comma.
    let delimiter = match options.delimiter {
        Some(delimiter) => Some(delimiter),
        None => detect_delimiter(&lines),
    };

    // Where the numbers start. An explicit `skip=` counts lines of the file as the user sees
    // them, blank and commented ones included, so it matches what an editor shows.
    let first_data = match options.skip {
        Some(skip) => lines
            .iter()
            .position(|(number, _)| *number > skip)
            .ok_or_else(|| WaveFileError::NoData {
                path: path.to_path_buf(),
            })?,
        None => lines
            .iter()
            .position(|(_, line)| starts_with_a_number(line, delimiter))
            .ok_or_else(|| WaveFileError::NoData {
                path: path.to_path_buf(),
            })?,
    };

    let width = split(lines[first_data].1, delimiter).count();
    if width < 2 {
        return Err(WaveFileError::OneColumn {
            path: path.to_path_buf(),
        });
    }

    // The header is the last preamble line with the data's own field count. Both orders occur:
    // a bare `X,CH1,CH2` on its own, and several metadata lines followed by the names.
    let header: Vec<String> = lines[..first_data]
        .iter()
        .rev()
        .map(|(_, line)| split(line, delimiter).collect::<Vec<_>>())
        .find(|fields| fields.len() == width)
        .map(|fields| fields.iter().map(|field| sanitise(field)).collect())
        .unwrap_or_default();

    let time_index = resolve_column(&options.time, &header, width, path)?;
    let value_indices: Vec<usize> = match &options.columns {
        Some(requested) => requested
            .iter()
            .map(|column| resolve_column(column, &header, width, path))
            .collect::<Result<_, _>>()?,
        None => (0..width).filter(|index| *index != time_index).collect(),
    };

    let names: Vec<String> = match &options.names {
        Some(given) => {
            if given.len() != value_indices.len() {
                return Err(WaveFileError::NameCountMismatch {
                    path: path.to_path_buf(),
                    names: given.len(),
                    columns: value_indices.len(),
                });
            }
            given.clone()
        }
        None => value_indices
            .iter()
            .map(|index| {
                header
                    .get(*index)
                    .filter(|name| !name.is_empty())
                    .cloned()
                    .unwrap_or_else(|| format!("c{}", index + 1))
            })
            .collect(),
    };

    let mut times: Vec<f64> = Vec::new();
    let mut values: Vec<Vec<f64>> = vec![Vec::new(); value_indices.len()];
    for (number, line) in &lines[first_data..] {
        let fields: Vec<&str> = split(line, delimiter).collect();
        if fields.len() != width {
            return Err(WaveFileError::RaggedRow {
                path: path.to_path_buf(),
                line: *number,
                expected: width,
                found: fields.len(),
            });
        }
        let time = number_at(&fields, time_index, *number, path)?;
        if let Some(previous) = times.last() {
            if time < *previous {
                return Err(WaveFileError::TimeNotMonotonic {
                    path: path.to_path_buf(),
                    line: *number,
                    previous: *previous,
                    found: time,
                });
            }
        }
        times.push(time);
        for (slot, index) in value_indices.iter().enumerate() {
            values[slot].push(number_at(&fields, *index, *number, path)?);
        }
    }

    if times.is_empty() {
        return Err(WaveFileError::NoData {
            path: path.to_path_buf(),
        });
    }

    let applied_shift = match options.time_shift {
        TimeShift::Auto => -times[0],
        TimeShift::None => 0.0,
        TimeShift::By(shift) => shift,
    };

    let columns = values
        .into_iter()
        .map(|column| {
            times
                .iter()
                .zip(column)
                .map(|(time, value)| (time + applied_shift, options.scale * value + options.offset))
                .collect()
        })
        .collect();

    Ok(WaveTable {
        names,
        columns,
        applied_shift,
        samples: times.len(),
    })
}

fn is_blank_or_comment(line: &str) -> bool {
    let trimmed = line.trim();
    // `#` is the near-universal comment; `%` is what Octave and the array-language family
    // write, and a capture exported from one of those is a file this is meant to read.
    trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with('%')
}

fn starts_with_a_number(line: &str, delimiter: Option<char>) -> bool {
    split(line, delimiter)
        .next()
        .map(|field| {
            // A decimal comma counts as data here even though it will be refused two steps
            // later. Treating it as preamble instead would skip every row of the file and
            // report "no line in it begins with a number", which is true and useless -- the
            // user needs to be told about the separator, not about the absence of data.
            parse_spice_number(field).is_some() || looks_like_a_decimal_comma(field)
        })
        .unwrap_or(false)
}

/// Comma, semicolon or tab, whichever appears most across the top of the file; `None` means
/// split on whitespace.
///
/// Counted over several lines rather than one, because the first line of a real export is as
/// likely to be `Record Length,1.0E+3,,` as it is to be the header, and a preamble line written
/// with spaces in a comma-separated file would otherwise decide it.
///
/// A semicolon matters: it is what an instrument set to a European locale writes, and it is the
/// one case where the delimiter and the decimal separator are not both a comma.
fn detect_delimiter(lines: &[(usize, &str)]) -> Option<char> {
    const SAMPLE: usize = 10;
    let head: Vec<&str> = lines.iter().take(SAMPLE).map(|(_, line)| *line).collect();
    let count = |candidate: char| -> usize {
        head.iter()
            .map(|line| line.matches(candidate).count())
            .sum()
    };

    // A European-locale export writes `0,001;1,23`: the comma is inside the numbers and the
    // semicolon separates them. Counting characters alone picks the comma, which splits the
    // numbers in half and produces "'000;1' is not a number" -- a message about the wrong
    // thing. So when a digit-comma-digit appears *and* the file also uses semicolons or tabs,
    // the comma is not a candidate: a file whose fields are comma-separated does not also
    // separate them with semicolons.
    //
    // The decimal comma is still refused (there is no reading that is right for both files) --
    // this only makes the refusal name it.
    let decimal_comma = head.iter().any(|line| has_digit_comma_digit(line));
    let candidates: &[char] = if decimal_comma && (count(';') > 0 || count('\t') > 0) {
        &[';', '\t']
    } else {
        &[',', ';', '\t']
    };

    candidates
        .iter()
        .copied()
        .map(|candidate| (candidate, count(candidate)))
        .filter(|(_, count)| *count > 0)
        .max_by_key(|(_, count)| *count)
        .map(|(candidate, _)| candidate)
}

/// Whether `line` holds a `digit,digit` run — the signature of a decimal comma.
fn has_digit_comma_digit(line: &str) -> bool {
    let bytes = line.as_bytes();
    bytes
        .iter()
        .enumerate()
        .skip(1)
        .take(bytes.len().saturating_sub(2))
        .any(|(index, byte)| {
            *byte == b',' && bytes[index - 1].is_ascii_digit() && bytes[index + 1].is_ascii_digit()
        })
}

fn split<'a>(line: &'a str, delimiter: Option<char>) -> Box<dyn Iterator<Item = &'a str> + 'a> {
    match delimiter {
        Some(delimiter) => Box::new(line.split(delimiter).map(str::trim)),
        None => Box::new(line.split_whitespace()),
    }
}

/// A column name an output can be addressed by: a netlist name, not arbitrary file text.
///
/// A scope's own headers are close to usable already (`CH1`, `Volt`, `X`) but not reliably so —
/// `Volt(V)` and `CH 1` both occur — and a name with a space or a parenthesis in it cannot be
/// written in a `ref=` field at all.
fn sanitise(field: &str) -> String {
    let cleaned: String = field
        .trim()
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || character == '_' {
                character
            } else {
                '_'
            }
        })
        .collect();
    cleaned.trim_matches('_').to_string()
}

fn resolve_column(
    column: &ColumnRef,
    header: &[String],
    width: usize,
    path: &Path,
) -> Result<usize, WaveFileError> {
    let found = match column {
        ColumnRef::Index(index) => (*index < width).then_some(*index),
        ColumnRef::Name(name) => header
            .iter()
            .position(|candidate| candidate.eq_ignore_ascii_case(&sanitise(name))),
    };
    found.ok_or_else(|| WaveFileError::NoSuchColumn {
        path: path.to_path_buf(),
        column: column.clone(),
        available: width,
        names: header
            .iter()
            .filter(|name| !name.is_empty())
            .cloned()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect(),
    })
}

fn number_at(
    fields: &[&str],
    index: usize,
    line: usize,
    path: &Path,
) -> Result<f64, WaveFileError> {
    let field = fields[index];
    if let Some(value) = parse_spice_number(field) {
        // `parse_spice_number` falls back to Rust's own float parser, which accepts `nan`,
        // `inf` and `-inf`. For a netlist field that is a question for another day; for a
        // capture it is not a question at all, and letting one through is worse than it
        // sounds in both columns:
        //
        //   * a NaN **value** becomes a breakpoint, and the first step that interpolates
        //     across that segment returns NaN for the rest of the run, with nothing anywhere
        //     naming the capture as the cause;
        //   * a NaN **time** defeats the monotonicity check below, because `time < previous`
        //     is false when either side is NaN. The breakpoint list then has no total order,
        //     and `segment_before`'s `partition_point` is documented as returning an
        //     unspecified index on an unsorted slice -- so the waveform is not wrong in some
        //     bounded way, it is arbitrary.
        //
        // Refused here rather than in `parse_spice_number` so that only the loader's contract
        // changes: this is the one place every time and every value in a capture passes
        // through.
        if !value.is_finite() {
            return Err(WaveFileError::NotFinite {
                path: path.to_path_buf(),
                line,
                column: index + 1,
                field: field.to_string(),
            });
        }
        return Ok(value);
    }
    // Before calling it junk: is it a number with the wrong decimal separator? Saying so is the
    // difference between a user re-exporting the file and a user wondering what is wrong with
    // their instrument.
    if looks_like_a_decimal_comma(field) {
        return Err(WaveFileError::DecimalComma {
            path: path.to_path_buf(),
            line,
            field: field.to_string(),
        });
    }
    Err(WaveFileError::NotANumber {
        path: path.to_path_buf(),
        line,
        column: index + 1,
        field: field.to_string(),
    })
}

fn looks_like_a_decimal_comma(field: &str) -> bool {
    let body = field.trim().trim_start_matches(['+', '-']);
    match body.split_once(',') {
        Some((before, after)) => {
            !before.is_empty()
                && !after.is_empty()
                && before.chars().all(|c| c.is_ascii_digit())
                // The tail may carry an exponent, which is still a number with the wrong
                // separator: `1,5e-3`.
                && after
                    .chars()
                    .next()
                    .map(|c| c.is_ascii_digit())
                    .unwrap_or(false)
        }
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn here() -> PathBuf {
        PathBuf::from("capture.csv")
    }

    fn load_text(text: &str) -> Result<WaveTable, WaveFileError> {
        load_str(text, &here(), &LoadOptions::default())
    }

    /// The plain case, and the one thing about it that is not plain: the header's names become
    /// the outputs' names, so a measurement can say `ref=CH1` rather than `ref=c2`.
    #[test]
    fn a_two_column_csv_loads_with_its_header_as_the_output_name() {
        let table = load_text("Time,CH1\n0,0\n1e-6,5\n2e-6,0\n").unwrap();
        assert_eq!(table.names, vec!["CH1"]);
        assert_eq!(table.columns.len(), 1);
        assert_eq!(table.columns[0], vec![(0.0, 0.0), (1e-6, 5.0), (2e-6, 0.0)]);
        assert_eq!(table.samples, 3);
    }

    /// One block, one output per column -- the shape a multi-channel capture needs, and the
    /// same `output_names` convention every other multi-output block kind uses.
    #[test]
    fn every_column_but_the_time_column_becomes_an_output() {
        let table = load_text("X,CH1,CH2,CH3\n0,1,2,3\n1,4,5,6\n").unwrap();
        assert_eq!(table.names, vec!["CH1", "CH2", "CH3"]);
        assert_eq!(table.columns[1], vec![(0.0, 2.0), (1.0, 5.0)]);
    }

    /// A file off a real instrument: several metadata lines, then the names, then the numbers --
    /// and a time column centred on the trigger, so it starts negative.
    ///
    /// This is the test that matters most. A clean CSV is not what anyone will point at this.
    #[test]
    fn a_scope_export_loads_preamble_negative_time_and_all() {
        let text = "\
Record Length,1.0E+3,,
Sample Interval,1.0E-6,,
Trigger Point,5.00000000000E+02,,
TIME,CH1,CH2
-5.0e-4,0.00,0.01
-4.99e-4,0.08,0.02
0.0,4.98,0.03
5.0e-4,0.01,0.04
";
        let table = load_str(text, &here(), &LoadOptions::default()).unwrap();
        assert_eq!(table.names, vec!["CH1", "CH2"]);
        // The capture starts at t = 0 and the shift is reported, not silent: a run that quietly
        // began 500 us before zero would put the whole comparison out of phase.
        assert_eq!(table.applied_shift, 5.0e-4);
        assert_eq!(table.columns[0][0], (0.0, 0.0));
        assert!((table.columns[0][2].0 - 5.0e-4).abs() < 1e-18);
        assert_eq!(table.columns[0][3].0, 1.0e-3);
    }

    /// The pre-trigger time is the file's own, when asked for. Both answers are defensible --
    /// which is exactly why the default has to be stated and reported rather than assumed.
    #[test]
    fn the_shift_can_be_declined_or_given_outright() {
        let text = "t,v\n-1,10\n0,20\n1,30\n";
        let kept = load_str(
            text,
            &here(),
            &LoadOptions {
                time_shift: TimeShift::None,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(kept.applied_shift, 0.0);
        assert_eq!(kept.columns[0][0], (-1.0, 10.0));

        let moved = load_str(
            text,
            &here(),
            &LoadOptions {
                time_shift: TimeShift::By(4.0),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(moved.columns[0][0], (3.0, 10.0));
    }

    /// A ×10 probe, and a removed DC offset. Applied to the values and never to the time, which
    /// is why they are one option and not two.
    #[test]
    fn scale_and_offset_are_applied_to_the_values() {
        let table = load_str(
            "t,v\n0,1\n1,2\n",
            &here(),
            &LoadOptions {
                scale: 10.0,
                offset: -0.5,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(table.columns[0], vec![(0.0, 9.5), (1.0, 19.5)]);
    }

    /// Semicolons, tabs and plain whitespace all occur -- a European-locale instrument writes
    /// the first, and anything printed by a microcontroller over a serial link tends to write
    /// the last.
    #[test]
    fn the_separator_is_detected_not_demanded() {
        for text in [
            "t;v\n0;1\n1;2\n",
            "t\tv\n0\t1\n1\t2\n",
            "t v\n0 1\n1 2\n",
            "t   v\n0   1\n1   2\n",
        ] {
            let table = load_text(text).unwrap_or_else(|e| panic!("{text:?}: {e}"));
            assert_eq!(table.columns[0], vec![(0.0, 1.0), (1.0, 2.0)], "{text:?}");
        }
    }

    /// A header is not required. Without one the outputs are named by position, and the first
    /// line is data rather than being eaten as names.
    #[test]
    fn a_file_with_no_header_is_data_from_its_first_line() {
        let table = load_text("0,1\n1,2\n").unwrap();
        assert_eq!(table.names, vec!["c2"]);
        assert_eq!(table.columns[0].len(), 2);
    }

    /// Names a scope writes are not netlist names. `Volt(V)` cannot be written in a `ref=`
    /// field at all, so it is cleaned rather than passed through to fail later somewhere else.
    #[test]
    fn a_header_name_is_cleaned_into_something_a_netlist_can_say() {
        let table = load_text("Time (s),Volt(V),CH 2\n0,1,2\n1,3,4\n").unwrap();
        assert_eq!(table.names, vec!["Volt_V", "CH_2"]);
    }

    /// Picking columns, by position or by the name the header gave them. A four-channel capture
    /// where only one channel is wanted should not become four outputs.
    #[test]
    fn a_column_can_be_chosen_by_name_or_by_position() {
        let text = "TIME,CH1,CH2\n0,1,2\n1,3,4\n";
        let by_name = load_str(
            text,
            &here(),
            &LoadOptions {
                columns: Some(vec![ColumnRef::Name("CH2".to_string())]),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(by_name.names, vec!["CH2"]);
        assert_eq!(by_name.columns[0], vec![(0.0, 2.0), (1.0, 4.0)]);

        let by_index = load_str(
            text,
            &here(),
            &LoadOptions {
                columns: Some(vec![ColumnRef::Index(1)]),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(by_index.columns[0], vec![(0.0, 1.0), (1.0, 3.0)]);
    }

    /// The time column is not always first: a logger that prints its channels and then a
    /// timestamp is a normal thing to meet.
    #[test]
    fn the_time_column_does_not_have_to_be_the_first_one() {
        let table = load_str(
            "CH1,CH2,TIME\n1,2,0\n3,4,1\n",
            &here(),
            &LoadOptions {
                time: ColumnRef::Name("TIME".to_string()),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(table.names, vec!["CH1", "CH2"]);
        assert_eq!(table.columns[0], vec![(0.0, 1.0), (1.0, 3.0)]);
    }

    /// `outputs=` renames, and a count that does not match the columns is refused rather than
    /// applied to the first few.
    #[test]
    fn names_can_be_given_and_a_wrong_count_is_refused() {
        let text = "t,a,b\n0,1,2\n1,3,4\n";
        let named = load_str(
            text,
            &here(),
            &LoadOptions {
                names: Some(vec!["gate".to_string(), "drain".to_string()]),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(named.names, vec!["gate", "drain"]);

        let error = load_str(
            text,
            &here(),
            &LoadOptions {
                names: Some(vec!["only_one".to_string()]),
                ..Default::default()
            },
        )
        .unwrap_err();
        assert!(
            matches!(error, WaveFileError::NameCountMismatch { .. }),
            "{error}"
        );
    }

    /// Every refusal names the file, the line and the problem. Asserted together because the
    /// message *is* the feature here: a capture that will not load is the one moment a user
    /// needs to be told exactly what is wrong with their file.
    #[test]
    fn every_refusal_says_what_is_wrong_and_where() {
        let cases: Vec<(&str, &str)> = vec![
            (
                "t,v\n0,1\n1\n",
                "line 3: 1 fields, but the first data row has 2",
            ),
            (
                "t,v\n0,1\n1,oops\n",
                "line 3 column 2: 'oops' is not a number",
            ),
            (
                "t,v\n0,1\n-1,2\n",
                "line 3: time went backwards, from 0 to -1",
            ),
            ("just a header\nand another\n", "has no data"),
            ("v\n1\n2\n", "has one column"),
        ];
        for (text, expected) in cases {
            let error = load_text(text).unwrap_err().to_string();
            assert!(
                error.contains(expected),
                "{text:?}\n  expected to contain: {expected}\n  got: {error}"
            );
            assert!(error.contains("capture.csv"), "no file name in: {error}");
        }
    }

    /// A European-locale export is the one failure worth its own message. `0,001;1,23` is four
    /// fields with a comma separator and two numbers without one, and no single reading is right
    /// for both files -- so it is named, with what to do about it, rather than guessed at.
    #[test]
    fn a_decimal_comma_is_named_rather_than_guessed_at() {
        let error = load_text("t;v\n0,000;1,23\n0,001;2,34\n")
            .unwrap_err()
            .to_string();
        assert!(error.contains("decimal comma"), "{error}");
        assert!(error.contains("re-export"), "{error}");
        assert!(error.contains("'0,000'"), "{error}");
    }

    /// A missing column says what the file does have. A user who guessed `CH3` on a two-channel
    /// capture should not have to open the file to find out what it holds.
    #[test]
    fn a_missing_column_lists_the_ones_that_exist() {
        let error = load_str(
            "TIME,CH1,CH2\n0,1,2\n",
            &here(),
            &LoadOptions {
                columns: Some(vec![ColumnRef::Name("CH3".to_string())]),
                ..Default::default()
            },
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("no column 'CH3'"), "{error}");
        assert!(error.contains("CH1"), "{error}");
        assert!(error.contains("CH2"), "{error}");
    }

    /// `skip=` counts the file's lines the way an editor numbers them, blank and commented ones
    /// included -- otherwise a user counting rows in their editor would be off by however many
    /// this loader had quietly dropped.
    #[test]
    fn an_explicit_skip_counts_the_lines_a_user_sees() {
        let text = "a header that starts with 1 number\n\n0,1\n1,2\n";
        let table = load_str(
            text,
            &here(),
            &LoadOptions {
                skip: Some(2),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(table.columns[0], vec![(0.0, 1.0), (1.0, 2.0)]);
    }

    /// Blank lines and `#`/`%` comments are ignored anywhere, not only at the top: Octave and
    /// the rest of the array-language family write `%`, and a capture exported from one of them
    /// is a file this has to read.
    #[test]
    fn blank_lines_and_comments_are_ignored_wherever_they_are() {
        let table =
            load_text("# exported 2026-11-19\n\nt,v\n0,1\n\n% a note mid-file\n1,2\n").unwrap();
        assert_eq!(table.names, vec!["v"]);
        assert_eq!(table.columns[0], vec![(0.0, 1.0), (1.0, 2.0)]);
    }

    /// The netlist's own number grammar, so a hand-written capture reads the way a netlist
    /// does -- and an instrument's `1.0E-6` reads too, which is the spelling that actually
    /// turns up.
    #[test]
    fn a_field_may_be_written_in_either_grammar() {
        let table = load_text("t,v\n0,1.0E-6\n1u,2\n2e-6,3m\n").unwrap();
        assert_eq!(table.columns[0][0].1, 1.0e-6);
        assert_eq!(table.columns[0][1].0, 1.0e-6);
        assert_eq!(table.columns[0][2].1, 3.0e-3);
    }

    /// Two samples at the same instant are a step, which is a real thing a logger writes and
    /// which PWL already handles. Only going *backwards* is an error.
    #[test]
    fn a_repeated_timestamp_is_a_step_not_an_error() {
        let table = load_text("t,v\n0,0\n1,0\n1,5\n2,5\n").unwrap();
        assert_eq!(table.columns[0].len(), 4);
        assert_eq!(table.columns[0][1], (1.0, 0.0));
        assert_eq!(table.columns[0][2], (1.0, 5.0));
    }

    /// 100 000 samples, which is a deep-memory export, loaded in one pass. Guards against the
    /// loader being quadratic in the sample count -- a `contains`-per-row or a re-scan per
    /// column would turn a 2 MB file into minutes.
    #[test]
    fn a_hundred_thousand_samples_load_in_one_pass() {
        let mut text = String::from("TIME,CH1,CH2\n");
        for index in 0..100_000 {
            let time = index as f64 * 1e-7;
            text.push_str(&format!("{time:e},{},{}\n", index % 7, index % 3));
        }
        let started = std::time::Instant::now();
        let table = load_text(&text).unwrap();
        assert_eq!(table.samples, 100_000);
        assert_eq!(table.columns.len(), 2);
        assert_eq!(table.columns[0].len(), 100_000);
        // Generous on purpose: this is a guard against an accidental O(n^2), not a benchmark.
        assert!(
            started.elapsed().as_secs() < 5,
            "loading 100k samples took {:?}",
            started.elapsed()
        );
    }

    /// A file with one data row is a constant, not an error: it is what a single-point reference
    /// is, and PWL already answers it everywhere.
    #[test]
    fn one_data_row_is_a_constant() {
        let table = load_text("t,v\n0,7\n").unwrap();
        assert_eq!(table.columns[0], vec![(0.0, 7.0)]);
    }

    /// An unreadable file names the path and the operating system's own reason, rather than a
    /// message this crate invented about a file it never saw.
    #[test]
    fn a_missing_file_names_the_path_and_the_reason() {
        let error = load(
            Path::new("definitely-not-here-7f3a.csv"),
            &LoadOptions::default(),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("definitely-not-here-7f3a.csv"), "{error}");
        assert!(error.contains("cannot read"), "{error}");
    }
}
