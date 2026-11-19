//! The loader behind `kind=filewave`, against a corpus of scope-export *shapes*.
//!
//! # Why a corpus, and why synthetic
//!
//! `wave_file.rs`'s unit tests are written in string literals: each one is a few lines chosen
//! to exercise one branch. That is the right shape for a branch and the wrong shape for the
//! question this file asks, which is **would a capture that came off an instrument load?** The
//! answer depends on things no literal happens to contain -- a preamble of a different width
//! from the data, CRLF, a trigger-centred time base, a units row between the names and the
//! numbers, a first column that is a sample index rather than seconds.
//!
//! No real export has reached this project yet, so every file in
//! `tests/fixtures/scope-exports/` was written by `generate.py` beside them. They are honest
//! about that: what they reproduce is the *shape* a capture arrives in, not any one vendor's
//! bytes. When a real export does arrive it goes in alongside these, not instead of them.
//!
//! # What each test is for
//!
//! Every fixture gets one test, and the test states the outcome in full -- names, sample count,
//! applied shift, the span of the time base. Not because all of it is interesting, but because
//! a change in any of it is a change in what a user's capture becomes, and "the test still
//! passes" should mean "their waveform is still the same waveform".
//!
//! Three tests pin behaviour that is **wrong**, each pointing at the issue that owns it. A
//! pinned defect is visible; an un-pinned one is rediscovered.

use std::path::{Path, PathBuf};

use general_mna::wave_file::{load, ColumnRef, LoadOptions, TimeShift, WaveFileError, WaveTable};

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/scope-exports")
        .join(name)
}

fn loaded(name: &str) -> WaveTable {
    load(&fixture(name), &LoadOptions::default())
        .unwrap_or_else(|error| panic!("'{name}' should load: {error}"))
}

fn loaded_with(name: &str, options: &LoadOptions) -> WaveTable {
    load(&fixture(name), options).unwrap_or_else(|error| panic!("'{name}' should load: {error}"))
}

fn refused(name: &str) -> WaveFileError {
    match load(&fixture(name), &LoadOptions::default()) {
        Ok(table) => panic!(
            "'{name}' should be refused, but loaded {} samples",
            table.samples
        ),
        Err(error) => error,
    }
}

/// The time of the first and last breakpoint of one column.
fn span(table: &WaveTable, column: usize) -> (f64, f64) {
    let points = &table.columns[column];
    (points[0].0, points[points.len() - 1].0)
}

fn close(left: f64, right: f64) -> bool {
    (left - right).abs() <= 1e-12 * right.abs().max(1.0)
}

// ---------------------------------------------------------------------------------------------
// Shapes that load
// ---------------------------------------------------------------------------------------------

/// Two columns, a one-line header, and a time base centred on the trigger.
///
/// The simplest export there is, and still not trivial: its time column runs from -5 ms to
/// +5 ms, and `TimeShift::Auto` must move the whole capture to start at zero and *say* by how
/// much, so a run can report it.
#[test]
fn two_columns_centred_on_the_trigger() {
    let table = loaded("two-column-trigger-centred.csv");

    assert_eq!(table.names, ["CH1"], "the header names the output");
    assert_eq!(table.samples, 21);
    assert!(
        close(table.applied_shift, 5.0e-3),
        "a -5 ms start is shifted forward by 5 ms, reported as {}",
        table.applied_shift
    );
    let (first, last) = span(&table, 0);
    assert!(close(first, 0.0), "the capture starts at zero, not {first}");
    assert!(
        close(last, 10.0e-3),
        "and spans its 10 ms window, not {last}"
    );
}

/// Seven narrow metadata lines, then a header of the data's own width, then three columns.
///
/// This is the shape the "the header is the last preamble line with the data's field count"
/// rule exists for. Every line above the header has two fields against the data's three, so a
/// loader that took the *first* line as the header would name the outputs `SYNTHETIC-2CH` and
/// nothing.
#[test]
fn a_metadata_preamble_does_not_become_the_header() {
    let table = loaded("metadata-preamble-two-channels.csv");

    assert_eq!(
        table.names,
        ["CH1", "CH2"],
        "both channels keep their instrument names"
    );
    assert_eq!(table.samples, 21);
    assert!(close(table.applied_shift, 5.0e-3));
    assert!(
        close(table.columns[0][0].1, 3.3),
        "CH1 is the drive, high at the trigger: {}",
        table.columns[0][0].1
    );
}

/// A units row sits between the channel names and the data, and **takes their place**.
///
/// Both `x-axis,CH1,CH2` and `second,Volt,Volt` are preamble lines of the data's width, so the
/// rule picks the last of them and the outputs are called `Volt` and `Volt`. Two outputs of one
/// block now share a name, and these are top-level signal names that a measurement's `ref=`
/// points at -- so `ref=Volt` cannot say which channel it means.
///
/// Pinned as it is, not as it should be. See #49.
#[test]
fn a_units_row_displaces_the_channel_names() {
    let table = loaded("units-row-under-header.csv");

    assert_eq!(
        table.names,
        ["Volt", "Volt"],
        "today the units row wins and both outputs are called Volt -- the defect this pins"
    );
    assert_eq!(table.samples, 21);
}

/// Tab-separated, which the delimiter sniffer has to find without being told.
#[test]
fn tabs_are_detected() {
    let table = loaded("tab-separated.tsv");

    assert_eq!(table.names, ["CH1"]);
    assert_eq!(table.samples, 21);
    assert!(close(table.applied_shift, 5.0e-3));
}

/// Aligned columns, no delimiter at all, under `#` and `%` comment lines.
///
/// The numbers load: whitespace splitting is the fallback and it works. The **names do not**.
/// `%  t  drive  vc` is a comment line, dropped before the header is looked for, so the outputs
/// fall back to `c2` and `c3` -- and a commented header row is exactly how an ASCII logger and
/// Octave's `save -ascii` name their columns.
///
/// Pinned as it is. See #51.
#[test]
fn an_ascii_logs_columns_load_but_lose_their_names() {
    let table = loaded("whitespace-aligned.txt");

    assert_eq!(
        table.names,
        ["c2", "c3"],
        "today a commented header is a comment first and a header never -- the defect this pins"
    );
    assert_eq!(table.samples, 21);
    assert!(close(table.applied_shift, 5.0e-3));
    assert!(
        close(table.columns[0][0].1, 3.3),
        "the numbers themselves are right"
    );
}

/// CRLF line endings and a trailing blank line: every export written on or for Windows.
///
/// A stray `\r` left on the last field of each row would make every value unparseable, and a
/// trailing blank line is a ragged row of zero fields if it is not dropped first.
#[test]
fn crlf_and_a_trailing_blank_line() {
    let table = loaded("crlf-trailing-blank.csv");

    assert_eq!(table.names, ["CH1"]);
    assert_eq!(table.samples, 21, "the trailing blank line is not a row");
    assert!(table.columns[0].iter().all(|(_, value)| value.is_finite()));
}

/// `E-03`, `e-3`, a leading `+`, a bare `0`, and a value with no exponent.
///
/// One file per vendor's idea of how to print a float would be a dozen files. This is every
/// spelling in one, which is the same test.
#[test]
fn every_spelling_of_a_float() {
    let table = loaded("exponent-and-sign-forms.csv");

    assert_eq!(table.samples, 7);
    let values: Vec<f64> = table.columns[0].iter().map(|(_, value)| *value).collect();
    assert_eq!(values, [0.0, 0.0, 1.2, 2.2, 3.0, 3.3, 3.3]);
    let times: Vec<f64> = table.columns[0].iter().map(|(time, _)| *time).collect();
    assert!(close(times[0], 0.0), "-5.0000000E-03 shifted to zero");
    assert!(
        close(times[1], 1.0e-3),
        "-4e-3 is the same number: {}",
        times[1]
    );
    assert!(
        close(times[5], 5.0e-3),
        "a bare 0 is the trigger instant: {}",
        times[5]
    );
}

/// 2 500 samples, which is a capture rather than a literal.
///
/// The size matters: it is why `segment_before` is a binary search. A scan would still pass
/// this test, so what is asserted here is only that the file arrives intact and ordered.
#[test]
fn a_capture_of_realistic_length() {
    let table = loaded("long-capture-2500.csv");

    assert_eq!(table.names, ["CH1", "CH2"]);
    assert_eq!(table.samples, 2500);
    for column in &table.columns {
        assert_eq!(column.len(), 2500);
        assert!(
            column.windows(2).all(|pair| pair[0].0 <= pair[1].0),
            "breakpoints stay ordered after the shift"
        );
        assert!(column.iter().all(|(_, value)| value.is_finite()));
    }
    let (first, last) = span(&table, 0);
    assert!(close(first, 0.0));
    assert!(close(last, 10.0e-3), "a 10 ms window, not {last}");
}

/// The first column is a sample **index**; the time base is in the header's `Start` and
/// `Increment`. The capture loads, and is wrong by three orders of magnitude.
///
/// Time should be `Start + Increment * x`. Taken as seconds, the index turns a 10 ms window
/// into a 20 *second* one, and every individual sample is plausible -- the kind of silently
/// wrong result the rule engine exists to catch, arriving before the rule engine can see it.
///
/// The header is not even available to fall back on: `X,CH1,CH2,Start,Increment` is five fields
/// against the data's three, so no preamble line has the data's width and the outputs are `c2`
/// and `c3`.
///
/// Pinned as it is. See #50.
#[test]
fn a_sample_index_is_read_as_seconds() {
    let table = loaded("sample-index-start-increment.csv");

    assert_eq!(
        table.names,
        ["c2", "c3"],
        "no preamble line has the data's width"
    );
    assert_eq!(table.samples, 21);
    let (first, last) = span(&table, 0);
    assert!(close(first, 0.0));
    assert!(
        close(last, 20.0),
        "today the index is seconds, so the 10 ms capture spans {last} s -- the defect this pins"
    );
}

/// A capture that already starts at zero: `TimeShift::Auto` must be a no-op, and must report it.
#[test]
fn auto_shift_leaves_a_zero_based_capture_alone() {
    let table = loaded("zero-based-time.csv");

    assert_eq!(table.samples, 21);
    assert!(
        close(table.applied_shift, 0.0),
        "nothing to shift, so nothing is reported: {}",
        table.applied_shift
    );
    let (first, last) = span(&table, 0);
    assert!(close(first, 0.0));
    assert!(close(last, 10.0e-3));
}

/// Two samples at one instant -- a logger's record of an edge. Legal: time never *decreases*.
///
/// It leaves a zero-width segment in the breakpoint list, which the linear arm has to survive
/// rather than divide by.
#[test]
fn two_samples_at_one_instant_are_an_edge_not_an_error() {
    let table = loaded("repeated-timestamp.csv");

    assert_eq!(table.samples, 4);
    let times: Vec<f64> = table.columns[0].iter().map(|(time, _)| *time).collect();
    assert!(close(times[1], 1.0e-3));
    assert!(
        close(times[2], 1.0e-3),
        "both samples keep their shared stamp"
    );
    let values: Vec<f64> = table.columns[0].iter().map(|(_, value)| *value).collect();
    assert_eq!(values, [0.0, 0.0, 3.3, 3.3], "and the step between them");
}

// ---------------------------------------------------------------------------------------------
// The options, exercised against the corpus rather than against literals
// ---------------------------------------------------------------------------------------------

/// `cols=` by header name picks one channel out of two.
#[test]
fn one_channel_chosen_by_its_name() {
    let options = LoadOptions {
        columns: Some(vec![ColumnRef::Name("ch2".into())]),
        ..LoadOptions::default()
    };
    let table = loaded_with("metadata-preamble-two-channels.csv", &options);

    assert_eq!(
        table.names,
        ["CH2"],
        "matched case-insensitively, named as the file spells it"
    );
    assert_eq!(table.columns.len(), 1);
    assert!(
        table.columns[0][0].1 < 1.0,
        "CH2 is the response, near zero at the trigger"
    );
}

/// `outputs=` renames the channels, which is what makes `ref=` readable for a file with none.
#[test]
fn explicit_names_override_the_header() {
    let options = LoadOptions {
        names: Some(vec!["vgate".into(), "vcap".into()]),
        ..LoadOptions::default()
    };
    let table = loaded_with("units-row-under-header.csv", &options);

    assert_eq!(
        table.names,
        ["vgate", "vcap"],
        "the way out of the duplicate-Volt defect"
    );
}

/// `outputs=` with the wrong count is refused rather than zipped short.
#[test]
fn a_name_per_column_or_none() {
    let options = LoadOptions {
        names: Some(vec!["only-one".into()]),
        ..LoadOptions::default()
    };
    match load(&fixture("metadata-preamble-two-channels.csv"), &options) {
        Err(WaveFileError::NameCountMismatch { names, columns, .. }) => {
            assert_eq!((names, columns), (1, 2));
        }
        other => panic!("expected NameCountMismatch, got {other:?}"),
    }
}

/// `tshift=none` keeps the instrument's own time base, negative times included.
#[test]
fn the_shift_can_be_declined() {
    let options = LoadOptions {
        time_shift: TimeShift::None,
        ..LoadOptions::default()
    };
    let table = loaded_with("two-column-trigger-centred.csv", &options);

    assert!(close(table.applied_shift, 0.0), "nothing was applied");
    let (first, last) = span(&table, 0);
    assert!(
        close(first, -5.0e-3),
        "the capture still begins before the trigger: {first}"
    );
    assert!(close(last, 5.0e-3));
}

/// `scale=` and `offset=` are the ×10 probe and the removed DC term, applied once at load.
#[test]
fn a_times_ten_probe_and_a_removed_offset() {
    let options = LoadOptions {
        scale: 10.0,
        offset: -1.0,
        ..LoadOptions::default()
    };
    let table = loaded_with("exponent-and-sign-forms.csv", &options);

    let values: Vec<f64> = table.columns[0].iter().map(|(_, value)| *value).collect();
    assert_eq!(
        values,
        [-1.0, -1.0, 11.0, 21.0, 29.0, 32.0, 32.0],
        "scale * v + offset"
    );
}

/// `time=` can name a column that is not the first one.
#[test]
fn the_time_column_can_be_named() {
    let options = LoadOptions {
        time: ColumnRef::Name("time".into()),
        ..LoadOptions::default()
    };
    let table = loaded_with("metadata-preamble-two-channels.csv", &options);

    assert_eq!(
        table.names,
        ["CH1", "CH2"],
        "the time column is excluded from the outputs"
    );
    assert!(close(table.applied_shift, 5.0e-3));
}

/// A column that is not there names what is, so the next attempt can be right.
#[test]
fn a_missing_column_lists_the_ones_that_exist() {
    let options = LoadOptions {
        columns: Some(vec![ColumnRef::Name("CH3".into())]),
        ..LoadOptions::default()
    };
    match load(&fixture("metadata-preamble-two-channels.csv"), &options) {
        Err(error @ WaveFileError::NoSuchColumn { .. }) => {
            let message = error.to_string();
            assert!(message.contains("column 'CH3'"), "{message}");
            assert!(
                message.contains("CH1"),
                "and says what it does have: {message}"
            );
        }
        other => panic!("expected NoSuchColumn, got {other:?}"),
    }
}

/// `skip=` counts the file's own lines, as an editor numbers them -- not "preamble lines".
///
/// `skip=1` on a file with two header rows therefore leaves the second one to be read as data,
/// and the refusal says which line and which field: the semantics are visible in the message
/// rather than guessed at. `skip=2` is the one that works here, and the header rule already
/// found the data without being told.
#[test]
fn an_explicit_skip_counts_lines_as_an_editor_does() {
    let one = LoadOptions {
        skip: Some(1),
        ..LoadOptions::default()
    };
    match load(&fixture("units-row-under-header.csv"), &one) {
        Err(WaveFileError::NotANumber {
            line,
            column,
            field,
            ..
        }) => {
            assert_eq!((line, column), (2, 1), "line 2 is now data, and is not");
            assert_eq!(field, "second");
        }
        other => panic!("expected the units row to be read as data, got {other:?}"),
    }

    let two = LoadOptions {
        skip: Some(2),
        ..LoadOptions::default()
    };
    let table = loaded_with("units-row-under-header.csv", &two);
    assert_eq!(
        table.samples, 21,
        "both header rows skipped, every data row read"
    );
}

// ---------------------------------------------------------------------------------------------
// Shapes that are refused, each by name
// ---------------------------------------------------------------------------------------------

/// A European-locale export: `;` between fields, `,` inside the numbers.
///
/// There is no reading that is right for both this file and a comma-separated one, so it is
/// named and refused. What matters is *which* refusal: the message must be about the separator,
/// not "no line begins with a number", which would be true and useless.
#[test]
fn a_decimal_comma_is_refused_by_name() {
    match refused("decimal-comma-semicolon.csv") {
        WaveFileError::DecimalComma { line, field, .. } => {
            assert_eq!(line, 2, "the first data row");
            assert!(field.contains(','), "quoting the field itself: {field}");
            assert!(
                WaveFileError::DecimalComma {
                    path: fixture("decimal-comma-semicolon.csv"),
                    line,
                    field,
                }
                .to_string()
                .contains("decimal point"),
                "and saying what to do about it"
            );
        }
        other => panic!("expected DecimalComma, got {other:?}"),
    }
}

/// A row one field short -- a truncated transfer, or a channel switched off mid-export.
#[test]
fn a_ragged_row_names_its_line_and_both_counts() {
    match refused("ragged-row.csv") {
        WaveFileError::RaggedRow {
            line,
            expected,
            found,
            ..
        } => assert_eq!((line, expected, found), (4, 3, 2)),
        other => panic!("expected RaggedRow, got {other:?}"),
    }
}

/// Two captures concatenated, so the time base starts over partway down.
#[test]
fn a_time_base_that_restarts_is_refused() {
    match refused("time-goes-backwards.csv") {
        WaveFileError::TimeNotMonotonic {
            line,
            previous,
            found,
            ..
        } => {
            assert_eq!(line, 5);
            assert!(
                close(previous, 2.0e-3) && close(found, 0.0),
                "{previous} -> {found}"
            );
        }
        other => panic!("expected TimeNotMonotonic, got {other:?}"),
    }
}

/// Amplitudes with no time column, which is a real export setting and not a waveform.
#[test]
fn values_without_a_time_column() {
    assert!(matches!(
        refused("one-column.csv"),
        WaveFileError::OneColumn { .. }
    ));
}

/// A setup file saved where a waveform was meant: keys and words, no numbers anywhere.
#[test]
fn a_setup_file_holds_no_waveform() {
    assert!(matches!(
        refused("setup-only.csv"),
        WaveFileError::NoData { .. }
    ));
}

/// Metadata in the left columns and the waveform in the right ones, on the same rows.
///
/// An older generation of export writes the key and its value into the first fields of the
/// first few rows and the waveform into the last two fields of *every* row, so a data row's
/// first field is empty rather than a number. No prefix of the file is preamble, and skipping
/// by shape cannot find the data.
///
/// `NoData` is true and useless here: the file is full of numbers, in columns four and five.
/// Pinned as it is. See #52.
#[test]
fn metadata_beside_the_data_defeats_the_preamble_rule() {
    match refused("interleaved-metadata-and-data.csv") {
        error @ WaveFileError::NoData { .. } => assert!(
            error
                .to_string()
                .contains("no line in it begins with a number"),
            "the message names the absence of data in a file that is mostly data -- \
             the defect this pins: {error}"
        ),
        other => panic!("expected NoData, got {other:?}"),
    }
}

/// An empty field keeps the row's width, so it is a number that is not one.
#[test]
fn a_missing_sample_left_blank() {
    match refused("empty-field.csv") {
        WaveFileError::NotANumber {
            line,
            column,
            field,
            ..
        } => {
            assert_eq!((line, column), (3, 2));
            assert!(
                field.is_empty(),
                "the field quoted is the empty one: {field:?}"
            );
        }
        other => panic!("expected NotANumber, got {other:?}"),
    }
}

/// A clipped sample written as `nan` is refused, naming the line and the column.
///
/// `parse_spice_number` falls back to Rust's own float parser, which accepts `nan`, `inf` and
/// `-inf`. For a netlist field that is a question for another day; for a capture it is not a
/// question at all. A NaN value would become a breakpoint, and the first step that interpolated
/// across that segment would return NaN for the rest of the run, with nothing anywhere naming
/// the capture as the cause.
#[test]
fn an_overrange_sample_is_refused() {
    match refused("overrange-sample.csv") {
        WaveFileError::NotFinite {
            line,
            column,
            field,
            ..
        } => {
            assert_eq!(
                (line, column),
                (4, 2),
                "the third data row, the value column"
            );
            assert_eq!(field, "nan");
        }
        other => panic!("expected NotFinite, got {other:?}"),
    }
}

/// `inf` and `-inf` are refused the same way, at the first of them.
#[test]
fn an_infinite_sample_is_refused() {
    match refused("infinite-sample.csv") {
        WaveFileError::NotFinite { line, field, .. } => {
            assert_eq!(
                line, 3,
                "refused at the first one, not after reading the file"
            );
            assert_eq!(field, "inf");
        }
        other => panic!("expected NotFinite, got {other:?}"),
    }
}

/// A `nan` **timestamp** is the worse half, and the reason the check cannot live in the
/// monotonicity test.
///
/// `time < previous` is false when either side is NaN, so a NaN timestamp passes the ordering
/// check and the breakpoint list it admits has no total order at all. `segment_before`'s
/// `partition_point` is documented as returning an unspecified index on an unsorted slice, so
/// the waveform that results is not wrong in some bounded way -- it is arbitrary.
///
/// This is refused before the ordering check ever runs, because `number_at` is the one place
/// every time and every value in a capture passes through.
#[test]
fn a_nan_timestamp_is_refused_rather_than_silently_admitted() {
    match refused("nan-timestamp.csv") {
        WaveFileError::NotFinite {
            line,
            column,
            field,
            ..
        } => {
            assert_eq!((line, column), (3, 1), "the time column, column one");
            assert_eq!(field, "nan");
        }
        other => panic!("expected NotFinite, got {other:?}"),
    }
}

/// The message says why `nan` is not a number *here*, which `NotANumber` could not.
#[test]
fn the_refusal_explains_itself() {
    let message = refused("nan-timestamp.csv").to_string();

    assert!(
        message.contains("nan-timestamp.csv"),
        "names the file: {message}"
    );
    assert!(message.contains("line 3"), "and the line: {message}");
    assert!(
        message.contains("not a finite one"),
        "and distinguishes this from junk: {message}"
    );
    assert!(
        message.contains("removed or replaced"),
        "and says what to do about it: {message}"
    );
}
