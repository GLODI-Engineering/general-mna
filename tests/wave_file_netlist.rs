//! `kind=filewave` — a measured waveform read off disk by a netlist line.
//!
//! The loader has its own unit tests, against text rather than files. These go through the real
//! path a user takes: a netlist naming a file on disk, parsed by `build_system`, producing a
//! block whose breakpoints are the file's numbers.

use std::path::PathBuf;

use general_mna::block_graph::{BlockKind, WaveAfter, WaveInterp};
use general_mna::{build_system, System};
use general_spice_core::dialect::Dialect;

/// A file in the temp directory, removed when the test ends however it ends.
struct Capture {
    path: PathBuf,
}

impl Capture {
    fn new(tag: &str, contents: &str) -> Capture {
        // Tagged with the test's own name and the process id: the suite runs in parallel and
        // two tests writing one path would fail in whichever order they happened to run.
        let path = std::env::temp_dir().join(format!(
            "general-mna-{tag}-{}-{:?}.csv",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::write(&path, contents).expect("the temp directory must be writable");
        Capture { path }
    }

    fn netlist_path(&self) -> String {
        self.path.display().to_string()
    }
}

impl Drop for Capture {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

fn blocks_of(source: &str) -> Result<Vec<general_mna::block_graph::BlockInstance>, String> {
    build_system(source, Dialect::Ngspice).map(|System { blocks, .. }| blocks)
}

/// The whole point: a netlist line reads a capture, and the block carries the file's numbers.
#[test]
fn a_netlist_line_reads_a_capture_off_disk() {
    let capture = Capture::new("basic", "TIME,CH1\n0,0\n1e-6,5\n2e-6,0\n");
    let source = format!("SCOPE kind=filewave path={}\n", capture.netlist_path());
    let blocks = blocks_of(&source).unwrap();

    assert_eq!(blocks.len(), 1);
    assert_eq!(blocks[0].name, "SCOPE");
    match &blocks[0].kind {
        BlockKind::FileWave {
            columns,
            output_names,
            interp,
            after,
            source: from,
        } => {
            assert_eq!(columns.len(), 1);
            assert_eq!(columns[0], vec![(0.0, 0.0), (1e-6, 5.0), (2e-6, 0.0)]);
            // The first output aliases the block's own name, as it does for every other
            // multi-output kind -- a single-channel capture is addressable as `SCOPE`.
            assert_eq!(output_names, &vec!["SCOPE".to_string()]);
            assert_eq!(*interp, WaveInterp::Linear);
            assert_eq!(*after, WaveAfter::Hold);
            assert_eq!(from, &capture.path);
        }
        other => panic!("expected a FileWave, got {other:?}"),
    }
}

/// A two-channel capture is one block with two outputs, named by the file's own header: the
/// first aliases the block's own name, the second is a top-level signal name of its own (`VOUT`,
/// not `SCOPE.VOUT`) -- the convention every other multi-output block kind already follows.
#[test]
fn a_two_channel_capture_is_one_block_with_two_named_outputs() {
    let capture = Capture::new("two-channel", "TIME,VGS,VOUT\n0,0,0\n1e-6,10,4.9\n");
    let source = format!("SCOPE kind=filewave path={}\n", capture.netlist_path());
    let blocks = blocks_of(&source).unwrap();

    match &blocks[0].kind {
        BlockKind::FileWave {
            columns,
            output_names,
            ..
        } => {
            assert_eq!(output_names, &vec!["SCOPE".to_string(), "VOUT".to_string()]);
            assert_eq!(columns.len(), 2);
            assert_eq!(columns[1], vec![(0.0, 0.0), (1e-6, 4.9)]);
        }
        other => panic!("expected a FileWave, got {other:?}"),
    }
}

/// `outputs=` renames the channels, which is how a capture whose header says `CH1,CH2` becomes
/// a netlist that says what those channels are.
#[test]
fn outputs_renames_the_channels() {
    let capture = Capture::new("renamed", "TIME,CH1,CH2\n0,0,0\n1,1,2\n");
    let source = format!(
        "SCOPE kind=filewave path={} outputs=gate,drain\n",
        capture.netlist_path()
    );
    match &blocks_of(&source).unwrap()[0].kind {
        BlockKind::FileWave { output_names, .. } => {
            // The first is still the block's own name: `outputs=` names the channels, and the
            // alias is not negotiable, or `SCOPE` would name nothing.
            assert_eq!(
                output_names,
                &vec!["SCOPE".to_string(), "drain".to_string()]
            );
        }
        other => panic!("expected a FileWave, got {other:?}"),
    }
}

/// Columns are chosen 1-based, the way a user counting columns in a spreadsheet says it, or by
/// the name the header gave them.
#[test]
fn a_column_is_chosen_one_based_or_by_name() {
    let capture = Capture::new("cols", "TIME,CH1,CH2\n0,1,2\n1,3,4\n");
    for selector in ["cols=3", "cols=CH2"] {
        let source = format!(
            "SCOPE kind=filewave path={} {selector}\n",
            capture.netlist_path()
        );
        match &blocks_of(&source).unwrap()[0].kind {
            BlockKind::FileWave { columns, .. } => {
                assert_eq!(columns.len(), 1, "{selector}");
                assert_eq!(columns[0], vec![(0.0, 2.0), (1.0, 4.0)], "{selector}");
            }
            other => panic!("{selector}: expected a FileWave, got {other:?}"),
        }
    }
}

/// `interp=` and `after=` are carried through, and an unknown value is refused with the list of
/// the ones that exist rather than silently falling back to the default.
#[test]
fn interp_and_after_are_carried_through_and_validated() {
    let capture = Capture::new("modes", "TIME,CH1\n0,0\n1,1\n");
    let source = format!(
        "SCOPE kind=filewave path={} interp=hold after=repeat\n",
        capture.netlist_path()
    );
    match &blocks_of(&source).unwrap()[0].kind {
        BlockKind::FileWave { interp, after, .. } => {
            assert_eq!(*interp, WaveInterp::Hold);
            assert_eq!(*after, WaveAfter::Repeat);
        }
        other => panic!("expected a FileWave, got {other:?}"),
    }

    for (field, value, expected) in [
        ("interp", "spline", "unknown interp 'spline'"),
        ("after", "wrap", "unknown after 'wrap'"),
        ("tshift", "soon", "neither auto, none, nor a number"),
    ] {
        let source = format!(
            "SCOPE kind=filewave path={} {field}={value}\n",
            capture.netlist_path()
        );
        let error = blocks_of(&source).unwrap_err();
        assert!(error.contains(expected), "{field}={value}: {error}");
        assert!(error.contains("device 'SCOPE'"), "{field}={value}: {error}");
    }
}

/// The time base, which is the detail most likely to go unnoticed: a scope export starts before
/// the trigger, so by default the capture is moved to start at `t = 0`.
#[test]
fn the_capture_starts_at_zero_unless_told_otherwise() {
    let capture = Capture::new("tshift", "TIME,CH1\n-5e-4,0\n0,5\n5e-4,0\n");
    let shifted = format!("SCOPE kind=filewave path={}\n", capture.netlist_path());
    match &blocks_of(&shifted).unwrap()[0].kind {
        BlockKind::FileWave { columns, .. } => {
            assert_eq!(columns[0][0].0, 0.0);
            assert_eq!(columns[0][2].0, 1e-3);
        }
        other => panic!("expected a FileWave, got {other:?}"),
    }

    let kept = format!(
        "SCOPE kind=filewave path={} tshift=none\n",
        capture.netlist_path()
    );
    match &blocks_of(&kept).unwrap()[0].kind {
        BlockKind::FileWave { columns, .. } => assert_eq!(columns[0][0].0, -5e-4),
        other => panic!("expected a FileWave, got {other:?}"),
    }

    let by = format!(
        "SCOPE kind=filewave path={} tshift=1m\n",
        capture.netlist_path()
    );
    match &blocks_of(&by).unwrap()[0].kind {
        // `1m` is one millisecond in this netlist's own number grammar -- the same grammar
        // every other field uses, rather than a second one for this block.
        BlockKind::FileWave { columns, .. } => assert_eq!(columns[0][0].0, 1e-3 - 5e-4),
        other => panic!("expected a FileWave, got {other:?}"),
    }
}

/// A ×10 probe and a removed DC offset, in the netlist's own number grammar.
#[test]
fn scale_and_offset_reach_the_breakpoints() {
    let capture = Capture::new("scaled", "TIME,CH1\n0,1\n1,2\n");
    let source = format!(
        "SCOPE kind=filewave path={} scale=10 offset=-500m\n",
        capture.netlist_path()
    );
    match &blocks_of(&source).unwrap()[0].kind {
        BlockKind::FileWave { columns, .. } => {
            assert_eq!(columns[0], vec![(0.0, 9.5), (1.0, 19.5)]);
        }
        other => panic!("expected a FileWave, got {other:?}"),
    }
}

/// `delim=` takes a word for a character a whitespace-separated netlist field cannot hold.
#[test]
fn the_separator_can_be_named_in_words() {
    let capture = Capture::new("delim", "TIME\tCH1\n0\t1\n1\t2\n");
    let source = format!(
        "SCOPE kind=filewave path={} delim=tab\n",
        capture.netlist_path()
    );
    match &blocks_of(&source).unwrap()[0].kind {
        BlockKind::FileWave { columns, .. } => {
            assert_eq!(columns[0], vec![(0.0, 1.0), (1.0, 2.0)]);
        }
        other => panic!("expected a FileWave, got {other:?}"),
    }
}

/// A file that is not there fails **at parse time**, with the netlist line and the file's own
/// reason. Loading lazily at first evaluation would report this ten seconds into a run, after
/// the user had waited for it.
#[test]
fn a_missing_file_fails_at_parse_time_naming_the_line_and_the_path() {
    let source = "SCOPE kind=filewave path=no-such-capture-9c2f.csv\n";
    let error = blocks_of(source).unwrap_err();
    assert!(error.contains("line 1"), "{error}");
    assert!(error.contains("device 'SCOPE'"), "{error}");
    assert!(error.contains("no-such-capture-9c2f.csv"), "{error}");
    assert!(error.contains("cannot read"), "{error}");
}

/// A missing `path=` is the generic missing-field error every other block kind gives, rather
/// than a special one this block invented.
#[test]
fn a_filewave_with_no_path_says_which_field_is_missing() {
    let error = blocks_of("SCOPE kind=filewave\n").unwrap_err();
    assert!(error.contains("missing field 'path'"), "{error}");
}

/// The loader's diagnosis survives the trip through the netlist layer: the file, the line in it,
/// and the problem, with the netlist's own position added in front.
#[test]
fn a_bad_row_keeps_the_loaders_own_message() {
    let capture = Capture::new("ragged", "TIME,CH1\n0,1\n1\n");
    let source = format!("SCOPE kind=filewave path={}\n", capture.netlist_path());
    let error = blocks_of(&source).unwrap_err();
    assert!(error.contains("line 1: device 'SCOPE'"), "{error}");
    assert!(error.contains("line 3"), "the file's own line: {error}");
    assert!(error.contains("the first data row has 2"), "{error}");
}

/// A capture driving a real circuit: the measured waveform becomes a voltage source through
/// `sig2phys`, which is the only thing a `V` source's magnitude may name. This is the shape a
/// user actually writes, and it asserts that nothing about the new block breaks that rule.
#[test]
fn a_capture_drives_a_voltage_source_through_sig2phys() {
    let capture = Capture::new("drives", "TIME,VIN\n0,0\n1e-3,5\n2e-3,0\n");
    let source = format!(
        "SCOPE kind=filewave path={}\n\
         VSRC kind=sig2phys domain=voltage in=SCOPE\n\
         V1 in 0 VSRC\n\
         R1 in out 1000\n\
         C1 out 0 1u\n",
        capture.netlist_path()
    );
    let System { mna, blocks, .. } = build_system(&source, Dialect::Ngspice).unwrap();
    assert_eq!(blocks.len(), 2);
    assert!(blocks.iter().any(|block| block.name == "SCOPE"));
    assert!(blocks.iter().any(|block| block.name == "VSRC"));
    // A real circuit was built around it, not just a block graph.
    assert!(!mna.unknowns.is_empty());
}
