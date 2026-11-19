# Scope-export shapes

Twenty-one capture files for the loader behind `kind=filewave`, one per **shape** a measured
waveform can arrive in. Loaded by `tests/wave_file_scope_exports.rs`; written by `generate.py`
in this directory.

They are **synthetic**. No capture off a real instrument has reached this project yet, so every
byte here was written by that script. What they reproduce is not any one vendor's export — a
loader that had to recognise a particular vendor would be wrong about the next one, which is
why `src/wave_file.rs` skips a preamble by shape instead — but the shapes themselves, which is
the level the loader actually works at.

The waveform in every file is the same textbook pair: a 1 kHz square drive and the step
response of a series RC across it ($R = 2.2\ \mathrm{k}\Omega$, $C = 100\ \mathrm{nF}$, so
$\tau = 220\ \mu\mathrm{s}$), sampled over a 10 ms window centred on the trigger. Which signal
it is does not matter to a loader; what matters is that a reader can see at a glance whether the
numbers survived the trip.

| File | Shape | Outcome |
|---|---|---|
| `two-column-trigger-centred.csv` | one channel, header, time from −5 ms | loads, shifted by 5 ms |
| `metadata-preamble-two-channels.csv` | narrow metadata preamble, then a header of the data's width | loads as `CH1`, `CH2` |
| `units-row-under-header.csv` | channel names, then a units row of the same width | loads as `Volt`, `Volt` — #49 |
| `tab-separated.tsv` | tabs | loads |
| `whitespace-aligned.txt` | aligned columns, `#`/`%` comments, commented header | loads as `c2`, `c3` — #51 |
| `crlf-trailing-blank.csv` | CRLF and a trailing blank line | loads |
| `exponent-and-sign-forms.csv` | `E-03`, `e-3`, `+`, a bare `0`, no exponent | loads |
| `long-capture-2500.csv` | 2 500 samples, two channels | loads |
| `sample-index-start-increment.csv` | sample index in column 1, time base in the header | loads as 20 **s** — #50 |
| `zero-based-time.csv` | already starts at zero | loads, shift 0 |
| `repeated-timestamp.csv` | two samples at one instant | loads |
| `overrange-sample.csv` | a clipped sample written `nan` | refused: `NotFinite` |
| `nan-timestamp.csv` | a `nan` in the **time** column | refused: `NotFinite` |
| `infinite-sample.csv` | `inf` and `-inf` samples | refused: `NotFinite` |
| `decimal-comma-semicolon.csv` | `;` separators, `,` decimal point | refused: `DecimalComma` |
| `ragged-row.csv` | a row one field short | refused: `RaggedRow` |
| `time-goes-backwards.csv` | two captures concatenated | refused: `TimeNotMonotonic` |
| `one-column.csv` | amplitudes, no time column | refused: `OneColumn` |
| `setup-only.csv` | a setup file, no numbers | refused: `NoData` |
| `empty-field.csv` | a sample left blank | refused: `NotANumber` |
| `interleaved-metadata-and-data.csv` | metadata in the left columns, data in the right | refused: `NoData` — #52 |

Four rows name an open issue: that is the outcome today, pinned by a test, and wrong. A pinned
defect is visible in CI output; an un-pinned one is rediscovered. A fifth defect was found the
same way and is fixed rather than pinned — the three `NotFinite` rows are what that fix looks
like from here (#48).

**When a real export arrives, add it alongside these rather than editing them.** A fixture whose
provenance is "measured" is worth more than one whose provenance is "written to pass", and the
two kinds should not be mistaken for each other.
