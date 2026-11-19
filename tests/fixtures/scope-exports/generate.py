#!/usr/bin/env python3
"""Writes the synthetic scope-export corpus that `tests/wave_file_scope_exports.rs` loads.

# What these files are, and what they are not

They are **synthetic**. Nobody has yet handed this project a capture off a real instrument, so
every byte here was written by this script. What they reproduce is not any one vendor's export
-- a sniffer that had to recognise a particular vendor would be wrong about the next one, which
is why `wave_file.rs` skips a preamble by shape instead -- but the **shapes** a capture can
arrive in, one file per shape:

  * a time column centred on the trigger, so it starts negative;
  * a metadata preamble of a different width from the data, with the channel names on the last
    line of it;
  * a units row between the channel names and the data;
  * a sample *index* in the first column, with the time base only in the header's `Start` and
    `Increment` fields;
  * tabs, aligned spaces, CRLF, `#` and `%` comments, a trailing blank line;
  * and the six ways a file can be unreadable, each of which must be refused by name.

The waveform in every file is the same textbook pair: a 1 kHz square drive and the step
response of a series RC across it. Which signal it is does not matter to a loader -- what
matters is that a reader can see at a glance whether the numbers survived the trip.

# Provenance of the shapes

The negative time base, the vendor-specific preamble and the channel-named header are stated in
`src/wave_file.rs`'s own module doc as the three things that decide whether a real capture
loads. The sample-index-plus-`Start`/`Increment` shape is how the `RigolWFM` project's reader
derives time from a Rigol DS2000/DS4000 CSV (`offset + scale * x`, two header lines).

The rest -- which exact keys appear in a preamble, how many digits a vendor prints -- is
plausible invention and labelled as such. When a real export does arrive, add it alongside
these rather than editing them: a fixture whose provenance is "measured" is worth more than one
whose provenance is "written to pass".

Run from the repository root:

    python3 tests/fixtures/scope-exports/generate.py
"""

from __future__ import annotations

import math
from pathlib import Path

HERE = Path(__file__).resolve().parent

# The capture: a 10 ms window centred on the trigger, as a scope hands it over.
WINDOW = 10.0e-3
DRIVE_HZ = 1.0e3
TAU = 220.0e-6  # R = 2.2 k, C = 100 n


def square(t: float) -> float:
    """The 1 kHz, 0/3.3 V drive. Phase referenced to t = 0, the trigger."""
    return 3.3 if math.sin(2.0 * math.pi * DRIVE_HZ * t) >= 0.0 else 0.0


def rc_response(t: float) -> float:
    """The capacitor voltage, integrated with enough steps that it is not itself a test case."""
    v = 0.0
    start = -WINDOW
    steps = 4000
    dt = (t - start) / steps
    time = start
    for _ in range(steps):
        v += dt * (square(time) - v) / TAU
        time += dt
    return v


def samples(count: int, *, centred: bool = True) -> list[tuple[float, float, float]]:
    """`count` rows of `(t, drive, response)` across the window."""
    rows = []
    for index in range(count):
        t = -WINDOW / 2.0 + WINDOW * index / (count - 1)
        if not centred:
            t += WINDOW / 2.0
        rows.append((t, square(t), rc_response(t)))
    return rows


def write(name: str, text: str, *, crlf: bool = False) -> None:
    body = text.replace("\n", "\r\n") if crlf else text
    (HERE / name).write_text(body, encoding="utf-8", newline="")
    print(f"  {name}  ({len(body)} bytes)")


def sci(value: float) -> str:
    return f"{value:.7e}"


# --------------------------------------------------------------------------------------------
# Shapes that must load
# --------------------------------------------------------------------------------------------


def two_column_trigger_centred() -> None:
    """The simplest real export: one channel, header, time starting at -5 ms."""
    rows = samples(21)
    lines = ["TIME,CH1"]
    lines += [f"{sci(t)},{sci(vc)}" for t, _, vc in rows]
    write("two-column-trigger-centred.csv", "\n".join(lines) + "\n")


def metadata_preamble_two_channels() -> None:
    """A two-field metadata preamble, then a three-field header, then three-field data.

    This is the case the "last preamble line with the data's field count" rule exists for: the
    header is line 8, not line 1, and every line above it is narrower than the data.
    """
    rows = samples(21)
    lines = [
        "Model,SYNTHETIC-2CH",
        "Firmware,0.0.0-synthetic",
        "Record Length,21",
        "Sample Interval,5.0000000e-04",
        "Trigger Point,10",
        "Horizontal Units,s",
        "Vertical Units,V",
        "TIME,CH1,CH2",
    ]
    lines += [f"{sci(t)},{sci(drive)},{sci(vc)}" for t, drive, vc in rows]
    write("metadata-preamble-two-channels.csv", "\n".join(lines) + "\n")


def units_row_under_header() -> None:
    """Channel names, then a units row of the same width, then the data.

    Both rows are preamble and both have the data's field count, so the rule picks the *last*
    one -- the units. Pinned deliberately: whatever the loader does with this, a reader of the
    test should be able to see it without running anything.
    """
    rows = samples(21)
    lines = ["x-axis,CH1,CH2", "second,Volt,Volt"]
    lines += [f"{sci(t)},{sci(drive)},{sci(vc)}" for t, drive, vc in rows]
    write("units-row-under-header.csv", "\n".join(lines) + "\n")


def tab_separated() -> None:
    rows = samples(21)
    lines = ["TIME\tCH1"]
    lines += [f"{sci(t)}\t{sci(vc)}" for t, _, vc in rows]
    write("tab-separated.tsv", "\n".join(lines) + "\n")


def whitespace_aligned() -> None:
    """Aligned columns with no delimiter at all, under `#` and `%` comment lines.

    What a logger prints over a serial link, and what Octave's `save -ascii` writes.
    """
    rows = samples(21)
    lines = [
        "# synthetic capture, aligned columns",
        "%  t             drive        vc",
    ]
    lines += [f"{t: .7e}  {drive: .7e}  {vc: .7e}" for t, drive, vc in rows]
    write("whitespace-aligned.txt", "\n".join(lines) + "\n")


def crlf_and_trailing_blank() -> None:
    """CRLF line endings and a trailing blank line -- what any Windows-side export writes."""
    rows = samples(21)
    lines = ["TIME,CH1"]
    lines += [f"{sci(t)},{sci(vc)}" for t, _, vc in rows]
    write("crlf-trailing-blank.csv", "\n".join(lines) + "\n\n", crlf=True)


def exponent_and_sign_forms() -> None:
    """`E-03`, `e-3`, a leading `+`, a bare `0`, and a value with no exponent at all.

    One file per vendor's idea of how to print a float would be a dozen files; this is all of
    those spellings in one, which is the same test.
    """
    lines = [
        "TIME,CH1",
        "-5.0000000E-03,+0.0000000E+00",
        "-4e-3,0",
        "-3.0e-03,+1.2",
        "-2.000000E-03,2.2000000e+00",
        "-1.0E-3,3.0",
        "0,+3.3000000E+00",
        "1.0E-3,3.3",
    ]
    write("exponent-and-sign-forms.csv", "\n".join(lines) + "\n")


def long_capture() -> None:
    """2 500 samples: the size that made `segment_before` a binary search rather than a scan."""
    rows = samples(2500)
    lines = [
        "Record Length,2500",
        "Sample Interval,4.0016006e-06",
        "TIME,CH1,CH2",
    ]
    lines += [f"{sci(t)},{sci(drive)},{sci(vc)}" for t, drive, vc in rows]
    write("long-capture-2500.csv", "\n".join(lines) + "\n")


def sample_index_start_increment() -> None:
    """The trap: column 0 is a sample *index*, and the time base is in the header.

    Time is `Start + Increment * x`, so the file's own first column is 0, 1, 2, ... . A loader
    that takes column 0 as seconds reads a 10 ms capture as a 2 500 *second* one and is wrong
    by seven orders of magnitude, with every sample individually plausible.
    """
    rows = samples(21)
    lines = [
        "X,CH1,CH2,Start,Increment",
        f"Sequence,Volt,Volt,{sci(-WINDOW / 2.0)},{sci(WINDOW / 20.0)}",
    ]
    lines += [f"{index},{sci(drive)},{sci(vc)}" for index, (_, drive, vc) in enumerate(rows)]
    write("sample-index-start-increment.csv", "\n".join(lines) + "\n")


def already_zero_based() -> None:
    """A capture that already starts at t = 0, so `tshift=auto` must change nothing."""
    rows = samples(21, centred=False)
    lines = ["TIME,CH1"]
    lines += [f"{sci(t)},{sci(vc)}" for t, _, vc in rows]
    write("zero-based-time.csv", "\n".join(lines) + "\n")


def repeated_timestamp() -> None:
    """Two samples at one instant: a logger's record of an edge, and legal (time never
    decreases). The linear arm has a zero-width segment to survive."""
    lines = [
        "TIME,CH1",
        "0.0000000e+00,0.0000000e+00",
        "1.0000000e-03,0.0000000e+00",
        "1.0000000e-03,3.3000000e+00",
        "2.0000000e-03,3.3000000e+00",
    ]
    write("repeated-timestamp.csv", "\n".join(lines) + "\n")


# --------------------------------------------------------------------------------------------
# Shapes that must be refused, by name
# --------------------------------------------------------------------------------------------


def decimal_comma_semicolon() -> None:
    """A European-locale export: `;` separates, `,` is the decimal point. Refused, not guessed."""
    rows = samples(11)
    lines = ["TIME;CH1;CH2"]
    for t, drive, vc in rows:
        lines.append(f"{sci(t)};{sci(drive)};{sci(vc)}".replace(".", ","))
    write("decimal-comma-semicolon.csv", "\n".join(lines) + "\n")


def ragged_row() -> None:
    """One row short of a field -- a truncated transfer, or a channel switched off mid-export."""
    lines = [
        "TIME,CH1,CH2",
        "0.0000000e+00,0.0000000e+00,0.0000000e+00",
        "1.0000000e-03,3.3000000e+00,2.1000000e+00",
        "2.0000000e-03,3.3000000e+00",
        "3.0000000e-03,0.0000000e+00,3.0000000e+00",
    ]
    write("ragged-row.csv", "\n".join(lines) + "\n")


def time_goes_backwards() -> None:
    """Two captures concatenated: the second one's time base starts over."""
    lines = [
        "TIME,CH1",
        "0.0000000e+00,0.0000000e+00",
        "1.0000000e-03,3.3000000e+00",
        "2.0000000e-03,3.3000000e+00",
        "0.0000000e+00,0.0000000e+00",
        "1.0000000e-03,3.3000000e+00",
    ]
    write("time-goes-backwards.csv", "\n".join(lines) + "\n")


def one_column() -> None:
    """Amplitudes with no time column -- what a scope set to "values only" writes."""
    lines = ["CH1"] + [f"{sci(vc)}" for _, _, vc in samples(11)]
    write("one-column.csv", "\n".join(lines) + "\n")


def setup_only() -> None:
    """A setup file saved instead of a waveform: metadata and no numbers anywhere."""
    lines = [
        "Model,SYNTHETIC-2CH",
        "Firmware,0.0.0-synthetic",
        "Channel,CH1",
        "Coupling,DC",
        "Probe Attenuation,x10",
        "Vertical Scale,one volt per division",
        "Horizontal Scale,one millisecond per division",
    ]
    write("setup-only.csv", "\n".join(lines) + "\n")


def interleaved_metadata_and_data() -> None:
    """Metadata in the left columns, data in the right ones, on the same rows.

    An older generation of export does this: the key and its value occupy the first fields of
    the first n rows, the waveform occupies the last two fields of every row, and a data row's
    first field is empty rather than a number. There is no prefix of this file that is
    preamble, so skipping by shape cannot find the data.
    """
    rows = samples(21)
    lines = []
    meta = [
        ("Record Length", "2.1000000e+01"),
        ("Sample Interval", "5.0000000e-04"),
        ("Trigger Point", "1.0000000e+01"),
        ("Source", "CH1"),
        ("Vertical Units", "V"),
        ("Horizontal Units", "s"),
    ]
    for index, (t, _, vc) in enumerate(rows):
        key, value = meta[index] if index < len(meta) else ("", "")
        lines.append(f"{key},{value},,{sci(t)},{sci(vc)},")
    write("interleaved-metadata-and-data.csv", "\n".join(lines) + "\n")


def overrange_sample() -> None:
    """A clipped sample written as a word rather than a number."""
    lines = [
        "TIME,CH1",
        "0.0000000e+00,0.0000000e+00",
        "1.0000000e-03,3.3000000e+00",
        "2.0000000e-03,nan",
        "3.0000000e-03,0.0000000e+00",
    ]
    write("overrange-sample.csv", "\n".join(lines) + "\n")


def empty_field() -> None:
    """A missing sample left as an empty field, which keeps the row's width."""
    lines = [
        "TIME,CH1",
        "0.0000000e+00,0.0000000e+00",
        "1.0000000e-03,",
        "2.0000000e-03,3.3000000e+00",
    ]
    write("empty-field.csv", "\n".join(lines) + "\n")


def main() -> None:
    print(f"writing the scope-export corpus into {HERE}")
    for builder in (
        two_column_trigger_centred,
        metadata_preamble_two_channels,
        units_row_under_header,
        tab_separated,
        whitespace_aligned,
        crlf_and_trailing_blank,
        exponent_and_sign_forms,
        long_capture,
        sample_index_start_increment,
        already_zero_based,
        repeated_timestamp,
        decimal_comma_semicolon,
        ragged_row,
        time_goes_backwards,
        one_column,
        setup_only,
        interleaved_metadata_and_data,
        overrange_sample,
        empty_field,
    ):
        builder()


if __name__ == "__main__":
    main()
