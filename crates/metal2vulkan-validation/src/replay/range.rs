//! A stream's recording range (`docs/REPLAY-STREAM.md`, sections 1 and 10): the frames through
//! which its recording is sound, where a play ends unless asked to go past it.

use std::path::{Path, PathBuf};

/// How far a stream's recording is sound.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecordingRange {
    /// Through frame `F`'s present, frames counted from 0.
    LastFrame(usize),
    /// The whole stream.
    Whole,
}

impl RecordingRange {
    /// The last frame a play plays, given `--until-frame`'s `until`: the earlier of `until` and
    /// the range's last frame, unless `past` (`--past-range yes`) takes the play past the range.
    pub fn last_frame(self, past: bool, until: Option<usize>) -> Option<usize> {
        match (self, past) {
            (RecordingRange::LastFrame(last), false) => {
                Some(until.map_or(last, |until| until.min(last)))
            }
            _ => until,
        }
    }
}

/// Where a stream's recording range came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RangeSource {
    /// The stream's sidecar ([`sidecar`]).
    Sidecar,
    /// A row of the table the player was given, by the row's stream label.
    Table(&'static str),
    /// Neither: nothing states the range, so it is the whole stream.
    Unstated,
}

/// A row of a recording range table: a stream's label, its first segment's file name and byte
/// length, and its range.
pub type RangeRow = (&'static str, &'static str, u64, RecordingRange);

/// The recording ranges of the streams recorded before their recorder wrote sidecars, each found
/// by its first segment's file name and byte length. A stream not here, and with no sidecar, is
/// sound as a whole.
pub const RECORDED_RANGES: &[RangeRow] = &[
    (
        "u2",
        "355866.0000.m2vr",
        1_076_366_025,
        RecordingRange::LastFrame(1128),
    ),
    (
        "u3",
        "982893.0000.m2vr",
        1_090_835_052,
        RecordingRange::LastFrame(1178),
    ),
    (
        "r3",
        "3143738.0000.m2vr",
        1_079_062_708,
        RecordingRange::LastFrame(1137),
    ),
    (
        "r4",
        "4121827.0000.m2vr",
        1_095_656_664,
        RecordingRange::LastFrame(1099),
    ),
    (
        "b3",
        "3403756.0000.m2vr",
        325_063_687,
        RecordingRange::Whole,
    ),
];

/// The sidecar of the stream whose first segment is `first`: `STEM.range` beside it, where the
/// segment is `STEM.0000.m2vr` (or `STEM.m2vr`, a stream of one segment).
pub fn sidecar(first: &Path) -> PathBuf {
    beside(first, "range")
}

/// The host's display library of the stream whose first segment is `first`:
/// `STEM.display.metallib` beside it, as [`sidecar`].
pub fn display_library(first: &Path) -> PathBuf {
    beside(first, "display.metallib")
}

/// `STEM.EXTENSION` beside the first segment `first` (`STEM.0000.m2vr`, or `STEM.m2vr`).
fn beside(first: &Path, extension: &str) -> PathBuf {
    let name = first.file_name().and_then(|n| n.to_str()).unwrap_or("");
    let stem = (name.strip_suffix(".0000.m2vr"))
        .or_else(|| name.strip_suffix(".m2vr"))
        .unwrap_or(name);
    first.with_file_name(format!("{stem}.{extension}"))
}

/// The recording range of the stream whose first segment is `first`, and where it came from: its
/// sidecar where it has one (one line, `last-frame F` or `whole`), else the row of `table` that
/// names its first segment's file name and length, else the whole stream. A sidecar that reads
/// as neither is an error naming it.
pub fn recording_range(
    first: &Path,
    table: &[RangeRow],
) -> Result<(RecordingRange, RangeSource), String> {
    let path = sidecar(first);
    match std::fs::read_to_string(&path) {
        Ok(text) => {
            let range = match text.trim_end_matches('\n').split_once(' ') {
                None if text.trim_end_matches('\n') == "whole" => Some(RecordingRange::Whole),
                Some(("last-frame", frame)) => frame.parse().ok().map(RecordingRange::LastFrame),
                _ => None,
            };
            let why = || {
                format!(
                    "{}: {text:?} is neither last-frame F nor whole",
                    path.display()
                )
            };
            return range
                .map(|range| (range, RangeSource::Sidecar))
                .ok_or_else(why);
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(format!("{}: {error}", path.display())),
    }
    let name = first.file_name().and_then(|n| n.to_str()).unwrap_or("");
    let metadata =
        std::fs::metadata(first).map_err(|error| format!("{}: {error}", first.display()))?;
    let row = table
        .iter()
        .find(|(_, file, bytes, _)| *file == name && *bytes == metadata.len());
    Ok(match row {
        Some((label, _, _, range)) => (*range, RangeSource::Table(label)),
        None => (RecordingRange::Whole, RangeSource::Unstated),
    })
}
