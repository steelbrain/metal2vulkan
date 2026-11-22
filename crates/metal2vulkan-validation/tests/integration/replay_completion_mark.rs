//! The player's report names its mode on its first line (`--states-only`), and the report of a
//! stream recorded before the recorder's completion fix says so on the line after it
//! (`docs/REPLAY-STREAM.md`, sections 1, 6 and 10), where a fixed stream's does not.

use metal2vulkan_validation::replay::{
    StreamWriter, FLAG_ANNOUNCED_BYTES_RECORDED, FLAG_COMPLETION_BYTES_RECORDED,
};
use metal2vulkan_validation::ScratchDir;
use std::fs::File;
use std::process::Command;

const MARK: &str =
    "recorded before the recorder's completion fix: the stream may differ from what the program did";

const FULL: &str = "mode full: every call";

/// The report `command-replay play` writes of a one-call stream with header `flags`, given a device
/// type no device has, so the play stops before any device opens, with `extra` arguments.
fn report(dir: &ScratchDir, flags: u32, extra: &[&str]) -> String {
    let stream = dir.path().join(format!("flags{flags}.0000.m2vr"));
    let mut writer = StreamWriter::with_flags(File::create(&stream).unwrap(), flags).unwrap();
    writer.call("commit", 1, 0, &[], &[]).unwrap();
    writer.finish().unwrap();
    let out = dir.path().join(format!("out{flags}"));
    let status = Command::new(bin_exe!("command-replay"))
        .arg("play")
        .arg("--stream")
        .arg(&stream)
        .arg("--out")
        .arg(&out)
        .args(["--device-type", "none", "--shader-cache"])
        .arg(dir.path().join("cache"))
        .args(extra)
        .status()
        .unwrap();
    assert_eq!(status.code(), Some(2));
    std::fs::read_to_string(out.join("report.txt")).unwrap()
}

/// A stream whose header lacks bit 1 (flags 1, as every stream recorded before the fix reads)
/// carries the line, by itself, after the mode line; a stream with it (flags 3) does not carry the
/// line.
#[test]
fn a_report_of_a_stream_without_the_completion_bit_begins_with_the_line() {
    let dir = ScratchDir::new("replay-completion-mark").unwrap();
    let before = report(&dir, FLAG_ANNOUNCED_BYTES_RECORDED, &[]);
    let lines = before.lines().collect::<Vec<_>>();
    assert_eq!(lines[..2], [FULL, MARK], "{before}");
    assert!(lines[2].starts_with("range "), "{before}");
    let fixed = report(
        &dir,
        FLAG_ANNOUNCED_BYTES_RECORDED | FLAG_COMPLETION_BYTES_RECORDED,
        &[],
    );
    let lines = fixed.lines().collect::<Vec<_>>();
    assert_eq!(lines[0], FULL, "{fixed}");
    assert!(lines[1].starts_with("range "), "{fixed}");
    assert!(!fixed.contains(MARK), "{fixed}");
}

/// The first line of a report names the play's mode: `--states-only yes` a states-only play,
/// `--states-only no` and no option a full one; any other value is a bad command line, reported on
/// that line.
#[test]
fn a_report_names_its_mode_on_its_first_line() {
    let dir = ScratchDir::new("replay-mode-line").unwrap();
    let flags = FLAG_ANNOUNCED_BYTES_RECORDED | FLAG_COMPLETION_BYTES_RECORDED;
    let states = report(&dir, flags, &["--states-only", "yes"]);
    assert!(
        states.starts_with(
            "mode states-only: the device, library, function and pipeline state records and the \
             presents alone (--states-only yes)\n"
        ),
        "{states}"
    );
    let full = report(&dir, flags, &["--states-only", "no"]);
    assert!(full.starts_with(&format!("{FULL}\n")), "{full}");
    let bad = report(&dir, flags, &["--states-only", "maybe"]);
    assert_eq!(
        bad.lines().next(),
        Some("error states-only: maybe is neither yes nor no"),
        "{bad}"
    );
}
