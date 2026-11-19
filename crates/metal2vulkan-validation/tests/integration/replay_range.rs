//! The player's report names the recording range a play uses, where it came from, and whether the
//! play ends there (`docs/REPLAY-STREAM.md`, sections 1 and 10).

use metal2vulkan_validation::replay::{
    StreamWriter, FLAG_ANNOUNCED_BYTES_RECORDED, FLAG_COMPLETION_BYTES_RECORDED,
};
use metal2vulkan_validation::ScratchDir;
use std::fs::File;
use std::path::Path;
use std::process::Command;

/// The exit status and report of `command-replay play` over the one-call stream `first` with
/// `extra` options, given a device type no device has, so the play stops before any device opens.
fn report(dir: &Path, first: &Path, run: &str, extra: &[&str]) -> (Option<i32>, String) {
    let out = dir.join(run);
    let status = Command::new(bin_exe!("command-replay"))
        .arg("play")
        .arg("--stream")
        .arg(first)
        .arg("--out")
        .arg(&out)
        .args(["--device-type", "none", "--shader-cache"])
        .arg(dir.join("cache"))
        .args(extra)
        .status()
        .unwrap();
    let text = std::fs::read_to_string(out.join("report.txt")).unwrap();
    (status.code(), text)
}

/// With no sidecar and no table row the range is the whole stream; a sidecar's `last-frame F` ends
/// the play there unless `--past-range yes` takes it past; a sidecar of neither form stops the
/// play before it starts.
#[test]
fn a_report_names_the_recording_range_the_play_used() {
    let dir = ScratchDir::new("replay-range-report").unwrap();
    let first = dir.path().join("s.0000.m2vr");
    let flags = FLAG_ANNOUNCED_BYTES_RECORDED | FLAG_COMPLETION_BYTES_RECORDED;
    let mut writer = StreamWriter::with_flags(File::create(&first).unwrap(), flags).unwrap();
    writer.call("commit", 1, 0, &[], &[]).unwrap();
    writer.finish().unwrap();
    let line = |run: &str, extra: &[&str]| {
        let (code, text) = report(dir.path(), &first, run, extra);
        assert_eq!(code, Some(2), "{text}");
        let lines = text.lines().map(str::to_owned).collect::<Vec<_>>();
        assert!(lines[1].starts_with("error device-type: "), "{text}");
        lines[0].clone()
    };
    assert_eq!(
        line("unstated", &[]),
        "range whole (no sidecar and no table row)"
    );
    std::fs::write(dir.path().join("s.range"), "last-frame 3\n").unwrap();
    assert_eq!(
        line("ends", &[]),
        "range last-frame 3 (the stream's sidecar); the play ends there"
    );
    assert_eq!(
        line("past", &["--past-range", "yes"]),
        "range last-frame 3 (the stream's sidecar); --past-range yes plays past it"
    );
    std::fs::write(dir.path().join("s.range"), "frames 3\n").unwrap();
    let (code, text) = report(dir.path(), &first, "bad", &[]);
    assert_eq!(code, Some(2), "{text}");
    assert!(text.starts_with("error range: "), "{text}");
}
