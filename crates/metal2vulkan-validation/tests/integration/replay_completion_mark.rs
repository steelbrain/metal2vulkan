//! The player's report of a stream recorded before the recorder's completion fix begins with the
//! line that says so (`docs/REPLAY-STREAM.md`, sections 1, 6 and 10), and a fixed stream's does
//! not.

use metal2vulkan_validation::replay::{
    StreamWriter, FLAG_ANNOUNCED_BYTES_RECORDED, FLAG_COMPLETION_BYTES_RECORDED,
};
use metal2vulkan_validation::ScratchDir;
use std::fs::File;
use std::process::Command;

const MARK: &str =
    "recorded before the recorder's completion fix: the stream may differ from what the program did";

/// The report `command-replay play` writes of a one-call stream with header `flags`, given a device
/// type no device has, so the play stops before any device opens.
fn report(dir: &ScratchDir, flags: u32) -> String {
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
        .status()
        .unwrap();
    assert_eq!(status.code(), Some(2));
    std::fs::read_to_string(out.join("report.txt")).unwrap()
}

/// A stream whose header lacks bit 1 (flags 1, as every stream recorded before the fix reads)
/// begins its report with the line, by itself; a stream with it (flags 3) does not carry the line.
#[test]
fn a_report_of_a_stream_without_the_completion_bit_begins_with_the_line() {
    let dir = ScratchDir::new("replay-completion-mark").unwrap();
    let before = report(&dir, FLAG_ANNOUNCED_BYTES_RECORDED);
    let lines = before.lines().collect::<Vec<_>>();
    assert_eq!(lines.first(), Some(&MARK), "{before}");
    assert!(lines[1].starts_with("error device-type: "), "{before}");
    let fixed = report(
        &dir,
        FLAG_ANNOUNCED_BYTES_RECORDED | FLAG_COMPLETION_BYTES_RECORDED,
    );
    assert!(fixed.starts_with("error device-type: "), "{fixed}");
    assert!(!fixed.contains(MARK), "{fixed}");
}
