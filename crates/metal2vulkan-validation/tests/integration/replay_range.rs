//! The player's report names the recording range a play uses, where it came from, and whether the
//! play ends there, and whether the stream has the host's display library beside it; every report
//! line goes to the standard error too (`docs/REPLAY-STREAM.md`, sections 1 and 10).

use metal2vulkan_validation::replay::{
    StreamWriter, FLAG_ANNOUNCED_BYTES_RECORDED, FLAG_COMPLETION_BYTES_RECORDED,
};
use metal2vulkan_validation::ScratchDir;
use std::fs::File;
use std::path::Path;
use std::process::Command;

/// The exit status and report of `command-replay play` over the one-call stream `first` with
/// `extra` options, given a device type no device has, so the play stops before any device opens.
/// Its standard error carries every line of the report, in order (section 10).
fn report(dir: &Path, first: &Path, run: &str, extra: &[&str]) -> (Option<i32>, String) {
    let out = dir.join(run);
    let played = Command::new(bin_exe!("command-replay"))
        .arg("play")
        .arg("--stream")
        .arg(first)
        .arg("--out")
        .arg(&out)
        .args(["--device-type", "none", "--shader-cache"])
        .arg(dir.join("cache"))
        .args(extra)
        .output()
        .unwrap();
    let text = std::fs::read_to_string(out.join("report.txt")).unwrap();
    let stderr = String::from_utf8_lossy(&played.stderr);
    let mut rest = stderr.lines();
    for line in text.lines() {
        assert!(
            rest.any(|written| written == line),
            "{line:?} not on the standard error in order: {stderr}"
        );
    }
    (played.status.code(), text)
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
        assert!(lines[2].starts_with("error device-type: "), "{text}");
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

/// The line after the range says whether the stream has the host's display library beside it,
/// `NAME.display.metallib`, and its length, SHA-256 and git blob id.
#[test]
fn a_report_names_the_display_library_beside_the_stream() {
    let dir = ScratchDir::new("replay-display-report").unwrap();
    let first = dir.path().join("s.0000.m2vr");
    let flags = FLAG_ANNOUNCED_BYTES_RECORDED | FLAG_COMPLETION_BYTES_RECORDED;
    let mut writer = StreamWriter::with_flags(File::create(&first).unwrap(), flags).unwrap();
    writer.call("commit", 1, 0, &[], &[]).unwrap();
    writer.finish().unwrap();
    let line = |run: &str| {
        let (code, text) = report(dir.path(), &first, run, &[]);
        assert_eq!(code, Some(2), "{text}");
        text.lines().nth(1).unwrap_or_default().to_owned()
    };
    assert_eq!(
        line("none"),
        "display library none: the host's library from source is skipped"
    );
    std::fs::write(dir.path().join("s.display.metallib"), b"MTLB!").unwrap();
    // `git hash-object` of the five bytes names the same blob.
    assert_eq!(
        line("beside"),
        "display library the stream's sidecar, 5 bytes, sha256 \
         6cfdce7eee577852d35aebca1eb2891e635bec8546b3038e31348ff5f9a5ce0c, git blob 8d7ffd7ee222d331e5a1c416934ff13a9ab9b1be"
    );
}
