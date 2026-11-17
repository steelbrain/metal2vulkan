//! Replay streams (`docs/REPLAY-STREAM.md`): play one on a Vulkan device through the old runtime,
//! or print its records.
//!
//! `command-replay play --stream FILE --out DIR --device-type cpu|gpu --shader-cache DIR
//! [--blobs DIR]` plays the stream whose first segment is `FILE` and writes `DIR/report.txt`:
//! `device NAME`, the player's report lines, `cut SEGMENT OFFSET` where the last segment ends in
//! a record cut short, then `ok` or `error WHAT`. The exit status is 0 for `ok`, 1 for a play that
//! stopped and 2 for a harness failure (a bad command line, no device, an unreadable stream).
//!
//! `command-replay dump --stream FILE` prints each call record as a line (a blob as its hash),
//! then `cut SEGMENT OFFSET` where the last segment ends in a record cut short.
//!
//! GUEST DATA: a stream recorded from a guest, its blobs and the reports are the consumer's: they
//! stay on the hosts the guest data rule allows and are never committed.

use metal2vulkan_runtime::{DeviceOptions, ShaderCacheOptions};
use metal2vulkan_validation::candidate::runtime_case::open_device;
use metal2vulkan_validation::gpu::DeviceType;
use metal2vulkan_validation::replay::{
    segments, OldRuntime, PlayError, Player, StreamError, StreamReader,
};
use std::collections::BTreeMap;
use std::io::Write as _;
use std::path::PathBuf;

const USAGE: &str = "usage: command-replay play --stream FILE --out DIR --device-type cpu|gpu \
                     --shader-cache DIR [--blobs DIR]\n       command-replay dump --stream FILE";

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (run, required): (fn(&BTreeMap<String, String>) -> i32, &[&str]) =
        match args.first().map(String::as_str) {
            Some("play") => (play, &["stream", "out", "device-type", "shader-cache"]),
            Some("dump") => (dump, &["stream"]),
            _ => {
                eprintln!("{USAGE}");
                std::process::exit(2);
            }
        };
    let code = match options(&args[1..], required) {
        Ok(options) => run(&options),
        Err(why) => {
            eprintln!("command-replay: {why}\n{USAGE}");
            2
        }
    };
    std::process::exit(code);
}

fn options(args: &[String], required: &[&str]) -> Result<BTreeMap<String, String>, String> {
    let mut out = BTreeMap::new();
    let mut it = args.iter();
    while let Some(flag) = it.next() {
        let name = flag
            .strip_prefix("--")
            .ok_or_else(|| format!("unexpected {flag:?}"))?;
        let value = it.next().ok_or_else(|| format!("{flag} needs a value"))?;
        out.insert(name.to_string(), value.clone());
    }
    for name in required {
        if !out.contains_key(*name) {
            return Err(format!("--{name} is required"));
        }
    }
    Ok(out)
}

fn reader(options: &BTreeMap<String, String>) -> Result<StreamReader, StreamError> {
    let first = PathBuf::from(&options["stream"]);
    StreamReader::open(&segments(&first), options.get("blobs").map(PathBuf::from))
}

fn dump(options: &BTreeMap<String, String>) -> i32 {
    let mut stdout = std::io::stdout().lock();
    let ended = reader(options).and_then(|mut reader| {
        while let Some(call) = reader.next_call()? {
            if writeln!(stdout, "{call}").is_err() {
                return Ok(());
            }
        }
        if let Some((segment, offset)) = reader.truncated() {
            let _ = writeln!(stdout, "cut {} {offset}", segment.display());
        }
        Ok(())
    });
    match ended {
        Ok(()) => 0,
        Err(error) => {
            eprintln!("command-replay: {error}");
            2
        }
    }
}

/// The report file, written as the play goes.
struct ReportFile {
    file: std::fs::File,
}

impl ReportFile {
    fn line(&mut self, line: &str) {
        let _ = writeln!(self.file, "{line}");
        let _ = self.file.flush();
    }
}

fn play(options: &BTreeMap<String, String>) -> i32 {
    let out = PathBuf::from(&options["out"]);
    if let Err(error) = std::fs::create_dir_all(&out) {
        eprintln!("command-replay: {}: {error}", out.display());
        return 2;
    }
    let mut report = match std::fs::File::create(out.join("report.txt")) {
        Ok(file) => ReportFile { file },
        Err(error) => {
            eprintln!("command-replay: report.txt: {error}");
            return 2;
        }
    };
    let device_type = match DeviceType::parse(&options["device-type"]) {
        Ok(device_type) => device_type,
        Err(why) => {
            report.line(&format!("error device-type: {why}"));
            return 2;
        }
    };
    let mut reader = match reader(options) {
        Ok(reader) => reader,
        Err(error) => {
            report.line(&format!("error {error}"));
            return 2;
        }
    };
    let mut device_options = DeviceOptions::default();
    device_options.shader_cache = ShaderCacheOptions::in_directory(&options["shader-cache"]);
    let device = match open_device(device_type, device_options) {
        Ok(device) => device,
        Err(error) => {
            report.line(&format!("error device: {error}"));
            return 2;
        }
    };
    report.line(&format!("device {}", device.name()));
    let mut backend = OldRuntime::new(device);
    let (played, ended) = Player::new(&mut backend).play(&mut reader);
    for line in played.lines() {
        report.line(&line);
    }
    if let Some((segment, offset)) = reader.truncated() {
        report.line(&format!("cut {} {offset}", segment.display()));
    }
    match ended {
        Ok(()) => {
            report.line("ok");
            0
        }
        Err(error) => {
            report.line(&format!("error {error}"));
            match error {
                PlayError::Stream(_) => 2,
                _ => 1,
            }
        }
    }
}
