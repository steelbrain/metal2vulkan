//! Replay streams (`docs/REPLAY-STREAM.md`): play one on a Vulkan device through the old runtime
//! or the new one, or print its records.
//!
//! `command-replay play --stream FILE --out DIR --device-type cpu|gpu --shader-cache DIR
//! [--blobs DIR] [--runtime old|rt] [--snapshots yes|no] [--managed one-copy|two-copy]
//! [--until-frame F] [--past-range yes|no] [--gpu-wait-timeout SECONDS]` plays the stream whose
//! first segment is `FILE` through the old runtime (the default) or the new one (`rt`), with
//! `--gpu-wait-timeout` bounding each device wait (10 s unless it says otherwise; a CPU device's
//! work can take longer than a GPU's), with `--snapshots no` taking no
//! `m2v.privateBytes` snapshot on either (so two runtimes' plays do the same work), `--managed`
//! telling either runtime how the recording device kept Managed buffers where the stream has no
//! `m2v.deviceIdentity` record (one copy: unified memory; two copies: a GPU copy; with neither, a
//! Managed buffer stops the new runtime's play and plays as the old runtime's own), and `--until-frame` stopping the play after frame `F`'s present
//! (frames counted from 0). The play ends at the stream's recording range (its sidecar's, else
//! its row of `RECORDED_RANGES`, else the whole stream) unless `--past-range yes` takes it past.
//! It writes `DIR/report.txt`, each line to the standard error too: first, where the stream's
//! header lacks `FLAG_COMPLETION_BYTES_RECORDED`, the line `recorded before the recorder's
//! completion fix: the stream may differ from what the program did` by itself, then `range ...`
//! (the recording range, where it came from, and whether the play ends there), then `device NAME
//! runtime old|rt snapshots yes|no`, each frame's `frame` and `end` lines as the frame ends
//! (flushed, so a play that never ends leaves the frames it finished), `addresses ...` (the
//! written words that look like addresses, which play as recorded, the address arguments and those
//! moved, and the objects whose ID differs from the recording's), the player's other report lines,
//! the
//! backend's own lines, `until-frame F` where the play stopped there (at `--until-frame` or at the
//! recording range), `cut SEGMENT OFFSET` where
//! the last
//! segment ends in a record cut
//! short, `skipped ...` where the backend skipped what its runtime does not build (such a play is
//! no gate result), `gaps ...` where it met a typed gap, on the new runtime where the play
//! finished its range and skipped nothing the matrix's judged metrics (`M2V-MATRIX metric
//! replay-rt/STEM/RANGE/NAME N`, `docs/REPLAY-STREAM.md`, section 10), then `ok`, `ids: ...` or
//! `error WHAT`, the skipped line and this one ending `; not a verdict past frame F, call C` where
//! the backend skipped anything (`F` and `C` the first skip's frame and stream position); a play
//! that ends where an object's ID differs, the compute pipeline states' typed gap aside, ends
//! `ids: played to the end; ...`. The exit status is 0 for `ok`, 1 for a play that stopped, 2 for
//! a harness failure (a bad command line, no device, an unreadable stream or sidecar) and 3 for
//! `ids`.
//!
//! With `--dump-command-buffers FIRST..LAST --dump-dir DIR`, the play also writes into `DIR`, once
//! each command buffer whose stream id lies in `FIRST..=LAST` completed, every buffer its encoders
//! named, with the ranges of it the cover rule held covered then (`docs/REPLAY-STREAM.md`, section
//! 10).
//!
//! `command-replay compare-dumps --a DIR --b DIR` compares two plays' dumps word by word, leaving
//! out the words either play held covered, and prints a line per buffer that differs outside every
//! cover, per dump only one play has and per dump a play could not read, then a summary ending
//! `agree` or `DIFFER`. The exit status is 0 where they agree, 1 where they differ, 2 for a bad
//! command line or an unreadable dump.
//!
//! `command-replay dump --stream FILE` prints each call record as a line (a blob as its hash),
//! then `cut SEGMENT OFFSET` where the last segment ends in a record cut short.
//!
//! `command-replay shapes --stream FILE` prints, for each selector, how many calls had each shape
//! of arguments and answers (`Value::shape`: kinds and map keys, no data), most first: what a
//! stream asks of a player without what it holds.
//!
//! GUEST DATA: a stream recorded from a guest, its blobs and the reports are the consumer's: they
//! stay on the hosts the guest data rule allows and are never committed.

use metal2vulkan_runtime::{DeviceOptions, ShaderCacheOptions};
use metal2vulkan_validation::candidate::runtime_case::open_device;
use metal2vulkan_validation::gpu::wait::{parse_wait_timeout, set_wait_timeout, WAIT_TIMEOUT_FLAG};
use metal2vulkan_validation::gpu::DeviceType;
use metal2vulkan_validation::replay::{
    compare_dumps, compare_plays, display_library, judged_records, recording_range, segments,
    Backend, ManagedCopies, NewRuntime, OldRuntime, PlayError, Player, RangeSource, RecordingRange,
    StreamError, StreamReader, Value, FLAG_COMPLETION_BYTES_RECORDED, RECORDED_RANGES,
};
use std::collections::BTreeMap;
use std::io::Write as _;
use std::ops::RangeInclusive;
use std::path::{Path, PathBuf};

const USAGE: &str = "usage: command-replay play --stream FILE --out DIR --device-type cpu|gpu \
                     --shader-cache DIR [--blobs DIR] [--runtime old|rt] [--snapshots yes|no] \
                     [--managed one-copy|two-copy] [--until-frame F] [--past-range yes|no] \
                     [--gpu-wait-timeout SECONDS] \
                     [--dump-command-buffers FIRST..LAST --dump-dir DIR]\n       \
                     command-replay compare-dumps --a DIR --b DIR\n       \
                     command-replay compare-plays --a REPORT --b REPORT [--frame F]\n       \
                     command-replay dump \
                     --stream FILE\n       command-replay shapes --stream FILE";

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let code = match args.first().map(String::as_str) {
        Some("play") => {
            let required = ["stream", "out", "device-type", "shader-cache"];
            with_options(&args[1..], &required, play)
        }
        Some("dump") => with_options(&args[1..], &["stream"], dump),
        Some("compare-dumps") => with_options(&args[1..], &["a", "b"], compare),
        Some("compare-plays") => with_options(&args[1..], &["a", "b"], compare_two_plays),
        Some("shapes") => with_options(&args[1..], &["stream"], shapes),
        _ => {
            eprintln!("{USAGE}");
            2
        }
    };
    std::process::exit(code);
}

/// `run` on the options `args` give, or 2 where they lack one of `required`.
fn with_options(
    args: &[String],
    required: &[&str],
    run: fn(&BTreeMap<String, String>) -> i32,
) -> i32 {
    match options(args, required) {
        Ok(options) => run(&options),
        Err(why) => {
            eprintln!("command-replay: {why}\n{USAGE}");
            2
        }
    }
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

/// `compare-dumps`: two plays' dumps compared, a line each, then the summary; 0 where they agree,
/// 1 where they differ, 2 where a dump cannot be read.
fn compare(options: &BTreeMap<String, String>) -> i32 {
    let (a, b) = (PathBuf::from(&options["a"]), PathBuf::from(&options["b"]));
    match compare_dumps(&a, &b) {
        Ok(comparison) => {
            for line in comparison.lines() {
                println!("{line}");
            }
            match comparison.agrees() {
                true => 0,
                false => 1,
            }
        }
        Err(error) => {
            eprintln!("command-replay: compare-dumps: {error}");
            2
        }
    }
}

/// `compare-plays`: two plays' reports (each a `report.txt` or the directory holding it) compared
/// for a timing verdict, a line each; 0 where the frame `--frame` names has a verdict (or none is
/// named), 1 where it has none, 2 where a report cannot be read.
fn compare_two_plays(options: &BTreeMap<String, String>) -> i32 {
    let frame = match options.get("frame").map(|f| f.parse::<u64>()) {
        None => None,
        Some(Ok(frame)) => Some(frame),
        Some(Err(_)) => {
            eprintln!("command-replay: --frame takes a frame number\n{USAGE}");
            return 2;
        }
    };
    let report = |name: &str| {
        let path = PathBuf::from(&options[name]);
        let path = match path.is_dir() {
            true => path.join("report.txt"),
            false => path,
        };
        std::fs::read_to_string(&path).map_err(|error| format!("{}: {error}", path.display()))
    };
    let (a, b) = match (report("a"), report("b")) {
        (Ok(a), Ok(b)) => (a, b),
        (Err(why), _) | (_, Err(why)) => {
            eprintln!("command-replay: compare-plays: {why}");
            return 2;
        }
    };
    let (lines, verdict) = compare_plays(&a, &b, frame);
    for line in lines {
        println!("{line}");
    }
    match verdict {
        true => 0,
        false => 1,
    }
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

/// The most shapes `shapes` prints of one selector.
const SHAPES_SHOWN: usize = 30;

fn shapes(options: &BTreeMap<String, String>) -> i32 {
    let mut counted: BTreeMap<(String, String), u64> = BTreeMap::new();
    let ended = reader(options).and_then(|mut reader| {
        while let Some(call) = reader.next_call()? {
            let args: Vec<String> = call.args.iter().map(Value::shape).collect();
            let answers: Vec<String> = (call.answers.iter())
                .map(|(key, value)| format!(" -> {key}: {}", value.shape()))
                .collect();
            let shape = format!("{}{}", args.join(" "), answers.concat());
            *counted
                .entry((call.selector.to_string(), shape))
                .or_default() += 1;
        }
        Ok(reader.truncated().cloned())
    });
    let mut by_selector: BTreeMap<String, Vec<(u64, String)>> = BTreeMap::new();
    for ((selector, shape), count) in counted {
        by_selector
            .entry(selector)
            .or_default()
            .push((count, shape));
    }
    let mut stdout = std::io::stdout().lock();
    for (selector, mut shapes) in by_selector {
        shapes.sort_unstable_by(|a, b| b.cmp(a));
        let total: u64 = shapes.iter().map(|(count, _)| count).sum();
        let _ = writeln!(stdout, "{selector}: {total} calls, {} shapes", shapes.len());
        let more = shapes.len().saturating_sub(SHAPES_SHOWN);
        for (count, shape) in shapes.into_iter().take(SHAPES_SHOWN) {
            let _ = writeln!(stdout, "  {count} {shape}");
        }
        if more > 0 {
            let _ = writeln!(stdout, "  ... {more} more shapes");
        }
    }
    match ended {
        Ok(cut) => {
            if let Some((segment, offset)) = cut {
                let _ = writeln!(stdout, "cut {} {offset}", segment.display());
            }
            0
        }
        Err(error) => {
            eprintln!("command-replay: {error}");
            2
        }
    }
}

/// The line by itself at the top of every report of a stream whose header lacks
/// `FLAG_COMPLETION_BYTES_RECORDED`, so that no table relayed from the report loses it.
const BEFORE_COMPLETION_FIX: &str =
    "recorded before the recorder's completion fix: the stream may differ from what the program did";

/// The report's `display library` line of a stream with the sidecar `metallib`: its length, its
/// SHA-256, and its git blob id (the SHA-1 of `blob LENGTH\0` and the bytes), which names the
/// file in the host's tree.
fn display_line(metallib: &[u8]) -> String {
    use sha1::Digest as _;
    let sha256 = metal2vulkan_validation::replay::hex(&sha2::Sha256::digest(metallib));
    let mut blob = sha1::Sha1::new();
    blob.update(format!("blob {}\0", metallib.len()).as_bytes());
    blob.update(metallib);
    let blob = metal2vulkan_validation::replay::hex(&blob.finalize());
    format!(
        "display library the stream's sidecar, {} bytes, sha256 {sha256}, git blob {blob}",
        metallib.len()
    )
}

/// The report's `range` line: the recording range the play used, where it came from, and whether
/// the play ends there or `--past-range yes` takes it past.
fn range_line(range: RecordingRange, source: RangeSource, past: bool) -> String {
    let source = match source {
        RangeSource::Sidecar => "the stream's sidecar".to_owned(),
        RangeSource::Table(label) => format!("the player's table, stream {label}"),
        RangeSource::Unstated => "no sidecar and no table row".to_owned(),
    };
    match (range, past) {
        (RecordingRange::Whole, _) => format!("range whole ({source})"),
        (RecordingRange::LastFrame(last), false) => {
            format!("range last-frame {last} ({source}); the play ends there")
        }
        (RecordingRange::LastFrame(last), true) => {
            format!("range last-frame {last} ({source}); --past-range yes plays past it")
        }
    }
}

/// The report file, written as the play goes: every line it takes, the frame lines among them,
/// goes to the standard error too, so a caller that loses the file keeps the lines.
struct ReportFile {
    file: std::fs::File,
}

impl ReportFile {
    fn line(&mut self, line: &str) {
        let _ = writeln!(self, "{line}");
        let _ = self.flush();
    }
}

impl std::io::Write for ReportFile {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.file.write_all(bytes)?;
        let _ = std::io::stderr().write_all(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.file.flush()
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
    let mut reader = match reader(options) {
        Ok(reader) => reader,
        Err(error) => {
            report.line(&format!("error {error}"));
            return 2;
        }
    };
    if reader.flags() & FLAG_COMPLETION_BYTES_RECORDED == 0 {
        report.line(BEFORE_COMPLETION_FIX);
    }
    let (range, source) = match recording_range(&PathBuf::from(&options["stream"]), RECORDED_RANGES)
    {
        Ok(found) => found,
        Err(why) => {
            report.line(&format!("error range: {why}"));
            return 2;
        }
    };
    let past = match options.get("past-range").map(String::as_str) {
        None | Some("no") => false,
        Some("yes") => true,
        Some(other) => {
            report.line(&format!("error past-range: {other} is neither yes nor no"));
            return 2;
        }
    };
    report.line(&range_line(range, source, past));
    // The host's display library, which its library from source of no arguments is made of.
    let display = display_library(&PathBuf::from(&options["stream"]));
    let display = match std::fs::read(&display) {
        Ok(metallib) => {
            report.line(&display_line(&metallib));
            Some(metallib)
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            report.line("display library none: the host's library from source is skipped");
            None
        }
        Err(error) => {
            report.line(&format!(
                "error display library: {}: {error}",
                display.display()
            ));
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
    let flag = WAIT_TIMEOUT_FLAG.trim_start_matches("--");
    if let Some(seconds) = options.get(flag) {
        if let Err(why) = parse_wait_timeout(seconds).and_then(set_wait_timeout) {
            report.line(&format!("error {flag}: {why}"));
            return 2;
        }
    }
    let snapshots = match options.get("snapshots").map(String::as_str) {
        None | Some("yes") => true,
        Some("no") => false,
        Some(other) => {
            report.line(&format!("error snapshots: {other} is neither yes nor no"));
            return 2;
        }
    };
    let managed = match options.get("managed").map(String::as_str) {
        None => None,
        Some("one-copy") => Some(ManagedCopies::One),
        Some("two-copy") => Some(ManagedCopies::Two),
        Some(other) => {
            report.line(&format!(
                "error managed: {other} is neither one-copy nor two-copy"
            ));
            return 2;
        }
    };
    let until = match options.get("until-frame").map(|f| f.parse::<usize>()) {
        None => None,
        Some(Ok(last)) => Some(last),
        Some(Err(_)) => {
            report.line(&format!(
                "error until-frame: {} is no frame number",
                options["until-frame"]
            ));
            return 2;
        }
    };
    let until = range.last_frame(past, until);
    let dump = match (options.get("dump-command-buffers"), options.get("dump-dir")) {
        (None, None) => None,
        (Some(listed), Some(dir)) => {
            let parsed = listed.split_once("..").and_then(|(first, last)| {
                Some(first.parse::<u64>().ok()?..=last.parse::<u64>().ok()?)
            });
            let Some(command_buffers) = parsed else {
                report.line(&format!(
                    "error dump-command-buffers: {listed} is no FIRST..LAST"
                ));
                return 2;
            };
            if let Err(error) = std::fs::create_dir_all(dir) {
                report.line(&format!("error dump-dir: {dir}: {error}"));
                return 2;
            }
            Some((command_buffers, PathBuf::from(dir)))
        }
        _ => {
            report.line("error dump: --dump-command-buffers and --dump-dir go together");
            return 2;
        }
    };
    // The first line says which mode ran: a timing table never comes from a snapshotting play
    // unnoticed.
    let mode = match snapshots {
        true => "snapshots yes",
        false => "snapshots no",
    };
    let cache = PathBuf::from(&options["shader-cache"]);
    match options.get("runtime").map(String::as_str) {
        None | Some("old") => {
            let mut device_options = DeviceOptions::default();
            device_options.shader_cache =
                ShaderCacheOptions::in_directory(&options["shader-cache"]);
            let device = match open_device(device_type, device_options) {
                Ok(device) => device,
                Err(error) => {
                    report.line(&format!("error device: {error}"));
                    return 2;
                }
            };
            report.line(&format!("device {} runtime old {mode}", device.name()));
            let mut backend = OldRuntime::new(device, device_type);
            if let Some(copies) = managed {
                backend = backend.with_managed(copies);
            }
            if let Some(metallib) = display {
                backend = backend.with_display_library(metallib);
            }
            run(
                &mut backend,
                &mut reader,
                &mut report,
                (snapshots, until, dump, None),
            )
        }
        Some("rt") => {
            let mut backend = match open_rt(device_type, cache) {
                Ok((backend, name)) => {
                    report.line(&format!("device {name} runtime rt {mode}"));
                    let backend = match managed {
                        Some(copies) => backend.with_managed(copies),
                        None => backend,
                    };
                    match display {
                        Some(metallib) => backend.with_display_library(metallib),
                        None => backend,
                    }
                }
                Err(error) => {
                    report.line(&format!("error device: {error}"));
                    return 2;
                }
            };
            let first = PathBuf::from(&options["stream"]);
            let code = run(
                &mut backend,
                &mut reader,
                &mut report,
                (snapshots, until, dump, Some(&first)),
            );
            // The device ends here, not in `exit()`, where the driver's own exit teardown would
            // run beside the runtime's threads. Its gauges read after it dropped go out as the
            // matrix's lines, which the matrix's zero-at-drop rule reads.
            let ended = backend.end();
            for line in ended.lines() {
                report.line(&line);
            }
            match ended.result() {
                Ok(()) => code,
                Err(error) => {
                    report.line(&format!("error teardown: {error}"));
                    code.max(1)
                }
            }
        }
        Some(other) => {
            report.line(&format!("error runtime: {other} is neither old nor rt"));
            2
        }
    }
}

/// The new runtime's backend on the first physical device of `device_type` that opens (a GPU:
/// discrete, then integrated, then virtual), its compiled kernels kept in `cache`, and its name.
/// A GPU opens only while the hang tripwire is absent, as the old runtime's device does.
fn open_rt(device_type: DeviceType, cache: PathBuf) -> Result<(NewRuntime, String), String> {
    use ash::vk::PhysicalDeviceType as Kind;
    if device_type == DeviceType::Gpu {
        metal2vulkan_validation::gpu::tripwire::refuse_if_tripped()?;
    }
    let instance = metal2vulkan_rt::Instance::new(&[]).map_err(|error| error.to_string())?;
    let (_, vk_instance) = instance.raw();
    let ranks: &[Kind] = match device_type {
        DeviceType::Cpu => &[Kind::CPU],
        DeviceType::Gpu => &[Kind::DISCRETE_GPU, Kind::INTEGRATED_GPU, Kind::VIRTUAL_GPU],
    };
    let mut candidates = Vec::new();
    for physical in instance
        .physical_devices()
        .map_err(|error| error.to_string())?
    {
        // SAFETY: `physical` is one of the instance's physical devices, and the instance is live.
        let properties = unsafe { vk_instance.get_physical_device_properties(physical.raw()) };
        if let Some(rank) = ranks
            .iter()
            .position(|&kind| kind == properties.device_type)
        {
            let name = (properties.device_name_as_c_str()).map_or_else(
                |_| "?".to_owned(),
                |name| name.to_string_lossy().into_owned(),
            );
            candidates.push((rank, name, physical));
        }
    }
    candidates.sort_by_key(|(rank, ..)| *rank);
    let mut refused = Vec::new();
    for (_, name, physical) in candidates {
        let mut options = metal2vulkan_rt::DeviceOptions::default();
        options.pipeline_cache_dir = Some(cache.clone());
        let opened = metal2vulkan_rt::Device::new(&physical, options)
            .and_then(|device| NewRuntime::new(device, device_type));
        match opened {
            Ok(backend) => return Ok((backend, name)),
            Err(why) => refused.push(format!("{name}: {why}")),
        }
    }
    Err(format!(
        "no {device_type:?} device opens: {}",
        refused.join("; ")
    ))
}

/// The command buffers a play dumps and where (`--dump-command-buffers`, `--dump-dir`).
type Dumped = Option<(RangeInclusive<u64>, PathBuf)>;

/// Play `reader` through `backend` into `report`, taking snapshots or not, to the last frame where
/// one is given, dumping the command buffers where asked, and on the new runtime, where `judged`
/// names the stream's first segment, writing the matrix's judged metrics of a play that finished
/// its range (`judged_records`): the exit status.
fn run<B: Backend>(
    backend: &mut B,
    reader: &mut StreamReader,
    report: &mut ReportFile,
    (snapshots, until, dump, judged): (bool, Option<usize>, Dumped, Option<&Path>),
) -> i32 {
    let player = Player::new(backend).with_frame_lines(&mut *report);
    let player = match snapshots {
        true => player,
        false => player.without_snapshots(),
    };
    let player = match until {
        Some(last) => player.until_frame(last),
        None => player,
    };
    let player = match dump {
        Some((command_buffers, dir)) => player.dumping(command_buffers, dir),
        None => player,
    };
    let (played, ended) = player.play(reader);
    let counters = backend.counters();
    // The objects whose ID the program saw that the runtime gives otherwise: a word that names one
    // in the program's memory names something else in the play. The IDs nothing in the play can
    // read are aside: the compute pipeline states' (a typed gap) and the skipped objects'.
    let unread = |key: &str| counters.get(key).copied().unwrap_or(0);
    let unread_ids = unread("gap_pipeline_ids") + unread("skipped_resource_ids");
    let differing_ids = (played.ids.len() as u64).saturating_sub(unread_ids);
    let stream = &played.total.stream;
    let fixed = |key: &str| stream.get(key).copied().unwrap_or(0);
    let notes = match played.escape_notes {
        0 => "the stream predates the escape note".to_owned(),
        notes => format!("escape notes {notes}"),
    };
    report.line(&format!(
        "addresses address_looking_words={} misaligned_address_words={} address_arguments={} \
         relocated_words={} differing_ids={differing_ids}: written words play as recorded; {notes}",
        fixed("address_looking_words"),
        fixed("misaligned_address_words"),
        fixed("address_arguments"),
        fixed("relocated_words"),
    ));
    for line in played.lines_but_frames() {
        report.line(&line);
    }
    for line in backend.lines() {
        report.line(&line);
    }
    if let Some(last) = until.filter(|&last| played.frames.len() > last) {
        report.line(&format!("until-frame {last}"));
    }
    if let Some((segment, offset)) = reader.truncated() {
        report.line(&format!("cut {} {offset}", segment.display()));
    }
    let counted = |prefix: &str| -> Vec<String> {
        (counters.iter())
            .filter(|(name, count)| name.starts_with(prefix) && **count > 0)
            .map(|(name, count)| format!("{name}={count}"))
            .collect()
    };
    let skips = counted("skipped_");
    // Past the first skip the play is no verdict: the skipped line and the verdict line say so.
    let note = (played.verdict_note()).map_or(String::new(), |note| format!("; {note}"));
    if !skips.is_empty() {
        report.line(&format!(
            "skipped {}: no gate result{note}",
            skips.join(" ")
        ));
    }
    let gaps = counted("gap_");
    if !gaps.is_empty() {
        report.line(&format!("gaps {}", gaps.join(" ")));
    }
    if let Some(first) = judged {
        for line in judged_records(first, &played, &ended, until) {
            report.line(&line);
        }
    }
    match ended {
        // Played to the end, the IDs a shader reads differing from the program's: no fault, hang
        // or refusal, its own status.
        Ok(()) if differing_ids > 0 => {
            report.line(&format!(
                "ids: played to the end; {differing_ids} objects' gpuResourceID differ from the \
                 recording's{note}"
            ));
            3
        }
        Ok(()) => {
            report.line(&format!("ok{note}"));
            0
        }
        Err(error) => {
            report.line(&format!("error {error}{note}"));
            match error {
                PlayError::Stream(_) => 2,
                _ => 1,
            }
        }
    }
}
