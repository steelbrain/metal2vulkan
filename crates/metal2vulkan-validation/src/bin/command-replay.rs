//! Replay streams (`docs/REPLAY-STREAM.md`): play one on a Vulkan device through the old runtime
//! or the new one, or print its records.
//!
//! `command-replay play --stream FILE --out DIR --device-type cpu|gpu --shader-cache DIR
//! [--blobs DIR] [--runtime old|rt] [--snapshots yes|no] [--managed one-copy|two-copy]
//! [--until-frame F]` plays the stream whose first segment is `FILE` through the old runtime (the
//! default) or the new one (`rt`), with `--snapshots no` taking no `m2v.privateBytes` snapshot on
//! either (so two runtimes' plays do the same work), `--managed` telling the new runtime how the
//! recording device kept Managed buffers where the stream has no `m2v.deviceIdentity` record (one
//! copy: unified memory; two copies: a GPU copy; with neither, a Managed buffer stops the play),
//! and `--until-frame` stopping the play after frame `F`'s present (frames counted from 0), and
//! writes `DIR/report.txt`: first, where the stream's header lacks
//! `FLAG_COMPLETION_BYTES_RECORDED`, the line `recorded before the recorder's completion fix: the
//! stream may differ from what the program did` by itself, then
//! `device NAME runtime old|rt snapshots yes|no`, each frame's `frame`
//! and `end` lines as the frame ends (flushed, so a play that never ends leaves the frames it
//! finished), `relocation ...` (the
//! words relocated, the address arguments, the writes into objects with no recorded range, and
//! the objects whose ID differs from the recording's), the player's other report lines, the
//! backend's own lines, `until-frame F` where the play stopped there, `cut SEGMENT OFFSET` where
//! the last
//! segment ends in a record cut
//! short, `skipped ...` where the backend skipped what its runtime does not build (such a play is
//! no gate result), `gaps ...` where it met a typed gap, then `ok`, `ids: ...` or `error WHAT`,
//! the skipped line and this one ending `; not a verdict past frame F, call C` where the backend
//! skipped anything (`F` and `C` the first skip's frame and stream position); a play that ends
//! where an object's ID differs, the compute pipeline states' typed gap aside, ends `ids: played
//! to the end; ...`. The exit status is 0 for `ok`, 1 for a play that stopped, 2 for a harness
//! failure (a bad command line, no device, an unreadable stream) and 3 for `ids`.
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
use metal2vulkan_validation::gpu::DeviceType;
use metal2vulkan_validation::replay::{
    segments, Backend, ManagedCopies, NewRuntime, OldRuntime, PlayError, Player, StreamError,
    StreamReader, Value, FLAG_COMPLETION_BYTES_RECORDED,
};
use std::collections::BTreeMap;
use std::io::Write as _;
use std::path::PathBuf;

const USAGE: &str = "usage: command-replay play --stream FILE --out DIR --device-type cpu|gpu \
                     --shader-cache DIR [--blobs DIR] [--runtime old|rt] [--snapshots yes|no] \
                     [--managed one-copy|two-copy] [--until-frame F]\n       command-replay dump \
                     --stream FILE\n       command-replay shapes --stream FILE";

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let code = match args.first().map(String::as_str) {
        Some("play") => {
            let required = ["stream", "out", "device-type", "shader-cache"];
            with_options(&args[1..], &required, play)
        }
        Some("dump") => with_options(&args[1..], &["stream"], dump),
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
    let device_type = match DeviceType::parse(&options["device-type"]) {
        Ok(device_type) => device_type,
        Err(why) => {
            report.line(&format!("error device-type: {why}"));
            return 2;
        }
    };
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
            run(&mut backend, &mut reader, &mut report, snapshots, until)
        }
        Some("rt") => {
            let mut backend = match open_rt(device_type, cache) {
                Ok((backend, name)) => {
                    report.line(&format!("device {name} runtime rt {mode}"));
                    match managed {
                        Some(copies) => backend.with_managed(copies),
                        None => backend,
                    }
                }
                Err(error) => {
                    report.line(&format!("error device: {error}"));
                    return 2;
                }
            };
            run(&mut backend, &mut reader, &mut report, snapshots, until)
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

/// Play `reader` through `backend` into `report`: the exit status.
fn run<B: Backend>(
    backend: &mut B,
    reader: &mut StreamReader,
    report: &mut ReportFile,
    snapshots: bool,
    until: Option<usize>,
) -> i32 {
    let player = Player::new(backend).with_frame_lines(&mut report.file);
    let player = match snapshots {
        true => player,
        false => player.without_snapshots(),
    };
    let player = match until {
        Some(last) => player.until_frame(last),
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
    report.line(&format!(
        "relocation relocated_words={} misaligned_address_words={} address_arguments={} \
         unrelocated_writes={} differing_ids={differing_ids}",
        fixed("relocated_words"),
        fixed("misaligned_address_words"),
        fixed("address_arguments"),
        fixed("unrelocated_writes"),
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
