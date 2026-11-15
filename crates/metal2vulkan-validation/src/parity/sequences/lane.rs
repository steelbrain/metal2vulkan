//! The sequences' device tests. On the device the cfg selects, each seed's sequence runs on the
//! runtime under the validation layer (`sequences_<batch>_match_metal`, eight seeds a test); its
//! records are compared with every committed line of Metal's, and its bytes with the model's.
//! A synchronization hazard the layer reports (`METAL2VULKAN_LANE_VALIDATION=sync`) between
//! commands that share no byte is explained (`hazards`): printed as a `SEQUENCE-SYNC` line, it
//! fails nothing; every other message fails the seed.
//! A seed a line holds as a Metal-side gap ([`super::METAL_GAP`]: Metal's driver crashed running it) is
//! not compared with that line; it still runs, and where no line answers it, it must complete
//! without a refusal and without a validation message, and the test names it. A seed that fails
//! is shrunk (`minimize`) on the runtime and reported with its seed, its sequence and the smaller
//! one. On a Mac with `--cfg m2v_gpu`, `record_metal` records Metal's answers, each seed in a
//! child process under API validation; a child that dies of a signal records the seed's gap.

use super::super::api::Observation;
use super::super::record::{compare, read_lines, stale, Line, Value};
use super::super::records_path;
use super::generate::generate;
use super::hazards::{sort, MetalOrder, Sorted, Verdict};
use super::minimize::minimize;
use super::model::{byte_differences, records as model_records};
use super::program::{
    case_name, At, BufferSpec, Command, CommandBufferSpec, Encoder, Kernel, Kind, Place, Program,
    Role, Step, BUFFER_BYTES,
};
use super::run::{run, HostPages};
use super::runtime::RuntimeSeq;
use super::{fingerprint, metal_gap, NAME, SEEDS};
use crate::runtime_lane::sync_coverage::{blind_to_kernel_writes, layer_sees_kernel_writes};
use metal2vulkan_runtime::DebugMessage;
use std::time::{Duration, Instant};

/// The seeds one test runs.
const BATCH: u64 = 8;

/// The most candidates the minimizer runs on the runtime for one seed.
const ATTEMPTS: usize = 150;

/// The longest a batch spends minimizing; a seed past it is reported with what was found.
const MINIMIZE_BUDGET: Duration = Duration::from_secs(300);

/// How a run on the runtime failed, which a smaller sequence must fail alike.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Failure {
    /// A call was refused: the runtime's words.
    Refused(String),
    /// The validation layer reported: the first message's id name.
    Layer(String),
    /// Bytes differ from the model's.
    Bytes,
}

/// A run's records and the validation layer's messages, sorted (`hazards::sort`).
type Ran = (Vec<(String, Value)>, Sorted);

/// One run of `program` as `seed`'s on a runtime device of its own. The host pages outlive the
/// device.
fn on_runtime(seed: u64, program: &Program) -> Result<Ran, String> {
    on_device(seed, program, RuntimeSeq::open()?)
}

/// One run of `program` as `seed`'s on `api`, which it drops before the host pages.
fn on_device(seed: u64, program: &Program, api: RuntimeSeq) -> Result<Ran, String> {
    let pages = HostPages::of(program);
    let mut observation = Observation::new(case_name(seed), false);
    run(&api, program, &pages, &mut observation)?;
    let records = observation.finish(NAME);
    let messages = api.messages.clone();
    let encoded = api.encoded.take();
    let (mut taken, from) = api.taken();
    drop(api);
    drop(pages);
    taken.extend(
        messages
            .take()
            .into_iter()
            .map(|message| (from.clone(), message)),
    );
    Ok((records, sort(program, &encoded, taken)))
}

/// How the run of `program` failed, if it did: a refusal first, then the layer's report, then,
/// when `against_model`, bytes that differ from the model's (`differences`, named).
fn failure(
    seed: u64,
    program: &Program,
    records: &[(String, Value)],
    messages: &[DebugMessage],
    against_model: bool,
) -> Option<(Failure, Vec<String>)> {
    if let Some((name, detail)) = records.iter().find_map(|(name, value)| match value {
        Value::Refused { detail, .. } => Some((name, detail)),
        _ => None,
    }) {
        return Some((
            Failure::Refused(detail.clone()),
            vec![format!("{name}: refused")],
        ));
    }
    if let Some(first) = messages.first() {
        let reported = messages
            .iter()
            .map(|message| format!("{}: {}", message.id_name, message.text))
            .collect();
        return Some((Failure::Layer(first.id_name.clone()), reported));
    }
    if !against_model {
        return None;
    }
    let differences = byte_differences(records, &model_records(case_name(seed), program));
    (!differences.is_empty()).then_some((Failure::Bytes, differences))
}

/// Run each seed of `batch` on the runtime and compare it with Metal's lines and the model.
fn batch(batch: u64) {
    let lines = read_lines(&records_path(NAME)).unwrap();
    let mut problems = Vec::new();
    for line in &lines {
        let environment = &line.fingerprint.metal_environment;
        let stale = stale(&line.fingerprint, &fingerprint(environment).unwrap());
        if !stale.is_empty() {
            problems.push(format!(
                "{NAME} on {environment}: stale ({}): record it again",
                stale.join(", ")
            ));
        }
    }
    let deadline = Instant::now() + MINIMIZE_BUDGET;
    for seed in batch * BATCH..(batch + 1) * BATCH {
        if let Err(problem) = check(seed, &lines, deadline) {
            problems.push(problem);
        }
    }
    if lines.is_empty() {
        problems.push(format!("{NAME}: no Metal record: record them on a Mac"));
    }
    assert!(problems.is_empty(), "{}", problems.join("\n\n"));
}

/// One seed: its run on the runtime against each Metal line of it and against the model. A line
/// that holds the seed as a Metal-side gap is not compared; a seed no line answers is checked for
/// completion and validation messages only, since no Metal answer confirms the model's for it. A
/// failure is shrunk until `deadline`.
fn check(seed: u64, lines: &[Line], deadline: Instant) -> Result<(), String> {
    let program = generate(seed);
    let (records, sorted) = on_runtime(seed, &program)?;
    for (hazard, verdict) in &sorted.explained {
        eprintln!("SEQUENCE-SYNC seed {seed}: {hazard}: {verdict}");
    }
    let messages = sorted.failing;
    let prefix = format!("{}/", case_name(seed));
    let mut differences = Vec::new();
    let mut gaps = Vec::new();
    for line in lines {
        if let Some(detail) = metal_gap(&line.records, seed) {
            gaps.push(format!("{}: {detail}", line.fingerprint.metal_environment));
            continue;
        }
        let current = fingerprint(&line.fingerprint.metal_environment)?;
        let of_seed = Line {
            fingerprint: line.fingerprint.clone(),
            records: line
                .records
                .iter()
                .filter(|(name, _)| name.starts_with(&prefix))
                .cloned()
                .collect(),
        };
        if let Err(difference) = compare(std::slice::from_ref(&of_seed), &current, &records) {
            differences.push(difference);
        }
    }
    let against_model = gaps.len() < lines.len();
    let failed = failure(seed, &program, &records, &messages, against_model);
    if differences.is_empty() && failed.is_none() {
        if against_model {
            eprintln!(
                "sequences: seed {seed} matches {} Metal line(s) and the model: {} commands, {} \
                 records",
                lines.len() - gaps.len(),
                program.command_count(),
                records.len()
            );
        }
        for gap in &gaps {
            eprintln!(
                "sequences: seed {seed} is a Metal-side gap on {gap}; on the runtime it completed \
                 without a refusal or a validation message ({} commands, {} records)",
                program.command_count(),
                records.len()
            );
        }
        return Ok(());
    }
    let mut report = format!(
        "seed {seed} ({} commands) differs:\n{}",
        program.command_count(),
        differences.join("\n")
    );
    for gap in &gaps {
        report.push_str(&format!("\na Metal-side gap on {gap}"));
    }
    if let Some((failure, details)) = failed {
        let against = if against_model {
            "against the model"
        } else {
            "on the runtime alone"
        };
        report.push_str(&format!("\n{against}: {failure:?}\n{}", details.join("\n")));
        let minimized = minimize(&program, ATTEMPTS, |candidate| {
            if Instant::now() > deadline {
                return false;
            }
            match on_runtime(seed, candidate) {
                Ok((records, sorted)) => {
                    failure_of(seed, candidate, &records, &sorted.failing, against_model)
                        .is_some_and(|found| found == failure)
                }
                Err(_) => false,
            }
        });
        report.push_str(&format!(
            "\nminimized after {} runs ({} commands{}):\n{}",
            minimized.attempts,
            minimized.program.command_count(),
            if Instant::now() > deadline {
                ", stopped at the batch's budget"
            } else {
                ""
            },
            minimized.program
        ));
    }
    report.push_str(&format!("\nthe sequence of seed {seed}:\n{program}"));
    Err(report)
}

fn failure_of(
    seed: u64,
    program: &Program,
    records: &[(String, Value)],
    messages: &[DebugMessage],
    against_model: bool,
) -> Option<Failure> {
    failure(seed, program, records, messages, against_model).map(|(failure, _)| failure)
}

#[test]
fn sequences_0_match_metal() {
    batch(0);
}

#[test]
fn sequences_1_match_metal() {
    batch(1);
}

#[test]
fn sequences_2_match_metal() {
    batch(2);
}

#[test]
fn sequences_3_match_metal() {
    batch(3);
}

#[test]
fn sequences_4_match_metal() {
    batch(4);
}

#[test]
fn sequences_5_match_metal() {
    batch(5);
}

#[test]
fn sequences_6_match_metal() {
    batch(6);
}

#[test]
fn sequences_7_match_metal() {
    batch(7);
}

/// Every seed is in a batch.
#[test]
fn the_batches_cover_every_seed() {
    assert_eq!(8 * BATCH, SEEDS);
}

/// Two untracked buffers, `seq_copy` writing word 256 of buffer 0 in one serial encoder and
/// reading buffer 0 at `read` in the next, into buffer 1: nothing orders the two (Metal leaves
/// untracked memory between encoders to the program). The layer's ranges, the descriptors the
/// runtime binds, meet wherever `read` is: each storage slot binds its buffer from the start to
/// the padded end and reads at its residual (`bind::planned_range`). The second writes another
/// buffer, so a read after write is the only hazard the two can make.
fn two_encoders(read: u64) -> Program {
    let copy = |a: u64, (buffer, b): (usize, u64)| Command::Dispatch {
        kernel: Kernel::Copy,
        a: At {
            buffer: 0,
            offset: a,
        },
        b: At { buffer, offset: b },
        indirect: None,
    };
    let serial = |command| {
        Step::Encoder(Encoder {
            kind: Kind::Compute { concurrent: false },
            waits: Vec::new(),
            updates: Vec::new(),
            commands: vec![command],
        })
    };
    let untracked = || BufferSpec {
        role: Role::Data,
        private: false,
        tracked: false,
        place: Place::Own,
        initial: vec![7; BUFFER_BYTES as usize],
    };
    Program {
        heaps: Vec::new(),
        buffers: vec![untracked(), untracked()],
        textures: Vec::new(),
        views: Vec::new(),
        queues: 1,
        events: 0,
        fences: 0,
        command_buffers: vec![CommandBufferSpec {
            queue: 0,
            steps: vec![serial(copy(0, (0, 256))), serial(copy(read, (1, 400)))],
        }],
    }
}

/// The lane's reading of the layer's hazards, on a device whose layer checks synchronization
/// whatever the lane's switch says: a read of the word the earlier encoder wrote is a hazard
/// between commands that share a byte, which fails; a read of the next word, inside the
/// descriptors' ranges the runtime binds (`hazards`), is one the layer's range alone makes, which
/// is explained as untracked memory and fails nothing. Both hazards are a kernel's read after a
/// kernel's write: where the layer sees no kernel's write
/// (`runtime_lane::sync_coverage::layer_sees_kernel_writes`, asked first), neither run may report
/// anything at all, and the line `LAYER-BLIND-KERNEL-WRITES` names what was not checked.
#[test]
fn a_hazard_fails_when_its_commands_share_a_byte_and_else_is_explained() {
    const TEST: &str = "a_hazard_fails_when_its_commands_share_a_byte_and_else_is_explained";
    let sees = layer_sees_kernel_writes();
    let run = |read| {
        let program = two_encoders(read);
        let (_, sorted) = on_device(0, &program, RuntimeSeq::open_synchronized().unwrap()).unwrap();
        for (hazard, verdict) in &sorted.explained {
            eprintln!("SEQUENCE-SYNC control read {read}: {hazard}: {verdict}");
        }
        for message in &sorted.failing {
            eprintln!(
                "SEQUENCE-SYNC control read {read}: failing {}: {}",
                message.id_name, message.text
            );
        }
        sorted
    };
    let shared = run(256);
    assert!(
        shared.explained.is_empty(),
        "explained: {:?}",
        shared.explained
    );
    if !sees {
        let apart = run(260);
        assert!(
            apart.explained.is_empty(),
            "explained: {:?}",
            apart.explained
        );
        let failing: Vec<DebugMessage> =
            (shared.failing.into_iter()).chain(apart.failing).collect();
        return blind_to_kernel_writes(
            TEST,
            "the failing read after write of commands sharing a byte, nor the explained one of \
             commands sharing a descriptor's range alone",
            &failing,
        );
    }
    assert!(
        shared
            .failing
            .iter()
            .any(|message| message.id_name == "SYNC-HAZARD-READ-AFTER-WRITE"
                && message.text.contains("share bytes")),
        "no hazard between the commands sharing a byte failed: {:#?}",
        shared.failing
    );
    let apart = run(260);
    assert!(apart.failing.is_empty(), "failing: {:#?}", apart.failing);
    assert!(
        apart.explained.iter().any(|(hazard, verdict)| hazard.earlier_writes
            && !hazard.later_writes
            && matches!(verdict, Verdict::Explained { metal, .. } if metal.contains(&MetalOrder::Untracked))),
        "no read after write explained as untracked memory: {:?}",
        apart.explained
    );
}

/// Metal's side: the recorder and its child.
#[cfg(all(m2v_gpu, target_os = "macos"))]
mod metal_recording {
    use super::super::super::api::{Observation, AT, RECORD};
    use super::super::super::record::{read_lines, stale, write_line, Line, Refusal, Value};
    use super::super::super::records_path;
    use super::super::generate::generate;
    use super::super::metal::MetalSeq;
    use super::super::model::{byte_differences, records as model_records};
    use super::super::program::{
        case_name, Command as SeqCommand, Kind, Place, Program, Step, BUFFER_BYTES,
    };
    use super::super::run::{run, HostPages};
    use super::super::{fingerprint, METAL_GAP, NAME, SEEDS};
    use std::os::unix::process::ExitStatusExt;
    use std::path::PathBuf;
    use std::process::{Command, Stdio};
    use std::time::{Duration, Instant, SystemTime};

    /// The variable that names the child's seed.
    const SEED_VAR: &str = "M2V_SEQUENCE_SEED";

    /// How long the recorder waits for macOS to write a crashed child's report.
    const REPORT_WAIT: Duration = Duration::from_secs(20);

    /// Record Metal's answers to every seed into the committed records file, as this Mac's
    /// environment's line, unless its line is current. A seed whose bytes differ from the model's
    /// is printed (`SEQUENCE-MODEL-DIFFERENCE`), and the line is written all the same: Metal's
    /// answer is the record, and `tests::committed_records_agree_with_the_model` fails on it. A
    /// seed whose child died of a signal (Metal's driver crashed running it) is recorded as the
    /// seed's Metal-side gap ([`METAL_GAP`]) with the crash's signature, and printed
    /// (`SEQUENCE-METAL-GAP`); its partial records are dropped.
    #[test]
    #[ignore = "writes the committed sequence records; a recording run on a Mac takes it"]
    fn record_metal() {
        let environment = crate::metal::identity().unwrap().environment_id();
        let fingerprint = fingerprint(&environment).unwrap();
        let lines = read_lines(&records_path(NAME)).unwrap();
        let current = lines.iter().any(|line| {
            line.fingerprint.metal_environment == environment
                && stale(&line.fingerprint, &fingerprint).is_empty()
        });
        if current {
            eprintln!("sequences: {NAME} on {environment} is current");
            return;
        }
        let mut records = Vec::new();
        let mut differ = 0;
        let mut gaps = Vec::new();
        for seed in 0..SEEDS {
            let ran = match child(seed).unwrap() {
                Child::Ran(ran) => ran,
                Child::Crashed(detail) => {
                    eprintln!("SEQUENCE-METAL-GAP seed {seed}: {detail}");
                    gaps.push(seed);
                    let name = format!("{}/{METAL_GAP}", case_name(seed));
                    let gap = Value::Refused {
                        refusal: Refusal::Gap,
                        detail,
                    };
                    records.push((name, gap));
                    continue;
                }
            };
            let expected = model_records(case_name(seed), &generate(seed));
            for difference in byte_differences(&ran, &expected) {
                eprintln!("SEQUENCE-MODEL-DIFFERENCE seed {seed}: {difference}");
                differ += 1;
            }
            records.extend(ran);
        }
        eprintln!(
            "sequences: recorded {SEEDS} seeds on {environment}, {} records, {differ} differences \
             from the model, Metal-side gaps {gaps:?}",
            records.len()
        );
        let line = Line {
            fingerprint,
            records,
        };
        write_line(&records_path(NAME), line).unwrap();
    }

    /// What a seed's child did: ran, with the records it printed, or died of a signal, with the
    /// crash's signature.
    enum Child {
        Ran(Vec<(String, Value)>),
        Crashed(String),
    }

    /// Run one seed in a child process under API validation. A child that API validation stopped
    /// is the generator's misuse, an error; a child that died of a signal crashed in Metal.
    fn child(seed: u64) -> Result<Child, String> {
        let started = SystemTime::now();
        let spawned = Command::new(std::env::current_exe().map_err(|e| e.to_string())?)
            .args([
                "--exact",
                "parity::sequences::lane::metal_recording::metal_child",
                "--ignored",
                "--nocapture",
                "--test-threads=1",
            ])
            .env(SEED_VAR, seed.to_string())
            .env("MTL_DEBUG_LAYER", "1")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|error| format!("seed {seed}: spawn: {error}"))?;
        let pid = spawned.id();
        let output = spawned
            .wait_with_output()
            .map_err(|error| format!("seed {seed}: wait: {error}"))?;
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        if let Some(signal) = output.status.signal() {
            let at = stdout
                .lines()
                .rev()
                .find_map(|line| line.strip_prefix(AT))
                .and_then(|named| named.split_once('/'))
                .map(|(_, op)| op.to_owned());
            return Ok(Child::Crashed(crash_detail(
                &generate(seed),
                signal,
                at.as_deref(),
                crash_report(pid, started),
            )));
        }
        let passed = output.status.success() && stdout.contains("test result: ok. 1 passed");
        if !passed {
            return Err(format!(
                "seed {seed}: the child failed ({}):\n{stdout}\n{stderr}",
                output.status
            ));
        }
        stdout
            .lines()
            .filter_map(|line| line.strip_prefix(RECORD))
            .map(|json| {
                serde_json::from_str(json).map_err(|error| format!("seed {seed}: {error}: {json}"))
            })
            .collect::<Result<_, _>>()
            .map(Child::Ran)
    }

    /// A crashed seed's gap detail: the signal, the operation it died in and its command, and,
    /// from macOS's crash report, the Metal call on the crashed thread and the frame that
    /// faulted.
    fn crash_detail(
        program: &Program,
        signal: i32,
        at: Option<&str>,
        report: Option<(String, String)>,
    ) -> String {
        let signal = match signal {
            4 => "SIGILL".to_owned(),
            5 => "SIGTRAP".to_owned(),
            6 => "SIGABRT".to_owned(),
            10 => "SIGBUS".to_owned(),
            11 => "SIGSEGV".to_owned(),
            other => format!("signal {other}"),
        };
        let at = match at {
            Some(op) => match command_at(program, op) {
                Some(command) => format!(" at {op} ({command})"),
                None => format!(" at {op}"),
            },
            None => String::new(),
        };
        let crash = match report {
            Some((call, frame)) => format!("; {call} faulted in {frame}"),
            None => "; no crash report".to_owned(),
        };
        format!("Metal's driver crashed: the recording child died of {signal}{at}{crash}")
    }

    /// The command a run's operation `op` (`cb<b> s<s> c<c>`) encodes, with its encoder and, for
    /// an indirect dispatch, the arguments buffer it reads.
    fn command_at(program: &Program, op: &str) -> Option<String> {
        let mut parts = op.split(' ');
        let mut index =
            |prefix: &str| -> Option<usize> { parts.next()?.strip_prefix(prefix)?.parse().ok() };
        let (cb, step, at) = (index("cb")?, index("s")?, index("c")?);
        let Step::Encoder(encoder) = program.command_buffers.get(cb)?.steps.get(step)? else {
            return None;
        };
        let command = encoder.commands.get(at)?;
        let kind = match encoder.kind {
            Kind::Blit => "a blit encoder",
            Kind::Compute { concurrent: true } => "a concurrent compute encoder",
            Kind::Compute { concurrent: false } => "a serial compute encoder",
            Kind::Render { .. } => "a render encoder",
        };
        let arguments = match command {
            SeqCommand::Dispatch {
                indirect: Some(arguments),
                ..
            } => {
                let buffer = &program.buffers[arguments.buffer];
                let place = match buffer.place {
                    Place::Own => "its own".to_owned(),
                    Place::NoCopy => "over a host page without a copy".to_owned(),
                    Place::Heap { heap, offset } => format!("in heap{heap} at {offset}"),
                };
                format!(
                    "; the arguments buffer{}: {BUFFER_BYTES} bytes, {}, {}, {place}, read at offset \
                     {}",
                    arguments.buffer,
                    if buffer.private { "private" } else { "shared" },
                    if buffer.tracked { "tracked" } else { "untracked" },
                    arguments.offset
                )
            }
            _ => String::new(),
        };
        Some(format!("{command} in {kind}{arguments}"))
    }

    /// The Metal call on a crashed child's crashed thread and its faulting frame, from the report
    /// macOS writes for process `pid` (`~/Library/Logs/DiagnosticReports/*.ips`, written after the
    /// crash): `None` when no report for it appears in [`REPORT_WAIT`].
    fn crash_report(pid: u32, started: SystemTime) -> Option<(String, String)> {
        let directory =
            PathBuf::from(std::env::var_os("HOME")?).join("Library/Logs/DiagnosticReports");
        let deadline = Instant::now() + REPORT_WAIT;
        loop {
            for entry in std::fs::read_dir(&directory).ok()?.filter_map(Result::ok) {
                let path = entry.path();
                let fresh = entry
                    .metadata()
                    .and_then(|metadata| metadata.modified())
                    .is_ok_and(|modified| modified >= started);
                if !fresh || path.extension().is_none_or(|extension| extension != "ips") {
                    continue;
                }
                if let Some(found) = std::fs::read_to_string(&path)
                    .ok()
                    .and_then(|text| signature(&text, pid))
                {
                    return Some(found);
                }
            }
            if Instant::now() > deadline {
                return None;
            }
            std::thread::sleep(Duration::from_millis(250));
        }
    }

    /// From a crash report's text, when it is process `pid`'s: the first Objective-C method on the
    /// crashed thread (the Metal call) and the thread's top frame (image and function, its
    /// template arguments and parameters left out).
    fn signature(text: &str, pid: u32) -> Option<(String, String)> {
        let (_, body) = text.split_once('\n')?;
        let report: serde_json::Value = serde_json::from_str(body).ok()?;
        if report.get("pid")?.as_u64()? != u64::from(pid) {
            return None;
        }
        let images = report.get("usedImages")?.as_array()?;
        let thread = report
            .get("threads")?
            .get(report.get("faultingThread")?.as_u64()? as usize)?;
        let frames = thread.get("frames")?.as_array()?;
        fn symbol(frame: &serde_json::Value) -> Option<&str> {
            frame.get("symbol")?.as_str()
        }
        let top = frames.first()?;
        let image = top
            .get("imageIndex")
            .and_then(|index| images.get(index.as_u64()? as usize))
            .and_then(|image| image.get("name"))
            .and_then(|name| name.as_str())
            .unwrap_or("?");
        let frame = format!("{image} {}", short(symbol(top).unwrap_or("?")));
        let call = frames
            .iter()
            .filter_map(symbol)
            .find(|symbol| symbol.starts_with("-[") || symbol.starts_with("+["))
            .unwrap_or("an unnamed call")
            .to_owned();
        Some((call, frame))
    }

    /// A C++ symbol without its template arguments and parameters.
    fn short(symbol: &str) -> String {
        let mut out = String::new();
        let mut depth = 0usize;
        for c in symbol.chars() {
            match c {
                '<' => depth += 1,
                '>' => depth = depth.saturating_sub(1),
                '(' if depth == 0 => break,
                _ if depth == 0 => out.push(c),
                _ => {}
            }
        }
        out
    }

    /// One seed on Metal, its records printed as they are made.
    #[test]
    #[ignore = "a child process of record_metal"]
    fn metal_child() {
        let seed: u64 = std::env::var(SEED_VAR)
            .expect("the seed to run")
            .parse()
            .expect("a seed");
        let program = generate(seed);
        // The child exits once it recorded: the pages stay until then.
        let pages = HostPages::of(&program);
        let api = MetalSeq::open().unwrap();
        let mut observation = Observation::new(case_name(seed), true);
        run(&api, &program, &pages, &mut observation).unwrap();
        observation.finish(NAME);
        drop(api);
        drop(pages);
    }
}
