//! The view check (`view_check.rs`) on a device: it starts as the device is made, settles with its
//! medians, and a later device of the same driver over the same shader cache reads its record as it
//! is made, before its first pipeline.
//!
//! - **Settles.** On lavapipe (`m2v_cpu_icd`), which writes timestamps, it is measured: a verdict,
//!   the per-lane ratio, and the report naming every arm's median. On a GPU it is measured, or
//!   unchecked with why; the report is printed either way.
//! - **Taken.** The compile device's profile carries exactly the answer's facts.
//! - **Kept.** Where it was measured, the next device over the same cache, made after the first
//!   went, answers the same at once, from the record, running no check.
//! - **Ended by a loss.** A recovery while it runs ends it on the lost generation, which it lets
//!   go of, and the new generation runs a check of its own.
//! - **Behind no batch.** Its timed dispatch's submission (`Device::submit_internal_if_idle`) is
//!   not made while a batch is in flight, and is once none is.
//! - **Its own work alone.** That submission carries none of the program's work: a new texture's
//!   first move queued for the staging queue stays queued for the program's next submission.
//! - **Bounded.** A device whose work never pauses keeps the check standing by until its
//!   stand-bys reach their most, where it ends unchecked and makes no further attempt; a quiet
//!   device over the same cache, which kept nothing of it, runs the check and answers.

use super::blit_holds::{options, Scratch, FILLED};
use crate::device::holds::Holder;
use crate::{
    CommandBufferStatus, CpuCacheMode, Device, HazardTrackingMode, Region, ResourceOptions,
    StorageMode, TextureDescriptor, ViewVerdict,
};
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use std::time::{Duration, Instant};

/// How long a test waits for the check: lavapipe runs its kernels on the CPU.
const WAIT: Duration = Duration::from_secs(120);

#[test]
fn the_check_settles_with_its_medians_and_a_later_device_reads_its_record() {
    let scratch = Scratch::new("view-check");
    let mut options = options(&scratch.0);
    options.view_facts = None;
    let device = Device::system_default_with(&options).unwrap();
    assert!(device.settle_view_check(WAIT), "the check answered");
    let costs = device.view_costs();
    let report = device.view_check_report();
    println!("view check on {}: {report}", device.name());
    assert_ne!(costs.verdict, ViewVerdict::Pending);
    if cfg!(m2v_cpu_icd) {
        assert_eq!(costs.verdict, ViewVerdict::Measured, "{report}");
    }
    let workarounds = device.compile_device().profile().workarounds().to_vec();
    for fact in [
        metal2vulkan::device::DriverWorkaround::UniformBufferPerLaneReadsSerialize,
        metal2vulkan::device::DriverWorkaround::ExternalUniformReadsSkipACache,
    ] {
        assert_eq!(
            workarounds.contains(&fact),
            costs.facts.contains(&fact),
            "{fact:?}: {report}"
        );
    }
    if costs.verdict == ViewVerdict::Unchecked {
        assert!(costs.facts.is_empty(), "{report}");
        return;
    }
    assert!(costs.per_lane.is_some(), "{report}");
    for named in [
        "Uniform/PerLane/device/1KiB",
        "Storage/PerLane/device/4KiB",
        "ran",
    ] {
        assert!(report.contains(named), "{named}: {report}");
    }
    drop(device);
    let again = Device::system_default_with(&options).unwrap();
    assert_eq!(again.view_costs(), costs, "the record's answer, at once");
    let report = again.view_check_report();
    assert!(
        report.contains("read from the shader cache's record"),
        "{report}"
    );
}

/// How long the program's work goes on in [`the_check_stands_by_while_the_device_s_work_goes_on`],
/// and how long into it the check's first kernel, begun as the device was made before any work,
/// has surely begun. A kernel's making counts in `compute_pipeline_create_nanos` only as it ends,
/// which on lavapipe beside the lane's tests came past 300 ms into the work (jobs m3-max-39472,
/// 39475, 39513): the test reads the kernels begun (`ViewCheck::kernels_begun`).
const STREAM: Duration = Duration::from_millis(900);
const SETTLED: Duration = Duration::from_millis(300);

/// The bytes of the buffer the program's work fills ([`Stream`]), and the fills of the command
/// buffer by which it measures the device's rate first.
const STREAM_BYTES: u64 = 16 << 20;
const STREAM_FILLS: usize = 32;

/// How long each of the stream's command buffers lasts at the rate its first one measured, and
/// the most fills one takes. The rate moves with the host's load: on lavapipe beside the lane's
/// tests a first command buffer measured 3.6 times slower than the later ones ran (job
/// m3-max-40006), so the queue holds less than [`STREAM_DEPTH`] times this, and may drain.
const STREAM_EACH: Duration = Duration::from_millis(225);
const STREAM_MOST_FILLS: usize = 8192;

/// The command buffers the program's work keeps committed or in flight ([`Stream`]): eight of
/// [`STREAM_EACH`], 1.8 s of queued work at the measured rate. On lavapipe the queue's work runs on
/// the CPUs the lane's tests share, at a rate that moves with their load: four command buffers of a
/// fixed 512 MiB of fills each queued drained there beside other tests (kernels begun while the
/// work went on, job m3-max-39758), as four of 64 MiB had (the device's queues unused for 26 to
/// 57 ms with nothing in flight, job m3-max-39574). Where the queue drains anyway, the work has
/// paused, and the check may go on: the stream counts each pause it finds ([`Stream::next`]).
const STREAM_DEPTH: usize = 8;

/// The program's work as a stream: command buffers of `queue`'s, each filling a buffer of
/// [`STREAM_BYTES`] whole `fills` times, [`STREAM_DEPTH`] of them committed or in flight.
struct Stream {
    queue: crate::CommandQueue,
    target: crate::Buffer,
    /// The fills of each command buffer: as many as last [`STREAM_EACH`] at the rate a first
    /// command buffer of [`STREAM_FILLS`] measured, committed with [`STREAM_DEPTH`] behind it so
    /// that the queue never drains as it is measured, at least that many and at most
    /// [`STREAM_MOST_FILLS`].
    fills: usize,
    queued: std::collections::VecDeque<(crate::CommandBuffer, Instant)>,
    /// The longest a command buffer took from its commit to its completion.
    longest: Duration,
}

impl Stream {
    fn new(device: &Device) -> Stream {
        let queue = device.new_command_queue().unwrap();
        let target = device
            .new_buffer(STREAM_BYTES, ResourceOptions::private())
            .unwrap();
        let mut stream = Stream {
            queue,
            target,
            fills: STREAM_FILLS,
            queued: std::collections::VecDeque::new(),
            longest: Duration::ZERO,
        };
        for _ in 0..STREAM_DEPTH {
            stream.next();
        }
        let one = stream.end_oldest();
        let scaled = STREAM_EACH.as_secs_f64() / one.as_secs_f64().max(1e-6);
        stream.fills =
            ((STREAM_FILLS as f64 * scaled) as usize).clamp(STREAM_FILLS, STREAM_MOST_FILLS);
        stream
    }

    /// Commit one more command buffer, first waiting for the oldest where [`STREAM_DEPTH`] are
    /// queued; whether the work had paused: none of the stream's command buffers was left
    /// uncompleted as this one was committed, the queue drained before it. A pause is seen as the
    /// host publishes the last one's status, after the queue's completed value the check reads, so
    /// one that ends within that publication is not seen.
    fn next(&mut self) -> bool {
        if self.queued.len() >= STREAM_DEPTH {
            self.wait_oldest();
        }
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        for fill in 0..self.fills {
            blit.fill_buffer(&self.target, 0..STREAM_BYTES, fill as u8)
                .unwrap();
        }
        blit.end_encoding().unwrap();
        let paused = (self.queued.back())
            .is_none_or(|(newest, _)| newest.status() == CommandBufferStatus::Completed);
        command_buffer.commit().unwrap();
        self.queued.push_back((command_buffer, Instant::now()));
        paused
    }

    /// Wait for the oldest command buffer queued: its commit to its completion.
    fn end_oldest(&mut self) -> Duration {
        let (command_buffer, committed) = self.queued.pop_front().expect("one queued");
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        committed.elapsed()
    }

    fn wait_oldest(&mut self) {
        let took = self.end_oldest();
        self.longest = self.longest.max(took);
    }

    /// Stop: wait for every command buffer queued; the longest commit to completion.
    fn end(mut self) -> Duration {
        while !self.queued.is_empty() {
            self.wait_oldest();
        }
        self.longest
    }
}

/// The program's work never waits on the check: while a queue keeps a command buffer committed or
/// in flight, the check stands by, beginning no kernel's making and submitting no dispatch, its
/// answer pending; once the work stops it answers, its stand-bys counted. Each command buffer of the
/// work completes in its own time (the longest is printed); none waits behind the check's work.
/// Where the stream found its work paused ([`Stream::next`]), the check may have gone on in the
/// pause: the test prints the pauses and holds the check to its answer and its stand-bys alone;
/// where it found none, to all of the above.
#[test]
fn the_check_stands_by_while_the_device_s_work_goes_on() {
    let scratch = Scratch::new("view-check-stand-by");
    let mut options = options(&scratch.0);
    options.view_facts = None;
    let device = Device::system_default_with(&options).unwrap();
    let mut stream = Stream::new(&device);
    let began = std::time::Instant::now();
    let check = device.view_check();
    let mut settled = None;
    let mut paused = 0;
    while began.elapsed() < STREAM {
        if stream.next() {
            paused += 1;
        }
        if settled.is_none() && began.elapsed() >= SETTLED {
            settled = Some((device.work_counts(), check.kernels_begun()));
        }
    }
    let (during, begun) = (device.work_counts(), check.kernels_begun());
    let pending = device.view_costs().verdict;
    let fills = stream.fills;
    let longest = stream.end();
    let (settled, settled_begun) = settled.expect("the work went on past SETTLED");
    println!(
        "the work's command buffers of {fills} fills, the longest commit to completion: {:.2} \
         ms, paused {paused} times; kernels begun before SETTLED {settled_begun} and by the \
         work's end {begun}, compute pipeline time by then {:.2} ms and by the work's end {:.2} \
         ms",
        longest.as_secs_f64() * 1e3,
        settled.compute_pipeline_create_nanos as f64 / 1e6,
        during.compute_pipeline_create_nanos as f64 / 1e6
    );
    if paused == 0 {
        assert_eq!(
            pending,
            ViewVerdict::Pending,
            "no answer while the work went on"
        );
        assert_eq!(
            begun, settled_begun,
            "no kernel begun while the work went on"
        );
        assert_eq!(
            during.internal_submissions, settled.internal_submissions,
            "no dispatch submitted while the work went on"
        );
    }
    assert!(device.settle_view_check(WAIT), "the check answered");
    let costs = device.view_costs();
    let checks = device.work_counts().device_checks;
    let report = device.view_check_report();
    if cfg!(m2v_cpu_icd) {
        assert_eq!(costs.verdict, ViewVerdict::Measured, "{report}");
    }
    assert!(checks.view_check_stand_bys >= 1, "{checks:?}");
    let stood = Duration::from_nanos(checks.view_check_stand_by_nanos);
    if paused == 0 {
        assert!(stood >= STREAM - SETTLED, "stood by {stood:?}: {checks:?}");
    }
}

/// How long the check takes, let go at its hold after its generation was lost, to let go of the
/// lost `VkDevice`.
const ENDS: Duration = Duration::from_secs(10);

/// A recovery under the check: held at its first round's step (`device/holds.rs`) while the device
/// recovers (`Device::recover_for_test`), the lost generation's check meets the loss there and lets
/// go of the lost `VkDevice`, which goes once nothing else holds it (`WorkCounts::vk_devices_live`
/// back to 1), and the new generation runs a check of its own, which answers. On a device whose
/// check answers at once, unchecked (no timestamps the host reads), it never holds, and the test
/// says so and ends. The holds are read through the generation serving, so the test's handle pins
/// none (`Device::shared_arc`).
#[test]
fn a_recovery_under_the_check_lets_the_lost_device_go_and_the_new_one_checks() {
    let scratch = Scratch::new("view-check-recovery");
    let mut options = options(&scratch.0);
    options.view_facts = None;
    let device = Device::system_default_with(&options).unwrap();
    let holds = std::sync::Arc::clone(&device.current().shared().holds);
    let held = holds.held(Holder::ViewCheck);
    held.hold();
    let began = Instant::now();
    while held.waiting() == 0 {
        if device.view_costs().verdict != ViewVerdict::Pending {
            held.release();
            let report = device.view_check_report();
            assert_eq!(
                device.view_costs().verdict,
                ViewVerdict::Unchecked,
                "{report}"
            );
            println!("view check on {}: never held, {report}", device.name());
            return;
        }
        assert!(began.elapsed() < WAIT, "the check never reached its hold");
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(device.work_counts().vk_devices_live, 1);
    device.recover_for_test().unwrap();
    assert_eq!(
        device.work_counts().vk_devices_live,
        2,
        "the check holds the lost VkDevice"
    );
    held.release();
    let released = Instant::now();
    while device.work_counts().vk_devices_live > 1 {
        assert!(
            released.elapsed() < ENDS,
            "the check let go of the lost VkDevice"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(
        device.settle_view_check(WAIT),
        "the new generation's check answered"
    );
    let report = device.view_check_report();
    println!("view check on {} after a recovery: {report}", device.name());
    assert_ne!(device.view_costs().verdict, ViewVerdict::Pending);
}

/// The most fills of [`FILLED`] bytes one busy command buffer of
/// [`an_idle_internal_submission_waits_for_no_batch_in_flight`] makes, the work it grows to until
/// it is still in flight as the submission is asked for.
const MOST_FILLS: u32 = 1 << 10;

/// The submission the view check times its dispatches by (`Device::submit_internal_if_idle`) is
/// not made while a batch is in flight: a command buffer of fills, committed and submitted, is in
/// flight as it is asked for, and it records nothing (no internal submission counted); once that
/// command buffer completed, it is made. Where the fills ended before the ask, the test commits
/// four times as many and asks again.
#[test]
fn an_idle_internal_submission_waits_for_no_batch_in_flight() {
    let scratch = Scratch::new("view-check-idle");
    let device = Device::system_default_with(&options(&scratch.0)).unwrap();
    let queue = device.new_command_queue().unwrap();
    let target = device
        .new_buffer(FILLED, ResourceOptions::private())
        .unwrap();
    let current = device.current();
    let mut fills = 16;
    loop {
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        for value in 0..fills {
            blit.fill_buffer(&target, 0..FILLED, value as u8).unwrap();
        }
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_scheduled().unwrap();
        let before = device.work_counts().internal_submissions;
        let made = idle_submission(&current);
        let after = device.work_counts().internal_submissions;
        let ended = command_buffer.status() == CommandBufferStatus::Completed;
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        if !made {
            assert_eq!(after, before, "a submission not made is not counted");
            println!("{fills} fills of {FILLED} bytes in flight: no submission made");
            break;
        }
        assert!(ended, "made while {fills} fills were in flight");
        assert!(fills < MOST_FILLS, "{fills} fills ended before the ask");
        fills *= 4;
    }
    let before = device.work_counts().internal_submissions;
    assert!(idle_submission(&current), "made with no batch in flight");
    assert_eq!(device.work_counts().internal_submissions, before + 1);
}

/// The submission the view check times its dispatches by (`Device::submit_internal_if_idle`)
/// carries none of the program's work: made while a new `Managed` texture's first move waits
/// queued for the device's staging queue (`WorkCounts::staging_moves_queued`), it is one internal
/// submission and submits nothing there (no staging submission or flush), waiting for nothing
/// there on the GPU (`staging_gpu_waits`); the move stays queued for the program's next
/// submission, and `get_bytes` reads the texture's zeros. A device that copies new textures
/// otherwise (no staging queue, or a start on the host) prints so.
#[test]
fn an_idle_internal_submission_carries_none_of_the_program_s_queued_work() {
    const SIDE: u64 = 64;
    let scratch = Scratch::new("view-check-own");
    let device = Device::system_default_with(&options(&scratch.0)).unwrap();
    let name = device.name().to_owned();
    let before = device.work_counts();
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::new(
        StorageMode::Managed,
        CpuCacheMode::DefaultCache,
        HazardTrackingMode::Default,
    );
    let texture = device.new_texture(&descriptor).unwrap();
    let made = device.work_counts();
    if made.staging_moves_queued == before.staging_moves_queued {
        println!("{name}: a new Managed texture queues no move for a staging queue");
        return;
    }
    assert!(
        idle_submission(&device.current()),
        "made with no batch in flight"
    );
    let after = device.work_counts();
    assert_eq!(
        (
            after.internal_submissions - made.internal_submissions,
            after.staging_submissions - made.staging_submissions,
            after.staging_flushes - made.staging_flushes,
            after.staging_gpu_waits - made.staging_gpu_waits,
        ),
        (1, 0, 0, 0),
        "{name}: internal, staging submissions, flushes, GPU waits for the staging queue"
    );
    let mut bytes = vec![0xee; (4 * SIDE * SIDE) as usize];
    let region = Region::new_2d(0, 0, SIDE, SIDE);
    texture
        .get_bytes(&mut bytes, 4 * SIDE, 0, region, 0, 0)
        .unwrap();
    assert!(
        bytes.iter().all(|&b| b == 0),
        "{name}: the new texture reads zero"
    );
}

/// An internal submission of no work of `device`'s, made only where no batch is in flight, inside
/// a section of the exit guard as the view check makes it: whether it was made.
fn idle_submission(device: &Device) -> bool {
    let _busy = crate::exit_guard::enter().expect("the process is not exiting");
    device.submit_internal_if_idle(|_, _| {}).unwrap()
}

/// The bound [`a_device_whose_work_never_pauses_ends_the_check_unchecked_at_its_bound`] sets on
/// the check's stand-bys.
const BOUND: Duration = Duration::from_millis(300);

/// How long the device stays quiet after the check ended, in which it makes no further attempt.
const AFTER: Duration = Duration::from_millis(300);

/// A device whose work never pauses (a queue keeps a command buffer committed or in flight) keeps
/// the check standing by, its answer pending, until its stand-bys reach their most ([`BOUND`],
/// set for the test; `view_check::STAND_BY_MOST` otherwise): it answers unchecked, saying the
/// work never paused, its stand-bys counted for at least that long, and once the work stopped it
/// makes no further attempt (no kernel made, no dispatch submitted, no stand-by, the verdict kept).
/// A device made after over the same cache, which kept nothing of the unchecked answer, stays
/// quiet, and no call waits for its answer (which would have it stand by for no one): it runs the
/// check and answers (measured on lavapipe).
#[test]
fn a_device_whose_work_never_pauses_ends_the_check_unchecked_at_its_bound() {
    let scratch = Scratch::new("view-check-bound");
    let mut options = options(&scratch.0);
    options.view_facts = None;
    let device = Device::system_default_with(&options).unwrap();
    device.view_check().stand_by_at_most(BOUND);
    let mut stream = Stream::new(&device);
    let began = Instant::now();
    while device.view_costs().verdict == ViewVerdict::Pending {
        assert!(began.elapsed() < WAIT, "the check never ended");
        stream.next();
    }
    stream.end();
    let report = device.view_check_report();
    println!(
        "view check on {} under work that never pauses, ended after {:.1} ms: {report}",
        device.name(),
        began.elapsed().as_secs_f64() * 1e3
    );
    assert_eq!(
        device.view_costs().verdict,
        ViewVerdict::Unchecked,
        "{report}"
    );
    assert!(report.contains("never paused"), "{report}");
    let ended = device.work_counts();
    let checks = ended.device_checks;
    let stood = Duration::from_nanos(checks.view_check_stand_by_nanos);
    assert!(stood >= BOUND, "stood by {stood:?}: {checks:?}");
    std::thread::sleep(AFTER);
    let later = device.work_counts();
    assert_eq!(
        (
            later.compute_pipeline_create_nanos,
            later.internal_submissions,
            later.device_checks.view_check_stand_bys,
        ),
        (
            ended.compute_pipeline_create_nanos,
            ended.internal_submissions,
            checks.view_check_stand_bys,
        ),
        "no attempt after the check ended"
    );
    assert_eq!(device.view_costs().verdict, ViewVerdict::Unchecked);
    drop(device);
    let quiet = Device::system_default_with(&options).unwrap();
    let made = Instant::now();
    while quiet.view_costs().verdict == ViewVerdict::Pending {
        assert!(
            made.elapsed() < WAIT,
            "the quiet device's check never answered"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
    let report = quiet.view_check_report();
    println!("view check on {} kept quiet: {report}", quiet.name());
    assert!(
        !report.contains("read from the shader cache's record"),
        "{report}"
    );
    assert!(!report.contains("never paused"), "{report}");
    if cfg!(m2v_cpu_icd) {
        assert_eq!(
            quiet.view_costs().verdict,
            ViewVerdict::Measured,
            "{report}"
        );
    }
}
