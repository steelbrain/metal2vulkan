//! A process that exits while the runtime still works, on the device the cfg selects, under the
//! validation layer.
//!
//! Each case runs in a child process ([`child`], this test binary again), which ends with `exit`
//! or at the end of `main` while the runtime's threads hold work, run the program's code, or tear
//! a device down. The parent checks that the child exited with status 0, not by a signal, and
//! that its exit handlers ended within [`EXIT_LIMIT`] of its last line, or of the end of the CPU
//! device's work its exit waits for where that is later, less the time the exiting thread waited
//! for a CPU meanwhile ([`exit_end`]): below the exit hook's own bound of 2 s, so a hook that
//! waited out its bound fails too. Where the runtime still ran its work past that bound, on Linux
//! (the GNU C library) the hook ends the process at once, the handlers registered before it
//! unrun, and the exit ends at the notice the runtime writes as it ends the process
//! ([`AT_ONCE`]), as the parent reads it: the kernel's teardown of the process after it is
//! printed, not counted. Each child reports the validation
//! layer's messages until then, which must be none. The parent prints each exit's time, with its
//! wall time, that wait, and how long the process took to end after its handlers. Where the hook
//! ran before the mark each child registers right after its first device
//! ([`after_the_later_devices`]), the exit's time ends at the mark: the handlers registered as
//! that device was made, its driver's and the layer's, run after it, and their time is printed,
//! not counted (on an RTX 5080 laptop beside other GPU work, 0.5 to 7 s). Each test that checks
//! that time runs alone in the suite (`.config/nextest.toml`): an exit's time holds the drivers'
//! calls the hook waits for, and, where the hook ran after the mark, the drivers' exit handlers,
//! which other processes' GPU work slows (NVIDIA's `vkDestroyDevice` took 54 ms alone and up to
//! 2 s beside other lanes on an RTX 3090).
//!
//! Metal's answers (an M4 with macOS 27.0.1, each case without and with
//! `MTL_DEBUG_LAYER=1`): every such exit was clean, in 15 to 45 ms, but one: a dropped command
//! buffer waiting for a plain `MTLEvent` held the exit back 5.03 s, which the runtime does not
//! copy. A dropped command buffer that waits for a shared event stays pending, and completes,
//! its handler running, once the event is signalled
//! ([`a_dropped_command_buffer_completes_once_its_event_is_signalled`]).
//!
//! On a CPU device the work submitted before an exit runs on the process's own threads, so no exit
//! handler or destructor runs beside it ([`an_exit_runs_no_handler_beside_a_cpu_device_s_work`]):
//! the exit waits for the work, and the handlers registered before the device run after it.
//! Measured on lavapipe, its queue thread compiling a dispatch's shader during the
//! destructors ended the process in the heap's checks, in 1 to 2 exits of 100 while a host step
//! waited.

use super::indirect_host_steps::Setup;
use super::{
    at_plan, bound_buffers, encode_case, layer_reported_nothing_at_all, pipeline_state,
    public_case, run_queue_waits, validated_device, validated_device_with, CaseResources, Messages,
    ScratchCache,
};
use crate::air_library::AirLibrary;
use crate::gpu::{DeviceType, TEST_DEVICE};
use metal2vulkan::library::{FunctionConstantValue, FunctionConstantValues};
use metal2vulkan::metal::{DataType, PixelFormat, TextureUsage};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    BlitOption, Buffer, CommandBuffer, CommandBufferStatus, CommandQueue, ComputePipelineState,
    Deallocator, Device, DeviceKind, EventRef, HostMemory, Origin, PipelineOptions,
    ResourceOptions, SharedEvent, SharedEventListener, Size, TextureDescriptor,
};
use std::collections::BTreeMap;
use std::io::Read;
use std::path::PathBuf;
use std::process::{Command, ExitStatus, Stdio};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicPtr, AtomicU64, Ordering};
use std::sync::{mpsc, Mutex, OnceLock, PoisonError};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// The longest an exit may take, from the child's last line to the end of its last exit handler
/// ([`exit_end`]; the runtime's notice where it ended the process at once and that handler did not
/// run, [`AT_ONCE`]; the mark before the first device's driver's and layer's handlers where the
/// hook ran before it, [`after_the_later_devices`]), or,
/// where the exit waits for a CPU device's work that ends later, from that work's end: the
/// work runs for as long as the host's load lets it ([`CPU_WORK_TIME`]), and the exit's own part
/// does not. Less the time the exiting thread waited for a CPU meanwhile, which is the host's
/// load. What follows the handlers, the libraries' destructors and the kernel's teardown of the
/// process, is not the exit's work: the parent prints it.
const EXIT_LIMIT: Duration = Duration::from_millis(1500);

/// The longest a child may run, or a command buffer of these tests take.
const WAIT: Duration = Duration::from_secs(60);

/// The environment variable that names the case to [`child`].
const CASE_VAR: &str = "M2V_LANE_EXIT_CASE";

/// The environment variable that names the shader cache directory to [`child`].
const CACHE_VAR: &str = "M2V_LANE_EXIT_CACHE_DIR";

/// The line a child reports on, right before it exits.
const REPORT: &str = "RUNTIME-EXIT";

/// The start of the line the runtime writes on standard error where it ends an exit at once from
/// its hook (Linux, the GNU C library: the driver still runs the runtime's work as the hook runs).
const AT_ONCE: &str = "metal2vulkan: the process's exit: ";

/// The bytes one fill of the in-flight case writes, and how many fills it encodes: enough work
/// that the device still runs it when a handler exits (4 GiB written on a GPU), and on a CPU
/// device, which writes it at memory speed, 64 MiB.
fn in_flight() -> (u64, usize) {
    match TEST_DEVICE {
        DeviceType::Gpu => (64 << 20, 64),
        DeviceType::Cpu => (16 << 20, 4),
    }
}

/// The bytes one fill of the CPU work case writes.
const CPU_WORK_FILL: u64 = 64 << 20;

/// How long the CPU work case's work runs, as one fill's time measured before it predicts: far
/// longer than the exit's own teardown (about 50 ms on lavapipe), so the hook finds it running,
/// and far shorter than the hook's bound of 2 s, within which the hook waits for it. A host's load stretches it: at a load of 30 to 45, 400 ms
/// of fills ran for 1.5 s.
const CPU_WORK_TIME: Duration = Duration::from_millis(200);

/// The fill byte of [`fill`].
const FILLED: u8 = 0x5a;

/// The marker the CPU work case fills after its work, which [`read_the_marker`] reads at exit: set
/// on a CPU device only. Registered before the device, the handler runs after the driver's own
/// exit-time destructors as well as the device's hook, and a GPU driver's may unmap what it mapped:
/// on the RTX 3090 (NVIDIA 615.71.09) the read of a marker it had mapped ended the process with
/// `SIGSEGV` in every exit (3 of 3, the fault at the mapping's first byte in
/// the handler). A CPU device's memory is the process's own, which no destructor unmaps, and only
/// a CPU device's exit waits for the work the marker follows.
static MARKER: AtomicPtr<u8> = AtomicPtr::new(std::ptr::null_mut());

/// A point the exit's own time may count from ([`exit_end`]): when, in milliseconds since the
/// epoch, and each of the process's threads' wait for a CPU by then, by thread id, where Linux
/// counts it (`super::run_queue_waits`).
struct Since {
    ms: u128,
    waits: BTreeMap<u64, Duration>,
}

impl Since {
    /// Now: the instant read before the waits, so that a wait between the two stays in the exit's
    /// time.
    fn now() -> Since {
        let ms = now_ms();
        Since {
            ms,
            waits: run_queue_waits(),
        }
    }
}

/// The child's last line ([`report`]).
static REPORTED: Mutex<Option<Since>> = Mutex::new(None);

/// When the marker's fill was first seen complete ([`watch_the_marker`], else
/// [`read_the_marker`]): where the CPU work the exit waits for ended.
static WORK_ENDED: Mutex<Option<Since>> = Mutex::new(None);

/// When the CPU work the exit waits for ended ([`WORK_ENDED`]), in milliseconds since the epoch:
/// noted now where it was not yet.
fn work_ended() -> u128 {
    WORK_ENDED
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .get_or_insert_with(Since::now)
        .ms
}

/// Watch the marker at `at` from a thread of its own, which goes on through the exit, and note in
/// [`WORK_ENDED`] when its fill is first seen complete: where the work the exit waits for ended,
/// which [`exits_cleanly`] counts the exit's own time from.
fn watch_the_marker(at: NonNull<u8>) {
    let at = at.as_ptr() as usize;
    std::thread::spawn(move || {
        let at = at as *const u8;
        loop {
            // SAFETY: the marker's 16 shared bytes, mapped for the process's life (as
            // `read_the_marker` reads them); read volatile, as the device writes them meanwhile.
            let written = (0..16).all(|i| unsafe { at.add(i).read_volatile() } == FILLED);
            if written {
                work_ended();
                return;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    });
}

/// An exit handler the CPU work case registers before it makes its device, so that it runs after
/// the device's exit hook where the hook returns (not where it ended the process at once past its
/// bound, on Linux): print whether the marker's fill, submitted after the work, completed (always
/// `written=0` without a marker, on a GPU), when the work ended (`seen_ms`: when
/// [`watch_the_marker`] saw the fill complete, else now; 0 where it is not written), and whether
/// the runtime answers that the process is exiting (`process_exiting`, which the hook's run makes
/// true on every device).
extern "C" fn read_the_marker() {
    let at = MARKER.load(Ordering::SeqCst);
    // SAFETY: the marker's 16 shared bytes of a CPU device, mapped for the process's life: the
    // case keeps the buffer, the exit drops nothing, and the driver's memory is the process's.
    let written = !at.is_null()
        && unsafe { std::slice::from_raw_parts(at, 16) }
            .iter()
            .all(|&b| b == FILLED);
    let exiting = u8::from(metal2vulkan_runtime::process_exiting());
    // Where the watcher has not run since the fill completed, the work ended by now at the latest.
    let seen_ms = match written {
        true => work_ended(),
        false => 0,
    };
    let line = format!(
        "{REPORT}-MARKER written={} exiting={exiting} seen_ms={seen_ms}\n",
        u8::from(written)
    );
    // SAFETY: a write of the line's bytes to standard output.
    unsafe { libc::write(1, line.as_ptr().cast(), line.len()) };
}

/// What the CPU work case leaves for [`release_the_gate`]: the event its work waits for at 1,
/// which nothing else sets, and the marker's command buffer, committed after the work on its
/// queue.
struct Gate {
    event: SharedEvent,
    marking: CommandBuffer,
}

/// The CPU work case's [`Gate`], set once both command buffers are committed.
static GATE: OnceLock<Gate> = OnceLock::new();

/// The hook's registrations in the process (`metal2vulkan_runtime::exit_registrations`) right
/// after the CPU work case registered [`release_the_gate`]: one more by the exit means the hook
/// was registered again after the gate's handler, so it ran first.
static AT_THE_GATE: AtomicU64 = AtomicU64::new(0);

/// The last device of the case `instance_after_handler_exit`, which [`drop_the_held_device`]
/// drops.
static HELD: Mutex<Option<Device>> = Mutex::new(None);

/// An exit handler the case `instance_after_handler_exit` registers after its first device's hook
/// and before its last device, so that it runs after the exit handler the validation layer
/// registers at the last device's instance (which forgets every device the layer knows): note
/// whether the hook ran (`hook_ran`), then drop the last device, whose teardown, once the hook
/// closed the guard, makes no Vulkan call.
extern "C" fn drop_the_held_device() {
    let hook_ran = metal2vulkan_runtime::process_exiting();
    let at_ms = now_ms();
    drop(HELD.lock().unwrap_or_else(PoisonError::into_inner).take());
    let line = format!(
        "{REPORT}-HELD hook_ran={} at_ms={at_ms} dropped_ms={}\n",
        u8::from(hook_ran),
        now_ms()
    );
    // SAFETY: a write of the line's bytes to standard output.
    unsafe { libc::write(1, line.as_ptr().cast(), line.len()) };
}

/// An exit handler the cases `program_instance_exit` and `adopted_instance_exit` register right
/// after the program's own instance, so after the exit handler the validation layer registered
/// as it was created (which forgets every device the layer knows), and before the program reports
/// that instance or the runtime adopts its device: note whether the runtime's hook ran by then
/// (`hook_ran`), as it does where the report or the adoption registered it again, after both.
extern "C" fn after_the_program_instance() {
    let line = format!(
        "{REPORT}-PROGRAM-INSTANCE hook_ran={}\n",
        u8::from(metal2vulkan_runtime::process_exiting())
    );
    // SAFETY: a write of the line's bytes to standard output.
    unsafe { libc::write(1, line.as_ptr().cast(), line.len()) };
}

/// An exit handler every child registers right after its first device, so that it runs after
/// every handler registered since (the case's own, the later instances' and devices', the hook's
/// later registrations, [`drop_the_held_device`], the destructors the driver's compiler registered
/// in a making) and before those registered as the first device was made (its driver's and
/// layer's): print whether the hook ran by then (`hook_ran`), when it ran (`at_ms`), and the
/// exiting thread's wait for a CPU until then (`queued_us`, as [`exit_end`]'s), where the exit's
/// own part ends once the hook ran ([`exits_cleanly`]).
extern "C" fn after_the_later_devices() {
    let hook_ran = metal2vulkan_runtime::process_exiting();
    let queued = exit_thread_queued().unwrap_or_default();
    let line = format!(
        "{REPORT}-MARK hook_ran={} at_ms={} queued_us={}\n",
        u8::from(hook_ran),
        now_ms(),
        queued.as_micros()
    );
    // SAFETY: a write of the line's bytes to standard output.
    unsafe { libc::write(1, line.as_ptr().cast(), line.len()) };
}

/// The longest [`release_the_gate`] waits for the gate's work to be submitted once it opened the
/// gate: a submission takes milliseconds on a loaded host, so a wait past it is a failure the
/// parent reports (`scheduled=0`), not an exit held until the parent's [`WAIT`] kills the child.
const SUBMISSION_WAIT: Duration = Duration::from_secs(10);

/// An exit handler the CPU work case registers last, right before it exits, so that it runs first
/// in the exit, before the device's hook: note whether the marker's command buffer was still held
/// (`held`: `Committed`, its work behind the closed gate, so neither had run when the exit began),
/// and whether the hook had run (`hook_ran`, which would leave the gate's work pending), with the
/// hook's registrations since this handler's own (`since`: any one runs the hook first) and the
/// process's by kind (`registrations`); then open
/// the gate, wait at most [`SUBMISSION_WAIT`] until the marker's command buffer is submitted, the
/// work's before it (`scheduled`), and note whether it still runs as the hook comes next
/// (`running`). Where the hook ran, nothing is submitted after it (it closed the device's guard),
/// so the handler does not wait.
extern "C" fn release_the_gate() {
    let Some(gate) = GATE.get() else {
        return;
    };
    let held = gate.marking.status() == CommandBufferStatus::Committed;
    let hook_ran = metal2vulkan_runtime::process_exiting();
    let now = metal2vulkan_runtime::exit_registrations();
    let since = now.total.saturating_sub(AT_THE_GATE.load(Ordering::SeqCst));
    gate.event.set_signaled_value(1);
    let deadline = Instant::now() + SUBMISSION_WAIT;
    while !hook_ran
        && gate.marking.status() == CommandBufferStatus::Committed
        && Instant::now() < deadline
    {
        std::thread::sleep(Duration::from_millis(1));
    }
    let status = gate.marking.status();
    let scheduled = matches!(
        status,
        CommandBufferStatus::Scheduled | CommandBufferStatus::Completed
    );
    let running = status == CommandBufferStatus::Scheduled;
    let line = format!(
        "{REPORT}-RELEASE held={} hook_ran={} scheduled={} running={} since={since} \
         registrations={now:?}\n",
        u8::from(held),
        u8::from(hook_ran),
        u8::from(scheduled),
        u8::from(running)
    );
    // SAFETY: a write of the line's bytes to standard output.
    unsafe { libc::write(1, line.as_ptr().cast(), line.len()) };
}

/// The handler case's command buffer, a fill of [`HANDLER_MARKER`]'s buffer, encoded and not
/// committed, which [`commit_at_exit`] commits.
static AT_EXIT: OnceLock<CommandBuffer> = OnceLock::new();

/// The 16 shared bytes the handler case's command buffer fills, kept to the process's end.
static HANDLER_MARKER: AtomicPtr<u8> = AtomicPtr::new(std::ptr::null_mut());

/// The hook's registrations in the process right after the handler case registered
/// [`commit_at_exit`]: one more by the exit means the hook was registered again after it.
static AT_THE_HANDLER: AtomicU64 = AtomicU64::new(0);

/// An exit handler the handler case registers once its device is made, as a program registers its
/// own: note whether the device's hook ran before it (`hook_ran`) and the hook's registrations
/// since its own (`since`); then commit the case's command buffer and wait until it completed, as
/// Metal runs a program's work handed over in its exit (`committed`, `completed`, and the marker
/// it fills, `written`). It runs before the handlers registered as the device was made (its
/// driver's and layer's), so the marker's pages are still mapped.
extern "C" fn commit_at_exit() {
    let Some(command_buffer) = AT_EXIT.get() else {
        return;
    };
    let hook_ran = metal2vulkan_runtime::process_exiting();
    let now = metal2vulkan_runtime::exit_registrations();
    let since = now
        .total
        .saturating_sub(AT_THE_HANDLER.load(Ordering::SeqCst));
    let committed = command_buffer.commit();
    let waited = command_buffer.wait_until_completed();
    let completed = command_buffer.status() == CommandBufferStatus::Completed;
    let at = HANDLER_MARKER.load(Ordering::SeqCst);
    // SAFETY: the marker's 16 shared bytes, mapped while the device's driver's handlers have not
    // run (they run after this one), and the case keeps the buffer.
    let written = !at.is_null()
        && unsafe { std::slice::from_raw_parts(at, 16) }
            .iter()
            .all(|&b| b == FILLED);
    let line = format!(
        "{REPORT}-HANDLER hook_ran={} committed={} completed={} written={} since={since} \
         waited={waited:?} registrations={now:?}\n",
        u8::from(hook_ran),
        u8::from(committed.is_ok()),
        u8::from(completed),
        u8::from(written)
    );
    // SAFETY: a write of the line's bytes to standard output.
    unsafe { libc::write(1, line.as_ptr().cast(), line.len()) };
}

/// The device the calls cases' handler calls ([`calls_at_exit`]) and the AIR of the kernel it
/// makes (`store_const`), kept to the process's end.
static CALLS_AT_EXIT: Mutex<Option<(Device, Vec<u8>)>> = Mutex::new(None);

/// An exit handler the calls cases register once the program's own thread made its common calls
/// ([`common_calls`]), as a program registers its own: the same calls again, on the exiting
/// thread, whose thread-locals the exit destroyed before it ran the handlers (macOS), those the
/// runtime made for that thread's earlier calls among them. Its line says whether the device's
/// hook ran before it (`hook_ran`) and what each call answered. Its pipeline's region (two threads
/// a threadgroup) is not the earlier one's (one), so its creation makes it on this thread.
extern "C" fn calls_at_exit() {
    let kept = CALLS_AT_EXIT
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .clone();
    let Some((device, air)) = kept else {
        return;
    };
    let hook_ran = metal2vulkan_runtime::process_exiting();
    let answered = common_calls(&device, &air, 2);
    let line = format!(
        "{REPORT}-CALLS hook_ran={} {answered}\n",
        u8::from(hook_ran)
    );
    // SAFETY: a write of the line's bytes to standard output.
    unsafe { libc::write(1, line.as_ptr().cast(), line.len()) };
}

/// The program's common calls on this thread, and what each answered, as a line's fields: three
/// shared buffers (`buffers`), a 4x4 `RGBA8Unorm` texture (`texture`) and `store_const`'s compute
/// pipeline of `threads` threads a threadgroup, so that its creation makes its pipeline on this
/// thread (`pipeline`: `made`, else the error); then a command buffer that copies 64 bytes from a
/// buffer into the texture and back out into another, and runs the kernel, which stores 42, where
/// its pipeline was made (`committed`, `waited`, `completed`); and what it wrote, read back
/// (`texels`: the bytes came back; `stored`: the kernel's word, `-` where no kernel ran). A call
/// that fails ends the line with its error.
fn common_calls(device: &Device, air: &[u8], threads: u64) -> String {
    let token = |what: &dyn std::fmt::Debug| {
        format!("{what:?}")
            .split_whitespace()
            .collect::<Vec<_>>()
            .join("_")
    };
    let bytes: Vec<u8> = (0u8..64).map(|at| at.wrapping_mul(37) ^ FILLED).collect();
    let shared = |length| device.new_buffer(length, ResourceOptions::shared());
    let (upload, back, out) = match (shared(64), shared(64), shared(4)) {
        (Ok(upload), Ok(back), Ok(out)) => (upload, back, out),
        (upload, back, out) => {
            return format!("buffers={}", token(&(upload.err(), back.err(), out.err())))
        }
    };
    let (Some(at), Some(word)) = (upload.contents(), out.contents()) else {
        return "buffers=no_contents".to_owned();
    };
    // SAFETY: the upload buffer's 64 shared bytes and the output's 4, alive with the buffers, and
    // no command buffer uses them yet.
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), at.as_ptr(), bytes.len());
        std::ptr::write_bytes(word.as_ptr(), 0, 4);
    }
    let descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 4, 4, false);
    let texture = match device.new_texture(&descriptor) {
        Ok(texture) => texture,
        Err(error) => return format!("buffers=1 texture={}", token(&error)),
    };
    let pipeline = device
        .new_library_with_air(air)
        .and_then(|library| library.new_function_with_name("store_const"))
        .and_then(|function| {
            let mut descriptor = ComputePipelineDescriptor::new(StageSource::Function(
                function.function_descriptor(),
            ));
            descriptor.required_threads_per_threadgroup = Some([threads as u32, 1, 1]);
            device.new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
        });
    let made = match &pipeline {
        Ok(_) => "made".to_owned(),
        Err(error) => token(error),
    };
    let encoded = || -> Result<CommandBuffer, metal2vulkan_runtime::Error> {
        let queue = device.new_command_queue()?;
        let command_buffer = queue.command_buffer()?;
        let mut blit = command_buffer.blit_command_encoder()?;
        let size = Size::new(4, 4, 1);
        let origin = Origin::new(0, 0, 0);
        let none = BlitOption::None;
        blit.copy_from_buffer_to_texture(&upload, 0, 16, 64, size, &texture, 0, 0, origin, none)?;
        blit.copy_from_texture_to_buffer(&texture, 0, 0, origin, size, &back, 0, 16, 64, none)?;
        blit.end_encoding()?;
        if let Ok(pipeline) = &pipeline {
            let mut encoder = command_buffer.compute_command_encoder()?;
            encoder.set_compute_pipeline_state(pipeline)?;
            encoder.set_buffer(Some(&out), 0, 0)?;
            encoder.dispatch_threadgroups(Size::new(1, 1, 1), Size::new(threads, 1, 1))?;
            encoder.end_encoding()?;
        }
        Ok(command_buffer)
    };
    let command_buffer = match encoded() {
        Ok(command_buffer) => command_buffer,
        Err(error) => {
            return format!(
                "buffers=1 texture=1 pipeline={made} encoded={}",
                token(&error)
            )
        }
    };
    let committed = match command_buffer.commit() {
        Ok(()) => "1".to_owned(),
        Err(error) => token(&error),
    };
    let waited = token(&command_buffer.wait_until_completed());
    let completed = command_buffer.status() == CommandBufferStatus::Completed;
    let texels = completed
        && back.contents().is_some_and(|at| {
            // SAFETY: the buffer's 64 shared bytes, alive with it; the command buffer that wrote
            // them completed.
            let came_back = unsafe { std::slice::from_raw_parts(at.as_ptr(), 64) };
            came_back == &bytes[..]
        });
    let stored = match (&pipeline, completed) {
        (Ok(_), true) => {
            // SAFETY: the output's 4 shared bytes, alive with it, which the completed command
            // buffer wrote.
            let word = unsafe { word.as_ptr().cast::<u32>().read_unaligned() };
            word.to_string()
        }
        _ => "-".to_owned(),
    };
    format!(
        "buffers=1 texture=1 pipeline={made} committed={committed} waited={waited} completed={} \
         texels={} stored={stored}",
        u8::from(completed),
        u8::from(texels)
    )
}

/// The `store_const` kernel's AIR (the engine fixtures'), which the calls cases make their
/// pipelines of.
fn store_const_air() -> Vec<u8> {
    llvm_assemble(&engine_dir().join("store_const.ll")).unwrap()
}

/// The command buffers the submitting case commits at once.
const COMMITS: usize = 256;

/// The queues the submitting case's command buffers go round: each holds at most 64 uncompleted,
/// as Metal's default queue does, and these hold 32 each.
const COMMIT_QUEUES: usize = 8;

/// The time now, in milliseconds since the epoch: the clock both processes read.
fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |since| since.as_millis())
}

/// Report the case's end and the layer's messages so far, then exit with status 0.
fn exit_now(case: &str, messages: &Messages) -> ! {
    report(case, messages);
    std::process::exit(0)
}

/// Print the line the parent reads, with how many of the layer's host-wait gap's messages
/// passed (`candidate::runtime_case::Messages`; each printed a `LAYER-GAP` line to stderr).
fn report(case: &str, messages: &Messages) {
    let layer_gaps = messages.layer_gaps();
    let messages = messages.take();
    for message in &messages {
        println!("{REPORT}-MESSAGE {message:?}");
    }
    let reported = Since::now();
    println!(
        "{REPORT} {case} at_ms={} messages={} layer_gaps={layer_gaps}",
        reported.ms,
        messages.len()
    );
    *REPORTED.lock().unwrap_or_else(PoisonError::into_inner) = Some(reported);
}

/// An exit handler every child registers first, before its device, so that it runs last of the
/// exit's handlers: print when the handlers ended (`at_ms`), and how long the exiting thread
/// waited for a CPU since the exit's own time began (`queued_us`), which is the host's load and
/// not the exit's. That time begins at the end of the CPU work the exit waits for where the work
/// ended after the child's last line, else at that line ([`exits_cleanly`]). The wait is Linux's
/// count for the thread (`super::run_queue_waits`), read before the instant, so a wait between the
/// two stays in the exit's time; where Linux gives none, or the thread was not yet running at that
/// line, it is 0.
extern "C" fn exit_end() {
    let queued = exit_thread_queued().unwrap_or_default();
    let line = format!(
        "{REPORT}-END at_ms={} queued_us={}\n",
        now_ms(),
        queued.as_micros()
    );
    // SAFETY: a write of the line's bytes to standard output.
    unsafe { libc::write(1, line.as_ptr().cast(), line.len()) };
}

/// The calling thread's wait for a CPU since the exit's own time began ([`exit_end`]).
fn exit_thread_queued() -> Option<Duration> {
    let reported = REPORTED.lock().unwrap_or_else(PoisonError::into_inner);
    let work_ended = WORK_ENDED.lock().unwrap_or_else(PoisonError::into_inner);
    let since = match (reported.as_ref(), work_ended.as_ref()) {
        (Some(reported), Some(ended)) if ended.ms > reported.ms => ended,
        (Some(reported), _) => reported,
        (None, _) => return None,
    };
    let thread = std::fs::read_link("/proc/thread-self").ok()?;
    let thread: u64 = thread.file_name()?.to_str()?.parse().ok()?;
    let waited = *run_queue_waits().get(&thread)?;
    Some(waited.saturating_sub(*since.waits.get(&thread)?))
}

/// A command buffer of `queue` that waits for `event` at 1, then fills 16 bytes of a new buffer.
fn waiting(queue: &CommandQueue, event: EventRef<'_>) -> CommandBuffer {
    let buffer = queue
        .device()
        .new_buffer(16, ResourceOptions::shared())
        .unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    command_buffer.encode_wait_for_event(event, 1).unwrap();
    fill(&command_buffer, &buffer, 16);
    command_buffer
}

/// Encode a fill of `buffer`'s first `length` bytes.
fn fill(command_buffer: &CommandBuffer, buffer: &Buffer, length: u64) {
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.fill_buffer(buffer, 0..length, FILLED).unwrap();
    blit.end_encoding().unwrap();
}

/// A compute state of `fold_config_kernel` (the fixture's `fold_config`), its two layers' opcodes
/// from the clock (shifted by `salt` bits), so no earlier process asked the driver for the same
/// pipeline and the driver compiles it rather than finding it in a cache of its own. Where every
/// dispatch has `threads` threads a threadgroup, its creation makes that region's pipeline on the
/// calling thread; else it makes none.
fn configured(device: &Device, threads: Option<u32>, salt: u32) -> ComputePipelineState {
    let air = llvm_assemble(&engine_dir().join("fold_config_kernel.ll")).unwrap();
    let library = device.new_library_with_air(&air).unwrap();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |since| since.subsec_nanos());
    let opcodes = |shift: u32| {
        (nanos >> (shift + salt))
            .to_le_bytes()
            .map(|byte| byte % 16)
    };
    let byte = |ty, value: u8| FunctionConstantValue::new(ty, vec![value]);
    let constants = FunctionConstantValues::defaults()
        .with_index(0, byte(DataType::Bool, 0))
        .with_index(1, byte(DataType::Bool, 1))
        .with_index(2, byte(DataType::Bool, 1))
        .with_index(3, byte(DataType::UChar, 1))
        .with_index(4, byte(DataType::UChar, 2))
        .with_index(5, FunctionConstantValue::new(DataType::UChar4, opcodes(0)))
        .with_index(6, FunctionConstantValue::new(DataType::UChar4, opcodes(3)))
        .with_index(7, byte(DataType::UChar, 2));
    let function = library
        .new_function_with_name_constant_values("fold_config", &constants)
        .unwrap();
    let mut descriptor =
        ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    descriptor.required_threads_per_threadgroup = threads.map(|threads| [threads, 1, 1]);
    device
        .new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// A command buffer of `queue` that fills 16 bytes of a new buffer.
fn small(queue: &CommandQueue) -> CommandBuffer {
    let buffer = queue
        .device()
        .new_buffer(16, ResourceOptions::shared())
        .unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    fill(&command_buffer, &buffer, 16);
    command_buffer
}

/// A command buffer of `queue` that runs `setup`'s writing kernel, then executes the commands it
/// writes: a batch whose host step reads them at its gate.
fn stepping(queue: &CommandQueue, setup: &Setup) -> CommandBuffer {
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    setup.write(&mut encoder);
    setup.execute(&mut encoder);
    encoder.end_encoding().unwrap();
    command_buffer
}

/// Block this thread: the case's exit happens on another one.
fn wait_for_the_exit(case: &str) -> ! {
    std::thread::sleep(WAIT);
    panic!("{case}: the process did not exit");
}

/// One case, run in a child process: the case [`CASE_VAR`] names, on a device whose shader cache
/// is in [`CACHE_VAR`]'s directory, which the parent made and removes.
#[test]
#[ignore = "run in a child process by this module's tests"]
fn child() {
    let case = std::env::var(CASE_VAR).expect("the case to run");
    let cache = PathBuf::from(std::env::var(CACHE_VAR).expect("the cache directory"));
    // SAFETY: a handler with no arguments that never unwinds.
    unsafe { libc::atexit(exit_end) };
    if case == "cpu_work_exit" {
        // SAFETY: a handler with no arguments that never unwinds.
        unsafe { libc::atexit(read_the_marker) };
    }
    let messages = Messages::default();
    let device = validated_device_with(TEST_DEVICE, &messages, &cache, at_plan).unwrap();
    // Right after the device: the hook's later registrations, and every handler a case registers,
    // run before it, and the device's driver's and layer's handlers after it.
    // SAFETY: a handler with no arguments that never unwinds.
    unsafe { libc::atexit(after_the_later_devices) };
    let case = case.as_str();
    match case {
        "pending_held_exit" => {
            let queue = device.new_command_queue().unwrap();
            let event = device.new_shared_event().unwrap();
            let command_buffer = waiting(&queue, EventRef::from(&event));
            command_buffer.commit().unwrap();
            assert_eq!(command_buffer.status(), CommandBufferStatus::Committed);
            exit_now(case, &messages);
        }
        "pending_dropped_exit" | "pending_dropped_return" => {
            {
                let queue = device.new_command_queue().unwrap();
                let event = device.new_shared_event().unwrap();
                waiting(&queue, EventRef::from(&event)).commit().unwrap();
            }
            drop(device);
            if case == "pending_dropped_exit" {
                exit_now(case, &messages);
            }
            report(case, &messages);
        }
        "host_step_pending_exit" => {
            // The setup's work once to completion first: its pipelines are made, so no driver's
            // making of one is in progress as the process exits, which the hook would wait for.
            let setup = Setup::new(&device, 2);
            let queue = device.new_command_queue().unwrap();
            let made = stepping(&queue, &setup);
            made.commit().unwrap();
            made.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(made.status(), CommandBufferStatus::Completed);
            // The second command buffer's batch waits at its gate for the fills before it.
            let (length, fills) = in_flight();
            let buffer = device
                .new_buffer(length, ResourceOptions::private())
                .unwrap();
            let filling = queue.command_buffer().unwrap();
            for _ in 0..fills {
                fill(&filling, &buffer, length);
            }
            let stepping = stepping(&queue, &setup);
            filling.commit().unwrap();
            stepping.commit().unwrap();
            exit_now(case, &messages);
        }
        "making_exit" | "first_making_exit" => {
            // A state whose creation makes its pipeline to its end on this thread: the making's
            // driver call, the driver's compile, returns before the exit. `first_making_exit`
            // makes none, so the pool's making below is the process's first of the kernel, and
            // the driver's compiler makes its statics for it while it runs.
            if case == "making_exit" {
                drop(configured(&device, Some(64), 0));
            }
            // A state whose creation makes none: its first dispatch hands its region's making to
            // the device's pool and goes on (`pipeline/dispatch_making.rs`), and the child exits
            // while the pool's thread is in the driver's compile.
            let pending = configured(&device, None, 3);
            let queue = device.new_command_queue().unwrap();
            let buffer = device
                .new_buffer(64 * 4, ResourceOptions::shared())
                .unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer.compute_command_encoder().unwrap();
            encoder.set_compute_pipeline_state(&pending).unwrap();
            encoder.set_buffer(Some(&buffer), 0, 0).unwrap();
            encoder.set_bytes(&[0; 80], 1).unwrap();
            encoder.set_buffer(Some(&buffer), 0, 2).unwrap();
            encoder
                .dispatch_threadgroups(Size::new(2, 1, 1), Size::new(32, 1, 1))
                .unwrap();
            encoder.end_encoding().unwrap();
            command_buffer.commit().unwrap();
            // The pool's pieces held as the child exits (its making in progress where 1), and the
            // hook's registrations: one after each making's driver call.
            println!(
                "{REPORT}-MAKING pool_busy={} {:?}",
                device.work_counts().pipeline_pool_busy,
                metal2vulkan_runtime::exit_registrations()
            );
            exit_now(case, &messages);
        }
        "cpu_work_exit" => {
            // The device's submission checks run on a thread of their own, whose sections register
            // the hook again (`exit_guard.rs`, "Registered again"): they end first, so that the
            // hook's last registration comes before `release_the_gate`'s, and the hook runs after
            // it.
            assert!(
                device.settle_submit_checks(WAIT),
                "{case}: the submission checks answered"
            );
            // Long work held at a gate, then a fill of a shared marker after it on the same queue;
            // the process exits with both held, and `release_the_gate`, first in the exit, opens
            // the gate and waits until both are submitted: the device runs the work as the
            // process exits, whatever the host's load did before.
            let queue = device.new_command_queue().unwrap();
            let length = CPU_WORK_FILL;
            let buffer = device
                .new_buffer(length, ResourceOptions::private())
                .unwrap();
            // As many fills as run for `CPU_WORK_TIME` by one fill's time, 8 to 64.
            let one = queue.command_buffer().unwrap();
            fill(&one, &buffer, length);
            let start = Instant::now();
            one.commit().unwrap();
            one.wait_until_completed().unwrap();
            let each = start.elapsed().as_nanos().max(1);
            let fills = usize::try_from(CPU_WORK_TIME.as_nanos() / each)
                .unwrap_or(64)
                .clamp(8, 64);
            let marker = device.new_buffer(16, ResourceOptions::shared()).unwrap();
            let event = device.new_shared_event().unwrap();
            let work = queue.command_buffer().unwrap();
            work.encode_wait_for_event(EventRef::from(&event), 1)
                .unwrap();
            for _ in 0..fills {
                fill(&work, &buffer, length);
            }
            let marking = queue.command_buffer().unwrap();
            fill(&marking, &marker, 16);
            work.commit().unwrap();
            marking.commit().unwrap();
            let cpu = device.properties().kind == DeviceKind::Cpu;
            if cpu {
                let at = marker.contents().expect("a shared buffer's contents");
                MARKER.store(at.as_ptr(), Ordering::SeqCst);
                watch_the_marker(at);
            }
            println!("{REPORT}-WORK {case} cpu={}", u8::from(cpu));
            println!(
                "{REPORT}-SWITCH {} at {:?}",
                device.queue_switch_check_report(),
                metal2vulkan_runtime::exit_registrations()
            );
            assert!(
                GATE.set(Gate { event, marking }).is_ok(),
                "one gate a process"
            );
            // Registered after every runtime call of the case, so after the device's hook.
            // SAFETY: a handler with no arguments that never unwinds.
            unsafe { libc::atexit(release_the_gate) };
            AT_THE_GATE.store(
                metal2vulkan_runtime::exit_registrations().total,
                Ordering::SeqCst,
            );
            exit_now(case, &messages);
        }
        "handler_work_exit" => {
            // The device's submission checks' sections end first, as in `cpu_work_exit`.
            assert!(
                device.settle_submit_checks(WAIT),
                "{case}: the submission checks answered"
            );
            let queue = device.new_command_queue().unwrap();
            let first = small(&queue);
            first.commit().unwrap();
            first.wait_until_completed().unwrap();
            let marker: &'static Buffer = Box::leak(Box::new(
                device.new_buffer(16, ResourceOptions::shared()).unwrap(),
            ));
            let at = marker.contents().expect("a shared buffer's contents");
            HANDLER_MARKER.store(at.as_ptr(), Ordering::SeqCst);
            let command_buffer = queue.command_buffer().unwrap();
            fill(&command_buffer, marker, 16);
            assert!(AT_EXIT.set(command_buffer).is_ok(), "one handler a process");
            // As a program registers its own exit handler once its device is made.
            // SAFETY: a handler with no arguments that never unwinds.
            unsafe { libc::atexit(commit_at_exit) };
            AT_THE_HANDLER.store(
                metal2vulkan_runtime::exit_registrations().total,
                Ordering::SeqCst,
            );
            // Then the program goes on: a pipeline made on its own thread, after its handler.
            let _made = configured(&device, Some(32), 5);
            println!(
                "{REPORT}-MADE cpu={} {:?}",
                u8::from(device.properties().kind == DeviceKind::Cpu),
                metal2vulkan_runtime::exit_registrations()
            );
            exit_now(case, &messages);
        }
        "handler_calls_exit" | "handler_calls_after_making_exit" => {
            // The device's submission checks' sections end first, as in `cpu_work_exit`.
            assert!(
                device.settle_submit_checks(WAIT),
                "{case}: the submission checks answered"
            );
            // The program's calls on its own thread first: the thread-locals the runtime keeps
            // for them are made on this thread, and the exit destroys them before the handlers.
            let air = store_const_air();
            let answered = common_calls(&device, &air, 1);
            println!("{REPORT}-OWN {answered}");
            assert!(
                answered.ends_with("completed=1 texels=1 stored=42"),
                "{case}: the program's own calls: {answered}"
            );
            *CALLS_AT_EXIT.lock().unwrap_or_else(PoisonError::into_inner) =
                Some((device.clone(), air));
            // As a program registers its own exit handler once its device is made.
            // SAFETY: a handler with no arguments that never unwinds.
            unsafe { libc::atexit(calls_at_exit) };
            if case == "handler_calls_after_making_exit" {
                // A pipeline made after the handler: the exit ends the makings before it.
                let _made = configured(&device, Some(32), 7);
            }
            println!(
                "{REPORT}-MADE cpu={}",
                u8::from(device.properties().kind == DeviceKind::Cpu)
            );
            exit_now(case, &messages);
        }
        "instance_after_handler_exit" => {
            // Each device another sink, so another instance; the first keeps the layer loaded. A
            // device made and dropped first: its making's sections register the hook again after
            // each entry point's first call since the first device's, which the last one's then
            // do not.
            drop(validated_device_with(TEST_DEVICE, &messages, &cache, at_plan).unwrap());
            // SAFETY: a handler with no arguments that never unwinds.
            unsafe { libc::atexit(drop_the_held_device) };
            let before = metal2vulkan_runtime::exit_registrations();
            let last = validated_device_with(TEST_DEVICE, &messages, &cache, at_plan).unwrap();
            let after = metal2vulkan_runtime::exit_registrations();
            println!("{REPORT}-REGISTRATIONS before={before:?} after={after:?}");
            *HELD.lock().unwrap_or_else(PoisonError::into_inner) = Some(last);
            exit_now(case, &messages);
        }
        "program_instance_exit" | "adopted_instance_exit" => {
            // The runtime's threads in sections at the exit: a loop without bound runs on.
            super::predication::loop_without_bound(&device, 3);
            // The program's own instance and device, with the validation layer, whose exit
            // handler the instance's creation registers; then a handler of the case's after it.
            let adopted = super::presenting::Adopted::open(false).unwrap();
            // SAFETY: a handler with no arguments that never unwinds.
            unsafe { libc::atexit(after_the_program_instance) };
            if case == "program_instance_exit" {
                metal2vulkan_runtime::program_instance_made();
            } else {
                // Kept to the process's end, as the instance and device it adopts are.
                std::mem::forget(adopted.runtime(&cache));
            }
            std::mem::forget(adopted);
            exit_now(case, &messages);
        }
        "pending_plain_event_exit" => {
            {
                let queue = device.new_command_queue().unwrap();
                let event = device.new_event().unwrap();
                waiting(&queue, EventRef::from(&event)).commit().unwrap();
            }
            drop(device);
            exit_now(case, &messages);
        }
        "completed_handler_exits" => {
            let queue = device.new_command_queue().unwrap();
            let command_buffer = small(&queue);
            let in_handler = messages.clone();
            command_buffer
                .add_completed_handler(Box::new(move |_: &CommandBuffer| {
                    exit_now("completed_handler_exits", &in_handler)
                }))
                .unwrap();
            command_buffer.commit().unwrap();
            wait_for_the_exit(case);
        }
        "scheduled_handler_exits_in_flight" => {
            // The handler runs once the work is submitted, while the GPU runs it and the
            // queue's waiter polls for it.
            let queue = device.new_command_queue().unwrap();
            let (length, fills) = in_flight();
            let buffer = device
                .new_buffer(length, ResourceOptions::private())
                .unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            for _ in 0..fills {
                fill(&command_buffer, &buffer, length);
            }
            let in_handler = messages.clone();
            command_buffer
                .add_scheduled_handler(Box::new(move |_: &CommandBuffer| {
                    exit_now("scheduled_handler_exits_in_flight", &in_handler)
                }))
                .unwrap();
            command_buffer.commit().unwrap();
            wait_for_the_exit(case);
        }
        "deallocator_exits" => {
            let queue = device.new_command_queue().unwrap();
            let page = host_page();
            let memory = anonymous(page);
            let in_deallocator = messages.clone();
            let deallocator: Deallocator = Box::new(move |_: NonNull<u8>, _: u64| {
                exit_now("deallocator_exits", &in_deallocator)
            });
            // SAFETY: the pages stay mapped for the life of the process.
            let made = unsafe {
                device.new_buffer_with_bytes_no_copy(
                    HostMemory::new(memory),
                    page as u64,
                    ResourceOptions::default(),
                    Some(deallocator),
                )
            };
            let buffer = match made {
                Ok(buffer) => buffer,
                Err(error) => {
                    // A device that refuses the import typed: nothing to release.
                    println!("{REPORT}-SKIPPED {case}: {error}");
                    exit_now(case, &messages);
                }
            };
            let command_buffer = queue.command_buffer().unwrap();
            fill(&command_buffer, &buffer, 64);
            // The command buffer holds the last reference: its release runs the deallocator on
            // a handler thread.
            drop(buffer);
            command_buffer.commit().unwrap();
            wait_for_the_exit(case);
        }
        "listener_exits" => {
            let event = device.new_shared_event().unwrap();
            let listener = SharedEventListener::new();
            let in_listener = messages.clone();
            event.notify_listener(
                &listener,
                1,
                Box::new(move |_: &SharedEvent, _| exit_now("listener_exits", &in_listener)),
            );
            event.set_signaled_value(1);
            wait_for_the_exit(case);
        }
        "submitting_exit" => {
            // Many command buffers encoded first, then committed at once: the scheduler is still
            // submitting them, and the queues' waiters polling, when the program exits right
            // after. (A commit calls no Vulkan function; the program's own Vulkan calls during
            // its exit would be its own race.)
            let queues: Vec<CommandQueue> = (0..COMMIT_QUEUES)
                .map(|_| device.new_command_queue().unwrap())
                .collect();
            let buffer = device.new_buffer(16, ResourceOptions::shared()).unwrap();
            let command_buffers: Vec<CommandBuffer> = (0..COMMITS)
                .map(|at| {
                    let command_buffer = queues[at % COMMIT_QUEUES].command_buffer().unwrap();
                    fill(&command_buffer, &buffer, 16);
                    command_buffer
                })
                .collect();
            for command_buffer in &command_buffers {
                command_buffer.commit().unwrap();
            }
            exit_now(case, &messages);
        }
        "status_learns_exit" => {
            let command_buffer = small(&device.new_command_queue().unwrap());
            command_buffer.commit().unwrap();
            drop(device);
            let deadline = std::time::Instant::now() + WAIT;
            while command_buffer.status() != CommandBufferStatus::Completed {
                assert!(std::time::Instant::now() < deadline, "never completed");
                std::thread::yield_now();
            }
            drop(command_buffer);
            exit_now(case, &messages);
        }
        "event_learns_exit" => {
            let (told, learn) = mpsc::channel();
            {
                let queue = device.new_command_queue().unwrap();
                let event = device.new_shared_event().unwrap();
                let listener = SharedEventListener::new();
                event.notify_listener(
                    &listener,
                    1,
                    Box::new(move |_: &SharedEvent, _| {
                        let _ = told.send(());
                    }),
                );
                let command_buffer = small(&queue);
                command_buffer
                    .encode_signal_event(EventRef::from(&event), 1)
                    .unwrap();
                command_buffer.commit().unwrap();
                drop(device);
                learn.recv_timeout(WAIT).expect("the listener heard");
            }
            exit_now(case, &messages);
        }
        "handler_learns_exit" => {
            let (told, learn) = mpsc::channel();
            {
                let command_buffer = small(&device.new_command_queue().unwrap());
                command_buffer
                    .add_completed_handler(Box::new(move |_: &CommandBuffer| {
                        let _ = told.send(());
                    }))
                    .unwrap();
                command_buffer.commit().unwrap();
                drop(device);
                learn.recv_timeout(WAIT).expect("the handler ran");
            }
            exit_now(case, &messages);
        }
        "released_import_exit" | "released_import_teardown_exit" => {
            released_import(device, case, &messages);
        }
        "handler_drops_last_exit" => {
            let (told, learn) = mpsc::channel();
            {
                let queue = device.new_command_queue().unwrap();
                let command_buffer = small(&queue);
                let (kept_queue, kept_buffer) = (queue.clone(), command_buffer.clone());
                command_buffer
                    .add_completed_handler(Box::new(move |_: &CommandBuffer| {
                        drop((kept_queue, kept_buffer));
                        let _ = told.send(());
                    }))
                    .unwrap();
                command_buffer.commit().unwrap();
            }
            drop(device);
            learn.recv_timeout(WAIT).expect("the handler ran");
            exit_now(case, &messages);
        }
        other => panic!("no case {other}"),
    }
}

/// The kernel the released-import cases dispatch after their fills: it samples a texture through a
/// sampler, both named by the IDs its argument buffer holds, so its command buffer binds the
/// device's heap set, which reaches every live texture.
const ID_READER: &str =
    "sample-an-unsigned-texture-through-a-sampler-held-beside-it-in-an-argument-buffer";

/// How long the released-import cases' fills keep the GPU busy, as one fill's time predicts.
const BUSY: Duration = Duration::from_millis(60);

/// The fills that keep the device busy for about [`BUSY`]: a buffer's length and how many fills
/// of it, as 16 fills' time measured on `queue` predicts on a GPU (16 to 1024 of them); on a CPU
/// device [`in_flight`]'s.
fn held_fills(device: &Device, queue: &CommandQueue) -> (u64, usize) {
    let (length, fills) = in_flight();
    if TEST_DEVICE == DeviceType::Cpu {
        return (length, fills);
    }
    let buffer = device
        .new_buffer(length, ResourceOptions::private())
        .unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    fills_of(&command_buffer, &buffer, length, 16);
    let start = Instant::now();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    let each = (start.elapsed() / 16).as_nanos().max(1);
    let fills = usize::try_from(BUSY.as_nanos() / each)
        .unwrap_or(1024)
        .clamp(16, 1024);
    (length, fills)
}

/// Encode `fills` fills of `buffer`'s first `length` bytes, in one blit encoder.
fn fills_of(command_buffer: &CommandBuffer, buffer: &Buffer, length: u64, fills: usize) {
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    for _ in 0..fills {
        blit.fill_buffer(buffer, 0..length, FILLED).unwrap();
    }
    blit.end_encoding().unwrap();
}

/// The released-import cases, in reims-vgpu's shape (a texture over a surface released with the
/// surface while a command buffer that binds the device's heap set runs): twice, a texture over a
/// no-copy buffer of fresh host pages, whose deallocator unmaps them; a command buffer of fills
/// that keep the device busy for about [`BUSY`], then a dispatch that reads another texture
/// through its ID ([`ID_READER`]); the texture and its buffer released once the command buffer is
/// scheduled, while the fills run.
///
/// - `released_import_exit`: each command buffer and deallocator waited for, then everything let
///   go of but the device, which the process keeps (a static of the program's), and the process
///   exits at the end of `main`, the runtime still releasing what the program let go of.
/// - `released_import_teardown_exit`: the second command buffer's completed handler tells the
///   program, which has let go of everything, the device included, and exits at once: the command
///   buffer's release on the handler thread lets go of the device's last reference, so its
///   teardown runs as the process exits.
fn released_import(device: Device, case: &str, messages: &Messages) {
    // As reims-vgpu's test: the device's own checks end first.
    let _ = device.settle_in_place_checks(WAIT);
    let _ = device.settle_queue_switch_check(WAIT);
    let _ = device.settle_submit_checks(WAIT);
    let queue = device.new_command_queue().unwrap();
    let reader = public_case(ID_READER).unwrap();
    let state = pipeline_state(&device, &reader).unwrap();
    let resources = CaseResources::new(&device, &reader).unwrap();
    let buffers = bound_buffers(&device, &reader.compute, &resources).unwrap();
    let (length, fills) = held_fills(&device, &queue);
    let filled = device
        .new_buffer(length, ResourceOptions::private())
        .unwrap();
    let page = host_page();
    let len = (64 * 32 * 4usize).div_ceil(page) * page;
    let teardown = case == "released_import_teardown_exit";
    for round in 0..2 {
        let (freed, seen) = mpsc::channel();
        let memory = anonymous(len);
        let deallocator: Deallocator = Box::new(move |at: NonNull<u8>, length: u64| {
            // SAFETY: the pages `anonymous` mapped for this buffer alone, which nothing reads
            // any more.
            let unmapped = unsafe { libc::munmap(at.as_ptr().cast(), length as usize) } == 0;
            let _ = freed.send(unmapped);
        });
        // SAFETY: the pages stay mapped until the deallocator runs.
        let made = unsafe {
            device.new_buffer_with_bytes_no_copy(
                HostMemory::new(memory),
                len as u64,
                ResourceOptions::shared(),
                Some(deallocator),
            )
        };
        let buffer = match made {
            Ok(buffer) => buffer,
            Err(error) => {
                // A device that refuses the import typed: nothing to release.
                println!("{REPORT}-SKIPPED {case}: {error}");
                exit_now(case, messages);
            }
        };
        let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, 64, 32, false);
        descriptor.usage = TextureUsage::SHADER_READ;
        descriptor.resource_options = ResourceOptions::shared();
        let texture = buffer.new_texture(&descriptor, 0, 64 * 4).unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        fills_of(&command_buffer, &filled, length, fills);
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encode_case(&mut encoder, &state, &reader.compute, &buffers, &resources).unwrap();
        encoder.end_encoding().unwrap();
        let (told, learn) = mpsc::channel();
        command_buffer
            .add_completed_handler(Box::new(move |_: &CommandBuffer| {
                let _ = told.send(());
            }))
            .unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_scheduled().unwrap();
        let running = command_buffer.status() == CommandBufferStatus::Scheduled;
        drop((texture, buffer));
        println!(
            "{REPORT}-ROUND {case} round={round} fills={fills} running={}",
            u8::from(running)
        );
        if teardown && round == 1 {
            drop((
                command_buffer,
                queue,
                state,
                resources,
                buffers,
                filled,
                device,
            ));
            learn.recv_timeout(WAIT).expect("the handler ran");
            exit_now(case, messages);
        }
        learn.recv_timeout(WAIT).expect("the handler ran");
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        assert_eq!(
            seen.recv_timeout(WAIT),
            Ok(true),
            "the deallocator unmapped the pages"
        );
    }
    // The program keeps its device for the process's life; the rest it lets go of, and its
    // releases run as the process exits.
    std::mem::forget(device);
    report(case, messages);
}

/// The host's page size.
fn host_page() -> usize {
    // SAFETY: `sysconf` reads a system constant.
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    usize::try_from(page).expect("a page size")
}

/// `len` fresh private anonymous bytes, never unmapped.
fn anonymous(len: usize) -> NonNull<u8> {
    // SAFETY: a fresh mapping.
    let at = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            len,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANON,
            -1,
            0,
        )
    };
    assert_ne!(at, libc::MAP_FAILED, "mmap");
    NonNull::new(at.cast()).expect("a mapping")
}

/// What a child process did.
struct Exited {
    status: ExitStatus,
    stdout: String,
    stderr: String,
    /// When the parent saw it end, in milliseconds since the epoch.
    ended_ms: u128,
    /// When the parent read the runtime's notice that it ends the process at once ([`AT_ONCE`])
    /// on its standard error, in milliseconds since the epoch; `None` where it wrote none.
    at_once_ms: Option<u128>,
}

/// Run `case` in a child process, for at most [`WAIT`] (then it is killed, and this fails).
fn run_child(case: &str) -> Exited {
    let cache = ScratchCache::new().unwrap();
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "runtime_lane::exits::child",
            "--ignored",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(CASE_VAR, case)
        .env(CACHE_VAR, cache.path())
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    // The pipe's text, and when `mark` first showed in it.
    let read = |mut pipe: Box<dyn Read + Send>, mark: &'static str| {
        std::thread::spawn(move || {
            let (mut bytes, mut chunk, mut marked) = (Vec::new(), [0; 4096], None);
            loop {
                match pipe.read(&mut chunk) {
                    Ok(0) => break,
                    Ok(read) => {
                        bytes.extend_from_slice(&chunk[..read]);
                        let shown = || bytes.windows(mark.len()).any(|w| w == mark.as_bytes());
                        if marked.is_none() && !mark.is_empty() && shown() {
                            marked = Some(now_ms());
                        }
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                    Err(_) => break,
                }
            }
            (String::from_utf8_lossy(&bytes).into_owned(), marked)
        })
    };
    let stdout = read(Box::new(child.stdout.take().unwrap()), "");
    let stderr = read(Box::new(child.stderr.take().unwrap()), AT_ONCE);
    let pid = child.id();
    let (ended, end) = mpsc::channel();
    std::thread::spawn(move || {
        let status = child.wait();
        let _ = ended.send((status, now_ms()));
    });
    let Ok((status, ended_ms)) = end.recv_timeout(WAIT) else {
        // SAFETY: a signal to the child this test started, which has not been reaped.
        unsafe { libc::kill(pid as libc::pid_t, libc::SIGKILL) };
        panic!("{case}: the child did not end within {WAIT:?}");
    };
    let (stdout, _) = stdout.join().unwrap();
    let (stderr, at_once_ms) = stderr.join().unwrap();
    Exited {
        status: status.unwrap(),
        stdout,
        stderr,
        ended_ms,
        at_once_ms,
    }
}

/// The first line of `exited`'s standard output from `start` on (the harness may print the test's
/// name on the same line first).
fn stdout_line<'a>(exited: &'a Exited, start: &str) -> Option<&'a str> {
    exited
        .stdout
        .lines()
        .find_map(|line| Some(&line[line.find(start)?..]))
}

/// Run each of `cases` in a child process, and check each exit: status 0, no signal, no
/// validation message, and its exit handlers ended ([`exit_end`]'s `at_ms`; where the child
/// printed none, the runtime's at-once notice as the parent read it, else the process's end)
/// within [`EXIT_LIMIT`] of the child's last line, or of the end of
/// the CPU device's work the exit waits for where that is later (the marker's `seen_ms`), less the
/// time the exiting thread waited for a CPU since (`queued_us`; none where the child gave no
/// count). Where the child marked the end of the handlers registered since its first device
/// ([`after_the_later_devices`]) and the hook ran before the mark, the exit's part ends at the
/// mark: the handlers after it are those registered as the first device was made, its driver's and
/// the layer's, and the hook's first registration, whose run finds no device and makes no Vulkan
/// call (`exit_guard::hook`), so their time is the driver's and the layer's (on an RTX 5080 laptop
/// beside other GPU work, 454 to 997 ms where the rest of the exit took 2 ms, and 1.5 to 5.9 s
/// beside the runtime suite), printed and not bounded. What each child did, in order.
fn exits_cleanly(cases: &[&str]) -> Vec<Exited> {
    let mut all = Vec::new();
    for &case in cases {
        let exited = run_child(case);
        let Exited {
            status,
            stdout,
            stderr,
            ended_ms,
            at_once_ms,
        } = &exited;
        let output = || format!("stdout:\n{stdout}\nstderr:\n{stderr}");
        #[cfg(unix)]
        {
            use std::os::unix::process::ExitStatusExt;
            assert_eq!(
                status.signal(),
                None,
                "{case}: ended by a signal\n{}",
                output()
            );
        }
        assert!(status.success(), "{case}: {status}\n{}", output());
        let line = stdout_line(&exited, &format!("{REPORT} {case} "))
            .unwrap_or_else(|| panic!("{case}: no report\n{}", output()));
        let field = |name: &str| -> u128 {
            line.split_whitespace()
                .find_map(|pair| pair.strip_prefix(name)?.strip_prefix('='))
                .and_then(|value| value.parse().ok())
                .unwrap_or_else(|| panic!("{case}: no {name} in {line}"))
        };
        assert_eq!(
            field("messages"),
            0,
            "{case}: the layer reported\n{}",
            output()
        );
        let at_ms = field("at_ms");
        // The work the exit waits for runs as long as the host's load lets it: the exit's own
        // time runs from its end.
        let work_ended_ms = stdout_line(&exited, &format!("{REPORT}-MARKER "))
            .and_then(|line| {
                line.split_whitespace()
                    .find_map(|pair| pair.strip_prefix("seen_ms="))?
                    .parse::<u128>()
                    .ok()
            })
            .unwrap_or(0);
        let from_ms = at_ms.max(work_ended_ms);
        let end = stdout_line(&exited, &format!("{REPORT}-END "));
        let end_field = |name: &str| -> Option<u128> {
            end?.split_whitespace()
                .find_map(|pair| pair.strip_prefix(name)?.strip_prefix('='))?
                .parse()
                .ok()
        };
        // Where the child marked the end of the handlers registered since its first device, the
        // hook's run among them, the exit's part ends at the mark: its first device's driver's and
        // layer's handlers after it are printed, not counted.
        let mark = stdout_line(&exited, &format!("{REPORT}-MARK "))
            .filter(|mark| mark.split_whitespace().any(|pair| pair == "hook_ran=1"));
        let mark_field = |name: &str| -> Option<u128> {
            mark?
                .split_whitespace()
                .find_map(|pair| pair.strip_prefix(name)?.strip_prefix('='))?
                .parse()
                .ok()
        };
        // Where the runtime ended the process at once, the exit's part ends at its notice: the
        // kernel's teardown of the process after it is printed, not counted.
        let handled_ms = end_field("at_ms").or(*at_once_ms).unwrap_or(*ended_ms);
        let own_ms = mark_field("at_ms").unwrap_or(handled_ms);
        let wall = Duration::from_millis(own_ms.saturating_sub(from_ms) as u64);
        let queued = mark_field("queued_us")
            .or_else(|| end_field("queued_us"))
            .map_or(Duration::ZERO, |us| {
                Duration::from_micros(u64::try_from(us).unwrap_or(u64::MAX))
            });
        let first_device = mark_field("at_ms").map(|mark| handled_ms.saturating_sub(mark));
        let took = wall.saturating_sub(queued);
        let after = Duration::from_millis(ended_ms.saturating_sub(handled_ms) as u64);
        let work = Duration::from_millis(from_ms.saturating_sub(at_ms) as u64);
        assert!(
            took < EXIT_LIMIT,
            "{case}: the exit took {took:?} ({wall:?} of wall time, less {queued:?} the exiting \
             thread waited for a CPU; the process ended {after:?} after its handlers){}\n{}",
            match work_ended_ms > at_ms {
                true => format!(
                    " after the CPU device's work it waited for ended ({work:?} after the last \
                     line)"
                ),
                false => String::new(),
            },
            output()
        );
        let skipped = stdout.contains(&format!("{REPORT}-SKIPPED {case}"));
        println!(
            "runtime exit, {case}: {} ms ({} ms of wall time, less {} ms the exiting thread \
             waited for a CPU; the process ended {} ms after its handlers){}{}{}",
            took.as_millis(),
            wall.as_millis(),
            queued.as_millis(),
            after.as_millis(),
            if work_ended_ms > at_ms {
                format!(
                    " after the work it waited for, which ended {} ms after the last line",
                    work.as_millis()
                )
            } else {
                String::new()
            },
            if skipped {
                " (host memory refused)"
            } else {
                ""
            },
            match first_device {
                Some(ms) => format!(
                    "; then the first device's driver's and layer's handlers, not counted, \
                     {ms} ms"
                ),
                None => String::new(),
            }
        );
        // The child exits before its message list goes, so its count is printed here.
        for line in stderr.lines().filter(|line| line.starts_with("LAYER-GAP ")) {
            println!("{line}");
        }
        let layer_gaps = field("layer_gaps");
        if layer_gaps != 0 {
            println!("LAYER-GAP count {case}: {layer_gaps} layer gap message(s) passed");
        }
        all.push(exited);
    }
    all
}

/// A command buffer still waiting for an event at exit (held, dropped, or for a plain event that
/// only the GPU can signal) stays pending, and the process exits at once, by `exit` or at the end
/// of `main`. Metal's exits were clean in 15 to 30 ms, and 5.03 s for the plain event.
#[test]
fn pending_work_stays_pending_at_exit() {
    exits_cleanly(&[
        "pending_held_exit",
        "pending_dropped_exit",
        "pending_dropped_return",
        "pending_plain_event_exit",
    ]);
}

/// An exit from the program's code on a runtime thread, which holds no section of the exit
/// guard: a completed handler, a scheduled handler while the GPU still runs its work and the
/// queue's waiter polls for it, a no-copy buffer's deallocator run by a command buffer's
/// release, and a listener's notification. The hook does not wait for the exiting thread.
#[test]
fn exits_from_the_runtime_s_threads_are_clean() {
    exits_cleanly(&[
        "completed_handler_exits",
        "scheduled_handler_exits_in_flight",
        "deallocator_exits",
        "listener_exits",
    ]);
}

/// An exit while a batch waits at its host step's gate for the work before it: the scheduler ends
/// at its next section, and what is pending stays pending. The same work ran once to completion
/// before, so that its pipelines are made and no driver's making of one is in progress at the
/// exit: the hook waits for such a making up to its own bound, which the runtime's
/// `the_hook_waits_past_its_bound_for_a_making_in_progress` tests.
#[test]
fn an_exit_while_a_host_step_waits_is_clean() {
    exits_cleanly(&["host_step_pending_exit"]);
}

/// An exit while a pool's thread makes a dispatch's pipeline, after a pipeline made to its end on
/// the program's thread, both new to the driver (`configured`): the hook, registered again after
/// the first making's driver call, runs before every destructor the driver's compiler registered
/// in it, and waits for the second making before them. Seen before: on an RTX 5080 a consumer's
/// test process ended with `SIGSEGV` at `0x18` inside `libnvidia-glvkspirv.so`, the driver's SPIR-V
/// compiler, during its exit after its test passed (one test of a whole run under load).
///
/// The exit's own part ends at the mark the child registers right after its device
/// ([`after_the_later_devices`]): the hook's handlers, its wait for the making, and the compiler's
/// destructors run before it, the device's driver's and layer's handlers after it. On an RTX 5080
/// laptop beside the runtime suite, the part before the mark ended 8 to 41 ms after the last line,
/// with the making ended by then, and the handlers after it took 1.5 to 5.9 s, printed and not
/// bounded ([`exits_cleanly`]).
///
/// And the same exit with no making before the pool's (`first_making_exit`): the pool's making is
/// the process's first of the kernel, so the driver's compiler makes its statics, and registers
/// their destructors, while it runs, and the registrar registers the hook again after them
/// (`exit_guard.rs`, "Registered again"). Measured on lavapipe before the registrar: in 1 of 3
/// such exits, destructors LLVM registered 1 to 2 ms before the exit ran before the hook, the
/// compile going on.
#[test]
fn an_exit_mid_making_is_clean() {
    let cases = ["making_exit", "first_making_exit"];
    for (case, exited) in cases.into_iter().zip(exits_cleanly(&cases)) {
        // Where the hook ended the process at once, no handler after it ran ([`AT_ONCE`]).
        let marked = stdout_line(&exited, &format!("{REPORT}-MARK ")).is_some();
        assert!(
            marked || exited.at_once_ms.is_some(),
            "{case}: no mark between the hook's handlers and the driver's"
        );
        let line = stdout_line(&exited, &format!("{REPORT}-MAKING "))
            .unwrap_or_else(|| panic!("{case}: no line of its making"));
        println!("{case}: {line}");
        assert!(
            case != "making_exit" || !line.contains("at_makings: 0,"),
            "{case}: no making registered the hook again: {line}"
        );
    }
}

/// An exit while a CPU device runs work submitted before it (fills it runs for about
/// [`CPU_WORK_TIME`], then a fill of a shared marker) runs no exit handler beside that work, which
/// runs on the process's own threads: the hook waits for the work, and the marker is written by
/// the time [`read_the_marker`] runs, after the device's hook, with no notice that the exit ended
/// at once ([`AT_ONCE`]). The exit's own time counts from the work's end, which a thread watching
/// the marker sees: the work's own time follows the host's load (1.585 s from the last line to the
/// end of an exit beside other lanes at a load of 30 to 45, 47 to 58 ms alone, on lavapipe).
///
/// That is the interleaving of an exit with a dispatch in flight on lavapipe, made certain: the
/// work waits at a gate until the exit has begun, where a handler that runs before the device's
/// hook ([`release_the_gate`]) opens it and waits until the work is submitted, so it runs as the
/// hook starts, for far longer than the exit's own teardown. Without the hook's wait the marker was
/// unwritten in 4 of 20 exits with a fixed 512 MiB of fills, which lavapipe ran within the
/// teardown's 50 ms in the other 16. On a GPU the hook does not wait for the work (as Metal's exit
/// does not), and the exit, with the work started in it, is clean: the hook waits for the sections
/// in progress (the completion waiter's wait for the work), returns, and the marker's handler
/// runs. The gate makes the work's state as the exit
/// begins certain on every device and at any load: before it, the child read the status once the
/// work was submitted, and on an RTX 3090 (NVIDIA 615.71.09) beside 24 CPU spinners the GPU
/// finished the work (15 to 18 fills of 64 MiB, sized by one fill's 11 to 13 ms round trip there)
/// in the 2 to 3 ms the child took to see its submission and read its status, in 2 of 20 runs; and
/// in 2 batches beside other lanes.
#[test]
fn an_exit_runs_no_handler_beside_a_cpu_device_s_work() {
    let case = "cpu_work_exit";
    let exited = exits_cleanly(&[case]).remove(0);
    let field = |line: &str, name: &str| -> Option<u8> {
        stdout_line(&exited, line)?
            .split_whitespace()
            .find_map(|pair| pair.strip_prefix(name)?.strip_prefix('='))?
            .parse()
            .ok()
    };
    let what = || format!("stdout:\n{}\nstderr:\n{}", exited.stdout, exited.stderr);
    let release = format!("{REPORT}-RELEASE ");
    let marker = format!("{REPORT}-MARKER ");
    let switch = format!("{REPORT}-SWITCH ");
    for line in [&switch, &release] {
        eprintln!("{}", stdout_line(&exited, line).unwrap_or_default());
    }
    assert_eq!(
        field(&release, "since"),
        Some(0),
        "the hook was registered again after the gate's handler, so it ran first: {}",
        what()
    );
    assert_eq!(
        field(&release, "held"),
        Some(1),
        "the work was not held at its gate when the exit began: {}",
        what()
    );
    assert_eq!(
        field(&release, "hook_ran"),
        Some(0),
        "the device's hook ran before the gate opened: {}",
        what()
    );
    assert_eq!(
        field(&release, "scheduled"),
        Some(1),
        "the work was not submitted once the gate opened: {}",
        what()
    );
    let cpu = field(&format!("{REPORT}-WORK {case} "), "cpu") == Some(1);
    if cpu {
        assert_eq!(
            field(&release, "running"),
            Some(1),
            "the work completed before the device's hook began: {}",
            what()
        );
    }
    let notices = exited.stderr.matches(AT_ONCE).count();
    assert!(
        notices <= 1,
        "the exit's notice was written {notices} times: {}",
        what()
    );
    assert_eq!(
        notices,
        0,
        "the exit ended at once, its work running past the hook's bound: {}",
        what()
    );
    assert_eq!(
        field(&marker, "exiting"),
        Some(1),
        "the handler registered before the device did not run, or the runtime did not answer that \
         the process is exiting: {}",
        what()
    );
    if cpu {
        assert_eq!(
            field(&marker, "written"),
            Some(1),
            "the exit did not wait for the CPU device's work: {}",
            what()
        );
    }
    println!(
        "runtime exit, {case} on a {}: the device's hook returned, and the handlers before it ran",
        match cpu {
            true => "CPU device",
            false => "GPU",
        },
    );
}

/// An exit handler the program registers once its device is made commits work as the process
/// exits and waits for it: Metal runs that work to completion, and so must the runtime, whatever
/// the program did after registering the handler (here, a pipeline made on its own thread). The
/// device's hook runs after the handler, so the handler finds the device's guard open; the hook's
/// run for the making's registration, before the handler, ended the makings alone. On a CPU
/// device, whose queues compile as they run, that run closes the guard (docs/API.md, accepted
/// differences): the handler finds the process exiting, and its wait answers so.
#[test]
fn an_exit_handler_registered_after_the_device_runs_its_work_to_completion() {
    let case = "handler_work_exit";
    let exited = exits_cleanly(&[case]).remove(0);
    let handler = format!("{REPORT}-HANDLER ");
    let line_field = |line: &str, name: &str| -> Option<u8> {
        stdout_line(&exited, line)?
            .split_whitespace()
            .find_map(|pair| pair.strip_prefix(name)?.strip_prefix('='))?
            .parse()
            .ok()
    };
    let field = |name: &str| line_field(&handler, name);
    let what = || format!("stdout:\n{}\nstderr:\n{}", exited.stdout, exited.stderr);
    eprintln!("{}", stdout_line(&exited, &handler).unwrap_or_default());
    if line_field(&format!("{REPORT}-MADE "), "cpu") == Some(1) {
        assert_eq!(
            (field("hook_ran"), field("completed")),
            (Some(1), Some(0)),
            "on a CPU device the making's run of the hook did not close the guard: {}",
            what()
        );
        assert!(
            stdout_line(&exited, &handler).is_some_and(|line| line.contains("Exiting")),
            "the handler's wait did not answer that the process is exiting: {}",
            what()
        );
        return;
    }
    assert_eq!(
        field("hook_ran"),
        Some(0),
        "the device's hook ran before the program's handler: {}",
        what()
    );
    assert_eq!(
        field("committed"),
        Some(1),
        "the handler's commit failed: {}",
        what()
    );
    assert_eq!(
        field("completed"),
        Some(1),
        "the handler's work did not complete: {}",
        what()
    );
    assert_eq!(
        field("written"),
        Some(1),
        "the handler's work did not fill its marker: {}",
        what()
    );
}

/// An exit handler registered after the device makes the program's common calls on the exiting
/// thread ([`common_calls`]: a buffer, a texture and a compute pipeline made, a copy through the
/// texture and a dispatch encoded, committed and waited for, what they wrote read back), after the
/// program's own thread made them before the exit, so the runtime's thread-locals for them were
/// made there and the exit destroyed them before it ran the handlers (macOS). Each call answers
/// as on any other thread, and the process exits cleanly. Where the program registered no handler
/// after a later pipeline making, every call runs and its work completes, on every device. Where
/// it made a pipeline after the handler, the handler's making answers that the process is exiting
/// (docs/API.md, accepted differences, "Process exit"); on a GPU the rest of its work completes,
/// and on a CPU device that making's run of the hook closed the runtime, so the handler's work
/// answers that the process is exiting too.
#[test]
fn an_exit_handler_s_common_calls_answer_as_on_any_thread() {
    let cases = ["handler_calls_exit", "handler_calls_after_making_exit"];
    for (case, exited) in cases.into_iter().zip(exits_cleanly(&cases)) {
        let calls = format!("{REPORT}-CALLS ");
        let what = || {
            format!(
                "{case}: stdout:\n{}\nstderr:\n{}",
                exited.stdout, exited.stderr
            )
        };
        let line = stdout_line(&exited, &calls)
            .unwrap_or_else(|| panic!("the handler wrote no line: {}", what()));
        eprintln!("{case}: {line}");
        let field = |name: &str| {
            line.split_whitespace()
                .find_map(|pair| pair.strip_prefix(name)?.strip_prefix('='))
        };
        let cpu = stdout_line(&exited, &format!("{REPORT}-MADE "))
            .is_some_and(|made| made.contains("cpu=1"));
        let all = (
            field("hook_ran"),
            field("pipeline"),
            field("committed"),
            field("completed"),
            field("texels"),
            field("stored"),
        );
        match (case, cpu) {
            ("handler_calls_exit", _) => assert_eq!(
                all,
                (
                    Some("0"),
                    Some("made"),
                    Some("1"),
                    Some("1"),
                    Some("1"),
                    Some("42")
                ),
                "the handler's calls did not all run: {}",
                what()
            ),
            (_, false) => {
                assert!(
                    field("pipeline").is_some_and(|made| made.contains("Exiting")),
                    "the handler's making did not answer that the process is exiting: {}",
                    what()
                );
                assert_eq!(
                    (all.0, all.2, all.3, all.4, all.5),
                    (Some("0"), Some("1"), Some("1"), Some("1"), Some("-")),
                    "the handler's other work did not complete: {}",
                    what()
                );
            }
            (_, true) => {
                assert!(
                    field("hook_ran") == Some("1") && field("completed") != Some("1"),
                    "on a CPU device the making's run of the hook did not close the runtime: {}",
                    what()
                );
                assert!(
                    line.contains("Exiting"),
                    "the handler's calls did not answer that the process is exiting: {}",
                    what()
                );
            }
        }
    }
}

/// An exit right after committing many command buffers: the scheduler is still submitting them,
/// and the queues' waiters polling.
#[test]
fn an_exit_while_submitting_is_clean() {
    exits_cleanly(&["submitting_exit"]);
}

/// A program that learns its work finished from the status, an event's listener or a completed
/// handler (or whose handler drops the last references) and then drops everything and exits at
/// once: the device's teardown runs on a runtime thread, and the exit waits for it or leaves the
/// device to the process's end. Metal's exits were clean in 25 to 45 ms.
#[test]
fn exits_right_after_learning_of_completion_are_clean() {
    exits_cleanly(&[
        "status_learns_exit",
        "event_learns_exit",
        "handler_learns_exit",
        "handler_drops_last_exit",
    ]);
}

/// An exit after a texture over a no-copy buffer's host pages was released while a command buffer
/// that binds the device's heap set ran (reims-vgpu's shape): with the device kept for the
/// process's life, the runtime still releasing what the program let go of, and with the device's
/// last reference let go of on the handler thread, its teardown racing the exit.
#[test]
fn exits_after_a_released_import_are_clean() {
    exits_cleanly(&["released_import_exit", "released_import_teardown_exit"]);
}

/// An exit handler of the program's, registered between two devices, finds the hook ran, and
/// drops the later device: the validation layer registers an exit handler at each instance, which
/// forgets every device it knows, after which its next call for one ends the process. The hook
/// registers again after each instance the runtime creates, so it runs before the layer's handler
/// and closes the guard, and the teardown makes no Vulkan call. Before, on the M3 Max, a pool's
/// making for a lane's third device, in progress at exit, ended its process with `SIGABRT` in the
/// layer ("The VkDevice dispatch handle was not found").
#[test]
fn an_exit_handler_registered_before_an_instance_runs_after_the_hook() {
    let exited = exits_cleanly(&["instance_after_handler_exit"]);
    let line = stdout_line(&exited[0], &format!("{REPORT}-HELD "))
        .unwrap_or_else(|| panic!("no line from the handler: {}", exited[0].stdout));
    assert!(
        line.contains("hook_ran=1"),
        "the handler ran before the hook: {line}"
    );
    let at = |start: &str, name: &str| -> Option<u128> {
        stdout_line(&exited[0], start)?
            .split_whitespace()
            .find_map(|pair| pair.strip_prefix(name)?.strip_prefix('='))?
            .parse()
            .ok()
    };
    let last = at(&format!("{REPORT} instance_after_handler_exit "), "at_ms");
    let held = at(&format!("{REPORT}-HELD "), "at_ms");
    let dropped = at(&format!("{REPORT}-HELD "), "dropped_ms");
    let mark = at(&format!("{REPORT}-MARK "), "at_ms");
    let end = at(&format!("{REPORT}-END "), "at_ms");
    if let (Some(last), Some(held), Some(dropped), Some(mark), Some(end)) =
        (last, held, dropped, mark, end)
    {
        println!(
            "runtime exit, instance_after_handler_exit's parts: {} ms to the program's handler \
             (the hook, the last instance's handlers), {} ms its drop of the last device, {} ms \
             the second device's handlers, {} ms the first device's",
            held.saturating_sub(last),
            dropped.saturating_sub(held),
            mark.saturating_sub(dropped),
            end.saturating_sub(mark)
        );
    }
}

/// An instance the program makes itself, reported (`program_instance_made`) or the instance of a
/// device the runtime adopts (`Device::from_raw`), registers the exit hook again after the exit
/// handler the validation layer registers as the instance is created, which forgets every device
/// the layer knows: the hook runs first and closes the guard, so no call of the runtime's (here a
/// loop without bound's, which runs on its own thread) reaches the layer after its handler ran.
/// Before, on the M3 Max, a child that made its own instance after its device while a loop without
/// bound ran on ended with `SIGABRT` inside the layer in 2 and 3 exits of 20 (the layer's lookup
/// of the device, at the loop's next call).
#[test]
fn an_instance_the_program_made_registers_the_exit_hook_after_its_layer_s_handler() {
    for exited in exits_cleanly(&["program_instance_exit", "adopted_instance_exit"]) {
        let line = stdout_line(&exited, &format!("{REPORT}-PROGRAM-INSTANCE "))
            .unwrap_or_else(|| panic!("no line from the handler: {}", exited.stdout));
        assert!(
            line.contains("hook_ran=1"),
            "the layer's handler ran before the hook: {line}"
        );
    }
}

/// A command buffer whose program dropped every handle but the shared event it waits for stays
/// pending, and completes once the event is signalled: its completed handler runs, and sees the
/// status `Completed` (Metal's answer: status 4).
#[test]
fn a_dropped_command_buffer_completes_once_its_event_is_signalled() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device: Device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let event = device.new_shared_event().unwrap();
        let (told, learn) = mpsc::channel();
        {
            let queue = device.new_command_queue().unwrap();
            let command_buffer = waiting(&queue, EventRef::from(&event));
            command_buffer
                .add_completed_handler(Box::new(move |command_buffer: &CommandBuffer| {
                    let _ = told.send(command_buffer.status());
                }))
                .unwrap();
            command_buffer.commit().unwrap();
        }
        drop(device);
        assert!(
            learn.recv_timeout(Duration::from_millis(200)).is_err(),
            "the handler ran before the signal"
        );
        event.set_signaled_value(1);
        assert_eq!(learn.recv_timeout(WAIT), Ok(CommandBufferStatus::Completed));
        // The device goes with the event's last handle, once the runtime let go of the command
        // buffer.
        drop(event);
        let deadline = std::time::Instant::now() + WAIT;
        while messages.holders() > 1 {
            assert!(
                std::time::Instant::now() < deadline,
                "the device never went"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
    }
    layer_reported_nothing_at_all(&messages);
}

/// Devices made again in a long-lived process register the exit hook no more while what they
/// load stays loaded: with one device held, whose instance keeps the Vulkan loader, the driver and
/// the layers loaded, 16 devices made and dropped after two that warm up add a registration at a
/// device only where the process loaded a shared object as that device was made (a host's driver
/// manifest whose library the loader loads and drops again at each instance loads one at each;
/// where nothing loads, none), one after each of their instances (a layer registers an exit
/// handler at each), and the sections' registrations stay within their bound (one after each
/// pipeline making, priming's included, aside).
#[test]
fn devices_made_again_register_the_exit_hook_no_more() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let held: Device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        for _ in 0..2 {
            drop(validated_device(TEST_DEVICE, &messages, cache.path()).unwrap());
        }
        let before = metal2vulkan_runtime::exit_registrations();
        let mut loads = before.loads;
        let mut made_after_a_load = 0;
        for _ in 0..16 {
            drop(validated_device(TEST_DEVICE, &messages, cache.path()).unwrap());
            let now = metal2vulkan_runtime::exit_registrations().loads;
            if now.is_none() || now != loads {
                made_after_a_load += 1;
            }
            loads = now;
        }
        let after = metal2vulkan_runtime::exit_registrations();
        println!(
            "runtime exit registrations: {before:?}, then after 16 devices, {made_after_a_load} \
             of them made as the process loaded a shared object, {after:?}"
        );
        assert_eq!(
            after.at_devices - before.at_devices,
            made_after_a_load,
            "devices made again registered the hook past one per device made after a load"
        );
        assert_eq!(
            after.at_instances - before.at_instances,
            16,
            "each instance made registers the hook once after it"
        );
        assert!(
            after.total
                - after.at_devices
                - after.at_instances
                - after.at_makings
                - after.during_makings
                <= after.section_bound,
            "the sections registered past their bound: {after:?}"
        );
        drop(held);
    }
    layer_reported_nothing_at_all(&messages);
}
