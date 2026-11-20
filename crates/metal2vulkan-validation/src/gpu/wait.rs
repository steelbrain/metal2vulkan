//! Finite host waits on device work, and the hang protocol.
//!
//! Every host wait on submitted device work in this crate goes through this module, and every
//! one of them is bounded by [`wait_timeout()`] (10 s unless a command overrides it with
//! [`WAIT_TIMEOUT_FLAG`]). A unit test fails if an unbounded wait appears anywhere else.
//!
//! When a wait times out, the submission may still be running on the device:
//!
//! 1. nothing the submission uses is freed: the process never returns to the caller, so no
//!    destructor of an in-flight resource runs;
//! 2. for a GPU, the tripwire `~/.cache/m2v-gpu/hang` is written ([`tripwire::HangRecord`]), which
//!    stops every later GPU entry point on the host until a human has followed the
//!    suspected-hang protocol in `docs/VALIDATION.md`;
//! 3. the record is printed and the process leaves through `_exit(86)`, without running device
//!    destructors or `atexit` handlers.
//!
//! A wait that fails without telling whether the submission finished is handled like a timeout.
//!
//! A submission that completes with a device-fault status (Vulkan `VK_ERROR_DEVICE_LOST`; Metal
//! `MTLCommandBufferError` timeout, page fault, access revoked, or device removed) is a suspected
//! GPU fault: it writes the tripwire too, then fails the operation normally, because nothing is
//! in flight any more; a wait on the new runtime's command buffers (`new_runtime`) leaves how
//! the command buffer ended to its caller instead. A CPU implementation (lavapipe, SwiftShader)
//! is an ordinary killable process: its timeouts and faults follow the same exit rules but never
//! write the host GPU tripwire.

use super::tripwire::{self, HangKind, HangRecord};
use super::DeviceType;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

/// Default bound on one host wait for device work.
pub const DEFAULT_WAIT_TIMEOUT: Duration = Duration::from_secs(10);

/// Largest bound a command may ask for. A longer wait is indistinguishable from a hang for the
/// operator watching the host.
pub const MAX_WAIT_TIMEOUT: Duration = Duration::from_secs(600);

/// Command-line flag with which device-executing binaries override [`DEFAULT_WAIT_TIMEOUT`].
pub const WAIT_TIMEOUT_FLAG: &str = "--gpu-wait-timeout";

/// Exit status of a process that stopped on a timed-out device wait.
pub const HANG_EXIT_CODE: i32 = 86;

static WAIT_TIMEOUT_MS: AtomicU64 = AtomicU64::new(DEFAULT_WAIT_TIMEOUT.as_millis() as u64);

/// The bound every device wait in this process uses.
pub fn wait_timeout() -> Duration {
    Duration::from_millis(WAIT_TIMEOUT_MS.load(Ordering::Relaxed))
}

/// Set the bound for this process. Commands call it once, from their [`WAIT_TIMEOUT_FLAG`]
/// argument, before opening a device.
pub fn set_wait_timeout(timeout: Duration) -> Result<(), String> {
    if timeout.is_zero() || timeout > MAX_WAIT_TIMEOUT {
        return Err(format!(
            "device wait timeout {timeout:?} is outside 1 ms ..= {MAX_WAIT_TIMEOUT:?}"
        ));
    }
    let millis = u64::try_from(timeout.as_millis()).expect("bounded by MAX_WAIT_TIMEOUT");
    WAIT_TIMEOUT_MS.store(millis, Ordering::Relaxed);
    Ok(())
}

/// Parse a [`WAIT_TIMEOUT_FLAG`] value: a whole number of seconds, 1 ..= 600.
pub fn parse_wait_timeout(value: &str) -> Result<Duration, String> {
    let seconds = value
        .parse::<u64>()
        .map_err(|error| format!("{WAIT_TIMEOUT_FLAG} {value:?}: {error}"))?;
    let timeout = Duration::from_secs(seconds);
    if seconds == 0 || timeout > MAX_WAIT_TIMEOUT {
        return Err(format!(
            "{WAIT_TIMEOUT_FLAG} {value:?}: expected whole seconds in 1..={}",
            MAX_WAIT_TIMEOUT.as_secs()
        ));
    }
    Ok(timeout)
}

/// The device a waited-on submission runs on, as recorded in the tripwire.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SubmissionBackend {
    /// A CPU Vulkan implementation. Never writes the host GPU tripwire.
    VulkanCpu,
    /// A Vulkan GPU.
    VulkanGpu,
    /// A Metal device.
    Metal,
}

impl SubmissionBackend {
    pub fn vulkan(device_type: DeviceType) -> Self {
        match device_type {
            DeviceType::Cpu => Self::VulkanCpu,
            DeviceType::Gpu => Self::VulkanGpu,
        }
    }

    /// Whether a stuck or faulted submission on this backend can wedge the host GPU.
    pub fn is_gpu(self) -> bool {
        !matches!(self, Self::VulkanCpu)
    }
}

/// What an executor is running on the current thread, so a hang record names it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SubmissionLabel {
    pub case_id: String,
    /// SHA-256 of the executed module: the SPIR-V for a Vulkan candidate, the AIR source for Metal.
    pub module_sha256: String,
}

thread_local! {
    // Device contexts are thread-local, so the work on a thread is the only work its waits can be
    // for.
    static CURRENT_LABEL: RefCell<Option<SubmissionLabel>> = const { RefCell::new(None) };
}

/// Run `body` with `label` naming every device wait it makes on this thread. The previous label
/// is restored afterwards, also when `body` unwinds.
pub fn with_submission_label<R>(label: SubmissionLabel, body: impl FnOnce() -> R) -> R {
    struct Restore(Option<SubmissionLabel>);
    impl Drop for Restore {
        fn drop(&mut self) {
            CURRENT_LABEL.with(|current| current.borrow_mut().clone_from(&self.0));
        }
    }
    let previous = CURRENT_LABEL.with(|current| current.borrow_mut().replace(label));
    let _restore = Restore(previous);
    body()
}

pub(crate) fn current_label() -> Option<SubmissionLabel> {
    CURRENT_LABEL.with(|current| current.borrow().clone())
}

/// How one bounded wait for device work ended.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum WaitOutcome {
    /// The work finished without error.
    Completed,
    /// The bound elapsed first; the work may still be running.
    TimedOut,
    /// The wait itself failed and the work may still be running.
    Unresolved(String),
    /// The work ended with a status that indicates a device fault.
    SuspectedFault(String),
    /// The work ended with an ordinary error.
    Failed(String),
}

/// Something that can be waited on with a bound: a Vulkan fence, a Metal command buffer, or a
/// test double.
pub(crate) trait CompletionSource {
    fn wait(&mut self, timeout: Duration) -> WaitOutcome;
}

/// The settled result of one wait, before any process-level effect.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Verdict {
    Completed,
    Failed(String),
    Fault(HangRecord),
    Hang(HangRecord),
}

/// Wait on `source` for at most `timeout` and classify the result. Pure apart from the wait
/// itself, so tests can drive it with a fake source.
pub(crate) fn judge(
    source: &mut dyn CompletionSource,
    backend: SubmissionBackend,
    operation: &str,
    timeout: Duration,
) -> Verdict {
    let record = |kind, detail: String| {
        let label = current_label();
        HangRecord::now(
            kind,
            backend,
            operation,
            detail,
            timeout,
            label.as_ref().map(|label| label.case_id.clone()),
            label.map(|label| label.module_sha256),
        )
    };
    match source.wait(timeout) {
        WaitOutcome::Completed => Verdict::Completed,
        WaitOutcome::Failed(detail) => Verdict::Failed(format!("{operation} failed: {detail}")),
        WaitOutcome::SuspectedFault(detail) => {
            Verdict::Fault(record(HangKind::SuspectedFault, detail))
        }
        WaitOutcome::TimedOut => Verdict::Hang(record(
            HangKind::Timeout,
            format!("no completion within {timeout:?}"),
        )),
        WaitOutcome::Unresolved(detail) => Verdict::Hang(record(HangKind::Unresolved, detail)),
    }
}

/// Wait on `source` with this process's bound and apply the hang protocol. Returns only when
/// the work is no longer in flight.
pub(crate) fn settle(
    source: &mut dyn CompletionSource,
    backend: SubmissionBackend,
    operation: &str,
) -> Result<(), String> {
    match judge(source, backend, operation, wait_timeout()) {
        Verdict::Completed => Ok(()),
        Verdict::Failed(error) => Err(error),
        Verdict::Fault(record) => {
            report_incident(&record, tripwire::write_host_record);
            Err(record.summary())
        }
        Verdict::Hang(record) => hang(&record),
    }
}

/// Record `record`: write the tripwire through `write_tripwire` for a GPU backend, and print it
/// either way. The tripwire comes first, and the printing never panics: a stderr nobody reads any
/// more (a closed pipe) loses the lines, while a panic would unwind past work still in flight.
pub(crate) fn report_incident(
    record: &HangRecord,
    write_tripwire: impl FnOnce(&HangRecord) -> Result<(), String>,
) {
    let note = |line: &str| {
        use std::io::Write as _;
        let _ = writeln!(std::io::stderr(), "m2v-gpu: {line}");
    };
    if record.backend.is_gpu() {
        if let Err(error) = write_tripwire(record) {
            note(&error);
        }
    }
    note(&record.summary());
    match serde_json::to_string(record) {
        Ok(json) => note(&json),
        Err(error) => note(&format!("cannot encode hang record: {error}")),
    }
}

/// The end of a timed-out wait: record it and leave without destroying anything the device may
/// still be using.
fn hang(record: &HangRecord) -> ! {
    report_incident(record, tripwire::write_host_record);
    // SAFETY: `_exit` ends the process immediately. Skipping Rust destructors and C `atexit`
    // handlers is the point: they would free memory and objects a live queue still references.
    unsafe { libc::_exit(HANG_EXIT_CODE) }
}

/// Bounded waits on Vulkan fences, timeline semaphores and queues.
#[cfg(any(target_os = "linux", target_os = "macos"))]
pub(crate) mod vulkan {
    use super::{settle, CompletionSource, SubmissionBackend, WaitOutcome};
    use ash::{vk, Device};
    use std::time::Duration;

    struct Fence<'a> {
        device: &'a Device,
        fence: vk::Fence,
    }

    impl CompletionSource for Fence<'_> {
        fn wait(&mut self, timeout: Duration) -> WaitOutcome {
            let nanos = u64::try_from(timeout.min(super::MAX_WAIT_TIMEOUT).as_nanos())
                .expect("the wait bound fits in u64 nanoseconds");
            match unsafe { self.device.wait_for_fences(&[self.fence], true, nanos) } {
                Ok(()) => WaitOutcome::Completed,
                Err(vk::Result::TIMEOUT) => WaitOutcome::TimedOut,
                Err(vk::Result::ERROR_DEVICE_LOST) => {
                    WaitOutcome::SuspectedFault("VK_ERROR_DEVICE_LOST".into())
                }
                // A failed wait (out of memory) says nothing about the submission. It is over
                // only if its fence is observably signaled; otherwise it may still be running.
                Err(error) => match unsafe { self.device.get_fence_status(self.fence) } {
                    Ok(true) => WaitOutcome::Failed(format!("fence wait: {error}")),
                    Ok(false) => WaitOutcome::Unresolved(format!(
                        "fence wait failed ({error}) and the fence is unsignaled"
                    )),
                    Err(status) => WaitOutcome::Unresolved(format!(
                        "fence wait failed ({error}) and its status is unknown ({status})"
                    )),
                },
            }
        }
    }

    /// Wait for `fence`, signaled by a submission `operation` just made, under the hang
    /// protocol. Returns only once the submission is no longer in flight.
    pub(crate) fn wait_for_fence(
        device: &Device,
        fence: vk::Fence,
        backend: SubmissionBackend,
        operation: &str,
    ) -> Result<(), String> {
        settle(&mut Fence { device, fence }, backend, operation)
    }

    /// Wait, bounded, until everything submitted to `queue` has finished, so the device may be
    /// destroyed. Replaces `vkDeviceWaitIdle`, which cannot time out: an empty submission with a
    /// fence signals once all earlier work on the queue has completed.
    ///
    /// `Err` means the device must be leaked rather than destroyed. A timeout does not return.
    pub(crate) fn drain_queue(
        device: &Device,
        queue: vk::Queue,
        backend: SubmissionBackend,
    ) -> Result<(), String> {
        let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }
            .map_err(|error| format!("create queue drain fence: {error}"))?;
        let result = unsafe { device.queue_submit(queue, &[], fence) }
            .map_err(|error| format!("submit queue drain fence: {error}"))
            .and_then(|()| wait_for_fence(device, fence, backend, "Vulkan queue drain"));
        if result.is_ok() {
            unsafe { device.destroy_fence(fence, None) };
        }
        result
    }
}

/// Bounded waits on the Metal-like runtime's command buffers (`metal2vulkan_runtime`): the
/// engine executor's runtime path.
pub(crate) mod runtime {
    use super::{settle, CompletionSource, SubmissionBackend, WaitOutcome};
    use metal2vulkan_runtime::{CommandBuffer, CommandBufferError, CommandBufferStatus};
    use std::time::Duration;

    struct Committed<'a>(&'a CommandBuffer);

    impl CompletionSource for Committed<'_> {
        fn wait(&mut self, timeout: Duration) -> WaitOutcome {
            if self
                .0
                .wait_until_completed_for(timeout.min(super::MAX_WAIT_TIMEOUT))
                .is_err()
            {
                return WaitOutcome::TimedOut;
            }
            match (self.0.status(), self.0.error()) {
                (CommandBufferStatus::Completed, _) => WaitOutcome::Completed,
                (_, Some(CommandBufferError::DeviceLost)) => {
                    WaitOutcome::SuspectedFault("the command buffer lost its device".into())
                }
                (status, error) => {
                    WaitOutcome::Failed(format!("the command buffer ended {status:?}: {error:?}"))
                }
            }
        }
    }

    /// Wait for `command_buffer`, committed by `operation`, under the hang protocol. Returns
    /// only once its work is no longer in flight.
    pub(crate) fn wait_for_command_buffer(
        command_buffer: &CommandBuffer,
        backend: SubmissionBackend,
        operation: &str,
    ) -> Result<(), String> {
        settle(&mut Committed(command_buffer), backend, operation)
    }
}

/// Bounded waits on the new runtime's command buffers (`metal2vulkan_rt`): the replay player's
/// backend over it. The runtime's own wait has no bound, so the wait is on the command buffer's
/// completed handler, after the queue's wait hint has sent it. How the command buffer ended is the
/// caller's to read (the player reads its error into the play's verdict): a wait here says only
/// that its work is no longer in flight, so an ending with a device fault writes no tripwire.
pub(crate) mod new_runtime {
    use super::WaitOutcome;
    use super::{hang, judge, wait_timeout, CompletionSource, SubmissionBackend, Verdict};
    use metal2vulkan_rt::{CommandBuffer, CommandQueue, Error};
    use std::sync::{Arc, Condvar, Mutex, PoisonError};
    use std::time::Duration;

    /// Whether a command buffer has retired: its completed handler ran, which the runtime runs
    /// once it has let go of everything the command buffer held.
    #[derive(Default)]
    pub(crate) struct Retired {
        done: Mutex<bool>,
        condvar: Condvar,
    }

    impl Retired {
        /// The flag of `command_buffer`, not yet committed, which its completed handler sets.
        pub(crate) fn watch(command_buffer: &CommandBuffer) -> Result<Arc<Self>, Error> {
            let retired = Arc::new(Self::default());
            let flag = Arc::clone(&retired);
            command_buffer.add_completed_handler(move |_| {
                *flag.done.lock().unwrap_or_else(PoisonError::into_inner) = true;
                flag.condvar.notify_all();
            })?;
            Ok(retired)
        }

        /// Whether the command buffer has retired.
        pub(crate) fn is_set(&self) -> bool {
            *self.done.lock().unwrap_or_else(PoisonError::into_inner)
        }
    }

    struct Waited<'a>(&'a Retired);

    impl CompletionSource for Waited<'_> {
        fn wait(&mut self, timeout: Duration) -> WaitOutcome {
            let done = self.0.done.lock().unwrap_or_else(PoisonError::into_inner);
            let (done, _) = (self.0.condvar)
                .wait_timeout_while(done, timeout, |done| !*done)
                .unwrap_or_else(PoisonError::into_inner);
            match *done {
                true => WaitOutcome::Completed,
                false => WaitOutcome::TimedOut,
            }
        }
    }

    /// Wait until `command_buffer`, committed on `queue`, has retired (`retired`), under the hang
    /// protocol: a wait past the process's bound records the hang, named by `operation`, and ends
    /// the process ([`HANG_EXIT_CODE`](super::HANG_EXIT_CODE)), so this returns only once it has
    /// retired.
    pub(crate) fn wait_until_retired(
        queue: &CommandQueue,
        command_buffer: &CommandBuffer,
        retired: &Retired,
        backend: SubmissionBackend,
        operation: &str,
    ) {
        queue.send_now(command_buffer);
        // The handler's flag is set or the bound elapsed: completed or a hang, nothing else.
        if let Verdict::Hang(record) =
            judge(&mut Waited(retired), backend, operation, wait_timeout())
        {
            hang(&record)
        }
    }
}

/// Bounded waits on Metal command buffers. [`commit_and_wait`](metal::commit_and_wait) is public
/// for the crate's binaries (the format scout's Metal side).
#[cfg(target_os = "macos")]
pub mod metal {
    use super::{settle, CompletionSource, SubmissionBackend, WaitOutcome};
    use block2::RcBlock;
    use core::ptr::NonNull;
    use objc2::runtime::ProtocolObject;
    use objc2_metal::{
        MTLCommandBuffer, MTLCommandBufferError, MTLCommandBufferErrorDomain,
        MTLCommandBufferStatus,
    };
    use std::sync::{Arc, Condvar, Mutex, PoisonError};
    use std::time::Duration;

    #[derive(Default)]
    struct Signal {
        completed: Mutex<bool>,
        condvar: Condvar,
    }

    struct CommandBuffer<'a> {
        command_buffer: &'a ProtocolObject<dyn MTLCommandBuffer>,
        signal: Arc<Signal>,
    }

    impl CompletionSource for CommandBuffer<'_> {
        fn wait(&mut self, timeout: Duration) -> WaitOutcome {
            let completed = self
                .signal
                .completed
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            let (completed, _) = self
                .signal
                .condvar
                .wait_timeout_while(completed, timeout, |completed| !*completed)
                .unwrap_or_else(PoisonError::into_inner);
            if !*completed {
                return WaitOutcome::TimedOut;
            }
            drop(completed);
            classify(self.command_buffer)
        }
    }

    /// Classify a command buffer whose completed handler has run.
    fn classify(command_buffer: &ProtocolObject<dyn MTLCommandBuffer>) -> WaitOutcome {
        let status = command_buffer.status();
        if status == MTLCommandBufferStatus::Completed {
            return WaitOutcome::Completed;
        }
        let Some(error) = command_buffer.error() else {
            return WaitOutcome::Failed(format!("ended with status {status:?}"));
        };
        // SAFETY: an immutable framework constant.
        let domain = unsafe { MTLCommandBufferErrorDomain };
        let fault = error.domain().isEqualToString(domain)
            && usize::try_from(error.code()).is_ok_and(|code| {
                [
                    MTLCommandBufferError::Timeout,
                    MTLCommandBufferError::PageFault,
                    MTLCommandBufferError::AccessRevoked,
                    MTLCommandBufferError::DeviceRemoved,
                ]
                .contains(&MTLCommandBufferError(code))
            });
        if fault {
            WaitOutcome::SuspectedFault(error.to_string())
        } else {
            WaitOutcome::Failed(error.to_string())
        }
    }

    /// A completed handler on a command buffer, which [`settle_watched`] waits on.
    pub(crate) struct Watch(Arc<Signal>);

    /// Add to `command_buffer` the completed handler [`settle_watched`] waits on. Made before
    /// the command buffer's `commit`, which the caller makes.
    pub(crate) fn watch(command_buffer: &ProtocolObject<dyn MTLCommandBuffer>) -> Watch {
        let signal = Arc::new(Signal::default());
        let handler_signal = Arc::clone(&signal);
        let handler = RcBlock::new(move |_: NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
            *handler_signal
                .completed
                .lock()
                .unwrap_or_else(PoisonError::into_inner) = true;
            handler_signal.condvar.notify_all();
        });
        // SAFETY: the block is valid for the call, and Metal copies it before returning. The
        // handler must be added before commit.
        unsafe { command_buffer.addCompletedHandler(RcBlock::as_ptr(&handler)) };
        Watch(signal)
    }

    /// Wait for `command_buffer`, which `watch` watches and which was committed since, under
    /// the hang protocol. `Ok` only when the command buffer reached `Completed`.
    pub(crate) fn settle_watched(
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
        watch: Watch,
        operation: &str,
    ) -> Result<(), String> {
        settle(
            &mut CommandBuffer {
                command_buffer,
                signal: watch.0,
            },
            SubmissionBackend::Metal,
            operation,
        )
    }

    /// Commit `command_buffer` and wait for it under the hang protocol. Replaces
    /// `commit` + `waitUntilCompleted`, which cannot time out. `Ok` only when the command buffer
    /// reached `Completed`.
    pub fn commit_and_wait(
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
        operation: &str,
    ) -> Result<(), String> {
        let watch = watch(command_buffer);
        command_buffer.commit();
        settle_watched(command_buffer, watch, operation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    struct Fake {
        outcome: WaitOutcome,
        waited: Option<Duration>,
    }

    impl Fake {
        fn new(outcome: WaitOutcome) -> Self {
            Self {
                outcome,
                waited: None,
            }
        }
    }

    impl CompletionSource for Fake {
        fn wait(&mut self, timeout: Duration) -> WaitOutcome {
            self.waited = Some(timeout);
            self.outcome.clone()
        }
    }

    /// A source that really blocks: completion arrives from another thread, or never.
    struct Delayed {
        receiver: std::sync::mpsc::Receiver<()>,
    }

    impl CompletionSource for Delayed {
        fn wait(&mut self, timeout: Duration) -> WaitOutcome {
            match self.receiver.recv_timeout(timeout) {
                Ok(()) => WaitOutcome::Completed,
                Err(_) => WaitOutcome::TimedOut,
            }
        }
    }

    fn label() -> SubmissionLabel {
        SubmissionLabel {
            case_id: "case-1".into(),
            module_sha256: "ab".repeat(32),
        }
    }

    #[test]
    fn completion_passes_and_ordinary_failure_is_an_error() {
        let timeout = Duration::from_millis(250);
        let mut completed = Fake::new(WaitOutcome::Completed);
        assert_eq!(
            judge(&mut completed, SubmissionBackend::VulkanGpu, "op", timeout),
            Verdict::Completed
        );
        assert_eq!(completed.waited, Some(timeout));
        let mut failed = Fake::new(WaitOutcome::Failed("out of memory".into()));
        assert_eq!(
            judge(&mut failed, SubmissionBackend::Metal, "op", timeout),
            Verdict::Failed("op failed: out of memory".into())
        );
    }

    #[test]
    fn timeout_is_a_hang_that_names_the_labelled_work() {
        let timeout = Duration::from_millis(20);
        let (sender, receiver) = std::sync::mpsc::channel();
        let verdict = with_submission_label(label(), || {
            judge(
                &mut Delayed { receiver },
                SubmissionBackend::VulkanGpu,
                "candidate dispatch",
                timeout,
            )
        });
        drop(sender);
        let Verdict::Hang(record) = verdict else {
            panic!("expected a hang, got {verdict:?}");
        };
        assert_eq!(record.kind, HangKind::Timeout);
        assert_eq!(record.backend, SubmissionBackend::VulkanGpu);
        assert_eq!(record.operation, "candidate dispatch");
        assert_eq!(record.case_id.as_deref(), Some("case-1"));
        assert_eq!(record.module_sha256, Some("ab".repeat(32)));
        assert_eq!(record.timeout_ms, 20);
        assert_eq!(record.pid, std::process::id());
        assert!(!record.command.is_empty());
        // The label does not outlive its scope.
        assert_eq!(current_label(), None);
    }

    #[test]
    fn an_unresolved_wait_is_a_hang() {
        let mut unresolved = Fake::new(WaitOutcome::Unresolved("fence status unknown".into()));
        let Verdict::Hang(record) = judge(
            &mut unresolved,
            SubmissionBackend::VulkanGpu,
            "op",
            DEFAULT_WAIT_TIMEOUT,
        ) else {
            panic!("expected a hang");
        };
        assert_eq!(record.kind, HangKind::Unresolved);
        assert_eq!(record.detail, "fence status unknown");
    }

    #[test]
    fn completion_from_another_thread_within_the_bound_passes() {
        let (sender, receiver) = std::sync::mpsc::channel();
        let completer = std::thread::spawn(move || sender.send(()).unwrap());
        let verdict = judge(
            &mut Delayed { receiver },
            SubmissionBackend::Metal,
            "op",
            Duration::from_secs(60),
        );
        completer.join().unwrap();
        assert_eq!(verdict, Verdict::Completed);
    }

    #[test]
    fn fault_status_is_a_suspected_fault() {
        let mut faulted = Fake::new(WaitOutcome::SuspectedFault("page fault".into()));
        let Verdict::Fault(record) = judge(
            &mut faulted,
            SubmissionBackend::Metal,
            "Metal compute dispatch",
            DEFAULT_WAIT_TIMEOUT,
        ) else {
            panic!("expected a fault");
        };
        assert_eq!(record.kind, HangKind::SuspectedFault);
        assert_eq!(record.detail, "page fault");
        assert_eq!(record.case_id, None);
    }

    fn scratch_tripwire() -> (crate::ScratchDir, PathBuf) {
        let scratch = crate::ScratchDir::new("gpu-wait").unwrap();
        let path = scratch.path().join("m2v-gpu.hang");
        (scratch, path)
    }

    #[test]
    fn gpu_incidents_write_the_tripwire_and_cpu_incidents_do_not() {
        let mut faulted = Fake::new(WaitOutcome::SuspectedFault("VK_ERROR_DEVICE_LOST".into()));
        for (backend, writes) in [
            (SubmissionBackend::VulkanCpu, false),
            (SubmissionBackend::VulkanGpu, true),
            (SubmissionBackend::Metal, true),
        ] {
            let (_scratch, path) = scratch_tripwire();
            let Verdict::Fault(record) = judge(&mut faulted, backend, "op", DEFAULT_WAIT_TIMEOUT)
            else {
                panic!("expected a fault");
            };
            report_incident(&record, |record| tripwire::write_record(&path, record));
            assert_eq!(path.exists(), writes, "{backend:?}");
            if writes {
                assert_eq!(tripwire::read_record(&path).unwrap(), record);
                assert!(tripwire::refuse_if_present(&path).is_err());
            }
        }
    }

    #[test]
    fn wait_timeout_argument_is_bounded() {
        assert_eq!(parse_wait_timeout("10").unwrap(), DEFAULT_WAIT_TIMEOUT);
        assert_eq!(parse_wait_timeout("600").unwrap(), MAX_WAIT_TIMEOUT);
        for value in ["0", "601", "", "-1", "1.5", "10s"] {
            assert!(parse_wait_timeout(value).is_err(), "{value:?}");
        }
        assert!(set_wait_timeout(Duration::ZERO).is_err());
        assert!(set_wait_timeout(MAX_WAIT_TIMEOUT + Duration::from_millis(1)).is_err());
        assert_eq!(wait_timeout(), DEFAULT_WAIT_TIMEOUT);
    }
}
