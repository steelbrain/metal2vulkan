//! Freed memory that waits for the work in flight (`freed_memory.rs`), in both settings: on a
//! device made to park freed memory through the crate's hook (`Device::park_freed_memory`, as
//! where the driver has `RuntimeDriverDefect::FreedAddressMemoryFailsPendingWork`), a host import
//! released while a batch submitted before it runs keeps its memory, and its deallocator waits,
//! until that batch completed, then goes on a handler thread; elsewhere it goes at once, as
//! before. The counts show which path ran (`WorkCounts::freed_memory_parked`,
//! `freed_memory_waiting`, `host_deallocators_delayed`). The same release waits too for the
//! program's own submission through `CommandQueue::with_vk_queue` pending as it goes, held back
//! by a timeline semaphore of the program's that the program signals once the release returned.
//! And what a block notes as it goes holds it for the batches, the device's own submissions and
//! the program's uses of a queue made before, and for none after.
//!
//! Where a host import goes as its release frees it: its free, then its deallocator, before the
//! release returns, on the releasing thread, as Metal runs the deallocator (an M3 Max under macOS
//! 27.0.1 ran an unused buffer's deallocator inside its final release, on the releasing thread,
//! every time); with neither a deallocator nor a keep-alive, its free before the release returns,
//! as the caller may unmap the pages then; with a keep-alive alone (a runtime extension), its free
//! and the keep-alive's drop on a handler thread, the release returning first.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use crate::freed_memory::Parking;
use crate::{
    BufferContents, BufferDescriptor, CommandBuffer, CommandQueue, Device, DeviceKind,
    DeviceOptions, Error, HostMemory, KeepAlive, ResourceOptions, ShaderCacheOptions,
};
use ash::vk;
use std::alloc::Layout;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread::ThreadId;
use std::time::{Duration, Instant};

/// The longest a command buffer, or a release after it, may take.
const WAIT: Duration = Duration::from_secs(60);

/// The no-copy buffer's bytes: a whole page of any host.
const BYTES: usize = 64 << 10;

/// The fills that keep the batch in flight: [`FILLS`] of [`FILLED`] bytes each.
const FILLS: u32 = 16;
const FILLED: u64 = 64 << 20;

/// A scratch directory for the device's shader cache, removed when dropped.
struct Scratch(PathBuf);

impl Scratch {
    fn new(case: &str) -> Scratch {
        let name = format!("m2v-freed-memory-{case}-{}", std::process::id());
        let path = std::env::temp_dir().join(name);
        std::fs::create_dir_all(&path).unwrap();
        Scratch(path)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Page-aligned host memory of [`BYTES`], freed when dropped: declared first, after the device
/// that used it.
pub(super) struct Host(NonNull<u8>);

impl Host {
    fn layout() -> Layout {
        Layout::from_size_align(BYTES, BYTES).unwrap()
    }

    pub(super) fn new() -> Host {
        // SAFETY: the layout's size is not zero.
        let pointer = unsafe { std::alloc::alloc_zeroed(Host::layout()) };
        Host(NonNull::new(pointer).unwrap())
    }
}

impl Drop for Host {
    fn drop(&mut self) {
        // SAFETY: allocated with this layout in `Host::new`.
        unsafe { std::alloc::dealloc(self.0.as_ptr(), Host::layout()) };
    }
}

/// The device the cfg selects: a CPU device under `m2v_cpu_icd`, else a GPU, its shader cache in
/// `cache`.
fn device(cache: &Path) -> Device {
    let options = DeviceOptions {
        kinds: match cfg!(m2v_cpu_icd) {
            true => vec![DeviceKind::Cpu],
            false => vec![
                DeviceKind::Discrete,
                DeviceKind::Integrated,
                DeviceKind::Virtual,
            ],
        },
        shader_cache: ShaderCacheOptions {
            directory: Some(cache.to_owned()),
            ..ShaderCacheOptions::default()
        },
        ..DeviceOptions::default()
    };
    Device::system_default_with(&options).unwrap()
}

/// The runs of a deallocator: how many, and the thread of the last.
#[derive(Default)]
pub(super) struct Runs {
    pub(super) count: AtomicUsize,
    thread: Mutex<Option<ThreadId>>,
}

/// A no-copy buffer over `host` that no command uses, its deallocator's runs in `runs`; `None`,
/// printed, where the device imports no host memory.
pub(super) fn unused(device: &Device, host: &Host, runs: &Arc<Runs>) -> Option<crate::Buffer> {
    let runs = Arc::clone(runs);
    // SAFETY: the memory outlives the buffer (dropped after the device), and nothing else uses it.
    let made = unsafe {
        device.new_buffer_with_bytes_no_copy(
            HostMemory::new(host.0),
            BYTES as u64,
            ResourceOptions::default(),
            Some(Box::new(move |_: NonNull<u8>, _: u64| {
                *runs.thread.lock().unwrap() = Some(std::thread::current().id());
                runs.count.fetch_add(1, Ordering::SeqCst);
            })),
        )
    };
    match made {
        Ok(buffer) => Some(buffer),
        Err(Error::Unsupported(gap)) => {
            eprintln!("FREED-MEMORY: the device imports no host memory: {gap:?}");
            None
        }
        Err(error) => panic!("new_buffer_with_bytes_no_copy: {error:?}"),
    }
}

/// A command buffer of `queue`'s that fills `target` [`FILLS`] times, committed, once its batch's
/// `vkQueueSubmit` was made: a kernel submission counted (`WorkCounts::queue_submissions`, counted
/// under the parking gate's read lock that the store of the batch's value holds too, so a block's
/// note after it sees the value). Its kernel times are no such sign: the scheduler stores its
/// start as it begins the batch, before replaying and submitting it, and under
/// `CompletionPolicy::Grouped` its end as it holds the batch for a kernel submission with later
/// ones.
fn filling(queue: &CommandQueue, target: &crate::Buffer) -> CommandBuffer {
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    for value in 0..FILLS {
        blit.fill_buffer(target, 0..FILLED, value as u8).unwrap();
    }
    blit.end_encoding().unwrap();
    let before = queue.device().work_counts().queue_submissions;
    command_buffer.commit().unwrap();
    let deadline = Instant::now() + WAIT;
    while queue.device().work_counts().queue_submissions == before {
        assert!(Instant::now() < deadline, "the batch was never submitted");
        std::thread::yield_now();
    }
    command_buffer
}

/// Wait until `done`, failing with `what` past [`WAIT`].
fn until(what: &str, done: impl Fn() -> bool) {
    let deadline = Instant::now() + WAIT;
    while !done() {
        assert!(Instant::now() < deadline, "{what}");
        std::thread::sleep(Duration::from_millis(1));
    }
}

/// Where the device parks freed memory, a host import released while a batch submitted before it
/// runs keeps its memory, its deallocator waiting, until that batch completed; then both go, on a
/// thread of the runtime's, and the wait was announced once. Where the fills completed before the
/// release, which the parking's count says (`WorkCounts::freed_memory_parked`), its memory and
/// its deallocator go inside the release, on the releasing thread, as Metal's do: the release
/// reads the value the queue's waiter published, which comes before the turn that finishes the
/// command buffer, so the command buffer's status says nothing about which path the release took.
#[test]
fn a_release_waits_for_the_batch_in_flight_where_the_device_parks_freed_memory() {
    let scratch = Scratch::new("parked");
    let host = Host::new();
    let device = device(&scratch.0);
    device.park_freed_memory();
    let queue = device.new_command_queue().unwrap();
    let runs = Arc::new(Runs::default());
    let Some(buffer) = unused(&device, &host, &runs) else {
        return;
    };
    let target = device
        .new_buffer(FILLED, ResourceOptions::private())
        .unwrap();
    let command_buffer = filling(&queue, &target);
    let before = device.work_counts();
    let status = command_buffer.status();
    drop(buffer);
    let after = device.work_counts();
    let parked = after.freed_memory_parked - before.freed_memory_parked;
    let delayed = after.host_deallocators_delayed - before.host_deallocators_delayed;
    let ran = runs.count.load(Ordering::SeqCst);
    eprintln!(
        "FREED-MEMORY: at the release ({status:?}): parked {parked}, waiting {}, deallocators \
         delayed {delayed}, run {ran}",
        after.freed_memory_waiting,
    );
    // Whether the release found work in flight is the parking's own answer, the one it acted on,
    // and no command buffer's status: the queue's waiter publishes the value it saw the queue's
    // timeline reach, which the release reads, before the completion turn that finishes the
    // command buffers of that value (`command_buffer/completion.rs`), so a status short of
    // `Completed` is no sign that the release waited, and the fills complete whenever the GPU
    // and a loaded host let them.
    let waited = parked == 1;
    match waited {
        true => {
            assert_eq!(after.freed_memory_waiting, 1);
            assert_eq!(delayed, 1, "the deallocator's wait is announced");
            assert_eq!(
                ran, 0,
                "the deallocator ran while the batch submitted before its release ran"
            );
        }
        false => {
            eprintln!("FREED-MEMORY: the fills completed before the release: none waited");
            assert_eq!(parked, 0, "a release parked one block, not {parked}");
            assert_eq!(
                after.freed_memory_waiting, 0,
                "a block waits although the release freed its memory"
            );
            assert_eq!(
                delayed, 0,
                "a deallocator's wait is announced although the release freed its memory"
            );
            // As Metal's release runs it, on the releasing thread (the module docs).
            assert_eq!(ran, 1, "the deallocator ran inside the release");
            assert_eq!(
                *runs.thread.lock().unwrap(),
                Some(std::thread::current().id()),
                "the deallocator ran on the releasing thread"
            );
        }
    }
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    until("the deallocator never ran", || {
        runs.count.load(Ordering::SeqCst) == 1
    });
    until("the memory never went", || {
        device.work_counts().freed_memory_waiting == 0
    });
    eprintln!(
        "FREED-MEMORY: after completion: waiting {}, run {}",
        device.work_counts().freed_memory_waiting,
        runs.count.load(Ordering::SeqCst)
    );
    // A release after its batch completed runs the deallocator at once, on the program's thread.
    if waited {
        assert_ne!(
            *runs.thread.lock().unwrap(),
            Some(std::thread::current().id()),
            "the deallocator ran on the program's thread after its batch completed"
        );
    }
}

/// The bytes of a private buffer that takes a block of its own, with a buffer of addresses (past
/// half a pooled block, `memory.rs`), and that the recycler parks for no later buffer (past its
/// bytes, `recycle.rs`): its release frees its block.
const OWN_BLOCK: u64 = 128 << 20;

/// The `vkDestroyBuffer` calls the memory's own code made so far (`memory.rs`): its blocks'
/// buffers of addresses (`WorkCounts::object_churn`'s release sites). Read from the counters
/// alone: a release waiting for the gate holds the memory's lock, which `Device::work_counts`
/// takes.
fn addresses_destroyed(device: &Device) -> u64 {
    let (churn, _) = device.shared().counts.churn.snapshot();
    (churn.release_sites.iter())
        .filter(|site| site.command == "vkDestroyBuffer" && site.file.ends_with("memory.rs"))
        .map(|site| site.calls)
        .sum()
}

/// Where the device parks freed memory, a block's buffer of addresses goes with no submission
/// inside its `vkQueueSubmit` (`RuntimeDriverDefect::DestroyedAddressBufferFailsSubmission`):
/// while another thread holds the parking's gate for reading, as a batch's and an own submission's
/// `vkQueueSubmit` do, the release of a buffer whose block then goes destroys no buffer of
/// addresses and waits; once that thread lets go, the release goes on, and the block's buffer of
/// addresses goes. Before, the release destroyed it at once and only then waited for the gate, to
/// note the batches: on MoltenVK, a `vkQueueSubmit` encoding meanwhile had listed it, and failed.
#[test]
fn a_block_s_buffer_of_addresses_goes_with_no_submission_inside_its_vk_queue_submit() {
    let scratch = Scratch::new("gated");
    let device = device(&scratch.0);
    device.park_freed_memory();
    let parking = Arc::clone(device.shared().parking.get().expect("the device parks"));
    let buffer = device
        .new_buffer(OWN_BLOCK, ResourceOptions::private())
        .unwrap();
    let before = addresses_destroyed(&device);
    let submitting = parking.submitting();
    let (released, done) = mpsc::channel();
    let releaser = std::thread::spawn(move || {
        drop(buffer);
        released.send(()).unwrap();
    });
    assert!(
        done.recv_timeout(Duration::from_millis(500)).is_err(),
        "the release went on while a submission was inside the gate"
    );
    assert_eq!(
        addresses_destroyed(&device),
        before,
        "a buffer of addresses went while a submission was inside its vkQueueSubmit"
    );
    drop(submitting);
    done.recv_timeout(WAIT)
        .expect("the release went on once the submission left the gate");
    releaser.join().unwrap();
    assert_eq!(
        addresses_destroyed(&device),
        before + 1,
        "the block's buffer of addresses went once the submission left"
    );
}

/// The rounds of [`blocks_go_while_submissions_that_make_buffers_run`].
const ROUNDS: usize = 24;

/// Where the device parks freed memory, blocks that go while another thread makes buffers and
/// submits work on them end, as does that work, without error: a release takes the memory's lock
/// and then the gate's write lock, and a submission takes the memory's lock (its buffer's making)
/// outside the gate and the gate's read lock inside its queue's lock alone (the lock order in
/// `freed_memory.rs`'s docs). A submission that took the memory's lock inside the gate would
/// wait for a release waiting for it, and neither would end.
#[test]
fn blocks_go_while_submissions_that_make_buffers_run() {
    let scratch = Scratch::new("orders");
    let device = device(&scratch.0);
    device.park_freed_memory();
    let (done, ended) = mpsc::channel();
    let submitter = {
        let device = device.clone();
        let done = done.clone();
        std::thread::spawn(move || {
            let queue = device.new_command_queue().unwrap();
            for round in 0..ROUNDS * 4 {
                let target = device
                    .new_buffer(1 << 20, ResourceOptions::private())
                    .unwrap();
                let command_buffer = queue.command_buffer().unwrap();
                let mut blit = command_buffer.blit_command_encoder().unwrap();
                blit.fill_buffer(&target, 0..1 << 20, round as u8).unwrap();
                blit.end_encoding().unwrap();
                command_buffer.commit().unwrap();
                command_buffer.wait_until_completed_for(WAIT).unwrap();
                assert_eq!(command_buffer.error(), None, "round {round}");
            }
            done.send("submissions").unwrap();
        })
    };
    let releaser = {
        let device = device.clone();
        std::thread::spawn(move || {
            for _ in 0..ROUNDS {
                let own = device
                    .new_buffer(OWN_BLOCK, ResourceOptions::private())
                    .unwrap();
                drop(own);
            }
            done.send("releases").unwrap();
        })
    };
    for _ in 0..2 {
        let what = ended
            .recv_timeout(WAIT)
            .expect("the submissions and releases end");
        eprintln!("FREED-MEMORY: the {what} ended");
    }
    submitter.join().unwrap();
    releaser.join().unwrap();
    assert!(
        addresses_destroyed(&device) >= ROUNDS as u64,
        "each release destroyed its block's buffer of addresses"
    );
}

/// Where the device parks no freed memory, the same release goes at once, on the releasing
/// thread, and nothing waits.
#[test]
fn a_release_goes_at_once_where_the_device_parks_no_freed_memory() {
    let scratch = Scratch::new("at-once");
    let host = Host::new();
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let runs = Arc::new(Runs::default());
    let Some(buffer) = unused(&device, &host, &runs) else {
        return;
    };
    let target = device
        .new_buffer(FILLED, ResourceOptions::private())
        .unwrap();
    let command_buffer = filling(&queue, &target);
    let parks = device
        .properties()
        .driver_defects
        .contains(&metal2vulkan::error::RuntimeDriverDefect::FreedAddressMemoryFailsPendingWork);
    drop(buffer);
    match parks {
        true => eprintln!("FREED-MEMORY: the device's driver has the defect: it parks"),
        false => {
            assert_eq!(
                runs.count.load(Ordering::SeqCst),
                1,
                "the release went at once"
            );
            assert_eq!(
                *runs.thread.lock().unwrap(),
                Some(std::thread::current().id())
            );
            let counts = device.work_counts();
            eprintln!(
                "FREED-MEMORY: at the release: parked {}, waiting {}, run {}",
                counts.freed_memory_parked,
                counts.freed_memory_waiting,
                runs.count.load(Ordering::SeqCst)
            );
            assert_eq!(
                (counts.freed_memory_parked, counts.freed_memory_waiting),
                (0, 0)
            );
        }
    }
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
}

/// A block's note holds it until each queue's timeline reached the value of the last batch
/// submitted there before it, and until every own submission of the device's numbered before it
/// completed; a batch or an own submission made after the note holds it not.
#[test]
fn a_note_holds_a_block_for_the_work_submitted_before_it_alone() {
    let scratch = Scratch::new("noted");
    let device = device(&scratch.0);
    let parking = Parking::new(2);
    let empty = parking.note_destroying(|| ()).1;
    assert!(parking.reached(&empty), "nothing was submitted");
    parking.submitted(1, 7);
    let batch = parking.note_destroying(|| ()).1;
    parking.submitted(1, 8);
    assert!(parking.reached(&empty), "a batch after the note");
    assert!(!parking.reached(&batch), "the batch in flight");
    parking.completed(1, 7);
    assert!(parking.reached(&batch), "the batch completed");
    parking.completed(1, 8);
    let own = parking.own_submission(&device);
    let during = parking.note_destroying(|| ()).1;
    assert!(parking.reached(&batch), "an own submission after the note");
    assert!(!parking.reached(&during), "the own submission in flight");
    let later = parking.own_submission(&device);
    assert!(
        !parking.reached(&during),
        "the own submission still in flight"
    );
    drop(own);
    assert!(
        parking.reached(&during),
        "the own submission completed; the later one is after"
    );
    let last = parking.note_destroying(|| ()).1;
    assert!(
        !parking.reached(&last),
        "the later own submission in flight"
    );
    drop(later);
    assert!(parking.reached(&last), "every own submission completed");
    // The program's uses of a queue, marked on a timeline of their own.
    let raw = device.shared().vk();
    let (_, first) = parking.program_use(1, raw).expect("the timeline of marks");
    let using = parking.note_destroying(|| ()).1;
    assert!(
        parking.reached(&last),
        "a use of the program's after the note"
    );
    assert!(!parking.reached(&using), "the program's use in flight");
    let (_, second) = parking.program_use(1, raw).unwrap();
    parking.program_completed(1, first);
    assert!(
        parking.reached(&using),
        "the program's use completed; the later one is after"
    );
    let after = parking.note_destroying(|| ()).1;
    assert!(!parking.reached(&after), "the later use in flight");
    parking.program_completed(1, second);
    assert!(
        parking.reached(&after),
        "every use of the program's completed"
    );
    // SAFETY: the timeline is the device's, and no submission or wait uses it.
    unsafe { parking.release(raw) };
}

/// Where the device parks freed memory, a host import released while the program's own
/// submission through `with_vk_queue` is pending keeps its memory, its deallocator waiting, until
/// that submission completed: the runtime sees neither what it lists nor when it completes, so
/// the use's mark holds the block. The submission waits on a timeline semaphore of the program's,
/// which the program signals once the release returned, so nothing else of the device's is in
/// flight meanwhile. It names none of the buffer's Vulkan objects: the block's buffer of addresses
/// is destroyed as it goes, which Vulkan forbids under a pending command that names it, and
/// MoltenVK fails a pending submission whose pipelines read device addresses whatever buffers it
/// names (`RuntimeDriverDefect::FreedAddressMemoryFailsPendingWork`).
#[test]
fn a_release_waits_for_the_program_s_own_submission_where_the_device_parks_freed_memory() {
    let scratch = Scratch::new("program");
    let host = Host::new();
    let device = device(&scratch.0);
    device.park_freed_memory();
    let queue = device.new_command_queue().unwrap();
    let runs = Arc::new(Runs::default());
    let Some(buffer) = unused(&device, &host, &runs) else {
        return;
    };
    // SAFETY: the handles are used only below, while the device lives; nothing here destroys one
    // of the runtime's, and the test submits only through `with_vk_queue`.
    let vk_device = unsafe { device.raw() }.device;
    let mut kind = vk::SemaphoreTypeCreateInfo::default()
        .semaphore_type(vk::SemaphoreType::TIMELINE)
        .initial_value(0);
    let info = vk::SemaphoreCreateInfo::default().push_next(&mut kind);
    // SAFETY: the device is live and enables `timelineSemaphore`.
    let gate = unsafe { vk_device.create_semaphore(&info, None) }.unwrap();
    let waits = [gate];
    let values = [1];
    let stages = [vk::PipelineStageFlags::ALL_COMMANDS];
    let mut timeline = vk::TimelineSemaphoreSubmitInfo::default().wait_semaphore_values(&values);
    let submit = vk::SubmitInfo::default()
        .wait_semaphores(&waits)
        .wait_dst_stage_mask(&stages)
        .push_next(&mut timeline);
    // SAFETY: the queue is used only inside `f`, under the runtime's lock; the submission waits
    // for a value of the program's semaphore that the program signals below.
    unsafe {
        queue.with_vk_queue(|vk_queue, _| {
            vk_device
                .queue_submit(vk_queue, &[submit], vk::Fence::null())
                .unwrap()
        })
    };
    let before = device.work_counts();
    drop(buffer);
    let after = device.work_counts();
    let ran = runs.count.load(Ordering::SeqCst);
    eprintln!(
        "FREED-MEMORY: at the release, the program's submission pending: parked {}, waiting {}, \
         deallocators delayed {}, run {ran}",
        after.freed_memory_parked - before.freed_memory_parked,
        after.freed_memory_waiting,
        after.host_deallocators_delayed - before.host_deallocators_delayed,
    );
    let signal = vk::SemaphoreSignalInfo::default().semaphore(gate).value(1);
    // SAFETY: the program's semaphore, signalled above its value, with no signal pending.
    unsafe { vk_device.signal_semaphore(&signal) }.unwrap();
    assert_eq!(
        ran, 0,
        "the deallocator ran while the program's submission before its release was pending"
    );
    assert_eq!(after.freed_memory_parked - before.freed_memory_parked, 1);
    assert_eq!(after.freed_memory_waiting, 1);
    until("the deallocator never ran", || {
        runs.count.load(Ordering::SeqCst) == 1
    });
    until("the memory never went", || {
        device.work_counts().freed_memory_waiting == 0
    });
    assert_ne!(
        *runs.thread.lock().unwrap(),
        Some(std::thread::current().id()),
        "the deallocator ran on the program's thread"
    );
    // SAFETY: the program's semaphore; the submission that waited on it completed (its mark
    // did, which the deallocator waited for).
    unsafe { vk_device.destroy_semaphore(gate, None) };
}

/// A no-copy buffer over `host` that no command uses, whose deallocator records its run in
/// `runs` with the device's live host imports as it ran, once `go` lets it (at most [`WAIT`]);
/// `None`, printed, where the device imports no host memory.
fn waiting(
    device: &Device,
    host: &Host,
    runs: &Arc<Runs>,
    go: mpsc::Receiver<()>,
) -> Option<(crate::Buffer, Arc<Mutex<Option<u64>>>)> {
    let live = Arc::new(Mutex::new(None));
    let (runs, seen, counted) = (Arc::clone(runs), Arc::clone(&live), device.clone());
    // SAFETY: the memory outlives the buffer (dropped after the device), and nothing else uses it.
    let made = unsafe {
        device.new_buffer_with_bytes_no_copy(
            HostMemory::new(host.0),
            BYTES as u64,
            ResourceOptions::default(),
            Some(Box::new(move |_: NonNull<u8>, _: u64| {
                let _ = go.recv_timeout(WAIT);
                *seen.lock().unwrap() = Some(counted.work_counts().host_imports_live);
                *runs.thread.lock().unwrap() = Some(std::thread::current().id());
                runs.count.fetch_add(1, Ordering::SeqCst);
            })),
        )
    };
    match made {
        Ok(buffer) => Some((buffer, live)),
        Err(Error::Unsupported(gap)) => {
            eprintln!("IMPORT-RELEASE: the device imports no host memory: {gap:?}");
            None
        }
        Err(error) => panic!("new_buffer_with_bytes_no_copy: {error:?}"),
    }
}

/// A no-copy buffer's release frees its import and runs its deallocator before it returns, on the
/// releasing thread, as Metal runs it at the release (an M3 Max under macOS 27.0.1 ran an unused
/// buffer's deallocator inside its final release, on the releasing thread, every time, released on
/// the main thread or another); no release is posted. A runtime that ran it later, or on another
/// thread, would break a program valid under Metal's contract.
#[test]
fn a_release_frees_and_runs_its_deallocator_before_it_returns_on_its_thread() {
    let scratch = Scratch::new("exact");
    let host = Host::new();
    let device = device(&scratch.0);
    let runs = Arc::new(Runs::default());
    let (go, wait) = mpsc::channel();
    go.send(()).unwrap();
    let Some((buffer, live_at_run)) = waiting(&device, &host, &runs, wait) else {
        return;
    };
    let before = device.work_counts();
    drop(buffer);
    let after = device.work_counts();
    assert_eq!(runs.count.load(Ordering::SeqCst), 1, "ran at the release");
    assert_eq!(
        *runs.thread.lock().unwrap(),
        Some(std::thread::current().id())
    );
    assert_eq!(
        *live_at_run.lock().unwrap(),
        Some(before.host_imports_live - 1),
        "the deallocator ran before the import was freed"
    );
    assert_eq!(
        after.host_import_releases_posted,
        before.host_import_releases_posted
    );
}

/// A no-copy buffer released with neither a deallocator nor a keep-alive has its import freed in
/// the release: the caller may unmap the pages once the release returns, and
/// `VK_EXT_external_memory_host` requires the imported range to stay valid for the imported
/// memory's whole life. No release is posted, and the import is gone from the live imports as the
/// release returns.
#[test]
fn a_release_with_nothing_waiting_frees_its_import_before_it_returns() {
    let scratch = Scratch::new("nothing-waiting");
    let host = Host::new();
    let device = device(&scratch.0);
    // SAFETY: the memory outlives the buffer (dropped after the device), and nothing else uses it.
    let made = unsafe {
        device.new_buffer_with_bytes_no_copy(
            HostMemory::new(host.0),
            BYTES as u64,
            ResourceOptions::default(),
            None,
        )
    };
    let buffer = match made {
        Ok(buffer) => buffer,
        Err(Error::Unsupported(gap)) => {
            eprintln!("IMPORT-RELEASE: the device imports no host memory: {gap:?}");
            return;
        }
        Err(error) => panic!("new_buffer_with_bytes_no_copy: {error:?}"),
    };
    let before = device.work_counts();
    drop(buffer);
    let after = device.work_counts();
    assert_eq!(
        after.host_import_releases_posted, before.host_import_releases_posted,
        "the release posted its free"
    );
    assert_eq!(
        after.host_imports_live,
        before.host_imports_live - 1,
        "the import outlived the release"
    );
}

/// A keep-alive's drop (a runtime extension) tells the caller the pages may go.
struct Dropped(Arc<Runs>);

impl Drop for Dropped {
    fn drop(&mut self) {
        *self.0.thread.lock().unwrap() = Some(std::thread::current().id());
        self.0.count.fetch_add(1, Ordering::SeqCst);
    }
}

/// A no-copy buffer released with a keep-alive alone (a runtime extension: Metal has none) has
/// its import's free, and the keep-alive's drop after it, posted to a handler thread, the release
/// returning first: no deallocator waits, so Metal's contract sees nothing late.
#[test]
fn a_release_with_a_keep_alive_alone_posts_its_free() {
    let scratch = Scratch::new("keep-alive");
    let host = Host::new();
    let device = device(&scratch.0);
    let runs = Arc::new(Runs::default());
    let keep_alive: KeepAlive = Box::new(Dropped(Arc::clone(&runs)));
    let contents = BufferContents::NoCopy {
        // SAFETY: the memory outlives the buffer (dropped after the device), and nothing else
        // uses it.
        memory: unsafe { HostMemory::new(host.0) },
        deallocator: None,
        keep_alive: Some(keep_alive),
    };
    let made = device.new_buffer_with_descriptor(BufferDescriptor::new(
        BYTES as u64,
        ResourceOptions::default(),
        contents,
    ));
    let buffer = match made {
        Ok(buffer) => buffer,
        Err(Error::Unsupported(gap)) => {
            eprintln!("IMPORT-RELEASE: the device imports no host memory: {gap:?}");
            return;
        }
        Err(error) => panic!("new_buffer_with_descriptor: {error:?}"),
    };
    let before = device.work_counts();
    drop(buffer);
    let after = device.work_counts();
    assert_eq!(
        after.host_import_releases_posted - before.host_import_releases_posted,
        1,
        "the release posted its free"
    );
    until("the keep-alive never dropped", || {
        runs.count.load(Ordering::SeqCst) == 1
    });
    until("the release never finished", || {
        device.work_counts().host_import_releases_pending == 0
    });
    assert_ne!(
        *runs.thread.lock().unwrap(),
        Some(std::thread::current().id()),
        "the keep-alive dropped on the releasing thread"
    );
}
