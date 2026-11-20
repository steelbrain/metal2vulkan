//! The submit thread: each queue's FIFO of command buffers, the holds at event waits, the groups
//! held while a queue's earlier submissions run, the upload that goes ahead of later commits, and
//! the submissions it passes to `vkQueueSubmit2`.
//!
//! Each queue groups on its own. While fewer than [`IN_FLIGHT`] of the queue's submissions are
//! not seen complete, each of its command buffers goes at once, alone. While that many run, the
//! ones committed meanwhile are held in one group of at most [`GROUP`], which goes when it fills,
//! when a completion leaves fewer running (the completion thread wakes the thread as it retires),
//! while the completion thread runs a round of retirements, completed handlers and listeners (it
//! sees no completion until the round ends, so a group is held only while it waits on the GPU),
//! when the host is to wait for one of them ([`Work::send`]), when a command buffer of another
//! queue that waits on a fence one of them updates is to go (`crate::fence`), or at teardown. The
//! driver runs a submission's command buffers as one kernel job, which a job timeout ends as a
//! whole (Linux's amdgpu ends a job past its timeout and resets the ring), so a command buffer
//! joins the group only while the group's expected GPU runs, summed, stay within [`GROUP_RUN`];
//! one whose queue has no run yet, or whose run is past the bound, goes alone, the group held
//! going first. The
//! bound is tested again as the group goes, with the expected run as it then stands (a completion
//! between a join and the send may have raised it): a group past it goes member by member, one
//! `vkQueueSubmit2` each, only the last signalling. At most [`UNSEEN_JOINS`] command buffers join
//! a group while none of the queue's retires, so command buffers that turn long all at once join
//! no more than that many on the estimate of the short ones before them. A segment that
//! signals an event ends its group, which goes at once; one that waits on a semaphore leads a new
//! group, the one held going first. A group is one submission with one signal of the progress
//! semaphore, which the completion thread follows in commit order. A command buffer that takes no
//! submission of its own (one with nothing for the GPU, or one ended with an error) and finds a
//! group held joins it, and the group goes at once: what it retires behind is then submitted, so
//! its completion waits on nothing that only a later completion or wait would send.
//!
//! The upload is cut at commits: the calls made between two commits record into an upload of
//! their own, numbered as the later commit ([`WorkState::current`]), which goes ahead of
//! everything in the first submission whose command buffers' last was committed at or after that
//! number. So a call goes ahead of every command buffer committed after it, and never in a
//! submission whose command buffers were all committed before it, whenever the submit thread
//! takes it: what each submission carries follows the program's calls and commits and the
//! submission's command buffers alone, by the copy-ins' rule (`crate::capture`).
//! A commit only takes its number; the first call after it begins the next upload. The last
//! upload a submission carries ends with the one barrier that makes their writes visible.
//!
//! The uploads not yet submitted are bounded: a call that finds them holding [`UPLOAD_ENTRIES`]
//! calls, or [`UPLOAD_BYTES`] of memory they alone keep, has them submitted and waits for them to
//! complete before it records. The release of a texture or a block one of them holds has them go
//! at once, with no wait ([`Work::release`]), so its memory goes back once they complete, with no
//! later call. A new texture's move to its home layout holds nothing until the
//! submission that takes the upload of its commit, which records the moves of every upload it
//! takes in one barrier where the textures still live. Copies stage their bytes in the device's
//! staging region ([`crate::staging`]), which the work lock guards with the uploads; they go
//! without a wait once their copies' spans take half the region.
//!
//! A command buffer's commit takes the Managed buffers' ranges named since the last capture
//! ([`crate::capture`]): its copy-in, GPU copies from the consumers' blocks into the device
//! copies, waits at that commit's number, and the first submission of a command buffer numbered
//! at or after it takes it, ahead of that command buffer's first segment and outside its waits.

use crate::buffer::{BufferInner, BufferObject, DeviceCopy};
use crate::capture::{self, Changes, CopyIn, Modified};
use crate::command::{self, CbInner, CommandBufferError, Handler, Pending, Segment};
use crate::complete;
use crate::counters::{Counters, Gauge, Metric, Site};
use crate::device::{lock, Core};
use crate::error::Error;
use crate::event::EventInner;
use crate::fill;
use crate::memory::{Block, Internal};
use crate::over::{self, Leans};
use crate::queue::QueueInner;
use crate::record::{self, Reach, Recorder};
use crate::staging::{At, Oldest, Region, Stage, Staging, REGION_LEN};
use crate::texture::{Homing, Image, OverCopy, TextureInner, TextureObject, Zeroing};
use crate::times;
use ash::vk;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError, Weak};

/// The command buffers one submission ends at most. A larger group is fewer submissions, until
/// the group grows past what a completion can release: on the UHD 630, sixteen ran a burst of 160
/// command buffers in half the time no bound did (probe 06860a3848, job m3-max-44105).
pub(crate) const GROUP: usize = 16;

/// The most expected GPU run, in nanoseconds, of the command buffers one group holds, summed: 16
/// ms, the owner's line for an intolerable wait. Each command buffer's
/// expected run is its queue's (`QueueInner::expected_run`), from the GPU times of the queue's
/// command buffers as they retire. Command buffers each within the driver's job timeout may pass
/// it summed in one job.
pub(crate) const GROUP_RUN: u64 = 16_000_000;

/// The most command buffers that join one group while none of their queue's retires with its GPU
/// times: they join on an expected run no completion has updated, so a program whose command
/// buffers turn long all at once puts at most this many of them in one job before the estimate
/// rises. Four: the most GPU time summed over consecutive
/// command buffers of one queue in a replay stream on an RTX 3090 was 385.1 ms over four, within
/// a quarter of amdgpu's default job timeout of 2,000 ms, and 1,536.6 ms over sixteen, past it
/// (job m3-max-49358).
///
/// Its measured price, against sixteen, is paid where command buffers come faster than they
/// complete. On the UHD 630, a burst of 2,000 command buffers of one dispatch each made 344
/// submissions a run against 127 and took 40.2 ms against 19.8. A burst of 160 command buffers of
/// two dispatches made 27 against 10 and took 3.28 ms against 1.85 (job m3-max-49576). On an RTX
/// 3090, sixteen ran such bursts 5.1-6.5% faster (job m3-max-49575). No replay stream pays it:
/// glbench, gb and windrag on a 780M made the same submissions in every frame under both, as none
/// of their groups passed four (jobs m3-max-49698 and m3-max-49721).
pub(crate) const UNSEEN_JOINS: usize = 4;

/// Whether a command buffer whose queue's expected run is `run` (none before one of its command
/// buffers retired with its GPU times) joins a held group whose members' expected runs sum to
/// `held`, `unseen` of them joined since the queue's last retirement: only while the sum with its
/// own stays within [`GROUP_RUN`] and fewer than [`UNSEEN_JOINS`] joined unseen. Else the group
/// goes first and the command buffer leads the next.
pub(crate) fn joins(held: u64, unseen: usize, run: Option<u64>) -> bool {
    run.is_some_and(|run| held.saturating_add(run) <= GROUP_RUN) && unseen < UNSEEN_JOINS
}

/// Whether a group of `members` command buffers goes member by member as it is sent, its queue's
/// expected run being `run` then: where the members' expected runs, summed at that run, pass
/// [`GROUP_RUN`], or the run is not known. One member is one submission either way.
pub(crate) fn apart(members: u64, run: Option<u64>) -> bool {
    members > 1 && run.is_none_or(|run| run.saturating_mul(members) > GROUP_RUN)
}

/// The submissions of a queue not yet seen complete at which its command buffers are held in a
/// group. Three, not two: with two, the GPU idles between a group's end and the next
/// submission's start whenever the completion that sends the held group comes late. Behind three,
/// the frame of five command buffers on the UHD 630 completed in 420-429 us against 895-1137 us,
/// and the second of four long command buffers was seen complete 74.7 ms after its commit against
/// 99.2 ms (76.5 against 96.8 on an RTX 2080, 46.8 against 57.7 on an RX 550), while a burst of
/// 160 kept its time within 0.5% on all three (probe b15b306d2f, jobs m3-max-44423, m3-max-44424
/// and m3-max-44437). On an RTX 3090 the frame's mean commit-to-complete came to 152-165 us
/// against 183-188 us, and the four long command buffers' to 32.6-32.9 ms against 44.3-67.2 ms,
/// the burst within 0.1% (jobs m3-max-44525 and m3-max-44546). Not four: behind four the UHD 630 ran that burst in 2768-2839 us against
/// 1850-1860 us, more of its command buffers going alone, and the RTX 2080's long rows came to
/// 78.1-78.2 ms against 42.8-43.2 ms (jobs m3-max-44442 and m3-max-44443).
pub(crate) const IN_FLIGHT: usize = 3;

/// The calls the uploads not yet submitted record at most: a call past them waits for them to
/// complete.
pub(crate) const UPLOAD_ENTRIES: usize = 4096;

/// The memory the uploads not yet submitted alone keep, at most: their staging buffers, and the
/// textures and blocks released while they held them. A call past it waits for them to complete;
/// one call larger than it goes alone.
pub(crate) const UPLOAD_BYTES: u64 = 64 << 20;

/// The new textures awaiting their home layout at which the dead among them are first dropped.
pub(crate) const HOMES_PRUNE: usize = 64;

/// For the device tests: the nanoseconds this thread waited for the work lock
/// ([`Work::lock`]), in all, since it last took them.
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
pub(crate) mod waits {
    use std::cell::Cell;

    thread_local! {
        static WAITED: Cell<u64> = const { Cell::new(0) };
    }

    pub(crate) fn add(nanos: u64) {
        WAITED.with(|waited| waited.set(waited.get().saturating_add(nanos)));
    }

    /// The nanoseconds waited since the last take, taken.
    pub(crate) fn take() -> u64 {
        WAITED.with(|waited| waited.replace(0))
    }
}

/// The submit side: the FIFOs and the upload, and the submit thread's wake-up.
pub(crate) struct Work {
    state: Mutex<WorkState>,
    changed: Condvar,
    /// A span of the staging region another call held was released, or the device closed or was
    /// lost: a staging call asleep for it looks again ([`Work::stage`]). Signalled only while one
    /// sleeps ([`WorkState::span_sleepers`]).
    spans: Condvar,
    /// A queue holds a group: the completion thread wakes the thread as submissions complete.
    keeping: AtomicBool,
    /// Bytes of textures and blocks released while an upload held them, since the last
    /// submission that left no upload unsent: they count against [`UPLOAD_BYTES`]. A texture's
    /// release adds without the work lock, which its drop may run under.
    released: AtomicU64,
    /// For the device tests: a commit began its count of the wait for the work lock.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) counting: AtomicBool,
    /// For the device tests: a submission began its count of the wait for the runtime's queue
    /// lock.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) asking: AtomicBool,
}

pub(crate) struct WorkState {
    /// The queues with command buffers enqueued, committed, held or in flight.
    fifos: Vec<Fifo>,
    /// The next command buffer's place in the commit order.
    next_seq: u64,
    /// The next FIFO entry's place in its queue's order.
    next_place: u64,
    /// The progress value of the last submission.
    progress: u64,
    /// The progress value of the last submission that took an upload, 0 before the first.
    uploaded: u64,
    /// The upload recording: zero fills, copies and new textures' moves to a home layout, the
    /// calls made since the last commit, or before it where none has been made since
    /// ([`current`](WorkState::current)).
    upload: Option<Upload>,
    /// Uploads a later call or a flush ended the recording of, oldest first, not yet submitted.
    sealed: VecDeque<Upload>,
    /// How many of `sealed`, from the oldest, the next submission takes whatever its command
    /// buffers, alone where no group goes ([`Work::flush`]); 0 for none.
    flush: usize,
    /// The staging region, from the first copy on.
    region: Option<Region>,
    /// The Managed buffers' ranges named since the last commit.
    changes: Changes,
    /// Something changed that may let a held segment or group go.
    dirty: bool,
    /// The submit thread is to end: what is held in groups goes, then what is held at waits ends.
    stop: bool,
    /// Quiesced or torn down: commits and uploads are refused.
    closed: bool,
    lost: bool,
    /// The staging calls asleep for a span of the region another call holds ([`Work::stage`]): a
    /// release signals [`Work::spans`] only while there are any, so with none it makes no call.
    span_sleepers: u32,
}

struct Fifo {
    queue: Arc<QueueInner>,
    entries: VecDeque<Entry>,
    /// The group staged and not yet submitted.
    group: Option<Batch>,
    /// The progress values of the queue's submissions not yet seen complete, in order.
    in_flight: VecDeque<u64>,
    /// A host wait is to come for the command buffer at this place: a group from it or before
    /// goes at once.
    send_through: u64,
    /// The progress value at which the last command buffer it ended with nothing held retires:
    /// the next retires no earlier.
    behind: u64,
    /// Its head's segment the submit thread holds for an event's value neither set nor promised
    /// by a submitted signal, as the command buffer's number and the segment's index, and the
    /// host time the thread, with nothing left to stage, found it so ([`WorkState::mark_holds`]).
    held_since: Option<(u64, usize, u64)>,
}

struct Entry {
    cb: Arc<CbInner>,
    /// Its place in its queue's order.
    place: u64,
    /// None while enqueued and not committed.
    pending: Option<Pending>,
    /// Ended before it was submitted: it retires with this error behind its queue's earlier
    /// work.
    error: Option<CommandBufferError>,
}

/// Commands recorded into one command buffer apart from any command buffer of the guest's (zero
/// fills, texture copies, and new textures' moves to their home layouts), the calls made before
/// one commit and after the one before it: submitted ahead of everything in the first submission
/// whose command buffers' last was committed at or after them, which keeps what they use until it
/// retires.
pub(crate) struct Upload {
    pub(crate) recorder: Recorder,
    command: vk::CommandBuffer,
    /// The commit number its calls come before ([`WorkState::current`]).
    number: u64,
    held: Vec<Held>,
    /// How many of `held` it held at the last release of an object it held
    /// ([`Work::release`]): a release of one held from there on is the first since.
    met: usize,
    /// The calls recorded, and one for the home layouts recorded as it is submitted.
    entries: usize,
    /// The bytes of the staging buffers of its own it holds.
    staging: u64,
    /// The staging region's spans its copies use, and their bytes.
    spans: Vec<u64>,
    span_bytes: u64,
    /// Where a surface waits for the moves this upload records ([`Work::ready_point`]): set to
    /// the progress value of the submission that takes it, 0 until then.
    ready: Option<Arc<AtomicU64>>,
    /// The textures made while it recorded, not held: the submission that takes it moves each
    /// that still lives to its home layout ([`Work::home`]).
    homes: Vec<Weak<Image>>,
    /// The length of `homes` at which the dead are dropped from it.
    prune_at: usize,
    /// The buffer ranges its zero fills wrote since its last barrier ([`Work::fill`]), at most
    /// one a call ([`UPLOAD_ENTRIES`]).
    filled: Vec<Filled>,
    /// Where it recorded a barrier between its fills: the ranges its fills wrote before the
    /// first, which a fill of an upload before it in the same submission may meet
    /// ([`order_fills`]).
    leading: Option<Vec<Filled>>,
}

/// A zero fill's buffer and its range of bytes.
type Filled = (vk::Buffer, u64, u64);

/// What an upload keeps until it retires.
pub(crate) enum Held {
    Block(Arc<Block>),
    /// A texture's image the upload moves to its home layout or copies bytes to or from.
    Texture(Arc<Image>),
    /// The staging buffer of such a copy.
    Staging(Arc<Staging>),
    /// A Managed buffer's device copy the upload copies into.
    Copy(Arc<DeviceCopy>),
}

impl Held {
    /// Whether `self` and `other` are the same object.
    fn is(&self, other: &Held) -> bool {
        match (self, other) {
            (Held::Block(a), Held::Block(b)) => Arc::ptr_eq(a, b),
            (Held::Texture(a), Held::Texture(b)) => Arc::ptr_eq(a, b),
            (Held::Staging(a), Held::Staging(b)) => Arc::ptr_eq(a, b),
            (Held::Copy(a), Held::Copy(b)) => Arc::ptr_eq(a, b),
            _ => false,
        }
    }
}

impl Upload {
    /// A new upload for the calls before the commit numbered `number`, its command buffer
    /// recording.
    ///
    /// Err: [`Error::Vulkan`] where a command pool or buffer cannot be made.
    fn begin(core: &Core, number: u64) -> Result<Upload, Error> {
        let device = &core.device;
        let mut recorder = core.recorders.take(device)?;
        match recorder.next(device) {
            Ok(command) => Ok(Upload {
                recorder,
                command,
                number,
                held: Vec::new(),
                met: 0,
                entries: 0,
                staging: 0,
                spans: Vec::new(),
                span_bytes: 0,
                ready: None,
                homes: Vec::new(),
                prune_at: HOMES_PRUNE,
                filled: Vec::new(),
                leading: None,
            }),
            Err(error) => {
                core.recorders.give_back(core, recorder);
                Err(error)
            }
        }
    }

    /// Records `homes`, images' moves from undefined to their home layouts
    /// ([`Image::homing`]): in one barrier, with the acquires of the buffers' ranges under images
    /// over a dma-buf's import (each such image's move counted in
    /// [`Metrics::external_acquires`](crate::Metrics::external_acquires)), and where any is
    /// zeroed, its zeroing after that barrier and its move home in a second, each counted in
    /// `core`'s counters; the upload holds each image.
    fn home(&self, core: &Core, homes: &[Homing]) {
        if homes.is_empty() {
            return;
        }
        let device = &core.device;
        let barrier = |moves: &[vk::ImageMemoryBarrier2<'static>],
                       buffers: &[vk::BufferMemoryBarrier2<'static>]| {
            let dependency = vk::DependencyInfo::default()
                .image_memory_barriers(moves)
                .buffer_memory_barriers(buffers);
            // SAFETY: the upload's command buffer is recording, under the work lock; each image
            // is the device's and held by the upload, and holds the import whose buffer an
            // acquire names, a dma-buf's, whose foreign family is enabled (`check_over`).
            unsafe { device.cmd_pipeline_barrier2(self.command, &dependency) };
            core.counters.count(Metric::Barriers);
        };
        let firsts: Vec<_> = homes.iter().map(|home| home.first).collect();
        let acquires: Vec<_> = homes.iter().filter_map(|home| home.buffer).collect();
        barrier(firsts.as_slice(), acquires.as_slice());
        for _ in &acquires {
            core.counters.count(Metric::ExternalAcquires);
        }
        let zeroings: Vec<&Zeroing> = homes.iter().filter_map(|home| home.zero.as_ref()).collect();
        if zeroings.is_empty() {
            return;
        }
        // Made before the upload holds an image whose zeroing copies from it (`Work::home`).
        let zeros = fill::made(core);
        for zeroing in &zeroings {
            // SAFETY: as above, outside any render pass; the first barrier moved the image to
            // the transfer destination layout.
            unsafe { zeroing.record(device, self.command, zeros) };
            core.counters.texture_cleared(zeroing.bytes);
        }
        let moves: Vec<_> = zeroings.iter().map(|zeroing| zeroing.home).collect();
        barrier(moves.as_slice(), &[]);
        core.counters.count(Metric::TextureClearBarriers);
    }
}

/// One queue's group: what one `vkQueueSubmit2` submits of it.
struct Batch {
    /// The place of the first command buffer with a segment in it.
    first: u64,
    segments: Vec<Staged>,
    /// Command buffers whose last segment is in it, and those that ended behind them, in the
    /// queue's order.
    finished: Vec<Finished>,
    /// The command buffers whose last segment is in it.
    count: usize,
    /// It holds segments of its queue's head, whose last is not yet staged.
    partial: bool,
    /// The expected GPU runs of its command buffers, summed, in nanoseconds ([`GROUP_RUN`]);
    /// `u64::MAX` where one's was not known.
    run: u64,
    /// Where each command buffer that leads in it begins among `segments`, the copy-ins it takes
    /// first, and its commit number, in the same order.
    starts: Vec<usize>,
    leads: Vec<u64>,
    /// The copy-ins it took, which it keeps until it retires (`crate::capture`).
    copy_ins: Vec<CopyIn>,
    /// Its queue's completed command buffers when the last of its joins was counted, and the
    /// command buffers that joined since one of them completed ([`UNSEEN_JOINS`]).
    seen: u64,
    unseen: usize,
    /// It goes member by member ([`GROUP_RUN`], tested again as it goes).
    apart: bool,
    /// Events it signals that the completion thread is not yet watching.
    watch: Vec<Arc<EventInner>>,
    /// What its segments wrote of buffers under textures over a buffer in an image of their own
    /// (`crate::over`).
    leans: Leans,
    /// The copies into those textures' images its upload makes first: one for each texture a
    /// segment leans on the leading copy for, and each texture over a Managed buffer's device
    /// copy a command buffer of it keeps that owes one (`crate::over::ManagedOvers`).
    overs: Vec<OverCopy>,
    /// The copies back owed first by textures over a Managed buffer's device copy whose ranges
    /// meet those, for writes already submitted, with their images, which the upload keeps.
    backs: Vec<over::Back>,
    /// The Managed buffers its segments wrote whose textures over their device copies owe their
    /// copies in once it is submitted.
    written: Vec<Arc<BufferInner>>,
    /// The textures over a Managed buffer's device copy its segments wrote, one for each segment
    /// that wrote one, whose copies back its submission settles.
    wrote: Vec<Arc<TextureInner>>,
}

/// One part of a submit info: a segment's or an upload's commands and its signals, and for a
/// submission's first, the wait for a sparse bind (`submit`).
#[derive(Default)]
pub(crate) struct Staged {
    pub(crate) waits: Vec<vk::SemaphoreSubmitInfo<'static>>,
    pub(crate) command: Option<vk::CommandBufferSubmitInfo<'static>>,
    pub(crate) signals: Vec<vk::SemaphoreSubmitInfo<'static>>,
}

/// One `VkSubmitInfo2` of a submission: the waits of its first segment, the commands of each of
/// its segments, and the signals of its last.
pub(crate) struct Info<'a> {
    pub(crate) waits: &'a [vk::SemaphoreSubmitInfo<'static>],
    pub(crate) commands: Vec<vk::CommandBufferSubmitInfo<'static>>,
    pub(crate) signals: &'a [vk::SemaphoreSubmitInfo<'static>],
}

/// The submit infos of a submission's segments, in order: a segment that waits on nothing joins
/// the info before it where that one signals nothing, so only a signal, or the wait for a sparse
/// bind, starts a new one. The GPU runs the command buffers of one info in order, as it runs those of consecutive
/// infos, and a driver may pay for each info: the UHD 630's paid about 160 us apiece (probe
/// 06860a3848, job m3-max-43935).
pub(crate) fn infos(segments: &[Staged]) -> Vec<Info<'_>> {
    let mut infos: Vec<Info<'_>> = Vec::with_capacity(1);
    for staged in segments {
        match infos.last_mut() {
            Some(last) if staged.waits.is_empty() && last.signals.is_empty() => {
                last.commands.extend(staged.command);
                last.signals = &staged.signals;
            }
            _ => infos.push(Info {
                waits: &staged.waits,
                commands: staged.command.into_iter().collect(),
                signals: &staged.signals,
            }),
        }
    }
    infos
}

/// A command buffer of a submission: with an error where it ended before it was submitted.
pub(crate) type Finished = (Arc<CbInner>, Pending, Option<CommandBufferError>);

/// A command buffer that retires with no submission of its own: with how it ended, or with no
/// error where it had nothing to submit.
pub(crate) type Ended = Finished;

/// A command buffer that joined a group, with its scheduled handlers.
pub(crate) type Joined = (Arc<CbInner>, Vec<Handler>);

/// What staging the next ready segment did.
enum Step {
    /// Nothing is ready.
    Idle,
    /// Something was staged or ended; no group goes yet.
    Staged,
    /// The group of the FIFO at this index goes now.
    Goes(usize),
}

impl Work {
    pub(crate) fn new() -> Work {
        Work {
            state: Mutex::new(WorkState {
                fifos: Vec::new(),
                next_seq: 0,
                next_place: 0,
                progress: 0,
                uploaded: 0,
                upload: None,
                sealed: VecDeque::new(),
                flush: 0,
                region: None,
                changes: Changes::default(),
                dirty: false,
                stop: false,
                closed: false,
                lost: false,
                span_sleepers: 0,
            }),
            changed: Condvar::new(),
            spans: Condvar::new(),
            keeping: AtomicBool::new(false),
            released: AtomicU64::new(0),
            #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
            counting: AtomicBool::new(false),
            #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
            asking: AtomicBool::new(false),
        }
    }

    pub(crate) fn lock(&self) -> MutexGuard<'_, WorkState> {
        #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
        let began = crate::times::host_time();
        let state = lock(&self.state);
        #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
        waits::add(crate::times::host_time().saturating_sub(began));
        state
    }

    /// Wakes the submit thread after a change [`WorkState`] recorded.
    pub(crate) fn wake(&self) {
        self.changed.notify_one();
    }

    /// A host value rose: held segments may go.
    pub(crate) fn poke(&self) {
        self.lock().dirty = true;
        self.wake();
    }

    /// Submissions completed: where a queue holds a group, it may go.
    pub(crate) fn completed(&self) {
        if self.keeping.load(Ordering::SeqCst) {
            self.poke();
        }
    }

    /// The host is to wait for `cb`: where it is committed and not yet submitted, the group it is
    /// in goes at once, or, where it is not yet staged, the group held ahead of it and the one it
    /// joins. Nothing for a command buffer submitted, ended or not committed. It takes the lock
    /// briefly and waits for no GPU work.
    pub(crate) fn send(&self, cb: &Arc<CbInner>) {
        let mut state = self.lock();
        let Some(fifo) = (state.fifos.iter_mut()).find(|fifo| fifo.queue.is(&cb.queue)) else {
            return;
        };
        let held = (fifo.group.as_ref()).and_then(|group| {
            (group.finished.iter())
                .any(|(joined, _, error)| error.is_none() && Arc::ptr_eq(joined, cb))
                .then_some(group.first)
        });
        let place = held.or_else(|| {
            (fifo.entries.iter())
                .find(|entry| {
                    Arc::ptr_eq(&entry.cb, cb) && entry.pending.is_some() && entry.error.is_none()
                })
                .map(|entry| entry.place)
        });
        let Some(place) = place.filter(|&place| place > fifo.send_through) else {
            return;
        };
        fifo.send_through = place;
        state.dirty = true;
        drop(state);
        self.wake();
    }

    /// Records into the upload of the next commit ([`WorkState::current`]) with `record`, which is
    /// given the device and the upload's command buffer, recording, under the work lock:
    /// commands that go ahead of every later commit, on every queue, in the first submission
    /// whose command buffers' last was committed after the call. The upload keeps each of `held`
    /// until it retires, and `stage`, the staging of a copy [`stage`](Self::stage) gave: its span
    /// of the region until the submission completes, or its own staging buffer. A texture of
    /// `held` not yet in its home layout is moved there first.
    ///
    /// Where the uploads not yet submitted are full ([`UPLOAD_ENTRIES`], [`UPLOAD_BYTES`]),
    /// they are submitted and the call waits for them to complete first, without the lock. Where
    /// their spans of the staging region reach half the region, they are submitted, and the call
    /// does not wait. Where the call fails, `stage`'s span is given up.
    ///
    /// Err: [`Error::Quiesced`]; [`Error::Vulkan`] where a command pool or buffer cannot be made,
    /// and with `VK_ERROR_DEVICE_LOST` where the device is lost while the call waits.
    pub(crate) fn upload<const N: usize>(
        &self,
        core: &Core,
        stage: Option<&Stage>,
        held: [Held; N],
        record: impl FnOnce(&ash::Device, vk::CommandBuffer),
    ) -> Result<(), Error> {
        let recorded = self.record_upload(core, stage, held, |device, upload| {
            record(device, upload.command)
        });
        if recorded.is_err() {
            if let Some(Stage::Region { number, .. }) = stage {
                self.give_up(*number);
            }
        }
        recorded
    }

    /// The work lock, once the uploads not yet submitted have room for a call of `staging` bytes
    /// of staging of its own: where they are full ([`WorkState::full`]), they are submitted and
    /// the call waits for them to complete, without the lock.
    ///
    /// Err: [`Error::Quiesced`]; [`Error::Vulkan`] with `VK_ERROR_DEVICE_LOST` where the device is
    /// lost while the call waits.
    fn room(&self, core: &Core, staging: u64) -> Result<MutexGuard<'_, WorkState>, Error> {
        let mut state = self.lock();
        loop {
            if state.closed {
                return Err(Error::Quiesced);
            }
            let released = self.released.load(Ordering::Relaxed);
            if !state.full(staging, released) {
                return Ok(state);
            }
            let value = state.flush();
            drop(state);
            self.wake();
            core.counters.count(Metric::UploadWaits);
            core.done.wait(core, value)?;
            state = self.lock();
        }
    }

    /// `didModifyRange:` of a Managed buffer with a device copy: the consumer's bytes
    /// `start..end` of `buffer` changed. The range is only named, merged with the buffer's others
    /// where they overlap or touch, holding the buffer weakly; the next commit of a command buffer
    /// with commands copies it in ([`capture`](Self::capture)) where the buffer still lives. It
    /// allocates nothing.
    ///
    /// Err: [`Error::Quiesced`].
    pub(crate) fn modify(
        &self,
        buffer: &Arc<BufferInner>,
        start: u64,
        end: u64,
    ) -> Result<(), Error> {
        let mut state = self.lock();
        if state.closed {
            return Err(Error::Quiesced);
        }
        state.changes.add(buffer, start, end);
        Ok(())
    }

    /// At a commit of a command buffer with commands, on the committing thread, under the work
    /// lock `state`, which it keeps: takes every Managed buffer's range named since the last
    /// capture (`crate::capture`) into one copy-in, GPU copies from the consumers' blocks into the
    /// device copies, recorded into a command buffer of its own ([`capture::copy_in`]); none
    /// where every buffer named is gone. It waits at the committing command buffer's number, the
    /// next in the commit order ([`WorkState::seq`]), for the first command buffer submitted that
    /// is numbered at or after it ([`capture::Captured`]). A commit that fails after its capture
    /// leaves it there, for the next.
    ///
    /// Err: [`Error::Quiesced`]; what making the copy-in's command buffer returns. The ranges are
    /// then named again for the next commit.
    pub(crate) fn capture<'a>(
        &'a self,
        core: &Core,
        mut state: MutexGuard<'a, WorkState>,
    ) -> Result<MutexGuard<'a, WorkState>, Error> {
        let named = state.changes.take();
        if named.is_empty() {
            return Ok(state);
        }
        if state.closed {
            return Err(self.unwind(state, &named, Error::Quiesced));
        }
        let copy_in = match capture::copy_in(core, &named) {
            Ok(copy_in) => copy_in,
            Err(error) => return Err(self.unwind(state, &named, error)),
        };
        core.counters
            .captured(named.iter().map(Modified::bytes).sum());
        let number = state.next_seq + 1;
        state.changes.captured.push(number, copy_in);
        Ok(state)
    }

    /// A capture that cannot finish: its ranges `named` are named again for the next commit (none
    /// once the device is closed: its submit thread forgot the ranges at its end), and `error` is
    /// the commit's.
    fn unwind(
        &self,
        mut state: MutexGuard<'_, WorkState>,
        named: &[Modified],
        error: Error,
    ) -> Error {
        let again = if state.closed { &[] } else { named };
        state.changes.give_up(again);
        error
    }

    /// [`upload`](Self::upload) up to the span's give-up on a failure, `record` given the upload
    /// itself.
    fn record_upload<const N: usize>(
        &self,
        core: &Core,
        stage: Option<&Stage>,
        held: [Held; N],
        record: impl FnOnce(&ash::Device, &mut Upload),
    ) -> Result<(), Error> {
        let staging = match stage {
            Some(Stage::Own(staging)) => staging.len,
            _ => 0,
        };
        let mut state = self.room(core, staging)?;
        let device = &core.device;
        let upload = state.current(core)?;
        let homes: Vec<Homing> = (held.iter())
            .filter_map(|item| match item {
                Held::Texture(image) => image.homing(),
                _ => None,
            })
            .collect();
        upload.home(core, &homes);
        record(device, upload);
        upload.entries += 1;
        upload.staging += staging;
        for item in held {
            if !upload.held.iter().any(|kept| kept.is(&item)) {
                upload.held.push(item);
            }
        }
        let mut half = false;
        match stage {
            Some(Stage::Region { number, len, .. }) => {
                upload.spans.push(*number);
                upload.span_bytes += len;
                if let Some(region) = &mut state.region {
                    region.ring.set(*number, At::Recorded);
                }
                self.release_spans(&state);
                let spans: u64 = state.unsent().map(|upload| upload.span_bytes).sum();
                half = spans >= REGION_LEN / 2;
            }
            Some(Stage::Own(staging)) => upload.held.push(Held::Staging(Arc::clone(staging))),
            None => {}
        }
        if half {
            state.flush();
            drop(state);
            self.wake();
        }
        Ok(())
    }

    /// Staging for a copy of `len` bytes: a span of the device's staging region, which the
    /// block-maker thread makes as its first work, so a copy before then waits for that make
    /// ([`Pool::await_made`](crate::memory::Pool::await_made)), and where that thread made none,
    /// the copy makes it ([`make_staging_region`](Self::make_staging_region)); a staging buffer
    /// of its own for a copy larger than the region. Where the region has
    /// no room, the call waits, without the lock, for what frees its oldest span: the submission
    /// that carries it, after submitting the uploads not yet submitted where it is in one. Where
    /// another call holds that span, still writing its bytes or reading them, the call sleeps on
    /// [`spans`](Work::spans) until that call releases it ([`release_spans`](Self::release_spans)),
    /// counted in [`Metrics::staging_sleeps`](crate::Metrics::staging_sleeps). `read` keeps the
    /// span until [`unread`](Self::unread), for a copy the host reads once it completed. The
    /// caller passes the stage to [`upload`](Self::upload).
    ///
    /// Err: [`Error::Quiesced`]; what making the region or the staging buffer can return;
    /// [`Error::Vulkan`] with `VK_ERROR_DEVICE_LOST` where the device is lost while the call
    /// waits.
    pub(crate) fn stage(&self, core: &Arc<Core>, len: u64, read: bool) -> Result<Stage, Error> {
        if len > REGION_LEN {
            if self.lock().closed {
                return Err(Error::Quiesced);
            }
            return Ok(Stage::Own(Arc::new(Staging::new(core, len)?)));
        }
        let mut state = self.lock();
        if state.closed {
            return Err(Error::Quiesced);
        }
        if state.region.is_none() {
            drop(state);
            core.pool.await_made(Internal::StagingRegion);
            self.make_staging_region(core)?;
            state = self.lock();
        }
        // The progress value known complete: what the completion thread has seen, or what this
        // call waited for.
        let mut completed = core.done.completed();
        loop {
            if state.closed {
                return Err(Error::Quiesced);
            }
            // Made above, and taken only at teardown.
            let Some(region) = &mut state.region else {
                return Err(Error::Quiesced);
            };
            region.ring.reclaim(completed.max(core.done.completed()));
            if let Some((number, offset)) = region.ring.take(len, read) {
                return Ok(Stage::Region {
                    number,
                    len,
                    buffer: region.buffer,
                    offset,
                    host: region.host(offset),
                });
            }
            let Some(oldest) = region.ring.oldest() else {
                unreachable!("a ring with no live span takes any span of at most its length");
            };
            let value = match oldest {
                Oldest::Submitted(value) => value,
                Oldest::Pending => state.flush(),
                // Another call finishes writing or reading its span: its release wakes this one.
                Oldest::Busy => {
                    if state.lost {
                        return Err(Error::Vulkan {
                            call: "vkWaitSemaphores",
                            result: vk::Result::ERROR_DEVICE_LOST,
                        });
                    }
                    core.counters.count(Metric::StagingSleeps);
                    state.span_sleepers += 1;
                    state = (self.spans.wait(state)).unwrap_or_else(PoisonError::into_inner);
                    state.span_sleepers -= 1;
                    continue;
                }
            };
            drop(state);
            self.wake();
            core.counters.count(Metric::UploadWaits);
            core.done.wait(core, value)?;
            completed = completed.max(value);
            state = self.lock();
        }
    }

    /// The host no longer reads span `number` of the staging region.
    pub(crate) fn unread(&self, number: u64) {
        let mut state = self.lock();
        if let Some(region) = &mut state.region {
            region.ring.unread(number);
        }
        self.release_spans(&state);
    }

    /// Gives up span `number` of the staging region, with no copy recorded.
    fn give_up(&self, number: u64) {
        let mut state = self.lock();
        if let Some(region) = &mut state.region {
            region.ring.set(number, At::Done);
            region.ring.unread(number);
        }
        self.release_spans(&state);
    }

    /// Wakes the staging calls asleep for a span another call holds ([`stage`](Self::stage)), where
    /// any sleep: after a change, under the work lock `state`, that can release the region's
    /// oldest span, or end the wait. Three changes release a span a call sleeps for: its copy
    /// recorded, its give-up, and its read's end. A span's later moves, to its submission or given
    /// up with an upload not yet submitted, are of recorded spans, for which no call sleeps: one
    /// finding its oldest span recorded submits the uploads and waits for the GPU instead.
    fn release_spans(&self, state: &WorkState) {
        if state.span_sleepers > 0 {
            self.spans.notify_all();
        }
    }

    /// Makes the staging region where none is made yet: on the block-maker thread, its first work
    /// (`crate::memory::Internal::StagingRegion`), or on a copy's own thread where that thread
    /// made none, counted in
    /// [`Metrics::staging_region_allocations`](crate::Metrics::staging_region_allocations).
    ///
    /// Err: what [`Region::new`] returns.
    pub(crate) fn make_staging_region(&self, core: &Core) -> Result<(), Error> {
        if self.lock().region.is_some() {
            return Ok(());
        }
        let made = Region::new(core, REGION_LEN, Site::StagingRegion)?;
        let mut state = self.lock();
        if state.region.is_none() && !state.closed {
            state.region = Some(made);
            core.counters.rise_by(Gauge::StagingBytes, REGION_LEN);
        } else {
            drop(state);
            // SAFETY: another call made the region first, or the device closed; nothing used this
            // one.
            unsafe { made.free(core) };
        }
        Ok(())
    }

    /// At teardown: the staging region, for the core to free.
    pub(crate) fn take_region(&mut self) -> Option<Region> {
        let state = self.state.get_mut().unwrap_or_else(PoisonError::into_inner);
        state.region.take()
    }

    /// A new texture's `image`, or a drawable's acquired again: the upload of the next commit
    /// ([`WorkState::current`]) keeps it, and the submission that takes that upload moves it to
    /// its home layout where it still lives, holding it from then; until then nothing holds it.
    /// Where its move home copies from the zero source, the zero source is made first
    /// ([`fill::zeros`]), off the work lock.
    ///
    /// Err: [`Error::Quiesced`]; [`Error::Vulkan`] where a command pool or buffer cannot be made;
    /// what [`fill::zeros`] returns.
    pub(crate) fn home(&self, core: &Core, image: &Arc<Image>) -> Result<(), Error> {
        if image.copies_zeros() {
            fill::zeros(core)?;
        }
        let mut state = self.lock();
        if state.closed {
            return Err(Error::Quiesced);
        }
        let upload = state.current(core)?;
        image.owe_home();
        if upload.homes.len() >= upload.prune_at {
            upload.homes.retain(|home| home.strong_count() > 0);
            upload.prune_at = (upload.homes.len() * 2).max(HOMES_PRUNE);
        }
        upload.homes.push(Arc::downgrade(image));
        Ok(())
    }

    /// A texture whose image an upload holds is released: its `bytes` count against the
    /// [`UPLOAD_BYTES`] of the uploads not yet submitted. An upload already submitted frees it
    /// soon; counting it anyway can only submit the others sooner.
    pub(crate) fn released(&self, bytes: u64) {
        self.released.fetch_add(bytes, Ordering::Relaxed);
    }

    /// The consumer released `block`: where an upload not yet submitted holds it, the release is
    /// a held one ([`release`](Self::release)), and its length counts against their
    /// [`UPLOAD_BYTES`].
    pub(crate) fn release_block(&self, core: &Core, block: &Arc<Block>) {
        let len = block.memory.len;
        if self.release(core, &Held::Block(Arc::clone(block)), len) {
            self.released(len);
        }
    }

    /// The last holder of `held` outside the uploads let go of it, `bytes` of a texture's image
    /// or of a block: whether an upload not yet submitted holds it. Such a release is counted
    /// ([`Metrics::held_releases`](crate::Metrics::held_releases), with its bytes), as one that
    /// starts a send where no such release came since those uploads came to hold it
    /// ([`Metrics::held_release_starts`](crate::Metrics::held_release_starts)), and every upload
    /// not yet submitted goes at once, alone where no group goes, with no wait
    /// ([`flush`](Self::flush)), so what they hold is freed once they complete, with no later
    /// call.
    pub(crate) fn release(&self, core: &Core, held: &Held, bytes: u64) -> bool {
        let mut state = self.lock();
        let mut holds = false;
        let mut starts = false;
        for upload in state.unsent() {
            if let Some(at) = (upload.held.iter()).position(|kept| kept.is(held)) {
                holds = true;
                starts |= at >= upload.met;
            }
        }
        if !holds {
            return false;
        }
        core.counters.held_release(bytes, starts);
        for upload in state.unsent_mut() {
            upload.met = upload.held.len();
        }
        state.flush();
        drop(state);
        self.wake();
        true
    }

    /// For the device tests: the calls of the uploads not yet submitted, the bytes they alone
    /// keep, and the new textures awaiting their home layout.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn pending(&self) -> (usize, u64, usize) {
        let state = self.lock();
        let released = self.released.load(Ordering::Relaxed);
        let (entries, staging) = state.unsent_calls();
        let homes = state.unsent().map(|upload| upload.homes.len()).sum();
        (entries, staging + released, homes)
    }

    /// For the device tests: the staging region's live spans, oldest first; none before the
    /// region is made.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn staging_spans(&self) -> Vec<(u64, u64, crate::staging::At)> {
        let state = self.lock();
        (state.region.as_ref()).map_or_else(Vec::new, |region| region.ring.live())
    }

    /// Records the zero fill of `offset..end` of `block` into the upload ([`fill`](Self::fill)).
    ///
    /// Err: as [`fill`](Self::fill).
    pub(crate) fn zero(
        &self,
        core: &Core,
        block: &Arc<Block>,
        offset: u64,
        end: u64,
    ) -> Result<(), Error> {
        let (buffer, base) = block.memory.range(0);
        let held = Held::Block(Arc::clone(block));
        self.fill(core, held, buffer, base + offset, base + end)
    }

    /// Records the zero fill of `start..end` of `buffer` into the upload, which keeps `held`, what
    /// holds the buffer, until it retires. The upload's commands are in no order of their own:
    /// a fill that meets bytes an earlier fill of the upload wrote goes after a barrier, so the
    /// two do not write the same bytes at once (a write-after-write hazard). Fills of other bytes
    /// take none. One that meets a fill of an earlier upload of the same submission goes after
    /// a barrier the submission records between them ([`order_fills`]).
    ///
    /// Err: as [`upload`](Self::upload); what [`fill::zeros_for`] returns.
    pub(crate) fn fill(
        &self,
        core: &Core,
        held: Held,
        buffer: vk::Buffer,
        start: u64,
        end: u64,
    ) -> Result<(), Error> {
        let zeros = fill::zeros_for(core, start, end)?;
        self.record_upload(core, None, [held], |device, upload| {
            let meets = |&(filled, from, to): &Filled| filled == buffer && from < end && start < to;
            if upload.filled.iter().any(meets) {
                record::barrier(device, upload.command, Reach::Commands);
                core.counters.count(Metric::Barriers);
                if upload.leading.is_none() {
                    upload.leading = Some(upload.filled.clone());
                }
                upload.filled.clear();
            }
            upload.filled.push((buffer, start, end));
            // SAFETY: the upload's command buffer is recording, under the lock; `held` holds the
            // buffer, which has every usage, `start..end` lies in it, and `zeros` is what
            // `zeros_for` answered for it.
            unsafe { fill::record(device, upload.command, zeros, buffer, start, end) }
        })
    }

    /// Has every upload not yet submitted go at once, alone where no group goes: the progress
    /// value at which everything recorded into uploads so far has completed and is visible to the
    /// host, for [`Done::wait`](crate::complete::Done::wait).
    pub(crate) fn flush(&self) -> u64 {
        let value = self.lock().flush();
        self.wake();
        value
    }

    /// The point at which every move to a home layout recorded so far has completed, with no
    /// submission of its own: the last upload's not yet submitted, whose submission sets it, and
    /// takes every one before it; where none is, the last submission that took one.
    /// [`ready_value`](Self::ready_value) reads it.
    pub(crate) fn ready_point(&self) -> Arc<AtomicU64> {
        let mut guard = self.lock();
        let state = &mut *guard;
        let uploaded = state.uploaded;
        match state.upload.as_mut().or(state.sealed.back_mut()) {
            Some(upload) => Arc::clone(
                upload
                    .ready
                    .get_or_insert_with(|| Arc::new(AtomicU64::new(0))),
            ),
            None => Arc::new(AtomicU64::new(uploaded)),
        }
    }

    /// The progress value at which `point` ([`ready_point`](Self::ready_point)) is reached: the
    /// value of the submission that took its upload, or where that upload is not yet submitted,
    /// the next submission's, which this call has take it at once ([`flush`](Self::flush)).
    pub(crate) fn ready_value(&self, point: &AtomicU64) -> u64 {
        let mut state = self.lock();
        // A submission takes the upload under this lock, setting the point as it does.
        match point.load(Ordering::Relaxed) {
            0 => {
                let value = state.flush();
                drop(state);
                self.wake();
                value
            }
            value => value,
        }
    }

    /// Ends `queue`'s command buffers not yet in a submission or a group with `NotRun`. A group
    /// it holds still goes: its command buffers are scheduled.
    pub(crate) fn abandon(&self, core: &Core, queue: &QueueInner) {
        let (ended, held) = {
            let mut guard = self.lock();
            let state = &mut *guard;
            let (ended, held) = match state.fifos.iter_mut().find(|fifo| fifo.queue.is(queue)) {
                Some(fifo) => {
                    let entries: Vec<Entry> = fifo.entries.drain(..).collect();
                    let mut ended = Vec::new();
                    for entry in entries {
                        fifo.end(entry, CommandBufferError::NotRun, &mut ended);
                    }
                    (ended, fifo.group.is_some())
                }
                None => (Vec::new(), false),
            };
            // The ended ones joined the group held, which goes at once.
            state.dirty |= held;
            (ended, held)
        };
        if held {
            self.wake();
        }
        core.done.end_all(core, ended);
    }

    /// Closes the device to commits and uploads, and tells the submit thread to end.
    pub(crate) fn stop(&self) {
        let mut state = self.lock();
        state.closed = true;
        state.stop = true;
        state.dirty = true;
        self.wake();
        self.release_spans(&state);
    }

    /// At a loss: the submit thread ends everything it holds.
    pub(crate) fn lose(&self) {
        let mut state = self.lock();
        state.lost = true;
        state.dirty = true;
        self.wake();
        self.release_spans(&state);
    }

    /// The progress value of the last submission.
    pub(crate) fn progress(&self) -> u64 {
        self.lock().progress
    }
}

impl WorkState {
    pub(crate) fn closed(&self) -> bool {
        self.closed
    }

    pub(crate) fn lost(&self) -> bool {
        self.lost
    }

    /// The next place in the commit order.
    pub(crate) fn seq(&mut self) -> u64 {
        self.next_seq += 1;
        self.next_seq
    }

    /// Has every upload not yet submitted go at once, alone where no group goes: the progress
    /// value at which everything recorded into uploads so far has completed. The upload
    /// recording ends here; a later call begins the next. The caller wakes the thread.
    fn flush(&mut self) -> u64 {
        self.sealed.extend(self.upload.take());
        if self.sealed.is_empty() {
            // Every upload so far went in a submission already made.
            return self.progress;
        }
        self.flush = self.sealed.len();
        self.dirty = true;
        // The next submission takes them.
        self.progress + 1
    }

    /// The upload a call made now records into: the one of the next commit, numbered as that
    /// commit will be. Where the upload recording holds calls made before the last commit, the
    /// call ends its recording ([`sealed`](WorkState::sealed)): it goes in the first submission
    /// whose command buffers' last was committed at or after its number, and the call begins the
    /// next. So what each upload holds follows the program's calls and commits alone, whenever
    /// the submit thread takes it. A commit only takes its number ([`seq`](Self::seq)).
    ///
    /// Err: [`Error::Vulkan`] where a command pool or buffer cannot be made.
    fn current(&mut self, core: &Core) -> Result<&mut Upload, Error> {
        let number = self.next_seq + 1;
        let earlier = self.upload.take_if(|upload| upload.number < number);
        self.sealed.extend(earlier);
        let upload = match self.upload.take() {
            Some(upload) => upload,
            None => Upload::begin(core, number)?,
        };
        Ok(self.upload.insert(upload))
    }

    /// The uploads not yet submitted, oldest first.
    fn unsent(&self) -> impl Iterator<Item = &Upload> {
        self.sealed.iter().chain(&self.upload)
    }

    /// [`unsent`](Self::unsent), to change.
    fn unsent_mut(&mut self) -> impl Iterator<Item = &mut Upload> {
        self.sealed.iter_mut().chain(&mut self.upload)
    }

    /// The calls the uploads not yet submitted hold, and the bytes of the staging buffers of their
    /// own they hold.
    fn unsent_calls(&self) -> (usize, u64) {
        (self.unsent()).fold((0, 0), |(entries, staging), upload| {
            (entries + upload.entries, staging + upload.staging)
        })
    }

    /// Whether a call of `staging` bytes waits for the uploads not yet submitted: they hold a
    /// call, and they are at [`UPLOAD_ENTRIES`], or the call would take what they alone keep
    /// (with `released`) past [`UPLOAD_BYTES`].
    fn full(&self, staging: u64, released: u64) -> bool {
        let (entries, kept) = self.unsent_calls();
        entries > 0
            && (entries >= UPLOAD_ENTRIES
                || kept.saturating_add(released).saturating_add(staging) > UPLOAD_BYTES)
    }

    /// Takes the uploads the submission of progress `value` carries, oldest first: each whose
    /// calls came before the commit numbered `through`, the last of its command buffers' (0 for
    /// none), and those a flush sends ([`flush`](Self::flush)). Their spans of the staging region
    /// are free once `value` completes. The moves to their home layouts of the textures they keep
    /// that still live are recorded in the last, in one barrier, which holds each texture from
    /// then. `released` starts again once no upload is left unsent.
    fn take_uploads(
        &mut self,
        core: &Core,
        released: &AtomicU64,
        value: u64,
        through: u64,
    ) -> Vec<Upload> {
        let within = (self.sealed.iter())
            .take_while(|upload| upload.number <= through)
            .count();
        let count = within.max(self.flush).min(self.sealed.len());
        self.flush = 0;
        let mut taken: Vec<Upload> = self.sealed.drain(..count).collect();
        if self.sealed.is_empty() {
            taken.extend(self.upload.take_if(|upload| upload.number <= through));
        }
        if taken.is_empty() {
            return taken;
        }
        if self.sealed.is_empty() && self.upload.is_none() {
            released.store(0, Ordering::Relaxed);
        }
        self.uploaded = value;
        let mut homes = Vec::new();
        let mut held = Vec::new();
        for upload in &mut taken {
            if let Some(ready) = &upload.ready {
                ready.store(value, Ordering::Relaxed);
            }
            if let Some(region) = &mut self.region {
                for &number in &upload.spans {
                    region.ring.set(number, At::Submitted(value));
                }
            }
            for image in upload.homes.drain(..).filter_map(|home| home.upgrade()) {
                // One an upload copied to or from was moved before its copy, and is held.
                if let Some(homing) = image.homing() {
                    homes.push(homing);
                    held.push(Held::Texture(image));
                }
            }
        }
        if let (false, Some(last)) = (homes.is_empty(), taken.last_mut()) {
            last.home(core, &homes);
            last.entries += 1;
            last.held.extend(held);
        }
        taken
    }

    /// Takes the uploads not yet submitted where they will never be: their spans of the staging
    /// region are given up, and the textures awaiting their home layout need none.
    fn drop_uploads(&mut self) -> Vec<Upload> {
        let mut dropped: Vec<Upload> = self.sealed.drain(..).collect();
        dropped.extend(self.upload.take());
        self.flush = 0;
        if let Some(region) = &mut self.region {
            for upload in &dropped {
                for &number in &upload.spans {
                    region.ring.set(number, At::Done);
                }
            }
        }
        dropped
    }

    fn fifo(&mut self, queue: &Arc<QueueInner>) -> &mut Fifo {
        let index = match self.fifos.iter().position(|fifo| fifo.queue.is(queue)) {
            Some(index) => index,
            None => {
                self.fifos.push(Fifo {
                    queue: Arc::clone(queue),
                    entries: VecDeque::new(),
                    group: None,
                    in_flight: VecDeque::new(),
                    send_through: 0,
                    behind: 0,
                    held_since: None,
                });
                self.fifos.len() - 1
            }
        };
        &mut self.fifos[index]
    }

    /// Takes `cb`'s place at its queue's end.
    fn push(&mut self, cb: &Arc<CbInner>) -> &mut Entry {
        self.next_place += 1;
        let place = self.next_place;
        let entries = &mut self.fifo(&cb.queue).entries;
        entries.push_back(Entry {
            cb: Arc::clone(cb),
            place,
            pending: None,
            error: None,
        });
        entries.back_mut().expect("an entry just pushed")
    }

    /// Takes `cb`'s place in its queue's FIFO.
    pub(crate) fn enqueue(&mut self, cb: &Arc<CbInner>) {
        self.push(cb);
    }

    /// Hands `cb` to the submit thread: at its place where it was enqueued, else at its queue's
    /// end. With `error` (it ended while enqueued, and its place went), at its queue's end, to
    /// retire with that error behind the work held or in flight before it.
    pub(crate) fn commit(
        &mut self,
        cb: &Arc<CbInner>,
        pending: Pending,
        error: Option<CommandBufferError>,
    ) {
        self.dirty = true;
        let enqueued = (self.fifo(&cb.queue).entries.iter_mut())
            .find(|entry| Arc::ptr_eq(&entry.cb, cb) && entry.pending.is_none());
        let entry = match enqueued {
            Some(entry) => entry,
            None => self.push(cb),
        };
        entry.pending = Some(pending);
        entry.error = error;
    }

    /// With nothing left to stage at `now`: each queue's head whose next segment waits for an
    /// event's value neither set nor promised by a submitted signal is held from `now`, where it
    /// was not held already ([`Metrics::event_holds`](crate::Metrics::event_holds)).
    fn mark_holds(&mut self, now: u64) {
        for fifo in &mut self.fifos {
            let Some(Entry {
                pending: Some(pending),
                error: None,
                ..
            }) = fifo.entries.front()
            else {
                continue;
            };
            let at = (pending.seq, pending.next);
            let marked = (fifo.held_since).is_some_and(|(seq, next, _)| (seq, next) == at);
            if !marked && !ready(&pending.gpu.segments[pending.next]) {
                fifo.held_since = Some((at.0, at.1, now));
            }
        }
    }

    /// The FIFO whose head goes next: of the committed heads that are ended or whose next segment
    /// is ready, the earliest committed.
    fn next_ready(&self) -> Option<usize> {
        let mut best: Option<(usize, u64)> = None;
        for (index, fifo) in self.fifos.iter().enumerate() {
            let Some(Entry {
                pending: Some(pending),
                error,
                ..
            }) = fifo.entries.front()
            else {
                continue;
            };
            if best.is_some_and(|(_, seq)| seq < pending.seq) {
                continue;
            }
            if error.is_some() || ready(&pending.gpu.segments[pending.next]) {
                best = Some((index, pending.seq));
            }
        }
        best.map(|(index, _)| index)
    }

    /// The FIFO, other than the one at `index`, whose held group holds a command buffer, finished
    /// or its queue's head, that updates a fence the head at `index` waits on.
    fn updater_held(&self, index: usize) -> Option<usize> {
        let waiter = &self.fifos[index].head()?.gpu.fences;
        if !waiter.waits() {
            return None;
        }
        (0..self.fifos.len()).find(|&other| {
            let fifo = &self.fifos[other];
            other != index
                && fifo.group.as_ref().is_some_and(|group| {
                    (group.finished.iter())
                        .any(|(_, pending, _)| pending.gpu.fences.updates_for(waiter))
                        || (group.partial
                            && (fifo.head())
                                .is_some_and(|pending| pending.gpu.fences.updates_for(waiter)))
                })
        })
    }

    /// Forgets the submissions seen complete by `completed`, and the FIFOs left with nothing.
    fn settle(&mut self, completed: u64) {
        self.fifos.retain_mut(|fifo| {
            while (fifo.in_flight.front()).is_some_and(|&value| value <= completed) {
                fifo.in_flight.pop_front();
            }
            !(fifo.entries.is_empty() && fifo.group.is_none() && fifo.in_flight.is_empty())
        });
    }

    /// Stages the next ready segment, in commit order, into its queue's group, or ends the next
    /// ended command buffer behind its queue's work. A command buffer whose last segment joins a
    /// group is scheduled at `now`, its scheduled handlers going to `joined`, but one that
    /// presents a drawable, which `submit` schedules once its submission has passed
    /// `vkQueueSubmit2`; one ended with nothing held goes to `ended`. With `busy` (the completion
    /// thread runs a round of retirements and handlers, `Done::busy`) a group goes as each
    /// command buffer joins it, as no completion would send it until the round ends.
    fn step(
        &mut self,
        now: u64,
        busy: bool,
        counters: &Counters,
        joined: &mut Vec<Joined>,
        ended: &mut Vec<Ended>,
    ) -> Step {
        let Some(index) = self.next_ready() else {
            return Step::Idle;
        };
        let stop = self.stop;
        let fifo = &mut self.fifos[index];
        let Some(entry) = fifo.entries.front_mut() else {
            return Step::Idle;
        };
        if let Some(error) = entry.error {
            if let Some(entry) = fifo.entries.pop_front() {
                fifo.end(entry, error, ended);
            }
            return Step::Staged;
        }
        // A command buffer that waits on a fence goes after each command buffer of another queue
        // held in a group that updates it, which Metal would have submitted at its commit
        // (`crate::fence`): that group goes first, as a host wait for it would send it.
        if let Some(other) = self.updater_held(index) {
            let fifo = &mut self.fifos[other];
            if let Some(group) = &fifo.group {
                fifo.send_through = fifo.send_through.max(group.first);
            }
            return Step::Goes(other);
        }
        let fifo = &mut self.fifos[index];
        let Some(entry) = fifo.entries.front_mut() else {
            return Step::Idle;
        };
        let Some(pending) = entry.pending.as_mut() else {
            return Step::Idle;
        };
        // A command buffer with nothing for the GPU takes no submission: it is scheduled now, and
        // completes with its queue's work before it.
        if pending.gpu.is_empty() {
            if let Some(Entry {
                cb,
                pending: Some(mut pending),
                ..
            }) = fifo.entries.pop_front()
            {
                let scheduled = std::mem::take(&mut pending.handlers.scheduled);
                cb.scheduled(now, !scheduled.is_empty());
                if !scheduled.is_empty() {
                    joined.push((Arc::clone(&cb), scheduled));
                }
                fifo.behind_held(cb, pending, None, ended);
            }
            return Step::Staged;
        }
        let segment = &pending.gpu.segments[pending.next];
        // A segment that leans on the leading copy for a buffer an earlier segment of the group
        // wrote leads its group, so its copy follows the write (`crate::over`).
        if (fifo.group.as_ref()).is_some_and(|group| group.leans.conflicts(&segment.overs)) {
            return Step::Goes(index);
        }
        // A command buffer joins the group held only while the group's expected GPU runs, summed,
        // stay within the bound: the group goes first where its own run is not known or would take
        // the sum past it.
        let run = fifo.queue.expected_run();
        let retired = fifo.queue.retired();
        let leads = pending.next == 0;
        if let (true, Some(group)) = (leads, &fifo.group) {
            let unseen = if group.seen == retired {
                group.unseen
            } else {
                0
            };
            if !joins(group.run, unseen, run) {
                return Step::Goes(index);
            }
        }
        let place = entry.place;
        let group = fifo.group.get_or_insert_with(|| Batch {
            first: place,
            segments: Vec::new(),
            finished: Vec::new(),
            count: 0,
            partial: false,
            run: 0,
            starts: Vec::new(),
            leads: Vec::new(),
            copy_ins: Vec::new(),
            seen: 0,
            unseen: 0,
            apart: false,
            watch: Vec::new(),
            leans: Leans::default(),
            overs: Vec::new(),
            backs: Vec::new(),
            written: Vec::new(),
            wrote: Vec::new(),
        });
        if leads {
            group.run = group.run.saturating_add(run.unwrap_or(u64::MAX));
            group.starts.push(group.segments.len());
            group.leads.push(pending.seq);
            if group.seen != retired {
                group.seen = retired;
                group.unseen = 0;
            }
            group.unseen += 1;
        }
        group.leans.join(&segment.overs);
        for buffer in &segment.managed_written {
            if !group.written.iter().any(|kept| kept.id() == buffer.id()) {
                group.written.push(buffer.clone());
            }
        }
        group.wrote.extend(segment.managed_wrote.iter().cloned());
        // Each texture over a Managed buffer's device copy the command buffer keeps that an event
        // owes its copy in (`crate::over::ManagedOvers`).
        if pending.gpu.managed {
            for texture in pending.gpu.kept.textures.iter() {
                let Some(buffer) = texture.managed_over() else {
                    continue;
                };
                let taken = buffer
                    .over_entries()
                    .take_into(texture.id(), &mut group.backs);
                group.overs.extend(taken);
            }
        }
        for copy in segment.overs.leaning() {
            if !group.overs.iter().any(|kept| kept.image == copy.image) {
                group.overs.push(copy);
            }
        }
        let staged = stage(segment, &mut group.watch);
        let at = (pending.seq, pending.next);
        if let Some((.., since)) =
            (fifo.held_since).take_if(|&mut (seq, next, _)| (seq, next) == at)
        {
            counters.event_hold(now.saturating_sub(since));
        }
        // An event's signal goes at once: nothing waits on it behind a held group. So does a
        // presenting command buffer's signal of the consumer's semaphore, which its present waits on
        // once the command buffer is scheduled.
        let signals = !staged.signals.is_empty();
        group.segments.push(staged);
        pending.next += 1;
        if pending.next < pending.gpu.segments.len() {
            group.partial = true;
            return if signals {
                Step::Goes(index)
            } else {
                Step::Staged
            };
        }
        let Some(Entry {
            cb,
            pending: Some(mut pending),
            ..
        }) = fifo.entries.pop_front()
        else {
            return Step::Staged;
        };
        if !pending.gpu.presents() {
            let scheduled = std::mem::take(&mut pending.handlers.scheduled);
            cb.scheduled(now, !scheduled.is_empty());
            if !scheduled.is_empty() {
                joined.push((Arc::clone(&cb), scheduled));
            }
        }
        group.finished.push((cb, pending, None));
        group.partial = false;
        group.count += 1;
        let goes = signals
            || stop
            || busy
            || group.count >= GROUP
            || fifo.in_flight.len() < IN_FLIGHT
            || fifo.send_through >= group.first;
        match goes {
            true => Step::Goes(index),
            false => Step::Staged,
        }
    }

    /// The FIFO at `index`'s group, given the next progress value and the uploads of the commits
    /// up to its last command buffer's ([`take_uploads`](Self::take_uploads)): the submission it
    /// makes.
    fn close(
        &mut self,
        index: usize,
        core: &Core,
        released: &AtomicU64,
    ) -> Option<(u64, Vec<Upload>, Batch)> {
        let fifo = &mut self.fifos[index];
        let mut batch = fifo.group.take()?;
        // The bound again, with the expected run as it now stands: a completion since a join may
        // have raised it.
        batch.apart = apart(batch.starts.len() as u64, fifo.queue.expected_run());
        // The copy-ins captured at or before the number of a command buffer that goes first in
        // it, each ahead of the first such one (`crate::capture`).
        batch.take(self.changes.captured.take(&batch.leads));
        self.progress += 1;
        let value = self.progress;
        fifo.in_flight.push_back(value);
        if batch.partial {
            if let Some(Entry {
                pending: Some(pending),
                ..
            }) = fifo.entries.front_mut()
            {
                pending.value = value;
                pending.gpu.fences.submitted(value);
            }
        }
        for (_, pending, error) in &mut batch.finished {
            pending.value = value;
            if error.is_none() {
                pending.gpu.fences.submitted(value);
            }
        }
        let through = batch.leads.last().copied().unwrap_or(0);
        let uploads = self.take_uploads(core, released, value, through);
        Some((value, uploads, batch))
    }

    /// The first FIFO whose group goes at the end of a pass: unless its queue has [`IN_FLIGHT`]
    /// submissions not seen complete, no host wait is to come for it, and the completion thread
    /// waits for the GPU (not `busy` with a round of retirements and handlers).
    fn unheld(&self, busy: bool) -> Option<usize> {
        self.fifos.iter().position(|fifo| {
            fifo.group.as_ref().is_some_and(|group| {
                self.stop
                    || busy
                    || fifo.in_flight.len() < IN_FLIGHT
                    || fifo.send_through >= group.first
            })
        })
    }
}

impl Fifo {
    /// Its head's pending part, where its head is committed.
    fn head(&self) -> Option<&Pending> {
        (self.entries.front()).and_then(|entry| entry.pending.as_ref())
    }

    /// Ends `entry` with `error` (its own, where it has one): behind the group held, which it
    /// joins to retire with; else once the queue's submissions in flight and the command buffers
    /// it ended before complete, at once where none is. One not committed ends at its commit.
    fn end(&mut self, entry: Entry, error: CommandBufferError, ended: &mut Vec<Ended>) {
        let Entry {
            cb,
            pending,
            error: own,
            ..
        } = entry;
        let error = own.unwrap_or(error);
        let Some(pending) = pending else {
            cb.end_uncommitted(error);
            return;
        };
        if let Some(group) = &mut self.group {
            // The head whose segments the group holds is ended now, if it was the head.
            group.partial = false;
        }
        self.behind_held(cb, pending, Some(error), ended);
    }

    /// Retires a committed command buffer with no submission of its own: behind the group held,
    /// which it joins to retire with and which goes at once; else once the queue's submissions in
    /// flight and the command buffers it ended before complete, at once where none is.
    fn behind_held(
        &mut self,
        cb: Arc<CbInner>,
        mut pending: Pending,
        error: Option<CommandBufferError>,
        ended: &mut Vec<Ended>,
    ) {
        match &mut self.group {
            Some(group) => {
                // Sent as a host wait for it would send it (`Work::send`): its completion then
                // waits only on submitted work.
                self.send_through = self.send_through.max(group.first);
                group.finished.push((cb, pending, error));
            }
            None => {
                // Behind its queue's work, and, where it waits on a fence, behind the last
                // submission that updates it, as Metal's fence wait holds a command buffer with no
                // work too (`crate::fence`).
                let in_flight = self.in_flight.back().copied().unwrap_or(0);
                let waited = pending.gpu.fences.waited_through();
                pending.value = pending.value.max(in_flight).max(self.behind).max(waited);
                self.behind = pending.value;
                ended.push((cb, pending, error));
            }
        }
    }
}

/// Whether every wait of `segment` is met by the host's value or by a submitted signal.
fn ready(segment: &Segment) -> bool {
    segment.waits.iter().all(|(event, value)| {
        let state = lock(&event.state);
        state.floor >= *value || state.submitted >= *value
    })
}

/// A ready segment's submit info. Its waits for events' values submit nothing: a ready segment's
/// every wait is met by the host's value or by a signal already submitted on the device's one
/// `VkQueue`, earlier in its submission order, and the barrier ending each segment with commands
/// (`record::end`) orders every later command after the signaller's, as a semaphore wait would. A
/// signal not above the highest value submitted or set is left out, so the semaphore only rises.
/// The consumer's binary semaphores of a presenting command buffer's last segment are signalled
/// with it.
fn stage(segment: &Segment, watch: &mut Vec<Arc<EventInner>>) -> Staged {
    let mut staged = Staged {
        command: segment
            .command
            .map(|command| vk::CommandBufferSubmitInfo::default().command_buffer(command)),
        ..Staged::default()
    };
    for (event, value) in &segment.signals {
        let mut state = lock(&event.state);
        if *value <= state.submitted.max(state.floor) {
            continue;
        }
        state.submitted = *value;
        state.pending.push_back(*value);
        if !state.watched {
            state.watched = true;
            watch.push(Arc::clone(event));
        }
        staged.signals.push(
            vk::SemaphoreSubmitInfo::default()
                .semaphore(event.semaphore)
                .value(*value)
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS),
        );
    }
    for &ready in &segment.binary {
        staged.signals.push(
            vk::SemaphoreSubmitInfo::default()
                .semaphore(ready)
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS),
        );
    }
    staged
}

/// Ends FIFO entries: a committed one now, one not yet committed at its commit; `error` answers
/// for each without its own, with its pending part where it is committed.
fn end(
    entries: VecDeque<Entry>,
    error: impl Fn(Option<&Pending>) -> CommandBufferError,
) -> Vec<Ended> {
    let mut ended = Vec::new();
    for Entry {
        cb,
        pending,
        error: own,
        ..
    } in entries
    {
        let error = own.unwrap_or_else(|| error(pending.as_ref()));
        match pending {
            Some(pending) => ended.push((cb, pending, Some(error))),
            None => cb.end_uncommitted(error),
        }
    }
    ended
}

/// The submit thread, until it is stopped.
pub(crate) fn run(core: Arc<Core>) {
    let work = &core.work;
    let mut state = work.lock();
    loop {
        if state.lost {
            let submitted = state.progress;
            let error = |pending: Option<&Pending>| CommandBufferError::DeviceLost {
                in_flight: pending.is_some_and(|pending| (1..=submitted).contains(&pending.value)),
            };
            let mut ended = Vec::new();
            for fifo in state.fifos.drain(..) {
                // A group held never reached the GPU: its command buffers end as in flight only
                // where an earlier segment of theirs did.
                for (cb, pending, own) in fifo.group.into_iter().flat_map(|group| group.finished) {
                    let error = own.unwrap_or_else(|| error(Some(&pending)));
                    ended.push((cb, pending, Some(error)));
                }
                ended.extend(end(fifo.entries, error));
            }
            // Nothing more runs: the textures awaiting their home layout need none.
            let uploads = state.drop_uploads();
            drop(state);
            for upload in uploads {
                upload.retire(&core);
            }
            core.done.end_all(&core, ended);
            state = work.lock();
        } else {
            state = pass(&core, state);
        }
        if state.stop {
            let ended: Vec<Ended> = (state.fifos.drain(..))
                .flat_map(|fifo| end(fifo.entries, |_| CommandBufferError::NotRun))
                .collect();
            let uploads = state.drop_uploads();
            // The copy-ins not taken hold device copies, which hold the core.
            let captured = state.changes.clear();
            drop(state);
            for upload in uploads {
                upload.retire(&core);
            }
            for copy_in in captured {
                copy_in.retire(&core);
            }
            core.done.end_all(&core, ended);
            // The last command buffers this thread ends went to the completion thread first, so
            // it retires them before it ends.
            core.done.stop(&core);
            return;
        }
        while !state.dirty {
            state = work
                .changed
                .wait(state)
                .unwrap_or_else(PoisonError::into_inner);
        }
        state.dirty = false;
    }
}

/// One pass of the submit thread: stages what is ready in commit order, submits each group that
/// goes as it closes, then each group its queue no longer holds, then an upload to flush alone.
fn pass<'a>(core: &'a Core, mut state: MutexGuard<'a, WorkState>) -> MutexGuard<'a, WorkState> {
    let work = &core.work;
    // Marked before the completed value and the completion thread's busy mark are read: a
    // completion the read misses, or a round of the completion thread's begun after its read,
    // sees the mark and wakes the thread.
    work.keeping.store(true, Ordering::SeqCst);
    state.settle(core.done.completed());
    let mut joined: Vec<Joined> = Vec::new();
    let mut ended: Vec<Ended> = Vec::new();
    let mut sent = false;
    loop {
        let busy = core.done.busy();
        let now = times::host_time();
        let index = match state.step(now, busy, &core.counters, &mut joined, &mut ended) {
            Step::Idle => match state.unheld(busy) {
                Some(index) => index,
                None => {
                    state.mark_holds(now);
                    break;
                }
            },
            Step::Staged => continue,
            Step::Goes(index) => index,
        };
        let Some((value, uploads, batch)) = state.close(index, core, &work.released) else {
            continue;
        };
        drop(state);
        // Scheduled handlers run here, with no lock held, before the submission they are in.
        scheduled(&mut joined);
        core.done.end_all(core, std::mem::take(&mut ended));
        submit(core, value, uploads, batch);
        sent = true;
        state = work.lock();
    }
    if state.flush > 0 && !sent {
        let value = state.progress + 1;
        let uploads = state.take_uploads(core, &work.released, value, 0);
        if !uploads.is_empty() {
            state.progress = value;
            drop(state);
            submit(core, value, uploads, Batch::empty());
            state = work.lock();
        }
    }
    if state.fifos.iter().all(|fifo| fifo.group.is_none()) {
        work.keeping.store(false, Ordering::SeqCst);
    }
    if !joined.is_empty() || !ended.is_empty() {
        drop(state);
        scheduled(&mut joined);
        core.done.end_all(core, ended);
        state = work.lock();
    }
    state
}

impl Batch {
    /// Puts each copy-in of `taken` ahead of the command buffer that takes it (its index in
    /// `leads`), as a part of its own with no waits ([`put_ahead`]). The batch keeps them until it
    /// retires.
    fn take(&mut self, taken: Vec<(usize, CopyIn)>) {
        let part = |copy_in: &CopyIn| Staged {
            command: Some(vk::CommandBufferSubmitInfo::default().command_buffer(copy_in.command)),
            ..Staged::default()
        };
        let kept = put_ahead(&mut self.segments, &mut self.starts, taken, part);
        self.copy_ins.extend(kept);
    }

    /// A submission of no command buffer's: an upload alone.
    fn empty() -> Batch {
        Batch {
            first: 0,
            segments: Vec::new(),
            finished: Vec::new(),
            count: 0,
            partial: false,
            run: 0,
            starts: Vec::new(),
            leads: Vec::new(),
            copy_ins: Vec::new(),
            seen: 0,
            unseen: 0,
            apart: false,
            watch: Vec::new(),
            leans: Leans::default(),
            overs: Vec::new(),
            backs: Vec::new(),
            written: Vec::new(),
            wrote: Vec::new(),
        }
    }
}

/// Puts each of `taken`, with the index in `starts` of the command buffer of a group that takes it,
/// where that command buffer begins among the group's `segments`, as the segment `part` makes of
/// it, after those taken before it: behind every segment of the command buffers before it in the
/// group, and ahead of the taker's first, which then begins with them. Returns the taken, in their
/// order.
pub(crate) fn put_ahead<S, C>(
    segments: &mut Vec<S>,
    starts: &mut [usize],
    taken: Vec<(usize, C)>,
    part: impl Fn(&C) -> S,
) -> Vec<C> {
    let mut kept = Vec::with_capacity(taken.len());
    for (lead, item) in taken.into_iter().rev() {
        segments.insert(starts[lead], part(&item));
        for start in &mut starts[lead + 1..] {
            *start += 1;
        }
        kept.push(item);
    }
    kept.reverse();
    kept
}

/// Orders the zero fills of the uploads one submission carries, oldest first: where a fill an
/// upload records before any barrier of its own meets bytes a fill of an earlier one wrote since
/// the last barrier, a barrier ends the upload before it, so the two do not write the same bytes
/// at once ([`Work::fill`]), counted in `core`'s counters.
fn order_fills(core: &Core, uploads: &[Upload]) {
    if uploads.len() < 2 {
        return;
    }
    let mut open: Vec<Filled> = Vec::new();
    for (index, upload) in uploads.iter().enumerate() {
        let first = upload.leading.as_ref().unwrap_or(&upload.filled);
        let meets = |&(buffer, start, end): &Filled| {
            (open.iter()).any(|&(other, from, to)| other == buffer && from < end && start < to)
        };
        if let (Some(before), true) = (index.checked_sub(1), first.iter().any(meets)) {
            record::barrier(&core.device, uploads[before].command, Reach::Commands);
            core.counters.count(Metric::Barriers);
            open.clear();
        }
        if upload.leading.is_some() {
            open.clear();
        }
        open.extend_from_slice(&upload.filled);
    }
}

/// Runs the scheduled handlers of the command buffers `joined` scheduled, in order, with no lock
/// held: before the submission they are in goes, and before any of them is handed on as ended.
fn scheduled(joined: &mut Vec<Joined>) {
    for (cb, handlers) in joined.drain(..) {
        command::run_scheduled(cb, handlers);
    }
}

/// Submits a group: its uploads first, oldest first, then each segment in order, the last
/// signalling the progress `value`. Whatever was passed to `vkQueueSubmit2` goes to the completion
/// thread as in flight; a failed submission is taken as a loss.
fn submit(core: &Core, value: u64, uploads: Vec<Upload>, batch: Batch) {
    let Batch {
        mut segments,
        mut finished,
        watch,
        overs,
        backs,
        written,
        wrote,
        starts,
        apart,
        copy_ins,
        ..
    } = batch;
    let device = &core.device;
    // An upload begun for textures gone before their move records nothing; one is kept, or begun,
    // for the copies into textures over a buffer in an image of their own.
    let sides = !overs.is_empty() || !backs.is_empty();
    let mut ok = true;
    let mut empty = None;
    let mut kept = Vec::with_capacity(uploads.len());
    for upload in uploads {
        match upload.entries {
            0 => {
                if let Some(earlier) = empty.replace(upload) {
                    earlier.retire(core);
                }
            }
            _ => kept.push(upload),
        }
    }
    match (sides && kept.is_empty(), empty) {
        (true, Some(upload)) => kept.push(upload),
        (true, None) => match Upload::begin(core, 0) {
            Ok(upload) => kept.push(upload),
            Err(_) => ok = false,
        },
        (false, Some(upload)) => upload.retire(core),
        (false, None) => {}
    }
    let mut uploads = kept;
    order_fills(core, &uploads);
    // What the uploads wrote is visible to every command after them, and to the host once the
    // submission completes: one barrier, at the end of the last.
    if let Some((last, earlier)) = uploads.split_last_mut() {
        for upload in earlier {
            // SAFETY: the upload's command buffer was recording under the work lock, which
            // `close` took it out under.
            ok &= unsafe { device.end_command_buffer(upload.command) }.is_ok();
        }
        // Each texture over a buffer in an image of its own that a segment leans on takes the
        // bytes shaders reach (`crate::over`).
        // SAFETY: the upload's command buffer is recording, and only this thread has it; the
        // command buffers of the submission hold each texture, which holds its buffer, until it
        // retires, and each image is in its home layout, any move there recorded before.
        unsafe {
            let copies: Vec<OverCopy> = backs.iter().map(|(_, copy, _)| *copy).collect();
            over::record_side(device, last.command, &copies, false, &core.counters);
            over::record_side(device, last.command, &overs, true, &core.counters);
        }
        (last.held).extend(backs.into_iter().map(|(_, _, image)| Held::Texture(image)));
        record::barrier(device, last.command, Reach::Host);
        core.counters.count(Metric::Barriers);
        // SAFETY: as above.
        ok &= unsafe { device.end_command_buffer(last.command) }.is_ok();
    }
    let ahead = uploads.len();
    segments.splice(
        0..0,
        uploads.iter().map(|upload| Staged {
            command: Some(vk::CommandBufferSubmitInfo::default().command_buffer(upload.command)),
            ..Staged::default()
        }),
    );
    // One part, or, where the group goes member by member, one for each command buffer that leads
    // in it, the uploads with the first.
    let mut parts: Vec<Vec<Staged>> = Vec::with_capacity(1);
    let bounds: Vec<usize> = match apart {
        true => (starts.iter())
            .map(|start| start + ahead)
            .filter(|&at| at > 0)
            .collect(),
        false => Vec::new(),
    };
    for &at in bounds.iter().rev() {
        parts.push(segments.split_off(at));
    }
    parts.push(segments);
    parts.reverse();
    for part in &mut parts {
        part.retain(|staged| staged.command.is_some() || !staged.signals.is_empty());
    }
    parts.retain(|part| !part.is_empty());
    let progress = vk::SemaphoreSubmitInfo::default()
        .semaphore(core.done.progress)
        .value(value)
        .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS);
    match parts.last_mut().and_then(|part| part.last_mut()) {
        Some(last) => last.signals.push(progress),
        None => parts.push(vec![Staged {
            signals: vec![progress],
            ..Staged::default()
        }]),
    }
    if ok {
        let asked = times::host_time();
        #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
        (core.work.asking).store(true, Ordering::SeqCst);
        let queue = lock(&core.queues.runtime);
        (core.counters).queue_lock_wait(times::host_time().saturating_sub(asked));
        // A bind of an import over several dma-bufs since the last submission: this one waits for
        // it, and every later command, in this submission and after it, is ordered after that
        // wait (`crate::import::Sparse`).
        #[cfg(unix)]
        if let Some((timeline, value)) = (core.imports.sparse.as_ref()).and_then(|s| s.owed()) {
            if let Some(first) = parts.first_mut().and_then(|part| part.first_mut()) {
                first.waits.push(
                    vk::SemaphoreSubmitInfo::default()
                        .semaphore(timeline)
                        .value(value)
                        .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS),
                );
            }
        }
        for part in &parts {
            let merged = infos(part);
            let infos: Vec<vk::SubmitInfo2<'_>> = merged
                .iter()
                .map(|info| {
                    vk::SubmitInfo2::default()
                        .wait_semaphore_infos(info.waits)
                        .command_buffer_infos(&info.commands)
                        .signal_semaphore_infos(info.signals)
                })
                .collect();
            // Counted before the call: the work can complete, and a thread waiting on it go on to
            // read the metrics, before the call returns here.
            core.counters.count(Metric::Submissions);
            core.counters.submitted_infos(infos.len() as u64);
            // Inside the driver from here until the call returns, for the block-maker thread's
            // wait after its memory calls (`crate::submit_calls`). One made during a free is
            // counted before the call too; one a free began beside, as it returns.
            let call = core.submit_calls.enter();
            if call.entered_in_free() {
                core.counters.count(Metric::SubmissionsDuringFrees);
            }
            let began = times::host_time();
            // SAFETY: every command buffer is executable and in no other pending submission;
            // the one wait, a sparse bind's, is on a signal submitted earlier, and every signal
            // raises its semaphore;
            // the queue is held.
            ok = unsafe { device.queue_submit2(*queue, &infos, vk::Fence::null()) }.is_ok();
            let nanos = times::host_time().saturating_sub(began);
            core.counters.submit_call(nanos);
            if call.met_free() {
                if !call.entered_in_free() {
                    core.counters.count(Metric::SubmissionsDuringFrees);
                }
                core.counters.submit_call_during_free(nanos);
            }
            drop(call);
            if !ok {
                break;
            }
        }
        drop(queue);
    }
    if ok {
        // A command buffer that presents a drawable is scheduled now that the submission that
        // signals the consumer's semaphores has passed `vkQueueSubmit2`: a present made from here,
        // its own scheduled handlers' included, waits on a submitted signal. They run before the
        // completion thread learns of it below, so before its completed handlers. One whose
        // submission failed is not scheduled; its scheduled handlers run as it retires.
        let now = times::host_time();
        for (cb, pending, error) in &mut finished {
            if error.is_none() && pending.gpu.presents() {
                let scheduled = std::mem::take(&mut pending.handlers.scheduled);
                cb.scheduled(now, !scheduled.is_empty());
                if !scheduled.is_empty() {
                    command::run_scheduled(Arc::clone(cb), scheduled);
                }
            }
        }
    } else {
        complete::lose(core);
    }
    core.done
        .submitted(core, value, uploads, copy_ins, finished, watch);
    // Device work of the submission wrote these Managed buffers: their textures over their device
    // copies owe their copies in from the next submission (`crate::over::ManagedOvers`).
    for buffer in written {
        buffer.over_entries().written();
    }
    // Its writes of textures over a Managed buffer's device copy are submitted: the upload that
    // leads a later submission may take their copies back.
    for texture in wrote {
        if let Some(buffer) = texture.managed_over() {
            buffer.over_entries().settle(texture.id());
        }
    }
}

impl Upload {
    /// Recycles the upload's recorder and gives up what it holds: once its submission completed,
    /// or where it never went.
    pub(crate) fn retire(self, core: &Core) {
        core.recorders.give_back(core, self.recorder);
    }
}
