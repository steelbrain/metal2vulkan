//! A device's spare command pools and descriptor pools: a command buffer that completed gives back
//! the command pool its encoders recorded into, with the command buffers made from it, and the
//! descriptor pools its commands' sets came from (`descriptor_arena.rs`); the next command buffer
//! takes them where it would make its own. Made, used once and destroyed, a command pool cost 38
//! to 48 us and a descriptor pool 23 to 26 us on RADV (Radeon 780M), 7 to 8 and 3 on ANV, against
//! 0.2 and 0.02, and 1.3 and 0.1, reset.
//!
//! - **Command pools are kept, never released while the device lives**. Every command pool a
//!   command buffer gives back is kept, whatever its status, and every command buffer dropped with
//!   one gives it back too: the runtime calls `vkDestroyCommandPool` on a command buffer's pool
//!   only as the device is torn down, after its queues are idle. A command buffer makes a pool only
//!   where no spare of its queue's family is free, so the device holds at most as many as its
//!   command buffers held at once (Metal bounds those by each queue's `maxCommandBufferCount`).
//!   Before, a pool given back past 8 spares, or one that made more than 64 command buffers, was
//!   released on the handler thread that completed its command buffer; on NVIDIA's driver
//!   615.71.09 that `vkDestroyCommandPool` intermittently never returned (the driver spinning,
//!   with its own `[vkrt] Analysis` thread), so the command buffer never completed and its queue
//!   stalled.
//! - **A pool's command buffers follow its recent uses.** A pool given back keeps, of each level,
//!   as many command buffers as the most that one of its last [`RECENT_USES`] uses, or one of its
//!   uses in its current and its last span of [`RECENT_FOR`], handed out (a secondary at least,
//!   below), and frees the rest; and the oldest spare no command buffer took for [`IDLE_FOR`]
//!   frees every command buffer it holds as a later give-back finds it ([`take_idle`]), then goes
//!   back among the spares as the oldest. Kept by their peak, a device's pools held, for its life,
//!   what the largest command buffer each ever served recorded, and each burst of command buffers
//!   at once added pools and filled them: on RADV each command buffer holds 96 KiB of GTT (buffer
//!   objects of 80 and 16 KiB) until its pool is trimmed. Kept by the last [`RECENT_USES`] uses
//!   alone, a pool freed what a use took again a few dozen uses later, and made it again then.
//!   Counted: `Spares::command_buffers_held` and `Spares::churn` (the crate's tests).
//! - **The spares hold at most [`HELD`] command buffers together.** Counted with them: the pool
//!   given back, and what each pool out of the spares held as it went out, taken by a command
//!   buffer, freeing command buffers or beginning a primary ahead ([`CommandPools::out`]), whose
//!   command buffers hold their last recordings all the same. A give-back that leaves more takes
//!   the oldest spares that hold any out, each to free every command buffer it holds as a spare
//!   left idle does, until they hold at most that many; where that leaves more, the pool given
//!   back frees its last made past them, a secondary kept ([`past_held`]). A command buffer
//!   holds what its recording took of the driver's device-wide pools until it is begun again,
//!   reset or freed, and a pool's recent uses bound only its own. ANV below Xe-HP (Mesa 26.2.4
//!   and 26.2.6, `anv_device.c`) takes a 64 KiB block of a 1 GiB binding-table pool for each
//!   command buffer begun, so a device holds at most 16,384 command buffers at once, and a
//!   recording past them fails with `VK_ERROR_OUT_OF_DEVICE_MEMORY`, which no message reports,
//!   or crashes in the driver. On ms01's Iris Xe one command buffer of render passes, each a draw
//!   beside a cleared depth target in a primary and a secondary of its own, failed so at the end
//!   of its 8,087th pass, the spares holding 424 command buffers; such a pass records inline in
//!   one now (`render/inline_first.rs`), and past the first 2,048 encoders of a command buffer
//!   its batch begins no command buffer of barriers before each (`encoder.rs`). A test's frames of 707 and then 1,413 such
//!   passes, each frame a command buffer of its own whose pool kept its peak, left the spares
//!   holding up to 9,771 command buffers of earlier frames before a frame of 1,413 passes, and
//!   failed so in 6 of 79 runs beside the runtime suite, each where they held 9,716 to 9,771
//!   (2,786 to 7,747 in the runs that passed). [`HELD`] leaves three quarters of that pool to
//!   the command buffers committed and in recording. What it costs there, that test alone, five
//!   runs each way (debug): its spares held 4,096 at most against 7,787 to 7,793; its pools'
//!   frees ran 220 times on the free thread against 14 to 17; it allocated 12,559 command buffers
//!   in four runs and 9,864 in the fifth against 10,193 in each; and it took 13.6 to 14.0 s
//!   against 13.6 to 13.9 s. Beside the runtime suite it failed in none of 76 runs.
//! - **Freed on the device's free thread** ([`PoolFrees`], `free_thread.rs`), not on the thread
//!   giving the pool back, which for a command buffer that completed is its queue's completion
//!   thread before its completed handlers run and its waiters return: [`FREED_AT_ONCE`] command
//!   buffers at a time, `vkFreeCommandBuffers` then `vkTrimCommandPool` (but on ANV, below),
//!   which returns what the driver keeps of freed command buffers (`release.rs`'s
//!   `PoolBuffers`), each inside a section of the exit guard. The pool stays out of the spares
//!   until its frees ran (a command pool needs external synchronization), so a taker meanwhile
//!   takes another spare or makes a pool. Where the free thread's queue is full, or the thread
//!   cannot start, they run on the giving thread.
//!   The device's teardown waits for those posted before it releases the spares. Counted (the
//!   crate's tests): `Spares::freed_on`, the threads that freed them. On a Radeon 780M (RADV), in
//!   release (`tests::spares_latency`), a give-back that freed a burst's command buffers on the
//!   completion thread held the command buffer's first completed handler 1.5 to 1.6 ms at the
//!   median after its status read final (2.2 ms at the slowest), and its waiter's return as long;
//!   with the frees posted, 6 us and 15 to 17 us (63 us at the slowest).
//!   [`FREED_AT_ONCE`] freed and trimmed took, in `tests::spares_bench`: on a Radeon 780M
//!   (RADV) 18 to 35 us to free them and 0.63 to 0.70 ms at the median to trim (3.0 ms at the
//!   slowest), which returned their GTT, where the trim of 252 at once took 5.2 to 6.3 ms; on an
//!   Iris Xe (ANV) 64 to 103 us and 43 to 47 us, the driver keeping their memory for the command
//!   buffers it makes next; on an RTX 5080 Laptop GPU (NVIDIA 615) 4 to 7 us and 0.4 to 48 us
//!   (0.22 ms at the slowest); on an Apple M3 Max (MoltenVK 1.4.2) 0.9 to 1.0 us each (3.5 us at
//!   the slowest), the process's physical footprint unchanged by them (its first 252 command
//!   buffers had added 176 KiB to it, which stayed); on lavapipe a few microseconds.
//! - **Not trimmed where a trim fails the kept barriers** (ANV:
//!   `RuntimeDriverDefect::TrimmedCommandBufferFailsResubmission`, `driver_defects.rs`). A
//!   batch's kept barriers are submitted again as recorded (`command_buffer/kept_barriers.rs`),
//!   and there each such submission names every command buffer that ever came next after one;
//!   once one of those is trimmed away, the submission fails and the device is lost. So the pool's
//!   command buffers are freed and the pool is not trimmed: the driver keeps each one it freed,
//!   reset, for the pool's later allocations (Mesa's common command pool, `vk_command_pool.c`), a
//!   pool holding for the device's life as many command buffers' objects as it held at once, but
//!   none of their recordings beyond each one's first batch buffer object. On the Iris Xe that
//!   left the process's memory as trimming did: a burst's command buffers freed and trimmed, freed
//!   untrimmed, or kept returned none of what the burst grew (`driver_defects.rs`).
//! - **Descriptor pools are bounded.** At most [`DESCRIPTOR_POOLS`] descriptor pools of at most
//!   [`DESCRIPTORS`] descriptors together; a descriptor pool given back past them releases the
//!   oldest spares first, so the spares follow the shapes in use. The descriptors bound the
//!   memory; the count is set by a frame's demand: a window server's frame of 40 command
//!   buffers, 31 of a pass and 22 of a dispatch (host_ab's `drag`), holds about 53 pools in
//!   flight, a draw's needs and a dispatch's each opening a pool of their own, and under a bound of
//!   32 it made about 21 pools anew a frame.
//! - **Fitted arenas keep spares of their own.** The pools of the runtime's own kernels' sets
//!   (`DescriptorArena::fitted`) go back to a list apart from the command buffers', bounded the
//!   same way: a command buffer's set 0 takes any spare with room, so in one list it took the
//!   small pools a texture's moves gave back, and under a window server's command buffers a frame
//!   the fitted arenas made about a third of their pools anew.
//! - **A pool whose command buffers all ended is not reset; its command buffers are, as they are
//!   begun.** Its pools are made with `VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT`, and
//!   Vulkan: "If a command buffer is in the invalid, or executable state, and the command buffer
//!   was allocated from a command pool with the VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT
//!   flag set, then vkBeginCommandBuffer implicitly resets the command buffer, behaving as if
//!   vkResetCommandBuffer had been called with VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT not
//!   set". A command buffer counts the command buffers of its pool it ended
//!   (`CommandBuffer::end_raw`, every `vkEndCommandBuffer` of one): where it completed and ended
//!   every one handed out, none is pending or recording, so each is executable or invalid. A
//!   pool given back otherwise may hold one still recording, which the validation layer forbids
//!   beginning (`VUID-vkBeginCommandBuffer-commandBuffer-00049`: "commandBuffer must not be in
//!   the recording or pending state"): by a command buffer that failed or was dropped
//!   unsubmitted (an encoder never ended, a recording an error cut short), or by one that
//!   completed with a recording it left unused, as its encoders may (a render pass instance's
//!   draws its pass replaced before any draw, a refusal past a begin, a segment's core dropped).
//!   Its taker resets that pool first, as every taker did before ([`CommandPool::recording`]),
//!   and counts it (`WorkCounts::spare_command_pools_reset`). ANV resets a command buffer at each begin whatever its pool
//!   (Mesa 26.2.4, `genX(BeginCommandBuffer)`), so a `vkResetCommandPool` before the begins reset
//!   each one twice: on ms01's Iris Xe, ready to record again, one command buffer took 8.0 to 20.5
//!   us of thread time begun alone against 12.1 to 25.1 us after the pool's reset, three 24.5 to
//!   50.3 against 36.6 to 63.2; the RTX 5080 (615.71.09) 0.13 to 0.24 us less; RADV (Radeon 780M)
//!   0.04 to 0.15 us less for one and 0.16 to 0.30 us more for three.
//! - **A pool given back holds a secondary command buffer.** One that made none makes one as it
//!   is given back, on the thread giving it back, inside a section of the exit guard. A pass
//!   whose first instance does not record inline begins its draws in a secondary command buffer
//!   as it opens (`render.rs`); taken from a pool that had served only passes recorded inline,
//!   it would be made there, on the thread that encodes: on the M3 Max (MoltenVK, macOS 27.0.1),
//!   0 to 5 of a run's 4020 pass opens made one, 16 allocations each against 14. A making that
//!   fails is left to that open.
//! - **A settled pool given back holds a primary already begun.** The thread that completed its
//!   command buffer begins one of its primaries once the completed handlers ran
//!   (`ONE_TIME_SUBMIT`, as every encoder begins its own; recording nothing), inside a section of
//!   the exit guard, where no command buffer took the pool since ([`Spares::begin_ahead`]). It
//!   takes the pool out of the spares under their lock and begins with no lock held, then keeps
//!   it again, the newest: a taker meanwhile takes another spare, or makes a pool, and never
//!   waits for that begin (a wait on the lock across it was 747 waits of the guest's encoding
//!   thread in 255 frames, 11 us at the median, on ANV). The command buffer's gate opens only
//!   once that thread's hold goes (`command_buffer/gate.rs`), so it begins none where a thread
//!   waits for the gate: a waiter's return waits for no begin, but for one wait that came after
//!   the check. The first encoder of the command buffer that takes the pool takes the primary
//!   begun (`EncoderCore::open`), so the thread that encodes makes no `vkBeginCommandBuffer` for
//!   it. On ANV, whose begin clears the command buffer's state (Mesa 26.2.4,
//!   `genX(BeginCommandBuffer)`), that begin was 8 to 20 us of the encoding thread's time a
//!   command buffer (above). Vulkan asks only external synchronization of a command buffer and
//!   its pool, never one thread: `vkBeginCommandBuffer`, and every command recorded, "Host access
//!   to commandBuffer must be externally synchronized" and "Host access to the VkCommandPool that
//!   commandBuffer was allocated from must be externally synchronized". The pool passes between
//!   threads through the spares, under their lock, and is never in use on two threads at once, so
//!   a command buffer begun on one thread is recorded on another. Only the first encoder takes
//!   it: the primaries every other path makes (segments, continuations, a batch's barriers) are
//!   never the begun one. A pool whose taker resets it (one given back unsettled) resets the begun
//!   primary with the rest, which is then an ordinary one again. A pool given back unsettled, or
//!   while a thread waited, or whose begin fails, holds none. A taker takes the newest spare
//!   holding one first ([`take_command_pool`]). Counted: `WorkCounts::primaries_begun_ahead`, and
//!   those encoders took, `WorkCounts::encoder_primaries_begun_ahead`.
//! - **Descriptor pools are reset by the taker**, on the thread that encodes: the program's own, or
//!   one running the program's code (a handler).
//! - **Taken outside sections alone.** A section of the exit guard (a host step's recording on the
//!   scheduler's thread) calls only the guard's table of entry points, which holds no reset of a
//!   pool, and takes no spare: a command buffer or command recorded there makes its
//!   pools as before; what it made is given back all the same. A command buffer that encoded
//!   nothing, whose submissions record only their barriers, in sections, takes its spare command
//!   pool at its commit, on the committing thread.
//! - **Given back once nothing of them is pending.** A command buffer that completed
//!   (`CommandBufferStatus::Completed`) gives back its command pool and its descriptor pools; one
//!   that failed, once its work ended, gives back its command pool and releases its descriptor
//!   pools; one dropped unsubmitted gives back its command pool.
//! - **Released with the device**, as its teardown begins (`DeviceShared`'s drop, after
//!   `vkDeviceWaitIdle`): every command buffer holds the device, so none is left to give one
//!   back.
//! - **Audited where the device's pending audit is on** (`pending_audit.rs`): a pool given back,
//!   and a spare pool its taker resets, has each of its command buffers checked against the
//!   submissions that carry it, its primary begun ahead among them; the begin ahead is checked as
//!   every begin is (`device/functions.rs`).

use super::descriptor_arena::{Room, SetNeeds};
use super::device::{lock, DeviceShared};
use super::error::{vulkan, Error};
use super::exit_guard::{self, Call};
use super::pending_audit::Freeing;
use super::release::{self, Released};
use ash::vk;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// The most spare descriptor pools a device keeps (of each list, a command buffer's arenas' and
/// the fitted arenas'): a few frames' worth of a window server's (the module docs).
pub(super) const DESCRIPTOR_POOLS: usize = 256;

/// The most descriptors the spare descriptor pools hold together.
pub(super) const DESCRIPTORS: u64 = 1 << 18;

/// The uses of a command pool whose command buffers it keeps: given back, it keeps of each level
/// as many as the most one of its last this many uses handed out (the module docs).
pub(super) const RECENT_USES: usize = 8;

/// The spans of a command pool's uses whose command buffers it keeps besides its last
/// [`RECENT_USES`]: given back, it keeps of each level as many as the most one of its uses in this
/// span and the one before handed out, so that one a use takes again within one to two spans
/// stays (the module docs).
pub(super) const RECENT_FOR: Duration = Duration::from_secs(1);

/// How long a spare command pool stays untaken before every command buffer it holds is freed (the
/// module docs).
pub(super) const IDLE_FOR: Duration = Duration::from_secs(10);

/// The most command buffers one `vkFreeCommandBuffers` frees, the pool trimmed after each where
/// the device trims (the module docs).
pub(super) const FREED_AT_ONCE: usize = 32;

/// The most command buffers a give-back leaves the spare command pools holding, with those the
/// pools out of them held as each went out and the primaries begun ahead: a quarter of the
/// command buffers with recordings a device on ANV below Xe-HP holds at once (the module docs).
pub(super) const HELD: usize = 4096;

/// The index of `level`'s list in a [`CommandPool`]: primary 0, secondary 1.
fn level_index(level: vk::CommandBufferLevel) -> usize {
    usize::from(level == vk::CommandBufferLevel::SECONDARY)
}

/// The command buffers a device allocated, by the place that asked, and those its spares freed, by
/// why, so far: for the crate's device tests.
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
#[derive(Clone, Debug, Default)]
pub(super) struct Churn {
    /// By `file:line` of the call that asked (`DeviceShared::allocate_command_buffers` and the
    /// calls that pass their caller's place on).
    pub(super) allocated: std::collections::BTreeMap<String, u64>,
    /// Freed from a pool given back, past its recent uses ([`CommandPool::given_back`]).
    pub(super) past_recent_uses: u64,
    /// Freed from a pool given back, past [`HELD`] ([`past_held`]).
    pub(super) past_held: u64,
    /// Freed by a spare left idle or shed past [`HELD`] ([`CommandPool::shed`]).
    pub(super) shed: u64,
}

/// A command pool and the command buffers made from it, by level: those a command buffer handed
/// out since it took the pool come first.
pub(super) struct CommandPool {
    pub(super) raw: vk::CommandPool,
    /// The queue family the pool was made for: its command buffers run on queues of it only.
    family: u32,
    made: [Vec<vk::CommandBuffer>; 2],
    handed: [usize; 2],
    /// A primary made from the pool, begun ahead while the pool was a spare and recording
    /// nothing, not among `made` until an encoder takes it ([`take_begun`](Self::take_begun)).
    pub(super) begun: Option<vk::CommandBuffer>,
    /// One of its command buffers may still be recording: the command buffer that gave it back
    /// did not complete, or left one it was handed unended (the module docs), so its taker
    /// resets it.
    pub(super) recording: bool,
    /// The command buffers each of its last [`RECENT_USES`] uses handed out, by level, the newest
    /// last ([`given_back`](Self::given_back)).
    uses: VecDeque<[usize; 2]>,
    /// When its current span of [`RECENT_FOR`] began, the most one of its uses in that span handed
    /// out, and the most one in the span before did, by level; `None` before a use.
    spans: Option<(Instant, [usize; 2], [usize; 2])>,
    /// When a command buffer last gave it back, or `None` before one did.
    given: Option<Instant>,
    /// The command buffers it held as it last went out of the spares, counted among those of the
    /// pools out of them until it comes back ([`CommandPools::out`]).
    out: usize,
}

impl CommandPool {
    /// A new pool, made for queue family `family`, of no command buffer yet.
    pub(super) fn new(raw: vk::CommandPool, family: u32) -> CommandPool {
        CommandPool {
            raw,
            family,
            made: [Vec::new(), Vec::new()],
            handed: [0, 0],
            begun: None,
            recording: false,
            uses: VecDeque::new(),
            spans: None,
            given: None,
            out: 0,
        }
    }

    /// A command buffer gives the pool back at `now`, having handed out what
    /// [`handed_out`](Self::handed_out) says: its use is noted, and of the command buffers of
    /// each level past the most one of its last [`RECENT_USES`] uses, or one of its uses in its
    /// current and its last span of [`RECENT_FOR`], handed out (of secondaries, past one at
    /// least), the last made, are taken out of it, to be freed (the module docs). Those it handed
    /// out come first, so none of them is among those.
    pub(super) fn given_back(&mut self, now: Instant) -> Vec<vk::CommandBuffer> {
        if self.uses.len() == RECENT_USES {
            self.uses.pop_front();
        }
        self.uses.push_back(self.handed);
        self.given = Some(now);
        let (start, mut current, mut last) = self.spans.unwrap_or((now, [0, 0], [0, 0]));
        let start = match now.saturating_duration_since(start) {
            since if since < RECENT_FOR => start,
            since if since < 2 * RECENT_FOR => {
                last = current;
                current = [0, 0];
                start + RECENT_FOR
            }
            _ => {
                (current, last) = ([0, 0], [0, 0]);
                now
            }
        };
        for at in 0..2 {
            current[at] = current[at].max(self.handed[at]);
        }
        self.spans = Some((start, current, last));
        let mut past = Vec::new();
        for at in 0..2 {
            let recent = self.uses.iter().map(|used| used[at]).max().unwrap_or(0);
            let used = recent.max(current[at]).max(last[at]);
            let keep = match at == level_index(vk::CommandBufferLevel::SECONDARY) {
                true => used.max(1),
                false => used,
            };
            let from = keep.min(self.made[at].len());
            past.extend(self.made[at].drain(from..));
        }
        past
    }

    /// Whether no command buffer took it since one gave it back [`IDLE_FOR`] or longer before
    /// `now`, and it holds a command buffer (the module docs).
    fn idle(&self, now: Instant) -> bool {
        let holds = self.begun.is_some() || self.made.iter().any(|made| !made.is_empty());
        let since = self.given.map(|given| now.saturating_duration_since(given));
        holds && since.is_some_and(|since| since >= IDLE_FOR)
    }

    /// Every command buffer it holds, taken out of it to be freed, the primary begun ahead first:
    /// a spare left idle (the module docs).
    pub(super) fn shed(&mut self) -> Vec<vk::CommandBuffer> {
        let mut shed: Vec<vk::CommandBuffer> = self.begun.take().into_iter().collect();
        for made in self.made.iter_mut().rev() {
            shed.append(made);
        }
        shed
    }

    /// Its command buffers past `keep` in all, the last made, taken out of it to be freed:
    /// secondaries first, one kept, then primaries; never the primary begun ahead (the module
    /// docs).
    pub(super) fn past(&mut self, keep: usize) -> Vec<vk::CommandBuffer> {
        let mut over = self.held().saturating_sub(keep);
        let mut past = Vec::new();
        for (at, least) in [(level_index(vk::CommandBufferLevel::SECONDARY), 1), (0, 0)] {
            let made = &mut self.made[at];
            let from = made.len() - over.min(made.len().saturating_sub(least));
            over -= made.len() - from;
            past.extend(made.drain(from..));
        }
        past
    }

    /// The command buffers it holds, the primary begun ahead among them.
    pub(super) fn held(&self) -> usize {
        self.made[0].len() + self.made[1].len() + usize::from(self.begun.is_some())
    }

    /// The pool's primary begun ahead (the module docs), handed out: taken by the first encoder
    /// of the command buffer that took the pool. `None` where it holds none.
    pub(super) fn take_begun(&mut self) -> Option<vk::CommandBuffer> {
        let buffer = self.begun.take()?;
        let at = level_index(vk::CommandBufferLevel::PRIMARY);
        self.made[at].insert(self.handed[at], buffer);
        self.handed[at] += 1;
        Some(buffer)
    }

    /// A command buffer of `level` made from the pool before a command buffer last took it, not
    /// handed out since, in any state but pending (its begin resets it: the module docs); `None`
    /// once each was.
    pub(super) fn reuse(&mut self, level: vk::CommandBufferLevel) -> Option<vk::CommandBuffer> {
        let at = level_index(level);
        let buffer = self.made[at].get(self.handed[at]).copied()?;
        self.handed[at] += 1;
        Some(buffer)
    }

    /// The command buffers handed out since a command buffer last took it, of both levels.
    pub(super) fn handed_out(&self) -> usize {
        self.handed[0] + self.handed[1]
    }

    /// `buffer`, of `level`, was just made from the pool, and is handed out.
    pub(super) fn made(&mut self, level: vk::CommandBufferLevel, buffer: vk::CommandBuffer) {
        let at = level_index(level);
        self.made[at].push(buffer);
        self.handed[at] = self.made[at].len();
    }

    /// A command buffer took the pool, reset where it was [`recording`](Self::recording): none of
    /// the command buffers made from it is handed out, and none is recording but the primary
    /// begun ahead, where the pool was not reset.
    pub(super) fn taken(&mut self) {
        if self.recording {
            // The reset left the primary begun ahead in the initial state.
            if let Some(buffer) = self.begun.take() {
                self.made[level_index(vk::CommandBufferLevel::PRIMARY)].push(buffer);
            }
        }
        self.handed = [0, 0];
        self.recording = false;
    }

    /// Every command buffer made from the pool, of both levels, the primary begun ahead among
    /// them: what its reset or give-back frees (the device's pending audit, `pending_audit.rs`).
    fn buffers(&self) -> Vec<vk::CommandBuffer> {
        let [primaries, secondaries] = &self.made;
        let made = primaries.iter().chain(secondaries).copied();
        made.chain(self.begun).collect()
    }

    /// Whether the pool made no secondary command buffer (the module docs).
    pub(super) fn wants_secondary(&self) -> bool {
        self.made[level_index(vk::CommandBufferLevel::SECONDARY)].is_empty()
    }

    /// `buffer`, a secondary command buffer just made from the pool, not handed out: the next
    /// command buffer that takes the pool hands it out first.
    pub(super) fn stock_secondary(&mut self, buffer: vk::CommandBuffer) {
        self.made[level_index(vk::CommandBufferLevel::SECONDARY)].push(buffer);
    }

    /// Make the pool a secondary command buffer where it [`wants_secondary`](Self::wants_secondary)
    /// one (the module docs): inside a section of the exit guard, on whatever thread gives the
    /// pool back; nothing where the guard is closing, or where the making fails.
    fn stock(&mut self, device: &DeviceShared) {
        if !self.wants_secondary() {
            return;
        }
        let Some(_busy) = exit_guard::enter() else {
            return;
        };
        let info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.raw)
            .level(vk::CommandBufferLevel::SECONDARY)
            .command_buffer_count(1);
        if let Ok(made) = device.allocate_command_buffers(&info) {
            for buffer in made {
                self.stock_secondary(buffer);
            }
        }
    }

    /// Whether the pool may begin a primary ahead: none of its command buffers is recording or
    /// pending, and it holds none begun (the module docs).
    fn begins_ahead(&self) -> bool {
        !self.recording && self.begun.is_none()
    }

    /// Begin one of the pool's primaries ahead, where it [`begins_ahead`](Self::begins_ahead): one
    /// it made, which its begin resets, or a new one. Inside a section of the exit guard, on a
    /// pool taken out of the spares; nothing where the guard is closing, and none begun where the
    /// making or the begin fails. Whether it began one.
    fn begin_ahead(&mut self, device: &DeviceShared) -> bool {
        if !self.begins_ahead() {
            return false;
        }
        let Some(_busy) = exit_guard::enter() else {
            return false;
        };
        let at = level_index(vk::CommandBufferLevel::PRIMARY);
        let buffer = match self.made[at].pop() {
            Some(buffer) => buffer,
            None => {
                let info = vk::CommandBufferAllocateInfo::default()
                    .command_pool(self.raw)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1);
                match device.allocate_command_buffers(&info) {
                    Ok(made) if !made.is_empty() => made[0],
                    _ => return false,
                }
            }
        };
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: the pool is out of the spares, this thread's alone, and none of its command
        // buffers is pending or recording (settled): the buffer is executable, invalid or
        // initial, and its begin resets it (`VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT`).
        let begun = unsafe { device.vk().begin_command_buffer(buffer, &begin) };
        exit_guard::called(Call::BeginCommandBuffer);
        match begun {
            Ok(()) => {
                self.begun = Some(buffer);
                true
            }
            // Begun again at its next use, which resets it.
            Err(_) => {
                self.made[at].push(buffer);
                false
            }
        }
    }
}

/// A descriptor pool, with the room it was made with.
pub(super) struct DescriptorPool {
    pub(super) raw: vk::DescriptorPool,
    pub(super) room: Room,
}

/// A device's spare command pools, and the command buffers the pools out of them hold.
#[derive(Default)]
pub(super) struct CommandPools {
    /// Oldest first.
    pub(super) spares: Vec<CommandPool>,
    /// The command buffers the pools out of the spares held as each went out: taken by a command
    /// buffer, out to free command buffers on the device's free thread, or to begin a primary
    /// ahead (the module docs).
    pub(super) out: usize,
}

impl CommandPools {
    /// `pool` went out of the spares: what it holds is counted until it comes back.
    pub(super) fn went_out(&mut self, pool: &mut CommandPool) {
        pool.out = pool.held();
        self.out += pool.out;
    }

    /// `pool` came back from out of the spares, or goes for good: no longer counted.
    pub(super) fn came_back(&mut self, pool: &mut CommandPool) {
        self.out -= pool.out;
        pool.out = 0;
    }
}

/// A device's spares.
pub(super) struct Spares {
    /// Shared with the device's free thread, which keeps a pool again once its command buffers
    /// are freed ([`PoolFrees`]).
    command_pools: Arc<Mutex<CommandPools>>,
    /// A pool is trimmed after its command buffers are freed, but where the driver fails a
    /// command buffer submitted again once one that came next after it is trimmed (the module
    /// docs).
    trims: bool,
    /// The names of the threads that freed spares' command buffers, one each time, for the
    /// crate's device tests.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    freed_on: Arc<Mutex<Vec<String>>>,
    /// The device's command buffers allocated and the spares' freed, for the crate's device tests.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    churn: Mutex<Churn>,
    /// The command buffers' arenas', oldest first.
    descriptor_pools: Mutex<VecDeque<DescriptorPool>>,
    /// The fitted arenas' (the module docs), oldest first.
    fitted_pools: Mutex<VecDeque<DescriptorPool>>,
}

impl Spares {
    /// The spares of a device, which trims a command pool after freeing its command buffers
    /// where `trims` (the module docs).
    pub(super) fn new(trims: bool) -> Spares {
        Spares {
            command_pools: Default::default(),
            trims,
            #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
            freed_on: Default::default(),
            #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
            churn: Default::default(),
            descriptor_pools: Default::default(),
            fitted_pools: Default::default(),
        }
    }

    /// A spare command pool of `device` made for queue family `family`, none of whose command
    /// buffers is pending or recording (reset where one may be, the module docs); `None` when
    /// there is none, or this thread is in a section.
    pub(super) fn command_pool(
        &self,
        device: &DeviceShared,
        family: u32,
    ) -> Result<Option<CommandPool>, Error> {
        if exit_guard::in_section() {
            return Ok(None);
        }
        let taken = {
            let mut pools = lock(&self.command_pools);
            take_command_pool(&mut pools.spares, family).map(|mut pool| {
                pools.went_out(&mut pool);
                pool
            })
        };
        let Some(mut pool) = taken else {
            return Ok(None);
        };
        if pool.recording {
            if let Some(audit) = device.pending_audit() {
                audit.check(device.vk(), Freeing::PoolReset, &pool.buffers());
            }
            // SAFETY: the pool is the caller's alone now, and nothing of it is pending: the
            // command buffer that gave it back ended its work or was never submitted.
            let reset = unsafe {
                device
                    .vk()
                    .reset_command_pool(pool.raw, vk::CommandPoolResetFlags::empty())
            };
            if let Err(error) = reset {
                lock(&self.command_pools).came_back(&mut pool);
                // SAFETY: as above.
                unsafe { release_in_section(device, pool.raw) };
                return Err(vulkan(error));
            }
            device.counts.spare_command_pool_reset();
        }
        pool.taken();
        device.counts.spare_command_pool_taken();
        Ok(Some(pool))
    }

    /// Keep `pool` of `device`, of a command buffer whose work ended or that was never
    /// submitted: nothing of it is pending, and nothing else holds it; `settled` where the
    /// command buffer completed and ended every command buffer of it handed out, so none is
    /// recording (the module docs). A pool that made no secondary command buffer makes one. Where
    /// it holds command buffers past its recent uses, those are freed; the oldest spare no
    /// command buffer took for [`IDLE_FOR`] frees all of its own ([`take_idle`]); and past
    /// [`HELD`] command buffers in the spares and the pool together, the oldest spares free all
    /// of theirs, and the pool those past it ([`past_held`]); each pool out of the spares until
    /// its frees ran on the device's free thread ([`PoolFrees`]). Never released here.
    pub(super) fn give_command_pool(
        &self,
        device: &DeviceShared,
        mut pool: CommandPool,
        settled: bool,
    ) {
        // The audit reads the device's timelines inside a section of the exit guard, on
        // whatever thread gives the pool back.
        if let Some(audit) = device.pending_audit() {
            if let Some(_busy) = exit_guard::enter() {
                audit.check(device.vk(), Freeing::GiveBack, &pool.buffers());
            }
        }
        let now = Instant::now();
        pool.recording = !settled;
        let mut past = pool.given_back(now);
        #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
        let recent = past.len() as u64;
        pool.stock(device);
        let (given, shed) = {
            let mut pools = lock(&self.command_pools);
            pools.came_back(&mut pool);
            let idle = take_idle(&mut pools.spares, now);
            let out = pools.out;
            let (oldest, held) = past_held(&mut pools.spares, out, &mut pool);
            #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
            {
                let mut churn = lock(&self.churn);
                churn.past_recent_uses += recent;
                churn.past_held += held.len() as u64;
            }
            past.extend(held);
            let given = match past.is_empty() {
                true => {
                    keep_command_pool(&mut pools.spares, pool);
                    None
                }
                false => {
                    pools.went_out(&mut pool);
                    Some(self.frees(pool, past, false))
                }
            };
            let shed: Vec<_> = (idle.into_iter().chain(oldest))
                .map(|mut pool| {
                    let shed = pool.shed();
                    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
                    {
                        lock(&self.churn).shed += shed.len() as u64;
                    }
                    pools.went_out(&mut pool);
                    self.frees(pool, shed, true)
                })
                .collect();
            (given, shed)
        };
        for frees in given.into_iter().chain(shed) {
            // SAFETY: nothing of a pool given back or of a spare is pending, and each is out of
            // the spares.
            if let Err(frees) = unsafe { device.frees.post_pool(frees) } {
                // Inside a section of the exit guard, which this thread may be outside of (a
                // command buffer completes on a handler thread). Once the process is exiting, the
                // command buffers are left to its end.
                if let Some(_busy) = exit_guard::enter() {
                    // SAFETY: as above.
                    unsafe { (*frees).free(device.vk()) };
                }
            }
        }
    }

    /// The frees of `buffers`, taken out of `pool`, which goes back among the spares once they
    /// ran, the oldest where `idle`, else the newest.
    fn frees(
        &self,
        pool: CommandPool,
        buffers: Vec<vk::CommandBuffer>,
        idle: bool,
    ) -> Box<PoolFrees> {
        Box::new(PoolFrees {
            pool,
            buffers,
            idle,
            trim: self.trims,
            spares: Arc::clone(&self.command_pools),
            #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
            freed_on: Arc::clone(&self.freed_on),
        })
    }

    /// The names of the threads that freed spares' command buffers so far, one each time, for
    /// the crate's device tests.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(super) fn freed_on(&self) -> Vec<String> {
        lock(&self.freed_on).clone()
    }

    /// `count` command buffers the device allocated, asked for `at`
    /// (`DeviceShared::allocate_command_buffers`), for the crate's device tests.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(super) fn allocated(&self, at: &std::panic::Location<'_>, count: u32) {
        let place = format!("{}:{}", at.file(), at.line());
        *lock(&self.churn).allocated.entry(place).or_default() += u64::from(count);
    }

    /// The device's command buffers allocated and the spares' freed so far, for the crate's
    /// device tests.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(super) fn churn(&self) -> Churn {
        lock(&self.churn).clone()
    }

    /// Whether a pool is trimmed after its command buffers are freed, for the crate's device
    /// tests.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(super) fn trims(&self) -> bool {
        self.trims
    }

    /// The command buffers the spare command pools hold, for the crate's device tests.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(super) fn command_buffers_held(&self) -> usize {
        lock(&self.command_pools)
            .spares
            .iter()
            .map(CommandPool::held)
            .sum()
    }

    /// Begin a primary ahead in the spare command pool `raw` of `device`, where it is still a
    /// spare, settled, and holds none begun (the module docs): taken out of the spares under
    /// their lock, begun with no lock held, so a taker meanwhile takes another spare rather than
    /// wait for the begin, and kept again, the newest. Counted (`WorkCounts::primaries_begun_ahead`)
    /// once it is kept again, so a taker that saw the count finds the pool among the spares.
    pub(super) fn begin_ahead(&self, device: &DeviceShared, raw: vk::CommandPool) {
        let taken = {
            let mut pools = lock(&self.command_pools);
            take_to_begin(&mut pools.spares, raw).map(|mut pool| {
                pools.went_out(&mut pool);
                pool
            })
        };
        let Some(mut pool) = taken else {
            return;
        };
        let begun = pool.begin_ahead(device);
        {
            let mut pools = lock(&self.command_pools);
            pools.came_back(&mut pool);
            keep_command_pool(&mut pools.spares, pool);
        }
        if begun {
            device.counts.primary_begun_ahead();
        }
    }

    /// A spare descriptor pool of `device`, reset, with room for one set of `needs`, the newest
    /// such: a command buffer's arena's, or, for a fitted arena (`fitted`, the most descriptors
    /// it takes a spare of), one of the fitted arenas' own; `None` when there is none, or this
    /// thread is in a section.
    pub(super) fn descriptor_pool(
        &self,
        device: &DeviceShared,
        needs: &SetNeeds,
        fitted: Option<u64>,
    ) -> Result<Option<DescriptorPool>, Error> {
        if exit_guard::in_section() {
            return Ok(None);
        }
        let pools = match fitted {
            Some(_) => &self.fitted_pools,
            None => &self.descriptor_pools,
        };
        let pool = take_descriptor_pool(&mut lock(pools), needs, fitted);
        let Some(pool) = pool else {
            return Ok(None);
        };
        // SAFETY: the pool is the caller's alone now, and nothing uses its sets: a command buffer
        // that completed gave it back.
        let reset = unsafe {
            device
                .vk()
                .reset_descriptor_pool(pool.raw, vk::DescriptorPoolResetFlags::empty())
        };
        if let Err(error) = reset {
            // SAFETY: as above.
            unsafe { release_in_section(device, pool.raw) };
            return Err(vulkan(error));
        }
        Ok(Some(pool))
    }

    /// Keep `pool` of `device`, whose sets nothing uses any more, among a command buffer's
    /// arena's spares or, where `fitted`, among the fitted arenas', releasing the oldest of that
    /// list past the bounds (or `pool` itself, when it alone is past them).
    pub(super) fn give_descriptor_pool(
        &self,
        device: &DeviceShared,
        pool: DescriptorPool,
        fitted: bool,
    ) {
        let pools = match fitted {
            true => &self.fitted_pools,
            false => &self.descriptor_pools,
        };
        let past = keep_descriptor_pool(&mut lock(pools), pool);
        for raw in past {
            // SAFETY: nothing uses the pool's sets (the caller's, or a spare's), and nothing else
            // holds it.
            unsafe { release_in_section(device, raw) };
        }
    }

    /// Release every spare of `device`: its teardown, when no command buffer is left.
    pub(super) fn release(&self, device: &DeviceShared) {
        for pool in lock(&self.command_pools).spares.drain(..) {
            // SAFETY: no command buffer is left, so nothing of the pool is pending.
            unsafe { release_in_section(device, pool.raw) };
        }
        for pools in [&self.descriptor_pools, &self.fitted_pools] {
            for pool in lock(pools).drain(..) {
                // SAFETY: as above.
                unsafe { release_in_section(device, pool.raw) };
            }
        }
    }
}

/// Release `object` of `device` inside a section of the exit guard, which this thread may be
/// outside of (a command buffer completes on a handler thread): the device's functions are
/// reached inside it. Once the process is exiting, the object is left to its end.
///
/// # Safety
///
/// As [`release::object`]'s: nothing pending uses the object.
pub(super) unsafe fn release_in_section<T: Released>(device: &DeviceShared, object: T) {
    let Some(_busy) = exit_guard::enter() else {
        return;
    };
    // SAFETY: the caller's.
    unsafe { release::object(device.vk(), object) };
}

/// A command pool out of the spares, and command buffers taken out of it to be freed: those past
/// its recent uses or past [`HELD`], or all it held as a spare left idle or shed past [`HELD`]
/// (the module docs). Run on the device's free thread (`free_thread.rs`), else where the frees
/// were found.
pub(crate) struct PoolFrees {
    pool: CommandPool,
    buffers: Vec<vk::CommandBuffer>,
    /// Kept again as the oldest spare (a spare left idle, or shed past [`HELD`]), else as the
    /// newest.
    idle: bool,
    /// The pool is trimmed after each free (the module docs).
    trim: bool,
    /// The device's spare command pools.
    spares: Arc<Mutex<CommandPools>>,
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    freed_on: Arc<Mutex<Vec<String>>>,
}

impl PoolFrees {
    /// Free the command buffers through `device`, the device's functions taken inside a section of
    /// the exit guard, [`FREED_AT_ONCE`] at a time, the pool trimmed after each where the device
    /// trims (`release.rs`'s `PoolBuffers`), then keep the pool among the spares again.
    ///
    /// # Safety
    ///
    /// None of the command buffers is pending, and nothing else holds the pool.
    pub(crate) unsafe fn free(self, device: &ash::Device) {
        let PoolFrees {
            mut pool,
            buffers,
            idle,
            trim,
            spares,
            #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
            freed_on,
        } = self;
        for some in buffers.chunks(FREED_AT_ONCE) {
            // SAFETY: the caller's.
            unsafe {
                release::object(
                    device,
                    release::PoolBuffers {
                        pool: pool.raw,
                        buffers: some,
                        trim,
                    },
                )
            };
        }
        #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
        lock(&freed_on).push(std::thread::current().name().unwrap_or_default().to_owned());
        let mut pools = lock(&spares);
        pools.came_back(&mut pool);
        match idle {
            true => keep_idle_command_pool(&mut pools.spares, pool),
            false => keep_command_pool(&mut pools.spares, pool),
        }
    }
}

/// The oldest of the spare command pools `pools` left idle at `now` ([`CommandPool::idle`]),
/// taken out, to free its command buffers (the module docs).
pub(super) fn take_idle(pools: &mut Vec<CommandPool>, now: Instant) -> Option<CommandPool> {
    let at = pools.iter().position(|pool| pool.idle(now))?;
    Some(pools.remove(at))
}

/// The spares past [`HELD`] command buffers held with `given`, a pool given back, and the `out`
/// the pools out of the spares hold, of the spare command pools `pools` (oldest first): the
/// oldest that hold any, taken out to shed theirs, until all hold at most that many; and, where
/// that leaves more, `given`'s command buffers past them, taken out of it ([`CommandPool::past`])
/// (the module docs).
pub(super) fn past_held(
    pools: &mut Vec<CommandPool>,
    out: usize,
    given: &mut CommandPool,
) -> (Vec<CommandPool>, Vec<vk::CommandBuffer>) {
    let mut held = given.held() + out + pools.iter().map(CommandPool::held).sum::<usize>();
    let mut oldest = Vec::new();
    while held > HELD {
        let Some(at) = pools.iter().position(|pool| pool.held() > 0) else {
            break;
        };
        let pool = pools.remove(at);
        held -= pool.held();
        oldest.push(pool);
    }
    let others = held - given.held();
    (oldest, given.past(HELD.saturating_sub(others)))
}

/// Keep `pool`, a spare whose command buffers its idleness or [`HELD`] freed, among the spare
/// command pools `pools` as the oldest: a taker takes the newest, which still holds its command
/// buffers.
pub(super) fn keep_idle_command_pool(pools: &mut Vec<CommandPool>, pool: CommandPool) {
    pools.insert(0, pool);
}

/// The newest of the spare command pools `pools` made for queue family `family` that holds a
/// primary begun ahead, else the newest of them, taken out.
pub(super) fn take_command_pool(pools: &mut Vec<CommandPool>, family: u32) -> Option<CommandPool> {
    let of_family = |pool: &CommandPool| pool.family == family;
    let at = (pools
        .iter()
        .rposition(|pool| of_family(pool) && pool.begun.is_some()))
    .or_else(|| pools.iter().rposition(of_family))?;
    Some(pools.remove(at))
}

/// The spare command pool `raw` of `pools`, taken out to begin a primary ahead in
/// ([`Spares::begin_ahead`]), where it is still among them and may begin one
/// ([`CommandPool::begins_ahead`]).
pub(super) fn take_to_begin(
    pools: &mut Vec<CommandPool>,
    raw: vk::CommandPool,
) -> Option<CommandPool> {
    let at = (pools.iter()).rposition(|pool| pool.raw == raw && pool.begins_ahead())?;
    Some(pools.remove(at))
}

/// Keep `pool` among the spare command pools `pools`, newest last: every one, however many
/// command buffers it made and however many spares there are (the module docs).
pub(super) fn keep_command_pool(pools: &mut Vec<CommandPool>, pool: CommandPool) {
    pools.push(pool);
}

/// The newest of the spare descriptor pools `pools` (oldest first) with room for one set of
/// `needs`, and of at most `most` descriptors where given, taken out.
pub(super) fn take_descriptor_pool(
    pools: &mut VecDeque<DescriptorPool>,
    needs: &SetNeeds,
    most: Option<u64>,
) -> Option<DescriptorPool> {
    let at = pools.iter().rposition(|pool| {
        pool.room.fits(needs) && most.is_none_or(|most| pool.room.descriptors() <= most)
    })?;
    pools.remove(at)
}

/// Keep `pool` among the spare descriptor pools `pools` (oldest first): the pools to release, the
/// oldest while `pools` holds more than [`DESCRIPTOR_POOLS`] or [`DESCRIPTORS`], or `pool` itself
/// when it alone holds more descriptors than that.
pub(super) fn keep_descriptor_pool(
    pools: &mut VecDeque<DescriptorPool>,
    pool: DescriptorPool,
) -> Vec<vk::DescriptorPool> {
    if pool.room.descriptors() > DESCRIPTORS {
        return vec![pool.raw];
    }
    pools.push_back(pool);
    let mut past = Vec::new();
    let mut held: u64 = pools.iter().map(|pool| pool.room.descriptors()).sum();
    while pools.len() > DESCRIPTOR_POOLS || held > DESCRIPTORS {
        let Some(oldest) = pools.pop_front() else {
            break;
        };
        held -= oldest.room.descriptors();
        past.push(oldest.raw);
    }
    past
}
