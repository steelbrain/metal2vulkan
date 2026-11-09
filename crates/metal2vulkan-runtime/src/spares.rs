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
//!   only as the device is torn down, after its queues are idle.
//!   The spares stay bounded all the same: a command buffer makes a pool only where no spare of its
//!   queue's family is free, so the device holds at most as many as its command buffers held at
//!   once (Metal bounds those by each queue's `maxCommandBufferCount`), each holding what its
//!   largest command buffer recorded. Before, a pool given back past 8 spares, or one that made
//!   more than 64 command buffers, was released on the handler thread that completed its command
//!   buffer; on NVIDIA's driver 615.71.09 that `vkDestroyCommandPool` intermittently never returned
//!   (the driver spinning, with its own `[vkrt] Analysis` thread), so the command buffer never
//!   completed and its queue stalled.
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
use std::sync::Mutex;

/// The most spare descriptor pools a device keeps (of each list, a command buffer's arenas' and
/// the fitted arenas'): a few frames' worth of a window server's (the module docs).
pub(super) const DESCRIPTOR_POOLS: usize = 256;

/// The most descriptors the spare descriptor pools hold together.
pub(super) const DESCRIPTORS: u64 = 1 << 18;

/// The index of `level`'s list in a [`CommandPool`]: primary 0, secondary 1.
fn level_index(level: vk::CommandBufferLevel) -> usize {
    usize::from(level == vk::CommandBufferLevel::SECONDARY)
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
        }
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

/// A device's spares.
#[derive(Default)]
pub(super) struct Spares {
    command_pools: Mutex<Vec<CommandPool>>,
    /// The command buffers' arenas', oldest first.
    descriptor_pools: Mutex<VecDeque<DescriptorPool>>,
    /// The fitted arenas' (the module docs), oldest first.
    fitted_pools: Mutex<VecDeque<DescriptorPool>>,
}

impl Spares {
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
        let Some(mut pool) = take_command_pool(&mut lock(&self.command_pools), family) else {
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
    /// recording (the module docs). A pool that made no secondary command buffer makes one
    /// first. Never released here.
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
        pool.recording = !settled;
        pool.stock(device);
        keep_command_pool(&mut lock(&self.command_pools), pool);
    }

    /// Begin a primary ahead in the spare command pool `raw` of `device`, where it is still a
    /// spare, settled, and holds none begun (the module docs): taken out of the spares under
    /// their lock, begun with no lock held, so a taker meanwhile takes another spare rather than
    /// wait for the begin, and kept again, the newest. Counted (`WorkCounts::primaries_begun_ahead`)
    /// once it is kept again, so a taker that saw the count finds the pool among the spares.
    pub(super) fn begin_ahead(&self, device: &DeviceShared, raw: vk::CommandPool) {
        let Some(mut pool) = take_to_begin(&mut lock(&self.command_pools), raw) else {
            return;
        };
        let begun = pool.begin_ahead(device);
        keep_command_pool(&mut lock(&self.command_pools), pool);
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
        for pool in lock(&self.command_pools).drain(..) {
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
