//! Device tests of the commit path's work on the device the cfg selects, under the validation
//! layer: each holds a piece of work the runtime once repeated at every commit, or at every change
//! of its live buffers, to a bound, through the device's work counts (`Device::work_counts`, not
//! part of the API). Deterministic: they count, and time nothing.
//!
//! - **Command pools.** A command buffer that encoded nothing takes, for any barrier its
//!   submission records, the spare command pool the one before gave back; before, it made one in
//!   the scheduler's section, where no spare is taken. A command buffer's first encoder records
//!   into its spare pool's primary begun ahead, as the command buffer that gave the pool back
//!   completed; before, it began one on the thread that encodes (8 to 20 us on ANV).
//! - **No empty submissions.** A command buffer that encoded nothing and waits for and signals
//!   nothing makes no `vkQueueSubmit` where no first work of the device's waits
//!   (`WorkCounts::command_buffers_unsubmitted`): it completes in its queue's order, once the work
//!   submitted before it did. Before, it was one `vkQueueSubmit` of no `VkCommandBuffer`, its
//!   signals alone, and before that it carried one of the barrier to the host, and ANV on i915
//!   lists every `VkDeviceMemory` of the device in a kernel submission of commands (47 objects at
//!   32 host imports on an Iris Xe), where it lists two in one of none.
//! - **First work rides along.** A new texture's first work (`texture/first_move.rs`) is a
//!   `VkSubmitInfo` of the next submission's own `vkQueueSubmit`, a commit's or an internal one's
//!   (handing its `VkImage` out): one call each, which Mesa's drivers run as one kernel submission;
//!   on MoltenVK the internal one is followed by the relay's mark of no work, as every use of a
//!   queue outside a batch is there (`device/relay.rs`).
//!   Before, it was a `vkQueueSubmit` of its own, a second kernel submission listing every buffer
//!   object of the device on ANV over i915.
//! - **Fences.** Command buffers that each end in one submission complete at their fences on a
//!   device whose queues' failures are relayed, which takes one for every batch (MoltenVK,
//!   `device/relay.rs`): the queues' waiters wait for no timeline value; before, they waited for
//!   one each, which MoltenVK wakes through a shared event's listener. Elsewhere they end at their
//!   values, and the device makes no fence for them; before, each commit could make one, which
//!   NVIDIA's driver held for up to 1.9 s while other processes' work ran. On a device the runtime
//!   adopts, which relays none, every command buffer still ends at a fence where the driver reads
//!   a command buffer's query pools after its value, which only the ones that record a query
//!   command can suffer: the fence is the faster wait there all the same
//!   (`command_buffer/fences.rs`).
//! - **The live buffers.** A buffer made or released between commits costs the next resident set
//!   no ask of a listed buffer and no new buffer to upload it to; before, each asked every
//!   listed buffer for its pages again and uploaded the set to a new buffer.
//! - **Command pools, kept and reused**. Fills in rounds of more command buffers at once than any
//!   earlier round, released as they complete: the command pools made stay bounded by the peak in
//!   flight, each given back is kept and reused by a later round, and the fills read correctly.
//!   Previously a pool given back past the spares (or that made more than 64 command buffers) was
//!   destroyed as its command buffer completed, where NVIDIA 615.71.09's `vkDestroyCommandPool`
//!   intermittently never returned.
//! - **The end in the work.** A batch ends in its last command buffer: the barrier to the host
//!   (and a timed command buffer's end timestamp) go there as the batch ends, and its submission
//!   holds no end of its own (`WorkCounts::finales_folded`), an event's signal included; before,
//!   each batch's end was a command buffer of its own, about 9 us of GPU time on an Iris Xe. Only
//!   an event's signal that waits for another `VkQueue`'s earlier one, still pending, keeps an
//!   end apart, after that wait (`WorkCounts::finales_apart`). The host reads what either wrote
//!   once it completed.
//! - **Completed work, waited for.** A batch's submission waits for each timeline whose
//!   completed value the host saw move since a batch of its `VkQueue` last waited for it, each
//!   queue's and the device's own (`WorkCounts::completed_waits`), which makes what completed
//!   work wrote visible to it. A batch with nothing planned before its first encoder and no start
//!   timestamp to write then begins with no command buffer of barriers
//!   (`WorkCounts::bare_starts`); before, each began with one that made completed work's writes
//!   visible, about 12 us of GPU time on an Iris Xe. Where the driver's submissions stall on
//!   semaphore operations (`DriverFact::SemaphoreOperationsStallSubmits`) each still begins with
//!   that command buffer and waits for nothing completed. Copies read what completed fills on
//!   another command queue and a host copy into a texture wrote.
//! - **Batches of one queue, ordered by their ends** (`BatchOrderPolicy::Ends`, on a driver
//!   measured to run a queue's submissions one after another: [`orders_by_ends`]). Every batch
//!   ends with a barrier from every write to the host's accesses and to every later command of
//!   its `VkQueue`, so a batch there plans no barrier from an earlier batch's accesses, still in
//!   flight or not: a chain of command buffers on one command queue, each copying what the one
//!   before wrote while a long fill ahead of them runs, each begins with no command buffer of
//!   barriers (`WorkCounts::bare_starts`), and the last reads the first's fill. Under
//!   `BatchOrderPolicy::Exact`, and on any other driver, each but the first begins with one that
//!   orders it after the batch before, and reads the same bytes.
//! - **Addressable buffers**. The program's buffers take their device addresses from their memory
//!   blocks, so making them makes no buffer with the device-address usage beyond a new block's;
//!   before, each was one, which MoltenVK made resident one by one in every encoder whose pipeline
//!   reads pointers (2.5 ms a commit at 32768 live buffers on an Apple M4).

use super::degrade_notices::{on_device, on_device_with};
use super::{bound_buffers, encode_case, pipeline_state, public_case, CaseResources, WAIT};
use metal2vulkan::device::{BatchOrderPolicy, DriverFact};
use metal2vulkan::error::RuntimeDriverDefect;
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use metal2vulkan::vk::DriverId;
use metal2vulkan_runtime::{
    BlitOption, Buffer, CommandBuffer, CommandBufferStatus, CommandQueue, Device, EventRef, Origin,
    Region, ResourceOptions, Size, TextureDescriptor,
};

/// A kernel whose root table searches the device's live buffers: its pointer walk reaches
/// buffers no encoder named.
const POINTER_WALKER: &str = "walk-a-device-pointer-chain-through-a-loop-phi";

/// Commit `command_buffer` and wait for it to complete.
fn finish(command_buffer: &CommandBuffer) {
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
}

/// The submissions of no work the runtime adds after a use of `device`'s queue outside a batch of
/// its own, an internal submission among them: one where it relays the driver's failures, a mark
/// with a fence of its own under the queue's lock (`device/relay.rs`), and none elsewhere. The
/// relay is the runtime's own branch on `DriverFact::QueueFailuresReportedByMessenger`, which on
/// a device the lane made is enough for it (the runtime's `relay_facts` test holds the two equal
/// there), so the fact is read rather than measured: the only use of a queue a program can make
/// that submits nothing of its own, `CommandQueue::with_vk_queue`, also takes the freed-memory
/// parking's mark (`freed_memory.rs`), which no use of the runtime's own takes, and so counts one
/// too many here.
pub(super) fn relay_marks(device: &Device) -> u64 {
    let relays =
        (device.properties().driver_facts).contains(&DriverFact::QueueFailuresReportedByMessenger);
    u64::from(relays)
}

/// A command buffer of a blit that fills `buffer`.
fn fill(queue: &CommandQueue, buffer: &Buffer) -> CommandBuffer {
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.fill_buffer(buffer, 0..buffer.length(), 0x5a).unwrap();
    blit.end_encoding().unwrap();
    command_buffer
}

/// Empty command buffers one after another make no command pool past the first: each takes the
/// spare the one before gave back as it completed.
#[test]
fn empty_command_buffers_take_the_spare_command_pool() {
    on_device(|device, _| {
        let queue = device.new_command_queue().unwrap();
        // The first may make the pool the later ones take.
        finish(&queue.command_buffer().unwrap());
        let before = device.work_counts();
        for _ in 0..8 {
            finish(&queue.command_buffer().unwrap());
        }
        let after = device.work_counts();
        assert_eq!(after.command_pools_made, before.command_pools_made);
    });
}

/// A command buffer's first encoder takes the primary its spare command pool holds begun ahead
/// (`spares.rs`): the command buffer before, which gave the pool back, began it as it completed,
/// where no thread waited for it (`WorkCounts::primaries_begun_ahead`), and the encoder begins
/// none of its own (`WorkCounts::encoder_primaries_begun_ahead`). The fills still read correctly.
#[test]
fn a_first_encoder_takes_the_primary_its_spare_pool_began_ahead() {
    const ROUNDS: u8 = 4;
    on_device(|device, _| {
        let queue = device.new_command_queue().unwrap();
        let buffer = device.new_buffer(256, ResourceOptions::shared()).unwrap();
        for round in 0..ROUNDS {
            let before = device.work_counts();
            let first = fill(&queue, &buffer);
            first.commit().unwrap();
            // Begun on the completing thread, which a wait for the command buffer would skip:
            // read from the counts instead, which count it once its pool is among the spares
            // again (the begin runs with the pool out of them), so the next command buffer
            // takes that pool.
            let deadline = std::time::Instant::now() + WAIT;
            while device.work_counts().primaries_begun_ahead == before.primaries_begun_ahead {
                assert!(
                    std::time::Instant::now() < deadline,
                    "round {round}: no primary begun ahead after a command buffer completed"
                );
                std::thread::yield_now();
            }
            assert_eq!(first.status(), CommandBufferStatus::Completed);
            let begun = device.work_counts();
            let command_buffer = queue.command_buffer().unwrap();
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            blit.fill_buffer(&buffer, 0..buffer.length(), round + 1)
                .unwrap();
            blit.end_encoding().unwrap();
            finish(&command_buffer);
            let after = device.work_counts();
            assert_eq!(
                after.encoder_primaries_begun_ahead - begun.encoder_primaries_begun_ahead,
                1,
                "round {round}: the encoder took its pool's primary begun ahead"
            );
            assert!(
                bytes(&buffer).iter().all(|&b| b == round + 1),
                "round {round}: the fill read back"
            );
        }
    });
}

/// Empty command buffers one after another make no `vkQueueSubmit` (the module docs).
#[test]
fn empty_command_buffers_make_no_submission() {
    on_device(|device, _| {
        let queue = device.new_command_queue().unwrap();
        // The first takes any first work the device's creation left.
        finish(&queue.command_buffer().unwrap());
        let before = device.work_counts();
        for _ in 0..8 {
            finish(&queue.command_buffer().unwrap());
        }
        let after = device.work_counts();
        assert_eq!(after.queue_submissions, before.queue_submissions);
        assert_eq!(
            after.command_buffers_unsubmitted - before.command_buffers_unsubmitted,
            8
        );
        assert_eq!(
            after.command_buffers_submitted,
            before.command_buffers_submitted
        );
    });
}

/// A commit after a new `Private` texture was made, and the internal submission that hands
/// another's `VkImage` out, are one `vkQueueSubmit` each, the texture's first work in it: the
/// empty command buffer's submission carries the first work's `VkCommandBuffer` alone (the module
/// docs). The internal submission, a use of the queue outside a batch, is followed by the relay's
/// mark of no work where the runtime relays the driver's failures ([`relay_marks`]).
#[test]
fn a_new_texture_s_first_work_rides_in_the_next_submission() {
    on_device(|device, _| {
        let queue = device.new_command_queue().unwrap();
        finish(&queue.command_buffer().unwrap());
        let marks = relay_marks(device);
        let texture = || {
            let mut descriptor =
                TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 64, 64, false);
            descriptor.usage = TextureUsage::SHADER_READ;
            descriptor.resource_options = ResourceOptions::private();
            device.new_texture(&descriptor).unwrap()
        };
        let before = device.work_counts();
        let committed = texture();
        finish(&queue.command_buffer().unwrap());
        let after = device.work_counts();
        assert_eq!(after.queue_submissions - before.queue_submissions, 1);
        assert_eq!(
            after.command_buffers_submitted - before.command_buffers_submitted,
            1
        );
        let handed = texture();
        let before = device.work_counts();
        // SAFETY: the image is only read for its handle; the texture lives past the read.
        let image = unsafe { handed.vk_image() };
        assert!(image.is_some(), "a texture of its own image");
        let after = device.work_counts();
        assert_eq!(after.internal_submissions - before.internal_submissions, 1);
        assert_eq!(
            after.queue_submissions - before.queue_submissions,
            1 + marks
        );
        drop((committed, handed));
    });
}

/// Rounds of 32 command buffers encoded, then committed, together, each a fill of its own
/// buffer, released as they complete: the runtime keeps every command pool given back while the
/// device lives and reuses it, so across the rounds the command pools made stay bounded by the peak
/// in flight (at most one round's), and every round's fills still read correctly. Previously a pool
/// given back past the device's spares was destroyed as its command buffer completed, where NVIDIA
/// 615.71.09's `vkDestroyCommandPool` intermittently never returned (the consumer's hang); and
/// MoltenVK 1.4.1 fills through `vkCmdUpdateBuffer` (`word_fill.rs`), whose scratch buffer a
/// completed handler gives back to its command pool's pool, so a program that destroyed its pool at
/// that value died (`mutex lock failed`).
#[test]
fn fills_across_rounds_reuse_a_bounded_set_of_command_pools() {
    const IN_FLIGHT: usize = 32;
    const ROUNDS: usize = 16;
    on_device(|device, _| {
        let queue = device.new_command_queue().unwrap();
        let buffers: Vec<Buffer> = (0..IN_FLIGHT)
            .map(|_| device.new_buffer(256, ResourceOptions::shared()).unwrap())
            .collect();
        let before = device.work_counts();
        for round in 0..ROUNDS {
            let byte = round as u8 + 1;
            // All encoded before any is committed, so each holds a command pool of its own.
            let committed: Vec<CommandBuffer> = buffers
                .iter()
                .map(|buffer| {
                    let command_buffer = queue.command_buffer().unwrap();
                    let mut blit = command_buffer.blit_command_encoder().unwrap();
                    blit.fill_buffer(buffer, 0..buffer.length(), byte).unwrap();
                    blit.end_encoding().unwrap();
                    command_buffer
                })
                .collect();
            for command_buffer in &committed {
                command_buffer.commit().unwrap();
            }
            for command_buffer in &committed {
                command_buffer.wait_until_completed_for(WAIT).unwrap();
                assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
            }
            drop(committed);
            for buffer in &buffers {
                let contents = buffer.contents().expect("a shared buffer");
                // SAFETY: the buffer holds `length` bytes at `contents`, alive with the buffer;
                // no command buffer uses it now.
                let bytes = unsafe {
                    std::slice::from_raw_parts(contents.as_ptr(), buffer.length() as usize)
                };
                assert!(bytes.iter().all(|&b| b == byte), "round {round}");
            }
        }
        let after = device.work_counts();
        let made = (after.command_pools_made - before.command_pools_made) as usize;
        eprintln!(
            "FILL-POOL-REUSE {}: {made} command pools made for {} command buffers over {ROUNDS} \
             rounds",
            device.name(),
            ROUNDS * IN_FLIGHT
        );
        // The pools made are bounded by the peak in flight and do not grow with the rounds: each
        // given back is kept and reused, not destroyed and remade. Before the fix this was about
        // `ROUNDS * (IN_FLIGHT - 8)` as pools past the spares were destroyed each round.
        assert!(
            made <= IN_FLIGHT,
            "{made} command pools made, more than the peak in flight"
        );
    });
}

/// The bytes of `buffer`, a shared one no command buffer uses now.
fn bytes(buffer: &Buffer) -> Vec<u8> {
    let contents = buffer.contents().expect("a shared buffer");
    // SAFETY: the buffer holds `length` bytes at `contents`, alive with the buffer; no command
    // buffer uses it now.
    unsafe { std::slice::from_raw_parts(contents.as_ptr(), buffer.length() as usize) }.to_vec()
}

/// A fill's batch ends in the fill's own command buffer, one that signals an event too (the
/// module docs): the host reads each fill once it completed. A fill on a second queue that signals
/// the next value of an event a first queue's fill signalled just before ends once, in its work
/// or, where its signal waits for the first's still pending on another `VkQueue`, apart; the
/// event reaches the second value.
#[test]
fn a_batch_ends_in_its_last_command_buffer() {
    on_device(|device, _| {
        let queue = device.new_command_queue().unwrap();
        let buffer = device.new_buffer(4096, ResourceOptions::shared()).unwrap();
        // The first takes any first work the device's creation left.
        finish(&fill(&queue, &buffer));
        let filled = |byte: u8, signal: Option<u64>| {
            let command_buffer = queue.command_buffer().unwrap();
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            blit.fill_buffer(&buffer, 0..buffer.length(), byte).unwrap();
            blit.end_encoding().unwrap();
            let event = device.new_event().unwrap();
            if let Some(value) = signal {
                command_buffer
                    .encode_signal_event(EventRef::from(&event), value)
                    .unwrap();
            }
            let before = device.work_counts();
            finish(&command_buffer);
            let after = device.work_counts();
            assert!(bytes(&buffer).iter().all(|&b| b == byte), "{byte:#x}");
            (
                after.finales_folded - before.finales_folded,
                after.finales_apart - before.finales_apart,
            )
        };
        assert_eq!(filled(0x11, None), (1, 0), "a fill");
        assert_eq!(
            filled(0x22, Some(1)),
            (1, 0),
            "a fill that signals an event"
        );
        assert_eq!(filled(0x33, None), (1, 0), "a fill after it");
        // Across two queues, the second's signal right after the first's.
        let other = device.new_command_queue().unwrap();
        let event = device.new_shared_event().unwrap();
        let large = device
            .new_buffer(64 << 20, ResourceOptions::private())
            .unwrap();
        let before = device.work_counts();
        let signalled: Vec<CommandBuffer> = [(&queue, &large, 1), (&other, &buffer, 2)]
            .into_iter()
            .map(|(on, target, value)| {
                let command_buffer = on.command_buffer().unwrap();
                let mut blit = command_buffer.blit_command_encoder().unwrap();
                blit.fill_buffer(target, 0..target.length(), 0x44).unwrap();
                blit.end_encoding().unwrap();
                command_buffer
                    .encode_signal_event(EventRef::from(&event), value)
                    .unwrap();
                command_buffer.commit().unwrap();
                command_buffer
            })
            .collect();
        for command_buffer in &signalled {
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        }
        let after = device.work_counts();
        assert!(bytes(&buffer).iter().all(|&b| b == 0x44), "the second fill");
        assert!(
            event.wait_until_signaled_value(2, WAIT),
            "{}",
            event.signaled_value()
        );
        let ends = (after.finales_folded - before.finales_folded)
            + (after.finales_apart - before.finales_apart);
        assert_eq!(ends, 2, "one end each");
    });
}

/// On a device made with `gpu_times_from_host`, whose command buffers write no start timestamp, a
/// batch with nothing planned before its first encoder begins with no command buffer of barriers,
/// and its submission waits for the timelines the host saw complete (the module docs): fills on
/// one command queue and copies of what each wrote on another each start bare and wait at least
/// once, and each copy reads its fill's bytes. Where the driver's submissions stall on semaphore
/// operations (`DriverFact::SemaphoreOperationsStallSubmits`), each instead begins with a
/// command buffer of barriers and waits for nothing completed, and reads the same bytes. A copy
/// out of a texture after `replace_region`, which may be the device's own submission or the
/// staging queue's, reads its bytes.
#[test]
fn batches_with_nothing_planned_wait_for_completed_work_instead_of_a_barrier() {
    on_device_with(
        |options| options.gpu_times_from_host = true,
        |device, _| {
            let by_barrier = device
                .properties()
                .driver_facts
                .contains(&DriverFact::SemaphoreOperationsStallSubmits);
            // Bare starts and at least one wait each, or none of either.
            let form = |what: &str, (bare, waits): (u64, u64)| match by_barrier {
                false => {
                    assert_eq!(bare, 1, "{what}: a bare start");
                    assert!(waits >= 1, "{what}: {waits} waits for completed work");
                }
                true => assert_eq!((bare, waits), (0, 0), "{what}: a barrier, no wait"),
            };
            let filling = device.new_command_queue().unwrap();
            let copying = device.new_command_queue().unwrap();
            let source = device.new_buffer(4096, ResourceOptions::private()).unwrap();
            let read = device.new_buffer(4096, ResourceOptions::shared()).unwrap();
            // The first takes any first work the device's creation left, and each buffer's.
            finish(&fill(&filling, &source));
            finish(&fill(&copying, &read));
            // The batches each began bare, and the waits for completed work they held.
            let counted = |command_buffer: CommandBuffer| {
                let before = device.work_counts();
                finish(&command_buffer);
                let after = device.work_counts();
                (
                    after.bare_starts - before.bare_starts,
                    after.completed_waits - before.completed_waits,
                )
            };
            for round in 0..4u8 {
                let byte = 0x10 + round;
                let command_buffer = filling.command_buffer().unwrap();
                let mut blit = command_buffer.blit_command_encoder().unwrap();
                blit.fill_buffer(&source, 0..source.length(), byte).unwrap();
                blit.end_encoding().unwrap();
                form(&format!("fill {round}"), counted(command_buffer));
                let command_buffer = copying.command_buffer().unwrap();
                let mut blit = command_buffer.blit_command_encoder().unwrap();
                blit.copy_from_buffer_to_buffer(&source, 0, &read, 0, source.length())
                    .unwrap();
                blit.end_encoding().unwrap();
                form(&format!("copy {round}"), counted(command_buffer));
                assert!(
                    bytes(&read).iter().all(|&b| b == byte),
                    "copy {round}: the fill's {byte:#x}"
                );
            }
            // A host copy into a texture, then a copy out of it.
            const SIDE: u64 = 16;
            let mut descriptor =
                TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
            descriptor.usage = TextureUsage::SHADER_READ;
            let texture = device.new_texture(&descriptor).unwrap();
            let texels: Vec<u8> = (0..SIDE * SIDE * 4).map(|i| (i * 7 % 251) as u8).collect();
            let before = device.work_counts();
            texture
                .replace_region(Region::new_2d(0, 0, SIDE, SIDE), 0, 0, &texels, SIDE * 4, 0)
                .unwrap();
            let command_buffer = copying.command_buffer().unwrap();
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            blit.copy_from_texture_to_buffer(
                &texture,
                0,
                0,
                Origin::new(0, 0, 0),
                Size::new(SIDE, SIDE, 1),
                &read,
                0,
                SIDE * 4,
                0,
                BlitOption::None,
            )
            .unwrap();
            blit.end_encoding().unwrap();
            finish(&command_buffer);
            let after = device.work_counts();
            eprintln!(
                "COMPLETED-WAITS {}: a host copy into a texture: {} internal submissions, {} \
                 staging ones; the copy out: {} bare starts, {} waits for completed work",
                device.name(),
                after.internal_submissions - before.internal_submissions,
                after.staging_submissions - before.staging_submissions,
                after.bare_starts - before.bare_starts,
                after.completed_waits - before.completed_waits,
            );
            assert_eq!(
                &bytes(&read)[..texels.len()],
                &texels[..],
                "the texture's bytes"
            );
        },
    );
}

/// Whether a device of the driver orders its batches by their ends under `batch_order`: the
/// runtime's table of drivers measured to run a queue's submissions one after another
/// (`batch_order.rs`).
pub(super) fn orders_by_ends(device: &Device, batch_order: BatchOrderPolicy) -> bool {
    batch_order == BatchOrderPolicy::Ends
        && matches!(
            device.properties().description.driver_id,
            Some(
                DriverId::INTEL_OPEN_SOURCE_MESA
                    | DriverId::MESA_RADV
                    | DriverId::NVIDIA_PROPRIETARY
            )
        )
}

/// Under the default, on a device made with `gpu_times_from_host` (so no start timestamp is
/// written before a batch's work): a long fill, waiting for a shared event, then a chain of
/// command buffers on the same command queue, each copying what the one before wrote, the first
/// copying the long one's last fill; the host raises the event once all are committed. Where the
/// device orders batches by their ends, every batch begins with no command buffer of barriers, as
/// the end of the batch before orders it (the module docs), and the chain's last copy reads the
/// long one's fill. Elsewhere
/// [`batches_of_one_queue_in_the_exact_form_begin_with_their_barriers`]'s.
#[test]
fn batches_of_one_queue_ordered_by_their_ends_begin_with_no_barrier() {
    chained_batches(BatchOrderPolicy::Ends);
}

/// The same under `BatchOrderPolicy::Exact`: each batch of the chain begins with a command buffer
/// of barriers for the copy before it exactly where that copy was in flight as the batch was
/// planned, and the last reads the same bytes.
#[test]
fn batches_of_one_queue_in_the_exact_form_begin_with_their_barriers() {
    chained_batches(BatchOrderPolicy::Exact);
}

/// The two tests above under `batch_order`. Where the driver's submissions stall on semaphore
/// operations (`DriverFact::SemaphoreOperationsStallSubmits`), every batch begins with its command
/// buffer of barriers under either.
///
/// The scheduler holds the long fill on the host until the event is raised (no batch is submitted
/// waiting for a value the host may set, `event_timeline.rs`), and the chain behind it, so every
/// batch is planned after the raise, while the work before it runs on the GPU or after it
/// completed, as the GPU's speed and the host's planning fall. Under the exact form a batch begins
/// with no barrier exactly where the host saw no earlier batch of its queue pending as it planned
/// it: `WorkCounts::bare_starts` equals `batches_with_none_pending`, both counted from the
/// completed value the plan took, the long fill always among them (what it depends on completed).
/// A run where the GPU finished every batch before the next was planned exercises only that side,
/// and says so; the planner's barrier for work in flight is `tests::tracker`'s
/// `a_chain_of_copies_orders_each_after_the_one_before_only_while_it_is_in_flight`, by
/// construction. The long fill writes 16 GiB, so the barrier side usually runs: on the Radeon RX
/// 7900 XTX every copy of 24 runs was planned behind work in flight.
fn chained_batches(batch_order: BatchOrderPolicy) {
    const CHAIN: usize = 8;
    const LONG_FILLS: u64 = 1024;
    on_device_with(
        |options| {
            options.gpu_times_from_host = true;
            options.policy.batch_order = batch_order;
        },
        |device, _| {
            let by_barrier = device
                .properties()
                .driver_facts
                .contains(&DriverFact::SemaphoreOperationsStallSubmits);
            let by_ends = orders_by_ends(device, batch_order);
            let queue = device.new_command_queue().unwrap();
            let scratch = device
                .new_buffer(16 << 20, ResourceOptions::private())
                .unwrap();
            let links: Vec<Buffer> = (0..CHAIN)
                .map(|_| device.new_buffer(4096, ResourceOptions::private()).unwrap())
                .collect();
            let read = device.new_buffer(4096, ResourceOptions::shared()).unwrap();
            // Each buffer's first work, completed.
            for buffer in links.iter().chain([&scratch, &read]) {
                finish(&fill(&queue, buffer));
            }
            let before = device.work_counts();
            let event = device.new_shared_event().unwrap();
            let long = queue.command_buffer().unwrap();
            long.encode_wait_for_event(EventRef::from(&event), 1)
                .unwrap();
            let mut blit = long.blit_command_encoder().unwrap();
            for round in 0..LONG_FILLS {
                blit.fill_buffer(&scratch, 0..scratch.length(), round as u8)
                    .unwrap();
            }
            blit.fill_buffer(&links[0], 0..4096, 0xc3).unwrap();
            blit.end_encoding().unwrap();
            long.commit().unwrap();
            let chain: Vec<CommandBuffer> = (1..CHAIN)
                .map(|at| (&links[at - 1], &links[at]))
                .chain([(&links[CHAIN - 1], &read)])
                .map(|(from, to)| {
                    let command_buffer = queue.command_buffer().unwrap();
                    let mut blit = command_buffer.blit_command_encoder().unwrap();
                    blit.copy_from_buffer_to_buffer(from, 0, to, 0, 4096)
                        .unwrap();
                    blit.end_encoding().unwrap();
                    command_buffer.commit().unwrap();
                    command_buffer
                })
                .collect();
            event.set_signaled_value(1);
            for command_buffer in chain.iter().chain([&long]) {
                command_buffer.wait_until_completed_for(WAIT).unwrap();
                assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
            }
            let after = device.work_counts();
            let bare = after.bare_starts - before.bare_starts;
            let none_pending = after.batches_with_none_pending - before.batches_with_none_pending;
            let exact = !by_barrier && !by_ends;
            let only_bare = match exact && none_pending == CHAIN as u64 + 1 {
                true => "; only the bare side ran: each batch's work before it had completed",
                false => "",
            };
            eprintln!(
                "CHAINED-BATCHES {} {batch_order:?}: {bare} bare starts of {} batches, \
                 {none_pending} planned with no earlier work in flight{only_bare}",
                device.name(),
                CHAIN + 1,
            );
            // Each copy plans a barrier for the copy before it where that one was in flight,
            // unless batches are ordered by their ends.
            let expected = match (by_barrier, by_ends) {
                (true, _) => 0,
                (false, true) => CHAIN as u64 + 1,
                (false, false) => none_pending,
            };
            assert_eq!(bare, expected, "bare starts");
            assert!(
                bytes(&read).iter().all(|&byte| byte == 0xc3),
                "the long fill's bytes through the chain"
            );
        },
    );
}

/// Command buffers that each end in one submission, empty or of a blit, complete at their fences
/// on a device whose queues' failures are relayed (`device/relay.rs`), which takes one for every
/// batch, and on one whose driver reads a command buffer's query pools after its value
/// (`RuntimeDriverDefect::ReadsCommandBuffersAfterTheirSignal`), which takes one for every
/// command buffer's last batch: the queues' waiters wait for no timeline value. Elsewhere they
/// end at their values alone: the device makes no fence for any of them, from its creation on, and
/// the queues' waiters wait for their values.
#[test]
fn command_buffers_end_at_fences_only_where_the_driver_reads_them_later() {
    on_device(|device, _| {
        let queue = device.new_command_queue().unwrap();
        let buffer = device.new_buffer(4096, ResourceOptions::default()).unwrap();
        let relayed = relay_marks(device) > 0;
        let before = device.work_counts();
        for _ in 0..4 {
            finish(&queue.command_buffer().unwrap());
            finish(&fill(&queue, &buffer));
        }
        let after = device.work_counts();
        let late_reads = (device.properties().driver_defects)
            .contains(&RuntimeDriverDefect::ReadsCommandBuffersAfterTheirSignal);
        eprintln!(
            "COMMAND-BUFFER-ENDS {}: reads after the value {late_reads}, failures relayed \
             {relayed}, {} fences made, {} fence waits and {} timeline waits for 8 command \
             buffers that record no query",
            device.name(),
            after.fences_made,
            after.fence_waits - before.fence_waits,
            after.timeline_waits - before.timeline_waits
        );
        // `fences::ends_at_fence` of the queue these command buffers were committed to.
        if relayed || late_reads {
            assert!(after.fence_waits > before.fence_waits, "fence waits");
            assert_eq!(
                after.timeline_waits, before.timeline_waits,
                "timeline waits"
            );
        } else {
            assert_eq!(after.fence_waits, before.fence_waits, "fence waits");
            assert!(
                after.timeline_waits > before.timeline_waits,
                "timeline waits"
            );
            assert_eq!(after.fences_made, 0, "fences made");
        }
    });
}

/// With 64 buffers listed, a buffer released and another made between each pointer walk's
/// commit puts the walk's resident set into the buffer of the set it replaces.
#[test]
fn a_buffer_made_or_released_costs_the_live_buffers_no_upload() {
    on_device(|device, _| {
        let queue = device.new_command_queue().unwrap();
        let case = public_case(POINTER_WALKER).unwrap();
        let state = pipeline_state(device, &case).unwrap();
        let resources = CaseResources::new(device, &case).unwrap();
        let bound = bound_buffers(device, &case.compute, &resources).unwrap();
        let walk = || {
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer.compute_command_encoder().unwrap();
            encode_case(&mut encoder, &state, &case.compute, &bound, &resources).unwrap();
            encoder.end_encoding().unwrap();
            finish(&command_buffer);
        };
        let new_buffer = || device.new_buffer(4096, ResourceOptions::default()).unwrap();
        let mut listed: Vec<Buffer> = (0..64).map(|_| new_buffer()).collect();
        // The first set, uploaded to a buffer made for it.
        walk();
        let before = device.work_counts();
        for _ in 0..8 {
            listed.remove(0);
            listed.push(new_buffer());
            walk();
        }
        let after = device.work_counts();
        assert_eq!(after.live_tables_uploaded, before.live_tables_uploaded);
        drop(listed);
    });
}

/// 64 `Shared` and 64 `Private` buffers of the program's make at most one buffer with the
/// device-address usage per memory class they or the runtime's own buffers took a new block of:
/// each takes its address from its block's buffer of addresses (`memory.rs`).
#[test]
fn buffers_made_take_their_addresses_from_their_blocks() {
    on_device(|device, _| {
        let before = device.work_counts();
        let made: Vec<Buffer> = [ResourceOptions::shared(), ResourceOptions::private()]
            .into_iter()
            .flat_map(|options| (0..64).map(move |_| options))
            .map(|options| device.new_buffer(256, options).unwrap())
            .collect();
        let after = device.work_counts();
        let addressable = after.addressable_buffers_made - before.addressable_buffers_made;
        assert!(
            addressable <= 4,
            "{} buffers made {addressable} buffers with the device-address usage",
            made.len()
        );
    });
}
