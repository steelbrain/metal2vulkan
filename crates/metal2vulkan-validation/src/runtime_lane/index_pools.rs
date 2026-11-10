//! Device tests of the index pool blocks (the runtime's `gpu_pools.rs`) on the device the cfg
//! selects, under the validation layer.
//!
//! Each command buffer here draws one indexed indirect draw of points with rasterization disabled
//! (the owned fixture `ob3_index_buffer`), from two 32-bit indices inside their buffer, on a
//! device opened without the robustness features, so every device's draw takes the index bounds
//! pass (`indirect-index-bounds-pass`), whose first pass in a command buffer leases its queue's
//! index pool block as it encodes and holds it until the command buffer completed.
//!
//! - The command buffers of one queue share its block, two in flight at once round after round,
//!   and the queue makes it once (`WorkCounts::gpu_pool_blocks_made`); another queue makes its
//!   own; a block no command buffer held for two seconds goes as a later one's lease ends, and a
//!   queue's goes with the queue: to the device's spare where it has none, which the next queue
//!   that needs a block takes instead of making one, else it is released
//!   (`WorkCounts::gpu_pool_blocks_released`). Before, a block that went was released, and the
//!   next queue to need one made one again. Before that, each command
//!   buffer in flight held a block of its own, made where none was spare: on the M3 Max (MoltenVK
//!   1.4.1, macOS 27.0.1) a block is 1 GiB, whose making took 24 to 39 ms inside
//!   `vkBindBufferMemory`, and wired its 1 GiB.
//! - A command buffer's completed handler runs while another thread makes a block: its uploads
//!   and pools go after its handlers. Before, its uploads went first, and on MoltenVK their
//!   `vkDestroyBuffer` waited for the lock that making holds, so the handler ran as the making
//!   ended, 20 to 27 ms after the command buffer completed.
//! - The command buffer returns from its submission before the making begins
//!   (`wait_until_scheduled`), and its draw's instances take a few milliseconds of GPU time, so it
//!   completes during the making: the runtime watches a batch for its completion from its
//!   `vkQueueSubmit`'s return on, and a driver may hold that return for the making. On the
//!   RTX 5080 Laptop GPU (NVIDIA 615.71.09, Linux) a `vkQueueSubmit` the scheduler's thread
//!   called while the making's `vkAllocateMemory` ran returned 0.01 to 0.25 ms after that did, in
//!   25 of 40 rounds of a probe (allocations of a 254.7 MiB block, 3.1 to 35.7 ms, beside another
//!   process's GPU work), twice after the GPU had ended the batch (23.9 ms into a 29.4 ms making,
//!   and 0.3 ms into a 3.7 ms one); another thread's `vkWaitSemaphores`, semaphore reads and fence
//!   reads took at most 0.03 ms during raw allocations of that size (1.9 to 213.6 ms). In the
//!   other 15 rounds the submission returned before the allocation began and the handler ran
//!   during it; in all 40 the handler ran 0.07 to 1.44 ms after the later of the two.

use super::{
    at_plan, buffer_with, layer_reported_nothing_at_all, validated_device_with, Messages,
    ScratchCache, WAIT,
};
use crate::air_library::AirLibrary;
use crate::coverage::ROBUSTNESS_FEATURES;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::metal::{IndexType, PrimitiveType};
use metal2vulkan::pipeline::{RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::DeviceOptions;
use metal2vulkan_runtime::{
    host_time, Buffer, CommandBuffer, CommandQueue, Device, PipelineOptions, RenderPassDescriptor,
    RenderPipelineState,
};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// The rounds of two command buffers in flight at once.
const ROUNDS: usize = 6;

/// The runtime's quiet period: a block no command buffer held this long goes as a later lease
/// ends (`gpu_pools::QUIET`).
const QUIET: Duration = Duration::from_secs(2);

/// The rounds of a command buffer completing while another queue makes its block. A round tells
/// only where the making took `MAKING_TELLS` or more and the command buffer completed in its first
/// half: on the RTX 5080 Laptop GPU a block's making took 4.9 to 189.5 ms in 240 rounds beside
/// another process's GPU work.
const MAKING_ROUNDS: usize = 12;

/// The shortest making of a block that tells a handler that waited for it from one that did not
/// (on the M3 Max a block took 24 to 39 ms; a handler that waits for nothing runs within a
/// millisecond of its command buffer's completion).
const MAKING_TELLS: f64 = 0.010;

/// The GPU time of a round's first command buffer, about: it completes this long into the making,
/// in its first half where the making takes `MAKING_TELLS` or more.
const WORK: f64 = 0.003;

/// The most instances of the draw that measure its GPU time.
const MOST_INSTANCES: u32 = 1 << 24;

/// The words' bytes.
fn words(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

/// The device, opened without the robustness features, the fixture's vertex-only pipeline state
/// and the draw's buffers.
struct Lane {
    device: Device,
    pipeline: RenderPipelineState,
    seen: Buffer,
    indices: Buffer,
    arguments: Buffer,
}

impl Lane {
    fn new(messages: &Messages, cache: &ScratchCache) -> Lane {
        Lane::new_with(|_| {}, messages, cache)
    }

    /// [`Lane::new`], `adjust` applied to the options of each device it makes (`at_plan` gives them
    /// their queue plan).
    fn new_with(adjust: fn(&mut DeviceOptions), messages: &Messages, cache: &ScratchCache) -> Lane {
        let device = validated_device_with(TEST_DEVICE, messages, cache.path(), |options| {
            options.withheld_features = ROBUSTNESS_FEATURES.into_iter().collect();
            adjust(options)
        })
        .unwrap();
        let bitcode = llvm_assemble(&engine_dir().join("ob3_index_buffer.ll")).unwrap();
        let library = device.new_library_with_air(&bitcode).unwrap();
        let function = library.new_function_with_name("ob3_index_buffer").unwrap();
        let mut descriptor =
            RenderPipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
        descriptor.rasterization_enabled = false;
        let pipeline = device
            .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
            .unwrap();
        // Two positions, both inside the buffer's two indices: a valid draw, which reserves
        // nothing of the pool its pass leases.
        let indices = buffer_with(&device, &words(&[1, 2])).unwrap();
        let arguments = buffer_with(&device, &words(&[2, 1, 0, 0, 0])).unwrap();
        let seen = buffer_with(&device, &[0; 32]).unwrap();
        Lane {
            device,
            pipeline,
            seen,
            indices,
            arguments,
        }
    }

    fn queue(&self) -> CommandQueue {
        self.device.new_command_queue().unwrap()
    }

    /// A command buffer of `queue` with the draw, encoded: its pass leased the queue's block.
    fn encoded(&self, queue: &CommandQueue) -> CommandBuffer {
        self.encoded_with(queue, &self.arguments)
    }

    /// A command buffer of `queue` with the draw of `arguments`, encoded.
    fn encoded_with(&self, queue: &CommandQueue, arguments: &Buffer) -> CommandBuffer {
        let mut pass = RenderPassDescriptor::new();
        pass.render_target_width = 1;
        pass.render_target_height = 1;
        pass.default_raster_sample_count = 1;
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encoder.set_render_pipeline_state(&self.pipeline).unwrap();
        encoder.set_vertex_buffer(Some(&self.seen), 0, 0).unwrap();
        encoder
            .draw_indexed_primitives_indirect(
                PrimitiveType::Point,
                IndexType::UInt32,
                &self.indices,
                0,
                arguments,
                0,
            )
            .unwrap();
        encoder.end_encoding().unwrap();
        command_buffer
    }

    /// One command buffer of `queue` with the draw, committed and completed.
    fn drawn(&self, queue: &CommandQueue) {
        let command_buffer = self.encoded(queue);
        command_buffer.commit().unwrap();
        completed(&command_buffer);
    }

    /// The arguments of a draw whose command buffer of `queue` took `WORK` of GPU time or more:
    /// its instances doubled from one until one did, or reached `MOST_INSTANCES`.
    fn working(&self, queue: &CommandQueue) -> Buffer {
        let mut instances = 1;
        loop {
            let arguments = buffer_with(&self.device, &words(&[2, instances, 0, 0, 0])).unwrap();
            let command_buffer = self.encoded_with(queue, &arguments);
            command_buffer.commit().unwrap();
            completed(&command_buffer);
            let took =
                command_buffer.gpu_end_time().unwrap() - command_buffer.gpu_start_time().unwrap();
            if took >= WORK || instances == MOST_INSTANCES {
                eprintln!(
                    "{instances} instances took {:.3} ms of GPU time",
                    took * 1e3
                );
                return arguments;
            }
            instances *= 2;
        }
    }

    /// The blocks of the device's GPU pools made and released so far.
    fn blocks(&self) -> (u64, u64) {
        let counts = self.device.work_counts();
        (counts.gpu_pool_blocks_made, counts.gpu_pool_blocks_released)
    }
}

/// Wait for `command_buffer`, which completes without an error.
fn completed(command_buffer: &CommandBuffer) {
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
}

/// Two command buffers of one queue encoded before either commits, so both hold its block,
/// round after round: the first round makes the one block, and every later one makes and
/// releases none (the module docs).
#[test]
fn command_buffers_of_one_queue_share_one_pool_block() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let lane = Lane::new(&messages, &cache);
        let queue = lane.queue();
        for round in 0..ROUNDS {
            let (made, released) = lane.blocks();
            let first = lane.encoded(&queue);
            let second = lane.encoded(&queue);
            first.commit().unwrap();
            second.commit().unwrap();
            completed(&first);
            completed(&second);
            let (made_now, released_now) = lane.blocks();
            let expected = if round == 0 { (1, 0) } else { (0, 0) };
            assert_eq!(
                (made_now - made, released_now - released),
                expected,
                "round {round}: blocks made and released"
            );
        }
        assert_eq!(
            lane.device.work_counts().gpu_pool_leases_peak,
            2,
            "two command buffers held the block at once at most"
        );
    }
    layer_reported_nothing_at_all(&messages);
}

/// A second queue's command buffer makes a block of its own; the first queue's block stays. Two
/// queues that go let their blocks go, the first to the device's spare and the next released,
/// and a queue made later takes the spare rather than make a block.
#[test]
fn a_second_queue_takes_its_own_pool_block() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let lane = Lane::new_with(at_plan, &messages, &cache);
        let queue = lane.queue();
        lane.drawn(&queue);
        assert_eq!(lane.blocks(), (1, 0), "the first queue's block");
        let other = lane.queue();
        lane.drawn(&other);
        assert_eq!(lane.blocks(), (2, 0), "the second queue's own block");
        lane.drawn(&queue);
        assert_eq!(lane.blocks(), (2, 0), "the first queue's block again");
        let third = lane.queue();
        lane.drawn(&third);
        assert_eq!(lane.blocks(), (3, 0), "the third queue's own block");
        drop((other, third));
        gone(&lane, 1);
        assert_eq!(
            lane.blocks(),
            (3, 1),
            "one of the two blocks went, the other is the spare"
        );
        let later = lane.queue();
        lane.drawn(&later);
        assert_eq!(lane.blocks(), (3, 1), "the later queue took the spare");
        lane.drawn(&queue);
        assert_eq!(lane.blocks(), (3, 1), "the first queue's block again");
    }
    layer_reported_nothing_at_all(&messages);
}

/// The blocks the lane's device holds, its spare's included.
fn live(lane: &Lane) -> u64 {
    let (made, released) = lane.blocks();
    made - released
}

/// Wait until `released` of the lane's blocks were released: a queue goes once its last command
/// buffer's hold goes, which may be on a handler thread just after the wait returned.
fn gone(lane: &Lane, released: u64) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while lane.blocks().1 < released && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(1));
    }
}

/// Two queues' blocks that no command buffer held for the quiet period go as another queue's
/// command buffer's lease ends, the first to the device's spare and the next released; that
/// queue's block, just held, stays. Of the two queues that rested, the first to draw again takes
/// the spare, and the next makes a block.
#[test]
fn a_pool_block_no_command_buffer_held_goes_after_the_quiet_period() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let lane = Lane::new_with(at_plan, &messages, &cache);
        let idle = lane.queue();
        let rested = lane.queue();
        let busy = lane.queue();
        lane.drawn(&idle);
        lane.drawn(&rested);
        lane.drawn(&busy);
        assert_eq!(lane.blocks(), (3, 0));
        std::thread::sleep(QUIET + Duration::from_millis(200));
        lane.drawn(&busy);
        assert_eq!(
            lane.blocks(),
            (3, 1),
            "the idle queues' blocks went, one to the spare"
        );
        lane.drawn(&busy);
        assert_eq!(lane.blocks(), (3, 1), "the busy queue's block stayed");
        lane.drawn(&rested);
        assert_eq!(lane.blocks(), (3, 1), "a queue that rested took the spare");
        lane.drawn(&idle);
        assert_eq!(lane.blocks(), (4, 1), "the other makes one again");
    }
    layer_reported_nothing_at_all(&messages);
}

/// A command buffer's completed handler runs while another queue's first command buffer makes
/// that queue's block on this thread, round after round, each with a queue of its own (and a
/// queue that takes the device's spare first, so the other makes one). The
/// command buffer returned from its submission before the making began, and draws for about
/// `WORK` of GPU time: where the making took long enough to tell and the command buffer completed
/// in its first half, the handler ran less than half the making after the later of its completion
/// and its submission's return (the module docs).
#[test]
fn a_completed_handler_waits_for_no_pool_block_made_meanwhile() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let lane = Lane::new_with(at_plan, &messages, &cache);
        let queue = lane.queue();
        // The pipeline's first draw, which makes the first queue's block.
        lane.drawn(&queue);
        let working = lane.working(&queue);
        let mut told = 0;
        for round in 0..MAKING_ROUNDS {
            let other = lane.queue();
            // Takes the block the round before let go to the device's spare, or makes one.
            let holder = lane.queue();
            let held = lane.encoded(&holder);
            let first = lane.encoded_with(&queue, &working);
            let ran: Arc<Mutex<Option<f64>>> = Arc::default();
            let log = Arc::clone(&ran);
            first
                .add_completed_handler(Box::new(move |_| {
                    *log.lock().unwrap() = Some(host_time());
                }))
                .unwrap();
            let (made, _) = lane.blocks();
            first.commit().unwrap();
            // Its submission returned: the making's allocation holds none of it.
            first.wait_until_scheduled().unwrap();
            let start = host_time();
            let second = lane.encoded(&other);
            let encoded = host_time();
            assert_eq!(
                lane.blocks().0 - made,
                1,
                "round {round}: the second queue's command buffer made a block"
            );
            second.commit().unwrap();
            completed(&first);
            completed(&second);
            let handled = ran.lock().unwrap().expect("the completed handler ran");
            let seen = first.gpu_end_time().unwrap();
            // Its last batch's `vkQueueSubmit` returned: from then on its queue's waiter watches
            // for its completion.
            let submitted = first.kernel_end_time();
            let reached = seen.max(submitted);
            let making = encoded - start;
            let lag = handled - reached;
            let tells = making >= MAKING_TELLS && (start..start + making / 2.0).contains(&reached);
            told += usize::from(tells);
            eprintln!(
                "round {round}: making {:.3} ms, completion {:+.3} ms and submission's return \
                 {:+.3} ms into it, handler {:.3} ms after both{}",
                making * 1e3,
                (seen - start) * 1e3,
                (submitted - start) * 1e3,
                lag * 1e3,
                if tells { "" } else { " (tells nothing)" }
            );
            if tells {
                assert!(
                    lag < making / 2.0,
                    "round {round}: the completed handler ran {:.3} ms after its command buffer \
                     completed and its submission returned, during a making of {:.3} ms",
                    lag * 1e3,
                    making * 1e3
                );
            }
            drop((first, second, held, other, holder));
            // The first queue's block and the spare stay; the round's two queues went (the
            // module docs).
            let deadline = Instant::now() + Duration::from_secs(5);
            while live(&lane) > 2 && Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(1));
            }
            assert_eq!(live(&lane), 2, "round {round}: the blocks left");
        }
        eprintln!("{told} of {MAKING_ROUNDS} rounds told");
    }
    layer_reported_nothing_at_all(&messages);
}
