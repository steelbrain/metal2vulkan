//! Command buffers let go of as they are committed, on another thread, while the GPU still runs
//! the work committed before them, with their queue dropped and another made meanwhile: the
//! runtime begins none of their `VkCommandBuffer`s again, and resets none of their command pools,
//! before the GPU ran them (`spares.rs`). The validation layer tracks each command buffer's pending
//! state through the waits that saw its submission complete, and reports a begin or a reset of a
//! pending one (`VUID-vkBeginCommandBuffer-commandBuffer-00049`,
//! `VUID-vkResetCommandPool-commandPool-00040`).
//!
//! The frames are a window server's: several command buffers on one queue, each of a pass recorded
//! inline (`render/inline_first.rs`) and a pass into a texture over host pages with a depth
//! attachment, whose draws record into a secondary command buffer, each released as it is
//! committed, as the consumer's platform releases a guest's. Beside them, another thread commits a
//! command buffer a frame on a queue of its own, waiting for the frame's last command buffer's
//! event signal and for a signal the releasing thread makes from the host as it lets go of it: its
//! batches are held until both, and go from the thread that signals. A secondary that ANV executes
//! from a primary is called, its return address written into its last command by the GPU as the
//! call runs: one begun again while that primary is pending ends with a jump to nothing.

use super::host_memory::Mapping;
use super::no_copy_textures::{floats, rects_beside, staged};
use super::{
    at_plan, layer_reported_nothing_at_all, validated_device_with, Messages, ScratchCache, WAIT,
};
use crate::gpu::TEST_DEVICE;
use metal2vulkan::metal::{PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan_runtime::{
    Buffer, CommandBuffer, CommandBufferStatus, CommandQueue, CpuCacheMode, Device, EventRef,
    HazardTrackingMode, HostMemory, IoSurfacePlane, LoadAction, RenderPassDescriptor,
    RenderPipelineState, ResourceOptions, SharedEvent, StorageMode, StoreAction, Texture,
    TextureDescriptor,
};
use std::sync::mpsc;

/// The frames, each of [`COMMAND_BUFFERS`] command buffers.
const FRAMES: usize = 120;
const COMMAND_BUFFERS: usize = 6;
/// Every this many frames the queue is dropped and another made at once.
const QUEUE_EVERY: usize = 20;
const WIDTH: u64 = 2048;
const HEIGHT: u64 = 1024;
const ROW: u64 = WIDTH * 4;

/// What the frames let go of on the releasing thread, and the host signals it makes.
enum Released {
    CommandBuffer(CommandBuffer),
    Queue(CommandQueue),
    /// Signal the host event to this value.
    Signal(u64),
}

/// The passes' pipelines and inputs.
struct Drawing {
    inline: RenderPipelineState,
    beside_depth: RenderPipelineState,
    rects: [Buffer; 3],
    color: Buffer,
}

impl Drawing {
    /// A pass loading and storing `surface` that blends three rectangles over it: recorded inline,
    /// or, with `depth` attached (cleared, not stored), into a secondary command buffer.
    fn pass(&self, command_buffer: &CommandBuffer, surface: &Texture, depth: Option<&Texture>) {
        let mut pass = RenderPassDescriptor::new();
        let attachment = &mut pass.color_attachments[0];
        attachment.texture = Some(surface.clone());
        attachment.load_action = LoadAction::Load;
        attachment.store_action = StoreAction::Store;
        if let Some(depth) = depth {
            pass.depth_attachment.texture = Some(depth.clone());
            pass.depth_attachment.load_action = LoadAction::Clear;
            pass.depth_attachment.store_action = StoreAction::DontCare;
        }
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        let pipeline = match depth {
            Some(_) => &self.beside_depth,
            None => &self.inline,
        };
        encoder.set_render_pipeline_state(pipeline).unwrap();
        encoder
            .set_fragment_buffer(Some(&self.color), 0, 0)
            .unwrap();
        for rect in &self.rects {
            encoder.set_vertex_buffer(Some(rect), 0, 0).unwrap();
            encoder
                .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
                .unwrap();
        }
        encoder.end_encoding().unwrap();
    }
}

/// A private render target of the surface's shape.
fn private_target(device: &Device) -> Texture {
    let mut d = TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, WIDTH, HEIGHT, false);
    d.usage = TextureUsage::RENDER_TARGET;
    d.resource_options = ResourceOptions::private();
    device.new_texture(&d).unwrap()
}

/// A private depth texture of the surface's shape.
fn private_depth(device: &Device) -> Texture {
    let mut d = TextureDescriptor::texture_2d(PixelFormat::Depth32Float, WIDTH, HEIGHT, false);
    d.usage = TextureUsage::RENDER_TARGET;
    d.resource_options = ResourceOptions::private();
    device.new_texture(&d).unwrap()
}

/// A `Managed` no-copy buffer over `mapping`'s first `len` bytes, and a texture of its plane.
fn surface(device: &Device, mapping: &Mapping, len: usize) -> (Buffer, Texture) {
    let options = ResourceOptions::new(
        StorageMode::Managed,
        CpuCacheMode::DefaultCache,
        HazardTrackingMode::Default,
    );
    // SAFETY: the mapping outlives every buffer over it (the test drops them first).
    let memory = unsafe { HostMemory::new(mapping.pointer()) };
    // SAFETY: as above.
    let buffer =
        unsafe { device.new_buffer_with_bytes_no_copy(memory, len as u64, options, None) }.unwrap();
    let mut d = TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, WIDTH, HEIGHT, false);
    d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    d.resource_options = buffer.resource_options();
    let texture = buffer
        .new_texture_with_iosurface_plane(&d, IoSurfacePlane::new(0, ROW, false))
        .unwrap();
    (buffer, texture)
}

/// The frames, each command buffer handed to another thread to let go of as it is committed, and
/// the queue dropped and made again every [`QUEUE_EVERY`] frames, with the other thread's waiting
/// command buffers beside them: the layer reports nothing, every command buffer committed last on
/// a queue completes without an error, and the pools went round through the spares while earlier
/// command buffers were still pending.
#[test]
fn command_buffers_let_go_of_while_pending_keep_their_pools_until_they_ran() {
    let len = ((ROW * HEIGHT) as usize).next_multiple_of(16384);
    let mapping = Mapping::anonymous(len);
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), at_plan).unwrap();
        let drawing = Drawing {
            inline: rects_beside(&device, PixelFormat::BGRA8Unorm, true, None),
            beside_depth: rects_beside(
                &device,
                PixelFormat::BGRA8Unorm,
                true,
                Some(PixelFormat::Depth32Float),
            ),
            rects: [
                staged(&device, &floats([-1.0, -1.0, 1.0, 1.0])),
                staged(&device, &floats([-1.0, -1.0, 0.0, 0.0])),
                staged(&device, &floats([0.0, 0.0, 1.0, 1.0])),
            ],
            color: staged(&device, &floats([0.25, 0.5, 0.75, 0.5])),
        };
        let (buffer, texture) = surface(&device, &mapping, len);
        let (inline_target, depth) = (private_target(&device), private_depth(&device));
        let (aside_target, aside_depth) = (private_target(&device), private_depth(&device));
        let framed: SharedEvent = device.new_shared_event().unwrap();
        let host: SharedEvent = device.new_shared_event().unwrap();
        let before = device.work_counts();
        let (release, released) = mpsc::channel::<Released>();
        let signalled = host.clone();
        let releasing = std::thread::spawn(move || {
            let mut pending = 0_usize;
            for item in released {
                match item {
                    Released::CommandBuffer(command_buffer) => {
                        if command_buffer.status() != CommandBufferStatus::Completed {
                            pending += 1;
                        }
                        drop(command_buffer);
                    }
                    Released::Queue(queue) => drop(queue),
                    Released::Signal(value) => signalled.set_signaled_value(value),
                }
            }
            pending
        });
        let mut lasts = Vec::new();
        let aside_last = std::thread::scope(|scope| {
            // The other thread's frames, each waiting for this thread's frame and the host's.
            let aside_release = release.clone();
            let (drawing, framed, host) = (&drawing, &framed, &host);
            let (aside_target, aside_depth) = (&aside_target, &aside_depth);
            let device = &device;
            let aside = scope.spawn(move || {
                let queue = device.new_command_queue().unwrap();
                let mut last = None;
                for frame in 0..FRAMES as u64 {
                    let command_buffer = queue.command_buffer().unwrap();
                    command_buffer
                        .encode_wait_for_event(EventRef::from(framed), frame + 1)
                        .unwrap();
                    command_buffer
                        .encode_wait_for_event(EventRef::from(host), frame + 1)
                        .unwrap();
                    drawing.pass(&command_buffer, aside_target, None);
                    drawing.pass(&command_buffer, aside_target, Some(aside_depth));
                    command_buffer.commit().unwrap();
                    last = Some(command_buffer.clone());
                    aside_release
                        .send(Released::CommandBuffer(command_buffer))
                        .unwrap();
                    aside_release.send(Released::Signal(frame + 1)).unwrap();
                }
                drop(queue);
                last.unwrap()
            });
            let mut queue = device.new_command_queue().unwrap();
            for frame in 0..FRAMES {
                if frame > 0 && frame % QUEUE_EVERY == 0 {
                    let next = device.new_command_queue().unwrap();
                    release.send(Released::Queue(queue)).unwrap();
                    queue = next;
                }
                for at in 0..COMMAND_BUFFERS {
                    let command_buffer = queue.command_buffer().unwrap();
                    drawing.pass(&command_buffer, &inline_target, None);
                    drawing.pass(&command_buffer, &texture, Some(&depth));
                    if at + 1 == COMMAND_BUFFERS {
                        command_buffer
                            .encode_signal_event(EventRef::from(framed), frame as u64 + 1)
                            .unwrap();
                    }
                    command_buffer.commit().unwrap();
                    let queue_ends = (frame + 1) % QUEUE_EVERY == 0 || frame + 1 == FRAMES;
                    if queue_ends && at + 1 == COMMAND_BUFFERS {
                        lasts.push(command_buffer.clone());
                    }
                    release
                        .send(Released::CommandBuffer(command_buffer))
                        .unwrap();
                }
            }
            drop(queue);
            aside.join().unwrap()
        });
        drop(release);
        let pending = releasing.join().unwrap();
        lasts.push(aside_last);
        for command_buffer in &lasts {
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
            assert_eq!(command_buffer.error(), None);
        }
        let after = device.work_counts();
        let taken = after.spare_command_pools_taken - before.spare_command_pools_taken;
        let secondaries =
            after.command_buffers_begun_secondaries - before.command_buffers_begun_secondaries;
        let inline = after.render_passes_begun_inline - before.render_passes_begun_inline;
        let ahead = after.primaries_begun_ahead - before.primaries_begun_ahead;
        let held = after.batches_held_past_a_look - before.batches_held_past_a_look;
        let met = after.event_waits_met - before.event_waits_met;
        let committed = FRAMES * (COMMAND_BUFFERS + 1);
        println!(
            "PENDING-POOLS on {}: {committed} command buffers, {pending} let go of before they \
             completed; spare pools taken {taken}, primaries begun ahead {ahead}, secondaries \
             begun {secondaries}, passes begun inline {inline}, batches held past a look {held}, \
             event waits met {met}",
            device.name(),
        );
        // The pools went round while command buffers were pending, and both kinds of pass ran.
        assert!(
            pending > 0,
            "no command buffer was pending as it was let go of"
        );
        assert!(taken > 0, "no spare pool was taken");
        assert!(
            secondaries > 0,
            "no pass recorded into a secondary command buffer"
        );
        assert!(inline > 0, "no pass recorded inline");
        drop((lasts, drawing, texture, buffer, depth, inline_target));
        drop((aside_target, aside_depth, framed, host));
    }
    layer_reported_nothing_at_all(&messages);
}
