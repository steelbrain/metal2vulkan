//! A heap's textures the program released go as the command buffers that used the heap while the
//! program held them complete, while later command buffers that use the heap are in flight
//! (`residency.rs`, `Heap::textures_from`): Metal's `useHeap:` holds the heap, not its textures,
//! so a later encoder's `use_heap` makes resident only the textures the program holds, not those
//! an earlier command buffer still holds for its own commands.
//!
//! An automatic `Private` heap holds one texture the program keeps, filled by a blit, and
//! [`RELEASED`] more. A first frame (a command buffer whose first command waits for an event the
//! host signals) uses the heap and reads the kept texture through its resource ID, so it holds
//! every texture of the heap until it completes. The program releases the [`RELEASED`] textures,
//! commits [`LATER`] frames of the same commands, each waiting for a later value of the event, and
//! signals the first frame's value alone: once the first frame completed, every released texture
//! must go while the later frames still wait. Then each frame read the kept texture's texels.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The function is the owned fixture
//! `validation/fixtures/public/engine/handles.ll`'s `argument_buffer_read` (`llvm-as`).

use super::blit_holds::{device, Scratch, WAIT};
use crate::{
    BlitOption, Buffer, CommandBuffer, CommandBufferStatus, CommandQueue, ComputePipelineState,
    Device, EventRef, Heap, HeapDescriptor, HeapType, Origin, PipelineOptions, ResourceOptions,
    SharedEvent, Size, Texture, TextureDescriptor,
};
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use std::time::{Duration, Instant};

/// The textures' width, in texels, and the threads of each dispatch.
const SIDE: u64 = 8;

/// The heap's textures the program releases.
const RELEASED: usize = 128;

/// The frames committed after the release, each waiting for its own value of the event.
const LATER: u64 = 3;

/// How long the released textures may take to go once the first frame completed.
const GONE_WITHIN: Duration = Duration::from_secs(10);

/// Texel `x` of the kept texture.
fn texel(x: u64) -> [f32; 4] {
    [x as f32 + 1.0, 2.0 * x as f32 + 1.0, 0.25, 1.0]
}

/// The kept texture's texels, as bytes.
fn texels() -> Vec<u8> {
    (0..SIDE)
        .flat_map(texel)
        .flat_map(f32::to_le_bytes)
        .collect()
}

/// A `Private` `RGBA32Float` texture of [`SIDE`] by 1 texels a function reads.
fn descriptor() -> TextureDescriptor {
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA32Float, SIDE, 1, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::private();
    descriptor
}

/// `argument_buffer_read` of the owned engine fixture `handles.ll` on `device`: it reads the
/// texture whose resource ID buffer 0 holds into buffer 1, a texel a thread.
fn state(device: &Device) -> ComputePipelineState {
    let ll = metal2vulkan_devtools::engine_dir().join("handles.ll");
    let bitcode = metal2vulkan_devtools::llvm_assemble(&ll).unwrap();
    let library = metal2vulkan::library::Library::from_air(&bitcode).unwrap();
    let function = device
        .new_library_with_compile_library(library)
        .new_function_with_name("argument_buffer_read")
        .unwrap();
    let descriptor =
        ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    device
        .new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// The kept texture, filled with [`texels`] by a blit from a buffer.
fn kept(device: &Device, queue: &CommandQueue, heap: &Heap) -> Texture {
    let kept = heap.new_texture(&descriptor()).unwrap();
    let source = device
        .new_buffer_with_bytes(&texels(), ResourceOptions::shared())
        .unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_buffer_to_texture(
        &source,
        0,
        16 * SIDE,
        0,
        Size::new(SIDE, 1, 1),
        &kept,
        0,
        0,
        Origin::new(0, 0, 0),
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    kept
}

/// What a frame holds: its command buffer and the buffer its function writes.
struct Frame {
    command_buffer: CommandBuffer,
    out: Buffer,
}

impl Frame {
    /// A frame committed on `queue` whose first command waits for `event` to reach `value`, then
    /// uses `heap` and reads the texture whose resource ID `ids` holds.
    fn commit(
        (device, queue, state): (&Device, &CommandQueue, &ComputePipelineState),
        (heap, ids): (&Heap, &Buffer),
        (event, value): (&SharedEvent, u64),
    ) -> Frame {
        let out = device
            .new_buffer(16 * SIDE, ResourceOptions::shared())
            .unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        (command_buffer.encode_wait_for_event(EventRef::Shared(event), value)).unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(state).unwrap();
        encoder.use_heap(heap).unwrap();
        encoder.set_buffer(Some(ids), 0, 0).unwrap();
        encoder.set_buffer(Some(&out), 0, 1).unwrap();
        encoder
            .dispatch_threads(Size::new(SIDE, 1, 1), Size::new(SIDE, 1, 1))
            .unwrap();
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        Frame {
            command_buffer,
            out,
        }
    }

    /// Wait for the frame, which must read the kept texture's texels.
    fn check(self, which: &str) {
        let command_buffer = &self.command_buffer;
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        assert_eq!(command_buffer.error(), None);
        let contents = self.out.contents().unwrap();
        let length = 16 * SIDE as usize;
        // SAFETY: the buffer's bytes, which the completed command buffer wrote.
        let read = unsafe { std::slice::from_raw_parts(contents.as_ptr(), length) };
        assert_eq!(
            read,
            texels(),
            "{which} reads the kept texture through its resource ID"
        );
    }
}

#[test]
fn released_heap_textures_go_while_later_users_of_the_heap_are_in_flight() {
    let scratch = Scratch::new("heap-use-released");
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let state = state(&device);
    let answer = device.heap_texture_size_and_align(&descriptor()).unwrap();
    let each = answer.size.next_multiple_of(answer.align);
    let bytes = each * (RELEASED as u64 + 1) + answer.align;
    let heap = device
        .new_heap(&HeapDescriptor::new(
            HeapType::Automatic,
            bytes,
            ResourceOptions::private(),
        ))
        .unwrap();
    let kept = kept(&device, &queue, &heap);
    let released: Vec<Texture> = (0..RELEASED)
        .map(|_| heap.new_texture(&descriptor()).unwrap())
        .collect();
    let watched: Vec<_> = released.iter().map(Texture::downgrade).collect();
    let id = kept.gpu_resource_id().0.to_le_bytes();
    let ids = device
        .new_buffer_with_bytes(&id, ResourceOptions::shared())
        .unwrap();
    let event = device.new_shared_event().unwrap();
    let on = (&device, &queue, &state);
    // The first frame uses the heap while the program holds every texture of it.
    let first = Frame::commit(on, (&heap, &ids), (&event, 1));
    drop(released);
    let later: Vec<Frame> = (2..=LATER + 1)
        .map(|value| Frame::commit(on, (&heap, &ids), (&event, value)))
        .collect();
    event.set_signaled_value(1);
    first.check("the first frame");
    let deadline = Instant::now() + GONE_WITHIN;
    let living = || {
        watched
            .iter()
            .filter(|weak| weak.strong_count() > 0)
            .count()
    };
    while living() > 0 && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(1));
    }
    let pending = later
        .iter()
        .filter(|frame| frame.command_buffer.status() != CommandBufferStatus::Completed)
        .count();
    assert_eq!(pending, LATER as usize, "the later frames still wait");
    assert_eq!(
        RELEASED - living(),
        RELEASED,
        "the released textures go once the first frame completed, while later frames that use \
         the heap are in flight"
    );
    event.set_signaled_value(LATER + 1);
    for frame in later {
        frame.check("a later frame");
    }
}
