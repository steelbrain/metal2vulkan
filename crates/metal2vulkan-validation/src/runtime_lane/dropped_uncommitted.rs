//! A command buffer dropped uncommitted, whose draws' pipelines are still being made, on the
//! device the cfg selects, under the validation layer: once the drop returns, nothing of it
//! touches host memory, so the program may unmap the pages its draws wrote at once.
//!
//! The consumer (reims-vgpu) unmaps a guest task's pages as soon as the task's objects are let go
//! of, and relies on this: a dropped, uncommitted command buffer's deferred work, a draw whose
//! pipeline is still being made among it, never reaches the scheduler, which takes only committed
//! batches and hands a host step over only for a committed line's head.
//!
//! The draws are of the owned fixture `ob3_index_buffer`, a vertex function with rasterization
//! disabled that marks `seen[min(vertex_id, 7)]`, here through a no-copy buffer over a second
//! mapping of a page of guest RAM (a shared memory object, as `runtime_lane::host_memory` maps
//! it). The state's creation predicts a point list's pipeline, so a line's and a triangle's draw
//! each hand their own pipeline's making to the device's pool, and the pass records them later
//! (`render/pending.rs`).

use super::host_memory::{no_copy, not_importable, on_device, GuestRam, GUEST_PAGE};
use super::{pipelines_settled, WAIT};
use crate::air_library::AirLibrary;
use metal2vulkan::metal::PrimitiveType;
use metal2vulkan::pipeline::{RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    Buffer, CommandBuffer, CommandBufferStatus, CommandQueue, Device, PipelineOptions,
    RenderPassDescriptor, RenderPipelineState, ResourceOptions,
};
use std::ptr::NonNull;

/// The byte every page of the guest RAM holds until a draw marks it.
const UNTOUCHED: u8 = 0xa5;

/// The words a draw's vertex function marks, from the buffer's start.
const MARKS: usize = 8;

/// The fixture's state: `ob3_index_buffer`, rasterization disabled.
fn unrasterized_state(device: &Device) -> RenderPipelineState {
    let bitcode = llvm_assemble(&engine_dir().join("ob3_index_buffer.ll")).unwrap();
    let library = device.new_library_with_air(&bitcode).unwrap();
    let function = library.new_function_with_name("ob3_index_buffer").unwrap();
    let mut descriptor =
        RenderPipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    descriptor.rasterization_enabled = false;
    device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// A no-copy buffer over `len` bytes at `at`, with the consumer's resource options (0).
///
/// # Safety
///
/// `at..at + len` stays mapped while the buffer lives.
unsafe fn over(device: &Device, at: NonNull<u8>, len: usize) -> Buffer {
    // SAFETY: the caller's contract.
    match unsafe { no_copy(device, at, len, ResourceOptions::default(), None) } {
        Ok(buffer) => buffer,
        Err(error) if not_importable(&error) => {
            panic!("this device imports no host memory exactly: {error}")
        }
        Err(error) => panic!("no-copy buffer: {error}"),
    }
}

/// Encode, into a new command buffer of `queue`, a pass without attachments whose draws of
/// `state` (a line's eight vertices, then a triangle's six) mark `seen` at vertex buffer 0. Not
/// committed.
fn encode_marks(queue: &CommandQueue, state: &RenderPipelineState, seen: &Buffer) -> CommandBuffer {
    let mut pass = RenderPassDescriptor::new();
    pass.render_target_width = 1;
    pass.render_target_height = 1;
    pass.default_raster_sample_count = 1;
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
    encoder.set_render_pipeline_state(state).unwrap();
    encoder.set_vertex_buffer(Some(seen), 0, 0).unwrap();
    (encoder.draw_primitives(PrimitiveType::Line, 0, MARKS as u32, 1, 0)).unwrap();
    (encoder.draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)).unwrap();
    encoder.end_encoding().unwrap();
    command_buffer
}

/// The guest RAM's bytes of a page the draws marked: each of the [`MARKS`] words 1, every other
/// byte untouched.
fn marked_page() -> Vec<u8> {
    let mut page = vec![UNTOUCHED; GUEST_PAGE];
    for word in page[..4 * MARKS].chunks_mut(4) {
        word.copy_from_slice(&1u32.to_le_bytes());
    }
    page
}

/// A command buffer whose draws' pipelines are still being made as its encoding ends, and whose
/// draws write a no-copy buffer over guest pages, is dropped uncommitted; the program lets go of
/// the buffer and unmaps the pages at once (replaced in place by an inaccessible mapping, so any
/// CPU access of them faults), then waits until the makings are done. The guest RAM's page keeps
/// every byte, the validation layer reports nothing, and a command buffer committed after it on
/// the same queue draws with the pipelines now made, through a no-copy buffer over the next
/// page, which it marks: the same draws reach the guest RAM when they run.
#[test]
fn a_dropped_uncommitted_command_buffer_touches_no_host_memory_once_dropped() {
    on_device(|device| {
        let ram = GuestRam::new(3 * GUEST_PAGE);
        ram.write(0, &vec![UNTOUCHED; 3 * GUEST_PAGE]);
        let rw = libc::PROT_READ | libc::PROT_WRITE;
        let state = unrasterized_state(device);
        let queue = device.new_command_queue().unwrap();
        // Nothing of the state's creation is left in the pool: what it holds after the encoding
        // is the draws'.
        let before = pipelines_settled(device);
        let dropped_view = ram.map(GUEST_PAGE, GUEST_PAGE, rw);
        // SAFETY: the view stays mapped while the buffer lives: the buffer goes first.
        let dropped_buffer = unsafe { over(device, dropped_view.pointer(), GUEST_PAGE) };
        let command_buffer = encode_marks(&queue, &state, &dropped_buffer);
        let encoded = device.work_counts();
        assert_eq!(
            encoded.pipelines_pooled - before.pipelines_pooled,
            2,
            "the line's and the triangle's pipelines went to the pool"
        );
        assert!(
            encoded.pipeline_pool_busy > 0,
            "the draws' pipelines were made before the encoding ended: no draw was left pending"
        );
        drop(command_buffer);
        drop(dropped_buffer);
        // SAFETY: the view's own range, which nothing else uses: the buffer over it is gone.
        let unmapped = unsafe {
            libc::mmap(
                dropped_view.pointer().as_ptr().cast(),
                GUEST_PAGE,
                libc::PROT_NONE,
                libc::MAP_PRIVATE | libc::MAP_ANON | libc::MAP_FIXED,
                -1,
                0,
            )
        };
        assert_eq!(
            unmapped,
            dropped_view.pointer().as_ptr().cast(),
            "unmap in place"
        );
        pipelines_settled(device);
        // The same queue, the pipelines made, over the next page.
        let next_view = ram.map(2 * GUEST_PAGE, GUEST_PAGE, rw);
        // SAFETY: the view stays mapped until the command buffer completed and the buffer went.
        let next_buffer = unsafe { over(device, next_view.pointer(), GUEST_PAGE) };
        let next = encode_marks(&queue, &state, &next_buffer);
        next.commit().unwrap();
        next.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(next.status(), CommandBufferStatus::Completed);
        assert_eq!(next.error(), None);
        drop(next);
        drop(next_buffer);
        drop(next_view);
        assert!(
            ram.read(GUEST_PAGE..2 * GUEST_PAGE) == vec![UNTOUCHED; GUEST_PAGE],
            "the dropped command buffer's draws reached the guest page"
        );
        assert!(
            ram.read(2 * GUEST_PAGE..3 * GUEST_PAGE) == marked_page(),
            "the next command buffer's draws marked their guest page"
        );
        assert!(
            ram.read(0..GUEST_PAGE) == vec![UNTOUCHED; GUEST_PAGE],
            "a page no buffer was over"
        );
        drop(dropped_view);
    });
}
