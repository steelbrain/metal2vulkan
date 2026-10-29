//! The order across queues of a batch that waits for a pipeline being made: Metal makes a render
//! pipeline at `newRenderPipelineState`, so a batch of its draws never waits at its submission, and
//! a batch committed after it on another queue runs at once unless the two share a resource one of
//! them writes (`CommandBuffer::footprint`, `scheduler/order.rs`). The runtime makes a draw's
//! `VkPipeline` on the device's pool (`pipeline/pool.rs`), whose test hold (`pool::held`) keeps
//! every making waiting here. The state's creation makes the pipeline of a triangle list's draw
//! (`pipeline/predicted.rs`), which a triangle strip shares where the topology is dynamic, so the
//! pass draws a line, of another class, whose pipeline only its draw makes: across the target's top
//! row, through its texels' centers.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The render pipeline is the owned
//! fixture `validation/fixtures/public/examples/runtime_examples.ll` (`llvm-as`), as the
//! examples run it.

use crate::pipeline::budget::Urgency;
use crate::pipeline::pool::held;
use crate::{
    BlitOption, Buffer, ClearColor, CommandBuffer, CommandBufferStatus, CommandQueue, Device,
    DeviceKind, DeviceOptions, EventRef, HostMemory, LoadAction, Origin, PipelineOptions,
    RenderPassDescriptor, RenderPipelineState, ResourceOptions, ShaderCacheOptions, Size,
    StoreAction, Texture, TextureDescriptor,
};
use metal2vulkan::device::PipelineWaitPolicy;
use metal2vulkan::metal::{PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use std::alloc::Layout;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// The longest a command buffer that runs may take.
const WAIT: Duration = Duration::from_secs(60);

/// How long a batch that waits for the pass is watched not to complete.
const WATCHED: Duration = Duration::from_millis(500);

/// The render target's side, and the texel the draw stores in its top row.
const SIDE: u64 = 8;
const DRAWN: [f32; 4] = [1.0, 0.2, 0.0, 1.0];
const DRAWN_TEXEL: [u8; 4] = [255, 51, 0, 255];

/// The no-copy buffer's bytes and alignment, and its texture's width, height and row pitch.
const NO_COPY: usize = 64 << 10;
const NO_COPY_ALIGN: usize = 64 << 10;
const LINEAR: (u64, u64, u64) = (64, 16, 256);

/// A scratch directory for the device's shader cache, removed when dropped.
struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Scratch {
        let path =
            std::env::temp_dir().join(format!("m2v-pipeline-wait-order-{}", std::process::id()));
        std::fs::create_dir_all(&path).unwrap();
        Scratch(path)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The device the cfg selects: a CPU device under `m2v_cpu_icd`, else a GPU, its shader cache
/// in `cache`.
fn device(cache: &Path) -> Device {
    let mut options = DeviceOptions {
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
    // The held making is waited for until it is made, whatever the hold's length: under
    // `PipelineWaitPolicy::Bounded` a draw whose making is held past its command buffer's bound
    // runs nothing (`tests/deferred_calls.rs` takes that form).
    options.policy.pipeline_waits = PipelineWaitPolicy::Exact;
    Device::system_default_with(&options).unwrap()
}

/// The fixture's render pipeline state: `colored_vertex` and `colored_fragment`, one
/// `RGBA8Unorm` target.
fn state(device: &Device) -> RenderPipelineState {
    blended_state(device, false)
}

/// [`state`]'s, its target blended where `blended`: a state of other pipelines.
fn blended_state(device: &Device, blended: bool) -> RenderPipelineState {
    let ll = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/fixtures/public/examples/runtime_examples.ll");
    let bitcode = metal2vulkan_devtools::llvm_assemble(&ll).unwrap();
    let library = metal2vulkan::library::Library::from_air(&bitcode).unwrap();
    let library = device.new_library_with_compile_library(library);
    let vertex = library.new_function_with_name("colored_vertex").unwrap();
    let fragment = library.new_function_with_name("colored_fragment").unwrap();
    let mut descriptor =
        RenderPipelineDescriptor::new(StageSource::Function(vertex.function_descriptor()));
    descriptor.fragment = Some(StageSource::Function(fragment.function_descriptor()));
    let mut target = ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm);
    target.blending_enabled = blended;
    descriptor.color_attachments[0] = Some(target);
    device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// A `Shared` buffer of `values`' bytes.
fn floats(device: &Device, values: impl IntoIterator<Item = f32>) -> Buffer {
    let bytes: Vec<u8> = values.into_iter().flat_map(f32::to_le_bytes).collect();
    device
        .new_buffer_with_bytes(&bytes, ResourceOptions::shared())
        .unwrap()
}

/// A `Shared` buffer over memory of the test's, whose deallocator frees it and sets `freed`,
/// and a 2D texture over its bytes.
fn no_copy_texture(device: &Device, freed: &Arc<AtomicBool>) -> (Buffer, Texture) {
    let layout = Layout::from_size_align(NO_COPY, NO_COPY_ALIGN).unwrap();
    // SAFETY: the layout's size is not zero.
    let memory = NonNull::new(unsafe { std::alloc::alloc_zeroed(layout) }).unwrap();
    let freed = Arc::clone(freed);
    // SAFETY: the memory is the test's, mapped and writable, aligned to 64 KiB, until the
    // deallocator frees it once the buffer's last reference goes; a refused call drops the
    // deallocator unrun, and the memory then leaks.
    let buffer = unsafe {
        device.new_buffer_with_bytes_no_copy(
            HostMemory::new(memory),
            NO_COPY as u64,
            ResourceOptions::shared(),
            Some(Box::new(move |pointer, _| {
                // SAFETY: the pointer is the memory allocated above with `layout`, freed once.
                std::alloc::dealloc(pointer.as_ptr(), layout);
                freed.store(true, Ordering::SeqCst);
            })),
        )
    }
    .unwrap();
    let (width, height, row) = LINEAR;
    let mut descriptor =
        TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, width, height, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::shared();
    let texture = buffer.new_texture(&descriptor, 0, row).unwrap();
    (buffer, texture)
}

/// The pool's holds are counted: a test's hold lives until its own drop, whatever holds the
/// tests running beside it in the process drop meanwhile.
#[test]
fn a_hold_of_the_pool_lives_until_its_own_drop() {
    let first = held::hold();
    let second = held::hold();
    drop(first);
    assert!(held::holding(), "another hold's drop let the pool go");
    drop(second);
}

/// Queue A's pass, whose draw's `VkPipeline` the pool is held from making, waits for it; a batch
/// committed after it on queue B that shares no resource with it (a copy into a texture over a
/// no-copy buffer) completes meanwhile, and once its handles go its buffer's memory is freed,
/// nothing holding the buffer or its import (the deallocator not delayed); a batch on queue B that reads the pass's target waits for the
/// pass, and reads what it drew.
#[test]
fn a_batch_waiting_for_a_pipeline_holds_only_the_batches_that_share_its_resources() {
    let scratch = Scratch::new();
    let device = device(&scratch.0);
    // The device's check of images over host pages (`in_place_check.rs`), which the texture over
    // the no-copy buffer would start, runs to its verdicts first: its rounds' internal
    // submissions, in flight as the buffer goes, would hold its memory and so its deallocator
    // (`freed_memory.rs` waits for the device's own submissions), which is not what this test
    // asks about. The texture then takes the path the verdict gives, as a program's does once
    // the device's verdicts are known.
    let _ = device.settle_in_place_checks(WAIT);
    let state = state(&device);
    let mut target = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
    target.usage = TextureUsage::RENDER_TARGET;
    target.resource_options = ResourceOptions::private();
    let target = device.new_texture(&target).unwrap();
    // The top row's texel centers are at `y = 1 - 1 / SIDE`; the line runs past both edges, so
    // the clip makes its ends and every rasterization rule draws the row's first texel.
    let row_center = 1.0 - 1.0 / SIDE as f32;
    let positions = floats(&device, [-2.0, row_center, 2.0, row_center]);
    let colors = floats(&device, (0..2).flat_map(|_| DRAWN));
    let (a, b) = (
        device.new_command_queue().unwrap(),
        device.new_command_queue().unwrap(),
    );
    let hold = held::hold();

    // Queue A: the pass, its pipeline being made.
    let mut pass = RenderPassDescriptor::new();
    let color = &mut pass.color_attachments[0];
    color.texture = Some(target.clone());
    color.load_action = LoadAction::Clear;
    color.store_action = StoreAction::Store;
    color.clear_color = ClearColor::new(0.0, 0.0, 1.0, 1.0);
    let pooled = device.work_counts().pipelines_pooled;
    let drawing = a.command_buffer().unwrap();
    let mut encoder = drawing.render_command_encoder(&pass).unwrap();
    encoder.set_render_pipeline_state(&state).unwrap();
    encoder.set_vertex_buffer(Some(&positions), 0, 0).unwrap();
    encoder.set_vertex_buffer(Some(&colors), 0, 1).unwrap();
    encoder
        .draw_primitives(PrimitiveType::Line, 0, 2, 1, 0)
        .unwrap();
    encoder.end_encoding().unwrap();
    assert_eq!(
        device.work_counts().pipelines_pooled,
        pooled + 1,
        "the line's making waits in the pool"
    );
    drawing.commit().unwrap();

    // Queue B: a copy into a texture over a no-copy buffer, which the pass does not use.
    let freed = Arc::new(AtomicBool::new(false));
    {
        let (buffer, texture) = no_copy_texture(&device, &freed);
        let (width, height, row) = LINEAR;
        let source = device
            .new_buffer_with_bytes(
                &vec![0x5a; (row * height) as usize],
                ResourceOptions::shared(),
            )
            .unwrap();
        let unrelated = b.command_buffer().unwrap();
        let mut blit = unrelated.blit_command_encoder().unwrap();
        blit.copy_from_buffer_to_texture(
            &source,
            0,
            row,
            row * height,
            Size::new(width, height, 1),
            &texture,
            0,
            0,
            Origin::new(0, 0, 0),
            BlitOption::None,
        )
        .unwrap();
        blit.end_encoding().unwrap();
        unrelated.commit().unwrap();
        unrelated
            .wait_until_completed_for(WAIT)
            .expect("a batch on another queue that shares nothing with the pass runs meanwhile");
        assert_eq!(unrelated.status(), CommandBufferStatus::Completed);
        let contents = buffer.contents().unwrap();
        // SAFETY: the buffer's `NO_COPY` bytes, which no command buffer writes now.
        let bytes = unsafe { std::slice::from_raw_parts(contents.as_ptr(), NO_COPY) };
        assert!(bytes[..(row * height) as usize].iter().all(|&b| b == 0x5a));
    }
    // Nothing of the runtime's holds the buffer: its memory is freed while the pass waits.
    let deadline = Instant::now() + WAIT;
    while !freed.load(Ordering::SeqCst) && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(10));
    }
    assert!(
        freed.load(Ordering::SeqCst),
        "the no-copy buffer's memory is freed"
    );
    // At its release, with its import freed first: the pass, committed but not yet submitted,
    // held no texture's image over the pages (`heap_sets.rs`), so nothing delayed the
    // deallocator, and the pass's submission lists no import of freed pages.
    assert_eq!(
        device.work_counts().host_deallocators_delayed,
        0,
        "the deallocator waited for the import"
    );
    assert_ne!(drawing.status(), CommandBufferStatus::Completed);

    // Queue B: a copy of the pass's target, which waits for the pass.
    let row = SIDE * 4;
    let readback = device
        .new_buffer(row * SIDE, ResourceOptions::shared())
        .unwrap();
    let conflicting = b.command_buffer().unwrap();
    let mut blit = conflicting.blit_command_encoder().unwrap();
    blit.copy_from_texture_to_buffer(
        &target,
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(SIDE, SIDE, 1),
        &readback,
        0,
        row,
        row * SIDE,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    conflicting.commit().unwrap();
    // The pool's hold is counted (`pool::held`): another test's drop never lets the pass's
    // making go meanwhile, so the copy cannot complete while the hold lives, however long it is
    // watched.
    let early = conflicting.wait_until_completed_for(WATCHED).is_ok();
    let drawn = drawing.status();
    assert!(
        !early,
        "a batch that reads the pass's target waits for the pass (the pass is {drawn:?}: its \
         making went while held where it completed, the copy missed its wait where not)"
    );
    assert_ne!(drawn, CommandBufferStatus::Completed);
    drop(hold);
    for command_buffer in [&drawing, &conflicting] {
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    }
    let contents = readback.contents().unwrap();
    // SAFETY: the buffer's bytes, which no command buffer writes now.
    let texels = unsafe { std::slice::from_raw_parts(contents.as_ptr(), (row * SIDE) as usize) };
    assert_eq!(
        texels[..4],
        DRAWN_TEXEL,
        "the copy reads what the pass drew"
    );
}

/// A command buffer on `queue` of one pass into a new target, whose line's `VkPipeline` of
/// `state` the device's pool makes: committed.
fn pass_of_a_pooled_line(
    device: &Device,
    queue: &CommandQueue,
    state: &RenderPipelineState,
) -> CommandBuffer {
    let mut target = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
    target.usage = TextureUsage::RENDER_TARGET;
    target.resource_options = ResourceOptions::private();
    let target = device.new_texture(&target).unwrap();
    let row_center = 1.0 - 1.0 / SIDE as f32;
    let positions = floats(device, [-2.0, row_center, 2.0, row_center]);
    let colors = floats(device, (0..2).flat_map(|_| DRAWN));
    let mut pass = RenderPassDescriptor::new();
    let color = &mut pass.color_attachments[0];
    color.texture = Some(target);
    color.load_action = LoadAction::Clear;
    color.store_action = StoreAction::Store;
    let pooled = device.work_counts().pipelines_pooled;
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
    encoder.set_render_pipeline_state(state).unwrap();
    encoder.set_vertex_buffer(Some(&positions), 0, 0).unwrap();
    encoder.set_vertex_buffer(Some(&colors), 0, 1).unwrap();
    encoder
        .draw_primitives(PrimitiveType::Line, 0, 2, 1, 0)
        .unwrap();
    encoder.end_encoding().unwrap();
    assert_eq!(
        device.work_counts().pipelines_pooled,
        pooled + 1,
        "the line's making waits in the pool"
    );
    command_buffer.commit().unwrap();
    command_buffer
}

/// The making a queue's first held batch waits for goes first in the device's pool: queue B's
/// first command buffer waits for an event the test signals last, and the pass committed behind
/// it there hands its line's making to the pool first; queue A's pass, committed after on a
/// target of its own, hands its own. Once the scheduler finds A's pass first in its queue's line,
/// its making ranks first (`budget::Urgency::Head`), before B's, whose batch cannot go before the
/// event anyway: Metal makes a pipeline state at its creation, so no submission waits behind
/// another queue's makings. Every command buffer then completes.
#[test]
fn a_queue_s_first_held_batch_s_making_goes_first_in_the_pool() {
    let scratch = Scratch::new();
    let device = device(&scratch.0);
    let (behind, first) = (state(&device), blended_state(&device, true));
    let (a, b) = (
        device.new_command_queue().unwrap(),
        device.new_command_queue().unwrap(),
    );
    let event = device.new_shared_event().unwrap();
    let hold = held::hold();

    // Queue B: a command buffer that waits for the event, then a pass behind it.
    let waiting = b.command_buffer().unwrap();
    waiting
        .encode_wait_for_event(EventRef::from(&event), 1)
        .unwrap();
    waiting.commit().unwrap();
    let held_behind = pass_of_a_pooled_line(&device, &b, &behind);
    // Queue A: a pass, first in its queue's line.
    let head = pass_of_a_pooled_line(&device, &a, &first);

    let pool = &device.shared().pipeline_pool;
    let deadline = Instant::now() + WAIT;
    while pool.first_urgency() != Some(Urgency::Head) && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(5));
    }
    assert_eq!(
        pool.first_urgency(),
        Some(Urgency::Head),
        "queue A's pass's making ranks first, before the earlier making of the pass behind B's \
         waiting command buffer"
    );
    drop(hold);
    head.wait_until_completed_for(WAIT)
        .expect("queue A's pass runs once its making is made");
    assert_ne!(
        held_behind.status(),
        CommandBufferStatus::Completed,
        "queue B's pass waits behind the event"
    );
    event.set_signaled_value(1);
    for command_buffer in [&waiting, &held_behind, &head] {
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    }
}
