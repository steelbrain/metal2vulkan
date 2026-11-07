//! A plain pass split by memory barriers over its render targets, each followed by a draw that
//! reads its own attachment at its own pixels: its instances after the first record inline too
//! (`render/inline_splits.rs`), each with its copy of the attachment taken where the pass's end
//! takes it, before the instance (`texture/attached_reads.rs`). So a draw reads the texels the
//! instances before its own drew, and none its own instance drew: the copies' contents are
//! those of the pass recorded with each instance in a secondary.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The functions are `tob_rect`,
//! `tob_solid` and `tob_read` of the owned fixture
//! `validation/fixtures/public/engine/tob_renders.ll` (`llvm-as`).

use crate::{
    BlitOption, Device, DeviceKind, DeviceOptions, LoadAction, Origin, PipelineOptions, Region,
    RenderCommandEncoder, RenderPassDescriptor, RenderPipelineState, ResourceOptions,
    ShaderCacheOptions, Size, StoreAction, TextureDescriptor,
};
use metal2vulkan::metal::{BarrierScope, PixelFormat, PrimitiveType, RenderStages, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use std::path::{Path, PathBuf};
use std::time::Duration;

/// The longest the command buffer may take.
const WAIT: Duration = Duration::from_secs(60);

/// The target's side, in pixels.
const SIDE: u64 = 32;

/// The target's red before the pass, the red the first fill writes, and the second's.
const BEFORE: u8 = 64;
const FIRST: u8 = 128;
const SECOND: u8 = 200;

/// A scratch directory for the device's shader cache, removed when dropped.
struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Scratch {
        let path = std::env::temp_dir().join(format!("m2v-inline-splits-{}", std::process::id()));
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

/// The fixture's state of `tob_rect` and `fragment`, one `RGBA8Unorm` target.
fn state(device: &Device, fragment: &str) -> RenderPipelineState {
    let ll = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/fixtures/public/engine/tob_renders.ll");
    let bitcode = metal2vulkan_devtools::llvm_assemble(&ll).unwrap();
    let library = metal2vulkan::library::Library::from_air(&bitcode).unwrap();
    let library = device.new_library_with_compile_library(library);
    let vertex = library.new_function_with_name("tob_rect").unwrap();
    let fragment = library.new_function_with_name(fragment).unwrap();
    let mut descriptor =
        RenderPipelineDescriptor::new(StageSource::Function(vertex.function_descriptor()));
    descriptor.fragment = Some(StageSource::Function(fragment.function_descriptor()));
    descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

fn floats(v: [f32; 4]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

/// The corners, in normalized device coordinates, of the target's pixels `[x0, x1) x [y0, y1)`.
fn corners([x0, y0, x1, y1]: [u64; 4]) -> Vec<u8> {
    let x = |p: u64| p as f32 / SIDE as f32 * 2.0 - 1.0;
    let y = |p: u64| 1.0 - p as f32 / SIDE as f32 * 2.0;
    floats([x(x0), y(y1), x(x1), y(y0)])
}

/// Fill the box `rect` with `red`.
fn fill(encoder: &mut RenderCommandEncoder, state: &RenderPipelineState, rect: [u64; 4], red: u8) {
    encoder.set_render_pipeline_state(state).unwrap();
    encoder.set_vertex_bytes(&corners(rect), 0).unwrap();
    let red = f32::from(red) / 255.0;
    encoder
        .set_fragment_bytes(&floats([red, 0.0, 0.0, 1.0]), 0)
        .unwrap();
    encoder
        .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
        .unwrap();
}

/// Draw the box `rect`, each fragment the texel under it in the target's copy.
fn read(encoder: &mut RenderCommandEncoder, state: &RenderPipelineState, rect: [u64; 4]) {
    encoder.set_render_pipeline_state(state).unwrap();
    encoder.set_vertex_bytes(&corners(rect), 0).unwrap();
    encoder
        .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
        .unwrap();
}

/// A barrier over the render targets from the fragment stage to the fragment stage.
fn barrier(encoder: &mut RenderCommandEncoder) {
    let fragment = RenderStages::FRAGMENT;
    encoder.memory_barrier_with_scope(BarrierScope::RENDER_TARGETS, fragment, fragment);
}

/// The first instance fills `[4, 28)`; after a barrier the second reads `[8, 16)`, fills it,
/// and reads `[10, 14)`, which reads its copy from before it (the first fill's); after another
/// the third reads `[8, 10)`, which reads the second fill.
#[test]
fn instances_after_splits_record_inline_and_read_the_copies_from_before_them() {
    let scratch = Scratch::new();
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let solid = state(&device, "tob_solid");
    let reader = state(&device, "tob_read");
    let mut d = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
    d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    d.resource_options = ResourceOptions::shared();
    let target = device.new_texture(&d).unwrap();
    let before = [BEFORE, 0, 0, 255].repeat((SIDE * SIDE) as usize);
    let region = Region::new_2d(0, 0, SIDE, SIDE);
    target
        .replace_region(region, 0, 0, &before, SIDE * 4, 0)
        .unwrap();
    let back = device
        .new_buffer(SIDE * SIDE * 4, ResourceOptions::shared())
        .unwrap();
    let counted = device.work_counts();
    let mut pass = RenderPassDescriptor::new();
    let color = &mut pass.color_attachments[0];
    color.texture = Some(target.clone());
    color.load_action = LoadAction::Load;
    color.store_action = StoreAction::Store;
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
    encoder.set_fragment_texture(Some(&target), 0).unwrap();
    fill(&mut encoder, &solid, [4, 4, 28, 28], FIRST);
    barrier(&mut encoder);
    read(&mut encoder, &reader, [8, 8, 16, 16]);
    fill(&mut encoder, &solid, [8, 8, 16, 16], SECOND);
    read(&mut encoder, &reader, [10, 10, 14, 14]);
    barrier(&mut encoder);
    read(&mut encoder, &reader, [8, 8, 10, 10]);
    encoder.end_encoding().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_texture_to_buffer(
        &target,
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(SIDE, SIDE, 1),
        &back,
        0,
        SIDE * 4,
        SIDE * SIDE * 4,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    let after = device.work_counts();
    // The path ran: the first instance inline, and the two after the barriers.
    assert_eq!(after.inline_passes - counted.inline_passes, 1);
    assert_eq!(after.inline_passes_split - counted.inline_passes_split, 1);
    assert_eq!(
        after.inline_split_instances - counted.inline_split_instances,
        2
    );
    // One render encoder, its render pass instances by where they began, and its splits by cause
    // (`WorkCounts::render_passes_begun_inline` and kin): each barrier over the render targets
    // orders the draws in their instance, and the draw after it that reads the attachment splits
    // the pass, twice, and nothing else does.
    let splits = |c: &crate::WorkCounts| {
        c.instance_splits_barrier
            + c.instance_splits_tile_dispatch
            + c.instance_splits_tile_memory
            + c.instance_splits_interlocked_writes
            + c.instance_splits_feedback
            + c.instance_splits_memoryless_write
            + c.instance_splits_pending_pipeline
            + c.instance_splits_execution
            + c.instance_splits_visibility
            + c.instance_splits_shape
    };
    assert_eq!(
        (
            after.render_encoders_ended - counted.render_encoders_ended,
            after.render_passes_begun_inline - counted.render_passes_begun_inline,
            after.render_passes_begun_inline_splits - counted.render_passes_begun_inline_splits,
            after.render_passes_begun_instances - counted.render_passes_begun_instances,
            after.render_passes_begun_resolves - counted.render_passes_begun_resolves,
            after.instance_splits_attached_read - counted.instance_splits_attached_read,
            splits(&after) - splits(&counted),
        ),
        (1, 1, 2, 0, 0, 2, 0),
        "the encoders, the instances by where they began, and the splits by cause"
    );
    let contents = back.contents().unwrap();
    // SAFETY: the buffer's bytes, which no command uses now.
    let bytes = unsafe { std::slice::from_raw_parts(contents.as_ptr(), back.length() as usize) };
    let inside = |p: u64, [x0, y0, x1, y1]: [u64; 4]| {
        let (x, y) = (p % SIDE, p / SIDE);
        (x0..x1).contains(&x) && (y0..y1).contains(&y)
    };
    for p in 0..SIDE * SIDE {
        let expected = if inside(p, [8, 8, 10, 10]) {
            // The third instance's read: the second fill, which its copy holds.
            SECOND
        } else if inside(p, [10, 10, 14, 14]) {
            // The second instance's second read: its copy, from before it.
            FIRST
        } else if inside(p, [8, 8, 16, 16]) {
            SECOND
        } else if inside(p, [4, 4, 28, 28]) {
            FIRST
        } else {
            BEFORE
        };
        let (x, y) = (p % SIDE, p / SIDE);
        assert_eq!(bytes[(p * 4) as usize], expected, "pixel ({x}, {y})");
    }
}
