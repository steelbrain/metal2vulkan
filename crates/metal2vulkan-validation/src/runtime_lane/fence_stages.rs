//! A render encoder's fence wait orders the draw stages it names and those after them, on the
//! device the cfg selects (`fence.rs` and `tracker.rs`'s `FenceUse` in the runtime). A blit encoder
//! fills a [`BIG`]-byte buffer (work to wait for) and then the first word of the depth buffer
//! `rt6_render_state`'s fullscreen triangle (`rt6_full`) reads in its vertex function, and updates
//! a fence; a render encoder waits for the fence and draws the triangle shaded white
//! (`rt6_white`) into a cleared `Depth32Float` target under `Less` with writes, so each texel
//! stores the depth its vertex function read: [`FILLED`] where it ran after the fill, [`BEFORE`]
//! where it ran before. A blit copies the depth out.
//!
//! - **Before the vertex stage**, under the validation layer: every texel holds [`FILLED`] (Metal:
//!   a wait before the vertex stage orders all of the draws' work).
//! - **Before the fragment stage**, without the validation layer: the vertex stage may run before
//!   the fill, as Metal's did on an M3 Max (the fence probe: a fragment-stage wait let the vertex
//!   function read the old value while the fragment function read the new). Either answer is
//!   Metal's, so the case prints how many texels held each and checks only that every texel holds
//!   one of the two. The validation layer is off there: the vertex function's read races the fill
//!   by Metal's rule, which synchronization validation reports as a hazard.

use super::{
    buffer_with, layer_reported_nothing, read_buffer, validated_device_with, Messages,
    ScratchCache, WAIT,
};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::metal::state::DepthStencilDescriptor;
use metal2vulkan::metal::{
    CompareFunction, PixelFormat, PrimitiveType, RenderStages, TextureUsage,
};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::llvm_assemble;
use metal2vulkan_runtime::{
    BlitOption, Device, LoadAction, Origin, PipelineOptions, RenderPassDescriptor, ResourceOptions,
    Size, StoreAction, TextureDescriptor,
};

/// The target's side.
const SIDE: u64 = 64;

/// The bytes the blit fills before the depth buffer's word: work long enough for a vertex
/// function that does not wait to run first, short of any watchdog.
const BIG: u64 = 64 << 20;

/// The depth the vertex function reads before the fill.
const BEFORE: f32 = 0.75;

/// The byte the blit fills the depth buffer's first word with, and the depth those bytes hold.
const FILL: u8 = 0x3e;
const FILLED: u32 = 0x3e3e_3e3e;

/// How many of the target's texels held [`FILLED`] and [`BEFORE`] after one wait before
/// `stages`, on `device`.
fn run(device: &Device, stages: RenderStages) -> (usize, usize) {
    let ll = metal2vulkan_devtools::engine_dir().join("rt6_render_state.ll");
    let library = device
        .new_library_with_air(&llvm_assemble(&ll).unwrap())
        .unwrap();
    let function = |name: &str| {
        StageSource::Function(
            library
                .new_function_with_name(name)
                .unwrap()
                .function_descriptor(),
        )
    };
    let mut descriptor = RenderPipelineDescriptor::new(function("rt6_full"));
    descriptor.fragment = Some(function("rt6_white"));
    descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    descriptor.depth_format = Some(PixelFormat::Depth32Float);
    let pipeline = device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap();
    let attachment = |format: PixelFormat| {
        let mut d = TextureDescriptor::texture_2d(format, SIDE, SIDE, false);
        d.usage = TextureUsage::RENDER_TARGET;
        d.resource_options = ResourceOptions::private();
        device.new_texture(&d).unwrap()
    };
    let target = attachment(PixelFormat::RGBA8Unorm);
    let depth = attachment(PixelFormat::Depth32Float);
    let mut less = DepthStencilDescriptor::default();
    less.depth_compare_function = CompareFunction::Less;
    less.depth_write_enabled = true;
    let less = device.new_depth_stencil_state(&less).unwrap();
    let mut words = vec![0u8; 256];
    words[..4].copy_from_slice(&BEFORE.to_le_bytes());
    let depths = buffer_with(device, &words).unwrap();
    let big = device.new_buffer(BIG, ResourceOptions::private()).unwrap();
    let out = device
        .new_buffer(SIDE * SIDE * 4, ResourceOptions::shared())
        .unwrap();
    let fence = device.new_fence();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.fill_buffer(&big, 0..BIG, 0x5a).unwrap();
    blit.fill_buffer(&depths, 0..4, FILL).unwrap();
    blit.update_fence(&fence).unwrap();
    blit.end_encoding().unwrap();
    let mut pass = RenderPassDescriptor::new();
    let color = &mut pass.color_attachments[0];
    color.texture = Some(target.clone());
    color.load_action = LoadAction::Clear;
    color.store_action = StoreAction::Store;
    pass.depth_attachment.texture = Some(depth.clone());
    pass.depth_attachment.load_action = LoadAction::Clear;
    pass.depth_attachment.clear_depth = 1.0;
    pass.depth_attachment.store_action = StoreAction::Store;
    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
    encoder.wait_for_fence(&fence, stages).unwrap();
    encoder.set_render_pipeline_state(&pipeline).unwrap();
    encoder.set_depth_stencil_state(&less).unwrap();
    encoder.set_vertex_buffer(Some(&depths), 0, 0).unwrap();
    encoder
        .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
        .unwrap();
    encoder.end_encoding().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    let size = Size {
        width: SIDE,
        height: SIDE,
        depth: 1,
    };
    let origin = Origin { x: 0, y: 0, z: 0 };
    blit.copy_from_texture_to_buffer(
        &depth,
        0,
        0,
        origin,
        size,
        &out,
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
    let bytes = read_buffer(&out, 0, SIDE * SIDE * 4).unwrap();
    let texels = bytes
        .chunks(4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]));
    let (mut filled, mut before) = (0, 0);
    for texel in texels {
        match texel {
            FILLED => filled += 1,
            t if t == BEFORE.to_bits() => before += 1,
            t => panic!("a texel of depth {t:#010x}, neither the fill's nor the one before"),
        }
    }
    (filled, before)
}

/// The module docs' wait before the vertex stage: every texel holds the fill's depth.
#[test]
fn a_wait_before_the_vertex_stage_orders_the_vertex_function() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |_| {}).unwrap();
    let name = device.name().to_owned();
    let (filled, before) = run(&device, RenderStages::VERTEX);
    assert_eq!(
        (filled, before),
        ((SIDE * SIDE) as usize, 0),
        "texels after the fill and before it on {name}"
    );
    drop(device);
    layer_reported_nothing(&messages, &format!("on {name}"));
}

/// The module docs' wait before the fragment stage: either answer, counted.
#[test]
fn a_wait_before_the_fragment_stage_may_let_the_vertex_function_run_first() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
        options.layers.clear();
    })
    .unwrap();
    let name = device.name().to_owned();
    let (filled, before) = run(&device, RenderStages::FRAGMENT);
    eprintln!(
        "FENCE-STAGES on {name}: fragment-stage wait, {filled} texels after the fill, \
         {before} before"
    );
    assert_eq!(filled + before, (SIDE * SIDE) as usize, "on {name}");
}
