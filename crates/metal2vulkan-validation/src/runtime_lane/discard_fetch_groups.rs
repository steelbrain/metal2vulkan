//! A fetching fragment function that discards, on the device the cfg selects, under the
//! validation layer: whether every live fragment's write lands. A device without
//! rasterization-order color access fetches through the pixel interlock, where the color's write
//! is a storage write the entry makes at its return; on MoltenVK on an Apple GPU, Apple's compiler
//! drops the storage writes of every live thread of a simdgroup (8 by 4 pixels) that holds a
//! discarded thread in some shapes of function (`DriverWorkaround::DiscardDropsSimdgroupStores`,
//! the translator's `lower/fragment/discard.rs`), which the translator defers such a discard past
//! (`fragment.discard.deferred`) or records. This lane draws that shape and counts the live pixels
//! whose write did not land, by their 8x4 groups.
//!
//! Each pass (`sfd_discard_fetch`) clears an 8-bit color, then draws in one encoder a fullscreen
//! `sfd_fill` (code [`FILL`]) and a draw of one of [`FRAGMENTS`] ([`Shape`]: one fullscreen
//! triangle, or a rectangle in each of 8 by 4 cells whose edges cut 2x2 quads) that fetches the
//! color, reads and samples a texture of one texel value ([`TEXEL`]), discards where the pixel's
//! `(x + 3 y) % 5` is 0 or never ([`Discards`]), samples again, takes a derivative and writes
//! fetched plus texel plus 1. The fragment functions differ in where float arithmetic the module
//! keeps apart (`NoContraction`, which SPIRV-Cross writes as `optnone` calls) runs: nowhere
//! (`sfd_alpha`), in straight-line code (`sfd_line`), in a branch only the discarded fragments
//! enter (`sfd_dropped`), or in one only the helper invocations past a rectangle's edges enter
//! (`sfd_edge`). So a pixel holds [`expected`]: that sum where the draw covers it and does not
//! discard it, else [`FILL`]. [`RUNS`] passes of each function, shape and discard pattern over
//! 224x128 and 1920x1080, the target kept across passes (fetched in place once it took its storage
//! form) or new each pass (fetched through a scratch image). Every pixel of every pass must hold
//! its code; each pass prints its wrong pixels, live and discarded apart, with their 8x4 groups and
//! how many of those are wrong in every live pixel; and each fragment stage's discard and helper
//! strategies, its `NoContraction` operations and its precision notes as the device's compile
//! profile lowers it, the path that ran.

use super::framebuffer_fetch::fetches;
use super::split_fetch_order::{codes, completed, encode_readback, texture, wrong_groups};
use super::{
    at_plan, layer_reported_nothing_at_all, validated_device_with, Messages, ScratchCache,
};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::metal::{PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, FunctionDescriptor, RenderPipelineDescriptor, StageSource,
};
use metal2vulkan::vk::ShaderStage;
use metal2vulkan::Budget;
use metal2vulkan_devtools::spirv::spirv_dis;
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    ClearColor, Device, Library, LoadAction, PipelineOptions, Region, RenderPassDescriptor,
    RenderPipelineState, ResourceOptions, SamplerState, StoreAction, Texture,
};

/// The code `sfd_fill` writes.
const FILL: u32 = 40;

/// The code every texel of the sampled texture holds.
const TEXEL: u32 = 90;

/// The passes of each fragment function, shape, discard pattern, target and size.
const RUNS: usize = 8;

/// The fetching fragment functions the lane draws with (the module docs).
const FRAGMENTS: [&str; 4] = ["sfd_alpha", "sfd_line", "sfd_dropped", "sfd_edge"];

/// The `sfd_alpha` draw's primitives.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Shape {
    /// One fullscreen triangle (`sfd_fan`).
    Fan,
    /// A rectangle in each of 8 by 4 cells (`sfd_cells`).
    Cells,
}

/// Where the fetching fragment function discards.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Discards {
    /// Where the pixel's `(x + 3 y) % 5` is 0.
    Pattern,
    /// Nowhere: the site is there, never taken.
    Never,
}

/// Whether `shape` covers pixel `(x, y)` of a target of `sides`: the rectangle of its cell from 3
/// right and 5 down of the cell's corner to 2 left and 2 up of the next (`sfd_cells`).
fn covers(shape: Shape, (width, height): (u64, u64), (x, y): (u64, u64)) -> bool {
    let (cell_x, cell_y) = (width / 8, height / 4);
    let (left, top) = (x / cell_x * cell_x, y / cell_y * cell_y);
    shape == Shape::Fan
        || (left + 3 <= x && x < left + cell_x - 2 && top + 5 <= y && y < top + cell_y - 2)
}

/// Whether the fetching fragment function discards pixel `(x, y)` under `discards`.
fn discards_at(discards: Discards, (x, y): (u64, u64)) -> bool {
    discards == Discards::Pattern && (x + 3 * y) % 5 == 0
}

/// The code pixel `(x, y)` of a target of `sides` holds after a pass (the module docs), and
/// whether the fetching draw wrote it.
fn expected(shape: Shape, discards: Discards, sides: (u64, u64), pixel: (u64, u64)) -> (u8, bool) {
    match covers(shape, sides, pixel) && !discards_at(discards, pixel) {
        true => (((FILL + TEXEL + 1) & 255) as u8, true),
        false => (FILL as u8, false),
    }
}

/// The pipelines a pass draws with.
struct Pipelines {
    fill: RenderPipelineState,
    fan: RenderPipelineState,
    cells: RenderPipelineState,
}

/// The descriptor of `vertex` and `fragment` of `library` into a `BGRA8Unorm` attachment.
fn descriptor(library: &Library, vertex: &str, fragment: &str) -> RenderPipelineDescriptor {
    let function = |name: &str| {
        StageSource::Function(
            library
                .new_function_with_name(name)
                .unwrap()
                .function_descriptor(),
        )
    };
    let mut descriptor = RenderPipelineDescriptor::new(function(vertex));
    descriptor.fragment = Some(function(fragment));
    descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::BGRA8Unorm));
    descriptor
}

/// The fragment stage's discard and helper strategies, its `NoContraction` operations and its
/// precision notes, as the device's compile profile lowers `fragment` drawn by `sfd_fan`.
fn lowering(device: &Device, ll: &std::path::Path, fragment: &str) -> String {
    let bitcode = llvm_assemble(ll).unwrap();
    let library = metal2vulkan::library::Library::from_air(&bitcode).unwrap();
    let function = |name: &str| {
        StageSource::Function(FunctionDescriptor::new(library.function(name).unwrap()))
    };
    let mut descriptor = RenderPipelineDescriptor::new(function("sfd_fan"));
    descriptor.fragment = Some(function(fragment));
    descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::BGRA8Unorm));
    let compiled = device
        .compile_device()
        .compile_render_pipeline(&descriptor, &Budget::standard())
        .unwrap();
    let Some(stage) = compiled
        .stages
        .iter()
        .find(|stage| stage.stage == ShaderStage::FRAGMENT)
    else {
        return "no fragment stage".to_owned();
    };
    let strategies: Vec<_> = stage
        .strategies
        .entries
        .iter()
        .filter(|(name, _)| {
            ["discard", "helper", "interlock"]
                .iter()
                .any(|n| name.contains(n))
        })
        .map(|(name, used)| format!("{name} x{}", used.uses))
        .collect();
    let kept_apart = spirv_dis(&stage.spirv).matches("NoContraction").count();
    format!(
        "strategies {strategies:?}, {kept_apart} NoContraction, notes {:?}",
        stage.precision_notes
    )
}

/// One pass of `shape` and `discards` over `target` of `sides`, the texture `sampled` bound
/// through `sampler`; every pixel's code after it.
fn pass(
    device: &Device,
    pipelines: &Pipelines,
    (target, sampled, sampler): (&Texture, &Texture, &SamplerState),
    (shape, discards): (Shape, Discards),
    sides: (u64, u64),
) -> Vec<u8> {
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut clear = RenderPassDescriptor::new();
    let color = &mut clear.color_attachments[0];
    color.texture = Some(target.clone());
    color.load_action = LoadAction::Clear;
    color.clear_color = ClearColor::new(0.0, 0.0, 0.0, 1.0);
    color.store_action = StoreAction::Store;
    let encoder = command_buffer.render_command_encoder(&clear).unwrap();
    encoder.end_encoding().unwrap();
    let mut descriptor = RenderPassDescriptor::new();
    let color = &mut descriptor.color_attachments[0];
    color.texture = Some(target.clone());
    color.load_action = LoadAction::Load;
    color.store_action = StoreAction::Store;
    let mut encoder = command_buffer.render_command_encoder(&descriptor).unwrap();
    encoder.set_render_pipeline_state(&pipelines.fill).unwrap();
    encoder
        .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
        .unwrap();
    let (pipeline, vertices, instances) = match shape {
        Shape::Fan => (&pipelines.fan, 3, 1),
        Shape::Cells => (&pipelines.cells, 6, 32),
    };
    encoder.set_render_pipeline_state(pipeline).unwrap();
    let size = [sides.0 as f32, sides.1 as f32];
    encoder
        .set_vertex_bytes(&size.map(f32::to_le_bytes).concat(), 0)
        .unwrap();
    // scale, bias, discard_mod, discard_hit (a hit past the modulus never discards), and the
    // cell's sides.
    let hit: u32 = match discards {
        Discards::Pattern => 0,
        Discards::Never => 7,
    };
    let globals = [
        1.0f32.to_le_bytes(),
        0.0f32.to_le_bytes(),
        5u32.to_le_bytes(),
        hit.to_le_bytes(),
        ((sides.0 / 8) as f32).to_le_bytes(),
        ((sides.1 / 4) as f32).to_le_bytes(),
    ]
    .concat();
    encoder.set_fragment_bytes(&globals, 0).unwrap();
    encoder.set_fragment_texture(Some(sampled), 0).unwrap();
    encoder
        .set_fragment_sampler_state(Some(sampler), 0)
        .unwrap();
    encoder
        .draw_primitives(PrimitiveType::Triangle, 0, vertices, instances, 0)
        .unwrap();
    encoder.end_encoding().unwrap();
    let buffer = encode_readback(&command_buffer, target, sides);
    command_buffer.commit().unwrap();
    completed(&command_buffer);
    codes(&buffer, sides)
}

/// Every live pixel of every pass holds its write (the module docs).
#[test]
fn a_discarding_fetch_lands_every_live_fragment_s_write() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), at_plan).unwrap();
    if !fetches(&device) {
        return;
    }
    let ll = engine_dir().join("sfd_discard_fetch.ll");
    let library = device
        .new_library_with_air(&llvm_assemble(&ll).unwrap())
        .unwrap();
    let make = |vertex, fragment| {
        let descriptor = descriptor(&library, vertex, fragment);
        device
            .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
            .unwrap()
    };
    let drawn: Vec<(&str, Pipelines)> = FRAGMENTS
        .iter()
        .map(|&fragment| {
            eprintln!(
                "DISCARD-FETCH on {}: {fragment} lowers with {}",
                device.name(),
                lowering(&device, &ll, fragment)
            );
            let pipelines = Pipelines {
                fill: make("sfd_fan", "sfd_fill"),
                fan: make("sfd_fan", fragment),
                cells: make("sfd_cells", fragment),
            };
            (fragment, pipelines)
        })
        .collect();
    let usage = (TextureUsage::SHADER_READ, ResourceOptions::shared());
    let sampled = texture(&device, PixelFormat::RGBA8Unorm, (4, 4), usage);
    let texel = [TEXEL as u8, TEXEL as u8, TEXEL as u8, 255];
    sampled
        .replace_region(Region::new_2d(0, 0, 4, 4), 0, 0, &texel.repeat(16), 16, 0)
        .unwrap();
    let sampler = device.new_sampler_state(&SamplerDescriptor::new()).unwrap();
    let target_usage = (
        TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ,
        ResourceOptions::private(),
    );
    let mut wrong_passes = Vec::new();
    for (fragment, pipelines) in &drawn {
        for sides in [(224u64, 128u64), (1920, 1080)] {
            let kept = texture(&device, PixelFormat::BGRA8Unorm, sides, target_usage);
            for shape in [Shape::Fan, Shape::Cells] {
                for discards in [Discards::Pattern, Discards::Never] {
                    let want: Vec<(u8, bool)> = (0..sides.1)
                        .flat_map(|y| (0..sides.0).map(move |x| (x, y)))
                        .map(|pixel| expected(shape, discards, sides, pixel))
                        .collect();
                    for fresh in [false, true] {
                        for run in 0..RUNS {
                            let target = match fresh {
                                true => {
                                    texture(&device, PixelFormat::BGRA8Unorm, sides, target_usage)
                                }
                                false => kept.clone(),
                            };
                            let bound = (&target, &sampled, &sampler);
                            let got = pass(&device, pipelines, bound, (shape, discards), sides);
                            let wrong: Vec<bool> =
                                got.iter().zip(&want).map(|(g, (w, _))| g != w).collect();
                            let live_wrong: Vec<bool> = wrong
                                .iter()
                                .zip(&want)
                                .map(|(&w, &(_, live))| w && live)
                                .collect();
                            let live = live_wrong.iter().filter(|&&w| w).count();
                            let other = wrong.iter().filter(|&&w| w).count() - live;
                            let groups = wrong_groups(&live_wrong, sides);
                            // Groups wrong in every live pixel they hold.
                            let whole = groups
                                .iter()
                                .filter(|&&(left, top)| {
                                    (top..(top + 4).min(sides.1))
                                        .flat_map(|y| {
                                            (left..(left + 8).min(sides.0)).map(move |x| (x, y))
                                        })
                                        .map(|(x, y)| (y * sides.0 + x) as usize)
                                        .filter(|&i| want[i].1)
                                        .all(|i| wrong[i])
                                })
                                .count();
                            let first: Vec<_> = groups.iter().take(6).collect();
                            eprintln!(
                            "DISCARD-FETCH on {}: {fragment} {}x{} {shape:?} {discards:?} fresh \
                             {fresh}, pass {run}: {live} live and {other} other pixels wrong, the \
                             live ones in {} 8x4 groups ({whole} wrong in every live pixel), first \
                             {first:?}",
                            device.name(),
                            sides.0,
                            sides.1,
                            groups.len(),
                        );
                            if live + other > 0 {
                                wrong_passes.push((
                                    fragment, sides, shape, discards, fresh, run, live, other,
                                ));
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(
        wrong_passes.is_empty(),
        "{}: passes whose fragments' writes did not land (fragment function, sides, shape, \
         discards, fresh, pass, live pixels, other pixels): {wrong_passes:?}",
        device.name()
    );
    drop(drawn);
    drop(device);
    layer_reported_nothing_at_all(&messages);
}
