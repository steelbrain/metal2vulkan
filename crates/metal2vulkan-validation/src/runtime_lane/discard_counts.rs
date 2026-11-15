//! Simdgroup operations after an MSL 2.2 discard on the device the cfg selects, under the
//! validation layer.
//!
//! The owned fixture `fragment_discard_counts` (compiled at MSL 2.2) draws Metal's fullscreen
//! triangle over an 8x4 `RGBA32Sint` target cleared to -7. Its `discard_counts` takes the
//! simdgroup's lane count before a discard and after it, and after it the count of lanes whose
//! quad keeps a thread, and a kept thread writes `(after - before, after - kept, before, after)`:
//! differences that hold whatever the simdgroup's size and however quads pack into it.
//!
//! Metal's answer, as an M4 gives it (from the fixture's AIR and from the
//! source compiled at MSL 2.2, with and without the debug layer), is `(0, 0, n, n)` in both modes:
//!
//! - **Half of each quad discarded** (`whole` 0, each quad's right column): a discarded thread
//!   runs on as a helper and stays a lane while its quad keeps a live thread.
//! - **Whole quads discarded** (`whole` 1, a checkerboard of 2x2 quads): a wholly discarded quad
//!   leaves the simdgroup, and is missing from the count before its discard too.
//!
//! The compile layer predicates the discard, so a discarded invocation stays a full participant,
//! and before each later non-quad simdgroup operation kills a quad whose every invocation is
//! discarded or a helper (the whole-quad exit). So every device counts Metal's lanes after the
//! discard. Metal's compiler moves the count ahead of the discard past it (nothing between reads
//! it or has a side effect), and canonicalization moves it there too (SEMANTICS 1094), so it also
//! counts Metal's lanes: every device gives Metal's answer in both modes. Before 1094 the count
//! ran ahead of the discard, counting the quads it drops (`Answer::BeforeCountsDroppedQuads`)
//! wherever a device packed a dropped quad beside a kept one.

use super::buffer_with;
use super::resolves::{lane, quiet, Lane};
use crate::air_library::AirLibrary;
use metal2vulkan::device::VkFeature;
use metal2vulkan::metal::{PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::llvm_assemble;
use metal2vulkan_runtime::{
    BlitOption, ClearColor, LoadAction, PipelineOptions, RenderPassDescriptor, RenderPipelineState,
    ResourceOptions, StoreAction, Texture, TextureDescriptor,
};

/// The target's width and height.
const W: u64 = 8;
const H: u64 = 4;

/// The clear value of every channel.
const CLEAR: i32 = -7;

/// The fixture's pipeline on `lane`'s device, drawing an `RGBA32Sint` target.
fn pipeline(lane: &Lane) -> RenderPipelineState {
    let ll = metal2vulkan_devtools::engine_dir().join("fragment_discard_counts.ll");
    let library = lane
        .device
        .new_library_with_air(&llvm_assemble(&ll).unwrap())
        .unwrap();
    let function = |name| {
        library
            .new_function_with_name(name)
            .unwrap()
            .function_descriptor()
    };
    let mut descriptor =
        RenderPipelineDescriptor::new(StageSource::Function(function("discard_counts_full")));
    descriptor.fragment = Some(StageSource::Function(function("discard_counts")));
    descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA32Sint));
    lane.device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// The private 8x4 `RGBA32Sint` render target.
fn target(lane: &Lane) -> Texture {
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA32Sint, W, H, false);
    descriptor.usage = TextureUsage::RENDER_TARGET;
    descriptor.resource_options = ResourceOptions::private();
    lane.device.new_texture(&descriptor).unwrap()
}

/// The texels, row by row, after one draw with `whole`.
fn drawn(lane: &Lane, pipeline: &RenderPipelineState, whole: u32) -> Vec<[i32; 4]> {
    let target = target(lane);
    let mut descriptor = RenderPassDescriptor::new();
    let attachment = &mut descriptor.color_attachments[0];
    attachment.texture = Some(target.clone());
    attachment.load_action = LoadAction::Clear;
    let clear = f64::from(CLEAR);
    attachment.clear_color = ClearColor::new(clear, clear, clear, clear);
    attachment.store_action = StoreAction::Store;
    lane.run(&descriptor, |lane, encoder, held| {
        encoder.set_render_pipeline_state(pipeline).unwrap();
        let mode = buffer_with(&lane.device, &whole.to_le_bytes()).unwrap();
        encoder.set_fragment_buffer(Some(&mode), 0, 0).unwrap();
        encoder
            .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
            .unwrap();
        held.push(mode);
    })
    .unwrap();
    lane.read(&target, (0, 0), 16, BlitOption::None)
        .as_chunks::<16>()
        .0
        .iter()
        .map(|texel| {
            let word = |i: usize| i32::from_le_bytes(texel[4 * i..4 * i + 4].try_into().unwrap());
            [word(0), word(1), word(2), word(3)]
        })
        .collect()
}

/// Whether the thread at `(x, y)` keeps running past the discard in mode `whole`: an even column
/// in mode 0, a quad of the checkerboard's even squares in mode 1.
fn kept(x: u64, y: u64, whole: u32) -> bool {
    if whole == 0 {
        x & 1 == 0
    } else {
        ((x >> 1) + (y >> 1)) & 1 == 0
    }
}

/// The texels of `whole`'s draw, one line a row.
fn rows(texels: &[[i32; 4]]) -> String {
    texels
        .as_chunks::<{ W as usize }>()
        .0
        .iter()
        .map(|row| format!("{row:?}"))
        .collect::<Vec<_>>()
        .join("\n  ")
}

/// How a mode's kept texels answer beside Metal's `(0, 0, n, n)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Answer {
    /// Every kept texel is `(0, 0, n, n)`: Metal's answer.
    Metals,
    /// Every kept texel counts Metal's lanes after the discard (`after == kept`), and some count
    /// whole quads more before it (`before - after` a positive multiple of 4): quads the discard
    /// drops, which Metal leaves out of the count before the discard too.
    BeforeCountsDroppedQuads,
}

/// The answer of `whole`'s texels: a discarded thread writes nothing, and a kept one counts
/// Metal's lanes after the discard and Metal's or whole quads more before it. `Err`: a texel
/// outside both answers.
fn answer(texels: &[[i32; 4]], whole: u32) -> Result<Answer, String> {
    let mut metals = true;
    for (index, texel) in texels.iter().enumerate() {
        let (x, y) = (index as u64 % W, index as u64 / W);
        let [moved, outside, before, after] = *texel;
        let fits = if kept(x, y, whole) {
            metals &= moved == 0;
            after > 0 && outside == 0 && moved <= 0 && moved % 4 == 0 && before == after - moved
        } else {
            *texel == [CLEAR; 4]
        };
        if !fits {
            return Err(format!("({x}, {y}) wrote {texel:?}"));
        }
    }
    Ok(if metals {
        Answer::Metals
    } else {
        Answer::BeforeCountsDroppedQuads
    })
}

/// Each mode's texels give Metal's answer.
#[test]
fn simdgroup_counts_around_a_discard_answer_as_metals() {
    let (messages, _cache, lane) = lane();
    let name = lane.device.name().to_owned();
    let description = &lane.device.properties().description;
    let maximal = description.has(VkFeature::ShaderMaximalReconvergence);
    let size = description.subgroup.size;
    let pipeline = pipeline(&lane);
    for whole in [0, 1] {
        let texels = drawn(&lane, &pipeline, whole);
        assert_eq!(texels.len(), (W * H) as usize);
        let answer = answer(&texels, whole)
            .unwrap_or_else(|why| panic!("whole {whole} on {name}: {why}\n  {}", rows(&texels)));
        eprintln!(
            "discard counts whole {whole} on {name} (simdgroup {size}, maximal reconvergence \
             {maximal}): {answer:?}\n  {}",
            rows(&texels)
        );
        assert_eq!(
            answer,
            Answer::Metals,
            "whole {whole} on {name} (simdgroup {size})\n  {}",
            rows(&texels)
        );
    }
    quiet(lane, &messages);
}
