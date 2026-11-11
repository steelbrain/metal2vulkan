//! Framebuffer fetch through input attachments ordered between draws alone, on the device the cfg
//! selects with rasterization-order attachment access and the fragment shader interlock withheld
//! (`DeviceOptions::withheld_extensions` and `withheld_features`): the form of a device that has
//! neither (Honeykrisp on Apple's M2 Pro; RADV on the RX 550, GCN4, which has neither of its own),
//! under the validation layer.
//!
//! The runtime reads each fetched color as the input attachment of its index and records a barrier
//! of the draws' subpass's dependency on itself, by region, before a draw that reads input
//! attachments after an earlier draw of its render pass instance wrote a color, and before a draw
//! that writes a color after an earlier one read them (`WorkCounts::fetch_draw_barriers`). Each
//! pass draws the fullscreen triangle (`rt6_resolves`'s `rt6_resolve_full`) over a 4² target:
//!
//! - **Draws in order**: an `R32Float` target cleared to 0.75, then `fetch_scalar`
//!   (`fragment_fetch`, half what it reads), `rt6_resolve_white` (1), `fetch_scalar` twice: 0.25,
//!   Metal's answer, which holds only where each draw reads what the draw before it stored (a read
//!   of the clear in each fetch leaves 0.375; a fetch that missed the white write, 0.09375). Three
//!   barriers: before the white write (after the first fetch's reads) and before each later fetch
//!   (after the write before it).
//! - **Integer read-modify-writes**: an `RGBA32Uint` target cleared to (1, 2, 3, 4), then
//!   `exf_fetch_count` (three times what it reads, plus `k + 1` in lane `k`) three times:
//!   `c · 27 + (k + 1) · 13`, exactly. Two barriers.
//! - **One draw over itself**: `fetch_scalar` drawn as two instances of the fullscreen triangle in
//!   one draw over the `R32Float` target cleared to 0.75. Metal halves each texel twice (0.1875).
//!   Nothing orders the second instance's fragments after the first's here, so each texel holds
//!   0.1875 or 0.375 (the second read before the first's write landed), printed with its count;
//!   the pipeline's fragment stage records `PrecisionNote::UnorderedFetchWithinDraw`. No barrier:
//!   one draw.
//!
//! The runtime announces the note (`unordered-fetch-within-draw`) once, where the first fetching
//! pipeline is made, as it announces a stage's other notes; no draw announces it again.
//!
//! Each fetching pipeline reads one input attachment and states no rasterization-order access,
//! which is asserted of the translator's pipeline.

use super::super::degrade_notices::counts;
use super::super::resolves::{lane_with, quiet, Lane, ONE, WHITE};
use super::super::{Messages, ScratchCache};
use super::{library, pass, pipeline, pipeline_descriptor, steps, TEXELS, Z};
use crate::runtime_lane::buffer_with;
use metal2vulkan::device::{VkExtension, VkFeature};
use metal2vulkan::metal::{PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan::pipeline::PrecisionNote;
use metal2vulkan::Budget;
use metal2vulkan_runtime::{BlitOption, ClearColor, Degradation, Library, StoreAction};

/// The lane's device with neither ordered fetch form: rasterization-order attachment access and
/// the fragment shader interlock withheld, extensions and features both.
fn unordered_lane() -> (Messages, ScratchCache, Lane) {
    lane_with(|options| {
        options.withheld_extensions.extend([
            VkExtension::ExtRasterizationOrderAttachmentAccess,
            VkExtension::ExtFragmentShaderInterlock,
        ]);
        options.withheld_features.extend([
            VkFeature::RasterizationOrderColorAttachmentAccess,
            VkFeature::RasterizationOrderDepthAttachmentAccess,
            VkFeature::RasterizationOrderStencilAttachmentAccess,
            VkFeature::FragmentShaderPixelInterlock,
            VkFeature::FragmentShaderSampleInterlock,
        ]);
    })
}

/// The device's `fetch_draw_barriers` so far.
fn barriers(lane: &Lane) -> u64 {
    lane.device.work_counts().fetch_draw_barriers
}

/// The kind the fetching fragment stage records.
fn note() -> Degradation {
    Degradation::from_precision_note(PrecisionNote::UnorderedFetchWithinDraw)
}

/// Every fetch of the lane, drawn in order: Metal's answers between draws, the note announced, the
/// barriers counted, and a draw over itself one of the orders of its fragments (the module docs).
#[test]
fn fetches_between_draws_read_every_earlier_draw_and_note_the_draw_itself() {
    let (messages, cache, lane) = unordered_lane();
    let device = lane.device.name().to_owned();
    let fragment = lane.device.compile_device().profile().fragment();
    assert!(
        !fragment.rasterization_order_color_access && !fragment.pixel_interlock,
        "{device}: an ordered fetch form left in the profile"
    );
    let vertices = library(&lane, "rt6_resolves");
    let fetch = library(&lane, "fragment_fetch");
    let count = library(&lane, "exf_fetch_count");
    let pf = PixelFormat::R32Float;
    // The translator's pipeline of each fetching function: one input attachment, no
    // rasterization-order access, no storage form.
    let inputs = |(library, entry): (&Library, &str), pf: PixelFormat| {
        let descriptor = pipeline_descriptor(&vertices, (library, entry), &[pf], 1);
        let made = (lane.device.compile_device())
            .compile_render_pipeline(&descriptor, &Budget::standard())
            .unwrap_or_else(|e| panic!("{entry} on {device}: {e}"));
        let graphics = &made.graphics;
        assert!(
            !graphics.rasterization_order_attachment_access,
            "{entry} on {device}: a fetch in rasterization order"
        );
        assert!(
            graphics.storage_attachments.is_empty(),
            "{entry} on {device}"
        );
        let read: Vec<u32> = (graphics.input_attachments.iter())
            .map(|input| input.color_index)
            .collect();
        assert_eq!(read, [0], "{entry} on {device}: its input attachments");
    };
    inputs((&fetch, "fetch_scalar"), pf);
    inputs((&count, "exf_fetch_count"), PixelFormat::RGBA32Uint);
    let _ = messages.take_notices();
    let halve = pipeline(&lane, &vertices, (&fetch, "fetch_scalar"), &[pf], 1).unwrap();
    let white = pipeline(&lane, &vertices, (&vertices, WHITE), &[pf], 1).unwrap();
    let announced = counts(&messages.take_notices());
    assert_eq!(announced.get(&note()), Some(&1), "{device}: {announced:?}");

    // Draws in order: 0.25, three barriers.
    let before = barriers(&lane);
    let target = lane.texture(pf, ONE, TextureUsage::RENDER_TARGET);
    let clear = ClearColor::new(0.75, 0.0, 0.0, 0.0);
    let targets = [(&target, clear, StoreAction::Store)];
    steps(
        &lane,
        &pass(&targets, None),
        &[&halve, &white, &halve, &halve],
    )
    .unwrap();
    let read = lane.read(&target, (0, 0), 4, BlitOption::None);
    assert_eq!(
        read,
        0.25f32.to_le_bytes().repeat(TEXELS),
        "{device}: draws in order"
    );
    assert_eq!(
        barriers(&lane) - before,
        3,
        "{device}: barriers between the draws"
    );
    eprintln!("FETCH-BETWEEN-DRAWS draws in order on {device}: 0.25 at every texel, 3 barriers");

    // Integer read-modify-writes: exact, two barriers.
    let integers = [PixelFormat::RGBA32Uint];
    let step = pipeline(&lane, &vertices, (&count, "exf_fetch_count"), &integers, 1).unwrap();
    let before = barriers(&lane);
    let target = lane.texture(PixelFormat::RGBA32Uint, ONE, TextureUsage::RENDER_TARGET);
    let clear = ClearColor::new(1.0, 2.0, 3.0, 4.0);
    let targets = [(&target, clear, StoreAction::Store)];
    steps(&lane, &pass(&targets, None), &[&step, &step, &step]).unwrap();
    let texel: Vec<u8> = (1u32..=4).flat_map(|c| (c * 40).to_le_bytes()).collect();
    assert_eq!(
        lane.read(&target, (0, 0), 16, BlitOption::None),
        texel.repeat(TEXELS),
        "{device}: integer read-modify-writes"
    );
    assert_eq!(
        barriers(&lane) - before,
        2,
        "{device}: barriers between the draws"
    );
    eprintln!("FETCH-BETWEEN-DRAWS integer read-modify-writes on {device}: exact, 2 barriers");

    // One draw over itself: each texel one of the two orders of its fragments, no barrier.
    let before = barriers(&lane);
    let target = lane.texture(pf, ONE, TextureUsage::RENDER_TARGET);
    let clear = ClearColor::new(0.75, 0.0, 0.0, 0.0);
    let targets = [(&target, clear, StoreAction::Store)];
    lane.run(&pass(&targets, None), |lane, encoder, held| {
        let depth = buffer_with(&lane.device, &Z.to_le_bytes()).unwrap();
        encoder.set_vertex_buffer(Some(&depth), 0, 0).unwrap();
        encoder.set_render_pipeline_state(&halve).unwrap();
        (encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 2, 0))
            .unwrap_or_else(|e| panic!("{device}: one draw over itself: {e}"));
        held.push(depth);
    })
    .unwrap();
    let read = lane.read(&target, (0, 0), 4, BlitOption::None);
    let (mut metals, mut other) = (0usize, 0usize);
    for texel in read.as_chunks::<4>().0 {
        let bits = u32::from_le_bytes(*texel);
        if bits == 0.1875f32.to_bits() {
            metals += 1;
        } else if bits == 0.375f32.to_bits() {
            other += 1;
        } else {
            panic!(
                "{device}: one draw over itself stored {}",
                f32::from_bits(bits)
            );
        }
    }
    assert_eq!(
        barriers(&lane) - before,
        0,
        "{device}: a barrier inside one draw"
    );
    eprintln!(
        "FETCH-BETWEEN-DRAWS one draw over itself on {device}: {metals} of {TEXELS} texels \
         Metal's 0.1875, {other} the other order's 0.375 (unordered-fetch-within-draw)"
    );
    // Announced once, at the first fetching pipeline's making: no later making or draw again.
    let again = counts(&messages.take_notices());
    assert_eq!(again.get(&note()), None, "{device}: {again:?}");
    drop(cache);
    quiet(lane, &messages);
}
