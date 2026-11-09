//! Plain color writers whose color a render pass moves into the interlocked storage form, on the
//! device the cfg selects, under the validation layer, against Metal's answer.
//!
//! An extended-range color that a pass clears between two of its codes holds its `half` shadow
//! from the first draw that blends into it, in the pixel interlock (`render/extended_shadow.rs`).
//! From then on every draw that writes the color runs the variant that writes it, and the
//! shadow, through the storage form inside the fragment function: before the depth and stencil
//! tests, which the variant selects from the draw's state (`BindState::fragment_tests`).
//!
//! Each case is one pass over a 4² `RGB10A8_2P_XR10` color cleared to (0, 0, 0, 0.5) (alpha
//! between two of its plane's codes) and a `Depth32Float_Stencil8` attachment cleared to depth 0.5
//! and stencil 0, of three draws of `smw_writers`' fullscreen triangle at depth 0.25: `smw_plain`
//! blended (premultiplied over) writing (0, 1, 0.5, 1), which starts the shadow and leaves its own
//! output, then a writer of (1, 0.5, 0, 1) under the case's depth-stencil state and stencil
//! reference: `smw_discard`, which discards the fragments of pixel column 0, or `smw_store`, which
//! does so and counts each run it does not discard in buffer 1; then `smw_plain` blended again,
//! writing (0, 0, 0, 0), which leaves every texel as it was and reads the color where the second
//! draw left it. Metal tests each fragment after the function: color 0 holds the second draw's codes
//! where the fragment survives its discard and its tests, the first draw's elsewhere, and
//! `smw_store` runs for every fragment it does not discard, those its tests then drop included (12
//! of 16). That is Metal's answer to every case, each pixel's codes and the count (read on an Apple
//! M3 Max, macOS 26.7, through the same fixture). One more case draws `smw_store` under a write mask
//! that covers none of the color's channels, which stores nothing: color 0 holds the first draw's
//! codes everywhere, as Metal's write mask leaves it (its run count, which no reading covers, is
//! printed and not checked).
//!
//! The path: a device that fetches in the pixel interlock holds the shadow through the second
//! draw, whose variant writes it, so the third draw blends onto it and nothing announces
//! `extended-range-blends-stored-codes`; a second draw that gave the shadow up would have the
//! third announce it. A device with raster-order attachment access (lavapipe) holds no shadow,
//! draws each as compiled, and announces the stored codes once, at the first case: the same
//! answer. The masked writer stores nothing, so no test bears on its write and its draw keeps the
//! shadow under tests that may drop fragments, which a writer that stores something does not
//! survive (its variant is refused, and the shadow ends).

use super::degrade_notices::counts;
use super::extended_range_blends::STORED_CODES;
use super::framebuffer_fetch::{fetches, through_inputs};
use super::resolves::{lane, quiet, Lane};
use crate::air_library::AirLibrary;
use crate::runtime_lane::{buffer_with, read_buffer};
use metal2vulkan::metal::state::DepthStencilDescriptor;
use metal2vulkan::metal::{
    BlendFactor, ColorWriteMask, CompareFunction, PixelFormat, PrimitiveType,
};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::llvm_assemble;
use metal2vulkan_runtime::{
    BlitOption, ClearColor, Library, LoadAction, PipelineOptions, RenderPassDescriptor,
    RenderPipelineState, StoreAction,
};
use std::path::Path;

/// The pass's color format.
const COLOR: PixelFormat = PixelFormat::RGB10A82PXR10;

/// The pass's depth and stencil format.
const DEPTH: PixelFormat = PixelFormat::Depth32FloatStencil8;

/// The 4² targets' side.
const SIDE: usize = 4;

/// The codes (red, green, blue) of the first draw's (0, 1, 0.5) and the second's (1, 0.5, 0):
/// `384 + 510 v` each, exact.
const FIRST: [u32; 3] = [384, 894, 639];
const SECOND: [u32; 3] = [894, 639, 384];

/// `smw_writers`' library on `lane`'s device.
fn library(lane: &Lane) -> Library {
    let ll = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/public/engine/smw_writers.ll");
    lane.device
        .new_library_with_air(&llvm_assemble(&ll).unwrap())
        .unwrap()
}

/// The pipeline of `smw_full` and `fragment` into the pass's attachments, blended premultiplied
/// over where `blended`, under a write mask that covers no channel where `masked`.
fn pipeline(
    lane: &Lane,
    library: &Library,
    fragment: &str,
    (blended, masked): (bool, bool),
) -> RenderPipelineState {
    let function = |name| {
        StageSource::Function(
            library
                .new_function_with_name(name)
                .unwrap()
                .function_descriptor(),
        )
    };
    let mut descriptor = RenderPipelineDescriptor::new(function("smw_full"));
    descriptor.fragment = Some(function(fragment));
    let mut color = ColorAttachmentDescriptor::new(COLOR);
    if blended {
        color.blending_enabled = true;
        color.destination_rgb_blend_factor = BlendFactor::OneMinusSourceAlpha;
        color.destination_alpha_blend_factor = BlendFactor::OneMinusSourceAlpha;
    }
    if masked {
        color.write_mask = ColorWriteMask::NONE;
    }
    descriptor.color_attachments[0] = Some(color);
    descriptor.depth_format = Some(DEPTH);
    descriptor.stencil_format = Some(DEPTH);
    lane.device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// `values` as `half` bytes.
fn halves(values: [f32; 4]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|&v| super::surface_aliases::half_bits(v).to_le_bytes())
        .collect()
}

/// A depth-stencil state comparing depth by `compare`, written where `write`.
fn depth(compare: CompareFunction, write: bool) -> DepthStencilDescriptor {
    let mut d = DepthStencilDescriptor::default();
    d.depth_compare_function = compare;
    d.depth_write_enabled = write;
    d
}

#[test]
fn a_moved_writer_keeps_what_metals_later_tests_drop() {
    let (messages, cache, lane) = lane();
    if !fetches(&lane.device) {
        return;
    }
    let device = lane.device.name();
    let library = library(&lane);
    let first = pipeline(&lane, &library, "smw_plain", (true, false));
    let mut equal = DepthStencilDescriptor::default();
    for face in [&mut equal.front_face_stencil, &mut equal.back_face_stencil] {
        face.stencil_compare_function = CompareFunction::Equal;
    }
    // The case, the second draw's writer, its state and reference, whether it stores the
    // fragments (each in a column but 0): its tests pass them and its mask writes the color, and
    // whether its mask covers no channel.
    let cases = [
        ("no state", "smw_discard", None, 0, true, false),
        (
            "depth less, unwritten",
            "smw_discard",
            Some(depth(CompareFunction::Less, false)),
            0,
            true,
            false,
        ),
        (
            "depth greater, unwritten",
            "smw_discard",
            Some(depth(CompareFunction::Greater, false)),
            0,
            false,
            false,
        ),
        (
            "depth less, written",
            "smw_discard",
            Some(depth(CompareFunction::Less, true)),
            0,
            true,
            false,
        ),
        (
            "stencil equal, failing",
            "smw_discard",
            Some(equal),
            1,
            false,
            false,
        ),
        (
            "stencil equal, passing",
            "smw_discard",
            Some(equal),
            0,
            true,
            false,
        ),
        (
            "counting, depth greater, unwritten",
            "smw_store",
            Some(depth(CompareFunction::Greater, false)),
            0,
            false,
            false,
        ),
        (
            "counting, depth less, written",
            "smw_store",
            Some(depth(CompareFunction::Less, true)),
            0,
            true,
            false,
        ),
        (
            "counting, write mask none, depth greater, unwritten",
            "smw_store",
            Some(depth(CompareFunction::Greater, false)),
            0,
            false,
            true,
        ),
    ];
    let mut stored_codes = 0;
    for (case, writer, ds, reference, passes, masked) in cases {
        let second = pipeline(&lane, &library, writer, (false, masked));
        let target = lane.single(COLOR);
        let depth_texture = lane.single(DEPTH);
        let mut pass = RenderPassDescriptor::new();
        let color = &mut pass.color_attachments[0];
        color.texture = Some(target.clone());
        color.load_action = LoadAction::Clear;
        color.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.5);
        color.store_action = StoreAction::Store;
        pass.depth_attachment.texture = Some(depth_texture.clone());
        pass.depth_attachment.load_action = LoadAction::Clear;
        pass.depth_attachment.clear_depth = 0.5;
        pass.depth_attachment.store_action = StoreAction::DontCare;
        pass.stencil_attachment.texture = Some(depth_texture.clone());
        pass.stencil_attachment.load_action = LoadAction::Clear;
        pass.stencil_attachment.store_action = StoreAction::DontCare;
        let ds_state = ds.map(|d| lane.device.new_depth_stencil_state(&d).unwrap());
        let runs = buffer_with(&lane.device, &[0u8; 4]).unwrap();
        let mut drawn = None;
        lane.run(&pass, |lane, encoder, held| {
            let z = buffer_with(&lane.device, &0.25f32.to_le_bytes()).unwrap();
            let tints = [
                halves([0.0, 1.0, 0.5, 1.0]),
                halves([1.0, 0.5, 0.0, 1.0]),
                halves([0.0; 4]),
            ]
            .map(|bytes| buffer_with(&lane.device, &bytes).unwrap());
            encoder.set_vertex_buffer(Some(&z), 0, 0).unwrap();
            encoder.set_render_pipeline_state(&first).unwrap();
            encoder.set_fragment_buffer(Some(&tints[0]), 0, 0).unwrap();
            encoder
                .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
                .unwrap();
            if let Some(ds_state) = &ds_state {
                encoder.set_depth_stencil_state(ds_state).unwrap();
            }
            encoder.set_stencil_reference_value(reference);
            encoder.set_render_pipeline_state(&second).unwrap();
            encoder.set_fragment_buffer(Some(&tints[1]), 0, 0).unwrap();
            encoder.set_fragment_buffer(Some(&runs), 0, 1).unwrap();
            drawn = Some(encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0));
            encoder.set_render_pipeline_state(&first).unwrap();
            encoder.set_fragment_buffer(Some(&tints[2]), 0, 0).unwrap();
            encoder
                .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
                .unwrap();
            held.extend([z]);
            held.extend(tints);
        })
        .unwrap_or_else(|error| panic!("{case}: the pass on {device}: {error}"));
        drawn
            .expect("the second draw")
            .unwrap_or_else(|error| panic!("{case}: the second draw on {device}: {error}"));
        // Plane 0's words: red, green and blue's 10-bit codes.
        let texels = lane.read(&target, (0, 0), 4, BlitOption::None);
        for (i, word) in texels.chunks(4).enumerate() {
            let word = u32::from_le_bytes([word[0], word[1], word[2], word[3]]);
            let codes = [(word >> 20) & 0x3FF, (word >> 10) & 0x3FF, word & 0x3FF];
            let stored = passes && i % SIDE != 0;
            let want = if stored { SECOND } else { FIRST };
            assert_eq!(codes, want, "{case}: pixel {i} on {device}");
        }
        let notices = counts(&messages.take_notices());
        stored_codes += notices.get(&STORED_CODES[0]).copied().unwrap_or(0);
        if writer == "smw_store" {
            let counted = u32::from_le_bytes(read_buffer(&runs, 0, 4).unwrap().try_into().unwrap());
            match masked {
                true => eprintln!("{case}: {counted} runs counted on {device}"),
                false => assert_eq!(counted, 12, "{case}: the runs counted on {device}"),
            }
        }
        eprintln!(
            "moved writer, {case}, on {device} {}: Metal's answer",
            match through_inputs(&lane) {
                true => "through input attachments",
                false => "in the pixel interlock, the shadow held",
            }
        );
    }
    // The shadow lasted through every writer where the device holds one.
    let held = !through_inputs(&lane);
    assert_eq!(
        stored_codes,
        if held { 0 } else { 1 },
        "the stored codes announced on {device}"
    );
    drop(cache);
    quiet(lane, &messages);
}
