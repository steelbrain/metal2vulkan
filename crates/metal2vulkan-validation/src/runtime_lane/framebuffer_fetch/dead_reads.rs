//! A framebuffer read that the function's constants make dead, on the device the cfg selects,
//! under the validation layer: the owned fixture `dfk_dead_fetch`'s `dfk_branch` (the read of
//! color 0 in the arm of a branch on the `bool` constant 0) and `dfk_scaled` (its fetched alpha
//! scaled by the `half` constant 1, under fast math), each returning one `half4` to color 0 at
//! pixel `i = 64 · y + x`: the loaded source, or, with the read live, the source times the fetched
//! color where the fetched alpha is not zero (`dfk_branch`) or the source with the fetched alpha
//! added to its own (`dfk_scaled`, weight one).
//!
//! Color 0 is a `SIDE`² private `BGRA8Unorm` target blended one over one minus the source alpha
//! for color and alpha (a composite's state), cleared to 0.4 in every channel (code 102) and
//! stored; the source is (0.2, 0.2, 0.2, 0) in `half` at every pixel. Specialized with the read
//! dead (constant 0 false, constant 1 zero), each function reads nothing of the attachment, so on
//! every device, one with no fetch form too, its pipeline is a plain one: as the translator makes
//! it for the device, color 0 is a color attachment the device blends, with no storage and no
//! input attachment. A pass of two draws stores (204, 204, 204, 102) in every texel, Metal's (the
//! source added to the destination twice, alpha kept: 0.4 + 2 · 0.2 is 0.8, code 204), each
//! channel within one code of it asserted and the exact ones counted; it takes no storage form for
//! the target (`WorkCounts::storage_forms_taken`), splits no render pass instance for interlocked
//! writes (`instance_splits_interlocked_writes`), and announces none of the fetch's notices
//! (`render-pass-split`, `memoryless-backed`, `fetch-through-scratch`, `fetched-target-storage`).
//!
//! Specialized live, on a device with a fetch form, each pipeline reads color 0 in that form (a
//! written storage attachment in the pixel interlock, an input attachment through
//! rasterization-order access), which is asserted of the translator's pipeline; the live pass's
//! notices, counts and first texel are printed as the contrast. Each function prints a
//! `DEAD-FETCH` line.

use super::{fetches, form, library, through_inputs, FETCHES, STORAGE_FORMS};
use crate::runtime_lane::degrade_notices::counts;
use crate::runtime_lane::resolves::{lane, quiet, Lane, Shape, ONE};
use crate::runtime_lane::{buffer_with, Messages};
use metal2vulkan::library::{FunctionConstantValue, FunctionConstantValues};
use metal2vulkan::metal::{
    BlendFactor, BlendOperation, DataType, PixelFormat, PrimitiveType, TextureUsage,
};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan::Budget;
use metal2vulkan_runtime::{
    BlitOption, ClearColor, Degradation, Library, LoadAction, PipelineOptions,
    RenderPassDescriptor, StoreAction,
};
use std::collections::BTreeSet;

/// The target's side: large enough that `STORAGE` costs the target its compression on Apple's GPU
/// (from 16² up), so a fetch there takes a storage form for it or a scratch.
const SIDE: u64 = 16;

/// The source buffer's pixels: the fixture indexes it by `64 · y + x`.
const SOURCES: usize = 64 * 64;

/// The fixture's functions whose read the constants make dead.
const ENTRIES: [&str; 2] = ["dfk_branch", "dfk_scaled"];

/// Metal's texel after the dead pass's two draws, in memory order (blue, green, red, alpha).
const DEAD_TEXEL: [u8; 4] = [204, 204, 204, 102];

/// The fixture's function `name` of `library`, specialized with the read live or not.
fn specialized(library: &Library, name: &str, live: bool) -> StageSource {
    let weight: u16 = if live { 0x3c00 } else { 0 };
    let constants = FunctionConstantValues::defaults()
        .with_index(
            0,
            FunctionConstantValue::new(DataType::Bool, [u8::from(live)]),
        )
        .with_index(
            1,
            FunctionConstantValue::new(DataType::Half, weight.to_le_bytes()),
        );
    StageSource::Function(
        library
            .new_function_with_name_constant_values(name, &constants)
            .unwrap()
            .function_descriptor(),
    )
}

/// The pipeline of `entry`, live or not, behind `dfk_vertex`: color 0 `BGRA8Unorm` blended one
/// over one minus the source alpha.
fn descriptor(library: &Library, entry: &str, live: bool) -> RenderPipelineDescriptor {
    let vertex = library
        .new_function_with_name("dfk_vertex")
        .unwrap()
        .function_descriptor();
    let mut d = RenderPipelineDescriptor::new(StageSource::Function(vertex));
    d.fragment = Some(specialized(library, entry, live));
    let mut color = ColorAttachmentDescriptor::new(PixelFormat::BGRA8Unorm);
    color.blending_enabled = true;
    color.source_rgb_blend_factor = BlendFactor::One;
    color.destination_rgb_blend_factor = BlendFactor::OneMinusSourceAlpha;
    color.rgb_blend_operation = BlendOperation::Add;
    color.source_alpha_blend_factor = BlendFactor::One;
    color.destination_alpha_blend_factor = BlendFactor::OneMinusSourceAlpha;
    color.alpha_blend_operation = BlendOperation::Add;
    d.color_attachments[0] = Some(color);
    d
}

/// What a pass of two draws of `descriptor`'s pipeline did: the target's texels, the fetch's
/// notices it announced, and how many storage forms it took and instances it split for
/// interlocked writes.
struct Drawn {
    texels: Vec<u8>,
    notices: BTreeSet<Degradation>,
    storage_forms: u64,
    interlocked_splits: u64,
}

/// Draw `descriptor`'s pipeline twice over a fresh target cleared to 0.4, the source
/// (0.2, 0.2, 0.2, 0) at every pixel.
fn two_draws(lane: &Lane, messages: &Messages, descriptor: &RenderPipelineDescriptor) -> Drawn {
    let state = lane
        .device
        .new_render_pipeline_state(descriptor, &PipelineOptions::default())
        .unwrap();
    let shape = Shape { size: SIDE, ..ONE };
    let target = lane.texture(PixelFormat::BGRA8Unorm, shape, TextureUsage::RENDER_TARGET);
    // 0.2 in `half` is 0x3266; the alpha 0.
    let source: Vec<u8> = [0x3266u16, 0x3266, 0x3266, 0]
        .iter()
        .flat_map(|h| h.to_le_bytes())
        .collect();
    let src = buffer_with(&lane.device, &source.repeat(SOURCES)).unwrap();
    let mut pass = RenderPassDescriptor::new();
    let attachment = &mut pass.color_attachments[0];
    attachment.texture = Some(target.clone());
    attachment.load_action = LoadAction::Clear;
    attachment.clear_color = ClearColor::new(0.4, 0.4, 0.4, 0.4);
    attachment.store_action = StoreAction::Store;
    let _ = messages.take_notices();
    let before = lane.device.work_counts();
    lane.run(&pass, |_, encoder, _| {
        encoder.set_render_pipeline_state(&state).unwrap();
        encoder.set_fragment_buffer(Some(&src), 0, 0).unwrap();
        for _ in 0..2 {
            encoder
                .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
                .unwrap();
        }
    })
    .unwrap();
    let texels = lane.read(&target, (0, 0), 4, BlitOption::None);
    let after = lane.device.work_counts();
    let fetch_kinds: BTreeSet<Degradation> = FETCHES.into_iter().chain(STORAGE_FORMS).collect();
    let notices = counts(&messages.take_notices())
        .into_keys()
        .filter(|k| fetch_kinds.contains(k))
        .collect();
    Drawn {
        texels,
        notices,
        storage_forms: after.storage_forms_taken - before.storage_forms_taken,
        interlocked_splits: after.instance_splits_interlocked_writes
            - before.instance_splits_interlocked_writes,
    }
}

#[test]
fn a_framebuffer_read_dead_after_specialization_draws_as_a_plain_pipeline() {
    let (messages, cache, lane) = lane();
    let library = library(&lane, "dfk_dead_fetch");
    let device = lane.device.name().to_owned();
    // Every dead pass first, so no live pass announces a kind of notice before them.
    for entry in ENTRIES {
        let what = format!("{entry} on {device}, dead");
        let dead = descriptor(&library, entry, false);
        let made = lane
            .device
            .compile_device()
            .compile_render_pipeline(&dead, &Budget::standard())
            .unwrap_or_else(|e| panic!("{what}: {e}"));
        assert!(
            made.graphics.storage_attachments.is_empty(),
            "{what}: {:?}",
            made.graphics.storage_attachments
        );
        assert!(
            made.graphics.input_attachments.is_empty(),
            "{what}: {:?}",
            made.graphics.input_attachments
        );
        let color = made.graphics.color_attachments.first().copied().flatten();
        assert!(
            color.is_some_and(|c| c.blend.is_some()),
            "{what}: color 0 {color:?}"
        );
        let drawn = two_draws(&lane, &messages, &dead);
        let mut exact = 0;
        for (i, texel) in drawn.texels.chunks_exact(4).enumerate() {
            for (c, (&got, &want)) in texel.iter().zip(&DEAD_TEXEL).enumerate() {
                assert!(
                    got.abs_diff(want) <= 1,
                    "{what}: texel {i} channel {c}: {got}, Metal's {want}"
                );
                exact += usize::from(got == want);
            }
        }
        assert_eq!(drawn.notices, BTreeSet::new(), "{what}: notices");
        assert_eq!(drawn.storage_forms, 0, "{what}: storage forms taken");
        assert_eq!(drawn.interlocked_splits, 0, "{what}: interlocked splits");
        let channels = drawn.texels.len();
        eprintln!(
            "DEAD-FETCH {what}: a plain pipeline, {exact} of {channels} channels Metal's \
             exactly, the rest within one code; no storage form, no fetch notice"
        );
    }
    if fetches(&lane.device) {
        for entry in ENTRIES {
            let what = format!("{entry} on {device}, live");
            let live = descriptor(&library, entry, true);
            let made = lane
                .device
                .compile_device()
                .compile_render_pipeline(&live, &Budget::standard())
                .unwrap_or_else(|e| panic!("{what}: {e}"));
            let storage: Vec<(u32, bool)> = (made.graphics.storage_attachments.iter())
                .map(|u| (u.color_index, u.writes))
                .collect();
            let inputs: Vec<u32> = (made.graphics.input_attachments.iter())
                .map(|u| u.color_index)
                .collect();
            match through_inputs(&lane) {
                true => {
                    assert_eq!(inputs, [0], "{what}");
                    assert_eq!(storage, [], "{what}");
                }
                false => {
                    assert_eq!(storage, [(0, true)], "{what}");
                    assert!(made.graphics.color_attachments[0].is_none(), "{what}");
                }
            }
            let drawn = two_draws(&lane, &messages, &live);
            eprintln!(
                "DEAD-FETCH {what}, through {}: notices {:?}, {} storage forms taken, {} \
                 interlocked splits, first texel {:?}",
                form(&lane),
                drawn.notices,
                drawn.storage_forms,
                drawn.interlocked_splits,
                &drawn.texels[..4]
            );
        }
    }
    drop(cache);
    quiet(lane, &messages);
}
