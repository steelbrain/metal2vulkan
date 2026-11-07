//! Device tests of attachments stored `DontCare` or `MultisampleResolve` (the runtime's
//! `Device::degrade_pass`, `DriverFact::DontCareStoreIsMetals`) on the device the cfg selects,
//! under the validation layer.
//!
//! Metal keeps such an attachment's earlier texels on Apple's GPUs unless the tiler overflows
//! (an M4 with macOS 27.0.1), but a pass that clears it, draws nothing, and resolves it alone,
//! which leaves the clear where Metal compresses the texture. Vulkan leaves the texels undefined.
//! On a driver with the fact (MoltenVK 1.4.1 on the Apple M4), which hands the store to Metal
//! unchanged, a pass recorded as one render pass instance keeps the texels and announces no
//! `dont-care-store`; every other driver announces it once, and each of its
//! texels must be one of Metal's two answers, the earlier texel or the one the same pass stores
//! with `Store` (Metal's under its tiler's overflow), as the notice's exactness (`OneOfMetals`)
//! states; the counts are printed (`DONT-CARE-TEXELS`). Only a driver measured to keep one of the
//! two (`DriverFact::DontCareStoreIsOneOfMetals`) is handed `DONT_CARE`; every other device stores
//! the attachment, and each of its texels must be one of the two too.
//!
//! Those cases run under the runtime's default `DontCareColorPolicy::Driver` (set over the lane's
//! `exact_answers()`), where a color attachment of a texture's own image stored `DontCare` is
//! stored as the driver stores it and no pass renders through a backing
//! (`WorkCounts::kept_texels_passes` stays 0). Under `DontCareColorPolicy::Kept`, which
//! `DevicePolicy::exact_answers()` selects, such an attachment keeps its earlier texels exactly
//! on every driver and announces no `dont-care-store`: on a driver with the fact, in a pass
//! recorded as one render pass instance, handed to Metal; everywhere else, and in a pass the
//! driver does not run unbroken (the two barriers below), through a backing its queue keeps
//! (`kept_bytes.rs`), each such pass counted.
//!
//! The negative cases announce it on every driver, each on a device of its own (the notice is
//! once per device), their texels printed: a barrier over the render targets between two draws,
//! inside the instance, where MoltenVK restarts its Metal pass and stores the attachment; one
//! over buffers, which splits the pass into two instances, the first storing; a pass that clears
//! a 4x attachment, draws nothing and resolves it; and a pass at custom sample positions. A pass
//! whose render area (the descriptor's target size) is smaller than its attachment is a positive
//! case: the runtime's framebuffer is the render area, so MoltenVK renders its whole framebuffer.
//!
//! The draws are the owned fixture `rt6_restart`'s (`clear_reach`'s lane): a triangle over the
//! whole target at depth 0.25, writing (1/255, 0, 0, 1), the depth always passing and written,
//! the stencil replaced by 0x41.

use super::clear_reach::{
    lane, lane_with, quiet, Draw, Lane, Shape, AREA, DRAWN, OLD, PLAIN, SIDE,
};
use super::degrade_notices::{
    assert_made_once, forced, resolve_kind, shared_in_system_memory, tie_cleared,
};
use super::{buffer_with, Messages, WAIT};
use metal2vulkan::device::{DontCareColorPolicy, DriverFact, VkExtension};
use metal2vulkan::metal::state::DepthStencilDescriptor;
use metal2vulkan::metal::{
    BarrierScope, CompareFunction, PixelFormat, PrimitiveType, RenderStages, TextureUsage,
};
use metal2vulkan::vk::SampleCountFlags;
use metal2vulkan_runtime::{
    BlitOption, ClearColor, Degradation, LoadAction, MultisampleDepthResolveFilter,
    RenderCommandEncoder, RenderPassDescriptor, RenderPipelineState, SamplePosition, StoreAction,
    Texture,
};
use std::collections::BTreeSet;

const RGBA8: PixelFormat = PixelFormat::RGBA8Unorm;
const TARGET: TextureUsage = TextureUsage::RENDER_TARGET;

/// A 4x texture of the lane's side.
const MSAA: Shape = Shape {
    samples: 4,
    ..PLAIN
};

/// The color a 4x texture is filled with before the pass (its earlier texels).
fn earlier() -> ClearColor {
    ClearColor::new(0.25, 0.5, 0.75, 1.0)
}

/// A pass over color attachment `texture`, loaded by `load` (clearing to (0.5, 0, 0, 1)) and stored
/// by `store` into `resolve` where given.
fn color_pass(
    texture: &Texture,
    load: LoadAction,
    store: StoreAction,
    resolve: Option<&Texture>,
) -> RenderPassDescriptor {
    let mut descriptor = RenderPassDescriptor::new();
    let target = &mut descriptor.color_attachments[0];
    target.texture = Some(texture.clone());
    target.load_action = load;
    target.clear_color = ClearColor::new(0.5, 0.0, 0.0, 1.0);
    target.store_action = store;
    target.resolve_texture = resolve.cloned();
    descriptor
}

/// A 4x color texture of `pf` holding [`earlier`] in every sample: cleared and stored by a pass
/// of its own.
fn earlier_msaa(lane: &Lane, pf: PixelFormat) -> Texture {
    let texture = lane.texture(pf, MSAA, TARGET);
    let mut descriptor = color_pass(&texture, LoadAction::Clear, StoreAction::Store, None);
    descriptor.color_attachments[0].clear_color = earlier();
    lane.run(&descriptor, Draw::None, None);
    texture
}

/// What 4x color texture `texture` of `pf` holds, as one texel a pixel: a pass that loads it,
/// draws nothing and resolves it into a new texture, read back (every sample alike resolves to
/// itself, as an untouched texture's does: compared with [`earlier_msaa`]'s).
fn resolved(lane: &Lane, texture: &Texture, pf: PixelFormat) -> Vec<u8> {
    let into = lane.texture(pf, PLAIN, TARGET);
    let descriptor = color_pass(
        texture,
        LoadAction::Load,
        StoreAction::MultisampleResolve,
        Some(&into),
    );
    lane.run(&descriptor, Draw::None, None);
    lane.read(&into, (0, 0), 4, BlitOption::None)
}

/// Encode `descriptor`'s pass through `pipeline`, its vertex buffer the triangle over the whole
/// target, `body` drawing it ([`triangle`]); then commit and wait.
fn encode(
    lane: &Lane,
    descriptor: &RenderPassDescriptor,
    pipeline: &RenderPipelineState,
    body: impl FnOnce(&mut RenderCommandEncoder),
) {
    let command_buffer = lane.queue.command_buffer().unwrap();
    let mut encoder = command_buffer.render_command_encoder(descriptor).unwrap();
    encoder.set_render_pipeline_state(pipeline).unwrap();
    let corners: [f32; 6] = [-1.0, -1.0, 3.0, -1.0, -1.0, 3.0];
    let bytes: Vec<u8> = corners.iter().flat_map(|v| v.to_le_bytes()).collect();
    let corners = buffer_with(&lane.device, &bytes).unwrap();
    encoder.set_vertex_buffer(Some(&corners), 0, 0).unwrap();
    body(&mut encoder);
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
}

/// Draw [`encode`]'s triangle.
fn triangle(encoder: &mut RenderCommandEncoder) {
    encoder
        .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
        .unwrap();
}

/// Whether the lane's device stores `DONT_CARE` as Metal's own `DontCare`.
fn metals(lane: &Lane) -> bool {
    lane.device
        .properties()
        .driver_facts
        .contains(&DriverFact::DontCareStoreIsMetals)
}

/// Whether the runtime hands the lane's device `DONT_CARE` for Metal's `DontCare`: a driver that
/// stores it as Metal's own, or keeps one of Metal's two answers; every other device stores.
fn discards(lane: &Lane) -> bool {
    let facts = &lane.device.properties().driver_facts;
    metals(lane) || facts.contains(&DriverFact::DontCareStoreIsOneOfMetals)
}

/// A pipeline of the fixture's by its colors, its depth and stencil formats and its samples
/// ([`Lane::made`]'s arguments).
type PipelineShape<'a> = (
    &'a [PixelFormat],
    (Option<PixelFormat>, Option<PixelFormat>),
    u64,
);

/// The kinds the steps of `lane` announce besides `dont-care-store`: the makings of the
/// fixture's pipelines of `pipelines` (colors, depth and stencil, samples), the resolves of 4x
/// color attachments of `resolves` (`degrade_notices::resolve_kind`), and where `buffers`, the
/// `Shared` buffers of the draws and the reads.
fn made(
    lane: &Lane,
    pipelines: &[PipelineShape<'_>],
    resolves: &[PixelFormat],
    buffers: bool,
) -> BTreeSet<Degradation> {
    let system_memory = buffers && shared_in_system_memory(&lane.device);
    let mut first = forced(&[(Degradation::SharedInSystemMemory, system_memory)]);
    for &(colors, attachments, samples) in pipelines {
        first.extend(lane.made(colors, attachments, samples));
    }
    if !resolves.is_empty() {
        first.extend(resolve_kind(&lane.device, MSAA.samples as u32));
    }
    first
}

/// `texels` are `want`, the earlier texels, on a device with the fact. On any other each texel
/// (of `unit` bytes) is one of Metal's two answers, the earlier texel or `stored`, the texel the
/// same pass stores with `Store` (Metal's under its tiler's overflow), as `dont-care-store`'s
/// exactness (`OneOfMetals`) states; the counts of each are printed. So too on a device the runtime
/// does not hand `DONT_CARE` ([`discards`]), which stores what the pass renders: not always the
/// texel a `Store` pass keeps, as a `DontCare` pass's clear reaches no farther than its render area
/// (lavapipe kept the earlier texels past a smaller one, which Metal keeps there too).
fn kept(lane: &Lane, case: &str, texels: &[u8], (want, stored): (&[u8], &[u8]), unit: usize) {
    if metals(lane) {
        assert!(texels == want, "{case}: the earlier texels are not kept");
        return;
    }
    let (mut earlier, mut rendered, mut other) = (0, 0, Vec::new());
    let units = texels
        .chunks(unit)
        .zip(want.chunks(unit))
        .zip(stored.chunks(unit));
    for (at, ((texel, want), stored)) in units.enumerate() {
        if texel == want {
            earlier += 1;
        } else if texel == stored {
            rendered += 1;
        } else if other.len() < 4 {
            other.push(format!(
                "{at}: {texel:02x?} (earlier {want:02x?}, stored {stored:02x?})"
            ));
        } else {
            other.push(String::new());
        }
    }
    let how = match discards(lane) {
        true => "DONT_CARE",
        false => "STORE",
    };
    eprintln!(
        "DONT-CARE-TEXELS {case} on {} ({how}): {earlier} earlier, {rendered} stored, {} other",
        lane.device.name(),
        other.len()
    );
    assert!(
        other.is_empty(),
        "{case}: texels neither the earlier ones nor the pass's stored ones, none of Metal's \
         answers: {other:?}"
    );
}

/// The notices `messages` holds announce `dont-care-store` where `announced`, and the kinds of
/// `first` the device had not announced before (`degrade_notices::assert_made_once`).
fn announces(messages: &Messages, case: &str, announced: bool, first: &BTreeSet<Degradation>) {
    let kinds = match announced {
        true => vec![Degradation::DontCareStore],
        false => Vec::new(),
    };
    let counts = assert_made_once(messages, &kinds, first);
    assert_eq!(
        counts.contains_key(&Degradation::DontCareStore),
        announced,
        "{case}: {counts:?}"
    );
}

/// Passes kept apart from one by the backing (`WorkCounts::kept_texels_passes`) since the device
/// was made.
fn kept_texels(lane: &Lane) -> u64 {
    lane.device.work_counts().kept_texels_passes
}

/// [`lane`] on a device whose colors stored `DontCare` keep their earlier texels
/// (`DontCareColorPolicy::Kept`, which `DevicePolicy::exact_answers()` selects).
fn kept_lane() -> (Messages, super::ScratchCache, Lane) {
    lane_with(|options| options.policy.dont_care_colors = DontCareColorPolicy::Kept)
}

/// [`lane`] on a device whose colors stored `DontCare` are stored as the driver stores them
/// (`DontCareColorPolicy::Driver`, the runtime's default), over the lane's `exact_answers()`.
fn driver_lane() -> (Messages, super::ScratchCache, Lane) {
    lane_with(|options| options.policy.dont_care_colors = DontCareColorPolicy::Driver)
}

/// Passes recorded as one render pass instance keep their attachments' earlier texels under
/// `DontCare` and `MultisampleResolve` on a driver that stores them as Metal's own (our probes'
/// answers), and announce no `dont-care-store` there: a color attachment cleared and drawn, loaded
/// and drawn, cleared and not drawn, and cleared and drawn over a render area smaller than it; a 4x
/// `RGBA8Unorm` and a 4x `BGRA8Unorm_sRGB` attachment cleared, drawn and resolved, or stored
/// `DontCare`; a `Depth32Float` and a `Depth32Float_Stencil8` attachment cleared and drawn; and a
/// 4x `Depth32Float` attachment cleared, drawn and resolved. Every other driver announces it once,
/// and prints whether it kept them. The resolve textures hold the draw everywhere. Under
/// `DontCareColorPolicy::Driver`, the runtime's default: no pass renders through a backing.
#[test]
fn dont_care_stores_keep_the_texels_where_the_driver_stores_them_as_metal_s() {
    let (messages, _cache, lane) = driver_lane();
    let pipeline = lane.pipeline(&[RGBA8], (None, None), 1);
    let old = OLD.repeat((SIDE * SIDE) as usize);
    for (case, load, draw, area) in [
        ("color_clear_draw", LoadAction::Clear, Draw::Full, None),
        ("color_load_draw", LoadAction::Load, Draw::Full, None),
        ("color_clear_nodraw", LoadAction::Clear, Draw::None, None),
        (
            "color_clear_draw_small_area",
            LoadAction::Clear,
            Draw::Full,
            Some(AREA),
        ),
    ] {
        let run = |store| {
            let texture = lane.old_color(RGBA8, PLAIN, TARGET);
            let mut descriptor = color_pass(&texture, load, store, None);
            if let Some(area) = area {
                descriptor.render_target_width = area;
                descriptor.render_target_height = area;
            }
            lane.run(&descriptor, draw, Some(&pipeline));
            lane.read(&texture, (0, 0), 4, BlitOption::None)
        };
        let stored = run(StoreAction::Store);
        let texels = run(StoreAction::DontCare);
        kept(&lane, case, &texels, (&old, &stored), 4);
    }
    for pf in [RGBA8, PixelFormat::BGRA8UnormSRGB] {
        let pipeline = lane.pipeline(&[pf], (None, None), 4);
        let reference = resolved(&lane, &earlier_msaa(&lane, pf), pf);
        for store in [StoreAction::MultisampleResolve, StoreAction::DontCare] {
            let case = format!("{pf:?}_ms4_clear_draw_{store:?}");
            let run = |store| {
                let texture = earlier_msaa(&lane, pf);
                let resolve = matches!(
                    store,
                    StoreAction::MultisampleResolve | StoreAction::StoreAndMultisampleResolve
                )
                .then(|| lane.texture(pf, PLAIN, TARGET));
                let descriptor = color_pass(&texture, LoadAction::Clear, store, resolve.as_ref());
                lane.run(&descriptor, Draw::Full, Some(&pipeline));
                (resolved(&lane, &texture, pf), resolve)
            };
            let (stored, _) = run(match store {
                StoreAction::MultisampleResolve => StoreAction::StoreAndMultisampleResolve,
                _ => StoreAction::Store,
            });
            let (texels, resolve) = run(store);
            kept(&lane, &case, &texels, (&reference, &stored), 4);
            if let (Some(resolve), true) = (resolve, pf == RGBA8) {
                assert_eq!(
                    lane.read(&resolve, (0, 0), 4, BlitOption::None),
                    DRAWN.repeat((SIDE * SIDE) as usize),
                    "{case}: the resolve"
                );
            }
        }
    }
    let word = |v: f32| v.to_le_bytes().repeat((SIDE * SIDE) as usize);
    let depth = PixelFormat::Depth32Float;
    let combined = PixelFormat::Depth32FloatStencil8;
    for (case, pf, stencil) in [
        ("depth32_clear_draw", depth, false),
        ("depth32_stencil8_clear_draw", combined, true),
    ] {
        let pipeline = lane.pipeline(&[], (Some(pf), stencil.then_some(pf)), 1);
        let option = match stencil {
            true => BlitOption::DepthFromDepthStencil,
            false => BlitOption::None,
        };
        // The depths, and the stencils where it has them, after the pass stores by `store`.
        let run = |store| {
            let texture = lane.old_depth_stencil(pf, PLAIN, (true, stencil));
            let mut descriptor = RenderPassDescriptor::new();
            descriptor.depth_attachment.texture = Some(texture.clone());
            descriptor.depth_attachment.load_action = LoadAction::Clear;
            descriptor.depth_attachment.clear_depth = 0.5;
            descriptor.depth_attachment.store_action = store;
            if stencil {
                descriptor.stencil_attachment.texture = Some(texture.clone());
                descriptor.stencil_attachment.load_action = LoadAction::Clear;
                descriptor.stencil_attachment.clear_stencil = 0x40;
                descriptor.stencil_attachment.store_action = store;
            }
            lane.run(&descriptor, Draw::Full, Some(&pipeline));
            let depths = lane.read(&texture, (0, 0), 4, option);
            let stencils = stencil
                .then(|| lane.read(&texture, (0, 0), 1, BlitOption::StencilFromDepthStencil));
            (depths, stencils)
        };
        let (stored, stored_stencils) = run(StoreAction::Store);
        let (depths, stencils) = run(StoreAction::DontCare);
        kept(&lane, case, &depths, (&word(0.75), &stored), 4);
        if let (Some(stencils), Some(stored)) = (stencils, stored_stencils) {
            let want = vec![0x22; (SIDE * SIDE) as usize];
            kept(
                &lane,
                &format!("{case}, stencil"),
                &stencils,
                (&want, &stored),
                1,
            );
        }
    }
    {
        let case = "depth32_ms4_clear_draw_resolve";
        let pipeline = lane.pipeline(&[], (Some(depth), None), 4);
        let resolve_into = |texture: &Texture, load, clear: f64, store| {
            let into = lane.texture(depth, PLAIN, TARGET);
            let mut descriptor = RenderPassDescriptor::new();
            let attachment = &mut descriptor.depth_attachment;
            attachment.texture = Some(texture.clone());
            attachment.load_action = load;
            attachment.clear_depth = clear;
            attachment.store_action = store;
            attachment.resolve_texture = Some(into.clone());
            attachment.depth_resolve_filter = MultisampleDepthResolveFilter::Sample0;
            (descriptor, into)
        };
        // The depths a pass that loads the texture resolves, after a pass that cleared, drew and
        // resolved it stored by `store`.
        let run = |store| {
            let texture = lane.old_depth_stencil(depth, MSAA, (true, false));
            let (descriptor, into) = resolve_into(&texture, LoadAction::Clear, 0.5, store);
            lane.run(&descriptor, Draw::Full, Some(&pipeline));
            assert_eq!(
                lane.read(&into, (0, 0), 4, BlitOption::None),
                word(0.25),
                "{case}: the resolve"
            );
            let load = StoreAction::MultisampleResolve;
            let (descriptor, into) = resolve_into(&texture, LoadAction::Load, 0.0, load);
            lane.run(&descriptor, Draw::None, None);
            lane.read(&into, (0, 0), 4, BlitOption::None)
        };
        let stored = run(StoreAction::StoreAndMultisampleResolve);
        let depths = run(StoreAction::MultisampleResolve);
        kept(&lane, case, &depths, (&word(0.75), &stored), 4);
    }
    let depth = Some(depth);
    let combined = Some(combined);
    let pipelines: [(&[PixelFormat], _, u64); 6] = [
        (&[RGBA8], (None, None), 1),
        (&[RGBA8], (None, None), 4),
        (&[PixelFormat::BGRA8UnormSRGB], (None, None), 4),
        (&[], (depth, None), 1),
        (&[], (combined, combined), 1),
        (&[], (depth, None), 4),
    ];
    let mut first = made(
        &lane,
        &pipelines,
        &[RGBA8, PixelFormat::BGRA8UnormSRGB],
        true,
    );
    // `color_clear_draw_small_area` clears over a render area smaller than its attachment.
    first.extend(tie_cleared(&lane.device, false));
    announces(&messages, "the kept passes", !metals(&lane), &first);
    // The default policy stores the colors as the driver stores them: no pass took a backing.
    assert_eq!(kept_texels(&lane), 0, "the passes through a backing");
    quiet(lane, &messages);
}

/// Passes a driver does not store as Metal's own announce `dont-care-store` on every driver, each
/// on a device of its own, their texels printed: a barrier over the render targets between two
/// draws (MoltenVK restarts its Metal pass there, storing the attachment); a barrier over
/// buffers, which splits the pass into two render pass instances, the first storing; a 4x
/// attachment cleared, not drawn and resolved (Metal leaves the clear where it compresses the
/// texture); and a 4x pass at custom sample positions, where the device takes them. Under
/// `DontCareColorPolicy::Driver`, the runtime's default.
#[test]
fn dont_care_stores_a_driver_may_store_otherwise_are_announced() {
    let old = OLD.repeat((SIDE * SIDE) as usize);
    for (case, scope) in [
        ("barrier_render_targets", BarrierScope::RENDER_TARGETS),
        ("barrier_buffers", BarrierScope::BUFFERS),
    ] {
        let (messages, _cache, lane) = driver_lane();
        let pipeline = lane.pipeline(&[RGBA8], (None, None), 1);
        let texture = lane.old_color(RGBA8, PLAIN, TARGET);
        let descriptor = color_pass(&texture, LoadAction::Clear, StoreAction::DontCare, None);
        encode(&lane, &descriptor, &pipeline, |encoder| {
            triangle(encoder);
            encoder.memory_barrier_with_scope(
                scope,
                RenderStages::FRAGMENT,
                RenderStages::FRAGMENT,
            );
            triangle(encoder);
        });
        let texels = lane.read(&texture, (0, 0), 4, BlitOption::None);
        eprintln!(
            "{case} on {}: the earlier texels {}",
            lane.device.name(),
            if texels == old {
                "are kept"
            } else {
                "are not kept"
            }
        );
        // A barrier over buffers ends the pass's render pass instance (`render-pass-split`).
        let mut first = made(&lane, &[(&[RGBA8], (None, None), 1)], &[], true);
        first.extend(forced(&[(
            Degradation::RenderPassSplit,
            scope == BarrierScope::BUFFERS,
        )]));
        first.extend(tie_cleared(&lane.device, true));
        announces(&messages, case, true, &first);
        assert_eq!(
            kept_texels(&lane),
            0,
            "{case}: the passes through a backing"
        );
        quiet(lane, &messages);
    }
    {
        let case = "ms4_clear_nodraw_resolve";
        let (messages, _cache, lane) = driver_lane();
        let reference = resolved(&lane, &earlier_msaa(&lane, RGBA8), RGBA8);
        let texture = earlier_msaa(&lane, RGBA8);
        let into = lane.texture(RGBA8, PLAIN, TARGET);
        let descriptor = color_pass(
            &texture,
            LoadAction::Clear,
            StoreAction::MultisampleResolve,
            Some(&into),
        );
        lane.run(&descriptor, Draw::None, None);
        let texels = resolved(&lane, &texture, RGBA8);
        eprintln!(
            "{case} on {}: the earlier texels {}",
            lane.device.name(),
            if texels == reference {
                "are kept"
            } else {
                "are not kept"
            }
        );
        let mut first = made(&lane, &[], &[RGBA8], true);
        first.extend(tie_cleared(&lane.device, true));
        announces(&messages, case, true, &first);
        quiet(lane, &messages);
    }
    {
        let case = "ms4_positions_clear_draw_resolve";
        let (messages, _cache, lane) = driver_lane();
        let pipeline = lane.pipeline(&[RGBA8], (None, None), 4);
        let reference = resolved(&lane, &earlier_msaa(&lane, RGBA8), RGBA8);
        let texture = earlier_msaa(&lane, RGBA8);
        let into = lane.texture(RGBA8, PLAIN, TARGET);
        let mut descriptor = color_pass(
            &texture,
            LoadAction::Clear,
            StoreAction::MultisampleResolve,
            Some(&into),
        );
        descriptor.sample_positions = vec![SamplePosition::new(0.5, 0.5); 4];
        // As `sample_positions`' lane: a device without the extension at 4 samples refuses them.
        let d = &lane.device.properties().description;
        let located = d.extensions.contains(&VkExtension::ExtSampleLocations)
            && d.limits
                .sample_location_sample_counts
                .contains(SampleCountFlags::from_raw(4));
        if located {
            lane.run(&descriptor, Draw::Full, Some(&pipeline));
            let texels = resolved(&lane, &texture, RGBA8);
            eprintln!(
                "{case} on {}: the earlier texels {}",
                lane.device.name(),
                if texels == reference {
                    "are kept"
                } else {
                    "are not kept"
                }
            );
            let mut first = made(&lane, &[(&[RGBA8], (None, None), 4)], &[RGBA8], true);
            first.extend(tie_cleared(&lane.device, true));
            announces(&messages, case, true, &first);
        } else {
            eprintln!("{case} on {}: no sample locations", lane.device.name());
        }
        quiet(lane, &messages);
    }
}

/// Under `DontCareColorPolicy::Kept` (`DevicePolicy::exact_answers()`), a color attachment of a
/// texture's own image stored `DontCare` keeps its earlier texels exactly on every driver
/// (Metal's answer, `kept_bytes.rs`), and announces no `dont-care-store`: cleared and drawn,
/// loaded and drawn, cleared and not drawn, cleared and drawn over a render area smaller than it,
/// and cleared and drawn twice with a barrier over the render targets or over buffers between the
/// draws. Each pass renders through a backing its queue keeps, counted, but where the driver
/// stores `DONT_CARE` as Metal's own and the pass is one render pass instance it runs unbroken
/// (MoltenVK, but for the barriers), handed to Metal. The controls stored `Store` keep the draw
/// where it covers the texel. On a device of its own, a 4x `RGBA8Unorm` and a 4x
/// `BGRA8Unorm_sRGB` attachment cleared, drawn and stored `DontCare` keep theirs too; the
/// resolves that read them back announce `dont-care-store` where the driver does not store as
/// Metal's own.
#[test]
fn exact_answers_keep_a_dont_care_color_s_earlier_texels_on_every_driver() {
    let (messages, _cache, lane) = kept_lane();
    let pipeline = lane.pipeline(&[RGBA8], (None, None), 1);
    let old = OLD.repeat((SIDE * SIDE) as usize);
    // Whether the device renders a pass of one unbroken instance through the backing.
    let backed = u64::from(!metals(&lane));
    for (case, load, draw, area) in [
        ("color_clear_draw", LoadAction::Clear, Draw::Full, None),
        ("color_load_draw", LoadAction::Load, Draw::Full, None),
        ("color_clear_nodraw", LoadAction::Clear, Draw::None, None),
        (
            "color_clear_draw_small_area",
            LoadAction::Clear,
            Draw::Full,
            Some(AREA),
        ),
    ] {
        let run = |store| {
            let texture = lane.old_color(RGBA8, PLAIN, TARGET);
            let mut descriptor = color_pass(&texture, load, store, None);
            if let Some(area) = area {
                descriptor.render_target_width = area;
                descriptor.render_target_height = area;
            }
            lane.run(&descriptor, draw, Some(&pipeline));
            lane.read(&texture, (0, 0), 4, BlitOption::None)
        };
        let stored = run(StoreAction::Store);
        if draw == Draw::Full && area.is_none() {
            assert_eq!(
                stored,
                DRAWN.repeat((SIDE * SIDE) as usize),
                "{case}: the control"
            );
        }
        let before = kept_texels(&lane);
        let texels = run(StoreAction::DontCare);
        assert!(texels == old, "{case}: the earlier texels are not kept");
        assert_eq!(
            kept_texels(&lane) - before,
            backed,
            "{case}: the backed passes"
        );
        eprintln!(
            "KEPT-TEXELS {case} on {}: kept, {} pass(es) through the backing",
            lane.device.name(),
            kept_texels(&lane) - before
        );
    }
    for (case, scope) in [
        ("barrier_render_targets", BarrierScope::RENDER_TARGETS),
        ("barrier_buffers", BarrierScope::BUFFERS),
    ] {
        let texture = lane.old_color(RGBA8, PLAIN, TARGET);
        let descriptor = color_pass(&texture, LoadAction::Clear, StoreAction::DontCare, None);
        let before = kept_texels(&lane);
        encode(&lane, &descriptor, &pipeline, |encoder| {
            triangle(encoder);
            encoder.memory_barrier_with_scope(
                scope,
                RenderStages::FRAGMENT,
                RenderStages::FRAGMENT,
            );
            triangle(encoder);
        });
        let texels = lane.read(&texture, (0, 0), 4, BlitOption::None);
        assert!(texels == old, "{case}: the earlier texels are not kept");
        // No driver runs these passes unbroken: each renders through the backing.
        assert_eq!(kept_texels(&lane) - before, 1, "{case}: the backed passes");
        eprintln!("KEPT-TEXELS {case} on {}: kept", lane.device.name());
    }
    // A barrier over buffers ends the pass's render pass instance (`render-pass-split`), and
    // `color_clear_draw_small_area` clears over a render area smaller than its attachment.
    let mut first = made(&lane, &[(&[RGBA8], (None, None), 1)], &[], true);
    first.insert(Degradation::RenderPassSplit);
    first.extend(tie_cleared(&lane.device, false));
    announces(&messages, "the kept colors", false, &first);
    quiet(lane, &messages);
    // The 4x colors, on a device of their own (the notice is once per device).
    let (messages, _cache, lane) = kept_lane();
    for pf in [RGBA8, PixelFormat::BGRA8UnormSRGB] {
        let pipeline = lane.pipeline(&[pf], (None, None), 4);
        let reference = resolved(&lane, &earlier_msaa(&lane, pf), pf);
        let case = format!("{pf:?}_ms4_clear_draw_DontCare");
        let texture = earlier_msaa(&lane, pf);
        let descriptor = color_pass(&texture, LoadAction::Clear, StoreAction::DontCare, None);
        let before = kept_texels(&lane);
        lane.run(&descriptor, Draw::Full, Some(&pipeline));
        let backed = kept_texels(&lane) - before;
        let texels = resolved(&lane, &texture, pf);
        assert!(
            texels == reference,
            "{case}: the earlier texels are not kept"
        );
        assert_eq!(
            backed,
            u64::from(!metals(&lane)),
            "{case}: the backed passes"
        );
        eprintln!(
            "KEPT-TEXELS {case} on {}: kept, {backed} pass(es) through the backing",
            lane.device.name()
        );
    }
    let pipelines: [(&[PixelFormat], _, u64); 2] = [
        (&[RGBA8], (None, None), 4),
        (&[PixelFormat::BGRA8UnormSRGB], (None, None), 4),
    ];
    let mut first = made(
        &lane,
        &pipelines,
        &[RGBA8, PixelFormat::BGRA8UnormSRGB],
        true,
    );
    // The 4x colors' clears cover their attachments.
    first.extend(tie_cleared(&lane.device, true));
    announces(&messages, "the kept 4x colors", !metals(&lane), &first);
    quiet(lane, &messages);
}

/// A pass of no depth attachment whose draw's state tests and writes depth renders into a hidden
/// depth and stencil of the runtime's own (`DepthStencilTarget::hidden`), cleared before the pass
/// and never stored, and of no texture of the program's: it announces no `dont-care-store`, on
/// every driver.
#[test]
fn a_hidden_depth_announces_no_dont_care_store() {
    let (messages, _cache, lane) = lane();
    let pipeline = lane.pipeline(&[RGBA8], (None, None), 1);
    let texture = lane.old_color(RGBA8, PLAIN, TARGET);
    let descriptor = color_pass(&texture, LoadAction::Clear, StoreAction::Store, None);
    let mut state = DepthStencilDescriptor::default();
    state.depth_compare_function = CompareFunction::Less;
    state.depth_write_enabled = true;
    let state = lane.device.new_depth_stencil_state(&state).unwrap();
    encode(&lane, &descriptor, &pipeline, |encoder| {
        encoder.set_depth_stencil_state(&state).unwrap();
        triangle(encoder);
    });
    assert_eq!(
        lane.read(&texture, (0, 0), 4, BlitOption::None),
        DRAWN.repeat((SIDE * SIDE) as usize),
        "hidden_depth: the draw passes the hidden depth of 1.0"
    );
    let mut first = made(&lane, &[(&[RGBA8], (None, None), 1)], &[], true);
    first.extend(tie_cleared(&lane.device, true));
    announces(&messages, "hidden_depth", false, &first);
    quiet(lane, &messages);
}

/// A 4x attachment cleared and resolved alone (`MultisampleResolve`) into its sRGB twin, in a
/// pass with no draw call, after a pass that cleared and stored it: Metal keeps the samples'
/// earlier texels, the texture compressed or not, and resolves the clear into the twin's channel
/// order (an Apple M4 under macOS 27.0.1; the consumer's M3 Max alike for `BGRA8Unorm` each way).
/// The clear's codes are 0x00 and 0xff, which both encodings' averages hold, so this case does
/// not read which encoding the resolve works in (`resolves.rs`'s
/// `a_resolve_into_the_srgb_twin_resolves_in_the_resolve_texture_s_encoding` does: the resolve
/// texture's). On every driver the samples resolve as
/// the earlier ones do, the resolve texture holds the clear (opaque red), and the pass announces
/// no `dont-care-store`. Each pair on a device of its own.
#[test]
fn a_clear_resolved_into_the_srgb_twin_without_a_draw_keeps_the_samples() {
    let pairs = [
        (
            PixelFormat::BGRA8Unorm,
            PixelFormat::BGRA8UnormSRGB,
            [0, 0, 0xff, 0xff],
        ),
        (
            PixelFormat::BGRA8UnormSRGB,
            PixelFormat::BGRA8Unorm,
            [0, 0, 0xff, 0xff],
        ),
        (
            PixelFormat::RGBA8Unorm,
            PixelFormat::RGBA8UnormSRGB,
            [0xff, 0, 0, 0xff],
        ),
        (
            PixelFormat::RGBA8UnormSRGB,
            PixelFormat::RGBA8Unorm,
            [0xff, 0, 0, 0xff],
        ),
    ];
    for (att, res, red) in pairs {
        let case = format!("ms4_clear_nodraw_resolve {att:?} into {res:?}");
        let (messages, _cache, lane) = lane();
        let texture = earlier_msaa(&lane, att);
        let into = lane.texture(res, PLAIN, TARGET);
        let mut descriptor = color_pass(
            &texture,
            LoadAction::Clear,
            StoreAction::MultisampleResolve,
            Some(&into),
        );
        descriptor.color_attachments[0].clear_color = ClearColor::new(1.0, 0.0, 0.0, 1.0);
        let _ = messages.take_notices();
        lane.run(&descriptor, Draw::None, None);
        // The pass resolves the 4x attachment: no draw, no buffer. The resolve is into the
        // attachment's format's sRGB twin, which the crate-owned resolve draws on every device
        // (`color_resolve.rs`), announced `shader-resolve`.
        let mut first = made(&lane, &[], &[], false);
        first.insert(Degradation::ShaderResolve);
        // Into an sRGB twin it decodes and encodes by the device's own sRGB conversions, every
        // device having the format (`bind::ColorResolveFormat::device_srgb`).
        if matches!(
            res,
            PixelFormat::BGRA8UnormSRGB | PixelFormat::RGBA8UnormSRGB
        ) {
            first.insert(Degradation::DeviceRoundedResolve);
        }
        announces(&messages, &case, false, &first);
        let resolve = lane.read(&into, (0, 0), 4, BlitOption::None);
        let samples = resolved(&lane, &texture, att);
        let reference = resolved(&lane, &earlier_msaa(&lane, att), att);
        eprintln!(
            "{case} on {}: the earlier texels {}, the resolve {:02x?}",
            lane.device.name(),
            if samples == reference {
                "are kept"
            } else {
                "are not kept"
            },
            &resolve[..4]
        );
        assert!(
            samples == reference,
            "{case}: the earlier texels are not kept"
        );
        assert!(
            resolve.chunks(4).all(|texel| texel == red),
            "{case}: the resolve {:02x?}",
            &resolve[..4]
        );
        quiet(lane, &messages);
    }
}
