//! Device tests of the counts of render passes whose first instance did not record inline, by the
//! first check that ruled it out (the runtime's `render/inline_first.rs`,
//! `WorkCounts::passes_not_inline_layouts` and kin), on the device the cfg selects, under the
//! validation layer, through the resolve lane's fixture and helpers (`resolves.rs`): a pass of one
//! sample cleared and stored records inline and counts no check; the same loaded `DontCare` counts
//! its actions; the same into a memoryless texture loaded and stored `DontCare` counts the draw's
//! write of its memoryless color, and cleared, its memoryless color; a pass of four samples
//! resolved counts its layout. A pass kept by its color's actions counts the first action ruled
//! out too: a `DontCare` load, a reach past the render area its colors do not share, a clear to a
//! NaN, a resolve, a store.
//!
//! And a memoryless color loaded and stored `DontCare` beside a stored one: a pass whose draw
//! writes only the stored color records inline, and one whose second draw writes the memoryless
//! color splits before it (`WorkCounts::inline_memoryless_write_splits`) into an instance that
//! records inline too (`WorkCounts::inline_split_instances`), each storing what plain Metal
//! stores.
//!
//! And a pass beside a depth/stencil attachment: one of the texture's own image cleared or loaded
//! and stored records inline, beside a color or alone, and so does one of a combined texture whose
//! depth alone it names; a depth loaded or stored `DontCare` counts its actions, a memoryless one
//! its memoryless, a depth and a stencil of separate textures its attachments. A pass recorded
//! inline beside its depth tests and writes it as Metal does.

use super::memoryless::memoryless;
use super::resolves::{color_pass, lane, quiet, Lane, Shape, FULL, ONE, SAMPLES, SIZE, WHITE};
use metal2vulkan::metal::{ColorWriteMask, PixelFormat, TextureType, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_runtime::{
    BlitOption, ClearColor, LoadAction, PipelineOptions, RenderPassDescriptor, StoreAction,
    WorkCounts,
};

const RGBA8: PixelFormat = PixelFormat::RGBA8Unorm;

/// The counts of passes whose first instance recorded inline and of those that did not, by
/// check: inline, encoders, layouts, memoryless, actions, buffer textures, attachments, storage,
/// stand-ins, memoryless writes, draws.
fn inline_counts(counts: &WorkCounts) -> [u64; 11] {
    [
        counts.inline_passes,
        counts.passes_not_inline_encoders,
        counts.passes_not_inline_layouts,
        counts.passes_not_inline_memoryless,
        counts.passes_not_inline_actions,
        counts.passes_not_inline_buffer_textures,
        counts.passes_not_inline_attachments,
        counts.passes_not_inline_storage,
        counts.passes_not_inline_stand_ins,
        counts.passes_not_inline_memoryless_writes,
        counts.passes_not_inline_draws,
    ]
}

/// Draw white over `descriptor`'s pass, with a pipeline of `samples`: what that pass added to
/// [`inline_counts`].
fn drawn(lane: &Lane, descriptor: &RenderPassDescriptor, samples: u64) -> [u64; 11] {
    let pipeline = lane.pipeline((FULL, WHITE), &[RGBA8], (None, None), samples);
    let before = inline_counts(&lane.device.work_counts());
    lane.run(descriptor, |l, e, h| l.draw(e, h, &pipeline, 0.5))
        .unwrap();
    let after = inline_counts(&lane.device.work_counts());
    std::array::from_fn(|at| after[at] - before[at])
}

#[test]
fn a_pass_that_does_not_record_inline_counts_the_first_check_that_ruled_it_out() {
    let (messages, _cache, lane) = lane();
    let target = lane.single(RGBA8);
    let msaa = lane.msaa(RGBA8);
    let stored = color_pass(&target, None, StoreAction::Store, (0, 0));
    assert_eq!(
        drawn(&lane, &stored, 1),
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "cleared and stored"
    );
    let mut undefined = stored.clone();
    undefined.color_attachments[0].load_action = LoadAction::DontCare;
    assert_eq!(
        drawn(&lane, &undefined, 1),
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        "loaded DontCare"
    );
    let scratch = memoryless(
        &lane,
        TextureType::Type2D,
        RGBA8,
        ONE,
        TextureUsage::RENDER_TARGET,
    )
    .unwrap();
    let mut transient = color_pass(&scratch, None, StoreAction::DontCare, (0, 0));
    transient.color_attachments[0].load_action = LoadAction::DontCare;
    assert_eq!(
        drawn(&lane, &transient, 1),
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        "memoryless, loaded and stored DontCare, drawn"
    );
    let mut cleared = transient.clone();
    cleared.color_attachments[0].load_action = LoadAction::Clear;
    assert_eq!(
        drawn(&lane, &cleared, 1),
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        "memoryless, cleared"
    );
    let resolved = color_pass(
        &msaa,
        Some(&target),
        StoreAction::MultisampleResolve,
        (0, 0),
    );
    assert_eq!(
        drawn(&lane, &resolved, 4),
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        "four samples resolved"
    );
    drop((target, msaa, scratch));
    quiet(lane, &messages);
}

/// The counts of passes kept from recording inline by a color's actions, then by the first
/// action ruled out: a `DontCare` load, a reach past the render area not shared, a resolve, a
/// store.
fn actions_counts(counts: &WorkCounts) -> [u64; 5] {
    [
        counts.passes_not_inline_actions,
        counts.passes_not_inline_actions_dont_care_load,
        counts.passes_not_inline_actions_unequal_reach,
        counts.passes_not_inline_actions_resolve,
        counts.passes_not_inline_actions_store,
    ]
}

/// The actions of a pass's color attachments that keep its first instance from recording
/// inline, each pass counted once by the first ruled out (`WorkCounts::passes_not_inline_actions`
/// and its `_dont_care_load` kin), white drawn over each: loaded `DontCare`; cleared over a render
/// area of half the texture's side, which Metal's clear of whole 32-pixel tiles reaches past,
/// beside a second color loaded, whose texels past the render area Metal keeps (where it is
/// alone, the pass records inline over the reach), and a larger texture cleared past a memoryless
/// color's side, which sets the render area; resolved, a single-sampled attachment's texels
/// copied into the resolve texture; and stored `DontCare`. An `RGBA16Float` texture cleared to a
/// NaN, which the device's clear converts, records inline.
#[test]
fn a_pass_kept_by_a_color_s_actions_counts_the_first_action_ruled_out() {
    let (messages, _cache, lane) = lane();
    let rgba16f = PixelFormat::RGBA16Float;
    let ran = |descriptor: &RenderPassDescriptor, pf: PixelFormat| -> [u64; 5] {
        let colors = descriptor
            .color_attachments
            .iter()
            .filter(|c| c.texture.is_some())
            .count();
        let pipeline = lane.pipeline((FULL, WHITE), &vec![pf; colors], (None, None), 1);
        let before = actions_counts(&lane.device.work_counts());
        lane.run(descriptor, |l, e, h| l.draw(e, h, &pipeline, 0.5))
            .unwrap();
        let after = actions_counts(&lane.device.work_counts());
        std::array::from_fn(|at| after[at] - before[at])
    };
    let target = lane.single(RGBA8);
    let stored = color_pass(&target, None, StoreAction::Store, (0, 0));
    let mut undefined = stored.clone();
    undefined.color_attachments[0].load_action = LoadAction::DontCare;
    assert_eq!(ran(&undefined, RGBA8), [1, 1, 0, 0, 0], "loaded DontCare");
    let mut part = stored.clone();
    (part.render_target_width, part.render_target_height) = (SIZE / 2, SIZE / 2);
    assert_eq!(
        ran(&part, RGBA8),
        [0, 0, 0, 0, 0],
        "cleared past half its side, alone"
    );
    let beside = lane.single(RGBA8);
    let loaded = &mut part.color_attachments[1];
    loaded.texture = Some(beside.clone());
    loaded.load_action = LoadAction::Load;
    loaded.store_action = StoreAction::Store;
    assert_eq!(
        ran(&part, RGBA8),
        [1, 0, 1, 0, 0],
        "cleared past half its side, beside a color loaded"
    );
    // A 64x64 texture cleared over the 4x4 render area of a memoryless color beside it, loaded and
    // stored `DontCare`: the clear reaches 32x32, past the memoryless texture.
    let scratch = memoryless(
        &lane,
        TextureType::Type2D,
        RGBA8,
        ONE,
        TextureUsage::RENDER_TARGET,
    )
    .unwrap();
    let wide = lane.texture(
        RGBA8,
        Shape { size: 64, ..ONE },
        TextureUsage::RENDER_TARGET,
    );
    let mut past = color_pass(&scratch, None, StoreAction::DontCare, (0, 0));
    past.color_attachments[0].load_action = LoadAction::DontCare;
    let cleared = &mut past.color_attachments[1];
    cleared.texture = Some(wide.clone());
    cleared.load_action = LoadAction::Clear;
    cleared.store_action = StoreAction::Store;
    assert_eq!(
        ran(&past, RGBA8),
        [1, 0, 1, 0, 0],
        "cleared past a memoryless color's side"
    );
    let half = lane.single(rgba16f);
    let mut nan = color_pass(&half, None, StoreAction::Store, (0, 0));
    nan.color_attachments[0].clear_color = ClearColor::new(f64::NAN, 0.0, 0.0, 1.0);
    assert_eq!(ran(&nan, rgba16f), [0, 0, 0, 0, 0], "cleared to a NaN");
    let resolve = lane.single(RGBA8);
    let resolved = color_pass(
        &target,
        Some(&resolve),
        StoreAction::MultisampleResolve,
        (0, 0),
    );
    assert_eq!(ran(&resolved, RGBA8), [1, 0, 0, 1, 0], "resolved");
    let unstored = color_pass(&target, None, StoreAction::DontCare, (0, 0));
    assert_eq!(ran(&unstored, RGBA8), [1, 0, 0, 0, 1], "stored DontCare");
    drop((target, beside, scratch, wide, half, resolve));
    quiet(lane, &messages);
}

/// A memoryless color loaded and stored `DontCare` beside a stored one cleared to zero (the module
/// docs): white through a pipeline that writes the stored color alone records inline; white
/// through one that leaves the stored color alone and then through one that writes the memoryless
/// color too splits before the second, which records inline too. Each stores the white plain
/// Metal stores, or the clear, and the second pass's memoryless color takes a backing from the
/// split on.
#[test]
fn a_memoryless_color_records_inline_until_a_draw_writes_it() {
    let (messages, _cache, lane) = lane();
    let stored = lane.single(RGBA8);
    let scratch = memoryless(
        &lane,
        TextureType::Type2D,
        RGBA8,
        ONE,
        TextureUsage::RENDER_TARGET,
    )
    .unwrap();
    // Color 0 the memoryless one, which `rt6_resolve_white` writes, and color 1 the stored one.
    let mut pass = color_pass(&scratch, None, StoreAction::DontCare, (0, 0));
    pass.color_attachments[0].load_action = LoadAction::DontCare;
    let other = &mut pass.color_attachments[1];
    other.texture = Some(stored.clone());
    other.load_action = LoadAction::Clear;
    other.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
    other.store_action = StoreAction::Store;
    let white = lane.pipeline((FULL, WHITE), &[RGBA8, RGBA8], (None, None), 1);
    let function = |name| {
        let function = lane.library.new_function_with_name(name).unwrap();
        StageSource::Function(function.function_descriptor())
    };
    let mut unwritten = RenderPipelineDescriptor::new(function(FULL));
    unwritten.fragment = Some(function(WHITE));
    let mut leaves = ColorAttachmentDescriptor::new(RGBA8);
    leaves.write_mask = ColorWriteMask::NONE;
    unwritten.color_attachments[0] = Some(leaves);
    unwritten.color_attachments[1] = Some(ColorAttachmentDescriptor::new(RGBA8));
    let unwritten = lane
        .device
        .new_render_pipeline_state(&unwritten, &PipelineOptions::default())
        .unwrap();
    let texels = (SIZE * SIZE) as usize;
    let cleared = vec![0u8; 4 * texels];
    // The memoryless color unwritten: one instance, inline.
    let splits = |lane: &Lane| lane.device.work_counts().inline_memoryless_write_splits;
    let later = |lane: &Lane| lane.device.work_counts().inline_split_instances;
    let (before, split_before) = (inline_counts(&lane.device.work_counts()), splits(&lane));
    let later_before = later(&lane);
    lane.run(&pass, |l, e, h| l.draw(e, h, &unwritten, 0.5))
        .unwrap();
    let after = inline_counts(&lane.device.work_counts());
    assert_eq!(
        std::array::from_fn::<u64, 11, _>(|at| after[at] - before[at]),
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "beside a memoryless color it leaves alone"
    );
    assert_eq!(splits(&lane), split_before, "no split");
    assert_eq!(later(&lane), later_before, "no instance after a split");
    assert_eq!(
        lane.read(&stored, (0, 0), 4, BlitOption::None),
        cleared,
        "the stored color, cleared"
    );
    // Then a draw that writes it: split before it.
    let before = inline_counts(&lane.device.work_counts());
    lane.run(&pass, |l, e, h| {
        l.draw(e, h, &unwritten, 0.5);
        l.draw(e, h, &white, 0.5);
    })
    .unwrap();
    let after = inline_counts(&lane.device.work_counts());
    assert_eq!(
        std::array::from_fn::<u64, 11, _>(|at| after[at] - before[at]),
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "a later draw writes the memoryless color"
    );
    assert_eq!(splits(&lane), split_before + 1, "a split before it");
    assert_eq!(
        later(&lane),
        later_before + 1,
        "the instance after it, inline"
    );
    assert_eq!(
        lane.read(&stored, (0, 0), 4, BlitOption::None),
        cleared,
        "the stored color, cleared"
    );
    drop((stored, scratch));
    quiet(lane, &messages);
}

/// What `descriptor`'s pass, white drawn over it at 0.5 through a pipeline of its color (where it
/// has one) and of `formats` for its depth and stencil, added to [`inline_counts`]. A pass of no
/// color draws through a pipeline of no fragment function.
fn drawn_beside(
    lane: &Lane,
    descriptor: &RenderPassDescriptor,
    formats: (Option<PixelFormat>, Option<PixelFormat>),
) -> [u64; 11] {
    let pipeline = match descriptor.color_attachments[0].texture.is_some() {
        true => lane.pipeline((FULL, WHITE), &[RGBA8], formats, 1),
        false => {
            let function = lane.library.new_function_with_name(FULL).unwrap();
            let mut d = RenderPipelineDescriptor::new(StageSource::Function(
                function.function_descriptor(),
            ));
            (d.depth_format, d.stencil_format) = formats;
            lane.device
                .new_render_pipeline_state(&d, &PipelineOptions::default())
                .unwrap()
        }
    };
    let before = inline_counts(&lane.device.work_counts());
    lane.run(descriptor, |l, e, h| l.draw(e, h, &pipeline, 0.5))
        .unwrap();
    let after = inline_counts(&lane.device.work_counts());
    std::array::from_fn(|at| after[at] - before[at])
}

/// A pass beside a depth/stencil attachment records its first instance inline where nothing the
/// pass's end decides changes that attachment there (the runtime's `render/inline_first.rs`): a
/// single-sampled `Depth32Float` cleared or loaded and stored, beside a color or alone, and a
/// `Depth32Float_Stencil8` whose depth alone the pass names, its stencil kept; and where it does
/// not, the first check that ruled it out is counted: a depth loaded `DontCare` (its actions), one
/// stored `DontCare` (its actions), a memoryless one (memoryless), a depth and a stencil of
/// separate textures, rendered through a scratch image (its attachments). Under the validation
/// layer, which reports nothing.
#[test]
fn a_pass_beside_a_depth_or_stencil_records_inline_where_its_attachment_allows() {
    let (messages, _cache, lane) = lane();
    let (depth32, combined) = (PixelFormat::Depth32Float, PixelFormat::Depth32FloatStencil8);
    let color = lane.single(RGBA8);
    let depth = lane.single(depth32);
    let both = lane.single(combined);
    let stencil = lane.single(PixelFormat::Stencil8);
    let beside = |texture: &metal2vulkan_runtime::Texture, load, store| {
        let mut descriptor = color_pass(&color, None, StoreAction::Store, (0, 0));
        let attachment = &mut descriptor.depth_attachment;
        attachment.texture = Some(texture.clone());
        attachment.load_action = load;
        attachment.store_action = store;
        descriptor
    };
    let with_depth = (Some(depth32), None);
    let cleared = beside(&depth, LoadAction::Clear, StoreAction::Store);
    let inline = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    assert_eq!(
        drawn_beside(&lane, &cleared, with_depth),
        inline,
        "a depth cleared and stored"
    );
    let loaded = beside(&depth, LoadAction::Load, StoreAction::Store);
    assert_eq!(
        drawn_beside(&lane, &loaded, with_depth),
        inline,
        "a depth loaded and stored"
    );
    let mut alone = cleared.clone();
    alone.color_attachments[0].texture = None;
    assert_eq!(
        drawn_beside(&lane, &alone, with_depth),
        inline,
        "a depth cleared and stored, no color"
    );
    let named_depth = beside(&both, LoadAction::Clear, StoreAction::Store);
    assert_eq!(
        drawn_beside(&lane, &named_depth, (Some(combined), None)),
        inline,
        "the depth of a depth and stencil texture, its stencil kept"
    );
    let actions = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0];
    let undefined = beside(&depth, LoadAction::DontCare, StoreAction::Store);
    let before = actions_counts(&lane.device.work_counts());
    assert_eq!(
        drawn_beside(&lane, &undefined, with_depth),
        actions,
        "a depth loaded DontCare"
    );
    let discarded = beside(&depth, LoadAction::Clear, StoreAction::DontCare);
    assert_eq!(
        drawn_beside(&lane, &discarded, with_depth),
        actions,
        "a depth stored DontCare"
    );
    let after = actions_counts(&lane.device.work_counts());
    assert_eq!(
        std::array::from_fn::<u64, 5, _>(|at| after[at] - before[at]),
        [2, 1, 0, 0, 1],
        "the depth's actions ruled out: a DontCare load, a store"
    );
    let scratch = memoryless(
        &lane,
        TextureType::Type2D,
        depth32,
        ONE,
        TextureUsage::RENDER_TARGET,
    )
    .unwrap();
    let transient = beside(&scratch, LoadAction::Clear, StoreAction::DontCare);
    assert_eq!(
        drawn_beside(&lane, &transient, with_depth),
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        "a memoryless depth"
    );
    let mut apart = cleared.clone();
    let attachment = &mut apart.stencil_attachment;
    attachment.texture = Some(stencil.clone());
    attachment.load_action = LoadAction::Clear;
    attachment.store_action = StoreAction::Store;
    assert_eq!(
        drawn_beside(&lane, &apart, (Some(depth32), Some(PixelFormat::Stencil8))),
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        "a depth and a stencil of separate textures"
    );
    drop((color, depth, both, stencil, scratch));
    quiet(lane, &messages);
}

/// A pass whose first instance records inline beside its depth tests and writes the depth as
/// Metal does: over a color cleared to zero and a `Depth32Float` cleared to 0.5, under `Less`
/// with writes, white at 0.75 draws nothing, the sample color at 0.25 draws everywhere (red 0,
/// green 0, blue a half, alpha 1, one sample), and white at 0.4 nothing; the depth is left at
/// 0.25 and the color at the sample color's.
#[test]
fn a_pass_recorded_inline_beside_its_depth_tests_and_writes_it() {
    let (messages, _cache, lane) = lane();
    let depth32 = PixelFormat::Depth32Float;
    let color = lane.single(RGBA8);
    let depth = lane.single(depth32);
    let mut descriptor = color_pass(&color, None, StoreAction::Store, (0, 0));
    let attachment = &mut descriptor.depth_attachment;
    attachment.texture = Some(depth.clone());
    attachment.load_action = LoadAction::Clear;
    attachment.store_action = StoreAction::Store;
    attachment.clear_depth = 0.5;
    let white = lane.pipeline((FULL, WHITE), &[RGBA8], (Some(depth32), None), 1);
    let colored = lane.pipeline((FULL, SAMPLES), &[RGBA8], (Some(depth32), None), 1);
    let mut less = metal2vulkan::metal::state::DepthStencilDescriptor::default();
    less.depth_compare_function = metal2vulkan::metal::CompareFunction::Less;
    less.depth_write_enabled = true;
    let less = lane.device.new_depth_stencil_state(&less).unwrap();
    let before = lane.device.work_counts().inline_passes;
    lane.run(&descriptor, |l, e, h| {
        e.set_depth_stencil_state(&less).unwrap();
        l.draw(e, h, &white, 0.75);
        l.draw(e, h, &colored, 0.25);
        l.draw(e, h, &white, 0.4);
    })
    .unwrap();
    assert_eq!(
        lane.device.work_counts().inline_passes - before,
        1,
        "the pass recorded inline"
    );
    let texels = lane.read(&color, (0, 0), 4, BlitOption::None);
    for texel in texels.chunks(4) {
        assert!(
            texel[0] == 0 && texel[1] == 0 && (127..=128).contains(&texel[2]) && texel[3] == 255,
            "the color {texel:?}, not the sample color's"
        );
    }
    let depths = lane.read(&depth, (0, 0), 4, BlitOption::None);
    assert_eq!(
        depths,
        0.25f32.to_le_bytes().repeat((SIZE * SIZE) as usize),
        "the depth, not 0.25"
    );
    drop((color, depth));
    quiet(lane, &messages);
}
