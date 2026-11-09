//! A render pipeline's color drawn into a pass attachment of another pixel format on the device the
//! cfg selects, under the validation layer. Metal's validation stops each; plain Metal (Metal
//! probes, an M4 with macOS 27.0.1) draws them:
//!
//! - **A channel-order twin** (`RGBA8Unorm` and `BGRA8Unorm`, their sRGB forms, `RGB10A2Unorm`
//!   and `BGR10A2Unorm`, each way) stores, blends, masks and resolves what a pipeline of the
//!   pass's own format does there: the color in the pass's channel order (probe 1
//!   `a_rgba8_in_bgra8` stored `bf8040ff`, a `BGRA8Unorm` pipeline's bytes; `b_rgba8_in_bgra8`
//!   blended, `m_rgba8_in_bgra8_ra` masked to red and alpha, and `s_rgba8_in_bgra8_blend` at 4
//!   samples resolved, each as the `BGRA8Unorm` pipeline; probe 2 `p_rgb10a2_in_bgr10a2` and the
//!   reverse, each word the pass format's own). Each case clears a 4² pass to (0.5, 0.25, 0.75, 1)
//!   and draws its left half in (0.2, 0.4, 0.6, 0.8), and its bytes are the same draw's by a
//!   pipeline of the pass's format; the 8-bit unorm pairs' bytes are Metal's too.
//! - **A color at another place in the per-sample record**: plain Metal packs the state's set
//!   colors from byte 0 and stores each place into the pass's attachment there (probes 2 to 5). A
//!   state of colors 0 and 1 in a pass of color 1 alone stores its color 0 into the pass's color 1,
//!   blended as the state's color 0, at 1 and 4 samples (probe 3 `g_rgba8_rgba8_in_none_rgba8` and
//!   `_blend0`): the bytes the state's color 0 stores into a pass of color 0. A state of color 1
//!   alone (`function_constant_interface`'s `fc_fragment_gated_color`, its second output red) in a
//!   pass of colors 0 and 1 stores red into the pass's color 0 and zero into its color 1 where it
//!   draws (probe 3 `g_none_rgba8_in_rgba8_rgba8`).
//! - **A pass color at no place of the record** is zeroed where the draw covers it only where
//!   Metal's compiled code reads no attachment (probes 4 and 5): kept under a partial write mask
//!   and a blend whose destination factor is `One`, each exact; kept under a blend over the
//!   output's alpha, read from a buffer as Metal keeps it (`buffer1`), with the
//!   `uncovered-color-kept` notice ([`KEPT_COLORS`]), as Metal's compiler zeroes it where it
//!   folds the output to a constant.
//! - **Another encoding through a view**: a state color whose encoding a view of the attachment's
//!   image stores, where the image takes that view (the sRGB and plain 8-bit forms, which every
//!   such image lists, and a texture made with `PixelFormatView`), stores what a pass of the view's
//!   format stores: an `RGBA8Unorm` state in a `BGRA8Unorm_sRGB` pass the plain encoding (probe 1
//!   `a_rgba8_in_bgra8s`), an `RGBA8Unorm` state in an `R32Float` pass its bytes (a probe).
//! - **Another encoding through a stand-in**: where the image takes no such view (an `R32Float`
//!   texture without `PixelFormatView` under an `RGBA8Unorm` state), a stand-in image of the
//!   view's format, copied in before the instance and back after it, stores the same bytes, and
//!   draws of the pass's own format before and after it read and leave what the view's do.
//! - **Refused**, `AttachmentFormat`: a pair no view stores (a signed normalized pass, which
//!   stores -128 as -127, probe 1), another encoding in a multisampled pass (unmeasured past the
//!   channel-order twins), and a color whose bytes plain Metal stores into part of an attachment,
//!   or more than one: an `R8Unorm` color 0 beside an `RGBA8Unorm` color 1 in a pass of color 1
//!   alone (probe 3 `g_r8_rgba8_in_none_rgba8` stored both colors' bytes into it).

use super::degrade_notices::shared_in_system_memory;
use super::degrade_notices::{
    assert_made_once, clear_in_question, forced, made_render_kinds, resolve_kind,
};
use super::resolves::{lane, quiet, Lane, MSAA, ONE};
use crate::air_library::AirLibrary;
use metal2vulkan::library::{FunctionConstantValue, FunctionConstantValues};
use metal2vulkan::metal::{
    BlendFactor, ColorWriteMask, DataType, PixelFormat, PrimitiveType, TextureUsage,
};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    Attachment, BlitOption, Buffer, ClearColor, Degradation, DrawProblem, Error, Invalid, Library,
    LoadAction, PipelineOptions, Problem, RenderPassDescriptor, RenderPipelineState, StoreAction,
    Texture,
};
use std::collections::BTreeSet;

/// The kind [`an_uncovered_color_is_kept_where_metal_s_code_reads_the_attachment`] takes.
pub(super) const KEPT_COLORS: [Degradation; 1] = [Degradation::UncoveredColorKept];

/// The probe's clear and draw colors, and the rectangle it draws: the target's left half.
const CLEAR: [f64; 4] = [0.5, 0.25, 0.75, 1.0];
const COLOR: [f32; 4] = [0.2, 0.4, 0.6, 0.8];
const LEFT: [f32; 4] = [-1.0, -1.0, 0.0, 1.0];

/// How a pipeline writes its color: its blend, and its write mask.
#[derive(Clone, Copy)]
struct Writes {
    blend: Blend,
    mask: ColorWriteMask,
}

/// A pipeline color's blend: none; source over destination by the output's alpha; or source
/// plus destination.
#[derive(Clone, Copy, PartialEq)]
enum Blend {
    Off,
    Over,
    Add,
}

const PLAIN: Writes = Writes {
    blend: Blend::Off,
    mask: ColorWriteMask::ALL,
};

const OVER: Writes = Writes {
    blend: Blend::Over,
    mask: ColorWriteMask::ALL,
};

/// The engine fixture `stem`'s library on `lane`'s device: `tob_renders` (a rectangle from a
/// buffer's corners and a solid color from a buffer) or `function_constant_interface`.
fn fixture(lane: &Lane, stem: &str) -> Library {
    let bitcode = llvm_assemble(&engine_dir().join(format!("{stem}.ll"))).unwrap();
    lane.device.new_library_with_air(&bitcode).unwrap()
}

/// What a pipeline's fragment function writes: `tob_solid`'s color 0 from a buffer, or
/// `fc_fragment_gated_color` with its second output on (color 0 the position's x and y, color 1
/// red).
#[derive(Clone, Copy)]
enum Fragment {
    Solid,
    Gated,
}

/// The rectangle pipeline drawing the solid color into `colors`, each written by `writes`, at
/// `samples`.
fn pipeline(
    lane: &Lane,
    library: &Library,
    colors: &[Option<PixelFormat>],
    writes: Writes,
    samples: u64,
) -> RenderPipelineState {
    drawing(lane, library, Fragment::Solid, colors, writes, samples)
}

/// The rectangle pipeline drawing `fragment` into `colors`, each written by `writes`, at
/// `samples`.
fn drawing(
    lane: &Lane,
    library: &Library,
    fragment: Fragment,
    colors: &[Option<PixelFormat>],
    writes: Writes,
    samples: u64,
) -> RenderPipelineState {
    let descriptor = drawing_descriptor(lane, library, fragment, colors, writes, samples);
    lane.device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// The kinds the making of [`drawing`]'s pipeline announces, and the `Shared` buffers of its
/// draws.
fn made(
    lane: &Lane,
    library: &Library,
    (fragment, colors): (Fragment, &[Option<PixelFormat>]),
    writes: Writes,
    samples: u64,
) -> BTreeSet<Degradation> {
    let descriptor = drawing_descriptor(lane, library, fragment, colors, writes, samples);
    let mut first = forced(&[(
        Degradation::SharedInSystemMemory,
        shared_in_system_memory(&lane.device),
    )]);
    first.extend(made_render_kinds(&lane.device, &descriptor));
    first
}

/// The descriptor of [`drawing`]'s pipeline.
fn drawing_descriptor(
    lane: &Lane,
    library: &Library,
    fragment: Fragment,
    colors: &[Option<PixelFormat>],
    writes: Writes,
    samples: u64,
) -> RenderPipelineDescriptor {
    let function = |name: &str| {
        StageSource::Function(
            library
                .new_function_with_name(name)
                .unwrap()
                .function_descriptor(),
        )
    };
    let mut descriptor = RenderPipelineDescriptor::new(function("tob_rect"));
    descriptor.fragment = Some(match fragment {
        Fragment::Solid => function("tob_solid"),
        Fragment::Gated => {
            let constants = FunctionConstantValues::defaults()
                .with_index(2, FunctionConstantValue::new(DataType::Bool, [1]));
            let function = fixture(lane, "function_constant_interface")
                .new_function_with_name_constant_values("fc_fragment_gated_color", &constants)
                .unwrap();
            StageSource::Function(function.function_descriptor())
        }
    });
    for (index, format) in colors.iter().enumerate() {
        descriptor.color_attachments[index] = format.map(|pf| {
            let mut color = ColorAttachmentDescriptor::new(pf);
            color.write_mask = writes.mask;
            let (source, destination) = match writes.blend {
                Blend::Off => return color,
                Blend::Over => (BlendFactor::SourceAlpha, BlendFactor::OneMinusSourceAlpha),
                Blend::Add => (BlendFactor::One, BlendFactor::One),
            };
            color.blending_enabled = true;
            color.source_rgb_blend_factor = source;
            color.destination_rgb_blend_factor = destination;
            color.source_alpha_blend_factor = BlendFactor::One;
            color.destination_alpha_blend_factor = match writes.blend {
                Blend::Add => BlendFactor::One,
                _ => BlendFactor::Zero,
            };
            color
        });
    }
    descriptor.raster_sample_count = samples as u32;
    descriptor
}

fn clear() -> ClearColor {
    ClearColor::new(CLEAR[0], CLEAR[1], CLEAR[2], CLEAR[3])
}

/// A pass of `targets` (each cleared, stored, and resolved into its resolve texture if any).
fn pass(targets: &[(&Texture, Option<&Texture>)]) -> RenderPassDescriptor {
    let targets: Vec<_> = targets.iter().copied().map(Some).collect();
    gapped(&targets)
}

/// A pass of `targets`, `None` for no attachment at that index (each cleared, stored, and
/// resolved into its resolve texture if any).
fn gapped(targets: &[Option<(&Texture, Option<&Texture>)>]) -> RenderPassDescriptor {
    let mut descriptor = RenderPassDescriptor::new();
    for (index, target) in targets.iter().enumerate() {
        let Some((texture, resolve)) = *target else {
            continue;
        };
        let target = &mut descriptor.color_attachments[index];
        target.texture = Some(texture.clone());
        target.load_action = LoadAction::Clear;
        target.clear_color = clear();
        target.store_action = match resolve {
            Some(_) => StoreAction::StoreAndMultisampleResolve,
            None => StoreAction::Store,
        };
        target.resolve_texture = resolve.cloned();
    }
    descriptor
}

/// Draw the left half by `pipeline` in `descriptor`'s pass.
fn draw(lane: &Lane, descriptor: &RenderPassDescriptor, pipeline: &RenderPipelineState) {
    let floats = |values: &[f32]| {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        super::buffer_with(&lane.device, &bytes).unwrap()
    };
    lane.run(descriptor, |_, encoder, held: &mut Vec<Buffer>| {
        encoder.set_render_pipeline_state(pipeline).unwrap();
        let (corners, color) = (floats(&LEFT), floats(&COLOR));
        encoder.set_vertex_buffer(Some(&corners), 0, 0).unwrap();
        encoder.set_fragment_buffer(Some(&color), 0, 0).unwrap();
        encoder
            .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
            .unwrap();
        held.extend([corners, color]);
    })
    .unwrap();
}

/// The bytes a pipeline of `state` writing by `writes` stores into a pass of `pass` at
/// `samples` (and what that pass resolves, at more than one), 4 bytes a texel.
fn stored(
    lane: &Lane,
    library: &Library,
    (state, pass_format): (PixelFormat, PixelFormat),
    writes: Writes,
    samples: u64,
) -> (Vec<u8>, Option<Vec<u8>>) {
    let usage = TextureUsage::RENDER_TARGET;
    let shape = if samples > 1 { MSAA } else { ONE };
    let target = lane.texture(pass_format, shape, usage);
    let resolve = (samples > 1).then(|| lane.texture(pass_format, ONE, usage));
    let state = pipeline(lane, library, &[Some(state)], writes, samples);
    draw(lane, &pass(&[(&target, resolve.as_ref())]), &state);
    let read = |texture: &Texture| lane.read(texture, (0, 0), 4, BlitOption::None);
    let drawn = match samples {
        1 => read(&target),
        _ => Vec::new(),
    };
    (drawn, resolve.as_ref().map(read))
}

/// Each channel-order twin, each way, stores what a pipeline of the pass's own format stores:
/// unblended, blended, masked to red and alpha, and blended at 4 samples, resolved.
#[test]
fn a_channel_order_twin_stores_the_pass_format_s_bytes() {
    let (messages, _cache, lane) = lane();
    let name = lane.device.name().to_owned();
    let library = fixture(&lane, "tob_renders");
    let twins = [
        (PixelFormat::RGBA8Unorm, PixelFormat::BGRA8Unorm),
        (PixelFormat::BGRA8Unorm, PixelFormat::RGBA8Unorm),
        (PixelFormat::RGBA8UnormSRGB, PixelFormat::BGRA8UnormSRGB),
        (PixelFormat::BGRA8UnormSRGB, PixelFormat::RGBA8UnormSRGB),
        (PixelFormat::RGB10A2Unorm, PixelFormat::BGR10A2Unorm),
        (PixelFormat::BGR10A2Unorm, PixelFormat::RGB10A2Unorm),
    ];
    let masked = Writes {
        blend: Blend::Off,
        mask: ColorWriteMask::RED | ColorWriteMask::ALPHA,
    };
    let ways = [
        ("plain", PLAIN, 1),
        ("blended", OVER, 1),
        ("masked", masked, 1),
        ("blended at 4x", OVER, 4),
    ];
    for (state, pass_format) in twins {
        for (how, writes, samples) in ways {
            let twin = stored(&lane, &library, (state, pass_format), writes, samples);
            let own = stored(&lane, &library, (pass_format, pass_format), writes, samples);
            assert_eq!(
                twin, own,
                "{state:?} {how} into {pass_format:?} on {name}: the pass format's bytes"
            );
        }
    }
    // Metal's bytes where the draw covers (the left two texels of each row; the clear's
    // rounding of 0.5 is the device's): blue first in a `BGRA8Unorm` pass, red first in an
    // `RGBA8Unorm` one.
    let (bgra, _) = stored(
        &lane,
        &library,
        (PixelFormat::RGBA8Unorm, PixelFormat::BGRA8Unorm),
        PLAIN,
        1,
    );
    let (rgba, _) = stored(
        &lane,
        &library,
        (PixelFormat::BGRA8Unorm, PixelFormat::RGBA8Unorm),
        PLAIN,
        1,
    );
    let drawn =
        |bytes: &[u8]| -> Vec<u8> { bytes.chunks(16).flat_map(|row| row[..8].to_vec()).collect() };
    assert_eq!(
        drawn(&bgra),
        [0x99, 0x66, 0x33, 0xcc].repeat(8),
        "on {name}"
    );
    assert_eq!(
        drawn(&rgba),
        [0x33, 0x66, 0x99, 0xcc].repeat(8),
        "on {name}"
    );
    eprintln!("channel-order twins on {name}: the pass format's bytes, as Metal");
    quiet(lane, &messages);
}

/// The bytes each pass color of `formats` (`None` for no attachment) holds after `state` draws
/// the left half at `samples`: each stored at 1 sample, resolved at more.
fn drawn(
    lane: &Lane,
    state: &RenderPipelineState,
    formats: &[Option<PixelFormat>],
    samples: u64,
) -> Vec<Option<Vec<u8>>> {
    let usage = TextureUsage::RENDER_TARGET;
    let shape = if samples > 1 { MSAA } else { ONE };
    let targets: Vec<Option<(Texture, Option<Texture>)>> = formats
        .iter()
        .map(|format| {
            let pf = (*format)?;
            let resolve = (samples > 1).then(|| lane.texture(pf, ONE, usage));
            Some((lane.texture(pf, shape, usage), resolve))
        })
        .collect();
    let borrowed: Vec<_> = targets
        .iter()
        .map(|t| {
            t.as_ref()
                .map(|(texture, resolve)| (texture, resolve.as_ref()))
        })
        .collect();
    draw(lane, &gapped(&borrowed), state);
    targets
        .iter()
        .map(|t| {
            let (texture, resolve) = t.as_ref()?;
            let read = resolve.as_ref().unwrap_or(texture);
            Some(lane.read(read, (0, 0), 4, BlitOption::None))
        })
        .collect()
}

/// The texels of 4² `RGBA8Unorm` bytes the left half covers, and the rest.
fn halves(bytes: &[u8]) -> (Vec<u8>, Vec<u8>) {
    let rows = bytes.chunks(16);
    let left = rows.clone().flat_map(|row| row[..8].to_vec()).collect();
    (left, rows.flat_map(|row| row[8..].to_vec()).collect())
}

const RGBA8: Option<PixelFormat> = Some(PixelFormat::RGBA8Unorm);

/// Each state color is drawn into the pass color at its place in Metal's per-sample record,
/// blended as the state's color there, either order, and a pass color at no place is zeroed
/// where Metal's code reads no attachment (the module docs).
#[test]
fn a_state_color_draws_into_the_pass_color_at_its_place_in_the_record() {
    let (messages, _cache, lane) = lane();
    let name = lane.device.name().to_owned();
    let solid = fixture(&lane, "tob_renders");
    // The kinds the pipelines' makings and the resolves at 4 samples announce.
    let mut first = BTreeSet::new();
    // A pass lacking color 0, under a state of colors 0 and 1: the pass's color 1 holds what the
    // state's color 0 stores into a pass of color 0 (probe 3 `g_rgba8_rgba8_in_none_rgba8`),
    // plain, blended (`_blend0`), and at 4 samples, resolved.
    for (how, writes, samples) in [
        ("plain", PLAIN, 1),
        ("blended", OVER, 1),
        ("at 4x", PLAIN, 4),
    ] {
        let both = pipeline(&lane, &solid, &[RGBA8, RGBA8], writes, samples);
        let alone = pipeline(&lane, &solid, &[RGBA8], writes, samples);
        for colors in [&[RGBA8, RGBA8][..], &[RGBA8]] {
            first.extend(made(
                &lane,
                &solid,
                (Fragment::Solid, colors),
                writes,
                samples,
            ));
        }
        if samples > 1 {
            first.extend(resolve_kind(&lane.device, samples as u32));
        }
        let routed = drawn(&lane, &both, &[None, RGBA8], samples);
        let own = drawn(&lane, &alone, &[RGBA8], samples);
        assert_eq!(
            routed[1], own[0],
            "{how} on {name}: the state's color 0's bytes"
        );
        if samples == 1 && writes.blend == Blend::Off {
            let (left, _) = halves(routed[1].as_ref().unwrap());
            assert_eq!(left, [0x33, 0x66, 0x99, 0xcc].repeat(8), "{how} on {name}");
        }
    }
    // A state of color 1 alone in a pass of colors 0 and 1: red into the pass's color 0, zero
    // into its color 1 where the draw covers, the clear elsewhere (probe 3
    // `g_none_rgba8_in_rgba8_rgba8`).
    let second = drawing(&lane, &solid, Fragment::Gated, &[None, RGBA8], PLAIN, 1);
    first.extend(made(
        &lane,
        &solid,
        (Fragment::Gated, &[None, RGBA8]),
        PLAIN,
        1,
    ));
    let both = drawn(&lane, &second, &[RGBA8, RGBA8], 1);
    let (left0, rest0) = halves(both[0].as_ref().unwrap());
    let (left1, rest1) = halves(both[1].as_ref().unwrap());
    assert_eq!(left0, [0xff, 0, 0, 0xff].repeat(8), "color 0 on {name}");
    assert_eq!(left1, [0; 32], "color 1 on {name}");
    assert_eq!(rest0, rest1, "the clear on {name}");
    // Every pass clears to `CLEAR`, whose red 0.5 is a tie in an 8-bit channel.
    first.extend(clear_in_question(&lane.device, true));
    let notices = assert_made_once(&messages, &[], &first);
    assert!(!notices.contains_key(&KEPT_COLORS[0]), "on {name}");
    eprintln!("colors at their places in the record on {name}: Metal's bytes");
    quiet(lane, &messages);
}

/// A pass color at no place of the state's record keeps its contents where Metal's compiled
/// code reads the attachment: exactly under a partial mask and a `One` destination factor; with
/// the `uncovered-color-kept` notice under a blend over the output's alpha (the module docs).
#[test]
fn an_uncovered_color_is_kept_where_metal_s_code_reads_the_attachment() {
    let (messages, _cache, lane) = lane();
    let name = lane.device.name().to_owned();
    let solid = fixture(&lane, "tob_renders");
    // Color 1 holds its clear where the draw covers as where it does not.
    let kept = |writes: Writes| {
        let state = pipeline(&lane, &solid, &[RGBA8], writes, 1);
        let colors = drawn(&lane, &state, &[RGBA8, RGBA8], 1);
        let (left, right) = halves(colors[1].as_ref().unwrap());
        assert_eq!(left, right, "kept on {name}");
    };
    let exact = [
        Writes {
            blend: Blend::Off,
            mask: ColorWriteMask::RED | ColorWriteMask::ALPHA,
        },
        Writes {
            blend: Blend::Add,
            mask: ColorWriteMask::ALL,
        },
    ];
    let mut first = BTreeSet::new();
    for writes in exact {
        kept(writes);
        first.extend(made(&lane, &solid, (Fragment::Solid, &[RGBA8]), writes, 1));
    }
    // Every pass clears to `CLEAR`, whose red 0.5 is a tie in an 8-bit channel.
    first.extend(clear_in_question(&lane.device, true));
    let notices = assert_made_once(&messages, &[], &first);
    assert!(!notices.contains_key(&KEPT_COLORS[0]), "exact on {name}");
    kept(OVER);
    kept(OVER);
    let first = made(&lane, &solid, (Fragment::Solid, &[RGBA8]), OVER, 1);
    assert_made_once(&messages, &KEPT_COLORS, &first);
    eprintln!("uncovered colors kept on {name}, as Metal");
    quiet(lane, &messages);
}

/// The bytes a pipeline of `state` writing by `writes` stores into a one-sample pass of
/// `pass_format` made with `usage`, cleared to `clear`, 4 bytes a texel.
fn stored_in(
    lane: &Lane,
    library: &Library,
    (state, pass_format): (PixelFormat, PixelFormat),
    (writes, usage): (Writes, TextureUsage),
    clear: ClearColor,
) -> Vec<u8> {
    let target = lane.texture(pass_format, ONE, usage | TextureUsage::RENDER_TARGET);
    let state = pipeline(lane, library, &[Some(state)], writes, 1);
    let mut descriptor = pass(&[(&target, None)]);
    descriptor.color_attachments[0].clear_color = clear;
    draw(lane, &descriptor, &state);
    lane.read(&target, (0, 0), 4, BlitOption::None)
}

/// A color of another encoding draws through a view of the attachment's image in the state's
/// encoding, in the pass's channel order, where the image takes the view: the sRGB and plain forms
/// of the 8-bit formats, whose images list both (probe 1 `a_rgba8_in_bgra8s` stored the plain
/// encoding, a probe `fmt_srgb_into_plain` the sRGB one), and a texture made with `PixelFormatView`
/// (a probe: `RGBA8Unorm` into `R32Float` stored the `RGBA8Unorm` bytes). Each pass is cleared to a
/// color both encodings store alike, and stores what a pass of the view's format stores, plain,
/// blended and masked.
#[test]
fn a_color_of_another_encoding_draws_through_a_view_of_the_attachment() {
    let (messages, _cache, lane) = lane();
    let name = lane.device.name().to_owned();
    let library = fixture(&lane, "tob_renders");
    let exact = ClearColor::new(1.0, 0.0, 1.0, 1.0);
    let masked = Writes {
        blend: Blend::Off,
        mask: ColorWriteMask::RED | ColorWriteMask::ALPHA,
    };
    let ways = [("plain", PLAIN), ("blended", OVER), ("masked", masked)];
    use PixelFormat as P;
    // (state, pass, the format of the view's bytes)
    let viewed = [
        (P::RGBA8UnormSRGB, P::RGBA8Unorm, P::RGBA8UnormSRGB),
        (P::RGBA8Unorm, P::RGBA8UnormSRGB, P::RGBA8Unorm),
        (P::BGRA8UnormSRGB, P::RGBA8Unorm, P::RGBA8UnormSRGB),
        (P::RGBA8Unorm, P::BGRA8UnormSRGB, P::BGRA8Unorm),
        (P::RGBA8UnormSRGB, P::BGRA8Unorm, P::BGRA8UnormSRGB),
        (P::BGRA8Unorm, P::RGBA8UnormSRGB, P::RGBA8Unorm),
    ];
    let none = TextureUsage::empty();
    for (state, pass_format, view) in viewed {
        for (how, writes) in ways {
            let drawn = stored_in(&lane, &library, (state, pass_format), (writes, none), exact);
            let own = stored_in(&lane, &library, (view, view), (writes, none), exact);
            assert_eq!(
                drawn, own,
                "{state:?} {how} into {pass_format:?} on {name}: the bytes of {view:?}"
            );
        }
    }
    // Metal's bytes where the draw covers: the plain encoding in a `BGRA8Unorm_sRGB` pass.
    let plain = stored_in(
        &lane,
        &library,
        (P::RGBA8Unorm, P::BGRA8UnormSRGB),
        (PLAIN, none),
        exact,
    );
    let (left, _) = halves(&plain);
    assert_eq!(left, [0x99, 0x66, 0x33, 0xcc].repeat(8), "on {name}");
    // An `RGBA8Unorm` state in an `R32Float` pass made with `PixelFormatView` stores its bytes.
    let floats = stored_in(
        &lane,
        &library,
        (P::RGBA8Unorm, P::R32Float),
        (PLAIN, TextureUsage::PIXEL_FORMAT_VIEW),
        exact,
    );
    let (left, _) = halves(&floats);
    assert_eq!(left, [0x33, 0x66, 0x99, 0xcc].repeat(8), "on {name}");
    eprintln!("colors of another encoding through views on {name}: the view's bytes, as Metal");
    quiet(lane, &messages);
}

/// The bytes of a one-sample `R32Float` pass made with `usage`, and how long its command buffer
/// took, after `rounds` rounds of three draws: an `R32Float` state over the whole target, an
/// `RGBA8Unorm` state blended over the left half (reading the first draw's bytes), and the
/// `R32Float` state over the right half (after the second's instance). Each round's second draw
/// renders through a view, or a stand-in where the image takes none, in an instance of its own.
fn rounds(
    lane: &Lane,
    library: &Library,
    usage: TextureUsage,
    rounds: usize,
) -> (Vec<u8>, std::time::Duration) {
    const FULL: [f32; 4] = [-1.0, -1.0, 1.0, 1.0];
    const RIGHT: [f32; 4] = [0.0, -1.0, 1.0, 1.0];
    let target = lane.texture(
        PixelFormat::R32Float,
        ONE,
        usage | TextureUsage::RENDER_TARGET,
    );
    let floats = pipeline(lane, library, &[Some(PixelFormat::R32Float)], PLAIN, 1);
    let bytes = pipeline(lane, library, &[RGBA8], OVER, 1);
    let buffer = |values: &[f32]| {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        super::buffer_with(&lane.device, &bytes).unwrap()
    };
    let started = std::time::Instant::now();
    lane.run(
        &pass(&[(&target, None)]),
        |_, encoder, held: &mut Vec<Buffer>| {
            let (full, left, right) = (buffer(&FULL), buffer(&LEFT), buffer(&RIGHT));
            let color = buffer(&COLOR);
            encoder.set_fragment_buffer(Some(&color), 0, 0).unwrap();
            for _ in 0..rounds {
                for (state, rect) in [(&floats, &full), (&bytes, &left), (&floats, &right)] {
                    encoder.set_render_pipeline_state(state).unwrap();
                    encoder.set_vertex_buffer(Some(rect), 0, 0).unwrap();
                    encoder
                        .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
                        .unwrap();
                }
            }
            held.extend([full, left, right, color]);
        },
    )
    .unwrap();
    let took = started.elapsed();
    (lane.read(&target, (0, 0), 4, BlitOption::None), took)
}

/// A color of another encoding whose attachment's image takes no view of it draws through a
/// stand-in image of the view's format, copied from the attachment before its instance and back
/// after it: an `RGBA8Unorm` state in an `R32Float` pass stores its bytes (a probe), and draws
/// before and after it, of the pass's own format, read and leave what a view of a `PixelFormatView`
/// texture does. Prints what the stand-in's copies cost against the view (`STAND-IN-COST`),
/// measured, not asserted.
#[test]
fn a_color_of_another_encoding_draws_through_a_stand_in_where_no_view_is_taken() {
    let (messages, _cache, lane) = lane();
    let name = lane.device.name().to_owned();
    let library = fixture(&lane, "tob_renders");
    let exact = ClearColor::new(1.0, 0.0, 1.0, 1.0);
    let none = TextureUsage::empty();
    let floats = stored_in(
        &lane,
        &library,
        (PixelFormat::RGBA8Unorm, PixelFormat::R32Float),
        (PLAIN, none),
        exact,
    );
    let (left, _) = halves(&floats);
    assert_eq!(left, [0x33, 0x66, 0x99, 0xcc].repeat(8), "on {name}");
    let view = TextureUsage::PIXEL_FORMAT_VIEW;
    let (viewed, _) = rounds(&lane, &library, view, 1);
    let (stood, _) = rounds(&lane, &library, none, 1);
    assert_eq!(stood, viewed, "three draws on {name}: the view's bytes");
    // Warm, then 64 rounds each way.
    rounds(&lane, &library, view, 2);
    rounds(&lane, &library, none, 2);
    let (_, through_view) = rounds(&lane, &library, view, 64);
    let (_, through_stand_in) = rounds(&lane, &library, none, 64);
    eprintln!(
        "STAND-IN-COST on {name}: 64 rounds of 3 draws (a 4x4 pass, 128 instances): through a \
         view {through_view:?}, through a stand-in {through_stand_in:?}"
    );
    quiet(lane, &messages);
}

/// A pair whose bytes no view stores, another encoding in a multisampled pass, and a color whose
/// bytes plain Metal stores into part of an attachment, stay refused at
/// `set_render_pipeline_state`.
#[test]
fn colors_no_view_stores_or_in_parts_stay_refused() {
    let (messages, _cache, lane) = lane();
    let library = fixture(&lane, "tob_renders");
    let refused = |index| {
        Err(Error::Invalid(Invalid::new(
            "set_render_pipeline_state",
            Problem::Draw(DrawProblem::AttachmentFormat(Attachment::Color(index))),
        )))
    };
    let usage = TextureUsage::RENDER_TARGET;
    let plain = pipeline(&lane, &library, &[RGBA8], PLAIN, 1);
    // A signed normalized pass (a probe: it stores -128 as -127).
    let snorm = lane.texture(PixelFormat::RGBA8Snorm, ONE, usage);
    lane.run(&pass(&[(&snorm, None)]), |_, encoder, _| {
        assert_eq!(encoder.set_render_pipeline_state(&plain), refused(0));
    })
    .unwrap();
    // A 4-sample `R32Float` pass under a 4-sample `RGBA8Unorm` state (unmeasured past the
    // channel-order twins), resolved.
    let samples = pipeline(&lane, &library, &[RGBA8], PLAIN, 4);
    let floats = lane.texture(PixelFormat::R32Float, MSAA, usage);
    let resolved = lane.texture(PixelFormat::R32Float, ONE, usage);
    lane.run(&pass(&[(&floats, Some(&resolved))]), |_, encoder, _| {
        assert_eq!(encoder.set_render_pipeline_state(&samples), refused(0));
    })
    .unwrap();
    // An `R8Unorm` color 0 beside an `RGBA8Unorm` color 1, in a pass of color 1 alone (probe 3
    // `g_r8_rgba8_in_none_rgba8`: both colors' bytes in the pass's color 1).
    let parts = pipeline(
        &lane,
        &library,
        &[Some(PixelFormat::R8Unorm), RGBA8],
        PLAIN,
        1,
    );
    let color1 = lane.texture(PixelFormat::RGBA8Unorm, ONE, usage);
    lane.run(&gapped(&[None, Some((&color1, None))]), |_, encoder, _| {
        assert_eq!(encoder.set_render_pipeline_state(&parts), refused(0));
    })
    .unwrap();
    quiet(lane, &messages);
}
