//! Blending into render targets of Apple's extended-range formats on the device the cfg selects,
//! under the validation layer, against Metal's bytes.
//!
//! Each case replays one of a Metal probe `rtp1`'s blend draws: a 128 x 64 target of each format
//! holding the probe's random raw texels (seed `1000 + format`, [`targets`]; plane 0 by
//! `replace_region`, a two-plane format's alpha bytes first by the fixture `xr_two_planes`'s
//! `float` write of `b / 255`), one full-screen draw of the owned fixture `rt6_samples`'s `half4`
//! fill (`rt6_fill_rgba16float`) of the probe's 8,192 outputs (seed 77, as
//! `extended_range_targets`) under the load action `Load` and a blend of source factor one, added,
//! for color and alpha, with destination factor zero (`one_zero`), one (`add`), or one minus the
//! source's alpha (`premul`, the consumer's). The runtime blends such an attachment in the shader,
//! inside the pixel interlock, from the texel's tile value (`extended_range::blend::native`), or on
//! a device with raster-order attachment access (lavapipe) over the attachment's input attachment;
//! any refusal fails.
//!
//! Metal's answer: the SHA-256 of the target's bytes after the draw, a two-plane target's plane 0
//! words and alpha bytes apart ([`METAL`]; an Apple M4 under macOS 27.0.1, plain
//! and under Metal's validation layer alike, byte-identical to the M3 Max's probe). A
//! two-plane target's words read back through a blit (Metal's plane 0), its alpha bytes as
//! `round(255 a)` of the fixture's `float` read. A differing case is printed with its first
//! differing texels on one `XR-BLENDS-DIFFERS` line.

use super::degrade_notices::shared_in_system_memory;
use super::degrade_notices::{assert_made_once, counts, forced, made_render_kinds};
use super::extended_range_targets::{
    digest, dispatch, half_bits, half_value, kernel, outputs, reads, COUNT, HEIGHT, SOURCE_HALF,
    WIDTH,
};
use super::resolves::{lane, lane_with, quiet, Lane, FULL, WHITE};
use super::two_plane_copies::Rng;
use super::{buffer_with, read_buffer};
use crate::candidate::runtime_samples::{library, shape};
use metal2vulkan::metal::format::{ClearRoute, ExtendedRange};
use metal2vulkan::metal::{
    BarrierScope, BlendFactor, ColorWriteMask, IndirectCommandType, PixelFormat, PrimitiveType,
    RenderStages, TextureUsage,
};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_runtime::{
    BlitOption, Buffer, ClearColor, DebugMessage, Degradation, Error,
    IndirectCommandBufferDescriptor, LoadAction, PipelineOptions, Region, RenderPassDescriptor,
    ResourceOptions, StoreAction, Texture, TextureDescriptor,
};
use std::collections::BTreeSet;

/// The formats, by the probe's names.
const FORMATS: [(PixelFormat, &str); 6] = [
    (PixelFormat::BGRA10XR, "bgra10xr"),
    (PixelFormat::BGRA10XRSRGB, "bgra10xr_srgb"),
    (PixelFormat::BGR10XR, "bgr10xr"),
    (PixelFormat::BGR10XRSRGB, "bgr10xr_srgb"),
    (PixelFormat::RGB10A82PXR10, "rgb10a8_2p"),
    (PixelFormat::RGB10A82PXR10SRGB, "rgb10a8_2p_srgb"),
];

/// The blends: the probe's name and the destination factor (color and alpha alike) under source
/// factor one, added.
const BLENDS: [(&str, BlendFactor); 3] = [
    ("one_zero", BlendFactor::Zero),
    ("add", BlendFactor::One),
    ("premul", BlendFactor::OneMinusSourceAlpha),
];

/// The SHA-256 of the probe's targets before the draws (`rt.<format>.dst0`, `.dsta`).
const TARGETS: [(&str, &str); 8] = [
    (
        "rt.bgra10xr.dst0",
        "5500646f25136cb1ee712c537bf52bd652692caf5ac75bfec166ad054851391c",
    ),
    (
        "rt.bgra10xr_srgb.dst0",
        "ec04259f2cf0abd058a45a43d2ae42eee1d2b69f9a75380a6e220f2b80ee3885",
    ),
    (
        "rt.bgr10xr.dst0",
        "86f240ed1326042e075dfb6b630025040e7d0900516125a06e202d05dd59b4ea",
    ),
    (
        "rt.bgr10xr_srgb.dst0",
        "10d671ec19741d21a6cd2d5684458fff5cb4bfcfdf9a6f7656497ad057874073",
    ),
    (
        "rt.rgb10a8_2p.dst0",
        "3a4d452aaf8e6c1ac7f54f37d6ae19ea7d9fff3ac8092124734437f6a21ec62f",
    ),
    (
        "rt.rgb10a8_2p.dsta",
        "773162c1efa6ad307364ff468da6540c4e991d2f70a7aed787dc303d77870d7b",
    ),
    (
        "rt.rgb10a8_2p_srgb.dst0",
        "5f100cc206d94f512a8a0e72c14685adb2de0bf1d22333bc7f6fc767dd55fbf1",
    ),
    (
        "rt.rgb10a8_2p_srgb.dsta",
        "9629c65d83e460c4a303776ab33ca4fe0a35ae89f9f69356792fa49bd82c183e",
    ),
];

/// Metal's targets after each draw: the probe's output and its SHA-256 (`.out0` plane 0, `.outa`
/// a two-plane format's alpha bytes).
const METAL: [(&str, &str); 24] = [
    (
        "rt.bgra10xr.h.one_zero.out0",
        "874c66dd7d289ca4727ab716199b96b05c2656777a28b0ff911e17e1c82226cc",
    ),
    (
        "rt.bgra10xr.h.add.out0",
        "8105b650b07b73a6762d2ce596b9a02cec1f2ba6e3e56ce3c0957a60db6de3a3",
    ),
    (
        "rt.bgra10xr.h.premul.out0",
        "ccf3234f3107c24c02d479b43e1c573fc498b026d248675a46cdbd3b77b530bf",
    ),
    (
        "rt.bgra10xr_srgb.h.one_zero.out0",
        "fa2120f896eee071a193eae70de62f00992eca84d2c66b9f7f5e94d9debec0b2",
    ),
    (
        "rt.bgra10xr_srgb.h.add.out0",
        "390d3abb63dd550203684a0c7d1ee478d8a0dc944b6e0c8070f996cdaa30dbe3",
    ),
    (
        "rt.bgra10xr_srgb.h.premul.out0",
        "32fd83f3bf10fa24ba5681e65c00909ad42697e1bef297cf59f19e2cfc205739",
    ),
    (
        "rt.bgr10xr.h.one_zero.out0",
        "1a944db25bbe9a9f104b732cd6f9c7eb4e32446f1fe5c9a8427ffa6e1e2d7753",
    ),
    (
        "rt.bgr10xr.h.add.out0",
        "a8afb7d94a14f3b97c2c86101b5e04a4d1118c9e0b432637a2de381b039051da",
    ),
    (
        "rt.bgr10xr.h.premul.out0",
        "c351886567031dfcefe833c8ece1d8a1773899aec7d6be49aa732430c034b924",
    ),
    (
        "rt.bgr10xr_srgb.h.one_zero.out0",
        "f330bb938200100f733d55341eea1c8c00244feeb3db70a15e424fe37e1b4be6",
    ),
    (
        "rt.bgr10xr_srgb.h.add.out0",
        "92852363a178020cbad91ad6963c616128f5aa627afc3b3ed551d79c66de9c81",
    ),
    (
        "rt.bgr10xr_srgb.h.premul.out0",
        "f65ae73b150e1830c78ca13c511380bc377800682b81b45075ac2452859c6b5c",
    ),
    (
        "rt.rgb10a8_2p.h.one_zero.out0",
        "385e9a912cd02e931250a0b7c1bcbe1698ba7f105e5745cf4f9e7c2a798d5daa",
    ),
    (
        "rt.rgb10a8_2p.h.one_zero.outa",
        "da175edfc5cbcb3d7428f49d8441ac42d4237a208f1b5b13fcde98f0b3b1384a",
    ),
    (
        "rt.rgb10a8_2p.h.add.out0",
        "191ffaf0a95b0d8a1d24cca7b3337b0927e5796f63b31dbb597510a03d24f6ad",
    ),
    (
        "rt.rgb10a8_2p.h.add.outa",
        "c755a8468b188b6bff7708b299c0fa4d8ef299db2c69fa1093c7e25314f8a090",
    ),
    (
        "rt.rgb10a8_2p.h.premul.out0",
        "12d9bbb76943ff3dd618d4bcae721eca7c38f7a30f53a549d0ff71f378d6e391",
    ),
    (
        "rt.rgb10a8_2p.h.premul.outa",
        "773d47247740043302924a7df179eb06e2c5f4fafac1182651987dc64049a059",
    ),
    (
        "rt.rgb10a8_2p_srgb.h.one_zero.out0",
        "336275f01108d01299469fbd930bf78fb97e298f3c9f8ebe6aebf8f2ab6f8454",
    ),
    (
        "rt.rgb10a8_2p_srgb.h.one_zero.outa",
        "da175edfc5cbcb3d7428f49d8441ac42d4237a208f1b5b13fcde98f0b3b1384a",
    ),
    (
        "rt.rgb10a8_2p_srgb.h.add.out0",
        "498e728fff2d7471a3df16defd5f4dbc4d055259831b191199172dbbfd49265e",
    ),
    (
        "rt.rgb10a8_2p_srgb.h.add.outa",
        "568be8ce0915916cce80be8c15e4531257ca8fdbdc0fadd34754ea063ddba746",
    ),
    (
        "rt.rgb10a8_2p_srgb.h.premul.out0",
        "475e90b4016b64d798a459e1d4541f41dc7c083dd4b95c2c1f67c07a6ef34792",
    ),
    (
        "rt.rgb10a8_2p_srgb.h.premul.outa",
        "599e8d1d35161c67b148b7c85c267cfc56ad03ce86a13bdc50c3dbf7b59b1b35",
    ),
];

fn metal(name: &str) -> &'static str {
    METAL
        .iter()
        .chain(TARGETS.iter())
        .find(|m| m.0 == name)
        .map(|m| m.1)
        .unwrap_or_else(|| panic!("no Metal digest of {name}"))
}

/// The probe's random raw texels of `form` (`gen_dst`, seed `1000 + raw`, the format's Metal
/// number): plane 0's bytes (a `BGRA10_XR` texel's four 16-bit fields, each a code shifted up 6
/// with 6 random low bits, alpha's code mostly in `[384, 894]`; a word of three codes and 2 random
/// top bits otherwise), and a two-plane format's alpha bytes.
pub(super) fn targets(form: ExtendedRange, raw: u64) -> (Vec<u8>, Vec<u8>) {
    let mut r = Rng::new(1000 + raw);
    let code = |r: &mut Rng, alpha: bool| -> u64 {
        if alpha && r.below(10) < 8 {
            return 384 + r.below(511);
        }
        r.below(1024)
    };
    let (mut plane, mut alpha) = (Vec::new(), Vec::new());
    for _ in 0..COUNT {
        match form {
            ExtendedRange::Bgra10 { .. } => {
                for c in 0..4 {
                    let field = (code(&mut r, c == 3) << 6 | r.below(64)) as u16;
                    plane.extend(field.to_le_bytes());
                }
            }
            _ => {
                let top = r.below(4) << 30;
                let (a, b) = (code(&mut r, false), code(&mut r, false));
                let word = (top | a << 20 | b << 10 | code(&mut r, false)) as u32;
                plane.extend(word.to_le_bytes());
                if matches!(form, ExtendedRange::Planes { .. }) {
                    alpha.push(r.below(256) as u8);
                }
            }
        }
    }
    (plane, alpha)
}

/// A shared texture of `pf` the draws render to and the kernels read and write.
fn target(lane: &Lane, pf: PixelFormat) -> Texture {
    let mut descriptor = TextureDescriptor::texture_2d(pf, WIDTH as u64, HEIGHT as u64, false);
    descriptor.usage =
        TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
    descriptor.resource_options = ResourceOptions::shared();
    lane.device
        .new_texture(&descriptor)
        .unwrap_or_else(|e| panic!("a {pf:?} texture: {e}"))
}

/// `pf`'s target holding the probe's texels `(plane, alpha)`, written as the probe writes them.
fn filled(lane: &Lane, pf: PixelFormat, (plane, alpha): (&[u8], &[u8])) -> Texture {
    let t = target(lane, pf);
    let (w, h) = (WIDTH as u64, HEIGHT as u64);
    if !alpha.is_empty() {
        let values: Vec<u8> = alpha
            .iter()
            .flat_map(|&b| [0.0, 0.0, 0.0, f32::from(b) / 255.0])
            .flat_map(f32::to_le_bytes)
            .collect();
        let values = buffer_with(&lane.device, &values).unwrap();
        dispatch(
            lane,
            &kernel(lane, "xr2p_write_float"),
            &t,
            (&values, 0),
            (w, h),
        );
    }
    let row = plane.len() as u64 / (w * h) * w;
    t.replace_region(Region::new_2d(0, 0, w, h), 0, 0, plane, row, 0)
        .unwrap();
    t
}

/// The descriptor of [`draws`]' pipeline: `rt6_fill_rgba16float`'s `half4` outputs into `color`.
fn draw_descriptor(lane: &Lane, color: ColorAttachmentDescriptor) -> RenderPipelineDescriptor {
    let library = library(&lane.device).unwrap();
    let function = |name: &str| {
        StageSource::Function(
            library
                .new_function_with_name(name)
                .unwrap()
                .function_descriptor(),
        )
    };
    let mut pipeline = RenderPipelineDescriptor::new(function("rt6_fill_vertex"));
    pipeline.fragment = Some(function("rt6_fill_rgba16float"));
    pipeline.color_attachments[0] = Some(color);
    pipeline
}

/// The kinds the makings of `descriptors` announce, and the `Shared` buffers of the draws.
fn made(lane: &Lane, descriptors: &[RenderPipelineDescriptor]) -> BTreeSet<Degradation> {
    let mut first = forced(&[(
        Degradation::SharedInSystemMemory,
        shared_in_system_memory(&lane.device),
    )]);
    for descriptor in descriptors {
        first.extend(made_render_kinds(&lane.device, descriptor));
    }
    first
}

/// One draw of the probe's `half4` outputs (`values`) into `target`, loaded and stored, through a
/// pipeline whose color 0 is `color`; the pipeline's error where the device refuses it.
fn draw(
    lane: &Lane,
    target: &Texture,
    values: &[u8],
    color: ColorAttachmentDescriptor,
) -> Result<(), Error> {
    draws(lane, target, values, color, (None, 1, false))
}

/// [`draw`]'s draw `count` times in its one pass, which clears `target` to `clear` where there
/// is one. Where `split`, a memory barrier of the buffers between each two draws, which splits the
/// pass there.
fn draws(
    lane: &Lane,
    target: &Texture,
    values: &[u8],
    color: ColorAttachmentDescriptor,
    (clear, count, split): (Option<ClearColor>, usize, bool),
) -> Result<(), Error> {
    let what = format!("{:?} under {color:?}", color.pixel_format);
    let pipeline = draw_descriptor(lane, color);
    let pipeline = lane
        .device
        .new_render_pipeline_state(&pipeline, &PipelineOptions::default())?;
    let values = buffer_with(&lane.device, values).unwrap();
    let shape = shape(&lane.device, WIDTH as u32, 1).unwrap();
    let mut pass = RenderPassDescriptor::new();
    let attachment = &mut pass.color_attachments[0];
    attachment.texture = Some(target.clone());
    attachment.load_action = LoadAction::Load;
    attachment.store_action = StoreAction::Store;
    if let Some(clear) = clear {
        attachment.load_action = LoadAction::Clear;
        attachment.clear_color = clear;
    }
    lane.run(&pass, |_, e, _| {
        e.set_render_pipeline_state(&pipeline).unwrap();
        e.set_fragment_buffer(Some(&values), 0, 0).unwrap();
        e.set_fragment_buffer(Some(&shape), 0, 1).unwrap();
        for k in 0..count {
            if split && k > 0 {
                e.memory_barrier_with_scope(
                    BarrierScope::BUFFERS,
                    RenderStages::FRAGMENT,
                    RenderStages::FRAGMENT,
                );
            }
            e.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
                .unwrap();
        }
    })
    .unwrap_or_else(|e| panic!("{what}: the pass: {e}"));
    Ok(())
}

/// The probe's `half4` outputs' bytes (`extended_range_targets`'s, checked against their digest).
fn halves() -> Vec<u8> {
    let halves: Vec<u8> = outputs()
        .iter()
        .flat_map(|v| v.map(half_bits))
        .flat_map(u16::to_le_bytes)
        .collect();
    assert_eq!(digest(&halves), SOURCE_HALF, "the probe's half outputs");
    halves
}

/// The first texels (of `size` bytes) of `got`, in hex.
fn first_texels(got: &[u8], size: usize) -> String {
    got.chunks(size)
        .take(4)
        .map(|t| t.iter().map(|b| format!("{b:02x}")).collect::<String>())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Each format's blends of the probe's `half` outputs over its random texels, of source factor
/// one, added, with destination factor zero, one, and one minus the source's alpha. Where the
/// blend computes nothing (one and zero) every format stores Metal's bytes, which is asserted.
/// Under one and one minus the source's alpha the function blends in the device's arithmetic,
/// where Metal's compiler rounds each step to `half` (`extended-range-blend-source`): how many
/// outputs differ from Metal's is printed, not asserted; the conformance suite's `xr-blend-*`
/// cases hold Metal's bytes and each site the device's arithmetic moves.
#[test]
fn blends_of_half_outputs_into_extended_range_texels() {
    let (messages, _cache, lane) = lane();
    let name = lane.device.name().to_owned();
    let halves = halves();
    let read = kernel(&lane, "xr2p_read_float");
    let size = (WIDTH as u64, HEIGHT as u64);
    let mut wrong = Vec::new();
    let mut moved = Vec::new();
    let mut checked = 0;
    for (pf, probe) in FORMATS {
        let form = ExtendedRange::of(pf).expect("an extended-range format");
        let (plane, alpha) = targets(form, u64::from(pf.as_raw()));
        let texel = plane.len() / COUNT;
        for (blend, dst) in BLENDS {
            let case = format!("rt.{probe}.h.{blend}");
            let t = filled(&lane, pf, (&plane, &alpha));
            let mut color = ColorAttachmentDescriptor::new(pf);
            color.blending_enabled = true;
            color.destination_rgb_blend_factor = dst;
            color.destination_alpha_blend_factor = dst;
            draw(&lane, &t, &halves, color).unwrap_or_else(|e| panic!("{case}: the pipeline: {e}"));
            let got = lane.read(&t, (0, 0), texel as u64, BlitOption::None);
            checked += 1;
            // The device's arithmetic answers a blend that computes something.
            let differs = match dst {
                BlendFactor::Zero => &mut wrong,
                _ => &mut moved,
            };
            if digest(&got) != metal(&format!("{case}.out0")) {
                differs.push(format!(
                    "{case}.out0: the first texels {}",
                    first_texels(&got, texel)
                ));
            }
            if alpha.is_empty() {
                continue;
            }
            let bytes: Vec<u8> = reads(&lane, &read, &t, size)
                .chunks(16)
                .map(|v| {
                    let a = f32::from_le_bytes([v[12], v[13], v[14], v[15]]);
                    (a * 255.0).round_ties_even() as u8
                })
                .collect();
            if digest(&bytes) != metal(&format!("{case}.outa")) {
                differs.push(format!(
                    "{case}.outa: the first alpha bytes {:02x?}",
                    &bytes[..8]
                ));
            }
        }
    }
    for line in wrong.iter().chain(&moved) {
        println!("XR-BLENDS-DIFFERS {name}: {line}");
    }
    println!(
        "XR-BLENDS {name}: {checked} blends of {COUNT} texels checked, {} outputs of one and zero \
         and {} of the device's arithmetic differ from Metal",
        wrong.len(),
        moved.len()
    );
    assert!(
        wrong.is_empty(),
        "{} outputs differ from Metal",
        wrong.len()
    );
    quiet(lane, &messages);
}

/// The generator gives the probe's targets: the SHA-256 of each format's texels before the draws
/// (the probe's files).
#[test]
fn the_targets_are_the_probes() {
    for (pf, probe) in FORMATS {
        let form = ExtendedRange::of(pf).expect("an extended-range format");
        let (plane, alpha) = targets(form, u64::from(pf.as_raw()));
        assert_eq!(
            digest(&plane),
            metal(&format!("rt.{probe}.dst0")),
            "{probe}"
        );
        if !alpha.is_empty() {
            assert_eq!(
                digest(&alpha),
                metal(&format!("rt.{probe}.dsta")),
                "{probe}"
            );
        }
    }
}

/// The kind a draw that blends onto an extended-range color an earlier draw of its pass wrote
/// announces; the lane site `degrade_notices` lists.
pub(super) const STORED_CODES: [Degradation; 1] = [Degradation::ExtendedRangeBlendsStoredCodes];

/// A pass of two draws blending (source one, destination one minus the source's alpha) into a
/// two-plane extended-range color announces nothing: the device that blends it (in the pixel
/// interlock) holds the color's `half` shadow from the second draw (`render/extended_shadow.rs`),
/// which blends onto the `half` the first wrote, as Metal's tile memory holds it. A memory
/// barrier between the two splits the pass, and the second draw's instance cannot hold the
/// shadow of what the first instance stored: it blends onto the codes the first stored and
/// announces `extended-range-blends-stored-codes` once, naming the color (the M3 Max's run of the
/// same probes). A pass of one blended draw, and one of two unblended
/// draws, announce none. A device with raster-order attachment access (lavapipe), which blends over
/// the color's input attachment and holds no shadow, announces it at the pass of two draws already,
/// once for the device.
#[test]
fn a_blend_onto_an_earlier_draw_announces_the_stored_codes_only_past_its_shadow() {
    let (messages, _cache, lane) = lane();
    let halves = halves();
    let pf = PixelFormat::RGB10A82PXR10;
    let form = ExtendedRange::of(pf).expect("an extended-range format");
    let (plane, alpha) = targets(form, u64::from(pf.as_raw()));
    let blended = || {
        let mut color = ColorAttachmentDescriptor::new(pf);
        color.blending_enabled = true;
        color.destination_rgb_blend_factor = BlendFactor::OneMinusSourceAlpha;
        color.destination_alpha_blend_factor = BlendFactor::OneMinusSourceAlpha;
        color
    };
    let shadowed = !super::framebuffer_fetch::through_inputs(&lane);
    let t = filled(&lane, pf, (&plane, &alpha));
    if let Err(e) = draws(
        &lane,
        &t,
        &halves,
        ColorAttachmentDescriptor::new(pf),
        (None, 2, false),
    ) {
        panic!("unblended: {e}");
    }
    let counts_unblended = counts(&messages.take_notices());
    assert!(
        !counts_unblended.contains_key(&STORED_CODES[0]),
        "{counts_unblended:?}"
    );
    draws(&lane, &t, &halves, blended(), (None, 1, false))
        .unwrap_or_else(|e| panic!("one blend: {e}"));
    let once = counts(&messages.take_notices());
    assert!(!once.contains_key(&STORED_CODES[0]), "{once:?}");
    draws(&lane, &t, &halves, blended(), (None, 2, false))
        .unwrap_or_else(|e| panic!("two blends: {e}"));
    let two = messages.take_notices();
    draws(&lane, &t, &halves, blended(), (None, 2, true))
        .unwrap_or_else(|e| panic!("two blends a barrier apart: {e}"));
    let split = messages.take_notices();
    // Announced once for the device: at the split pass past the shadow, at the pass of two
    // draws without one.
    let (notices, quiet_pass) = match shadowed {
        true => (split, two),
        false => (two, split),
    };
    assert!(
        !counts(&quiet_pass).contains_key(&STORED_CODES[0]),
        "{quiet_pass:?}"
    );
    assert_eq!(
        counts(&notices).get(&STORED_CODES[0]),
        Some(&1),
        "{notices:?}"
    );
    let named = notices
        .iter()
        .filter_map(|m| m.degradation.as_ref())
        .find(|n| n.degradation == STORED_CODES[0])
        .map(|n| n.object.clone());
    assert!(
        named
            .as_deref()
            .is_some_and(|o| o.contains("color 0, RGB10A82PXR10")),
        "{named:?}"
    );
}

/// A blended draw onto a two-plane extended-range color its pass clears, where the clear's
/// alpha, 0.5, lies between two of the alpha plane's codes (Metal's tile memory holds the
/// clear's half 0.5, the image the byte 128, which reads 128/255:
/// `ExtendedRange::clear_read_back`), announces nothing: the device that blends it (in the pixel
/// interlock) holds the color's `half` shadow from that draw, set to the clear's halves
/// (`render/extended_shadow.rs`). A second draw a memory barrier later, in an instance of its
/// own, still holds it: the shadow, once started, lasts through the pass's splits. A pass that
/// clears to zero, every channel a code's value, announces none. A device with raster-order
/// attachment access (lavapipe), which blends over the color's input attachment and holds no
/// shadow, announces `extended-range-blends-stored-codes` at the first pass between two codes, once
/// for the device.
#[test]
fn a_blend_onto_a_clear_between_two_codes_reads_the_clears_half() {
    let (messages, _cache, lane) = lane();
    let halves = halves();
    let pf = PixelFormat::RGB10A82PXR10;
    let form = ExtendedRange::of(pf).expect("an extended-range format");
    let (plane, alpha) = targets(form, u64::from(pf.as_raw()));
    let mut blended = ColorAttachmentDescriptor::new(pf);
    blended.blending_enabled = true;
    blended.destination_rgb_blend_factor = BlendFactor::OneMinusSourceAlpha;
    blended.destination_alpha_blend_factor = BlendFactor::OneMinusSourceAlpha;
    let t = filled(&lane, pf, (&plane, &alpha));
    let zero = Some(ClearColor::new(0.0, 0.0, 0.0, 0.0));
    draws(&lane, &t, &halves, blended, (zero, 1, false))
        .unwrap_or_else(|e| panic!("the pipeline: {e}"));
    let cleared = counts(&messages.take_notices());
    assert!(!cleared.contains_key(&STORED_CODES[0]), "{cleared:?}");
    let shadowed = !super::framebuffer_fetch::through_inputs(&lane);
    let between = Some(ClearColor::new(0.0, 0.0, 0.0, 0.5));
    for (count, split) in [(1, false), (2, true)] {
        draws(&lane, &t, &halves, blended, (between, count, split))
            .unwrap_or_else(|e| panic!("between two codes, {count} draws: {e}"));
        let notices = counts(&messages.take_notices());
        let announced = !shadowed && count == 1;
        assert_eq!(
            notices.get(&STORED_CODES[0]).copied(),
            announced.then_some(1),
            "{count} draws: {notices:?}"
        );
    }
}

/// A pipeline blending (premultiplied over) an output into a two-plane extended-range color
/// announces its blend's note once, whatever the output computes: the function blends in the
/// shader in the device's arithmetic, where Metal's compiler rounds each step to `half`. The
/// `float4` output (`rt6_samples`' `rt6_fill_rgba32float`) announces
/// `extended-range-float-blend-source`, and the `half4` one (`rt6_fill_rgba16float`)
/// `extended-range-blend-source`, each never the other's; the same color unblended announces
/// neither. A device with raster-order attachment access (lavapipe) blends over the color's input
/// attachment and announces alike.
#[test]
fn a_blend_into_extended_range_announces_its_source_note() {
    let (messages, _cache, lane) = lane();
    let pf = PixelFormat::RGB10A82PXR10;
    let source = Degradation::ExtendedRangeBlendSource;
    let float_source = Degradation::ExtendedRangeFloatBlendSource;
    let fills = library(&lane.device).unwrap();
    let function = |name: &str| {
        StageSource::Function(
            fills
                .new_function_with_name(name)
                .unwrap()
                .function_descriptor(),
        )
    };
    let made = |name: &str, blended: bool| {
        let mut pipeline = RenderPipelineDescriptor::new(function("rt6_fill_vertex"));
        pipeline.fragment = Some(function(name));
        let mut color = ColorAttachmentDescriptor::new(pf);
        color.blending_enabled = blended;
        color.destination_rgb_blend_factor = BlendFactor::OneMinusSourceAlpha;
        color.destination_alpha_blend_factor = BlendFactor::OneMinusSourceAlpha;
        pipeline.color_attachments[0] = Some(color);
        lane.device
            .new_render_pipeline_state(&pipeline, &PipelineOptions::default())
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        counts(&messages.take_notices())
    };
    for name in ["rt6_fill_rgba32float", "rt6_fill_rgba16float"] {
        let notices = made(name, false);
        assert!(!notices.contains_key(&source), "{name}: {notices:?}");
        assert!(!notices.contains_key(&float_source), "{name}: {notices:?}");
    }
    let notices = made("rt6_fill_rgba32float", true);
    assert_eq!(notices.get(&float_source), Some(&1), "{notices:?}");
    assert_eq!(notices.get(&source), None, "{notices:?}");
    let notices = made("rt6_fill_rgba16float", true);
    assert_eq!(notices.get(&source), Some(&1), "{notices:?}");
    assert_eq!(notices.get(&float_source), None, "{notices:?}");
}

/// The kind a draw in a pass that loads an extended-range, padded or bfloat color announces.
pub(super) const EXTENDED_RANGE_BITS: [Degradation; 1] = [Degradation::ExtendedRangeBitsKept];

/// A draw into `BGR10_XR` holding the probe's random texels under a write mask of none of its
/// channels: every texel keeps its bytes, the device's store (`render/tile_store.rs`), where
/// Metal's pass leaves each tile a draw reaches with its bits 30 and 31 ones (on an Apple M4 and
/// an Apple M3 Max; the full-screen triangle reaches every tile); `extended-range-bits-kept` is
/// announced once. A draw under every channel, whose texels the draw writes, announces it no
/// more (once per device).
#[test]
fn a_draw_under_an_empty_mask_keeps_the_unused_bits_and_says_so() {
    let (messages, _cache, lane) = lane();
    let halves = halves();
    let pf = PixelFormat::BGR10XR;
    let form = ExtendedRange::of(pf).expect("an extended-range format");
    let (plane, alpha) = targets(form, u64::from(pf.as_raw()));
    let t = filled(&lane, pf, (&plane, &alpha));
    let mut none = ColorAttachmentDescriptor::new(pf);
    none.write_mask = ColorWriteMask::NONE;
    let first = made(&lane, &[draw_descriptor(&lane, none)]);
    draw(&lane, &t, &halves, none).expect("no channel");
    let got = lane.read(&t, (0, 0), 4, BlitOption::None);
    let canonical = plane
        .chunks(4)
        .filter(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]) >> 30 != 3)
        .count();
    println!("EXTENDED-RANGE-BITS-KEPT {canonical} texels Metal's pass would rewrite");
    assert!(
        canonical > 0,
        "the probe's texels hold no bits Metal rewrites"
    );
    assert!(
        got == plane,
        "the texels differ from their bytes: {}",
        first_texels(&got, 4)
    );
    assert_made_once(&messages, &EXTENDED_RANGE_BITS, &first);
    let t = filled(&lane, pf, (&plane, &alpha));
    let every = ColorAttachmentDescriptor::new(pf);
    let first = made(&lane, &[draw_descriptor(&lane, every)]);
    draw(&lane, &t, &halves, every).expect("every channel");
    let counts = assert_made_once(&messages, &[], &first);
    assert!(!counts.contains_key(&EXTENDED_RANGE_BITS[0]), "{counts:?}");
    quiet(lane, &messages);
}

/// A pass that loads a 4x multisampled `BGR10_XR` attachment and draws announces
/// `extended-range-bits-kept` once, naming color 0: the attachment keeps the bits its draws do not
/// write, which Metal's pass rewrites in each tile a draw reaches.
#[test]
fn a_multisampled_extended_range_color_keeps_its_unused_bits_and_says_so() {
    let (messages, _cache, lane) = lane();
    let pf = PixelFormat::BGR10XR;
    let msaa = lane.msaa(pf);
    let pipeline = lane.pipeline((FULL, WHITE), &[pf], (None, None), 4);
    let mut pass = RenderPassDescriptor::new();
    let attachment = &mut pass.color_attachments[0];
    attachment.texture = Some(msaa.clone());
    attachment.load_action = LoadAction::Load;
    attachment.store_action = StoreAction::Store;
    lane.run(&pass, |l, e, h| l.draw(e, h, &pipeline, 0.5))
        .unwrap();
    let notices = messages.take_notices();
    for message in &notices {
        let notice = message.degradation.as_ref().unwrap();
        if notice.degradation == EXTENDED_RANGE_BITS[0] {
            assert!(notice.object.contains("color 0"), "{}", message.text);
        }
    }
    // `lane.pipeline`'s descriptor: `FULL` and `WHITE` into the 4x color.
    let function = |name| {
        StageSource::Function(
            lane.library
                .new_function_with_name(name)
                .unwrap()
                .function_descriptor(),
        )
    };
    let mut descriptor = RenderPipelineDescriptor::new(function(FULL));
    descriptor.fragment = Some(function(WHITE));
    descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(pf));
    descriptor.raster_sample_count = 4;
    messages.restore_notices(notices);
    assert_made_once(&messages, &EXTENDED_RANGE_BITS, &made(&lane, &[descriptor]));
    quiet(lane, &messages);
}

/// A pass of a `BGR10_XR` color 0, an `R10Unorm_X6` color 1 and an `RG10Unorm_X12` color 2 (over
/// buffers of words of all ones), every one loaded, of a state that writes white to color 0
/// under a write mask of none of its channels, on a device that asks for the kinds at every
/// occurrence (`DeviceOptions::announce_each`), with a draw and an indirect command buffer's
/// execution of a draw: `extended-range-bits-kept` is announced, naming the three loaded colors,
/// and the padded colors, which the function does not write, keep their bytes, the device's store
/// (Metal's pass leaves each word `0xffc0`, its low six bits zeroed). A pass that clears and
/// stores the padded colors names color 0 alone; one that draws nothing announces nothing.
#[test]
fn a_pass_of_extended_range_and_padded_colors_names_each_loaded_color() {
    let kinds = [EXTENDED_RANGE_BITS[0]];
    let (messages, _cache, lane) =
        lane_with(|options| options.announce_each = kinds.into_iter().collect());
    let (w, h) = (WIDTH as u64, HEIGHT as u64);
    let xr = PixelFormat::BGR10XR;
    let padded = [PixelFormat::R10UnormX6, PixelFormat::RG10UnormX12];
    let bpr = 4 * w;
    let buffers: Vec<Buffer> = padded
        .iter()
        .map(|_| buffer_with(&lane.device, &vec![0xff; (bpr * h) as usize]).unwrap())
        .collect();
    let over = |pf: PixelFormat, buffer: &Buffer| {
        let mut descriptor = TextureDescriptor::texture_2d(pf, w, h, false);
        descriptor.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
        descriptor.resource_options = ResourceOptions::shared();
        buffer
            .new_texture(&descriptor, 0, bpr)
            .unwrap_or_else(|e| panic!("{pf:?} over a buffer: {e}"))
    };
    let targets = [
        target(&lane, xr),
        over(padded[0], &buffers[0]),
        over(padded[1], &buffers[1]),
    ];
    let function = |name: &str| {
        StageSource::Function(
            lane.library
                .new_function_with_name(name)
                .unwrap()
                .function_descriptor(),
        )
    };
    let mut descriptor = RenderPipelineDescriptor::new(function("rt6_resolve_full"));
    descriptor.fragment = Some(function("rt6_resolve_white"));
    let mut none = ColorAttachmentDescriptor::new(xr);
    none.write_mask = ColorWriteMask::NONE;
    descriptor.color_attachments[0] = Some(none);
    for (index, pf) in padded.into_iter().enumerate() {
        descriptor.color_attachments[index + 1] = Some(ColorAttachmentDescriptor::new(pf));
    }
    descriptor.support_indirect_command_buffers = true;
    let pipeline = lane
        .device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap();
    let mut icb = IndirectCommandBufferDescriptor::new();
    icb.command_types = IndirectCommandType::DRAW;
    icb.inherit_pipeline_state = true;
    icb.inherit_buffers = true;
    let icb = lane
        .device
        .new_indirect_command_buffer(&icb, 1, ResourceOptions::shared())
        .unwrap();
    icb.indirect_render_command_at(0)
        .unwrap()
        .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
        .unwrap();
    // A pass of the targets, the padded ones loaded or cleared.
    let pass = |padded_load: LoadAction| {
        let mut pass = RenderPassDescriptor::new();
        for (index, texture) in targets.iter().enumerate() {
            let attachment = &mut pass.color_attachments[index];
            attachment.texture = Some(texture.clone());
            attachment.load_action = match index {
                0 => LoadAction::Load,
                _ => padded_load,
            };
            attachment.store_action = StoreAction::Store;
        }
        pass
    };
    let depth = buffer_with(&lane.device, &0.5f32.to_le_bytes()).unwrap();
    // The passes' draws: a draw, then an execution of the buffer's draw.
    let drawn = |load: LoadAction, draws: bool| {
        lane.run(&pass(load), |_, e, _| {
            e.set_render_pipeline_state(&pipeline).unwrap();
            e.set_vertex_buffer(Some(&depth), 0, 0).unwrap();
            if draws {
                e.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
                    .unwrap();
                e.execute_commands_in_buffer(&icb, 0..1).unwrap();
            }
        })
        .unwrap();
        messages.take_notices()
    };
    // The colors each announcement names, of the passes' notices.
    let named = |notices: &[DebugMessage]| -> Vec<bool> {
        let colors = ["color 0", "color 1", "color 2"];
        let mut named = vec![false; colors.len()];
        for message in notices {
            let notice = message.degradation.as_ref().unwrap();
            if notice.degradation != EXTENDED_RANGE_BITS[0] {
                continue;
            }
            for (seen, color) in named.iter_mut().zip(colors) {
                *seen |= notice.object.contains(color);
            }
        }
        named
    };
    let notices = drawn(LoadAction::Load, true);
    assert!(
        counts(&notices).contains_key(&EXTENDED_RANGE_BITS[0]),
        "{:?}",
        counts(&notices)
    );
    assert_eq!(named(&notices), [true, true, true]);
    for (pf, buffer) in padded.iter().zip(&buffers) {
        let bytes = read_buffer(buffer, 0, bpr * h).unwrap();
        for (y, line) in bytes.chunks(bpr as usize).enumerate() {
            assert!(
                line.iter().all(|&b| b == 0xff),
                "{pf:?} row {y}: {:02x?}",
                &line[..8]
            );
        }
    }
    let notices = drawn(LoadAction::Clear, true);
    assert!(!notices.is_empty(), "a pass of a loaded BGR10_XR color");
    assert_eq!(named(&notices), [true, false, false]);
    let announced = counts(&drawn(LoadAction::Load, false));
    assert!(announced.is_empty(), "{announced:?}");
    quiet(lane, &messages);
}

/// `n / 510` correctly rounded to `f32` (`|n| <= 639`): the `f64` quotient's neighbours compared
/// exactly (each `f32` times 510 is exact in `f64`).
fn ratio(n: i32) -> f32 {
    let q = f64::from(n) / 510.0;
    let y = q as f32;
    let err = |c: f32| (f64::from(c) * 510.0 - f64::from(n)).abs();
    [
        y,
        f32::from_bits(y.to_bits().wrapping_sub(1)),
        f32::from_bits(y.to_bits() + 1),
    ]
    .into_iter()
    .filter(|c| c.is_finite())
    .min_by(|a, b| err(*a).total_cmp(&err(*b)))
    .map_or(0.0, |c| if n == 0 { 0.0 } else { c })
}

/// `x` (exact in `f64`, below the largest `half`) rounded to `half`, ties away from zero, as an
/// `f32`.
fn half_away(x: f64) -> f32 {
    if x == 0.0 {
        return x as f32;
    }
    let e = ((x.abs().to_bits() >> 52) & 0x7FF) as i32 - 1023;
    let ulp = 2f64.powi(e.max(-14) - 10);
    ((x / ulp).round() * ulp) as f32
}

/// Metal's tile value of an `RGB10A8_2P_XR10` texel of color word `word` and alpha byte `alpha`:
/// each color code clamped to 1022, `(code - 384) / 510` correctly rounded, alpha `alpha / 255`,
/// each rounded to `half` with ties away from zero.
pub(super) fn tile_of(word: u32, alpha: u8) -> [f32; 4] {
    let color = |shift: u32| ratio(((word >> shift) & 0x3FF).min(1022) as i32 - 384);
    let a = (f64::from(alpha) / 255.0) as f32;
    [color(20), color(10), color(0), a].map(|v| half_away(f64::from(v)))
}

/// `x` rounded to `half`, nearest even, as an `f32`.
fn half_even(x: f32) -> f32 {
    half_value(half_bits(x))
}

/// Metal's read of a linear `RGB10A8_2P_XR10` tile holding the halves `held`: each color lane
/// clamped to `[-1542/2048, 1281/1024]`, alpha to `[0, 1]`, a NaN to the low end
/// (`lower::arith::extended_range::tile_read`).
fn tile_read(held: [f32; 4]) -> [f32; 4] {
    let (lo, hi) = (-1542.0 / 2048.0, 1281.0 / 1024.0);
    std::array::from_fn(|k| {
        let (lo, hi) = if k == 3 { (0.0, 1.0) } else { (lo, hi) };
        match held[k].is_nan() {
            true => lo,
            false => held[k].clamp(lo, hi),
        }
    })
}

/// A fetched `RGB10A8_2P_XR10` attachment written back under every channel with blending
/// disabled, in the shape of a guest's pipeline: `exf_fetch_blend` reads color 0 and returns
/// `(source + destination) / 2`, its source `(x, y, 1, -0.5)` of the pixel's center, beside an
/// `RG16Float` color 1 and a second `RGB10A8_2P_XR10` attachment, both of mask none, and a
/// `Depth32Float_Stencil8` attachment. Color 0 holds the probe's random texels (`targets`), and
/// three full-screen draws run in one pass, each over what the one before it wrote. Each texel ends
/// as Metal's rule gives it: the first draw reads the tile value (`tile_of`), each computes the
/// function's `float` arithmetic and Metal's unblended write of the `float` output, rounded to
/// `half` to nearest even, which the tile holds unclamped; the next draw reads that `half` clamped
/// to the source range and alpha to `[0, 1]` (`tile_read`; where the earlier model,
/// each draw reading the codes the one before stored, differed from Metal at 3, 558 of 4, 096
/// texels of the probe's own such pass), and the store writes the last `half`'s encode as a
/// kernel's write (`ExtendedRange::clear_texel`); colors 1 and 2 keep their bytes, the device's
/// store, where Metal's pass zeroes color 2's color words' top two bits (`extended-range-bits-kept`
/// says so, `render/tile_store.rs`). The pipeline is made on every device:
/// one that fetches through the interlock writes the storage form and holds color 0's `half` shadow
/// from the second draw on (`render/extended_shadow.rs`), one with raster-order attachment access
/// reads an input attachment of the raw texels and writes the raw output, each draw reading the
/// codes the one before stored, which the runtime announces (`extended-range-blends-stored-codes`).
#[test]
fn a_fetched_two_plane_attachment_written_unblended_stores_metals_write() {
    use crate::air_library::AirLibrary;
    use metal2vulkan_devtools::{engine_dir, llvm_assemble};
    let (messages, _cache, lane) = lane();
    let name = lane.device.name().to_owned();
    let pf = PixelFormat::RGB10A82PXR10;
    let form = ExtendedRange::of(pf).expect("an extended-range format");
    let (plane, alpha) = targets(form, u64::from(pf.as_raw()));
    let samples = library(&lane.device).unwrap();
    let bitcode = llvm_assemble(&engine_dir().join("exf_fetch_blend.ll")).unwrap();
    let fetching = lane.device.new_library_with_air(&bitcode).unwrap();
    let function = |library: &metal2vulkan_runtime::Library, name: &str| {
        StageSource::Function(
            library
                .new_function_with_name(name)
                .unwrap()
                .function_descriptor(),
        )
    };
    let mut pipeline = RenderPipelineDescriptor::new(function(&samples, "rt6_fill_vertex"));
    pipeline.fragment = Some(function(&fetching, "exf_fetch_blend"));
    let mut covered = ColorAttachmentDescriptor::new(PixelFormat::RG16Float);
    covered.write_mask = ColorWriteMask::NONE;
    let mut unwritten = ColorAttachmentDescriptor::new(pf);
    unwritten.write_mask = ColorWriteMask::NONE;
    pipeline.color_attachments[0] = Some(ColorAttachmentDescriptor::new(pf));
    pipeline.color_attachments[1] = Some(covered);
    pipeline.color_attachments[2] = Some(unwritten);
    pipeline.depth_format = Some(PixelFormat::Depth32FloatStencil8);
    pipeline.stencil_format = Some(PixelFormat::Depth32FloatStencil8);
    let state = lane
        .device
        .new_render_pipeline_state(&pipeline, &PipelineOptions::default())
        .unwrap_or_else(|e| panic!("the guest's pipeline on {name}: {e}"));
    let through = match super::framebuffer_fetch::through_inputs(&lane) {
        true => "input attachments",
        false => "the pixel interlock",
    };
    let t = filled(&lane, pf, (&plane, &alpha));
    let (other_plane, other_alpha) = targets(form, 551);
    let other = filled(&lane, pf, (&other_plane, &other_alpha));
    let covered_texels: Vec<u8> = (0..COUNT)
        .flat_map(|i| [half_bits(i as f32 / 64.0), half_bits(-0.25)])
        .flat_map(u16::to_le_bytes)
        .collect();
    let covered = target(&lane, PixelFormat::RG16Float);
    covered
        .replace_region(
            Region::new_2d(0, 0, WIDTH as u64, HEIGHT as u64),
            0,
            0,
            &covered_texels,
            4 * WIDTH as u64,
            0,
        )
        .unwrap();
    let mut depth = TextureDescriptor::texture_2d(
        PixelFormat::Depth32FloatStencil8,
        WIDTH as u64,
        HEIGHT as u64,
        false,
    );
    depth.usage = TextureUsage::RENDER_TARGET;
    depth.resource_options = ResourceOptions::private();
    let depth = lane.device.new_texture(&depth).unwrap();
    let mut pass = RenderPassDescriptor::new();
    for (index, texture) in [&t, &covered, &other].into_iter().enumerate() {
        let attachment = &mut pass.color_attachments[index];
        attachment.texture = Some(texture.clone());
        attachment.load_action = LoadAction::Load;
        attachment.store_action = StoreAction::Store;
    }
    pass.depth_attachment.texture = Some(depth.clone());
    pass.depth_attachment.load_action = LoadAction::Clear;
    pass.depth_attachment.clear_depth = 1.0;
    pass.depth_attachment.store_action = StoreAction::DontCare;
    pass.stencil_attachment.texture = Some(depth.clone());
    pass.stencil_attachment.load_action = LoadAction::Clear;
    pass.stencil_attachment.store_action = StoreAction::DontCare;
    lane.run(&pass, |_, e, _| {
        e.set_render_pipeline_state(&state).unwrap();
        for _ in 0..3 {
            e.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
                .unwrap();
        }
    })
    .unwrap_or_else(|e| panic!("the pass on {name}: {e}"));
    // Metal's rule, texel by texel: through the shadow in the interlock, else through the codes.
    let shadowed = !super::framebuffer_fetch::through_inputs(&lane);
    let mut want_plane = Vec::with_capacity(4 * COUNT);
    let mut want_alpha = Vec::with_capacity(COUNT);
    for (i, (word, &a)) in plane.chunks(4).zip(&alpha).enumerate() {
        let (x, y) = ((i % WIDTH) as f32 + 0.5, (i / WIDTH) as f32 + 0.5);
        let word = u32::from_le_bytes([word[0], word[1], word[2], word[3]]);
        let mut texel = [word, u32::from(a)];
        let mut d = tile_of(word, a);
        for _ in 0..3 {
            let source = [x, y, 1.0, -0.5];
            let out: [f64; 4] = std::array::from_fn(|k| f64::from((source[k] + d[k]) * 0.5));
            let raw = form.clear_texel(out, ClearRoute::ThroughHalf);
            texel = [raw[0], raw[1]];
            d = match shadowed {
                true => tile_read(out.map(|v| half_even(v as f32))),
                false => tile_of(texel[0], texel[1] as u8),
            };
        }
        want_plane.extend(texel[0].to_le_bytes());
        want_alpha.push(texel[1] as u8);
    }
    let got = lane.read(&t, (0, 0), 4, BlitOption::None);
    let read = kernel(&lane, "xr2p_read_float");
    let size = (WIDTH as u64, HEIGHT as u64);
    let alpha_of = |texture: &Texture| -> Vec<u8> {
        reads(&lane, &read, texture, size)
            .chunks(16)
            .map(|v| {
                let a = f32::from_le_bytes([v[12], v[13], v[14], v[15]]);
                (a * 255.0).round_ties_even() as u8
            })
            .collect()
    };
    let got_alpha = alpha_of(&t);
    let words = |bytes: &[u8]| -> Vec<u32> {
        bytes
            .chunks(4)
            .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
            .collect()
    };
    let (got_words, want_words) = (words(&got), words(&want_plane));
    let differ: Vec<String> = (0..COUNT)
        .filter(|&i| got_words[i] != want_words[i] || got_alpha[i] != want_alpha[i])
        .take(4)
        .map(|i| {
            format!(
                "texel {i}: {:08x}/{:02x}, want {:08x}/{:02x}",
                got_words[i], got_alpha[i], want_words[i], want_alpha[i]
            )
        })
        .collect();
    let count = (0..COUNT)
        .filter(|&i| got_words[i] != want_words[i] || got_alpha[i] != want_alpha[i])
        .count();
    // Color 2, loaded and unwritten, keeps its bytes, the device's store, where Metal's pass
    // zeroes its color words' top two bits in each tile a draw reaches.
    let kept = lane.read(&other, (0, 0), 4, BlitOption::None) == other_plane
        && alpha_of(&other) == other_alpha
        && lane.read(&covered, (0, 0), 4, BlitOption::None) == covered_texels;
    println!(
        "XR-FETCH-UNBLENDED {name}: made, through {through}; {count} of {COUNT} texels differ \
         from Metal's rule{}; colors 1 and 2 {}",
        if differ.is_empty() {
            String::new()
        } else {
            format!(" ({})", differ.join(", "))
        },
        if kept { "kept" } else { "CHANGED" }
    );
    assert_eq!(count, 0, "texels differ from Metal's rule on {name}");
    assert!(kept, "colors 1 and 2 changed on {name}");
    quiet(lane, &messages);
}
