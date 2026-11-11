//! A blend into a padded 10-bit attachment (`R10Unorm_X6`, `RG10Unorm_X12`, their `_sRGB` twins
//! and `RGB10A2Unorm_sRGB`), which Metal makes and draws (the M3 Max under macOS 27.0.1:
//! `probes/padded_and_bfloat_blends.m` of the conformance crate), and which no device blends: the
//! attachment holds the raw words in an unsigned integer format.
//!
//! The function computes the blend over an ordered read of the attachment, as an extended-range
//! form's (`lower::fragment::fetch`, `lower::fragment::input_blend`). On every committed profile
//! that orders fetch through the pixel interlock alone, `rt6_samples`' `half4` and `float4` fills
//! compile with color 0 in the storage form under premultiplied source-over and one plus one, the
//! fact naming the blend, and pass L3; so do `xrb_dual`'s second-source blends, whose index-1
//! output the blend reads, and a write mask of some of the channels. On every one that orders it
//! through rasterization-order input attachments the whole pipeline is made with color 0's blend
//! computed over its input attachment (`RuntimeFact::ExtendedRangeInputBlend`), the attachment a
//! color attachment of the raw format with no Vulkan blend. On one that orders it in neither form
//! the pipeline is refused naming the two orderings. Before the blend was built every one of these
//! was refused `NotImplemented(Format)`. No GPU is used.
//!
//! `llvm-as` assembles the fixtures; a missing tool skips visibly.

use std::path::Path;

use crate::budget::Budget;
use crate::device::{Device, DeviceDescription, DevicePolicy, DeviceProfile};
use crate::error::{DescriptorError, Error, MetalAnswer, Needs};
use crate::library::{FunctionArtifact, Library};
use crate::metal::{BlendFactor, BlendOperation, ColorWriteMask, PixelFormat};
use crate::pipeline::artifact::{PrecisionNote, RenderPipeline};
use crate::pipeline::descriptor::{
    ColorAttachmentDescriptor, FunctionDescriptor, RenderPipelineDescriptor, StageSource,
};
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::link::RenderLinkInputs;
use crate::pipeline::lower_inputs::{BlendEquation, FramebufferFetchInputs, LoweringInputs};
use crate::pipeline::testing::{
    compile_with, engine_dir, fragment_inputs, link_and_validate, AssembleError,
};
use crate::spirv::testing::committed_profiles;
use crate::vk;

/// Fixture `stem` assembled; `None` (skipped) without `llvm-as`.
fn assembled(stem: &str) -> Option<Vec<u8>> {
    let text =
        std::fs::read_to_string(engine_dir().join(format!("{stem}.ll"))).expect("the fixture");
    match metal2vulkan_devtools::llvm_assemble_text(&text) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {stem}: {why}"),
    }
}

/// The padded 10-bit formats.
const PADDED: [PixelFormat; 5] = [
    PixelFormat::R10UnormX6,
    PixelFormat::R10UnormX6SRGB,
    PixelFormat::RG10UnormX12,
    PixelFormat::RG10UnormX12SRGB,
    PixelFormat::RGB10A2UnormSRGB,
];

/// The bfloat formats, which Metal blends nowhere.
const BFLOAT: [PixelFormat; 3] = [
    PixelFormat::R16Bfloat,
    PixelFormat::RG16Bfloat,
    PixelFormat::RGBA16Bfloat,
];

/// Premultiplied source-over: one over one minus the source alpha, added, for color and alpha.
const OVER: BlendEquation = BlendEquation {
    source_rgb: BlendFactor::One,
    destination_rgb: BlendFactor::OneMinusSourceAlpha,
    rgb_operation: BlendOperation::Add,
    source_alpha: BlendFactor::One,
    destination_alpha: BlendFactor::OneMinusSourceAlpha,
    alpha_operation: BlendOperation::Add,
};

/// One plus one, added, for color and alpha.
const ADD: BlendEquation = BlendEquation {
    source_rgb: BlendFactor::One,
    destination_rgb: BlendFactor::One,
    rgb_operation: BlendOperation::Add,
    source_alpha: BlendFactor::One,
    destination_alpha: BlendFactor::One,
    alpha_operation: BlendOperation::Add,
};

/// The second source's blend: its color and one minus it, its alpha and one minus it.
const SECOND: BlendEquation = BlendEquation {
    source_rgb: BlendFactor::Source1Color,
    destination_rgb: BlendFactor::OneMinusSource1Color,
    rgb_operation: BlendOperation::Add,
    source_alpha: BlendFactor::Source1Alpha,
    destination_alpha: BlendFactor::OneMinusSource1Alpha,
    alpha_operation: BlendOperation::Add,
};

/// Color 0 of `pf` blended by `eq` under write mask `mask`.
fn attachment(
    pf: PixelFormat,
    eq: BlendEquation,
    mask: ColorWriteMask,
) -> ColorAttachmentDescriptor {
    let mut c = ColorAttachmentDescriptor::new(pf);
    c.blending_enabled = true;
    c.source_rgb_blend_factor = eq.source_rgb;
    c.destination_rgb_blend_factor = eq.destination_rgb;
    c.rgb_blend_operation = eq.rgb_operation;
    c.source_alpha_blend_factor = eq.source_alpha;
    c.destination_alpha_blend_factor = eq.destination_alpha;
    c.alpha_blend_operation = eq.alpha_operation;
    c.write_mask = mask;
    assert_eq!(BlendEquation::of(&c), Some(eq));
    c
}

/// The lowering inputs of a fragment function writing color 0 of `pf` blended by `eq` under
/// `mask` on `p`, as the render pipeline compile sets them.
fn state(
    p: &DeviceProfile,
    pf: PixelFormat,
    eq: BlendEquation,
    mask: ColorWriteMask,
) -> LoweringInputs {
    let mut r = RenderLinkInputs::new();
    r.color_attachments[0] = Some(attachment(pf, eq, mask));
    let mut inputs = fragment_inputs(&r, p);
    inputs.framebuffer_fetch = Some(FramebufferFetchInputs::of(
        [0],
        &r.color_attachments,
        false,
        false,
        1,
    ));
    inputs
}

/// Color 0's storage form in `a`: whether it writes, and its blend.
fn fetched(a: &FunctionArtifact) -> Option<(bool, Option<BlendEquation>)> {
    a.facts.iter().find_map(|f| match f {
        RuntimeFact::InterlockedFramebufferFetch {
            color: 0,
            writes,
            blend,
            ..
        } => Some((*writes, *blend)),
        _ => None,
    })
}

/// The device of committed profile `stem`.
fn device(stem: &str) -> Device {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles")
        .join(format!("{stem}.json"));
    let text = std::fs::read_to_string(path).expect("profile");
    let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    Device::new(desc, DevicePolicy::default()).expect("device")
}

/// Whether `e` is a refusal the device's lack of a capability makes, whatever the blend (a
/// raw format without the storage access the form needs).
fn device_gap(e: &Error) -> bool {
    matches!(e, Error::Unsupported(u) if u.needs != Needs::NotImplemented)
}

/// The pipeline of `bc`'s `vertex` and `fragment` with color 0 `c`, on `dev`.
fn pipeline(
    dev: &Device,
    (bc, vertex, fragment): (&[u8], &str, &str),
    c: ColorAttachmentDescriptor,
) -> Result<RenderPipeline, Error> {
    let function = |name: &str| {
        FunctionDescriptor::new(
            Library::from_air(bc)
                .expect("library")
                .function(name)
                .expect("function"),
        )
    };
    let mut d = RenderPipelineDescriptor::new(StageSource::Function(function(vertex)));
    d.fragment = Some(StageSource::Function(function(fragment)));
    d.color_attachments[0] = Some(c);
    dev.compile_render_pipeline(&d, &Budget::standard())
}

/// The blends a pipeline computes over color 0's input attachment, with the channels each keeps.
fn input_blends(a: &RenderPipeline) -> Vec<(BlendEquation, vk::ColorComponentFlags)> {
    a.dependencies
        .iter()
        .filter_map(|d| match d.fact {
            RuntimeFact::ExtendedRangeInputBlend {
                color: 0,
                blend,
                kept,
            } => Some((blend, kept)),
            _ => None,
        })
        .collect()
}

#[test]
fn a_blend_into_a_padded_attachment_is_computed_in_the_shader() {
    let (Some(fills), Some(dual)) = (assembled("rt6_samples"), assembled("xrb_dual")) else {
        return;
    };
    // Each function, its vertex function, and the blends it takes.
    let entries: [(&[u8], &str, &str, &[BlendEquation]); 4] = [
        (
            &fills,
            "rt6_fill_vertex",
            "rt6_fill_rgba16float",
            &[OVER, ADD],
        ),
        (
            &fills,
            "rt6_fill_vertex",
            "rt6_fill_rgba32float",
            &[OVER, ADD],
        ),
        (&dual, "xrb_dual_vertex", "xrb_dual_half", &[SECOND]),
        (&dual, "xrb_dual_vertex", "xrb_dual_float", &[SECOND]),
    ];
    // The masks of some of a format's channels: red alone, and the color channels alone.
    let rgb = ColorWriteMask::RED
        .union(ColorWriteMask::GREEN)
        .union(ColorWriteMask::BLUE);
    let (mut interlocked, mut inputs, mut unordered) = (0, 0, 0);
    for (stem, p) in committed_profiles().expect("profiles") {
        let dev = device(&stem);
        for pf in PADDED {
            for &(bc, vertex, fragment, blends) in &entries {
                for &eq in blends {
                    for mask in [ColorWriteMask::ALL, ColorWriteMask::RED, rgb] {
                        let what = format!("{stem} {pf:?} {fragment} {eq:?} {mask:?}");
                        let made = pipeline(&dev, (bc, vertex, fragment), attachment(pf, eq, mask));
                        if crate::lower::fragment::interlock_orders(&p) {
                            let a = match compile_with(bc, fragment, &p, &state(&p, pf, eq, mask)) {
                                Ok(a) => a,
                                Err(e) if device_gap(&e) => {
                                    eprintln!("{what}: {e}");
                                    continue;
                                }
                                Err(e) => panic!("{what}: {e}"),
                            };
                            assert_eq!(
                                fetched(&a),
                                Some((true, Some(eq))),
                                "{what}: {:?}",
                                a.facts
                            );
                            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
                            let made = made.unwrap_or_else(|e| panic!("{what}: the pipeline: {e}"));
                            assert!(
                                !made.graphics.storage_attachments.is_empty(),
                                "{what}: no storage attachment"
                            );
                            interlocked += 1;
                        } else if crate::lower::fragment::inputs_order(&p) {
                            let a = made.unwrap_or_else(|e| panic!("{what}: {e}"));
                            let blends = input_blends(&a);
                            assert_eq!(blends.len(), 1, "{what}: {blends:?}");
                            assert_eq!(blends[0].0, eq, "{what}");
                            let state0 = a.graphics.color_attachments[0].expect("color 0");
                            let form = crate::metal::format::ExtendedRange::of(pf).expect("padded");
                            assert_eq!(state0.format, form.raw_format(), "{what}");
                            assert!(state0.blend.is_none(), "{what}");
                            inputs += 1;
                        } else {
                            let e = made.map(|_| ()).expect_err(&what);
                            let named = crate::lower::fragment::fetch::unordered_extended_fetch();
                            assert!(
                                matches!(&e, Error::Unsupported(u) if u.needs == named.needs),
                                "{what}: {e}"
                            );
                            unordered += 1;
                        }
                    }
                }
            }
        }
    }
    assert!(interlocked > 0, "no profile blends in the storage form");
    assert!(inputs > 0, "no profile blends over the input attachment");
    assert!(unordered > 0, "no profile orders fetch in neither form");
}

/// A blend into a bfloat attachment (`R16Bfloat`, `RG16Bfloat`, `RGBA16Bfloat`) is Metal's
/// pipeline error, whatever the function writes there: Metal's pipeline validation asserts that
/// the format is not blendable (the M3 Max under macOS 27.0.1, plain and under the validation
/// layer, with a fragment function writing the attachment, under a mask of no channel, and with no
/// fragment function). So the pipeline is `InvalidDescriptor(ColorAttachment)` on every committed
/// profile, as is the fragment function's translation under that state; the same pipeline
/// unblended is made where the device makes the attachment (the negative control).
#[test]
fn a_blend_into_a_bfloat_attachment_is_metals_pipeline_error() {
    let Some(fills) = assembled("rt6_samples") else {
        return;
    };
    let invalid = Error::InvalidDescriptor(DescriptorError::ColorAttachment {
        index: 0,
        metal: MetalAnswer::EndsProcess,
    });
    let fills = (fills.as_slice(), "rt6_fill_vertex", "rt6_fill_rgba16float");
    let (mut refused, mut unblended) = (0, 0);
    for (stem, p) in committed_profiles().expect("profiles") {
        let dev = device(&stem);
        for pf in BFLOAT {
            for mask in [ColorWriteMask::ALL, ColorWriteMask::NONE] {
                let what = format!("{stem} {pf:?} {mask:?}");
                let made = pipeline(&dev, fills, attachment(pf, OVER, mask));
                assert_eq!(made.map(drop), Err(invalid.clone()), "{what}");
                let compiled = compile_with(fills.0, fills.2, &p, &state(&p, pf, OVER, mask));
                assert_eq!(compiled.map(drop), Err(invalid.clone()), "{what}");
                refused += 1;
            }
            let what = format!("{stem} {pf:?} without a fragment function");
            let mut d =
                RenderPipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(
                    Library::from_air(fills.0)
                        .expect("library")
                        .function(fills.1)
                        .expect("function"),
                )));
            d.color_attachments[0] = Some(attachment(pf, OVER, ColorWriteMask::ALL));
            let made = dev.compile_render_pipeline(&d, &Budget::standard());
            assert_eq!(made.map(drop), Err(invalid.clone()), "{what}");
            // The negative control: the same pipeline unblended.
            let mut c = ColorAttachmentDescriptor::new(pf);
            c.blending_enabled = false;
            match pipeline(&dev, fills, c) {
                Ok(_) => unblended += 1,
                Err(e) => assert!(device_gap(&e), "{stem} {pf:?} unblended: {e}"),
            }
        }
    }
    assert!(refused > 0 && unblended > 0, "{refused} {unblended}");
}

/// A fetch of a padded 10-bit attachment reads the tile value of its codes
/// (`extended_range::tile_value`), as Metal's fetch does: each loaded code's value rounded to
/// `half`, a `float4` input's too (the M3 Max under macOS 27.0.1, `probes/padded_fetch.m` of the
/// conformance crate). `pfx_fetch`'s `float4` and `half4` reads of color 0 (mask none), stored into
/// an `RGBA32Float` color 1, make the whole pipeline on every committed profile that orders fetch:
/// through the pixel interlock with color 0 a storage attachment of the raw format the function
/// does not write, or through rasterization-order input attachments with color 0 an attachment of
/// the raw format read as an unsigned input attachment. On one that orders it in neither form it
/// takes that input attachment too, ordered between draws alone, its fragment stage noting
/// `UnorderedFetchWithinDraw`. Before the read was built each was refused
/// `NotImplemented(FramebufferFetch)`. The negative control: a fetch of a bfloat attachment, whose
/// read Metal answers unmeasured, stays `NotImplemented(Format)` on every profile.
#[test]
fn a_fetch_of_a_padded_attachment_reads_the_tile_value_of_its_codes() {
    use crate::error::Feature;
    use crate::lower::fragment::{inputs_order, interlock_orders};
    use crate::metal::format::ExtendedRange;
    let Some(bc) = assembled("pfx_fetch") else {
        return;
    };
    let made = |dev: &Device, fragment: &str, pf: PixelFormat| {
        let function = |name: &str| {
            FunctionDescriptor::new(
                Library::from_air(&bc)
                    .expect("library")
                    .function(name)
                    .expect("function"),
            )
        };
        let mut d =
            RenderPipelineDescriptor::new(StageSource::Function(function("pfx_fetch_vertex")));
        d.fragment = Some(StageSource::Function(function(fragment)));
        let mut read = ColorAttachmentDescriptor::new(pf);
        read.write_mask = ColorWriteMask::NONE;
        d.color_attachments[0] = Some(read);
        d.color_attachments[1] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA32Float));
        dev.compile_render_pipeline(&d, &Budget::standard())
    };
    let (mut interlocked, mut inputs, mut unordered, mut bfloat) = (0, 0, 0, 0);
    for (stem, p) in committed_profiles().expect("profiles") {
        let dev = device(&stem);
        for fragment in ["pfx_fetch_read", "pfx_fetch_read_half"] {
            for pf in PADDED {
                let what = format!("{stem} {pf:?} {fragment}");
                let raw = ExtendedRange::of(pf).expect("padded").raw_format();
                let made = made(&dev, fragment, pf);
                if interlock_orders(&p) {
                    let a = match made {
                        Ok(a) => a,
                        Err(e) if device_gap(&e) => {
                            eprintln!("{what}: {e}");
                            continue;
                        }
                        Err(e) => panic!("{what}: {e}"),
                    };
                    let uses: Vec<(u32, vk::Format, bool)> = a
                        .graphics
                        .storage_attachments
                        .iter()
                        .map(|s| (s.color_index, s.format, s.writes))
                        .collect();
                    assert_eq!(uses, [(0, raw, false)], "{what}");
                    interlocked += 1;
                } else {
                    // Rasterization-order input attachments, or with neither ordered form input
                    // attachments ordered between draws alone, noted.
                    let a = made.unwrap_or_else(|e| panic!("{what}: {e}"));
                    let state0 = a.graphics.color_attachments[0].expect("color 0");
                    assert_eq!(state0.format, raw, "{what}");
                    let read: Vec<u32> = a
                        .graphics
                        .input_attachments
                        .iter()
                        .map(|i| i.color_index)
                        .collect();
                    assert_eq!(read, [0], "{what}");
                    let noted = (a.stages.iter()).any(|s| {
                        (s.precision_notes).contains(&PrecisionNote::UnorderedFetchWithinDraw)
                    });
                    assert_eq!(noted, !inputs_order(&p), "{what}");
                    match inputs_order(&p) {
                        true => inputs += 1,
                        false => unordered += 1,
                    }
                }
            }
            for pf in BFLOAT {
                let what = format!("{stem} {pf:?} {fragment}");
                match made(&dev, fragment, pf).map(drop) {
                    Err(Error::Unsupported(u))
                        if u.feature == Feature::Format && u.needs == Needs::NotImplemented =>
                    {
                        bfloat += 1
                    }
                    Err(e) if device_gap(&e) => eprintln!("{what}: {e}"),
                    other => panic!("{what}: {other:?}"),
                }
            }
        }
    }
    assert!(interlocked > 0, "no profile reads in the storage form");
    assert!(inputs > 0, "no profile reads through the input attachment");
    assert!(unordered > 0, "no profile orders fetch in neither form");
    assert!(bfloat > 0, "no bfloat fetch was refused");
}

/// An unblended bfloat attachment (`R16Bfloat`, `RG16Bfloat`, `RGBA16Bfloat`) the draw moves into
/// the interlocked storage form (named in `LoweringInputs::framebuffer_fetch`, as a render pass
/// instance names a color it keeps out of its subpass for other draws) is written there as the
/// fixed-function path writes it: Metal's bfloat code of the output's `half`
/// (`extended_color::bfloat_output`), and under a mask of some channels the texel's kept channels
/// as the attachment held them (the raw lanes of the texel the write read, as the fixed-function
/// write mask keeps them). On every committed profile that orders fetch through the pixel
/// interlock alone, `rt6_samples`' `float4` and `half4` fills compile with color 0 a written
/// storage attachment under every mask and pass L3, the `float4` fill recording the pipeline's
/// `textureWriteRoundingMode` its write rounds by. The same functions without the naming take no
/// storage form (the control). Before, a write of some channels failed inside the lowering, and
/// a write of every one stored a kernel write's code, which drops the low bits.
#[test]
fn an_unblended_bfloat_attachment_is_written_in_the_storage_form() {
    use crate::lower::fragment::interlock_orders;
    let Some(fills) = assembled("rt6_samples") else {
        return;
    };
    let masks = [
        ColorWriteMask::ALL,
        ColorWriteMask::RED,
        ColorWriteMask::GREEN.union(ColorWriteMask::ALPHA),
        ColorWriteMask::NONE,
    ];
    let inputs = |p: &DeviceProfile, pf: PixelFormat, mask: ColorWriteMask, named: bool| {
        let mut r = RenderLinkInputs::new();
        let mut c = ColorAttachmentDescriptor::new(pf);
        c.write_mask = mask;
        r.color_attachments[0] = Some(c);
        let mut inputs = fragment_inputs(&r, p);
        if named {
            inputs.framebuffer_fetch = Some(FramebufferFetchInputs::of(
                [0],
                &r.color_attachments,
                false,
                false,
                1,
            ));
        }
        inputs
    };
    let mut checked = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        if !interlock_orders(&p) {
            continue;
        }
        for pf in BFLOAT {
            for fill in ["rt6_fill_rgba32float", "rt6_fill_rgba16float"] {
                for mask in masks {
                    let what = format!("{stem} {pf:?} {fill} {mask:?}");
                    let a = match compile_with(&fills, fill, &p, &inputs(&p, pf, mask, true)) {
                        Ok(a) => a,
                        Err(e) if device_gap(&e) => {
                            eprintln!("{what}: {e}");
                            continue;
                        }
                        Err(e) => panic!("{what}: {e}"),
                    };
                    assert_eq!(fetched(&a), Some((true, None)), "{what}: {:?}", a.facts);
                    // A `float` output rounds by the pipeline's mode, which link holds.
                    if mask != ColorWriteMask::NONE {
                        let rounds = (fill == "rt6_fill_rgba32float")
                            .then_some(crate::metal::TextureWriteRoundingMode::TowardZero);
                        assert_eq!(a.interface.io.texture_write_rounding, rounds, "{what}");
                    }
                    link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
                    // The control: unnamed, the attachment is the fixed function's.
                    let plain = compile_with(&fills, fill, &p, &inputs(&p, pf, mask, false))
                        .unwrap_or_else(|e| panic!("{what}: unnamed: {e}"));
                    assert_eq!(fetched(&plain), None, "{what}");
                    checked += 1;
                }
            }
        }
    }
    assert!(
        checked > 0,
        "no interlock profile wrote a bfloat attachment"
    );
}
