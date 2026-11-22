//! A blend into a padded 10-bit attachment (`R10Unorm_X6`, `RG10Unorm_X12`, their `_sRGB` twins
//! and `RGB10A2Unorm_sRGB`), which Metal makes and draws (the M3 Max under macOS 27.0.1:
//! `probes/padded_and_bfloat_blends.m` of the conformance crate), and which no device blends: the
//! attachment holds the raw words in an unsigned integer format.
//!
//! The function's own blend over an ordered read of the attachment is not built, so on every
//! committed profile `rt6_samples`' `half4` and `float4` fills under premultiplied source-over and
//! one plus one, `xrb_dual`'s second-source blends and a write mask of some of the channels are
//! refused `NotImplemented(FramebufferFetch)`; a blend into one of Apple's extended-range forms,
//! which Metal's tile holds as `half`, is refused (`blended_pipelines`). No GPU is used.
//!
//! `llvm-as` assembles the fixtures; a missing tool skips visibly.

use std::path::Path;

use crate::budget::Budget;
use crate::device::{Device, DeviceDescription, DevicePolicy, DeviceProfile};
use crate::error::{DescriptorError, Error, Feature, MetalAnswer, Needs};
use crate::library::Library;
use crate::metal::{BlendFactor, ColorWriteMask, PixelFormat};
use crate::pipeline::artifact::{PrecisionNote, RenderPipeline};
use crate::pipeline::descriptor::{
    ColorAttachmentDescriptor, FunctionDescriptor, RenderPipelineDescriptor, StageSource,
};
use crate::pipeline::link::RenderLinkInputs;
use crate::pipeline::lower_inputs::LoweringInputs;
use crate::pipeline::testing::{compile_with, engine_dir, fragment_inputs, AssembleError};
use crate::spirv::testing::committed_profiles;

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

/// A blend's source and destination factors for color and for alpha, each added.
type Factors = [BlendFactor; 4];

/// Premultiplied source-over: one over one minus the source alpha, for color and alpha.
const OVER: Factors = [
    BlendFactor::One,
    BlendFactor::OneMinusSourceAlpha,
    BlendFactor::One,
    BlendFactor::OneMinusSourceAlpha,
];

/// One plus one, for color and alpha.
const ADD: Factors = [BlendFactor::One; 4];

/// The second source's blend: its color and one minus it, its alpha and one minus it.
const SECOND: Factors = [
    BlendFactor::Source1Color,
    BlendFactor::OneMinusSource1Color,
    BlendFactor::Source1Alpha,
    BlendFactor::OneMinusSource1Alpha,
];

/// Color 0 of `pf` blended by `factors`, added, under write mask `mask`.
fn attachment(
    pf: PixelFormat,
    factors: Factors,
    mask: ColorWriteMask,
) -> ColorAttachmentDescriptor {
    let mut c = ColorAttachmentDescriptor::new(pf);
    c.blending_enabled = true;
    c.source_rgb_blend_factor = factors[0];
    c.destination_rgb_blend_factor = factors[1];
    c.source_alpha_blend_factor = factors[2];
    c.destination_alpha_blend_factor = factors[3];
    c.write_mask = mask;
    c
}

/// The lowering inputs of a fragment function writing color 0 of `pf` blended by `factors` under
/// `mask` on `p`, as the render pipeline compile sets them.
fn state(
    p: &DeviceProfile,
    pf: PixelFormat,
    factors: Factors,
    mask: ColorWriteMask,
) -> LoweringInputs {
    let mut r = RenderLinkInputs::new();
    r.color_attachments[0] = Some(attachment(pf, factors, mask));
    fragment_inputs(&r, p)
}

/// The device of committed profile `stem`.
fn device(stem: &str) -> Device {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles")
        .join(format!("{stem}.json"));
    let text = std::fs::read_to_string(path).expect("profile");
    let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    Device::from_queries(desc, DevicePolicy::RESIDENT_TABLES).expect("device")
}

/// Whether `e` is a refusal the device's lack of a capability makes, whatever the blend.
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

/// A blend into a padded attachment is refused `NotImplemented(FramebufferFetch)` on every
/// committed profile, under every blend and mask here (the module docs).
#[test]
fn a_blend_into_a_padded_attachment_is_refused_on_every_device() {
    let (Some(fills), Some(dual)) = (assembled("rt6_samples"), assembled("xrb_dual")) else {
        return;
    };
    // Each function, its vertex function, and the blends it takes.
    let entries: [(&[u8], &str, &str, &[Factors]); 4] = [
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
    let mut refused = 0;
    for (stem, _) in committed_profiles().expect("profiles") {
        let dev = device(&stem);
        for pf in PADDED {
            for &(bc, vertex, fragment, blends) in &entries {
                for &factors in blends {
                    for mask in [ColorWriteMask::ALL, ColorWriteMask::RED, rgb] {
                        let what = format!("{stem} {pf:?} {fragment} {factors:?} {mask:?}");
                        let made =
                            pipeline(&dev, (bc, vertex, fragment), attachment(pf, factors, mask));
                        match made.map(drop) {
                            Err(Error::Unsupported(u)) => assert_eq!(
                                (u.feature, u.needs),
                                (Feature::FramebufferFetch, Needs::NotImplemented),
                                "{what}"
                            ),
                            other => panic!("{what}: {other:?}"),
                        }
                        refused += 1;
                    }
                }
            }
        }
    }
    assert!(refused > 0);
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
/// an `RGBA32Float` color 1, make the whole pipeline on every committed profile, color 0 an
/// attachment of the raw format read as an unsigned input attachment, ordered between draws alone,
/// its fragment stage noting `UnorderedFetchWithinDraw`. The negative control: a fetch of a
/// bfloat attachment, whose read Metal answers unmeasured, stays `NotImplemented(Format)` on every
/// profile.
#[test]
fn a_fetch_of_a_padded_attachment_reads_the_tile_value_of_its_codes() {
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
    let (mut read_padded, mut bfloat) = (0, 0);
    for (stem, _) in committed_profiles().expect("profiles") {
        let dev = device(&stem);
        for fragment in ["pfx_fetch_read", "pfx_fetch_read_half"] {
            for pf in PADDED {
                let what = format!("{stem} {pf:?} {fragment}");
                let raw = ExtendedRange::of(pf).expect("padded").raw_format();
                // Input attachments ordered between draws alone, noted.
                let a = made(&dev, fragment, pf).unwrap_or_else(|e| panic!("{what}: {e}"));
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
                assert!(noted, "{what}");
                read_padded += 1;
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
    assert!(read_padded > 0, "no padded fetch was made");
    assert!(bfloat > 0, "no bfloat fetch was refused");
}
