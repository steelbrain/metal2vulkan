//! Whole render pipelines into an extended-range attachment in the shapes a wide-gamut
//! `CAMetalLayer` client (Impeller's) makes them: color 0 `BGRA10_XR` or its `_sRGB` twin, under
//! Impeller's pipeline blends (premultiplied source-over among them) and straight alpha-over, write
//! masks of every channel and of none, no depth or stencil attachment, `Stencil8` or
//! `Depth32Float_Stencil8`, one sample and four, a `half4` and a `float4` output; and the shapes
//! beside them: a pipeline without a fragment function, a fragment function that writes no color,
//! and write masks of some channels.
//!
//! On the profiles of the reported device's family (`nvidia-rtx50-linux`, `nvidia-rtx5080-laptop`),
//! on MoltenVK's, on lavapipe's and on the runtime's floor, a written blended attachment is refused
//! `NotImplemented(Format)`: Metal's tile holds the attachment as `half` through the pass, which
//! the codes it holds do not keep (`lower::fragment::fetch::codes_round_the_tile`). Every other
//! pipeline is made (but lavapipe's of four samples, below): an unwritten one with no blend and an
//! empty mask, a partly masked one under the mask of its fields. The committed lavapipe profile
//! states no multisampled color attachment of the raw format (`R16G16B16A16_UINT`: its
//! `framebufferIntegerColorSampleCounts` is one sample, and it records no count of that format's
//! own), so a pipeline of four samples is refused there naming the properties it needs. An
//! unblended mask of some of the codes one word holds (`BGR10_XR`'s, the two-plane forms') is
//! refused `NotImplemented(FramebufferFetch)` on every device: their merge over an ordered read of
//! the attachment is not built. No GPU is used.
//!
//! `llvm-as` assembles the fixture `rt6_samples`; without it the tests skip visibly.

use super::*;
use crate::error::{Error, Needs, Unsupported};
use crate::metal::{BlendFactor, ColorWriteMask};
use crate::pipeline::descriptor::{ColorAttachmentDescriptor, RenderPipelineDescriptor};
use crate::pipeline::RenderPipeline;
use crate::vk;

/// The profiles: the reported device's family, MoltenVK's and lavapipe's.
const VENUES: [&str; 4] = [
    "nvidia-rtx50-linux",
    "nvidia-rtx5080-laptop",
    "moltenvk-m3",
    "lavapipe",
];

/// The formats a wide-gamut layer asks for.
const FORMATS: [PixelFormat; 2] = [PixelFormat::BGRA10XR, PixelFormat::BGRA10XRSRGB];

/// The device of committed profile `stem`.
fn device(stem: &str) -> Device {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles")
        .join(format!("{stem}.json"));
    let text = std::fs::read_to_string(path).expect("profile");
    let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    Device::from_queries(desc, DevicePolicy::RESIDENT_TABLES).expect("device")
}

/// The fixture's library; `None` (skipped) without `llvm-as`.
fn library() -> Option<Library> {
    match llvm_assemble(&crate::pipeline::testing::engine_dir().join("rt6_samples.ll")) {
        Ok(b) => Some(Library::from_air(&b).expect("library")),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected rt6_samples.ll: {why}"),
    }
}

/// Whether `e` is the lavapipe profile's gap at four samples: no multisampled color attachment of
/// the raw format it states.
fn lacks_samples(e: &Error) -> bool {
    matches!(e, Error::Unsupported(u)
        if u.feature == Feature::RasterSampleCount && matches!(u.needs, Needs::AnyOf(_)))
}

/// Compile `d` on `dev` (profile `stem`): the pipeline, `None` for lavapipe's gap at four samples,
/// or a panic naming `what`.
fn made(
    stem: &str,
    dev: &Device,
    d: &RenderPipelineDescriptor,
    what: &str,
) -> Option<RenderPipeline> {
    match dev.compile_render_pipeline(d, &Budget::standard()) {
        Ok(pipe) => Some(pipe),
        Err(e) if stem == "lavapipe" && d.raster_sample_count > 1 && lacks_samples(&e) => None,
        Err(e) => panic!("{what}: {e:?}"),
    }
}

/// Compile `d` on `dev`, which must refuse it as a blend into one of Apple's extended-range forms:
/// `NotImplemented(Format)` at color 0. A panic naming `what` otherwise.
fn refused(dev: &Device, d: &RenderPipelineDescriptor, what: &str) {
    match dev.compile_render_pipeline(d, &Budget::standard()) {
        Err(Error::Unsupported(u)) => {
            assert_eq!(u.feature, Feature::Format, "{what}: {u:?}");
            assert_eq!(u.needs, Needs::NotImplemented, "{what}: {u:?}");
            let at = u.location.as_ref().and_then(|l| l.attachment());
            assert_eq!(
                at,
                Some(crate::error::Attachment::Color(0)),
                "{what}: {u:?}"
            );
        }
        other => panic!("{what}: {other:?}"),
    }
}

/// Color 0 of `pf` blended (where `factors`: source and destination for color, then alpha) under
/// `mask`.
fn attachment(
    pf: PixelFormat,
    factors: Option<[BlendFactor; 4]>,
    mask: ColorWriteMask,
) -> ColorAttachmentDescriptor {
    let mut a = ColorAttachmentDescriptor::new(pf);
    if let Some([src, dst, src_a, dst_a]) = factors {
        a.blending_enabled = true;
        a.source_rgb_blend_factor = src;
        a.destination_rgb_blend_factor = dst;
        a.source_alpha_blend_factor = src_a;
        a.destination_alpha_blend_factor = dst_a;
    }
    a.write_mask = mask;
    a
}

/// Every shape a wide-gamut client makes writes the blended attachment, so each is refused on every
/// venue, under a mask of every channel and of none alike.
#[test]
fn wide_gamut_client_pipelines_are_refused() {
    let Some(library) = library() else {
        return;
    };
    let function = |name: &str| {
        StageSource::Function(FunctionDescriptor::new(
            library.function(name).expect("function"),
        ))
    };
    use BlendFactor as F;
    let (sa, da) = (F::SourceAlpha, F::DestinationAlpha);
    let (msa, mda) = (F::OneMinusSourceAlpha, F::OneMinusDestinationAlpha);
    // Impeller's pipeline blends (source and destination, color then alpha) and straight
    // alpha-over.
    let blends: [(&str, [F; 4]); 12] = [
        ("source-over", [F::One, msa, F::One, msa]),
        ("straight-over", [sa, msa, F::One, msa]),
        ("clear", [F::Zero, F::Zero, F::Zero, F::Zero]),
        ("destination-over", [mda, F::One, mda, F::One]),
        ("source-in", [da, F::Zero, da, F::Zero]),
        ("destination-in", [F::Zero, sa, F::Zero, sa]),
        ("source-out", [mda, F::Zero, mda, F::Zero]),
        ("destination-out", [F::Zero, msa, F::Zero, msa]),
        ("source-atop", [da, msa, da, msa]),
        ("xor", [mda, msa, mda, msa]),
        ("plus", [F::One, F::One, F::One, F::One]),
        ("modulate", [F::Zero, F::SourceColor, F::Zero, sa]),
    ];
    let depth_stencil = [
        None,
        Some(PixelFormat::Stencil8),
        Some(PixelFormat::Depth32FloatStencil8),
    ];
    let mut refused_count = 0;
    for stem in VENUES.into_iter().chain(["runtime-floor"]) {
        let dev = device(stem);
        for pf in FORMATS {
            for (name, factors) in blends {
                for mask in [ColorWriteMask::ALL, ColorWriteMask::NONE] {
                    for ds in depth_stencil {
                        for samples in [1, 4] {
                            for fill in ["rt6_fill_rgba16float", "rt6_fill_rgba32float"] {
                                let mut d =
                                    RenderPipelineDescriptor::new(function("rt6_fill_vertex"));
                                d.fragment = Some(function(fill));
                                d.color_attachments[0] = Some(attachment(pf, Some(factors), mask));
                                d.raster_sample_count = samples;
                                d.stencil_format = ds;
                                if ds == Some(PixelFormat::Depth32FloatStencil8) {
                                    d.depth_format = ds;
                                }
                                let what = format!(
                                    "{stem} {pf:?} {name} mask {} {ds:?} x{samples} {fill}",
                                    mask.as_raw()
                                );
                                refused(&dev, &d, &what);
                                refused_count += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    // Every shape at both sample counts on each venue.
    let shapes = FORMATS.len() * blends.len() * 2 * depth_stencil.len() * 2 * 2;
    assert_eq!(refused_count, (VENUES.len() + 1) * shapes);
}

/// A pipeline without a fragment function and one whose function writes no color (depth alone)
/// leave a blended attachment as it is, under every mask: no blend, an empty mask. An unblended
/// mask of some of the format's channels is made, the mask of the fields, red and blue traded. A
/// written blended attachment is refused under every mask (`refused`).
#[test]
fn unwritten_and_masked_extended_range_attachments_are_made() {
    let Some(library) = library() else {
        return;
    };
    let function = |name: &str| {
        StageSource::Function(FunctionDescriptor::new(
            library.function(name).expect("function"),
        ))
    };
    use vk::ColorComponentFlags as C;
    let over = [
        BlendFactor::One,
        BlendFactor::OneMinusSourceAlpha,
        BlendFactor::One,
        BlendFactor::OneMinusSourceAlpha,
    ];
    let rgb = ColorWriteMask::RED
        .union(ColorWriteMask::GREEN)
        .union(ColorWriteMask::BLUE);
    // Each mask, and the mask of `BGRA10_XR`'s fields it is.
    let masks = [
        (ColorWriteMask::ALL, C::R | C::G | C::B | C::A),
        (ColorWriteMask::NONE, C::empty()),
        (rgb, C::R | C::G | C::B),
        (ColorWriteMask::ALPHA, C::A),
        (ColorWriteMask::RED, C::B),
        (
            ColorWriteMask::BLUE.union(ColorWriteMask::GREEN),
            C::R | C::G,
        ),
    ];
    let fragments = [
        None,
        Some("rt6_fill_depth32float"),
        Some("rt6_fill_rgba16float"),
    ];
    let mut made_count = 0;
    for stem in VENUES {
        let dev = device(stem);
        for pf in FORMATS {
            for blended in [false, true] {
                for (mask, fields) in masks {
                    for fragment in fragments {
                        for samples in [1, 4] {
                            let factors = blended.then_some(over);
                            let mut d = RenderPipelineDescriptor::new(function("rt6_fill_vertex"));
                            d.fragment = fragment.map(function);
                            d.color_attachments[0] = Some(attachment(pf, factors, mask));
                            d.raster_sample_count = samples;
                            if fragment == Some("rt6_fill_depth32float") {
                                d.depth_format = Some(PixelFormat::Depth32Float);
                            }
                            let what = format!(
                                "{stem} {pf:?} blended {blended} mask {} {fragment:?} x{samples}",
                                mask.as_raw()
                            );
                            let written = fragment == Some("rt6_fill_rgba16float");
                            if blended && written {
                                refused(&dev, &d, &what);
                                continue;
                            }
                            let Some(pipe) = made(stem, &dev, &d, &what) else {
                                continue;
                            };
                            made_count += 1;
                            let a = pipe.graphics.color_attachments[0].expect("attachment");
                            assert!(a.blend.is_none(), "{what}");
                            assert_eq!(a.format, vk::Format::R16G16B16A16_UINT, "{what}");
                            let expected = match written {
                                false => C::empty(),
                                true => fields,
                            };
                            assert_eq!(a.write_mask, expected, "{what}");
                        }
                    }
                }
            }
        }
    }
    // Every shape but the written blended ones, on the three venues of four samples, and
    // lavapipe's single-sampled half.
    let shapes = FORMATS.len() * (2 * masks.len() * fragments.len() - masks.len());
    assert_eq!(made_count, 3 * shapes * 2 + shapes);
}

/// On the runtime's floor (no `fragmentShaderPixelInterlock`), a written blended attachment of
/// either format is refused `NotImplemented(Format)`, as on every device: no device would make it.
/// The unwritten attachment, which takes no blend, is still made there.
#[test]
fn the_floor_refuses_an_extended_range_blend_as_every_device_does() {
    let Some(library) = library() else {
        return;
    };
    let function = |name: &str| {
        StageSource::Function(FunctionDescriptor::new(
            library.function(name).expect("function"),
        ))
    };
    use BlendFactor as F;
    let over = [
        F::One,
        F::OneMinusSourceAlpha,
        F::One,
        F::OneMinusSourceAlpha,
    ];
    let wanted = Unsupported::not_implemented(Feature::Format);
    for stem in ["runtime-floor"] {
        let dev = device(stem);
        for pf in FORMATS {
            for fill in ["rt6_fill_rgba16float", "rt6_fill_rgba32float"] {
                let mut d = RenderPipelineDescriptor::new(function("rt6_fill_vertex"));
                d.fragment = Some(function(fill));
                d.color_attachments[0] = Some(attachment(pf, Some(over), ColorWriteMask::ALL));
                let what = format!("{stem} {pf:?} {fill}");
                match dev.compile_render_pipeline(&d, &Budget::standard()) {
                    Err(Error::Unsupported(u)) => {
                        assert_eq!(u.feature, wanted.feature, "{what}: {u:?}");
                        assert_eq!(u.needs, wanted.needs, "{what}: {u:?}");
                    }
                    other => panic!("{what}: {other:?}"),
                }
            }
            let mut d = RenderPipelineDescriptor::new(function("rt6_fill_vertex"));
            d.color_attachments[0] = Some(attachment(pf, Some(over), ColorWriteMask::ALL));
            let pipe = dev
                .compile_render_pipeline(&d, &Budget::standard())
                .unwrap_or_else(|e| panic!("{stem} {pf:?} unwritten: {e:?}"));
            let a = pipe.graphics.color_attachments[0].expect("attachment");
            assert!(
                a.blend.is_none() && a.write_mask.is_empty(),
                "{stem} {pf:?}"
            );
        }
    }
}

/// An unblended write of some of the codes one word holds (`BGR10_XR`'s and its twin's word, the
/// two-plane forms' color word: `color_writes::merged_extended`), which no mask of the raw
/// components gives, from a `half4` output and a `float4` one, is refused
/// `NotImplemented(FramebufferFetch)` on every device here, lavapipe and the floor included: the
/// merge over an ordered read of the attachment is not built. At four samples a device that lacks
/// the raw format's samples may name that instead.
#[test]
fn a_mask_of_some_codes_of_a_word_is_refused_on_every_device() {
    let Some(library) = library() else {
        return;
    };
    let function = |name: &str| {
        StageSource::Function(FunctionDescriptor::new(
            library.function(name).expect("function"),
        ))
    };
    let formats = [
        PixelFormat::BGR10XR,
        PixelFormat::BGR10XRSRGB,
        PixelFormat::RGB10A82PXR10,
        PixelFormat::RGB10A82PXR10SRGB,
    ];
    let masks = [
        ColorWriteMask::RED,
        ColorWriteMask::BLUE.union(ColorWriteMask::GREEN),
        ColorWriteMask::RED.union(ColorWriteMask::ALPHA),
    ];
    let stems = [
        "nvidia-rtx50-linux",
        "nvidia-rtx5080-laptop",
        "moltenvk-m3",
        "intel-uhd770-rpls-anv",
        "lavapipe",
        "runtime-floor",
    ];
    let mut refused = 0;
    for stem in stems {
        let dev = device(stem);
        for pf in formats {
            for mask in masks {
                for fill in ["rt6_fill_rgba16float", "rt6_fill_rgba32float"] {
                    for samples in [1, 4] {
                        let mut d = RenderPipelineDescriptor::new(function("rt6_fill_vertex"));
                        d.fragment = Some(function(fill));
                        d.color_attachments[0] = Some(attachment(pf, None, mask));
                        d.raster_sample_count = samples;
                        let what =
                            format!("{stem} {pf:?} mask {} {fill} x{samples}", mask.as_raw());
                        match dev.compile_render_pipeline(&d, &Budget::standard()) {
                            Err(Error::Unsupported(u))
                                if (u.feature, &u.needs)
                                    == (Feature::FramebufferFetch, &Needs::NotImplemented) =>
                            {
                                refused += 1;
                            }
                            Err(e) if samples > 1 && lacks_samples(&e) => {}
                            other => panic!("{what}: {:?}", other.map(|_| ())),
                        }
                    }
                }
            }
        }
    }
    assert!(refused >= stems.len() * formats.len() * masks.len() * 2);
}
