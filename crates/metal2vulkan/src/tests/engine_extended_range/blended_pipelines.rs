//! Whole render pipelines into an extended-range attachment in the shapes a wide-gamut
//! `CAMetalLayer` client (Impeller's) makes them: color 0 `BGRA10_XR` or its `_sRGB` twin, under
//! Impeller's pipeline blends (premultiplied source-over among them) and straight alpha-over, write
//! masks of every channel and of none, no depth or stencil attachment, `Stencil8` or
//! `Depth32Float_Stencil8`, one sample and four, a `half4` and a `float4` output; and the shapes
//! beside them: a pipeline without a fragment function, a fragment function that writes no color,
//! and write masks of some channels.
//!
//! On the profiles of the reported device's family (`nvidia-rtx50-linux`, `nvidia-rtx5080-laptop`),
//! on MoltenVK's and on lavapipe's, every pipeline is made (but lavapipe's of four samples, below):
//! a written blended attachment in the shader's blend (the interlocked storage form, or over the
//! input attachment on lavapipe), an unwritten one with no blend and an empty mask, a partly masked
//! one under the mask of its fields. The committed lavapipe profile states no multisampled color
//! attachment of the raw format (`R16G16B16A16_UINT`: its `framebufferIntegerColorSampleCounts` is
//! one sample, and it records no count of that format's own), so a pipeline of four samples is
//! refused there naming the properties it needs. Before, a blended attachment the function does
//! not write and a mask of some channels were each refused `NotImplemented(Format)` at `Color(0)`,
//! the reported refusal. An unblended mask of some of the codes one word holds (`BGR10_XR`'s, the
//! two-plane forms') merges them in the storage form where the device orders fetch through the
//! interlock. No GPU is used.
//!
//! `llvm-as` assembles the fixture `rt6_samples`; without it the tests skip visibly.

use super::*;
use crate::error::{Error, Needs};
use crate::metal::{BlendFactor, ColorWriteMask};
use crate::pipeline::descriptor::{ColorAttachmentDescriptor, RenderPipelineDescriptor};
use crate::pipeline::RenderPipeline;
use crate::spirv::testing::check_words_valid;
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
    Device::new(desc, DevicePolicy::default()).expect("device")
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

/// Whether `pipe` reads and writes color 0 in the interlocked storage form.
fn stored(pipe: &RenderPipeline) -> bool {
    (pipe.graphics.storage_attachments.iter()).any(|u| u.color_index == 0)
}

#[test]
fn wide_gamut_client_pipelines_are_made() {
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
    let mut made_count = 0;
    for stem in VENUES {
        let dev = device(stem);
        let interlock = crate::lower::fragment::interlock_orders(dev.profile());
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
                                let Some(pipe) = made(stem, &dev, &d, &what) else {
                                    continue;
                                };
                                made_count += 1;
                                // The shader blends: in the storage form on an interlock device,
                                // over the input attachment on lavapipe, with no Vulkan blend.
                                if interlock {
                                    assert!(stored(&pipe), "{what}");
                                } else {
                                    let a = pipe.graphics.color_attachments[0].expect("attachment");
                                    assert!(a.blend.is_none(), "{what}");
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    // Every shape on the three venues of four samples, and lavapipe's single-sampled half.
    let shapes = FORMATS.len() * blends.len() * 2 * depth_stencil.len() * 2;
    assert_eq!(made_count, 3 * shapes * 2 + shapes);
}

/// A pipeline without a fragment function and one whose function writes no color (depth alone)
/// leave a blended attachment as it is, under every mask: no blend, an empty mask. A mask of some
/// of the format's channels, blended or not, is made: unblended it is the mask of the fields, red
/// and blue traded; blended, the shader's blend keeps the channels it leaves out.
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
        let interlock = crate::lower::fragment::interlock_orders(dev.profile());
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
                            let Some(pipe) = made(stem, &dev, &d, &what) else {
                                continue;
                            };
                            made_count += 1;
                            let written = fragment == Some("rt6_fill_rgba16float");
                            if blended && written && interlock {
                                assert!(stored(&pipe), "{what}");
                                continue;
                            }
                            assert!(!stored(&pipe), "{what}");
                            let a = pipe.graphics.color_attachments[0].expect("attachment");
                            assert!(a.blend.is_none(), "{what}");
                            assert_eq!(a.format, vk::Format::R16G16B16A16_UINT, "{what}");
                            let expected = match (written, blended) {
                                (false, _) => C::empty(),
                                (true, false) => fields,
                                // Over the input attachment the write stores every raw component
                                // (none where the mask keeps every channel).
                                (true, true) if fields.is_empty() => C::empty(),
                                (true, true) => C::R | C::G | C::B | C::A,
                            };
                            assert_eq!(a.write_mask, expected, "{what}");
                        }
                    }
                }
            }
        }
    }
    let shapes = FORMATS.len() * 2 * masks.len() * fragments.len();
    assert_eq!(made_count, 3 * shapes * 2 + shapes);
}

/// On a device that orders framebuffer fetch in neither form (no `fragmentShaderPixelInterlock`,
/// no `rasterizationOrderColorAttachmentAccess`: the `vk12-floor` and `roadmap-2022` profiles), a
/// written blended attachment of either format, which the function blends over its own read of
/// the codes, is refused as framebuffer fetch needing either feature, the refusal the public door
/// (`metal::format::extended_range_blend`) gives, where it was `NotImplemented(Format)`. The
/// unwritten attachment, which takes no blend, is still made there.
#[test]
fn an_unordered_device_names_what_an_extended_range_blend_needs() {
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
    let wanted = crate::lower::fragment::fetch::unordered_extended_fetch();
    for stem in ["vk12-floor", "roadmap-2022"] {
        let dev = device(stem);
        assert!(
            crate::lower::fragment::fetch::fetch_unordered(dev.profile()),
            "{stem}"
        );
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
/// components gives: where the device orders fetch only through the pixel interlock, the function
/// merges its codes into the word it reads first, in the interlocked storage form (color 0 leaves
/// the subpass), at one sample and four, from a `half4` output and a `float4` one, and every stage
/// passes L3. On lavapipe, which orders fetch through rasterization-order input attachments, the
/// function merges them over the input attachment: color 0 stays a color attachment of the raw
/// format with no blend and a write of every raw component, the stage states
/// `RuntimeFact::WordInputMerge` with the channels the mask keeps, and it passes L3 (four samples
/// are its profile's gap). On a device that orders fetch in neither form the refusal names the
/// orderings the merge needs. Before, every venue refused the mask `NotImplemented(Format)` at link.
#[test]
fn a_mask_of_some_codes_of_a_word_merges_them_in_the_storage_form() {
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
    let unordered = crate::lower::fragment::fetch::unordered_extended_fetch();
    let mut merged = 0;
    for stem in [
        "nvidia-rtx50-linux",
        "nvidia-rtx5080-laptop",
        "moltenvk-m3",
        "intel-uhd770-rpls-anv",
        "lavapipe",
        "roadmap-2022",
    ] {
        let dev = device(stem);
        let interlock = crate::lower::fragment::interlock_orders(dev.profile());
        let inputs = crate::lower::fragment::inputs_order(dev.profile());
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
                            Ok(pipe) => {
                                assert!(interlock || inputs, "{what}");
                                assert_eq!(stored(&pipe), interlock, "{what}");
                                if inputs {
                                    let a = pipe.graphics.color_attachments[0].expect("attachment");
                                    let form = ExtendedRange::of(pf).expect("a form");
                                    assert_eq!(a.format, form.raw_format(), "{what}");
                                    let whole = match form.raw_components() {
                                        1 => vk::ColorComponentFlags::R,
                                        _ => {
                                            vk::ColorComponentFlags::R | vk::ColorComponentFlags::G
                                        }
                                    };
                                    assert_eq!(a.write_mask, whole, "{what}");
                                    assert!(a.blend.is_none(), "{what}");
                                    let kept = crate::metal::state::color_write_mask(
                                        crate::lower::fragment::kept_channels(pf, mask, true),
                                    )
                                    .expect("mask");
                                    let fact = crate::pipeline::RuntimeFact::WordInputMerge {
                                        color: 0,
                                        kept,
                                    };
                                    assert!(
                                        pipe.dependencies.iter().any(|d| d.fact == fact),
                                        "{what}: {:?}",
                                        pipe.dependencies
                                    );
                                }
                                for m in &pipe.stages {
                                    check_words_valid(&m.spirv, &pipe.requirements, dev.profile())
                                        .unwrap_or_else(|e| panic!("{what}: L3: {e}"));
                                }
                                merged += 1;
                            }
                            Err(Error::Unsupported(u)) if !interlock && !inputs => {
                                assert_eq!(u.feature, unordered.feature, "{what}: {u:?}");
                                assert_eq!(u.needs, unordered.needs, "{what}: {u:?}");
                            }
                            Err(e) if inputs && samples > 1 && lacks_samples(&e) => {}
                            Err(e) => panic!("{what}: {e:?}"),
                        }
                    }
                }
            }
        }
    }
    // Four interlock venues at both counts, lavapipe at one sample.
    assert_eq!(merged, (4 * 2 + 1) * formats.len() * masks.len() * 2);
}
