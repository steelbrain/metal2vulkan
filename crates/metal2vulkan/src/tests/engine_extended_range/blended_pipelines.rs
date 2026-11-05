//! Whole render pipelines that blend into an extended-range attachment, in the shape a wide-gamut
//! `CAMetalLayer` client makes them: color 0 `BGRA10_XR` (or its `_sRGB` twin), alpha-over blending
//! (premultiplied and straight), one sample and four, a `half4` and a `float4` output.

use super::*;
use crate::metal::{BlendFactor, ColorWriteMask};
use crate::pipeline::descriptor::{ColorAttachmentDescriptor, RenderPipelineDescriptor};

/// The profiles of the venues the report and the suite run on.
const VENUES: [&str; 4] = [
    "nvidia-rtx50-linux",
    "nvidia-rtx5080-laptop",
    "moltenvk-m3",
    "lavapipe",
];

fn device(stem: &str) -> Device {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles")
        .join(format!("{stem}.json"));
    let text = std::fs::read_to_string(path).expect("profile");
    let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    Device::new(desc, DevicePolicy::default()).expect("device")
}

#[test]
fn blended_extended_range_pipelines_are_made() {
    let bc = match llvm_assemble(&crate::pipeline::testing::engine_dir().join("rt6_samples.ll")) {
        Ok(b) => b,
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            return;
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected rt6_samples.ll: {why}"),
    };
    let library = Library::from_air(&bc).expect("library");
    let function = |name: &str| {
        StageSource::Function(FunctionDescriptor::new(
            library.function(name).expect("function"),
        ))
    };
    use BlendFactor as F;
    // Impeller's pipeline blends (`ContentContextOptions::ApplyToPipelineDescriptor`): source
    // and destination factors for color and alpha alike, and the straight alpha-over.
    let blends = [
        (
            "source-over",
            F::One,
            F::OneMinusSourceAlpha,
            F::One,
            F::OneMinusSourceAlpha,
        ),
        (
            "straight-over",
            F::SourceAlpha,
            F::OneMinusSourceAlpha,
            F::One,
            F::OneMinusSourceAlpha,
        ),
        ("clear", F::Zero, F::Zero, F::Zero, F::Zero),
        (
            "destination-over",
            F::OneMinusDestinationAlpha,
            F::One,
            F::OneMinusDestinationAlpha,
            F::One,
        ),
        (
            "source-in",
            F::DestinationAlpha,
            F::Zero,
            F::DestinationAlpha,
            F::Zero,
        ),
        (
            "destination-in",
            F::Zero,
            F::SourceAlpha,
            F::Zero,
            F::SourceAlpha,
        ),
        (
            "source-out",
            F::OneMinusDestinationAlpha,
            F::Zero,
            F::OneMinusDestinationAlpha,
            F::Zero,
        ),
        (
            "destination-out",
            F::Zero,
            F::OneMinusSourceAlpha,
            F::Zero,
            F::OneMinusSourceAlpha,
        ),
        (
            "source-atop",
            F::DestinationAlpha,
            F::OneMinusSourceAlpha,
            F::DestinationAlpha,
            F::OneMinusSourceAlpha,
        ),
        (
            "xor",
            F::OneMinusDestinationAlpha,
            F::OneMinusSourceAlpha,
            F::OneMinusDestinationAlpha,
            F::OneMinusSourceAlpha,
        ),
        ("plus", F::One, F::One, F::One, F::One),
        ("modulate", F::Zero, F::SourceColor, F::Zero, F::SourceAlpha),
    ];
    let mut failures = Vec::new();
    let mut made = 0;
    for stem in VENUES {
        let dev = device(stem);
        for pf in [PixelFormat::BGRA10XR, PixelFormat::BGRA10XRSRGB] {
            for &(name, src, dst, src_a, dst_a) in &blends {
                for mask in [ColorWriteMask::ALL, ColorWriteMask::NONE] {
                    for ds in [
                        None,
                        Some(PixelFormat::Stencil8),
                        Some(PixelFormat::Depth32FloatStencil8),
                    ] {
                        for samples in [1, 4] {
                            for fill in ["rt6_fill_rgba16float", "rt6_fill_rgba32float"] {
                                let mut a = ColorAttachmentDescriptor::new(pf);
                                a.blending_enabled = true;
                                a.source_rgb_blend_factor = src;
                                a.destination_rgb_blend_factor = dst;
                                a.source_alpha_blend_factor = src_a;
                                a.destination_alpha_blend_factor = dst_a;
                                a.write_mask = mask;
                                let mut d =
                                    RenderPipelineDescriptor::new(function("rt6_fill_vertex"));
                                d.fragment = Some(function(fill));
                                d.color_attachments[0] = Some(a);
                                d.raster_sample_count = samples;
                                d.stencil_format = ds;
                                if ds == Some(PixelFormat::Depth32FloatStencil8) {
                                    d.depth_format = ds;
                                }
                                let what = format!(
                                    "{stem} {pf:?} {name} mask {:?} {ds:?} x{samples} {fill}",
                                    mask.as_raw()
                                );
                                match dev.compile_render_pipeline(&d, &Budget::standard()) {
                                    Ok(_) => made += 1,
                                    Err(e) => {
                                        eprintln!("REFUSED {what}: {e:?}");
                                        failures.push(what);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    eprintln!("made {made}, refused {}", failures.len());
    assert!(failures.is_empty(), "{} refused", failures.len());
}

/// The other shapes such a client makes: a pipeline without a fragment function, a fragment
/// function that writes no color (depth alone), and write masks of some channels, blended and not.
#[test]
fn other_extended_range_pipeline_shapes_are_made() {
    let bc = match llvm_assemble(&crate::pipeline::testing::engine_dir().join("rt6_samples.ll")) {
        Ok(b) => b,
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            return;
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected rt6_samples.ll: {why}"),
    };
    let library = Library::from_air(&bc).expect("library");
    let function = |name: &str| {
        StageSource::Function(FunctionDescriptor::new(
            library.function(name).expect("function"),
        ))
    };
    use BlendFactor as F;
    let masks = [
        ColorWriteMask::ALL,
        ColorWriteMask::NONE,
        ColorWriteMask::RED
            .union(ColorWriteMask::GREEN)
            .union(ColorWriteMask::BLUE),
        ColorWriteMask::ALPHA,
        ColorWriteMask::RED,
    ];
    let mut failures = Vec::new();
    let mut made = 0;
    for stem in VENUES {
        let dev = device(stem);
        for pf in [PixelFormat::BGRA10XR, PixelFormat::BGRA10XRSRGB] {
            for blended in [false, true] {
                for mask in masks {
                    for fragment in [
                        None,
                        Some("rt6_fill_depth32float"),
                        Some("rt6_fill_rgba16float"),
                    ] {
                        for samples in [1, 4] {
                            let mut a = ColorAttachmentDescriptor::new(pf);
                            a.blending_enabled = blended;
                            a.source_rgb_blend_factor = F::One;
                            a.destination_rgb_blend_factor = F::OneMinusSourceAlpha;
                            a.source_alpha_blend_factor = F::One;
                            a.destination_alpha_blend_factor = F::OneMinusSourceAlpha;
                            a.write_mask = mask;
                            let mut d = RenderPipelineDescriptor::new(function("rt6_fill_vertex"));
                            d.fragment = fragment.map(function);
                            d.color_attachments[0] = Some(a);
                            d.raster_sample_count = samples;
                            if fragment == Some("rt6_fill_depth32float") {
                                d.depth_format = Some(PixelFormat::Depth32Float);
                            }
                            let what = format!(
                                "{stem} {pf:?} blended {blended} mask {} {fragment:?} x{samples}",
                                mask.as_raw()
                            );
                            match dev.compile_render_pipeline(&d, &Budget::standard()) {
                                Ok(_) => made += 1,
                                Err(e) => {
                                    eprintln!("REFUSED {what}: {e:?}");
                                    failures.push(what);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    eprintln!("made {made}, refused {}", failures.len());
    assert!(failures.is_empty(), "{} refused", failures.len());
}
