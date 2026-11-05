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
    let mut failures = Vec::new();
    for stem in VENUES {
        let dev = device(stem);
        for pf in [PixelFormat::BGRA10XR, PixelFormat::BGRA10XRSRGB] {
            for (src, name) in [
                (BlendFactor::One, "premultiplied"),
                (BlendFactor::SourceAlpha, "straight"),
            ] {
                for samples in [1, 4] {
                    for fill in ["rt6_fill_rgba16float", "rt6_fill_rgba32float"] {
                        let mut a = ColorAttachmentDescriptor::new(pf);
                        a.blending_enabled = true;
                        a.source_rgb_blend_factor = src;
                        a.source_alpha_blend_factor = BlendFactor::One;
                        a.destination_rgb_blend_factor = BlendFactor::OneMinusSourceAlpha;
                        a.destination_alpha_blend_factor = BlendFactor::OneMinusSourceAlpha;
                        a.write_mask = ColorWriteMask::ALL;
                        let mut d = RenderPipelineDescriptor::new(function("rt6_fill_vertex"));
                        d.fragment = Some(function(fill));
                        d.color_attachments[0] = Some(a);
                        d.raster_sample_count = samples;
                        let what = format!("{stem} {pf:?} {name} x{samples} {fill}");
                        match dev.compile_render_pipeline(&d, &Budget::standard()) {
                            Ok(_) => eprintln!("{what}: made"),
                            Err(e) => {
                                eprintln!("{what}: {e:?}");
                                failures.push(what);
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}
