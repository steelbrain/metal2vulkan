//! `ARGB8Unorm` color attachments: an image holding Metal's alpha, red, green and blue in its red,
//! green, blue and alpha.
//!
//! The owned fixture `engine/stage_io`'s render pipeline, its color 0 in `ARGB8Unorm`, goes
//! through `Device::compile_render_pipeline` on every committed profile:
//! - the fragment function writes its `float4` color rotated (an `OpVectorShuffle` of lanes 3, 0,
//!   1, 2) and records the attachment (`StageIo::rotated_colors`);
//! - link gives the attachment `R8G8B8A8_UNORM` and the descriptor's write mask rotated;
//! - a blend the image answers in place keeps Vulkan's factors of it; one whose color factors
//!   read Metal's source alpha (`One` or `SourceAlpha` over `OneMinusSourceAlpha`) reads a second
//!   output of copies of it (`StageIo::rotated_factors`, an output of index 1 the function's
//!   last shuffle fills) through Vulkan's `SRC1` factors where the device blends dual-source and
//!   the function writes no more attachments than its dual-source limit, else names the feature
//!   or the limit; one no Vulkan factor gives the function computes over the texel where the
//!   device orders framebuffer fetch (the interlocked storage form, or over the input attachment,
//!   `RotatedInputBlend`), noting the device's conversion, and a device that orders it in neither
//!   form refuses naming both orderings; alpha to coverage over the image is refused
//!   `NotImplemented(Format)` at the attachment;
//! - where the device's blend is Apple's, straight alpha takes the measured premultiplied form: a
//!   second output of `(0, 0, 0, a)` and Metal's blue times Metal's alpha in the image's alpha; a
//!   blend in none of the measured forms records `PrecisionNote::RotatedBlendRounding`.
//!
//! No GPU is used. `llvm-as` assembles the fixture; a missing tool skips visibly.

use crate::budget::Budget;
use crate::device::requirements::{VkFeature, VkLimit, VkProperty};
use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::error::{Attachment, Capability, Error, Feature, Needs};
use crate::library::Library;
use crate::metal::{BlendFactor, BlendOperation, ColorWriteMask, PixelFormat, VertexFormat};
use crate::pipeline::descriptor::{
    ColorAttachmentDescriptor, VertexAttributeDescriptor, VertexBufferLayoutDescriptor,
    VertexDescriptor,
};
use crate::pipeline::testing::{engine_dir, llvm_assemble, AssembleError};
use crate::pipeline::{
    FunctionDescriptor, PrecisionNote, RenderPipeline, RenderPipelineDescriptor, StageSource,
};
use crate::vk;

/// Every committed profile as a device under the default policy, by stem.
fn devices() -> Vec<(String, Device)> {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles");
    let mut out = Vec::new();
    for entry in std::fs::read_dir(&dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display())) {
        let path = entry.expect("profile directory entry").path();
        if path.extension().is_none_or(|x| x != "json") {
            continue;
        }
        let stem = path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        let text = std::fs::read_to_string(&path).expect("read profile");
        let desc = DeviceDescription::from_khronos_json(&text, None).expect("profile parses");
        let dev = Device::new(desc, DevicePolicy::default()).expect("device derives");
        out.push((stem, dev));
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

fn fixture() -> Option<Vec<u8>> {
    match llvm_assemble(&engine_dir().join("stage_io.ll")) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected stage_io: {why}"),
    }
}

/// `stage_io`'s vertex and fragment functions, color 0 of `color`, color 2 `RGBA32Uint` where
/// `color2` (else the function's color 2 has no attachment, and it writes color 0 alone).
fn descriptor(
    bc: &[u8],
    color: ColorAttachmentDescriptor,
    color2: bool,
) -> RenderPipelineDescriptor {
    let lib = Library::from_air(bc).expect("library");
    let v = lib.function("stage_io_vertex").expect("vertex");
    let f = lib.function("stage_io_fragment").expect("fragment");
    let mut layout = VertexDescriptor::new();
    layout
        .layouts
        .insert(0, VertexBufferLayoutDescriptor::new(Some(24)));
    for (index, format, offset) in [
        (0, VertexFormat::Float3, 0),
        (1, VertexFormat::Half2, 12),
        (4, VertexFormat::UShort2, 16),
        (7, VertexFormat::UChar4, 20),
    ] {
        layout
            .attributes
            .insert(index, VertexAttributeDescriptor::new(format, offset, 0));
    }
    let mut d = RenderPipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(v)));
    d.fragment = Some(StageSource::Function(FunctionDescriptor::new(f)));
    d.vertex_descriptor = Some(layout);
    d.color_attachments[0] = Some(color);
    if color2 {
        d.color_attachments[2] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA32Uint));
    }
    d.depth_format = Some(PixelFormat::Depth32Float);
    d
}

/// The operands of every instruction of opcode `op` in the pipeline's fragment module.
fn fragment_instructions(pipeline: &RenderPipeline, op: crate::spv::Op) -> Vec<Vec<u32>> {
    let mut out = Vec::new();
    for s in pipeline
        .stages
        .iter()
        .filter(|s| s.stage == vk::ShaderStage::FRAGMENT)
    {
        let words = &s.spirv;
        let mut at = 5;
        while at < words.len() {
            let count = (words[at] >> 16) as usize;
            if count == 0 {
                break;
            }
            if words[at] & 0xffff == op as u32 {
                out.push(words[at + 1..at + count].to_vec());
            }
            at += count;
        }
    }
    out
}

/// Whether the fragment module shuffles a `float4` into `lanes`.
fn shuffles(pipeline: &RenderPipeline, lanes: [u32; 4]) -> bool {
    fragment_instructions(pipeline, crate::spv::Op::VectorShuffle)
        .iter()
        .any(|o| o.len() == 8 && o[4..] == lanes)
}

/// Whether the fragment module rotates a `float4`: an `OpVectorShuffle` of lanes 3, 0, 1, 2.
fn rotates(pipeline: &RenderPipeline) -> bool {
    shuffles(pipeline, [3, 0, 1, 2])
}

/// Whether the fragment module declares an output of dual-source index 1.
fn second_output(pipeline: &RenderPipeline) -> bool {
    fragment_instructions(pipeline, crate::spv::Op::Decorate)
        .iter()
        .any(|d| d.get(1) == Some(&(crate::spv::Decoration::Index as u32)) && d.get(2) == Some(&1))
}

/// Whether the fragment module premultiplies a lane: a product of two values each clamped by
/// `GLSL.std.450`'s `NClamp`, an `OpFMul` or, where the builder writes an exact product so, a
/// `GLSL.std.450` `Fma` of them with `-0.0`.
fn premultiplies(pipeline: &RenderPipeline) -> bool {
    const NCLAMP: u32 = 81;
    const FMA: u32 = 50;
    let ext = fragment_instructions(pipeline, crate::spv::Op::ExtInst);
    let clamped: Vec<u32> = ext
        .iter()
        .filter(|o| o.get(3) == Some(&NCLAMP))
        .map(|o| o[1])
        .collect();
    let of_clamped = |a: u32, b: u32| clamped.contains(&a) && clamped.contains(&b);
    fragment_instructions(pipeline, crate::spv::Op::FMul)
        .iter()
        .any(|o| of_clamped(o[2], o[3]))
        || ext
            .iter()
            .any(|o| o.len() == 7 && o[3] == FMA && of_clamped(o[4], o[5]))
}

/// Whether the pipeline's fragment stage records `note`.
fn notes(pipeline: &RenderPipeline, note: PrecisionNote) -> bool {
    pipeline
        .stages
        .iter()
        .filter(|s| s.stage == vk::ShaderStage::FRAGMENT)
        .any(|s| s.precision_notes.contains(&note))
}

/// Whether the pipeline's fragment stage records `PrecisionNote::RotatedBlendRounding`.
fn announces_rounding(pipeline: &RenderPipeline) -> bool {
    notes(pipeline, PrecisionNote::RotatedBlendRounding)
}

fn blended(
    source: BlendFactor,
    destination: BlendFactor,
    rgb: BlendOperation,
    alpha: BlendOperation,
) -> ColorAttachmentDescriptor {
    let mut c = ColorAttachmentDescriptor::new(PixelFormat::ARGB8Unorm);
    c.blending_enabled = true;
    c.source_rgb_blend_factor = source;
    c.source_alpha_blend_factor = source;
    c.destination_rgb_blend_factor = destination;
    c.destination_alpha_blend_factor = destination;
    c.rgb_blend_operation = rgb;
    c.alpha_blend_operation = alpha;
    c
}

#[test]
fn argb8_attachments_take_the_output_mask_and_blend_rotated() {
    let Some(bc) = fixture() else {
        return;
    };
    let mut checked = 0;
    for (stem, dev) in devices() {
        let compile = |c, color2| {
            dev.compile_render_pipeline(&descriptor(&bc, c, color2), &Budget::standard())
        };
        // Red and alpha of Metal's channels: the image's green and red.
        let mut masked = ColorAttachmentDescriptor::new(PixelFormat::ARGB8Unorm);
        masked.write_mask = ColorWriteMask::RED | ColorWriteMask::ALPHA;
        let p = match compile(masked, true) {
            Ok(p) => p,
            // A device without the image's attachment refuses it as a need of the device.
            Err(Error::Unsupported(u)) if u.needs != Needs::NotImplemented => continue,
            Err(e) => panic!("{stem}: {e}"),
        };
        checked += 1;
        let state = p.graphics.color_attachments[0].expect("color 0");
        assert_eq!(state.format, vk::Format::R8G8B8A8_UNORM, "{stem}");
        assert_eq!(
            state.write_mask,
            vk::ColorComponentFlags::R | vk::ColorComponentFlags::G,
            "{stem}"
        );
        assert!(state.blend.is_none(), "{stem}");
        assert!(rotates(&p), "{stem}");
        assert!(!second_output(&p), "{stem}");
        // The same function for `RGBA8Unorm` rotates nothing.
        let plain = compile(
            ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm),
            true,
        )
        .unwrap_or_else(|e| panic!("{stem}: RGBA8Unorm: {e}"));
        assert!(!rotates(&plain), "{stem}");
        use BlendFactor as F;
        use BlendOperation as O;
        // Blends the image answers in place.
        for c in [
            blended(F::SourceColor, F::OneMinusDestinationColor, O::Add, O::Add),
            blended(F::One, F::One, O::ReverseSubtract, O::ReverseSubtract),
            blended(F::SourceAlpha, F::DestinationAlpha, O::Min, O::Min),
        ] {
            match compile(c, true) {
                Ok(p) => {
                    let blend = p.graphics.color_attachments[0].and_then(|s| s.blend);
                    assert!(blend.is_some_and(|b| !b.reads_second_output()), "{stem}");
                    assert!(!second_output(&p), "{stem}: {c:?}");
                    // In none of the measured forms but a minimum's: announced.
                    let unmeasured = c.rgb_blend_operation != O::Min;
                    assert_eq!(announces_rounding(&p), unmeasured, "{stem}: {c:?}");
                }
                Err(Error::Unsupported(u)) if u.needs != Needs::NotImplemented => {}
                Err(e) => panic!("{stem}: {c:?}: {e}"),
            }
        }
        // Blends whose color factors read Metal's source alpha: a second output of copies of it.
        let dual = dev.profile().fragment().dual_src_blend;
        let apple = dev
            .profile()
            .fields()
            .enabled
            .properties
            .contains(&VkProperty::MetalColorWrites);
        for c in [
            blended(F::One, F::OneMinusSourceAlpha, O::Add, O::Add),
            blended(F::SourceAlpha, F::OneMinusSourceAlpha, O::Add, O::Add),
        ] {
            match compile(c, false) {
                Ok(p) => {
                    assert!(dual, "{stem}");
                    let blend = p.graphics.color_attachments[0].and_then(|s| s.blend);
                    assert!(blend.is_some_and(|b| b.reads_second_output()), "{stem}");
                    assert!(second_output(&p) && rotates(&p), "{stem}: {c:?}");
                    // Straight alpha on Apple's blend: the premultiplied form, the second output
                    // `(0, 0, 0, a)` and Metal's blue times Metal's alpha in the image's alpha.
                    // Elsewhere copies of Metal's source alpha. Measured either way: no notice.
                    let premultiplied = apple && c.source_rgb_blend_factor == F::SourceAlpha;
                    assert_eq!(shuffles(&p, [4, 4, 4, 3]), premultiplied, "{stem}: {c:?}");
                    assert_eq!(shuffles(&p, [3, 3, 3, 3]), !premultiplied, "{stem}: {c:?}");
                    assert_eq!(premultiplies(&p), premultiplied, "{stem}: {c:?}");
                    assert!(!announces_rounding(&p), "{stem}: {c:?}");
                }
                Err(Error::Unsupported(u)) if !dual => {
                    assert_eq!(u.feature, Feature::DualSourceBlending, "{stem}");
                    assert_eq!(
                        u.needs,
                        Needs::AnyOf(vec![vec![Capability::Feature(VkFeature::DualSrcBlend)]]),
                        "{stem}"
                    );
                }
                Err(Error::Unsupported(u)) if u.needs != Needs::NotImplemented => {}
                Err(e) => panic!("{stem}: {c:?}: {e}"),
            }
            // Beside a second attachment the function writes: within the dual-source limit only.
            let max = dev.profile().limits().max_fragment_dual_src_attachments;
            match compile(c, true) {
                Ok(_) => assert!(dual && max >= 2, "{stem}"),
                Err(Error::Unsupported(u)) if dual && max < 2 => {
                    assert_eq!(u.feature, Feature::DualSourceBlending, "{stem}");
                    assert_eq!(
                        u.needs,
                        Needs::AnyOf(vec![vec![Capability::Limit {
                            limit: VkLimit::MaxFragmentDualSrcAttachments,
                            required: 2,
                            available: u64::from(max),
                        }]]),
                        "{stem}"
                    );
                }
                Err(Error::Unsupported(u)) if u.needs != Needs::NotImplemented => {}
                Err(e) => panic!("{stem}: {c:?} beside color 2: {e}"),
            }
        }
        // Blends no Vulkan factor gives on the image: the function computes Metal's 8-bit unorm
        // blend over the texel where the device orders framebuffer fetch, in the interlocked
        // storage form (color 0 is no attachment of the subpass) or over the attachment's input
        // attachment (`RotatedInputBlend`, no Vulkan blend there), and a device that orders it in
        // neither form names both orderings.
        let interlock = crate::lower::fragment::interlock_orders(dev.profile());
        let inputs = crate::lower::fragment::inputs_order(dev.profile());
        for c in [
            blended(F::DestinationAlpha, F::Zero, O::Add, O::Add),
            blended(F::BlendColor, F::Zero, O::Add, O::Add),
            blended(F::One, F::One, O::Add, O::Subtract),
            blended(F::One, F::One, O::Min, O::Max),
        ] {
            let d = descriptor(&bc, c, true);
            match dev.compile_render_pipeline(&d, &Budget::standard()) {
                Ok(p) if interlock => {
                    assert!(p.graphics.color_attachments[0].is_none(), "{stem}: {c:?}");
                    let stored = &p.graphics.storage_attachments;
                    assert!(
                        stored.iter().any(|s| s.color_index == 0
                            && s.writes
                            && s.format == vk::Format::R8G8B8A8_UNORM),
                        "{stem}: {c:?}: {stored:?}"
                    );
                    assert!(notes(&p, PrecisionNote::DeviceColorConversion), "{stem}");
                }
                Ok(p) if inputs => {
                    let state = p.graphics.color_attachments[0].expect("color 0");
                    assert_eq!(state.blend, None, "{stem}: {c:?}");
                    assert_eq!(state.format, vk::Format::R8G8B8A8_UNORM, "{stem}");
                    let fact = p.dependencies.iter().any(|d| {
                        matches!(
                            d.fact,
                            crate::pipeline::RuntimeFact::RotatedInputBlend { color: 0, .. }
                        )
                    });
                    assert!(fact, "{stem}: {c:?}");
                    assert!(notes(&p, PrecisionNote::DeviceColorConversion), "{stem}");
                }
                Ok(_) => panic!("{stem}: {c:?}: made on a device that orders no fetch"),
                Err(Error::Unsupported(u)) if !interlock && !inputs => {
                    assert_eq!(u.feature, Feature::FramebufferFetch, "{stem}: {c:?}");
                    assert_eq!(
                        u.needs,
                        Needs::AnyOf(vec![
                            vec![Capability::Feature(VkFeature::FragmentShaderPixelInterlock)],
                            vec![Capability::Feature(
                                VkFeature::RasterizationOrderColorAttachmentAccess
                            )],
                        ]),
                        "{stem}: {c:?}"
                    );
                    let at = u.location.as_ref().and_then(|l| l.attachment());
                    assert_eq!(at, Some(Attachment::Color(0)), "{stem}: {c:?}");
                }
                // A device that orders fetch but lacks what the form needs names it.
                Err(Error::Unsupported(u)) if u.needs != Needs::NotImplemented => {}
                other => panic!("{stem}: {c:?}: {:?}", other.map(|_| ())),
            }
        }
        // Alpha to coverage over the image.
        let mut d = descriptor(
            &bc,
            ColorAttachmentDescriptor::new(PixelFormat::ARGB8Unorm),
            true,
        );
        d.alpha_to_coverage = true;
        match dev.compile_render_pipeline(&d, &Budget::standard()) {
            Err(Error::Unsupported(u)) => {
                assert_eq!(
                    (u.feature, u.needs),
                    (Feature::Format, Needs::NotImplemented)
                );
                let at = u.location.as_ref().and_then(|l| l.attachment());
                assert_eq!(at, Some(Attachment::Color(0)), "{stem}");
            }
            other => panic!("{stem}: alpha to coverage: {:?}", other.map(|_| ())),
        }
    }
    assert!(checked > 0);
}
