//! Metal's private fixed-function render pipeline state (`PrivateRenderState`) over the owned
//! fixtures `engine/rt6_render_state` (Metal's fullscreen triangle and a white fragment
//! function) and `engine/a27_a4_atomic_load_only` (a kernel), on the `lavapipe` and
//! `moltenvk-m3` profiles at one and four samples:
//!
//! - Metal's defaults are the ones measured, and the properties that change nothing Metal draws
//!   keep the pipeline's key, modules and graphics state at any value.
//! - The sample mask is `GraphicsState::sample_mask`, its bits past the sample count dropped; it
//!   changes the key, never the modules, and only where it drops a sample; a pipeline without a
//!   fragment function ignores it, as Metal's does. A mask that drops one
//!   is refused where the fragment function could tell Metal's mask, applied after it runs, from
//!   Vulkan's (the owned fixtures `engine/sample_rate`, reading `[[sample_mask]]`, and
//!   `engine/multisample`, writing memory).
//! - The alpha test is run by the fragment function, against the encoder's reference as a push
//!   field; `Always` keeps the pipeline.
//! - The logic operation leaves the normalized, sRGB and integer attachments unblended (unwritten
//!   under a no-op), so copy and no-op draw on every device; the others take Vulkan's `logicOp`
//!   where the device has it (lavapipe), and are refused naming it where it does not
//!   (MoltenVK), and where Vulkan's operation is not Metal's.
//! - The sample coverage is Metal's masks by the window pixel's place in its quad, ANDed with the
//!   sample mask: alike at every place the Vulkan mask, else applied by the fragment function from
//!   its pixel.
//! - Each property not built refuses a value other than Metal's default `NotImplemented`, naming
//!   it (the coverage at 8 samples, where Metal was not measured). A compute pipeline's `needs_custom_border_color_samplers`, which changes nothing, keeps
//!   its key.
//!
//! `llvm-as` assembles the fixtures; a missing tool skips visibly. No GPU is used.

use std::path::Path;

use crate::budget::Budget;
use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::error::{Error, Feature, PrivateRenderField, RuntimeInput, Unsupported};
use crate::library::Library;
use crate::metal::PixelFormat;
use crate::pipeline::descriptor::{
    ColorAttachmentDescriptor, ComputePipelineDescriptor, FunctionDescriptor, PrivateRenderState,
    RenderPipelineDescriptor, StageSource,
};
use crate::pipeline::testing::{engine_dir, llvm_assemble, AssembleError};
use crate::pipeline::RenderPipeline;
use crate::spirv::testing::check_words_valid;

const STEMS: [&str; 2] = ["lavapipe", "moltenvk-m3"];

fn fixture(stem: &str) -> Option<Library> {
    match llvm_assemble(&engine_dir().join(format!("{stem}.ll"))) {
        Ok(b) => Some(Library::from_air(&b).expect("library")),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {stem}.ll: {why}"),
    }
}

fn device(stem: &str) -> Device {
    device_with(stem, DevicePolicy::default())
}

fn device_with(stem: &str, policy: DevicePolicy) -> Device {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles")
        .join(format!("{stem}.json"));
    let text = std::fs::read_to_string(path).expect("profile");
    let description = DeviceDescription::from_khronos_json(&text, None).expect("description");
    Device::new(description, policy).expect("device")
}

/// The fullscreen triangle drawn white into one `RGBA8Unorm` attachment at `samples`.
fn render(lib: &Library, samples: u32) -> RenderPipelineDescriptor {
    let stage = |name: &str| {
        StageSource::Function(FunctionDescriptor::new(
            lib.function(name).expect("function"),
        ))
    };
    let mut d = RenderPipelineDescriptor::new(stage("rt6_full"));
    d.fragment = Some(stage("rt6_white"));
    d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    d.raster_sample_count = samples;
    d
}

fn compile(dev: &Device, d: &RenderPipelineDescriptor, what: &str) -> RenderPipeline {
    dev.compile_render_pipeline(d, &Budget::standard())
        .unwrap_or_else(|e| panic!("{what}: {e}"))
}

fn same_modules(a: &RenderPipeline, b: &RenderPipeline) -> bool {
    a.stages.len() == b.stages.len()
        && a.stages
            .iter()
            .zip(&b.stages)
            .all(|(x, y)| x.stage == y.stage && x.spirv == y.spirv)
}

fn refused(field: PrivateRenderField) -> Error {
    Unsupported::not_implemented(Feature::PrivateRenderState(field)).into()
}

#[test]
fn defaults_and_properties_that_change_nothing_keep_the_pipeline() {
    // Metal's defaults, as an M4 answers a new descriptor's.
    let s = PrivateRenderState::new();
    assert_eq!(s.sample_mask, u64::MAX);
    assert_eq!(s.sample_coverage, 1.0);
    assert_eq!(s.clip_distance_enable_mask, 0xff);
    assert_eq!(
        (s.alpha_test_function, s.logic_operation),
        (0, 0),
        "raw enum values"
    );
    assert_eq!(
        (
            s.vertex_depth_compare_clamp_mask,
            s.fragment_depth_compare_clamp_mask
        ),
        (0, 0)
    );
    assert!(![
        s.sample_coverage_invert,
        s.alpha_test_enabled,
        s.logic_operation_enabled,
        s.point_smooth_enabled,
        s.point_coord_lower_left,
        s.point_size_output_vs,
        s.two_side_enabled,
        s.depth_stencil_write_disabled,
        s.needs_custom_border_color_samplers,
        s.open_gl_mode_enabled,
    ]
    .contains(&true));
    assert_eq!(
        (s.vertex_amplification_mode, s.explicit_visibility_group_id),
        (0, u32::MAX)
    );
    assert_eq!(PrivateRenderState::default(), s);
    let Some(lib) = fixture("rt6_render_state") else {
        return;
    };
    for stem in STEMS {
        let dev = device(stem);
        for samples in [1, 4] {
            let what = format!("{stem} {samples}x");
            let base = compile(&dev, &render(&lib, samples), &what);
            assert_eq!(base.graphics.sample_mask, None, "{what}");
            let mut d = render(&lib, samples);
            let p = &mut d.private;
            p.sample_coverage_invert = true;
            p.alpha_test_function = 3;
            p.clip_distance_enable_mask = 0;
            p.point_coord_lower_left = true;
            p.point_size_output_vs = true;
            p.two_side_enabled = true;
            p.vertex_depth_compare_clamp_mask = u32::MAX;
            p.depth_stencil_write_disabled = true;
            p.needs_custom_border_color_samplers = true;
            p.open_gl_mode_enabled = true;
            p.vertex_amplification_mode = 1;
            // A pipeline that does not amplify.
            p.explicit_visibility_group_id = 3;
            // Every sample's bit set, and bits past the sample count cleared or set.
            p.sample_mask = ((1 << samples) - 1) | 0xf0_0000_0000;
            let kept = compile(&dev, &d, &what);
            assert_eq!(kept.key, base.key, "{what}");
            assert_eq!(kept.graphics, base.graphics, "{what}");
            assert!(same_modules(&kept, &base), "{what}");
        }
    }
}

#[test]
fn the_sample_mask_is_its_bits_within_the_sample_count() {
    let Some(lib) = fixture("rt6_render_state") else {
        return;
    };
    let cases: [(u32, u64, Option<u32>); 6] = [
        (4, 0x5, Some(0x5)),
        (4, 0, Some(0)),
        (4, 0xffff_fff5, Some(0x5)),
        (4, 0x1_0000_000f, None),
        (1, 0x2, Some(0)),
        (1, 0x3, None),
    ];
    for stem in STEMS {
        let dev = device(stem);
        for (samples, mask, effective) in cases {
            let what = format!("{stem} {samples}x {mask:#x}");
            let base = compile(&dev, &render(&lib, samples), &what);
            let mut d = render(&lib, samples);
            d.private.sample_mask = mask;
            let masked = compile(&dev, &d, &what);
            assert_eq!(masked.graphics.sample_mask, effective, "{what}");
            assert_eq!(masked.key == base.key, effective.is_none(), "{what}");
            assert!(same_modules(&masked, &base), "{what}");
            let mut graphics = masked.graphics.clone();
            graphics.sample_mask = None;
            assert_eq!(graphics, base.graphics, "{what}");
        }
        // Without a fragment function Metal ignores the mask.
        for samples in [1, 4] {
            let what = format!("{stem} {samples}x without a fragment function");
            let mut d = render(&lib, samples);
            d.fragment = None;
            d.depth_format = Some(PixelFormat::Depth32Float);
            let base = compile(&dev, &d, &what);
            d.private.sample_mask = 0;
            let masked = compile(&dev, &d, &what);
            assert_eq!(masked.graphics.sample_mask, None, "{what}");
            assert_eq!(masked.key, base.key, "{what}");
        }
    }
}

/// The storage classes of the variables `words` decorates `BuiltIn SampleMask` (1 `Input`, 3
/// `Output`), sorted.
fn sample_mask_variables(words: &[u32]) -> Vec<u32> {
    const DECORATE: u32 = 71;
    const VARIABLE: u32 = 59;
    const BUILT_IN: u32 = 11;
    const SAMPLE_MASK: u32 = 20;
    let (mut ids, mut classes) = (Vec::new(), Vec::new());
    let mut i = 5;
    while i < words.len() {
        let (op, n) = (words[i] & 0xffff, (words[i] >> 16) as usize);
        let w = &words[i..i + n.max(1)];
        if op == DECORATE && n >= 4 && w[2] == BUILT_IN && w[3] == SAMPLE_MASK {
            ids.push(w[1]);
        }
        if op == VARIABLE && n >= 4 && ids.contains(&w[2]) {
            classes.push(w[3]);
        }
        i += n.max(1);
    }
    classes.sort();
    classes
}

#[test]
fn a_function_that_could_tell_applies_the_mask_itself() {
    let (Some(lib), Some(rate), Some(multi), Some(early)) = (
        fixture("rt6_render_state"),
        fixture("sample_rate"),
        fixture("multisample"),
        fixture("cfb_depth_never"),
    ) else {
        return;
    };
    let stage = |lib: &Library, name: &str| {
        StageSource::Function(FunctionDescriptor::new(
            lib.function(name).expect("function"),
        ))
    };
    // Reads `[[sample_mask]]`: Metal's is the rasterized coverage, unmasked.
    let mut reads = RenderPipelineDescriptor::new(stage(&lib, "rt6_full"));
    reads.fragment = Some(stage(&rate, "sample_rate_coverage"));
    reads.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA32Uint));
    // Writes memory: Metal runs it for the masked samples too.
    let mut writes = RenderPipelineDescriptor::new(stage(&multi, "multisample_vertex"));
    writes.fragment = Some(stage(&multi, "multisample_counter"));
    writes.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA16Float));
    // Early fragment tests: Metal's mask keeps every sample's depth and stencil.
    let mut tests = RenderPipelineDescriptor::new(stage(&lib, "rt6_full"));
    tests.fragment = Some(stage(&early, "cfb_dn_red_early"));
    tests.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    tests.depth_format = Some(PixelFormat::Depth32Float);
    // (name, the function's own SampleMask variables).
    let cases = [
        ("reads", &reads, vec![1]),
        ("writes", &writes, vec![]),
        ("early tests", &tests, vec![]),
    ];
    for stem in STEMS {
        let dev = device(stem);
        for (name, d, own) in &cases {
            let what = format!("{stem} {name}");
            let base = compile(&dev, d, &what);
            assert_eq!(base.graphics.sample_mask, None, "{what}");
            // A mask keeping the pipeline's one sample changes nothing.
            let mut kept = (*d).clone();
            kept.private.sample_mask = 0x1;
            let kept = compile(&dev, &kept, &what);
            assert_eq!(kept.key, base.key, "{what}");
            assert!(same_modules(&kept, &base), "{what}");
            let mut masked = (*d).clone();
            masked.private.sample_mask = 0;
            let p = compile(&dev, &masked, &what);
            // No Vulkan mask: the fragment function writes its `SampleMask` output.
            assert_eq!(p.graphics.sample_mask, None, "{what}");
            let fragment = p
                .stages
                .iter()
                .find(|m| m.stage == crate::vk::ShaderStage::FRAGMENT)
                .expect("fragment");
            let mut want = own.clone();
            want.push(3);
            assert_eq!(sample_mask_variables(&fragment.spirv), want, "{what}");
            for m in &p.stages {
                check_words_valid(&m.spirv, &p.requirements, dev.profile())
                    .unwrap_or_else(|e| panic!("{what} {:?}: L3: {e}", m.stage));
            }
            // The keys a consumer looks the pipeline up by name the masked compile.
            let keys = dev
                .render_pipeline_key(&masked, &Budget::standard())
                .unwrap_or_else(|e| panic!("{what}: {e}"));
            assert_eq!(keys.pipeline, p.key, "{what}");
            let inputs = keys.fragment_inputs.expect("fragment inputs");
            assert_eq!(inputs.sample_mask, Some([0; 4]), "{what}");
            // The function compiled without the mask, under the inputs the unmasked pipeline
            // gives it, is compiled for the pipeline again.
            let unmasked = dev
                .render_pipeline_key(d, &Budget::standard())
                .unwrap_or_else(|e| panic!("{what}: {e}"))
                .fragment_inputs
                .expect("fragment inputs");
            let Some(StageSource::Function(f)) = &d.fragment else {
                unreachable!("a function fragment stage")
            };
            let mut compiled = masked.clone();
            compiled.fragment = Some(StageSource::Compiled(
                dev.compile_function(f, &unmasked, &Budget::standard())
                    .unwrap_or_else(|e| panic!("{what}: {e}")),
            ));
            match dev.compile_render_pipeline(&compiled, &Budget::standard()) {
                Err(e) => assert_eq!(e, Error::NeedsInput(RuntimeInput::SampleMask), "{what}"),
                Ok(_) => panic!("{what}: compiled"),
            }
        }
    }
}

/// The `SampleMask` output's constant operands in `words`: the masks a function applies itself
/// (the one mask, or the four masks packed a byte each where they differ between a quad's
/// pixels) are among them.
fn constants(words: &[u32]) -> Vec<u32> {
    const CONSTANT: u32 = 43;
    let mut out = Vec::new();
    let mut i = 5;
    while i < words.len() {
        let (op, n) = (words[i] & 0xffff, (words[i] >> 16) as usize);
        if op == CONSTANT && n == 4 {
            out.push(words[i + 3]);
        }
        i += n.max(1);
    }
    out
}

/// The sample coverage: Metal's masks by the window pixel's place in its 2x2 quad
/// (`private_state::coverage_masks`), ANDed with the sample mask. Masks alike at every place are
/// the sample mask's (a Vulkan mask, the modules unchanged); masks that differ are applied by the
/// fragment function from its pixel (packed a byte each), with no Vulkan mask, whatever the
/// function observes; a coverage of 1 keeps the pipeline; a coverage keeping nothing at one
/// sample is a mask of 0.
#[test]
fn the_sample_coverage_is_applied_by_the_pixels_place_in_its_quad() {
    use crate::pipeline::private_state::coverage_masks;
    // Metal's rule at the measured counts (`pipeline::private_state`'s docs).
    assert_eq!(coverage_masks(0.5, 1), Some([0; 4]));
    assert_eq!(
        coverage_masks(f32::from_bits(0.5f32.to_bits() + 1), 1),
        Some([1; 4])
    );
    assert_eq!(coverage_masks(f32::NAN, 4), Some([0; 4]));
    assert_eq!(coverage_masks(f32::INFINITY, 4), Some([0xf; 4]));
    assert_eq!(coverage_masks(-1.0, 2), Some([0; 4]));
    // 0.3: code 77, 5 of a quad's 16 samples: (0, 0) two (samples 0, 3), the rest one.
    assert_eq!(coverage_masks(0.3, 4), Some([0x9, 0x1, 0x1, 0x1]));
    // 2 of a quad's 8: (0, 0) and (1, 1).
    assert_eq!(coverage_masks(0.3, 2), Some([0x1, 0, 0, 0x1]));
    // 0.0295: code 8, the first level.
    assert_eq!(coverage_masks(0.0295, 4), Some([0x1, 0, 0, 0]));
    assert_eq!(coverage_masks(0.029, 4), Some([0; 4]));
    assert_eq!(coverage_masks(0.5, 8), None);
    let Some(lib) = fixture("rt6_render_state") else {
        return;
    };
    for stem in STEMS {
        let dev = device(stem);
        for samples in [1, 4] {
            let what = format!("{stem} {samples}x");
            let base = compile(&dev, &render(&lib, samples), &what);
            // A coverage of 1 keeps the pipeline.
            let mut d = render(&lib, samples);
            d.private.sample_coverage = 1.0;
            assert_eq!(compile(&dev, &d, &what).key, base.key, "{what}");
            // Alike at every place: the Vulkan mask.
            let (alike, mask) = match samples {
                1 => (0.25, 0),
                _ => (0.25, 0x1),
            };
            d.private.sample_coverage = alike;
            let p = compile(&dev, &d, &what);
            assert_eq!(p.graphics.sample_mask, Some(mask), "{what}");
            assert!(same_modules(&p, &base), "{what}");
            assert_ne!(p.key, base.key, "{what}");
        }
        // Different at the places: the function applies them from its pixel.
        let what = format!("{stem} 4x 0.3");
        let base = compile(&dev, &render(&lib, 4), &what);
        let mut d = render(&lib, 4);
        d.private.sample_coverage = 0.3;
        // ANDed with the sample mask.
        d.private.sample_mask = 0b1110;
        let p = compile(&dev, &d, &what);
        assert_eq!(p.graphics.sample_mask, None, "{what}");
        assert_ne!(p.key, base.key, "{what}");
        let fragment = p
            .stages
            .iter()
            .find(|m| m.stage == crate::vk::ShaderStage::FRAGMENT)
            .expect("fragment");
        assert_eq!(sample_mask_variables(&fragment.spirv), vec![3], "{what}");
        // (0, 0) 0x8, (1, 0) 0x0, (0, 1) 0x0, (1, 1) 0x0, a byte each.
        assert!(constants(&fragment.spirv).contains(&0x8), "{what}");
        for m in &p.stages {
            check_words_valid(&m.spirv, &p.requirements, dev.profile())
                .unwrap_or_else(|e| panic!("{what} {:?}: L3: {e}", m.stage));
        }
        let keys = dev
            .render_pipeline_key(&d, &Budget::standard())
            .unwrap_or_else(|e| panic!("{what}: {e}"));
        assert_eq!(keys.pipeline, p.key, "{what}");
        let inputs = keys.fragment_inputs.expect("fragment inputs");
        assert_eq!(inputs.sample_mask, Some([0x8, 0, 0, 0]), "{what}");
        // 0.6 alone: code 153, 10 of 16: (0, 0) and (1, 1) three, the others two.
        d.private.sample_mask = u64::MAX;
        d.private.sample_coverage = 0.6;
        let inputs = dev
            .render_pipeline_key(&d, &Budget::standard())
            .unwrap_or_else(|e| panic!("{what}: {e}"))
            .fragment_inputs
            .expect("fragment inputs");
        assert_eq!(inputs.sample_mask, Some([0xb, 0x9, 0x9, 0xb]), "{what}");
        // The function compiled without the masks is compiled for the pipeline again.
        let unmasked = dev
            .render_pipeline_key(&render(&lib, 4), &Budget::standard())
            .unwrap_or_else(|e| panic!("{what}: {e}"))
            .fragment_inputs
            .expect("fragment inputs");
        let Some(StageSource::Function(f)) = &d.fragment else {
            unreachable!("a function fragment stage")
        };
        let mut compiled = d.clone();
        compiled.fragment = Some(StageSource::Compiled(
            dev.compile_function(f, &unmasked, &Budget::standard())
                .unwrap_or_else(|e| panic!("{what}: {e}")),
        ));
        match dev.compile_render_pipeline(&compiled, &Budget::standard()) {
            Err(e) => assert_eq!(e, Error::NeedsInput(RuntimeInput::SampleMask), "{what}"),
            Ok(_) => panic!("{what}: compiled"),
        }
        // Without a fragment function Metal ignores the coverage.
        let mut bare = render(&lib, 4);
        bare.fragment = None;
        bare.depth_format = Some(PixelFormat::Depth32Float);
        let base = compile(&dev, &bare, &what);
        bare.private.sample_coverage = 0.3;
        let p = compile(&dev, &bare, &what);
        assert_eq!((p.key, p.graphics.sample_mask), (base.key, None), "{what}");
    }
}

#[test]
fn each_property_not_built_refuses_a_value_other_than_metals_default() {
    use PrivateRenderField as F;
    let Some(lib) = fixture("rt6_render_state") else {
        return;
    };
    let cases: [(fn(&mut PrivateRenderState), PrivateRenderField); 2] = [
        (
            |p| {
                p.alpha_test_enabled = true;
                p.alpha_test_function = 8;
            },
            F::AlphaTest,
        ),
        (|p| p.point_smooth_enabled = true, F::PointSmooth),
    ];
    for stem in STEMS {
        let dev = device(stem);
        for (set, field) in cases {
            let mut d = render(&lib, 4);
            set(&mut d.private);
            match dev.compile_render_pipeline(&d, &Budget::standard()) {
                Err(e) => assert_eq!(e, refused(field), "{stem} {field:?}"),
                Ok(_) => panic!("{stem} {field:?}: compiled"),
            }
        }
        // A coverage at 8 samples, where Metal was not measured (a profile that draws at 8).
        if stem == "lavapipe" {
            for coverage in [0.5, f32::NAN] {
                let mut d = render(&lib, 8);
                d.private.sample_coverage = coverage;
                match dev.compile_render_pipeline(&d, &Budget::standard()) {
                    Err(e) => assert_eq!(e, refused(F::SampleCoverage), "{stem} {coverage}"),
                    Ok(_) => panic!("{stem}: coverage {coverage} at 8 samples compiled"),
                }
            }
        }
        // A visibility group where the pipeline amplifies.
        let mut d = render(&lib, 1);
        d.max_vertex_amplification_count = 2;
        d.private.explicit_visibility_group_id = 0;
        match dev.compile_render_pipeline(&d, &Budget::standard()) {
            Err(e) => assert_eq!(e, refused(F::ExplicitVisibilityGroup), "{stem}"),
            Ok(_) => panic!("{stem}: an amplified visibility group compiled"),
        }
        // A mode Metal's descriptor asserts at.
        let mut d = render(&lib, 1);
        d.private.vertex_amplification_mode = 2;
        match dev.compile_render_pipeline(&d, &Budget::standard()) {
            Err(Error::InvalidDescriptor(_)) => {}
            other => panic!("{stem}: vertexAmplificationMode 2: {other:?}"),
        }
    }
}

/// The private logic operation: every normalized, sRGB and integer attachment is unblended, and
/// under a no-op unwritten, which draws copy and no-op on both profiles with no Vulkan operation;
/// another operation (xor, raw 10, and raw 26, whose low four bits are xor's) is Vulkan's
/// `logicOp` on lavapipe, its requirements naming the feature, and refused naming the feature on
/// MoltenVK, which lacks it; a float attachment keeps its blend under copy and no-op and takes
/// Vulkan's operation unblended; an sRGB attachment under xor, a blended float attachment under xor
/// and a format whose operation Metal was not measured with are refused `NotImplemented`.
#[test]
fn the_logic_operation_draws_as_metal_s_or_is_refused() {
    use crate::device::requirements::VkFeature;
    use crate::error::{Attachment, Capability};
    use crate::metal::BlendFactor;
    use crate::vk;
    let Some(lib) = fixture("rt6_render_state") else {
        return;
    };
    let with = |pf: PixelFormat, blended: bool, op: Option<u64>| {
        let mut d = render(&lib, 1);
        let mut a = ColorAttachmentDescriptor::new(pf);
        if blended {
            a.blending_enabled = true;
            a.source_rgb_blend_factor = BlendFactor::One;
            a.destination_rgb_blend_factor = BlendFactor::One;
        }
        d.color_attachments[0] = Some(a);
        if let Some(op) = op {
            d.private.logic_operation_enabled = true;
            d.private.logic_operation = op;
        }
        d
    };
    let not_built = |index: u32| {
        Error::Unsupported(Unsupported::not_implemented(Feature::PrivateRenderState(
            PrivateRenderField::LogicOperation,
        )))
        .at_attachment(Attachment::Color(index))
    };
    for stem in STEMS {
        let dev = device(stem);
        let has = stem == "lavapipe";
        let state = |d: &RenderPipelineDescriptor, what: &str| {
            let p = compile(&dev, d, &format!("{stem} {what}"));
            let a = p.graphics.color_attachments[0].expect("attachment 0");
            (p.graphics.logic_op, a.blend.is_some(), a.write_mask, p)
        };
        // Copy and no-op on every device, the blend dropped, no-op writing nothing.
        let (op, blends, mask, _) = state(&with(PixelFormat::RGBA8Unorm, true, Some(2)), "copy");
        assert_eq!((op, blends), (None, false), "{stem} copy");
        assert_eq!(mask, vk::ColorComponentFlags::known_bits(), "{stem} copy");
        let (op, blends, mask, _) = state(&with(PixelFormat::RGBA8Unorm, true, Some(4)), "no-op");
        assert_eq!(
            (op, blends, mask),
            (None, false, vk::ColorComponentFlags::empty())
        );
        let (op, blends, _, _) = state(&with(PixelFormat::RGBA8UnormSRGB, true, Some(2)), "srgb");
        assert_eq!((op, blends), (None, false), "{stem} sRGB copy");
        // A float attachment keeps its blend under copy.
        let (op, blends, _, _) = state(&with(PixelFormat::RGBA16Float, true, Some(2)), "float");
        assert_eq!((op, blends), (None, true), "{stem} float copy");
        for raw in [10, 26] {
            for pf in [
                PixelFormat::RGBA8Unorm,
                PixelFormat::RGBA8Snorm,
                PixelFormat::RGB10A2Unorm,
                PixelFormat::R16Unorm,
                PixelFormat::RGBA16Float,
            ] {
                let blended = pf != PixelFormat::RGBA16Float;
                let d = with(pf, blended, Some(raw));
                match (has, dev.compile_render_pipeline(&d, &Budget::standard())) {
                    (true, Ok(p)) => {
                        assert_eq!(p.graphics.logic_op, Some(vk::LogicOp::XOR), "{pf:?}");
                        let a = p.graphics.color_attachments[0].expect("attachment 0");
                        assert!(a.blend.is_none(), "{stem} {pf:?} blends");
                        assert!(p.requirements.features.contains(&VkFeature::LogicOp));
                    }
                    (false, Err(e)) => assert_eq!(
                        e,
                        Error::Unsupported(Unsupported::needs(
                            Feature::PrivateRenderState(PrivateRenderField::LogicOperation),
                            Capability::Feature(VkFeature::LogicOp),
                        )),
                        "{stem} {pf:?}"
                    ),
                    (_, other) => panic!("{stem} {pf:?} xor: {other:?}"),
                }
            }
        }
        // Where Vulkan's operation is not Metal's, on either device.
        for (pf, blended) in [
            (PixelFormat::RGBA8UnormSRGB, false),
            (PixelFormat::RGBA16Float, true),
            (PixelFormat::RG11B10Float, false),
        ] {
            match dev.compile_render_pipeline(&with(pf, blended, Some(10)), &Budget::standard()) {
                Err(e) => assert_eq!(e, not_built(0), "{stem} {pf:?}"),
                Ok(_) => panic!("{stem} {pf:?} xor compiled"),
            }
        }
        // Each Vulkan operation keys a pipeline of its own, apart from copy's, whose attachments
        // are the same unblended ones.
        if has {
            let key = |raw: u64| {
                compile(&dev, &with(PixelFormat::RGBA8Unorm, true, Some(raw)), "key").key
            };
            let keys = [key(0), key(10), key(15), key(2)];
            for (i, a) in keys.iter().enumerate() {
                assert!(keys[i + 1..].iter().all(|b| a != b), "{stem} keys {keys:?}");
            }
        }
        // Off, the pipeline is the one without the property.
        let off = compile(&dev, &with(PixelFormat::RGBA8Unorm, true, None), "off");
        let mut d = with(PixelFormat::RGBA8Unorm, true, None);
        d.private.logic_operation = 10;
        let unset = compile(&dev, &d, "off, an operation set");
        assert!(off.graphics.logic_op.is_none() && unset.graphics.logic_op.is_none());
        assert_eq!(off.graphics, unset.graphics, "{stem}");
    }
}

/// The private alpha test: the fragment function runs it at its returns, reading the encoder's
/// reference as a push field for every function but `Never`, which kills every fragment; `Always`
/// keeps the pipeline as without the test. The keys a consumer looks the pipeline up by name the
/// tested compile, and a function compiled without the test is compiled for the pipeline again. A
/// pipeline without a fragment function, and a function past `Always`, are refused.
#[test]
fn the_alpha_test_runs_in_the_fragment_function() {
    use crate::metal::CompareFunction;
    use crate::pipeline::sym::{PushField, Stage};
    let Some(lib) = fixture("rt6_render_state") else {
        return;
    };
    let reads_reference = |p: &RenderPipeline| {
        p.interface.push.iter().any(|pl| {
            pl.stage == Stage::Fragment
                && pl
                    .fields
                    .iter()
                    .any(|f| f.field == PushField::AlphaTestReference)
        })
    };
    for stem in STEMS {
        let dev = device(stem);
        let plain = render(&lib, 1);
        let base = compile(&dev, &plain, stem);
        let untested = dev
            .render_pipeline_key(&plain, &Budget::standard())
            .unwrap_or_else(|e| panic!("{stem}: {e}"))
            .fragment_inputs
            .expect("fragment inputs");
        assert_eq!(untested.alpha_test, None, "{stem}");
        for raw in 0..=7u64 {
            let what = format!("{stem} function {raw}");
            let mut d = render(&lib, 1);
            d.private.alpha_test_enabled = true;
            d.private.alpha_test_function = raw;
            let p = compile(&dev, &d, &what);
            let function = CompareFunction::try_from(raw as u32).expect("function");
            if function == CompareFunction::Always {
                assert_eq!(p.key, base.key, "{what}");
                assert!(same_modules(&p, &base), "{what}");
                continue;
            }
            assert_ne!(p.key, base.key, "{what}");
            assert!(!same_modules(&p, &base), "{what}");
            assert_eq!(
                reads_reference(&p),
                function != CompareFunction::Never,
                "{what}"
            );
            for m in &p.stages {
                check_words_valid(&m.spirv, &p.requirements, dev.profile())
                    .unwrap_or_else(|e| panic!("{what} {:?}: L3: {e}", m.stage));
            }
            let keys = dev
                .render_pipeline_key(&d, &Budget::standard())
                .unwrap_or_else(|e| panic!("{what}: {e}"));
            assert_eq!(keys.pipeline, p.key, "{what}");
            let inputs = keys.fragment_inputs.expect("fragment inputs");
            assert_eq!(inputs.alpha_test, Some(function), "{what}");
            let Some(StageSource::Function(f)) = &d.fragment else {
                unreachable!("a function fragment stage")
            };
            let mut compiled = d.clone();
            compiled.fragment = Some(StageSource::Compiled(
                dev.compile_function(f, &untested, &Budget::standard())
                    .unwrap_or_else(|e| panic!("{what}: {e}")),
            ));
            match dev.compile_render_pipeline(&compiled, &Budget::standard()) {
                Err(e) => assert_eq!(e, Error::NeedsInput(RuntimeInput::AlphaTest), "{what}"),
                Ok(_) => panic!("{what}: compiled"),
            }
        }
        let mut bare = render(&lib, 1);
        bare.fragment = None;
        bare.private.alpha_test_enabled = true;
        bare.private.alpha_test_function = 1;
        match dev.compile_render_pipeline(&bare, &Budget::standard()) {
            Err(e) => assert_eq!(e, refused(PrivateRenderField::AlphaTest), "{stem}"),
            Ok(_) => panic!("{stem}: compiled without a fragment function"),
        }
    }
}

/// `fragmentDepthCompareClampMask` over the owned fixture `engine/pdc_compare_clamp`: a fragment
/// function clamps the reference of a comparison through `[[sampler(s)]]` where bit `s` is set and
/// through a constexpr sampler where bit 31 is, and records the samplers it compares through and
/// those it clamps; a mask over none of them keeps its module. A function compiled for another
/// mask over its samplers is compiled for the pipeline again, and one comparing through an
/// argument buffer's sampler is refused under a mask other than 0.
#[test]
fn the_depth_compare_clamp_mask_clamps_its_samplers_references() {
    use crate::pipeline::stage_io::DepthCompares;
    let Some(lib) = fixture("pdc_compare_clamp") else {
        return;
    };
    let pipeline = |fragment: &str, mask: u32| {
        let stage = |name: &str| {
            StageSource::Function(FunctionDescriptor::new(
                lib.function(name).expect("function"),
            ))
        };
        let mut d = RenderPipelineDescriptor::new(stage("pdc_vertex"));
        d.fragment = Some(stage(fragment));
        d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
        d.private.fragment_depth_compare_clamp_mask = mask;
        d
    };
    // (fragment, its comparisons' samplers, (mask, clamped)).
    let slot = 1u32 << 2;
    let cases: [(&str, u32, [(u32, u32); 4]); 2] = [
        (
            "pdc_slot",
            slot,
            [(0, 0), (slot, slot), (!slot, 0), (u32::MAX, slot)],
        ),
        (
            "pdc_constexpr",
            DepthCompares::CONSTEXPR,
            [
                (0, 0),
                (DepthCompares::CONSTEXPR, DepthCompares::CONSTEXPR),
                (0x7fff_ffff, 0),
                (u32::MAX, DepthCompares::CONSTEXPR),
            ],
        ),
    ];
    let devices = STEMS.map(|stem| (stem, device(stem)));
    for (stem, dev) in devices {
        for (fragment, samplers, masks) in cases {
            let base = compile(&dev, &pipeline(fragment, 0), stem);
            for (mask, clamped) in masks {
                let what = format!("{stem} {fragment} mask {mask:#x}");
                let d = pipeline(fragment, mask);
                let p = compile(&dev, &d, &what);
                assert_eq!(same_modules(&p, &base), clamped == 0, "{what}");
                for m in &p.stages {
                    check_words_valid(&m.spirv, &p.requirements, dev.profile())
                        .unwrap_or_else(|e| panic!("{what} {:?}: L3: {e}", m.stage));
                }
                let inputs = dev
                    .render_pipeline_key(&d, &Budget::standard())
                    .unwrap_or_else(|e| panic!("{what}: {e}"))
                    .fragment_inputs
                    .expect("fragment inputs");
                assert_eq!(inputs.depth_compare_clamp_mask, mask, "{what}");
                let Some(StageSource::Function(f)) = &d.fragment else {
                    unreachable!("a function fragment stage")
                };
                let own = dev
                    .compile_function(f, &inputs, &Budget::standard())
                    .unwrap_or_else(|e| panic!("{what}: {e}"));
                let compares = own.artifact().interface.io.depth_compares;
                assert_eq!(compares.samplers, samplers, "{what}");
                assert_eq!(compares.clamped, clamped, "{what}");
                // The function compiled without the mask clamps nothing.
                let unclamped = dev
                    .compile_function(
                        f,
                        &crate::pipeline::LoweringInputs {
                            depth_compare_clamp_mask: 0,
                            ..inputs
                        },
                        &Budget::standard(),
                    )
                    .unwrap_or_else(|e| panic!("{what}: {e}"));
                let mut compiled = d.clone();
                compiled.fragment = Some(StageSource::Compiled(unclamped));
                match (
                    dev.compile_render_pipeline(&compiled, &Budget::standard()),
                    clamped,
                ) {
                    (Ok(_), 0) => {}
                    (Err(e), 0) => panic!("{what}: {e}"),
                    (Err(e), _) => assert_eq!(
                        e,
                        Error::NeedsInput(RuntimeInput::DepthCompareClamp),
                        "{what}"
                    ),
                    (Ok(_), _) => panic!("{what}: compiled without the clamp"),
                }
            }
        }
        // An argument buffer's sampler: no mask bit names it.
        compile(&dev, &pipeline("pdc_argument", 0), stem);
        for mask in [1, slot, DepthCompares::CONSTEXPR] {
            match dev.compile_render_pipeline(&pipeline("pdc_argument", mask), &Budget::standard())
            {
                // Refused at the comparison, which the refusal locates.
                Err(Error::Unsupported(u)) => assert_eq!(
                    u.feature,
                    Feature::PrivateRenderState(PrivateRenderField::FragmentDepthCompareClamp),
                    "{stem} {mask:#x}"
                ),
                Err(e) => panic!("{stem} {mask:#x}: {e}"),
                Ok(_) => panic!("{stem} {mask:#x}: compiled"),
            }
        }
    }
}

#[test]
fn the_compute_custom_border_color_flag_keeps_the_pipeline() {
    let Some(kernels) = fixture("a27_a4_atomic_load_only") else {
        return;
    };
    let f = kernels.function("a4_atomic_load_only").expect("kernel");
    let mut d = ComputePipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(f)));
    for stem in STEMS {
        let dev = device(stem);
        d.needs_custom_border_color_samplers = false;
        let base = dev
            .compile_compute_pipeline(&d, &Budget::standard())
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        d.needs_custom_border_color_samplers = true;
        let kept = dev
            .compile_compute_pipeline(&d, &Budget::standard())
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert_eq!(kept.key, base.key, "{stem}");
    }
}
