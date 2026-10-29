//! Fragment functions that fetch a blended attachment of a format whose Metal blend arithmetic
//! the translation does not build (the owned fixtures `engine/fbl_fetch_blend8`, a four-lane output
//! to color 0 beside color 1 of mask none, under fast math and not, and
//! `engine/fbl_fetch_blend16_narrow`, outputs of one to four lanes to color 1 beside colors 0 and 2
//! of mask none).
//!
//! On every committed profile that orders fetch through the pixel interlock alone, each such
//! attachment the device can view as a storage image takes the storage form with the generic
//! float blend (`StorageView::generic_blend`): a 16-bit normalized, 8-bit signed normalized,
//! 10-bit, packed float, 32-bit float and `A8Unorm` attachment of a four-lane output, and an 8-bit
//! unorm or 32-bit float one of fewer lanes, under several blends and either math. Each names the blend in its `RuntimeFact::InterlockedFramebufferFetch`, records
//! `PrecisionNote::UnmeasuredFetchBlend`, pushes the blend color as `PushField::HalfBlendColor`
//! exactly where the blend reads it, and its module passes L3. Before, each was refused
//! `NotImplemented(FramebufferFetch)` at its color.
//!
//! The whole pipeline is made for `RGBA32Float` there, color 0 a written storage attachment and
//! no color attachment of the subpass, its stages passing L3 and its fragment stage carrying the
//! generic blend's note.
//! A blend that reads the second source, or any blend beside a second (index 1) output (the owned
//! fixture `engine/fbl_fetch_dual`), takes the generic blend too, a 16-bit float and an 8-bit
//! unorm attachment's among them; one of a function that writes no second output is refused as
//! Metal refuses its pipeline, `InvalidDescriptor(ColorAttachment)`. No GPU is used.
//!
//! `llvm-as` assembles the fixtures; a missing tool skips visibly.

use std::path::Path;
use std::sync::Arc;

use crate::budget::Budget;
use crate::device::{Device, DeviceDescription, DevicePolicy, DeviceProfile};
use crate::error::{DescriptorError, Error, Needs};
use crate::library::{FunctionArtifact, Library};
use crate::metal::{BlendFactor, BlendOperation, ColorWriteMask, PixelFormat};
use crate::pipeline::artifact::PrecisionNote;
use crate::pipeline::descriptor::{
    ColorAttachmentDescriptor, FunctionDescriptor, RenderPipelineDescriptor, StageSource,
};
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::link::RenderLinkInputs;
use crate::pipeline::lower_inputs::{BlendEquation, FramebufferFetchInputs, LoweringInputs};
use crate::pipeline::sym::PushField;
use crate::pipeline::testing::{
    compile_with, engine_dir, fragment_inputs, link_and_validate, AssembleError,
};
use crate::spirv::testing::{check_words_valid, committed_profiles};
use crate::vk;

/// Fixture `stem` assembled, under fast math or (`safe`) without; `None` (skipped) without
/// `llvm-as`.
fn assembled(stem: &str, safe: bool) -> Option<Vec<u8>> {
    let text =
        std::fs::read_to_string(engine_dir().join(format!("{stem}.ll"))).expect("the fixture");
    let option = "!{!\"air.compile.fast_math_enable\"}";
    assert_eq!(text.matches(option).count(), 1, "{stem}'s math mode");
    let text = match safe {
        true => text.replacen(option, "!{!\"air.compile.fast_math_disable\"}", 1),
        false => text,
    };
    match metal2vulkan_devtools::llvm_assemble_text(&text) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {stem}: {why}"),
    }
}

/// One over one minus the source alpha, added, for color and alpha.
const OVER: BlendEquation = BlendEquation {
    source_rgb: BlendFactor::One,
    destination_rgb: BlendFactor::OneMinusSourceAlpha,
    rgb_operation: BlendOperation::Add,
    source_alpha: BlendFactor::One,
    destination_alpha: BlendFactor::OneMinusSourceAlpha,
    alpha_operation: BlendOperation::Add,
};

/// Blends of several factors and operations, each with whether it reads the blend color.
fn equations() -> [(BlendEquation, bool); 4] {
    use BlendFactor as F;
    use BlendOperation as O;
    [
        (OVER, false),
        (
            BlendEquation {
                source_rgb: F::BlendColor,
                destination_rgb: F::OneMinusBlendAlpha,
                rgb_operation: O::ReverseSubtract,
                source_alpha: F::DestinationAlpha,
                destination_alpha: F::OneMinusSourceColor,
                alpha_operation: O::Subtract,
            },
            true,
        ),
        // The minimum and the maximum, which read no factor.
        (
            BlendEquation {
                rgb_operation: O::Max,
                alpha_operation: O::Min,
                ..OVER
            },
            false,
        ),
        (
            BlendEquation {
                source_rgb: F::SourceAlphaSaturated,
                destination_rgb: F::DestinationColor,
                ..OVER
            },
            false,
        ),
    ]
}

/// The pipeline: color `color` of `pf` blended by `eq` under `mask`, the colors below it and the
/// one above it `RGBA16Float` of mask none.
fn attachments(
    color: usize,
    pf: PixelFormat,
    eq: BlendEquation,
    mask: ColorWriteMask,
) -> RenderLinkInputs {
    let mut blended = ColorAttachmentDescriptor::new(pf);
    blended.blending_enabled = true;
    blended.source_rgb_blend_factor = eq.source_rgb;
    blended.destination_rgb_blend_factor = eq.destination_rgb;
    blended.rgb_blend_operation = eq.rgb_operation;
    blended.source_alpha_blend_factor = eq.source_alpha;
    blended.destination_alpha_blend_factor = eq.destination_alpha;
    blended.alpha_blend_operation = eq.alpha_operation;
    blended.write_mask = mask;
    assert_eq!(BlendEquation::of(&blended), Some(eq));
    let mut unwritten = ColorAttachmentDescriptor::new(PixelFormat::RGBA16Float);
    unwritten.write_mask = ColorWriteMask::NONE;
    let mut r = RenderLinkInputs::new();
    for c in 0..=color + 1 {
        r.color_attachments[c] = Some(unwritten);
    }
    r.color_attachments[color] = Some(blended);
    r
}

/// The lowering inputs of a function that reads color `color` in pipeline `r` on `p`.
fn state(color: u32, r: &RenderLinkInputs, p: &DeviceProfile) -> LoweringInputs {
    let mut inputs = fragment_inputs(r, p);
    inputs.framebuffer_fetch = Some(FramebufferFetchInputs::of(
        [color],
        &r.color_attachments,
        false,
        false,
        1,
    ));
    inputs
}

/// Color `color`'s storage form in `a`: whether it writes, and its blend.
fn fetched(a: &FunctionArtifact, color: u32) -> Option<(bool, Option<BlendEquation>)> {
    a.facts.iter().find_map(|f| match *f {
        RuntimeFact::InterlockedFramebufferFetch {
            color: c,
            writes,
            blend,
            ..
        } if c == color => Some((writes, blend)),
        _ => None,
    })
}

/// `compile_with`, `None` where the device lacks the format's storage (refused whatever the
/// blend).
fn compiled(
    bc: &[u8],
    entry: &str,
    p: &DeviceProfile,
    inputs: &LoweringInputs,
    what: &str,
) -> Option<Arc<FunctionArtifact>> {
    match compile_with(bc, entry, p, inputs) {
        Ok(a) => Some(a),
        Err(Error::Unsupported(u)) if u.needs != Needs::NotImplemented => {
            eprintln!("{what}: {u}");
            None
        }
        Err(e) => panic!("{what}: {e}"),
    }
}

/// Check `a`, compiled for a blend of color `color` by `eq` (reading the blend color where
/// `reads_color`): the generic blend in the storage form, noted, its module valid.
fn check_generic(
    a: &FunctionArtifact,
    p: &DeviceProfile,
    (color, eq, reads_color): (u32, BlendEquation, bool),
    what: &str,
) {
    assert_eq!(
        fetched(a, color),
        Some((true, Some(eq))),
        "{what}: {:?}",
        a.facts
    );
    assert!(
        a.precision.contains(&PrecisionNote::UnmeasuredFetchBlend),
        "{what}: {:?}",
        a.precision
    );
    let pushes = |field: PushField| a.interface.push.fields.iter().any(|x| x.field == field);
    assert_eq!(pushes(PushField::HalfBlendColor), reads_color, "{what}");
    link_and_validate(a, p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
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

#[test]
fn a_blended_fetch_of_another_format_computes_the_generic_blend() {
    let (Some(fast), Some(safe)) = (
        assembled("fbl_fetch_blend8", false),
        assembled("fbl_fetch_blend8", true),
    ) else {
        return;
    };
    let formats = [
        PixelFormat::RGBA32Float,
        PixelFormat::RGBA16Unorm,
        PixelFormat::RGBA16Snorm,
        PixelFormat::RGBA8Snorm,
        PixelFormat::RGB10A2Unorm,
        PixelFormat::BGR10A2Unorm,
        PixelFormat::RG11B10Float,
        PixelFormat::A8Unorm,
    ];
    // Each module and its entry, under fast math and not.
    let entries = [
        (&fast, "fbl8_over_loaded"),
        (&fast, "fbl8_over_loaded_float"),
        (&safe, "fbl8_over_product"),
    ];
    let mut checked = 0;
    // Profiles on which `RGBA32Float` took the form.
    let mut float32 = Vec::new();
    for (stem, p) in committed_profiles().expect("profiles") {
        if !crate::lower::fragment::interlock_orders(&p) {
            continue;
        }
        for pf in formats {
            for (eq, reads_color) in equations() {
                let r = attachments(0, pf, eq, ColorWriteMask::ALL);
                let inputs = state(0, &r, &p);
                for &(bc, entry) in &entries {
                    let what = format!("{stem} {pf:?} {entry} {eq:?}");
                    let Some(a) = compiled(bc, entry, &p, &inputs, &what) else {
                        continue;
                    };
                    check_generic(&a, &p, (0, eq, reads_color), &what);
                    checked += 1;
                    if pf == PixelFormat::RGBA32Float && !float32.contains(&stem) {
                        float32.push(stem.clone());
                    }
                }
            }
        }
    }
    assert!(checked > 0, "no interlock profile took the generic blend");
    assert!(
        float32.iter().any(|s| s.starts_with("moltenvk")),
        "no MoltenVK profile took RGBA32Float's form: {float32:?}"
    );
    // The whole pipeline, on each profile that took `RGBA32Float`'s form.
    for stem in float32 {
        let dev = device(&stem);
        let function = |name: &str| {
            StageSource::Function(FunctionDescriptor::new(
                Library::from_air(&fast)
                    .expect("library")
                    .function(name)
                    .expect("function"),
            ))
        };
        let what = format!("{stem}: the pipeline");
        let mut d = RenderPipelineDescriptor::new(function("fbl8_vertex"));
        d.fragment = Some(function("fbl8_over_loaded"));
        let r = attachments(0, PixelFormat::RGBA32Float, OVER, ColorWriteMask::ALL);
        d.color_attachments[0] = r.color_attachments[0];
        d.color_attachments[1] = r.color_attachments[1];
        let pipe = dev
            .compile_render_pipeline(&d, &Budget::standard())
            .unwrap_or_else(|e| panic!("{what}: {e}"));
        let storage: Vec<(u32, bool)> = (pipe.graphics.storage_attachments.iter())
            .map(|u| (u.color_index, u.writes))
            .collect();
        assert_eq!(storage, [(0, true)], "{what}");
        assert!(pipe.graphics.color_attachments[0].is_none(), "{what}");
        let fragment = (pipe.stages.iter())
            .find(|s| s.stage == vk::ShaderStage::FRAGMENT)
            .expect("a fragment stage");
        // The shader computes the blend, as the device's blend unit would, and says so.
        assert!(
            fragment
                .precision_notes
                .contains(&PrecisionNote::UnmeasuredFetchBlend),
            "{what}: {:?}",
            fragment.precision_notes
        );
        for s in &pipe.stages {
            check_words_valid(&s.spirv, &pipe.requirements, dev.profile())
                .unwrap_or_else(|e| panic!("{what}: L3: {e}"));
        }
    }
}

/// Outputs of fewer than four lanes into the one- and two-channel 8-bit unorm and 32-bit float
/// formats (an 8-bit unorm blend of such an output is not measured), and a partial write mask:
/// the generic blend, under fast math and not.
#[test]
fn a_narrow_output_takes_the_generic_blend() {
    let (Some(fast), Some(safe)) = (
        assembled("fbl_fetch_blend16_narrow", false),
        assembled("fbl_fetch_blend16_narrow", true),
    ) else {
        return;
    };
    use ColorWriteMask as M;
    // Each format, the entries of its channels' lanes, and its write masks.
    let cases = [
        (
            PixelFormat::R8Unorm,
            ["fbl_narrow_half", "fbl_narrow_float"],
            vec![M::ALL],
        ),
        (
            PixelFormat::RG8Unorm,
            ["fbl_narrow_half2", "fbl_narrow_float2"],
            vec![M::ALL, M::GREEN],
        ),
        (
            PixelFormat::R32Float,
            ["fbl_narrow_half", "fbl_narrow_float"],
            vec![M::ALL],
        ),
        (
            PixelFormat::RG32Float,
            ["fbl_narrow_half2", "fbl_narrow_float2"],
            vec![M::ALL, M::RED],
        ),
    ];
    let mut checked = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        if !crate::lower::fragment::interlock_orders(&p) {
            continue;
        }
        for (pf, entries, masks) in &cases {
            for mask in masks {
                for (eq, reads_color) in equations() {
                    let r = attachments(1, *pf, eq, *mask);
                    let inputs = state(1, &r, &p);
                    for bc in [&fast, &safe] {
                        for entry in entries {
                            let what = format!("{stem} {pf:?} {mask:?} {entry} {eq:?}");
                            let Some(a) = compiled(bc, entry, &p, &inputs, &what) else {
                                continue;
                            };
                            check_generic(&a, &p, (1, eq, reads_color), &what);
                            checked += 1;
                        }
                    }
                }
            }
        }
    }
    assert!(checked > 0, "no interlock profile took the storage form");
}

/// A blend of the second source, or any blend beside a second (index 1) output, of a function that
/// reads the attachment (the owned fixture `engine/fbl_fetch_dual`, under fast math and not): the
/// generic blend, which reads the second output where the blend names it, into a 32-bit float, a
/// 16-bit normalized, a 16-bit float and an 8-bit unorm attachment alike (Metal makes and draws
/// such pipelines; its blend of a 16-bit float or 8-bit unorm one with the second source is not
/// measured). Noted, its module valid.
#[test]
fn a_second_source_blend_takes_the_generic_blend() {
    let (Some(fast), Some(safe)) = (
        assembled("fbl_fetch_dual", false),
        assembled("fbl_fetch_dual", true),
    ) else {
        return;
    };
    use BlendFactor as F;
    let formats = [
        PixelFormat::RGBA32Float,
        PixelFormat::RGBA16Unorm,
        PixelFormat::RGBA16Float,
        PixelFormat::RGBA8Unorm,
    ];
    let equations = [
        BlendEquation {
            source_rgb: F::Source1Color,
            destination_rgb: F::OneMinusSource1Color,
            ..OVER
        },
        BlendEquation {
            source_alpha: F::Source1Alpha,
            destination_alpha: F::OneMinusSource1Alpha,
            ..OVER
        },
        OVER,
    ];
    let mut checked = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        if !crate::lower::fragment::interlock_orders(&p) {
            continue;
        }
        for pf in formats {
            for eq in equations {
                let r = attachments(0, pf, eq, ColorWriteMask::ALL);
                let inputs = state(0, &r, &p);
                for bc in [&fast, &safe] {
                    for entry in ["fbd_dual_half", "fbd_dual_float"] {
                        let what = format!("{stem} {pf:?} {entry} {eq:?}");
                        let Some(a) = compiled(bc, entry, &p, &inputs, &what) else {
                            continue;
                        };
                        check_generic(&a, &p, (0, eq, false), &what);
                        checked += 1;
                    }
                }
            }
        }
    }
    assert!(checked > 0, "no interlock profile took the storage form");
}

/// A blend of the second source of a function that writes no second output: Metal refuses the
/// pipeline ("Fragment shader does not write to render target color(0), index(1) that is required
/// for blending", plain Metal on the M3 Max under macOS 27.0.1, a fetching function and a plain
/// one, `half4` and `float4`), and so is it refused, `InvalidDescriptor(ColorAttachment)` of color
/// 0, whatever the format.
#[test]
fn a_second_source_blend_without_its_output_is_refused_as_metal_refuses_it() {
    let Some(fast) = assembled("fbl_fetch_blend8", false) else {
        return;
    };
    use BlendFactor as F;
    let cases = [
        (
            PixelFormat::RGBA32Float,
            BlendEquation {
                source_rgb: F::Source1Color,
                ..OVER
            },
        ),
        (
            PixelFormat::RGBA16Unorm,
            BlendEquation {
                destination_alpha: F::OneMinusSource1Alpha,
                ..OVER
            },
        ),
        (
            PixelFormat::RGBA16Float,
            BlendEquation {
                destination_rgb: F::Source1Alpha,
                ..OVER
            },
        ),
    ];
    let mut checked = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        if !crate::lower::fragment::interlock_orders(&p) {
            continue;
        }
        for (pf, eq) in cases {
            let r = attachments(0, pf, eq, ColorWriteMask::ALL);
            let inputs = state(0, &r, &p);
            let what = format!("{stem} {pf:?} {eq:?}");
            match compile_with(&fast, "fbl8_over_loaded", &p, &inputs) {
                Err(Error::InvalidDescriptor(DescriptorError::ColorAttachment { index: 0 })) => {
                    checked += 1;
                }
                Err(Error::Unsupported(u)) => eprintln!("{what}: {u}"),
                other => panic!("{what}: {:?}", other.map(|a| a.facts.clone())),
            }
        }
    }
    assert!(checked > 0, "no interlock profile refused");
}
