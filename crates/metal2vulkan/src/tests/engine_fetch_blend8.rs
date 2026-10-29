//! A fragment function that fetches a blended 8-bit unorm attachment, in the shape of a
//! CoreAnimation composite of the x86 macOS guest (the owned fixtures `engine/fbl_fetch_blend8`,
//! under fast math, and `engine/fbl_fetch_blend8_safe`, without): color 0 blended with one over
//! one minus the source alpha for color and alpha, color 1 `RGBA16Float` of mask none, each
//! function reading color 0.
//!
//! On every committed profile that orders fetch through the pixel interlock alone, each function
//! compiles with color 0 in the storage form for each 8-bit unorm format (`BGRA8Unorm`,
//! `RGBA8Unorm`, `RG8Unorm`, `R8Unorm`) and the blend it computes
//! (`RuntimeFact::InterlockedFramebufferFetch` names the blend, rounded without fast math), and
//! its module passes L3. Every output, the fast-math product of `fbl8_over_product` too, blends
//! as Metal does (Metal clamps the output before the blend, so nothing folds into it): no
//! function records `HalfOutputFastMathBlend` or `HalfOutputPrecision`. The storage image's store
//! converts the blend's result to its code, the device's conversion: every function records
//! `DeviceColorConversion`; an sRGB attachment's encode and decode are the device's
//! (`DevicePackConversion`).
//! The guest's format (`BGRA8Unorm`) takes the form on every such profile (ANV's through its
//! RGBA-order twin, the shader exchanging red and blue around the blend), and the guest's whole
//! pipeline is made there, color 0 a written storage attachment and no color attachment of the
//! subpass, its stages passing L3. Before the shader computed the
//! blend, each was refused `NotImplemented(FramebufferFetch)` at color 0. Other blends (the blend
//! color's among them) and write masks that keep channels compile the same way; a blend that reads
//! the second source of a function that writes none is refused as Metal refuses its pipeline
//! (`InvalidDescriptor(ColorAttachment)`). No GPU is used.
//!
//! `llvm-as` assembles the fixtures; a missing tool skips visibly.

use std::path::Path;

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
use crate::pipeline::testing::{
    compile_with, engine_dir, fragment_inputs, link_and_validate, AssembleError,
};
use crate::spirv::testing::{check_words_valid, committed_profiles};
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

/// The guest's blend: one over one minus the source alpha, added, for color and alpha.
const OVER: BlendEquation = BlendEquation {
    source_rgb: BlendFactor::One,
    destination_rgb: BlendFactor::OneMinusSourceAlpha,
    rgb_operation: BlendOperation::Add,
    source_alpha: BlendFactor::One,
    destination_alpha: BlendFactor::OneMinusSourceAlpha,
    alpha_operation: BlendOperation::Add,
};

/// The 8-bit unorm formats.
const FORMATS: [PixelFormat; 4] = [
    PixelFormat::BGRA8Unorm,
    PixelFormat::RGBA8Unorm,
    PixelFormat::RG8Unorm,
    PixelFormat::R8Unorm,
];

/// The notes a draw announces where the blend's arithmetic may be another than Metal's.
const NOTES: [PrecisionNote; 2] = [
    PrecisionNote::HalfOutputFastMathBlend,
    PrecisionNote::HalfOutputPrecision,
];

/// The attachments: color 0 of `pf` blended by `eq` under write mask `mask`, color 1 of mask none.
fn attachments(pf: PixelFormat, eq: BlendEquation, mask: ColorWriteMask) -> RenderLinkInputs {
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
    r.color_attachments[0] = Some(blended);
    r.color_attachments[1] = Some(unwritten);
    r
}

/// The lowering inputs of a function that reads color 0 in [`attachments`] on `p`.
fn state(
    p: &DeviceProfile,
    pf: PixelFormat,
    eq: BlendEquation,
    mask: ColorWriteMask,
) -> LoweringInputs {
    let r = attachments(pf, eq, mask);
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

/// Color 0's storage form in `a`: whether it writes, its blend, and whether that is rounded.
fn fetched(a: &FunctionArtifact) -> Option<(bool, Option<BlendEquation>, bool)> {
    a.facts.iter().find_map(|f| match f {
        RuntimeFact::InterlockedFramebufferFetch {
            color: 0,
            writes,
            blend,
            rounded,
            ..
        } => Some((*writes, *blend, *rounded)),
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
/// format without the storage access the form needs).
fn device_gap(e: &Error) -> bool {
    matches!(e, Error::Unsupported(u) if u.needs != Needs::NotImplemented)
}

#[test]
fn a_blended_eight_bit_unorm_fetch_computes_metals_blend() {
    let (Some(fast), Some(safe)) = (
        assembled("fbl_fetch_blend8"),
        assembled("fbl_fetch_blend8_safe"),
    ) else {
        return;
    };
    // Each fixture's functions, and whether its blend is rounded.
    let entries = [
        (&fast, "fbl8_over_loaded", false),
        (&fast, "fbl8_over_loaded_float", false),
        (&fast, "fbl8_over_product", false),
        (&safe, "fbl8_safe_loaded", true),
        (&safe, "fbl8_safe_loaded_float", true),
        (&safe, "fbl8_safe_product", true),
    ];
    let mut checked = 0;
    // An ANV profile compiled the guest's shape.
    let mut anv = false;
    for (stem, p) in committed_profiles().expect("profiles") {
        if !crate::lower::fragment::interlock_orders(&p) {
            continue;
        }
        for pf in FORMATS {
            let inputs = state(&p, pf, OVER, ColorWriteMask::ALL);
            for &(bc, entry, rounded) in &entries {
                let what = format!("{stem} {pf:?} {entry}");
                // The guest's format takes the form on every such profile, ANV's through its
                // RGBA-order twin; another may lack its storage there.
                let a = match compile_with(bc, entry, &p, &inputs) {
                    Ok(a) => a,
                    Err(e) if device_gap(&e) && pf != PixelFormat::BGRA8Unorm => {
                        eprintln!("{what}: {e}");
                        continue;
                    }
                    Err(e) => panic!("{what}: {e}"),
                };
                anv |= stem.contains("anv") && pf == PixelFormat::BGRA8Unorm;
                assert_eq!(
                    fetched(&a),
                    Some((true, Some(OVER), rounded)),
                    "{what}: {:?}",
                    a.facts
                );
                for note in NOTES {
                    assert!(!a.precision.contains(&note), "{what}: {:?}", a.precision);
                }
                assert!(
                    a.precision.contains(&PrecisionNote::DeviceColorConversion),
                    "{what}: {:?}",
                    a.precision
                );
                link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
                checked += 1;
            }
        }
        // The whole pipeline, as the guest makes it.
        let dev = device(&stem);
        let function = |name: &str| {
            StageSource::Function(FunctionDescriptor::new(
                Library::from_air(&fast)
                    .expect("library")
                    .function(name)
                    .expect("function"),
            ))
        };
        for entry in ["fbl8_over_loaded", "fbl8_over_product"] {
            let what = format!("{stem} {entry}: the pipeline");
            let mut d = RenderPipelineDescriptor::new(function("fbl8_vertex"));
            d.fragment = Some(function(entry));
            let r = attachments(PixelFormat::BGRA8Unorm, OVER, ColorWriteMask::ALL);
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
            for note in NOTES {
                assert!(
                    !fragment.precision_notes.contains(&note),
                    "{what}: {:?}",
                    fragment.precision_notes
                );
            }
            assert!(
                (fragment.precision_notes).contains(&PrecisionNote::DeviceColorConversion),
                "{what}: {:?}",
                fragment.precision_notes
            );
            for s in &pipe.stages {
                check_words_valid(&s.spirv, &pipe.requirements, dev.profile())
                    .unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            }
        }
    }
    assert!(checked > 0, "no interlock profile took the storage form");
    assert!(anv, "no ANV profile compiled the guest's shape");
}

/// A blended 8-bit four-channel sRGB attachment (`RGBA8Unorm_sRGB`, `BGRA8Unorm_sRGB`): on every
/// committed profile that orders fetch through the pixel interlock alone, each function compiles
/// in the storage form through the format's `R32_UINT` view with the blend it computes (Metal's
/// 8-bit unorm blend over the decoded texel, encoded: `extended_range::Blend::srgb8`), rounded
/// without fast math, and announces nothing, a write mask that keeps channels and the blend color
/// too; and its module passes L3. Before, each was refused `NotImplemented(FramebufferFetch)`.
/// The second source's blend (the owned fixture `engine/fbl_fetch_dual`, fast math), which Metal's
/// 8-bit unorm blend does not build, takes its rounded forms, noted `UnmeasuredFetchBlend`.
#[test]
fn a_blended_four_channel_srgb_fetch_computes_metals_blend() {
    let (Some(fast), Some(safe), Some(dual)) = (
        assembled("fbl_fetch_blend8"),
        assembled("fbl_fetch_blend8_safe"),
        assembled("fbl_fetch_dual"),
    ) else {
        return;
    };
    use BlendFactor as F;
    let entries = [
        (&fast, "fbl8_over_loaded", false),
        (&fast, "fbl8_over_loaded_float", false),
        (&safe, "fbl8_safe_loaded", true),
        (&safe, "fbl8_safe_loaded_float", true),
    ];
    let lerp = BlendEquation {
        source_rgb: F::BlendColor,
        destination_rgb: F::OneMinusBlendColor,
        ..OVER
    };
    let states = [
        (OVER, ColorWriteMask::ALL),
        (OVER, ColorWriteMask::RED.union(ColorWriteMask::ALPHA)),
        (lerp, ColorWriteMask::ALL),
    ];
    let second = BlendEquation {
        source_rgb: F::Source1Color,
        destination_rgb: F::OneMinusSource1Color,
        source_alpha: F::Source1Alpha,
        destination_alpha: F::OneMinusSource1Alpha,
        ..OVER
    };
    let mut checked = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        if !crate::lower::fragment::interlock_orders(&p) {
            continue;
        }
        for pf in [PixelFormat::RGBA8UnormSRGB, PixelFormat::BGRA8UnormSRGB] {
            for (eq, mask) in states {
                let inputs = state(&p, pf, eq, mask);
                for &(bc, entry, rounded) in &entries {
                    let what = format!("{stem} {pf:?} {eq:?} {mask:?} {entry}");
                    let a = compile_with(bc, entry, &p, &inputs)
                        .unwrap_or_else(|e| panic!("{what}: {e}"));
                    assert_eq!(fetched(&a), Some((true, Some(eq), rounded)), "{what}");
                    for note in NOTES.into_iter().chain([
                        PrecisionNote::UnmeasuredFetchBlend,
                        PrecisionNote::DeviceColorConversion,
                    ]) {
                        assert!(!a.precision.contains(&note), "{what}: {:?}", a.precision);
                    }
                    assert!(
                        a.precision.contains(&PrecisionNote::DevicePackConversion),
                        "{what}: {:?}",
                        a.precision
                    );
                    link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
                    checked += 1;
                }
            }
            let inputs = state(&p, pf, second, ColorWriteMask::ALL);
            for entry in ["fbd_dual_half", "fbd_dual_float"] {
                let what = format!("{stem} {pf:?} {entry}");
                let a = compile_with(&dual, entry, &p, &inputs)
                    .unwrap_or_else(|e| panic!("{what}: {e}"));
                assert_eq!(fetched(&a), Some((true, Some(second), false)), "{what}");
                assert!(
                    a.precision.contains(&PrecisionNote::UnmeasuredFetchBlend),
                    "{what}: {:?}",
                    a.precision
                );
                link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
                checked += 1;
            }
        }
    }
    assert!(checked > 0, "no interlock profile took the storage form");
}

/// Other blends and write masks of an 8-bit unorm attachment: on every committed profile that
/// orders fetch through the pixel interlock alone, each compiles in the storage form with the
/// blend it computes, rounded without fast math, and announces only the device's conversion of its
/// result (`DeviceColorConversion`); one reading the blend
/// color pushes it as `half` (`PushField::HalfBlendColor`), and its module passes L3.
#[test]
fn another_eight_bit_unorm_blend_or_mask_computes_metals_blend() {
    let (Some(fast), Some(safe)) = (
        assembled("fbl_fetch_blend8"),
        assembled("fbl_fetch_blend8_safe"),
    ) else {
        return;
    };
    use crate::pipeline::sym::PushField;
    use BlendFactor as F;
    use BlendOperation as O;
    let rgb = ColorWriteMask::RED
        .union(ColorWriteMask::GREEN)
        .union(ColorWriteMask::BLUE);
    // Each blend, its write mask, and whether it reads the blend color.
    let cases = [
        (
            BlendEquation {
                source_rgb: F::SourceAlpha,
                ..OVER
            },
            ColorWriteMask::ALL,
            false,
        ),
        (
            BlendEquation {
                destination_rgb: F::One,
                destination_alpha: F::One,
                ..OVER
            },
            ColorWriteMask::ALL,
            false,
        ),
        (
            BlendEquation {
                source_rgb: F::BlendColor,
                destination_rgb: F::OneMinusBlendAlpha,
                rgb_operation: O::ReverseSubtract,
                source_alpha: F::DestinationAlpha,
                destination_alpha: F::OneMinusSourceColor,
                alpha_operation: O::Subtract,
            },
            ColorWriteMask::ALL,
            true,
        ),
        (
            BlendEquation {
                rgb_operation: O::Max,
                alpha_operation: O::Min,
                ..OVER
            },
            ColorWriteMask::ALL,
            false,
        ),
        (OVER, rgb, false),
        (OVER, ColorWriteMask::BLUE, false),
        (OVER, ColorWriteMask::ALPHA, false),
        (
            BlendEquation {
                source_rgb: F::OneMinusDestinationAlpha,
                destination_rgb: F::SourceColor,
                rgb_operation: O::ReverseSubtract,
                ..OVER
            },
            ColorWriteMask::GREEN.union(ColorWriteMask::ALPHA),
            false,
        ),
    ];
    let entries = [
        (&fast, "fbl8_over_loaded", false),
        (&safe, "fbl8_safe_loaded", true),
    ];
    let mut checked = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        if !crate::lower::fragment::interlock_orders(&p) {
            continue;
        }
        for (eq, mask, reads_color) in cases {
            let inputs = state(&p, PixelFormat::BGRA8Unorm, eq, mask);
            for &(bc, entry, rounded) in &entries {
                let what = format!("{stem} {entry} {eq:?} {mask:?}");
                let a =
                    compile_with(bc, entry, &p, &inputs).unwrap_or_else(|e| panic!("{what}: {e}"));
                assert_eq!(
                    fetched(&a),
                    Some((true, Some(eq), rounded)),
                    "{what}: {:?}",
                    a.facts
                );
                for note in NOTES {
                    assert!(!a.precision.contains(&note), "{what}: {:?}", a.precision);
                }
                assert!(
                    a.precision.contains(&PrecisionNote::DeviceColorConversion),
                    "{what}: {:?}",
                    a.precision
                );
                let pushes =
                    |field: PushField| a.interface.push.fields.iter().any(|x| x.field == field);
                assert_eq!(pushes(PushField::HalfBlendColor), reads_color, "{what}");
                assert!(!pushes(PushField::BlendColor), "{what}");
                link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
                checked += 1;
            }
        }
    }
    assert!(checked > 0, "no interlock profile took the storage form");
}

/// A blend of an 8-bit unorm attachment that reads the second source, of a function that writes no
/// second output: Metal refuses the pipeline ("Fragment shader does not write to render target
/// color(0), index(1) that is required for blending", plain Metal on the M3 Max under macOS
/// 27.0.1), and on every committed profile that orders fetch through the pixel interlock alone it
/// is refused so, `InvalidDescriptor(ColorAttachment)` of color 0.
#[test]
fn a_second_source_blend_without_its_output_is_refused_as_metal_refuses_it() {
    let Some(fast) = assembled("fbl_fetch_blend8") else {
        return;
    };
    use BlendFactor as F;
    let cases = [
        BlendEquation {
            source_rgb: F::Source1Color,
            ..OVER
        },
        BlendEquation {
            destination_alpha: F::OneMinusSource1Alpha,
            ..OVER
        },
    ];
    let mut checked = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        if !crate::lower::fragment::interlock_orders(&p) {
            continue;
        }
        for eq in cases {
            let inputs = state(&p, PixelFormat::BGRA8Unorm, eq, ColorWriteMask::ALL);
            let what = format!("{stem} {eq:?}");
            match compile_with(&fast, "fbl8_over_loaded", &p, &inputs) {
                Err(Error::InvalidDescriptor(DescriptorError::ColorAttachment { index: 0 })) => {
                    checked += 1;
                }
                Err(e) if device_gap(&e) => eprintln!("{what}: {e}"),
                other => panic!("{what}: {:?}", other.map(|a| a.facts.clone())),
            }
        }
    }
    assert!(checked > 0, "no interlock profile refused");
}
