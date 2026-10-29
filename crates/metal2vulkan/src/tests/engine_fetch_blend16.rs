//! A fragment function that fetches a blended 16-bit float attachment, in the shape of a macOS 27
//! window-server composite (the owned fixture `engine/fbl_fetch_blend16`): color 0 `RGBA16Float`
//! blended with one over one minus the source alpha for color and alpha, color 1 `RGBA16Float` of
//! mask none, each function reading color 0 under fast math.
//!
//! On every committed profile that orders fetch through the pixel interlock alone, each function
//! compiles with color 0 in the storage form and the blend it computes
//! (`RuntimeFact::InterlockedFramebufferFetch` names the blend, not rounded), and its module
//! passes L3; the whole pipeline is made there, color 0 a written storage attachment and no color
//! attachment of the subpass, its stages passing L3. `fbl_over_loaded` (a loaded `half4`) and
//! `fbl_over_loaded_float` (a loaded `float4`) output what Metal's compiler cannot fold into its
//! blend, so the blend is Metal's exactly and neither records `HalfOutputFastMathBlend`;
//! `fbl_over_product`'s product Metal folds into the blend, so it records the note. The same module
//! stating no fast math (`air.compile.fast_math_disable`) computes the rounded forms, exact, noted
//! by none, rounded in the fact. Before the shader computed the blend, each was refused
//! `NotImplemented(FramebufferFetch)` at color 0. A NaN of the blend is the device's arithmetic's,
//! which the storage image's store converts: every module records `DeviceNanEncoding`. No GPU is
//! used.
//!
//! `llvm-as` assembles the fixture; a missing tool skips visibly.

use std::path::Path;

use crate::budget::Budget;
use crate::device::{Device, DeviceDescription, DevicePolicy, DeviceProfile};
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

/// The fixture's text.
fn text() -> String {
    std::fs::read_to_string(engine_dir().join("fbl_fetch_blend16.ll")).expect("the fixture")
}

/// `text` assembled; `None` (skipped) without `llvm-as`.
fn assembled(text: &str) -> Option<Vec<u8>> {
    match metal2vulkan_devtools::llvm_assemble_text(text) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected fbl_fetch_blend16: {why}"),
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

/// The attachments: color 0 blended by `eq` (the guest's [`OVER`]), color 1 of mask none.
fn attachments(eq: BlendEquation) -> RenderLinkInputs {
    let mut blended = ColorAttachmentDescriptor::new(PixelFormat::RGBA16Float);
    blended.blending_enabled = true;
    blended.source_rgb_blend_factor = eq.source_rgb;
    blended.destination_rgb_blend_factor = eq.destination_rgb;
    blended.rgb_blend_operation = eq.rgb_operation;
    blended.source_alpha_blend_factor = eq.source_alpha;
    blended.destination_alpha_blend_factor = eq.destination_alpha;
    blended.alpha_blend_operation = eq.alpha_operation;
    assert_eq!(BlendEquation::of(&blended), Some(eq));
    let mut unwritten = ColorAttachmentDescriptor::new(PixelFormat::RGBA16Float);
    unwritten.write_mask = ColorWriteMask::NONE;
    let mut r = RenderLinkInputs::new();
    r.color_attachments[0] = Some(blended);
    r.color_attachments[1] = Some(unwritten);
    r
}

/// The lowering inputs of a function that reads color 0 in [`attachments`]`(eq)` on `p`.
fn state(p: &DeviceProfile, eq: BlendEquation) -> LoweringInputs {
    let r = attachments(eq);
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

#[test]
fn a_blended_sixteen_bit_float_fetch_computes_metals_blend() {
    let fast_text = text();
    let option = "!{!\"air.compile.fast_math_enable\"}";
    assert_eq!(
        fast_text.matches(option).count(),
        1,
        "the fixture's math mode"
    );
    let safe_text = fast_text.replacen(option, "!{!\"air.compile.fast_math_disable\"}", 1);
    let (Some(fast), Some(safe)) = (assembled(&fast_text), assembled(&safe_text)) else {
        return;
    };
    // Each entry, and whether Metal's compiler folds its output into the blend under fast math.
    let entries = [
        ("fbl_over_loaded", false),
        ("fbl_over_loaded_float", false),
        ("fbl_over_product", true),
    ];
    let note = PrecisionNote::HalfOutputFastMathBlend;
    let mut checked = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        if !crate::lower::fragment::interlock_orders(&p) {
            continue;
        }
        let inputs = state(&p, OVER);
        for (entry, folds) in entries {
            for (bc, rounded) in [(&fast, false), (&safe, true)] {
                let what = format!("{stem} {entry} rounded {rounded}");
                let a = match compile_with(bc, entry, &p, &inputs) {
                    Ok(a) => a,
                    // A device without the format's storage refuses the form whatever the blend.
                    Err(crate::error::Error::Unsupported(u))
                        if u.needs != crate::error::Needs::NotImplemented =>
                    {
                        eprintln!("{what}: {u}");
                        continue;
                    }
                    Err(e) => panic!("{what}: {e}"),
                };
                assert_eq!(
                    fetched(&a),
                    Some((true, Some(OVER), rounded)),
                    "{what}: {:?}",
                    a.facts
                );
                assert_eq!(
                    a.precision.contains(&note),
                    folds && !rounded,
                    "{what}: {:?}",
                    a.precision
                );
                // The blend's NaN is the device's arithmetic's, which the store converts, noted on
                // every profile.
                assert!(
                    a.precision.contains(&PrecisionNote::DeviceNanEncoding),
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
        for (entry, folds) in entries {
            let what = format!("{stem} {entry}: the pipeline");
            let mut d = RenderPipelineDescriptor::new(function("fbl_vertex"));
            d.fragment = Some(function(entry));
            let r = attachments(OVER);
            d.color_attachments[0] = r.color_attachments[0];
            d.color_attachments[1] = r.color_attachments[1];
            let pipe = match dev.compile_render_pipeline(&d, &Budget::standard()) {
                Ok(pipe) => pipe,
                Err(crate::error::Error::Unsupported(u))
                    if u.needs != crate::error::Needs::NotImplemented =>
                {
                    eprintln!("{what}: {u}");
                    continue;
                }
                Err(e) => panic!("{what}: {e}"),
            };
            let storage: Vec<(u32, bool)> = (pipe.graphics.storage_attachments.iter())
                .map(|u| (u.color_index, u.writes))
                .collect();
            assert_eq!(storage, [(0, true)], "{what}");
            assert!(pipe.graphics.color_attachments[0].is_none(), "{what}");
            let fragment = (pipe.stages.iter())
                .find(|s| s.stage == vk::ShaderStage::FRAGMENT)
                .expect("a fragment stage");
            assert_eq!(
                fragment.precision_notes.contains(&note),
                folds,
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
}

/// Each blend constant's difference into color 0 under fast math (`BlendColor` with
/// `OneMinusBlendColor`, `BlendAlpha` with `OneMinusBlendAlpha`, either as the source factor, under
/// `Subtract` and `ReverseSubtract`, for color and alpha): on every committed profile that orders
/// fetch through the pixel interlock alone, `fbl_over_loaded` and `fbl_over_loaded_float` compute
/// it in the storage form, Metal's form (the minuend's product fused, the
/// subtrahend's rounded first), so neither records `HalfOutputFastMathBlend`, and pass L3. Before,
/// the color channels' blend color difference took a rounded near-equivalent and every difference
/// recorded the note.
#[test]
fn a_fetched_blend_constant_difference_records_no_note() {
    let Some(fast) = assembled(&text()) else {
        return;
    };
    use BlendFactor as F;
    let pairs = [
        (F::BlendColor, F::OneMinusBlendColor),
        (F::OneMinusBlendColor, F::BlendColor),
        (F::BlendAlpha, F::OneMinusBlendAlpha),
        (F::OneMinusBlendAlpha, F::BlendAlpha),
    ];
    let note = PrecisionNote::HalfOutputFastMathBlend;
    let mut checked = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        if !crate::lower::fragment::interlock_orders(&p) {
            continue;
        }
        for (fs, fd) in pairs {
            for op in [BlendOperation::Subtract, BlendOperation::ReverseSubtract] {
                let eq = BlendEquation {
                    source_rgb: fs,
                    destination_rgb: fd,
                    rgb_operation: op,
                    source_alpha: fs,
                    destination_alpha: fd,
                    alpha_operation: op,
                };
                let inputs = state(&p, eq);
                for entry in ["fbl_over_loaded", "fbl_over_loaded_float"] {
                    let what = format!("{stem} {entry} {fs:?} {fd:?} {op:?}");
                    let a = match compile_with(&fast, entry, &p, &inputs) {
                        Ok(a) => a,
                        // A device without the format's storage refuses the form whatever the
                        // blend.
                        Err(crate::error::Error::Unsupported(u))
                            if u.needs != crate::error::Needs::NotImplemented =>
                        {
                            eprintln!("{what}: {u}");
                            continue;
                        }
                        Err(e) => panic!("{what}: {e}"),
                    };
                    assert_eq!(
                        fetched(&a),
                        Some((true, Some(eq), false)),
                        "{what}: {:?}",
                        a.facts
                    );
                    assert!(!a.precision.contains(&note), "{what}: {:?}", a.precision);
                    link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
                    checked += 1;
                }
            }
        }
    }
    assert!(checked > 0, "no interlock profile took the storage form");
}
