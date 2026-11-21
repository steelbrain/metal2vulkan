//! Fragment functions that fetch a blended 16-bit float attachment through an output of one to
//! four lanes (the owned fixture `engine/fbl_fetch_blend16_narrow`), each reading color 1 in its
//! output's type and writing it beside colors 0 and 2 of mask none, under fast math, and the same
//! module stating no fast math (`air.compile.fast_math_disable`).
//!
//! The guest's shape, a macOS 26 window-server pass: color 1 `RG16Float` blended with the
//! destination color times the source plus zero for color and alpha under a mask of red and green,
//! through a `half2` (`fbl_narrow_half2`) or `float2` (`fbl_narrow_float2`) output. On every
//! committed profile that orders fetch through the pixel interlock alone, each compiles with color
//! 1 in the storage form and the blend it computes (`RuntimeFact::InterlockedFramebufferFetch`),
//! exact, noted by none (the blend's one term, of an output Metal's compiler cannot fold into
//! it), and its module passes L3; the whole pipeline is made there, color 1 a written storage
//! attachment and no color attachment of the subpass, its stages passing L3. Before the shader
//! blended an output of fewer than four lanes, each was refused `NotImplemented(FramebufferFetch)`
//! at color 1.
//!
//! The class, on the first such profile that takes the form: every output width into each 16-bit
//! float format with no more channels than it has lanes (Metal refuses fewer), under blends that
//! put every factor but the second source's in each of the four factors and every operation in
//! each class, the guest's, and the color channels' blend color difference, under the format's
//! partial write masks (one color channel of `RGBA16Float` among them), each under fast math and
//! not in turn: each compiles in the storage form, the blend its fact names, and passes L3. Before,
//! besides the narrow outputs, the fast-math blend color difference was refused
//! `ExtendedRangeBlend(ConstantColorDifference)`. No GPU is used.
//!
//! `llvm-as` assembles the fixture; a missing tool skips visibly.

use std::path::Path;
use std::sync::Arc;

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
    std::fs::read_to_string(engine_dir().join("fbl_fetch_blend16_narrow.ll")).expect("the fixture")
}

/// `text` assembled; `None` (skipped) without `llvm-as`.
fn assembled(text: &str) -> Option<Vec<u8>> {
    match metal2vulkan_devtools::llvm_assemble_text(text) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => {
            panic!("llvm-as rejected fbl_fetch_blend16_narrow: {why}")
        }
    }
}

/// Each entry and its output's lanes.
const ENTRIES: [(&str, u8); 8] = [
    ("fbl_narrow_half", 1),
    ("fbl_narrow_half2", 2),
    ("fbl_narrow_half3", 3),
    ("fbl_narrow_half4", 4),
    ("fbl_narrow_float", 1),
    ("fbl_narrow_float2", 2),
    ("fbl_narrow_float3", 3),
    ("fbl_narrow_float4", 4),
];

/// The guest's blend: the destination color times the source plus zero, for color and alpha.
const GUEST: BlendEquation = BlendEquation {
    source_rgb: BlendFactor::DestinationColor,
    destination_rgb: BlendFactor::Zero,
    rgb_operation: BlendOperation::Add,
    source_alpha: BlendFactor::DestinationAlpha,
    destination_alpha: BlendFactor::Zero,
    alpha_operation: BlendOperation::Add,
};

/// Every factor but the second source's.
const FACTORS: [BlendFactor; 15] = [
    BlendFactor::Zero,
    BlendFactor::One,
    BlendFactor::SourceColor,
    BlendFactor::OneMinusSourceColor,
    BlendFactor::SourceAlpha,
    BlendFactor::OneMinusSourceAlpha,
    BlendFactor::DestinationColor,
    BlendFactor::OneMinusDestinationColor,
    BlendFactor::DestinationAlpha,
    BlendFactor::OneMinusDestinationAlpha,
    BlendFactor::SourceAlphaSaturated,
    BlendFactor::BlendColor,
    BlendFactor::OneMinusBlendColor,
    BlendFactor::BlendAlpha,
    BlendFactor::OneMinusBlendAlpha,
];

/// Every operation.
const OPERATIONS: [BlendOperation; 5] = [
    BlendOperation::Add,
    BlendOperation::Subtract,
    BlendOperation::ReverseSubtract,
    BlendOperation::Min,
    BlendOperation::Max,
];

/// Blends that put every factor of [`FACTORS`] in each of the four factors (each position a
/// permutation of them) and every operation in each class, then the guest's and the color
/// channels' blend color difference.
fn equations() -> Vec<BlendEquation> {
    let n = FACTORS.len();
    let ops = OPERATIONS.len();
    let mut out: Vec<BlendEquation> = (0..n)
        .map(|i| BlendEquation {
            source_rgb: FACTORS[i],
            destination_rgb: FACTORS[(7 * i + 3) % n],
            rgb_operation: OPERATIONS[i % ops],
            source_alpha: FACTORS[(4 * i + 5) % n],
            destination_alpha: FACTORS[(11 * i + 2) % n],
            alpha_operation: OPERATIONS[(i + 2) % ops],
        })
        .collect();
    out.push(GUEST);
    out.push(BlendEquation {
        source_rgb: BlendFactor::BlendColor,
        destination_rgb: BlendFactor::OneMinusBlendColor,
        rgb_operation: BlendOperation::Subtract,
        source_alpha: BlendFactor::One,
        destination_alpha: BlendFactor::OneMinusSourceAlpha,
        alpha_operation: BlendOperation::Add,
    });
    out
}

/// A format's channels, and the write masks that store some of them.
fn formats() -> [(PixelFormat, u8, Vec<ColorWriteMask>); 3] {
    use ColorWriteMask as M;
    [
        (PixelFormat::R16Float, 1, vec![M::ALL, M::RED]),
        (
            PixelFormat::RG16Float,
            2,
            vec![M::ALL, M::RED, M::GREEN, M::RED.union(M::GREEN)],
        ),
        (
            PixelFormat::RGBA16Float,
            4,
            vec![
                M::ALL,
                M::RED,
                M::GREEN.union(M::ALPHA),
                M::RED.union(M::BLUE),
                M::ALPHA,
            ],
        ),
    ]
}

/// The attachments: color 1 of `pf` blended by `eq` under `mask`, colors 0 and 2 `RGBA16Float` of
/// mask none.
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
    r.color_attachments[0] = Some(unwritten);
    r.color_attachments[1] = Some(blended);
    r.color_attachments[2] = Some(unwritten);
    r
}

/// The lowering inputs of a function that reads color 1 in pipeline `r` on `p`.
fn state(r: &RenderLinkInputs, p: &DeviceProfile) -> LoweringInputs {
    let mut inputs = fragment_inputs(r, p);
    inputs.framebuffer_fetch = Some(FramebufferFetchInputs::of(
        [1],
        &r.color_attachments,
        false,
        false,
        1,
    ));
    inputs
}

/// Color 1's storage form in `a`: whether it writes, its blend, and whether that is rounded.
fn fetched(a: &FunctionArtifact) -> Option<(bool, Option<BlendEquation>, bool)> {
    a.facts.iter().find_map(|f| match f {
        RuntimeFact::InterlockedFramebufferFetch {
            color: 1,
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
    Device::from_queries(desc, DevicePolicy::default()).expect("device")
}

/// `compile_with`, `None` where the device lacks a format's storage (refused whatever the blend).
fn compiled(
    bc: &[u8],
    entry: &str,
    p: &DeviceProfile,
    inputs: &LoweringInputs,
    what: &str,
) -> Option<Arc<FunctionArtifact>> {
    match compile_with(bc, entry, p, inputs) {
        Ok(a) => Some(a),
        Err(crate::error::Error::Unsupported(u))
            if u.needs != crate::error::Needs::NotImplemented =>
        {
            eprintln!("{what}: {u}");
            None
        }
        Err(e) => panic!("{what}: {e}"),
    }
}

#[test]
fn a_narrow_output_blends_a_fetched_sixteen_bit_float_attachment() {
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
    let note = PrecisionNote::HalfOutputFastMathBlend;
    let guest_mask = ColorWriteMask::RED.union(ColorWriteMask::GREEN);
    let mut class: Option<DeviceProfile> = None;
    let mut checked = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        if !crate::lower::fragment::interlock_orders(&p) {
            continue;
        }
        let r = attachments(PixelFormat::RG16Float, GUEST, guest_mask);
        let inputs = state(&r, &p);
        let mut took = false;
        for entry in ["fbl_narrow_half2", "fbl_narrow_float2"] {
            for (bc, rounded) in [(&fast, false), (&safe, true)] {
                let what = format!("{stem} {entry} rounded {rounded}");
                let Some(a) = compiled(bc, entry, &p, &inputs, &what) else {
                    continue;
                };
                assert_eq!(
                    fetched(&a),
                    Some((true, Some(GUEST), rounded)),
                    "{what}: {:?}",
                    a.facts
                );
                assert!(!a.precision.contains(&note), "{what}: {:?}", a.precision);
                // A `float2` output's blend result is the storage image's store's to round to
                // `half`, the device's, noted. A `half2` output's is a `half`.
                let float = entry == "fbl_narrow_float2";
                assert_eq!(
                    a.precision.contains(&PrecisionNote::DeviceColorConversion),
                    float,
                    "{what}: {:?}",
                    a.precision
                );
                link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
                took = true;
                checked += 1;
            }
        }
        if took && class.is_none() {
            class = Some(p.clone());
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
        for entry in ["fbl_narrow_half2", "fbl_narrow_float2"] {
            let what = format!("{stem} {entry}: the pipeline");
            let mut d = RenderPipelineDescriptor::new(function("fbl_narrow_vertex"));
            d.fragment = Some(function(entry));
            for (k, c) in r.color_attachments.iter().enumerate().take(3) {
                d.color_attachments[k] = *c;
            }
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
            assert_eq!(storage, [(1, true)], "{what}");
            assert!(pipe.graphics.color_attachments[1].is_none(), "{what}");
            let fragment = (pipe.stages.iter())
                .find(|s| s.stage == vk::ShaderStage::FRAGMENT)
                .expect("a fragment stage");
            assert!(
                !fragment.precision_notes.contains(&note),
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
    // The class, on the first profile that took the form.
    let p = class.expect("a profile that took the form");
    let equations = equations();
    let mut built = 0;
    for (n, (entry, lanes)) in ENTRIES.into_iter().enumerate() {
        for (pf, channels, masks) in formats() {
            // Metal refuses an output of fewer lanes than the format's channels.
            if lanes < channels {
                continue;
            }
            for (i, eq) in equations.iter().enumerate() {
                let mask = masks[(i + n) % masks.len()];
                let (bc, rounded) = match (i + n) % 2 {
                    0 => (&fast, false),
                    _ => (&safe, true),
                };
                let what = format!("{entry} {pf:?} {eq:?} {mask:?} rounded {rounded}");
                let r = attachments(pf, *eq, mask);
                let Some(a) = compiled(bc, entry, &p, &state(&r, &p), &what) else {
                    continue;
                };
                assert_eq!(
                    fetched(&a),
                    Some((true, Some(*eq), rounded)),
                    "{what}: {:?}",
                    a.facts
                );
                link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
                built += 1;
            }
        }
    }
    assert!(built > 0, "the class built nothing");
}

/// The attachments of [`attachments`], color 1 written unblended.
fn unblended(pf: PixelFormat) -> RenderLinkInputs {
    let mut r = attachments(pf, GUEST, ColorWriteMask::ALL);
    if let Some(c) = r.color_attachments[1].as_mut() {
        c.blending_enabled = false;
    }
    r
}

/// A one- or two-channel 8-bit sRGB attachment (`R8Unorm_sRGB`, `RG8Unorm_sRGB`) the function
/// fetches, on every committed profile that orders fetch through the pixel interlock alone and
/// renders the format itself: each entry of as many lanes as the format's channels or more, fast
/// and safe math, compiles in the storage form through the format's integer view, unblended and
/// under source-alpha over and the guest's blend, which the function computes
/// (`extended_range::Blend::srgb8`): noted `UnmeasuredFetchBlend` exactly where that is not
/// Metal's (a no-alpha form the 8-bit unorm blend does not build under fast math, or an output of
/// fewer than four lanes, which blends as four whose missing lanes are zero), and its module passes
/// L3. Before, each was refused `NotImplemented(FramebufferFetch)`.
#[test]
fn a_fetched_one_or_two_channel_srgb_attachment_is_written_and_blended() {
    use crate::lower::arith::extended_range::blend::{BlendArith, BlendMath};
    use crate::lower::arith::extended_range::Blend;
    let fast_text = text();
    let option = "!{!\"air.compile.fast_math_enable\"}";
    let safe_text = fast_text.replacen(option, "!{!\"air.compile.fast_math_disable\"}", 1);
    let (Some(fast), Some(safe)) = (assembled(&fast_text), assembled(&safe_text)) else {
        return;
    };
    let over = BlendEquation {
        source_rgb: BlendFactor::SourceAlpha,
        destination_rgb: BlendFactor::OneMinusSourceAlpha,
        rgb_operation: BlendOperation::Add,
        source_alpha: BlendFactor::One,
        destination_alpha: BlendFactor::OneMinusSourceAlpha,
        alpha_operation: BlendOperation::Add,
    };
    let note = PrecisionNote::UnmeasuredFetchBlend;
    let mut checked = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        if !crate::lower::fragment::interlock_orders(&p) {
            continue;
        }
        for (pf, channels) in [
            (PixelFormat::R8UnormSRGB, 1),
            (PixelFormat::RG8UnormSRGB, 2),
        ] {
            // A device that renders the format through a word view: not this form.
            if crate::metal::format::WordAttachment::of(pf, &p).is_some() {
                continue;
            }
            for (entry, lanes) in ENTRIES {
                if lanes < channels {
                    continue;
                }
                for (bc, math) in [(&fast, BlendMath::Fast), (&safe, BlendMath::Rounded)] {
                    let rounded = math == BlendMath::Rounded;
                    let what = format!("{stem} {pf:?} {entry} rounded {rounded}");
                    let r = unblended(pf);
                    let a = compile_with(bc, entry, &p, &state(&r, &p))
                        .unwrap_or_else(|e| panic!("{what} unblended: {e}"));
                    assert_eq!(
                        fetched(&a).map(|f| (f.0, f.1)),
                        Some((true, None)),
                        "{what}"
                    );
                    assert!(!a.precision.contains(&note), "{what}: {:?}", a.precision);
                    link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
                    for eq in [over, GUEST] {
                        let what = format!("{what} {eq:?}");
                        let r = attachments(pf, eq, ColorWriteMask::ALL);
                        let a = compile_with(bc, entry, &p, &state(&r, &p))
                            .unwrap_or_else(|e| panic!("{what}: {e}"));
                        assert_eq!(fetched(&a), Some((true, Some(eq), rounded)), "{what}");
                        let half = (math, BlendArith::Half);
                        let exact = lanes == 4
                            && Blend::srgb8(&eq, pf, ColorWriteMask::ALL, half)
                                .is_ok_and(|(_, exact)| exact);
                        assert_eq!(a.precision.contains(&note), !exact, "{what}");
                        link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
                        checked += 1;
                    }
                }
            }
        }
    }
    assert!(checked > 0, "no interlock profile took the storage form");
    // A `half` output under fast math: its storage write is noted `HalfOutputFastMathBlend` where
    // it stores a channel, and not under a mask that names none of the format's (blue into
    // `R8Unorm_sRGB`), which stores nothing.
    let fast_note = PrecisionNote::HalfOutputFastMathBlend;
    for (stem, p) in committed_profiles().expect("profiles") {
        let pf = PixelFormat::R8UnormSRGB;
        if !crate::lower::fragment::interlock_orders(&p)
            || crate::metal::format::WordAttachment::of(pf, &p).is_some()
        {
            continue;
        }
        for (mask, noted) in [(ColorWriteMask::RED, true), (ColorWriteMask::BLUE, false)] {
            let mut r = unblended(pf);
            if let Some(c) = r.color_attachments[1].as_mut() {
                c.write_mask = mask;
            }
            let what = format!("{stem} {mask:?}");
            let a = compile_with(&fast, "fbl_narrow_half4", &p, &state(&r, &p))
                .unwrap_or_else(|e| panic!("{what}: {e}"));
            assert_eq!(
                a.precision.contains(&fast_note),
                noted,
                "{what}: {:?}",
                a.precision
            );
        }
    }
}
