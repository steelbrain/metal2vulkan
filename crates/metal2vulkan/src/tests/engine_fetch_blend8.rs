//! A fragment function that fetches a blended 8-bit unorm attachment, in the shape of a
//! CoreAnimation composite of the x86 macOS guest (the owned fixture `engine/fbl_fetch_blend8`,
//! under fast math): color 0 blended with one over one minus the source alpha for color and alpha,
//! color 1 `RGBA16Float` of mask none, the function reading color 0. On a profile that orders
//! fetch through rasterization-order input attachments, a color input of an `ARGB8Unorm`
//! attachment, whose input attachment would read its image's rotated channels, is refused
//! `NotImplemented(Format)` at color 0. No GPU is used.
//!
//! `llvm-as` assembles the fixtures; a missing tool skips visibly.

use crate::device::DeviceProfile;
use crate::error::{Attachment, Error, Feature, Needs};
use crate::metal::{BlendFactor, BlendOperation, ColorWriteMask, PixelFormat};
use crate::pipeline::descriptor::ColorAttachmentDescriptor;
use crate::pipeline::link::RenderLinkInputs;
use crate::pipeline::lower_inputs::{BlendEquation, FramebufferFetchInputs, LoweringInputs};
use crate::pipeline::testing::{compile_with, engine_dir, fragment_inputs, AssembleError};
use crate::spirv::testing::committed_profiles;

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
    inputs.framebuffer_fetch = Some(FramebufferFetchInputs::of([0], &r.color_attachments));
    inputs
}

/// A color input of an `ARGB8Unorm` attachment, whose image holds Metal's channels rotated, on
/// every committed profile that orders fetch through rasterization-order input attachments: the
/// input attachment would read the image's order, so the function's lowering refuses it
/// `NotImplemented(Format)` at the attachment (link admits such an attachment's input attachment
/// only for the function's own blend over it, `RuntimeFact::RotatedInputBlend`).
#[test]
fn a_color_input_of_a_rotated_attachment_is_refused_over_its_input_attachment() {
    let Some(fast) = assembled("fbl_fetch_blend8") else {
        return;
    };
    let mut checked = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        if !crate::lower::fragment::inputs_order(&p) {
            continue;
        }
        let inputs = state(&p, PixelFormat::ARGB8Unorm, OVER, ColorWriteMask::ALL);
        match compile_with(&fast, "fbl8_over_loaded", &p, &inputs) {
            Err(Error::Unsupported(u)) => {
                assert_eq!(
                    (u.feature, u.needs),
                    (Feature::Format, Needs::NotImplemented),
                    "{stem}"
                );
                let at = u.location.as_ref().and_then(|l| l.attachment());
                assert_eq!(at, Some(Attachment::Color(0)), "{stem}");
            }
            other => panic!("{stem}: {:?}", other.map(|_| ())),
        }
        checked += 1;
    }
    assert!(
        checked > 0,
        "no profile orders fetch through input attachments"
    );
}
