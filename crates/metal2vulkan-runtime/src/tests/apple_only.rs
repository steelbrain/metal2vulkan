//! `apple_only`: every Apple-only row (class A) answers typed.

use crate::{AppleOnly, AppleOnlyCall, DEFAULT_PROTECTION_OPTIONS};

/// The 11 Apple-only calls of the table, each at Metal's default where it has one and at values
/// off it, with the runtime's answer. The getter (`protectionOptions`) is
/// `no_getter_answers_off_metals_default`'s. A texture's rotation is no longer the table's: the
/// runtime builds it.
#[test]
fn every_apple_only_call_answers_typed() {
    use AppleOnlyCall as C;
    let table: [(&str, AppleOnlyCall, Result<(), AppleOnly>); 30] = [
        // Compressed-texture controls: no compression is visible on Vulkan.
        (
            "invalidateCompressedTexture:",
            C::InvalidateCompressedTexture,
            Ok(()),
        ),
        (
            "invalidateCompressedTexture:slice:level:",
            C::InvalidateCompressedTextureSlice,
            Ok(()),
        ),
        (
            "insertCompressedTextureReinterpretationFlush",
            C::CompressedTextureReinterpretationFlush,
            Ok(()),
        ),
        // Render setters.
        (
            "setAlphaTestReferenceValue:",
            C::SetAlphaTestReferenceValue(0.5),
            Ok(()),
        ),
        // Every width: the encoder's lines are drawn at Metal's quantized width
        // (`render/line_width.rs`).
        ("setLineWidth:", C::SetLineWidth(1.0), Ok(())),
        ("setLineWidth:", C::SetLineWidth(2.0), Ok(())),
        ("setLineWidth:", C::SetLineWidth(0.5), Ok(())),
        ("setLineWidth:", C::SetLineWidth(f32::NAN), Ok(())),
        ("setLineWidth:", C::SetLineWidth(-1.0), Ok(())),
        ("setLineWidth:", C::SetLineWidth(f32::INFINITY), Ok(())),
        ("setPointSize:", C::SetPointSize(1.0), Ok(())),
        ("setPointSize:", C::SetPointSize(f32::NAN), Ok(())),
        (
            "setProvokingVertexMode:",
            C::SetProvokingVertexMode(0),
            Ok(()),
        ),
        (
            "setProvokingVertexMode:",
            C::SetProvokingVertexMode(1),
            Ok(()),
        ),
        (
            "setProvokingVertexMode:",
            C::SetProvokingVertexMode(u64::MAX),
            Ok(()),
        ),
        (
            "setProvokingVertexMode:",
            C::SetProvokingVertexMode(1 << 63),
            Ok(()),
        ),
        (
            "setProvokingVertexMode:",
            C::SetProvokingVertexMode(2),
            Ok(()),
        ),
        (
            "setProvokingVertexMode:",
            C::SetProvokingVertexMode(i64::MAX as u64),
            Ok(()),
        ),
        (
            "setViewportTransformEnabled:",
            C::SetViewportTransformEnabled(true),
            Ok(()),
        ),
        (
            "setViewportTransformEnabled:",
            C::SetViewportTransformEnabled(false),
            Err(AppleOnly::ViewportTransformDisabled),
        ),
        // The amplification mode's count: the value's set bits in mode 1, the value otherwise;
        // 1 to 8 taken, 0 (copies that change from run to run) and past 8 refused.
        (
            "setVertexAmplificationMode:value:",
            C::SetVertexAmplificationMode { mode: 0, value: 0 },
            Err(AppleOnly::VertexAmplificationMode),
        ),
        (
            "setVertexAmplificationMode:value:",
            C::SetVertexAmplificationMode { mode: 1, value: 2 },
            Ok(()),
        ),
        (
            "setVertexAmplificationMode:value:",
            C::SetVertexAmplificationMode {
                mode: 1,
                value: 255,
            },
            Ok(()),
        ),
        (
            "setVertexAmplificationMode:value:",
            C::SetVertexAmplificationMode {
                mode: 1,
                value: 257,
            },
            Ok(()),
        ),
        (
            "setVertexAmplificationMode:value:",
            C::SetVertexAmplificationMode { mode: 0, value: 8 },
            Ok(()),
        ),
        (
            "setVertexAmplificationMode:value:",
            C::SetVertexAmplificationMode {
                mode: 255,
                value: 9,
            },
            Err(AppleOnly::VertexAmplificationMode),
        ),
        (
            "setVertexAmplificationMode:value:",
            C::SetVertexAmplificationMode {
                mode: 1,
                value: u64::MAX,
            },
            Err(AppleOnly::VertexAmplificationMode),
        ),
        // Depth and stencil "cleared" change nothing Metal draws.
        ("setDepthCleared", C::SetDepthCleared, Ok(())),
        ("setStencilCleared", C::SetStencilCleared, Ok(())),
        (
            "setAlphaTestReferenceValue:",
            C::SetAlphaTestReferenceValue(f32::NAN),
            Ok(()),
        ),
    ];
    let mut selectors: Vec<&str> = Vec::new();
    for (selector, call, answer) in table {
        assert_eq!(call.selector(), selector, "{call:?}");
        assert_eq!(call.check(), answer, "{call:?}");
        if !selectors.contains(&selector) {
            selectors.push(selector);
        }
    }
    // The 11 calls of the table's 28 rows (some calls have several rows).
    assert_eq!(selectors.len(), 11, "{selectors:?}");
}

/// The getter answers Metal's default, the protection options of every runtime texture.
#[test]
fn no_getter_answers_off_metals_default() {
    assert_eq!(DEFAULT_PROTECTION_OPTIONS, 0);
}

/// Each refused Apple-only feature names why ([`AppleOnly::gap`]), and none is a device's gap:
/// each refusal's answer on Metal is not measured. The error's text names the gap.
#[test]
fn each_apple_only_refusal_names_its_gap() {
    use crate::AppleOnlyGap::Unmeasured;
    let table = [
        (AppleOnly::ProtectedContent, Unmeasured),
        (AppleOnly::ViewportTransformDisabled, Unmeasured),
        (AppleOnly::VertexAmplificationMode, Unmeasured),
        (AppleOnly::SurfaceClientDirection, Unmeasured),
    ];
    for (feature, gap) in table {
        assert_eq!(feature.gap(), gap, "{feature:?}");
        let text = crate::Error::AppleOnly(feature).to_string();
        assert!(text.contains(&format!("({gap:?})")), "{text}");
        assert!(!text.contains("not on Vulkan"), "{text}");
    }
}
