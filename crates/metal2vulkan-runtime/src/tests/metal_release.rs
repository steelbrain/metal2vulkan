//! `metal_release`: the answers of the release every device answers as are what it measured, and
//! the descriptor check and the alignment queries answer by them.

use crate::capabilities::linear_alignment_rule;
use crate::metal_release::{DEPTH_STENCIL_RESOURCE_ID, MACOS_RELEASE};
use crate::pixel_format_info::describes;
use crate::texture::heap::no_texture_alignment;
use crate::texture_descriptor::{Creator, Refused, TextureDescriptor, TextureProblem};
use crate::texture_descriptor::{TextureRules, APPLE_SAMPLE_COUNTS};
use crate::{BufferTextureProblem, Error, Invalid, Problem, ResourceId};
use metal2vulkan::metal::PixelFormat as P;
use metal2vulkan::metal::{TextureType, TextureUsage};

/// The 12 three-channel formats of the macOS 27.0 SDK and `RGB16Bfloat`: formats macOS 27.0.1
/// describes none of.
const THREE_CHANNEL: [P; 13] = [
    P::RGB8Unorm,
    P::RGB8Snorm,
    P::RGB8Uint,
    P::RGB8Sint,
    P::RGB16Unorm,
    P::RGB16Snorm,
    P::RGB16Uint,
    P::RGB16Sint,
    P::RGB16Float,
    P::RGB16Bfloat,
    P::RGB32Uint,
    P::RGB32Sint,
    P::RGB32Float,
];

/// The formats from 646 to 650, which the release does not describe.
const TWELVE_BIT_UINT: [P; 5] = [
    P::R12UintX4,
    P::R12UintPACKED,
    P::RG12UintX8,
    P::RG12UintPACKED,
    P::RGB10A8Uint2P,
];

/// The formats from 630 to 645: made by an Apple M4 under macOS 27.0.1, described by none on an
/// Apple M3 Max at that release (the consumer's macOS 27 host lane, and where the alignment query
/// asserts and the descriptor's validation stops a texture of each, "has invalid pixelFormat").
const FROM_630: [P; 16] = [
    P::BGRA10UintPACKED,
    P::BGRA10XRPACKED,
    P::BGRA10XRPACKEDSRGB,
    P::BGRA10UnormPACKED,
    P::BGRA10UnormPACKEDSRGB,
    P::BGRA10UnormPACKEDHLG,
    P::BGRA10UnormPACKEDPQ,
    P::BGRA10Uint,
    P::BGRA10Unorm,
    P::BGRA10UnormSRGB,
    P::BGRA10UnormHLG,
    P::BGRA10UnormPQ,
    P::R10UintX6,
    P::R10UintPACKED,
    P::RG10UintX12,
    P::RG10UintPACKED,
];

/// The rules of a device of an Apple GPU's sample counts.
const RULES: TextureRules = TextureRules {
    sample_counts: APPLE_SAMPLE_COUNTS,
};

/// The release is macOS 27.0.1's, as macOS names it.
#[test]
fn the_release_is_macos_27_0_1() {
    assert_eq!(MACOS_RELEASE, "27.0.1");
}

/// The release describes 264 values on an Apple M3 Max (`MTLPixelFormatGetInfoForDevice`, our
/// probe and the consumer's macOS 27 host lane): every value the runtime names but
/// `Depth24Unorm_Stencil8` and `X24_Stencil8`, which no Apple GPU has, the three-channel formats,
/// 646 to 650, and the 16 from 630 to 645, which an Apple M4 there describes.
#[test]
fn the_release_describes_what_it_measured() {
    let apple_lacks = [P::Depth24UnormStencil8, P::X24Stencil8];
    let named = P::ALL.iter().filter(|&&pf| pf != P::Invalid).count();
    let described = P::ALL.iter().filter(|&&pf| describes(pf)).count();
    assert_eq!(described, 264);
    assert_eq!(named - described, 36);
    for &pf in P::ALL {
        let an_m3s = !THREE_CHANNEL.contains(&pf)
            && !TWELVE_BIT_UINT.contains(&pf)
            && !FROM_630.contains(&pf)
            && !apple_lacks.contains(&pf)
            && pf != P::Invalid;
        assert_eq!(describes(pf), an_m3s, "{pf:?}");
    }
}

/// The release answers a depth/stencil state's `gpuResourceID` below Apple10 0 (our probe on an
/// M3 Max, plain Metal), and answers the heap query of a texture of a format of which Metal makes
/// none alone, with no planes, 128 for every type but a texture buffer's 16.
#[test]
fn the_release_s_resource_ids_and_nil_alignments() {
    assert_eq!(DEPTH_STENCIL_RESOURCE_ID, ResourceId(0));
    use TextureType as T;
    let types = [
        (T::Type1D, 128),
        (T::Type2D, 128),
        (T::Type2DMultisample, 128),
        (T::TextureBuffer, 16),
        (T::Type1DArray, 128),
        (T::Type2DArray, 128),
        (T::Cube, 128),
        (T::CubeArray, 128),
        (T::Type3D, 128),
        (T::Type2DMultisampleArray, 128),
    ];
    for pf in [
        P::Unspecialized,
        P::RGBA16FloatXR,
        P::AnyDepth,
        P::AnyStencil,
    ] {
        for (ty, align) in types {
            assert_eq!(no_texture_alignment(pf, ty), align, "{pf:?} {ty:?}");
        }
    }
}

/// A texture of a format Metal describes none of is stopped as `Invalid` is, from every creator
/// and the heap size query, before every other rule; one of a format it describes is not (plain
/// Metal ends its process over it, `newTextureWithDescriptor:` and the heap size query alike, the
/// API validation layer off: our probes, an M3 Max under macOS 27.0.1). The 16 formats from 630 to
/// 645 are stopped, as the M3 Max stops them.
#[test]
fn textures_of_the_formats_metal_describes_none_of_are_stopped() {
    let creators = [
        Creator::Device,
        Creator::Buffer,
        Creator::OverSurface,
        Creator::Shared,
        Creator::SizeQuery,
    ];
    let stop = Err(Refused::Stops(TextureProblem::NotAFormat));
    for &pf in P::ALL {
        let mut d = TextureDescriptor::texture_2d(pf, 16, 16, false);
        d.usage = TextureUsage::SHADER_READ;
        for creator in creators {
            let checked = d.check_made(creator, RULES);
            if describes(pf) {
                assert_ne!(checked, stop, "{pf:?} {creator:?}");
            } else {
                assert_eq!(checked, stop, "{pf:?} {creator:?}");
            }
        }
        // Before a rule it breaks too: a zero extent.
        let mut zero = d;
        zero.width = 0;
        if !describes(pf) {
            assert_eq!(zero.check(RULES), stop, "{pf:?}");
        }
    }
    for pf in FROM_630 {
        let d = TextureDescriptor::texture_2d(pf, 16, 16, false);
        assert_eq!(d.check(RULES), stop, "{pf:?}");
    }
}

/// `minimumLinearTextureAlignmentForPixelFormat:` follows an Apple M3 Max's rule: an assertion
/// for the formats from 630 to 645 ("MTLPixelFormatInvalid is not supported on this device"; an
/// Apple M4 under 27.0.1 answers 16), which the runtime refuses `Invalid`, and 16 bytes at least
/// for the others (the device's answer is a probe of its images, `texture/linear_alignment.rs`).
#[test]
fn the_linear_alignment_answers_the_m3_max_s() {
    let call = "minimum_linear_texture_alignment_for_pixel_format";
    for pf in FROM_630 {
        assert_eq!(
            linear_alignment_rule(pf),
            Err(Error::Invalid(Invalid::new(
                call,
                Problem::BufferTexture(BufferTextureProblem::NotLinear { pixel_format: pf }),
            ))),
            "{pf:?}"
        );
    }
    for pf in [P::RGBA8Unorm, P::R10UnormX6, P::RGB10A82P] {
        assert_eq!(linear_alignment_rule(pf), Ok(16));
    }
}

/// Metal describes neither `Depth24Unorm_Stencil8` nor `X24_Stencil8`, which an Apple GPU has
/// neither of: the descriptor check stops a texture of each as one of no format, first, from every
/// creator, as Metal on an Apple M4 ("has invalid pixelFormat (255)") and an M3 Max (the
/// consumer's run) stops them, whatever the Vulkan driver has.
#[test]
fn neither_24_bit_depth_format_is_described() {
    let creators = [
        Creator::Device,
        Creator::Buffer,
        Creator::OverSurface,
        Creator::Shared,
        Creator::SizeQuery,
    ];
    let stop = Err(Refused::Stops(TextureProblem::NotAFormat));
    for pf in [P::Depth24UnormStencil8, P::X24Stencil8] {
        assert!(!describes(pf), "{pf:?}");
        let mut d = TextureDescriptor::texture_2d(pf, 16, 16, false);
        d.usage = TextureUsage::RENDER_TARGET;
        for creator in creators {
            assert_eq!(d.check_made(creator, RULES), stop, "{pf:?} {creator:?}");
        }
    }
}
