//! What Metal does with the `MTLPixelFormat` values no case ever put to it: the two ASTC
//! footprints Apple's headers do not declare, Apple's one-plane video formats, its two-plane
//! 10-bit one, the padded 10- and 12-bit display formats, and the two plain stragglers
//! `BGRA4Unorm` and `ARGB8Unorm_sRGB`.
//!
//! # Why this family exists
//!
//! `docs/CONFORMANCE-COVERAGE.md`'s option-space axis read 68 of `MTLPixelFormat`'s 301 values as
//! compared against nothing, the largest single number on that axis. The 68 were one
//! undifferentiated figure and they are not one thing: **35 of them cannot be put in a texture
//! descriptor at all** without ending Metal's process, and 33 can. This family compares the 33.
//! The 35 are settled, with the measurement that settles them, under *The thirty-five no
//! descriptor may name* below.
//!
//! The 33 were nowhere in the suite. [`crate::cases::textures`] covers every *uncompressed colour*
//! format whose texels are Metal's own bytes, [`crate::cases::astc_textures`] the fourteen ASTC
//! footprints Apple's headers declare, [`crate::cases::ycbcr_textures`] and the extended-range
//! families their two-plane formats, and [`crate::cases::heap_metal_sizes`] a handful of
//! descriptors Metal makes no texture of. Not one of the 68 is named in any of them: measured by
//! matching every `PixelFormat::` spelling in the suite's sources against the list, which found
//! none of the 68.
//!
//! # What each case compares
//!
//! One case a format family. Each holds a table of [`Q`] descriptors and records, per descriptor,
//! the two facts [`crate::cases::heap_metal_sizes`] established — read rather than copied, so the
//! two families cannot drift apart in what they ask:
//!
//! - what `heapTextureSizeAndAlignWithDescriptor:` answers, Metal's own layout arithmetic;
//! - whether `newTextureWithDescriptor:` answers a texture, which is what makes a 0-byte size mean
//!   something.
//!
//! Then, for every format Metal makes a texture of, a **host round trip** in the shape
//! [`crate::cases::textures`] uses: a 16 by 16 `Shared` texture read fresh, written whole by
//! `replaceRegion:`, and read back, both reads into a destination that already holds [`BEFORE`] so
//! a byte the copy did not write is a recorded value rather than a zero nobody can tell from a
//! written one.
//!
//! ## What each sweep can tell apart, measured
//!
//! - **Every texture type Metal allows for the format**, in both `Shared` and `Private` storage.
//!   Not a formality: measured on an Apple M3 Max under Metal 27.0.1, the nine one-plane `YCBCR*`
//!   formats and the four `RGB10A8_2P` ones answer **nil** for `Cube` and `CubeArray` and a
//!   texture for the other six types, while `BGRA4Unorm`, `ARGB8Unorm_sRGB` and the padded 10- and
//!   12-bit formats answer a texture for all eight. A side that made a cube of any of the
//!   thirteen, or refused one of the rest, answers differently at that site and nowhere else.
//! - **Width 1 to 17** at height 1, and **height 1 to 17** at width 16. These are what separate
//!   the ASTC footprints, and the only thing that does: `ASTC_4x2_*`'s height sweep steps at
//!   heights 5, 9 and 17, `ASTC_8x4_*`'s at 17 alone, so a side that read either footprint as 4 by
//!   4 — ASTC's smallest declared one — answers neither. Their 64 by 64 answers (8192 and 2048
//!   bytes) and their whole width sweeps are identical within each pair of twins, because the
//!   128-byte floor hides the block width, which is why the height sweep is in the table.
//! - **The round trip's bytes**, which are the format's texel or block size: Metal's
//!   `getBytes:` fills one row of `blocks-or-texels-per-row x block-bytes` and leaves the rest of
//!   the 256-byte destination row as it was. Measured per format on the same M3 Max: 32 bytes for
//!   `BGRA4Unorm` and the 4:2:2 `YCBCR*` formats, 64 for `ARGB8Unorm_sRGB`, the 4:4:4 ones, the
//!   `RGB10A2Unorm_*`/`BGR10A2Unorm_*` twins and the `RGB10A8_2P` family, 64 for `ASTC_4x2_*` (4
//!   blocks of 16) and 32 for `ASTC_8x4_*` (2 blocks of 16).
//! - **Whether the round trip moves anything at all**, which is a per-format fact and not a
//!   property of the pitch. Seven of the formats — `R12Unorm_X4`, `R12Unorm_X4_PQ`,
//!   `RG12Unorm_X8`, `R10Unorm_X6_PQ`, `R10Unorm_X6_HLG`, `R12Unorm_X4_HLG` and
//!   `RGB10A2Unorm_HLG` — take `replaceRegion:` without complaint and then leave the whole
//!   destination of `getBytes:` untouched, at every pitch from 16 to 256 bytes. That is not an
//!   anomaly of this family's pitch: `R10Unorm_X6`, which [`crate::cases::textures`] already
//!   compares, behaves the same, while `RGB10A2Unorm_PQ` and `BGR10A2Unorm_sRGB` — the same texel
//!   size, the same allocation — come back in full. So the pair `RGB10A2Unorm_PQ` against
//!   `RGB10A2Unorm_HLG` is a site where two formats a side would plausibly treat alike answer
//!   differently, and the family records both.
//!
//! # The thirty-five no descriptor may name
//!
//! Settled, not pending, and **not to be re-opened as a coverage gap**:
//!
//! `MTLPixelFormatRGB8Unorm`, `MTLPixelFormatRGB8Snorm`, `MTLPixelFormatRGB8Uint`,
//! `MTLPixelFormatRGB8Sint`, `MTLPixelFormatRGB16Unorm`, `MTLPixelFormatRGB16Snorm`,
//! `MTLPixelFormatRGB16Uint`, `MTLPixelFormatRGB16Sint`, `MTLPixelFormatRGB16Float`,
//! `MTLPixelFormatRGB16Bfloat`, `MTLPixelFormatRGB32Uint`, `MTLPixelFormatRGB32Sint`,
//! `MTLPixelFormatRGB32Float`, `MTLPixelFormatX24_Stencil8`, `MTLPixelFormatBGRA10Uint_PACKED`,
//! `MTLPixelFormatBGRA10_XR_PACKED`, `MTLPixelFormatBGRA10_XR_PACKED_sRGB`,
//! `MTLPixelFormatBGRA10Unorm_PACKED`, `MTLPixelFormatBGRA10Unorm_PACKED_sRGB`,
//! `MTLPixelFormatBGRA10Unorm_PACKED_HLG`, `MTLPixelFormatBGRA10Unorm_PACKED_PQ`,
//! `MTLPixelFormatBGRA10Uint`, `MTLPixelFormatBGRA10Unorm`, `MTLPixelFormatBGRA10Unorm_sRGB`,
//! `MTLPixelFormatBGRA10Unorm_HLG`, `MTLPixelFormatBGRA10Unorm_PQ`, `MTLPixelFormatR10Uint_X6`,
//! `MTLPixelFormatR10Uint_PACKED`, `MTLPixelFormatRG10Uint_X12`, `MTLPixelFormatRG10Uint_PACKED`,
//! `MTLPixelFormatR12Uint_X4`, `MTLPixelFormatR12Uint_PACKED`, `MTLPixelFormatRG12Uint_X8`,
//! `MTLPixelFormatRG12Uint_PACKED`, `MTLPixelFormatRGB10A8Uint_2P`.
//!
//! **Measured** on an Apple M3 Max under macOS 27.0.1, plain Metal with no API validation layer,
//! in a throwaway process forked per (format, texture type) so a shape that ended the process cost
//! that shape alone: for every one of these 35 values, in **every** one of the eight
//! non-multisample texture types, in both storage modes, and for the size query as well as the
//! creator, `-[MTLTextureDescriptorInternal validateWithDevice:]` fails its assertion with
//!
//! ```text
//! Texture Descriptor Validation
//! MTLTextureDescriptor has invalid pixelFormat (N).
//! ```
//!
//! and the process ends on SIGABRT. 35 formats x 10 descriptor shapes, 350 shapes, every one fatal
//! and not one of them answering nil. The same 35 end the process from the one public per-format
//! device query too: `minimumLinearTextureAlignmentForPixelFormat:` asserts **`MTLPixelFormatInvalid`
//! is not supported on this device**, naming the value `Invalid` rather than echoing it back,
//! which is Metal saying the raw value is not a pixel format of this device.
//!
//! So nothing reached from `MTLDevice` or `MTLTextureDescriptor` answers for one of them: a
//! texture, a view, a heap size and the alignment query each end the process and nothing returns.
//! `MTLTextureDescriptor`'s own `description` still prints each value's Metal name, which is how
//! `scripts/regen-metal-enums/` learned the names and why they are declared at all — a name in a
//! description is not a format a device holds.
//!
//! The list is held by `tests::UNREACHABLE` and `tests::the_unreachable_and_the_tabled_are_the_sixty_eight`,
//! under `#[cfg(test)]` on purpose: `scripts/conformance-coverage/` reads a `PixelFormat::` variant
//! in a module with a recorded Metal run as a value the module put to Metal, and these 35 are the
//! opposite of that. A test holds the split total and disjoint; the prose above is what a reader
//! finds.
//!
//! # Why every case here is safe
//!
//! Nothing here reaches the GPU: the size query is CPU arithmetic over a descriptor, the textures
//! are a 64 by 64 level at most, and the round trip is the host copying bytes in and out. No
//! shader is compiled, no command buffer is made and nothing is dispatched.
//!
//! The real hazard is the whole subject of this family and it is sharper here than anywhere else
//! in the suite: a descriptor whose pixel format Metal does not hold **ends Metal's process**, a
//! failed assertion rather than a nil answer, and `heapTextureSizeAndAlignWithDescriptor:` reaches
//! that validation too — so the "pure getter" is not safe for an arbitrary format value. Every
//! format in this family's tables is one measured to survive it. The exclusions are measured, not
//! reasoned, and are named where they are made:
//!
//! - the 35 above, in every texture type and in the size query;
//! - `Type1D`, `Type1DArray` and `TextureBuffer` for the six ASTC footprints, where Metal's
//!   validation stops the descriptor with "compressed pixel formats require texture type of
//!   MTLTextureType2D[Array], MTLTextureType3D or MTLTextureTypeCube[Array]";
//! - the multisample types everywhere, which `heap_metal_sizes` records as a process-ending shape
//!   for a format Metal describes no texture of;
//! - `ShaderAtomic` usage, which Metal ends its own process on for any format but `R32Uint`.
//!
//! `device_tests::no_descriptor_of_the_family_is_one_metals_validation_stops` is what keeps that
//! honest rather than hopeful, the same guard and for the same reason as `heap_metal_sizes`': the
//! table is replayed on the runtime first, where `Error::Invalid` — "Metal's validation stops it,
//! ending the process" — is a recorded value and nothing else, and a table that answers `Invalid`
//! anywhere never reaches Metal. It passed over every descriptor of this family on MoltenVK over
//! this M3 Max before any of it was recorded.
//!
//! # Why a case stops at its first refusal
//!
//! The runtime holds **no image of any of the 33**: every one of them is refused
//! `Unsupported { feature: Format, needs: NotImplemented }`, measured over every descriptor of
//! every case of this family. A case that recorded a refusal at each site and went on — which is
//! what `heap_metal_sizes` does, because the runtime answers all of its descriptors — would print
//! some 2900 `Refused` rows and need a `known-gaps.tsv` row for each, which is a ledger nobody
//! reads and therefore no ledger. So each case stops at the first site it is refused, and the
//! family's six refusal rows stand for the gap: the full list of the 33 is in the orchestrator's
//! todos, where work belongs, and the recordings hold Metal's answer for every one, which is what
//! an implementation of them will be held to.
//!
//! # Exactness
//!
//! Every field is a number, a bool or bytes, and the family declares no tolerance. A format the
//! runtime holds no image of refuses its descriptor with a typed error, which the comparison reads
//! as [`Refused`](crate::verdict::Verdict::Refused) against Metal's answer — never a pass. That is
//! the point: the gap was a number in a generated document and it is now a row of the report
//! naming the format, the descriptor and both answers.

use crate::case::{Abort, Case, Input, MetalSource, Side, TextureRole, TextureSpec};
use crate::texture_query::{HeapTextureQuery as Q, Storage};
use crate::transcript::{Aspect, ByteLayout, Element, Gaps, Node, Observed, Region, Sites, Value};
use metal2vulkan::metal::{PixelFormat, TextureType};

/// The two storage modes every case of the family sweeps.
const BOTH_STORAGE: [Storage; 2] = [Storage::Shared, Storage::Private];

/// The round trip's texture side, in texels.
const SIDE: u32 = 16;

/// The round trip's row pitch, in bytes: wide enough for a 16-byte texel at [`SIDE`] texels, so one
/// pitch serves every format of the family and the bytes a format's own row does **not** reach are
/// part of what is compared.
const PITCH: u32 = 256;

/// The byte every read-back destination holds before the copy, so a byte the copy did not write is
/// a recorded value rather than a zero nobody can tell from a written one. The same byte
/// [`crate::cases::textures`] uses.
pub const BEFORE: u8 = 0xee;

/// The case names, as the report and the recordings spell them.
pub const ASTC_FOOTPRINTS: &str = "format-footprints-astc-4x2-and-8x4";
pub const PLAIN_STRAGGLERS: &str = "format-footprints-plain-stragglers";
pub const ONE_PLANE_YCBCR: &str = "format-footprints-one-plane-ycbcr";
pub const PADDED_WIDE_CHANNELS: &str = "format-footprints-padded-wide-channels";
pub const TWO_PLANE_TEN_BIT: &str = "format-footprints-two-plane-ten-bit";
pub const PACKED_WITHOUT_TEXTURES: &str = "format-footprints-packed-without-textures";

/// The site a descriptor's two answers sit under, and the round trip's own three. The first two
/// are the names [`crate::cases::heap_metal_sizes`] uses, so a reader of one family's diff reads
/// the other's.
mod site {
    /// The heap size query's answer.
    pub const QUERY: &str = "size-and-align";
    /// Whether `newTextureWithDescriptor:` answered a texture.
    pub const TEXTURE: &str = "makes-texture";
    /// What a fresh level 0 reads.
    pub const FRESH: &str = "fresh";
    /// What `replaceRegion:` answered.
    pub const WROTE: &str = "replace_region";
    /// What the level reads after the write.
    pub const BACK: &str = "round-trip";
}

/// The five texture types every format of this family is asked in, with the depth and slice count
/// each takes.
///
/// `Type1D`, `Type1DArray` and `TextureBuffer` are in [`LINEAR_TYPES`] instead, because the six
/// ASTC footprints must not be asked in them: Metal's validation stops a compressed format's
/// one-dimensional or buffer descriptor and ends its process. The two multisample types are in
/// neither, for the reason `heap_metal_sizes` states.
const BLOCKY_TYPES: [(TextureType, u32, u32); 5] = [
    (TextureType::Type2D, 1, 1),
    (TextureType::Type2DArray, 1, 2),
    (TextureType::Cube, 1, 1),
    (TextureType::CubeArray, 1, 2),
    (TextureType::Type3D, 2, 1),
];

/// The three texture types a linear format takes on top of [`BLOCKY_TYPES`].
const LINEAR_TYPES: [(TextureType, u32, u32); 3] = [
    (TextureType::Type1D, 1, 1),
    (TextureType::Type1DArray, 1, 2),
    (TextureType::TextureBuffer, 1, 1),
];

/// ASTC's 4x2 and 8x4 footprints, LDR, sRGB and HDR: the six block formats Apple's headers do not
/// declare and `astc_textures` therefore could not reach, which Metal nonetheless makes a 2D
/// texture of.
const ASTC: [PixelFormat; 6] = [
    PixelFormat::ASTC4x2SRGB,
    PixelFormat::ASTC8x4SRGB,
    PixelFormat::ASTC4x2LDR,
    PixelFormat::ASTC8x4LDR,
    PixelFormat::ASTC4x2HDR,
    PixelFormat::ASTC8x4HDR,
];

/// The two formats of the 68 that are an ordinary colour texel in an ordinary layout:
/// `BGRA4Unorm`, four bits a channel, and `ARGB8Unorm_sRGB`, the sRGB twin of the rotated
/// `ARGB8Unorm` the suite already compares.
const PLAIN: [PixelFormat; 2] = [PixelFormat::BGRA4Unorm, PixelFormat::ARGB8UnormSRGB];

/// Apple's **one-plane** YCbCr formats: 4:2:2 and 4:4:4, their sRGB twins, the 4:4:4 with alpha,
/// the reversed 4:2:2, and the three transfer functions of the 10-bit 4:4:4 (PQ, HLG and extended
/// range).
///
/// They share a shape the rest of the family does not: Metal answers **nil** for a cube and a cube
/// array of any of them and a texture for the other six types, and their heap size has an
/// alignment of 16 rather than 128.
const ONE_PLANE: [PixelFormat; 9] = [
    PixelFormat::YCBCR84221P,
    PixelFormat::YCBCR104441P,
    PixelFormat::YCBCR84221PSRGB,
    PixelFormat::YCBCR104441PSRGB,
    PixelFormat::YCBCRA84441P,
    PixelFormat::YCBCR84221PREV,
    PixelFormat::YCBCR104441PPQ,
    PixelFormat::YCBCR104441PXR,
    PixelFormat::YCBCR104441PHLG,
];

/// The padded 10- and 12-bit display formats: a 10- or 12-bit code in the top bits of its own
/// little-endian word with the low bits unread, and the transfer-function twins of `RGB10A2Unorm`
/// and `BGR10A2Unorm`.
///
/// `RGB10A2Unorm_PQ` and `BGR10A2Unorm_sRGB`/`_PQ` round-trip their bytes and the other seven do
/// not, which is why they are one case: the pair that differs sits beside the pair that does not.
const PADDED: [PixelFormat; 10] = [
    PixelFormat::RGB10A2UnormPQ,
    PixelFormat::R12UnormX4,
    PixelFormat::R12UnormX4PQ,
    PixelFormat::RG12UnormX8,
    PixelFormat::R10UnormX6PQ,
    PixelFormat::R10UnormX6HLG,
    PixelFormat::R12UnormX4HLG,
    PixelFormat::RGB10A2UnormHLG,
    PixelFormat::BGR10A2UnormSRGB,
    PixelFormat::BGR10A2UnormPQ,
];

/// Apple's two-plane 10-bit-plus-alpha format and its three transfer-function twins, whose heap
/// size starts at 16385 bytes for a 1 by 1 texture: plane 0 is 16 KiB-aligned and plane 1 follows
/// it.
const TWO_PLANE: [PixelFormat; 4] = [
    PixelFormat::RGB10A82P,
    PixelFormat::RGB10A82PSRGB,
    PixelFormat::RGB10A82PPQ,
    PixelFormat::RGB10A82PHLG,
];

/// The two formats of the 68 whose descriptor Metal accepts and whose texture it refuses:
/// `R10Unorm_PACKED` and `RG10Unorm_PACKED` answer 0 bytes at an alignment of 16 in every texture
/// type, and `newTextureWithDescriptor:` answers nil for every one.
///
/// The `heap_metal_sizes` shape where the 0-byte answer and the nil texture are recorded together,
/// which is the only way either one means anything. They take no round trip: there is no texture
/// to write to.
const PACKED_NO_TEXTURE: [PixelFormat; 2] =
    [PixelFormat::R10UnormPACKED, PixelFormat::RG10UnormPACKED];

/// A case over one family of formats: Metal's size query and creation answer for each format in
/// every texture type it survives, the width and height sweeps that pin its footprint, and the
/// host round trip where Metal makes a texture.
pub struct FootprintCase {
    name: &'static str,
    queries: Vec<Q>,
    /// The formats a round trip runs for, in order. Empty where Metal makes no texture of any
    /// format of the case.
    round_trips: Vec<PixelFormat>,
}

/// Every case of the family.
pub fn all() -> Vec<FootprintCase> {
    vec![
        astc_footprints(),
        plain_stragglers(),
        one_plane_ycbcr(),
        padded_wide_channels(),
        two_plane_ten_bit(),
        packed_without_textures(),
    ]
}

/// The descriptors one format contributes: every texture type of `types` in both storage modes at
/// 64 by 64, then the width sweep at height 1 and the height sweep at width 16, both `Private`.
fn queries_for(format: PixelFormat, types: &[(TextureType, u32, u32)], out: &mut Vec<Q>) {
    for &(texture_type, depth, slices) in types {
        for storage in BOTH_STORAGE {
            out.push(Q::texture_2d(format, [64, 64], storage).of_type(texture_type, depth, slices));
        }
    }
    for width in 1..=17 {
        out.push(Q::texture_2d(format, [width, 1], Storage::Private));
    }
    // From 2, because the width sweep's 16th row **is** 16 by 1 and a site name is a descriptor's
    // own fields: two sites of one name under one parent is a transcript the comparison cannot
    // pair, and it says so rather than guessing.
    for height in 2..=17 {
        out.push(Q::texture_2d(format, [16, height], Storage::Private));
    }
}

/// The eight texture types a linear format takes.
fn all_types() -> Vec<(TextureType, u32, u32)> {
    BLOCKY_TYPES
        .iter()
        .chain(LINEAR_TYPES.iter())
        .copied()
        .collect()
}

/// A case over `formats`, each asked in `types`, with a round trip for each where `round_trips`.
fn case_over(
    name: &'static str,
    formats: &[PixelFormat],
    types: &[(TextureType, u32, u32)],
    round_trips: bool,
) -> FootprintCase {
    let mut queries = Vec::new();
    for &format in formats {
        queries_for(format, types, &mut queries);
    }
    FootprintCase {
        name,
        queries,
        round_trips: if round_trips {
            formats.to_vec()
        } else {
            Vec::new()
        },
    }
}

/// ASTC's 4x2 and 8x4 footprints: the one case whose formats are block formats, the one whose
/// height sweep is the only site telling its formats apart, and the one asked in five texture
/// types rather than eight.
fn astc_footprints() -> FootprintCase {
    case_over(ASTC_FOOTPRINTS, &ASTC, &BLOCKY_TYPES, true)
}

/// `BGRA4Unorm` and `ARGB8Unorm_sRGB`: an ordinary 2-byte texel and an ordinary 4-byte one, whose
/// answers are what a reader compares the rest of the family against.
fn plain_stragglers() -> FootprintCase {
    case_over(PLAIN_STRAGGLERS, &PLAIN, &all_types(), true)
}

/// Apple's nine one-plane YCbCr formats, whose cube and cube-array descriptors answer a size and
/// **no texture**.
fn one_plane_ycbcr() -> FootprintCase {
    case_over(ONE_PLANE_YCBCR, &ONE_PLANE, &all_types(), true)
}

/// The padded 10- and 12-bit display formats and the transfer-function twins of the two
/// `*10A2Unorm` ones.
fn padded_wide_channels() -> FootprintCase {
    case_over(PADDED_WIDE_CHANNELS, &PADDED, &all_types(), true)
}

/// The two-plane 10-bit-plus-alpha family, whose plane 0 is 16 KiB-aligned.
fn two_plane_ten_bit() -> FootprintCase {
    case_over(TWO_PLANE_TEN_BIT, &TWO_PLANE, &all_types(), true)
}

/// `R10Unorm_PACKED` and `RG10Unorm_PACKED`: 0 bytes and no texture, in every texture type, and so
/// no round trip.
fn packed_without_textures() -> FootprintCase {
    case_over(
        PACKED_WITHOUT_TEXTURES,
        &PACKED_NO_TEXTURE,
        &all_types(),
        false,
    )
}

/// The site a creator's answer records at, with the side's own error kept out of it, exactly as
/// [`crate::cases::textures`] does and for the same reason: Metal has one way to refuse a texture
/// and no typed error at all, so the comparable fact is whether a texture was made.
fn made_value(made: bool) -> Value {
    if made {
        Value::ok(Value::Absent)
    } else {
        Value::error("NotMade", Value::Absent)
    }
}

/// The whole destination of a read-back, before the copy.
fn before() -> Vec<u8> {
    vec![BEFORE; (PITCH * SIDE) as usize]
}

/// The bytes `replace_region` writes for `format`: a pattern seeded by the format's own ordinal, so
/// two formats of one texel size write different bytes and a read-back cannot be another format's.
fn written(format: PixelFormat) -> Vec<u8> {
    let seed = format.as_raw() as u8;
    (0..(PITCH * SIDE) as usize)
        .map(|at| (at as u8).wrapping_mul(31).wrapping_add(seed))
        .collect()
}

/// The layout a read-back's bytes compare under: [`SIDE`] rows of [`PITCH`] bytes, every byte an
/// element, so the bytes a format's own row does not reach are compared rather than ignored. That
/// is what makes "how many bytes came back" a compared fact instead of an inference.
fn layout() -> ByteLayout {
    ByteLayout {
        region: Region::Texture {
            level: 0,
            slice: 0,
            aspect: Aspect::Color,
            sample: 0,
            origin: [0, 0, 0],
            size: [SIDE, SIDE, 1],
        },
        element: Element::U8,
        row_offset: 0,
        row_elements: PITCH,
        row_pitch: PITCH,
        rows: SIDE,
        image_pitch: PITCH * SIDE,
        images: 1,
        gaps: Gaps::Compared,
    }
}

impl FootprintCase {
    /// One format's round trip, under a branch of the format's own name.
    fn round_trip(
        &self,
        format: PixelFormat,
        side: &mut dyn Side,
        sites: &mut Sites,
    ) -> Result<(), Abort> {
        let at = |leaf: &str| format!("{}/{leaf}", format.name());
        let spec = TextureSpec::new(format, [SIDE, SIDE], PITCH, TextureRole::Read);
        let mut node = Node::branch(format.name(), Vec::new());
        let texture = match side.texture_2d_in_role(&spec, &[]) {
            Ok(texture) => {
                node = node.with(Node::leaf(site::TEXTURE, Observed::value(made_value(true))));
                texture
            }
            Err(failure) => {
                sites.push(node.with(Node::leaf(
                    site::TEXTURE,
                    Observed::value(made_value(false)),
                )));
                return Err(Abort {
                    at: at(site::TEXTURE),
                    failure,
                });
            }
        };
        let before = before();
        match side.read_texture_over(texture, [SIDE, SIDE], PITCH, &before) {
            Ok(bytes) => {
                node = node.with(Node::leaf(site::FRESH, Observed::bytes(bytes, layout())))
            }
            Err(failure) => {
                sites.push(node.with(Node::leaf(site::FRESH, Observed::value(failure.recorded()))));
                return Err(Abort {
                    at: at(site::FRESH),
                    failure,
                });
            }
        }
        match side.replace_region(texture, [SIDE, SIDE], PITCH, &written(format)) {
            Ok(()) => {
                node = node.with(Node::leaf(
                    site::WROTE,
                    Observed::value(Value::ok(Value::Absent)),
                ))
            }
            Err(failure) => {
                sites.push(node.with(Node::leaf(site::WROTE, Observed::value(failure.recorded()))));
                return Err(Abort {
                    at: at(site::WROTE),
                    failure,
                });
            }
        }
        match side.read_texture_over(texture, [SIDE, SIDE], PITCH, &before) {
            Ok(bytes) => {
                sites.push(node.with(Node::leaf(site::BACK, Observed::bytes(bytes, layout()))));
                Ok(())
            }
            Err(failure) => {
                sites.push(node.with(Node::leaf(site::BACK, Observed::value(failure.recorded()))));
                Err(Abort {
                    at: at(site::BACK),
                    failure,
                })
            }
        }
    }
}

impl Case for FootprintCase {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        // The descriptors are the whole of what the case feeds, and the order it asks them in: a
        // table that gains, loses or re-points a row is a change to the case's inputs, so a
        // recording cannot decay into the answers of descriptors nobody asked about.
        let mut words = Vec::new();
        for query in &self.queries {
            words.extend(query.words());
        }
        let mut trips = Vec::new();
        for format in &self.round_trips {
            trips.extend(format.as_raw().to_le_bytes());
            trips.extend(written(*format));
        }
        Ok(vec![
            Input::bytes("descriptors", &words),
            Input::bytes(
                "asks",
                b"heapTextureSizeAndAlign,newTextureWithDescriptor".as_slice(),
            ),
            Input::bytes("before", &before()),
            Input::bytes("round-trips", &trips),
        ])
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        for query in &self.queries {
            let answered = match side.heap_texture_size_and_align(query) {
                Ok(answered) => Value::ok(answered.value()),
                Err(failure) => {
                    sites.push(
                        Node::branch(query.site(), Vec::new())
                            .with(Node::leaf(site::QUERY, Observed::value(failure.recorded()))),
                    );
                    return Err(Abort {
                        at: format!("{}/{}", query.site(), site::QUERY),
                        failure,
                    });
                }
            };
            let made = match side.makes_texture(query) {
                Ok(made) => Value::ok(Value::Bool(made)),
                Err(failure) => {
                    sites.push(
                        Node::branch(query.site(), Vec::new())
                            .with(Node::leaf(site::QUERY, Observed::value(answered)))
                            .with(Node::leaf(
                                site::TEXTURE,
                                Observed::value(failure.recorded()),
                            )),
                    );
                    return Err(Abort {
                        at: format!("{}/{}", query.site(), site::TEXTURE),
                        failure,
                    });
                }
            };
            sites.push(
                Node::branch(query.site(), Vec::new())
                    .with(Node::leaf(site::QUERY, Observed::value(answered)))
                    .with(Node::leaf(site::TEXTURE, Observed::value(made))),
            );
        }
        for &format in &self.round_trips {
            self.round_trip(format, side, sites)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// The 35 values of the 68 that **no case of this suite may put in a texture descriptor**,
    /// settled rather than pending, with the measurement in the module's own documentation.
    ///
    /// Under `#[cfg(test)]` on purpose: `scripts/conformance-coverage/` reads a `PixelFormat::`
    /// variant named in a module that has a recorded Metal run as a value the module put to
    /// Metal, and these 35 are the opposite of that. It excludes a `#[cfg(test)]` item, so the
    /// list can be held by a test without the coverage axis counting it as compared.
    const UNREACHABLE: [PixelFormat; 35] = [
        PixelFormat::RGB8Unorm,
        PixelFormat::RGB8Snorm,
        PixelFormat::RGB8Uint,
        PixelFormat::RGB8Sint,
        PixelFormat::RGB16Unorm,
        PixelFormat::RGB16Snorm,
        PixelFormat::RGB16Uint,
        PixelFormat::RGB16Sint,
        PixelFormat::RGB16Float,
        PixelFormat::RGB16Bfloat,
        PixelFormat::RGB32Uint,
        PixelFormat::RGB32Sint,
        PixelFormat::RGB32Float,
        PixelFormat::X24Stencil8,
        PixelFormat::BGRA10UintPACKED,
        PixelFormat::BGRA10XRPACKED,
        PixelFormat::BGRA10XRPACKEDSRGB,
        PixelFormat::BGRA10UnormPACKED,
        PixelFormat::BGRA10UnormPACKEDSRGB,
        PixelFormat::BGRA10UnormPACKEDHLG,
        PixelFormat::BGRA10UnormPACKEDPQ,
        PixelFormat::BGRA10Uint,
        PixelFormat::BGRA10Unorm,
        PixelFormat::BGRA10UnormSRGB,
        PixelFormat::BGRA10UnormHLG,
        PixelFormat::BGRA10UnormPQ,
        PixelFormat::R10UintX6,
        PixelFormat::R10UintPACKED,
        PixelFormat::RG10UintX12,
        PixelFormat::RG10UintPACKED,
        PixelFormat::R12UintX4,
        PixelFormat::R12UintPACKED,
        PixelFormat::RG12UintX8,
        PixelFormat::RG12UintPACKED,
        PixelFormat::RGB10A8Uint2P,
    ];

    /// Every format of the family's tables, with the 35 unreachable ones, is exactly the 68 values
    /// `docs/CONFORMANCE-COVERAGE.md` read as compared against nothing — no format in two tables
    /// and none in neither.
    ///
    /// The 68 are not derivable from the translator's own sources: the formats it holds no image of
    /// are a superset, since it holds none of several the suite already compares through other
    /// families. So the split is spelled here, and what this test holds is the one thing that can
    /// rot on its own — a stream moving a format between the piles must keep them total and
    /// disjoint.
    #[test]
    fn the_unreachable_and_the_tabled_are_the_sixty_eight() {
        let mut tabled: Vec<PixelFormat> = Vec::new();
        tabled.extend(ASTC);
        tabled.extend(PLAIN);
        tabled.extend(ONE_PLANE);
        tabled.extend(PADDED);
        tabled.extend(TWO_PLANE);
        tabled.extend(PACKED_NO_TEXTURE);
        let compared: BTreeSet<PixelFormat> = tabled.iter().copied().collect();
        assert_eq!(compared.len(), tabled.len(), "a format is in two tables");
        assert_eq!(compared.len(), 33, "the compared tables hold other than 33");
        let settled: BTreeSet<PixelFormat> = UNREACHABLE.iter().copied().collect();
        assert_eq!(settled.len(), UNREACHABLE.len(), "a format repeats");
        assert!(
            compared.is_disjoint(&settled),
            "a format is both compared and settled unreachable"
        );
        assert_eq!(compared.len() + settled.len(), 68);
    }

    /// Every case names its own formats and nothing else, every format of a case's table has a
    /// descriptor in it, and a case's round trips are its own formats. Without this a table could
    /// lose a format to an editing slip and the family's coverage would shrink with every case
    /// still green.
    #[test]
    fn every_case_asks_about_every_format_of_its_table() {
        let tables: [(&str, &[PixelFormat], bool); 6] = [
            (ASTC_FOOTPRINTS, &ASTC, true),
            (PLAIN_STRAGGLERS, &PLAIN, true),
            (ONE_PLANE_YCBCR, &ONE_PLANE, true),
            (PADDED_WIDE_CHANNELS, &PADDED, true),
            (TWO_PLANE_TEN_BIT, &TWO_PLANE, true),
            (PACKED_WITHOUT_TEXTURES, &PACKED_NO_TEXTURE, false),
        ];
        let cases = all();
        assert_eq!(cases.len(), tables.len());
        for (case, (name, formats, trips)) in cases.iter().zip(tables) {
            assert_eq!(case.name, name);
            let wanted: BTreeSet<PixelFormat> = formats.iter().copied().collect();
            let asked: BTreeSet<PixelFormat> =
                case.queries.iter().map(|query| query.format).collect();
            assert_eq!(
                asked, wanted,
                "{name} asks about other formats than its table"
            );
            let tripped: BTreeSet<PixelFormat> = case.round_trips.iter().copied().collect();
            assert_eq!(
                tripped,
                if trips { wanted } else { BTreeSet::new() },
                "{name} round-trips other formats than its table"
            );
        }
    }

    /// No case asks a block format in a one-dimensional or buffer texture type. Metal's validation
    /// stops such a descriptor and **ends its process**, so this is the guard that keeps a later
    /// edit from turning a recording run into a crash.
    #[test]
    fn no_block_format_is_asked_in_a_linear_texture_type() {
        let block: BTreeSet<PixelFormat> = ASTC.iter().copied().collect();
        for case in all() {
            for query in &case.queries {
                if block.contains(&query.format) {
                    assert!(
                        !matches!(
                            query.texture_type,
                            TextureType::Type1D
                                | TextureType::Type1DArray
                                | TextureType::TextureBuffer
                        ),
                        "{}: {} would end Metal's process",
                        case.name,
                        query.site()
                    );
                }
            }
        }
    }

    /// No case asks a multisample descriptor, and none sets the `ShaderAtomic` usage bit: both end
    /// Metal's process for the formats of this family, as `heap_metal_sizes` records.
    #[test]
    fn no_descriptor_is_multisampled_or_atomic() {
        for case in all() {
            for query in &case.queries {
                assert_eq!(query.samples, 1, "{}: {}", case.name, query.site());
                assert!(
                    !query
                        .usage
                        .contains(metal2vulkan::metal::TextureUsage::SHADER_ATOMIC),
                    "{}: {}",
                    case.name,
                    query.site()
                );
            }
        }
    }

    /// The height sweep is what tells ASTC's 4x2 footprint from its 8x4 one, so it must reach a
    /// height past both block heights. A sweep that stopped at 4 would compare two footprints that
    /// answer alike at every height it asked.
    #[test]
    fn the_height_sweep_passes_both_astc_block_heights() {
        let case = astc_footprints();
        let heights: BTreeSet<u32> = case
            .queries
            .iter()
            .filter(|query| query.size[0] == 16)
            .map(|query| query.size[1])
            .collect();
        for height in [1, 2, 3, 4, 5, 9, 17] {
            assert!(heights.contains(&height), "the sweep skips height {height}");
        }
        // No descriptor appears twice: a site name is a descriptor's own fields, so a repeat is a
        // transcript whose sites the comparison cannot pair. The width sweep's 16th row and the
        // height sweep's first would be the same descriptor, which is why the latter starts at 2.
        for case in all() {
            let mut names: Vec<String> = case.queries.iter().map(|query| query.site()).collect();
            let before = names.len();
            names.sort();
            names.dedup();
            assert_eq!(before, names.len(), "{} repeats a descriptor", case.name);
        }
    }

    /// The pattern each round trip writes differs from the byte the destination holds before a
    /// read, and differs between formats, so a read-back that copied nothing — or that copied
    /// another format's bytes — reads as a difference rather than as a match.
    #[test]
    fn each_round_trips_pattern_is_its_own_and_not_the_byte_before_it() {
        let mut patterns = BTreeSet::new();
        for case in all() {
            for &format in &case.round_trips {
                let bytes = written(format);
                assert_eq!(bytes.len(), before().len(), "{}", format.name());
                assert!(
                    bytes.iter().any(|&byte| byte != BEFORE),
                    "{}",
                    format.name()
                );
                assert!(
                    patterns.insert(bytes),
                    "{} repeats a pattern",
                    format.name()
                );
            }
        }
        assert_eq!(patterns.len(), 31, "other than 31 formats round-trip");
    }

    /// The round trip's pitch is wide enough for every format of the family, so one pitch serves
    /// all of them and the bytes past a format's own row are the compared padding rather than a
    /// buffer overrun.
    #[test]
    fn the_pitch_holds_the_widest_texel_the_family_has() {
        assert!(PITCH >= SIDE * 16, "a 16-byte texel would not fit");
    }
}

/// The safety guard this family rests on, run on a device.
///
/// The same guard `heap_metal_sizes` carries, in the family that needs it most: the subject here is
/// formats whose descriptor Metal's own validation may stop, which ends Metal's process. The
/// runtime's model is what tells the two apart — `Error::Invalid` is "Metal's validation stops it,
/// ending the process", `Error::Nil` is "plain Metal answers nil, the process going on" — so a
/// table that makes the runtime answer `Invalid` anywhere must never be recorded.
///
/// Unlike `heap_metal_sizes`' guard this one tolerates every **other** typed error. The runtime
/// holds no image of any of this family's 33 formats and refuses each `NotImplemented(Format)`,
/// which is the gap this family exists to make visible; only `Invalid` says the descriptor itself
/// is one Metal stops.
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod device_tests {
    use super::*;
    use crate::device::{self, Notices, ScratchCache, Venues};
    use crate::runtime_side::RuntimeSide;

    /// The venue the guard opens: a GPU where the tests are built for one, else a CPU ICD.
    const VENUE: Venues = if cfg!(m2v_gpu) {
        Venues::Gpu
    } else {
        Venues::Cpu
    };

    #[test]
    fn no_descriptor_of_the_family_is_one_metals_validation_stops() {
        let mut notices = Notices::default();
        let cache = ScratchCache::new().unwrap();
        let runtime = device::open(VENUE, &notices, &cache).unwrap();
        let mut side = RuntimeSide::new(
            &runtime,
            notices.open("no_descriptor_of_the_family_is_one_metals_validation_stops"),
        );
        let mut stopped = Vec::new();
        for case in all() {
            for query in &case.queries {
                for answer in [
                    side.heap_texture_size_and_align(query)
                        .err()
                        .map(|failure| failure.recorded()),
                    side.makes_texture(query)
                        .err()
                        .map(|failure| failure.recorded()),
                ] {
                    let Some(recorded) = answer else {
                        continue;
                    };
                    if format!("{recorded:?}").contains("Invalid") {
                        stopped.push(format!("{} {}: {recorded:?}", case.name, query.site()));
                    }
                }
            }
        }
        assert!(
            stopped.is_empty(),
            "these descriptors answer `Invalid` on the runtime, so Metal's validation stops them \
             and record mode would end Metal's process; take them out of the table:\n{}",
            stopped.join("\n")
        );
    }
}
