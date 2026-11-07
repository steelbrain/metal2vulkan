//! `MTLReadWriteTextureTier` and `MTLArgumentBuffersTier`: the two tiers a device reports, and the
//! behaviour each one **promises**, per pixel format and per texture type.
//!
//! `docs/CONFORMANCE-COVERAGE.md` read both types as 0 of 3 and 0 of 2 values compared, and the
//! suite's sources named neither. They are device capabilities the runtime answers to every
//! program, so nothing checked our answer against Metal's, and the integer alone is the smaller
//! half of the question: **a side that reports tier 2 and then refuses a format tier 2 names is
//! wrong in a way no comparison of the number can see.** So each case here records the tier and
//! what it promises, and the family's own invariant is the one a reader should look for first:
//! *no format the side's own tier promises is a format the side refuses.*
//!
//! # Where Metal enforces the read-write tier, measured
//!
//! Not at the descriptor, and not at the texture type. Measured on this Apple M3 Max under macOS
//! 27.0.1 (plain Metal, no API validation layer), over the 30 uncompressed colour formats of
//! `ROWS` in each of the **eight non-multisample texture types**, descriptors of
//! `ShaderRead | ShaderWrite` usage and private storage: Metal answered a texture for all 240,
//! each reporting `usage` 0x3, and answered nil for none. A 1D read-write texture and a
//! `TextureBuffer` one are made as readily as a 2D one, and so are a cube and a 3D texture.
//!
//! The **texture type** is gated by the language rather than by the device: `access::read_write`
//! on a multisample type is a `static_assert` of Metal's own headers,
//! *"invalid access value 'a' for 'texture2d_ms'"*, for every component type (float, half, int,
//! short, uint, ushort) — so the two multisample types can carry no read-write argument at all,
//! which is why no case here states one and why the type axis of this family is eight wide. Every
//! other type compiles for every component type.
//!
//! What is left is the format, and the only place a format's read-write support can be observed is
//! a **dispatch that reads and writes the texture**, which is what
//! `read-write-tier-round-trip-*` does.
//!
//! ## The shapes no case may state
//!
//! Metal's descriptor validation ends the process rather than answering nil for these, measured in
//! a throwaway process per cell on the same host, and `-[MTLTextureDescriptorInternal
//! validateWithDevice:]:1421` is the frame in each:
//!
//! - a **compressed** format (`BC1_RGBA`, `ASTC_4x4_sRGB`) in `Type1D`, `Type1DArray` or
//!   `TextureBuffer`: *`"compressed pixel formats require texture type of
//!   MTLTextureType2D[Array], MTLTextureType3D or MTLTextureTypeCube[Array]"`*;
//! - a **depth or stencil** format (`Depth32Float`, `Stencil8`, `Depth32Float_Stencil8`) in
//!   `Type1D`, `Type1DArray`, `Type3D` or `TextureBuffer`: *`"depth/stencil pixel formats require
//!   texture type of MTLTextureType2D[Array] or MTLTextureTypeCube[Array]"`*.
//!
//! In the types that are left, each of those five formats answers a read-write texture of `usage`
//! 0x3 — Metal refuses read-write usage on a compressed or depth format nowhere — so
//! `BLOCKED` holds them with the types they may be asked in, and no fatal cell is reachable from
//! a table of this family.
//!
//! # What a dispatch finds, and why the controls are in the table
//!
//! `readWriteTextureSupport` on this host answers `MTLReadWriteTextureTier2`, whose formats are
//! Metal's own two sets, the ones the runtime derives its answer from
//! (`READ_WRITE_TIER1_FORMATS` and `READ_WRITE_TIER2_FORMATS` in
//! `crates/metal2vulkan/src/device/metal.rs`, which no public path exports): three single-channel
//! 32-bit formats at tier 1, and fifteen more at tier 2. A sweep of those eighteen alone would pass on an implementation that supported exactly
//! them, which is the opposite of what this suite is for: **twelve formats neither tier names
//! round-trip a read-write texel on this Metal**, measured by the dispatch this family runs, and
//! they are in the table as `beyond` rows for that reason. An implementation that refused one of
//! them because no tier promises it answers differently at its own site, and the owner's rule is
//! the one that applies: if Metal allows it, we must too.
//!
//! The round trip is the fixture `format_scout`'s own kernels, which are our MSL: the texture is
//! loaded with `fill`'s bytes, `scout_read_write_*` reads each texel through
//! `access::read_write` and writes the `yxwz` swizzle of it back, and `scout_read_*` then reads
//! every texel into a buffer of four words a texel. So a side that cannot read, cannot write, or
//! transposes the channels of a read-write texture answers differently, and the recorded value is
//! the whole 16 by 16 grid rather than one texel.
//!
//! `fill` is bytes of `0x11 * ((i % 7) + 1)`, which is no infinity and no NaN in any format of
//! the table — checked per format class: a float32's exponent byte never reaches 0xFF, a half's
//! never 0x1F, and the 11- and 10-bit channels of `RG11B10Float` cannot reach their own. A code
//! read as a float and written back is the same code in every format here but the two sRGB ones,
//! whose encode is Metal's own rounding and whose answer this family records rather than reasons
//! about.
//!
//! # The crossing: a writable texture an argument buffer holds
//!
//! `read-write-tier-held` is where the two tiers meet, which is the interaction neither value's
//! own sweep can reach: `MTLArgumentBuffersTier2` is what promises a **writable** texture inside
//! an argument buffer, so a case that reaches a read-write texture through one asks both tiers at
//! once. The fixture is `a8r_buffers`' `a8r_held`, whose argument-buffer struct holds a
//! `texture2d<float, access::read_write>` at `[[id(0)]]` and a `texture2d<float, access::write>`
//! at `[[id(1)]]`: the kernel reads every texel of the read-write one into a buffer and writes
//! both. The case records what the read gave and what the write left in the texture, per format.
//!
//! # Why the reported tiers are a case of their own
//!
//! `device-read-write-tiers` holds the two numbers and nothing else, deliberately. They are a
//! **reading of the device** rather than a model of Metal's rule, which is the reason
//! [`crate::device_queries`] excludes them: a venue with a smaller device answers a lower tier and
//! is not diverging from Metal, and the transcript has no vocabulary for that. Keeping them in one
//! tiny case is what stops that costing the behaviour sweeps: every other case of this family
//! compares behaviour, and its tier only enters through the invariant above, which is the same
//! claim on every device — the formats *this* side's *own* tier promises and *this* side refused.
//!
//! # What this family still cannot reach
//!
//! `MTLReadWriteTextureTier1` and `MTLArgumentBuffersTier1` are values **no device of the venue
//! set answers**: this oracle reports tier 2 for both, so no run can put a lower tier to Metal and
//! the two values stay uncompared on the option-space axis. That is a property of the hardware a
//! venue has and not of the family: the formats tier 1 names are swept here as `tier1` rows
//! whatever tier the side reports, and the invariant is what carries their meaning, so a venue that
//! does report tier 1 compares the same sites with a smaller promise and needs no new case.
//!
//! # Why every case here is safe
//!
//! No descriptor of any table is one Metal's validation stops (`BLOCKED` above, and
//! `tests::no_descriptor_of_the_family_would_end_metals_process` holds every row of every table to
//! [`crate::texture_query::HeapTextureQuery::forbidden`] with no device at all). The textures are
//! 16 by 16 and 8 by 8, one level, one sample; the dispatches are one thread a texel with no loop,
//! no sampling and no atomic; and `ShaderAtomic` usage, which Metal ends its process over for any
//! format but `R32Uint`, is in no table.
//!
//! # Exactness
//!
//! Every value is a word or a byte and the family declares no tolerance beyond the one notice a
//! venue without unified memory announces for the `Shared` buffer each case reads its answer back
//! through. A format the runtime holds no read-write image of refuses its site with a typed error,
//! which the comparison reads as [`Refused`](crate::verdict::Verdict::Refused) and never as a
//! pass.

use crate::allowance::Allowance;
use crate::case::{
    made, probe, Abort, ArgumentSlot, Bindings, Case, Failure, Handle, Input, MetalSource, Side,
};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::texture::{TextureRequest, Upload};
use crate::texture_query::{HeapTextureQuery as Q, Storage};
use crate::transcript::{
    Aspect, ByteLayout, Digest, Element, Gaps, Node, Observed, Region, Sites, Value,
};
use metal2vulkan::metal::{
    PixelFormat, ReadWriteTextureTier, TextureSwizzleChannels, TextureType, TextureUsage,
};
use std::collections::BTreeSet;

/// The fixture whose read-write kernels the round trip runs: our own MSL, one kernel a component
/// class.
const SCOUT: &str = "format_scout";

/// The fixture whose kernel reaches a read-write texture through an argument buffer.
const HELD_FIXTURE: &str = "a8r_buffers";

/// The entry point of [`HELD_FIXTURE`] the crossing runs.
const HELD_ENTRY: &str = "a8r_held";

/// The grid `format_scout`'s kernels are written for: each indexes `out[g.y * 16 + g.x]`.
const SCOUT_SIDE: u32 = 16;

/// The grid `a8r_held` is written for: it indexes `out[g.y * 8 + g.x]`.
const HELD_SIDE: u32 = 8;

/// The words one texel of a kernel's read is: a `float4`, an `int4` or a `uint4`.
const WORDS_A_TEXEL: u32 = 4;

/// The usage a read-write texture is made with: both bits, which is how Metal spells one.
const READ_WRITE: TextureUsage = TextureUsage::SHADER_READ.union(TextureUsage::SHADER_WRITE);

/// The eight texture types that can carry a read-write texture argument.
///
/// Every type but the two multisample ones, which Metal's own headers refuse `access::read_write`
/// for in a `static_assert` (the module docs hold its words), so no shader can take one and no case
/// can ask the question.
const TYPES: [TextureType; 8] = [
    TextureType::Type1D,
    TextureType::Type1DArray,
    TextureType::Type2D,
    TextureType::Type2DArray,
    TextureType::Cube,
    TextureType::CubeArray,
    TextureType::Type3D,
    TextureType::TextureBuffer,
];

/// The component class a kernel reads a format's texels in, which decides which of the fixture's
/// kernels a row runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Class {
    /// `texture2d<float, …>`: every normalized and floating-point format.
    Float,
    /// `texture2d<int, …>`.
    Sint,
    /// `texture2d<uint, …>`.
    Uint,
}

impl Class {
    /// The word the case's name and its sites carry.
    const fn word(self) -> &'static str {
        match self {
            Class::Float => "float",
            Class::Sint => "sint",
            Class::Uint => "uint",
        }
    }

    /// The fixture's `access::read_write` kernel for this class.
    const fn read_write_entry(self) -> &'static str {
        match self {
            Class::Float => "scout_read_write_f",
            Class::Sint => "scout_read_write_i",
            Class::Uint => "scout_read_write_u",
        }
    }

    /// The fixture's `access::read` kernel for this class, which reads the grid into a buffer.
    const fn read_entry(self) -> &'static str {
        match self {
            Class::Float => "scout_read_f",
            Class::Sint => "scout_read_i",
            Class::Uint => "scout_read_u",
        }
    }
}

/// Which tier of `MTLReadWriteTextureTier` names a format, as Metal's tiers are documented and as
/// the runtime's own model holds them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Promise {
    /// A tier-1 format: promised by `Tier1` and by `Tier2`.
    Tier1,
    /// A format tier 2 adds: promised by `Tier2` alone.
    Tier2,
    /// A format **no** tier names, kept in the tables as a control: Metal read-writes it anyway,
    /// so a side that refused it for not being promised answers differently at its own site.
    Beyond,
}

impl Promise {
    /// Whether a side reporting `tier` has promised this format a read-write texture.
    const fn promised_by(self, tier: ReadWriteTextureTier) -> bool {
        match self {
            Promise::Tier1 => !matches!(tier, ReadWriteTextureTier::None),
            Promise::Tier2 => matches!(tier, ReadWriteTextureTier::Tier2),
            Promise::Beyond => false,
        }
    }

    /// The word a site's name carries.
    const fn word(self) -> &'static str {
        match self {
            Promise::Tier1 => "tier1",
            Promise::Tier2 => "tier2",
            Promise::Beyond => "beyond",
        }
    }
}

/// One format of the sweep: how a kernel reads it, what one texel takes, and which tier names it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Row {
    format: PixelFormat,
    class: Class,
    /// Bytes one texel takes, which is the pitch the case loads and reads at.
    texel_bytes: u32,
    promise: Promise,
}

/// The formats this family sweeps: the three `MTLReadWriteTextureTier1` names, the twelve more
/// `Tier2` adds, and twelve controls no tier names.
///
/// The tier sets are the two the runtime derives its own reported tier from
/// (`READ_WRITE_TIER1_FORMATS` and `READ_WRITE_TIER2_FORMATS` in
/// `crates/metal2vulkan/src/device/metal.rs`), so a case of this family asks exactly what the
/// implementation claims; they are written out here because no public path exports them. The controls are one of every other shape a
/// colour texel has: a reversed byte order, both sRGB encodings, 16-bit normalized, two-channel
/// 32-bit float, the two packed 10- and 11-bit formats, two-channel half and byte formats, both
/// signed-normalized widths, and the alpha-only format.
const ROWS: [Row; 30] = [
    // MTLReadWriteTextureTier1.
    row(PixelFormat::R32Float, Class::Float, 4, Promise::Tier1),
    row(PixelFormat::R32Uint, Class::Uint, 4, Promise::Tier1),
    row(PixelFormat::R32Sint, Class::Sint, 4, Promise::Tier1),
    // What MTLReadWriteTextureTier2 adds.
    row(PixelFormat::RGBA32Float, Class::Float, 16, Promise::Tier2),
    row(PixelFormat::RGBA32Uint, Class::Uint, 16, Promise::Tier2),
    row(PixelFormat::RGBA32Sint, Class::Sint, 16, Promise::Tier2),
    row(PixelFormat::RGBA16Float, Class::Float, 8, Promise::Tier2),
    row(PixelFormat::RGBA16Uint, Class::Uint, 8, Promise::Tier2),
    row(PixelFormat::RGBA16Sint, Class::Sint, 8, Promise::Tier2),
    row(PixelFormat::RGBA8Unorm, Class::Float, 4, Promise::Tier2),
    row(PixelFormat::RGBA8Uint, Class::Uint, 4, Promise::Tier2),
    row(PixelFormat::RGBA8Sint, Class::Sint, 4, Promise::Tier2),
    row(PixelFormat::R16Float, Class::Float, 2, Promise::Tier2),
    row(PixelFormat::R16Uint, Class::Uint, 2, Promise::Tier2),
    row(PixelFormat::R16Sint, Class::Sint, 2, Promise::Tier2),
    row(PixelFormat::R8Unorm, Class::Float, 1, Promise::Tier2),
    row(PixelFormat::R8Uint, Class::Uint, 1, Promise::Tier2),
    row(PixelFormat::R8Sint, Class::Sint, 1, Promise::Tier2),
    // Beyond both tiers.
    row(PixelFormat::BGRA8Unorm, Class::Float, 4, Promise::Beyond),
    row(
        PixelFormat::RGBA8UnormSRGB,
        Class::Float,
        4,
        Promise::Beyond,
    ),
    row(
        PixelFormat::BGRA8UnormSRGB,
        Class::Float,
        4,
        Promise::Beyond,
    ),
    row(PixelFormat::RGBA16Unorm, Class::Float, 8, Promise::Beyond),
    row(PixelFormat::RG32Float, Class::Float, 8, Promise::Beyond),
    row(PixelFormat::RG11B10Float, Class::Float, 4, Promise::Beyond),
    row(PixelFormat::BGR10A2Unorm, Class::Float, 4, Promise::Beyond),
    row(PixelFormat::RG16Float, Class::Float, 4, Promise::Beyond),
    row(PixelFormat::RG8Unorm, Class::Float, 2, Promise::Beyond),
    row(PixelFormat::R8Snorm, Class::Float, 1, Promise::Beyond),
    row(PixelFormat::R16Snorm, Class::Float, 2, Promise::Beyond),
    row(PixelFormat::A8Unorm, Class::Float, 1, Promise::Beyond),
];

/// A format whose descriptor Metal's validation stops in some texture types, with the types it may
/// be asked in: a compressed or a depth and stencil format, which the descriptor sweep asks for
/// read-write usage in every type that is not fatal.
///
/// The fatal cells and Metal's own words for them are in the module docs. Nothing here is in
/// `ROWS`: no kernel of this family reads a compressed or a depth texel, and a `depth2d`
/// argument is a different declaration from a `texture2d` one.
const BLOCKED: [(PixelFormat, &[TextureType]); 5] = [
    (PixelFormat::BC1RGBA, &COMPRESSED_TYPES),
    (PixelFormat::ASTC4x4SRGB, &COMPRESSED_TYPES),
    (PixelFormat::Depth32Float, &DEPTH_TYPES),
    (PixelFormat::Stencil8, &DEPTH_TYPES),
    (PixelFormat::Depth32FloatStencil8, &DEPTH_TYPES),
];

/// The types a compressed format may be asked in: Metal's own list.
const COMPRESSED_TYPES: [TextureType; 5] = [
    TextureType::Type2D,
    TextureType::Type2DArray,
    TextureType::Cube,
    TextureType::CubeArray,
    TextureType::Type3D,
];

/// The types a depth or stencil format may be asked in: Metal's own list.
const DEPTH_TYPES: [TextureType; 4] = [
    TextureType::Type2D,
    TextureType::Type2DArray,
    TextureType::Cube,
    TextureType::CubeArray,
];

/// One row of `ROWS`, as a `const` function so the table reads as a table.
const fn row(format: PixelFormat, class: Class, texel_bytes: u32, promise: Promise) -> Row {
    Row {
        format,
        class,
        texel_bytes,
        promise,
    }
}

/// The bytes a texture of this family is loaded with: `0x11 * ((i % 7) + 1)`.
///
/// Seven values, so the four channels of a texel differ in every texel size of the table and the
/// `yxwz` swizzle the read-write kernel writes is visible, and none of them puts an infinity or a
/// NaN in any format here (the module docs say why, per class).
fn fill(bytes: usize) -> Vec<u8> {
    (0..bytes)
        .map(|index| 0x11 * ((index % 7) as u8 + 1))
        .collect()
}

/// The descriptor a cell of the descriptor sweep asks about: `format` in `texture_type`, one level,
/// one sample, read-write usage, private storage.
fn queried(format: PixelFormat, texture_type: TextureType) -> Q {
    let slices = match texture_type {
        TextureType::Type1DArray | TextureType::Type2DArray | TextureType::CubeArray => 2,
        _ => 1,
    };
    Q::texture_2d(format, [4, 4], Storage::Private)
        .of_type(texture_type, 4, slices)
        .with_usage(READ_WRITE)
}

/// The site a cell of the descriptor sweep sits at: the format, the type, and which tier names the
/// format, so a reader of a diff sees the promise beside the answer.
fn cell(format: PixelFormat, texture_type: TextureType, promise: Option<Promise>) -> String {
    match promise {
        Some(promise) => format!("{}/{}/{texture_type:?}", format.name(), promise.word()),
        None => format!("{}/{texture_type:?}", format.name()),
    }
}

/// The site the two reported tiers sit at.
const READ_WRITE_SITE: &str = "readWriteTextureSupport";
const ARGUMENT_BUFFERS_SITE: &str = "argumentBuffersSupport";

/// The site the family's invariant sits at: the formats the side's **own** reported tier promises
/// that the side would not give a read-write texture.
const BROKEN_SITE: &str = "promised-and-refused";

/// Records the two tiers a side reports, and nothing else.
pub struct ReportedTiers;

impl Case for ReportedTiers {
    fn name(&self) -> &'static str {
        "device-read-write-tiers"
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        // The two Metal type names, so a case re-pointed at another query is unrecorded rather
        // than compared against these answers.
        Ok(vec![Input::bytes(
            "queries",
            format!("{READ_WRITE_SITE} {ARGUMENT_BUFFERS_SITE}").as_bytes(),
        )])
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let tier = side.read_write_texture_tier();
        probe(sites, READ_WRITE_SITE, tier, |tier| {
            Value::unit_variant(tier.name())
        });
        let tier = side.argument_buffers_tier();
        probe(sites, ARGUMENT_BUFFERS_SITE, tier, |tier| {
            Value::unit_variant(tier.name())
        });
        Ok(())
    }
}

/// Whether each format of the tables can be a read-write texture of each texture type, with the
/// family's invariant beside it.
pub struct ReadWriteDescriptors;

impl Case for ReadWriteDescriptors {
    fn name(&self) -> &'static str {
        "read-write-tier-descriptors"
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // The tables name block-compressed formats, which a device with no Vulkan format for one
        // makes through a decoded twin carried beside the blocks, and the runtime says so: a
        // performance notice at the notices site alone, every answer of the tables still
        // compared. Measured firing on llvmpipe (LLVM 21.1.8).
        vec![crate::replay::notice_allowance(
            metal2vulkan_runtime::Degradation::CarriedDecodedTwin.code(),
        )]
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let mut words = Vec::new();
        for row in ROWS {
            words.extend(row.format.as_raw().to_le_bytes());
            words.extend(row.promise.word().as_bytes());
        }
        for (format, types) in BLOCKED {
            words.extend(format.as_raw().to_le_bytes());
            for texture_type in types {
                words.extend(texture_type.as_raw().to_le_bytes());
            }
        }
        for texture_type in TYPES {
            words.extend(texture_type.as_raw().to_le_bytes());
        }
        words.extend(READ_WRITE.as_raw().to_le_bytes());
        Ok(vec![Input::bytes("descriptors", &words)])
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let tier = side.read_write_texture_tier();
        let tier = probe(sites, READ_WRITE_SITE, tier, |tier| {
            Value::unit_variant(tier.name())
        });
        let mut broken = BTreeSet::new();
        for row in ROWS {
            for texture_type in TYPES {
                let at = cell(row.format, texture_type, Some(row.promise));
                let made = side.makes_texture(&queried(row.format, texture_type));
                let answer = probe(sites, &at, made, |made| Value::Bool(*made));
                // The invariant is about the side's own promise: a format its tier names that it
                // would not make a read-write texture of, in a type Metal makes one of.
                let kept = answer == Some(true);
                if let Some(tier) = tier {
                    if row.promise.promised_by(tier) && !kept {
                        broken.insert(at);
                    }
                }
            }
        }
        for (format, types) in BLOCKED {
            for &texture_type in types {
                let at = cell(format, texture_type, None);
                let made = side.makes_texture(&queried(format, texture_type));
                probe(sites, &at, made, |made| Value::Bool(*made));
            }
        }
        sites.value(BROKEN_SITE, Value::set(broken));
        Ok(())
    }
}

/// A read-write texel round trip over every format of one component class: load the texels, swizzle
/// each one through `access::read_write`, and read the grid back.
pub struct ReadWriteRoundTrip {
    name: &'static str,
    class: Class,
}

/// One case a component class.
pub fn round_trips() -> Vec<ReadWriteRoundTrip> {
    vec![
        ReadWriteRoundTrip {
            name: "read-write-tier-round-trip-float",
            class: Class::Float,
        },
        ReadWriteRoundTrip {
            name: "read-write-tier-round-trip-sint",
            class: Class::Sint,
        },
        ReadWriteRoundTrip {
            name: "read-write-tier-round-trip-uint",
            class: Class::Uint,
        },
    ]
}

impl ReadWriteRoundTrip {
    /// The rows this case sweeps: its class's formats, in the table's order.
    fn rows(&self) -> impl Iterator<Item = Row> + use<'_> {
        ROWS.into_iter().filter(|row| row.class == self.class)
    }

    /// The fixture under the class's read-write kernel.
    fn fixture(&self) -> Fixture {
        Fixture {
            name: SCOUT,
            entry: self.class.read_write_entry(),
        }
    }

    /// The words the read-back buffer holds: four a texel of the grid.
    fn words(&self) -> u32 {
        SCOUT_SIDE * SCOUT_SIDE * WORDS_A_TEXEL
    }
}

impl Case for ReadWriteRoundTrip {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // Every row reads its answer back through a `Shared` buffer and a `Shared` texture, so on
        // a device without unified memory the runtime announces `shared-in-system-memory`
        // (`crate::replay::shared_memory_notice` holds the whole reason). It declares the notice's
        // site alone, so every value still compares against Metal's exactly.
        vec![crate::replay::shared_memory_notice()]
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let mut shaders = Vec::new();
        for extension in [RUNTIME_TEXT, METAL_TEXT] {
            shaders.push(ShaderRef {
                fixture: format!("{SCOUT}{extension}"),
                entry: self.class.read_write_entry().to_owned(),
                source_sha256: crate::sha256_hex(self.fixture().text(extension)?.as_bytes()),
            });
        }
        Ok(shaders)
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let fixture = self.fixture();
        let mut words = Vec::new();
        for row in self.rows() {
            words.extend(row.format.as_raw().to_le_bytes());
            words.extend(row.texel_bytes.to_le_bytes());
            words.extend(row.promise.word().as_bytes());
        }
        Ok(vec![
            Input {
                name: "air".to_owned(),
                digest: Digest::sha256(fixture.text(RUNTIME_TEXT)?.as_bytes()),
            },
            Input {
                name: "air-typed".to_owned(),
                digest: Digest::sha256(fixture.text(METAL_TEXT)?.as_bytes()),
            },
            Input::bytes("class", self.class.word().as_bytes()),
            Input::bytes("formats", &words),
            Input::bytes(
                "texture:fill",
                &fill((SCOUT_SIDE * SCOUT_SIDE * 16) as usize),
            ),
        ])
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let loaded = self.fixture().loaded().map_err(|why| Abort {
            at: "library".to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        })?;
        let library = made(sites, "library", side.library(&loaded.source()))?;
        let swizzle = made(
            sites,
            "pipeline/read-write",
            side.compute_pipeline(library, self.class.read_write_entry()),
        )?;
        let read = made(
            sites,
            "pipeline/read",
            side.compute_pipeline(library, self.class.read_entry()),
        )?;
        let tier = side.read_write_texture_tier();
        let tier = probe(sites, READ_WRITE_SITE, tier, |tier| {
            Value::unit_variant(tier.name())
        });
        let mut broken = BTreeSet::new();
        for row in self.rows() {
            let at = format!("{}/{}", row.format.name(), row.promise.word());
            let kept = self.sweep_one(side, sites, &at, row, swizzle, read);
            if let Some(tier) = tier {
                if row.promise.promised_by(tier) && !kept {
                    broken.insert(at);
                }
            }
        }
        sites.value(BROKEN_SITE, Value::set(broken));
        Ok(())
    }
}

impl ReadWriteRoundTrip {
    /// One format's round trip: the texture, the swizzle through `access::read_write`, the read of
    /// the whole grid, and the words it gave. Answers whether the side carried it through.
    fn sweep_one(
        &self,
        side: &mut dyn Side,
        sites: &mut Sites,
        at: &str,
        row: Row,
        swizzle: Handle,
        read: Handle,
    ) -> bool {
        let pitch = SCOUT_SIDE * row.texel_bytes;
        let bytes = fill((pitch * SCOUT_SIDE) as usize);
        let uploads = [Upload::level_0(
            [SCOUT_SIDE, SCOUT_SIDE],
            pitch,
            SCOUT_SIDE,
            &bytes,
        )];
        let request = TextureRequest {
            format: row.format.as_raw(),
            texture_type: TextureType::Type2D,
            size: [SCOUT_SIDE, SCOUT_SIDE, 1],
            levels: 1,
            slices: 1,
            usage: READ_WRITE,
            private: false,
            gpu_optimized: true,
            swizzle: TextureSwizzleChannels::DEFAULT,
            over: None,
            uploads: &uploads,
        };
        let texture = side.texture(&request);
        let Some(texture) = probe(sites, &format!("{at}/texture"), texture, |_| Value::Absent)
        else {
            return false;
        };
        let out = side.zeroed_buffer((self.words() * 4) as usize);
        let Some(out) = probe(sites, &format!("{at}/buffer"), out, |_| Value::Absent) else {
            return false;
        };
        let grid = [SCOUT_SIDE, SCOUT_SIDE, 1];
        let only = [texture];
        let swizzled = Bindings::textures(&only);
        let written = side.dispatch_threads(swizzle, &swizzled, &[], grid, grid);
        let Some(written) = probe(sites, &format!("{at}/read-write"), written, |_| {
            Value::Absent
        }) else {
            return false;
        };
        let status = side.wait(written);
        let Some(status) = probe(
            sites,
            &format!("{at}/read-write/status"),
            status,
            Clone::clone,
        ) else {
            return false;
        };
        if status != Value::unit_variant("Completed") {
            return false;
        }
        let textures = [texture];
        let buffers = [out];
        let bindings = Bindings::new(&textures, &buffers);
        let back = side.dispatch_threads(read, &bindings, &[], grid, grid);
        let Some(back) = probe(sites, &format!("{at}/read"), back, |_| Value::Absent) else {
            return false;
        };
        let status = side.wait(back);
        let Some(status) = probe(sites, &format!("{at}/read/status"), status, Clone::clone) else {
            return false;
        };
        if status != Value::unit_variant("Completed") {
            return false;
        }
        let words = side.read_buffer(out, 0, (self.words() * 4) as usize);
        let Some(words) = probe(sites, &format!("{at}/read-back"), words, |_| Value::Absent) else {
            return false;
        };
        let length = u64::from(self.words() * 4);
        sites.push(Node::leaf(
            format!("{at}/texels"),
            Observed::bytes(words, ByteLayout::buffer(0, length, Element::U32)),
        ));
        true
    }
}

/// The crossing of the two tiers: a read-write texture an argument buffer holds, read and written
/// through it.
pub struct HeldReadWrite;

impl HeldReadWrite {
    /// The fixture under the kernel whose argument buffer holds the read-write texture.
    fn fixture(&self) -> Fixture {
        Fixture {
            name: HELD_FIXTURE,
            entry: HELD_ENTRY,
        }
    }

    /// The two colours the kernel writes: `color[0]` into the read-write texture, `color[1]` into
    /// the write-only one. Zeros and ones, which every format of the sweep stores exactly, so the
    /// case's subject is the path and not a rounding.
    fn colors() -> Vec<u8> {
        let mut bytes = Vec::new();
        for value in [0.0f32, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0] {
            bytes.extend(value.to_le_bytes());
        }
        bytes
    }

    /// The rows the crossing sweeps: the float-class formats, which are what the kernel's
    /// `texture2d<float, access::read_write>` reads.
    fn rows() -> impl Iterator<Item = Row> {
        ROWS.into_iter().filter(|row| row.class == Class::Float)
    }

    /// One format: the two textures, the argument buffer that holds them, the dispatch, and both
    /// answers — what the read through the argument buffer gave, and what its write left.
    fn sweep_one(
        &self,
        side: &mut dyn Side,
        sites: &mut Sites,
        at: &str,
        row: Row,
        pipeline: Handle,
        colors: Handle,
    ) -> bool {
        let pitch = HELD_SIDE * row.texel_bytes;
        let bytes = fill((pitch * HELD_SIDE) as usize);
        let uploads = [Upload::level_0(
            [HELD_SIDE, HELD_SIDE],
            pitch,
            HELD_SIDE,
            &bytes,
        )];
        let request = TextureRequest {
            format: row.format.as_raw(),
            texture_type: TextureType::Type2D,
            size: [HELD_SIDE, HELD_SIDE, 1],
            levels: 1,
            slices: 1,
            usage: READ_WRITE,
            private: false,
            gpu_optimized: true,
            swizzle: TextureSwizzleChannels::DEFAULT,
            over: None,
            uploads: &uploads,
        };
        let read_write = side.texture(&request);
        let Some(read_write) = probe(sites, &format!("{at}/texture/0"), read_write, |_| {
            Value::Absent
        }) else {
            return false;
        };
        let write = side.texture(&request);
        let Some(write) = probe(sites, &format!("{at}/texture/1"), write, |_| Value::Absent) else {
            return false;
        };
        let held = side.argument_buffer(&[
            ArgumentSlot::Texture(read_write),
            ArgumentSlot::Texture(write),
        ]);
        let Some(held) = probe(sites, &format!("{at}/argument-buffer"), held, |_| {
            Value::Absent
        }) else {
            return false;
        };
        let words = (HELD_SIDE * HELD_SIDE * WORDS_A_TEXEL * 4) as usize;
        let out = side.zeroed_buffer(words);
        let Some(out) = probe(sites, &format!("{at}/buffer"), out, |_| Value::Absent) else {
            return false;
        };
        let grid = [HELD_SIDE, HELD_SIDE, 1];
        let buffers = [colors, out, held];
        let bindings = Bindings::buffers(&buffers);
        let dispatched = side.dispatch_threads(pipeline, &bindings, &[], grid, grid);
        let Some(dispatched) = probe(sites, &format!("{at}/dispatch"), dispatched, |_| {
            Value::Absent
        }) else {
            return false;
        };
        let status = side.wait(dispatched);
        let Some(status) = probe(sites, &format!("{at}/status"), status, Clone::clone) else {
            return false;
        };
        if status != Value::unit_variant("Completed") {
            return false;
        }
        let read = side.read_buffer(out, 0, words);
        let Some(read) = probe(sites, &format!("{at}/read-back"), read, |_| Value::Absent) else {
            return false;
        };
        sites.push(Node::leaf(
            format!("{at}/texels"),
            Observed::bytes(read, ByteLayout::buffer(0, words as u64, Element::U32)),
        ));
        let stored = side.read_texture(read_write, [HELD_SIDE, HELD_SIDE], pitch);
        let Some(stored) = probe(sites, &format!("{at}/stored"), stored, |_| Value::Absent) else {
            return false;
        };
        sites.push(Node::leaf(
            format!("{at}/stored-bytes"),
            Observed::bytes(
                stored,
                ByteLayout {
                    region: Region::Texture {
                        level: 0,
                        slice: 0,
                        aspect: Aspect::Color,
                        sample: 0,
                        origin: [0, 0, 0],
                        size: [HELD_SIDE, HELD_SIDE, 1],
                    },
                    element: Element::U8,
                    row_offset: 0,
                    row_elements: pitch,
                    row_pitch: pitch,
                    rows: HELD_SIDE,
                    image_pitch: pitch * HELD_SIDE,
                    images: 1,
                    gaps: Gaps::Ignored,
                },
            ),
        ));
        true
    }
}

impl Case for HeldReadWrite {
    fn name(&self) -> &'static str {
        "read-write-tier-held"
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // The venue's memory model for the `Shared` read-back ([`crate::replay::shared_memory_notice`]),
        // and the notice a device announces for a held texture whose format its heap array cannot
        // view (`unnamed-texture-unviewable`): the array holds the class's neutral at the
        // texture's ID. It fired on lavapipe and the RTX 5080 for the two sRGB formats and
        // `BGR10A2Unorm` until the heap viewed them through their storage aliases
        // (`bind::heap::storage_conversions`); it stays declared for a device whose alias for a
        // swept format the heap's formatless declaration cannot serve. Both sit at the notices
        // site alone: every texel is still compared. And the held textures' heap set,
        // which MoltenVK's layout query refuses ([`crate::replay::set_past_layout_notice`]):
        // announced on the M3 Max under MoltenVK 1.4.2, not on lavapipe.
        vec![
            crate::replay::shared_memory_notice(),
            crate::replay::notice_allowance(
                metal2vulkan_runtime::Degradation::UnnamedTextureUnviewable.code(),
            ),
            crate::replay::set_past_layout_notice(),
        ]
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let mut shaders = Vec::new();
        for extension in [RUNTIME_TEXT, METAL_TEXT] {
            shaders.push(ShaderRef {
                fixture: format!("{HELD_FIXTURE}{extension}"),
                entry: HELD_ENTRY.to_owned(),
                source_sha256: crate::sha256_hex(self.fixture().text(extension)?.as_bytes()),
            });
        }
        Ok(shaders)
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let fixture = self.fixture();
        let mut words = Vec::new();
        for row in Self::rows() {
            words.extend(row.format.as_raw().to_le_bytes());
            words.extend(row.texel_bytes.to_le_bytes());
            words.extend(row.promise.word().as_bytes());
        }
        Ok(vec![
            Input {
                name: "air".to_owned(),
                digest: Digest::sha256(fixture.text(RUNTIME_TEXT)?.as_bytes()),
            },
            Input {
                name: "air-typed".to_owned(),
                digest: Digest::sha256(fixture.text(METAL_TEXT)?.as_bytes()),
            },
            Input::bytes("formats", &words),
            Input::bytes("buffer:0", &Self::colors()),
            Input::bytes("texture:fill", &fill((HELD_SIDE * HELD_SIDE * 16) as usize)),
        ])
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let loaded = self.fixture().loaded().map_err(|why| Abort {
            at: "library".to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        })?;
        let library = made(sites, "library", side.library(&loaded.source()))?;
        let pipeline = made(
            sites,
            "pipeline",
            side.compute_pipeline(library, HELD_ENTRY),
        )?;
        let colors = made(sites, "buffer/0", side.buffer(&Self::colors()))?;
        let tier = side.argument_buffers_tier();
        probe(sites, ARGUMENT_BUFFERS_SITE, tier, |tier| {
            Value::unit_variant(tier.name())
        });
        let tier = side.read_write_texture_tier();
        let tier = probe(sites, READ_WRITE_SITE, tier, |tier| {
            Value::unit_variant(tier.name())
        });
        let mut broken = BTreeSet::new();
        for row in Self::rows() {
            let at = format!("{}/{}", row.format.name(), row.promise.word());
            let kept = self.sweep_one(side, sites, &at, row, pipeline, colors);
            if let Some(tier) = tier {
                if row.promise.promised_by(tier) && !kept {
                    broken.insert(at);
                }
            }
        }
        sites.value(BROKEN_SITE, Value::set(broken));
        Ok(())
    }
}

/// Every case of the family.
pub fn all() -> Vec<Box<dyn Case>> {
    let mut cases: Vec<Box<dyn Case>> =
        vec![Box::new(ReportedTiers), Box::new(ReadWriteDescriptors)];
    for case in round_trips() {
        cases.push(Box::new(case));
    }
    cases.push(Box::new(HeldReadWrite));
    cases
}

#[cfg(test)]
mod tests {
    use super::*;

    /// No descriptor of any table of this family is one Metal's own validation stops, which it
    /// answers for with no device at all: the proof that a row added by mistake cannot end Metal's
    /// process, taken over the whole family rather than over the rows a run happened to reach.
    #[test]
    fn no_descriptor_of_the_family_would_end_metals_process() {
        for row in ROWS {
            for texture_type in TYPES {
                let query = queried(row.format, texture_type);
                assert_eq!(
                    query.forbidden(),
                    None,
                    "{} {texture_type:?}",
                    row.format.name()
                );
            }
        }
        for (format, types) in BLOCKED {
            for &texture_type in types {
                let query = queried(format, texture_type);
                assert_eq!(
                    query.forbidden(),
                    None,
                    "{} {texture_type:?}",
                    format.name()
                );
            }
        }
    }

    /// No two rows name one format, and every row's texel size is the pitch its uploads state: a
    /// table whose sizes slipped would load a texture with bytes of another format's shape.
    #[test]
    fn every_row_is_its_own_format() {
        let mut seen = BTreeSet::new();
        for row in ROWS {
            assert!(seen.insert(row.format.as_raw()), "{}", row.format.name());
            assert!(row.texel_bytes.is_power_of_two(), "{}", row.format.name());
        }
        for (format, _) in BLOCKED {
            assert!(
                !seen.contains(&format.as_raw()),
                "{} is both swept and blocked",
                format.name()
            );
        }
    }

    /// The fill is no infinity and no NaN in any format of the table, which is what keeps the
    /// round trip's subject the read-write path rather than a NaN payload: checked over every
    /// 32-bit and 16-bit window the bytes make.
    #[test]
    fn the_fill_holds_no_infinity_and_no_nan() {
        let bytes = fill(4096);
        for window in bytes.windows(4) {
            let word = u32::from_le_bytes([window[0], window[1], window[2], window[3]]);
            assert_ne!(
                (word >> 23) & 0xff,
                0xff,
                "{word:#x} is an f32 infinity or NaN"
            );
            // Both halves of the word, as a half each.
            for half in [word as u16, (word >> 16) as u16] {
                assert_ne!(
                    (half >> 10) & 0x1f,
                    0x1f,
                    "{half:#x} is a half infinity or NaN"
                );
            }
            // `RG11B10Float`'s three channels: 11, 11 and 10 bits, each with five exponent bits.
            assert_ne!(
                (word >> 6) & 0x1f,
                0x1f,
                "{word:#x} is an 11-bit infinity or NaN"
            );
            assert_ne!(
                (word >> 17) & 0x1f,
                0x1f,
                "{word:#x} is an 11-bit infinity or NaN"
            );
            assert_ne!(
                (word >> 27) & 0x1f,
                0x1f,
                "{word:#x} is a 10-bit infinity or NaN"
            );
        }
    }
}
