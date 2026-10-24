//! **Crossings** of `MTLTextureDescriptor`'s fields, which is where the per-value axis ends and
//! most of what an implementation can get wrong begins.
//!
//! `docs/CONFORMANCE-COVERAGE.md` says of its own option-space axis: *"Every column here is per
//! value. A type whose values are all compared says nothing about the pairs and triples of them,
//! which is where most of the behaviour an implementation can get wrong lives."* Every case here
//! is a pair or a triple, and every one of them is chosen because **neither field alone shows it**:
//! a sweep over texture types at one level count, beside a sweep over level counts of one type,
//! passes every case in this module while answering the wrong number for a 3D texture.
//!
//! Every call is a **pure getter or a plain allocation**: `heapTextureSizeAndAlignWithDescriptor:`
//! is CPU arithmetic over a descriptor, `newTextureWithDescriptor:` allocates at most a 64 by 64
//! by 64 private texture, and nothing here encodes, dispatches, commits or reaches a queue.
//!
//! # What each case crosses, and what one axis alone cannot see
//!
//! - **`texture-pairs-type-levels-slices`** — texture type against `mipmapLevelCount` against
//!   `arrayLength`, over three element sizes. The level-size rule is **per type**, and the
//!   measurements below are each a cell no single-axis sweep reaches.
//! - **`texture-pairs-usage-crossings`** — `MTLTextureUsage` against the pixel format and against
//!   the level-0 extent, with the texture's own answered getters beside the footprint. This is
//!   where `MTLTextureUsageUnknown`, `NoConcurrentAccess` and `Video` are compared, and where the
//!   compression gate's **conjunction** is read.
//! - **`texture-pairs-storage-crossings`** — the storage mode against usage, type and levels,
//!   including `MTLStorageModeMemoryless`, which no case of this suite had ever stated.
//! - **`texture-pairs-plain-contents`** — `allowGPUOptimizedContents` against the extent, the
//!   type, the usage and the storage mode.
//! - **`texture-pairs-heap-texture-getters`** — a texture **placed in a heap**, whose
//!   `MTLResource` getters [`crate::cases::resource_facts`] reaches for a buffer, a placed buffer,
//!   a device texture and a view, and for no heap texture at all.
//!
//! ## The type and level crossings, measured
//!
//! Each of these is a number an implementation that applied one type's rule everywhere would get
//! wrong, measured on an Apple M3 Max under macOS 27.0.1, `RGBA8Unorm`, `ShaderRead`, `Private`:
//!
//! - **A 1D texture forbids mip levels and a texture buffer of the same one row takes them.** A
//!   `Type1D` descriptor of 64 by 1 with a `mipmapLevelCount` of 2 fails
//!   `-[MTLTextureDescriptorInternal validateWithDevice:]` — *"requires that mipmapLevelCount is
//!   1"* — and ends Metal's process, while a `TextureBuffer` of exactly 64 by 1 takes a
//!   `mipmapLevelCount` of 7 and answers **7** back. Two types that are each one row of texels,
//!   opposite answers. The refusal is a ledger row
//!   ([`crate::texture_query::HeapTextureQuery::forbidden`]) rather than a site, because a site of
//!   it would end the recording's process; what the case compares is the texture buffer's seven.
//! - **An array of one slice is not a texture of the same shape.** A `Type2DArray` of 64 by 64,
//!   `arrayLength` 1, one level answers 16512 bytes at an alignment of **16384**, where the
//!   `Type2D` descriptor of the same extent answers the same 16512 at **128**. At seven levels the
//!   sizes part too: 33152 against 22528. So the alignment moves at one level and the size moves
//!   at seven, and a side that read an `arrayLength` of 1 as "not an array" answers neither.
//! - **A 3D texture's depth counts toward the levels Metal allows and not toward the chain's
//!   bytes.** An 8 by 8 by 64 descriptor takes a `mipmapLevelCount` of 7 — `floor(log2(64)) + 1`,
//!   the depth's own full chain — and ends the process at 8. But its footprint **saturates at
//!   four**: levels 4, 5 and 7 each answer 65536 bytes, which is 64 chains of the 1024 bytes an 8
//!   by 8 chain takes. A 16 by 16 by 256 descriptor is the same shape of answer: it takes 9 levels
//!   and ends the process at 10, while levels 5 and 9 both answer 589824. So the chain's levels are
//!   the **width and height's** full chain and the allowed count is the **largest axis's**, and
//!   only a descriptor whose depth exceeds its sides, with more than one level, can tell.
//!   A side that fed the depth into the chain would answer every other row of this case correctly.
//! - **A cube is six chains and a cube array is six times its `arrayLength`.** 64 by 64 at seven
//!   levels: 2D 22528, an array of one slice 33152, a cube 198912 (six of the array's chain), an
//!   array of three 99456, a cube array of three 596736. Five numbers from one extent and one
//!   level count.
//! - **A multisample type takes no mip levels either**, and a multisample array of three slices
//!   answers 198144 at an alignment of 16384 against the single's 66048 at 128.
//!
//! ## The usage crossings, measured
//!
//! Metal keeps a texture losslessly compressed, which costs 8 bytes of metadata per 16 by 16
//! square and so shows in the footprint, only where **three** things hold at once: the level 0's
//! shorter side is at least 16 texels, `allowGPUOptimizedContents` is on, and the usage asks for
//! none of `ShaderWrite`, `PixelFormatView` or `ShaderAtomic`. A conjunction is the shape no
//! per-value sweep can read, and these are its cells:
//!
//! - 64 by 64, `ShaderRead` 16512 against `ShaderRead | PixelFormatView` 16384: the usage moves it;
//! - 16 by 16, 1152 against 1024: the usage still moves it at the smallest compressed side;
//! - **16 by 15, 1024 against 1024**: one texel shorter and the usage moves nothing. The cell that
//!   makes the extent part of the rule rather than a backdrop;
//! - **a 1D texture of 5000 texels, 32768 against 32768**: its height is 1, so no usage of it is
//!   ever compressed. A second shape of the same cell, reached through the type instead of the
//!   extent;
//! - `RenderTarget` alone, and `ShaderRead | RenderTarget`, both 16512: neither bit is in the
//!   conjunction, so the pair that would be a plausible guess moves nothing.
//!
//! And the three values of `MTLTextureUsage` that no recording of this suite held:
//!
//! - **`Unknown`**, which is no bits at all: the texture answers `usage` **0x17** —
//!   `ShaderRead | ShaderWrite | RenderTarget | PixelFormatView` — and its footprint is the
//!   uncompressed one, because the usage Metal resolved it to asks for `ShaderWrite`. A value that
//!   is observable only through what Metal made of it, which is why this case reads the getters
//!   and not only the footprint;
//! - **`NoConcurrentAccess`** (0x4000) and **`Video`** (0x8000): each is answered back verbatim,
//!   alone and beside `ShaderRead`, and neither changes the footprint — so neither is in the
//!   compression conjunction, and neither alone is read as `Unknown`, which a side that tested
//!   "no usage bits I know" would have got wrong;
//! - beside them the control, usage **0x8**, the one bit `MTLTextureUsage` declares no name for
//!   and Metal's texture drops: the texture answers `usage` **0** and keeps the compressed
//!   footprint. So "the texture answers 0" and "the descriptor asked for 0" are different
//!   descriptors with different footprints, which is the one site that holds the `Unknown`
//!   resolution to being a resolution.
//!
//! **Metal refuses none of the format crossings.** `ShaderWrite` on `BC1_RGBA`, on `ASTC_4x4_LDR`
//! and on `Depth32Float` each answer a texture whose `usage` is 0x3, and `RenderTarget` on the two
//! block-compressed formats answers one whose usage is 0x4. The descriptor is where a guess would
//! put the refusal and Metal does not put it there, so the case compares the acceptance.
//!
//! ## The storage crossings, measured
//!
//! - **`Shared`, `Managed` and `Private` give one footprint.** 64 by 64 at seven levels answers
//!   22528 in all three, and a three-slice array answers 99456 in all three. The storage mode and
//!   the heap footprint are **independent** among those three, and the case records the rows that
//!   say so rather than leaving it a belief.
//! - **`Memoryless` is the one mode that is not.** It answers the **uncompressed** footprint
//!   everywhere: 22144 at seven levels against the other three modes' 22528, 16384 against 16512
//!   at one. So it is a third switch into the same conjunction as `PixelFormatView` usage and
//!   `allowGPUOptimizedContents` off, and the rows read together say which of the three moved.
//! - **Plain Metal takes `Memoryless` on a descriptor of any usage and any type.** `ShaderRead`,
//!   `ShaderWrite`, `RenderTarget`, their union, `Unknown`, `PixelFormatView`,
//!   `NoConcurrentAccess` and `Video` each answer a texture whose `storageMode` is 3, in every
//!   texture type including `TextureBuffer` and a four-sample multisample one. So *"memoryless is
//!   for render targets"* is not a rule the descriptor enforces, and this family records what it
//!   does instead. What Apple's own API validation layer says of the same descriptors is the
//!   recording's second oracle and is recorded beside it.
//!
//! ## What `compressionType` would add, and why no case here reaches it
//!
//! `MTLTextureDescriptor`'s `compressionType` halves the footprint — 8320 bytes against 16512 for
//! a 64 by 64 `RGBA8Unorm` — and Metal's validation ends the process for every one of five
//! crossings of it: with `ShaderWrite` usage, with any storage mode but `Private`, with
//! `allowGPUOptimizedContents` off, with a depth format and with a block-compressed one, each
//! naming the pair in its own words. It is exactly the interaction family this module is for.
//!
//! **No case can compare it**, because our own side has no such field: the runtime's
//! `TextureDescriptor` declares no `compressionType`, so there is nothing for a case to state. The
//! gap is the runtime's, it is recorded here and in the tree's open work, and closing it is a new
//! field on a published crate rather than a case.
//!
//! # Why every case here is safe
//!
//! Every descriptor of every table is held, with no device at all, to
//! [`crate::texture_query::HeapTextureQuery::forbidden`] — the shapes
//! `-[MTLTextureDescriptorInternal validateWithDevice:]` fails its assertion for, which end Metal's
//! process rather than answering nil. `tests::no_descriptor_of_the_family_would_end_metals_process`
//! is that proof, and the Metal side runs the same guard before each call it makes, so a row added
//! by mistake is refused rather than fatal.
//!
//! The usages are what that leaves: `ShaderAtomic` is in no table, because Metal ends its process
//! for it on any format but `R32Uint` (measured again this round on `RGBA8Unorm`), and the formats
//! are the plain ones this tree already makes textures of.

use crate::case::{made, Abort, Case, Input, MetalSource, Side};
use crate::heap_query::{HeapSpec, QueriedResource};
use crate::sparse_query::SparseTextureSpec;
use crate::texture_query::{HeapTextureQuery as Q, Storage};
use crate::transcript::{Node, Observed, Sites, Value};
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};

/// The case names, as the report and the recordings spell them.
pub const TYPE_LEVELS: &str = "texture-pairs-type-levels-slices";
pub const USAGE: &str = "texture-pairs-usage-crossings";
pub const STORAGE: &str = "texture-pairs-storage-crossings";
pub const PLAIN_CONTENTS: &str = "texture-pairs-plain-contents";
pub const HEAP_TEXTURES: &str = "texture-pairs-heap-texture-getters";

/// The sites a descriptor's answers sit under.
mod site {
    /// `heapTextureSizeAndAlignWithDescriptor:`.
    pub const QUERY: &str = "size-and-align";
    /// Every `MTLTexture` getter of the texture `newTextureWithDescriptor:` answered; the absent
    /// field where plain Metal answered nil.
    pub const FACTS: &str = "texture";
}

/// What a case asks of each descriptor of its table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Asks {
    /// The heap footprint alone: the whole table is about Metal's layout arithmetic.
    Footprint,
    /// The footprint and the texture's own answered getters, which is the only way a usage Metal
    /// **resolves** is observable.
    FootprintAndGetters,
}

impl Asks {
    /// The words the case fingerprints the asking as, so a case that grows a call is a change to
    /// its inputs.
    fn words(self) -> &'static [u8] {
        match self {
            Asks::Footprint => b"heapTextureSizeAndAlign",
            Asks::FootprintAndGetters => b"heapTextureSizeAndAlign,newTextureWithDescriptor",
        }
    }
}

/// A case over one table of crossed descriptors.
pub struct DescriptorPairs {
    name: &'static str,
    queries: Vec<Q>,
    asks: Asks,
}

/// Every case of the family.
pub fn all() -> Vec<Box<dyn Case>> {
    vec![
        Box::new(type_levels_slices()),
        Box::new(usage_crossings()),
        Box::new(storage_crossings()),
        Box::new(plain_contents()),
        Box::new(HeapTextureGetters) as Box<dyn Case>,
    ]
}

/// The colour format every table's base row carries: four bytes a texel, whose tile is 64 by 64
/// elements.
const BASE: PixelFormat = PixelFormat::RGBA8Unorm;

/// The one bit `MTLTextureUsage` declares no name for and Metal's texture drops (0x8), as a usage
/// of its own.
///
/// Its own function because it is the control the `Unknown` rows are read against, and because it
/// is the one usage this family states that no named constant spells: `TextureUsage::DROPPED` is
/// the raw bits, and `try_from` is what turns them into a usage a descriptor can carry.
fn dropped_bit() -> TextureUsage {
    TextureUsage::try_from(TextureUsage::DROPPED).expect("MTLTextureUsage's own dropped bit")
}

/// Texture type against `mipmapLevelCount` against `arrayLength`, over three element sizes.
///
/// The extents are the ones that make the per-type rule visible: 64 by 64, whose full chain is
/// seven levels in every type; 37 by 5, whose odd extent rounds; 8 by 8 by 64 and 16 by 16 by 256,
/// whose depth exceeds their sides, which is the only shape where a 3D texture's chain parts from
/// its allowed level count; and one row of 64 texels, where `Type1D` and `TextureBuffer` disagree
/// about mip levels.
fn type_levels_slices() -> DescriptorPairs {
    let mut queries = Vec::new();
    // Each row is (type, [width, height, depth], levels, arrayLength). Every one of them passes
    // `forbidden`, which the module's own test proves with no device.
    const ROWS: &[(TextureType, [u32; 3], u32, u32)] = &[
        // One row of texels, two types: the one that refuses mip levels at one level, and the one
        // that takes seven.
        (TextureType::Type1D, [64, 1, 1], 1, 1),
        (TextureType::Type1D, [5000, 1, 1], 1, 1),
        (TextureType::Type1DArray, [64, 1, 1], 1, 1),
        (TextureType::Type1DArray, [64, 1, 1], 1, 3),
        (TextureType::TextureBuffer, [64, 1, 1], 1, 1),
        (TextureType::TextureBuffer, [64, 1, 1], 2, 1),
        (TextureType::TextureBuffer, [64, 1, 1], 7, 1),
        // A 2D texture and an array of one slice of the same extent, at one level and at seven.
        (TextureType::Type2D, [64, 64, 1], 1, 1),
        (TextureType::Type2D, [64, 64, 1], 2, 1),
        (TextureType::Type2D, [64, 64, 1], 6, 1),
        (TextureType::Type2D, [64, 64, 1], 7, 1),
        (TextureType::Type2DArray, [64, 64, 1], 1, 1),
        (TextureType::Type2DArray, [64, 64, 1], 7, 1),
        (TextureType::Type2DArray, [64, 64, 1], 1, 3),
        (TextureType::Type2DArray, [64, 64, 1], 7, 3),
        // An odd extent, whose levels round.
        (TextureType::Type2D, [37, 5, 1], 1, 1),
        (TextureType::Type2D, [37, 5, 1], 3, 1),
        (TextureType::Type2DArray, [37, 5, 1], 3, 3),
        // Six chains, and six times an array length.
        (TextureType::Cube, [8, 8, 1], 1, 1),
        (TextureType::Cube, [8, 8, 1], 4, 1),
        (TextureType::Cube, [64, 64, 1], 1, 1),
        (TextureType::Cube, [64, 64, 1], 7, 1),
        (TextureType::CubeArray, [64, 64, 1], 1, 1),
        (TextureType::CubeArray, [64, 64, 1], 7, 1),
        (TextureType::CubeArray, [8, 8, 1], 4, 2),
        (TextureType::CubeArray, [64, 64, 1], 1, 3),
        (TextureType::CubeArray, [64, 64, 1], 7, 3),
        // A 3D texture whose depth is at most its sides: the chain and the allowed level count
        // agree, so these are the control rows.
        (TextureType::Type3D, [8, 8, 8], 1, 1),
        (TextureType::Type3D, [8, 8, 8], 4, 1),
        (TextureType::Type3D, [64, 64, 8], 1, 1),
        (TextureType::Type3D, [64, 64, 8], 7, 1),
        (TextureType::Type3D, [64, 64, 64], 1, 1),
        (TextureType::Type3D, [64, 64, 64], 7, 1),
        // And the rows where they part. 8 by 8 by 64 takes seven levels and its chain holds four;
        // 16 by 16 by 256 takes nine and its chain holds five.
        (TextureType::Type3D, [8, 8, 64], 1, 1),
        (TextureType::Type3D, [8, 8, 64], 4, 1),
        (TextureType::Type3D, [8, 8, 64], 5, 1),
        (TextureType::Type3D, [8, 8, 64], 7, 1),
        (TextureType::Type3D, [16, 16, 256], 1, 1),
        (TextureType::Type3D, [16, 16, 256], 5, 1),
        (TextureType::Type3D, [16, 16, 256], 9, 1),
    ];
    for format in [BASE, PixelFormat::R8Unorm, PixelFormat::RGBA16Float] {
        for (texture_type, size, levels, slices) in ROWS {
            queries.push(
                Q::texture_2d(format, [size[0], size[1]], Storage::Private)
                    .of_type(*texture_type, size[2], *slices)
                    .with_levels(*levels),
            );
        }
    }
    // The two multisample types, which take one level and whose array of three is a different
    // alignment from the single's.
    for (texture_type, slices) in [
        (TextureType::Type2DMultisample, 1),
        (TextureType::Type2DMultisampleArray, 1),
        (TextureType::Type2DMultisampleArray, 3),
    ] {
        queries.push(
            Q::texture_2d(BASE, [64, 64], Storage::Private)
                .of_type(texture_type, 1, slices)
                .with_samples(4),
        );
    }
    DescriptorPairs {
        name: TYPE_LEVELS,
        queries,
        asks: Asks::Footprint,
    }
}

/// `MTLTextureUsage` against the pixel format and against the level-0 extent, with the texture's
/// own getters beside the footprint.
fn usage_crossings() -> DescriptorPairs {
    let mut queries = Vec::new();
    // The compression conjunction's cells: the usage that switches it, at an extent that takes it,
    // at the smallest that does, at one texel shorter, and in a type whose height is always 1.
    const EXTENTS: &[(TextureType, [u32; 3], u32)] = &[
        (TextureType::Type2D, [64, 64, 1], 1),
        (TextureType::Type2D, [64, 64, 1], 7),
        (TextureType::Type2D, [16, 16, 1], 1),
        (TextureType::Type2D, [16, 15, 1], 1),
        (TextureType::Type1D, [5000, 1, 1], 1),
        (TextureType::Type3D, [64, 64, 8], 1),
        (TextureType::TextureBuffer, [64, 1, 1], 1),
    ];
    // Every usage a colour texture of this family states. `ShaderAtomic` is in no row: Metal ends
    // its own process for it on any format but `R32Uint`.
    let usages = [
        TextureUsage::UNKNOWN,
        TextureUsage::SHADER_READ,
        TextureUsage::SHADER_WRITE,
        TextureUsage::SHADER_READ.union(TextureUsage::SHADER_WRITE),
        TextureUsage::RENDER_TARGET,
        TextureUsage::SHADER_READ.union(TextureUsage::RENDER_TARGET),
        TextureUsage::SHADER_READ.union(TextureUsage::PIXEL_FORMAT_VIEW),
        TextureUsage::NO_CONCURRENT_ACCESS,
        TextureUsage::SHADER_READ.union(TextureUsage::NO_CONCURRENT_ACCESS),
        TextureUsage::VIDEO,
        TextureUsage::SHADER_READ.union(TextureUsage::VIDEO),
        TextureUsage::SHADER_READ
            .union(TextureUsage::VIDEO)
            .union(TextureUsage::NO_CONCURRENT_ACCESS),
        // The control for `Unknown`: the one bit `MTLTextureUsage` declares no name for, which
        // Metal's texture drops. A descriptor of it asks for a usage and answers none, where
        // `Unknown` asks for none and answers four.
        dropped_bit(),
    ];
    for usage in usages {
        for (texture_type, size, levels) in EXTENTS {
            queries.push(
                Q::texture_2d(BASE, [size[0], size[1]], Storage::Private)
                    .of_type(*texture_type, size[2], 1)
                    .with_levels(*levels)
                    .with_usage(usage),
            );
        }
    }
    // Usage against the format: a write and a render target asked of formats a shader cannot write
    // and a pass cannot target. Metal refuses none of them at the descriptor, which is the fact
    // these rows record. `BASE` is not among them, and needs not be: the extent table above holds
    // that exact descriptor at each of these four usages, so the plain colour format's control
    // rows are already in the case and a second copy of them would be two rows at one site.
    for format in [
        PixelFormat::RGBA8UnormSRGB,
        PixelFormat::RGBA32Uint,
        PixelFormat::Depth32Float,
        PixelFormat::BC1RGBA,
        PixelFormat::ASTC4x4LDR,
        PixelFormat::RG11B10Float,
    ] {
        for usage in [
            TextureUsage::SHADER_READ,
            TextureUsage::SHADER_READ.union(TextureUsage::SHADER_WRITE),
            TextureUsage::RENDER_TARGET,
            TextureUsage::SHADER_READ.union(TextureUsage::PIXEL_FORMAT_VIEW),
        ] {
            queries.push(
                Q::texture_2d(format, [64, 64], Storage::Private)
                    .of_type(TextureType::Type2D, 1, 1)
                    .with_usage(usage),
            );
        }
    }
    // The one format `ShaderAtomic` is safe on, which is how the bit is compared at all.
    queries.push(
        Q::texture_2d(PixelFormat::R32Uint, [64, 64], Storage::Private)
            .with_usage(TextureUsage::SHADER_READ.union(TextureUsage::SHADER_ATOMIC)),
    );
    queries.push(
        Q::texture_2d(PixelFormat::R32Uint, [64, 64], Storage::Private)
            .with_usage(TextureUsage::SHADER_READ),
    );
    DescriptorPairs {
        name: USAGE,
        queries,
        asks: Asks::FootprintAndGetters,
    }
}

/// The storage mode against usage, type and levels: the three modes that answer one footprint, and
/// the fourth that does not.
fn storage_crossings() -> DescriptorPairs {
    let mut queries = Vec::new();
    const ROWS: &[(TextureType, [u32; 3], u32, u32)] = &[
        (TextureType::Type2D, [64, 64, 1], 1, 1),
        (TextureType::Type2D, [64, 64, 1], 7, 1),
        (TextureType::Type2DArray, [64, 64, 1], 7, 3),
        (TextureType::Cube, [64, 64, 1], 7, 1),
        (TextureType::Type3D, [64, 64, 8], 7, 1),
        (TextureType::Type3D, [8, 8, 64], 4, 1),
        (TextureType::TextureBuffer, [64, 1, 1], 1, 1),
    ];
    // Every mode against every row: the three that coincide are as much a measurement as the one
    // that does not, and a reader of the recording can see which.
    for storage in [
        Storage::Shared,
        Storage::Managed,
        Storage::Private,
        Storage::Memoryless,
    ] {
        for (texture_type, size, levels, slices) in ROWS {
            queries.push(
                Q::texture_2d(BASE, [size[0], size[1]], storage)
                    .of_type(*texture_type, size[2], *slices)
                    .with_levels(*levels),
            );
        }
    }
    // Memoryless against the usage, which is the question the mode's own name begs: the descriptor
    // takes every one of them. `ShaderRead` is not in this list and needs not be — it is the
    // default the mode table above states for both modes at this very descriptor, so the row is
    // already in the case and a second copy would be two rows at one site.
    for usage in [
        TextureUsage::UNKNOWN,
        TextureUsage::SHADER_WRITE,
        TextureUsage::SHADER_READ.union(TextureUsage::SHADER_WRITE),
        TextureUsage::RENDER_TARGET,
        TextureUsage::SHADER_READ.union(TextureUsage::RENDER_TARGET),
        TextureUsage::SHADER_READ.union(TextureUsage::PIXEL_FORMAT_VIEW),
        TextureUsage::NO_CONCURRENT_ACCESS,
        TextureUsage::VIDEO,
    ] {
        for storage in [Storage::Memoryless, Storage::Private] {
            queries.push(
                Q::texture_2d(BASE, [64, 64], storage)
                    .of_type(TextureType::Type2D, 1, 1)
                    .with_usage(usage),
            );
        }
    }
    // And memoryless against a depth format and a multisample type, the two shapes a memoryless
    // attachment is actually used in.
    queries.push(
        Q::texture_2d(PixelFormat::Depth32Float, [64, 64], Storage::Memoryless)
            .with_usage(TextureUsage::RENDER_TARGET),
    );
    queries.push(
        Q::texture_2d(PixelFormat::Depth32Float, [64, 64], Storage::Private)
            .with_usage(TextureUsage::RENDER_TARGET),
    );
    for storage in [Storage::Memoryless, Storage::Private] {
        queries.push(
            Q::texture_2d(BASE, [64, 64], storage)
                .of_type(TextureType::Type2DMultisample, 1, 1)
                .with_samples(4)
                .with_usage(TextureUsage::RENDER_TARGET),
        );
    }
    DescriptorPairs {
        name: STORAGE,
        queries,
        asks: Asks::FootprintAndGetters,
    }
}

/// `allowGPUOptimizedContents` against the extent, the type, the usage and the storage mode.
///
/// Every row states a **colour** format: with the property off the runtime announces
/// `optimized-depth-stencil-texture` on MoltenVK for a depth or stencil texture, and a case whose
/// subject is Metal's arithmetic should not carry a device's announcement as well.
fn plain_contents() -> DescriptorPairs {
    let mut queries = Vec::new();
    const ROWS: &[(TextureType, [u32; 3], u32, u32)] = &[
        // The extents either side of the compressed side, which is where the property's effect
        // begins and ends.
        (TextureType::Type2D, [64, 64, 1], 1, 1),
        (TextureType::Type2D, [64, 64, 1], 7, 1),
        (TextureType::Type2D, [16, 16, 1], 1, 1),
        (TextureType::Type2D, [16, 15, 1], 1, 1),
        (TextureType::Type2D, [8, 8, 1], 1, 1),
        // The types, each of which lays its chains out its own way.
        (TextureType::Type1D, [5000, 1, 1], 1, 1),
        (TextureType::Type2DArray, [64, 64, 1], 7, 3),
        (TextureType::Cube, [64, 64, 1], 7, 1),
        (TextureType::CubeArray, [64, 64, 1], 7, 3),
        (TextureType::Type3D, [64, 64, 8], 7, 1),
        (TextureType::Type3D, [8, 8, 64], 4, 1),
        (TextureType::TextureBuffer, [64, 1, 1], 1, 1),
    ];
    for gpu_optimized in [true, false] {
        for (texture_type, size, levels, slices) in ROWS {
            let query = Q::texture_2d(BASE, [size[0], size[1]], Storage::Private)
                .of_type(*texture_type, size[2], *slices)
                .with_levels(*levels);
            queries.push(match gpu_optimized {
                true => query,
                false => query.plain_contents(),
            });
        }
    }
    // The property against the usage, where both switch the same thing: the four combinations say
    // whether either alone is enough and whether both together do more. The `Private` arm of
    // `ShaderRead` is left out of this loop alone — the extent table above holds that descriptor
    // either side of the property already, so a second copy would be two rows at one site.
    for gpu_optimized in [true, false] {
        for usage in [
            TextureUsage::SHADER_READ,
            TextureUsage::SHADER_READ.union(TextureUsage::PIXEL_FORMAT_VIEW),
        ] {
            for storage in [Storage::Private, Storage::Shared, Storage::Managed] {
                if usage == TextureUsage::SHADER_READ && storage == Storage::Private {
                    continue;
                }
                let query = Q::texture_2d(BASE, [64, 64], storage).with_usage(usage);
                queries.push(match gpu_optimized {
                    true => query,
                    false => query.plain_contents(),
                });
            }
        }
    }
    // And against the one storage mode that already switches it, where neither can do it twice.
    for gpu_optimized in [true, false] {
        let query = Q::texture_2d(BASE, [64, 64], Storage::Memoryless)
            .with_usage(TextureUsage::RENDER_TARGET);
        queries.push(match gpu_optimized {
            true => query,
            false => query.plain_contents(),
        });
    }
    DescriptorPairs {
        name: PLAIN_CONTENTS,
        queries,
        asks: Asks::FootprintAndGetters,
    }
}

impl Case for DescriptorPairs {
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
        Ok(vec![
            Input::bytes("descriptors", &words),
            Input::bytes("asks", self.asks.words()),
        ])
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        for query in &self.queries {
            let mut node = Node::branch(query.site(), Vec::new());
            // Each descriptor's answer is a site of its own, and a call that did not answer is
            // recorded at its site and the sweep goes on: one descriptor a venue refuses compares
            // as a refusal at that one site, and the rest of the table stays readable.
            let answered = match side.heap_texture_size_and_align(query) {
                Ok(answered) => Value::ok(answered.value()),
                Err(failure) => failure.recorded(),
            };
            node = node.with(Node::leaf(site::QUERY, Observed::value(answered)));
            if self.asks == Asks::FootprintAndGetters {
                node = node.with(match side.queried_texture_facts(query) {
                    // The absent field for a descriptor plain Metal makes no texture of, so "no
                    // texture" and "a texture whose getters are these" are one shape apart.
                    Ok(None) => Node::leaf(site::FACTS, Observed::value(Value::Absent)),
                    Ok(Some(facts)) => facts.node(site::FACTS),
                    Err(failure) => Node::leaf(site::FACTS, Observed::value(failure.recorded())),
                });
            }
            sites.push(node);
        }
        Ok(())
    }
}

/// The heap a texture of [`HeapTextureGetters`] is placed in: 1 MiB, as
/// [`crate::cases::resource_facts`]' heaps are.
const HEAP_SIZE: u64 = 1 << 20;

/// What a **heap-placed texture** answers about itself, which no case of this suite asked.
///
/// [`crate::cases::resource_facts`] reads every `MTLResource` getter of five kinds of object — a
/// device buffer, a device texture, a view, a heap buffer and a placement-heap buffer — and of no
/// heap **texture**. So four of its answers were unmeasured for the one kind of resource whose
/// shape a descriptor decides: its `resourceOptions`, its `hazardTrackingMode`, its `heapOffset`
/// and its `isAliasable`.
///
/// What makes it a crossing rather than a sixth row is that the heap decides three of those and the
/// **descriptor** decides where the next texture lands:
///
/// - a heap-placed texture never had a descriptor stating a CPU cache mode or a hazard tracking
///   mode, so what it answers is the **heap's**, composed by Metal;
/// - `heapOffset` is where Metal's suballocator put it, which depends on the footprint of every
///   texture placed before it — so the table places a run of textures of **different shapes** in
///   one heap, in a stated order, and the offsets are a sequence rather than a value;
/// - the heap's own `usedSize` after each placement is Metal's allocator's answer to the same
///   question from the other side, and is read beside each offset.
///
/// The positive control is the sequence itself: three textures of three footprints cannot all
/// answer offset 0, so a side that answered 0 always differs at two sites, and a side whose
/// suballocator packs differently differs in the `usedSize` run as well.
pub struct HeapTextureGetters;

/// The textures placed in one heap, in order: a shape whose footprint differs from the next, so the
/// offsets Metal's suballocator answers are a sequence.
fn placed_textures() -> Vec<(&'static str, SparseTextureSpec)> {
    vec![
        ("2d-64-l1", SparseTextureSpec::texture_2d(BASE, [64, 64], 1)),
        ("2d-64-l7", SparseTextureSpec::texture_2d(BASE, [64, 64], 7)),
        (
            "2darr-64-s3-l7",
            SparseTextureSpec::texture_2d(BASE, [64, 64], 7).of_type(
                TextureType::Type2DArray,
                1,
                3,
            ),
        ),
        (
            "cube-8-l4",
            SparseTextureSpec::texture_2d(BASE, [8, 8], 4).of_type(TextureType::Cube, 1, 1),
        ),
        (
            "3d-8x8x64-l4",
            SparseTextureSpec::texture_2d(BASE, [8, 8], 4).of_type(TextureType::Type3D, 64, 1),
        ),
    ]
}

/// The heaps the case places into: the heap modes a placed texture inherits, crossed one at a time
/// so a difference names the mode that moved.
///
/// All three are **`Private` storage at Metal's own CPU cache mode**, and the hazard tracking mode
/// is the one that varies. That is a limit of the suite's vocabulary rather than a choice:
/// [`SparseTextureSpec`] is the only texture a case can place on a heap and it states `Private`
/// storage and no cache mode, while Metal requires a heap-placed resource's modes to be the
/// heap's — measured this round, a texture of the default cache mode on a write-combined heap
/// fails `-[IOGPUMetalHeap newSubResourceWithLength:alignment:options:offset:]`'s assertion,
/// *"The requested CPU cache mode (MTLCPUCacheModeDefaultCache) does not match the heap's mode
/// (MTLCPUCacheModeWriteCombined)"*, and ends Metal's process. So a shared or write-combined heap
/// is unreachable until that type states a storage and a cache mode, and a case that asked for
/// one would be fatal rather than divergent.
///
/// What is left is still the crossing the case is for: the **hazard tracking mode** is the one
/// Metal **resolves** rather than echoing, and reading it back off a texture whose descriptor
/// never stated it is Metal's composition. [`crate::cases::resource_facts`] measured that
/// resolution for a heap **buffer** — `Default` and `Untracked` both answering 1, `Tracked`
/// answering 2, against 2 for a texture outside a heap — and this is the same question asked of
/// the one resource kind that family never placed.
fn heaps() -> Vec<HeapSpec> {
    use metal2vulkan_runtime::HazardTrackingMode;
    vec![
        HeapSpec::automatic(HEAP_SIZE, Storage::Private),
        HeapSpec::automatic(HEAP_SIZE, Storage::Private).tracked(HazardTrackingMode::Tracked),
        HeapSpec::automatic(HEAP_SIZE, Storage::Private).tracked(HazardTrackingMode::Untracked),
    ]
}

impl Case for HeapTextureGetters {
    fn name(&self) -> &'static str {
        HEAP_TEXTURES
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<crate::allowance::Allowance> {
        // `heap-texture-lossless-layout` is announced as each of these textures is made: Metal
        // lays a heap texture it keeps losslessly compressed out in the heap's bytes in a way the
        // device's image does not reproduce, so the runtime places the texture's own tiles
        // otherwise and says so. Metal announces nothing, because the layout is Metal's own, so
        // the notice is a difference and the declaration is what accounts for it.
        //
        // It loosens no value. Every `MTLResource` getter of every row compares exactly with
        // Metal's, and the whole of what the declaration covers is the announcement: this case
        // reads where a texture lives and what may be done to it, never a byte of its contents,
        // so a layout of the heap's bytes is not something any site of it can observe. A case that
        // reads a heap texture's texels is a case of its own and would have to compare them.
        //
        // The notice is announced once per device per kind, so which case of a whole-set run
        // provokes it depends on what ran before; measured announced by this case on an Apple M3
        // Max under macOS 27.0.1, in the run this declaration was taken from.
        vec![crate::replay::notice_allowance(
            metal2vulkan_runtime::Degradation::HeapTextureLosslessLayout.code(),
        )]
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let mut words = Vec::new();
        for heap in heaps() {
            words.extend(heap.words());
        }
        for (_, spec) in placed_textures() {
            words.extend(spec.words());
        }
        Ok(vec![
            Input::bytes("heaps-and-textures", &words),
            Input::bytes(
                "asks",
                b"newTextureWithDescriptor:onHeap,MTLResource getters,usedSize".as_slice(),
            ),
        ])
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        for spec in heaps() {
            let at = |site: &str| format!("{}/{site}", spec.site());
            let heap = made(sites, &at("heap"), side.heap(&spec))?;
            let mut rows = Node::branch(spec.site(), Vec::new());
            for (word, texture) in placed_textures() {
                // A texture a venue cannot place is recorded at its own site and the run goes on,
                // so one refusal costs its row rather than the heap's whole sequence.
                let placed = match side.heap_texture(heap, &texture) {
                    Ok(placed) => placed,
                    Err(failure) => {
                        rows = rows.with(Node::leaf(word, Observed::value(failure.recorded())));
                        continue;
                    }
                };
                let mut row = Node::branch(word, Vec::new());
                row = row.with(
                    match side.resource_facts(QueriedResource::Texture(placed), Some(heap)) {
                        Ok(facts) => facts.node("resource"),
                        Err(failure) => Node::leaf("resource", Observed::value(failure.recorded())),
                    },
                );
                // The heap's own answer to the same question from the other side, read after each
                // placement so the run of them says how Metal's suballocator packed the textures.
                row = row.with(match side.heap_own_facts(heap) {
                    Ok(facts) => Node::leaf(
                        "heap-used-size",
                        Observed::value(Value::Uint(facts.used_size)),
                    ),
                    Err(failure) => {
                        Node::leaf("heap-used-size", Observed::value(failure.recorded()))
                    }
                });
                rows = rows.with(row);
            }
            sites.push(rows);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every descriptor of every table, with no device at all.
    fn every_query() -> Vec<(&'static str, Q)> {
        let mut out = Vec::new();
        for case in [
            type_levels_slices(),
            usage_crossings(),
            storage_crossings(),
            plain_contents(),
        ] {
            for query in case.queries {
                out.push((case.name, query));
            }
        }
        out
    }

    /// The guard this family rests on, run with no device.
    ///
    /// Record mode hands every descriptor of these tables to Metal, and a descriptor Metal's own
    /// validation stops **ends Metal's process**. So a table that trips the guard must never be
    /// recorded, and this is the check that says so before anybody runs `conformance record`.
    #[test]
    fn no_descriptor_of_the_family_would_end_metals_process() {
        let mut stopped = Vec::new();
        for (name, query) in every_query() {
            if let Some(why) = query.forbidden() {
                stopped.push(format!("{name} {}: {why}", query.site()));
            }
        }
        assert!(
            stopped.is_empty(),
            "these descriptors are ones Metal's validation stops, so record mode would end \
             Metal's process; take them out of the table:\n{}",
            stopped.join("\n")
        );
    }

    /// The guard refuses each shape it is written for, shown on a descriptor of this family's own
    /// tables rather than on a value nobody states. A guard nothing trips is a guard nobody has
    /// run, and these are the four shapes measured fatal.
    #[test]
    fn the_guard_refuses_each_shape_it_was_measured_for() {
        let one_row = Q::texture_2d(BASE, [64, 1], Storage::Private);
        // A 1D texture of more than one level, where a texture buffer of the same row takes seven.
        assert!(one_row
            .of_type(TextureType::Type1D, 1, 1)
            .with_levels(2)
            .forbidden()
            .is_some());
        assert!(one_row
            .of_type(TextureType::TextureBuffer, 1, 1)
            .with_levels(7)
            .forbidden()
            .is_none());
        // An arrayLength past 1 on a type that is not an array type.
        let flat = Q::texture_2d(BASE, [64, 64], Storage::Private);
        assert!(flat
            .of_type(TextureType::Type2D, 1, 3)
            .forbidden()
            .is_some());
        assert!(flat
            .of_type(TextureType::Type2DArray, 1, 3)
            .forbidden()
            .is_none());
        // A multisample type of more than one level.
        assert!(flat
            .of_type(TextureType::Type2DMultisample, 1, 1)
            .with_samples(4)
            .with_levels(2)
            .forbidden()
            .is_some());
        // A level count past the full chain of the largest axis, which for a 3D descriptor is its
        // depth even though its chain's bytes are its width and height's.
        assert!(flat.with_levels(7).forbidden().is_none());
        assert!(flat.with_levels(8).forbidden().is_some());
        let deep =
            Q::texture_2d(BASE, [8, 8], Storage::Private).of_type(TextureType::Type3D, 64, 1);
        assert!(deep.with_levels(7).forbidden().is_none());
        assert!(deep.with_levels(8).forbidden().is_some());
    }

    /// Every table of this family is a crossing: each of its axes takes more than one value, and
    /// the rows are distinct descriptors.
    ///
    /// A table whose every row shared one level count, or one usage, would be a per-value sweep
    /// wearing this module's name, and the whole point of the family is that it is not.
    #[test]
    fn every_table_crosses_more_than_one_value_of_each_axis_it_names() {
        for case in [
            type_levels_slices(),
            usage_crossings(),
            storage_crossings(),
            plain_contents(),
        ] {
            let count = |mut values: Vec<u64>| {
                values.sort_unstable();
                values.dedup();
                values.len()
            };
            let types = count(
                case.queries
                    .iter()
                    .map(|q| u64::from(q.texture_type.as_raw()))
                    .collect(),
            );
            let levels = count(case.queries.iter().map(|q| u64::from(q.levels)).collect());
            assert!(
                types > 1 || levels > 1,
                "{} varies neither the type nor the level count",
                case.name
            );
            let sites = count(
                case.queries
                    .iter()
                    .map(|q| {
                        use std::hash::{Hash, Hasher};
                        let mut hasher = std::collections::hash_map::DefaultHasher::new();
                        q.site().hash(&mut hasher);
                        hasher.finish()
                    })
                    .collect(),
            );
            assert_eq!(
                sites,
                case.queries.len(),
                "{} holds two descriptors at one site",
                case.name
            );
        }
    }

    /// The usage table states each of the three values of `MTLTextureUsage` no recording of this
    /// suite held, and the control beside them.
    #[test]
    fn the_usage_table_states_the_three_uncompared_values() {
        let stated: Vec<u32> = usage_crossings()
            .queries
            .iter()
            .map(|q| q.usage.as_raw())
            .collect();
        for (word, raw) in [
            ("Unknown", TextureUsage::UNKNOWN.as_raw()),
            (
                "NoConcurrentAccess",
                TextureUsage::NO_CONCURRENT_ACCESS.as_raw(),
            ),
            ("Video", TextureUsage::VIDEO.as_raw()),
            ("the dropped bit", TextureUsage::DROPPED),
        ] {
            assert!(
                stated.contains(&raw),
                "no row of {USAGE} states {word} ({raw:#x})"
            );
        }
    }

    /// The storage table states all four `MTLStorageMode` values, including the one no case of this
    /// suite had ever stated.
    #[test]
    fn the_storage_table_states_every_mode() {
        let stated: Vec<&'static str> = storage_crossings()
            .queries
            .iter()
            .map(|q| q.storage.word())
            .collect();
        for word in ["shared", "managed", "private", "memoryless"] {
            assert!(stated.contains(&word), "no row of {STORAGE} states {word}");
        }
    }

    /// Every heap this family places into is one Metal makes, and the textures it places are of
    /// more than one footprint — which is what makes the offsets a sequence rather than a value.
    #[test]
    fn the_heap_table_is_well_formed_and_its_textures_differ() {
        for spec in heaps() {
            assert!(
                spec.is_well_formed().is_none(),
                "{} is a heap Metal's validation stops",
                spec.site()
            );
            // Metal requires a heap-placed resource's storage and CPU cache modes to be the
            // heap's, and the only texture a case can place states `Private` and Metal's default
            // cache. A heap of any other pair ends Metal's process rather than refusing, measured
            // this round, so no row may state one.
            assert!(
                matches!(spec.storage, Storage::Private),
                "{} is not Private, and a placed texture of this vocabulary is",
                spec.site()
            );
            assert!(
                matches!(spec.cache, metal2vulkan_runtime::CpuCacheMode::DefaultCache),
                "{} is not the default cache mode, and a placed texture of this vocabulary is",
                spec.site()
            );
        }
        let placed = placed_textures();
        let mut words: Vec<Vec<u8>> = placed.iter().map(|(_, spec)| spec.words()).collect();
        let before = words.len();
        words.sort();
        words.dedup();
        assert_eq!(
            words.len(),
            before,
            "two placed textures are one descriptor"
        );
        assert!(before > 2, "a sequence of offsets needs more than two rows");
    }
}
