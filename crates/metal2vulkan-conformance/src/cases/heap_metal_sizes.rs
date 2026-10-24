//! `heapTextureSizeAndAlignWithDescriptor:`: Metal's own layout arithmetic, asked directly.
//!
//! This family is the suite's first **pure getter sweep**. Each case holds a table of descriptors
//! ([`crate::texture_query::HeapTextureQuery`]) and records, per descriptor, the two numbers Metal
//! answers; some also record whether `newTextureWithDescriptor:` answers a texture at all, which is
//! the fact that makes a 0-byte answer mean something.
//!
//! # Why this family came first
//!
//! `runtime_lane`'s `heap_metal_sizes` carries the numbers these cases measure — 44 bytes for a
//! 6 x 3 4:2:2 texture, 20480 for a 64 x 64 two-plane extended-range one, and some 240 more — and
//! until now nothing tied them to a Metal run: they were taken on an Apple M4 by hand, written into
//! assertions, and the probe was thrown away. The runtime answers them from a **model** of Metal's
//! arithmetic (`texture/heap_tiled.rs`) that is the same on every device at one Metal level, so the
//! numbers are a contract with Metal and with nothing else, and the only thing that can check them
//! is Metal. Record mode now does, on the oracle, as often as anybody asks.
//!
//! # Why every case here is safe
//!
//! The query compiles nothing, allocates nothing and reaches no GPU: it is CPU arithmetic over a
//! descriptor. `newTextureWithDescriptor:` is asked **only of descriptors Metal answers nil for**,
//! so it allocates nothing either.
//!
//! The one real hazard is the descriptors Metal's own validation stops, which **end Metal's
//! process** — the runtime spells that distinction out: `Error::Nil` is "plain Metal answers nil,
//! the process going on", while `Error::Invalid` is "Metal's validation stops it, ending the
//! process". Every such descriptor is left out of this family by construction, and the exclusions
//! are named where they are made:
//!
//! - `ShaderAtomic` usage of any format but `R32Uint` (a failed assertion without the debug layer
//!   too), so no case here sets that bit;
//! - a multisample descriptor of a format Metal describes no texture of;
//! - a one-dimensional, 3D or buffer descriptor of `AnyDepth` or `AnyStencil`;
//! - an extent of 0 and a side past 16384.
//!
//! The order of work is what keeps that honest rather than hopeful: a new descriptor is replayed on
//! the runtime first, where an `Invalid` answer is a recorded value and nothing else, and only a
//! table that answers no `Invalid` anywhere is ever handed to Metal.

use crate::case::{Abort, Case, Input, MetalSource, Side};
use crate::texture_query::{HeapTextureQuery as Q, Storage};
use crate::transcript::{Node, Observed, Sites, Value};
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};

/// The two storage modes every case of the family sweeps.
const BOTH_STORAGE: [Storage; 2] = [Storage::Shared, Storage::Private];

/// The case names, as the report and the recordings spell them.
pub const FOUR_TWO_TWO: &str = "heap-sizes-422";
pub const X32_STENCIL8: &str = "heap-sizes-x32-stencil8";
pub const TWO_PLANE: &str = "heap-sizes-two-plane-extended-range";
pub const TWO_PLANE_CUBES: &str = "heap-sizes-two-plane-cubes";
pub const UNSPECIALIZED: &str = "heap-sizes-unspecialized";
pub const NO_TEXTURE_FORMATS: &str = "heap-sizes-formats-without-textures";
pub const BLOCK_WRITES_ONLY: &str = "heap-sizes-block-writes-only";

/// The site a descriptor's two answers sit under.
mod site {
    /// The heap size query's answer.
    pub const QUERY: &str = "size-and-align";
    /// Whether `newTextureWithDescriptor:` answered a texture.
    pub const TEXTURE: &str = "makes-texture";
}

/// A case over one table of descriptors: the heap size query's answer for each, and for a table of
/// descriptors Metal makes no texture of, that fact too.
pub struct HeapSizeCase {
    name: &'static str,
    queries: Vec<Q>,
    /// Whether the case also asks `newTextureWithDescriptor:`.
    ///
    /// True only for a table of descriptors Metal answers **nil** for, so the call allocates
    /// nothing and cannot end Metal's process.
    asks_texture: bool,
}

/// Every case of the family.
pub fn all() -> Vec<HeapSizeCase> {
    vec![
        four_two_two(),
        x32_stencil8(),
        two_plane(),
        two_plane_cubes(),
        unspecialized(),
        formats_without_textures(),
        block_writes_only(),
    ]
}

/// Metal's answer for a 4:2:2 texture, whose rows of two-byte texels Metal pads to a multiple of
/// 16 bytes with the last row unpadded, at an alignment of 16.
///
/// The five shapes are the ones whose arithmetic the padding rule shows: a power-of-two extent, a
/// small one, two whose rows need padding, and one 1000 texels wide.
fn four_two_two() -> HeapSizeCase {
    let mut queries = Vec::new();
    for format in [PixelFormat::GBGR422, PixelFormat::BGRG422] {
        for size in [[64, 32], [8, 8], [6, 3], [10, 5], [1000, 3]] {
            for storage in BOTH_STORAGE {
                queries.push(Q::texture_2d(format, size, storage));
            }
        }
    }
    HeapSizeCase {
        name: FOUR_TWO_TWO,
        queries,
        asks_texture: false,
    }
}

/// `X32_Stencil8` beside `Stencil8` over one set of shapes: both answers are recorded, so the
/// relation the runtime lane asserts — that the two descriptors answer alike — is a fact a reader
/// of the recording can check, and a release where Apple parts them fails the case rather than
/// passing under a rule nobody re-measured.
fn x32_stencil8() -> HeapSizeCase {
    let mut queries = Vec::new();
    for format in [PixelFormat::X32Stencil8, PixelFormat::Stencil8] {
        for (texture_type, size, levels, slices) in [
            (TextureType::Type2D, [64, 32], 1, 1),
            (TextureType::Type2D, [6, 3], 1, 1),
            (TextureType::Type2D, [1000, 3], 1, 1),
            (TextureType::Type2D, [64, 64], 7, 1),
            (TextureType::Type2DArray, [64, 64], 1, 3),
            (TextureType::Cube, [32, 32], 6, 1),
        ] {
            queries.push(
                Q::texture_2d(format, size, Storage::Private)
                    .of_type(texture_type, 1, slices)
                    .with_levels(levels),
            );
        }
    }
    HeapSizeCase {
        name: X32_STENCIL8,
        queries,
        asks_texture: false,
    }
}

/// A two-plane extended-range texture in every texture type Metal makes one of, in both storage
/// modes: the family whose answers the consumer's video path depends on, and the one whose numbers
/// are least guessable (a 1 x 1 texture answers 16385 bytes).
fn two_plane() -> HeapSizeCase {
    let mut queries = Vec::new();
    let usage = TextureUsage::SHADER_READ.union(TextureUsage::SHADER_WRITE);
    for format in [PixelFormat::RGB10A82PXR10, PixelFormat::RGB10A82PXR10SRGB] {
        for (texture_type, size, levels, slices) in [
            (TextureType::Type2D, [64, 64, 1], 1, 1),
            (TextureType::Type2D, [1, 1, 1], 1, 1),
            (TextureType::Type2D, [7, 3, 1], 1, 1),
            (TextureType::Type2D, [1000, 3, 1], 1, 1),
            (TextureType::Type2D, [64, 65, 1], 1, 1),
            (TextureType::Type2D, [7, 600, 1], 1, 1),
            (TextureType::Type2D, [1920, 1080, 1], 1, 1),
            (TextureType::Type2D, [64, 64, 1], 7, 1),
            (TextureType::Type2D, [17, 9, 1], 5, 1),
            (TextureType::Type2DArray, [64, 64, 1], 1, 3),
            (TextureType::Type2DArray, [7, 3, 1], 1, 5),
            (TextureType::Type2DArray, [64, 65, 1], 1, 2),
            (TextureType::Type2DArray, [17, 9, 1], 3, 2),
            (TextureType::Type1D, [64, 1, 1], 1, 1),
            (TextureType::Type1D, [5000, 1, 1], 1, 1),
            (TextureType::Type1DArray, [7, 1, 1], 1, 5),
            (TextureType::Type1DArray, [5000, 1, 1], 1, 4),
            (TextureType::Type3D, [64, 64, 4], 1, 1),
            (TextureType::Type3D, [32, 32, 20], 1, 1),
            (TextureType::Type3D, [64, 64, 4], 3, 1),
        ] {
            for storage in BOTH_STORAGE {
                queries.push(
                    Q::texture_2d(format, [size[0], size[1]], storage)
                        .of_type(texture_type, size[2], slices)
                        .with_levels(levels)
                        .with_usage(usage),
                );
            }
        }
    }
    HeapSizeCase {
        name: TWO_PLANE,
        queries,
        asks_texture: false,
    }
}

/// A two-plane extended-range cube and cube array: 0 bytes at an alignment of 16, and no texture.
///
/// Its own case rather than a row of [`two_plane`], because it is the one shape of the family where
/// the 0-byte answer only means something beside the second fact: a descriptor Metal describes no
/// texture of.
fn two_plane_cubes() -> HeapSizeCase {
    let mut queries = Vec::new();
    for format in [PixelFormat::RGB10A82PXR10, PixelFormat::RGB10A82PXR10SRGB] {
        for (texture_type, slices) in [(TextureType::Cube, 1), (TextureType::CubeArray, 2)] {
            queries.push(Q::texture_2d(format, [32, 32], Storage::Private).of_type(
                texture_type,
                1,
                slices,
            ));
        }
    }
    HeapSizeCase {
        name: TWO_PLANE_CUBES,
        queries,
        asks_texture: true,
    }
}

/// The eight texture types a descriptor of a format with no texture is asked in, with the depth and
/// slice count each type takes. The two multisample types are left out of every such case: Metal's
/// validation stops a multisample descriptor of a format it describes no texture of, which ends its
/// process.
const STACKABLE_TYPES: [(TextureType, u32, u32); 8] = [
    (TextureType::Type1D, 1, 1),
    (TextureType::Type1DArray, 1, 2),
    (TextureType::Type2D, 1, 1),
    (TextureType::Type2DArray, 1, 2),
    (TextureType::Cube, 1, 1),
    (TextureType::CubeArray, 1, 2),
    (TextureType::Type3D, 2, 1),
    (TextureType::TextureBuffer, 1, 1),
];

/// `Unspecialized`, Apple's private format: 0 bytes at the alignment of its texture type, and no
/// texture from any creator.
///
/// The alignment is the subject. It is a Metal-level fact — 16 KiB for the stacked types under
/// macOS 26.7, 128 under 27.0.1 — which is exactly the kind of answer a recording keyed by Metal
/// level exists to hold, and exactly the kind a comment cannot.
///
/// `ShaderAtomic` is not among the usages, and must not be: Metal ends its own process on it.
fn unspecialized() -> HeapSizeCase {
    let mut queries = Vec::new();
    for storage in BOTH_STORAGE {
        for (texture_type, depth, slices) in STACKABLE_TYPES {
            for usage in [
                TextureUsage::SHADER_READ,
                TextureUsage::SHADER_READ.union(TextureUsage::SHADER_WRITE),
                TextureUsage::RENDER_TARGET,
            ] {
                queries.push(
                    Q::texture_2d(PixelFormat::Unspecialized, [16, 16], storage)
                        .of_type(texture_type, depth, slices)
                        .with_usage(usage),
                );
            }
        }
    }
    HeapSizeCase {
        name: UNSPECIALIZED,
        queries,
        asks_texture: true,
    }
}

/// The formats Metal describes no texture of alone: `RGBA16Float_XR`, `AnyDepth`, `AnyStencil` and
/// a two-plane YCbCr format. Each answers 0 bytes at its texture type's alignment, and no texture.
///
/// `AnyDepth` and `AnyStencil` are asked in the 2D and cube types alone: Metal's validation stops a
/// one-dimensional, 3D or buffer descriptor of either, which ends its process.
fn formats_without_textures() -> HeapSizeCase {
    let mut queries = Vec::new();
    for format in [
        PixelFormat::RGBA16FloatXR,
        PixelFormat::AnyDepth,
        PixelFormat::AnyStencil,
        PixelFormat::YCBCR84202P,
    ] {
        let any_depth = matches!(format, PixelFormat::AnyDepth | PixelFormat::AnyStencil);
        for (texture_type, depth, slices) in STACKABLE_TYPES {
            let one_dimensional =
                matches!(texture_type, TextureType::Type1D | TextureType::Type1DArray);
            let volume_or_buffer = matches!(
                texture_type,
                TextureType::Type3D | TextureType::TextureBuffer
            );
            if any_depth && (one_dimensional || volume_or_buffer) {
                continue;
            }
            for storage in BOTH_STORAGE {
                queries.push(Q::texture_2d(format, [64, 64], storage).of_type(
                    texture_type,
                    depth,
                    slices,
                ));
            }
        }
    }
    HeapSizeCase {
        name: NO_TEXTURE_FORMATS,
        queries,
        asks_texture: true,
    }
}

/// Apple's private `BlockWritesOnly` usage bit (0x10000) beside the same descriptor without it:
/// both answers are recorded, so the relation the runtime lane asserts — that the bit changes no
/// heap answer — is a fact a reader can check rather than a rule.
fn block_writes_only() -> HeapSizeCase {
    let mut queries = Vec::new();
    for (format, texture_type, size, levels, slices) in [
        (PixelFormat::BGRA8Unorm, TextureType::Type2D, [16, 16], 1, 1),
        (
            PixelFormat::RGBA8Unorm,
            TextureType::Type2D,
            [100, 37],
            1,
            1,
        ),
        (
            PixelFormat::RGBA16Float,
            TextureType::Type2D,
            [256, 256],
            9,
            1,
        ),
        (
            PixelFormat::BGRA8Unorm,
            TextureType::Type2DArray,
            [64, 64],
            1,
            4,
        ),
    ] {
        for storage in BOTH_STORAGE {
            for base in [
                TextureUsage::SHADER_READ,
                TextureUsage::SHADER_READ.union(TextureUsage::RENDER_TARGET),
            ] {
                for usage in [base, base.union(TextureUsage::BLOCK_WRITES_ONLY)] {
                    queries.push(
                        Q::texture_2d(format, size, storage)
                            .of_type(texture_type, 1, slices)
                            .with_levels(levels)
                            .with_usage(usage),
                    );
                }
            }
        }
    }
    HeapSizeCase {
        name: BLOCK_WRITES_ONLY,
        queries,
        asks_texture: false,
    }
}

/// The rows a driver that is not Metal answers a **larger** heap texture size or alignment for,
/// which the runtime announces as `heap-texture-larger-than-metal`.
///
/// Measured on the RTX 5080 Laptop (615.71.09) and on llvmpipe (Mesa 26.2.4):
/// `heap-sizes-x32-stencil8` announced it on both and `heap-sizes-block-writes-only` on the 5080,
/// each with the case declaring it nowhere; the other five rows were exact on both
/// (replay-nvidia.log, replay-llvmpipe.log).
///
/// The declaration is at the notices site alone, which matters here more than elsewhere: both rows
/// also answer **sizes and alignments** that are not Metal's -- `X32_Stencil8` aligns at 1024
/// where Metal aligns at 128 -- and those values stay compared exactly, so the rows stay divergent
/// on them. That difference is the finding; the undeclared notice was only hiding it.
const LARGER_THAN_METALS: [&str; 2] = [X32_STENCIL8, BLOCK_WRITES_ONLY];

impl Case for HeapSizeCase {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<crate::allowance::Allowance> {
        // The family reads no buffer back -- it asks the device for a heap texture's size and
        // alignment -- so the notice a card without unified memory announces for a `Shared` buffer
        // cannot reach it, and no row declares one. Measured: five of the seven rows were exact on
        // both the 5080 and llvmpipe.
        match LARGER_THAN_METALS.contains(&self.name) {
            true => vec![crate::replay::notice_allowance(
                metal2vulkan_runtime::Degradation::HeapTextureLargerThanMetal.code(),
            )],
            false => Vec::new(),
        }
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
            Input::bytes(
                "asks",
                if self.asks_texture {
                    b"heapTextureSizeAndAlign,newTextureWithDescriptor".as_slice()
                } else {
                    b"heapTextureSizeAndAlign".as_slice()
                },
            ),
        ])
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        for query in &self.queries {
            let mut node = Node::branch(query.site(), Vec::new());
            // Each descriptor's answer is a site of its own, and a call that did not answer is
            // recorded at its site and the sweep goes on: one descriptor the runtime refuses
            // compares as a refusal at that one site, and the other 200 answers stay readable
            // instead of being lost behind an abort.
            let answered = match side.heap_texture_size_and_align(query) {
                Ok(answered) => Value::ok(answered.value()),
                Err(failure) => failure.recorded(),
            };
            node = node.with(Node::leaf(site::QUERY, Observed::value(answered)));
            if self.asks_texture {
                let made = match side.makes_texture(query) {
                    Ok(made) => Value::ok(Value::Bool(made)),
                    Err(failure) => failure.recorded(),
                };
                node = node.with(Node::leaf(site::TEXTURE, Observed::value(made)));
            }
            sites.push(node);
        }
        Ok(())
    }
}

/// The safety guard this family rests on, run on a device.
///
/// Record mode hands every descriptor of these tables to Metal. A descriptor Metal's own validation
/// stops **ends Metal's process**, and the runtime's model is what tells the two apart:
/// `Error::Invalid` is "Metal's validation stops it, ending the process", while `Error::Nil` is
/// "plain Metal answers nil, the process going on". So a table that makes the runtime answer
/// `Invalid` anywhere is a table that must never be recorded, and this is the check that says so
/// before anybody runs `conformance record`.
///
/// It is also the check that would catch a table growing a row somebody assumed was harmless. Its
/// failure mode was exercised while writing it: a 2D descriptor of width 0 added to one table fails
/// here, naming the descriptor and the problem.
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod tests {
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
                let answer = side.heap_texture_size_and_align(query);
                if let Err(failure) = &answer {
                    stopped.push(format!(
                        "{} {}: {:?}",
                        case.name,
                        query.site(),
                        failure.recorded()
                    ));
                }
                if case.asks_texture {
                    if let Err(failure) = side.makes_texture(query) {
                        stopped.push(format!(
                            "{} {} makes_texture: {:?}",
                            case.name,
                            query.site(),
                            failure.recorded()
                        ));
                    }
                }
            }
        }
        assert!(
            stopped.is_empty(),
            "these descriptors do not answer on the runtime, so Metal's validation may stop them \
             and record mode would end Metal's process; take them out of the table:\n{}",
            stopped.join("\n")
        );
    }
}
