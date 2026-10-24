//! A sparse heap (`MTLHeapTypeSparse`) and the textures made over it, as both answer for
//! themselves.
//!
//! Three Metal calls and six getters: `newHeapWithDescriptor:` of a sparse heap, the heap's `type`,
//! `size` and `usedSize`, `newTextureWithDescriptor:` on that heap, and the texture's `isSparse`,
//! `firstMipmapInTail` and `tailSizeInBytes`.
//!
//! # What the port adds to the lane
//!
//! `runtime_lane::sparse_textures` reads these getters of exactly one texture — a 256 by 256
//! `R32Float` of 9 levels — and asserts `firstMipmapInTail == 3` and `tailSizeInBytes == 16384`
//! inside a much larger test. The runtime answers both from models (`sparse::first_mipmap_in_tail`,
//! `sparse::tail_size_in_bytes`) whose comments say "measured" four times over, including the two
//! claims that decide every other answer:
//!
//! - **the chain is the full mip chain**, `floor(log2(largest axis)) + 1` levels, *whatever the
//!   texture's own level count* — so `firstMipmapInTail` may be past the texture's own levels;
//! - **a 3D texture's tail is summed per depth slice**, each rounded up to whole pages separately.
//!
//! Nothing in the tree tied either to a Metal run. One texture cannot: the first claim only shows
//! where a texture's levels are fewer than its chain, and the second only on a 3D texture. The
//! sweep here records Metal's two numbers for 123 textures — level counts from 1 to the full chain
//! and past it, non-square and odd extents, every sparse texture type, four tile widths, and the
//! multisample pair — so the models are checked against Metal rather than against their own
//! comments.
//!
//! The heap's three getters go beside them because `usedSize` is what every mapping case counts in,
//! and a fresh sparse heap's `usedSize` being 0 — with a texture made over it and nothing mapped —
//! is the baseline the rest of the lane measures against.
//!
//! **Padded formats are in.** This round found that Metal places textures of the padded 10-bit
//! extended-range formats in automatic, placement **and sparse** heaps, so `BGR10_XR` and
//! `BGRA10_XR` are swept here rather than skipped. The two-plane extended-range pair stays out: the
//! runtime refuses it typed for a sparse heap (`NotImplemented(Format)`), which is its own row to
//! settle and not this family's.
//!
//! # Why every case here is safe
//!
//! Nothing is encoded, committed or mapped: a heap is made, textures are made over it, and six
//! getters are read. A sparse heap holds no memory until a mapping takes a tile, so no GPU work and
//! no fault is possible, and a guard runs the whole table on the runtime first, where a descriptor
//! Metal's validation would stop shows up as a typed `Invalid` and a descriptor Metal answers nil
//! for as a typed `Nil`. It earned its place at once: the level-count table first held counts past
//! the mip chain, and the guard named all three as descriptors Metal's validation stops, which
//! record mode would have died on. The textures left out by construction, and why, are named in the
//! `EXCLUDED` constant's own documentation.

use crate::case::{Abort, Case, Handle, Input, MetalSource, Side};
use crate::replay::notice_allowance;
use crate::sparse_query::{SparseHeapSpec, SparseTextureSpec as T};
use crate::transcript::{Node, Observed, Sites, Value};
use metal2vulkan::metal::{PixelFormat, TextureType};
use metal2vulkan_runtime::Degradation;

/// The case names, as the report and the recordings spell them.
pub const HEAP_FACTS: &str = "sparse-heap-facts";
pub const TAILS_BY_LEVELS: &str = "sparse-tails-by-level-count";
pub const TAILS_BY_SHAPE: &str = "sparse-tails-by-shape";
pub const TAILS_BY_TYPE: &str = "sparse-tails-by-texture-type";
pub const TAILS_WIDE: &str = "sparse-tails-wide-and-short";
pub const TAILS_MULTISAMPLED: &str = "sparse-tails-multisampled";

/// The heap every case of the family makes: 1 MiB in 16-KiB pages, the lane's own.
const HEAP: SparseHeapSpec = SparseHeapSpec::new(1 << 20);

/// The padded 10-bit extended-range formats this family sweeps.
///
/// Three places in this tree claimed a padded format has no heap texture at all, and this round
/// measured that false: Metal places them in automatic, placement and sparse heaps alike. So they
/// belong in the sweep, and a release where Apple stops making one fails the case rather than
/// passing under a skip.
const PADDED_FORMATS: [PixelFormat; 2] = [PixelFormat::BGR10XR, PixelFormat::BGRA10XR];

/// What this family leaves out, and why. The guard holds the tables to it.
///
/// - **1D, 1D array and texture-buffer types.** A sparse heap makes no texture of them: Metal
///   answers nil and the runtime answers a typed `Nil`. That is a fact worth a case, but it is a
///   case about what a descriptor *makes*, which is `heap_metal_sizes`' subject, not this one's.
/// - **Level counts past a shape's own mip chain.** Metal's validation stops the descriptor.
/// - **The two-plane extended-range pair.** The runtime refuses it for a sparse heap
///   (`NotImplemented(Format)`), so a case over it would be a `Refused` row standing for an open
///   product question rather than a measurement of Metal's tail arithmetic.
/// - **Depth and stencil formats.** The runtime refuses a `Depth32Float` sparse texture that would
///   need a depth twin (`NotImplemented(SparseHeaps)`) and refuses a dual-plane format outright.
///   Their *tile* is recorded in [`crate::cases::sparse_tiles`], where no texture is made.
/// - **Sample counts other than 1, 2 and 4**, and a multisample texture of more than one level,
///   which Metal describes none of.
pub const EXCLUDED: &str = "see this module's documentation";

/// The texture types a sparse heap makes a single-sampled texture of.
const SINGLE_SAMPLED_TYPES: [TextureType; 5] = [
    TextureType::Type2D,
    TextureType::Type2DArray,
    TextureType::Cube,
    TextureType::CubeArray,
    TextureType::Type3D,
];

/// A case over one table of textures made from a sparse heap.
pub struct SparseHeapCase {
    name: &'static str,
    textures: Vec<T>,
    /// Whether the case also reads the heap's own three getters, before and after the textures are
    /// made.
    ///
    /// One case reads them, because the answer depends on the heap alone: a fresh sparse heap's
    /// `usedSize` is 0 and stays 0 however many textures are made over it, and that is one fact,
    /// not one per texture.
    asks_heap: bool,
}

/// Every case of the family.
pub fn all() -> Vec<SparseHeapCase> {
    vec![
        heap_facts(),
        tails_by_level_count(),
        tails_by_shape(),
        tails_by_texture_type(),
        tails_wide_and_short(),
        tails_multisampled(),
    ]
}

/// The heap's own three getters, read before any texture is made and again after two are, beside
/// those two textures' facts.
///
/// Two textures and not one, so "`usedSize` stays 0 however many textures are made over the heap"
/// is a fact the recording states rather than a rule a reader infers.
fn heap_facts() -> SparseHeapCase {
    SparseHeapCase {
        name: HEAP_FACTS,
        textures: vec![
            T::texture_2d(PixelFormat::R32Float, [256, 256], 9),
            T::texture_2d(PixelFormat::RGBA8Unorm, [256, 256], 1),
        ],
        asks_heap: true,
    }
}

/// Every level count from 1 to 9 of a 256 by 256 `R32Float` texture, whose full chain is 9.
///
/// This is the case that checks the chain claim: counts 1 to 8 are **short** of the chain and 9 is
/// the chain. If `firstMipmapInTail` really is computed over the full chain whatever the texture's
/// own count, Metal answers 3 for every count above 1 — which is a level *past the texture's own
/// levels* for counts 2 and 3, the shape the model's "measured" comment is about — and a count of 1
/// has no tail at all.
///
/// A count **past** the chain is not here, and the guard is why. The table held 10, 11 and 12
/// first; the runtime answered `Invalid(MipmapLevelCount { max: 9 })` for each, which is "Metal's
/// own validation stops this descriptor, and recording it would end Metal's process". So the
/// texture Metal describes none of is left to a case about what a descriptor makes, not recorded
/// here.
fn tails_by_level_count() -> SparseHeapCase {
    let textures = (1..=9)
        .map(|levels| T::texture_2d(PixelFormat::R32Float, [256, 256], levels))
        .collect();
    SparseHeapCase {
        name: TAILS_BY_LEVELS,
        textures,
        asks_heap: false,
    }
}

/// Shapes and formats over a 2D texture: four tile widths, square and non-square extents, odd
/// extents, extents at and below one tile, and the padded 10-bit pair.
///
/// The tile's width is what decides `firstMipmapInTail` — the first level at most half a tile wide
/// or high — so a format of each tile width shows the rule moving, and a non-square extent shows
/// which of the two axes Metal tests.
fn tails_by_shape() -> SparseHeapCase {
    let mut textures = Vec::new();
    let formats = [
        PixelFormat::R8Unorm,     // a 128 x 128 tile
        PixelFormat::RG8Unorm,    // 128 x 64
        PixelFormat::R32Float,    // 64 x 64
        PixelFormat::RGBA32Float, // 32 x 32
    ];
    for format in formats.into_iter().chain(PADDED_FORMATS) {
        for size in [
            [256, 256],
            [256, 64],
            [64, 256],
            [100, 37],
            [64, 64],
            [32, 32],
            [1, 1],
        ] {
            // One level, and four where the shape has four. A count past a shape's own chain is a
            // descriptor Metal's validation stops, which the guard caught in this table: a 1 by 1
            // texture holds one level, so it is asked for one.
            let largest: u32 = size.into_iter().max().unwrap_or(1).max(1);
            let chain = 1 + largest.ilog2();
            for levels in [1, 4] {
                if levels <= chain {
                    textures.push(T::texture_2d(format, size, levels));
                }
            }
        }
    }
    SparseHeapCase {
        name: TAILS_BY_SHAPE,
        textures,
        asks_heap: false,
    }
}

/// Every sparse texture type: the five single-sampled ones with slices and depths, and the
/// multisample pair at 2 and 4 samples.
///
/// The 3D rows are the ones that check the per-depth-slice tail sum, which no 2D texture can show;
/// the array and cube rows check that a slice's tail is counted once, not once per slice.
fn tails_by_texture_type() -> SparseHeapCase {
    let mut textures = Vec::new();
    for texture_type in SINGLE_SAMPLED_TYPES {
        let (depth, slices) = match texture_type {
            TextureType::Type3D => (64, 1),
            TextureType::Type2DArray => (1, 4),
            TextureType::CubeArray => (1, 2),
            _ => (1, 1),
        };
        for levels in [1, 4, 9] {
            for format in [PixelFormat::R32Float, PixelFormat::RGBA8Unorm] {
                textures.push(T::texture_2d(format, [256, 256], levels).of_type(
                    texture_type,
                    depth,
                    slices,
                ));
            }
        }
    }
    SparseHeapCase {
        name: TAILS_BY_TYPE,
        textures,
        asks_heap: false,
    }
}

/// Wide, short textures, whose recorded Metal answer is what **found two rules the model was
/// missing**.
///
/// These are the shapes where `firstMipmapInTail` is level 0 — one axis is already at most half a
/// tile — so the whole mip chain is tail, and the chain is long because the other axis is wide,
/// which is where a tail holds many levels only a few texels wide. The model summed each tail
/// level's blocks and rounded the sum up to whole pages once; Metal answered more, by a different
/// number of pages per format, on 13 of this case's rows. Measured on this M3 Max at 27.0.1:
///
/// | texture | Metal | the old model |
/// |---|---|---|
/// | `R32Float` 1000x3, 4 levels | 32768 | 16384 |
/// | `RGBA32Float` 1000x3, 1 level | 65536 | 49152 |
/// | `RGBA32Float` 1000x3, 4 levels | 98304 | 65536 |
/// | `BGR10_XR` 1000x3, 4 levels | 32768 | 16384 |
/// | `BGRA10_XR` 1000x3, 4 levels | 49152 | 32768 |
///
/// What the 13 rows established, fitted against every tail this family recorded, is that a tail
/// level's bytes are **not** its blocks' bytes: each axis is padded to a power of two blocks, and
/// a level counts at least one tile row however small it is. Both are needed — the plain product
/// is wrong on all 13 and the padded product alone on 5, each short by exactly one page — and the
/// two together reproduce every recorded row. `sparse::tail_size_in_bytes` now does that, and
/// `metal2vulkan-runtime`'s `tests::sparse::every_recorded_mip_tail_is_metals` holds it to this
/// family's recordings.
///
/// The case stays apart from the shapes the model always got right, because that is what made the
/// difference readable: a family that mixed them would have reported 13 differences among 100
/// agreements. Nothing here loosens the field — Metal is the oracle and the model is what moved.
fn tails_wide_and_short() -> SparseHeapCase {
    let mut textures = Vec::new();
    for format in [
        PixelFormat::R32Float,
        PixelFormat::RGBA32Float,
        PixelFormat::R8Unorm,
        PixelFormat::BGR10XR,
        PixelFormat::BGRA10XR,
    ] {
        for size in [[1000, 3], [1000, 1], [512, 2], [2048, 4], [3, 1000]] {
            let largest: u32 = size.into_iter().max().unwrap_or(1).max(1);
            let chain = 1 + largest.ilog2();
            for levels in [1, 4] {
                if levels <= chain {
                    textures.push(T::texture_2d(format, size, levels));
                }
            }
        }
    }
    SparseHeapCase {
        name: TAILS_WIDE,
        textures,
        asks_heap: false,
    }
}

/// The multisample pair at 2 and 4 samples, in a case of its own.
///
/// Apart from the single-sampled types because a venue's **sample-count** gap is the one gap in
/// this family a device honestly has: `sparseResidency2Samples` and `4Samples` are optional Vulkan
/// features, and llvmpipe has neither. With these rows inside the type sweep, the four textures it
/// cannot make labelled the other 176 it answered perfectly, because the suite's `LacksFeature`
/// verdict is a whole-case outcome produced only where a run **aborts**, and this case records each
/// texture's answer and goes on. Here the gap has its own row and the type sweep stays readable.
///
/// It still reads as `Refused` rather than `LacksFeature` on such a venue, and that is a product
/// bug rather than anything this case can arrange: the runtime refuses with
/// `Unsupported { feature: SparseTextures { missing: SparseResidencySamples(2) }, needs:
/// NotRepresentable }`, and `verdict::device_gap_of` counts only `Needs::AnyOf` and the two driver
/// defects as a device's gap. `NotRepresentable` means "no Vulkan 1.2+ mechanism exists for these
/// semantics", which is not what is true here: the mechanism exists and the device has not got it,
/// which is `Needs::AnyOf`.
///
/// A multisampled sparse texture is one level and a render target, which is what Metal describes
/// one of.
fn tails_multisampled() -> SparseHeapCase {
    let mut textures = Vec::new();
    for samples in [2, 4] {
        for format in [PixelFormat::RGBA8Unorm, PixelFormat::R32Float] {
            textures.push(
                T::texture_2d(format, [256, 128], 1)
                    .of_type(TextureType::Type2DMultisample, 1, 1)
                    .multisampled(samples),
            );
        }
    }
    SparseHeapCase {
        name: TAILS_MULTISAMPLED,
        textures,
        asks_heap: false,
    }
}

/// The sites a case emits.
mod site {
    /// The heap's own three getters, as the heap was made.
    pub const HEAP_FRESH: &str = "heap-fresh";
    /// The heap's three getters again, after every texture of the case was made over it.
    pub const HEAP_AFTER: &str = "heap-after-textures";
    /// A texture's three sparse getters, under the texture's own node.
    pub const TEXTURE: &str = "texture";
}

/// The degrade notices a device may announce as it makes a sparse heap or a sparse texture.
///
/// Each gets an allowance over the notices site, so a venue that announced one may hold it in its
/// set while a venue that announced none compares the set exactly against Metal's empty one. The
/// set is genuinely per-venue and measured, not guessed: MoltenVK binds no sparse memory, so it
/// makes ordinary images and announces `emulated-sparse-texture`; lavapipe makes **real** sparse
/// images but is not `residencyNonResidentStrict`, so it announces `non-strict-residency`. Several
/// allowances on one path is the shape the suite already uses for this, and a difference is
/// justified where **any** of them holds.
const SPARSE_NOTICES: [Degradation; 4] = [
    Degradation::EmulatedSparseTexture,
    Degradation::NonStrictResidency,
    Degradation::FirstSparseResourceWait,
    Degradation::HostReadMappingArguments,
];

impl Case for SparseHeapCase {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let mut words = HEAP.words();
        for texture in &self.textures {
            words.extend(texture.words());
        }
        Ok(vec![
            Input::bytes("heap-and-textures", &words),
            Input::bytes(
                "asks-heap",
                if self.asks_heap {
                    b"type,size,usedSize".as_slice()
                } else {
                    b"".as_slice()
                },
            ),
        ])
    }

    fn allowances(&self) -> Vec<crate::allowance::Allowance> {
        SPARSE_NOTICES
            .iter()
            .map(|kind| notice_allowance(kind.code()))
            .collect()
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        // The heap is the one call of the family that can stop the run: a device that cannot make
        // a sparse heap refuses it typed, and every texture of the case would be a texture of a
        // heap that does not exist. So this one aborts, and the sites after it are not emitted.
        let heap = crate::case::call(sites, "sparse_heap", side.sparse_heap(&HEAP), |_| {
            Value::Absent
        })?;
        if self.asks_heap {
            emit_heap(side, sites, heap, site::HEAP_FRESH);
        }
        for texture in &self.textures {
            // Each texture is a site of its own, and one the venue cannot make is recorded at its
            // site and the sweep goes on: the other answers stay readable instead of being lost
            // behind an abort.
            let mut node = Node::branch(texture.site(), Vec::new());
            let (answered, lacks) = match side.heap_texture(heap, texture) {
                Ok(made) => match side.sparse_texture_facts(made) {
                    Ok(facts) => (Value::ok(facts.value()), None),
                    Err(failure) => (failure.recorded(), failure.lacks),
                },
                Err(failure) => (failure.recorded(), failure.lacks),
            };
            node = node.with(Node::leaf(site::TEXTURE, Observed::value(answered)));
            sites.push(node);
            // A venue that honestly cannot make this texture -- lavapipe has no
            // `sparseResidencyNSamples`, so no multisampled sparse texture at all -- is the
            // venue's gap at this site, never a refusal of ours. The fact has to be recorded with
            // the answer, because the sweep carries on and so there is no abort to read it from.
            if let Some(needs) = lacks {
                sites.device_gap(&format!("{}/{}", texture.site(), site::TEXTURE), needs);
            }
        }
        if self.asks_heap {
            emit_heap(side, sites, heap, site::HEAP_AFTER);
        }
        Ok(())
    }
}

/// Emit the heap's three getters at `at`.
fn emit_heap(side: &mut dyn Side, sites: &mut Sites, heap: Handle, at: &str) {
    let (answered, lacks) = match side.heap_facts(heap) {
        Ok(facts) => (Value::ok(facts.value()), None),
        Err(failure) => (failure.recorded(), failure.lacks),
    };
    sites.push(Node::leaf(at, Observed::value(answered)));
    // As above: a venue that cannot answer these getters at all is its own gap here.
    if let Some(needs) = lacks {
        sites.device_gap(at, needs);
    }
}

/// The guards this family's safety argument rests on, run on a device.
///
/// Record mode hands the heap descriptor and every texture descriptor of these tables to Metal. A
/// descriptor Metal's own API validation stops **ends Metal's process**, and the runtime's model is
/// what tells the two apart: `Error::Invalid` is "Metal's validation stops it, ending the process",
/// while `Error::Nil` is "plain Metal answers nil, the process going on". So a table that makes the
/// runtime answer `Invalid` anywhere is a table that must never be recorded.
///
/// A refusal that is **ours** — the runtime has not built the form, or the device lacks sparse
/// residency — is a different matter: it is a legitimate recorded value, which is the whole point
/// of the `Refused` verdict, and it does not make the Metal leg unsafe. So the guard sorts the
/// answers by kind rather than demanding every one succeed, which is what keeps it from being
/// either vacuous or a blanket ban on recording a gap.
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod tests {
    use super::*;
    use crate::device::{self, Notices, ScratchCache, Venues};
    use crate::runtime_side::RuntimeSide;

    /// Whether a recorded failure is the runtime's `Error::Invalid`, which is the one kind that
    /// says "Metal's own validation stops this descriptor, and recording it would end Metal's
    /// process". `Failure` keeps the error as the variant path a transcript holds, not as the typed
    /// error, so the kind is read from there.
    fn is_invalid(value: &crate::transcript::Value) -> bool {
        let crate::transcript::Value::Variant { name, payload } = value else {
            return false;
        };
        if name != "Err" {
            return false;
        }
        matches!(&**payload, crate::transcript::Value::Variant { name, .. } if name == "Error::Invalid")
    }

    /// The venue the guard opens: a GPU where the tests are built for one, else a CPU ICD.
    const VENUE: Venues = if cfg!(m2v_gpu) {
        Venues::Gpu
    } else {
        Venues::Cpu
    };

    /// No descriptor of the family is one Metal's own validation stops.
    ///
    /// Verified to fail: a `T::texture_2d(PixelFormat::R32Float, [0, 256], 1)` row added to a table
    /// fails here, naming the texture and the `Invalid` the runtime answered.
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
        let heap = match side.sparse_heap(&HEAP) {
            Ok(heap) => heap,
            // A device that cannot make a sparse heap at all has nothing to say about Metal's
            // validation of the textures, and its own refusal is a recorded value.
            Err(failure) => {
                println!("this venue refuses a sparse heap: {:?}", failure.recorded());
                return;
            }
        };
        for case in all() {
            for texture in &case.textures {
                if let Err(failure) = side.heap_texture(heap, texture) {
                    // `Error::Invalid` alone: the refusals that are ours to answer for, and `Nil`,
                    // leave Metal's process alive and are recorded values.
                    if is_invalid(&failure.recorded()) {
                        stopped.push(format!(
                            "{} {}: {:?}",
                            case.name,
                            texture.site(),
                            failure.recorded()
                        ));
                    }
                }
            }
        }
        assert!(
            stopped.is_empty(),
            "these descriptors are ones Metal's validation stops, so record mode would end \
             Metal's process; take them out of the table:\n{}",
            stopped.join("\n")
        );
    }

    /// The family's tables hold no texture the module's own exclusion list names.
    ///
    /// Verified to fail: a `TextureBuffer` row, a `Depth32Float` row and a three-sample row, each
    /// added to a table, fail here naming the texture and which exclusion it breaks.
    #[test]
    fn no_texture_of_the_family_is_one_the_exclusion_list_names() {
        let mut bad = Vec::new();
        for case in all() {
            for texture in &case.textures {
                let multisample = matches!(
                    texture.texture_type,
                    TextureType::Type2DMultisample | TextureType::Type2DMultisampleArray
                );
                if !multisample && !SINGLE_SAMPLED_TYPES.contains(&texture.texture_type) {
                    bad.push(format!(
                        "{}: {} is not a type a sparse heap makes a texture of",
                        case.name,
                        texture.site()
                    ));
                }
                if multisample && (texture.levels != 1 || !texture.render_target) {
                    bad.push(format!(
                        "{}: {} is a multisampled texture of more than one level or not a target",
                        case.name,
                        texture.site()
                    ));
                }
                if !multisample && texture.samples != 1 {
                    bad.push(format!(
                        "{}: {} is a single-sampled type at more than one sample",
                        case.name,
                        texture.site()
                    ));
                }
                if multisample && ![2, 4].contains(&texture.samples) {
                    bad.push(format!(
                        "{}: {} is a sample count the oracle was not measured at",
                        case.name,
                        texture.site()
                    ));
                }
                if metal2vulkan::metal::format::is_depth_stencil(texture.format) {
                    bad.push(format!(
                        "{}: {} names a depth or stencil format, whose sparse texture the runtime \
                         refuses typed",
                        case.name,
                        texture.site()
                    ));
                }
            }
        }
        assert!(bad.is_empty(), "{}", bad.join("\n"));
    }
}
