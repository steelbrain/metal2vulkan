//! `MTLResourceStateCommandEncoder`, swept: the only implemented container the suite reached **no
//! item of at all**.
//!
//! Seven implemented items, none exercised before this family, so nothing in the tree held Metal
//! to its sparse mapping behaviour. Six of the seven are reachable — the seventh is the protocol
//! declaration, which no call site can name — and all six are here:
//! `updateTextureMapping:mode:region:mipLevel:slice:`,
//! `updateTextureMappings:mode:regions:mipLevels:slices:numRegions:`,
//! `updateTextureMapping:mode:indirectBuffer:indirectBufferOffset:`,
//! `moveTextureMappingsFromTexture:…`, `updateFence:` and `waitForFence:`.
//!
//! # Depth, not breadth
//!
//! Touching each item once would say almost nothing: what an update *means* is its mode, its
//! region, its level and its slice, and those are the space an implementation gets wrong. So the
//! family sweeps that space rather than adding six one-call cases:
//!
//! | parameter | values compared | of | left out, and why |
//! |---|---|---|---|
//! | `MTLSparseTextureMappingMode` | `Map`, `Unmap` | 2 | — |
//! | region size, in tiles | 0, 1, a row, a level's whole grid, the last tile | — | nothing: the bounds are 0 and the level's own grid, and both are rows |
//! | region origin | 0, the last tile of the level | — | — |
//! | mip level | 0, 1, 2, `firstMipmapInTail`, past the texture's levels | 5 of 9 | levels 4 to 8, which are past `firstMipmapInTail` and whose update Metal maps as the tail's |
//! | slice | 0 and 1 of a two-slice array, and one past the texture's two | 3 | — |
//! | update selector | singular, plural, indirect | 3 | — |
//! | indirect argument count | 0, 1, several | — | a count past the buffer, which the runtime refuses typed and whose Metal behaviour is a refusal case of its own |
//! | indirect offset | 0, a nonzero multiple of 4 | — | an offset that is not a multiple of 4, which the runtime documents as read at the multiple below and which belongs to a refusal case |
//! | move overlap | onto an unmapped tile, onto a mapped one, within a texture, between two | — | — |
//! | move past the bounds | a source clipped on one axis, a source clipped on two, a destination's row overflowing, and both sides inside | — | a destination run past the level's **last row**, and a region deeper than one image: neither is measured, so no row asserts either |
//!
//! # Every swept value changes what is read, and here is which
//!
//! A sweep makes vacuity cheap to hide, so the measurement is chosen to move with each value: the
//! heap's **`usedSize`**, read once before anything is mapped and again after every commit. A
//! sparse heap holds no memory until a mapping takes a tile, so
//!
//! - **the mode** moves it: a map raises it by the tiles it took, an unmap lowers it;
//! - **the region's size** moves it: a region of 0 tiles leaves it where it was, one tile raises
//!   it by one page, a level's whole grid by that level's tiles;
//! - **the level** moves it: level 0 of this texture is 16 tiles, level 1 is 4, level 2 is 1, and
//!   the tail is one page however small a region names it;
//! - **the slice** moves it: a second slice's tiles are tiles of their own, and a slice **past**
//!   the texture's own moves it, plain Metal's and ours alike, by a page for one
//!   ([`a_slice_past_the_textures`], which also reads the texels four mapped tiles hold either
//!   side of such an update, because `usedSize` cannot say which tiles one reached);
//! - **the move** does not move it, which is the point of its row: the runtime's own
//!   documentation says a move leaves `usedSize` alone, and nothing had measured that.
//!
//! Two parameters are **indistinguishable by construction** in `usedSize` and are named rather
//! than left implied. The three update selectors map the same tiles, so the plural and indirect
//! rows are read against the singular row's own numbers — they hold Metal to its three spellings
//! of one update, and a difference between them is a bug of ours. And the two fences change no
//! byte at all here: Metal orders a command buffer's encoders anyway, so
//! [`mapping_fences`] states plainly that it pins the pair encoding **cleanly** and reaching the
//! items, not that it proves an ordering.
//!
//! Beside `usedSize`, one row reads **texels**: [`mapped_tiles_hold_their_texels`] writes known
//! bytes into tiles it has mapped and reads them back, which is what says a mapping backed real
//! memory rather than only moving a counter.
//!
//! # Why nothing here can fault
//!
//! Sparse mapping is the shape in this round's territory where an out-of-range region could fault
//! a device, so the bound is held three ways, none of them care:
//!
//! - every region is stated in **tiles**, bounded by the texture's own grid
//!   ([`crate::sparse_mapping::TileGrid`]);
//! - `every_region_lies_inside_its_level` checks every row's every region against that grid with
//!   no device at all, so a row that grew past it fails in the test suite;
//! - the side **reads the venue's own tile size first** and aborts typed where it is not the grid
//!   the rows were written for, so a venue whose tiles are another size records that refusal
//!   rather than being handed a region past its grid.
//!
//! A **move** is the one step whose region may run past that grid, because that is the question
//! the four move rows are about and plain Metal's answer to one is measured rather than guessed:
//! eight shapes on an M3 Max, every command buffer completed with no error and no heap's
//! `usedSize` moved (`probes/sparse_move_past_the_bounds.m`). The relief is the step's kind and
//! not a row's name, and it is bounded by what was measured — each side's origin inside its
//! level's own tiles, no axis stating more tiles than the level holds, one image deep — so a
//! region of 0xffff_ffff tiles is still a test failure.
//!
//! A fill and a read name tiles the program mapped and nothing else
//! (`every_fill_and_read_lies_in_mapped_tiles`), which for the move rows means the tiles the move
//! itself mapped: the guard applies Metal's own measured rule for which destination tiles a move
//! takes, so a read of one it did not take is caught as loudly as a read of a tile no step ever
//! mapped.
//!
//! Nothing dispatches, nothing draws, no shader runs, every wait is bounded, and the heap is 4
//! MiB of which the rows map at most a few dozen pages.

use crate::allowance::Allowance;
use crate::case::{Abort, Case, Input, MetalSource, Side};
use crate::replay::notice_allowance;
use crate::sparse_mapping::{MappingMode, MappingStep, SparseMappingProgram, TileGrid, TileRect};
use crate::sparse_query::{SparseHeapSpec, SparseTextureSpec};
use crate::transcript::{ByteLayout, Element, Node, Observed, Sites};
use metal2vulkan::metal::{PixelFormat, TextureType};
use metal2vulkan_runtime::Degradation;

/// The sparse heap every case of the family makes: 4 MiB in 16-KiB pages, which is 256 pages — far
/// more than the few dozen tiles any row maps.
const HEAP: SparseHeapSpec = SparseHeapSpec::new(4 << 20);

/// The texture extent every row uses, and the levels it declares.
///
/// 256 by 256 `R32Float` is the shape `runtime_lane::sparse_textures` measured on this oracle:
/// a 16-KiB page makes its tile 64 by 64 texels, so level 0 holds 4 by 4 tiles, level 1 holds
/// 2 by 2, level 2 holds exactly one, and `firstMipmapInTail` is 3. Nine levels is the full mip
/// chain, so the rows can name a level past the tail as well as inside it.
const WIDTH: u32 = 256;
const LEVELS: u32 = 9;
const FORMAT: PixelFormat = PixelFormat::R32Float;

/// The tile grid the rows' regions are stated in, and the bytes a texel takes.
///
/// The side checks it against the venue's own
/// `sparseTileSizeWithTextureType:pixelFormat:sampleCount:sparsePageSize:` before it encodes
/// anything, so a venue whose tiles are another size records a typed refusal rather than being
/// handed a region past its grid.
const TILE: TileGrid = TileGrid::new(64, 64, 4);

/// The level `firstMipmapInTail` is expected to be for [`WIDTH`] at [`TILE`]: level 3, whose 32 by
/// 32 texels are smaller than one tile.
///
/// It is not read from the device: the rows that name the tail name **this** level, and the
/// recording holds the device's own `firstMipmapInTail` beside the answer, so a venue where the
/// two differ shows it in the transcript rather than in a region nobody bounded.
const TAIL_LEVEL: u64 = 3;

/// A level past the texture's own levels: an update of it maps nothing, which is the negative fact
/// its row records.
const PAST_THE_LEVELS: u64 = LEVELS as u64;

/// The slices the array rows' texture holds.
const SLICES: u32 = 2;

/// A slice past the texture's own: the shape [`A_SLICE_PAST_THE_TEXTURES`] states.
const PAST_THE_SLICES: u64 = SLICES as u64;

/// The byte [`mapped_tiles_hold_their_texels`] fills its mapped tiles with. Not zero: a tile that
/// read zero would be indistinguishable from one the copy never reached.
const FILL: u8 = 0xc3;

/// The four bytes [`A_SLICE_PAST_THE_TEXTURES`] fills its four mapped tiles with, in the order
/// (slice 0, tile 0), (slice 0, tile 1), (slice 1, tile 0), (slice 1, tile 1).
///
/// **Four distinct values and none of them zero**, which is what makes the row's reading able to
/// separate the rules rather than only to pass. A side that mapped one slice for both, or one tile
/// for both, reads another row's byte where its own belongs; and a tile an update unmapped reads
/// zero through a blit on this oracle (`probes/sparse_unmapped_blit.m`), which no fill here can,
/// so "this tile lost its mapping" and "this tile is as it was" are two readings and not one.
const SLICE_FILLS: [u8; 4] = [0xc3, 0x3c, 0xa5, 0x5a];

/// The notices **every** row of the family takes.
///
/// A sparse texture is emulated wherever the device offers no sparse residency of its own — which
/// on this oracle's MoltenVK is every sparse texture — and the runtime announces that once per
/// texture. It is a device gap, announced honestly, so each row declares it and then compares
/// every number exactly; a venue with native sparse residency announces nothing and the
/// declaration simply does not fire there.
///
/// A device that *has* sparse residency announces the other of the pair instead:
/// `non-strict-residency` says the device does not guarantee that a read of an unmapped tile
/// answers zero, where Metal's sparse texture does. Both are the device's own and neither loosens
/// a number: every row still compares each texel and each count with Metal's exactly, so a texel
/// a non-strict read moved would be this family's finding and not its tolerance.
///
/// Measured on llvmpipe (LLVM 23.1.1, 256 bits, Mesa 26.2.4) in a whole-suite replay: all nine
/// rows announce `non-strict-residency`, and eight of them announce no
/// `emulated-sparse-texture` at all, which is the declaration that leaves the other unfired.
const EVERY_ROWS_NOTICES: [Degradation; 2] = [
    Degradation::EmulatedSparseTexture,
    Degradation::NonStrictResidency,
];

/// A single-sampled 2D sparse texture of [`WIDTH`] by [`WIDTH`].
fn plain() -> SparseTextureSpec {
    SparseTextureSpec::texture_2d(FORMAT, [WIDTH, WIDTH], LEVELS)
}

/// The same texture as a [`SLICES`]-slice array, for the rows that sweep the slice.
fn array() -> SparseTextureSpec {
    plain().of_type(TextureType::Type2DArray, 1, SLICES)
}

/// The tiles level `level` of one of these textures holds, on each axis.
fn tiles_at(level: u32) -> [u64; 2] {
    TILE.tiles_at(WIDTH as u64, WIDTH as u64, level)
}

/// A region covering every tile of level `level`.
fn whole_level(level: u32) -> TileRect {
    TileRect::new([0, 0], tiles_at(level))
}

/// The one tile at the far corner of level `level`: the boundary either side of which an
/// implementation's rounding shows.
fn last_tile(level: u32) -> TileRect {
    let [across, down] = tiles_at(level);
    TileRect::one([across - 1, down - 1])
}

/// Whether a program reaches a `Shared` buffer of its own: it reads tiles out to one
/// ([`MappingStep::ReadRegion`], which the fills before it write into), or it makes one for an
/// indirect update to read its regions from. A program that only maps, moves and fences allocates
/// none, so a device's own memory model has nothing to announce for it.
///
/// It is a structural test of the program rather than a flag on the row, so a row that grows a
/// read or an indirect buffer declares the notice without anybody remembering to say so.
fn reaches_a_shared_buffer(program: &SparseMappingProgram) -> bool {
    !program.buffers.is_empty()
        || program
            .steps
            .iter()
            .any(|step| matches!(step, MappingStep::ReadRegion { .. }))
}

/// One row of the family: a case name, and the program it runs.
struct Row {
    /// The case's name: its recording's file name, and its row in the report.
    case: &'static str,
    /// The program the row runs, built with no device at all so a guard can read it.
    program: fn() -> SparseMappingProgram,
    /// The notices the row takes **beyond** [`EVERY_ROWS_NOTICES`], each a degradation the
    /// runtime announces on the path this row alone drives. Empty for every row but the indirect
    /// one.
    notices: &'static [Degradation],
}

/// The program a row states, over `textures`, with `buffers`, `fences` and `steps`.
fn program(
    textures: Vec<SparseTextureSpec>,
    buffers: Vec<Vec<u8>>,
    fences: usize,
    steps: Vec<MappingStep>,
) -> SparseMappingProgram {
    SparseMappingProgram {
        heap: HEAP,
        textures,
        buffers,
        fences,
        tile: TILE,
        steps,
    }
}

/// An update of one region of one level and slice of texture 0, then a commit: the shape most of
/// the sweep is made of, so the sweep reads as the values it varies.
fn update(mode: MappingMode, region: TileRect, level: u64, slice: u64) -> Vec<MappingStep> {
    vec![
        MappingStep::One {
            texture: 0,
            mode,
            region,
            level,
            slice,
        },
        MappingStep::Commit,
    ]
}

/// **Both modes, over the region sizes Metal documents a special case for.**
///
/// Nine commits, so nine `usedSize` readings beside the baseline: a map of no tiles (which maps
/// nothing), of one tile, of a row of tiles, of a level's whole grid, of the last tile of the
/// level, and then the same regions unmapped. `usedSize` moves with each, which is what makes the
/// sweep a measurement rather than five encodings of one answer.
static MODES_AND_SIZES: Row = Row {
    case: "sparse-mapping-modes-and-region-sizes",
    notices: &[],
    program: || {
        let mut steps = Vec::new();
        for mode in MappingMode::ALL {
            for region in [
                // A region of no tiles: Metal maps nothing for it, whatever the mode.
                TileRect::none([0, 0]),
                TileRect::one([0, 0]),
                // A row of tiles, so a side that mapped a column shows.
                TileRect::new([0, 0], [tiles_at(0)[0], 1]),
                whole_level(0),
                // The far corner, which is where a rounding error lands.
                last_tile(0),
            ] {
                steps.extend(update(mode, region, 0, 0));
            }
        }
        program(vec![plain()], Vec::new(), 0, steps)
    },
};

/// **Every level inside the texture, and the tail.**
///
/// Level 0, 1 and 2 hold 16, 4 and 1 tiles, so each raises `usedSize` by its own number. Level
/// [`TAIL_LEVEL`] is `firstMipmapInTail`, where Metal maps the slice's **whole tail** however small
/// the region — documented behaviour nothing in this tree had measured.
///
/// A level **past** the texture's own is [`A_LEVEL_PAST_THE_TEXTURES`], a row of its own, because
/// our side refuses it and an abort there would leave every number below uncompared.
static LEVELS_AND_THE_TAIL: Row = Row {
    case: "sparse-mapping-levels-and-the-tail",
    notices: &[],
    program: || {
        let mut steps = Vec::new();
        for level in [0, 1, 2] {
            steps.extend(update(
                MappingMode::Map,
                whole_level(level),
                level as u64,
                0,
            ));
        }
        // The tail, named by one tile: Metal maps the whole tail for it.
        steps.extend(update(
            MappingMode::Map,
            TileRect::one([0, 0]),
            TAIL_LEVEL,
            0,
        ));
        // And unmapped in the same order, so every rise has a fall to read against.
        for level in [0, 1, 2] {
            steps.extend(update(
                MappingMode::Unmap,
                whole_level(level),
                level as u64,
                0,
            ));
        }
        steps.extend(update(
            MappingMode::Unmap,
            TileRect::one([0, 0]),
            TAIL_LEVEL,
            0,
        ));
        program(vec![plain()], Vec::new(), 0, steps)
    },
};

/// **A level past the texture's own**, on its own, because the two oracles disagree about it.
///
/// Plain Metal takes the call and maps nothing: measured here, `usedSize` unchanged. Metal's own
/// API validation layer stops it (`mipLevel(9) must be <= firstMipmapInTail(3)`), so the two
/// oracles disagree, and the runtime follows plain Metal — the behavioural one — on the direct
/// selector as it already did on the indirect one. It refused the call typed until this row made
/// the disagreement visible.
///
/// It is one step and a commit, so the reading costs no other number: that is why it is not a row
/// of [`LEVELS_AND_THE_TAIL`].
static A_LEVEL_PAST_THE_TEXTURES: Row = Row {
    case: "sparse-mapping-a-level-past-the-textures",
    notices: &[],
    program: || {
        program(
            vec![plain()],
            Vec::new(),
            0,
            update(MappingMode::Map, TileRect::one([0, 0]), PAST_THE_LEVELS, 0),
        )
    },
};

/// **A slice past the texture's own**, on its own, and the one row of the family that reads
/// texels to answer a question `usedSize` cannot.
///
/// Plain Metal **takes** the call, as it takes a level past the texture's: measured on this M3 Max
/// through the direct selector (`probes/sparse_mapping_past_the_bounds.m`), where a map of one tile
/// of slice 2 of a two-slice texture raised the heap's `usedSize` by one page and the command
/// buffer completed with no error. So a typed refusal of ours would be a refusal where Metal
/// allows, and the runtime maps nothing and counts nothing for it
/// (`UnitGrid::clipped_region`). Metal's own API validation layer stops the call, which is the
/// second oracle and not this one.
///
/// **A page in `usedSize` does not say which tiles an update reached, and that is why this row
/// reads texels.** A heap counter moves for a tile anywhere, or for no tile at all, so a case
/// built on it alone passes while the question stays open — which is exactly how the gap this row
/// closes survived. Four tiles are mapped and filled with [`SLICE_FILLS`]' four distinct bytes,
/// two in each real slice, and read back **before and after** an update of the slice past the
/// texture's. The two readings separate the rules:
///
/// - **the update reaches no tile** (what the runtime does): all four tiles read their own byte
///   in both readings, and only `usedSize` differs;
/// - **the slice is read as slice 0**: slice 0's tile (1, 0) is unmapped by the row's `Unmap`
///   step and reads zero in the second reading, where no fill of this row writes zero;
/// - **the slice is clamped to the last**: slice 1's tile (1, 0) reads zero instead;
/// - **the update reaches a tile of the slice it names and nothing visible**: the four tiles are
///   as they were and `usedSize` moved, which is the counter-without-a-tile answer and is
///   distinguishable from the first because `usedSize` moved.
///
/// Both modes are stated, because they are measured to differ here: a map of such a region took a
/// page and an unmap of one can take the count below zero. The `Unmap` is the mode the texel
/// reading hangs on — it names a tile both real slices hold **mapped and filled**, so a side that
/// reached slice 0 or slice 1 with it loses a byte somebody wrote, and every read of this row is
/// still of a tile the program itself mapped.
static A_SLICE_PAST_THE_TEXTURES: Row = Row {
    case: "sparse-mapping-a-slice-past-the-textures",
    notices: &[],
    program: || {
        // The two tiles of level 0 each real slice holds mapped: the second is the one the
        // out-of-bounds `Unmap` names.
        const HELD: [TileRect; 2] = [TileRect::one([0, 0]), TileRect::one([1, 0])];
        // The four tiles read back, in slice then tile order, so the reads line up with
        // [`SLICE_FILLS`].
        let reads = || {
            let mut steps = Vec::new();
            for slice in 0..SLICES as u64 {
                for region in HELD {
                    steps.push(MappingStep::ReadRegion {
                        texture: 0,
                        level: 0,
                        slice,
                        region,
                    });
                }
            }
            steps
        };
        let mut steps = Vec::new();
        for slice in 0..SLICES as u64 {
            for region in HELD {
                steps.push(MappingStep::One {
                    texture: 0,
                    mode: MappingMode::Map,
                    region,
                    level: 0,
                    slice,
                });
            }
        }
        for slice in 0..SLICES as u64 {
            for (which, region) in HELD.iter().enumerate() {
                steps.push(MappingStep::FillRegion {
                    texture: 0,
                    level: 0,
                    slice,
                    region: *region,
                    value: SLICE_FILLS[slice as usize * HELD.len() + which],
                });
            }
        }
        // The control reading: four mapped tiles holding four distinct bytes. Without it a second
        // reading of all zeroes would be indistinguishable from a fill that never landed.
        steps.extend(reads());
        steps.push(MappingStep::Commit);
        // A map of a region of a slice past the texture's own, at a tile no real slice has
        // mapped, so a page Metal takes for it is a page this row did not ask for anywhere.
        steps.extend(update(
            MappingMode::Map,
            TileRect::one([2, 0]),
            0,
            PAST_THE_SLICES,
        ));
        // And an unmap of one, at a tile both real slices hold mapped and filled: the step whose
        // effect the reading after it can actually see.
        steps.extend(update(MappingMode::Unmap, HELD[1], 0, PAST_THE_SLICES));
        steps.extend(reads());
        steps.push(MappingStep::Commit);
        program(vec![array()], Vec::new(), 0, steps)
    },
};

/// **Both slices of an array texture**, mapped one at a time and then together.
///
/// A slice's tiles are tiles of its own, so `usedSize` rises twice; an implementation that mapped
/// slice 0 for both reads the first rise twice and the case moves.
static SLICES_OF_AN_ARRAY: Row = Row {
    case: "sparse-mapping-slices-of-an-array",
    notices: &[],
    program: || {
        let mut steps = Vec::new();
        for slice in 0..SLICES as u64 {
            steps.extend(update(MappingMode::Map, whole_level(0), 0, slice));
        }
        for slice in 0..SLICES as u64 {
            steps.extend(update(MappingMode::Unmap, whole_level(0), 0, slice));
        }
        program(vec![array()], Vec::new(), 0, steps)
    },
};

/// **The plural selector against the singular one**, over the same tiles.
///
/// `updateTextureMappings:mode:regions:mipLevels:slices:numRegions:` takes a region, a level and a
/// slice each, so this row states several of each in one call and reads `usedSize` against the
/// singular row's own numbers for the same tiles. A list of one is in too, because a range setter
/// over one element is the shape a side is most likely to read as a range of none.
static PLURAL_UPDATES: Row = Row {
    case: "sparse-mapping-plural-updates",
    notices: &[],
    program: || {
        let regions = vec![
            (whole_level(0), 0, 0),
            (whole_level(1), 1, 1),
            (TileRect::one([0, 0]), 2, 0),
            // A region of no tiles among regions that hold some, so a side that gave up on the
            // list at the empty one shows.
            (TileRect::none([0, 0]), 0, 1),
            (last_tile(0), 0, 1),
        ];
        let steps = vec![
            // A list of one first, so its own number stands apart from the list of five.
            MappingStep::Many {
                texture: 0,
                mode: MappingMode::Map,
                regions: vec![(TileRect::one([0, 0]), 2, 0)],
            },
            MappingStep::Commit,
            MappingStep::Many {
                texture: 0,
                mode: MappingMode::Map,
                regions: regions.clone(),
            },
            MappingStep::Commit,
            MappingStep::Many {
                texture: 0,
                mode: MappingMode::Unmap,
                regions,
            },
            MappingStep::Commit,
        ];
        program(vec![array()], Vec::new(), 0, steps)
    },
};

/// **The indirect selector**, over argument counts of 0, 1 and several, at two offsets.
///
/// Metal reads a 32-bit count and then that many `MTLMapIndirectArguments` from the buffer: each
/// is the region's origin and size in tiles, the level and the slice, as 32-bit words. The row
/// states three such blocks in one buffer and names each by its offset, so the count and the
/// offset are both swept, and a count of 0 is the negative fact beside them.
static INDIRECT_UPDATES: Row = Row {
    case: "sparse-mapping-indirect-updates",
    // The runtime reads the indirect regions on the **host**, once the work before the update
    // completed, and announces that it did. The case takes the notice and compares every number
    // exactly; what it may not do is take a different number because of one, and it does not.
    notices: &[Degradation::HostReadMappingArguments],
    program: || {
        let steps = vec![
            // A count of 0 at offset 0: nothing maps.
            MappingStep::Indirect {
                texture: 0,
                mode: MappingMode::Map,
                buffer: 0,
                offset: 0,
            },
            MappingStep::Commit,
            // One region at the second block's own nonzero, four-byte-aligned offset.
            MappingStep::Indirect {
                texture: 0,
                mode: MappingMode::Map,
                buffer: 0,
                offset: ONE_REGION_AT,
            },
            MappingStep::Commit,
            // Three regions at the third block's offset.
            MappingStep::Indirect {
                texture: 0,
                mode: MappingMode::Map,
                buffer: 0,
                offset: THREE_REGIONS_AT,
            },
            MappingStep::Commit,
            // And the three unmapped again, so each rise has its fall.
            MappingStep::Indirect {
                texture: 0,
                mode: MappingMode::Unmap,
                buffer: 0,
                offset: THREE_REGIONS_AT,
            },
            MappingStep::Commit,
        ];
        program(vec![plain()], vec![indirect_arguments()], 0, steps)
    },
};

/// Where the three argument blocks of [`INDIRECT_UPDATES`]'s buffer lie: offset 0 holds the count
/// of 0, and the other two a nonzero multiple of 4 each.
const ONE_REGION_AT: u64 = 4;
const THREE_REGIONS_AT: u64 = 4 + 4 + ARGUMENT_WORDS * 4;

/// The words of one `MTLMapIndirectArguments`: the region's origin and its size in tiles, then the
/// level and the slice. Eight, which is the layout Metal's header declares and the runtime reads.
const ARGUMENT_WORDS: u64 = 8;

/// The buffer [`INDIRECT_UPDATES`] reads: a count of 0, then a count of 1 with its region, then a
/// count of 3 with theirs, each `MTLMapIndirectArguments` six 32-bit words — the origin and the
/// size in tiles, the level and the slice.
fn indirect_arguments() -> Vec<u8> {
    let argument = |region: &TileRect, level: u32, slice: u32| {
        [
            region.origin[0] as u32,
            region.origin[1] as u32,
            region.origin[2] as u32,
            region.size[0] as u32,
            region.size[1] as u32,
            region.size[2] as u32,
            level,
            slice,
        ]
    };
    // Block 0 at offset 0: a count of 0 and no arguments.
    let mut words: Vec<u32> = vec![0];
    // Block 1 at `ONE_REGION_AT`: one region.
    words.push(1);
    words.extend(argument(&TileRect::one([0, 0]), 0, 0));
    // Block 2 at `THREE_REGIONS_AT`: three, one of them of no tiles.
    words.push(3);
    for (rect, level) in [
        (TileRect::one([1, 1]), 0u32),
        (whole_level(1), 1),
        (TileRect::none([0, 0]), 0),
    ] {
        words.extend(argument(&rect, level, 0));
    }
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

/// **`moveTextureMappingsFromTexture:…`**, onto an unmapped tile and onto a mapped one, within one
/// texture and between two.
///
/// The runtime's own documentation says a move leaves both heaps' `usedSize` alone and that a tile
/// whose destination is already mapped stays where it is. Nothing had measured either. This row
/// states both: `usedSize` after the move is read against `usedSize` before it, and the move onto
/// a mapped tile is the one that would raise it if an implementation took a page for it.
static MOVES: Row = Row {
    case: "sparse-mapping-moves",
    notices: &[],
    program: || {
        let one = [1, 1, 1];
        let steps = vec![
            // Tile (0,0) of texture 0 mapped; its neighbour is not.
            MappingStep::One {
                texture: 0,
                mode: MappingMode::Map,
                region: TileRect::one([0, 0]),
                level: 0,
                slice: 0,
            },
            MappingStep::Commit,
            // Onto an unmapped tile of the same texture: the mapping moves, `usedSize` does not.
            MappingStep::Move {
                from: 0,
                from_slice: 0,
                from_level: 0,
                from_origin: [0, 0, 0],
                size: one,
                to: 0,
                to_slice: 0,
                to_level: 0,
                to_origin: [1, 0, 0],
            },
            MappingStep::Commit,
            // Map the tile it came from, then move onto it: the destination is mapped, so Metal
            // leaves the source where it is and takes no page for it.
            MappingStep::One {
                texture: 0,
                mode: MappingMode::Map,
                region: TileRect::one([0, 0]),
                level: 0,
                slice: 0,
            },
            MappingStep::Commit,
            MappingStep::Move {
                from: 0,
                from_slice: 0,
                from_level: 0,
                from_origin: [1, 0, 0],
                size: one,
                to: 0,
                to_slice: 0,
                to_level: 0,
                to_origin: [0, 0, 0],
            },
            MappingStep::Commit,
            // Between two textures of the one heap, onto an unmapped tile.
            MappingStep::Move {
                from: 0,
                from_slice: 0,
                from_level: 0,
                from_origin: [0, 0, 0],
                size: one,
                to: 1,
                to_slice: 0,
                to_level: 0,
                to_origin: [0, 0, 0],
            },
            MappingStep::Commit,
            // A move of no tiles: nothing moves, and nothing is taken.
            MappingStep::Move {
                from: 0,
                from_slice: 0,
                from_level: 0,
                from_origin: [0, 0, 0],
                size: [0, 0, 0],
                to: 1,
                to_slice: 0,
                to_level: 0,
                to_origin: [1, 0, 0],
            },
            MappingStep::Commit,
        ];
        program(vec![plain(), plain()], Vec::new(), 0, steps)
    },
};

/// The byte tile (`x`, `y`) of texture `texture` is filled with, where a row of the move family
/// needs each tile's provenance read off its own value.
///
/// **Each field has its own bits**, so two tiles cannot share a byte: the texture at bit 5, the
/// row at bits 2 and 3, the column at bits 0 and 1, over a constant that makes every answer
/// nonzero and none of them `0xff`. Those two readings a texel can have for a reason other than a
/// fill of ours — an unmapped tile reads 0 through a blit on this oracle, and a freshly mapped
/// tile reads `0xff` (`probes/sparse_unmapped_blit.m`) — so a marker of either value would be
/// indistinguishable from a tile nobody wrote.
///
/// It is written as fields rather than as a table because the probe this family's move rows came
/// from got it wrong the other way: its first form OR'd the row into a constant that already held
/// that bit, so two rows shared a word and a tile's provenance was recoverable only by
/// elimination. `tests::every_tiles_marker_is_its_own` reads every marker of the grid back and
/// fails on a repeat, so the distinctness is checked rather than argued.
const fn marker(texture: usize, x: u64, y: u64) -> u8 {
    0x40 | ((texture as u8) << 5) | ((y as u8) << 2) | (x as u8)
}

/// The source texture of every move row, and the destination.
const SOURCE: usize = 0;
const DESTINATION: usize = 1;

/// The tiles of `rect`, in Metal's own order (x, then y).
fn tiles_of(rect: &TileRect) -> Vec<[u64; 2]> {
    let mut out = Vec::new();
    for down in 0..rect.size[1] {
        for across in 0..rect.size[0] {
            out.push([rect.origin[0] + across, rect.origin[1] + down]);
        }
    }
    out
}

/// The program a move row runs: the tiles `held` names mapped and filled with their own markers
/// and read back as the control, then `moves`, then the tiles `after` names read back.
///
/// Three command buffers, as `probes/sparse_move_past_the_bounds.m` uses three: a mapping is bound
/// before the encoders after it in the same command buffer, but what a **move** rebound is read
/// after a commit of its own, so the reading cannot be of a tile whose mapping had not landed.
///
/// `held` is read **before** the move as well as filled, and that control is what makes the
/// reading after it a measurement: four tiles holding four distinct markers say the fills landed
/// and say which tile each one is, so a later all-zero reading is a tile the move did not reach
/// rather than a fill that never ran.
///
/// Both textures are made with `allowGPUOptimizedContents` off. Under Metal's default this oracle
/// keeps such a shader-read texture losslessly compressed, and a tile a move rebound in one then
/// reads its own texels in only 128 of its 4,096 words and 0 in the rest, where every descriptor
/// that turns compression off reads all 4,096 (`probes/sparse_move_compression.m`). A move is
/// documented to move the tile's mapping, and the rows read which tiles it moved, so they read
/// them where compression's own state cannot hide one; docs/API.md's accepted differences say
/// what a compressed texture reads.
fn moved(
    held: &[(usize, TileRect)],
    moves: MappingStep,
    after: &[(usize, [u64; 2])],
) -> SparseMappingProgram {
    let mut steps = Vec::new();
    for (texture, rect) in held {
        steps.push(MappingStep::One {
            texture: *texture,
            mode: MappingMode::Map,
            region: *rect,
            level: 0,
            slice: 0,
        });
    }
    for (texture, rect) in held {
        for at in tiles_of(rect) {
            steps.push(MappingStep::FillRegion {
                texture: *texture,
                level: 0,
                slice: 0,
                region: TileRect::one(at),
                value: marker(*texture, at[0], at[1]),
            });
        }
    }
    // The control: every tile this row filled, read one tile at a time so each answer is one
    // tile's own bytes rather than four tiles interleaved by texel row.
    for (texture, rect) in held {
        for at in tiles_of(rect) {
            steps.push(MappingStep::ReadRegion {
                texture: *texture,
                level: 0,
                slice: 0,
                region: TileRect::one(at),
            });
        }
    }
    steps.push(MappingStep::Commit);
    steps.push(moves);
    steps.push(MappingStep::Commit);
    for (texture, at) in after {
        steps.push(MappingStep::ReadRegion {
            texture: *texture,
            level: 0,
            slice: 0,
            region: TileRect::one(*at),
        });
    }
    steps.push(MappingStep::Commit);
    let texture = plain().unoptimized();
    program(vec![texture, texture], Vec::new(), 0, steps)
}

/// A move of `size` tiles at `from` of level 0 slice 0 of [`SOURCE`] onto `to` of [`DESTINATION`].
fn move_of(from: [u64; 2], size: [u64; 2], to: [u64; 2]) -> MappingStep {
    MappingStep::Move {
        from: SOURCE,
        from_slice: 0,
        from_level: 0,
        from_origin: [from[0], from[1], 0],
        size: [size[0], size[1], 1],
        to: DESTINATION,
        to_slice: 0,
        to_level: 0,
        to_origin: [to[0], to[1], 0],
    }
}

/// **A move whose two sides both lie inside the level, read off the texels**: the control the
/// three rows past an edge are read against.
///
/// 2 by 2 tiles at (0, 0) onto (1, 1), every tile of both sides inside level 0's four by four.
/// Metal moved them to the region's own offsets — B(1,1), B(2,1), B(1,2) and B(2,2) holding
/// A(0,0), A(1,0), A(0,1) and A(1,1) — and this is the row that says so, because
/// [`moves`] reads `usedSize` alone and a move moves none of it. **A move that cannot be seen to
/// move anything makes the rows below measurements of nothing**, which is why this row exists
/// rather than being folded into one of them.
///
/// It separates two rules on its own. The destination as the stated **region** reads these four
/// tiles; the destination as a sequence of pages from the origin would read B(1,1), B(2,0), B(3,0)
/// and B(2,1). Where the stated destination region fits in the level, Metal's run of tiles is
/// exactly that region, so the two agree here and the rows past an edge are where the run shows.
static MOVE_INSIDE_THE_LEVEL: Row = Row {
    case: "sparse-mapping-move-inside-the-level",
    notices: &[],
    program: || {
        moved(
            &[(SOURCE, TileRect::new([0, 0], [2, 2]))],
            move_of([0, 0], [2, 2], [1, 1]),
            &[
                (DESTINATION, [1, 1]),
                (DESTINATION, [2, 1]),
                (DESTINATION, [1, 2]),
                (DESTINATION, [2, 2]),
            ],
        )
    },
};

/// **A destination row that overflows the level, which does not clip**: the reading nobody could
/// have guessed and the one our own clipping got wrong.
///
/// Source (0, 0) of 4 by 1 tiles — every tile inside level 0's four — onto destination
/// origin (2, 0), where the level has two tiles to the right of the origin and the stated region
/// would need four. Metal moved **all four** source tiles: B(2,0), B(3,0), B(2,1) and B(3,1) took
/// A(0,0), A(1,0), A(2,0) and A(3,0), each read off the destination tile's own first texel, and
/// A's whole row 0 read zero afterwards (`probes/sparse_move_past_the_bounds.m`, shape
/// `region-dest`, an M3 Max under macOS 27.0.1, plain Metal, every tile's verdict read off all
/// 4,096 of its texels, and a destination tile reading as mixed rather than as that word
/// throughout).
///
/// So the destination is a **run of tiles rather than a region**: it starts at the destination
/// origin, is laid out `sourceSize.width` tiles to a row — clipped only by the tiles to the right
/// of the origin — and takes the next row when one fills. A clip of the destination region moves
/// **two**, leaving B(2,1) and B(3,1) unmapped and reading zero, which is what the runtime did
/// until the rule was measured; this row is what would catch it coming back, and what would catch
/// another implementation doing it.
static MOVE_PAST_THE_DESTINATIONS_ROW: Row = Row {
    case: "sparse-mapping-move-past-the-destinations-row",
    notices: &[],
    program: || {
        moved(
            &[(SOURCE, TileRect::new([0, 0], [4, 1]))],
            move_of([0, 0], [4, 1], [2, 0]),
            &[
                (DESTINATION, [2, 0]),
                (DESTINATION, [3, 0]),
                (DESTINATION, [2, 1]),
                (DESTINATION, [3, 1]),
            ],
        )
    },
};

/// **A source region past the level's own tiles**: the part outside moves nowhere, and the
/// destination tiles the stated region named but the run did not reach keep their own texels.
///
/// Source (2, 0) of 4 by 1 tiles where level 0 holds four, so tiles (2,0) and (3,0) are inside and
/// the two past the edge are not, onto destination origin (0, 0). Metal moved the two: B(0,0) and
/// B(1,0) took A(2,0) and A(3,0), each read off the destination tile's own first texel, and A's
/// row 1 was whole (`probes/sparse_move_past_the_bounds.m`, shape `region-open`). So the overhang
/// does not **wrap** — a rule that continued into the next row would have put A(0,1) and A(1,1)
/// into B(2,0) and B(3,0) and left A's row 1 unmapped, and this row reads A(0,1) and A(1,1) after
/// the move for exactly that reason.
///
/// And B(2,0) and B(3,0) are **mapped and filled with their own markers before the move**, which
/// is the probe's `region-mapped` shape: the stated destination region names them and the run
/// never reaches them. Metal left both holding what they held, so a rule that zeroed the
/// destination tiles it named but did not fill reads two markers of ours as zero here. That
/// separation needs a known **nonzero** byte in those tiles before the move, which an unmapped
/// tile cannot carry — an unmapped tile reads zero either way — so it is the one place this family
/// maps a destination tile ahead of a move.
static MOVE_OF_A_SOURCE_PAST_THE_LEVEL: Row = Row {
    case: "sparse-mapping-move-of-a-source-past-the-level",
    notices: &[],
    program: || {
        moved(
            &[
                // The two source tiles that survive the clip, and the two a wrap would take.
                (SOURCE, TileRect::new([2, 0], [2, 1])),
                (SOURCE, TileRect::new([0, 1], [2, 1])),
                // The destination tiles the stated region names past the run's own two.
                (DESTINATION, TileRect::new([2, 0], [2, 1])),
            ],
            move_of([2, 0], [4, 1], [0, 0]),
            &[
                (DESTINATION, [0, 0]),
                (DESTINATION, [1, 0]),
                (DESTINATION, [2, 0]),
                (DESTINATION, [3, 0]),
                (SOURCE, [0, 1]),
                (SOURCE, [1, 1]),
            ],
        )
    },
};

/// **A source clipped on both axes**, which is the shape a pairing by list index gets wrong.
///
/// Source (2, 2) of 4 by 4 tiles on a level holding four by four, so 2 by 2 survive, onto
/// destination origin (1, 1), where three tiles lie to the right. B(1,1), B(2,1), B(3,1) and
/// B(1,2) took A(2,2), A(3,2), A(2,3) and A(3,3), each read off the destination tile's own first
/// texel (`probes/sparse_move_past_the_bounds.m`, shape `src-clip-2d`).
///
/// Its third tile at (3, 1) is the whole reading: the run's row is the **stated** width's three
/// and not the clipped source's two, which would have put it at (1, 2). And pairing two clipped
/// lists by index is wrong here in a second way — a source clipped to 2 by 2 against a destination
/// region of 4 by 4 lists four sources against sixteen destinations, and index 2 is the source's
/// second **row** against the destination's third **column**. Two of the four pairs were wrong in
/// the runtime and nothing held it; this row is what holds it.
static MOVE_OF_A_CLIPPED_SOURCE: Row = Row {
    case: "sparse-mapping-move-of-a-clipped-source",
    notices: &[],
    program: || {
        moved(
            &[(SOURCE, TileRect::new([2, 2], [2, 2]))],
            move_of([2, 2], [4, 4], [1, 1]),
            &[
                (DESTINATION, [1, 1]),
                (DESTINATION, [2, 1]),
                (DESTINATION, [3, 1]),
                (DESTINATION, [1, 2]),
            ],
        )
    },
};

/// **`updateFence:` and `waitForFence:` of a resource state encoder**, between two encoders of one
/// command buffer.
///
/// Stated plainly: **Metal orders a command buffer's encoders anyway**, so the fence is redundant
/// here and this row does not prove it orders anything. What it pins is that encoding the pair
/// changes none of the tiles and none of the bytes, which is what a consumer that fences
/// defensively needs, and it is what makes the two items exercised rather than named. The row
/// after it, [`mapped_tiles_hold_their_texels`], is where a mapping's effect is actually read.
static FENCES: Row = Row {
    case: "sparse-mapping-fences",
    notices: &[],
    program: || {
        let steps = vec![
            MappingStep::One {
                texture: 0,
                mode: MappingMode::Map,
                region: whole_level(0),
                level: 0,
                slice: 0,
            },
            MappingStep::UpdateFence { fence: 0 },
            // A second resource state encoder of the same command buffer, which waits for it.
            MappingStep::NextEncoder,
            MappingStep::WaitForFence { fence: 0 },
            MappingStep::One {
                texture: 0,
                mode: MappingMode::Map,
                region: whole_level(1),
                level: 1,
                slice: 0,
            },
            MappingStep::Commit,
        ];
        program(vec![plain()], Vec::new(), 1, steps)
    },
};

/// **The mapping backed real memory**: known bytes written into mapped tiles and read back.
///
/// `usedSize` says a page was taken; this says the page is the texture's. One tile of level 0 is
/// mapped, filled with [`FILL`] through a blit, and read back through another; every byte of the
/// read-back compares. The fill and the read name **only** tiles this program mapped, which
/// `every_fill_and_read_lies_in_mapped_tiles` holds: a blit over an unmapped tile is a shape
/// nobody here has measured.
static MAPPED_TEXELS: Row = Row {
    case: "sparse-mapping-mapped-tiles-hold-their-texels",
    notices: &[],
    program: || {
        let region = TileRect::one([0, 0]);
        let steps = vec![
            MappingStep::One {
                texture: 0,
                mode: MappingMode::Map,
                region,
                level: 0,
                slice: 0,
            },
            // One command buffer: the map is bound before the encoders after it, so the fill
            // reaches a tile that is mapped.
            MappingStep::FillRegion {
                texture: 0,
                level: 0,
                slice: 0,
                region,
                value: FILL,
            },
            MappingStep::ReadRegion {
                texture: 0,
                level: 0,
                slice: 0,
                region,
            },
            MappingStep::Commit,
        ];
        program(vec![plain()], Vec::new(), 0, steps)
    },
};

/// Every row of the family, in the order the report lists them.
#[cfg(test)]
static EVERY_ROW: [&Row; 14] = [
    &MODES_AND_SIZES,
    &LEVELS_AND_THE_TAIL,
    &A_LEVEL_PAST_THE_TEXTURES,
    &SLICES_OF_AN_ARRAY,
    &A_SLICE_PAST_THE_TEXTURES,
    &PLURAL_UPDATES,
    &INDIRECT_UPDATES,
    &MOVES,
    &MOVE_INSIDE_THE_LEVEL,
    &MOVE_PAST_THE_DESTINATIONS_ROW,
    &MOVE_OF_A_SOURCE_PAST_THE_LEVEL,
    &MOVE_OF_A_CLIPPED_SOURCE,
    &FENCES,
    &MAPPED_TEXELS,
];

/// A case of the family: one row.
pub struct SparseMappings {
    row: &'static Row,
}

impl Case for SparseMappings {
    fn name(&self) -> &'static str {
        self.row.case
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // **No value is loosened**: every number and every byte of every row compares exactly. The
        // only declarations are the notices a row's own path announces, which the runtime must be
        // allowed to say; a notice a row does not declare reads as `UndeclaredNotice`, naming the
        // code and the case, which is the loud form.
        //
        // Two kinds are declared beyond a row's own. A device that keeps a sparse texture's
        // residency less strictly than Metal says so (`non-strict-residency`), which llvmpipe
        // announced for every row of the family. And the `Shared` buffer a row that reads texels
        // back carries ([`crate::replay::shared_memory_notice`]), which the RTX 5080 announced for
        // the two rows that read any -- `sparse-mapping-mapped-tiles-hold-their-texels` and
        // `sparse-mapping-indirect-updates`. The rest answer mappings and numbers alone, so they
        // allocate no such buffer and declare nothing for one.
        let mut named: Vec<Allowance> = EVERY_ROWS_NOTICES
            .iter()
            .chain(self.row.notices)
            .chain(std::iter::once(&Degradation::NonStrictResidency))
            .map(|code| notice_allowance(code.code()))
            .collect();
        if reaches_a_shared_buffer(&(self.row.program)()) {
            named.push(crate::replay::shared_memory_notice());
        }
        named
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        // The program itself: the heap, every texture, the indirect arguments, the tile grid the
        // regions are stated in, and every command with its own numbers. A program whose regions,
        // modes or order moved is a change to the case's inputs, so a recording cannot stay the
        // answer to a program that is no longer this one.
        Ok(vec![Input::bytes("program", &(self.row.program)().words())])
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let program = (self.row.program)();
        // The call's own site and the program's answer take **different** names: two siblings
        // under one name are paired by whichever namesake the comparison reaches first, which is
        // how a round reported six cases divergent having compared no byte of either side. The
        // suite holds it for every recording (`tests::no_recorded_site_shares_its_name_with_a_
        // sibling`), and it caught this family's first recordings.
        let mapped = crate::case::made(sites, "sparse-mapping", side.sparse_mapping(&program))?;
        sites.push(mapped.node("program", &program.steps));
        // A refused step stops the case here, after every step's own site is emitted: the numbers
        // a refused update did not change would otherwise read as divergences of their own, each a
        // row of the ledger for one bug.
        if let Some((at, failure)) = mapped.first_refusal() {
            return Err(Abort {
                at: format!("program/step/{at}/{}", program.steps[at].call()),
                failure: failure.clone(),
            });
        }
        for (at, bytes) in mapped.reads.iter().enumerate() {
            let length = bytes.len() as u64;
            sites.push(Node::leaf(
                format!("read/{at}"),
                Observed::bytes(bytes.clone(), ByteLayout::buffer(0, length, Element::U32)),
            ));
        }
        Ok(())
    }
}

/// The case of both mapping modes over the region sizes Metal documents a special case for.
pub fn modes_and_region_sizes() -> SparseMappings {
    SparseMappings {
        row: &MODES_AND_SIZES,
    }
}

/// The case of every level a region can name, the tail and one past the texture's own included.
pub fn levels_and_the_tail() -> SparseMappings {
    SparseMappings {
        row: &LEVELS_AND_THE_TAIL,
    }
}

/// The case of a level past the texture's own, which our side refuses and plain Metal takes.
pub fn a_level_past_the_textures() -> SparseMappings {
    SparseMappings {
        row: &A_LEVEL_PAST_THE_TEXTURES,
    }
}

/// The case of both slices of an array texture.
pub fn slices_of_an_array() -> SparseMappings {
    SparseMappings {
        row: &SLICES_OF_AN_ARRAY,
    }
}

/// The case of a slice past the texture's own, which plain Metal and our side both take a page
/// for, with the texels four mapped tiles hold either side of it (plain Metal's page lands in one
/// of them, ours nowhere: an accepted difference, `docs/API.md`).
pub fn a_slice_past_the_textures() -> SparseMappings {
    SparseMappings {
        row: &A_SLICE_PAST_THE_TEXTURES,
    }
}

/// The case of the plural update selector.
pub fn plural_updates() -> SparseMappings {
    SparseMappings {
        row: &PLURAL_UPDATES,
    }
}

/// The case of the indirect update selector.
pub fn indirect_updates() -> SparseMappings {
    SparseMappings {
        row: &INDIRECT_UPDATES,
    }
}

/// The case of the mapping moves.
pub fn moves() -> SparseMappings {
    SparseMappings { row: &MOVES }
}

/// The case of a move whose two sides both lie inside the level, read off the texels: the control
/// the three rows past an edge are read against.
pub fn move_inside_the_level() -> SparseMappings {
    SparseMappings {
        row: &MOVE_INSIDE_THE_LEVEL,
    }
}

/// The case of a destination row that overflows the level, which Metal does not clip.
pub fn move_past_the_destinations_row() -> SparseMappings {
    SparseMappings {
        row: &MOVE_PAST_THE_DESTINATIONS_ROW,
    }
}

/// The case of a source region past the level's own tiles, whose overhang moves nowhere.
pub fn move_of_a_source_past_the_level() -> SparseMappings {
    SparseMappings {
        row: &MOVE_OF_A_SOURCE_PAST_THE_LEVEL,
    }
}

/// The case of a source clipped on both axes, which a pairing by list index gets wrong.
pub fn move_of_a_clipped_source() -> SparseMappings {
    SparseMappings {
        row: &MOVE_OF_A_CLIPPED_SOURCE,
    }
}

/// The case of a fence between two resource state encoders of one command buffer.
pub fn mapping_fences() -> SparseMappings {
    SparseMappings { row: &FENCES }
}

/// The case that reads the texels of a tile it mapped.
pub fn mapped_tiles_hold_their_texels() -> SparseMappings {
    SparseMappings {
        row: &MAPPED_TEXELS,
    }
}

/// Every case of the family, in the order the report lists them.
pub fn all() -> Vec<Box<dyn Case>> {
    vec![
        Box::new(modes_and_region_sizes()),
        Box::new(levels_and_the_tail()),
        Box::new(a_level_past_the_textures()),
        Box::new(slices_of_an_array()),
        Box::new(a_slice_past_the_textures()),
        Box::new(plural_updates()),
        Box::new(indirect_updates()),
        Box::new(moves()),
        Box::new(move_inside_the_level()),
        Box::new(move_past_the_destinations_row()),
        Box::new(move_of_a_source_past_the_level()),
        Box::new(move_of_a_clipped_source()),
        Box::new(mapping_fences()),
        Box::new(mapped_tiles_hold_their_texels()),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeMap, BTreeSet};

    /// The cases name distinct recordings, and `all` holds every row.
    #[test]
    fn the_cases_are_distinct_and_complete() {
        let named: BTreeSet<&str> = all().iter().map(|case| case.name()).collect();
        assert_eq!(named.len(), all().len());
        assert_eq!(named.len(), EVERY_ROW.len());
        let rows: BTreeSet<&str> = EVERY_ROW.iter().map(|row| row.case).collect();
        assert_eq!(rows, named);
    }

    /// Every case states its inputs and loosens nothing, and each row's program fingerprints
    /// differently.
    ///
    /// Without the last half a recording would stay the answer to a program whose regions or modes
    /// had moved, which for this family is the difference between mapping one tile and mapping a
    /// whole level.
    #[test]
    fn every_case_states_its_inputs_and_loosens_nothing() {
        let mut digests = BTreeSet::new();
        for case in all() {
            let inputs = case
                .inputs()
                .unwrap_or_else(|why| panic!("{}: {why}", case.name()));
            assert!(!inputs.is_empty(), "{}", case.name());
            // Every row declares the notices every row takes and the one a device's own sparse
            // residency provokes; the indirect row adds its own, and the two rows that reach a
            // `Shared` buffer add the one a device's own memory model provokes. No value
            // tolerance appears anywhere in this family.
            let allowed = case.allowances();
            let expects = EVERY_ROWS_NOTICES.len()
                + 1
                + match case.name() == INDIRECT_UPDATES.case {
                    true => INDIRECT_UPDATES.notices.len(),
                    false => 0,
                }
                + usize::from(reaches_a_shared_buffer(&(EVERY_ROW
                    .iter()
                    .find(|row| row.case == case.name())
                    .expect("a row of the family")
                    .program)()));
            assert_eq!(allowed.len(), expects, "{}", case.name());
            for allowance in &allowed {
                assert!(
                    matches!(allowance.loosening, crate::allowance::Loosening::Unchecked),
                    "{}: a value tolerance in a family that loosens none",
                    case.name()
                );
            }
            digests.insert(crate::sha256_hex(
                &inputs[0].digest.hex.clone().into_bytes(),
            ));
        }
        assert_eq!(digests.len(), EVERY_ROW.len());
    }

    /// **Every region of every row lies inside its level's own tiles, and a move's runs past them
    /// only as far as the measurement did.**
    ///
    /// This is the family's safety rule, held by a test rather than by care: a region past a
    /// level's tiles is the one shape here that could fault a device, and no row may grow one by
    /// accident. A level past the texture's own is allowed and is a row of its own — Metal maps
    /// nothing for it — so the check bounds the region against the **tail's** grid there, which is
    /// the smallest any level has.
    ///
    /// A **slice** past the texture's own is allowed on the same terms, and only by exactly one:
    /// [`A_SLICE_PAST_THE_TEXTURES`] is the row that names it, plain Metal is measured to take such
    /// an update, and the region it names is still bounded by its level's own tiles here — which is
    /// the bound that could fault a device. The slice indexes images and not memory within one, so
    /// a slice one past the texture's cannot reach past a level's grid however the device reads it.
    ///
    /// # A move is bounded differently, because Metal's own answer to one is measured
    ///
    /// A **move** may state a region past its level's tiles, and the four move rows past an edge
    /// are the whole point of their existence: what Metal does with one is the rule the runtime
    /// implements, and the mapping selectors' bound is what the clause above keeps. The relief is
    /// structural — it is the step's kind and not a row's name — and it is bounded by the
    /// measurement rather than opened:
    ///
    /// - each side's **origin lies inside** its level's own tiles, so neither side starts outside
    ///   the grid. A move from or onto an origin outside it is one nothing has read, and
    ///   [`moved_tiles`] answers no pair for one, so a row stating one would assert nothing;
    /// - the region's size on an axis is **at most the tiles that level holds** on it, which is
    ///   the largest overhang `probes/sparse_move_past_the_bounds.m` measured (a region of four on
    ///   a level of four, at an origin of two). So every tile a move's region names lies within
    ///   one grid of the level, and `size.width` of 0xffff_ffff is not a thing a row can grow into;
    /// - the depth stays one image, as every other region here does: a region deeper than one is
    ///   the part of the rule no shape measured.
    ///
    /// The last two clauses count what they allowed, because a relief nothing uses is a relief
    /// that stopped holding anything: the family states moves both inside the bounds and past
    /// them, and this fails if either disappears.
    #[test]
    fn every_region_lies_inside_its_level() {
        let (mut inside, mut past) = (0usize, 0usize);
        for row in EVERY_ROW {
            let program = (row.program)();
            let slices = u64::from(slices_of(&program));
            for (at, step) in program.steps.iter().enumerate() {
                let moving = matches!(step, MappingStep::Move { .. });
                for (region, level, slice) in regions_of(step) {
                    let grid = tiles_at(level.min(u64::from(LEVELS) - 1) as u32);
                    assert!(
                        slice < slices
                            || (slice == slices && row.case == A_SLICE_PAST_THE_TEXTURES.case),
                        "{}: step {at} names slice {slice} of a texture of {slices}",
                        row.case
                    );
                    for (axis, tiles) in grid.iter().enumerate() {
                        let past_the_grid = region.origin[axis] + region.size[axis];
                        if !moving {
                            assert!(
                                past_the_grid <= *tiles,
                                "{}: step {at} reaches tile {past_the_grid} of {} on axis {axis} \
                                 of level {level}",
                                row.case,
                                tiles
                            );
                            continue;
                        }
                        // A move's own bound: the origin inside the level, and no more tiles
                        // stated than the level holds.
                        assert!(
                            region.origin[axis] < *tiles,
                            "{}: step {at} moves from or onto tile {} of {} on axis {axis} of \
                             level {level}, which is outside the level and is a shape nothing has \
                             measured",
                            row.case,
                            region.origin[axis],
                            tiles
                        );
                        assert!(
                            region.size[axis] <= *tiles,
                            "{}: step {at} states {} tiles on axis {axis} of a level holding {}, \
                             which is a larger overhang than any shape measured",
                            row.case,
                            region.size[axis],
                            tiles
                        );
                        match past_the_grid <= *tiles {
                            true => inside += 1,
                            false => past += 1,
                        }
                    }
                }
                assert!(
                    regions_of(step)
                        .iter()
                        .all(|(region, _, _)| region.size[2] == region.size[2].min(1)),
                    "{}: step {at} states a region deeper than one image",
                    row.case
                );
            }
        }
        // Neither half of the relief may vanish: the family states moves inside the bounds and
        // moves past them, and a count of zero either way means this check stopped reading one.
        assert!(inside > 0, "no row states a move inside its level's tiles");
        assert!(past > 0, "no row states a move past its level's tiles");
    }

    /// The regions a step names, each with the level and the slice it names them in. A step that
    /// names none answers none.
    fn regions_of(step: &MappingStep) -> Vec<(TileRect, u64, u64)> {
        match step {
            MappingStep::One {
                region,
                level,
                slice,
                ..
            } => vec![(*region, *level, *slice)],
            MappingStep::Many { regions, .. } => regions.clone(),
            MappingStep::Move {
                from_origin,
                size,
                from_level,
                from_slice,
                to_origin,
                to_level,
                to_slice,
                ..
            } => vec![
                (
                    TileRect {
                        origin: *from_origin,
                        size: *size,
                    },
                    *from_level,
                    *from_slice,
                ),
                (
                    TileRect {
                        origin: *to_origin,
                        size: *size,
                    },
                    *to_level,
                    *to_slice,
                ),
            ],
            MappingStep::FillRegion {
                region,
                level,
                slice,
                ..
            }
            | MappingStep::ReadRegion {
                region,
                level,
                slice,
                ..
            } => vec![(*region, *level, *slice)],
            // The indirect selector's regions are the buffer's, checked by their own guard; the
            // rest name none.
            MappingStep::Indirect { .. }
            | MappingStep::UpdateFence { .. }
            | MappingStep::WaitForFence { .. }
            | MappingStep::NextEncoder
            | MappingStep::Commit => Vec::new(),
        }
    }

    /// The slices every texture of `program` holds, which is the fewest any step may name.
    fn slices_of(program: &SparseMappingProgram) -> u32 {
        program
            .textures
            .iter()
            .map(|spec| spec.array_length.max(1))
            .min()
            .unwrap_or(1)
    }

    /// **The indirect buffer's own regions lie inside their level's tiles too**, and its counts
    /// really are at the offsets the row names.
    ///
    /// The regions the side maps there come out of the buffer rather than out of a step, so the
    /// guard above cannot see them. Without this the row would hand Metal whatever lay at those
    /// offsets, which is the one place in this family a region could reach past a level.
    #[test]
    fn the_indirect_arguments_lie_inside_their_levels() {
        let bytes = indirect_arguments();
        let word = |at: usize| {
            u32::from_le_bytes(bytes[at * 4..at * 4 + 4].try_into().expect("four bytes")) as u64
        };
        for (offset, wanted) in [(0u64, 0u64), (ONE_REGION_AT, 1), (THREE_REGIONS_AT, 3)] {
            assert_eq!(offset % 4, 0, "the offset is not four-byte aligned");
            let at = (offset / 4) as usize;
            assert!(at < bytes.len() / 4, "the count lies past the buffer");
            let count = word(at);
            assert_eq!(count, wanted, "the count at {offset}");
            for region in 0..count as usize {
                let from = at + 1 + region * ARGUMENT_WORDS as usize;
                assert!(
                    (from + ARGUMENT_WORDS as usize) * 4 <= bytes.len(),
                    "argument {region} at {offset} runs past the buffer"
                );
                let level = word(from + 6);
                let slice = word(from + 7);
                let grid = tiles_at(level as u32);
                assert!(level < u64::from(LEVELS), "level {level}");
                assert_eq!(slice, 0, "slice {slice} of a texture of one");
                for (axis, tiles) in grid.iter().enumerate() {
                    let past = word(from + axis) + word(from + 3 + axis);
                    assert!(
                        past <= *tiles,
                        "argument {region} at {offset} reaches tile {past} of {tiles} on axis \
                         {axis} of level {level}"
                    );
                }
            }
        }
    }

    /// **Every fill and every read names tiles the program mapped and has not unmapped** — a
    /// move's own mapping included.
    ///
    /// A blit over an unmapped sparse tile reads zero on this oracle
    /// (`probes/sparse_unmapped_blit.m`), so a row that stated one would record a reading that
    /// says nothing, and the four move rows hang on telling a tile the move reached from one it
    /// did not. The check walks each program's own steps in order, keeping the tiles mapped so
    /// far, and fails naming the step that reached outside them.
    ///
    /// # A move moves a mapping, and this models which
    ///
    /// Until the move rows landed the walk treated a move as touching nothing, which was true of
    /// every row there was and is false of theirs: a move unmaps the source tiles it moved and
    /// maps the destination tiles they landed on, and the rows read exactly those. So the walk
    /// applies [`moved_tiles`] — **Metal's own measured rule**, written here rather than asked of
    /// the runtime, because the runtime is the thing under test — and a read of a tile that rule
    /// maps is legal while a read of one it does not is caught as loudly as before.
    ///
    /// A pair whose **destination is already mapped** moves nothing, which is Metal's own
    /// documented rule and which [`MOVES`] states a move for, so the clause is exercised rather
    /// than assumed: the source keeps its mapping and the destination keeps its own texels.
    /// [`MOVE_OF_A_SOURCE_PAST_THE_LEVEL`] maps two destination tiles ahead of its move for the
    /// other half of that rule — tiles the stated region names and the **run never reaches**,
    /// which Metal left holding their own texels rather than zeroing.
    ///
    /// The tile keys carry the **texture** as well as the slice, the level and the two axes, which
    /// they did not before. Every row that fills had one texture, so nothing had noticed that a
    /// program could map tile (0, 0) of texture 0 and read tile (0, 0) of texture 1 and pass; the
    /// move rows have two textures and fill both, so the key had to become the tile's own identity.
    #[test]
    fn every_fill_and_read_lies_in_mapped_tiles() {
        let mut fills = 0usize;
        let mut moved = 0usize;
        let mut onto_the_mapped = 0usize;
        for row in EVERY_ROW {
            let program = (row.program)();
            let mut mapped: BTreeSet<Tile> = BTreeSet::new();
            for (at, step) in program.steps.iter().enumerate() {
                match step {
                    MappingStep::One {
                        texture,
                        mode,
                        region,
                        level,
                        slice,
                    } => {
                        for tile in tiles_in(*texture, region, *level, *slice) {
                            match mode {
                                MappingMode::Map => mapped.insert(tile),
                                MappingMode::Unmap => mapped.remove(&tile),
                            };
                        }
                    }
                    MappingStep::Many {
                        texture,
                        mode,
                        regions,
                    } => {
                        for (region, level, slice) in regions {
                            for tile in tiles_in(*texture, region, *level, *slice) {
                                match mode {
                                    MappingMode::Map => mapped.insert(tile),
                                    MappingMode::Unmap => mapped.remove(&tile),
                                };
                            }
                        }
                    }
                    MappingStep::FillRegion {
                        texture,
                        level,
                        slice,
                        region,
                        ..
                    }
                    | MappingStep::ReadRegion {
                        texture,
                        level,
                        slice,
                        region,
                    } => {
                        for tile in tiles_in(*texture, region, *level, *slice) {
                            assert!(
                                mapped.contains(&tile),
                                "{}: step {at} reaches the unmapped tile {tile:?}",
                                row.case
                            );
                            fills += 1;
                        }
                    }
                    // A move moves a mapping: the source tiles it moved are unmapped afterwards
                    // and the destination tiles they landed on are mapped, so a read after one is
                    // bounded by what the move left rather than by what was mapped before it.
                    MappingStep::Move { .. } => {
                        for (source, destination) in moved_tiles(&program, step) {
                            // Metal moves nothing onto a tile that is already mapped: the source
                            // keeps its mapping and the destination keeps its own texels.
                            if mapped.contains(&destination) {
                                onto_the_mapped += 1;
                                continue;
                            }
                            mapped.remove(&source);
                            mapped.insert(destination);
                            moved += 1;
                        }
                    }
                    // An indirect update, a fence, an encoder boundary and a commit touch no
                    // texel and bind no tile this walk can see, so a fill after one is still
                    // bounded by what was mapped before it.
                    MappingStep::Indirect { .. }
                    | MappingStep::UpdateFence { .. }
                    | MappingStep::WaitForFence { .. }
                    | MappingStep::NextEncoder
                    | MappingStep::Commit => {}
                }
            }
        }
        // And the check is not vacuous: six rows fill and read, so the walk above ran. One tile
        // filled and one read by `mapped_tiles_hold_their_texels`; four filled and eight read
        // (two readings of four tiles) by `a_slice_past_the_textures`; four filled and eight read
        // by each of the three move rows whose reading is of four tiles, and six filled and
        // twelve read by `move-of-a-source-past-the-level`.
        assert_eq!(
            fills,
            2 + 12 + 3 * 12 + 18,
            "no row fills or reads a tile at all"
        );
        // The move clause moved something, and the already-mapped clause fired. `MOVES` states
        // four moves of one tile and one of none: two move, one is onto a tile already mapped and
        // one names no tile. The four move rows move four, four, two (the other two tiles of the
        // source region lie past the level) and four.
        assert_eq!(moved, 2 + 4 + 4 + 2 + 4, "the walk modelled no move at all");
        assert_eq!(
            onto_the_mapped, 1,
            "no row moves onto a tile already mapped"
        );
    }

    /// A tile's own identity: its texture, its slice, its level and its two axes.
    type Tile = (usize, u64, u64, u64, u64);

    /// The tiles `region` of `level` and `slice` of `texture` holds, as the keys the walk above
    /// tracks.
    fn tiles_in(texture: usize, region: &TileRect, level: u64, slice: u64) -> Vec<Tile> {
        let mut out = Vec::new();
        for down in 0..region.size[1] {
            for across in 0..region.size[0] {
                out.push((
                    texture,
                    slice,
                    level,
                    region.origin[0] + across,
                    region.origin[1] + down,
                ));
            }
        }
        out
    }

    /// **Metal's own rule for which tiles a move moves**, each source tile with the tile it moves
    /// onto, in Metal's order — written here so the suite's expectation is Metal's rather than the
    /// runtime's.
    ///
    /// Measured off the texels either side of the move on an Apple M3 Max under macOS 27.0.1,
    /// plain Metal, one shape per process, each tile's verdict read off all 4,096 of its texels
    /// (`probes/sparse_move_past_the_bounds.m`). Eight shapes and one rule answers all eight:
    ///
    /// - the **source region is clipped** to the level's own tiles on each axis, and its surviving
    ///   tiles move in Metal's order (x, then y). The part outside moves nothing anywhere: it does
    ///   not wrap and it takes no destination tile. A source level or slice past the texture's own
    ///   moves nothing at all;
    /// - the **destination is a run of tiles, not a region**: it starts at the destination origin,
    ///   is laid out `size.width` tiles to a row — the width **as stated**, clipped only by the
    ///   tiles the level has to the right of the origin — and takes the next row when one fills,
    ///   one tile per surviving source tile. The width is clipped and the height is **not**.
    ///
    /// Two things are **unmeasured** and this answers no pair for either, so no row can assert
    /// one: a run that would pass the level's **last row**, and a region deeper than one image.
    /// The **tail** is a third: at `firstMipmapInTail` a side's units are one each however many
    /// tiles the region names, which this does not model, so it refuses a move that names a tail
    /// level rather than answering a pairing nobody measured.
    fn moved_tiles(program: &SparseMappingProgram, step: &MappingStep) -> Vec<(Tile, Tile)> {
        let MappingStep::Move {
            from,
            from_slice,
            from_level,
            from_origin,
            size,
            to,
            to_slice,
            to_level,
            to_origin,
        } = step
        else {
            return Vec::new();
        };
        for level in [from_level, to_level] {
            assert!(
                *level < TAIL_LEVEL,
                "a move of level {level}, at or past the tail, whose pairing this rule does not \
                 model"
            );
        }
        let slices = u64::from(slices_of(program));
        // A level or slice past the texture's own moves nothing at all, on either side.
        if *from_slice >= slices || *to_slice >= slices {
            return Vec::new();
        }
        let source = tiles_at(*from_level as u32);
        let destination = tiles_at(*to_level as u32);
        // An origin outside its level, a size of no tiles, and a region deeper than one image each
        // move nothing this rule can answer for.
        if (0..2).any(|axis| {
            from_origin[axis] >= source[axis]
                || to_origin[axis] >= destination[axis]
                || size[axis] == 0
        }) || size[2] != 1
        {
            return Vec::new();
        }
        let held: [u64; 2] =
            std::array::from_fn(|axis| size[axis].min(source[axis] - from_origin[axis]));
        // The run's row: the width as the move states it, clipped by the tiles to the right of the
        // destination origin. The stated width and not the clipped source's, which is the reading
        // `move-of-a-clipped-source` is about.
        let per_row = size[0].min(destination[0] - to_origin[0]).max(1);
        let mut pairs = Vec::new();
        for (at, tile) in tiles_in(
            *from,
            &TileRect::new([from_origin[0], from_origin[1]], held),
            *from_level,
            *from_slice,
        )
        .into_iter()
        .enumerate()
        {
            let at = at as u64;
            let (x, y) = (to_origin[0] + at % per_row, to_origin[1] + at / per_row);
            // Past the level's last row the destination is unmeasured, so nothing moves there
            // rather than moving somewhere nobody has read.
            if y >= destination[1] {
                continue;
            }
            pairs.push((tile, (*to, *to_slice, *to_level, x, y)));
        }
        pairs
    }

    /// **Every tile's marker is its own**, and none of them is a byte a texel can read for a
    /// reason other than a fill of ours.
    ///
    /// The probe these rows came from got this wrong: its first per-tile word OR'd the row into a
    /// constant that already held that bit, so rows 0 and 1 shared a word and rows 2 and 3 shared
    /// one, and a tile's provenance was recoverable only by elimination. A marker table is worth
    /// nothing unless the markers differ, so this reads every one of them back.
    #[test]
    fn every_tiles_marker_is_its_own() {
        let mut seen: BTreeMap<u8, (usize, u64, u64)> = BTreeMap::new();
        let [across, down] = tiles_at(0);
        for texture in [SOURCE, DESTINATION] {
            for y in 0..down {
                for x in 0..across {
                    let byte = marker(texture, x, y);
                    assert_ne!(
                        byte, 0,
                        "texture {texture} tile ({x}, {y}) marks itself zero"
                    );
                    assert_ne!(
                        byte, 0xff,
                        "texture {texture} tile ({x}, {y}) marks itself 0xff"
                    );
                    if let Some(held) = seen.insert(byte, (texture, x, y)) {
                        panic!("{byte:#04x} marks both {held:?} and ({texture}, {x}, {y})");
                    }
                }
            }
        }
        // Two textures of sixteen tiles each, every one with a byte of its own.
        assert_eq!(seen.len(), 2 * (across * down) as usize);
        // And each row of a move actually uses distinct markers, which is the property the
        // readings hang on rather than the table's own.
        for row in [
            &MOVE_INSIDE_THE_LEVEL,
            &MOVE_PAST_THE_DESTINATIONS_ROW,
            &MOVE_OF_A_SOURCE_PAST_THE_LEVEL,
            &MOVE_OF_A_CLIPPED_SOURCE,
        ] {
            let filled: Vec<u8> = (row.program)()
                .steps
                .iter()
                .filter_map(|step| match step {
                    MappingStep::FillRegion { value, .. } => Some(*value),
                    _ => None,
                })
                .collect();
            let distinct: BTreeSet<u8> = filled.iter().copied().collect();
            assert!(
                filled.len() >= 4,
                "{}: too few fills to tell apart",
                row.case
            );
            assert_eq!(
                distinct.len(),
                filled.len(),
                "{}: two tiles share a marker",
                row.case
            );
        }
    }

    /// **Each move row's pairing is the reading `probes/sparse_move_past_the_bounds.m` took off
    /// Metal**, asserted against the shape that produced it, with no device at all.
    ///
    /// [`moved_tiles`] is what the walk above believes a move does, and what the move rows' reads
    /// are placed by. Believing it is not the same as checking it: this holds it to Metal's own
    /// four readings, by tile, so a change to the rule that still let the guard pass fails here
    /// instead. The runtime has the mirror of this test over the same eight shapes
    /// (`metal2vulkan-runtime`'s `tests::sparse::a_move_is_metals_run_of_destination_tiles`); this
    /// one is the suite's, and the two were written from the same probe output rather than from
    /// each other.
    #[test]
    fn every_move_rows_pairing_is_the_probes_own_reading() {
        // Metal's own reading for each row: the source tile, then the destination tile it landed
        // on, in Metal's order.
        /// One of Metal's readings: the source tile, and the destination tile it landed on.
        type Reading = ([u64; 2], [u64; 2]);
        let readings: [(&Row, &[Reading]); 4] = [
            // dest-2d: both sides inside, so the run is exactly the stated region.
            (
                &MOVE_INSIDE_THE_LEVEL,
                &[
                    ([0, 0], [1, 1]),
                    ([1, 0], [2, 1]),
                    ([0, 1], [1, 2]),
                    ([1, 1], [2, 2]),
                ],
            ),
            // region-dest: all four move, the last two onto the row below.
            (
                &MOVE_PAST_THE_DESTINATIONS_ROW,
                &[
                    ([0, 0], [2, 0]),
                    ([1, 0], [3, 0]),
                    ([2, 0], [2, 1]),
                    ([3, 0], [3, 1]),
                ],
            ),
            // region-open: the two tiles past the level's edge move nowhere.
            (
                &MOVE_OF_A_SOURCE_PAST_THE_LEVEL,
                &[([2, 0], [0, 0]), ([3, 0], [1, 0])],
            ),
            // src-clip-2d: the run's row is the stated width's three, so the third tile is at
            // (3, 1) and not at (1, 2).
            (
                &MOVE_OF_A_CLIPPED_SOURCE,
                &[
                    ([2, 2], [1, 1]),
                    ([3, 2], [2, 1]),
                    ([2, 3], [3, 1]),
                    ([3, 3], [1, 2]),
                ],
            ),
        ];
        for (row, metal) in readings {
            let program = (row.program)();
            let moves: Vec<&MappingStep> = program
                .steps
                .iter()
                .filter(|step| matches!(step, MappingStep::Move { .. }))
                .collect();
            assert_eq!(moves.len(), 1, "{}: not one move", row.case);
            let pairs: Vec<Reading> = moved_tiles(&program, moves[0])
                .into_iter()
                .map(|(source, destination)| {
                    assert_eq!(
                        source.0, SOURCE,
                        "{}: a source of another texture",
                        row.case
                    );
                    assert_eq!(
                        destination.0, DESTINATION,
                        "{}: a destination of another texture",
                        row.case
                    );
                    ([source.3, source.4], [destination.3, destination.4])
                })
                .collect();
            assert_eq!(pairs, metal.to_vec(), "{}", row.case);
            // And the row reads **every** destination tile Metal's reading names, so a pairing
            // that moved a tile the row never looks at cannot pass unnoticed.
            let read: BTreeSet<[u64; 2]> = program
                .steps
                .iter()
                .skip_while(|step| !matches!(step, MappingStep::Move { .. }))
                .filter_map(|step| match step {
                    MappingStep::ReadRegion {
                        texture, region, ..
                    } if *texture == DESTINATION => Some([region.origin[0], region.origin[1]]),
                    _ => None,
                })
                .collect();
            for (_, destination) in metal {
                assert!(
                    read.contains(destination),
                    "{}: nothing reads the destination tile {destination:?} Metal moved onto",
                    row.case
                );
            }
        }
    }

    /// **Every row ends on a commit, and reads exactly what its steps asked for.**
    ///
    /// A program that left a command buffer uncommitted would record `usedSize` readings for work
    /// that never ran, and a read whose commit never came would answer the guard bytes.
    #[test]
    fn every_row_ends_on_a_commit_and_reads_after_one() {
        for row in EVERY_ROW {
            let program = (row.program)();
            assert_eq!(
                program.steps.last(),
                Some(&MappingStep::Commit),
                "{}: the program does not end on a commit",
                row.case
            );
            assert!(program.commits() > 0, "{}", row.case);
            // Every read lies before a commit that follows it.
            let mut after_last_commit = 0usize;
            for step in &program.steps {
                match step {
                    MappingStep::Commit => after_last_commit = 0,
                    step if step.reads() => after_last_commit += 1,
                    _ => {}
                }
            }
            assert_eq!(
                after_last_commit, 0,
                "{}: a read with no commit after it",
                row.case
            );
        }
    }

    /// **The sweep really is a sweep**: both mapping modes appear, every update selector appears,
    /// and the levels and slices named across the family are the ones the module's table claims.
    ///
    /// This is the guard against the sweep quietly shrinking. A row edited down to one mode, or a
    /// selector dropped, would leave the table in the module docs claiming coverage the rows no
    /// longer have.
    #[test]
    fn the_sweep_covers_the_values_the_table_claims() {
        let mut modes: BTreeSet<&'static str> = BTreeSet::new();
        let mut calls: BTreeSet<&'static str> = BTreeSet::new();
        let mut levels: BTreeSet<u64> = BTreeSet::new();
        let mut slices: BTreeSet<u64> = BTreeSet::new();
        let mut sizes: BTreeMap<u64, usize> = BTreeMap::new();
        for row in EVERY_ROW {
            for step in &(row.program)().steps {
                calls.insert(step.call());
                match step {
                    MappingStep::One { mode, .. } | MappingStep::Many { mode, .. } => {
                        modes.insert(mode.word());
                    }
                    MappingStep::Indirect { mode, .. } => {
                        modes.insert(mode.word());
                    }
                    _ => {}
                }
                for (region, level, slice) in regions_of(step) {
                    levels.insert(level);
                    slices.insert(slice);
                    *sizes.entry(region.tiles()).or_default() += 1;
                }
            }
        }
        // Both values of `MTLSparseTextureMappingMode`.
        assert_eq!(modes, BTreeSet::from(["map", "unmap"]));
        // Every one of the encoder's six reachable items.
        for call in [
            "updateTextureMapping",
            "updateTextureMappings",
            "updateTextureMappingIndirect",
            "moveTextureMappingsFromTexture",
            "updateFence",
            "waitForFence",
        ] {
            assert!(calls.contains(call), "no row calls {call}");
        }
        // Levels 0, 1, 2, the tail, and one past the texture's own.
        for level in [0, 1, 2, TAIL_LEVEL, PAST_THE_LEVELS] {
            assert!(levels.contains(&level), "no row names level {level}");
        }
        // 0 and 1 of the array, and one past the texture's two.
        assert_eq!(slices, BTreeSet::from([0, 1, PAST_THE_SLICES]));
        // A region of no tiles, of one, and of a level's whole grid.
        for tiles in [0, 1, tiles_at(0)[0] * tiles_at(0)[1]] {
            assert!(sizes.contains_key(&tiles), "no row states {tiles} tiles");
        }
        // And the move shapes the table claims: a source clipped on one axis, a source clipped on
        // two, a destination run that overflows its row, and a move with both sides inside. Each
        // is read off its own program rather than off a row's name, so a row edited into another
        // shape takes its claim with it.
        let mut shapes: BTreeSet<&'static str> = BTreeSet::new();
        for row in EVERY_ROW {
            let program = (row.program)();
            for step in &program.steps {
                let MappingStep::Move {
                    from_origin,
                    size,
                    to_origin,
                    ..
                } = step
                else {
                    continue;
                };
                if size[0] == 0 || size[1] == 0 {
                    continue;
                }
                let source = tiles_at(0);
                let clipped = (0..2)
                    .filter(|axis| from_origin[*axis] + size[*axis] > source[*axis])
                    .count();
                let overflows = to_origin[0] + size[0] > source[0];
                shapes.insert(match (clipped, overflows) {
                    (0, false) => "both sides inside",
                    (0, true) => "the destination's row overflows",
                    (1, _) => "the source is clipped on one axis",
                    _ => "the source is clipped on two axes",
                });
            }
        }
        assert_eq!(
            shapes,
            BTreeSet::from([
                "both sides inside",
                "the destination's row overflows",
                "the source is clipped on one axis",
                "the source is clipped on two axes",
            ])
        );
    }

    /// The tile grid the rows are written for is the one the module's prose names, and the levels
    /// it claims hold the tiles it claims.
    ///
    /// The numbers in the docs are what the rows' regions were bounded against, so they are
    /// checked rather than trusted. The device's own tile size is checked by the side, at run
    /// time, against this same grid.
    #[test]
    fn the_tile_grid_is_the_one_the_rows_were_written_for() {
        assert_eq!(TILE.tile, [64, 64, 1]);
        assert_eq!(tiles_at(0), [4, 4]);
        assert_eq!(tiles_at(1), [2, 2]);
        assert_eq!(tiles_at(2), [1, 1]);
        // The tail begins where a level is smaller than one tile.
        assert!((WIDTH as u64 >> TAIL_LEVEL) < TILE.tile[0]);
        assert!((WIDTH as u64 >> (TAIL_LEVEL - 1)) >= TILE.tile[0]);
        // One tile of level 0 is one page of the heap, which is what makes `usedSize` readable as
        // tiles.
        let (per_row, bytes) = TILE.bytes_of(&TileRect::one([0, 0]));
        assert_eq!(per_row, 64 * 4);
        assert_eq!(bytes, 64 * 64 * 4);
        assert_eq!(bytes, 16 << 10);
        // And the heap holds far more pages than any row maps.
        assert!(HEAP.size / bytes > 64);
    }
}
