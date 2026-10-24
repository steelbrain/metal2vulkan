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
//! | slice | 0 and 1 of a two-slice array | 2 | — |
//! | update selector | singular, plural, indirect | 3 | — |
//! | indirect argument count | 0, 1, several | — | a count past the buffer, which the runtime refuses typed and whose Metal behaviour is a refusal case of its own |
//! | indirect offset | 0, a nonzero multiple of 4 | — | an offset that is not a multiple of 4, which the runtime documents as read at the multiple below and which belongs to a refusal case |
//! | move overlap | onto an unmapped tile, onto a mapped one, within a texture, between two | — | — |
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
//! - **the slice** moves it: a second slice's tiles are tiles of their own;
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
//! A fill and a read name tiles the program mapped and nothing else
//! (`every_fill_and_read_lies_in_mapped_tiles`): a blit over an unmapped tile is a shape nobody
//! has measured here, and it stays out of the suite until somebody does — named in this round's
//! open items rather than guessed at.
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

/// The byte [`mapped_tiles_hold_their_texels`] fills its mapped tiles with. Not zero: a tile that
/// read zero would be indistinguishable from one the copy never reached.
const FILL: u8 = 0xc3;

/// The notices **every** row of the family takes.
///
/// A sparse texture is emulated wherever the device offers no sparse residency of its own — which
/// on this oracle's MoltenVK is every sparse texture — and the runtime announces that once per
/// texture. It is a device gap, announced honestly, so each row declares it and then compares
/// every number exactly; a venue with native sparse residency announces nothing and the
/// declaration simply does not fire there.
const EVERY_ROWS_NOTICES: [Degradation; 1] = [Degradation::EmulatedSparseTexture];

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
static EVERY_ROW: [&Row; 9] = [
    &MODES_AND_SIZES,
    &LEVELS_AND_THE_TAIL,
    &A_LEVEL_PAST_THE_TEXTURES,
    &SLICES_OF_AN_ARRAY,
    &PLURAL_UPDATES,
    &INDIRECT_UPDATES,
    &MOVES,
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
        Box::new(plural_updates()),
        Box::new(indirect_updates()),
        Box::new(moves()),
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

    /// **Every region of every row lies inside its level's own tiles.**
    ///
    /// This is the family's safety rule, held by a test rather than by care: a region past a
    /// level's tiles is the one shape here that could fault a device, and no row may grow one by
    /// accident. A level past the texture's own is allowed and is a row of its own — Metal maps
    /// nothing for it — so the check bounds the region against the **tail's** grid there, which is
    /// the smallest any level has.
    #[test]
    fn every_region_lies_inside_its_level() {
        for row in EVERY_ROW {
            let program = (row.program)();
            for (at, step) in program.steps.iter().enumerate() {
                for (region, level, slice) in regions_of(step) {
                    let grid = tiles_at(level.min(u64::from(LEVELS) - 1) as u32);
                    assert!(
                        slice < u64::from(slices_of(&program)),
                        "{}: step {at} names slice {slice}",
                        row.case
                    );
                    for (axis, tiles) in grid.iter().enumerate() {
                        let past = region.origin[axis] + region.size[axis];
                        assert!(
                            past <= *tiles,
                            "{}: step {at} reaches tile {past} of {} on axis {axis} of level \
                             {level}",
                            row.case,
                            tiles
                        );
                    }
                    assert_eq!(region.size[2], region.size[2].min(1));
                }
            }
        }
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

    /// **Every fill and every read names tiles the program mapped and has not unmapped.**
    ///
    /// A blit over an unmapped sparse tile is a shape nobody here has measured, so no row may
    /// state one. The check walks each program's own steps in order, keeping the tiles mapped so
    /// far, and fails naming the step that reached outside them.
    #[test]
    fn every_fill_and_read_lies_in_mapped_tiles() {
        let mut fills = 0usize;
        for row in EVERY_ROW {
            let program = (row.program)();
            let mut mapped: BTreeSet<(u64, u64, u64, u64)> = BTreeSet::new();
            for (at, step) in program.steps.iter().enumerate() {
                match step {
                    MappingStep::One {
                        mode,
                        region,
                        level,
                        slice,
                        ..
                    } => {
                        for tile in tiles_in(region, *level, *slice) {
                            match mode {
                                MappingMode::Map => mapped.insert(tile),
                                MappingMode::Unmap => mapped.remove(&tile),
                            };
                        }
                    }
                    MappingStep::Many { mode, regions, .. } => {
                        for (region, level, slice) in regions {
                            for tile in tiles_in(region, *level, *slice) {
                                match mode {
                                    MappingMode::Map => mapped.insert(tile),
                                    MappingMode::Unmap => mapped.remove(&tile),
                                };
                            }
                        }
                    }
                    MappingStep::FillRegion {
                        level,
                        slice,
                        region,
                        ..
                    }
                    | MappingStep::ReadRegion {
                        level,
                        slice,
                        region,
                        ..
                    } => {
                        for tile in tiles_in(region, *level, *slice) {
                            assert!(
                                mapped.contains(&tile),
                                "{}: step {at} reaches the unmapped tile {tile:?}",
                                row.case
                            );
                            fills += 1;
                        }
                    }
                    // A move, an indirect update, a fence, an encoder boundary and a commit touch
                    // no texel, so a fill after one is still bounded by what was mapped before it.
                    // No row of this family fills after a move.
                    MappingStep::Move { .. }
                    | MappingStep::Indirect { .. }
                    | MappingStep::UpdateFence { .. }
                    | MappingStep::WaitForFence { .. }
                    | MappingStep::NextEncoder
                    | MappingStep::Commit => {}
                }
            }
        }
        // And the check is not vacuous: one row does fill and read, so the walk above ran.
        assert_eq!(fills, 2, "no row fills or reads a tile at all");
    }

    /// The tiles `region` of `level` and `slice` holds, as the keys the walk above tracks.
    fn tiles_in(region: &TileRect, level: u64, slice: u64) -> Vec<(u64, u64, u64, u64)> {
        let mut out = Vec::new();
        for down in 0..region.size[1] {
            for across in 0..region.size[0] {
                out.push((
                    slice,
                    level,
                    region.origin[0] + across,
                    region.origin[1] + down,
                ));
            }
        }
        out
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
        assert_eq!(slices, BTreeSet::from([0, 1]));
        // A region of no tiles, of one, and of a level's whole grid.
        for tiles in [0, 1, tiles_at(0)[0] * tiles_at(0)[1]] {
            assert!(sizes.contains_key(&tiles), "no row states {tiles} tiles");
        }
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
