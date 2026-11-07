//! A sparse texture's mappings: Metal's units (tiles and per-slice tails) and the device's (blocks
//! and mip tails), how they cover each other, the texels each holds, and the state Metal's updates
//! leave.
//!
//! A device unit holds memory while any Metal unit it holds texels of is mapped, so Metal's 16-KiB
//! tiles map on a device whose blocks are 64 KiB: a batch binds a unit when the first of its Metal
//! units maps and zeroes all its texels ([`UnitGrid::bound_boxes`]), copies a move's texels tile by
//! tile ([`UnitGrid::move_boxes`]), zeroes the texels of each Metal unit it unmaps inside a unit
//! that stays bound ([`UnitGrid::unmapped_boxes`]), and unbinds a unit when the last of its Metal
//! units unmaps. An unmapped tile inside a bound unit therefore reads zero, Metal's answer; a unit
//! that holds no mapped tile reads the device's value.

use super::emulated::Written;
use super::texture::{SparseShape, VkSparseLayout};
use crate::values::{Origin, Region, Size};
use metal2vulkan::bind::TileMapping;
use metal2vulkan::metal::TextureType;
use std::collections::{BTreeSet, HashMap, HashSet};

/// One of Metal's units of a sparse texture: a tile of a level above the mip tail, or one slice's
/// whole tail.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) enum MetalUnit {
    Tile {
        level: u32,
        slice: u32,
        at: [u32; 3],
    },
    Tail {
        slice: u32,
    },
}

impl MetalUnit {
    /// Whether it is a slice's whole tail.
    pub(crate) fn is_tail(self) -> bool {
        matches!(self, MetalUnit::Tail { .. })
    }
}

/// One of the device's units of a sparse image: a block of a level above its mip tail, or the
/// mip tail of a layer (layer 0 for a tail shared by all layers).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) enum VkUnit {
    Block {
        level: u32,
        layer: u32,
        at: [u32; 3],
    },
    Tail {
        layer: u32,
    },
}

/// A box of texels of one level and layer of a texture's image.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct TexelBox {
    pub(crate) level: u32,
    pub(crate) layer: u32,
    pub(crate) origin: [u32; 3],
    pub(crate) size: [u32; 3],
}

impl TexelBox {
    /// The texels both boxes hold, if any.
    pub(crate) fn intersect(&self, other: &TexelBox) -> Option<TexelBox> {
        if (self.level, self.layer) != (other.level, other.layer) {
            return None;
        }
        let start = [0, 1, 2].map(|axis| self.origin[axis].max(other.origin[axis]));
        let end = [0, 1, 2].map(|axis| {
            (self.origin[axis] + self.size[axis]).min(other.origin[axis] + other.size[axis])
        });
        if (0..3).any(|axis| end[axis] <= start[axis]) {
            return None;
        }
        Some(TexelBox {
            level: self.level,
            layer: self.layer,
            origin: start,
            size: [0, 1, 2].map(|axis| end[axis] - start[axis]),
        })
    }

    /// The box in pieces of at most `piece` texels along each axis, from its origin.
    pub(crate) fn pieces(&self, piece: [u32; 3]) -> Vec<TexelBox> {
        let piece = piece.map(|axis| axis.max(1));
        let mut pieces = Vec::new();
        let end = [0, 1, 2].map(|axis| self.origin[axis] + self.size[axis]);
        let mut z = self.origin[2];
        while z < end[2] {
            let mut y = self.origin[1];
            while y < end[1] {
                let mut x = self.origin[0];
                while x < end[0] {
                    let origin = [x, y, z];
                    let size = [0, 1, 2].map(|axis| piece[axis].min(end[axis] - origin[axis]));
                    pieces.push(TexelBox {
                        level: self.level,
                        layer: self.layer,
                        origin,
                        size,
                    });
                    x += piece[0];
                }
                y += piece[1];
            }
            z += piece[2];
        }
        pieces
    }
}

/// What a texture has mapped and bound: Metal's mapped units, the memory of each bound device
/// unit, and the units of slices past its own that its heap counts
/// ([`UnitGrid::clipped_region`]).
#[derive(Default)]
pub(crate) struct SparseState {
    pub(crate) mapped: HashSet<MetalUnit>,
    pub(crate) bound: HashMap<VkUnit, crate::memory::Placement>,
    /// Units of a slice past the texture's own that a map took: its heap counts each, as Metal's
    /// `usedSize` does, and nothing holds texels of them.
    pub(crate) charged: HashSet<MetalUnit>,
}

/// The units of a sparse texture, as data: Metal's (its tile, and a slice's tail from
/// `firstMipmapInTail`) and the device's (its sparse layout), over its shape and its layers
/// (Metal's slices, a cube's faces each; 1 for a 3D texture).
#[derive(Clone, Copy)]
pub(crate) struct UnitGrid {
    pub(crate) shape: SparseShape,
    pub(crate) layers: u32,
    /// Metal's tile, in texels.
    pub(crate) tile: [u64; 3],
    pub(crate) first_mipmap_in_tail: u64,
    pub(crate) layout: VkSparseLayout,
}

/// The units of `unit` texels along each axis that cover `[start, end)`.
fn covering(start: [u32; 3], end: [u32; 3], unit: [u32; 3]) -> impl Iterator<Item = [u32; 3]> {
    let first = [0, 1, 2].map(|axis| start[axis] / unit[axis].max(1));
    let last = [0, 1, 2].map(|axis| end[axis].div_ceil(unit[axis].max(1)));
    (first[2]..last[2]).flat_map(move |z| {
        (first[1]..last[1]).flat_map(move |y| (first[0]..last[0]).map(move |x| [x, y, z]))
    })
}

/// An origin's three axes as the unit grid's own calls take them.
fn axes_of(origin: Origin) -> [u32; 3] {
    [origin.x, origin.y, origin.z].map(|axis| axis as u32)
}

impl UnitGrid {
    /// The extent of `level`, in texels: depth 1 but for a 3D texture.
    pub(crate) fn level_extent(&self, level: u32) -> [u32; 3] {
        let mut extent = self
            .shape
            .extent
            .map(|axis| u32::try_from((axis >> level.min(63)).max(1)).unwrap_or(u32::MAX));
        if self.shape.texture_type != TextureType::Type3D {
            extent[2] = 1;
        }
        extent
    }

    /// Metal's tile, in texels.
    pub(crate) fn tile32(&self) -> [u32; 3] {
        self.tile
            .map(|axis| u32::try_from(axis).unwrap_or(u32::MAX).max(1))
    }

    fn levels(&self) -> u32 {
        u32::try_from(self.shape.levels).unwrap_or(u32::MAX)
    }

    /// Metal's `firstMipmapInTail`.
    pub(crate) fn first_tail(&self) -> u32 {
        u32::try_from(self.first_mipmap_in_tail).unwrap_or(u32::MAX)
    }

    /// The level's grid of Metal tiles.
    pub(crate) fn tiles(&self, level: u32) -> [u32; 3] {
        let extent = self.level_extent(level);
        let tile = self.tile32();
        [0, 1, 2].map(|axis| extent[axis].div_ceil(tile[axis]))
    }

    /// The device layer of Metal's `slice`'s mip tail.
    fn tail_layer(&self, slice: u32) -> u32 {
        if self.layout.single_mip_tail {
            0
        } else {
            slice
        }
    }

    /// The Metal slices: the image's layers (a cube's faces each), 1 for a 3D texture.
    pub(crate) fn slices(&self) -> u32 {
        self.layers
    }

    /// Metal's units an update of `region` (in tiles) of `level` and `slice` touches, in Metal's
    /// order (x, then y, then z): the slice's tail at `firstMipmapInTail` whatever the region,
    /// nothing past it or for a region of no tiles. The caller checked the level, the slice, and
    /// a tiled level's region against the texture.
    pub(crate) fn update_units(
        &self,
        level: u32,
        slice: u32,
        origin: [u32; 3],
        size: [u32; 3],
    ) -> Vec<MetalUnit> {
        let first = self.first_tail();
        if level == first {
            return vec![MetalUnit::Tail { slice }];
        }
        if level > first || size.contains(&0) {
            return Vec::new();
        }
        let end = [0, 1, 2].map(|axis| origin[axis] + size[axis]);
        covering(origin, end, [1, 1, 1])
            .map(|at| MetalUnit::Tile { level, slice, at })
            .collect()
    }

    /// The device units `unit` lies in.
    pub(crate) fn vk_units_of(&self, unit: MetalUnit) -> Vec<VkUnit> {
        let vk_tail = self.layout.mip_tail_first_lod;
        match unit {
            MetalUnit::Tile { level, slice, at } => {
                if level >= vk_tail {
                    return vec![VkUnit::Tail {
                        layer: self.tail_layer(slice),
                    }];
                }
                let extent = self.level_extent(level);
                let tile = self.tile32();
                let start = [0, 1, 2].map(|axis| at[axis].saturating_mul(tile[axis]));
                let end = [0, 1, 2]
                    .map(|axis| (at[axis] + 1).saturating_mul(tile[axis]).min(extent[axis]));
                self.blocks(level, slice, start, end)
            }
            MetalUnit::Tail { slice } => {
                let mut units = Vec::new();
                for level in self.first_tail()..self.levels() {
                    if level >= vk_tail {
                        units.push(VkUnit::Tail {
                            layer: self.tail_layer(slice),
                        });
                        break;
                    }
                    let extent = self.level_extent(level);
                    units.extend(self.blocks(level, slice, [0, 0, 0], extent));
                }
                units
            }
        }
    }

    /// The blocks of `level` and `layer` that cover texels `[start, end)`.
    fn blocks(&self, level: u32, layer: u32, start: [u32; 3], end: [u32; 3]) -> Vec<VkUnit> {
        covering(start, end, self.layout.granularity)
            .map(|at| VkUnit::Block { level, layer, at })
            .collect()
    }

    /// The Metal units `unit` holds texels of.
    pub(crate) fn metal_units_of(&self, unit: VkUnit) -> Vec<MetalUnit> {
        let first = self.first_tail();
        match unit {
            VkUnit::Block { level, layer, at } => {
                if level >= first {
                    return vec![MetalUnit::Tail { slice: layer }];
                }
                let (start, size) = self.block_texels(level, at);
                let end = [0, 1, 2].map(|axis| start[axis] + size[axis]);
                covering(start, end, self.tile32())
                    .map(|at| MetalUnit::Tile {
                        level,
                        slice: layer,
                        at,
                    })
                    .collect()
            }
            VkUnit::Tail { layer } => {
                let slices = if self.layout.single_mip_tail {
                    0..self.slices()
                } else {
                    layer..layer + 1
                };
                let mut units = Vec::new();
                for slice in slices {
                    for level in self.layout.mip_tail_first_lod..first.min(self.levels()) {
                        let tiles = self.tiles(level);
                        units.extend(
                            covering([0, 0, 0], tiles, [1, 1, 1]).map(|at| MetalUnit::Tile {
                                level,
                                slice,
                                at,
                            }),
                        );
                    }
                    if first < self.levels() {
                        units.push(MetalUnit::Tail { slice });
                    }
                }
                units
            }
        }
    }

    /// Whether device unit `unit` holds memory while `mapped` are Metal's mapped units: some
    /// Metal unit it holds texels of is mapped.
    pub(crate) fn wanted(&self, unit: VkUnit, mapped: &HashSet<MetalUnit>) -> bool {
        self.metal_units_of(unit)
            .iter()
            .any(|metal| mapped.contains(metal))
    }

    /// The texels of Metal unit `unit`: its tile's, clipped at its level's edge, or each level of
    /// its slice's tail, whole.
    pub(crate) fn metal_boxes(&self, unit: MetalUnit) -> Vec<TexelBox> {
        match unit {
            MetalUnit::Tile { level, slice, at } => {
                let extent = self.level_extent(level);
                let tile = self.tile32();
                let origin = [0, 1, 2].map(|axis| at[axis].saturating_mul(tile[axis]));
                if (0..3).any(|axis| origin[axis] >= extent[axis]) {
                    return Vec::new();
                }
                let size = [0, 1, 2].map(|axis| tile[axis].min(extent[axis] - origin[axis]));
                vec![TexelBox {
                    level,
                    layer: slice,
                    origin,
                    size,
                }]
            }
            MetalUnit::Tail { slice } => (self.first_tail()..self.levels())
                .map(|level| self.level_box(level, slice))
                .collect(),
        }
    }

    /// The texels of device unit `unit`: its block's, clipped at its level's edge, or each level
    /// of its layer's tail (of every layer, for a tail shared by all), whole.
    pub(crate) fn vk_boxes(&self, unit: VkUnit) -> Vec<TexelBox> {
        match unit {
            VkUnit::Block { level, layer, at } => {
                let (origin, size) = self.block_texels(level, at);
                vec![TexelBox {
                    level,
                    layer,
                    origin,
                    size,
                }]
            }
            VkUnit::Tail { layer } => {
                let layers = if self.layout.single_mip_tail {
                    0..self.slices()
                } else {
                    layer..layer + 1
                };
                layers
                    .flat_map(|layer| {
                        (self.layout.mip_tail_first_lod..self.levels())
                            .map(move |level| (level, layer))
                    })
                    .map(|(level, layer)| self.level_box(level, layer))
                    .collect()
            }
        }
    }

    /// Level `level` of `layer`, whole.
    fn level_box(&self, level: u32, layer: u32) -> TexelBox {
        TexelBox {
            level,
            layer,
            origin: [0; 3],
            size: self.level_extent(level),
        }
    }

    /// The texels of the device units `bound`, which a batch binds to new memory: zeroed whole
    /// before anything is copied into them.
    pub(crate) fn bound_boxes(&self, bound: &BTreeSet<VkUnit>) -> BTreeSet<TexelBox> {
        bound.iter().flat_map(|&unit| self.vk_boxes(unit)).collect()
    }

    /// The texels of each of Metal's units `unmapped` inside each device unit that `held` says
    /// holds memory: zeroed after a batch, or after a blit wrote them, so an unmapped tile reads
    /// Metal's zero. An emulated texture's every unit holds memory, its own image's.
    pub(crate) fn unmapped_boxes(
        &self,
        unmapped: impl IntoIterator<Item = MetalUnit>,
        held: impl Fn(VkUnit) -> bool,
    ) -> BTreeSet<TexelBox> {
        let mut boxes = BTreeSet::new();
        for unit in unmapped {
            let texels = self.metal_boxes(unit);
            for vk_unit in self.vk_units_of(unit) {
                if !held(vk_unit) {
                    continue;
                }
                for held_box in self.vk_boxes(vk_unit) {
                    boxes.extend(texels.iter().filter_map(|b| b.intersect(&held_box)));
                }
            }
        }
        boxes
    }

    /// The texel boxes a move of Metal unit `from` of this grid to Metal unit `to` of grid
    /// `destination` copies, each source box with its destination: each level of the unit, from
    /// its origin, in the extent both hold. None where the two grids' formats, sample counts or
    /// tiles differ (a move Metal's validation stops), or where one unit is a tail and the other a
    /// tile (a move it takes): Metal's destination reads the moved bytes in another layout, and
    /// the runtime's reads zero there (a near-equivalent).
    pub(crate) fn move_boxes(
        &self,
        from: MetalUnit,
        destination: &UnitGrid,
        to: MetalUnit,
    ) -> Vec<(TexelBox, TexelBox)> {
        if self.shape.pixel_format != destination.shape.pixel_format
            || self.shape.samples != destination.shape.samples
            || self.tile != destination.tile
            || from.is_tail() != to.is_tail()
        {
            return Vec::new();
        }
        self.metal_boxes(from)
            .into_iter()
            .zip(destination.metal_boxes(to))
            .map(|(source, target)| {
                let size = [0, 1, 2].map(|axis| source.size[axis].min(target.size[axis]));
                (TexelBox { size, ..source }, TexelBox { size, ..target })
            })
            .filter(|(source, _)| !source.size.contains(&0))
            .collect()
    }

    /// The Metal units `written` touches: its slices' tails where its level is in the tail, else
    /// the tiles its box meets.
    pub(crate) fn units_within(&self, written: &Written) -> Vec<MetalUnit> {
        let first = self.first_tail();
        let tile = self.tile32();
        let mut units = Vec::new();
        for slice in written.layers.clone() {
            if written.level >= first {
                units.push(MetalUnit::Tail { slice });
                continue;
            }
            let end = [0, 1, 2].map(|axis| written.origin[axis] + written.size[axis]);
            units.extend(
                covering(written.origin, end, tile).map(|at| MetalUnit::Tile {
                    level: written.level,
                    slice,
                    at,
                }),
            );
        }
        units
    }

    /// Metal's units an update's `region` (in tiles) of `level` and `slice` touches, as plain Metal
    /// maps arguments its validation stops (the M3 Max under macOS 27.0.1): none at a level past
    /// the texture's, nor for a tiled level's region whose origin is past its tiles or whose size
    /// is 0 on an axis; else the region's tiles inside the level, the rest of it dropped. A level
    /// past `firstMipmapInTail` touches its slice's tail. A slice past the texture's own names the
    /// units the same region names on an in-bounds slice, at that slice
    /// ([`past_slices`](Self::past_slices)): units Metal's heap counts and nothing holds.
    ///
    /// One rule for all three mapping selectors — the direct update, the plural one and the
    /// indirect one — and, with the slice bounded ([`clipped_size`](Self::clipped_size)), for the
    /// move. Plain Metal **takes** each of these arguments where its own
    /// API validation layer stops them, so a typed refusal of ours would be a refusal where Metal
    /// allows. Measured through the direct
    /// `updateTextureMapping:mode:region:mipLevel:slice:` on this M3 Max, macOS 27.0.1
    /// (`metal2vulkan-conformance/probes/sparse_mapping_past_the_bounds.m`), over the suite's own
    /// 256 by 256 nine-level `R32Float` texture whose level 0 holds 4 by 4 tiles, the heap's
    /// `usedSize` 0 before each and the command buffer `Completed` with no error after: level 9
    /// maps nothing; a region of 8 by 8 tiles at the origin maps 16 pages, the level's own tiles
    /// and no more; an origin of (4, 4) maps nothing; and one tile of level 0 maps one page, which
    /// is the control that says the measurement works at all. The three move shapes complete with
    /// no error too, leaving `usedSize` where it was.
    ///
    /// Metal **does not bound the slice**, where it bounds the level and the origin. A region of
    /// a slice past the texture's costs Metal's heap exactly what the same region on an in-bounds
    /// slice costs — the region clipped to this level's tiles, one page each — and that charge
    /// is then an ordinary one: idempotent per tile, given back on an unmap of it, and never
    /// taking the count below where it stood. So the runtime counts those units in its heap and
    /// maps them nowhere (`SparseState::charged`). Measured through **both** mapping selectors on
    /// this M3 Max (the direct one above, and the indirect one over nineteen configurations in
    /// `metal2vulkan-conformance/probes/sparse_indirect_usedsize.m`), over level 0's tiles, which
    /// read alike: one page for one such tile, sixteen for a region of 8 by 8 on a level holding
    /// 4 by 4, one per out-of-bounds slice named, and nothing at all for a level or an origin
    /// past the bounds. Another level's region takes the same rule, a tail among them, through
    /// the direct selector, the indirect one and a view alike
    /// (`metal2vulkan-conformance/probes/sparse_slice_past_levels_usedsize.m`, on this M3 Max: a
    /// tile of level 1 or 2 one page, 8 by 8 tiles on a level holding 2 by 2 four, a tile of the
    /// first tail level the slice's tail, a level inside the tail past it nothing, and an unmap
    /// giving the page back). `tests::sparse` holds the readings row by row.
    ///
    /// Plain Metal's page for such a tile also lands at an address its own page layout derives,
    /// in a block of another slice's mapped tile (`sparse-mapping-a-slice-past-the-textures`
    /// reads it): not mirrored, an accepted difference (`docs/API.md`), since Vulkan binds
    /// nothing past an image's layers (`VUID-VkSparseImageMemoryBindInfo-subresource-01723`) and
    /// the address is not derivable from what is measured.
    pub(crate) fn clipped_region(&self, level: u64, slice: u64, region: &Region) -> Vec<MetalUnit> {
        let Some(clipped) = self.clipped_level(level, region) else {
            return Vec::new();
        };
        self.update_units(
            level as u32,
            u32::try_from(slice).unwrap_or(u32::MAX),
            axes_of(region.origin),
            clipped.map(|axis| axis as u32),
        )
    }

    /// Whether `unit` is of a slice past the texture's own: one Metal's heap counts and nothing
    /// holds ([`clipped_region`](Self::clipped_region)).
    pub(crate) fn past_slices(&self, unit: MetalUnit) -> bool {
        let (MetalUnit::Tile { slice, .. } | MetalUnit::Tail { slice }) = unit;
        slice >= self.slices()
    }

    /// The tiles of `region` of `level` and `slice` that lie inside this grid, on each axis, or
    /// `None` where the region names no unit of it at all.
    ///
    /// The one place [`clipped_region`](Self::clipped_region)'s rule is written down, so the move
    /// below reads the same bounds the three mapping selectors do, the slice's too (they read
    /// [`clipped_level`](Self::clipped_level) on a slice past the texture's). A level at
    /// `firstMipmapInTail` answers one unit however large the region is — the slice's whole tail —
    /// which is what `[1, 1, 1]` stands for here; a level past the tail, a level or slice past the
    /// texture's own, an origin past the level's tiles and a size of 0 on an axis each answer
    /// `None`.
    fn clipped_size(&self, level: u64, slice: u64, region: &Region) -> Option<[u64; 3]> {
        if slice >= u64::from(self.slices()) {
            return None;
        }
        self.clipped_level(level, region)
    }

    /// [`clipped_size`](Self::clipped_size) on any slice.
    fn clipped_level(&self, level: u64, region: &Region) -> Option<[u64; 3]> {
        if level >= self.shape.levels {
            return None;
        }
        let level32 = level as u32;
        match level32.cmp(&self.first_tail()) {
            std::cmp::Ordering::Equal => return Some([1; 3]),
            std::cmp::Ordering::Greater => return None,
            std::cmp::Ordering::Less => {}
        }
        let tiles = self.tiles(level32).map(u64::from);
        let origin = [region.origin.x, region.origin.y, region.origin.z];
        let size = [region.size.width, region.size.height, region.size.depth];
        if (0..3).any(|axis| origin[axis] >= tiles[axis] || size[axis] == 0) {
            return None;
        }
        Some(std::array::from_fn(|axis| {
            size[axis].min(tiles[axis] - origin[axis])
        }))
    }

    /// The Metal units a move of `size` tiles at `source`'s origin of its level and slice of this
    /// grid, onto `into`'s origin of its level and slice of `destination`, moves — each source
    /// unit with the unit it moves onto.
    ///
    /// **This is plain Metal's own rule, measured off the texels either side of the move** on an
    /// Apple M3 Max under macOS 27.0.1
    /// (`metal2vulkan-conformance/probes/sparse_move_past_the_bounds.m`, one shape per process,
    /// each tile's verdict read off all 4,096 of its texels). A move changes no `usedSize`, so a
    /// counter cannot see any of it; eight shapes settle it and one rule answers all eight:
    ///
    /// - **the source region is clipped** to the level's own tiles, on each axis, and its
    ///   surviving tiles move in Metal's order (x, then y, then z). The part outside moves nothing
    ///   anywhere: it does not wrap into another row and it takes no destination tile. A level or
    ///   slice past the texture's own, an origin past the level's tiles and a size of 0 on an axis
    ///   each move nothing at all ([`clipped_size`](Self::clipped_size)'s `None`).
    /// - **the destination is a run of tiles, not a region.** It starts at `into`'s origin and is
    ///   laid out `size.width` tiles to a row — the width **as stated**, clipped only by the tiles
    ///   the level has to the right of the origin — taking the next row when one fills, one
    ///   destination per surviving source tile. So the destination's width is clipped and its
    ///   height is **not**: a destination region that runs past the level's right edge does not
    ///   lose those tiles, it continues on the rows below.
    ///
    /// Where the stated destination region fits inside the level, `size.width` is the whole of it
    /// and the run is exactly the region, so the pairing is the tile's own offset within it —
    /// which is what an ordinary move looks like, and is why the run shows only past an edge.
    ///
    /// Two things here are **not** measured and take the conservative form. A destination run that
    /// would pass the level's **last row** drops the tiles past it, rather than wrapping on to
    /// somewhere nobody has read. And a region deeper than one image pairs per image, the run
    /// restarting at the destination's own row and column for each: every measured shape is one
    /// image deep, and `size.width` tiles to a row is the one part of the rule a 3D move could
    /// read differently.
    ///
    /// At `firstMipmapInTail` on either side the units are one each — the slice's whole tail — so
    /// a tail moves onto the other side's tail or onto one tile of a tiled level, and one tile
    /// onto a whole tail, which is what Metal's validation takes and Metal moves.
    pub(crate) fn move_pairs(
        &self,
        source: MoveSide,
        size: Size,
        destination: &UnitGrid,
        into: MoveSide,
    ) -> Vec<(MetalUnit, MetalUnit)> {
        let from = Region {
            origin: source.origin,
            size,
        };
        let to = Region {
            origin: into.origin,
            size,
        };
        let (Some(held), Some(takes)) = (
            self.clipped_size(source.level, source.slice, &from),
            destination.clipped_size(into.level, into.slice, &to),
        ) else {
            return Vec::new();
        };
        let (level, slice) = (into.level as u32, into.slice as u32);
        let sources = self.update_units(
            source.level as u32,
            source.slice as u32,
            axes_of(source.origin),
            held.map(|axis| axis as u32),
        );
        // A tail on either side is one unit however many tiles the region names, so the two lists
        // are one each and the pairing is the zip of them.
        if source.level as u32 >= self.first_tail() || level >= destination.first_tail() {
            let targets =
                destination.update_units(level, slice, axes_of(into.origin), takes.map(|_| 1));
            return sources.into_iter().zip(targets).collect();
        }
        let tiles = destination.tiles(level).map(u64::from);
        // The run's row: the width as the move states it, clipped by the tiles to the right of the
        // origin. The stated width and not the clipped source's, which is what the measurement
        // pins down -- a source of 4 by 4 tiles clipped to 2 by 2 onto an origin of (1, 1) put its
        // third tile at (3, 1), which is a row of three and not of two.
        let per_row = size.width.min(tiles[0] - into.origin.x).max(1);
        let image = held[0].max(1) * held[1].max(1);
        let mut pairs = Vec::new();
        for (at, unit) in sources.into_iter().enumerate() {
            let at = at as u64;
            // Where the tile lies in its own image of the source region, and which image it is:
            // the run restarts at the destination's origin for each image, which is the part of
            // the rule no measured shape reaches.
            let (within, slab) = (at % image, at / image);
            let (x, y) = (
                into.origin.x + within % per_row,
                into.origin.y + within / per_row,
            );
            let z = into.origin.z + slab;
            // Past the level's last row or image the destination is unmeasured, so nothing moves
            // there rather than moving somewhere nobody has read.
            if y >= tiles[1] || z >= tiles[2].max(1) {
                continue;
            }
            pairs.push((
                unit,
                MetalUnit::Tile {
                    level,
                    slice,
                    at: [x as u32, y as u32, z as u32],
                },
            ));
        }
        pairs
    }

    /// The bytes of memory `unit` binds.
    pub(crate) fn unit_bytes(&self, unit: VkUnit) -> u64 {
        match unit {
            VkUnit::Block { .. } => self.layout.block_bytes,
            VkUnit::Tail { .. } => self
                .layout
                .mip_tail_size
                .next_multiple_of(self.layout.block_bytes),
        }
    }

    /// A block's texels: its offset and extent in its level, clipped at the level's edge.
    pub(crate) fn block_texels(&self, level: u32, at: [u32; 3]) -> ([u32; 3], [u32; 3]) {
        let extent = self.level_extent(level);
        let granularity = self.layout.granularity;
        let offset = [0, 1, 2].map(|axis| at[axis].saturating_mul(granularity[axis]));
        let size =
            [0, 1, 2].map(|axis| granularity[axis].min(extent[axis].saturating_sub(offset[axis])));
        (offset, size)
    }
}

/// One side of a move: which level, slice and origin of one texture's tile grid it names.
///
/// The two sides share the move's one `sourceSize`, as Metal's selector does, so the size is not
/// here: it is [`UnitGrid::move_pairs`]'s own argument.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MoveSide {
    pub(crate) level: u64,
    pub(crate) slice: u64,
    pub(crate) origin: Origin,
}

/// The units of one mapping update, as encoded: what the scheduler applies in order.
#[derive(Clone, Debug)]
pub(crate) struct UnitUpdate {
    pub(crate) mode: TileMapping,
    pub(crate) units: Vec<MetalUnit>,
}
