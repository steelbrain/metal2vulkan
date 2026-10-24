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
use crate::values::Region;
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

/// What a texture has mapped and bound: Metal's mapped units, and the memory of each bound
/// device unit.
#[derive(Default)]
pub(crate) struct SparseState {
    pub(crate) mapped: HashSet<MetalUnit>,
    pub(crate) bound: HashMap<VkUnit, crate::memory::Placement>,
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
    /// maps arguments its validation stops (the M3 Max under macOS 27.0.1): none at a level or
    /// slice past the texture's, nor for a tiled level's region whose origin is past its tiles or
    /// whose size is 0 on an axis; else the region's tiles inside the level, the rest of it
    /// dropped. A level past `firstMipmapInTail` touches its slice's tail.
    ///
    /// One rule for all three mapping selectors — the direct update, the plural one and the
    /// indirect one — and for the move. Plain Metal **takes** each of these arguments where its own
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
    /// Differs: Metal's `usedSize` over a region of a **slice** past the texture's varies — a tile
    /// more beside other regions, and an unmap of it can take the count below zero, and the direct
    /// selector alone took one page for it in the measurement above — where the runtime maps and
    /// counts none. Which tiles a move past these bounds moves is not measured: `usedSize` cannot
    /// show it and the probe reads no texels.
    pub(crate) fn clipped_region(&self, level: u64, slice: u64, region: &Region) -> Vec<MetalUnit> {
        if level >= self.shape.levels || slice >= u64::from(self.slices()) {
            return Vec::new();
        }
        let (level32, slice32) = (level as u32, slice as u32);
        if level32 >= self.first_tail() {
            return self.update_units(level32, slice32, [0; 3], [0; 3]);
        }
        let tiles = self.tiles(level32).map(u64::from);
        let origin = [region.origin.x, region.origin.y, region.origin.z];
        let size = [region.size.width, region.size.height, region.size.depth];
        if (0..3).any(|axis| origin[axis] >= tiles[axis] || size[axis] == 0) {
            return Vec::new();
        }
        let clipped: [u64; 3] =
            std::array::from_fn(|axis| size[axis].min(tiles[axis] - origin[axis]));
        self.update_units(
            level32,
            slice32,
            origin.map(|axis| axis as u32),
            clipped.map(|axis| axis as u32),
        )
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

/// The units of one mapping update, as encoded: what the scheduler applies in order.
#[derive(Clone, Debug)]
pub(crate) struct UnitUpdate {
    pub(crate) mode: TileMapping,
    pub(crate) units: Vec<MetalUnit>,
}
