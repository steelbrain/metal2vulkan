//! What the Metal oracle makes of a sparse texture's mappings.
//!
//! A case names its mapped texels ([`crate::case::sparse`]); the device divides the texture into
//! its own tiles, Metal's sparse tiles (`sparseTileSizeWithTextureType:...` at the heap's page
//! size), with a first level kept in a mip tail (`firstMipmapInTail`). The oracle makes each
//! mapping through [`device_mappings`] under its [`SparseTiling`], so a region that is not a whole
//! number of the device's tiles, or that covers part of a level in its mip tail, is refused before
//! anything runs. A case's mip tail is its slice's one level whole (`case::sparse`): a device that
//! keeps that level in its tail maps the tail, and one that does not maps the level's tiles; a
//! region covering a whole level in the tail maps the tail. On a device that makes every mapping,
//! the mapped texels are exactly the regions and tails the case names ([`mapped_texels`]), which is
//! what the oracle uploads ([`literal_offset`]). The engine's candidate runs no sparse texture
//! (`ToolingRequirement::SparseTexture`).

use crate::case::sparse::{sparse_extent, SparseMapping, LITERAL_LEVELS};
use crate::literal::LiteralTexture;

// A case's mip tail is its slice's one level whole (`case::sparse`): a literal with more levels
// needs the tail stated in texels.
const _: () = assert!(LITERAL_LEVELS == 1);

/// How one device divides a sparse texture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SparseTiling {
    /// A tile's extent in texels (x, y, z): Metal's sparse tile.
    pub tile: [u32; 3],
    /// The first mipmap level in the mip tail, or the texture's level count when it has no tail
    /// (Metal's `firstMipmapInTail`).
    pub first_level_in_tail: u32,
}

/// Texels `origin .. origin + size` of one level and slice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TexelRegion {
    pub level: u32,
    pub slice: u32,
    pub origin: [u32; 3],
    pub size: [u32; 3],
}

/// One mapping of a sparse texture as a device makes it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceMapping {
    /// A region of whole tiles: its texels, and the same region in tiles (Metal maps a region in
    /// tiles; Vulkan binds its texels, each block a whole one or one cut by the level's edge).
    Tiles {
        texels: TexelRegion,
        tile_origin: [u32; 3],
        tile_count: [u32; 3],
    },
    /// The mip tail of slice `slice`, which the device maps as one unit.
    MipTail { slice: u32 },
}

/// The extent of level `level` of a texture whose level 0 is `extent`.
pub fn level_extent(extent: [u32; 3], level: u32) -> [u32; 3] {
    extent.map(|axis| axis.checked_shr(level).unwrap_or(0).max(1))
}

/// `texture`'s level-0 extent and slice count, or why it is not a sparse texture of the case
/// model.
fn extent_of(texture: &LiteralTexture) -> Result<([u32; 3], u32), String> {
    sparse_extent(texture.texture_type, texture.dimensions).ok_or_else(|| {
        format!(
            "texture {} is a {:?} texture, which the case model does not make sparse",
            texture.binding, texture.texture_type
        )
    })
}

/// `texture`'s mappings as a device with `tiling` makes them, in the case's order, or why that
/// device cannot make one of them: a region that covers part of a level in the mip tail, or that
/// is not a whole number of tiles (its origin a multiple of the tile, its end a multiple of it or
/// the level's edge). A mip tail, and a region covering a whole level the device keeps in its
/// tail, map as the tail where the device keeps level 0 there (the tail is then exactly that one
/// level: the case model's textures have one), and a mip tail as level 0's tiles where it does
/// not. `Ok(None)` for a texture that is not sparse.
pub fn device_mappings(
    texture: &LiteralTexture,
    tiling: SparseTiling,
) -> Result<Option<Vec<DeviceMapping>>, String> {
    let Some(sparse) = &texture.sparse else {
        return Ok(None);
    };
    if tiling.tile.contains(&0) {
        return Err(format!(
            "texture {}: the device's sparse tile {:?} is empty",
            texture.binding, tiling.tile
        ));
    }
    let (extent, _) = extent_of(texture)?;
    sparse
        .mappings
        .iter()
        .enumerate()
        .map(|(index, mapping)| {
            let at = format!("texture {} mapping {index}", texture.binding);
            match *mapping {
                SparseMapping::Region {
                    level,
                    slice,
                    origin,
                    size,
                } => {
                    let edge = level_extent(extent, level);
                    if level >= tiling.first_level_in_tail {
                        // The tail is the levels from its first; a literal's is its one level.
                        let whole = origin == [0; 3] && size == edge;
                        let alone =
                            level == tiling.first_level_in_tail && level + 1 == LITERAL_LEVELS;
                        if whole && alone {
                            return Ok(DeviceMapping::MipTail { slice });
                        }
                        return Err(format!(
                            "{at}: texels {origin:?} + {size:?} of level {level} are part of the \
                             device's mip tail (from level {}), which maps only as a whole",
                            tiling.first_level_in_tail
                        ));
                    }
                    let mut tile_origin = [0; 3];
                    let mut tile_count = [0; 3];
                    for axis in 0..3 {
                        let tile = tiling.tile[axis];
                        let end = u64::from(origin[axis]) + u64::from(size[axis]);
                        let whole = end % u64::from(tile) == 0 || end == u64::from(edge[axis]);
                        if size[axis] == 0 || end > u64::from(edge[axis]) {
                            return Err(format!(
                                "{at}: texels {origin:?} + {size:?} are not a region of level \
                                 {level}'s {edge:?}"
                            ));
                        }
                        if origin[axis] % tile != 0 || !whole {
                            return Err(format!(
                                "{at}: texels {origin:?} + {size:?} of level {level} are not \
                                 whole {:?}-texel tiles of the device (each axis must start on a \
                                 tile and end on one or at the level's edge {edge:?})",
                                tiling.tile
                            ));
                        }
                        tile_origin[axis] = origin[axis] / tile;
                        tile_count[axis] = size[axis].div_ceil(tile);
                    }
                    Ok(DeviceMapping::Tiles {
                        texels: TexelRegion {
                            level,
                            slice,
                            origin,
                            size,
                        },
                        tile_origin,
                        tile_count,
                    })
                }
                // The case's tail is its one level whole: the device's tail where it keeps that
                // level in one, else the level's tiles.
                SparseMapping::MipTail { slice } if tiling.first_level_in_tail == 0 => {
                    Ok(DeviceMapping::MipTail { slice })
                }
                SparseMapping::MipTail { slice } => {
                    let size = level_extent(extent, 0);
                    let mut tile_count = [0; 3];
                    for axis in 0..3 {
                        tile_count[axis] = size[axis].div_ceil(tiling.tile[axis]);
                    }
                    Ok(DeviceMapping::Tiles {
                        texels: TexelRegion {
                            level: 0,
                            slice,
                            origin: [0; 3],
                            size,
                        },
                        tile_origin: [0; 3],
                        tile_count,
                    })
                }
            }
        })
        .collect::<Result<Vec<_>, String>>()
        .map(Some)
}

/// The texels `mappings` (of `texture`, made under `tiling`) map, each one level of one slice: a
/// region's own texels, and each level of a slice's mip tail whole. What each executor uploads
/// of the authored bytes, in mapping order.
pub fn mapped_texels(
    texture: &LiteralTexture,
    tiling: SparseTiling,
    mappings: &[DeviceMapping],
) -> Result<Vec<TexelRegion>, String> {
    let (extent, _) = extent_of(texture)?;
    let mut regions = Vec::new();
    for mapping in mappings {
        match *mapping {
            DeviceMapping::Tiles { texels, .. } => regions.push(texels),
            DeviceMapping::MipTail { slice } => {
                regions.extend((tiling.first_level_in_tail..LITERAL_LEVELS).map(|level| {
                    TexelRegion {
                        level,
                        slice,
                        origin: [0; 3],
                        size: level_extent(extent, level),
                    }
                }));
            }
        }
    }
    Ok(regions)
}

/// The byte offset in `texture`'s authored bytes of `region`'s first texel: the literal packs
/// level 0's slices one after another, each its planes (z), each its rows. A copy of the region
/// reads rows of the literal's width and images of its height from there.
pub fn literal_offset(texture: &LiteralTexture, region: &TexelRegion) -> Result<u64, String> {
    let (extent, _) = extent_of(texture)?;
    if region.level != 0 {
        return Err(format!(
            "texture {}: the literal holds level 0 only, not level {}",
            texture.binding, region.level
        ));
    }
    if region.size.contains(&0) {
        return Err(format!(
            "texture {}: an empty region {region:?}",
            texture.binding
        ));
    }
    let texel = texture.format.bytes_per_pixel() as u64;
    let [width, height, depth] = extent.map(u64::from);
    let [x, y, z] = region.origin.map(u64::from);
    let offset = ((u64::from(region.slice) * depth + z) * height + y) * width * texel + x * texel;
    let [w, h, d] = region.size.map(u64::from);
    let last = ((u64::from(region.slice) * depth + z + d - 1) * height + y + h - 1) * width * texel
        + (x + w) * texel;
    if last > texture.bytes.len() as u64 {
        return Err(format!(
            "texture {}: region {region:?} reaches byte {last} of {} authored bytes",
            texture.binding,
            texture.bytes.len()
        ));
    }
    Ok(offset)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::case::sparse::SparseTextureResource;
    use crate::case::{ResourceRole, TextureFormat, TextureType};

    fn sparse(
        texture_type: TextureType,
        dimensions: [u32; 3],
        mappings: Vec<SparseMapping>,
    ) -> LiteralTexture {
        let layout = crate::literal::texture_layout(texture_type, dimensions, 1).unwrap();
        let texels = layout.width * layout.height * layout.depth * layout.array_layers;
        LiteralTexture {
            binding: 2,
            role: ResourceRole::Input,
            texture_type,
            format: TextureFormat::R32Float,
            dimensions,
            sample_count: 1,
            bytes: vec![0; 4 * texels as usize],
            sparse: Some(SparseTextureResource { mappings }),
            mip_levels: 1,
        }
    }

    fn region(slice: u32, origin: [u32; 3], size: [u32; 3]) -> SparseMapping {
        SparseMapping::Region {
            level: 0,
            slice,
            origin,
            size,
        }
    }

    const NO_TAIL: u32 = LITERAL_LEVELS;

    /// A region of whole tiles maps as those tiles on any device whose tile divides it; the last
    /// tile of an axis may be cut by the level's edge, and the same region is refused where it is
    /// not whole tiles (a 128-texel block cuts the 64-texel tiles a 16 KiB page makes).
    #[test]
    fn a_region_maps_as_whole_tiles_or_is_refused() {
        let left = sparse(
            TextureType::D2,
            [256, 100, 1],
            vec![region(0, [0, 0, 0], [128, 100, 1])],
        );
        let metal = SparseTiling {
            tile: [64, 64, 1],
            first_level_in_tail: NO_TAIL,
        };
        assert_eq!(
            device_mappings(&left, metal).unwrap().unwrap(),
            [DeviceMapping::Tiles {
                texels: TexelRegion {
                    level: 0,
                    slice: 0,
                    origin: [0, 0, 0],
                    size: [128, 100, 1],
                },
                tile_origin: [0, 0, 0],
                tile_count: [2, 2, 1],
            }]
        );
        let vulkan = SparseTiling {
            tile: [128, 128, 1],
            first_level_in_tail: NO_TAIL,
        };
        assert!(device_mappings(&left, vulkan).unwrap().is_some());
        let quarter = sparse(
            TextureType::D2,
            [256, 128, 1],
            vec![region(0, [64, 0, 0], [64, 64, 1])],
        );
        assert!(device_mappings(&quarter, metal).unwrap().is_some());
        let error = device_mappings(&quarter, vulkan).unwrap_err();
        assert!(
            error.contains("not whole [128, 128, 1]-texel tiles"),
            "{error}"
        );
        // A region that ends short of a tile inside the level is not whole tiles either.
        let short = sparse(
            TextureType::D2,
            [256, 128, 1],
            vec![region(0, [0, 0, 0], [100, 128, 1])],
        );
        assert!(device_mappings(&short, metal).is_err());
    }

    /// A case's mip tail is its slice's one level whole on every device: where the device keeps
    /// that level in its tail (Metal's 16 KiB page for a 16x16 texture), it maps the tail; where it
    /// keeps no tail (lavapipe's 128x128 blocks), it maps the level's one block, cut by the level's
    /// edge. The same texels either way. A region covering the whole level maps the tail too, and
    /// one covering part of it is refused there.
    #[test]
    fn a_mip_tail_is_the_whole_level_on_every_device() {
        let small = sparse(
            TextureType::D2Array,
            [16, 16, 2],
            vec![SparseMapping::MipTail { slice: 1 }],
        );
        let tailed = SparseTiling {
            tile: [64, 64, 1],
            first_level_in_tail: 0,
        };
        let untailed = SparseTiling {
            tile: [128, 128, 1],
            first_level_in_tail: NO_TAIL,
        };
        let level = TexelRegion {
            level: 0,
            slice: 1,
            origin: [0; 3],
            size: [16, 16, 1],
        };
        let mappings = device_mappings(&small, tailed).unwrap().unwrap();
        assert_eq!(mappings, [DeviceMapping::MipTail { slice: 1 }]);
        assert_eq!(mapped_texels(&small, tailed, &mappings).unwrap(), [level]);
        let mappings = device_mappings(&small, untailed).unwrap().unwrap();
        assert_eq!(
            mappings,
            [DeviceMapping::Tiles {
                texels: level,
                tile_origin: [0; 3],
                tile_count: [1, 1, 1],
            }]
        );
        assert_eq!(mapped_texels(&small, untailed, &mappings).unwrap(), [level]);
        let whole = sparse(
            TextureType::D2Array,
            [16, 16, 2],
            vec![region(1, [0, 0, 0], [16, 16, 1])],
        );
        assert_eq!(
            device_mappings(&whole, tailed).unwrap().unwrap(),
            [DeviceMapping::MipTail { slice: 1 }]
        );
        assert!(matches!(
            device_mappings(&whole, untailed).unwrap().unwrap()[..],
            [DeviceMapping::Tiles { .. }]
        ));
        let part = sparse(
            TextureType::D2,
            [16, 16, 1],
            vec![region(0, [0, 0, 0], [16, 8, 1])],
        );
        let error = device_mappings(&part, tailed).unwrap_err();
        assert!(error.contains("part of the device's mip tail"), "{error}");
        assert!(device_mappings(&part, untailed).is_err());
    }

    /// The upload reads each region from the literal's packing: slices, then planes, then rows.
    #[test]
    fn a_region_is_copied_from_the_literals_packing() {
        let array = sparse(TextureType::D2Array, [4, 3, 2], vec![]);
        let at = |slice, origin| TexelRegion {
            level: 0,
            slice,
            origin,
            size: [1, 1, 1],
        };
        assert_eq!(literal_offset(&array, &at(0, [0, 0, 0])), Ok(0));
        assert_eq!(
            literal_offset(&array, &at(0, [1, 2, 0])),
            Ok(4 * (2 * 4 + 1))
        );
        assert_eq!(
            literal_offset(&array, &at(1, [3, 2, 0])),
            Ok(4 * (12 + 2 * 4 + 3))
        );
        let volume = sparse(TextureType::D3, [4, 3, 2], vec![]);
        assert_eq!(literal_offset(&volume, &at(0, [0, 1, 1])), Ok(4 * (12 + 4)));
        let past = TexelRegion {
            level: 0,
            slice: 1,
            origin: [0, 0, 0],
            size: [4, 3, 1],
        };
        assert!(literal_offset(&volume, &past).is_err());
        // An ordinary texture has no mappings.
        let mut ordinary = sparse(TextureType::D2, [4, 4, 1], vec![]);
        ordinary.sparse = None;
        assert_eq!(
            device_mappings(
                &ordinary,
                SparseTiling {
                    tile: [1, 1, 1],
                    first_level_in_tail: NO_TAIL
                }
            ),
            Ok(None)
        );
    }
}
