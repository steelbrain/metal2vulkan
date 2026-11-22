//! A mipmapped texture's literal: `mip_levels` levels, every one authored.
//!
//! A texture that declares `mip_levels` above 1 has that many levels on both sides: the Metal
//! oracle makes it with that `mipmapLevelCount` and writes each level with
//! `replaceRegion:mipmapLevel:slice:`, and the engine's candidate makes its runtime texture
//! with the same level count and writes each level with `replace_region` at that level. Its bytes
//! are every level's in order, level 0 first, each laid out as a one-level texture of that
//! level's extent ([`level_dimensions`]): its slices in order, each slice's images and rows
//! tightly packed. Each axis Metal halves (the width; a 2D, 2D array or cube texture's height; a
//! 3D texture's height and depth) is the level 0 extent shifted right by the level, at least 1;
//! the layers are every level's.
//!
//! The checker accepts levels only where each executor makes them exactly: a single-sample 2D, 2D
//! array, 3D, cube, or cube array texture that is an input (an executor writes it and reads none
//! back), not sparse (a sparse literal has one level, `sparse::LITERAL_LEVELS`), with at most
//! [`full_chain`] levels. One level, the default, is the field's only spelling for it, so a case
//! written before the field keeps its identity.

use super::{ResourceRole, TextureResource, TextureType};

/// The extent, in the case's spelling (`[width, height or layers, depth or layers]`), of level
/// `level` of a texture of `texture_type` whose level 0 is `dimensions`.
pub fn level_dimensions(texture_type: TextureType, dimensions: [u32; 3], level: u32) -> [u32; 3] {
    let half = |extent: u32| extent.checked_shr(level).unwrap_or(0).max(1);
    let [width, second, third] = dimensions;
    match texture_type {
        TextureType::Buffer | TextureType::D1 | TextureType::D1Array => {
            [half(width), second, third]
        }
        TextureType::D2
        | TextureType::D2Array
        | TextureType::D2Multisample
        | TextureType::D2MultisampleArray
        | TextureType::Cube
        | TextureType::CubeArray => [half(width), half(second), third],
        TextureType::D3 => [half(width), half(second), half(third)],
    }
}

/// The most levels a texture of `texture_type` and level 0 extent `dimensions` has: Metal's full
/// chain, `floor(log2(largest halved axis)) + 1`.
pub fn full_chain(texture_type: TextureType, dimensions: [u32; 3]) -> u32 {
    let [width, second, third] = dimensions;
    let largest = match texture_type {
        TextureType::Buffer | TextureType::D1 | TextureType::D1Array => width,
        TextureType::D3 => width.max(second).max(third),
        _ => width.max(second),
    };
    u32::BITS - largest.max(1).leading_zeros()
}

/// Whether the case model makes a texture of `texture_type` with more than one level.
pub fn mipmappable(texture_type: TextureType) -> bool {
    matches!(
        texture_type,
        TextureType::D2
            | TextureType::D2Array
            | TextureType::D3
            | TextureType::Cube
            | TextureType::CubeArray
    )
}

/// Check `texture`'s level count.
pub(super) fn validate(texture: &TextureResource, errors: &mut Vec<String>) {
    let levels = texture.mip_levels;
    if levels == 1 {
        return;
    }
    let label = format!("texture binding {}", texture.binding);
    if levels == 0 {
        errors.push(format!("{label}: mip_levels is at least 1"));
        return;
    }
    if !mipmappable(texture.texture_type) {
        errors.push(format!(
            "{label}: a mipmapped texture is 2D, 2D array, 3D, cube, or cube array (the case \
             model makes no other type with levels), not {:?}",
            texture.texture_type
        ));
    }
    if texture.sample_count != 1 {
        errors.push(format!("{label}: a multisampled texture has one level"));
    }
    if texture.sparse.is_some() {
        errors.push(format!("{label}: a sparse texture literal has one level"));
    }
    if texture.role != ResourceRole::Input {
        errors.push(format!(
            "{label}: a mipmapped texture is an input (no executor reads a level back)"
        ));
    }
    let most = full_chain(texture.texture_type, texture.dimensions);
    if levels > most {
        errors.push(format!(
            "{label}: {levels} levels, but a texture of dimensions {:?} has at most {most}",
            texture.dimensions
        ));
    }
}

/// The byte length of a texture literal of `levels` levels: each level's extent times
/// `bytes_per_texel`, summed; `None` on overflow.
pub fn literal_len(
    texture_type: TextureType,
    dimensions: [u32; 3],
    levels: u32,
    bytes_per_texel: usize,
) -> Option<usize> {
    (0..levels).try_fold(0usize, |total, level| {
        let level = level_dimensions(texture_type, dimensions, level)
            .iter()
            .try_fold(bytes_per_texel, |size, extent| {
                size.checked_mul(*extent as usize)
            })?;
        total.checked_add(level)
    })
}
