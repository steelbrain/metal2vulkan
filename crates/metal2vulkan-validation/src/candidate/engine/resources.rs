//! Textures and samplers of an engine case in the binder's terms.
//!
//! The executor binds exactly what the Metal oracle binds: each authored texture is a Metal
//! texture of the same type, pixel format, extent, and sample count (one mip level), a texture
//! buffer is a view of a buffer holding the authored bytes at offset 0 whose length is its width
//! in texels, and each authored sampler is `MTLSamplerDescriptor`'s defaults with the authored
//! address, filter, and coordinate modes. Element `k` of a texture array at `[[texture(n)]]` is
//! the texture bound at slot `n + k` ([`bound_textures`]): the Metal oracle binds it there, and the
//! product reads that element through that slot's own binding (a handle-array element is the slot
//! `n + k`, `memory::handles`), so the binder needs no descriptor array. Everything here is
//! device-free; the Vulkan objects are `platform::plan_resources`.

use crate::case::{
    ArgumentBufferSamplerResource, SamplerAddressMode, SamplerCompareFunction, SamplerFilter,
    SamplerMipFilter, SamplerResource, TextureFormat, TextureType,
};
use crate::literal::{LiteralTexture, LiteralTextureArray};
use metal2vulkan::bind::{BoundSampler, BoundTexture, SamplerDescriptor, TextureFacts};
use metal2vulkan::metal::{self, PixelFormat};
use std::collections::{BTreeMap, BTreeSet};

/// The Metal pixel format of an authored texture format.
pub fn pixel_format(format: TextureFormat) -> PixelFormat {
    use TextureFormat as F;
    match format {
        F::R8Unorm => PixelFormat::R8Unorm,
        F::Rgba8Unorm => PixelFormat::RGBA8Unorm,
        F::Rgba8Uint => PixelFormat::RGBA8Uint,
        F::Rgba8Sint => PixelFormat::RGBA8Sint,
        F::R16Float => PixelFormat::R16Float,
        F::R16Uint => PixelFormat::R16Uint,
        F::Rg16Float => PixelFormat::RG16Float,
        F::Rg32Float => PixelFormat::RG32Float,
        F::Rgba16Float => PixelFormat::RGBA16Float,
        F::Rgba16Uint => PixelFormat::RGBA16Uint,
        F::Rgba16Sint => PixelFormat::RGBA16Sint,
        F::R32Uint => PixelFormat::R32Uint,
        F::R32Sint => PixelFormat::R32Sint,
        F::R32Float => PixelFormat::R32Float,
        F::Rgba32Uint => PixelFormat::RGBA32Uint,
        F::Rgba32Sint => PixelFormat::RGBA32Sint,
        F::Rgba32Float => PixelFormat::RGBA32Float,
        F::Depth32Float => PixelFormat::Depth32Float,
        F::Rg32Uint => PixelFormat::RG32Uint,
        F::R8Snorm => PixelFormat::R8Snorm,
        F::Rgba8Snorm => PixelFormat::RGBA8Snorm,
        F::R16Unorm => PixelFormat::R16Unorm,
        F::Rgba16Unorm => PixelFormat::RGBA16Unorm,
        F::R16Snorm => PixelFormat::R16Snorm,
        F::Rg16Snorm => PixelFormat::RG16Snorm,
        F::Rgba16Snorm => PixelFormat::RGBA16Snorm,
        F::Rgba8UnormSrgb => PixelFormat::RGBA8UnormSRGB,
        F::Bgra8Unorm => PixelFormat::BGRA8Unorm,
        F::Bgra8UnormSrgb => PixelFormat::BGRA8UnormSRGB,
        F::R8UnormSrgb => PixelFormat::R8UnormSRGB,
        F::Rg8Unorm => PixelFormat::RG8Unorm,
        F::Rg8UnormSrgb => PixelFormat::RG8UnormSRGB,
        F::Rg8Snorm => PixelFormat::RG8Snorm,
        F::Rg16Unorm => PixelFormat::RG16Unorm,
        F::A8Unorm => PixelFormat::A8Unorm,
        F::B5g6r5Unorm => PixelFormat::B5G6R5Unorm,
        F::A1bgr5Unorm => PixelFormat::A1BGR5Unorm,
        F::Abgr4Unorm => PixelFormat::ABGR4Unorm,
        F::Bgr5a1Unorm => PixelFormat::BGR5A1Unorm,
        F::Rgb10a2Unorm => PixelFormat::RGB10A2Unorm,
        F::Bgr10a2Unorm => PixelFormat::BGR10A2Unorm,
        F::Rg11b10Float => PixelFormat::RG11B10Float,
        F::Rgb9e5Float => PixelFormat::RGB9E5Float,
    }
}

/// The Metal texture type of an authored texture type.
pub fn texture_type(texture_type: TextureType) -> metal::TextureType {
    use metal::TextureType as M;
    match texture_type {
        TextureType::Buffer => M::TextureBuffer,
        TextureType::D1 => M::Type1D,
        TextureType::D1Array => M::Type1DArray,
        TextureType::D2 => M::Type2D,
        TextureType::D2Array => M::Type2DArray,
        TextureType::D2Multisample => M::Type2DMultisample,
        TextureType::D2MultisampleArray => M::Type2DMultisampleArray,
        TextureType::D3 => M::Type3D,
        TextureType::Cube => M::Cube,
        TextureType::CubeArray => M::CubeArray,
    }
}

/// The Metal texture slot of element `element` of the texture array at `binding`: `binding +
/// element`, or `None` when it overflows.
pub fn texture_array_slot(binding: u32, element: u32) -> Option<u32> {
    binding.checked_add(element)
}

/// Each texture a function binds, by Metal texture slot: each of `textures` at its binding, then
/// each element of each of `arrays` at its [`texture_array_slot`], in the order the Metal oracle
/// binds them (`setTexture:atIndex:`, textures first). Metal's later bind wins, so element 0 of
/// an array that declares `overrides_texture_at_base` replaces the texture at its base slot (the
/// function-constant alternative the case checker admits there); any other slot bound twice is
/// refused (the checker refuses each first). An element is its array's type, format, and sample
/// count with its own extent and contents, at its slot.
pub fn bound_textures(
    textures: &[LiteralTexture],
    arrays: &[LiteralTextureArray],
) -> Result<BTreeMap<u32, LiteralTexture>, String> {
    let mut bound = BTreeMap::new();
    for texture in textures {
        if bound.insert(texture.binding, texture.clone()).is_some() {
            return Err(format!("texture {} bound twice", texture.binding));
        }
    }
    let mut elements = BTreeSet::new();
    for array in arrays {
        for (element, texture) in array.elements.iter().enumerate() {
            let slot = u32::try_from(element)
                .ok()
                .and_then(|element| texture_array_slot(array.binding, element))
                .ok_or_else(|| {
                    format!(
                        "texture array {} element {element}'s slot overflows",
                        array.binding
                    )
                })?;
            let replaces = element == 0 && array.overrides_texture_at_base;
            let mut texture = texture.clone();
            texture.binding = slot;
            let displaced = bound.insert(slot, texture).is_some();
            if !elements.insert(slot) || (displaced && !replaces) {
                return Err(format!(
                    "texture slot {slot} is bound twice (texture array {} element {element})",
                    array.binding
                ));
            }
        }
    }
    Ok(bound)
}

/// The bytes a texture buffer's view covers: its width in texels (the Metal oracle's
/// `newTextureWithDescriptor:offset:bytesPerRow:` view of the authored bytes at offset 0).
pub fn texel_buffer_length(texture: &LiteralTexture) -> Result<u64, String> {
    let length = u64::from(texture.dimensions[0])
        .checked_mul(texture.format.bytes_per_pixel() as u64)
        .ok_or_else(|| format!("texture buffer {} length overflows", texture.binding))?;
    if length == 0 || length > texture.bytes.len() as u64 {
        return Err(format!(
            "texture buffer {} is {length} bytes wide but authors {} bytes",
            texture.binding,
            texture.bytes.len()
        ));
    }
    Ok(length)
}

/// What the binder sees bound for an authored texture: a texture buffer's view (offset 0, its
/// width in texels), or an image texture's facts (Metal's `arrayLength` counts cubes for a cube
/// array, its levels, shader read and write usage, and `isSparse` for a sparse texture).
pub fn bound_texture(texture: &LiteralTexture) -> Result<BoundTexture, String> {
    let pixel_format = pixel_format(texture.format);
    if texture.texture_type == TextureType::Buffer {
        let length = texel_buffer_length(texture)?;
        return Ok(BoundTexture::Buffer {
            pixel_format,
            offset: 0,
            length,
            bytes_per_row: length,
            resource_id: None,
            swizzle: metal2vulkan::metal::TextureSwizzleChannels::DEFAULT,
            parts: 1,
        });
    }
    let layout = texture.layout()?;
    let mut facts = TextureFacts::new(self::texture_type(texture.texture_type), pixel_format);
    facts.sample_count = layout.sample_count;
    facts.usage = metal::TextureUsage::SHADER_READ | metal::TextureUsage::SHADER_WRITE;
    facts.array_length = match texture.texture_type {
        TextureType::D1Array | TextureType::D2Array | TextureType::D2MultisampleArray => {
            layout.array_layers
        }
        TextureType::CubeArray => layout.array_layers / 6,
        _ => 1,
    };
    facts.sparse = texture.sparse.is_some();
    facts.mipmap_level_count = texture.mip_levels;
    Ok(BoundTexture::Texture {
        facts,
        resource_id: None,
    })
}

/// The `MTLSamplerDescriptor` the Metal oracle creates for `sampler`: the defaults, with the
/// authored address mode on every axis, the filters, and the coordinate mode.
pub fn sampler_descriptor(sampler: &SamplerResource) -> SamplerDescriptor {
    let address = match sampler.address_mode {
        SamplerAddressMode::ClampToEdge => metal::SamplerAddressMode::ClampToEdge,
        SamplerAddressMode::ClampToZero => metal::SamplerAddressMode::ClampToZero,
        SamplerAddressMode::Repeat => metal::SamplerAddressMode::Repeat,
        SamplerAddressMode::MirroredRepeat => metal::SamplerAddressMode::MirrorRepeat,
    };
    let filter = |filter| match filter {
        SamplerFilter::Nearest => metal::SamplerMinMagFilter::Nearest,
        SamplerFilter::Linear => metal::SamplerMinMagFilter::Linear,
    };
    let mut desc = SamplerDescriptor::new();
    desc.min_filter = filter(sampler.min_filter);
    desc.mag_filter = filter(sampler.mag_filter);
    desc.mip_filter = match sampler.mip_filter {
        SamplerMipFilter::NotMipmapped => metal::SamplerMipFilter::NotMipmapped,
        SamplerMipFilter::Nearest => metal::SamplerMipFilter::Nearest,
        SamplerMipFilter::Linear => metal::SamplerMipFilter::Linear,
    };
    desc.s_address_mode = address;
    desc.t_address_mode = address;
    desc.r_address_mode = address;
    desc.normalized_coordinates = sampler.normalized_coordinates;
    desc
}

/// The `MTLSamplerDescriptor` the Metal oracle creates for the argument-buffer sampler `sampler`:
/// [`sampler_descriptor`]'s of its modes, with its comparison function (absent is Metal's default,
/// `never`).
pub fn argument_sampler_descriptor(sampler: &ArgumentBufferSamplerResource) -> SamplerDescriptor {
    let mut desc = sampler_descriptor(&SamplerResource {
        binding: sampler.buffer_binding,
        address_mode: sampler.address_mode,
        min_filter: sampler.min_filter,
        mag_filter: sampler.mag_filter,
        mip_filter: sampler.mip_filter,
        normalized_coordinates: sampler.normalized_coordinates,
    });
    desc.compare_function = match sampler.compare_function {
        None | Some(SamplerCompareFunction::Never) => metal::CompareFunction::Never,
        Some(SamplerCompareFunction::Less) => metal::CompareFunction::Less,
        Some(SamplerCompareFunction::Equal) => metal::CompareFunction::Equal,
        Some(SamplerCompareFunction::LessEqual) => metal::CompareFunction::LessEqual,
        Some(SamplerCompareFunction::Greater) => metal::CompareFunction::Greater,
        Some(SamplerCompareFunction::NotEqual) => metal::CompareFunction::NotEqual,
        Some(SamplerCompareFunction::GreaterEqual) => metal::CompareFunction::GreaterEqual,
        Some(SamplerCompareFunction::Always) => metal::CompareFunction::Always,
    };
    desc
}

/// The bound sampler of an authored sampler.
pub fn bound_sampler(sampler: &SamplerResource) -> BoundSampler {
    BoundSampler::new(sampler_descriptor(sampler))
}

/// The observed bytes of a texture output: the `(origin, dimensions)` selection, in the case's
/// spelling, of `texture` whose `bytes` are its final contents packed like the authored literal
/// (slice-major rows). A multisampled texture has no such packing on either executor.
pub fn select_texture(
    texture: &LiteralTexture,
    origin: [u32; 3],
    dimensions: [u32; 3],
) -> Result<Vec<u8>, String> {
    if texture.sample_count != 1 {
        return Err(format!(
            "texture {} is multisampled; its samples are not an observation",
            texture.binding
        ));
    }
    let layout = texture.layout()?;
    // Metal reads the selection back through `subresource_span`; it must name a valid region of
    // this texture's type before the packed bytes are sliced.
    crate::literal::subresource_span(texture.texture_type, origin, dimensions, 1)?;
    let planes = if texture.texture_type == TextureType::D3 {
        layout.depth
    } else {
        layout.array_layers
    };
    let fits = |axis: usize, extent: u32| {
        origin[axis]
            .checked_add(dimensions[axis])
            .is_some_and(|end| end <= extent)
    };
    let rows = if texture.texture_type == TextureType::D1Array {
        layout.array_layers
    } else {
        layout.height
    };
    let third = if texture.texture_type == TextureType::D1Array {
        1
    } else {
        planes
    };
    if !(fits(0, layout.width) && fits(1, rows) && fits(2, third)) {
        return Err(format!(
            "output selection {origin:?}+{dimensions:?} exceeds texture {}",
            texture.binding
        ));
    }
    texture.select(origin, dimensions)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::case::ResourceRole;

    fn texture(texture_type: TextureType, dimensions: [u32; 3], bytes: Vec<u8>) -> LiteralTexture {
        LiteralTexture {
            binding: 3,
            role: ResourceRole::InOut,
            texture_type,
            format: TextureFormat::R8Unorm,
            dimensions,
            sample_count: 1,
            bytes,
            sparse: None,
            mip_levels: 1,
        }
    }

    #[test]
    fn authored_textures_bind_what_the_metal_oracle_creates() {
        match bound_texture(&texture(TextureType::CubeArray, [2, 2, 12], vec![0; 48])).unwrap() {
            BoundTexture::Texture { facts, .. } => {
                assert_eq!(facts.texture_type, metal::TextureType::CubeArray);
                assert_eq!(facts.pixel_format, PixelFormat::R8Unorm);
                assert_eq!((facts.array_length, facts.sample_count), (2, 1));
                assert_eq!(facts.mipmap_level_count, 1);
            }
            other => panic!("{other:?}"),
        }
        match bound_texture(&texture(TextureType::D1Array, [4, 3, 1], vec![0; 12])).unwrap() {
            BoundTexture::Texture { facts, .. } => assert_eq!(facts.array_length, 3),
            other => panic!("{other:?}"),
        }
        // A texture buffer's view is its width, not every authored byte.
        match bound_texture(&texture(TextureType::Buffer, [5, 1, 1], vec![0; 8])).unwrap() {
            BoundTexture::Buffer { offset, length, .. } => assert_eq!((offset, length), (0, 5)),
            other => panic!("{other:?}"),
        }
        assert!(bound_texture(&texture(TextureType::Buffer, [9, 1, 1], vec![0; 8])).is_err());
        assert!(bound_texture(&texture(TextureType::Cube, [2, 3, 6], vec![0; 36])).is_err());
    }

    #[test]
    fn samplers_are_metals_defaults_with_the_authored_modes() {
        let desc = sampler_descriptor(&SamplerResource {
            binding: 0,
            address_mode: SamplerAddressMode::MirroredRepeat,
            min_filter: SamplerFilter::Linear,
            mag_filter: SamplerFilter::Nearest,
            mip_filter: SamplerMipFilter::Linear,
            normalized_coordinates: false,
        });
        let mut expected = SamplerDescriptor::new();
        expected.min_filter = metal::SamplerMinMagFilter::Linear;
        expected.mip_filter = metal::SamplerMipFilter::Linear;
        expected.s_address_mode = metal::SamplerAddressMode::MirrorRepeat;
        expected.t_address_mode = metal::SamplerAddressMode::MirrorRepeat;
        expected.r_address_mode = metal::SamplerAddressMode::MirrorRepeat;
        expected.normalized_coordinates = false;
        assert_eq!(desc, expected);
    }

    /// Selections read the final contents in the literal's packing: layers of a 2D array by
    /// origin z, the layers of a 1D array by origin y, and nothing outside the texture.
    #[test]
    fn texture_outputs_select_the_final_contents() {
        let array = texture(TextureType::D2Array, [2, 2, 2], (0..8).collect());
        assert_eq!(select_texture(&array, [1, 0, 1], [1, 2, 1]), Ok(vec![5, 7]));
        assert!(select_texture(&array, [0, 0, 1], [2, 2, 2]).is_err());
        let rows = texture(TextureType::D1Array, [3, 2, 1], (0..6).collect());
        assert_eq!(
            select_texture(&rows, [0, 1, 0], [3, 1, 1]),
            Ok(vec![3, 4, 5])
        );
        let buffer = texture(TextureType::Buffer, [4, 1, 1], vec![9, 8, 7, 6]);
        assert_eq!(
            select_texture(&buffer, [1, 0, 0], [2, 1, 1]),
            Ok(vec![8, 7])
        );
        let mut multisampled = texture(TextureType::D2Multisample, [1, 1, 1], vec![0; 4]);
        multisampled.sample_count = 4;
        assert!(select_texture(&multisampled, [0, 0, 0], [1, 1, 1]).is_err());
    }
}
