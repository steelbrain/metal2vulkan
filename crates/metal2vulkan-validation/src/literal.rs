//! Canonical preparation of authored literal resources.
//!
//! Schema validation establishes shape and ownership. This module performs the one execution-time
//! interpretation shared by every backend: choose the bytes implied by a resource role, decode
//! base64, and preserve function-constant bits.

use crate::case::{
    AttributeFormat, AuthoredCase, FragmentImageblockFormat, OutputSelection, RateMapLiteral,
    ResourceRole, ScalarType, TextureFormat, TextureType,
};
use base64::Engine as _;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LiteralBuffer {
    pub binding: u32,
    /// The bytes a Vulkan executor binds: for a rasterization-rate map, the product's parameter
    /// data of its compile of the map (`RateMapLiteral::compiled`), whether or not the case's
    /// render pass carries a map.
    pub bytes: Vec<u8>,
    /// The rasterization-rate map whose parameter data the buffer holds: the Metal oracle binds
    /// Metal's data of the same map instead of `bytes`.
    pub rasterization_rate_map: Option<RateMapLiteral>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LiteralArgumentBufferBuffer {
    pub buffer_binding: u32,
    pub field_offset: u32,
    /// The bytes a Vulkan executor encodes: for a rasterization-rate map, the product's parameter
    /// data of its compile of the map (`RateMapLiteral::compiled`), whether or not the case's
    /// render pass carries a map.
    pub bytes: Vec<u8>,
    /// The rasterization-rate map whose parameter data the member buffer holds: the Metal oracle
    /// encodes a buffer of Metal's data of the same map instead of `bytes`.
    pub rasterization_rate_map: Option<RateMapLiteral>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LiteralDeviceBufferArray {
    pub binding: u32,
    pub length: u32,
    pub elements: Vec<LiteralDeviceBufferArrayElement>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LiteralDeviceBufferArrayElement {
    pub index: u32,
    pub role: ResourceRole,
    pub bytes: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LiteralStageInput {
    pub location: u32,
    pub format: AttributeFormat,
    pub stride: u32,
    /// Its layout's stride is `MTLBufferLayoutStrideDynamic`, and its data is bound with the
    /// stride form at `stride` (`AttributeInput::dynamic_stride`).
    pub dynamic_stride: bool,
    pub bytes: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LiteralTexture {
    pub binding: u32,
    pub role: ResourceRole,
    pub texture_type: TextureType,
    pub format: TextureFormat,
    pub dimensions: [u32; 3],
    pub sample_count: u32,
    pub bytes: Vec<u8>,
    /// What is mapped of a sparse texture before the work; `None` for an ordinary texture, and for
    /// every texture an executor makes itself (an element of a texture array, an attachment, the
    /// imageblock's storage).
    pub sparse: Option<crate::case::sparse::SparseTextureResource>,
    /// Its mipmap levels, `bytes` holding every one, level 0 first ([`crate::case::mips`]); 1 for
    /// every texture but a direct binding that declares more.
    pub mip_levels: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LiteralArgumentBufferTexture {
    pub buffer_binding: u32,
    pub field_offset: u32,
    pub role: ResourceRole,
    pub texture_type: TextureType,
    pub format: TextureFormat,
    pub dimensions: [u32; 3],
    pub sample_count: u32,
    pub bytes: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LiteralTextureArray {
    pub binding: u32,
    pub elements: Vec<LiteralTexture>,
    pub overrides_texture_at_base: bool,
}

/// A render pass's color attachment: `samples` samples of each pixel of `dimensions`, each
/// pixel's samples in order (one sample is the plain pixel layout).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LiteralRenderTarget {
    pub index: u32,
    pub format: TextureFormat,
    pub dimensions: [u32; 2],
    pub samples: u32,
    pub bytes: Vec<u8>,
}

/// A render pass's depth/stencil attachment: each present aspect holds `samples` samples of each
/// pixel of `dimensions`, each pixel's samples in order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LiteralDepthStencil {
    pub dimensions: [u32; 2],
    pub samples: u32,
    pub depth: Option<Vec<u8>>,
    pub stencil: Option<Vec<u8>>,
}

impl LiteralDepthStencil {
    /// The `(origin, dimensions)` region of the depth aspect (`depth`) or the stencil aspect,
    /// every sample of each pixel, rows in order.
    pub fn select(
        &self,
        depth: bool,
        origin: [u32; 2],
        dimensions: [u32; 2],
    ) -> Result<Vec<u8>, String> {
        let (bytes, texel) = if depth {
            (self.depth.as_ref(), 4)
        } else {
            (self.stencil.as_ref(), 1)
        };
        let bytes =
            bytes.ok_or_else(|| "the selected depth/stencil aspect is absent".to_string())?;
        select_tightly_packed_2d(
            bytes,
            self.dimensions,
            origin,
            dimensions,
            texel * self.samples as usize,
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LiteralFragmentImageblockMember {
    pub semantic: String,
    pub format: FragmentImageblockFormat,
    pub role: ResourceRole,
    pub bytes: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LiteralFragmentImageblock {
    pub dimensions: [u32; 2],
    pub members: Vec<LiteralFragmentImageblockMember>,
}

impl LiteralRenderTarget {
    /// The `(origin, dimensions)` region, every sample of each pixel, rows in order.
    pub fn select(&self, origin: [u32; 2], dimensions: [u32; 2]) -> Result<Vec<u8>, String> {
        select_tightly_packed_2d(
            &self.bytes,
            self.dimensions,
            origin,
            dimensions,
            self.format.bytes_per_pixel() * self.samples as usize,
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TextureLayout {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub array_layers: u32,
    pub sample_count: u32,
}

impl TextureLayout {
    /// The `arrayLength` a Metal texture of `texture_type` with this layout is made with: its
    /// layers for a 1D, 2D or multisampled 2D array, its cubes for a cube array, and 1 otherwise.
    pub fn metal_array_length(self, texture_type: TextureType) -> u32 {
        match texture_type {
            TextureType::D1Array | TextureType::D2Array | TextureType::D2MultisampleArray => {
                self.array_layers
            }
            TextureType::CubeArray => self.array_layers / 6,
            _ => 1,
        }
    }
}

/// The Metal subresource a `(origin, dimensions)` selection names, resolved once.
///
/// An authored case spells a texture's extent the way Metal declares it, which is not the way
/// `replaceRegion`/`getBytes` want it: a `D1Array` carries its layer count in component 1 and has
/// no height, while every other arrayed type carries layers in component 2. Every caller that
/// re-derived that rule from raw `dimensions` indices was one texture type away from being wrong,
/// and both copies in the Metal executor were -- they read a `D1Array` as a `width x layers`
/// rectangle out of slice 0, so an upload landed nowhere and a readback came back all zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SubresourceSpan {
    /// Origin within one slice: x, y, z.
    pub origin: [u32; 3],
    /// Extent of one slice: width, height, depth.
    pub size: [u32; 3],
    pub base_slice: u32,
    pub slice_count: u32,
}

/// Resolve `(origin, dimensions)` in the case's spelling into the Metal subresource it names.
pub fn subresource_span(
    texture_type: TextureType,
    origin: [u32; 3],
    dimensions: [u32; 3],
    sample_count: u32,
) -> Result<SubresourceSpan, String> {
    let layout = texture_layout(texture_type, dimensions, sample_count)?;
    // Which origin component means "layer" is the same question `texture_layout` already answers
    // for the extent, asked about the offset instead. The two must be read off the same table.
    let (y, base_slice) = match texture_type {
        TextureType::Buffer | TextureType::D1 => (0, 0),
        TextureType::D1Array => (0, origin[1]),
        TextureType::D2Array
        | TextureType::D2MultisampleArray
        | TextureType::Cube
        | TextureType::CubeArray => (origin[1], origin[2]),
        TextureType::D2 | TextureType::D2Multisample | TextureType::D3 => (origin[1], 0),
    };
    let z = match texture_type {
        TextureType::D3 => origin[2],
        _ => 0,
    };
    Ok(SubresourceSpan {
        origin: [origin[0], y, z],
        size: [layout.width, layout.height, layout.depth],
        base_slice,
        slice_count: layout.array_layers,
    })
}

impl LiteralTexture {
    pub fn layout(&self) -> Result<TextureLayout, String> {
        texture_layout(self.texture_type, self.dimensions, self.sample_count)
    }

    /// Level `level`'s extent, in the case's spelling, and its bytes, laid out as a one-level
    /// texture of that extent ([`crate::case::mips`]). Level 0 of a one-level texture is its extent
    /// and every byte.
    pub fn level(&self, level: u32) -> Result<([u32; 3], &[u8]), String> {
        use crate::case::mips::{level_dimensions, literal_len};
        if level >= self.mip_levels {
            return Err(format!(
                "texture {} has {} level(s), not a level {level}",
                self.binding, self.mip_levels
            ));
        }
        let texel = self.format.bytes_per_pixel() * self.sample_count as usize;
        let overflow = || format!("texture {} byte size overflows", self.binding);
        let start =
            literal_len(self.texture_type, self.dimensions, level, texel).ok_or_else(overflow)?;
        let dimensions = level_dimensions(self.texture_type, self.dimensions, level);
        let len = literal_len(self.texture_type, dimensions, 1, texel).ok_or_else(overflow)?;
        let bytes = start
            .checked_add(len)
            .and_then(|end| self.bytes.get(start..end))
            .ok_or_else(|| {
                format!(
                    "texture {} supplies {} bytes, short of its level {level}",
                    self.binding,
                    self.bytes.len()
                )
            })?;
        Ok((dimensions, bytes))
    }

    pub fn bytes_per_row(&self) -> Result<usize, String> {
        (self.dimensions[0] as usize)
            .checked_mul(self.sample_count as usize)
            .and_then(|width| width.checked_mul(self.format.bytes_per_pixel()))
            .ok_or_else(|| format!("texture {} row byte size overflows", self.binding))
    }

    pub fn bytes_per_image(&self) -> Result<usize, String> {
        self.bytes_per_row()?
            .checked_mul(self.dimensions[1] as usize)
            .ok_or_else(|| format!("texture {} image byte size overflows", self.binding))
    }

    pub fn select(&self, origin: [u32; 3], dimensions: [u32; 3]) -> Result<Vec<u8>, String> {
        let pixel_size = self.format.bytes_per_pixel();
        let row_stride = self.bytes_per_row()?;
        let image_stride = self.bytes_per_image()?;
        let texel_size = pixel_size * self.sample_count as usize;
        let selected_row = dimensions[0] as usize * texel_size;
        let mut output =
            Vec::with_capacity(selected_row * dimensions[1] as usize * dimensions[2] as usize);
        for z in origin[2]..origin[2] + dimensions[2] {
            for y in origin[1]..origin[1] + dimensions[1] {
                let start = z as usize * image_stride
                    + y as usize * row_stride
                    + origin[0] as usize * texel_size;
                let end = start + selected_row;
                let row = self.bytes.get(start..end).ok_or_else(|| {
                    format!("selected region exceeds texture {} bytes", self.binding)
                })?;
                output.extend_from_slice(row);
            }
        }
        Ok(output)
    }
}

impl LiteralArgumentBufferTexture {
    pub fn layout(&self) -> Result<TextureLayout, String> {
        texture_layout(self.texture_type, self.dimensions, self.sample_count)
    }

    pub fn bytes_per_row(&self) -> Result<usize, String> {
        (self.dimensions[0] as usize)
            .checked_mul(self.sample_count as usize)
            .and_then(|width| width.checked_mul(self.format.bytes_per_pixel()))
            .ok_or_else(|| format!("{} row byte size overflows", self.label()))
    }

    pub fn bytes_per_image(&self) -> Result<usize, String> {
        self.bytes_per_row()?
            .checked_mul(self.dimensions[1] as usize)
            .ok_or_else(|| format!("{} image byte size overflows", self.label()))
    }

    pub fn select(&self, origin: [u32; 3], dimensions: [u32; 3]) -> Result<Vec<u8>, String> {
        let pixel_size = self.format.bytes_per_pixel();
        let row_stride = self.bytes_per_row()?;
        let image_stride = self.bytes_per_image()?;
        let texel_size = pixel_size * self.sample_count as usize;
        let selected_row = dimensions[0] as usize * texel_size;
        let mut output =
            Vec::with_capacity(selected_row * dimensions[1] as usize * dimensions[2] as usize);
        for z in origin[2]..origin[2] + dimensions[2] {
            for y in origin[1]..origin[1] + dimensions[1] {
                let start = z as usize * image_stride
                    + y as usize * row_stride
                    + origin[0] as usize * texel_size;
                let end = start + selected_row;
                let row = self
                    .bytes
                    .get(start..end)
                    .ok_or_else(|| format!("selected region exceeds {} bytes", self.label()))?;
                output.extend_from_slice(row);
            }
        }
        Ok(output)
    }

    pub fn label(&self) -> String {
        format!(
            "argument-buffer texture {}+{}",
            self.buffer_binding, self.field_offset
        )
    }
}

pub fn texture_layout(
    texture_type: TextureType,
    dimensions: [u32; 3],
    sample_count: u32,
) -> Result<TextureLayout, String> {
    let [width, second, third] = dimensions;
    let invalid = || {
        format!(
            "invalid {texture_type:?} dimensions {dimensions:?} and sample count {sample_count}"
        )
    };
    let layout = match texture_type {
        TextureType::Buffer if second == 1 && third == 1 && sample_count == 1 => TextureLayout {
            width,
            height: 1,
            depth: 1,
            array_layers: 1,
            sample_count: 1,
        },
        TextureType::D1 if second == 1 && third == 1 && sample_count == 1 => TextureLayout {
            width,
            height: 1,
            depth: 1,
            array_layers: 1,
            sample_count: 1,
        },
        TextureType::D1Array if third == 1 && sample_count == 1 => TextureLayout {
            width,
            height: 1,
            depth: 1,
            array_layers: second,
            sample_count: 1,
        },
        TextureType::D2 if third == 1 && sample_count == 1 => TextureLayout {
            width,
            height: second,
            depth: 1,
            array_layers: 1,
            sample_count: 1,
        },
        TextureType::D2Array if sample_count == 1 => TextureLayout {
            width,
            height: second,
            depth: 1,
            array_layers: third,
            sample_count: 1,
        },
        TextureType::D2Multisample if third == 1 && matches!(sample_count, 2 | 4 | 8) => {
            TextureLayout {
                width,
                height: second,
                depth: 1,
                array_layers: 1,
                sample_count,
            }
        }
        TextureType::D2MultisampleArray if matches!(sample_count, 2 | 4 | 8) => TextureLayout {
            width,
            height: second,
            depth: 1,
            array_layers: third,
            sample_count,
        },
        TextureType::D3 if sample_count == 1 => TextureLayout {
            width,
            height: second,
            depth: third,
            array_layers: 1,
            sample_count: 1,
        },
        TextureType::Cube if width == second && third == 6 && sample_count == 1 => TextureLayout {
            width,
            height: width,
            depth: 1,
            array_layers: 6,
            sample_count: 1,
        },
        TextureType::CubeArray
            if width == second && third.is_multiple_of(6) && sample_count == 1 =>
        {
            TextureLayout {
                width,
                height: width,
                depth: 1,
                array_layers: third,
                sample_count: 1,
            }
        }
        _ => return Err(invalid()),
    };
    if width == 0
        || layout.height == 0
        || layout.depth == 0
        || layout.array_layers == 0
        || layout.sample_count == 0
    {
        return Err(invalid());
    }
    Ok(layout)
}

pub(crate) fn select_tightly_packed_2d(
    bytes: &[u8],
    extent: [u32; 2],
    origin: [u32; 2],
    dimensions: [u32; 2],
    pixel_size: usize,
) -> Result<Vec<u8>, String> {
    let invalid = || "selected attachment range exceeds its layout or bytes".to_string();
    if pixel_size == 0
        || (0..2).any(|axis| {
            origin[axis]
                .checked_add(dimensions[axis])
                .is_none_or(|end| end > extent[axis])
        })
    {
        return Err(invalid());
    }
    if dimensions.contains(&0) {
        return Ok(Vec::new());
    }
    let source_row = (extent[0] as usize)
        .checked_mul(pixel_size)
        .ok_or_else(invalid)?;
    let selected_row = (dimensions[0] as usize)
        .checked_mul(pixel_size)
        .ok_or_else(invalid)?;
    let first = (origin[1] as usize)
        .checked_mul(source_row)
        .and_then(|offset| {
            (origin[0] as usize)
                .checked_mul(pixel_size)
                .and_then(|x| offset.checked_add(x))
        })
        .ok_or_else(invalid)?;
    let last_end = ((dimensions[1] - 1) as usize)
        .checked_mul(source_row)
        .and_then(|offset| first.checked_add(offset))
        .and_then(|offset| offset.checked_add(selected_row))
        .filter(|end| *end <= bytes.len())
        .ok_or_else(invalid)?;
    // Validate the complete span before allocating output. The checked, nonoverlapping rows
    // bound the output size by the available source bytes, including on narrow hosts.
    let mut output = Vec::with_capacity(selected_row * dimensions[1] as usize);
    for row in bytes[first..last_end].chunks(source_row) {
        output.extend_from_slice(&row[..selected_row]);
    }
    Ok(output)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LiteralFunctionConstant {
    pub index: u32,
    pub scalar_type: ScalarType,
    pub lanes: u32,
    pub bytes: Vec<u8>,
}

/// Decode only the authored function constants needed by product specialization.
///
/// Translation-only workers use this instead of materializing every execution resource.
pub fn function_constants(case: &AuthoredCase) -> Result<Vec<LiteralFunctionConstant>, String> {
    case.function_constants
        .iter()
        .map(|constant| {
            Ok(LiteralFunctionConstant {
                index: constant.index,
                scalar_type: constant.scalar_type,
                lanes: constant.lanes,
                bytes: decode(
                    &constant.bytes_b64,
                    &format!("function constant {}", constant.index),
                )?,
            })
        })
        .collect()
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LiteralTessellation {
    pub factors: Vec<crate::case::TessellationFactors>,
    pub instance_count: u32,
    pub amplification_count: u32,
    pub control_points: Vec<LiteralStageInput>,
    pub patch_inputs: Vec<LiteralStageInput>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LiteralResources {
    pub buffers: Vec<LiteralBuffer>,
    pub argument_buffer_buffers: Vec<LiteralArgumentBufferBuffer>,
    pub device_buffer_arrays: Vec<LiteralDeviceBufferArray>,
    pub vertex_inputs: Vec<LiteralStageInput>,
    pub kernel_stage_inputs: Vec<LiteralStageInput>,
    pub textures: Vec<LiteralTexture>,
    pub texture_arrays: Vec<LiteralTextureArray>,
    pub argument_buffer_textures: Vec<LiteralArgumentBufferTexture>,
    pub render_targets: Vec<LiteralRenderTarget>,
    pub depth_stencil: Option<LiteralDepthStencil>,
    pub fragment_imageblock: Option<LiteralFragmentImageblock>,
    pub function_constants: Vec<LiteralFunctionConstant>,
    pub tessellation: Option<LiteralTessellation>,
}

impl LiteralResources {
    pub fn prepare(case: &AuthoredCase) -> Result<Self, String> {
        let buffers = case
            .buffers
            .iter()
            .map(|resource| {
                if let Some(map) = &resource.rasterization_rate_map {
                    return Ok(LiteralBuffer {
                        binding: resource.binding,
                        bytes: map
                            .parameter_data()
                            .map_err(|error| format!("buffer {}: {error}", resource.binding))?,
                        rasterization_rate_map: Some(map.clone()),
                    });
                }
                let encoded = role_bytes(
                    resource.role,
                    resource.bytes_b64.as_deref(),
                    resource.initial_bytes_b64.as_deref(),
                )
                .ok_or_else(|| format!("buffer {} has no literal bytes", resource.binding))?;
                Ok(LiteralBuffer {
                    binding: resource.binding,
                    bytes: decode(encoded, &format!("buffer {}", resource.binding))?,
                    rasterization_rate_map: None,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let argument_buffer_buffers = case
            .argument_buffer_buffers
            .iter()
            .map(|resource| {
                let label = format!(
                    "argument-buffer buffer {}+{}",
                    resource.buffer_binding, resource.field_offset
                );
                if let Some(map) = &resource.rasterization_rate_map {
                    return Ok(LiteralArgumentBufferBuffer {
                        buffer_binding: resource.buffer_binding,
                        field_offset: resource.field_offset,
                        bytes: map
                            .parameter_data()
                            .map_err(|error| format!("{label}: {error}"))?,
                        rasterization_rate_map: Some(map.clone()),
                    });
                }
                let encoded = role_bytes(
                    resource.role,
                    resource.bytes_b64.as_deref(),
                    resource.initial_bytes_b64.as_deref(),
                )
                .ok_or_else(|| format!("{label} has no literal bytes"))?;
                Ok(LiteralArgumentBufferBuffer {
                    buffer_binding: resource.buffer_binding,
                    field_offset: resource.field_offset,
                    bytes: decode(encoded, &label)?,
                    rasterization_rate_map: None,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let device_buffer_arrays = case
            .device_buffer_arrays
            .iter()
            .map(|array| {
                let elements = array
                    .elements
                    .iter()
                    .map(|resource| {
                        let label = format!(
                            "device-buffer-array {} element {}",
                            array.binding, resource.index
                        );
                        let encoded = role_bytes(
                            resource.role,
                            resource.bytes_b64.as_deref(),
                            resource.initial_bytes_b64.as_deref(),
                        )
                        .ok_or_else(|| format!("{label} has no literal bytes"))?;
                        Ok(LiteralDeviceBufferArrayElement {
                            index: resource.index,
                            role: resource.role,
                            bytes: decode(encoded, &label)?,
                        })
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                Ok(LiteralDeviceBufferArray {
                    binding: array.binding,
                    length: array.length,
                    elements,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let kernel_stage_inputs = case
            .kernel_stage_inputs
            .iter()
            .map(|resource| {
                Ok(LiteralStageInput {
                    location: resource.location,
                    format: resource.format,
                    stride: resource.stride,
                    dynamic_stride: resource.dynamic_stride,
                    bytes: decode(
                        &resource.bytes_b64,
                        &format!("kernel stage input {}", resource.location),
                    )?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let vertex_inputs = case
            .vertex_inputs
            .iter()
            .map(|resource| {
                Ok(LiteralStageInput {
                    location: resource.location,
                    format: resource.format,
                    stride: resource.stride,
                    dynamic_stride: resource.dynamic_stride,
                    bytes: decode(
                        &resource.bytes_b64,
                        &format!("vertex input {}", resource.location),
                    )?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let textures = case
            .textures
            .iter()
            .map(|resource| {
                let encoded = role_bytes(
                    resource.role,
                    resource.bytes_b64.as_deref(),
                    resource.initial_bytes_b64.as_deref(),
                )
                .ok_or_else(|| format!("texture {} has no literal bytes", resource.binding))?;
                Ok(LiteralTexture {
                    binding: resource.binding,
                    role: resource.role,
                    texture_type: resource.texture_type,
                    format: resource.format,
                    dimensions: resource.dimensions,
                    sample_count: resource.sample_count,
                    bytes: decode(encoded, &format!("texture {}", resource.binding))?,
                    sparse: resource.sparse.clone(),
                    mip_levels: resource.mip_levels,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let argument_buffer_textures = case
            .argument_buffer_textures
            .iter()
            .map(|resource| {
                let label = format!(
                    "argument-buffer texture {}+{}",
                    resource.buffer_binding, resource.field_offset
                );
                let encoded = role_bytes(
                    resource.role,
                    resource.bytes_b64.as_deref(),
                    resource.initial_bytes_b64.as_deref(),
                )
                .ok_or_else(|| format!("{label} has no literal bytes"))?;
                Ok(LiteralArgumentBufferTexture {
                    buffer_binding: resource.buffer_binding,
                    field_offset: resource.field_offset,
                    role: resource.role,
                    texture_type: resource.texture_type,
                    format: resource.format,
                    dimensions: resource.dimensions,
                    sample_count: resource.sample_count,
                    bytes: decode(encoded, &label)?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let texture_arrays = case
            .texture_arrays
            .iter()
            .map(|array| {
                let elements = array
                    .elements
                    .iter()
                    .enumerate()
                    .map(|(index, element)| {
                        let label = format!("texture-array {} element {index}", array.binding);
                        let encoded = role_bytes(
                            array.role,
                            element.bytes_b64.as_deref(),
                            element.initial_bytes_b64.as_deref(),
                        )
                        .ok_or_else(|| format!("{label} has no literal bytes"))?;
                        Ok(LiteralTexture {
                            binding: array.binding,
                            role: array.role,
                            texture_type: array.texture_type,
                            format: array.format,
                            dimensions: element.dimensions,
                            sample_count: array.sample_count,
                            bytes: decode(encoded, &label)?,
                            sparse: None,
                            mip_levels: 1,
                        })
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                Ok(LiteralTextureArray {
                    binding: array.binding,
                    elements,
                    overrides_texture_at_base: array.overrides_texture_at_base,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let function_constants = function_constants(case)?;
        let render_targets = case
            .render_targets
            .iter()
            .map(|target| {
                Ok(LiteralRenderTarget {
                    index: target.index,
                    format: target.format,
                    dimensions: target.dimensions,
                    samples: case.raster_samples(),
                    bytes: decode(
                        &target.initial_bytes_b64,
                        &format!("render target {}", target.index),
                    )?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let fragment_imageblock = case
            .fragment_imageblock
            .as_ref()
            .map(|imageblock| {
                let members = imageblock
                    .members
                    .iter()
                    .map(|member| {
                        let encoded = role_bytes(
                            member.role,
                            member.bytes_b64.as_deref(),
                            member.initial_bytes_b64.as_deref(),
                        )
                        .ok_or_else(|| {
                            format!(
                                "fragment imageblock member {} has no literal bytes",
                                member.semantic
                            )
                        })?;
                        Ok(LiteralFragmentImageblockMember {
                            semantic: member.semantic.clone(),
                            format: member.format,
                            role: member.role,
                            bytes: decode(
                                encoded,
                                &format!("fragment imageblock member {}", member.semantic),
                            )?,
                        })
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                Ok::<_, String>(LiteralFragmentImageblock {
                    dimensions: imageblock.dimensions,
                    members,
                })
            })
            .transpose()?;
        let depth_stencil = case
            .depth_stencil
            .as_ref()
            .map(|attachment| {
                Ok::<_, String>(LiteralDepthStencil {
                    dimensions: attachment.dimensions,
                    samples: case.raster_samples(),
                    depth: attachment
                        .initial_depth_b64
                        .as_deref()
                        .map(|bytes| decode(bytes, "depth attachment"))
                        .transpose()?,
                    stencil: attachment
                        .initial_stencil_b64
                        .as_deref()
                        .map(|bytes| decode(bytes, "stencil attachment"))
                        .transpose()?,
                })
            })
            .transpose()?;
        let tessellation = case
            .tessellation
            .as_ref()
            .map(|tessellation| {
                Ok::<_, String>(LiteralTessellation {
                    factors: tessellation.factors.clone(),
                    instance_count: tessellation.instance_count,
                    amplification_count: tessellation.amplification_count,
                    control_points: decode_stage_inputs(
                        "tessellation control point",
                        &tessellation.control_points,
                    )?,
                    patch_inputs: decode_stage_inputs(
                        "tessellation patch input",
                        &tessellation.patch_inputs,
                    )?,
                })
            })
            .transpose()?;
        Ok(Self {
            buffers,
            argument_buffer_buffers,
            device_buffer_arrays,
            vertex_inputs,
            kernel_stage_inputs,
            textures,
            texture_arrays,
            argument_buffer_textures,
            render_targets,
            depth_stencil,
            fragment_imageblock,
            function_constants,
            tessellation,
        })
    }

    /// The authored bytes of the output `output` selects, in the layout a run reads them back.
    pub fn selected_initial(&self, output: &OutputSelection) -> Result<Vec<u8>, String> {
        match output {
            OutputSelection::None => Ok(Vec::new()),
            OutputSelection::Buffer {
                binding,
                offset,
                length,
            } => {
                let resource = self
                    .buffers
                    .iter()
                    .find(|resource| resource.binding == *binding)
                    .ok_or_else(|| format!("missing output buffer binding {binding}"))?;
                let all = &resource.bytes;
                let start = *offset as usize;
                let end = start
                    .checked_add(*length as usize)
                    .ok_or_else(|| "selected output range overflow".to_string())?;
                all.get(start..end)
                    .map(<[u8]>::to_vec)
                    .ok_or_else(|| "selected output range exceeds buffer".into())
            }
            OutputSelection::ArgumentBufferBuffer {
                buffer_binding,
                field_offset,
                offset,
                length,
            } => {
                let resource = self
                    .argument_buffer_buffers
                    .iter()
                    .find(|resource| {
                        resource.buffer_binding == *buffer_binding
                            && resource.field_offset == *field_offset
                    })
                    .ok_or_else(|| {
                        format!("missing argument-buffer buffer {buffer_binding}+{field_offset}")
                    })?;
                let start = *offset as usize;
                let end = start
                    .checked_add(*length as usize)
                    .ok_or_else(|| "selected output range overflow".to_string())?;
                resource
                    .bytes
                    .get(start..end)
                    .map(<[u8]>::to_vec)
                    .ok_or_else(|| "selected output range exceeds argument-buffer buffer".into())
            }
            OutputSelection::DeviceBufferArrayElement {
                binding,
                element,
                offset,
                length,
            } => {
                let resource = self
                    .device_buffer_arrays
                    .iter()
                    .find(|array| array.binding == *binding)
                    .and_then(|array| array.elements.iter().find(|item| item.index == *element))
                    .ok_or_else(|| {
                        format!("missing device-buffer-array binding {binding} element {element}")
                    })?;
                let start = *offset as usize;
                let end = start
                    .checked_add(*length as usize)
                    .ok_or_else(|| "selected output range overflow".to_string())?;
                resource
                    .bytes
                    .get(start..end)
                    .map(<[u8]>::to_vec)
                    .ok_or_else(|| {
                        "selected output range exceeds device-buffer-array element".into()
                    })
            }
            OutputSelection::Texture {
                binding,
                origin,
                dimensions,
            } => self
                .textures
                .iter()
                .find(|resource| resource.binding == *binding)
                .ok_or_else(|| format!("missing output texture binding {binding}"))?
                .select(*origin, *dimensions),
            OutputSelection::TextureArrayElement {
                binding,
                element,
                origin,
                dimensions,
            } => self
                .texture_arrays
                .iter()
                .find(|array| array.binding == *binding)
                .and_then(|array| array.elements.get(*element as usize))
                .ok_or_else(|| {
                    format!("missing texture-array binding {binding} element {element}")
                })?
                .select(*origin, *dimensions),
            OutputSelection::ArgumentBufferTexture {
                buffer_binding,
                field_offset,
                origin,
                dimensions,
            } => self
                .argument_buffer_textures
                .iter()
                .find(|resource| {
                    resource.buffer_binding == *buffer_binding
                        && resource.field_offset == *field_offset
                })
                .ok_or_else(|| {
                    format!("missing argument-buffer texture {buffer_binding}+{field_offset}")
                })?
                .select(*origin, *dimensions),
            OutputSelection::RenderTarget {
                index,
                origin,
                dimensions,
            } => self
                .render_targets
                .iter()
                .find(|target| target.index == *index)
                .ok_or_else(|| format!("missing render target {index}"))?
                .select(*origin, *dimensions),
            OutputSelection::Depth { origin, dimensions }
            | OutputSelection::Stencil { origin, dimensions } => {
                let attachment = self
                    .depth_stencil
                    .as_ref()
                    .ok_or_else(|| "missing depth/stencil attachment".to_string())?;
                attachment.select(
                    matches!(output, OutputSelection::Depth { .. }),
                    *origin,
                    *dimensions,
                )
            }
            OutputSelection::FragmentImageblock {
                semantic,
                origin,
                dimensions,
            } => {
                let imageblock = self
                    .fragment_imageblock
                    .as_ref()
                    .ok_or_else(|| "missing fragment imageblock".to_string())?;
                let member = imageblock
                    .members
                    .iter()
                    .find(|member| member.semantic == *semantic)
                    .ok_or_else(|| format!("missing fragment imageblock member {semantic}"))?;
                select_tightly_packed_2d(
                    &member.bytes,
                    imageblock.dimensions,
                    *origin,
                    *dimensions,
                    member.format.byte_size(),
                )
            }
        }
    }
}

fn decode_stage_inputs(
    label: &str,
    inputs: &[crate::case::AttributeInput],
) -> Result<Vec<LiteralStageInput>, String> {
    inputs
        .iter()
        .map(|input| {
            Ok(LiteralStageInput {
                location: input.location,
                format: input.format,
                stride: input.stride,
                dynamic_stride: input.dynamic_stride,
                bytes: decode(&input.bytes_b64, &format!("{label} {}", input.location))?,
            })
        })
        .collect()
}

fn role_bytes<'a>(
    role: ResourceRole,
    input: Option<&'a str>,
    initial: Option<&'a str>,
) -> Option<&'a str> {
    match role {
        ResourceRole::Input => input,
        ResourceRole::Output | ResourceRole::InOut => initial,
    }
}

fn decode(encoded: &str, label: &str) -> Result<Vec<u8>, String> {
    base64::engine::general_purpose::STANDARD
        .decode(encoded)
        .map_err(|error| format!("decode {label}: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn texture_layout_distinguishes_depth_layers_faces_and_samples() {
        assert_eq!(
            texture_layout(TextureType::Buffer, [8, 1, 1], 1).unwrap(),
            TextureLayout {
                width: 8,
                height: 1,
                depth: 1,
                array_layers: 1,
                sample_count: 1,
            }
        );
        assert_eq!(
            texture_layout(TextureType::D3, [2, 3, 4], 1).unwrap(),
            TextureLayout {
                width: 2,
                height: 3,
                depth: 4,
                array_layers: 1,
                sample_count: 1,
            }
        );
        assert_eq!(
            texture_layout(TextureType::CubeArray, [2, 2, 12], 1).unwrap(),
            TextureLayout {
                width: 2,
                height: 2,
                depth: 1,
                array_layers: 12,
                sample_count: 1,
            }
        );
        assert_eq!(
            texture_layout(TextureType::D2Multisample, [2, 3, 1], 4).unwrap(),
            TextureLayout {
                width: 2,
                height: 3,
                depth: 1,
                array_layers: 1,
                sample_count: 4,
            }
        );
    }

    #[test]
    fn texture_region_selection_is_tightly_packed_across_slices() {
        let texture = LiteralTexture {
            binding: 7,
            role: ResourceRole::Output,
            texture_type: TextureType::D2Array,
            format: TextureFormat::R8Unorm,
            dimensions: [3, 2, 2],
            sample_count: 1,
            bytes: (0..12).collect(),
            sparse: None,
            mip_levels: 1,
        };
        assert_eq!(
            texture.select([1, 0, 0], [2, 2, 2]).unwrap(),
            vec![1, 2, 4, 5, 7, 8, 10, 11]
        );
    }

    #[test]
    fn render_target_selection_is_tightly_packed() {
        let target = LiteralRenderTarget {
            index: 2,
            format: TextureFormat::R8Unorm,
            dimensions: [3, 2],
            samples: 1,
            bytes: (0..6).collect(),
        };
        assert_eq!(target.select([1, 0], [2, 2]).unwrap(), vec![1, 2, 4, 5]);
        // Every sample of each selected pixel, in order.
        let multisampled = LiteralRenderTarget {
            samples: 2,
            bytes: (0..12).collect(),
            ..target
        };
        assert_eq!(
            multisampled.select([1, 0], [2, 2]).unwrap(),
            vec![2, 3, 4, 5, 8, 9, 10, 11]
        );
    }

    #[test]
    fn attachment_selection_rejects_geometry_and_overflow_before_allocating() {
        let bytes = [0u8; 16];
        for (extent, origin, dimensions, pixel_size) in [
            ([4, 4], [3, 0], [2, 1], 1),
            ([4, 4], [0, 4], [1, 1], 1),
            ([u32::MAX, 4], [u32::MAX, 0], [1, 1], 1),
            ([4, 4], [0, 0], [1, 1], usize::MAX),
            ([u32::MAX; 2], [0, 0], [u32::MAX; 2], 4),
            ([4, 4], [0, 0], [1, 1], 0),
        ] {
            assert!(
                select_tightly_packed_2d(&bytes, extent, origin, dimensions, pixel_size).is_err()
            );
        }
        assert_eq!(
            select_tightly_packed_2d(&bytes, [4, 4], [4, 4], [0, 0], 1).unwrap(),
            Vec::<u8>::new()
        );
        assert_eq!(
            select_tightly_packed_2d(&bytes, [u32::MAX; 2], [0, 0], [0, u32::MAX], 4).unwrap(),
            Vec::<u8>::new()
        );
    }

    #[test]
    fn depth_stencil_selection_is_tightly_packed() {
        let bytes = (0u8..16).collect::<Vec<_>>();
        assert_eq!(
            select_tightly_packed_2d(&bytes, [4, 4], [1, 1], [2, 2], 1).unwrap(),
            [5, 6, 9, 10]
        );
    }

    /// Every texture type's `(origin, dimensions)` spelling, resolved to the Metal subresource it
    /// names. `D1Array` is the whole reason this table exists: it is the one type whose layer count
    /// sits in component 1, so a reader that assumed component 1 is always the height and component
    /// 2 is always the layer asked Metal for a `width x layers` rectangle out of slice 0.
    #[test]
    fn a_subresource_span_reads_each_texture_type_the_way_that_type_spells_itself() {
        let span = |ty, origin, dimensions| subresource_span(ty, origin, dimensions, 1).unwrap();

        let d1 = span(TextureType::D1, [2, 0, 0], [8, 1, 1]);
        assert_eq!(d1.size, [8, 1, 1]);
        assert_eq!((d1.base_slice, d1.slice_count), (0, 1));

        // The layer count is component 1, the height is always one, and the layer origin is the
        // base slice rather than a y offset.
        let d1_array = span(TextureType::D1Array, [2, 1, 0], [6, 3, 1]);
        assert_eq!(d1_array.origin, [2, 0, 0]);
        assert_eq!(d1_array.size, [6, 1, 1]);
        assert_eq!((d1_array.base_slice, d1_array.slice_count), (1, 3));

        let d2 = span(TextureType::D2, [1, 2, 0], [5, 4, 1]);
        assert_eq!(d2.origin, [1, 2, 0]);
        assert_eq!(d2.size, [5, 4, 1]);
        assert_eq!((d2.base_slice, d2.slice_count), (0, 1));

        // Here component 1 really is the height and component 2 really is the layer.
        let d2_array = span(TextureType::D2Array, [1, 2, 3], [5, 4, 7]);
        assert_eq!(d2_array.origin, [1, 2, 0]);
        assert_eq!(d2_array.size, [5, 4, 1]);
        assert_eq!((d2_array.base_slice, d2_array.slice_count), (3, 7));

        // A 3D texture is the only one that keeps a z offset and a depth greater than one.
        let d3 = span(TextureType::D3, [1, 2, 3], [5, 4, 6]);
        assert_eq!(d3.origin, [1, 2, 3]);
        assert_eq!(d3.size, [5, 4, 6]);
        assert_eq!((d3.base_slice, d3.slice_count), (0, 1));

        let cube = span(TextureType::Cube, [0, 0, 2], [4, 4, 6]);
        assert_eq!(cube.size, [4, 4, 1]);
        assert_eq!((cube.base_slice, cube.slice_count), (2, 6));

        let cube_array = span(TextureType::CubeArray, [0, 0, 6], [4, 4, 12]);
        assert_eq!(cube_array.size, [4, 4, 1]);
        assert_eq!((cube_array.base_slice, cube_array.slice_count), (6, 12));

        let buffer = span(TextureType::Buffer, [3, 0, 0], [9, 1, 1]);
        assert_eq!(buffer.origin, [3, 0, 0]);
        assert_eq!(buffer.size, [9, 1, 1]);
        assert_eq!((buffer.base_slice, buffer.slice_count), (0, 1));
    }

    /// The byte strides the Metal executor derives from a span. A `D1Array` slice is one row, so a
    /// three-layer 6-texel array advances 96 bytes per slice, not the 288 a height-of-three would.
    #[test]
    fn a_subresource_span_gives_each_arrayed_type_a_one_slice_byte_stride() {
        let stride = |ty, dimensions| {
            let span = subresource_span(ty, [0, 0, 0], dimensions, 1).unwrap();
            let row = span.size[0] as usize * TextureFormat::Rgba32Uint.bytes_per_pixel();
            (
                row * span.size[1] as usize * span.size[2] as usize,
                span.slice_count as usize,
            )
        };
        assert_eq!(stride(TextureType::D1Array, [6, 3, 1]), (96, 3));
        assert_eq!(stride(TextureType::D2Array, [6, 2, 3]), (192, 3));
        assert_eq!(stride(TextureType::D3, [6, 2, 3]), (576, 1));
        assert_eq!(stride(TextureType::Cube, [4, 4, 6]), (256, 6));
    }

    /// A multisample selection keeps its sample count out of the span's extent, which is what lets
    /// the caller multiply it back in at whatever texel size that path uses.
    #[test]
    fn a_multisample_span_carries_its_layers_without_its_samples() {
        let span =
            subresource_span(TextureType::D2MultisampleArray, [0, 0, 1], [4, 4, 3], 4).unwrap();
        assert_eq!(span.size, [4, 4, 1]);
        assert_eq!((span.base_slice, span.slice_count), (1, 3));
    }
}
