//! Native texture ownership, declared usage, uploads, and multisample initialization.

use super::{commit_and_wait, mtl_size, Buffer};
use crate::case::{ResourceRole, TextureFormat, TextureType};
use crate::check::interface::{
    self, declared_texture_atomic, declared_texture_writable, AirInterface,
};
use crate::gpu::compile;
use crate::literal::{LiteralResources, LiteralTexture};
use core::ffi::c_void;
use core::ptr::NonNull;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
    MTLCompareFunction, MTLDepthStencilDescriptor, MTLDevice, MTLHeap, MTLHeapDescriptor,
    MTLHeapType, MTLLibrary, MTLLoadAction, MTLOrigin, MTLPixelFormat, MTLPrimitiveType, MTLRegion,
    MTLRenderCommandEncoder, MTLRenderPassDescriptor, MTLRenderPipelineDescriptor,
    MTLResourceOptions, MTLResourceStateCommandEncoder, MTLSize, MTLSparsePageSize,
    MTLSparseTextureMappingMode, MTLStorageMode, MTLStoreAction, MTLTexture, MTLTextureDescriptor,
    MTLTextureType, MTLTextureUsage,
};

type RawTexture = Retained<ProtocolObject<dyn MTLTexture>>;

pub(super) struct Texture {
    pub(super) object: RawTexture,
    pub(super) _backing: Option<Buffer>,
    /// The sparse heap a sparse texture's tiles are mapped from.
    pub(super) _heap: Option<Retained<ProtocolObject<dyn MTLHeap>>>,
}

impl std::ops::Deref for Texture {
    type Target = ProtocolObject<dyn MTLTexture>;

    fn deref(&self) -> &Self::Target {
        &self.object
    }
}

/// The Metal usage a texture argument needs ([`interface::texture_shader_usage`]).
pub(super) fn texture_shader_usage(role: ResourceRole, writable: Option<bool>) -> MTLTextureUsage {
    MTLTextureUsage(interface::texture_shader_usage(role, writable).as_raw() as usize)
}

pub(super) fn make_textures(
    device: &ProtocolObject<dyn MTLDevice>,
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    resources: &LiteralResources,
    interface: &AirInterface,
) -> Result<Vec<(u32, Texture)>, String> {
    resources
        .textures
        .iter()
        .map(|resource| {
            let label = format!("texture {}", resource.binding);
            let mut usage = texture_shader_usage(
                resource.role,
                declared_texture_writable(interface, resource.binding),
            );
            // A texture atomic needs the texture made for it: without the usage, a 2D array's
            // atomic at slice 1 reads and writes slice 0 on Apple M3.
            if declared_texture_atomic(interface, resource.binding) {
                usage |= MTLTextureUsage::ShaderAtomic;
            }
            if resource.sparse.is_some() {
                let texture = make_sparse_texture(device, queue, &label, usage, resource)?;
                return Ok((resource.binding, texture));
            }
            let texture = make_texture(
                device,
                queue,
                MetalTextureLiteral {
                    label: &label,
                    usage,
                    texture_type: resource.texture_type,
                    format: resource.format,
                    dimensions: resource.dimensions,
                    sample_count: resource.sample_count,
                    mip_levels: resource.mip_levels,
                    bytes: &resource.bytes,
                },
            )?;
            Ok((resource.binding, texture))
        })
        .collect()
}

pub(super) type TextureArray = (u32, Vec<Texture>);

pub(super) fn make_texture_arrays(
    device: &ProtocolObject<dyn MTLDevice>,
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    resources: &LiteralResources,
    interface: &AirInterface,
) -> Result<Vec<TextureArray>, String> {
    resources
        .texture_arrays
        .iter()
        .map(|array| {
            let writable = declared_texture_writable(interface, array.binding);
            let elements = array
                .elements
                .iter()
                .enumerate()
                .map(|(element, resource)| {
                    let label = format!("texture-array {} element {element}", array.binding);
                    make_texture(
                        device,
                        queue,
                        MetalTextureLiteral {
                            label: &label,
                            usage: texture_shader_usage(resource.role, writable),
                            texture_type: resource.texture_type,
                            format: resource.format,
                            dimensions: resource.dimensions,
                            sample_count: resource.sample_count,
                            mip_levels: 1,
                            bytes: &resource.bytes,
                        },
                    )
                })
                .collect::<Result<Vec<_>, String>>()?;
            Ok((array.binding, elements))
        })
        .collect()
}

pub(super) struct MetalTextureLiteral<'a> {
    pub(super) label: &'a str,
    pub(super) usage: MTLTextureUsage,
    pub(super) texture_type: TextureType,
    pub(super) format: TextureFormat,
    pub(super) dimensions: [u32; 3],
    pub(super) sample_count: u32,
    /// Its levels, `bytes` holding every one, level 0 first.
    pub(super) mip_levels: u32,
    pub(super) bytes: &'a [u8],
}

pub(super) fn make_texture(
    device: &ProtocolObject<dyn MTLDevice>,
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    literal: MetalTextureLiteral<'_>,
) -> Result<Texture, String> {
    let MetalTextureLiteral {
        label,
        usage,
        texture_type,
        format,
        dimensions,
        sample_count,
        mip_levels,
        bytes,
    } = literal;
    let layout = crate::literal::texture_layout(texture_type, dimensions, sample_count)?;
    let descriptor = MTLTextureDescriptor::new();
    descriptor.setTextureType(metal_texture_type(texture_type));
    descriptor.setPixelFormat(metal_pixel_format(format));
    unsafe {
        descriptor.setWidth(layout.width as usize);
        descriptor.setHeight(layout.height as usize);
        descriptor.setDepth(layout.depth as usize);
        descriptor.setArrayLength(layout.metal_array_length(texture_type) as usize);
        descriptor.setSampleCount(layout.sample_count as usize);
        descriptor.setMipmapLevelCount(mip_levels as usize);
    }
    descriptor.setStorageMode(if layout.sample_count == 1 {
        MTLStorageMode::Shared
    } else {
        MTLStorageMode::Private
    });
    descriptor.setUsage(if layout.sample_count == 1 {
        usage
    } else {
        usage | MTLTextureUsage::RenderTarget
    });
    if texture_type == TextureType::Buffer {
        let pointer = NonNull::new(bytes.as_ptr().cast_mut().cast::<c_void>())
            .ok_or_else(|| format!("{label} bytes pointer is null"))?;
        let backing = unsafe {
            device.newBufferWithBytes_length_options(
                pointer,
                bytes.len(),
                MTLResourceOptions::StorageModeShared,
            )
        }
        .ok_or_else(|| format!("create Metal buffer backing for {label}"))?;
        let texture = backing
            .newTextureWithDescriptor_offset_bytesPerRow(&descriptor, 0, bytes.len())
            .ok_or_else(|| format!("create Metal texture-buffer view for {label}"))?;
        return Ok(Texture {
            object: texture,
            _backing: Some(backing),
            _heap: None,
        });
    }
    let texture = device
        .newTextureWithDescriptor(&descriptor)
        .ok_or_else(|| format!("create Metal {label}"))?;
    if layout.sample_count != 1 {
        initialize_multisample_texture(device, queue, &texture, label, layout, format, bytes)?;
        return Ok(Texture {
            object: texture,
            _backing: None,
            _heap: None,
        });
    }
    // Each level's bytes, level 0 first, each laid out as a one-level texture of its extent; a
    // one-level texture's level 0 is every byte.
    let texel = format.bytes_per_pixel() * sample_count as usize;
    let mut start = 0usize;
    for level in 0..mip_levels {
        let level_dimensions = crate::case::mips::level_dimensions(texture_type, dimensions, level);
        let len = crate::case::mips::literal_len(texture_type, level_dimensions, 1, texel)
            .ok_or_else(|| format!("{label} level {level} byte size overflows"))?;
        let level_bytes = if mip_levels == 1 {
            bytes
        } else {
            start
                .checked_add(len)
                .and_then(|end| bytes.get(start..end))
                .ok_or_else(|| {
                    format!(
                        "{label} supplies {} bytes, short of level {level}",
                        bytes.len()
                    )
                })?
        };
        upload_texture_level(
            &texture,
            label,
            texture_type,
            sample_count,
            format,
            LevelBytes {
                level,
                dimensions: level_dimensions,
                bytes: level_bytes,
            },
        )?;
        start += len;
    }
    Ok(Texture {
        object: texture,
        _backing: None,
        _heap: None,
    })
}

/// The page size of the oracle's sparse heaps: the smallest Metal offers, so the most regions are
/// whole tiles.
const SPARSE_PAGE_SIZE: MTLSparsePageSize = MTLSparsePageSize::Size16;

/// A sparse texture: made from its own sparse heap (private storage, the oracle's page size), with
/// exactly the case's mappings made, in order, by one resource-state encoder, and then the authored
/// bytes of exactly the mapped texels copied in by a blit, each in its own command buffer that
/// completes before the next is committed, so both complete before the work. A mapping this device
/// cannot make (`executor_contract::sparse::device_mappings` under its tile size and first level in
/// the mip tail) is refused before anything is mapped. One stderr line per texture records the
/// device's tiling (`metal-sparse-texture ...`, never recorded).
fn make_sparse_texture(
    device: &ProtocolObject<dyn MTLDevice>,
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    label: &str,
    usage: MTLTextureUsage,
    literal: &LiteralTexture,
) -> Result<Texture, String> {
    use crate::executor_contract::sparse::{
        device_mappings, literal_offset, mapped_texels, DeviceMapping, SparseTiling,
    };
    let (extent, slices) =
        crate::case::sparse::sparse_extent(literal.texture_type, literal.dimensions).ok_or_else(
            || {
                format!(
                    "{label}: a {:?} texture is not sparse",
                    literal.texture_type
                )
            },
        )?;
    let layout = literal.layout()?;
    let texture_type = metal_texture_type(literal.texture_type);
    let pixel_format = metal_pixel_format(literal.format);
    let descriptor = MTLTextureDescriptor::new();
    descriptor.setTextureType(texture_type);
    descriptor.setPixelFormat(pixel_format);
    unsafe {
        descriptor.setWidth(layout.width as usize);
        descriptor.setHeight(layout.height as usize);
        descriptor.setDepth(layout.depth as usize);
        descriptor.setArrayLength(layout.metal_array_length(literal.texture_type) as usize);
        descriptor.setSampleCount(1);
        descriptor.setMipmapLevelCount(crate::case::sparse::LITERAL_LEVELS as usize);
    }
    descriptor.setStorageMode(MTLStorageMode::Private);
    descriptor.setUsage(usage);
    // SAFETY: one sample, the only count a sparse texture of the case model has.
    let tile = unsafe {
        device.sparseTileSizeWithTextureType_pixelFormat_sampleCount_sparsePageSize(
            texture_type,
            pixel_format,
            1,
            SPARSE_PAGE_SIZE,
        )
    };
    let tile_bytes = device.sparseTileSizeInBytesForSparsePageSize(SPARSE_PAGE_SIZE);
    let tile = [tile.width, tile.height, tile.depth].map(|axis| axis as u32);
    if tile.contains(&0) || tile_bytes == 0 {
        return Err(format!(
            "{label}: the device has no sparse tile for {pixel_format:?} ({tile:?}, {tile_bytes} \
             bytes)"
        ));
    }
    // Room for every tile of every slice and one tile more per slice for its mip tail: no
    // mapping of the case can need more.
    let tiles_per_slice: usize = (0..3)
        .map(|axis| extent[axis].div_ceil(tile[axis]) as usize)
        .product();
    let heap_bytes = (tiles_per_slice + 1) * slices as usize * tile_bytes;
    let heap_descriptor = MTLHeapDescriptor::new();
    heap_descriptor.setType(MTLHeapType::Sparse);
    heap_descriptor.setStorageMode(MTLStorageMode::Private);
    heap_descriptor.setSparsePageSize(SPARSE_PAGE_SIZE);
    heap_descriptor.setSize(heap_bytes);
    let heap = device
        .newHeapWithDescriptor(&heap_descriptor)
        .ok_or_else(|| format!("create the sparse heap of {label} ({heap_bytes} bytes)"))?;
    let texture = heap
        .newTextureWithDescriptor(&descriptor)
        .ok_or_else(|| format!("create sparse {label} from its heap"))?;
    if !texture.isSparse() {
        return Err(format!("{label}: the heap's texture is not sparse"));
    }
    let tiling = SparseTiling {
        tile,
        first_level_in_tail: texture.firstMipmapInTail() as u32,
    };
    eprintln!(
        "metal-sparse-texture {label} format={pixel_format:?} tile={tile:?} \
         tile_bytes={tile_bytes} first_mipmap_in_tail={} tail_bytes={} heap_bytes={heap_bytes}",
        tiling.first_level_in_tail,
        texture.tailSizeInBytes()
    );
    let mappings = device_mappings(literal, tiling)?
        .ok_or_else(|| format!("{label} declares no sparse mappings"))?;
    if !mappings.is_empty() {
        let command = queue
            .commandBuffer()
            .ok_or_else(|| "MTLCommandQueue::commandBuffer returned nil".to_string())?;
        let encoder = command
            .resourceStateCommandEncoder()
            .ok_or_else(|| format!("create the resource-state encoder of {label}"))?;
        for mapping in &mappings {
            let (region, level, slice) = match *mapping {
                DeviceMapping::Tiles {
                    texels,
                    tile_origin,
                    tile_count,
                } => (
                    MTLRegion {
                        origin: MTLOrigin {
                            x: tile_origin[0] as usize,
                            y: tile_origin[1] as usize,
                            z: tile_origin[2] as usize,
                        },
                        size: mtl_size(tile_count),
                    },
                    texels.level,
                    texels.slice,
                ),
                // The tail is one unit: its first level's one tile maps every level in it.
                DeviceMapping::MipTail { slice } => (
                    MTLRegion {
                        origin: MTLOrigin { x: 0, y: 0, z: 0 },
                        size: mtl_size([1, 1, 1]),
                    },
                    tiling.first_level_in_tail,
                    slice,
                ),
            };
            // SAFETY: the texture is a live sparse texture of this device, the region lies in
            // its tiles (`device_mappings`), and the command buffer is waited for below.
            unsafe {
                encoder.updateTextureMapping_mode_region_mipLevel_slice(
                    &texture,
                    MTLSparseTextureMappingMode::Map,
                    region,
                    level as usize,
                    slice as usize,
                );
            }
        }
        encoder.endEncoding();
        commit_and_wait(&command, &format!("Metal sparse mapping of {label}"))?;
    }
    let regions = mapped_texels(literal, tiling, &mappings)?;
    if !regions.is_empty() {
        let pointer = NonNull::new(literal.bytes.as_ptr().cast_mut().cast::<c_void>())
            .ok_or_else(|| format!("{label} bytes pointer is null"))?;
        // SAFETY: `pointer` addresses `literal.bytes`, which outlives the call that copies them.
        let staging = unsafe {
            device.newBufferWithBytes_length_options(
                pointer,
                literal.bytes.len(),
                MTLResourceOptions::StorageModeShared,
            )
        }
        .ok_or_else(|| format!("create the staging buffer of {label}"))?;
        let bytes_per_row = extent[0] as usize * literal.format.bytes_per_pixel();
        let bytes_per_image = if literal.texture_type == TextureType::D3 {
            bytes_per_row * extent[1] as usize
        } else {
            0
        };
        let command = queue
            .commandBuffer()
            .ok_or_else(|| "MTLCommandQueue::commandBuffer returned nil".to_string())?;
        let blit = command
            .blitCommandEncoder()
            .ok_or_else(|| format!("create the upload encoder of {label}"))?;
        for region in &regions {
            let offset = literal_offset(literal, region)?;
            // SAFETY: the region lies in the literal's bytes (`literal_offset`) and in mapped
            // texels of the texture's level and slice; the command buffer is waited for below.
            unsafe {
                blit.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                    &staging,
                    offset as usize,
                    bytes_per_row,
                    bytes_per_image,
                    mtl_size(region.size),
                    &texture,
                    region.slice as usize,
                    region.level as usize,
                    MTLOrigin {
                        x: region.origin[0] as usize,
                        y: region.origin[1] as usize,
                        z: region.origin[2] as usize,
                    },
                );
            }
        }
        blit.endEncoding();
        commit_and_wait(&command, &format!("Metal sparse upload of {label}"))?;
    }
    Ok(Texture {
        object: texture,
        _backing: None,
        _heap: Some(heap),
    })
}

pub(super) fn upload_texture(
    texture: &ProtocolObject<dyn MTLTexture>,
    label: &str,
    texture_type: TextureType,
    dimensions: [u32; 3],
    sample_count: u32,
    format: TextureFormat,
    bytes: &[u8],
) -> Result<(), String> {
    upload_texture_level(
        texture,
        label,
        texture_type,
        sample_count,
        format,
        LevelBytes {
            level: 0,
            dimensions,
            bytes,
        },
    )
}

/// One level of a texture literal: its index, its extent in the case's spelling, and its bytes,
/// laid out as a one-level texture of that extent.
struct LevelBytes<'a> {
    level: u32,
    dimensions: [u32; 3],
    bytes: &'a [u8],
}

/// Write every slice of `level` into `texture`.
fn upload_texture_level(
    texture: &ProtocolObject<dyn MTLTexture>,
    label: &str,
    texture_type: TextureType,
    sample_count: u32,
    format: TextureFormat,
    level: LevelBytes<'_>,
) -> Result<(), String> {
    let LevelBytes {
        level,
        dimensions,
        bytes,
    } = level;
    let span = crate::literal::subresource_span(texture_type, [0, 0, 0], dimensions, sample_count)?;
    let bytes_per_row = span.size[0] as usize * format.bytes_per_pixel();
    let bytes_per_image = bytes_per_row * span.size[1] as usize;
    let bytes_per_slice = bytes_per_image * span.size[2] as usize;
    let region = MTLRegion {
        origin: MTLOrigin { x: 0, y: 0, z: 0 },
        size: MTLSize {
            width: span.size[0] as usize,
            height: span.size[1] as usize,
            depth: span.size[2] as usize,
        },
    };
    // The length check is also what makes the per-slice pointer arithmetic below sound, and it
    // is the only guard against a null base pointer now that the slice is known non-empty.
    let needed = bytes_per_slice * span.slice_count as usize;
    if bytes.len() < needed || needed == 0 {
        return Err(format!(
            "{label} supplies {} bytes but its extent needs {needed}",
            bytes.len()
        ));
    }
    for slice in 0..span.slice_count as usize {
        let slice_pointer = unsafe {
            NonNull::new_unchecked(bytes.as_ptr().add(slice * bytes_per_slice) as *mut c_void)
        };
        unsafe {
            texture.replaceRegion_mipmapLevel_slice_withBytes_bytesPerRow_bytesPerImage(
                region,
                level as usize,
                span.base_slice as usize + slice,
                slice_pointer,
                bytes_per_row,
                bytes_per_image,
            );
        }
    }
    Ok(())
}

pub(super) fn initialize_multisample_texture(
    device: &ProtocolObject<dyn MTLDevice>,
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    texture: &ProtocolObject<dyn MTLTexture>,
    label: &str,
    layout: crate::literal::TextureLayout,
    format: TextureFormat,
    bytes: &[u8],
) -> Result<(), String> {
    let (value_type, result_type, expression, depth) = match format {
        TextureFormat::R8Unorm => ("uchar", "float", "float(value) / 255.0f", false),
        TextureFormat::Rgba8Unorm => ("uchar4", "float4", "float4(value) / 255.0f", false),
        TextureFormat::Rgba8Uint => ("uchar4", "uint4", "uint4(value)", false),
        TextureFormat::Rgba8Sint => ("char4", "int4", "int4(value)", false),
        TextureFormat::R16Float => ("half", "half", "value", false),
        TextureFormat::R16Uint => ("ushort", "uint", "uint(value)", false),
        TextureFormat::Rg16Float => ("half2", "half2", "value", false),
        TextureFormat::Rg32Float => ("float2", "float2", "value", false),
        TextureFormat::Rgba16Float => ("half4", "half4", "value", false),
        TextureFormat::Rgba16Uint => ("ushort4", "uint4", "uint4(value)", false),
        TextureFormat::Rgba16Sint => ("short4", "int4", "int4(value)", false),
        TextureFormat::R32Uint => ("uint", "uint", "value", false),
        TextureFormat::R32Sint => ("int", "int", "value", false),
        TextureFormat::R32Float => ("float", "float", "value", false),
        TextureFormat::Rgba32Uint => ("uint4", "uint4", "value", false),
        TextureFormat::Rgba32Sint => ("int4", "int4", "value", false),
        TextureFormat::Rgba32Float => ("float4", "float4", "value", false),
        TextureFormat::Depth32Float => ("float", "DepthValue", "DepthValue { value }", true),
        TextureFormat::R8Snorm
        | TextureFormat::Rgba8Snorm
        | TextureFormat::R16Unorm
        | TextureFormat::Rgba16Unorm
        | TextureFormat::R16Snorm
        | TextureFormat::Rg16Snorm
        | TextureFormat::Rgba16Snorm
        | TextureFormat::Rgba8UnormSrgb
        | TextureFormat::Bgra8Unorm
        | TextureFormat::Bgra8UnormSrgb
        | TextureFormat::R8UnormSrgb
        | TextureFormat::Rg8Unorm
        | TextureFormat::Rg8UnormSrgb
        | TextureFormat::Rg8Snorm
        | TextureFormat::Rg16Unorm
        | TextureFormat::A8Unorm
        | TextureFormat::B5g6r5Unorm
        | TextureFormat::A1bgr5Unorm
        | TextureFormat::Abgr4Unorm
        | TextureFormat::Bgr5a1Unorm
        | TextureFormat::Rgb10a2Unorm
        | TextureFormat::Bgr10a2Unorm
        | TextureFormat::Rg11b10Float
        | TextureFormat::Rgb9e5Float => {
            return Err(format!(
                "{label}: {format:?} is a sampled-only format, which no case multisamples"
            ));
        }
        TextureFormat::Rg32Uint => {
            return Err(format!(
                "{label}: {format:?} is the ulong texture format, which no case multisamples"
            ));
        }
    };
    let depth_declaration = if depth {
        "struct DepthValue { float depth [[depth(any)]]; };\n"
    } else {
        ""
    };
    let source = format!(
        r#"
#include <metal_stdlib>
using namespace metal;
struct LiteralVertex {{ float4 position [[position]]; }};
{depth_declaration}vertex LiteralVertex metal2vulkan_literal_vertex(uint id [[vertex_id]]) {{
    const float2 positions[3] = {{ float2(-1.0, -1.0), float2(3.0, -1.0), float2(-1.0, 3.0) }};
    return LiteralVertex {{ float4(positions[id], 0.0, 1.0) }};
}}
fragment {result_type} metal2vulkan_literal_fragment(
    LiteralVertex input [[stage_in]], uint sample [[sample_id]],
    const device {value_type} *values [[buffer(0)]]) {{
    uint x = uint(input.position.x);
    uint y = uint(input.position.y);
    {value_type} value = values[(y * {width}u + x) * {samples}u + sample];
    return {expression};
}}
"#,
        width = layout.width,
        samples = layout.sample_count,
    );
    let library = compile::bounded("multisample initializer library compilation", || {
        device.newLibraryWithSource_options_error(&NSString::from_str(&source), None)
    })
    .map_err(|error| format!("compile multisample initializer for {label}: {error}"))?;
    let vertex = library
        .newFunctionWithName(&NSString::from_str("metal2vulkan_literal_vertex"))
        .ok_or_else(|| format!("multisample initializer for {label} has no vertex function"))?;
    let fragment = library
        .newFunctionWithName(&NSString::from_str("metal2vulkan_literal_fragment"))
        .ok_or_else(|| format!("multisample initializer for {label} has no fragment function"))?;
    let pipeline_descriptor = MTLRenderPipelineDescriptor::new();
    pipeline_descriptor.setVertexFunction(Some(&vertex));
    pipeline_descriptor.setFragmentFunction(Some(&fragment));
    pipeline_descriptor.setRasterSampleCount(layout.sample_count as usize);
    if depth {
        pipeline_descriptor.setDepthAttachmentPixelFormat(metal_pixel_format(format));
    } else {
        let attachment = unsafe {
            pipeline_descriptor
                .colorAttachments()
                .objectAtIndexedSubscript(0)
        };
        attachment.setPixelFormat(metal_pixel_format(format));
    }
    let pipeline = compile::bounded("multisample initializer pipeline creation", || {
        device.newRenderPipelineStateWithDescriptor_error(&pipeline_descriptor)
    })
    .map_err(|error| format!("create multisample initializer for {label}: {error}"))?;
    let pointer = NonNull::new(bytes.as_ptr().cast_mut().cast::<c_void>())
        .ok_or_else(|| format!("{label} bytes pointer is null"))?;
    let source_buffer = unsafe {
        device.newBufferWithBytes_length_options(
            pointer,
            bytes.len(),
            MTLResourceOptions::StorageModeShared,
        )
    }
    .ok_or_else(|| format!("create multisample source buffer for {label}"))?;
    let depth_state = if depth {
        let descriptor = MTLDepthStencilDescriptor::new();
        descriptor.setDepthWriteEnabled(true);
        descriptor.setDepthCompareFunction(MTLCompareFunction::Always);
        Some(
            device
                .newDepthStencilStateWithDescriptor(&descriptor)
                .ok_or_else(|| format!("create multisample depth state for {label}"))?,
        )
    } else {
        None
    };
    let command = queue
        .commandBuffer()
        .ok_or_else(|| "MTLCommandQueue::commandBuffer returned nil".to_string())?;
    let layer_bytes = layout.width as usize
        * layout.height as usize
        * layout.sample_count as usize
        * format.bytes_per_pixel();
    for layer in 0..layout.array_layers as usize {
        let pass = MTLRenderPassDescriptor::new();
        if depth {
            let attachment = pass.depthAttachment();
            attachment.setTexture(Some(texture));
            attachment.setSlice(layer);
            attachment.setLoadAction(MTLLoadAction::DontCare);
            attachment.setStoreAction(MTLStoreAction::Store);
        } else {
            let attachment = unsafe { pass.colorAttachments().objectAtIndexedSubscript(0) };
            attachment.setTexture(Some(texture));
            attachment.setSlice(layer);
            attachment.setLoadAction(MTLLoadAction::DontCare);
            attachment.setStoreAction(MTLStoreAction::Store);
        }
        let encoder = command
            .renderCommandEncoderWithDescriptor(&pass)
            .ok_or_else(|| format!("create multisample initializer encoder for {label}"))?;
        encoder.setRenderPipelineState(&pipeline);
        if let Some(state) = &depth_state {
            encoder.setDepthStencilState(Some(state));
        }
        unsafe {
            encoder.setFragmentBuffer_offset_atIndex(Some(&*source_buffer), layer * layer_bytes, 0);
            encoder.drawPrimitives_vertexStart_vertexCount(MTLPrimitiveType::Triangle, 0, 3);
        }
        encoder.endEncoding();
    }
    commit_and_wait(
        &command,
        &format!("Metal multisample initialization for {label}"),
    )
}

fn metal_texture_type(texture_type: TextureType) -> MTLTextureType {
    match texture_type {
        TextureType::Buffer => MTLTextureType::TypeTextureBuffer,
        TextureType::D1 => MTLTextureType::Type1D,
        TextureType::D1Array => MTLTextureType::Type1DArray,
        TextureType::D2 => MTLTextureType::Type2D,
        TextureType::D2Array => MTLTextureType::Type2DArray,
        TextureType::D2Multisample => MTLTextureType::Type2DMultisample,
        TextureType::D2MultisampleArray => MTLTextureType::Type2DMultisampleArray,
        TextureType::D3 => MTLTextureType::Type3D,
        TextureType::Cube => MTLTextureType::TypeCube,
        TextureType::CubeArray => MTLTextureType::TypeCubeArray,
    }
}

pub(super) fn metal_pixel_format(format: TextureFormat) -> MTLPixelFormat {
    match format {
        TextureFormat::R8Unorm => MTLPixelFormat::R8Unorm,
        TextureFormat::Rgba8Unorm => MTLPixelFormat::RGBA8Unorm,
        TextureFormat::Rgba8Uint => MTLPixelFormat::RGBA8Uint,
        TextureFormat::Rgba8Sint => MTLPixelFormat::RGBA8Sint,
        TextureFormat::R16Float => MTLPixelFormat::R16Float,
        TextureFormat::R16Uint => MTLPixelFormat::R16Uint,
        TextureFormat::Rg16Float => MTLPixelFormat::RG16Float,
        TextureFormat::Rg32Float => MTLPixelFormat::RG32Float,
        TextureFormat::Rgba16Float => MTLPixelFormat::RGBA16Float,
        TextureFormat::Rgba16Uint => MTLPixelFormat::RGBA16Uint,
        TextureFormat::Rgba16Sint => MTLPixelFormat::RGBA16Sint,
        TextureFormat::R32Uint => MTLPixelFormat::R32Uint,
        TextureFormat::R32Sint => MTLPixelFormat::R32Sint,
        TextureFormat::R32Float => MTLPixelFormat::R32Float,
        TextureFormat::Rgba32Uint => MTLPixelFormat::RGBA32Uint,
        TextureFormat::Rgba32Sint => MTLPixelFormat::RGBA32Sint,
        TextureFormat::Rgba32Float => MTLPixelFormat::RGBA32Float,
        TextureFormat::Depth32Float => MTLPixelFormat::Depth32Float,
        TextureFormat::Rg32Uint => MTLPixelFormat::RG32Uint,
        TextureFormat::R8Snorm => MTLPixelFormat::R8Snorm,
        TextureFormat::Rgba8Snorm => MTLPixelFormat::RGBA8Snorm,
        TextureFormat::R16Unorm => MTLPixelFormat::R16Unorm,
        TextureFormat::Rgba16Unorm => MTLPixelFormat::RGBA16Unorm,
        TextureFormat::R16Snorm => MTLPixelFormat::R16Snorm,
        TextureFormat::Rg16Snorm => MTLPixelFormat::RG16Snorm,
        TextureFormat::Rgba16Snorm => MTLPixelFormat::RGBA16Snorm,
        TextureFormat::Rgba8UnormSrgb => MTLPixelFormat::RGBA8Unorm_sRGB,
        TextureFormat::Bgra8Unorm => MTLPixelFormat::BGRA8Unorm,
        TextureFormat::Bgra8UnormSrgb => MTLPixelFormat::BGRA8Unorm_sRGB,
        TextureFormat::R8UnormSrgb => MTLPixelFormat::R8Unorm_sRGB,
        TextureFormat::Rg8Unorm => MTLPixelFormat::RG8Unorm,
        TextureFormat::Rg8UnormSrgb => MTLPixelFormat::RG8Unorm_sRGB,
        TextureFormat::Rg8Snorm => MTLPixelFormat::RG8Snorm,
        TextureFormat::Rg16Unorm => MTLPixelFormat::RG16Unorm,
        TextureFormat::A8Unorm => MTLPixelFormat::A8Unorm,
        TextureFormat::B5g6r5Unorm => MTLPixelFormat::B5G6R5Unorm,
        TextureFormat::A1bgr5Unorm => MTLPixelFormat::A1BGR5Unorm,
        TextureFormat::Abgr4Unorm => MTLPixelFormat::ABGR4Unorm,
        TextureFormat::Bgr5a1Unorm => MTLPixelFormat::BGR5A1Unorm,
        TextureFormat::Rgb10a2Unorm => MTLPixelFormat::RGB10A2Unorm,
        TextureFormat::Bgr10a2Unorm => MTLPixelFormat::BGR10A2Unorm,
        TextureFormat::Rg11b10Float => MTLPixelFormat::RG11B10Float,
        TextureFormat::Rgb9e5Float => MTLPixelFormat::RGB9E5Float,
    }
}
