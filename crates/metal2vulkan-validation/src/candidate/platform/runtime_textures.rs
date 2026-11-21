//! The textures and samplers a run through the runtime binds ([`super::runtime`]), made as the
//! Metal oracle makes them (`metal::platform::textures`).
//!
//! Each bound texture is a texture of its type, pixel format, extent, array length and levels
//! (`engine::resources`), Shared, with the usage its function declares of its slot
//! (`RuntimeKernel::texture_usages`), at an index of the run's texture grant, and every level's
//! slices written from the case's bytes before the work (`replaceRegion:`, level 0 first). Each
//! bound sampler is the oracle's descriptor of it (`engine::resources::sampler_descriptor`), at an
//! index of the run's sampler grant counted down from its top, so it never meets the constexpr
//! samplers the pipeline numbers up from the bottom. A texture output is read back as the oracle
//! reads it (`getBytes:`): the selected region of level 0, slice by slice.

use crate::candidate::engine::{resources, ComputeCase, RuntimeKernel};
use crate::check::admission::ExecutorError;
use crate::literal::LiteralTexture;
use metal2vulkan_runtime as rt;
use std::collections::BTreeMap;

/// The textures and samplers of a run, by the Metal index each binds at, and the grant their
/// texture indices are taken from.
pub(super) struct Bound {
    pub(super) textures: BTreeMap<u32, rt::Texture>,
    pub(super) samplers: BTreeMap<u32, rt::SamplerState>,
    _grant: Option<rt::IndexGrant>,
}

/// Make `case`'s bound textures and samplers on `device`, with the usages `kernel` declares, as
/// the module documentation says. The run already holds its grant of the whole sampler range;
/// `error` maps the runtime's errors as the run's.
pub(super) fn make(
    device: &rt::Device,
    case: &ComputeCase,
    kernel: &RuntimeKernel<'_>,
    error: impl Fn(&str, rt::Error) -> ExecutorError,
) -> Result<Bound, ExecutorError> {
    let first = rt::IndexClass::Texture.range().start;
    let grant = match u32::try_from(case.textures.len()) {
        Ok(0) => None,
        Ok(count) => Some(
            device
                .grant_indices(rt::IndexClass::Texture, first, count)
                .map_err(|e| error("grant the texture indices", e))?,
        ),
        Err(_) => {
            return Err(ExecutorError::Failed(
                "the case binds more textures than an index grant holds".into(),
            ))
        }
    };
    let mut textures = BTreeMap::new();
    for (index, (&binding, texture)) in (first..).zip(&case.textures) {
        let usage = kernel
            .texture_usages
            .get(&binding)
            .copied()
            .ok_or_else(|| {
                ExecutorError::Failed(format!("texture {binding} has no declared usage"))
            })?;
        textures.insert(
            binding,
            made_texture(device, texture, usage, index, &error)?,
        );
    }
    let mut samplers = BTreeMap::new();
    for (index, (&binding, sampler)) in rt::IndexClass::Sampler.range().rev().zip(&case.samplers) {
        let state = device
            .new_sampler_state(&resources::sampler_descriptor(sampler), index)
            .map_err(|e| error(&format!("sampler {binding}"), e))?;
        samplers.insert(binding, state);
    }
    Ok(Bound {
        textures,
        samplers,
        _grant: grant,
    })
}

/// `texture` made with `usage` at `index`, every level's slices written from its bytes.
fn made_texture(
    device: &rt::Device,
    texture: &LiteralTexture,
    usage: rt::TextureUsage,
    index: u32,
    error: &impl Fn(&str, rt::Error) -> ExecutorError,
) -> Result<rt::Texture, ExecutorError> {
    let what = format!("texture {}", texture.binding);
    let layout = texture.layout().map_err(ExecutorError::Failed)?;
    let mut descriptor = rt::TextureDescriptor::new(
        resources::texture_type(texture.texture_type),
        resources::pixel_format(texture.format),
        layout.width,
        layout.height,
    );
    descriptor.depth = layout.depth;
    descriptor.array_length = layout.metal_array_length(texture.texture_type);
    descriptor.sample_count = layout.sample_count;
    descriptor.mipmap_level_count = texture.mip_levels;
    descriptor.usage = usage;
    descriptor.storage_mode = rt::StorageMode::Shared;
    let made = device
        .new_texture(&descriptor, Some(index))
        .map_err(|e| error(&what, e))?;
    for level in 0..texture.mip_levels {
        let (dimensions, bytes) = texture.level(level).map_err(ExecutorError::Failed)?;
        let span = crate::literal::subresource_span(
            texture.texture_type,
            [0, 0, 0],
            dimensions,
            texture.sample_count,
        )
        .map_err(ExecutorError::Failed)?;
        let rows = Rows::of(texture, span.size);
        if bytes.len() < rows.slice * span.slice_count as usize || rows.slice == 0 {
            return Err(ExecutorError::Failed(format!(
                "{what} supplies {} bytes, short of its level {level}",
                bytes.len()
            )));
        }
        for (slice, chunk) in bytes
            .chunks_exact(rows.slice)
            .take(span.slice_count as usize)
            .enumerate()
        {
            made.replace_region(
                rt::Region {
                    origin: [0, 0, 0],
                    size: span.size,
                },
                level,
                span.base_slice + slice as u32,
                chunk.into(),
                rows.row as u64,
                rows.image as u64,
            )
            .map_err(|e| error(&format!("{what} level {level}"), e))?;
        }
    }
    Ok(made)
}

/// The packing of one slice of a region of `size` texels of a texture, as the case's literal and
/// the oracle's reads lay it out: tightly packed rows, images and slices.
struct Rows {
    row: usize,
    image: usize,
    slice: usize,
}

impl Rows {
    fn of(texture: &LiteralTexture, size: [u32; 3]) -> Rows {
        let row = size[0] as usize * texture.format.bytes_per_pixel();
        let image = row * size[1] as usize;
        Rows {
            row,
            image,
            slice: image * size[2] as usize,
        }
    }
}

/// The `(origin, dimensions)` selection of level 0 of `made`, the run's texture of `texture`,
/// in the case's spelling, read back slice by slice as the oracle reads a texture output.
pub(super) fn read_back(
    made: &rt::Texture,
    texture: &LiteralTexture,
    origin: [u32; 3],
    dimensions: [u32; 3],
    error: impl Fn(&str, rt::Error) -> ExecutorError,
) -> Result<Vec<u8>, ExecutorError> {
    if texture.sample_count != 1 {
        return Err(ExecutorError::Failed(format!(
            "texture {} is multisampled; its samples are not an observation",
            texture.binding
        )));
    }
    let span = crate::literal::subresource_span(texture.texture_type, origin, dimensions, 1)
        .map_err(ExecutorError::Failed)?;
    let rows = Rows::of(texture, span.size);
    let mut output = vec![0u8; rows.slice * span.slice_count as usize];
    for (slice, chunk) in output.chunks_exact_mut(rows.slice.max(1)).enumerate() {
        made.get_bytes(
            chunk,
            rows.row as u64,
            rows.image as u64,
            rt::Region {
                origin: span.origin,
                size: span.size,
            },
            0,
            span.base_slice + slice as u32,
        )
        .map_err(|e| error(&format!("texture {} read-back", texture.binding), e))?;
    }
    Ok(output)
}
