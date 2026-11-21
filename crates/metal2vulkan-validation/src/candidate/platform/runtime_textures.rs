//! The textures and samplers a run through the runtime binds or places in an argument buffer
//! ([`super::runtime`]), made as the Metal oracle makes them (`metal::platform::textures`,
//! `metal::platform::argument_buffers`).
//!
//! Each texture is a texture of its type, pixel format, extent, array length and levels
//! (`engine::resources`), Shared, with the usage its function declares of its slot or its
//! argument-buffer member (`RuntimeKernel::texture_usages`, `RuntimeKernel::member_usages`), at an
//! index of the run's texture grant (the bound textures first, then the members), and every
//! level's slices written from the case's bytes before the work (`replaceRegion:`, level 0
//! first), but for a 32-bit float depth's, which the run renders (`runtime_depth`), with the
//! texture made for rendering too. Each sampler is the oracle's descriptor of it (`engine::resources::sampler_descriptor`,
//! `argument_sampler_descriptor`), at an index of the run's sampler grant counted down from its
//! top, so it never meets the constexpr samplers the pipeline numbers up from the bottom. A
//! member's `gpuResourceID` is the word its owner holds at its field ([`Bound::words`]), as a
//! program writes one into an argument buffer. A texture output is read back as the oracle reads
//! it (`getBytes:`): the selected region of level 0, slice by slice.

use super::runtime_depth;
use crate::candidate::engine::argument_buffers::FieldKey;
use crate::candidate::engine::{resources, ComputeCase, RuntimeKernel};
use crate::case::{ArgumentBufferSamplerResource, SamplerResource, TextureFormat};
use crate::check::admission::ExecutorError;
use crate::gpu::DeviceType;
use crate::literal::LiteralTexture;
use metal2vulkan::metal::TextureUsage;
use metal2vulkan_runtime as rt;
use std::collections::BTreeMap;

/// The textures and samplers of a run: the bound ones by the Metal index each binds at, the
/// argument-buffer members by their owner's index and field offset, and the grant their texture
/// indices are taken from.
pub(super) struct Bound {
    pub(super) textures: BTreeMap<u32, rt::Texture>,
    pub(super) samplers: BTreeMap<u32, rt::SamplerState>,
    pub(super) members: BTreeMap<FieldKey, rt::Texture>,
    member_samplers: BTreeMap<FieldKey, rt::SamplerState>,
    _grant: Option<rt::IndexGrant>,
}

impl Bound {
    /// The `gpuResourceID` of each argument-buffer texture and sampler, by its field.
    pub(super) fn words(&self) -> BTreeMap<FieldKey, u64> {
        let textures = self
            .members
            .iter()
            .map(|(&key, texture)| (key, texture.gpu_resource_id().to_bits()));
        let samplers = self
            .member_samplers
            .iter()
            .map(|(&key, sampler)| (key, sampler.gpu_resource_id().to_bits()));
        textures.chain(samplers).collect()
    }
}

/// What one function of a run binds of textures and samplers, bound and in argument buffers,
/// with the usage it declares of each texture.
pub(super) struct Wanted<'a> {
    pub(super) textures: &'a BTreeMap<u32, LiteralTexture>,
    pub(super) samplers: &'a BTreeMap<u32, SamplerResource>,
    pub(super) members: &'a BTreeMap<FieldKey, LiteralTexture>,
    pub(super) member_samplers: &'a BTreeMap<FieldKey, ArgumentBufferSamplerResource>,
    pub(super) texture_usages: &'a BTreeMap<u32, TextureUsage>,
    pub(super) member_usages: &'a BTreeMap<FieldKey, TextureUsage>,
}

impl<'a> Wanted<'a> {
    /// A kernel case's, with the usages `kernel` declares.
    pub(super) fn of_kernel(case: &'a ComputeCase, kernel: &'a RuntimeKernel<'a>) -> Wanted<'a> {
        Wanted {
            textures: &case.textures,
            samplers: &case.samplers,
            members: &case.argument_buffer_textures,
            member_samplers: &case.argument_buffer_samplers,
            texture_usages: kernel.texture_usages,
            member_usages: kernel.member_usages,
        }
    }
}

/// The indices a run's next textures and samplers take: textures up from the bottom of the
/// texture class, samplers down from the top of the sampler class.
pub(super) struct Indices {
    texture: u32,
    sampler: u32,
}

impl Indices {
    /// A run's first.
    pub(super) fn new() -> Indices {
        Indices {
            texture: rt::IndexClass::Texture.range().start,
            sampler: rt::IndexClass::Sampler.range().end,
        }
    }

    fn sampler(&mut self) -> Result<u32, ExecutorError> {
        match self.sampler.checked_sub(1) {
            Some(index) if index >= rt::IndexClass::Sampler.range().start => {
                self.sampler = index;
                Ok(index)
            }
            _ => Err(ExecutorError::Failed(
                "the case binds more samplers than the sampler class holds".into(),
            )),
        }
    }
}

/// A grant of `count` texture indices of `device`'s at the next of `indices`, none for no
/// texture, and the indices it holds: each texture a shader reads or writes and each color
/// attachment takes one.
pub(super) fn texture_indices(
    device: &rt::Device,
    indices: &mut Indices,
    count: usize,
    error: &impl Fn(&str, rt::Error) -> ExecutorError,
) -> Result<(Option<rt::IndexGrant>, std::ops::Range<u32>), ExecutorError> {
    let first = indices.texture;
    let end = u32::try_from(count)
        .ok()
        .and_then(|count| first.checked_add(count))
        .ok_or_else(|| {
            ExecutorError::Failed("the case binds more textures than an index grant holds".into())
        })?;
    let grant = match end - first {
        0 => None,
        count => Some(
            device
                .grant_indices(rt::IndexClass::Texture, first, count)
                .map_err(|e| error("grant the texture indices", e))?,
        ),
    };
    indices.texture = end;
    Ok((grant, first..end))
}

/// Make `wanted`'s textures and samplers on `device` at the next of `indices`, as the module
/// documentation says. The run already holds its grant of the whole sampler range; `error` maps
/// the runtime's errors as the run's.
pub(super) fn make(
    device: &rt::Device,
    device_type: DeviceType,
    wanted: &Wanted<'_>,
    indices: &mut Indices,
    error: impl Fn(&str, rt::Error) -> ExecutorError,
) -> Result<Bound, ExecutorError> {
    let count = wanted.textures.len() + wanted.members.len();
    let (grant, mut at) = texture_indices(device, indices, count, &error)?;
    let mut fills = Vec::new();
    let mut textures = BTreeMap::new();
    for ((&binding, texture), index) in wanted.textures.iter().zip(&mut at) {
        let usage = wanted
            .texture_usages
            .get(&binding)
            .copied()
            .ok_or_else(|| {
                ExecutorError::Failed(format!("texture {binding} has no declared usage"))
            })?;
        textures.insert(
            binding,
            made_texture(device, texture, usage, index, &error, &mut fills)?,
        );
    }
    let mut members = BTreeMap::new();
    for ((&key, texture), index) in wanted.members.iter().zip(&mut at) {
        let usage = wanted.member_usages.get(&key).copied().ok_or_else(|| {
            ExecutorError::Failed(format!(
                "argument-buffer texture {}+{} has no declared usage",
                key.0, key.1
            ))
        })?;
        let made = made_texture(device, texture, usage, index, &error, &mut fills)?;
        members.insert(key, made);
    }
    if !fills.is_empty() {
        runtime_depth::filled(device, device_type, &fills, &error)?;
    }
    let mut samplers = BTreeMap::new();
    for (&binding, sampler) in wanted.samplers {
        let state = device
            .new_sampler_state(&resources::sampler_descriptor(sampler), indices.sampler()?)
            .map_err(|e| error(&format!("sampler {binding}"), e))?;
        samplers.insert(binding, state);
    }
    let mut member_samplers = BTreeMap::new();
    for (&key, sampler) in wanted.member_samplers {
        let descriptor = resources::argument_sampler_descriptor(sampler);
        let state = device
            .new_sampler_state(&descriptor, indices.sampler()?)
            .map_err(|e| error(&format!("argument-buffer sampler {}+{}", key.0, key.1), e))?;
        member_samplers.insert(key, state);
    }
    Ok(Bound {
        textures,
        samplers,
        members,
        member_samplers,
        _grant: grant,
    })
}

/// `texture` made with `usage` at `index`, every level's slices written from its bytes; a 32-bit
/// float depth's are pushed to `fills` instead, which the run renders (`runtime_depth`).
fn made_texture(
    device: &rt::Device,
    texture: &LiteralTexture,
    usage: rt::TextureUsage,
    index: u32,
    error: &impl Fn(&str, rt::Error) -> ExecutorError,
    fills: &mut Vec<runtime_depth::Fill>,
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
    // A 32-bit float depth is filled by a draw, which renders into the texture (`runtime_depth`).
    let depth = texture.format == TextureFormat::Depth32Float;
    descriptor.usage = match depth {
        true => usage | rt::TextureUsage::RENDER_TARGET,
        false => usage,
    };
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
            let slice = span.base_slice + slice as u32;
            if depth {
                fills.push(runtime_depth::Fill {
                    texture: made.clone(),
                    level,
                    slice,
                    width: span.size[0],
                    depths: chunk.to_vec(),
                });
                continue;
            }
            made.replace_region(
                rt::Region {
                    origin: [0, 0, 0],
                    size: span.size,
                },
                level,
                slice,
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
