//! The members of an engine case's argument buffers, bound the same way by the compute and the
//! render executor.
//!
//! A case authors each member by its owner (the Metal buffer index of the argument buffer that
//! holds it) and the byte offset of its field in the owner. The Metal oracle creates the member,
//! writes it into the owner with its argument encoder (Tier 2), and makes it resident for reading
//! and writing (`useResource:usage:`, on a render encoder with `stages: Vertex | Fragment`). The
//! executor does the same in the binder's terms: the owner, a buffer of the function's own stage,
//! is created holding at each field the 8 bytes the encoder writes there ([`field_words`],
//! [`owner_bytes`]), and each member is what the binder takes for it:
//!
//! - a buffer is its guest address in the owner and a used resource of class `Buffer`
//!   ([`used_buffer`]), so the resident set reaches it and a loaded pointer is translated through
//!   it;
//! - a texture is its `gpuResourceID` in the owner and a used resource of class `Texture` with
//!   that resource id ([`used_texture`]): a heap member, which the plan writes into every heap
//!   array whose class holds it and enters in each stage's handle table, so a handle loaded from
//!   the field finds its element;
//! - a sampler is its `gpuResourceID` in the owner and an argument sampler of the bind state with
//!   that id ([`embed_samplers`], [`sampler_ids`]): the plan writes it into each sampler array
//!   and enters it in each stage's sampler table, so a sampler loaded from the field finds its
//!   element.

use super::resources;
use super::BufferAddress;
use crate::case::{ArgumentBufferSamplerResource, TextureType};
use crate::literal::{LiteralArgumentBufferBuffer, LiteralArgumentBufferTexture, LiteralTexture};
use metal2vulkan::bind::{ArgumentSampler, BoundTexture, ResourceClass, UsedResource};
use metal2vulkan::metal::{RenderStages, ResourceUsage};
use std::collections::BTreeMap;

/// An argument-buffer member's key: its owner's Metal buffer index and its field's byte offset
/// in the owner.
pub type FieldKey = (u32, u32);

/// Bytes of one member's field: a `gpuAddress` or a `gpuResourceID`.
const FIELD_BYTES: u32 = 8;

/// Grow the owner of `key` among `buffers` with zeros to cover the member's 8-byte field (the
/// oracle's owner is padded to its declared size, which covers it). An owner that is not bound
/// and a field offset that overflows are refused.
fn cover(buffers: &mut BTreeMap<u32, Vec<u8>>, key: FieldKey, what: &str) -> Result<(), String> {
    let owner = buffers
        .get_mut(&key.0)
        .ok_or_else(|| format!("{what} {}+{} without its owner buffer", key.0, key.1))?;
    let end = key
        .1
        .checked_add(FIELD_BYTES)
        .and_then(|end| usize::try_from(end).ok())
        .ok_or("an argument-buffer field offset overflows")?;
    if owner.len() < end {
        owner.resize(end, 0);
    }
    Ok(())
}

/// The argument-buffer buffers `literal` of a function whose bound buffers are `buffers`: each
/// one's initial bytes by its key, each owner grown with zeros to cover its 8-byte field. A key
/// authored twice is refused.
pub fn embed(
    buffers: &mut BTreeMap<u32, Vec<u8>>,
    literal: &[LiteralArgumentBufferBuffer],
) -> Result<BTreeMap<FieldKey, Vec<u8>>, String> {
    let mut embedded = BTreeMap::new();
    for buffer in literal {
        let key = (buffer.buffer_binding, buffer.field_offset);
        cover(buffers, key, "argument-buffer buffer")?;
        if embedded.insert(key, buffer.bytes.clone()).is_some() {
            return Err(format!(
                "argument-buffer buffer {}+{} bound twice",
                key.0, key.1
            ));
        }
    }
    Ok(embedded)
}

/// The argument-buffer textures `literal` of a function whose bound buffers are `buffers`: each
/// one as a texture of its authored type, format, extent, sample count, and contents, by its key
/// (its `binding` is its owner's index, which only labels it), each owner grown with zeros to
/// cover its 8-byte field. A texture buffer is refused, as its case's requirement refuses it first
/// ([`TextureBufferMember`](crate::requirement::ToolingRequirement::TextureBufferMember)): the
/// binder's planned heap makes no texture buffer a member, so a handle naming one misses. A key
/// authored twice is refused.
pub fn embed_textures(
    buffers: &mut BTreeMap<u32, Vec<u8>>,
    literal: &[LiteralArgumentBufferTexture],
) -> Result<BTreeMap<FieldKey, LiteralTexture>, String> {
    let mut embedded = BTreeMap::new();
    for texture in literal {
        let key = (texture.buffer_binding, texture.field_offset);
        if texture.texture_type == TextureType::Buffer {
            return Err(format!(
                "{} is a texture buffer, which no heap array holds",
                texture.label()
            ));
        }
        cover(buffers, key, "argument-buffer texture")?;
        let bound = LiteralTexture {
            binding: texture.buffer_binding,
            role: texture.role,
            texture_type: texture.texture_type,
            format: texture.format,
            dimensions: texture.dimensions,
            sample_count: texture.sample_count,
            bytes: texture.bytes.clone(),
            sparse: None,
            mip_levels: 1,
        };
        if embedded.insert(key, bound).is_some() {
            return Err(format!("{} bound twice", texture.label()));
        }
    }
    Ok(embedded)
}

/// The argument-buffer samplers `literal` of a function whose bound buffers are `buffers`: each
/// one's authored descriptor by its key, each owner grown with zeros to cover its 8-byte field. A
/// key authored twice is refused.
pub fn embed_samplers(
    buffers: &mut BTreeMap<u32, Vec<u8>>,
    literal: &[ArgumentBufferSamplerResource],
) -> Result<BTreeMap<FieldKey, ArgumentBufferSamplerResource>, String> {
    let mut embedded = BTreeMap::new();
    for sampler in literal {
        let key = (sampler.buffer_binding, sampler.field_offset);
        cover(buffers, key, "argument-buffer sampler")?;
        if embedded.insert(key, sampler.clone()).is_some() {
            return Err(format!(
                "argument-buffer sampler {}+{} bound twice",
                key.0, key.1
            ));
        }
    }
    Ok(embedded)
}

/// The synthetic `gpuResourceID` of the used resource whose consumer token is `token`: nonzero
/// (a handle table never enters id 0, the null entry) and distinct per token.
pub fn resource_id(token: u32) -> u64 {
    (1 << 62) | (u64::from(token) + 1)
}

/// The synthetic `gpuResourceID` of the argument-buffer sampler at `position` among a case's:
/// nonzero (a sampler table never enters id 0, the null entry), distinct per position, and apart
/// from every texture's [`resource_id`].
pub fn sampler_id(position: usize) -> Result<u64, String> {
    u64::try_from(position)
        .ok()
        .and_then(|position| position.checked_add(1))
        .filter(|&id| id < 1 << 61)
        .map(|id| (1 << 61) | id)
        .ok_or_else(|| "too many argument-buffer samplers".into())
}

/// The [`sampler_id`] of each argument-buffer sampler `keys` (in order), by key.
pub fn sampler_ids<K: Ord>(keys: impl Iterator<Item = K>) -> Result<BTreeMap<K, u64>, String> {
    keys.enumerate()
        .map(|(position, key)| Ok((key, sampler_id(position)?)))
        .collect()
}

/// The bind state's argument samplers of `samplers`, in the order [`sampler_ids`] takes their keys:
/// each one's authored descriptor ([`resources::argument_sampler_descriptor`]) with its
/// [`sampler_id`].
pub fn argument_samplers<'a, K>(
    samplers: impl Iterator<Item = (K, &'a ArgumentBufferSamplerResource)>,
) -> Result<Vec<ArgumentSampler>, String> {
    samplers
        .enumerate()
        .map(|(position, (_, sampler))| {
            Ok(ArgumentSampler::new(
                resources::argument_sampler_descriptor(sampler),
                sampler_id(position)?,
            ))
        })
        .collect()
}

/// The consumer token of each argument-buffer texture `keys` (in order) of a case whose
/// argument-buffer buffers number `buffers`: its position after them, so the buffers keep theirs.
pub fn texture_tokens<K: Ord>(
    buffers: usize,
    keys: impl Iterator<Item = K>,
) -> Result<BTreeMap<K, u32>, String> {
    keys.enumerate()
        .map(|(position, key)| {
            let token = buffers
                .checked_add(position)
                .and_then(|token| u32::try_from(token).ok())
                .ok_or("too many argument-buffer members")?;
            Ok((key, token))
        })
        .collect()
}

/// The 8 bytes Metal's argument encoder writes at each member's field, as a little-endian word
/// by key: each argument-buffer buffer's guest address (`buffers`), each argument-buffer
/// texture's [`resource_id`] (`textures`, by consumer token), and each argument-buffer sampler's
/// id (`samplers`, [`sampler_ids`]). A key named by two members is refused.
pub fn field_words<K: Ord + Copy + std::fmt::Debug>(
    buffers: &BTreeMap<K, BufferAddress>,
    textures: &BTreeMap<K, u32>,
    samplers: &BTreeMap<K, u64>,
) -> Result<BTreeMap<K, u64>, String> {
    let mut words = BTreeMap::new();
    let members = buffers
        .iter()
        .map(|(&key, address)| (key, address.guest))
        .chain(
            textures
                .iter()
                .map(|(&key, &token)| (key, resource_id(token))),
        )
        .chain(samplers.iter().map(|(&key, &id)| (key, id)));
    for (key, word) in members {
        if words.insert(key, word).is_some() {
            return Err(format!("argument-buffer field {key:?} holds two members"));
        }
    }
    Ok(words)
}

/// The bytes the buffer at Metal index `index` of a function is created with: its initial bytes
/// (`buffers`) with each of its members' words (`fields`, by key, [`field_words`]) written at
/// the member's field, little endian. Two members whose fields overlap, and a field the owner
/// does not cover, are refused.
pub fn owner_bytes(
    buffers: &BTreeMap<u32, Vec<u8>>,
    index: u32,
    fields: &BTreeMap<FieldKey, u64>,
) -> Result<Vec<u8>, String> {
    let mut bytes = buffers
        .get(&index)
        .ok_or_else(|| format!("buffer {index} is not bound"))?
        .clone();
    // Keys sort by owner, then offset: each field must start at or past the previous one's end.
    let mut end = 0;
    for (&(owner, field), word) in fields.range((index, 0)..=(index, u32::MAX)) {
        if field < end {
            return Err(format!(
                "buffer {owner}'s argument-buffer fields overlap at {field}"
            ));
        }
        end = field.saturating_add(FIELD_BYTES);
        let at = field as usize;
        bytes
            .get_mut(at..at + FIELD_BYTES as usize)
            .ok_or_else(|| format!("buffer {owner} does not cover its field at {field}"))?
            .copy_from_slice(&word.to_le_bytes());
    }
    Ok(bytes)
}

/// The used resource of the argument-buffer buffer of `length` bytes at `address` whose consumer
/// token is `token` (its position among the bind state's argument-buffer buffers), resident for
/// reading and writing in `stages`, as the Metal oracle's `useResource` makes it: no stages on a
/// compute encoder, the vertex and fragment stages on a render encoder.
pub fn used_buffer(
    token: usize,
    length: usize,
    address: BufferAddress,
    stages: RenderStages,
) -> Result<UsedResource, String> {
    let token = u32::try_from(token).map_err(|_| "too many argument-buffer buffers")?;
    let mut used = UsedResource::new(ResourceClass::Buffer, token);
    used.gpu_address = Some(address.guest);
    used.device_address = address.host;
    used.length = length as u64;
    used.usage = ResourceUsage::READ | ResourceUsage::WRITE;
    used.stages = stages;
    Ok(used)
}

/// The used resource of the argument-buffer texture `texture` whose consumer token is `token`
/// ([`texture_tokens`]): a `Texture` of the facts a bound texture of it has
/// ([`resources::bound_texture`]), whose resource id is [`resource_id`] of its token, resident
/// for reading and writing in `stages` as the Metal oracle's `useResource` makes it.
pub fn used_texture(
    token: u32,
    texture: &LiteralTexture,
    stages: RenderStages,
) -> Result<UsedResource, String> {
    let BoundTexture::Texture { facts, .. } = resources::bound_texture(texture)? else {
        return Err(format!(
            "argument-buffer texture {token} is a texture buffer, which no heap array holds"
        ));
    };
    let mut used = UsedResource::new(ResourceClass::Texture(facts), token);
    used.usage = ResourceUsage::READ | ResourceUsage::WRITE;
    used.stages = stages;
    used.resource_id = Some(resource_id(token));
    Ok(used)
}

/// The guest address of each argument-buffer buffer of `keys` (in token order) on a device
/// without device addresses, by key: its [`BufferAddress::synthetic_used`].
pub fn synthetic_addresses<K: Ord>(
    keys: impl Iterator<Item = K>,
) -> Result<BTreeMap<K, BufferAddress>, String> {
    keys.enumerate()
        .map(|(token, key)| {
            let token = u32::try_from(token).map_err(|_| "too many argument-buffer buffers")?;
            Ok((key, BufferAddress::synthetic_used(token)))
        })
        .collect()
}
