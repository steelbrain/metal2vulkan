//! The members of an engine case's argument buffers, bound the same way by the compute and the
//! render executor.
//!
//! A case authors each member by its owner (the Metal buffer index of the argument buffer that
//! holds it) and the byte offset of its field in the owner. The Metal oracle creates the member,
//! writes it into the owner with its argument encoder (Tier 2), and makes it resident for reading
//! and writing (`useResource:usage:`, on a render encoder with `stages: Vertex | Fragment`). A run
//! through the runtime does the same: the owner, a buffer of the function's own stage, is created
//! holding at each field the 8 bytes the encoder writes there ([`owner_bytes`]): a buffer's
//! `gpuAddress`, a texture's or a sampler's `gpuResourceID`, as the runtime gives each.

use crate::case::ArgumentBufferSamplerResource;
use crate::literal::{LiteralArgumentBufferBuffer, LiteralArgumentBufferTexture, LiteralTexture};
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
/// cover its 8-byte field; a texture buffer among them, which a run through the runtime makes over
/// a buffer of its bytes (`platform::runtime_textures`). A key authored twice is refused.
pub fn embed_textures(
    buffers: &mut BTreeMap<u32, Vec<u8>>,
    literal: &[LiteralArgumentBufferTexture],
) -> Result<BTreeMap<FieldKey, LiteralTexture>, String> {
    let mut embedded = BTreeMap::new();
    for texture in literal {
        let key = (texture.buffer_binding, texture.field_offset);
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

/// The bytes the buffer at Metal index `index` of a function is created with: its initial bytes
/// (`buffers`) with each of its members' words (`fields`, by key) written at the member's field,
/// little endian. Two members whose fields overlap, and a field the owner
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
