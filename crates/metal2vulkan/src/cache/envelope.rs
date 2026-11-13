//! The self-checking frame around an encoded artifact.
//!
//! ```text
//! magic        8 bytes   "M2VARTF\0"
//! format       u32       ARTIFACT_FORMAT_VERSION
//! semantics    u32       SEMANTICS_VERSION
//! kind         u32       what the body encodes (function, compute or render pipeline, a render
//!                        descriptor's linked-use answers or reflection)
//! key          32 bytes  the artifact's key (FunctionCompileKey or PipelineKey digest; a render
//!                        descriptor's key for its answers and reflection)
//! profile      32 bytes  the fingerprint of the profile it was compiled for
//! body length  u64
//! body         the `codec` encoding of the artifact
//! checksum     u64       XXH64 (seed 0) of every byte before it (`checksum`)
//! ```
//!
//! Integers are little-endian. [`open`] checks, in order: magic and both versions
//! ([`ArtifactError::Stale`]), length and checksum ([`ArtifactError::Corrupt`]), the kind
//! (`Corrupt`), and the profile fingerprint against the loading device
//! ([`ArtifactError::WrongDevice`]).

use crate::device::Device;

use super::checksum::{self, Checksum};
use super::{ArtifactError, Hash256, ARTIFACT_FORMAT_VERSION, SEMANTICS_VERSION};

const MAGIC: [u8; 8] = *b"M2VARTF\0";

/// Bytes before the body.
const HEADER: usize = 8 + 4 + 4 + 4 + 32 + 32 + 8;
/// Bytes after the body.
const CHECKSUM: usize = 8;

/// What an artifact's body encodes. A mismatch is `Corrupt`: the bytes are not the artifact the
/// caller asked for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub(crate) enum Kind {
    Function = 1,
    ComputePipeline = 2,
    RenderPipeline = 3,
    /// A render descriptor's `LinkedUseAnswers`, under its descriptor key.
    LinkedUseAnswers = 4,
    /// A render descriptor's `RenderPipelineReflection`, under its descriptor key.
    RenderReflection = 5,
}

/// A verified frame: the key it names and its body.
pub(crate) struct Opened<'a> {
    pub key: Hash256,
    pub body: &'a [u8],
}

/// Append the header of a frame whose body is `body_len` bytes to `out`.
fn put_header(out: &mut Vec<u8>, kind: Kind, key: &Hash256, profile: &Hash256, body_len: u64) {
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&ARTIFACT_FORMAT_VERSION.to_le_bytes());
    out.extend_from_slice(&SEMANTICS_VERSION.to_le_bytes());
    out.extend_from_slice(&(kind as u32).to_le_bytes());
    out.extend_from_slice(key);
    out.extend_from_slice(profile);
    out.extend_from_slice(&body_len.to_le_bytes());
}

/// Frame `body`.
pub(crate) fn seal(kind: Kind, key: &Hash256, profile: &Hash256, body: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(HEADER + body.len() + CHECKSUM);
    put_header(&mut out, kind, key, profile, body.len() as u64);
    out.extend_from_slice(body);
    let checksum = checksum::of(&out);
    out.extend_from_slice(&checksum.to_le_bytes());
    out
}

/// Frame the body `body` hands its sink into `out`, a chunk at a time: the bytes [`seal`] gives
/// for that body, never held whole. `body` runs twice, first to count the bytes for the header,
/// then to write them, so it must hand over the same bytes both times.
pub(crate) fn seal_to(
    kind: Kind,
    key: &Hash256,
    profile: &Hash256,
    body: &dyn Fn(&mut dyn FnMut(&[u8])),
    out: &mut dyn FnMut(&[u8]),
) {
    let mut body_len = 0u64;
    body(&mut |chunk| body_len = body_len.saturating_add(chunk.len() as u64));
    let mut header = Vec::with_capacity(HEADER);
    put_header(&mut header, kind, key, profile, body_len);
    let mut checksum = Checksum::default();
    let mut emit = |chunk: &[u8]| {
        checksum.update(chunk);
        out(chunk);
    };
    emit(&header);
    body(&mut emit);
    out(&checksum.finish().to_le_bytes());
}

fn u32_at(bytes: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_le_bytes(bytes.get(at..at + 4)?.try_into().ok()?))
}

fn hash_at(bytes: &[u8], at: usize) -> Option<Hash256> {
    bytes.get(at..at + 32)?.try_into().ok()
}

/// Check the frame of `bytes` for an artifact of `kind` compiled for `device`'s profile.
pub(crate) fn open<'a>(
    bytes: &'a [u8],
    kind: Kind,
    device: &Device,
) -> Result<Opened<'a>, ArtifactError> {
    // Identity of the encoding: anything else is another format or another semantics.
    if bytes.get(..8) != Some(&MAGIC[..])
        || u32_at(bytes, 8) != Some(ARTIFACT_FORMAT_VERSION)
        || u32_at(bytes, 12) != Some(SEMANTICS_VERSION)
    {
        return Err(ArtifactError::Stale);
    }
    // Integrity: the declared length and the checksum over everything before it.
    let body_len = bytes
        .get(HEADER - 8..HEADER)
        .and_then(|b| b.try_into().ok())
        .map(u64::from_le_bytes)
        .ok_or(ArtifactError::Corrupt)?;
    let end = usize::try_from(body_len)
        .ok()
        .and_then(|n| n.checked_add(HEADER))
        .ok_or(ArtifactError::Corrupt)?;
    if bytes.len() != end.checked_add(CHECKSUM).ok_or(ArtifactError::Corrupt)? {
        return Err(ArtifactError::Corrupt);
    }
    if bytes[end..] != checksum::of(&bytes[..end]).to_le_bytes() {
        return Err(ArtifactError::Corrupt);
    }
    if u32_at(bytes, 16) != Some(kind as u32) {
        return Err(ArtifactError::Corrupt);
    }
    let key = hash_at(bytes, 20).ok_or(ArtifactError::Corrupt)?;
    let profile = hash_at(bytes, 52).ok_or(ArtifactError::Corrupt)?;
    // Trust: only bytes compiled for this device's profile load.
    if profile != device.profile_fingerprint() {
        return Err(ArtifactError::WrongDevice);
    }
    Ok(Opened {
        key,
        body: &bytes[HEADER..end],
    })
}
