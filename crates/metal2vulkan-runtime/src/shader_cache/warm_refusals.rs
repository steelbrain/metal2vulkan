//! The failures a warm pass's creations met that are facts of the record, the device, its driver
//! and this build (`warm/pass.rs`): kept in the build's directory, so a later pass over the same
//! directory answers the record from them at once rather than making its creation again.
//!
//! - **Why.** A kernel the device refuses (one past the stage bound its driver compiles intact,
//!   `Unsupported{StageSize}`, or past the private memory a thread of it holds,
//!   `Unsupported{PrivateMemory}`) is refused only once its translation has decoded, specialized,
//!   planned and emitted it: up to 1.4 s of a warm pass's work on the Iris Xe under ANV, made
//!   again at every boot whose consumer hands the pass the record again.
//! - **Named by what decides the failure.** A refusal's name is the SHA-256 of [`NAME_TAG`], the
//!   driver's identity (vendor, device, driver version and `pipelineCacheUUID`,
//!   `driver::Identity`) and the record's bytes (its descriptor, with its libraries by content
//!   hash), kept under the runtime's `RUNTIME_VERSION` and the device's profile fingerprint
//!   (`creation::record_name`) in the build's directory, which the translator's version,
//!   semantics and sources name (`this_build`). So another device or driver, another profile, a
//!   translator or a runtime of another version names it otherwise, finds none, and makes the
//!   creation again.
//! - **What it holds.** The kind of the failure, as the pass's warning names it
//!   (`warm::failure_kind`: the runtime's error and, for a translator refusal, the refused
//!   feature and what it needs), and nothing of the record's shaders: the pass counts and
//!   announces the record answered from it as one whose creation failed of that kind.
//! - **Which failures.** Only those the pass decides are permanent for the record on this device
//!   (`warm::pass`'s `kept_refusal`); a transient failure is never kept, so the next pass makes
//!   the creation again.
//!
//! ```text
//! magic        8 bytes   "M2VWRF1\0"
//! name         32 bytes  the refusal's name
//! kind length  u32
//! kind         UTF-8
//! checksum     32 bytes  SHA-256 of every byte before it
//! ```

use super::driver::Identity;
use super::{EntryKind, ShaderCache};
use metal2vulkan::cache::Hash256;
use metal2vulkan::device::Device as CompileDevice;
use sha2::{Digest, Sha256};

/// The tag a refusal's name hashes first.
const NAME_TAG: &[u8] = b"metal2vulkan-runtime warm refusal name v1\0";

/// The first bytes of a refusal's record.
const MAGIC: &[u8; 8] = b"M2VWRF1\0";

/// The longest kind a record holds: a refusal's feature and needs are far shorter.
const MOST_KIND: usize = 4096;

/// The name of the refusal of the record `record` on the device of the driver `identity`.
pub(crate) fn refusal_name(identity: &Identity, record: &[u8]) -> Hash256 {
    let mut hash = Sha256::new();
    hash.update(NAME_TAG);
    hash.update(identity.bytes());
    hash.update(record);
    hash.finalize().into()
}

/// The record of the refusal `name` of `kind` (the module docs' frame).
pub(crate) fn refusal_record(name: &Hash256, kind: &str) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(8 + 32 + 4 + kind.len() + 32);
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(name);
    bytes.extend_from_slice(&(kind.len() as u32).to_le_bytes());
    bytes.extend_from_slice(kind.as_bytes());
    let checksum: Hash256 = Sha256::digest(&bytes).into();
    bytes.extend_from_slice(&checksum);
    bytes
}

/// A refusal record's refusal: not a record of this kind, of another name, or damaged.
#[derive(Debug)]
pub(crate) struct NotTheRecord;

/// The kind the record `bytes` of the refusal `name` holds, refused unless they are exactly such
/// a record ([`refusal_record`]).
pub(crate) fn refusal_kind(bytes: &[u8], name: &Hash256) -> Result<String, NotTheRecord> {
    let head = 8 + 32 + 4;
    let sum = bytes.len().checked_sub(32).ok_or(NotTheRecord)?;
    if sum < head
        || &bytes[..8] != MAGIC
        || &bytes[8..40] != name
        || bytes[sum..] != <[u8; 32]>::from(Sha256::digest(&bytes[..sum]))
    {
        return Err(NotTheRecord);
    }
    let length = u32::from_le_bytes(bytes[40..head].try_into().map_err(|_| NotTheRecord)?);
    let length = usize::try_from(length).map_err(|_| NotTheRecord)?;
    if length > MOST_KIND || head + length != sum {
        return Err(NotTheRecord);
    }
    String::from_utf8(bytes[head..sum].to_vec()).map_err(|_| NotTheRecord)
}

impl ShaderCache {
    /// The kind of the failure an earlier warm pass over this directory kept for the record
    /// `record` on this device, driver, profile and build ([`keep_warm_refusal`]), if any: read
    /// as a creation's record is, counted as no lookup (it is no translation).
    ///
    /// [`keep_warm_refusal`]: Self::keep_warm_refusal
    pub(crate) fn warm_refusal(&self, compile: &CompileDevice, record: &[u8]) -> Option<String> {
        let name = refusal_name(&self.records.identity, record);
        self.read_answer(EntryKind::WarmRefusal, &name, compile, |bytes| {
            refusal_kind(bytes, &name)
        })
    }

    /// Keep `kind` as the failure the creation of the record `record` meets for good on this
    /// device, driver, profile and build, for a later warm pass to answer the record with.
    pub(crate) fn keep_warm_refusal(&self, compile: &CompileDevice, record: &[u8], kind: &str) {
        let name = refusal_name(&self.records.identity, record);
        self.keep_answer(EntryKind::WarmRefusal, &name, compile, || {
            refusal_record(&name, kind)
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A refusal's name moves with each part of the driver's identity and with the record, and its
    /// record gives its kind back under its own name only; a damaged or cut record is refused.
    #[test]
    fn a_refusal_names_its_driver_and_record_and_gives_its_kind_back() {
        let identity = Identity::new(0x8086, 0xa7a0, 7, [1; 16]);
        let record = b"a record's bytes";
        let name = refusal_name(&identity, record);
        for other in [
            Identity::new(0x8087, 0xa7a0, 7, [1; 16]),
            Identity::new(0x8086, 0xa7a1, 7, [1; 16]),
            Identity::new(0x8086, 0xa7a0, 8, [1; 16]),
            Identity::new(0x8086, 0xa7a0, 7, [2; 16]),
        ] {
            assert_ne!(refusal_name(&other, record), name, "{other:?}");
        }
        assert_ne!(refusal_name(&identity, b"another record"), name);
        let kind = "Unsupported{StageSize; DriverDefect(SerializedStageWrapsLargeModules)}";
        let bytes = refusal_record(&name, kind);
        assert_eq!(refusal_kind(&bytes, &name).unwrap(), kind);
        assert!(refusal_kind(&bytes, &refusal_name(&identity, b"another")).is_err());
        for at in [0, 8, 40, 44, bytes.len() - 33, bytes.len() - 1] {
            let mut damaged = bytes.clone();
            damaged[at] ^= 1;
            assert!(refusal_kind(&damaged, &name).is_err(), "byte {at}");
        }
        assert!(refusal_kind(&bytes[..bytes.len() - 1], &name).is_err());
        assert!(refusal_kind(&[], &name).is_err());
    }
}
