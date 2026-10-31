//! Metallib container layer: MTLB, bitcode wrapper, raw bitcode.
//!
//! [`parse`] accepts exactly these forms, told apart by the first four bytes (no scanning for a
//! magic inside the input):
//!
//! | Bytes | Form | Result |
//! |---|---|---|
//! | `MTLB` | Metal library, one bitcode module per function | [`Library::Metallib`] |
//! | `CB FE BA BE` | Metal's own universal metallib: slices for the AIR and for GPUs | [`Library::Metallib`] of its AIR slice |
//! | `CA FE BA BE`, `CA FE BA BF` | a Mach-O universal file of the same slices | [`Library::Metallib`] of its AIR slice |
//! | `DE C0 17 0B` | bitcode wrapper (`0x0b17c0de`, an `.air` file) | [`Library::Module`] |
//! | `BC C0 DE` | raw bitcode stream | [`Library::Module`] |
//!
//! A universal metallib holds its slices behind a big-endian header: the magic, a `u32` slice
//! count, and a record per slice, in one of three headers:
//!
//! - **Metal's own**, what `MTLDynamicLibrary`'s `serializeToURL:error:` writes (a probe, an M3 Max
//!   and an M4): the magic `0xCBFEBABE`, and per slice five big-endian `u32`s (CPU type, subtype,
//!   offset, size, alignment).
//! - **A Mach-O universal file** (`<mach-o/fat.h>`), the form of the system's universal
//!   metallibs: [`FAT_MAGIC`] with the same records (`fat_arch`), or [`FAT_MAGIC_64`], whose
//!   records (`fat_arch_64`) hold the offset and size as big-endian `u64`s, then the alignment and
//!   a reserved `u32`.
//!
//! The slice of CPU type `0x01000017` is the library's AIR, a metallib, which is then the result's
//! [`Metallib::bytes`]; the others hold code compiled for one GPU, which is not read. Metal's own
//! file without an AIR slice is `Container(NoAirSlice)`, as Metal refuses it ("its AIR slice is
//! missing"). A Mach-O universal file is a metallib only through its AIR slice, since files that
//! are no metallib share its magic (a CPU binary, a Java class): one whose records do not fit it,
//! or that holds no AIR slice, is `Container(BadMagic)`, as any bytes that are no metallib are.
//!
//! A metallib's functions are enumerated from its function table, every known tag is validated
//! strictly, unknown tags are kept as raw tags (translation semantics come from the bitcode and
//! AIR metadata, never from container tags), and every function's `HASH` is verified against its
//! bitcode (linear: function bitcode ranges must be disjoint). Selecting a function by name is the
//! Metal contract (`newFunctionWithName:`), not a name-keyed heuristic.
//!
//! A wrapper or raw module may hold several entry points; enumerating them needs its AIR metadata,
//! which is decoded by `air::bitcode`/`air::abi`, so this layer only locates the stream.
//!
//! Parsing is cheap and bounded by the input: every offset and size is checked with checked
//! arithmetic against the slice, and no allocation is sized from a count without a bound derived
//! from the remaining bytes.

mod mtlb;
mod tags;
#[cfg(test)]
mod tests;

#[cfg(test)]
pub use mtlb::VertexAttribute;
pub use mtlb::{
    ContainerConstant, DynamicLinkage, FunctionPublicMetadata, LibraryFunction, LibraryType,
    Metallib,
};
#[cfg(test)]
pub use tags::RawTag;

use sha2::{Digest, Sha256};

use crate::error::Error;
use crate::error::{ContainerError, InputError, InputErrorKind, Location};

/// The first four bytes of a metallib.
pub const MTLB_MAGIC: [u8; 4] = *b"MTLB";
/// The bitcode wrapper magic `0x0b17c0de`, little-endian.
pub const WRAPPER_MAGIC: [u8; 4] = [0xDE, 0xC0, 0x17, 0x0B];
/// The LLVM bitcode magic `BC 0xC0DE`.
pub const BITCODE_MAGIC: [u8; 4] = [b'B', b'C', 0xC0, 0xDE];
/// The first four bytes of Metal's own universal metallib (`0xCBFEBABE`, big-endian).
pub const UNIVERSAL_MAGIC: [u8; 4] = [0xCB, 0xFE, 0xBA, 0xBE];
/// The first four bytes of a Mach-O universal file (`FAT_MAGIC`, `0xCAFEBABE`, big-endian).
pub const FAT_MAGIC: [u8; 4] = [0xCA, 0xFE, 0xBA, 0xBE];
/// The first four bytes of a Mach-O universal file of 64-bit records (`FAT_MAGIC_64`,
/// `0xCAFEBABF`, big-endian).
pub const FAT_MAGIC_64: [u8; 4] = [0xCA, 0xFE, 0xBA, 0xBF];
/// The CPU type of a universal metallib's AIR slice.
const AIR_SLICE_CPU_TYPE: u32 = 0x0100_0017;
/// The bytes of a universal metallib's header.
const UNIVERSAL_HEADER_LEN: usize = 8;
/// Size of the bitcode wrapper header.
const WRAPPER_HEADER_LEN: usize = 20;

/// A SHA-256 digest.
pub type Hash256 = [u8; 32];

/// A parsed container.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Library<'a> {
    /// A metallib with its function table.
    Metallib(Metallib<'a>),
    /// A single bitcode module (wrapper or raw).
    Module(Module<'a>),
}

/// A bitcode module given directly, wrapped or raw.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Module<'a> {
    /// The wrapper header, when the input was wrapped.
    pub wrapper: Option<WrapperHeader>,
    /// The whole input.
    pub bytes: &'a [u8],
    /// The bitcode stream (starts with [`BITCODE_MAGIC`]).
    pub stream: &'a [u8],
    /// SHA-256 of the whole input, as a metallib `HASH` covers a function's wrapped bitcode.
    pub hash: Hash256,
}

/// The bitcode wrapper header. Only the stream range is used; the rest is recorded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WrapperHeader {
    /// The wrapper's version field, as read.
    pub version: u32,
    /// The byte offset of the bitcode stream from the start of the input.
    pub offset: u32,
    /// The bitcode stream's size in bytes.
    pub size: u32,
    /// The wrapper's target CPU type field, as read.
    pub cputype: u32,
}

/// The pipeline stage a metallib `TYPE` tag names. Values not observed in Apple output are kept as
/// `Other`; AIR metadata is the authority, this is cross-checked against it. The observed values,
/// with the `MTLFunctionType` Metal answers for each
/// (`validation/fixtures/public/reflection/metallib-tags.txt`): 0 vertex, 1 fragment, 2 kernel (a
/// tile function too), 4 visible, 6 intersection, 7 mesh, 8 object; 3, a function a dynamic
/// library exports, for which Metal makes no `MTLFunction`; and 5, a Core Image kernel library's
/// extern function, which Metal answers as `functionType` 4.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FunctionStage {
    /// A vertex function (`TYPE` 0).
    Vertex,
    /// A fragment function (`TYPE` 1).
    Fragment,
    /// A kernel, or a tile function (`TYPE` 2).
    Kernel,
    /// A mesh function (`TYPE` 7).
    Mesh,
    /// An object function (`TYPE` 8).
    Object,
    /// A `[[visible]]` function (`!air.visible`): a function a pipeline links
    /// (`MTLLinkedFunctions`), not a stage. A bitcode module lists each it exports; a metallib tags
    /// one `TYPE` 4.
    Visible,
    /// An `[[intersection(...)]]` function: a function an intersection function table holds, not
    /// a stage (`TYPE` 6).
    Intersection,
    /// A function a dynamic library exports (`TYPE` 3): its symbol, which another library's
    /// external declaration names. Not a stage: Metal's `newFunctionWithName:` stops at an
    /// assertion for it (a probe, an M3 Max and an M4), even for a `kernel` its source defined.
    Exported,
    /// An extern function (`TYPE` 5): a Core Image kernel that a Core Image kernel library holds
    /// (`metal -fcikernel`, `!air.ci`), which Core Image links into a kernel of its own through
    /// Metal. Not a stage: Metal's `functionNames` omits it, `newFunctionWithName:` answers it
    /// with `functionType` 4 (which the SDK names no enumerator for), and a pipeline made from it
    /// fails as one made from a name the library lacks (a probe, an M3 Max under macOS 27.0.1).
    Extern,
    /// A `TYPE` value not observed in Apple's output, as read.
    Other(u8),
}

impl FunctionStage {
    /// The stage for a `TYPE` tag value.
    pub fn from_type_tag(value: u8) -> FunctionStage {
        match value {
            0 => FunctionStage::Vertex,
            1 => FunctionStage::Fragment,
            2 => FunctionStage::Kernel,
            3 => FunctionStage::Exported,
            4 => FunctionStage::Visible,
            5 => FunctionStage::Extern,
            6 => FunctionStage::Intersection,
            7 => FunctionStage::Mesh,
            8 => FunctionStage::Object,
            other => FunctionStage::Other(other),
        }
    }
}

/// Parse `bytes` as a metallib, bitcode wrapper, or raw bitcode stream.
pub fn parse(bytes: &[u8]) -> Result<Library<'_>, Error> {
    if let Some(form) = bytes.get(..4).and_then(UniversalForm::of) {
        // Offsets in an error inside the slice are the slice's own.
        return Ok(Library::Metallib(mtlb::parse(air_slice(bytes, form)?)?));
    }
    match bytes.get(..4) {
        Some(m) if m == MTLB_MAGIC => Ok(Library::Metallib(mtlb::parse(bytes)?)),
        Some(m) if m == WRAPPER_MAGIC || m == BITCODE_MAGIC => {
            let (wrapper, stream) = bitcode_stream(bytes)?;
            Ok(Library::Module(Module {
                wrapper,
                bytes,
                stream,
                hash: sha256(bytes),
            }))
        }
        Some(_) => Err(container_error(ContainerError::BadMagic, 0)),
        None => Err(container_error(
            ContainerError::Truncated {
                need: 4,
                have: bytes.len() as u64,
            },
            0,
        )),
    }
}

/// Locate the bitcode stream of wrapped or raw bitcode.
pub fn bitcode_stream(bytes: &[u8]) -> Result<(Option<WrapperHeader>, &[u8]), Error> {
    match bytes.get(..4) {
        Some(m) if m == BITCODE_MAGIC => Ok((None, bytes)),
        Some(m) if m == WRAPPER_MAGIC => {
            let have = bytes.len() as u64;
            let Some(header) = bytes.get(..WRAPPER_HEADER_LEN) else {
                return Err(container_error(
                    ContainerError::Truncated {
                        need: WRAPPER_HEADER_LEN as u64,
                        have,
                    },
                    0,
                ));
            };
            let word = |i: usize| {
                header
                    .get(i..i + 4)
                    .and_then(|b| <[u8; 4]>::try_from(b).ok())
                    .map_or(0, u32::from_le_bytes)
            };
            let wrapper = WrapperHeader {
                version: word(4),
                offset: word(8),
                size: word(12),
                cputype: word(16),
            };
            let start = u64::from(wrapper.offset);
            let end = start + u64::from(wrapper.size);
            if start < WRAPPER_HEADER_LEN as u64 {
                return Err(container_error(ContainerError::BadOffset, 8));
            }
            if end > have {
                return Err(container_error(
                    ContainerError::Truncated { need: end, have },
                    12,
                ));
            }
            let stream = usize::try_from(start)
                .ok()
                .zip(usize::try_from(end).ok())
                .and_then(|(s, e)| bytes.get(s..e))
                .ok_or_else(|| container_error(ContainerError::BadOffset, 8))?;
            if stream.get(..4) != Some(&BITCODE_MAGIC[..]) {
                return Err(container_error(ContainerError::BadMagic, start));
            }
            Ok((Some(wrapper), stream))
        }
        Some(_) => Err(container_error(ContainerError::BadMagic, 0)),
        None => Err(container_error(
            ContainerError::Truncated {
                need: 4,
                have: bytes.len() as u64,
            },
            0,
        )),
    }
}

/// The header of a universal metallib (the module docs).
#[derive(Clone, Copy)]
enum UniversalForm {
    /// Metal's own ([`UNIVERSAL_MAGIC`]): five big-endian `u32`s a slice.
    Metal,
    /// A Mach-O universal file ([`FAT_MAGIC`]): `fat_arch`, the same five `u32`s.
    MachO,
    /// A Mach-O universal file of 64-bit records ([`FAT_MAGIC_64`]): `fat_arch_64`, the CPU type
    /// and subtype as `u32`s, the offset and size as `u64`s, then the alignment and a reserved
    /// `u32`.
    MachO64,
}

impl UniversalForm {
    /// The header whose magic `head` is, if any.
    fn of(head: &[u8]) -> Option<UniversalForm> {
        match <[u8; 4]>::try_from(head).ok()? {
            UNIVERSAL_MAGIC => Some(UniversalForm::Metal),
            FAT_MAGIC => Some(UniversalForm::MachO),
            FAT_MAGIC_64 => Some(UniversalForm::MachO64),
            _ => None,
        }
    }

    /// The bytes of one slice record.
    fn record_len(self) -> usize {
        match self {
            UniversalForm::Metal | UniversalForm::MachO => 20,
            UniversalForm::MachO64 => 32,
        }
    }

    /// What a file of this header that yields no AIR slice is refused as: `error` in Metal's own,
    /// and `BadMagic` in a Mach-O universal file, which is a metallib only through its AIR slice.
    fn unread(self, error: Error) -> Error {
        match self {
            UniversalForm::Metal => error,
            UniversalForm::MachO | UniversalForm::MachO64 => {
                container_error(ContainerError::BadMagic, 0)
            }
        }
    }
}

/// The AIR slice of a universal metallib of `form` (the module docs).
fn air_slice(bytes: &[u8], form: UniversalForm) -> Result<&[u8], Error> {
    let be = |at: usize, width: usize| -> Result<u64, Error> {
        at.checked_add(width)
            .and_then(|e| bytes.get(at..e))
            .map(|b| b.iter().fold(0u64, |v, &byte| (v << 8) | u64::from(byte)))
            .ok_or_else(|| {
                container_error(
                    ContainerError::Truncated {
                        need: (at as u64).saturating_add(width as u64),
                        have: bytes.len() as u64,
                    },
                    at as u64,
                )
            })
    };
    let count = usize::try_from(be(4, 4).map_err(|e| form.unread(e))?).unwrap_or(usize::MAX);
    if count > bytes.len().saturating_sub(UNIVERSAL_HEADER_LEN) / form.record_len() {
        return Err(form.unread(container_error(ContainerError::BadOffset, 4)));
    }
    for k in 0..count {
        let record = UNIVERSAL_HEADER_LEN + k * form.record_len();
        if be(record, 4)? != u64::from(AIR_SLICE_CPU_TYPE) {
            continue;
        }
        let (at, size) = match form {
            UniversalForm::Metal | UniversalForm::MachO => {
                (be(record + 8, 4)?, be(record + 12, 4)?)
            }
            UniversalForm::MachO64 => (be(record + 8, 8)?, be(record + 16, 8)?),
        };
        let start = usize::try_from(at).unwrap_or(usize::MAX);
        let slice = start
            .checked_add(usize::try_from(size).unwrap_or(usize::MAX))
            .and_then(|end| bytes.get(start..end))
            .ok_or_else(|| container_error(ContainerError::BadOffset, record as u64 + 8))?;
        if slice.get(..4) != Some(&MTLB_MAGIC[..]) {
            return Err(container_error(ContainerError::BadMagic, at));
        }
        return Ok(slice);
    }
    Err(form.unread(container_error(ContainerError::NoAirSlice, 0)))
}

/// SHA-256 of `bytes`.
pub fn sha256(bytes: &[u8]) -> Hash256 {
    Sha256::digest(bytes).into()
}

fn container_error(kind: ContainerError, offset: u64) -> Error {
    InputError::new(InputErrorKind::Container(kind), Location::container(offset)).into()
}
