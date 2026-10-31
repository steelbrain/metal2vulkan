//! MTLB: header, function table, extended header, per-function public/private metadata, and the
//! bitcode section.
//!
//! Layout (little-endian, packed), as observed in Apple output of formats 2.7–2.9:
//!
//! - Header (0x58 bytes): `MTLB`, `platform u16`, `version u16 u16`, `file_type u8` (0 an
//!   executable library, 1 a Core Image kernel library, 2 a dynamic library), `os u8`,
//!   `os_version u16 u16`, then `u64`
//!   `file_size` and `(offset, size)` pairs for the function list, public metadata, private
//!   metadata, and bitcode sections.
//! - Function list: `u32 count`, then `count` entries of `u32 entry_size` (counting itself) and a
//!   tag list; the list size in the header is the sum of the entry sizes. The extended-header tag
//!   list follows the last entry.
//! - Per function, `OFFT` holds its public-metadata, private-metadata, and bitcode offsets relative
//!   to their section bases. Each metadata record is `u32 size` and a tag list of exactly that
//!   size. The bitcode is `MDSZ` bytes of wrapped bitcode, and `HASH` is its SHA-256.
//! - No two functions share or overlap a public-metadata record, a private-metadata record, or a
//!   bitcode range. Apple output never does, and requiring it keeps parsing (and the parsed
//!   result) linear in the file size: otherwise many entries could re-parse one large record.
//!   The exceptions are a dynamic library (`MTLLibraryTypeDynamic`), whose functions Apple's
//!   toolchain lists as the symbols it exports (`TYPE` 3, each as `NAME`) over the library's one
//!   module, and a Core Image kernel library (`metallib -cikernel`), whose kernels it lists as
//!   extern functions (`TYPE` 5) over the one module `metal -fcikernel` compiled: their functions
//!   may share exactly one bitcode range, hashed once.
//! - The extended header's `HDYN` and `ILST` tags each hold an `(offset, size)` of a region of
//!   the file. `HDYN`'s is a tag list: `NAME`, the library's install name (a dynamic library's
//!   `installName`; an executable's file name), and one `DYNL` per dynamic library it was linked
//!   against (`MTLCompileOptions.libraries`, `metal -l`), by install name, in link order.
//!   `ILST`'s is a `u32` count of records, each a `u32` size (counting itself) and a tag list
//!   naming (`NAME`) one symbol the library imports from them. Observed in Apple output of format
//!   2.9 (a probe, an M3 Max and an M4).

use super::tags::{self, RawTag, Tag, TagReader};
use super::{container_error, sha256, FunctionStage, Hash256};
use crate::error::{ContainerError, InputError, InputErrorKind, Location};
use crate::error::{Error, Feature};

const HEADER_LEN: usize = 0x58;
/// The smallest function entry: its size word and `ENDT`.
const MIN_ENTRY_LEN: u64 = 8;
/// Accepted `version_major` values (only 2.x observed).
const SUPPORTED_MAJOR: [u16; 2] = [1, 2];
/// `file_type` of an executable library.
const FILE_TYPE_EXECUTABLE: u8 = 0;
/// `file_type` of a Core Image kernel library (`metallib -cikernel`).
const FILE_TYPE_CORE_IMAGE: u8 = 1;
/// `file_type` of a dynamic library (`MTLLibraryTypeDynamic`).
const FILE_TYPE_DYNAMIC: u8 = 2;

/// A metallib's library type (`MTLLibraryType`), from its header's `file_type`.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LibraryType {
    /// `MTLLibraryTypeExecutable` (`file_type` 0): a library of entry points. A Core Image kernel
    /// library (`file_type` 1) is one too, as Metal answers it: its functions are extern functions
    /// (`TYPE` 5), which Core Image links into a kernel of its own.
    Executable,
    /// `MTLLibraryTypeDynamic` (`file_type` 2): a library whose functions resolve other
    /// libraries' external symbols (`MTLDynamicLibrary`).
    Dynamic,
}

/// A section of the file: `[offset, offset + size)`, checked against the file size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Section {
    pub offset: u64,
    pub size: u64,
}

impl Section {
    fn end(self) -> u64 {
        self.offset + self.size
    }
}

/// The MTLB header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MtlbHeader {
    pub platform: u16,
    pub version: (u16, u16),
    pub file_type: u8,
    pub os: u8,
    pub os_version: (u16, u16),
    pub file_size: u64,
    pub function_list: Section,
    pub public_metadata: Section,
    pub private_metadata: Section,
    pub bitcode: Section,
}

/// A parsed metallib.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Metallib<'a> {
    /// The metallib's own bytes, which its header's ranges count in and its functions' slices lie
    /// in: the input, or a universal file's AIR slice (the module docs).
    pub bytes: &'a [u8],
    /// The file header: its version, target, and section ranges.
    pub header: MtlbHeader,
    /// The function list's entries, in file order.
    pub functions: Vec<LibraryFunction<'a>>,
    /// The extended-header tags, all kept raw.
    pub extended: Vec<RawTag<'a>>,
    /// What its `HDYN` and `ILST` regions say about dynamic libraries.
    pub linkage: DynamicLinkage<'a>,
}

impl Metallib<'_> {
    /// The library's type, from the header's `file_type`.
    pub fn library_type(&self) -> LibraryType {
        match self.header.file_type {
            FILE_TYPE_DYNAMIC => LibraryType::Dynamic,
            _ => LibraryType::Executable,
        }
    }
}

/// A metallib's install name and the dynamic libraries it was linked against (its `HDYN` and
/// `ILST` regions); empty when it has neither.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DynamicLinkage<'a> {
    /// `HDYN`'s `NAME`: a dynamic library's install name, or the name an executable was written
    /// under.
    pub install_name: Option<&'a str>,
    /// `HDYN`'s `DYNL`s: the install names of the dynamic libraries it was linked against, in
    /// link order.
    pub dependencies: Vec<&'a str>,
    /// `ILST`'s `NAME`s: the symbols it imports from them, in their order.
    pub imports: Vec<&'a str>,
    /// A dynamic library's one module: its bitcode section, which every function it exports
    /// shares (observed: each `OFFT` bitcode offset 0, each `MDSZ` the section's size). `None`
    /// for an executable library.
    pub module: Option<&'a [u8]>,
}

/// One function of a metallib.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LibraryFunction<'a> {
    /// `NAME`: the function's name.
    pub name: &'a str,
    /// `TYPE`: the function's stage.
    pub stage: FunctionStage,
    /// The verified SHA-256 of [`LibraryFunction::bitcode`]: the function's content key.
    pub hash: Hash256,
    /// `VERS`: the AIR version, major and minor.
    pub air_version: (u16, u16),
    /// `VERS`: the Metal language version, major and minor.
    pub language_version: (u16, u16),
    /// The function's wrapped bitcode (its own module).
    pub bitcode: &'a [u8],
    /// The function's public metadata tags.
    pub public: FunctionPublicMetadata<'a>,
    /// Private metadata tags, kept raw (no content observed).
    pub private: Vec<RawTag<'a>>,
    /// Function-table tags this layer does not interpret (`LAYR`, `TESS`, `RFLT`, …).
    pub raw_tags: Vec<RawTag<'a>>,
}

impl<'a> LibraryFunction<'a> {
    /// The bitcode stream inside the function's wrapper.
    #[cfg(test)]
    pub fn stream(&self) -> Result<&'a [u8], Error> {
        super::bitcode_stream(self.bitcode).map(|(_, stream)| stream)
    }
}

/// Per-function public metadata. Cross-checked against the AIR metadata ABI by `air::abi`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FunctionPublicMetadata<'a> {
    /// `CNST`: function constants.
    pub constants: Vec<ContainerConstant<'a>>,
    /// `VATT`: vertex attributes.
    pub vertex_attributes: Vec<VertexAttribute<'a>>,
    /// `VATY`: vertex attribute `MTLDataType`s, raw.
    pub vertex_attribute_types: Vec<u8>,
    /// Tags this layer does not interpret (`ARGR`, `RETR`, …).
    pub raw_tags: Vec<RawTag<'a>>,
}

/// One `CNST` entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ContainerConstant<'a> {
    /// The constant's name.
    pub name: &'a str,
    /// `MTLDataType`, raw.
    pub data_type: u8,
    /// The constant's index (`[[function_constant(n)]]`).
    pub index: u16,
    /// Raw flag byte.
    pub flags: u8,
}

/// One `VATT` entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VertexAttribute<'a> {
    /// The attribute's name.
    pub name: &'a str,
    /// The attribute index and flags as stored; decoded by the ABI cross-check.
    pub packed: u16,
}

/// Little-endian fixed-offset reads from the header.
struct Fields<'a>(&'a [u8]);

impl Fields<'_> {
    fn bytes<const N: usize>(&self, at: usize) -> [u8; N] {
        // The header length was checked; a short read yields zeros and cannot happen.
        at.checked_add(N)
            .and_then(|e| self.0.get(at..e))
            .and_then(|b| <[u8; N]>::try_from(b).ok())
            .unwrap_or([0; N])
    }
    fn u8(&self, at: usize) -> u8 {
        let [b] = self.bytes::<1>(at);
        b
    }
    fn u16(&self, at: usize) -> u16 {
        u16::from_le_bytes(self.bytes(at))
    }
    fn u64(&self, at: usize) -> u64 {
        u64::from_le_bytes(self.bytes(at))
    }
}

pub(super) fn parse(bytes: &[u8]) -> Result<Metallib<'_>, Error> {
    let have = bytes.len() as u64;
    let Some(head) = bytes.get(..HEADER_LEN) else {
        return Err(container_error(
            ContainerError::Truncated {
                need: HEADER_LEN as u64,
                have,
            },
            0,
        ));
    };
    let h = Fields(head);
    let section = |at: usize| Section {
        offset: h.u64(at),
        size: h.u64(at + 8),
    };
    let header = MtlbHeader {
        platform: h.u16(4),
        version: (h.u16(6), h.u16(8)),
        file_type: h.u8(10),
        os: h.u8(11),
        os_version: (h.u16(12), h.u16(14)),
        file_size: h.u64(16),
        function_list: section(24),
        public_metadata: section(40),
        private_metadata: section(56),
        bitcode: section(72),
    };
    if !SUPPORTED_MAJOR.contains(&header.version.0) {
        return Err(container_error(
            ContainerError::UnsupportedContainerVersion {
                major: header.version.0,
                minor: header.version.1,
            },
            6,
        ));
    }
    if !matches!(
        header.file_type,
        FILE_TYPE_EXECUTABLE | FILE_TYPE_CORE_IMAGE | FILE_TYPE_DYNAMIC
    ) {
        return Err(InputError::new(
            InputErrorKind::Unsupported(Feature::LibraryKind(header.file_type)),
            Location::container(10),
        )
        .into());
    }
    if header.file_size > have {
        return Err(container_error(
            ContainerError::Truncated {
                need: header.file_size,
                have,
            },
            16,
        ));
    }
    let file = usize::try_from(header.file_size)
        .ok()
        .and_then(|n| bytes.get(..n))
        .ok_or_else(|| container_error(ContainerError::BadOffset, 16))?;
    for (s, at) in [
        (header.function_list, 24),
        (header.public_metadata, 40),
        (header.private_metadata, 56),
        (header.bitcode, 72),
    ] {
        let fits = s
            .offset
            .checked_add(s.size)
            .is_some_and(|end| s.offset >= HEADER_LEN as u64 && end <= header.file_size);
        if !fits {
            return Err(container_error(ContainerError::BadOffset, at));
        }
    }

    // Function list.
    let list = header.function_list;
    let count_at = offset(list.offset)?;
    let count = u64::from(read_u32(file, count_at)?);
    let entries_start = count_at + 4;
    let entries_end = list
        .offset
        .checked_add(4)
        .and_then(|s| s.checked_add(list.size))
        .filter(|e| *e <= header.file_size)
        .ok_or_else(|| container_error(ContainerError::BadOffset, 32))?;
    let entries_end = offset(entries_end)?;
    if count > list.size / MIN_ENTRY_LEN {
        return Err(container_error(ContainerError::BadOffset, list.offset));
    }
    // `count` is untrusted and only loosely bounded above, so nothing is reserved from it: the
    // list grows one entry at a time, and each entry is pushed only after its own bytes parsed.
    let mut parsed = Vec::new();
    let mut at = entries_start;
    for _ in 0..count {
        let size = offset(u64::from(read_u32(file, at)?))?;
        let end = at
            .checked_add(size)
            .filter(|e| size as u64 >= MIN_ENTRY_LEN && *e <= entries_end)
            .ok_or_else(|| container_error(ContainerError::BadTag, at as u64))?;
        let (list, list_end) = tags::read_list(file, at + 4, end)?;
        if list_end != end {
            return Err(container_error(ContainerError::BadTag, at as u64));
        }
        parsed.push(function(file, &header, at as u64, &list)?);
        at = end;
    }
    if at != entries_end {
        return Err(container_error(ContainerError::BadOffset, 32));
    }
    let (extended, _) = tags::read_list(file, at, file.len())?;
    let mut linkage = linkage(file, &extended)?;
    if header.file_type == FILE_TYPE_DYNAMIC {
        let section = header.bitcode;
        let (start, end) = (offset(section.offset)?, offset(section.end())?);
        linkage.module = Some(
            file.get(start..end)
                .ok_or_else(|| container_error(ContainerError::BadOffset, 72))?,
        );
    }

    // Function bitcode ranges are disjoint, so verifying every hash is linear in the file size. A
    // dynamic library's functions, and a Core Image kernel library's, may share one range exactly,
    // which is hashed once.
    let shared = matches!(header.file_type, FILE_TYPE_DYNAMIC | FILE_TYPE_CORE_IMAGE);
    let mut ranges: Vec<(u64, u64, u64, usize)> = parsed
        .iter()
        .enumerate()
        .map(|(i, p)| (p.bitcode_start, p.function.bitcode.len() as u64, p.entry, i))
        .collect();
    ranges.sort_unstable();
    for pair in ranges.windows(2) {
        if let [(a, a_len, _, _), (b, b_len, b_entry, _)] = pair {
            let same = shared && a == b && a_len == b_len;
            if a + a_len > *b && !same {
                return Err(container_error(ContainerError::BadOffset, *b_entry));
            }
        }
    }
    let mut hashed: Option<(u64, u64, Hash256)> = None;
    for &(start, len, entry, i) in &ranges {
        let p = parsed
            .get(i)
            .ok_or_else(|| container_error(ContainerError::BadOffset, entry))?;
        let digest = match hashed {
            Some((s, l, digest)) if s == start && l == len => digest,
            _ => {
                let digest = sha256(p.function.bitcode);
                hashed = Some((start, len, digest));
                digest
            }
        };
        if digest != p.function.hash {
            return Err(container_error(ContainerError::HashMismatch, entry));
        }
    }

    // Metadata records are parsed only after their offsets are known to be disjoint, so each
    // record's bytes are read (and its tags held) once.
    let public = records(
        file,
        header.public_metadata,
        parsed.iter().map(|p| (p.public_at, p.entry)),
    )?;
    let private = records(
        file,
        header.private_metadata,
        parsed.iter().map(|p| (p.private_at, p.entry)),
    )?;
    let mut functions = Vec::with_capacity(parsed.len());
    for ((p, public), private) in parsed.into_iter().zip(public).zip(private) {
        let mut f = p.function;
        f.public = public_metadata(&public)?;
        f.private = private.into_iter().map(|t| t.raw).collect();
        functions.push(f);
    }

    Ok(Metallib {
        bytes,
        header,
        functions,
        extended: extended.into_iter().map(|t| t.raw).collect(),
        linkage,
    })
}

/// The region of `file` an extended-header tag's `(offset, size)` names.
fn region<'a>(file: &'a [u8], tag: &Tag<'a>) -> Result<(usize, usize), Error> {
    let mut r = TagReader::new(tag);
    let (at, size) = (r.u64()?, r.u64()?);
    r.finish()?;
    let bad = || container_error(ContainerError::BadOffset, tag.offset);
    let end = at.checked_add(size).ok_or_else(bad)?;
    let (at, end) = (offset(at)?, offset(end)?);
    if at < HEADER_LEN || end > file.len() {
        return Err(bad());
    }
    Ok((at, end))
}

/// The `HDYN` and `ILST` regions of the extended header (the module docs), each read once.
fn linkage<'a>(file: &'a [u8], extended: &[Tag<'a>]) -> Result<DynamicLinkage<'a>, Error> {
    let mut out = DynamicLinkage::default();
    let (mut hdyn, mut ilst) = (false, false);
    for tag in extended {
        match &tag.raw.fourcc {
            b"HDYN" => {
                if hdyn {
                    return Err(container_error(ContainerError::DuplicateTag, tag.offset));
                }
                hdyn = true;
                let (at, end) = region(file, tag)?;
                let (list, list_end) = tags::read_list(file, at, end)?;
                if list_end != end {
                    return Err(container_error(ContainerError::BadTag, at as u64));
                }
                for t in &list {
                    let mut r = TagReader::new(t);
                    match &t.raw.fourcc {
                        b"NAME" => {
                            if !set(&mut out.install_name, r.cstr()?) {
                                return Err(container_error(
                                    ContainerError::DuplicateTag,
                                    t.offset,
                                ));
                            }
                        }
                        b"DYNL" => out.dependencies.push(r.cstr()?),
                        _ => continue,
                    }
                    r.finish()?;
                }
            }
            b"ILST" => {
                if ilst {
                    return Err(container_error(ContainerError::DuplicateTag, tag.offset));
                }
                ilst = true;
                let (at, end) = region(file, tag)?;
                let count = u64::from(read_u32(file, at)?);
                // Each record is at least its size word and `ENDT`.
                if count > (end - at) as u64 / MIN_ENTRY_LEN {
                    return Err(container_error(ContainerError::BadOffset, at as u64));
                }
                let mut next = at + 4;
                for _ in 0..count {
                    // The record's size counts its size word, as a function entry's does.
                    let size = offset(u64::from(read_u32(file, next)?))?;
                    let record_end = next
                        .checked_add(size)
                        .filter(|e| size as u64 >= MIN_ENTRY_LEN && *e <= end)
                        .ok_or_else(|| container_error(ContainerError::BadTag, next as u64))?;
                    let (list, list_end) = tags::read_list(file, next + 4, record_end)?;
                    if list_end != record_end {
                        return Err(container_error(ContainerError::BadTag, next as u64));
                    }
                    let mut name = None;
                    for t in &list {
                        if &t.raw.fourcc == b"NAME" {
                            let mut r = TagReader::new(t);
                            if !set(&mut name, r.cstr()?) {
                                return Err(container_error(
                                    ContainerError::DuplicateTag,
                                    t.offset,
                                ));
                            }
                            r.finish()?;
                        }
                    }
                    out.imports.push(
                        name.ok_or_else(|| {
                            container_error(ContainerError::MissingTag, next as u64)
                        })?,
                    );
                    next = record_end;
                }
                if next != end {
                    return Err(container_error(ContainerError::BadOffset, at as u64));
                }
            }
            _ => {}
        }
    }
    Ok(out)
}

/// A function with the file offsets used for validation and diagnostics. Its metadata fields are
/// still empty; they are filled once every function's metadata offsets have been checked.
struct Parsed<'a> {
    function: LibraryFunction<'a>,
    /// The function-table entry.
    entry: u64,
    /// The function's bitcode.
    bitcode_start: u64,
    /// `OFFT`'s public-metadata offset, relative to the section.
    public_at: u64,
    /// `OFFT`'s private-metadata offset, relative to the section.
    private_at: u64,
}

/// Parse one function-table entry starting at `entry` from its tags. Metadata records are not read
/// here (see [`records`]).
fn function<'a>(
    file: &'a [u8],
    header: &MtlbHeader,
    entry: u64,
    list: &[Tag<'a>],
) -> Result<Parsed<'a>, Error> {
    let mut name = None;
    let mut stage = None;
    let mut hash = None;
    let mut size = None;
    let mut offsets = None;
    let mut versions = None;
    let mut raw_tags = Vec::new();
    for tag in list {
        let mut r = TagReader::new(tag);
        let slot_filled = match &tag.raw.fourcc {
            b"NAME" => set(&mut name, r.cstr()?),
            b"TYPE" => set(&mut stage, FunctionStage::from_type_tag(r.u8()?)),
            b"HASH" => set(&mut hash, r.array::<32>()?),
            b"MDSZ" => set(&mut size, r.u64()?),
            b"OFFT" => set(&mut offsets, (r.u64()?, r.u64()?, r.u64()?)),
            b"VERS" => set(&mut versions, ((r.u16()?, r.u16()?), (r.u16()?, r.u16()?))),
            _ => {
                raw_tags.push(tag.raw);
                continue;
            }
        };
        if !slot_filled {
            return Err(container_error(ContainerError::DuplicateTag, tag.offset));
        }
        r.finish()?;
    }
    let missing = || container_error(ContainerError::MissingTag, entry);
    let (public_at, private_at, bitcode_at) = offsets.ok_or_else(missing)?;
    let size = size.ok_or_else(missing)?;
    let (air_version, language_version) = versions.ok_or_else(missing)?;

    let bitcode_start = header
        .bitcode
        .offset
        .checked_add(bitcode_at)
        .ok_or_else(|| container_error(ContainerError::BadOffset, entry))?;
    let bitcode = bitcode_start
        .checked_add(size)
        .map(|e| (bitcode_start, e))
        .filter(|(_, e)| *e <= header.bitcode.end())
        .and_then(|(s, e)| Some((usize::try_from(s).ok()?, usize::try_from(e).ok()?)))
        .and_then(|(s, e)| file.get(s..e))
        .ok_or_else(|| container_error(ContainerError::BadOffset, entry))?;
    Ok(Parsed {
        function: LibraryFunction {
            name: name.ok_or_else(missing)?,
            stage: stage.ok_or_else(missing)?,
            hash: hash.ok_or_else(missing)?,
            air_version,
            language_version,
            bitcode,
            public: FunctionPublicMetadata::default(),
            private: Vec::new(),
            raw_tags,
        },
        entry,
        bitcode_start,
        public_at,
        private_at,
    })
}

/// The metadata records in `section` at the given `(offset, entry)` pairs, one per function and in
/// the same order. Records are read in file order, and one that starts before the previous record
/// ends (including a shared record) is `BadOffset` at the later function's entry. The records are
/// therefore disjoint, and the total work and the returned tags are linear in the section size.
fn records<'a>(
    file: &'a [u8],
    section: Section,
    offsets: impl Iterator<Item = (u64, u64)>,
) -> Result<Vec<Vec<Tag<'a>>>, Error> {
    let mut order: Vec<(u64, u64, usize)> = offsets
        .enumerate()
        .map(|(i, (at, entry))| (at, entry, i))
        .collect();
    order.sort_unstable();
    let mut out: Vec<Option<Vec<Tag<'a>>>> = vec![None; order.len()];
    let mut end = 0;
    for (at, entry, i) in order {
        if at < end {
            return Err(container_error(ContainerError::BadOffset, entry));
        }
        let (list, list_end) = metadata(file, section, at, entry)?;
        end = list_end - section.offset;
        if let Some(slot) = out.get_mut(i) {
            *slot = Some(list);
        }
    }
    Ok(out.into_iter().map(Option::unwrap_or_default).collect())
}

/// Set a once-only tag value. Returns `false` when it was already set.
fn set<T>(slot: &mut Option<T>, value: T) -> bool {
    if slot.is_some() {
        return false;
    }
    *slot = Some(value);
    true
}

/// The metadata record at `at` in `section`: `u32 size`, then a tag list ending in `ENDT`.
///
/// Two size conventions occur in Apple output, and both are accepted exactly: the tag-list length
/// (written by `metallib` when linking `.air` files) and the tag-list length plus the size word
/// itself (written by `metal` when compiling straight to a metallib). Any other size is `BadTag`.
///
/// Returns the tags and the file offset where the record ends.
fn metadata(
    file: &[u8],
    section: Section,
    at: u64,
    entry: u64,
) -> Result<(Vec<Tag<'_>>, u64), Error> {
    let start = section
        .offset
        .checked_add(at)
        .filter(|s| s.checked_add(4).is_some_and(|e| e <= section.end()))
        .ok_or_else(|| container_error(ContainerError::BadOffset, entry))?;
    let size = u64::from(read_u32(file, offset(start)?)?);
    let list_start = start + 4;
    let bound = list_start.saturating_add(size).min(section.end());
    let (list, list_end) = tags::read_list(file, offset(list_start)?, offset(bound)?)?;
    let len = list_end as u64 - list_start;
    if size != len && size != len + 4 {
        return Err(container_error(ContainerError::BadTag, start));
    }
    Ok((list, list_end as u64))
}

fn public_metadata<'a>(list: &[Tag<'a>]) -> Result<FunctionPublicMetadata<'a>, Error> {
    let mut md = FunctionPublicMetadata::default();
    let (mut cnst, mut vatt, mut vaty) = (None, None, None);
    for tag in list {
        let mut r = TagReader::new(tag);
        let first = match &tag.raw.fourcc {
            b"CNST" => {
                let n = r.u16()?;
                // Each entry is at least 5 bytes (empty name, type, index, flags).
                if usize::from(n) > r.remaining() / 5 {
                    return Err(container_error(ContainerError::BadTag, tag.offset));
                }
                md.constants.reserve_exact(usize::from(n));
                for _ in 0..n {
                    md.constants.push(ContainerConstant {
                        name: r.cstr()?,
                        data_type: r.u8()?,
                        index: r.u16()?,
                        flags: r.u8()?,
                    });
                }
                set(&mut cnst, ())
            }
            b"VATT" => {
                let n = r.u16()?;
                if usize::from(n) > r.remaining() / 3 {
                    return Err(container_error(ContainerError::BadTag, tag.offset));
                }
                md.vertex_attributes.reserve_exact(usize::from(n));
                for _ in 0..n {
                    md.vertex_attributes.push(VertexAttribute {
                        name: r.cstr()?,
                        packed: r.u16()?,
                    });
                }
                set(&mut vatt, ())
            }
            b"VATY" => {
                let n = r.u16()?;
                if usize::from(n) > r.remaining() {
                    return Err(container_error(ContainerError::BadTag, tag.offset));
                }
                md.vertex_attribute_types.reserve_exact(usize::from(n));
                for _ in 0..n {
                    md.vertex_attribute_types.push(r.u8()?);
                }
                set(&mut vaty, ())
            }
            _ => {
                md.raw_tags.push(tag.raw);
                continue;
            }
        };
        if !first {
            return Err(container_error(ContainerError::DuplicateTag, tag.offset));
        }
        r.finish()?;
    }
    Ok(md)
}

fn read_u32(file: &[u8], at: usize) -> Result<u32, Error> {
    at.checked_add(4)
        .and_then(|e| file.get(at..e))
        .and_then(|b| <[u8; 4]>::try_from(b).ok())
        .map(u32::from_le_bytes)
        .ok_or_else(|| {
            container_error(
                ContainerError::Truncated {
                    need: at as u64 + 4,
                    have: file.len() as u64,
                },
                at as u64,
            )
        })
}

fn offset(value: u64) -> Result<usize, Error> {
    usize::try_from(value).map_err(|_| container_error(ContainerError::BadOffset, 0))
}
