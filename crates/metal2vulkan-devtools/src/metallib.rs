//! A metallib writer for tests: functions in the layout Apple's toolchain writes (the AIR front
//! end's `container::mtlb` reads it), with the tags a test chooses. It writes synthetic
//! containers around bitcode a test supplies; no Apple-owned bytes.

use sha2::{Digest, Sha256};

/// The LLVM bitcode wrapper header's magic (`0x0B17C0DE`, little-endian).
const WRAPPER_MAGIC: [u8; 4] = [0xDE, 0xC0, 0x17, 0x0B];

fn sha256(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

/// `stream` in a bitcode wrapper.
pub fn wrap(stream: &[u8]) -> Vec<u8> {
    if stream.starts_with(&WRAPPER_MAGIC) {
        return stream.to_vec();
    }
    let mut out = Vec::new();
    out.extend_from_slice(&WRAPPER_MAGIC);
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&20u32.to_le_bytes());
    out.extend_from_slice(&(stream.len() as u32).to_le_bytes());
    out.extend_from_slice(&u32::MAX.to_le_bytes());
    out.extend_from_slice(stream);
    out
}

fn tag(out: &mut Vec<u8>, fourcc: &[u8; 4], bytes: &[u8]) {
    out.extend_from_slice(fourcc);
    out.extend_from_slice(&(bytes.len() as u16).to_le_bytes());
    out.extend_from_slice(bytes);
}

/// One metallib function: name, `TYPE`, wrapped bitcode, `CNST` entries (name, `MTLDataType`,
/// index; required), further public-metadata tags, and further function-table tags.
pub struct MtlbFunction<'a> {
    pub name: &'a str,
    pub type_code: u8,
    pub bitcode: Vec<u8>,
    pub constants: Vec<(&'a str, u8, u16)>,
    /// Further public-metadata tags (fourcc, bytes), after `CNST`: `VATT`, `VATY`, or a `CNST`
    /// with other flags in place of `constants`.
    pub public: Vec<([u8; 4], Vec<u8>)>,
    /// Further function-table tags (fourcc, bytes), after `VERS`: `RFLT`, `LAYR`, ...
    pub tags: Vec<([u8; 4], Vec<u8>)>,
}

/// What a metallib records of dynamic libraries: whether it is one (file type 2, its functions
/// sharing the first one's bitcode, written once), its install name and the install names of the
/// dynamic libraries it was linked against (`HDYN`), and the symbols it imports from them (`ILST`),
/// in the layout Apple's toolchain writes (a probe).
#[derive(Clone, Debug, Default)]
pub struct MtlbLinkage<'a> {
    pub dynamic: bool,
    pub install_name: Option<&'a str>,
    pub dependencies: Vec<&'a str>,
    pub imports: Vec<&'a str>,
}

/// A metallib in the layout Apple's toolchain writes.
pub fn metallib(functions: &[MtlbFunction<'_>]) -> Vec<u8> {
    metallib_linked(functions, &MtlbLinkage::default())
}

/// [`metallib`] with `linkage`: its regions follow the bitcode, as in Apple's output.
pub fn metallib_linked(functions: &[MtlbFunction<'_>], linkage: &MtlbLinkage<'_>) -> Vec<u8> {
    let (mut public, mut private, mut bitcode, mut entries) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let md = |tags: &[([u8; 4], Vec<u8>)]| {
        let mut list = Vec::new();
        for (fourcc, bytes) in tags {
            tag(&mut list, fourcc, bytes);
        }
        list.extend_from_slice(b"ENDT");
        let mut out = (list.len() as u32).to_le_bytes().to_vec();
        out.extend(list);
        out
    };
    for f in functions {
        let mut pub_tags = Vec::new();
        if !f.constants.is_empty() {
            let mut cnst = (f.constants.len() as u16).to_le_bytes().to_vec();
            for (name, ty, index) in &f.constants {
                cnst.extend(name.as_bytes());
                cnst.push(0);
                cnst.push(*ty);
                cnst.extend(index.to_le_bytes());
                cnst.push(1);
            }
            pub_tags.push((*b"CNST", cnst));
        }
        pub_tags.extend(f.public.iter().cloned());
        let (pub_at, priv_at) = (public.len(), private.len());
        public.extend(md(&pub_tags));
        private.extend(md(&[]));
        let bc_at = match (linkage.dynamic, bitcode.is_empty()) {
            (true, false) => 0,
            _ => {
                let at = bitcode.len();
                bitcode.extend_from_slice(&f.bitcode);
                at
            }
        };
        let mut list = Vec::new();
        let mut name = f.name.as_bytes().to_vec();
        name.push(0);
        tag(&mut list, b"NAME", &name);
        tag(&mut list, b"TYPE", &[f.type_code]);
        tag(&mut list, b"HASH", &sha256(&f.bitcode));
        tag(&mut list, b"MDSZ", &(f.bitcode.len() as u64).to_le_bytes());
        let mut offt = Vec::new();
        for v in [pub_at, priv_at, bc_at] {
            offt.extend_from_slice(&(v as u64).to_le_bytes());
        }
        tag(&mut list, b"OFFT", &offt);
        tag(&mut list, b"VERS", &[2, 0, 8, 0, 4, 0, 0, 0]);
        for (fourcc, bytes) in &f.tags {
            tag(&mut list, fourcc, bytes);
        }
        list.extend_from_slice(b"ENDT");
        let mut entry = ((list.len() + 4) as u32).to_le_bytes().to_vec();
        entry.extend(list);
        entries.push(entry);
    }
    let entries_len: usize = entries.iter().map(Vec::len).sum();
    // The regions: `HDYN`'s tag list, and `ILST`'s count and records (each record's size counts
    // its size word).
    let mut regions: Vec<([u8; 4], Vec<u8>)> = Vec::new();
    if linkage.install_name.is_some() || !linkage.dependencies.is_empty() {
        let mut list = Vec::new();
        let cstr = |s: &str| {
            let mut v = s.as_bytes().to_vec();
            v.push(0);
            v
        };
        if let Some(name) = linkage.install_name {
            tag(&mut list, b"NAME", &cstr(name));
        }
        for d in &linkage.dependencies {
            tag(&mut list, b"DYNL", &cstr(d));
        }
        list.extend_from_slice(b"ENDT");
        regions.push((*b"HDYN", list));
    }
    if !linkage.imports.is_empty() {
        let mut out = (linkage.imports.len() as u32).to_le_bytes().to_vec();
        for symbol in &linkage.imports {
            let mut name = symbol.as_bytes().to_vec();
            name.push(0);
            let mut list = Vec::new();
            tag(&mut list, b"NAME", &name);
            list.extend_from_slice(b"ENDT");
            out.extend_from_slice(&((list.len() + 4) as u32).to_le_bytes());
            out.extend(list);
        }
        regions.push((*b"ILST", out));
    }
    let ext_len = regions.len() * 22 + 4;
    let list_off = 0x58usize;
    let pub_off = list_off + 4 + entries_len + ext_len;
    let priv_off = pub_off + public.len();
    let bc_off = priv_off + private.len();
    let mut region_at = bc_off + bitcode.len();
    let mut ext = Vec::new();
    for (fourcc, bytes) in &regions {
        let mut at = (region_at as u64).to_le_bytes().to_vec();
        at.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
        tag(&mut ext, fourcc, &at);
        region_at += bytes.len();
    }
    ext.extend_from_slice(b"ENDT");
    let file_size = region_at;
    let mut out = Vec::new();
    out.extend_from_slice(b"MTLB");
    out.extend_from_slice(&0x8001u16.to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&9u16.to_le_bytes());
    out.push(if linkage.dynamic { 2 } else { 0 });
    out.push(0x81);
    out.extend_from_slice(&26u16.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    for v in [
        file_size,
        list_off,
        entries_len,
        pub_off,
        public.len(),
        priv_off,
        private.len(),
        bc_off,
        bitcode.len(),
    ] {
        out.extend_from_slice(&(v as u64).to_le_bytes());
    }
    out.extend_from_slice(&(functions.len() as u32).to_le_bytes());
    for e in entries {
        out.extend(e);
    }
    out.extend(ext);
    out.extend(public);
    out.extend(private);
    out.extend(bitcode);
    for (_, bytes) in regions {
        out.extend(bytes);
    }
    assert_eq!(out.len(), file_size);
    out
}

/// The header of a universal file ([`universal`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UniversalHeader {
    /// Metal's own (`0xCBFEBABE`), as `serializeToURL:error:` writes a dynamic library.
    Metal,
    /// A Mach-O universal file (`FAT_MAGIC`, `0xCAFEBABE`: `fat_arch` records), the form of the
    /// system's universal metallibs.
    MachO,
    /// A Mach-O universal file of 64-bit records (`FAT_MAGIC_64`, `0xCAFEBABF`: `fat_arch_64`).
    MachO64,
}

impl UniversalHeader {
    /// Every header.
    pub const ALL: [UniversalHeader; 3] = [
        UniversalHeader::Metal,
        UniversalHeader::MachO,
        UniversalHeader::MachO64,
    ];
}

/// `slices` (CPU type, bytes) in a universal file of `header`, each slice at a 16-byte boundary
/// (the AIR slice is of CPU type `0x01000017`).
pub fn universal(header: UniversalHeader, slices: &[(u32, &[u8])]) -> Vec<u8> {
    let (magic, wide) = match header {
        UniversalHeader::Metal => ([0xCB, 0xFE, 0xBA, 0xBE], false),
        UniversalHeader::MachO => ([0xCA, 0xFE, 0xBA, 0xBE], false),
        UniversalHeader::MachO64 => ([0xCA, 0xFE, 0xBA, 0xBF], true),
    };
    let record = if wide { 32 } else { 20 };
    let mut out = magic.to_vec();
    out.extend_from_slice(&(slices.len() as u32).to_be_bytes());
    let mut at = (8 + record * slices.len()).next_multiple_of(16);
    let mut starts = Vec::with_capacity(slices.len());
    for (cpu, bytes) in slices {
        out.extend_from_slice(&cpu.to_be_bytes());
        out.extend_from_slice(&0u32.to_be_bytes());
        if wide {
            out.extend_from_slice(&(at as u64).to_be_bytes());
            out.extend_from_slice(&(bytes.len() as u64).to_be_bytes());
            out.extend_from_slice(&[0; 8]);
        } else {
            out.extend_from_slice(&(at as u32).to_be_bytes());
            out.extend_from_slice(&(bytes.len() as u32).to_be_bytes());
            out.extend_from_slice(&[0; 4]);
        }
        starts.push(at);
        at = (at + bytes.len()).next_multiple_of(16);
    }
    for ((_, bytes), start) in slices.iter().zip(starts) {
        out.resize(start, 0);
        out.extend_from_slice(bytes);
    }
    out
}
