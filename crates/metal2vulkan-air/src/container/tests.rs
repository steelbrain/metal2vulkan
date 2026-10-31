use super::*;
use crate::bitstream::writer::Writer;
use crate::bitstream::{Entry, Reader};
use crate::budget::{Budget, Meter};
use crate::error::ContainerError;
use crate::error::{Feature, Unsupported};
use metal2vulkan_devtools::metallib::{universal, UniversalHeader};

/// A small bitcode module whose content depends on `seed`: one block with one record.
fn module_stream(seed: u64) -> Vec<u8> {
    let mut w = Writer::bitcode();
    w.enter(13, 3);
    w.unabbrev(1, &[seed, seed + 1]);
    w.end();
    w.finish()
}

/// `stream` in a bitcode wrapper, followed by `pad` bytes inside the wrapped region.
fn wrap(stream: &[u8], pad: usize) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&WRAPPER_MAGIC);
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&20u32.to_le_bytes());
    out.extend_from_slice(&(stream.len() as u32).to_le_bytes());
    out.extend_from_slice(&u32::MAX.to_le_bytes());
    out.extend_from_slice(stream);
    out.extend(std::iter::repeat_n(0, pad));
    out
}

fn tag(out: &mut Vec<u8>, fourcc: &[u8; 4], bytes: &[u8]) {
    out.extend_from_slice(fourcc);
    out.extend_from_slice(&(bytes.len() as u16).to_le_bytes());
    out.extend_from_slice(bytes);
}

fn cstr(s: &str) -> Vec<u8> {
    let mut v = s.as_bytes().to_vec();
    v.push(0);
    v
}

#[derive(Clone)]
struct FnSpec {
    name: String,
    type_code: u8,
    bitcode: Vec<u8>,
    public: Vec<([u8; 4], Vec<u8>)>,
    private: Vec<([u8; 4], Vec<u8>)>,
    extra: Vec<([u8; 4], Vec<u8>)>,
    /// Replace the computed hash.
    hash: Option<[u8; 32]>,
    /// Omit these per-function tags.
    omit: Vec<[u8; 4]>,
}

impl FnSpec {
    fn new(name: &str, type_code: u8, seed: u64) -> FnSpec {
        FnSpec {
            name: name.to_owned(),
            type_code,
            bitcode: wrap(&module_stream(seed), 12),
            public: vec![],
            private: vec![],
            extra: vec![],
            hash: None,
            omit: vec![],
        }
    }
}

/// Assembles a metallib in the layout Apple's toolchain writes (see `mtlb.rs`).
#[derive(Clone)]
struct Builder {
    version: (u16, u16),
    file_type: u8,
    functions: Vec<FnSpec>,
    extended: Vec<([u8; 4], Vec<u8>)>,
    /// Bytes appended after the bitcode section (reflection data in real files).
    trailer: Vec<u8>,
    /// Metadata record sizes count the size word itself (`metal` straight to a metallib) rather
    /// than only the tag list (`metallib` linking `.air` files).
    inclusive_md_size: bool,
    /// Every function's bitcode is the first function's, written once (a dynamic library).
    shared_bitcode: bool,
    /// Extended-header tags holding the `(offset, size)` of a region written after the trailer,
    /// with the region's bytes (`HDYN`, `ILST`).
    regions: Vec<([u8; 4], Vec<u8>)>,
}

impl Builder {
    fn new(functions: Vec<FnSpec>) -> Builder {
        Builder {
            version: (2, 9),
            file_type: 0,
            functions,
            extended: vec![],
            trailer: vec![],
            inclusive_md_size: false,
            shared_bitcode: false,
            regions: vec![],
        }
    }

    fn build(&self) -> Vec<u8> {
        let mut public = Vec::new();
        let mut private = Vec::new();
        let mut bitcode = Vec::new();
        let mut entries = Vec::new();
        for f in &self.functions {
            let md = |tags: &[([u8; 4], Vec<u8>)]| {
                let mut list = Vec::new();
                for (fourcc, bytes) in tags {
                    tag(&mut list, fourcc, bytes);
                }
                list.extend_from_slice(b"ENDT");
                let size = list.len() + if self.inclusive_md_size { 4 } else { 0 };
                let mut out = (size as u32).to_le_bytes().to_vec();
                out.extend(list);
                out
            };
            let (pub_at, priv_at) = (public.len(), private.len());
            public.extend(md(&f.public));
            private.extend(md(&f.private));
            let bc_at = match (self.shared_bitcode, bitcode.is_empty()) {
                (true, false) => 0,
                _ => {
                    let at = bitcode.len();
                    bitcode.extend_from_slice(&f.bitcode);
                    at
                }
            };
            let mut list = Vec::new();
            let mut put = |fourcc: &[u8; 4], bytes: &[u8]| {
                if !f.omit.contains(fourcc) {
                    tag(&mut list, fourcc, bytes);
                }
            };
            put(b"NAME", &cstr(&f.name));
            put(b"TYPE", &[f.type_code]);
            put(b"HASH", &f.hash.unwrap_or_else(|| sha256(&f.bitcode)));
            put(b"MDSZ", &(f.bitcode.len() as u64).to_le_bytes());
            let mut offt = Vec::new();
            for v in [pub_at, priv_at, bc_at] {
                offt.extend_from_slice(&(v as u64).to_le_bytes());
            }
            put(b"OFFT", &offt);
            put(b"VERS", &[2, 0, 8, 0, 4, 0, 0, 0]);
            for (fourcc, bytes) in &f.extra {
                tag(&mut list, fourcc, bytes);
            }
            list.extend_from_slice(b"ENDT");
            let mut entry = ((list.len() + 4) as u32).to_le_bytes().to_vec();
            entry.extend(list);
            entries.push(entry);
        }
        let entries_len: usize = entries.iter().map(Vec::len).sum();
        // Each region tag is 6 + 16 bytes; its offset is known once the layout is.
        let ext_len = self
            .extended
            .iter()
            .map(|(_, b)| 6 + b.len())
            .sum::<usize>()
            + self.regions.len() * 22
            + 4;
        let list_off = 0x58usize;
        let pub_off = list_off + 4 + entries_len + ext_len;
        let priv_off = pub_off + public.len();
        let bc_off = priv_off + private.len();
        let mut region_at = bc_off + bitcode.len() + self.trailer.len();
        let mut ext = Vec::new();
        for (fourcc, bytes) in &self.extended {
            tag(&mut ext, fourcc, bytes);
        }
        for (fourcc, bytes) in &self.regions {
            let mut at = (region_at as u64).to_le_bytes().to_vec();
            at.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
            tag(&mut ext, fourcc, &at);
            region_at += bytes.len();
        }
        ext.extend_from_slice(b"ENDT");
        assert_eq!(ext.len(), ext_len);
        let file_size = region_at;

        let mut out = Vec::new();
        out.extend_from_slice(b"MTLB");
        out.extend_from_slice(&0x8001u16.to_le_bytes());
        out.extend_from_slice(&self.version.0.to_le_bytes());
        out.extend_from_slice(&self.version.1.to_le_bytes());
        out.push(self.file_type);
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
        assert_eq!(out.len(), 0x58);
        out.extend_from_slice(&(self.functions.len() as u32).to_le_bytes());
        for e in entries {
            out.extend(e);
        }
        out.extend(ext);
        out.extend(public);
        out.extend(private);
        out.extend(bitcode);
        out.extend_from_slice(&self.trailer);
        for (_, bytes) in &self.regions {
            out.extend_from_slice(bytes);
        }
        assert_eq!(out.len(), file_size);
        out
    }
}

/// A tag list of `tags`, ending in `ENDT`.
fn tag_list(tags: &[(&[u8; 4], Vec<u8>)]) -> Vec<u8> {
    let mut out = Vec::new();
    for (fourcc, bytes) in tags {
        tag(&mut out, fourcc, bytes);
    }
    out.extend_from_slice(b"ENDT");
    out
}

/// An `HDYN` region: the install name and the dependencies' install names.
fn hdyn(install_name: &str, dependencies: &[&str]) -> Vec<u8> {
    let mut tags = vec![(b"NAME", cstr(install_name))];
    tags.extend(dependencies.iter().map(|d| (b"DYNL", cstr(d))));
    tag_list(&tags)
}

/// An `ILST` region: a count, then per symbol a record whose size counts its size word.
fn ilst(symbols: &[&str]) -> Vec<u8> {
    let mut out = (symbols.len() as u32).to_le_bytes().to_vec();
    for s in symbols {
        let list = tag_list(&[(b"NAME", cstr(s))]);
        out.extend_from_slice(&((list.len() + 4) as u32).to_le_bytes());
        out.extend(list);
    }
    out
}

/// A dynamic library exporting `symbols` over one module, installed as `libD`, linked against
/// `libA` and `libB`, and importing `imported`.
fn dylib(symbols: &[&str]) -> Builder {
    let functions = symbols.iter().map(|s| FnSpec::new(s, 3, 9)).collect();
    let mut b = Builder::new(functions);
    b.file_type = 2;
    b.shared_bitcode = true;
    b.regions = vec![
        (*b"HDYN", hdyn("libD", &["libA", "libB"])),
        (*b"ILST", ilst(&["imported"])),
    ];
    b.extended.push((*b"UUID", vec![7; 16]));
    b
}

/// A dynamic library's functions are its exports, over one module and one hash; its install
/// name, its dependencies (in link order), and its imports are read from `HDYN` and `ILST`.
#[test]
fn a_dynamic_library_parses_its_exports_and_linkage() {
    let bytes = dylib(&["_Z1fi", "_Z1gi"]).build();
    let lib = metallib(&bytes);
    assert_eq!(lib.library_type(), LibraryType::Dynamic);
    assert_eq!(lib.functions.len(), 2);
    for f in &lib.functions {
        assert_eq!(f.stage, FunctionStage::Exported);
        assert_eq!(f.bitcode, lib.functions[0].bitcode);
    }
    assert_eq!(lib.linkage.install_name, Some("libD"));
    assert_eq!(lib.linkage.dependencies, ["libA", "libB"]);
    assert_eq!(lib.linkage.imports, ["imported"]);
    assert_eq!(lib.linkage.module, Some(lib.functions[0].bitcode));
    // An executable names itself and its dependencies the same way.
    let mut exe = sample();
    exe.regions = vec![(*b"HDYN", hdyn("client.metallib", &["libD"]))];
    let bytes = exe.build();
    let lib = metallib(&bytes);
    assert_eq!(lib.library_type(), LibraryType::Executable);
    assert_eq!(lib.linkage.install_name, Some("client.metallib"));
    assert_eq!(lib.linkage.dependencies, ["libD"]);
    let plain = sample().build();
    assert_eq!(metallib(&plain).linkage, DynamicLinkage::default());
}

/// Only a dynamic library's functions may share a bitcode range, and only exactly.
#[test]
fn only_a_dynamic_library_shares_bitcode() {
    let mut exe = dylib(&["a", "b"]);
    exe.file_type = 0;
    for f in &mut exe.functions {
        f.type_code = 2;
    }
    let err = parse(&exe.build()).unwrap_err();
    assert_eq!(container_kind(&err), ContainerError::BadOffset, "{err:?}");
    // A hash that does not match the shared module is still refused.
    let mut bad = dylib(&["a", "b"]);
    bad.functions[1].hash = Some([0; 32]);
    let err = parse(&bad.build()).unwrap_err();
    assert_eq!(
        container_kind(&err),
        ContainerError::HashMismatch,
        "{err:?}"
    );
}

/// Malformed `HDYN` and `ILST` regions are container errors.
#[test]
fn malformed_linkage_regions() {
    let with = |regions: Vec<([u8; 4], Vec<u8>)>| {
        let mut b = dylib(&["f"]);
        b.regions = regions;
        parse(&b.build()).map(|_| ())
    };
    let kind = |r: Result<(), Error>| container_kind(&r.unwrap_err());
    // Two `HDYN`s, or two names in one.
    assert_eq!(
        kind(with(vec![
            (*b"HDYN", hdyn("x", &[])),
            (*b"HDYN", hdyn("y", &[]))
        ])),
        ContainerError::DuplicateTag
    );
    let two_names = tag_list(&[(b"NAME", cstr("x")), (b"NAME", cstr("y"))]);
    assert_eq!(
        kind(with(vec![(*b"HDYN", two_names)])),
        ContainerError::DuplicateTag
    );
    // An `ILST` record without a name, with a size past the region, or a count past it.
    let unnamed = {
        let list = tag_list(&[(b"XXXX", vec![1])]);
        let mut out = 1u32.to_le_bytes().to_vec();
        out.extend_from_slice(&((list.len() + 4) as u32).to_le_bytes());
        out.extend(list);
        out
    };
    assert_eq!(
        kind(with(vec![(*b"ILST", unnamed)])),
        ContainerError::MissingTag
    );
    let mut long = ilst(&["s"]);
    long[4] += 1;
    assert_eq!(kind(with(vec![(*b"ILST", long)])), ContainerError::BadTag);
    let mut counted = ilst(&["s"]);
    counted[0] = 9;
    assert_eq!(
        kind(with(vec![(*b"ILST", counted)])),
        ContainerError::BadOffset
    );
    // A region outside the file.
    let mut b = dylib(&["f"]);
    b.regions = vec![(*b"HDYN", hdyn("x", &[]))];
    let mut bytes = b.build();
    let at = find(&bytes, b"HDYN") + 6;
    bytes[at..at + 8].copy_from_slice(&u64::MAX.to_le_bytes());
    assert_eq!(
        container_kind(&parse(&bytes).unwrap_err()),
        ContainerError::BadOffset
    );
}

/// A universal file is read through its AIR slice, in each of the three headers, and the result's
/// bytes are that slice. Metal's own file without an AIR slice is `NoAirSlice`; a Mach-O one is
/// `BadMagic`, as other files of its magic are; an AIR slice that is no metallib is `BadMagic`; a
/// file cut inside its records or its slice is an error.
#[test]
fn a_universal_metallib_is_read_through_its_air_slice() {
    let air = dylib(&["f"]).build();
    let gpu = vec![0xCF, 0xFA, 0xED, 0xFE, 1, 2, 3, 4];
    for header in UniversalHeader::ALL {
        let bytes = universal(header, &[(0x0100_0013, &gpu), (0x0100_0017, &air)]);
        let lib = metallib(&bytes);
        assert_eq!(lib.bytes, &air[..], "{header:?}");
        assert_eq!(lib.linkage.install_name, Some("libD"), "{header:?}");
        assert_eq!(lib.functions.len(), 1, "{header:?}");
        let missing = match header {
            UniversalHeader::Metal => ContainerError::NoAirSlice,
            UniversalHeader::MachO | UniversalHeader::MachO64 => ContainerError::BadMagic,
        };
        let none = universal(header, &[(0x0100_0013, &gpu)]);
        assert_eq!(
            container_kind(&parse(&none).unwrap_err()),
            missing,
            "{header:?}"
        );
        let not_mtlb = universal(header, &[(0x0100_0017, &gpu)]);
        assert_eq!(
            container_kind(&parse(&not_mtlb).unwrap_err()),
            ContainerError::BadMagic,
            "{header:?}"
        );
        for len in [6, 12, 40, bytes.len() - 1] {
            let mut cut = bytes.clone();
            cut.truncate(len);
            assert!(parse(&cut).is_err(), "{header:?} cut at {len}");
        }
    }
    // A Java class shares FAT_MAGIC: its version reads as a count of 65 records, which a short
    // file cannot hold and a long one holds no AIR slice among.
    for len in [8, 2048] {
        let mut class = FAT_MAGIC.to_vec();
        class.extend_from_slice(&[0, 0, 0, 65]);
        class.resize(len, 0);
        assert_eq!(
            container_kind(&parse(&class).unwrap_err()),
            ContainerError::BadMagic,
            "{len} bytes"
        );
    }
}

/// A `fat_arch_64` record's offset and size are each read as a 64-bit value: a high word that is
/// not zero in either puts the slice past the file.
#[test]
fn a_wide_universal_record_reads_64_bit_offsets_and_sizes() {
    let air = dylib(&["f"]).build();
    let bytes = universal(UniversalHeader::MachO64, &[(0x0100_0017, &air)]);
    assert_eq!(metallib(&bytes).bytes, &air[..]);
    // The record starts at byte 8: the offset's high word is bytes 16..20, the size's 24..28.
    for high in [19, 27] {
        let mut wide = bytes.clone();
        wide[high] = 1;
        assert_eq!(
            container_kind(&parse(&wide).unwrap_err()),
            ContainerError::BadOffset,
            "byte {high}"
        );
    }
}

/// A three-function library with public metadata and unknown tags everywhere they occur.
fn sample() -> Builder {
    let mut kernel = FnSpec::new("k_main", 2, 1);
    // CNST: 2 constants.
    let mut cnst = 2u16.to_le_bytes().to_vec();
    cnst.extend(cstr("use_b"));
    cnst.extend([0x35, 0, 0, 1]);
    cnst.extend(cstr("scale"));
    cnst.extend([0x03, 5, 0, 1]);
    kernel.public.push((*b"CNST", cnst));
    kernel.public.push((*b"ARGR", vec![1, 2, 3]));
    kernel.private.push((*b"PRIV", vec![9]));
    kernel.extra.push((*b"RFLT", 0u64.to_le_bytes().to_vec()));

    let mut vertex = FnSpec::new("v_main", 0, 2);
    let mut vatt = 2u16.to_le_bytes().to_vec();
    vatt.extend(cstr("pos"));
    vatt.extend(0x8000u16.to_le_bytes());
    vatt.extend(cstr("uv"));
    vatt.extend(0x8003u16.to_le_bytes());
    vertex.public.push((*b"VATT", vatt));
    vertex.public.push((*b"VATY", vec![2, 0, 6, 0x11]));
    vertex.extra.push((*b"LAYR", vec![1]));

    let fragment = FnSpec::new("f_main", 1, 3);
    let mut b = Builder::new(vec![kernel, vertex, fragment]);
    b.extended.push((*b"UUID", vec![7; 16]));
    b.trailer = vec![0xAA; 20];
    b
}

fn container_kind(e: &Error) -> ContainerError {
    match e {
        Error::InvalidInput(InputError {
            kind: InputErrorKind::Container(k),
            ..
        }) => k.clone(),
        other => panic!("expected a container error, got {other:?}"),
    }
}

fn metallib(bytes: &[u8]) -> Metallib<'_> {
    match parse(bytes).unwrap() {
        Library::Metallib(m) => m,
        other => panic!("expected a metallib, got {other:?}"),
    }
}

/// Walk a bitstream to its end; returns the number of entries.
fn walk(stream: &[u8]) -> Result<usize, Error> {
    let mut m = Meter::new(&Budget::default());
    let mut r = Reader::new(stream)?;
    let mut n = 0;
    loop {
        if r.advance(&mut m)? == Entry::End {
            return Ok(n);
        }
        n += 1;
    }
}

#[test]
fn parses_every_function_and_its_metadata() {
    let bytes = sample().build();
    let lib = metallib(&bytes);
    assert_eq!(lib.header.version, (2, 9));
    assert_eq!(lib.header.platform, 0x8001);
    assert_eq!(lib.header.file_size, bytes.len() as u64);
    assert_eq!(
        lib.extended,
        vec![RawTag {
            fourcc: *b"UUID",
            bytes: &[7; 16]
        }]
    );
    let names: Vec<_> = lib.functions.iter().map(|f| (f.name, f.stage)).collect();
    assert_eq!(
        names,
        vec![
            ("k_main", FunctionStage::Kernel),
            ("v_main", FunctionStage::Vertex),
            ("f_main", FunctionStage::Fragment),
        ]
    );
    let k = &lib.functions[0];
    assert_eq!(k.air_version, (2, 8));
    assert_eq!(k.language_version, (4, 0));
    assert_eq!(k.hash, sha256(k.bitcode));
    assert_eq!(
        k.public.constants,
        vec![
            ContainerConstant {
                name: "use_b",
                data_type: 0x35,
                index: 0,
                flags: 1
            },
            ContainerConstant {
                name: "scale",
                data_type: 3,
                index: 5,
                flags: 1
            },
        ]
    );
    assert_eq!(k.public.raw_tags[0].fourcc, *b"ARGR");
    assert_eq!(k.private[0].fourcc, *b"PRIV");
    assert_eq!(k.raw_tags[0].fourcc, *b"RFLT");
    let v = &lib.functions[1];
    assert_eq!(v.name, "v_main");
    assert_eq!(
        v.public.vertex_attributes,
        vec![
            VertexAttribute {
                name: "pos",
                packed: 0x8000
            },
            VertexAttribute {
                name: "uv",
                packed: 0x8003
            },
        ]
    );
    assert_eq!(v.public.vertex_attribute_types, vec![6, 0x11]);
    assert_eq!(v.raw_tags[0].fourcc, *b"LAYR");
    // Every function is its own bitcode module, and the streams differ.
    let mut streams = Vec::new();
    for f in &lib.functions {
        let stream = f.stream().unwrap();
        assert_eq!(walk(stream).unwrap(), 3);
        streams.push(stream);
    }
    assert_ne!(streams[0], streams[1]);
}

#[test]
fn stage_codes() {
    for (code, stage) in [
        (0, FunctionStage::Vertex),
        (1, FunctionStage::Fragment),
        (2, FunctionStage::Kernel),
        (4, FunctionStage::Visible),
        (6, FunctionStage::Intersection),
        (7, FunctionStage::Mesh),
        (8, FunctionStage::Object),
        (3, FunctionStage::Exported),
        (5, FunctionStage::Extern),
        (9, FunctionStage::Other(9)),
    ] {
        assert_eq!(FunctionStage::from_type_tag(code), stage);
    }
}

#[test]
fn hash_is_verified() {
    let mut b = sample();
    b.functions[1].hash = Some([0; 32]);
    let bytes = b.build();
    let err = parse(&bytes).unwrap_err();
    assert_eq!(container_kind(&err), ContainerError::HashMismatch);
    // Flipping one bitcode byte also fails the hash.
    let bytes = sample().build();
    let lib = metallib(&bytes);
    let at = lib.functions[2].bitcode.as_ptr() as usize - bytes.as_ptr() as usize + 25;
    let mut flipped = bytes.clone();
    flipped[at] ^= 1;
    assert_eq!(
        container_kind(&parse(&flipped).unwrap_err()),
        ContainerError::HashMismatch
    );
}

#[test]
fn unsupported_versions_and_library_kinds() {
    let mut b = sample();
    b.version = (3, 0);
    assert_eq!(
        container_kind(&parse(&b.build()).unwrap_err()),
        ContainerError::UnsupportedContainerVersion { major: 3, minor: 0 }
    );
    let mut b = sample();
    b.version = (1, 2);
    assert!(parse(&b.build()).is_ok());
    // A Core Image kernel library (`file_type` 1) is read; a kind Apple's toolchain is not seen
    // to write is refused.
    let mut b = sample();
    b.file_type = 1;
    match parse(&b.build()) {
        Ok(Library::Metallib(ml)) => assert_eq!(ml.library_type(), LibraryType::Executable),
        other => panic!("{other:?}"),
    }
    let mut b = sample();
    b.file_type = 3;
    match parse(&b.build()).unwrap_err() {
        Error::Unsupported(Unsupported {
            feature: Feature::LibraryKind(3),
            ..
        }) => {}
        other => panic!("{other:?}"),
    }
}

#[test]
fn malformed_function_tags() {
    let check = |edit: &dyn Fn(&mut FnSpec), want: ContainerError| {
        let mut b = sample();
        edit(&mut b.functions[0]);
        let err = parse(&b.build()).unwrap_err();
        assert_eq!(container_kind(&err), want, "{err:?}");
    };
    for missing in [*b"NAME", *b"TYPE", *b"HASH", *b"MDSZ", *b"OFFT", *b"VERS"] {
        check(&|f| f.omit.push(missing), ContainerError::MissingTag);
    }
    check(
        &|f| f.extra.push((*b"TYPE", vec![2])),
        ContainerError::DuplicateTag,
    );
    check(
        &|f| f.extra.push((*b"NAME", cstr("again"))),
        ContainerError::DuplicateTag,
    );
    // TYPE with two bytes; NAME without NUL, with bytes after the NUL, or not UTF-8.
    check(
        &|f| {
            f.omit.push(*b"TYPE");
            f.extra.push((*b"TYPE", vec![2, 0]));
        },
        ContainerError::BadTag,
    );
    check(
        &|f| {
            f.omit.push(*b"NAME");
            f.extra.push((*b"NAME", b"abc".to_vec()));
        },
        ContainerError::BadTag,
    );
    check(
        &|f| {
            f.omit.push(*b"NAME");
            f.extra.push((*b"NAME", b"ab\0c\0".to_vec()));
        },
        ContainerError::BadTag,
    );
    check(
        &|f| {
            f.omit.push(*b"NAME");
            f.extra.push((*b"NAME", vec![0xFF, 0]));
        },
        ContainerError::BadTag,
    );
    // Public metadata: short or overlong CNST, duplicate VATY, VATY count mismatch.
    check(
        &|f| f.public.push((*b"CNST", vec![1, 0, b'a', 0, 3])),
        ContainerError::BadTag,
    );
    check(
        &|f| f.public.push((*b"VATY", vec![1, 0, 6, 7])),
        ContainerError::BadTag,
    );
    check(
        &|f| {
            f.public.push((*b"VATY", vec![0, 0]));
            f.public.push((*b"VATY", vec![0, 0]));
        },
        ContainerError::DuplicateTag,
    );
    check(
        &|f| f.public.push((*b"VATT", vec![0xFF, 0xFF])),
        ContainerError::BadTag,
    );
}

/// The file offset of the first occurrence of `needle` in `hay`.
fn find(hay: &[u8], needle: &[u8]) -> usize {
    hay.windows(needle.len())
        .position(|w| w == needle)
        .expect("needle")
}

#[test]
fn malformed_layout() {
    let good = sample().build();
    let set_u64 = |b: &mut Vec<u8>, at: usize, v: u64| {
        b[at..at + 8].copy_from_slice(&v.to_le_bytes());
    };
    let expect = |b: &[u8], want: ContainerError| {
        let err = parse(b).unwrap_err();
        assert_eq!(container_kind(&err), want, "{err:?}");
    };
    // Bad magic, short header, file size beyond the input.
    let mut b = good.clone();
    b[0] = b'X';
    expect(&b, ContainerError::BadMagic);
    expect(
        &good[..0x40],
        ContainerError::Truncated {
            need: 0x58,
            have: 0x40,
        },
    );
    expect(&good[..2], ContainerError::Truncated { need: 4, have: 2 });
    let mut b = good.clone();
    set_u64(&mut b, 16, good.len() as u64 + 1);
    expect(
        &b,
        ContainerError::Truncated {
            need: good.len() as u64 + 1,
            have: good.len() as u64,
        },
    );
    // A section outside the file, or overlapping the header.
    for at in [24usize, 40, 56, 72] {
        let mut b = good.clone();
        set_u64(&mut b, at, good.len() as u64);
        expect(&b, ContainerError::BadOffset);
        let mut b = good.clone();
        set_u64(&mut b, at, 0x10);
        expect(&b, ContainerError::BadOffset);
    }
    // Function list size that disagrees with the entries.
    let mut b = good.clone();
    let size = u64::from_le_bytes(b[32..40].try_into().unwrap());
    set_u64(&mut b, 32, size - 4);
    assert!(parse(&b).is_err());
    let mut b = good.clone();
    set_u64(&mut b, 32, size + 4);
    expect(&b, ContainerError::BadOffset);
    // A function count beyond what the list can hold.
    let mut b = good.clone();
    b[0x58..0x5C].copy_from_slice(&u32::MAX.to_le_bytes());
    expect(&b, ContainerError::BadOffset);
    // An entry size that runs past the list.
    let mut b = good.clone();
    b[0x5C..0x60].copy_from_slice(&0xFFFFu32.to_le_bytes());
    expect(&b, ContainerError::BadTag);
    // An MDSZ running past the bitcode section.
    let mut b = good.clone();
    let mdsz = find(&b, b"MDSZ") + 6;
    set_u64(&mut b, mdsz, 1 << 20);
    expect(&b, ContainerError::BadOffset);
    // A public-metadata offset outside its section, and a metadata size that runs past it.
    let mut b = good.clone();
    let offt = find(&b, b"OFFT") + 6;
    set_u64(&mut b, offt, 1 << 20);
    expect(&b, ContainerError::BadOffset);
    let mut b = good.clone();
    let pub_off = u64::from_le_bytes(b[40..48].try_into().unwrap()) as usize;
    b[pub_off..pub_off + 4].copy_from_slice(&0xFFFFu32.to_le_bytes());
    expect(&b, ContainerError::BadTag);
    // A metadata size that ends before its ENDT.
    let mut b = good.clone();
    let size = u32::from_le_bytes(b[pub_off..pub_off + 4].try_into().unwrap());
    b[pub_off..pub_off + 4].copy_from_slice(&(size - 1).to_le_bytes());
    expect(&b, ContainerError::BadTag);
}

#[test]
fn overlapping_bitcode_is_rejected() {
    let bytes = sample().build();
    let mut b = bytes.clone();
    // Point the second function's bitcode at the first one's.
    let second = find(&b[find(&b, b"v_main")..], b"OFFT") + find(&b, b"v_main") + 6 + 16;
    b[second..second + 8].copy_from_slice(&0u64.to_le_bytes());
    let err = parse(&b).unwrap_err();
    assert_eq!(container_kind(&err), ContainerError::BadOffset, "{err:?}");
}

#[test]
fn shared_or_overlapping_metadata_is_rejected() {
    let bytes = sample().build();
    // The second function's OFFT: public, private, and bitcode offsets.
    let offt = find(&bytes[find(&bytes, b"v_main")..], b"OFFT") + find(&bytes, b"v_main") + 6;
    // Point its public or private record at the first function's (offset 0), or into the middle of
    // it (just past the size word).
    for (field, at) in [(0, 0u64), (8, 0), (0, 4), (8, 4)] {
        let mut b = bytes.clone();
        b[offt + field..offt + field + 8].copy_from_slice(&at.to_le_bytes());
        let err = parse(&b).unwrap_err();
        assert_eq!(
            container_kind(&err),
            ContainerError::BadOffset,
            "{field}/{at}: {err:?}"
        );
    }
}

#[test]
fn wrapper_and_raw_modules() {
    let stream = module_stream(4);
    let wrapped = wrap(&stream, 12);
    let Library::Module(m) = parse(&wrapped).unwrap() else {
        panic!()
    };
    assert_eq!(m.stream, &stream[..]);
    assert_eq!(m.hash, sha256(&wrapped));
    assert_eq!(
        m.wrapper,
        Some(WrapperHeader {
            version: 0,
            offset: 20,
            size: stream.len() as u32,
            cputype: u32::MAX
        })
    );
    assert_eq!(walk(m.stream).unwrap(), 3);

    let Library::Module(raw) = parse(&stream).unwrap() else {
        panic!()
    };
    assert_eq!(raw.wrapper, None);
    assert_eq!(raw.stream, &stream[..]);

    // Stream range outside the input, inside the header, or not bitcode.
    let mut b = wrapped.clone();
    b[12..16].copy_from_slice(&(stream.len() as u32 + 100).to_le_bytes());
    assert!(matches!(
        container_kind(&parse(&b).unwrap_err()),
        ContainerError::Truncated { .. }
    ));
    let mut b = wrapped.clone();
    b[8..12].copy_from_slice(&4u32.to_le_bytes());
    assert_eq!(
        container_kind(&parse(&b).unwrap_err()),
        ContainerError::BadOffset
    );
    let mut b = wrapped.clone();
    b[20] = b'X';
    assert_eq!(
        container_kind(&parse(&b).unwrap_err()),
        ContainerError::BadMagic
    );
    assert!(matches!(
        container_kind(&parse(&wrapped[..10]).unwrap_err()),
        ContainerError::Truncated { need: 20, .. }
    ));
}

#[test]
fn truncation_anywhere_is_an_error() {
    let bytes = sample().build();
    for cut in 0..bytes.len() - sample().trailer.len() {
        let err = parse(&bytes[..cut]).expect_err("truncated metallib must fail");
        assert!(matches!(err, Error::InvalidInput(_)), "cut {cut}: {err:?}");
    }
}

#[test]
fn bit_flips_never_panic() {
    let bytes = sample().build();
    let bits = bytes.len() as u64 * 8;
    let mut rng = crate::bitstream::writer::XorShift(0xD1B5_4A32_D192_ED03);
    let mut failed = 0;
    for _ in 0..4000 {
        let mut b = bytes.clone();
        for _ in 0..1 + rng.next() % 3 {
            let bit = rng.next() % bits;
            b[(bit / 8) as usize] ^= 1 << (bit % 8);
        }
        match parse(&b) {
            Ok(Library::Metallib(lib)) => {
                // Anything accepted still has verified hashes and walkable wrappers or a typed error.
                for f in &lib.functions {
                    assert_eq!(sha256(f.bitcode), f.hash);
                    if let Ok(stream) = f.stream() {
                        let _ = walk(stream);
                    }
                }
            }
            Ok(Library::Module(_)) => panic!("an MTLB never parses as a module"),
            Err(Error::InvalidInput(_) | Error::InvalidRequest(_) | Error::Unsupported(_)) => {
                failed += 1
            }
            Err(other) => panic!("unexpected error class: {other:?}"),
        }
    }
    assert!(failed > 0);
}

/// Parses the committed Apple AIR bitcode of `engine/` and, where `xcrun metallib` exists, links
/// it into a real metallib and parses that too (the metallib half skips visibly elsewhere; CPU
/// only). Every `.air` the directory holds joins that link, so one left there beside the committed
/// set fails this: `xcrun metallib` resolves each shared `air.*` symbol to one definition for the
/// whole link, and which deployment target it comes from depends on the whole input set.
/// `scripts/regen-air-fixtures` writes none, and `git status` names one that is there.
#[test]
fn committed_air_fixtures_parse_and_link() {
    let dir = metal2vulkan_devtools::engine_dir();
    let mut airs: Vec<_> = std::fs::read_dir(&dir)
        .map(|d| {
            d.filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().is_some_and(|x| x == "air"))
                .collect()
        })
        .unwrap_or_default();
    airs.sort();
    assert!(!airs.is_empty(), "{} commits .air bitcode", dir.display());
    // A fixture may hold several kernels; the metallib has one function per entry point.
    let mut entries = 0;
    for air in &airs {
        let bytes = std::fs::read(air).unwrap();
        let Library::Module(m) = parse(&bytes).unwrap() else {
            panic!("{} is not a module", air.display())
        };
        assert!(m.wrapper.is_some(), "{}", air.display());
        assert!(walk(m.stream).unwrap() > 0, "{}", air.display());
        let decoded = crate::decode(&bytes, &mut Meter::new(&Budget::default())).unwrap();
        entries += decoded.abi.entries.len();
    }
    let has_metallib = std::process::Command::new("xcrun")
        .args(["-sdk", "macosx", "-f", "metallib"])
        .output()
        .is_ok_and(|o| o.status.success());
    if !has_metallib {
        eprintln!("SKIPPED metallib half: xcrun metallib not available");
        return;
    }
    let out = std::env::temp_dir().join(format!("m2v-container-{}.metallib", std::process::id()));
    let status = std::process::Command::new("xcrun")
        .args(["-sdk", "macosx", "metallib"])
        .args(&airs)
        .arg("-o")
        .arg(&out)
        .status()
        .unwrap();
    let bytes = std::fs::read(&out);
    let _ = std::fs::remove_file(&out);
    assert!(status.success());
    let bytes = bytes.unwrap();
    let lib = metallib(&bytes);
    assert_eq!(lib.functions.len(), entries);
    for f in &lib.functions {
        assert!(walk(f.stream().unwrap()).unwrap() > 0, "{}", f.name);
    }
}

#[test]
fn both_metadata_size_conventions_parse_and_nothing_else() {
    let mut b = sample();
    b.inclusive_md_size = true;
    let inclusive = b.build();
    let exclusive = sample().build();
    let (a, c) = (metallib(&inclusive), metallib(&exclusive));
    assert_eq!(a.functions, c.functions);
    // Any other size is rejected: bump the first public record's size by 4 over inclusive.
    let pub_off = u64::from_le_bytes(inclusive[40..48].try_into().unwrap()) as usize;
    let size = u32::from_le_bytes(inclusive[pub_off..pub_off + 4].try_into().unwrap());
    for delta in [1u32, 2, 3, 4] {
        let mut bad = inclusive.clone();
        bad[pub_off..pub_off + 4].copy_from_slice(&(size + delta).to_le_bytes());
        let err = parse(&bad).unwrap_err();
        assert_eq!(
            container_kind(&err),
            ContainerError::BadTag,
            "{delta}: {err:?}"
        );
    }
}
