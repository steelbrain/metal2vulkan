//! A large metallib's content hash is computed beside its parse (`library/function.rs`,
//! `parse_hashed`): the two passes over its bytes, the content hash and the check of each
//! function's `HASH`, run side by side from [`HASHED_BESIDE`] bytes on, and give the library what
//! the inline path gives over the same bytes: a universal file holding the metallib as its AIR
//! slice takes the inline path (its content is the slice, which only the parse finds), so the two
//! are compared byte for byte. Below that size, and for a refused library, nothing changes. No GPU
//! is used.

use metal2vulkan_devtools::metallib::{metallib, universal, wrap, MtlbFunction, UniversalHeader};

use crate::air::container;
use crate::library::function::{HASHED_BESIDE, HASHED_BESIDE_PARSES};
use crate::library::Library;

/// A metallib of `count` functions, each its own `size` bytes of bitcode (a bitcode wrapper around
/// bytes no test decodes), and its bytes.
fn library_of(count: usize, size: usize) -> Vec<u8> {
    let names: Vec<String> = (0..count).map(|i| format!("kernel_{i:04}")).collect();
    let functions: Vec<MtlbFunction<'_>> = (names.iter().enumerate())
        .map(|(i, name)| {
            let mut stream = b"BC\xc0\xde".to_vec();
            stream.extend((0..size).map(|at| (at ^ i) as u8));
            MtlbFunction {
                name,
                type_code: 2,
                bitcode: wrap(&stream),
                constants: Vec::new(),
                public: Vec::new(),
                tags: Vec::new(),
            }
        })
        .collect();
    metallib(&functions)
}

/// How many parses on this thread hashed beside their parse, around `parse`.
fn beside(parse: impl FnOnce()) -> u64 {
    let before = HASHED_BESIDE_PARSES.with(|n| n.get());
    parse();
    HASHED_BESIDE_PARSES.with(|n| n.get()) - before
}

/// The AIR slice's CPU type in a universal file.
const AIR_SLICE: u32 = 0x0100_0017;

#[test]
fn a_large_metallib_is_hashed_beside_its_parse_to_the_inline_hash() {
    let bytes = library_of(64, 32 << 10);
    assert!(bytes.len() >= HASHED_BESIDE, "{} bytes", bytes.len());
    let wrapped = universal(UniversalHeader::MachO, &[(AIR_SLICE, &bytes)]);
    for parse in [Library::from_metallib, Library::parse] {
        let mut inline = None;
        let counted = beside(|| inline = Some(parse(&wrapped).expect("the universal file parses")));
        assert_eq!(
            counted, 0,
            "a universal file's content hash is computed after its parse"
        );
        let inline = inline.expect("parsed");
        let mut library = None;
        let counted = beside(|| library = Some(parse(&bytes).expect("the metallib parses")));
        assert_eq!(counted, 1, "its content hash was computed beside its parse");
        let library = library.expect("parsed");
        assert_eq!(library.content_hash(), inline.content_hash());
        assert_eq!(library.content_hash(), &container::sha256(&bytes));
        assert_eq!(library.bytes(), inline.bytes());
        let table = |l: &Library| {
            l.functions()
                .map(|f| (f.name.to_owned(), f.stage, f.air_hash))
                .collect::<Vec<_>>()
        };
        assert_eq!(table(&library), table(&inline));
        assert!(library == inline, "the whole library is the inline path's");
        assert_eq!(library.len(), 64);
    }
}

#[test]
fn a_small_metallib_is_hashed_after_its_parse_to_the_same_hash() {
    let bytes = library_of(4, 1 << 10);
    assert!(bytes.len() < HASHED_BESIDE, "{} bytes", bytes.len());
    let mut library = None;
    let counted = beside(|| library = Some(Library::from_metallib(&bytes).expect("parses")));
    assert_eq!(counted, 0, "no thread for a small library");
    let library = library.expect("parsed");
    assert_eq!(library.content_hash(), &container::sha256(&bytes));
}

#[test]
fn a_large_metallib_whose_hash_tag_is_wrong_is_refused_as_inline() {
    let mut bytes = library_of(64, 32 << 10);
    // The last byte of the last function's bitcode: its `HASH` no longer matches it.
    let last = bytes.len() - 1;
    bytes[last] ^= 1;
    let wrapped = universal(UniversalHeader::MachO, &[(AIR_SLICE, &bytes)]);
    let inline = Library::from_metallib(&wrapped).expect_err("refused inline");
    let mut refused = None;
    assert_eq!(beside(|| refused = Some(Library::from_metallib(&bytes))), 1);
    let refused = refused.expect("ran").expect_err("refused beside");
    assert_eq!(
        format!("{refused:?}"),
        format!("{inline:?}"),
        "the same refusal, its hash beside or not"
    );
}
