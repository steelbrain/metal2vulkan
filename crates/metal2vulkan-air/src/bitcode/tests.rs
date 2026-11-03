//! Bitcode reader tests for the module tables: hand-assembled modules (valid and malformed),
//! deterministic mutation robustness, and the public fixtures assembled at test time. Function
//! bodies have their own hand-assembled tests in `function_tests`.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU32, Ordering as AtomicOrdering};

use super::*;
use crate::bitstream::writer::{DefOp, Enc, Field, Writer, XorShift};
use crate::budget::Budget;
use crate::container;
use crate::error::{BitcodeError, InputErrorKind};
use crate::error::{Feature, Limit};
use crate::ir::{
    Const, FuncId, Linkage, Md, MemoryEffects, ModRef, Module, ParamAttrs, Section, Type,
    UnnamedAddr, Visibility,
};

pub(super) fn meter() -> Meter {
    Meter::new(&Budget::default())
}

pub(super) fn chars(s: &str) -> Vec<u64> {
    s.bytes().map(u64::from).collect()
}

/// A record with a blob, through a one-off abbreviation (`[literal code, vbr6 x n, blob]`).
pub(super) fn blob_record(w: &mut Writer, code: u64, ops: &[u64], blob: &[u8]) {
    let mut def = vec![DefOp::Literal(code)];
    def.extend(ops.iter().map(|_| DefOp::Vbr(6)));
    def.push(DefOp::Blob);
    w.define(&def);
    let mut encs = vec![Enc::Literal];
    encs.extend(ops.iter().map(|_| Enc::Vbr(6)));
    encs.push(Enc::Blob);
    let mut fields = vec![Field::Scalar(code)];
    fields.extend(ops.iter().map(|&v| Field::Scalar(v)));
    fields.push(Field::Blob(blob.to_vec()));
    // Application abbrev ids start at 4; each call site defines exactly one per block.
    w.abbreviated(4, &encs, &fields);
}

/// A bitcode stream: identification (epoch `epoch`), a module whose body `body` writes after
/// `VERSION = version`, and a string table.
pub(super) fn stream(
    version: u64,
    epoch: u64,
    strtab: &[u8],
    body: impl FnOnce(&mut Writer),
) -> Vec<u8> {
    let mut w = Writer::bitcode();
    w.enter(u64::from(codes::IDENTIFICATION_BLOCK), 5);
    w.unabbrev(1, &chars("test"));
    w.unabbrev(2, &[epoch]);
    w.end();
    w.enter(u64::from(codes::MODULE_BLOCK), 3);
    w.unabbrev(1, &[version]);
    body(&mut w);
    w.end();
    w.enter(u64::from(codes::STRTAB_BLOCK), 3);
    blob_record(&mut w, 1, &[], strtab);
    w.end();
    w.finish()
}

pub(super) fn block(w: &mut Writer, id: u32, records: &[(u64, Vec<u64>)]) {
    w.enter(u64::from(id), 4);
    for (code, ops) in records {
        w.unabbrev(*code, ops);
    }
    w.end();
}

/// LSB-first VBR6 bits, padded to a 32-bit word (the `METADATA_STRINGS` length stream).
fn vbr6_words(values: &[u64]) -> Vec<u8> {
    let mut bits = Vec::new();
    for &v in values {
        let mut v = v;
        loop {
            let chunk = v & 0x1F;
            v >>= 5;
            let c = if v == 0 { chunk } else { chunk | 0x20 };
            for i in 0..6 {
                bits.push((c >> i) & 1);
            }
            if v == 0 {
                break;
            }
        }
    }
    while bits.len() % 32 != 0 {
        bits.push(0);
    }
    bits.chunks(8)
        .map(|b| b.iter().enumerate().map(|(i, &x)| (x as u8) << i).sum())
        .collect()
}

pub(super) fn md_strings(w: &mut Writer, strings: &[&str]) {
    let lengths: Vec<u64> = strings.iter().map(|s| s.len() as u64).collect();
    let mut blob = vbr6_words(&lengths);
    let offset = blob.len() as u64;
    for s in strings {
        blob.extend_from_slice(s.as_bytes());
    }
    blob_record(w, 35, &[strings.len() as u64, offset], &blob);
}

/// The type table of the valid synthetic module:
/// 0 `i32`, 1 `%S addrspace(1)*` (a forward reference), 2 `%S = { i32, %S addrspace(1)* }`,
/// 3 `void`, 4 `void (%S addrspace(1)*)`, 5 `[2 x i32]`.
fn good_types() -> Vec<(u64, Vec<u64>)> {
    vec![
        (1, vec![6]),
        (7, vec![32]),
        (8, vec![2, 1]),
        (19, chars("S")),
        (20, vec![0, 0, 1]),
        (2, vec![]),
        (21, vec![0, 3, 1]),
        (11, vec![2, 0]),
    ]
}

/// The body of the valid synthetic module. Values: 0 `@g`, 1 `@f`, 2 the aggregate `[i32 7, i32
/// 1]` (forward references to 3 and 4), 3 `i32 7`, 4 `i32 1`.
fn good_body(w: &mut Writer) {
    block(w, codes::TYPE_BLOCK, &good_types());
    w.unabbrev(3, &chars("e-p:64:64:64-i32:32:32"));
    w.unabbrev(5, &chars("air.static_init"));
    // @g: [2 x i32] addrspace(2) constant, init value 2, internal, align 4, section 1.
    w.unabbrev(7, &[0, 1, 5, 1 | 2 | (2 << 2), 3, 3, 3, 1]);
    // @f: declare void (%S addrspace(1)*).
    w.unabbrev(8, &[1, 1, 4, 0, 1, 0, 0, 0, 0]);
    block(
        w,
        codes::CONSTANTS_BLOCK,
        &[
            (1, vec![5]),
            (7, vec![3, 4]),
            (1, vec![0]),
            (4, vec![14]),
            (4, vec![2]),
        ],
    );
    w.enter(u64::from(codes::METADATA_KIND_BLOCK), 3);
    w.unabbrev(6, &[0, b'x'.into()]);
    w.unabbrev(6, &[[1].as_slice(), &chars("llvm.loop")].concat());
    w.end();
    w.enter(u64::from(codes::METADATA_BLOCK), 4);
    md_strings(w, &["air.x", "hi"]);
    w.unabbrev(2, &[0, 3]);
    w.unabbrev(3, &[1, 3, 0]);
    w.unabbrev(4, &chars("air.kernel"));
    w.unabbrev(10, &[3]);
    w.end();
    block(
        w,
        codes::SYNC_SCOPE_NAMES_BLOCK,
        &[(1, chars("singlethread")), (1, vec![])],
    );
}

fn good() -> Vec<u8> {
    stream(2, 0, b"gf", good_body)
}

/// Decode as translation does ([`DecodeOptions::default`]).
pub(super) fn decode(bytes: &[u8]) -> Result<ModuleTables, Error> {
    decode_module(bytes, DecodeOptions::default(), &mut meter())
}

/// Every option on: what a decode keeps for the writer and for comparisons across LLVM majors.
pub(super) const KEEP_ALL: DecodeOptions = DecodeOptions {
    enum_attrs: true,
    drop_unused_markers: true,
};

/// Decode keeping the uninterpreted enum attributes.
fn decode_keeping_enums(bytes: &[u8]) -> Result<ModuleTables, Error> {
    let opts = DecodeOptions {
        enum_attrs: true,
        ..DecodeOptions::default()
    };
    decode_module(bytes, opts, &mut meter())
}

/// Decode for a comparison across LLVM majors: unused marker declarations removed.
pub(super) fn decode_compared(bytes: &[u8]) -> Result<ModuleTables, Error> {
    let opts = DecodeOptions {
        drop_unused_markers: true,
        ..DecodeOptions::default()
    };
    decode_module(bytes, opts, &mut meter())
}

/// Verify the decoded tables and bodies through the IR verifier.
pub(super) fn verify_module_level(t: &ModuleTables) -> Result<Module, Error> {
    Module::new(t.parts.clone(), &mut meter())
}

pub(super) fn bitcode_kind(e: &Error) -> Option<&BitcodeError> {
    match e {
        Error::InvalidInput(i) => match &i.kind {
            InputErrorKind::Bitcode(b) => Some(b),
            _ => None,
        },
        _ => None,
    }
}

pub(super) fn unsupported_feature(e: &Error) -> Option<&Feature> {
    match e {
        Error::Unsupported(u) => Some(&u.feature),
        _ => None,
    }
}

#[test]
fn synthetic_module_decodes_every_module_level_table() {
    let t = decode(&good()).unwrap();
    let p = &t.parts;
    let sym = |s| p.symbols.get(s).unwrap();

    // Types: the pointer is erased; its pointee is kept by bitcode index (a forward reference).
    assert_eq!(t.types.len(), 6);
    assert_eq!(t.types.pointee(1), Some(2));
    let ty = |i| p.types.get(t.types.ty(i).unwrap()).unwrap().clone();
    assert_eq!(
        ty(1),
        Type::Ptr {
            space: crate::ir::AddrSpace::Device
        }
    );
    match ty(2) {
        Type::Struct { fields, name, .. } => {
            assert_eq!(fields.len(), 2);
            assert_eq!(sym(name.unwrap()), b"S");
        }
        other => panic!("not a struct: {other:?}"),
    }
    assert_eq!(t.types.params(4), Some(&[1u32][..]));

    // Globals and functions, named from the string table.
    let g = p.globals.values().next().unwrap();
    assert_eq!(sym(g.name), b"g");
    assert_eq!(g.space, crate::ir::AddrSpace::Constant);
    assert!(g.constant);
    assert_eq!(g.linkage, Linkage::Internal);
    assert_eq!(g.align.map(|a| a.bytes()), Some(4));
    assert_eq!(g.section, Some(Section::StaticInit));
    let init = p.consts.get(g.init.unwrap()).unwrap();
    let Const::Aggregate(elems) = &init.value else {
        panic!("initializer is not an aggregate: {init:?}");
    };
    let ints: Vec<_> = elems
        .iter()
        .map(|&e| p.consts.get(e).unwrap().value.clone())
        .collect();
    assert_eq!(ints, vec![Const::Int(7), Const::Int(1)]);
    let f = p.funcs.values().next().unwrap();
    assert_eq!(sym(f.name), b"f");
    assert!(f.body.is_none());
    assert_eq!(f.params.len(), 1);
    assert!(t.defined.is_empty());
    assert_eq!(t.function_blocks, 0);

    // Value numbering: globals, functions, then module constants in record order.
    assert_eq!(t.values.len(), 5);
    assert!(matches!(
        p.consts.get(t.value(0).unwrap()).unwrap().value,
        Const::GlobalAddr(_)
    ));
    assert!(matches!(
        p.consts.get(t.value(1).unwrap()).unwrap().value,
        Const::FuncAddr(_)
    ));
    // `@g` uses the explicit-type spelling and `@f` names its function type directly: neither
    // record names the value's own type, and each keeps the type of what it addresses.
    assert_eq!(t.values[0].bc_type, None);
    assert_eq!(t.values[0].object_bc_type, Some(5));
    assert_eq!(t.values[1].bc_type, None);
    assert_eq!(t.values[1].object_bc_type, Some(4));
    assert_eq!(t.values[2].bc_type, Some(5));
    for v in &t.values[2..] {
        assert!(v.bc_type.is_some());
        assert_eq!(v.object_bc_type, None);
    }

    // Metadata: two strings, a value, a node with a null operand, one named list.
    let md: Vec<_> = p.metadata.values().cloned().collect();
    assert_eq!(md.len(), 4);
    assert!(matches!(md[0], Md::String(s) if sym(s) == b"air.x"));
    assert!(matches!(md[1], Md::String(s) if sym(s) == b"hi"));
    assert_eq!(md[2], Md::Value(t.value(3).unwrap()));
    assert!(
        matches!(&md[3], Md::Node { distinct: false, ops } if ops.len() == 3 && ops[2].is_none())
    );
    assert_eq!(p.named_md.len(), 1);
    assert_eq!(sym(p.named_md[0].name), b"air.kernel");

    assert_eq!(t.md_kinds.loop_hints, Some(1));
    assert_eq!(t.md_kinds.range, None);
    assert_eq!(
        t.sync_scopes,
        vec![
            Some(crate::ir::SyncScope::SingleThread),
            Some(crate::ir::SyncScope::System)
        ]
    );
    assert_eq!(p.data_layout.pointer(0).size_bits, 64);

    verify_module_level(&t).unwrap();
}

#[test]
fn decoding_is_deterministic() {
    let bytes = good();
    assert_eq!(decode(&bytes).unwrap(), decode(&bytes).unwrap());
}

fn with_body(extra: impl FnOnce(&mut Writer)) -> Result<ModuleTables, Error> {
    decode(&stream(2, 0, b"gf", |w| {
        good_body(w);
        extra(w);
    }))
}

#[test]
fn version_and_identification_are_checked() {
    let e = decode(&stream(1, 0, b"", |_| {})).unwrap_err();
    assert_eq!(
        bitcode_kind(&e),
        Some(&BitcodeError::UnsupportedBitcodeVersion { version: 1 })
    );
    let e = decode(&stream(2, 1, b"", |_| {})).unwrap_err();
    assert_eq!(bitcode_kind(&e), Some(&BitcodeError::BadIdentification));
    // A module record before VERSION.
    let mut w = Writer::bitcode();
    w.enter(u64::from(codes::MODULE_BLOCK), 3);
    w.unabbrev(2, &chars("air64"));
    w.end();
    let e = decode(&w.finish()).unwrap_err();
    assert_eq!(
        bitcode_kind(&e),
        Some(&BitcodeError::MalformedRecord {
            block: codes::MODULE_BLOCK,
            code: codes::MODULE_VERSION
        })
    );
    // No module at all.
    let mut w = Writer::bitcode();
    w.enter(u64::from(codes::SYMTAB_BLOCK), 3);
    w.end();
    let e = decode(&w.finish()).unwrap_err();
    assert!(matches!(
        bitcode_kind(&e),
        Some(BitcodeError::MalformedRecord { .. })
    ));
    // Not a bitcode stream.
    let mut w = Writer::with_magic(*b"MTLB");
    w.enter(8, 3);
    w.end();
    assert!(decode(&w.finish()).is_err());
}

#[test]
fn unknown_and_unsupported_records_are_typed() {
    let e = with_body(|w| w.unabbrev(99, &[])).unwrap_err();
    assert_eq!(
        bitcode_kind(&e),
        Some(&BitcodeError::UnknownRecord {
            block: codes::MODULE_BLOCK,
            code: 99
        })
    );
    // MODULE_CODE_ALIAS and the legacy MODULE_CODE_PURGEVALS.
    let e = with_body(|w| w.unabbrev(14, &[0, 0, 0, 0])).unwrap_err();
    assert_eq!(
        unsupported_feature(&e),
        Some(&Feature::ModuleRecord { code: 14 })
    );
    let e = with_body(|w| w.unabbrev(10, &[0])).unwrap_err();
    assert_eq!(
        unsupported_feature(&e),
        Some(&Feature::ModuleRecord { code: 10 })
    );
    // LLVM 23's MODULE_CODE_ASM_PROPERTY qualifies module asm, itself refused.
    let e = with_body(|w| w.unabbrev(19, &chars("k\0v"))).unwrap_err();
    assert_eq!(
        unsupported_feature(&e),
        Some(&Feature::ModuleRecord { code: 19 })
    );
    // Legacy codes LLVM defines in other blocks: METADATA_OLD_NODE, PARAMATTR_CODE_ENTRY_OLD.
    let e = with_body(|w| block(w, codes::METADATA_BLOCK, &[(8, vec![0, 0])])).unwrap_err();
    assert_eq!(
        unsupported_feature(&e),
        Some(&Feature::BitcodeRecord {
            block: codes::METADATA_BLOCK,
            code: 8
        })
    );
    let e = with_body(|w| block(w, codes::PARAMATTR_BLOCK, &[(1, vec![0, 0])])).unwrap_err();
    assert_eq!(
        unsupported_feature(&e),
        Some(&Feature::BitcodeRecord {
            block: codes::PARAMATTR_BLOCK,
            code: 1
        })
    );
    // An unknown module sub-block.
    let e = with_body(|w| block(w, 60, &[])).unwrap_err();
    assert_eq!(
        bitcode_kind(&e),
        Some(&BitcodeError::UnknownBlock { block: 60 })
    );
    // A thread-local global.
    let e = with_body(|w| w.unabbrev(7, &[0, 1, 0, 2, 0, 0, 0, 0, 0, 1])).unwrap_err();
    assert_eq!(
        unsupported_feature(&e),
        Some(&Feature::ModuleRecord { code: 7 })
    );
    // A calling convention other than C or fast.
    let e = with_body(|w| w.unabbrev(8, &[1, 1, 4, 9, 1, 0, 0, 0, 0])).unwrap_err();
    assert_eq!(
        unsupported_feature(&e),
        Some(&Feature::CallingConvention(9))
    );
    // A global in an address space outside AIR's.
    let e = with_body(|w| w.unabbrev(7, &[0, 1, 0, 2 | (8 << 2), 0, 0, 0, 0])).unwrap_err();
    assert_eq!(unsupported_feature(&e), Some(&Feature::AddressSpace(8)));
}

fn types_only(records: &[(u64, Vec<u64>)]) -> Result<ModuleTables, Error> {
    decode(&stream(2, 0, b"", |w| block(w, codes::TYPE_BLOCK, records)))
}

#[test]
fn type_table_errors_are_typed() {
    // A struct containing itself by value is a cycle erasure cannot break.
    let e = types_only(&[(1, vec![1]), (20, vec![0, 0])]).unwrap_err();
    assert_eq!(bitcode_kind(&e), Some(&BitcodeError::BadType));
    // A reference past the table.
    let e = types_only(&[(1, vec![1]), (11, vec![4, 7])]).unwrap_err();
    assert_eq!(bitcode_kind(&e), Some(&BitcodeError::BadType));
    // More entries than NUMENTRY declared.
    let e = types_only(&[(1, vec![1]), (2, vec![]), (2, vec![])]).unwrap_err();
    assert!(matches!(
        bitcode_kind(&e),
        Some(BitcodeError::MalformedRecord { .. })
    ));
    // Fewer entries than declared.
    let e = types_only(&[(1, vec![3]), (2, vec![])]).unwrap_err();
    assert_eq!(bitcode_kind(&e), Some(&BitcodeError::BadType));
    // x86_fp80, token, i128, scalable vectors: valid LLVM outside the typed model.
    for (code, ops, feature_code) in [(13u64, vec![], 13u32), (22, vec![], 22), (7, vec![128], 7)] {
        let e = types_only(&[(1, vec![1]), (code, ops)]).unwrap_err();
        assert_eq!(
            unsupported_feature(&e),
            Some(&Feature::Type { code: feature_code })
        );
    }
    let e = types_only(&[(1, vec![2]), (7, vec![32]), (12, vec![4, 0, 1])]).unwrap_err();
    assert_eq!(unsupported_feature(&e), Some(&Feature::Type { code: 12 }));
    // LLVM 23's byte type (`b8`).
    let e = types_only(&[(1, vec![1]), (27, vec![8])]).unwrap_err();
    assert_eq!(unsupported_feature(&e), Some(&Feature::Type { code: 27 }));
    // An unknown address space and an unknown type code.
    let e = types_only(&[(1, vec![1]), (25, vec![9])]).unwrap_err();
    assert_eq!(unsupported_feature(&e), Some(&Feature::AddressSpace(9)));
    let e = types_only(&[(1, vec![1]), (77, vec![])]).unwrap_err();
    assert_eq!(
        bitcode_kind(&e),
        Some(&BitcodeError::UnknownRecord {
            block: codes::TYPE_BLOCK,
            code: 77
        })
    );
    // Zero-width integers are malformed.
    let e = types_only(&[(1, vec![1]), (7, vec![0])]).unwrap_err();
    assert!(matches!(
        bitcode_kind(&e),
        Some(BitcodeError::MalformedRecord { .. })
    ));
}

#[test]
fn opaque_pointers_have_no_pointee() {
    let t = types_only(&[(1, vec![2]), (25, vec![3]), (8, vec![0, 3])]).unwrap();
    assert_eq!(t.types.pointee(0), None);
    assert_eq!(t.types.pointee(1), Some(0));
    // Both erase to the same pointer type.
    assert_eq!(t.types.ty(0), t.types.ty(1));
}

/// A module with one attribute group (`records`) used by function `@f` as list 1.
fn with_attrs(group: &[u64]) -> Result<ModuleTables, Error> {
    decode_keeping_enums(&stream(2, 0, b"f", |w| {
        block(w, codes::TYPE_BLOCK, &good_types());
        block(w, codes::PARAMATTR_GROUP_BLOCK, &[(3, group.to_vec())]);
        block(w, codes::PARAMATTR_BLOCK, &[(2, vec![1])]);
        w.unabbrev(8, &[0, 1, 4, 0, 1, 0, 1, 0, 0]);
    }))
}

#[test]
fn attributes_decode_the_kept_kinds() {
    // Function level: convergent, argmemonly + writeonly, a string attribute.
    let mut group = vec![1, 0xFFFF_FFFF, 0, 43, 0, 45, 0, 52, 4];
    group.extend(chars("k"));
    group.push(0);
    group.extend(chars("v"));
    group.push(0);
    let t = with_attrs(&group).unwrap();
    let f = t.parts.funcs.values().next().unwrap();
    assert!(f.attrs.convergent);
    assert_eq!(
        f.attrs.memory,
        Some(MemoryEffects {
            argmem: ModRef::Mod,
            inaccessible: ModRef::NoModRef,
            other: ModRef::NoModRef,
        })
    );
    let (k, v) = f.attrs.strings[0];
    assert_eq!(t.parts.symbols.get(k), Some(&b"k"[..]));
    assert_eq!(t.parts.symbols.get(v), Some(&b"v"[..]));

    // `memory(argmem: write)` in the version-1 encoding (errno and target memory fold into other).
    let t = with_attrs(&[1, 0xFFFF_FFFF, 1, 86, (1 << 56) | 2 | (1 << 4)]).unwrap();
    let f = t.parts.funcs.values().next().unwrap();
    assert_eq!(
        f.attrs.memory,
        Some(MemoryEffects {
            argmem: ModRef::Mod,
            inaccessible: ModRef::NoModRef,
            other: ModRef::Ref,
        })
    );
    // Version 2 (LLVM 23) keeps version 1's layout.
    let t = with_attrs(&[1, 0xFFFF_FFFF, 1, 86, (2 << 56) | 2 | (1 << 4)]).unwrap();
    let f = t.parts.funcs.values().next().unwrap();
    assert_eq!(
        f.attrs.memory,
        Some(MemoryEffects {
            argmem: ModRef::Mod,
            inaccessible: ModRef::NoModRef,
            other: ModRef::Ref,
        })
    );
    // Version 0: argmem, inaccessible, other.
    let t = with_attrs(&[1, 0xFFFF_FFFF, 1, 86, 1 | (3 << 4)]).unwrap();
    let f = t.parts.funcs.values().next().unwrap();
    assert_eq!(
        f.attrs.memory,
        Some(MemoryEffects {
            argmem: ModRef::Ref,
            inaccessible: ModRef::NoModRef,
            other: ModRef::ModRef,
        })
    );

    // Parameter 1: align 16, noalias, readonly, captures(none), byval(i32), a range (dropped).
    let t = with_attrs(&[
        1, 1, 1, 1, 16, 0, 9, 0, 21, 1, 102, 0, 6, 3, 0, 7, 92, 32, 0, 20,
    ])
    .unwrap();
    let f = t.parts.funcs.values().next().unwrap();
    let a = &f.params[0].attrs;
    assert_eq!(a.align.map(|a| a.bytes()), Some(16));
    assert!(a.noalias && a.nocapture);
    assert_eq!(a.access, Some(ModRef::Ref));
    assert_eq!(a.byval, t.types.ty(0));
    // The untyped `byval` spelling takes the pointee of the parameter's typed pointer.
    let t = with_attrs(&[1, 1, 0, 3]).unwrap();
    let f = t.parts.funcs.values().next().unwrap();
    assert_eq!(f.params[0].attrs.byval, t.types.ty(2));
    verify_module_level(&t).unwrap();
}

/// The enum kinds of `t`'s first function.
fn first_enums(t: &ModuleTables) -> SiteEnums {
    t.enum_attrs
        .func(FuncId::new(0))
        .cloned()
        .unwrap_or_default()
}

/// Every enum attribute the IR does not interpret is recorded beside it when asked, for the
/// function, the return value, and each parameter, and groups for one position merge into their
/// union. A translation's decode records none.
#[test]
fn uninterpreted_enum_attributes_are_kept_per_position() {
    // Group 1: the function is nounwind (18) and willreturn (61), and convergent (interpreted);
    // group 2: the function is also speculatable (53); group 3: parameter 1 is noundef (68) and
    // noalias (interpreted); group 4: the return value is noundef.
    let bytes = stream(2, 0, b"f", |w| {
        block(w, codes::TYPE_BLOCK, &good_types());
        block(
            w,
            codes::PARAMATTR_GROUP_BLOCK,
            &[
                (3, vec![1, 0xFFFF_FFFF, 0, 18, 0, 61, 0, 43]),
                (3, vec![2, 0xFFFF_FFFF, 0, 53]),
                (3, vec![3, 1, 0, 68, 0, 9]),
                (3, vec![4, 0, 0, 68]),
            ],
        );
        block(w, codes::PARAMATTR_BLOCK, &[(2, vec![1, 2, 3, 4])]);
        w.unabbrev(8, &[0, 1, 4, 0, 1, 0, 1, 0, 0]);
    });
    let t = decode_keeping_enums(&bytes).unwrap();
    let f = t.parts.funcs.values().next().unwrap();
    assert!(f.attrs.convergent);
    let e = first_enums(&t);
    assert_eq!(
        e.func.kinds().collect::<Vec<_>>(),
        [
            EnumAttrs::NO_UNWIND,
            EnumAttrs::SPECULATABLE,
            EnumAttrs::WILL_RETURN
        ]
    );
    assert!(f.params[0].attrs.noalias);
    assert_eq!(e.param(0).kinds().collect::<Vec<_>>(), [68]);
    assert_eq!(e.ret.kinds().collect::<Vec<_>>(), [68]);
    verify_module_level(&t).unwrap();
    // As translation decodes it: the same module, nothing recorded beside it.
    let plain = decode(&bytes).unwrap();
    assert!(plain.enum_attrs.is_empty());
    assert_eq!(plain.parts, t.parts);
}

/// Visibility and `unnamed_addr` decode as LLVM's reader applies them: a symbol with local linkage
/// keeps the default visibility, an unknown value of either field is the default, and a record too
/// short to carry them has the defaults.
#[test]
fn visibility_and_unnamed_addr_decode_as_llvms_reader_applies_them() {
    let decoded = |global: &[u64], function: &[u64]| {
        let t = decode(&stream(2, 0, b"gf", |w| {
            block(w, codes::TYPE_BLOCK, &good_types());
            w.unabbrev(7, global);
            w.unabbrev(8, function);
        }))
        .unwrap();
        let g = t.parts.globals.values().next().unwrap();
        let f = t.parts.funcs.values().next().unwrap();
        (
            (g.visibility, g.unnamed_addr),
            (f.visibility, f.unnamed_addr),
        )
    };
    // GLOBALVAR [.., linkage 5, .., visibility 8, threadlocal 9, unnamed_addr 10];
    // FUNCTION [.., linkage 5, .., visibility 9, gc 10, unnamed_addr 11].
    let global = |linkage, vis, unnamed| vec![0, 1, 0, 2, 0, linkage, 0, 0, vis, 0, unnamed];
    let function = |linkage, vis, unnamed| vec![1, 1, 4, 0, 1, linkage, 0, 0, 0, vis, 0, unnamed];
    assert_eq!(
        decoded(&global(0, 1, 2), &function(0, 2, 1)),
        (
            (Visibility::Hidden, UnnamedAddr::Local),
            (Visibility::Protected, UnnamedAddr::Global)
        )
    );
    // Internal (3) and private (9) linkage keep the default visibility.
    assert_eq!(
        decoded(&global(3, 1, 1), &function(9, 1, 2)),
        (
            (Visibility::Default, UnnamedAddr::Global),
            (Visibility::Default, UnnamedAddr::Local)
        )
    );
    // Unknown values read as the defaults.
    assert_eq!(
        decoded(&global(0, 7, 9), &function(0, 7, 9)),
        (
            (Visibility::Default, UnnamedAddr::Significant),
            (Visibility::Default, UnnamedAddr::Significant)
        )
    );
    // Records that end before the fields.
    assert_eq!(
        decoded(&[0, 1, 0, 2, 0, 0, 0, 0], &[1, 1, 4, 0, 1, 0, 0, 0, 0]),
        (
            (Visibility::Default, UnnamedAddr::Significant),
            (Visibility::Default, UnnamedAddr::Significant)
        )
    );
}

/// One `PARAMATTR` list naming many groups at distinct positions decodes in work linear in its
/// operand count (a per-group scan of the positions seen so far would be quadratic and unmetered).
/// Groups for one position still merge in record order.
#[test]
fn attribute_lists_decode_in_linear_work() {
    const N: u64 = 50_000;
    // Group i (id i) applies to parameter i with an uninterpreted attribute only (`nofree`, kind
    // 62, kept beside the IR on request; LLVM refuses a group without any); group N+1 and N+2 both
    // apply to parameter 1 with an alignment, the later one winning.
    let mut groups: Vec<(u64, Vec<u64>)> = (1..=N).map(|i| (3, vec![i, i, 0, 62])).collect();
    groups.push((3, vec![N + 1, 1, 1, 1, 8]));
    groups.push((3, vec![N + 2, 1, 1, 1, 16]));
    // Named in reverse, so the list is not already in position order.
    let list: Vec<u64> = (1..=N + 2).rev().collect();
    let ops = list.len() as u64;
    let bytes = stream(2, 0, b"", |w| {
        block(w, codes::TYPE_BLOCK, &good_types());
        block(w, codes::PARAMATTR_GROUP_BLOCK, &groups);
        block(w, codes::PARAMATTR_BLOCK, &[(2, list)]);
    });
    let cap = 16 * (ops + groups.len() as u64) + 4096;
    let mut m = Meter::with_caps(&Budget::default(), cap, 256 << 20);
    let t = decode_module(&bytes, KEEP_ALL, &mut m).unwrap();
    assert!(m.work() <= cap);
    let n = usize::try_from(N).unwrap();
    let r = t.attr_lists[0]
        .resolve(&t.types, n, |_| None, true, &mut m)
        .unwrap()
        .unwrap();
    assert_eq!(r.params.len(), n);
    // Reversed naming puts N+2 before N+1, so N+1's align 8 is the later group for parameter 1.
    assert_eq!(r.params[0].align.map(|a| a.bytes()), Some(8));
    let mut nofree = EnumAttrs::default();
    nofree.insert(EnumAttrs::NO_FREE);
    assert_eq!(r.params[1], ParamAttrs::default());
    assert_eq!(r.enums.param(1), nofree);
    assert_eq!(r.enums.params.len(), n);
}

/// The kinds LLVM 23 adds are known: `denormal_fpenv` (106, its value checked) and
/// `dead_on_return` (103) in its new integer form are advisory and dropped; the enum kinds
/// `nooutline`, `flatten`, `noipa` (107 to 109) and `dead_on_return`'s older enum spelling are
/// kept uninterpreted.
#[test]
fn llvm_23_attribute_kinds_decode() {
    // `denormal_fpenv(preservesign, float: dynamic|ieee)`, nooutline, flatten, noipa.
    let t = with_attrs(&[1, 0xFFFF_FFFF, 1, 106, 0x35, 0, 107, 0, 108, 0, 109]).unwrap();
    let f = t.parts.funcs.values().next().unwrap();
    assert_eq!(
        first_enums(&t).func.kinds().collect::<Vec<_>>(),
        [107, 108, 109]
    );
    assert_eq!(f.attrs, crate::ir::FnAttrs::default());
    // `dead_on_return(16)` on parameter 1, and its older enum spelling.
    let t = with_attrs(&[1, 1, 1, 103, 16]).unwrap();
    let f = t.parts.funcs.values().next().unwrap();
    assert_eq!(f.params[0].attrs, Default::default());
    assert!(t.enum_attrs.is_empty());
    let t = with_attrs(&[1, 1, 0, 103]).unwrap();
    let f = t.parts.funcs.values().next().unwrap();
    assert_eq!(f.params[0].attrs, Default::default());
    assert_eq!(first_enums(&t).param(0).kinds().collect::<Vec<_>>(), [103]);
    // A `denormal_fpenv` value wider than its two modes.
    let e = with_attrs(&[1, 0xFFFF_FFFF, 1, 106, 0x100]).unwrap_err();
    assert_eq!(bitcode_kind(&e), Some(&BitcodeError::UnknownAttribute));
}

/// A list naming a group id that no group entry defines reads that id as no attributes, as
/// LLVM's reader does (its group map's `operator[]`); the list's other groups still apply. A
/// CoreImage library module's `llvm.lifetime` declarations name such an id.
#[test]
fn an_undefined_attribute_group_names_no_attributes() {
    let bytes = |list: Vec<u64>| {
        stream(2, 0, b"f", |w| {
            block(w, codes::TYPE_BLOCK, &good_types());
            // Group 1: the function is convergent; group 3: parameter 1 is noalias.
            block(
                w,
                codes::PARAMATTR_GROUP_BLOCK,
                &[(3, vec![1, 0xFFFF_FFFF, 0, 43]), (3, vec![3, 1, 0, 9])],
            );
            block(w, codes::PARAMATTR_BLOCK, &[(2, list)]);
            w.unabbrev(8, &[0, 1, 4, 0, 1, 0, 1, 0, 0]);
        })
    };
    let t = decode(&bytes(vec![1, 2, 3])).unwrap();
    let f = t.parts.funcs.values().next().unwrap();
    assert!(f.attrs.convergent);
    assert!(f.params[0].attrs.noalias);
    // Only undefined ids: an empty list.
    let t = decode(&bytes(vec![2, 7])).unwrap();
    let f = t.parts.funcs.values().next().unwrap();
    assert_eq!(f.attrs, Default::default());
    assert_eq!(f.params[0].attrs, Default::default());
    verify_module_level(&t).unwrap();
}

/// An undefined group id names no attributes only where LLVM reads it so: LLVM keys groups by a
/// 32-bit id and resolves a list as it reads it, so a wider id, a group block after a list block,
/// and a second block of either kind are malformed rather than silently dropping the attributes
/// LLVM would apply (a lost `byval` would change what a call copies).
#[test]
fn attribute_blocks_keep_llvms_order_and_32_bit_ids() {
    const GROUPS: u32 = codes::PARAMATTR_GROUP_BLOCK;
    const LISTS: u32 = codes::PARAMATTR_BLOCK;
    let module = |blocks: &[(u32, Vec<(u64, Vec<u64>)>)]| {
        decode(&stream(2, 0, b"f", |w| {
            block(w, codes::TYPE_BLOCK, &good_types());
            for (id, records) in blocks {
                block(w, *id, records);
            }
            w.unabbrev(8, &[0, 1, 4, 0, 1, 0, 1, 0, 0]);
        }))
    };
    // Group 1: the function is convergent; group 3: parameter 1 is noalias.
    let groups = || {
        (
            GROUPS,
            vec![(3, vec![1, 0xFFFF_FFFF, 0, 43]), (3, vec![3, 1, 0, 9])],
        )
    };
    let lists = |ids: Vec<u64>| (LISTS, vec![(2, ids)]);
    let malformed = |blocks: &[(u32, Vec<(u64, Vec<u64>)>)], block: u32, code: u32| {
        let e = module(blocks).unwrap_err();
        assert_eq!(
            bitcode_kind(&e),
            Some(&BitcodeError::MalformedRecord { block, code }),
            "{e:?}"
        );
    };
    // In LLVM's order both groups apply.
    let t = module(&[groups(), lists(vec![1, 3])]).unwrap();
    let f = t.parts.funcs.values().next().unwrap();
    assert!(f.attrs.convergent && f.params[0].attrs.noalias);
    // The list before the groups: LLVM refuses the group block once the list has looked an id up.
    malformed(&[lists(vec![1, 3]), groups()], GROUPS, 0);
    // A listed id above `u32::MAX`, which LLVM truncates onto group 3.
    malformed(&[groups(), lists(vec![1, (1 << 32) + 3])], LISTS, 2);
    // A group id or position above `u32::MAX`.
    let wide_id = (GROUPS, vec![(3, vec![(1 << 32) + 1, 0xFFFF_FFFF, 0, 43])]);
    malformed(&[wide_id, lists(vec![1])], GROUPS, 3);
    let wide_position = (GROUPS, vec![(3, vec![1, (1 << 32) + 1, 0, 9])]);
    malformed(&[wide_position, lists(vec![1])], GROUPS, 3);
    // A second block of either kind.
    malformed(&[groups(), groups(), lists(vec![1])], GROUPS, 0);
    malformed(&[groups(), lists(vec![1]), lists(vec![3])], LISTS, 0);
}

/// A group entry needs its id, its position, and at least one attribute operand: LLVM's reader
/// refuses a shorter entry ("Invalid grp record", through 23.1), so it is malformed rather than an
/// empty group.
#[test]
fn a_group_entry_without_attributes_is_malformed() {
    const GROUPS: u32 = codes::PARAMATTR_GROUP_BLOCK;
    let module = |entry: Vec<u64>| {
        decode(&stream(2, 0, b"f", |w| {
            block(w, codes::TYPE_BLOCK, &good_types());
            block(w, GROUPS, &[(3, entry)]);
            block(w, codes::PARAMATTR_BLOCK, &[(2, vec![1])]);
            w.unabbrev(8, &[0, 1, 4, 0, 1, 0, 1, 0, 0]);
        }))
    };
    for entry in [vec![], vec![1], vec![1, 0xFFFF_FFFF]] {
        let e = module(entry.clone()).unwrap_err();
        assert_eq!(
            bitcode_kind(&e),
            Some(&BitcodeError::MalformedRecord {
                block: GROUPS,
                code: 3
            }),
            "{entry:?}: {e:?}"
        );
    }
    // One attribute operand is enough (`convergent`).
    let t = module(vec![1, 0xFFFF_FFFF, 0, 43]).unwrap();
    assert!(t.parts.funcs.values().next().unwrap().attrs.convergent);
}

#[test]
fn attribute_errors_are_typed() {
    // Kind 110 is not in LLVM's table (through 23.1).
    let e = with_attrs(&[1, 0xFFFF_FFFF, 0, 110]).unwrap_err();
    assert_eq!(bitcode_kind(&e), Some(&BitcodeError::UnknownAttribute));
    // An unknown attribute form.
    let e = with_attrs(&[1, 0xFFFF_FFFF, 9, 1]).unwrap_err();
    assert_eq!(bitcode_kind(&e), Some(&BitcodeError::UnknownAttribute));
    // An unknown memory encoding version.
    let e = with_attrs(&[1, 0xFFFF_FFFF, 1, 86, 3 << 56]).unwrap_err();
    assert_eq!(bitcode_kind(&e), Some(&BitcodeError::UnknownAttribute));
    // inalloca changes the calling ABI.
    let e = with_attrs(&[1, 1, 6, 38, 0]).unwrap_err();
    assert_eq!(
        unsupported_feature(&e),
        Some(&Feature::Attribute { kind: 38 })
    );
    // A non-power-of-two alignment and an unterminated string.
    for group in [
        vec![1, 1, 1, 1, 12],
        vec![1, 0xFFFF_FFFF, 3, u64::from(b'k')],
    ] {
        let e = with_attrs(&group).unwrap_err();
        assert!(
            matches!(bitcode_kind(&e), Some(BitcodeError::MalformedRecord { .. })),
            "{e:?}"
        );
    }
    // Attributes on a parameter the signature does not have.
    let e = with_attrs(&[1, 5, 0, 9]).unwrap_err();
    assert!(matches!(
        bitcode_kind(&e),
        Some(BitcodeError::MalformedRecord { .. })
    ));
}

fn constants(records: &[(u64, Vec<u64>)]) -> Result<ModuleTables, Error> {
    decode(&stream(2, 0, b"", |w| {
        block(w, codes::TYPE_BLOCK, &good_types());
        block(w, codes::CONSTANTS_BLOCK, records);
    }))
}

#[test]
fn constants_decode_and_expand() {
    let t = constants(&[
        (1, vec![0]),
        // i32 -1 (sign-rotated VBR), undef, poison, null (canonicalized to 0).
        (4, vec![3]),
        (3, vec![]),
        (26, vec![]),
        (2, vec![]),
        (1, vec![5]),
        // [2 x i32] from DATA.
        (22, vec![5, 6]),
        // add nsw (value 0 + value 0) is an i32 expression.
        (1, vec![0]),
        (10, vec![0, 0, 0, 2]),
    ])
    .unwrap();
    let c = |i: u64| {
        t.parts
            .consts
            .get(t.value(i).unwrap())
            .unwrap()
            .value
            .clone()
    };
    assert_eq!(c(0), Const::Int(u64::from(u32::MAX)));
    assert_eq!(c(1), Const::Undef);
    assert_eq!(c(2), Const::Poison);
    assert_eq!(c(3), Const::Int(0));
    let Const::Aggregate(elems) = c(4) else {
        panic!("DATA is not an aggregate");
    };
    assert_eq!(elems.len(), 2);
    assert!(matches!(
        c(5),
        Const::Expr(crate::ir::ConstExpr::BinOp { .. })
    ));
    verify_module_level(&t).unwrap();
}

/// A vector `FLOAT` or `INTEGER` record is a splat (LLVM 23 writes floating-point vector splats
/// so): it decodes to the same constant as the `DATA` spelling older writers use.
#[test]
fn vector_splats_decode_like_their_data_spelling() {
    // Types: 0 `i32`, 1 `float`, 2 `<4 x float>`, 3 `<2 x i32>`.
    let splats = |records: &[(u64, Vec<u64>)]| {
        decode(&stream(2, 0, b"", |w| {
            block(
                w,
                codes::TYPE_BLOCK,
                &[
                    (1, vec![4]),
                    (7, vec![32]),
                    (3, vec![]),
                    (12, vec![4, 1]),
                    (12, vec![2, 0]),
                ],
            );
            block(w, codes::CONSTANTS_BLOCK, records);
        }))
    };
    let t = splats(&[
        (1, vec![2]),
        (6, vec![0x3D80_0000]),
        (22, vec![0x3D80_0000; 4]),
        (1, vec![3]),
        // i32 -7 (sign-rotated VBR).
        (4, vec![15]),
        (22, vec![0xFFFF_FFF9; 2]),
    ])
    .unwrap();
    assert_eq!(t.value(0), t.value(1));
    assert_eq!(t.value(2), t.value(3));
    let Const::Aggregate(lanes) = &t.parts.consts.get(t.value(0).unwrap()).unwrap().value else {
        panic!("a splat is an aggregate");
    };
    assert_eq!(lanes.len(), 4);
    verify_module_level(&t).unwrap();
    // A splat whose value does not fit the element type, or of the other scalar kind.
    for records in [
        vec![(1, vec![2]), (6, vec![1 << 32])],
        vec![(1, vec![2]), (4, vec![2])],
        vec![(1, vec![3]), (6, vec![0])],
    ] {
        let e = splats(&records).unwrap_err();
        assert!(
            matches!(bitcode_kind(&e), Some(BitcodeError::MalformedRecord { .. })),
            "{records:?}: {e:?}"
        );
    }
}

#[test]
fn constant_errors_are_typed() {
    // A cycle between two aggregates.
    let e = constants(&[(1, vec![5]), (7, vec![1, 1]), (7, vec![0, 0])]).unwrap_err();
    assert_eq!(bitcode_kind(&e), Some(&BitcodeError::BadValueRef));
    // A reference past the block.
    let e = constants(&[(1, vec![5]), (7, vec![0, 9])]).unwrap_err();
    assert_eq!(bitcode_kind(&e), Some(&BitcodeError::BadValueRef));
    // A record before any SETTYPE, and an INTEGER of an aggregate type.
    let e = constants(&[(2, vec![])]).unwrap_err();
    assert!(matches!(
        bitcode_kind(&e),
        Some(BitcodeError::MalformedRecord { .. })
    ));
    let e = constants(&[(1, vec![5]), (4, vec![2])]).unwrap_err();
    assert!(matches!(
        bitcode_kind(&e),
        Some(BitcodeError::MalformedRecord { .. })
    ));
    // Block addresses, `select` expressions, and LLVM 23's byte constants are valid LLVM outside
    // the typed model.
    for code in [21u32, 13, 35, 36] {
        let e = constants(&[(1, vec![0]), (u64::from(code), vec![0, 0, 0])]).unwrap_err();
        assert_eq!(unsupported_feature(&e), Some(&Feature::Constant { code }));
    }
    let e = constants(&[(1, vec![0]), (60, vec![])]).unwrap_err();
    assert_eq!(
        bitcode_kind(&e),
        Some(&BitcodeError::UnknownRecord {
            block: codes::CONSTANTS_BLOCK,
            code: 60
        })
    );
}

/// `records` in a module-level constants block over [`good_types`] and three more: 6 `void ()`,
/// 7 `void ()*` (a typed pointer in address space 0), 8 `ptr` (opaque, address space 0).
fn asm_constants(records: &[(u64, Vec<u64>)]) -> Result<ModuleTables, Error> {
    let mut types = good_types();
    types[0] = (1, vec![9]);
    types.extend([(21, vec![0, 3]), (8, vec![6, 0]), (25, vec![0])]);
    decode(&stream(2, 0, b"", |w| {
        block(w, codes::TYPE_BLOCK, &types);
        block(w, codes::CONSTANTS_BLOCK, records);
    }))
}

/// `[len, bytes...]`: an inline assembly string's operands.
fn asm_text(s: &str) -> Vec<u64> {
    [vec![s.len() as u64], chars(s)].concat()
}

/// Every spelling of LLVM's inline assembly record decodes to its text and flags: `INLINEASM`
/// (code 30, naming its function type), and the typed-pointer `INLINEASM_OLD3` (28),
/// `INLINEASM_OLD2` (23) and `INLINEASM_OLD` (18), typed by their pointer's pointee. The constant
/// is a `ptr` in address space 0, and the module verifies.
#[test]
fn inline_asm_records_decode_in_every_spelling() {
    let empty = [asm_text(""), asm_text("")].concat();
    let text = [asm_text("nop \"x\""), asm_text("~{memory}")].concat();
    let t = asm_constants(&[
        (1, vec![8]),
        // asm sideeffect "", "" of type void (): what an MSL `asm volatile("")` compiles to.
        (30, [vec![6, 1], empty.clone()].concat()),
        // Every flag.
        (30, [vec![6, 0b1111], text.clone()].concat()),
        (1, vec![7]),
        (28, [vec![0b1010], text.clone()].concat()),
        (23, [vec![0b100], empty.clone()].concat()),
        (18, [vec![0b11], text.clone()].concat()),
    ])
    .unwrap();
    let asm = |i: u64| match &t.parts.consts.get(t.value(i).unwrap()).unwrap().value {
        Const::InlineAsm(a) => (**a).clone(),
        other => panic!("value {i}: {other:?}"),
    };
    let flags =
        |a: &crate::ir::InlineAsm| [a.side_effects, a.align_stack, a.intel_dialect, a.can_throw];
    assert_eq!(asm(0).spelling(), r#"asm sideeffect "", """#);
    assert_eq!(flags(&asm(0)), [true, false, false, false]);
    assert_eq!(&*asm(1).template, b"nop \"x\"");
    assert_eq!(&*asm(1).constraints, b"~{memory}");
    assert_eq!(
        asm(1).spelling(),
        r#"asm sideeffect alignstack inteldialect unwind "nop \22x\22", "~{memory}""#
    );
    assert_eq!(flags(&asm(2)), [false, true, false, true]);
    assert_eq!(flags(&asm(3)), [false, false, true, false]);
    assert_eq!(&*asm(3).template, b"");
    assert_eq!(flags(&asm(4)), [true, true, false, false]);
    assert_eq!(&*asm(4).constraints, b"~{memory}");
    let ptr0 = |i: u64| t.parts.consts.ty(t.value(i).unwrap()).unwrap();
    assert_eq!(ptr0(0), ptr0(2), "both pointer spellings erase to one ptr");
    verify_module_level(&t).unwrap();
}

/// An inline assembly record that does not fit its spelling is malformed: a flag bit the spelling
/// does not define, a function type that is none, a string running past the record, a byte above
/// 255, an operand after the constraints, an old spelling over an opaque pointer (no pointee to
/// type it), and a type other than a pointer. Inline assembly as another constant's operand fails
/// verification: LLVM has no such constant.
#[test]
fn malformed_inline_asm_records_are_typed() {
    let empty = [asm_text(""), asm_text("")].concat();
    for records in [
        vec![
            (1, vec![8]),
            (30, [vec![6, 1 << 4], empty.clone()].concat()),
        ],
        vec![(1, vec![7]), (28, [vec![1 << 4], empty.clone()].concat())],
        vec![(1, vec![7]), (23, [vec![1 << 3], empty.clone()].concat())],
        vec![(1, vec![7]), (18, [vec![1 << 2], empty.clone()].concat())],
        vec![(1, vec![8]), (30, [vec![0, 1], empty.clone()].concat())],
        vec![(1, vec![8]), (30, vec![6, 1, 3, 1, 2])],
        vec![(1, vec![8]), (30, vec![6, 1, 1, 256, 0])],
        vec![
            (1, vec![8]),
            (30, [vec![6, 1], empty.clone(), vec![0]].concat()),
        ],
        vec![(1, vec![8]), (30, vec![6, 1, 0])],
        vec![(1, vec![8]), (28, [vec![1], empty.clone()].concat())],
        vec![(1, vec![0]), (30, [vec![6, 1], empty.clone()].concat())],
    ] {
        let e = asm_constants(&records).unwrap_err();
        assert!(
            matches!(bitcode_kind(&e), Some(BitcodeError::MalformedRecord { .. })),
            "{records:?}: {e:?}"
        );
    }
    // A struct `{ ptr }` (type 9) holding an inline assembly expression.
    let mut types = good_types();
    types[0] = (1, vec![10]);
    types.extend([
        (21, vec![0, 3]),
        (8, vec![6, 0]),
        (25, vec![0]),
        (18, vec![0, 8]),
    ]);
    let t = decode(&stream(2, 0, b"", |w| {
        block(w, codes::TYPE_BLOCK, &types);
        block(
            w,
            codes::CONSTANTS_BLOCK,
            &[
                (1, vec![8]),
                (30, [vec![6, 1], empty.clone()].concat()),
                (1, vec![9]),
                (7, vec![0]),
            ],
        );
    }))
    .unwrap();
    let e = verify_module_level(&t).unwrap_err();
    assert!(
        matches!(&e, Error::InvalidInput(i)
            if i.kind == InputErrorKind::Verify(crate::error::VerifyError::MalformedConstant)),
        "{e:?}"
    );
}

#[test]
fn metadata_and_string_table_errors_are_typed() {
    // A node referring past the metadata.
    let e = with_body(|w| block(w, codes::METADATA_BLOCK, &[(3, vec![40])])).unwrap_err();
    assert_eq!(bitcode_kind(&e), Some(&BitcodeError::BadMetadata));
    // A value of the wrong type (value 3 is an i32, type 5 is [2 x i32]).
    let e = with_body(|w| block(w, codes::METADATA_BLOCK, &[(2, vec![5, 3])])).unwrap_err();
    assert_eq!(bitcode_kind(&e), Some(&BitcodeError::BadValueRef));
    // A name without its named node.
    let e = with_body(|w| block(w, codes::METADATA_BLOCK, &[(4, chars("n"))])).unwrap_err();
    assert_eq!(bitcode_kind(&e), Some(&BitcodeError::BadMetadata));
    // A string count the length stream cannot hold.
    let e = with_body(|w| {
        w.enter(u64::from(codes::METADATA_BLOCK), 4);
        blob_record(w, 35, &[1000, 4], &[0, 0, 0, 0]);
        w.end();
    })
    .unwrap_err();
    assert!(matches!(
        bitcode_kind(&e),
        Some(BitcodeError::MalformedRecord { .. })
    ));
    // A name outside the string table.
    let e = decode(&stream(2, 0, b"g", good_body)).unwrap_err();
    assert_eq!(bitcode_kind(&e), Some(&BitcodeError::BadStringTable));
}

#[test]
fn global_attachments_are_checked_and_dropped() {
    const ATTACH: u64 = codes::METADATA_GLOBAL_DECL_ATTACHMENT as u64;
    let attach = |ops: Vec<u64>| with_body(|w| block(w, codes::METADATA_BLOCK, &[(ATTACH, ops)]));
    // `@g` (value 0) and `@f` (value 1) may carry nodes; nothing is added to the IR.
    let t = attach(vec![0, 0, 3, 1, 3]).unwrap();
    assert_eq!(t.parts.metadata.len(), 4);
    assert!(attach(vec![1, 7, 3]).is_ok());
    // A module constant (value 3) or a missing value is not a global object.
    for v in [3, 40] {
        let e = attach(vec![v, 0, 3]).unwrap_err();
        assert_eq!(bitcode_kind(&e), Some(&BitcodeError::BadValueRef));
    }
    // An attachment must name a node that exists: not a string, not past the metadata.
    for md in [0, 4, 40] {
        let e = attach(vec![0, 0, md]).unwrap_err();
        assert_eq!(bitcode_kind(&e), Some(&BitcodeError::BadMetadata));
    }
    // No pairs, or an unpaired kind.
    for ops in [vec![], vec![0], vec![0, 0]] {
        let e = attach(ops).unwrap_err();
        assert_eq!(
            bitcode_kind(&e),
            Some(&BitcodeError::MalformedRecord {
                block: codes::METADATA_BLOCK,
                code: codes::METADATA_GLOBAL_DECL_ATTACHMENT
            })
        );
    }
}

/// LLVM 23's `GUIDLIST` is checked and dropped: 32-bit halves in pairs, one pair per module value
/// id (later records continue the numbering), and a non-zero GUID only on a global or function.
#[test]
fn guid_lists_are_checked_and_dropped() {
    const LIST: u64 = codes::MODULE_GUIDLIST as u64;
    let with_lists = |lists: &[Vec<u64>]| {
        with_body(|w| {
            for ops in lists {
                w.unabbrev(LIST, ops);
            }
        })
    };
    let malformed = |r: Result<ModuleTables, Error>| {
        let e = r.unwrap_err();
        assert_eq!(
            bitcode_kind(&e),
            Some(&BitcodeError::MalformedRecord {
                block: codes::MODULE_BLOCK,
                code: codes::MODULE_GUIDLIST
            }),
            "{e}"
        );
    };
    // GUIDs on `@g` (value 0) and `@f` (value 1), none on the constants; an empty list; a list
    // shorter than the value table; the same assignment split over two records.
    let plain = with_body(|_| {}).unwrap();
    for lists in [
        vec![vec![0, 7, 0x8000_0000, 1, 0, 0, 0, 0, 0, 0]],
        vec![vec![]],
        vec![vec![0, 7]],
        vec![vec![0, 7], vec![0xFFFF_FFFF, 0xFFFF_FFFF]],
    ] {
        assert_eq!(with_lists(&lists).unwrap(), plain, "{lists:?}");
    }
    // An odd operand count, and a half wider than 32 bits.
    malformed(with_lists(&[vec![0, 7, 0]]));
    malformed(with_lists(&[vec![0, 1 << 32]]));
    // A GUID on a module constant (value 3), directly or through a continuing record, and one
    // past the value table.
    malformed(with_lists(&[vec![0, 0, 0, 0, 0, 0, 0, 1]]));
    malformed(with_lists(&[vec![0, 0, 0, 0], vec![0, 0, 0, 1]]));
    malformed(with_lists(&[vec![0; 18]
        .into_iter()
        .chain([0, 1])
        .collect()]));
}

/// LLVM 23's `prefalign` (function record operand 21) is checked like `align` and dropped.
#[test]
fn function_preferred_alignment_is_checked_and_dropped() {
    let with_prefalign = |raw: u64| {
        let mut ops = vec![0, 1, 4, 0, 1, 0, 0, 0, 0];
        ops.resize(21, 0);
        ops.push(raw);
        decode(&stream(2, 0, b"f", |w| {
            block(w, codes::TYPE_BLOCK, &good_types());
            w.unabbrev(8, &ops);
        }))
    };
    let t = with_prefalign(5).unwrap();
    assert_eq!(t, with_prefalign(0).unwrap());
    assert_eq!(t.parts.funcs.values().next().unwrap().align, None);
    // Alignments are powers of two up to 2^32 (`log2 + 1` encoded).
    let e = with_prefalign(200).unwrap_err();
    assert_eq!(
        bitcode_kind(&e),
        Some(&BitcodeError::MalformedRecord {
            block: codes::MODULE_BLOCK,
            code: codes::MODULE_FUNCTION
        })
    );
}

#[test]
fn the_ledger_refuses_before_allocating() {
    let bytes = good();
    let mut m = Meter::with_caps(&Budget::default(), u64::MAX, 256);
    let e = decode_module(&bytes, DecodeOptions::default(), &mut m).unwrap_err();
    assert_eq!(e, Error::LimitExceeded(Limit::LedgerBytes));
    let mut m = Meter::with_caps(&Budget::default(), 16, u64::MAX);
    let e = decode_module(&bytes, DecodeOptions::default(), &mut m).unwrap_err();
    assert_eq!(e, Error::LimitExceeded(Limit::WorkUnits));
}

/// Truncate at every word and flip bits at fixed seeds: always a result, never a panic, and
/// bounded work. Every option is on, so the optional steps see the mutations too.
pub(super) fn mutate(bytes: &[u8]) {
    let words = bytes.len() / 4;
    for cut in 0..words {
        let mut m = Meter::with_caps(&Budget::default(), 1 << 24, 64 << 20);
        let _ = decode_module(&bytes[..cut * 4], KEEP_ALL, &mut m);
        assert!(m.work() <= 1 << 24);
    }
    let mut rng = XorShift(0x9E37_79B9_7F4A_7C15);
    for _ in 0..2000 {
        let mut b = bytes.to_vec();
        for _ in 0..1 + rng.next() % 3 {
            let bit = (rng.next() % (b.len() as u64 * 8)) as usize;
            b[bit / 8] ^= 1 << (bit % 8);
        }
        let mut m = Meter::with_caps(&Budget::default(), 1 << 24, 64 << 20);
        let _ = decode_module(&b, KEEP_ALL, &mut m);
    }
}

#[test]
fn mutated_synthetic_modules_never_panic() {
    mutate(&good());
}

// ---- Public fixtures, assembled at test time ----

static SCRATCH: AtomicU32 = AtomicU32::new(0);

/// A fresh scratch directory, removed by the guard.
struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Scratch {
        let n = SCRATCH.fetch_add(1, AtomicOrdering::Relaxed);
        let dir =
            std::env::temp_dir().join(format!("m2v-bitcode-tests-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        Scratch(dir)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// `llvm-as`: the `METAL2VULKAN_LLVM_AS` override, else `PATH`, else the Homebrew LLVM prefixes.
fn llvm_as() -> Option<PathBuf> {
    if let Some(p) = metal2vulkan_devtools::tools::tool_path_override("llvm-as") {
        return Some(PathBuf::from(p));
    }
    let path = std::env::var_os("PATH").unwrap_or_default();
    std::env::split_paths(&path)
        .chain(
            ["/opt/homebrew/opt/llvm/bin", "/usr/local/opt/llvm/bin"]
                .iter()
                .map(PathBuf::from),
        )
        .map(|d| d.join("llvm-as"))
        .find(|p| p.is_file())
}

fn fixtures_dir() -> PathBuf {
    metal2vulkan_devtools::fixtures_dir()
}

/// Every committed opaque-pointer `.ll` fixture (top level and `engine/`), sorted.
fn opaque_fixtures() -> Vec<PathBuf> {
    let mut out = Vec::new();
    for dir in [fixtures_dir(), fixtures_dir().join("engine")] {
        for e in std::fs::read_dir(dir).unwrap() {
            let p = e.unwrap().path();
            let name = p.file_name().unwrap().to_string_lossy().into_owned();
            if name.ends_with(".ll") && !name.ends_with(".air.ll") {
                out.push(p);
            }
        }
    }
    out.sort();
    out
}

fn assemble(tool: &Path, args: &[&std::ffi::OsStr], out: &Path) -> Option<Vec<u8>> {
    let status = Command::new(tool)
        .args(args)
        .arg("-o")
        .arg(out)
        .output()
        .ok()?;
    if !status.status.success() {
        return None;
    }
    std::fs::read(out).ok()
}

fn stream_of(bytes: &[u8]) -> &[u8] {
    container::bitcode_stream(bytes).unwrap().1
}

/// Counts read from the text the bitcode was assembled from. `llvm-as` auto-upgrades `llvm.*`
/// declarations as it reads the text (renaming or rewriting them, and LLVM 23 drops an unused
/// `declare void @llvm.lifetime.start.i64(i64)`), so, as in the bitcode-reader audit, only the
/// other declarations are counted.
struct TextCounts {
    defines: usize,
    declares: usize,
    globals: usize,
    named_md: Vec<String>,
}

fn text_counts(text: &str) -> TextCounts {
    let lines = || text.lines().map(str::trim_start);
    TextCounts {
        defines: lines().filter(|l| l.starts_with("define ")).count(),
        declares: lines()
            .filter(|l| {
                l.starts_with("declare ")
                    && !l
                        .find('@')
                        .is_some_and(|at| l[at + 1..].starts_with("llvm."))
            })
            .count(),
        globals: lines()
            .filter(|l| l.starts_with('@') && l.contains(" = "))
            .count(),
        named_md: lines()
            .filter(|l| {
                l.starts_with('!')
                    && !l[1..].starts_with(|c: char| c.is_ascii_digit())
                    && l.contains(" = !{")
            })
            .map(|l| l[1..l.find(" = ").unwrap()].to_string())
            .collect(),
    }
}

fn check_fixture(t: &ModuleTables, text: &str, what: &Path) {
    let c = text_counts(text);
    let p = &t.parts;
    assert_eq!(t.defined.len(), c.defines, "{what:?}: definitions");
    assert_eq!(
        t.function_blocks as usize, c.defines,
        "{what:?}: function blocks"
    );
    let counted = p
        .funcs
        .values()
        .filter(|f| f.body.is_some() || !p.symbols.get(f.name).unwrap().starts_with(b"llvm."))
        .count();
    assert_eq!(counted, c.defines + c.declares, "{what:?}: functions");
    assert_eq!(p.globals.len(), c.globals, "{what:?}: globals");
    let mut named: Vec<String> = p
        .named_md
        .iter()
        .map(|n| String::from_utf8_lossy(p.symbols.get(n.name).unwrap()).into_owned())
        .collect();
    let mut expected = c.named_md;
    named.sort();
    expected.sort();
    assert_eq!(named, expected, "{what:?}: named metadata");
    // Value numbering: every global, then every function (a marker declaration decode removed
    // without a constant), then the module constants.
    let removed = t.values.iter().filter(|v| v.id.is_none()).count();
    let (g, f) = (p.globals.len(), p.funcs.len() + removed);
    for (i, v) in t.values.iter().enumerate() {
        let c = v.id.map(|c| &p.consts.get(c).unwrap().value);
        let ok = match c {
            Some(Const::GlobalAddr(_)) => i < g,
            None | Some(Const::FuncAddr(_)) => (g..g + f).contains(&i),
            Some(_) => i >= g + f,
        };
        assert!(ok, "{what:?}: value {i} is {c:?}");
    }
    // Every definition has its body, and every declaration none.
    for (id, f) in p.funcs.iter() {
        assert_eq!(
            f.body.is_some(),
            t.defined.contains(&id),
            "{what:?}: body of {:?}",
            f.name
        );
    }
    if let Err(e) = verify_module_level(t) {
        panic!("{what:?}: the decoded module does not verify: {e}");
    }
}

#[test]
fn public_fixtures_decode_and_verify() {
    let Some(tool) = llvm_as() else {
        eprintln!(
            "SKIP public_fixtures_decode_and_verify: llvm-as not found (set METAL2VULKAN_LLVM_AS)"
        );
        return;
    };
    let scratch = Scratch::new();
    let out = scratch.0.join("fixture.bc");
    let (mut decoded, mut skipped) = (0, Vec::new());
    for ll in opaque_fixtures() {
        let Some(bytes) = assemble(&tool, &[ll.as_os_str()], &out) else {
            // The fixture is not valid for this llvm-as; its text is not our input.
            skipped.push(ll);
            continue;
        };
        let text = std::fs::read_to_string(&ll).unwrap();
        let t = decode(stream_of(&bytes)).unwrap_or_else(|e| panic!("{ll:?}: {e}"));
        check_fixture(&t, &text, &ll);
        assert_eq!(t, decode(stream_of(&bytes)).unwrap(), "{ll:?}: determinism");
        decoded += 1;
    }
    if !skipped.is_empty() {
        eprintln!("SKIP (llvm-as rejected the text): {skipped:?}");
    }
    assert!(decoded > 100, "only {decoded} fixtures assembled");
}

#[test]
fn a_public_fixture_survives_mutation() {
    let Some(tool) = llvm_as() else {
        eprintln!("SKIP a_public_fixture_survives_mutation: llvm-as not found");
        return;
    };
    let scratch = Scratch::new();
    let out = scratch.0.join("store_const.bc");
    let ll = fixtures_dir().join("engine/store_const.ll");
    let bytes = assemble(&tool, &[ll.as_os_str()], &out).expect("llvm-as store_const.ll");
    mutate(stream_of(&bytes));
}

/// Whether `xcrun metal-as` exists. Tests that need it skip visibly when it does not, and fail
/// when it exists but rejects a committed fixture.
fn have_metal_as() -> bool {
    cfg!(target_os = "macos")
        && Command::new("xcrun")
            .args(["-f", "metal-as"])
            .output()
            .is_ok_and(|o| o.status.success())
}

/// An argument-buffer member of a module Apple's compiler wrote before AIR 2.5 states no
/// `air.address_space`, and the holding struct's field at the member's byte offset is what gives
/// it. `engine/air_2_4_front_end` is our own MSL compiled at `-mmacosx-version-min=12.0`
/// (`air64_v24`), whose argument buffer is
/// `<{ texture2d, sampler, ptr addrspace(1), ptr addrspace(2), i32, [4 x i8] }>` with a
/// `device float4 *writable` at byte 16 and a `constant float4 *readable` at byte 24: through
/// `xcrun metal-as` the parameter keeps its pointee and the two members read device and constant,
/// and through `llvm-as` -- whose opaque pointers erase the pointee, so the module states no
/// holding type -- both read as the documented device default. One fixture, two assemblers, so
/// the field is demonstrably what decides it.
#[test]
fn an_argument_buffer_member_takes_its_space_from_the_holding_struct() {
    let Some(tool) = llvm_as() else {
        eprintln!("SKIP an_argument_buffer_member_takes_its_space_from_the_holding_struct: llvm-as not found");
        return;
    };
    if !have_metal_as() {
        eprintln!(
            "SKIP an_argument_buffer_member_takes_its_space_from_the_holding_struct: xcrun metal-as not found"
        );
        return;
    }
    let scratch = Scratch::new();
    let dir = fixtures_dir().join("engine");
    // The buffer members of the kernel's argument buffer, in layout order.
    let spaces = |bytes: &[u8]| -> Vec<crate::ir::AddrSpace> {
        let d = super::decode(stream_of(bytes), &mut meter()).expect("decode");
        let e = d
            .abi
            .entries
            .iter()
            .find(|e| matches!(e.stage, crate::abi::Stage::Kernel { .. }))
            .expect("the kernel");
        let crate::abi::ParamRole::ArgumentBuffer(arg) = &e.params[0].role else {
            panic!("an argument buffer, not {:?}", e.params[0].role)
        };
        d.abi
            .struct_info(arg.layout.expect("its layout"))
            .expect("the layout")
            .members
            .iter()
            .filter_map(|mem| match &mem.kind {
                crate::abi::MemberKind::Indirect(member) => match &member.role {
                    crate::abi::ParamRole::Buffer(b) => Some(b.space),
                    _ => None,
                },
                _ => None,
            })
            .collect()
    };
    let air_ll = dir.join("air_2_4_front_end.air.ll");
    let args = [std::ffi::OsStr::new("metal-as"), air_ll.as_os_str()];
    let typed = assemble(Path::new("xcrun"), &args, &scratch.0.join("t.air"))
        .unwrap_or_else(|| panic!("xcrun metal-as rejected {air_ll:?}"));
    assert_eq!(
        spaces(&typed),
        [crate::ir::AddrSpace::Device, crate::ir::AddrSpace::Constant],
        "typed-pointer bitcode reads each member's own field"
    );
    let ll = dir.join("air_2_4_front_end.ll");
    let opaque =
        assemble(&tool, &[ll.as_os_str()], &scratch.0.join("o.bc")).expect("llvm-as the .ll form");
    assert_eq!(
        spaces(&opaque),
        [crate::ir::AddrSpace::Device, crate::ir::AddrSpace::Device],
        "opaque pointers state no holding type, so the device default stands"
    );
}

/// Typed-pointer bitcode from the committed `.air.ll` companions through `xcrun metal-as`.
#[test]
fn typed_pointer_fixtures_keep_their_pointees() {
    if !have_metal_as() {
        eprintln!("SKIP typed_pointer_fixtures_keep_their_pointees: xcrun metal-as not found");
        return;
    }
    let scratch = Scratch::new();
    let out = scratch.0.join("fixture.air");
    let dir = fixtures_dir().join("engine");
    let mut airs: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.to_string_lossy().ends_with(".air.ll"))
        .collect();
    airs.sort();
    let xcrun = Path::new("xcrun");
    let mut decoded = 0;
    for air_ll in airs {
        let args = [std::ffi::OsStr::new("metal-as"), air_ll.as_os_str()];
        let bytes = assemble(xcrun, &args, &out)
            .unwrap_or_else(|| panic!("xcrun metal-as rejected {air_ll:?}"));
        let text = std::fs::read_to_string(&air_ll).unwrap();
        let t = decode(stream_of(&bytes)).unwrap_or_else(|e| panic!("{air_ll:?}: {e}"));
        check_fixture(&t, &text, &air_ll);
        // Every pointer parameter of every function keeps its pointee.
        let fn_types = function_bc_types(&t);
        assert_eq!(fn_types.len(), t.parts.funcs.len());
        for (_, fn_idx) in fn_types {
            let params = t.types.params(u64::from(fn_idx)).unwrap();
            for &p in params {
                let bc = t.types.get(u64::from(p)).unwrap();
                if matches!(t.parts.types.get(bc.ty.unwrap()), Some(Type::Ptr { .. })) {
                    assert!(
                        bc.pointee.is_some(),
                        "{air_ll:?}: untyped parameter pointer"
                    );
                }
            }
        }
        decoded += 1;
    }
    assert!(decoded > 0);
}

/// The bitcode function type of every function, from the module value numbering.
fn function_bc_types(t: &ModuleTables) -> Vec<(crate::ir::FuncId, u32)> {
    t.values
        .iter()
        .filter_map(|v| match t.parts.consts.get(v.id?)?.value {
            Const::FuncAddr(f) => Some((f, v.object_bc_type.expect("function type recorded"))),
            _ => None,
        })
        .collect()
}

#[test]
fn functions_differing_only_in_pointees_keep_their_own_types() {
    // 0 `float`, 1 `i32`, 2 `float addrspace(1)*`, 3 `i32 addrspace(1)*`, 4 `void`,
    // 5 `void (float addrspace(1)*)`, 6 `void (i32 addrspace(1)*)`, 7 `void (i32 addrspace(1)*)*`,
    // 8 `float addrspace(2)*`.
    let types = [
        (1, vec![9]),
        (3, vec![]),
        (7, vec![32]),
        (8, vec![0, 1]),
        (8, vec![1, 1]),
        (2, vec![]),
        (21, vec![0, 4, 2]),
        (21, vec![0, 4, 3]),
        (8, vec![6, 0]),
        (8, vec![0, 2]),
    ];
    let t = decode(&stream(2, 0, b"abcgh", |w| {
        block(w, codes::TYPE_BLOCK, &types);
        // @g: explicit-type `i32 addrspace(2)` global; @h: the older spelling, typed pointer 8.
        w.unabbrev(7, &[3, 1, 1, 1 | 2 | (2 << 2), 0, 0, 0, 0]);
        w.unabbrev(7, &[4, 1, 8, 1, 0, 0, 0, 0]);
        // @a: type 5, @b: type 6, @c: the oldest spelling, a pointer (7) to type 6.
        w.unabbrev(8, &[0, 1, 5, 0, 1, 0, 0, 0, 0]);
        w.unabbrev(8, &[1, 1, 6, 0, 1, 0, 0, 0, 0]);
        w.unabbrev(8, &[2, 1, 7, 0, 1, 0, 0, 0, 0]);
    }))
    .unwrap();
    let p = &t.parts;

    // The two function types erase to one IR type.
    let funcs: Vec<_> = p.funcs.values().collect();
    assert_eq!(funcs.len(), 3);
    assert_eq!(funcs[0].ty, funcs[1].ty);
    assert_eq!(funcs[1].ty, funcs[2].ty);

    // Each function still resolves to its own bitcode type and parameter pointees.
    let fn_types = function_bc_types(&t);
    let recorded: Vec<u32> = fn_types.iter().map(|&(_, bc)| bc).collect();
    assert_eq!(recorded, vec![5, 6, 6]);
    let pointees: Vec<Vec<Option<u32>>> = fn_types
        .iter()
        .map(|&(_, bc)| {
            t.types
                .params(u64::from(bc))
                .unwrap()
                .iter()
                .map(|&q| t.types.pointee(u64::from(q)))
                .collect()
        })
        .collect();
    assert_eq!(pointees, vec![vec![Some(0)], vec![Some(1)], vec![Some(1)]]);

    // Value numbering: @g, @h, @a, @b, @c.
    let own: Vec<_> = t.values.iter().map(|v| v.bc_type).collect();
    let object: Vec<_> = t.values.iter().map(|v| v.object_bc_type).collect();
    assert_eq!(own, vec![None, Some(8), None, None, Some(7)]);
    assert_eq!(object, vec![Some(1), Some(0), Some(5), Some(6), Some(6)]);
    let globals: Vec<_> = p.globals.values().collect();
    assert_eq!(globals[0].value_ty, t.types.ty(1).unwrap());
    assert_eq!(globals[1].value_ty, t.types.ty(0).unwrap());
    assert_eq!(globals[1].space, crate::ir::AddrSpace::Constant);
    verify_module_level(&t).unwrap();
}

#[test]
fn erased_types_are_interned_once() {
    // Two typed pointers into one space erase to one IR type; the table stays acyclic.
    let t = types_only(&[
        (1, vec![4]),
        (7, vec![8]),
        (7, vec![32]),
        (8, vec![0, 1]),
        (8, vec![1, 1]),
    ])
    .unwrap();
    assert_eq!(t.types.ty(2), t.types.ty(3));
    for (id, ty) in t.parts.types.iter() {
        for c in ty.children() {
            assert!(c < id);
        }
    }
}

/// `name` with each overload token of an `llvm.*` intrinsic that spells a typed pointer
/// (`p<space><pointee>`, as in `llvm.memcpy.p1i8.p1i8.i64`) spelled opaque (`p<space>`): the
/// pointee is the only part of such a name the typed spelling adds. A pointee whose mangling holds
/// a `.` is not modelled.
fn opaque_intrinsic_name(name: &str) -> String {
    if !name.starts_with("llvm.") {
        return name.to_owned();
    }
    let token = |t: &str| -> String {
        let digits = t.strip_prefix('p').map_or(0, |r| {
            r.len() - r.trim_start_matches(|c: char| c.is_ascii_digit()).len()
        });
        if digits > 0 {
            t[..1 + digits].to_owned()
        } else {
            t.to_owned()
        }
    };
    name.split('.').map(token).collect::<Vec<_>>().join(".")
}

/// What the typed and opaque spellings of one fixture must agree on (they differ only in pointees,
/// including the pointee manglings of overloaded `llvm.*` names, and in attribute spellings LLVM
/// upgrades, such as `argmemonly writeonly` versus `memory(argmem: write)` and `nocapture` versus
/// `captures(none)`). Definitions are compared in order and declarations as a set: the opaque text
/// is `llvm-dis` of the typed module, and LLVM's reader re-mangles a typed-pointer intrinsic
/// (`llvm.memcpy.p0i8.p0i8.i64` to `llvm.memcpy.p0.p0.i64`) into a new declaration after the
/// module's others, so a declaration's position is the reader's.
fn pointee_free_summary(t: &ModuleTables) -> Vec<String> {
    let p = &t.parts;
    let name = |s| opaque_intrinsic_name(&String::from_utf8_lossy(p.symbols.get(s).unwrap()));
    let mut out = Vec::new();
    let mut declared = Vec::new();
    for (id, f) in p.funcs.iter() {
        let params: Vec<_> = f
            .params
            .iter()
            .map(|q| {
                let a = &q.attrs;
                (a.align, a.noalias, a.nocapture, a.access, a.byval.is_some())
            })
            .collect();
        let line = format!(
            "fn {} body={} {:?} convergent={} memory={:?} params={params:?}",
            name(f.name),
            t.defined.contains(&id),
            f.linkage,
            f.attrs.convergent,
            f.attrs.memory,
        );
        if f.body.is_some() {
            out.push(line);
        } else {
            declared.push(line);
        }
    }
    declared.sort();
    out.extend(declared);
    for g in p.globals.values() {
        out.push(format!(
            "global {} {:?} const={} init={}",
            name(g.name),
            g.space,
            g.constant,
            g.init.is_some()
        ));
    }
    for n in &p.named_md {
        out.push(format!("!{} {}", name(n.name), n.ops.len()));
    }
    out.push(format!("metadata {}", p.metadata.len()));
    out
}

#[test]
fn typed_and_opaque_fixtures_agree_modulo_pointees() {
    let Some(tool) = llvm_as() else {
        eprintln!("SKIP typed_and_opaque_fixtures_agree_modulo_pointees: llvm-as not found");
        return;
    };
    if !have_metal_as() {
        eprintln!("SKIP typed_and_opaque_fixtures_agree_modulo_pointees: xcrun metal-as not found");
        return;
    }
    let scratch = Scratch::new();
    let (bc, air) = (scratch.0.join("o.bc"), scratch.0.join("t.air"));
    let dir = fixtures_dir().join("engine");
    let mut compared = 0;
    for ll in opaque_fixtures()
        .into_iter()
        .filter(|p| p.starts_with(&dir))
    {
        let air_ll = ll.with_extension("air.ll");
        let opaque = assemble(&tool, &[ll.as_os_str()], &bc).expect("llvm-as");
        let args = [std::ffi::OsStr::new("metal-as"), air_ll.as_os_str()];
        let typed = assemble(Path::new("xcrun"), &args, &air)
            .unwrap_or_else(|| panic!("xcrun metal-as rejected {air_ll:?}"));
        let o = decode_compared(stream_of(&opaque)).unwrap();
        let t = decode_compared(stream_of(&typed)).unwrap();
        assert_eq!(pointee_free_summary(&o), pointee_free_summary(&t), "{ll:?}");
        assert!(o.types.iter().all(|b| b.pointee.is_none()), "{ll:?}");
        assert!(t.types.iter().any(|b| b.pointee.is_some()), "{ll:?}");
        compared += 1;
    }
    assert!(compared > 0);
}
