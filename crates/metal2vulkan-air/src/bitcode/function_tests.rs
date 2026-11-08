//! Function-body tests: hand-assembled bodies (valid and malformed), the symbol-table
//! cross-checks, and mutation robustness.

use std::cell::Cell;

use super::codes;
use super::tests::{
    bitcode_kind, block, chars, decode, decode_compared, md_strings, mutate, stream, stream_v1,
    unsupported_feature, verify_module_level, vst_name,
};
use super::ModuleTables;
use crate::bitstream::writer::Writer;
use crate::error::BitcodeError;
use crate::error::{Error, Feature};
use crate::ir::print::print;
use crate::ir::{
    BlockId, Callee, Const, FuncId, InstId, InstKind, IntFlags, IntPredicate, Type, Unroll, Value,
};

const STRTAB: &[u8] = b"kair.x";
/// Forward-reference offsets are 32-bit wrapping: `next - id`.
const fn rel(next: u64, id: u64) -> u64 {
    next.wrapping_sub(id) & 0xFFFF_FFFF
}

/// Types: 0 `i32`, 1 `i1`, 2 `void`, 3 `i32 addrspace(1)*`, 4 `void (i32 addrspace(1)*, i32)`,
/// 5 `i32 (i32)`. Functions: `@k` (value 0, defined, type 4) and `@air.x` (value 1, declared,
/// type 5). Module constants: value 2 `i32 0`, value 3 `i32 7`. Metadata: 0 the string
/// `llvm.loop.mustprogress`, 1 `i32 0`, 2 `i32 7`, 3 `!{i32 0, i32 7}`, 4 `!{!0}`, 5 `distinct
/// !{!5, !4}`. Kinds: 0 `range`, 1 `llvm.loop`.
fn prefix(w: &mut Writer, k_defined: bool) {
    prefix_in(w, k_defined, true);
}

/// [`prefix`], the function records with their string-table fields (version 2) or without
/// (version 1).
fn prefix_in(w: &mut Writer, k_defined: bool, strtab: bool) {
    let name = |offset: u64, size: u64| if strtab { vec![offset, size] } else { vec![] };
    block(
        w,
        codes::TYPE_BLOCK,
        &[
            (1, vec![6]),
            (7, vec![32]),
            (7, vec![1]),
            (2, vec![]),
            (8, vec![0, 1]),
            (21, vec![0, 2, 3, 0]),
            (21, vec![0, 0, 0]),
        ],
    );
    w.unabbrev(
        8,
        &[name(0, 1), vec![4, 0, u64::from(!k_defined), 0, 0, 0, 0]].concat(),
    );
    w.unabbrev(8, &[name(1, 5), vec![5, 0, 1, 0, 0, 0, 0]].concat());
    block(
        w,
        codes::CONSTANTS_BLOCK,
        &[(1, vec![0]), (2, vec![]), (4, vec![14])],
    );
    w.enter(u64::from(codes::METADATA_KIND_BLOCK), 3);
    w.unabbrev(6, &[[0].as_slice(), &chars("range")].concat());
    w.unabbrev(6, &[[1].as_slice(), &chars("llvm.loop")].concat());
    w.end();
    w.enter(u64::from(codes::METADATA_BLOCK), 4);
    md_strings(w, &["llvm.loop.mustprogress"]);
    w.unabbrev(2, &[0, 2]);
    w.unabbrev(2, &[0, 3]);
    w.unabbrev(3, &[2, 3]);
    w.unabbrev(3, &[1]);
    w.unabbrev(5, &[6, 5]);
    w.end();
}

/// The valid body of `@k(%p, %n)`; values 4 `%p`, 5 `%n`, 6 the local `i32 3`:
///
/// ```text
/// bb0:  %7 = call i32 @air.x(i32 %n), !range !3      ; inst 0
///       br label %loop                               ; inst 1
/// loop: %8 = phi i32 [%7, bb0], [%9, loop]           ; inst 2 (forward reference to %9)
///       %9 = add nsw i32 %8, 3                       ; inst 3
///       %10 = icmp ult i32 %9, %n                    ; inst 4
///       br i1 %10, label %loop, label %bb2, !llvm.loop !5  ; inst 5
/// bb2:  store i32 %9, i32 addrspace(1)* %p, align 4  ; inst 6
///       ret void                                     ; inst 7
/// ```
fn good_records() -> Vec<(u64, Vec<u64>)> {
    vec![
        (1, vec![3]),
        (34, vec![0, 1 << 15, 5, rel(7, 1), rel(7, 5)]),
        (11, vec![1]),
        (16, vec![0, 2, 0, 3, 1]),
        (2, vec![rel(9, 8), rel(9, 6), 0, 2]),
        (28, vec![rel(10, 9), rel(10, 5), 36]),
        (11, vec![1, 2, rel(11, 10)]),
        (44, vec![rel(11, 4), rel(11, 9), 3, 0]),
        (10, vec![]),
    ]
}

/// A function block: `DECLAREBLOCKS` and the local constant `i32 3` first, then `records`,
/// then `tail` (symbol table, attachments).
fn function_block(w: &mut Writer, records: &[(u64, Vec<u64>)], tail: impl FnOnce(&mut Writer)) {
    w.enter(u64::from(codes::FUNCTION_BLOCK), 4);
    let (first, rest) = records.split_first().unwrap();
    w.unabbrev(first.0, &first.1);
    block(w, codes::CONSTANTS_BLOCK, &[(1, vec![0]), (4, vec![6])]);
    for (code, ops) in rest {
        w.unabbrev(*code, ops);
    }
    tail(w);
    w.end();
}

fn good_tail(w: &mut Writer) {
    let mut bb = vec![1];
    bb.extend(chars("loop"));
    block(
        w,
        codes::VALUE_SYMTAB_BLOCK,
        &[(1, vec![4, u64::from(b'p')]), (2, bb)],
    );
    block(
        w,
        codes::METADATA_ATTACHMENT_BLOCK,
        &[(11, vec![0, 0, 3]), (11, vec![5, 1, 5])],
    );
}

fn module(records: &[(u64, Vec<u64>)]) -> Vec<u8> {
    stream(2, 0, STRTAB, |w| {
        prefix(w, true);
        function_block(w, records, good_tail);
    })
}

fn good() -> Vec<u8> {
    module(&good_records())
}

fn body_of(t: &ModuleTables) -> &crate::ir::Body {
    t.parts
        .funcs
        .get(FuncId::new(0))
        .unwrap()
        .body
        .as_ref()
        .unwrap()
}

fn with_records(records: &[(u64, Vec<u64>)]) -> Result<ModuleTables, Error> {
    decode(&module(records))
}

/// `good_records()` with record `i` (counting `DECLAREBLOCKS` as 0) replaced.
fn replaced(i: usize, record: (u64, Vec<u64>)) -> Result<ModuleTables, Error> {
    let mut r = good_records();
    r[i] = record;
    with_records(&r)
}

#[test]
fn a_synthetic_body_decodes_every_part() {
    let t = decode(&good()).unwrap();
    assert_eq!(t.function_blocks, 1);
    let p = &t.parts;
    let body = body_of(&t);
    let i = |n: u32| &body.insts.get(InstId::new(n)).unwrap().kind;

    // Three blocks tile eight instructions; the loop block keeps its name.
    let spans: Vec<_> = body
        .blocks
        .values()
        .map(|b| (b.start.raw(), b.end.raw()))
        .collect();
    assert_eq!(spans, vec![(0, 2), (2, 6), (6, 8)]);
    let name = body.blocks.get(BlockId::new(1)).unwrap().name.unwrap();
    assert_eq!(p.symbols.get(name).unwrap(), b"loop");
    let k = p.funcs.get(FuncId::new(0)).unwrap();
    assert_eq!(p.symbols.get(k.params[0].name.unwrap()).unwrap(), b"p");
    assert_eq!(k.params[1].name, None);

    // `air.x` is outside the closed intrinsic set, so its call stays an unknown intrinsic.
    let InstKind::Call(call) = i(0) else {
        panic!("inst 0 is {:?}", i(0));
    };
    assert_eq!(call.callee, Callee::UnknownIntrinsic(FuncId::new(1)));
    assert_eq!(call.args[..], [Value::Arg(crate::ir::ArgIdx::new(1))]);
    assert_eq!(call.range.as_ref().unwrap().ranges[..], [(0, 7)]);

    // The PHI's forward reference resolves to the add.
    assert_eq!(
        i(2),
        &InstKind::Phi {
            incoming: Box::new([
                (Value::Inst(InstId::new(0)), BlockId::new(0)),
                (Value::Inst(InstId::new(3)), BlockId::new(1)),
            ]),
            fmf: Default::default(),
        }
    );
    let InstKind::BinOp { flags, a, b, .. } = i(3) else {
        panic!("inst 3 is {:?}", i(3));
    };
    assert_eq!(*flags, IntFlags::NSW);
    assert_eq!(*a, Value::Inst(InstId::new(2)));
    let Value::Const(three) = b else {
        panic!("the local constant is {b:?}");
    };
    assert_eq!(p.consts.get(*three).unwrap().value, Const::Int(3));
    assert!(matches!(
        i(4),
        InstKind::ICmp {
            pred: IntPredicate::Ult,
            ..
        }
    ));
    let InstKind::CondBr {
        t: yes,
        f: no,
        hints,
        ..
    } = i(5)
    else {
        panic!("inst 5 is {:?}", i(5));
    };
    assert_eq!((*yes, *no), (BlockId::new(1), BlockId::new(2)));
    let hints = hints.unwrap();
    assert!(hints.must_progress);
    assert_eq!(hints.unroll, Unroll::Default);
    assert!(matches!(i(6), InstKind::Store { .. }));
    assert_eq!(i(7), &InstKind::Ret(None));

    // The `i32 0` written as NULL is the integer zero.
    assert_eq!(
        p.consts.get(t.value(2).unwrap()).unwrap().value,
        Const::Int(0)
    );

    let module = verify_module_level(&t).unwrap();
    let text = print(&module);
    assert!(text.contains("phi"), "{text}");
}

#[test]
fn bodies_decode_deterministically() {
    let bytes = good();
    let a = decode(&bytes).unwrap();
    assert_eq!(a, decode(&bytes).unwrap());
    let (a, b) = (
        verify_module_level(&a).unwrap(),
        verify_module_level(&a).unwrap(),
    );
    assert_eq!(print(&a), print(&b));
}

#[test]
fn every_instruction_family_decodes() {
    // bb0 only: exercises records the loop body does not.
    let records = vec![
        (1, vec![1]),
        // %7 = alloca i32, i32 3 (the size is the local constant, by absolute id 6), align 4
        (19, vec![0, 0, 6, (1 << 6) | 3]),
        // %8 = load i32, ptr %7, align 4
        (20, vec![rel(8, 7), 0, 3, 0]),
        // %9 = atomicrmw add ptr %p, i32 %8 monotonic (system)
        (59, vec![rel(9, 4), rel(9, 8), 1, 0, 2, 1]),
        // %10 = cmpxchg ptr %p, i32 %8, i32 %9 acq_rel acquire (system)
        (46, vec![rel(10, 4), rel(10, 8), rel(10, 9), 0, 5, 1, 3, 0]),
        // %11 = extractvalue { i32, i1 } %10, 1
        (26, vec![rel(11, 10), 1]),
        // %12 = select i1 %11, i32 %8, i32 %9
        (29, vec![rel(12, 8), rel(12, 9), rel(12, 11)]),
        // fence seq_cst (singlethread)
        (36, vec![6, 0]),
        // %13 = getelementptr inbounds i32, ptr %p, i32 %12
        (43, vec![1, 0, rel(13, 4), rel(13, 12)]),
        // %14 = freeze i32 %12
        (58, vec![rel(14, 12)]),
        (10, vec![]),
    ];
    let t = stream(2, 0, STRTAB, |w| {
        prefix(w, true);
        function_block(w, &records, |_| {});
    });
    let t = decode(&t).unwrap();
    let body = body_of(&t);
    let kinds: Vec<_> = body.insts.values().map(|i| i.kind.opcode()).collect();
    use crate::ir::Opcode::*;
    assert_eq!(
        kinds,
        vec![
            Alloca,
            Load,
            AtomicRmw,
            CmpXchg,
            ExtractValue,
            Select,
            Fence,
            Gep,
            Freeze,
            Ret
        ]
    );
    let cmpxchg = body.insts.get(InstId::new(3)).unwrap();
    assert!(matches!(
        t.parts.types.get(cmpxchg.ty),
        Some(Type::Struct { fields, .. }) if fields.len() == 2
    ));
    verify_module_level(&t).unwrap();
}

/// Cast flags: `zext` has only `nneg`; LLVM 23 gives `uitofp` fast-math flags above its `nneg`
/// bit (and `sitofp` unshifted ones). All are dropped; a bit outside them is malformed.
#[test]
fn cast_flags_follow_llvm_23() {
    // Types 0 `i32`, 1 `float`, 2 `void`, 3 `void (i32)`, 4 `i64`; `@k(i32 %n)` is value 0, `%n`
    // value 1, and the cast value 2.
    let cast = |to: u64, opc: u64, flags: u64| {
        decode(&stream(2, 0, b"k", |w| {
            block(
                w,
                codes::TYPE_BLOCK,
                &[
                    (1, vec![5]),
                    (7, vec![32]),
                    (3, vec![]),
                    (2, vec![]),
                    (21, vec![0, 2, 0]),
                    (7, vec![64]),
                ],
            );
            w.unabbrev(8, &[0, 1, 3, 0, 0, 0, 0, 0, 0]);
            block(
                w,
                codes::FUNCTION_BLOCK,
                &[
                    (1, vec![1]),
                    (3, vec![rel(2, 1), to, opc, flags]),
                    (10, vec![]),
                ],
            );
        }))
    };
    // `uitofp nneg ... fast`, `sitofp fast`, `zext nneg`.
    for (to, opc, flags) in [(1, 5, 1 | (0xFF << 1)), (1, 6, 0xFF), (4, 1, 1)] {
        let t = cast(to, opc, flags).unwrap_or_else(|e| panic!("{opc} {flags:#x}: {e}"));
        assert_eq!(t, cast(to, opc, 0).unwrap());
        verify_module_level(&t).unwrap();
    }
    for (to, opc, flags) in [(1, 5, 1 << 9), (1, 6, 1 << 8), (4, 1, 2)] {
        let e = cast(to, opc, flags).unwrap_err();
        assert_eq!(
            bitcode_kind(&e),
            Some(&BitcodeError::MalformedRecord { block: 12, code: 3 }),
            "{opc} {flags:#x}: {e}"
        );
    }
}

#[test]
fn switches_decode_absolute_case_values() {
    let records = vec![
        (1, vec![3]),
        // switch i32 %n, label %bb2 [i32 0, label %bb1; i32 7, label %bb2]
        (12, vec![0, rel(7, 5), 2, 2, 1, 3, 2]),
        (10, vec![]),
        (10, vec![]),
    ];
    let t = decode(&stream(2, 0, STRTAB, |w| {
        prefix(w, true);
        function_block(w, &records, |_| {});
    }))
    .unwrap();
    let InstKind::Switch { default, cases, .. } =
        &body_of(&t).insts.get(InstId::new(0)).unwrap().kind
    else {
        panic!("not a switch");
    };
    assert_eq!(*default, BlockId::new(2));
    assert_eq!(cases[..], [(0, BlockId::new(1)), (7, BlockId::new(2))]);
    verify_module_level(&t).unwrap();
}

#[test]
fn record_errors_are_typed() {
    let malformed = |r: Result<ModuleTables, Error>, block: u32, code: u32| {
        let e = r.unwrap_err();
        assert_eq!(
            bitcode_kind(&e),
            Some(&BitcodeError::MalformedRecord { block, code }),
            "{e}"
        );
    };
    let bad_ref = |r: Result<ModuleTables, Error>| {
        let e = r.unwrap_err();
        assert_eq!(bitcode_kind(&e), Some(&BitcodeError::BadValueRef), "{e}");
    };
    let unsupported = |r: Result<ModuleTables, Error>, f: Feature| {
        let e = r.unwrap_err();
        assert_eq!(unsupported_feature(&e), Some(&f), "{e}");
    };

    // Instructions outside the closed set, and codes LLVM does not define.
    unsupported(replaced(8, (13, vec![])), Feature::Instruction { code: 13 });
    let e = replaced(8, (99, vec![])).unwrap_err();
    assert_eq!(
        bitcode_kind(&e),
        Some(&BitcodeError::UnknownRecord {
            block: 12,
            code: 99
        })
    );
    // A load without an explicit type (the typed-pointer spelling) and a calling convention
    // other than C or fast.
    unsupported(
        replaced(7, (20, vec![rel(11, 4), 3, 0])),
        Feature::Instruction { code: 20 },
    );
    unsupported(
        replaced(
            1,
            (34, vec![0, (1 << 15) | (9 << 1), 5, rel(7, 1), rel(7, 5)]),
        ),
        Feature::CallingConvention(9),
    );
    // A forward reference that is never defined, or defined with another type (the PHI names
    // `%10`, an `i1`, as an `i32`: signed offset 8 - 10 = -2, encoded 5).
    bad_ref(replaced(7, (44, vec![rel(11, 4), rel(11, 40), 0, 3, 0])));
    bad_ref(replaced(3, (16, vec![0, 2, 0, 5, 1])));
    // Backward references of the wrong type (`%p` as an `i32`; `%7` as an `i1`).
    bad_ref(replaced(4, (2, vec![rel(9, 8), rel(9, 4), 0])));
    bad_ref(replaced(3, (16, vec![1, 2, 0, 3, 1])));
    // Branches to undeclared blocks; a flag operand that is not 0/1; unknown flag bits.
    malformed(replaced(2, (11, vec![3])), 12, 11);
    malformed(
        replaced(7, (44, vec![rel(11, 4), rel(11, 9), 3, 2])),
        12,
        44,
    );
    malformed(replaced(4, (2, vec![rel(9, 8), rel(9, 6), 0, 4])), 12, 2);
    // Fewer blocks than declared, and an instruction after the last declared block.
    let mut r = good_records();
    r[0] = (1, vec![4]);
    malformed(with_records(&r), 12, 1);
    let mut r = good_records();
    r.push((10, vec![]));
    malformed(with_records(&r), 12, 10);
    // A second constants block after the first instruction.
    let e = decode(&stream(2, 0, STRTAB, |w| {
        prefix(w, true);
        function_block(w, &good_records(), |w| {
            block(w, codes::CONSTANTS_BLOCK, &[(1, vec![0]), (4, vec![2])]);
        });
    }))
    .unwrap_err();
    assert_eq!(
        bitcode_kind(&e),
        Some(&BitcodeError::MalformedRecord { block: 11, code: 0 })
    );
    // A sync scope outside the module's table, and a custom named scope.
    malformed(replaced(8, (36, vec![6, 7])), 12, 36);
    let e = decode(&stream(2, 0, STRTAB, |w| {
        prefix(w, true);
        block(
            w,
            codes::SYNC_SCOPE_NAMES_BLOCK,
            &[(1, chars("singlethread")), (1, vec![]), (1, chars("agent"))],
        );
        let mut r = good_records();
        r.insert(8, (36, vec![6, 2]));
        function_block(w, &r, |_| {});
    }))
    .unwrap_err();
    assert_eq!(unsupported_feature(&e), Some(&Feature::SyncScope), "{e}");
}

#[test]
fn attachments_are_checked() {
    let with_tail = |tail: &dyn Fn(&mut Writer)| {
        decode(&stream(2, 0, STRTAB, |w| {
            prefix(w, true);
            function_block(w, &good_records(), tail);
        }))
    };
    // `!range` on a branch and `!llvm.loop` on a call are malformed; a missing node is bad
    // metadata; an instruction out of range is malformed; other kinds are dropped.
    for (record, block_err) in [
        (
            vec![1, 0, 3],
            BitcodeError::MalformedRecord {
                block: 16,
                code: 11,
            },
        ),
        (
            vec![0, 1, 5],
            BitcodeError::MalformedRecord {
                block: 16,
                code: 11,
            },
        ),
        (vec![0, 0, 60], BitcodeError::BadMetadata),
        (
            vec![40, 7, 3],
            BitcodeError::MalformedRecord {
                block: 16,
                code: 11,
            },
        ),
    ] {
        let e = with_tail(&|w| block(w, codes::METADATA_ATTACHMENT_BLOCK, &[(11, record.clone())]))
            .unwrap_err();
        assert_eq!(bitcode_kind(&e), Some(&block_err), "{record:?}: {e}");
    }
    let t =
        with_tail(&|w| block(w, codes::METADATA_ATTACHMENT_BLOCK, &[(11, vec![6, 7, 3])])).unwrap();
    verify_module_level(&t).unwrap();
}

#[test]
fn function_blocks_match_the_defined_functions() {
    // A definition without a body, and a body without a definition.
    let e = decode(&stream(2, 0, STRTAB, |w| prefix(w, true))).unwrap_err();
    assert_eq!(
        bitcode_kind(&e),
        Some(&BitcodeError::MalformedRecord { block: 12, code: 0 })
    );
    let e = decode(&stream(2, 0, STRTAB, |w| {
        prefix(w, false);
        function_block(w, &good_records(), good_tail);
    }))
    .unwrap_err();
    assert_eq!(
        bitcode_kind(&e),
        Some(&BitcodeError::MalformedRecord { block: 12, code: 0 })
    );
}

/// A module with `VSTOFFSET = vst` and a module symbol table naming `@k` at word `entry`,
/// plus the bit positions of the function block and the symbol table.
fn with_vst(vst: Option<u64>, entry: u64) -> (Vec<u8>, u64, u64) {
    let at = Cell::new((0, 0));
    let bytes = stream(2, 0, STRTAB, |w| {
        if let Some(v) = vst {
            w.unabbrev(13, &[v]);
        }
        prefix(w, true);
        let f = w.bit();
        function_block(w, &good_records(), good_tail);
        let v = w.bit();
        block(w, codes::VALUE_SYMTAB_BLOCK, &[(3, vec![0, entry])]);
        at.set((f, v));
    });
    let (f, v) = at.get();
    (bytes, f, v)
}

#[test]
fn the_module_symbol_table_locates_every_body() {
    // Iterate to the fixed point where the offsets name the positions they are written at.
    let (mut vst, mut entry) = (0, 0);
    for _ in 0..8 {
        let (_, f, v) = with_vst(Some(vst), entry);
        if (vst, entry) == (v / 32, f / 32) {
            break;
        }
        (vst, entry) = (v / 32, f / 32);
    }
    let (bytes, f, v) = with_vst(Some(vst), entry);
    assert_eq!((f % 32, v % 32), (0, 0));
    assert_eq!((vst, entry), (v / 32, f / 32));
    let t = decode(&bytes).unwrap();
    assert_eq!(t.fn_entries, vec![(FuncId::new(0), entry)]);

    let err = |bytes: Vec<u8>| bitcode_kind(&decode(&bytes).unwrap_err()).cloned();
    let vst_err = Some(BitcodeError::MalformedRecord {
        block: codes::MODULE_BLOCK,
        code: codes::MODULE_VSTOFFSET,
    });
    assert_eq!(err(with_vst(Some(vst + 1), entry).0), vst_err);
    assert_eq!(err(with_vst(None, entry).0), vst_err);
    assert_eq!(
        err(with_vst(Some(vst), entry + 1).0),
        Some(BitcodeError::MalformedRecord {
            block: codes::VALUE_SYMTAB_BLOCK,
            code: codes::VST_FNENTRY,
        })
    );
}

/// The module `build(vst, entry)` writes at the fixed point where `VSTOFFSET = vst` and `@k`'s
/// offset `entry` name the words its symbol table and function block are written at; `build`
/// returns the bytes and the bit positions of the function block and the symbol table.
fn located(build: impl Fn(u64, u64) -> (Vec<u8>, u64, u64)) -> (Vec<u8>, u64, u64) {
    let (mut vst, mut entry) = (0, 0);
    for _ in 0..8 {
        let (_, f, v) = build(vst, entry);
        if (vst, entry) == (v / 32, f / 32) {
            break;
        }
        (vst, entry) = (v / 32, f / 32);
    }
    let (bytes, f, v) = build(vst, entry);
    assert_eq!((f % 32, v % 32), (0, 0));
    assert_eq!((vst, entry), (v / 32, f / 32));
    (bytes, vst, entry)
}

/// [`good`] in module version 1: `VSTOFFSET = vst` when given, the function records without
/// name fields, `@k`'s body, then the module symbol table `names(entry)`; with the bit positions
/// of the function block and the symbol table.
fn v1_module(
    vst: Option<u64>,
    entry: u64,
    names: &dyn Fn(u64) -> Vec<(u64, Vec<u64>)>,
) -> (Vec<u8>, u64, u64) {
    let at = Cell::new((0, 0));
    let bytes = stream_v1(|w| {
        if let Some(v) = vst {
            w.unabbrev(13, &[v]);
        }
        prefix_in(w, true, false);
        let f = w.bit();
        function_block(w, &good_records(), good_tail);
        let v = w.bit();
        block(w, codes::VALUE_SYMTAB_BLOCK, &names(entry));
        at.set((f, v));
    });
    let (f, v) = at.get();
    (bytes, f, v)
}

/// What LLVM 3.8 and later write in version 1: `@k`'s name with its block's offset, and the
/// declaration's name.
fn v1_names(entry: u64) -> Vec<(u64, Vec<u64>)> {
    vec![
        vst_name(codes::VST_FNENTRY, &[0, entry], "k"),
        vst_name(codes::VST_ENTRY, &[1], "air.x"),
    ]
}

fn v1_good() -> (Vec<u8>, u64) {
    let (bytes, _, entry) = located(|vst, entry| v1_module(Some(vst), entry, &v1_names));
    (bytes, entry)
}

fn decoded(bytes: &[u8]) -> super::Decoded {
    super::decoded::decode(bytes, &mut super::tests::meter()).unwrap()
}

/// A version-1 module (names in the module symbol table, which LLVM writes after the function
/// blocks) decodes to the module its version-2 twin does: the declaration's name, read ahead of
/// the body that calls it, classifies it as the version-2 name does. So does the layout before
/// LLVM 3.8 (no `VSTOFFSET`, the symbol table ahead of the bodies, which follow record order).
#[test]
fn a_version_1_body_decodes_as_its_version_2_twin() {
    let (v2, _, _) = located(|vst, entry| with_vst(Some(vst), entry));
    let (v1, entry) = v1_good();
    let t = decode(&v1).unwrap();
    assert_eq!(t.fn_entries, vec![(FuncId::new(0), entry)]);
    assert_eq!(t.parts, decode(&v2).unwrap().parts);
    assert_eq!(decoded(&v1), decoded(&v2));

    let early = stream_v1(|w| {
        prefix_in(w, true, false);
        block(
            w,
            codes::VALUE_SYMTAB_BLOCK,
            &[
                vst_name(codes::VST_ENTRY, &[0], "k"),
                vst_name(codes::VST_ENTRY, &[1], "air.x"),
            ],
        );
        function_block(w, &good_records(), good_tail);
    });
    assert!(decode(&early).unwrap().fn_entries.is_empty());
    assert_eq!(decoded(&early), decoded(&v2));

    // LLVM's version-1 writer names only named values: an unnamed function has no entry, and its
    // body is found by record order.
    let (unnamed, _, _) = located(|vst, entry| {
        v1_module(Some(vst), entry, &|_| {
            vec![vst_name(codes::VST_ENTRY, &[1], "air.x")]
        })
    });
    let t = decode(&unnamed).unwrap();
    assert!(t.fn_entries.is_empty());
    let k = t.parts.funcs.get(FuncId::new(0)).unwrap();
    assert_eq!(t.parts.symbols.get(k.name), Some(&b""[..]));
    assert!(k.body.is_some());
}

#[test]
fn a_version_1_symbol_table_locates_every_body() {
    let err = |bytes: Vec<u8>| bitcode_kind(&decode(&bytes).unwrap_err()).cloned();
    let vst_err = Some(BitcodeError::MalformedRecord {
        block: codes::MODULE_BLOCK,
        code: codes::MODULE_VSTOFFSET,
    });
    let entry_err = Some(BitcodeError::MalformedRecord {
        block: codes::VALUE_SYMTAB_BLOCK,
        code: codes::VST_FNENTRY,
    });
    let (_, vst, entry) = located(|vst, entry| v1_module(Some(vst), entry, &v1_names));
    // A wrong table offset, a function offset with no table offset, a wrong function offset.
    assert_eq!(err(v1_module(Some(vst + 1), entry, &v1_names).0), vst_err);
    assert_eq!(err(v1_module(None, entry, &v1_names).0), vst_err);
    assert_eq!(err(v1_module(Some(vst), entry + 1, &v1_names).0), entry_err);
    // A function offset for a declaration.
    let on_declaration = |entry| {
        vec![
            vst_name(codes::VST_FNENTRY, &[0, entry], "k"),
            vst_name(codes::VST_FNENTRY, &[1, entry], "air.x"),
        ]
    };
    assert_eq!(
        err(v1_module(Some(vst), entry, &on_declaration).0),
        entry_err
    );
}

#[test]
fn mutated_bodies_never_panic() {
    mutate(&good());
    mutate(&v1_good().0);
}

#[test]
fn decoded_modules_carry_their_parameter_pointees() {
    let d = super::decoded::decode(&good(), &mut super::tests::meter()).unwrap();
    let i32t = d.module.types().lookup(&Type::Int(32)).unwrap();
    assert_eq!(
        d.pointees.params(FuncId::new(0)),
        Some(&[Some(i32t), None][..])
    );
    assert_eq!(d.pointees.param(FuncId::new(0), 0), Some(i32t));
    assert_eq!(d.pointees.params(FuncId::new(1)), Some(&[None][..]));
    assert_eq!(d.pointees.ret(FuncId::new(0)), None);
    assert!(!d.pointees.is_empty());
    // The wrapper spelling decodes to the same module.
    let raw = good();
    let mut wrapped = Vec::new();
    for word in [0x0B17_C0DE_u32, 0, 20, raw.len() as u32, 0] {
        wrapped.extend_from_slice(&word.to_le_bytes());
    }
    wrapped.extend_from_slice(&raw);
    let w = super::decoded::decode(&wrapped, &mut super::tests::meter()).unwrap();
    assert_eq!(w, d);
}

#[test]
fn function_metadata_continues_the_module_numbering() {
    // The module has metadata 0..=5; the function's own block defines 6 `!{!0}` and 7
    // `distinct !{!7, !6}`, and a second block 8 onwards.
    let with_md = |extra: &[(u64, Vec<u64>)], attach: u64| {
        let extra = extra.to_vec();
        decode(&stream(2, 0, STRTAB, move |w| {
            prefix(w, true);
            function_block(w, &good_records(), move |w| {
                block(w, codes::METADATA_BLOCK, &[(3, vec![1]), (5, vec![8, 7])]);
                block(w, codes::METADATA_BLOCK, &extra);
                block(
                    w,
                    codes::METADATA_ATTACHMENT_BLOCK,
                    &[(11, vec![5, 1, attach])],
                );
            });
        }))
    };
    for attach in [7, 8] {
        let t = with_md(&[(5, vec![9, 7])], attach).unwrap();
        let InstKind::CondBr { hints, .. } = &body_of(&t).insts.get(InstId::new(5)).unwrap().kind
        else {
            panic!("inst 5 is not a conditional branch");
        };
        assert!(hints.unwrap().must_progress, "attachment {attach}");
        // Function metadata does not leak into the module.
        assert_eq!(t.parts.metadata.len(), 6);
    }
    // A reference past the function's metadata, and `LocalAsMetadata` of an argument.
    let e = with_md(&[(3, vec![11])], 7).unwrap_err();
    assert_eq!(bitcode_kind(&e), Some(&BitcodeError::BadMetadata), "{e}");
    let e = with_md(&[(2, vec![0, 5])], 7).unwrap_err();
    assert_eq!(
        unsupported_feature(&e),
        Some(&Feature::BitcodeRecord {
            block: codes::METADATA_BLOCK,
            code: 2
        }),
        "{e}"
    );
}

/// How a module spells its `llvm.lifetime.*` declarations.
#[derive(Clone, Copy)]
enum Markers {
    /// As LLVM 21 writes them, `void (i64, ptr)`, declared before the kernel.
    SizedFirst,
    /// As LLVM 22 onwards writes them, `void (ptr)`, declared after the kernel (where LLVM's
    /// reader appends a re-mangled declaration).
    UnsizedLast,
}

/// `@k(ptr %p, i64 %n)` storing `i64 8` to `%p` between `lifetime.start` and `lifetime.end` of
/// `%p`, with the markers spelled as `markers` says. Types: 0 `i64`, 1 `void`, 2 `ptr`, then the
/// kernel's `void (ptr, i64)` and the markers' type, in declaration order. With `taken`, `@k` also
/// stores `lifetime.start`'s address to `%p`. The sized spelling's size is the stored `i64 8`, or
/// with `own_size` a constant `i64 16` nothing else uses, numbered first (as LLVM 21 writes a size
/// no instruction shares: its plane sorted by use count).
fn marker_module(markers: Markers, taken: bool, own_size: bool) -> Vec<u8> {
    const NAMES: &[u8] = b"kllvm.lifetime.start.p0llvm.lifetime.end.p0";
    let kernel = (21, vec![0, 1, 2, 0]);
    let (types, k_ty, marker_ty, order): (_, u64, u64, [usize; 3]) = match markers {
        Markers::SizedFirst => (vec![kernel, (21, vec![0, 1, 0, 2])], 3, 4, [1, 2, 0]),
        Markers::UnsizedLast => (vec![(21, vec![0, 1, 2]), kernel], 4, 3, [0, 1, 2]),
    };
    // Function records: (strtab offset, size, type, isproto) of k, start, end.
    let records = [
        (0, 1, k_ty, 0),
        (1, 22, marker_ty, 1),
        (23, 20, marker_ty, 1),
    ];
    // Each function's value id is its position in record order.
    let value = |f: usize| order.iter().position(|&o| o == f).unwrap() as u64;
    // Values 3 `%p` and 4 `%n`, then the local constants: `i64 16` when it is the size's own,
    // then `i64 8`.
    let own_size = own_size && matches!(markers, Markers::SizedFirst);
    let mut consts = vec![(1, vec![0])];
    if own_size {
        consts.push((4, vec![32]));
    }
    consts.push((4, vec![16]));
    let eight = if own_size { 6 } else { 5 };
    let next = eight + 1;
    let marker_call = |f: usize| {
        let mut ops = vec![0, 1 << 15, marker_ty, rel(next, value(f))];
        if let Markers::SizedFirst = markers {
            ops.push(rel(next, 5));
        }
        ops.push(rel(next, 3));
        (34, ops)
    };
    let mut body = vec![
        (1, vec![1]),
        marker_call(1),
        (44, vec![rel(next, 3), rel(next, eight), 4, 0]),
    ];
    if taken {
        body.push((44, vec![rel(next, 3), rel(next, value(1)), 4, 0]));
    }
    body.push(marker_call(2));
    body.push((10, vec![]));
    stream(2, 0, NAMES, |w| {
        let mut t = vec![(1, vec![5]), (7, vec![64]), (2, vec![]), (25, vec![0])];
        t.extend(types);
        block(w, codes::TYPE_BLOCK, &t);
        for f in order {
            let (offset, size, ty, proto) = records[f];
            w.unabbrev(8, &[offset, size, ty, 0, proto, 0, 0, 0, 0]);
        }
        // Values: the three functions, %p 3, %n 4, the local constants from 5.
        w.enter(u64::from(codes::FUNCTION_BLOCK), 4);
        let (first, rest) = body.split_first().unwrap();
        w.unabbrev(first.0, &first.1);
        block(w, codes::CONSTANTS_BLOCK, &consts);
        for (code, ops) in rest {
            w.unabbrev(*code, ops);
        }
        w.end();
    })
}

/// Asked to (`DecodeOptions::drop_unused_markers`), decode removes the marker declarations no use
/// survives, so the two spellings decode alike; a translation's decode leaves them, as pruning
/// drops them anyway.
#[test]
fn unused_marker_declarations_decode_alike_however_llvm_spells_them() {
    let sized = decode_compared(&marker_module(Markers::SizedFirst, false, false)).unwrap();
    let unsized_ = decode_compared(&marker_module(Markers::UnsizedLast, false, false)).unwrap();
    let text = |t: &ModuleTables| print(&verify_module_level(t).unwrap());
    // Each spelling's own declarations, types, and address constants are gone, and what remains
    // is numbered in its old order: the two decode to one module.
    assert_eq!(text(&sized), text(&unsized_));
    assert_eq!(sized.parts.types, unsized_.parts.types);
    assert_eq!(sized.parts.consts, unsized_.parts.consts);
    for t in [&sized, &unsized_] {
        assert_eq!(t.parts.funcs.len(), 1);
        assert!(!text(t).contains("lifetime"), "{}", text(t));
        assert_eq!(t.defined, [FuncId::new(0)]);
        // The kernel's value keeps its id; the markers' values have no constant.
        let kept: Vec<_> = t.values.iter().take(3).map(|v| v.id.is_some()).collect();
        let k = t.values.iter().position(|v| v.id.is_some()).unwrap();
        assert_eq!(kept.iter().filter(|&&k| k).count(), 1);
        assert_eq!(
            t.parts.consts.get(t.values[k].id.unwrap()).unwrap().value,
            Const::FuncAddr(FuncId::new(0))
        );
        // The markers' type (the second function type) has no IR type left.
        assert_eq!(t.types.iter().filter(|b| b.ty.is_none()).count(), 1);
    }
    // The type table: i64, void, ptr, and the kernel's type.
    assert_eq!(sized.parts.types.len(), 4);
    // Without the option, both declarations stay (their calls are still dropped).
    let plain = decode(&marker_module(Markers::SizedFirst, false, false)).unwrap();
    assert_eq!(plain.parts.funcs.len(), 3);
    assert!(plain.values.iter().all(|v| v.id.is_some()));
}

/// LLVM 21 writes `lifetime.*`'s size operand, LLVM 22 none, so a size no kept instruction uses
/// is a constant only the older spelling has. Asked to, decode removes it with the calls: the
/// spellings decode alike, and the constant after it takes its place. Without the option it
/// stays; a size another instruction uses stays either way (the shared `i64 8` above).
#[test]
fn a_constant_only_dropped_marker_calls_named_goes_with_them() {
    let own = marker_module(Markers::SizedFirst, false, true);
    let sized = decode_compared(&own).unwrap();
    let unsized_ = decode_compared(&marker_module(Markers::UnsizedLast, false, true)).unwrap();
    let text = |t: &ModuleTables| print(&verify_module_level(t).unwrap());
    assert_eq!(text(&sized), text(&unsized_));
    assert_eq!(sized.parts.consts, unsized_.parts.consts);
    assert_eq!(sized.parts.types, unsized_.parts.types);
    let sixteen = |t: &ModuleTables| {
        t.parts
            .consts
            .iter()
            .any(|(_, c)| c.value == Const::Int(16))
    };
    assert!(!sixteen(&sized), "{}", text(&sized));
    assert!(sixteen(&decode(&own).unwrap()));
}

/// `@k()` calling `llvm.assume(i1 true)` when `assumed`, else only returning. Types: 0 `void`, 1
/// `void ()`, then with the call 2 `i1` and 3 `void (i1)`: `i1` only `true` and the marker use.
fn assume_module(assumed: bool) -> Vec<u8> {
    stream(2, 0, b"kllvm.assume", |w| {
        let mut types = vec![
            (1, vec![2 + 2 * u64::from(assumed)]),
            (2, vec![]),
            (21, vec![0, 0]),
        ];
        if assumed {
            types.extend([(7, vec![1]), (21, vec![0, 0, 2])]);
        }
        block(w, codes::TYPE_BLOCK, &types);
        w.unabbrev(8, &[0, 1, 1, 0, 0, 0, 0, 0, 0]);
        if assumed {
            w.unabbrev(8, &[1, 11, 3, 0, 1, 0, 0, 0, 0]);
        }
        // Values: `@k` 0, `@llvm.assume` 1, the local `i1 true` 2.
        w.enter(u64::from(codes::FUNCTION_BLOCK), 4);
        w.unabbrev(1, &[1]);
        if assumed {
            block(w, codes::CONSTANTS_BLOCK, &[(1, vec![2]), (4, vec![3])]);
            w.unabbrev(34, &[0, 1 << 15, 3, rel(3, 1), rel(3, 2)]);
        }
        w.unabbrev(10, &[]);
        w.end();
    })
}

/// A type only a removed constant and the removed declaration used goes with them: the module
/// with an `llvm.assume(i1 true)` decodes as the one without it.
#[test]
fn a_type_only_a_dropped_markers_operands_used_goes_with_them() {
    let text = |t: &ModuleTables| print(&verify_module_level(t).unwrap());
    let assumed = decode_compared(&assume_module(true)).unwrap();
    let plain = decode_compared(&assume_module(false)).unwrap();
    assert_eq!(text(&assumed), text(&plain));
    assert_eq!(assumed.parts.types, plain.parts.types);
    assert_eq!(assumed.parts.consts, plain.parts.consts);
    assert_eq!(assumed.parts.funcs.len(), 1);
    // Without the option the declaration, `i1 true`, and `i1` stay.
    let kept = decode(&assume_module(true)).unwrap();
    assert_eq!(kept.parts.funcs.len(), 2);
    assert!(kept.parts.types.iter().any(|(_, t)| *t == Type::Int(1)));
}

#[test]
fn a_marker_declaration_with_another_use_is_kept() {
    let t = decode_compared(&marker_module(Markers::SizedFirst, true, false)).unwrap();
    let text = print(&verify_module_level(&t).unwrap());
    // `lifetime.start`'s address is stored, so its declaration stays; `lifetime.end` goes.
    assert_eq!(t.parts.funcs.len(), 2, "{text}");
    assert!(text.contains("llvm.lifetime.start.p0"), "{text}");
    assert!(!text.contains("llvm.lifetime.end.p0"), "{text}");
    // Both calls are still dropped.
    let calls = body_of_named(&t, b"k")
        .insts
        .values()
        .filter(|i| matches!(i.kind, InstKind::Call(_)))
        .count();
    assert_eq!(calls, 0);
}

fn body_of_named<'t>(t: &'t ModuleTables, name: &[u8]) -> &'t crate::ir::Body {
    t.parts
        .funcs
        .values()
        .find(|f| t.parts.symbols.get(f.name) == Some(name))
        .and_then(|f| f.body.as_ref())
        .unwrap()
}
