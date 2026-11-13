//! A gated-off argument stored into private memory: the store is observed only once an observed
//! read reaches its bytes, or the allocation escapes. An unobserved read of the argument is
//! `undef`; an observed one reads zero, so each test tells the two apart by the value the
//! specialized body stores or passes.

use super::*;
use crate::intrinsics::{Elem, LlvmIntrinsic, Math, MathFn, NumTy, Precision};

/// How [`private_store`]'s kernel reads its private struct `s`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Reach {
    /// The present field only.
    Kept,
    /// The gated-off field.
    Gone,
    /// The whole struct, as one aggregate load (only the present field is extracted).
    Whole,
    /// One byte of the gated-off field, through a byte offset.
    Byte,
    /// The gated-off field through the pointer loaded back from the cell holding `&s`.
    Cell,
    /// All of `s`, copied to `out` by `memcpy`.
    Copy,
    /// The present field, with `&s` also stored to `out` as an integer.
    Escape,
    /// The present field, beside a `memset` of an unread private array filled with `gone`.
    Fill,
    /// The present field, beside a `memset` of an unread private array `gone` bytes long.
    Length,
    /// The present field, beside a `memcpy` of `s` into an unread private array, `gone` bytes long.
    CopyLength,
    /// The present field, beside a store into an unread private array at index `gone`.
    Index,
}

/// `kernel k(uint kept, uint gone [[off]], device uint* out)` stores `kept` and `gone` into the two
/// fields of a private `struct { uint, uint } s`, stores `&s` into a private pointer cell `{ thread
/// void*, uint }` (as a surface function's argument struct holds it), then reads as `reach` says
/// and stores what it read to `out`. The last four shapes also write an unread private `uchar
/// arr[16]`.
fn private_store(reach: Reach) -> Result<Entry, Error> {
    let mut m = ModuleBuilder::new();
    let void = m.void();
    let i1 = m.int(1);
    let i8t = m.int(8);
    let i32t = m.int(32);
    let i64t = m.int(64);
    let dev = m.ptr(AddrSpace::Device);
    let thread = m.ptr(AddrSpace::Thread);
    let pair = m.struct_ty(&[i32t, i32t]);
    let cell_ty = m.struct_ty(&[thread, i32t]);
    let off = predicate(&mut m, "off", 0);
    let memcpy_ty = m.fn_ty(void, &[dev, thread, i64t, i1]);
    let memcpy = m.declare("llvm.memcpy.p1.p0.i64", memcpy_ty);
    let local_memcpy_ty = m.fn_ty(void, &[thread, thread, i64t, i1]);
    let local_memcpy = m.declare("llvm.memcpy.p0.p0.i64", local_memcpy_ty);
    let memset_ty = m.fn_ty(void, &[thread, i8t, i64t, i1]);
    let memset = m.declare("llvm.memset.p0.i64", memset_ty);
    let bytes = m.array(i8t, 16);
    let fty = m.fn_ty(void, &[i32t, i32t, dev]);
    let one = m.const_int(i32t, 1);
    let (field0, field1) = (m.const_int(i32t, 0), m.const_int(i32t, 1));
    let zero = m.const_int(i64t, 0);
    let four = m.const_int(i64t, 4);
    let eight = m.const_int(i64t, 8);
    let sixteen = m.const_int(i64t, 16);
    let zero8 = m.const_int(i8t, 0);
    let no = m.const_int(i1, 0);
    let mut f = m.define("k", fty);
    let alloca = |f: &mut FunctionBuilder<'_>, ty: TypeId| {
        f.push(
            thread,
            InstKind::Alloca {
                ty,
                count: one,
                align: None,
            },
        )
    };
    let gep = |f: &mut FunctionBuilder<'_>, src_ty: TypeId, base: Value, indices: &[Value]| {
        f.push(
            thread,
            InstKind::Gep {
                src_ty,
                base,
                indices: indices.into(),
                inbounds: true,
            },
        )
    };
    let s = alloca(&mut f, pair);
    let kept = gep(&mut f, pair, s, &[zero, field0]);
    store(&mut f, arg(0), kept);
    let gone = gep(&mut f, pair, s, &[zero, field1]);
    store(&mut f, arg(1), gone);
    let cell = alloca(&mut f, cell_ty);
    let slot = gep(&mut f, cell_ty, cell, &[zero, field0]);
    store(&mut f, s, slot);
    let read = match reach {
        Reach::Kept => Some(load(&mut f, i32t, kept)),
        Reach::Gone => Some(load(&mut f, i32t, gone)),
        Reach::Whole => {
            let w = load(&mut f, pair, s);
            Some(f.push(
                i32t,
                InstKind::ExtractValue {
                    agg: w,
                    indices: [0].into(),
                },
            ))
        }
        Reach::Byte => {
            let b = gep(&mut f, i8t, s, &[four]);
            Some(load(&mut f, i8t, b))
        }
        Reach::Cell => {
            let p = load(&mut f, thread, slot);
            let q = gep(&mut f, pair, p, &[zero, field1]);
            Some(load(&mut f, i32t, q))
        }
        Reach::Copy => {
            let callee = Callee::Llvm {
                decl: memcpy,
                op: LlvmIntrinsic::Memcpy {
                    dst: AddrSpace::Device,
                    src: AddrSpace::Thread,
                    len: Elem::I64,
                    volatile: false,
                },
            };
            call(&mut f, void, callee, &[arg(2), s, eight, no]);
            None
        }
        Reach::Escape => {
            let a = f.push(
                i64t,
                InstKind::Cast {
                    op: CastOp::PtrToInt,
                    v: s,
                },
            );
            store(&mut f, a, arg(2));
            Some(load(&mut f, i32t, kept))
        }
        Reach::Fill | Reach::Length | Reach::CopyLength | Reach::Index => {
            let arr = alloca(&mut f, bytes);
            let cast = |f: &mut FunctionBuilder<'_>, op: CastOp, ty: TypeId| {
                f.push(ty, InstKind::Cast { op, v: arg(1) })
            };
            let memset_call = |f: &mut FunctionBuilder<'_>, fill: Value, len: Value| {
                let op = LlvmIntrinsic::Memset {
                    dst: AddrSpace::Thread,
                    len: Elem::I64,
                    volatile: false,
                };
                call(
                    f,
                    void,
                    Callee::Llvm { decl: memset, op },
                    &[arr, fill, len, no],
                );
            };
            match reach {
                Reach::Fill => {
                    let fill = cast(&mut f, CastOp::Trunc, i8t);
                    memset_call(&mut f, fill, sixteen);
                }
                Reach::Length => {
                    let len = cast(&mut f, CastOp::ZExt, i64t);
                    memset_call(&mut f, zero8, len);
                }
                Reach::CopyLength => {
                    let len = cast(&mut f, CastOp::ZExt, i64t);
                    let op = LlvmIntrinsic::Memcpy {
                        dst: AddrSpace::Thread,
                        src: AddrSpace::Thread,
                        len: Elem::I64,
                        volatile: false,
                    };
                    let callee = Callee::Llvm {
                        decl: local_memcpy,
                        op,
                    };
                    call(&mut f, void, callee, &[arr, s, len, no]);
                }
                _ => {
                    let at = cast(&mut f, CastOp::ZExt, i64t);
                    let p = gep(&mut f, bytes, arr, &[zero, at]);
                    store(&mut f, zero8, p);
                }
            }
            Some(load(&mut f, i32t, kept))
        }
    };
    if let Some(x) = read {
        store(&mut f, x, arg(2));
    }
    f.ret(None);
    let k = f.finish();
    let module = m.finish().unwrap();
    let builtin = |param: u32, predicate: Option<Predicate>, name: &str| ParamBinding {
        param,
        role: ParamRole::Builtin(crate::abi::Builtin::ThreadPositionInGrid),
        unused: false,
        predicate,
        ty: ArgTypeInfo::default(),
        name: Some(DiagName::new(name)),
    };
    let params = vec![
        builtin(0, None, "kept"),
        builtin(1, Some(off), "gone"),
        buffer(2, 0),
    ];
    let abi = abi(vec![kernel(k, params)], Vec::new());
    specialize(module, &abi, &request(&[]), &mut meter())
}

/// A gated-off value stored into a private field that no observed read reaches, in a struct whose
/// address sits in a pointer cell nothing reads, is not observed. Nothing observes the store or the
/// cell's either, so both go, and the present field's one store decides its read
/// (`canon::forward`): what stays stores the present value into `out`.
#[test]
fn an_absent_value_stored_into_private_bytes_nothing_reads_is_unobserved() {
    let e = private_store(Reach::Kept).unwrap();
    let text = body_text(&e);
    assert!(!text.contains("undef"), "{text}");
    assert!(!text.contains("%a1"), "{text}");
    assert_eq!(text.matches("store ").count(), 1, "{text}");
}

/// Every read that reaches the gated-off field's bytes observes the store: the field itself, an
/// aggregate load covering it, one byte of it, the field through the pointer loaded back from the
/// cell, a copy of the struct, and any read once the struct's address escapes. The observed value
/// reads zero.
#[test]
fn a_read_that_reaches_an_absent_value_in_private_memory_observes_it() {
    for reach in [
        Reach::Gone,
        Reach::Whole,
        Reach::Byte,
        Reach::Cell,
        Reach::Copy,
        Reach::Escape,
    ] {
        let e = private_store(reach).unwrap_or_else(|e| panic!("{reach:?}: {e}"));
        let text = body_text(&e);
        assert!(
            text.contains("store i32 zeroinitializer"),
            "{reach:?}: {text}"
        );
        assert!(!text.contains("undef"), "{reach:?}: {text}");
        assert!(!text.contains("%a1"), "{reach:?}: {text}");
    }
}

/// Only the bytes a private write writes wait for a read. A `memset` whose fill byte is the
/// gated-off value, into memory nothing reads, does not observe it.
#[test]
fn an_absent_fill_byte_nothing_reads_is_unobserved() {
    let e = private_store(Reach::Fill).unwrap();
    let text = body_text(&e);
    assert!(!text.contains("%a1"), "{text}");
}

/// The length of a private `memset` or `memcpy` decides how much it writes, so a gated-off length
/// is observed (it reads zero, never `undef`). Nothing reads the written memory, so the write goes,
/// and its length with it.
#[test]
fn an_absent_length_of_a_private_write_nothing_reads_goes_with_it() {
    for reach in [Reach::Length, Reach::CopyLength] {
        let e = private_store(reach).unwrap_or_else(|e| panic!("{reach:?}: {e}"));
        let text = body_text(&e);
        assert!(!text.contains("llvm.mem"), "{reach:?}: {text}");
        assert!(!text.contains("zext"), "{reach:?}: {text}");
        assert!(!text.contains("undef"), "{reach:?}: {text}");
        assert!(!text.contains("%a1"), "{reach:?}: {text}");
    }
}

/// The index that places a private store decides where it writes, so a gated-off index is observed
/// (it reads zero). Nothing reads the written memory, so the store goes, and the address its index
/// places with it. The present field's one store decides its read (`canon::forward`), so only the
/// store into `out` stays.
#[test]
fn an_absent_index_of_a_private_write_nothing_reads_goes_with_it() {
    let e = private_store(Reach::Index).unwrap();
    let text = body_text(&e);
    assert!(!text.contains("zext"), "{text}");
    assert!(!text.contains("undef"), "{text}");
    assert!(!text.contains("%a1"), "{text}");
    assert_eq!(text.matches("store ").count(), 1, "{text}");
}

/// The work of a kernel whose private `uint arr[n]` is covered by one `memset`, then written
/// element by element and read back element by element into `out`, beside a gated-off argument.
fn covered_array_work(n: u64) -> u64 {
    let mut m = ModuleBuilder::new();
    let void = m.void();
    let i1 = m.int(1);
    let i8t = m.int(8);
    let i32t = m.int(32);
    let i64t = m.int(64);
    let dev = m.ptr(AddrSpace::Device);
    let thread = m.ptr(AddrSpace::Thread);
    let arr_ty = m.array(i32t, n);
    let off = predicate(&mut m, "off", 0);
    let memset_ty = m.fn_ty(void, &[thread, i8t, i64t, i1]);
    let memset = m.declare("llvm.memset.p0.i64", memset_ty);
    let fty = m.fn_ty(void, &[i32t, i32t, dev]);
    let one = m.const_int(i32t, 1);
    let zero = m.const_int(i64t, 0);
    let len = m.const_int(i64t, 4 * n);
    let zero8 = m.const_int(i8t, 0);
    let no = m.const_int(i1, 0);
    let index: Vec<Value> = (0..n).map(|i| m.const_int(i64t, i)).collect();
    let mut f = m.define("k", fty);
    let arr = f.push(
        thread,
        InstKind::Alloca {
            ty: arr_ty,
            count: one,
            align: None,
        },
    );
    let op = LlvmIntrinsic::Memset {
        dst: AddrSpace::Thread,
        len: Elem::I64,
        volatile: false,
    };
    call(
        &mut f,
        void,
        Callee::Llvm { decl: memset, op },
        &[arr, zero8, len, no],
    );
    let elem = |f: &mut FunctionBuilder<'_>, i: Value| {
        f.push(
            thread,
            InstKind::Gep {
                src_ty: arr_ty,
                base: arr,
                indices: [zero, i].into(),
                inbounds: true,
            },
        )
    };
    for &i in &index {
        let p = elem(&mut f, i);
        store(&mut f, arg(0), p);
    }
    for &i in &index {
        let p = elem(&mut f, i);
        let x = load(&mut f, i32t, p);
        store(&mut f, x, arg(2));
    }
    // The gated-off argument goes into memory nothing reads.
    let unread = f.push(
        thread,
        InstKind::Alloca {
            ty: i32t,
            count: one,
            align: None,
        },
    );
    store(&mut f, arg(1), unread);
    f.ret(None);
    let k = f.finish();
    let module = m.finish().unwrap();
    let builtin = |param: u32, predicate: Option<Predicate>, name: &str| ParamBinding {
        param,
        role: ParamRole::Builtin(crate::abi::Builtin::ThreadPositionInGrid),
        unused: false,
        predicate,
        ty: ArgTypeInfo::default(),
        name: Some(DiagName::new(name)),
    };
    let params = vec![
        builtin(0, None, "kept"),
        builtin(1, Some(off), "gone"),
        buffer(2, 0),
    ];
    let abi = abi(vec![kernel(k, params)], Vec::new());
    let mut meter = meter();
    specialize(module, &abi, &request(&[]), &mut meter).unwrap();
    meter.work()
}

/// A read visits only the writes it overlaps, each once overall. With a `memset` covering every
/// element, a scan from the covering write's start would visit about `n²/2` writes; the work of the
/// whole specialization instead grows with `n` (up to the tree's logarithm): four times the
/// elements cost well under six times the work.
#[test]
fn a_read_of_private_memory_visits_only_the_writes_it_overlaps() {
    let (small, large) = (covered_array_work(512), covered_array_work(2048));
    assert!(large < 6 * small, "{small} -> {large}");
}

/// `sincos` stores its cosine through a pointer, so a call whose result nothing reads still writes
/// memory: `kernel k(float x [[off]], device float* out)` stores `cos(x)`, written by `sincos` into
/// a private float, to `out`, which observes `x`: it reads zero, not `undef`.
#[test]
fn a_call_that_writes_through_a_pointer_observes_its_operands() {
    let mut m = ModuleBuilder::new();
    let void = m.void();
    let f32t = m.float();
    let i32t = m.int(32);
    let dev = m.ptr(AddrSpace::Device);
    let thread = m.ptr(AddrSpace::Thread);
    let off = predicate(&mut m, "off", 0);
    let sincos_ty = m.fn_ty(f32t, &[f32t, thread]);
    let sincos = m.declare("air.fast_sincos.f32", sincos_ty);
    let fty = m.fn_ty(void, &[f32t, dev]);
    let one = m.const_int(i32t, 1);
    let mut f = m.define("k", fty);
    let c = f.push(
        thread,
        InstKind::Alloca {
            ty: f32t,
            count: one,
            align: None,
        },
    );
    let op = AirIntrinsic::Math(Math {
        f: MathFn::SinCos,
        precision: Precision::Fast,
        ty: NumTy {
            elem: Elem::F32,
            lanes: 1,
        },
    });
    call(&mut f, f32t, Callee::Air { decl: sincos, op }, &[arg(0), c]);
    let cos = load(&mut f, f32t, c);
    store(&mut f, cos, arg(1));
    f.ret(None);
    let k = f.finish();
    let module = m.finish().unwrap();
    let x = ParamBinding {
        param: 0,
        role: ParamRole::Builtin(crate::abi::Builtin::ThreadPositionInGrid),
        unused: false,
        predicate: Some(off),
        ty: ArgTypeInfo::default(),
        name: Some(DiagName::new("x")),
    };
    let abi = abi(vec![kernel(k, vec![x, buffer(1, 0)])], Vec::new());
    let e = specialize(module, &abi, &request(&[]), &mut meter()).unwrap();
    let text = body_text(&e);
    assert!(text.contains("float zeroinitializer"), "{text}");
    assert!(!text.contains("undef") && !text.contains("%a0"), "{text}");
}
