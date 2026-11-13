//! Thread-memory forwarding (`canon::forward`): a load of a stack allocation that one dominating
//! store wrote reads the stored value, and an allocation no read is left of goes with its stores.
//! Each kernel reads two per-thread values `a` and `b` and writes what it computes to `out`; a
//! load the pass must keep shows as a `load` in the specialized body.

use super::*;
use crate::intrinsics::{Elem, LlvmIntrinsic};

/// How [`kernel_of`]'s kernel writes and reads its private `int x`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Shape {
    /// A closure `{ int* }` holds `&x`; the read goes through the pointer loaded back from it,
    /// as a lambda capturing `x` by reference reads it once inlined.
    Closure,
    /// `x = a`, then `x = b` under a branch on `a`, then the read after the branches join.
    TwoStores,
    /// `x = a` under a branch on `a` only, then the read after the branches join.
    NotDominating,
    /// `x = a`, then `&x` also stored to `out` as an integer, then the read.
    Escaped,
    /// `x` is a `long` written whole, then read as the `int` at its offset 0.
    Part,
    /// `x = a`, then `memcpy(&x, &y, 4)` from a private `int y = b`, then the read.
    Copied,
    /// The read, then `x = a`, in the one block.
    ReadFirst,
    /// `x` is a `long` holding `out`'s bits, read back and turned back into a pointer, which `a` is
    /// stored through; the kernel's value is `b`.
    Bits,
    /// `x` is a `{ int*, int }` stored whole as `{ out, a }`, then read whole and its `int` taken.
    Pair,
    /// `x` is a `thread int*` set to null, read back and compared with null.
    NullSlot,
}

/// `kernel k(int a, int b, device int* out)` that stores and reads its private `x` as `shape`
/// says, and stores what it read to `out`.
fn kernel_of(shape: Shape) -> Entry {
    let mut m = ModuleBuilder::new();
    let void = m.void();
    let i1 = m.int(1);
    let i32t = m.int(32);
    let i64t = m.int(64);
    let dev = m.ptr(AddrSpace::Device);
    let thread = m.ptr(AddrSpace::Thread);
    let closure_ty = m.struct_ty(&[thread]);
    let pair = m.struct_ty(&[dev, i32t]);
    let fty = m.fn_ty(void, &[i32t, i32t, dev]);
    let memcpy_ty = m.fn_ty(void, &[thread, thread, i64t, i1]);
    let memcpy = m.declare("llvm.memcpy.p0.p0.i64", memcpy_ty);
    let one = m.const_int(i32t, 1);
    let zero32 = m.const_int(i32t, 0);
    let zero = m.const_int(i64t, 0);
    let four = m.const_int(i64t, 4);
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
    let wide = matches!(shape, Shape::Part | Shape::Bits);
    let x = match shape {
        Shape::Pair => alloca(&mut f, pair),
        Shape::NullSlot => alloca(&mut f, thread),
        _ if wide => alloca(&mut f, i64t),
        _ => alloca(&mut f, i32t),
    };
    let read = match shape {
        Shape::Closure => {
            store(&mut f, arg(0), x);
            let cl = alloca(&mut f, closure_ty);
            let field = f.push(
                thread,
                InstKind::Gep {
                    src_ty: closure_ty,
                    base: cl,
                    indices: [zero, zero32].into(),
                    inbounds: true,
                },
            );
            store(&mut f, x, field);
            let p = load(&mut f, thread, field);
            load(&mut f, i32t, p)
        }
        Shape::TwoStores | Shape::NotDominating => {
            if shape == Shape::TwoStores {
                store(&mut f, arg(0), x);
            }
            let c = f.push(
                i1,
                InstKind::ICmp {
                    pred: IntPredicate::Eq,
                    a: arg(0),
                    b: zero32,
                },
            );
            let (t, join) = (f.block(), f.block());
            f.cond_br(c, t, join);
            f.switch_to(t);
            let v = if shape == Shape::TwoStores {
                arg(1)
            } else {
                arg(0)
            };
            store(&mut f, v, x);
            f.br(join);
            f.switch_to(join);
            load(&mut f, i32t, x)
        }
        Shape::Escaped => {
            store(&mut f, arg(0), x);
            let a = f.push(
                i64t,
                InstKind::Cast {
                    op: CastOp::PtrToInt,
                    v: x,
                },
            );
            store(&mut f, a, arg(2));
            load(&mut f, i32t, x)
        }
        Shape::Part => {
            let wide = f.push(
                i64t,
                InstKind::Cast {
                    op: CastOp::ZExt,
                    v: arg(0),
                },
            );
            store(&mut f, wide, x);
            load(&mut f, i32t, x)
        }
        Shape::Copied => {
            store(&mut f, arg(0), x);
            let y = alloca(&mut f, i32t);
            store(&mut f, arg(1), y);
            let op = LlvmIntrinsic::Memcpy {
                dst: AddrSpace::Thread,
                src: AddrSpace::Thread,
                len: Elem::I64,
                volatile: false,
            };
            call(
                &mut f,
                void,
                Callee::Llvm { decl: memcpy, op },
                &[x, y, four, no],
            );
            load(&mut f, i32t, x)
        }
        Shape::ReadFirst => {
            let v = load(&mut f, i32t, x);
            store(&mut f, arg(0), x);
            v
        }
        Shape::Bits => {
            let bits = f.push(
                i64t,
                InstKind::Cast {
                    op: CastOp::PtrToInt,
                    v: arg(2),
                },
            );
            store(&mut f, bits, x);
            let w = load(&mut f, i64t, x);
            let q = f.push(
                dev,
                InstKind::Cast {
                    op: CastOp::IntToPtr,
                    v: w,
                },
            );
            store(&mut f, arg(0), q);
            arg(1)
        }
        Shape::Pair => {
            let poison = Value::Const(f.module().constant(pair, Const::Poison));
            let mut agg = poison;
            for (val, k) in [(arg(2), 0), (arg(0), 1)] {
                agg = f.push(
                    pair,
                    InstKind::InsertValue {
                        agg,
                        val,
                        indices: Box::new([k]),
                    },
                );
            }
            store(&mut f, agg, x);
            let whole = load(&mut f, pair, x);
            f.push(
                i32t,
                InstKind::ExtractValue {
                    agg: whole,
                    indices: Box::new([1]),
                },
            )
        }
        Shape::NullSlot => {
            let null = Value::Const(f.module().constant(thread, Const::Null));
            store(&mut f, null, x);
            let p = load(&mut f, thread, x);
            let c = f.push(
                i1,
                InstKind::ICmp {
                    pred: IntPredicate::Eq,
                    a: p,
                    b: null,
                },
            );
            f.push(
                i32t,
                InstKind::Cast {
                    op: CastOp::ZExt,
                    v: c,
                },
            )
        }
    };
    store(&mut f, read, arg(2));
    f.ret(None);
    let k = f.finish();
    let module = m.finish().unwrap();
    let builtin = |param: u32, name: &str| ParamBinding {
        param,
        role: ParamRole::Builtin(crate::abi::Builtin::ThreadPositionInGrid),
        unused: false,
        predicate: None,
        ty: ArgTypeInfo::default(),
        name: Some(DiagName::new(name)),
    };
    let params = vec![builtin(0, "a"), builtin(1, "b"), buffer(2, 0)];
    let abi = abi(vec![kernel(k, params)], Vec::new());
    specialize(module, &abi, &request(&[]), &mut meter()).unwrap()
}

/// A lambda's closure holding `&x`, read back through the pointer it holds, is read as `x`'s own
/// address, and `x`'s one store as its value: no allocation, load or private store is left, and
/// `out` gets `a` itself.
#[test]
fn a_closure_of_captured_addresses_reads_as_the_captured_values() {
    let text = body_text(&kernel_of(Shape::Closure));
    assert!(!text.contains("alloca"), "{text}");
    assert!(!text.contains("load"), "{text}");
    assert_eq!(text.matches("store ").count(), 1, "{text}");
    assert!(text.contains("store %a0,"), "{text}");
}

/// A pointer's bits stored, read back and turned back into a pointer of its type are the pointer
/// itself: `a` is stored to `out` directly, with no allocation or `inttoptr` left.
#[test]
fn a_pointer_s_bits_read_back_as_a_pointer_are_the_pointer() {
    let text = body_text(&kernel_of(Shape::Bits));
    assert!(!text.contains("alloca"), "{text}");
    assert!(!text.contains("inttoptr"), "{text}");
    assert!(text.contains("store %a0, %a2,"), "{text}");
    assert!(text.contains("store %a1, %a2,"), "{text}");
}

/// A load of an aggregate that holds a pointer, and a load of a thread pointer one store set to
/// null, stay, and their allocations with them: lowering reads such an aggregate's members through
/// the chain that built it or through memory, and holds a thread pointer's null only in memory.
#[test]
fn a_load_of_an_aggregate_holding_a_pointer_or_of_a_thread_null_stays() {
    for shape in [Shape::Pair, Shape::NullSlot] {
        let text = body_text(&kernel_of(shape));
        assert!(text.contains("alloca"), "{shape:?}: {text}");
    }
}

/// A load the rule does not cover stays a load of its allocation: a field two stores write, a
/// store that runs on one path only, an address that escapes, a part of a wider store, a field a
/// copy writes too, and a read before the store.
#[test]
fn a_load_one_dominating_store_does_not_decide_stays() {
    for shape in [
        Shape::TwoStores,
        Shape::NotDominating,
        Shape::Escaped,
        Shape::Part,
        Shape::Copied,
        Shape::ReadFirst,
    ] {
        let text = body_text(&kernel_of(shape));
        assert!(text.contains("alloca"), "{shape:?}: {text}");
        assert!(text.contains("load i32"), "{shape:?}: {text}");
    }
}
