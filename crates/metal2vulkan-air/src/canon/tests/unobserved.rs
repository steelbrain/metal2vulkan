//! What nothing observes after specialization goes before lowering: a value no side effect, branch,
//! or return depends on, and a write into a stack allocation no load reaches. A side effect stays
//! although nothing uses its result.

use super::*;
use crate::abi::TextureArg;
use crate::ir::{Atomic, AtomicRmwOp, Ordering, SyncScope};

/// `kernel k(device i32* out [[buffer(0)]], uint tid, texture2d<float, write> t [[texture(0)]])`,
/// its body built by `body` (before the `ret`).
fn entry(body: impl FnOnce(&mut FunctionBuilder<'_>)) -> Entry {
    let mut m = ModuleBuilder::new();
    let void = m.void();
    let i32t = m.int(32);
    let dev = m.ptr(AddrSpace::Device);
    let fty = m.fn_ty(void, &[dev, i32t, dev]);
    let mut f = m.define("k", fty);
    body(&mut f);
    f.ret(None);
    let k = f.finish();
    let module = m.finish().unwrap();
    let tid = ParamBinding {
        param: 1,
        role: ParamRole::Builtin(crate::abi::Builtin::ThreadPositionInGrid),
        unused: false,
        predicate: None,
        ty: ArgTypeInfo::default(),
        name: Some(DiagName::new("tid")),
    };
    let texture = ParamBinding {
        role: ParamRole::Texture(TextureArg {
            location: Slot {
                index: AbiInt::Const(0),
                count: AbiInt::Const(1),
            },
            access: Access::Write,
            raster_order_group: None,
        }),
        ..buffer(2, 0)
    };
    let abi = abi(
        vec![kernel(k, vec![buffer(0, 0), tid, texture])],
        Vec::new(),
    );
    specialize(module, &abi, &request(&[]), &mut meter()).unwrap()
}

/// A call of the `air.*` intrinsic `name` of type `ret(params)`.
fn air_call(
    f: &mut FunctionBuilder<'_>,
    name: &str,
    ret: TypeId,
    params: &[TypeId],
    args: &[Value],
) -> Value {
    let ty = f.module().fn_ty(ret, params);
    let decl = f.module().declare(name, ty);
    let op = AirIntrinsic::parse(name).unwrap_or_else(|| panic!("{name}"));
    call(f, ret, Callee::Air { decl, op }, args)
}

/// Arithmetic, an element address, and a plain load whose results nothing observes go, with what
/// only they read; the store that observes `tid` stays.
#[test]
fn values_nothing_observes_go() {
    let e = entry(|f| {
        let i32t = f.module().int(32);
        let i64t = f.module().int(64);
        let dev = f.module().ptr(AddrSpace::Device);
        let one = f.module().const_int(i32t, 1);
        let two = f.module().const_int(i64t, 2);
        let sum = f.bin(BinOp::Add, arg(1), one);
        f.bin(BinOp::Mul, sum, sum);
        let x = load(f, i32t, arg(0));
        f.bin(BinOp::Xor, x, one);
        f.push(
            dev,
            InstKind::Gep {
                src_ty: i32t,
                base: arg(0),
                indices: [two].into(),
                inbounds: true,
            },
        );
        store(f, arg(1), arg(0));
    });
    let text = body_text(&e);
    for gone in [
        " = add ",
        " = mul ",
        " = xor ",
        " = load ",
        " = getelementptr",
    ] {
        assert!(!text.contains(gone), "{gone}: {text}");
    }
    assert_eq!(text.matches("store ").count(), 1, "{text}");
}

/// A side effect stays although nothing uses its result, and so does every value it reads: a store
/// into device memory, a store into threadgroup memory, an `atomicrmw` and an AIR atomic whose old
/// values nothing reads, a volatile load nothing reads, a barrier, and a texture write.
#[test]
fn a_side_effect_nothing_uses_stays() {
    let e = entry(|f| {
        let void = f.module().void();
        let i1 = f.module().int(1);
        let i32t = f.module().int(32);
        let float = f.module().float();
        let dev = f.module().ptr(AddrSpace::Device);
        let coord_ty = f.module().vector(i32t, 2);
        let color_ty = f.module().vector(float, 4);
        let g = f
            .module()
            .simple_global("g", i32t, AddrSpace::Threadgroup, None);
        let g = f.module().global_addr(g);
        let (zero, one, two) = (
            f.module().const_int(i32t, 0),
            f.module().const_int(i32t, 1),
            f.module().const_int(i32t, 2),
        );
        let no = f.module().const_int(i1, 0);
        let coord = f.module().constant(coord_ty, Const::Null);
        let color = f.module().constant(color_ty, Const::Null);
        let sum = f.bin(BinOp::Add, arg(1), one);
        store(f, sum, arg(0));
        let twice = f.bin(BinOp::Shl, arg(1), one);
        store(f, twice, g);
        let third = f.bin(BinOp::Mul, arg(1), two);
        f.push(
            i32t,
            InstKind::AtomicRmw {
                op: AtomicRmwOp::Add,
                ptr: arg(0),
                val: third,
                atomic: Atomic {
                    ordering: Ordering::Monotonic,
                    scope: SyncScope::System,
                },
                volatile: false,
                align: Align::from_bytes(4),
            },
        );
        let fourth = f.bin(BinOp::Or, arg(1), two);
        air_call(
            f,
            "air.atomic.global.add.u.i32",
            i32t,
            &[dev, i32t, i32t, i32t, i1],
            &[arg(0), fourth, zero, two, no],
        );
        f.push(
            i32t,
            InstKind::Load {
                ptr: arg(0),
                align: Align::from_bytes(4),
                volatile: true,
                atomic: None,
                range: None,
            },
        );
        air_call(f, "air.wg.barrier", void, &[i32t, i32t], &[two, one]);
        air_call(
            f,
            "air.write_texture_2d.v4f32",
            void,
            &[dev, coord_ty, color_ty, i32t, i32t],
            &[arg(2), Value::Const(coord), Value::Const(color), zero, two],
        );
    });
    let text = body_text(&e);
    for kept in [
        " = add ",
        " = shl ",
        " = mul ",
        " = or ",
        "atomicrmw",
        "[air Atomic(",
        "load volatile",
        "[air Sync(Barrier",
        "[air Texture(Texture { op: Write",
    ] {
        assert!(text.contains(kept), "{kept}: {text}");
    }
    assert_eq!(text.matches("store ").count(), 2, "{text}");
    assert!(text.contains("addrspace(3) @g"), "{text}");
}

/// A write into a stack allocation stays once the allocation escapes; one nothing reads goes, and
/// so does one whose only load reads it whole at its own type, which reads the stored value
/// (`canon::forward`). `private_memory` covers each reach.
#[test]
fn a_private_write_goes_only_when_no_load_reaches_it() {
    let shaped = |read: bool, escape: bool| {
        entry(|f| {
            let i32t = f.module().int(32);
            let i64t = f.module().int(64);
            let thread = f.module().ptr(AddrSpace::Thread);
            let one = f.module().const_int(i32t, 1);
            let s = f.push(
                thread,
                InstKind::Alloca {
                    ty: i32t,
                    count: one,
                    align: None,
                },
            );
            let sum = f.bin(BinOp::Add, arg(1), one);
            store(f, sum, s);
            if read {
                let x = load(f, i32t, s);
                store(f, x, arg(0));
            }
            if escape {
                let a = f.push(
                    i64t,
                    InstKind::Cast {
                        op: CastOp::PtrToInt,
                        v: s,
                    },
                );
                let p = f.push(
                    thread,
                    InstKind::Cast {
                        op: CastOp::IntToPtr,
                        v: a,
                    },
                );
                let x = load(f, i32t, p);
                store(f, x, arg(0));
            }
        })
    };
    let alone = body_text(&shaped(false, false));
    assert!(!alone.contains(" = add "), "{alone}");
    assert_eq!(alone.matches("store ").count(), 0, "{alone}");
    let read = body_text(&shaped(true, false));
    assert!(read.contains(" = add "), "{read}");
    assert!(!read.contains("alloca"), "{read}");
    assert_eq!(read.matches("store ").count(), 1, "{read}");
    let escaped = body_text(&shaped(false, true));
    assert!(escaped.contains(" = add "), "{escaped}");
    assert_eq!(escaped.matches("store ").count(), 2, "{escaped}");
}
