//! `spirv-val` sweep, continued: memory, calls, atomics, barriers.

use super::families::tys;
use super::{assert_spirv_valid, compute, cval, elem_ptr, finish_compute, gid_x, scalar, ssbo};
use crate::spirv::builder::ModuleBuilder;
use crate::spirv::ir::{
    AtomicOp, BinOp, MemAccess, MemoryOrder, Scope, Semantics, StorageClass, Type,
};

#[test]
fn memory_and_calls() {
    let mut b = ModuleBuilder::new();
    let t = tys(&mut b);
    let buf = ssbo(&mut b, 0, t.u32_, 4);
    let zero = scalar(&mut b, t.u32_, 0);
    let private = b
        .plain_global(StorageClass::Private, t.u32_, Some(zero))
        .expect("private");
    let shared = b
        .plain_global(StorageClass::Workgroup, t.vec4, None)
        .expect("workgroup");
    let st = b
        .ty(Type::Struct {
            members: [t.u32_, t.vec4].into(),
            layout: None,
        })
        .expect("struct");
    // A callee taking a value and a `Function` pointer, returning a value.
    let fptr = b
        .ty(Type::Pointer {
            class: StorageClass::Function,
            pointee: t.u32_,
        })
        .expect("fn ptr");
    let callee = {
        let mut f = b.function(t.u32_).expect("callee");
        let x = f.param(t.u32_).expect("x");
        let p = f.param(fptr).expect("p");
        let y = f.load(p, MemAccess::default()).expect("load");
        let s = f.binary(BinOp::IAdd, x, y).expect("add");
        f.store(p, s, MemAccess::default()).expect("store");
        f.ret(Some(s)).expect("ret");
        f.finish().expect("finish")
    };
    // A void callee (its call has a void result id in SPIR-V).
    let void_callee = {
        let mut f = b.function(t.void).expect("void callee");
        f.ret(None).expect("ret");
        f.finish().expect("finish")
    };
    let main = compute(&mut b, |f| {
        let i = gid_x(f);
        let src = elem_ptr(f, buf, i);
        let x = f
            .load(
                src,
                MemAccess {
                    volatile: true,
                    nontemporal: true,
                    ..MemAccess::default()
                },
            )
            .expect("load");
        let var = f.local_var(t.u32_, Some(zero)).expect("var");
        let vp = f.var_ptr(var).expect("ptr");
        let r = f.call(callee, &[x, vp]).expect("call").expect("value");
        assert_eq!(f.call(void_callee, &[]).expect("void call"), None);
        let svar = f.local_var(st, None).expect("struct var");
        let sp = f.var_ptr(svar).expect("ptr");
        let one = cval(f, t.u32_, 1);
        let zero_v = cval(f, t.u32_, 0);
        let member = f.access_chain(sp, &[zero_v]).expect("member");
        f.store(member, r, MemAccess::default())
            .expect("store member");
        let comp = f.access_chain(sp, &[one, i]).expect("dynamic component");
        f.load(comp, MemAccess::default()).expect("load comp");
        let pp = f.module().global_value(private).expect("private");
        f.store(pp, r, MemAccess::default()).expect("private store");
        let wp = f.module().global_value(shared).expect("shared");
        let v4 = f.undef(t.vec4).expect("undef");
        f.store(wp, v4, MemAccess::default()).expect("shared store");
        let base = f.module().global_value(buf).expect("buf");
        let len = f.array_length(base, 0).expect("array length");
        f.store(src, len, MemAccess::default()).expect("store len");
    });
    assert_spirv_valid(finish_compute(b, main));
}

fn sem(order: MemoryOrder, uniform: bool, workgroup: bool) -> Semantics {
    Semantics {
        order,
        uniform_memory: uniform,
        workgroup_memory: workgroup,
        ..Semantics::RELAXED
    }
}

#[test]
fn integer_atomics_and_barriers() {
    let mut b = ModuleBuilder::new();
    let t = tys(&mut b);
    let bu = ssbo(&mut b, 0, t.u32_, 4);
    let bi = ssbo(&mut b, 1, t.i32_, 4);
    let b64 = ssbo(&mut b, 2, t.u64_, 8);
    let wg = b
        .plain_global(StorageClass::Workgroup, t.u32_, None)
        .expect("workgroup counter");
    let main = compute(&mut b, |f| {
        let i = gid_x(f);
        let pu = elem_ptr(f, bu, i);
        let pi = elem_ptr(f, bi, i);
        let p64 = elem_ptr(f, b64, i);
        let pw = f.module().global_value(wg).expect("wg");
        let one = cval(f, t.u32_, 1);
        let ione = cval(f, t.i32_, 1);
        let one64 = cval(f, t.u64_, 1);
        let relaxed = Semantics::RELAXED;
        for (p, v, scope, uniform, workgroup) in [
            (pu, one, Scope::Device, true, false),
            (pi, ione, Scope::Device, true, false),
            (p64, one64, Scope::Device, true, false),
            (pw, one, Scope::Workgroup, false, true),
        ] {
            let acq = sem(MemoryOrder::Acquire, uniform, workgroup);
            let rel = sem(MemoryOrder::Release, uniform, workgroup);
            let acq_rel = sem(MemoryOrder::AcquireRelease, uniform, workgroup);
            f.atomic_load(p, scope, relaxed).expect("relaxed load");
            f.atomic_load(p, scope, acq).expect("acquire load");
            f.atomic_store(p, scope, relaxed, v).expect("relaxed store");
            f.atomic_store(p, scope, rel, v).expect("release store");
            for op in [
                AtomicOp::Exchange,
                AtomicOp::IAdd,
                AtomicOp::ISub,
                AtomicOp::SMin,
                AtomicOp::UMin,
                AtomicOp::SMax,
                AtomicOp::UMax,
                AtomicOp::And,
                AtomicOp::Or,
                AtomicOp::Xor,
            ] {
                for s in [relaxed, acq_rel] {
                    f.atomic(op, p, scope, s, v)
                        .unwrap_or_else(|e| panic!("{op:?}: {e}"));
                }
            }
            f.atomic_compare_exchange(p, scope, relaxed, relaxed, v, v)
                .expect("relaxed compare exchange");
            f.atomic_compare_exchange(p, scope, acq_rel, acq, v, v)
                .expect("compare exchange");
        }
        f.control_barrier(
            Scope::Workgroup,
            Scope::Workgroup,
            sem(MemoryOrder::AcquireRelease, false, true),
        )
        .expect("workgroup barrier");
        f.control_barrier(
            Scope::Workgroup,
            Scope::Device,
            sem(MemoryOrder::AcquireRelease, true, false),
        )
        .expect("device memory barrier");
        f.control_barrier(Scope::Subgroup, Scope::Subgroup, relaxed)
            .expect("subgroup barrier");
        f.memory_barrier(Scope::Device, sem(MemoryOrder::AcquireRelease, true, false))
            .expect("memory barrier");
        f.memory_barrier(Scope::Workgroup, sem(MemoryOrder::Release, false, true))
            .expect("workgroup memory barrier");
    });
    assert_spirv_valid(finish_compute(b, main));
}

#[test]
fn float_atomics() {
    let mut b = ModuleBuilder::new();
    let t = tys(&mut b);
    let bf = ssbo(&mut b, 0, t.f32_, 4);
    let wg = b
        .plain_global(StorageClass::Workgroup, t.f32_, None)
        .expect("workgroup");
    let main = compute(&mut b, |f| {
        let i = gid_x(f);
        let p = elem_ptr(f, bf, i);
        let pw = f.module().global_value(wg).expect("wg");
        let x = cval(f, t.f32_, 0x3f80_0000);
        let r = Semantics::RELAXED;
        for (p, scope) in [(p, Scope::Device), (pw, Scope::Workgroup)] {
            f.atomic_load(p, scope, r).expect("load");
            f.atomic_store(p, scope, r, x).expect("store");
            f.atomic(AtomicOp::Exchange, p, scope, r, x)
                .expect("exchange");
        }
    });
    assert_spirv_valid(finish_compute(b, main));
}
