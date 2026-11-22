//! The pass's domain: a marked variable written by parts is left as it is.

use super::*;
use crate::budget::Budget;
use crate::spirv::builder::constructs::{ArmEnd, LoopSpec};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ir::{BinOp, ExecMode, ExecModel, Type};

fn meter() -> Meter {
    Meter::new(&Budget::default())
}

fn c32(f: &mut FunctionBuilder<'_>, v: u32) -> Value {
    let c = f.u32_const(v).expect("constant");
    f.module().const_value(c).expect("value")
}

/// A loop of three trips over `v`, a `vec2<u32>` `Function` variable marked trip-local: each trip
/// reads `v` whole and adds `v.x` to `acc` (the value the last trip left), stores `v = (0, 0)` on
/// the first trip alone, then writes `i + 1` into `v.x`: through an access chain where
/// `by_parts`, else as the whole `v = (i + 1, 0)`. The body's first block holds only the trip
/// test, so `v` is live into it through the next trip's read, and every whole store of `v` is in
/// the loop.
fn module(by_parts: bool) -> TypedModule {
    let mut b = ModuleBuilder::new();
    let void = b.ty(Type::Void).expect("void");
    let mut f = b.function(void).expect("function");
    let body = |f: &mut FunctionBuilder<'_>| -> BResult<()> {
        let u32_ = f.module().ty(Type::Int {
            width: 32,
            signed: false,
        })?;
        let v2 = f.module().ty(Type::Vector { elem: u32_, len: 2 })?;
        let z = f.u32_const(0)?;
        let i = f.local_var(u32_, Some(z))?;
        let acc = f.local_var(u32_, Some(z))?;
        let v = f.local_var(v2, None)?;
        f.mark_trip_local(v)?;
        let (ip, ap, vp) = (f.var_ptr(i)?, f.var_ptr(acc)?, f.var_ptr(v)?);
        f.loop_(
            &mut (),
            LoopSpec::default(),
            Box::new(move |f, _, lc| {
                let t = f.load(ip, MemAccess::default())?;
                let three = c32(f, 3);
                let done = f.binary(BinOp::UGreaterThanEqual, t, three)?;
                f.if_(
                    &mut (),
                    done,
                    Box::new(|f, _| {
                        lc.break_(f)?;
                        Ok(ArmEnd::Exits)
                    }),
                    None,
                )?;
                let w = f.load(vp, MemAccess::default())?;
                let x = f.composite_extract(w, &[0])?;
                let a = f.load(ap, MemAccess::default())?;
                let a1 = f.binary(BinOp::IAdd, a, x)?;
                f.store(ap, a1, MemAccess::default())?;
                let zero = c32(f, 0);
                let first = f.binary(BinOp::IEqual, t, zero)?;
                f.if_(
                    &mut (),
                    first,
                    Box::new(move |f, _| {
                        let zeros = f.composite_construct(v2, &[zero, zero])?;
                        f.store(vp, zeros, MemAccess::default())?;
                        Ok(ArmEnd::FallThrough(Vec::new()))
                    }),
                    None,
                )?;
                let one = c32(f, 1);
                let t1 = f.binary(BinOp::IAdd, t, one)?;
                if by_parts {
                    let px = f.access_chain(vp, &[zero])?;
                    f.store(px, t1, MemAccess::default())?;
                } else {
                    let w1 = f.composite_construct(v2, &[t1, zero])?;
                    f.store(vp, w1, MemAccess::default())?;
                }
                f.store(ip, t1, MemAccess::default())
            }),
        )?;
        f.ret(None)
    };
    body(&mut f).expect("body");
    let main = f.finish().expect("finish");
    b.entry(
        ExecModel::GLCompute,
        main,
        vec![ExecMode::LocalSize { x: 1, y: 1, z: 1 }],
    )
    .expect("entry");
    b.finalize().expect("finalize")
}

/// A marked variable a trip writes by parts, which the next trip reads, is never ended: a store
/// of an undefined value at the next trip's top would wipe the part. The same loop with the
/// variable written whole takes the store (its variable breaks the mark's promise, which only
/// shows that the access chain alone keeps the first one out).
#[test]
fn a_variable_written_by_parts_is_left_alone() {
    let mut m = module(true);
    let written = end_dead_carries(&mut m, &mut meter()).expect("pass");
    assert_eq!(
        written, 0,
        "a variable written through an access chain is ended"
    );
    assert_eq!(dead_carries(&m), 0);
    let mut whole = module(false);
    let written = end_dead_carries(&mut whole, &mut meter()).expect("pass");
    assert_eq!(
        written, 1,
        "the loop's variable written whole takes its store"
    );
}
