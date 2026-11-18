//! Interpreter self-tests on hand-built modules: arithmetic, memory, control flow, calls, barriers
//! and the Q3 model, subgroup operations with maximal reconvergence at widths 32 and 64, both
//! scheduling orders, and every trap class.

use super::*;
use crate::pipeline::sym::{Axis, ResourceKind, ResourceSlot, SpecKey, Stage};
use crate::spirv::builder::constructs::{ArmEnd, ArmFn, LoopFn, LoopSpec};
use crate::spirv::builder::ops::group::GroupMode;
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{GlobalId, TypeId, Value};
use crate::spirv::ir::{
    natural_layout, AtomicOp, BinOp, ConstBits, Constant, ConvOp, ExecMode, ExecModel, GroupArith,
    ImageOperands, IoDir, LayoutRule, MemAccess, MemoryOrder, Scope, Semantics, ShuffleOp,
    StorageClass, TexelMemory, Type, TypedModule, UnOp,
};

mod derivative;
mod float_props;
mod harness;
mod memory;
mod structured_trace;
mod subgroup;

// ---- Fixture ----

fn u32_ty(b: &mut ModuleBuilder) -> TypeId {
    b.ty(Type::Int {
        width: 32,
        signed: false,
    })
    .expect("u32")
}

/// A `u32` constant's value.
fn c32(f: &mut FunctionBuilder<'_>, v: u32) -> Value {
    let c = f.u32_const(v).expect("constant");
    f.module().const_value(c).expect("value")
}

fn slot(index: u32) -> ResourceSlot {
    ResourceSlot {
        stage: Stage::Kernel,
        kind: ResourceKind::Buffer,
        index,
    }
}

/// A `Block { u32 data[]; }` storage buffer at buffer slot `index`.
fn buffer(b: &mut ModuleBuilder, index: u32) -> GlobalId {
    let u32_ = u32_ty(b);
    let rta = b
        .ty(Type::RuntimeArray {
            elem: u32_,
            stride: Some(4),
        })
        .expect("runtime array");
    let layout = natural_layout(b.types(), LayoutRule::Std430, true, &[rta]).expect("layout");
    let block = b
        .ty(Type::Struct {
            members: [rta].into(),
            layout: Some(layout),
        })
        .expect("block");
    b.resource(slot(index), StorageClass::StorageBuffer, block)
        .expect("resource")
}

/// Load a `u32` (or `uvec3` component 0) compute builtin.
fn builtin(f: &mut FunctionBuilder<'_>, bi: crate::spv::BuiltIn) -> Value {
    let u32_ = u32_ty(f.module());
    let vec3 = matches!(
        bi,
        crate::spv::BuiltIn::GlobalInvocationId
            | crate::spv::BuiltIn::LocalInvocationId
            | crate::spv::BuiltIn::WorkgroupId
            | crate::spv::BuiltIn::NumWorkgroups
    );
    let ty = if vec3 {
        f.module()
            .ty(Type::Vector { elem: u32_, len: 3 })
            .expect("uvec3")
    } else {
        u32_
    };
    let g = f.module().builtin(bi, IoDir::Input, ty).expect("builtin");
    let p = f.module().global_value(g).expect("global");
    let v = f.load(p, MemAccess::default()).expect("load");
    if vec3 {
        f.composite_extract(v, &[0]).expect("extract")
    } else {
        v
    }
}

fn lid(f: &mut FunctionBuilder<'_>) -> Value {
    builtin(f, crate::spv::BuiltIn::LocalInvocationIndex)
}

fn gid(f: &mut FunctionBuilder<'_>) -> Value {
    builtin(f, crate::spv::BuiltIn::GlobalInvocationId)
}

/// A pointer to `buf.data[i]`.
fn elem(f: &mut FunctionBuilder<'_>, buf: GlobalId, i: Value) -> BResult<Value> {
    let zero = c32(f, 0);
    let base = f.module().global_value(buf)?;
    f.access_chain(base, &[zero, i])
}

fn store_at(f: &mut FunctionBuilder<'_>, buf: GlobalId, i: Value, v: Value) -> BResult<()> {
    let p = elem(f, buf, i)?;
    f.store(p, v, MemAccess::default())
}

fn load_at(f: &mut FunctionBuilder<'_>, buf: GlobalId, i: Value) -> BResult<Value> {
    let p = elem(f, buf, i)?;
    f.load(p, MemAccess::default())
}

fn wg_barrier(f: &mut FunctionBuilder<'_>) -> BResult<()> {
    let sem = Semantics {
        order: MemoryOrder::AcquireRelease,
        workgroup_memory: true,
        ..Semantics::RELAXED
    };
    f.control_barrier(Scope::Workgroup, Scope::Workgroup, sem)
}

/// A compute module with local size `local`, `nbuf` `u32` storage buffers, and entry body `body`
/// (a return is added when the body leaves a block open).
fn kernel(
    local: [u32; 3],
    nbuf: u32,
    body: impl FnOnce(&mut FunctionBuilder<'_>, &[GlobalId]) -> BResult<()>,
) -> TypedModule {
    kernel_with(local, nbuf, Vec::new(), body)
}

/// [`kernel`] whose entry also declares `modes`.
fn kernel_with(
    local: [u32; 3],
    nbuf: u32,
    modes: Vec<ExecMode>,
    body: impl FnOnce(&mut FunctionBuilder<'_>, &[GlobalId]) -> BResult<()>,
) -> TypedModule {
    let mut b = ModuleBuilder::new();
    let bufs: Vec<GlobalId> = (0..nbuf).map(|i| buffer(&mut b, i)).collect();
    let void = b.ty(Type::Void).expect("void");
    let mut f = b.function(void).expect("function");
    body(&mut f, &bufs).expect("body");
    if f.current_block().is_some() {
        f.ret(None).expect("ret");
    }
    let id = f.finish().expect("finish");
    let [x, y, z] = local;
    let mut all = vec![ExecMode::LocalSize { x, y, z }];
    all.extend(modes);
    b.entry(ExecModel::GLCompute, id, all).expect("entry");
    b.finalize().expect("finalize")
}

/// Zeroed buffers of `words` `u32`s each, bound to slots 0, 1, ….
fn buffers(words: &[usize]) -> Resources {
    Resources {
        buffers: words
            .iter()
            .enumerate()
            .map(|(i, n)| Buffer {
                slot: Some(slot(i as u32)),
                address: None,
                bytes: vec![0; n * 4],
            })
            .collect(),
        push: Vec::new(),
        spec: Default::default(),
    }
}

fn words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn cfg(groups: u32, subgroup: SubgroupSize, order: Order) -> Config {
    Config {
        workgroups: [groups, 1, 1],
        subgroup,
        order,
        ..Config::default()
    }
}

/// Run in both orders; the outputs must agree (no race). Returns the forward outcome.
fn run_both(m: &TypedModule, c: Config, res: &Resources) -> Outcome {
    let fwd = run(
        m,
        &Config {
            order: Order::Forward,
            ..c
        },
        res.clone(),
    )
    .expect("forward run");
    let rev = run(
        m,
        &Config {
            order: Order::Reverse,
            ..c
        },
        res.clone(),
    )
    .expect("reverse run");
    assert_eq!(fwd, rev, "the two scheduling orders disagree");
    fwd
}

fn trap_of(r: IResult<Outcome>) -> TrapKind {
    match r {
        Err(InterpError::Trap(t)) => t.kind,
        other => panic!("expected a trap, got {other:?}"),
    }
}

/// A loop running `body` `n` times with counter `k` in a `Function` variable.
fn counted<'a, C: 'a>(
    f: &mut FunctionBuilder<'_>,
    ctx: &mut C,
    n: Value,
    body: impl FnOnce(&mut FunctionBuilder<'_>, &mut C, Value) -> BResult<()> + 'a,
) -> BResult<()> {
    let u32_ = u32_ty(f.module());
    let var = f.local_var(u32_, None)?;
    let counter = f.var_ptr(var)?;
    let zero = c32(f, 0);
    f.store(counter, zero, MemAccess::default())?;
    let body: LoopFn<'a, C> = Box::new(move |fb, ctx, lc| {
        let k = fb.load(counter, MemAccess::default())?;
        let done = fb.binary(BinOp::UGreaterThanEqual, k, n)?;
        let brk: ArmFn<'_, C> = Box::new(|fb, _| {
            lc.break_(fb)?;
            Ok(ArmEnd::Exits)
        });
        fb.if_(ctx, done, brk, None)?;
        let one = c32(fb, 1);
        let k1 = fb.binary(BinOp::IAdd, k, one)?;
        fb.store(counter, k1, MemAccess::default())?;
        body(fb, ctx, k)
    });
    f.loop_(ctx, LoopSpec::default(), body)
}

// ---- Arithmetic and dispatch ----

#[test]
fn dispatch_writes_every_invocation() {
    let m = kernel([64, 1, 1], 1, |f, b| {
        let g = gid(f);
        let three = c32(f, 3);
        let one = c32(f, 1);
        let x = f.binary(BinOp::IMul, g, three)?;
        let x = f.binary(BinOp::IAdd, x, one)?;
        store_at(f, b[0], g, x)
    });
    for sg in [SubgroupSize::W32, SubgroupSize::W64] {
        let out = run_both(&m, cfg(3, sg, Order::Forward), &buffers(&[192]));
        let w = words(&out.buffers[0]);
        assert_eq!(w, (0..192u32).map(|g| g * 3 + 1).collect::<Vec<_>>());
        assert_eq!(out.invocations.len(), 192);
        assert_eq!(out.invocations[100].global_id, [100, 0, 0]);
    }
}

/// A module whose local size is the `BuiltIn WorkgroupSize` composite of specialization
/// constants (the form the pipeline emits under Vulkan 1.2), with no local-size mode: the
/// interpreter dispatches the composite's defaults.
#[test]
fn workgroup_size_composite_sets_the_local_size() {
    let mut b = ModuleBuilder::new();
    let buf = buffer(&mut b, 0);
    let u32_ = u32_ty(&mut b);
    let [x, y, z] = [(Axis::X, 16), (Axis::Y, 1), (Axis::Z, 1)].map(|(a, n)| {
        b.spec(SpecKey::LocalSize(a), u32_, ConstBits(n))
            .expect("spec")
    });
    b.workgroup_size(x, y, z).expect("workgroup size");
    let void = b.ty(Type::Void).expect("void");
    let mut f = b.function(void).expect("function");
    let g = gid(&mut f);
    let one = c32(&mut f, 1);
    let v = f.binary(BinOp::IAdd, g, one).expect("add");
    store_at(&mut f, buf, g, v).expect("store");
    f.ret(None).expect("ret");
    let id = f.finish().expect("finish");
    b.entry(ExecModel::GLCompute, id, Vec::new())
        .expect("entry");
    let m = b.finalize().expect("finalize");
    let out = run_both(
        &m,
        cfg(2, SubgroupSize::W32, Order::Forward),
        &buffers(&[32]),
    );
    assert_eq!(words(&out.buffers[0]), (1..=32u32).collect::<Vec<_>>());
    assert_eq!(out.invocations.len(), 32);

    // Specialization values replace the defaults (the local size included); a key the module
    // does not declare is ignored, and a value wider than the constant's type is refused.
    let mut res = buffers(&[32]);
    res.spec.insert(SpecKey::LocalSize(Axis::X), 4);
    res.spec.insert(SpecKey::MultiRegion, 1);
    let out = run_both(&m, cfg(2, SubgroupSize::W32, Order::Forward), &res);
    let mut want = vec![0u32; 32];
    for (i, w) in want.iter_mut().take(8).enumerate() {
        *w = i as u32 + 1;
    }
    assert_eq!(words(&out.buffers[0]), want);
    assert_eq!(out.invocations.len(), 8);
    let mut wide = buffers(&[32]);
    wide.spec.insert(SpecKey::LocalSize(Axis::X), 1 << 32);
    assert!(matches!(
        run(&m, &cfg(1, SubgroupSize::W32, Order::Forward), wide),
        Err(InterpError::Config(_))
    ));
}

#[test]
fn compute_builtins() {
    let m = kernel([8, 2, 1], 1, |f, b| {
        let li = lid(f);
        let sgid = builtin(f, crate::spv::BuiltIn::SubgroupId);
        let sgl = builtin(f, crate::spv::BuiltIn::SubgroupLocalInvocationId);
        let nsg = builtin(f, crate::spv::BuiltIn::NumSubgroups);
        let wg = builtin(f, crate::spv::BuiltIn::WorkgroupId);
        let four = c32(f, 4);
        let base = f.binary(BinOp::IMul, li, four)?;
        let sixty_four = c32(f, 64);
        let wgoff = f.binary(BinOp::IMul, wg, sixty_four)?;
        let base = f.binary(BinOp::IAdd, base, wgoff)?;
        for (k, v) in [sgid, sgl, nsg, li].into_iter().enumerate() {
            let kk = c32(f, k as u32);
            let i = f.binary(BinOp::IAdd, base, kk)?;
            store_at(f, b[0], i, v)?;
        }
        Ok(())
    });
    let out = run_both(
        &m,
        cfg(2, SubgroupSize::W32, Order::Forward),
        &buffers(&[128]),
    );
    let w = words(&out.buffers[0]);
    for wg in 0..2usize {
        for li in 0..16usize {
            let at = wg * 64 + li * 4;
            assert_eq!(&w[at..at + 4], &[0, li as u32, 1, li as u32]);
        }
    }
}

#[test]
fn integer_float_and_conversion_ops() {
    let m = kernel([1, 1, 1], 1, |f, b| {
        let i32_ = f
            .module()
            .ty(Type::Int {
                width: 32,
                signed: true,
            })
            .expect("i32");
        let f32_ = f.module().ty(Type::Float { width: 32 }).expect("f32");
        let u32_ = u32_ty(f.module());
        let k = |f: &mut FunctionBuilder<'_>, ty: TypeId, bits: u64| -> Value {
            let c = f
                .module()
                .constant(Constant::Scalar {
                    ty,
                    bits: ConstBits(bits),
                })
                .expect("const");
            f.module().const_value(c).expect("value")
        };
        let mut outs = Vec::new();
        // -7 / 2 = -3, -7 srem 2 = -1, -7 smod 2 = 1.
        let m7 = k(f, i32_, (-7i32) as u32 as u64);
        let two = k(f, i32_, 2);
        for op in [BinOp::SDiv, BinOp::SRem, BinOp::SMod] {
            let r = f.binary(op, m7, two)?;
            outs.push(f.convert(ConvOp::Bitcast, r, u32_)?);
        }
        // Arithmetic shift right of -8 by 1 = -4.
        let m8 = k(f, i32_, (-8i32) as u32 as u64);
        let one = k(f, i32_, 1);
        let r = f.binary(BinOp::ShiftRightArithmetic, m8, one)?;
        outs.push(f.convert(ConvOp::Bitcast, r, u32_)?);
        // 0.1f + 0.2f, rounded in f32.
        let a = k(f, f32_, u64::from(0.1f32.to_bits()));
        let bb = k(f, f32_, u64::from(0.2f32.to_bits()));
        let s = f.binary(BinOp::FAdd, a, bb)?;
        outs.push(f.convert(ConvOp::Bitcast, s, u32_)?);
        // float -> signed int truncates toward zero: -2.75 -> -2.
        let x = k(f, f32_, u64::from((-2.75f32).to_bits()));
        let r = f.convert(ConvOp::ConvertFToS, x, i32_)?;
        outs.push(f.convert(ConvOp::Bitcast, r, u32_)?);
        // u32::MAX -> f32 rounds to 2^32.
        let big = k(f, u32_, u64::from(u32::MAX));
        let r = f.convert(ConvOp::ConvertUToF, big, f32_)?;
        outs.push(f.convert(ConvOp::Bitcast, r, u32_)?);
        // Bit reverse and bit count.
        let v = k(f, u32_, 0x0000_00f1);
        outs.push(f.unary(UnOp::BitReverse, v)?);
        outs.push(f.unary(UnOp::BitCount, v)?);
        // Extract bits [4, 8) of 0xf1 sign-extended: 0b1111 -> -1.
        let four = c32(f, 4);
        let r = f.bit_field_extract(true, v, four, four)?;
        outs.push(r);
        for (i, o) in outs.into_iter().enumerate() {
            let idx = c32(f, i as u32);
            store_at(f, b[0], idx, o)?;
        }
        Ok(())
    });
    let out = run_both(
        &m,
        cfg(1, SubgroupSize::W32, Order::Forward),
        &buffers(&[10]),
    );
    let w = words(&out.buffers[0]);
    assert_eq!(
        w,
        vec![
            (-3i32) as u32,
            (-1i32) as u32,
            1,
            (-4i32) as u32,
            (0.1f32 + 0.2f32).to_bits(),
            (-2i32) as u32,
            4294967296f32.to_bits(),
            0x8f00_0000,
            5,
            u32::MAX,
        ]
    );
}

#[test]
fn arithmetic_traps() {
    type Op = fn(&mut FunctionBuilder<'_>, Value, Value) -> BResult<Value>;
    let cases: [(Op, u32, TrapKind); 3] = [
        (
            |f, a, b| f.binary(BinOp::UDiv, a, b),
            0,
            TrapKind::DivideByZero,
        ),
        (
            |f, a, b| f.binary(BinOp::ShiftLeftLogical, a, b),
            32,
            TrapKind::ShiftOutOfRange {
                shift: 32,
                width: 32,
            },
        ),
        (
            |f, a, b| f.bit_field_extract(false, a, b, b),
            20,
            TrapKind::BitFieldOutOfRange,
        ),
    ];
    for (op, rhs, want) in cases {
        let m = kernel([1, 1, 1], 1, |f, b| {
            let zero = c32(f, 0);
            // The operand comes from memory so nothing folds it.
            let x = load_at(f, b[0], zero)?;
            let seven = c32(f, 7);
            let x = f.binary(BinOp::IAdd, x, seven)?;
            let r = c32(f, rhs);
            let y = op(f, x, r)?;
            store_at(f, b[0], zero, y)
        });
        let got = trap_of(run(&m, &Config::default(), buffers(&[1])));
        assert_eq!(got, want);
    }
}

// ---- Control flow and calls ----

#[test]
fn loops_selections_switches_and_phis() {
    let m = kernel([4, 1, 1], 1, |f, b| {
        let li = lid(f);
        let u32_ = u32_ty(f.module());
        let acc = f.local_var(u32_, None)?;
        let accp = f.var_ptr(acc)?;
        let zero = c32(f, 0);
        f.store(accp, zero, MemAccess::default())?;
        // acc = sum over k < 10 of (k odd ? k : 2k).
        let ten = c32(f, 10);
        counted(f, &mut (), ten, move |fb, ctx, k| {
            let one = c32(fb, 1);
            let low = fb.binary(BinOp::BitwiseAnd, k, one)?;
            let odd = fb.binary(BinOp::INotEqual, low, zero)?;
            let t: ArmFn<'_, ()> = Box::new(move |_, _| Ok(ArmEnd::FallThrough(vec![k])));
            let e: ArmFn<'_, ()> = Box::new(move |fb, _| {
                let d = fb.binary(BinOp::IAdd, k, k)?;
                Ok(ArmEnd::FallThrough(vec![d]))
            });
            let merged = fb.if_(ctx, odd, t, Some(e))?;
            let v = merged.0[0];
            let a = fb.load(accp, MemAccess::default())?;
            let s = fb.binary(BinOp::IAdd, a, v)?;
            fb.store(accp, s, MemAccess::default())
        })?;
        let a = f.load(accp, MemAccess::default())?;
        // Switch on the lane: 0 -> +100, 1|2 -> +200, default -> +300.
        let add = |n: u32| -> ArmFn<'_, ()> {
            Box::new(move |fb, _| Ok(ArmEnd::FallThrough(vec![c32(fb, n)])))
        };
        let merged = f.switch_(
            &mut (),
            li,
            vec![(vec![0], add(100)), (vec![1, 2], add(200))],
            add(300),
        )?;
        let r = f.binary(BinOp::IAdd, a, merged.0[0])?;
        store_at(f, b[0], li, r)
    });
    let out = run_both(
        &m,
        cfg(1, SubgroupSize::W32, Order::Forward),
        &buffers(&[4]),
    );
    // odd k: 1+3+5+7+9 = 25; even 2k: 0+4+8+12+16 = 40.
    assert_eq!(words(&out.buffers[0]), vec![165, 265, 265, 365]);
    // The header runs once per iteration and once more to exit.
    let li = m.loops().len();
    assert_eq!(li, 1);
    for inv in &out.invocations {
        assert_eq!(inv.header_visits, vec![11]);
    }
}

#[test]
fn calls_pass_pointers_and_return_values() {
    let mut b = ModuleBuilder::new();
    let out = buffer(&mut b, 0);
    let u32_ = u32_ty(&mut b);
    let ptr_ty = b
        .ty(Type::Pointer {
            class: StorageClass::Function,
            pointee: u32_,
        })
        .expect("ptr");
    // fn bump(p: *u32, d: u32) -> u32 { *p += d; return *p * 2 }
    let bump = {
        let mut f = b.function(u32_).expect("function");
        let p = f.param(ptr_ty).expect("param");
        let d = f.param(u32_).expect("param");
        let v = f.load(p, MemAccess::default()).expect("load");
        let s = f.binary(BinOp::IAdd, v, d).expect("add");
        f.store(p, s, MemAccess::default()).expect("store");
        let two = c32(&mut f, 2);
        let r = f.binary(BinOp::IMul, s, two).expect("mul");
        f.ret(Some(r)).expect("ret");
        f.finish().expect("finish")
    };
    let void = b.ty(Type::Void).expect("void");
    let mut f = b.function(void).expect("function");
    let li = lid(&mut f);
    let var = f.local_var(u32_, None).expect("var");
    let vp = f.var_ptr(var).expect("ptr");
    f.store(vp, li, MemAccess::default()).expect("store");
    let five = c32(&mut f, 5);
    let r1 = f.call(bump, &[vp, five]).expect("call").expect("value");
    let r2 = f.call(bump, &[vp, r1]).expect("call").expect("value");
    store_at(&mut f, out, li, r2).expect("store");
    f.ret(None).expect("ret");
    let id = f.finish().expect("finish");
    b.entry(
        ExecModel::GLCompute,
        id,
        vec![ExecMode::LocalSize { x: 3, y: 1, z: 1 }],
    )
    .expect("entry");
    let m = b.finalize().expect("finalize");
    let out = run_both(
        &m,
        cfg(1, SubgroupSize::W32, Order::Forward),
        &buffers(&[3]),
    );
    // v = li + 5; r1 = 2v; v' = v + r1 = 3v; r2 = 6v.
    assert_eq!(words(&out.buffers[0]), vec![30, 36, 42]);
}

#[test]
fn step_bound_stops_an_infinite_loop() {
    let m = kernel([1, 1, 1], 1, |f, b| {
        let body: LoopFn<'_, ()> = Box::new(|fb, _, _| {
            let zero = c32(fb, 0);
            let x = load_at(fb, b[0], zero)?;
            let one = c32(fb, 1);
            let y = fb.binary(BinOp::IAdd, x, one)?;
            store_at(fb, b[0], zero, y)
        });
        f.loop_(&mut (), LoopSpec::default(), body)?;
        Ok(())
    });
    let c = Config {
        step_limit: 10_000,
        ..Config::default()
    };
    assert_eq!(
        trap_of(run(&m, &c, buffers(&[1]))),
        TrapKind::StepLimit(10_000)
    );
}

#[test]
fn unreachable_traps() {
    let m = kernel([1, 1, 1], 0, |f, _| f.unreachable());
    assert_eq!(
        trap_of(run(&m, &Config::default(), Resources::default())),
        TrapKind::Unreachable
    );
}

#[test]
fn non_compute_entries_are_unmodelled() {
    let mut b = ModuleBuilder::new();
    let void = b.ty(Type::Void).expect("void");
    let mut f = b.function(void).expect("function");
    f.ret(None).expect("ret");
    let id = f.finish().expect("finish");
    b.entry(ExecModel::Fragment, id, Vec::new()).expect("entry");
    let m = b.finalize().expect("finalize");
    assert_eq!(
        run(&m, &Config::default(), Resources::default()),
        Err(InterpError::Unmodelled(Unmodelled::ExecModel(
            ExecModel::Fragment
        )))
    );
}

/// A vertex entry runs one invocation per vertex and instance of a `[vertices, instances, 1]` draw
/// from vertex and instance 0 (the bases read 0), and writes its outputs into memory of its own.
#[test]
fn vertex_entries_draw_every_vertex_of_every_instance() {
    use crate::spv::BuiltIn as B;
    let mut b = ModuleBuilder::new();
    let out = buffer(&mut b, 0);
    let void = b.ty(Type::Void).expect("void");
    let f32_ = b.ty(Type::Float { width: 32 }).expect("f32");
    let vec4 = b.ty(Type::Vector { elem: f32_, len: 4 }).expect("vec4");
    let position = b
        .builtin(B::Position, IoDir::Output, vec4)
        .expect("position");
    let mut f = b.function(void).expect("function");
    let (vertex, instance) = (
        builtin(&mut f, B::VertexIndex),
        builtin(&mut f, B::InstanceIndex),
    );
    let (bv, bi) = (
        builtin(&mut f, B::BaseVertex),
        builtin(&mut f, B::BaseInstance),
    );
    let four = c32(&mut f, 4);
    let at = f.binary(BinOp::IMul, instance, four).expect("mul");
    let at = f.binary(BinOp::IAdd, at, vertex).expect("add");
    let sixteen = c32(&mut f, 16);
    let v = f
        .binary(BinOp::ShiftLeftLogical, instance, sixteen)
        .expect("shl");
    let v = f.binary(BinOp::BitwiseOr, v, vertex).expect("or");
    let v = f.binary(BinOp::IAdd, v, bv).expect("base vertex");
    let v = f.binary(BinOp::IAdd, v, bi).expect("base instance");
    store_at(&mut f, out, at, v).expect("store");
    let x = f.convert(ConvOp::ConvertUToF, vertex, f32_).expect("x");
    let pos = f.composite_construct(vec4, &[x, x, x, x]).expect("pos");
    let p = f.module().global_value(position).expect("global");
    f.store(p, pos, MemAccess::default())
        .expect("store position");
    f.ret(None).expect("ret");
    let id = f.finish().expect("finish");
    b.entry(ExecModel::Vertex, id, Vec::new()).expect("entry");
    let m = b.finalize().expect("finalize");
    let c = Config {
        workgroups: [3, 2, 1],
        ..Config::default()
    };
    let o = run(&m, &c, buffers(&[8])).expect("run");
    let mut want = vec![0u32; 8];
    for i in 0..2u32 {
        for v in 0..3u32 {
            want[(i * 4 + v) as usize] = i << 16 | v;
        }
    }
    assert_eq!(words(&o.buffers[0]), want);
    assert_eq!(o.invocations.len(), 6);
}

// ---- Barriers ----

/// `wg[li] = li; barrier; out[li] = wg[(li + 1) % n]` over a `Workgroup` array.
fn neighbour_exchange(n: u32) -> TypedModule {
    kernel([n, 1, 1], 1, |f, b| {
        let u32_ = u32_ty(f.module());
        let len = f.u32_const(n)?;
        let arr = f.module().ty(Type::Array {
            elem: u32_,
            len,
            stride: None,
        })?;
        let g = f
            .module()
            .plain_global(StorageClass::Workgroup, arr, None)?;
        let base = f.module().global_value(g)?;
        let li = lid(f);
        let p = f.access_chain(base, &[li])?;
        f.store(p, li, MemAccess::default())?;
        wg_barrier(f)?;
        let one = c32(f, 1);
        let nn = c32(f, n);
        let j = f.binary(BinOp::IAdd, li, one)?;
        let j = f.binary(BinOp::UMod, j, nn)?;
        let q = f.access_chain(base, &[j])?;
        let v = f.load(q, MemAccess::default())?;
        store_at(f, b[0], li, v)
    })
}

#[test]
fn barrier_orders_workgroup_memory() {
    let m = neighbour_exchange(96);
    for sg in [SubgroupSize::W32, SubgroupSize::W64] {
        let out = run_both(&m, cfg(2, sg, Order::Forward), &buffers(&[96]));
        let w = words(&out.buffers[0]);
        assert_eq!(w, (0..96u32).map(|i| (i + 1) % 96).collect::<Vec<_>>());
        assert!(out.invocations.iter().all(|i| i.barriers == 1));
    }
}

/// A `Workgroup` array whose length is an `OpSpecConstantOp` of a specialization constant is sized
/// by the run's specialization value (as Vulkan sizes it at pipeline creation), not the default:
/// each invocation writes and reads back its own word past the default length.
#[test]
fn a_spec_op_length_follows_the_specialization() {
    use crate::spirv::ir::SpecOp;
    let m = kernel([64, 1, 1], 1, |f, b| {
        let u32_ = u32_ty(f.module());
        let len = f.module().spec(SpecKey::TgLength(0), u32_, ConstBits(16))?;
        let two = f.u32_const(2)?;
        let words = f
            .module()
            .spec_op(SpecOp::ShiftRightLogical, u32_, &[len, two])?;
        let arr = f.module().ty(Type::Array {
            elem: u32_,
            len: words,
            stride: None,
        })?;
        let g = f
            .module()
            .plain_global(StorageClass::Workgroup, arr, None)?;
        let base = f.module().global_value(g)?;
        let li = lid(f);
        let p = f.access_chain(base, &[li])?;
        f.store(p, li, MemAccess::default())?;
        let v = f.load(p, MemAccess::default())?;
        let n = f.module().const_value(words)?;
        let v = f.binary(BinOp::IAdd, v, n)?;
        store_at(f, b[0], li, v)
    });
    let mut res = buffers(&[64]);
    res.spec.insert(SpecKey::TgLength(0), 256);
    let out = run(&m, &cfg(1, SubgroupSize::W32, Order::Forward), res).expect("run");
    assert_eq!(
        words(&out.buffers[0]),
        (0..64u32).map(|i| i + 64).collect::<Vec<_>>()
    );
    // With the default (16 bytes, 4 words), invocation 4 indexes past the array.
    let r = run(
        &m,
        &cfg(1, SubgroupSize::W32, Order::Forward),
        buffers(&[64]),
    );
    match r {
        Err(InterpError::Trap(t)) => {
            assert_eq!(t.kind, TrapKind::IndexOutOfRange { index: 4, len: 4 });
        }
        other => panic!("{other:?}"),
    }
}

/// `SpecOp::eval` compares at the operand width it is given: `4 < 8` and not `9 < 8` for `u32`.
#[test]
fn spec_op_eval_compares_at_the_operand_width() {
    use crate::spirv::ir::SpecOp;
    assert_eq!(SpecOp::ULessThan.eval(32, &[4, 8]), Some(1));
    assert_eq!(SpecOp::ULessThan.eval(32, &[9, 8]), Some(0));
    assert_eq!(SpecOp::ULessThan.eval(32, &[0, 8]), Some(1));
    assert_eq!(SpecOp::Select.eval(32, &[1, 8, 2]), Some(8));
    assert_eq!(SpecOp::Select.eval(32, &[0, 8, 2]), Some(2));
}

/// Invocation `t < n` writes `t` to word `t` of the `Workgroup` `u32` array `array` declares,
/// reads it back after a barrier and stores it to `out[t]`; the others write and read word 0
/// (the value 0, as invocation 0 does) and store 0.
fn workgroup_words_kernel(
    n: u32,
    array: impl FnOnce(&mut ModuleBuilder) -> BResult<GlobalId>,
) -> TypedModule {
    kernel([64, 1, 1], 1, move |f, b| {
        let g = array(f.module())?;
        let base = f.module().global_value(g)?;
        let li = lid(f);
        let nv = c32(f, n);
        let zero = c32(f, 0);
        let inside = f.binary(BinOp::ULessThan, li, nv)?;
        let idx = f.select(inside, li, zero)?;
        let p = f.access_chain(base, &[idx])?;
        f.store(p, idx, MemAccess::default())?;
        wg_barrier(f)?;
        let v = f.load(p, MemAccess::default())?;
        let v = f.select(inside, v, zero)?;
        store_at(f, b[0], li, v)
    })
}

/// A `u32` scalar constant.
fn mconst(mb: &mut ModuleBuilder, v: u64) -> BResult<crate::spirv::ids::ConstId> {
    let ty = u32_ty(mb);
    mb.constant(Constant::Scalar {
        ty,
        bits: ConstBits(v),
    })
}

/// An `OpSpecConstantOp` `ULessThan` of two `u32`s compares at 32 bits (not at its `bool` result's
/// width), so `select(len < 8, 8, len)` floors the array length at 8 words for a specialized `len`
/// under 8 and keeps it from 8 up.
#[test]
fn a_spec_op_unsigned_less_than_compares_its_operands() {
    use crate::spirv::ir::SpecOp;
    let m = workgroup_words_kernel(8, |mb| {
        let u32_ = u32_ty(mb);
        let len = mb.spec(SpecKey::TgLength(0), u32_, ConstBits(4))?;
        let eight = mconst(mb, 8)?;
        let bool_ = mb.ty(Type::Bool)?;
        let small = mb.spec_op(SpecOp::ULessThan, bool_, &[len, eight])?;
        let words = mb.spec_op(SpecOp::Select, u32_, &[small, eight, len])?;
        let arr = mb.ty(Type::Array {
            elem: u32_,
            len: words,
            stride: None,
        })?;
        mb.plain_global(StorageClass::Workgroup, arr, None)
    });
    for len in [0u64, 4, 7, 8, 9] {
        let mut res = buffers(&[64]);
        res.spec.insert(SpecKey::TgLength(0), len);
        let out = run(&m, &cfg(1, SubgroupSize::W32, Order::Forward), res)
            .unwrap_or_else(|e| panic!("len {len}: {e:?}"));
        let want: Vec<u32> = (0..64u32).map(|t| if t < 8 { t } else { 0 }).collect();
        assert_eq!(words(&out.buffers[0]), want, "len {len}");
    }
}

/// A threadgroup arena whose specialized size is under `MIN_ARENA_BYTES` (no static part and a
/// region of 0 or 16 bytes, or a 4-byte static part and a 0-byte region) is 8 words on the
/// interpreter, as Vulkan sizes it from the serialized `OpSpecConstantOp`s, so every word a
/// contained 32-byte load at offset 0 reads is inside the array.
#[test]
fn a_small_specialized_threadgroup_arena_is_floored_at_eight_words() {
    use crate::memory::arena::{declare_threadgroup, TgShape, MIN_ARENA_BYTES};
    let min_words = (MIN_ARENA_BYTES / 4) as u32;
    for (static_size, len) in [(0u64, 0u64), (0, 16), (4, 0), (0, 32), (0, 48)] {
        let m = workgroup_words_kernel(min_words, |mb| {
            let shape = TgShape {
                static_size,
                regions: vec![(0, 0)],
                imageblock: None,
                tile: false,
                fragment: None,
                words: Default::default(),
            };
            Ok(declare_threadgroup(mb, &shape)?.decl.global)
        });
        let mut res = buffers(&[64]);
        res.spec.insert(SpecKey::TgLength(0), len);
        let out = run(&m, &cfg(1, SubgroupSize::W32, Order::Forward), res)
            .unwrap_or_else(|e| panic!("static {static_size}, region {len}: {e:?}"));
        let want: Vec<u32> = (0..64u32)
            .map(|t| if t < min_words { t } else { 0 })
            .collect();
        assert_eq!(
            words(&out.buffers[0]),
            want,
            "static {static_size}, region {len}"
        );
    }
    // Past the floor the array is exactly the specialized size: 48 bytes are 12 words.
    let m = workgroup_words_kernel(13, |mb| {
        let shape = TgShape {
            static_size: 0,
            regions: vec![(0, 0)],
            imageblock: None,
            tile: false,
            fragment: None,
            words: Default::default(),
        };
        Ok(declare_threadgroup(mb, &shape)?.decl.global)
    });
    let mut res = buffers(&[64]);
    res.spec.insert(SpecKey::TgLength(0), 48);
    match run(&m, &cfg(1, SubgroupSize::W32, Order::Forward), res) {
        Err(InterpError::Trap(t)) => {
            assert_eq!(t.kind, TrapKind::IndexOutOfRange { index: 12, len: 12 });
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn uninitialized_workgroup_read_traps() {
    // Without the barrier, the first subgroup reads a neighbour's slot of the next subgroup
    // before it is written.
    let m = kernel([64, 1, 1], 1, |f, b| {
        let u32_ = u32_ty(f.module());
        let len = f.u32_const(64)?;
        let arr = f.module().ty(Type::Array {
            elem: u32_,
            len,
            stride: None,
        })?;
        let g = f
            .module()
            .plain_global(StorageClass::Workgroup, arr, None)?;
        let base = f.module().global_value(g)?;
        let li = lid(f);
        let p = f.access_chain(base, &[li])?;
        f.store(p, li, MemAccess::default())?;
        let one = c32(f, 1);
        let n = c32(f, 64);
        let j = f.binary(BinOp::IAdd, li, one)?;
        let j = f.binary(BinOp::UMod, j, n)?;
        let q = f.access_chain(base, &[j])?;
        let v = f.load(q, MemAccess::default())?;
        store_at(f, b[0], li, v)
    });
    let r = run(
        &m,
        &cfg(1, SubgroupSize::W32, Order::Forward),
        buffers(&[64]),
    );
    match r {
        Err(InterpError::Trap(t)) => {
            assert_eq!(t.kind, TrapKind::UninitializedRead { offset: 128 });
            assert_eq!(t.invocation, Some([31, 0, 0]));
        }
        other => panic!("{other:?}"),
    }
}

/// `if li >= 40 return; wg[li] = li; barrier; out[li] = wg[39 - li]` over 64 invocations.
fn early_return_kernel() -> TypedModule {
    kernel([64, 1, 1], 1, |f, b| {
        let u32_ = u32_ty(f.module());
        let len = f.u32_const(64)?;
        let arr = f.module().ty(Type::Array {
            elem: u32_,
            len,
            stride: None,
        })?;
        let g = f
            .module()
            .plain_global(StorageClass::Workgroup, arr, None)?;
        let base = f.module().global_value(g)?;
        let li = lid(f);
        let forty = c32(f, 40);
        let late = f.binary(BinOp::UGreaterThanEqual, li, forty)?;
        let ret: ArmFn<'_, ()> = Box::new(|fb, _| {
            fb.ret(None)?;
            Ok(ArmEnd::Exits)
        });
        f.if_(&mut (), late, ret, None)?;
        let p = f.access_chain(base, &[li])?;
        f.store(p, li, MemAccess::default())?;
        wg_barrier(f)?;
        let top = c32(f, 39);
        let j = f.binary(BinOp::ISub, top, li)?;
        let q = f.access_chain(base, &[j])?;
        let v = f.load(q, MemAccess::default())?;
        store_at(f, b[0], li, v)
    })
}

#[test]
fn early_return_before_a_barrier_is_the_q3_model() {
    let m = early_return_kernel();
    let out = run_both(
        &m,
        cfg(1, SubgroupSize::W32, Order::Forward),
        &buffers(&[64]),
    );
    let w = words(&out.buffers[0]);
    let want: Vec<u32> = (0..64u32)
        .map(|i| if i < 40 { 39 - i } else { 0 })
        .collect();
    assert_eq!(w, want);
}

/// Under Vulkan's rule a workgroup barrier some invocations skip traps, and the barrier check
/// reports it statically.
#[test]
fn early_return_before_a_barrier_traps_under_the_vulkan_rule() {
    let m = early_return_kernel();
    let c = Config {
        barriers: BarrierRule::Vulkan,
        ..cfg(1, SubgroupSize::W32, Order::Forward)
    };
    assert_eq!(
        trap_of(run(&m, &c, buffers(&[64]))),
        TrapKind::NonUniformBarrier
    );
    let report = crate::spirv::testing::barriers::workgroup_barriers(&m).expect("check");
    assert_eq!(report.barriers, 1);
    assert_eq!(report.nonuniform.len(), 1, "{report:?}");
}

/// A block that ends in `OpUnreachable` is never reached in a defined execution, so the barrier
/// check proves uniform a barrier after a divergent construct whose only other way out is
/// `OpUnreachable` (`if (li == 0) { if (b[0] != 0) unreachable; b[1] = 1; } barrier`, the shape
/// of a `switch` whose default Metal's compiler marks unreachable); it ran with `b[0] == 0`
/// under the Vulkan rule without a trap. With an early return in its place, the check reports
/// the barrier, as before.
#[test]
fn the_barrier_check_takes_no_path_into_unreachable() {
    for end in ["unreachable", "return"] {
        let m = kernel([64, 1, 1], 1, |f, b| {
            let buf = b[0];
            let li = lid(f);
            let zero = c32(f, 0);
            let one = c32(f, 1);
            let first = f.binary(BinOp::IEqual, li, zero)?;
            let arm: ArmFn<'_, ()> = Box::new(move |fb, _| {
                let w = load_at(fb, buf, zero)?;
                let bad = fb.binary(BinOp::INotEqual, w, zero)?;
                let leave: ArmFn<'_, ()> = Box::new(move |fb, _| {
                    if end == "unreachable" {
                        fb.unreachable()?;
                    } else {
                        fb.ret(None)?;
                    }
                    Ok(ArmEnd::Exits)
                });
                fb.if_(&mut (), bad, leave, None)?;
                store_at(fb, buf, one, one)?;
                Ok(ArmEnd::FallThrough(Vec::new()))
            });
            f.if_(&mut (), first, arm, None)?;
            wg_barrier(f)?;
            let two = c32(f, 2);
            store_at(f, buf, two, li)
        });
        let report = crate::spirv::testing::barriers::workgroup_barriers(&m).expect("check");
        assert_eq!(report.barriers, 1, "{end}");
        assert_eq!(
            report.nonuniform.is_empty(),
            end == "unreachable",
            "{end}: {report:?}"
        );
        if end == "unreachable" {
            let c = Config {
                barriers: BarrierRule::Vulkan,
                ..cfg(1, SubgroupSize::W32, Order::Forward)
            };
            run(&m, &c, buffers(&[64])).expect("no invocation takes the unreachable arm");
        }
    }
}

/// The barrier check proves a barrier uniform behind a loop that exits on a workgroup vote (a word
/// cleared, set, and read between workgroup barriers), and reports the same loop exiting on the
/// lane's own flag.
#[test]
fn the_barrier_check_reads_a_vote_as_uniform() {
    for vote in [true, false] {
        // live = li < 40; loop { if !any(live) break; barrier; live = false }
        let m = kernel([64, 1, 1], 0, |f, _| {
            let u32_ = u32_ty(f.module());
            let word = f
                .module()
                .plain_global(StorageClass::Workgroup, u32_, None)?;
            let p = f.module().global_value(word)?;
            let li = lid(f);
            let forty = c32(f, 40);
            let first = f.binary(BinOp::ULessThan, li, forty)?;
            let bool_ = f.module().ty(Type::Bool)?;
            let live = f.local_var(bool_, None)?;
            let lp = f.var_ptr(live)?;
            f.store(lp, first, MemAccess::default())?;
            let body: LoopFn<'_, ()> = Box::new(move |fb, _, lc| {
                let lp = fb.var_ptr(live)?;
                let l = fb.load(lp, MemAccess::default())?;
                let go = if vote {
                    let zero = c32(fb, 0);
                    let one = c32(fb, 1);
                    let relaxed = Semantics::RELAXED;
                    fb.atomic_store(p, Scope::Workgroup, relaxed, zero)?;
                    wg_barrier(fb)?;
                    let bit = fb.select(l, one, zero)?;
                    fb.atomic(AtomicOp::Or, p, Scope::Workgroup, relaxed, bit)?;
                    wg_barrier(fb)?;
                    let x = fb.atomic_load(p, Scope::Workgroup, relaxed)?;
                    wg_barrier(fb)?;
                    fb.binary(BinOp::INotEqual, x, zero)?
                } else {
                    l
                };
                let stop = fb.unary(UnOp::LogicalNot, go)?;
                fb.if_(
                    &mut (),
                    stop,
                    Box::new(|fb, _| {
                        lc.break_(fb)?;
                        Ok(ArmEnd::Exits)
                    }),
                    None,
                )?;
                wg_barrier(fb)?;
                let no = fb.module().constant(Constant::Scalar {
                    ty: bool_,
                    bits: ConstBits(0),
                })?;
                let no = fb.module().const_value(no)?;
                let lp = fb.var_ptr(live)?;
                fb.store(lp, no, MemAccess::default())?;
                Ok(())
            });
            f.loop_(&mut (), LoopSpec::default(), body)?;
            Ok(())
        });
        let report = crate::spirv::testing::barriers::workgroup_barriers(&m).expect("check");
        let barriers = if vote { 4 } else { 1 };
        assert_eq!(report.barriers, barriers, "vote {vote}");
        assert_eq!(
            report.nonuniform.is_empty(),
            vote,
            "vote {vote}: {report:?}"
        );
        let c = Config {
            barriers: BarrierRule::Vulkan,
            ..cfg(1, SubgroupSize::W32, Order::Forward)
        };
        let r = run(&m, &c, Resources::default());
        if vote {
            r.expect("a voted loop is uniform");
        } else {
            assert_eq!(trap_of(r), TrapKind::NonUniformBarrier);
        }
    }
}

#[test]
fn divergent_barriers_trap() {
    // if li < 16 { barrier } else { barrier }: two different barriers.
    let m = kernel([32, 1, 1], 0, |f, _| {
        let li = lid(f);
        let sixteen = c32(f, 16);
        let low = f.binary(BinOp::ULessThan, li, sixteen)?;
        let arm = || -> ArmFn<'static, ()> {
            Box::new(|fb, _| {
                wg_barrier(fb)?;
                Ok(ArmEnd::FallThrough(Vec::new()))
            })
        };
        f.if_(&mut (), low, arm(), Some(arm()))?;
        Ok(())
    });
    assert_eq!(
        trap_of(run(&m, &Config::default(), Resources::default())),
        TrapKind::NonUniformBarrier
    );
}

#[test]
fn a_race_shows_as_an_order_difference() {
    // Every invocation writes wg[0] = li without synchronization, then reads it after a barrier.
    let m = kernel([64, 1, 1], 1, |f, b| {
        let u32_ = u32_ty(f.module());
        let g = f
            .module()
            .plain_global(StorageClass::Workgroup, u32_, None)?;
        let p = f.module().global_value(g)?;
        let li = lid(f);
        f.store(p, li, MemAccess::default())?;
        wg_barrier(f)?;
        let v = f.load(p, MemAccess::default())?;
        store_at(f, b[0], li, v)
    });
    let res = buffers(&[64]);
    let c = cfg(1, SubgroupSize::W32, Order::Forward);
    let fwd = run(&m, &c, res.clone()).expect("forward");
    let rev = run(
        &m,
        &Config {
            order: Order::Reverse,
            ..c
        },
        res,
    )
    .expect("reverse");
    assert_ne!(fwd.buffers, rev.buffers);
    assert_eq!(words(&fwd.buffers[0])[0], 63);
    assert_eq!(words(&rev.buffers[0])[0], 0);
}

// ---- Composites and GLSL ----

#[test]
fn composites_and_exact_glsl() {
    use crate::spirv::ir::GlslOp;
    let m = kernel([4, 1, 1], 1, |f, b| {
        let li = lid(f);
        let u32_ = u32_ty(f.module());
        let f32_ = f.module().ty(Type::Float { width: 32 })?;
        let uvec4 = f.module().ty(Type::Vector { elem: u32_, len: 4 })?;
        let (one, two, three, nine) = (c32(f, 1), c32(f, 2), c32(f, 3), c32(f, 9));
        let v = f.composite_construct(uvec4, &[li, one, two, three])?;
        let v = f.composite_insert(nine, v, &[2])?;
        let v = f.vector_shuffle(v, v, &[3, 2, 1, 0])?;
        // Reversed: (3, 9, 1, li); lane li picks component li.
        let picked = f.vector_extract_dynamic(v, li)?;
        let fc = |f: &mut FunctionBuilder<'_>, x: f32| -> BResult<Value> {
            let c = f.module().constant(Constant::Scalar {
                ty: f32_,
                bits: ConstBits(u64::from(x.to_bits())),
            })?;
            f.module().const_value(c)
        };
        let (x, lo, hi) = (fc(f, 5.5)?, fc(f, 0.0)?, fc(f, 2.0)?);
        let clamped = f.glsl(GlslOp::FClamp, &[x, lo, hi])?;
        let clamped = f.convert(ConvOp::Bitcast, clamped, u32_)?;
        let v256 = c32(f, 0x100);
        let msb = f.glsl(GlslOp::FindUMsb, &[v256])?;
        let third = fc(f, 1.0 / 3.0)?;
        let fma = f.glsl(GlslOp::Fma, &[third, hi, x])?;
        let fma = f.convert(ConvOp::Bitcast, fma, u32_)?;
        let four = c32(f, 4);
        let base = f.binary(BinOp::IMul, li, four)?;
        for (k, val) in [picked, clamped, msb, fma].into_iter().enumerate() {
            let kk = c32(f, k as u32);
            let i = f.binary(BinOp::IAdd, base, kk)?;
            store_at(f, b[0], i, val)?;
        }
        Ok(())
    });
    let out = run_both(
        &m,
        cfg(1, SubgroupSize::W32, Order::Forward),
        &buffers(&[16]),
    );
    let w = words(&out.buffers[0]);
    let fma = (1.0f32 / 3.0).mul_add(2.0, 5.5).to_bits();
    for li in 0..4usize {
        let picked = [3, 9, 1, li as u32][li];
        assert_eq!(&w[li * 4..li * 4 + 4], &[picked, 2.0f32.to_bits(), 8, fma]);
    }
}

/// `Length` sums the squares in order, rounding each step, and rounds the root once: on a vector
/// and on a scalar, whose length is its magnitude.
#[test]
fn length_is_the_rounded_root_of_the_summed_squares() {
    use crate::spirv::ir::GlslOp;
    let m = kernel([1, 1, 1], 1, |f, buf| {
        let u32_ = u32_ty(f.module());
        let f32_ = f.module().ty(Type::Float { width: 32 })?;
        let vec3 = f.module().ty(Type::Vector { elem: f32_, len: 3 })?;
        let fc = |f: &mut FunctionBuilder<'_>, x: f32| -> BResult<Value> {
            let k = f.module().constant(Constant::Scalar {
                ty: f32_,
                bits: ConstBits(u64::from(x.to_bits())),
            })?;
            f.module().const_value(k)
        };
        let (a, b, c, s) = (fc(f, 1.0)?, fc(f, 2.0)?, fc(f, 3.0)?, fc(f, -2.5)?);
        let v = f.composite_construct(vec3, &[a, b, c])?;
        for (k, x) in [v, s].into_iter().enumerate() {
            let l = f.glsl(GlslOp::Length, &[x])?;
            let l = f.convert(ConvOp::Bitcast, l, u32_)?;
            let at = c32(f, k as u32);
            store_at(f, buf[0], at, l)?;
        }
        Ok(())
    });
    let out = run_both(
        &m,
        cfg(1, SubgroupSize::W32, Order::Forward),
        &buffers(&[2]),
    );
    let w = words(&out.buffers[0]);
    assert_eq!(&w[..2], &[14.0f32.sqrt().to_bits(), 2.5f32.to_bits()]);
}

/// `OpFmaKHR` rounds once: on `a = 1 + 2^-23`, `b = 1 - 2^-24`, `c = -RN(a * b)` (A1 case
/// 52343cc4's lane) it gives the product's rounding error, where a separate multiply and add give
/// zero; per lane of a vector too.
#[test]
fn fma_khr_rounds_once() {
    let a = 1.0f32 + f32::EPSILON;
    let b = 1.0f32 - f32::EPSILON / 2.0;
    let c = -(a * b);
    let m = kernel([1, 1, 1], 1, |f, buf| {
        let u32_ = u32_ty(f.module());
        let f32_ = f.module().ty(Type::Float { width: 32 })?;
        let vec2 = f.module().ty(Type::Vector { elem: f32_, len: 2 })?;
        let fc = |f: &mut FunctionBuilder<'_>, x: f32| -> BResult<Value> {
            let k = f.module().constant(Constant::Scalar {
                ty: f32_,
                bits: ConstBits(u64::from(x.to_bits())),
            })?;
            f.module().const_value(k)
        };
        let (x, y, z) = (fc(f, a)?, fc(f, b)?, fc(f, c)?);
        let r = f.fma(x, y, z)?;
        let r = f.convert(ConvOp::Bitcast, r, u32_)?;
        let zero = c32(f, 0);
        store_at(f, buf[0], zero, r)?;
        let two = fc(f, 2.0)?;
        let xv = f.composite_construct(vec2, &[x, two])?;
        let yv = f.composite_construct(vec2, &[y, two])?;
        let zv = f.composite_construct(vec2, &[z, two])?;
        let rv = f.fma(xv, yv, zv)?;
        for k in 0..2u32 {
            let lane = f.composite_extract(rv, &[k])?;
            let lane = f.convert(ConvOp::Bitcast, lane, u32_)?;
            let at = c32(f, k + 1);
            store_at(f, buf[0], at, lane)?;
        }
        Ok(())
    });
    let out = run_both(
        &m,
        cfg(1, SubgroupSize::W32, Order::Forward),
        &buffers(&[3]),
    );
    let w = words(&out.buffers[0]);
    let fused = a.mul_add(b, c);
    assert_ne!(fused, 0.0);
    assert_eq!(a * b + c, 0.0);
    assert_eq!(
        &w[..3],
        &[fused.to_bits(), fused.to_bits(), 6.0f32.to_bits()]
    );
}

#[test]
fn dynamic_extract_out_of_range_and_transcendentals() {
    use crate::spirv::ir::GlslOp;
    let m = kernel([5, 1, 1], 1, |f, b| {
        let li = lid(f);
        let u32_ = u32_ty(f.module());
        let uvec4 = f.module().ty(Type::Vector { elem: u32_, len: 4 })?;
        let v = f.composite_construct(uvec4, &[li, li, li, li])?;
        let x = f.vector_extract_dynamic(v, li)?;
        store_at(f, b[0], li, x)
    });
    assert_eq!(
        trap_of(run(&m, &Config::default(), buffers(&[5]))),
        TrapKind::IndexOutOfRange { index: 4, len: 4 }
    );
    let m = kernel([1, 1, 1], 1, |f, b| {
        let f32_ = f.module().ty(Type::Float { width: 32 })?;
        let u32_ = u32_ty(f.module());
        let c = f.module().constant(Constant::Null { ty: f32_ })?;
        let z = f.module().const_value(c)?;
        let s = f.glsl(GlslOp::Sin, &[z])?;
        let s = f.convert(ConvOp::Bitcast, s, u32_)?;
        let zero = c32(f, 0);
        store_at(f, b[0], zero, s)
    });
    assert_eq!(
        run(&m, &Config::default(), buffers(&[1])),
        Err(InterpError::Unmodelled(Unmodelled::Glsl(GlslOp::Sin)))
    );
}

// ---- Switches split past the pair limit ----

/// One switch shape: its arms (literals, the value it falls through with, or `None` when it
/// returns after storing its marker), and the default's (`None`: returns).
struct SwitchShape {
    arms: Vec<(Vec<u32>, Option<u32>)>,
    default: Option<u32>,
}

/// A small deterministic generator (xorshift).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n.max(1)
    }
}

/// A shape of `arms` arms, each of 1 to `2 * limit + 1` literals (so some arm alone holds more
/// than one `OpSwitch` may), drawn from a narrow range and from the whole `u32` range, the
/// arms' literals interleaved; about one arm in five, and the default every third shape, return.
fn switch_shape(rng: &mut Rng, arms: usize, limit: usize, k: u64) -> SwitchShape {
    let mut seen = std::collections::BTreeSet::new();
    let mut out = Vec::with_capacity(arms);
    for a in 0..arms {
        let n = 1 + rng.below(2 * limit as u64 + 1) as usize;
        let mut lits = Vec::with_capacity(n);
        while lits.len() < n {
            let l = match rng.below(4) {
                0 => u32::MAX - rng.below(8) as u32,
                1 => rng.next() as u32,
                _ => rng.below(64 * arms as u64) as u32,
            };
            if seen.insert(l) {
                lits.push(l);
            }
        }
        let value = (rng.below(5) != 0).then_some(1000 + a as u32);
        out.push((lits, value));
    }
    SwitchShape {
        arms: out,
        default: (!k.is_multiple_of(3)).then_some(7),
    }
}

/// What the switch of `s` leaves in an invocation's output word for selector `x`: the value its
/// arm falls through with, or the arm's return marker `2000 + arm` (`3000` for the default's).
fn switch_expected(s: &SwitchShape, x: u32) -> u32 {
    for (a, (lits, value)) in s.arms.iter().enumerate() {
        if lits.contains(&x) {
            return value.unwrap_or(2000 + a as u32);
        }
    }
    s.default.unwrap_or(3000)
}

/// A kernel of `local` invocations per group that runs the switch of `s` on its selector
/// (`buf0[gid]`) with `kind`, and stores what it leaves to `buf1[gid]`.
fn switch_kernel(
    s: &SwitchShape,
    kind: crate::spirv::builder::constructs::SwitchKind,
) -> TypedModule {
    kernel([64, 1, 1], 2, |f, b| {
        let (sel_buf, out) = (b[0], b[1]);
        let g = gid(f);
        let x = load_at(f, sel_buf, g)?;
        let arm = |value: Option<u32>, marker: u32| -> ArmFn<'_, ()> {
            Box::new(move |fb, _| match value {
                Some(v) => Ok(ArmEnd::FallThrough(vec![c32(fb, v)])),
                None => {
                    let k = c32(fb, marker);
                    store_at(fb, out, g, k)?;
                    fb.ret(None)?;
                    Ok(ArmEnd::Exits)
                }
            })
        };
        let cases = s
            .arms
            .iter()
            .enumerate()
            .map(|(a, (lits, value))| (lits.clone(), arm(*value, 2000 + a as u32)))
            .collect();
        let merged = f.switch_within(&mut (), x, cases, arm(s.default, 3000), kind)?;
        if f.current_block().is_some() {
            match merged.0.as_slice() {
                [v] => store_at(f, out, g, *v)?,
                _ => panic!("a switch with fall-through arms merges one value"),
            }
        }
        Ok(())
    })
}

/// A switch of more literals than its limit is split into switches within it, and runs exactly as
/// one switch would: for every selector value (each literal, its neighbours, values in the gaps,
/// and both ends of the `u32` range), the arm its literal names runs, and every other value runs
/// the default, with fall-through values merged and returns taken from inside the arms. Shapes
/// include arms with more literals than the limit, literals of several arms interleaved, and
/// switches-at-use; every `OpSwitch` stays within the limit, and the module passes the owned
/// verifier.
#[test]
fn a_switch_split_at_its_limit_selects_every_arm_as_one_switch() {
    use crate::spirv::builder::constructs::SwitchKind;
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    for k in 0..36u64 {
        let limit = [3usize, 4, 5, 8][(k % 4) as usize];
        let arms = 1 + rng.below(24) as usize;
        let s = switch_shape(&mut rng, arms, limit, k);
        let kind = SwitchKind {
            at_use: k % 2 == 1,
            limit,
        };
        let m = switch_kernel(&s, kind);
        let mut meter = crate::budget::Meter::new(&crate::budget::Budget::default());
        crate::spirv::verify::check_structure(&m, &mut meter)
            .unwrap_or_else(|e| panic!("shape {k}: the owned verifier rejected it: {e}"));
        let total: usize = s.arms.iter().map(|(l, _)| l.len()).sum();
        let mut switches = 0;
        for b in m.blocks() {
            if let crate::spirv::ir::Terminator::Switch { cases, .. } = &b.term {
                switches += 1;
                assert!(cases.len() <= limit, "shape {k}: {} pairs", cases.len());
            }
        }
        assert!(switches >= 1, "shape {k}: no switch");
        if total <= limit {
            assert_eq!(
                switches, 1,
                "shape {k}: a switch within its limit is one OpSwitch"
            );
        }
        // Every literal, its neighbours, and values in the gaps and at the ends.
        let mut xs: Vec<u32> = vec![0, 1, u32::MAX, u32::MAX - 1];
        for (lits, _) in &s.arms {
            for &l in lits {
                xs.extend([l, l.wrapping_sub(1), l.wrapping_add(1)]);
            }
        }
        for _ in 0..16 {
            xs.push(rng.next() as u32);
        }
        xs.sort_unstable();
        xs.dedup();
        let groups = xs.len().div_ceil(64) as u32;
        xs.resize(groups as usize * 64, 0);
        let mut res = buffers(&[xs.len(), xs.len()]);
        res.buffers[0].bytes = xs.iter().flat_map(|x| x.to_le_bytes()).collect();
        let out = run_both(&m, cfg(groups, SubgroupSize::W32, Order::Forward), &res);
        let got = words(&out.buffers[1]);
        for (i, x) in xs.iter().enumerate() {
            assert_eq!(
                got[i],
                switch_expected(&s, *x),
                "shape {k} (limit {limit}, {total} literals): selector {x}"
            );
        }
    }
}

// ---- Robust images ----

/// A load outside a robust image returns the invocation's own earlier store there (Vulkan's
/// permission, which the model takes) unless the load is volatile; under the Vulkan memory model
/// that is the read's `VolatileTexel`, the form of the `Volatile` decoration that model bans. One
/// invocation writes 7 at `(5, 0)` of a 1x1 robust image and reads it back.
#[test]
fn a_volatile_texel_read_never_returns_a_discarded_store() {
    let tex = ResourceSlot {
        stage: Stage::Kernel,
        kind: ResourceKind::Texture,
        index: 0,
    };
    for volatile in [false, true] {
        let mut b = ModuleBuilder::new();
        let out = buffer(&mut b, 0);
        let u32_ = u32_ty(&mut b);
        let ty = b
            .ty(Type::Image(crate::spirv::ir::ImageType {
                sampled_type: u32_,
                dim: crate::spirv::ir::Dim::D2,
                depth: false,
                arrayed: false,
                multisampled: false,
                usage: crate::spirv::ir::ImageUsage::Storage,
                format: crate::spv::ImageFormat::R32ui,
            }))
            .expect("image type");
        let image = b
            .resource(tex, StorageClass::UniformConstant, ty)
            .expect("image");
        let uvec2 = b.ty(Type::Vector { elem: u32_, len: 2 }).expect("uvec2");
        let uvec4 = b.ty(Type::Vector { elem: u32_, len: 4 }).expect("uvec4");
        let void = b.ty(Type::Void).expect("void");
        let mut f = b.function(void).expect("function");
        let (zero, five, seven) = (c32(&mut f, 0), c32(&mut f, 5), c32(&mut f, 7));
        let at = f
            .composite_construct(uvec2, &[five, zero])
            .expect("coordinate");
        let texel = f
            .composite_construct(uvec4, &[seven, zero, zero, zero])
            .expect("texel");
        let p = f.module().global_value(image).expect("image pointer");
        let img = f.load(p, MemAccess::default()).expect("image");
        let none = ImageOperands::default();
        f.write(img, at, texel, &none, TexelMemory::PRIVATE)
            .expect("write");
        let t = f
            .read(img, at, &none, TexelMemory::Private { volatile })
            .expect("read");
        let x = f.composite_extract(t, &[0]).expect("x");
        store_at(&mut f, out, zero, x).expect("store");
        f.ret(None).expect("ret");
        let main = f.finish().expect("finish");
        b.entry(
            ExecModel::GLCompute,
            main,
            vec![ExecMode::LocalSize { x: 1, y: 1, z: 1 }],
        )
        .expect("entry");
        let m = b.finalize().expect("finalize");
        let mut robust = Image::new(tex, [1, 1, 1], 1, 1, 1);
        robust.robust = Some([0, 0, 0, 1]);
        let (o, _) =
            run_with_images(&m, &Config::default(), buffers(&[1]), vec![robust]).expect("run");
        let want = if volatile { 0 } else { 7 };
        assert_eq!(words(&o.buffers[0]), [want], "volatile {volatile}");
    }
}
