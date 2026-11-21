//! A set of regions written as a kept routine computes what the regions computed, a region that
//! differs stays in line, and a variable whose value carries from one run of its region to the next
//! is passed to the routine, not made its own. A region repeated more often than a search tries
//! extents against is still one routine.

use super::*;
use crate::budget::Budget;
use crate::device::profile::DeviceProfile;
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage};
use crate::spirv::builder::constructs::{ArmEnd, LoopSpec};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::GlobalId;
use crate::spirv::interp::{run, Buffer, Config, Resources};
use crate::spirv::ir::{natural_layout, BinOp, ExecMode, ExecModel, LayoutRule, MemAccess};
use crate::spirv::testing::{assert_valid, committed_profiles};

fn meter() -> Meter {
    Meter::new(&Budget::default())
}

fn u32_ty(b: &mut ModuleBuilder) -> TypeId {
    b.ty(Type::Int {
        width: 32,
        signed: false,
    })
    .expect("u32")
}

fn c32(f: &mut FunctionBuilder<'_>, v: u32) -> Value {
    let c = f.u32_const(v).expect("constant");
    f.module().const_value(c).expect("value")
}

fn slot(kind: ResourceKind, index: u32) -> ResourceSlot {
    ResourceSlot {
        stage: Stage::Kernel,
        kind,
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
    b.resource(
        slot(ResourceKind::Buffer, index),
        StorageClass::StorageBuffer,
        block,
    )
    .expect("resource")
}

fn elem(f: &mut FunctionBuilder<'_>, buf: GlobalId, i: Value) -> BResult<Value> {
    let zero = c32(f, 0);
    let base = f.module().global_value(buf)?;
    f.access_chain(base, &[zero, i])
}

fn load_at(f: &mut FunctionBuilder<'_>, buf: GlobalId, i: u32) -> BResult<Value> {
    let i = c32(f, i);
    let p = elem(f, buf, i)?;
    f.load(p, MemAccess::default())
}

fn store_at(f: &mut FunctionBuilder<'_>, buf: GlobalId, i: Value, v: Value) -> BResult<()> {
    let p = elem(f, buf, i)?;
    f.store(p, v, MemAccess::default())
}

fn words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn resources(input: &[u32], out_words: usize) -> Resources {
    Resources {
        buffers: vec![
            Buffer {
                slot: Some(slot(ResourceKind::Buffer, 0)),
                address: None,
                bytes: input.iter().flat_map(|w| w.to_le_bytes()).collect(),
            },
            Buffer {
                slot: Some(slot(ResourceKind::Buffer, 1)),
                address: None,
                bytes: vec![0; out_words * 4],
            },
        ],
        push: Vec::new(),
        spec: Default::default(),
    }
}

fn moltenvk() -> DeviceProfile {
    committed_profiles()
        .expect("profiles")
        .into_iter()
        .find(|(s, _)| s == "moltenvk-m3")
        .expect("moltenvk-m3")
        .1
}

/// The eight operations of a region's chain, its fourth `fourth`.
fn chain(fourth: BinOp) -> [(BinOp, u32); 8] {
    [
        (BinOp::IMul, 3),
        (BinOp::BitwiseXor, 5),
        (BinOp::IAdd, 7),
        (fourth, 11),
        (BinOp::BitwiseXor, 13),
        (BinOp::IAdd, 17),
        (BinOp::IMul, 19),
        (BinOp::BitwiseXor, 23),
    ]
}

/// [`chain`] computed on `x`.
fn apply_chain(x: u32, fourth: BinOp) -> u32 {
    chain(fourth).iter().fold(x, |p, (op, k)| match op {
        BinOp::IMul => p.wrapping_mul(*k),
        BinOp::BitwiseXor => p ^ k,
        BinOp::IAdd => p.wrapping_add(*k),
        BinOp::ISub => p.wrapping_sub(*k),
        _ => panic!("no such chain operation"),
    })
}

/// Region `j`: `x = seed + 10 (j + 1) + dep` through a variable, [`chain`] on `x` (`p`), `y = x >
/// 25 ? 2 x : x + 1` through a selection's phi, a loop counting a variable initialized to 0 up to
/// 3, and `out[base + 2 + j] = y + count + p`. Returns `y`.
fn region(
    f: &mut FunctionBuilder<'_>,
    out: GlobalId,
    (seed, base, dep): (Value, Value, Value),
    j: u32,
    fourth: BinOp,
) -> BResult<Value> {
    let u32_ = u32_ty(f.module());
    let v = f.local_var(u32_, None)?;
    let vp = f.var_ptr(v)?;
    let add = c32(f, 10 * (j + 1));
    let x0 = f.binary(BinOp::IAdd, seed, add)?;
    let x0 = f.binary(BinOp::IAdd, x0, dep)?;
    f.store(vp, x0, MemAccess::default())?;
    let x = f.load(vp, MemAccess::default())?;
    let mut p = x;
    for (op, k) in chain(fourth) {
        let k = c32(f, k);
        p = f.binary(op, p, k)?;
    }
    let c25 = c32(f, 25);
    let big = f.binary(BinOp::UGreaterThan, x, c25)?;
    let y = f.if_(
        &mut (),
        big,
        Box::new(move |f, _| {
            let two = c32(f, 2);
            Ok(ArmEnd::FallThrough(vec![f.binary(BinOp::IMul, x, two)?]))
        }),
        Some(Box::new(move |f, _| {
            let one = c32(f, 1);
            Ok(ArmEnd::FallThrough(vec![f.binary(BinOp::IAdd, x, one)?]))
        })),
    )?;
    let y = *y.0.first().expect("phi");
    let zero = f.u32_const(0)?;
    let cnt = f.local_var(u32_, Some(zero))?;
    let cp = f.var_ptr(cnt)?;
    f.loop_(
        &mut (),
        LoopSpec::default(),
        Box::new(move |f, _, lc| {
            let c = f.load(cp, MemAccess::default())?;
            let three = c32(f, 3);
            let done = f.binary(BinOp::UGreaterThanEqual, c, three)?;
            f.if_(
                &mut (),
                done,
                Box::new(|f, _| {
                    lc.break_(f)?;
                    Ok(ArmEnd::Exits)
                }),
                None,
            )?;
            let one = c32(f, 1);
            let c1 = f.binary(BinOp::IAdd, c, one)?;
            f.store(cp, c1, MemAccess::default())
        }),
    )?;
    let n = f.load(cp, MemAccess::default())?;
    let r = f.binary(BinOp::IAdd, y, n)?;
    let r = f.binary(BinOp::IAdd, r, p)?;
    let at = c32(f, 2 + j);
    let idx = f.binary(BinOp::IAdd, base, at)?;
    store_at(f, out, idx, r)?;
    Ok(y)
}

/// One [`region`] after another, region `j`'s chain's fourth operation `fourths[j]`, reading `in =
/// [seed, base, seed2]` and writing `out`; then `out[0] = y + 100` of the last region. With
/// `cross`, each region but the first adds the previous region's `y` where the first adds `seed2`.
fn kernel(fourths: &[BinOp], cross: bool) -> TypedModule {
    let mut b = ModuleBuilder::new();
    let input = buffer(&mut b, 0);
    let out = buffer(&mut b, 1);
    let void = b.ty(Type::Void).expect("void");
    let mut f = b.function(void).expect("function");
    let body = |f: &mut FunctionBuilder<'_>| -> BResult<()> {
        let seed = load_at(f, input, 0)?;
        let base = load_at(f, input, 1)?;
        let seed2 = load_at(f, input, 2)?;
        let mut y = seed2;
        for (j, fourth) in fourths.iter().enumerate() {
            let dep = if cross && j > 0 { y } else { seed2 };
            y = region(f, out, (seed, base, dep), j as u32, *fourth)?;
        }
        let c100 = c32(f, 100);
        let z = f.binary(BinOp::IAdd, y, c100)?;
        let zero = c32(f, 0);
        store_at(f, out, zero, z)?;
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

/// The words [`kernel`] writes, `base` 0.
fn expected(fourths: &[BinOp], cross: bool, seed: u32, seed2: u32) -> Vec<u32> {
    let mut out = vec![0u32; 2 + fourths.len()];
    let mut y = seed2;
    for (j, fourth) in fourths.iter().enumerate() {
        let dep = if cross && j > 0 { y } else { seed2 };
        let x = seed.wrapping_add(10 * (j as u32 + 1)).wrapping_add(dep);
        let p = apply_chain(x, *fourth);
        y = if x > 25 {
            x.wrapping_mul(2)
        } else {
            x.wrapping_add(1)
        };
        out[2 + j] = y.wrapping_add(3).wrapping_add(p);
    }
    out[0] = y.wrapping_add(100);
    out
}

/// Each kept routine of `m`, with the calls of it in the entry.
fn routines(m: &TypedModule) -> Vec<(FuncId, usize)> {
    let entry = &m.functions()[m.entry().func.index()];
    let mut out = Vec::new();
    for (i, f) in m.functions().iter().enumerate() {
        if !f.kept {
            continue;
        }
        let id = FuncId::new(i as u32);
        let calls = entry
            .blocks
            .iter()
            .flat_map(|b| &m.blocks()[b.index()].insts)
            .filter(|n| matches!(n.inst, Inst::Call { func, .. } if func == id))
            .count();
        out.push((id, calls));
    }
    out
}

/// The `OpFunction`s (opcode 54) of a module's words whose control is `DontInline`.
fn dont_inline(words: &[u32]) -> usize {
    let mut out = 0;
    let mut at = 5;
    while at < words.len() {
        let (count, op) = ((words[at] >> 16) as usize, words[at] & 0xFFFF);
        if op == 54 && words[at + 3] & 0x2 != 0 {
            out += 1;
        }
        at += count.max(1);
    }
    out
}

/// Run `before` and `after` over `input` and check both write `want`.
fn same_words(before: &TypedModule, after: &TypedModule, input: &[u32], want: &[u32]) {
    let n = want.len();
    let a = run(before, &Config::default(), resources(input, n)).expect("run in line");
    let b = run(after, &Config::default(), resources(input, n)).expect("run as calls");
    assert_eq!(words(&a.buffers[1]), want, "in line, input {input:?}");
    assert_eq!(words(&b.buffers[1]), want, "as calls, input {input:?}");
}

/// Three regions that are the same code but their constants, each with its own variables (one
/// with an initializer), a selection merging a value by phi and a counting loop, become calls of
/// one kept routine, which takes their selection and loop; the last region's `y`, read after it,
/// is the routine's result. The module is valid, declares the routine `DontInline`, and writes
/// what the regions wrote, for inputs on both sides of the selection. Each region is under the
/// size measured to run no slower as a routine, so the pass as compiles run it leaves them.
#[test]
fn three_equal_regions_become_calls_of_one_kept_routine_that_computes_the_same() {
    let fourths = [BinOp::IMul; 3];
    let orig = kernel(&fourths, false);
    let mut kept = orig.clone();
    assert_eq!(outline_entry(&mut kept, &mut meter()).expect("outline"), 0);
    let mut m = orig.clone();
    assert_eq!(outline(&mut m, 1, &mut meter()).expect("outline"), 1);
    assert_eq!(routines(&m), [(FuncId::new(1), 3)]);
    let routine = &m.functions()[1];
    assert_eq!(routine.vars.len(), 2, "the region's two variables");
    assert_eq!(orig.loops().len(), 3);
    assert_eq!(m.loops().len(), 1, "the routine's loop");
    let spirv = assert_valid(m.clone(), &moltenvk());
    assert_eq!(dont_inline(&spirv), 1);
    for (seed, seed2) in [(1u32, 2u32), (20, 0), (0, 0), (7, 30)] {
        let want = expected(&fourths, false, seed, seed2);
        same_words(&orig, &m, &[seed, 0, seed2], &want);
    }
    // Nothing is left to write as a routine.
    assert_eq!(outline(&mut m, 1, &mut meter()).expect("outline again"), 0);
}

/// A chain: each region but the first reads the previous region's `y` where the first reads
/// `seed2`. Every region's `y` is the routine's result, and each call after the first is passed
/// the call before's.
#[test]
fn a_chain_of_regions_passes_each_call_the_result_of_the_one_before() {
    let fourths = [BinOp::IMul; 3];
    let orig = kernel(&fourths, true);
    let mut m = orig.clone();
    assert_eq!(outline(&mut m, 1, &mut meter()).expect("outline"), 1);
    assert_eq!(routines(&m), [(FuncId::new(1), 3)]);
    assert_valid(m.clone(), &moltenvk());
    for (seed, seed2) in [(3u32, 4u32), (20, 0), (0, 0), (1, 30)] {
        let want = expected(&fourths, true, seed, seed2);
        same_words(&orig, &m, &[seed, 0, seed2], &want);
    }
}

/// Of three regions, the middle one's chain subtracts where the others' multiply: the first and
/// last become calls (their keys agree one item past the first region's end, where the next
/// region's first addition reads a value from before it and the kernel's last reads the last
/// region's `y`, so the set is compared again without that item), and the middle one stays in
/// line. Two regions that differ so become nothing.
#[test]
fn a_region_that_differs_stays_in_line_beside_the_calls_of_the_equal_ones() {
    let fourths = [BinOp::IMul, BinOp::ISub, BinOp::IMul];
    let orig = kernel(&fourths, false);
    let mut m = orig.clone();
    assert_eq!(outline(&mut m, 1, &mut meter()).expect("outline"), 1);
    assert_eq!(routines(&m), [(FuncId::new(1), 2)]);
    assert_eq!(
        m.loops().len(),
        2,
        "the routine's loop and the middle region's"
    );
    assert_valid(m.clone(), &moltenvk());
    for (seed, seed2) in [(1u32, 2u32), (20, 0), (7, 30)] {
        let want = expected(&fourths, false, seed, seed2);
        same_words(&orig, &m, &[seed, 0, seed2], &want);
    }
    let mut two = kernel(&[BinOp::IMul, BinOp::ISub], false);
    assert_eq!(outline(&mut two, 1, &mut meter()).expect("outline"), 0);
}

/// Eighty equal regions, more repeats than a window's extents are tried against, are one routine
/// called eighty times, made by one search: a search that took only the repeats it tried would
/// leave the rest for another search and another routine of the same code, and a function of
/// many repeats would be searched once per few dozen of them.
#[test]
fn a_region_repeated_past_the_repeats_tried_is_one_routine_called_at_every_repeat() {
    let fourths = [BinOp::IMul; 80];
    let orig = kernel(&fourths, false);
    let mut m = orig.clone();
    assert_eq!(outline(&mut m, 1, &mut meter()).expect("outline"), 1);
    assert_eq!(routines(&m), [(FuncId::new(1), 80)]);
    assert_valid(m.clone(), &moltenvk());
    for (seed, seed2) in [(1u32, 2u32), (20, 0)] {
        let want = expected(&fourths, false, seed, seed2);
        same_words(&orig, &m, &[seed, 0, seed2], &want);
    }
}

/// Region `j` of the looped kernel: `a = acc`, `t = a * 5` through a variable it stores first,
/// [`chain`] on `t` (`p`), `acc = a + p`, and `out[2 + j] = acc`.
fn looped_region(f: &mut FunctionBuilder<'_>, out: GlobalId, acc: VarId, j: u32) -> BResult<()> {
    let u32_ = u32_ty(f.module());
    let ap = f.var_ptr(acc)?;
    let a = f.load(ap, MemAccess::default())?;
    let tmp = f.local_var(u32_, None)?;
    let tp = f.var_ptr(tmp)?;
    let five = c32(f, 5);
    let t0 = f.binary(BinOp::IMul, a, five)?;
    f.store(tp, t0, MemAccess::default())?;
    let t = f.load(tp, MemAccess::default())?;
    let mut p = t;
    for (op, k) in chain(BinOp::IMul) {
        let k = c32(f, k);
        p = f.binary(op, p, k)?;
    }
    let s = f.binary(BinOp::IAdd, a, p)?;
    f.store(ap, s, MemAccess::default())?;
    let at = c32(f, 2 + j);
    store_at(f, out, at, s)
}

/// A loop of four trips running `k` [`looped_region`]s one after another, region `j` on its own
/// accumulator, a variable initialized to `j`.
fn looped_kernel(k: u32) -> TypedModule {
    let mut b = ModuleBuilder::new();
    let input = buffer(&mut b, 0);
    let out = buffer(&mut b, 1);
    let void = b.ty(Type::Void).expect("void");
    let mut f = b.function(void).expect("function");
    let body = |f: &mut FunctionBuilder<'_>| -> BResult<()> {
        let u32_ = u32_ty(f.module());
        let first = load_at(f, input, 0)?;
        let mut accs = Vec::new();
        for j in 0..k {
            let c = f.u32_const(j)?;
            accs.push(f.local_var(u32_, Some(c))?);
        }
        let zero = f.u32_const(0)?;
        let it = f.local_var(u32_, Some(zero))?;
        let ip = f.var_ptr(it)?;
        f.loop_(
            &mut (),
            LoopSpec::default(),
            Box::new(move |f, _, lc| {
                let c = f.load(ip, MemAccess::default())?;
                let four = c32(f, 4);
                let done = f.binary(BinOp::UGreaterThanEqual, c, four)?;
                f.if_(
                    &mut (),
                    done,
                    Box::new(|f, _| {
                        lc.break_(f)?;
                        Ok(ArmEnd::Exits)
                    }),
                    None,
                )?;
                for (j, acc) in accs.iter().enumerate() {
                    looped_region(f, out, *acc, j as u32)?;
                }
                let one = c32(f, 1);
                let c1 = f.binary(BinOp::IAdd, c, one)?;
                f.store(ip, c1, MemAccess::default())
            }),
        )?;
        let zero = c32(f, 0);
        store_at(f, out, zero, first)?;
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

/// The words [`looped_kernel`] writes for `in[0] = first`.
fn looped_expected(k: u32, first: u32) -> Vec<u32> {
    let mut out = vec![0u32; 2 + k as usize];
    let mut accs: Vec<u32> = (0..k).collect();
    for _ in 0..4 {
        for (j, acc) in accs.iter_mut().enumerate() {
            let a = *acc;
            *acc = a.wrapping_add(apply_chain(a.wrapping_mul(5), BinOp::IMul));
            out[2 + j] = *acc;
        }
    }
    out[0] = first;
    out
}

/// Two regions in a loop, each on its own accumulator: the accumulator, read before it is written,
/// carries its value from one trip to the next, so the routine is passed it as a pointer; the
/// variable each region stores before it reads it becomes the routine's own. Both regions are
/// one block each, and the module writes what they wrote.
#[test]
fn a_variable_read_before_it_is_written_in_a_loop_is_passed_to_the_routine() {
    let orig = looped_kernel(2);
    let mut m = orig.clone();
    assert_eq!(outline(&mut m, 1, &mut meter()).expect("outline"), 1);
    assert_eq!(routines(&m), [(FuncId::new(1), 2)]);
    let routine = &m.functions()[1];
    assert_eq!(routine.vars.len(), 1, "the variable stored first");
    let pointers = routine
        .params
        .iter()
        .filter(|p| {
            let t = m.value(**p).expect("parameter").ty;
            matches!(m.types().get(t).expect("type"), Type::Pointer { .. })
        })
        .count();
    assert_eq!(pointers, 1, "the accumulator");
    assert_valid(m.clone(), &moltenvk());
    for first in [0u32, 9] {
        same_words(&orig, &m, &[first], &looped_expected(2, first));
    }
}
