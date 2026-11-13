//! A rolled run computes what the run computed, and a run that is not the same code stays as it is.

use super::*;
use crate::budget::Budget;
use crate::device::profile::DeviceProfile;
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage};
use crate::spirv::builder::constructs::{ArmEnd, LoopSpec};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::interp::{run, run_with_images, Buffer, Config, Image, Resources};
use crate::spirv::ir::{
    natural_layout, Dim, ExecMode, ExecModel, ImageOperands, ImageType, ImageUsage, LayoutRule,
    TexelMemory,
};
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

/// Region `j` of the run kernel: `x = seed + 10 (j + 1) + dep` through a variable, `y = x > 25
/// ? 2 x : x + 1` through a selection's phi, a loop counting a variable initialized to 0 up to
/// 3, and `out[base + 2 + j] = y + count`. Returns `y`.
fn region(
    f: &mut FunctionBuilder<'_>,
    out: GlobalId,
    ins: (Value, Value, Value),
    j: u32,
) -> BResult<Value> {
    region_adding(f, out, ins, j, 10 * (j + 1))
}

/// [`region`], adding `add` where it adds `10 (j + 1)`.
fn region_adding(
    f: &mut FunctionBuilder<'_>,
    out: GlobalId,
    (seed, base, dep): (Value, Value, Value),
    j: u32,
    add: u32,
) -> BResult<Value> {
    let u32_ = u32_ty(f.module());
    let v = f.local_var(u32_, None)?;
    let vp = f.var_ptr(v)?;
    let add = c32(f, add);
    let x0 = f.binary(BinOp::IAdd, seed, add)?;
    let x0 = f.binary(BinOp::IAdd, x0, dep)?;
    f.store(vp, x0, MemAccess::default())?;
    let x = f.load(vp, MemAccess::default())?;
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
    let at = c32(f, 2 + j);
    let idx = f.binary(BinOp::IAdd, base, at)?;
    store_at(f, out, idx, r)?;
    Ok(y)
}

/// `k` regions ([`region`]) reading `in = [seed, base, seed2]` and writing `out`; then
/// `out[0] = y + 100` of the last region. With `cross`, each region but the first adds the
/// previous region's `y` where the first adds `seed2`.
fn run_kernel(k: u32, cross: bool) -> TypedModule {
    kernel_adding(k, cross, |j| 10 * (j + 1))
}

/// [`run_kernel`] whose region `j` adds `add(j)` where it adds `10 (j + 1)`.
fn kernel_adding(k: u32, cross: bool, add: fn(u32) -> u32) -> TypedModule {
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
        for j in 0..k {
            let dep = if cross && j > 0 { y } else { seed2 };
            y = region_adding(f, out, (seed, base, dep), j, add(j))?;
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

/// The words the run kernel writes: per region `y + 3` at `2 + j`, and the last `y + 100` at 0.
fn expected(k: u32, cross: bool, seed: u32, seed2: u32) -> Vec<u32> {
    expected_adding(k, cross, seed, seed2, |j| 10 * (j + 1))
}

/// [`expected`] of [`kernel_adding`].
fn expected_adding(k: u32, cross: bool, seed: u32, seed2: u32, add: fn(u32) -> u32) -> Vec<u32> {
    let mut out = vec![0u32; 2 + k as usize];
    let mut y = seed2;
    for j in 0..k {
        let dep = if cross && j > 0 { y } else { seed2 };
        let x = seed + add(j) + dep;
        y = if x > 25 { 2 * x } else { x + 1 };
        out[2 + j as usize] = y + 3;
    }
    out[0] = y + 100;
    out
}

/// The loop control of each `OpLoopMerge` (opcode 246) in a module's words.
fn loop_controls(words: &[u32]) -> Vec<u32> {
    let mut out = Vec::new();
    let mut at = 5;
    while at < words.len() {
        let (count, op) = ((words[at] >> 16) as usize, words[at] & 0xFFFF);
        if op == 246 {
            out.push(words[at + 3]);
        }
        at += count.max(1);
    }
    out
}

fn entry_insts(m: &TypedModule) -> usize {
    m.functions()[m.entry().func.index()]
        .blocks
        .iter()
        .map(|b| m.blocks()[b.index()].insts.len())
        .sum()
}

/// Three regions that differ in their constants, each with its own variables (one with an
/// initializer), a selection merging a value by phi, and a counting loop, roll into one loop of
/// three trips around one region; the last region's `y`, read after the run, is the first's. The
/// module is valid, holds one region's instructions, and writes what the three regions wrote, for
/// inputs on both sides of the selection.
#[test]
fn three_regions_roll_into_one_loop_that_computes_the_same() {
    let orig = run_kernel(3, false);
    let mut m = orig.clone();
    let mut mt = meter();
    let w0 = entry_weight(&m, &mut mt).expect("weight");
    let loops = roll(&mut m, 0, 1, &mut mt).expect("roll");
    assert_eq!(loops, 1, "one run rolled");
    let w1 = entry_weight(&m, &mut mt).expect("weight");
    let one = entry_weight(&run_kernel(1, false), &mut mt).expect("weight");
    // One region, and the loop: six blocks, the trip index's load, store, count, and tests, the
    // two differing constants' strides, and the counter's initializer.
    assert!(
        w1 < w0 && w1 <= one + 20,
        "the entry holds one region: {w0} -> {w1} (one region: {one})"
    );
    // The three regions' loops become the rolled loop and the one region's loop.
    assert_eq!(orig.loops().len(), 3);
    assert_eq!(m.loops().len(), 2);
    assert!(entry_insts(&m) < entry_insts(&orig));
    // The rolled loop asks `DontUnroll`, so a downstream compiler keeps it; the region's own
    // loop asks nothing.
    let spirv = assert_valid(m.clone(), &moltenvk());
    let mut controls = loop_controls(&spirv);
    controls.sort_unstable();
    assert_eq!(controls, [0, 2]);
    assert_eq!(
        m.loops()
            .iter()
            .filter(|l| l.unrolling == Unrolling::DontUnroll)
            .count(),
        1,
        "the rolled loop"
    );
    for (seed, seed2) in [(1u32, 2u32), (20, 0), (0, 0), (7, 30)] {
        let input = [seed, 0, seed2];
        let a = run(&orig, &Config::default(), resources(&input, 5)).expect("run unrolled");
        let b = run(&m, &Config::default(), resources(&input, 5)).expect("run rolled");
        let want = expected(3, false, seed, seed2);
        assert_eq!(words(&a.buffers[1]), want, "the unrolled kernel");
        assert_eq!(words(&b.buffers[1]), want, "the rolled kernel");
    }
    // Nothing is left to roll.
    let again = roll(&mut m, 0, 1, &mut mt).expect("roll again");
    assert_eq!(again, 0);
}

/// A chain: each region but the first reads the previous region's `y` where the first reads
/// `seed2`, from before the run; the regions roll into one loop that carries `y` from trip to trip
/// in a variable, holds one region, and writes what the regions wrote, for inputs on both sides of
/// the selection; six regions too.
#[test]
fn a_chain_of_regions_rolls_carrying_the_previous_trips_value() {
    for k in [3u32, 6] {
        let orig = run_kernel(k, true);
        let mut m = orig.clone();
        let mut mt = meter();
        let loops = roll(&mut m, 0, 1, &mut mt).expect("roll");
        assert_eq!(loops, 1, "{k} regions: one run rolled");
        let (w0, w1) = (
            entry_weight(&orig, &mut mt).expect("weight"),
            entry_weight(&m, &mut mt).expect("weight"),
        );
        // The regions' counting loops become the rolled loop and one region's.
        assert_eq!(orig.loops().len(), k as usize);
        assert_eq!(m.loops().len(), 2, "{k} regions: one region is left");
        assert!(w1 < w0, "{k} regions: {w0} -> {w1}");
        assert_valid(m.clone(), &moltenvk());
        for (seed, seed2) in [(3u32, 4u32), (20, 0), (0, 0), (1, 30)] {
            let input = [seed, 0, seed2];
            let words_out = 2 + k as usize;
            let a = run(&orig, &Config::default(), resources(&input, words_out)).expect("unrolled");
            let b = run(&m, &Config::default(), resources(&input, words_out)).expect("rolled");
            let want = expected(k, true, seed, seed2);
            assert_eq!(
                words(&a.buffers[1]),
                want,
                "{k} regions: the unrolled kernel"
            );
            assert_eq!(words(&b.buffers[1]), want, "{k} regions: the rolled kernel");
        }
    }
}

/// `k` regions ([`region`]) reading `in = [seed, base, a0, a1, a2, a3]` and writing `out`, as
/// [`run_kernel`]'s, each adding a lane of `v = (a0, a1, a2, a3)`, built before the run, where
/// [`run_kernel`]'s add `seed2`: region `j` reads lane `j % 4` of `v`, and, with `shuffled`, lane
/// 0 of `v`'s lanes `(j + 1) % 4` and `j % 4` shuffled, plus that read's lane 1.
fn lane_kernel(k: u32, shuffled: bool) -> TypedModule {
    let mut b = ModuleBuilder::new();
    let input = buffer(&mut b, 0);
    let out = buffer(&mut b, 1);
    let u32_ = u32_ty(&mut b);
    let uvec4 = b.ty(Type::Vector { elem: u32_, len: 4 }).expect("uvec4");
    let void = b.ty(Type::Void).expect("void");
    let mut f = b.function(void).expect("function");
    let body = |f: &mut FunctionBuilder<'_>| -> BResult<()> {
        let seed = load_at(f, input, 0)?;
        let base = load_at(f, input, 1)?;
        let mut lanes = Vec::new();
        for i in 0..4 {
            lanes.push(load_at(f, input, 2 + i)?);
        }
        let v = f.composite_construct(uvec4, &lanes)?;
        let mut y = seed;
        for j in 0..k {
            let lane = f.composite_extract(v, &[j % 4])?;
            let dep = if shuffled {
                let pair = f.vector_shuffle(v, v, &[(j + 1) % 4, j % 4])?;
                let first = f.composite_extract(pair, &[0])?;
                let second = f.composite_extract(pair, &[1])?;
                let both = f.binary(BinOp::IAdd, first, second)?;
                f.binary(BinOp::IAdd, lane, both)?
            } else {
                lane
            };
            y = region(f, out, (seed, base, dep), j)?;
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

/// The words [`lane_kernel`] writes for `lanes`.
fn lane_expected(k: u32, shuffled: bool, seed: u32, lanes: [u32; 4]) -> Vec<u32> {
    let mut out = vec![0u32; 2 + k as usize];
    let mut y = seed;
    for j in 0..k {
        let lane = lanes[(j % 4) as usize];
        let dep = match shuffled {
            true => lane
                .wrapping_add(lanes[((j + 1) % 4) as usize])
                .wrapping_add(lanes[(j % 4) as usize]),
            false => lane,
        };
        let x = seed.wrapping_add(10 * (j + 1)).wrapping_add(dep);
        y = if x > 25 {
            x.wrapping_mul(2)
        } else {
            x.wrapping_add(1)
        };
        out[2 + j as usize] = y.wrapping_add(3);
    }
    out[0] = y.wrapping_add(100);
    out
}

/// Lane reads that differ by region: regions that read their own lane of a vector built before the
/// run, and their own shuffle of it, roll into one loop whose reads are computed before it and
/// chosen by trip; the module holds one region and writes what the regions wrote.
#[test]
fn regions_reading_their_own_lanes_roll_through_lifted_reads() {
    for (k, shuffled) in [(4u32, false), (3, true), (6, true)] {
        let what = format!("{k} regions, shuffled {shuffled}");
        let orig = lane_kernel(k, shuffled);
        let mut m = orig.clone();
        let mut mt = meter();
        let loops = roll(&mut m, 0, 1, &mut mt).expect("roll");
        assert_eq!(loops, 1, "{what}: one run rolled");
        let (w0, w1) = (
            entry_weight(&orig, &mut mt).expect("weight"),
            entry_weight(&m, &mut mt).expect("weight"),
        );
        assert_eq!(orig.loops().len(), k as usize);
        assert_eq!(m.loops().len(), 2, "{what}: one region is left");
        assert!(w1 < w0, "{what}: {w0} -> {w1}");
        assert_valid(m.clone(), &moltenvk());
        for (seed, lanes) in [(1u32, [1u32, 2, 3, 4]), (20, [0, 40, 0, 7]), (0, [9; 4])] {
            let input = [seed, 0, lanes[0], lanes[1], lanes[2], lanes[3]];
            let words_out = 2 + k as usize;
            let a = run(&orig, &Config::default(), resources(&input, words_out)).expect("unrolled");
            let b = run(&m, &Config::default(), resources(&input, words_out)).expect("rolled");
            let want = lane_expected(k, shuffled, seed, lanes);
            assert_eq!(words(&a.buffers[1]), want, "{what}: the unrolled kernel");
            assert_eq!(words(&b.buffers[1]), want, "{what}: the rolled kernel");
        }
    }
}

/// Below the size it rolls above, and with regions lighter than the smallest worth a loop, nothing
/// changes; neither does a module below [`ROLL_ABOVE`], whatever the profile.
#[test]
fn small_modules_stay_as_they_are() {
    let orig = run_kernel(3, false);
    let mut m = orig.clone();
    assert_eq!(roll(&mut m, u64::MAX, 1, &mut meter()).expect("roll"), 0);
    assert_eq!(roll(&mut m, 0, u64::MAX, &mut meter()).expect("roll"), 0);
    assert!(entry_weight(&m, &mut meter()).expect("weight") <= ROLL_ABOVE);
    assert_eq!(roll_entry(&mut m, &mut meter()).expect("roll"), 0);
    assert_eq!(m.blocks(), orig.blocks());
}

/// Two regions, each loading its own storage image and, when `x = seed + 5 (j + 1)` is not
/// zero, writing `(x, j, 0, 0)` at `(j, 0)` of it.
fn image_kernel() -> TypedModule {
    let mut b = ModuleBuilder::new();
    let input = buffer(&mut b, 0);
    let u32_ = u32_ty(&mut b);
    let ty = b
        .ty(Type::Image(ImageType {
            sampled_type: u32_,
            dim: Dim::D2,
            depth: false,
            arrayed: false,
            multisampled: false,
            usage: ImageUsage::Storage,
            format: crate::spv::ImageFormat::Rgba32ui,
        }))
        .expect("image type");
    let images: Vec<GlobalId> = (0..2)
        .map(|i| {
            b.resource(
                slot(ResourceKind::Texture, i),
                StorageClass::UniformConstant,
                ty,
            )
            .expect("image")
        })
        .collect();
    let uvec2 = b.ty(Type::Vector { elem: u32_, len: 2 }).expect("uvec2");
    let uvec4 = b.ty(Type::Vector { elem: u32_, len: 4 }).expect("uvec4");
    let void = b.ty(Type::Void).expect("void");
    let mut f = b.function(void).expect("function");
    let body = |f: &mut FunctionBuilder<'_>| -> BResult<()> {
        let seed = load_at(f, input, 0)?;
        for (j, img) in images.iter().enumerate() {
            let j = j as u32;
            let add = c32(f, 5 * (j + 1));
            let x = f.binary(BinOp::IAdd, seed, add)?;
            let p = f.module().global_value(*img)?;
            let t = f.load(p, MemAccess::default())?;
            let zero = c32(f, 0);
            let jj = c32(f, j);
            let jv = f.binary(BinOp::IAdd, seed, jj)?;
            let jv = f.binary(BinOp::ISub, jv, seed)?;
            let at = f.composite_construct(uvec2, &[jv, zero])?;
            let texel = f.composite_construct(uvec4, &[x, jv, zero, zero])?;
            let nz = f.binary(BinOp::INotEqual, x, zero)?;
            f.if_(
                &mut (),
                nz,
                Box::new(move |f, _| {
                    f.write(
                        t,
                        at,
                        texel,
                        &ImageOperands::default(),
                        TexelMemory::PRIVATE,
                    )?;
                    Ok(ArmEnd::FallThrough(Vec::new()))
                }),
                None,
            )?;
        }
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

/// Two regions that write two images roll into a loop whose image write loads its trip's image in a
/// switch on the trip; the images receive what the two regions wrote.
#[test]
fn regions_writing_their_own_images_roll_through_a_switch() {
    let orig = image_kernel();
    let mut m = orig.clone();
    let loops = roll(&mut m, 0, 1, &mut meter()).expect("roll");
    assert_eq!(loops, 1);
    let switches = m
        .blocks()
        .iter()
        .filter(|b| matches!(b.term, Terminator::Switch { .. }))
        .count();
    assert_eq!(switches, 1, "one switch picks the trip's image");
    assert_valid(m.clone(), &moltenvk());
    let images = || {
        (0..2)
            .map(|i| Image::new(slot(ResourceKind::Texture, i), [2, 1, 1], 1, 1, 1))
            .collect::<Vec<_>>()
    };
    for seed in [0u32, 3, u32::MAX - 4] {
        let res = || Resources {
            buffers: vec![Buffer {
                slot: Some(slot(ResourceKind::Buffer, 0)),
                address: None,
                bytes: seed.to_le_bytes().to_vec(),
            }],
            push: Vec::new(),
            spec: Default::default(),
        };
        let (_, a) = run_with_images(&orig, &Config::default(), res(), images()).expect("unrolled");
        let (_, b) = run_with_images(&m, &Config::default(), res(), images()).expect("rolled");
        let texels = |imgs: &[Image]| -> Vec<Vec<[u32; 4]>> {
            imgs.iter().map(|i| i.texels.clone()).collect()
        };
        assert_eq!(texels(&a), texels(&b), "seed {seed}");
        if seed == 3 {
            assert_eq!(a[0].texels[0], [8, 0, 0, 0]);
            assert_eq!(a[1].texels[1], [13, 1, 0, 0]);
        }
    }
}

/// `for_each_value_mut` visits the operands `for_each_value` visits, in the same order.
#[test]
fn mutable_operand_visits_match_the_shared_ones() {
    for m in [run_kernel(3, false), image_kernel(), lane_kernel(3, true)] {
        for b in m.blocks() {
            for node in &b.insts {
                let mut seen = Vec::new();
                node.inst.for_each_value(|v| seen.push(v));
                let mut inst = node.inst.clone();
                let mut visited = Vec::new();
                inst.for_each_value_mut(|v| {
                    visited.push(*v);
                    *v = Value::new(v.0 + 1_000_000);
                });
                assert_eq!(visited, seen);
                let mut after = Vec::new();
                inst.for_each_value(|v| after.push(v.0));
                assert_eq!(
                    after,
                    seen.iter().map(|v| v.0 + 1_000_000).collect::<Vec<_>>()
                );
            }
            let mut seen = Vec::new();
            b.term.for_each_value(|v| seen.push(v));
            let mut term = b.term.clone();
            let mut visited = Vec::new();
            term.for_each_value_mut(|v| visited.push(*v));
            assert_eq!(visited, seen);
        }
    }
}

/// The values the regions of [`nested_kernel`] read, loaded before its selections.
#[derive(Clone, Copy)]
struct Nested {
    seed: Value,
    base: Value,
    seed2: Value,
    obase: Value,
    v4: Value,
    v2: Value,
}

/// The regions of [`nested_kernel`], in the inner selection's arm.
fn nested_regions(
    f: &mut FunctionBuilder<'_>,
    (input, out): (GlobalId, GlobalId),
    n: Nested,
    k: u32,
    bound: bool,
) -> BResult<()> {
    let u32_ = u32_ty(f.module());
    let mut y = n.seed2;
    for j in 0..k {
        let lane = if j < 4 {
            f.composite_extract(n.v4, &[j])?
        } else {
            f.composite_extract(n.v2, &[j - 4])?
        };
        let var = f.local_var(u32_, None)?;
        let vp = f.var_ptr(var)?;
        let x = f.binary(BinOp::IAdd, n.seed, lane)?;
        let x = f.binary(BinOp::IAdd, x, y)?;
        f.store(vp, x, MemAccess::default())?;
        let c25 = c32(f, 25);
        let big = f.binary(BinOp::UGreaterThan, x, c25)?;
        let base = n.base;
        f.if_(
            &mut (),
            big,
            Box::new(move |f, _| {
                // The byte offset of the region's word: `base + 16 j`, the first region's
                // `base` itself; with `bound`, plus `x & 28`, a value of the region's own.
                let mut off = base;
                if bound {
                    let mask = c32(f, 28);
                    let t = f.binary(BinOp::BitwiseAnd, x, mask)?;
                    let t = if j > 0 {
                        let c = c32(f, 16 * j);
                        f.binary(BinOp::IAdd, t, c)?
                    } else {
                        t
                    };
                    off = f.binary(BinOp::IAdd, off, t)?;
                } else if j > 0 {
                    let c = c32(f, 16 * j);
                    off = f.binary(BinOp::IAdd, off, c)?;
                }
                let two = c32(f, 2);
                let word = f.binary(BinOp::ShiftRightLogical, off, two)?;
                let p = elem(f, input, word)?;
                let d = f.load(p, MemAccess::default())?;
                let x2 = f.binary(BinOp::IMul, x, two)?;
                let r = f.binary(BinOp::IAdd, x2, d)?;
                f.store(vp, r, MemAccess::default())?;
                Ok(ArmEnd::FallThrough(Vec::new()))
            }),
            None,
        )?;
        y = f.load(vp, MemAccess::default())?;
        let at = c32(f, 2 + j);
        let idx = f.binary(BinOp::IAdd, n.obase, at)?;
        store_at(f, out, idx, y)?;
    }
    let c100 = c32(f, 100);
    let z = f.binary(BinOp::IAdd, y, c100)?;
    let zero = c32(f, 0);
    store_at(f, out, zero, z)
}

/// `k` (up to 6) regions inside two selections, the shape of the consumer's large fragment
/// functions, reading `in = [seed, base, gate, seed2, obase, a0, a1, a2, a3, data...]` and writing
/// `out`. Where `gate != 0` and then `seed != 7`, region `j` stores `x = seed + lane + y` to a
/// variable of its own, `lane` lane `j` of `(a0, a1, a2, a3)` or lane `j - 4` of `(a2, a3)`, and
/// `y` the previous region's result (the first region's: `seed2`); where `x > 25` it stores `2 x +
/// in[(base + 16 j) / 4]` there instead, the byte offset computed in the selection's arm (the first
/// region's is `base` itself, with no add); with `bound`, the offset also adds `x & 28`, before the
/// region's constant, so the add the first region lacks reads a value of the region. Its result
/// `y`, the variable's value, goes to `out[obase + 2 + j]`, and the last `y + 100` to `out[0]`.
fn nested_kernel(k: u32, bound: bool) -> TypedModule {
    let mut b = ModuleBuilder::new();
    let input = buffer(&mut b, 0);
    let out = buffer(&mut b, 1);
    let u32_ = u32_ty(&mut b);
    let uvec4 = b.ty(Type::Vector { elem: u32_, len: 4 }).expect("uvec4");
    let uvec2 = b.ty(Type::Vector { elem: u32_, len: 2 }).expect("uvec2");
    let void = b.ty(Type::Void).expect("void");
    let mut f = b.function(void).expect("function");
    let body = |f: &mut FunctionBuilder<'_>| -> BResult<()> {
        let seed = load_at(f, input, 0)?;
        let base = load_at(f, input, 1)?;
        let gate = load_at(f, input, 2)?;
        let seed2 = load_at(f, input, 3)?;
        let obase = load_at(f, input, 4)?;
        let mut lanes = Vec::new();
        for i in 0..4 {
            lanes.push(load_at(f, input, 5 + i)?);
        }
        let v4 = f.composite_construct(uvec4, &lanes)?;
        let v2 = f.composite_construct(uvec2, &lanes[2..])?;
        let n = Nested {
            seed,
            base,
            seed2,
            obase,
            v4,
            v2,
        };
        let zero = c32(f, 0);
        let open = f.binary(BinOp::INotEqual, gate, zero)?;
        f.if_(
            &mut (),
            open,
            Box::new(move |f, _| {
                let seven = c32(f, 7);
                let go = f.binary(BinOp::INotEqual, n.seed, seven)?;
                f.if_(
                    &mut (),
                    go,
                    Box::new(move |f, _| {
                        nested_regions(f, (input, out), n, k, bound)?;
                        Ok(ArmEnd::FallThrough(Vec::new()))
                    }),
                    None,
                )?;
                Ok(ArmEnd::FallThrough(Vec::new()))
            }),
            None,
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

/// The words [`nested_kernel`] writes for `input` (`obase` 0).
fn nested_expected(k: u32, bound: bool, input: &[u32]) -> Vec<u32> {
    let (seed, base, gate, seed2) = (input[0], input[1], input[2], input[3]);
    let lanes = [input[5], input[6], input[7], input[8], input[7], input[8]];
    let mut out = vec![0u32; 2 + k as usize];
    if gate != 0 && seed != 7 {
        let mut y = seed2;
        for j in 0..k as usize {
            let x = seed.wrapping_add(lanes[j]).wrapping_add(y);
            y = x;
            if x > 25 {
                let mut off = base.wrapping_add(16 * j as u32);
                if bound {
                    off = off.wrapping_add(x & 28);
                }
                y = x.wrapping_mul(2).wrapping_add(input[(off >> 2) as usize]);
            }
            out[2 + j] = y;
        }
        out[0] = y.wrapping_add(100);
    }
    out
}

/// The entry's instructions.
fn insts(m: &TypedModule) -> impl Iterator<Item = &Inst> {
    m.functions()[m.entry().func.index()]
        .blocks
        .iter()
        .flat_map(|b| &m.blocks()[b.index()].insts)
        .map(|n| &n.inst)
}

/// The plain constant `v` is, if it is one.
fn word_of(m: &TypedModule, v: Value) -> Option<u64> {
    match m.value(v).ok()?.origin {
        ValueOrigin::Const(id) => m.consts.scalar_bits(id).map(|b| b.0),
        _ => None,
    }
}

/// The entry's `OpIMul`s by 2: one per region of [`nested_kernel`] written out (a stride's
/// multiply by its step is not one).
fn muls(m: &TypedModule) -> usize {
    insts(m)
        .filter(
            |i| matches!(i, Inst::Binary { op: BinOp::IMul, b, .. } if word_of(m, *b) == Some(2)),
        )
        .count()
}

/// The entry's `OpSelect`s.
fn selects(m: &TypedModule) -> usize {
    insts(m)
        .filter(|i| matches!(i, Inst::Select { .. }))
        .count()
}

/// Inputs of [`nested_kernel`]: both gates open and each closed, `x` on both sides of 25.
fn nested_inputs() -> Vec<Vec<u32>> {
    let mut out = Vec::new();
    for (seed, gate, seed2, lanes) in [
        (1u32, 1u32, 2u32, [1u32, 2, 3, 4]),
        (20, 5, 0, [0, 40, 0, 7]),
        (0, 1, 0, [9; 4]),
        (7, 1, 30, [1, 2, 3, 4]),
        (3, 0, 30, [1, 2, 3, 4]),
        (2, 1, 1, [30, 0, 100, 5]),
    ] {
        let mut w = vec![seed, 36, gate, seed2, 0];
        w.extend(lanes);
        w.extend((0..31u32).map(|i| 1000 + 7 * i));
        out.push(w);
    }
    out
}

/// Chained regions inside two selections, each storing its result to a variable of its own that the
/// next reads, and each but the first adding its constant to a byte offset from before the
/// selections, in its selection's arm: all of them roll into one loop inside the inner arm, the
/// adds computed before the loop and chosen by trip; the module is valid, holds one region, and
/// writes what the regions wrote, gates open or closed.
#[test]
fn a_chain_inside_two_selections_rolls_with_its_offsets_computed_before_the_loop() {
    for k in [3u32, 6] {
        let orig = nested_kernel(k, false);
        let mut m = orig.clone();
        let mut mt = meter();
        let loops = roll(&mut m, 0, 1, &mut mt).expect("roll");
        assert_eq!(loops, 1, "{k} regions: one run rolled");
        assert_eq!(muls(&orig), k as usize);
        assert_eq!(muls(&m), 1, "{k} regions: every region in the loop");
        assert_eq!(m.loops().len(), 1);
        assert_eq!(m.loops()[0].unrolling, Unrolling::DontUnroll);
        assert_valid(m.clone(), &moltenvk());
        for input in nested_inputs() {
            let want = nested_expected(k, false, &input);
            let words_out = 2 + k as usize;
            let a = run(&orig, &Config::default(), resources(&input, words_out)).expect("unrolled");
            let b = run(&m, &Config::default(), resources(&input, words_out)).expect("rolled");
            assert_eq!(
                words(&a.buffers[1]),
                want,
                "{k} regions: the unrolled kernel"
            );
            assert_eq!(words(&b.buffers[1]), want, "{k} regions: the rolled kernel");
        }
    }
}

/// The roll is exact: where the add the first region lacks reads a value of the region's own, it
/// does not float, the first region stays as it is, and the others roll.
#[test]
fn a_region_whose_extra_add_reads_its_own_value_stays_out_of_the_loop() {
    let orig = nested_kernel(6, true);
    let mut m = orig.clone();
    let loops = roll(&mut m, 0, 1, &mut meter()).expect("roll");
    assert_eq!(loops, 1);
    assert_eq!(muls(&orig), 6);
    assert_eq!(
        muls(&m),
        2,
        "the first region, and the loop of the other five"
    );
    assert_valid(m.clone(), &moltenvk());
    for input in nested_inputs() {
        let want = nested_expected(6, true, &input);
        let a = run(&orig, &Config::default(), resources(&input, 8)).expect("unrolled");
        let b = run(&m, &Config::default(), resources(&input, 8)).expect("rolled");
        assert_eq!(words(&a.buffers[1]), want, "the unrolled kernel");
        assert_eq!(words(&b.buffers[1]), want, "the rolled kernel");
    }
}

/// `k` regions reading `in = [seed, obase, sel_0, ...]`, each a switch on a selector loaded
/// before the run, one per region, so the run's first block ends in a switch whose selector
/// differs by trip: region `j` computes `x = seed + 10 (j + 1)`, then `y = 2 x` for selector 0,
/// `x + 7` for 1, and `x + 1` otherwise, and stores `y` to `out[obase + 2 + j]`.
fn switch_kernel(k: u32) -> TypedModule {
    let mut b = ModuleBuilder::new();
    let input = buffer(&mut b, 0);
    let out = buffer(&mut b, 1);
    let void = b.ty(Type::Void).expect("void");
    let mut f = b.function(void).expect("function");
    let body = |f: &mut FunctionBuilder<'_>| -> BResult<()> {
        let seed = load_at(f, input, 0)?;
        let obase = load_at(f, input, 1)?;
        let mut sels = Vec::with_capacity(k as usize);
        for j in 0..k {
            sels.push(load_at(f, input, 2 + j)?);
        }
        for (j, sel) in (0..k).zip(sels) {
            let add = c32(f, 10 * (j + 1));
            let x = f.binary(BinOp::IAdd, seed, add)?;
            let y = f.switch_(
                &mut (),
                sel,
                vec![
                    (
                        vec![0],
                        Box::new(move |f: &mut FunctionBuilder<'_>, _: &mut ()| {
                            let two = c32(f, 2);
                            Ok(ArmEnd::FallThrough(vec![f.binary(BinOp::IMul, x, two)?]))
                        }),
                    ),
                    (
                        vec![1],
                        Box::new(move |f: &mut FunctionBuilder<'_>, _: &mut ()| {
                            let seven = c32(f, 7);
                            Ok(ArmEnd::FallThrough(vec![f.binary(
                                BinOp::IAdd,
                                x,
                                seven,
                            )?]))
                        }),
                    ),
                ],
                Box::new(move |f: &mut FunctionBuilder<'_>, _: &mut ()| {
                    let one = c32(f, 1);
                    Ok(ArmEnd::FallThrough(vec![f.binary(BinOp::IAdd, x, one)?]))
                }),
            )?;
            let y = *y.0.first().expect("phi");
            let at = c32(f, 2 + j);
            let idx = f.binary(BinOp::IAdd, obase, at)?;
            store_at(f, out, idx, y)?;
        }
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

/// A differing operand of the run's first block's terminator: the switch moves into the loop's
/// first block with the region and takes its trip's selector there. The rolled module is valid and
/// stores what the regions stored, for every pattern of selectors. (Placing that operand once
/// overflowed an index.)
#[test]
fn a_switch_on_a_selector_from_before_the_run_takes_its_trips_selector() {
    let k = 4u32;
    let orig = switch_kernel(k);
    let mut m = orig.clone();
    let loops = roll(&mut m, 0, 1, &mut meter()).expect("roll");
    assert_eq!(loops, 1, "one run rolled");
    assert_eq!(
        m.loops()
            .iter()
            .filter(|l| l.unrolling == Unrolling::DontUnroll)
            .count(),
        1,
        "the rolled loop"
    );
    assert_valid(m.clone(), &moltenvk());
    for pattern in 0u32..81 {
        let seed = 3 + pattern;
        let sels: Vec<u32> = (0..k).map(|j| (pattern / 3u32.pow(j)) % 3).collect();
        let mut input = vec![seed, 0];
        input.extend(&sels);
        let want: Vec<u32> = [0, 0]
            .into_iter()
            .chain((0..k).map(|j| {
                let x = seed + 10 * (j + 1);
                match sels[j as usize] {
                    0 => 2 * x,
                    1 => x + 7,
                    _ => x + 1,
                }
            }))
            .collect();
        let words_out = 2 + k as usize;
        let a = run(&orig, &Config::default(), resources(&input, words_out)).expect("unrolled");
        let b = run(&m, &Config::default(), resources(&input, words_out)).expect("rolled");
        assert_eq!(
            words(&a.buffers[1]),
            want,
            "selectors {sels:?}: the unrolled kernel"
        );
        assert_eq!(
            words(&b.buffers[1]),
            want,
            "selectors {sels:?}: the rolled kernel"
        );
    }
}

/// A differing `u32` constant that grows by a stride from region to region (each region's add `10
/// (j + 1)` and output word `2 + j`) is computed from the trip index, with no select; the module is
/// valid and writes what the regions wrote, for three regions and six.
#[test]
fn constants_that_grow_by_a_stride_come_from_the_trip_index() {
    for k in [3u32, 6] {
        let orig = run_kernel(k, false);
        let mut m = orig.clone();
        assert_eq!(roll(&mut m, 0, 1, &mut meter()).expect("roll"), 1);
        assert_eq!(selects(&orig), 0);
        assert_eq!(selects(&m), 0, "{k} regions: strides, no selects");
        assert_valid(m.clone(), &moltenvk());
        for (seed, seed2) in [(1u32, 2u32), (20, 0), (0, 0), (7, 30)] {
            let input = [seed, 0, seed2];
            let words_out = 2 + k as usize;
            let b = run(&m, &Config::default(), resources(&input, words_out)).expect("rolled");
            assert_eq!(
                words(&b.buffers[1]),
                expected(k, false, seed, seed2),
                "{k} regions: the rolled kernel"
            );
        }
    }
}

/// The roll is exact: constants off one stride (each region's add `10 (j + 1)^2`) keep their
/// selects, while the output word's `2 + j` beside them takes the stride; the rolled kernel
/// writes what the regions wrote.
#[test]
fn constants_off_a_stride_keep_their_selects() {
    let add: fn(u32) -> u32 = |j| 10 * (j + 1) * (j + 1);
    let orig = kernel_adding(3, false, add);
    let mut m = orig.clone();
    assert_eq!(roll(&mut m, 0, 1, &mut meter()).expect("roll"), 1);
    assert_eq!(
        selects(&m),
        2,
        "one select per trip but the last, for the adds alone"
    );
    assert_valid(m.clone(), &moltenvk());
    for (seed, seed2) in [(1u32, 2u32), (20, 0), (0, 0), (7, 30)] {
        let input = [seed, 0, seed2];
        let a = run(&orig, &Config::default(), resources(&input, 5)).expect("unrolled");
        let b = run(&m, &Config::default(), resources(&input, 5)).expect("rolled");
        let want = expected_adding(3, false, seed, seed2, add);
        assert_eq!(words(&a.buffers[1]), want, "the unrolled kernel");
        assert_eq!(words(&b.buffers[1]), want, "the rolled kernel");
    }
}

/// A 2D image of `u32` texels, sampled (`Sampled = 1`).
fn sampled_image_ty(b: &mut ModuleBuilder) -> TypeId {
    let u32_ = u32_ty(b);
    b.ty(Type::Image(ImageType {
        sampled_type: u32_,
        dim: Dim::D2,
        depth: false,
        arrayed: false,
        multisampled: false,
        usage: ImageUsage::Sampled,
        format: crate::spv::ImageFormat::Unknown,
    }))
    .expect("image type")
}

/// The `ivec2` constant `(x, y)`.
fn ivec2(f: &mut FunctionBuilder<'_>, x: i32, y: i32) -> BResult<Value> {
    let i32_ = f.module().ty(Type::Int {
        width: 32,
        signed: true,
    })?;
    let ivec2 = f.module().ty(Type::Vector { elem: i32_, len: 2 })?;
    let mut parts = Vec::new();
    for c in [x, y] {
        parts.push(f.module().constant(Constant::Scalar {
            ty: i32_,
            bits: ConstBits(u64::from(c as u32)),
        })?);
    }
    let c = f.module().constant(Constant::Composite {
        ty: ivec2,
        parts: parts.into(),
    })?;
    f.module().const_value(c)
}

/// The constant offset of fetch kernel region `j`: `(j - 2, 0)`, every region's its own.
fn fetch_offset(j: u32) -> i32 {
    j as i32 - 2
}

/// How the regions of [`fetch_kernel`] read.
#[derive(Clone, Copy, PartialEq)]
enum Fetches {
    /// Region `j` fetches image 0 at `(seed, 0)` plus its own constant offset ([`fetch_offset`]).
    Offsets,
    /// Region `j` fetches its own image `j` at `(seed, 0)` plus its own constant offset.
    ImagesAndOffsets,
    /// Region `j` fetches image 0 at `(x, 0)` with no offset, `x` the previous region's texel's
    /// first component (the first region's: `seed`): a coordinate carried from region to region.
    Carried,
    /// [`Fetches::Offsets`], where region 1 also stores its texel to `out[obase + k + 2]` before
    /// its selection: a side effect inside it that the other regions lack.
    SideEffect,
    /// [`Fetches::Offsets`], where region 1 stores to buffer slot 2 where the others store to
    /// `out`: the same instructions on another buffer.
    OtherBuffer,
}

/// `k` regions reading `in = [seed, gate, obase]` and images of `u32` texels at texture slots
/// `0..k`: region `j` fetches a texel as `how` says and, where `gate != 0` (a selection, which
/// anchors the search's period), stores its first component times `j + 1` to
/// `out[obase + 2 + j]`; the last region's first component goes to `out[0]`.
fn fetch_kernel(k: u32, how: Fetches) -> TypedModule {
    let mut b = ModuleBuilder::new();
    let input = buffer(&mut b, 0);
    let out = buffer(&mut b, 1);
    let other = buffer(&mut b, 2);
    let ty = sampled_image_ty(&mut b);
    let images: Vec<GlobalId> = (0..k)
        .map(|i| {
            b.resource(
                slot(ResourceKind::Texture, i),
                StorageClass::UniformConstant,
                ty,
            )
            .expect("image")
        })
        .collect();
    let u32_ = u32_ty(&mut b);
    let uvec2 = b.ty(Type::Vector { elem: u32_, len: 2 }).expect("uvec2");
    let void = b.ty(Type::Void).expect("void");
    let mut f = b.function(void).expect("function");
    let body = |f: &mut FunctionBuilder<'_>| -> BResult<()> {
        let seed = load_at(f, input, 0)?;
        let gate = load_at(f, input, 1)?;
        let obase = load_at(f, input, 2)?;
        let mut x = seed;
        for j in 0..k {
            let image = match how {
                Fetches::ImagesAndOffsets => images[j as usize],
                _ => images[0],
            };
            let p = f.module().global_value(image)?;
            let t = f.load(p, MemAccess::default())?;
            let zero = c32(f, 0);
            let at = match how {
                Fetches::Carried => f.composite_construct(uvec2, &[x, zero])?,
                _ => f.composite_construct(uvec2, &[seed, zero])?,
            };
            let ops = match how {
                Fetches::Carried => ImageOperands::default(),
                _ => ImageOperands {
                    const_offset: Some(ivec2(f, fetch_offset(j), 0)?),
                    ..ImageOperands::default()
                },
            };
            let texel = f.fetch(t, at, &ops)?;
            let first = f.composite_extract(texel, &[0])?;
            if how == Fetches::SideEffect && j == 1 {
                let at = c32(f, k + 2);
                let idx = f.binary(BinOp::IAdd, obase, at)?;
                store_at(f, out, idx, first)?;
            }
            let open = f.binary(BinOp::INotEqual, gate, zero)?;
            let to = match how {
                Fetches::OtherBuffer if j == 1 => other,
                _ => out,
            };
            f.if_(
                &mut (),
                open,
                Box::new(move |f, _| {
                    let w = c32(f, j + 1);
                    let r = f.binary(BinOp::IMul, first, w)?;
                    let at = c32(f, 2 + j);
                    let idx = f.binary(BinOp::IAdd, obase, at)?;
                    store_at(f, to, idx, r)?;
                    Ok(ArmEnd::FallThrough(Vec::new()))
                }),
                None,
            )?;
            x = first;
        }
        let zero = c32(f, 0);
        store_at(f, out, zero, x)?;
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

/// The texel at `x` of fetch kernel image `i` (16 x 1): its first component `(5 x + 3 i + 1) mod
/// 16`, which keeps a carried coordinate inside the image.
fn fetch_texel(i: u32, x: u32) -> u32 {
    (5 * x + 3 * i + 1) % 16
}

/// The images [`fetch_kernel`] reads: `n` of 16 x 1 texels ([`fetch_texel`]).
fn fetch_images(n: u32) -> Vec<Image> {
    (0..n)
        .map(|i| {
            let mut img = Image::new(slot(ResourceKind::Texture, i), [16, 1, 1], 1, 1, 1);
            for (x, t) in img.texels.iter_mut().enumerate() {
                *t = [fetch_texel(i, x as u32), 7, 0, 1];
            }
            img
        })
        .collect()
}

/// [`resources`] with buffer slot 2 (the other buffer of [`Fetches::OtherBuffer`]) too.
fn fetch_resources(input: &[u32], out_words: usize) -> Resources {
    let mut res = resources(input, out_words);
    res.buffers.push(Buffer {
        slot: Some(slot(ResourceKind::Buffer, 2)),
        address: None,
        bytes: vec![0; out_words * 4],
    });
    res
}

/// The words [`fetch_kernel`] writes for `seed` and `gate` (`k + 3` of them).
fn fetch_expected(k: u32, how: Fetches, seed: u32, gate: u32) -> Vec<u32> {
    let mut out = vec![0u32; k as usize + 3];
    let mut x = seed;
    for j in 0..k {
        let (image, at) = match how {
            Fetches::Offsets | Fetches::SideEffect | Fetches::OtherBuffer => {
                (0, seed as i32 + fetch_offset(j))
            }
            Fetches::ImagesAndOffsets => (j, seed as i32 + fetch_offset(j)),
            Fetches::Carried => (0, x as i32),
        };
        let first = fetch_texel(image, at as u32);
        if gate != 0 && !(how == Fetches::OtherBuffer && j == 1) {
            out[2 + j as usize] = first * (j + 1);
        }
        if how == Fetches::SideEffect && j == 1 {
            out[k as usize + 2] = first;
        }
        x = first;
    }
    out[0] = x;
    out
}

/// The entry's switches, each with its case count and the arms' instructions of `pick`.
fn switch_arms(m: &TypedModule, pick: impl Fn(&Inst) -> bool + Copy) -> Vec<(usize, Vec<&Inst>)> {
    let f = &m.functions()[m.entry().func.index()];
    let mut out = Vec::new();
    for b in &f.blocks {
        let Terminator::Switch { default, cases, .. } = &m.blocks()[b.index()].term else {
            continue;
        };
        let mut arms: Vec<BlockId> = cases.iter().map(|(_, a)| *a).collect();
        arms.push(*default);
        arms.dedup();
        let insts = arms
            .iter()
            .flat_map(|a| &m.blocks()[a.index()].insts)
            .map(|n| &n.inst)
            .filter(|i| pick(i))
            .collect();
        out.push((cases.len(), insts));
    }
    out
}

/// The `ConstOffset` of an image instruction, as its `(x, y)`.
fn offset_of(m: &TypedModule, inst: &Inst) -> Option<(i32, i32)> {
    let o = match inst {
        Inst::ImageFetch { ops, .. } | Inst::ImageSample { ops, .. } => ops.const_offset?,
        _ => return None,
    };
    let ValueOrigin::Const(id) = m.value(o).ok()?.origin else {
        return None;
    };
    let crate::spirv::ir::ConstKind::Plain(Constant::Composite { parts, .. }) =
        &m.consts.get(id).ok()?.kind
    else {
        return None;
    };
    let lane =
        |i: usize| -> Option<i32> { Some(m.consts.scalar_bits(*parts.get(i)?)?.0 as u32 as i32) };
    Some((lane(0)?, lane(1)?))
}

/// Regions that fetch at their own constant offsets (an operand SPIR-V and Metal take only as a
/// constant) roll into one loop whose fetch is in a switch on the trip, each arm the fetch at its
/// trip's offset; with images of their own too, each arm loads its trip's image. The module is
/// valid and writes what the regions wrote (the interpreter).
#[test]
fn regions_fetching_at_their_own_constant_offsets_roll_through_a_switch() {
    for (k, how) in [
        (5u32, Fetches::Offsets),
        (3, Fetches::Offsets),
        (4, Fetches::ImagesAndOffsets),
    ] {
        let orig = fetch_kernel(k, how);
        let mut m = orig.clone();
        let loops = roll(&mut m, 0, 1, &mut meter()).expect("roll");
        assert_eq!(loops, 1, "{k} regions: one run rolled");
        let fetches = |m: &TypedModule| {
            insts(m)
                .filter(|i| matches!(i, Inst::ImageFetch { .. }))
                .count()
        };
        assert_eq!(fetches(&orig), k as usize);
        let arms = switch_arms(&m, |i| matches!(i, Inst::ImageFetch { .. }));
        assert_eq!(arms.len(), 1, "{k} regions: one switch");
        let (cases, arm_fetches) = &arms[0];
        assert_eq!(*cases, k as usize - 1, "a case per trip but the last");
        let offsets: Vec<_> = arm_fetches.iter().map(|i| offset_of(&m, i)).collect();
        let want: Vec<_> = (0..k).map(|j| Some((fetch_offset(j), 0))).collect();
        assert_eq!(
            offsets, want,
            "{k} regions: each arm fetches at its trip's offset"
        );
        assert_valid(m.clone(), &moltenvk());
        for (seed, gate) in [(2u32, 1u32), (5, 1), (9, 0), (13, 1)] {
            let res = || fetch_resources(&[seed, gate, 0], k as usize + 3);
            let (a, _) = run_with_images(&orig, &Config::default(), res(), fetch_images(k))
                .expect("unrolled");
            let (b, _) =
                run_with_images(&m, &Config::default(), res(), fetch_images(k)).expect("rolled");
            let want = fetch_expected(k, how, seed, gate);
            assert_eq!(
                words(&a.buffers[1]),
                want,
                "{k} regions, seed {seed}: unrolled"
            );
            assert_eq!(
                words(&b.buffers[1]),
                want,
                "{k} regions, seed {seed}: rolled"
            );
        }
    }
}

/// A coordinate carried from region to region (each region fetches where the previous region's
/// texel points) rolls into one loop that carries it in a variable, and the rolled module fetches
/// what the regions fetched (the interpreter).
#[test]
fn a_coordinate_carried_from_the_previous_region_rolls_and_reads_what_it_read() {
    let k = 6u32;
    let orig = fetch_kernel(k, Fetches::Carried);
    let mut m = orig.clone();
    let loops = roll(&mut m, 0, 1, &mut meter()).expect("roll");
    assert_eq!(loops, 1, "one run rolled");
    assert_eq!(
        insts(&m)
            .filter(|i| matches!(i, Inst::ImageFetch { .. }))
            .count(),
        1,
        "one fetch, in the loop"
    );
    assert!(
        switch_arms(&m, |_| true).is_empty(),
        "nothing differs by trip but the coordinate"
    );
    assert_valid(m.clone(), &moltenvk());
    for (seed, gate) in [(0u32, 1u32), (3, 1), (7, 0), (15, 1)] {
        let res = || fetch_resources(&[seed, gate, 0], k as usize + 3);
        let (a, _) =
            run_with_images(&orig, &Config::default(), res(), fetch_images(1)).expect("unrolled");
        let (b, _) =
            run_with_images(&m, &Config::default(), res(), fetch_images(1)).expect("rolled");
        let want = fetch_expected(k, Fetches::Carried, seed, gate);
        assert_eq!(words(&a.buffers[1]), want, "seed {seed}: unrolled");
        assert_eq!(words(&b.buffers[1]), want, "seed {seed}: rolled");
    }
}

/// Regions are not rolled where one does what the others do not: a store inside region 1 that the
/// other regions lack, or region 1's store to another buffer (its instructions keyed alike, its
/// pointer not the same). Each module stays as it was and writes what it wrote.
#[test]
fn a_region_with_a_side_effect_of_its_own_is_not_rolled() {
    for how in [Fetches::SideEffect, Fetches::OtherBuffer] {
        let orig = fetch_kernel(3, how);
        let mut m = orig.clone();
        assert_eq!(roll(&mut m, 0, 1, &mut meter()).expect("roll"), 0);
        assert_eq!(m.loops().len(), 0);
        let res = fetch_resources(&[5, 1, 0], 6);
        let (a, _) = run_with_images(&m, &Config::default(), res, fetch_images(1)).expect("run");
        assert_eq!(words(&a.buffers[1]), fetch_expected(3, how, 5, 1));
    }
}

/// `k` regions sampling (explicit LOD 0) a sampled image of `u32` texels through one sampler at
/// `(j / 4, 1 / 2)` and constant offset `(j - 2, 0)`, then sampling the same sampled image again
/// at `(1 / 2, 1 / 2)` with offset `(1, 0)` in every region, and, where `in[0] != 0`, storing
/// both first components to `out[in[1] + 2 j + 2]` and `out[in[1] + 2 j + 3]`.
fn sample_kernel(k: u32) -> TypedModule {
    let mut b = ModuleBuilder::new();
    let input = buffer(&mut b, 0);
    let out = buffer(&mut b, 1);
    let ty = sampled_image_ty(&mut b);
    let image = b
        .resource(
            slot(ResourceKind::Texture, 0),
            StorageClass::UniformConstant,
            ty,
        )
        .expect("image");
    let sampler_ty = b.ty(Type::Sampler).expect("sampler type");
    let sampler = b
        .resource(
            slot(ResourceKind::Sampler, 0),
            StorageClass::UniformConstant,
            sampler_ty,
        )
        .expect("sampler");
    let f32_ = b.ty(Type::Float { width: 32 }).expect("f32");
    let vec2 = b.ty(Type::Vector { elem: f32_, len: 2 }).expect("vec2");
    let void = b.ty(Type::Void).expect("void");
    let mut f = b.function(void).expect("function");
    let body = |f: &mut FunctionBuilder<'_>| -> BResult<()> {
        let float = |f: &mut FunctionBuilder<'_>, x: f32| -> BResult<Value> {
            let c = f.module().constant(Constant::Scalar {
                ty: f32_,
                bits: ConstBits(u64::from(x.to_bits())),
            })?;
            f.module().const_value(c)
        };
        let gate = load_at(f, input, 0)?;
        let obase = load_at(f, input, 1)?;
        for j in 0..k {
            let p = f.module().global_value(image)?;
            let t = f.load(p, MemAccess::default())?;
            let sp = f.module().global_value(sampler)?;
            let s = f.load(sp, MemAccess::default())?;
            let si = f.sampled_image(t, s)?;
            let u = float(f, j as f32 / 4.0)?;
            let half = float(f, 0.5)?;
            let lod = float(f, 0.0)?;
            let at = f.composite_construct(vec2, &[u, half])?;
            let ops = ImageOperands {
                lod: Some(lod),
                const_offset: Some(ivec2(f, j as i32 - 2, 0)?),
                ..ImageOperands::default()
            };
            let r = f.sample(si, at, None, false, &ops)?;
            let mid = f.composite_construct(vec2, &[half, half])?;
            let again = ImageOperands {
                lod: Some(lod),
                const_offset: Some(ivec2(f, 1, 0)?),
                ..ImageOperands::default()
            };
            let r2 = f.sample(si, mid, None, false, &again)?;
            let firsts = [
                f.composite_extract(r, &[0])?,
                f.composite_extract(r2, &[0])?,
            ];
            let zero = c32(f, 0);
            let open = f.binary(BinOp::INotEqual, gate, zero)?;
            f.if_(
                &mut (),
                open,
                Box::new(move |f, _| {
                    for (w, first) in [2 * j + 2, 2 * j + 3].into_iter().zip(firsts) {
                        let at = c32(f, w);
                        let idx = f.binary(BinOp::IAdd, obase, at)?;
                        store_at(f, out, idx, first)?;
                    }
                    Ok(ArmEnd::FallThrough(Vec::new()))
                }),
                None,
            )?;
        }
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

/// Regions that sample at their own constant offsets roll into one loop whose sample is in a switch
/// on the trip, each arm making the sampled image again (SPIR-V consumes a sampled image in its own
/// block) and sampling at its trip's offset; the second sample, after the switch, makes its sampled
/// image again there too. `spirv-val` takes the module.
#[test]
fn regions_sampling_at_their_own_constant_offsets_roll_with_their_sampled_images() {
    let k = 6u32;
    let orig = sample_kernel(k);
    let mut m = orig.clone();
    assert_eq!(roll(&mut m, 0, 1, &mut meter()).expect("roll"), 1);
    let samples = |m: &TypedModule| {
        insts(m)
            .filter(|i| matches!(i, Inst::ImageSample { .. }))
            .count()
    };
    assert_eq!(samples(&orig), 2 * k as usize);
    assert_eq!(
        samples(&m),
        k as usize + 1,
        "an arm per trip, and the second sample"
    );
    let arms = switch_arms(&m, |i| {
        matches!(i, Inst::ImageSample { .. } | Inst::SampledImage { .. })
    });
    assert_eq!(arms.len(), 1, "one switch");
    let (cases, arm_insts) = &arms[0];
    assert_eq!(*cases, k as usize - 1);
    let offsets: Vec<_> = arm_insts
        .iter()
        .filter(|i| matches!(i, Inst::ImageSample { .. }))
        .map(|i| offset_of(&m, i))
        .collect();
    let want: Vec<_> = (0..k).map(|j| Some((j as i32 - 2, 0))).collect();
    assert_eq!(offsets, want, "each arm samples at its trip's offset");
    assert_eq!(
        arm_insts
            .iter()
            .filter(|i| matches!(i, Inst::SampledImage { .. }))
            .count(),
        k as usize,
        "each arm makes its sampled image"
    );
    assert_valid(m, &moltenvk());
}

/// [`switched_operand`] names the operand [`Inst::for_each_value`] visits as the `ConstOffset`.
#[test]
fn the_switched_operand_is_the_constant_offset() {
    for m in [fetch_kernel(3, Fetches::Offsets), sample_kernel(2)] {
        let mut seen = 0;
        for b in m.blocks() {
            for node in &b.insts {
                let Some(at) = switched_operand(&node.inst) else {
                    continue;
                };
                let want = match node.inst {
                    Inst::ImageFetch { ops, .. } | Inst::ImageSample { ops, .. } => {
                        ops.const_offset
                    }
                    _ => None,
                };
                let mut operands = Vec::new();
                node.inst.for_each_value(|v| operands.push(v));
                assert_eq!(operands.get(at).copied(), want);
                seen += 1;
            }
        }
        assert!(seen > 0);
    }
}
