//! Seeded random well-typed builder programs: every program the generator can build serializes to a
//! module `spirv-val` accepts for Vulkan 1.2 and 1.3, and building the same seed twice yields the
//! same words, and relocating it under random link tables writes only its relocation slots. The
//! property also runs through the owned verifier and L3 (`assert_valid`) on every committed device
//! profile.
//!
//! The generator keeps a scoped value pool, as a lowering must: values defined inside a construct
//! arm or loop body are usable only there, and only the construct's merged results escape it.

use super::{
    assert_relocation_writes_only_literals, assert_spirv_valid, cval, elem_ptr, finish_compute,
    float, gid_x, int, link_all, ssbo,
};
use crate::spirv::builder::constructs::{ArmEnd, ArmFn, LoopCtx, LoopSpec};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{GlobalId, TypeId, Value};
use crate::spirv::ir::{BinOp, ConvOp, MemAccess, TypedModule, UnOp};
use crate::spirv::serialize::{relocate, serialize};
use crate::spirv::testing::{check_valid, committed_profiles, spirv_dis};

/// A small deterministic PRNG (xorshift64*), so the property needs no dependency.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Rng {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    U32,
    I32,
    F32,
    Bool,
}

const KINDS: [Kind; 4] = [Kind::U32, Kind::I32, Kind::F32, Kind::Bool];

#[derive(Clone, Copy)]
struct Tys {
    u32_: TypeId,
    i32_: TypeId,
    f32_: TypeId,
}

/// The values in scope, by kind.
#[derive(Clone, Default)]
struct Pool(Vec<(Kind, Value)>);

impl Pool {
    fn pick(&self, rng: &mut Rng, k: Kind) -> Value {
        let of: Vec<Value> = self
            .0
            .iter()
            .filter(|(x, _)| *x == k)
            .map(|p| p.1)
            .collect();
        of[rng.below(of.len())]
    }

    fn push(&mut self, k: Kind, v: Value) {
        self.0.push((k, v));
    }
}

/// The generator's context, threaded through construct closures.
struct Gen {
    rng: Rng,
    t: Tys,
    out: GlobalId,
    budget: u32,
}

impl Gen {
    fn take(&mut self) -> bool {
        if self.budget == 0 {
            return false;
        }
        self.budget -= 1;
        true
    }
}

/// One random straight-line operation over `pool`, adding its result.
fn op(f: &mut FunctionBuilder<'_>, g: &mut Gen, pool: &mut Pool) -> BResult<()> {
    let k = KINDS[g.rng.below(KINDS.len())];
    let (a, b) = (pool.pick(&mut g.rng, k), pool.pick(&mut g.rng, k));
    let r = match k {
        Kind::U32 => {
            const OPS: [BinOp; 8] = [
                BinOp::IAdd,
                BinOp::ISub,
                BinOp::IMul,
                BinOp::UDiv,
                BinOp::UMod,
                BinOp::BitwiseAnd,
                BinOp::BitwiseXor,
                BinOp::ShiftLeftLogical,
            ];
            match g.rng.below(OPS.len() + 2) {
                8 => f.unary(UnOp::BitCount, a)?,
                9 => {
                    let s = pool.pick(&mut g.rng, Kind::I32);
                    f.convert(ConvOp::Bitcast, s, g.t.u32_)?
                }
                i => f.binary(OPS[i], a, b)?,
            }
        }
        Kind::I32 => {
            const OPS: [BinOp; 6] = [
                BinOp::IAdd,
                BinOp::ISub,
                BinOp::IMul,
                BinOp::SDiv,
                BinOp::BitwiseOr,
                BinOp::ShiftRightArithmetic,
            ];
            match g.rng.below(OPS.len() + 2) {
                6 => f.unary(UnOp::SNegate, a)?,
                7 => {
                    let x = pool.pick(&mut g.rng, Kind::F32);
                    f.convert(ConvOp::ConvertFToS, x, g.t.i32_)?
                }
                i => f.binary(OPS[i], a, b)?,
            }
        }
        Kind::F32 => {
            const OPS: [BinOp; 4] = [BinOp::FAdd, BinOp::FSub, BinOp::FMul, BinOp::FDiv];
            match g.rng.below(OPS.len() + 2) {
                4 => f.unary(UnOp::FNegate, a)?,
                5 => {
                    let x = pool.pick(&mut g.rng, Kind::U32);
                    f.convert(ConvOp::ConvertUToF, x, g.t.f32_)?
                }
                i => f.binary(OPS[i], a, b)?,
            }
        }
        Kind::Bool => {
            let (x, y) = (
                pool.pick(&mut g.rng, Kind::U32),
                pool.pick(&mut g.rng, Kind::U32),
            );
            let (p, q) = (
                pool.pick(&mut g.rng, Kind::F32),
                pool.pick(&mut g.rng, Kind::F32),
            );
            match g.rng.below(5) {
                0 => f.binary(BinOp::ULessThan, x, y)?,
                1 => f.binary(BinOp::IEqual, x, y)?,
                2 => f.binary(BinOp::FOrdLessThan, p, q)?,
                3 => f.binary(BinOp::LogicalAnd, a, b)?,
                _ => f.unary(UnOp::LogicalNot, a)?,
            }
        }
    };
    pool.push(k, r);
    // A `select` over the new kind now and then.
    if k != Kind::Bool && g.rng.below(4) == 0 {
        let c = pool.pick(&mut g.rng, Kind::Bool);
        let (x, y) = (pool.pick(&mut g.rng, k), pool.pick(&mut g.rng, k));
        let s = f.select(c, x, y)?;
        pool.push(k, s);
    }
    Ok(())
}

/// Store a random `u32` of `pool` to the output buffer at a random index.
fn store(f: &mut FunctionBuilder<'_>, g: &mut Gen, pool: &Pool) -> BResult<()> {
    let i = pool.pick(&mut g.rng, Kind::U32);
    let v = pool.pick(&mut g.rng, Kind::U32);
    let p = elem_ptr(f, g.out, i);
    f.store(p, v, MemAccess::default())
}

/// A random region: operations, stores, and nested constructs, down to `depth`.
fn region(f: &mut FunctionBuilder<'_>, g: &mut Gen, pool: &mut Pool, depth: u32) -> BResult<()> {
    let n = 1 + g.rng.below(5);
    for _ in 0..n {
        if !g.take() {
            return Ok(());
        }
        match g.rng.below(if depth == 0 { 2 } else { 5 }) {
            0 => op(f, g, pool)?,
            1 => store(f, g, pool)?,
            2 => if_region(f, g, pool, depth - 1)?,
            3 => switch_region(f, g, pool, depth - 1)?,
            _ => loop_region(f, g, pool, depth - 1)?,
        }
    }
    Ok(())
}

/// An arm that runs a region in its own scope, then merges one `u32` and one `f32`.
fn arm<'a>(pool: &'a Pool, depth: u32) -> ArmFn<'a, Gen> {
    Box::new(move |fb: &mut FunctionBuilder<'_>, g: &mut Gen| {
        let mut local = pool.clone();
        region(fb, g, &mut local, depth)?;
        let u = local.pick(&mut g.rng, Kind::U32);
        let x = local.pick(&mut g.rng, Kind::F32);
        Ok(ArmEnd::FallThrough(vec![u, x]))
    })
}

fn merge(pool: &mut Pool, merged: &[Value]) {
    pool.push(Kind::U32, merged[0]);
    pool.push(Kind::F32, merged[1]);
}

fn if_region(f: &mut FunctionBuilder<'_>, g: &mut Gen, pool: &mut Pool, depth: u32) -> BResult<()> {
    let c = pool.pick(&mut g.rng, Kind::Bool);
    let scope = pool.clone();
    let m = f.if_(g, c, arm(&scope, depth), Some(arm(&scope, depth)))?;
    merge(pool, &m.0);
    Ok(())
}

fn switch_region(
    f: &mut FunctionBuilder<'_>,
    g: &mut Gen,
    pool: &mut Pool,
    depth: u32,
) -> BResult<()> {
    let sel = pool.pick(&mut g.rng, Kind::U32);
    let scope = pool.clone();
    let cases = vec![
        (vec![0], arm(&scope, depth)),
        (vec![1, 5], arm(&scope, depth)),
        (vec![u32::MAX], arm(&scope, depth)),
    ];
    let m = f.switch_(g, sel, cases, arm(&scope, depth))?;
    merge(pool, &m.0);
    Ok(())
}

/// A counted loop whose body is a random region; its values do not escape.
fn loop_region(
    f: &mut FunctionBuilder<'_>,
    g: &mut Gen,
    pool: &mut Pool,
    depth: u32,
) -> BResult<()> {
    let u32_ = g.t.u32_;
    let var = f.local_var(u32_, None)?;
    let counter = f.var_ptr(var)?;
    let zero = cval(f, u32_, 0);
    f.store(counter, zero, MemAccess::default())?;
    let one = cval(f, u32_, 1);
    let limit = cval(f, u32_, 4);
    let scope = pool.clone();
    f.loop_(
        g,
        LoopSpec::default(),
        Box::new(
            move |fb: &mut FunctionBuilder<'_>, g: &mut Gen, lc: &LoopCtx| {
                let n = fb.load(counter, MemAccess::default())?;
                let done = fb.binary(BinOp::UGreaterThanEqual, n, limit)?;
                fb.if_(
                    g,
                    done,
                    Box::new(|fb: &mut FunctionBuilder<'_>, _: &mut Gen| {
                        lc.break_(fb)?;
                        Ok(ArmEnd::Exits)
                    }),
                    None,
                )?;
                let n1 = fb.binary(BinOp::IAdd, n, one)?;
                fb.store(counter, n1, MemAccess::default())?;
                let mut local = scope.clone();
                local.push(Kind::U32, n);
                region(fb, g, &mut local, depth)
            },
        ),
    )?;
    Ok(())
}

/// The program for `seed`.
fn program(seed: u64) -> TypedModule {
    let mut b = ModuleBuilder::new();
    // Only the types the generator uses: every program then needs only Vulkan 1.2 core, so
    // every committed profile must accept every program.
    let void = b.ty(crate::spirv::ir::Type::Void).expect("void");
    let t = Tys {
        u32_: int(&mut b, 32, false),
        i32_: int(&mut b, 32, true),
        f32_: float(&mut b, 32),
    };
    let out = ssbo(&mut b, 0, t.u32_, 4);
    let mut f = b.function(void).expect("main");
    let mut g = Gen {
        rng: Rng::new(seed),
        t,
        out,
        budget: 40,
    };
    let mut pool = Pool::default();
    let i = gid_x(&mut f);
    pool.push(Kind::U32, i);
    let seeds = [
        (Kind::U32, t.u32_, 7),
        (Kind::I32, t.i32_, 0xFFFF_FFFD),
        (Kind::F32, t.f32_, 0x3FC0_0000),
    ];
    for (k, ty, bits) in seeds {
        let v = cval(&mut f, ty, bits);
        pool.push(k, v);
    }
    let c = f
        .binary(BinOp::ULessThan, i, pool.0[1].1)
        .expect("seed bool");
    pool.push(Kind::Bool, c);
    region(&mut f, &mut g, &mut pool, 3).expect("region");
    store(&mut f, &mut g, &pool).expect("final store");
    f.ret(None).expect("ret");
    let main = f.finish().expect("finish");
    finish_compute(b, main)
}

fn words(m: TypedModule) -> Vec<u32> {
    let r = serialize(m).expect("serialize");
    relocate(&r, &link_all(&r)).expect("relocate")
}

#[test]
fn random_programs_pass_spirv_val() {
    // The generator must actually reach every construct, or the property proves less than it
    // says.
    let mut seen = [false; 4];
    for seed in 0..48 {
        eprintln!("seed {seed}");
        let text = spirv_dis(&assert_spirv_valid(program(seed)));
        for (s, op) in seen
            .iter_mut()
            .zip(["OpLoopMerge", "OpSwitch", "OpPhi", "OpSelect"])
        {
            *s |= text.contains(op);
        }
    }
    assert_eq!(seen, [true; 4]);
}

#[test]
fn random_programs_relocate_only_decoration_literals() {
    for seed in 0..48u32 {
        let r = serialize(program(u64::from(seed))).expect("serialize");
        assert_relocation_writes_only_literals(&r, seed, 4);
    }
}

#[test]
fn random_programs_build_deterministically() {
    for seed in 0..48 {
        assert_eq!(words(program(seed)), words(program(seed)), "seed {seed}");
    }
}

#[test]
fn random_programs_pass_l3_on_every_committed_profile() {
    // The owned verifier (with requirements ⊆ profile) and `spirv-val` under each profile's
    // environments and layout relaxations accept every program on every profile.
    let profiles = committed_profiles().expect("committed profiles");
    assert!(!profiles.is_empty());
    for seed in 0..48 {
        let m = program(seed);
        for (stem, p) in &profiles {
            if let Err(e) = check_valid(m.clone(), p) {
                panic!("seed {seed} on {stem}: {e}");
            }
        }
    }
}
