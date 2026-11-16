//! A uniform view's load that every trip of its loop reads alike leaves the loop where it may be
//! read early, a texel form's fetch too, and every other load stays; the kernel computes the same
//! either way.

use super::*;
use crate::budget::Budget;
use crate::device::profile::DeviceProfile;
use crate::memory::views::{declare_at, read_word, View, ViewKind};
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage};
use crate::spirv::builder::constructs::{ArmEnd, LoopSpec};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::GlobalId;
use crate::spirv::interp::{run, run_with_images, Buffer, Config, Image, Resources};
use crate::spirv::ir::{
    natural_layout, BinOp, ConstBits, Constant, ExecMode, ExecModel, LayoutRule, MemAccess,
};
use crate::spirv::testing::{assert_valid, committed_profiles};

const ROBUST: Containment = Containment::On {
    robust: Robustness::Rba2,
};

fn meter() -> Meter {
    Meter::new(&Budget::default())
}

fn slot(index: u32) -> ResourceSlot {
    ResourceSlot {
        stage: Stage::Kernel,
        kind: ResourceKind::Buffer,
        index,
    }
}

fn lavapipe() -> DeviceProfile {
    committed_profiles()
        .expect("profiles")
        .into_iter()
        .find(|(s, _)| s == "lavapipe")
        .expect("lavapipe")
        .1
}

fn c32(f: &mut FunctionBuilder<'_>, v: u32) -> Value {
    let c = f.u32_const(v).expect("constant");
    f.module().const_value(c).expect("value")
}

/// Word `i` of the `{ u32 words[]; }` or `{ u32 words[N]; }` block `buf`.
fn word(f: &mut FunctionBuilder<'_>, buf: GlobalId, i: Value) -> BResult<Value> {
    let zero = c32(f, 0);
    let base = f.module().global_value(buf)?;
    let p = f.access_chain(base, &[zero, i])?;
    f.load(p, MemAccess::default())
}

fn store_word(f: &mut FunctionBuilder<'_>, buf: GlobalId, i: Value, v: Value) -> BResult<()> {
    let zero = c32(f, 0);
    let base = f.module().global_value(buf)?;
    let p = f.access_chain(base, &[zero, i])?;
    f.store(p, v, MemAccess::default())
}

/// What [`kernel`]'s loop reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Shape {
    /// `p[k + 2]` on odd trips and `p[3]` on even ones, each under its own branch.
    Guarded,
    /// `p[1]` at the top of every trip, before the trip test.
    AtTop,
    /// `p[i & 7]`: a word that changes from trip to trip.
    ByTrip,
    /// [`Shape::Guarded`], with `out[i + 1] = i` stored on every trip.
    StoresInLoop,
    /// [`Shape::Guarded`], with `out[1] = n` stored before the loop.
    StoresBefore,
    /// `in[k + 2]` on odd trips, from the storage buffer.
    FromStorage,
}

/// `p` the uniform view (slot 1) of 8 words, `in` and `out` storage buffers (slots 0 and 2):
/// `n = in[0]`, `k = in[1]`, then `acc` summed over `n` trips as `shape` says, and `out[0] = acc`.
fn kernel(shape: Shape) -> TypedModule {
    kernel_of(shape, ViewKind::UniformWords { words: 8 })
}

/// [`kernel`], its uniform view `p` of the view kind `kind`, read through `memory::views`.
fn kernel_of(shape: Shape, kind: ViewKind) -> TypedModule {
    let mut b = ModuleBuilder::new();
    let input = declare_at(&mut b, slot(0), ViewKind::Storage)
        .expect("in")
        .global;
    let p: View = declare_at(&mut b, slot(1), kind).expect("p");
    let out = declare_at(&mut b, slot(2), ViewKind::Storage)
        .expect("out")
        .global;
    let void = b.ty(Type::Void).expect("void");
    let mut f = b.function(void).expect("function");
    let body = |f: &mut FunctionBuilder<'_>| -> BResult<()> {
        let u32_ = f.module().ty(Type::Int {
            width: 32,
            signed: false,
        })?;
        let zero = c32(f, 0);
        let one = c32(f, 1);
        let n = word(f, input, zero)?;
        let k = word(f, input, one)?;
        if shape == Shape::StoresBefore {
            store_word(f, out, one, n)?;
        }
        let z = f.u32_const(0)?;
        let acc = f.local_var(u32_, Some(z))?;
        let i = f.local_var(u32_, Some(z))?;
        let (ap, ip) = (f.var_ptr(acc)?, f.var_ptr(i)?);
        f.loop_(
            &mut (),
            LoopSpec::default(),
            Box::new(move |f, _, lc| {
                let t = f.load(ip, MemAccess::default())?;
                let a = f.load(ap, MemAccess::default())?;
                let top = match shape {
                    Shape::AtTop => {
                        let one = c32(f, 1);
                        Some(read_word(f, p, one)?)
                    }
                    _ => None,
                };
                let done = f.binary(BinOp::UGreaterThanEqual, t, n)?;
                f.if_(
                    &mut (),
                    done,
                    Box::new(|f, _| {
                        lc.break_(f)?;
                        Ok(ArmEnd::Exits)
                    }),
                    None,
                )?;
                let add = match (shape, top) {
                    (_, Some(w)) => w,
                    (Shape::ByTrip, _) => {
                        let seven = c32(f, 7);
                        let at = f.binary(BinOp::BitwiseAnd, t, seven)?;
                        read_word(f, p, at)?
                    }
                    _ => {
                        let one = c32(f, 1);
                        let odd = f.binary(BinOp::BitwiseAnd, t, one)?;
                        let odd = f.binary(BinOp::INotEqual, odd, zero)?;
                        let storage = shape == Shape::FromStorage;
                        let merged = f.if_(
                            &mut (),
                            odd,
                            Box::new(move |f, _| {
                                let two = c32(f, 2);
                                let at = f.binary(BinOp::IAdd, k, two)?;
                                let read = match storage {
                                    true => word(f, input, at)?,
                                    false => read_word(f, p, at)?,
                                };
                                Ok(ArmEnd::FallThrough(vec![read]))
                            }),
                            Some(Box::new(move |f, _| {
                                let three = c32(f, 3);
                                Ok(ArmEnd::FallThrough(vec![read_word(f, p, three)?]))
                            })),
                        )?;
                        *merged.0.first().expect("phi")
                    }
                };
                if shape == Shape::StoresInLoop {
                    let one = c32(f, 1);
                    let at = f.binary(BinOp::IAdd, t, one)?;
                    store_word(f, out, at, t)?;
                }
                let a1 = f.binary(BinOp::IAdd, a, add)?;
                f.store(ap, a1, MemAccess::default())?;
                let one = c32(f, 1);
                let t1 = f.binary(BinOp::IAdd, t, one)?;
                f.store(ip, t1, MemAccess::default())
            }),
        )?;
        let total = f.load(ap, MemAccess::default())?;
        store_word(f, out, zero, total)?;
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

/// The loads of uniform views in each block of `m`, by block.
fn uniform_loads(m: &TypedModule) -> Vec<(BlockId, usize)> {
    let s = Scratch {
        body: Vec::new(),
        reach: Vec::new(),
        seen: Vec::new(),
        movable: Vec::new(),
        stamp: 0,
        preds: Vec::new(),
        pos: {
            let mut pos = vec![u32::MAX; m.values.len()];
            for b in &m.blocks {
                for (i, n) in b.insts.iter().enumerate() {
                    if let Some(r) = n.result {
                        pos[r.index()] = i as u32;
                    }
                }
            }
            pos
        },
    };
    m.blocks
        .iter()
        .enumerate()
        .map(|(i, b)| {
            let n = b
                .insts
                .iter()
                .filter(|n| matches!(n.inst, Inst::Load { ptr, .. } if uniform_view(m, &s, ptr)))
                .count();
            (BlockId::new(i as u32), n)
        })
        .filter(|&(_, n)| n > 0)
        .collect()
}

fn resources(input: &[u32], p: &[u32]) -> Resources {
    let bytes = |w: &[u32]| w.iter().flat_map(|w| w.to_le_bytes()).collect();
    Resources {
        buffers: vec![
            Buffer {
                slot: Some(slot(0)),
                address: None,
                bytes: bytes(input),
            },
            Buffer {
                slot: Some(slot(1)),
                address: None,
                bytes: bytes(p),
            },
            Buffer {
                slot: Some(slot(2)),
                address: None,
                bytes: vec![0; 16 * 4],
            },
        ],
        push: Vec::new(),
        spec: Default::default(),
    }
}

fn out_word(m: &TypedModule, input: &[u32], p: &[u32]) -> u32 {
    let r = run(m, &Config::default(), resources(input, p)).expect("run");
    let b = &r.buffers[2];
    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

/// Hoist `shape`'s kernel under `containment`; returns the moves, the kernel before and after.
fn hoisted(shape: Shape, containment: Containment) -> (u32, TypedModule, TypedModule) {
    let before = kernel(shape);
    let mut after = before.clone();
    let moved = hoist_uniform_loads(&mut after, containment, &mut meter()).expect("hoist");
    assert_valid(after.clone(), &lavapipe());
    (moved, before, after)
}

/// The guarded loads at a trip-invariant word leave the loop for its preheader under robust
/// buffer access, and the kernel sums the same words.
#[test]
fn a_guarded_uniform_load_at_a_fixed_word_leaves_its_loop() {
    let (moved, before, after) = hoisted(Shape::Guarded, ROBUST);
    assert_eq!(moved, 2);
    let pre = after.loops()[0].preheader;
    assert_eq!(uniform_loads(&after), [(pre, 2)]);
    assert!(!uniform_loads(&before).iter().any(|&(b, _)| b == pre));
    let p = [10, 20, 30, 40, 50, 60, 70, 80];
    for (n, k) in [(0, 0), (1, 0), (5, 1), (6, 4)] {
        let input = [n, k];
        let want = (0..n)
            .map(|t| if t & 1 == 1 { p[k as usize + 2] } else { p[3] })
            .sum::<u32>();
        assert_eq!(out_word(&before, &input, &p), want, "before, n {n} k {k}");
        assert_eq!(out_word(&after, &input, &p), want, "after, n {n} k {k}");
    }
}

/// Without robust buffer access, a guarded load stays, and a load at the top of every trip still
/// leaves.
#[test]
fn without_robust_access_only_a_load_every_entry_runs_leaves() {
    let (moved, _, after) = hoisted(Shape::Guarded, Containment::UnsafeOff);
    assert_eq!(moved, 0);
    assert!(!uniform_loads(&after)
        .iter()
        .any(|&(b, _)| b == after.loops()[0].preheader));
    let (moved, before, after) = hoisted(Shape::AtTop, Containment::UnsafeOff);
    assert_eq!(moved, 1);
    let p = [0, 7, 0, 0, 0, 0, 0, 0];
    for n in [0, 1, 3] {
        assert_eq!(out_word(&before, &[n, 0], &p), 7 * n);
        assert_eq!(out_word(&after, &[n, 0], &p), 7 * n);
    }
}

/// A load whose word changes from trip to trip, a loop that stores to a buffer, a store to a buffer
/// before the loop, and a load of a storage buffer all stay.
#[test]
fn loads_a_trip_or_a_buffer_write_may_change_stay() {
    for shape in [Shape::ByTrip, Shape::StoresInLoop, Shape::StoresBefore] {
        let (moved, _, after) = hoisted(shape, ROBUST);
        assert_eq!(moved, 0, "{shape:?}");
        assert!(
            !uniform_loads(&after)
                .iter()
                .any(|&(b, _)| b == after.loops()[0].preheader),
            "{shape:?}"
        );
    }
    // The even trips' uniform load leaves; the odd trips' storage load stays.
    let (moved, _, _) = hoisted(Shape::FromStorage, ROBUST);
    assert_eq!(moved, 1);
}

/// No module holds a `Uniform` resource that is not a `Block` (what `BufferBlock` decorates in
/// old SPIR-V: a writable buffer; the crate's grammar names no `BufferBlock`): the builder refuses
/// one. The crate's own uniform view is a view.
#[test]
fn only_a_block_is_a_uniform_view() {
    let mut b = ModuleBuilder::new();
    let u32_ = b
        .ty(Type::Int {
            width: 32,
            signed: false,
        })
        .expect("u32");
    let eight = b
        .constant(Constant::Scalar {
            ty: u32_,
            bits: ConstBits(8),
        })
        .expect("8");
    let arr = b
        .ty(Type::Array {
            elem: u32_,
            len: eight,
            stride: Some(4),
        })
        .expect("array");
    let layout = natural_layout(b.types(), LayoutRule::Std430, false, &[arr]).expect("layout");
    let st = b
        .ty(Type::Struct {
            members: [arr].into(),
            layout: Some(layout),
        })
        .expect("struct");
    assert!(b.resource(slot(1), StorageClass::Uniform, st).is_err());
    let k = kernel(Shape::Guarded);
    let s = Scratch {
        body: Vec::new(),
        reach: Vec::new(),
        seen: Vec::new(),
        movable: Vec::new(),
        stamp: 0,
        preds: Vec::new(),
        pos: Vec::new(),
    };
    let view = (k.globals.iter())
        .find(|g| g.class == StorageClass::Uniform)
        .expect("the view");
    assert!(uniform_view(&k, &s, view.value));
}

/// The texel form's words `p` as a uniform texel buffer of slot 1, four words a texel.
fn texels(p: &[u32]) -> Image {
    let texels: Vec<[u32; 4]> = p.as_chunks::<4>().0.to_vec();
    let mut image = Image::new(slot(1), [texels.len() as u32, 1, 1], 1, 1, 1);
    image.texels = texels;
    image
}

/// [`out_word`] of a kernel whose `p` is a texel form, `p` its texels.
fn texel_out_word(m: &TypedModule, input: &[u32], p: &[u32]) -> u32 {
    let mut res = resources(input, &[]);
    res.buffers.retain(|b| b.slot != Some(slot(1)));
    let (r, _) = run_with_images(m, &Config::default(), res, vec![texels(p)]).expect("run");
    let b = &r.buffers[1];
    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

/// The texel fetches in each block of `m`, by block.
fn fetches(m: &TypedModule) -> Vec<(BlockId, usize)> {
    (m.blocks.iter().enumerate())
        .map(|(i, b)| {
            let n = (b.insts.iter())
                .filter(|n| matches!(n.inst, Inst::ImageFetch { .. }))
                .count();
            (BlockId::new(i as u32), n)
        })
        .filter(|&(_, n)| n > 0)
        .collect()
}

/// A uniform view's texel form moves as its block does: under robust buffer access the guarded
/// fetches at a trip-invariant word leave the loop with their images' loads (four moves), and the
/// kernel sums the same words; a fetch whose texel changes from trip to trip, and one in a loop
/// that stores to a buffer, stay.
#[test]
fn a_texel_form_s_fetch_at_a_fixed_word_leaves_its_loop() {
    let texel = ViewKind::UniformTexel { vecs: 2 };
    let before = kernel_of(Shape::Guarded, texel);
    let mut after = before.clone();
    let moved = hoist_uniform_loads(&mut after, ROBUST, &mut meter()).expect("hoist");
    assert_valid(after.clone(), &lavapipe());
    assert_eq!(moved, 4);
    let pre = after.loops()[0].preheader;
    assert_eq!(fetches(&after), [(pre, 2)]);
    assert!(!fetches(&before).iter().any(|&(b, _)| b == pre));
    let p = [10, 20, 30, 40, 50, 60, 70, 80];
    for (n, k) in [(0, 0), (1, 0), (5, 1), (6, 4)] {
        let input = [n, k];
        let want = (0..n)
            .map(|t| if t & 1 == 1 { p[k as usize + 2] } else { p[3] })
            .sum::<u32>();
        assert_eq!(
            texel_out_word(&before, &input, &p),
            want,
            "before, n {n} k {k}"
        );
        assert_eq!(
            texel_out_word(&after, &input, &p),
            want,
            "after, n {n} k {k}"
        );
    }
    for shape in [Shape::ByTrip, Shape::StoresInLoop] {
        let mut m = kernel_of(shape, texel);
        let moved = hoist_uniform_loads(&mut m, ROBUST, &mut meter()).expect("hoist");
        let pre = m.loops()[0].preheader;
        assert!(!fetches(&m).iter().any(|&(b, _)| b == pre), "{shape:?}");
        // The image's load may leave alone: it reads the descriptor, the same on every trip.
        assert!(moved <= 1, "{shape:?}: {moved}");
    }
}
