//! Builtin derivation per tier on the CPU interpreter: every Metal kernel builtin, in both
//! local-size forms, with `MultiRegion` 0 and 1, against Metal's definitions evaluated in Rust.
//! `dispatchThreads` grids are run as their regions (interior and edges, each with its exact local
//! size) and must cover every Metal thread exactly once with Metal's values; group-count splits
//! must be indistinguishable from the single dispatch.

use super::*;
use crate::device::DevicePolicy;
use crate::pipeline::sym::{Axis, ResourceKind, ResourceSlot, Stage};
use crate::spirv::interp::{run, Buffer, Config, Order, Resources, SubgroupSize};
use crate::spirv::ir::{natural_layout, ExecModel, LayoutRule, StorageClass, TypedModule};
use crate::spirv::testing::assert_valid;
use std::collections::BTreeMap;

/// The words one invocation records: every builtin in [`KernelBuiltin::ALL`] order, three words
/// for a vector builtin and one for a scalar.
fn record_words() -> usize {
    KernelBuiltin::ALL
        .iter()
        .map(|b| if b.is_vector() { 3 } else { 1 })
        .sum()
}

fn plan(form: LocalSizeForm, multi_region: bool) -> DispatchPlan {
    let all = BuiltinSet::of(&KernelBuiltin::ALL);
    DispatchPlan {
        local_size: form,
        builtins: all,
        multi_region,
        params: Vec::new(),
        tile: false,
        tile_sized: false,
        flat_grid: false,
        keys: SpecKey::dispatch_keys(Stage::Kernel),
        folded_groups: false,
        draw_pieces: false,
        wrapper_groups: false,
        object_grid: false,
        partial_groups: false,
    }
}

/// A kernel that stores every builtin of `dp` for each invocation at record
/// `WorkgroupIndex · N + LocalInvocationIndex` of buffer 0 (Vulkan's own numbering, independent
/// of the derivation under test). Under [`DispatchPlan::partial_groups`] `N` is the threads the
/// workgroup runs, `sx · sy · sz`, and each record begins with a word that is 1 where the
/// invocation is live ([`KernelDecl::live`]).
fn module(dp: &DispatchPlan) -> TypedModule {
    let mut mb = ModuleBuilder::new();
    let decl = KernelDecl::declare(&mut mb, dp, &SpecDefaults::default()).expect("declare");
    let u32_ = mb
        .ty(Type::Int {
            width: 32,
            signed: false,
        })
        .expect("u32");
    let uvec3 = mb.ty(Type::Vector { elem: u32_, len: 3 }).expect("uvec3");
    let rta = mb
        .ty(Type::RuntimeArray {
            elem: u32_,
            stride: Some(4),
        })
        .expect("rta");
    let layout = natural_layout(mb.types(), LayoutRule::Std430, true, &[rta]).expect("layout");
    let block = mb
        .ty(Type::Struct {
            members: [rta].into(),
            layout: Some(layout),
        })
        .expect("block");
    let out = mb
        .resource(
            ResourceSlot {
                stage: Stage::Kernel,
                kind: ResourceKind::Buffer,
                index: 0,
            },
            StorageClass::StorageBuffer,
            block,
        )
        .expect("resource");
    let push = if dp.multi_region {
        let v4 = mb.ty(Type::Vector { elem: u32_, len: 4 }).expect("uvec4");
        let two = mb
            .constant(Constant::Scalar {
                ty: u32_,
                bits: ConstBits(2),
            })
            .expect("2");
        let arr = mb
            .ty(Type::Array {
                elem: v4,
                len: two,
                stride: Some(16),
            })
            .expect("array");
        mb.push_member(PushField::DispatchRegion, arr, 0)
            .expect("member");
        Some(mb.push_global().expect("push"))
    } else {
        None
    };
    let index_var = mb
        .builtin(
            crate::spv::BuiltIn::LocalInvocationIndex,
            IoDir::Input,
            u32_,
        )
        .expect("index");
    let wg_var = mb
        .builtin(crate::spv::BuiltIn::WorkgroupId, IoDir::Input, uvec3)
        .expect("wg");
    let nwg_var = mb
        .builtin(crate::spv::BuiltIn::NumWorkgroups, IoDir::Input, uvec3)
        .expect("nwg");
    let void = mb.ty(Type::Void).expect("void");
    let mut fb = mb.function(void).expect("function");
    let f = &mut fb;
    let load = |f: &mut FunctionBuilder<'_>, g| {
        let p = f.module().global_value(g).expect("global");
        f.load(p, MemAccess::default()).expect("load")
    };
    let k = |f: &mut FunctionBuilder<'_>, n: u32| {
        let c = f.u32_const(n).expect("const");
        f.module().const_value(c).expect("value")
    };
    // slot = ((wg.z · G.y + wg.y) · G.x + wg.x) · N + LocalInvocationIndex, N = sx·sy·sz.
    let wg = load(f, wg_var);
    let nwg = load(f, nwg_var);
    let c = |f: &mut FunctionBuilder<'_>, v, i| f.composite_extract(v, &[i]).expect("extract");
    let (wx, wy, wz) = (c(f, wg, 0), c(f, wg, 1), c(f, wg, 2));
    let (gx, gy) = (c(f, nwg, 0), c(f, nwg, 1));
    let t = f.binary(BinOp::IMul, wz, gy).expect("mul");
    let t = f.binary(BinOp::IAdd, t, wy).expect("add");
    let t = f.binary(BinOp::IMul, t, gx).expect("mul");
    let group = f.binary(BinOp::IAdd, t, wx).expect("add");
    let size = KernelBuiltin::ThreadsPerThreadgroup;
    let mut no_region = |_: &mut FunctionBuilder<'_>| -> BResult<SValue> {
        Err(crate::spirv::error::BuildError::Dangling(
            crate::spirv::error::Dangling::Value(SValue::new(u32::MAX)),
        ))
    };
    let s = match dp.partial_groups {
        true => decl.running_threads(f).expect("size"),
        false => decl.value(f, size, &mut no_region).expect("size"),
    };
    let n = KernelDecl::volume(f, s).expect("volume");
    let first = f.binary(BinOp::IMul, group, n).expect("mul");
    let li = load(f, index_var);
    let inv = f.binary(BinOp::IAdd, first, li).expect("add");
    let words = record_words() as u32 + u32::from(dp.partial_groups);
    let per = k(f, words);
    let mut at = f.binary(BinOp::IMul, inv, per).expect("mul");
    let base = f.module().global_value(out).expect("out");
    let zero = k(f, 0);
    let one = k(f, 1);
    let mut region = |f: &mut FunctionBuilder<'_>| -> BResult<SValue> {
        let g = push.expect("a region read without a push block");
        let p = f.module().global_value(g)?;
        let idx = f.u32_const(0)?;
        let idx = f.module().const_value(idx)?;
        let p = f.access_chain(p, &[idx])?;
        f.load(p, MemAccess::default())
    };
    if dp.partial_groups {
        let live = decl
            .live(f, &mut region)
            .expect("live")
            .expect("a live flag");
        let word = f.select(live, one, zero).expect("select");
        let p = f.access_chain(base, &[zero, at]).expect("chain");
        f.store(p, word, MemAccess::default()).expect("store");
        at = f.binary(BinOp::IAdd, at, one).expect("next");
    }
    for b in KernelBuiltin::ALL {
        let v = decl.value(f, b, &mut region).expect("value");
        let parts: Vec<SValue> = if b.is_vector() {
            (0..3).map(|i| c(f, v, i)).collect()
        } else {
            vec![v]
        };
        for x in parts {
            let p = f.access_chain(base, &[zero, at]).expect("chain");
            f.store(p, x, MemAccess::default()).expect("store");
            at = f.binary(BinOp::IAdd, at, one).expect("next");
        }
    }
    fb.ret(None).expect("ret");
    let id = fb.finish().expect("finish");
    mb.entry(ExecModel::GLCompute, id, Vec::new())
        .expect("entry");
    mb.finalize().expect("finalize")
}

/// What the planner writes for one Vulkan dispatch of a Metal dispatch.
#[derive(Clone, Copy, Debug)]
struct Region {
    /// The exact local size (`sx, sy, sz`).
    local: [u32; 3],
    groups: [u32; 3],
    /// `(base, threads_per_grid, nominal)` with `MultiRegion = 1`.
    multi: Option<([u32; 3], [u32; 3], [u32; 3])>,
}

/// Run `m` for region `r`; returns each invocation's `(Vulkan workgroup, local index, record)`.
fn run_region(
    m: &TypedModule,
    form: LocalSizeForm,
    r: &Region,
    sg: SubgroupSize,
) -> Vec<([u32; 3], u32, Vec<u32>)> {
    let n = r.local.iter().product::<u32>();
    let groups = r.groups.iter().product::<u32>();
    let words = record_words();
    let mut spec = BTreeMap::new();
    for (a, v) in [Axis::X, Axis::Y, Axis::Z].into_iter().zip(r.local) {
        spec.insert(SpecKey::LocalSize(a), u64::from(v));
    }
    if form == LocalSizeForm::Flat {
        spec.insert(SpecKey::FlatLocalSize, u64::from(n));
    }
    let mut push = Vec::new();
    if let Some((base, threads, nominal)) = r.multi {
        spec.insert(SpecKey::MultiRegion, 1);
        push = region_bytes(base, threads, nominal).expect("pack").to_vec();
    } else {
        push.resize(32, 0);
    }
    let res = Resources {
        buffers: vec![Buffer {
            slot: Some(ResourceSlot {
                stage: Stage::Kernel,
                kind: ResourceKind::Buffer,
                index: 0,
            }),
            address: None,
            bytes: vec![0; (n * groups) as usize * words * 4],
        }],
        push,
        spec,
    };
    let mut outs = Vec::new();
    for order in [Order::Forward, Order::Reverse] {
        let cfg = Config {
            workgroups: r.groups,
            subgroup: sg,
            order,
            ..Config::default()
        };
        let out = run(m, &cfg, res.clone()).expect("run");
        outs.push(out.buffers[0].clone());
    }
    assert_eq!(outs[0], outs[1], "order independent");
    let w: Vec<u32> = outs[0]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let mut recs = Vec::new();
    for g in 0..groups {
        let wg = [
            g % r.groups[0],
            (g / r.groups[0]) % r.groups[1],
            g / (r.groups[0] * r.groups[1]),
        ];
        for li in 0..n {
            let at = ((g * n + li) as usize) * words;
            recs.push((wg, li, w[at..at + words].to_vec()));
        }
    }
    recs
}

/// Metal's builtins for the thread at grid position `p` of a Metal grid of `threads` threads with
/// nominal threadgroup size `l` (`dispatchThreads`; `Threadgroups` is the case `threads = G·l`).
fn metal(p: [u32; 3], threads: [u32; 3], l: [u32; 3]) -> Vec<u32> {
    let tg: [u32; 3] = std::array::from_fn(|i| p[i] / l[i]);
    let pos: [u32; 3] = std::array::from_fn(|i| p[i] % l[i]);
    let size: [u32; 3] = std::array::from_fn(|i| l[i].min(threads[i] - tg[i] * l[i]));
    let groups: [u32; 3] = std::array::from_fn(|i| threads[i].div_ceil(l[i]));
    let n: u32 = size.iter().product();
    let lin = pos[0] + size[0] * (pos[1] + size[1] * pos[2]);
    let nominal: u32 = l.iter().product();
    let mut out = Vec::new();
    for b in KernelBuiltin::ALL {
        use KernelBuiltin as K;
        match b {
            K::ThreadPositionInGrid => out.extend(p),
            K::ThreadPositionInThreadgroup => out.extend(pos),
            K::ThreadgroupPositionInGrid => out.extend(tg),
            K::ThreadsPerThreadgroup => out.extend(size),
            K::ThreadsPerGrid => out.extend(threads),
            K::ThreadgroupsPerGrid => out.extend(groups),
            K::ThreadIndexInThreadgroup => out.push(lin),
            K::ThreadIndexInSimdgroup => out.push(lin % 32),
            K::SimdgroupIndexInThreadgroup => out.push(lin / 32),
            K::ThreadsPerSimdgroup => out.push(32),
            K::SimdgroupsPerThreadgroup => out.push(n.div_ceil(32)),
            K::ThreadIndexInQuadgroup => out.push(lin % 4),
            K::QuadgroupIndexInThreadgroup => out.push(lin / 4),
            K::QuadgroupsPerThreadgroup => out.push(n.div_ceil(4)),
            K::DispatchThreadsPerThreadgroup => out.extend(l),
            K::DispatchSimdgroupsPerThreadgroup => out.push(nominal.div_ceil(32)),
            // Not a compute kernel's builtin (`ALL` leaves it out).
            K::RenderTargetArrayIndex => {}
        }
    }
    out
}

/// The Metal grid position a record reports (its first three words).
fn position(rec: &[u32]) -> [u32; 3] {
    [rec[0], rec[1], rec[2]]
}

/// Run every region of a Metal dispatch of `threads` threads (nominal size `l`) and check that
/// the records cover each Metal thread exactly once with Metal's builtins.
fn check_dispatch(
    m: &TypedModule,
    form: LocalSizeForm,
    regions: &[Region],
    threads: [u32; 3],
    l: [u32; 3],
    sg: SubgroupSize,
) {
    let mut seen: BTreeMap<[u32; 3], Vec<u32>> = BTreeMap::new();
    for r in regions {
        for (wg, li, rec) in run_region(m, form, r, sg) {
            let p = position(&rec);
            assert!(
                seen.insert(p, rec.clone()).is_none(),
                "{p:?} twice (region {r:?}, wg {wg:?}, li {li})"
            );
        }
    }
    let total: u32 = threads.iter().product();
    assert_eq!(seen.len(), total as usize, "every Metal thread once");
    for (p, rec) in seen {
        assert_eq!(rec, metal(p, threads, l), "thread {p:?}");
    }
}

/// The regions of `dispatchThreads(threads, l)`: per axis, the full groups (if any) and one edge
/// group of the remainder (if any), each region with its exact local size and base.
fn thread_regions(threads: [u32; 3], l: [u32; 3]) -> Vec<Region> {
    let axis = |i: usize| {
        let (t, n) = (threads[i], l[i]);
        let mut pieces = Vec::new();
        if t / n > 0 {
            pieces.push((0, t / n, n));
        }
        if t % n > 0 {
            pieces.push((t / n, 1, t % n));
        }
        pieces
    };
    let mut out = Vec::new();
    for x in axis(0) {
        for y in axis(1) {
            for z in axis(2) {
                out.push(Region {
                    local: [x.2, y.2, z.2],
                    groups: [x.1, y.1, z.1],
                    multi: Some(([x.0, y.0, z.0], threads, l)),
                });
            }
        }
    }
    out
}

/// `MultiRegion = 0`: a `Threadgroups` dispatch reads `NumWorkgroups` and no push field, in both
/// local-size forms and both modelled subgroup widths; so does an object function's grid
/// (`DispatchPlan::object_grid`), whose threads per grid here wrap on no axis.
#[test]
fn single_region_builtins_match_metal() {
    for form in [LocalSizeForm::Spec, LocalSizeForm::Flat] {
        for (mr, object) in [(false, false), (true, false), (false, true), (true, true)] {
            let mut dp = plan(form, mr);
            dp.object_grid = object;
            let m = module(&dp);
            for sg in [SubgroupSize::W32, SubgroupSize::W64] {
                // Flattened tiers need N % S = 0 (LocalSizeConstraint); 64 fits both widths.
                for (l, groups) in [([8, 4, 2], [3, 2, 1]), ([64, 1, 1], [2, 1, 2])] {
                    let r = Region {
                        local: l,
                        groups,
                        multi: None,
                    };
                    let threads = std::array::from_fn(|i| l[i] * groups[i]);
                    check_dispatch(&m, form, &[r], threads, l, sg);
                }
            }
        }
    }
}

/// `MultiRegion = 1`: `dispatchThreads` with partial edges on two axes, as its four regions, gives
/// every thread Metal's position, partial threadgroup size, and grid sizes.
#[test]
fn dispatch_threads_regions_match_metal() {
    let m = module(&plan(LocalSizeForm::Spec, true));
    for (threads, l) in [
        ([10, 3, 2], [4, 2, 1]),
        ([37, 1, 1], [32, 1, 1]),
        ([5, 7, 3], [2, 3, 2]),
    ] {
        let regions = thread_regions(threads, l);
        check_dispatch(
            &m,
            LocalSizeForm::Spec,
            &regions,
            threads,
            l,
            SubgroupSize::W32,
        );
    }
}

/// `MultiRegion = 1` for a group-count split: a `Threadgroups` dispatch of 5×3 groups split at 2
/// groups per axis is indistinguishable from the single dispatch, in both forms.
#[test]
fn group_count_splits_match_the_single_dispatch() {
    let (l, groups, max) = ([32u32, 2, 1], [5u32, 3, 1], 2u32);
    let threads: [u32; 3] = std::array::from_fn(|i| l[i] * groups[i]);
    for form in [LocalSizeForm::Spec, LocalSizeForm::Flat] {
        let m = module(&plan(form, true));
        let split = |g: u32| -> Vec<(u32, u32)> {
            (0..g.div_ceil(max))
                .map(|k| (k * max, max.min(g - k * max)))
                .collect()
        };
        let mut regions = Vec::new();
        for (bx, cx) in split(groups[0]) {
            for (by, cy) in split(groups[1]) {
                regions.push(Region {
                    local: l,
                    groups: [cx, cy, 1],
                    multi: Some(([bx, by, 0], threads, l)),
                });
            }
        }
        check_dispatch(&m, form, &regions, threads, l, SubgroupSize::W64);
    }
}

/// A partial object threadgroups module's run of a draw of `threads` threads in threadgroups of
/// `l` (one dispatch of the threadgroups the threads fill, rounded up, each of `l` invocations,
/// `MultiRegion` 1), its region flagged where `flagged` ([`partial_region_bytes`]): each
/// invocation's live word and builtins.
fn run_partial(
    m: &TypedModule,
    form: LocalSizeForm,
    threads: [u32; 3],
    l: [u32; 3],
    flagged: bool,
    sg: SubgroupSize,
) -> Vec<(u32, Vec<u32>)> {
    let groups: [u32; 3] = std::array::from_fn(|i| threads[i].div_ceil(l[i]));
    let n = l.iter().product::<u32>();
    let count = groups.iter().product::<u32>() * n;
    let words = record_words() + 1;
    let mut spec = BTreeMap::new();
    for (a, v) in [Axis::X, Axis::Y, Axis::Z].into_iter().zip(l) {
        spec.insert(SpecKey::LocalSize(a), u64::from(v));
    }
    if form == LocalSizeForm::Flat {
        spec.insert(SpecKey::FlatLocalSize, u64::from(n));
    }
    spec.insert(SpecKey::MultiRegion, 1);
    let push = match flagged {
        true => partial_region_bytes(threads, l),
        false => region_bytes([0; 3], threads, l),
    };
    let res = Resources {
        buffers: vec![Buffer {
            slot: Some(ResourceSlot {
                stage: Stage::Kernel,
                kind: ResourceKind::Buffer,
                index: 0,
            }),
            address: None,
            bytes: vec![0; count as usize * words * 4],
        }],
        push: push.expect("pack").to_vec(),
        spec,
    };
    let cfg = Config {
        workgroups: groups,
        subgroup: sg,
        ..Config::default()
    };
    let out = run(m, &cfg, res).expect("run");
    let w: Vec<u32> = out.buffers[0]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    w.chunks(words)
        .map(|rec| (rec[0], rec[1..].to_vec()))
        .collect()
}

/// An object function's partial threadgroups (`DispatchPlan::partial_groups`): a draw of the
/// threadgroups `threads` fill, rounded up, at the whole size `L` with its region flagged
/// `NONUNIFORM`, runs every Metal thread of `dispatchThreads(threads, L)` exactly once among its
/// live invocations, with Metal's builtins (the last threadgroup along an axis answering the
/// threads left, its positions and index over them), and leaves the rest not live; with the
/// region unflagged and whole threadgroups, every invocation is live with Metal's answers, in both
/// local-size forms.
#[test]
fn partial_object_threadgroups_match_metal() {
    for form in [LocalSizeForm::Spec, LocalSizeForm::Flat] {
        let mut dp = plan(form, true);
        dp.partial_groups = true;
        let m = module(&dp);
        for sg in [SubgroupSize::W32, SubgroupSize::W64] {
            // Flattened tiers need N % S = 0 (LocalSizeConstraint); 64 fits both widths.
            let cases: &[([u32; 3], [u32; 3])] = match form {
                LocalSizeForm::Spec => &[
                    ([6, 1, 1], [4, 1, 1]),
                    ([5, 3, 1], [4, 2, 1]),
                    ([37, 2, 3], [32, 1, 2]),
                    ([64, 1, 1], [64, 1, 1]),
                ],
                LocalSizeForm::Flat => &[
                    ([100, 1, 1], [64, 1, 1]),
                    ([70, 3, 1], [32, 2, 1]),
                    ([37, 2, 3], [32, 1, 2]),
                ],
            };
            for &(threads, l) in cases {
                let mut seen: BTreeMap<[u32; 3], Vec<u32>> = BTreeMap::new();
                let mut dead = 0;
                for (live, rec) in run_partial(&m, form, threads, l, true, sg) {
                    if live == 0 {
                        dead += 1;
                        continue;
                    }
                    let p = position(&rec);
                    assert!(seen.insert(p, rec).is_none(), "{p:?} twice");
                }
                let total: u32 = threads.iter().product();
                let runs: u32 = (0..3).map(|i| threads[i].div_ceil(l[i]) * l[i]).product();
                assert_eq!(seen.len(), total as usize, "every Metal thread once");
                assert_eq!(
                    dead,
                    runs - total,
                    "{threads:?} {l:?}: the invocations past them"
                );
                for (p, rec) in seen {
                    assert_eq!(
                        rec,
                        metal(p, threads, l),
                        "{form:?} {threads:?} {l:?}: {p:?}"
                    );
                }
            }
            // Unflagged, of whole threadgroups: the uniform answers, every invocation live.
            let (threads, l) = ([64, 4, 2], [32, 2, 1]);
            for (live, rec) in run_partial(&m, form, threads, l, false, sg) {
                assert_eq!(live, 1);
                assert_eq!(rec, metal(position(&rec), threads, l), "{form:?}");
            }
        }
    }
}

/// The nominal size packs 1..=1024 per axis and refuses 0 or more; the region field is `(base,
/// nominal)`, `(threads, 0)`.
#[test]
fn region_bytes_pack_the_nominal_size() {
    assert_eq!(pack_nominal([1, 1, 1]), Some(0));
    assert_eq!(pack_nominal([1024, 1, 1]), Some(1023));
    assert_eq!(pack_nominal([4, 2, 3]), Some(3 | 1 << 10 | 2 << 20));
    assert_eq!(pack_nominal([0, 1, 1]), None);
    assert_eq!(pack_nominal([1, 1025, 1]), None);
    let b = region_bytes([1, 2, 3], [10, 20, 30], [4, 2, 3]).expect("bytes");
    let w: Vec<u32> = b
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(w, vec![1, 2, 3, 3 | 1 << 10 | 2 << 20, 10, 20, 30, 0]);
}

/// Builtin shapes, grid dependence, and the lane-arithmetic class; the projections the lowered
/// representations admit.
#[test]
fn builtin_facts_and_projections() {
    use KernelBuiltin as K;
    assert_eq!(
        KernelBuiltin::ALL.iter().filter(|b| b.is_vector()).count(),
        7
    );
    for b in KernelBuiltin::ALL {
        let from_abi = match b {
            K::ThreadPositionInGrid => Builtin::ThreadPositionInGrid,
            K::ThreadPositionInThreadgroup => Builtin::ThreadPositionInThreadgroup,
            K::ThreadgroupPositionInGrid => Builtin::ThreadgroupPositionInGrid,
            K::ThreadsPerThreadgroup => Builtin::ThreadsPerThreadgroup,
            K::ThreadsPerGrid => Builtin::ThreadsPerGrid,
            K::ThreadgroupsPerGrid => Builtin::ThreadgroupsPerGrid,
            K::ThreadIndexInThreadgroup => Builtin::ThreadIndexInThreadgroup,
            K::ThreadIndexInSimdgroup => Builtin::ThreadIndexInSimdgroup,
            K::SimdgroupIndexInThreadgroup => Builtin::SimdgroupIndexInThreadgroup,
            K::ThreadsPerSimdgroup => Builtin::ThreadsPerSimdgroup,
            K::SimdgroupsPerThreadgroup => Builtin::SimdgroupsPerThreadgroup,
            K::ThreadIndexInQuadgroup => Builtin::ThreadIndexInQuadgroup,
            K::QuadgroupIndexInThreadgroup => Builtin::QuadgroupIndexInThreadgroup,
            K::QuadgroupsPerThreadgroup => Builtin::QuadgroupsPerThreadgroup,
            K::DispatchThreadsPerThreadgroup => Builtin::DispatchThreadsPerThreadgroup,
            K::DispatchSimdgroupsPerThreadgroup => Builtin::DispatchSimdgroupsPerThreadgroup,
            K::RenderTargetArrayIndex => Builtin::RenderTargetArrayIndex,
        };
        assert_eq!(KernelBuiltin::of(&from_abi), Ok(b));
    }
    // The stage-in region's builtins are the stage-IO family's.
    for b in [Builtin::StageInGridOrigin, Builtin::StageInGridSize] {
        assert_eq!(KernelBuiltin::of(&b), Err(not_lowered()));
    }
    // A tile function's; `plan_kernel` refuses it in a kernel without tile inputs.
    assert_eq!(
        KernelBuiltin::of(&Builtin::RenderTargetArrayIndex),
        Ok(K::RenderTargetArrayIndex)
    );
    assert_eq!(KernelBuiltin::of(&Builtin::VertexId), Err(not_lowered()));
    let lanes: Vec<K> = KernelBuiltin::ALL
        .into_iter()
        .filter(|b| b.lane_arithmetic())
        .collect();
    assert_eq!(
        lanes,
        vec![
            K::ThreadIndexInSimdgroup,
            K::SimdgroupIndexInThreadgroup,
            K::ThreadsPerSimdgroup,
            K::SimdgroupsPerThreadgroup
        ]
    );
    let int = |bits| ScalarTy::Int { bits };
    let pos = K::ThreadPositionInGrid;
    let idx = K::ThreadIndexInThreadgroup;
    assert_eq!(
        Projection::of(pos, &DemoteTy::Scalar(int(32))),
        Some(Projection::Scalar { bits: 32 })
    );
    assert_eq!(
        Projection::of(idx, &DemoteTy::Scalar(int(16))),
        Some(Projection::Scalar { bits: 16 })
    );
    let v = |len| DemoteTy::Vector { elem: int(32), len };
    assert_eq!(
        Projection::of(pos, &v(2)),
        Some(Projection::Vector { bits: 32, len: 2 })
    );
    assert_eq!(Projection::of(idx, &v(2)), None, "a scalar builtin");
    assert_eq!(Projection::of(pos, &v(4)), None);
    assert_eq!(Projection::of(pos, &DemoteTy::Scalar(int(8))), None);
    assert_eq!(
        Projection::of(pos, &DemoteTy::Scalar(ScalarTy::Float { bits: 32 })),
        None
    );
    for p in [
        Projection::Scalar { bits: 16 },
        Projection::Vector { bits: 32, len: 3 },
    ] {
        assert_eq!(Projection::of(pos, &p.lowered()), Some(p));
    }
}

/// G-val: both local-size forms, with and without regions, pass L3 on the committed profiles
/// whose kernel stage has subgroups (the flattened form reads subgroup builtins).
#[test]
fn derivation_modules_are_valid() {
    for stem in ["moltenvk-m3", "lunarg-desktop-baseline-2023"] {
        let p = crate::pipeline::testing::profile(stem, DevicePolicy::default()).expect(stem);
        for form in [LocalSizeForm::Spec, LocalSizeForm::Flat] {
            for mr in [false, true] {
                assert_valid(module(&plan(form, mr)), &p);
            }
            let mut dp = plan(form, true);
            dp.partial_groups = true;
            assert_valid(module(&dp), &p);
        }
    }
}

/// The flat variant's 64-bit arm divides a `u64` by a `u32` in 32-bit operations
/// ([`KernelDecl::divmod_64_by_32`]): on the CPU interpreter its quotient and remainder are Rust's
/// for dividends at and around every power-of-two boundary of interest (2^16, 2^32, 2^58, 2^63,
/// 2^64 - 1) and a 1206 x 2622 grid's index range, and divisors 1, small, 1206 and 2622, and at
/// both sides of 2^31 and 2^32 (where `2r + bit` carries past 32 bits).
#[test]
fn the_flat_variant_s_64_bit_division_is_exact() {
    let mut cases: Vec<(u64, u32)> = Vec::new();
    let ns = [
        0u64,
        1,
        1 << 16,
        3_162_132,
        u64::from(u32::MAX),
        1 << 32,
        (1 << 32) + 1,
        1 << 58,
        1 << 63,
        u64::MAX,
    ];
    let ds = [
        1u32,
        2,
        3,
        1206,
        2622,
        1 << 31,
        (1 << 31) + 1,
        u32::MAX - 1,
        u32::MAX,
    ];
    for &n in &ns {
        for k in [0u64, 1, 2] {
            for &d in &ds {
                cases.push((n.wrapping_sub(k), d));
                cases.push((n.wrapping_add(k), d));
            }
        }
    }
    let mut mb = ModuleBuilder::new();
    let u32_ = mb
        .ty(Type::Int {
            width: 32,
            signed: false,
        })
        .expect("u32");
    let rta = mb
        .ty(Type::RuntimeArray {
            elem: u32_,
            stride: Some(4),
        })
        .expect("rta");
    let layout = natural_layout(mb.types(), LayoutRule::Std430, true, &[rta]).expect("layout");
    let block = mb
        .ty(Type::Struct {
            members: [rta].into(),
            layout: Some(layout),
        })
        .expect("block");
    let slot = ResourceSlot {
        stage: Stage::Kernel,
        kind: ResourceKind::Buffer,
        index: 0,
    };
    let buf = mb
        .resource(slot, StorageClass::StorageBuffer, block)
        .expect("resource");
    let uvec3 = mb.ty(Type::Vector { elem: u32_, len: 3 }).expect("uvec3");
    let wg_var = mb
        .builtin(crate::spv::BuiltIn::WorkgroupId, IoDir::Input, uvec3)
        .expect("wg");
    let void = mb.ty(Type::Void).expect("void");
    let mut fb = mb.function(void).expect("function");
    let f = &mut fb;
    let k = |f: &mut FunctionBuilder<'_>, n: u32| {
        let c = f.u32_const(n).expect("const");
        f.module().const_value(c).expect("value")
    };
    // Case `i` (the workgroup's x) reads `[lo, hi, d]` at words `3i..` and writes
    // `[q_lo, q_hi, r]` there.
    let p = f.module().global_value(wg_var).expect("global");
    let wg = f.load(p, MemAccess::default()).expect("load");
    let i = f.composite_extract(wg, &[0]).expect("x");
    let three = k(f, 3);
    let at = f.binary(BinOp::IMul, i, three).expect("mul");
    let base = f.module().global_value(buf).expect("buf");
    let zero = k(f, 0);
    let word = |f: &mut FunctionBuilder<'_>, j: u32| {
        let j = k(f, j);
        let a = f.binary(BinOp::IAdd, at, j).expect("add");
        f.access_chain(base, &[zero, a]).expect("chain")
    };
    let (pl, ph, pd) = (word(f, 0), word(f, 1), word(f, 2));
    let lo = f.load(pl, MemAccess::default()).expect("lo");
    let hi = f.load(ph, MemAccess::default()).expect("hi");
    let d = f.load(pd, MemAccess::default()).expect("d");
    let (q_lo, q_hi, r) = KernelDecl::divmod_64_by_32(f, [lo, hi], d).expect("divide");
    f.store(pl, q_lo, MemAccess::default()).expect("store");
    f.store(ph, q_hi, MemAccess::default()).expect("store");
    f.store(pd, r, MemAccess::default()).expect("store");
    fb.ret(None).expect("ret");
    let id = fb.finish().expect("finish");
    mb.entry(
        ExecModel::GLCompute,
        id,
        vec![ExecMode::LocalSize { x: 1, y: 1, z: 1 }],
    )
    .expect("entry");
    let m = mb.finalize().expect("finalize");
    let p = crate::pipeline::testing::profile("intel-arl-anv", DevicePolicy::default())
        .expect("profile");
    assert_valid(m.clone(), &p);
    let bytes: Vec<u8> = cases
        .iter()
        .flat_map(|&(n, d)| [n as u32, (n >> 32) as u32, d])
        .flat_map(u32::to_le_bytes)
        .collect();
    let res = Resources {
        buffers: vec![Buffer {
            slot: Some(slot),
            address: None,
            bytes,
        }],
        push: Vec::new(),
        spec: BTreeMap::new(),
    };
    let cfg = Config {
        workgroups: [cases.len() as u32, 1, 1],
        ..Config::default()
    };
    let out = run(&m, &cfg, res).expect("run");
    let w: Vec<u32> = out.buffers[0]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect();
    for (j, &(n, d)) in cases.iter().enumerate() {
        let q = u64::from(w[3 * j]) | (u64::from(w[3 * j + 1]) << 32);
        assert_eq!(
            (q, w[3 * j + 2]),
            (n / u64::from(d), (n % u64::from(d)) as u32),
            "{n} / {d}"
        );
    }
}
