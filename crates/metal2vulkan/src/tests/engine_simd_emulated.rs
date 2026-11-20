//! Simdgroups emulated through workgroup memory.
//!
//! A kernel whose device cannot run a 32-lane simdgroup on subgroups (`lavapipe` cannot pin 32
//! lanes, nor `roadmap-2022`, whose subgroups are 4 lanes; `vk12-floor` has no subgroup
//! operations) is planned in the `Emulated` tier: simdgroup
//! `k` is the linear threads `[32k, 32k + 32)`, and every operation that reads other lanes
//! publishes through a workgroup scratch between workgroup barriers. The owned fixtures
//! `engine/simd_uniform` and the workgroup-uniform kernels of `engine/simd_ops` go
//! through the whole pipeline, pass L3 (`spirv-val` under each profile), and run on the CPU
//! interpreter at 8, 16, 32, and 64 subgroup lanes in both scheduling orders. Every kernel's
//! output is checked against an expectation derived from the thread index and the threadgroup
//! size, including threadgroups whose last simdgroup or quad is partially occupied (a source lane
//! outside the threadgroup reads zero bits, Metal's answer; `engine_simd_past_n` checks its
//! readings). An operation outside workgroup-uniform control flow runs predicated: `engine/
//! simd_divergent`, the divergent kernels of `simd_ops`, `a3_quad_votes_divergent`, and
//! `a3_multi_level_break_tail` check that exactly the lanes that execute an operation take part in
//! it, after early returns, in branch arms, in loops with per-thread trip counts, breaks,
//! continues, and returns, and in the tail of a two-level break. A matrix multiply-add in a
//! divergent block reads the cells of a lane that is not live as zero, Metal's answer
//! (`engine_simd_partly_live` checks its readings), and a kernel whose threadgroup memory and
//! scratch exceed `maxComputeSharedMemorySize` as `ThreadgroupMemorySize`, at compile or at plan
//! time. On a profile whose kernels run their simdgroups on whole subgroups (`moltenvk-m3`), a
//! dispatch with a partial simdgroup asks for the kernel's emulated-simdgroups variant, which gives
//! the same reading at 1 to 32 threads. Every run takes Vulkan's barrier rule
//! (`BarrierRule::Vulkan`): an invocation that returned while others wait at a workgroup barrier
//! traps. Under the identity domains every one of those kernels gives the same readings, with one
//! scratch region and as many barriers as under the default domains. No GPU is used.
//!
//! `llvm-as` assembles the fixtures; a missing tool skips visibly.

use std::collections::BTreeMap;

use crate::bind::{BindState, BoundBuffer, Dispatch, StageTable};
use crate::budget::Budget;
use crate::cache::VariantKey;
use crate::device::profile::{DeviceProfile, HandleDomain, PointerDomain};
use crate::device::subgroup::{OpBits, SimdClass, SubgroupRequirement, Tier};
use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::error::{BindError, Error, Feature};
use crate::library::Library;
use crate::lower::dispatch::{region_bytes, LocalSizeForm};
use crate::lower::select::Plan;
use crate::pipeline::facts::{Mechanism, RuntimeFact};
use crate::pipeline::interface::PushLoc;

use crate::pipeline::sym::{Axis, PushField, ResourceKind, ResourceSlot, SpecKey, Stage};
use crate::pipeline::testing::{
    compile, engine_dir, link_and_validate, llvm_assemble, profile, typed_module,
    typed_module_with, AssembleError, Scratch,
};
use crate::pipeline::{
    ComputePipeline, ComputePipelineDescriptor, FunctionDescriptor, LoweringInputs, Reconvergence,
    StageSource,
};
use crate::spirv::interp::{run, BarrierRule, Buffer, Config, Order, Resources, SubgroupSize};
use crate::spirv::ir::TypedModule;
use crate::spirv::testing::spirv_dis;
use crate::vk;

/// The committed profiles whose kernels' simdgroups are emulated.
const PROFILES: [&str; 2] = ["lavapipe", "roadmap-2022"];

const WIDTHS: [SubgroupSize; 4] = [
    SubgroupSize::W8,
    SubgroupSize::W16,
    SubgroupSize::W32,
    SubgroupSize::W64,
];

/// Full, multi-simdgroup, 2D and 3D, and partially occupied threadgroups (35 and 40 threads end
/// in a partial simdgroup and, for 35, a partial quad; one thread is a simdgroup of one).
const LOCALS: [[u32; 3]; 7] = [
    [32, 1, 1],
    [8, 8, 1],
    [96, 1, 1],
    [4, 8, 2],
    [40, 1, 1],
    [7, 5, 1],
    [1, 1, 1],
];

fn assemble_file(path: &std::path::Path, what: &str) -> Option<Vec<u8>> {
    match llvm_assemble(path) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {what}: {why}"),
    }
}

fn assemble(stem: &str) -> Option<Vec<u8>> {
    assemble_file(&engine_dir().join(format!("{stem}.ll")), stem)
}

/// `text` assembled from a scratch file.
fn assemble_text(tag: &str, text: &str) -> Option<Vec<u8>> {
    let scratch = Scratch::new(&format!("m2v-simd-emulated-{tag}"));
    let dir = scratch.path();
    let path = dir.join("kernel.ll");
    std::fs::write(&path, text).expect("write");
    assemble_file(&path, tag)
}

/// The policy of the identity domains: the guest's own addresses and handles.
fn identity() -> DevicePolicy {
    DevicePolicy {
        pointer_domain: PointerDomain::Identity,
        handle_domain: HandleDomain::Identity,
        ..DevicePolicy::default()
    }
}

/// The device address of buffer `index` under the identity domains.
fn guest_base(index: u32) -> u64 {
    (u64::from(index) + 1) << 32
}

/// Every word of buffer 0 after one workgroup of local size `local` ran `m` (planned as `plan`)
/// with `lanes`-wide subgroups, identical in both scheduling orders. Buffer 1, when the kernel
/// binds it, holds `constant`. Under the identity domains each buffer is at its
/// [`guest_base`], which the push block carries.
fn run_all(
    plan: &Plan,
    m: &TypedModule,
    local: [u32; 3],
    lanes: SubgroupSize,
    constant: Option<u32>,
) -> Vec<u32> {
    let mut spec = BTreeMap::new();
    for (a, v) in [Axis::X, Axis::Y, Axis::Z].into_iter().zip(local) {
        spec.insert(SpecKey::LocalSize(a), u64::from(v));
    }
    let n: u32 = local.iter().product();
    spec.insert(SpecKey::FlatLocalSize, u64::from(n));
    let mut push = vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)];
    let mut addressed = false;
    for f in &plan.push.fields {
        let PushLoc::Push { offset } = f.loc else {
            panic!("{f:?} spilled");
        };
        let o = offset as usize;
        match f.field {
            // Zero residuals: each buffer is bound at its start.
            PushField::SlotBindOffset(_) => {}
            PushField::SlotGuestBase(s) => {
                let word = match plan.families.access.identity {
                    true => super::engine_word_phases::slot_word(guest_base(s.index)),
                    false => guest_base(s.index),
                };
                push[o..o + 8].copy_from_slice(&word.to_le_bytes());
                addressed = true;
            }
            // The output's word per thread, and the constant's one word.
            PushField::SlotBounds(s) => {
                let bytes = [n as usize * 4, 4][s.index as usize];
                let word = super::engine_word_phases::bounds_word(bytes, 0);
                push[o..o + 8].copy_from_slice(&word);
            }
            PushField::ZeroAddress => {
                let word = super::engine_word_phases::ZERO_UNMAPPED;
                push[o..o + 8].copy_from_slice(&word.to_le_bytes());
            }
            PushField::DispatchRegion => {
                push[o..o + 32].copy_from_slice(&region_bytes([0; 3], local, local).expect("packs"))
            }
            _ => panic!("unexpected push field {f:?}"),
        }
    }
    let buffer = |index, bytes| Buffer {
        slot: (!addressed).then_some(ResourceSlot {
            stage: Stage::Kernel,
            kind: ResourceKind::Buffer,
            index,
        }),
        address: addressed.then(|| guest_base(index)),
        bytes,
    };
    let mut buffers = vec![buffer(0, vec![0; n as usize * 4])];
    if let Some(c) = constant {
        buffers.push(buffer(1, c.to_le_bytes().to_vec()));
    }
    let res = Resources {
        buffers,
        push,
        spec,
    };
    words(&run_orders(m, lanes, &res)[0])
}

/// The buffers after one workgroup ran `m` on `res` with `lanes`-wide subgroups, identical in
/// both scheduling orders.
fn run_orders(m: &TypedModule, lanes: SubgroupSize, res: &Resources) -> Vec<Vec<u8>> {
    let mut outs = Vec::new();
    for order in [Order::Forward, Order::Reverse] {
        let cfg = Config {
            workgroups: [1, 1, 1],
            subgroup: lanes,
            order,
            // Vulkan's rule: every invocation reaches each workgroup barrier.
            barriers: BarrierRule::Vulkan,
            ..Config::default()
        };
        outs.push(run(m, &cfg, res.clone()).expect("run").buffers);
    }
    assert_eq!(outs[0], outs[1], "order independent");
    outs.swap_remove(0)
}

fn words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// The threadgroup's shape as the expectations see it: `n` threads, simdgroup `t & !31` of
/// lanes `t & 31`, quad `t & !3`.
#[derive(Clone, Copy)]
struct Group {
    n: u32,
}

impl Group {
    fn lane(t: u32) -> u32 {
        t & 31
    }

    fn base(t: u32) -> u32 {
        t & !31
    }

    /// The threads of `t`'s simdgroup inside the threadgroup, in lane order.
    fn simd(self, t: u32) -> std::ops::Range<u32> {
        Self::base(t)..(Self::base(t) + 32).min(self.n)
    }

    /// The threads of `t`'s quad inside the threadgroup.
    fn quad(self, t: u32) -> std::ops::Range<u32> {
        (t & !3)..((t & !3) + 4).min(self.n)
    }

    /// Source thread `src` when it is inside the threadgroup.
    fn get(self, src: u32) -> Option<u32> {
        (src < self.n).then_some(src)
    }

    /// The thread index `src` holds, as a shuffle reads it: zero bits past the threadgroup's
    /// thread count, Metal's answer.
    fn read(self, src: u32) -> u32 {
        self.get(src).unwrap_or(0)
    }
}

/// The expected `out[t]` of each workgroup-uniform kernel of `simd_ops` and `simd_uniform` in a
/// threadgroup of `g.n` threads (see the fixtures' MSL); `steps` is `loop_k`'s trip count.
fn expected(kernel: &str, g: Group, t: u32, steps: u32) -> u32 {
    let l = Group::lane(t);
    let b = Group::base(t);
    let q = t & 3;
    let qb = t & !3;
    match kernel {
        // simd_ops
        "simd_shuffle_k" => g.read(b + ((l * 7 + 3) & 31)),
        "simd_broadcast_k" => g.read(b + 9),
        "simd_shuffle_up_k" => {
            if l >= 3 {
                t - 3
            } else {
                t
            }
        }
        "simd_shuffle_down_k" => {
            if l + 5 < 32 {
                g.read(t + 5)
            } else {
                t
            }
        }
        "simd_shuffle_xor_k" => g.read(t ^ 13),
        "simd_rotate_k" => g.read(b + ((l + 7) & 31)),
        "simd_fill_down_k" => {
            let s = l % 8;
            let src = g.get(t - s + (s + 3) % 8);
            if s + 3 < 8 {
                src.unwrap_or(0)
            } else {
                src.map_or(0, |u| u + 1000)
            }
        }
        "simd_fill_up_k" => {
            if l >= 2 {
                t - 2
            } else {
                g.get(t - l + (l + 30) % 32).map_or(0, |u| u + 1000)
            }
        }
        "simd_sum_k" => {
            let s: u32 = g.simd(t).sum();
            let f: u32 = g.simd(t).map(|u| u & 7).sum();
            s + f * 100_000
        }
        "simd_half_sum_k" => g.simd(t).map(Group::lane).sum(),
        "simd_minmax_k" => {
            let max = g.simd(t).map(Group::lane).max().unwrap_or(0);
            (1000 - 20) + (1000 + max - 20) * 10_000
        }
        "simd_bits_k" => {
            let or = g.simd(t).fold(0u32, |a, u| a | (1 << (u & 15)));
            let and = g.simd(t).fold(0xFFFFu32, |a, u| a & ((u | 0xF0) & 0xFFFF));
            or | (and << 16)
        }
        "simd_xor_k" => g.simd(t).fold(0u32, |a, u| a ^ ((u * u) & 0xFFFF)),
        "quad_ops_k" => {
            let sum: u32 = g.quad(t).sum();
            g.read(t ^ 1) + g.read(qb + 2) * 1000 + sum * 1_000_000
        }
        "quad_relative_k" => {
            let up = if q >= 1 { t - 1 } else { t };
            let down = if q + 2 < 4 { g.read(t + 2) } else { t };
            let rot = g.read(qb + ((q + 3) & 3));
            up + down * 1000 + rot * 1_000_000
        }
        "quad_dynamic_k" => {
            let max = g.quad(t).max().unwrap_or(t);
            g.read(t ^ ((t >> 2) & 3)) + max * 1000
        }
        // simd_uniform
        "vote_k" => {
            let has40 = g.simd(t).any(|u| u == 40);
            u32::from(!has40) + u32::from(has40) * 2 + u32::from(l == 0) * 4 + b * 8
        }
        "ballot_k" => g
            .simd(t)
            .filter(|u| u % 3 == 0)
            .fold(0u32, |a, u| a | (1 << Group::lane(u))),
        "prefix_k" => {
            let incl: u32 = (b..=t).sum();
            let excl: u32 = (0..l).sum();
            incl * 2 + excl * 100_000
        }
        "float_prefix_k" => {
            let excl: u32 = (0..l).sum();
            let incl: u32 = (b..=t).map(|u| u & 7).sum();
            excl + incl * 10_000
        }
        "quad_vote_k" => {
            let quad = g.quad(t);
            let mask = (1u32 << quad.len()) - 1;
            let has9 = quad.clone().any(|u| u == 9);
            u32::from(t == qb) + mask * 2 + u32::from(!has9) * 64 + u32::from(has9) * 128 + qb * 256
        }
        "vector_k" => g.get(t ^ 5).map_or(0, |s| {
            s | ((s + 1) << 8) | ((s ^ 0x55) << 16) | ((255 - s) << 24)
        }),
        "half_vector_k" => {
            let s = if l + 3 < 32 { g.read(t + 3) } else { t };
            (s & 63) | ((s >> 1) << 16)
        }
        "loop_k" => {
            let mut v: Vec<u32> = (0..g.n).collect();
            for i in 0..steps {
                v = (0..g.n)
                    .map(|u| {
                        let src = g.get(Group::base(u) + ((Group::lane(u) + 1) & 31));
                        src.map_or(0, |s| v[s as usize]) + i
                    })
                    .collect();
            }
            v[t as usize]
        }
        "barrier_k" => {
            let src = b | ((l + 1) & 31);
            if src < g.n {
                src * 3
            } else {
                0
            }
        }
        // The divergent kernels of simd_ops and simd_uniform: the lanes that do not run the
        // operation are inactive, as are the lanes past the threadgroup's end.
        "simd_prefix_k" => {
            let active = |u: u32| !Group::lane(u).is_multiple_of(3);
            if !active(t) {
                return 7;
            }
            let lanes: Vec<u32> = g.simd(t).filter(|&u| active(u)).collect();
            let incl: u32 = lanes.iter().filter(|&&u| u <= t).sum();
            let excl: u32 = lanes
                .iter()
                .filter(|&&u| u < t)
                .map(|&u| Group::lane(u))
                .sum();
            incl * 2 + excl * 100_000
        }
        "simd_float_prefix_k" => {
            if l < 4 {
                7
            } else {
                (4..l).sum()
            }
        }
        "simd_vote_k" => {
            let active = |u: u32| Group::lane(u) != 6;
            if !active(t) {
                return 0;
            }
            let lanes: Vec<u32> = g.simd(t).filter(|&u| active(u)).collect();
            u32::from(lanes.iter().all(|&u| Group::lane(u) != 6))
                + u32::from(lanes.contains(&40)) * 2
                + u32::from(lanes.iter().all(|&u| u < 40)) * 4
        }
        "simd_ballot_k" => {
            if l < 2 {
                return 1;
            }
            g.simd(t)
                .filter(|&u| Group::lane(u) >= 2 && u % 3 == 0)
                .fold(0u32, |a, u| a | (1 << Group::lane(u)))
        }
        "simd_first_k" => {
            if l < 5 {
                3
            } else {
                u32::from(l == 5) + (b + 5) * 10
            }
        }
        // Metal's backend speculates the branch's arm (`air::canon::speculate`), so every lane of
        // the quad votes, one that skips the arm too: the M3 Max (macOS 27.0.1) answers 95 on
        // thread 0 (its mask 0xF) and 159 on thread 8 (lane 9 in the mask and the `any`).
        "quad_ballot_k" => {
            if t % 7 == 2 {
                return 99;
            }
            let quad: Vec<u32> = g.quad(t).collect();
            let mask = quad.iter().fold(0u32, |a, &u| a | (1 << (u & 3)));
            let first = quad.first().copied() == Some(t);
            u32::from(first)
                + mask * 2
                + u32::from(quad.iter().all(|&u| u != 9)) * 64
                + u32::from(quad.contains(&9)) * 128
        }
        "divergent_k" => {
            if t < 16 {
                g.simd(t).filter(|&u| u < 16).sum()
            } else {
                0
            }
        }
        // simd_divergent. A thread that returned before its store leaves 0.
        "guard_k" => {
            if t >= LIMIT {
                return 0;
            }
            let s: u32 = g.simd(t).filter(|&u| u < LIMIT).sum();
            s + u32::from(t == b) * 100_000 + b * 1_000_000
        }
        "guard_shuffle_k" => {
            if t >= LIMIT {
                return 0;
            }
            // The source lane when it is active; zero past the thread count and where it returned
            // (Metal's answer for a lane that does not take part).
            let src = if l + 1 >= 32 {
                t
            } else if t + 1 >= g.n || t + 1 >= LIMIT {
                0
            } else {
                t + 1
            };
            let all = g.simd(t).filter(|&u| u < LIMIT).all(|u| u < 20);
            src + (l + 1) * 1000 + u32::from(all) * 100_000
        }
        "branch_k" => {
            if t % 3 == 1 {
                g.simd(t).filter(|&u| u % 3 == 1).sum()
            } else {
                let lanes: Vec<u32> = g.simd(t).filter(|&u| u % 3 != 1).collect();
                let max = lanes.iter().copied().max().unwrap_or(0);
                max * 1000 + lanes.iter().filter(|&&u| u < t).count() as u32
            }
        }
        "nested_k" => {
            if t & 1 == 0 {
                return 5;
            }
            let min = g.simd(t).filter(|&u| u & 1 == 1).min().unwrap_or(0);
            let inner = if t & 3 == 3 {
                g.simd(t).filter(|&u| u & 3 == 3).count() as u32 * 1000
            } else {
                0
            };
            min + inner
        }
        "trip_k" => (0..(t & 7))
            .map(|i| g.simd(t).filter(|&u| (u & 7) > i).count() as u32 * (i + 1))
            .sum(),
        "break_k" => {
            // The iterations in which lane `u` reaches the sum: it skips `u & 3`, and leaves at
            // the first other `i` above `u & 7`.
            let runs =
                |u: u32, i: u32| i != (u & 3) && (0..=i).all(|j| j == (u & 3) || j <= (u & 7));
            (0..8u32)
                .filter(|&i| runs(t, i))
                .map(|i| (i + 1) * g.simd(t).filter(|&u| runs(u, i)).count() as u32)
                .sum()
        }
        "loop_return_k" => {
            // Lane `u` returns in iteration `u & 3` when `u & 4`, before that iteration's sum.
            let runs = |u: u32, i: u32| u & 4 == 0 || (u & 3) > i;
            let count = |i: u32| g.simd(t).filter(|&u| runs(u, i)).count() as u32;
            if t & 4 != 0 {
                (0..(t & 3)).map(count).sum::<u32>() + 500
            } else {
                (0..4).map(count).sum()
            }
        }
        "guard_barrier_k" => {
            if t >= LIMIT {
                return 0;
            }
            let live = LIMIT.min(g.n);
            ((t + 1) % live) * 3 + g.simd(t).filter(|&u| u < LIMIT).count() as u32 * 1000
        }
        "switch_k" => match t & 3 {
            0 => g.simd(t).filter(|&u| u & 3 == 0).sum(),
            1 => g.simd(t).filter(|&u| u & 3 == 1).max().unwrap_or(0) + 1000,
            _ => u32::from(g.simd(t).any(|u| u & 3 >= 2 && u == 6)) + 7,
        },
        "quad_guard_k" => {
            let active = |u: u32| u % 5 != 3 && u < g.n;
            if !active(t) {
                return 0;
            }
            let quad: Vec<u32> = g.quad(t).filter(|&u| active(u)).collect();
            let sum: u32 = quad.iter().sum();
            // Zero past the thread count and where the source is not live (Metal's answer for a
            // lane that does not take part).
            let src = if t ^ 1 < g.n && active(t ^ 1) {
                t ^ 1
            } else {
                0
            };
            let first = quad.first().copied() == Some(t);
            let mask = quad.iter().fold(0u32, |a, &u| a | (1 << (u & 3)));
            sum + src * 1000 + u32::from(first) * 1_000_000 + mask * 10_000_000
        }
        "guard_simd_barrier_k" => {
            if t >= LIMIT {
                return 0;
            }
            let src = (t & !31) | ((t + 1) & 31);
            if src < LIMIT.min(g.n) {
                src * 5
            } else {
                7
            }
        }
        "loop_simd_barrier_k" => (0..(t & 3)).fold(0, |acc, i| (acc + i + 1) * 2),
        // Every thread breaks (or continues) at iteration `LIMIT & 7` (5), after its sum and its
        // odd iteration's add of `i`, before that iteration's add of 1000.
        "uniform_break_k" | "uniform_continue_k" => {
            let all = g.simd(t).count() as u32;
            let loop_ = if kernel == "uniform_break_k" {
                6 * all + (1 + 3 + 5) + 5 * 1000
            } else {
                8 * all + (1 + 3 + 5 + 7) + 7 * 1000
            };
            let odd = g.simd(t).filter(|&u| u & 1 == 1).count() as u32;
            loop_ + (t & 1) * odd * 100_000
        }
        // Every thread leaves at `i == 5` through its own exit, after six sums and five steps.
        "uniform_exit_k" => {
            let all = g.simd(t).count() as u32;
            let xor = g.get(t ^ 1).map_or(0, |s| 2 * s + 3);
            let odd = g.simd(t).filter(|&u| u & 1 == 1).count() as u32;
            6 * all + 5 * 1000 + xor + (t & 1) * odd * 100_000
        }
        // The fixture: quad q runs the votes on the lanes of the bits of q & 15.
        "a3_quad_votes_divergent" => {
            let active = |u: u32| u < g.n && (((u >> 2) & 15) >> (u & 3)) & 1 != 0;
            if !active(t) {
                return 0xFF00_0000;
            }
            let quad: Vec<u32> = g.quad(t).filter(|&u| active(u)).collect();
            let mask = quad.iter().fold(0u32, |a, &u| a | (1 << (u & 3)));
            u32::from(quad.first().copied() == Some(t))
                | (mask << 1)
                | (u32::from(quad.iter().all(|&u| u & 1 == 0)) << 5)
                | (u32::from(quad.iter().any(|&u| u % 5 == 0)) << 6)
                | (u32::from(quad.iter().all(|&u| u != 9)) << 7)
                | (u32::from(quad.contains(&9)) << 8)
        }
        // `simd_masks`: a quad ballot's bit `j` is quad lane `j`; a simdgroup's active mask (its
        // zero upper half folded in) has bit `i` for each active lane `i`.
        // The divergent kernel's arm is speculated as Metal's backend does
        // (`air::canon::speculate`): a lane that skips it votes too (the M3 Max, macOS 27.0.1,
        // answers 2 on thread 8, the skipping lane 9's).
        "mask_quad_ballot_k" | "mask_quad_ballot_divergent_k" => {
            if kernel == "mask_quad_ballot_divergent_k" && t % 7 == 2 {
                return 99;
            }
            g.quad(t)
                .filter(|&u| u % 3 == 0)
                .fold(0u32, |a, u| a | (1 << (u & 3)))
        }
        "mask_simd_active_k" | "mask_simd_active_divergent_k" => {
            let divergent = kernel == "mask_simd_active_divergent_k";
            let active = |u: u32| !divergent || Group::lane(u) >= 3;
            if !active(t) {
                return 1;
            }
            g.simd(t)
                .filter(|&u| active(u))
                .fold(0u32, |a, u| a | (1 << Group::lane(u)))
        }
        other => panic!("no expectation for {other}"),
    }
}

/// The workgroup-uniform kernels of `simd_ops`: every simdgroup and quad operation runs on every
/// thread.
const SIMD_OPS: [&str; 16] = [
    "simd_shuffle_k",
    "simd_broadcast_k",
    "simd_shuffle_up_k",
    "simd_shuffle_down_k",
    "simd_shuffle_xor_k",
    "simd_rotate_k",
    "simd_fill_down_k",
    "simd_fill_up_k",
    "simd_sum_k",
    "simd_half_sum_k",
    "simd_minmax_k",
    "simd_bits_k",
    "simd_xor_k",
    "quad_ops_k",
    "quad_relative_k",
    "quad_dynamic_k",
];

/// The kernels of `simd_ops` that run their operation on some lanes only.
const SIMD_OPS_DIVERGENT: [&str; 6] = [
    "simd_prefix_k",
    "simd_float_prefix_k",
    "simd_vote_k",
    "simd_ballot_k",
    "simd_first_k",
    "quad_ballot_k",
];

/// The kernels of `simd_divergent`: operations after an exit guard, in both arms of a branch,
/// in nested branches, in loops whose trip count, `continue`, `break`, or `return` is the
/// thread's, after a threadgroup barrier that returned threads skip, in a switch, quad
/// operations after an exit guard, a simdgroup barrier (the only simdgroup operation) after an
/// exit guard and in a loop whose trip count is the thread's, a uniform `break` and `continue`
/// in a loop every thread runs together, and exchanges in a uniform loop exit of its own (walked
/// as an arm of the exit dispatch, guarded although every thread takes it).
const SIMD_DIVERGENT: [&str; 15] = [
    "guard_k",
    "guard_shuffle_k",
    "branch_k",
    "nested_k",
    "trip_k",
    "break_k",
    "loop_return_k",
    "guard_barrier_k",
    "switch_k",
    "quad_guard_k",
    "guard_simd_barrier_k",
    "loop_simd_barrier_k",
    "uniform_break_k",
    "uniform_continue_k",
    "uniform_exit_k",
];

/// The threads of `simd_divergent`'s guard kernels that do not return early (buffer 1).
const LIMIT: u32 = 37;

/// The kernels of `simd_uniform` with the words each lane publishes (0: none, so no scratch).
const SIMD_UNIFORM: [(&str, u32); 9] = [
    ("vote_k", 1),
    ("ballot_k", 1),
    ("prefix_k", 1),
    ("float_prefix_k", 1),
    ("quad_vote_k", 1),
    ("vector_k", 4),
    ("half_vector_k", 2),
    ("loop_k", 1),
    ("barrier_k", 0),
];

/// Plan `kernel` of `bc` on profile `stem`, check its emulated tier, and check the expectation
/// at every width and local size.
fn run_kernel(stem: &str, bc: &[u8], kernel: &str, words: Option<u32>) {
    run_kernel_under(stem, DevicePolicy::default(), bc, kernel, words);
}

/// [`run_kernel`] on profile `stem` under `policy`.
fn run_kernel_under(stem: &str, policy: DevicePolicy, bc: &[u8], kernel: &str, words: Option<u32>) {
    let p = profile(stem, policy).expect("profile");
    let (plan, m) = typed_module(bc, kernel, &p).unwrap_or_else(|e| panic!("{stem} {kernel}: {e}"));
    check_emulated(stem, kernel, &plan, words);
    run_expected(stem, kernel, &plan, &m, &LOCALS);
}

/// `plan` is `kernel`'s in the emulated tier: no subgroup requirement, the local size kept, and
/// (when `words` is given) the scratch of `words` words per lane.
fn check_emulated(stem: &str, kernel: &str, plan: &Plan, words: Option<u32>) {
    assert_eq!(plan.simd.1, Tier::Emulated, "{stem} {kernel}");
    assert_eq!(plan.subgroup, SubgroupRequirement::NONE, "{stem} {kernel}");
    assert_eq!(
        plan.dispatch.as_ref().map(|d| d.local_size),
        Some(LocalSizeForm::Spec),
        "{stem} {kernel}: the local size is kept"
    );
    let fact = plan.facts.iter().find_map(|f| match f {
        RuntimeFact::SimdgroupEmulation {
            bytes_per_simdgroup,
            ..
        } => Some(*bytes_per_simdgroup),
        _ => None,
    });
    if let Some(w) = words {
        assert_eq!(
            fact,
            (w > 0).then_some(128 * w),
            "{stem} {kernel}: the scratch"
        );
    }
}

/// `kernel`'s module `m` (planned as `plan`) gives the expectation at every width and each of
/// `locals`.
fn run_expected(stem: &str, kernel: &str, plan: &Plan, m: &TypedModule, locals: &[[u32; 3]]) {
    let steps = 5;
    let constant = match kernel {
        "loop_k" => Some(steps),
        "guard_k"
        | "guard_shuffle_k"
        | "guard_barrier_k"
        | "guard_simd_barrier_k"
        | "uniform_break_k"
        | "uniform_continue_k"
        | "uniform_exit_k" => Some(LIMIT),
        _ => None,
    };
    for lanes in WIDTHS {
        for &local in locals {
            let g = Group {
                n: local.iter().product(),
            };
            let out = run_all(plan, m, local, lanes, constant);
            for (t, got) in out.iter().enumerate() {
                let want = expected(kernel, g, t as u32, steps);
                assert_eq!(*got, want, "{stem} {kernel} {local:?} {lanes:?} thread {t}");
            }
        }
    }
}

/// L3: the kernel links, relocates, and passes `spirv-val` under the profile, and uses no
/// subgroup instruction or capability. The plan's size estimate bounds the emitted words.
fn validate(stem: &str, bc: &[u8], kernel: &str) {
    let p = profile(stem, DevicePolicy::default()).expect("profile");
    let a = compile(bc, kernel, &p).unwrap_or_else(|e| panic!("{stem} {kernel}: {e}"));
    let (plan, _) = typed_module(bc, kernel, &p).expect("plan");
    let emitted = a.module.words.len() as u64;
    assert!(
        plan.estimate.words >= emitted,
        "{stem} {kernel}: estimate {} < emitted {emitted}",
        plan.estimate.words
    );
    let (_, words) = link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem} {kernel}: {e}"));
    let text = spirv_dis(&words);
    assert!(
        !text.contains("OpGroupNonUniform") && !text.contains("GroupNonUniform"),
        "{stem} {kernel}: a subgroup instruction or capability"
    );
    assert_eq!(a.subgroup, SubgroupRequirement::NONE, "{stem} {kernel}");
}

/// The workgroup-uniform kernels of `simd_ops` compute Metal's results on emulated simdgroups at
/// every subgroup width, in full and partially occupied threadgroups.
#[test]
fn simd_ops_run_emulated_at_every_width() {
    let Some(bc) = assemble("simd_ops") else {
        return;
    };
    for stem in PROFILES {
        for kernel in SIMD_OPS {
            validate(stem, &bc, kernel);
            run_kernel(stem, &bc, kernel, None);
        }
    }
}

/// Votes, ballots, `is_first`, `broadcast_first`, prefix sums, quad votes and masks, vector and
/// `half` values, repeated exchanges in a loop, and a simdgroup barrier over threadgroup memory, at
/// every subgroup width.
#[test]
fn simd_uniform_runs_emulated_at_every_width() {
    let Some(bc) = assemble("simd_uniform") else {
        return;
    };
    for stem in PROFILES {
        for (kernel, words) in SIMD_UNIFORM {
            validate(stem, &bc, kernel);
            run_kernel(stem, &bc, kernel, Some(words));
        }
    }
}

/// An emulated reduction or prefix sum is five tree rounds (a float exclusive sum six) and a vote
/// or ballot one word ORed by every lane, not a read per lane of the simdgroup, so its kernel stays
/// within the 3x of the same kernel on whole subgroups (`moltenvk-m3`, one subgroup instruction per
/// operation). The per-lane forms were 6.8 to 21.9 times larger (`simd_sum_k` 3,576 words on
/// `lavapipe` against 387, `simd_prefix_k` 6,681 against 454, `simd_vote_k` 10,011 against 457). A
/// quad vote is one gathered word too (the per-lane fold put the divergent quad votes at 4.20x),
/// and a divergent float min or max seeds its inactive lanes with the first active lane's operand
/// instead of exchanging a flag in every round (1,275 words on `lavapipe`, 2.59x, with the flag;
/// 1,163, 2.36x, seeded; held at 2.5x). Each push field loads once at entry, which took one reload
/// out of both forms (1,145 against 462 words, 2.48x, to 1,136 against 453, 2.51x): the emulation's
/// 683 words over the subgroup form did not change, so the hold moved to 2.55x. Through
/// `Device::from_queries` the kernel is 1,164 words on `lavapipe` against 453 (2.57x), held at
/// 2.6x, still below the flag form's 112 extra words (about 2.75x against 453). The subgroup form is `moltenvk-m3`'s
/// without its driver's measured deviation (`MetalCanonicalNans`), which neither emulated profile
/// has, so both forms carry the same NaN selects (the input's `fadd` before the reduction: 14
/// words, 1,150 against 467 on `lavapipe`, 2.46x) and the ratio stays the emulation's. A profile
/// with no measured half-rounding fact (`lavapipe`, `vk12-floor`) rounds each `half` add in integer
/// arithmetic, twenty instructions, where `moltenvk-m3`'s measured conversion takes three. That
/// added 270 words to each kernel with a `half` reduction or scan and none to its subgroup form:
/// `simd_half_sum_k` went from 1,148 against 512 words (2.24x) to 1,418 (2.77x), and
/// `simd_uniform`'s `float_prefix_k`, whose `half` inclusive sum is one of two scans, from 1,552
/// against 572 (2.71x) to 1,822 (3.19x), held at 3.2x. The words are the exact rounding's, not
/// the emulation's. The interpreter runs of this file check their results.
#[test]
fn emulated_reductions_scans_and_votes_stay_compact() {
    let Some(ops) = assemble("simd_ops") else {
        return;
    };
    let Some(uniform) = assemble("simd_uniform") else {
        return;
    };
    let Some(float_min) = float_reduction_kernels("min", false) else {
        return;
    };
    let Some(quads) = assemble("a3_quad_votes_divergent") else {
        return;
    };
    let direct = whole_subgroups_without_the_nan_fact();
    // The divergent float min, the only kernel held below 3x, and the `half` scan the integer
    // rounding put above it.
    let tight = (&float_min, "simd_float_prefix_k", 2.6);
    let rounded = (&uniform, "float_prefix_k", 3.2);
    let kernels = [
        (&float_min, "simd_minmax_k"),
        (tight.0, tight.1),
        (&ops, "simd_sum_k"),
        (&ops, "simd_half_sum_k"),
        (&ops, "simd_minmax_k"),
        (&ops, "simd_bits_k"),
        (&ops, "simd_prefix_k"),
        (&ops, "simd_float_prefix_k"),
        (&ops, "simd_vote_k"),
        (&ops, "simd_ballot_k"),
        (&ops, "simd_first_k"),
        (&uniform, "vote_k"),
        (&uniform, "ballot_k"),
        (&uniform, "prefix_k"),
        (rounded.0, rounded.1),
        (&ops, "quad_ops_k"),
        (&quads, "a3_quad_votes_divergent"),
    ];
    let mut over = Vec::new();
    for stem in PROFILES {
        let p = profile(stem, DevicePolicy::default()).expect("profile");
        for (bc, kernel) in kernels {
            let emulated = compile(bc, kernel, &p).expect("compile").module.words.len();
            let whole = compile(bc, kernel, &direct)
                .expect("compile")
                .module
                .words
                .len();
            let ratio = emulated as f64 / whole as f64;
            eprintln!("{stem} {kernel}: {emulated} words, {whole} on subgroups ({ratio:.2}x)");
            let ceiling = [tight, rounded]
                .into_iter()
                .find(|held| (bc, kernel) == (held.0, held.1))
                .map_or(3.0, |held| held.2);
            if ratio > ceiling {
                over.push(format!(
                    "{stem} {kernel}: {emulated} against {whole} ({ratio:.2}x)"
                ));
            }
        }
    }
    assert!(
        over.is_empty(),
        "above the ceiling over the subgroup form:\n{}",
        over.join("\n")
    );
}

/// `moltenvk-m3`'s profile (whole subgroups), whose NaN encoding is the device's: a float result's
/// NaN select is then the same on both sides of a ratio.
fn whole_subgroups_without_the_nan_fact() -> DeviceProfile {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles/moltenvk-m3.json");
    let text = std::fs::read_to_string(&path).expect("profile");
    let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    Device::from_queries(desc, DevicePolicy::default())
        .expect("device")
        .profile()
        .clone()
}

/// A copy of `simd_ops` whose `simd_minmax_k` (every thread) and `simd_float_prefix_k` (lanes 4
/// and up of each simdgroup, the others store 7) store the bits of `air.simd_<op>.f32` of
/// [`float_input`] of the thread instead.
fn float_reduction_kernels(op: &str, nan: bool) -> Option<Vec<u8>> {
    let text = std::fs::read_to_string(engine_dir().join("simd_ops.ll")).expect("read");
    let base = "  %s = mul i32 %1, 7\n  %r = urem i32 %s, 23\n  %rf = uitofp i32 %r to float\n  \
                %b = fadd float %rf, -1.100000e+01\n";
    let nan_of = |name: &str, from: &str| {
        if nan {
            format!(
                "  %k = urem i32 %1, 3\n  %isnan = icmp eq i32 %k, 1\n  \
                 {name} = select i1 %isnan, float 0x7FF8000000000000, float {from}\n"
            )
        } else {
            format!("  {name} = select i1 true, float {from}, float {from}\n")
        }
    };
    let uniform_from = "  %3 = trunc i32 %1 to i16\n";
    let uniform_to =
        "  store i32 %12, ptr addrspace(1) %14, align 4, !tbaa !43, !alias.scope !77\n";
    let (Some(u0), Some(u1)) = (text.find(uniform_from), text.find(uniform_to)) else {
        panic!("simd_minmax_k");
    };
    let uniform = format!(
        "{base}{}  %red = tail call float @air.simd_{op}.f32(float %x) #3\n  \
         %bits = bitcast float %red to i32\n  %at = zext i32 %1 to i64\n  \
         %p = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %at\n  \
         store i32 %bits, ptr addrspace(1) %p, align 4, !tbaa !43, !alias.scope !77\n",
        nan_of("%x", "%b")
    );
    let text = format!("{}{uniform}{}", &text[..u0], &text[u1 + uniform_to.len()..]);
    let divergent_from = concat!(
        "  %6 = tail call fast float @air.convert.f.f32.u.i32(i32 %2) #4\n",
        "  %7 = tail call fast float @air.simd_prefix_exclusive_sum.f32(float %6) #3\n",
        "  %8 = tail call i32 @air.convert.u.i32.f.f32(float %7) #4\n",
    );
    assert!(text.contains(divergent_from), "simd_float_prefix_k");
    let divergent = format!(
        "{base}  %low = icmp ult i32 %2, 4\n  %top = icmp eq i32 %2, 3\n  \
         %far = select i1 %top, float 1.000000e+03, float -1.000000e+03\n  \
         %v = select i1 %low, float %far, float %b\n{}  \
         %7 = tail call float @air.simd_{op}.f32(float %6) #3\n  %8 = bitcast float %7 to i32\n",
        nan_of("%6", "%v")
    );
    let mut text = text.replacen(divergent_from, &divergent, 1);
    let declare = format!("declare float @air.simd_{op}.f32(float) local_unnamed_addr #2\n");
    if !text.contains(&declare) {
        text.push_str(&declare);
    }
    assemble_text(&format!("float-{op}-{nan}"), &text)
}

/// Thread `t`'s operand in [`float_reduction_kernels`]: `(7t mod 23) - 11`, `-1000` in lanes 0
/// to 2 and `1000` in lane 3 of the divergent kernel (which do not take part), and with `nan` a
/// NaN in every third thread.
fn float_input(t: u32, divergent: bool, nan: bool) -> f32 {
    let l = Group::lane(t);
    if nan && t % 3 == 1 {
        f32::NAN
    } else if divergent && l < 4 {
        if l == 3 {
            1000.0
        } else {
            -1000.0
        }
    } else {
        ((t * 7) % 23) as f32 - 11.0
    }
}

/// The `GLSL.std.450` `FMin`, `FMax`, `NMin`, and `NMax` operands of `words` that a NaN constant
/// reaches through selects, composites, copies, and phis, as `(operation, constant id)`. A select
/// whose condition tests its second value for a NaN (`OpIsNan`, or the bits test `(bits &
/// 0x7FFFFFFF) > 0x7F800000`) answers its first, Metal's NaN, only where that value is a NaN, so it
/// introduces none: only its second value is walked.
fn nans_reaching_min_or_max(words: &[u32]) -> Vec<(&'static str, u32)> {
    let mut floats = BTreeMap::new();
    let mut consts: BTreeMap<u32, (u32, Vec<u32>)> = BTreeMap::new();
    let mut defs: BTreeMap<u32, (u16, Vec<u32>)> = BTreeMap::new();
    // The NaN tests' instructions: OpBitcast, OpIsNan, OpUGreaterThan, OpBitwiseAnd.
    let mut tests: BTreeMap<u32, (u16, Vec<u32>)> = BTreeMap::new();
    let mut glsl = None;
    let mut uses = Vec::new();
    let mut i = 5;
    while i < words.len() {
        let n = ((words[i] >> 16) as usize).max(1);
        let op = (words[i] & 0xFFFF) as u16;
        let w = &words[i + 1..(i + n).min(words.len())];
        match op {
            // OpExtInstImport
            11 => {
                let name: Vec<u8> = w[1..].iter().flat_map(|x| x.to_le_bytes()).collect();
                if name.starts_with(b"GLSL.std.450\0") {
                    glsl = Some(w[0]);
                }
            }
            // OpTypeFloat
            22 => {
                floats.insert(w[0], w[1]);
            }
            // OpConstant
            43 => {
                if let Some(&width) = floats.get(&w[0]) {
                    consts.insert(w[1], (width, w[2..].to_vec()));
                }
            }
            // OpConstantComposite, OpCompositeConstruct, OpCopyObject, OpSelect, OpPhi
            44 | 80 | 83 | 169 | 245 => {
                defs.insert(w[1], (op, w[2..].to_vec()));
            }
            // OpBitcast, OpIsNan, OpUGreaterThan, OpBitwiseAnd
            124 | 156 | 172 | 199 => {
                tests.insert(w[1], (op, w[2..].to_vec()));
            }
            // OpExtInst
            12 if Some(w[2]) == glsl => {
                let name = match w[3] {
                    37 => "FMin",
                    40 => "FMax",
                    79 => "NMin",
                    80 => "NMax",
                    _ => "",
                };
                if !name.is_empty() {
                    uses.extend(w[4..].iter().map(|&a| (name, a)));
                }
            }
            _ => {}
        }
        i += n;
    }
    // Whether `cond` is a NaN test of `x`.
    let tests_nan = |cond: u32, x: u32| match tests.get(&cond) {
        Some((156, a)) => a.first() == Some(&x),
        Some((172, a)) => a.first().and_then(|m| tests.get(m)).is_some_and(|(op, m)| {
            *op == 199
                && m.first()
                    .and_then(|b| tests.get(b))
                    .is_some_and(|(op, b)| *op == 124 && b.first() == Some(&x))
        }),
        _ => false,
    };
    let nan = |width: u32, v: &[u32]| match width {
        16 => v[0] & 0x7C00 == 0x7C00 && v[0] & 0x3FF != 0,
        32 => v[0] & 0x7F80_0000 == 0x7F80_0000 && v[0] & 0x7F_FFFF != 0,
        _ => v[1] & 0x7FF0_0000 == 0x7FF0_0000 && (v[1] & 0xF_FFFF != 0 || v[0] != 0),
    };
    let mut found = Vec::new();
    for (name, operand) in uses {
        let mut seen = std::collections::BTreeSet::new();
        let mut stack = vec![operand];
        while let Some(id) = stack.pop() {
            if !seen.insert(id) {
                continue;
            }
            if consts.get(&id).is_some_and(|(width, v)| nan(*width, v)) {
                found.push((name, id));
            }
            match defs.get(&id) {
                // OpSelect of Metal's NaN where its other value is a NaN: that value.
                Some((169, a)) if a.len() == 3 && tests_nan(a[0], a[2]) => stack.push(a[2]),
                // OpSelect: the two objects, not the condition.
                Some((169, a)) => stack.extend(&a[1..]),
                // OpPhi: the values, not the parent blocks.
                Some((245, a)) => stack.extend(a.iter().step_by(2)),
                Some((_, a)) => stack.extend(a),
                None => {}
            }
        }
    }
    found
}

/// No NaN reaches the `NMin` or `NMax` of an emulated float min or max. A NaN is their identity
/// only under `SignedZeroInfNanPreserve`, without which a device may assume that no operand is a
/// NaN, so a tree round skips an absent partial with a select instead, in a workgroup-uniform block
/// and in a divergent one (where each partial carries whether it holds an active lane). Every
/// operand of `NMin`, `NMax`, `FMin`, and `FMax` is walked through selects, composites, copies, and
/// phis to the constants it can take.
#[test]
fn no_nan_reaches_an_emulated_float_min_or_max() {
    for op in ["min", "max"] {
        let Some(bc) = float_reduction_kernels(op, false) else {
            return;
        };
        for stem in PROFILES {
            let p = profile(stem, DevicePolicy::default()).expect("profile");
            for kernel in ["simd_minmax_k", "simd_float_prefix_k"] {
                validate(stem, &bc, kernel);
                let a = compile(&bc, kernel, &p).unwrap_or_else(|e| panic!("{stem} {kernel}: {e}"));
                let (_, words) =
                    link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem} {kernel}: {e}"));
                let found = nans_reaching_min_or_max(&words);
                assert!(
                    found.is_empty(),
                    "{stem} {kernel} ({op}): a NaN reaches {found:?}"
                );
            }
        }
    }
}

/// The float operations of [`float_reduction_kernels`] the tests take.
const FLOAT_OPS: [&str; 5] = [
    "min",
    "max",
    "sum",
    "prefix_inclusive_sum",
    "prefix_exclusive_sum",
];

/// An emulated float min, max, sum, or prefix sum takes exactly the active lanes, whatever the
/// others hold, in full and partially occupied simdgroups (every local size) and in a divergent
/// block whose inactive lanes hold values that would win (`simd_float_prefix_k`'s lanes 0 to 3,
/// `-1000` and `1000`); an exclusive sum with no active lane below reads `+0`. Every third thread's
/// min or max operand is a NaN, which `NMin` and `NMax` skip as Metal's `fmin` and `fmax` do; a
/// simdgroup whose active lanes are all NaN would read NaN.
#[test]
fn emulated_float_reductions_and_scans_take_exactly_the_active_lanes() {
    for op in FLOAT_OPS {
        let nan = matches!(op, "min" | "max");
        let Some(bc) = float_reduction_kernels(op, nan) else {
            return;
        };
        for stem in PROFILES {
            let p = profile(stem, DevicePolicy::default()).expect("profile");
            for kernel in ["simd_minmax_k", "simd_float_prefix_k"] {
                let divergent = kernel == "simd_float_prefix_k";
                let (plan, m) = typed_module(&bc, kernel, &p)
                    .unwrap_or_else(|e| panic!("{stem} {kernel}: {e}"));
                check_emulated(stem, kernel, &plan, None);
                for lanes in WIDTHS {
                    for local in LOCALS {
                        let g = Group {
                            n: local.iter().product(),
                        };
                        let out = run_all(&plan, &m, local, lanes, None);
                        for (t, got) in out.iter().enumerate() {
                            let t = t as u32;
                            let at = format!("{stem} {kernel} {op} {local:?} {lanes:?} thread {t}");
                            if divergent && Group::lane(t) < 4 {
                                assert_eq!(*got, 7, "{at}");
                                continue;
                            }
                            let vals = g
                                .simd(t)
                                .filter(|&u| !divergent || Group::lane(u) >= 4)
                                .filter(|&u| match op {
                                    "prefix_inclusive_sum" => u <= t,
                                    "prefix_exclusive_sum" => u < t,
                                    _ => true,
                                })
                                .map(|u| float_input(u, divergent, nan))
                                .filter(|x| !x.is_nan());
                            let want = match op {
                                "min" => vals.reduce(f32::min),
                                "max" => vals.reduce(f32::max),
                                _ => Some(vals.fold(0.0, |a, b| a + b)),
                            };
                            match want {
                                Some(w) => assert_eq!(*got, w.to_bits(), "{at}"),
                                None => assert!(f32::from_bits(*got).is_nan(), "{at}"),
                            }
                        }
                    }
                }
            }
        }
    }
}

/// The lanes that do not contribute to an emulated float exclusive prefix sum add `-0`, and a lane
/// with no active lane below it reads `+0` (Vulkan's scan identity), in a workgroup-uniform block
/// (`float_prefix_k`) and a divergent one (`simd_float_prefix_k`, lanes 4 and up). Every operand
/// here is `-0` and the sum's bits are stored, so the lowest contributing lane holds `+0` and every
/// other lane `-0`.
#[test]
fn an_emulated_float_exclusive_sum_keeps_the_sign_of_zero() {
    let edit = |stem: &str, from: &str, to: &str| -> Option<Vec<u8>> {
        let text = std::fs::read_to_string(engine_dir().join(format!("{stem}.ll"))).expect("read");
        assert!(text.contains(from), "{stem}");
        assemble_text(&format!("signed-zero-{stem}"), &text.replacen(from, to, 1))
    };
    let Some(uniform) = edit(
        "simd_uniform",
        "  %5 = tail call fast float @air.simd_prefix_exclusive_sum.f32(float %4) #7\n  \
         %6 = tail call i32 @air.convert.u.i32.f.f32(float %5) #8\n",
        "  %n = fmul float %4, -0.000000e+00\n  \
         %5 = tail call float @air.simd_prefix_exclusive_sum.f32(float %n) #7\n  \
         %6 = bitcast float %5 to i32\n",
    ) else {
        return;
    };
    let Some(divergent) = edit(
        "simd_ops",
        "  %7 = tail call fast float @air.simd_prefix_exclusive_sum.f32(float %6) #3\n  \
         %8 = tail call i32 @air.convert.u.i32.f.f32(float %7) #4\n",
        "  %n = fmul float %6, -0.000000e+00\n  \
         %7 = tail call float @air.simd_prefix_exclusive_sum.f32(float %n) #3\n  \
         %8 = bitcast float %7 to i32\n",
    ) else {
        return;
    };
    let neg = (-0.0f32).to_bits();
    for stem in PROFILES {
        let p = profile(stem, DevicePolicy::default()).expect("profile");
        for (bc, kernel) in [
            (&uniform, "float_prefix_k"),
            (&divergent, "simd_float_prefix_k"),
        ] {
            validate(stem, bc, kernel);
            let (plan, m) =
                typed_module(bc, kernel, &p).unwrap_or_else(|e| panic!("{stem} {kernel}: {e}"));
            check_emulated(stem, kernel, &plan, None);
            for lanes in WIDTHS {
                for local in LOCALS {
                    let out = run_all(&plan, &m, local, lanes, None);
                    for (t, got) in out.iter().enumerate() {
                        let l = Group::lane(t as u32);
                        let want = match kernel {
                            // The half inclusive sum of `t & 7` beside it, times 10,000.
                            "float_prefix_k" => {
                                let b = Group::base(t as u32);
                                let incl: u32 = (b..=t as u32).map(|u| u & 7).sum();
                                (incl * 10_000).wrapping_add(if l == 0 { 0 } else { neg })
                            }
                            _ if l < 4 => 7,
                            _ if l == 4 => 0,
                            _ => neg,
                        };
                        assert_eq!(*got, want, "{stem} {kernel} {local:?} {lanes:?} thread {t}");
                    }
                }
            }
        }
    }
}

/// The emulated simdgroup barrier is a workgroup barrier (the simdgroup's threads may be in
/// different subgroups), and the scratch is exchanged between workgroup barriers.
#[test]
fn emulated_barriers_are_workgroup_barriers() {
    let Some(bc) = assemble("simd_uniform") else {
        return;
    };
    for stem in PROFILES {
        let p = profile(stem, DevicePolicy::default()).expect("profile");
        for kernel in ["barrier_k", "vote_k"] {
            let a = compile(&bc, kernel, &p).expect("compile");
            let (_, words) = link_and_validate(&a, &p).expect("valid");
            let text = spirv_dis(&words);
            assert!(
                text.contains("OpControlBarrier %uint_2 %uint_2"),
                "{stem} {kernel}\n{text}"
            );
            assert!(
                !text.contains("OpControlBarrier %uint_3"),
                "{stem} {kernel}: no subgroup-scope barrier"
            );
        }
    }
}

/// An operation outside workgroup-uniform control flow runs predicated: the lanes that do not run
/// it (their simdgroup's other arm, or past the threadgroup's end) take part in its exchange as
/// inactive lanes, so votes, ballots, `is_first`, `broadcast_first`, prefix sums, reductions, and
/// the quad's active mask see exactly the active lanes, at every subgroup width.
fn divergent_operations_run_predicated(stem: &str) {
    let Some(ops) = assemble("simd_ops") else {
        return;
    };
    let Some(uniform) = assemble("simd_uniform") else {
        return;
    };
    let Some(quads) = assemble("a3_quad_votes_divergent") else {
        return;
    };
    let kernels = SIMD_OPS_DIVERGENT.iter().map(|k| (&ops, *k)).chain([
        (&uniform, "divergent_k"),
        (&quads, "a3_quad_votes_divergent"),
    ]);
    for (bc, kernel) in kernels {
        validate(stem, bc, kernel);
        run_kernel(stem, bc, kernel, None);
    }
}

#[test]
fn divergent_operations_run_predicated_on_lavapipe() {
    divergent_operations_run_predicated("lavapipe");
}

#[test]
fn divergent_operations_run_predicated_on_vk12_floor() {
    divergent_operations_run_predicated("vk12-floor");
}

/// Operations in divergent control flow of every shape the walk predicates run exactly on the lanes
/// that execute them, at every subgroup width and in full and partial threadgroups.
fn simd_divergent_runs_predicated(stem: &str) {
    let Some(bc) = assemble("simd_divergent") else {
        return;
    };
    for kernel in SIMD_DIVERGENT {
        validate(stem, &bc, kernel);
        run_kernel(stem, &bc, kernel, None);
    }
}

#[test]
fn simd_divergent_runs_predicated_on_lavapipe() {
    simd_divergent_runs_predicated("lavapipe");
}

#[test]
fn simd_divergent_runs_predicated_on_roadmap_2022() {
    simd_divergent_runs_predicated("roadmap-2022");
}

/// The `OpControlBarrier`s of `kernel` of `bc` compiled on profile `stem` under `policy`.
fn barriers(stem: &str, policy: DevicePolicy, bc: &[u8], kernel: &str) -> usize {
    let p = profile(stem, policy).expect("profile");
    let a = compile(bc, kernel, &p).unwrap_or_else(|e| panic!("{stem} {kernel}: {e}"));
    let (_, words) = link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem} {kernel}: {e}"));
    spirv_dis(&words).matches("OpControlBarrier").count()
}

/// Under the identity domains the emulated tier's exchanges and votes use one scratch region and
/// each closes with a barrier, as under the default domains: every kernel of `simd_ops`,
/// `simd_uniform` and `simd_divergent` (shuffles, reductions, prefix sums, votes, ballots, quad
/// operations, vector and `half` values, exchanges in uniform and per-thread loops, after early
/// returns and in branch arms) gives the same expectation at every subgroup width in both
/// scheduling orders, in full and partial threadgroups, plans the scratch of one region, and has as
/// many barriers as under the default domains.
#[test]
fn identity_exchanges_keep_one_scratch_region() {
    const STEM: &str = "lavapipe";
    let (Some(ops), Some(uniform), Some(divergent)) = (
        assemble("simd_ops"),
        assemble("simd_uniform"),
        assemble("simd_divergent"),
    ) else {
        return;
    };
    let kernels = SIMD_OPS
        .iter()
        .chain(&SIMD_OPS_DIVERGENT)
        .map(|k| (&ops, *k, None))
        .chain(SIMD_UNIFORM.iter().map(|(k, w)| (&uniform, *k, Some(*w))))
        .chain([(&uniform, "divergent_k", None)])
        .chain(SIMD_DIVERGENT.iter().map(|k| (&divergent, *k, None)));
    for (bc, kernel, words) in kernels {
        run_kernel_under(STEM, identity(), bc, kernel, words);
        assert_eq!(
            barriers(STEM, identity(), bc, kernel),
            barriers(STEM, DevicePolicy::default(), bc, kernel),
            "{kernel}: barriers under the identity domains"
        );
    }
}

/// The tail of `a3_multi_level_break_tail`'s two-level break runs after the inner loop, once per
/// outer iteration, with every lane that left the inner loop through it in that iteration, when the
/// simdgroup is emulated, at every subgroup width. The fixture's 32 lanes stop at flat iteration
/// `7t mod 17` of a 3 x 4 nest; the expected words are the reading
/// `engine_metal_readings::a_multi_level_break_tail_runs_once_per_outer_iteration` checks
/// against Metal's.
#[test]
fn a_multi_level_break_tail_runs_once_per_outer_iteration_emulated() {
    const LANES: u32 = 32;
    const TRIPS: [u32; 2] = [3, 4];
    let kernel = "a3_multi_level_break_tail";
    let Some(bc) = assemble(kernel) else {
        return;
    };
    let stop: Vec<u32> = (0..LANES).map(|t| 7 * t % 17).collect();
    let (inner, flat) = (TRIPS[1], TRIPS[0] * TRIPS[1]);
    let want: Vec<u32> = stop
        .iter()
        .flat_map(|&s| {
            let mask = (0..LANES)
                .filter(|&u| {
                    let v = stop[u as usize];
                    s < flat && v < flat && v / inner == s / inner
                })
                .fold(0u32, |m, u| m | (1 << u));
            let at = if s < flat {
                ((s / inner) << 8) | (s % inner)
            } else {
                0xffff
            };
            [mask, at]
        })
        .collect();
    let bytes = |w: &[u32]| -> Vec<u8> { w.iter().flat_map(|x| x.to_le_bytes()).collect() };
    for stem in PROFILES {
        validate(stem, &bc, kernel);
        let p = profile(stem, DevicePolicy::default()).expect("profile");
        let (plan, m) = typed_module(&bc, kernel, &p).expect("plan");
        assert_eq!(plan.simd.1, Tier::Emulated, "{stem}");
        let slot = |kind, index| {
            Some(ResourceSlot {
                stage: Stage::Kernel,
                kind,
                index,
            })
        };
        let buffers: Vec<Buffer> = [bytes(&stop), bytes(&TRIPS), vec![0xab; 8 * LANES as usize]]
            .into_iter()
            .zip(0u32..)
            .map(|(b, i)| Buffer {
                slot: slot(ResourceKind::Buffer, i),
                address: None,
                bytes: b,
            })
            .collect();
        let mut spec = BTreeMap::new();
        spec.insert(SpecKey::LocalSize(Axis::X), u64::from(LANES));
        spec.insert(SpecKey::LocalSize(Axis::Y), 1);
        spec.insert(SpecKey::LocalSize(Axis::Z), 1);
        spec.insert(SpecKey::FlatLocalSize, u64::from(LANES));
        let res = Resources {
            buffers,
            push: vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)],
            spec,
        };
        for lanes in WIDTHS {
            let out = run_orders(&m, lanes, &res);
            assert_eq!(words(&out[2]), want, "{stem} {lanes:?}");
        }
    }
}

/// A matrix multiply-add in a divergent block compiles and passes L3 on every profile that emulates
/// the kernel's simdgroups, device addresses or none: a lane that is not live takes part with zero
/// cells, which needs no runtime input.
#[test]
fn a_divergent_matrix_multiply_add_compiles_on_every_emulating_profile() {
    let text = std::fs::read_to_string(engine_dir().join("simd_matrix.ll")).expect("read");
    // `matrix_thread_elements` runs only on lanes 0-15 of each simdgroup.
    let text = text.replacen(
        "  %5 = shl i32 %3, 1\n",
        "  %c = icmp ult i32 %3, 16\n  br i1 %c, label %go, label %done\n\ngo:\n  %5 = shl i32 %3, 1\n",
        1,
    );
    let text = text.replacen(
        "<2 x i64> zeroinitializer) #5\n  ret void\n}",
        "<2 x i64> zeroinitializer) #5\n  br label %done\n\ndone:\n  ret void\n}",
        1,
    );
    assert!(text.contains("label %go") && text.contains("\ndone:\n"));
    let Some(bc) = assemble_text("divergent-matrix", &text) else {
        return;
    };
    for stem in PROFILES {
        let p = profile(stem, DevicePolicy::default()).expect("profile");
        let a =
            compile(&bc, "matrix_thread_elements", &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert!(
            a.facts
                .iter()
                .any(|f| matches!(f, RuntimeFact::SimdgroupEmulation { .. })),
            "{stem}"
        );
        assert!(!a.facts.contains(&RuntimeFact::ExecutionStatus), "{stem}");
    }
}

/// `barrier_k` with its threadgroup array resized to `words` words and its stored value moved
/// through a `simd_shuffle_xor`, so the kernel exchanges one word per lane beside its arena.
fn arena_kernel(words: u32) -> Option<Vec<u8>> {
    let text = std::fs::read_to_string(engine_dir().join("simd_uniform.ll")).expect("read");
    let text = text.replace("[128 x i32]", &format!("[{words} x i32]"));
    let text = text.replacen(
        "  %6 = mul i32 %1, 3\n",
        "  %m = mul i32 %1, 3\n  %6 = tail call i32 @air.simd_shuffle_xor.u.i32(i32 %m, i16 1)\n",
        1,
    );
    let text = text.replacen(
        "\ndeclare void @air.simdgroup.barrier",
        "\ndeclare i32 @air.simd_shuffle_xor.u.i32(i32, i16)\n\ndeclare void @air.simdgroup.barrier",
        1,
    );
    assert!(text.contains("@air.simd_shuffle_xor.u.i32(i32 %m"));
    assemble_text(&format!("arena-{words}"), &text)
}

/// Metal reports a kernel's static threadgroup memory rounded up to 16 bytes: an arena of 257
/// words (1028 bytes) answers 1040, as `threadgroup uint a[256]; threadgroup uint b;` does on the
/// M3 Max (macOS 27.0.1).
#[test]
fn the_static_length_is_rounded_up_to_16_bytes_as_metal_reports_it() {
    let Some(bc) = arena_kernel(257) else {
        return;
    };
    let dev = lavapipe();
    let pipe = pipeline(&dev, &bc, "barrier_k");
    assert_eq!(pipe.metal_info.static_threadgroup_memory_length, 1040);
}

/// The scratch takes 128 bytes per simdgroup and word beside the kernel's threadgroup memory. A
/// kernel whose arena leaves less than one simdgroup's scratch is refused at compile time;
/// otherwise the pipeline reports the threads whose scratch fits as its maximum and its own
/// threadgroup memory as its static length, as Metal reports it, and the planner refuses a larger
/// threadgroup as `ThreadgroupMemorySize`.
#[test]
fn the_scratch_must_fit_beside_the_threadgroup_memory() {
    // roadmap-2022: 16 KiB. An arena of exactly 16 KiB leaves nothing for the scratch.
    let Some(full) = arena_kernel(4096) else {
        return;
    };
    let floor = profile("roadmap-2022", DevicePolicy::default()).expect("profile");
    match compile(&full, "barrier_k", &floor) {
        Err(Error::Unsupported(u)) => {
            assert_eq!(u.feature, Feature::ThreadgroupMemorySize);
            let needs = format!("{:?}", u.needs);
            assert!(
                needs.contains("MaxComputeSharedMemorySize")
                    && needs.contains("required: 16512")
                    && needs.contains("available: 16384"),
                "{needs}"
            );
        }
        other => panic!("{other:?}"),
    }
    // lavapipe: 32 KiB. An arena of 31 KiB leaves 1 KiB, the scratch of 8 simdgroups, which the
    // reported static length, the kernel's own 31 KiB, does not count.
    let Some(bc) = arena_kernel(31 * 256) else {
        return;
    };
    let dev = lavapipe();
    let pipe = pipeline(&dev, &bc, "barrier_k");
    assert_eq!(pipe.metal_info.max_total_threads_per_threadgroup, 256);
    assert_eq!(pipe.metal_info.static_threadgroup_memory_length, 31 * 1024);
    let state = bind_state(None);
    let plan = pipe.plan(&dev, &state.shape()).expect("plan");
    assert!(plan.instantiate_dispatch(&state, &threads(256)).is_ok());
    assert_memory_refused(plan.instantiate_dispatch(&state, &threads(257)), 32896);
}

/// A kernel that reads a `[[threadgroup(0)]]` region declares the threadgroup arena, a `u32` array
/// of at least `MIN_ARENA_BYTES` even when the region is empty, beside the scratch. `region_k`
/// publishes eight words per lane (1 KiB per simdgroup): lavapipe's 32 KiB fits 31 simdgroups
/// beside the 32-byte arena, not 32. The reported static length is the kernel's own, 0, as Metal
/// reports it: neither the arena's minimum nor the scratch, which the planner still counts, so the
/// remainder a Metal client gives the region beside that length is refused at the most threads
/// (the device's gap, `ThreadgroupMemorySize`), and a region that fits beside the scratch runs.
#[test]
fn an_empty_region_declares_the_arena_beside_the_scratch() {
    let Some(bc) = assemble("simd_uniform") else {
        return;
    };
    let dev = lavapipe();
    let pipe = pipeline(&dev, &bc, "region_k");
    let state = bind_state(Some(0));
    let plan = pipe.plan(&dev, &state.shape()).expect("plan");
    assert_memory_refused(
        plan.instantiate_dispatch(&state, &threads(1024)),
        32 + 32 * 1024,
    );
    assert!(plan.instantiate_dispatch(&state, &threads(31 * 32)).is_ok());
    assert_eq!(pipe.metal_info.max_total_threads_per_threadgroup, 31 * 32);
    assert_eq!(pipe.metal_info.static_threadgroup_memory_length, 0);
    // The remainder beside the reported static length, at the most threads: the arena and the
    // scratch of 31 simdgroups do not fit beside it.
    let remainder = 32 * 1024 - pipe.metal_info.static_threadgroup_memory_length;
    let state = bind_state(Some(remainder));
    let plan = pipe.plan(&dev, &state.shape()).expect("plan");
    match plan.instantiate_dispatch(&state, &threads(31 * 32)) {
        Err(BindError::Unsupported(u)) => assert_eq!(u.feature, Feature::ThreadgroupMemorySize),
        other => panic!("{other:?}"),
    }
    // A region that fits beside the arena and the scratch runs.
    let state = bind_state(Some(32 * 1024 - 32 - 31 * 1024));
    let plan = pipe.plan(&dev, &state.shape()).expect("plan");
    assert!(plan.instantiate_dispatch(&state, &threads(31 * 32)).is_ok());
}

/// An emulated operation exchanges through workgroup memory between workgroup barriers in
/// workgroup-uniform control flow, so its result does not depend on how subgroups reconverge; a
/// threadgroup barrier keeps the profile's reconvergence (`roadmap-2022` has no maximal
/// reconvergence).
#[test]
fn emulated_operations_do_not_depend_on_reconvergence() {
    let Some(bc) = assemble("simd_uniform") else {
        return;
    };
    for stem in PROFILES {
        let p = profile(stem, DevicePolicy::default()).expect("profile");
        for kernel in ["vote_k", "prefix_k", "quad_vote_k", "barrier_k"] {
            let a = compile(&bc, kernel, &p).unwrap_or_else(|e| panic!("{stem} {kernel}: {e}"));
            assert_eq!(
                a.reconvergence,
                Reconvergence::Guaranteed,
                "{stem} {kernel}"
            );
        }
    }
    let floor = profile("roadmap-2022", DevicePolicy::default()).expect("profile");
    assert!(!floor.fields().subgroup.maximal_reconvergence);
    let a = compile(&bc, "region_k", &floor).expect("region_k");
    assert_eq!(a.reconvergence, Reconvergence::ImplementationDefined);
}

/// On lavapipe without maximal reconvergence (its description with `shaderMaximalReconvergence`
/// cleared, as the runtime lane withholds the feature): the shuffles of `dsm2_shuffle_past_n` run
/// on the emulated tier (8-lane subgroups cannot hold Metal's simdgroups), every tangled
/// instruction is an emulated site, and the kernel's reconvergence is `Guaranteed`, compiled alone
/// and as the compute pipeline the runtime makes.
#[test]
fn emulated_shuffles_keep_metals_reconvergence_without_maximal_reconvergence() {
    let Some(bc) = assemble("dsm2_shuffle_past_n") else {
        return;
    };
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles/lavapipe.json");
    let text = std::fs::read_to_string(path).expect("profile");
    let cleared = text.replace(
        "\"shaderMaximalReconvergence\": true",
        "\"shaderMaximalReconvergence\": false",
    );
    assert_ne!(
        cleared, text,
        "the description names shaderMaximalReconvergence"
    );
    let desc = DeviceDescription::from_khronos_json(&cleared, None).expect("description");
    let dev = Device::from_queries(desc, DevicePolicy::default()).expect("device");
    let p = dev.profile();
    assert!(!p.fields().subgroup.maximal_reconvergence);
    let a = compile(&bc, "dsm2_shuffle_past_n", p).expect("compile");
    let names: Vec<&String> = a.strategies.entries.keys().collect();
    assert_eq!(
        a.reconvergence,
        Reconvergence::Guaranteed,
        "compiled alone: {names:?}"
    );
    let pipe = pipeline(&dev, &bc, "dsm2_shuffle_past_n");
    assert_eq!(pipe.stage.reconvergence, Reconvergence::Guaranteed);
}

/// The committed lavapipe profile (32 KiB of shared memory, 8-lane subgroups).
fn lavapipe() -> Device {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles/lavapipe.json");
    let text = std::fs::read_to_string(path).expect("profile");
    let desc =
        crate::device::DeviceDescription::from_khronos_json(&text, None).expect("description");
    let dev = Device::from_queries(desc, DevicePolicy::default()).expect("device");
    assert_eq!(
        dev.profile().fields().limits.max_compute_shared_memory_size,
        32 * 1024
    );
    dev
}

fn pipeline(dev: &Device, bc: &[u8], kernel: &str) -> ComputePipeline {
    let lib = Library::from_air(bc).expect("library");
    let f = lib.function(kernel).expect("kernel");
    let d = ComputePipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(f)));
    dev.compile_compute_pipeline(&d, &Budget::standard())
        .unwrap_or_else(|e| panic!("{kernel}: {e}"))
}

/// Buffer 0 bound, and `[[threadgroup(0)]]` set to `region` bytes.
fn bind_state(region: Option<u32>) -> BindState {
    let mut table = StageTable::default();
    table.buffers.insert(
        0,
        BoundBuffer::Buffer {
            length: 4096,
            offset: 0,
            gpu_address: 0x1_0000,
            device_address: None,
            attribute_stride: None,
        },
    );
    if let Some(len) = region {
        table.threadgroup_lengths.insert(0, len);
    }
    let mut state = BindState::default();
    state.stages.insert(Stage::Kernel, table);
    state
}

fn threads(n: u32) -> Dispatch {
    Dispatch::Threadgroups {
        groups: [1, 1, 1],
        threads_per_group: [n, 1, 1],
    }
}

fn assert_memory_refused<T: std::fmt::Debug>(r: Result<T, BindError>, required: u64) {
    match r {
        Err(BindError::Unsupported(u)) => {
            assert_eq!(u.feature, Feature::ThreadgroupMemorySize);
            let needs = format!("{:?}", u.needs);
            assert!(needs.contains(&format!("required: {required},")), "{needs}");
        }
        other => panic!("{other:?}"),
    }
}

/// The committed profile whose kernels run their simdgroups on whole 32-lane subgroups
/// (`Direct`, `REQUIRE_FULL_SUBGROUPS`): a threadgroup with a partial simdgroup runs on the
/// kernel's emulated-simdgroups variant.
const DIRECT: &str = "moltenvk-m3";

/// The authored cases' partial threadgroups (1, 4, 8, and 16 threads), a 2D threadgroup of 15
/// threads, and one full simdgroup.
const PARTIAL: [[u32; 3]; 6] = [
    [1, 1, 1],
    [4, 1, 1],
    [8, 1, 1],
    [16, 1, 1],
    [5, 3, 1],
    [32, 1, 1],
];

/// The lowering inputs of the emulated-simdgroups variant.
fn emulated_inputs() -> LoweringInputs {
    let mut inputs = LoweringInputs::new();
    inputs.emulated_simdgroups = true;
    inputs
}

/// The emulated-simdgroups variant's key.
fn emulated_key() -> VariantKey {
    let mut key = VariantKey::new();
    key.emulated_simdgroups = true;
    key
}

fn one_row_key() -> VariantKey {
    let mut key = VariantKey::new();
    key.one_row = true;
    key
}

/// `simd_divergent`'s `guard_k` (a simdgroup sum in divergent control flow) followed by `stages`
/// diamonds of plain arithmetic, each a branch on a per-thread value and a phi: a large body whose
/// predicated walk synchronizes in one block.
fn large_body(stages: u32) -> Option<Vec<u8>> {
    let text = std::fs::read_to_string(engine_dir().join("simd_divergent.ll")).expect("read");
    let start = text.find("define void @guard_k(").expect("guard_k");
    let merge = start + text[start..].find("\n16:").expect("guard_k's merge") + 1;
    let end = merge + text[merge..].find("\n}\n").expect("guard_k's end") + 1;
    let mut body = String::from("16:\n  br label %s0\n");
    let mut prev = String::from("%2");
    for i in 0..stages {
        let next = i + 1;
        body += &format!(
            "s{i}:\n  %k{i} = and i32 {prev}, 7\n  %c{i} = icmp ult i32 %k{i}, 3\n  \
             br i1 %c{i}, label %a{i}, label %b{i}\n\
             a{i}:\n  %y{i} = mul i32 {prev}, 3\n  br label %m{i}\n\
             b{i}:\n  %z{i} = add i32 {prev}, {next}\n  br label %m{i}\n\
             m{i}:\n  %w{i} = phi i32 [ %y{i}, %a{i} ], [ %z{i}, %b{i} ]\n  br label %s{next}\n"
        );
        prev = format!("%w{i}");
    }
    body += &format!(
        "s{stages}:\n  %at = zext i32 %2 to i64\n  \
         %out = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %at\n  \
         store i32 {prev}, ptr addrspace(1) %out, align 4\n  ret void\n"
    );
    let text = format!("{}{body}{}", &text[..merge], &text[end..]);
    assemble_text(&format!("large-body-{stages}"), &text)
}

/// A large body under the emulated-simdgroups input ([`large_body`]): its estimate bounds the
/// words it emits and stays within a small multiple of them (the walk is charged per construct it
/// emits), so it compiles under a words cap of that multiple of its emitted size.
#[test]
fn a_large_predicated_body_compiles_under_a_cap_it_fits() {
    let Some(bc) = large_body(256) else {
        return;
    };
    let p = profile("lavapipe", DevicePolicy::default()).expect("profile");
    let inputs = emulated_inputs();
    let (plan, _) = typed_module_with(&bc, "guard_k", &p, &inputs).expect("plan");
    assert!(
        plan.families.simd.predication.is_some(),
        "walked predicated"
    );
    let a = crate::pipeline::testing::compile_with(&bc, "guard_k", &p, &inputs).expect("compile");
    let emitted = a.module.words.len() as u64;
    assert!(
        plan.estimate.words >= emitted,
        "estimate {} < emitted {emitted}",
        plan.estimate.words
    );
    let cap = 8 * emitted;
    let capped = crate::pipeline::compile::compile_bitcode(
        &bc,
        crate::pipeline::compile::EntryRef::Name("guard_k"),
        &crate::library::FunctionConstantValues::defaults(),
        &p,
        &inputs,
        &mut crate::pipeline::testing::meter(),
        cap,
    );
    assert!(
        capped.is_ok(),
        "estimate {} under cap {cap} (emitted {emitted}): {:?}",
        plan.estimate.words,
        capped.err()
    );
}

/// `kernel` of `bc` on `DIRECT`: its own tier is `Direct` with a multiple-of-32 constraint, and
/// under the emulated-simdgroups input the `Emulated` tier, which depends on the variant, passes
/// L3 without a subgroup instruction, and gives the reading (a partial simdgroup's active lanes are
/// those below the thread count) at every width and partial threadgroup size.
fn run_variant(bc: &[u8], kernel: &str, words: Option<u32>) {
    let dev = direct();
    let p = dev.profile();
    let (own, _) = typed_module(bc, kernel, p).unwrap_or_else(|e| panic!("{kernel}: {e}"));
    assert_eq!(own.simd.1, Tier::Direct, "{kernel}");
    assert!(
        own.facts
            .contains(&RuntimeFact::LocalSizeConstraint { multiple: 32 }),
        "{kernel}"
    );
    assert!(!own.facts.contains(&RuntimeFact::EmulatedSimdgroups));
    let (plan, m) = typed_module_with(bc, kernel, p, &emulated_inputs())
        .unwrap_or_else(|e| panic!("{kernel}: {e}"));
    check_emulated(DIRECT, kernel, &plan, words);
    assert!(
        plan.facts.contains(&RuntimeFact::EmulatedSimdgroups),
        "{kernel}"
    );
    assert!(
        !plan
            .facts
            .iter()
            .any(|f| matches!(f, RuntimeFact::LocalSizeConstraint { .. })),
        "{kernel}: no multiple"
    );
    let f = Library::from_air(bc)
        .and_then(|lib| lib.function(kernel))
        .expect("kernel");
    let compiled = dev
        .compile_function(
            &FunctionDescriptor::new(f),
            &emulated_inputs(),
            &Budget::standard(),
        )
        .unwrap_or_else(|e| panic!("{kernel}: {e}"));
    let a = compiled.artifact();
    assert!(
        plan.estimate.words >= a.module.words.len() as u64,
        "{kernel}"
    );
    assert_eq!(a.subgroup, SubgroupRequirement::NONE, "{kernel}");
    let (_, words) = link_and_validate(a, p).unwrap_or_else(|e| panic!("{kernel}: {e}"));
    let text = spirv_dis(&words);
    assert!(!text.contains("GroupNonUniform"), "{kernel}: a subgroup op");
    run_expected(DIRECT, kernel, &plan, &m, &PARTIAL);
}

/// Shuffles, broadcasts, fills, reductions, bitwise folds, and quad operations of `simd_ops`;
/// votes, ballots, prefix sums, vector and `half` exchanges, a loop of exchanges, and a simdgroup
/// barrier over threadgroup memory of `simd_uniform`; and an exit guard, a threadgroup barrier
/// after it, and a per-thread trip count of `simd_divergent` (predicated) run on a `Direct`
/// kernel's emulated-simdgroups variant at 1, 4, 8, 16, 15 (5 x 3), and 32 threads.
#[test]
fn partial_simdgroups_run_on_the_emulated_variant_of_a_direct_kernel() {
    let Some(ops) = assemble("simd_ops") else {
        return;
    };
    let Some(uniform) = assemble("simd_uniform") else {
        return;
    };
    let Some(divergent) = assemble("simd_divergent") else {
        return;
    };
    for kernel in SIMD_OPS {
        run_variant(&ops, kernel, None);
    }
    for (kernel, words) in SIMD_UNIFORM {
        run_variant(&uniform, kernel, Some(words));
    }
    for kernel in [
        "guard_k",
        "guard_barrier_k",
        "trip_k",
        "guard_simd_barrier_k",
    ] {
        run_variant(&divergent, kernel, None);
    }
}

/// Through the public API on `DIRECT`: the plan of a kernel whose simdgroups run on whole subgroups
/// asks for the emulated-simdgroups variant exactly for the dispatches with a partial simdgroup
/// (and a `dispatchThreads` grid whose edge has one), instead of refusing them; the pipeline
/// compiled under that key states it in its plan and runs each of them, and a full one, with
/// Metal's local size and no subgroup requirement. A profile that emulates every kernel's
/// simdgroups never asks.
#[test]
fn the_plan_asks_for_the_emulated_variant_only_for_partial_simdgroups() {
    let Some(bc) = assemble("simd_ops") else {
        return;
    };
    let dev = direct();
    let budget = Budget::standard();
    let f = Library::from_air(&bc)
        .and_then(|lib| lib.function("simd_sum_k"))
        .expect("kernel");
    let fd = FunctionDescriptor::new(f);
    let base = pipeline(&dev, &bc, "simd_sum_k");
    let state = bind_state(None);
    let shape = state.shape();
    let plan = base.plan(&dev, &shape).expect("plan");
    assert!(plan.variant.is_empty());
    let key = emulated_key();
    let edge = Dispatch::Threads {
        threads: [40, 1, 1],
        threads_per_group: [32, 1, 1],
    };
    for d in [threads(1), threads(4), threads(8), threads(16), edge] {
        assert_eq!(plan.dispatch_variant(&d), Ok(key.clone()), "{d:?}");
        assert_eq!(
            plan.instantiate_dispatch(&state, &d),
            Err(BindError::NeedsVariant {
                variant: Box::new(key.clone())
            }),
            "{d:?}"
        );
    }
    // Whole simdgroups keep the flattened pipeline.
    for n in [32, 64] {
        assert_eq!(plan.dispatch_variant(&threads(n)), Ok(VariantKey::new()));
        let c = plan
            .instantiate_dispatch(&state, &threads(n))
            .expect("full");
        assert!(c.regions[0]
            .stage_flags
            .contains(vk::PipelineShaderStageCreateFlags::REQUIRE_FULL_SUBGROUPS));
    }

    // The variant the plan asked for.
    let compiled = dev
        .compile_function(&fd, &key.lowering_inputs(Stage::Kernel), &budget)
        .expect("variant");
    assert_ne!(
        *compiled.key(),
        dev.function_compile_key(&fd, &LoweringInputs::new())
    );
    let variant = dev
        .compile_compute_pipeline(
            &ComputePipelineDescriptor::new(StageSource::Compiled(compiled)),
            &budget,
        )
        .expect("variant pipeline");
    assert_ne!(variant.key, base.key);
    assert!(variant
        .dependencies
        .iter()
        .any(|d| d.fact == RuntimeFact::EmulatedSimdgroups && d.mechanism == Mechanism::Variant));
    assert_eq!(variant.stage.required_subgroup_size, None);
    assert!(variant.stage.stage_flags.is_empty());
    let vplan = variant.plan(&dev, &shape).expect("variant plan");
    assert_eq!(vplan.variant, key, "the variant's plan states its key");
    for d in [threads(1), threads(4), threads(16), threads(32), edge] {
        assert_eq!(vplan.dispatch_variant(&d), Ok(key.clone()), "{d:?}");
        let c = vplan.instantiate_dispatch(&state, &d).expect("runs");
        for r in &c.regions {
            assert!(r.stage_flags.is_empty(), "{d:?}");
            assert_eq!(r.required_subgroup_size, None, "{d:?}");
        }
    }

    // lavapipe emulates every kernel's simdgroups: no dispatch asks for a variant.
    let lava = lavapipe();
    let own = pipeline(&lava, &bc, "simd_sum_k");
    let lplan = own.plan(&lava, &shape).expect("plan");
    assert_eq!(lplan.dispatch_variant(&threads(16)), Ok(VariantKey::new()));
    assert!(lplan.instantiate_dispatch(&state, &threads(16)).is_ok());
}

/// The emulated tier's own limits stay refusals when a partial simdgroup asks for the variant. A
/// matrix multiply-add in a divergent block compiles in both; a kernel whose threadgroup memory
/// leaves no room for one simdgroup's scratch is `ThreadgroupMemorySize` at the variant's compile;
/// and one that leaves room for some is refused at the variant's plan for a threadgroup whose
/// scratch does not fit, which the `Direct` pipeline runs when its simdgroups are whole.
#[test]
fn the_emulated_tiers_limits_stay_refused_for_partial_simdgroups() {
    let dev = direct();
    let p = dev.profile();
    let emulated = |bc: &[u8], kernel: &str| {
        let f = Library::from_air(bc)
            .and_then(|lib| lib.function(kernel))
            .expect("kernel");
        dev.compile_function(
            &FunctionDescriptor::new(f),
            &emulated_inputs(),
            &Budget::standard(),
        )
    };

    let text = std::fs::read_to_string(engine_dir().join("simd_matrix.ll")).expect("read");
    let text = text.replacen(
        "  %5 = shl i32 %3, 1\n",
        "  %c = icmp ult i32 %3, 16\n  br i1 %c, label %go, label %done\n\ngo:\n  %5 = shl i32 %3, 1\n",
        1,
    );
    let text = text.replacen(
        "<2 x i64> zeroinitializer) #5\n  ret void\n}",
        "<2 x i64> zeroinitializer) #5\n  br label %done\n\ndone:\n  ret void\n}",
        1,
    );
    assert!(text.contains("label %go") && text.contains("\ndone:\n"));
    let Some(bc) = assemble_text("direct-divergent-matrix", &text) else {
        return;
    };
    compile(&bc, "matrix_thread_elements", p).expect("Direct");
    let variant = emulated(&bc, "matrix_thread_elements").expect("the variant");
    assert!(variant
        .artifact()
        .facts
        .contains(&RuntimeFact::EmulatedSimdgroups));

    // 32 KiB of shared memory: an arena of 32 KiB leaves nothing for the scratch.
    assert_eq!(p.fields().limits.max_compute_shared_memory_size, 32 * 1024);
    let Some(full) = arena_kernel(8192) else {
        return;
    };
    compile(&full, "barrier_k", p).expect("Direct");
    match emulated(&full, "barrier_k") {
        Err(Error::Unsupported(u)) => {
            assert_eq!(u.feature, Feature::ThreadgroupMemorySize);
            let needs = format!("{:?}", u.needs);
            assert!(needs.contains("required: 32896"), "{needs}");
        }
        other => panic!("{other:?}"),
    }

    // An arena of 31 KiB leaves the scratch of 8 simdgroups.
    let Some(bc) = arena_kernel(31 * 256) else {
        return;
    };
    let state = bind_state(None);
    let shape = state.shape();
    let base = pipeline(&dev, &bc, "barrier_k");
    let plan = base.plan(&dev, &shape).expect("plan");
    assert!(plan.instantiate_dispatch(&state, &threads(512)).is_ok());
    assert_eq!(
        plan.dispatch_variant(&threads(260)),
        Ok(emulated_key()),
        "a partial simdgroup"
    );
    let variant = dev
        .compile_compute_pipeline(
            &ComputePipelineDescriptor::new(StageSource::Compiled(
                emulated(&bc, "barrier_k").expect("variant"),
            )),
            &Budget::standard(),
        )
        .expect("variant pipeline");
    let vplan = variant.plan(&dev, &shape).expect("variant plan");
    assert!(vplan.instantiate_dispatch(&state, &threads(250)).is_ok());
    assert_memory_refused(
        vplan.instantiate_dispatch(&state, &threads(260)),
        31 * 1024 + 9 * 128,
    );
}

/// A `dispatchThreads` grid whose edge is a partial simdgroup runs whole on the variant, so its
/// interior is checked against the variant's limits, not the flattened tier's. On
/// nvidia-rtx50-linux (`Direct`, z at most 64), an interior of `[1, 1, 96]`, which the flattened
/// pipeline runs alone, asks for no variant; a grid of `[1, 1, 100]` in threadgroups of
/// `[1, 1, 96]` is a `LocalSize` refusal up front, and one in threadgroups of `[1, 1, 64]` asks for
/// the variant, whose plan runs both regions.
#[test]
fn a_mixed_dispatch_is_checked_against_the_variants_limits() {
    let Some(bc) = assemble("simd_ops") else {
        return;
    };
    let dev = device("nvidia-rtx50-linux");
    assert_eq!(
        dev.profile().fields().limits.max_compute_work_group_size[2],
        64
    );
    let (own, _) = typed_module(&bc, "simd_sum_k", dev.profile()).expect("simd_sum_k");
    assert_eq!(own.simd.1, Tier::Direct);
    let budget = Budget::standard();
    let state = bind_state(None);
    let shape = state.shape();
    let plan = pipeline(&dev, &bc, "simd_sum_k")
        .plan(&dev, &shape)
        .expect("plan");
    let f = Library::from_air(&bc)
        .and_then(|lib| lib.function("simd_sum_k"))
        .expect("kernel");
    let compiled = dev
        .compile_function(&FunctionDescriptor::new(f), &emulated_inputs(), &budget)
        .expect("variant");
    let vplan = dev
        .compile_compute_pipeline(
            &ComputePipelineDescriptor::new(StageSource::Compiled(compiled)),
            &budget,
        )
        .expect("variant pipeline")
        .plan(&dev, &shape)
        .expect("variant plan");
    let mixed = |z| Dispatch::Threads {
        threads: [1, 1, 100],
        threads_per_group: [1, 1, z],
    };

    let interior = Dispatch::Threads {
        threads: [1, 1, 96],
        threads_per_group: [1, 1, 96],
    };
    assert_eq!(plan.dispatch_variant(&interior), Ok(VariantKey::new()));
    assert!(plan.instantiate_dispatch(&state, &interior).is_ok());
    match plan.dispatch_variant(&mixed(96)) {
        Err(BindError::Unsupported(u)) => assert_eq!(
            u.feature,
            Feature::LocalSize {
                axis: Some(Axis::Z)
            }
        ),
        other => panic!("{other:?}"),
    }
    assert_eq!(plan.dispatch_variant(&mixed(64)), Ok(emulated_key()));
    let c = vplan
        .instantiate_dispatch(&state, &mixed(64))
        .expect("the variant runs both regions");
    assert_eq!(c.regions.len(), 2);
}

/// The committed moltenvk-m3 profile (32-lane subgroups that require whole subgroups).
fn direct() -> Device {
    device(DIRECT)
}

/// The committed profile `name`.
fn device(name: &str) -> Device {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles")
        .join(format!("{name}.json"));
    let text = std::fs::read_to_string(path).expect("profile");
    let desc =
        crate::device::DeviceDescription::from_khronos_json(&text, None).expect("description");
    Device::from_queries(desc, DevicePolicy::default()).expect("device")
}

/// `quad_ballot` and `simd_active_threads_mask` on emulated simdgroups, uniform and divergent (run
/// predicated), at every subgroup width and in full and partially occupied threadgroups: exactly
/// the lanes that run the operation set their bits.
fn simd_masks_run_emulated(stem: &str) {
    let Some(bc) = assemble("simd_masks") else {
        return;
    };
    for kernel in [
        "mask_quad_ballot_k",
        "mask_quad_ballot_divergent_k",
        "mask_simd_active_k",
        "mask_simd_active_divergent_k",
    ] {
        validate(stem, &bc, kernel);
        run_kernel(stem, &bc, kernel, None);
    }
}

#[test]
fn simd_masks_run_emulated_on_lavapipe() {
    simd_masks_run_emulated("lavapipe");
}

#[test]
fn simd_masks_run_emulated_on_vk12_floor() {
    simd_masks_run_emulated("vk12-floor");
}

/// The scratch holds one `u32[W]` slot per lane, so an exchange reaches a lane's word with one
/// access chain at the lane and the constant word, and no slot address is multiplied out (`lane ·
/// W + i` was an `OpIMul` and an `OpIAdd` per word). The predicated shuffle publishes its value and
/// the active flag (`W = 2`).
#[test]
fn an_emulated_exchange_reaches_a_word_with_one_access_chain() {
    const VARIABLE: u32 = 59;
    const ACCESS_CHAIN: u32 = 65;
    const IMUL: u32 = 132;
    const WORKGROUP: u32 = 4;
    let Some(bc) = assemble("simd_divergent") else {
        return;
    };
    for stem in PROFILES {
        let p = profile(stem, DevicePolicy::default()).expect("profile");
        let words = compile(&bc, "guard_shuffle_k", &p)
            .expect("compile")
            .module
            .words
            .clone();
        let mut insts = Vec::new();
        let mut i = 5;
        while let Some(w) = words.get(i) {
            let len = ((w >> 16) as usize).max(1);
            insts.push(&words[i..(i + len).min(words.len())]);
            i += len;
        }
        let op = |inst: &[u32]| inst[0] & 0xFFFF;
        let scratch: Vec<u32> = insts
            .iter()
            .filter(|x| op(x) == VARIABLE && x.get(3) == Some(&WORKGROUP))
            .map(|x| x[2])
            .collect();
        let products: Vec<u32> = insts
            .iter()
            .filter(|x| op(x) == IMUL)
            .map(|x| x[2])
            .collect();
        let chains: Vec<&[u32]> = insts
            .iter()
            .filter(|x| op(x) == ACCESS_CHAIN && scratch.contains(&x[3]))
            .copied()
            .collect();
        assert!(
            chains.iter().any(|c| c.len() == 6),
            "{stem}: no lane-and-word access chain into the scratch"
        );
        for c in chains {
            assert!(
                c[4..].iter().all(|ix| !products.contains(ix)),
                "{stem}: a scratch word addressed through a product: {c:?}"
            );
        }
    }
}

/// `kernel void name(device uint *b [[buffer(0)]], constant uint &n [[buffer(1)]], uint gid
/// [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]])`: `if (gid < n)
/// b[index] = b[index] * 3 + extra`, `index` and `extra` the instructions `index` and `extra`
/// compute (`%x`, an `i64`, and `%e`, an `i32`) from `%gid` and `%lane`. The read and the write
/// of one word stand where only some threads of a simdgroup run, so the lockstep pass orders
/// them with a simdgroup barrier, a tangle barrier on the RTX 5080's maximal reconvergence.
fn row_kernel(name: &str, index: &str, extra: &str) -> String {
    format!(
        r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

define void @{name}(ptr addrspace(1) %b, ptr addrspace(2) %np, i32 %gid, i32 %lane) #0 {{
entry:
  %n = load i32, ptr addrspace(2) %np, align 4
  %in = icmp ult i32 %gid, %n
  br i1 %in, label %then, label %end
then:
  {index}
  {extra}
  %p = getelementptr inbounds i32, ptr addrspace(1) %b, i64 %x
  %v = load i32, ptr addrspace(1) %p, align 4
  %m = mul i32 %v, 3
  %w = add i32 %m, %e
  store i32 %w, ptr addrspace(1) %p, align 4
  br label %end
end:
  ret void
}}

attributes #0 = {{ convergent mustprogress nounwind willreturn }}

!air.kernel = !{{!0}}
!air.version = !{{!7}}
!air.language_version = !{{!8}}

!0 = !{{ptr @{name}, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4, !5, !6}}
!3 = !{{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"b"}}
!4 = !{{i32 1, !"air.buffer", !"air.buffer_size", i32 4, !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"n"}}
!5 = !{{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}}
!6 = !{{i32 3, !"air.thread_index_in_simdgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"lane"}}
!7 = !{{i32 2, i32 8, i32 0}}
!8 = !{{!"Metal", i32 4, i32 0, i32 0}}
"#
    )
}

/// Buffers 0 and 1 bound.
fn two_buffers() -> BindState {
    let mut state = bind_state(None);
    if let Some(table) = state.stages.get_mut(&Stage::Kernel) {
        table.buffers.insert(
            1,
            BoundBuffer::Buffer {
                length: 4,
                offset: 0,
                gpu_address: 0x2_0000,
                device_address: None,
                attribute_stride: None,
            },
        );
    }
    state
}

/// A kernel whose only simdgroup semantics are the barrier its lockstep order took, between a
/// read and a write of `b[gid]` that two threads of one simdgroup meet at only where the
/// simdgroup spans two rows of its threadgroup (the shape of OpenCL C's `if (i < n) b[i] *= k`),
/// keeps the RTX 5080's `Direct` tier and records `RowLockstep`. A threadgroup of one row (1, 16
/// or 48 threads, or a whole 32) asks for its one-row variant (`VariantKey::one_row`), and the
/// planned pipeline still runs one without whole subgroups: no emulated-simdgroups variant is
/// compiled, so none is announced. A threadgroup of two rows that is not whole subgroups still
/// asks for the emulated variant. The same kernel at `b[gid >> 1]` (two threads of a row meet) or
/// reading its simdgroup lane keeps every partial threadgroup on the emulated variant and a whole
/// one on its pinned pipeline.
#[test]
fn a_row_of_a_lockstep_only_kernel_takes_one_compile() {
    let x = "%x = zext i32 %gid to i64";
    let none = "%e = add i32 0, 0";
    let kernels = [
        ("row_scale", x, none, true),
        (
            "row_scale_pairs",
            "%h = lshr i32 %gid, 1\n  %x = zext i32 %h to i64",
            none,
            false,
        ),
        ("row_scale_lane", x, "%e = add i32 %lane, 0", false),
    ];
    let dev = device("nvidia-rtx5080-laptop");
    let p = dev.profile();
    let state = two_buffers();
    let shape = state.shape();
    for (name, index, extra, rows) in kernels {
        let Some(bc) = assemble_text(name, &row_kernel(name, index, extra)) else {
            return;
        };
        let (own, _) = typed_module(&bc, name, p).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(own.simd.1, Tier::Direct, "{name}");
        assert!(
            own.facts
                .contains(&RuntimeFact::LocalSizeConstraint { multiple: 32 }),
            "{name}"
        );
        assert_eq!(
            own.facts.contains(&RuntimeFact::RowLockstep),
            rows,
            "{name}"
        );
        let base = pipeline(&dev, &bc, name);
        let plan = base.plan(&dev, &shape).expect("plan");
        for n in [1, 16, 48] {
            let want = if rows { one_row_key() } else { emulated_key() };
            assert_eq!(
                plan.dispatch_variant(&threads(n)),
                Ok(want),
                "{name} at {n}"
            );
        }
        for n in [32, 64] {
            let want = if rows {
                one_row_key()
            } else {
                VariantKey::new()
            };
            assert_eq!(
                plan.dispatch_variant(&threads(n)),
                Ok(want),
                "{name} at {n}"
            );
        }
        let two_rows = Dispatch::Threadgroups {
            groups: [1, 1, 1],
            threads_per_group: [8, 2, 1],
        };
        assert_eq!(
            plan.dispatch_variant(&two_rows),
            Ok(emulated_key()),
            "{name}"
        );
        if rows {
            let c = plan
                .instantiate_dispatch(&state, &threads(16))
                .expect("one row on the planned pipeline");
            assert_eq!(
                c.regions[0].stage_flags,
                vk::PipelineShaderStageCreateFlags::empty()
            );
            assert_eq!(c.regions[0].required_subgroup_size, Some(32));
            let c = plan
                .instantiate_dispatch(&state, &threads(32))
                .expect("whole subgroups");
            assert!(c.regions[0]
                .stage_flags
                .contains(vk::PipelineShaderStageCreateFlags::REQUIRE_FULL_SUBGROUPS));
        }
    }
}

/// `kernel void name(device uint *out [[buffer(0)]], uint2 tid [[thread_position_in_threadgroup]],
/// uint t [[thread_index_in_threadgroup]])`, a tile's shape as a compiler spells it:
///
/// - every thread stores `t` at word `t` of `@nb`, and after a barrier reads word `t ^ 8` and
///   overwrites its own with 100 in the same interval, in workgroup-uniform flow (a pair the
///   lockstep pass orders with a threadgroup barrier, in every threadgroup shape);
/// - then, where `run` holds (`x >= 32` in row 0, `x < 16` in row 1, where only some threads of a
///   simdgroup run), it stores `t << 4 | k` at word `(x & 31) + 4 (x & -32) + 32 k` of `@grid`
///   and `t << 4 | 8 | k` 512 words on, `k` below 4. Two threads' anchors differ by a multiple of
///   128 words plus at most 31: the rows `k` never meet, and the halves meet only where two
///   threads of one simdgroup share a lane and lie 128 apart in `x`, as in a threadgroup 144 wide
///   and 2 tall (row 0's `x` 128 to 143 and row 1's 0 to 15 are simdgroup 4), a pair a simdgroup
///   barrier orders and no row of a threadgroup holds;
/// - after a barrier, `out[t]` is its own word `k = 0` of the tile where it ran (0 elsewhere),
///   with what it read from `@nb` 20 bits up.
///
/// `extra` and `decl` add an instruction to the entry and its declaration.
fn row_tile_kernel(name: &str, extra: &str, decl: &str) -> String {
    format!(
        r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

@grid = internal unnamed_addr addrspace(3) global [2048 x i32] undef, align 4
@nb = internal unnamed_addr addrspace(3) global [512 x i32] undef, align 4

define void @{name}(ptr addrspace(1) %out, <2 x i32> %tid, i32 %t) #0 {{
entry:
  {extra}
  %tz = zext i32 %t to i64
  %own = getelementptr inbounds [512 x i32], ptr addrspace(3) @nb, i64 0, i64 %tz
  store i32 %t, ptr addrspace(3) %own, align 4
  tail call void @air.wg.barrier(i32 2, i32 1)
  %t8 = xor i32 %t, 8
  %t8z = zext i32 %t8 to i64
  %other = getelementptr inbounds [512 x i32], ptr addrspace(3) @nb, i64 0, i64 %t8z
  %nv = load i32, ptr addrspace(3) %other, align 4
  store i32 100, ptr addrspace(3) %own, align 4
  %x = extractelement <2 x i32> %tid, i64 0
  %y = extractelement <2 x i32> %tid, i64 1
  %lane = and i32 %x, 31
  %high = and i32 %x, -32
  %l = zext i32 %lane to i64
  %h4 = shl nuw nsw i32 %high, 2
  %h = zext i32 %h4 to i64
  %col = getelementptr inbounds i32, ptr addrspace(3) @grid, i64 %l
  %at = getelementptr inbounds i32, ptr addrspace(3) %col, i64 %h
  %row0 = icmp eq i32 %y, 0
  %right = icmp uge i32 %x, 32
  %left = icmp ult i32 %x, 16
  %run = select i1 %row0, i1 %right, i1 %left
  br i1 %run, label %fill, label %done
fill:
  %v0 = shl i32 %t, 4
  store i32 %v0, ptr addrspace(3) %at, align 4
  %a1 = getelementptr inbounds i32, ptr addrspace(3) %at, i64 32
  %v1 = or i32 %v0, 1
  store i32 %v1, ptr addrspace(3) %a1, align 4
  %a2 = getelementptr inbounds i32, ptr addrspace(3) %at, i64 64
  %v2 = or i32 %v0, 2
  store i32 %v2, ptr addrspace(3) %a2, align 4
  %a3 = getelementptr inbounds i32, ptr addrspace(3) %at, i64 96
  %v3 = or i32 %v0, 3
  store i32 %v3, ptr addrspace(3) %a3, align 4
  %b0 = getelementptr inbounds i32, ptr addrspace(3) %at, i64 512
  %w0 = or i32 %v0, 8
  store i32 %w0, ptr addrspace(3) %b0, align 4
  %b1 = getelementptr inbounds i32, ptr addrspace(3) %at, i64 544
  %w1 = or i32 %v0, 9
  store i32 %w1, ptr addrspace(3) %b1, align 4
  %b2 = getelementptr inbounds i32, ptr addrspace(3) %at, i64 576
  %w2 = or i32 %v0, 10
  store i32 %w2, ptr addrspace(3) %b2, align 4
  %b3 = getelementptr inbounds i32, ptr addrspace(3) %at, i64 608
  %w3 = or i32 %v0, 11
  store i32 %w3, ptr addrspace(3) %b3, align 4
  br label %done
done:
  tail call void @air.wg.barrier(i32 2, i32 1)
  br i1 %run, label %read, label %write
read:
  %got = load i32, ptr addrspace(3) %at, align 4
  br label %write
write:
  %word = phi i32 [ %got, %read ], [ 0, %done ]
  %up = shl i32 %nv, 20
  %both = or i32 %word, %up
  %o = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %tz
  store i32 %both, ptr addrspace(1) %o, align 4
  ret void
}}

declare void @air.wg.barrier(i32, i32) #1
{decl}

attributes #0 = {{ convergent mustprogress nounwind willreturn }}
attributes #1 = {{ convergent mustprogress nounwind willreturn }}

!air.kernel = !{{!0}}
!air.version = !{{!6}}
!air.language_version = !{{!7}}

!0 = !{{ptr @{name}, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4, !5}}
!3 = !{{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}}
!4 = !{{i32 1, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"uint2", !"air.arg_name", !"tid"}}
!5 = !{{i32 2, !"air.thread_index_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"t"}}
!6 = !{{i32 2, i32 8, i32 0}}
!7 = !{{!"Metal", i32 4, i32 0, i32 0}}
"#
    )
}

/// The words [`row_tile_kernel`] leaves in one threadgroup of `local` threads, as an Apple M3
/// Max runs it: a thread where it ran reads its own tile word, but in row 0 at `x` 128 to 143 of
/// a threadgroup of two rows, where row 1's thread `x - 128` (one simdgroup with it) stores its
/// second half there one instruction later, which lockstep leaves; each reads `t ^ 8` from
/// `@nb` before its neighbour overwrites it.
fn row_tile_words(local: [u32; 3]) -> Vec<u32> {
    let w = local[0];
    (0..local[0] * local[1])
        .map(|t| {
            let (x, y) = (t % w, t / w);
            let run = if y == 0 { x >= 32 } else { x < 16 };
            let word = if !run {
                0
            } else if y == 0 && local[1] > 1 && (128..144).contains(&x) {
                ((w + x - 128) << 4) | 8
            } else {
                t << 4
            };
            word | ((t ^ 8) << 20)
        })
        .collect()
}

/// Every word of buffer 0 after one threadgroup of `local` threads ran `m` (planned as `plan`)
/// with `lanes`-wide subgroups, in both scheduling orders, its local size also flattened (the
/// pinned tiers' Vulkan workgroup).
fn run_local(plan: &Plan, m: &TypedModule, local: [u32; 3], lanes: SubgroupSize) -> Vec<u32> {
    let mut spec = BTreeMap::new();
    for (a, v) in [Axis::X, Axis::Y, Axis::Z].into_iter().zip(local) {
        spec.insert(SpecKey::LocalSize(a), u64::from(v));
    }
    let n: u32 = local.iter().product();
    spec.insert(SpecKey::FlatLocalSize, u64::from(n));
    let res = Resources {
        buffers: vec![Buffer {
            slot: Some(ResourceSlot {
                stage: Stage::Kernel,
                kind: ResourceKind::Buffer,
                index: 0,
            }),
            address: None,
            bytes: vec![0; n as usize * 4],
        }],
        push: vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)],
        spec,
    };
    words(&run_orders(m, lanes, &res)[0])
}

/// A threadgroup of one row runs a tile whose halves meet only across rows without its lockstep
/// simdgroup barrier ([`row_tile_kernel`], on the Iris Xe, whose maximal reconvergence on
/// 32-lane subgroups makes that barrier a tangle barrier). Its own pipeline keeps the barrier,
/// takes `CrossLane(BASIC)` and `Direct` (32 lanes pinned), records `RowLockstep`, and leaves
/// Metal's words in a threadgroup 144 wide and 2 tall, which runs on it, the shared lane's word
/// included. A dispatch of one row (128 or 48 threads) asks for the one-row variant, which keeps
/// the threadgroup barrier the `@nb` pair takes, has no simdgroup barrier, no simd class and no
/// pinned subgroup, states `OneRow`, and leaves Metal's words at every width; its plan sends a
/// threadgroup of two rows back. The same kernel with a simdgroup sum keeps its pinned pipeline
/// for every threadgroup of whole simdgroups, one row or not.
#[test]
fn a_one_row_dispatch_drops_the_lockstep_simdgroup_barriers() {
    let dev = device("intel-irisxe-rplp-anv");
    let p = dev.profile();
    let name = "row_tile";
    let Some(bc) = assemble_text(name, &row_tile_kernel(name, "", "")) else {
        return;
    };
    let wide = [144, 2, 1];

    // The kernel's own pipeline.
    let (own, m) = typed_module(&bc, name, p).unwrap_or_else(|e| panic!("{name}: {e}"));
    let uses = |plan: &Plan, k: &str| plan.census().get(k).map_or(0, |u| u.0);
    assert!(
        uses(&own, "barrier.lockstep.simdgroup") > 0,
        "{:?}",
        own.census()
    );
    assert_eq!(
        uses(&own, "barrier.lockstep.threadgroup"),
        1,
        "{:?}",
        own.census()
    );
    assert_eq!(
        own.simd,
        (SimdClass::CrossLane(OpBits::BASIC), Tier::Direct)
    );
    assert!(own.facts.contains(&RuntimeFact::RowLockstep));
    assert!(!own.facts.contains(&RuntimeFact::OneRow));
    assert_eq!(
        run_local(&own, &m, wide, SubgroupSize::W32),
        row_tile_words(wide)
    );

    // The one-row variant.
    let mut inputs = LoweringInputs::new();
    inputs.one_row = true;
    let (row, rm) =
        typed_module_with(&bc, name, p, &inputs).unwrap_or_else(|e| panic!("{name}: {e}"));
    assert_eq!(
        uses(&row, "barrier.lockstep.simdgroup"),
        0,
        "{:?}",
        row.census()
    );
    assert_eq!(
        uses(&row, "barrier.lockstep.threadgroup"),
        1,
        "{:?}",
        row.census()
    );
    assert_eq!(row.simd, (SimdClass::None, Tier::NoSubgroup));
    assert_eq!(row.subgroup, SubgroupRequirement::NONE);
    assert!(row.facts.contains(&RuntimeFact::OneRow));
    assert!(!row.facts.contains(&RuntimeFact::RowLockstep));
    for local in [[128, 1, 1], [48, 1, 1]] {
        for lanes in [SubgroupSize::W8, SubgroupSize::W16, SubgroupSize::W32] {
            assert_eq!(
                run_local(&row, &rm, local, lanes),
                row_tile_words(local),
                "{local:?} at {lanes:?}"
            );
        }
    }

    // The binder's choice.
    let state = bind_state(None);
    let shape = state.shape();
    let plan = pipeline(&dev, &bc, name).plan(&dev, &shape).expect("plan");
    let two_rows = Dispatch::Threadgroups {
        groups: [1, 1, 1],
        threads_per_group: wide,
    };
    assert_eq!(plan.dispatch_variant(&threads(128)), Ok(one_row_key()));
    assert_eq!(plan.dispatch_variant(&threads(48)), Ok(one_row_key()));
    assert_eq!(plan.dispatch_variant(&two_rows), Ok(VariantKey::new()));
    let f = Library::from_air(&bc)
        .ok()
        .and_then(|lib| lib.function(name).ok())
        .expect("kernel");
    let budget = Budget::standard();
    let compiled = dev
        .compile_function(
            &FunctionDescriptor::new(f),
            &one_row_key().lowering_inputs(Stage::Kernel),
            &budget,
        )
        .expect("variant");
    let variant = dev
        .compile_compute_pipeline(
            &ComputePipelineDescriptor::new(StageSource::Compiled(compiled)),
            &budget,
        )
        .expect("variant pipeline");
    assert_eq!(variant.stage.required_subgroup_size, None);
    let vplan = variant.plan(&dev, &shape).expect("variant plan");
    assert_eq!(
        vplan.variant,
        one_row_key(),
        "the variant's plan states its key"
    );
    assert_eq!(vplan.dispatch_variant(&threads(128)), Ok(one_row_key()));
    assert!(vplan.instantiate_dispatch(&state, &threads(48)).is_ok());
    assert_eq!(vplan.dispatch_variant(&two_rows), Ok(VariantKey::new()));
    assert_eq!(
        vplan.instantiate_dispatch(&state, &two_rows),
        Err(BindError::NeedsVariant {
            variant: Box::new(VariantKey::new())
        })
    );

    // With a simdgroup operation of its own the kernel keeps its pinned pipeline.
    let summed = "row_tile_sum";
    let Some(bc) = assemble_text(
        summed,
        &row_tile_kernel(
            summed,
            "%s = tail call i32 @air.simd_sum.u.i32(i32 %t)",
            "declare i32 @air.simd_sum.u.i32(i32) #1",
        ),
    ) else {
        return;
    };
    let (own, _) = typed_module(&bc, summed, p).unwrap_or_else(|e| panic!("{summed}: {e}"));
    assert_eq!(own.simd.1, Tier::Direct);
    assert!(!own.facts.contains(&RuntimeFact::RowLockstep));
    let plan = pipeline(&dev, &bc, summed)
        .plan(&dev, &shape)
        .expect("plan");
    for d in [threads(32), threads(128), two_rows] {
        assert_eq!(plan.dispatch_variant(&d), Ok(VariantKey::new()), "{d:?}");
    }
}

/// A wave64-only device: the committed RDNA3 description, which reconverges maximally, with its
/// subgroup size range narrowed to 64, where a simdgroup is half a subgroup (`Cluster32`).
fn wave64() -> Device {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles/amd-rdna3-radv.json");
    let text = std::fs::read_to_string(path).expect("profile");
    let narrowed = text.replace("\"minSubgroupSize\": 32", "\"minSubgroupSize\": 64");
    assert_ne!(narrowed, text, "the description names minSubgroupSize");
    let desc = DeviceDescription::from_khronos_json(&narrowed, None).expect("description");
    Device::from_queries(desc, DevicePolicy::default()).expect("device")
}

/// `kernel void name(texture2d_ms<uint> t [[texture(0)]], device uint *out [[buffer(0)]], uint2
/// g [[thread_position_in_grid]])`, `wsm_samples`'s `wsm_read_u` as the Metal compiler emits it:
/// `n = t.get_num_samples()`, and where `g` is inside the texture and `guard` holds (`%iny`, `g.y`
/// inside, and `%some`, `n != 0`), `for (s = 0; ; s++) { out[s + q] = t.read(g, s).r; if (tested
/// == n) break; }` with `q = (g.y * width + g.x) * mult`. The compiler's own spelling is `guard`
/// `select i1 %iny, i1 %some, i1 false`, `mult` `%n` and `tested` `%next` (`s + 1`).
fn cells_kernel(name: &str, guard: &str, mult: &str, tested: &str) -> String {
    format!(
        r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

define void @{name}(ptr addrspace(1) %t, ptr addrspace(1) %out, <2 x i32> %g) #0 {{
entry:
  %n = tail call i32 @air.get_num_samples_texture_2d_ms(ptr addrspace(1) %t)
  %x = extractelement <2 x i32> %g, i64 0
  %w = tail call i32 @air.get_width_texture_2d_ms(ptr addrspace(1) %t)
  %inx = icmp ult i32 %x, %w
  br i1 %inx, label %row, label %end
row:
  %y = extractelement <2 x i32> %g, i64 1
  %h = tail call i32 @air.get_height_texture_2d_ms(ptr addrspace(1) %t)
  %iny = icmp ult i32 %y, %h
  %some = icmp ne i32 %n, 0
  %go = {guard}
  br i1 %go, label %pre, label %end
pre:
  %yw = mul i32 %w, %y
  %p = add i32 %yw, %x
  %q = mul i32 %p, {mult}
  br label %loop
loop:
  %s = phi i32 [ 0, %pre ], [ %next, %loop ]
  %smp = tail call ptr addrspace(2) @air.get_read_sampler()
  %r = tail call {{ <4 x i32>, i8 }} @air.read_texture_2d_ms.u.v4i32(ptr addrspace(1) %t, ptr addrspace(2) %smp, <2 x i32> %g, i32 %s, i32 1)
  %rv = extractvalue {{ <4 x i32>, i8 }} %r, 0
  %v = extractelement <4 x i32> %rv, i64 0
  %i = add i32 %s, %q
  %iz = zext i32 %i to i64
  %at = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %iz
  store i32 %v, ptr addrspace(1) %at, align 4
  %next = add nuw i32 %s, 1
  %done = icmp eq i32 {tested}, %n
  br i1 %done, label %end, label %loop
end:
  ret void
}}

declare i32 @air.get_num_samples_texture_2d_ms(ptr addrspace(1))
declare i32 @air.get_width_texture_2d_ms(ptr addrspace(1))
declare i32 @air.get_height_texture_2d_ms(ptr addrspace(1))
declare ptr addrspace(2) @air.get_read_sampler()
declare {{ <4 x i32>, i8 }} @air.read_texture_2d_ms.u.v4i32(ptr addrspace(1), ptr addrspace(2), <2 x i32>, i32, i32)

attributes #0 = {{ convergent mustprogress nounwind willreturn }}

!air.kernel = !{{!0}}
!air.version = !{{!6}}
!air.language_version = !{{!7}}

!0 = !{{ptr @{name}, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4, !5}}
!3 = !{{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"texture2d_ms<uint, read>", !"air.arg_name", !"t"}}
!4 = !{{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}}
!5 = !{{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"g"}}
!6 = !{{i32 2, i32 8, i32 0}}
!7 = !{{!"Metal", i32 4, i32 0, i32 0}}
"#
    )
}

/// A multisampled texture of four samples in `format` bound at texture 0 for reading, and
/// buffer 0.
fn samples_state(format: crate::metal::PixelFormat) -> BindState {
    let mut state = bind_state(None);
    let mut facts =
        crate::bind::TextureFacts::new(crate::metal::TextureType::Type2DMultisample, format);
    facts.sample_count = 4;
    facts.usage = crate::metal::TextureUsage::SHADER_READ;
    if let Some(table) = state.stages.get_mut(&Stage::Kernel) {
        table.textures.insert(
            0,
            crate::bind::BoundTexture::Texture {
                facts,
                resource_id: None,
            },
        );
    }
    state
}

/// A texel's samples read into a buffer, `out[(y * w + x) * n + s]` over `s < n` with `n` the
/// texture's sample count (`wsm_samples`'s kernels, as the Metal compiler emits them): the store
/// meets itself in a later iteration only where two threads of a simdgroup share a pixel, which
/// the analysis cannot rule out where a simdgroup spans two rows of its threadgroup (a grid deeper
/// than one gives two threads one `(x, y)`), so on a wave64 device the kernel keeps its barrier, a
/// `Cluster32` tier, and a multiple of 64 threads. In a row its threads' `x` differ and their `y`
/// are equal, and each thread's cells `(y * w + x) * n + s` lie below the next pixel's, `s` a
/// counter that starts at 0 below `n` (the branch before the loop tested `n != 0`, as a `select`
/// or an `and`) and leaves the loop on reaching it: the kernel records `RowLockstep`, and a
/// threadgroup of one row (1, 16, 32 or 48 threads) asks for its one-row variant, not the
/// emulated-simdgroups variant. Two rows of 16 still ask for the latter. Negative controls keep
/// every partial threadgroup on the variant: the row scaled by the width rather than by `n`
/// (cells of two pixels may meet), the latch testing the counter rather than the next value (the
/// body runs at `s == n`, the next pixel's first cell), and the loop entered without the `n != 0`
/// test (at `n == 0` every pixel writes cell 0).
#[test]
fn a_texel_s_samples_in_a_row_need_no_emulated_simdgroups_on_wave64() {
    let compiler = "select i1 %iny, i1 %some, i1 false";
    let mut kernels: Vec<(String, Vec<u8>, crate::metal::PixelFormat, bool)> = Vec::new();
    let Some(fixture) = assemble("wsm_samples") else {
        return;
    };
    for (name, format) in [
        ("wsm_read_f", crate::metal::PixelFormat::RGBA32Float),
        ("wsm_read_u", crate::metal::PixelFormat::R32Uint),
    ] {
        kernels.push((name.into(), fixture.clone(), format, true));
    }
    for (name, guard, mult, tested, rows) in [
        ("cells", compiler, "%n", "%next", true),
        ("cells_and", "and i1 %iny, %some", "%n", "%next", true),
        ("cells_by_width", compiler, "%w", "%next", false),
        ("cells_through_n", compiler, "%n", "%s", false),
        (
            "cells_unguarded",
            "select i1 %iny, i1 true, i1 false",
            "%n",
            "%next",
            false,
        ),
    ] {
        let Some(bc) = assemble_text(name, &cells_kernel(name, guard, mult, tested)) else {
            return;
        };
        kernels.push((name.into(), bc, crate::metal::PixelFormat::R32Uint, rows));
    }
    let dev = wave64();
    let p = dev.profile();
    for (name, bc, format, rows) in kernels {
        let (own, _) = typed_module(&bc, &name, p).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(own.simd.1, Tier::Cluster32, "{name}");
        assert!(
            own.facts
                .contains(&RuntimeFact::LocalSizeConstraint { multiple: 64 }),
            "{name}: {:?}",
            own.facts
        );
        assert_eq!(
            own.facts.contains(&RuntimeFact::RowLockstep),
            rows,
            "{name}"
        );
        let state = samples_state(format);
        let plan = pipeline(&dev, &bc, &name)
            .plan(&dev, &state.shape())
            .unwrap_or_else(|e| panic!("{name}: {e:?}"));
        let want = if rows { one_row_key() } else { emulated_key() };
        for n in [1, 16, 32, 48] {
            assert_eq!(
                plan.dispatch_variant(&threads(n)),
                Ok(want.clone()),
                "{name} at {n}"
            );
        }
        let two_rows = Dispatch::Threadgroups {
            groups: [1, 1, 1],
            threads_per_group: [16, 2, 1],
        };
        assert_eq!(
            plan.dispatch_variant(&two_rows),
            Ok(emulated_key()),
            "{name}"
        );
    }
}

/// What a [`row_store_kernel`] stores.
#[derive(Clone, Copy, PartialEq, Eq)]
enum RowStores {
    /// The three words `3 cell + k` of its cell `cell = y * stride + x`, in `int` arithmetic as
    /// Apple compiles it (`nsw` on every step, the index sign-extended), which two threads of one
    /// simdgroup in different rows may share for all the IR shows.
    Cells,
    /// The same words in `uint` arithmetic, whose steps carry no flags: a step may wrap, and the
    /// order does not prove two threads of one row apart in such an index.
    WrappingCells,
    /// 1 at `x` and 2 at `x + 1`, which two threads of one row share.
    Neighbours,
}

/// A kernel over its position in the grid `(x, y)` that loads `src[x]` and stores to `dst`, of
/// buffers 0 and 1 and the uniform `stride` at buffer 2, what `stores` says; where `guarded`, only
/// a thread whose `x` is under `stride` does, past a branch on that test, as a kernel that bounds
/// its grid is compiled.
fn row_store_kernel(name: &str, stores: RowStores, guarded: bool) -> String {
    let cells = |flags: &str, ext: &str| {
        let mut body = format!(
            "%row = mul{flags} i32 %w, %y
  %cell = add{flags} i32 %row, %x
  %d0 = mul{flags} i32 %cell, 3"
        );
        for k in 0..3 {
            if k > 0 {
                body += &format!("\n  %d{k} = add{flags} i32 %d0, {k}");
            }
            body += &format!(
                "\n  %e{k} = {ext} i32 %d{k} to i64
  %p{k} = getelementptr inbounds i32, ptr addrspace(1) %dst, i64 %e{k}
  store i32 %v, ptr addrspace(1) %p{k}, align 4"
            );
        }
        body
    };
    let body = match stores {
        RowStores::Cells => cells(" nsw", "sext"),
        RowStores::WrappingCells => cells("", "zext"),
        RowStores::Neighbours => "%e0 = zext i32 %x to i64
  %p0 = getelementptr inbounds i32, ptr addrspace(1) %dst, i64 %e0
  store i32 1, ptr addrspace(1) %p0, align 4
  %x1 = add i32 %x, 1
  %e1 = zext i32 %x1 to i64
  %p1 = getelementptr inbounds i32, ptr addrspace(1) %dst, i64 %e1
  store i32 2, ptr addrspace(1) %p1, align 4"
            .to_owned(),
    };
    let (test, body) = match guarded {
        true => (
            "%inside = icmp ult i32 %x, %w
  br i1 %inside, label %body, label %done

body:
  ",
            format!("{body}\n  br label %done\n\ndone:"),
        ),
        false => ("", body),
    };
    format!(
        r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

define void @{name}(ptr addrspace(1) %src, ptr addrspace(1) %dst, ptr addrspace(2) %stride, <2 x i32> %position) {{
  %y = extractelement <2 x i32> %position, i64 1
  %w = load i32, ptr addrspace(2) %stride, align 4
  %x = extractelement <2 x i32> %position, i64 0
  {test}%c = zext i32 %x to i64
  %ps = getelementptr inbounds i32, ptr addrspace(1) %src, i64 %c
  %v = load i32, ptr addrspace(1) %ps, align 4
  {body}
  ret void
}}

!air.kernel = !{{!0}}
!air.version = !{{!20}}
!air.language_version = !{{!21}}

!0 = !{{ptr @{name}, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4, !5, !6}}
!3 = !{{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"src"}}
!4 = !{{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"dst"}}
!5 = !{{i32 2, !"air.buffer", !"air.buffer_size", i32 4, !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"stride"}}
!6 = !{{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"position"}}
!20 = !{{i32 2, i32 8, i32 0}}
!21 = !{{!"Metal", i32 4, i32 0, i32 0}}
"#
    )
}

/// On lavapipe's `Device::from_queries` profile the lockstep order gives the stores of every
/// [`row_store_kernel`] barriers under the default domains, and none under the identity domains
/// (the runtime's), whose kernels take no order: there each kernel's module has no lockstep
/// barrier, states no `RowLockstep` and takes no simd class, it does not observe its threadgroup
/// partition with partial threadgroups, and that variant's pipeline runs its grid in tiles
/// (`GridTiles`).
#[test]
fn the_identity_domains_order_no_lockstep_pair() {
    let desc = crate::pipeline::testing::description("lavapipe").unwrap_or_else(|e| panic!("{e}"));
    let resident = Device::from_queries(desc.clone(), DevicePolicy::default())
        .unwrap_or_else(|e| panic!("{e}"));
    let dev = Device::from_queries(desc, identity()).unwrap_or_else(|e| panic!("{e}"));
    let budget = Budget::standard();
    let lockstep = |plan: &Plan| {
        (plan.census().keys())
            .filter(|name| name.starts_with("barrier.lockstep"))
            .count()
    };
    for (name, stores, guarded) in [
        ("row_cells", RowStores::Cells, false),
        ("row_cells_wrapping", RowStores::WrappingCells, false),
        ("row_neighbours", RowStores::Neighbours, false),
        ("row_cells_guarded", RowStores::Cells, true),
        ("row_cells_wrapping_guarded", RowStores::WrappingCells, true),
        ("row_neighbours_guarded", RowStores::Neighbours, true),
    ] {
        let Some(bc) = assemble_text(name, &row_store_kernel(name, stores, guarded)) else {
            return;
        };
        let (ordered, _) =
            typed_module(&bc, name, resident.profile()).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(lockstep(&ordered) > 0, "{name}: {:?}", ordered.census());
        let (plain, _) =
            typed_module(&bc, name, dev.profile()).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(lockstep(&plain), 0, "{name}: {:?}", plain.census());
        assert!(!plain.facts.contains(&RuntimeFact::RowLockstep), "{name}");
        assert_eq!(plain.simd, (SimdClass::None, Tier::NoSubgroup), "{name}");
        let f = Library::from_air(&bc)
            .ok()
            .and_then(|lib| lib.function(name).ok())
            .map(FunctionDescriptor::new)
            .expect("kernel");
        let entries = crate::pipeline::StageEntries::new();
        let mut partial = LoweringInputs::new();
        partial.partial_threadgroups = true;
        let observes = dev
            .kernel_observes_partition(&f, &partial, &budget, &entries)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(!observes, "{name}: the partial variant");
        let compiled = dev
            .compile_function(&f, &partial, &budget)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(!compiled.observes_threadgroup_partition(), "{name}");
        let pipeline = dev
            .compile_compute_pipeline(
                &ComputePipelineDescriptor::new(StageSource::Compiled(compiled)),
                &budget,
            )
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(
            (pipeline.dependencies.iter()).any(|d| d.fact == RuntimeFact::GridTiles),
            "{name}: {:?}",
            pipeline.dependencies
        );
    }
}
