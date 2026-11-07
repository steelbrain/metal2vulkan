//! Simdgroups emulated through workgroup memory.
//!
//! A kernel whose device cannot run a 32-lane simdgroup on subgroups (`lavapipe` cannot pin 32
//! lanes; `vk12-floor` has no subgroup operations) is planned in the `Emulated` tier: simdgroup
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
//! traps. No GPU is used.
//!
//! `llvm-as` assembles the fixtures; a missing tool skips visibly.

use std::collections::BTreeMap;

use crate::bind::{BindState, BoundBuffer, Dispatch, StageTable};
use crate::budget::Budget;
use crate::cache::VariantKey;
use crate::device::profile::DeviceProfile;
use crate::device::subgroup::{SubgroupRequirement, Tier};
use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::error::{BindError, Error, Feature};
use crate::library::Library;
use crate::lower::dispatch::LocalSizeForm;
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
const PROFILES: [&str; 2] = ["lavapipe", "vk12-floor"];

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

/// Every word of buffer 0 after one workgroup of local size `local` ran `m` (planned as `plan`)
/// with `lanes`-wide subgroups, identical in both scheduling orders. Buffer 1, when the kernel
/// binds it, holds `constant`.
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
    let push = vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)];
    for f in &plan.push.fields {
        assert!(matches!(f.loc, PushLoc::Push { .. }), "{f:?}");
        // Zero residuals: each buffer is bound at its start.
        assert!(
            matches!(f.field, PushField::SlotBindOffset(_)),
            "unexpected push field {f:?}"
        );
    }
    let slot = |index| {
        Some(ResourceSlot {
            stage: Stage::Kernel,
            kind: ResourceKind::Buffer,
            index,
        })
    };
    let mut buffers = vec![Buffer {
        slot: slot(0),
        address: None,
        bytes: vec![0; n as usize * 4],
    }];
    if let Some(c) = constant {
        buffers.push(Buffer {
            slot: slot(1),
            address: None,
            bytes: c.to_le_bytes().to_vec(),
        });
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
    let p = profile(stem, DevicePolicy::default()).expect("profile");
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
/// 683 words over the subgroup form did not change, so the hold moved to 2.55x, still below the
/// flag form's 112 extra words (about 2.75x against 453). The subgroup form is `moltenvk-m3`'s
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
    let tight = (&float_min, "simd_float_prefix_k", 2.55);
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

/// `moltenvk-m3`'s profile (whole subgroups) without its driver's deviation `MetalCanonicalNans`,
/// as [`PROFILES`] lack it: a float result's NaN select is then the same on both sides of a ratio.
fn whole_subgroups_without_the_nan_fact() -> DeviceProfile {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles/moltenvk-m3.json");
    let text = std::fs::read_to_string(&path).expect("profile");
    let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    let nans = crate::error::DriverDeviation::MetalCanonicalNans;
    let with = Device::new(desc.clone(), DevicePolicy::default()).expect("device");
    assert_eq!(
        with.profile().fields().nan_encoding,
        crate::device::profile::NanEncoding::MetalCanonical,
        "moltenvk-m3's driver has the deviation"
    );
    Device::with_deviation(desc, DevicePolicy::default(), nans, false)
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
fn simd_divergent_runs_predicated_on_vk12_floor() {
    simd_divergent_runs_predicated("vk12-floor");
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

/// The scratch takes 128 bytes per simdgroup and word beside the kernel's threadgroup memory. A
/// kernel whose arena leaves less than one simdgroup's scratch is refused at compile time;
/// otherwise the pipeline reports the threads whose scratch fits as its maximum and its own
/// threadgroup memory as its static length, as Metal reports it, and the planner refuses a larger
/// threadgroup as `ThreadgroupMemorySize`.
#[test]
fn the_scratch_must_fit_beside_the_threadgroup_memory() {
    // vk12-floor: 16 KiB. An arena of exactly 16 KiB leaves nothing for the scratch.
    let Some(full) = arena_kernel(4096) else {
        return;
    };
    let floor = profile("vk12-floor", DevicePolicy::default()).expect("profile");
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
/// threadgroup barrier keeps the profile's reconvergence (`vk12-floor` has no maximal
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
    let floor = profile("vk12-floor", DevicePolicy::default()).expect("profile");
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
    let dev = Device::new(desc, DevicePolicy::default()).expect("device");
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
    let dev = Device::new(desc, DevicePolicy::default()).expect("device");
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
/// pipeline runs alone, is a `LocalSize` refusal up front, as the variant's plan refuses it; an
/// interior of `[1, 1, 64]` asks for the variant, whose plan runs both regions.
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
    for r in [
        plan.dispatch_variant(&mixed(96)).map(drop),
        plan.instantiate_dispatch(&state, &mixed(96)).map(drop),
        vplan.instantiate_dispatch(&state, &mixed(96)).map(drop),
    ] {
        match r {
            Err(BindError::Unsupported(u)) => assert_eq!(
                u.feature,
                Feature::LocalSize {
                    axis: Some(Axis::Z)
                }
            ),
            other => panic!("{other:?}"),
        }
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
    Device::new(desc, DevicePolicy::default()).expect("device")
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
