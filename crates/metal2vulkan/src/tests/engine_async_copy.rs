//! A simdgroup's asynchronous 2D copies and their events as Metal was measured to run them, on the
//! CPU interpreter.
//!
//! The owned fixture `rfs_async_copy` fills 4 KiB of threadgroup memory with `0xee`, has every
//! simdgroup copy the tile its parameters describe from a patterned device buffer (elements of 1, 2
//! and 4 bytes) and wait, and writes the threadgroup memory and each simdgroup's `is_null` of its
//! copy's event and of the null event out. Each kernel is compiled for a profile of each simdgroup
//! tier, passes L3, and runs on the interpreter in both scheduling orders, in threadgroups of 64
//! threads and of 40 (a second simdgroup of 8 lanes); every word must be the reference's
//! ([`reference`], the reading the Metal probes matched). `llvm-as` assembles
//! the fixture; without it the tests skip visibly.

use std::collections::BTreeMap;

use crate::budget::SPIRV_WORDS;
use crate::device::DevicePolicy;
use crate::library::FunctionConstantValues;
use crate::lower::select::Plan;
use crate::pipeline::compile::{compile_bitcode, EntryRef};
use crate::pipeline::interface::PushLoc;
use crate::pipeline::lower_inputs::LoweringInputs;
use crate::pipeline::sym::{PushField, ResourceKind, ResourceSlot, Stage};
use crate::pipeline::testing::{
    engine_dir, link_and_validate, llvm_assemble, meter, profile, typed_module_specialized,
    AssembleError,
};
use crate::spirv::interp::{run, Buffer, Config, Order, Resources, SubgroupSize};
use crate::spirv::ir::TypedModule;

/// Profiles and the subgroup widths their devices run: 32-lane subgroups, the simdgroup itself
/// (MoltenVK on Apple silicon; RADV pinned to 32 lanes), and lavapipe's 8, where a kernel with
/// cross-lane operations emulates its simdgroups.
const MOLTENVK: (&str, &[SubgroupSize]) = ("moltenvk-m3", &[SubgroupSize::W32]);
const RADV: (&str, &[SubgroupSize]) = ("amd-rdna2-apu-radv", &[SubgroupSize::W32]);
const LAVAPIPE: (&str, &[SubgroupSize]) = ("lavapipe", &[SubgroupSize::W8]);

/// The fixture's threadgroup memory.
const TG_BYTES: usize = 4096;

/// The source buffer's bytes.
const SRC_BYTES: usize = 4096;

#[test]
fn copies_are_metals_on_moltenvk() {
    copies_are_metals(MOLTENVK);
}

#[test]
fn copies_are_metals_on_radv() {
    copies_are_metals(RADV);
}

#[test]
fn copies_are_metals_on_lavapipe() {
    copies_are_metals(LAVAPIPE);
}

fn bitcode() -> Option<Vec<u8>> {
    match llvm_assemble(&engine_dir().join("rfs_async_copy.ll")) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("rfs_async_copy: {why}"),
    }
}

struct Kernel {
    what: String,
    plan: Plan,
    module: TypedModule,
}

/// Kernel `entry` of `bc` for profile `stem`, checked through L3: its plan and typed module.
fn kernel(bc: &[u8], entry: &str, stem: &str) -> Kernel {
    let p = profile(stem, DevicePolicy::RESIDENT_TABLES).expect("profile");
    let consts = FunctionConstantValues::defaults();
    let what = format!("{entry} on {stem}");
    let a = compile_bitcode(
        bc,
        EntryRef::Name(entry),
        &consts,
        &p,
        &LoweringInputs::new(),
        &mut meter(),
        SPIRV_WORDS,
    )
    .unwrap_or_else(|e| panic!("{what}: {e}"));
    link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: {e}"));
    let (plan, module) =
        typed_module_specialized(bc, entry, &p, &consts).unwrap_or_else(|e| panic!("{what}: {e}"));
    Kernel { what, plan, module }
}

impl Kernel {
    /// One threadgroup of `threads` threads with buffers `init`, at each subgroup width in
    /// `widths` and in both scheduling orders, which must agree: the final buffers.
    fn run(&self, widths: &[SubgroupSize], threads: u32, init: &[Vec<u8>]) -> Vec<Vec<u8>> {
        let mut outs: Vec<Vec<Vec<u8>>> = Vec::new();
        for &w in widths {
            for order in [Order::Forward, Order::Reverse] {
                let cfg = Config {
                    subgroup: w,
                    order,
                    ..Config::default()
                };
                let out = run(&self.module, &cfg, resources(&self.plan, threads, init))
                    .unwrap_or_else(|e| panic!("{} {w:?} {threads}: {e:?}", self.what));
                outs.push(out.buffers);
            }
        }
        for o in &outs[1..] {
            assert!(o == &outs[0], "{}: the runs disagree", self.what);
        }
        outs.swap_remove(0)
    }
}

fn resources(plan: &Plan, threads: u32, init: &[Vec<u8>]) -> Resources {
    let mut spec = BTreeMap::new();
    spec.extend(crate::pipeline::threadgroup_constants([threads, 1, 1]));
    let push = vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)];
    for f in &plan.push.fields {
        assert!(
            matches!(
                (f.field, f.loc),
                (
                    PushField::SlotBindOffset(_) | PushField::DispatchRegion,
                    PushLoc::Push { .. }
                )
            ),
            "unexpected push field {f:?}"
        );
    }
    let buffers = init
        .iter()
        .zip(0u32..)
        .map(|(b, index)| Buffer {
            slot: Some(ResourceSlot {
                stage: Stage::Kernel,
                kind: ResourceKind::Buffer,
                index,
            }),
            address: None,
            bytes: b.clone(),
        })
        .collect();
    Resources {
        buffers,
        push,
        spec,
    }
}

/// The source's byte `k`.
fn src_byte(k: usize) -> u8 {
    ((k * 131 + 7) % 255 + 1) as u8
}

/// A copy's parameters (the fixture's `p`): `[dst_epr, dst_stride, dst_tile.x, dst_tile.y,
/// src_epr, src_stride, src_tile.x, src_tile.y, offset.x, offset.y, dst byte offset, src byte
/// offset, the byte offset between the simdgroups' destinations]`.
type Params = [i64; 13];

/// Configurations: a plain tile, a smaller source, positive, negative and mixed offsets, a
/// smaller destination, a destination pitch, element strides, a tile wider than the simdgroup,
/// an empty destination, an offset past the source, and a source extent below zero.
const CONFIGS: [(&str, Params); 12] = [
    ("plain", [8, 1, 8, 4, 16, 1, 8, 4, 0, 0, 0, 0, 1024]),
    ("src-smaller", [8, 1, 8, 4, 16, 1, 5, 3, 0, 0, 0, 0, 1024]),
    ("offset+", [8, 1, 8, 4, 16, 1, 8, 4, 1, 2, 0, 0, 1024]),
    ("offset-", [8, 1, 8, 4, 16, 1, 8, 4, -1, -2, 0, 1024, 1024]),
    ("dst-smaller", [8, 1, 4, 2, 16, 1, 8, 4, 0, 0, 0, 0, 1024]),
    ("dst-pitch", [12, 1, 8, 4, 16, 1, 8, 4, 0, 0, 16, 0, 1024]),
    ("stride2", [16, 2, 4, 4, 16, 2, 4, 4, 0, 0, 0, 0, 1024]),
    ("33x3", [33, 1, 33, 3, 40, 1, 33, 3, 0, 0, 0, 0, 1024]),
    ("empty-dst", [8, 1, 0, 4, 16, 1, 8, 4, 0, 0, 0, 0, 1024]),
    ("offset-past", [8, 1, 8, 4, 16, 1, 8, 4, 9, 0, 0, 0, 1024]),
    (
        "offset-mixed",
        [8, 1, 8, 4, 16, 1, 6, 3, 2, -1, 0, 1024, 1024],
    ),
    (
        "src-negative-extent",
        [8, 1, 8, 4, 16, 1, -2, 4, 0, 0, 0, 0, 1024],
    ),
];

/// The reference: the threadgroup memory after every simdgroup's copy of `size`-byte elements,
/// then per simdgroup its event's `is_null` (0) and the null event's (1). Destination element
/// `(x, y)` of the tile takes source element `(x + offset.x, y + offset.y)` when each coordinate is
/// below the source tile's extent compared unsigned, else zero.
fn reference(p: &Params, size: usize, simdgroups: usize) -> Vec<u32> {
    let mut m = vec![0xeeu8; TG_BYTES];
    for s in 0..simdgroups {
        let dst = p[10] + s as i64 * p[12];
        for y in 0..p[3].max(0) {
            for x in 0..p[2].max(0) {
                let (sx, sy) = (x + p[8], y + p[9]);
                let inside = (sx as u64) < (p[6] as u64) && (sy as u64) < (p[7] as u64);
                let d = (dst + (y * p[0] + x * p[1]) * size as i64) as usize;
                let from = (p[11] + (sy * p[4] + sx * p[5]) * size as i64) as usize;
                for b in 0..size {
                    m[d + b] = if inside { src_byte(from + b) } else { 0 };
                }
            }
        }
    }
    let mut words: Vec<u32> = m
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    for _ in 0..simdgroups {
        words.extend([0, 1]);
    }
    words
}

/// Every configuration's threadgroup memory and events are Metal's on `profile`, for each element
/// size, in whole and partial simdgroups.
fn copies_are_metals((stem, widths): (&str, &[SubgroupSize])) {
    let Some(bc) = bitcode() else {
        return;
    };
    let src: Vec<u8> = (0..SRC_BYTES).map(src_byte).collect();
    for size in [1usize, 2, 4] {
        let k = kernel(&bc, &format!("rfs_async_copy_{size}"), stem);
        for (name, p) in &CONFIGS {
            for threads in [64u32, 40] {
                let simdgroups = threads.div_ceil(32) as usize;
                let words = TG_BYTES / 4 + 2 * simdgroups;
                let out = vec![0xabu8; 4 * words];
                let params: Vec<u8> = p.iter().flat_map(|v| v.to_le_bytes()).collect();
                let got = k.run(widths, threads, &[src.clone(), out, params]);
                let got: Vec<u32> = got[1]
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                let want = reference(p, size, simdgroups);
                if let Some(i) = (0..words).find(|&i| got[i] != want[i]) {
                    panic!(
                        "{} {name} at {threads} threads: word {i} is {:#010x}, Metal's {:#010x}",
                        k.what, got[i], want[i]
                    );
                }
            }
        }
    }
}
