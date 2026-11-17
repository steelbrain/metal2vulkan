//! The identity domains (`PointerDomain::Identity` with `HandleDomain::Identity`) in the lowering.
//!
//! - A kernel over buffers alone (`engine/a1_bool_from_float`) compiles with each slot reached
//!   through its own device address, pushed (`SlotGuestBase`), with no root table and no
//!   descriptor. It passes L3, and on the CPU interpreter, with its buffers above 4 GiB and bound
//!   at offsets that are not 0, it computes the program's answer and writes nothing else.
//! - A kernel whose slot addresses do not fit the 128-byte window (`many`, generated: 20 read
//!   buffers and an output) pushes its parameter block's address first
//!   (`ArgumentTableAddress`), the addresses that fit after it, and reads the rest from the block
//!   at their placed offsets, with no descriptor; it passes L3 and computes its answer on the CPU
//!   interpreter.
//! - A kernel that reads a texture (`engine/a2_dynamic_texel_offset`) is still refused as
//!   `Unsupported{IdentityDomains}`, not implemented, where the default domains compile it.
//!
//! `llvm-as` assembles the kernels; without it the tests skip visibly.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::device::profile::{HandleDomain, PointerDomain};
use crate::device::DevicePolicy;
use crate::error::{Error, Feature, Needs};
use crate::lower::dispatch::region_bytes;
use crate::pipeline::interface::PushLoc;
use crate::pipeline::sym::{Axis, PushField, SlotClass, SlotRef, SpecKey, Stage};
use crate::pipeline::testing::{
    compile, compile_kept, engine_dir, link_and_validate, llvm_assemble, profile, AssembleError,
};
use crate::spirv::interp::{run, Buffer, Config, Resources};

/// Profiles with `bufferDeviceAddress` and `shaderInt64`.
const PROFILES: [&str; 2] = ["lavapipe", "moltenvk-m3"];

fn identity() -> DevicePolicy {
    DevicePolicy {
        pointer_domain: PointerDomain::Identity,
        handle_domain: HandleDomain::Identity,
        ..DevicePolicy::default()
    }
}

fn assemble(name: &str) -> Option<Vec<u8>> {
    assembled(
        name,
        llvm_assemble(&engine_dir().join(format!("{name}.ll"))),
    )
}

fn assembled(name: &str, r: Result<Vec<u8>, AssembleError>) -> Option<Vec<u8>> {
    match r {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(e) => panic!("{name}: {e:?}"),
    }
}

/// `text` assembled in a scratch directory of its own, removed once `llvm-as` is done.
fn assemble_text(name: &str, text: &str) -> Option<Vec<u8>> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "m2v-identity-{}-{}-{name}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).expect("scratch");
    let ll = dir.join("k.ll");
    std::fs::write(&ll, text).expect("write");
    let r = llvm_assemble(&ll);
    let _ = std::fs::remove_dir_all(&dir);
    assembled(name, r)
}

/// The spec map of a dispatch of `group` threads a threadgroup, in one dimension.
fn local_size(group: u32) -> BTreeMap<SpecKey, u64> {
    BTreeMap::from([
        (SpecKey::LocalSize(Axis::X), u64::from(group)),
        (SpecKey::LocalSize(Axis::Y), 1),
        (SpecKey::LocalSize(Axis::Z), 1),
        (SpecKey::FlatLocalSize, u64::from(group)),
    ])
}

fn slot(index: u32) -> SlotRef {
    SlotRef {
        stage: Stage::Kernel,
        class: SlotClass::Buffer,
        index,
    }
}

/// `float` patterns none of which is a NaN or a denormal (the fixture compiles with fast math).
const FLOATS: [f32; 6] = [0.0, -0.0, 0.5, 2.0, 1.0, -3.0];
/// The same values' `half` bits.
const HALVES: [u16; 6] = [0x0000, 0x8000, 0x3800, 0x4000, 0x3c00, 0xc200];

/// Thread `i`'s input lanes: indices into [`FLOATS`] and [`HALVES`].
fn lanes(i: usize) -> ([usize; 4], [usize; 4]) {
    (
        std::array::from_fn(|j| (i + j) % FLOATS.len()),
        std::array::from_fn(|j| (3 * i + j) % HALVES.len()),
    )
}

/// What `a1_bool_from_float` writes for thread `i`: each lane's `x != 0`, the `float` lanes in
/// bits 0-3 and 4-7, the `half` lanes in bits 8-11 and 12-15.
fn want(i: usize) -> u32 {
    let (f, h) = lanes(i);
    let mut r = 0u32;
    for (j, (&fi, &hi)) in f.iter().zip(&h).enumerate() {
        let fb = u32::from(FLOATS[fi] != 0.0);
        let hb = u32::from(HALVES[hi] & 0x7fff != 0);
        r |= fb << j | fb << (4 + j) | hb << (8 + j) | hb << (12 + j);
    }
    r
}

/// A kernel over buffers alone compiles under the identity domains to a module that reads each
/// slot's device address from its own push field and nothing else, and computes the program's
/// answer there.
#[test]
fn a_buffer_kernel_reads_each_slot_at_its_pushed_device_address() {
    let Some(bc) = assemble("a1_bool_from_float") else {
        return;
    };
    const GROUP: u32 = 8;
    const GROUPS: u32 = 4;
    let threads = (GROUP * GROUPS) as usize;
    // (the buffer's device address, the bind offset): above 4 GiB, so a dropped high word shows.
    let at = [
        (0x1_0000_0000u64, 32u64),
        (0x2_0000_1000, 16),
        (0x3_0000_2000, 4),
    ];
    let mut bits = vec![0u8; at[0].1 as usize];
    let mut halves = vec![0u8; at[1].1 as usize];
    for i in 0..threads {
        let (f, h) = lanes(i);
        for (&fi, &hi) in f.iter().zip(&h) {
            bits.extend_from_slice(&FLOATS[fi].to_bits().to_le_bytes());
            halves.extend_from_slice(&HALVES[hi].to_le_bytes());
        }
    }
    // One spare word past the output's end, which the kernel must not write.
    let out = vec![0u8; at[2].1 as usize + 4 * threads + 4];
    for stem in PROFILES {
        let p = profile(stem, identity()).unwrap_or_else(|e| panic!("{e}"));
        let (a, plan, module) =
            compile_kept(&bc, "a1_bool_from_float", &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert!(a.interface.descriptors.is_empty(), "{stem}");
        assert!(!a.interface.needs_pointer_translation, "{stem}");
        assert_eq!(plan.push.argument_table_size, 0, "{stem}");
        link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
        let mut push = vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)];
        let mut read = Vec::new();
        // The grid builtin's region, as the identity domains fill it: base 0, the threadgroup
        // size, and the grid.
        let region = region_bytes([0; 3], [GROUP * GROUPS, 1, 1], [GROUP, 1, 1]).expect("packs");
        for f in &plan.push.fields {
            let o = match f.loc {
                PushLoc::Push { offset } => offset as usize,
                PushLoc::ArgumentTable { .. } => panic!("{stem}: {f:?} spilled"),
            };
            match f.field {
                PushField::SlotGuestBase(s) => {
                    let (base, bind) = at[s.index as usize];
                    push[o..o + 8].copy_from_slice(&(base + bind).to_le_bytes());
                    read.push(s);
                }
                PushField::DispatchRegion => push[o..o + 32].copy_from_slice(&region),
                _ => panic!("{stem}: unexpected push field {f:?}"),
            }
        }
        read.sort();
        assert_eq!(read, [slot(0), slot(1), slot(2)], "{stem}");
        let buffer = |(address, _): (u64, u64), bytes: &Vec<u8>| Buffer {
            slot: None,
            address: Some(address),
            bytes: bytes.clone(),
        };
        let mut expected = vec![0u8; at[2].1 as usize];
        for i in 0..threads {
            expected.extend_from_slice(&want(i).to_le_bytes());
        }
        expected.extend_from_slice(&[0; 4]);
        // The region read (`MultiRegion` 1) and not (0) give the same grid.
        for multi in [0, 1] {
            let mut spec = local_size(GROUP);
            spec.insert(SpecKey::MultiRegion, multi);
            let res = Resources {
                buffers: vec![
                    buffer(at[0], &bits),
                    buffer(at[1], &halves),
                    buffer(at[2], &out),
                ],
                push: push.clone(),
                spec,
            };
            let cfg = Config {
                workgroups: [GROUPS, 1, 1],
                ..Config::default()
            };
            let got = run(&module, &cfg, res).unwrap_or_else(|e| panic!("{stem}: {e}"));
            assert_eq!(got.buffers[2], expected, "{stem} {multi}: out");
            assert_eq!(got.buffers[0], bits, "{stem} {multi}: bits");
            assert_eq!(got.buffers[1], halves, "{stem} {multi}: halves");
        }
    }
}

/// A kernel that reads a texture is not lowered under the identity domains yet.
#[test]
fn a_texture_kernel_is_refused_under_the_identity_domains() {
    let Some(bc) = assemble("a2_dynamic_texel_offset") else {
        return;
    };
    for stem in PROFILES {
        let default = profile(stem, DevicePolicy::default()).unwrap_or_else(|e| panic!("{e}"));
        compile(&bc, "a2_dynamic_texel_offset", &default)
            .unwrap_or_else(|e| panic!("{stem}: default domains: {e}"));
        let p = profile(stem, identity()).unwrap_or_else(|e| panic!("{e}"));
        match compile(&bc, "a2_dynamic_texel_offset", &p) {
            Err(Error::Unsupported(u)) => {
                assert_eq!(
                    (u.feature, u.needs),
                    (Feature::IdentityDomains, Needs::NotImplemented),
                    "{stem}"
                );
            }
            other => panic!("{stem}: {other:?}"),
        }
    }
}

/// The read buffers of [`many`]: with its output, 21 slot addresses of 8 bytes, past the window.
const READS: u32 = 20;

/// A kernel reading one `uint` from each of [`READS`] buffers and storing their ordered fold
/// (`s · 3 + v`, so two swapped addresses change it) to the next buffer.
fn many() -> String {
    let mut params = Vec::new();
    let mut body = String::new();
    let mut meta = Vec::new();
    for k in 0..=READS {
        params.push(format!(
            "ptr addrspace(1) noundef \"air-buffer-no-alias\" %{k}"
        ));
        let (access, name) = match k == READS {
            true => ("air.read_write", "out".to_string()),
            false => ("air.read", format!("b{k}")),
        };
        meta.push(format!(
            "!{} = !{{i32 {k}, !\"air.buffer\", !\"air.location_index\", i32 {k}, i32 1, !\"{access}\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"{name}\"}}",
            100 + k
        ));
    }
    body.push_str("  %s0 = load i32, ptr addrspace(1) %0, align 4\n");
    for k in 1..READS {
        body.push_str(&format!(
            "  %v{k} = load i32, ptr addrspace(1) %{k}, align 4\n  %m{k} = mul i32 %s{}, 3\n  %s{k} = add i32 %m{k}, %v{k}\n",
            k - 1
        ));
    }
    let last = READS - 1;
    let args: Vec<String> = (0..=READS).map(|k| format!("!{}", 100 + k)).collect();
    format!(
        "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\n\
target triple = \"air64_v28-apple-macosx26.0.0\"\n\n\
define void @many({}) {{\nentry:\n{body}  store i32 %s{last}, ptr addrspace(1) %{READS}, align 4\n  ret void\n}}\n\n\
!air.kernel = !{{!9}}\n!air.version = !{{!17}}\n!air.language_version = !{{!18}}\n\
!9 = !{{ptr @many, !10, !11}}\n!10 = !{{}}\n!11 = !{{{}}}\n{}\n\
!17 = !{{i32 2, i32 8, i32 0}}\n!18 = !{{!\"Metal\", i32 4, i32 0, i32 0}}\n",
        params.join(", "),
        args.join(", "),
        meta.join("\n"),
    )
}

/// A kernel whose slot addresses do not fit the push window reads the rest from its parameter
/// block, whose address the window holds first, and computes its answer there.
#[test]
fn slot_addresses_past_the_window_are_read_from_the_pushed_parameter_block() {
    let Some(bc) = assemble_text("many", &many()) else {
        return;
    };
    // Buffer `k` at `BASE + k · STRIDE`, holding one word; the block at `BLOCK`.
    const BASE: u64 = 0x5_0000_0000;
    const STRIDE: u64 = 0x100;
    const BLOCK: u64 = 0x7_0000_0000;
    let value = |k: u32| 0x0101_0101u32.wrapping_mul(k + 1) ^ 0x5a5a;
    let want = (1..READS).fold(value(0), |s, k| s.wrapping_mul(3).wrapping_add(value(k)));
    for stem in PROFILES {
        let p = profile(stem, identity()).unwrap_or_else(|e| panic!("{e}"));
        let (a, plan, module) =
            compile_kept(&bc, "many", &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert!(a.interface.descriptors.is_empty(), "{stem}");
        assert!(!a.interface.needs_pointer_translation, "{stem}");
        link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
        let first = plan.push.fields.first().map(|f| (f.field, f.loc));
        assert_eq!(
            first,
            Some((PushField::ArgumentTableAddress, PushLoc::Push { offset: 0 })),
            "{stem}"
        );
        let mut push = vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)];
        let mut block = vec![0u8; plan.push.argument_table_size as usize];
        let (mut pushed, mut spilled) = (Vec::new(), Vec::new());
        for f in &plan.push.fields {
            let (bytes, into, offset) = match (f.field, f.loc) {
                (PushField::ArgumentTableAddress, PushLoc::Push { offset }) => {
                    (BLOCK, &mut push, offset)
                }
                (PushField::SlotGuestBase(s), PushLoc::Push { offset }) => {
                    pushed.push(s);
                    (BASE + u64::from(s.index) * STRIDE, &mut push, offset)
                }
                (PushField::SlotGuestBase(s), PushLoc::ArgumentTable { offset }) => {
                    spilled.push(s);
                    (BASE + u64::from(s.index) * STRIDE, &mut block, offset)
                }
                other => panic!("{stem}: unexpected push field {other:?}"),
            };
            let o = offset as usize;
            into[o..o + 8].copy_from_slice(&bytes.to_le_bytes());
        }
        // The window's 120 bytes past the block address hold 15 slot addresses.
        assert_eq!(pushed.len(), 15, "{stem}");
        assert_eq!(spilled.len(), (READS + 1) as usize - 15, "{stem}");
        assert_eq!(plan.push.argument_table_size, 48, "{stem}");
        let mut buffers: Vec<Buffer> = (0..=READS)
            .map(|k| Buffer {
                slot: None,
                address: Some(BASE + u64::from(k) * STRIDE),
                bytes: match k == READS {
                    true => vec![0; 4],
                    false => value(k).to_le_bytes().to_vec(),
                },
            })
            .collect();
        buffers.push(Buffer {
            slot: None,
            address: Some(BLOCK),
            bytes: block,
        });
        let res = Resources {
            buffers,
            push,
            spec: local_size(1),
        };
        let got = run(&module, &Config::default(), res).unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert_eq!(
            got.buffers[READS as usize],
            want.to_le_bytes(),
            "{stem}: out"
        );
    }
}
