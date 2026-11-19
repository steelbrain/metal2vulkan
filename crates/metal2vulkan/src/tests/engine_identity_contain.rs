//! Containment under the identity domains (`lower::access::emit`'s `prepare`).
//!
//! - A kernel (`contain`, generated) that loads `data[t + k]`, stores it to `out[t + k]` and adds 1
//!   to `out[t]` atomically, `k` the word at offset 0 of a `constant` buffer, compiles on every
//!   committed profile `Device::from_queries` derives under the identity domains with
//!   `SlotRemaining` for its two device slots, none for the `constant` slot read at a constant
//!   offset, and one `ZeroAddress`. In its module every device-address load at a run-time offset
//!   reads through an address selected in the shader (one select, toward the zero range), the
//!   constant-offset load through an address no select makes, and every device-address store and
//!   atomic sits in the true arm of a conditional branch.
//! - On the CPU interpreter, with each slot's pushed length: in bounds it moves the program's
//!   words; past `data`'s length its loads read zero; past `out`'s pushed length (shorter than the
//!   buffer the interpreter holds) its stores land nowhere; a slot pushed shorter than one access
//!   and a nil slot (length 0) read zero; and the zero range stays zeros. No access traps.
//!
//! `llvm-as` assembles the kernel; without it the tests skip visibly.

use std::collections::{BTreeMap, BTreeSet};

use super::engine_word_phases::{identity_profiles, slot_word};
use crate::lower::dispatch::region_bytes;
use crate::pipeline::interface::PushLoc;
use crate::pipeline::sym::{Axis, PushField, SlotClass, SlotRef, SpecKey, Stage};
use crate::pipeline::testing::{compile_kept, link_and_validate, llvm_assemble, AssembleError};
use crate::spirv::interp::{run, Buffer, Config, Resources};

/// Threads of the dispatch, one threadgroup.
const THREADS: u32 = 4;
/// The slots' addresses: `data`, `out`, `k`, and the zero range.
const DATA: u64 = 0x1_0000_0000;
const OUT: u64 = 0x2_0000_1000;
const K: u64 = 0x3_0000_2000;
const ZERO: u64 = 0x7_0000_0000;
/// Where a nil slot's `data` lies: the runtime's nil buffer.
const NIL: u64 = 0x5_0000_0000;
/// Words of `data` and of the interpreter's `out`.
const DATA_WORDS: usize = 8;
const OUT_WORDS: usize = 16;
/// What `out` holds before the dispatch.
const SENTINEL: u32 = 0xAAAA_0000;

fn contain_kernel() -> String {
    "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\n\
target triple = \"air64_v28-apple-macosx26.0.0\"\n\n\
define void @contain(ptr addrspace(1) noundef \"air-buffer-no-alias\" %0, ptr addrspace(1) noundef \"air-buffer-no-alias\" %1, ptr addrspace(2) noundef %2, <3 x i32> noundef %3) {\n\
entry:\n\
  %t = extractelement <3 x i32> %3, i64 0\n\
  %k = load i32, ptr addrspace(2) %2, align 4\n\
  %i = add i32 %t, %k\n\
  %ie = zext i32 %i to i64\n\
  %p = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %ie\n\
  %v = load i32, ptr addrspace(1) %p, align 4\n\
  %q = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %ie\n\
  store i32 %v, ptr addrspace(1) %q, align 4\n\
  %te = zext i32 %t to i64\n\
  %r = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %te\n\
  %o = atomicrmw add ptr addrspace(1) %r, i32 1 monotonic, align 4\n\
  ret void\n\
}\n\n\
!air.kernel = !{!9}\n!air.version = !{!17}\n!air.language_version = !{!18}\n\
!9 = !{ptr @contain, !10, !11}\n!10 = !{}\n!11 = !{!12, !13, !14, !15}\n\
!12 = !{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"data\"}\n\
!13 = !{i32 1, !\"air.buffer\", !\"air.location_index\", i32 1, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"out\"}\n\
!14 = !{i32 2, !\"air.buffer\", !\"air.buffer_size\", i32 4, !\"air.location_index\", i32 2, i32 1, !\"air.read\", !\"air.address_space\", i32 2, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"k\"}\n\
!15 = !{i32 3, !\"air.thread_position_in_grid\", !\"air.arg_type_name\", !\"uint3\", !\"air.arg_name\", !\"tid\"}\n\
!17 = !{i32 2, i32 8, i32 0}\n!18 = !{!\"Metal\", i32 4, i32 0, i32 0}\n"
        .to_owned()
}

/// The kernel assembled in a scratch directory of its own, removed once `llvm-as` is done.
fn assemble() -> Option<Vec<u8>> {
    let dir = std::env::temp_dir().join(format!("m2v-contain-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("scratch");
    let ll = dir.join("k.ll");
    std::fs::write(&ll, contain_kernel()).expect("write");
    let r = llvm_assemble(&ll);
    let _ = std::fs::remove_dir_all(&dir);
    match r {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(e) => panic!("contain: {e:?}"),
    }
}

fn slot(index: u32) -> SlotRef {
    SlotRef {
        stage: Stage::Kernel,
        class: SlotClass::Buffer,
        index,
    }
}

/// The device-address accesses of serialized module `words`: the loads through an address an
/// `OpSelect` makes and through one it does not, and the stores and atomics in the true arm of
/// an `OpBranchConditional` and elsewhere.
fn accesses(words: &[u32]) -> ([usize; 2], [usize; 2]) {
    const TYPE_POINTER: u32 = 32;
    const PHYSICAL_STORAGE_BUFFER: u32 = 5349;
    const CONVERT_U_TO_PTR: u32 = 120;
    const ACCESS_CHAIN: u32 = 65;
    const SELECT: u32 = 169;
    const LOAD: u32 = 61;
    const STORE: u32 = 62;
    const ATOMIC_IADD: u32 = 234;
    const LABEL: u32 = 248;
    const BRANCH_CONDITIONAL: u32 = 250;
    let insts = || {
        let mut at = 5;
        std::iter::from_fn(move || {
            let w = *words.get(at)?;
            let len = ((w >> 16) as usize).max(1);
            let inst = &words[at..(at + len).min(words.len())];
            at += len;
            Some((w & 0xFFFF, inst))
        })
    };
    let pointers: BTreeSet<u32> = insts()
        .filter(|(op, i)| *op == TYPE_POINTER && i.get(2) == Some(&PHYSICAL_STORAGE_BUFFER))
        .map(|(_, i)| i[1])
        .collect();
    let selects: BTreeSet<u32> = insts()
        .filter(|(op, _)| *op == SELECT)
        .map(|(_, i)| i[2])
        .collect();
    let guarded: BTreeSet<u32> = insts()
        .filter(|(op, _)| *op == BRANCH_CONDITIONAL)
        .map(|(_, i)| i[2])
        .collect();
    // Each device place, whether its address is a select's.
    let mut bases = BTreeMap::new();
    let mut places = BTreeMap::new();
    let (mut loads, mut writes) = ([0; 2], [0; 2]);
    let mut block = 0;
    for (op, i) in insts() {
        match op {
            LABEL => block = i[1],
            CONVERT_U_TO_PTR if pointers.contains(&i[1]) => {
                bases.insert(i[2], selects.contains(&i[3]));
            }
            ACCESS_CHAIN => {
                if let Some(&selected) = bases.get(&i[3]) {
                    places.insert(i[2], selected);
                }
            }
            LOAD => {
                if let Some(&selected) = places.get(&i[3]).or(bases.get(&i[3])) {
                    loads[usize::from(!selected)] += 1;
                }
            }
            STORE if places.contains_key(&i[1]) || bases.contains_key(&i[1]) => {
                writes[usize::from(!guarded.contains(&block))] += 1;
            }
            ATOMIC_IADD if places.contains_key(&i[3]) || bases.contains_key(&i[3]) => {
                writes[usize::from(!guarded.contains(&block))] += 1;
            }
            _ => {}
        }
    }
    (loads, writes)
}

/// Every device-address access at a run-time offset carries the check on every committed
/// profile under the identity domains: its slot reads its `SlotRemaining`, a load reads through a
/// selected address, a store and an atomic sit in the check's true arm; the constant-offset load
/// takes none of it.
#[test]
fn every_run_time_offset_access_of_a_slot_carries_the_check() {
    let Some(bc) = assemble() else {
        return;
    };
    let profiles = identity_profiles();
    assert!(!profiles.is_empty());
    for (stem, p) in &profiles {
        let (a, plan, _) =
            compile_kept(&bc, "contain", p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        link_and_validate(&a, p).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
        let fields: BTreeSet<PushField> = plan.push.fields.iter().map(|f| f.field).collect();
        for s in [0, 1] {
            assert!(
                fields.contains(&PushField::SlotRemaining(slot(s))),
                "{stem}: slot {s} reads no length"
            );
        }
        assert!(
            !fields.contains(&PushField::SlotRemaining(slot(2))),
            "{stem}: the constant-offset slot reads a length"
        );
        assert!(
            fields.contains(&PushField::ZeroAddress),
            "{stem}: no zero range"
        );
        // The run-time load is selected, the constant-offset one is not; the store and the
        // atomic are guarded.
        assert_eq!(accesses(&a.module.words), ([1, 1], [2, 0]), "{stem}");
    }
}

/// One dispatch of the kernel at `k`, with `data`'s and `out`'s pushed lengths `lengths` and
/// `data` at `data_at`; `out` and the zero range after it.
fn dispatch(k: u32, lengths: [u32; 2], data_at: u64) -> (Vec<u32>, Vec<u8>) {
    let Some(bc) = assemble() else {
        return (Vec::new(), Vec::new());
    };
    let (stem, p) = identity_profiles()
        .into_iter()
        .find(|(stem, _)| stem == "lavapipe")
        .expect("lavapipe");
    let (_, plan, module) = compile_kept(&bc, "contain", &p).unwrap_or_else(|e| panic!("{e}"));
    let mut push = vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)];
    for f in &plan.push.fields {
        let PushLoc::Push { offset } = f.loc else {
            panic!("{stem}: {f:?} spilled");
        };
        let o = offset as usize;
        match f.field {
            PushField::SlotGuestBase(s) => {
                let at = [data_at, OUT, K][s.index as usize];
                push[o..o + 8].copy_from_slice(&slot_word(at).to_le_bytes());
            }
            PushField::SlotRemaining(s) => {
                push[o..o + 4].copy_from_slice(&lengths[s.index as usize].to_le_bytes())
            }
            PushField::ZeroAddress => push[o..o + 8].copy_from_slice(&ZERO.to_le_bytes()),
            PushField::DispatchRegion => {
                let region = region_bytes([0; 3], [THREADS, 1, 1], [THREADS, 1, 1]).expect("packs");
                push[o..o + 32].copy_from_slice(&region);
            }
            _ => panic!("{stem}: unexpected push field {f:?}"),
        }
    }
    // `data`'s words; at `DATA + 32` (the case before the bind offset) the 32 bytes below the
    // slot's address are `data` too, which no access may read.
    let data: Vec<u8> = (0..DATA_WORDS as u32)
        .flat_map(|w| (0x100 + w).to_le_bytes())
        .collect();
    let (data_at, data) = match data_at {
        a if a == DATA + 32 => (DATA, [data.clone(), data].concat()),
        a => (a, data),
    };
    let out: Vec<u8> = (0..OUT_WORDS)
        .flat_map(|_| SENTINEL.to_le_bytes())
        .collect();
    let buffer = |address: u64, bytes: Vec<u8>| Buffer {
        slot: None,
        address: Some(address),
        bytes,
    };
    let res = Resources {
        buffers: vec![
            buffer(data_at, data),
            buffer(OUT, out),
            buffer(K, k.to_le_bytes().to_vec()),
            buffer(ZERO, vec![0; 16]),
        ],
        push,
        spec: BTreeMap::from([
            (SpecKey::LocalSize(Axis::X), u64::from(THREADS)),
            (SpecKey::LocalSize(Axis::Y), 1),
            (SpecKey::LocalSize(Axis::Z), 1),
            (SpecKey::FlatLocalSize, u64::from(THREADS)),
        ]),
    };
    let got = run(&module, &Config::default(), res).unwrap_or_else(|e| panic!("{stem}: {e}"));
    let words = got.buffers[1]
        .chunks(4)
        .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
        .collect();
    (words, got.buffers[3].clone())
}

/// What `out` holds after a dispatch at `k` whose `data` reads as `read(word)` and whose stores
/// land below word `stored`.
fn expected(k: u32, read: impl Fn(u32) -> u32, stored: u32) -> Vec<u32> {
    let mut out = vec![SENTINEL; OUT_WORDS];
    for t in 0..THREADS {
        let i = t.wrapping_add(k);
        if i < stored {
            out[i as usize] = read(i);
        }
        // The atomic reaches word `t`, inside every pushed length here.
        out[t as usize] = out[t as usize].wrapping_add(1);
    }
    out
}

/// On the CPU interpreter a load outside its slot's pushed length reads zero, a store outside it
/// lands nowhere, a slot shorter than one access and a nil slot read zero, and nothing touches
/// the zero range; inside, the kernel moves its words.
#[test]
fn an_access_outside_the_pushed_length_reads_zero_and_writes_nothing() {
    let data_len = (4 * DATA_WORDS) as u32;
    let word = |i: u32| {
        if (i as usize) < DATA_WORDS {
            0x100 + i
        } else {
            0
        }
    };
    let cases: [(&str, u32, [u32; 2], u64, Vec<u32>); 5] = [
        (
            "inside",
            0,
            [data_len, 4 * OUT_WORDS as u32],
            DATA,
            expected(0, word, OUT_WORDS as u32),
        ),
        // Threads 2 and 3 read past `data` and store past `out`'s pushed 8 words.
        ("past", 6, [data_len, 32], DATA, expected(6, word, 8)),
        // Two bytes: shorter than one access.
        ("short", 0, [2, 64], DATA, expected(0, |_| 0, 16)),
        // Nil: the runtime pushes its nil buffer's address and length 0.
        ("nil", 0, [0, 64], NIL, expected(0, |_| 0, 16)),
        // 24 to 12 bytes before the bind offset, inside the buffer the interpreter holds: the
        // offset wraps to the top of its 32 bits, outside, and the stores land nowhere.
        (
            "before",
            u32::MAX - 5,
            [data_len, 64],
            DATA + 32,
            expected(u32::MAX - 5, |_| 0, 0),
        ),
    ];
    for (case, k, lengths, data_at, want) in cases {
        let (got, zero) = dispatch(k, lengths, data_at);
        if got.is_empty() {
            return;
        }
        assert_eq!(got, want, "{case}: out");
        assert_eq!(zero, vec![0; 16], "{case}: the zero range");
    }
}
