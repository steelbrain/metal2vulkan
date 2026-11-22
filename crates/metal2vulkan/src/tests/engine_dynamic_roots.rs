//! Dynamic buffer roots end to end.
//!
//! - A pointer loaded from a buffer (its 8-byte guest address) resolves through the root table's
//!   resident set: in-bounds accesses through it are byte-exact, including accesses at 64-bit
//!   offsets computed after the load; out-of-bounds accesses (past the end, below the start,
//!   wrapped offsets near 2³² and 2⁶⁴) never trap, load 0, and store only into the sink.
//! - A null pointer or a guest address outside the resident set is root 0: every access through
//!   it is bounded and counted in the `ExecutionStatus` miss word
//!   and bit.
//! - `inttoptr` resolves every address of resident sets of 0 to 33 entries exactly (the binary
//!   search's loop), including a pointer one past an allocation's end, and the probed addresses
//!   of sets of 2^16 to about 3 * 2^16 entries.
//! - A buffer parameter merged with a loaded pointer (a linked-list walk in a loop) is a wide
//!   pointer reached through its root entry (`BufferForm::Psb`, `DeviceAddresses`).
//! - `icmp` of loaded pointers with null and with a parameter compares Metal addresses across
//!   roots (a ring walk returns to its head; a null-terminated walk stops at null).
//! - Profile F refuses a dereference of a loaded pointer, naming what it lacks of
//!   `bufferDeviceAddress` and `shaderInt64`.
//! - A volatile load of a pointer, volatile accesses and `memcpy`s through it, and a volatile
//!   pointer load through it carry `Volatile` on every word, counted per storage class, and are
//!   exact.
//!
//! Every module passes L3 on the tested profiles and runs on the CPU interpreter. `llvm-as`
//! assembles the kernels; without it the tests skip visibly.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::bind::status::StatusWord;
use crate::device::profile::DeviceProfile;
use crate::device::requirements::VkFeature;
use crate::device::DevicePolicy;
use crate::error::{Capability, Error, Feature, Needs};
use crate::lower::select::Plan;
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::interface::{BufferForm, PushLoc, SlotInterface};
use crate::pipeline::sym::{PushField, ResourceKind, ResourceSlot, SlotClass, SlotRef, Stage};
use crate::pipeline::testing::{
    compile, device_f, link_and_validate, llvm_assemble, typed_module, AssembleError, PROFILE_F,
};
use crate::spirv::interp::{run, Buffer, Config, Resources};
use crate::spirv::ir::{Inst, StorageClass, Type, TypedModule};

/// The `ExecutionStatus` buffer's bytes, every word 0 at the start: through its last word.
const STATUS_BYTES: usize =
    crate::bind::status::StatusWord::SeqCstFences.byte_offset() as usize + 4;

const HEADER: &str = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n";

/// Device addresses of the interpreter's address-backed buffers.
const TABLE: u64 = 0x0000_0010_0000_0000;
const SINK: u64 = 0x0000_0011_0000_0000;
const STATUS: u64 = 0x0000_0012_0000_0000;
/// Host (device) and guest (Metal) addresses of resident allocation `k` (distinct domains, so a
/// mix-up of the two faults the interpreter or reads the wrong bytes).
const fn host(k: u64) -> u64 {
    0x0000_0020_0000_0000 + k * 0x1_0000_0000
}
const fn guest(k: u64) -> u64 {
    0x0000_0001_0000_0000 + k * 0x100
}
/// Bytes of each resident allocation (a multiple of 16).
const ALLOC: u64 = 64;

fn profile(stem: &str, policy: DevicePolicy) -> DeviceProfile {
    crate::pipeline::testing::profile(stem, policy).expect("committed profile")
}

/// The Profile-A profiles.
fn profiles() -> Vec<(String, DeviceProfile)> {
    ["moltenvk-m3", "lavapipe"]
        .into_iter()
        .map(|stem| {
            (
                stem.to_owned(),
                profile(stem, DevicePolicy::RESIDENT_TABLES),
            )
        })
        .collect()
}

fn assemble_text(name: &str, text: &str) -> Option<Vec<u8>> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "m2v-dynamic-{}-{}-{name}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).expect("scratch");
    let ll = dir.join("k.ll");
    std::fs::write(&ll, text).expect("write");
    let r = llvm_assemble(&ll);
    let _ = std::fs::remove_dir_all(&dir);
    match r {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {name}: {why}\n{text}"),
    }
}

/// Kernel metadata for `name` with three `device` buffers: 0 (read), 1 (read-write), 2 (read).
fn meta(name: &str) -> String {
    format!(
        "!air.kernel = !{{!9}}\n!air.version = !{{!17}}\n!air.language_version = !{{!18}}\n\
         !9 = !{{ptr @{name}, !10, !11}}\n!10 = !{{}}\n!11 = !{{!12, !13, !14}}\n\
         !12 = !{{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 8, !\"air.arg_type_align_size\", i32 8, !\"air.arg_type_name\", !\"ulong\", !\"air.arg_name\", !\"ptrs\"}}\n\
         !13 = !{{i32 1, !\"air.buffer\", !\"air.location_index\", i32 1, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"out\"}}\n\
         !14 = !{{i32 2, !\"air.buffer\", !\"air.location_index\", i32 2, i32 1, !\"air.read\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 8, !\"air.arg_type_align_size\", i32 8, !\"air.arg_type_name\", !\"ulong\", !\"air.arg_name\", !\"params\"}}\n\
         !17 = !{{i32 2, i32 8, i32 0}}\n!18 = !{{!\"Metal\", i32 4, i32 0, i32 0}}\n"
    )
}

fn kernel(name: &str, body: &str) -> String {
    format!(
        "{HEADER}define void @{name}(ptr addrspace(1) noundef \"air-buffer-no-alias\" %0, ptr addrspace(1) noundef \"air-buffer-no-alias\" %1, ptr addrspace(1) noundef \"air-buffer-no-alias\" %2) {{\nentry:\n{body}  ret void\n}}\n\n{}",
        meta(name)
    )
}

/// The pointer at element `params[0]` (a `u32`) of `ptrs`, displaced by the `i64` `params[1]`.
const CHASE: &str = "  %k = load i32, ptr addrspace(1) %2, align 8\n\
  %op = getelementptr i8, ptr addrspace(1) %2, i64 8\n\
  %off = load i64, ptr addrspace(1) %op, align 8\n\
  %pp = getelementptr ptr addrspace(1), ptr addrspace(1) %0, i32 %k\n\
  %p = load ptr addrspace(1), ptr addrspace(1) %pp, align 8\n\
  %q = getelementptr i8, ptr addrspace(1) %p, i64 %off\n";

/// One access kind through the chased pointer.
struct Leaf {
    name: &'static str,
    bytes: u64,
    body: &'static str,
    /// The stored bytes, for a store.
    store: Option<&'static [u8]>,
}

const LEAVES: [Leaf; 4] = [
    Leaf {
        name: "dyn_load_i32",
        bytes: 4,
        body: "  %v = load i32, ptr addrspace(1) %q, align 4\n  store i32 %v, ptr addrspace(1) %1, align 4\n",
        store: None,
    },
    Leaf {
        name: "dyn_load_i8",
        bytes: 1,
        body: "  %v = load i8, ptr addrspace(1) %q, align 1\n  store i8 %v, ptr addrspace(1) %1, align 4\n",
        store: None,
    },
    Leaf {
        name: "dyn_store_i32",
        bytes: 4,
        body: "  store i32 305419896, ptr addrspace(1) %q, align 4\n",
        store: Some(&[0x78, 0x56, 0x34, 0x12]),
    },
    Leaf {
        name: "dyn_store_i16",
        bytes: 2,
        body: "  store i16 -21829, ptr addrspace(1) %q, align 1\n",
        store: Some(&[0xBB, 0xAA]),
    },
];

/// A root table: entry 0 null, entry 1 the sink with the resident range, `statics` parameter
/// entries from root 2, and the resident entries `(host, guest, size)` from root `2 + statics`.
fn root_table(statics: &[(u64, u64, u64, u64)], resident: &[(u64, u64, u64)]) -> Vec<u8> {
    let first = 2 + statics.len() as u64;
    let n = first as usize + resident.len();
    let mut t = vec![0u8; 32 * n.max(2)];
    let mut put = |root: usize, field: usize, v: u64| {
        let at = 32 * root + 8 * field;
        t[at..at + 8].copy_from_slice(&v.to_le_bytes());
    };
    put(1, 0, SINK);
    put(1, 1, first);
    put(1, 2, resident.len() as u64);
    for (i, (h, g, s, b)) in statics.iter().enumerate() {
        put(2 + i, 0, *h);
        put(2 + i, 1, *g);
        put(2 + i, 2, *s);
        put(2 + i, 3, *b);
    }
    for (i, (h, g, s)) in resident.iter().enumerate() {
        let r = first as usize + i;
        put(r, 0, *h);
        put(r, 1, *g);
        put(r, 2, *s);
    }
    t
}

/// The push bytes of `plan`: the root-table and status addresses, and zero residuals.
fn push_bytes(plan: &Plan) -> Vec<u8> {
    let mut out = vec![0u8; 128];
    for f in &plan.push.fields {
        let PushLoc::Push { offset } = f.loc else {
            panic!("{:?} spilled to the ArgumentTable", f.field);
        };
        let at = offset as usize;
        match f.field {
            PushField::RootTableAddress => out[at..at + 8].copy_from_slice(&TABLE.to_le_bytes()),
            PushField::ExecutionStatusAddress => {
                out[at..at + 8].copy_from_slice(&STATUS.to_le_bytes())
            }
            PushField::SlotBindOffset(_) => {}
            other => panic!("unexpected push field {other:?}"),
        }
    }
    out
}

fn kslot(index: u32) -> ResourceSlot {
    ResourceSlot {
        stage: Stage::Kernel,
        kind: ResourceKind::Buffer,
        index,
    }
}

fn slot_buffer(index: u32, bytes: Vec<u8>) -> Buffer {
    Buffer {
        slot: Some(kslot(index)),
        address: None,
        bytes,
    }
}

fn addressed(address: u64, bytes: Vec<u8>) -> Buffer {
    Buffer {
        slot: None,
        address: Some(address),
        bytes,
    }
}

/// Distinct bytes for allocation `k`.
fn pattern(k: u64) -> Vec<u8> {
    (0..ALLOC)
        .map(|i| {
            (i as u8)
                .wrapping_mul(7)
                .wrapping_add(0x21u8.wrapping_add((k as u8).wrapping_mul(0x45)))
        })
        .collect()
}

fn status_words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
        .collect()
}

/// The resident-set search (its loop of ⌈log₂ N⌉ steps) finds every address of every entry of sets
/// of 0 to 33 entries, a pointer one past an entry's end included (its accesses are out of bounds, not
/// misses), and misses addresses in the gaps, below the first entry, and above the last.
#[test]
fn inttoptr_resolves_every_resident_address() {
    let body = "  %g = load i64, ptr addrspace(1) %2, align 8\n\
  %p = inttoptr i64 %g to ptr addrspace(1)\n\
  %v = load i32, ptr addrspace(1) %p, align 4\n\
  store i32 %v, ptr addrspace(1) %1, align 4\n";
    let Some(bc) = assemble_text("from_int", &kernel("from_int", body)) else {
        return;
    };
    let p = profile("moltenvk-m3", DevicePolicy::RESIDENT_TABLES);
    let a = compile(&bc, "from_int", &p).expect("compile");
    link_and_validate(&a, &p).expect("L3");
    let (plan, m) = typed_module(&bc, "from_int", &p).expect("module");
    for n in [0u64, 1, 2, 3, 5, 8, 16, 17, 33] {
        let resident: Vec<(u64, u64, u64)> = (0..n).map(|k| (host(k), guest(k), ALLOC)).collect();
        let mut probes: Vec<(u64, Option<(u64, u64)>)> = vec![(0, None), (guest(0) - 4, None)];
        for k in 0..n {
            probes.push((guest(k), Some((k, 0))));
            probes.push((guest(k) + 12, Some((k, 12))));
            probes.push((guest(k) + ALLOC - 4, Some((k, ALLOC - 4))));
            // One past the end: this allocation's, out of bounds; then the gap (a miss).
            probes.push((guest(k) + ALLOC, Some((k, ALLOC))));
            probes.push((guest(k) + ALLOC + 4, None));
        }
        probes.push((guest(n) + 8, None));
        for (addr, want) in probes {
            let mut buffers = vec![
                slot_buffer(0, vec![0; 16]),
                slot_buffer(1, vec![0; 16]),
                slot_buffer(2, addr.to_le_bytes().to_vec()),
                addressed(TABLE, root_table(&[(0, 0, 0, 0); 3], &resident)),
                addressed(SINK, vec![0; 64]),
                addressed(STATUS, vec![0; STATUS_BYTES]),
            ];
            for k in 0..n {
                buffers.push(addressed(host(k), pattern(k)));
            }
            let res = Resources {
                buffers,
                push: push_bytes(&plan),
                spec: Default::default(),
            };
            let case = format!("n={n} addr={addr:#x}");
            let out = run(&m, &Config::default(), res).unwrap_or_else(|e| panic!("{case}: {e}"));
            let got = &out.buffers[1][..4];
            let expect = match want {
                Some((k, rel)) if rel + 4 <= ALLOC => {
                    pattern(k)[rel as usize..rel as usize + 4].to_vec()
                }
                _ => vec![0; 4],
            };
            assert_eq!(got, &expect[..], "{case}");
            let misses = status_words(&out.buffers[5])[StatusWord::ResidentMisses.index() as usize];
            assert_eq!(misses, u32::from(want.is_none()), "{case}: misses");
        }
    }
}

/// The search finds the addresses of sets of 2^16 to about 3 * 2^16 entries (16 to 18 steps):
/// every entry probed, at its start, inside it, and one past its end, those around 2^16 among
/// them; the gaps, the address below the first entry and the one above the last miss.
#[test]
fn inttoptr_resolves_addresses_of_large_sets() {
    let body = "  %g = load i64, ptr addrspace(1) %2, align 8\n\
  %p = inttoptr i64 %g to ptr addrspace(1)\n\
  %v = load i32, ptr addrspace(1) %p, align 4\n\
  store i32 %v, ptr addrspace(1) %1, align 4\n";
    let Some(bc) = assemble_text("from_int", &kernel("from_int", body)) else {
        return;
    };
    let p = profile("moltenvk-m3", DevicePolicy::RESIDENT_TABLES);
    let a = compile(&bc, "from_int", &p).expect("compile");
    link_and_validate(&a, &p).expect("L3");
    let (plan, m) = typed_module(&bc, "from_int", &p).expect("module");
    let reach = 1u64 << 16;
    for n in [reach, reach + 1, 3 * reach - 7] {
        let resident: Vec<(u64, u64, u64)> = (0..n).map(|k| (host(k), guest(k), ALLOC)).collect();
        let table = root_table(&[(0, 0, 0, 0); 3], &resident);
        let mut probes: Vec<(u64, Option<(u64, u64)>)> = vec![(0, None), (guest(0) - 4, None)];
        for k in [0, 1, reach - 1, reach, reach + 1, n / 2, n - 2, n - 1] {
            if k >= n {
                continue;
            }
            probes.push((guest(k), Some((k, 0))));
            probes.push((guest(k) + 12, Some((k, 12))));
            probes.push((guest(k) + ALLOC, Some((k, ALLOC))));
            probes.push((guest(k) + ALLOC + 4, None));
        }
        probes.push((guest(n) + 8, None));
        for (addr, want) in probes {
            let mut buffers = vec![
                slot_buffer(0, vec![0; 16]),
                slot_buffer(1, vec![0; 16]),
                slot_buffer(2, addr.to_le_bytes().to_vec()),
                addressed(TABLE, table.clone()),
                addressed(SINK, vec![0; 64]),
                addressed(STATUS, vec![0; STATUS_BYTES]),
            ];
            if let Some((k, _)) = want {
                buffers.push(addressed(host(k), pattern(k)));
            }
            let res = Resources {
                buffers,
                push: push_bytes(&plan),
                spec: Default::default(),
            };
            let case = format!("n={n} addr={addr:#x}");
            let out = run(&m, &Config::default(), res).unwrap_or_else(|e| panic!("{case}: {e}"));
            let got = &out.buffers[1][..4];
            let expect = match want {
                Some((k, rel)) if rel + 4 <= ALLOC => {
                    pattern(k)[rel as usize..rel as usize + 4].to_vec()
                }
                _ => vec![0; 4],
            };
            assert_eq!(got, &expect[..], "{case}");
            let misses = status_words(&out.buffers[5])[StatusWord::ResidentMisses.index() as usize];
            assert_eq!(misses, u32::from(want.is_none()), "{case}: misses");
        }
    }
}

/// A linked-list walk: the head is the `[[buffer(0)]]` parameter and each next pointer is loaded
/// from the current node, so the loop's pointer phi merges the parameter with loaded pointers.
/// The parameter's slot is reached through its root entry (`Psb`, `DeviceAddresses`) and the phi
/// is a wide pointer.
fn walk_kernel() -> String {
    let body = "  %n = load i32, ptr addrspace(1) %2, align 8\n\
  br label %loop\n\
loop:\n\
  %p = phi ptr addrspace(1) [ %0, %entry ], [ %next, %loop ]\n\
  %i = phi i32 [ 0, %entry ], [ %i1, %loop ]\n\
  %acc = phi i32 [ 0, %entry ], [ %acc1, %loop ]\n\
  %vp = getelementptr inbounds i8, ptr addrspace(1) %p, i64 8\n\
  %v = load i32, ptr addrspace(1) %vp, align 4\n\
  %m = mul i32 %acc, 31\n\
  %acc1 = add i32 %m, %v\n\
  %next = load ptr addrspace(1), ptr addrspace(1) %p, align 8\n\
  %i1 = add i32 %i, 1\n\
  %done = icmp eq i32 %i1, %n\n\
  br i1 %done, label %exit, label %loop\n\
exit:\n\
  store i32 %acc1, ptr addrspace(1) %1, align 4\n";
    kernel("walk", body)
}

#[test]
fn a_parameter_merged_with_loaded_pointers_walks_a_list() {
    let Some(bc) = assemble_text("walk", &walk_kernel()) else {
        return;
    };
    for (what, p) in profiles() {
        let a = compile(&bc, "walk", &p).unwrap_or_else(|e| panic!("{what}: {e}"));
        link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
        let head = SlotRef {
            stage: Stage::Kernel,
            class: SlotClass::Buffer,
            index: 0,
        };
        match a.interface.slots.get(&head) {
            Some(SlotInterface::Buffer(b)) => assert_eq!(b.form, BufferForm::Psb, "{what}"),
            other => panic!("{what}: {other:?}"),
        }
        let (plan, m) = typed_module(&bc, "walk", &p).expect("module");
        assert!(
            plan.facts
                .contains(&RuntimeFact::DeviceAddresses { slot: head }),
            "{what}"
        );
        // Nodes: the head (bound at offset 16 of its allocation) and resident nodes 0..3, each
        // `{ next: ptr, value: u32 }`; the last node points at itself.
        let node = |next: u64, value: u32| {
            let mut b = vec![0u8; ALLOC as usize];
            b[..8].copy_from_slice(&next.to_le_bytes());
            b[8..12].copy_from_slice(&value.to_le_bytes());
            b
        };
        let head_host = 0x0000_0030_0000_0000u64;
        let head_guest = 0x0000_0000_4000_0000u64;
        let mut head_alloc = vec![0xEEu8; 16];
        head_alloc.extend(node(guest(0), 7));
        let nodes: Vec<Vec<u8>> = vec![
            node(guest(1) + 32, 11),
            {
                // Node 1 lives at byte 32 of allocation 1.
                let mut b = vec![0u8; 32];
                b.extend(node(guest(2), 13));
                b.truncate(ALLOC as usize);
                b
            },
            node(guest(2), 17),
        ];
        let status = vec![0; STATUS_BYTES];
        for steps in [1u32, 2, 3, 6] {
            let resident = [
                (host(0), guest(0), ALLOC),
                (host(1), guest(1), ALLOC),
                (host(2), guest(2), ALLOC),
            ];
            let statics = [
                (head_host, head_guest, head_alloc.len() as u64, 16),
                (0, 0, 0, 0),
                (0, 0, 0, 0),
            ];
            let mut params = vec![0u8; 16];
            params[..4].copy_from_slice(&steps.to_le_bytes());
            let mut buffers = vec![
                slot_buffer(1, vec![0; 16]),
                slot_buffer(2, params),
                addressed(TABLE, root_table(&statics, &resident)),
                addressed(SINK, vec![0; 64]),
                addressed(STATUS, status.clone()),
                addressed(head_host, head_alloc.clone()),
            ];
            for (k, n) in nodes.iter().enumerate() {
                buffers.push(addressed(host(k as u64), n.clone()));
            }
            let res = Resources {
                buffers,
                push: push_bytes(&plan),
                spec: Default::default(),
            };
            let out = run(&m, &Config::default(), res).unwrap_or_else(|e| panic!("{what}: {e}"));
            let values = [7u32, 11, 13, 17, 17, 17];
            let want = values
                .iter()
                .take(steps as usize)
                .fold(0u32, |acc, v| acc.wrapping_mul(31).wrapping_add(*v));
            let got = u32::from_le_bytes(out.buffers[0][..4].try_into().expect("word"));
            assert_eq!(got, want, "{what} steps={steps}");
            let st = status_words(&out.buffers[4]);
            assert_eq!(st[StatusWord::ResidentMisses.index() as usize], 0, "{what}");
            assert_eq!(st[StatusWord::Status.index() as usize], 0, "{what}");
        }
    }
}

/// Profile F has no device addresses: dereferencing a loaded pointer is refused, naming
/// `bufferDeviceAddress`.
#[test]
fn profile_f_refuses_a_loaded_pointer_dereference() {
    let leaf = &LEAVES[0];
    let Some(bc) = assemble_text(
        "floor",
        &kernel(leaf.name, &format!("{CHASE}{}", leaf.body)),
    ) else {
        return;
    };
    let f = device_f(DevicePolicy::RESIDENT_TABLES).expect(PROFILE_F);
    match compile(&bc, leaf.name, f.profile()) {
        Err(Error::Unsupported(u)) => {
            assert_eq!(u.feature, Feature::DereferenceLoadedPointer);
            assert_eq!(
                u.needs,
                Needs::AnyOf(vec![vec![Capability::Feature(
                    VkFeature::BufferDeviceAddress
                )]])
            );
        }
        other => panic!("expected a refusal, got {other:?}"),
    }
}

/// A walk that stops at a null next pointer or when the list returns to its head: `icmp` of a
/// loaded pointer with null and with the head parameter compares Metal addresses across roots
/// (the loaded head pointer resolves to the head's resident entry, the parameter is root 2 with a
/// bind offset), so both stops are found exactly.
fn ring_kernel() -> String {
    let body = "  %n = load i32, ptr addrspace(1) %2, align 8\n\
  br label %loop\n\
loop:\n\
  %p = phi ptr addrspace(1) [ %0, %entry ], [ %next, %loop ]\n\
  %i = phi i32 [ 0, %entry ], [ %i1, %loop ]\n\
  %acc = phi i32 [ 0, %entry ], [ %acc1, %loop ]\n\
  %vp = getelementptr inbounds i8, ptr addrspace(1) %p, i64 8\n\
  %v = load i32, ptr addrspace(1) %vp, align 4\n\
  %m = mul i32 %acc, 31\n\
  %acc1 = add i32 %m, %v\n\
  %next = load ptr addrspace(1), ptr addrspace(1) %p, align 8\n\
  %i1 = add i32 %i, 1\n\
  %back = icmp eq ptr addrspace(1) %next, %0\n\
  %nul = icmp eq ptr addrspace(1) %next, null\n\
  %stop = or i1 %back, %nul\n\
  %lim = icmp eq i32 %i1, %n\n\
  %done = or i1 %stop, %lim\n\
  br i1 %done, label %exit, label %loop\n\
exit:\n\
  store i32 %acc1, ptr addrspace(1) %1, align 4\n\
  %o1 = getelementptr inbounds i8, ptr addrspace(1) %1, i64 4\n\
  store i32 %i1, ptr addrspace(1) %o1, align 4\n";
    kernel("ring", body)
}

#[test]
fn loaded_pointers_compare_by_metal_address() {
    let Some(bc) = assemble_text("ring", &ring_kernel()) else {
        return;
    };
    let head_host = 0x0000_0030_0000_0000u64;
    let head_guest = 0x0000_0000_4000_0000u64;
    let node = |next: u64, value: u32| {
        let mut b = vec![0u8; ALLOC as usize];
        b[..8].copy_from_slice(&next.to_le_bytes());
        b[8..12].copy_from_slice(&value.to_le_bytes());
        b
    };
    for (what, p) in profiles() {
        let a = compile(&bc, "ring", &p).unwrap_or_else(|e| panic!("{what}: {e}"));
        link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
        let (plan, m) = typed_module(&bc, "ring", &p).expect("module");
        let census = plan.census();
        assert!(
            census.contains_key("access.dynamic.compare"),
            "{what}: {census:?}"
        );
        // (last node's next pointer, expected steps): back to the head (bound at offset 16),
        // null, and neither (the step limit of 9 stops a self-loop at node 2).
        for (last, steps) in [(head_guest + 16, 4u32), (0, 4), (guest(2), 9)] {
            let mut head_alloc = vec![0xEEu8; 16];
            head_alloc.extend(node(guest(0), 7));
            let nodes = [node(guest(1), 11), node(guest(2), 13), node(last, 17)];
            let status = vec![0; STATUS_BYTES];
            // The head's allocation is resident too (a bound argument), below the nodes.
            let resident = [
                (head_host, head_guest, head_alloc.len() as u64),
                (host(0), guest(0), ALLOC),
                (host(1), guest(1), ALLOC),
                (host(2), guest(2), ALLOC),
            ];
            let statics = [
                (head_host, head_guest, head_alloc.len() as u64, 16),
                (0, 0, 0, 0),
                (0, 0, 0, 0),
            ];
            let mut params = vec![0u8; 16];
            params[..4].copy_from_slice(&9u32.to_le_bytes());
            let mut buffers = vec![
                slot_buffer(1, vec![0; 16]),
                slot_buffer(2, params),
                addressed(TABLE, root_table(&statics, &resident)),
                addressed(SINK, vec![0; 64]),
                addressed(STATUS, status),
                addressed(head_host, head_alloc),
            ];
            for (k, n) in nodes.iter().enumerate() {
                buffers.push(addressed(host(k as u64), n.clone()));
            }
            let res = Resources {
                buffers,
                push: push_bytes(&plan),
                spec: Default::default(),
            };
            let case = format!("{what} last={last:#x}");
            let out = run(&m, &Config::default(), res).unwrap_or_else(|e| panic!("{case}: {e}"));
            let values = [7u32, 11, 13, 17, 17, 17, 17, 17, 17];
            let want = values
                .iter()
                .take(steps as usize)
                .fold(0u32, |acc, v| acc.wrapping_mul(31).wrapping_add(*v));
            let words = status_words(&out.buffers[0]);
            assert_eq!(words[..2], [want, steps], "{case}");
            assert_eq!(
                status_words(&out.buffers[4])[StatusWord::ResidentMisses.index() as usize],
                0,
                "{case}"
            );
        }
    }
}

/// Counts by storage class.
type ByClass = Vec<(StorageClass, u32)>;

/// The volatile loads and stores of `m` by the storage class of their pointer.
fn volatiles(m: &TypedModule) -> (ByClass, ByClass) {
    let class = |ptr| match m.value(ptr).and_then(|d| m.types().get(d.ty)) {
        Ok(Type::Pointer { class, .. }) => *class,
        other => panic!("an access through {other:?}"),
    };
    let (mut loads, mut stores) = (BTreeMap::new(), BTreeMap::new());
    for n in m.blocks().iter().flat_map(|b| &b.insts) {
        match &n.inst {
            Inst::Load { ptr, access } if access.volatile => {
                *loads.entry(class(*ptr)).or_insert(0) += 1
            }
            Inst::Store { ptr, access, .. } if access.volatile => {
                *stores.entry(class(*ptr)).or_insert(0) += 1
            }
            _ => {}
        }
    }
    (loads.into_iter().collect(), stores.into_iter().collect())
}

const SB: StorageClass = StorageClass::StorageBuffer;
const PSB: StorageClass = StorageClass::PhysicalStorageBuffer;

/// `llvm.memcpy` between device pointers.
const MEMCPY: &str =
    "declare void @llvm.memcpy.p1.p1.i64(ptr addrspace(1), ptr addrspace(1), i64, i1)\n";

/// A volatile load of the pointer itself and volatile accesses through it, a leaf's and a volatile
/// `memcpy`'s whose one endpoint is the wide pointer: the guest address's two words are volatile
/// loads of the slot, the leaf's word a volatile access of the allocation by device address, and
/// each access is exact in bounds.
#[test]
fn volatile_accesses_through_a_loaded_pointer_are_exact() {
    let chase = CHASE.replace("load ptr addrspace(1)", "load volatile ptr addrspace(1)");
    let leaves = [
        (
            Leaf {
                name: "dyn_volatile_load",
                bytes: 4,
                body: "  %v = load volatile i32, ptr addrspace(1) %q, align 4\n  store i32 %v, ptr addrspace(1) %1, align 4\n",
                store: None,
            },
            (vec![(SB, 2), (PSB, 1)], vec![]),
        ),
        (
            Leaf {
                name: "dyn_volatile_store",
                bytes: 4,
                body: "  store volatile i32 305419896, ptr addrspace(1) %q, align 4\n",
                store: Some(&[0x78, 0x56, 0x34, 0x12]),
            },
            (vec![(SB, 2)], vec![(PSB, 1)]),
        ),
        (
            Leaf {
                name: "dyn_volatile_copy_from",
                bytes: 4,
                body: "  call void @llvm.memcpy.p1.p1.i64(ptr addrspace(1) align 4 %1, ptr addrspace(1) align 4 %q, i64 4, i1 true)\n",
                store: None,
            },
            (vec![(SB, 2), (PSB, 1)], vec![(SB, 1)]),
        ),
        (
            // `params[0..4]`, the element index 1.
            Leaf {
                name: "dyn_volatile_copy_to",
                bytes: 4,
                body: "  call void @llvm.memcpy.p1.p1.i64(ptr addrspace(1) align 4 %q, ptr addrspace(1) align 4 %2, i64 4, i1 true)\n",
                store: Some(&[1, 0, 0, 0]),
            },
            (vec![(SB, 3)], vec![(PSB, 1)]),
        ),
    ];
    // Element 1 of `ptrs`: allocation 1 at byte 16.
    let ptrs: Vec<u8> = [guest(0), guest(1) + 16]
        .iter()
        .flat_map(|g| g.to_le_bytes())
        .collect();
    for (leaf, want) in &leaves {
        let (name, store) = (leaf.name, leaf.store);
        let text = format!("{}{MEMCPY}", kernel(name, &format!("{chase}{}", leaf.body)));
        let Some(bc) = assemble_text(name, &text) else {
            return;
        };
        for (what, p) in profiles() {
            let what = format!("{name} {what}");
            let a = compile(&bc, name, &p).unwrap_or_else(|e| panic!("{what}: {e}"));
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            let (plan, m) = typed_module(&bc, name, &p).expect("module");
            assert_eq!(&volatiles(&m), want, "{what}");
            let resident = [(host(0), guest(0), ALLOC), (host(1), guest(1), ALLOC)];
            // In bounds of the 64-byte allocation from byte 16.
            for off in [0u64, 4, 44] {
                let mut params = vec![0u8; 16];
                params[..4].copy_from_slice(&1u32.to_le_bytes());
                params[8..].copy_from_slice(&off.to_le_bytes());
                let data: Vec<Vec<u8>> = (0..2).map(pattern).collect();
                let res = Resources {
                    buffers: vec![
                        slot_buffer(0, ptrs.clone()),
                        slot_buffer(1, vec![0; 16]),
                        slot_buffer(2, params),
                        addressed(TABLE, root_table(&[(0, 0, 0, 0); 3], &resident)),
                        addressed(SINK, vec![0; 64]),
                        addressed(STATUS, vec![0; STATUS_BYTES]),
                        addressed(host(0), data[0].clone()),
                        addressed(host(1), data[1].clone()),
                    ],
                    push: push_bytes(&plan),
                    spec: Default::default(),
                };
                let case = format!("{what} off={off}");
                let out =
                    run(&m, &Config::default(), res).unwrap_or_else(|e| panic!("{case}: {e}"));
                let rel = 16 + off as usize;
                match store {
                    None => {
                        let mut want = vec![0u8; 16];
                        let n = leaf.bytes as usize;
                        want[..n].copy_from_slice(&data[1][rel..rel + n]);
                        assert_eq!(out.buffers[1], want, "{case}");
                    }
                    Some(bytes) => {
                        let mut want = data[1].clone();
                        want[rel..rel + bytes.len()].copy_from_slice(bytes);
                        assert_eq!(out.buffers[7], want, "{case}");
                        assert_eq!(out.buffers[6], data[0], "{case}");
                    }
                }
            }
        }
    }
}

/// A volatile load of a pointer from the allocation a loaded pointer reaches (a pointer loaded
/// through a wide pointer): its guest address, two whole words, is one volatile load by device
/// address, and the pointer it yields resolves to its allocation.
#[test]
fn a_volatile_pointer_load_through_a_loaded_pointer_is_exact() {
    let name = "dyn_volatile_chase";
    let body = "  %p2 = load volatile ptr addrspace(1), ptr addrspace(1) %q, align 8\n\
  %v = load i32, ptr addrspace(1) %p2, align 4\n\
  store i32 %v, ptr addrspace(1) %1, align 4\n";
    let Some(bc) = assemble_text(name, &kernel(name, &format!("{CHASE}{body}"))) else {
        return;
    };
    // Element 1 of `ptrs` is allocation 1 at byte 16; at byte 24 of it, a pointer to byte 12 of
    // allocation 0.
    let ptrs: Vec<u8> = [guest(0), guest(1) + 16]
        .iter()
        .flat_map(|g| g.to_le_bytes())
        .collect();
    let mut data: Vec<Vec<u8>> = (0..2).map(pattern).collect();
    data[1][24..32].copy_from_slice(&(guest(0) + 12).to_le_bytes());
    let mut params = vec![0u8; 16];
    params[..4].copy_from_slice(&1u32.to_le_bytes());
    params[8..].copy_from_slice(&8u64.to_le_bytes());
    let resident = [(host(0), guest(0), ALLOC), (host(1), guest(1), ALLOC)];
    for (what, p) in profiles() {
        let a = compile(&bc, name, &p).unwrap_or_else(|e| panic!("{what}: {e}"));
        link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
        let (plan, m) = typed_module(&bc, name, &p).expect("module");
        assert_eq!(volatiles(&m), (vec![(PSB, 1)], vec![]), "{what}");
        let res = Resources {
            buffers: vec![
                slot_buffer(0, ptrs.clone()),
                slot_buffer(1, vec![0; 16]),
                slot_buffer(2, params.clone()),
                addressed(TABLE, root_table(&[(0, 0, 0, 0); 3], &resident)),
                addressed(SINK, vec![0; 64]),
                addressed(STATUS, vec![0; STATUS_BYTES]),
                addressed(host(0), data[0].clone()),
                addressed(host(1), data[1].clone()),
            ],
            push: push_bytes(&plan),
            spec: Default::default(),
        };
        let out = run(&m, &Config::default(), res).unwrap_or_else(|e| panic!("{what}: {e}"));
        let mut want = vec![0u8; 16];
        want[..4].copy_from_slice(&data[0][12..16]);
        assert_eq!(out.buffers[1], want, "{what}");
        let status = status_words(&out.buffers[5]);
        assert_eq!(
            status[StatusWord::ResidentMisses.index() as usize],
            0,
            "{what}"
        );
    }
}

/// A kernel that loads the pointers at elements `0..n` of `ptrs` and stores the word at offset
/// 4 of each to `out[i]`.
fn loads_kernel(name: &str, n: u32) -> String {
    let mut body = String::new();
    for i in 0..n {
        body.push_str(&format!(
            "  %pp{i} = getelementptr ptr addrspace(1), ptr addrspace(1) %0, i32 {i}\n\
             %p{i} = load ptr addrspace(1), ptr addrspace(1) %pp{i}, align 8\n\
             %q{i} = getelementptr i8, ptr addrspace(1) %p{i}, i64 4\n\
             %v{i} = load i32, ptr addrspace(1) %q{i}, align 4\n\
             %o{i} = getelementptr i32, ptr addrspace(1) %1, i32 {i}\n\
             store i32 %v{i}, ptr addrspace(1) %o{i}, align 4\n"
        ));
    }
    kernel(name, &body)
}

/// A module with several resident-set searches defines the search once and every loaded pointer
/// calls it, so a further pointer load adds its call and the access through the pointer (384 words
/// on `moltenvk-m3`), not another 16-step search (about 1,100 words in place); a single search is
/// emitted in place, without a function. The census charges the search once, as
/// `access.dynamic.resolve`.
#[test]
fn the_resident_set_search_is_one_function_per_module() {
    const FUNCTION: u32 = 54;
    const CALL: u32 = 57;
    let mut sizes = Vec::new();
    for (name, n) in [("one_load", 1u32), ("three_loads", 3)] {
        let Some(bc) = assemble_text(name, &loads_kernel(name, n)) else {
            return;
        };
        for (what, p) in profiles() {
            let a = compile(&bc, name, &p).unwrap_or_else(|e| panic!("{what} {name}: {e}"));
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what} {name}: L3: {e}"));
            let words = &a.module.words;
            let mut ops = BTreeMap::new();
            let mut i = 5;
            while let Some(w) = words.get(i) {
                *ops.entry(w & 0xFFFF).or_insert(0u32) += 1;
                i += ((w >> 16) as usize).max(1);
            }
            let (functions, calls) = if n == 1 { (1, None) } else { (2, Some(&n)) };
            assert_eq!(
                ops.get(&FUNCTION),
                Some(&functions),
                "{what} {name}: functions"
            );
            assert_eq!(ops.get(&CALL), calls, "{what} {name}: calls");
            let (plan, _) = typed_module(&bc, name, &p).expect("module");
            let census = plan.census();
            assert_eq!(
                census.get("access.dynamic.resolve").map(|(_, w)| *w),
                Some(u64::from(crate::lower::access::dynamic::SEARCH_WORDS)),
                "{what} {name}: the search is charged once"
            );
            sizes.push((what, n, words.len()));
        }
    }
    for (what, n, words) in &sizes {
        if *n != 3 {
            continue;
        }
        let one = sizes
            .iter()
            .find(|(w, k, _)| w == what && *k == 1)
            .map(|(_, _, s)| *s)
            .expect("the one-load module");
        assert!(
            words - one < 2 * 600,
            "{what}: two more pointer loads add {} words",
            words - one
        );
    }
}
