//! Pointer values end to end.
//!
//! - Compares of device pointers are compares of Metal addresses: two pointers into one slot by
//!   their offsets, any other pair by `SlotGuestBase` plus displacement (a `u64`, or two words
//!   without `shaderInt64`), so two parameters bound to one buffer compare equal, an unbound
//!   (`nil`) parameter compares equal to null, and a loaded pointer compares with a parameter by
//!   address.
//! - `null`, `undef`, and `poison` pointers are the null root: as a `select` side, a phi's
//!   incoming value, a GEP base, or an access's pointer, narrow or wide.
//! - Pointers into one constant global compare by their offsets.
//! - On Profile F a static pointer converts to, and is stored as, its Metal address; stored
//!   volatile, its words are volatile stores.
//!
//! Every module passes L3 on the tested profiles and runs on the CPU interpreter. `llvm-as`
//! assembles the kernels; without it the tests skip visibly.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::budget::SPIRV_WORDS;
use crate::device::profile::{BufferAccess, DeviceProfile};
use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::error::{Error, Feature, Needs};
use crate::library::FunctionConstantValues;
use crate::lower::select::{LowerInputs, Plan};
use crate::pipeline::compile::{compile_bitcode, EntryRef};
use crate::pipeline::interface::PushLoc;
use crate::pipeline::sym::{PushField, ResourceKind, ResourceSlot, Stage, SyntheticBinding};
use crate::pipeline::testing::{
    compile, device_f, link_and_validate, llvm_assemble, meter, typed_module_with, AssembleError,
    PROFILE_F,
};
use crate::spirv::interp::{run, Buffer, Config, Order, Resources};
use crate::spirv::ir::{Inst, StorageClass, Type, TypedModule};

const HEADER: &str = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n";

/// Device addresses of the interpreter's address-backed buffers.
const TABLE: u64 = 0x0000_0010_0000_0000;
const SINK: u64 = 0x0000_0011_0000_0000;
const STATUS: u64 = 0x0000_0012_0000_0000;

/// Host (device) and default guest (Metal) addresses of `[[buffer(i)]]`.
const fn slot_host(i: u64) -> u64 {
    0x0000_0020_0000_0000 + i * 0x1_0000_0000
}
const fn slot_guest(i: u64) -> u64 {
    0x0000_0001_0000_0000 + i * 0x1_0000
}

/// Committed profile `stem`, or [`PROFILE_F`]'s (`runtime-floor` without `bufferDeviceAddress`).
fn profile(stem: &str) -> DeviceProfile {
    if stem == PROFILE_F {
        return device_f(DevicePolicy::RESIDENT_TABLES)
            .expect(PROFILE_F)
            .profile()
            .clone();
    }
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles")
        .join(format!("{stem}.json"));
    let text = std::fs::read_to_string(path).expect("profile");
    let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    Device::from_queries(desc, DevicePolicy::RESIDENT_TABLES)
        .expect("device")
        .profile()
        .clone()
}

/// Profile A, and Profile F ([`PROFILE_F`], below the runtime's floor).
fn profiles() -> Vec<(String, DeviceProfile)> {
    ["moltenvk-m3", "lavapipe", PROFILE_F]
        .into_iter()
        .map(|stem| (stem.to_owned(), profile(stem)))
        .collect()
}

fn addressed(p: &DeviceProfile) -> bool {
    p.fields().buffer_access == BufferAccess::Addressed
}

fn assemble(name: &str, text: &str) -> Option<Vec<u8>> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "m2v-pointer-values-{}-{}-{name}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).expect("scratch");
    let ll = dir.join("k.ll");
    std::fs::write(&ll, format!("{HEADER}{text}")).expect("write");
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

/// Kernel metadata for `name`: `buffers` read-write `device` buffers of element type `uint`,
/// then the thread's index in its threadgroup.
fn meta(name: &str, buffers: u32) -> String {
    let mut s = format!(
        "!air.kernel = !{{!0}}\n!air.version = !{{!90}}\n!air.language_version = !{{!91}}\n\
         !0 = !{{ptr @{name}, !1, !2}}\n!1 = !{{}}\n!2 = !{{"
    );
    let args: Vec<String> = (0..=buffers).map(|i| format!("!{}", 10 + i)).collect();
    s.push_str(&args.join(", "));
    s.push_str("}\n");
    for i in 0..buffers {
        s.push_str(&format!(
            "!{} = !{{i32 {i}, !\"air.buffer\", !\"air.location_index\", i32 {i}, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"b{i}\"}}\n",
            10 + i
        ));
    }
    s.push_str(&format!(
        "!{} = !{{i32 {buffers}, !\"air.thread_index_in_threadgroup\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"t\"}}\n\
         !90 = !{{i32 2, i32 8, i32 0}}\n!91 = !{{!\"Metal\", i32 4, i32 0, i32 0}}\n",
        10 + buffers
    ));
    s
}

/// A kernel `name(b0, .., b{buffers-1}, t)` whose entry block is `body` (which may open more
/// blocks and must end in a terminator-free block), with module-scope `globals`.
fn kernel(name: &str, buffers: u32, body: &str, globals: &str) -> String {
    let params: Vec<String> = (0..buffers)
        .map(|i| format!("ptr addrspace(1) noundef \"air-buffer-no-alias\" %b{i}"))
        .collect();
    format!(
        "{globals}\ndefine void @{name}({}, i32 %t) {{\nentry:\n{body}  ret void\n}}\n\n{}",
        params.join(", "),
        meta(name, buffers)
    )
}

fn kslot(index: u32) -> ResourceSlot {
    ResourceSlot {
        stage: Stage::Kernel,
        kind: ResourceKind::Buffer,
        index,
    }
}

fn words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn le_words(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

/// The root table: null, the sink with the resident range, `[[buffer(i)]]` at root `2 + i`, and
/// the resident set (every slot) sorted by guest address.
fn root_table(slots: &[Vec<u8>]) -> Vec<u8> {
    let first = 2 + slots.len() as u64;
    let n = first as usize + slots.len();
    let mut t = vec![0u8; 32 * n];
    let mut put = |root: usize, field: usize, v: u64| {
        let at = 32 * root + 8 * field;
        t[at..at + 8].copy_from_slice(&v.to_le_bytes());
    };
    put(1, 0, SINK);
    put(1, 1, first);
    put(1, 2, slots.len() as u64);
    for (i, b) in slots.iter().enumerate() {
        for root in [2 + i, first as usize + i] {
            put(root, 0, slot_host(i as u64));
            put(root, 1, slot_guest(i as u64));
            put(root, 2, b.len() as u64);
        }
    }
    t
}

/// The resources of `plan` over `slots` for `threads` invocations in one threadgroup, with each
/// slot's `SlotGuestBase` from `guest` (the bound buffer's Metal address; 0 for a `nil` one).
/// The buffers the run's output lists are the slots, then the root table, the sink, the status
/// buffer, and the constant blob when the module reads one.
fn resources(
    plan: &Plan,
    slots: &[Vec<u8>],
    threads: u32,
    guest: &dyn Fn(usize) -> u64,
) -> Resources {
    let mut spec = BTreeMap::new();
    spec.extend(crate::pipeline::threadgroup_constants([threads, 1, 1]));
    let mut push = vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)];
    for f in &plan.push.fields {
        let PushLoc::Push { offset } = f.loc else {
            panic!("{:?} spilled to the ArgumentTable", f.field);
        };
        let at = offset as usize;
        match f.field {
            PushField::RootTableAddress => push[at..at + 8].copy_from_slice(&TABLE.to_le_bytes()),
            PushField::ExecutionStatusAddress => {
                push[at..at + 8].copy_from_slice(&STATUS.to_le_bytes())
            }
            PushField::SlotBindOffset(_) | PushField::DispatchRegion => {
                push[at..at + f.size as usize].fill(0)
            }
            PushField::SlotGuestBase(s) => {
                assert_eq!(f.size, 8, "a guest base is eight bytes");
                push[at..at + 8].copy_from_slice(&guest(s.index as usize).to_le_bytes())
            }
            other => panic!("unexpected push field {other:?}"),
        }
    }
    let mut buffers: Vec<Buffer> = slots
        .iter()
        .enumerate()
        .map(|(i, b)| Buffer {
            slot: Some(kslot(i as u32)),
            address: Some(slot_host(i as u64)),
            bytes: b.clone(),
        })
        .collect();
    buffers.push(Buffer {
        slot: None,
        address: Some(TABLE),
        bytes: root_table(slots),
    });
    buffers.push(Buffer {
        slot: None,
        address: Some(SINK),
        bytes: vec![0; 64],
    });
    buffers.push(Buffer {
        slot: None,
        address: Some(STATUS),
        bytes: vec![0; crate::bind::status::StatusWord::SeqCstFences.byte_offset() as usize + 4],
    });
    if let Some(b) = &plan.families.access.blob {
        let mut bytes = b.bytes.to_vec();
        bytes.resize(bytes.len().next_multiple_of(16), 0);
        buffers.push(Buffer {
            slot: Some(ResourceSlot {
                stage: Stage::Kernel,
                kind: ResourceKind::Synthetic(SyntheticBinding::ConstantBlob),
                index: 0,
            }),
            address: None,
            bytes,
        });
    }
    Resources {
        buffers,
        push,
        spec,
    }
}

/// Compile entry `entry` of `bc` for `p`, pass L3, and return the plan and the typed module the
/// interpreter runs.
fn checked(bc: &[u8], entry: &str, p: &DeviceProfile, what: &str) -> (Plan, TypedModule) {
    let inputs = LowerInputs::new();
    let a = compile_bitcode(
        bc,
        EntryRef::Name(entry),
        &FunctionConstantValues::defaults(),
        p,
        &inputs,
        &mut meter(),
        SPIRV_WORDS,
    )
    .unwrap_or_else(|e| panic!("{what}: {e}"));
    link_and_validate(&a, p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
    let (plan, m) = typed_module_with(bc, entry, p, &inputs).expect("module");
    (plan, m)
}

/// Run `m` in both scheduling orders: each order's output buffers.
fn run_orders(m: &TypedModule, res: &Resources, what: &str) -> Vec<Vec<Vec<u8>>> {
    [Order::Forward, Order::Reverse]
        .into_iter()
        .map(|order| {
            let cfg = Config {
                order,
                ..Config::default()
            };
            run(m, &cfg, res.clone())
                .unwrap_or_else(|e| panic!("{what} {order:?}: {e}"))
                .buffers
        })
        .collect()
}

/// The uses of the strategies whose census name starts with `prefix`.
fn uses(plan: &Plan, prefix: &str) -> u32 {
    plan.census()
        .iter()
        .filter(|(k, _)| k.starts_with(prefix))
        .map(|(_, (n, _))| *n)
        .sum()
}

/// Thread `t` compares pointers and stores each result (0 or 1) at `b0[8t + k]`: `p = b1 + t`
/// and `q = b1 + (t & ~1)` (one slot: equal for an even `t`, `p > q` for an odd one), `b1` with
/// `b2` for equality and order, `b2` (which nothing accesses) and `p` with null, and `b1 + 4`
/// with `b2 + 4` (a displacement that carries into the high word of a guest base near 2³²).
const COMPARE_BODY: &str = "  %h = lshr i32 %t, 1
  %e = shl i32 %h, 1
  %p = getelementptr inbounds i32, ptr addrspace(1) %b1, i32 %t
  %q = getelementptr inbounds i32, ptr addrspace(1) %b1, i32 %e
  %v = load i32, ptr addrspace(1) %p, align 4
  %c0 = icmp eq ptr addrspace(1) %p, %q
  %c1 = icmp ugt ptr addrspace(1) %p, %q
  %c2 = icmp eq ptr addrspace(1) %b1, %b2
  %c3 = icmp ult ptr addrspace(1) %b1, %b2
  %c4 = icmp eq ptr addrspace(1) %b2, null
  %c5 = icmp ne ptr addrspace(1) %p, null
  %b14 = getelementptr inbounds i8, ptr addrspace(1) %b1, i64 4
  %b24 = getelementptr inbounds i8, ptr addrspace(1) %b2, i64 4
  %c6 = icmp uge ptr addrspace(1) %b14, %b24
  %c7 = icmp sle ptr addrspace(1) %b14, %b24
  %base = shl i32 %t, 3
  %r0 = zext i1 %c0 to i32
  %o0 = getelementptr inbounds i32, ptr addrspace(1) %b0, i32 %base
  store i32 %r0, ptr addrspace(1) %o0, align 4
  %i1 = add i32 %base, 1
  %r1 = zext i1 %c1 to i32
  %o1 = getelementptr inbounds i32, ptr addrspace(1) %b0, i32 %i1
  store i32 %r1, ptr addrspace(1) %o1, align 4
  %i2 = add i32 %base, 2
  %r2 = zext i1 %c2 to i32
  %o2 = getelementptr inbounds i32, ptr addrspace(1) %b0, i32 %i2
  store i32 %r2, ptr addrspace(1) %o2, align 4
  %i3 = add i32 %base, 3
  %r3 = zext i1 %c3 to i32
  %o3 = getelementptr inbounds i32, ptr addrspace(1) %b0, i32 %i3
  store i32 %r3, ptr addrspace(1) %o3, align 4
  %i4 = add i32 %base, 4
  %r4 = zext i1 %c4 to i32
  %o4 = getelementptr inbounds i32, ptr addrspace(1) %b0, i32 %i4
  store i32 %r4, ptr addrspace(1) %o4, align 4
  %i5 = add i32 %base, 5
  %r5 = zext i1 %c5 to i32
  %o5 = getelementptr inbounds i32, ptr addrspace(1) %b0, i32 %i5
  store i32 %r5, ptr addrspace(1) %o5, align 4
  %i6 = add i32 %base, 6
  %r6 = zext i1 %c6 to i32
  %o6 = getelementptr inbounds i32, ptr addrspace(1) %b0, i32 %i6
  store i32 %r6, ptr addrspace(1) %o6, align 4
  %i7 = add i32 %base, 7
  %r7 = zext i1 %c7 to i32
  %o7 = getelementptr inbounds i32, ptr addrspace(1) %b0, i32 %i7
  store i32 %r7, ptr addrspace(1) %o7, align 4
";

/// Compares of static pointers are compares of Metal addresses. Two pointers into one slot compare
/// their offsets (no guest base); two slots compare `SlotGuestBase` plus displacement, so two
/// parameters bound to one buffer are equal, a `nil` (unbound, guest base 0) one is null, and the
/// order of two bases that differ only in the high word (or whose displacement carries into it) is
/// exact in two words as in a `u64`.
#[test]
fn static_pointer_compares_are_metal_address_compares() {
    const N: u32 = 8;
    let Some(bc) = assemble("compare", &kernel("compare", 3, COMPARE_BODY, "")) else {
        return;
    };
    // (name, guest base of b1, of b2)
    let cases: [(&str, u64, u64); 5] = [
        ("distinct", slot_guest(1), slot_guest(2)),
        ("aliased", slot_guest(1), slot_guest(1)),
        ("b2 nil", slot_guest(1), 0),
        ("high words", 0x2_0000_0000, 0x1_FFFF_FFF0),
        ("carry", 0x1_FFFF_FFFC, 0x2_0000_0000),
    ];
    for (what, p) in profiles() {
        let (plan, m) = checked(&bc, "compare", &p, &what);
        assert_eq!(uses(&plan, "access.pointer.compare_offsets"), 2, "{what}");
        assert_eq!(uses(&plan, "access.pointer.compare_addresses"), 6, "{what}");
        let words_form = !p.fields().leaf.int64;
        assert_eq!(
            uses(&plan, "access.pointer.compare_addresses.words"),
            if words_form { 6 } else { 0 },
            "{what}"
        );
        for (case, g1, g2) in cases {
            let slots = vec![
                vec![0xEE; 32 * N as usize],
                vec![0; 4 * N as usize],
                vec![0; 16],
            ];
            let guest = |i: usize| match i {
                1 => g1,
                2 => g2,
                _ => slot_guest(i as u64),
            };
            let res = resources(&plan, &slots, N, &guest);
            let (a, b) = (g1 + 4, g2 + 4);
            for out in run_orders(&m, &res, &format!("{what} {case}")) {
                let w = words(&out[0]);
                for t in 0..N as usize {
                    let want = [
                        u32::from(t % 2 == 0),
                        u32::from(t % 2 == 1),
                        u32::from(g1 == g2),
                        u32::from(g1 < g2),
                        u32::from(g2 == 0),
                        1,
                        u32::from(a >= b),
                        u32::from((a as i64) <= (b as i64)),
                    ];
                    assert_eq!(&w[8 * t..8 * t + 8], &want, "{what} {case}: thread {t}");
                }
            }
        }
    }
}

/// A pointer whose offset range the analysis does not bound lies in its slot's first 2³² bytes when
/// the slot's window does. `b1`'s only access is `b1[16]` (the window `[64, 68)`); `p = b1 + x` for
/// a loaded `x` compares below it (`p < b1 + 64` exactly when `x < 16`, which the window's span
/// alone would get wrong) and against `b2` by address.
#[test]
fn an_unbounded_pointer_lies_in_its_slots_first_span() {
    let body = "  %x = load i32, ptr addrspace(1) %b0, align 4
  %p = getelementptr inbounds i32, ptr addrspace(1) %b1, i32 %x
  %q = getelementptr inbounds i32, ptr addrspace(1) %b1, i64 16
  %v = load i32, ptr addrspace(1) %q, align 4
  %lt = icmp ult ptr addrspace(1) %p, %q
  %ge = icmp uge ptr addrspace(1) %p, %b2
  %r1 = zext i1 %lt to i32
  %o1 = getelementptr inbounds i32, ptr addrspace(1) %b0, i64 1
  store i32 %r1, ptr addrspace(1) %o1, align 4
  %r2 = zext i1 %ge to i32
  %o2 = getelementptr inbounds i32, ptr addrspace(1) %b0, i64 2
  store i32 %r2, ptr addrspace(1) %o2, align 4
  %o3 = getelementptr inbounds i32, ptr addrspace(1) %b0, i64 3
  store i32 %v, ptr addrspace(1) %o3, align 4
";
    let Some(bc) = assemble("span", &kernel("span", 3, body, "")) else {
        return;
    };
    let gap = slot_guest(2) - slot_guest(1);
    for (what, p) in profiles() {
        let (plan, m) = checked(&bc, "span", &p, &what);
        assert_eq!(uses(&plan, "access.pointer.compare_offsets"), 1, "{what}");
        assert_eq!(uses(&plan, "access.pointer.compare_addresses"), 1, "{what}");
        for x in [
            0u32,
            1,
            15,
            16,
            17,
            1000,
            (gap / 4) as u32 - 1,
            (gap / 4) as u32,
        ] {
            let b1: Vec<u32> = (0..32).map(|i| 100 + i).collect();
            let slots = vec![le_words(&[x, 0, 0, 0]), le_words(&b1), vec![0; 16]];
            let res = resources(&plan, &slots, 1, &|i| slot_guest(i as u64));
            for out in run_orders(&m, &res, &format!("{what} x={x}")) {
                let want = [
                    x,
                    u32::from(x < 16),
                    u32::from(u64::from(x) * 4 >= gap),
                    116,
                ];
                assert_eq!(words(&out[0]), want, "{what}: x = {x}");
            }
        }
    }
}

/// Null constants as a `select` side, a phi's incoming value, a GEP base, and an access's
/// pointer (device and constant), each through the null root.
const NULL_BODY: &str = "  %lo = icmp ult i32 %t, 4
  %pa = getelementptr inbounds i32, ptr addrspace(1) %b1, i32 %t
  %s = select i1 %lo, ptr addrspace(1) %pa, ptr addrspace(1) null
  %sv = load i32, ptr addrspace(1) %s, align 4
  %sn = icmp eq ptr addrspace(1) %s, null
  %n16 = getelementptr inbounds i8, ptr addrspace(1) null, i64 16
  %nv = load i32, ptr addrspace(1) %n16, align 4
  store i32 7, ptr addrspace(1) %n16, align 4
  %cv = load i32, ptr addrspace(2) null, align 4
  %cg = getelementptr inbounds i32, ptr addrspace(2) null, i64 3
  %cgv = load i32, ptr addrspace(2) %cg, align 4
  %u = select i1 %lo, ptr addrspace(1) undef, ptr addrspace(1) %pa
  %uv = load i32, ptr addrspace(1) %u, align 4
  %t16 = add i32 %t, 16
  %pw = getelementptr inbounds i32, ptr addrspace(1) %b2, i32 %t16
  %w = select i1 %lo, ptr addrspace(1) poison, ptr addrspace(1) %pw
  %wv = select i1 %lo, i32 9, i32 %t
  store i32 %wv, ptr addrspace(1) %w, align 4
  %odd = trunc i32 %t to i1
  br i1 %odd, label %a, label %m
a:
  %t8 = add i32 %t, 8
  %qa = getelementptr inbounds i32, ptr addrspace(1) %b1, i32 %t8
  br label %m
m:
  %ph = phi ptr addrspace(1) [ %qa, %a ], [ null, %entry ]
  %pv = load i32, ptr addrspace(1) %ph, align 4
  %pn = icmp eq ptr addrspace(1) %ph, null
  %base = shl i32 %t, 3
  %o0 = getelementptr inbounds i32, ptr addrspace(1) %b0, i32 %base
  store i32 %sv, ptr addrspace(1) %o0, align 4
  %i1 = add i32 %base, 1
  %r1 = zext i1 %sn to i32
  %o1 = getelementptr inbounds i32, ptr addrspace(1) %b0, i32 %i1
  store i32 %r1, ptr addrspace(1) %o1, align 4
  %i2 = add i32 %base, 2
  %sum = add i32 %nv, %cv
  %sum2 = add i32 %sum, %cgv
  %o2 = getelementptr inbounds i32, ptr addrspace(1) %b0, i32 %i2
  store i32 %sum2, ptr addrspace(1) %o2, align 4
  %i3 = add i32 %base, 3
  %o3 = getelementptr inbounds i32, ptr addrspace(1) %b0, i32 %i3
  store i32 %uv, ptr addrspace(1) %o3, align 4
  %i4 = add i32 %base, 4
  %o4 = getelementptr inbounds i32, ptr addrspace(1) %b0, i32 %i4
  store i32 %pv, ptr addrspace(1) %o4, align 4
  %i5 = add i32 %base, 5
  %r5 = zext i1 %pn to i32
  %o5 = getelementptr inbounds i32, ptr addrspace(1) %b0, i32 %i5
  store i32 %r5, ptr addrspace(1) %o5, align 4
";

/// Null constants are the null root, with the rule for its dereference (which Metal leaves
/// undefined): a pointer that can only be null loads 0 and drops its stores, and one whose other
/// candidate is a single slot dereferences that slot at the null pointer's offset (contained
/// there). `b1[i] = 100 + i`; thread `t` reads `b1[t]` through a `select` with null (`b1[0]` for
/// `t >= 4`, where the pointer compares equal to null), 0 through a GEP on null and through
/// constant-space null pointers, `b1[t]` or `b1[0]` through a `select` with `undef`, and `b1[t +
/// 8]` or `b1[0]` through a phi with a null incoming value; it stores through a `select` with
/// `poison`: `t` at `b2[t + 16]` for `t >= 4`, else 9 at `b2[0]`. Its store through null reaches no
/// buffer.
#[test]
fn null_constants_are_the_null_root() {
    const N: u32 = 8;
    let Some(bc) = assemble("nulls", &kernel("nulls", 3, NULL_BODY, "")) else {
        return;
    };
    for (what, p) in profiles() {
        let (plan, m) = checked(&bc, "nulls", &p, &what);
        assert_eq!(uses(&plan, "access.pointer.compare_addresses"), 2, "{what}");
        let b1: Vec<u32> = (0..32).map(|i| 100 + i).collect();
        let slots = vec![vec![0xEE; 32 * N as usize], le_words(&b1), vec![0; 128]];
        let res = resources(&plan, &slots, N, &|i| slot_guest(i as u64));
        for out in run_orders(&m, &res, &what) {
            let w = words(&out[0]);
            for t in 0..N {
                let lo = t < 4;
                let odd = t % 2 == 1;
                let want = [
                    if lo { 100 + t } else { 100 },
                    u32::from(!lo),
                    0,
                    if lo { 100 } else { 100 + t },
                    if odd { 108 + t } else { 100 },
                    u32::from(!odd),
                ];
                let at = 8 * t as usize;
                assert_eq!(&w[at..at + 6], &want, "{what}: thread {t}");
            }
            assert_eq!(out[1], le_words(&b1), "{what}: b1");
            let got = words(&out[2]);
            for (i, x) in got.iter().enumerate() {
                let i = i as u32;
                let want = match i {
                    0 => 9,
                    20..24 => i - 16,
                    _ => 0,
                };
                assert_eq!(*x, want, "{what}: b2[{i}]");
            }
        }
    }
}

/// A loaded (wide) pointer merges with null constants: `p = b0[0]` (`b2`'s guest address); thread
/// `t` reads `b2[t]` through `select(t < 4, p + t, null)` and `b2[t + 8]` or 0 through a phi with a
/// null incoming value, and compares the loaded pointer with the static parameter `b2` (which it
/// points into) by address. Profile A only (a loaded pointer needs device addresses).
#[test]
fn wide_pointers_merge_with_null_and_compare_with_parameters() {
    const N: u32 = 8;
    let body = "  %p = load ptr addrspace(1), ptr addrspace(1) %b0, align 8
  %lo = icmp ult i32 %t, 4
  %pt = getelementptr inbounds i32, ptr addrspace(1) %p, i32 %t
  %s = select i1 %lo, ptr addrspace(1) %pt, ptr addrspace(1) null
  %sv = load i32, ptr addrspace(1) %s, align 4
  %eq = icmp eq ptr addrspace(1) %pt, %b2
  %lt = icmp ult ptr addrspace(1) %b2, %pt
  %odd = trunc i32 %t to i1
  br i1 %odd, label %a, label %m
a:
  %t8 = add i32 %t, 8
  %qa = getelementptr inbounds i32, ptr addrspace(1) %p, i32 %t8
  br label %m
m:
  %ph = phi ptr addrspace(1) [ %qa, %a ], [ null, %entry ]
  %pv = load i32, ptr addrspace(1) %ph, align 4
  %base = shl i32 %t, 2
  %o0 = getelementptr inbounds i32, ptr addrspace(1) %b1, i32 %base
  store i32 %sv, ptr addrspace(1) %o0, align 4
  %i1 = add i32 %base, 1
  %o1 = getelementptr inbounds i32, ptr addrspace(1) %b1, i32 %i1
  store i32 %pv, ptr addrspace(1) %o1, align 4
  %i2 = add i32 %base, 2
  %r2 = zext i1 %eq to i32
  %o2 = getelementptr inbounds i32, ptr addrspace(1) %b1, i32 %i2
  store i32 %r2, ptr addrspace(1) %o2, align 4
  %i3 = add i32 %base, 3
  %r3 = zext i1 %lt to i32
  %o3 = getelementptr inbounds i32, ptr addrspace(1) %b1, i32 %i3
  store i32 %r3, ptr addrspace(1) %o3, align 4
";
    let Some(bc) = assemble("wide_nulls", &kernel("wide_nulls", 3, body, "")) else {
        return;
    };
    for (what, p) in profiles().into_iter().filter(|(_, p)| addressed(p)) {
        let (plan, m) = checked(&bc, "wide_nulls", &p, &what);
        assert_eq!(uses(&plan, "access.pointer.compare_addresses"), 2, "{what}");
        let b2: Vec<u32> = (0..32).map(|i| 200 + i).collect();
        let slots = vec![
            slot_guest(2).to_le_bytes().to_vec(),
            vec![0xEE; 16 * N as usize],
            le_words(&b2),
        ];
        let res = resources(&plan, &slots, N, &|i| slot_guest(i as u64));
        for out in run_orders(&m, &res, &what) {
            let w = words(&out[1]);
            for t in 0..N {
                let want = [
                    if t < 4 { 200 + t } else { 0 },
                    if t % 2 == 1 { 208 + t } else { 0 },
                    u32::from(t == 0),
                    u32::from(t > 0),
                ];
                let at = 4 * t as usize;
                assert_eq!(&w[at..at + 4], &want, "{what}: thread {t}");
            }
        }
    }
}

/// Pointers into one constant global compare by their offsets. Thread `t` sums `tab[t & 3 ..]` by
/// walking a pointer to the end of the table (the loop stops when the advanced pointer equals the
/// end pointer) and also stores whether its start is below the end.
#[test]
fn constant_global_pointers_compare_by_offset() {
    const N: u32 = 8;
    let globals = "@tab = internal addrspace(2) constant [4 x i32] [i32 1, i32 20, i32 300, i32 4000], align 4";
    let body = "  %s = and i32 %t, 3
  %start = getelementptr inbounds [4 x i32], ptr addrspace(2) @tab, i32 0, i32 %s
  %end = getelementptr inbounds [4 x i32], ptr addrspace(2) @tab, i32 0, i32 4
  %below = icmp ult ptr addrspace(2) %start, %end
  br label %loop
loop:
  %p = phi ptr addrspace(2) [ %start, %entry ], [ %p1, %loop ]
  %acc = phi i32 [ 0, %entry ], [ %a1, %loop ]
  %v = load i32, ptr addrspace(2) %p, align 4
  %a1 = add i32 %acc, %v
  %p1 = getelementptr inbounds i32, ptr addrspace(2) %p, i32 1
  %done = icmp eq ptr addrspace(2) %p1, %end
  br i1 %done, label %exit, label %loop
exit:
  %base = shl i32 %t, 1
  %o0 = getelementptr inbounds i32, ptr addrspace(1) %b0, i32 %base
  store i32 %a1, ptr addrspace(1) %o0, align 4
  %i1 = add i32 %base, 1
  %r1 = zext i1 %below to i32
  %o1 = getelementptr inbounds i32, ptr addrspace(1) %b0, i32 %i1
  store i32 %r1, ptr addrspace(1) %o1, align 4
";
    let Some(bc) = assemble("walk", &kernel("walk", 1, body, globals)) else {
        return;
    };
    let tab = [1u32, 20, 300, 4000];
    for (what, p) in profiles() {
        let (plan, m) = checked(&bc, "walk", &p, &what);
        assert_eq!(uses(&plan, "access.pointer.compare_offsets"), 2, "{what}");
        let slots = vec![vec![0xEE; 8 * N as usize]];
        let res = resources(&plan, &slots, N, &|i| slot_guest(i as u64));
        for out in run_orders(&m, &res, &what) {
            let w = words(&out[0]);
            for t in 0..N as usize {
                let sum: u32 = tab[t & 3..].iter().sum();
                assert_eq!(&w[2 * t..2 * t + 2], &[sum, 1], "{what}: thread {t}");
            }
        }
    }
}

/// Conversions and stores of static pointers: `ptrtoint` of `b1 + 12` to `i64`, `i32`, and `i1`,
/// and `b1 + 12` and null stored as pointers.
const CONVERT_BODY: &str = "  %p = getelementptr inbounds i8, ptr addrspace(1) %b1, i64 12
  %g = ptrtoint ptr addrspace(1) %p to i64
  store i64 %g, ptr addrspace(1) %b0, align 8
  %g32 = ptrtoint ptr addrspace(1) %p to i32
  %o8 = getelementptr inbounds i8, ptr addrspace(1) %b0, i64 8
  store i32 %g32, ptr addrspace(1) %o8, align 4
  %g1 = ptrtoint ptr addrspace(1) %p to i1
  %g1z = zext i1 %g1 to i32
  %o12 = getelementptr inbounds i8, ptr addrspace(1) %b0, i64 12
  store i32 %g1z, ptr addrspace(1) %o12, align 4
  %o16 = getelementptr inbounds i8, ptr addrspace(1) %b0, i64 16
  store ptr addrspace(1) %p, ptr addrspace(1) %o16, align 8
  %o24 = getelementptr inbounds i8, ptr addrspace(1) %b0, i64 24
  store ptr addrspace(1) null, ptr addrspace(1) %o24, align 8
  %v = load i32, ptr addrspace(1) %p, align 4
  %o32 = getelementptr inbounds i8, ptr addrspace(1) %b0, i64 32
  store i32 %v, ptr addrspace(1) %o32, align 4
";

/// On Profile F (no root entries) a static pointer converts to, and is stored as, its Metal address
/// `SlotGuestBase + displacement`, computed in two words without `shaderInt64` (the guest base
/// `0x1_FFFF_FFF8` makes the displacement carry into the high word); on Profile A the same kernel
/// takes the wide form. Either way every result is exact. With the pointer stores volatile, each
/// stored address's two words are volatile stores of the slot, and nothing else is.
#[test]
fn static_pointers_convert_to_their_metal_addresses() {
    let volatile = CONVERT_BODY.replace("  store ptr", "  store volatile ptr");
    assert_eq!(volatile.matches("store volatile ptr").count(), 2);
    for (form, body, want_volatile) in [
        ("plain", CONVERT_BODY, (vec![], vec![])),
        ("volatile", volatile.as_str(), (vec![], vec![(SB, 4)])),
    ] {
        let Some(bc) = assemble("convert", &kernel("convert", 2, body, "")) else {
            return;
        };
        check_convert(&bc, form, &want_volatile);
    }
}

fn check_convert(bc: &[u8], form: &str, want_volatile: &(ByClass, ByClass)) {
    let g1: u64 = 0x1_FFFF_FFF8;
    for (what, p) in profiles() {
        let what = format!("{what} {form}");
        let (plan, m) = checked(bc, "convert", &p, &what);
        // The pointers are stored through `b0`, a descriptor view, so the stores' names carry no
        // reach; a static-root pointer store exists only without device addresses, which would
        // otherwise make the stored pointer wide.
        let census = plan.census();
        if addressed(&p) {
            assert_eq!(uses(&plan, "access.dynamic.ptrtoint"), 3, "{what}");
            assert_eq!(
                census.get("access.dynamic.store_pointer").map(|e| e.0),
                Some(2),
                "{what}"
            );
        } else {
            assert_eq!(uses(&plan, "access.pointer.ptrtoint"), 3, "{what}");
            assert_eq!(
                census.get("access.pointer.store_pointer").map(|e| e.0),
                Some(2),
                "{what}"
            );
        }
        assert_eq!(&volatiles(&m), want_volatile, "{what}");
        let b1: Vec<u32> = (0..8).map(|i| 50 + i).collect();
        let slots = vec![vec![0xEE; 40], le_words(&b1)];
        // On Profile A the root entry of `b1` carries its guest address; the push field and the
        // entry agree.
        let guest = |i: usize| {
            if i == 1 && !addressed(&p) {
                g1
            } else {
                slot_guest(i as u64)
            }
        };
        let res = resources(&plan, &slots, 1, &guest);
        let a = guest(1) + 12;
        let mut want = a.to_le_bytes().to_vec();
        want.extend_from_slice(&(a as u32).to_le_bytes());
        want.extend_from_slice(&u32::from(a & 1 == 1).to_le_bytes());
        want.extend_from_slice(&a.to_le_bytes());
        want.extend_from_slice(&0u64.to_le_bytes());
        want.extend_from_slice(&53u32.to_le_bytes());
        want.extend_from_slice(&[0xEE; 4]);
        for out in run_orders(&m, &res, &what) {
            assert_eq!(out[0], want, "{what}");
        }
    }
}

/// Counts by storage class.
type ByClass = Vec<(StorageClass, u32)>;

const SB: StorageClass = StorageClass::StorageBuffer;

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

/// Profile F still refuses what needs device addresses: a pointer made from an integer (its
/// dereference reads memory no descriptor binds), naming what the profile lacks.
#[test]
fn profile_f_refuses_a_pointer_made_from_an_integer() {
    let body = "  %g = load i64, ptr addrspace(1) %b0, align 8
  %p = inttoptr i64 %g to ptr addrspace(1)
  %v = load i32, ptr addrspace(1) %p, align 4
  store i32 %v, ptr addrspace(1) %b1, align 4
";
    let Some(bc) = assemble("from_int", &kernel("from_int", 2, body, "")) else {
        return;
    };
    let stem = PROFILE_F;
    let p = profile(stem);
    let e = compile(&bc, "from_int", &p).expect_err("refused");
    let Error::Unsupported(u) = e else {
        panic!("{stem}: {e:?}");
    };
    assert_eq!(
        (u.feature, u.needs),
        (
            Feature::DereferenceLoadedPointer,
            Needs::AnyOf(vec![p.addressed_needs()])
        ),
        "{stem}"
    );
}

/// The null root is a device or constant pointer's. A threadgroup or private pointer merged with
/// its `null` (the arena family's, which has no null offset) stays a typed refusal or lowers; it is
/// never an internal error.
#[test]
fn a_null_threadgroup_or_private_select_side_is_no_internal_error() {
    let tg = "@tg = internal addrspace(3) global [8 x i32] undef, align 4";
    let bodies = [
        (
            "tg_select",
            tg,
            "  %c = icmp ult i32 %t, 4
  %p = getelementptr inbounds [8 x i32], ptr addrspace(3) @tg, i32 0, i32 %t
  store i32 %t, ptr addrspace(3) %p, align 4
  %s = select i1 %c, ptr addrspace(3) %p, ptr addrspace(3) null
  %v = load i32, ptr addrspace(3) %s, align 4
  store i32 %v, ptr addrspace(1) %b0, align 4
",
        ),
        (
            "private_select",
            "",
            "  %a = alloca [8 x i32], align 4
  %c = icmp ult i32 %t, 4
  %i = and i32 %t, 7
  %p = getelementptr inbounds [8 x i32], ptr %a, i32 0, i32 %i
  store i32 %t, ptr %p, align 4
  %s = select i1 %c, ptr %p, ptr null
  %v = load i32, ptr %s, align 4
  store i32 %v, ptr addrspace(1) %b0, align 4
",
        ),
        (
            "private_bytes_select",
            "",
            "  %a = alloca [16 x i8], align 4
  %c = icmp ult i32 %t, 4
  %i = and i32 %t, 3
  %o = shl i32 %i, 2
  %p = getelementptr inbounds i8, ptr %a, i32 %o
  store i32 %t, ptr %p, align 4
  %q = getelementptr inbounds i8, ptr %a, i32 1
  store i8 3, ptr %q, align 1
  %s = select i1 %c, ptr %p, ptr null
  %v = load i32, ptr %s, align 4
  store i32 %v, ptr addrspace(1) %b0, align 4
",
        ),
    ];
    for (name, globals, body) in bodies {
        let Some(bc) = assemble(name, &kernel(name, 1, body, globals)) else {
            return;
        };
        for stem in ["moltenvk-m3", PROFILE_F] {
            match compile(&bc, name, &profile(stem)) {
                Ok(_) | Err(Error::Unsupported(_)) => {}
                Err(e) => panic!("{name} on {stem}: {e:?}"),
            }
        }
    }
}
