//! Buffers a kernel reaches through a device address held in memory or made from a slot pointer,
//! under the identity domains (`PointerDomain::Identity` with `HandleDomain::Identity`), on every
//! committed profile `Device::from_queries` derives there (our own MSL: `engine/ab_pointers` and
//! `engine/ab_tier2`'s `ab_indexed`).
//!
//! - `ab_sum` reads two buffers through the two pointer members of an argument buffer,
//!   `ab_address` one through a `gpuAddress` written into a constant buffer's bytes, and
//!   `ab_indexed` the texture of an argument buffer a pointer member of the bound one reaches:
//!   each loaded word is the device address of the bytes it names, used as it is. No plan pushes
//!   `RootTableAddress` (there is no root table) or `ExecutionStatusAddress` (no access through a
//!   loaded address is counted), the buffer kernels declare no descriptor, and each passes L3.
//! - `ab_slot_address` converts a slot pointer to an integer and increments the element through
//!   it: the slot stays a static root, reached at its pushed address and checked against its
//!   pushed bounds (`SlotBounds`), and no parameter is a wide pointer.
//! - On the CPU interpreter, with every buffer above 4 GiB and the loaded addresses and the slot
//!   bound at offsets past their buffers' starts, `ab_sum` and `ab_address` store the words the
//!   addresses name, and `ab_slot_address` stores each element's Metal address (the slot's
//!   address, `gpuAddress` plus its bind offset, plus the element's offset) and increments it.
//!   Nothing else is written.
//! - A slot pointer stored through another slot (`sp_into_slot`, generated) is stored as its
//!   address and passes L3; one stored through a pointer loaded from memory
//!   (`sp_through_loaded`) or into a thread cell whose bytes are read at an offset only the run
//!   knows, an open cell (`sp_open_cell`), is `Unsupported{Lowering(Access)}`, not implemented.
//!
//! `llvm-as` assembles the fixtures; without it the tests skip visibly.

use std::collections::BTreeMap;

use super::engine_word_phases::{bounds_word, identity_profiles, slot_word, ZERO_UNMAPPED};
use crate::device::profile::DeviceProfile;
use crate::error::{Error, Feature, LowerFamily, Needs};
use crate::lower::dispatch::region_bytes;
use crate::lower::select::Plan;
use crate::pipeline::interface::PushLoc;
use crate::pipeline::sym::{Axis, PushField, SlotClass, SlotRef, SpecKey, Stage};
use crate::pipeline::testing::{
    compile, compile_kept, engine_dir, link_and_validate, llvm_assemble, llvm_assemble_text,
    AssembleError,
};
use crate::spirv::interp::{run, Buffer, Config, Resources};
use crate::spirv::ir::TypedModule;

/// Threads a threadgroup, and threadgroups.
const GROUP: u32 = 8;
const GROUPS: u32 = 2;
const THREADS: usize = (GROUP * GROUPS) as usize;

/// Where the interpreter holds the buffers: the two slots, and the two buffers only a loaded
/// address names. Each lies above 4 GiB, so a dropped high word shows.
const SLOT0: u64 = 0x1_0000_0000;
const SLOT1: u64 = 0x4_0000_0000;
const FIRST: u64 = 0x2_0000_0000;
const SECOND: u64 = 0x3_0000_1000;

fn assemble(name: &str) -> Option<Vec<u8>> {
    match llvm_assemble(&engine_dir().join(format!("{name}.ll"))) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(e) => panic!("{name}: {e:?}"),
    }
}

fn slot(index: u32) -> SlotRef {
    SlotRef {
        stage: Stage::Kernel,
        class: SlotClass::Buffer,
        index,
    }
}

fn words(values: impl IntoIterator<Item = u32>) -> Vec<u8> {
    values.into_iter().flat_map(u32::to_le_bytes).collect()
}

fn addresses(values: &[u64]) -> Vec<u8> {
    values.iter().flat_map(|a| a.to_le_bytes()).collect()
}

/// `entry` of `bc` compiled on `stem`'s profile `p`, in the identity form: no
/// `RootTableAddress` and no `ExecutionStatusAddress` among its push fields, no descriptor where it
/// reaches memory through `buffers` alone (a texture is an element of set 1's arrays), and an
/// artifact that passes L3.
fn compiled(
    bc: &[u8],
    (entry, buffers): (&str, bool),
    stem: &str,
    p: &DeviceProfile,
) -> (Plan, TypedModule) {
    let (a, plan, module) =
        compile_kept(bc, entry, p).unwrap_or_else(|e| panic!("{stem}: {entry}: {e}"));
    if buffers {
        assert!(a.interface.descriptors.is_empty(), "{stem}: {entry}");
    }
    let fields: Vec<PushField> = plan.push.fields.iter().map(|f| f.field).collect();
    for absent in [
        PushField::RootTableAddress,
        PushField::ExecutionStatusAddress,
    ] {
        assert!(!fields.contains(&absent), "{stem}: {entry}: {fields:?}");
    }
    link_and_validate(&a, p).unwrap_or_else(|e| panic!("{stem}: {entry}: L3: {e}"));
    (plan, module)
}

/// The push bytes of `plan`, slot `i` bound at `slots[i]`: its buffer's address, its bind offset
/// and its buffer's length.
fn push(plan: &Plan, what: &str, slots: &[(u64, u64, usize)]) -> Vec<u8> {
    let mut push = vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)];
    let region = region_bytes([0; 3], [GROUP * GROUPS, 1, 1], [GROUP, 1, 1]).expect("packs");
    for f in &plan.push.fields {
        let PushLoc::Push { offset } = f.loc else {
            panic!("{what}: {f:?} spilled");
        };
        let o = offset as usize;
        match f.field {
            PushField::SlotGuestBase(s) => {
                let (base, bind, _) = slots[s.index as usize];
                push[o..o + 8].copy_from_slice(&slot_word(base + bind).to_le_bytes());
            }
            PushField::SlotBounds(s) => {
                let (_, bind, bytes) = slots[s.index as usize];
                push[o..o + 8].copy_from_slice(&bounds_word(bytes, bind));
            }
            PushField::ZeroAddress => push[o..o + 8].copy_from_slice(&ZERO_UNMAPPED.to_le_bytes()),
            PushField::DispatchRegion => push[o..o + 32].copy_from_slice(&region),
            _ => panic!("{what}: unexpected push field {f:?}"),
        }
    }
    push
}

/// The buffers at `at`, each with its bytes.
fn buffers(at: &[(u64, &[u8])]) -> Vec<Buffer> {
    (at.iter())
        .map(|(address, bytes)| Buffer {
            slot: None,
            address: Some(*address),
            bytes: bytes.to_vec(),
        })
        .collect()
}

fn spec() -> BTreeMap<SpecKey, u64> {
    BTreeMap::from([
        (SpecKey::LocalSize(Axis::X), u64::from(GROUP)),
        (SpecKey::LocalSize(Axis::Y), 1),
        (SpecKey::LocalSize(Axis::Z), 1),
        (SpecKey::FlatLocalSize, u64::from(GROUP)),
    ])
}

fn config() -> Config {
    Config {
        workgroups: [GROUPS, 1, 1],
        ..Config::default()
    }
}

/// An argument buffer's two pointer members, each the address of a buffer, the second 16 bytes
/// past its buffer's start as an encoder's `setBuffer:offset:` writes it: thread `i` stores
/// `a[i] + b[i]` through them, and the plan reads no root table.
#[test]
fn an_argument_buffers_pointer_members_are_the_device_addresses_they_hold() {
    let Some(bc) = assemble("ab_pointers") else {
        return;
    };
    let n = THREADS as u32 + 4;
    let a = words((0..n).map(|k| 0x1000 + k));
    let b = words((0..n).map(|k| 0x7000_0000 + 3 * k));
    let args = addresses(&[FIRST, SECOND + 16]);
    // One spare word past the output's end, which the kernel must not write.
    let out = vec![0u8; 4 * THREADS + 4];
    let mut want = words((0..THREADS as u32).map(|i| (0x1000 + i) + (0x7000_0000 + 3 * (4 + i))));
    want.extend_from_slice(&[0; 4]);
    for (stem, p) in identity_profiles() {
        let (plan, module) = compiled(&bc, ("ab_sum", true), &stem, &p);
        let res = Resources {
            push: push(
                &plan,
                &stem,
                &[(SLOT0, 0, args.len()), (SLOT1, 0, out.len())],
            ),
            buffers: buffers(&[(SLOT0, &args), (SLOT1, &out), (FIRST, &a), (SECOND, &b)]),
            spec: spec(),
        };
        let got = run(&module, &config(), res).unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert_eq!(got.buffers[1], want, "{stem}: out");
        assert_eq!(
            (&got.buffers[2], &got.buffers[3]),
            (&a, &b),
            "{stem}: read only"
        );
    }
}

/// A `gpuAddress` the program wrote into a constant buffer's bytes, 8 bytes past its buffer's
/// start: thread `i` stores the word `i` past it.
#[test]
fn a_stored_gpu_address_is_the_device_address_it_holds() {
    let Some(bc) = assemble("ab_pointers") else {
        return;
    };
    let n = THREADS as u32 + 2;
    let words_at = words((0..n).map(|k| 0xA000_0000 | k));
    let held = addresses(&[FIRST + 8]);
    let out = vec![0u8; 4 * THREADS + 4];
    let mut want = words((0..THREADS as u32).map(|i| 0xA000_0000 | (2 + i)));
    want.extend_from_slice(&[0; 4]);
    for (stem, p) in identity_profiles() {
        let (plan, module) = compiled(&bc, ("ab_address", true), &stem, &p);
        let res = Resources {
            push: push(
                &plan,
                &stem,
                &[(SLOT0, 0, held.len()), (SLOT1, 0, out.len())],
            ),
            buffers: buffers(&[(SLOT0, &held), (SLOT1, &out), (FIRST, &words_at)]),
            spec: spec(),
        };
        let got = run(&module, &config(), res).unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert_eq!(got.buffers[1], want, "{stem}: out");
    }
}

/// A slot pointer converted to an integer is its Metal address: the slot's address (bound 16
/// bytes past its buffer's start) plus the element's offset. The slot stays a static root, its
/// run-time accesses checked against its pushed bounds, and the increment through the same
/// pointer lands in the element.
#[test]
fn a_slot_pointer_converted_to_an_integer_is_its_metal_address() {
    let Some(bc) = assemble("ab_pointers") else {
        return;
    };
    const BIND: u64 = 16;
    let elements: Vec<u32> = (0..THREADS as u32 + 1).map(|k| 0x500 + 7 * k).collect();
    let mut buf = vec![0xEE; BIND as usize];
    buf.extend_from_slice(&words(elements.iter().copied()));
    let out = vec![0u8; 8 * THREADS];
    let want_out = addresses(
        &(0..THREADS as u64)
            .map(|i| SLOT0 + BIND + 4 * i)
            .collect::<Vec<_>>(),
    );
    let mut want_buf = vec![0xEE; BIND as usize];
    want_buf.extend_from_slice(&words(elements[..THREADS].iter().map(|v| v + 1)));
    want_buf.extend_from_slice(&elements[THREADS].to_le_bytes());
    for (stem, p) in identity_profiles() {
        let (plan, module) = compiled(&bc, ("ab_slot_address", true), &stem, &p);
        assert!(
            plan.families.access.dynamic.params.is_empty(),
            "{stem}: a wide parameter"
        );
        assert!(
            (plan.push.fields.iter()).any(|f| f.field == PushField::SlotBounds(slot(0))),
            "{stem}: the slot is not checked: {:?}",
            plan.push.fields
        );
        let res = Resources {
            push: push(
                &plan,
                &stem,
                &[(SLOT0, BIND, buf.len()), (SLOT1, 0, out.len())],
            ),
            buffers: buffers(&[(SLOT0, &buf), (SLOT1, &out)]),
            spec: spec(),
        };
        let got = run(&module, &config(), res).unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert_eq!(got.buffers[1], want_out, "{stem}: out");
        assert_eq!(got.buffers[0], want_buf, "{stem}: buf");
    }
}

/// The texture of an argument buffer a pointer member of the bound one reaches (`ab_indexed`)
/// compiles with no root table: its handle is read at the loaded address.
#[test]
fn a_texture_behind_an_argument_buffers_pointer_member_is_read_at_the_loaded_address() {
    let Some(bc) = assemble("ab_tier2") else {
        return;
    };
    for (stem, p) in identity_profiles() {
        compiled(&bc, ("ab_indexed", false), &stem, &p);
    }
}

/// Kernels that store a slot pointer, `buf + i`: through another slot (`sp_into_slot`), through a
/// pointer loaded from the first slot (`sp_through_loaded`), and into a thread cell of two
/// pointers whose bytes are read back as an integer at an index only the run knows
/// (`sp_open_cell`).
const STORED_SLOT_POINTERS: &str = r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

define void @sp_into_slot(ptr addrspace(1) noundef %cells, ptr addrspace(1) noundef %buf, i32 noundef %i) {
  %z = zext i32 %i to i64
  %cell = getelementptr inbounds ptr addrspace(1), ptr addrspace(1) %cells, i64 %z
  %p = getelementptr inbounds i32, ptr addrspace(1) %buf, i64 %z
  store ptr addrspace(1) %p, ptr addrspace(1) %cell, align 8
  ret void
}

define void @sp_through_loaded(ptr addrspace(1) noundef %holder, ptr addrspace(1) noundef %buf, i32 noundef %i) {
  %cells = load ptr addrspace(1), ptr addrspace(1) %holder, align 8
  %z = zext i32 %i to i64
  %cell = getelementptr inbounds ptr addrspace(1), ptr addrspace(1) %cells, i64 %z
  %p = getelementptr inbounds i32, ptr addrspace(1) %buf, i64 %z
  store ptr addrspace(1) %p, ptr addrspace(1) %cell, align 8
  ret void
}

define void @sp_open_cell(ptr addrspace(1) noundef %out, ptr addrspace(1) noundef %buf, i32 noundef %i) {
  %cells = alloca [2 x ptr addrspace(1)], align 8
  %z = zext i32 %i to i64
  %p = getelementptr inbounds i32, ptr addrspace(1) %buf, i64 %z
  %q = getelementptr inbounds i32, ptr addrspace(1) %p, i64 1
  store ptr addrspace(1) %p, ptr %cells, align 8
  %second = getelementptr inbounds [2 x ptr addrspace(1)], ptr %cells, i64 0, i64 1
  store ptr addrspace(1) %q, ptr %second, align 8
  %k = and i64 %z, 1
  %at = getelementptr inbounds [2 x i64], ptr %cells, i64 0, i64 %k
  %bits = load i64, ptr %at, align 8
  %o = getelementptr inbounds i64, ptr addrspace(1) %out, i64 %z
  store i64 %bits, ptr addrspace(1) %o, align 8
  ret void
}

!air.kernel = !{!0, !6, !8}
!air.version = !{!11}
!air.language_version = !{!12}
!0 = !{ptr @sp_into_slot, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"uint*", !"air.arg_name", !"cells"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"buf"}
!5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"i"}
!6 = !{ptr @sp_through_loaded, !1, !7}
!7 = !{!13, !4, !5}
!8 = !{ptr @sp_open_cell, !1, !9}
!9 = !{!10, !4, !5}
!10 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"ulong", !"air.arg_name", !"out"}
!11 = !{i32 2, i32 8, i32 0}
!12 = !{!"Metal", i32 4, i32 0, i32 0}
!13 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"uint**", !"air.arg_name", !"holder"}
"#;

/// A slot pointer stored through another slot is its address, stored as an 8-byte leaf through a
/// static destination: `sp_into_slot` compiles in the identity form and passes L3. Stored through
/// a pointer loaded from memory, or into an open thread cell, it is refused, not implemented,
/// as `Unsupported{Lowering(Access)}` on every profile the identity domains derive.
#[test]
fn a_slot_pointer_stored_through_a_loaded_pointer_or_into_an_open_cell_is_refused() {
    let bc = match llvm_assemble_text(STORED_SLOT_POINTERS) {
        Ok(bc) => bc,
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            return;
        }
        Err(e) => panic!("stored slot pointers: {e:?}"),
    };
    let profiles = identity_profiles();
    assert!(
        !profiles.is_empty(),
        "no profile the identity domains derive"
    );
    for (stem, p) in profiles {
        compiled(&bc, ("sp_into_slot", true), &stem, &p);
        for entry in ["sp_through_loaded", "sp_open_cell"] {
            match compile(&bc, entry, &p) {
                Err(Error::Unsupported(u)) => assert_eq!(
                    (u.feature, u.needs),
                    (
                        Feature::Lowering(LowerFamily::Access),
                        Needs::NotImplemented
                    ),
                    "{stem}: {entry}"
                ),
                other => panic!("{stem}: {entry}: {:?}", other.map(|_| ())),
            }
        }
    }
}
