//! The old engine's must-fallback specifics that no other engine test pins.
//!
//! `tests/must_fallback.rs` pinned the old engine's clean refusals until 0.2.0 deleted both: an
//! unsupported input gives an
//! `Err`, never valid-looking SPIR-V that computes something else. Most of its classes are pinned
//! for the engine elsewhere (`src/tests/engine_refused_families.rs`, the truncation tests of
//! `air::bitstream` and `air::container`, the ABI strictness tests of `air::abi`, and the
//! function-constant and function-table boundaries of `air::canon`), or the engine lowers them
//! (execution masks, texture atomics, function tables, intersection function buffers). Three were
//! pinned nowhere:
//!
//! - **A module with no function definitions.** It is a library with no functions, and asking it
//!   for one is a typed refusal, not an empty module.
//! - **A byte-view vector store into a word block** was the old engine's GAP, pinned because its
//!   two tempting lowerings are wrong: a non-atomic read-modify-write of the surrounding words
//!   races another thread writing the neighbouring bytes, and a plain word store is byte-exact
//!   only at a four-byte-aligned offset. The engine's leaf model (`memory::leaf`) decides the
//!   path from the access's own alignment and writes a partial word's bytes with two relaxed
//!   atomics, so the gap is closed, and this pins the closed form: the old input's shape, made
//!   well-formed, stores exactly its bytes on the CPU interpreter from eight threads writing
//!   neighbouring 12-byte vectors, aligned and unaligned, in both scheduling orders.
//! - **A kernel whose only write is into an imageblock** was refused because the old engine had no
//!   attachment to resolve the imageblock into, so a tile clear kernel compiled to a module that
//!   cleared nothing. The engine compiles a tile function as a tile pipeline whose staging
//!   stores the cells to the attachments (`src/tests/engine_tile.rs`'s
//!   `every_tile_stages_its_pixels`), and a compute kernel's imageblock is threadgroup staging
//!   whose only output is the slices it writes, as in Metal. There is no refusal left to pin.
//!
//! `llvm-as` assembles the inputs; without it the tests skip visibly.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::device::profile::DeviceProfile;
use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::error::{DescriptorError, Error};
use crate::library::Library;
use crate::lower::select::Plan;
use crate::pipeline::interface::PushLoc;
use crate::pipeline::sym::{Axis, PushField, ResourceKind, ResourceSlot, SpecKey, Stage};
use crate::pipeline::testing::{
    compile, link_and_validate, llvm_assemble, typed_module, AssembleError,
};
use crate::spirv::interp::{run, Buffer, Config, Order, Resources};

const HEADER: &str = r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"
"#;

const TRAILER: &str = r#"!air.version = !{!93}
!air.language_version = !{!94}
!93 = !{i32 2, i32 8, i32 0}
!94 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

/// `llvm-as` of `text`, through a scratch file that is removed at once. `None` when `llvm-as` is
/// not installed (the reason is printed).
fn assemble(text: &str) -> Option<Vec<u8>> {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let path = std::env::temp_dir().join(format!(
        "m2v-must-fallback-{}-{}.ll",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::write(&path, text).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let assembled = llvm_assemble(&path);
    std::fs::remove_file(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    match assembled {
        Ok(bitcode) => Some(bitcode),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejects an input:\n{why}\n{text}"),
    }
}

fn device(stem: &str) -> Device {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles")
        .join(format!("{stem}.json"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let description = DeviceDescription::from_khronos_json(&text, None).expect(stem);
    Device::from_queries(description, DevicePolicy::RESIDENT_TABLES).expect(stem)
}

/// A module that defines no function is a library of none, and asking it for one is refused as
/// a missing entry, by index and by name.
#[test]
fn a_module_with_no_function_definitions_is_an_empty_library() {
    let Some(bitcode) = assemble(&format!("{HEADER}\n{TRAILER}")) else {
        return;
    };
    let library = Library::from_air(&bitcode).expect("a module without functions is a library");
    assert_eq!(library.functions().count(), 0);
    for refusal in [library.function_at(0).err(), library.function("k").err()] {
        match refusal {
            Some(Error::InvalidDescriptor(DescriptorError::EntryNotFound {
                available, ..
            })) => {
                assert_eq!(available, 0)
            }
            other => panic!("a function of an empty library must be EntryNotFound: {other:?}"),
        }
    }
}

/// The old engine's reduced input (a `device uchar*` view of a buffer the kernel also reads as
/// `uint`, the pointer kept in a private struct and reached across a call), made well-formed: the
/// kernel's arguments match its signature, and each thread stores its own `<3 x float>` at byte
/// offset `BASE + 12 * gid` (`ALIGN` is the alignment that offset has). The vectors start past a
/// 16-byte header, whose first word the kernel loads and nothing writes.
const BYTE_VIEW: &str = r#"%struct.view = type { ptr addrspace(2), ptr addrspace(1) }

define void @k(ptr addrspace(1) noundef %data, i32 noundef %gid) {
entry:
  %view = alloca %struct.view, align 8
  %slot = getelementptr inbounds %struct.view, ptr %view, i64 0, i32 1
  store ptr addrspace(1) %data, ptr %slot, align 8
  %first = load i32, ptr addrspace(1) %data, align 4
  call fastcc void @store_vec(ptr %view, i32 %gid, i32 %first)
  ret void
}

define internal fastcc void @store_vec(ptr %view, i32 %gid, i32 %first) {
entry:
  br label %reach

reach:
  %slot = getelementptr inbounds %struct.view, ptr %view, i64 0, i32 1
  %data = load ptr addrspace(1), ptr %slot, align 8
  %lane = mul i32 %gid, 12
  %at = add i32 %lane, BASE
  %offset = zext i32 %at to i64
  %dst = getelementptr inbounds i8, ptr addrspace(1) %data, i64 %offset
  %x = uitofp i32 %gid to float
  %y = fadd float %x, 5.000000e-01
  %z = fsub float -1.000000e+00, %x
  %v0 = insertelement <3 x float> undef, float %x, i32 0
  %v1 = insertelement <3 x float> %v0, float %y, i32 1
  %v2 = insertelement <3 x float> %v1, float %z, i32 2
  store <3 x float> %v2, ptr addrspace(1) %dst, align ALIGN
  br label %done

done:
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"data"}
!4 = !{i32 1, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}
"#;

/// Threads in the one threadgroup, each storing one vector.
const THREADS: u32 = 8;

/// The bytes of thread `gid`'s vector.
fn vector_bytes(gid: u32) -> Vec<u8> {
    let x = gid as f32;
    [x, x + 0.5, -1.0 - x]
        .iter()
        .flat_map(|lane| lane.to_le_bytes())
        .collect()
}

/// The resources of one dispatch of `THREADS` threads over `bytes` at `[[buffer(0)]]`, bound at
/// its start.
fn resources(plan: &Plan, bytes: Vec<u8>) -> Resources {
    let mut spec = BTreeMap::new();
    spec.insert(SpecKey::LocalSize(Axis::X), u64::from(THREADS));
    spec.insert(SpecKey::LocalSize(Axis::Y), 1);
    spec.insert(SpecKey::LocalSize(Axis::Z), 1);
    spec.insert(SpecKey::FlatLocalSize, u64::from(THREADS));
    let mut push = vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)];
    for field in &plan.push.fields {
        match (field.field, field.loc) {
            (PushField::SlotBindOffset(_), PushLoc::Push { offset }) => {
                push[offset as usize..offset as usize + 4].fill(0)
            }
            // A single-region dispatch: the region is unused.
            (PushField::DispatchRegion, PushLoc::Push { .. }) => {}
            other => panic!("unexpected push field {other:?}"),
        }
    }
    Resources {
        buffers: vec![Buffer {
            slot: Some(ResourceSlot {
                stage: Stage::Kernel,
                kind: ResourceKind::Buffer,
                index: 0,
            }),
            address: None,
            bytes,
        }],
        push,
        spec,
    }
}

/// Profiles that bind buffers, without 8- and 16-bit storage and with them. On each, the aligned
/// store is three plain word stores, and the unaligned one writes its partial first and last words
/// with the two atomics each.
const PROFILES: [&str; 2] = ["runtime-floor", "moltenvk-m3"];

/// The old engine refused this rather than write it wrongly; the engine lowers it, and every
/// thread's twelve bytes land exactly while every byte around them keeps its value.
#[test]
fn a_byte_view_vector_store_into_a_word_block_is_exact() {
    for (base, align) in [(16u32, 4u32), (17, 1)] {
        let text = BYTE_VIEW
            .replace("BASE", &base.to_string())
            .replace("ALIGN", &align.to_string());
        let Some(bitcode) = assemble(&format!("{HEADER}\n{text}\n{TRAILER}")) else {
            return;
        };
        let len = (base + 12 * THREADS + 15).next_multiple_of(16) as usize;
        let before: Vec<u8> = (0..len).map(|i| 0xA0 ^ (i as u8)).collect();
        let mut want = before.clone();
        for gid in 0..THREADS {
            let at = (base + 12 * gid) as usize;
            want[at..at + 12].copy_from_slice(&vector_bytes(gid));
        }
        for stem in PROFILES {
            let device = device(stem);
            let profile: &DeviceProfile = device.profile();
            let what = format!("{stem}, offset {base} + 12 * gid, align {align}");
            let artifact =
                compile(&bitcode, "k", profile).unwrap_or_else(|e| panic!("{what}: compile: {e}"));
            link_and_validate(&artifact, profile).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            let (plan, module) =
                typed_module(&bitcode, "k", profile).unwrap_or_else(|e| panic!("{what}: {e}"));
            for order in [Order::Forward, Order::Reverse] {
                let config = Config {
                    order,
                    ..Config::default()
                };
                let out = run(&module, &config, resources(&plan, before.clone()))
                    .unwrap_or_else(|e| panic!("{what}, {order:?}: run: {e:?}"));
                assert_eq!(out.buffers[0], want, "{what}, {order:?}");
            }
        }
    }
}
