//! The access family end to end: descriptor views, word-composed leaves, sub-word stores, and both
//! containment encodings.
//!
//! - The end-of-buffer fixtures (`engine/buffer_end_*`) compile on every tested profile, pass
//!   L3, and compute byte-exact results on the CPU interpreter, including the untouched bytes of a
//!   sub-word store and the padding, with a zero and a non-zero bind residual.
//! - Generated kernels load and store every leaf path (aligned words, one word, byte by byte) at
//!   offsets around the end of the buffer and at `0xFFFF_FFF0..=0xFFFF_FFFF`: with the in-shader
//!   encoding no access traps, an in-bounds access is exact (a `half` NaN loaded a NaN where the
//!   module reads a `half` by the device's widening, `PrecisionNote::NativeNanEncoding`), and an
//!   out-of-bounds load is 0 and an out-of-bounds store changes nothing.
//! - Slot forms: the `constant` fixture gets a uniform view (`u32[]` with standard layout,
//!   `uvec4[]` without) with robust descriptors and a storage view in shader, where an allocation
//!   shorter than the window is contained; several slots are storage views decorated `Aliased`, a
//!   pointer selected between two slots is dispatched by root, and a `Whole` window declares
//!   `BufferRangeLimit`.
//! - Device addresses (Profile A): a `BufferRangeLimit` variant and the per-stage
//!   descriptor-limit fallback reach slots through the root table, with every leaf path exact in
//!   bounds and contained (the sink) out of bounds; Profile F refuses both. A window that starts
//!   below the bind offset and reaches 2³¹ or more above it is exact on both sides; a `Psb`
//!   window longer than 2³² is refused.
//! - Constant GEP indices whose products exceed 128 bits wrap to the offset's low word.
//!
//! `llvm-as` assembles the fixtures; without it the tests skip visibly.

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::air;
use crate::budget::SPIRV_WORDS;
use crate::device::profile::{BufferAccess, DeviceProfile};
use crate::device::requirements::VkFeature;
use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::error::{Error, Feature, LowerFamily, Needs, Unsupported};
use crate::library::FunctionConstantValues;
use crate::lower::emit::emit_module;
use crate::lower::select::{select, LowerInputs, Plan};
use crate::memory::static_root::analyze_memory;
use crate::pipeline::artifact::PrecisionNote;
use crate::pipeline::compile::{compile_bitcode, plan_bitcode, EntryRef};
use crate::pipeline::facts::{RuntimeFact, UniformViewChoice};
use crate::pipeline::interface::{BufferForm, DescriptorView, PushLoc, SlotInterface, Window};
use crate::pipeline::sym::{
    PushField, ResourceKind, ResourceSlot, SlotClass, SlotRef, SpecKey, Stage, Symbol,
};
use crate::pipeline::testing::{
    compile, engine_dir, link_and_validate, llvm_assemble, meter, typed_module, AssembleError,
};
use crate::spirv::interp::{run, Buffer, Config, InterpError, Resources};
use crate::spirv::ir::{GlobalKind, TypedModule};
use crate::spirv::verify::verify;

/// Profiles with and without 64-bit integers and 8/16-bit storage.
const PROFILES: [&str; 4] = [
    "moltenvk-m3",
    "lunarg-desktop-baseline-2023",
    "vk12-floor",
    "lavapipe",
];

fn policy() -> DevicePolicy {
    DevicePolicy::default()
}

/// Committed profile `stem`, optionally with buffer robustness disabled (the in-shader encoding)
/// and optionally without `uniformBufferStandardLayout`.
fn profile_with(stem: &str, robust: bool, standard_layout: bool) -> DeviceProfile {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles")
        .join(format!("{stem}.json"));
    let text = std::fs::read_to_string(path).expect("profile");
    let mut desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    if !robust {
        desc.features.remove(&VkFeature::RobustBufferAccess);
        desc.features.remove(&VkFeature::RobustBufferAccess2);
    }
    if !standard_layout {
        desc.features
            .remove(&VkFeature::UniformBufferStandardLayout);
    }
    Device::new(desc, policy())
        .expect("device")
        .profile()
        .clone()
}

fn assemble_file(name: &str) -> Option<Vec<u8>> {
    match llvm_assemble(&engine_dir().join(format!("{name}.ll"))) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {name}.ll: {why}"),
    }
}

fn assemble_text(name: &str, text: &str) -> Option<Vec<u8>> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "m2v-access-{}-{}-{name}",
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

fn kslot(index: u32) -> ResourceSlot {
    ResourceSlot {
        stage: Stage::Kernel,
        kind: ResourceKind::Buffer,
        index,
    }
}

fn sref(index: u32) -> SlotRef {
    SlotRef {
        stage: Stage::Kernel,
        class: SlotClass::Buffer,
        index,
    }
}

/// The push bytes with each slot's `SlotBindOffset` residual set.
fn push_bytes(plan: &Plan, residuals: &[(u32, u32)]) -> Vec<u8> {
    let mut out = vec![0u8; 128];
    for f in &plan.push.fields {
        let (PushField::SlotBindOffset(s), PushLoc::Push { offset }) = (f.field, f.loc) else {
            continue;
        };
        let r = residuals
            .iter()
            .find(|(i, _)| *i == s.index)
            .map_or(0, |(_, r)| *r);
        let at = offset as usize;
        out[at..at + 4].copy_from_slice(&r.to_le_bytes());
    }
    out
}

/// A buffer bound at residual `r` over `data` padded to a multiple of 16 with `pad` bytes.
fn bound(index: u32, r: u32, data: &[u8], pad: u8) -> Buffer {
    let mut bytes = vec![0xEEu8; r as usize];
    bytes.extend_from_slice(data);
    while !bytes.len().is_multiple_of(16) {
        bytes.push(pad);
    }
    Buffer {
        slot: Some(kslot(index)),
        address: None,
        bytes,
    }
}

fn check_l3(bitcode: &[u8], entry: &str, p: &DeviceProfile, what: &str) {
    let a = compile(bitcode, entry, p).unwrap_or_else(|e| panic!("{what}: compile: {e}"));
    link_and_validate(&a, p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
}

/// Every tested profile in both encodings.
fn encodings() -> Vec<(String, DeviceProfile)> {
    let mut out = Vec::new();
    for stem in PROFILES {
        for robust in [true, false] {
            out.push((
                format!("{stem} robust={robust}"),
                profile_with(stem, robust, true),
            ));
        }
    }
    out
}

/// The last byte of a 6-byte `device` buffer through the word view, with residuals 0 and 8.
#[test]
fn the_last_byte_of_a_six_byte_buffer_loads_exactly() {
    let Some(bc) = assemble_file("buffer_end_byte_load") else {
        return;
    };
    let data = [1u8, 2, 3, 4, 5, 0x9C];
    for (what, p) in encodings() {
        check_l3(&bc, "buffer_end_byte_load", &p, &what);
        let (plan, m) = typed_module(&bc, "buffer_end_byte_load", &p).expect("module");
        for r in [0u32, 8] {
            let res = Resources {
                buffers: vec![bound(0, r, &data, 0x77), bound(1, 0, &[0; 4], 0)],
                push: push_bytes(&plan, &[(0, r)]),
                spec: Default::default(),
            };
            let out = run(&m, &Config::default(), res).expect("run");
            assert_eq!(out.buffers[1][..4], [0x9C, 0, 0, 0], "{what} r={r}");
        }
    }
}

/// The last half (bytes 8..9) of a 10-byte buffer.
#[test]
fn the_last_half_of_a_ten_byte_buffer_loads_exactly() {
    let Some(bc) = assemble_file("buffer_end_half_load") else {
        return;
    };
    let data = [0u8, 1, 2, 3, 4, 5, 6, 7, 0x34, 0x92];
    for (what, p) in encodings() {
        check_l3(&bc, "buffer_end_half_load", &p, &what);
        let (plan, m) = typed_module(&bc, "buffer_end_half_load", &p).expect("module");
        for r in [0u32, 4] {
            let res = Resources {
                buffers: vec![bound(0, r, &data, 0x55), bound(1, 0, &[0; 4], 0)],
                push: push_bytes(&plan, &[(0, r)]),
                spec: Default::default(),
            };
            let out = run(&m, &Config::default(), res).expect("run");
            assert_eq!(out.buffers[1][..4], [0x34, 0x92, 0, 0], "{what} r={r}");
        }
    }
}

/// A byte store at the end of a 6-byte buffer changes only that byte; its neighbours and the
/// padding are preserved.
#[test]
fn a_byte_store_at_the_end_preserves_its_neighbours() {
    let Some(bc) = assemble_file("buffer_end_byte_store") else {
        return;
    };
    let data = [0x10u8, 0x11, 0x12, 0x13, 0x14, 0x15];
    for (what, p) in encodings() {
        check_l3(&bc, "buffer_end_byte_store", &p, &what);
        let (plan, m) = typed_module(&bc, "buffer_end_byte_store", &p).expect("module");
        for r in [0u32, 4, 12] {
            let before = bound(0, r, &data, 0x66);
            let mut want = before.bytes.clone();
            want[r as usize + 5] = 0xA5;
            let res = Resources {
                buffers: vec![before],
                push: push_bytes(&plan, &[(0, r)]),
                spec: Default::default(),
            };
            let out = run(&m, &Config::default(), res).expect("run");
            assert_eq!(out.buffers[0], want, "{what} r={r}");
        }
    }
}

/// The `constant` fixture's slot is a uniform view (`u32[]` with `uniformBufferStandardLayout`,
/// `uvec4[]` without) and reads the last byte exactly.
#[test]
fn a_bounded_constant_buffer_is_a_uniform_view() {
    let Some(bc) = assemble_file("buffer_end_byte_load_constant") else {
        return;
    };
    let data = [9u8, 8, 7, 6, 5, 0xC3];
    for stem in PROFILES {
        for standard in [true, false] {
            // Robust descriptors only: the in-shader encoding uses a storage view.
            let what = format!("{stem} standard={standard}");
            let p = profile_with(stem, true, standard);
            let a = compile(&bc, "buffer_end_byte_load_constant", &p)
                .unwrap_or_else(|e| panic!("{what}: {e}"));
            let form = match a.interface.slots.get(&sref(0)) {
                Some(SlotInterface::Buffer(b)) => b.form,
                other => panic!("{what}: {other:?}"),
            };
            let want = if standard {
                DescriptorView::UniformWords
            } else {
                DescriptorView::UniformVec4
            };
            assert_eq!(form, BufferForm::Descriptor(want), "{what}");
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            let (plan, m) = typed_module(&bc, "buffer_end_byte_load_constant", &p).expect("module");
            for r in [0u32, 4] {
                // The uniform view spans `uniform_range(1, align)` bytes. The interpreter models
                // no robustness, so this in-bounds run binds that much.
                let mut ubo = bound(0, r, &data, 0x44);
                ubo.bytes.resize(ubo.bytes.len().max(512), 0x44);
                let res = Resources {
                    buffers: vec![ubo, bound(1, 0, &[0; 4], 0)],
                    push: push_bytes(&plan, &[(0, r)]),
                    spec: Default::default(),
                };
                let out = run(&m, &Config::default(), res).expect("run");
                assert_eq!(out.buffers[1][..4], [0xC3, 0, 0, 0], "{what} r={r}");
            }
        }
    }
}

/// A uniform view's slot declares its `RangeAtBindOffset` specialization: where the binder states
/// the range starts at the bind offset, the read takes the Metal offset alone, whatever the
/// residual pushed (an alignment-checked binder pushes 0 there); elsewhere the residual plus the
/// offset, as before. With `uniformBufferStandardLayout` (`u32[]`) and without (`uvec4[]`).
#[test]
fn a_uniform_view_read_at_its_bind_offset_drops_the_residual() {
    let Some(bc) = assemble_file("buffer_end_byte_load_constant") else {
        return;
    };
    let data = [9u8, 8, 7, 6, 5, 0xC3];
    let key = SpecKey::RangeAtBindOffset(sref(0));
    for stem in PROFILES {
        for standard in [true, false] {
            let what = format!("{stem} standard={standard}");
            let p = profile_with(stem, true, standard);
            let a = compile(&bc, "buffer_end_byte_load_constant", &p)
                .unwrap_or_else(|e| panic!("{what}: {e}"));
            assert!(a.interface.read_keys().contains(&key), "{what}");
            let (plan, m) = typed_module(&bc, "buffer_end_byte_load_constant", &p).expect("module");
            // The data at byte 0, a residual of 8 pushed: byte 5 when the range starts at the bind
            // offset, byte 13 (the padding) when it does not.
            for (at_bind, want) in [(1u64, 0xC3u8), (0, 0x44)] {
                let mut ubo = bound(0, 0, &data, 0x44);
                ubo.bytes.resize(512, 0x44);
                let res = Resources {
                    buffers: vec![ubo, bound(1, 0, &[0; 4], 0)],
                    push: push_bytes(&plan, &[(0, 8)]),
                    spec: [(key, at_bind)].into(),
                };
                let out = run(&m, &Config::default(), res).expect("run");
                assert_eq!(
                    out.buffers[1][..4],
                    [want, 0, 0, 0],
                    "{what} at_bind={at_bind}"
                );
            }
        }
    }
}

/// A `constant` struct of 6 bytes whose first and last bytes are read: its window starts at 0.
const CONSTANT_ENDS: &str = r#"define void @constant_ends(ptr addrspace(2) noundef "air-buffer-no-alias" %0, ptr addrspace(1) noundef "air-buffer-no-alias" %1) {
entry:
  %a = load i8, ptr addrspace(2) %0, align 1
  %p = getelementptr inbounds i8, ptr addrspace(2) %0, i64 5
  %b = load i8, ptr addrspace(2) %p, align 1
  %x = zext i8 %a to i32
  %y = zext i8 %b to i32
  %s = shl i32 %y, 8
  %v = or i32 %x, %s
  store i32 %v, ptr addrspace(1) %1, align 4
  ret void
}
!air.kernel = !{!9}
!air.version = !{!17}
!air.language_version = !{!18}
!9 = !{ptr @constant_ends, !10, !11}
!10 = !{}
!11 = !{!12, !13}
!12 = !{i32 0, !"air.buffer", !"air.buffer_size", i32 6, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 6, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"Six", !"air.arg_name", !"in"}
!13 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!17 = !{i32 2, i32 8, i32 0}
!18 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

/// A uniform view is sized for a descriptor base aligned to `max(minUniformBufferOffsetAlignment,
/// 16)`. A window starting at byte 0 is bound from the multiple of that alignment at or below the
/// bind offset, so its residual is any value below it: on `amd-rdna3-radv` (an alignment of 4)
/// residuals 12 and 15 put the struct's last byte in words 4 and 5, past the 16 bytes an alignment
/// of 4 alone would declare. The view covers them and reads both ends exactly, with
/// `uniformBufferStandardLayout` (`u32[]`) and without (`uvec4[]`).
#[test]
fn a_uniform_view_covers_every_residual_of_a_sixteen_aligned_base() {
    let Some(bc) = assemble_text("constant_ends", CONSTANT_ENDS) else {
        return;
    };
    let data = [0x5Au8, 8, 7, 6, 5, 0xC3];
    for stem in ["amd-rdna3-radv", "moltenvk-m3"] {
        for standard in [true, false] {
            let what = format!("{stem} standard={standard}");
            let p = profile_with(stem, true, standard);
            let a = compile(&bc, "constant_ends", &p).unwrap_or_else(|e| panic!("{what}: {e}"));
            let form = match a.interface.slots.get(&sref(0)) {
                Some(SlotInterface::Buffer(b)) => b.form,
                other => panic!("{what}: {other:?}"),
            };
            let want = if standard {
                DescriptorView::UniformWords
            } else {
                DescriptorView::UniformVec4
            };
            assert_eq!(form, BufferForm::Descriptor(want), "{what}");
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            let (plan, m) = typed_module(&bc, "constant_ends", &p).expect("module");
            for r in [0u32, 4, 12, 15] {
                let res = Resources {
                    buffers: vec![bound(0, r, &data, 0x44), bound(1, 0, &[0; 4], 0)],
                    push: push_bytes(&plan, &[(0, r)]),
                    spec: Default::default(),
                };
                let out = run(&m, &Config::default(), res).expect("run");
                assert_eq!(out.buffers[1][..4], [0x5A, 0xC3, 0, 0], "{what} r={r}");
            }
        }
    }
}

/// Without robust descriptors a `constant` slot is a storage view, contained against the range
/// actually bound. The binder clamps that range to the padded allocation, so it can be shorter than
/// the proven window: here a 16-byte allocation bound at offset 12 leaves 4 of the struct's 6
/// bytes, and the constant-offset read of byte 5 lands past the bound range. It must not trap and
/// must read 0; the same short allocation at offsets 0 and 4 reads byte 5 exactly.
#[test]
fn a_constant_buffer_shorter_than_its_window_is_contained_in_shader() {
    let Some(bc) = assemble_file("buffer_end_byte_load_constant") else {
        return;
    };
    let alloc: [u8; 16] = std::array::from_fn(|i| 0x10 + i as u8);
    for stem in PROFILES {
        for standard in [true, false] {
            let what = format!("{stem} standard={standard}");
            let p = profile_with(stem, false, standard);
            let a = compile(&bc, "buffer_end_byte_load_constant", &p)
                .unwrap_or_else(|e| panic!("{what}: {e}"));
            let form = match a.interface.slots.get(&sref(0)) {
                Some(SlotInterface::Buffer(b)) => b.form,
                other => panic!("{what}: {other:?}"),
            };
            assert_eq!(
                form,
                BufferForm::Descriptor(DescriptorView::Storage),
                "{what}"
            );
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            let (plan, m) = typed_module(&bc, "buffer_end_byte_load_constant", &p).expect("module");
            for (r, want) in [(0u32, alloc[5]), (4, alloc[9]), (12, 0)] {
                let short = Buffer {
                    slot: Some(kslot(0)),
                    address: None,
                    bytes: alloc.to_vec(),
                };
                let res = Resources {
                    buffers: vec![short, bound(1, 0, &[0xAA; 4], 0)],
                    push: push_bytes(&plan, &[(0, r)]),
                    spec: Default::default(),
                };
                let out = run(&m, &Config::default(), res)
                    .unwrap_or_else(|e| panic!("{what} r={r}: {e:?}"));
                assert_eq!(out.buffers[1][..4], [want, 0, 0, 0], "{what} r={r}");
            }
        }
    }
}

const HEADER: &str = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n";

/// The metadata of a kernel `name` with buffers `data` (0, read-write), `out` (1), and `params`
/// (2, read) — all `device`, element size 1.
fn meta(name: &str) -> String {
    format!(
        "!air.kernel = !{{!9}}\n!air.version = !{{!17}}\n!air.language_version = !{{!18}}\n\
         !9 = !{{ptr @{name}, !10, !11}}\n!10 = !{{}}\n!11 = !{{!12, !13, !14}}\n\
         !12 = !{{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 1, !\"air.arg_type_align_size\", i32 1, !\"air.arg_type_name\", !\"uchar\", !\"air.arg_name\", !\"data\"}}\n\
         !13 = !{{i32 1, !\"air.buffer\", !\"air.location_index\", i32 1, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 1, !\"air.arg_type_align_size\", i32 1, !\"air.arg_type_name\", !\"uchar\", !\"air.arg_name\", !\"out\"}}\n\
         !14 = !{{i32 2, !\"air.buffer\", !\"air.location_index\", i32 2, i32 1, !\"air.read\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"params\"}}\n\
         !17 = !{{i32 2, i32 8, i32 0}}\n!18 = !{{!\"Metal\", i32 4, i32 0, i32 0}}\n"
    )
}

/// One generated leaf kernel: load `ty` from `data + params[0]` (align `align`) into `out`, or
/// store `value` there.
struct LeafKernel {
    name: &'static str,
    ty: &'static str,
    bytes: u32,
    align: u32,
    /// The stored constant and its little-endian bytes, for a store kernel.
    store: Option<(&'static str, Vec<u8>)>,
}

fn leaf_kernel(k: &LeafKernel) -> String {
    let name = k.name;
    let ty = k.ty;
    let a = k.align;
    let body = match &k.store {
        None => format!(
            "  %v = load {ty}, ptr addrspace(1) %p, align {a}\n  store {ty} %v, ptr addrspace(1) %1, align 16\n"
        ),
        Some((c, _)) => format!("  store {ty} {c}, ptr addrspace(1) %p, align {a}\n"),
    };
    format!(
        "{HEADER}define void @{name}(ptr addrspace(1) noundef \"air-buffer-no-alias\" %0, ptr addrspace(1) noundef \"air-buffer-no-alias\" %1, ptr addrspace(1) noundef \"air-buffer-no-alias\" %2) {{\nentry:\n  %o = load i32, ptr addrspace(1) %2, align 4\n  %p = getelementptr i8, ptr addrspace(1) %0, i32 %o\n{body}  ret void\n}}\n\n{}",
        meta(name)
    )
}

fn leaf_kernels() -> Vec<LeafKernel> {
    let k = |name, ty, bytes, align| LeafKernel {
        name,
        ty,
        bytes,
        align,
        store: None,
    };
    let s = |name, ty, bytes, align, c, b: &[u8]| LeafKernel {
        name,
        ty,
        bytes,
        align,
        store: Some((c, b.to_vec())),
    };
    vec![
        k("load_i8", "i8", 1, 1),
        k("load_i16_aligned", "i16", 2, 2),
        k("load_i16_bytes", "i16", 2, 1),
        k("load_i32_aligned", "i32", 4, 4),
        k("load_i32_bytes", "i32", 4, 1),
        k("load_float3", "<3 x float>", 12, 16),
        k("load_uint4", "<4 x i32>", 16, 16),
        k("load_uchar4_bytes", "<4 x i8>", 4, 1),
        k("load_short2", "<2 x i16>", 4, 4),
        // Narrow floats are held widened; their leaves round-trip the 16 format bits, a `half`
        // NaN's as the device's widening encodes it where the module reads a `half` so.
        k("load_half", "half", 2, 2),
        k("load_half_bytes", "half", 2, 1),
        k("load_half3", "<3 x half>", 6, 8),
        k("load_bfloat", "bfloat", 2, 2),
        k("load_bfloat2", "<2 x bfloat>", 4, 4),
        // Vector leaves convert their lanes as one vector; under-aligned leaves join the words they
        // touch at the offset's shift, reading the word after them only when the leaf reaches into
        // it.
        k("load_i32_half_aligned", "i32", 4, 2),
        k("load_float2_bytes", "<2 x float>", 8, 1),
        k("load_short3_half_aligned", "<3 x i16>", 6, 2),
        k("load_half4", "<4 x half>", 8, 8),
        k("load_half2_bytes", "<2 x half>", 4, 1),
        k("load_uchar4", "<4 x i8>", 4, 4),
        k("load_uchar3_bytes", "<3 x i8>", 3, 1),
        k("load_float3_word_aligned", "<3 x float>", 12, 4),
        k("load_int4_bytes", "<4 x i32>", 16, 1),
        // A vector of more than four lanes is one leaf of chunked register value.
        k("load_int6", "<6 x i32>", 24, 32),
        k("load_int6_word_aligned", "<6 x i32>", 24, 4),
        k("load_half5_bytes", "<5 x half>", 10, 1),
        k("load_uchar8", "<8 x i8>", 8, 8),
        // A 16-byte leaf aligned to 16 is one `uvec4` of the slot's `uvec4` view: one bitcast for
        // four 32-bit or two 64-bit lanes (two words each without `shaderInt64`), the vector's
        // words for a chunked one; aligned to 4 it moves word by word.
        k("load_float4", "<4 x float>", 16, 16),
        k("load_long2", "<2 x i64>", 16, 16),
        k("load_half8", "<8 x half>", 16, 16),
        k("load_short8", "<8 x i16>", 16, 16),
        k("load_uchar16", "<16 x i8>", 16, 32),
        k("load_uint4_word_aligned", "<4 x i32>", 16, 4),
        s("store_i8", "i8", 1, 1, "-91", &[0xA5]),
        s("store_i16_aligned", "i16", 2, 2, "-4403", &[0xCD, 0xEE]),
        s("store_i16_bytes", "i16", 2, 1, "-4403", &[0xCD, 0xEE]),
        s(
            "store_i32_bytes",
            "i32",
            4,
            1,
            "305419896",
            &0x1234_5678u32.to_le_bytes(),
        ),
        s(
            "store_i32_aligned",
            "i32",
            4,
            4,
            "305419896",
            &0x1234_5678u32.to_le_bytes(),
        ),
        s(
            "store_uint4",
            "<4 x i32>",
            16,
            16,
            "<i32 1, i32 2, i32 3, i32 4>",
            &[1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0],
        ),
        s("store_half", "half", 2, 2, "0xHBC01", &[0x01, 0xBC]),
        s("store_half_bytes", "half", 2, 1, "0xH7E01", &[0x01, 0x7E]),
        s("store_bfloat", "bfloat", 2, 2, "0xRC0A1", &[0xA1, 0xC0]),
        s(
            "store_uchar3",
            "<3 x i8>",
            3,
            4,
            "<i8 7, i8 8, i8 9>",
            &[7, 8, 9],
        ),
        s(
            "store_half3",
            "<3 x half>",
            6,
            8,
            "<half 0xH3C00, half 0xHBC01, half 0xH7E01>",
            &[0x00, 0x3C, 0x01, 0xBC, 0x01, 0x7E],
        ),
        s(
            "store_i32_half_aligned",
            "i32",
            4,
            2,
            "305419896",
            &0x1234_5678u32.to_le_bytes(),
        ),
        s(
            "store_float2_bytes",
            "<2 x float>",
            8,
            1,
            "<float 1.5, float -2.0>",
            &[0, 0, 0xC0, 0x3F, 0, 0, 0, 0xC0],
        ),
        s(
            "store_short3_half_aligned",
            "<3 x i16>",
            6,
            2,
            "<i16 258, i16 -2, i16 772>",
            &[0x02, 0x01, 0xFE, 0xFF, 0x04, 0x03],
        ),
        s(
            "store_half4",
            "<4 x half>",
            8,
            8,
            "<half 0xH3C00, half 0xHBC01, half 0xH7E01, half 0xH0001>",
            &[0x00, 0x3C, 0x01, 0xBC, 0x01, 0x7E, 0x01, 0x00],
        ),
        s(
            "store_half2_bytes",
            "<2 x half>",
            4,
            1,
            "<half 0xH3C00, half 0xHBC01>",
            &[0x00, 0x3C, 0x01, 0xBC],
        ),
        s(
            "store_uchar4",
            "<4 x i8>",
            4,
            4,
            "<i8 1, i8 -2, i8 3, i8 -4>",
            &[1, 0xFE, 3, 0xFC],
        ),
        s(
            "store_int4_bytes",
            "<4 x i32>",
            16,
            1,
            "<i32 1, i32 -2, i32 305419896, i32 4>",
            &[
                1, 0, 0, 0, 0xFE, 0xFF, 0xFF, 0xFF, 0x78, 0x56, 0x34, 0x12, 4, 0, 0, 0,
            ],
        ),
        s(
            "store_int6",
            "<6 x i32>",
            24,
            8,
            "<i32 1, i32 -2, i32 3, i32 305419896, i32 5, i32 -6>",
            &[
                1, 0, 0, 0, 0xFE, 0xFF, 0xFF, 0xFF, 3, 0, 0, 0, 0x78, 0x56, 0x34, 0x12, 5, 0, 0, 0,
                0xFA, 0xFF, 0xFF, 0xFF,
            ],
        ),
        s(
            "store_float4",
            "<4 x float>",
            16,
            16,
            "<float 1.5, float -2.0, float 3.25, float 0.0>",
            &[
                0, 0, 0xC0, 0x3F, 0, 0, 0, 0xC0, 0, 0, 0x50, 0x40, 0, 0, 0, 0,
            ],
        ),
        s(
            "store_long2",
            "<2 x i64>",
            16,
            16,
            "<i64 81985529216486895, i64 -2>",
            &[
                0xEF, 0xCD, 0xAB, 0x89, 0x67, 0x45, 0x23, 0x01, 0xFE, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
                0xFF, 0xFF,
            ],
        ),
        s(
            "store_half8",
            "<8 x half>",
            16,
            16,
            "<half 0xH3C00, half 0xHBC01, half 0xH7E01, half 0xH0001, half 0xH8400, half 0xH7BFF, half 0xHFC00, half 0xH0000>",
            &[
                0x00, 0x3C, 0x01, 0xBC, 0x01, 0x7E, 0x01, 0x00, 0x00, 0x84, 0xFF, 0x7B, 0x00, 0xFC,
                0x00, 0x00,
            ],
        ),
        s(
            "store_uchar16",
            "<16 x i8>",
            16,
            16,
            "<i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15, i8 -16>",
            &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0xF0],
        ),
        s(
            "store_uint4_word_aligned",
            "<4 x i32>",
            16,
            4,
            "<i32 1, i32 2, i32 3, i32 4>",
            &[1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0],
        ),
        s(
            "store_half5_half_aligned",
            "<5 x half>",
            10,
            2,
            "<half 0xH3C00, half 0xHBC01, half 0xH7E01, half 0xH0001, half 0xH8400>",
            &[0x00, 0x3C, 0x01, 0xBC, 0x01, 0x7E, 0x01, 0x00, 0x00, 0x84],
        ),
    ]
}

fn pattern(n: usize) -> Vec<u8> {
    (0..n)
        .map(|i| (i as u8).wrapping_mul(29).wrapping_add(11))
        .collect()
}

/// The probe offsets for a buffer of `size` bytes (the L0 set).
fn offsets(size: u64) -> Vec<u64> {
    let mut v: Vec<u64> = vec![0, 1, 2, 3];
    for d in 0..=17u64 {
        v.push(size.saturating_sub(d));
        v.push(size + d);
    }
    v.extend(0xFFFF_FFF0u64..=0xFFFF_FFFF);
    v.sort_unstable();
    v.dedup();
    v
}

/// Every leaf path, in-shader encoding, at offsets around the end of a 32-byte buffer and near 2^32
/// (with residuals 0 and 4 wrapping them further): no trap, exact in bounds, 0 or unchanged out of
/// bounds. Probes are skipped where the offset breaks the access's stated alignment (undefined in
/// Metal). A `half` NaN loaded, or stored from a constant, need only come back a NaN where the
/// module reads a `half` from its bits by the device's widening or encodes one by its narrowing
/// (`PrecisionNote::NativeNanEncoding`), whose NaN encoding is the device's.
#[test]
fn every_leaf_path_is_exact_in_bounds_and_contained_out_of_bounds() {
    let size = 32u64;
    for stem in ["moltenvk-m3", "vk12-floor"] {
        let p = profile_with(stem, false, true);
        for k in leaf_kernels() {
            let what = format!("{stem} {}", k.name);
            let Some(bc) = assemble_text(k.name, &leaf_kernel(&k)) else {
                return;
            };
            check_l3(&bc, k.name, &p, &what);
            let native_halves = k.ty.contains("half")
                && compile(&bc, k.name, &p)
                    .unwrap_or_else(|e| panic!("{what}: {e}"))
                    .precision
                    .contains(&PrecisionNote::NativeNanEncoding);
            let (plan, m) = typed_module(&bc, k.name, &p).unwrap_or_else(|e| panic!("{what}: {e}"));
            for r in [0u32, 4] {
                for off in offsets(size) {
                    let o = off as u32;
                    let desc = u64::from(r.wrapping_add(o));
                    if desc % u64::from(k.align) != 0 {
                        continue;
                    }
                    let data = pattern(size as usize);
                    let mut params = vec![0u8; 16];
                    params[..4].copy_from_slice(&o.to_le_bytes());
                    let mut dbuf = bound(0, r, &data, 0);
                    dbuf.bytes.truncate(r as usize + size as usize);
                    let res = Resources {
                        buffers: vec![
                            dbuf.clone(),
                            bound(1, 0, &[0; 32], 0),
                            bound(2, 0, &params, 0),
                        ],
                        push: push_bytes(&plan, &[(0, r)]),
                        spec: Default::default(),
                    };
                    let dsize = dbuf.bytes.len() as u64;
                    let n = u64::from(k.bytes);
                    let inb = dsize >= n && desc <= dsize - n;
                    let out = match run(&m, &Config::default(), res) {
                        Ok(o) => o,
                        Err(InterpError::Trap(t)) => {
                            panic!("{what} r={r} off={off:#x}: trapped {t:?}")
                        }
                        Err(e) => panic!("{what}: {e}"),
                    };
                    let at = desc as usize;
                    match &k.store {
                        None => {
                            let want: Vec<u8> = if inb {
                                dbuf.bytes[at..at + k.bytes as usize].to_vec()
                            } else {
                                vec![0; k.bytes as usize]
                            };
                            let got = &out.buffers[1][..k.bytes as usize];
                            if native_halves {
                                assert!(
                                    same_halves(got, &want),
                                    "{what} r={r} off={off:#x}: {got:?} for {want:?}"
                                );
                            } else {
                                assert_eq!(got, &want[..], "{what} r={r} off={off:#x}");
                            }
                        }
                        Some((_, bytes)) => {
                            let mut want = dbuf.bytes.clone();
                            if inb {
                                want[at..at + bytes.len()].copy_from_slice(bytes);
                            }
                            if native_halves {
                                assert!(
                                    same_halves(&out.buffers[0], &want),
                                    "{what} r={r} off={off:#x}: {:?}",
                                    out.buffers[0]
                                );
                            } else {
                                assert_eq!(out.buffers[0], want, "{what} r={r} off={off:#x}");
                            }
                        }
                    }
                }
            }
        }
    }
}

/// A kernel that stores a word through a pointer selected between two slots: the root dispatch
/// (switch-at-use) writes the chosen slot only, and both views are storage views decorated
/// `Aliased`.
const SELECT_TWO: &str = r#"
define void @select_two(ptr addrspace(1) noundef "air-buffer-no-alias" %0, ptr addrspace(1) noundef "air-buffer-no-alias" %1, ptr addrspace(1) noundef "air-buffer-no-alias" %2) {
entry:
  %c = load i32, ptr addrspace(1) %2, align 4
  %b = icmp eq i32 %c, 0
  %p = select i1 %b, ptr addrspace(1) %0, ptr addrspace(1) %1
  %q = getelementptr i8, ptr addrspace(1) %p, i32 4
  store i32 305419896, ptr addrspace(1) %q, align 4
  %r = getelementptr i8, ptr addrspace(1) %p, i32 9
  %v = load i8, ptr addrspace(1) %r, align 1
  %w = getelementptr i8, ptr addrspace(1) %2, i32 8
  store i8 %v, ptr addrspace(1) %w, align 1
  ret void
}
"#;

#[test]
fn a_pointer_selected_between_two_slots_dispatches_by_root() {
    let text = format!("{HEADER}{SELECT_TWO}\n{}", meta("select_two")).replace(
        "!\"air.read\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4",
        "!\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4",
    );
    let Some(bc) = assemble_text("select_two", &text) else {
        return;
    };
    for (what, p) in encodings() {
        let a = compile(&bc, "select_two", &p).unwrap_or_else(|e| panic!("{what}: {e}"));
        assert!(
            a.strategies
                .entries
                .keys()
                .any(|k| k.starts_with("access.store.switch")),
            "{what}: {:?}",
            a.strategies.entries.keys().collect::<Vec<_>>()
        );
        let (_, words) = link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: {e}"));
        // Three storage slots, each view `OpDecorate %v Aliased` (opcode 71, decoration 20): the
        // word views, and where the profile has 8-bit storage access and `shaderInt8`, each
        // slot's `u8[]` view (each slot moves a byte).
        let leaf = &p.fields().leaf;
        let views = if leaf.storage_buffer_8bit && leaf.int8 {
            6
        } else {
            3
        };
        let mut aliased = 0;
        let mut i = 5;
        while i < words.len() {
            let (op, len) = (words[i] & 0xFFFF, (words[i] >> 16) as usize);
            if op == 71 && words.get(i + 2) == Some(&20) {
                aliased += 1;
            }
            i += len.max(1);
        }
        assert_eq!(aliased, views, "{what}");
        let (plan, m) = typed_module(&bc, "select_two", &p).expect("module");
        for c in [0u32, 1] {
            let mut params = vec![0u8; 16];
            params[..4].copy_from_slice(&c.to_le_bytes());
            let a0 = pattern(16);
            let a1: Vec<u8> = pattern(32).into_iter().skip(16).collect();
            let res = Resources {
                buffers: vec![
                    bound(0, 0, &a0, 0),
                    bound(1, 0, &a1, 0),
                    bound(2, 0, &params, 0),
                ],
                push: push_bytes(&plan, &[]),
                spec: Default::default(),
            };
            let out = run(&m, &Config::default(), res).expect("run");
            let (hit, miss, src) = if c == 0 { (0, 1, &a0) } else { (1, 0, &a1) };
            let mut want = src.clone();
            want[4..8].copy_from_slice(&0x1234_5678u32.to_le_bytes());
            assert_eq!(out.buffers[hit], want, "{what} c={c}");
            let untouched = if c == 0 { &a1 } else { &a0 };
            assert_eq!(&out.buffers[miss], untouched, "{what} c={c}");
            assert_eq!(out.buffers[2][8], src[9], "{what} c={c}");
        }
    }
}

/// A storage slot whose window is `Whole` (a dynamic offset into an unsized pointer) cannot be
/// proven within `maxStorageBufferRange`, so it declares `BufferRangeLimit`; a variant moving it to
/// a device address needs the PSB leaf (not implemented yet) on Profile A and a device address
/// Profile F lacks.
#[test]
fn an_unbounded_storage_slot_declares_its_range_limit() {
    let k = LeafKernel {
        name: "load_i32_aligned",
        ty: "i32",
        bytes: 4,
        align: 4,
        store: None,
    };
    let Some(bc) = assemble_text("range_limit", &leaf_kernel(&k)) else {
        return;
    };
    for stem in PROFILES {
        let p = profile_with(stem, true, true);
        let a = compile(&bc, k.name, &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert!(
            a.facts
                .contains(&RuntimeFact::BufferRangeLimit { slot: sref(0) }),
            "{stem}"
        );
        match a.interface.slots.get(&sref(0)) {
            Some(SlotInterface::Buffer(b)) => {
                assert_eq!(b.window, Window::Whole, "{stem}");
                assert_eq!(b.form, BufferForm::Descriptor(DescriptorView::Storage));
            }
            other => panic!("{stem}: {other:?}"),
        }
        // The constant-offset slots (`out`, `params`) are bounded and declare nothing.
        assert!(
            !a.facts
                .contains(&RuntimeFact::BufferRangeLimit { slot: sref(2) }),
            "{stem}"
        );
        let mut inputs = LowerInputs::new();
        inputs.psb_slots.insert(sref(0));
        let r = plan_bitcode(
            &bc,
            EntryRef::Name(k.name),
            &FunctionConstantValues::defaults(),
            &p,
            &inputs,
            &mut meter(),
        );
        if p.fields().buffer_access == BufferAccess::Addressed {
            // Profile A: the variant reaches the slot by device address.
            let plan = r.unwrap_or_else(|e| panic!("{stem}: {e}"));
            let form = plan
                .slots
                .iter()
                .flatten()
                .find(|c| c.slot == sref(0))
                .map(|c| c.form);
            assert_eq!(form, Some(BufferForm::Psb), "{stem}");
            assert!(plan
                .facts
                .contains(&RuntimeFact::DeviceAddresses { slot: sref(0) }));
        } else {
            // Profile F names what it lacks of Profile A.
            let want = Unsupported::needs_any_of(Feature::BufferRange, vec![p.addressed_needs()]);
            match r {
                Err(Error::Unsupported(u)) => {
                    assert_eq!((u.feature, u.needs), (want.feature, want.needs), "{stem}")
                }
                other => panic!("{stem}: expected a refusal, got {other:?}"),
            }
        }
    }
}

/// The typed module of `entry` under `inputs` (as `pipeline::testing::typed_module`, with
/// lowering inputs).
fn typed_module_with(
    bc: &[u8],
    entry: &str,
    p: &DeviceProfile,
    inputs: &LowerInputs,
) -> (Plan, TypedModule) {
    let m = &mut meter();
    let decoded = air::decode(bc, m).expect("decode");
    let req = air::SpecRequest {
        call_stack_depth: 1,
        entry: entry.to_owned(),
        ..air::SpecRequest::default()
    };
    let spec = air::specialize(decoded, &req, m).expect("specialize");
    let memory = analyze_memory(&spec, p, m).expect("analyze");
    let plan = select(&spec, &memory, p, inputs, m).expect("select");
    let module = emit_module(&spec, &plan, m).expect("emit").module;
    verify(&module, p, m).expect("verify");
    (plan, module)
}

const TABLE: u64 = 0x0000_0010_0000_0000;
const DATA: u64 = 0x0000_0020_0000_0000;
const SINK: u64 = 0x0000_0030_0000_0000;

/// The root table (32-byte entries `host, guest, size, bind_offset`) with the sink at root 1 and
/// `[[buffer(0)]]` at root 2, as a buffer at `TABLE`.
fn root_table(size: u64, bind: u64) -> Buffer {
    let mut t = vec![0u8; 32 * 5];
    t[32..40].copy_from_slice(&SINK.to_le_bytes());
    t[48..56].copy_from_slice(&64u64.to_le_bytes());
    t[64..72].copy_from_slice(&DATA.to_le_bytes());
    t[72..80].copy_from_slice(&DATA.to_le_bytes());
    t[80..88].copy_from_slice(&size.to_le_bytes());
    t[88..96].copy_from_slice(&bind.to_le_bytes());
    Buffer {
        slot: None,
        address: Some(TABLE),
        bytes: t,
    }
}

/// The push bytes of `plan` with the root-table address and zero residuals.
fn push_with_table(plan: &Plan) -> Vec<u8> {
    let mut out = push_bytes(plan, &[]);
    for f in &plan.push.fields {
        if let (PushField::RootTableAddress, PushLoc::Push { offset }) = (f.field, f.loc) {
            let at = offset as usize;
            out[at..at + 8].copy_from_slice(&TABLE.to_le_bytes());
        }
    }
    out
}

/// A `BufferRangeLimit` variant reaches the slot through its device address from the root table.
/// Every leaf path, with bind offsets 0 and 8, at offsets around the end of the allocation and near
/// 2^32: no trap, exact in bounds, 0 out of bounds, and an out-of-bounds store lands in the sink
/// and leaves the allocation unchanged. Robust buffer access does not cover device addresses, so
/// the check is emitted on a robust profile too. A `half` NaN loaded, or stored from a constant,
/// need only come back a NaN where the module reads a `half` from its bits by the device's
/// widening or encodes one by its narrowing (`PrecisionNote::NativeNanEncoding`), whose NaN
/// encoding is the device's.
#[test]
fn a_device_address_slot_is_exact_and_contained() {
    let size = 32u64;
    let p = profile_with("moltenvk-m3", true, true);
    assert_eq!(p.fields().buffer_access, BufferAccess::Addressed);
    let mut inputs = LowerInputs::new();
    inputs.psb_slots.insert(sref(0));
    for k in leaf_kernels() {
        let what = k.name;
        let Some(bc) = assemble_text(k.name, &leaf_kernel(&k)) else {
            return;
        };
        let a = compile_bitcode(
            &bc,
            EntryRef::Name(k.name),
            &FunctionConstantValues::defaults(),
            &p,
            &inputs,
            &mut meter(),
            SPIRV_WORDS,
        )
        .unwrap_or_else(|e| panic!("{what}: {e}"));
        link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
        let native_halves =
            k.ty.contains("half") && a.precision.contains(&PrecisionNote::NativeNanEncoding);
        let (plan, m) = typed_module_with(&bc, k.name, &p, &inputs);
        for bind in [0u64, 8] {
            for off in offsets(size) {
                let o = off as u32;
                let rel = bind + u64::from(o);
                if rel % u64::from(k.align) != 0 {
                    continue;
                }
                let data = pattern(size as usize);
                let mut params = vec![0u8; 16];
                params[..4].copy_from_slice(&o.to_le_bytes());
                let res = Resources {
                    buffers: vec![
                        Buffer {
                            slot: None,
                            address: Some(DATA),
                            bytes: data.clone(),
                        },
                        Buffer {
                            slot: None,
                            address: Some(SINK),
                            bytes: vec![0; 64],
                        },
                        root_table(size, bind),
                        bound(1, 0, &[0; 32], 0),
                        bound(2, 0, &params, 0),
                    ],
                    push: push_with_table(&plan),
                    spec: Default::default(),
                };
                let n = u64::from(k.bytes);
                let inb = size >= n && rel <= size - n;
                let out = match run(&m, &Config::default(), res) {
                    Ok(o) => o,
                    Err(InterpError::Trap(t)) => {
                        panic!("{what} bind={bind} off={off:#x}: trapped {t:?}")
                    }
                    Err(e) => panic!("{what}: {e}"),
                };
                let at = rel as usize;
                match &k.store {
                    None => {
                        let want: Vec<u8> = if inb {
                            data[at..at + k.bytes as usize].to_vec()
                        } else {
                            vec![0; k.bytes as usize]
                        };
                        let got = &out.buffers[3][..k.bytes as usize];
                        if native_halves {
                            assert!(
                                same_halves(got, &want),
                                "{what} bind={bind} off={off:#x}: {got:?} for {want:?}"
                            );
                        } else {
                            assert_eq!(got, &want[..], "{what} bind={bind} off={off:#x}");
                        }
                    }
                    Some((_, bytes)) => {
                        let mut want = data.clone();
                        if inb {
                            want[at..at + bytes.len()].copy_from_slice(bytes);
                        }
                        if native_halves {
                            assert!(
                                same_halves(&out.buffers[0], &want),
                                "{what} bind={bind} off={off:#x}: {:?}",
                                out.buffers[0]
                            );
                        } else {
                            assert_eq!(out.buffers[0], want, "{what} bind={bind} off={off:#x}");
                        }
                    }
                }
            }
        }
    }
}

/// The constant offset `2³² − 2³⁰ − 16` (3 GiB − 16): at least 2³¹ above the bind offset.
const FAR: u64 = 0xBFFF_FFF0;

/// A kernel whose `data` slot has the window `[-16, FAR + 8)` (start below the bind offset,
/// 3 GiB long): it loads the words at -16 and `FAR` into `out[0]` and `out[1]`, and stores
/// `0x12345678` at -12 and `0x9ABCDEF0` at `FAR + 4`.
fn far_window_kernel() -> String {
    let far = FAR;
    let far4 = FAR + 4;
    format!(
        "{HEADER}define void @far_window(ptr addrspace(1) noundef \"air-buffer-no-alias\" %0, ptr addrspace(1) noundef \"air-buffer-no-alias\" %1, ptr addrspace(1) noundef \"air-buffer-no-alias\" %2) {{\n\
entry:\n\
  %lo = getelementptr i8, ptr addrspace(1) %0, i64 -16\n\
  %a = load i32, ptr addrspace(1) %lo, align 4\n\
  store i32 %a, ptr addrspace(1) %1, align 4\n\
  %hi = getelementptr i8, ptr addrspace(1) %0, i64 {far}\n\
  %b = load i32, ptr addrspace(1) %hi, align 4\n\
  %o1 = getelementptr i8, ptr addrspace(1) %1, i64 4\n\
  store i32 %b, ptr addrspace(1) %o1, align 4\n\
  %lo2 = getelementptr i8, ptr addrspace(1) %0, i64 -12\n\
  store i32 305419896, ptr addrspace(1) %lo2, align 4\n\
  %hi2 = getelementptr i8, ptr addrspace(1) %0, i64 {far4}\n\
  store i32 -1698898192, ptr addrspace(1) %hi2, align 4\n\
  ret void\n\
}}\n\n{}",
        meta("far_window")
    )
}

/// A device-address slot whose window starts below the bind offset and reaches 2³¹ or more above
/// it. The `u32` pointer offset maps back into the window, so accesses on both sides of a non-zero
/// bind offset are exact (a sign extension would have sent the access at `FAR` to `bind − 2³² +
/// FAR`, out of bounds). The 3 GiB allocation is modelled sparsely: two buffers at its device
/// addresses around the bind offset and around `FAR`, with the root table's `size` deciding
/// containment. Out of bounds on either side, a load is 0 and a store leaves the allocation
/// unchanged.
#[test]
fn a_device_address_slot_below_its_bind_offset_is_exact_and_contained() {
    let Some(bc) = assemble_text("far_window", &far_window_kernel()) else {
        return;
    };
    let mut inputs = LowerInputs::new();
    inputs.psb_slots.insert(sref(0));
    for robust in [true, false] {
        let p = profile_with("moltenvk-m3", robust, true);
        let what = format!("robust={robust}");
        assert_eq!(p.fields().buffer_access, BufferAccess::Addressed);
        let a = compile_bitcode(
            &bc,
            EntryRef::Name("far_window"),
            &FunctionConstantValues::defaults(),
            &p,
            &inputs,
            &mut meter(),
            SPIRV_WORDS,
        )
        .unwrap_or_else(|e| panic!("{what}: {e}"));
        match a.interface.slots.get(&sref(0)) {
            Some(SlotInterface::Buffer(b)) => {
                assert_eq!(
                    b.window,
                    Window::Bounded {
                        start: -16,
                        len: FAR + 24
                    },
                    "{what}"
                );
                assert_eq!(b.form, BufferForm::Psb, "{what}");
            }
            other => panic!("{what}: {other:?}"),
        }
        link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
        let (plan, m) = typed_module_with(&bc, "far_window", &p, &inputs);
        // (bind offset, size): all in bounds; the far side out of bounds; the near side out of
        // bounds (bind − 16 wraps below the allocation).
        for (bind, size) in [(64u64, 64 + FAR + 8), (64, 64 + FAR), (8, 8 + FAR + 8)] {
            let near = pattern(128);
            let far: Vec<u8> = pattern(160).into_iter().skip(128).collect();
            // `far` holds allocation bytes `bind + FAR - 16 ..`.
            let far_at = bind + FAR - 16;
            let res = Resources {
                buffers: vec![
                    Buffer {
                        slot: None,
                        address: Some(DATA),
                        bytes: near.clone(),
                    },
                    Buffer {
                        slot: None,
                        address: Some(DATA + far_at),
                        bytes: far.clone(),
                    },
                    Buffer {
                        slot: None,
                        address: Some(SINK),
                        bytes: vec![0; 64],
                    },
                    root_table(size, bind),
                    bound(1, 0, &[0; 16], 0),
                ],
                push: push_with_table(&plan),
                spec: Default::default(),
            };
            let near_inb = bind >= 16;
            let far_inb = bind + FAR + 8 <= size;
            let out = match run(&m, &Config::default(), res) {
                Ok(o) => o,
                Err(InterpError::Trap(t)) => panic!("{what} bind={bind}: trapped {t:?}"),
                Err(e) => panic!("{what}: {e}"),
            };
            let what = format!("{what} bind={bind} size={size:#x}");
            let lo = bind.wrapping_sub(16) as usize;
            let mut want_out = vec![0u8; 16];
            let mut want_near = near.clone();
            let mut want_far = far.clone();
            if near_inb {
                want_out[..4].copy_from_slice(&near[lo..lo + 4]);
                want_near[lo + 4..lo + 8].copy_from_slice(&0x1234_5678u32.to_le_bytes());
            }
            if far_inb {
                want_out[4..8].copy_from_slice(&far[16..20]);
                want_far[20..24].copy_from_slice(&0x9ABC_DEF0u32.to_le_bytes());
            }
            assert_eq!(out.buffers[4], want_out, "{what}: out");
            assert_eq!(out.buffers[0], want_near, "{what}: near");
            assert_eq!(out.buffers[1], want_far, "{what}: far");
        }
    }
}

/// A device-address slot whose `Bounded` window is longer than 2³² holds constant offsets one `u32`
/// pointer offset cannot tell apart, so selection refuses it (wide static pointers are not
/// implemented) instead of aliasing them.
#[test]
fn a_device_address_window_longer_than_four_gib_is_refused() {
    let text = format!(
        "{HEADER}define void @wide_window(ptr addrspace(1) noundef \"air-buffer-no-alias\" %0, ptr addrspace(1) noundef \"air-buffer-no-alias\" %1, ptr addrspace(1) noundef \"air-buffer-no-alias\" %2) {{\n\
entry:\n\
  %a = load i32, ptr addrspace(1) %0, align 4\n\
  store i32 %a, ptr addrspace(1) %1, align 4\n\
  %hi = getelementptr i8, ptr addrspace(1) %0, i64 5368709120\n\
  %b = load i32, ptr addrspace(1) %hi, align 4\n\
  %o1 = getelementptr i8, ptr addrspace(1) %1, i64 4\n\
  store i32 %b, ptr addrspace(1) %o1, align 4\n\
  ret void\n\
}}\n\n{}",
        meta("wide_window")
    );
    let Some(bc) = assemble_text("wide_window", &text) else {
        return;
    };
    let p = profile_with("moltenvk-m3", true, true);
    let mut inputs = LowerInputs::new();
    inputs.psb_slots.insert(sref(0));
    let r = plan_bitcode(
        &bc,
        EntryRef::Name("wide_window"),
        &FunctionConstantValues::defaults(),
        &p,
        &inputs,
        &mut meter(),
    );
    match r {
        Err(Error::Unsupported(u)) => assert_eq!(
            (u.feature, u.needs),
            (
                Feature::Lowering(LowerFamily::Access),
                Needs::NotImplemented
            )
        ),
        other => panic!("expected a refusal, got {other:?}"),
    }
}

/// Constant GEP indices whose products and sum exceed 128 bits (a 2⁶⁴ − 2-byte array of pairs,
/// indexed at `i64::MIN` twice and -1) compose in wrapping arithmetic: the offset is the low word
/// of the sum, 0xFFFF_FFFF, and selection gives a result instead of overflowing.
#[test]
fn huge_constant_gep_indices_wrap_to_the_low_word() {
    let text = format!(
        "{HEADER}define void @huge_gep(ptr addrspace(1) noundef \"air-buffer-no-alias\" %0, ptr addrspace(1) noundef \"air-buffer-no-alias\" %1, ptr addrspace(1) noundef \"air-buffer-no-alias\" %2) {{\n\
entry:\n\
  %p = getelementptr [9223372036854775807 x [2 x i8]], ptr addrspace(1) %0, i64 -9223372036854775808, i64 -9223372036854775808, i64 -1\n\
  %q = getelementptr i8, ptr addrspace(1) %p, i32 1\n\
  %a = load i8, ptr addrspace(1) %q, align 1\n\
  store i8 %a, ptr addrspace(1) %1, align 1\n\
  ret void\n\
}}\n\n{}",
        meta("huge_gep")
    );
    let Some(bc) = assemble_text("huge_gep", &text) else {
        return;
    };
    for (what, p) in encodings() {
        let (plan, m) = match typed_module(&bc, "huge_gep", &p) {
            Ok(x) => x,
            Err(e) => panic!("{what}: {e}"),
        };
        let data = pattern(16);
        let res = Resources {
            buffers: vec![bound(0, 0, &data, 0), bound(1, 0, &[0xAB; 16], 0)],
            push: push_bytes(&plan, &[]),
            spec: Default::default(),
        };
        let out = run(&m, &Config::default(), res).unwrap_or_else(|e| panic!("{what}: {e}"));
        // 0xFFFF_FFFF + 1 wraps to offset 0.
        assert_eq!(out.buffers[1][0], data[0], "{what}");
    }
}

/// On Profile A, when the per-stage storage-buffer limit is exceeded, the highest-index storage
/// slot moves to its device address (a deterministic per-function choice), declares
/// `DeviceAddresses`, and the function still validates. Profile F views a read-only slot as a
/// uniform buffer instead: the `device` params word (its window bounds its range) before the
/// `Whole` data slot, which stays a spare candidate; the view depends on the alias check. Without
/// robust descriptors (the in-shader encoding), or when only written slots are left to move,
/// Profile F refuses naming the limit.
#[test]
fn the_highest_storage_slots_fall_back_to_device_addresses() {
    let k = LeafKernel {
        name: "load_i32_aligned",
        ty: "i32",
        bytes: 4,
        align: 4,
        store: None,
    };
    let Some(bc) = assemble_text("fallback", &leaf_kernel(&k)) else {
        return;
    };
    for stem in PROFILES {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../metal2vulkan-validation/profiles")
            .join(format!("{stem}.json"));
        let text = std::fs::read_to_string(path).expect("profile");
        let mut desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
        desc.limits.max_per_stage_descriptor_storage_buffers = 2;
        let p = Device::new(desc, policy())
            .expect("device")
            .profile()
            .clone();
        let r = compile(&bc, k.name, &p);
        if p.fields().buffer_access == BufferAccess::Addressed {
            let a = r.unwrap_or_else(|e| panic!("{stem}: {e}"));
            let form = |i| match a.interface.slots.get(&sref(i)) {
                Some(SlotInterface::Buffer(b)) => b.form,
                other => panic!("{stem}: {other:?}"),
            };
            assert_eq!(form(2), BufferForm::Psb, "{stem}");
            assert_eq!(form(1), BufferForm::Descriptor(DescriptorView::Storage));
            assert_eq!(form(0), BufferForm::Descriptor(DescriptorView::Storage));
            assert!(a
                .facts
                .contains(&RuntimeFact::DeviceAddresses { slot: sref(2) }));
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
        } else {
            let a = r.unwrap_or_else(|e| panic!("{stem}: {e}"));
            let form = |i| match a.interface.slots.get(&sref(i)) {
                Some(SlotInterface::Buffer(b)) => b.form,
                other => panic!("{stem}: {other:?}"),
            };
            assert_eq!(
                form(2),
                BufferForm::Descriptor(DescriptorView::UniformWords),
                "{stem}"
            );
            assert_eq!(form(1), BufferForm::Descriptor(DescriptorView::Storage));
            assert_eq!(form(0), BufferForm::Descriptor(DescriptorView::Storage));
            assert!(a
                .facts
                .contains(&RuntimeFact::UniformViewAliasing { slot: sref(2) }));
            assert!(a.facts.contains(&RuntimeFact::UniformViewRange {
                slot: sref(0),
                choice: UniformViewChoice::Spare,
            }));
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
            // No robust descriptors, or no read-only slot left to move: the limit is refused.
            let mut strict = profile_desc(stem);
            strict.limits.max_per_stage_descriptor_storage_buffers = 2;
            strict.features.remove(&VkFeature::RobustBufferAccess);
            strict.features.remove(&VkFeature::RobustBufferAccess2);
            let mut written = profile_desc(stem);
            written.limits.max_per_stage_descriptor_storage_buffers = 0;
            for desc in [strict, written] {
                let p = Device::new(desc, policy())
                    .expect("device")
                    .profile()
                    .clone();
                match compile(&bc, k.name, &p) {
                    Err(Error::Unsupported(u)) => {
                        assert_eq!(u.feature, Feature::DescriptorCount, "{stem}")
                    }
                    other => panic!("{stem}: expected a refusal, got {other:?}"),
                }
            }
        }
    }
}

/// The description of committed profile `stem`.
fn profile_desc(stem: &str) -> DeviceDescription {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles")
        .join(format!("{stem}.json"));
    let text = std::fs::read_to_string(path).expect("profile");
    DeviceDescription::from_khronos_json(&text, None).expect("description")
}

/// A kernel with a loop reserves no status binding (its modules carry no execution bound), so the
/// descriptor fallback counts the Metal slots alone. With three storage slots and a per-stage limit
/// of two, moving one slot brings the storage count to two: the highest slot to a device address
/// where the profile has them, else (Profile F) the uniform-view fallback moves only the bounded
/// params word, and the `Whole` data slot stays a storage descriptor.
#[test]
fn a_looping_kernel_reserves_no_status_binding_in_the_fallback() {
    let name = "fallback_loop";
    let text = format!(
        "{HEADER}define void @{name}(ptr addrspace(1) noundef \"air-buffer-no-alias\" %0, ptr addrspace(1) noundef \"air-buffer-no-alias\" %1, ptr addrspace(1) noundef \"air-buffer-no-alias\" %2) {{\n\
         entry:\n  %n = load i32, ptr addrspace(1) %2, align 4\n  %z = icmp eq i32 %n, 0\n  br i1 %z, label %exit, label %loop\n\
         loop:\n  %i = phi i32 [ 0, %entry ], [ %next, %loop ]\n\
         \x20 %p = getelementptr i8, ptr addrspace(1) %0, i32 %i\n  %v = load i8, ptr addrspace(1) %p, align 1\n\
         \x20 %q = getelementptr i8, ptr addrspace(1) %1, i32 %i\n  store i8 %v, ptr addrspace(1) %q, align 1\n\
         \x20 %next = add i32 %i, 1\n  %c = icmp ult i32 %next, %n\n  br i1 %c, label %loop, label %exit\n\
         exit:\n  ret void\n}}\n\n{}",
        meta(name)
    );
    let Some(bc) = assemble_text(name, &text) else {
        return;
    };
    for stem in PROFILES {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../metal2vulkan-validation/profiles")
            .join(format!("{stem}.json"));
        let text = std::fs::read_to_string(path).expect("profile");
        let mut desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
        desc.limits.max_per_stage_descriptor_storage_buffers = 2;
        let p = Device::new(desc, policy())
            .expect("device")
            .profile()
            .clone();
        let r = compile(&bc, name, &p);
        let a = r.unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert!(!a.facts.contains(&RuntimeFact::ExecutionStatus), "{stem}");
        if p.fields().buffer_access != BufferAccess::Addressed {
            let form = |i| match a.interface.slots.get(&sref(i)) {
                Some(SlotInterface::Buffer(b)) => b.form,
                other => panic!("{stem}: {other:?}"),
            };
            let storage = BufferForm::Descriptor(DescriptorView::Storage);
            assert_eq!(
                form(2),
                BufferForm::Descriptor(DescriptorView::UniformWords),
                "{stem}"
            );
            assert_eq!((form(0), form(1)), (storage, storage), "{stem}: one move");
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
            continue;
        }
        let form = |i| match a.interface.slots.get(&sref(i)) {
            Some(SlotInterface::Buffer(b)) => b.form,
            other => panic!("{stem}: {other:?}"),
        };
        assert_eq!(form(2), BufferForm::Psb, "{stem}");
        assert_eq!(
            form(1),
            BufferForm::Descriptor(DescriptorView::Storage),
            "{stem}: one move is enough"
        );
        assert_eq!(form(0), BufferForm::Descriptor(DescriptorView::Storage));
        link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
    }
}

/// The plan's size estimate bounds the emitted words of every leaf kernel, in both encodings and in
/// the device-address form.
#[test]
fn the_estimate_bounds_every_leaf_kernel() {
    let mut psb = LowerInputs::new();
    psb.psb_slots.insert(sref(0));
    for robust in [true, false] {
        let p = profile_with("moltenvk-m3", robust, true);
        for inputs in [LowerInputs::new(), psb.clone()] {
            for k in leaf_kernels() {
                let what = format!(
                    "{} robust={robust} psb={}",
                    k.name,
                    inputs != LowerInputs::new()
                );
                let Some(bc) = assemble_text(k.name, &leaf_kernel(&k)) else {
                    return;
                };
                let entry = EntryRef::Name(k.name);
                let consts = FunctionConstantValues::defaults();
                let plan = plan_bitcode(&bc, entry, &consts, &p, &inputs, &mut meter())
                    .unwrap_or_else(|e| panic!("{what}: {e}"));
                let a =
                    compile_bitcode(&bc, entry, &consts, &p, &inputs, &mut meter(), SPIRV_WORDS)
                        .unwrap_or_else(|e| panic!("{what}: {e}"));
                let words = a.module.words.len() as u64;
                assert!(
                    plan.estimate.words >= words,
                    "{what}: estimate {} < emitted {words}",
                    plan.estimate.words
                );
            }
        }
    }
}
/// The opcodes of a SPIR-V module's instructions, in order.
fn opcodes(words: &[u32]) -> Vec<u16> {
    let mut out = Vec::new();
    let mut i = 5;
    while let Some(w) = words.get(i) {
        out.push((w & 0xFFFF) as u16);
        i += ((w >> 16) as usize).max(1);
    }
    out
}

/// A kernel that loads `n` words of `data` at `params[0] + 8i` and stores their sum to `out`.
fn summing_kernel(name: &str, n: u32) -> String {
    let mut body = String::from("  %s0 = add i32 0, 0\n");
    for i in 0..n {
        body.push_str(&format!(
            "  %o{i} = add i32 %o, {}\n  %p{i} = getelementptr i8, ptr addrspace(1) %0, i32 %o{i}\n  %v{i} = load i32, ptr addrspace(1) %p{i}, align 4\n  %s{} = add i32 %s{i}, %v{i}\n",
            8 * i,
            i + 1
        ));
    }
    format!(
        "{HEADER}define void @{name}(ptr addrspace(1) noundef \"air-buffer-no-alias\" %0, ptr addrspace(1) noundef \"air-buffer-no-alias\" %1, ptr addrspace(1) noundef \"air-buffer-no-alias\" %2) {{\nentry:\n  %o = load i32, ptr addrspace(1) %2, align 4\n{body}  store i32 %s{n}, ptr addrspace(1) %1, align 4\n  ret void\n}}\n\n{}",
        meta(name)
    )
}

/// A block loads a slot's bind offset once, however many of its accesses use the slot, so each
/// further aligned word load of the slot adds exactly its one `OpLoad`; the sum is exact.
#[test]
fn a_block_loads_each_push_field_once() {
    const LOAD: u16 = 61;
    for (what, p) in encodings() {
        let mut loads = Vec::new();
        for n in [1u32, 3] {
            let name = if n == 1 { "sum_one" } else { "sum_three" };
            let Some(bc) = assemble_text(name, &summing_kernel(name, n)) else {
                return;
            };
            check_l3(&bc, name, &p, &what);
            let a = compile(&bc, name, &p).unwrap_or_else(|e| panic!("{what}: {e}"));
            let words = &a.module.words;
            loads.push(opcodes(words).iter().filter(|o| **o == LOAD).count());
            if n == 3 {
                let (plan, m) = typed_module(&bc, name, &p).unwrap_or_else(|e| panic!("{e}"));
                let data = pattern(64);
                let mut params = vec![0u8; 16];
                params[..4].copy_from_slice(&4u32.to_le_bytes());
                let res = Resources {
                    buffers: vec![
                        bound(0, 0, &data, 0),
                        bound(1, 0, &[0; 16], 0),
                        bound(2, 0, &params, 0),
                    ],
                    push: push_bytes(&plan, &[(0, 0)]),
                    spec: Default::default(),
                };
                let out = run(&m, &Config::default(), res).unwrap_or_else(|e| panic!("{e:?}"));
                let word = |at: usize| u32::from_le_bytes(data[at..at + 4].try_into().unwrap());
                let want = word(4).wrapping_add(word(12)).wrapping_add(word(20));
                assert_eq!(out.buffers[1][..4], want.to_le_bytes(), "{what}");
            }
        }
        assert_eq!(loads[1], loads[0] + 2, "{what}: loads {loads:?}");
    }
}

/// A kernel storing 7 through `out + 8` when `params[0] == 0`; the pointer is computed in the
/// entry block (`crossing`) or in the block that stores through it.
fn guarded_store(name: &str, crossing: bool) -> String {
    let gep = "  %p = getelementptr i8, ptr addrspace(1) %1, i32 8\n";
    let (early, late) = if crossing { (gep, "") } else { ("", gep) };
    format!(
        "{HEADER}define void @{name}(ptr addrspace(1) noundef \"air-buffer-no-alias\" %0, ptr addrspace(1) noundef \"air-buffer-no-alias\" %1, ptr addrspace(1) noundef \"air-buffer-no-alias\" %2) {{\nentry:\n  %o = load i32, ptr addrspace(1) %2, align 4\n{early}  %c = icmp eq i32 %o, 0\n  br i1 %c, label %a, label %b\na:\n{late}  store i32 7, ptr addrspace(1) %p, align 4\n  br label %b\nb:\n  ret void\n}}\n\n{}",
        meta(name)
    )
}

/// A block-crossing value whose emitted value is a constant (a buffer pointer at a constant offset)
/// is used as that constant where another block imports it: the function body is the one of the
/// kernel computing the pointer where it stores through it (no copy of a loaded variable, no
/// extraction of its offset).
#[test]
fn a_constant_crossing_value_is_used_as_the_constant() {
    const FUNCTION: u16 = 54;
    for (what, p) in encodings() {
        let mut bodies = Vec::new();
        for (name, crossing) in [("store_late", false), ("store_crossing", true)] {
            let Some(bc) = assemble_text(name, &guarded_store(name, crossing)) else {
                return;
            };
            check_l3(&bc, name, &p, &what);
            let a = compile(&bc, name, &p).unwrap_or_else(|e| panic!("{what}: {e}"));
            let ops = opcodes(&a.module.words);
            let start = ops.iter().position(|o| *o == FUNCTION).expect("a function");
            bodies.push(ops[start..].to_vec());
        }
        assert_eq!(bodies[0], bodies[1], "{what}");
    }
}

/// A kernel loading `data[params[0]]` in the entry block and, when it is 0, storing it (`second`:
/// plus `data[params[0] + 8]`, loaded in that block) to `out`.
fn two_block_kernel(name: &str, second: bool) -> String {
    let late = if second {
        "  %q = getelementptr i8, ptr addrspace(1) %0, i32 %o8\n  %w = load i32, ptr addrspace(1) %q, align 4\n  %s = add i32 %v, %w\n"
    } else {
        "  %s = add i32 %v, 0\n"
    };
    format!(
        "{HEADER}define void @{name}(ptr addrspace(1) noundef \"air-buffer-no-alias\" %0, ptr addrspace(1) noundef \"air-buffer-no-alias\" %1, ptr addrspace(1) noundef \"air-buffer-no-alias\" %2) {{\nentry:\n  %o = load i32, ptr addrspace(1) %2, align 4\n  %o8 = add i32 %o, 8\n  %p = getelementptr i8, ptr addrspace(1) %0, i32 %o\n  %v = load i32, ptr addrspace(1) %p, align 4\n  %c = icmp eq i32 %v, 0\n  br i1 %c, label %a, label %b\na:\n{late}  store i32 %s, ptr addrspace(1) %1, align 4\n  br label %b\nb:\n  ret void\n}}\n\n{}",
        meta(name)
    )
}

/// A push field the entry block loaded is not loaded again by a later block, so a later block's
/// further word load of the slot adds exactly its one `OpLoad`.
#[test]
fn a_push_field_the_entry_block_loaded_is_not_reloaded() {
    const LOAD: u16 = 61;
    for (what, p) in encodings() {
        let mut loads = Vec::new();
        for (name, second) in [("entry_only", false), ("entry_and_later", true)] {
            let Some(bc) = assemble_text(name, &two_block_kernel(name, second)) else {
                return;
            };
            check_l3(&bc, name, &p, &what);
            let a = compile(&bc, name, &p).unwrap_or_else(|e| panic!("{what}: {e}"));
            loads.push(
                opcodes(&a.module.words)
                    .iter()
                    .filter(|o| **o == LOAD)
                    .count(),
            );
        }
        assert_eq!(loads[1], loads[0] + 1, "{what}: loads {loads:?}");
    }
}

/// The word cap holds on the emitted words, not only on the estimate. A public fixture whose
/// estimate is below its emitted words (a storage texture write) compiles at a cap equal to its
/// emitted words and is refused with `SpirvWords` one word below, although its estimate passes both
/// caps.
#[test]
fn the_word_cap_holds_on_the_emitted_words() {
    let Some(bc) = assemble_file("storage_texture_write") else {
        return;
    };
    // (`DriverDeviation::MetalNormStorageWrites`): on an unmeasured device the write's rounding
    // routine, counted whole, lifts the estimate past the emitted words.
    let p = profile_with("moltenvk-m3", true, true);
    let entry = EntryRef::Name("storage_texture_write");
    let consts = FunctionConstantValues::defaults();
    let inputs = LowerInputs::new();
    let estimate = plan_bitcode(&bc, entry, &consts, &p, &inputs, &mut meter())
        .expect("plan")
        .estimate
        .words;
    let words = compile_bitcode(&bc, entry, &consts, &p, &inputs, &mut meter(), SPIRV_WORDS)
        .expect("compile")
        .module
        .words
        .len() as u64;
    assert!(
        estimate < words - 1,
        "the fixture no longer emits more than its estimate ({estimate} >= {words} - 1)"
    );
    compile_bitcode(&bc, entry, &consts, &p, &inputs, &mut meter(), words)
        .expect("compiles at a cap of its emitted words");
    assert_eq!(
        compile_bitcode(&bc, entry, &consts, &p, &inputs, &mut meter(), words - 1).unwrap_err(),
        Error::LimitExceeded(crate::error::Limit::SpirvWords)
    );
}

/// Profile `stem` with the in-shader encoding and without `features`.
fn profile_without(stem: &str, features: &[VkFeature]) -> DeviceProfile {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles")
        .join(format!("{stem}.json"));
    let text = std::fs::read_to_string(path).expect("profile");
    let mut desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    desc.features.remove(&VkFeature::RobustBufferAccess);
    desc.features.remove(&VkFeature::RobustBufferAccess2);
    for f in features {
        desc.features.remove(f);
    }
    Device::new(desc, policy())
        .expect("device")
        .profile()
        .clone()
}

/// With typed narrow views (`u16[]` on `moltenvk-m3` without 8-bit storage access, `u8[]` on the
/// 2023 baseline, both on `moltenvk-m3`), every generated leaf kernel is exact in bounds and
/// contained out of bounds at bind residuals that are not multiples of 4: a narrow unit's index
/// follows the binder contract (descriptor base congruent to the Metal address modulo 4), and a
/// sub-word store through a unit leaves its neighbours unchanged. A `half` NaN loaded, or stored
/// from a constant, need only come back a NaN where the module reads a `half` from its bits by the
/// device's widening or encodes one by its narrowing (`PrecisionNote::NativeNanEncoding`), whose
/// NaN encoding is the device's.
#[test]
fn narrow_units_are_exact_at_every_bind_residual() {
    let size = 32u64;
    const NO8: &[VkFeature] = &[VkFeature::StorageBuffer8BitAccess];
    let profiles: [(&str, &[VkFeature]); 3] = [
        ("moltenvk-m3", &[]),
        ("moltenvk-m3", NO8),
        ("lunarg-desktop-baseline-2023", &[]),
    ];
    for (stem, removed) in profiles {
        let p = profile_without(stem, removed);
        for k in leaf_kernels() {
            let what = format!("{stem} -{removed:?} {}", k.name);
            let Some(bc) = assemble_text(k.name, &leaf_kernel(&k)) else {
                return;
            };
            check_l3(&bc, k.name, &p, &what);
            let native_halves = k.ty.contains("half")
                && compile(&bc, k.name, &p)
                    .unwrap_or_else(|e| panic!("{what}: {e}"))
                    .precision
                    .contains(&PrecisionNote::NativeNanEncoding);
            let (plan, m) = typed_module(&bc, k.name, &p).unwrap_or_else(|e| panic!("{what}: {e}"));
            let mut probes: Vec<u32> = (0..=9).collect();
            probes.extend(size as u32 - 9..=size as u32 + 1);
            for r in [1u32, 2, 3, 6] {
                for &o in &probes {
                    let desc = u64::from(r + o);
                    if desc % u64::from(k.align) != 0 {
                        continue;
                    }
                    let data = pattern(size as usize);
                    let mut params = vec![0u8; 16];
                    params[..4].copy_from_slice(&o.to_le_bytes());
                    let mut dbuf = bound(0, r, &data, 0);
                    dbuf.bytes.truncate(r as usize + size as usize);
                    // A whole number of words is bound.
                    while !dbuf.bytes.len().is_multiple_of(4) {
                        dbuf.bytes.push(0x5A);
                    }
                    let res = Resources {
                        buffers: vec![
                            dbuf.clone(),
                            bound(1, 0, &[0; 32], 0),
                            bound(2, 0, &params, 0),
                        ],
                        push: push_bytes(&plan, &[(0, r)]),
                        spec: Default::default(),
                    };
                    let dsize = dbuf.bytes.len() as u64;
                    let inb = desc + u64::from(k.bytes) <= dsize;
                    let out = run(&m, &Config::default(), res)
                        .unwrap_or_else(|e| panic!("{what} r={r} off={o}: {e:?}"));
                    let at = desc as usize;
                    match &k.store {
                        None => {
                            let want: Vec<u8> = if inb {
                                dbuf.bytes[at..at + k.bytes as usize].to_vec()
                            } else {
                                vec![0; k.bytes as usize]
                            };
                            let got = &out.buffers[1][..k.bytes as usize];
                            if native_halves {
                                assert!(
                                    same_halves(got, &want),
                                    "{what} r={r} off={o}: {got:?} for {want:?}"
                                );
                            } else {
                                assert_eq!(got, &want[..], "{what} r={r} off={o}");
                            }
                        }
                        Some((_, bytes)) => {
                            let mut want = dbuf.bytes.clone();
                            if inb {
                                want[at..at + bytes.len()].copy_from_slice(bytes);
                            }
                            if native_halves {
                                assert!(
                                    same_halves(&out.buffers[0], &want),
                                    "{what} r={r} off={o}: {:?}",
                                    out.buffers[0]
                                );
                            } else {
                                assert_eq!(out.buffers[0], want, "{what} r={r} off={o}");
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Whether the little-endian `half` patterns `got` are `want`'s, a NaN pattern any NaN.
fn same_halves(got: &[u8], want: &[u8]) -> bool {
    let nan = |h: u16| h & 0x7C00 == 0x7C00 && h & 0x3FF != 0;
    got.len() == want.len()
        && got
            .as_chunks::<2>()
            .0
            .iter()
            .zip(want.as_chunks::<2>().0)
            .all(|(g, w)| {
                let (g, w) = (u16::from_le_bytes(*g), u16::from_le_bytes(*w));
                g == w || (nan(g) && nan(w))
            })
}

/// The variables bound to kernel buffer slot `index`: its word view and its narrow views.
fn views_of(m: &TypedModule, index: u32) -> usize {
    m.globals()
        .iter()
        .filter(|g| g.kind == GlobalKind::Resource(kslot(index)))
        .count()
}

/// A narrow-view case: the profile, its removed features, the kernel, the views of `data` and of
/// `out`, and whether a sub-word store's atomics remain.
type NarrowCase = (
    &'static str,
    &'static [VkFeature],
    &'static str,
    usize,
    usize,
    bool,
);

/// A storage slot gets a narrow view of a unit exactly when one of its accesses reads or writes a
/// leaf in whole units of that width and the profile has the width's storage access and integer
/// type; a store through one is a plain store, without the sub-word store's atomics.
#[test]
fn narrow_views_are_declared_where_the_profile_accesses_the_unit() {
    const ATOMIC_AND: u16 = 240;
    const ATOMIC_OR: u16 = 241;
    const NO16: &[VkFeature] = &[VkFeature::StorageBuffer16BitAccess];
    const NO8: &[VkFeature] = &[VkFeature::StorageBuffer8BitAccess];
    let cases: [NarrowCase; 14] = [
        ("moltenvk-m3", &[], "store_half", 2, 0, false),
        ("moltenvk-m3", &[], "store_i8", 2, 0, false),
        ("moltenvk-m3", &[], "store_uchar3", 3, 0, false),
        ("moltenvk-m3", &[], "store_half3", 2, 0, false),
        // The loaded half is stored to `out` as an aligned leaf with a 2-byte partial word.
        ("moltenvk-m3", &[], "load_half", 2, 2, false),
        ("moltenvk-m3", &[], "load_i32_aligned", 1, 1, false),
        // An under-aligned leaf joins words: no unit holds it.
        ("moltenvk-m3", &[], "store_i16_bytes", 1, 0, true),
        ("moltenvk-m3", NO16, "store_half", 1, 0, true),
        ("moltenvk-m3", NO8, "store_half", 2, 0, false),
        ("moltenvk-m3", NO8, "store_i8", 1, 0, true),
        // The floor has no narrow storage access: every sub-word store joins its word.
        ("vk12-floor", &[], "store_half", 1, 0, true),
        ("vk12-floor", &[], "store_i8", 1, 0, true),
        (
            "lunarg-desktop-baseline-2023",
            &[],
            "store_half",
            1,
            0,
            true,
        ),
        ("lunarg-desktop-baseline-2023", &[], "store_i8", 2, 0, false),
    ];
    let kernels = leaf_kernels();
    for (stem, removed, name, data_views, out_views, atomics) in cases {
        let what = format!("{stem} -{removed:?} {name}");
        let k = kernels.iter().find(|k| k.name == name).expect("kernel");
        let p = profile_without(stem, removed);
        let Some(bc) = assemble_text(name, &leaf_kernel(k)) else {
            return;
        };
        check_l3(&bc, name, &p, &what);
        let (_, m) = typed_module(&bc, name, &p).unwrap_or_else(|e| panic!("{what}: {e}"));
        assert_eq!(views_of(&m, 0), data_views, "{what}: views of data");
        assert_eq!(views_of(&m, 1), out_views, "{what}: views of out");
        let a = compile(&bc, name, &p).unwrap_or_else(|e| panic!("{what}: {e}"));
        let ops = opcodes(&a.module.words);
        assert_eq!(
            ops.iter().any(|o| *o == ATOMIC_AND || *o == ATOMIC_OR),
            atomics,
            "{what}: sub-word store atomics"
        );
    }
}

/// A fragment function over one `read_write` buffer `data`: `body` reads and writes it and
/// defines the `i32` `%r`, which the function returns as its color's red channel.
fn fragment_over_data(name: &str, body: &str) -> String {
    format!(
        r#"define <4 x float> @{name}(<4 x float> noundef %pos, ptr addrspace(1) noundef %data) {{
{body}  %f = uitofp i32 %r to float
  %o = insertelement <4 x float> zeroinitializer, float %f, i64 0
  ret <4 x float> %o
}}
!air.fragment = !{{!0}}
!air.version = !{{!9}}
!air.language_version = !{{!10}}
!0 = !{{ptr @{name}, !1, !2}}
!1 = !{{!11}}
!11 = !{{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}}
!2 = !{{!12, !3}}
!12 = !{{i32 0, !"air.position", !"air.center", !"air.no_perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"pos"}}
!3 = !{{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"data"}}
!9 = !{{i32 2, i32 8, i32 0}}
!10 = !{{!"Metal", i32 4, i32 0, i32 0}}
"#
    )
}

/// The buffer-slot variables of serialized module words `words` (relocations `relocs`): each
/// variable's id with its binding symbol, and the ids decorated `NonWritable`.
fn buffer_views(words: &[u32], relocs: &[(Symbol, u32)]) -> (Vec<(u32, ResourceSlot)>, Vec<u32>) {
    const DECORATE: u32 = 71;
    const NON_WRITABLE: u32 = 24;
    let views = relocs
        .iter()
        .filter_map(|(s, at)| match s {
            Symbol::Binding(slot) if slot.kind == ResourceKind::Buffer => words
                .get((*at as usize).checked_sub(2)?)
                .map(|id| (*id, *slot)),
            _ => None,
        })
        .collect();
    let mut non_writable = Vec::new();
    let mut i = 5;
    while let Some(w) = words.get(i) {
        let len = ((w >> 16) as usize).max(1);
        let inst = &words[i..(i + len).min(words.len())];
        if w & 0xFFFF == DECORATE && inst.get(2) == Some(&NON_WRITABLE) {
            non_writable.push(inst[1]);
        }
        i += len;
    }
    (views, non_writable)
}

/// A buffer slot's word view and its narrow and `uvec4` views are one memory, so a graphics stage
/// decorates none of them `NonWritable` when any of them is written: a slot read as a word and
/// written as a unit, or read as a unit and written as a word, keeps every view writable. A slot
/// only read keeps both views `NonWritable`.
#[test]
fn every_view_of_a_written_slot_stays_writable() {
    // (unit type, its byte offset, its alignment, the bytes it takes, the unit of word `%w`, and
    // the word of unit `%b`)
    let units = [
        ("i8", 5, 1, 1u32, "trunc i32 %w to i8", "zext i8 %b to i32"),
        ("i16", 6, 2, 2, "trunc i32 %w to i16", "zext i16 %b to i32"),
        (
            "<4 x i32>",
            16,
            16,
            16,
            "insertelement <4 x i32> zeroinitializer, i32 %w, i64 0",
            "extractelement <4 x i32> %b, i64 0",
        ),
    ];
    // `moltenvk-m3` without 8-bit storage has the `u16` view only.
    let profiles: [(&str, &[VkFeature]); 4] = [
        ("moltenvk-m3", &[]),
        ("lavapipe", &[]),
        ("amd-rdna3-radv", &[]),
        ("moltenvk-m3", &[VkFeature::StorageBuffer8BitAccess]),
    ];
    for (stem, removed) in profiles {
        let p = profile_without(stem, removed);
        let has = crate::memory::leaf::Units::storage_on(&p);
        for (ty, off, align, bytes, to_unit, from_unit) in units {
            let word = "  %w = load i32, ptr addrspace(1) %data, align 4\n";
            let at =
                format!("  %u = getelementptr inbounds i8, ptr addrspace(1) %data, i64 {off}\n");
            let cases = [
                (
                    "word_read_unit_write",
                    format!("{word}{at}  %t = {to_unit}\n  store {ty} %t, ptr addrspace(1) %u, align {align}\n  %r = add i32 %w, 0\n"),
                    true,
                ),
                (
                    "unit_read_word_write",
                    format!("{at}  %b = load {ty}, ptr addrspace(1) %u, align {align}\n  %z = {from_unit}\n  store i32 %z, ptr addrspace(1) %data, align 4\n  %r = add i32 %z, 0\n"),
                    true,
                ),
                (
                    "word_and_unit_read",
                    format!("{word}{at}  %b = load {ty}, ptr addrspace(1) %u, align {align}\n  %z = {from_unit}\n  %r = add i32 %w, %z\n"),
                    false,
                ),
            ];
            for (name, body, written) in cases {
                let what = format!("{stem} -{removed:?} {ty} {name}");
                let text = fragment_over_data(name, &body);
                let Some(bc) = assemble_text(name, &text) else {
                    return;
                };
                let a = compile(&bc, name, &p).unwrap_or_else(|e| panic!("{what}: {e}"));
                link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
                let (views, non_writable) = buffer_views(&a.module.words, &a.module.relocs);
                let data: Vec<u32> = views
                    .iter()
                    .filter(|(_, slot)| slot.index == 0)
                    .map(|(id, _)| *id)
                    .collect();
                assert_eq!(
                    data.len(),
                    if has.has(bytes) { 2 } else { 1 },
                    "{what}: views of data"
                );
                // The layout counts the slot's views as its binding's variables.
                let inputs = crate::pipeline::link::LinkInputs::standalone();
                let linked = crate::pipeline::link::link(&[&a], &inputs, &p)
                    .unwrap_or_else(|e| panic!("{what}: link: {e}"));
                let variables = linked.layout.sets[0]
                    .bindings
                    .iter()
                    .find(|b| {
                        matches!(b.source,
                            crate::pipeline::layout::BindingSource::Buffer(s) if s.index == 0)
                    })
                    .map(|b| b.variables);
                assert_eq!(
                    variables,
                    Some(data.len() as u32),
                    "{what}: variables of data's binding"
                );
                for id in &data {
                    assert_eq!(
                        non_writable.contains(id),
                        !written,
                        "{what}: view %{id} of data, NonWritable {non_writable:?}"
                    );
                }
            }
        }
    }
}

/// The `uvec4` residuals the 16-byte units are checked at: zero, below 16, a multiple of 16, past
/// 32, and wrapped.
const VEC4_RESIDUALS: [u32; 6] = [0, 4, 12, 16, 36, 0xFFFF_FFF0];

/// A 16-byte leaf whose alignment is at least 16 is one element of the slot's `uvec4` view
/// (declared beside the word view, and stored with one `OpStore`), in both encodings, on every
/// tested profile; aligned to 4 it moves word by word (four stores, no `uvec4` view of the slot it
/// stores to). At every residual a descriptor offset that is a multiple of 16 reads and writes
/// exactly the leaf's 16 bytes of a 48-byte range (a `half` NaN stored from a constant a NaN where
/// the module encodes it by the device's narrowing, `PrecisionNote::NativeNanEncoding`), and with
/// the in-shader check the offset past the range reads 0 and writes nothing.
#[test]
fn sixteen_byte_leaves_move_as_one_uvec4() {
    const OP_STORE: u16 = 62;
    let kernels = leaf_kernels();
    let sixteen: Vec<&LeafKernel> = kernels
        .iter()
        .filter(|k| k.bytes == 16 && k.align >= 4)
        .collect();
    assert_eq!(sixteen.len(), 13, "the aligned 16-byte leaf kernels");
    for (what, p) in encodings() {
        let robust = what.ends_with("robust=true");
        for k in &sixteen {
            let what = format!("{what} {}", k.name);
            let Some(bc) = assemble_text(k.name, &leaf_kernel(k)) else {
                return;
            };
            check_l3(&bc, k.name, &p, &what);
            let (plan, m) = typed_module(&bc, k.name, &p).unwrap_or_else(|e| panic!("{what}: {e}"));
            let vec4 = k.align >= 16;
            // A load kernel stores the loaded leaf to `out` at alignment 16.
            assert_eq!(
                views_of(&m, 0),
                if vec4 { 2 } else { 1 },
                "{what}: views of data"
            );
            assert_eq!(
                views_of(&m, 1),
                if k.store.is_none() { 2 } else { 0 },
                "{what}: views of out"
            );
            let a = compile(&bc, k.name, &p).unwrap_or_else(|e| panic!("{what}: {e}"));
            let native_halves =
                k.ty.contains("half") && a.precision.contains(&PrecisionNote::NativeNanEncoding);
            let stores = opcodes(&a.module.words)
                .iter()
                .filter(|o| **o == OP_STORE)
                .count();
            let want = if k.store.is_some() && !vec4 { 4 } else { 1 };
            assert_eq!(stores, want, "{what}: stores");
            let data = pattern(48);
            for r in VEC4_RESIDUALS {
                // The interpreter models no robustness: robust encodings run in bounds only.
                let descs: &[u32] = if robust {
                    &[0, 16, 32]
                } else {
                    &[0, 16, 32, 48]
                };
                for &desc in descs {
                    let o = desc.wrapping_sub(r);
                    let mut params = vec![0u8; 16];
                    params[..4].copy_from_slice(&o.to_le_bytes());
                    let res = Resources {
                        buffers: vec![
                            Buffer {
                                slot: Some(kslot(0)),
                                address: None,
                                bytes: data.clone(),
                            },
                            bound(1, 0, &[0; 32], 0),
                            bound(2, 0, &params, 0),
                        ],
                        push: push_bytes(&plan, &[(0, r)]),
                        spec: Default::default(),
                    };
                    let out = run(&m, &Config::default(), res)
                        .unwrap_or_else(|e| panic!("{what} r={r:#x} desc={desc}: {e:?}"));
                    let at = desc as usize;
                    let inb = at + 16 <= data.len();
                    match &k.store {
                        None => {
                            let want: Vec<u8> = if inb {
                                data[at..at + 16].to_vec()
                            } else {
                                vec![0; 16]
                            };
                            assert_eq!(out.buffers[1][..16], want[..], "{what} r={r:#x} {desc}");
                        }
                        Some((_, bytes)) => {
                            let mut want = data.clone();
                            if inb {
                                want[at..at + 16].copy_from_slice(bytes);
                            }
                            if native_halves {
                                assert!(
                                    same_halves(&out.buffers[0], &want),
                                    "{what} r={r:#x} {desc}: {:?}",
                                    out.buffers[0]
                                );
                            } else {
                                assert_eq!(out.buffers[0], want, "{what} r={r:#x} {desc}");
                            }
                        }
                    }
                }
            }
        }
    }
}

/// A kernel reading the `float4` at byte 32 of a `constant` buffer (a `Bounded` window, so a
/// uniform view under robust descriptors) into `out`.
const CONSTANT_VEC4: &str = r#"define void @constant_vec4(ptr addrspace(2) noundef "air-buffer-no-alias" %0, ptr addrspace(1) noundef "air-buffer-no-alias" %1) {
entry:
  %p = getelementptr inbounds i8, ptr addrspace(2) %0, i64 32
  %v = load <4 x float>, ptr addrspace(2) %p, align 16
  store <4 x float> %v, ptr addrspace(1) %1, align 16
  ret void
}
!air.kernel = !{!9}
!air.version = !{!17}
!air.language_version = !{!18}
!9 = !{ptr @constant_vec4, !10, !11}
!10 = !{}
!11 = !{!12, !13}
!12 = !{i32 0, !"air.buffer", !"air.buffer_size", i32 48, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 48, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"Three", !"air.arg_name", !"in"}
!13 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"out"}
!17 = !{i32 2, i32 8, i32 0}
!18 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

/// Under robust descriptors a `constant` slot's 16-aligned `float4` is one `uvec4` of its uniform
/// view: a `Uniform { uvec4[N] }` view beside the `u32[]` words with `uniformBufferStandardLayout`,
/// the std140 `uvec4[]` word view itself without. It reads the vector's 16 bytes exactly at every
/// residual the binder can give the window `[32, 48)` with a 16-aligned bind offset: the window's
/// first byte at descriptor byte `0, 16, ..., A - 16` for the base alignment `A` (the residual
/// wraps below the window).
#[test]
fn a_constant_vector_is_one_uvec4_of_its_uniform_view() {
    let text = format!("{HEADER}{CONSTANT_VEC4}");
    let Some(bc) = assemble_text("constant_vec4", &text) else {
        return;
    };
    for stem in PROFILES {
        for standard in [true, false] {
            let what = format!("{stem} standard={standard}");
            let p = profile_with(stem, true, standard);
            let a = compile(&bc, "constant_vec4", &p).unwrap_or_else(|e| panic!("{what}: {e}"));
            let form = match a.interface.slots.get(&sref(0)) {
                Some(SlotInterface::Buffer(b)) => b.form,
                other => panic!("{what}: {other:?}"),
            };
            let want = if standard {
                DescriptorView::UniformWords
            } else {
                DescriptorView::UniformVec4
            };
            assert_eq!(form, BufferForm::Descriptor(want), "{what}");
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            let (plan, m) = typed_module(&bc, "constant_vec4", &p).expect("module");
            assert_eq!(
                views_of(&m, 0),
                if standard { 2 } else { 1 },
                "{what}: views of in"
            );
            let data = pattern(512);
            let base = p
                .fields()
                .limits
                .min_uniform_buffer_offset_alignment
                .max(16) as u32;
            // Metal byte 32 is descriptor byte `r + 32`.
            for at in (0..base).step_by(16) {
                let r = at.wrapping_sub(32);
                let at = at as usize;
                let res = Resources {
                    buffers: vec![
                        Buffer {
                            slot: Some(kslot(0)),
                            address: None,
                            bytes: data.clone(),
                        },
                        bound(1, 0, &[0; 16], 0),
                    ],
                    push: push_bytes(&plan, &[(0, r)]),
                    spec: Default::default(),
                };
                let out = run(&m, &Config::default(), res)
                    .unwrap_or_else(|e| panic!("{what} r={r:#x}: {e:?}"));
                assert_eq!(out.buffers[1][..16], data[at..at + 16], "{what} r={r:#x}");
            }
        }
    }
}
