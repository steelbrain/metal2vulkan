//! Narrow GEP indices end to end (static roots need no `Int64`).
//!
//! - Dynamic-index kernels whose `i64` GEP indices are `zext`/`sext` of an `i1`, `i8`, `i16`, or
//!   `i32` value (a loaded index, negative ones included, and `thread_position_in_grid`) compile
//!   on `runtime-floor` without `shaderInt64` (below the runtime's floor) and `moltenvk-m3`, in
//!   both containment encodings, pass
//!   L3, select the empty `arith.index_extension` strategy, and compute byte-exact results on the
//!   CPU interpreter.
//! - An extension defined in another block than its GEPs (the loop-kernel shape: the index is
//!   extended in the entry block and used after a loop) is read narrowed too: the extension's
//!   narrow operand is the demoted value, so no 64-bit value is formed. So is an extension of a
//!   loop-header phi used after the loop, which reads the phi's value from the last iteration even
//!   when the latch has already written the next one.
//! - An `i64` index that is not such an extension (a 64-bit load, a shift right of the extended
//!   value) is a 64-bit value: `moltenvk-m3` holds it as a `u64`, `runtime-floor` without
//!   `shaderInt64` as two words, and both compile it exactly.
//!
//! `llvm-as` assembles the kernels; without it the tests skip visibly.

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::device::requirements::VkFeature;
use crate::device::{Device, DeviceDescription, DevicePolicy, DeviceProfile};
use crate::lower::select::Plan;
use crate::pipeline::interface::PushLoc;
use crate::pipeline::sym::{PushField, ResourceKind, ResourceSlot, Stage};
use crate::pipeline::testing::{
    compile, link_and_validate, llvm_assemble, typed_module, AssembleError,
};
use crate::spirv::ids::{DemoteTy, ScalarTy};
use crate::spirv::interp::{run, Buffer, Config, Resources};

const HEADER: &str = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n";

/// `runtime-floor` without `shaderInt64`: below the runtime's floor, a device that holds a 64-bit
/// value as two words.
const NO_INT64: &str = "runtime-floor without shaderInt64";

/// A profile without and one with `shaderInt64`.
const PROFILES: [&str; 2] = [NO_INT64, "moltenvk-m3"];

/// Committed profile `stem` (or [`NO_INT64`]'s device), optionally with buffer robustness disabled
/// (the in-shader encoding).
fn profile_with(stem: &str, robust: bool) -> DeviceProfile {
    let file = if stem == NO_INT64 {
        "runtime-floor"
    } else {
        stem
    };
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles")
        .join(format!("{file}.json"));
    let text = std::fs::read_to_string(path).expect("profile");
    let mut desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    if stem == NO_INT64 {
        desc.features.remove(&VkFeature::ShaderInt64);
    }
    if !robust {
        desc.features.remove(&VkFeature::RobustBufferAccess);
        desc.features.remove(&VkFeature::RobustBufferAccess2);
    }
    Device::from_queries(desc, DevicePolicy::default())
        .expect("device")
        .profile()
        .clone()
}

/// Every tested profile in both encodings.
fn encodings() -> Vec<(String, DeviceProfile)> {
    let mut out = Vec::new();
    for stem in PROFILES {
        for robust in [true, false] {
            out.push((
                format!("{stem} robust={robust}"),
                profile_with(stem, robust),
            ));
        }
    }
    out
}

fn assemble_text(name: &str, text: &str) -> Option<Vec<u8>> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "m2v-gep-index-{}-{}-{name}",
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

/// A kernel `name` over buffers `data` (0, read-write), `out` (1), and `params` (2, read, element
/// `params_ty` of `params_size` bytes), plus `thread_position_in_grid` as `%3` when `tid`.
fn kernel(name: &str, body: &str, params: (&str, u32), tid: bool) -> String {
    let (params_ty, params_size) = params;
    let tid_param = if tid { ", i32 noundef %3" } else { "" };
    let tid_meta = if tid { ", !15" } else { "" };
    format!(
        "{HEADER}define void @{name}(ptr addrspace(1) noundef \"air-buffer-no-alias\" %0, ptr addrspace(1) noundef \"air-buffer-no-alias\" %1, ptr addrspace(1) noundef \"air-buffer-no-alias\" %2{tid_param}) {{\nentry:\n{body}  ret void\n}}\n\n\
         !air.kernel = !{{!9}}\n!air.version = !{{!17}}\n!air.language_version = !{{!18}}\n\
         !9 = !{{ptr @{name}, !10, !11}}\n!10 = !{{}}\n!11 = !{{!12, !13, !14{tid_meta}}}\n\
         !12 = !{{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 1, !\"air.arg_type_align_size\", i32 1, !\"air.arg_type_name\", !\"uchar\", !\"air.arg_name\", !\"data\"}}\n\
         !13 = !{{i32 1, !\"air.buffer\", !\"air.location_index\", i32 1, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 1, !\"air.arg_type_align_size\", i32 1, !\"air.arg_type_name\", !\"uchar\", !\"air.arg_name\", !\"out\"}}\n\
         !14 = !{{i32 2, !\"air.buffer\", !\"air.location_index\", i32 2, i32 1, !\"air.read\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 {params_size}, !\"air.arg_type_align_size\", i32 {params_size}, !\"air.arg_type_name\", !\"{params_ty}\", !\"air.arg_name\", !\"params\"}}\n\
         !15 = !{{i32 3, !\"air.thread_position_in_grid\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"t\"}}\n\
         !17 = !{{i32 2, i32 8, i32 0}}\n!18 = !{{!\"Metal\", i32 4, i32 0, i32 0}}\n"
    )
}

fn kslot(index: u32) -> ResourceSlot {
    ResourceSlot {
        stage: Stage::Kernel,
        kind: ResourceKind::Buffer,
        index,
    }
}

/// A buffer bound at its start over `data` padded with zeros to a multiple of 16.
fn bound(index: u32, data: &[u8]) -> Buffer {
    let mut bytes = data.to_vec();
    while !bytes.len().is_multiple_of(16) {
        bytes.push(0);
    }
    Buffer {
        slot: Some(kslot(index)),
        address: None,
        bytes,
    }
}

/// Push bytes of `plan` with zero residuals (every field is in the push window).
fn push_bytes(plan: &Plan) -> Vec<u8> {
    for f in &plan.push.fields {
        assert!(
            matches!(f.loc, PushLoc::Push { .. }),
            "{:?} in the argument table",
            f.field
        );
        assert!(
            matches!(
                f.field,
                PushField::SlotBindOffset(_) | PushField::DispatchRegion
            ),
            "{:?}",
            f.field
        );
    }
    vec![0u8; 128]
}

fn pattern(n: usize) -> Vec<u8> {
    (0..n)
        .map(|i| (i as u8).wrapping_mul(29).wrapping_add(11) ^ ((i >> 8) as u8))
        .collect()
}

/// One loaded-index kernel: `params[0]` (of `params`) is extended by `ext` to `i64` and indexes
/// elements of `elem` (`bytes` wide) from `data + base`; the element is copied to `out` and
/// overwritten in `data` with `marker`.
struct IndexKernel {
    name: &'static str,
    params: (&'static str, u32),
    /// The IR computing `%x` from `%o` (the loaded param), empty when `%x` is `%o`.
    prep: &'static str,
    /// The IR type of `%x`.
    x_ty: &'static str,
    ext: &'static str,
    elem: &'static str,
    bytes: u32,
    base: u32,
    marker: &'static str,
    data: usize,
    /// The params values to run, as raw `params` bytes, with the expected index value.
    cases: Vec<(Vec<u8>, i64)>,
}

impl IndexKernel {
    fn text(&self) -> String {
        let (pty, _) = self.params;
        let load_ty = match pty {
            "short" | "ushort" => "i16",
            "char" | "uchar" => "i8",
            _ => "i32",
        };
        let x = if self.prep.is_empty() { "%o" } else { "%x" };
        let (e, xt, el, b, mk, a) = (
            self.ext,
            self.x_ty,
            self.elem,
            self.base,
            self.marker,
            self.bytes,
        );
        let body = format!(
            "  %o = load {load_ty}, ptr addrspace(1) %2, align 4\n{}  %s = {e} {xt} {x} to i64\n  %mid = getelementptr i8, ptr addrspace(1) %0, i64 {b}\n  %p = getelementptr inbounds {el}, ptr addrspace(1) %mid, i64 %s\n  %v = load {el}, ptr addrspace(1) %p, align {a}\n  store {el} %v, ptr addrspace(1) %1, align 4\n  store {el} {mk}, ptr addrspace(1) %p, align {a}\n",
            self.prep
        );
        kernel(self.name, &body, self.params, false)
    }
}

fn le(v: u32, n: usize) -> Vec<u8> {
    v.to_le_bytes()[..n].to_vec()
}

fn index_kernels() -> Vec<IndexKernel> {
    let words = |vals: &[i32]| -> Vec<(Vec<u8>, i64)> {
        vals.iter()
            .map(|v| (le(*v as u32, 4), i64::from(*v)))
            .collect()
    };
    vec![
        IndexKernel {
            name: "sext_i32_index",
            params: ("int", 4),
            prep: "",
            x_ty: "i32",
            ext: "sext",
            elem: "i32",
            bytes: 4,
            base: 64,
            marker: "-559038737",
            data: 128,
            cases: words(&(-16..16).collect::<Vec<_>>()),
        },
        IndexKernel {
            name: "zext_i32_index",
            params: ("uint", 4),
            prep: "",
            x_ty: "i32",
            ext: "zext",
            elem: "i32",
            bytes: 4,
            base: 0,
            marker: "-559038737",
            data: 128,
            cases: words(&(0..32).collect::<Vec<_>>()),
        },
        IndexKernel {
            name: "sext_i16_index",
            params: ("short", 2),
            prep: "",
            x_ty: "i16",
            ext: "sext",
            elem: "i8",
            bytes: 1,
            base: 32768,
            marker: "-91",
            data: 65536,
            cases: [0u16, 1, 0x7FFF, 0x8000, 0xFFF0, 0xFFFF]
                .iter()
                .map(|v| (le(u32::from(*v), 2), i64::from(*v as i16)))
                .collect(),
        },
        IndexKernel {
            name: "zext_i16_index",
            params: ("ushort", 2),
            prep: "",
            x_ty: "i16",
            ext: "zext",
            elem: "i8",
            bytes: 1,
            base: 0,
            marker: "-91",
            data: 65536,
            cases: [0u16, 1, 0x7FFF, 0x8000, 0xFFF0, 0xFFFF]
                .iter()
                .map(|v| (le(u32::from(*v), 2), i64::from(*v)))
                .collect(),
        },
        IndexKernel {
            name: "sext_i8_index",
            params: ("char", 1),
            prep: "",
            x_ty: "i8",
            ext: "sext",
            elem: "i16",
            bytes: 2,
            base: 256,
            marker: "-4403",
            data: 512,
            cases: (0..=255u8).map(|v| (vec![v], i64::from(v as i8))).collect(),
        },
        IndexKernel {
            name: "zext_i8_index",
            params: ("uchar", 1),
            prep: "",
            x_ty: "i8",
            ext: "zext",
            elem: "i16",
            bytes: 2,
            base: 0,
            marker: "-4403",
            data: 512,
            cases: (0..=255u8).map(|v| (vec![v], i64::from(v))).collect(),
        },
        IndexKernel {
            name: "sext_i1_index",
            params: ("uint", 4),
            prep: "  %x = icmp ne i32 %o, 0\n",
            x_ty: "i1",
            ext: "sext",
            elem: "i32",
            bytes: 4,
            base: 16,
            marker: "-559038737",
            data: 32,
            cases: vec![(le(0, 4), 0), (le(5, 4), -1)],
        },
        IndexKernel {
            name: "zext_i1_index",
            params: ("uint", 4),
            prep: "  %x = icmp ne i32 %o, 0\n",
            x_ty: "i1",
            ext: "zext",
            elem: "i32",
            bytes: 4,
            base: 16,
            marker: "-559038737",
            data: 32,
            cases: vec![(le(0, 4), 0), (le(5, 4), 1)],
        },
    ]
}

/// The little-endian bytes of IR integer constant `marker` of `bytes` bytes.
fn marker_bytes(marker: &str, bytes: u32) -> Vec<u8> {
    let v: i64 = marker.parse().expect("marker");
    v.to_le_bytes()[..bytes as usize].to_vec()
}

/// Every loaded-index kernel compiles without `shaderInt64`, selects the empty extension strategy,
/// passes L3, and reads and writes exactly the element its extended index names.
#[test]
fn extended_loaded_indices_are_exact_without_int64() {
    for k in index_kernels() {
        let Some(bc) = assemble_text(k.name, &k.text()) else {
            return;
        };
        for (what, p) in encodings() {
            let what = format!("{what} {}", k.name);
            let a = compile(&bc, k.name, &p).unwrap_or_else(|e| panic!("{what}: {e}"));
            let names: Vec<&String> = a.strategies.entries.keys().collect();
            assert!(
                names.iter().any(|n| *n == "arith.index_extension"),
                "{what}: {names:?}"
            );
            assert!(
                !names.iter().any(|n| *n == "arith.cast"),
                "{what}: {names:?}"
            );
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            let (plan, m) = typed_module(&bc, k.name, &p).expect("module");
            let data = pattern(k.data);
            let marker = marker_bytes(k.marker, k.bytes);
            for (params, index) in &k.cases {
                let at = i64::from(k.base) + index * i64::from(k.bytes);
                let at = usize::try_from(at).expect("in range");
                let n = k.bytes as usize;
                let res = Resources {
                    buffers: vec![bound(0, &data), bound(1, &[0; 16]), bound(2, params)],
                    push: push_bytes(&plan),
                    spec: Default::default(),
                };
                let out = run(&m, &Config::default(), res)
                    .unwrap_or_else(|e| panic!("{what} {index}: {e:?}"));
                assert_eq!(out.buffers[1][..n], data[at..at + n], "{what} {index}");
                let mut want = data.clone();
                want[at..at + n].copy_from_slice(&marker);
                assert_eq!(out.buffers[0][..k.data], want[..], "{what} {index}");
            }
        }
    }
}

/// Over `thread_position_in_grid`: `out[t] = data[t]` through `zext i32 %t to i64`, with one
/// extension feeding both GEPs.
#[test]
fn a_grid_position_index_is_exact_without_int64() {
    let body = "  %e = zext i32 %3 to i64\n  %p = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %e\n  %v = load i32, ptr addrspace(1) %p, align 4\n  %q = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %e\n  store i32 %v, ptr addrspace(1) %q, align 4\n";
    let Some(bc) = assemble_text("grid_copy", &kernel("grid_copy", body, ("uint", 4), true)) else {
        return;
    };
    for (what, p) in encodings() {
        let a = compile(&bc, "grid_copy", &p).unwrap_or_else(|e| panic!("{what}: {e}"));
        assert!(
            a.strategies.entries.contains_key("arith.index_extension"),
            "{what}"
        );
        link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
        let (plan, m) = typed_module(&bc, "grid_copy", &p).expect("module");
        let data = pattern(32);
        let res = Resources {
            buffers: vec![bound(0, &data), bound(1, &[0; 32]), bound(2, &[0; 16])],
            push: push_bytes(&plan),
            spec: Default::default(),
        };
        let cfg = Config {
            workgroups: [8, 1, 1],
            ..Config::default()
        };
        let out = run(&m, &cfg, res).unwrap_or_else(|e| panic!("{what}: {e:?}"));
        assert_eq!(out.buffers[1][..32], data[..], "{what}");
    }
}

/// A loop kernel: `params` holds `(index, n)`; the entry block extends `index` (by `ext`, or
/// `thread_position_in_grid` when `tid`) to `i64`, a loop sums `0..n`, and the exit block stores
/// `data[index] + sum` to `out[index]` and `-1` to `data[index]` through GEPs on the extension.
fn loop_kernel(name: &str, ext: &str, tid: bool) -> String {
    let x = if tid { "%3" } else { "%o" };
    let body = format!(
        "  %o = load i32, ptr addrspace(1) %2, align 4\n  %np = getelementptr inbounds i32, ptr addrspace(1) %2, i64 1\n  %n = load i32, ptr addrspace(1) %np, align 4\n  %e = {ext} i32 {x} to i64\n  br label %loop\nloop:\n  %i = phi i32 [ 0, %entry ], [ %i1, %loop ]\n  %acc = phi i32 [ 0, %entry ], [ %acc1, %loop ]\n  %acc1 = add i32 %acc, %i\n  %i1 = add i32 %i, 1\n  %c = icmp ult i32 %i1, %n\n  br i1 %c, label %loop, label %exit\nexit:\n  %mid = getelementptr i8, ptr addrspace(1) %0, i64 64\n  %p = getelementptr inbounds i32, ptr addrspace(1) %mid, i64 %e\n  %v = load i32, ptr addrspace(1) %p, align 4\n  %s = add i32 %v, %acc1\n  %omid = getelementptr i8, ptr addrspace(1) %1, i64 64\n  %q = getelementptr inbounds i32, ptr addrspace(1) %omid, i64 %e\n  store i32 %s, ptr addrspace(1) %q, align 4\n  store i32 -1, ptr addrspace(1) %p, align 4\n"
    );
    kernel(name, &body, ("uint", 8), tid)
}

/// An index extension in the entry block used by GEPs after a loop is read narrowed, on every
/// profile and encoding: no `shaderInt64`, no 64-bit demotion, and exact results for negative
/// (`sext`) and large (`zext`) loaded indices and for the grid position.
#[test]
fn an_extension_used_after_a_loop_is_exact_without_int64() {
    let wrap = |v: i32| (v as u32).to_le_bytes();
    for (name, ext, tid) in [
        ("loop_sext_index", "sext", false),
        ("loop_zext_index", "zext", false),
        ("loop_grid_index", "zext", true),
    ] {
        let Some(bc) = assemble_text(name, &loop_kernel(name, ext, tid)) else {
            return;
        };
        for (what, p) in encodings() {
            let what = format!("{what} {name}");
            let a = compile(&bc, name, &p).unwrap_or_else(|e| panic!("{what}: {e}"));
            let names: Vec<&String> = a.strategies.entries.keys().collect();
            assert!(
                names.iter().any(|n| *n == "arith.index_extension"),
                "{what}: {names:?}"
            );
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            let (plan, m) = typed_module(&bc, name, &p).expect("module");
            assert!(
                plan.cross
                    .iter()
                    .all(|(_, t)| !matches!(t, DemoteTy::Scalar(ScalarTy::Int { bits: 64 }))),
                "{what}: {:?}",
                plan.cross
            );
            let data = pattern(128);
            let indices: Vec<i32> = match (ext, tid) {
                (_, true) => vec![0],
                ("sext", _) => vec![-16, -3, 0, 5, 15],
                _ => vec![0, 7, 15],
            };
            for index in indices {
                let n = 5u32;
                let mut params = wrap(index).to_vec();
                params.extend_from_slice(&n.to_le_bytes());
                let res = Resources {
                    buffers: vec![bound(0, &data), bound(1, &[0; 128]), bound(2, &params)],
                    push: push_bytes(&plan),
                    spec: Default::default(),
                };
                let groups = if tid { 4 } else { 1 };
                let cfg = Config {
                    workgroups: [groups, 1, 1],
                    ..Config::default()
                };
                let out = run(&m, &cfg, res).unwrap_or_else(|e| panic!("{what} {index}: {e:?}"));
                let lanes: Vec<i64> = if tid {
                    (0..i64::from(groups)).collect()
                } else {
                    vec![i64::from(index)]
                };
                let mut want_out = [0u8; 128];
                let mut want_data = data.clone();
                for lane in lanes {
                    let at = usize::try_from(64 + 4 * lane).expect("in range");
                    let v = u32::from_le_bytes(data[at..at + 4].try_into().expect("word"));
                    let s = v.wrapping_add((0..n).sum::<u32>());
                    want_out[at..at + 4].copy_from_slice(&s.to_le_bytes());
                    want_data[at..at + 4].copy_from_slice(&[0xFF; 4]);
                }
                assert_eq!(out.buffers[1][..128], want_out[..], "{what} {index}");
                assert_eq!(out.buffers[0][..128], want_data[..], "{what} {index}");
            }
        }
    }
}

/// A loop kernel whose index is an extension (by `ext`) of the loop-header phi `%i`, used after
/// the loop. `params` holds `(start, n)`; `%i` counts from `start` while `%k` counts the `n ≥ 1`
/// iterations, and the exit block copies `data[%e]` to `out[%e]` and stores `-1` to `data[%e]`
/// (both from byte 64). Shapes:
///
/// - `"latch"`: one block is header and latch, holds the extension, and exits after the parallel
///   copy has written `%i + 1` into the phi's incoming variable, so the exit must read the value
///   `%i` had in the last iteration, `start + n - 1`, not `start + n`.
/// - `"body"`: the extension is in the body between the header and a latch that exits, which
///   reads the same `start + n - 1`.
/// - `"header"`: the header holds the extension and exits before its latch, so the exit reads the
///   header's last value, `start + n`.
fn loop_phi_kernel(name: &str, ext: &str, shape: &str) -> String {
    let head = "  %o = load i32, ptr addrspace(1) %2, align 4\n  %np = getelementptr inbounds i32, ptr addrspace(1) %2, i64 1\n  %n = load i32, ptr addrspace(1) %np, align 4\n  br label %loop\n";
    let exit = "exit:\n  %mid = getelementptr i8, ptr addrspace(1) %0, i64 64\n  %p = getelementptr inbounds i32, ptr addrspace(1) %mid, i64 %e\n  %v = load i32, ptr addrspace(1) %p, align 4\n  %omid = getelementptr i8, ptr addrspace(1) %1, i64 64\n  %q = getelementptr inbounds i32, ptr addrspace(1) %omid, i64 %e\n  store i32 %v, ptr addrspace(1) %q, align 4\n  store i32 -1, ptr addrspace(1) %p, align 4\n";
    let step = "  %i1 = add i32 %i, 1\n  %k1 = add i32 %k, 1\n";
    let loop_ = match shape {
        "latch" => format!(
            "loop:\n  %i = phi i32 [ %o, %entry ], [ %i1, %loop ]\n  %k = phi i32 [ 0, %entry ], [ %k1, %loop ]\n  %e = {ext} i32 %i to i64\n{step}  %c = icmp ult i32 %k1, %n\n  br i1 %c, label %loop, label %exit\n"
        ),
        "body" => format!(
            "loop:\n  %i = phi i32 [ %o, %entry ], [ %i1, %latch ]\n  %k = phi i32 [ 0, %entry ], [ %k1, %latch ]\n  br label %body\nbody:\n  %e = {ext} i32 %i to i64\n  br label %latch\nlatch:\n{step}  %c = icmp ult i32 %k1, %n\n  br i1 %c, label %loop, label %exit\n"
        ),
        "header" => format!(
            "loop:\n  %i = phi i32 [ %o, %entry ], [ %i1, %latch ]\n  %k = phi i32 [ 0, %entry ], [ %k1, %latch ]\n  %e = {ext} i32 %i to i64\n  %c = icmp ult i32 %k, %n\n  br i1 %c, label %latch, label %exit\nlatch:\n{step}  br label %loop\n"
        ),
        other => panic!("no loop shape {other}"),
    };
    kernel(name, &format!("{head}{loop_}{exit}"), ("uint", 8), false)
}

/// Over a loop-header phi: an extension of the phi, used after the loop, reads the phi's own value
/// where the use stands, never the incoming variable the latch has already overwritten. Exact on
/// every profile and encoding, without `shaderInt64` or a 64-bit demotion.
#[test]
fn an_extension_of_a_loop_header_phi_used_after_the_loop_is_exact() {
    for (shape, last) in [("latch", -1), ("body", -1), ("header", 0)] {
        for ext in ["sext", "zext"] {
            let name = format!("loop_phi_{shape}_{ext}");
            let Some(bc) = assemble_text(&name, &loop_phi_kernel(&name, ext, shape)) else {
                return;
            };
            let starts: &[i32] = if ext == "sext" {
                &[-12, -3, 0, 4]
            } else {
                &[0, 5]
            };
            for (what, p) in encodings() {
                let what = format!("{what} {name}");
                let a = compile(&bc, &name, &p).unwrap_or_else(|e| panic!("{what}: {e}"));
                assert!(
                    a.strategies.entries.contains_key("arith.index_extension"),
                    "{what}"
                );
                link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
                let (plan, m) = typed_module(&bc, &name, &p).expect("module");
                assert!(
                    plan.cross
                        .iter()
                        .chain(&plan.phis)
                        .all(|(_, t)| !matches!(t, DemoteTy::Scalar(ScalarTy::Int { bits: 64 }))),
                    "{what}: {:?} {:?}",
                    plan.cross,
                    plan.phis
                );
                let data = pattern(128);
                for &start in starts {
                    for n in [1u32, 3, 7] {
                        let index = i64::from(start) + i64::from(n) + last;
                        let mut params = start.to_le_bytes().to_vec();
                        params.extend_from_slice(&n.to_le_bytes());
                        let res = Resources {
                            buffers: vec![bound(0, &data), bound(1, &[0; 128]), bound(2, &params)],
                            push: push_bytes(&plan),
                            spec: Default::default(),
                        };
                        let out = run(&m, &Config::default(), res)
                            .unwrap_or_else(|e| panic!("{what} {start} {n}: {e:?}"));
                        let at = usize::try_from(64 + 4 * index).expect("in range");
                        let mut want_out = [0u8; 128];
                        want_out[at..at + 4].copy_from_slice(&data[at..at + 4]);
                        let mut want_data = data.clone();
                        want_data[at..at + 4].copy_from_slice(&[0xFF; 4]);
                        let case = format!("{what} start {start} n {n}");
                        assert_eq!(out.buffers[1][..128], want_out[..], "{case}");
                        assert_eq!(out.buffers[0][..128], want_data[..], "{case}");
                    }
                }
            }
        }
    }
}

/// A kernel name, its params element, its body, the params value, and the element index read.
type WideCase = (
    &'static str,
    (&'static str, u32),
    &'static str,
    Vec<u8>,
    i64,
);

/// An `i64` index that is not an index extension is a 64-bit value. `moltenvk-m3` holds it as a
/// native `u64`; [`NO_INT64`] has no `shaderInt64` and holds it as two words. Both compile it and
/// read the element exactly.
#[test]
fn an_i64_index_that_is_not_an_extension_is_a_64_bit_value() {
    let cases: [WideCase; 2] = [
        (
            "wide_loaded_index",
            ("ulong", 8),
            "  %w = load i64, ptr addrspace(1) %2, align 8\n  %p = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %w\n  %v = load i32, ptr addrspace(1) %p, align 4\n  store i32 %v, ptr addrspace(1) %1, align 4\n",
            5u64.to_le_bytes().to_vec(),
            5,
        ),
        // A shift right is no index word (its low word needs the high one).
        (
            "extension_with_shift_right",
            ("int", 4),
            "  %o = load i32, ptr addrspace(1) %2, align 4\n  %s = sext i32 %o to i64\n  %t = ashr i64 %s, 1\n  %mid = getelementptr i8, ptr addrspace(1) %0, i64 64\n  %p = getelementptr inbounds i32, ptr addrspace(1) %mid, i64 %t\n  %v = load i32, ptr addrspace(1) %p, align 4\n  store i32 %v, ptr addrspace(1) %1, align 4\n",
            (-6i32).to_le_bytes().to_vec(),
            -3 + 16,
        ),
    ];
    for (name, params, body, value, index) in cases {
        let Some(bc) = assemble_text(name, &kernel(name, body, params, false)) else {
            return;
        };
        for stem in PROFILES {
            let p = profile_with(stem, true);
            assert_eq!(p.fields().leaf.int64, stem == "moltenvk-m3");
            let a = compile(&bc, name, &p).unwrap_or_else(|e| panic!("{name} {stem}: {e}"));
            assert!(
                !a.strategies.entries.contains_key("arith.index_extension"),
                "{name} {stem}"
            );
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{name} {stem}: L3: {e}"));
            let (plan, m) = typed_module(&bc, name, &p).expect("module");
            let data = pattern(128);
            let res = Resources {
                buffers: vec![bound(0, &data), bound(1, &[0; 16]), bound(2, &value)],
                push: push_bytes(&plan),
                spec: Default::default(),
            };
            let out =
                run(&m, &Config::default(), res).unwrap_or_else(|e| panic!("{name} {stem}: {e:?}"));
            let at = usize::try_from(index * 4).expect("in range");
            assert_eq!(out.buffers[1][..4], data[at..at + 4], "{name} {stem}");
        }
    }
}
