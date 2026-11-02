//! Typed threadgroup objects (`memory::plan::Planner::typed_threadgroup`).
//!
//! - A kernel's static `half` (or 16-bit) threadgroup array whose every access is a plain load or
//!   store of its element is a `Workgroup` variable of its own type, each element in its held form:
//!   its stores and loads are plain, with no word atomics and no conversion to and from the `half`
//!   bits, and every `half` bit pattern (NaN payloads, signalling NaNs, denormals, both zeros)
//!   reads back as it was stored, on the CPU interpreter, but that a NaN's bits are the device's
//!   where the module reads a `half` from its bits by the device's widening
//!   (`PrecisionNote::NativeNanEncoding`).
//! - The reported static threadgroup length stays Metal's layout's, and the arena holds only the
//!   other objects.
//! - Where the structure does not allow it the array stays in the arena: an access through a
//!   punned pointer, a pointer merging two objects, a `[[threadgroup(n)]]` region beside it, an
//!   integer threadgroup address, and a held form past the device's limit.
//!
//! `llvm-as` assembles the kernels; without it the tests skip visibly.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::device::profile::DeviceProfile;
use crate::device::DevicePolicy;
use crate::lower::select::Plan;
use crate::pipeline::interface::PushLoc;
use crate::pipeline::sym::{Axis, PushField, ResourceKind, ResourceSlot, SpecKey, Stage};
use crate::pipeline::testing::{
    compile, device_nan_bits, link_and_validate, llvm_assemble, profile, typed_module,
    AssembleError,
};
use crate::spirv::interp::{run, Buffer, Config, Order, Resources, SubgroupSize};
use crate::spirv::ir::{StorageClass, TypedModule};

const PROFILES: [&str; 3] = ["moltenvk-m3", "vk12-floor", "lavapipe"];

const DATALAYOUT: &str = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32";

fn p(stem: &str) -> DeviceProfile {
    profile(stem, DevicePolicy::default()).expect("profile")
}

fn assemble(name: &str, body: &str) -> Option<Vec<u8>> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let text = format!(
        "target datalayout = \"{DATALAYOUT}\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n{body}"
    );
    let dir = std::env::temp_dir().join(format!(
        "m2v-typed-tg-{}-{}-{name}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).expect("scratch");
    let ll = dir.join("k.ll");
    std::fs::write(&ll, &text).expect("write");
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

const KERNEL_MD: &str = r#"declare void @air.wg.barrier(i32, i32)

!air.kernel = !{!0}
!0 = !{ptr @NAME, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!4 = !{i32 1, !"air.thread_index_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"t"}
"#;

/// The half bit pattern thread `t` stores: spread over the 16 bits, NaN payloads (quiet and
/// signalling, both signs), infinities, denormals and zeros among the 64.
fn pattern(t: u32) -> u32 {
    (t.wrapping_mul(1021) ^ 0x7C01) & 0xFFFF
}

/// `@h: [64 x half]`: thread `t` stores the half of bits `pattern(t)`, and after a barrier reads
/// the one of thread `63 - t` back as bits into `out[t]`. `extra` adds lines after the read that
/// define the `i32` `%e`, which the kernel stores to `out[64 + t]` (no test reads it); `globals`
/// adds globals.
fn halves(name: &str, globals: &str, extra: &str) -> String {
    let extra = if extra.is_empty() {
        "  %e = add i32 %t, 0"
    } else {
        extra
    };
    format!(
        r#"@h = internal addrspace(3) global [64 x half] undef, align 2
{globals}
define void @{name}(ptr addrspace(1) %out, i32 %t) {{
  %m = mul i32 %t, 1021
  %c = xor i32 %m, 31745
  %bits = trunc i32 %c to i16
  %v = bitcast i16 %bits to half
  %p = getelementptr inbounds [64 x half], ptr addrspace(3) @h, i32 0, i32 %t
  store half %v, ptr addrspace(3) %p, align 2
  tail call void @air.wg.barrier(i32 2, i32 1)
  %r = sub i32 63, %t
  %q = getelementptr inbounds [64 x half], ptr addrspace(3) @h, i32 0, i32 %r
  %x = load half, ptr addrspace(3) %q, align 2
  %xb = bitcast half %x to i16
  %w = zext i16 %xb to i32
{extra}
  %o = getelementptr inbounds i32, ptr addrspace(1) %out, i32 %t
  store i32 %w, ptr addrspace(1) %o, align 4
  %t64 = add i32 %t, 64
  %oe = getelementptr inbounds i32, ptr addrspace(1) %out, i32 %t64
  store i32 %e, ptr addrspace(1) %oe, align 4
  ret void
}}

{}"#,
        KERNEL_MD.replace("NAME", name)
    )
}

fn out_slot() -> ResourceSlot {
    ResourceSlot {
        stage: Stage::Kernel,
        kind: ResourceKind::Buffer,
        index: 0,
    }
}

/// Output buffer 0 of 128 zero words and 64 threads.
fn resources(plan: &Plan) -> Resources {
    let mut s = BTreeMap::new();
    s.insert(SpecKey::LocalSize(Axis::X), 64);
    s.insert(SpecKey::LocalSize(Axis::Y), 1);
    s.insert(SpecKey::LocalSize(Axis::Z), 1);
    s.insert(SpecKey::FlatLocalSize, 64);
    let mut push = vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)];
    for f in &plan.push.fields {
        match (f.field, f.loc) {
            (PushField::SlotBindOffset(_), PushLoc::Push { offset }) => {
                push[offset as usize..offset as usize + 4].fill(0)
            }
            other => panic!("unexpected push field {other:?}"),
        }
    }
    Resources {
        buffers: vec![Buffer {
            slot: Some(out_slot()),
            address: None,
            bytes: vec![0; 128 * 4],
        }],
        push,
        spec: s,
    }
}

/// The first 64 output words, in both scheduling orders (which must agree).
fn run_both(m: &TypedModule, res: &Resources) -> Vec<u32> {
    let mut outs = Vec::new();
    for order in [Order::Forward, Order::Reverse] {
        let cfg = Config {
            subgroup: SubgroupSize::W32,
            order,
            ..Config::default()
        };
        let out = run(m, &cfg, res.clone()).expect("run");
        outs.push(
            out.buffers[0]
                .as_chunks::<4>()
                .0
                .iter()
                .take(64)
                .map(|c| u32::from_le_bytes(*c))
                .collect::<Vec<u32>>(),
        );
    }
    assert_eq!(outs[0], outs[1], "scheduling orders disagree");
    outs.swap_remove(0)
}

/// Whether entry `entry` of `bc` on `prof` reads a `half` from its bits by the device's widening
/// (`PrecisionNote::NativeNanEncoding`, or `VkProperty::MetalNanMoves` on a device the interpreter
/// is not: `device_nan_bits`), whose reinterpretation of a NaN's bits is the device's.
fn native_halves(bc: &[u8], entry: &str, prof: &DeviceProfile) -> bool {
    device_nan_bits(&compile(bc, entry, prof).expect("compile"))
}

/// `got` against the bit patterns `want`. Where the module reads a `half` from its bits by the
/// device's widening (`noted`, [`native_halves`]), a NaN pattern need only read back as a NaN.
/// Elsewhere every pattern is exact.
fn assert_bits(noted: bool, got: &[u32], want: &[u32], what: &str) {
    let nan = |h: u32| h & 0x7C00 == 0x7C00 && h & 0x3FF != 0;
    if !noted {
        assert_eq!(got, want, "{what}");
        return;
    }
    assert_eq!(got.len(), want.len(), "{what}");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            g == w || (nan(*g) && nan(*w)),
            "{what}: lane {i}: {g:#x} for {w:#x}"
        );
    }
}

fn l3(bc: &[u8], entry: &str, prof: &DeviceProfile, what: &str) {
    let a = compile(bc, entry, prof).unwrap_or_else(|e| panic!("{what}: compile: {e}"));
    link_and_validate(&a, prof).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
}

/// The `Workgroup` variables the module declares.
fn workgroup_variables(m: &TypedModule) -> usize {
    m.globals()
        .iter()
        .filter(|g| g.class == StorageClass::Workgroup)
        .count()
}

/// The `half` array is one `Workgroup` variable of 64 held elements: plain typed stores and loads
/// (no arena, no word atomic), and every stored bit pattern reads back exactly.
#[test]
fn a_half_threadgroup_array_is_a_typed_workgroup_object() {
    let Some(bc) = assemble("halves", &halves("halves", "", "")) else {
        return;
    };
    for stem in PROFILES {
        let prof = p(stem);
        l3(&bc, "halves", &prof, stem);
        let (plan, m) = typed_module(&bc, "halves", &prof).expect("module");
        let local = &plan.families.access.local;
        assert_eq!(local.tg, None, "{stem}: no arena");
        assert_eq!(local.typed.len(), 1, "{stem}");
        let census = plan.census();
        for name in [
            "access.threadgroup.typed_store.clamped",
            "access.threadgroup.typed_load.clamped",
        ] {
            assert!(census.contains_key(name), "{stem}: {name} in {census:?}");
        }
        assert!(
            !census
                .keys()
                .any(|k| k.starts_with("access.threadgroup.store")),
            "{stem}: {census:?}"
        );
        assert_eq!(workgroup_variables(&m), 1, "{stem}");
        let want: Vec<u32> = (0..64u32).map(|t| pattern(63 - t)).collect();
        let noted = native_halves(&bc, "halves", &prof);
        assert_bits(noted, &run_both(&m, &resources(&plan)), &want, stem);
        // Metal's static length: the 128 bytes of the array.
        let a = compile(&bc, "halves", &prof).expect("compile");
        assert_eq!(a.kernel.static_threadgroup_bytes, 128, "{stem}");
    }
}

/// A `float` array beside the `half` one is a typed object too (an element is one typed access,
/// not a contained word), and the reported static length is Metal's layout of both.
#[test]
fn a_float_array_beside_the_half_one_is_typed_too() {
    let extra = r#"  %pf = getelementptr inbounds [64 x float], ptr addrspace(3) @f, i32 0, i32 %t
  store float 1.5, ptr addrspace(3) %pf, align 4
  %lf = load float, ptr addrspace(3) %pf, align 4
  %e = bitcast float %lf to i32"#;
    let Some(bc) = assemble(
        "mixed",
        &halves(
            "mixed",
            "@f = internal addrspace(3) global [64 x float] undef, align 4",
            extra,
        ),
    ) else {
        return;
    };
    for stem in PROFILES {
        let prof = p(stem);
        l3(&bc, "mixed", &prof, stem);
        let (plan, m) = typed_module(&bc, "mixed", &prof).expect("module");
        let local = &plan.families.access.local;
        assert_eq!(local.tg, None, "{stem}: no arena");
        assert_eq!(local.typed.len(), 2, "{stem}");
        assert_eq!(workgroup_variables(&m), 2, "{stem}");
        let want: Vec<u32> = (0..64u32).map(|t| pattern(63 - t)).collect();
        let noted = native_halves(&bc, "mixed", &prof);
        assert_bits(noted, &run_both(&m, &resources(&plan)), &want, stem);
        let a = compile(&bc, "mixed", &prof).expect("compile");
        assert_eq!(a.kernel.static_threadgroup_bytes, 384, "{stem}");
    }
}

/// Each structural condition that fails keeps the array in the arena, word-composed with its
/// sub-word stores' atomics, and the answer is the same.
#[test]
fn the_array_stays_in_the_arena_where_the_structure_does_not_allow_a_typed_object() {
    // A word load through a punned pointer: the array's accesses are not all of its element.
    let punned = r#"  %pw = getelementptr inbounds [64 x half], ptr addrspace(3) @h, i32 0, i32 2
  %e = load i32, ptr addrspace(3) %pw, align 4"#;
    // A pointer that may be either of two arrays (at run time always the written one).
    let merged = r#"  %far = icmp ugt i32 %t, 1000
  %sel = select i1 %far, ptr addrspace(3) @h2, ptr addrspace(3) @h
  %ls = load half, ptr addrspace(3) %sel, align 2
  %lb = bitcast half %ls to i16
  %e = zext i16 %lb to i32"#;
    // An integer threadgroup address.
    let literal =
        r#"  %e = load i32, ptr addrspace(3) inttoptr (i64 4 to ptr addrspace(3)), align 4"#;
    let cases = [
        ("punned", "", punned),
        (
            "merged",
            "@h2 = internal addrspace(3) global [64 x half] undef, align 2",
            merged,
        ),
        ("literal", "", literal),
    ];
    for (name, globals, extra) in cases {
        let Some(bc) = assemble(name, &halves(name, globals, extra)) else {
            return;
        };
        for stem in PROFILES {
            let prof = p(stem);
            let what = format!("{name} on {stem}");
            l3(&bc, name, &prof, &what);
            let (plan, m) = typed_module(&bc, name, &prof).expect("module");
            let local = &plan.families.access.local;
            assert!(local.tg.is_some(), "{what}: an arena");
            assert!(local.typed.is_empty(), "{what}");
            let census = plan.census();
            assert!(
                census.contains_key("access.threadgroup.store.checked"),
                "{what}: {census:?}"
            );
            let want: Vec<u32> = (0..64u32).map(|t| pattern(63 - t)).collect();
            let noted = native_halves(&bc, name, &prof);
            assert_bits(noted, &run_both(&m, &resources(&plan)), &want, &what);
        }
    }
}

/// A `[[threadgroup(0)]]` region beside the array keeps it in the arena: the encoder's length
/// must fit beside the static length Metal reports, which the binder checks.
#[test]
fn a_threadgroup_region_keeps_the_array_in_the_arena() {
    let text = r#"@h = internal addrspace(3) global [64 x half] undef, align 2

define void @region(ptr addrspace(1) %out, ptr addrspace(3) %a, i32 %t) {
  %p = getelementptr inbounds [64 x half], ptr addrspace(3) @h, i32 0, i32 %t
  store half 0xH3C00, ptr addrspace(3) %p, align 2
  %pa = getelementptr inbounds i32, ptr addrspace(3) %a, i32 %t
  store i32 %t, ptr addrspace(3) %pa, align 4
  tail call void @air.wg.barrier(i32 2, i32 1)
  %x = load half, ptr addrspace(3) %p, align 2
  %xb = bitcast half %x to i16
  %w = zext i16 %xb to i32
  %o = getelementptr inbounds i32, ptr addrspace(1) %out, i32 %t
  store i32 %w, ptr addrspace(1) %o, align 4
  ret void
}

declare void @air.wg.barrier(i32, i32)

!air.kernel = !{!0}
!0 = !{ptr @region, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 3, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"a"}
!5 = !{i32 2, !"air.thread_index_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"t"}
"#;
    let Some(bc) = assemble("region", text) else {
        return;
    };
    for stem in PROFILES {
        let prof = p(stem);
        l3(&bc, "region", &prof, stem);
        let (plan, _) = typed_module(&bc, "region", &prof).expect("module");
        let local = &plan.families.access.local;
        let tg = local.tg.as_ref().expect("an arena");
        assert_eq!(tg.static_size, 128, "{stem}");
        assert!(local.typed.is_empty(), "{stem}");
    }
}

/// A `half` array whose held form would pass the device's limit (32 KiB of halves, 64 KiB held)
/// stays in the arena, exactly the 32 KiB Metal takes.
#[test]
fn an_array_whose_held_form_passes_the_limit_stays_in_the_arena() {
    let text = format!(
        r#"@tile = internal addrspace(3) global [16384 x half] undef, align 2

define void @big(ptr addrspace(1) %out, i32 %t) {{
  %p = getelementptr inbounds [16384 x half], ptr addrspace(3) @tile, i32 0, i32 %t
  store half 0xH3C00, ptr addrspace(3) %p, align 2
  tail call void @air.wg.barrier(i32 2, i32 1)
  %x = load half, ptr addrspace(3) %p, align 2
  %xb = bitcast half %x to i16
  %w = zext i16 %xb to i32
  %o = getelementptr inbounds i32, ptr addrspace(1) %out, i32 %t
  store i32 %w, ptr addrspace(1) %o, align 4
  ret void
}}

{}"#,
        KERNEL_MD.replace("NAME", "big")
    );
    let Some(bc) = assemble("big", &text) else {
        return;
    };
    for stem in ["moltenvk-m3", "lavapipe"] {
        let prof = p(stem);
        l3(&bc, "big", &prof, stem);
        let (plan, _) = typed_module(&bc, "big", &prof).expect("module");
        let local = &plan.families.access.local;
        let tg = local.tg.as_ref().expect("an arena");
        assert_eq!(tg.static_size, 32768, "{stem}");
        assert!(local.typed.is_empty(), "{stem}");
    }
}
