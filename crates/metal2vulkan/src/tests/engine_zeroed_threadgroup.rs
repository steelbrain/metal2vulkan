//! Threadgroup memory before the program writes it (`device::profile::ThreadgroupStart`).
//!
//! - Through `Device::from_queries` every `Workgroup` variable that holds a function's threadgroup
//!   memory has a null initializer: the arena of a declared array and a `[[threadgroup(0)]]`
//!   region, whose length specialization gives, and a typed threadgroup object. A thread that
//!   reads its words before any thread writes them reads zero on the CPU interpreter, in both
//!   scheduling orders and in each of two threadgroups, and the module passes `spirv-val`.
//! - The emulated-SIMD-group variant of the same kernel lowers under the zero form: its arena is
//!   null-initialized and its exchange scratch, which the emulation writes before it reads, is not.
//! - A profile without `shaderZeroInitializeWorkgroupMemory` refuses such a function through
//!   `Device::from_queries` (`Unsupported{ZeroedThreadgroupMemory}`).
//!
//! `llvm-as` assembles the kernels; without it the tests skip visibly.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::device::profile::DeviceProfile;
use crate::device::requirements::VkFeature;
use crate::device::{Device, DevicePolicy};
use crate::error::{Error, Feature};
use crate::lower::access::local::Home;
use crate::lower::select::{LowerInputs, Plan};
use crate::pipeline::interface::PushLoc;
use crate::pipeline::sym::{Axis, PushField, ResourceKind, ResourceSlot, SpecKey, Stage};
use crate::pipeline::testing::{
    compile, compile_with, description, link_and_validate, llvm_assemble, typed_module,
    typed_module_with, AssembleError,
};
use crate::spirv::interp::{run, Buffer, Config, InterpError, Order, Resources};
use crate::spirv::ir::{ConstKind, Constant, GlobalKind, StorageClass, TypedModule};

/// Profiles of four drivers, each with `shaderZeroInitializeWorkgroupMemory`.
const PROFILES: [&str; 4] = [
    "moltenvk-m3",
    "lavapipe",
    "amd-780m-radv",
    "nvidia-rtx50-linux",
];

/// The threads of a threadgroup, and the words of each threadgroup object.
const THREADS: u32 = 64;

const HEADER: &str = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n";

fn assemble(name: &str, text: &str) -> Option<Vec<u8>> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "m2v-tg-zeroed-{}-{}-{name}",
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

/// `tg_unwritten`: thread `t` loads word `t` of the declared `[64 x i32]` array `@g` and of the
/// `[[threadgroup(0)]]` region `%a` before any thread stores; then (with `simd`) `simd_sum(t)`; a
/// barrier; each stores `t + 1` at both words. `out[t]` is the array's word, `out[64 + t]` the
/// region's and `out[128 + t]` the sum (0 without `simd`). The region beside it keeps the array in
/// the arena, whose length is a specialization operation.
fn unwritten(simd: bool) -> String {
    let (sum, declare) = match simd {
        true => (
            "  %s = call i32 @air.simd_sum.u.i32(i32 %t)\n",
            "declare i32 @air.simd_sum.u.i32(i32) #0\n",
        ),
        false => ("  %s = add i32 0, 0\n", ""),
    };
    format!(
        "@g = internal unnamed_addr addrspace(3) global [64 x i32] undef, align 4\n\n\
define void @tg_unwritten(ptr addrspace(3) noundef %a, ptr addrspace(1) noundef %out, i32 noundef %t) {{\n\
entry:\n\
  %tz = zext i32 %t to i64\n\
  %pg = getelementptr inbounds [64 x i32], ptr addrspace(3) @g, i64 0, i64 %tz\n\
  %x = load i32, ptr addrspace(3) %pg, align 4\n\
  %pa = getelementptr inbounds i32, ptr addrspace(3) %a, i64 %tz\n\
  %y = load i32, ptr addrspace(3) %pa, align 4\n\
{sum}\
  tail call void @air.wg.barrier(i32 2, i32 1) #0\n\
  %t1 = add i32 %t, 1\n\
  store i32 %t1, ptr addrspace(3) %pg, align 4\n\
  store i32 %t1, ptr addrspace(3) %pa, align 4\n\
  %po = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %tz\n\
  store i32 %x, ptr addrspace(1) %po, align 4\n\
  %t64 = add i64 %tz, 64\n\
  %pr = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %t64\n\
  store i32 %y, ptr addrspace(1) %pr, align 4\n\
  %t128 = add i64 %tz, 128\n\
  %ps = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %t128\n\
  store i32 %s, ptr addrspace(1) %ps, align 4\n\
  ret void\n\
}}\n\n\
declare void @air.wg.barrier(i32, i32) #0\n\
{declare}\n\
attributes #0 = {{ convergent mustprogress nounwind willreturn }}\n\n\
!air.kernel = !{{!0}}\n!air.version = !{{!6}}\n!air.language_version = !{{!7}}\n\
!0 = !{{ptr @tg_unwritten, !1, !2}}\n!1 = !{{}}\n!2 = !{{!3, !4, !5}}\n\
!3 = !{{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", !\"air.address_space\", i32 3, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"a\"}}\n\
!4 = !{{i32 1, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"out\"}}\n\
!5 = !{{i32 2, !\"air.thread_index_in_threadgroup\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"t\"}}\n\
!6 = !{{i32 2, i32 8, i32 0}}\n!7 = !{{!\"Metal\", i32 4, i32 0, i32 0}}\n"
    )
}

/// `tg_typed`: thread `t` loads element `t` of the declared `[64 x half]` array `@h` before any
/// thread stores, a barrier, and stores `t` there; `out[t]` is the loaded `half`'s bits. Its
/// every access is a plain load or store of its element, so the array is a typed threadgroup
/// object of its own (`memory::plan::SharedPlan`).
const TYPED: &str = r#"@h = internal unnamed_addr addrspace(3) global [64 x half] undef, align 2

define void @tg_typed(ptr addrspace(1) noundef %out, i32 noundef %t) {
entry:
  %tz = zext i32 %t to i64
  %ph = getelementptr inbounds [64 x half], ptr addrspace(3) @h, i64 0, i64 %tz
  %x = load half, ptr addrspace(3) %ph, align 2
  tail call void @air.wg.barrier(i32 2, i32 1) #0
  %f = uitofp i32 %t to half
  store half %f, ptr addrspace(3) %ph, align 2
  %b = bitcast half %x to i16
  %w = zext i16 %b to i32
  %po = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %tz
  store i32 %w, ptr addrspace(1) %po, align 4
  ret void
}

declare void @air.wg.barrier(i32, i32) #0

attributes #0 = { convergent mustprogress nounwind willreturn }

!air.kernel = !{!0}
!air.version = !{!5}
!air.language_version = !{!6}
!0 = !{ptr @tg_typed, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!4 = !{i32 1, !"air.thread_index_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"t"}
!5 = !{i32 2, i32 8, i32 0}
!6 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

/// `stem` through `Device::from_queries`.
fn queried(stem: &str) -> DeviceProfile {
    let desc = description(stem).expect("description");
    Device::from_queries(desc, DevicePolicy::default())
        .expect("device")
        .profile()
        .clone()
}

/// Each `Workgroup` variable the module declares: whether it has an initializer, and whether that
/// initializer is a null.
fn workgroup_inits(m: &TypedModule) -> Vec<(bool, bool)> {
    m.globals()
        .iter()
        .filter(|g| g.class == StorageClass::Workgroup && g.kind == GlobalKind::Plain)
        .map(|g| match g.init {
            Some(c) => {
                let kind = &m.consts().get(c).expect("constant").kind;
                (
                    true,
                    matches!(kind, ConstKind::Plain(Constant::Null { .. })),
                )
            }
            None => (false, false),
        })
        .collect()
}

/// Output buffer 0 of `words` zero words, threadgroups of [`THREADS`], and `spec` besides.
fn resources(plan: &Plan, words: usize, spec: &[(SpecKey, u64)]) -> Resources {
    let mut s = BTreeMap::new();
    s.insert(SpecKey::LocalSize(Axis::X), u64::from(THREADS));
    s.insert(SpecKey::LocalSize(Axis::Y), 1);
    s.insert(SpecKey::LocalSize(Axis::Z), 1);
    s.insert(SpecKey::FlatLocalSize, u64::from(THREADS));
    s.extend(spec.iter().copied());
    // Zero push bytes: the output is bound at its start, and the dispatch is one region at the
    // grid's origin.
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
    Resources {
        buffers: vec![Buffer {
            slot: Some(ResourceSlot {
                stage: Stage::Kernel,
                kind: ResourceKind::Buffer,
                index: 0,
            }),
            address: None,
            bytes: vec![0; words * 4],
        }],
        push,
        spec: s,
    }
}

/// Two threadgroups' run of `m`, in `order`.
fn run_in(m: &TypedModule, res: &Resources, order: Order) -> Result<Vec<u32>, InterpError> {
    let cfg = Config {
        workgroups: [2, 1, 1],
        order,
        ..Config::default()
    };
    let out = run(m, &cfg, res.clone())?;
    Ok(out.buffers[0]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect())
}

/// The region's length: 64 words.
const REGION: [(SpecKey, u64); 1] = [(SpecKey::TgLength(0), 4 * THREADS as u64)];

/// Through `from_queries` the arena of a declared array beside a region, and a typed threadgroup
/// object, are null-initialized, and every word read before the program writes it reads zero, in
/// both scheduling orders; the modules pass `spirv-val`.
#[test]
fn threadgroup_memory_reads_zero_until_the_program_writes_it_through_from_queries() {
    let Some(arena) = assemble("tg_unwritten", &unwritten(false)) else {
        return;
    };
    let Some(typed) = assemble("tg_typed", TYPED) else {
        return;
    };
    for stem in PROFILES {
        let prof = queried(stem);
        assert!(
            prof.fields()
                .enabled
                .features
                .contains(&VkFeature::ShaderZeroInitializeWorkgroupMemory),
            "{stem}"
        );
        for (bc, entry, words, spec) in [
            (&arena, "tg_unwritten", 192, &REGION[..]),
            (&typed, "tg_typed", 64, &[][..]),
        ] {
            let what = format!("{entry} on {stem}");
            let a = compile(bc, entry, &prof).unwrap_or_else(|e| panic!("{what}: {e}"));
            link_and_validate(&a, &prof).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            let (plan, m) =
                typed_module(bc, entry, &prof).unwrap_or_else(|e| panic!("{what}: {e}"));
            let local = &plan.families.access.local;
            assert!(local.zeroed, "{what}");
            if entry == "tg_typed" {
                assert!(
                    local
                        .typed
                        .iter()
                        .any(|o| matches!(o.home, Home::Shared(_))),
                    "{what}: no typed threadgroup object"
                );
            } else {
                assert!(local.tg.as_ref().is_some_and(|tg| !tg.regions.is_empty()));
            }
            assert_eq!(workgroup_inits(&m), vec![(true, true)], "{what}");
            let res = resources(&plan, words, spec);
            for order in [Order::Forward, Order::Reverse] {
                let out = run_in(&m, &res, order).unwrap_or_else(|e| panic!("{what}: {e:?}"));
                assert_eq!(out, vec![0; words], "{what}, {order:?}");
            }
        }
    }
}

/// The emulated-SIMD-group variant of a kernel whose arena holds a declared array beside a
/// region lowers through `from_queries`: the arena, its length a specialization operation, is
/// null-initialized, and the exchange scratch, of a length the threadgroup's size gives, has no
/// initializer. A null of the scratch's type would be one too, so the scratch staying
/// uninitialized is the translator's choice (it writes every word it reads), not the builder's.
#[test]
fn the_emulated_variant_zeroes_its_arena_and_not_its_scratch() {
    let Some(bc) = assemble("tg_unwritten_simd", &unwritten(true)) else {
        return;
    };
    let mut inputs = LowerInputs::new();
    inputs.emulated_simdgroups = true;
    for stem in PROFILES {
        let prof = queried(stem);
        let what = format!("tg_unwritten's emulated variant on {stem}");
        let a = compile_with(&bc, "tg_unwritten", &prof, &inputs)
            .unwrap_or_else(|e| panic!("{what}: {e}"));
        link_and_validate(&a, &prof).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
        let (plan, m) = typed_module_with(&bc, "tg_unwritten", &prof, &inputs)
            .unwrap_or_else(|e| panic!("{what}: {e}"));
        assert_eq!(
            plan.simd.1,
            crate::device::subgroup::Tier::Emulated,
            "{what}"
        );
        let mut inits = workgroup_inits(&m);
        inits.sort_unstable();
        assert_eq!(inits, vec![(false, false), (true, true)], "{what}");
    }
}

/// A profile without `shaderZeroInitializeWorkgroupMemory` refuses a function that uses
/// threadgroup memory through `from_queries`, typed.
#[test]
fn a_profile_without_zeroed_workgroup_memory_refuses_threadgroup_memory_through_from_queries() {
    let Some(bc) = assemble("tg_unwritten", &unwritten(false)) else {
        return;
    };
    const FLOOR: &str = "vk12-floor";
    let prof = queried(FLOOR);
    assert!(!prof
        .fields()
        .enabled
        .features
        .contains(&VkFeature::ShaderZeroInitializeWorkgroupMemory));
    match compile(&bc, "tg_unwritten", &prof) {
        Err(Error::Unsupported(u)) => assert_eq!(u.feature, Feature::ZeroedThreadgroupMemory),
        other => panic!("{other:?}"),
    }
}
