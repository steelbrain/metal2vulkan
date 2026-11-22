//! The flat and folded compute variants end to end on the CPU interpreter. No device is opened.
//!
//! `icb_replay_grid_template` observes every grid builtin of its region (its threadgroup's size and
//! position, the nominal size, the grid in threadgroups and threads) and its neighbour in
//! threadgroup memory; `host_replay` runs a command list of it as the consumer records each
//! command's own dispatch, and a model of the source computes what Metal does.
//!
//! A kernel that reads only `thread_position_in_grid` and `threads_per_grid` (`flat`) compiles as a
//! flat variant (`LoweringInputs::flat_grid`, `VariantKey::flat_grid`), which binds as any kernel
//! does and reads its grid from the region record at its `ReplayRegionAddress`: a direct
//! `dispatchThreads:` of its plan, ragged or not, is one region that runs each grid thread once, as
//! Metal computes it. A kernel that observes its partition is refused under the input.
//!
//! The grid template, which observes its partition, compiles as a folded variant
//! (`VariantKey::folded_groups`), which states its key; a direct or indirect dispatch of its plan
//! is refused.
//!
//! `llvm-as` assembles the fixture; without it the tests skip visibly.

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::budget::Budget;
use crate::cache::VariantKey;
use crate::device::profile::BufferAccess;
use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::error::{Error, Feature, Needs};
use crate::library::Library;
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::sym::Stage;
use crate::pipeline::testing::{llvm_assemble, AssembleError};
use crate::pipeline::{
    ComputePipeline, ComputePipelineDescriptor, FunctionDescriptor, LoweringInputs, StageSource,
};

const HEADER: &str = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n";

/// `text` (an AIR module body) assembled, with the AIR target header.
fn assemble(name: &str, text: &str) -> Option<Vec<u8>> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "m2v-device-replay-{}-{}-{name}",
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

/// Every committed profile as a device, by stem.
fn devices() -> Vec<(String, Device)> {
    devices_with(|_| {})
}

/// Every committed profile as a device, by stem, with its description changed by `tweak`.
fn devices_with(tweak: impl Fn(&mut DeviceDescription)) -> Vec<(String, Device)> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../metal2vulkan-validation/profiles");
    let mut out: Vec<(String, Device)> = std::fs::read_dir(&dir)
        .expect("profiles")
        .map(|e| e.expect("entry").path())
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .map(|path| {
            let stem = path
                .file_stem()
                .expect("stem")
                .to_string_lossy()
                .into_owned();
            let text = std::fs::read_to_string(&path).expect("profile");
            let mut desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
            tweak(&mut desc);
            (
                stem,
                Device::from_queries(desc, DevicePolicy::RESIDENT_TABLES).expect("device"),
            )
        })
        .collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

/// Function `entry` of `bc` compiled under `inputs`, through the public API.
fn pipeline(
    dev: &Device,
    bc: &[u8],
    entry: &str,
    inputs: &LoweringInputs,
) -> Result<ComputePipeline, Error> {
    let lib = Library::from_air(bc).expect("library");
    let f = FunctionDescriptor::new(lib.function(entry).expect("function"));
    let compiled = dev.compile_function(&f, inputs, &Budget::standard())?;
    let d = ComputePipelineDescriptor::new(StageSource::Compiled(compiled));
    dev.compile_compute_pipeline(&d, &Budget::standard())
}

/// `flat(b0, b1, p, g)`: each thread adds `b0[0] + 3 p.x + 5 p.y + 7 p.z + 11 g.x + 13 g.y +
/// 17 g.z` to `b1[p.x + p.y g.x + p.z g.x g.y]` by a relaxed atomic whose result it does not
/// read, reading only `thread_position_in_grid` and `threads_per_grid`: it does not observe its
/// threadgroup partition, and a thread run twice adds twice. (The same addition as a load and a
/// store is [`flat_rmw`], which takes a lockstep barrier and so has no flat variant.)
const FLAT: &str = r#"define void @flat(ptr addrspace(1) noundef "air-buffer-no-alias" %b0, ptr addrspace(1) noundef "air-buffer-no-alias" %b1, <3 x i32> noundef %p, <3 x i32> noundef %g) {
entry:
  %base = load i32, ptr addrspace(1) %b0, align 4
  %px = extractelement <3 x i32> %p, i64 0
  %py = extractelement <3 x i32> %p, i64 1
  %pz = extractelement <3 x i32> %p, i64 2
  %gx = extractelement <3 x i32> %g, i64 0
  %gy = extractelement <3 x i32> %g, i64 1
  %gz = extractelement <3 x i32> %g, i64 2
  %a0 = mul i32 %px, 3
  %a1 = mul i32 %py, 5
  %a2 = mul i32 %pz, 7
  %a3 = mul i32 %gx, 11
  %a4 = mul i32 %gy, 13
  %a5 = mul i32 %gz, 17
  %s0 = add i32 %base, %a0
  %s1 = add i32 %s0, %a1
  %s2 = add i32 %s1, %a2
  %s3 = add i32 %s2, %a3
  %s4 = add i32 %s3, %a4
  %v = add i32 %s4, %a5
  %row = mul i32 %py, %gx
  %plane = mul i32 %gx, %gy
  %layer = mul i32 %pz, %plane
  %i0 = add i32 %px, %row
  %i = add i32 %i0, %layer
  %i64 = zext i32 %i to i64
  %o = getelementptr inbounds i32, ptr addrspace(1) %b1, i64 %i64
  %old = atomicrmw add ptr addrspace(1) %o, i32 %v monotonic, align 4
  ret void
}

!air.kernel = !{!0}
!air.version = !{!20}
!air.language_version = !{!21}
!0 = !{ptr @flat, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !6}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"b0"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"b1"}
!5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint3", !"air.arg_name", !"p"}
!6 = !{i32 3, !"air.threads_per_grid", !"air.arg_type_name", !"uint3", !"air.arg_name", !"g"}
!20 = !{i32 2, i32 8, i32 0}
!21 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

/// [`FLAT`] adding to `b1[p.x + p.y g.x + p.z g.x g.y]` by a load and a store.
fn flat_rmw() -> String {
    FLAT.replace(
        "  %old = atomicrmw add ptr addrspace(1) %o, i32 %v monotonic, align 4\n",
        "  %cur = load i32, ptr addrspace(1) %o, align 4\n  %n = add i32 %cur, %v\n  store i32 \
         %n, ptr addrspace(1) %o, align 4\n",
    )
}

/// Known gap: [`flat_rmw`], the flat kernel's addition as a plain load and store of `b1[p.x + p.y
/// g.x + p.z g.x g.y]`, takes a device lockstep barrier between them. The analysis proves two
/// lanes' words apart only through their index in the simdgroup or the threadgroup, and two lanes
/// of one simdgroup do meet at that index in a grid of more than 2^32 threads, whose 32-bit index
/// wraps (Metal bounds each of the grid's dimensions, not their product): Apple's lockstep then
/// runs both loads before either store. The barrier is the conservative and exact answer; it
/// observes the threadgroup partition, so the kernel has no flat variant and its GPU-encoded
/// commands are refused, where [`FLAT`]'s atomic addition has one.
#[test]
fn a_flat_read_modify_write_of_its_grid_index_has_no_flat_variant() {
    let Some(rmw) = assemble("flat_rmw", &flat_rmw()) else {
        return;
    };
    let mut key = VariantKey::new();
    key.flat_grid = true;
    let plain = key.lowering_inputs(Stage::Kernel);
    let mut ran = 0;
    for (stem, dev) in devices() {
        if dev.profile().fields().buffer_access != BufferAccess::Addressed {
            continue;
        }
        ran += 1;
        match pipeline(&dev, &rmw, "flat", &plain) {
            Err(Error::Unsupported(u)) => {
                assert_eq!(u.feature, Feature::GpuEncodedIndirectCommands, "{stem}");
                assert_eq!(u.needs, Needs::NotRepresentable, "{stem}");
            }
            other => panic!("{stem}: {:?}", other.map(|_| ())),
        }
    }
    assert!(ran > 0, "no profile with device addresses");
}

/// `lanes(b0, p, l)`: each thread stores its index in its simdgroup (`thread_index_in_simdgroup`)
/// at `b0[p]`, `p` its position in the grid: it observes its threadgroup partition through its
/// simdgroups alone, with no subgroup operation (its lanes are counted from its thread's index).
const LANES: &str = r#"define void @lanes(ptr addrspace(1) noundef "air-buffer-no-alias" %b0, i32 noundef %p, i32 noundef %l) {
entry:
  %i = zext i32 %p to i64
  %o = getelementptr inbounds i32, ptr addrspace(1) %b0, i64 %i
  store i32 %l, ptr addrspace(1) %o, align 4
  ret void
}

!air.kernel = !{!0}
!air.version = !{!20}
!air.language_version = !{!21}
!0 = !{ptr @lanes, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"b0"}
!4 = !{i32 1, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"p"}
!5 = !{i32 2, !"air.thread_index_in_simdgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"l"}
!20 = !{i32 2, i32 8, i32 0}
!21 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

/// [`LANES`] with a threadgroup barrier before its store: it observes its partition otherwise.
fn lanes_barrier() -> String {
    LANES
        .replace(
            "  store i32 %l,",
            "  call void @air.wg.barrier(i32 1, i32 1)\n  store i32 %l,",
        )
        .replace(
            "\n!air.kernel",
            "\ndeclare void @air.wg.barrier(i32, i32)\n\n!air.kernel",
        )
}

/// [`LANES`] storing the sum of its simdgroup's lanes (`simd_sum`) after a simdgroup barrier: it
/// observes its partition through its simdgroups alone, by subgroup operations.
fn lanes_sum() -> String {
    LANES
        .replace(
            "  store i32 %l,",
            "  call void @air.simdgroup.barrier(i32 1, i32 4)\n  %s = call i32 \
             @air.simd_sum.u.i32(i32 %l)\n  store i32 %s,",
        )
        .replace(
            "\n!air.kernel",
            "\ndeclare void @air.simdgroup.barrier(i32, i32)\ndeclare i32 \
             @air.simd_sum.u.i32(i32)\n\n!air.kernel",
        )
}

/// A kernel that observes its threadgroup partition through its simdgroups alone has no flat
/// variant of any threadgroups, and has one of the call's (`VariantKey::call_threadgroups`): a
/// lane's index in its simdgroup ([`LANES`]) on every profile with device addresses, and a
/// simdgroup barrier and reduction ([`lanes_sum`]) at least where the kernel's simdgroups are its
/// subgroups (full subgroups: `Direct`, `Cluster32`); elsewhere that barrier would be a workgroup
/// barrier or an emulated simdgroup's, and the variant is refused. The variant states
/// `RuntimeFact::FlatGrid` and `RuntimeFact::CallThreadgroups`. A kernel with a threadgroup
/// barrier has neither variant, and the input is refused without the flat grid and with the
/// replay's bindings.
#[test]
fn a_flat_grid_of_the_call_s_threadgroups_admits_a_kernel_of_simdgroups() {
    let (Some(lanes), Some(barrier), Some(sum)) = (
        assemble("lanes", LANES),
        assemble("lanes_barrier", &lanes_barrier()),
        assemble("lanes_sum", &lanes_sum()),
    ) else {
        return;
    };
    let mut key = VariantKey::new();
    key.flat_grid = true;
    let any = key.lowering_inputs(Stage::Kernel);
    key.call_threadgroups = true;
    let call = key.lowering_inputs(Stage::Kernel);
    let mut alone = LoweringInputs::new();
    alone.call_threadgroups = true;
    let mut replayed = call.clone();
    replayed.replay_bindings = true;
    let stated = |pipe: &ComputePipeline| {
        [RuntimeFact::FlatGrid, RuntimeFact::CallThreadgroups]
            .map(|fact| pipe.dependencies.iter().any(|d| d.fact == fact))
    };
    let (mut ran, mut sums) = (0, 0);
    for (stem, dev) in devices() {
        if dev.profile().fields().buffer_access != BufferAccess::Addressed {
            continue;
        }
        ran += 1;
        let refused = |r: Result<ComputePipeline, Error>, needs: Needs, what: &str| match r {
            Err(Error::Unsupported(u)) => {
                assert_eq!(
                    u.feature,
                    Feature::GpuEncodedIndirectCommands,
                    "{stem} {what}"
                );
                assert_eq!(u.needs, needs, "{stem} {what}");
            }
            other => panic!("{stem} {what}: {:?}", other.map(|_| ())),
        };
        for (bc, what) in [(&lanes, "lanes"), (&sum, "sum")] {
            refused(
                pipeline(&dev, bc, "lanes", &any),
                Needs::NotRepresentable,
                what,
            );
            refused(
                pipeline(&dev, bc, "lanes", &alone),
                Needs::NotImplemented,
                what,
            );
            refused(
                pipeline(&dev, bc, "lanes", &replayed),
                Needs::NotImplemented,
                what,
            );
        }
        let observed = pipeline(&dev, &barrier, "lanes", &call);
        refused(observed, Needs::NotRepresentable, "barrier");
        let pipe = pipeline(&dev, &lanes, "lanes", &call).unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert_eq!(stated(&pipe), [true, true], "{stem}");
        let lib = Library::from_air(&sum).expect("library");
        let f = FunctionDescriptor::new(lib.function("lanes").expect("function"));
        let base = dev
            .compile_function(&f, &LoweringInputs::new(), &Budget::standard())
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        match pipeline(&dev, &sum, "lanes", &call) {
            Ok(pipe) => {
                assert_eq!(stated(&pipe), [true, true], "{stem}");
                sums += 1;
            }
            Err(e) if base.requires_full_subgroups() => panic!("{stem}: {e}"),
            refusal => refused(refusal, Needs::NotImplemented, "sum"),
        }
    }
    assert!(ran > 0, "no profile with device addresses");
    assert!(sums > 0, "no profile ran the simdgroup sum flat");
}
