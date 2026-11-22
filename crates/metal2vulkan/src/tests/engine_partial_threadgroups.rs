//! A kernel's partial-threadgroups variant (`LoweringInputs::partial_threadgroups`): it is
//! refused with another form of the kernel's (a flat grid, folded groups, a replay's bindings, or
//! emulated simdgroups). No GPU is used.
//!
//! `llvm-as` assembles the kernels; a missing tool skips visibly.

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::budget::Budget;
use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::error::{Error, Feature};
use crate::library::Library;
use crate::pipeline::testing::{llvm_assemble, typed_module_with, AssembleError};
use crate::pipeline::{
    ComputePipeline, ComputePipelineDescriptor, FunctionDescriptor, LoweringInputs, StageSource,
};
use crate::spirv::ir::TypedModule;

/// `text` assembled by `llvm-as`, or `None` (printed as skipped) where the tool is missing.
fn assemble_text(name: &str, text: &str) -> Option<Vec<u8>> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "m2v-binder-{}-{}-{name}",
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

/// The device of the committed profile `stem`, its description changed by `tweak`, under the
/// resident tables.
fn device(stem: &str, tweak: impl FnOnce(&mut DeviceDescription)) -> Device {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles")
        .join(format!("{stem}.json"));
    let text = std::fs::read_to_string(path).expect("profile");
    let mut desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    tweak(&mut desc);
    Device::from_queries(desc, DevicePolicy::RESIDENT_TABLES).expect("device")
}

const HEADER: &str = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n";

/// `neighbour`: `a[t] = 3t + 1; threadgroup barrier; out[pos] = n << 24 | a[(t + 1) mod n] << 12 |
/// simd_sum(t)`, `t` the thread's `thread_index_in_threadgroup` and `n` its threadgroup's
/// `threads_per_threadgroup`; without `simd`, the sum's word is 0.
fn neighbour(simd: bool) -> String {
    let (sum, declare) = match simd {
        true => (
            "  %s = call i32 @air.simd_sum.u.i32(i32 %t)\n",
            "declare i32 @air.simd_sum.u.i32(i32) #0\n",
        ),
        false => ("  %s = add i32 0, 0\n", ""),
    };
    format!(
        "{HEADER}@a = internal unnamed_addr addrspace(3) global [64 x i32] undef, align 4\n\n\
define void @neighbour(ptr addrspace(1) noundef %out, i32 noundef %pos, i32 noundef %t, i32 noundef %n) {{\n\
entry:\n\
  %tz = zext i32 %t to i64\n\
  %pa = getelementptr inbounds [64 x i32], ptr addrspace(3) @a, i64 0, i64 %tz\n\
  %v0 = mul i32 %t, 3\n\
  %v1 = add i32 %v0, 1\n\
  store i32 %v1, ptr addrspace(3) %pa, align 4\n\
  tail call void @air.wg.barrier(i32 2, i32 1) #0\n\
  %t1 = add i32 %t, 1\n\
  %u = urem i32 %t1, %n\n\
  %uz = zext i32 %u to i64\n\
  %pu = getelementptr inbounds [64 x i32], ptr addrspace(3) @a, i64 0, i64 %uz\n\
  %got = load i32, ptr addrspace(3) %pu, align 4\n\
{sum}\
  %hi = shl i32 %n, 24\n\
  %mid = shl i32 %got, 12\n\
  %hm = or i32 %hi, %mid\n\
  %w = or i32 %hm, %s\n\
  %pz = zext i32 %pos to i64\n\
  %po = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %pz\n\
  store i32 %w, ptr addrspace(1) %po, align 4\n\
  ret void\n\
}}\n\n\
declare void @air.wg.barrier(i32, i32) #0\n\
{declare}\n\
attributes #0 = {{ convergent mustprogress nounwind willreturn }}\n\n\
!air.kernel = !{{!0}}\n!air.version = !{{!7}}\n!air.language_version = !{{!8}}\n\
!0 = !{{ptr @neighbour, !1, !2}}\n!1 = !{{}}\n!2 = !{{!3, !4, !5, !6}}\n\
!3 = !{{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"out\"}}\n\
!4 = !{{i32 1, !\"air.thread_position_in_grid\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"pos\"}}\n\
!5 = !{{i32 2, !\"air.thread_index_in_threadgroup\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"t\"}}\n\
!6 = !{{i32 3, !\"air.threads_per_threadgroup\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"n\"}}\n\
!7 = !{{i32 2, i32 8, i32 0}}\n!8 = !{{!\"Metal\", i32 4, i32 0, i32 0}}\n"
    )
}

/// The compute pipeline of `entry` in `bc` on `dev` under `inputs`, and the typed module the
/// interpreter runs.
fn pipeline(
    dev: &Device,
    bc: &[u8],
    entry: &str,
    inputs: &LoweringInputs,
) -> Result<(ComputePipeline, TypedModule), Error> {
    let lib = Library::from_air(bc).expect("library");
    let f = FunctionDescriptor::new(lib.function(entry).expect("function"));
    let compiled = dev.compile_function(&f, inputs, &Budget::standard())?;
    let d = ComputePipelineDescriptor::new(StageSource::Compiled(compiled));
    let pipe = dev.compile_compute_pipeline(&d, &Budget::standard())?;
    let (_, m) = typed_module_with(bc, entry, dev.profile(), inputs)?;
    Ok((pipe, m))
}

/// The variant is refused with another form of the kernel's: a flat grid, folded groups, a
/// replay's bindings, or emulated simdgroups.
#[test]
fn partial_threadgroups_are_refused_with_another_form() {
    let Some(bc) = assemble_text("neighbour_forms", &neighbour(false)) else {
        return;
    };
    let dev = device("moltenvk-m3", |_| {});
    let forms: [fn(&mut LoweringInputs); 4] = [
        |i| i.flat_grid = true,
        |i| i.folded_groups = true,
        |i| i.replay_bindings = true,
        |i| i.emulated_simdgroups = true,
    ];
    for (n, form) in forms.into_iter().enumerate() {
        let mut inputs = LoweringInputs::new();
        inputs.partial_threadgroups = true;
        form(&mut inputs);
        match pipeline(&dev, &bc, "neighbour", &inputs) {
            Err(Error::Unsupported(u)) => {
                assert_eq!(u.feature, Feature::PartialThreadgroups, "form {n}");
            }
            other => panic!("form {n}: {:?}", other.map(|_| ())),
        }
    }
}
