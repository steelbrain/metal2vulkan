//! Whether a kernel's module is inert (`ModuleStats::inert`, `spirv::verify::effects`): no loop,
//! and no write but to its invocations' own and its threadgroups' memory. A kernel that does
//! nothing, one that writes only threadgroup or private memory, and one that changes threadgroup
//! memory atomically are inert; one that stores or writes atomically to a device buffer is not,
//! and neither is one that loops, though it writes only threadgroup memory. The compiled artifact
//! states it and its encoding keeps it. `llvm-as` assembles the module; without it the test skips
//! visibly.

use crate::device::{Device, DevicePolicy};
use crate::library::CompiledFunction;
use crate::pipeline::testing::{
    compile, description, link_and_validate, llvm_assemble, AssembleError, Scratch,
};

const PROFILES: [&str; 4] = [
    "amd-rdna3-radv",
    "nvidia-rtx5080-laptop",
    "lavapipe",
    "moltenvk-m3",
];

/// Each kernel and whether its module is inert.
const KERNELS: [(&str, bool); 7] = [
    ("inert_empty", true),
    ("inert_threadgroup_store", true),
    ("inert_private_store", true),
    ("inert_threadgroup_atomic", true),
    ("writes_device_store", false),
    ("writes_device_atomic", false),
    ("loops_threadgroup_only", false),
];

/// A kernel of the buffer `%0` and the thread's grid position `%1` whose body ends in `ret void`.
fn kernel(name: &str, body: &str) -> String {
    format!(
        "define void @{name}(ptr addrspace(1) noundef \"air-buffer-no-alias\" %0, i32 noundef %1) {{
entry:
{body}  ret void
}}

"
    )
}

/// `%p`, the word of `@tg` the thread's position picks (modulo 64).
const TG_WORD: &str = "  %i = and i32 %1, 63
  %iw = zext i32 %i to i64
  %p = getelementptr inbounds [64 x i32], ptr addrspace(3) @tg, i64 0, i64 %iw
";

fn module() -> String {
    let mut s = String::from(
        "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n@tg = internal addrspace(3) global [64 x i32] undef, align 4\n\n",
    );
    s.push_str(&kernel("inert_empty", ""));
    s.push_str(&kernel(
        "inert_threadgroup_store",
        &format!("{TG_WORD}  store i32 %1, ptr addrspace(3) %p, align 4\n"),
    ));
    s.push_str(&kernel(
        "inert_private_store",
        "  %arr = alloca [4 x i32], align 4
  %i = and i32 %1, 3
  %iw = zext i32 %i to i64
  %p = getelementptr inbounds [4 x i32], ptr %arr, i64 0, i64 %iw
  store i32 %1, ptr %p, align 4
  %v = load i32, ptr %p, align 4
  %q = getelementptr inbounds [4 x i32], ptr %arr, i64 0, i64 0
  store i32 %v, ptr %q, align 4
",
    ));
    s.push_str(&kernel(
        "inert_threadgroup_atomic",
        &format!("{TG_WORD}  %o = atomicrmw add ptr addrspace(3) %p, i32 1 monotonic, align 4\n"),
    ));
    s.push_str(&kernel(
        "writes_device_store",
        "  %iw = zext i32 %1 to i64
  %p = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %iw
  store i32 %1, ptr addrspace(1) %p, align 4
",
    ));
    s.push_str(&kernel(
        "writes_device_atomic",
        "  %o = atomicrmw add ptr addrspace(1) %0, i32 1 monotonic, align 4\n",
    ));
    s.push_str(
        "define void @loops_threadgroup_only(ptr addrspace(1) noundef \"air-buffer-no-alias\" %0, i32 noundef %1) {
entry:
  %n = load i32, ptr addrspace(1) %0, align 4
  %none = icmp eq i32 %n, 0
  br i1 %none, label %done, label %body
body:
  %i = phi i32 [ 0, %entry ], [ %i.next, %body ]
  %k = and i32 %i, 63
  %kw = zext i32 %k to i64
  %p = getelementptr inbounds [64 x i32], ptr addrspace(3) @tg, i64 0, i64 %kw
  store i32 %i, ptr addrspace(3) %p, align 4
  %i.next = add nuw i32 %i, 1
  %more = icmp ult i32 %i.next, %n
  br i1 %more, label %body, label %done
done:
  ret void
}

",
    );
    let refs: Vec<String> = (0..KERNELS.len()).map(|i| format!("!{}", 20 + i)).collect();
    s.push_str(&format!(
        "!llvm.module.flags = !{{!0, !1}}\n!air.kernel = !{{{}}}\n!air.version = !{{!17}}\n!air.language_version = !{{!18}}\n\n",
        refs.join(", ")
    ));
    s.push_str("!0 = !{i32 7, !\"air.max_device_buffers\", i32 31}\n!1 = !{i32 7, !\"air.max_threadgroup_buffers\", i32 31}\n!10 = !{}\n!11 = !{!12, !13}\n!12 = !{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"buf\"}\n!13 = !{i32 1, !\"air.thread_position_in_grid\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"gid\"}\n!17 = !{i32 2, i32 8, i32 0}\n!18 = !{!\"Metal\", i32 4, i32 0, i32 0}\n");
    for (i, (k, _)) in KERNELS.iter().enumerate() {
        s.push_str(&format!("!{} = !{{ptr @{k}, !10, !11}}\n", 20 + i));
    }
    s
}

/// Assemble the module, or `None` (skipped visibly) without `llvm-as`.
fn assemble() -> Option<Vec<u8>> {
    let scratch = Scratch::new("m2v-inert");
    let ll = scratch.path().join("inert.ll");
    std::fs::write(&ll, module()).expect("write");
    match llvm_assemble(&ll) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(t)) => {
            eprintln!("SKIPPED: {t}");
            None
        }
        Err(AssembleError::Rejected(e)) => panic!("llvm-as rejected the module: {e}"),
    }
}

/// Each kernel's module is inert exactly where it loops nowhere and writes no memory but its
/// invocations' own and its threadgroups': the artifact states it, its module links and validates,
/// and the encoded artifact decodes to the same answer.
#[test]
fn a_kernel_is_inert_where_it_loops_nowhere_and_writes_only_its_threadgroups_memory() {
    let Some(bitcode) = assemble() else {
        return;
    };
    for stem in PROFILES {
        let device = Device::new(
            description(stem).expect("description"),
            DevicePolicy::default(),
        )
        .expect("device");
        let p = device.profile();
        for (entry, inert) in KERNELS {
            let what = format!("{entry} on {stem}");
            let a = compile(&bitcode, entry, p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
            assert_eq!(a.stats.inert, inert, "{what}: {:?}", a.stats);
            assert_eq!(
                a.stats.loops > 0,
                entry.starts_with("loops"),
                "{what}: {:?}",
                a.stats
            );
            link_and_validate(&a, p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            let bytes = CompiledFunction::new(a).to_bytes();
            let decoded = CompiledFunction::from_bytes(&bytes, &device).expect("decode");
            assert_eq!(decoded.module_stats().inert, inert, "{what}: decoded");
        }
    }
}
