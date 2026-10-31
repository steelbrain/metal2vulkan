//! Whole arrays and structs loaded from and stored to buffers, end to end.
//!
//! Metal's own linker passes a field of a buffer's struct to a function by value as one aggregate
//! load (a Core Image kernel's linked `ciKernelMain` loads its `float4x4` transform so), and a
//! pointer cast to its own type beside it. The kernel below is that shape, written for this test:
//! the rows of a `constant` struct's `float4x4` are loaded whole through a byte offset and a
//! same-type cast, passed by value to a function that adds the struct's `float4` to one of them,
//! and stored whole into the output buffer after that sum. It translates on the tested profiles
//! and runs on the CPU interpreter, which computes what Metal does. `llvm-as` assembles it;
//! without it the test skips visibly.

use crate::device::DevicePolicy;
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage};
use crate::pipeline::testing::{compile_kept, llvm_assemble, profile, AssembleError, Scratch};
use crate::spirv::interp::{run, Buffer, Config, Resources};

/// `struct P { float4 rect; float4x4 m; }`, `kernel k(constant P &p [[buffer(0)]], device float4
/// *out [[buffer(1)]])`: `out[0] = row(p.m, 1) + p.rect`, then `p.m`'s rows into `out[1..5]`.
const KERNEL: &str = r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v29-apple-macosx27.0.0"

define internal fastcc <4 x float> @row(<4 x float> %r, [4 x <4 x float>] %m) unnamed_addr #0 {
  %a = extractvalue [4 x <4 x float>] %m, 1
  %s = fadd fast <4 x float> %a, %r
  ret <4 x float> %s
}

define void @k(ptr addrspace(2) noundef readonly align 16 captures(none) dereferenceable(80) "air-buffer-no-alias" %p, ptr addrspace(1) noundef "air-buffer-no-alias" %out) local_unnamed_addr #0 {
  %at = getelementptr i8, ptr addrspace(2) %p, i64 16
  %same = bitcast ptr addrspace(2) %at to ptr addrspace(2)
  %m = load [4 x <4 x float>], ptr addrspace(2) %same, align 16
  %r = load <4 x float>, ptr addrspace(2) %p, align 16
  %sum = tail call fastcc <4 x float> @row(<4 x float> %r, [4 x <4 x float>] %m)
  store <4 x float> %sum, ptr addrspace(1) %out, align 16
  %rows = getelementptr i8, ptr addrspace(1) %out, i64 16
  store [4 x <4 x float>] %m, ptr addrspace(1) %rows, align 16
  ret void
}

attributes #0 = { nounwind }

!air.kernel = !{!0}
!air.compile_options = !{!5, !6}
!air.version = !{!7}
!air.language_version = !{!8}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.buffer_size", i32 80, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !9, !"air.arg_type_size", i32 80, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"P", !"air.arg_name", !"p"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"out"}
!5 = !{!"air.compile.denorms_disable"}
!6 = !{!"air.compile.fast_math_enable"}
!7 = !{i32 2, i32 9, i32 0}
!8 = !{!"Metal", i32 4, i32 1, i32 0}
!9 = !{i32 0, i32 16, i32 0, !"float4", !"rect", i32 16, i32 64, i32 0, !"float4x4", !"m"}
"#;

fn buffer(index: u32, bytes: Vec<u8>) -> Buffer {
    Buffer {
        slot: Some(ResourceSlot {
            stage: Stage::Kernel,
            kind: ResourceKind::Buffer,
            index,
        }),
        address: None,
        bytes,
    }
}

fn floats(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// The aggregate load and store move every member, and the by-value argument reads the right one,
/// on each profile.
#[test]
fn an_aggregate_buffer_access_moves_every_member() {
    let scratch = Scratch::new("m2v-aggregate-access");
    let path = scratch.path().join("k.ll");
    std::fs::write(&path, KERNEL).expect("write");
    let bitcode = match llvm_assemble(&path) {
        Ok(b) => b,
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            return;
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected the kernel: {why}"),
    };
    // `rect`, then the 16 floats of `m`, row by row.
    let rect = [100.0f32, 200.0, 300.0, 400.0];
    let m: Vec<f32> = (1..=16).map(|i| i as f32).collect();
    let mut p = rect.to_vec();
    p.extend(&m);
    let mut want = vec![105.0f32, 206.0, 307.0, 408.0];
    want.extend(&m);
    for stem in ["moltenvk-m3", "lavapipe", "vk12-floor"] {
        let profile = profile(stem, DevicePolicy::default()).expect("profile");
        let (_, _, module) =
            compile_kept(&bitcode, "k", &profile).unwrap_or_else(|e| panic!("{stem}: {e}"));
        let res = Resources {
            buffers: vec![buffer(0, floats(&p)), buffer(1, vec![0; 80])],
            push: vec![0; 128],
            spec: Default::default(),
        };
        let cfg = Config {
            workgroups: [1, 1, 1],
            ..Config::default()
        };
        let out = run(&module, &cfg, res).unwrap_or_else(|e| panic!("{stem}: {e:?}"));
        let got: Vec<f32> = out.buffers[1]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        assert_eq!(got, want, "{stem}");
    }
}
