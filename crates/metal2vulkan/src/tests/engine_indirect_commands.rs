//! Indirect-command objects and the encoder operations that take them (`lower::access::object`).
//!
//! A kernel's encoder operation through its argument buffer's command-buffer member (a buffer
//! setter, a pipeline state, a dispatch, a barrier, a reset, a copy of another command; a render
//! command's buffer setter, draw, mesh draw and render state setter) is refused under the
//! resident-table domains on every committed profile and on Profile F,
//! `Unsupported{GpuEncodedIndirectCommands, NotImplemented}` (the identity domains' forms are
//! `tests::engine_identity_icb`'s). A kernel
//! whose argument buffer holds a `depth_stencil_state` compiles with it unread and is refused
//! `NotImplemented(IndirectRenderState)` where it reads it.
//!
//! `llvm-as` assembles the kernels; without it the tests skip visibly.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::device::profile::DeviceProfile;
use crate::error::{Error, Feature, Needs};
use crate::pipeline::testing::{compile, device_f, llvm_assemble, AssembleError, PROFILE_F};
use crate::spirv::testing::committed_profiles;

const DATALAYOUT: &str = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32";

fn assemble(name: &str, body: &str) -> Option<Vec<u8>> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let text = format!(
        "target datalayout = \"{DATALAYOUT}\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n{body}"
    );
    let dir = std::env::temp_dir().join(format!(
        "m2v-encode-{}-{}-{name}",
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

/// The committed profiles and Profile F, by stem.
fn profiles() -> Vec<(String, DeviceProfile)> {
    let mut all = committed_profiles().expect("profiles");
    let f = device_f(crate::device::DevicePolicy::RESIDENT_TABLES).expect(PROFILE_F);
    all.push((PROFILE_F.into(), f.profile().clone()));
    all
}

/// The argument buffer `{command_buffer commands; compute_pipeline_state pipeline;}` at Metal
/// buffer 0 (16 bytes, read), then the kernel's own buffers and the thread position.
const COMPUTE_META: &str = r#"
!air.kernel = !{!0}
!air.version = !{!20}
!air.language_version = !{!21}
!0 = !{ptr @encode, !1, !2}
!1 = !{}
!2 = !{!3, !7, !8, !9, !10}
!3 = !{i32 0, !"air.indirect_buffer", !"air.buffer_size", i32 16, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.struct_type_info", !4, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"Encoder", !"air.arg_name", !"enc"}
!4 = !{i32 0, i32 8, i32 0, !"command_buffer", !"commands", !"air.indirect_argument", !5, i32 8, i32 8, i32 0, !"compute_pipeline_state", !"pipeline", !"air.indirect_argument", !6}
!5 = !{i32 0, !"air.command_buffer", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"command_buffer", !"air.arg_name", !"commands"}
!6 = !{i32 1, !"air.compute_pipeline_state", !"air.location_index", i32 1, i32 1, !"air.arg_type_name", !"compute_pipeline_state", !"air.arg_name", !"pipeline"}
!7 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"data"}
!8 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"consts"}
!9 = !{i32 3, !"air.buffer", !"air.location_index", i32 3, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 32, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"Params", !"air.arg_name", !"params"}
!10 = !{i32 4, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"t"}
!20 = !{i32 2, i32 8, i32 0}
!21 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

/// A kernel whose body loads the argument buffer's command buffer (and, where `pipeline`, its
/// pipeline state as `%pso`), then calls `@air.{op}` on command `t` with `args` after the index;
/// `types` are those operands' types.
fn encoder_kernel(op: &str, args: &str, types: &str, pipeline: bool, meta: &str) -> String {
    let pso = match pipeline {
        true => {
            "  %pp = getelementptr inbounds %enc, ptr addrspace(1) %enc, i64 0, i32 1\n  \
             %pso = load ptr addrspace(1), ptr addrspace(1) %pp, align 8\n"
        }
        false => "",
    };
    format!(
        r#"%enc = type {{ ptr addrspace(1), ptr addrspace(1) }}

define void @encode(ptr addrspace(1) %enc, ptr addrspace(1) %data, ptr addrspace(2) %consts, ptr addrspace(1) %params, i32 %t) {{
entry:
  %pc = getelementptr inbounds %enc, ptr addrspace(1) %enc, i64 0, i32 0
  %cmd = load ptr addrspace(1), ptr addrspace(1) %pc, align 8
{pso}  call void @air.{op}(ptr addrspace(1) %cmd, i32 %t{args})
  ret void
}}

declare void @air.{op}(ptr addrspace(1), i32{types})
{meta}"#
    )
}

/// On every committed profile and Profile F, each encoder operation of a compute or a render
/// command is `Unsupported{GpuEncodedIndirectCommands, NotImplemented}`: no device lowers one.
#[test]
fn every_encoder_operation_is_refused_on_every_profile() {
    let render = COMPUTE_META.replace("compute_pipeline_state", "render_pipeline_state");
    let grid = "<3 x i32> <i32 1, i32 1, i32 1>";
    let groups = format!(", {grid}, <3 x i32> <i32 32, i32 1, i32 1>");
    let mesh = format!(", {grid}, {grid}, {grid}");
    let vec3 = ", <3 x i32>, <3 x i32>";
    let vec3s = ", <3 x i32>, <3 x i32>, <3 x i32>";
    let ops: [(&str, &str, &str, bool, &str); 10] = [
        ("set_barrier_compute_command", "", "", false, COMPUTE_META),
        (
            "concurrent_dispatch_threadgroups_compute_command",
            &groups,
            vec3,
            false,
            COMPUTE_META,
        ),
        (
            "set_kernel_buffer_compute_command.p1i8",
            ", ptr addrspace(1) %data, i64 -1, i32 0",
            ", ptr addrspace(1), i64, i32",
            false,
            COMPUTE_META,
        ),
        (
            "set_pipeline_state_compute_command",
            ", ptr addrspace(1) %pso",
            ", ptr addrspace(1)",
            true,
            COMPUTE_META,
        ),
        ("reset_compute_command", "", "", false, COMPUTE_META),
        (
            "copy_compute_command",
            ", ptr addrspace(1) %cmd, i32 0",
            ", ptr addrspace(1), i32",
            false,
            COMPUTE_META,
        ),
        (
            "set_vertex_buffer_render_command.p1i8",
            ", ptr addrspace(1) %data, i64 -1, i32 0",
            ", ptr addrspace(1), i64, i32",
            false,
            &render,
        ),
        (
            "draw_primitives_render_command",
            ", i32 3, i32 0, i32 3, i32 1, i32 0",
            ", i32, i32, i32, i32, i32",
            false,
            &render,
        ),
        (
            "draw_mesh_threadgroups_render_command",
            &mesh,
            vec3s,
            false,
            &render,
        ),
        (
            "set_cull_mode_render_command",
            ", i32 2",
            ", i32",
            false,
            &render,
        ),
    ];
    let all = profiles();
    assert!(!all.is_empty());
    for (op, args, types, pipeline, meta) in ops {
        let Some(bc) = assemble(op, &encoder_kernel(op, args, types, pipeline, meta)) else {
            return;
        };
        for (stem, p) in &all {
            let Error::Unsupported(u) = compile(&bc, "encode", p).expect_err(stem) else {
                panic!("{op} on {stem}: not Unsupported");
            };
            assert_eq!(
                (u.feature, u.needs),
                (Feature::GpuEncodedIndirectCommands, Needs::NotImplemented),
                "{op} on {stem}"
            );
        }
    }
}

/// A kernel whose argument buffer holds a `depth_stencil_state` it never reads compiles on every
/// committed profile, as Metal's compiler compiles such a kernel on an M4; one that reads the
/// state for anything but a setter is refused `NotImplemented(IndirectRenderState)`, the state's
/// one use being a render command's.
#[test]
fn a_depth_stencil_state_member_compiles_unread_and_is_refused_read() {
    let meta = COMPUTE_META.replace("compute_pipeline_state", "depth_stencil_state");
    let kernel = |body: &str| {
        format!(
            r#"%enc = type {{ ptr addrspace(1), ptr addrspace(1) }}

define void @encode(ptr addrspace(1) %enc, ptr addrspace(1) %data, ptr addrspace(2) %consts, ptr addrspace(1) %params, i32 %t) {{
entry:
{body}  ret void
}}
{meta}"#
        )
    };
    let unread = kernel("");
    let read = kernel(
        "  %ps = getelementptr inbounds %enc, ptr addrspace(1) %enc, i64 0, i32 1\n  \
         %ds = load ptr addrspace(1), ptr addrspace(1) %ps, align 8\n  \
         store ptr addrspace(1) %ds, ptr addrspace(1) %params, align 8\n",
    );
    let (Some(unread), Some(read)) = (assemble("ds_unread", &unread), assemble("ds_read", &read))
    else {
        return;
    };
    let all = committed_profiles().expect("profiles");
    assert!(!all.is_empty());
    for (stem, p) in all {
        if let Err(e) = compile(&unread, "encode", &p) {
            panic!("an unread depth/stencil state on {stem}: {e}");
        }
        let Error::Unsupported(u) = compile(&read, "encode", &p).expect_err(&stem) else {
            panic!("a read depth/stencil state on {stem}: not Unsupported");
        };
        assert_eq!(
            (u.feature, u.needs),
            (Feature::IndirectRenderState, Needs::NotImplemented),
            "{stem}"
        );
    }
}
