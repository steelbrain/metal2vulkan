//! GPU-encoded indirect commands through device addresses.
//!
//! An argument buffer's command-buffer member holds its resource ID, whose record buffer lies at
//! the guest address `records_address` of it, and a pipeline-state member its template's
//! host-registered token. On every committed profile with device addresses, a kernel that encodes
//! compute commands (buffer setters in the device and the constant space, a pipeline state, a
//! threadgroup length, a dispatch of threadgroups or threads, a barrier set or cleared, a stage-in
//! region, an imageblock size, a reset, a copy of another command) or render commands (vertex and
//! fragment buffers, a draw; object and mesh buffers, a barrier set or cleared, a mesh draw of
//! threadgroups or threads) compiles, links, and validates, and the CPU interpreter's run of it
//! leaves records that `CommandBuffer::decode` reads back exactly as the host contract states them
//! (`HostRecordsV1`): the round trip. A threadgroup length is kept as set, a multiple of 16 or not.
//! Each command a kernel writes (a copy's destination too) has its written flag past the records
//! set, and no other (`written_flag_offset`). A write the descriptor policy does not allow (a
//! command index at the capacity, a binding past its count, a dynamic stride the descriptor does
//! not enable, a pipeline set while inheriting it, a compute command in a render buffer, a copy
//! whose result is Metal's private record layout) sets the header's fault flag and writes nothing,
//! its written flag included, so `decode` fails as an encoding fault. A record buffer outside the
//! resident set is root 0: contained, and counted under isolation. On the profiles without device
//! addresses every encoder operation is `GpuEncodedIndirectCommands`, naming what they lack.
//!
//! `llvm-as` assembles the kernels; without it the tests skip visibly.

use std::collections::BTreeMap;
use std::num::NonZeroU64;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::bind::status::{ExecutionStatus, StatusBits, StatusWord};
use crate::device::profile::{BufferAccess, DeviceProfile};
use crate::error::{Error, Feature, Needs, Unsupported};
use crate::indirect_commands::written_flag_offset;
use crate::indirect_commands::{
    layout, Action, BufferBinding, BufferStage, CommandBuffer, CommandBufferError,
    CommandDescriptor, CommandRecord, ComputeDispatch, Inheritance, RenderDraws, StateSource,
    BINDING_COUNT, COMMAND_BUFFER_HEADER_BYTES, COMMAND_BYTES, WRITTEN_FLAG_BYTES,
};
use crate::lower::select::Plan;
use crate::pipeline::interface::PushLoc;
use crate::pipeline::sym::{Axis, PushField, ResourceKind, ResourceSlot, SpecKey, Stage};
use crate::pipeline::testing::{
    compile, link_and_validate, llvm_assemble, typed_module, AssembleError,
};
use crate::spirv::interp::{run, Buffer, Config, InterpError, Resources};
use crate::spirv::ir::TypedModule;
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

/// Committed profiles by stem.
type Profiles = Vec<(String, DeviceProfile)>;

/// One parameter that makes a write invalid, by what it breaks.
type Spoil = (&'static str, fn(&mut Params));

/// The committed profiles with device addresses, and those without.
fn profiles() -> (Profiles, Profiles) {
    committed_profiles()
        .expect("profiles")
        .into_iter()
        .partition(|(_, p)| p.fields().buffer_access == BufferAccess::Addressed)
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

/// Each thread `t` reads its parameters `params[t]` (`{u32 index, u32 binding, u32 length, u32
/// tg_binding, i64 stride, u32 grid_x, u32 pad}`, 32 bytes) and encodes command `index`: the
/// 16 bytes of `data` at `16 t` at kernel buffer `binding` with `stride`, `consts` at `binding +
/// 1`, the pipeline state, `length` at threadgroup binding `tg_binding`, then (in two blocks,
/// with the command handle carried across them) a dispatch of `(grid_x, 2, 3)` threadgroups of
/// `(32, 1, 1)` and a barrier for an even `t`, a dispatch of `(grid_x, 2, 3)` threads of `(16, 2,
/// 1)` for an odd `t`.
fn compute_kernel() -> String {
    format!(
        r#"%enc = type {{ ptr addrspace(1), ptr addrspace(1) }}

define void @encode(ptr addrspace(1) %enc, ptr addrspace(1) %data, ptr addrspace(2) %consts, ptr addrspace(1) %params, i32 %t) {{
entry:
  %pc = getelementptr inbounds %enc, ptr addrspace(1) %enc, i64 0, i32 0
  %cmd = load ptr addrspace(1), ptr addrspace(1) %pc, align 8
  %pp = getelementptr inbounds %enc, ptr addrspace(1) %enc, i64 0, i32 1
  %pso = load ptr addrspace(1), ptr addrspace(1) %pp, align 8
  %t64 = zext i32 %t to i64
  %prm = getelementptr inbounds [8 x i32], ptr addrspace(1) %params, i64 %t64
  %index = load i32, ptr addrspace(1) %prm, align 8
  %p1 = getelementptr inbounds i32, ptr addrspace(1) %prm, i64 1
  %binding = load i32, ptr addrspace(1) %p1, align 4
  %p2 = getelementptr inbounds i32, ptr addrspace(1) %prm, i64 2
  %length = load i32, ptr addrspace(1) %p2, align 4
  %p3 = getelementptr inbounds i32, ptr addrspace(1) %prm, i64 3
  %tgb = load i32, ptr addrspace(1) %p3, align 4
  %p4 = getelementptr inbounds i64, ptr addrspace(1) %prm, i64 2
  %stride = load i64, ptr addrspace(1) %p4, align 8
  %p6 = getelementptr inbounds i32, ptr addrspace(1) %prm, i64 6
  %gx = load i32, ptr addrspace(1) %p6, align 4
  %off = mul i64 %t64, 16
  %d = getelementptr inbounds i8, ptr addrspace(1) %data, i64 %off
  call void @air.set_kernel_buffer_compute_command.p1i8(ptr addrspace(1) %cmd, i32 %index, ptr addrspace(1) %d, i64 %stride, i32 %binding)
  %b1 = add i32 %binding, 1
  call void @air.set_kernel_buffer_compute_command.p2i8(ptr addrspace(1) %cmd, i32 %index, ptr addrspace(2) %consts, i64 -1, i32 %b1)
  call void @air.set_pipeline_state_compute_command(ptr addrspace(1) %cmd, i32 %index, ptr addrspace(1) %pso)
  call void @air.set_threadgroup_memory_length_compute_command(ptr addrspace(1) %cmd, i32 %index, i32 %length, i32 %tgb)
  %g0 = insertelement <3 x i32> <i32 poison, i32 2, i32 3>, i32 %gx, i64 0
  %odd = and i32 %t, 1
  %isodd = icmp ne i32 %odd, 0
  br i1 %isodd, label %threads, label %groups
groups:
  call void @air.concurrent_dispatch_threadgroups_compute_command(ptr addrspace(1) %cmd, i32 %index, <3 x i32> %g0, <3 x i32> <i32 32, i32 1, i32 1>)
  call void @air.set_barrier_compute_command(ptr addrspace(1) %cmd, i32 %index)
  br label %done
threads:
  call void @air.concurrent_dispatch_threads_compute_command(ptr addrspace(1) %cmd, i32 %index, <3 x i32> %g0, <3 x i32> <i32 16, i32 2, i32 1>)
  br label %done
done:
  ret void
}}

declare void @air.set_kernel_buffer_compute_command.p1i8(ptr addrspace(1), i32, ptr addrspace(1), i64, i32)
declare void @air.set_kernel_buffer_compute_command.p2i8(ptr addrspace(1), i32, ptr addrspace(2), i64, i32)
declare void @air.set_pipeline_state_compute_command(ptr addrspace(1), i32, ptr addrspace(1))
declare void @air.set_threadgroup_memory_length_compute_command(ptr addrspace(1), i32, i32, i32)
declare void @air.concurrent_dispatch_threadgroups_compute_command(ptr addrspace(1), i32, <3 x i32>, <3 x i32>)
declare void @air.concurrent_dispatch_threads_compute_command(ptr addrspace(1), i32, <3 x i32>, <3 x i32>)
declare void @air.set_barrier_compute_command(ptr addrspace(1), i32)
{COMPUTE_META}"#
    )
}

/// A kernel of the same argument buffer encoding render command `t`: the 16 bytes of `data` at
/// `16 t` as vertex buffer 0 (static stride) and fragment buffer 1, and a draw of 3 vertices from
/// `3 t`, 2 instances from 5.
fn render_kernel() -> String {
    let meta = COMPUTE_META.replace("compute_pipeline_state", "render_pipeline_state");
    format!(
        r#"%enc = type {{ ptr addrspace(1), ptr addrspace(1) }}

define void @encode(ptr addrspace(1) %enc, ptr addrspace(1) %data, ptr addrspace(2) %consts, ptr addrspace(1) %params, i32 %t) {{
entry:
  %pc = getelementptr inbounds %enc, ptr addrspace(1) %enc, i64 0, i32 0
  %cmd = load ptr addrspace(1), ptr addrspace(1) %pc, align 8
  %t64 = zext i32 %t to i64
  %off = mul i64 %t64, 16
  %d = getelementptr inbounds i8, ptr addrspace(1) %data, i64 %off
  call void @air.set_vertex_buffer_render_command.p1i8(ptr addrspace(1) %cmd, i32 %t, ptr addrspace(1) %d, i64 -1, i32 0)
  call void @air.set_fragment_buffer_render_command.p1i8(ptr addrspace(1) %cmd, i32 %t, ptr addrspace(1) %d, i32 1)
  %start = mul i32 %t, 3
  call void @air.draw_primitives_render_command(ptr addrspace(1) %cmd, i32 %t, i32 3, i32 %start, i32 3, i32 2, i32 5)
  ret void
}}

declare void @air.set_vertex_buffer_render_command.p1i8(ptr addrspace(1), i32, ptr addrspace(1), i64, i32)
declare void @air.set_fragment_buffer_render_command.p1i8(ptr addrspace(1), i32, ptr addrspace(1), i32)
declare void @air.draw_primitives_render_command(ptr addrspace(1), i32, i32, i32, i32, i32, i32)
{meta}"#
    )
}

/// Each thread `t` sets command `t`'s render pipeline state, the 16 bytes of `data` at `16 t` as
/// object buffer 0 and `consts` as mesh buffer 0, then for an even `t` its barrier and a mesh
/// draw of `(t + 1, 2, 1)` threadgroups of 4 object and 2 mesh threads, for an odd `t` a cleared
/// barrier and a mesh draw of `(12, 4, 1)` threads, 1 object thread and `(4, 2, 1)` mesh threads
/// (the forms of the owned fixture `mesh_commands`).
fn render_mesh_kernel() -> String {
    let meta = COMPUTE_META.replace("compute_pipeline_state", "render_pipeline_state");
    format!(
        r#"%enc = type {{ ptr addrspace(1), ptr addrspace(1) }}

define void @encode(ptr addrspace(1) %enc, ptr addrspace(1) %data, ptr addrspace(2) %consts, ptr addrspace(1) %params, i32 %t) {{
entry:
  %pc = getelementptr inbounds %enc, ptr addrspace(1) %enc, i64 0, i32 0
  %cmd = load ptr addrspace(1), ptr addrspace(1) %pc, align 8
  %pp = getelementptr inbounds %enc, ptr addrspace(1) %enc, i64 0, i32 1
  %pso = load ptr addrspace(1), ptr addrspace(1) %pp, align 8
  %t64 = zext i32 %t to i64
  %off = mul i64 %t64, 16
  %d = getelementptr inbounds i8, ptr addrspace(1) %data, i64 %off
  call void @air.set_pipeline_state_render_command(ptr addrspace(1) %cmd, i32 %t, ptr addrspace(1) %pso)
  call void @air.set_object_buffer_render_command.p1i8(ptr addrspace(1) %cmd, i32 %t, ptr addrspace(1) %d, i32 0)
  call void @air.set_mesh_buffer_render_command.p2i8(ptr addrspace(1) %cmd, i32 %t, ptr addrspace(2) %consts, i32 0)
  %bit = and i32 %t, 1
  %odd = icmp ne i32 %bit, 0
  br i1 %odd, label %threads, label %groups
groups:
  call void @air.set_barrier_render_command(ptr addrspace(1) %cmd, i32 %t)
  %x = add i32 %t, 1
  %grid = insertelement <3 x i32> <i32 0, i32 2, i32 1>, i32 %x, i64 0
  call void @air.draw_mesh_threadgroups_render_command(ptr addrspace(1) %cmd, i32 %t, <3 x i32> %grid, <3 x i32> <i32 4, i32 1, i32 1>, <3 x i32> <i32 2, i32 1, i32 1>)
  br label %exit
threads:
  call void @air.clear_barrier_render_command(ptr addrspace(1) %cmd, i32 %t)
  call void @air.draw_mesh_threads_render_command(ptr addrspace(1) %cmd, i32 %t, <3 x i32> <i32 12, i32 4, i32 1>, <3 x i32> <i32 1, i32 1, i32 1>, <3 x i32> <i32 4, i32 2, i32 1>)
  br label %exit
exit:
  ret void
}}

declare void @air.set_pipeline_state_render_command(ptr addrspace(1), i32, ptr addrspace(1))
declare void @air.set_object_buffer_render_command.p1i8(ptr addrspace(1), i32, ptr addrspace(1), i32)
declare void @air.set_mesh_buffer_render_command.p2i8(ptr addrspace(1), i32, ptr addrspace(2), i32)
declare void @air.set_barrier_render_command(ptr addrspace(1), i32)
declare void @air.clear_barrier_render_command(ptr addrspace(1), i32)
declare void @air.draw_mesh_threadgroups_render_command(ptr addrspace(1), i32, <3 x i32>, <3 x i32>, <3 x i32>)
declare void @air.draw_mesh_threads_render_command(ptr addrspace(1), i32, <3 x i32>, <3 x i32>, <3 x i32>)
{meta}"#
    )
}

/// Each thread `t` sets command `t`'s render pipeline state, then for an even `t` an indexed draw
/// of 6 `ushort` indices from `data` at `16 t` (triangles, 2 instances, base vertex `-t`, base
/// instance 7), for an odd `t` one of 4 `uint` indices from `consts` (points, 1 instance); thread
/// 3 then resets its command.
fn render_indexed_kernel() -> String {
    let meta = COMPUTE_META.replace("compute_pipeline_state", "render_pipeline_state");
    format!(
        r#"%enc = type {{ ptr addrspace(1), ptr addrspace(1) }}

define void @encode(ptr addrspace(1) %enc, ptr addrspace(1) %data, ptr addrspace(2) %consts, ptr addrspace(1) %params, i32 %t) {{
entry:
  %pc = getelementptr inbounds %enc, ptr addrspace(1) %enc, i64 0, i32 0
  %cmd = load ptr addrspace(1), ptr addrspace(1) %pc, align 8
  %pp = getelementptr inbounds %enc, ptr addrspace(1) %enc, i64 0, i32 1
  %pso = load ptr addrspace(1), ptr addrspace(1) %pp, align 8
  %t64 = zext i32 %t to i64
  %off = mul i64 %t64, 16
  %d = getelementptr inbounds i8, ptr addrspace(1) %data, i64 %off
  call void @air.set_pipeline_state_render_command(ptr addrspace(1) %cmd, i32 %t, ptr addrspace(1) %pso)
  %bit = and i32 %t, 1
  %odd = icmp ne i32 %bit, 0
  br i1 %odd, label %wide, label %narrow
narrow:
  %bv = sub i32 0, %t
  call void @air.draw_indexed_primitives_render_command.p1i16(ptr addrspace(1) %cmd, i32 %t, i32 3, i32 6, ptr addrspace(1) %d, i32 2, i32 %bv, i32 7)
  br label %done
wide:
  call void @air.draw_indexed_primitives_render_command.p2i32(ptr addrspace(1) %cmd, i32 %t, i32 0, i32 4, ptr addrspace(2) %consts, i32 1, i32 0, i32 0)
  br label %done
done:
  %last = icmp eq i32 %t, 3
  br i1 %last, label %reset, label %exit
reset:
  call void @air.reset_render_command(ptr addrspace(1) %cmd, i32 %t)
  br label %exit
exit:
  ret void
}}

declare void @air.set_pipeline_state_render_command(ptr addrspace(1), i32, ptr addrspace(1))
declare void @air.draw_indexed_primitives_render_command.p1i16(ptr addrspace(1), i32, i32, i32, ptr addrspace(1), i32, i32, i32)
declare void @air.draw_indexed_primitives_render_command.p2i32(ptr addrspace(1), i32, i32, i32, ptr addrspace(2), i32, i32, i32)
declare void @air.reset_render_command(ptr addrspace(1), i32)
{meta}"#
    )
}

/// Each thread `t` encodes command `t`'s patch draw: for an even `t`, 4 control points from patch
/// `t`, 2 patches, no patch index buffer, 3 instances from 1, the factors in `data` at `16 t` with
/// an instance stride of 48 and a scale of 1; for an odd `t`, 3 control points, 1 patch, the
/// patch indices in `data` at `16 t`, 1 instance, the factors in `consts`, a scale of 0.5.
fn render_patches_kernel() -> String {
    let meta = COMPUTE_META.replace("compute_pipeline_state", "render_pipeline_state");
    format!(
        r#"%enc = type {{ ptr addrspace(1), ptr addrspace(1) }}

define void @encode(ptr addrspace(1) %enc, ptr addrspace(1) %data, ptr addrspace(2) %consts, ptr addrspace(1) %params, i32 %t) {{
entry:
  %pc = getelementptr inbounds %enc, ptr addrspace(1) %enc, i64 0, i32 0
  %cmd = load ptr addrspace(1), ptr addrspace(1) %pc, align 8
  %t64 = zext i32 %t to i64
  %off = mul i64 %t64, 16
  %d = getelementptr inbounds i8, ptr addrspace(1) %data, i64 %off
  %bit = and i32 %t, 1
  %odd = icmp ne i32 %bit, 0
  br i1 %odd, label %indexed, label %plain
plain:
  call void @air.draw_patches_render_command.p1i32.p1i8(ptr addrspace(1) %cmd, i32 %t, i32 4, i32 %t, i32 2, ptr addrspace(1) null, i32 3, i32 1, ptr addrspace(1) %d, i32 48, float 1.000000e+00)
  br label %exit
indexed:
  call void @air.draw_patches_render_command.p1i32.p2i8(ptr addrspace(1) %cmd, i32 %t, i32 3, i32 0, i32 1, ptr addrspace(1) %d, i32 1, i32 0, ptr addrspace(2) %consts, i32 0, float 5.000000e-01)
  br label %exit
exit:
  ret void
}}

declare void @air.draw_patches_render_command.p1i32.p1i8(ptr addrspace(1), i32, i32, i32, i32, ptr addrspace(1), i32, i32, ptr addrspace(1), i32, float)
declare void @air.draw_patches_render_command.p1i32.p2i8(ptr addrspace(1), i32, i32, i32, i32, ptr addrspace(1), i32, i32, ptr addrspace(2), i32, float)
{meta}"#
    )
}

/// Each thread `t` encodes command `t`'s indexed patch draw: for an even `t`, 4 control points
/// from patch `t`, 2 patches, no patch index or control-point index buffer, 3 instances from 1,
/// the factors in `data` at `16 t` with an instance stride of 48 and a scale of 1; for an odd `t`,
/// 3 control points, 1 patch, the patch indices and the factors in `data` at `16 t`, the
/// control-point indices in `consts`, 1 instance, a scale of 0.5.
fn render_indexed_patches_kernel() -> String {
    let meta = COMPUTE_META.replace("compute_pipeline_state", "render_pipeline_state");
    format!(
        r#"%enc = type {{ ptr addrspace(1), ptr addrspace(1) }}

define void @encode(ptr addrspace(1) %enc, ptr addrspace(1) %data, ptr addrspace(2) %consts, ptr addrspace(1) %params, i32 %t) {{
entry:
  %pc = getelementptr inbounds %enc, ptr addrspace(1) %enc, i64 0, i32 0
  %cmd = load ptr addrspace(1), ptr addrspace(1) %pc, align 8
  %t64 = zext i32 %t to i64
  %off = mul i64 %t64, 16
  %d = getelementptr inbounds i8, ptr addrspace(1) %data, i64 %off
  %bit = and i32 %t, 1
  %odd = icmp ne i32 %bit, 0
  br i1 %odd, label %indexed, label %plain
plain:
  call void @air.draw_indexed_patches_render_command.p1i32.p1i8.p1i8(ptr addrspace(1) %cmd, i32 %t, i32 4, i32 %t, i32 2, ptr addrspace(1) null, ptr addrspace(1) null, i32 3, i32 1, ptr addrspace(1) %d, i32 48, float 1.000000e+00)
  br label %exit
indexed:
  call void @air.draw_indexed_patches_render_command.p1i32.p2i8.p1i8(ptr addrspace(1) %cmd, i32 %t, i32 3, i32 0, i32 1, ptr addrspace(1) %d, ptr addrspace(2) %consts, i32 1, i32 0, ptr addrspace(1) %d, i32 0, float 5.000000e-01)
  br label %exit
exit:
  ret void
}}

declare void @air.draw_indexed_patches_render_command.p1i32.p1i8.p1i8(ptr addrspace(1), i32, i32, i32, i32, ptr addrspace(1), ptr addrspace(1), i32, i32, ptr addrspace(1), i32, float)
declare void @air.draw_indexed_patches_render_command.p1i32.p2i8.p1i8(ptr addrspace(1), i32, i32, i32, i32, ptr addrspace(1), ptr addrspace(2), i32, i32, ptr addrspace(1), i32, float)
{meta}"#
    )
}

// ---- The interpreter's run ----

/// Device addresses of the interpreter's address-backed buffers (host), and the guest addresses
/// the run's Metal objects have (a distinct domain, so a mix-up reads the wrong bytes or faults).
const TABLE: u64 = 0x0000_0010_0000_0000;
const SINK: u64 = 0x0000_0011_0000_0000;
const STATUS: u64 = 0x0000_0012_0000_0000;
const DATA_HOST: u64 = 0x0000_0020_0000_0000;
const CONSTS_HOST: u64 = 0x0000_0021_0000_0000;
const RECORDS_HOST: u64 = 0x0000_0022_0000_0000;
const DATA_GUEST: u64 = 0x0000_0001_0000_1000;
const CONSTS_GUEST: u64 = 0x0000_0001_0000_8000;

/// A buffer slot set plainly to `address`, at the pipeline's stride.
fn plain(address: u64) -> Option<BufferBinding> {
    Some(BufferBinding {
        address,
        attribute_stride: None,
    })
}
/// The record buffer's resource ID, the command-buffer word, and the guest address of its records.
const RECORDS_ID: u64 = 3;
const RECORDS_GUEST: u64 = crate::indirect_commands::records_address(RECORDS_ID);
/// A copy's second record buffer ([`run_records`]).
const SOURCE_HOST: u64 = 0x0000_0023_0000_0000;
const SOURCE_ID: u64 = 4;
const SOURCE_GUEST: u64 = crate::indirect_commands::records_address(SOURCE_ID);
/// The pipeline state's host-registered token.
const TOKEN: u64 = 0x0123_4567_89ab_cdef;
/// Threads per run, and the record buffer's capacity.
const THREADS: u32 = 4;
const DATA_BYTES: u64 = 16 * THREADS as u64;
const CONSTS_BYTES: u64 = 16;

fn slot(index: u32) -> ResourceSlot {
    ResourceSlot {
        stage: Stage::Kernel,
        kind: ResourceKind::Buffer,
        index,
    }
}

/// The root table: null, the sink with the resident range, the four buffer slots (roots 2 to 5;
/// the wide ones read their entries), and the resident set from root 6 sorted by guest address.
fn root_table(resident: &[(u64, u64, u64)]) -> Vec<u8> {
    let statics = [
        (0, 0, 0),
        (DATA_HOST, DATA_GUEST, DATA_BYTES),
        (CONSTS_HOST, CONSTS_GUEST, CONSTS_BYTES),
        (0, 0, 0),
    ];
    let first = 2 + statics.len();
    let mut t = vec![0u8; 32 * (first + resident.len())];
    let mut put = |root: usize, field: usize, v: u64| {
        let at = 32 * root + 8 * field;
        t[at..at + 8].copy_from_slice(&v.to_le_bytes());
    };
    put(1, 0, SINK);
    put(1, 1, first as u64);
    put(1, 2, resident.len() as u64);
    for (i, (h, g, s)) in statics.iter().enumerate() {
        put(2 + i, 0, *h);
        put(2 + i, 1, *g);
        put(2 + i, 2, *s);
    }
    for (i, (h, g, s)) in resident.iter().enumerate() {
        put(first + i, 0, *h);
        put(first + i, 1, *g);
        put(first + i, 2, *s);
    }
    t
}

fn push_bytes(plan: &Plan) -> Vec<u8> {
    let mut push = vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)];
    for f in &plan.push.fields {
        let PushLoc::Push { offset } = f.loc else {
            panic!("{:?} spilled to the ArgumentTable", f.field);
        };
        let at = offset as usize;
        match f.field {
            PushField::RootTableAddress => push[at..at + 8].copy_from_slice(&TABLE.to_le_bytes()),
            PushField::ExecutionStatusAddress => {
                push[at..at + 8].copy_from_slice(&STATUS.to_le_bytes())
            }
            // A zero residual (every slot is bound at its start), and a single region.
            PushField::SlotBindOffset(_) | PushField::DispatchRegion => {}
            other => panic!("unexpected push field {other:?}"),
        }
    }
    push
}

/// The status buffer, at its device address.
fn status() -> Buffer {
    Buffer {
        slot: None,
        address: Some(STATUS),
        bytes: ExecutionStatus::initial().to_vec(),
    }
}

/// One thread's parameters ([`compute_kernel`]).
#[derive(Clone, Copy, Debug)]
struct Params {
    index: u32,
    binding: u32,
    length: u32,
    tg_binding: u32,
    stride: i64,
    grid_x: u32,
}

impl Params {
    fn valid(t: u32) -> Params {
        Params {
            index: (t + 1) % THREADS,
            binding: t % 2,
            length: 16 * (t + 1),
            tg_binding: t % 3,
            stride: -1,
            grid_x: 5 + t,
        }
    }

    fn bytes(self) -> Vec<u8> {
        let mut b = Vec::with_capacity(32);
        for w in [self.index, self.binding, self.length, self.tg_binding] {
            b.extend_from_slice(&w.to_le_bytes());
        }
        b.extend_from_slice(&self.stride.to_le_bytes());
        b.extend_from_slice(&self.grid_x.to_le_bytes());
        b.extend_from_slice(&0u32.to_le_bytes());
        b
    }
}

/// The little-endian `u32`s of `bytes`.
fn words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| u32::from_le_bytes(*w))
        .collect()
}

/// What one run left: the record buffer (its header and records), its written flags, a copy's
/// source record buffer, the status words, and the sink.
struct Run {
    records: Vec<u8>,
    flags: Vec<u32>,
    source: Option<Vec<u8>>,
    status: Vec<u32>,
    sink: Vec<u8>,
}

/// Run `m` (planned as `plan`) over `THREADS` threads with record buffer `records` (initialized)
/// resident at `RECORDS_GUEST` when `resident`, the argument buffer naming it by its ID, and each
/// thread's `params`.
fn run_encoder(
    plan: &Plan,
    m: &TypedModule,
    records: Vec<u8>,
    params: &[Params],
    resident: bool,
) -> Result<Run, InterpError> {
    let mut enc = RECORDS_ID.to_le_bytes().to_vec();
    enc.extend_from_slice(&TOKEN.to_le_bytes());
    let params: Vec<u8> = params.iter().flat_map(|p| p.bytes()).collect();
    run_records(plan, m, enc, records, None, params, resident)
}

/// [`run_encoder`] with argument buffer `enc`, the parameters' bytes `params`, and a second
/// record buffer `source` resident at `SOURCE_GUEST`, when there is one. The record buffer holds
/// a written flag a command past its records, each clear (`written_flag_offset`).
fn run_records(
    plan: &Plan,
    m: &TypedModule,
    enc: Vec<u8>,
    mut records: Vec<u8>,
    source: Option<Vec<u8>>,
    params: Vec<u8>,
    resident: bool,
) -> Result<Run, InterpError> {
    let length = records.len();
    let commands = (length - COMMAND_BUFFER_HEADER_BYTES) / COMMAND_BYTES;
    assert_eq!(
        written_flag_offset(commands as u32, 0),
        length as u64,
        "the flags follow the records"
    );
    records.resize(length + commands * WRITTEN_FLAG_BYTES, 0);
    let mut set = vec![
        (DATA_HOST, DATA_GUEST, DATA_BYTES),
        (CONSTS_HOST, CONSTS_GUEST, CONSTS_BYTES),
    ];
    if resident {
        set.push((RECORDS_HOST, RECORDS_GUEST, records.len() as u64));
    }
    if let Some(source) = &source {
        set.push((SOURCE_HOST, SOURCE_GUEST, source.len() as u64));
    }
    set.sort_by_key(|e| e.1);
    let mut spec = BTreeMap::new();
    spec.insert(SpecKey::LocalSize(Axis::X), u64::from(THREADS));
    spec.insert(SpecKey::LocalSize(Axis::Y), 1);
    spec.insert(SpecKey::LocalSize(Axis::Z), 1);
    spec.insert(SpecKey::FlatLocalSize, u64::from(THREADS));
    let mut buffers = vec![
        Buffer {
            slot: Some(slot(0)),
            address: None,
            bytes: enc,
        },
        Buffer {
            slot: Some(slot(1)),
            address: Some(DATA_HOST),
            bytes: vec![0; DATA_BYTES as usize],
        },
        Buffer {
            slot: Some(slot(2)),
            address: Some(CONSTS_HOST),
            bytes: vec![0; CONSTS_BYTES as usize],
        },
        Buffer {
            slot: Some(slot(3)),
            address: None,
            bytes: params,
        },
        Buffer {
            slot: None,
            address: Some(TABLE),
            bytes: root_table(&set),
        },
        Buffer {
            slot: None,
            address: Some(SINK),
            bytes: vec![0; 64],
        },
        status(),
        Buffer {
            slot: None,
            address: Some(RECORDS_HOST),
            bytes: records,
        },
    ];
    let copies = source.is_some();
    if let Some(bytes) = source {
        buffers.push(Buffer {
            slot: None,
            address: Some(SOURCE_HOST),
            bytes,
        });
    }
    let res = Resources {
        buffers,
        push: push_bytes(plan),
        spec,
    };
    let out = run(m, &Config::default(), res)?;
    let (records, flags) = out.buffers[7].split_at(length);
    Ok(Run {
        records: records.to_vec(),
        flags: words(flags),
        source: copies.then(|| out.buffers[8].clone()),
        status: out.buffers[6]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
            .collect(),
        sink: out.buffers[5].clone(),
    })
}

const OWNED: Inheritance = Inheritance {
    pipeline: StateSource::Command,
    buffers: StateSource::Command,
};

fn compute_descriptor(dynamic: bool) -> CommandDescriptor {
    CommandDescriptor::Compute {
        dispatch: ComputeDispatch::Both,
        max_buffer_bind_count: 3,
        max_threadgroup_memory_bind_count: 3,
        support_dynamic_attribute_stride: dynamic,
    }
}

/// Initialized storage for `THREADS` commands under `descriptor` and `inheritance`.
fn storage(descriptor: &CommandDescriptor, inheritance: Inheritance) -> Vec<u8> {
    let mut bytes = vec![0xa5; COMMAND_BUFFER_HEADER_BYTES + THREADS as usize * COMMAND_BYTES];
    CommandBuffer::initialize(&mut bytes, THREADS, descriptor, inheritance).expect("initialize");
    bytes
}

/// The record thread `t` with parameters `p` encodes ([`compute_kernel`]), its pipeline the
/// kernel's (`CommandRecord::kernel_pipeline`).
fn compute_record(t: u32, p: Params) -> CommandRecord {
    let mut r = CommandRecord {
        kernel_pipeline: true,
        ..CommandRecord::default()
    };
    let grid = [p.grid_x, 2, 3];
    r.action = if t % 2 == 1 {
        Action::DispatchThreads {
            grid,
            local_size: [16, 2, 1],
        }
    } else {
        r.barrier = true;
        Action::DispatchThreadgroups {
            grid,
            local_size: [32, 1, 1],
        }
    };
    r.bindings.pipeline = NonZeroU64::new(TOKEN);
    let kernel = r.bindings.buffers_mut(BufferStage::Kernel);
    kernel[p.binding as usize] = Some(BufferBinding {
        address: DATA_GUEST + 16 * u64::from(t),
        attribute_stride: (p.stride != -1).then_some(p.stride as u64),
    });
    kernel[p.binding as usize + 1] = plain(CONSTS_GUEST);
    r.threadgroup_memory_lengths[p.tg_binding as usize] = p.length;
    r
}

/// The records of `bytes` (decoded under `descriptor` and the owned state) in index order.
fn decoded(
    bytes: &[u8],
    descriptor: &CommandDescriptor,
) -> Result<Vec<CommandRecord>, CommandBufferError> {
    Ok(CommandBuffer::decode(bytes, descriptor, OWNED)?
        .records(0..THREADS)?
        .collect())
}

fn compiled(name: &str, text: &str) -> Option<Vec<(String, DeviceProfile, Plan, TypedModule)>> {
    let bc = assemble(name, text)?;
    let (addressed, _) = profiles();
    assert!(!addressed.is_empty());
    Some(
        addressed
            .into_iter()
            .map(|(stem, p)| {
                let a = compile(&bc, "encode", &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
                link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
                let (plan, m) = typed_module(&bc, "encode", &p).expect("module");
                (stem, p, plan, m)
            })
            .collect(),
    )
}

/// On every profile with device addresses, the records a kernel encodes through its argument
/// buffer's command buffer decode exactly as the host contract states them: each thread's command
/// index, its kernel buffers (the device buffer's guest address at its displacement, the constant
/// buffer's, the static or dynamic stride), the pipeline state's token, the threadgroup length at
/// its binding as set (one not a multiple of 16 included), the dispatch (threadgroups with a
/// barrier, or threads), carried across blocks; the header and every untouched word as `initialize`
/// left them; and no write in the sink. Every command starts with a pipeline a reset retained,
/// which the pipeline setter replaces with its own, reported.
#[test]
fn encoded_compute_records_round_trip_through_the_host_contract() {
    let Some(runs) = compiled("compute", &compute_kernel()) else {
        return;
    };
    for (stem, _, plan, m) in &runs {
        for dynamic in [false, true] {
            let descriptor = compute_descriptor(dynamic);
            let params: Vec<Params> = (0..THREADS)
                .map(|t| Params {
                    stride: if dynamic && t == 2 { 48 } else { -1 },
                    length: if t == 3 { 20 } else { Params::valid(t).length },
                    ..Params::valid(t)
                })
                .collect();
            let retained = CommandRecord {
                retained_pipeline: NonZeroU64::new(5),
                ..CommandRecord::default()
            };
            let initial = encoded(&descriptor, OWNED, &vec![retained; THREADS as usize]);
            let run = run_encoder(plan, m, initial, &params, true)
                .unwrap_or_else(|e| panic!("{stem}: {e}"));
            let mut want = vec![CommandRecord::default(); THREADS as usize];
            for (t, p) in params.iter().enumerate() {
                want[p.index as usize] = compute_record(t as u32, *p);
            }
            let got = decoded(&run.records, &descriptor)
                .unwrap_or_else(|e| panic!("{stem} dynamic={dynamic}: {e}"));
            assert_eq!(got, want, "{stem} dynamic={dynamic}");
            // The bytes are exactly the encoded records: nothing else was written.
            let mut exact = storage(&descriptor, OWNED);
            for (i, r) in want.iter().enumerate() {
                let at = COMMAND_BUFFER_HEADER_BYTES + i * COMMAND_BYTES;
                r.encode_into(&mut exact[at..at + COMMAND_BYTES]).unwrap();
            }
            assert_eq!(run.records, exact, "{stem} dynamic={dynamic}: bytes");
            // Each written command's flag set, and only those.
            let mut flags = vec![0; THREADS as usize];
            for p in &params {
                flags[p.index as usize] = 1;
            }
            assert_eq!(run.flags, flags, "{stem} dynamic={dynamic}: written flags");
            assert!(run.sink.iter().all(|b| *b == 0), "{stem}: sink written");
            assert_eq!(
                run.status[StatusWord::ResidentMisses.index() as usize],
                0,
                "{stem}: misses"
            );
        }
    }
}

/// A render command's vertex buffer (static stride), fragment buffer, and draw decode under a
/// render descriptor.
#[test]
fn encoded_render_records_round_trip_through_the_host_contract() {
    let Some(runs) = compiled("render", &render_kernel()) else {
        return;
    };
    let descriptor = CommandDescriptor::Render {
        draws: RenderDraws::PRIMITIVES,
        max_vertex_buffer_bind_count: 1,
        max_fragment_buffer_bind_count: 2,
        max_object_buffer_bind_count: 0,
        max_mesh_buffer_bind_count: 0,
        support_dynamic_attribute_stride: false,
    };
    for (stem, _, plan, m) in &runs {
        let run = run_encoder(plan, m, storage(&descriptor, OWNED), &[], true)
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        let want: Vec<CommandRecord> = (0..THREADS)
            .map(|t| {
                let mut r = CommandRecord {
                    action: Action::DrawPrimitives {
                        primitive_type: 3,
                        vertex_start: 3 * t,
                        vertex_count: 3,
                        instance_count: 2,
                        base_instance: 5,
                    },
                    ..CommandRecord::default()
                };
                let address = DATA_GUEST + 16 * u64::from(t);
                r.bindings.buffers_mut(BufferStage::Vertex)[0] = plain(address);
                r.bindings.buffers_mut(BufferStage::Fragment)[1] = plain(address);
                r
            })
            .collect();
        let got = decoded(&run.records, &descriptor).unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert_eq!(got, want, "{stem}");
    }
}

/// Each thread `t` resets command `t` (`air.reset_render_command`) where `reset`, else reads and
/// writes nothing.
fn render_reset_kernel(reset: bool) -> String {
    let meta = COMPUTE_META.replace("compute_pipeline_state", "render_pipeline_state");
    let body = match reset {
        true => {
            "%pc = getelementptr inbounds %enc, ptr addrspace(1) %enc, i64 0, i32 0
  %cmd = load ptr addrspace(1), ptr addrspace(1) %pc, align 8
  call void @air.reset_render_command(ptr addrspace(1) %cmd, i32 %t)"
        }
        false => "",
    };
    format!(
        r#"%enc = type {{ ptr addrspace(1), ptr addrspace(1) }}

define void @encode(ptr addrspace(1) %enc, ptr addrspace(1) %data, ptr addrspace(2) %consts, ptr addrspace(1) %params, i32 %t) {{
entry:
  {body}
  ret void
}}

declare void @air.reset_render_command(ptr addrspace(1), i32)
{meta}"#
    )
}

/// A GPU reset clears what a command may have set and nothing else: its command words, the
/// bindings below each stage's count in the header, and the words past the bindings
/// (`default_record`). Every record starts holding a draw, a barrier, a vertex and a fragment
/// binding below the counts (2 and 2), and threadgroup-length words; each byte of a binding past
/// the counts (vertex 2, fragment 2, every object and mesh one) holds a mark, which no writer of
/// the records puts there. After the reset every byte below the counts and of the words reads zero,
/// and every mark stands: the reset wrote 64 + 4 x 16 + 136 bytes of each record, not 2248. The
/// negative control: the same storage under a kernel that does not reset keeps every stale byte,
/// so the zeros are the reset's.
#[test]
fn a_reset_clears_the_command_words_and_the_counted_bindings_alone() {
    let descriptor = CommandDescriptor::Render {
        draws: RenderDraws::PRIMITIVES,
        max_vertex_buffer_bind_count: 2,
        max_fragment_buffer_bind_count: 2,
        max_object_buffer_bind_count: 0,
        max_mesh_buffer_bind_count: 0,
        support_dynamic_attribute_stride: false,
    };
    const MARK: u8 = 0xab;
    let binding = |stage, slot: usize| {
        let at = (layout::buffers(stage) + slot as u64 * layout::BINDING) as usize;
        at..at + layout::BINDING as usize
    };
    // Each record's ranges a reset clears, and the marked ones it leaves.
    let words = 0..layout::buffers(BufferStage::Kernel) as usize;
    let mut cleared = vec![words];
    let mut marked = Vec::new();
    for slot in 0..BINDING_COUNT {
        let below = slot < 2;
        for stage in [BufferStage::Vertex, BufferStage::Fragment] {
            match below {
                true => cleared.push(binding(stage, slot)),
                false => marked.push(binding(stage, slot)),
            }
        }
        marked.push(binding(BufferStage::Mesh, slot));
        marked.push(binding(BufferStage::Object, slot));
    }
    cleared.push(layout::THREADGROUP as usize..COMMAND_BYTES);
    let mut stale = storage(&descriptor, OWNED);
    for t in 0..THREADS as usize {
        let record = COMMAND_BUFFER_HEADER_BYTES + t * COMMAND_BYTES;
        for range in &cleared {
            stale[record + range.start..record + range.end].fill(0x5a);
        }
        for range in &marked {
            stale[record + range.start..record + range.end].fill(MARK);
        }
    }
    for (reset, stem) in [(true, "reset"), (false, "unreset")] {
        let Some(runs) = compiled(stem, &render_reset_kernel(reset)) else {
            return;
        };
        for (profile, _, plan, m) in &runs {
            let run = run_encoder(plan, m, stale.clone(), &[], true)
                .unwrap_or_else(|e| panic!("{profile}: {e}"));
            for t in 0..THREADS as usize {
                let record = COMMAND_BUFFER_HEADER_BYTES + t * COMMAND_BYTES;
                let bytes = &run.records[record..record + COMMAND_BYTES];
                for range in &cleared {
                    let want = match reset {
                        true => 0,
                        false => 0x5a,
                    };
                    let got = &bytes[range.clone()];
                    assert!(
                        got.iter().all(|&b| b == want),
                        "{profile} {stem} {t} {range:?}"
                    );
                }
                for range in &marked {
                    let got = &bytes[range.clone()];
                    assert!(
                        got.iter().all(|&b| b == MARK),
                        "{profile} {stem} {t} {range:?}"
                    );
                }
            }
        }
    }
}

/// `line` without the `i64` operand of its buffer setter: a declaration's `, i64` or a call's
/// `, i64 <value>`.
fn drop_stride(line: &str) -> String {
    let Some(at) = line.find(", i64") else {
        return line.to_string();
    };
    let rest = &line[at + 2..];
    let end = rest.find(", ").expect("an operand follows the stride");
    format!("{}{}", &line[..at], &rest[end..])
}

/// `text` as AIR 2.5 (Metal compiler 31001.723) spells it: its kernel and vertex buffer setters
/// declared and called without the attribute stride, MSL's plain setter, and the module's AIR
/// version 2.5 of language Metal 2.4.
fn stride_less(text: &str) -> String {
    let lines = text.lines().map(|line| {
        let setter = ["@air.set_kernel_buffer_", "@air.set_vertex_buffer_"]
            .iter()
            .any(|family| line.contains(family));
        match setter {
            true => drop_stride(line),
            false => line.to_string(),
        }
    });
    let mut out = lines.collect::<Vec<_>>().join("\n");
    for (from, to) in [
        (
            "!20 = !{i32 2, i32 8, i32 0}",
            "!20 = !{i32 2, i32 5, i32 0}",
        ),
        (
            "!21 = !{!\"Metal\", i32 4, i32 0, i32 0}",
            "!21 = !{!\"Metal\", i32 2, i32 4, i32 0}",
        ),
    ] {
        assert!(out.contains(from), "{from}");
        out = out.replace(from, to);
    }
    assert!(!out.contains(", i64 %stride") && !out.contains("i64, i32)"));
    out
}

/// AIR 2.5's kernel and vertex buffer setters, which take no attribute stride (MSL's plain
/// setter), write exactly the records of today's form called with a stride of all ones, as Metal's
/// compiler spells the plain setter now, on every profile with device addresses: the buffer's
/// address at its slot and the pipeline's own stride, the slot `set_kernel_buffer` (Metal's
/// `setKernelBuffer:offset:atIndex:`) leaves on the host, over a dynamic stride set before too (the
/// kernel-side plain setter writes all ones over it).
#[test]
fn air_2_5_buffer_setters_write_the_plain_setters_records() {
    let (Some(today_runs), Some(air25_runs)) = (
        compiled("compute", &compute_kernel()),
        compiled("compute-air25", &stride_less(&compute_kernel())),
    ) else {
        return;
    };
    for ((stem, _, sp, sm), (_, _, pp, pm)) in today_runs.iter().zip(&air25_runs) {
        for dynamic in [false, true] {
            let descriptor = compute_descriptor(dynamic);
            let params: Vec<Params> = (0..THREADS).map(Params::valid).collect();
            // Each command's slot holds a dynamic stride before, where the descriptor allows one.
            let before: Vec<CommandRecord> = (0..THREADS)
                .map(|i| {
                    let mut r = CommandRecord::default();
                    if dynamic {
                        let at = params.iter().find(|p| p.index == i).expect("a writer");
                        r.bindings.buffers_mut(BufferStage::Kernel)[at.binding as usize] =
                            Some(BufferBinding {
                                address: DATA_GUEST,
                                attribute_stride: Some(48),
                            });
                    }
                    r
                })
                .collect();
            let initial = encoded(&descriptor, OWNED, &before);
            let today = run_encoder(sp, sm, initial.clone(), &params, true)
                .unwrap_or_else(|e| panic!("{stem}: {e}"));
            let air25 = run_encoder(pp, pm, initial, &params, true)
                .unwrap_or_else(|e| panic!("{stem} AIR 2.5: {e}"));
            assert_eq!(air25.records, today.records, "{stem} dynamic={dynamic}");
            let mut want = vec![CommandRecord::default(); THREADS as usize];
            for (t, p) in params.iter().enumerate() {
                want[p.index as usize] = compute_record(t as u32, *p);
                let kernel = want[p.index as usize]
                    .bindings
                    .buffers_mut(BufferStage::Kernel);
                let address = DATA_GUEST + 16 * t as u64;
                assert_eq!(kernel[p.binding as usize], plain(address), "{stem}");
            }
            let got = decoded(&air25.records, &descriptor)
                .unwrap_or_else(|e| panic!("{stem} dynamic={dynamic}: {e}"));
            assert_eq!(got, want, "{stem} dynamic={dynamic}");
            assert!(air25.sink.iter().all(|b| *b == 0), "{stem}: sink written");
        }
    }
    let (Some(today_runs), Some(air25_runs)) = (
        compiled("render", &render_kernel()),
        compiled("render-air25", &stride_less(&render_kernel())),
    ) else {
        return;
    };
    let descriptor = CommandDescriptor::Render {
        draws: RenderDraws::PRIMITIVES,
        max_vertex_buffer_bind_count: 1,
        max_fragment_buffer_bind_count: 2,
        max_object_buffer_bind_count: 0,
        max_mesh_buffer_bind_count: 0,
        support_dynamic_attribute_stride: false,
    };
    for ((stem, _, sp, sm), (_, _, pp, pm)) in today_runs.iter().zip(&air25_runs) {
        let today = run_encoder(sp, sm, storage(&descriptor, OWNED), &[], true)
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        let air25 = run_encoder(pp, pm, storage(&descriptor, OWNED), &[], true)
            .unwrap_or_else(|e| panic!("{stem} AIR 2.5: {e}"));
        assert_eq!(air25.records, today.records, "{stem}");
        let got = decoded(&air25.records, &descriptor).unwrap_or_else(|e| panic!("{stem}: {e}"));
        for (t, r) in got.iter().enumerate() {
            let address = DATA_GUEST + 16 * t as u64;
            assert_eq!(
                r.bindings.buffers(BufferStage::Vertex)[0],
                plain(address),
                "{stem}"
            );
        }
    }
}

/// A render descriptor of mesh draws (of threadgroups where `threadgroups`, else of threads)
/// alone, with one object and one mesh buffer, `primitive` vertex buffers and twice as many
/// fragment buffers.
fn mesh_descriptor(threadgroups: bool, primitive: u32) -> CommandDescriptor {
    CommandDescriptor::Render {
        draws: RenderDraws {
            primitives: false,
            mesh_threadgroups: threadgroups,
            mesh_threads: !threadgroups,
            ..RenderDraws::PRIMITIVES
        },
        max_vertex_buffer_bind_count: primitive,
        max_fragment_buffer_bind_count: 2 * primitive,
        max_object_buffer_bind_count: 1,
        max_mesh_buffer_bind_count: 1,
        support_dynamic_attribute_stride: false,
    }
}

/// A kernel's mesh draws of either kind decode under a render descriptor of either mesh kind, whose
/// records hold both (a Metal probe): the grid and the object and mesh threadgroups' threads, the
/// barrier set or cleared, the pipeline state's token, the object buffer's guest address (device
/// memory) and the mesh buffer's (constant memory), which take the words of a compute command's
/// kernel buffers. Under a descriptor of draws of primitives, whose records do not hold them, the
/// header's fault flag fails the decode, as the GPU faults on Metal.
#[test]
fn encoded_mesh_draws_round_trip_through_the_host_contract() {
    let Some(runs) = compiled("render_mesh", &render_mesh_kernel()) else {
        return;
    };
    for (stem, _, plan, m) in &runs {
        for threadgroups in [true, false] {
            let descriptor = mesh_descriptor(threadgroups, 0);
            let run = run_encoder(plan, m, storage(&descriptor, OWNED), &[], true)
                .unwrap_or_else(|e| panic!("{stem}: {e}"));
            let want: Vec<CommandRecord> = (0..THREADS)
                .map(|t| {
                    let action = if t % 2 == 0 {
                        Action::DrawMeshThreadgroups {
                            threadgroups: [t + 1, 2, 1],
                            object_threads: [4, 1, 1],
                            mesh_threads: [2, 1, 1],
                        }
                    } else {
                        Action::DrawMeshThreads {
                            threads: [12, 4, 1],
                            object_threads: [1, 1, 1],
                            mesh_threads: [4, 2, 1],
                        }
                    };
                    let mut r = CommandRecord {
                        action,
                        barrier: t % 2 == 0,
                        ..CommandRecord::default()
                    };
                    r.bindings.pipeline = NonZeroU64::new(TOKEN);
                    r.kernel_pipeline = true;
                    r.bindings.buffers_mut(BufferStage::Object)[0] =
                        plain(DATA_GUEST + 16 * u64::from(t));
                    r.bindings.buffers_mut(BufferStage::Mesh)[0] = plain(CONSTS_GUEST);
                    r
                })
                .collect();
            let got = decoded(&run.records, &descriptor)
                .unwrap_or_else(|e| panic!("{stem} threadgroups={threadgroups}: {e}"));
            assert_eq!(got, want, "{stem} threadgroups={threadgroups}");
        }
        let primitives = CommandDescriptor::Render {
            draws: RenderDraws::PRIMITIVES,
            max_vertex_buffer_bind_count: 0,
            max_fragment_buffer_bind_count: 0,
            max_object_buffer_bind_count: 1,
            max_mesh_buffer_bind_count: 1,
            support_dynamic_attribute_stride: false,
        };
        let run = run_encoder(plan, m, storage(&primitives, OWNED), &[], true)
            .unwrap_or_else(|e| panic!("{stem} primitives: {e}"));
        assert_eq!(
            decoded(&run.records, &primitives).err(),
            Some(CommandBufferError::EncodingFault(1)),
            "{stem} primitives"
        );
    }
}

/// A kernel's draw of primitives on a buffer of mesh draws alone is dropped: it writes nothing and
/// sets no fault, as it draws nothing on Metal without a fault (a Metal probe); its buffer setters
/// write as on any render buffer.
#[test]
fn a_draw_of_primitives_on_a_buffer_of_mesh_draws_is_dropped() {
    let Some(runs) = compiled("render_dropped", &render_kernel()) else {
        return;
    };
    for (stem, _, plan, m) in &runs {
        let descriptor = mesh_descriptor(true, 1);
        let run = run_encoder(plan, m, storage(&descriptor, OWNED), &[], true)
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        let want: Vec<CommandRecord> = (0..THREADS)
            .map(|t| {
                let mut r = CommandRecord::default();
                let address = DATA_GUEST + 16 * u64::from(t);
                r.bindings.buffers_mut(BufferStage::Vertex)[0] = plain(address);
                r.bindings.buffers_mut(BufferStage::Fragment)[1] = plain(address);
                r
            })
            .collect();
        let got = decoded(&run.records, &descriptor).unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert_eq!(got, want, "{stem}");
    }
}

/// A write the descriptor policy does not allow sets the header's fault flag and writes nothing of
/// its own, so `decode` fails the run as an encoding fault: each invalid parameter alone, a compute
/// descriptor that inherits the pipeline, and a render descriptor (every compute operation is
/// outside its policy).
#[test]
fn a_write_outside_the_descriptor_policy_faults_and_writes_nothing() {
    let Some(runs) = compiled("faults", &compute_kernel()) else {
        return;
    };
    let invalid: [Spoil; 4] = [
        ("index at capacity", |p| p.index = THREADS),
        ("binding past its count", |p| p.binding = 2),
        ("threadgroup binding past its count", |p| p.tg_binding = 3),
        ("dynamic stride not enabled", |p| p.stride = 32),
    ];
    for (stem, _, plan, m) in &runs {
        let descriptor = compute_descriptor(false);
        for (what, spoil) in invalid {
            let mut params: Vec<Params> = (0..THREADS).map(Params::valid).collect();
            spoil(&mut params[1]);
            let run = run_encoder(plan, m, storage(&descriptor, OWNED), &params, true)
                .unwrap_or_else(|e| panic!("{stem} {what}: {e}"));
            assert_eq!(
                decoded(&run.records, &descriptor).err(),
                Some(CommandBufferError::EncodingFault(1)),
                "{stem} {what}"
            );
            // Thread 1's own command keeps the words of every write that faulted.
            if what == "index at capacity" {
                let mut want = storage(&descriptor, OWNED);
                want[4..8].copy_from_slice(&1u32.to_le_bytes());
                for (t, p) in params.iter().enumerate().filter(|(t, _)| *t != 1) {
                    let at = COMMAND_BUFFER_HEADER_BYTES + p.index as usize * COMMAND_BYTES;
                    compute_record(t as u32, *p)
                        .encode_into(&mut want[at..at + COMMAND_BYTES])
                        .unwrap();
                }
                assert_eq!(run.records, want, "{stem} {what}");
                // Thread 1's writes all faulted: the flags of the others' commands alone are set.
                let mut flags = vec![0; THREADS as usize];
                for (_, p) in params.iter().enumerate().filter(|(t, _)| *t != 1) {
                    flags[p.index as usize] = 1;
                }
                assert_eq!(run.flags, flags, "{stem} {what}: written flags");
            }
        }
        let inherit = Inheritance {
            pipeline: StateSource::Parent,
            buffers: StateSource::Command,
        };
        let params: Vec<Params> = (0..THREADS).map(Params::valid).collect();
        let run = run_encoder(plan, m, storage(&descriptor, inherit), &params, true)
            .unwrap_or_else(|e| panic!("{stem} inherited: {e}"));
        assert_eq!(
            CommandBuffer::decode(&run.records, &descriptor, inherit).err(),
            Some(CommandBufferError::EncodingFault(1)),
            "{stem} inherited pipeline"
        );
        let render = CommandDescriptor::Render {
            draws: RenderDraws::PRIMITIVES,
            max_vertex_buffer_bind_count: 3,
            max_fragment_buffer_bind_count: 3,
            max_object_buffer_bind_count: 0,
            max_mesh_buffer_bind_count: 0,
            support_dynamic_attribute_stride: false,
        };
        let initial = storage(&render, OWNED);
        let run = run_encoder(plan, m, initial.clone(), &params, true)
            .unwrap_or_else(|e| panic!("{stem} render: {e}"));
        let mut want = initial;
        want[4..8].copy_from_slice(&1u32.to_le_bytes());
        assert_eq!(run.records, want, "{stem} render descriptor");
        assert_eq!(
            run.flags,
            vec![0; THREADS as usize],
            "{stem} render descriptor: no written flag"
        );
    }
}

/// A record buffer no resident allocation holds is root 0: nothing of it is read or written (the
/// header loads read 0, so every write faults into the sink), and each operation's access is
/// counted as a resident-set miss, which the executor reads as no evidence.
#[test]
fn a_record_buffer_outside_the_resident_set_is_contained_and_counted() {
    let Some(runs) = compiled("miss", &compute_kernel()) else {
        return;
    };
    for (stem, _, plan, m) in &runs {
        let descriptor = compute_descriptor(false);
        let initial = storage(&descriptor, OWNED);
        let params: Vec<Params> = (0..THREADS).map(Params::valid).collect();
        let run = run_encoder(plan, m, initial.clone(), &params, false)
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert_eq!(run.records, initial, "{stem}");
        // One per operation: six for an even thread (with the barrier), five for an odd one.
        assert_eq!(
            run.status[StatusWord::ResidentMisses.index() as usize],
            (0..THREADS).map(|t| 6 - t % 2).sum::<u32>(),
            "{stem}"
        );
        assert!(
            StatusBits(run.status[StatusWord::Status.index() as usize])
                .contains(StatusBits::RESIDENT_MISS),
            "{stem}"
        );
    }
}

/// Without device addresses every encoder operation is `GpuEncodedIndirectCommands`, at the first
/// one, naming what the profile lacks of Profile A.
#[test]
fn encoding_without_device_addresses_names_what_the_profile_lacks() {
    let Some(bc) = assemble("floor", &compute_kernel()) else {
        return;
    };
    let (_, floor) = profiles();
    assert!(!floor.is_empty());
    for (stem, p) in floor {
        let Error::Unsupported(u) = compile(&bc, "encode", &p).expect_err(&stem) else {
            panic!("{stem}: not Unsupported");
        };
        let want = Unsupported::needs_any_of(
            Feature::GpuEncodedIndirectCommands,
            vec![p.addressed_needs()],
        );
        assert!(
            !matches!(u.needs, Needs::NotImplemented | Needs::NotRepresentable),
            "{stem}"
        );
        assert_eq!((u.feature, &u.needs), (want.feature, &want.needs), "{stem}");
        assert_eq!(
            u.location
                .as_ref()
                .and_then(|l| l.in_function())
                .map(|(_, b, i)| (b, i)),
            Some((Some(0), Some(19))),
            "{stem}: the first encoder operation"
        );
    }
}

/// A kernel's render state setter (cull mode, depth bias, depth clip mode, front-facing winding,
/// triangle fill mode, depth/stencil state; the declarations Metal's compiler writes) decodes, and
/// on every profile with device addresses the lowering refuses it
/// `NotImplemented(IndirectRenderState)`: the record holds no render state, and Metal's compiler
/// refuses every one on an M4. The depth/stencil state is the argument buffer's second member,
/// `air.depth_stencil_state`, in place of a pipeline state.
#[test]
fn render_state_setters_refuse_as_indirect_render_state() {
    let render = COMPUTE_META.replace("compute_pipeline_state", "render_pipeline_state");
    let depth_stencil = COMPUTE_META.replace("compute_pipeline_state", "depth_stencil_state");
    let state = "  %ps = getelementptr inbounds %enc, ptr addrspace(1) %enc, i64 0, i32 1\n  \
                 %ds = load ptr addrspace(1), ptr addrspace(1) %ps, align 8\n";
    for (name, setter, types, values, prelude, meta) in [
        (
            "cull",
            "set_cull_mode_render_command",
            "i32",
            "i32 2",
            "",
            &render,
        ),
        (
            "depth_bias",
            "set_depth_bias_render_command",
            "float, float, float",
            "float 1.0, float 2.0, float 0.5",
            "",
            &render,
        ),
        (
            "depth_clip",
            "set_depth_clip_mode_render_command",
            "i32",
            "i32 1",
            "",
            &render,
        ),
        (
            "winding",
            "set_front_facing_winding_render_command",
            "i32",
            "i32 1",
            "",
            &render,
        ),
        (
            "fill",
            "set_triangle_fill_mode_render_command",
            "i32",
            "i32 1",
            "",
            &render,
        ),
        (
            "depth_stencil",
            "set_depth_stencil_state_render_command",
            "ptr addrspace(1)",
            "ptr addrspace(1) %ds",
            state,
            &depth_stencil,
        ),
    ] {
        let text = format!(
            r#"%enc = type {{ ptr addrspace(1), ptr addrspace(1) }}

define void @encode(ptr addrspace(1) %enc, ptr addrspace(1) %data, ptr addrspace(2) %consts, ptr addrspace(1) %params, i32 %t) {{
entry:
  %pc = getelementptr inbounds %enc, ptr addrspace(1) %enc, i64 0, i32 0
  %cmd = load ptr addrspace(1), ptr addrspace(1) %pc, align 8
{prelude}  call void @air.{setter}(ptr addrspace(1) %cmd, i32 %t, {values})
  ret void
}}

declare void @air.{setter}(ptr addrspace(1), i32, {types})
{meta}"#
        );
        let Some(bc) = assemble(name, &text) else {
            return;
        };
        let (addressed, _) = profiles();
        assert!(!addressed.is_empty());
        let want = Unsupported::not_implemented(Feature::IndirectRenderState);
        for (stem, p) in addressed {
            let Error::Unsupported(u) = compile(&bc, "encode", &p).expect_err(&stem) else {
                panic!("{name} on {stem}: not Unsupported");
            };
            assert_eq!(
                (u.feature, &u.needs),
                (want.feature.clone(), &want.needs),
                "{name} on {stem}"
            );
        }
    }
}

/// A kernel whose argument buffer holds a `depth_stencil_state` it never reads compiles on every
/// profile with device addresses, as Metal's compiler compiles such a kernel's draw on an M4; one
/// that reads the state for anything but a setter is refused `NotImplemented(IndirectRenderState)`,
/// the state's one use being a render command's.
#[test]
fn a_depth_stencil_state_member_compiles_unread_and_is_refused_read() {
    let meta = COMPUTE_META.replace("compute_pipeline_state", "depth_stencil_state");
    let kernel = |body: &str| {
        format!(
            r#"%enc = type {{ ptr addrspace(1), ptr addrspace(1) }}

define void @encode(ptr addrspace(1) %enc, ptr addrspace(1) %data, ptr addrspace(2) %consts, ptr addrspace(1) %params, i32 %t) {{
entry:
  %pc = getelementptr inbounds %enc, ptr addrspace(1) %enc, i64 0, i32 0
  %cmd = load ptr addrspace(1), ptr addrspace(1) %pc, align 8
{body}  ret void
}}

declare void @air.set_barrier_render_command(ptr addrspace(1), i32)
{meta}"#
        )
    };
    let unread =
        kernel("  call void @air.set_barrier_render_command(ptr addrspace(1) %cmd, i32 %t)\n");
    let read = kernel(
        "  %ps = getelementptr inbounds %enc, ptr addrspace(1) %enc, i64 0, i32 1\n  \
         %ds = load ptr addrspace(1), ptr addrspace(1) %ps, align 8\n  \
         store ptr addrspace(1) %ds, ptr addrspace(1) %params, align 8\n",
    );
    let (Some(unread), Some(read)) = (assemble("ds_unread", &unread), assemble("ds_read", &read))
    else {
        return;
    };
    let (addressed, _) = profiles();
    assert!(!addressed.is_empty());
    for (stem, p) in addressed {
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

/// A kernel whose encoder argument buffer `{command_buffer commands; compute_pipeline_state
/// pipeline; ulong n;}` (Metal buffer 0, read-write) counts its threads in `n` with a `u64`
/// atomic, and sets a barrier on command `t`.
fn counted_kernel() -> String {
    r#"%enc = type { ptr addrspace(1), ptr addrspace(1), i64 }

define void @encode(ptr addrspace(1) %enc, ptr addrspace(1) %data, i32 %t) {
entry:
  %pc = getelementptr inbounds %enc, ptr addrspace(1) %enc, i64 0, i32 0
  %cmd = load ptr addrspace(1), ptr addrspace(1) %pc, align 8
  call void @air.set_barrier_compute_command(ptr addrspace(1) %cmd, i32 %t)
  %pn = getelementptr inbounds %enc, ptr addrspace(1) %enc, i64 0, i32 2
  %n = atomicrmw add ptr addrspace(1) %pn, i64 1 monotonic, align 8
  %w = trunc i64 %n to i32
  store i32 %w, ptr addrspace(1) %data, align 4
  ret void
}

declare void @air.set_barrier_compute_command(ptr addrspace(1), i32)

!air.kernel = !{!0}
!air.version = !{!20}
!air.language_version = !{!21}
!0 = !{ptr @encode, !1, !2}
!1 = !{}
!2 = !{!3, !7, !10}
!3 = !{i32 0, !"air.indirect_buffer", !"air.buffer_size", i32 24, !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.struct_type_info", !4, !"air.arg_type_size", i32 24, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"Encoder", !"air.arg_name", !"enc"}
!4 = !{i32 0, i32 8, i32 0, !"command_buffer", !"commands", !"air.indirect_argument", !5, i32 8, i32 8, i32 0, !"compute_pipeline_state", !"pipeline", !"air.indirect_argument", !6, i32 16, i32 8, i32 0, !"ulong", !"n", !"air.indirect_argument", !11}
!5 = !{i32 0, !"air.command_buffer", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"command_buffer", !"air.arg_name", !"commands"}
!6 = !{i32 1, !"air.compute_pipeline_state", !"air.location_index", i32 1, i32 1, !"air.arg_type_name", !"compute_pipeline_state", !"air.arg_name", !"pipeline"}
!11 = !{i32 2, !"air.indirect_constant", !"air.location_index", i32 2, i32 1, !"air.arg_type_name", !"ulong", !"air.arg_name", !"n"}
!7 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"data"}
!10 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"t"}
!20 = !{i32 2, i32 8, i32 0}
!21 = !{!"Metal", i32 4, i32 0, i32 0}
"#
    .to_owned()
}

/// An object word's load is named by the memory it reaches. Read through the argument buffer's
/// descriptor view, [`compute_kernel`]'s command-buffer and pipeline-state words are
/// `access.encode.word`; once a `u64` atomic on the buffer's counter lowers the buffer by device
/// address, the command-buffer word is `access.encode.word.psb`. A profile with device addresses
/// but no `u64` buffer atomics refuses the counter naming them.
#[test]
fn a_command_word_is_named_by_the_memory_it_reaches() {
    let Some(runs) = compiled("view", &compute_kernel()) else {
        return;
    };
    for (stem, _, plan, _) in &runs {
        let census = plan.census();
        let word = |name: &str| census.get(name).map_or(0, |(n, _)| *n);
        assert_eq!(word("access.encode.word"), 2, "{stem}");
        assert_eq!(word("access.encode.word.psb"), 0, "{stem}");
    }
    let Some(bc) = assemble("counted", &counted_kernel()) else {
        return;
    };
    let (addressed, _) = profiles();
    let mut compiled = 0;
    for (stem, p) in addressed {
        if !p.fields().leaf.buffer_int64_atomics {
            let Error::Unsupported(u) = compile(&bc, "encode", &p).expect_err(&stem) else {
                panic!("{stem}: not Unsupported");
            };
            assert_eq!(u.feature, Feature::Int64Atomics, "{stem}");
            continue;
        }
        let a = compile(&bc, "encode", &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
        let (plan, _) = typed_module(&bc, "encode", &p).expect("module");
        let census = plan.census();
        let word = |name: &str| census.get(name).map_or(0, |(n, _)| *n);
        assert_eq!(word("access.encode.word.psb"), 1, "{stem}");
        assert_eq!(word("access.encode.word"), 0, "{stem}");
        assert_eq!(word("access.encode.barrier"), 1, "{stem}");
        compiled += 1;
    }
    assert!(
        compiled > 0,
        "no committed profile has device addresses and u64 atomics"
    );
}

/// A kernel's `clear_barrier`, `set_stage_in_region`, `set_imageblock_size` and `reset` on a
/// compute command (the symbols and shapes Apple's compiler emits for `compute_command`'s members)
/// round-trip on every profile with device addresses: each even thread's
/// command, which they follow the dispatch of, decodes with its barrier cleared, its region, its
/// size (a width of 65535 held zero-extended), or as the default record but for the pipeline it
/// executes with, retained and reported cleared, and the bytes are exactly the encoded records.
#[test]
fn compute_command_setters_round_trip_through_the_host_contract() {
    let barrier = "call void @air.set_barrier_compute_command(ptr addrspace(1) %cmd, i32 %index)";
    type Expect = fn(&mut CommandRecord);
    let cases: [(&str, &str, &str, Expect); 4] = [
        (
            "clear",
            "call void @air.clear_barrier_compute_command(ptr addrspace(1) %cmd, i32 %index)",
            "declare void @air.clear_barrier_compute_command(ptr addrspace(1), i32)",
            |r| r.barrier = false,
        ),
        (
            "stage-in",
            "call void @air.set_stage_in_region_compute_command(ptr addrspace(1) %cmd, i32 %index, \
             <3 x i32> <i32 1, i32 2, i32 3>, <3 x i32> <i32 4, i32 5, i32 6>)",
            "declare void @air.set_stage_in_region_compute_command(ptr addrspace(1), i32, \
             <3 x i32>, <3 x i32>)",
            |r| {
                r.barrier = false;
                r.stage_in_region = [1, 2, 3, 4, 5, 6];
            },
        ),
        (
            "imageblock",
            "call void @air.set_imageblock_size_compute_command(ptr addrspace(1) %cmd, \
             i32 %index, <2 x i16> <i16 -1, i16 8>)",
            "declare void @air.set_imageblock_size_compute_command(ptr addrspace(1), i32, \
             <2 x i16>)",
            |r| {
                r.barrier = false;
                r.imageblock_size = [65535, 8];
            },
        ),
        (
            "reset",
            "call void @air.reset_compute_command(ptr addrspace(1) %cmd, i32 %index)",
            "declare void @air.reset_compute_command(ptr addrspace(1), i32)",
            // The pipeline the command executes with stays, reported cleared, and a kernel's
            // still: the one its write resolved.
            |r| {
                *r = CommandRecord {
                    retained_pipeline: r.bindings.pipeline,
                    kernel_pipeline: r.kernel_pipeline,
                    ..CommandRecord::default()
                }
            },
        ),
    ];
    for (name, call, declaration, expect) in cases {
        let kernel = compute_kernel();
        assert_eq!(kernel.matches(barrier).count(), 1);
        let kernel = kernel.replace(barrier, call).replace(
            "declare void @air.set_barrier_compute_command(ptr addrspace(1), i32)",
            declaration,
        );
        let Some(runs) = compiled(name, &kernel) else {
            return;
        };
        let descriptor = compute_descriptor(false);
        for (stem, _, plan, m) in &runs {
            let params: Vec<Params> = (0..THREADS).map(Params::valid).collect();
            let run = run_encoder(plan, m, storage(&descriptor, OWNED), &params, true)
                .unwrap_or_else(|e| panic!("{name} on {stem}: {e}"));
            let mut want = vec![CommandRecord::default(); THREADS as usize];
            for (t, p) in params.iter().enumerate() {
                let mut r = compute_record(t as u32, *p);
                if t % 2 == 0 {
                    expect(&mut r);
                }
                want[p.index as usize] = r;
            }
            let got = decoded(&run.records, &descriptor)
                .unwrap_or_else(|e| panic!("{name} on {stem}: {e}"));
            assert_eq!(got, want, "{name} on {stem}");
            let mut exact = storage(&descriptor, OWNED);
            for (i, r) in want.iter().enumerate() {
                let at = COMMAND_BUFFER_HEADER_BYTES + i * COMMAND_BYTES;
                r.encode_into(&mut exact[at..at + COMMAND_BYTES]).unwrap();
            }
            assert_eq!(run.records, exact, "{name} on {stem}: bytes");
            assert!(run.sink.iter().all(|b| *b == 0), "{name} on {stem}: sink");
        }
    }
}

/// A kernel whose argument buffer `{command_buffer commands; command_buffer source;}` (Metal
/// buffer 0, as [`compute_kernel`]'s) copies, in thread `t`, command `params[t].from` of `source`
/// onto command `params[t].index` of `commands` (`{u32 index, u32 from, ...}`, 32 bytes a
/// thread), then makes `after` of the copied command.
fn copy_kernel(after: &str) -> String {
    let meta = COMPUTE_META
        .replace(
            r#"i32 8, i32 8, i32 0, !"compute_pipeline_state", !"pipeline""#,
            r#"i32 8, i32 8, i32 0, !"command_buffer", !"source""#,
        )
        .replace(
            r#"!6 = !{i32 1, !"air.compute_pipeline_state", !"air.location_index", i32 1, i32 1, !"air.arg_type_name", !"compute_pipeline_state", !"air.arg_name", !"pipeline"}"#,
            r#"!6 = !{i32 1, !"air.command_buffer", !"air.location_index", i32 1, i32 1, !"air.arg_type_name", !"command_buffer", !"air.arg_name", !"source"}"#,
        );
    assert!(
        !meta.contains("pipeline"),
        "the source member replaced the pipeline's"
    );
    format!(
        r#"%enc = type {{ ptr addrspace(1), ptr addrspace(1) }}

define void @encode(ptr addrspace(1) %enc, ptr addrspace(1) %data, ptr addrspace(2) %consts, ptr addrspace(1) %params, i32 %t) {{
entry:
  %pc = getelementptr inbounds %enc, ptr addrspace(1) %enc, i64 0, i32 0
  %cmd = load ptr addrspace(1), ptr addrspace(1) %pc, align 8
  %ps = getelementptr inbounds %enc, ptr addrspace(1) %enc, i64 0, i32 1
  %src = load ptr addrspace(1), ptr addrspace(1) %ps, align 8
  %t64 = zext i32 %t to i64
  %prm = getelementptr inbounds [8 x i32], ptr addrspace(1) %params, i64 %t64
  %index = load i32, ptr addrspace(1) %prm, align 8
  %p1 = getelementptr inbounds i32, ptr addrspace(1) %prm, i64 1
  %from = load i32, ptr addrspace(1) %p1, align 4
  call void @air.copy_compute_command(ptr addrspace(1) %cmd, i32 %index, ptr addrspace(1) %src, i32 %from)
  {after}
  ret void
}}

declare void @air.copy_compute_command(ptr addrspace(1), i32, ptr addrspace(1), i32)
declare void @air.clear_barrier_compute_command(ptr addrspace(1), i32)
{meta}"#
    )
}

/// Run a [`copy_kernel`]: thread `t` copies command `copies[t].1` onto command `copies[t].0`, from
/// the record buffer `source` when there is one, else within `records`.
fn run_copies(
    plan: &Plan,
    m: &TypedModule,
    records: Vec<u8>,
    source: Option<Vec<u8>>,
    copies: [(u32, u32); THREADS as usize],
) -> Result<Run, InterpError> {
    let mut enc = RECORDS_ID.to_le_bytes().to_vec();
    let from = if source.is_some() {
        SOURCE_ID
    } else {
        RECORDS_ID
    };
    enc.extend_from_slice(&from.to_le_bytes());
    let mut params = Vec::new();
    for (index, from) in copies {
        params.extend_from_slice(&index.to_le_bytes());
        params.extend_from_slice(&from.to_le_bytes());
        params.extend_from_slice(&[0; 24]);
    }
    run_records(plan, m, enc, records, source, params, true)
}

/// Storage under `descriptor` and `inheritance` whose commands are `records`, as the host's
/// encoder writes them.
fn encoded(
    descriptor: &CommandDescriptor,
    inheritance: Inheritance,
    records: &[CommandRecord],
) -> Vec<u8> {
    let mut bytes = storage(descriptor, inheritance);
    for (i, r) in records.iter().enumerate() {
        let at = COMMAND_BUFFER_HEADER_BYTES + i * COMMAND_BYTES;
        r.encode_into(&mut bytes[at..at + COMMAND_BYTES]).unwrap();
    }
    bytes
}

/// Four full compute commands: [`compute_record`]'s, command 2 with a dynamic stride, each with
/// a stage-in region and an imageblock size of its own.
fn copied_sources() -> Vec<CommandRecord> {
    (0..THREADS)
        .map(|t| {
            let p = Params {
                stride: if t == 2 { 48 } else { -1 },
                ..Params::valid(t)
            };
            // The host encodes them: their pipelines are not a kernel's.
            let mut r = compute_record(t, p);
            r.kernel_pipeline = false;
            r.stage_in_region = [t, 1, 2, 3 + t, 4, 5];
            r.imageblock_size = [8 + t as u16, 4];
            r
        })
        .collect()
}

/// A kernel's `copy_command` between buffers of equal descriptors (on Metal, M3 and M4: the
/// destination executes as its source, and a setter after the copy writes over it) copies the whole
/// record on every profile with device addresses: between two buffers (each command from another
/// index, over full commands), within one (two commands onto empty ones, two onto themselves), and
/// with the barrier cleared after the copy; the source is left as it was and the bytes are exactly
/// the copied records.
#[test]
fn copied_commands_round_trip_between_equal_descriptors() {
    let clear = "call void @air.clear_barrier_compute_command(ptr addrspace(1) %cmd, i32 %index)";
    let descriptor = compute_descriptor(true);
    let sources = copied_sources();
    let full = encoded(&descriptor, OWNED, &sources);
    for (name, after) in [("copy", ""), ("copy then clear", clear)] {
        let Some(runs) = compiled(name, &copy_kernel(after)) else {
            return;
        };
        let cleared = |mut r: CommandRecord| {
            if !after.is_empty() {
                r.barrier = false;
            }
            r
        };
        for (stem, _, plan, m) in &runs {
            let census = plan.census();
            assert_eq!(census.get("access.encode.copy").map(|c| c.0), Some(1));
            // Between two buffers: command t from command 3 - t, over full commands.
            let run = run_copies(
                plan,
                m,
                full.clone(),
                Some(full.clone()),
                [(0, 3), (1, 2), (2, 1), (3, 0)],
            )
            .unwrap_or_else(|e| panic!("{name} on {stem}: {e}"));
            let want: Vec<CommandRecord> = sources.iter().rev().cloned().map(cleared).collect();
            assert_eq!(
                run.records,
                encoded(&descriptor, OWNED, &want),
                "{name} on {stem}"
            );
            assert_eq!(run.source.as_ref(), Some(&full), "{name} on {stem}: source");
            assert_eq!(decoded(&run.records, &descriptor).unwrap(), want);
            // Each destination's written flag set.
            let flags = vec![1; THREADS as usize];
            assert_eq!(run.flags, flags, "{name} on {stem}: written flags");
            // Within one buffer: commands 0 and 1 onto the empty 2 and 3, and onto themselves.
            let initial = encoded(&descriptor, OWNED, &sources[..2]);
            let run = run_copies(plan, m, initial, None, [(2, 0), (3, 1), (0, 0), (1, 1)])
                .unwrap_or_else(|e| panic!("{name} on {stem}: {e}"));
            let pair = [sources[0].clone(), sources[1].clone()].map(cleared);
            let want = [pair.clone(), pair].concat();
            assert_eq!(
                run.records,
                encoded(&descriptor, OWNED, &want),
                "{name} on {stem}"
            );
            assert_eq!(run.flags, flags, "{name} on {stem}: written flags");
            assert!(run.sink.iter().all(|b| *b == 0), "{name} on {stem}: sink");
        }
    }
}

/// A copy into a buffer that inherits its pipeline and its buffers from one that inherits neither
/// (of the same descriptor otherwise, the destination binding no kernel buffer) copies every word
/// but the pipeline's and the buffer bindings', which stay the destination's: on Metal (M3 and M4)
/// the destination executes the source's dispatch, barrier and threadgroup length with the
/// encoder's pipeline and buffers. The records decode under the destination's inheritance. The
/// destination holds five commands and the copies land on commands 1 to 4: a copy onto command 0
/// faults (Metal's runs it with zero grid sizes; the fault test).
#[test]
fn a_copy_into_a_buffer_that_inherits_keeps_its_pipeline_and_buffers() {
    let Some(runs) = compiled("into_inheriting", &copy_kernel("")) else {
        return;
    };
    let source_descriptor = compute_descriptor(false);
    let descriptor = CommandDescriptor::Compute {
        dispatch: ComputeDispatch::Both,
        max_buffer_bind_count: 0,
        max_threadgroup_memory_bind_count: 3,
        support_dynamic_attribute_stride: false,
    };
    let inherit = Inheritance {
        pipeline: StateSource::Parent,
        buffers: StateSource::Parent,
    };
    // The host encodes them: their pipelines are not a kernel's.
    let sources: Vec<CommandRecord> = (0..THREADS)
        .map(|t| CommandRecord {
            kernel_pipeline: false,
            ..compute_record(t, Params::valid(t))
        })
        .collect();
    let source = encoded(&source_descriptor, OWNED, &sources);
    let capacity = THREADS + 1;
    let mut initial = vec![0xa5; COMMAND_BUFFER_HEADER_BYTES + capacity as usize * COMMAND_BYTES];
    CommandBuffer::initialize(&mut initial, capacity, &descriptor, inherit).expect("initialize");
    let mut want: Vec<CommandRecord> = sources
        .iter()
        .rev()
        .map(|r| CommandRecord {
            bindings: Default::default(),
            ..r.clone()
        })
        .collect();
    want.insert(0, CommandRecord::default());
    let mut bytes = initial.clone();
    for (i, r) in want.iter().enumerate() {
        let at = COMMAND_BUFFER_HEADER_BYTES + i * COMMAND_BYTES;
        r.encode_into(&mut bytes[at..at + COMMAND_BYTES]).unwrap();
    }
    for (stem, _, plan, m) in &runs {
        let run = run_copies(
            plan,
            m,
            initial.clone(),
            Some(source.clone()),
            [(1, 3), (2, 2), (3, 1), (4, 0)],
        )
        .unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert_eq!(run.records, bytes, "{stem}");
        let got: Vec<CommandRecord> = CommandBuffer::decode(&run.records, &descriptor, inherit)
            .unwrap_or_else(|e| panic!("{stem}: {e}"))
            .records(0..capacity)
            .unwrap()
            .collect();
        assert_eq!(got, want, "{stem}");
        assert_eq!(run.source.as_ref(), Some(&source), "{stem}: source");
    }
}

/// A copy from a command that reports no pipeline (none set, or one its reset retained) keeps the
/// destination's pipeline as it was, reported or retained, and copies every other word; a copy from
/// one that reports a pipeline replaces the destination's (Metal's copy, measured on the M3 Max and
/// the M4: a copy of a fresh or a reset command leaves the destination's pipeline readable and
/// running).
#[test]
fn a_copy_from_a_command_without_a_pipeline_keeps_the_destinations() {
    let Some(runs) = compiled("copy_pipelines", &copy_kernel("")) else {
        return;
    };
    let descriptor = compute_descriptor(false);
    let own = NonZeroU64::new(TOKEN);
    let other = NonZeroU64::new(0x77);
    let sources = copied_sources();
    let named = |mut r: CommandRecord, pipeline| {
        r.bindings.pipeline = pipeline;
        r
    };
    let retaining = |mut r: CommandRecord, pipeline| {
        r.bindings.pipeline = None;
        r.retained_pipeline = pipeline;
        r
    };
    // Destination: full commands 0, 1 and 3 (pipeline `own`), command 2 reset (`own` retained).
    let destination = [
        named(sources[0].clone(), own),
        named(sources[1].clone(), own),
        retaining(CommandRecord::default(), own),
        named(sources[3].clone(), own),
    ];
    // Source: 0 fresh, 1 reset with `other` retained, 2 fresh, 3 with `other` set.
    let source = [
        CommandRecord::default(),
        retaining(sources[1].clone(), other),
        CommandRecord::default(),
        named(sources[3].clone(), other),
    ];
    let want = [
        named(CommandRecord::default(), own),
        named(retaining(sources[1].clone(), None), own),
        retaining(CommandRecord::default(), own),
        named(sources[3].clone(), other),
    ];
    for (stem, _, plan, m) in &runs {
        let run = run_copies(
            plan,
            m,
            encoded(&descriptor, OWNED, &destination),
            Some(encoded(&descriptor, OWNED, &source)),
            [(0, 0), (1, 1), (2, 2), (3, 3)],
        )
        .unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert_eq!(decoded(&run.records, &descriptor).unwrap(), want, "{stem}");
        assert_eq!(run.records, encoded(&descriptor, OWNED, &want), "{stem}");
    }
}

/// A copy whose result on Metal depends on its private record layout, or was not measured, sets the
/// destination's sticky fault and copies nothing, so `decode` fails the run as an encoding fault: a
/// copy between descriptors of other dispatch types (on Metal the destination ran a dispatch its
/// descriptor does not allow), of fewer bindings (Metal's overwrote the next command), from a
/// buffer that inherits (Metal's executed a garbage pipeline), of another dynamic stride, into one
/// that inherits the pipeline alone or binds kernel buffers while inheriting them (not measured),
/// onto command 0 of one that inherits both (Metal's ran it with `threads_per_grid` and
/// `threadgroups_per_grid` zero, M3 Max and M4), a source or destination index at its capacity
/// (Metal's corrupted a command), and into a render command buffer. Every thread makes the same
/// copy, from full commands.
#[test]
fn a_copy_whose_result_is_metals_private_layout_faults_and_copies_nothing() {
    let Some(runs) = compiled("copy_faults", &copy_kernel("")) else {
        return;
    };
    let compute = |dispatch, buffers, dynamic| CommandDescriptor::Compute {
        dispatch,
        max_buffer_bind_count: buffers,
        max_threadgroup_memory_bind_count: 3,
        support_dynamic_attribute_stride: dynamic,
    };
    let pipeline = Inheritance {
        pipeline: StateSource::Parent,
        buffers: StateSource::Command,
    };
    let both = Inheritance {
        pipeline: StateSource::Parent,
        buffers: StateSource::Parent,
    };
    let render = CommandDescriptor::Render {
        draws: RenderDraws::PRIMITIVES,
        max_vertex_buffer_bind_count: 3,
        max_fragment_buffer_bind_count: 3,
        max_object_buffer_bind_count: 0,
        max_mesh_buffer_bind_count: 0,
        support_dynamic_attribute_stride: false,
    };
    let owned = || compute(ComputeDispatch::Both, 3, false);
    let groups = compute(ComputeDispatch::Threadgroups, 3, false);
    let threads = compute(ComputeDispatch::Threads, 3, false);
    // (what, destination, its inheritance, source, its inheritance, each thread's copy)
    type Fault = (
        &'static str,
        CommandDescriptor,
        Inheritance,
        CommandDescriptor,
        Inheritance,
        (u32, u32),
    );
    let faults: [Fault; 10] = [
        ("other dispatch type", threads, OWNED, groups, OWNED, (0, 0)),
        (
            "fewer bindings",
            compute(ComputeDispatch::Both, 2, false),
            OWNED,
            owned(),
            OWNED,
            (0, 0),
        ),
        (
            "other dynamic stride",
            compute(ComputeDispatch::Both, 3, true),
            OWNED,
            owned(),
            OWNED,
            (0, 0),
        ),
        (
            "inheriting the pipeline alone",
            owned(),
            pipeline,
            owned(),
            OWNED,
            (1, 0),
        ),
        (
            "from a buffer that inherits",
            owned(),
            OWNED,
            compute(ComputeDispatch::Both, 0, false),
            both,
            (0, 0),
        ),
        (
            "into one that inherits and binds kernel buffers",
            owned(),
            both,
            owned(),
            OWNED,
            (1, 0),
        ),
        (
            "onto command 0 of one that inherits",
            compute(ComputeDispatch::Both, 0, false),
            both,
            owned(),
            OWNED,
            (0, 1),
        ),
        (
            "source index at capacity",
            owned(),
            OWNED,
            owned(),
            OWNED,
            (0, THREADS),
        ),
        (
            "destination index at capacity",
            owned(),
            OWNED,
            owned(),
            OWNED,
            (THREADS, 0),
        ),
        (
            "into a render command buffer",
            render,
            OWNED,
            owned(),
            OWNED,
            (0, 0),
        ),
    ];
    for (stem, _, plan, m) in &runs {
        for (what, descriptor, inheritance, from, from_inheritance, copy) in &faults {
            let records: Vec<CommandRecord> = copied_sources()
                .into_iter()
                .map(|r| match from_inheritance.buffers {
                    StateSource::Parent => CommandRecord {
                        bindings: Default::default(),
                        ..r
                    },
                    StateSource::Command => r,
                })
                .collect();
            let source = encoded(from, *from_inheritance, &records);
            let initial = storage(descriptor, *inheritance);
            let run = run_copies(plan, m, initial.clone(), Some(source.clone()), [*copy; 4])
                .unwrap_or_else(|e| panic!("{what} on {stem}: {e}"));
            let mut want = initial;
            want[4..8].copy_from_slice(&1u32.to_le_bytes());
            assert_eq!(run.records, want, "{what} on {stem}");
            assert_eq!(
                CommandBuffer::decode(&run.records, descriptor, *inheritance).err(),
                Some(CommandBufferError::EncodingFault(1)),
                "{what} on {stem}"
            );
            assert_eq!(
                run.source.as_ref(),
                Some(&source),
                "{what} on {stem}: source"
            );
        }
    }
}

/// A kernel's render pipeline setter, indexed draws (`ushort` indices in device memory, `uint`
/// ones in constant memory, a negative base vertex) and render reset decode under a render
/// descriptor of indexed draws, each value where the record format holds it: the index buffer as
/// its guest address, the index type from the overload. Under a descriptor of draws of primitives
/// alone, the indexed draw is outside the policy: the header's fault flag fails the decode.
#[test]
fn encoded_indexed_draws_round_trip_through_the_host_contract() {
    let Some(runs) = compiled("render_indexed", &render_indexed_kernel()) else {
        return;
    };
    let indexed = CommandDescriptor::Render {
        draws: RenderDraws {
            primitives: false,
            indexed: true,
            patches: false,
            indexed_patches: false,
            mesh_threadgroups: false,
            mesh_threads: false,
        },
        max_vertex_buffer_bind_count: 1,
        max_fragment_buffer_bind_count: 1,
        max_object_buffer_bind_count: 0,
        max_mesh_buffer_bind_count: 0,
        support_dynamic_attribute_stride: false,
    };
    for (stem, _, plan, m) in &runs {
        let run = run_encoder(plan, m, storage(&indexed, OWNED), &[], true)
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        let want: Vec<CommandRecord> = (0..THREADS)
            .map(|t| {
                if t == 3 {
                    return CommandRecord::default();
                }
                let action = if t % 2 == 0 {
                    Action::DrawIndexedPrimitives {
                        primitive_type: 3,
                        index_count: 6,
                        index_type: 0,
                        index_buffer: DATA_GUEST + 16 * u64::from(t),
                        instance_count: 2,
                        base_vertex: -(t as i32),
                        base_instance: 7,
                    }
                } else {
                    Action::DrawIndexedPrimitives {
                        primitive_type: 0,
                        index_count: 4,
                        index_type: 1,
                        index_buffer: CONSTS_GUEST,
                        instance_count: 1,
                        base_vertex: 0,
                        base_instance: 0,
                    }
                };
                let mut r = CommandRecord {
                    action,
                    kernel_pipeline: true,
                    ..CommandRecord::default()
                };
                r.bindings.pipeline = NonZeroU64::new(TOKEN);
                r
            })
            .collect();
        let got = decoded(&run.records, &indexed).unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert_eq!(got, want, "{stem}");
        let primitives = CommandDescriptor::Render {
            draws: RenderDraws::PRIMITIVES,
            max_vertex_buffer_bind_count: 1,
            max_fragment_buffer_bind_count: 1,
            max_object_buffer_bind_count: 0,
            max_mesh_buffer_bind_count: 0,
            support_dynamic_attribute_stride: false,
        };
        let run = run_encoder(plan, m, storage(&primitives, OWNED), &[], true)
            .unwrap_or_else(|e| panic!("{stem} primitives: {e}"));
        assert_eq!(
            decoded(&run.records, &primitives).err(),
            Some(CommandBufferError::EncodingFault(1)),
            "{stem} primitives"
        );
    }
}

/// A kernel's indexed patch draws decode under a render descriptor of indexed patch draws: a patch
/// draw's arguments and the control-point index buffer's guest address (0 for null), from device
/// and constant memory, where the record keeps it. A descriptor of patch draws alone holds them
/// too; under one of indexed draws alone, whose records do not, the indexed patch draw is outside
/// the policy: the decode fails.
#[test]
fn encoded_indexed_patch_draws_round_trip_through_the_host_contract() {
    let Some(runs) = compiled("render_indexed_patches", &render_indexed_patches_kernel()) else {
        return;
    };
    let draws = |patches, indexed_patches| CommandDescriptor::Render {
        draws: RenderDraws {
            primitives: false,
            indexed: false,
            patches,
            indexed_patches,
            mesh_threadgroups: false,
            mesh_threads: false,
        },
        max_vertex_buffer_bind_count: 1,
        max_fragment_buffer_bind_count: 1,
        max_object_buffer_bind_count: 0,
        max_mesh_buffer_bind_count: 0,
        support_dynamic_attribute_stride: false,
    };
    let indexed = draws(false, true);
    for (stem, _, plan, m) in &runs {
        let run = run_encoder(plan, m, storage(&indexed, OWNED), &[], true)
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        let want: Vec<CommandRecord> = (0..THREADS)
            .map(|t| {
                let data = DATA_GUEST + 16 * u64::from(t);
                let action = if t % 2 == 0 {
                    Action::DrawIndexedPatches {
                        control_points: 4,
                        patch_start: t,
                        patch_count: 2,
                        patch_index_buffer: 0,
                        control_point_index_buffer: 0,
                        instance_count: 3,
                        base_instance: 1,
                        factor_buffer: data,
                        factor_instance_stride: 48,
                        factor_scale: 1.0f32.to_bits(),
                    }
                } else {
                    Action::DrawIndexedPatches {
                        control_points: 3,
                        patch_start: 0,
                        patch_count: 1,
                        patch_index_buffer: data,
                        control_point_index_buffer: CONSTS_GUEST,
                        instance_count: 1,
                        base_instance: 0,
                        factor_buffer: data,
                        factor_instance_stride: 0,
                        factor_scale: 0.5f32.to_bits(),
                    }
                };
                CommandRecord {
                    action,
                    ..CommandRecord::default()
                }
            })
            .collect();
        let got = decoded(&run.records, &indexed).unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert_eq!(got, want, "{stem}");
        let patches = draws(true, false);
        let run = run_encoder(plan, m, storage(&patches, OWNED), &[], true)
            .unwrap_or_else(|e| panic!("{stem} patches: {e}"));
        let got = decoded(&run.records, &patches).unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert_eq!(got, want, "{stem} patches");
        let indexed_draws = CommandDescriptor::Render {
            draws: RenderDraws {
                primitives: false,
                indexed: true,
                patches: false,
                indexed_patches: false,
                mesh_threadgroups: false,
                mesh_threads: false,
            },
            max_vertex_buffer_bind_count: 1,
            max_fragment_buffer_bind_count: 1,
            max_object_buffer_bind_count: 0,
            max_mesh_buffer_bind_count: 0,
            support_dynamic_attribute_stride: false,
        };
        let run = run_encoder(plan, m, storage(&indexed_draws, OWNED), &[], true)
            .unwrap_or_else(|e| panic!("{stem} indexed draws: {e}"));
        assert_eq!(
            decoded(&run.records, &indexed_draws).err(),
            Some(CommandBufferError::EncodingFault(1)),
            "{stem} indexed draws"
        );
    }
}

/// A kernel's patch draws decode under a render descriptor of patch draws: the patch index and
/// factor buffers as guest addresses (0 for a null patch index buffer), from device and constant
/// memory, and the factor scale's bits, as the record format holds them. Under a descriptor of
/// draws of primitives alone, the patch draw is outside the policy: the decode fails.
#[test]
fn encoded_patch_draws_round_trip_through_the_host_contract() {
    let Some(runs) = compiled("render_patches", &render_patches_kernel()) else {
        return;
    };
    let patches = CommandDescriptor::Render {
        draws: RenderDraws {
            primitives: false,
            indexed: false,
            patches: true,
            indexed_patches: false,
            mesh_threadgroups: false,
            mesh_threads: false,
        },
        max_vertex_buffer_bind_count: 1,
        max_fragment_buffer_bind_count: 1,
        max_object_buffer_bind_count: 0,
        max_mesh_buffer_bind_count: 0,
        support_dynamic_attribute_stride: false,
    };
    for (stem, _, plan, m) in &runs {
        let run = run_encoder(plan, m, storage(&patches, OWNED), &[], true)
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        let want: Vec<CommandRecord> = (0..THREADS)
            .map(|t| {
                let data = DATA_GUEST + 16 * u64::from(t);
                let action = if t % 2 == 0 {
                    Action::DrawPatches {
                        control_points: 4,
                        patch_start: t,
                        patch_count: 2,
                        patch_index_buffer: 0,
                        instance_count: 3,
                        base_instance: 1,
                        factor_buffer: data,
                        factor_instance_stride: 48,
                        factor_scale: 1.0f32.to_bits(),
                    }
                } else {
                    Action::DrawPatches {
                        control_points: 3,
                        patch_start: 0,
                        patch_count: 1,
                        patch_index_buffer: data,
                        instance_count: 1,
                        base_instance: 0,
                        factor_buffer: CONSTS_GUEST,
                        factor_instance_stride: 0,
                        factor_scale: 0.5f32.to_bits(),
                    }
                };
                CommandRecord {
                    action,
                    ..CommandRecord::default()
                }
            })
            .collect();
        let got = decoded(&run.records, &patches).unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert_eq!(got, want, "{stem}");
        let primitives = CommandDescriptor::Render {
            draws: RenderDraws::PRIMITIVES,
            max_vertex_buffer_bind_count: 1,
            max_fragment_buffer_bind_count: 1,
            max_object_buffer_bind_count: 0,
            max_mesh_buffer_bind_count: 0,
            support_dynamic_attribute_stride: false,
        };
        let run = run_encoder(plan, m, storage(&primitives, OWNED), &[], true)
            .unwrap_or_else(|e| panic!("{stem} primitives: {e}"));
        assert_eq!(
            decoded(&run.records, &primitives).err(),
            Some(CommandBufferError::EncodingFault(1)),
            "{stem} primitives"
        );
    }
}
