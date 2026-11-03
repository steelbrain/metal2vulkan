//! Mesh draws in indirect command buffers on the device the cfg selects, under the validation
//! layer: a Metal probe drawn through `metal2vulkan-runtime` with the probe's functions, the owned
//! fixtures `mesh_draws` (the grid and object pipelines over two `RGBA32Uint` attachments, as
//! `mesh_draws.rs` draws them), `mesh_barrier` (a writer and an object reader over one) and
//! `mesh_commands` (the kernels that encode the commands, and the vertex function `vtx`). Each case
//! executes a buffer's commands in one 8x8 pass, its attachments cleared to zero and stored, and
//! prints the pixels as the probe printed them.
//!
//! [`ANSWERS`] holds the probe's lines, as an M4 (macOS 27.0.1) printed them without API
//! validation; the M3 Max (macOS 26.7) printed the same, with and without it. Among them:
//!
//! - A fresh descriptor binds no object or mesh buffers. A buffer of draws of primitives beside
//!   mesh draws, and one of both mesh kinds, are made; one of mesh draws beside dispatches is a
//!   buffer of compute commands, as plain Metal makes it where its validation stops it.
//! - Commands with pipeline states of their own, an object buffer, and a mesh draw of each kind
//!   draw what the encoder's mesh draws draw, set by the CPU and by a kernel (`own`).
//! - A kernel's mesh draw of either kind runs on a buffer of either mesh kind; on a buffer of
//!   draws of primitives Metal's GPU faults, and the runtime runs no command and reports the
//!   encoding fault. A kernel's draw of primitives on a buffer of mesh draws alone draws nothing
//!   on Metal, and the runtime drops it (`kinds`).
//! - The CPU's mesh draw of the other mesh kind draws, as plain Metal draws it where its
//!   validation stops it; one on a buffer of draws of primitives ends Metal's process, and the
//!   runtime refuses it (`CommandType`).
//! - A command's barrier, set by the CPU or a kernel, orders its draw after the earlier one: the
//!   reader's object function reads the words the writer's mesh function stored. Without one,
//!   Metal's reader read the words the CPU wrote, but the two draws are not ordered: the lane
//!   takes either order's words, and prints which (a device that emulates mesh draws orders
//!   them).
//! - A kernel's reset of a command leaves it drawing nothing.
//! - A mesh command that reads what an earlier draw of its pass wrote (an M4): Metal's reader reads
//!   none of a fragment's writes, behind a barrier or not, and a vertex function's behind the
//!   command's barrier; a device that emulates mesh draws reads so, and refuses the vertex
//!   function's cases without that barrier (`NotImplemented(IndirectRenderCommands)`).
//! - A pipeline state set on a command of a buffer whose command types hold no draw of its kind
//!   (the vertex pipeline on a buffer of mesh draws alone, a mesh pipeline on a buffer of draws
//!   of primitives, of indexed draws or of patch draws) ends Metal's process in the setter (an M4,
//!   with and without API validation), and the runtime refuses it (`PipelineKind`); so does the
//!   CPU's draw of primitives on a buffer of mesh draws alone, refused `CommandType`.
//!
//! Commands a kernel writes are drawn by a host step, which the device announces once; a device
//! without `VK_EXT_mesh_shader` announces its emulated mesh draws once.

use super::degrade_notices::on_device;
use super::mesh_draws::emulates;
use super::render_state::{texels, texture};
use super::{buffer_with, read_buffer, WAIT};
use crate::air_library::AirLibrary;
use metal2vulkan::bind::ReplayFaults;
use metal2vulkan::metal::{
    BarrierScope, IndirectCommandType, PixelFormat, PrimitiveTopologyClass, PrimitiveType,
    RenderStages, ResourceUsage,
};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, ComputePipelineDescriptor, MeshPipelineDescriptor,
    RenderPipelineDescriptor, StageSource,
};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    Buffer, ClearColor, CommandBuffer, ComputePipelineState, Device, Error, IndirectCommandBuffer,
    IndirectCommandBufferDescriptor, IndirectCommandBufferProblem, Invalid, Library, LoadAction,
    PipelineOptions, Problem, RenderCommandEncoder, RenderPassDescriptor, RenderPipelineState,
    ResourceOptions, RuntimeGap, Size, StoreAction,
};

/// a Metal probe, as an M4 printed it without API validation: `PX <case> <x> <y> a=<words>
/// [b=<words in hex>]` for each pixel a case drew, `RESULT <case> drawn=<pixels>`, `CTL <case>
/// <words>` (the control words after the pass), and `case <case> marks=<words in hex>` (the words
/// the vertex function marked); then the cases of
/// [`mesh_commands_read_earlier_draws_writes_as_metal_orders_them`], as the same M4 printed them
/// without API validation (macOS 27.0.1; with it the same, but `enc_frag_barrier`, whose barrier
/// from the fragment stage Metal's validation stops), the vertex writer's control words in
/// hexadecimal.
const ANSWERS: &str = "\
PX cpu_own 0 0 a=0,0,3,2 b=c,4,100804,0
PX cpu_own 1 0 a=1,0,3,2 b=c,4,100804,4
PX cpu_own 2 0 a=2,0,3,2 b=c,4,100804,8
PX cpu_own 0 1 a=0,1,3,2 b=c,4,100804,20000
PX cpu_own 1 1 a=1,1,3,2 b=c,4,100804,20004
PX cpu_own 2 1 a=2,1,3,2 b=c,4,100804,20008
PX cpu_own 0 2 a=1,65539,1049604,65548 b=0,0,10003,100402
PX cpu_own 1 2 a=1,65539,1049604,65548 b=1,0,10003,100402
PX cpu_own 2 2 a=1,65539,1049604,65548 b=2,0,10003,100402
PX cpu_own 0 4 a=2,65539,1049604,65548 b=0,0,10004,100402
PX cpu_own 1 4 a=2,65539,1049604,65548 b=1,0,10004,100402
PX cpu_own 2 4 a=2,65539,1049604,65548 b=2,0,10004,100402
PX cpu_own 3 4 a=2,65539,1049604,65548 b=3,0,10004,100402
RESULT cpu_own drawn=13
PX kernel_own 0 0 a=0,0,3,2 b=c,4,100804,0
PX kernel_own 1 0 a=1,0,3,2 b=c,4,100804,4
PX kernel_own 2 0 a=2,0,3,2 b=c,4,100804,8
PX kernel_own 0 1 a=0,1,3,2 b=c,4,100804,20000
PX kernel_own 1 1 a=1,1,3,2 b=c,4,100804,20004
PX kernel_own 2 1 a=2,1,3,2 b=c,4,100804,20008
PX kernel_own 0 2 a=1,65539,1049604,65548 b=0,0,10003,100402
PX kernel_own 1 2 a=1,65539,1049604,65548 b=1,0,10003,100402
PX kernel_own 2 2 a=1,65539,1049604,65548 b=2,0,10003,100402
PX kernel_own 0 4 a=2,65539,1049604,65548 b=0,0,10004,100402
PX kernel_own 1 4 a=2,65539,1049604,65548 b=1,0,10004,100402
PX kernel_own 2 4 a=2,65539,1049604,65548 b=2,0,10004,100402
PX kernel_own 3 4 a=2,65539,1049604,65548 b=3,0,10004,100402
RESULT kernel_own drawn=13
PX kernel_groups_on_groups 0 0 a=0,0,3,2 b=c,2,100404,0
PX kernel_groups_on_groups 1 0 a=1,0,3,2 b=c,2,100404,4
PX kernel_groups_on_groups 2 0 a=2,0,3,2 b=c,2,100404,8
PX kernel_groups_on_groups 0 1 a=0,1,3,2 b=c,2,100404,10000
PX kernel_groups_on_groups 1 1 a=1,1,3,2 b=c,2,100404,10004
PX kernel_groups_on_groups 2 1 a=2,1,3,2 b=c,2,100404,10008
RESULT kernel_groups_on_groups drawn=6
case kernel_groups_on_groups marks=0,0,0,0
PX kernel_groups_on_threads 0 0 a=0,0,3,2 b=c,2,100404,0
PX kernel_groups_on_threads 1 0 a=1,0,3,2 b=c,2,100404,4
PX kernel_groups_on_threads 2 0 a=2,0,3,2 b=c,2,100404,8
PX kernel_groups_on_threads 0 1 a=0,1,3,2 b=c,2,100404,10000
PX kernel_groups_on_threads 1 1 a=1,1,3,2 b=c,2,100404,10004
PX kernel_groups_on_threads 2 1 a=2,1,3,2 b=c,2,100404,10008
RESULT kernel_groups_on_threads drawn=6
case kernel_groups_on_threads marks=0,0,0,0
case kernel_groups_on_draw status=5
case kernel_groups_on_draw marks=0,0,0,0
PX kernel_threads_on_groups 0 0 a=0,0,3,2 b=c,4,100804,0
PX kernel_threads_on_groups 1 0 a=1,0,3,2 b=c,4,100804,4
PX kernel_threads_on_groups 2 0 a=2,0,3,2 b=c,4,100804,8
PX kernel_threads_on_groups 0 1 a=0,1,3,2 b=c,4,100804,20000
PX kernel_threads_on_groups 1 1 a=1,1,3,2 b=c,4,100804,20004
PX kernel_threads_on_groups 2 1 a=2,1,3,2 b=c,4,100804,20008
RESULT kernel_threads_on_groups drawn=6
case kernel_threads_on_groups marks=0,0,0,0
PX kernel_threads_on_threads 0 0 a=0,0,3,2 b=c,4,100804,0
PX kernel_threads_on_threads 1 0 a=1,0,3,2 b=c,4,100804,4
PX kernel_threads_on_threads 2 0 a=2,0,3,2 b=c,4,100804,8
PX kernel_threads_on_threads 0 1 a=0,1,3,2 b=c,4,100804,20000
PX kernel_threads_on_threads 1 1 a=1,1,3,2 b=c,4,100804,20004
PX kernel_threads_on_threads 2 1 a=2,1,3,2 b=c,4,100804,20008
RESULT kernel_threads_on_threads drawn=6
case kernel_threads_on_threads marks=0,0,0,0
case kernel_threads_on_draw status=5
case kernel_threads_on_draw marks=0,0,0,0
RESULT kernel_draw_on_groups drawn=0
case kernel_draw_on_groups marks=0,0,0,0
RESULT kernel_draw_on_threads drawn=0
case kernel_draw_on_threads marks=0,0,0,0
RESULT kernel_draw_on_draw drawn=0
case kernel_draw_on_draw marks=d0000000,d0000001,d0000002,d0000003
PX cpu_threads_on_groups 0 0 a=0,0,3,2 b=c,4,100804,0
PX cpu_threads_on_groups 1 0 a=1,0,3,2 b=c,4,100804,4
PX cpu_threads_on_groups 2 0 a=2,0,3,2 b=c,4,100804,8
PX cpu_threads_on_groups 0 1 a=0,1,3,2 b=c,4,100804,20000
PX cpu_threads_on_groups 1 1 a=1,1,3,2 b=c,4,100804,20004
PX cpu_threads_on_groups 2 1 a=2,1,3,2 b=c,4,100804,20008
RESULT cpu_threads_on_groups drawn=6
PX cpu_groups_on_threads 0 0 a=0,0,3,2 b=c,2,100404,0
PX cpu_groups_on_threads 1 0 a=1,0,3,2 b=c,2,100404,4
PX cpu_groups_on_threads 2 0 a=2,0,3,2 b=c,2,100404,8
PX cpu_groups_on_threads 0 1 a=0,1,3,2 b=c,2,100404,10000
PX cpu_groups_on_threads 1 1 a=1,1,3,2 b=c,2,100404,10004
PX cpu_groups_on_threads 2 1 a=2,1,3,2 b=c,2,100404,10008
RESULT cpu_groups_on_threads drawn=6
PX cpu_barrier 0 0 a=0,0,1,7
PX cpu_barrier 1 0 a=1,0,1,7
PX cpu_barrier 2 0 a=2,0,1,7
PX cpu_barrier 0 1 a=0,1,1,7
PX cpu_barrier 1 1 a=1,1,1,7
PX cpu_barrier 2 1 a=2,1,1,7
PX cpu_barrier 7 7 a=43690,0,0,0
RESULT cpu_barrier drawn=7
CTL cpu_barrier 3,2,1,7
PX cpu_nobarrier 0 0 a=0,0,1,0
PX cpu_nobarrier 7 7 a=43690,0,0,0
RESULT cpu_nobarrier drawn=2
CTL cpu_nobarrier 3,2,1,7
PX kernel_barrier 0 0 a=0,0,1,7
PX kernel_barrier 1 0 a=1,0,1,7
PX kernel_barrier 2 0 a=2,0,1,7
PX kernel_barrier 0 1 a=0,1,1,7
PX kernel_barrier 1 1 a=1,1,1,7
PX kernel_barrier 2 1 a=2,1,1,7
PX kernel_barrier 7 7 a=43690,0,0,0
RESULT kernel_barrier drawn=7
CTL kernel_barrier 3,2,1,7
PX kernel_nobarrier 0 0 a=0,0,1,0
PX kernel_nobarrier 7 7 a=43690,0,0,0
RESULT kernel_nobarrier drawn=2
CTL kernel_nobarrier 3,2,1,7
PX kernel_reset 0 0 a=0,0,3,2 b=c,2,100404,0
PX kernel_reset 1 0 a=1,0,3,2 b=c,2,100404,4
PX kernel_reset 2 0 a=2,0,3,2 b=c,2,100404,8
PX kernel_reset 0 1 a=0,1,3,2 b=c,2,100404,10000
PX kernel_reset 1 1 a=1,1,3,2 b=c,2,100404,10004
PX kernel_reset 2 1 a=2,1,3,2 b=c,2,100404,10008
RESULT kernel_reset drawn=6
PX icb_frag_object_nobarrier 0 0 a=0,0,1,0
PX icb_frag_object_nobarrier 7 7 a=48059,0,0,0
RESULT icb_frag_object_nobarrier drawn=2
CTL icb_frag_object_nobarrier 3,2,1,9
PX icb_frag_object_barrier 0 0 a=0,0,1,0
PX icb_frag_object_barrier 7 7 a=48059,0,0,0
RESULT icb_frag_object_barrier drawn=2
CTL icb_frag_object_barrier 3,2,1,9
PX icb_frag_grid_nobarrier 0 0 a=0,0,1,0
PX icb_frag_grid_nobarrier 7 7 a=48059,0,0,0
RESULT icb_frag_grid_nobarrier drawn=2
CTL icb_frag_grid_nobarrier 3,2,1,9
PX icb_frag_grid_barrier 0 0 a=0,0,1,0
PX icb_frag_grid_barrier 7 7 a=48059,0,0,0
RESULT icb_frag_grid_barrier drawn=2
CTL icb_frag_grid_barrier 3,2,1,9
PX enc_frag_nobarrier 0 0 a=0,0,1,0
PX enc_frag_nobarrier 7 7 a=48059,0,0,0
RESULT enc_frag_nobarrier drawn=2
CTL enc_frag_nobarrier 3,2,1,9
PX enc_frag_barrier 0 0 a=0,0,1,0
PX enc_frag_barrier 7 7 a=48059,0,0,0
RESULT enc_frag_barrier drawn=2
CTL enc_frag_barrier 3,2,1,9
PX enc_mesh_barrier 0 0 a=0,0,1,7
PX enc_mesh_barrier 1 0 a=1,0,1,7
PX enc_mesh_barrier 2 0 a=2,0,1,7
PX enc_mesh_barrier 0 1 a=0,1,1,7
PX enc_mesh_barrier 1 1 a=1,1,1,7
PX enc_mesh_barrier 2 1 a=2,1,1,7
PX enc_mesh_barrier 7 7 a=43690,0,0,0
RESULT enc_mesh_barrier drawn=7
CTL enc_mesh_barrier 3,2,1,7
PX icb_vtx_nobarrier 0 0 a=0,0,1,0
RESULT icb_vtx_nobarrier drawn=1
CTL icb_vtx_nobarrier d0000000,d0000001,d0000002,d0000003
PX icb_vtx_barrier 0 0 a=0,0,3489660930,3489660931
PX icb_vtx_barrier 1 0 a=1,0,3489660930,3489660931
PX icb_vtx_barrier 2 0 a=2,0,3489660930,3489660931
PX icb_vtx_barrier 3 0 a=3,0,3489660930,3489660931
PX icb_vtx_barrier 0 1 a=0,1,3489660930,3489660931
PX icb_vtx_barrier 1 1 a=1,1,3489660930,3489660931
PX icb_vtx_barrier 2 1 a=2,1,3489660930,3489660931
PX icb_vtx_barrier 3 1 a=3,1,3489660930,3489660931
PX icb_vtx_barrier 0 2 a=0,2,3489660930,3489660931
PX icb_vtx_barrier 1 2 a=1,2,3489660930,3489660931
PX icb_vtx_barrier 2 2 a=2,2,3489660930,3489660931
PX icb_vtx_barrier 3 2 a=3,2,3489660930,3489660931
RESULT icb_vtx_barrier drawn=12
CTL icb_vtx_barrier d0000000,d0000001,d0000002,d0000003
PX enc_vtx_command_barrier 0 0 a=0,0,3489660930,3489660931
PX enc_vtx_command_barrier 1 0 a=1,0,3489660930,3489660931
PX enc_vtx_command_barrier 2 0 a=2,0,3489660930,3489660931
PX enc_vtx_command_barrier 3 0 a=3,0,3489660930,3489660931
PX enc_vtx_command_barrier 0 1 a=0,1,3489660930,3489660931
PX enc_vtx_command_barrier 1 1 a=1,1,3489660930,3489660931
PX enc_vtx_command_barrier 2 1 a=2,1,3489660930,3489660931
PX enc_vtx_command_barrier 3 1 a=3,1,3489660930,3489660931
PX enc_vtx_command_barrier 0 2 a=0,2,3489660930,3489660931
PX enc_vtx_command_barrier 1 2 a=1,2,3489660930,3489660931
PX enc_vtx_command_barrier 2 2 a=2,2,3489660930,3489660931
PX enc_vtx_command_barrier 3 2 a=3,2,3489660930,3489660931
RESULT enc_vtx_command_barrier drawn=12
CTL enc_vtx_command_barrier d0000000,d0000001,d0000002,d0000003
PX enc_vtx_encoder_barrier 0 0 a=0,0,3489660930,3489660931
PX enc_vtx_encoder_barrier 1 0 a=1,0,3489660930,3489660931
PX enc_vtx_encoder_barrier 2 0 a=2,0,3489660930,3489660931
PX enc_vtx_encoder_barrier 3 0 a=3,0,3489660930,3489660931
PX enc_vtx_encoder_barrier 0 1 a=0,1,3489660930,3489660931
PX enc_vtx_encoder_barrier 1 1 a=1,1,3489660930,3489660931
PX enc_vtx_encoder_barrier 2 1 a=2,1,3489660930,3489660931
PX enc_vtx_encoder_barrier 3 1 a=3,1,3489660930,3489660931
PX enc_vtx_encoder_barrier 0 2 a=0,2,3489660930,3489660931
PX enc_vtx_encoder_barrier 1 2 a=1,2,3489660930,3489660931
PX enc_vtx_encoder_barrier 2 2 a=2,2,3489660930,3489660931
PX enc_vtx_encoder_barrier 3 2 a=3,2,3489660930,3489660931
RESULT enc_vtx_encoder_barrier drawn=12
CTL enc_vtx_encoder_barrier d0000000,d0000001,d0000002,d0000003
PX enc_vtx_nobarrier 0 0 a=0,0,3489660930,3489660931
PX enc_vtx_nobarrier 1 0 a=1,0,3489660930,3489660931
PX enc_vtx_nobarrier 2 0 a=2,0,3489660930,3489660931
PX enc_vtx_nobarrier 3 0 a=3,0,3489660930,3489660931
PX enc_vtx_nobarrier 0 1 a=0,1,3489660930,3489660931
PX enc_vtx_nobarrier 1 1 a=1,1,3489660930,3489660931
PX enc_vtx_nobarrier 2 1 a=2,1,3489660930,3489660931
PX enc_vtx_nobarrier 3 1 a=3,1,3489660930,3489660931
PX enc_vtx_nobarrier 0 2 a=0,2,3489660930,3489660931
PX enc_vtx_nobarrier 1 2 a=1,2,3489660930,3489660931
PX enc_vtx_nobarrier 2 2 a=2,2,3489660930,3489660931
PX enc_vtx_nobarrier 3 2 a=3,2,3489660930,3489660931
RESULT enc_vtx_nobarrier drawn=12
CTL enc_vtx_nobarrier d0000000,d0000001,d0000002,d0000003
";

/// The pass's side in pixels.
const SIDE: u64 = 8;

/// The probe's lines of case `name`.
fn metals(name: &str) -> Vec<String> {
    ANSWERS
        .lines()
        .filter(|line| {
            let mut words = line.split(' ');
            words.next();
            words.next() == Some(name)
        })
        .map(str::to_owned)
        .collect()
}

/// A draw kind of the probe: a mesh draw of threadgroups or of threads, or a draw of primitives.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Groups,
    Threads,
    Draw,
}

impl Kind {
    const ALL: [Kind; 3] = [Kind::Groups, Kind::Threads, Kind::Draw];

    fn name(self) -> &'static str {
        match self {
            Kind::Groups => "groups",
            Kind::Threads => "threads",
            Kind::Draw => "draw",
        }
    }

    fn command_type(self) -> IndirectCommandType {
        match self {
            Kind::Groups => IndirectCommandType::DRAW_MESH_THREADGROUPS,
            Kind::Threads => IndirectCommandType::DRAW_MESH_THREADS,
            Kind::Draw => IndirectCommandType::DRAW,
        }
    }

    /// The fixture's kernel that encodes command 0's draw of this kind.
    fn kernel(self) -> &'static str {
        match self {
            Kind::Groups => "enc_groups",
            Kind::Threads => "enc_threads",
            Kind::Draw => "enc_draw",
        }
    }
}

/// Both mesh kinds.
const MESH_KINDS: IndirectCommandType =
    IndirectCommandType::DRAW_MESH_THREADGROUPS.union(IndirectCommandType::DRAW_MESH_THREADS);

/// The lane's device and the three fixtures' libraries.
struct Lane {
    device: Device,
    draws: Library,
    barrier: Library,
    commands: Library,
}

/// The probe's draws of `mesh_draws`' functions: the object pipeline's 3 threadgroups of 4
/// object threads and 2 mesh threads, and the grid pipeline's 12 by 4 threads in mesh
/// threadgroups of 4 by 2.
const OBJECT_DRAW: [[u64; 3]; 3] = [[3, 1, 1], [4, 1, 1], [2, 1, 1]];
const GRID_THREADS: [[u64; 3]; 3] = [[12, 4, 1], [1, 1, 1], [4, 2, 1]];
/// The CPU's grid draw of threadgroups: 3 by 2 of 4 mesh threads.
const GRID_GROUPS: [[u64; 3]; 3] = [[3, 2, 1], [1, 1, 1], [4, 1, 1]];

fn size(axes: [u64; 3]) -> Size {
    Size::new(axes[0], axes[1], axes[2])
}

/// `words` as a buffer of `device`'s, 16 bytes at least.
fn words(device: &Device, words: &[u32]) -> Buffer {
    let mut bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    bytes.resize(bytes.len().max(16), 0);
    buffer_with(device, &bytes).unwrap()
}

impl Lane {
    fn new(device: &Device) -> Lane {
        let library = |stem: &str| {
            let bitcode = llvm_assemble(&engine_dir().join(format!("{stem}.ll"))).unwrap();
            device.new_library_with_air(&bitcode).unwrap()
        };
        Lane {
            device: device.clone(),
            draws: library("mesh_draws"),
            barrier: library("mesh_barrier"),
            commands: library("mesh_commands"),
        }
    }

    /// A mesh pipeline of `library`'s `mesh` (and `object`, where given) with `fragment`, over
    /// `attachments` attachments, made with `supportIndirectCommandBuffers`.
    fn mesh_pipeline(
        &self,
        library: &Library,
        object: Option<&str>,
        mesh: &str,
        fragment: &str,
        attachments: usize,
    ) -> Result<RenderPipelineState, Error> {
        let function = |name| {
            StageSource::Function(
                library
                    .new_function_with_name(name)
                    .unwrap()
                    .function_descriptor(),
            )
        };
        let mut descriptor = MeshPipelineDescriptor::new(function(mesh));
        descriptor.object = object.map(function);
        descriptor.fragment = Some(function(fragment));
        for attachment in &mut descriptor.color_attachments[..attachments] {
            *attachment = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA32Uint));
        }
        descriptor.support_indirect_command_buffers = true;
        self.device
            .new_mesh_render_pipeline_state(&descriptor, &PipelineOptions::default())
    }

    /// `mesh_draws`' object pipeline, and its grid pipeline.
    fn draws_pipelines(&self) -> (RenderPipelineState, RenderPipelineState) {
        (
            self.mesh_pipeline(
                &self.draws,
                Some("mesh_draws_object"),
                "mesh_draws_mesh",
                "mesh_draws_fragment",
                2,
            )
            .unwrap(),
            self.mesh_pipeline(
                &self.draws,
                None,
                "mesh_draws_grid",
                "mesh_draws_fragment",
                2,
            )
            .unwrap(),
        )
    }

    /// The vertex pipeline of `vtx`, rasterization disabled, over the pass's two attachments.
    fn vertex_pipeline(&self) -> RenderPipelineState {
        let function = self.commands.new_function_with_name("vtx").unwrap();
        let mut descriptor =
            RenderPipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
        descriptor.rasterization_enabled = false;
        descriptor.input_primitive_topology = PrimitiveTopologyClass::Point;
        for attachment in &mut descriptor.color_attachments[..2] {
            *attachment = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA32Uint));
        }
        descriptor.support_indirect_command_buffers = true;
        self.device
            .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
            .unwrap()
    }

    /// A buffer of `count` commands of `types`, inheriting the pipeline state and the buffers
    /// where `inherit`, else binding one object and one mesh buffer, as the probe's.
    fn icb(&self, types: IndirectCommandType, inherit: bool, count: u64) -> IndirectCommandBuffer {
        let mut descriptor = IndirectCommandBufferDescriptor::new();
        descriptor.command_types = types;
        descriptor.inherit_pipeline_state = inherit;
        descriptor.inherit_buffers = inherit;
        if !inherit {
            descriptor.max_object_buffer_bind_count = 1;
            descriptor.max_mesh_buffer_bind_count = 1;
        }
        self.device
            .new_indirect_command_buffer(&descriptor, count, ResourceOptions::shared())
            .unwrap()
    }

    /// The state of `mesh_commands`' kernel `name`.
    fn kernel(&self, name: &str) -> ComputePipelineState {
        let function = self.commands.new_function_with_name(name).unwrap();
        let descriptor =
            ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
        self.device
            .new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
            .unwrap()
    }

    /// Kernel `name` encodes `icb` in `threads` threads, its argument buffer the buffer's ID then
    /// `states`' IDs, and `buffers` at indices 1 on; complete before it returns.
    fn produce(
        &self,
        icb: &IndirectCommandBuffer,
        name: &str,
        states: &[&RenderPipelineState],
        buffers: &[&Buffer],
        threads: u64,
    ) {
        let kernel = self.kernel(name);
        let mut ids = vec![icb.gpu_resource_id().0];
        ids.extend(states.iter().map(|state| state.gpu_resource_id().0));
        let arguments: Vec<u8> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
        let arguments = buffer_with(&self.device, &arguments).unwrap();
        let queue = self.device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(&kernel).unwrap();
        encoder.set_buffer(Some(&arguments), 0, 0).unwrap();
        for (index, buffer) in buffers.iter().enumerate() {
            encoder
                .set_buffer(Some(buffer), 0, index as u32 + 1)
                .unwrap();
        }
        encoder
            .use_resource(icb.as_resource(), ResourceUsage::WRITE)
            .unwrap();
        let grid = Size::new(threads, 1, 1);
        encoder.dispatch_threads(grid, grid).unwrap();
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None, "{name}");
    }

    /// Execute every command of `icb` in a pass of `attachments` attachments under `state`
    /// (none where `None`), with `marks` at vertex buffer 2 where given and `used` resident for
    /// the object, mesh and fragment stages: the command buffer, and the probe's lines of the
    /// pixels drawn, as case `name`.
    fn execute(
        &self,
        name: &str,
        icb: &IndirectCommandBuffer,
        attachments: usize,
        state: Option<&RenderPipelineState>,
        marks: Option<&Buffer>,
        used: &[&Buffer],
    ) -> (CommandBuffer, Vec<String>) {
        let setup = |encoder: &mut RenderCommandEncoder| {
            if let Some(state) = state {
                encoder.set_render_pipeline_state(state).unwrap();
            }
            if let Some(marks) = marks {
                encoder.set_vertex_buffer(Some(marks), 0, 2).unwrap();
            }
        };
        self.execute_after(name, icb, attachments, setup, used)
            .unwrap()
    }

    /// [`execute`](Self::execute), the encoder given to `setup` first (its states, buffers and
    /// draws before the execution), or the commit's refusal.
    fn execute_after(
        &self,
        name: &str,
        icb: &IndirectCommandBuffer,
        attachments: usize,
        setup: impl FnOnce(&mut RenderCommandEncoder),
        used: &[&Buffer],
    ) -> Result<(CommandBuffer, Vec<String>), Error> {
        let targets: Vec<_> = (0..attachments)
            .map(|_| texture(&self.device, PixelFormat::RGBA32Uint, SIDE))
            .collect();
        let mut pass = RenderPassDescriptor::new();
        for (attachment, target) in pass.color_attachments.iter_mut().zip(&targets) {
            attachment.texture = Some(target.clone());
            attachment.load_action = LoadAction::Clear;
            attachment.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
            attachment.store_action = StoreAction::Store;
        }
        let queue = self.device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        setup(&mut encoder);
        for buffer in used {
            encoder
                .use_resource(
                    buffer.as_resource(),
                    ResourceUsage::READ | ResourceUsage::WRITE,
                    RenderStages::OBJECT
                        .union(RenderStages::MESH)
                        .union(RenderStages::FRAGMENT),
                )
                .unwrap();
        }
        encoder
            .execute_commands_in_buffer(icb, 0..icb.size())
            .unwrap();
        encoder.end_encoding().unwrap();
        command_buffer.commit()?;
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None, "{name}");
        let planes: Vec<Vec<u32>> = targets
            .iter()
            .map(|t| {
                texels(t, 16)
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|w| u32::from_le_bytes(*w))
                    .collect()
            })
            .collect();
        let mut lines = Vec::new();
        for y in 0..SIDE as usize {
            for x in 0..SIDE as usize {
                let at = 4 * (y * SIDE as usize + x);
                if planes.iter().all(|p| p[at..at + 4].iter().all(|&w| w == 0)) {
                    continue;
                }
                let decimal: Vec<String> =
                    planes[0][at..at + 4].iter().map(u32::to_string).collect();
                let mut line = format!("PX {name} {x} {y} a={}", decimal.join(","));
                if let Some(b) = planes.get(1) {
                    let hex: Vec<String> = b[at..at + 4].iter().map(|w| format!("{w:x}")).collect();
                    line.push_str(&format!(" b={}", hex.join(",")));
                }
                lines.push(line);
            }
        }
        lines.push(format!("RESULT {name} drawn={}", lines.len()));
        Ok((command_buffer, lines))
    }
}

/// The words of `buffer`'s first `n`, joined by commas, in `hex` or decimal.
fn joined(buffer: &Buffer, n: u64, hex: bool) -> String {
    read_buffer(buffer, 0, 4 * n)
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| u32::from_le_bytes(*w))
        .map(|w| if hex { format!("{w:x}") } else { w.to_string() })
        .collect::<Vec<_>>()
        .join(",")
}

/// A fresh descriptor binds no object or mesh buffers; buffers of draws of primitives beside
/// mesh draws, of both mesh kinds, and of mesh draws beside dispatches (a buffer of compute
/// commands) are made, as plain Metal makes them.
#[test]
fn descriptors_and_buffers_answer_as_metals() {
    let descriptor = IndirectCommandBufferDescriptor::new();
    assert_eq!(descriptor.max_object_buffer_bind_count, 0);
    assert_eq!(descriptor.max_mesh_buffer_bind_count, 0);
    on_device(|device, _| {
        for types in [
            IndirectCommandType::DRAW.union(IndirectCommandType::DRAW_MESH_THREADGROUPS),
            MESH_KINDS,
            IndirectCommandType::DRAW_MESH_THREADGROUPS
                .union(IndirectCommandType::CONCURRENT_DISPATCH),
        ] {
            let mut descriptor = IndirectCommandBufferDescriptor::new();
            descriptor.command_types = types;
            descriptor.inherit_pipeline_state = true;
            descriptor.inherit_buffers = true;
            device
                .new_indirect_command_buffer(&descriptor, 1, ResourceOptions::shared())
                .unwrap_or_else(|e| panic!("{types:?}: {e}"));
        }
    });
}

/// Commands with pipeline states of their own and an object buffer draw what Metal's draw, set
/// by the CPU (`cpu_own`) and by a kernel (`kernel_own`, `enc_own`): command 0 the object
/// pipeline's mesh draw of threadgroups, command 1 the grid pipeline's of threads.
#[test]
fn own_commands_draw_metals_words() {
    on_device(|device, _| {
        let lane = Lane::new(device);
        let (object, grid) = lane.draws_pipelines();
        for kernel in [false, true] {
            let name = if kernel { "kernel_own" } else { "cpu_own" };
            let icb = lane.icb(MESH_KINDS, false, 2);
            let ctl = words(device, &[2, 1, 0, 0]);
            if kernel {
                lane.produce(&icb, "enc_own", &[&object, &grid], &[&ctl], 2);
            } else {
                encode_own(&icb, (&object, &grid), &ctl);
            }
            let (_, lines) = lane.execute(name, &icb, 2, None, None, &[&ctl]);
            assert_eq!(lines, metals(name), "on {}", device.name());
        }
    });
}

/// The CPU's `own` commands on `icb`: command 0 the object pipeline's mesh draw of threadgroups,
/// `ctl` its object buffer, command 1 the grid pipeline's of threads.
fn encode_own(
    icb: &IndirectCommandBuffer,
    (object, grid): (&RenderPipelineState, &RenderPipelineState),
    ctl: &Buffer,
) {
    let [groups, object_threads, mesh_threads] = OBJECT_DRAW;
    let first = icb.indirect_render_command_at(0).unwrap();
    first.set_render_pipeline_state(object).unwrap();
    first.set_object_buffer(Some(ctl), 0, 0).unwrap();
    first
        .draw_mesh_threadgroups(size(groups), size(object_threads), size(mesh_threads))
        .unwrap();
    let [threads, object_threads, mesh_threads] = GRID_THREADS;
    let second = icb.indirect_render_command_at(1).unwrap();
    second.set_render_pipeline_state(grid).unwrap();
    second
        .draw_mesh_threads(size(threads), size(object_threads), size(mesh_threads))
        .unwrap();
}

/// A buffer of mesh draws alone keeps no chunk, as no mesh draw is plain: the CPU's `own`
/// commands' first execution begins as many command buffers as their second, which records them
/// again, none for a chunk recorded apart only to be found not plain. An execution of another
/// buffer runs first, so neither pays for the device's first work.
#[test]
fn a_buffer_of_mesh_draws_alone_records_no_chunk_apart() {
    on_device(|device, _| {
        let lane = Lane::new(device);
        let (object, grid) = lane.draws_pipelines();
        let ctl = words(device, &[2, 1, 0, 0]);
        let begun = || device.work_counts().command_buffers_begun;
        let warm = lane.icb(MESH_KINDS, false, 2);
        encode_own(&warm, (&object, &grid), &ctl);
        lane.execute("cpu_own", &warm, 2, None, None, &[&ctl]);
        let icb = lane.icb(MESH_KINDS, false, 2);
        encode_own(&icb, (&object, &grid), &ctl);
        let mut costs = Vec::new();
        for _ in 0..2 {
            let before = begun();
            let (_, lines) = lane.execute("cpu_own", &icb, 2, None, None, &[&ctl]);
            costs.push(begun() - before);
            assert_eq!(lines, metals("cpu_own"), "on {}", device.name());
        }
        assert_eq!(
            costs[0], costs[1],
            "command buffers begun by the first and second executions"
        );
    });
}

/// A kernel's draw of each kind on a buffer of each kind (`kernel_<kind>_on_<kind>`), the
/// buffer inheriting the encoder's state (the grid pipeline, or for draws of primitives the
/// vertex pipeline) and its vertex buffer 2, the marks: a mesh draw on a buffer of either mesh
/// kind draws Metal's words; on a buffer of draws of primitives the runtime runs no command and
/// reports the encoding fault, where Metal's GPU faulted (`status=5`); a draw of primitives on a
/// buffer of mesh draws draws nothing, and on its own kind's marks its 4 vertices.
#[test]
fn a_kernel_s_draws_of_each_kind_run_as_metals() {
    on_device(|device, _| {
        let lane = Lane::new(device);
        let (_, grid) = lane.draws_pipelines();
        let vertex = lane.vertex_pipeline();
        // `vtx` stores its marks: the state states that its vertex function writes memory, and
        // no point size.
        assert!(vertex.vertex_writes_memory() && !vertex.writes_point_size());
        for kind in Kind::ALL {
            for on in Kind::ALL {
                let name = format!("kernel_{}_on_{}", kind.name(), on.name());
                let icb = lane.icb(on.command_type(), true, 1);
                lane.produce(&icb, kind.kernel(), &[], &[], 1);
                let marks = words(device, &[0; 16]);
                let state = if kind == Kind::Draw { &vertex } else { &grid };
                let (command_buffer, mut lines) =
                    lane.execute(&name, &icb, 2, Some(state), Some(&marks), &[]);
                lines.push(format!("case {name} marks={}", joined(&marks, 4, true)));
                let report = command_buffer.execution_report().unwrap();
                let mut want = metals(&name);
                if kind != Kind::Draw && on == Kind::Draw {
                    // Metal's GPU faulted the command buffer; the runtime runs no command and
                    // reports the encoding fault.
                    assert_eq!(want.remove(0), format!("case {name} status=5"));
                    want.insert(0, format!("RESULT {name} drawn=0"));
                    assert!(report.replay_fault, "{name}");
                    assert_eq!(report.replay_faults, ReplayFaults::ENCODING, "{name}");
                } else {
                    assert!(!report.replay_fault, "{name}");
                }
                assert_eq!(lines, want, "on {}", device.name());
            }
        }
    });
}

/// The CPU's mesh draw of the other mesh kind draws as plain Metal draws it, where Metal's
/// validation stops it (`cpu_threads_on_groups`, `cpu_groups_on_threads`); a mesh draw on a
/// buffer of draws of primitives, which ends Metal's process, is refused.
#[test]
fn the_cpu_s_mesh_draws_of_the_other_kind_draw_as_plain_metals() {
    on_device(|device, _| {
        let lane = Lane::new(device);
        let (_, grid) = lane.draws_pipelines();
        for (kind, on) in [(Kind::Threads, Kind::Groups), (Kind::Groups, Kind::Threads)] {
            let name = format!("cpu_{}_on_{}", kind.name(), on.name());
            let icb = lane.icb(on.command_type(), true, 1);
            let command = icb.indirect_render_command_at(0).unwrap();
            match kind {
                Kind::Groups => {
                    let [groups, object, mesh] = GRID_GROUPS;
                    command.draw_mesh_threadgroups(size(groups), size(object), size(mesh))
                }
                _ => {
                    let [threads, object, mesh] = GRID_THREADS;
                    command.draw_mesh_threads(size(threads), size(object), size(mesh))
                }
            }
            .unwrap();
            let (_, lines) = lane.execute(&name, &icb, 2, Some(&grid), None, &[]);
            assert_eq!(lines, metals(&name), "on {}", device.name());
        }
        let icb = lane.icb(IndirectCommandType::DRAW, true, 1);
        let [groups, object, mesh] = GRID_GROUPS;
        assert_eq!(
            icb.indirect_render_command_at(0)
                .unwrap()
                .draw_mesh_threadgroups(size(groups), size(object), size(mesh)),
            Err(Error::Invalid(Invalid::new(
                "draw_mesh_threadgroups",
                Problem::IndirectCommandBuffer(IndirectCommandBufferProblem::CommandType(
                    IndirectCommandType::DRAW_MESH_THREADGROUPS
                ))
            )))
        );
    });
}

/// A command's barrier, set by the CPU or a kernel (`enc_barrier`), orders the reader's draw
/// after the writer's: the reader's object function reads the control words the writer's mesh
/// function stored, and launches their 3 by 2 mesh threadgroups. Without one the draws are not
/// ordered: Metal's reader read the words the CPU wrote, and the lane takes either order's.
#[test]
fn a_command_s_barrier_orders_its_mesh_draw() {
    on_device(|device, _| {
        let lane = Lane::new(device);
        let writer = lane
            .mesh_pipeline(
                &lane.barrier,
                None,
                "mesh_barrier_writer",
                "mesh_barrier_fragment",
                1,
            )
            .unwrap();
        let reader = lane
            .mesh_pipeline(
                &lane.barrier,
                Some("mesh_barrier_object"),
                "mesh_barrier_mesh",
                "mesh_barrier_fragment",
                1,
            )
            .unwrap();
        for kernel in [false, true] {
            for barrier in [true, false] {
                let name = format!(
                    "{}_{}",
                    if kernel { "kernel" } else { "cpu" },
                    if barrier { "barrier" } else { "nobarrier" }
                );
                let icb = lane.icb(IndirectCommandType::DRAW_MESH_THREADGROUPS, false, 2);
                let ctl = words(device, &[1, 1, 1, 0]);
                let one = Size::new(1, 1, 1);
                if kernel {
                    let flag = words(device, &[u32::from(barrier)]);
                    lane.produce(&icb, "enc_barrier", &[&writer, &reader], &[&ctl, &flag], 1);
                } else {
                    let first = icb.indirect_render_command_at(0).unwrap();
                    first.set_render_pipeline_state(&writer).unwrap();
                    first.set_mesh_buffer(Some(&ctl), 0, 0).unwrap();
                    first.draw_mesh_threadgroups(one, one, one).unwrap();
                    let second = icb.indirect_render_command_at(1).unwrap();
                    if barrier {
                        second.set_barrier().unwrap();
                    }
                    second.set_render_pipeline_state(&reader).unwrap();
                    second.set_object_buffer(Some(&ctl), 0, 0).unwrap();
                    second.draw_mesh_threadgroups(one, one, one).unwrap();
                }
                let (_, mut lines) = lane.execute(&name, &icb, 1, None, None, &[&ctl]);
                lines.push(format!("CTL {name} {}", joined(&ctl, 4, false)));
                let metals = metals(&name);
                if barrier {
                    assert_eq!(lines, metals, "on {}", device.name());
                    continue;
                }
                // The ordered draws' words: the barrier case's, as this case.
                let ordered: Vec<String> = self::metals(&name.replace("nobarrier", "barrier"))
                    .into_iter()
                    .map(|line| line.replacen("_barrier", "_nobarrier", 1))
                    .collect();
                let order = if lines == metals {
                    "Metal's order"
                } else if lines == ordered {
                    "the draws' order"
                } else {
                    panic!(
                        "{name} on {}: {lines:#?}, neither Metal's {metals:#?} nor the ordered \
                         {ordered:#?}",
                        device.name()
                    )
                };
                eprintln!(
                    "{name} on {}: unordered draws read in {order}",
                    device.name()
                );
            }
        }
    });
}

/// The writer of a case of [`mesh_commands_read_earlier_draws_writes_as_metal_orders_them`].
#[derive(Clone, Copy, PartialEq, Eq)]
enum Writer {
    /// `mesh_barrier`'s fragment writer: `(3, 2, 1, 9)`, covering pixel (7, 7) with `0xBBBB`.
    Fragment,
    /// `mesh_barrier`'s mesh-stage writer: `(3, 2, 1, 7)`, covering pixel (7, 7) with `0xAAAA`.
    Mesh,
    /// `mesh_commands`' `vtx` of 4 points: `0xd0000000 | vid` at word `vid`.
    Vertex,
}

impl Writer {
    /// The lines of case `name` whose reader read the writer's words (`writers`) or the CPU's,
    /// `(1, 1, 1, 0)`: the reader's pixels and the writer's, and the control words after the
    /// pass (hexadecimal for the vertex writer's, as our probe printed them).
    fn lines(self, name: &str, writers: bool) -> Vec<String> {
        let (z, w, cover, ctl): (u32, u32, Option<u32>, &str) = match self {
            Writer::Fragment => (1, 9, Some(0xBBBB), "3,2,1,9"),
            Writer::Mesh => (1, 7, Some(0xAAAA), "3,2,1,7"),
            Writer::Vertex => (
                0xd000_0002,
                0xd000_0003,
                None,
                "d0000000,d0000001,d0000002,d0000003",
            ),
        };
        // The reader's mesh threadgroups that draw: those below the words' first two, of its 4
        // by 3 grid (one threadgroup for the object reader, which launches the grid itself).
        let (columns, rows) = match (self, writers) {
            (_, false) => (1, 1),
            (Writer::Vertex, true) => (4, 3),
            (_, true) => (3, 2),
        };
        let (z, w) = if writers { (z, w) } else { (1, 0) };
        let mut lines: Vec<String> = (0..rows)
            .flat_map(|y| (0..columns).map(move |x| format!("PX {name} {x} {y} a={x},{y},{z},{w}")))
            .collect();
        lines.extend(cover.map(|cover| format!("PX {name} 7 7 a={cover},0,0,0")));
        lines.push(format!("RESULT {name} drawn={}", lines.len()));
        lines.push(format!("CTL {name} {ctl}"));
        lines
    }
}

/// Mesh commands that read what an earlier draw of their pass wrote, as Metal orders them (an M4):
/// `mesh_barrier`'s fragment writer, then its object or grid reader, as commands 0 and 1 of one
/// buffer, with and without the reader's barrier, and the encoder's own fragment writer before an
/// execution of the object reader, without a barrier and behind one from the fragment stage to the
/// object stage; the encoder's mesh-stage writer behind one from the mesh stage; and
/// `mesh_commands`' `vtx` writing from its vertex function before the
/// grid reader, as command 0 of a buffer of draws and mesh draws with and without the reader's
/// barrier, and as the encoder's draw before an execution of the reader behind the command's
/// barrier, the encoder's, or neither.
///
/// Metal's reader reads none of a fragment's writes, behind any barrier ([`ANSWERS`]; the M3 Max
/// alike): on a device that emulates mesh draws the reader's object and mesh
/// functions run before the pass, reading the CPU's words as Metal's do; on a device with
/// `VK_EXT_mesh_shader` they read a copy of the words taken before the pass
/// (`render/pass_start_reads.rs`), the CPU's words too. Metal's reader reads a vertex function's
/// words behind the command's barrier or the encoder's, and after the encoder's draw without
/// either, but not after an earlier command's without the barrier. The emulating device runs the
/// reader's work after the vertex function behind the command's barrier, and refuses the commit
/// `NotImplemented(IndirectRenderCommands)` without it; a native device reads the writer's words
/// behind either barrier and either words without one. Behind the mesh stage's barrier every
/// device reads the mesh function's words, as Metal's.
#[test]
fn mesh_commands_read_earlier_draws_writes_as_metal_orders_them() {
    on_device(|device, _| {
        let lane = Lane::new(device);
        let pipeline = |object, mesh, fragment| {
            lane.mesh_pipeline(&lane.barrier, object, mesh, fragment, 1)
                .unwrap()
        };
        let fragment_writer = pipeline(None, "mesh_barrier_cover", "mesh_barrier_fragment_writer");
        let mesh_writer = pipeline(None, "mesh_barrier_writer", "mesh_barrier_fragment");
        let object_reader = pipeline(
            Some("mesh_barrier_object"),
            "mesh_barrier_mesh",
            "mesh_barrier_fragment",
        );
        let grid_reader = pipeline(None, "mesh_barrier_grid", "mesh_barrier_fragment");
        let vertex_writer = {
            let function = lane.commands.new_function_with_name("vtx").unwrap();
            let mut descriptor = RenderPipelineDescriptor::new(StageSource::Function(
                function.function_descriptor(),
            ));
            descriptor.rasterization_enabled = false;
            descriptor.input_primitive_topology = PrimitiveTopologyClass::Point;
            descriptor.color_attachments[0] =
                Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA32Uint));
            descriptor.support_indirect_command_buffers = true;
            device
                .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
                .unwrap()
        };
        let one = Size::new(1, 1, 1);
        // A buffer of `count` draws and mesh draws with their own states and buffers.
        let icb = |count| {
            let mut descriptor = IndirectCommandBufferDescriptor::new();
            descriptor.command_types =
                IndirectCommandType::DRAW.union(IndirectCommandType::DRAW_MESH_THREADGROUPS);
            descriptor.inherit_pipeline_state = false;
            descriptor.inherit_buffers = false;
            descriptor.max_vertex_buffer_bind_count = 3;
            descriptor.max_fragment_buffer_bind_count = 1;
            descriptor.max_object_buffer_bind_count = 1;
            descriptor.max_mesh_buffer_bind_count = 1;
            device
                .new_indirect_command_buffer(&descriptor, count, ResourceOptions::shared())
                .unwrap()
        };
        // Command `at` of `icb`: the object reader of `ctl`, or the grid reader of 4 by 3
        // threadgroups where `grid`, behind its barrier where `barrier`.
        let read = |icb: &IndirectCommandBuffer, at, grid: bool, barrier: bool, ctl: &Buffer| {
            let command = icb.indirect_render_command_at(at).unwrap();
            if barrier {
                command.set_barrier().unwrap();
            }
            if grid {
                command.set_render_pipeline_state(&grid_reader).unwrap();
                command.set_mesh_buffer(Some(ctl), 0, 0).unwrap();
                command
                    .draw_mesh_threadgroups(Size::new(4, 3, 1), one, one)
                    .unwrap();
            } else {
                command.set_render_pipeline_state(&object_reader).unwrap();
                command.set_object_buffer(Some(ctl), 0, 0).unwrap();
                command.draw_mesh_threadgroups(one, one, one).unwrap();
            }
        };
        // The writer's draw by the encoder, behind the encoder's barrier from `after` to the
        // reader's stage where given.
        let own = |encoder: &mut RenderCommandEncoder, writer, ctl: &Buffer, after| {
            match writer {
                Writer::Fragment => {
                    encoder.set_render_pipeline_state(&fragment_writer).unwrap();
                    encoder.set_fragment_buffer(Some(ctl), 0, 0).unwrap();
                    encoder.draw_mesh_threadgroups(one, one, one).unwrap();
                }
                Writer::Mesh => {
                    encoder.set_render_pipeline_state(&mesh_writer).unwrap();
                    encoder.set_mesh_buffer(Some(ctl), 0, 0).unwrap();
                    encoder.draw_mesh_threadgroups(one, one, one).unwrap();
                }
                Writer::Vertex => {
                    encoder.set_render_pipeline_state(&vertex_writer).unwrap();
                    encoder.set_vertex_buffer(Some(ctl), 0, 2).unwrap();
                    encoder
                        .draw_primitives(PrimitiveType::Point, 0, 4, 1, 0)
                        .unwrap();
                }
            }
            let before = match writer {
                Writer::Vertex => RenderStages::MESH,
                Writer::Fragment | Writer::Mesh => RenderStages::OBJECT,
            };
            if let Some(after) = after {
                encoder.memory_barrier_with_scope(BarrierScope::BUFFERS, after, before);
            }
        };
        let emulated = emulates(device);
        let device_name = device.name();
        // Each case: its name, its writer, whether the command's barrier and the encoder's stand,
        // and its lines, or the commit's refusal.
        let mut cases = Vec::new();
        // Case `name`: the reader in `icb`, after the encoder's own draw of `writer` where `own`,
        // behind the encoder's barrier from `after` where given.
        let mut run = |name: String,
                       (writer, command): (Writer, bool),
                       icb: &IndirectCommandBuffer,
                       (own_draw, after): (bool, Option<RenderStages>),
                       ctl: &Buffer| {
            let setup = |encoder: &mut RenderCommandEncoder| {
                if own_draw {
                    own(encoder, writer, ctl, after);
                }
            };
            let lines = lane
                .execute_after(&name, icb, 1, setup, &[ctl])
                .map(|(_, mut lines)| {
                    let hex = writer == Writer::Vertex;
                    lines.push(format!("CTL {name} {}", joined(ctl, 4, hex)));
                    lines
                });
            cases.push((name, writer, command, after.is_some(), lines));
        };
        // The writer as command 0, the reader as command 1.
        for (name, writer, grid) in [
            ("icb_frag_object", Writer::Fragment, false),
            ("icb_frag_grid", Writer::Fragment, true),
            ("icb_vtx", Writer::Vertex, true),
        ] {
            for barrier in [false, true] {
                let ctl = words(device, &[1, 1, 1, 0]);
                let icb = icb(2);
                let first = icb.indirect_render_command_at(0).unwrap();
                if writer == Writer::Vertex {
                    first.set_render_pipeline_state(&vertex_writer).unwrap();
                    first.set_vertex_buffer(Some(&ctl), 0, 2).unwrap();
                    first
                        .draw_primitives(PrimitiveType::Point, 0, 4, 1, 0)
                        .unwrap();
                } else {
                    first.set_render_pipeline_state(&fragment_writer).unwrap();
                    first.set_fragment_buffer(Some(&ctl), 0, 0).unwrap();
                    first.draw_mesh_threadgroups(one, one, one).unwrap();
                }
                read(&icb, 1, grid, barrier, &ctl);
                let ordered = if barrier { "barrier" } else { "nobarrier" };
                let name = format!("{name}_{ordered}");
                run(name, (writer, barrier), &icb, (false, None), &ctl);
            }
        }
        // The encoder's writer, then an execution of the reader, behind the command's barrier or
        // the encoder's from a stage.
        for (name, writer, command, after) in [
            ("enc_frag_nobarrier", Writer::Fragment, false, None),
            (
                "enc_frag_barrier",
                Writer::Fragment,
                false,
                Some(RenderStages::FRAGMENT),
            ),
            (
                "enc_mesh_barrier",
                Writer::Mesh,
                false,
                Some(RenderStages::MESH),
            ),
            ("enc_vtx_command_barrier", Writer::Vertex, true, None),
            (
                "enc_vtx_encoder_barrier",
                Writer::Vertex,
                false,
                Some(RenderStages::VERTEX),
            ),
            ("enc_vtx_nobarrier", Writer::Vertex, false, None),
        ] {
            let ctl = words(device, &[1, 1, 1, 0]);
            let icb = icb(1);
            read(&icb, 0, writer == Writer::Vertex, command, &ctl);
            run(
                name.to_owned(),
                (writer, command),
                &icb,
                (true, after),
                &ctl,
            );
        }
        for (name, writer, command, encoder, lines) in cases {
            let writers = writer.lines(&name, true);
            let cpus = writer.lines(&name, false);
            let metal = metals(&name);
            let metals = if metal == writers {
                "the writer's"
            } else if metal == cpus {
                "the CPU's"
            } else {
                panic!("{name}: Metal's lines {metal:#?} are neither order's")
            };
            // An emulating device runs a reader of a vertex function's words behind the
            // command's barrier alone.
            let refused = emulated && writer == Writer::Vertex && !command;
            let lines = match lines {
                Err(Error::NotImplemented(RuntimeGap::IndirectRenderCommands)) if refused => {
                    eprintln!("{name} on {device_name}: refused, emulated; Metal's read {metals}");
                    continue;
                }
                Err(error) => panic!("{name} on {device_name}: {error}"),
                Ok(_) if refused => panic!("{name} on {device_name}: taken, where refused"),
                Ok(lines) => lines,
            };
            let expected = match writer {
                // Metal's order: the emulating device's work runs before the pass, and the
                // native one's reads a copy taken before it.
                Writer::Fragment => Some(&cpus),
                // Each barrier honoured by a native device, whose draws split there.
                Writer::Vertex if command || encoder => Some(&writers),
                Writer::Mesh => Some(&writers),
                // Unordered.
                Writer::Vertex => None,
            };
            match expected {
                Some(expected) => assert_eq!(&lines, expected, "{name} on {device_name}"),
                None => assert!(
                    lines == writers || lines == cpus,
                    "{name} on {device_name}: {lines:#?}, neither the writer's words \
                     {writers:#?} nor the CPU's {cpus:#?}"
                ),
            }
            let read = if lines == writers {
                "the writer's"
            } else {
                "the CPU's"
            };
            eprintln!("{name} on {device_name}: read {read} words; Metal's read {metals}");
        }
    });
}

/// A kernel's reset of command 1 (`enc_reset`) leaves it drawing nothing: only command 0 draws.
#[test]
fn a_reset_command_draws_nothing() {
    on_device(|device, _| {
        let lane = Lane::new(device);
        let (_, grid) = lane.draws_pipelines();
        let icb = lane.icb(MESH_KINDS, true, 2);
        lane.produce(&icb, "enc_reset", &[], &[], 1);
        let (_, lines) = lane.execute("kernel_reset", &icb, 2, Some(&grid), None, &[]);
        assert_eq!(lines, metals("kernel_reset"), "on {}", device.name());
    });
}

/// A pipeline state set on a command of a buffer whose command types hold no draw of its kind is
/// refused `PipelineKind`, and the CPU's draw of primitives on a buffer of mesh draws alone
/// `CommandType`, where Metal's process ends (the module docs): the vertex pipeline on a buffer
/// of either mesh kind or both, the grid pipeline on a buffer of draws of primitives, of indexed
/// draws or of patch draws. A buffer of draws of primitives beside mesh draws takes both states.
#[test]
fn a_pipeline_on_a_buffer_of_no_draw_of_its_kind_is_refused() {
    on_device(|device, _| {
        let lane = Lane::new(device);
        let (_, grid) = lane.draws_pipelines();
        let vertex = lane.vertex_pipeline();
        let refused = |call, problem| {
            Err(Error::Invalid(Invalid::new(
                call,
                Problem::IndirectCommandBuffer(problem),
            )))
        };
        let groups = IndirectCommandType::DRAW_MESH_THREADGROUPS;
        let threads = IndirectCommandType::DRAW_MESH_THREADS;
        for (state, types) in [
            (&vertex, groups),
            (&vertex, threads),
            (&vertex, MESH_KINDS),
            (&grid, IndirectCommandType::DRAW),
            (&grid, IndirectCommandType::DRAW_INDEXED),
            (&grid, IndirectCommandType::DRAW_PATCHES),
        ] {
            let icb = lane.icb(types, false, 1);
            let at = icb.indirect_render_command_at(0).unwrap();
            assert_eq!(
                at.set_render_pipeline_state(state),
                refused(
                    "set_render_pipeline_state",
                    IndirectCommandBufferProblem::PipelineKind
                ),
                "{types:?} on {}",
                device.name()
            );
            if types.difference(MESH_KINDS).is_empty() {
                assert_eq!(
                    at.draw_primitives(PrimitiveType::Point, 0, 1, 1, 0),
                    refused(
                        "draw_primitives",
                        IndirectCommandBufferProblem::CommandType(IndirectCommandType::DRAW)
                    ),
                    "a draw of primitives on {types:?} on {}",
                    device.name()
                );
            }
        }
        let icb = lane.icb(IndirectCommandType::DRAW.union(groups), false, 1);
        let at = icb.indirect_render_command_at(0).unwrap();
        for state in [&vertex, &grid] {
            assert_eq!(
                at.set_render_pipeline_state(state),
                Ok(()),
                "on {}",
                device.name()
            );
        }
    });
}

/// An executed command's mesh draw runs its object function with the threadgroup memory lengths
/// the command set, never the encoder's, as Metal does (the M3 Max under macOS 27.0.1: an
/// encoder's own draw and each case below, `icb_object_threadgroup`'s function): argument 0 of
/// 256 bytes keeps `a`'s 64 words off `b`, so b[0] and a[63] read 0xb and 0xa; one of 16 bytes
/// lets them reach `b`, and a[63], past both, reads 0. From the CPU's records at the commit, and
/// from the records a host step reads where an encoder made the buffer resident for writing.
///
/// Each object threadgroup is one thread: the function's barriers lie where its thread 0 alone
/// reaches them, which both Metal's and Vulkan's rules allow only where every thread of the
/// threadgroup reaches them (NVIDIA's GPU waited at them for good with 32 threads). The M3 Max
/// answers each case alike with 1 thread and with 32.
#[test]
fn a_command_s_mesh_draw_runs_with_its_own_object_threadgroup_lengths() {
    on_device(|device, _| {
        let lane = Lane::new(device);
        let bitcode = llvm_assemble(&engine_dir().join("icb_object_threadgroup.ll")).unwrap();
        let library = device.new_library_with_air(&bitcode).unwrap();
        let state = lane
            .mesh_pipeline(
                &library,
                Some("icb_object_threadgroup_object"),
                "icb_object_threadgroup_mesh",
                "icb_object_threadgroup_fragment",
                1,
            )
            .unwrap();
        // One object thread (the doc), 32 mesh threads.
        let (one, threads) = (Size::new(1, 1, 1), Size::new(32, 1, 1));
        // The encoder's lengths, the command's, and Metal's b[0] and a[63].
        let cases: [([u32; 2], [u32; 2], [u32; 2]); 2] = [
            ([16, 16], [256, 16], [0xb, 0xa]),
            ([256, 16], [16, 16], [0xa, 0]),
        ];
        let set_encoder = |encoder: &mut RenderCommandEncoder, lengths: [u32; 2]| {
            for (index, length) in (0u32..).zip(lengths) {
                encoder
                    .set_object_threadgroup_memory_length(length, index)
                    .unwrap();
            }
        };
        let read = |out: &Buffer| {
            let bytes = read_buffer(out, 0, 8).unwrap();
            let words = bytes.as_chunks::<4>().0;
            [0, 1].map(|at| u32::from_le_bytes(words[at]))
        };
        for written in [false, true] {
            for (encoder_lengths, command_lengths, metal) in cases {
                let out = words(device, &[0; 4]);
                let mut descriptor = IndirectCommandBufferDescriptor::new();
                descriptor.command_types = IndirectCommandType::DRAW_MESH_THREADGROUPS;
                descriptor.inherit_pipeline_state = false;
                descriptor.inherit_buffers = false;
                descriptor.max_object_buffer_bind_count = 1;
                descriptor.max_mesh_buffer_bind_count = 1;
                descriptor.max_object_threadgroup_memory_bind_count = 2;
                let icb = device
                    .new_indirect_command_buffer(&descriptor, 1, ResourceOptions::shared())
                    .unwrap();
                let command = icb.indirect_render_command_at(0).unwrap();
                command.set_render_pipeline_state(&state).unwrap();
                command.set_object_buffer(Some(&out), 0, 0).unwrap();
                for (index, length) in (0u32..).zip(command_lengths) {
                    command
                        .set_object_threadgroup_memory_length(length, index)
                        .unwrap();
                }
                command.draw_mesh_threadgroups(one, one, threads).unwrap();
                if written {
                    // An earlier encoder makes the buffer resident for writing: its execution's
                    // commands are then a host step's.
                    let queue = device.new_command_queue().unwrap();
                    let command_buffer = queue.command_buffer().unwrap();
                    let mut encoder = command_buffer.compute_command_encoder().unwrap();
                    encoder
                        .use_resource(icb.as_resource(), ResourceUsage::WRITE)
                        .unwrap();
                    encoder.end_encoding().unwrap();
                    command_buffer.commit().unwrap();
                    command_buffer.wait_until_completed_for(WAIT).unwrap();
                }
                let name = format!("{encoder_lengths:?} {command_lengths:?} written {written}");
                let setup =
                    |encoder: &mut RenderCommandEncoder| set_encoder(encoder, encoder_lengths);
                lane.execute_after(&name, &icb, 1, setup, &[&out]).unwrap();
                assert_eq!(read(&out), metal, "{name} on {}", device.name());
            }
        }
        // The encoder's own draw, under each case's encoder lengths, as Metal's.
        for (encoder_lengths, _, _) in &cases {
            let out = words(device, &[0; 4]);
            let icb = lane.icb(IndirectCommandType::DRAW_MESH_THREADGROUPS, true, 1);
            let name = format!("encoder {encoder_lengths:?}");
            let setup = |encoder: &mut RenderCommandEncoder| {
                set_encoder(encoder, *encoder_lengths);
                encoder.set_render_pipeline_state(&state).unwrap();
                encoder.set_object_buffer(Some(&out), 0, 0).unwrap();
                encoder.draw_mesh_threadgroups(one, one, threads).unwrap();
            };
            lane.execute_after(&name, &icb, 1, setup, &[&out]).unwrap();
            let metal = match encoder_lengths[0] {
                256 => [0xb, 0xa],
                _ => [0xa, 0],
            };
            assert_eq!(read(&out), metal, "{name} on {}", device.name());
        }
    });
}
