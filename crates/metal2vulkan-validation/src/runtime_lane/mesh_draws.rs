//! Mesh draws on the device the cfg selects, under the validation layer: the Metal probe's grid and
//! object cases drawn through `metal2vulkan-runtime` with the probe's functions (the owned fixture
//! `mesh_draws`). Each case is one draw into an 8x8 pass of two `RGBA32Uint` attachments, cleared
//! to zero and stored; each mesh threadgroup draws one triangle over the centre of one pixel, its
//! per-primitive data the builtins it read, which the fragment function writes. Without an object
//! function (`mesh_draws_grid`) the pixel is its threadgroup's position, and the data its position
//! and the grid's threadgroups in the first attachment; the grid's threads, its threadgroup's size
//! packed 10 bits an axis, and its thread 0's position in the grid, `x | y << 16`, in the second.
//! Under the object function (`mesh_draws_object`, whose thread 0 states its own builtins in the
//! payload and launches `ctl.x + group.x` by `ctl.y` mesh threadgroups, `ctl` its buffer 0) the
//! pixel is the mesh threadgroup's x and the object threadgroup's linear index times 2 plus its
//! y, and the data the payload's object words, then the mesh threadgroup's position, grid and
//! size.
//!
//! Every answer is Metal's, which the M3 Max (macOS 26.7) and an M4 (macOS 27.0.1) measured
//! alike, with and without API validation: [`GRID_ANSWERS`] and [`OBJECT_ANSWERS`] hold the probe's
//! lines, which the drawn pixels are printed as and compared with. Among them: `drawMeshThreads`
//! without an object function launches the whole threadgroups its threads fill, whose threads are
//! the grid the builtins report (10 by 3 threads in threadgroups of 4 by 2: 2 by 1 threadgroups of
//! an 8 by 2 grid); a grid with a zero axis draws nothing, and so do threads that fill no
//! threadgroup; an object threadgroup that sets no grid launches nothing.
//!
//! On a device with `VK_EXT_mesh_shader` an indirect draw reads the counts its arguments pass
//! writes, which must draw Metal's words too; a grid Metal draws nothing for (an axis past 65,535,
//! or more than 1,048,575 threadgroups: [`PAST_CASES`], measured on the M3 Max under macOS 27.0.1,
//! direct and indirect alike) draws nothing, and an object threadgroup that sets one launches
//! nothing ([`OBJECT_PAST_CASES`]). Object grids of any count run as Metal runs them (the owned
//! fixture `mesh_object_grid` and Metal probe 5, [`object_grids_run_as_metal_runs_them`]): in
//! pieces within the task stage's limits, threads per grid wrapped to 32 bits as Metal's, and
//! one of more than 2^29 threadgroups, which Metal's GPU discards (probe 6), failing the command
//! buffer with Metal's error (`CommandBufferError::GpuRecoveryVictim`), a direct draw's from its
//! encoder on.
//! A `drawMeshThreads` that leaves a partial object threadgroup is refused
//! (`NotImplemented(MeshShading)`); Metal's answers for it are printed (a partial object
//! threadgroup runs the threads left). What Metal's validation
//! stops draws as plain Metal does: a threadgroup of more than 1024 threads and a misaligned
//! argument offset draw nothing, and a draw of the other kind, which stops plain Metal too, is
//! refused typed.
//!
//! A device without `VK_EXT_mesh_shader` draws every case through the emulation, and the words must
//! be Metal's there too: the object function's compute dispatch writes each object threadgroup's
//! grid and payload, a scan places their mesh threadgroups (or an indirect draw's, from Metal's
//! arguments), and the mesh function's compute dispatch writes records that a synthesized vertex
//! stage draws. There indirect draws draw too: an object function's through an object arguments
//! pass, which reads Metal's arguments, reserves the object threadgroups' slots in the records
//! pool and writes the object dispatch's grid. A partial object threadgroup is refused
//! (`NotImplemented(MeshShading)`), as on a device with the extension.

use super::degrade_notices::{assert_made_once, made_mesh_kinds};
use super::gpu_times::read_kinds;
use super::render_state::{texels, texture};
use super::shared_kinds;
use super::{buffer_with, read_buffer, validated_device, Messages, ScratchCache, WAIT};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::device::{VkExtension, VkFeature};
use metal2vulkan::error::{BindError, Feature, Needs};
use metal2vulkan::metal::{PixelFormat, PrimitiveType};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, MeshPipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    Buffer, ClearColor, CommandBufferError, CommandQueue, Degradation, Device, DrawProblem, Error,
    Invalid, Library, LoadAction, PipelineOptions, Problem, RenderCommandEncoder,
    RenderPassDescriptor, RenderPipelineState, Size, StoreAction,
};

/// the Metal probe's group `g3` (`mesh_draws_grid` without an object function), as it printed it on
/// both Macs: `PX <case> <x> <y> a=<words> b=<words in hex>` for each pixel a case drew, then
/// `RESULT <case> drawn=<pixels>`.
const GRID_ANSWERS: &str = "\
PX grid-groups-3x2-by-4 0 0 a=0,0,3,2 b=c,2,100404,0
PX grid-groups-3x2-by-4 1 0 a=1,0,3,2 b=c,2,100404,4
PX grid-groups-3x2-by-4 2 0 a=2,0,3,2 b=c,2,100404,8
PX grid-groups-3x2-by-4 0 1 a=0,1,3,2 b=c,2,100404,10000
PX grid-groups-3x2-by-4 1 1 a=1,1,3,2 b=c,2,100404,10004
PX grid-groups-3x2-by-4 2 1 a=2,1,3,2 b=c,2,100404,10008
RESULT grid-groups-3x2-by-4 drawn=6
PX grid-groups-2x2-by-2x3 0 0 a=0,0,2,2 b=4,6,100c02,0
PX grid-groups-2x2-by-2x3 1 0 a=1,0,2,2 b=4,6,100c02,2
PX grid-groups-2x2-by-2x3 0 1 a=0,1,2,2 b=4,6,100c02,30000
PX grid-groups-2x2-by-2x3 1 1 a=1,1,2,2 b=4,6,100c02,30002
RESULT grid-groups-2x2-by-2x3 drawn=4
PX grid-threads-12x4-by-4x2 0 0 a=0,0,3,2 b=c,4,100804,0
PX grid-threads-12x4-by-4x2 1 0 a=1,0,3,2 b=c,4,100804,4
PX grid-threads-12x4-by-4x2 2 0 a=2,0,3,2 b=c,4,100804,8
PX grid-threads-12x4-by-4x2 0 1 a=0,1,3,2 b=c,4,100804,20000
PX grid-threads-12x4-by-4x2 1 1 a=1,1,3,2 b=c,4,100804,20004
PX grid-threads-12x4-by-4x2 2 1 a=2,1,3,2 b=c,4,100804,20008
RESULT grid-threads-12x4-by-4x2 drawn=6
PX grid-threads-10x3-by-4x2 0 0 a=0,0,2,1 b=8,2,100804,0
PX grid-threads-10x3-by-4x2 1 0 a=1,0,2,1 b=8,2,100804,4
RESULT grid-threads-10x3-by-4x2 drawn=2
RESULT grid-threads-3x1-by-4 drawn=0
PX grid-indirect-3x1-by-8 0 0 a=0,0,3,1 b=18,1,100408,0
PX grid-indirect-3x1-by-8 1 0 a=1,0,3,1 b=18,1,100408,8
PX grid-indirect-3x1-by-8 2 0 a=2,0,3,1 b=18,1,100408,10
RESULT grid-indirect-3x1-by-8 drawn=3
RESULT grid-groups-0 drawn=0
";

/// the Metal probe's group `g4` (`mesh_draws_object` and `mesh_draws_mesh`), as it printed it on
/// both Macs, in [`GRID_ANSWERS`]' form.
const OBJECT_ANSWERS: &str = "\
PX object-groups-3-by-4-mesh-2 0 0 a=0,65539,1049604,65548 b=0,0,10002,100402
PX object-groups-3-by-4-mesh-2 1 0 a=0,65539,1049604,65548 b=1,0,10002,100402
PX object-groups-3-by-4-mesh-2 0 2 a=1,65539,1049604,65548 b=0,0,10003,100402
PX object-groups-3-by-4-mesh-2 1 2 a=1,65539,1049604,65548 b=1,0,10003,100402
PX object-groups-3-by-4-mesh-2 2 2 a=1,65539,1049604,65548 b=2,0,10003,100402
PX object-groups-3-by-4-mesh-2 0 4 a=2,65539,1049604,65548 b=0,0,10004,100402
PX object-groups-3-by-4-mesh-2 1 4 a=2,65539,1049604,65548 b=1,0,10004,100402
PX object-groups-3-by-4-mesh-2 2 4 a=2,65539,1049604,65548 b=2,0,10004,100402
PX object-groups-3-by-4-mesh-2 3 4 a=2,65539,1049604,65548 b=3,0,10004,100402
RESULT object-groups-3-by-4-mesh-2 drawn=9
PX object-groups-2-mesh-2x2 0 0 a=0,65538,1049602,65540 b=0,0,20001,100802
PX object-groups-2-mesh-2x2 0 1 a=0,65538,1049602,65540 b=0,1,20001,100802
PX object-groups-2-mesh-2x2 0 2 a=1,65538,1049602,65540 b=0,0,20002,100802
PX object-groups-2-mesh-2x2 1 2 a=1,65538,1049602,65540 b=1,0,20002,100802
PX object-groups-2-mesh-2x2 0 3 a=1,65538,1049602,65540 b=0,1,20002,100802
PX object-groups-2-mesh-2x2 1 3 a=1,65538,1049602,65540 b=1,1,20002,100802
RESULT object-groups-2-mesh-2x2 drawn=6
PX object-threads-6-by-4 0 0 a=0,65538,1049604,65542 b=0,0,10002,100402
PX object-threads-6-by-4 1 0 a=0,65538,1049604,65542 b=1,0,10002,100402
PX object-threads-6-by-4 0 2 a=1,65538,1049602,65542 b=0,0,10003,100402
PX object-threads-6-by-4 1 2 a=1,65538,1049602,65542 b=1,0,10003,100402
PX object-threads-6-by-4 2 2 a=1,65538,1049602,65542 b=2,0,10003,100402
RESULT object-threads-6-by-4 drawn=5
PX object-threads-5x3-by-4x2 0 0 a=0,131074,1050628,196613 b=0,0,10001,100402
PX object-threads-5x3-by-4x2 0 2 a=1,131074,1050625,196613 b=0,0,10002,100402
PX object-threads-5x3-by-4x2 1 2 a=1,131074,1050625,196613 b=1,0,10002,100402
PX object-threads-5x3-by-4x2 0 4 a=65536,131074,1049604,196613 b=0,0,10001,100402
PX object-threads-5x3-by-4x2 0 6 a=65537,131074,1049601,196613 b=0,0,10002,100402
PX object-threads-5x3-by-4x2 1 6 a=65537,131074,1049601,196613 b=1,0,10002,100402
RESULT object-threads-5x3-by-4x2 drawn=6
PX object-grid-0-then-1 0 2 a=1,65538,1049601,65538 b=0,0,10001,100402
RESULT object-grid-0-then-1 drawn=1
PX object-indirect-2 0 0 a=0,65538,1049604,65544 b=0,0,10001,100402
PX object-indirect-2 0 2 a=1,65538,1049604,65544 b=0,0,10002,100402
PX object-indirect-2 1 2 a=1,65538,1049604,65544 b=1,0,10002,100402
RESULT object-indirect-2 drawn=3
";

/// The kind [`object_draws_store_metals_words`] takes on a device without `VK_EXT_mesh_shader`,
/// whose mesh draws are emulated, and not elsewhere.
pub(super) const EMULATED_MESH: [Degradation; 1] = [Degradation::EmulatedMeshDraws];

/// Whether `device` emulates mesh draws: its profile lacks `VK_EXT_mesh_shader` or its
/// `meshShader` feature, as the compile layer decides it.
pub(super) fn emulates(device: &Device) -> bool {
    let enabled = device.compile_device().profile().enabled();
    !enabled.extensions.contains(&VkExtension::ExtMeshShader)
        || !enabled.features.contains(&VkFeature::MeshShader)
}

/// The pass's side in pixels.
const SIDE: u64 = 8;

/// One mesh draw of the probe: its grid, then its threads per object threadgroup and per mesh
/// threadgroup.
#[derive(Clone, Copy)]
enum Draw {
    /// `drawMeshThreadgroups` of these threadgroups.
    Groups([u64; 3], [u64; 3], [u64; 3]),
    /// `drawMeshThreads` of these threads.
    Threads([u64; 3], [u64; 3], [u64; 3]),
    /// `drawMeshThreadgroupsWithIndirectBuffer` of these counts at this offset of a zeroed
    /// 64-byte buffer.
    Indirect([u32; 3], u64, [u64; 3], [u64; 3]),
}

/// The probe's cases of `mesh_draws_grid`, its threads per object threadgroup 1.
const GRID_CASES: [(&str, Draw); 7] = [
    (
        "grid-groups-3x2-by-4",
        Draw::Groups([3, 2, 1], [1; 3], [4, 1, 1]),
    ),
    (
        "grid-groups-2x2-by-2x3",
        Draw::Groups([2, 2, 1], [1; 3], [2, 3, 1]),
    ),
    (
        "grid-threads-12x4-by-4x2",
        Draw::Threads([12, 4, 1], [1; 3], [4, 2, 1]),
    ),
    (
        "grid-threads-10x3-by-4x2",
        Draw::Threads([10, 3, 1], [1; 3], [4, 2, 1]),
    ),
    (
        "grid-threads-3x1-by-4",
        Draw::Threads([3, 1, 1], [1; 3], [4, 1, 1]),
    ),
    (
        "grid-indirect-3x1-by-8",
        Draw::Indirect([3, 1, 1], 8, [1; 3], [8, 1, 1]),
    ),
    ("grid-groups-0", Draw::Groups([0, 2, 1], [1; 3], [4, 1, 1])),
];

/// Grids of `mesh_draws_grid` Metal draws nothing for (the module docs), each `RESULT <case>
/// drawn=0` on the M3 Max: a direct and an indirect grid with an axis past 65,535, and an
/// indirect grid of 2^20 threadgroups.
const PAST_CASES: [(&str, Draw); 4] = [
    (
        "grid-groups-65536",
        Draw::Groups([65536, 1, 1], [1; 3], [1, 1, 1]),
    ),
    (
        "grid-indirect-1x1x65536",
        Draw::Indirect([1, 1, 65536], 0, [1; 3], [1, 1, 1]),
    ),
    (
        "grid-indirect-1024x1024",
        Draw::Indirect([1024, 1024, 1], 0, [1; 3], [1, 1, 1]),
    ),
    (
        "grid-groups-1024x1024",
        Draw::Groups([1024, 1024, 1], [1; 3], [1, 1, 1]),
    ),
];

/// Object draws whose object threadgroups each set a mesh grid Metal launches nothing for (the
/// module docs; `ctl.x + group.x` by `ctl.y`): 1024 by 1024 and 1025 by 1024, 65,536 and 65,537
/// along `x`. The M3 Max (macOS 27.0.1) launches no mesh threadgroup from an object threadgroup
/// that sets such a grid, whatever the others set.
const OBJECT_PAST_CASES: [(&str, [u32; 4], Draw); 2] = [
    (
        "object-grid-1024x1024",
        [1024, 1024, 0, 0],
        Draw::Groups([2, 1, 1], [1; 3], [1, 1, 1]),
    ),
    (
        "object-grid-65536",
        [65536, 1, 0, 0],
        Draw::Groups([2, 1, 1], [1; 3], [1, 1, 1]),
    ),
];

/// The probe's cases of `mesh_draws_object` and `mesh_draws_mesh`, each with the object
/// function's `ctl` (bound at object buffer 0).
const OBJECT_CASES: [(&str, [u32; 4], Draw); 6] = [
    (
        "object-groups-3-by-4-mesh-2",
        [2, 1, 0, 0],
        Draw::Groups([3, 1, 1], [4, 1, 1], [2, 1, 1]),
    ),
    (
        "object-groups-2-mesh-2x2",
        [1, 2, 0, 0],
        Draw::Groups([2, 1, 1], [2, 1, 1], [2, 2, 1]),
    ),
    (
        "object-threads-6-by-4",
        [2, 1, 0, 0],
        Draw::Threads([6, 1, 1], [4, 1, 1], [2, 1, 1]),
    ),
    (
        "object-threads-5x3-by-4x2",
        [1, 1, 0, 0],
        Draw::Threads([5, 3, 1], [4, 2, 1], [2, 1, 1]),
    ),
    (
        "object-grid-0-then-1",
        [0, 1, 0, 0],
        Draw::Groups([2, 1, 1], [1; 3], [2, 1, 1]),
    ),
    (
        "object-indirect-2",
        [1, 1, 0, 0],
        Draw::Indirect([2, 1, 1], 4, [4, 1, 1], [2, 1, 1]),
    ),
];

fn size(axes: [u64; 3]) -> Size {
    Size::new(axes[0], axes[1], axes[2])
}

/// The lane's device, queue and the fixture's library.
struct Lane {
    device: Device,
    queue: CommandQueue,
    library: Library,
}

impl Lane {
    fn new(device: Device) -> Lane {
        let bitcode = llvm_assemble(&engine_dir().join("mesh_draws.ll")).unwrap();
        Lane {
            queue: device.new_command_queue().unwrap(),
            library: device.new_library_with_air(&bitcode).unwrap(),
            device,
        }
    }

    /// The state of `mesh` (and `object`, where given) with the fragment function, over the
    /// pass's two attachments.
    fn pipeline(&self, object: Option<&str>, mesh: &str) -> Result<RenderPipelineState, Error> {
        self.device.new_mesh_render_pipeline_state(
            &self.descriptor(object, mesh),
            &PipelineOptions::default(),
        )
    }

    /// The descriptor of [`pipeline`](Self::pipeline)'s state.
    fn descriptor(&self, object: Option<&str>, mesh: &str) -> MeshPipelineDescriptor {
        let function = |name| {
            StageSource::Function(
                self.library
                    .new_function_with_name(name)
                    .unwrap()
                    .function_descriptor(),
            )
        };
        let mut descriptor = MeshPipelineDescriptor::new(function(mesh));
        descriptor.object = object.map(function);
        descriptor.fragment = Some(function("mesh_draws_fragment"));
        for attachment in &mut descriptor.color_attachments[..2] {
            *attachment = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA32Uint));
        }
        descriptor
    }

    /// A pass of `state` whose commands `body` encodes: the probe's lines of the pixels it drew,
    /// as case `name`, or the first refusal `body` returns (the pass then ends without it).
    fn run(
        &self,
        name: &str,
        state: &RenderPipelineState,
        body: impl FnOnce(&mut RenderCommandEncoder) -> Result<(), Error>,
    ) -> Result<Vec<String>, Error> {
        let targets = [0, 1].map(|_| texture(&self.device, PixelFormat::RGBA32Uint, SIDE));
        let mut pass = RenderPassDescriptor::new();
        for (attachment, target) in pass.color_attachments.iter_mut().zip(&targets) {
            attachment.texture = Some(target.clone());
            attachment.load_action = LoadAction::Clear;
            attachment.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
            attachment.store_action = StoreAction::Store;
        }
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encoder.set_render_pipeline_state(state).unwrap();
        let encoded = body(&mut encoder);
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
        encoded?;
        // Each `VkPipeline` the draws created (an emulated mesh draw's compute pipeline beside
        // its graphics one) within the creation ceiling, and the pass's GPU time, printed.
        let created: Vec<std::time::Duration> = command_buffer
            .executions()
            .iter()
            .flat_map(|e| e.pipeline_creation().to_vec())
            .collect();
        for made in &created {
            crate::budgets::creation_within_ceiling(name, *made).unwrap();
        }
        let gpu = match (
            command_buffer.gpu_start_time(),
            command_buffer.gpu_end_time(),
        ) {
            (Ok(start), Ok(end)) => format!("{:.1} us", (end - start) * 1e6),
            _ => "untimed".to_owned(),
        };
        eprintln!(
            "TIMING {name}: created {:?} ms, gpu {gpu}",
            created.iter().map(|d| d.as_millis()).collect::<Vec<_>>()
        );
        let words = |t| -> Vec<u32> {
            texels(t, 16)
                .as_chunks::<4>()
                .0
                .iter()
                .map(|w| u32::from_le_bytes(*w))
                .collect()
        };
        let (a, b) = (words(&targets[0]), words(&targets[1]));
        let mut lines = Vec::new();
        for y in 0..SIDE as usize {
            for x in 0..SIDE as usize {
                let at = 4 * (y * SIDE as usize + x);
                let (wa, wb) = (&a[at..at + 4], &b[at..at + 4]);
                if wa.iter().chain(wb).all(|&w| w == 0) {
                    continue;
                }
                let decimal: Vec<String> = wa.iter().map(u32::to_string).collect();
                let hex: Vec<String> = wb.iter().map(|w| format!("{w:x}")).collect();
                lines.push(format!(
                    "PX {name} {x} {y} a={} b={}",
                    decimal.join(","),
                    hex.join(",")
                ));
            }
        }
        lines.push(format!("RESULT {name} drawn={}", lines.len()));
        Ok(lines)
    }
}

/// `draw` on `e`, its indirect buffer one of `device`'s, kept in `held`.
fn encode(
    device: &Device,
    e: &mut RenderCommandEncoder,
    draw: Draw,
    held: &mut Vec<Buffer>,
) -> Result<(), Error> {
    match draw {
        Draw::Groups(groups, object, mesh) => {
            e.draw_mesh_threadgroups(size(groups), size(object), size(mesh))
        }
        Draw::Threads(threads, object, mesh) => {
            e.draw_mesh_threads(size(threads), size(object), size(mesh))
        }
        Draw::Indirect(counts, offset, object, mesh) => {
            let mut bytes = vec![0u8; 64];
            for (i, count) in counts.iter().enumerate() {
                let at = offset as usize + 4 * i;
                bytes[at..at + 4].copy_from_slice(&count.to_le_bytes());
            }
            let buffer = buffer_with(device, &bytes).unwrap();
            let drawn =
                e.draw_mesh_threadgroups_indirect(&buffer, offset, size(object), size(mesh));
            held.push(buffer);
            drawn
        }
    }
}

/// The probe's lines of case `name` among `answers`.
fn metals(answers: &str, name: &str) -> Vec<String> {
    answers
        .lines()
        .filter(|line| {
            let mut words = line.split(' ');
            words.next();
            words.next() == Some(name)
        })
        .map(str::to_owned)
        .collect()
}

/// Whether `refused` is the pipeline refusal of a device without mesh shading, printed.
fn mesh_refused(refused: &Error, device: &str) -> bool {
    match refused {
        Error::Unsupported(u) if u.feature == Feature::MeshShading => {
            eprintln!("refused: mesh render pipelines on {device}: {u}");
            true
        }
        _ => false,
    }
}

/// Each grid case draws the words Metal drew (the module docs), and each of [`PAST_CASES`]
/// nothing.
#[test]
fn mesh_draws_store_metals_words() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let name = device.name().to_owned();
        let lane = Lane::new(device);
        let started = std::time::Instant::now();
        let state = match lane.pipeline(None, "mesh_draws_grid") {
            Ok(state) => state,
            Err(refused) if mesh_refused(&refused, &name) => return,
            Err(error) => panic!("mesh_draws_grid on {name}: {error}"),
        };
        eprintln!(
            "TIMING state on {name}: {} ms",
            started.elapsed().as_millis()
        );
        for (case, draw) in GRID_CASES {
            let mut held = Vec::new();
            let drawn = lane.run(case, &state, |e| encode(&lane.device, e, draw, &mut held));
            match drawn {
                Ok(lines) => assert_eq!(lines, metals(GRID_ANSWERS, case), "{case} on {name}"),
                Err(error) => panic!("{case} on {name}: {error}"),
            }
        }
        for (case, draw) in PAST_CASES {
            let mut held = Vec::new();
            let drawn = lane.run(case, &state, |e| encode(&lane.device, e, draw, &mut held));
            match drawn {
                Ok(lines) => assert_eq!(lines, [format!("RESULT {case} drawn=0")], "{case}"),
                Err(error) => panic!("{case} on {name}: {error}"),
            }
        }
    }
    let reported = messages.take();
    assert!(
        reported.is_empty(),
        "the validation layer reported: {reported:#?}"
    );
}

/// Each object case draws the words Metal drew (the module docs), but those that leave a partial
/// object threadgroup, refused, and each of [`OBJECT_PAST_CASES`] nothing; a
/// device without `VK_EXT_mesh_shader` announces its emulated draws once ([`EMULATED_MESH`]), and
/// another none.
#[test]
fn object_draws_store_metals_words() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let name = device.name().to_owned();
        let lane = Lane::new(device);
        let state = match lane.pipeline(Some("mesh_draws_object"), "mesh_draws_mesh") {
            Ok(state) => state,
            Err(refused) if mesh_refused(&refused, &name) => return,
            Err(error) => panic!("mesh_draws_object on {name}: {error}"),
        };
        // Each case with `ctl` in a buffer, then set as inline bytes, which it reads alike.
        let both = OBJECT_CASES.iter().flat_map(|&c| [(c, false), (c, true)]);
        for ((case, ctl, draw), inline) in both {
            let mut held = Vec::new();
            let bytes: Vec<u8> = ctl.iter().flat_map(|w| w.to_le_bytes()).collect();
            let ctl = buffer_with(&lane.device, &bytes).unwrap();
            let drawn = lane.run(case, &state, |e| {
                match inline {
                    true => e.set_object_bytes(&bytes, 0)?,
                    false => e.set_object_buffer(Some(&ctl), 0, 0)?,
                }
                encode(&lane.device, e, draw, &mut held)
            });
            let metals = metals(OBJECT_ANSWERS, case);
            match (draw, drawn) {
                (Draw::Threads(..), Err(Error::Bind(BindError::Unsupported(u))))
                    if u.feature == Feature::MeshShading && u.needs == Needs::NotImplemented =>
                {
                    eprintln!(
                        "{case} on {name}: refused, a partial object threadgroup; Metal's: \
                         {metals:?}"
                    )
                }
                (_, Ok(lines)) => assert_eq!(lines, metals, "{case} on {name}"),
                (_, Err(error)) => panic!("{case} on {name}: {error}"),
            }
        }
        for (case, ctl, draw) in OBJECT_PAST_CASES {
            let mut held = Vec::new();
            let bytes: Vec<u8> = ctl.iter().flat_map(|w| w.to_le_bytes()).collect();
            let drawn = lane.run(case, &state, |e| {
                e.set_object_bytes(&bytes, 0)?;
                encode(&lane.device, e, draw, &mut held)
            });
            match drawn {
                Ok(lines) => assert_eq!(lines, [format!("RESULT {case} drawn=0")], "{case}"),
                Err(error) => panic!("{case} on {name}: {error}"),
            }
        }
        // Exactly the emulation's kind where it draws, beside what the state's making forces
        // (its stages' notes: `FastMathRewrites` on ANV), the `Shared` buffers, and the reads of
        // each pass's GPU times its timing line prints (`gpu_times::read_kinds`:
        // `GpuTimesFromHost` on MoltenVK); before, a device that does not emulate let any kind
        // pass.
        let descriptor = lane.descriptor(Some("mesh_draws_object"), "mesh_draws_mesh");
        let mut made = made_mesh_kinds(&lane.device, &descriptor);
        made.extend(shared_kinds(&lane.device));
        made.extend(read_kinds(&lane.device));
        let emulated: &[Degradation] = match emulates(&lane.device) {
            true => &EMULATED_MESH,
            false => &[],
        };
        assert_made_once(&messages, emulated, &made);
    }
    let reported = messages.take();
    assert!(
        reported.is_empty(),
        "the validation layer reported: {reported:#?}"
    );
}

/// One case of Metal probe 5 over the owned fixture `mesh_object_grid`: the draw's object
/// threadgroups and their threads, whether it is direct, and the twelve counter words the M3 Max
/// (macOS 27.0.1) left: the threadgroups counted, their linear positions and those positions'
/// squares summed (each wrapping), the largest position, then the grid's `x`, `y`, `z` and the
/// threads per grid's `x` each threadgroup read, or-ed, then and-ed (from all ones).
struct GridCase {
    groups: [u32; 3],
    threads: [u64; 3],
    direct: bool,
    words: [u32; 12],
}

/// The words of a draw that ran no object threadgroup.
const RAN_NONE: [u32; 12] = [0, 0, 0, 0, 0, 0, 0, 0, !0, !0, !0, !0];

/// The words of `n` threadgroups in a row along `x`, each reading a grid of `grid` (and threads
/// per grid of `threads` along `x`): the sums Metal's counters wrap to.
const fn row(n: u32, sum: u32, squares: u32, grid: [u32; 3], threads: u32) -> [u32; 12] {
    let [x, y, z] = grid;
    [n, sum, squares, n - 1, x, y, z, threads, x, y, z, threads]
}

/// Probe 5's cases, each as Metal ran it (the module docs): grids past every device's task axis
/// and total, which run in pieces; grids whose threads per grid wrap to 32 bits, which a direct
/// draw runs as the wrapped threads fill and an indirect one whole, reporting the wrapped grid;
/// and grids with a zero count. lavapipe does not run a case of more than 1,048,576 threadgroups in
/// time, and skips it.
const OBJECT_GRID_CASES: [GridCase; 17] = {
    const fn case(groups: [u32; 3], threads: [u64; 3], direct: bool, words: [u32; 12]) -> GridCase {
        GridCase {
            groups,
            threads,
            direct,
            words,
        }
    }
    let one = [1, 1, 1];
    [
        case(
            [70_000, 1, 1],
            one,
            true,
            row(70_000, 2_449_965_000, 3_148_892_776, [70_000, 1, 1], 70_000),
        ),
        case(
            [70_000, 1, 1],
            one,
            false,
            row(70_000, 2_449_965_000, 3_148_892_776, [70_000, 1, 1], 70_000),
        ),
        case(
            [300, 300, 2],
            [4, 2, 1],
            true,
            row(180_000, 3_315_008_112, 4_292_449_072, [300, 300, 2], 1200),
        ),
        case(
            [300, 300, 2],
            [4, 2, 1],
            false,
            row(180_000, 3_315_008_112, 4_292_449_072, [300, 300, 2], 1200),
        ),
        case(
            [131_070, 2, 1],
            one,
            true,
            row(
                262_140,
                4_293_787_658,
                1_436_942_306,
                [131_070, 2, 1],
                131_070,
            ),
        ),
        case(
            [131_070, 2, 1],
            one,
            false,
            row(
                262_140,
                4_293_787_658,
                1_436_942_306,
                [131_070, 2, 1],
                131_070,
            ),
        ),
        case(
            [65536, 70, 1],
            one,
            true,
            row(
                4_587_520,
                4_292_673_536,
                1_432_420_352,
                [65536, 70, 1],
                65536,
            ),
        ),
        case(
            [65536, 70, 1],
            one,
            false,
            row(
                4_587_520,
                4_292_673_536,
                1_432_420_352,
                [65536, 70, 1],
                65536,
            ),
        ),
        case(
            [4_194_307, 1, 1],
            one,
            true,
            row(
                4_194_307,
                10_485_763,
                1_457_520_645,
                [4_194_307, 1, 1],
                4_194_307,
            ),
        ),
        case(
            [4_194_307, 1, 1],
            one,
            false,
            row(
                4_194_307,
                10_485_763,
                1_457_520_645,
                [4_194_307, 1, 1],
                4_194_307,
            ),
        ),
        case(
            [2, 16_777_217, 1],
            [1, 128, 1],
            true,
            row(33_554_434, 50_331_649, 2_936_012_801, [2, 16_777_217, 1], 2),
        ),
        case([3, 0, 1], one, true, RAN_NONE),
        case([3, 0, 1], one, false, RAN_NONE),
        case([33_554_432, 1, 1], [128, 1, 1], true, RAN_NONE),
        case(
            [33_554_432, 1, 1],
            [128, 1, 1],
            false,
            row(33_554_432, 4_278_190_080, 2_868_903_936, [0, 1, 1], 0),
        ),
        case(
            [33_554_433, 1, 1],
            [128, 1, 1],
            true,
            row(1, 0, 0, [1, 1, 1], 128),
        ),
        case(
            [33_554_433, 1, 1],
            [128, 1, 1],
            false,
            row(33_554_433, 16_777_216, 2_868_903_936, [1, 1, 1], 128),
        ),
    ]
};

/// An object grid of more threadgroups than Metal's GPU runs within its watchdog
/// (`bind::METAL_OBJECT_GRID_TOTAL`), direct and indirect, which the M3 Max discards
/// (`MTLCommandBufferErrorInternal`, nothing running from the draw's encoder on; probe 6 measured
/// 2^29 of this fixture's function and 2^30 of one that does next to nothing so): drawn nothing,
/// failing the command buffer with Metal's error (`CommandBufferError::GpuRecoveryVictim`), a
/// direct draw's from its encoder on, so a blit encoder after it runs nothing.
const PAST_GRID: [u32; 3] = [(1 << 29) + 1, 1, 1];

/// Each of [`OBJECT_GRID_CASES`] leaves Metal's words on a device with `VK_EXT_mesh_shader`, its
/// grid folded into the task stage's limits where it passes them, and each draw of [`PAST_GRID`]
/// draws nothing and fails its command buffer as Metal's GPU discards it, a direct one from its
/// encoder on. The emulation refuses a direct grid past its compute limits
/// (`Unsupported{MeshShading}`), and an indirect one past them draws nothing and fails its command
/// buffer (`MeshGridPastLimits`), each printed.
///
/// A case runs only where the device can finish it within the lane's wait, measured on the cases
/// before it (`TIMING` lines print each case's milliseconds, and each skipped case's reason). A
/// device takes its own time per object threadgroup -- an RTX 5080 Laptop GPU 2.4 microseconds,
/// where Metal's own GPU ran the largest grids here inside its 5 s watchdog -- and Metal's largest
/// are 33.5 million threadgroups.
#[test]
fn object_grids_run_as_metal_runs_them() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let name = device.name().to_owned();
        let bitcode = llvm_assemble(&engine_dir().join("mesh_object_grid.ll")).unwrap();
        let library = device.new_library_with_air(&bitcode).unwrap();
        let function = |name| {
            StageSource::Function(
                library
                    .new_function_with_name(name)
                    .unwrap()
                    .function_descriptor(),
            )
        };
        let mut descriptor = MeshPipelineDescriptor::new(function("mesh_object_grid_mesh"));
        descriptor.object = Some(function("mesh_object_grid_object"));
        descriptor.fragment = Some(function("mesh_object_grid_fragment"));
        descriptor.color_attachments[0] =
            Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA32Uint));
        let state =
            match device.new_mesh_render_pipeline_state(&descriptor, &PipelineOptions::default()) {
                Ok(state) => state,
                Err(refused) if mesh_refused(&refused, &name) => return,
                Err(error) => panic!("mesh_object_grid on {name}: {error}"),
            };
        let queue = device.new_command_queue().unwrap();
        let target = texture(&device, PixelFormat::RGBA32Uint, SIDE);
        let run = |groups: [u32; 3], threads: [u64; 3], direct: bool| {
            let mut pass = RenderPassDescriptor::new();
            let attachment = &mut pass.color_attachments[0];
            attachment.texture = Some(target.clone());
            attachment.load_action = LoadAction::Clear;
            attachment.store_action = StoreAction::Store;
            let mut initial = vec![0u8; 32];
            initial.extend([0xFF; 16]);
            let counters = buffer_with(&device, &initial).unwrap();
            let bytes: Vec<u8> = groups.iter().flat_map(|w| w.to_le_bytes()).collect();
            let arguments = buffer_with(&device, &bytes).unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
            encoder.set_render_pipeline_state(&state).unwrap();
            encoder.set_object_buffer(Some(&counters), 0, 0).unwrap();
            let one = Size::new(1, 1, 1);
            let drawn = match direct {
                true => {
                    encoder.draw_mesh_threadgroups(size(groups.map(u64::from)), size(threads), one)
                }
                false => encoder.draw_mesh_threadgroups_indirect(&arguments, 0, size(threads), one),
            };
            encoder.end_encoding().unwrap();
            // An encoder after the draw's, which runs unless the command buffer fails from the
            // draw's encoder on.
            let tail = buffer_with(&device, &[0u8; 4]).unwrap();
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            blit.fill_buffer(&tail, 0..4, 0x22).unwrap();
            blit.end_encoding().unwrap();
            let started = std::time::Instant::now();
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            let took = started.elapsed();
            let words: Vec<u32> = read_buffer(&counters, 0, 48)
                .unwrap()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|w| u32::from_le_bytes(*w))
                .collect();
            let what = format!(
                "{groups:?} of {threads:?}, {} on {name}",
                if direct { "direct" } else { "indirect" }
            );
            let tail = read_buffer(&tail, 0, 4).unwrap() == [0x22; 4];
            eprintln!(
                "TIMING {what}: {} ms, {:?}",
                took.as_millis(),
                command_buffer.error()
            );
            (drawn, command_buffer.error(), words, what, tail, took)
        };
        // The emulation's refusal of a direct grid past its compute limits.
        let refused = |drawn: &Result<(), Error>| match drawn {
            Err(Error::Bind(BindError::Unsupported(u))) if u.feature == Feature::MeshShading => {
                emulates(&device)
            }
            _ => false,
        };
        // An indirect grid the emulation cannot dispatch: an axis past its compute limits.
        let limits = device
            .compile_device()
            .profile()
            .limits()
            .max_compute_work_group_count;
        let undispatched = |c: &GridCase| {
            emulates(&device)
                && !c.direct
                && !c.groups.contains(&0)
                && (0..3).any(|i| c.groups[i] > limits[i])
        };
        // What the device takes per object threadgroup, from the cases it has run: a grid of tens
        // of millions of them outlasts `WAIT` on a device slower per threadgroup than Apple's,
        // whose GPU ran Metal's largest here inside its 5 s watchdog. The cases run in ascending
        // order of the threadgroups Metal ran, so the measurement is there before the largest; one
        // the measurement puts past half of `WAIT` is printed and skipped, rather than left
        // grinding on the device after the wait has given up.
        let mut per_group = 0.0f64;
        for c in &OBJECT_GRID_CASES {
            // The threadgroups Metal ran.
            if cfg!(m2v_cpu_icd) && c.words[0] > 1 << 20 {
                continue;
            }
            let groups = f64::from(c.words[0]);
            if per_group > 0.0 && groups * per_group > WAIT.as_secs_f64() / 2.0 {
                eprintln!(
                    "TIMING {:?} of {:?}, {} on {name}: {groups} threadgroups at {:.2} us each is \
                     past half of the {} s wait, skipped",
                    c.groups,
                    c.threads,
                    if c.direct { "direct" } else { "indirect" },
                    per_group * 1e6,
                    WAIT.as_secs(),
                );
                continue;
            }
            let (drawn, error, words, what, tail, took) = run(c.groups, c.threads, c.direct);
            if groups > 1e6 {
                per_group = per_group.max(took.as_secs_f64() / groups);
            }
            if refused(&drawn) {
                eprintln!("{what}: refused, emulated: {drawn:?}");
                continue;
            }
            if let Err(e) = drawn {
                panic!("{what}: {e}");
            }
            if undispatched(c) {
                eprintln!("{what}: past the emulation's dispatch: {error:?}");
                assert_eq!(
                    error,
                    Some(CommandBufferError::MeshGridPastLimits),
                    "{what}"
                );
                assert_eq!(words, RAN_NONE, "{what}");
                assert!(tail, "{what}: the encoder after the draw's");
                continue;
            }
            assert_eq!(error, None, "{what}");
            assert_eq!(words, c.words, "{what}");
            assert!(tail, "{what}: the encoder after the draw's");
        }
        for direct in [true, false] {
            let (drawn, error, words, what, tail, _) = run(PAST_GRID, [1; 3], direct);
            if refused(&drawn) {
                eprintln!("{what}: refused, emulated: {drawn:?}");
                continue;
            }
            if let Err(e) = drawn {
                panic!("{what}: {e}");
            }
            assert_eq!(error, Some(CommandBufferError::GpuRecoveryVictim), "{what}");
            assert_eq!(words, RAN_NONE, "{what}");
            // Metal runs nothing from the draw's encoder on; an indirect draw's count is the GPU's,
            // and the runtime runs the encoders after it (docs/API.md, "Accepted differences").
            assert_eq!(tail, !direct, "{what}: the encoder after the draw's");
        }
    }
    let reported = messages.take();
    assert!(
        reported.is_empty(),
        "the validation layer reported: {reported:#?}"
    );
}

/// The pixels Metal draws for `cases` drawn in order in one pass, as case `name`: each case's
/// pixels from its probe lines (`answers`), a later case's over an earlier one's, since Metal
/// rasterizes an encoder's primitives in order and the pass neither blends nor tests depth.
fn overlaid(cases: &[(&str, &str)], name: &str) -> Vec<String> {
    let mut pixels = std::collections::BTreeMap::new();
    for (answers, case) in cases {
        for line in metals(answers, case) {
            let words: Vec<&str> = line.split(' ').collect();
            if let ["PX", _, x, y, a, b] = words[..] {
                let at: (u32, u32) = (y.parse().unwrap(), x.parse().unwrap());
                pixels.insert(at, format!("{a} {b}"));
            }
        }
    }
    let mut lines: Vec<String> = pixels
        .iter()
        .map(|((y, x), data)| format!("PX {name} {x} {y} {data}"))
        .collect();
    lines.push(format!("RESULT {name} drawn={}", lines.len()));
    lines
}

/// Draws of both pipelines in one pass draw the words Metal draws for each in order (the
/// module docs): two draws of the object function, whose second's mesh threadgroups and, on a
/// device without `VK_EXT_mesh_shader`, records follow the first's in the command buffer's
/// records pool, and then a draw without one.
#[test]
fn draws_of_both_pipelines_in_one_pass_store_metals_words() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let name = device.name().to_owned();
        let lane = Lane::new(device);
        let (object, grid) = match (
            lane.pipeline(Some("mesh_draws_object"), "mesh_draws_mesh"),
            lane.pipeline(None, "mesh_draws_grid"),
        ) {
            (Ok(object), Ok(grid)) => (object, grid),
            (Err(refused), _) | (_, Err(refused)) if mesh_refused(&refused, &name) => return,
            (Err(error), _) | (_, Err(error)) => panic!("the mesh pipelines on {name}: {error}"),
        };
        let ctl = |words: [u32; 4]| {
            let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            buffer_with(&lane.device, &bytes).unwrap()
        };
        let (first, second) = (ctl([1, 2, 0, 0]), ctl([2, 1, 0, 0]));
        let one = [1; 3];
        let lines = lane
            .run("both", &object, |e| {
                e.set_object_buffer(Some(&first), 0, 0)?;
                e.draw_mesh_threadgroups(size([2, 1, 1]), size([2, 1, 1]), size([2, 2, 1]))?;
                e.set_object_buffer(Some(&second), 0, 0)?;
                e.draw_mesh_threadgroups(size([3, 1, 1]), size([4, 1, 1]), size([2, 1, 1]))?;
                e.set_render_pipeline_state(&grid)?;
                e.draw_mesh_threadgroups(size([2, 2, 1]), size(one), size([2, 3, 1]))
            })
            .unwrap();
        let metals = overlaid(
            &[
                (OBJECT_ANSWERS, "object-groups-2-mesh-2x2"),
                (OBJECT_ANSWERS, "object-groups-3-by-4-mesh-2"),
                (GRID_ANSWERS, "grid-groups-2x2-by-2x3"),
            ],
            "both",
        );
        assert_eq!(lines, metals, "on {name}");
    }
    let reported = messages.take();
    assert!(
        reported.is_empty(),
        "the validation layer reported: {reported:#?}"
    );
}

/// What Metal's validation stops draws as plain Metal does (a probe): more than 1024 threads in a
/// threadgroup and a misaligned argument offset draw nothing; what stops plain Metal too, and
/// arguments past the buffer, are refused typed. The pass draws nothing of it.
#[test]
fn mesh_draws_answer_as_plain_metal_and_refuse_the_rest() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let name = device.name().to_owned();
        let lane = Lane::new(device);
        let state = match lane.pipeline(None, "mesh_draws_grid") {
            Ok(state) => state,
            Err(refused) if mesh_refused(&refused, &name) => return,
            Err(error) => panic!("mesh_draws_grid on {name}: {error}"),
        };
        let invalid =
            |call, problem| Err(Error::Invalid(Invalid::new(call, Problem::Draw(problem))));
        let one = size([1; 3]);
        let arguments = buffer_with(&lane.device, &[0; 16]).unwrap();
        let lines = lane
            .run("refused", &state, |e| {
                // Probe 1b: more than 1024 threads in a mesh threadgroup draw nothing.
                assert_eq!(
                    e.draw_mesh_threadgroups(one, one, size([1025, 1, 1])),
                    Ok(())
                );
                assert_eq!(
                    e.draw_mesh_threads(size([64, 1, 1]), one, size([32, 33, 1])),
                    Ok(())
                );
                // Probe 1b: a misaligned argument offset draws nothing; arguments past the
                // buffer are refused, as every indirect draw's.
                let indirect = "draw_mesh_threadgroups_indirect";
                assert_eq!(
                    e.draw_mesh_threadgroups_indirect(&arguments, 2, one, one),
                    Ok(())
                );
                assert_eq!(
                    e.draw_mesh_threadgroups_indirect(&arguments, 8, one, one),
                    invalid(indirect, DrawProblem::IndirectBufferRange)
                );
                // Probe 1b: a draw of primitives with a mesh render pipeline's state.
                assert_eq!(
                    e.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0),
                    invalid("draw_primitives", DrawProblem::DrawKind)
                );
                Ok(())
            })
            .unwrap();
        assert_eq!(lines, ["RESULT refused drawn=0"], "on {name}");
    }
    let reported = messages.take();
    assert!(
        reported.is_empty(),
        "the validation layer reported: {reported:#?}"
    );
}
