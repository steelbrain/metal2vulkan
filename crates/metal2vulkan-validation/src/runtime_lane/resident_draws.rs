//! Draws that each bind a resource of their own while the pass makes every one resident, as a
//! page's render bundle does with a bind group a draw, on the device the cfg selects: what they
//! draw, and what they cost.
//!
//! Two shapes of draw, `resident_draws.metal`, each placing one small triangle into a 512x512
//! `RGBA8Unorm` target:
//! - **slot**: draw `i` binds position `i` (16 bytes) as vertex buffer 0 (`rd_slot_vertex`,
//!   `rd_color_fragment`); the pass makes the positions resident for its vertex stage.
//! - **entry**: draw `i` binds entry `i` of one argument table (24 bytes: the position's address,
//!   a texture's resource ID and a sampler's) as vertex and fragment buffer 0 (`rd_entry_vertex`
//!   reads the position through the pointer, `rd_entry_fragment` samples the texture); the pass
//!   makes the positions, the table and the texture resident.
//!
//! Each with the positions in a buffer of their own each (`own`), or at their offsets in one
//! (`one`). Three paths for each: the encoder's own draws (`encoder`), an indirect command
//! buffer's execution of the same draws made resident for reading (`read`: `commit` records the
//! CPU's records), and the same execution after a compute encoder made the buffer resident for
//! writing, with `rd_touch`, a kernel that calls no indirect-command operation (`write`: `commit`
//! records it from the CPU's records too, as no function that may write them met the buffer).
//! Off the measured paths, `encode` makes the buffer resident for writing with `rd_encode`, which
//! may reset a command (it resets none): a host step records each execution, resolving each
//! record's addresses among the resources made resident.
//!
//! [`each_path_draws_what_the_encoder_draws`] checks, under the validation layer, that every path
//! and binding of a shape draws the same pixels. [`resident_draws`] measures, without the layer:
//! printed as `RESIDENT-DRAWS` lines, for each shape, binding, draw count and path, the frame's
//! time (encoding to completion) over [`frames`] frames after one to warm, median and largest,
//! per draw, and the device's counters over those frames, each case's held to what its resident
//! resources' count does not set (`assert_bounded`). [`COUNTS_VAR`] lists the draw counts
//! (500, 1000, 2000 and 5000 by default), [`FRAMES_VAR`] the frames timed, [`ONLY_VAR`] keeps the
//! cases whose name (`shape binding path`, such as `slot own write`) contains it, and
//! [`DUMP_VAR`] prints every counter a case's frames moved, and a hash of what its last frame
//! drew, as `RESIDENT-DRAWS-COUNTS` lines, to compare two builds. It is ignored by default: run it
//! by name with `--run-ignored all`, in a build with `--cfg m2v_system_alloc`.

use super::degrade_notices::on_device;
use super::{buffer_with, validated_device_with, Messages, ScratchCache, WAIT};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::metal::{
    IndirectCommandType, PixelFormat, PrimitiveType, RenderStages, ResourceUsage, TextureUsage,
};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, ComputePipelineDescriptor, RenderPipelineDescriptor, StageSource,
};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    Buffer, CommandBufferStatus, CommandQueue, ComputePipelineState, Degradation, Device,
    IndirectCommandBuffer, IndirectCommandBufferDescriptor, LoadAction, PipelineOptions, Region,
    RenderCommandEncoder, RenderPassDescriptor, RenderPipelineState, ResourceOptions, ResourceRef,
    SamplerState, SamplerStateOptions, Size, StoreAction, Texture, TextureDescriptor, WorkCounts,
};
use std::time::Instant;

/// The variable that sets [`counts`].
const COUNTS_VAR: &str = "RESIDENT_DRAWS_COUNTS";
/// The variable that sets [`frames`].
const FRAMES_VAR: &str = "RESIDENT_DRAWS_FRAMES";
/// The variable that keeps the cases whose name contains it.
const ONLY_VAR: &str = "RESIDENT_DRAWS_ONLY";
/// The variable that prints every counter a case moved, and its pixels' hash.
const DUMP_VAR: &str = "RESIDENT_DRAWS_DUMP";

/// The target's width and height.
const SIDE: u64 = 512;

/// The draw counts: 500, 1000, 2000 and 5000 unless [`COUNTS_VAR`] lists others, comma-separated.
fn counts() -> Vec<u64> {
    std::env::var(COUNTS_VAR).map_or(vec![500, 1000, 2000, 5000], |list| {
        list.split(',')
            .map(|count| count.trim().parse().expect(COUNTS_VAR))
            .collect()
    })
}

/// The frames timed after the one that warms: 10 unless [`FRAMES_VAR`] says.
fn frames() -> usize {
    std::env::var(FRAMES_VAR).map_or(10, |frames| frames.parse().expect(FRAMES_VAR))
}

/// What a draw binds (the module docs).
#[derive(Clone, Copy, Debug)]
enum Shape {
    Slot,
    Entry,
}

/// How a pass's draws reach the device (the module docs).
#[derive(Clone, Copy, Debug)]
enum Path {
    Encoder,
    Read,
    Write,
    Encode,
}

/// The measured paths, in the order a buffer's executions take them.
const PATHS: [Path; 3] = [Path::Encoder, Path::Read, Path::Write];

/// The bytes of an argument-table entry.
const ENTRY: u64 = 24;

/// The functions' states, the texture and sampler the entries name, the target and a queue.
struct Scene {
    device: Device,
    slot: RenderPipelineState,
    entry: RenderPipelineState,
    touch: ComputePipelineState,
    encode: ComputePipelineState,
    count: Buffer,
    sampled: Texture,
    sampler: SamplerState,
    target: Texture,
    queue: CommandQueue,
}

impl Scene {
    fn new(device: &Device) -> Scene {
        let bitcode = llvm_assemble(&engine_dir().join("resident_draws.ll")).unwrap();
        let library = device.new_library_with_air(&bitcode).unwrap();
        let function = |name: &str| {
            StageSource::Function(
                library
                    .new_function_with_name(name)
                    .unwrap()
                    .function_descriptor(),
            )
        };
        let state = |vertex: &str, fragment: &str| {
            let mut descriptor = RenderPipelineDescriptor::new(function(vertex));
            descriptor.fragment = Some(function(fragment));
            descriptor.color_attachments[0] =
                Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
            descriptor.support_indirect_command_buffers = true;
            device
                .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
                .unwrap()
        };
        let kernel = |name: &str| {
            device
                .new_compute_pipeline_state(
                    &ComputePipelineDescriptor::new(function(name)),
                    &PipelineOptions::default(),
                )
                .unwrap()
        };
        let mut sampled = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 4, 4, false);
        sampled.usage = TextureUsage::SHADER_READ;
        sampled.resource_options = ResourceOptions::shared();
        let sampled = device.new_texture(&sampled).unwrap();
        let texels: Vec<u8> = [40u8, 160, 220, 255].repeat(16);
        (sampled.replace_region(Region::new_2d(0, 0, 4, 4), 0, 0, &texels, 16, 0)).unwrap();
        let mut options = SamplerStateOptions::default();
        options.support_argument_buffers = true;
        let sampler = device
            .new_sampler_state_with_options(&SamplerDescriptor::new(), options)
            .unwrap();
        let mut target = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
        target.usage = TextureUsage::RENDER_TARGET;
        target.resource_options = ResourceOptions::shared();
        Scene {
            slot: state("rd_slot_vertex", "rd_color_fragment"),
            entry: state("rd_entry_vertex", "rd_entry_fragment"),
            touch: kernel("rd_touch"),
            encode: kernel("rd_encode"),
            count: buffer_with(device, &[0; 4]).unwrap(),
            sampled,
            sampler,
            target: device.new_texture(&target).unwrap(),
            queue: device.new_command_queue().unwrap(),
            device: device.clone(),
        }
    }

    /// The target's bytes.
    fn pixels(&self) -> Vec<u8> {
        let mut bytes = vec![0u8; (4 * SIDE * SIDE) as usize];
        let region = Region::new_2d(0, 0, SIDE, SIDE);
        (self.target.get_bytes(&mut bytes, 4 * SIDE, 0, region, 0, 0)).unwrap();
        bytes
    }
}

/// One shape's draws of one binding and count: the positions, the argument table (the entry
/// shape's), and the indirect command buffer that holds the same draws.
struct Draws<'s> {
    scene: &'s Scene,
    shape: Shape,
    positions: Vec<Buffer>,
    /// Where position `i` lies: its buffer's index among `positions` and its offset there.
    at: Vec<(usize, u64)>,
    table: Option<Buffer>,
    icb: IndirectCommandBuffer,
    /// `rd_encode`'s table: the buffer's resource ID.
    commands: Buffer,
}

impl<'s> Draws<'s> {
    fn new(scene: &'s Scene, shape: Shape, own: bool, n: u64) -> Draws<'s> {
        let device = &scene.device;
        let position = |i: u64| -> Vec<u8> {
            [
                (i % 71) as f32 / 36.0 - 1.0,
                (i / 71) as f32 / 36.0 - 1.0,
                0.0,
                1.0,
            ]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect()
        };
        let (positions, at): (Vec<Buffer>, Vec<(usize, u64)>) = match own {
            true => (
                (0..n)
                    .map(|i| buffer_with(device, &position(i)).unwrap())
                    .collect(),
                (0..n as usize).map(|i| (i, 0)).collect(),
            ),
            false => (
                vec![buffer_with(device, &(0..n).flat_map(position).collect::<Vec<_>>()).unwrap()],
                (0..n).map(|i| (0, 16 * i)).collect(),
            ),
        };
        let table = matches!(shape, Shape::Entry).then(|| {
            let texture = scene.sampled.gpu_resource_id().0;
            let sampler = scene.sampler.gpu_resource_id().0;
            let bytes: Vec<u8> = at
                .iter()
                .flat_map(|&(buffer, offset)| {
                    let address = positions[buffer].gpu_address().0 + offset;
                    [address, texture, sampler]
                })
                .flat_map(u64::to_le_bytes)
                .collect();
            buffer_with(device, &bytes).unwrap()
        });
        let mut descriptor = IndirectCommandBufferDescriptor::new();
        descriptor.command_types = IndirectCommandType::DRAW;
        descriptor.max_vertex_buffer_bind_count = 1;
        descriptor.max_fragment_buffer_bind_count = 1;
        let icb = device
            .new_indirect_command_buffer(&descriptor, n, ResourceOptions::shared())
            .unwrap();
        let commands = buffer_with(device, &icb.gpu_resource_id().0.to_le_bytes()).unwrap();
        let draws = Draws {
            scene,
            shape,
            positions,
            at,
            table,
            icb,
            commands,
        };
        for i in 0..n {
            let command = draws.icb.indirect_render_command_at(i).unwrap();
            command.set_render_pipeline_state(draws.state()).unwrap();
            let (buffer, offset) = draws.binding(i);
            command.set_vertex_buffer(Some(buffer), offset, 0).unwrap();
            if let Shape::Entry = shape {
                command
                    .set_fragment_buffer(Some(buffer), offset, 0)
                    .unwrap();
            }
            command
                .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
                .unwrap();
        }
        draws
    }

    /// The draws' state.
    fn state(&self) -> &'s RenderPipelineState {
        match self.shape {
            Shape::Slot => &self.scene.slot,
            Shape::Entry => &self.scene.entry,
        }
    }

    /// The buffer and offset draw `i` binds.
    fn binding(&self, i: u64) -> (&Buffer, u64) {
        match &self.table {
            Some(table) => (table, ENTRY * i),
            None => {
                let (buffer, offset) = self.at[i as usize];
                (&self.positions[buffer], offset)
            }
        }
    }

    /// Make the pass's resources resident (the module docs).
    fn resident(&self, encoder: &mut RenderCommandEncoder) {
        let positions: Vec<ResourceRef> = self.positions.iter().map(ResourceRef::Buffer).collect();
        encoder
            .use_resources(&positions, ResourceUsage::READ, RenderStages::VERTEX)
            .unwrap();
        if let Some(table) = &self.table {
            let both = RenderStages::VERTEX | RenderStages::FRAGMENT;
            encoder
                .use_resource(table.as_resource(), ResourceUsage::READ, both)
                .unwrap();
            encoder
                .use_resource(
                    self.scene.sampled.as_resource(),
                    ResourceUsage::READ,
                    RenderStages::FRAGMENT,
                )
                .unwrap();
        }
    }

    /// One frame of `path`, committed and waited for: its time from encoding to completion, in
    /// milliseconds.
    fn frame(&self, path: Path) -> f64 {
        let scene = self.scene;
        let started = Instant::now();
        let command_buffer = scene.queue.command_buffer().unwrap();
        if let Path::Write | Path::Encode = path {
            let mut writing = command_buffer.compute_command_encoder().unwrap();
            match path {
                Path::Encode => {
                    writing.set_compute_pipeline_state(&scene.encode).unwrap();
                    writing.set_buffer(Some(&self.commands), 0, 1).unwrap();
                }
                _ => writing.set_compute_pipeline_state(&scene.touch).unwrap(),
            }
            writing.set_buffer(Some(&scene.count), 0, 0).unwrap();
            writing
                .use_resource(self.icb.as_resource(), ResourceUsage::WRITE)
                .unwrap();
            let one = Size::new(1, 1, 1);
            writing.dispatch_threads(one, one).unwrap();
            writing.end_encoding().unwrap();
        }
        let mut pass = RenderPassDescriptor::new();
        let color = &mut pass.color_attachments[0];
        color.texture = Some(scene.target.clone());
        color.load_action = LoadAction::Clear;
        color.store_action = StoreAction::Store;
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        self.resident(&mut encoder);
        let n = self.at.len() as u64;
        match path {
            Path::Encoder => {
                encoder.set_render_pipeline_state(self.state()).unwrap();
                for i in 0..n {
                    let (buffer, offset) = self.binding(i);
                    encoder.set_vertex_buffer(Some(buffer), offset, 0).unwrap();
                    if let Shape::Entry = self.shape {
                        encoder
                            .set_fragment_buffer(Some(buffer), offset, 0)
                            .unwrap();
                    }
                    encoder
                        .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
                        .unwrap();
                }
            }
            Path::Read | Path::Write | Path::Encode => {
                encoder.execute_commands_in_buffer(&self.icb, 0..n).unwrap();
            }
        }
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        started.elapsed().as_secs_f64() * 1e3
    }
}

/// Each counter `after` holds above `before`, by name, from their `Debug` forms (`WorkCounts`
/// derives it, a `name: value` a field).
fn moved(before: &WorkCounts, after: &WorkCounts) -> Vec<(String, i128)> {
    let fields = |counts: &WorkCounts| -> Vec<(String, u64)> {
        let text = format!("{counts:?}");
        let inner = text.split_once('{').map_or("", |(_, rest)| rest);
        (inner.trim_end_matches('}').split(','))
            .filter_map(|field| {
                let (name, value) = field.split_once(':')?;
                Some((name.trim().to_owned(), value.trim().parse().ok()?))
            })
            .collect()
    };
    (fields(before).into_iter().zip(fields(after)))
        .filter(|((_, a), (_, b))| a != b)
        .map(|((name, a), (_, b))| (name, i128::from(b) - i128::from(a)))
        .collect()
}

/// What a frame of `n` draws of `shape` on `path` may cost of the resources made resident, at
/// most, whatever their count: the ranges its root tables note of their own resident sets, the
/// draw's own binding in each of the two stages that read through it (the entry shape's; the
/// slot shape's functions read no pointer, so their draws upload no root table) and, on the
/// `encode` path, `rd_encode`'s own two buffers, as it reads its table through a pointer; one index
/// of the resident buffers by address for the execution's commands to resolve in (none for the
/// encoder's draws, which resolve nothing), and one whole comparison of what was made resident
/// with the list the execution's kept chunks were recorded under (none where it keeps none).
/// Each frame's `moved` counters, of `frames` frames.
fn assert_bounded(
    case: &str,
    (shape, path, n): (Shape, Path, u64),
    moved: &[(String, i128)],
    frames: usize,
) {
    let count = |name: &str| count(moved, name);
    let frames = frames as i128;
    let kernel = match path {
        Path::Encode => 2,
        Path::Encoder | Path::Read | Path::Write => 0,
    };
    let ranges = match shape {
        Shape::Slot => 0,
        Shape::Entry => 2 * i128::from(n),
    } + kernel;
    let noted = count("root_table_ranges_noted");
    assert!(
        noted <= ranges * frames,
        "{case} N {n}: {noted} root-table ranges noted"
    );
    let indexes = match path {
        Path::Encoder => 0,
        Path::Read | Path::Write | Path::Encode => 1,
    };
    let made = count("resident_address_indexes_made");
    assert!(
        made <= indexes * frames,
        "{case} N {n}: {made} address indexes made"
    );
    // One execution a frame, whose kept chunks hold one list of what was made resident.
    let compared = count("kept_resident_keys_compared");
    assert!(
        compared <= frames,
        "{case} N {n}: {compared} resident lists compared whole"
    );
}

/// FNV-1a of `bytes`.
fn hashed(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, &byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x100_0000_01b3)
    })
}

/// Every path of each shape, with a position buffer a draw and with one buffer at offsets, draws
/// what the encoder's own draws of positions in one buffer draw: 300 draws, past a chunk of an
/// indirect command buffer's records, each resolving its own buffer's address among the many made
/// resident (the `encode` path's host step resolves each record's binding there), at a cost of
/// the resources made resident that their count does not set (`assert_bounded`).
#[test]
fn each_path_draws_what_the_encoder_draws() {
    on_device(|device, _| {
        let scene = Scene::new(device);
        for shape in [Shape::Slot, Shape::Entry] {
            let one = Draws::new(&scene, shape, false, 300);
            one.frame(Path::Encoder);
            let expected = scene.pixels();
            let drawn = expected.chunks(4).filter(|p| p[3] != 0).count();
            assert!(drawn > 300, "{shape:?}: {drawn} pixels drawn");
            for own in [true, false] {
                let draws = Draws::new(&scene, shape, own, 300);
                // Three frames a path: an indirect command buffer's first execution keeps
                // nothing, its second records its chunks to keep, and its third runs them. Once
                // a kernel that may encode met the buffer, it stays written by the GPU.
                for path in PATHS.into_iter().chain([Path::Encode]) {
                    for frame in 0..3 {
                        let before = device.work_counts();
                        draws.frame(path);
                        let moved = moved(&before, &device.work_counts());
                        let case = format!("{shape:?} own {own} {path:?} frame {frame}");
                        assert_bounded(&case, (shape, path, 300), &moved, 1);
                        assert!(scene.pixels() == expected, "{case}");
                    }
                }
            }
        }
    });
}

/// Dispatch `kernel` once, with the count and `table` bound, in a command buffer of its own that
/// makes no indirect command buffer resident: so no later frame's dispatch waits for its pipeline,
/// which a host step of its own would.
fn warm(scene: &Scene, kernel: &ComputePipelineState, table: &Buffer) {
    let command_buffer = scene.queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(kernel).unwrap();
    encoder.set_buffer(Some(&scene.count), 0, 0).unwrap();
    encoder.set_buffer(Some(table), 0, 1).unwrap();
    let one = Size::new(1, 1, 1);
    encoder.dispatch_threads(one, one).unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
}

/// How many of the counter `name` `moved` holds.
fn count(moved: &[(String, i128)], name: &str) -> i128 {
    let found = moved.iter().find(|(field, _)| field == name);
    found.map_or(0, |&(_, count)| count)
}

/// The `write` path, as a page's frame encodes it (a kernel that counts made the buffer resident
/// for writing), leaves the buffer unmarked: `rd_touch` calls no indirect-command operation, so
/// no host step records an execution, the second execution records its chunks to keep and the
/// third runs them, as on the `read` path, whose pixels it draws; each commit counts the buffer it
/// left unmarked.
#[test]
fn a_kernel_that_cannot_encode_leaves_the_buffer_unmarked() {
    on_device(|device, _| {
        let scene = Scene::new(device);
        let read = Draws::new(&scene, Shape::Slot, false, 300);
        warm(&scene, &scene.touch, &read.commands);
        read.frame(Path::Read);
        let expected = scene.pixels();
        let draws = Draws::new(&scene, Shape::Slot, false, 300);
        for frame in 0..3 {
            let before = device.work_counts();
            draws.frame(Path::Write);
            let moved = moved(&before, &device.work_counts());
            let listed: Vec<String> = (moved.iter())
                .filter(|(field, _)| !field.ends_with("nanos"))
                .map(|(field, count)| format!("{field}={count}"))
                .collect();
            let steps = count(&moved, "host_steps_recorded");
            assert_eq!(steps, 0, "frame {frame}: {}", listed.join(" "));
            let unmarked = count(&moved, "indirect_write_residencies_unmarked");
            assert_eq!(unmarked, 1, "frame {frame}");
            // 300 commands are two chunks of 256.
            let kept = if frame == 2 { 2 } else { 0 };
            assert_eq!(count(&moved, "indirect_chunks_kept"), kept, "frame {frame}");
            assert!(!draws.icb.gpu_written(), "frame {frame}");
            assert!(scene.pixels() == expected, "frame {frame}");
        }
    });
}

/// The `encode` path marks the buffer at its commit: `rd_encode` may reset a command, so a host
/// step records each execution, from the first, with no chunk kept, and draws what the `read`
/// path draws (it resets none); the host-replay notice names the kernel and its operation.
#[test]
fn a_kernel_that_may_encode_marks_the_buffer() {
    on_device(|device, messages| {
        let scene = Scene::new(device);
        let read = Draws::new(&scene, Shape::Slot, false, 300);
        warm(&scene, &scene.encode, &read.commands);
        read.frame(Path::Read);
        let expected = scene.pixels();
        let draws = Draws::new(&scene, Shape::Slot, false, 300);
        for frame in 0..3 {
            let before = device.work_counts();
            draws.frame(Path::Encode);
            let moved = moved(&before, &device.work_counts());
            assert!(count(&moved, "host_steps_recorded") > 0, "frame {frame}");
            let unmarked = count(&moved, "indirect_write_residencies_unmarked");
            assert_eq!(unmarked, 0, "frame {frame}");
            assert_eq!(count(&moved, "indirect_chunks_kept"), 0, "frame {frame}");
            assert!(draws.icb.gpu_written(), "frame {frame}");
            assert!(scene.pixels() == expected, "frame {frame}");
        }
        let notices = messages.take_notices();
        let reason = (notices.iter())
            .filter_map(|message| message.degradation.as_ref())
            .find(|notice| notice.degradation == Degradation::HostReplayedIndirectCommands)
            .map(|notice| notice.reason.clone())
            .unwrap_or_default();
        assert!(
            reason.contains("rd_encode") && reason.contains("(reset)"),
            "{reason}"
        );
    });
}

#[test]
#[ignore = "a measurement, not a check: run by name with --run-ignored all"]
fn resident_draws() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
        options.layers.clear();
        // A consumer's device keeps no execution record.
        options.record_executions = false;
    })
    .unwrap();
    let name = device.name().to_owned();
    let scene = Scene::new(&device);
    let frames = frames();
    let only = std::env::var(ONLY_VAR).ok();
    let dump = std::env::var(DUMP_VAR).is_ok_and(|dump| dump == "1");
    for shape in [Shape::Slot, Shape::Entry] {
        for own in [true, false] {
            let binding = if own { "own" } else { "one" };
            for n in counts() {
                let draws = Draws::new(&scene, shape, own, n);
                for path in PATHS {
                    let case = format!("{shape:?} {binding} {path:?}").to_lowercase();
                    if only.as_deref().is_some_and(|only| !case.contains(only)) {
                        continue;
                    }
                    draws.frame(path);
                    let before = device.work_counts();
                    let mut times: Vec<f64> = (0..frames).map(|_| draws.frame(path)).collect();
                    let after = device.work_counts();
                    assert_bounded(&case, (shape, path, n), &moved(&before, &after), frames);
                    times.sort_by(f64::total_cmp);
                    let median = times[times.len() / 2];
                    let per = |name: &str| {
                        let moved = moved(&before, &after);
                        let count = moved.iter().find(|(field, _)| field == name);
                        count.map_or(0.0, |&(_, count)| count as f64 / frames as f64)
                    };
                    eprintln!(
                        "RESIDENT-DRAWS {name}: {case} N {n}: frame {median:.2} ms median, \
                         {:.2} largest, {:.2} us a draw; a frame's host steps {:.0} taking \
                         {:.2} ms, records read {:.0}, chunks kept {:.0}, heap tables worked out \
                         {:.0}, resident resources joined {:.0}, root-table ranges noted {:.0}, \
                         address indexes made {:.0}",
                        times[times.len() - 1],
                        median * 1e3 / n as f64,
                        per("host_steps_recorded"),
                        per("host_step_nanos") / 1e6,
                        per("indirect_records_read"),
                        per("indirect_chunks_kept"),
                        per("heap_tables_worked_out"),
                        per("resident_resources_joined"),
                        per("root_table_ranges_noted"),
                        per("resident_address_indexes_made"),
                    );
                    if dump {
                        let moved = moved(&before, &after);
                        let listed: Vec<String> = (moved.iter())
                            .filter(|(field, _)| !field.ends_with("nanos"))
                            .map(|(field, count)| format!("{field}={count}"))
                            .collect();
                        eprintln!(
                            "RESIDENT-DRAWS-COUNTS {case} N {n}: pixels {:016x}; {}",
                            hashed(&scene.pixels()),
                            listed.join(" ")
                        );
                    }
                }
            }
        }
    }
}
