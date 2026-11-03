//! Executions of commands the CPU encoded that run their chunks' kept recordings again (the
//! runtime's `kept_commands.rs`, `render/executions/kept.rs`), on the device the cfg selects,
//! under the validation layer. `icb_merged_draws`' vertex function places vertex `vid` at
//! `xy[vid]`, one pixel, and carries its command's mark (vertex buffer 0) and `vid` flat into a
//! `RGBA32Uint` target. Command `c` draws vertex `c`, its mark `100 + c`, at pixel `c` of the
//! target, so every command's draw is a pixel of its own. They draw as Metal's draws of the
//! commands do, and:
//!
//! - a range's first execution keeps nothing, and its second records each chunk to keep;
//! - an execution of commands no write reached since an earlier one recorded them to keep runs
//!   that recording again, chunk by chunk, reading no record: in another command buffer, into
//!   another target, and twice in one pass;
//! - a command encoded again or reset records its chunk again, and only it;
//! - a write of the GPU's (a blit's reset) runs no kept recording;
//! - another set of resident resources, or another range, records its chunks again;
//! - an execution with a check keeps its chunks as an unchecked one does where the check's needs
//!   rule out what the records may hold, and keeps nothing where they do not.
//!
//! And so do compute commands: `icb_replay_template`, command `c` binding its own parameters and
//! output windows, so every command's dispatch writes words of its own.

use super::degrade_notices::on_device;
use super::indirect_host_steps::complete;
use super::render_executions::icb;
use super::{buffer_with, read_buffer};
use crate::air_library::AirLibrary;
use metal2vulkan::metal::{IndirectCommandType, PixelFormat, PrimitiveType, RenderStages};
use metal2vulkan::metal::{ResourceUsage, TextureUsage};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, ComputePipelineDescriptor, RenderPipelineDescriptor, StageSource,
};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    Buffer, ClearColor, CommandQueue, ComputePipelineState, Device, IndirectCommandBuffer,
    IndirectCommandBufferDescriptor, IndirectCommandCheck, IndirectCommandCheckNeeds, LoadAction,
    PipelineOptions, Region, RenderPassDescriptor, RenderPipelineState, ResourceOptions,
    ResourceRef, Size, StoreAction, Texture, TextureDescriptor,
};
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// The target's width and height.
const W: u64 = 64;
const H: u64 = 48;
/// The commands: three chunks of the runtime's 256, the last a part.
const N: u64 = 600;

/// An execution's check: what it looks for, and the count of the commands it was called with.
type Checked<'c> = (IndirectCommandCheckNeeds, &'c Arc<AtomicU64>);

/// A check of `checked`'s needs that counts each command it is called with into `checked`'s
/// count.
fn counting((needs, calls): Checked<'_>) -> (IndirectCommandCheck, IndirectCommandCheckNeeds) {
    let calls = Arc::clone(calls);
    let check: IndirectCommandCheck = Box::new(move |_, _| {
        calls.fetch_add(1, Ordering::Relaxed);
    });
    (check, needs)
}

/// What the executions draw with: the commands' state, their buffers, the target and a queue.
struct Scene {
    state: RenderPipelineState,
    marks: Buffer,
    frame: Buffer,
    xy: Buffer,
    queue: CommandQueue,
}

impl Scene {
    fn new(device: &Device) -> Scene {
        let bitcode = llvm_assemble(&engine_dir().join("icb_merged_draws.ll")).unwrap();
        let library = device.new_library_with_air(&bitcode).unwrap();
        let function = |name: &str| {
            StageSource::Function(
                library
                    .new_function_with_name(name)
                    .unwrap()
                    .function_descriptor(),
            )
        };
        let mut descriptor = RenderPipelineDescriptor::new(function("icb_merged_vertex"));
        descriptor.fragment = Some(function("icb_merged_fragment"));
        descriptor.color_attachments[0] =
            Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA32Uint));
        descriptor.support_indirect_command_buffers = true;
        let state = device
            .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
            .unwrap();
        let words = |values: Vec<u32>| -> Vec<u8> {
            values.into_iter().flat_map(u32::to_le_bytes).collect()
        };
        let floats = |values: Vec<f32>| -> Vec<u8> {
            values.into_iter().flat_map(f32::to_le_bytes).collect()
        };
        let pixels = (0..W * H).flat_map(|p| [(p % W) as f32 + 0.5, (p / W) as f32 + 0.5]);
        Scene {
            state,
            marks: buffer_with(device, &words((0..W * H).map(|c| 100 + c as u32).collect()))
                .unwrap(),
            frame: buffer_with(device, &floats(vec![W as f32, H as f32, 0.0, 0.0])).unwrap(),
            xy: buffer_with(device, &floats(pixels.collect())).unwrap(),
            queue: device.new_command_queue().unwrap(),
        }
    }

    /// A buffer of [`N`] commands that inherit the encoder's state, each written
    /// ([`write`](Self::write)).
    fn icb(&self, device: &Device) -> IndirectCommandBuffer {
        let mut descriptor = IndirectCommandBufferDescriptor::new();
        descriptor.command_types = IndirectCommandType::DRAW;
        descriptor.inherit_pipeline_state = true;
        descriptor.max_vertex_buffer_bind_count = 3;
        descriptor.max_fragment_buffer_bind_count = 0;
        let icb = icb(device, &descriptor, N);
        for c in 0..N {
            self.write(&icb, c, c);
        }
        icb
    }

    /// Write command `c` of `icb`: its mark, the frame and the pixels as vertex buffers 0 to 2,
    /// and a draw of vertex `vertex` alone.
    fn write(&self, icb: &IndirectCommandBuffer, c: u64, vertex: u64) {
        let command = icb.indirect_render_command_at(c).unwrap();
        command
            .set_vertex_buffer(Some(&self.marks), 4 * c, 0)
            .unwrap();
        command.set_vertex_buffer(Some(&self.frame), 0, 1).unwrap();
        command.set_vertex_buffer(Some(&self.xy), 0, 2).unwrap();
        command
            .draw_primitives(PrimitiveType::Point, vertex, 1, 1, 0)
            .unwrap();
    }

    /// A target: `RGBA32Uint`, `W` by `H`, `Shared`, a render target.
    fn target(device: &Device) -> Texture {
        let mut target = TextureDescriptor::texture_2d(PixelFormat::RGBA32Uint, W, H, false);
        target.usage = TextureUsage::RENDER_TARGET;
        target.resource_options = ResourceOptions::shared();
        device.new_texture(&target).unwrap()
    }

    /// In a command buffer of its own, committed and waited for: a pass over `target`, cleared,
    /// that makes the commands' buffers and `extra` resident and executes `range` of `icb`
    /// `times` times.
    fn run(
        &self,
        target: &Texture,
        icb: &IndirectCommandBuffer,
        (range, times): (Range<u64>, usize),
        extra: Option<&Buffer>,
    ) {
        self.run_with(target, icb, (range, times), extra, None);
    }

    /// [`run`](Self::run) of `range` once, the execution given a check of `checked`'s needs
    /// that counts the commands it is called with.
    fn run_checked(
        &self,
        target: &Texture,
        icb: &IndirectCommandBuffer,
        range: Range<u64>,
        checked: Checked<'_>,
    ) {
        self.run_with(target, icb, (range, 1), None, Some(checked));
    }

    /// [`run`](Self::run), each execution given a check of `checked`'s where there is one.
    fn run_with(
        &self,
        target: &Texture,
        icb: &IndirectCommandBuffer,
        (range, times): (Range<u64>, usize),
        extra: Option<&Buffer>,
        checked: Option<Checked<'_>>,
    ) {
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut pass = RenderPassDescriptor::new();
        let color = &mut pass.color_attachments[0];
        color.texture = Some(target.clone());
        color.load_action = LoadAction::Clear;
        color.store_action = StoreAction::Store;
        color.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encoder.set_render_pipeline_state(&self.state).unwrap();
        for buffer in [&self.marks, &self.frame, &self.xy]
            .into_iter()
            .chain(extra)
        {
            encoder
                .use_resource(
                    buffer.as_resource(),
                    ResourceUsage::READ,
                    RenderStages::VERTEX,
                )
                .unwrap();
        }
        for _ in 0..times {
            match checked {
                None => encoder.execute_commands_in_buffer(icb, range.clone()),
                Some(checked) => {
                    let (check, needs) = counting(checked);
                    encoder.execute_commands_in_buffer_checked(icb, range.clone(), check, needs)
                }
            }
            .unwrap();
        }
        encoder.end_encoding().unwrap();
        complete(&command_buffer);
    }
}

/// `target`'s pixels' marks and vertices, pixel by pixel.
fn pixels(target: &Texture) -> Vec<[u32; 2]> {
    let mut bytes = vec![0u8; (16 * W * H) as usize];
    target
        .get_bytes(&mut bytes, 16 * W, 0, Region::new_2d(0, 0, W, H), 0, 0)
        .unwrap();
    let word = |p: &[u8; 16], i: usize| u32::from_le_bytes(p[4 * i..4 * i + 4].try_into().unwrap());
    let pixels = bytes.as_chunks::<16>().0.iter();
    pixels.map(|p| [word(p, 0), word(p, 1)]).collect()
}

/// What Metal's draws of commands `drawn` leave: command `c`'s mark at the pixel of the vertex
/// it draws, `vertex(c)`.
fn drawn(drawn: impl IntoIterator<Item = u64>, vertex: impl Fn(u64) -> u64) -> Vec<[u32; 2]> {
    let mut out = vec![[0; 2]; (W * H) as usize];
    for c in drawn {
        let v = vertex(c);
        out[v as usize] = [100 + c as u32, v as u32];
    }
    out
}

/// Across `run`: the records the host read, the chunks run from kept recordings, the commands
/// those held, and the chunks recorded to keep, each in a context of its own.
fn counted(device: &Device, run: impl FnOnce()) -> [u64; 4] {
    let before = device.work_counts();
    run();
    let after = device.work_counts();
    [
        after.indirect_records_read - before.indirect_records_read,
        after.indirect_chunks_kept - before.indirect_chunks_kept,
        after.indirect_commands_kept - before.indirect_commands_kept,
        after.kept_contexts_made - before.kept_contexts_made,
    ]
}

/// A range's first execution reads every record and keeps nothing; its second records each
/// chunk to keep; and an execution of commands no write reached since one recorded them to keep
/// runs that recording again, reading no record: in another command buffer, into another target,
/// and twice in one pass.
#[test]
fn unchanged_commands_run_their_kept_recordings() {
    on_device(|device, _| {
        let scene = Scene::new(device);
        let icb = scene.icb(device);
        let (a, b) = (Scene::target(device), Scene::target(device));
        let all = drawn(0..N, |c| c);
        let first = counted(device, || scene.run(&a, &icb, (0..N, 1), None));
        assert_eq!(first, [N, 0, 0, 0], "the first execution keeps nothing");
        assert_eq!(pixels(&a), all);
        let second = counted(device, || scene.run(&a, &icb, (0..N, 1), None));
        assert_eq!(
            second,
            [N, 0, 0, 3],
            "the second records three chunks to keep"
        );
        assert_eq!(pixels(&a), all);
        let again = counted(device, || scene.run(&a, &icb, (0..N, 1), None));
        assert_eq!(again, [0, 3, N, 0], "three chunks kept");
        assert_eq!(pixels(&a), all);
        let other = counted(device, || scene.run(&b, &icb, (0..N, 1), None));
        assert_eq!(other, [0, 3, N, 0], "kept into another target");
        assert_eq!(pixels(&b), all);
        let twice = counted(device, || scene.run(&a, &icb, (0..N, 2), None));
        assert_eq!(twice, [0, 6, 2 * N, 0], "kept twice in one pass");
        assert_eq!(pixels(&a), all);
    });
}

/// A command encoded again, or reset, records its chunk again, and only it, keeping nothing of
/// it at that first execution since the write; the next records it to keep, and the one after
/// runs every chunk's kept recording again.
#[test]
fn a_command_written_again_records_its_chunk_alone() {
    on_device(|device, _| {
        let scene = Scene::new(device);
        let icb = scene.icb(device);
        let target = Scene::target(device);
        scene.run(&target, &icb, (0..N, 1), None);
        scene.run(&target, &icb, (0..N, 1), None);
        // Command 300 draws vertex 1000 now; command 520 is reset, and draws nothing.
        scene.write(&icb, 300, 1000);
        icb.reset_with_range(520..521).unwrap();
        let vertex = |c| if c == 300 { 1000 } else { c };
        let expected = drawn((0..N).filter(|&c| c != 520), vertex);
        let written = counted(device, || scene.run(&target, &icb, (0..N, 1), None));
        let recorded = 256 + (N - 512);
        assert_eq!(
            written,
            [recorded, 1, 256, 0],
            "chunks 1 and 2 recorded again"
        );
        assert_eq!(pixels(&target), expected);
        let kept = counted(device, || scene.run(&target, &icb, (0..N, 1), None));
        assert_eq!(kept, [recorded, 1, 256, 2], "and recorded to keep");
        assert_eq!(pixels(&target), expected);
        let kept = counted(device, || scene.run(&target, &icb, (0..N, 1), None));
        assert_eq!(kept, [0, 3, N, 0]);
        assert_eq!(pixels(&target), expected);
    });
}

/// A write of the GPU's into the buffer (a blit's reset of commands of every chunk) runs no kept
/// recording: the reset commands draw nothing.
#[test]
fn a_gpu_write_runs_no_kept_recording() {
    on_device(|device, _| {
        let scene = Scene::new(device);
        let icb = scene.icb(device);
        let target = Scene::target(device);
        scene.run(&target, &icb, (0..N, 1), None);
        scene.run(&target, &icb, (0..N, 1), None);
        let command_buffer = scene.queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.reset_commands_in_buffer(&icb, 200..550).unwrap();
        blit.end_encoding().unwrap();
        complete(&command_buffer);
        let [_, chunks, ..] = counted(device, || scene.run(&target, &icb, (0..N, 1), None));
        assert_eq!(chunks, 0, "no chunk kept after the GPU's write");
        let expected = drawn((0..200).chain(550..N), |c| c);
        assert_eq!(pixels(&target), expected);
    });
}

/// Another set of resident resources records the chunks again, and so does another range; each
/// runs its own recording again after.
#[test]
fn another_residency_or_range_records_its_chunks_again() {
    on_device(|device, _| {
        let scene = Scene::new(device);
        let icb = scene.icb(device);
        let target = Scene::target(device);
        let extra = buffer_with(device, &[0; 16]).unwrap();
        let all = drawn(0..N, |c| c);
        scene.run(&target, &icb, (0..N, 1), None);
        scene.run(&target, &icb, (0..N, 1), None);
        let resident = counted(device, || scene.run(&target, &icb, (0..N, 1), Some(&extra)));
        assert_eq!(resident, [N, 0, 0, 3], "another residency records again");
        assert_eq!(pixels(&target), all);
        let again = counted(device, || scene.run(&target, &icb, (0..N, 1), Some(&extra)));
        assert_eq!(again, [0, 3, N, 0]);
        assert_eq!(pixels(&target), all);
        // Commands 100..400: chunk 0 from 100, and chunk 1 to 400, both recorded again.
        let part = counted(device, || {
            scene.run(&target, &icb, (100..400, 1), Some(&extra))
        });
        assert_eq!(part, [300, 0, 0, 2], "another range records again");
        assert_eq!(pixels(&target), drawn(100..400, |c| c));
        let part = counted(device, || {
            scene.run(&target, &icb, (100..400, 1), Some(&extra))
        });
        assert_eq!(part, [0, 2, 300, 0]);
        assert_eq!(pixels(&target), drawn(100..400, |c| c));
        // The whole range again: chunk 2 kept from the first executions.
        let whole = counted(device, || scene.run(&target, &icb, (0..N, 1), Some(&extra)));
        assert_eq!(whole, [512, 1, N - 512, 2]);
        assert_eq!(pixels(&target), all);
    });
}

/// An execution with a check whose needs rule out what the records may hold (commands under
/// states that may encode indirect commands: the encoder's state, which the commands inherit,
/// may not) calls the check at no commit and keeps its chunks from its second execution, as an
/// unchecked one does (`WorkCounts::indirect_chunks_kept`, `kept_contexts_made`). One whose check
/// looks for draws of points, which the records hold, keeps nothing, and the check sees every
/// command at each commit.
#[test]
fn a_checked_execution_keeps_where_its_check_sees_nothing() {
    on_device(|device, _| {
        let scene = Scene::new(device);
        let icb = scene.icb(device);
        let target = Scene::target(device);
        let all = drawn(0..N, |c| c);
        let calls = Arc::new(AtomicU64::new(0));
        let run = |needs| {
            counted(device, || {
                scene.run_checked(&target, &icb, 0..N, (needs, &calls));
            })
        };
        let blind = IndirectCommandCheckNeeds::new(false, true);
        assert_eq!(
            run(blind),
            [N, 0, 0, 0],
            "the first execution keeps nothing"
        );
        assert_eq!(
            run(blind),
            [N, 0, 0, 3],
            "the second records three chunks to keep"
        );
        assert_eq!(run(blind), [0, 3, N, 0], "three chunks kept");
        assert_eq!(pixels(&target), all);
        assert_eq!(
            calls.load(Ordering::Relaxed),
            0,
            "a check with nothing to see"
        );
        let points = IndirectCommandCheckNeeds::new(true, false);
        for _ in 0..3 {
            assert_eq!(run(points), [N, 0, 0, 0], "a check that sees keeps nothing");
        }
        assert_eq!(pixels(&target), all);
        assert_eq!(
            calls.load(Ordering::Relaxed),
            3 * N,
            "the check sees every command at each commit"
        );
    });
}

/// What the compute executions dispatch with: the commands' state, each command's parameters
/// (16 bytes apart: command `c`'s are `c + 1` and `1000 c`), their output, and a queue.
/// Command `c`'s four threads write `(c + 1)(t + 1) + 1000 c` to word `t` of its 16-byte window
/// of the output.
struct Dispatches {
    state: ComputePipelineState,
    params: Buffer,
    out: Buffer,
    queue: CommandQueue,
}

impl Dispatches {
    fn new(device: &Device) -> Dispatches {
        let bitcode = llvm_assemble(&engine_dir().join("icb_replay_template.ll")).unwrap();
        let library = device.new_library_with_air(&bitcode).unwrap();
        let kernel = library
            .new_function_with_name("icb_replay_template")
            .unwrap();
        let mut descriptor =
            ComputePipelineDescriptor::new(StageSource::Function(kernel.function_descriptor()));
        descriptor.support_indirect_command_buffers = true;
        let state = device
            .new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
            .unwrap();
        let params: Vec<u8> = (0..N as u32)
            .flat_map(|c| [c + 1, 1000 * c, 0, 0])
            .flat_map(u32::to_le_bytes)
            .collect();
        Dispatches {
            state,
            params: buffer_with(device, &params).unwrap(),
            out: buffer_with(device, &vec![0; 16 * N as usize]).unwrap(),
            queue: device.new_command_queue().unwrap(),
        }
    }

    /// A buffer of [`N`] concurrent dispatch commands, each written
    /// ([`write`](Self::write)).
    fn icb(&self, device: &Device) -> IndirectCommandBuffer {
        let mut descriptor = IndirectCommandBufferDescriptor::new();
        descriptor.command_types = IndirectCommandType::CONCURRENT_DISPATCH;
        descriptor.max_kernel_buffer_bind_count = 2;
        descriptor.max_kernel_threadgroup_memory_bind_count = 1;
        let icb = icb(device, &descriptor, N);
        for c in 0..N {
            self.write(&icb, c, c);
        }
        icb
    }

    /// Write command `c` of `icb`: the state, command `params`'s parameters and command `c`'s
    /// output window, and one threadgroup of four threads.
    fn write(&self, icb: &IndirectCommandBuffer, c: u64, params: u64) {
        let command = icb.indirect_compute_command_at(c).unwrap();
        command.set_compute_pipeline_state(&self.state).unwrap();
        command
            .set_kernel_buffer(Some(&self.params), 16 * params, 0)
            .unwrap();
        command
            .set_kernel_buffer(Some(&self.out), 16 * c, 1)
            .unwrap();
        command.set_threadgroup_memory_length(16, 0).unwrap();
        command
            .concurrent_dispatch_threadgroups(Size::new(1, 1, 1), Size::new(4, 1, 1))
            .unwrap();
    }

    /// In a command buffer of its own, committed and waited for: the output zeroed, then a
    /// compute encoder that makes the buffers resident and executes `range` of `icb`.
    fn run(&self, icb: &IndirectCommandBuffer, range: Range<u64>) {
        self.run_with(icb, range, None);
    }

    /// [`run`](Self::run), the execution given a check of `checked`'s where there is one.
    fn run_with(
        &self,
        icb: &IndirectCommandBuffer,
        range: Range<u64>,
        checked: Option<Checked<'_>>,
    ) {
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.fill_buffer(&self.out, 0..16 * N, 0).unwrap();
        blit.end_encoding().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        let used = [
            ResourceRef::Buffer(&self.params),
            ResourceRef::Buffer(&self.out),
        ];
        encoder
            .use_resources(&used, ResourceUsage::READ.union(ResourceUsage::WRITE))
            .unwrap();
        match checked {
            None => encoder.execute_commands_in_buffer(icb, range),
            Some(checked) => {
                let (check, needs) = counting(checked);
                encoder.execute_commands_in_buffer_checked(icb, range, check, needs)
            }
        }
        .unwrap();
        encoder.end_encoding().unwrap();
        complete(&command_buffer);
    }

    /// The output's words.
    fn out(&self) -> Vec<u32> {
        let bytes = read_buffer(&self.out, 0, 16 * N).unwrap();
        let words = bytes.as_chunks::<4>().0.iter();
        words.map(|word| u32::from_le_bytes(*word)).collect()
    }
}

/// What Metal's dispatches of commands `ran` leave in the output: command `c`'s parameters
/// `params(c)`'s words in its window.
fn dispatched(ran: impl IntoIterator<Item = u64>, params: impl Fn(u64) -> u64) -> Vec<u32> {
    let mut out = vec![0; 4 * N as usize];
    for c in ran {
        let p = params(c) as u32;
        for t in 0..4u32 {
            out[(4 * c) as usize + t as usize] = (p + 1) * (t + 1) + 1000 * p;
        }
    }
    out
}

/// Compute commands keep nothing at their range's first execution, record each chunk to keep at
/// its second, and run that recording again from then on while no write reaches them, reading no
/// record; a command encoded again records its chunk again, and only it, keeping it from the
/// execution after; and after a blit's reset of commands, no kept recording runs.
#[test]
fn compute_commands_run_their_kept_recordings_until_written() {
    on_device(|device, _| {
        let work = Dispatches::new(device);
        let icb = work.icb(device);
        let all = dispatched(0..N, |c| c);
        let first = counted(device, || work.run(&icb, 0..N));
        assert_eq!(first, [N, 0, 0, 0], "the first execution keeps nothing");
        assert_eq!(work.out(), all);
        let second = counted(device, || work.run(&icb, 0..N));
        assert_eq!(
            second,
            [N, 0, 0, 3],
            "the second records three chunks to keep"
        );
        assert_eq!(work.out(), all);
        let again = counted(device, || work.run(&icb, 0..N));
        assert_eq!(again, [0, 3, N, 0], "three chunks kept");
        assert_eq!(work.out(), all);
        // Command 300 takes command 7's parameters now.
        work.write(&icb, 300, 7);
        let params = |c| if c == 300 { 7 } else { c };
        let written = counted(device, || work.run(&icb, 0..N));
        assert_eq!(written, [256, 2, N - 256, 0], "chunk 1 recorded again");
        assert_eq!(work.out(), dispatched(0..N, params));
        let kept = counted(device, || work.run(&icb, 0..N));
        assert_eq!(kept, [256, 2, N - 256, 1], "and recorded to keep");
        assert_eq!(work.out(), dispatched(0..N, params));
        let again = counted(device, || work.run(&icb, 0..N));
        assert_eq!(again, [0, 3, N, 0]);
        assert_eq!(work.out(), dispatched(0..N, params));
        let command_buffer = work.queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.reset_commands_in_buffer(&icb, 200..550).unwrap();
        blit.end_encoding().unwrap();
        complete(&command_buffer);
        let [_, chunks, ..] = counted(device, || work.run(&icb, 0..N));
        assert_eq!(chunks, 0, "no chunk kept after the GPU's write");
        assert_eq!(work.out(), dispatched((0..200).chain(550..N), params));
    });
}

/// A compute execution with a check of every kind, whose needs the records rule out (no draw,
/// and no state that may encode indirect commands), calls the check at no commit and keeps its
/// chunks from its second execution, as an unchecked one does.
#[test]
fn a_checked_compute_execution_keeps_where_its_check_sees_nothing() {
    on_device(|device, _| {
        let work = Dispatches::new(device);
        let icb = work.icb(device);
        let all = dispatched(0..N, |c| c);
        let calls = Arc::new(AtomicU64::new(0));
        let checked = (IndirectCommandCheckNeeds::ALL, &calls);
        let run = || counted(device, || work.run_with(&icb, 0..N, Some(checked)));
        assert_eq!(run(), [N, 0, 0, 0], "the first execution keeps nothing");
        assert_eq!(
            run(),
            [N, 0, 0, 3],
            "the second records three chunks to keep"
        );
        assert_eq!(run(), [0, 3, N, 0], "three chunks kept");
        assert_eq!(work.out(), all);
        assert_eq!(
            calls.load(Ordering::Relaxed),
            0,
            "a check with nothing to see"
        );
    });
}
