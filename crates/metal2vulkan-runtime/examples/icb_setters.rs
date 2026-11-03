//! Not a consumer's example: the CPU cost of an indirect command buffer's command setters, the
//! shape of a consumer that encodes every field of each command on the CPU, one thread, nothing
//! executing.
//!
//! ```sh
//! cargo run --release -p metal2vulkan-runtime --example icb_setters [-- [render|compute|all] [--repetitions N] [--calls] [--kept]]
//! ```
//!
//! - **render**: a buffer of [`COMMANDS`] commands of `DRAW` alone, 2 vertex buffers and 1
//!   fragment buffer, nothing inherited. Each command: `indirect_render_command_at`,
//!   `set_render_pipeline_state`, `set_vertex_buffer` twice, `set_fragment_buffer` and
//!   `draw_primitives`.
//! - **compute**: a buffer of [`COMMANDS`] commands of `CONCURRENT_DISPATCH`, 3 kernel buffers,
//!   nothing inherited. Each command: `indirect_compute_command_at`,
//!   `set_compute_pipeline_state`, `set_kernel_buffer` three times and
//!   `concurrent_dispatch_threadgroups`.
//!
//! Each repetition encodes every command of the buffer once, over the commands the repetition
//! before encoded (as a consumer encodes a buffer again each frame), and is timed as a whole: the
//! wall time and, on macOS, the calling thread's own instructions and cycles
//! (`thread_selfcounts`, libSystem's, readable unprivileged; other systems print the wall time
//! alone). Printed: the median over the repetitions of each a command, with the lowest and the
//! highest. With `--calls`, each call kind is also timed alone, a pass of that one call over
//! every command of the encoded buffer, a setter's cost; and the record format's own writes of
//! the same fields (`CommandRecord::write_field` into one record's bytes, no lock and no check),
//! the floor a setter's write costs.
//!
//! With `--kept`, the compute row also runs its encoding after executions that kept the
//! buffer's chunks' recordings (`kept_commands.rs`), as a consumer's frame executes the buffer and
//! then encodes it again: before each repetition, untimed, a compute encoder executes every
//! command twice, each in a command buffer committed and waited for, the second running every
//! chunk from the first's recordings (the printed `indirect_chunks_kept` of each second one). That
//! mode submits work to the GPU; the rest submits nothing.
//!
//! The pipeline states are the owned fixture's (`runtime_examples.ll`'s `colored_vertex`,
//! `colored_fragment` and `scale_and_offset`), assembled by `llvm-as` from `PATH` or
//! `METAL2VULKAN_LLVM_AS`.

use metal2vulkan::bind::indirect_commands::{
    Action, BufferBinding, BufferStage, CommandRecord, RecordField, COMMAND_BYTES,
};
use metal2vulkan::metal::{IndirectCommandType, PixelFormat, PrimitiveType, ResourceUsage};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, ComputePipelineDescriptor, RenderPipelineDescriptor, StageSource,
};
use metal2vulkan_runtime::{
    Buffer, CommandBufferStatus, CommandQueue, ComputePipelineState, Device, IndirectCommandBuffer,
    IndirectCommandBufferDescriptor, Library, PipelineOptions, RenderPipelineState,
    ResourceOptions, ResourceRef, Size,
};
use std::hint::black_box;
use std::num::NonZeroU64;
use std::path::Path;
use std::time::{Duration, Instant};

/// The example's result: every runtime refusal is a typed `metal2vulkan_runtime::Error`.
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// The commands of each buffer, and of each repetition.
const COMMANDS: u32 = 16384;

/// The repetitions a row runs by default, after one untimed.
const REPETITIONS: usize = 30;

/// The calling thread's own counts so far: instructions and cycles, where the system counts them.
#[derive(Clone, Copy, Default)]
struct Counts {
    instructions: u64,
    cycles: u64,
}

#[cfg(target_os = "macos")]
fn counts() -> Option<Counts> {
    unsafe extern "C" {
        /// libSystem's per-thread counters; `THSC_CPI` (1) answers the instructions and the
        /// cycles, two `u64`s.
        fn thread_selfcounts(kind: i32, buffer: *mut u64, bytes: usize) -> i32;
    }
    let mut words = [0u64; 2];
    // SAFETY: the buffer holds the two words `THSC_CPI` writes.
    let answer = unsafe { thread_selfcounts(1, words.as_mut_ptr(), size_of_val(&words)) };
    (answer == 0).then_some(Counts {
        instructions: words[0],
        cycles: words[1],
    })
}

#[cfg(not(target_os = "macos"))]
fn counts() -> Option<Counts> {
    None
}

/// One repetition's measure: the wall time and the thread's counts, over `commands` commands.
#[derive(Clone, Copy)]
struct Sample {
    nanos: f64,
    counts: Option<Counts>,
}

/// Run `body` once untimed, then `repetitions` times, each measured.
fn measure(repetitions: usize, body: impl FnMut()) -> Vec<Sample> {
    measure_after(repetitions, || {}, body)
}

/// [`measure`], with `prepare` run untimed before each run of `body`.
fn measure_after(
    repetitions: usize,
    mut prepare: impl FnMut(),
    mut body: impl FnMut(),
) -> Vec<Sample> {
    prepare();
    body();
    (0..repetitions)
        .map(|_| {
            prepare();
            let (before, start) = (counts(), Instant::now());
            body();
            let nanos = start.elapsed().as_nanos() as f64;
            let after = counts();
            let counts = before.zip(after).map(|(before, after)| Counts {
                instructions: after.instructions - before.instructions,
                cycles: after.cycles - before.cycles,
            });
            Sample { nanos, counts }
        })
        .collect()
}

/// The median, the lowest and the highest of `values`.
fn spread(mut values: Vec<f64>) -> (f64, f64, f64) {
    values.sort_by(f64::total_cmp);
    let n = values.len();
    let median = match n % 2 {
        1 => values[n / 2],
        _ => (values[n / 2 - 1] + values[n / 2]) / 2.0,
    };
    (median, values[0], values[n - 1])
}

/// Print `row`'s samples, each of `per` units (commands, or calls).
fn report(row: &str, samples: &[Sample], per: f64) {
    let (ns, low, high) = spread(samples.iter().map(|s| s.nanos / per).collect());
    let mut line = format!("{row:<34} {ns:>9.1} ns [{low:.1}, {high:.1}]");
    if samples.iter().all(|s| s.counts.is_some()) {
        let of = |pick: fn(Counts) -> u64| {
            spread(
                samples
                    .iter()
                    .filter_map(|s| s.counts.map(|c| pick(c) as f64 / per))
                    .collect(),
            )
            .0
        };
        let (instructions, cycles) = (of(|c| c.instructions), of(|c| c.cycles));
        line += &format!("  {instructions:>9.1} instructions  {cycles:>8.1} cycles");
    }
    println!("{line}");
}

/// The owned fixture's library (the module docs).
fn library(device: &Device) -> Result<Library> {
    let ll = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/fixtures/public/examples/runtime_examples.ll");
    let bitcode = metal2vulkan_devtools::llvm_assemble(&ll)
        .map_err(|error| format!("llvm-as {}: {error:?}", ll.display()))?;
    let library = metal2vulkan::library::Library::from_air(&bitcode)?;
    Ok(device.new_library_with_compile_library(library))
}

/// The render row's state: `colored_vertex` and `colored_fragment`, for indirect command buffers.
fn render_state(device: &Device) -> Result<RenderPipelineState> {
    let library = library(device)?;
    let vertex = library.new_function_with_name("colored_vertex")?;
    let fragment = library.new_function_with_name("colored_fragment")?;
    let mut descriptor =
        RenderPipelineDescriptor::new(StageSource::Function(vertex.function_descriptor()));
    descriptor.fragment = Some(StageSource::Function(fragment.function_descriptor()));
    descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    descriptor.support_indirect_command_buffers = true;
    Ok(device.new_render_pipeline_state(&descriptor, &PipelineOptions::default())?)
}

/// The compute row's state: `scale_and_offset`, for indirect command buffers.
fn compute_state(device: &Device) -> Result<ComputePipelineState> {
    let function = library(device)?.new_function_with_name("scale_and_offset")?;
    let mut descriptor =
        ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    descriptor.support_indirect_command_buffers = true;
    Ok(device.new_compute_pipeline_state(&descriptor, &PipelineOptions::default())?)
}

/// The render row (the module docs).
fn render(device: &Device, repetitions: usize, calls: bool) -> Result<()> {
    let state = render_state(device)?;
    let mut descriptor = IndirectCommandBufferDescriptor::new();
    descriptor.command_types = IndirectCommandType::DRAW;
    descriptor.max_vertex_buffer_bind_count = 2;
    descriptor.max_fragment_buffer_bind_count = 1;
    let icb = device.new_indirect_command_buffer(
        &descriptor,
        u64::from(COMMANDS),
        ResourceOptions::shared(),
    )?;
    let (a, b) = (
        device.new_buffer(4096, ResourceOptions::shared())?,
        device.new_buffer(4096, ResourceOptions::shared())?,
    );
    let encode = |icb: &IndirectCommandBuffer| {
        for index in 0..COMMANDS {
            let command = icb.indirect_render_command_at(u64::from(index)).unwrap();
            command.set_render_pipeline_state(&state).unwrap();
            command
                .set_vertex_buffer(Some(&a), u64::from(index % 16) * 16, 0)
                .unwrap();
            command.set_vertex_buffer(Some(&b), 0, 1).unwrap();
            command.set_fragment_buffer(Some(&a), 256, 0).unwrap();
            command
                .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
                .unwrap();
        }
    };
    let per = f64::from(COMMANDS);
    report(
        "render command",
        &measure(repetitions, || encode(&icb)),
        per,
    );
    if !calls {
        return Ok(());
    }
    let commands: Vec<_> = (0..COMMANDS)
        .map(|index| icb.indirect_render_command_at(u64::from(index)).unwrap())
        .collect();
    let each = |call: &dyn Fn(&metal2vulkan_runtime::IndirectRenderCommand)| {
        measure(repetitions, || commands.iter().for_each(call))
    };
    let rows: [(&str, Vec<Sample>); 5] = [
        (
            "  indirect_render_command_at",
            measure(repetitions, || {
                for index in 0..COMMANDS {
                    black_box(icb.indirect_render_command_at(u64::from(index)).unwrap());
                }
            }),
        ),
        (
            "  set_render_pipeline_state",
            each(&|command| command.set_render_pipeline_state(&state).unwrap()),
        ),
        (
            "  set_vertex_buffer",
            each(&|command| command.set_vertex_buffer(Some(&b), 0, 1).unwrap()),
        ),
        (
            "  set_fragment_buffer",
            each(&|command| command.set_fragment_buffer(Some(&a), 256, 0).unwrap()),
        ),
        (
            "  draw_primitives",
            each(&|command| {
                command
                    .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
                    .unwrap()
            }),
        ),
    ];
    for (row, samples) in &rows {
        report(row, samples, per);
    }
    let fields = [
        RecordField::Pipeline(NonZeroU64::new(state.gpu_resource_id().0).unwrap()),
        buffer_field(BufferStage::Vertex, 0, &a, 16),
        buffer_field(BufferStage::Vertex, 1, &b, 0),
        buffer_field(BufferStage::Fragment, 0, &a, 256),
        RecordField::Action(Action::DrawPrimitives {
            primitive_type: 3,
            vertex_start: 0,
            vertex_count: 3,
            instance_count: 1,
            base_instance: 0,
        }),
    ];
    report(
        "  floor: the 5 fields' write_field",
        &floor(repetitions, &fields),
        per,
    );
    Ok(())
}

/// The compute row (the module docs).
fn compute(device: &Device, repetitions: usize, (calls, kept): (bool, bool)) -> Result<()> {
    let state = compute_state(device)?;
    let mut descriptor = IndirectCommandBufferDescriptor::new();
    descriptor.command_types = IndirectCommandType::CONCURRENT_DISPATCH;
    descriptor.max_kernel_buffer_bind_count = 3;
    let icb = device.new_indirect_command_buffer(
        &descriptor,
        u64::from(COMMANDS),
        ResourceOptions::shared(),
    )?;
    let buffers = [
        device.new_buffer(4096, ResourceOptions::shared())?,
        device.new_buffer(4096, ResourceOptions::shared())?,
        device.new_buffer(256, ResourceOptions::shared())?,
    ];
    let (threadgroups, threads) = (Size::new(4, 1, 1), Size::new(64, 1, 1));
    let encode = |icb: &IndirectCommandBuffer| {
        for index in 0..COMMANDS {
            let command = icb.indirect_compute_command_at(u64::from(index)).unwrap();
            command.set_compute_pipeline_state(&state).unwrap();
            command
                .set_kernel_buffer(Some(&buffers[0]), u64::from(index % 16) * 16, 0)
                .unwrap();
            command.set_kernel_buffer(Some(&buffers[1]), 0, 1).unwrap();
            command.set_kernel_buffer(Some(&buffers[2]), 0, 2).unwrap();
            command
                .concurrent_dispatch_threadgroups(threadgroups, threads)
                .unwrap();
        }
    };
    let per = f64::from(COMMANDS);
    report(
        "compute command",
        &measure(repetitions, || encode(&icb)),
        per,
    );
    let commands: Vec<_> = (0..COMMANDS)
        .map(|index| icb.indirect_compute_command_at(u64::from(index)).unwrap())
        .collect();
    if kept {
        let queue = device.new_command_queue()?;
        let chunks = std::cell::RefCell::new(Vec::new());
        // Twice: the second execution runs every chunk from the first's kept recordings.
        let prepare = || {
            execute(&queue, &icb, &buffers);
            let before = device.work_counts().indirect_chunks_kept;
            execute(&queue, &icb, &buffers);
            let ran = device.work_counts().indirect_chunks_kept - before;
            chunks.borrow_mut().push(ran);
        };
        let rows = [
            (
                "compute command, chunks kept",
                measure_after(repetitions, prepare, || encode(&icb)),
            ),
            (
                "  set_kernel_buffer, chunks kept",
                measure_after(repetitions, prepare, || {
                    for command in &commands {
                        command.set_kernel_buffer(Some(&buffers[1]), 0, 1).unwrap();
                    }
                }),
            ),
        ];
        for (row, samples) in &rows {
            report(row, samples, per);
        }
        let mut chunks = chunks.into_inner();
        chunks.dedup();
        println!("  (chunks each second execution ran from kept recordings: {chunks:?})");
    }
    if !calls {
        return Ok(());
    }
    let each = |call: &dyn Fn(&metal2vulkan_runtime::IndirectComputeCommand)| {
        measure(repetitions, || commands.iter().for_each(call))
    };
    let rows: [(&str, Vec<Sample>); 4] = [
        (
            "  indirect_compute_command_at",
            measure(repetitions, || {
                for index in 0..COMMANDS {
                    black_box(icb.indirect_compute_command_at(u64::from(index)).unwrap());
                }
            }),
        ),
        (
            "  set_compute_pipeline_state",
            each(&|command| command.set_compute_pipeline_state(&state).unwrap()),
        ),
        (
            "  set_kernel_buffer",
            each(&|command| command.set_kernel_buffer(Some(&buffers[1]), 0, 1).unwrap()),
        ),
        (
            "  concurrent_dispatch_threadgroups",
            each(&|command| {
                command
                    .concurrent_dispatch_threadgroups(threadgroups, threads)
                    .unwrap()
            }),
        ),
    ];
    for (row, samples) in &rows {
        report(row, samples, per);
    }
    let fields = [
        RecordField::Pipeline(NonZeroU64::new(state.gpu_resource_id().0).unwrap()),
        buffer_field(BufferStage::Kernel, 0, &buffers[0], 16),
        buffer_field(BufferStage::Kernel, 1, &buffers[1], 0),
        buffer_field(BufferStage::Kernel, 2, &buffers[2], 0),
        RecordField::Action(Action::DispatchThreadgroups {
            grid: [4, 1, 1],
            local_size: [64, 1, 1],
        }),
    ];
    report(
        "  floor: the 5 fields' write_field",
        &floor(repetitions, &fields),
        per,
    );
    Ok(())
}

/// Execute every command of `icb`, which dispatches over `buffers`, in a command buffer of
/// `queue`, committed and waited for.
fn execute(queue: &CommandQueue, icb: &IndirectCommandBuffer, buffers: &[Buffer]) {
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    for buffer in buffers {
        let usage = ResourceUsage::READ | ResourceUsage::WRITE;
        encoder
            .use_resource(ResourceRef::Buffer(buffer), usage)
            .unwrap();
    }
    encoder
        .execute_commands_in_buffer(icb, 0..u64::from(COMMANDS))
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer
        .wait_until_completed_for(Duration::from_secs(60))
        .unwrap();
    assert_eq!(
        command_buffer.status(),
        CommandBufferStatus::Completed,
        "{:?}",
        command_buffer.error()
    );
}

/// The field a buffer setter writes: `buffer` at `offset`, at slot `slot` of `stage`.
fn buffer_field(stage: BufferStage, slot: usize, buffer: &Buffer, offset: u64) -> RecordField {
    RecordField::Buffer {
        stage,
        slot,
        binding: Some(BufferBinding {
            address: buffer.gpu_address().0 + offset,
            attribute_stride: None,
        }),
    }
}

/// `fields` written into one record's bytes by the record format alone, [`COMMANDS`] times a
/// repetition: what the setters' writes cost without a lock, a check or the bookkeeping.
fn floor(repetitions: usize, fields: &[RecordField]) -> Vec<Sample> {
    let mut bytes = vec![0u8; COMMAND_BYTES];
    CommandRecord::default().encode_into(&mut bytes).unwrap();
    measure(repetitions, || {
        for _ in 0..COMMANDS {
            for &field in fields {
                black_box(CommandRecord::write_field(black_box(&mut bytes), field).unwrap());
            }
        }
    })
}

fn main() -> Result<()> {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let mode = arguments.first().map_or("all", String::as_str);
    let repetitions = arguments
        .iter()
        .position(|argument| argument == "--repetitions")
        .and_then(|at| arguments.get(at + 1))
        .map_or(Ok(REPETITIONS), |count| count.parse())?;
    let calls = arguments.iter().any(|argument| argument == "--calls");
    let kept = arguments.iter().any(|argument| argument == "--kept");
    let device = Device::system_default()?;
    println!(
        "device: {}; {COMMANDS} commands a repetition, median of {repetitions}",
        device.name()
    );
    if counts().is_none() {
        println!("(this system's thread counters are not read: wall time alone)");
    }
    match mode {
        "render" => render(&device, repetitions, calls),
        "compute" => compute(&device, repetitions, (calls, kept)),
        "all" => {
            render(&device, repetitions, calls)?;
            compute(&device, repetitions, (calls, kept))
        }
        other => Err(format!("unknown mode {other}: render, compute or all").into()),
    }
}
