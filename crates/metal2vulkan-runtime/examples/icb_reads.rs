//! Not a consumer's example: what an indirect command buffer's records cost the GPU and the host,
//! where a kernel or the CPU writes them and a render pass executes them.
//!
//! ```sh
//! cargo run --release -p metal2vulkan-runtime --example icb_reads -- \
//!     [--commands N] [--runs R] [--only SHAPE,...] [--idle S]
//! ```
//!
//! Each shape's buffer holds `N` draw commands (8192 by default) that inherit the pipeline state,
//! each of one triangle of its own over a 1024 by 1024 `RGBA8Unorm` target (an 8 by 8 texel
//! cell's lower half, in rows), its vertices read by vertex ID from vertex buffer 0, which every
//! command binds, with fragment buffer 1 (`kernel_icb_encode_render_commands`' bindings); the
//! fragment function returns opaque magenta. The shapes:
//!
//! - **`gpu-shared`** and **`gpu-private`**: a buffer of `Shared`, or `Private`, storage whose
//!   commands the owned kernel `kernel_icb_encode_render_commands` writes, one thread a command,
//!   in one command buffer, and a render pass of another command buffer executes;
//! - **`cpu-shared`**: a `Shared` buffer whose commands the CPU writes through each command's
//!   setters (its two buffers and its draw), then a render pass executes.
//!
//! Each line is the median over `R` runs (9 by default) after one not counted, of each command
//! buffer's GPU time (`gpu_start_time` to `gpu_end_time`) a command, the writer's (`encode`, the
//! kernel's command buffer; none for the CPU's) and the pass's (`execute`), and the host's time a
//! command in the CPU's setters (`cpu_encode`), with the runtime's counts a run that moved over
//! the counted runs (`WorkCounts`, by name from their `Debug` form). Once a shape's runs ended its
//! target is read back: every texel a triangle covers must be magenta.
//!
//! ```text
//! ICB-READS <shape> commands <N> gpu_us_per_command encode <median> execute <median>
//!     cpu_encode_us_per_command <median> <device>
//! ICB-READS-COUNTS <shape> <name>=<per run> ...
//! ```
//!
//! With `--idle S`, the CPU's shape first waits `S` seconds with no work on the device, then
//! writes one buffer's commands, commits its pass and waits for it, printing the longest single
//! setter call, the writes' and the commit's times and the commit to the completion:
//!
//! ```text
//! ICB-IDLE <S> longest_setter_ms <ms> encode_ms <ms> commit_ms <ms> commit_to_completed_ms <ms>
//! ```
//!
//! The render functions are LLVM text here and the kernel the owned fixture's (`metal2vulkan-
//! validation/fixtures/public/kernel_icb_encode_render_commands.ll`, built in), each assembled by
//! `llvm-as` from `PATH`.

use metal2vulkan::metal::{
    IndirectCommandType, PixelFormat, PrimitiveType, RenderStages, ResourceUsage, TextureUsage,
};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, ComputePipelineDescriptor, RenderPipelineDescriptor, StageSource,
};
use metal2vulkan_runtime::{
    BlitOption, Buffer, ClearColor, CommandBuffer, CommandBufferStatus, ComputePipelineState,
    Device, IndirectCommandBuffer, IndirectCommandBufferDescriptor, Library, LoadAction, Origin,
    PipelineOptions, RenderPassDescriptor, RenderPipelineState, ResourceOptions, Size, StoreAction,
    Texture, TextureDescriptor, WorkCounts,
};
use std::collections::BTreeMap;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// The example's result: every runtime refusal is a typed `metal2vulkan_runtime::Error`.
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// The longest the example waits for a command buffer.
const WAIT: Duration = Duration::from_secs(60);

/// The target's side, and a cell's.
const SIDE: u64 = 1024;
const CELL: u64 = 8;

/// The fragment function's color, exact in `RGBA8Unorm`.
const TEXEL: [u8; 4] = [255, 0, 255, 255];

/// The writer: the owned fixture's kernel.
const WRITER: &str = include_str!(
    "../../metal2vulkan-validation/fixtures/public/kernel_icb_encode_render_commands.ll"
);

/// `icb_vertex`: the `float4` at vertex `id` of vertex buffer 0; `icb_fragment`: opaque magenta.
const FUNCTIONS: &str = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\n\
target triple = \"air64_v28-apple-macosx26.0.0\"\n\n\
define <4 x float> @icb_vertex(ptr addrspace(1) noundef readonly %v, i32 noundef %id) {\n\
entry:\n\
  %i = zext i32 %id to i64\n\
  %p = getelementptr inbounds <4 x float>, ptr addrspace(1) %v, i64 %i\n\
  %x = load <4 x float>, ptr addrspace(1) %p, align 16\n\
  ret <4 x float> %x\n\
}\n\n\
define <4 x float> @icb_fragment() {\n\
entry:\n\
  ret <4 x float> <float 1.0, float 0.0, float 1.0, float 1.0>\n\
}\n\n\
!llvm.module.flags = !{!0, !1}\n!air.vertex = !{!20}\n!air.fragment = !{!30}\n\
!air.version = !{!8}\n!air.language_version = !{!9}\n\
!0 = !{i32 7, !\"air.max_device_buffers\", i32 31}\n\
!1 = !{i32 7, !\"air.max_constant_buffers\", i32 31}\n\
!8 = !{i32 2, i32 8, i32 0}\n!9 = !{!\"Metal\", i32 4, i32 0, i32 0}\n\
!20 = !{ptr @icb_vertex, !21, !23}\n\
!21 = !{!22}\n\
!22 = !{!\"air.position\", !\"air.arg_type_name\", !\"float4\"}\n\
!23 = !{!24, !25}\n\
!24 = !{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 16, !\"air.arg_type_align_size\", i32 16, !\"air.arg_type_name\", !\"float4\", !\"air.arg_name\", !\"v\"}\n\
!25 = !{i32 1, !\"air.vertex_id\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"id\"}\n\
!26 = !{!27}\n\
!27 = !{!\"air.render_target\", i32 0, i32 0, !\"air.arg_type_name\", !\"float4\"}\n\
!28 = !{}\n\
!30 = !{ptr @icb_fragment, !26, !28}\n";

/// A shape: who writes the commands, and the buffer's storage.
#[derive(Clone, Copy, PartialEq)]
enum Shape {
    GpuShared,
    GpuPrivate,
    CpuShared,
}

impl Shape {
    const ALL: [Shape; 3] = [Shape::GpuShared, Shape::GpuPrivate, Shape::CpuShared];

    fn name(self) -> &'static str {
        match self {
            Shape::GpuShared => "gpu-shared",
            Shape::GpuPrivate => "gpu-private",
            Shape::CpuShared => "cpu-shared",
        }
    }

    fn options(self) -> ResourceOptions {
        match self {
            Shape::GpuPrivate => ResourceOptions::private(),
            _ => ResourceOptions::shared(),
        }
    }
}

/// The run's arguments.
struct Args {
    commands: u64,
    runs: usize,
    only: Vec<String>,
    idle: Option<u64>,
}

impl Args {
    fn parse() -> Result<Args> {
        let mut parsed = Args {
            commands: 8192,
            runs: 9,
            only: Vec::new(),
            idle: None,
        };
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            let mut value = || args.next().ok_or_else(|| format!("{arg} takes a value"));
            match arg.as_str() {
                "--commands" => parsed.commands = value()?.parse()?,
                "--runs" => parsed.runs = value()?.parse()?,
                "--only" => parsed.only = value()?.split(',').map(str::to_owned).collect(),
                "--idle" => parsed.idle = Some(value()?.parse()?),
                other => return Err(format!("unknown argument {other}").into()),
            }
        }
        let cells = (SIDE / CELL) * (SIDE / CELL);
        if parsed.commands == 0 || parsed.commands > cells || parsed.runs == 0 {
            return Err(format!("commands must be 1 to {cells}, and runs at least 1").into());
        }
        Ok(parsed)
    }

    fn runs(&self, shape: Shape) -> bool {
        self.only.is_empty() || self.only.iter().any(|name| name == shape.name())
    }
}

/// Run `tool` from `PATH` over `input`: its output.
fn tool(tool: &str, args: &[&str], input: &str) -> Result<Vec<u8>> {
    let mut child = Command::new(tool)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("cannot run {tool} (put it on PATH): {error}"))?;
    child
        .stdin
        .take()
        .ok_or("no stdin")?
        .write_all(input.as_bytes())?;
    let output = child.wait_with_output()?;
    if !output.status.success() {
        return Err(format!("{tool}: {}", String::from_utf8_lossy(&output.stderr)).into());
    }
    Ok(output.stdout)
}

/// The counts of `counts` by name, read from their `Debug` form.
fn counts(counts: &WorkCounts) -> BTreeMap<String, u64> {
    let text = format!("{counts:?}");
    text.split([',', '{'])
        .filter_map(|part| {
            let (name, value) = part.split_once(':')?;
            Some((name.trim().to_owned(), value.trim().parse().ok()?))
        })
        .collect()
}

/// Commit `command_buffer` and wait for it: an error where it did not complete.
fn run(command_buffer: &CommandBuffer) -> Result<()> {
    command_buffer.commit()?;
    wait(command_buffer)
}

/// Wait for `command_buffer`: an error where it did not complete.
fn wait(command_buffer: &CommandBuffer) -> Result<()> {
    command_buffer
        .wait_until_completed_for(WAIT)
        .map_err(|_| format!("a command buffer did not complete within {WAIT:?}"))?;
    match command_buffer.status() {
        CommandBufferStatus::Completed => Ok(()),
        status => Err(format!(
            "a command buffer ended {status:?}: {:?}",
            command_buffer.error()
        )
        .into()),
    }
}

/// `command_buffer`'s GPU time in microseconds.
fn gpu_us(command_buffer: &CommandBuffer) -> Result<f64> {
    Ok((command_buffer.gpu_end_time()? - command_buffer.gpu_start_time()?) * 1e6)
}

/// What every run uses.
struct Ours<'a> {
    device: &'a Device,
    writer: ComputePipelineState,
    state: RenderPipelineState,
    target: Texture,
    /// Three vertices a command: the lower half of its cell.
    vertices: Buffer,
    /// The writer's third argument, a word a thread.
    out: Buffer,
    commands: u64,
}

impl Ours<'_> {
    /// A new buffer of the shape's storage for the commands.
    fn icb(&self, shape: Shape) -> Result<IndirectCommandBuffer> {
        let mut descriptor = IndirectCommandBufferDescriptor::new();
        descriptor.command_types = IndirectCommandType::DRAW;
        descriptor.inherit_pipeline_state = true;
        descriptor.max_vertex_buffer_bind_count = 2;
        descriptor.max_fragment_buffer_bind_count = 2;
        Ok(self
            .device
            .new_indirect_command_buffer(&descriptor, self.commands, shape.options())?)
    }

    /// The kernel's writes of every command of `icb`, committed and waited for: its GPU time.
    fn write_on_gpu(&self, icb: &IndirectCommandBuffer) -> Result<f64> {
        let arguments = self.device.new_buffer_with_bytes(
            &icb.gpu_resource_id().0.to_le_bytes(),
            ResourceOptions::shared(),
        )?;
        let queue = self.device.new_command_queue()?;
        let command_buffer = queue.command_buffer()?;
        let mut encoder = command_buffer.compute_command_encoder()?;
        encoder.set_compute_pipeline_state(&self.writer)?;
        encoder.set_buffer(Some(&arguments), 0, 0)?;
        encoder.set_buffer(Some(&self.vertices), 0, 1)?;
        encoder.set_buffer(Some(&self.out), 0, 2)?;
        encoder.use_resource(icb.as_resource(), ResourceUsage::WRITE)?;
        encoder.dispatch_threads(Size::new(self.commands, 1, 1), Size::new(64, 1, 1))?;
        encoder.end_encoding()?;
        run(&command_buffer)?;
        gpu_us(&command_buffer)
    }

    /// The CPU's writes of every command of `icb`: the longest single call and the whole.
    fn write_on_cpu(&self, icb: &IndirectCommandBuffer) -> Result<(Duration, Duration)> {
        let started = Instant::now();
        let mut longest = Duration::ZERO;
        let mut timed = |call: &mut dyn FnMut() -> Result<()>| -> Result<()> {
            let at = Instant::now();
            call()?;
            longest = longest.max(at.elapsed());
            Ok(())
        };
        for index in 0..self.commands {
            let command = icb.indirect_render_command_at(index)?;
            timed(&mut || Ok(command.set_vertex_buffer(Some(&self.vertices), 0, 0)?))?;
            timed(&mut || Ok(command.set_fragment_buffer(Some(&self.vertices), 0, 1)?))?;
            let start = 3 * index;
            timed(&mut || Ok(command.draw_primitives(PrimitiveType::Triangle, start, 3, 1, 0)?))?;
        }
        Ok((longest, started.elapsed()))
    }

    /// A render pass that executes every command of `icb`, encoded, not committed.
    fn execute(&self, icb: &IndirectCommandBuffer) -> Result<CommandBuffer> {
        let queue = self.device.new_command_queue()?;
        let command_buffer = queue.command_buffer()?;
        let mut pass = RenderPassDescriptor::new();
        let color = &mut pass.color_attachments[0];
        color.texture = Some(self.target.clone());
        color.load_action = LoadAction::Clear;
        color.store_action = StoreAction::Store;
        color.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
        let mut encoder = command_buffer.render_command_encoder(&pass)?;
        encoder.set_render_pipeline_state(&self.state)?;
        encoder.use_resource(
            self.vertices.as_resource(),
            ResourceUsage::READ,
            RenderStages::VERTEX,
        )?;
        encoder.execute_commands_in_buffer(icb, 0..self.commands)?;
        encoder.end_encoding()?;
        Ok(command_buffer)
    }

    /// Every texel a command's triangle covers fully is [`TEXEL`].
    fn check(&self, shape: Shape) -> Result<()> {
        let row = SIDE * 4;
        let readback = self
            .device
            .new_buffer(row * SIDE, ResourceOptions::shared())?;
        let queue = self.device.new_command_queue()?;
        let command_buffer = queue.command_buffer()?;
        let mut blit = command_buffer.blit_command_encoder()?;
        blit.copy_from_texture_to_buffer(
            &self.target,
            0,
            0,
            Origin::new(0, 0, 0),
            Size::new(SIDE, SIDE, 1),
            &readback,
            0,
            row,
            row * SIDE,
            BlitOption::None,
        )?;
        blit.end_encoding()?;
        run(&command_buffer)?;
        let contents = readback.contents().ok_or("the read-back has no contents")?;
        // SAFETY: `contents` holds the buffer's bytes, which no command buffer writes now.
        let texels =
            unsafe { std::slice::from_raw_parts(contents.as_ptr(), readback.length() as usize) };
        let across = SIDE / CELL;
        for command in 0..self.commands {
            // Texel (1, 6) of the cell, in Metal's rows from the top: inside the lower half's
            // triangle (`vertices`) whatever the rasterizer's rules for its edges.
            let (cx, cy) = (command % across, command / across);
            let x = cx * CELL + 1;
            let y = SIDE - 1 - (cy * CELL + 1);
            let at = ((y * SIDE + x) * 4) as usize;
            if texels[at..at + 4] != TEXEL {
                return Err(format!(
                    "{}: command {command}'s texel ({x}, {y}) is {:?}, not {TEXEL:?}",
                    shape.name(),
                    &texels[at..at + 4]
                )
                .into());
            }
        }
        Ok(())
    }
}

/// The library of the LLVM text `text`, assembled by `llvm-as`.
fn library(device: &Device, text: &str) -> Result<Library> {
    let bitcode = tool("llvm-as", &["-o", "-", "-"], text)?;
    let library = metal2vulkan::library::Library::from_air(&bitcode)?;
    Ok(device.new_library_with_compile_library(library))
}

/// `bytes` in a new `Private` buffer, copied in from a `Shared` one.
fn private_with(device: &Device, bytes: &[u8]) -> Result<Buffer> {
    let staging = device.new_buffer_with_bytes(bytes, ResourceOptions::shared())?;
    let buffer = device.new_buffer(bytes.len() as u64, ResourceOptions::private())?;
    let queue = device.new_command_queue()?;
    let command_buffer = queue.command_buffer()?;
    let mut blit = command_buffer.blit_command_encoder()?;
    blit.copy_from_buffer_to_buffer(&staging, 0, &buffer, 0, bytes.len() as u64)?;
    blit.end_encoding()?;
    run(&command_buffer)?;
    Ok(buffer)
}

/// Three vertices a command, in normalized device coordinates: its cell's lower half, the cells in
/// rows from the bottom left.
fn vertices(commands: u64) -> Vec<u8> {
    let across = SIDE / CELL;
    let step = 2.0 * CELL as f32 / SIDE as f32;
    (0..commands)
        .flat_map(|command| {
            let x = -1.0 + step * (command % across) as f32;
            let y = -1.0 + step * (command / across) as f32;
            [[x, y], [x + step, y], [x, y + step]]
        })
        .flat_map(|[x, y]| [x, y, 0.0, 1.0])
        .flat_map(f32::to_le_bytes)
        .collect()
}

fn main() -> Result<()> {
    let args = Args::parse()?;
    let device = Device::system_default()?;
    let name = device.name().to_string();
    println!("device: {name}");
    let functions = library(&device, FUNCTIONS)?;
    let mut descriptor = RenderPipelineDescriptor::new(StageSource::Function(
        functions
            .new_function_with_name("icb_vertex")?
            .function_descriptor(),
    ));
    descriptor.fragment = Some(StageSource::Function(
        functions
            .new_function_with_name("icb_fragment")?
            .function_descriptor(),
    ));
    descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    descriptor.support_indirect_command_buffers = true;
    let state = device.new_render_pipeline_state(&descriptor, &PipelineOptions::default())?;
    let writer = library(&device, WRITER)?;
    let writer = ComputePipelineDescriptor::new(StageSource::Function(
        writer
            .new_function_with_name("kernel_icb_encode_render_commands")?
            .function_descriptor(),
    ));
    let writer = device.new_compute_pipeline_state(&writer, &PipelineOptions::default())?;
    let mut target = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
    target.usage = TextureUsage::RENDER_TARGET;
    target.resource_options = ResourceOptions::private();
    let ours = Ours {
        device: &device,
        writer,
        state,
        target: device.new_texture(&target)?,
        vertices: private_with(&device, &vertices(args.commands))?,
        out: device.new_buffer(4 * args.commands, ResourceOptions::private())?,
        commands: args.commands,
    };
    if let Some(seconds) = args.idle {
        let icb = ours.icb(Shape::CpuShared)?;
        std::thread::sleep(Duration::from_secs(seconds));
        let (longest, encode) = ours.write_on_cpu(&icb)?;
        let command_buffer = ours.execute(&icb)?;
        let at = Instant::now();
        command_buffer.commit()?;
        let commit = at.elapsed();
        wait(&command_buffer)?;
        println!(
            "ICB-IDLE {seconds} longest_setter_ms {:.3} encode_ms {:.3} commit_ms {:.3} \
             commit_to_completed_ms {:.3}",
            longest.as_secs_f64() * 1e3,
            encode.as_secs_f64() * 1e3,
            commit.as_secs_f64() * 1e3,
            at.elapsed().as_secs_f64() * 1e3,
        );
    }
    let per = |us: f64| us / args.commands as f64;
    for shape in Shape::ALL {
        if !args.runs(shape) {
            continue;
        }
        let mut samples: [Vec<f64>; 3] = Default::default();
        let mut moved: BTreeMap<String, u64> = BTreeMap::new();
        for round in 0..=args.runs {
            let before = counts(&device.work_counts());
            let icb = ours.icb(shape)?;
            let (encode, cpu) = match shape {
                Shape::CpuShared => (0.0, ours.write_on_cpu(&icb)?.1.as_secs_f64() * 1e6),
                _ => (ours.write_on_gpu(&icb)?, 0.0),
            };
            let command_buffer = ours.execute(&icb)?;
            run(&command_buffer)?;
            let execute = gpu_us(&command_buffer)?;
            if round == 0 {
                continue;
            }
            for (sample, value) in samples.iter_mut().zip([encode, execute, cpu]) {
                sample.push(per(value));
            }
            for (count, value) in counts(&device.work_counts()) {
                let delta = value.saturating_sub(before.get(&count).copied().unwrap_or(0));
                if delta > 0 {
                    *moved.entry(count).or_default() += delta;
                }
            }
        }
        ours.check(shape)?;
        let median = |sample: &mut Vec<f64>| {
            sample.sort_by(f64::total_cmp);
            sample[sample.len() / 2]
        };
        let [encode, execute, cpu] = &mut samples;
        println!(
            "ICB-READS {} commands {} gpu_us_per_command encode {:.4} execute {:.4} \
             cpu_encode_us_per_command {:.4} {name}",
            shape.name(),
            args.commands,
            median(encode),
            median(execute),
            median(cpu),
        );
        let line: Vec<String> = (moved.iter())
            .map(|(count, sum)| format!("{count}={:.2}", *sum as f64 / args.runs as f64))
            .collect();
        println!("ICB-READS-COUNTS {} {}", shape.name(), line.join(" "));
    }
    Ok(())
}
