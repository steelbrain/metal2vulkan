//! Not a consumer's example: what an indirect command buffer's records cost the GPU and the host,
//! where a kernel or the CPU writes them and a render pass executes them.
//!
//! ```sh
//! cargo run --release -p metal2vulkan-runtime --example icb_reads -- \
//!     [--commands N] [--runs R] [--only SHAPE,...] [--idle S] [--heap-textures H] [--no-twin] \
//!     [--rate-map] [--wide-every W]
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
//! - **`gpu-reset`**: `gpu-shared` with each thread first resetting its command on the GPU
//!   (`air.reset_render_command`), as a kernel that encodes commands anew each frame does;
//! - **`cpu-shared`**: a `Shared` buffer whose commands the CPU writes through each command's
//!   setters (its two buffers and its draw), then a render pass executes;
//! - **`gpu-step`**: as `gpu-shared`, but each command's fragment buffer 1 is a material: an
//!   argument buffer of [`TEXTURES`] textures, which its fragment function samples (Metal takes no
//!   direct texture argument in a state that supports indirect command buffers), of the `H` (512
//!   by default) 64 by 64 textures of a heap the pass uses (`use_heap`), as a 3D renderer's
//!   materials lie in one. The kernel is the owned one with its fragment buffer the material in
//!   place of the vertices, and no output; the CPU set a barrier on the last command and cleared
//!   it first, so the buffer may hold barriers and its execution takes the host step
//!   (`render/executions.rs`), as a renderer's commands a culling kernel writes may. The fragment
//!   function adds the samples, times zero, to magenta. Before the shapes run, the device makes a
//!   no-copy buffer of host pages with the private option bit (`ResourceOptions::PRIVATE_BIT`),
//!   which takes a read twin where the device imports its pages in place, as a guest's window
//!   server's buffers do, so each command's slots are looked at for read twins
//!   (`binding::TwinSlots`); `--no-twin` makes none.
//!
//! With `--rate-map`, every shape's pass is over a rasterization-rate map of a guest's rate-mapped
//! frame (`rate_mapped_pieces`' in the validation crate's runtime lane): one layer over a
//! [`MAP_SCREEN`] screen whose quality falls from 1 at the centre to 0.3 at the edges on both axes
//! ([`MAP_QUALITY`]), into a target of its physical size, so the `gpu-step` shape's host step fans
//! the commands' triangles out to the map's regions after a probe (`rate_mapped_pass.rs`,
//! **Fan-out**). With `--wide-every W` too, every `W`th command's triangle covers the screen,
//! which touches more regions than the fan-out's copies: the probe flags it, and its draw is drawn
//! in parts. The read-back check is left out over a map, whose regions move the texels.
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
//!     cpu_encode_us_per_command <median> host_step_ns_per_command <median> <device>
//! ICB-READS-COUNTS <shape> <name>=<per run> ...
//! ```
//!
//! `host_step_ns_per_command` is a run's `WorkCounts::host_step_nanos` over its commands: the time
//! the host step's thread spent recording the execution, 0 where none took the step. The read
//! twin's making prints `ICB-TWIN read_twins_made <count>`.
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

use metal2vulkan::bind::{RateMapDescriptor, RateMapLayerDescriptor};
use metal2vulkan::metal::{
    IndirectCommandType, PixelFormat, PrimitiveType, RenderStages, ResourceUsage, TextureUsage,
};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, ComputePipelineDescriptor, RenderPipelineDescriptor, StageSource,
};
use metal2vulkan_runtime::{
    BlitOption, Buffer, ClearColor, CommandBuffer, CommandBufferStatus, ComputePipelineState,
    Device, Heap, HeapDescriptor, HeapType, HostMemory, IndirectCommandBuffer,
    IndirectCommandBufferDescriptor, Library, LoadAction, Origin, PipelineOptions,
    RasterizationRateMap, RenderPassDescriptor, RenderPipelineState, ResourceOptions, Size,
    StoreAction, Texture, TextureDescriptor, WorkCounts,
};
use std::collections::BTreeMap;
use std::io::Write;
use std::process::{Command, Stdio};
use std::ptr::NonNull;
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

/// The screen of `--rate-map`'s map, and its quality samples on both axes.
const MAP_SCREEN: [u32; 2] = [3648, 1734];
const MAP_QUALITY: [f32; 5] = [0.3, 0.6, 1.0, 0.6, 0.3];

/// The textures of a `gpu-step` command's material, which its fragment function samples.
const TEXTURES: u32 = 8;

/// The side of each texture of the `gpu-step` shape's heap.
const TEXTURE_SIDE: u64 = 64;

/// `icb_vertex` (as [`FUNCTIONS`]'), and `icb_sampled`: opaque magenta plus zero times the sum
/// of the samples of the [`TEXTURES`] textures of the material at fragment buffer 1 (an argument
/// buffer of their resource IDs), at the fragment's position, each through one constexpr sampler.
fn sampled_functions() -> String {
    let n = TEXTURES as usize;
    let mut body = String::from(
        "  %xy = shufflevector <4 x float> %pos, <4 x float> poison, <2 x i32> <i32 0, i32 1>\n  \
         %uv = fmul <2 x float> %xy, <float 0.015625, float 0.015625>\n",
    );
    let mut sum = String::from("zeroinitializer");
    for t in 0..n {
        body += &format!(
            "  %p{t} = getelementptr inbounds i8, ptr addrspace(2) %material, i64 {}\n  \
             %t{t} = load ptr addrspace(1), ptr addrspace(2) %p{t}, align 8\n  \
             %s{t} = call {{ <4 x float>, i8 }} @air.sample_texture_2d.v4f32(ptr addrspace(1) \
             %t{t}, ptr addrspace(2) @__air_sampler_state, <2 x float> %uv, i1 true, <2 x i32> \
             zeroinitializer, i1 false, float 0.0, float 0.0, i32 0)\n  %v{t} = extractvalue \
             {{ <4 x float>, i8 }} %s{t}, 0\n  %a{t} = fadd <4 x float> {sum}, %v{t}\n",
            8 * t
        );
        sum = format!("%a{t}");
    }
    body += &format!(
        "  %z = fmul <4 x float> {sum}, zeroinitializer\n  %out = fadd <4 x float> <float 1.0, \
         float 0.0, float 1.0, float 1.0>, %z\n"
    );
    let fields: Vec<String> = (0..n)
        .map(|t| {
            format!(
                "i32 {}, i32 8, i32 0, !\"texture2d<float, sample>\", !\"t{t}\", \
                 !\"air.indirect_argument\", !{}",
                8 * t,
                60 + t
            )
        })
        .collect();
    let textures: Vec<String> = (0..n)
        .map(|t| {
            format!(
                "!{} = !{{i32 {t}, !\"air.texture\", !\"air.location_index\", i32 {t}, i32 1, \
                 !\"air.sample\", !\"air.arg_type_name\", !\"texture2d<float, sample>\", \
                 !\"air.arg_name\", !\"t{t}\"}}\n",
                60 + t
            )
        })
        .collect();
    let bytes = 8 * n;
    let vertex_at = FUNCTIONS
        .find("define <4 x float> @icb_vertex")
        .expect("icb_vertex");
    let vertex_end = vertex_at + FUNCTIONS[vertex_at..].find("}\n").expect("its end") + 2;
    format!(
        "{head}@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 \
         34901797601020489, i64 0], align 8\n\n{vertex}\n\
         define <4 x float> @icb_sampled(<4 x float> %pos, ptr addrspace(2) %material) \
         {{\nentry:\n{body}  ret <4 x float> %out\n}}\n\n\
         declare {{ <4 x float>, i8 }} @air.sample_texture_2d.v4f32(ptr addrspace(1), ptr \
         addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)\n\n\
         !llvm.module.flags = !{{!0, !1}}\n!air.vertex = !{{!20}}\n!air.fragment = !{{!40}}\n\
         !air.sampler_states = !{{!45}}\n!air.version = !{{!8}}\n!air.language_version = !{{!9}}\n\
         !0 = !{{i32 7, !\"air.max_device_buffers\", i32 31}}\n\
         !1 = !{{i32 7, !\"air.max_constant_buffers\", i32 31}}\n\
         !8 = !{{i32 2, i32 8, i32 0}}\n!9 = !{{!\"Metal\", i32 4, i32 0, i32 0}}\n\
         {vertex_metadata}\
         !26 = !{{!27}}\n\
         !27 = !{{!\"air.render_target\", i32 0, i32 0, !\"air.arg_type_name\", !\"float4\"}}\n\
         !40 = !{{ptr @icb_sampled, !26, !41}}\n!41 = !{{!49, !50}}\n\
         !45 = !{{!\"air.sampler_state\", ptr addrspace(2) @__air_sampler_state}}\n\
         !49 = !{{i32 0, !\"air.position\", !\"air.center\", !\"air.no_perspective\", \
         !\"air.arg_type_name\", !\"float4\", !\"air.arg_name\", !\"position\"}}\n\
         !50 = !{{i32 1, !\"air.indirect_buffer\", !\"air.buffer_size\", i32 {bytes}, \
         !\"air.location_index\", i32 1, i32 1, !\"air.read\", !\"air.address_space\", i32 2, \
         !\"air.struct_type_info\", !51, !\"air.arg_type_size\", i32 {bytes}, \
         !\"air.arg_type_align_size\", i32 8, !\"air.arg_type_name\", !\"Material\", \
         !\"air.arg_name\", !\"material\"}}\n!51 = !{{{fields}}}\n{textures}",
        head = &FUNCTIONS[..FUNCTIONS.find("define").expect("a definition")],
        vertex = &FUNCTIONS[vertex_at..vertex_end],
        vertex_metadata = (FUNCTIONS.lines())
            .filter(|line| ["!20 ", "!21 ", "!22 ", "!23 ", "!24 ", "!25 "]
                .iter()
                .any(|id| line.starts_with(id)))
            .map(|line| format!("{line}\n"))
            .collect::<String>(),
        fields = fields.join(", "),
        textures = textures.concat(),
    )
}

/// A shape: who writes the commands, and the buffer's storage.
#[derive(Clone, Copy, PartialEq)]
enum Shape {
    GpuShared,
    GpuReset,
    GpuPrivate,
    CpuShared,
    GpuStep,
}

impl Shape {
    const ALL: [Shape; 5] = [
        Shape::GpuShared,
        Shape::GpuReset,
        Shape::GpuPrivate,
        Shape::CpuShared,
        Shape::GpuStep,
    ];

    fn name(self) -> &'static str {
        match self {
            Shape::GpuShared => "gpu-shared",
            Shape::GpuReset => "gpu-reset",
            Shape::GpuPrivate => "gpu-private",
            Shape::CpuShared => "cpu-shared",
            Shape::GpuStep => "gpu-step",
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
    heap_textures: u64,
    twin: bool,
    rate_map: bool,
    wide_every: Option<u64>,
}

impl Args {
    fn parse() -> Result<Args> {
        let mut parsed = Args {
            commands: 8192,
            runs: 9,
            only: Vec::new(),
            idle: None,
            heap_textures: 512,
            twin: true,
            rate_map: false,
            wide_every: None,
        };
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            let mut value = || args.next().ok_or_else(|| format!("{arg} takes a value"));
            match arg.as_str() {
                "--commands" => parsed.commands = value()?.parse()?,
                "--runs" => parsed.runs = value()?.parse()?,
                "--only" => parsed.only = value()?.split(',').map(str::to_owned).collect(),
                "--idle" => parsed.idle = Some(value()?.parse()?),
                "--heap-textures" => parsed.heap_textures = value()?.parse()?,
                "--no-twin" => parsed.twin = false,
                "--rate-map" => parsed.rate_map = true,
                "--wide-every" => parsed.wide_every = Some(value()?.parse()?),
                other => return Err(format!("unknown argument {other}").into()),
            }
        }
        let cells = (SIDE / CELL) * (SIDE / CELL);
        if parsed.commands == 0 || parsed.commands > cells || parsed.runs == 0 {
            return Err(format!("commands must be 1 to {cells}, and runs at least 1").into());
        }
        if parsed.heap_textures < u64::from(TEXTURES) {
            return Err(format!("the heap holds at least {TEXTURES} textures").into());
        }
        if parsed
            .wide_every
            .is_some_and(|every| every == 0 || !parsed.rate_map)
        {
            return Err("--wide-every takes a count of 1 or more, with --rate-map".into());
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
    /// The writer with a GPU reset of each command before its writes ([`resetting`]).
    resetter: ComputePipelineState,
    /// The writer of commands whose fragment buffer is the material ([`materials`]).
    material_writer: ComputePipelineState,
    state: RenderPipelineState,
    /// `icb_vertex` with `icb_sampled` ([`sampled_functions`]), the heap of textures, and the
    /// material: the resource IDs of its first [`TEXTURES`] textures.
    sampled: RenderPipelineState,
    heap: Heap,
    material: Buffer,
    target: Texture,
    /// The passes' rasterization-rate map, with `--rate-map`.
    rate_map: Option<RasterizationRateMap>,
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
        let icb =
            self.device
                .new_indirect_command_buffer(&descriptor, self.commands, shape.options())?;
        // A barrier the CPU set and cleared: the buffer may hold barriers, which sends its
        // execution to the host step.
        if shape == Shape::GpuStep {
            let last = icb.indirect_render_command_at(self.commands - 1)?;
            last.set_barrier()?;
            last.clear_barrier()?;
        }
        Ok(icb)
    }

    /// `writer`'s writes of every command of `icb`, committed and waited for: its GPU time. Its
    /// third buffer is `third`.
    fn write_on_gpu(
        &self,
        icb: &IndirectCommandBuffer,
        (writer, third): (&ComputePipelineState, &Buffer),
    ) -> Result<f64> {
        let arguments = self.device.new_buffer_with_bytes(
            &icb.gpu_resource_id().0.to_le_bytes(),
            ResourceOptions::shared(),
        )?;
        let queue = self.device.new_command_queue()?;
        let command_buffer = queue.command_buffer()?;
        let mut encoder = command_buffer.compute_command_encoder()?;
        encoder.set_compute_pipeline_state(writer)?;
        encoder.set_buffer(Some(&arguments), 0, 0)?;
        encoder.set_buffer(Some(&self.vertices), 0, 1)?;
        encoder.set_buffer(Some(third), 0, 2)?;
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
    fn execute(&self, icb: &IndirectCommandBuffer, shape: Shape) -> Result<CommandBuffer> {
        let queue = self.device.new_command_queue()?;
        let command_buffer = queue.command_buffer()?;
        let mut pass = RenderPassDescriptor::new();
        let color = &mut pass.color_attachments[0];
        color.texture = Some(self.target.clone());
        color.load_action = LoadAction::Clear;
        color.store_action = StoreAction::Store;
        color.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
        pass.rasterization_rate_map = self.rate_map.clone();
        let mut encoder = command_buffer.render_command_encoder(&pass)?;
        match shape {
            Shape::GpuStep => {
                encoder.set_render_pipeline_state(&self.sampled)?;
                encoder.use_heap(&self.heap, RenderStages::FRAGMENT)?;
                let read = ResourceUsage::READ;
                encoder.use_resource(self.material.as_resource(), read, RenderStages::FRAGMENT)?;
            }
            _ => encoder.set_render_pipeline_state(&self.state)?,
        }
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
/// The writer, each thread first resetting its command on the GPU (`air.reset_render_command`),
/// as a kernel that encodes commands anew each frame does.
fn resetting() -> String {
    let load = "  %7 = bitcast";
    let at = WRITER.find(load).expect("the fixture's buffer cast");
    let declare = at
        + WRITER[at..]
            .find("; Function Attrs")
            .expect("the fixture's declarations");
    format!(
        "{}  tail call void @air.reset_render_command(ptr addrspace(1) captures(none) %6, i32 %3) \
         #2\n{}declare void @air.reset_render_command(ptr addrspace(1) captures(none), i32) \
         local_unnamed_addr #1\n\n{}",
        &WRITER[..at],
        &WRITER[at..declare],
        &WRITER[declare..]
    )
}

/// The writer of commands whose fragment buffer 1 is the kernel's third buffer (a material) in
/// place of its vertices, the owned kernel's output store left out.
fn materials() -> String {
    let fragment = "@air.set_fragment_buffer_render_command.p1i8(ptr addrspace(1) captures(none) \
                    %6, i32 %3, ptr addrspace(1) %7, i32 1)";
    assert!(
        WRITER.contains(fragment),
        "the fixture's fragment buffer setter"
    );
    let own = fragment.replace("%7, i32 1)", "%2, i32 1)");
    (WRITER.lines())
        .filter(|line| !line.contains("store i32 %10"))
        .map(|line| format!("{}\n", line.replace(fragment, &own)))
        .collect()
}

/// The writer of `text`'s kernel.
fn writer(device: &Device, text: &str) -> Result<ComputePipelineState> {
    let writer = library(device, text)?;
    let writer = ComputePipelineDescriptor::new(StageSource::Function(
        writer
            .new_function_with_name("kernel_icb_encode_render_commands")?
            .function_descriptor(),
    ));
    Ok(device.new_compute_pipeline_state(&writer, &PipelineOptions::default())?)
}

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

/// A no-copy buffer of `bytes` bytes of fresh host pages with the private option bit
/// (`ResourceOptions::PRIVATE_BIT`): a read twin where the device imports the pages in place.
fn twinned(device: &Device, bytes: usize) -> Result<Buffer> {
    #[cfg(unix)]
    // SAFETY: a fresh anonymous mapping, read-write, never unmapped.
    let pages = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            bytes,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_ANON | libc::MAP_PRIVATE,
            -1,
            0,
        )
    };
    #[cfg(windows)]
    // SAFETY: a new allocation of its own, at an address the system picks.
    let pages = unsafe {
        windows_sys::Win32::System::Memory::VirtualAlloc(
            std::ptr::null(),
            bytes,
            windows_sys::Win32::System::Memory::MEM_RESERVE
                | windows_sys::Win32::System::Memory::MEM_COMMIT,
            windows_sys::Win32::System::Memory::PAGE_READWRITE,
        )
    };
    #[cfg(unix)]
    if pages == libc::MAP_FAILED {
        return Err(format!("mmap: {}", std::io::Error::last_os_error()).into());
    }
    let pages = NonNull::new(pages.cast::<u8>()).ok_or("a null mapping")?;
    let options =
        ResourceOptions::from_raw(ResourceOptions::shared().raw() | ResourceOptions::PRIVATE_BIT)?;
    // SAFETY: page-aligned read-write memory of `bytes` bytes, never unmapped, which outlives the
    // buffer.
    let buffer = unsafe {
        device.new_buffer_with_bytes_no_copy(HostMemory::new(pages), bytes as u64, options, None)
    }?;
    Ok(buffer)
}

/// Three vertices a command, in normalized device coordinates: its cell's lower half, the cells in
/// rows from the bottom left; with `wide_every`, every that many commands' a triangle over the
/// whole screen instead.
fn vertices(commands: u64, wide_every: Option<u64>) -> Vec<u8> {
    let across = SIDE / CELL;
    let step = 2.0 * CELL as f32 / SIDE as f32;
    (0..commands)
        .flat_map(|command| {
            if wide_every.is_some_and(|every| command % every == every - 1) {
                return [[-1.0, -1.0], [3.0, -1.0], [-1.0, 3.0]];
            }
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
    let sampling = library(&device, &sampled_functions())?;
    let mut descriptor = RenderPipelineDescriptor::new(StageSource::Function(
        sampling
            .new_function_with_name("icb_vertex")?
            .function_descriptor(),
    ));
    descriptor.fragment = Some(StageSource::Function(
        sampling
            .new_function_with_name("icb_sampled")?
            .function_descriptor(),
    ));
    descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    descriptor.support_indirect_command_buffers = true;
    let sampled = device.new_render_pipeline_state(&descriptor, &PipelineOptions::default())?;
    let mut sampled_texture =
        TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, TEXTURE_SIDE, TEXTURE_SIDE, false);
    sampled_texture.usage = TextureUsage::SHADER_READ;
    sampled_texture.resource_options = ResourceOptions::private();
    let fit = device.heap_texture_size_and_align(&sampled_texture)?;
    let room = args.heap_textures * fit.size.next_multiple_of(fit.align);
    let heap = device.new_heap(&HeapDescriptor::new(
        HeapType::Automatic,
        room,
        ResourceOptions::private(),
    ))?;
    let textures = (0..args.heap_textures)
        .map(|_| Ok(heap.new_texture(&sampled_texture)?))
        .collect::<Result<Vec<Texture>>>()?;
    let ids: Vec<u8> = (textures.iter().take(TEXTURES as usize))
        .flat_map(|texture| texture.gpu_resource_id().0.to_le_bytes())
        .collect();
    let material = device.new_buffer_with_bytes(&ids, ResourceOptions::shared())?;
    // Held to the end: the device has a read twin while it lives.
    let _twin = match args.twin {
        true => {
            let twin = twinned(&device, 1 << 16)?;
            println!(
                "ICB-TWIN read_twins_made {}",
                device.work_counts().read_twins_made
            );
            Some(twin)
        }
        false => None,
    };
    let rate_map = match args.rate_map {
        true => {
            let mut map = RateMapDescriptor::new(MAP_SCREEN);
            map.layers = vec![RateMapLayerDescriptor::new(
                MAP_QUALITY.to_vec(),
                MAP_QUALITY.to_vec(),
            )];
            Some(device.new_rasterization_rate_map(&map)?)
        }
        false => None,
    };
    let (width, height) = match &rate_map {
        Some(map) => {
            let size = map.physical_size_for_layer(0);
            (size.width, size.height)
        }
        None => (SIDE, SIDE),
    };
    let mut target = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, width, height, false);
    target.usage = TextureUsage::RENDER_TARGET;
    target.resource_options = ResourceOptions::private();
    let ours = Ours {
        device: &device,
        writer: writer(&device, WRITER)?,
        resetter: writer(&device, &resetting())?,
        material_writer: writer(&device, &materials())?,
        state,
        sampled,
        heap,
        material,
        target: device.new_texture(&target)?,
        rate_map,
        vertices: private_with(&device, &vertices(args.commands, args.wide_every))?,
        out: device.new_buffer(4 * args.commands, ResourceOptions::private())?,
        commands: args.commands,
    };
    if let Some(seconds) = args.idle {
        let icb = ours.icb(Shape::CpuShared)?;
        std::thread::sleep(Duration::from_secs(seconds));
        let (longest, encode) = ours.write_on_cpu(&icb)?;
        let command_buffer = ours.execute(&icb, Shape::CpuShared)?;
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
        let mut samples: [Vec<f64>; 4] = Default::default();
        let mut moved: BTreeMap<String, u64> = BTreeMap::new();
        for round in 0..=args.runs {
            let before = counts(&device.work_counts());
            let icb = ours.icb(shape)?;
            let (encode, cpu) = match shape {
                Shape::CpuShared => (0.0, ours.write_on_cpu(&icb)?.1.as_secs_f64() * 1e6),
                Shape::GpuReset => (ours.write_on_gpu(&icb, (&ours.resetter, &ours.out))?, 0.0),
                Shape::GpuStep => {
                    let writing = (&ours.material_writer, &ours.material);
                    (ours.write_on_gpu(&icb, writing)?, 0.0)
                }
                _ => (ours.write_on_gpu(&icb, (&ours.writer, &ours.out))?, 0.0),
            };
            let command_buffer = ours.execute(&icb, shape)?;
            run(&command_buffer)?;
            let execute = gpu_us(&command_buffer)?;
            if round == 0 {
                continue;
            }
            let after = counts(&device.work_counts());
            let moved_by = |name: &str| {
                let at = |of: &BTreeMap<String, u64>| of.get(name).copied().unwrap_or(0);
                at(&after).saturating_sub(at(&before))
            };
            // The step's nanoseconds, which `per` gives a command.
            let step = moved_by("host_step_nanos") as f64;
            for (sample, value) in samples.iter_mut().zip([encode, execute, cpu, step]) {
                sample.push(per(value));
            }
            for (count, value) in after {
                let delta = value.saturating_sub(before.get(&count).copied().unwrap_or(0));
                if delta > 0 {
                    *moved.entry(count).or_default() += delta;
                }
            }
        }
        if ours.rate_map.is_none() {
            ours.check(shape)?;
        }
        let median = |sample: &mut Vec<f64>| {
            sample.sort_by(f64::total_cmp);
            sample[sample.len() / 2]
        };
        let [encode, execute, cpu, step] = &mut samples;
        println!(
            "ICB-READS {} commands {} gpu_us_per_command encode {:.4} execute {:.4} \
             cpu_encode_us_per_command {:.4} host_step_ns_per_command {:.1} {name}",
            shape.name(),
            args.commands,
            median(encode),
            median(execute),
            median(cpu),
            median(step),
        );
        let line: Vec<String> = (moved.iter())
            .map(|(count, sum)| format!("{count}={:.2}", *sum as f64 / args.runs as f64))
            .collect();
        println!("ICB-READS-COUNTS {} {}", shape.name(), line.join(" "));
    }
    Ok(())
}
