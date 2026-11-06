//! Not a consumer's example: what a draw costs the thread that encodes it, in three shapes of a
//! consumer's render work (a guest's command buffers as its host replays them on the runtime),
//! through the runtime's public calls alone, on one thread.
//!
//! ```sh
//! cargo run --release -p metal2vulkan-runtime --example draw_costs [-- [ws|many|10k]... [--runs N] [--calls] [--metallib PATH]]
//! ```
//!
//! Each shape is command buffers committed back to back, each of one render pass over a 512x512
//! `BGRA8Unorm` target, cleared and stored. A command buffer's pass sets the first pipeline state
//! and a constants buffer at vertex index 1 and fragment index 0, then draws:
//!
//! - **ws**, a compositor's frames: 8 command buffers of 500 draws. Each draw is a 4-vertex
//!   triangle strip after `set_fragment_texture` (one of 16 64x64 textures), `set_fragment_buffer`
//!   and `set_vertex_buffer`, each buffer at a 256-byte slot of its own; every 4 draws,
//!   `set_render_pipeline_state` (the next of 4) and `set_fragment_sampler_states` again.
//! - **many**, a guest's frame in short command buffers: 64 command buffers of 16 draws, each a
//!   `set_vertex_buffer` at the next slot and a 3-vertex triangle list.
//! - **10k**: 3 command buffers of 10,000 such draws.
//!
//! The slots lie in 64 KiB no-copy buffers over host pages (`new_buffer_with_bytes_no_copy`), as
//! a guest's upload chunks do, each command buffer's in chunks of its own. A command buffer ends
//! with `end_encoding`, `add_completed_handler` and `commit`. The four pipeline states pair one
//! vertex function (positions from vertex buffer 0, a colour from buffer 1) with a fragment
//! function that returns fragment buffer 0's first value or samples texture 0 with sampler 0,
//! each opaque and blended over (`draw_costs.metal`). The device is opened with GPU times from
//! the host's clock and a message sink, the rest at its defaults.
//!
//! A run is 2 untimed repetitions then 6 counted ones. Each repetition counts the calling
//! thread's CPU time and, on macOS, its instructions and cycles (`thread_selfcounts`, libSystem's,
//! readable unprivileged) from the first command buffer's creation through the last commit, then
//! waits for every command buffer's handler outside the count. Printed per shape and figure: the
//! figure a draw, the median over the runs with the lowest and the highest (`DRAW-COST` lines).
//! Instructions are the figure to compare between builds: a busy host moves them little where it
//! moves time. The thread asks for the performance cores (`QOS_CLASS_USER_INTERACTIVE`).
//!
//! With `--calls`, a run also counts each call's instructions apart, reading the counter around
//! each call (the read's own cost, measured as the run starts, taken off): printed per call
//! (`DRAW-CALL` lines), its calls a draw, its instructions a call and its share of a draw, the
//! median over the runs, with the lowest and highest a call. The reads cost the totals their own
//! instructions, so a run's totals are read without `--calls`.
//!
//! The library is the metallib `--metallib` names, or else `draw_costs.metal` compiled by
//! `xcrun metal` in a scratch directory, removed once read.

use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::metal::{BlendFactor, PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_runtime::{
    Buffer, ClearColor, CommandBuffer, CommandBufferStatus, CommandQueue, Device, DeviceOptions,
    HostMemory, Library, LoadAction, MessageSink, PipelineOptions, RenderPassDescriptor,
    RenderPipelineState, ResourceOptions, SamplerState, StoreAction, Texture, TextureDescriptor,
};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;
use std::time::Duration;

/// The example's result: every runtime refusal is a typed `metal2vulkan_runtime::Error`.
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// A run's untimed repetitions, then its counted ones; the runs by default.
const WARM: usize = 2;
const COUNTED: usize = 6;
const RUNS: usize = 9;

/// An upload chunk's bytes, and each draw's slot in it.
const CHUNK: u64 = 0x1_0000;
const SLOT: u64 = 256;

/// A compositor's pipeline states and textures, and the draws a pipeline state runs for.
const PIPELINES: usize = 4;
const TEXTURES: usize = 16;
const RUN: usize = 4;

/// The target's side, and a texture's.
const SIDE: u64 = 512;
const TEXTURE_SIDE: u64 = 64;

/// The longest a repetition waits for a command buffer's handler.
const WAIT: Duration = Duration::from_secs(30);

/// The notices the device's sink heard.
static NOTICES: AtomicU64 = AtomicU64::new(0);

/// What a shape's draws set.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Stream {
    /// Each draw a vertex buffer at its slot, one pipeline state.
    Bytes,
    /// A compositor's: each draw a texture and its fragment and vertex slots, the pipeline state
    /// and the sampler again every [`RUN`] draws.
    Layers,
}

/// Command buffers of `draws` draws of `stream`'s each.
#[derive(Clone, Copy)]
struct Shape {
    name: &'static str,
    buffers: usize,
    draws: usize,
    stream: Stream,
}

/// The shapes, in the order they run.
const SHAPES: [Shape; 3] = [
    Shape {
        name: "ws",
        buffers: 8,
        draws: 500,
        stream: Stream::Layers,
    },
    Shape {
        name: "many",
        buffers: 64,
        draws: 16,
        stream: Stream::Bytes,
    },
    Shape {
        name: "10k",
        buffers: 3,
        draws: 10_000,
        stream: Stream::Bytes,
    },
];

impl Shape {
    /// The slots a draw takes: its vertex bytes, and its fragment bytes in a compositor's stream.
    fn slots(self) -> u64 {
        match self.stream {
            Stream::Bytes => 1,
            Stream::Layers => 2,
        }
    }

    /// The chunks a command buffer's draws take.
    fn chunks_a_buffer(self) -> u64 {
        (self.draws as u64 * self.slots() * SLOT).div_ceil(CHUNK)
    }

    /// The chunk and the offset of slot `slot` of command buffer `buffer`.
    fn slot(self, buffer: usize, slot: u64) -> (usize, u64) {
        let at = slot * SLOT;
        let chunk = buffer as u64 * self.chunks_a_buffer() + at / CHUNK;
        (chunk as usize, at % CHUNK)
    }
}

/// The calls `--calls` counts apart, by their index in a [`Tally`].
const CALLS: [&str; 11] = [
    "command_buffer",
    "render_command_encoder",
    "set_render_pipeline_state",
    "set_vertex_buffer",
    "set_fragment_buffer",
    "set_fragment_texture",
    "set_fragment_sampler_states",
    "draw_primitives",
    "end_encoding",
    "add_completed_handler",
    "commit",
];

/// Each call's count and instructions, over a run's counted repetitions.
type Tally = [(u64, u64); CALLS.len()];

/// Run `body` as call `kind`, counting its instructions into `tally` where there is one.
fn call<T>(tally: &mut Option<&mut Tally>, kind: usize, body: impl FnOnce() -> T) -> T {
    let Some(tally) = tally else {
        return body();
    };
    let before = instructions_and_cycles();
    let answer = body();
    if let Some((after, before)) = instructions_and_cycles().zip(before) {
        tally[kind].0 += 1;
        tally[kind].1 += after[0] - before[0];
    }
    answer
}

/// The instructions a read of the counter pair costs between them, the median of many.
fn read_cost() -> f64 {
    let pairs = (0..10_000)
        .filter_map(|_| {
            let before = instructions_and_cycles()?;
            let after = instructions_and_cycles()?;
            Some((after[0] - before[0]) as f64)
        })
        .collect::<Vec<_>>();
    match pairs.is_empty() {
        true => 0.0,
        false => spread(pairs).0,
    }
}

/// One call of a command buffer's pass, its objects looked up before a count starts.
#[derive(Clone, Copy)]
enum Step<'a> {
    Pipeline(&'a RenderPipelineState),
    VertexBuffer(&'a Buffer, u64, u32),
    FragmentBuffer(&'a Buffer, u64, u32),
    FragmentTexture(&'a Texture),
    FragmentSampler(&'a SamplerState),
    Draw(PrimitiveType, u32),
}

/// The objects every shape draws with.
struct Bench {
    device: Device,
    queue: CommandQueue,
    pass: RenderPassDescriptor,
    pipelines: Vec<RenderPipelineState>,
    textures: Vec<Texture>,
    sampler: SamplerState,
    constants: Buffer,
}

impl Bench {
    fn new(device: Device, library: &Library) -> Result<Bench> {
        let function = |name: &str| -> Result<StageSource> {
            let function = library.new_function_with_name(name)?;
            Ok(StageSource::Function(function.function_descriptor()))
        };
        let (buffer, texture) = (
            function("draw_costs_buffer")?,
            function("draw_costs_texture")?,
        );
        let pipelines = (0..PIPELINES)
            .map(|n| {
                let mut descriptor = RenderPipelineDescriptor::new(function("draw_costs_vertex")?);
                descriptor.fragment = Some(match n % 2 {
                    0 => buffer.clone(),
                    _ => texture.clone(),
                });
                let mut color = ColorAttachmentDescriptor::new(PixelFormat::BGRA8Unorm);
                if n >= 2 {
                    color.blending_enabled = true;
                    color.source_rgb_blend_factor = BlendFactor::One;
                    color.source_alpha_blend_factor = BlendFactor::One;
                    color.destination_rgb_blend_factor = BlendFactor::OneMinusSourceAlpha;
                    color.destination_alpha_blend_factor = BlendFactor::OneMinusSourceAlpha;
                }
                descriptor.color_attachments[0] = Some(color);
                Ok(device.new_render_pipeline_state(&descriptor, &PipelineOptions::default())?)
            })
            .collect::<Result<Vec<_>>>()?;
        let mut sampled = TextureDescriptor::texture_2d(
            PixelFormat::BGRA8Unorm,
            TEXTURE_SIDE,
            TEXTURE_SIDE,
            false,
        );
        sampled.usage = TextureUsage::SHADER_READ;
        let textures = (0..TEXTURES)
            .map(|_| device.new_texture(&sampled))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let mut target = TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, SIDE, SIDE, false);
        target.usage = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
        let mut pass = RenderPassDescriptor::new();
        let color = &mut pass.color_attachments[0];
        color.texture = Some(device.new_texture(&target)?);
        color.load_action = LoadAction::Clear;
        color.store_action = StoreAction::Store;
        color.clear_color = ClearColor::new(0.0, 0.0, 0.0, 1.0);
        Ok(Bench {
            queue: device.new_command_queue()?,
            sampler: device.new_sampler_state(&SamplerDescriptor::new())?,
            constants: device.new_buffer(4096, ResourceOptions::shared())?,
            device,
            pass,
            pipelines,
            textures,
        })
    }

    /// `shape`'s chunks: no-copy buffers over fresh host pages, kept mapped for the process's
    /// life, so no import outlives its pages.
    fn chunks(&self, shape: Shape) -> Result<Vec<Buffer>> {
        let count = shape.chunks_a_buffer() * shape.buffers as u64;
        let bytes = (count * CHUNK) as usize;
        // SAFETY: a fresh anonymous mapping, read-write, never unmapped.
        let pages = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                bytes,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_ANON | libc::MAP_SHARED,
                -1,
                0,
            )
        };
        if pages == libc::MAP_FAILED {
            return Err(format!("mmap: {}", std::io::Error::last_os_error()).into());
        }
        (0..count)
            .map(|n| {
                // SAFETY: inside the mapping, which stays mapped and writable for the process's
                // life; CHUNK is a multiple of the host's page.
                let at = NonNull::new(unsafe { pages.cast::<u8>().add((n * CHUNK) as usize) })
                    .ok_or("a null mapping")?;
                // SAFETY: as above: page-aligned read-write memory that outlives the buffer.
                let buffer = unsafe {
                    self.device.new_buffer_with_bytes_no_copy(
                        HostMemory::new(at),
                        CHUNK,
                        ResourceOptions::shared(),
                        None,
                    )
                }?;
                Ok(buffer)
            })
            .collect()
    }

    /// Command buffer `buffer`'s calls of a repetition of `shape`, over `chunks`.
    fn steps<'a>(&'a self, chunks: &'a [Buffer], shape: Shape, buffer: usize) -> Vec<Step<'a>> {
        let mut steps = vec![
            Step::Pipeline(&self.pipelines[0]),
            Step::VertexBuffer(&self.constants, 0, 1),
            Step::FragmentBuffer(&self.constants, 0, 0),
        ];
        for draw in 0..shape.draws {
            let slot = draw as u64 * shape.slots();
            match shape.stream {
                Stream::Bytes => {
                    let (chunk, offset) = shape.slot(buffer, slot);
                    steps.push(Step::VertexBuffer(&chunks[chunk], offset, 0));
                    steps.push(Step::Draw(PrimitiveType::Triangle, 3));
                }
                Stream::Layers => {
                    if draw % RUN == 0 {
                        steps.push(Step::Pipeline(&self.pipelines[draw / RUN % PIPELINES]));
                        steps.push(Step::FragmentSampler(&self.sampler));
                    }
                    steps.push(Step::FragmentTexture(&self.textures[draw % TEXTURES]));
                    let (chunk, offset) = shape.slot(buffer, slot + 1);
                    steps.push(Step::FragmentBuffer(&chunks[chunk], offset, 0));
                    let (chunk, offset) = shape.slot(buffer, slot);
                    steps.push(Step::VertexBuffer(&chunks[chunk], offset, 0));
                    steps.push(Step::Draw(PrimitiveType::TriangleStrip, 4));
                }
            }
        }
        steps
    }

    /// One repetition: every command buffer's calls, counted from the first one's creation
    /// through the last commit, then each handler waited for; each call counted apart into
    /// `tally` where there is one.
    fn repetition(
        &self,
        buffers: &[Vec<Step<'_>>],
        mut tally: Option<&mut Tally>,
    ) -> Result<Counts> {
        let (sender, completions) = mpsc::channel();
        let start = Counts::now();
        for steps in buffers {
            let command_buffer = call(&mut tally, 0, || self.queue.command_buffer())?;
            let mut encoder = call(&mut tally, 1, || {
                command_buffer.render_command_encoder(&self.pass)
            })?;
            for &step in steps {
                match step {
                    Step::Pipeline(state) => {
                        call(&mut tally, 2, || encoder.set_render_pipeline_state(state))
                    }
                    Step::VertexBuffer(buffer, offset, index) => call(&mut tally, 3, || {
                        encoder.set_vertex_buffer(Some(buffer), offset, index)
                    }),
                    Step::FragmentBuffer(buffer, offset, index) => call(&mut tally, 4, || {
                        encoder.set_fragment_buffer(Some(buffer), offset, index)
                    }),
                    Step::FragmentTexture(texture) => call(&mut tally, 5, || {
                        encoder.set_fragment_texture(Some(texture), 0)
                    }),
                    Step::FragmentSampler(sampler) => call(&mut tally, 6, || {
                        encoder.set_fragment_sampler_states(&[Some(sampler)], None, 0)
                    }),
                    Step::Draw(primitive, count) => call(&mut tally, 7, || {
                        encoder.draw_primitives(primitive, 0, count, 1, 0)
                    }),
                }?;
            }
            call(&mut tally, 8, || encoder.end_encoding())?;
            let sender = sender.clone();
            call(&mut tally, 9, || {
                command_buffer.add_completed_handler(Box::new(move |done: &CommandBuffer| {
                    let _ = sender.send(done.status());
                }))
            })?;
            call(&mut tally, 10, || command_buffer.commit())?;
        }
        let spent = Counts::now().since(start);
        for _ in buffers {
            match completions.recv_timeout(WAIT) {
                Ok(CommandBufferStatus::Completed) => {}
                Ok(status) => return Err(format!("a command buffer ended {status:?}").into()),
                Err(_) => {
                    return Err(format!("a command buffer did not complete in {WAIT:?}").into())
                }
            }
        }
        Ok(spent)
    }
}

/// The calling thread's CPU time, and its instructions and cycles where the system counts them.
#[derive(Clone, Copy, Default)]
struct Counts {
    nanos: u64,
    instructions: Option<[u64; 2]>,
}

impl Counts {
    fn now() -> Counts {
        let mut at = libc::timespec {
            tv_sec: 0,
            tv_nsec: 0,
        };
        // SAFETY: `at` is a timespec, which the call writes.
        let answer = unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut at) };
        assert_eq!(answer, 0, "clock_gettime");
        Counts {
            nanos: at.tv_sec as u64 * 1_000_000_000 + at.tv_nsec as u64,
            instructions: instructions_and_cycles(),
        }
    }

    /// What the thread spent from `start` until `self`.
    fn since(self, start: Counts) -> Counts {
        Counts {
            nanos: self.nanos - start.nanos,
            instructions: self
                .instructions
                .zip(start.instructions)
                .map(|(now, then)| [now[0] - then[0], now[1] - then[1]]),
        }
    }

    fn plus(self, more: Counts) -> Counts {
        Counts {
            nanos: self.nanos + more.nanos,
            instructions: self
                .instructions
                .zip(more.instructions)
                .map(|(a, b)| [a[0] + b[0], a[1] + b[1]]),
        }
    }
}

/// The calling thread's instructions and cycles so far: libSystem's `thread_selfcounts`, whose
/// `THSC_CPI` (1) answers the two as `u64`s.
#[cfg(target_os = "macos")]
fn instructions_and_cycles() -> Option<[u64; 2]> {
    unsafe extern "C" {
        fn thread_selfcounts(kind: i32, buffer: *mut u64, bytes: usize) -> i32;
    }
    let mut words = [0u64; 2];
    // SAFETY: the buffer holds the two words `THSC_CPI` writes.
    let answer = unsafe { thread_selfcounts(1, words.as_mut_ptr(), size_of_val(&words)) };
    (answer == 0).then_some(words)
}

/// Elsewhere the CPU time alone.
#[cfg(not(target_os = "macos"))]
fn instructions_and_cycles() -> Option<[u64; 2]> {
    None
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

/// Run `shape` `runs` times and print its figures a draw, and with `calls` each call's.
fn measure(bench: &Bench, shape: Shape, runs: usize, calls: bool) -> Result<()> {
    let chunks = bench.chunks(shape)?;
    let buffers: Vec<Vec<Step<'_>>> = (0..shape.buffers)
        .map(|buffer| bench.steps(&chunks, shape, buffer))
        .collect();
    let draws = (COUNTED * shape.buffers * shape.draws) as f64;
    let mut seen = Vec::with_capacity(runs);
    let mut tallies = Vec::with_capacity(runs);
    for _ in 0..runs {
        let mut total: Option<Counts> = None;
        let mut tally: Tally = [(0, 0); CALLS.len()];
        for repetition in 0..WARM + COUNTED {
            let counted = repetition >= WARM;
            let spent = bench.repetition(&buffers, (calls && counted).then_some(&mut tally))?;
            if counted {
                total = Some(total.map_or(spent, |total| total.plus(spent)));
            }
        }
        seen.push(total.unwrap_or_default());
        tallies.push(tally);
    }
    println!(
        "{}: {} command buffers of {} draws, {runs} runs of {COUNTED} counted repetitions; a draw:",
        shape.name, shape.buffers, shape.draws
    );
    let mut figures: Vec<(&str, Vec<f64>)> = vec![(
        "cpu_ns",
        seen.iter().map(|c| c.nanos as f64 / draws).collect(),
    )];
    if seen.iter().all(|c| c.instructions.is_some()) {
        for (figure, index) in [("instructions", 0), ("cycles", 1)] {
            let values = seen
                .iter()
                .filter_map(|c| c.instructions.map(|i| i[index] as f64 / draws))
                .collect();
            figures.push((figure, values));
        }
    }
    for (figure, values) in figures {
        let (median, low, high) = spread(values);
        println!(
            "DRAW-COST {} {figure} {median:.1} [{low:.1}, {high:.1}]",
            shape.name
        );
    }
    if calls && seen.iter().all(|c| c.instructions.is_some()) {
        let read = read_cost();
        println!(
            "{}: a counter read's own instructions: {read:.1}",
            shape.name
        );
        for (kind, name) in CALLS.iter().enumerate() {
            let made = tallies[0][kind].0;
            if made == 0 {
                continue;
            }
            let each = tallies
                .iter()
                .map(|tally| tally[kind].1 as f64 / tally[kind].0 as f64 - read)
                .collect();
            let (median, low, high) = spread(each);
            let a_draw = made as f64 / draws;
            println!(
                "DRAW-CALL {} {name} calls_a_draw {a_draw:.3} a_call {median:.1} [{low:.1}, \
                 {high:.1}] a_draw {:.1}",
                shape.name,
                median * a_draw
            );
        }
    }
    Ok(())
}

/// The library: the metallib at `metallib`, or `draw_costs.metal` compiled by `xcrun metal`.
fn library(device: &Device, metallib: Option<&Path>) -> Result<Library> {
    if let Some(path) = metallib {
        return Ok(device.new_library_with_data(&std::fs::read(path)?)?);
    }
    let scratch = std::env::temp_dir().join(format!("draw-costs-{}", std::process::id()));
    std::fs::create_dir_all(&scratch)?;
    let compiled = compile(&scratch).and_then(|metallib| Ok(std::fs::read(metallib)?));
    std::fs::remove_dir_all(&scratch)?;
    Ok(device.new_library_with_data(&compiled?)?)
}

/// `draw_costs.metal`, written into `scratch` and compiled there.
fn compile(scratch: &Path) -> Result<PathBuf> {
    let source = scratch.join("draw_costs.metal");
    std::fs::write(&source, include_str!("draw_costs.metal"))?;
    let out = scratch.join("draw_costs.metallib");
    let status = Command::new("xcrun")
        .args(["-sdk", "macosx", "metal", "-o"])
        .arg(&out)
        .arg(&source)
        .status()
        .map_err(|error| format!("cannot run xcrun metal (or pass --metallib): {error}"))?;
    match status.success() {
        true => Ok(out),
        false => Err(format!("xcrun metal: {status}").into()),
    }
}

fn main() -> Result<()> {
    let mut shapes = Vec::new();
    let (mut runs, mut metallib, mut calls) = (RUNS, None, false);
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--runs" => runs = args.next().ok_or("--runs N")?.parse()?,
            "--calls" => calls = true,
            "--metallib" => metallib = Some(PathBuf::from(args.next().ok_or("--metallib PATH")?)),
            name => shapes.push(
                *SHAPES
                    .iter()
                    .find(|shape| shape.name == name)
                    .ok_or_else(|| format!("no shape {name}: ws, many or 10k"))?,
            ),
        }
    }
    if shapes.is_empty() {
        shapes = SHAPES.to_vec();
    }
    // On a Mac, the performance cores, which an efficiency core's slower draws would otherwise mix
    // into the figures.
    #[cfg(target_os = "macos")]
    // SAFETY: sets the calling thread's own class; it takes no pointer.
    unsafe {
        libc::pthread_set_qos_class_self_np(libc::qos_class_t::QOS_CLASS_USER_INTERACTIVE, 0);
    }
    let mut options = DeviceOptions::default();
    options.gpu_times_from_host = true;
    options.messages = Some(MessageSink::new(|message| {
        if message.degradation.is_some() {
            NOTICES.fetch_add(1, Ordering::Relaxed);
        }
    }));
    let device = Device::system_default_with(&options)?;
    println!("device: {}", device.name());
    let library = library(&device, metallib.as_deref())?;
    let bench = Bench::new(device, &library)?;
    for shape in shapes {
        measure(&bench, shape, runs, calls)?;
    }
    println!("notices heard: {}", NOTICES.load(Ordering::Relaxed));
    Ok(())
}
