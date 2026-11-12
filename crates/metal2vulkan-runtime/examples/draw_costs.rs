//! Not a consumer's example: what a draw costs the thread that encodes it, in four shapes of a
//! consumer's render work (a guest's command buffers as its host replays them on the runtime),
//! through the runtime's public calls alone, on one thread.
//!
//! ```sh
//! cargo run --release -p metal2vulkan-runtime --example draw_costs [-- [ws|many|10k|clips|damage|compositor]... [--runs N] [--calls] [--counts] [--heap] [--shadowed] [--pages] [--threads N] [--metallib PATH]]
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
//! - **clips**, a compositor's frames under clips: 8 command buffers of 500 draws, each a 4-vertex
//!   triangle strip after `set_vertex_buffer` at the next slot; before every second one,
//!   `set_scissor_rect` (the top half of the target, then the bottom, in turn); every 4 draws
//!   instead `set_render_pipeline_state` (the next of 4), `set_fragment_sampler_states` and
//!   `set_fragment_texture` (the next of 16).
//! - **damage**, a compositor's frames of damaged rectangles: as **clips**, but each draw after a
//!   `set_scissor_rect` of its own, the next cell of a grid of 25 by 20 cells of 20x25 pixels, so
//!   a pass's 500 draws are bounded by 500 rectangles.
//! - **compositor**, a window compositor's frame as the consumer's per-draw table measures it:
//!   8 command buffers of 6 passes and 35 draws. Five passes each load a layer's surface
//!   (1600x1000, 1280x800, 1024x768, 800x600, 512x512) and draw 4 quads into it, then one clears
//!   a 3840x2400 framebuffer surface and draws 15. Every surface is `BGRA8Unorm`, a texture over
//!   its own buffer's one IOSurface plane at packed rows (`new_texture_with_iosurface_plane`):
//!   a `Private` buffer, as a host keeps a surface's device copy, or with `--pages` a no-copy
//!   buffer over host pages of its own. Each pass begins with `set_viewport`; each draw sets the
//!   pipeline state where its kind differs from the last draw's (plain, masked or backdrop, each
//!   blended), vertex buffers 0 (corners) and 1 (placement) and fragment buffer 0 (a tint) at
//!   slots of one no-copy buffer over host pages that move every draw, fragment texture 0 (one
//!   of 12 sampled surfaces: a 3840x2400 desktop picture, then windows of 1920x1200, 1024x768,
//!   512x512 and 256x256), a masked draw's texture 1 (another of them) or a backdrop draw's
//!   (its own pass's color attachment, read at the fragment's pixel), a sampler and a scissor
//!   rectangle, then draws 6 vertices, three draws of five indexed (`UInt16`). One draw in each
//!   pass is a backdrop.
//!
//! The slots lie in 64 KiB no-copy buffers over host pages (`new_buffer_with_bytes_no_copy`), as
//! a guest's upload chunks do, each command buffer's in chunks of its own. With `--heap`, the
//! chunks are instead buffers placed at their offsets in one `Placement` heap over the pages
//! (`new_heap_with_bytes_no_copy`, then `new_buffer_at_offset`), as the consumer places a guest's
//! upload chunks in a heap over the guest's own RAM, beside a no-copy buffer over pages that hold
//! the heap's and one chunk more, which stands for the guest RAM's own key over them: the
//! heap's bytes meet it, so each use of them takes a use of it too (`host_aliases.rs`'s
//! placements), as the guest RAM's do. A command buffer ends
//! with `end_encoding`, `add_completed_handler` and `commit`. The four pipeline states pair one
//! vertex function (positions from vertex buffer 0, a colour from buffer 1) with a fragment
//! function that returns fragment buffer 0's first value or samples texture 0 with sampler 0,
//! each opaque and blended over (`draw_costs.metal`). The device is opened with GPU times from
//! the host's clock and a message sink, the rest at its defaults.
//!
//! With `METAL2VULKAN_LOG_SHADOWS=1` (`metal2vulkan::env_vars`), each shadow the device makes is
//! printed on stderr with why, and so is each degrade notice's text.
//!
//! With `--shadowed`, the target is a texture over a no-copy buffer of host pages, as a guest's
//! window surface is, each pass loading it rather than clearing it, and the device is opened with
//! `HostPageTexturePolicy::Shadow`, so the target is a shadow on every device: each pass's end
//! plans its copy back, and counts its tiles against its draws' scissor rectangles
//! (`WorkCounts::shadow_attachment_tiles_covered`).
//!
//! A run is 2 untimed repetitions then 6 counted ones. Each repetition counts the calling
//! thread's CPU time and its instructions and cycles from the first command buffer's creation
//! through the last commit, then waits for every command buffer's handler outside the count, and
//! the process's page faults and voluntary context switches meanwhile (`getrusage`). On macOS the
//! instructions and cycles are libSystem's `thread_selfcounts` (readable unprivileged), which hold
//! the kernel's where it ran on the thread; on Linux they are the thread's own user-space counts
//! (`perf_event_open`, where `kernel.perf_event_paranoid` is at most 2; on a hybrid Intel CPU its
//! performance cores' counters alone, so pin the run there with `taskset`). Printed per shape and
//! figure: the figure a draw, the median over the runs with the lowest and the highest
//! (`DRAW-COST` lines). Instructions are the figure to compare between builds: a busy host moves
//! them little where it moves time. The thread asks for the performance cores
//! (`QOS_CLASS_USER_INTERACTIVE`).
//!
//! With `--threads N`, N threads encode at once on the one device and queue, each its own command
//! buffers of the shape over chunks of its own (in the one heap, with `--heap`), every repetition
//! started together, as the consumer walks a queue's command buffers on threads of their own:
//! each figure is then the threads' together a draw, so a cost that grows with the threads is
//! work they contend on (a lock, a counter every thread writes).
//!
//! With `--calls`, a run also counts each call's instructions apart, reading the counter around
//! each call (the read's own cost, measured as the run starts, taken off): printed per call
//! (`DRAW-CALL` lines), its calls a draw, its instructions a call and its share of a draw, the
//! median over the runs, with the lowest and highest a call. The reads cost the totals their own
//! instructions, so a run's totals are read without `--calls`.
//!
//! With `--counts`, each shape's runs are bracketed by the device's work counts
//! (`Device::work_counts`), printed whole before and after (`WORK-COUNTS` lines) with the draws
//! between, warm repetitions included.
//!
//! The library is the metallib `--metallib` names, or else `draw_costs.metal` compiled by
//! `xcrun metal` in a scratch directory, removed once read.

use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::device::HostPageTexturePolicy;
use metal2vulkan::metal::{
    BlendFactor, IndexType, PixelFormat, PrimitiveType, SamplerMinMagFilter, TextureUsage,
};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_runtime::{
    Buffer, ClearColor, CommandBuffer, CommandBufferStatus, CommandQueue, Device, DeviceOptions,
    Heap, HeapDescriptor, HeapType, HostMemory, IoSurfacePlane, Library, LoadAction, MessageSink,
    PipelineOptions, RenderPassDescriptor, RenderPipelineState, ResourceOptions, SamplerState,
    ScissorRect, StoreAction, Texture, TextureDescriptor, Viewport,
};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Barrier};
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
    /// A compositor's under clips: each draw its vertex slot, every second one a scissor
    /// rectangle of [`CLIPS`] before it, and every [`RUN`] draws instead the pipeline state, the
    /// sampler and a texture again.
    Clipped,
    /// A compositor's under damaged rectangles: as [`Stream::Clipped`], each draw after a
    /// scissor rectangle of its own ([`damaged`]).
    Damaged,
    /// A window compositor's frame of passes over surfaces ([`Compositor`]).
    Composited,
}

/// The columns of the grid of damaged rectangles a [`Stream::Damaged`] pass's draws take in turn,
/// and a cell's side in pixels.
const DAMAGE_COLUMNS: u64 = 25;
const DAMAGE_CELL: [u64; 2] = [20, 25];

/// Draw `draw`'s scissor rectangle in a [`Stream::Damaged`] pass: the next cell of the grid.
fn damaged(draw: usize) -> ScissorRect {
    let (column, row) = (draw as u64 % DAMAGE_COLUMNS, draw as u64 / DAMAGE_COLUMNS);
    ScissorRect {
        x: column * DAMAGE_CELL[0],
        y: row * DAMAGE_CELL[1] % SIDE,
        width: DAMAGE_CELL[0],
        height: DAMAGE_CELL[1],
    }
}

/// The scissor rectangles a clipped stream's draws take in turn: the target's top half, then its
/// bottom half.
const CLIPS: [ScissorRect; 2] = [
    ScissorRect {
        x: 0,
        y: 0,
        width: SIDE,
        height: SIDE / 2,
    },
    ScissorRect {
        x: 0,
        y: SIDE / 2,
        width: SIDE,
        height: SIDE / 2,
    },
];

/// Command buffers of `draws` draws of `stream`'s each.
#[derive(Clone, Copy)]
struct Shape {
    name: &'static str,
    buffers: usize,
    draws: usize,
    stream: Stream,
}

/// The shapes, in the order they run.
const SHAPES: [Shape; 6] = [
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
    Shape {
        name: "clips",
        buffers: 8,
        draws: 500,
        stream: Stream::Clipped,
    },
    Shape {
        name: "damage",
        buffers: 8,
        draws: 500,
        stream: Stream::Damaged,
    },
    Shape {
        name: "compositor",
        buffers: 8,
        draws: COMPOSITOR_DRAWS,
        stream: Stream::Composited,
    },
];

impl Shape {
    /// The slots a draw takes: its vertex bytes, and its fragment bytes in a compositor's stream.
    fn slots(self) -> u64 {
        match self.stream {
            Stream::Bytes | Stream::Clipped | Stream::Damaged => 1,
            Stream::Layers => 2,
            // Its slots lie in the compositor's own buffer.
            Stream::Composited => 0,
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
const CALLS: [&str; 14] = [
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
    "set_scissor_rect",
    "set_viewport",
    "draw_indexed_primitives",
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
    /// End the open encoder, if any, and open the compositor's pass of this index, its viewport
    /// set; a command buffer whose steps start otherwise opens the bench's one pass first.
    Pass(usize),
    Pipeline(&'a RenderPipelineState),
    VertexBuffer(&'a Buffer, u64, u32),
    FragmentBuffer(&'a Buffer, u64, u32),
    FragmentTexture(&'a Texture, u32),
    FragmentSampler(&'a SamplerState),
    Scissor(ScissorRect),
    Draw(PrimitiveType, u32),
    /// Six `UInt16` indices of a triangle list at this offset of the buffer.
    DrawIndexed(&'a Buffer, u64),
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
    /// With `--shadowed`, the no-copy buffer the target is a texture over.
    _target_pages: Option<Buffer>,
    /// The compositor shape's objects, where it runs.
    compositor: Option<Compositor>,
}

/// A shape's chunks, and with `--heap` the heap they are placed in and the buffer over the guest
/// RAM's pages that stands beside it ([`Bench::chunks`]).
struct Chunks {
    buffers: Vec<Buffer>,
    _heap: Option<Heap>,
    _ram: Option<Buffer>,
}

impl Bench {
    fn new(
        device: Device,
        library: &Library,
        shadowed: bool,
        compositor: Option<bool>,
    ) -> Result<Bench> {
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
        // With `--shadowed`, a texture over host pages at its packed rows, loaded.
        let (texture, target_pages, load) = match shadowed {
            false => (device.new_texture(&target)?, None, LoadAction::Clear),
            true => {
                let bytes = SIDE * SIDE * 4;
                // SAFETY: page-aligned read-write memory of `bytes` bytes, never unmapped, which
                // outlives the buffer.
                let pages = unsafe {
                    device.new_buffer_with_bytes_no_copy(
                        HostMemory::new(host_pages(bytes as usize)?),
                        bytes,
                        ResourceOptions::shared(),
                        None,
                    )
                }?;
                target.resource_options = ResourceOptions::shared();
                let texture = pages.new_texture(&target, 0, SIDE * 4)?;
                (texture, Some(pages), LoadAction::Load)
            }
        };
        let mut pass = RenderPassDescriptor::new();
        let color = &mut pass.color_attachments[0];
        color.texture = Some(texture);
        color.load_action = load;
        color.store_action = StoreAction::Store;
        color.clear_color = ClearColor::new(0.0, 0.0, 0.0, 1.0);
        let compositor = match compositor {
            Some(pages) => Some(Compositor::new(&device, library, pages)?),
            None => None,
        };
        Ok(Bench {
            queue: device.new_command_queue()?,
            sampler: device.new_sampler_state(&SamplerDescriptor::new())?,
            constants: device.new_buffer(4096, ResourceOptions::shared())?,
            device,
            pass,
            pipelines,
            textures,
            _target_pages: target_pages,
            compositor,
        })
    }

    /// `shape`'s chunks, `threads` times over: no-copy buffers over fresh host pages, or with
    /// `heap` buffers placed in one heap over them beside a buffer over them and one chunk more
    /// (the module docs), kept mapped for the process's life, so no import outlives its pages.
    fn chunks(&self, shape: Shape, threads: usize, heap: bool) -> Result<Chunks> {
        let count = shape.chunks_a_buffer() * (shape.buffers * threads) as u64;
        if count == 0 {
            return Ok(Chunks {
                buffers: Vec::new(),
                _heap: None,
                _ram: None,
            });
        }
        // With `heap`, one chunk before the heap's pages that only the guest RAM's buffer covers.
        let ahead = u64::from(heap) * CHUNK;
        let bytes = (ahead + count * CHUNK) as usize;
        let pages = host_pages(bytes)?;
        let at = |offset: u64| {
            // SAFETY: inside the mapping, which stays mapped and writable for the process's
            // life; CHUNK is a multiple of the host's page.
            NonNull::new(unsafe { pages.as_ptr().add(offset as usize) }).ok_or("a null mapping")
        };
        if !heap {
            let buffers = (0..count)
                .map(|n| {
                    // SAFETY: page-aligned read-write memory that outlives the buffer.
                    let buffer = unsafe {
                        self.device.new_buffer_with_bytes_no_copy(
                            HostMemory::new(at(n * CHUNK)?),
                            CHUNK,
                            ResourceOptions::shared(),
                            None,
                        )
                    }?;
                    Ok(buffer)
                })
                .collect::<Result<Vec<_>>>()?;
            return Ok(Chunks {
                buffers,
                _heap: None,
                _ram: None,
            });
        }
        // SAFETY: as above, the whole mapping.
        let ram = unsafe {
            self.device.new_buffer_with_bytes_no_copy(
                HostMemory::new(at(0)?),
                bytes as u64,
                ResourceOptions::shared(),
                None,
            )
        }?;
        let descriptor = HeapDescriptor::new(
            HeapType::Placement,
            count * CHUNK,
            ResourceOptions::shared(),
        );
        // SAFETY: as above, the mapping past its first chunk.
        let placed = unsafe {
            self.device.new_heap_with_bytes_no_copy(
                HostMemory::new(at(ahead)?),
                &descriptor,
                None,
                None,
            )
        }?;
        let buffers = (0..count)
            .map(|n| {
                Ok(placed.new_buffer_at_offset(CHUNK, ResourceOptions::shared(), n * CHUNK)?)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Chunks {
            buffers,
            _heap: Some(placed),
            _ram: Some(ram),
        })
    }

    /// Command buffer `buffer`'s calls of a repetition of `shape`, over `chunks`.
    fn steps<'a>(&'a self, chunks: &'a [Buffer], shape: Shape, buffer: usize) -> Vec<Step<'a>> {
        if let (Stream::Composited, Some(compositor)) = (shape.stream, &self.compositor) {
            return compositor.steps(buffer);
        }
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
                    steps.push(Step::FragmentTexture(&self.textures[draw % TEXTURES], 0));
                    let (chunk, offset) = shape.slot(buffer, slot + 1);
                    steps.push(Step::FragmentBuffer(&chunks[chunk], offset, 0));
                    let (chunk, offset) = shape.slot(buffer, slot);
                    steps.push(Step::VertexBuffer(&chunks[chunk], offset, 0));
                    steps.push(Step::Draw(PrimitiveType::TriangleStrip, 4));
                }
                Stream::Composited => {}
                Stream::Clipped | Stream::Damaged => {
                    if draw % RUN == 0 {
                        steps.push(Step::Pipeline(&self.pipelines[draw / RUN % PIPELINES]));
                        steps.push(Step::FragmentSampler(&self.sampler));
                        steps.push(Step::FragmentTexture(
                            &self.textures[draw / RUN % TEXTURES],
                            0,
                        ));
                    } else if draw % 2 == 1 && shape.stream == Stream::Clipped {
                        steps.push(Step::Scissor(CLIPS[draw / 2 % CLIPS.len()]));
                    }
                    if shape.stream == Stream::Damaged {
                        steps.push(Step::Scissor(damaged(draw)));
                    }
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
            let mut open = match steps.first() {
                Some(Step::Pass(_)) => None,
                _ => Some(call(&mut tally, 1, || {
                    command_buffer.render_command_encoder(&self.pass)
                })?),
            };
            for &step in steps {
                if let Step::Pass(at) = step {
                    if let Some(encoder) = open.take() {
                        call(&mut tally, 8, || encoder.end_encoding())?;
                    }
                    let compositor = self
                        .compositor
                        .as_ref()
                        .ok_or("a pass with no compositor")?;
                    let (descriptor, viewport) = &compositor.passes[at];
                    let mut encoder = call(&mut tally, 1, || {
                        command_buffer.render_command_encoder(descriptor)
                    })?;
                    call(&mut tally, 12, || encoder.set_viewport(*viewport))?;
                    open = Some(encoder);
                    continue;
                }
                let encoder = open.as_mut().ok_or("a step before its pass")?;
                match step {
                    Step::Pass(_) => Ok(()),
                    Step::Pipeline(state) => {
                        call(&mut tally, 2, || encoder.set_render_pipeline_state(state))
                    }
                    Step::VertexBuffer(buffer, offset, index) => call(&mut tally, 3, || {
                        encoder.set_vertex_buffer(Some(buffer), offset, index)
                    }),
                    Step::FragmentBuffer(buffer, offset, index) => call(&mut tally, 4, || {
                        encoder.set_fragment_buffer(Some(buffer), offset, index)
                    }),
                    Step::FragmentTexture(texture, index) => call(&mut tally, 5, || {
                        encoder.set_fragment_texture(Some(texture), index)
                    }),
                    Step::FragmentSampler(sampler) => call(&mut tally, 6, || {
                        encoder.set_fragment_sampler_states(&[Some(sampler)], None, 0)
                    }),
                    Step::Scissor(rect) => call(&mut tally, 11, || encoder.set_scissor_rect(rect)),
                    Step::Draw(primitive, count) => call(&mut tally, 7, || {
                        encoder.draw_primitives(primitive, 0, count, 1, 0)
                    }),
                    Step::DrawIndexed(buffer, offset) => call(&mut tally, 13, || {
                        encoder.draw_indexed_primitives(
                            PrimitiveType::Triangle,
                            6,
                            IndexType::UInt16,
                            buffer,
                            offset,
                            1,
                            0,
                            0,
                        )
                    }),
                }?;
            }
            if let Some(encoder) = open {
                call(&mut tally, 8, || encoder.end_encoding())?;
            }
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

/// The compositor shape's framebuffer surface, its five layer surfaces, and the draws of a layer's
/// pass and of the framebuffer's.
const FRAMEBUFFER: (u64, u64) = (3840, 2400);
const LAYERS: [(u64, u64); 5] = [
    (1600, 1000),
    (1280, 800),
    (1024, 768),
    (800, 600),
    (512, 512),
];
const LAYER_DRAWS: usize = 4;
const FRAMEBUFFER_DRAWS: usize = 15;
const COMPOSITOR_DRAWS: usize = LAYERS.len() * LAYER_DRAWS + FRAMEBUFFER_DRAWS;

/// The sampled surfaces: the first a desktop picture of the framebuffer's size, the rest windows
/// of these sizes in turn.
const SOURCES: usize = 12;
const SOURCE_SIZES: [(u64, u64); 4] = [(1920, 1200), (1024, 768), (512, 512), (256, 256)];

/// The compositor's per-draw data in its buffer over host pages: `COMPOSITOR_SLOTS` slots of each
/// kind, a draw taking the next: 4 corners (64 bytes a slot), 6 corners (128), 6 `UInt16`
/// indices (64), a placement (256) and a tint (256).
const COMPOSITOR_SLOTS: u64 = 256;
const CORNERS4: u64 = 0;
const CORNERS6: u64 = 64 << 10;
const INDICES: u64 = 128 << 10;
const PLACEMENT: u64 = 256 << 10;
const TINT: u64 = 512 << 10;
const COMPOSITOR_BYTES: u64 = 16 << 20;

/// A compositor draw's fragment function: its pipeline state's index.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Quad {
    Plain,
    Masked,
    Backdrop,
}

/// One draw of a compositor frame: its pass, its kind, whether it is indexed, its scissor
/// rectangle and its pass's size.
#[derive(Clone, Copy)]
struct FrameDraw {
    pass: usize,
    quad: Quad,
    indexed: bool,
    scissor: ScissorRect,
    size: (u64, u64),
}

/// The compositor shape's objects (the module docs): its passes with their viewports, their
/// targets, the sampled surfaces, its three pipeline states, its sampler and the buffer of its
/// per-draw data, and the buffers its surfaces lie in.
struct Compositor {
    passes: Vec<(RenderPassDescriptor, Viewport)>,
    targets: Vec<Texture>,
    sources: Vec<Texture>,
    pipelines: Vec<RenderPipelineState>,
    sampler: SamplerState,
    data: Buffer,
    _surfaces: Vec<Buffer>,
}

impl Compositor {
    fn new(device: &Device, library: &Library, pages: bool) -> Result<Compositor> {
        let function = |name: &str| -> Result<StageSource> {
            let function = library.new_function_with_name(name)?;
            Ok(StageSource::Function(function.function_descriptor()))
        };
        let pipelines = [
            "draw_costs_quad",
            "draw_costs_quad_masked",
            "draw_costs_quad_backdrop",
        ]
        .iter()
        .map(|fragment| {
            let mut descriptor = RenderPipelineDescriptor::new(function("draw_costs_quad_vertex")?);
            descriptor.fragment = Some(function(fragment)?);
            let mut color = ColorAttachmentDescriptor::new(PixelFormat::BGRA8Unorm);
            color.blending_enabled = true;
            color.source_rgb_blend_factor = BlendFactor::One;
            color.source_alpha_blend_factor = BlendFactor::One;
            color.destination_rgb_blend_factor = BlendFactor::OneMinusSourceAlpha;
            color.destination_alpha_blend_factor = BlendFactor::OneMinusSourceAlpha;
            descriptor.color_attachments[0] = Some(color);
            Ok(device.new_render_pipeline_state(&descriptor, &PipelineOptions::default())?)
        })
        .collect::<Result<Vec<_>>>()?;
        let mut sampler = SamplerDescriptor::new();
        sampler.min_filter = SamplerMinMagFilter::Linear;
        sampler.mag_filter = SamplerMinMagFilter::Linear;
        let sampler = device.new_sampler_state(&sampler)?;
        let mut surfaces = Vec::new();
        let mut surface = |(width, height): (u64, u64), target: bool| -> Result<Texture> {
            let (row, mut descriptor) = (
                width * 4,
                TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, width, height, false),
            );
            descriptor.usage = match target {
                true => TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ,
                false => TextureUsage::SHADER_READ,
            };
            let buffer = match pages {
                // SAFETY: page-aligned read-write memory of the surface's bytes, never unmapped,
                // which outlives the buffer.
                true => unsafe {
                    device.new_buffer_with_bytes_no_copy(
                        HostMemory::new(host_pages((row * height) as usize)?),
                        row * height,
                        ResourceOptions::shared(),
                        None,
                    )?
                },
                false => {
                    descriptor.resource_options = ResourceOptions::private();
                    device.new_buffer(row * height, ResourceOptions::private())?
                }
            };
            let plane = IoSurfacePlane::new(0, row, false);
            let texture = buffer.new_texture_with_iosurface_plane(&descriptor, plane)?;
            surfaces.push(buffer);
            Ok(texture)
        };
        let targets = (LAYERS.iter().copied().chain([FRAMEBUFFER]))
            .map(|size| surface(size, true))
            .collect::<Result<Vec<_>>>()?;
        let sources = (0..SOURCES)
            .map(|at| match at {
                0 => surface(FRAMEBUFFER, false),
                at => surface(SOURCE_SIZES[(at - 1) % SOURCE_SIZES.len()], false),
            })
            .collect::<Result<Vec<_>>>()?;
        let passes = (targets.iter().zip(LAYERS.iter().chain([&FRAMEBUFFER])))
            .enumerate()
            .map(|(at, (target, &(width, height)))| {
                let mut descriptor = RenderPassDescriptor::new();
                let color = &mut descriptor.color_attachments[0];
                color.texture = Some(target.clone());
                color.store_action = StoreAction::Store;
                match at < LAYERS.len() {
                    true => color.load_action = LoadAction::Load,
                    false => {
                        color.load_action = LoadAction::Clear;
                        color.clear_color = ClearColor::new(0.1, 0.1, 0.1, 1.0);
                    }
                }
                let viewport = Viewport {
                    origin_x: 0.0,
                    origin_y: 0.0,
                    width: width as f64,
                    height: height as f64,
                    znear: 0.0,
                    zfar: 1.0,
                };
                (descriptor, viewport)
            })
            .collect();
        let memory = host_pages(COMPOSITOR_BYTES as usize)?;
        // SAFETY: page-aligned read-write memory of its length, never unmapped, which outlives
        // the buffer.
        let data = unsafe {
            device.new_buffer_with_bytes_no_copy(
                HostMemory::new(memory),
                COMPOSITOR_BYTES,
                ResourceOptions::shared(),
                None,
            )?
        };
        let write = |at: u64, values: &[f32]| {
            let bytes: Vec<u8> = values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect();
            // SAFETY: inside the mapping, which nothing reads yet.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    bytes.as_ptr(),
                    memory.as_ptr().add(at as usize),
                    bytes.len(),
                )
            }
        };
        // A corner is a position in [0, 1] and its texture coordinate, y down for both.
        let [tl, tr, bl, br] =
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0f32]].map(|[x, y]| [x, y, x, y]);
        let plan = Compositor::plan();
        for slot in 0..COMPOSITOR_SLOTS {
            write(CORNERS4 + slot * 64, &[tl, tr, bl, br].concat());
            write(CORNERS6 + slot * 128, &[tl, tr, bl, bl, tr, br].concat());
            write(TINT + slot * 256, &[1.0, 1.0, 1.0, 0.9]);
            let FrameDraw { scissor, size, .. } = plan[slot as usize % plan.len()];
            let (w, h) = (size.0 as f32, size.1 as f32);
            let placement = [
                2.0 * scissor.width as f32 / w,
                -2.0 * scissor.height as f32 / h,
                2.0 * scissor.x as f32 / w - 1.0,
                1.0 - 2.0 * scissor.y as f32 / h,
            ];
            write(PLACEMENT + slot * 256, &placement);
            let indices: Vec<u8> = [0u16, 1, 2, 2, 1, 3]
                .iter()
                .flat_map(|index| index.to_le_bytes())
                .collect();
            // SAFETY: as above.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    indices.as_ptr(),
                    memory.as_ptr().add((INDICES + slot * 64) as usize),
                    indices.len(),
                )
            }
        }
        Ok(Compositor {
            passes,
            targets,
            sources,
            pipelines,
            sampler,
            data,
            _surfaces: surfaces,
        })
    }

    /// A frame's draws in order.
    fn plan() -> Vec<FrameDraw> {
        let mut draws = Vec::with_capacity(COMPOSITOR_DRAWS);
        for pass in 0..=LAYERS.len() {
            let (count, (width, height)) = match pass < LAYERS.len() {
                true => (LAYER_DRAWS, LAYERS[pass]),
                false => (FRAMEBUFFER_DRAWS, FRAMEBUFFER),
            };
            for k in 0..count {
                let d = draws.len();
                let backdrop = (pass < LAYERS.len() && k == 1) || (pass == LAYERS.len() && k == 7);
                let kind = match () {
                    _ if backdrop => Quad::Backdrop,
                    _ if d % 3 == 2 => Quad::Masked,
                    _ => Quad::Plain,
                };
                let scissor = ScissorRect {
                    x: (d * 97 % 8) as u64 * width / 16,
                    y: (d * 53 % 8) as u64 * height / 16,
                    width: width / 2,
                    height: height / 2,
                };
                draws.push(FrameDraw {
                    pass,
                    quad: kind,
                    indexed: matches!(d % 5, 0 | 2 | 3),
                    scissor,
                    size: (width, height),
                });
            }
        }
        draws
    }

    /// Command buffer `buffer`'s steps: its six passes and their draws, each draw's slots the
    /// next of the buffer's.
    fn steps(&self, buffer: usize) -> Vec<Step<'_>> {
        let (mut steps, mut kind, mut open) = (Vec::new(), None, None);
        for (d, draw) in Compositor::plan().into_iter().enumerate() {
            let FrameDraw {
                pass,
                quad,
                indexed,
                scissor,
                ..
            } = draw;
            if open != Some(pass) {
                steps.push(Step::Pass(pass));
                (open, kind) = (Some(pass), None);
            }
            let slot = (buffer * COMPOSITOR_DRAWS + d) as u64 % COMPOSITOR_SLOTS;
            if kind != Some(quad) {
                steps.push(Step::Pipeline(&self.pipelines[quad as usize]));
                kind = Some(quad);
            }
            let corners = match indexed {
                true => CORNERS4 + slot * 64,
                false => CORNERS6 + slot * 128,
            };
            steps.push(Step::VertexBuffer(&self.data, corners, 0));
            steps.push(Step::VertexBuffer(&self.data, PLACEMENT + slot * 256, 1));
            steps.push(Step::FragmentBuffer(&self.data, TINT + slot * 256, 0));
            steps.push(Step::FragmentTexture(
                &self.sources[(d * 5 + buffer) % SOURCES],
                0,
            ));
            match quad {
                Quad::Plain => {}
                Quad::Masked => {
                    let mask = &self.sources[(d * 7 + 3 + buffer) % SOURCES];
                    steps.push(Step::FragmentTexture(mask, 1));
                }
                Quad::Backdrop => steps.push(Step::FragmentTexture(&self.targets[pass], 1)),
            }
            steps.push(Step::FragmentSampler(&self.sampler));
            steps.push(Step::Scissor(scissor));
            steps.push(match indexed {
                true => Step::DrawIndexed(&self.data, INDICES + slot * 64),
                false => Step::Draw(PrimitiveType::Triangle, 6),
            });
        }
        steps
    }
}

/// The calling thread's CPU time, and its instructions and cycles where the system counts them;
/// the process's page faults that needed no I/O and its voluntary context switches, every
/// thread's (`getrusage`): what the kernel did, which the thread's counts hold where it ran on it.
#[derive(Clone, Copy, Default)]
struct Counts {
    nanos: u64,
    instructions: Option<[u64; 2]>,
    process: [u64; 2],
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
        // SAFETY: a `rusage` is plain integers, all zero valid, which the call writes.
        let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
        // SAFETY: `usage` is a `rusage`, which the call writes.
        let answer = unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };
        assert_eq!(answer, 0, "getrusage");
        Counts {
            nanos: at.tv_sec as u64 * 1_000_000_000 + at.tv_nsec as u64,
            instructions: instructions_and_cycles(),
            process: [usage.ru_minflt as u64, usage.ru_nvcsw as u64],
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
            process: [
                self.process[0] - start.process[0],
                self.process[1] - start.process[1],
            ],
        }
    }

    fn plus(self, more: Counts) -> Counts {
        Counts {
            nanos: self.nanos + more.nanos,
            instructions: self
                .instructions
                .zip(more.instructions)
                .map(|(a, b)| [a[0] + b[0], a[1] + b[1]]),
            process: [
                self.process[0] + more.process[0],
                self.process[1] + more.process[1],
            ],
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

/// On Linux, the calling thread's user-space instructions and cycles since its counters opened
/// at its first call (`perf_event_open`, where `kernel.perf_event_paranoid` is at most 2): on a
/// hybrid Intel CPU its performance cores' counters (`cpu_core`), which count only there.
#[cfg(target_os = "linux")]
fn instructions_and_cycles() -> Option<[u64; 2]> {
    use std::cell::Cell;

    /// `struct perf_event_attr` as its first published size (`PERF_ATTR_SIZE_VER0`).
    #[repr(C)]
    struct Attr {
        kind: u32,
        size: u32,
        config: u64,
        sample_period: u64,
        sample_type: u64,
        read_format: u64,
        flags: u64,
        wakeup_events: u32,
        bp_type: u32,
        config1: u64,
    }

    /// `PERF_TYPE_HARDWARE`; `PERF_COUNT_HW_CPU_CYCLES` and `PERF_COUNT_HW_INSTRUCTIONS`; the
    /// `exclude_kernel` and `exclude_hv` flags.
    const HARDWARE: u32 = 0;
    const CYCLES: u64 = 0;
    const INSTRUCTIONS: u64 = 1;
    const EXCLUDE_KERNEL_HV: u64 = (1 << 5) | (1 << 6);

    thread_local! {
        /// The thread's two counters' descriptors, opened at its first read; -1 where one is not.
        static COUNTERS: Cell<Option<[i32; 2]>> = const { Cell::new(None) };
    }

    fn open(event: u64) -> i32 {
        // A hybrid CPU's performance cores' PMU, in the config's upper half; none elsewhere.
        let pmu = std::fs::read_to_string("/sys/bus/event_source/devices/cpu_core/type")
            .ok()
            .and_then(|text| text.trim().parse::<u64>().ok())
            .unwrap_or(0);
        let attr = Attr {
            kind: HARDWARE,
            size: size_of::<Attr>() as u32,
            config: event | (pmu << 32),
            sample_period: 0,
            sample_type: 0,
            read_format: 0,
            flags: EXCLUDE_KERNEL_HV,
            wakeup_events: 0,
            bp_type: 0,
            config1: 0,
        };
        // SAFETY: `attr` is a valid `perf_event_attr` of its stated size, live for the call; the
        // calling thread (0), any CPU (-1), no group (-1), no flags.
        let fd = unsafe {
            libc::syscall(
                libc::SYS_perf_event_open,
                &attr as *const Attr,
                0,
                -1,
                -1,
                0,
            )
        };
        i32::try_from(fd).unwrap_or(-1)
    }

    fn read(fd: i32) -> Option<u64> {
        let mut count = 0u64;
        // SAFETY: `count` is 8 writable bytes, the size of the counter's value.
        let read = unsafe { libc::read(fd, (&mut count as *mut u64).cast(), 8) };
        (read == 8).then_some(count)
    }

    let [instructions, cycles] = COUNTERS.with(|cell| match cell.get() {
        Some(fds) => fds,
        None => {
            let fds = [open(INSTRUCTIONS), open(CYCLES)];
            cell.set(Some(fds));
            fds
        }
    });
    Some([read(instructions)?, read(cycles)?])
}

/// Elsewhere the CPU time alone.
#[cfg(not(any(target_os = "macos", target_os = "linux")))]
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

/// How a run is made beside its shape: `--calls`, `--counts`, `--heap` and `--threads`.
#[derive(Clone, Copy)]
struct Options {
    calls: bool,
    counts: bool,
    heap: bool,
    threads: usize,
}

/// One thread's run of its command buffers' `buffers`: its counted repetitions' counts together,
/// and its calls' tally where `calls`, each repetition started with the other threads' at
/// `start`.
fn thread_run(
    bench: &Bench,
    buffers: &[Vec<Step<'_>>],
    calls: bool,
    start: &Barrier,
) -> Result<(Counts, Tally)> {
    // On a Mac, the performance cores, as the main thread asks for them.
    #[cfg(target_os = "macos")]
    // SAFETY: sets the calling thread's own class; it takes no pointer.
    unsafe {
        libc::pthread_set_qos_class_self_np(libc::qos_class_t::QOS_CLASS_USER_INTERACTIVE, 0);
    }
    let mut total: Option<Counts> = None;
    let mut tally: Tally = [(0, 0); CALLS.len()];
    for repetition in 0..WARM + COUNTED {
        let counted = repetition >= WARM;
        start.wait();
        let spent = bench.repetition(buffers, (calls && counted).then_some(&mut tally))?;
        if counted {
            total = Some(total.map_or(spent, |total| total.plus(spent)));
        }
    }
    Ok((total.unwrap_or_default(), tally))
}

/// Run `shape` `runs` times and print its figures a draw, and with `--calls` each call's.
fn measure(bench: &Bench, shape: Shape, runs: usize, options: Options) -> Result<()> {
    let Options {
        calls,
        counts,
        heap,
        threads,
    } = options;
    let chunks = bench.chunks(shape, threads, heap)?;
    let before = counts.then(|| bench.device.work_counts());
    // Each thread's command buffers, the chunks of thread `t`'s from its `t * shape.buffers`th.
    let buffers: Vec<Vec<Vec<Step<'_>>>> = (0..threads)
        .map(|thread| {
            let first = thread * shape.buffers;
            (first..first + shape.buffers)
                .map(|buffer| bench.steps(&chunks.buffers, shape, buffer))
                .collect()
        })
        .collect();
    let draws = (COUNTED * threads * shape.buffers * shape.draws) as f64;
    let mut seen = Vec::with_capacity(runs);
    let mut tallies = Vec::with_capacity(runs);
    for _ in 0..runs {
        let start = Barrier::new(threads);
        let ran: Vec<Result<(Counts, Tally)>> = match threads {
            1 => vec![thread_run(bench, &buffers[0], calls, &start)],
            _ => std::thread::scope(|scope| {
                // Each thread's refusal as its text, which crosses threads.
                let running: Vec<_> = (buffers.iter())
                    .map(|own| {
                        let start = &start;
                        scope.spawn(move || {
                            thread_run(bench, own, calls, start).map_err(|error| error.to_string())
                        })
                    })
                    .collect();
                (running.into_iter())
                    .map(|thread| match thread.join() {
                        Ok(ran) => ran.map_err(Into::into),
                        Err(_) => Err("a thread panicked".into()),
                    })
                    .collect()
            }),
        };
        let mut total: Option<Counts> = None;
        let mut tally: Tally = [(0, 0); CALLS.len()];
        for answer in ran {
            let (spent, own) = answer?;
            total = Some(total.map_or(spent, |total| total.plus(spent)));
            for (all, one) in tally.iter_mut().zip(own) {
                *all = (all.0 + one.0, all.1 + one.1);
            }
        }
        // The process's counts each thread took over the same span: theirs once.
        let mut total = total.unwrap_or_default();
        total.process = total.process.map(|count| count / threads as u64);
        seen.push(total);
        tallies.push(tally);
    }
    if let Some(before) = before {
        let drawn = (WARM + COUNTED) * runs * threads * shape.buffers * shape.draws;
        println!("WORK-COUNTS {} draws {drawn}", shape.name);
        println!("WORK-COUNTS {} before {before:?}", shape.name);
        println!(
            "WORK-COUNTS {} after {:?}",
            shape.name,
            bench.device.work_counts()
        );
    }
    println!(
        "{}: {} command buffers of {} draws on {threads} thread(s), chunks in {}, {runs} runs of \
         {COUNTED} counted repetitions; a draw:",
        shape.name,
        shape.buffers,
        shape.draws,
        match heap {
            true => "a heap",
            false => "buffers",
        }
    );
    let mut figures: Vec<(&str, Vec<f64>)> = vec![(
        "cpu_ns",
        seen.iter().map(|c| c.nanos as f64 / draws).collect(),
    )];
    for (figure, index) in [("process_faults", 0), ("process_switches", 1)] {
        let values = seen
            .iter()
            .map(|c| c.process[index] as f64 / draws)
            .collect();
        figures.push((figure, values));
    }
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
            "DRAW-COST {} {figure} {median:.3} [{low:.3}, {high:.3}]",
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

/// `bytes` bytes of fresh host pages, read-write, never unmapped: private, as RADV refuses to
/// import shared anonymous memory (`host_memory.rs`'s `import_refused`).
fn host_pages(bytes: usize) -> Result<NonNull<u8>> {
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
    if pages == libc::MAP_FAILED {
        return Err(format!("mmap: {}", std::io::Error::last_os_error()).into());
    }
    Ok(NonNull::new(pages.cast::<u8>()).ok_or("a null mapping")?)
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
    let (mut runs, mut metallib, mut shadowed, mut pages) = (RUNS, None, false, false);
    let mut options = Options {
        calls: false,
        counts: false,
        heap: false,
        threads: 1,
    };
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--runs" => runs = args.next().ok_or("--runs N")?.parse()?,
            "--calls" => options.calls = true,
            "--counts" => options.counts = true,
            "--heap" => options.heap = true,
            "--shadowed" => shadowed = true,
            "--pages" => pages = true,
            "--threads" => {
                options.threads = args.next().ok_or("--threads N")?.parse()?;
                if options.threads == 0 {
                    return Err("--threads N: at least 1".into());
                }
            }
            "--metallib" => metallib = Some(PathBuf::from(args.next().ok_or("--metallib PATH")?)),
            name => shapes.push(*SHAPES.iter().find(|shape| shape.name == name).ok_or_else(
                || format!("no shape {name}: ws, many, 10k, clips, damage or compositor"),
            )?),
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
    let mut device_options = DeviceOptions::default();
    device_options.gpu_times_from_host = true;
    if shadowed {
        device_options.policy.host_page_textures = HostPageTexturePolicy::Shadow;
    }
    // With the shadow log on, its lines and each notice's text on stderr.
    let log = metal2vulkan::env_vars::log_shadows();
    device_options.messages = Some(MessageSink::new(move |message| {
        if message.degradation.is_some() {
            NOTICES.fetch_add(1, Ordering::Relaxed);
        }
        if log && (message.degradation.is_some() || message.id_name == "metal2vulkan-shadows") {
            eprintln!("{}: {}", message.id_name, message.text);
        }
    }));
    let device = Device::system_default_with(&device_options)?;
    println!("device: {}", device.name());
    let library = library(&device, metallib.as_deref())?;
    let compositor = (shapes.iter())
        .any(|shape| shape.stream == Stream::Composited)
        .then_some(pages);
    let bench = Bench::new(device, &library, shadowed, compositor)?;
    for shape in shapes {
        measure(&bench, shape, runs, options)?;
    }
    println!("notices heard: {}", NOTICES.load(Ordering::Relaxed));
    Ok(())
}
