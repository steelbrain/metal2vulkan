//! What a window server's composite of many render passes into one target costs the thread that
//! encodes it, on the device the cfg selects, without the validation layer: report only, in
//! release builds, to weigh a render pass's opening and end, and the binds and draws inside it,
//! against a bare Vulkan loop of the same commands on the same device.
//!
//! The workload is a guest window server's Mission Control composite as the consumer (macOS
//! guests over the runtime) measured it: one command buffer of [`PASSES`] render passes, each an
//! encoder of its own into one `BGRA8Unorm` target of [`TARGET_SIDE`] square, the first clearing
//! it and every later one loading it, each storing it. A pass sets its pipeline state and sampler,
//! vertex buffer 1 and fragment buffer 0, then draws [`DRAWS`] layers: each a fragment texture
//! (one of [`TEXTURES`]), then a fragment buffer and a vertex buffer at offsets of the layer's own
//! in one 64 KiB chunk of host pages (a no-copy buffer over them), then a 4-vertex triangle strip
//! (`zwd_quad`, `zwd_masked`, which read vertex and fragment buffer 0, fragment texture 0 and
//! sampler 0; vertex buffer 1 is bound and read by nothing). Every [`RESET`]th draw of the frame
//! but a pass's first sets the pipeline state and sampler again before its binds. The frame waits
//! for its command buffer at its end, outside the times below.
//!
//! - **`passes`**: the composite as above.
//! - **`one pass`**: the same calls in one encoder: one opening and one end where `passes` has
//!   [`PASSES`] of each, so the two frames' difference over `PASSES - 1` is what a pass's opening
//!   and end add, its effect on the draws after it included.
//! - **`barriers`**: `one pass` with a memory barrier over buffers, textures and render targets
//!   from the vertex and fragment stages to both before each pass's calls but the first: the
//!   encoder's own order of everything before the barrier ahead of everything after it, as an
//!   encoder's end orders it ahead of the next encoder's.
//! - **`bare sets`** and **`bare pushed`**: the same frame in plain Vulkan on the runtime's own
//!   device, through the driver's commands (loaded with `vkGetDeviceProcAddr`, not the runtime's
//!   counted ones): per pass a `vkCmdBeginRenderPass` (one render pass object clearing, one
//!   loading, both storing, with a dependency from the earlier passes' color writes), the
//!   pipeline, viewport and scissor; per draw its texture and its two buffer ranges, bound as one
//!   of [`TEXTURES`] descriptor sets made once with two dynamic offsets (`sets`) or pushed as
//!   three descriptor writes (`pushed`, where the device has `vkCmdPushDescriptorSetKHR`), and a
//!   `vkCmdDraw` of 4 vertices; the pipeline bound again where the runtime's frame sets it again;
//!   then `vkCmdEndRenderPass`. The SPIR-V is the owned synthetic fixtures under
//!   `fixtures/public/bare/`.
//!
//! Each round measures every shape in turn on one device: [`WARM`] frames untimed, then
//! [`TIMED`] frames printed as one `COMPOSITE` line each shape: the medians of the encoding
//! thread's CPU time and wall time a frame (from the first opening through the commit's return,
//! or for the bare loop through `vkEndCommandBuffer`), of a pass's opening and end together (and
//! each alone), and of a pass's binds and draws a call (`passes` and `one pass`) and a draw; then
//! [`COUNTED`] frames of their own, where the kernel lets a thread count its own instructions
//! (Linux), give the encoding thread's user-space instructions a pass's opening and end, a call
//! and a draw take on average. A `COMPOSITE-PASS` line then gives the difference of `passes` and
//! `one pass` per pass, and a `COMPOSITE-COUNTS` line per runtime shape the work counts a frame
//! moved (`Device::work_counts`, every count that moved).
//!
//! `composite_costs` measures [`ROUNDS`] rounds; `composite_costs_passes_alone` and
//! `composite_costs_bare_alone` encode [`PROFILED`] frames of `passes` and `bare sets` alone, in
//! a process of their own, for a profile of each (`perf record -g` of the test's process).
//!
//! Ignored by default: a measurement, not a check. Run it by name with `--run-ignored all`, in a
//! build with `--cfg m2v_system_alloc`, whose lib test binary keeps the system allocator as a
//! consumer's process does.

mod bare;

use super::frame_costs::{instructions, sampled, shape, step, thread_cpu, Step, TEXTURES};
use super::host_memory::{host_page, no_copy, Mapping};
use super::{loader, open_device, ScratchCache, WAIT};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::library::{FunctionConstantValue, FunctionConstantValues};
use metal2vulkan::metal::{
    BarrierScope, BlendFactor, DataType, PixelFormat, PrimitiveType, RenderStages,
    SamplerMinMagFilter, TextureUsage,
};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    Buffer, ClearColor, CommandBufferStatus, CommandQueue, Device, DeviceOptions, Library,
    LoadAction, PipelineOptions, RenderCommandEncoder, RenderPassDescriptor, RenderPipelineState,
    ResourceOptions, SamplerState, ShaderCacheOptions, StoreAction, Texture, TextureDescriptor,
    WorkCounts,
};
use std::collections::BTreeMap;
use std::time::{Duration, Instant};

/// The untimed and the timed frames of a shape in a round, the frames whose instructions it
/// counts, the rounds, and the frames a profiling run encodes.
const WARM: usize = 20;
const TIMED: usize = 100;
const COUNTED: usize = 20;
const ROUNDS: usize = 3;
const PROFILED: usize = 3000;
/// The passes of a frame, and the draws of each.
const PASSES: usize = 92;
const DRAWS: usize = 3;
/// Every `RESET`th draw of the frame but a pass's first sets the pipeline state and sampler again.
const RESET: usize = 4;
/// The target's side.
const TARGET_SIDE: u64 = 512;
/// The host chunk's bytes, and the bytes each draw's buffers take in it: the vertex buffer's at
/// the slot's start, the fragment buffer's 256 bytes past it (each a `ZwdShape`), whole multiples
/// of 256 as constant buffer offsets want.
const CHUNK: usize = 64 << 10;
const SLOT: usize = 512;

/// A frame's shape through the runtime.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Shape {
    Passes,
    OnePass,
    Barriers,
}

impl Shape {
    fn label(self) -> &'static str {
        match self {
            Shape::Passes => "passes",
            Shape::OnePass => "one pass",
            Shape::Barriers => "barriers",
        }
    }
}

/// One frame's steps: each encoder's opening and end, and each pass's binds and draws with the
/// calls they made; the frame's CPU and wall time.
#[derive(Default)]
struct Frame {
    open: Vec<Step>,
    ops: Vec<(Step, usize)>,
    end: Vec<Step>,
    cpu: Duration,
    wall: Duration,
}

/// The device, its queue, and what the frames draw with.
struct Composite {
    device: Device,
    queue: CommandQueue,
    pipeline: RenderPipelineState,
    sampler: SamplerState,
    textures: Vec<Texture>,
    target: Texture,
    /// The no-copy buffer over the chunk, and the chunk's pages, which outlive it (declared after).
    chunk: Buffer,
    _pages: Mapping,
}

/// A function of `library`, its constant 0 (`zwd_variant`) 0.
fn function(library: &Library, name: &str) -> StageSource {
    let constants = FunctionConstantValues::defaults().with_index(
        0,
        FunctionConstantValue::new(DataType::UInt, 0u32.to_le_bytes().to_vec()),
    );
    StageSource::Function(
        library
            .new_function_with_name_constant_values(name, &constants)
            .unwrap()
            .function_descriptor(),
    )
}

impl Composite {
    fn new(cache: &ScratchCache) -> Composite {
        let mut options = DeviceOptions::default();
        options.shader_cache = ShaderCacheOptions::in_directory(cache.path());
        options.loader = loader();
        options.gpu_times_from_host = true;
        let device = open_device(TEST_DEVICE, options).unwrap();
        let zwd = device
            .new_library_with_air(&llvm_assemble(&engine_dir().join("zwd_shapes.ll")).unwrap())
            .unwrap();
        let mut descriptor = RenderPipelineDescriptor::new(function(&zwd, "zwd_quad"));
        descriptor.fragment = Some(function(&zwd, "zwd_masked"));
        let mut color = ColorAttachmentDescriptor::new(PixelFormat::BGRA8Unorm);
        color.blending_enabled = true;
        color.source_rgb_blend_factor = BlendFactor::One;
        color.source_alpha_blend_factor = BlendFactor::One;
        color.destination_rgb_blend_factor = BlendFactor::OneMinusSourceAlpha;
        color.destination_alpha_blend_factor = BlendFactor::OneMinusSourceAlpha;
        descriptor.color_attachments[0] = Some(color);
        let pipeline = device
            .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
            .unwrap();
        let mut sampler = SamplerDescriptor::new();
        sampler.min_filter = SamplerMinMagFilter::Linear;
        sampler.mag_filter = SamplerMinMagFilter::Linear;
        let sampler = device.new_sampler_state(&sampler).unwrap();
        let textures = (0..TEXTURES)
            .map(|n| sampled(&device, [n as u8 * 16, 128, 255 - n as u8 * 16, 255]))
            .collect();
        let mut target =
            TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, TARGET_SIDE, TARGET_SIDE, false);
        target.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
        target.resource_options = ResourceOptions::private();
        let target = device.new_texture(&target).unwrap();
        let pages = Mapping::anonymous(CHUNK.next_multiple_of(host_page()));
        for slot in 0..CHUNK / SLOT {
            let bytes = shape(slot);
            pages.write(slot * SLOT, &bytes);
            pages.write(slot * SLOT + 256, &bytes);
        }
        // SAFETY: the pages are this process's, mapped read and write, and outlive the buffer
        // (`Composite`'s order).
        let chunk = unsafe {
            no_copy(
                &device,
                pages.pointer(),
                CHUNK,
                ResourceOptions::shared(),
                None,
            )
        }
        .unwrap();
        Composite {
            queue: device.new_command_queue().unwrap(),
            pipeline,
            sampler,
            textures,
            target,
            chunk,
            _pages: pages,
            device,
        }
    }

    /// The descriptor of a pass: the target cleared where `first`, else loaded; stored.
    fn descriptor(&self, first: bool) -> RenderPassDescriptor {
        let mut descriptor = RenderPassDescriptor::new();
        let color = &mut descriptor.color_attachments[0];
        color.texture = Some(self.target.clone());
        color.load_action = match first {
            true => LoadAction::Clear,
            false => LoadAction::Load,
        };
        color.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
        color.store_action = StoreAction::Store;
        descriptor
    }

    /// One pass's binds and draws into `encoder`, the frame's draws so far `n`: the calls made.
    fn pass_ops(&self, encoder: &mut RenderCommandEncoder, n: &mut usize) -> usize {
        encoder.set_render_pipeline_state(&self.pipeline).unwrap();
        encoder
            .set_fragment_sampler_state(Some(&self.sampler), 0)
            .unwrap();
        encoder.set_vertex_buffer(Some(&self.chunk), 0, 1).unwrap();
        encoder
            .set_fragment_buffer(Some(&self.chunk), 0, 0)
            .unwrap();
        let mut calls = 4;
        for draw in 0..DRAWS {
            if draw > 0 && (*n).is_multiple_of(RESET) {
                encoder.set_render_pipeline_state(&self.pipeline).unwrap();
                encoder
                    .set_fragment_sampler_state(Some(&self.sampler), 0)
                    .unwrap();
                calls += 2;
            }
            let slot = (*n % (CHUNK / SLOT)) * SLOT;
            encoder
                .set_fragment_texture(Some(&self.textures[*n % TEXTURES]), 0)
                .unwrap();
            encoder
                .set_fragment_buffer(Some(&self.chunk), (slot + 256) as u64, 0)
                .unwrap();
            encoder
                .set_vertex_buffer(Some(&self.chunk), slot as u64, 0)
                .unwrap();
            encoder
                .draw_primitives(PrimitiveType::TriangleStrip, 0, 4, 1, 0)
                .unwrap();
            calls += 4;
            *n += 1;
        }
        calls
    }

    /// Encode and commit one frame of `shape`, counting the encoding thread's instructions in
    /// each step where `counted`, then wait for it.
    fn frame(&self, shape: Shape, counted: bool) -> Frame {
        let mut frame = Frame::default();
        let (encoders, passes) = match shape {
            Shape::Passes => (PASSES, 1),
            Shape::OnePass | Shape::Barriers => (1, PASSES),
        };
        let command_buffer = self.queue.command_buffer().unwrap();
        let cpu = thread_cpu();
        let start = Instant::now();
        let mut n = 0;
        for encoder in 0..encoders {
            let descriptor = self.descriptor(encoder == 0);
            let (mut encoder, open) = step(counted, || {
                command_buffer.render_command_encoder(&descriptor).unwrap()
            });
            frame.open.push(open);
            for pass in 0..passes {
                let barrier = shape == Shape::Barriers && pass > 0;
                let (calls, ops) = step(counted, || {
                    if barrier {
                        encoder.memory_barrier_with_scope(
                            BarrierScope::BUFFERS
                                | BarrierScope::TEXTURES
                                | BarrierScope::RENDER_TARGETS,
                            RenderStages::VERTEX | RenderStages::FRAGMENT,
                            RenderStages::VERTEX | RenderStages::FRAGMENT,
                        );
                    }
                    self.pass_ops(&mut encoder, &mut n) + usize::from(barrier)
                });
                frame.ops.push((ops, calls));
            }
            let ((), end) = step(counted, || encoder.end_encoding().unwrap());
            frame.end.push(end);
        }
        command_buffer.commit().unwrap();
        frame.wall = start.elapsed();
        frame.cpu = thread_cpu().saturating_sub(cpu);
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        frame
    }
}

/// The median of `values`.
fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

/// Microseconds of `duration`.
fn micros(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1e6
}

/// What a shape's timed and counted frames measured, printed as its `COMPOSITE` line.
struct Measured {
    /// The medians of a frame's CPU and wall time, in milliseconds.
    frame_cpu: f64,
    frame_wall: f64,
    /// The medians of a pass's opening and end together, and each alone, in microseconds.
    open_end: f64,
    open: f64,
    end: f64,
    /// The medians of a call and a draw of a pass's binds and draws, in nanoseconds: a call of
    /// the encoder's, or for the bare loop a Vulkan command.
    call: f64,
    draw: f64,
    /// The mean instructions of a pass's opening and end, a call and a draw, where counted.
    instructions: Option<[f64; 3]>,
}

impl Measured {
    fn print(&self, device: &str, label: &str) {
        let instructions = match self.instructions {
            Some([pass, call, draw]) => format!(
                "; instructions: a pass's opening and end {pass:.0}, a call {call:.0}, a draw \
                 {draw:.0}"
            ),
            None => String::new(),
        };
        println!(
            "COMPOSITE {device}: {label}: a frame {:.3} ms of CPU, {:.3} ms wall; a pass's \
             opening and end {:.2} us (opening {:.2}, end {:.2}); its binds and draws {:.0} ns a \
             draw, {:.0} ns a call{instructions}",
            self.frame_cpu,
            self.frame_wall,
            self.open_end,
            self.open,
            self.end,
            self.draw,
            self.call,
        );
    }
}

/// The work counts that moved from `before` to `after`, by their names (nested ones joined with
/// a dot), read out of the derived `Debug`: every leaf that parses as a count.
fn moved(before: &WorkCounts, after: &WorkCounts) -> BTreeMap<String, u64> {
    fn leaves(counts: &WorkCounts) -> BTreeMap<String, u64> {
        let mut path: Vec<String> = Vec::new();
        let mut read = BTreeMap::new();
        for line in format!("{counts:#?}").lines() {
            let body = line.trim();
            if body.starts_with('}') || body.starts_with(']') {
                path.pop();
                continue;
            }
            let Some((name, value)) = body.split_once(':') else {
                if body.ends_with('{') || body.ends_with('[') {
                    path.push(String::new());
                }
                continue;
            };
            let value = value.trim().trim_end_matches(',');
            if value.ends_with('{') || value.ends_with('[') {
                path.push(name.trim().to_owned());
                continue;
            }
            if let Ok(count) = value.parse::<u64>() {
                let mut full: Vec<&str> = path.iter().map(String::as_str).collect();
                full.push(name.trim());
                let full: Vec<&str> = full.into_iter().filter(|part| !part.is_empty()).collect();
                read.insert(full.join("."), count);
            }
        }
        read
    }
    let before = leaves(before);
    leaves(after)
        .into_iter()
        .filter_map(|(name, count)| {
            let was = before.get(&name).copied().unwrap_or(0);
            (count > was).then(|| (name, count - was))
        })
        .collect()
}

/// What `frames` measured, and `counted`, frames of the same shape whose instructions were
/// counted, where they were.
fn summarize(frames: &[Frame], counted: Option<&[Frame]>) -> Measured {
    let instructions = counted.map(|counted| {
        let sum = |steps: &mut dyn Iterator<Item = &Step>| -> f64 {
            steps.map(|step| step.instructions.unwrap() as f64).sum()
        };
        let pass = sum(&mut counted.iter().flat_map(|f| f.open.iter().chain(&f.end)));
        let ops = sum(&mut counted.iter().flat_map(|f| f.ops.iter().map(|(s, _)| s)));
        let calls: usize = counted.iter().flat_map(|f| &f.ops).map(|(_, c)| c).sum();
        let encoders = counted.iter().map(|f| f.open.len()).sum::<usize>() as f64;
        let draws = (counted.len() * PASSES * DRAWS) as f64;
        [pass / encoders, ops / calls as f64, ops / draws]
    });
    let each = |of: fn(&Frame) -> &Vec<Step>| -> Vec<f64> {
        (frames.iter())
            .flat_map(|frame| of(frame).iter().map(|step| micros(step.wall)))
            .collect()
    };
    let open_end = (frames.iter())
        .flat_map(|frame| (frame.open.iter().zip(&frame.end)).map(|(o, e)| micros(o.wall + e.wall)))
        .collect();
    let ops = |each: &dyn Fn(usize) -> f64| -> f64 {
        median(
            (frames.iter())
                .flat_map(|frame| &frame.ops)
                .map(|(step, calls)| micros(step.wall) * 1e3 / each(*calls))
                .collect(),
        )
    };
    Measured {
        frame_cpu: median(frames.iter().map(|f| f.cpu.as_secs_f64() * 1e3).collect()),
        frame_wall: median(frames.iter().map(|f| f.wall.as_secs_f64() * 1e3).collect()),
        open_end: median(open_end),
        open: median(each(|f| &f.open)),
        end: median(each(|f| &f.end)),
        call: ops(&|calls| calls as f64),
        draw: ops(&|_| DRAWS as f64),
        instructions,
    }
}

impl Composite {
    /// One shape's timed and counted frames, warmed first.
    fn measure(&self, shape: Shape) -> Measured {
        for _ in 0..WARM {
            self.frame(shape, false);
        }
        let frames: Vec<Frame> = (0..TIMED).map(|_| self.frame(shape, false)).collect();
        let counted = instructions::read().is_some().then(|| {
            (0..COUNTED)
                .map(|_| self.frame(shape, true))
                .collect::<Vec<_>>()
        });
        summarize(&frames, counted.as_deref())
    }

    /// The work counts one frame of `shape` moves, once warm.
    fn counts(&self, shape: Shape) -> BTreeMap<String, u64> {
        self.frame(shape, false);
        let before = self.device.work_counts();
        self.frame(shape, false);
        moved(&before, &self.device.work_counts())
    }
}

/// [`ROUNDS`] rounds of every shape, printed (the module docs).
fn measure() {
    let cache = ScratchCache::new().unwrap();
    let composite = Composite::new(&cache);
    let name = composite.device.name().to_owned();
    let bare = bare::Bare::new(&composite);
    for round in 0..ROUNDS {
        println!("COMPOSITE {name}: round {}", round + 1);
        let passes = composite.measure(Shape::Passes);
        passes.print(&name, Shape::Passes.label());
        let one = composite.measure(Shape::OnePass);
        one.print(&name, Shape::OnePass.label());
        let barriers = composite.measure(Shape::Barriers);
        barriers.print(&name, Shape::Barriers.label());
        println!(
            "COMPOSITE-PASS {name}: (passes - one pass) / {}: {:.2} us of CPU, {:.2} us wall a \
             pass",
            PASSES - 1,
            (passes.frame_cpu - one.frame_cpu) * 1e3 / (PASSES - 1) as f64,
            (passes.frame_wall - one.frame_wall) * 1e3 / (PASSES - 1) as f64,
        );
        println!(
            "COMPOSITE-PASS {name}: (barriers - one pass) / {}: {:.2} us of CPU, {:.2} us wall a \
             barrier",
            PASSES - 1,
            (barriers.frame_cpu - one.frame_cpu) * 1e3 / (PASSES - 1) as f64,
            (barriers.frame_wall - one.frame_wall) * 1e3 / (PASSES - 1) as f64,
        );
        for pushed in [false, true] {
            if let Some(measured) = bare.measure(pushed) {
                measured.print(&name, bare::label(pushed));
            }
        }
    }
    for shape in [Shape::Passes, Shape::OnePass, Shape::Barriers] {
        let counts = composite.counts(shape);
        let text: Vec<String> = counts
            .iter()
            .map(|(name, count)| format!("{name}={count}"))
            .collect();
        println!(
            "COMPOSITE-COUNTS {name}: {}: {}",
            shape.label(),
            text.join(" ")
        );
    }
}

#[test]
#[ignore = "a measurement, not a check: run by name with --run-ignored all"]
fn composite_costs() {
    measure();
}

#[test]
#[ignore = "a measurement, not a check: run by name with --run-ignored all"]
fn composite_costs_passes_alone() {
    let cache = ScratchCache::new().unwrap();
    let composite = Composite::new(&cache);
    for _ in 0..PROFILED {
        composite.frame(Shape::Passes, false);
    }
}

#[test]
#[ignore = "a measurement, not a check: run by name with --run-ignored all"]
fn composite_costs_bare_alone() {
    let cache = ScratchCache::new().unwrap();
    let composite = Composite::new(&cache);
    let bare = bare::Bare::new(&composite);
    for _ in 0..PROFILED {
        bare.frame(false, false);
    }
}
