//! Not a consumer's example: what a frame of indexed draws costs the GPU where a compute dispatch
//! before each draw writes that draw's indices into a buffer of its own, as a layer over another
//! graphics API that fixes up each draw's index buffer does, beside the same frame with its indices
//! written once before.
//!
//! ```sh
//! cargo run --release -p metal2vulkan-runtime --example fixed_index_draws -- \
//!     [--frames F] [--draws N] [--group G] [--pass-draws R] [--cb-passes Q] [--indices I] \
//!     [--side S] [--apart] [--only SHAPE,...]
//! ```
//!
//! A frame is `N` draws (1,280 by default) of `I` indices each (1,536: 512 triangles), split into
//! groups of `G` draws (256). A group's first command buffer opens with a serial compute encoder of
//! `G` dispatches of the owned fixture's `fix_indices`
//! (`validation/fixtures/public/examples/fixed_indices.metal`), one per draw: each binds the
//! frame's one `Private` source of 16-bit indices at the draw's place, the draw's own `Private`
//! buffer of `I` 32-bit indices, and the draw's place and count as bytes, and writes the widened
//! indices into the draw's buffer, one thread an index in threadgroups of 64. Then the group's
//! draws, `R` (26) to a render encoder, `Q` (4) render encoders to a command buffer: each pass loads
//! and stores one `Private` `RGBA8Unorm` target of `S` (1024) texels a side (`--apart`: not loaded,
//! so its draws record in a secondary command buffer the pass executes at its end), and each draw
//! draws its triangles from its buffer through `placed_vertex` and `flat_fragment`, every triangle
//! a few texels wide. The defaults are the shape a GL benchmark's frames took through Apple's
//! GL-on-Metal layer on the runtime (each second: about 470 compute encoders of about 256
//! dispatches, 4,650 render passes of about 26 indexed draws, 1,270 command buffers).
//!
//! Two frames are in flight, each with its own draws' buffers, on one queue. The shapes:
//!
//! - **`fixed`**: the frame as above;
//! - **`prefixed`**: the same command buffers and passes without the compute encoders: each draw
//!   reads the indices one `fixed` frame wrote before the timed ones;
//! - **`dispatches`**: the compute encoders alone, each in a command buffer of its own.
//!
//! Each frame's GPU time is the union of its command buffers' spans from `gpu_start_time` to
//! `gpu_end_time` (`busy`), and its span from its first command buffer's start to its last's end;
//! each line is the median over `F` frames (60) after 10 not counted, with the runtime's counts per
//! frame that moved (`WorkCounts`, by name from their `Debug` form). Then each draw's indices of
//! the last frame are read back and checked against the source's.
//!
//! ```text
//! FIXED <shape> draws <N> group <G> pass_draws <R> cb_passes <Q> indices <I> gpu_ms <busy>
//!     least <least busy> span_ms <span> us_per_draw <busy / N> <device>
//! FIXED-COUNTS <shape> <name>=<per frame> ...
//! ```
//!
//! Measured on an RTX 5080 Laptop GPU (release, the defaults, 60 frames): `fixed` 0.526 ms of GPU
//! busy a frame, `prefixed` 0.404 ms, `dispatches` 0.116 ms (0.09 us a dispatch); with `--apart`
//! 0.555, 0.455 and 0.116 ms. So the compute encoders cost their dispatches' own time and the
//! frame's 5 compute-to-render transitions next to nothing: no barrier records between the
//! dispatches (each writes a buffer of its own), and 15 `vkCmdPipelineBarrier` a frame.

use metal2vulkan::metal::{IndexType, PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_runtime::{
    Buffer, CommandBuffer, CommandBufferStatus, CommandQueue, ComputePipelineState, Device,
    DeviceOptions, LoadAction, PipelineOptions, RenderPassDescriptor, RenderPipelineState,
    ResourceOptions, Size, StoreAction, TextureDescriptor, WorkCounts,
};
use std::collections::{BTreeMap, VecDeque};
use std::process::Command;
use std::time::Duration;

/// The example's result: every runtime refusal is a typed `metal2vulkan_runtime::Error`.
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// The longest the example waits for a command buffer.
const WAIT: Duration = Duration::from_secs(60);

/// The threads of a `fix_indices` threadgroup.
const THREADS: u64 = 64;

/// Frames run before the timed ones.
const WARM: usize = 10;

/// Frames in flight.
const IN_FLIGHT: usize = 2;

/// The vertices the source's indices name: triangles of three in a row, each a few texels wide.
const VERTICES: u32 = 3 * 4096;

/// The shapes, in the order they run.
const SHAPES: [&str; 3] = ["fixed", "prefixed", "dispatches"];

/// The run's shape (the module docs).
struct Shape {
    frames: usize,
    draws: u32,
    group: u32,
    pass_draws: u32,
    cb_passes: u32,
    indices: u32,
    side: u64,
    apart: bool,
    only: Option<Vec<String>>,
}

impl Shape {
    fn from_args() -> Result<Shape> {
        let mut shape = Shape {
            frames: 60,
            draws: 1280,
            group: 256,
            pass_draws: 26,
            cb_passes: 4,
            indices: 1536,
            side: 1024,
            apart: false,
            only: None,
        };
        let mut args = std::env::args().skip(1);
        while let Some(flag) = args.next() {
            if flag == "--apart" {
                shape.apart = true;
                continue;
            }
            let value = args.next().ok_or(format!("{flag} takes a value"))?;
            match flag.as_str() {
                "--frames" => shape.frames = value.parse()?,
                "--draws" => shape.draws = value.parse()?,
                "--group" => shape.group = value.parse()?,
                "--pass-draws" => shape.pass_draws = value.parse()?,
                "--cb-passes" => shape.cb_passes = value.parse()?,
                "--indices" => shape.indices = value.parse()?,
                "--side" => shape.side = value.parse()?,
                "--only" => shape.only = Some(value.split(',').map(str::to_owned).collect()),
                _ => return Err(format!("unknown flag {flag}").into()),
            }
        }
        if shape.frames == 0 || shape.draws == 0 || shape.group == 0 || shape.pass_draws == 0 {
            return Err("--frames, --draws, --group and --pass-draws take 1 or more".into());
        }
        if shape.cb_passes == 0 || shape.indices % 3 != 0 || shape.indices == 0 {
            return Err("--cb-passes takes 1 or more, --indices a multiple of 3".into());
        }
        if shape.indices > VERTICES {
            return Err(format!("--indices takes at most {VERTICES}").into());
        }
        Ok(shape)
    }

    fn runs(&self, name: &str) -> bool {
        self.only
            .as_ref()
            .is_none_or(|only| only.iter().any(|shape| shape == name))
    }
}

/// The owned fixture's functions, assembled by `llvm-as` from `PATH`.
struct Functions {
    fix: ComputePipelineState,
    draw: RenderPipelineState,
}

impl Functions {
    fn new(device: &Device) -> Result<Functions> {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../metal2vulkan-validation/fixtures/public/examples/fixed_indices.ll"
        );
        let output = Command::new("llvm-as")
            .args(["-o", "-", path])
            .output()
            .map_err(|error| format!("cannot run llvm-as (put it on PATH): {error}"))?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("llvm-as {path}: {stderr}").into());
        }
        let library = device.new_library_with_compile_library(
            metal2vulkan::library::Library::from_air(&output.stdout)?,
        );
        let options = PipelineOptions::default();
        let kernel = library.new_function_with_name("fix_indices")?;
        let fix = device.new_compute_pipeline_state_with_function(&kernel, &options)?;
        let vertex = library.new_function_with_name("placed_vertex")?;
        let fragment = library.new_function_with_name("flat_fragment")?;
        let mut descriptor =
            RenderPipelineDescriptor::new(StageSource::Function(vertex.function_descriptor()));
        descriptor.fragment = Some(StageSource::Function(fragment.function_descriptor()));
        descriptor.color_attachments[0] =
            Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
        let draw = device.new_render_pipeline_state(&descriptor, &options)?;
        Ok(Functions { fix, draw })
    }
}

/// A `Private` buffer holding `bytes`, copied in from a `Shared` one by a blit.
fn private_with(device: &Device, queue: &CommandQueue, bytes: &[u8]) -> Result<Buffer> {
    let staging = device.new_buffer_with_bytes(bytes, ResourceOptions::shared())?;
    let buffer = device.new_buffer(bytes.len() as u64, ResourceOptions::private())?;
    let command_buffer = queue.command_buffer()?;
    let mut blit = command_buffer.blit_command_encoder()?;
    blit.copy_from_buffer_to_buffer(&staging, 0, &buffer, 0, bytes.len() as u64)?;
    blit.end_encoding()?;
    finish(&command_buffer)?;
    Ok(buffer)
}

/// Commit `command_buffer` and wait for it.
fn finish(command_buffer: &CommandBuffer) -> Result<()> {
    command_buffer.commit()?;
    completed(command_buffer)
}

/// Wait for a committed `command_buffer`: an error where it did not complete.
fn completed(command_buffer: &CommandBuffer) -> Result<()> {
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

/// What every frame draws from: the source's indices, the vertices' positions, the target's
/// pass, and each in-flight frame's draws' buffers.
struct Scene {
    source: Buffer,
    /// The source's indices, as the check reads them.
    indices: Vec<u16>,
    positions: Buffer,
    pass: RenderPassDescriptor,
    /// Per frame in flight, one buffer per draw.
    fixed: Vec<Vec<Buffer>>,
}

impl Scene {
    fn new(device: &Device, queue: &CommandQueue, shape: &Shape) -> Result<Scene> {
        let count = shape.draws as usize * shape.indices as usize;
        // Each draw's indices start at a triangle of their own, the source's indices name the
        // vertices in a row, wrapping at `VERTICES`.
        let indices: Vec<u16> = (0..count).map(|at| (at as u32 % VERTICES) as u16).collect();
        let index_bytes: Vec<u8> = indices
            .iter()
            .flat_map(|index| index.to_le_bytes())
            .collect();
        let source = private_with(device, queue, &index_bytes)?;
        // Triangle `t`'s three corners lie a texel or two apart at a cell of a 64 by 64 grid.
        let texel = 2.0 / shape.side as f32;
        let positions: Vec<u8> = (0..VERTICES)
            .flat_map(|vertex| {
                let cell = (vertex / 3) % 4096;
                let x = -1.0 + (cell % 64) as f32 * 2.0 / 64.0 + texel;
                let y = -1.0 + (cell / 64) as f32 * 2.0 / 64.0 + texel;
                let (dx, dy) = match vertex % 3 {
                    0 => (0.0, 0.0),
                    1 => (2.0 * texel, 0.0),
                    _ => (0.0, 2.0 * texel),
                };
                [x + dx, y + dy]
            })
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let positions = private_with(device, queue, &positions)?;
        let mut texture =
            TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, shape.side, shape.side, false);
        texture.usage = TextureUsage::RENDER_TARGET;
        texture.resource_options = ResourceOptions::private();
        let target = device.new_texture(&texture)?;
        let mut pass = RenderPassDescriptor::new();
        let color = &mut pass.color_attachments[0];
        color.texture = Some(target);
        color.load_action = match shape.apart {
            true => LoadAction::DontCare,
            false => LoadAction::Load,
        };
        color.store_action = StoreAction::Store;
        let bytes = u64::from(shape.indices) * 4;
        let fixed = (0..IN_FLIGHT)
            .map(|_| {
                (0..shape.draws)
                    .map(|_| Ok(device.new_buffer(bytes, ResourceOptions::private())?))
                    .collect::<Result<Vec<Buffer>>>()
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Scene {
            source,
            indices,
            positions,
            pass,
            fixed,
        })
    }
}

/// One frame of `name`'s command buffers, committed, for the frame in flight `slot`.
fn frame(
    queue: &CommandQueue,
    functions: &Functions,
    scene: &Scene,
    shape: &Shape,
    (name, slot): (&str, usize),
) -> Result<Vec<CommandBuffer>> {
    let fixed = &scene.fixed[slot];
    let mut committed = Vec::new();
    let threadgroups = u64::from(shape.indices).div_ceil(THREADS);
    for first in (0..shape.draws).step_by(shape.group as usize) {
        let draws = first..(first + shape.group).min(shape.draws);
        let mut command_buffer = queue.command_buffer()?;
        if name != "prefixed" {
            let mut encoder = command_buffer.compute_command_encoder()?;
            encoder.set_compute_pipeline_state(&functions.fix)?;
            for draw in draws.clone() {
                let place = draw * shape.indices;
                let arguments: Vec<u8> = [place, shape.indices, 0]
                    .iter()
                    .flat_map(|word| word.to_le_bytes())
                    .collect();
                encoder.set_buffer(Some(&scene.source), 0, 0)?;
                encoder.set_buffer(Some(&fixed[draw as usize]), 0, 1)?;
                encoder.set_bytes(&arguments, 2)?;
                encoder.dispatch_threadgroups(
                    Size::new(threadgroups, 1, 1),
                    Size::new(THREADS, 1, 1),
                )?;
            }
            encoder.end_encoding()?;
        }
        if name == "dispatches" {
            command_buffer.commit()?;
            committed.push(command_buffer);
            continue;
        }
        let passes: Vec<u32> = draws.clone().step_by(shape.pass_draws as usize).collect();
        for (at, pass_first) in passes.iter().enumerate() {
            if at > 0 && at % shape.cb_passes as usize == 0 {
                command_buffer.commit()?;
                committed.push(command_buffer);
                command_buffer = queue.command_buffer()?;
            }
            let mut encoder = command_buffer.render_command_encoder(&scene.pass)?;
            encoder.set_render_pipeline_state(&functions.draw)?;
            encoder.set_vertex_buffer(Some(&scene.positions), 0, 0)?;
            for draw in *pass_first..(pass_first + shape.pass_draws).min(draws.end) {
                encoder.draw_indexed_primitives(
                    PrimitiveType::Triangle,
                    shape.indices,
                    IndexType::UInt32,
                    &fixed[draw as usize],
                    0,
                    1,
                    0,
                    0,
                )?;
            }
            encoder.end_encoding()?;
        }
        command_buffer.commit()?;
        committed.push(command_buffer);
    }
    Ok(committed)
}

/// A completed frame's GPU times in seconds: the union of its command buffers' spans, and its span.
fn times(frame: &[CommandBuffer]) -> Result<(f64, f64)> {
    let mut spans = Vec::with_capacity(frame.len());
    for command_buffer in frame {
        completed(command_buffer)?;
        spans.push((
            command_buffer.gpu_start_time()?,
            command_buffer.gpu_end_time()?,
        ));
    }
    spans.sort_by(|a, b| a.0.total_cmp(&b.0));
    let (mut busy, mut reached) = (0.0, f64::MIN);
    for &(start, end) in &spans {
        let start = start.max(reached);
        if end > start {
            busy += end - start;
        }
        reached = reached.max(end);
    }
    let first = spans.first().map_or(0.0, |span| span.0);
    Ok((busy, reached.max(first) - first))
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

/// Run `name`'s frames: the `FIXED` and `FIXED-COUNTS` lines.
fn run(
    device: &Device,
    queue: &CommandQueue,
    (functions, scene): (&Functions, &Scene),
    shape: &Shape,
    name: &str,
) -> Result<()> {
    // The frames in flight, each with whether it is timed.
    let mut flight: VecDeque<(bool, Vec<CommandBuffer>)> = VecDeque::new();
    let mut samples = Vec::with_capacity(shape.frames);
    let mut before = None;
    for at in 0..WARM + shape.frames {
        if at == WARM {
            // Every warm frame completed: the counts from here are the timed frames' alone.
            while let Some((_, earlier)) = flight.pop_front() {
                times(&earlier)?;
            }
            before = Some(counts(&device.work_counts()));
        }
        if flight.len() == IN_FLIGHT {
            if let Some((timed, earlier)) = flight.pop_front() {
                let measured = times(&earlier)?;
                if timed {
                    samples.push(measured);
                }
            }
        }
        let committed = frame(queue, functions, scene, shape, (name, at % IN_FLIGHT))?;
        flight.push_back((at >= WARM, committed));
    }
    while let Some((_, earlier)) = flight.pop_front() {
        samples.push(times(&earlier)?);
    }
    let after = counts(&device.work_counts());
    let before = before.ok_or("no timed frame")?;
    let mut busy: Vec<f64> = samples.iter().map(|sample| sample.0).collect();
    let mut span: Vec<f64> = samples.iter().map(|sample| sample.1).collect();
    busy.sort_by(f64::total_cmp);
    span.sort_by(f64::total_cmp);
    let median = busy[busy.len() / 2];
    println!(
        "FIXED {name} draws {} group {} pass_draws {} cb_passes {} indices {} gpu_ms {:.3} least \
         {:.3} span_ms {:.3} us_per_draw {:.3} {}",
        shape.draws,
        shape.group,
        shape.pass_draws,
        shape.cb_passes,
        shape.indices,
        median * 1e3,
        busy[0] * 1e3,
        span[span.len() / 2] * 1e3,
        median * 1e6 / f64::from(shape.draws),
        device.name(),
    );
    let frames = shape.frames as f64;
    let moved: Vec<String> = (after.iter())
        .filter_map(|(name, value)| {
            let moved = value.saturating_sub(before.get(name).copied().unwrap_or(0));
            (moved > 0).then(|| format!("{name}={:.1}", moved as f64 / frames))
        })
        .collect();
    println!("FIXED-COUNTS {name} {}", moved.join(" "));
    Ok(())
}

/// Each draw's indices in the frame in flight `slot`, read back: the source's, widened.
fn check(
    device: &Device,
    queue: &CommandQueue,
    scene: &Scene,
    shape: &Shape,
    slot: usize,
) -> Result<()> {
    let bytes = u64::from(shape.indices) * 4;
    let readback = device.new_buffer(bytes * u64::from(shape.draws), ResourceOptions::shared())?;
    let command_buffer = queue.command_buffer()?;
    let mut blit = command_buffer.blit_command_encoder()?;
    for (draw, fixed) in scene.fixed[slot].iter().enumerate() {
        blit.copy_from_buffer_to_buffer(fixed, 0, &readback, draw as u64 * bytes, bytes)?;
    }
    blit.end_encoding()?;
    finish(&command_buffer)?;
    let contents = readback.contents().ok_or("the readback has no contents")?;
    // SAFETY: `contents` holds the buffer's `length` bytes, and no command buffer writes them now.
    let read = unsafe { std::slice::from_raw_parts(contents.as_ptr(), readback.length() as usize) };
    for (at, word) in read.as_chunks::<4>().0.iter().enumerate() {
        let got = u32::from_le_bytes(*word);
        let want = u32::from(scene.indices[at]);
        if got != want {
            return Err(format!("index {at} of the fixed draws is {got}, not {want}").into());
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let shape = Shape::from_args()?;
    let device = Device::system_default_with(&DeviceOptions::default())?;
    println!("device: {}", device.name());
    let queue = device.new_command_queue()?;
    let functions = Functions::new(&device)?;
    let scene = Scene::new(&device, &queue, &shape)?;
    // Both frames in flight's draws' buffers written once, for `prefixed`.
    for slot in 0..IN_FLIGHT {
        for command_buffer in frame(&queue, &functions, &scene, &shape, ("fixed", slot))? {
            completed(&command_buffer)?;
        }
        check(&device, &queue, &scene, &shape, slot)?;
    }
    for name in SHAPES.into_iter().filter(|name| shape.runs(name)) {
        run(&device, &queue, (&functions, &scene), &shape, name)?;
        check(&device, &queue, &scene, &shape, 0)?;
    }
    Ok(())
}
