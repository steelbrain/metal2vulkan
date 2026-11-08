//! What a compositor's reads of its own pass's attachment cost the GPU, on the device the cfg
//! selects, without the validation layer: report only, in release builds, to weigh the copies a
//! draw that samples its pass's attachment takes (`texture/attached_reads.rs`, `shadow.rs`) before
//! and after a change to them.
//!
//! The workload is a window server's composite as the consumer's guest replays it, over a
//! [`WIDTH`]x[`HEIGHT`] `RGBA8Unorm` target of its own image: [`PASSES`] command buffers a frame,
//! each one render pass that loads and stores the target, of [`STEPS`] steps, each a fill of a
//! window's rectangle through its scissor, a memory barrier over the render targets and a draw
//! that reads the target at its own pixels over that rectangle (`tob_read`); then, in every other
//! pass, a barrier and a draw over a window that samples the target at other pixels
//! (`a4_fragment_select_sample`). The windows cover more than a fifth of the target, so no own-pixel
//! read takes a snapshot (`render/snapshot.rs`). The frame waits for its last command buffer.
//!
//! [`WARM`] frames run untimed, then [`TIMED`] frames are printed as one `ATTACHMENT-READ-COST`
//! line: the GPU time per frame (each command buffer's GPU end less its start, summed), the wall
//! time per frame from the first pass's opening through the last command buffer's completion, and
//! the device's work counts per frame: reads through copies, own-pixel reads, splits, snapshots,
//! shadow copies, the copies in of what changed within a rectangle, and the texels each encoder's
//! first copy in of an image's shadow moves.
//!
//! `snapshot_crossover` weighs the two exact forms of a draw that reads its pass's attachment at
//! its own pixels after a barrier (`render/snapshot.rs`): the snapshot and the split. Over a
//! [`WIDTH`]x[`HEIGHT`] `BGRA8Unorm` target, a surface over host pages (a texture of an IOSurface
//! plane over a no-copy buffer, as the consumer's guest makes its framebuffer) or a private
//! texture, each frame is [`CROSS_PASSES`] command buffers of one pass of [`CROSS_STEPS`] steps:
//! a fill of a rectangle at the corner of the step's scissor, a barrier over the render targets,
//! and a reader (`tob_read`) through the scissor whose rectangle, at the same corner, covers a
//! share of it. The cells cross the scissor's sides ([`SCISSORS`]), the share the reader covers
//! ([`COVERAGES`]) and the fill: the whole scissor, or at most [`GUEST_FILL`]. The cells run in
//! turn, round after round, so the GPU's clock weighs on each alike; printed per cell
//! (`SNAPSHOT-CROSSOVER`): the median GPU time per pass, the readers the device snapshotted and
//! split, the texels copied in per pass, and the clock (`gt_cur_freq_mhz`) at the first and last
//! round.
//!
//! `pass_start_barrier_costs` weighs a memory barrier over the render targets at the start of a
//! pass that clears its target, before a draw that reads the target (`texture/attached_reads.rs`):
//! over a display-sized and a backdrop-sized `BGRA8Unorm` private target, a reader at its own pixels
//! (`psb_masked`, beside an `A8Unorm` mask over a buffer) or sampling in pixel coordinates
//! (`psb_sample`), through a scissor, with the barrier and without it.
//!
//! `pass_start_clear_read_costs` weighs the same barrier in a compositor's shape: many passes in
//! one command buffer over one 4K-wide private target, each clearing it, its first draw sampling a
//! quarter of it and three blended draws sampling other textures.
//!
//! `self_reads` is the consumer's guest's self-read shape (its `selfread` probe): one command
//! buffer a frame of one render pass over a [`SELF_SIZE`] `BGRA8Unorm` target (a surface over host
//! pages, as the guest's framebuffer is, its rows 64-byte aligned as IOSurface lays them out, on a
//! device of each setting of `host_page_textures`; or a private texture), loaded but in the first
//! frame, which clears it, and stored; [`SELF_PAIRS`] times a fill of a [`SELF_RECT`] rectangle, a
//! memory barrier over the render targets and a draw over the same rectangle that reads the target
//! at its own pixels (`tob_read`), through no scissor; each target also without the barriers and
//! readers, the baseline. Printed per case (`SELF-READ-COST`): the GPU time per frame, and per
//! frame the splits, snapshots (and of them those pinned to the target's own image,
//! `WorkCounts::reading_draws_pinned`), render pass instances begun, shadow copies, texels copied
//! in (and of them those of a surface's shadow copied in again), keeps and their texels, merges, the
//! shadows made of render targets over host pages and the textures in place over them (once the
//! device's in-place check answered, each line and its time printed, `SELF-READ-CHECK`), and the
//! images made with feedback loop usage.
//!
//! Ignored by default: measurements, not checks. Run them by name with `--run-ignored all`.

use super::attachment_reads::{floats, function};
use super::first_use_writes::CHECK_WAIT;
use super::host_memory::{no_copy, Mapping, GUEST_PAGE};
use super::{open_device, WAIT};
use crate::gpu::TEST_DEVICE;
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::device::HostPageTexturePolicy;
use metal2vulkan::metal::{
    BarrierScope, BlendFactor, PixelFormat, PrimitiveType, RenderStages, TextureUsage,
};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor};
use metal2vulkan_runtime::{
    Buffer, ClearColor, CommandBuffer, CommandQueue, Device, DeviceOptions, IoSurfacePlane,
    LoadAction, PipelineOptions, RenderCommandEncoder, RenderPassDescriptor, RenderPipelineState,
    ResourceOptions, SamplerState, ScissorRect, StoreAction, Texture, TextureDescriptor,
    WorkCounts,
};
use std::collections::BTreeMap;
use std::time::Instant;

/// The target's size, a 1080p framebuffer's.
const WIDTH: u64 = 1920;
const HEIGHT: u64 = 1080;
/// The command buffers of a frame, each one pass.
const PASSES: u64 = 4;
/// The fill, barrier and own-pixel read steps of a pass.
const STEPS: u64 = 3;
/// The untimed and the timed frames.
const WARM: usize = 10;
const TIMED: usize = 60;
const RGBA8: PixelFormat = PixelFormat::RGBA8Unorm;

/// Step `step` of pass `pass`'s window: 960x720 pixels, at an offset of its own.
fn window(pass: u64, step: u64) -> [u64; 4] {
    let (x, y) = (
        (pass * 211 + step * 97) % 960,
        (pass * 53 + step * 131) % 360,
    );
    [x, y, x + 960, y + 720]
}

/// The corners, in normalized device coordinates, of the target's pixels `[x0, x1) x [y0, y1)`.
fn corners([x0, y0, x1, y1]: [u64; 4]) -> Vec<u8> {
    let x = |p: u64| p as f32 / WIDTH as f32 * 2.0 - 1.0;
    let y = |p: u64| 1.0 - p as f32 / HEIGHT as f32 * 2.0;
    floats([x(x0), y(y1), x(x1), y(y0)])
}

/// The `tob_rect` state with the fragment function `name` of `fixture`, into an unblended color
/// of `format`.
fn state(device: &Device, name: &str, fixture: &str, format: PixelFormat) -> RenderPipelineState {
    let mut d = RenderPipelineDescriptor::new(function(device, "tob_rect", "tob_renders.ll"));
    d.fragment = Some(function(device, name, fixture));
    d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(format));
    device
        .new_render_pipeline_state(&d, &PipelineOptions::default())
        .unwrap()
}

/// Draw over `rect`, through a scissor of it.
fn draw_over(encoder: &mut RenderCommandEncoder, [x0, y0, x1, y1]: [u64; 4]) {
    let rect = ScissorRect {
        x: x0,
        y: y0,
        width: x1 - x0,
        height: y1 - y0,
    };
    encoder.set_scissor_rect(rect).unwrap();
    encoder
        .set_vertex_bytes(&corners([x0, y0, x1, y1]), 0)
        .unwrap();
    encoder
        .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
        .unwrap();
}

fn barrier(encoder: &mut RenderCommandEncoder) {
    encoder.memory_barrier_with_scope(
        BarrierScope::RENDER_TARGETS,
        RenderStages::FRAGMENT,
        RenderStages::FRAGMENT,
    );
}

/// The workload's pipelines and sampler.
struct Draws {
    fill: RenderPipelineState,
    read: RenderPipelineState,
    sample: RenderPipelineState,
    sampler: SamplerState,
}

/// One frame over `target` on `queue` (the module docs): its command buffers, committed, the last
/// waited for.
fn frame(queue: &CommandQueue, target: &Texture, draws: &Draws) -> Vec<CommandBuffer> {
    let mut d = RenderPassDescriptor::new();
    let color = &mut d.color_attachments[0];
    color.texture = Some(target.clone());
    color.load_action = LoadAction::Load;
    color.store_action = StoreAction::Store;
    let mut committed = Vec::new();
    for pass in 0..PASSES {
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(&d).unwrap();
        for step in 0..STEPS {
            let rect = window(pass, step);
            encoder.set_render_pipeline_state(&draws.fill).unwrap();
            let red = (pass * STEPS + step) as f32 / (PASSES * STEPS) as f32;
            encoder
                .set_fragment_bytes(&floats([red, 0.5, 0.0, 1.0]), 0)
                .unwrap();
            draw_over(&mut encoder, rect);
            barrier(&mut encoder);
            encoder.set_render_pipeline_state(&draws.read).unwrap();
            encoder.set_fragment_texture(Some(target), 0).unwrap();
            draw_over(&mut encoder, rect);
        }
        if pass % 2 == 1 {
            barrier(&mut encoder);
            encoder.set_render_pipeline_state(&draws.sample).unwrap();
            encoder.set_fragment_texture(Some(target), 0).unwrap();
            encoder.set_fragment_texture(Some(target), 1).unwrap();
            encoder
                .set_fragment_sampler_state(Some(&draws.sampler), 0)
                .unwrap();
            draw_over(&mut encoder, window(pass, STEPS));
        }
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        committed.push(command_buffer);
    }
    let last = committed.last().unwrap();
    last.wait_until_completed_for(WAIT).unwrap();
    for command_buffer in &committed {
        assert_eq!(command_buffer.error(), None);
    }
    committed
}

/// The work counts the module docs name, per frame over `frames` frames.
fn per_frame(before: &WorkCounts, after: &WorkCounts, frames: usize) -> String {
    let per = |a: u64, b: u64| (a - b) as f64 / frames as f64;
    format!(
        "reads through copies {:.1}, own-pixel reads {:.1}, splits {:.1}, snapshotted {:.1}, \
         shadow copies {:.1}, bounded {:.1}, image first-copy texels {:.0}",
        per(
            after.attachment_reads_through_copies,
            before.attachment_reads_through_copies
        ),
        per(after.own_pixel_reads, before.own_pixel_reads),
        per(after.attachment_read_splits, before.attachment_read_splits),
        per(
            after.reading_draws_snapshotted,
            before.reading_draws_snapshotted
        ),
        per(after.shadow_copies, before.shadow_copies),
        per(after.bounded_shadow_copies, before.bounded_shadow_copies),
        per(
            after.shadow_texels_first_copied_in_images,
            before.shadow_texels_first_copied_in_images
        ),
    )
}

#[test]
#[ignore = "a release-mode measurement: run by name"]
fn attachment_read_costs() {
    let device = open_device(TEST_DEVICE, DeviceOptions::default()).unwrap();
    let draws = Draws {
        fill: state(&device, "tob_solid", "tob_renders.ll", RGBA8),
        read: state(&device, "tob_read", "tob_renders.ll", RGBA8),
        sample: state(
            &device,
            "a4_fragment_select_sample",
            "a4_fragment_select_sample.ll",
            RGBA8,
        ),
        sampler: device.new_sampler_state(&SamplerDescriptor::new()).unwrap(),
    };
    let mut d = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, WIDTH, HEIGHT, false);
    d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    d.resource_options = ResourceOptions::private();
    let target = device.new_texture(&d).unwrap();
    let queue = device.new_command_queue().unwrap();
    for _ in 0..WARM {
        frame(&queue, &target, &draws);
    }
    let before = device.work_counts();
    let (mut gpu, started) = (0.0, Instant::now());
    for _ in 0..TIMED {
        for command_buffer in frame(&queue, &target, &draws) {
            gpu +=
                command_buffer.gpu_end_time().unwrap() - command_buffer.gpu_start_time().unwrap();
        }
    }
    let wall = started.elapsed().as_secs_f64();
    let after = device.work_counts();
    println!(
        "ATTACHMENT-READ-COST on {}: {WIDTH}x{HEIGHT} target of its own image, {PASSES} passes of \
         {STEPS} steps: GPU {:.3} ms/frame, wall {:.3} ms/frame; {}",
        device.name(),
        gpu * 1e3 / TIMED as f64,
        wall * 1e3 / TIMED as f64,
        per_frame(&before, &after, TIMED)
    );
}

/// The sides of each crossover reader's scissor (`snapshot_crossover`).
const SCISSORS: [[u64; 2]; 4] = [[320, 180], [960, 540], [1280, 720], [WIDTH, HEIGHT]];
/// The share of its scissor each crossover reader's rectangle covers, in thousandths.
const COVERAGES: [u64; 4] = [15, 125, 500, 1000];
/// The fill's sides at most, where it is not the whole scissor: about the 394K texels the guest's
/// copies of what changed within a rectangle moved each, on average.
const GUEST_FILL: [u64; 2] = [640, 616];
/// The crossover's command buffers a frame, each one pass, and its steps a pass.
const CROSS_PASSES: u64 = 4;
const CROSS_STEPS: u64 = 5;
/// The crossover's rounds of every cell, the first untimed.
const ROUNDS: usize = 8;

/// One crossover cell (the module docs).
#[derive(Clone, Copy, Debug)]
struct Cell {
    /// The target is the surface over host pages, else the private texture.
    host: bool,
    scissor: [u64; 2],
    /// The reader's rectangle's sides, and the thousandths of the scissor they cover.
    reader: [u64; 2],
    coverage: u64,
    fill: [u64; 2],
}

/// Every crossover cell, in the order each round runs them.
fn cells() -> Vec<Cell> {
    let mut cells = Vec::new();
    for host in [true, false] {
        for scissor in SCISSORS {
            let capped = [scissor[0].min(GUEST_FILL[0]), scissor[1].min(GUEST_FILL[1])];
            let fills = match capped == scissor {
                true => vec![scissor],
                false => vec![scissor, capped],
            };
            for coverage in COVERAGES {
                let side = |s: u64| {
                    let side = s as f64 * (coverage as f64 / 1000.0).sqrt();
                    (side.round() as u64).clamp(1, s)
                };
                for &fill in &fills {
                    let reader = [side(scissor[0]), side(scissor[1])];
                    cells.push(Cell {
                        host,
                        scissor,
                        reader,
                        coverage,
                        fill,
                    });
                }
            }
        }
    }
    cells
}

/// A [`WIDTH`]x[`HEIGHT`] `BGRA8Unorm` surface over `mapping`'s host pages, render target and
/// shader read usage, as the consumer makes an IOSurface's texture; and its buffer.
fn surface(device: &Device, mapping: &Mapping) -> (Texture, Buffer) {
    let row = 4 * WIDTH;
    let len = ((row * HEIGHT) as usize).next_multiple_of(GUEST_PAGE);
    // SAFETY: the measurement drops the texture and the buffer before the mapping.
    let buffer = unsafe {
        no_copy(
            device,
            mapping.pointer(),
            len,
            ResourceOptions::default(),
            None,
        )
    }
    .unwrap();
    let mut d = TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, WIDTH, HEIGHT, false);
    d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    d.resource_options = buffer.resource_options();
    let texture = buffer
        .new_texture_with_iosurface_plane(&d, IoSurfacePlane::new(0, row, false))
        .unwrap();
    (texture, buffer)
}

/// The scissor of step `step` of pass `pass` of `cell`: its sides, at a place of its own.
fn scissor_of(cell: &Cell, pass: u64, step: u64) -> [u64; 2] {
    let n = pass * CROSS_STEPS + step;
    let [width, height] = cell.scissor;
    [
        n * 211 % (WIDTH - width + 1),
        n * 131 % (HEIGHT - height + 1),
    ]
}

/// One crossover frame of `cell` over `target` on `queue` (the module docs): its GPU time, in
/// microseconds.
fn cross_frame(
    queue: &CommandQueue,
    target: &Texture,
    fill: &RenderPipelineState,
    read: &RenderPipelineState,
    cell: &Cell,
) -> f64 {
    let mut d = RenderPassDescriptor::new();
    let color = &mut d.color_attachments[0];
    color.texture = Some(target.clone());
    color.load_action = LoadAction::Load;
    color.store_action = StoreAction::Store;
    let mut committed = Vec::new();
    for pass in 0..CROSS_PASSES {
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(&d).unwrap();
        for step in 0..CROSS_STEPS {
            let [x, y] = scissor_of(cell, pass, step);
            encoder.set_render_pipeline_state(fill).unwrap();
            let red = (pass * CROSS_STEPS + step) as f32 / (CROSS_PASSES * CROSS_STEPS) as f32;
            encoder
                .set_fragment_bytes(&floats([red, 0.5, 0.0, 1.0]), 0)
                .unwrap();
            draw_over(&mut encoder, [x, y, x + cell.fill[0], y + cell.fill[1]]);
            barrier(&mut encoder);
            encoder.set_render_pipeline_state(read).unwrap();
            encoder.set_fragment_texture(Some(target), 0).unwrap();
            let [width, height] = cell.scissor;
            encoder
                .set_scissor_rect(ScissorRect {
                    x,
                    y,
                    width,
                    height,
                })
                .unwrap();
            let [w, h] = cell.reader;
            encoder
                .set_vertex_bytes(&corners([x, y, x + w, y + h]), 0)
                .unwrap();
            encoder
                .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
                .unwrap();
        }
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        committed.push(command_buffer);
    }
    let last = committed.last().unwrap();
    last.wait_until_completed_for(WAIT).unwrap();
    let mut gpu = 0.0;
    for command_buffer in &committed {
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
        gpu += command_buffer.gpu_end_time().unwrap() - command_buffer.gpu_start_time().unwrap();
    }
    gpu * 1e6
}

/// The GPU's clock now, in MHz, where a DRM card states it (i915's and Xe's `gt_cur_freq_mhz`).
fn clock() -> String {
    let mut cards = std::fs::read_dir("/sys/class/drm")
        .into_iter()
        .flatten()
        .flatten();
    cards
        .find_map(|card| std::fs::read_to_string(card.path().join("gt_cur_freq_mhz")).ok())
        .map_or_else(|| "-".to_owned(), |mhz| mhz.trim().to_owned())
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

#[test]
#[ignore = "a release-mode measurement: run by name"]
fn snapshot_crossover() {
    let device = open_device(TEST_DEVICE, DeviceOptions::default()).unwrap();
    let bgra = PixelFormat::BGRA8Unorm;
    let fill = state(&device, "tob_solid", "tob_renders.ll", bgra);
    let read = state(&device, "tob_read", "tob_renders.ll", bgra);
    let mapping = Mapping::anonymous(((4 * WIDTH * HEIGHT) as usize).next_multiple_of(GUEST_PAGE));
    let (host, buffer) = surface(&device, &mapping);
    let mut d = TextureDescriptor::texture_2d(bgra, WIDTH, HEIGHT, false);
    d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    d.resource_options = ResourceOptions::private();
    let private = device.new_texture(&d).unwrap();
    let queue = device.new_command_queue().unwrap();
    let cells = cells();
    let mut times: BTreeMap<usize, Vec<f64>> = BTreeMap::new();
    let mut moves: BTreeMap<usize, (WorkCounts, WorkCounts)> = BTreeMap::new();
    let mut clocks: BTreeMap<usize, Vec<String>> = BTreeMap::new();
    for round in 0..ROUNDS {
        for (index, cell) in cells.iter().enumerate() {
            let target = if cell.host { &host } else { &private };
            clocks.entry(index).or_default().push(clock());
            let before = device.work_counts();
            let gpu = cross_frame(&queue, target, &fill, &read, cell);
            let after = device.work_counts();
            if round == 0 {
                continue;
            }
            if round == 1 {
                moves.insert(index, (before, after));
            }
            times.entry(index).or_default().push(gpu);
        }
    }
    let name = device.name().to_owned();
    for (index, cell) in cells.iter().enumerate() {
        let gpu = median(times.remove(&index).unwrap());
        let (before, after) = &moves[&index];
        let per = |a: u64, b: u64| (b - a) as f64 / CROSS_PASSES as f64;
        let clocks = &clocks[&index];
        println!(
            "SNAPSHOT-CROSSOVER {name} {} scissor {}x{} reader {}x{} ({}/1000) fill {}x{}: GPU \
             {:.1} us/pass; per pass: snapshotted {:.2}, splits {:.2}, texels copied in {:.0}; \
             clock {} .. {} MHz",
            if cell.host { "host" } else { "private" },
            cell.scissor[0],
            cell.scissor[1],
            cell.reader[0],
            cell.reader[1],
            cell.coverage,
            cell.fill[0],
            cell.fill[1],
            gpu / CROSS_PASSES as f64,
            per(
                before.reading_draws_snapshotted,
                after.reading_draws_snapshotted
            ),
            per(before.attachment_read_splits, after.attachment_read_splits),
            per(
                before.shadow_texels_copied_in,
                after.shadow_texels_copied_in
            ),
            clocks[1],
            clocks[clocks.len() - 1],
        );
    }
    drop((host, private, buffer));
    drop(mapping);
}

/// The targets of `pass_start_barrier_costs`: a display-sized compositor target and a
/// backdrop-sized one, each with the scissor its reader draws through.
const START_TARGETS: [([u64; 2], [u64; 2]); 2] =
    [([1728, 1216], [453, 100]), ([896, 576], [841, 521])];
/// `pass_start_barrier_costs`'s command buffers a frame, each one pass, and its rounds of every
/// cell, the first untimed.
const START_PASSES: usize = 8;
const START_ROUNDS: usize = 12;

/// One cell of `pass_start_barrier_costs`: the target, the reader (`psb_masked`, which reads at
/// its own pixel, or `psb_sample`, which samples in pixel coordinates), and whether the pass begins
/// with a memory barrier over the render targets.
#[derive(Clone, Copy, Debug)]
struct StartCell {
    target: usize,
    sample: bool,
    barrier: bool,
}

/// One frame of `pass_start_barrier_costs` over `target`: [`START_PASSES`] command buffers of one
/// pass each that clears it, `barrier` first where asked, then one draw of `reader` over the whole
/// target through `scissor`, the target at fragment texture 0 and `mask` at 3. The GPU time of
/// its command buffers, summed, and the CPU time from each command buffer's making through its
/// commit, summed.
fn start_frame(
    queue: &CommandQueue,
    (target, mask): (&Texture, &Texture),
    reader: &RenderPipelineState,
    scissor: [u64; 2],
    barrier: bool,
) -> (f64, f64) {
    let mut d = RenderPassDescriptor::new();
    let color = &mut d.color_attachments[0];
    color.texture = Some(target.clone());
    color.load_action = LoadAction::Clear;
    color.store_action = StoreAction::Store;
    let (width, height) = (target.width(), target.height());
    let corners = {
        let x = |p: u64| p as f32 / width as f32 * 2.0 - 1.0;
        let y = |p: u64| 1.0 - p as f32 / height as f32 * 2.0;
        floats([x(0), y(height), x(width), y(0)])
    };
    let mut committed = Vec::new();
    let mut cpu = 0.0;
    for _ in 0..START_PASSES {
        let started = Instant::now();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(&d).unwrap();
        if barrier {
            encoder.memory_barrier_with_scope(
                BarrierScope::RENDER_TARGETS,
                RenderStages::FRAGMENT,
                RenderStages::FRAGMENT,
            );
        }
        encoder.set_render_pipeline_state(reader).unwrap();
        encoder.set_vertex_bytes(&corners, 0).unwrap();
        let rect = ScissorRect {
            x: 0,
            y: 0,
            width: scissor[0],
            height: scissor[1],
        };
        encoder.set_scissor_rect(rect).unwrap();
        encoder.set_fragment_texture(Some(target), 0).unwrap();
        encoder.set_fragment_texture(Some(mask), 3).unwrap();
        encoder
            .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
            .unwrap();
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        cpu += started.elapsed().as_secs_f64();
        committed.push(command_buffer);
    }
    committed
        .last()
        .unwrap()
        .wait_until_completed_for(WAIT)
        .unwrap();
    let mut gpu = 0.0;
    for command_buffer in &committed {
        assert_eq!(command_buffer.error(), None);
        gpu += command_buffer.gpu_end_time().unwrap() - command_buffer.gpu_start_time().unwrap();
    }
    (gpu, cpu)
}

/// What a memory barrier over the render targets at the start of a pass that clears its target
/// costs where the pass's draw reads the target (`texture/attached_reads.rs`): the draw records
/// into an instance after the pass's first, which runs the clear alone, or reads through snapshots.
/// Each cell (the module docs' `StartCell`) runs [`START_PASSES`] passes a frame, the cells in
/// turn, round after round; printed per cell (`PASS-START-BARRIER-COST`): the median GPU time per
/// pass, the median CPU time per pass from its command buffer's making through its commit, and per
/// pass the splits, snapshots, render pass instances begun and texels copied in.
#[test]
#[ignore = "a release-mode measurement: run by name"]
fn pass_start_barrier_costs() {
    let device = open_device(TEST_DEVICE, DeviceOptions::default()).unwrap();
    let bgra = PixelFormat::BGRA8Unorm;
    let psb = |name: &str| {
        let mut d = RenderPipelineDescriptor::new(function(&device, "psb_rect", "psb_reads.ll"));
        d.fragment = Some(function(&device, name, "psb_reads.ll"));
        d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(bgra));
        device
            .new_render_pipeline_state(&d, &PipelineOptions::default())
            .unwrap()
    };
    let readers = [psb("psb_masked"), psb("psb_sample")];
    let targets: Vec<Texture> = START_TARGETS
        .iter()
        .map(|&([width, height], _)| {
            let mut d = TextureDescriptor::texture_2d(bgra, width, height, false);
            d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
            d.resource_options = ResourceOptions::private();
            device.new_texture(&d).unwrap()
        })
        .collect();
    // A 64x16 `A8Unorm` mask over a buffer, rows 256 bytes apart.
    let mask_buffer = device
        .new_buffer(256 * 16, ResourceOptions::shared())
        .unwrap();
    let mut d = TextureDescriptor::texture_2d(PixelFormat::A8Unorm, 64, 16, false);
    d.usage = TextureUsage::SHADER_READ;
    d.resource_options = ResourceOptions::shared();
    let mask = mask_buffer.new_texture(&d, 0, 256).unwrap();
    let queue = device.new_command_queue().unwrap();
    let mut cells = Vec::new();
    for target in 0..START_TARGETS.len() {
        for sample in [false, true] {
            for barrier in [false, true] {
                cells.push(StartCell {
                    target,
                    sample,
                    barrier,
                });
            }
        }
    }
    let mut times: BTreeMap<usize, (Vec<f64>, Vec<f64>)> = BTreeMap::new();
    let mut moves: BTreeMap<usize, (WorkCounts, WorkCounts)> = BTreeMap::new();
    for round in 0..START_ROUNDS {
        for (index, cell) in cells.iter().enumerate() {
            let before = device.work_counts();
            let (gpu, cpu) = start_frame(
                &queue,
                (&targets[cell.target], &mask),
                &readers[usize::from(cell.sample)],
                START_TARGETS[cell.target].1,
                cell.barrier,
            );
            let after = device.work_counts();
            if round == 0 {
                continue;
            }
            if round == 1 {
                moves.insert(index, (before, after));
            }
            let held = times.entry(index).or_default();
            held.0.push(gpu);
            held.1.push(cpu);
        }
    }
    let name = device.name().to_owned();
    for (index, cell) in cells.iter().enumerate() {
        let (gpu, cpu) = times.remove(&index).unwrap();
        let (gpu, cpu) = (median(gpu), median(cpu));
        let (before, after) = &moves[&index];
        let per = |a: u64, b: u64| (b - a) as f64 / START_PASSES as f64;
        let begun = |c: &WorkCounts| c.render_passes_begun_instances + c.render_passes_begun_inline;
        let ([width, height], [sw, sh]) = START_TARGETS[cell.target];
        println!(
            "PASS-START-BARRIER-COST {name} {width}x{height} scissor {sw}x{sh} {} {}: GPU {:.1} \
             us/pass, CPU {:.1} us/pass; per pass: splits {:.2}, snapshotted {:.2}, instances \
             begun {:.2}, texels copied in {:.0}",
            if cell.sample { "sample" } else { "own-pixel" },
            if cell.barrier {
                "barrier"
            } else {
                "no-barrier"
            },
            gpu * 1e6 / START_PASSES as f64,
            cpu * 1e6 / START_PASSES as f64,
            per(before.attachment_read_splits, after.attachment_read_splits),
            per(
                before.reading_draws_snapshotted,
                after.reading_draws_snapshotted
            ),
            per(begun(before), begun(after)),
            per(
                before.shadow_texels_copied_in + before.shadow_texels_copied_in_images,
                after.shadow_texels_copied_in + after.shadow_texels_copied_in_images
            ),
        );
    }
}

/// The target of `pass_start_clear_read_costs`: a 4K-wide private `BGRA8Unorm` texture.
const CLEAR_READ_TARGET: [u64; 2] = [3840, 2400];
/// Its passes in one command buffer, and the command buffers timed after one untimed.
const CLEAR_READ_PASSES: usize = 60;
const CLEAR_READ_FRAMES: usize = 12;

/// What a memory barrier over the render targets at the start of a pass that clears its target
/// costs where the pass's first draw samples the target (`texture/attached_reads.rs`), in a
/// compositor's shape: [`CLEAR_READ_PASSES`] passes in one command buffer over one
/// [`CLEAR_READ_TARGET`] private `BGRA8Unorm` target, each loading `Clear` (to whole codes) and
/// storing, its first command the barrier (after the fragment stage and before it), then a quad
/// over a quarter of the target that samples the target itself (`psb_sample`), then three blended
/// quads that sample other textures. Printed (`PASS-START-CLEAR-READ-COST`): the median GPU time
/// per pass over [`CLEAR_READ_FRAMES`] command buffers, the CPU time per pass from the command
/// buffer's making through its commit, and per pass the splits, render pass instances begun and
/// texels copied in.
#[test]
#[ignore = "a release-mode measurement: run by name"]
fn pass_start_clear_read_costs() {
    let device = open_device(TEST_DEVICE, DeviceOptions::default()).unwrap();
    let bgra = PixelFormat::BGRA8Unorm;
    let psb = |name: &str, blended: bool| {
        let mut d = RenderPipelineDescriptor::new(function(&device, "psb_rect", "psb_reads.ll"));
        d.fragment = Some(function(&device, name, "psb_reads.ll"));
        let mut color = ColorAttachmentDescriptor::new(bgra);
        if blended {
            color.blending_enabled = true;
            color.source_rgb_blend_factor = BlendFactor::SourceAlpha;
            color.destination_rgb_blend_factor = BlendFactor::OneMinusSourceAlpha;
            color.source_alpha_blend_factor = BlendFactor::One;
            color.destination_alpha_blend_factor = BlendFactor::OneMinusSourceAlpha;
        }
        d.color_attachments[0] = Some(color);
        device
            .new_render_pipeline_state(&d, &PipelineOptions::default())
            .unwrap()
    };
    let (read, blend) = (psb("psb_sample", false), psb("psb_sample", true));
    let [width, height] = CLEAR_READ_TARGET;
    let mut d = TextureDescriptor::texture_2d(bgra, width, height, false);
    d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    d.resource_options = ResourceOptions::private();
    let target = device.new_texture(&d).unwrap();
    let others: Vec<Texture> = (0..3)
        .map(|_| {
            let mut d = TextureDescriptor::texture_2d(bgra, 1024, 1024, false);
            d.usage = TextureUsage::SHADER_READ;
            d.resource_options = ResourceOptions::private();
            device.new_texture(&d).unwrap()
        })
        .collect();
    let ndc = |[x0, y0, x1, y1]: [u64; 4]| {
        let x = |p: u64| p as f32 / width as f32 * 2.0 - 1.0;
        let y = |p: u64| 1.0 - p as f32 / height as f32 * 2.0;
        floats([x(x0), y(y1), x(x1), y(y0)])
    };
    let quads = [
        [width / 4, height / 4, 3 * width / 4, 3 * height / 4],
        [0, 0, width / 2, height / 2],
        [width / 2, 0, width, height / 2],
        [width / 4, height / 2, 3 * width / 4, height],
    ];
    let queue = device.new_command_queue().unwrap();
    let mut gpu = Vec::new();
    let mut cpu = Vec::new();
    let mut moved = None;
    for frame in 0..=CLEAR_READ_FRAMES {
        let before = device.work_counts();
        let started = Instant::now();
        let command_buffer = queue.command_buffer().unwrap();
        for _ in 0..CLEAR_READ_PASSES {
            let mut d = RenderPassDescriptor::new();
            let color = &mut d.color_attachments[0];
            color.texture = Some(target.clone());
            color.load_action = LoadAction::Clear;
            color.store_action = StoreAction::Store;
            color.clear_color = ClearColor::new(0.0, 0.0, 0.0, 1.0);
            let mut encoder = command_buffer.render_command_encoder(&d).unwrap();
            barrier(&mut encoder);
            encoder.set_render_pipeline_state(&read).unwrap();
            encoder.set_fragment_texture(Some(&target), 0).unwrap();
            encoder.set_vertex_bytes(&ndc(quads[0]), 0).unwrap();
            encoder
                .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
                .unwrap();
            encoder.set_render_pipeline_state(&blend).unwrap();
            for (quad, other) in quads[1..].iter().zip(&others) {
                encoder.set_fragment_texture(Some(other), 0).unwrap();
                encoder.set_vertex_bytes(&ndc(*quad), 0).unwrap();
                encoder
                    .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
                    .unwrap();
            }
            encoder.end_encoding().unwrap();
        }
        command_buffer.commit().unwrap();
        let encoded = started.elapsed().as_secs_f64();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
        let after = device.work_counts();
        if frame == 0 {
            continue;
        }
        if frame == 1 {
            moved = Some((before, after));
        }
        gpu.push(command_buffer.gpu_end_time().unwrap() - command_buffer.gpu_start_time().unwrap());
        cpu.push(encoded);
    }
    let (before, after) = moved.unwrap();
    let per = |a: u64, b: u64| (b - a) as f64 / CLEAR_READ_PASSES as f64;
    let begun = |c: &WorkCounts| c.render_passes_begun_instances + c.render_passes_begun_inline;
    println!(
        "PASS-START-CLEAR-READ-COST {} {width}x{height}, {CLEAR_READ_PASSES} passes a command \
         buffer: GPU {:.1} us/pass, CPU {:.1} us/pass; per pass: splits {:.2}, instances begun \
         {:.2}, texels copied in {:.0}",
        device.name(),
        median(gpu) * 1e6 / CLEAR_READ_PASSES as f64,
        median(cpu) * 1e6 / CLEAR_READ_PASSES as f64,
        per(before.attachment_read_splits, after.attachment_read_splits),
        per(begun(&before), begun(&after)),
        per(
            before.shadow_texels_copied_in + before.shadow_texels_copied_in_images,
            after.shadow_texels_copied_in + after.shadow_texels_copied_in_images
        ),
    );
}

/// The `self_reads` target's size: the guest's desktop framebuffer's.
const SELF_SIZE: [u64; 2] = [3662, 2039];
/// The fill and reader pairs of a `self_reads` pass, and the rectangle each covers.
const SELF_PAIRS: u64 = 30;
const SELF_RECT: [u64; 2] = [600, 400];
/// The untimed and the timed `self_reads` frames of each case.
const SELF_WARM: usize = 3;
const SELF_TIMED: usize = 20;

/// Pair `pair`'s rectangle in frame `frame` of `self_reads`, at a place of its own.
fn self_rect(frame: usize, pair: u64) -> [u64; 4] {
    let ([width, height], [w, h]) = (SELF_SIZE, SELF_RECT);
    let n = frame as u64 + pair;
    let (x, y) = (n * 1373 % (width - w), (n * 3 + 1) * 787 % (height - h));
    [x, y, x + w, y + h]
}

/// One `self_reads` frame (the module docs) over `target`, its `frame`th, on `queue`; with no
/// barriers or readers where `baseline`. Its GPU time, in seconds.
fn self_frame(
    queue: &CommandQueue,
    target: &Texture,
    (fill, read): (&RenderPipelineState, &RenderPipelineState),
    frame: usize,
    baseline: bool,
) -> f64 {
    let mut d = RenderPassDescriptor::new();
    let color = &mut d.color_attachments[0];
    color.texture = Some(target.clone());
    color.load_action = match frame {
        0 => LoadAction::Clear,
        _ => LoadAction::Load,
    };
    color.store_action = StoreAction::Store;
    let [width, height] = SELF_SIZE;
    let corners = |[x0, y0, x1, y1]: [u64; 4]| {
        let x = |p: u64| p as f32 / width as f32 * 2.0 - 1.0;
        let y = |p: u64| 1.0 - p as f32 / height as f32 * 2.0;
        floats([x(x0), y(y1), x(x1), y(y0)])
    };
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.render_command_encoder(&d).unwrap();
    for pair in 0..SELF_PAIRS {
        let rect = corners(self_rect(frame, pair));
        encoder.set_render_pipeline_state(fill).unwrap();
        let red = pair as f32 / SELF_PAIRS as f32;
        encoder
            .set_fragment_bytes(&floats([red, 0.5, 0.0, 1.0]), 0)
            .unwrap();
        encoder.set_vertex_bytes(&rect, 0).unwrap();
        encoder
            .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
            .unwrap();
        if baseline {
            continue;
        }
        barrier(&mut encoder);
        encoder.set_render_pipeline_state(read).unwrap();
        encoder.set_fragment_texture(Some(target), 0).unwrap();
        encoder
            .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
            .unwrap();
    }
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    command_buffer.gpu_end_time().unwrap() - command_buffer.gpu_start_time().unwrap()
}

#[test]
#[ignore = "a release-mode measurement: run by name"]
fn self_reads() {
    // Each setting of `host_page_textures`: the surface in place where the device takes it so,
    // and its shadow; the private target once.
    for policy in [
        HostPageTexturePolicy::InPlace,
        HostPageTexturePolicy::Shadow,
    ] {
        let mut options = DeviceOptions::default();
        options.policy.host_page_textures = policy;
        let device = open_device(TEST_DEVICE, options).unwrap();
        self_reads_on(&device, policy);
    }
}

/// [`self_reads`] on `device`, of `policy`: its surface, and under `InPlace` the private target.
fn self_reads_on(device: &Device, policy: HostPageTexturePolicy) {
    // The surface takes the path the device's check answers, not a shadow made before it did.
    if policy == HostPageTexturePolicy::InPlace {
        let settled = device.settle_in_place_checks(CHECK_WAIT);
        for line in device.in_place_check_report() {
            println!(
                "SELF-READ-CHECK {}: settled {settled}: {line}",
                device.name()
            );
        }
    }
    println!(
        "SELF-READ-CHECK {}: took {:?}",
        device.name(),
        device.in_place_check_took()
    );
    let bgra = PixelFormat::BGRA8Unorm;
    let draws = (
        state(device, "tob_solid", "tob_renders.ll", bgra),
        state(device, "tob_read", "tob_renders.ll", bgra),
    );
    let [width, height] = SELF_SIZE;
    // A surface's rows at a multiple of 64 bytes, as IOSurface lays them out (a plane of a
    // surface without planes takes a multiple of 16 alone).
    let row = (4 * width).next_multiple_of(64);
    let len = ((row * height) as usize).next_multiple_of(GUEST_PAGE);
    let mapping = Mapping::anonymous(len);
    // SAFETY: the measurement drops the texture and the buffer before the mapping.
    let buffer = unsafe {
        no_copy(
            device,
            mapping.pointer(),
            len,
            ResourceOptions::default(),
            None,
        )
    }
    .unwrap();
    let mut d = TextureDescriptor::texture_2d(bgra, width, height, false);
    d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    d.resource_options = buffer.resource_options();
    let surface = buffer
        .new_texture_with_iosurface_plane(&d, IoSurfacePlane::new(0, row, false))
        .unwrap();
    d.resource_options = ResourceOptions::private();
    let private = device.new_texture(&d).unwrap();
    let queue = device.new_command_queue().unwrap();
    let name = device.name().to_owned();
    let mut targets = vec![(&surface, format!("surface ({policy:?})"))];
    if policy == HostPageTexturePolicy::InPlace {
        targets.push((&private, "private".to_owned()));
    }
    for (target, kind) in targets {
        for baseline in [true, false] {
            let mut frames = 0..;
            let mut next = || frames.next().unwrap();
            for _ in 0..SELF_WARM {
                self_frame(&queue, target, (&draws.0, &draws.1), next(), baseline);
            }
            let before = device.work_counts();
            let mut gpu = Vec::new();
            for _ in 0..SELF_TIMED {
                gpu.push(self_frame(
                    &queue,
                    target,
                    (&draws.0, &draws.1),
                    next(),
                    baseline,
                ));
            }
            let after = device.work_counts();
            let per = |a: u64, b: u64| (b - a) as f64 / SELF_TIMED as f64;
            let keeps = |c: &WorkCounts| c.shadow_keeps.render_cut + c.shadow_keeps.render_whole;
            let kept = |c: &WorkCounts| {
                c.shadow_keeps.render_cut_texels + c.shadow_keeps.render_whole_texels
            };
            println!(
                "SELF-READ-COST {name} {kind} {width}x{height} rows of {row} bytes {} pairs {} \
                 {}x{}: GPU {:.3} ms/frame (median); per frame: splits {:.2}, snapshotted \
                 {:.2}, pinned {:.2}, instances {:.2}, shadow copies {:.2}, bounded {:.2}, texels \
                 copied in {:.0}, of a surface again {:.0}, keeps {:.2}, texels kept {:.0}, \
                 merges {:.2}; \
                 render-target shadows over host pages {}, textures in place over host pages \
                 {}, images made with feedback loop usage: shadows {}, aliases {}",
                SELF_PAIRS,
                if baseline { "baseline" } else { "self-read" },
                SELF_RECT[0],
                SELF_RECT[1],
                median(gpu) * 1e3,
                per(before.attachment_read_splits, after.attachment_read_splits),
                per(
                    before.reading_draws_snapshotted,
                    after.reading_draws_snapshotted
                ),
                per(before.reading_draws_pinned, after.reading_draws_pinned),
                per(
                    before.render_passes_begun_instances,
                    after.render_passes_begun_instances
                ),
                per(before.shadow_copies, after.shadow_copies),
                per(before.bounded_shadow_copies, after.bounded_shadow_copies),
                per(
                    before.shadow_texels_copied_in,
                    after.shadow_texels_copied_in
                ),
                per(
                    before.shadow_texels_recopied_in_surfaces,
                    after.shadow_texels_recopied_in_surfaces
                ),
                per(keeps(&before), keeps(&after)),
                per(kept(&before), kept(&after)),
                per(before.shadow_merges_fetched, after.shadow_merges_fetched),
                after.host_texture_shadows_render_target,
                after.host_textures_in_place,
                after.feedback_loop_shadow_images,
                after.feedback_loop_alias_images,
            );
        }
    }
    drop((surface, private, buffer));
    drop(mapping);
}
