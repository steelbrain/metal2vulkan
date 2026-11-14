//! What a compositor's frame costs the GPU, by how its draws read their passes' colors: one
//! command buffer of 105 render encoders over a 1920x1080 framebuffer and four 1024x576 layers:
//! 1.3 to 1.7 render pass instances and 2.4 to 3.6 barriers an encoder on the RTX 5080 Laptop GPU
//! and the Iris Xe, against the consumer's guest walks' 1.5 to 1.9 and 4.4 to 5.1. Fifteen rounds a
//! frame, each:
//!
//! - a layer's pass that clears 768x432 of it, fills, samples the layer and fills;
//! - a framebuffer pass of a fill, a barrier, a reader through 1300x850, a fill, a barrier and a
//!   reader through 260x120;
//! - a framebuffer pass that samples the layer and fills;
//! - the layer's pass that loads it, samples it and fills;
//! - a framebuffer pass of two fills each followed by a barrier and a reader through the fill's
//!   1200x800;
//! - two framebuffer passes of four fills.
//!
//! A barrier is `memory_barrier_with_scope` over the render targets, fragment to fragment; a
//! reader reads the framebuffer at fragment texture 0 at its own pixels; a sampling draw samples
//! its own pass's color, which reads a copy taken before its render pass instance. Each shape
//! ([`Shape`]) runs over a private framebuffer and over one over host pages, a guest surface's
//! kind (a no-copy buffer's IOSurface plane, its rows 64-byte aligned), interleaved round by round,
//! and prints a `COMPOSITOR-COST` line: the median GPU time a frame (its command buffer's GPU end
//! less its start) and, a frame, render pass instances begun, barriers recorded, the readers'
//! splits and snapshots (in their instance and leading one), the splits a draw that reads through
//! snapshots made apart (`instance_splits_feedback`) and texels first copied in.
//!
//! Ignored by default: a measurement, not a check. Run it by name with `--run-ignored all`, in a
//! release build.

use super::attachment_read_costs::{median, state};
use super::attachment_reads::floats;
use super::first_use_writes::CHECK_WAIT;
use super::host_memory::{no_copy, Mapping, GUEST_PAGE};
use super::{open_device, WAIT};
use crate::gpu::TEST_DEVICE;
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::metal::{BarrierScope, PixelFormat, PrimitiveType, RenderStages, TextureUsage};
use metal2vulkan_runtime::{
    Buffer, ClearColor, CommandBuffer, CommandQueue, Device, DeviceOptions, IoSurfacePlane,
    LoadAction, RenderCommandEncoder, RenderPassDescriptor, RenderPipelineState, ResourceOptions,
    SamplerState, ScissorRect, StoreAction, Texture, TextureDescriptor, WorkCounts,
};
use std::collections::BTreeMap;

/// The framebuffer's size.
const FB: [u64; 2] = [1920, 1080];
/// The layers' size, and the part of it a layer's clearing pass renders.
const LAYER: [u64; 2] = [1024, 576];
const LAYER_AREA: [u64; 2] = [768, 432];
const LAYERS: usize = 4;
/// The rounds of encoders a frame (the module docs).
const ROUNDS: u64 = 15;
/// The rounds of the measurement, each one untimed frame and [`FRAMES`] timed ones of every cell.
const PARTS: usize = 3;
const FRAMES: usize = 5;
const BGRA8: PixelFormat = PixelFormat::BGRA8Unorm;

/// How the frame's readers and sampling draws read their passes' colors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Shape {
    /// As the module docs say: readers read with `tob_read`, after their barriers.
    Readers,
    /// The readers read with `wlk_read_branch`, whose texel decides a loop's trip count: a slot
    /// no snapshot serves (`SnapshotRefusal::Branch`), so each splits its instance.
    BranchReaders,
    /// The same calls without the barriers: each reader reads the copy taken before its instance.
    NoBarrier,
    /// No barrier, and every reader and sampling draw a fill: no draw reads its pass's color.
    NoReads,
}

const SHAPES: [Shape; 4] = [
    Shape::Readers,
    Shape::BranchReaders,
    Shape::NoBarrier,
    Shape::NoReads,
];

/// The frame's pipelines and sampler.
struct Draws {
    fill: RenderPipelineState,
    read: RenderPipelineState,
    branch: RenderPipelineState,
    sample: RenderPipelineState,
    sampler: SamplerState,
}

impl Draws {
    fn new(device: &Device) -> Draws {
        Draws {
            fill: state(device, "tob_solid", "tob_renders.ll", BGRA8),
            read: state(device, "tob_read", "tob_renders.ll", BGRA8),
            branch: state(device, "wlk_read_branch", "wlk_readers.ll", BGRA8),
            sample: state(device, "tob_sample", "tob_renders.ll", BGRA8),
            sampler: device.new_sampler_state(&SamplerDescriptor::new()).unwrap(),
        }
    }
}

/// One frame's recording: its draws, its shape, and its encoder.
struct Pass<'a> {
    encoder: RenderCommandEncoder,
    draws: &'a Draws,
    shape: Shape,
    size: [u64; 2],
}

impl Pass<'_> {
    /// Draw over `rect`, through a scissor of it.
    fn over(&mut self, [x0, y0, x1, y1]: [u64; 4]) {
        let rect = ScissorRect {
            x: x0,
            y: y0,
            width: x1 - x0,
            height: y1 - y0,
        };
        self.encoder.set_scissor_rect(rect).unwrap();
        let x = |p: u64| p as f32 / self.size[0] as f32 * 2.0 - 1.0;
        let y = |p: u64| 1.0 - p as f32 / self.size[1] as f32 * 2.0;
        let corners = floats([x(x0), y(y1), x(x1), y(y0)]);
        self.encoder.set_vertex_bytes(&corners, 0).unwrap();
        self.encoder
            .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
            .unwrap();
    }

    /// A fill of `rect` with a color of `seed`.
    fn fill(&mut self, rect: [u64; 4], seed: u64) {
        self.encoder
            .set_render_pipeline_state(&self.draws.fill)
            .unwrap();
        let c = |k: u64| ((seed * 37 + k * 71) % 251) as f32 / 255.0;
        self.encoder
            .set_fragment_bytes(&floats([c(0), c(1), c(2), 1.0]), 0)
            .unwrap();
        self.over(rect);
    }

    fn barrier(&mut self) {
        if matches!(self.shape, Shape::Readers | Shape::BranchReaders) {
            self.encoder.memory_barrier_with_scope(
                BarrierScope::RENDER_TARGETS,
                RenderStages::FRAGMENT,
                RenderStages::FRAGMENT,
            );
        }
    }

    /// A draw over `rect` that reads `texture` at its own pixels, by the shape.
    fn read(&mut self, texture: &Texture, rect: [u64; 4], seed: u64) {
        let state = match self.shape {
            Shape::NoReads => return self.fill(rect, seed),
            Shape::BranchReaders => &self.draws.branch,
            Shape::Readers | Shape::NoBarrier => &self.draws.read,
        };
        self.encoder.set_render_pipeline_state(state).unwrap();
        self.encoder.set_fragment_texture(Some(texture), 0).unwrap();
        self.over(rect);
    }

    /// A draw over `rect` that samples `texture`, by the shape.
    fn sample(&mut self, texture: &Texture, rect: [u64; 4], seed: u64) {
        if self.shape == Shape::NoReads {
            return self.fill(rect, seed);
        }
        self.encoder
            .set_render_pipeline_state(&self.draws.sample)
            .unwrap();
        self.encoder.set_fragment_texture(Some(texture), 0).unwrap();
        self.encoder
            .set_fragment_sampler_state(Some(&self.draws.sampler), 0)
            .unwrap();
        self.over(rect);
    }

    fn end(self) {
        self.encoder.end_encoding().unwrap();
    }
}

/// A pass over `texture`, loaded, or cleared over `area`.
fn descriptor(texture: &Texture, cleared: Option<[u64; 2]>) -> RenderPassDescriptor {
    let mut d = RenderPassDescriptor::new();
    let color = &mut d.color_attachments[0];
    color.texture = Some(texture.clone());
    color.store_action = StoreAction::Store;
    match cleared {
        Some([width, height]) => {
            color.load_action = LoadAction::Clear;
            color.clear_color = ClearColor {
                red: 0.1,
                green: 0.2,
                blue: 0.3,
                alpha: 1.0,
            };
            d.render_target_width = width;
            d.render_target_height = height;
        }
        None => color.load_action = LoadAction::Load,
    }
    d
}

/// Round `r`'s encoders into `command_buffer` (the module docs).
fn round(
    command_buffer: &CommandBuffer,
    (draws, shape): (&Draws, Shape),
    fb: &Texture,
    layers: &[Texture],
    r: u64,
) {
    let layer = &layers[r as usize % layers.len()];
    let o = |k: u64| (r * 53 + k * 29) % 128;
    let open = |texture: &Texture, cleared: Option<[u64; 2]>, size: [u64; 2]| Pass {
        encoder: (command_buffer)
            .render_command_encoder(&descriptor(texture, cleared))
            .unwrap(),
        draws,
        shape,
        size,
    };
    let mut e = open(layer, Some(LAYER_AREA), LAYER);
    e.fill([0, 0, 640, 400], r);
    e.sample(layer, [o(0), o(1), 700, 420], r);
    e.fill([100, 100, 300, 200], r + 1);
    e.end();
    let mut e = open(fb, None, FB);
    e.fill([o(2), o(3), 1400 + o(2), 900 + o(3)], r + 2);
    e.barrier();
    e.read(fb, [o(2), o(3), 1300 + o(2), 850 + o(3)], r);
    e.fill([600 + o(4), 300, 900 + o(4), 450], r + 3);
    e.barrier();
    e.read(fb, [600 + o(4), 300, 860 + o(4), 420], r + 1);
    e.end();
    let mut e = open(fb, None, FB);
    e.sample(layer, [200 + o(5), 100, 1224 + o(5), 676], r);
    e.fill([0, 1000, 1920, 1080], r + 4);
    e.end();
    let mut e = open(layer, None, LAYER);
    e.sample(layer, [o(6), o(7), 900, 500], r + 1);
    e.fill([10, 10, 200, 100], r + 5);
    e.end();
    let mut e = open(fb, None, FB);
    for k in 0..2 {
        let rect = [o(8 + k), o(9 + k), 1200 + o(8 + k), 800 + o(9 + k)];
        e.fill(rect, r + 6 + k);
        e.barrier();
        e.read(fb, rect, r + k);
    }
    e.end();
    for k in 0..2 {
        let mut e = open(fb, None, FB);
        for f in 0..4 {
            let x = (r * 131 + k * 300 + f * 211) % 1600;
            let y = (r * 71 + k * 200 + f * 97) % 800;
            e.fill([x, y, x + 320, y + 280], r + 8 + k + f);
        }
        e.end();
    }
}

/// One frame of `shape` on `queue`, committed and waited for: its GPU time in seconds.
fn frame(
    queue: &CommandQueue,
    (draws, shape): (&Draws, Shape),
    fb: &Texture,
    layers: &[Texture],
) -> f64 {
    let command_buffer = queue.command_buffer().unwrap();
    for r in 0..ROUNDS {
        round(&command_buffer, (draws, shape), fb, layers, r);
    }
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    command_buffer.gpu_end_time().unwrap() - command_buffer.gpu_start_time().unwrap()
}

/// A private target of `size`, rendered and sampled.
fn target(device: &Device, [width, height]: [u64; 2]) -> Texture {
    let mut d = TextureDescriptor::texture_2d(BGRA8, width, height, false);
    d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    d.resource_options = ResourceOptions::private();
    device.new_texture(&d).unwrap()
}

/// A framebuffer over a no-copy buffer over `mapping`'s `len` bytes of host pages, an IOSurface
/// plane of rows of `row` bytes, and the buffer.
fn surface(device: &Device, (mapping, len): (&Mapping, usize), row: u64) -> (Texture, Buffer) {
    let [width, height] = FB;
    // SAFETY: the measurement drops the texture and the buffer before the mapping.
    let pages = unsafe {
        no_copy(
            device,
            mapping.pointer(),
            len,
            ResourceOptions::default(),
            None,
        )
    }
    .unwrap();
    let mut d = TextureDescriptor::texture_2d(BGRA8, width, height, false);
    d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    d.resource_options = pages.resource_options();
    let plane = IoSurfacePlane::new(0, row, false);
    let texture = pages.new_texture_with_iosurface_plane(&d, plane).unwrap();
    (texture, pages)
}

/// A count a `COMPOSITOR-COST` line prints: its name, and how it is read.
type Count = (&'static str, fn(&WorkCounts) -> u64);

/// The counts a `COMPOSITOR-COST` line prints, a frame.
const COUNTS: [Count; 8] = [
    ("render pass instances", |c| {
        c.commands_recorded.cmd_begin_render_pass
    }),
    ("barriers", |c| c.commands_recorded.cmd_pipeline_barrier),
    ("reader splits", |c| c.attachment_read_splits),
    ("snapshotted", |c| c.reading_draws_snapshotted),
    ("of them leading", |c| c.reading_draws_snapshotted_leading),
    ("feedback splits", |c| c.instance_splits_feedback),
    ("branch splits", |c| {
        c.reading_draws_split_unreadable_own_pixel_branch
    }),
    ("texels first copied in", |c| {
        c.shadow_texels_first_copied_in
    }),
];

#[test]
#[ignore = "a release-mode measurement: run by name"]
fn compositor_costs() {
    let device = open_device(TEST_DEVICE, DeviceOptions::default()).unwrap();
    let settled = device.settle_in_place_checks(CHECK_WAIT);
    let draws = Draws::new(&device);
    let queue = device.new_command_queue().unwrap();
    let layers: Vec<Texture> = (0..LAYERS).map(|_| target(&device, LAYER)).collect();
    let row = (4 * FB[0]).next_multiple_of(64);
    let len = ((row * FB[1]) as usize).next_multiple_of(GUEST_PAGE);
    let mapping = Mapping::anonymous(len);
    let (pages_fb, pages) = surface(&device, (&mapping, len), row);
    let fbs = [("private", target(&device, FB)), ("host pages", pages_fb)];
    let counts = device.work_counts();
    println!(
        "COMPOSITOR-COST {}: in-place checks settled {settled}; textures in place over host pages \
         {}, images made with feedback loop usage: aliases {}, shadows {}",
        device.name(),
        counts.host_textures_in_place,
        counts.feedback_loop_alias_images,
        counts.feedback_loop_shadow_images,
    );
    // Per cell (framebuffer, shape): its timed frames' GPU times, and the counts they moved.
    let mut cells: BTreeMap<(usize, usize), (Vec<f64>, [u64; COUNTS.len()])> = BTreeMap::new();
    for _ in 0..PARTS {
        for (f, (_, fb)) in fbs.iter().enumerate() {
            for (s, &shape) in SHAPES.iter().enumerate() {
                frame(&queue, (&draws, shape), fb, &layers);
                let before = device.work_counts();
                let gpu: Vec<f64> = (0..FRAMES)
                    .map(|_| frame(&queue, (&draws, shape), fb, &layers))
                    .collect();
                let after = device.work_counts();
                let cell = cells.entry((f, s)).or_default();
                cell.0.extend(gpu);
                for (sum, (_, count)) in cell.1.iter_mut().zip(COUNTS) {
                    *sum += count(&after) - count(&before);
                }
            }
        }
    }
    let timed = (PARTS * FRAMES) as f64;
    for ((f, s), (gpu, moved)) in cells {
        let counts: Vec<String> = (COUNTS.iter().zip(moved))
            .map(|((name, _), sum)| format!("{name} {:.1}", sum as f64 / timed))
            .collect();
        println!(
            "COMPOSITOR-COST {} {} framebuffer, {:?}: GPU {:.3} ms/frame (median of {}); a frame: \
             {}",
            device.name(),
            fbs[f].0,
            SHAPES[s],
            median(gpu.clone()) * 1e3,
            gpu.len(),
            counts.join(", "),
        );
    }
    drop((fbs, pages, layers));
    drop(mapping);
}
