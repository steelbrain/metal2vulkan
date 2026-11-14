//! A render encoder's tile dispatch whose `VkPipeline` is being made (`render/pending_tile.rs`):
//! the dispatch returns once encoded, its pass records once the making lands, and its command
//! buffer completes in its queue's order with the tile function's stores, as Metal's, which made
//! the tile state's pipeline at its creation. The device's pool's test hold (`pool::held`) keeps
//! every making waiting here.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The tile function is the owned fixture
//! `validation/fixtures/public/engine/k10ts_tile_grid.ll` (`llvm-as`): over an implicit
//! imageblock of an `R32Float` and two `R32Uint` attachments, each thread reads the 2x2 cells at
//! twice its position and writes `f * 2 + 1`, `v * 3 + k` (`k = 1 + x % 2 + 2 * (y % 2)` of the
//! cell) and a tag of its tile and cell.

use super::blit_holds::{device, Scratch, WAIT};
use crate::pipeline::pool::held;
use crate::{
    Buffer, CommandBuffer, CommandBufferStatus, CommandQueue, Device, LoadAction, PipelineOptions,
    Region, RenderPassDescriptor, RenderPipelineState, ResourceOptions, Size, StoreAction, Texture,
    TextureDescriptor,
};
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use metal2vulkan::pipeline::{StageSource, TileRenderPipelineDescriptor};
use std::path::Path;
use std::time::{Duration, Instant};

/// How long a command buffer that waits for a making is watched not to complete.
const WATCHED: Duration = Duration::from_millis(500);

/// The attachments' formats: the fixture's implicit imageblock.
pub(super) const FORMATS: [PixelFormat; 3] = [
    PixelFormat::R32Float,
    PixelFormat::R32Uint,
    PixelFormat::R32Uint,
];

/// The attachments' extent: two rows of 16x16 tiles, the last column and row partial.
const AREA: [u64; 2] = [40, 24];

/// The fixture's `out` records: 512 of 12 words.
const RECORDS: u64 = 512 * 12 * 4;

/// A state of the fixture's tile function.
pub(super) fn tile_state(device: &Device) -> RenderPipelineState {
    let ll = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/fixtures/public/engine/k10ts_tile_grid.ll");
    let bitcode = metal2vulkan_devtools::llvm_assemble(&ll).unwrap();
    let library = metal2vulkan::library::Library::from_air(&bitcode).unwrap();
    let library = device.new_library_with_compile_library(library);
    let function = library.new_function_with_name("k10ts_tile_grid").unwrap();
    let mut descriptor =
        TileRenderPipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    for (slot, &format) in descriptor.color_attachment_formats.iter_mut().zip(&FORMATS) {
        *slot = Some(format);
    }
    device
        .new_tile_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// The pass's attachments and the fixture's records.
pub(super) struct Targets {
    pub(super) colors: [Texture; 3],
    pub(super) out: Buffer,
}

/// The value of attachment 0 and attachment 1 at pixel (`x`, `y`) before the pass.
fn f(x: u64, y: u64) -> f32 {
    (x + 100 * y) as f32
}

fn v(x: u64, y: u64) -> u32 {
    (7 * x + y) as u32
}

impl Targets {
    pub(super) fn new(device: &Device) -> Targets {
        let attachment = |format, texel: &dyn Fn(u64, u64) -> u32| {
            let [width, height] = AREA;
            let mut descriptor = TextureDescriptor::texture_2d(format, width, height, false);
            descriptor.usage = TextureUsage::RENDER_TARGET;
            descriptor.resource_options = ResourceOptions::shared();
            let texture = device.new_texture(&descriptor).unwrap();
            let bytes: Vec<u8> = pixels()
                .flat_map(|(x, y)| texel(x, y).to_le_bytes())
                .collect();
            let region = Region::new_2d(0, 0, width, height);
            texture
                .replace_region(region, 0, 0, &bytes, width * 4, 0)
                .unwrap();
            texture
        };
        Targets {
            colors: [
                attachment(FORMATS[0], &|x, y| f(x, y).to_bits()),
                attachment(FORMATS[1], &v),
                attachment(FORMATS[2], &|_, _| 0),
            ],
            out: device
                .new_buffer(RECORDS, ResourceOptions::shared())
                .unwrap(),
        }
    }

    /// Whether every pixel holds what the fixture's dispatch over every tile stores, as Metal's
    /// does: `f * 2 + 1`, `v * 3 + k` and its tile's and cell's tag.
    pub(super) fn stored(&self) -> bool {
        let words = |texture: &Texture| {
            let [width, height] = AREA;
            let mut bytes = vec![0u8; (width * height * 4) as usize];
            let region = Region::new_2d(0, 0, width, height);
            texture
                .get_bytes(&mut bytes, width * 4, 0, region, 0, 0)
                .unwrap();
            let (words, _) = bytes.as_chunks::<4>();
            words
                .iter()
                .map(|w| u32::from_le_bytes(*w))
                .collect::<Vec<_>>()
        };
        let [fs, vs, tags] = self.colors.each_ref().map(words);
        pixels().enumerate().all(|(i, (x, y))| {
            let k = 1 + (x % 2) as u32 + 2 * (y % 2) as u32;
            let (tile, cell) = ((x / 16, y / 16), (x % 16, y % 16));
            let tag = 0xA000_0000
                | ((tile.1 as u32) << 20)
                | ((tile.0 as u32) << 16)
                | ((cell.1 as u32) << 8)
                | cell.0 as u32;
            f32::from_bits(fs[i]) == f(x, y) * 2.0 + 1.0
                && vs[i] == v(x, y) * 3 + k
                && tags[i] == tag
        })
    }
}

/// Every pixel's (x, y), in row order.
fn pixels() -> impl Iterator<Item = (u64, u64)> {
    (0..AREA[1]).flat_map(|y| (0..AREA[0]).map(move |x| (x, y)))
}

/// A command buffer of `queue` whose one pass over `targets`, of 16x16 tiles, each attachment
/// loaded and stored, dispatches `state` on every tile, 8x8 threads a tile: encoded, uncommitted.
fn dispatch(queue: &CommandQueue, state: &RenderPipelineState, targets: &Targets) -> CommandBuffer {
    let mut pass = RenderPassDescriptor::new();
    for (attachment, target) in pass.color_attachments.iter_mut().zip(&targets.colors) {
        attachment.texture = Some(target.clone());
        attachment.load_action = LoadAction::Load;
        attachment.store_action = StoreAction::Store;
    }
    pass.tile_width = 16;
    pass.tile_height = 16;
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
    encoder.set_render_pipeline_state(state).unwrap();
    encoder.set_tile_buffer(Some(&targets.out), 0, 0).unwrap();
    encoder
        .dispatch_threads_per_tile(Size::new(8, 8, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer
}

/// Wait until `command_buffer` completed, at most [`WAIT`], and check it completed.
fn completed(command_buffer: &CommandBuffer) {
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(
        command_buffer.status(),
        CommandBufferStatus::Completed,
        "{:?} {:?}",
        command_buffer.error(),
        command_buffer.failed_pipeline()
    );
}

/// A tile dispatch whose pipeline the pool is held from making returns at once, its making
/// handed to the pool; its command buffer waits for it, and so does one committed after it on its
/// queue. Released, the dispatch stores what Metal's stores, and the later command buffer
/// completes after it.
#[test]
fn a_tile_dispatch_waits_for_no_making_and_its_command_buffer_completes_after_it() {
    let scratch = Scratch::new("tile-pending");
    let device = device(&scratch.0);
    let state = tile_state(&device);
    let targets = Targets::new(&device);
    let queue = device.new_command_queue().unwrap();
    let hold = held::hold();

    let before = device.work_counts();
    let pending = dispatch(&queue, &state, &targets);
    let after = queue.command_buffer().unwrap();
    pending.commit().unwrap();
    after.commit().unwrap();
    let held_counts = device.work_counts();
    assert_eq!(
        held_counts.tile_pipelines_pooled,
        before.tile_pipelines_pooled + 1,
        "the dispatch handed its making to the pool"
    );
    assert_eq!(
        held_counts.variant_pipelines_made, before.variant_pipelines_made,
        "nothing was made while the pool is held"
    );
    assert!(
        after.wait_until_completed_for(WATCHED).is_err(),
        "a command buffer after the pending dispatch's on its queue waits for it"
    );
    assert_ne!(pending.status(), CommandBufferStatus::Completed);

    drop(hold);
    completed(&pending);
    completed(&after);
    assert!(targets.stored(), "the dispatch stored what Metal's stores");
}

/// A tile state released while its dispatch's making waits in the pool stops it, returning at
/// once: its dispatch's command buffer was dropped uncommitted, so nothing else holds the state. A
/// second state of the same function shares its variant, and its dispatch took that making as it
/// stood, handing nothing to the pool; once the making failed, its command buffer makes the
/// pipeline itself, on the scheduler's thread while the pool is still held, and completes, storing
/// what Metal's stores.
#[test]
fn a_released_tile_state_stops_its_making_and_a_dispatch_that_took_it_makes_its_own() {
    let scratch = Scratch::new("tile-release");
    let device = device(&scratch.0);
    let first = tile_state(&device);
    let second = tile_state(&device);
    let targets = Targets::new(&device);
    let queue = device.new_command_queue().unwrap();
    let hold = held::hold();

    let before = device.work_counts();
    let dropped = dispatch(&queue, &first, &targets);
    let pending = dispatch(&queue, &second, &targets);
    assert_eq!(
        device.work_counts().tile_pipelines_pooled,
        before.tile_pipelines_pooled + 1,
        "the second state's dispatch took the first's making in flight"
    );
    pending.commit().unwrap();
    let released = Instant::now();
    drop(dropped);
    drop(first);
    assert!(
        released.elapsed() < WATCHED,
        "the release waited for its making"
    );
    completed(&pending);
    assert!(targets.stored(), "the dispatch stored what Metal's stores");
    drop(hold);
}
