//! Tile dispatches through the runtime on the device the cfg selects, under the validation layer.
//!
//! - Every committed public tile kernel runs through the runtime as the Metal oracle dispatches
//!   it (`runtime_case::encoded_tile`) against its committed Metal observation, or the output of
//!   the engine candidate run afresh where the two differ ([`census_with`]).
//!
//! The owned fixture `k10ts_tile_grid` is the tile function of the rest: over an implicit
//! imageblock of an `R32Float` and two `R32Uint` attachments, each of its threads reads the 2x2
//! cells at twice its position, writes `f * 2 + 1`, `v * 3 + k` (`k = 1 + x % 2 + 2 * (y % 2)` of
//! the cell) and a tag of its tile and cell.
//!
//! - A dispatch runs every tile of the pass's tile size over the render area, a partial tile
//!   staging the attachments' pixels it covers, and its stores reach the attachments.
//! - A dispatch sees the pass's clears and an earlier dispatch's stores.
//! - A dispatch's scratch images are announced once (`tile-dispatch-scratch`), and so is its
//!   pass's split, its reason naming the dispatches (`render-pass-split`).
//! - A region tile dispatch runs the tiles its region touches, as Metal takes the region.
//! - A pass's tile size and memory are refused where Metal's validation refuses them, and a
//!   dispatch where it refuses its state or its threads per tile, but for a depth other than 1,
//!   which runs as 1, and threads per tile past Metal's limit for the state, which run nothing
//!   whether or not they are also past the tile, as plain Metal runs them.
//! - A tile state of a color more than the pass's, past the ones its function reaches, and a pass
//!   of a color more than the state's, run as the state of the pass's colors runs, the color the
//!   function does not reach untouched, as plain Metal runs them.

use super::degrade_notices::{assert_once, made_tile_kinds};
use super::kernel_textures::census_with;
use super::shared_kinds;
use super::tile_states::on_device_with;
use super::tile_states::{descriptor, on_device};
use super::{
    at_plan, layer_reported_nothing_at_all, public_cases, read_buffer, validated_device_with,
    Messages, ScratchCache, WAIT,
};
use crate::case::Stage;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use metal2vulkan_runtime::{
    ClearColor, CommandBufferStatus, Degradation, Device, DrawProblem, Error, Invalid, LoadAction,
    Nil, PipelineOptions, Problem, Region, RenderCommandEncoder, RenderPassDescriptor,
    RenderPassProblem, RenderPipelineState, ResourceOptions, Size, StoreAction, Texture,
    TextureDescriptor,
};

/// The kind a tile dispatch takes (`degrade_notices`' closure).
pub(super) const TILE_DISPATCHES: [Degradation; 1] = [Degradation::TileDispatchScratch];

/// Every committed public tile kernel, a kernel over a tile pass's render targets (its implicit
/// imageblock's coverage), passes the [`census_with`].
#[test]
fn tile_kernels_match_metal() {
    let cases = public_cases(|case| {
        case.stage == Stage::Kernel
            && case.public_source.is_some()
            && case
                .imageblock
                .as_ref()
                .is_some_and(|imageblock| imageblock.implicit_coverage.is_some())
    })
    .unwrap();
    assert!(!cases.is_empty(), "no committed case is a tile kernel");
    census_with(at_plan, "runtime tile kernels", &cases);
}

/// The attachments' formats: the fixture's implicit imageblock.
const FORMATS: [PixelFormat; 3] = [
    PixelFormat::R32Float,
    PixelFormat::R32Uint,
    PixelFormat::R32Uint,
];

/// The attachments' extent: two rows of tiles, the last column and row partial.
const AREA: [u64; 2] = [40, 24];

/// The fixture's `out` records: 512 of 12 words.
const RECORDS: u64 = 512 * 12 * 4;

/// The fixture's tile state.
fn tile_state(device: &Device) -> RenderPipelineState {
    device
        .new_tile_render_pipeline_state(
            &descriptor(device, "k10ts_tile_grid", &FORMATS),
            &PipelineOptions::default(),
        )
        .unwrap()
}

/// An attachment of `format` over [`AREA`] holding `texel` of each pixel.
fn attachment(device: &Device, format: PixelFormat, texel: impl Fn(u64, u64) -> u32) -> Texture {
    let [width, height] = AREA;
    let mut descriptor = TextureDescriptor::texture_2d(format, width, height, false);
    descriptor.usage = TextureUsage::RENDER_TARGET;
    descriptor.resource_options = ResourceOptions::shared();
    let texture = device.new_texture(&descriptor).unwrap();
    let bytes: Vec<u8> = (0..height)
        .flat_map(|y| (0..width).map(move |x| (x, y)))
        .flat_map(|(x, y)| texel(x, y).to_le_bytes())
        .collect();
    texture
        .replace_region(
            Region::new_2d(0, 0, width, height),
            0,
            0,
            &bytes,
            width * 4,
            0,
        )
        .unwrap();
    texture
}

/// Every texel of `texture`, as 32-bit words in row order.
fn words(texture: &Texture) -> Vec<u32> {
    let [width, height] = AREA;
    let mut bytes = vec![0u8; (width * height * 4) as usize];
    texture
        .get_bytes(
            &mut bytes,
            width * 4,
            0,
            Region::new_2d(0, 0, width, height),
            0,
            0,
        )
        .unwrap();
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
        .collect()
}

/// A pass over `targets` of 16x16 tiles, each attachment loaded with `load` (cleared to
/// `clear`) and stored.
fn pass(targets: &[Texture], load: LoadAction, clear: f64) -> RenderPassDescriptor {
    let mut pass = RenderPassDescriptor::new();
    for (attachment, target) in pass.color_attachments.iter_mut().zip(targets) {
        attachment.texture = Some(target.clone());
        attachment.load_action = load;
        attachment.store_action = StoreAction::Store;
        attachment.clear_color = ClearColor::new(clear, 0.0, 0.0, 0.0);
    }
    pass.tile_width = 16;
    pass.tile_height = 16;
    pass
}

/// Encode `pass` on a new command buffer of `device`, `encode` its commands, and run it.
fn run(
    device: &Device,
    pass: &RenderPassDescriptor,
    encode: impl FnOnce(&mut RenderCommandEncoder),
) {
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.render_command_encoder(pass).unwrap();
    encode(&mut encoder);
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
}

/// The fixture's tag of the pixel at (`x`, `y`) in 16x16 tiles.
fn tag(x: u64, y: u64) -> u32 {
    let (tile, cell) = ((x / 16, y / 16), (x % 16, y % 16));
    0xA000_0000
        | ((tile.1 as u32) << 20)
        | ((tile.0 as u32) << 16)
        | ((cell.1 as u32) << 8)
        | cell.0 as u32
}

/// The fixture's `k` of the pixel at (`x`, `y`).
fn k(x: u64, y: u64) -> u32 {
    1 + (x % 2) as u32 + 2 * (y % 2) as u32
}

/// Every pixel's (x, y), in row order.
fn pixels() -> impl Iterator<Item = (u64, u64)> {
    (0..AREA[1]).flat_map(|y| (0..AREA[0]).map(move |x| (x, y)))
}

/// A tile dispatch runs each tile of the pass, and the device counts it once
/// (`WorkCounts::tile_dispatches_recorded`).
#[test]
fn a_tile_dispatch_runs_each_tile_of_the_pass() {
    on_device(|device| {
        let state = tile_state(device);
        let f = |x: u64, y: u64| (x + 100 * y) as f32;
        let v = |x: u64, y: u64| (7 * x + y) as u32;
        let targets = [
            attachment(device, FORMATS[0], |x, y| f(x, y).to_bits()),
            attachment(device, FORMATS[1], v),
            attachment(device, FORMATS[2], |_, _| 0),
        ];
        let out = device
            .new_buffer(RECORDS, ResourceOptions::shared())
            .unwrap();
        let recorded = device.work_counts().tile_dispatches_recorded;
        run(device, &pass(&targets, LoadAction::Load, 0.0), |encoder| {
            assert_eq!(
                (encoder.tile_width(), encoder.tile_height()),
                (Ok(16), Ok(16))
            );
            encoder.set_render_pipeline_state(&state).unwrap();
            encoder.set_tile_buffer(Some(&out), 0, 0).unwrap();
            encoder
                .dispatch_threads_per_tile(Size::new(8, 8, 1))
                .unwrap();
        });
        assert_eq!(
            device.work_counts().tile_dispatches_recorded,
            recorded + 1,
            "the tile dispatches counted"
        );
        let [fs, vs, tags] = [&targets[0], &targets[1], &targets[2]].map(words);
        for (i, (x, y)) in pixels().enumerate() {
            assert_eq!(
                f32::from_bits(fs[i]),
                f(x, y) * 2.0 + 1.0,
                "f at ({x}, {y})"
            );
            assert_eq!(vs[i], v(x, y) * 3 + k(x, y), "v at ({x}, {y})");
            assert_eq!(tags[i], tag(x, y), "tag at ({x}, {y})");
        }
    });
}

/// A tile state whose colors differ from the pass's past the last its function reaches, which
/// Metal's validation stops, runs as plain Metal runs it (Metal probes on an M4: a state of one
/// color more, `R32Float` or `RGBA32Float`, than its pass, and a pass of one color more than the
/// state, ran every tile, the colors the function reaches stored as under matching formats and the
/// other color untouched): the fixture's state with a fourth color, and its state in a pass of a
/// fourth, store the matching dispatch's attachments and records, the pass's fourth color kept.
#[test]
fn a_tile_state_of_other_colors_past_the_ones_its_function_reaches_runs() {
    on_device_with(at_plan, |device| {
        let f = |x: u64, y: u64| ((x + 100 * y) as f32).to_bits();
        let v = |x: u64, y: u64| (7 * x + y) as u32;
        let kept = |x: u64, y: u64| (0x5000_0000 + 64 * y + x) as u32;
        let dispatched = |state: &RenderPipelineState, fourth: bool| {
            let mut targets = vec![
                attachment(device, FORMATS[0], f),
                attachment(device, FORMATS[1], v),
                attachment(device, FORMATS[2], |_, _| 0),
            ];
            if fourth {
                targets.push(attachment(device, PixelFormat::R32Uint, kept));
            }
            let out = device
                .new_buffer(RECORDS, ResourceOptions::shared())
                .unwrap();
            run(device, &pass(&targets, LoadAction::Load, 0.0), |encoder| {
                encoder.set_render_pipeline_state(state).unwrap();
                encoder.set_tile_buffer(Some(&out), 0, 0).unwrap();
                encoder
                    .dispatch_threads_per_tile(Size::new(8, 8, 1))
                    .unwrap();
            });
            let words: Vec<Vec<u32>> = targets.iter().map(words).collect();
            (words, read_buffer(&out, 0, RECORDS).unwrap())
        };
        let matching = dispatched(&tile_state(device), false);
        let mut more = FORMATS.to_vec();
        more.push(PixelFormat::R32Float);
        let wider = device
            .new_tile_render_pipeline_state(
                &descriptor(device, "k10ts_tile_grid", &more),
                &PipelineOptions::default(),
            )
            .unwrap();
        assert!(
            dispatched(&wider, false) == matching,
            "a state of a color more than the pass's"
        );
        let (mut words, records) = dispatched(&tile_state(device), true);
        let fourth = words.pop().unwrap();
        assert!(
            (words, records) == matching,
            "a pass of a color more than the state's"
        );
        assert!(
            pixels()
                .zip(&fourth)
                .all(|((x, y), &word)| word == kept(x, y)),
            "the pass's fourth color kept"
        );
    });
}

/// Threads per tile of depth 2 and 4, which Metal's validation stops, run as depth 1, as plain
/// Metal runs them (a Metal probe on an M4: 64 threads a tile of 8x8 at either depth, each `z` 0):
/// the attachments and the records are the one-deep dispatch's. A second layer of threads would
/// read and write each cell twice and record past each tile's 64 records.
#[test]
fn a_tile_dispatch_deeper_than_one_runs_one_deep() {
    on_device_with(at_plan, |device| {
        let state = tile_state(device);
        let dispatched = |depth: u64| {
            let targets = [
                attachment(device, FORMATS[0], |x, y| ((x + 100 * y) as f32).to_bits()),
                attachment(device, FORMATS[1], |x, y| (7 * x + y) as u32),
                attachment(device, FORMATS[2], |_, _| 0),
            ];
            let out = device
                .new_buffer(RECORDS, ResourceOptions::shared())
                .unwrap();
            run(device, &pass(&targets, LoadAction::Load, 0.0), |encoder| {
                encoder.set_render_pipeline_state(&state).unwrap();
                encoder.set_tile_buffer(Some(&out), 0, 0).unwrap();
                encoder
                    .dispatch_threads_per_tile(Size::new(8, 8, depth))
                    .unwrap();
            });
            (
                [&targets[0], &targets[1], &targets[2]].map(words),
                read_buffer(&out, 0, RECORDS).unwrap(),
            )
        };
        let one = dispatched(1);
        for depth in [2, 4] {
            assert!(dispatched(depth) == one, "threads per tile 8x8x{depth}");
        }
    });
}

#[test]
fn a_tile_dispatch_sees_a_clear_and_an_earlier_dispatch() {
    on_device(|device| {
        let state = tile_state(device);
        let targets = FORMATS.map(|format| attachment(device, format, |_, _| 0xDEAD_BEEF));
        let out = device
            .new_buffer(RECORDS, ResourceOptions::shared())
            .unwrap();
        run(device, &pass(&targets, LoadAction::Clear, 2.0), |encoder| {
            encoder.set_render_pipeline_state(&state).unwrap();
            encoder.set_tile_buffer(Some(&out), 0, 0).unwrap();
            for _ in 0..2 {
                encoder
                    .dispatch_threads_per_tile(Size::new(8, 8, 1))
                    .unwrap();
            }
        });
        let [fs, vs, tags] = [&targets[0], &targets[1], &targets[2]].map(words);
        for (i, (x, y)) in pixels().enumerate() {
            // 2 -> 5 -> 11; 2 -> 6 + k -> 18 + 4k.
            assert_eq!(f32::from_bits(fs[i]), 11.0, "f at ({x}, {y})");
            assert_eq!(vs[i], 18 + 4 * k(x, y), "v at ({x}, {y})");
            assert_eq!(tags[i], tag(x, y), "tag at ({x}, {y})");
        }
    });
}

/// Two passes of two dispatches each on one device: the dispatches' scratch images are announced
/// once, and the passes' split once, its reason naming the dispatches; beside them, what the tile
/// state's stage records ([`made_tile_kinds`]).
#[test]
fn a_tile_dispatch_announces_its_scratch_and_split_once() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    // Beside the test's kinds, what its `Shared` buffer forces on the device and its tile state's
    // making announces.
    let mut kinds = vec![
        Degradation::TileDispatchScratch,
        Degradation::RenderPassSplit,
    ];
    {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), at_plan).unwrap();
        kinds.extend(shared_kinds(&device));
        let state = tile_state(&device);
        kinds.extend(made_tile_kinds(
            &device,
            &descriptor(&device, "k10ts_tile_grid", &FORMATS),
        ));
        let targets = FORMATS.map(|format| attachment(&device, format, |_, _| 0));
        let out = device
            .new_buffer(RECORDS, ResourceOptions::shared())
            .unwrap();
        for _ in 0..2 {
            run(&device, &pass(&targets, LoadAction::Load, 0.0), |encoder| {
                encoder.set_render_pipeline_state(&state).unwrap();
                encoder.set_tile_buffer(Some(&out), 0, 0).unwrap();
                for _ in 0..2 {
                    encoder
                        .dispatch_threads_per_tile(Size::new(8, 8, 1))
                        .unwrap();
                }
            });
        }
    }
    layer_reported_nothing_at_all(&messages);
    let notices = messages.take_notices();
    assert_once(&notices, &kinds);
    let split = notices
        .iter()
        .filter_map(|message| message.degradation.as_ref())
        .find(|notice| notice.degradation == Degradation::RenderPassSplit)
        .unwrap();
    assert!(
        split.reason.contains("it runs 2 tile dispatches"),
        "{}",
        split.reason
    );
}

/// Threads per tile past Metal's limit for the state run nothing and the dispatch is taken, as
/// plain Metal runs it, whether or not they are also past the pass's tile (an Apple M3 Max, macOS
/// 27.0.1, probe `refusals_without_a_named_device` shapes `tilec-32x64` and `tilec-64x64` over a
/// 32 by 32 target in 16 by 16 tiles, four fresh processes each: a visit counter per position per
/// tile counted no thread at all, with no error, where 32 by 32 threads per tile -- the limit
/// itself -- ran 2304 of them). So the pass keeps every pixel, where a dispatch that runs stores
/// `f * 2 + 1`. Threads per tile past the tile but within the limit stay refused, the limit itself
/// among them (`DrawProblem::ThreadsPerTilePastTile`, whose doc holds what plain Metal runs).
#[test]
fn a_tile_dispatch_past_metals_limit_runs_nothing_past_the_tile_too() {
    on_device(|device| {
        let state = tile_state(device);
        let f = |x: u64, y: u64| (x + 100 * y) as f32;
        let v = |x: u64, y: u64| (7 * x + y) as u32;
        let targets = [
            attachment(device, FORMATS[0], |x, y| f(x, y).to_bits()),
            attachment(device, FORMATS[1], v),
            attachment(device, FORMATS[2], |_, _| 0),
        ];
        let out = device
            .new_buffer(RECORDS, ResourceOptions::shared())
            .unwrap();
        let recorded = device.work_counts().tile_dispatches_recorded;
        run(device, &pass(&targets, LoadAction::Load, 0.0), |encoder| {
            encoder.set_render_pipeline_state(&state).unwrap();
            encoder.set_tile_buffer(Some(&out), 0, 0).unwrap();
            // 2048 and 4096 threads, past the state's Metal limit of 1024 (the fixture's
            // descriptor sets none) and past the pass's 16 by 16 tile.
            for threads in [Size::new(32, 64, 1), Size::new(64, 64, 1)] {
                assert_eq!(
                    encoder.dispatch_threads_per_tile(threads),
                    Ok(()),
                    "{threads:?} past Metal's limit"
                );
            }
            // 1024 threads, the limit itself, past the tile.
            assert_eq!(
                encoder.dispatch_threads_per_tile(Size::new(64, 16, 1)),
                Err(Error::Invalid(Invalid::new(
                    "dispatch_threads_per_tile",
                    Problem::Draw(DrawProblem::ThreadsPerTilePastTile {
                        threads: [64, 16],
                        tile: [16, 16],
                    }),
                )))
            );
        });
        assert_eq!(
            device.work_counts().tile_dispatches_recorded,
            recorded + 2,
            "the dispatches past Metal's limit counted, the refused one not"
        );
        let [fs, vs, tags] = [&targets[0], &targets[1], &targets[2]].map(words);
        for (i, (x, y)) in pixels().enumerate() {
            assert_eq!(f32::from_bits(fs[i]), f(x, y), "f at ({x}, {y})");
            assert_eq!(vs[i], v(x, y), "v at ({x}, {y})");
            assert_eq!(tags[i], 0, "tag at ({x}, {y})");
        }
    });
}

/// The passes and dispatches Metal refuses are refused, and a refused dispatch is not counted
/// (`WorkCounts::tile_dispatches_recorded`).
#[test]
fn a_tile_pass_and_dispatch_are_refused_where_metal_refuses_them() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let targets = FORMATS.map(|format| attachment(device, format, |_, _| 0));
        // The pass: Metal's tile sizes only, and its tile memory; plain Metal answers nil for
        // each refused pass (`tile_pass.rs`).
        for ([width, height], problem) in [
            (
                [8, 8],
                RenderPassProblem::TileSize {
                    width: 8,
                    height: 8,
                },
            ),
            (
                [16, 32],
                RenderPassProblem::TileSize {
                    width: 16,
                    height: 32,
                },
            ),
        ] {
            let mut descriptor = pass(&targets, LoadAction::Load, 0.0);
            descriptor.tile_width = width;
            descriptor.tile_height = height;
            let command_buffer = queue.command_buffer().unwrap();
            assert_eq!(
                command_buffer
                    .render_command_encoder(&descriptor)
                    .map(|_| ()),
                Err(Error::Nil(Nil::new(
                    "render_command_encoder",
                    Problem::RenderPass(problem),
                )))
            );
        }
        let mut descriptor = pass(&targets, LoadAction::Load, 0.0);
        descriptor.imageblock_sample_length = 65;
        let command_buffer = queue.command_buffer().unwrap();
        assert_eq!(
            command_buffer
                .render_command_encoder(&descriptor)
                .map(|_| ()),
            Err(Error::Nil(Nil::new(
                "render_command_encoder",
                Problem::RenderPass(RenderPassProblem::PerSampleStorage { bytes: 72 }),
            )))
        );
        // The dispatch: a tile state set, and threads per tile that fit the tile.
        let state = tile_state(device);
        let mut unset = pass(&targets, LoadAction::Load, 0.0);
        unset.tile_width = 0;
        unset.tile_height = 0;
        let command_buffer = queue.command_buffer().unwrap();
        let recorded = device.work_counts().tile_dispatches_recorded;
        let mut encoder = command_buffer.render_command_encoder(&unset).unwrap();
        assert_eq!(
            (encoder.tile_width(), encoder.tile_height()),
            (Ok(32), Ok(32))
        );
        let refused = |problem| {
            Err(Error::Invalid(Invalid::new(
                "dispatch_threads_per_tile",
                problem,
            )))
        };
        assert_eq!(
            encoder.dispatch_threads_per_tile(Size::new(8, 8, 1)),
            refused(Problem::NoPipelineState)
        );
        encoder.set_render_pipeline_state(&state).unwrap();
        assert_eq!(
            encoder.dispatch_threads_per_tile(Size::new(0, 8, 1)),
            refused(Problem::EmptyThreadgroup)
        );
        assert_eq!(
            encoder.dispatch_threads_per_tile(Size::new(64, 8, 1)),
            refused(Problem::Draw(DrawProblem::ThreadsPerTilePastTile {
                threads: [64, 8],
                tile: [32, 32],
            }))
        );
        encoder.end_encoding().unwrap();
        assert_eq!(
            device.work_counts().tile_dispatches_recorded,
            recorded,
            "refused dispatches counted"
        );
    });
}

/// A region tile dispatch runs the tiles its region touches as Metal does (on an M3 Max and an M4):
/// over [`AREA`] in 16x16 tiles (3 by 2), the tiles from the origin's to the one holding the end,
/// each origin and size taken in 32 bits; a region of no width at a tile's edge runs nothing and
/// one inside a tile runs that tile; a region past the pass is cut to it; an end that wraps past
/// `2^32 - 1` to 0 runs to the pass's last tile and one that wraps further runs nothing. The array
/// index draws into the pass's one layer. Every other pixel keeps its texels.
#[test]
fn a_region_tile_dispatch_runs_the_tiles_its_region_touches() {
    /// A region, its array index, and the tiles (column, row) it runs.
    type RegionCase = (Region, Option<u32>, &'static [(u64, u64)]);
    const WRAP: u64 = 1 << 32;
    let cases: [RegionCase; 9] = [
        (Region::new_2d(8, 4, 20, 10), None, &[(0, 0), (1, 0)]),
        (Region::new_2d(20, 20, 0, 0), None, &[(1, 1)]),
        (Region::new_2d(16, 0, 0, 16), None, &[]),
        (Region::new_2d(32, 16, 100, 100), None, &[(2, 1)]),
        (Region::new_2d(48, 0, 16, 16), None, &[]),
        (Region::new_2d(WRAP + 16, 0, 16, 16), None, &[(1, 0)]),
        (
            Region::new_2d(16, 0, WRAP - 16, 16),
            None,
            &[(1, 0), (2, 0)],
        ),
        (Region::new_2d(16, 0, WRAP - 8, 16), None, &[]),
        (Region::new_2d(0, 16, 16, 8), Some(3), &[(0, 1)]),
    ];
    on_device_with(at_plan, |device| {
        let state = tile_state(device);
        let f = |x: u64, y: u64| (x + 100 * y) as f32;
        let v = |x: u64, y: u64| (7 * x + y) as u32;
        for (region, index, tiles) in cases {
            let targets = [
                attachment(device, FORMATS[0], |x, y| f(x, y).to_bits()),
                attachment(device, FORMATS[1], v),
                attachment(device, FORMATS[2], |_, _| 0),
            ];
            let out = device
                .new_buffer(RECORDS, ResourceOptions::shared())
                .unwrap();
            run(device, &pass(&targets, LoadAction::Load, 0.0), |encoder| {
                encoder.set_render_pipeline_state(&state).unwrap();
                encoder.set_tile_buffer(Some(&out), 0, 0).unwrap();
                encoder
                    .dispatch_threads_per_tile_in_region(Size::new(8, 8, 1), region, index)
                    .unwrap();
            });
            let [fs, vs, tags] = [&targets[0], &targets[1], &targets[2]].map(words);
            for (i, (x, y)) in pixels().enumerate() {
                let ran = tiles.contains(&(x / 16, y / 16));
                let expected = if ran {
                    (f(x, y) * 2.0 + 1.0, v(x, y) * 3 + k(x, y), tag(x, y))
                } else {
                    (f(x, y), v(x, y), 0)
                };
                assert_eq!(
                    (f32::from_bits(fs[i]), vs[i], tags[i]),
                    expected,
                    "{region:?}: the pixel at ({x}, {y})"
                );
            }
        }
        // Refused as `dispatch_threads_per_tile` is.
        let targets = FORMATS.map(|format| attachment(device, format, |_, _| 0));
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer
            .render_command_encoder(&pass(&targets, LoadAction::Load, 0.0))
            .unwrap();
        encoder.set_render_pipeline_state(&state).unwrap();
        assert_eq!(
            encoder.dispatch_threads_per_tile_in_region(
                Size::new(32, 32, 1),
                Region::new_2d(0, 0, 16, 16),
                None
            ),
            Err(Error::Invalid(Invalid::new(
                "dispatch_threads_per_tile_in_region",
                Problem::Draw(DrawProblem::ThreadsPerTilePastTile {
                    threads: [32, 32],
                    tile: [16, 16],
                }),
            )))
        );
        encoder.end_encoding().unwrap();
    });
}
