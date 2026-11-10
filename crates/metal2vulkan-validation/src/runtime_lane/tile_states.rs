//! Tile pipeline states through the runtime on the device the cfg selects, under the validation
//! layer. The tile functions are the owned fixtures `k10ts_tile_grid` (an implicit imageblock of
//! three 32-bit attachments) and `k10ts_alias_probe` (an explicit imageblock, `Mixed`, aliased onto
//! four attachments).
//!
//! - A tile state answers Metal's queries as an M3 Max and an M4 do: `imageblockSampleLength` is
//!   the attachments' texels packed, or the aliased layout's size whatever the formats pack to; its
//!   memory length for a tile is that times the tile's cells, unrounded; it supports no indirect
//!   command buffers and echoes the descriptor's `threadgroupSizeMatchesTileSize`. Its thread limit
//!   is Metal's 1024 where the device runs it, lowered by the descriptor's. Where the descriptor
//!   requires a threadgroup size, the limit is that size's threads.
//! - Creation refuses what Metal's does: a function reaching an attachment the descriptor gives
//!   no format.
//! - A binary archive holds a tile pipeline it was given (`add_tile_render_pipeline_functions`),
//!   and a creation that fails on a miss is refused for one it was not, naming the tile function.
//! - A render encoder takes a tile state whose formats and sample count are its pass's and whose
//!   imageblock fits the pass's texels, refuses the rest as Metal's validation does, and refuses
//!   a draw with a tile state set.

use super::render_state::texture;
use super::{layer_reported_nothing_at_all, validated_device_with, Messages, ScratchCache};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::error::DescriptorError;
use metal2vulkan::metal::{PixelFormat, PrimitiveType};
use metal2vulkan::pipeline::{StageSource, TileRenderPipelineDescriptor};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::DeviceOptions;
use metal2vulkan_runtime::{
    ArchivedPipeline, Attachment, BinaryArchiveDescriptor, BinaryArchiveMiss, Device, DrawProblem,
    Error, Invalid, Library, PipelineOption, PipelineOptions, Problem, RenderPassDescriptor,
    RenderPipelineState, Size,
};

/// The implicit fixture's attachments.
const GRID: [PixelFormat; 3] = [
    PixelFormat::R32Float,
    PixelFormat::R32Uint,
    PixelFormat::R32Uint,
];

/// The aliased fixture's attachments: they pack to 18 bytes, its `Mixed` layout is 24.
const ALIASED: [PixelFormat; 4] = [
    PixelFormat::R16Float,
    PixelFormat::R32Float,
    PixelFormat::RGBA16Float,
    PixelFormat::R16Float,
];

/// Run `test` on a validated device of the lane, then fail on any validation message.
pub(super) fn on_device(test: impl FnOnce(&Device)) {
    on_device_with(|_| {}, test)
}

/// [`on_device`], `adjust` applied to the options of each device it makes (`at_plan` gives them
/// their queue plan).
pub(super) fn on_device_with(adjust: fn(&mut DeviceOptions), test: impl FnOnce(&Device)) {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), adjust).unwrap();
        test(&device);
    }
    layer_reported_nothing_at_all(&messages);
}

/// The library of the owned fixture `stem` on `device`.
fn library(device: &Device, stem: &str) -> Library {
    let bitcode = llvm_assemble(&engine_dir().join(format!("{stem}.ll"))).unwrap();
    device.new_library_with_air(&bitcode).unwrap()
}

/// The tile descriptor of `stem`'s tile function with color attachments of `formats`.
pub(super) fn descriptor(
    device: &Device,
    stem: &str,
    formats: &[PixelFormat],
) -> TileRenderPipelineDescriptor {
    let function = library(device, stem).new_function_with_name(stem).unwrap();
    let mut descriptor =
        TileRenderPipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    for (slot, &format) in descriptor.color_attachment_formats.iter_mut().zip(formats) {
        *slot = Some(format);
    }
    descriptor
}

/// The state of `descriptor` under the default options.
fn state(
    device: &Device,
    descriptor: &TileRenderPipelineDescriptor,
) -> Result<RenderPipelineState, Error> {
    device.new_tile_render_pipeline_state(descriptor, &PipelineOptions::default())
}

/// The aliased fixture's state, or `None` where the device stores no 16-bit attachment its tile
/// staging needs (`Unsupported`).
fn aliased_state(device: &Device) -> Option<RenderPipelineState> {
    match state(device, &descriptor(device, "k10ts_alias_probe", &ALIASED)) {
        Ok(state) => Some(state),
        Err(Error::Unsupported(why)) => {
            eprintln!("k10ts_alias_probe not made on this device: {why}");
            None
        }
        Err(error) => panic!("k10ts_alias_probe: {error}"),
    }
}

#[test]
fn a_tile_state_answers_metals_queries() {
    on_device(|device| {
        let mut grid = descriptor(device, "k10ts_tile_grid", &GRID);
        let tile = state(device, &grid).unwrap();
        assert_eq!(tile.imageblock_sample_length(), Ok(12));
        for (width, height, length) in
            [(16, 16, 3072), (32, 16, 6144), (32, 32, 12288), (3, 5, 180)]
        {
            assert_eq!(
                tile.imageblock_memory_length_for_dimensions(Size::new(width, height, 7)),
                Ok(length),
                "{width}x{height}"
            );
        }
        assert!(!tile.support_indirect_command_buffers());
        // The tile function takes no indirect command buffer.
        assert!(!tile.can_encode_indirect_commands());
        // Nor does it write a texture: its imageblock is no texture.
        assert!(!tile.tile_writes_textures() && !tile.fragment_writes_textures());
        assert!(!tile.threadgroup_size_matches_tile_size());
        let max = tile.max_total_threads_per_threadgroup();
        assert!((1..=1024).contains(&max), "{max}");
        grid.threadgroup_size_matches_tile_size = true;
        grid.max_total_threads_per_threadgroup = Some(64);
        let limited = state(device, &grid).unwrap();
        assert!(limited.threadgroup_size_matches_tile_size());
        assert_eq!(limited.max_total_threads_per_threadgroup(), max.min(64));
        if let Some(aliased) = aliased_state(device) {
            assert_eq!(aliased.imageblock_sample_length(), Ok(24));
            assert_eq!(
                aliased.imageblock_memory_length_for_dimensions(Size::new(16, 16, 1)),
                Ok(24 * 256)
            );
        }
    });
}

/// A tile state whose descriptor requires a threadgroup size answers that size's threads as its
/// `maxTotalThreadsPerThreadgroup`, as Metal's tile state does where Metal compiles it for the
/// first time (an M3 Max, macOS 27.0.1: 64 for 8x8 and 256 for 16x16). It answers 1 for 1x1,
/// where Metal rounds the threads down to a multiple of 4 and answers 0 (docs/API.md, accepted
/// differences). Where the device holds fewer threads than the size, the answer is the device's
/// limit. The same descriptor with no size keeps today's answer.
#[test]
fn a_tile_state_with_a_required_size_answers_its_threads() {
    on_device(|device| {
        let mut grid = descriptor(device, "k10ts_tile_grid", &GRID);
        let unrequired = state(device, &grid)
            .unwrap()
            .max_total_threads_per_threadgroup();
        assert!((1..=1024).contains(&unrequired), "{unrequired}");
        for (required, threads) in [([1, 1, 1], 1), ([8, 8, 1], 64), ([16, 16, 1], 256)] {
            grid.required_threads_per_threadgroup = Some(required);
            let made = state(device, &grid).unwrap();
            assert_eq!(
                made.max_total_threads_per_threadgroup(),
                threads.min(unrequired),
                "{required:?}"
            );
        }
        grid.required_threads_per_threadgroup = None;
        assert_eq!(
            state(device, &grid)
                .unwrap()
                .max_total_threads_per_threadgroup(),
            unrequired
        );
    });
}

#[test]
fn a_tile_state_is_refused_where_metal_refuses_it() {
    on_device(|device| {
        // The function stages attachment 0, which this descriptor gives no format: Metal's
        // creation answers "Invalid pixel format for render target 0".
        let missing = descriptor(device, "k10ts_tile_grid", &[]);
        assert!(
            matches!(
                state(device, &missing),
                Err(Error::Compile(
                    metal2vulkan::error::Error::InvalidDescriptor(
                        DescriptorError::ColorAttachment { index: 0 }
                    )
                ))
            ),
            "{:?}",
            state(device, &missing).map(|_| ())
        );
    });
}

#[test]
fn a_binary_archive_holds_the_tile_pipelines_added_to_it() {
    on_device(|device| {
        let grid = descriptor(device, "k10ts_tile_grid", &GRID);
        let archive = device
            .new_binary_archive(&BinaryArchiveDescriptor::new(None))
            .unwrap();
        let mut options = PipelineOptions::new(PipelineOption::FAIL_ON_BINARY_ARCHIVE_MISS);
        options.binary_archives = vec![archive.clone()];
        assert_eq!(
            device
                .new_tile_render_pipeline_state(&grid, &options)
                .map(|_| ()),
            Err(Error::BinaryArchiveMiss(BinaryArchiveMiss::new(
                "k10ts_tile_grid",
                ArchivedPipeline::Tile,
            )))
        );
        // Metal's text for a tile miss (the consumer's report).
        assert_eq!(
            BinaryArchiveMiss::new("k10ts_tile_grid", ArchivedPipeline::Tile).to_string(),
            "Unable to find function \"k10ts_tile_grid\" (tile pipeline) in binary archives."
        );
        archive.add_tile_render_pipeline_functions(&grid).unwrap();
        let held = device
            .new_tile_render_pipeline_state(&grid, &options)
            .unwrap();
        assert_eq!(held.imageblock_sample_length(), Ok(12));
        // Another threadgroup limit is another archive key.
        let mut limited = grid.clone();
        limited.max_total_threads_per_threadgroup = Some(64);
        assert!(matches!(
            device.new_tile_render_pipeline_state(&limited, &options),
            Err(Error::BinaryArchiveMiss(_))
        ));
        // An add answers its creation's error.
        assert!(archive
            .add_tile_render_pipeline_functions(&descriptor(device, "k10ts_tile_grid", &[]))
            .is_err());
    });
}

#[test]
fn a_render_encoder_takes_a_tile_state_of_its_pass() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let tile = state(device, &descriptor(device, "k10ts_tile_grid", &GRID)).unwrap();
        // A pass of `formats`, 16 pixels a side, its encoder given to `test`.
        let in_pass =
            |formats: &[PixelFormat],
             test: &dyn Fn(&mut metal2vulkan_runtime::RenderCommandEncoder)| {
                let targets: Vec<_> = formats.iter().map(|&pf| texture(device, pf, 16)).collect();
                let mut pass = RenderPassDescriptor::new();
                for (attachment, target) in pass.color_attachments.iter_mut().zip(&targets) {
                    attachment.texture = Some(target.clone());
                }
                let command_buffer = queue.command_buffer().unwrap();
                let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
                test(&mut encoder);
                encoder.end_encoding().unwrap();
            };
        let refused = |problem| {
            Err(Error::Invalid(Invalid::new(
                "set_render_pipeline_state",
                Problem::Draw(problem),
            )))
        };
        in_pass(&GRID, &|encoder| {
            encoder.set_render_pipeline_state(&tile).unwrap();
            assert_eq!(
                encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0),
                Err(Error::Invalid(Invalid::new(
                    "draw_primitives",
                    Problem::Draw(DrawProblem::TileState),
                )))
            );
        });
        in_pass(&GRID[..2], &|encoder| {
            assert_eq!(
                encoder.set_render_pipeline_state(&tile),
                refused(DrawProblem::AttachmentFormat(Attachment::Color(2)))
            );
        });
        in_pass(
            &[
                PixelFormat::R32Float,
                PixelFormat::R32Float,
                PixelFormat::R32Uint,
            ],
            &|encoder| {
                assert_eq!(
                    encoder.set_render_pipeline_state(&tile),
                    refused(DrawProblem::AttachmentFormat(Attachment::Color(1)))
                );
            },
        );
        // The aliased layout needs 24 bytes a sample where the pass's texels give 18: Metal's
        // "per sample imageBlock usage" refusal, until the pass gives more.
        if let Some(aliased) = aliased_state(device) {
            in_pass(&ALIASED, &|encoder| {
                assert_eq!(
                    encoder.set_render_pipeline_state(&aliased),
                    refused(DrawProblem::ImageblockSampleLength {
                        pipeline: 24,
                        pass: 18,
                    })
                );
            });
        }
    });
}
