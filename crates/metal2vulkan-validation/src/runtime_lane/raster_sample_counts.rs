//! Pipeline states of a `rasterSampleCount` an Apple GPU does not take, on the device the cfg
//! selects, under the validation layer. Plain Metal on an Apple M4 under macOS 27.0.1 (with and
//! without API validation) answers nil with an `NSError` for a render or mesh state of any count
//! but 1, 2 and 4, and for a tile state of 0: code 2, "rasterSampleCount (8) is not supported by
//! device.". The runtime answers so on every device, before it compiles anything, whatever samples
//! the Vulkan driver rasterizes (lavapipe, RADV, ANV and NVIDIA rasterize 8; MoltenVK does not).
//! The functions are the public fixture `icb_direct_arguments`' plain ones.

use super::host_memory::on_device;
use crate::air_library::AirLibrary;
use metal2vulkan::metal::PixelFormat;
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, MeshPipelineDescriptor, RenderPipelineDescriptor, StageSource,
    TileRenderPipelineDescriptor,
};
use metal2vulkan_devtools::llvm_assemble;
use metal2vulkan_runtime::{Error, MetalError, PipelineOptions, Problem};

/// Metal's error for `call` of a pipeline state of `count` raster samples, which an Apple GPU
/// does not take. (shared with `runtime_lane::coverage_loop`.)
pub(super) fn refusal(call: &'static str, count: u32) -> Error {
    Error::MetalError(MetalError::with_description(
        call,
        Problem::RasterSampleCount { count },
        "AGXMetalG15X_M1",
        2,
        format!("rasterSampleCount ({count}) is not supported by device."),
    ))
}

/// A render state, with rasterization or without, a mesh state, and a tile state of each count
/// an Apple GPU does not take are Metal's error; a render and a mesh state of 1, 2 and 4 are
/// made or the device's gap (`Unsupported`), and a tile state of more than one sample, which Metal
/// makes, is never Metal's error (the compile layer refuses it, `Unsupported(TileShading)`).
#[test]
fn a_state_of_a_raster_sample_count_an_apple_gpu_lacks_is_metal_s_error_on_every_device() {
    on_device(|device| {
        let ll = metal2vulkan_devtools::fixtures_dir().join("examples/icb_direct_arguments.ll");
        let library = device
            .new_library_with_air(&llvm_assemble(&ll).unwrap())
            .unwrap();
        let function = |name: &str| {
            StageSource::Function(
                library
                    .new_function_with_name(name)
                    .unwrap()
                    .function_descriptor(),
            )
        };
        let color = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
        let render = |count: u32, rasterized: bool| {
            let mut descriptor = RenderPipelineDescriptor::new(function("plain_vertex"));
            descriptor.fragment = Some(function("plain_fragment"));
            descriptor.color_attachments[0] = color;
            descriptor.rasterization_enabled = rasterized;
            descriptor.raster_sample_count = count;
            device
                .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
                .map(drop)
        };
        let mesh = |count: u32| {
            let mut descriptor = MeshPipelineDescriptor::new(function("plain_mesh"));
            descriptor.fragment = Some(function("plain_fragment"));
            descriptor.color_attachments[0] = color;
            descriptor.raster_sample_count = count;
            device
                .new_mesh_render_pipeline_state(&descriptor, &PipelineOptions::default())
                .map(drop)
        };
        // The count is checked before the tile function is compiled.
        let tile = |count: u32| {
            let mut descriptor = TileRenderPipelineDescriptor::new(function("buffer_kernel"));
            descriptor.color_attachment_formats[0] = Some(PixelFormat::RGBA8Unorm);
            descriptor.raster_sample_count = count;
            device
                .new_tile_render_pipeline_state(&descriptor, &PipelineOptions::default())
                .map(drop)
        };
        let refused = |call: &'static str, count: u32| Err(refusal(call, count));
        const RENDER: &str = "new_render_pipeline_state";
        const MESH: &str = "new_mesh_render_pipeline_state";
        const TILE: &str = "new_tile_render_pipeline_state";
        for count in [0, 3, 8, 16, 32] {
            assert_eq!(render(count, true), refused(RENDER, count), "{count}");
            assert_eq!(render(count, false), refused(RENDER, count), "{count}");
            assert_eq!(mesh(count), refused(MESH, count), "{count}");
        }
        let mut answers = Vec::new();
        for count in [1, 2, 4] {
            for (kind, made) in [("render", render(count, true)), ("mesh", mesh(count))] {
                match made {
                    Ok(()) => answers.push(format!("{kind} {count}x made")),
                    Err(Error::Unsupported(u)) => {
                        answers.push(format!("{kind} {count}x refused, {:?}", u.needs));
                    }
                    Err(error) => panic!("{kind} {count}x: {error:?}"),
                }
            }
        }
        assert_eq!(tile(0), refused(TILE, 0));
        for count in [3, 8, 16, 64] {
            let made = tile(count);
            assert!(
                !matches!(made, Err(Error::MetalError(_))),
                "{count}: {made:?}"
            );
        }
        eprintln!(
            "raster sample counts on {}: Metal's error at 0, 3, 8, 16 and 32; {}",
            device.name(),
            answers.join("; ")
        );
    });
}
