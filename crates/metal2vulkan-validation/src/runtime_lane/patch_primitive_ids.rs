//! Device tests of a fragment function's `[[primitive_id]]` behind a post-tessellation vertex
//! function, on the device the cfg selects, under the validation layer. The owned fixture
//! `k7_tess_triangle_probe` draws each triangle patch over the clip-space right triangle its
//! per-patch attribute states, and `stage_io_fragment_primitive` writes its primitive id as a
//! float into a 4x1 `RGBA32Float` target cleared to -1. Patch `k`'s attribute covers column `k`
//! alone, and every factor is 1.
//!
//! What Metal gives, as a Metal probe on an M3 Max with macOS 27.0.1 measured it: patch `i` of a
//! draw reads its primitive id as its position `patchStart + i`, the same in every instance (two
//! instances from base instance 3), with or without a patch index buffer; with one, the patch's
//! attribute is the patch the buffer's `uint` at that position names (`patch_indices.rs`). The
//! indirect draws are expected to answer as the direct draws of their arguments.
//!
//! A device's tessellation stages refuse the fragment function (`NotImplemented(Tessellation)`);
//! the emulation on a device without `tessellationShader` writes the position to a flat varying
//! (`patches::Path`).

use super::patches::on_path_with;
use super::patches::Path;
use super::{at_plan, buffer_with, WAIT};
use crate::air_library::AirLibrary;
use metal2vulkan::error::Feature;
use metal2vulkan::metal::{PixelFormat, TextureUsage, VertexFormat, VertexStepFunction};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource, TessellationDescriptor,
    VertexAttributeDescriptor, VertexBufferLayoutDescriptor, VertexDescriptor,
};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::DeviceOptions;
use metal2vulkan_runtime::{
    Buffer, ClearColor, CommandBufferStatus, Device, Error, LoadAction, PipelineOptions, Region,
    RenderPassDescriptor, RenderPipelineState, ResourceOptions, StoreAction, TextureDescriptor,
};

/// The target's columns, one per patch of the attribute buffer.
const COLUMNS: u64 = 4;

/// Patch `i` of four is patch `[1, 2, 3, 0][i]`.
const ROTATED: [u32; 4] = [1, 2, 3, 0];

/// A cleared column's value.
const CLEAR: f32 = -1.0;

/// The direct draw of 2 patches from patch 1, two instances from base instance 3, and the columns
/// its patches 1 and 2 cover.
const DIRECT: [f32; 4] = [CLEAR, 1.0, 2.0, CLEAR];

/// The draw of 3 patches from position 1 through [`ROTATED`], two instances from base instance 3:
/// positions 1, 2 and 3 draw patches 2, 3 and 0.
const THROUGH_INDICES: [f32; 4] = [3.0, CLEAR, 1.0, 2.0];

/// One draw: its patch index buffer, if any, and its arguments (`patchCount`, `instanceCount`,
/// `patchStart`, `baseInstance`), direct or through an indirect buffer.
struct Draw {
    name: &'static str,
    indices: bool,
    arguments: [u32; 4],
    indirect: bool,
}

#[test]
fn a_fragment_reads_a_patch_s_position_as_its_primitive_id() {
    primitive_ids(Path::Native);
}

/// [`a_fragment_reads_a_patch_s_position_as_its_primitive_id`], emulated.
#[test]
fn emulated_a_fragment_reads_a_patch_s_position_as_its_primitive_id() {
    primitive_ids_with(at_plan, Path::Emulated);
}

fn primitive_ids(path: Path) {
    primitive_ids_with(|_| {}, path)
}

/// [`primitive_ids`], `adjust` applied to the options of each device it makes (`at_plan` gives them
/// their queue plan).
fn primitive_ids_with(adjust: fn(&mut DeviceOptions), path: Path) {
    on_path_with(adjust, path, |device, name| {
        let pipeline = match pipeline(&device) {
            Ok(pipeline) => pipeline,
            Err(Error::Unsupported(u)) => {
                assert_eq!(path, Path::Native, "on {name} the emulation refused: {u:?}");
                assert_eq!(u.feature, Feature::Tessellation, "on {name}: {u:?}");
                eprintln!("patch primitive ids on {name}: the pipeline is refused, {u:?}");
                return;
            }
            Err(error) => panic!("the pipeline on {name}: {error}"),
        };
        let draws = [
            (
                Draw {
                    name: "direct",
                    indices: false,
                    arguments: [2, 2, 1, 3],
                    indirect: false,
                },
                DIRECT,
            ),
            (
                Draw {
                    name: "indirect",
                    indices: false,
                    arguments: [2, 2, 1, 3],
                    indirect: true,
                },
                DIRECT,
            ),
            (
                Draw {
                    name: "through_indices",
                    indices: true,
                    arguments: [3, 2, 1, 3],
                    indirect: false,
                },
                THROUGH_INDICES,
            ),
            (
                Draw {
                    name: "indirect_through_indices",
                    indices: true,
                    arguments: [3, 2, 1, 3],
                    indirect: true,
                },
                THROUGH_INDICES,
            ),
        ];
        for (draw, expected) in draws {
            assert_eq!(
                columns(&device, &pipeline, &draw),
                expected,
                "{} on {name}",
                draw.name
            );
        }
    });
}

/// The pipeline of the fixture's vertex function, its attribute read per patch from buffer 0,
/// and `stage_io_fragment_primitive` into an `RGBA32Float` attachment.
fn pipeline(device: &Device) -> Result<RenderPipelineState, Error> {
    let function = |file: &str, name: &str| {
        let bitcode = llvm_assemble(&engine_dir().join(file)).unwrap();
        StageSource::Function(
            device
                .new_library_with_air(&bitcode)
                .unwrap()
                .new_function_with_name(name)
                .unwrap()
                .function_descriptor(),
        )
    };
    let mut descriptor = RenderPipelineDescriptor::new(function(
        "k7_tess_triangle_probe.ll",
        "k7_tess_triangle_probe",
    ));
    descriptor.fragment = Some(function("stage_io.ll", "stage_io_fragment_primitive"));
    descriptor.color_attachments[0] =
        Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA32Float));
    let mut vertex = VertexDescriptor::new();
    let mut layout = VertexBufferLayoutDescriptor::new(Some(16));
    layout.step_function = VertexStepFunction::PerPatch;
    vertex.layouts.insert(0, layout);
    vertex.attributes.insert(
        0,
        VertexAttributeDescriptor::new(VertexFormat::Float4, 0, 0),
    );
    descriptor.vertex_descriptor = Some(vertex);
    descriptor.tessellation = Some(TessellationDescriptor::default());
    device.new_render_pipeline_state(&descriptor, &PipelineOptions::default())
}

/// A shared buffer of `words`' bytes.
fn buffer_of<T: Copy>(device: &Device, words: &[T], bytes: impl Fn(T) -> [u8; 4]) -> Buffer {
    let bytes: Vec<u8> = words.iter().flat_map(|&w| bytes(w)).collect();
    buffer_with(device, &bytes).unwrap()
}

/// The first channel of each column after `draw` of `pipeline` into a target cleared to
/// [`CLEAR`].
fn columns(device: &Device, pipeline: &RenderPipelineState, draw: &Draw) -> [f32; 4] {
    let mut texture = TextureDescriptor::texture_2d(PixelFormat::RGBA32Float, COLUMNS, 1, false);
    texture.usage = TextureUsage::RENDER_TARGET;
    texture.resource_options = ResourceOptions::shared();
    let target = device.new_texture(&texture).unwrap();
    // Patch `k`'s triangle: its corner at the left edge of column `k` and the bottom of the
    // target, 1.5 columns wide and 4 tall, so it covers column `k`'s centre and not the next.
    let width = 2.0 / COLUMNS as f32;
    let rects: Vec<f32> = (0..COLUMNS)
        .flat_map(|k| [-1.0 + width * k as f32, -1.0, 1.5 * width, 4.0])
        .collect();
    let rects = buffer_of(device, &rects, f32::to_le_bytes);
    // Every patch's three edge factors and inside factor 1 (half 0x3c00).
    let factors = buffer_of(
        device,
        &[0x3c00_3c00u32; 2 * COLUMNS as usize],
        u32::to_le_bytes,
    );
    let indices = buffer_of(device, &ROTATED, u32::to_le_bytes);
    let arguments = buffer_of(device, &draw.arguments, u32::to_le_bytes);
    let mut pass = RenderPassDescriptor::new();
    let color = &mut pass.color_attachments[0];
    color.texture = Some(target.clone());
    color.load_action = LoadAction::Clear;
    color.clear_color = ClearColor::new(f64::from(CLEAR), 0.0, 0.0, 0.0);
    color.store_action = StoreAction::Store;
    let command_buffer = device
        .new_command_queue()
        .unwrap()
        .command_buffer()
        .unwrap();
    let mut e = command_buffer.render_command_encoder(&pass).unwrap();
    e.set_render_pipeline_state(pipeline).unwrap();
    e.set_vertex_buffer(Some(&rects), 0, 0).unwrap();
    e.set_tessellation_factor_buffer(Some(&factors), 0, 0)
        .unwrap();
    let indices = draw.indices.then_some(&indices);
    let [count, instances, start, base] = draw.arguments;
    match draw.indirect {
        true => e.draw_patches_indirect(1, indices, 0, &arguments, 0),
        false => e.draw_patches(1, start, count, indices, 0, instances, base),
    }
    .unwrap();
    e.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    let mut texels = vec![0u8; 16 * COLUMNS as usize];
    target
        .get_bytes(
            &mut texels,
            16 * COLUMNS,
            16 * COLUMNS,
            Region::new_2d(0, 0, COLUMNS, 1),
            0,
            0,
        )
        .unwrap();
    std::array::from_fn(|k| f32::from_le_bytes(texels[16 * k..16 * k + 4].try_into().unwrap()))
}
