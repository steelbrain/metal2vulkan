//! Thread pointers held in a thread's array of structs at indices only the run knows, among the
//! structs' data, on the device the cfg selects, under the validation layer: the owned fixture
//! `pooled_cells`' kernels answer what the M3 Max answers (`pooled_cells.m`, macOS 27.0.1).
//!
//! - `pooled_closures`: a renderer's closure array, each element keeping a thread pointer to room
//!   at the array's end.
//! - `pooled_bytes`: a pointer into another array copied into the array as bytes, over a length
//!   the run reads, read back and written through.
//! - `pooled_spill`: the same pointer's bits stored into device memory as an integer and read back
//!   into the array. A thread pointer converted to an integer is not lowered, so the runtime
//!   refuses the pipeline, typed (`Unsupported`), where Metal answers as for `pooled_bytes`.
//! - `pooled_context`: the closures beside a context struct holding a constant pointer to an
//!   argument buffer (`Anc`) whose `held` points at a buffer of two textures' resource IDs, 4 by 1
//!   `R32Float`, 1 2 3 4 and 10 20 30 40; thread t reads texture t & 1 at x = t & 3.
//!
//! Eight threads in one threadgroup over `in[i] = i + 0.5` (16 words); the fixture's header says
//! what each word is. Every answer is a small integer or a half, exact in `float`.

use super::{
    at_plan, buffer_with, layer_reported_nothing_at_all, read_buffer, validated_device_with,
    Messages, ScratchCache, WAIT,
};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::metal::{PixelFormat, ResourceUsage, TextureUsage};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use metal2vulkan_devtools::llvm_assemble;
use metal2vulkan_runtime::{
    Buffer, CommandBufferStatus, ComputePipelineState, Device, Error, PipelineOptions, Region,
    ResourceOptions, ResourceRef, Size, Texture, TextureDescriptor,
};
use std::path::Path;

/// The threads, all in one threadgroup.
const THREADS: usize = 8;

/// The words each thread writes.
const WORDS: usize = 8;

/// What the M3 Max answers for `pooled_closures`.
#[rustfmt::skip]
const CLOSURES: [[f32; WORDS]; THREADS] = [
    [10.5, -1.0, -1.0, 1.0, 0.0, 1.0, 0.5, 1.0],
    [11.5, 2.5, -1.0, 2.0, 0.0, 3.0, 1.5, 1.0],
    [12.5, 23.5, -1.0, 2.0, 0.0, 3.0, 5.0, 2.0],
    [13.5, 24.5, 5.5, 3.0, 0.0, 6.0, 7.0, 2.0],
    [14.5, -1.0, -1.0, 1.0, 0.0, 1.0, 4.5, 1.0],
    [15.5, 6.5, -1.0, 2.0, 0.0, 3.0, 5.5, 1.0],
    [16.5, 27.5, -1.0, 2.0, 0.0, 3.0, 13.0, 2.0],
    [17.5, 28.5, 9.5, 3.0, 0.0, 6.0, 15.0, 2.0],
];

/// What the M3 Max answers for `pooled_bytes` and `pooled_spill` alike.
#[rustfmt::skip]
const COPIED: [[f32; WORDS]; THREADS] = [
    [0.5, 100.5, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    [2.5, 102.5, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0],
    [4.5, 104.5, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    [6.5, 106.5, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0],
    [4.5, 104.5, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    [6.5, 106.5, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0],
    [8.5, 108.5, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    [10.5, 110.5, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0],
];

/// What the M3 Max answers for `pooled_context`.
#[rustfmt::skip]
const CONTEXT: [[f32; WORDS]; THREADS] = [
    [1.0, -1.0, -1.0, 1.0, 0.0, 1.0, 0.5, 1.0],
    [20.0, 2.5, -1.0, 2.0, 0.0, 3.0, 1.5, 1.0],
    [3.0, 23.5, -1.0, 2.0, 0.0, 3.0, 5.0, 2.0],
    [40.0, 24.5, 5.5, 3.0, 0.0, 6.0, 7.0, 2.0],
    [1.0, -1.0, -1.0, 1.0, 0.0, 1.0, 4.5, 1.0],
    [20.0, 6.5, -1.0, 2.0, 0.0, 3.0, 5.5, 1.0],
    [3.0, 27.5, -1.0, 2.0, 0.0, 3.0, 13.0, 2.0],
    [40.0, 28.5, 9.5, 3.0, 0.0, 6.0, 15.0, 2.0],
];

/// A 4 by 1 `R32Float` texture holding `texels`.
fn texture(device: &Device, texels: [f32; 4]) -> Texture {
    let mut d = TextureDescriptor::texture_2d(PixelFormat::R32Float, 4, 1, false);
    d.usage = TextureUsage::SHADER_READ;
    d.resource_options = ResourceOptions::shared();
    let texture = device.new_texture(&d).unwrap();
    let bytes: Vec<u8> = texels.iter().flat_map(|t| t.to_le_bytes()).collect();
    texture
        .replace_region(Region::new_2d(0, 0, 4, 1), 0, 0, &bytes, 16, 0)
        .unwrap();
    texture
}

/// Kernel `entry`'s pipeline, or the runtime's refusal of it.
fn pipeline(device: &Device, entry: &str) -> Result<ComputePipelineState, Error> {
    let ll = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/public/engine/pooled_cells.ll");
    let library = device
        .new_library_with_air(&llvm_assemble(&ll).unwrap())
        .unwrap();
    let function = library.new_function_with_name(entry).unwrap();
    device.new_compute_pipeline_state(
        &ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor())),
        &PipelineOptions::default(),
    )
}

/// The words kernel `entry` writes over `in[i] = i + 0.5`, with `third` at buffer 2 when given and
/// `used` resident.
fn dispatched(
    device: &Device,
    entry: &str,
    third: Option<&Buffer>,
    used: &[ResourceRef<'_>],
) -> Vec<f32> {
    let pipeline = pipeline(device, entry).unwrap();
    let input: Vec<u8> = (0..16u16)
        .flat_map(|i| (f32::from(i) + 0.5).to_le_bytes())
        .collect();
    let out = buffer_with(device, &[0; 4 * WORDS * THREADS]).unwrap();
    let input = buffer_with(device, &input).unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(&pipeline).unwrap();
    encoder.set_buffer(Some(&out), 0, 0).unwrap();
    encoder.set_buffer(Some(&input), 0, 1).unwrap();
    if let Some(third) = third {
        encoder.set_buffer(Some(third), 0, 2).unwrap();
    }
    for r in used {
        encoder.use_resource(*r, ResourceUsage::READ).unwrap();
    }
    encoder
        .dispatch_threads(
            Size::new(THREADS as u64, 1, 1),
            Size::new(THREADS as u64, 1, 1),
        )
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    read_buffer(&out, 0, (4 * WORDS * THREADS) as u64)
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| f32::from_le_bytes(*w))
        .collect()
}

fn check(entry: &str, got: &[f32], metal: &[[f32; WORDS]; THREADS]) {
    for (t, (got, want)) in got.chunks(WORDS).zip(metal.iter()).enumerate() {
        assert_eq!(got, want, "{entry}: thread {t}");
    }
}

/// The fixture's kernels read, write and compare the pointers their closure array holds as Metal
/// does: pointers stored at run-time indices, copied in as bytes, and beside them an argument
/// buffer's textures reached through a context struct. The kernel that copies one in as an integer
/// of its bits is refused, typed.
#[test]
fn pointers_a_closure_array_holds_answer_as_metal() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    {
        // A command queue for each dispatch: the lane's queue limits would change the second
        // one's placement, so the device takes its plan.
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), at_plan).unwrap();
        check(
            "pooled_closures",
            &dispatched(&device, "pooled_closures", None, &[]),
            &CLOSURES,
        );
        let len = buffer_with(&device, &8u32.to_le_bytes()).unwrap();
        check(
            "pooled_bytes",
            &dispatched(&device, "pooled_bytes", Some(&len), &[]),
            &COPIED,
        );
        let Err(Error::Unsupported(refused)) = pipeline(&device, "pooled_spill") else {
            panic!("pooled_spill: made, though a thread pointer's integer is not lowered");
        };
        println!("POOLED-SPILL refused: {refused}");
        let t0 = texture(&device, [1.0, 2.0, 3.0, 4.0]);
        let t1 = texture(&device, [10.0, 20.0, 30.0, 40.0]);
        let ids: Vec<u8> = [&t0, &t1]
            .iter()
            .flat_map(|t| t.gpu_resource_id().0.to_le_bytes())
            .collect();
        let held = buffer_with(&device, &ids).unwrap();
        let anc = buffer_with(&device, &held.gpu_address().0.to_le_bytes()).unwrap();
        check(
            "pooled_context",
            &dispatched(
                &device,
                "pooled_context",
                Some(&anc),
                &[t0.as_resource(), t1.as_resource(), held.as_resource()],
            ),
            &CONTEXT,
        );
    }
    layer_reported_nothing_at_all(&messages);
}
