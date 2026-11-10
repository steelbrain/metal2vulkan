//! An argument buffer's texture read past its edge through a sampler (Metal 4.1's
//! `read(sampler, coord)`) reads Metal's border texel of its format and swizzle, which the
//! runtime writes into the texture's handle-table entry (`heap_sets.rs`, `handle_entry_flags`):
//! an `R32Float` texture reads `(1, 0, 0, 1)` through clamp-to-border opaque white and
//! `(0, 0, 0, 1)` through clamp-to-zero, and an `RGBA8Unorm` texture swizzled `(G, R, 1, 0)` reads
//! `(0, 0, 1, 0)` through clamp-to-zero, where an entry without them would read a four-channel
//! format's `(1, 1, 1, 1)` and `(0, 0, 0, 0)`. Each kernel names the texture by its resource ID in
//! the argument buffer, as Metal's `gpuResourceID` does.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The kernels are
//! `kernel_read_through.ll`'s `rt_heap_ow` and `rt_heap_zero`.

use super::blit_holds::{device, Scratch, WAIT};
use crate::{
    Buffer, CommandBufferStatus, ComputePipelineState, Device, PipelineOptions, Region,
    ResourceOptions, Size, Texture, TextureDescriptor,
};
use metal2vulkan::metal::{PixelFormat, TextureSwizzle, TextureSwizzleChannels, TextureUsage};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};

/// The textures' width and height, in texels.
const SIDE: u64 = 4;

/// The reads, `(x, y, layer, level)`: one inside, then one past each edge.
const POSITIONS: [[i32; 4]; 5] = [
    [1, 2, 0, 0],
    [-1, 0, 0, 0],
    [4, 1, 0, 0],
    [0, -1, 0, 0],
    [2, 4, 0, 0],
];

/// The compute state of `entry` of `kernel_read_through.ll` on `device`.
fn state(device: &Device, entry: &str) -> ComputePipelineState {
    let ll = metal2vulkan_devtools::engine_dir().join("kernel_read_through.ll");
    let bitcode = metal2vulkan_devtools::llvm_assemble(&ll).unwrap();
    let library = metal2vulkan::library::Library::from_air(&bitcode).unwrap();
    let function = device
        .new_library_with_compile_library(library)
        .new_function_with_name(entry)
        .unwrap();
    let descriptor =
        ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    device
        .new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// A shared buffer holding `bytes`.
fn holding(device: &Device, bytes: &[u8]) -> Buffer {
    let buffer = device
        .new_buffer(bytes.len() as u64, ResourceOptions::shared())
        .unwrap();
    let contents = buffer.contents().unwrap();
    // SAFETY: the new buffer's bytes, which no command uses yet.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr(), bytes.len()) };
    buffer
}

/// A `SIDE` by `SIDE` texture of `format` made with `swizzle`, every texel `texel`.
fn made(
    device: &Device,
    format: PixelFormat,
    swizzle: TextureSwizzleChannels,
    texel: &[u8],
) -> Texture {
    let mut descriptor = TextureDescriptor::texture_2d(format, SIDE, SIDE, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::shared();
    descriptor.swizzle = swizzle;
    let texture = device.new_texture(&descriptor).unwrap();
    let bytes = texel.repeat((SIDE * SIDE) as usize);
    let whole = Region::new_2d(0, 0, SIDE, SIDE);
    let row = texel.len() as u64 * SIDE;
    texture.replace_region(whole, 0, 0, &bytes, row, 0).unwrap();
    texture
}

/// What `entry` reads of `texture`, named by its resource ID in an argument buffer, at each of
/// [`POSITIONS`].
fn read(device: &Device, entry: &str, texture: &Texture) -> Vec<[f32; 4]> {
    let read = state(device, entry);
    let argument = holding(device, &texture.gpu_resource_id().0.to_le_bytes());
    let positions: Vec<u8> = POSITIONS
        .as_flattened()
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let positions = holding(device, &positions);
    let out = holding(device, &[0; 16 * POSITIONS.len()]);
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(&read).unwrap();
    encoder.set_buffer(Some(&positions), 0, 0).unwrap();
    encoder.set_buffer(Some(&out), 0, 1).unwrap();
    encoder.set_buffer(Some(&argument), 0, 2).unwrap();
    let threads = Size::new(POSITIONS.len() as u64, 1, 1);
    encoder.dispatch_threads(threads, threads).unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    assert_eq!(command_buffer.error(), None);
    let contents = out.contents().unwrap();
    // SAFETY: the buffer's bytes, which the completed command buffer wrote.
    let bytes = unsafe { std::slice::from_raw_parts(contents.as_ptr(), 16 * POSITIONS.len()) };
    let words: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&word| f32::from_le_bytes(word))
        .collect();
    words.as_chunks::<4>().0.to_vec()
}

#[test]
fn an_argument_buffer_texture_reads_its_formats_border_through_the_runtimes_table_entry() {
    use TextureSwizzle::{Green as G, One as O, Red as R, Zero as Z};
    let scratch = Scratch::new("heap-border-texels");
    let device = device(&scratch.0);
    let red = made(
        &device,
        PixelFormat::R32Float,
        TextureSwizzleChannels::DEFAULT,
        &0.5f32.to_le_bytes(),
    );
    let swizzled = made(
        &device,
        PixelFormat::RGBA8Unorm,
        TextureSwizzleChannels::new(G, R, O, Z),
        &[255, 0, 255, 0],
    );
    let cases = [
        (
            "rt_heap_ow",
            &red,
            [0.5, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
        ),
        (
            "rt_heap_zero",
            &red,
            [0.5, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ),
        (
            "rt_heap_zero",
            &swizzled,
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ),
    ];
    for (entry, texture, inside, border) in cases {
        let format = texture.pixel_format();
        let mut want = vec![border; POSITIONS.len()];
        want[0] = inside;
        assert_eq!(
            read(&device, entry, texture),
            want,
            "{entry} of a {format:?} texture: the texel inside, then Metal's border texel past \
             each edge"
        );
    }
}
