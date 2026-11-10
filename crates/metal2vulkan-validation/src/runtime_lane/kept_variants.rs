//! Released variants kept for a state made again, on the device the cfg selects, under the
//! validation layer: a pipeline state released and made again takes back its variant's shader
//! modules and `VkPipeline`s, which the device's work counts show (`Device::work_counts`, not part
//! of the API); a state of another compiled pipeline shares none; the least recently released go
//! first past the bound; a kept variant holds no resource, so a no-copy buffer a released state's
//! commands used is freed while its variant is kept; and the device's teardown destroys what it
//! kept, the layer reporting nothing, leaks included (each test's `on_device`). The pipelines are
//! the owned fixture `validation/fixtures/public/examples/runtime_examples.ll` and, for the bound,
//! specializations of the owned fixture `engine/unspecialized_stage_functions.ll`'s kernel
//! `unspecialized_guarded`.

use super::degrade_notices::on_device;
use super::WAIT;
use crate::air_library::AirLibrary;
use metal2vulkan::library::{FunctionConstantValue, FunctionConstantValues};
use metal2vulkan::metal::{DataType, PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, ComputePipelineDescriptor, RenderPipelineDescriptor, StageSource,
};
use metal2vulkan_devtools::llvm_assemble;
use metal2vulkan_runtime::{
    Buffer, ClearColor, CommandBuffer, CommandBufferStatus, ComputePipelineState, Device,
    HostMemory, Library, LoadAction, PipelineOptions, RenderPassDescriptor, RenderPipelineState,
    ResourceOptions, Size, StoreAction, Texture, TextureDescriptor, WorkCounts,
};
use std::alloc::Layout;
use std::path::Path;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// The render target's side, and the color the fixture's triangles draw.
const SIDE: u64 = 8;
const DRAWN: [f32; 4] = [1.0, 0.2, 0.0, 1.0];

/// A no-copy buffer's bytes and alignment, and its texture's width, height and row pitch.
const NO_COPY: usize = 64 << 10;
const NO_COPY_ALIGN: usize = 64 << 10;
const LINEAR: (u64, u64, u64) = (16, 16, 256);

/// Twice the kernel states the device keeps the variants of (`KEPT`, 256).
const MANY: u32 = 512;

/// The owned fixture's library on `device`.
fn library(device: &Device) -> Library {
    fixture_library(device, "examples/runtime_examples.ll")
}

/// The library of the owned fixture `fixtures/public/<path>` on `device`.
fn fixture_library(device: &Device, path: &str) -> Library {
    let ll = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures/public")
        .join(path);
    let bitcode = llvm_assemble(&ll).unwrap();
    device.new_library_with_air(&bitcode).unwrap()
}

/// The compute state of `library`'s kernel `unspecialized_guarded` with its constant
/// `optional_count` at `n`: each `n` its own compiled pipeline.
fn guarded_state(device: &Device, library: &Library, n: u32) -> ComputePipelineState {
    let count = FunctionConstantValue::new(DataType::UInt, n.to_le_bytes());
    let function = library
        .new_function_with_name_constant_values(
            "unspecialized_guarded",
            &FunctionConstantValues::defaults().with_name("optional_count", count),
        )
        .unwrap();
    let descriptor =
        ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    device
        .new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// The fixture's render pipeline state: `colored_vertex` and `colored_fragment`, one target of
/// `format`.
fn render_state(device: &Device, format: PixelFormat) -> RenderPipelineState {
    let library = library(device);
    let vertex = library.new_function_with_name("colored_vertex").unwrap();
    let fragment = library.new_function_with_name("colored_fragment").unwrap();
    let mut descriptor =
        RenderPipelineDescriptor::new(StageSource::Function(vertex.function_descriptor()));
    descriptor.fragment = Some(StageSource::Function(fragment.function_descriptor()));
    descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(format));
    device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// The fixture's compute pipeline state of kernel `name`.
pub(super) fn kernel_state(device: &Device, name: &str) -> ComputePipelineState {
    let function = library(device).new_function_with_name(name).unwrap();
    let descriptor =
        ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    device
        .new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// Commit `command_buffer` and wait for it to complete.
fn finish(command_buffer: &CommandBuffer) {
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
}

/// A `Shared` buffer over memory of the test's, whose deallocator frees it and sets `freed`.
fn no_copy_buffer(device: &Device, freed: &Arc<AtomicBool>) -> Buffer {
    let layout = Layout::from_size_align(NO_COPY, NO_COPY_ALIGN).unwrap();
    // SAFETY: the layout's size is not zero.
    let memory = NonNull::new(unsafe { std::alloc::alloc_zeroed(layout) }).unwrap();
    let freed = Arc::clone(freed);
    // SAFETY: the memory is the test's, mapped and writable, aligned to 64 KiB, until the
    // deallocator frees it once the buffer's last reference goes; a refused call drops the
    // deallocator unrun, and the memory then leaks.
    unsafe {
        device.new_buffer_with_bytes_no_copy(
            HostMemory::new(memory),
            NO_COPY as u64,
            ResourceOptions::shared(),
            Some(Box::new(move |pointer, _| {
                // SAFETY: the pointer is the memory allocated above with `layout`, freed once.
                std::alloc::dealloc(pointer.as_ptr(), layout);
                freed.store(true, Ordering::SeqCst);
            })),
        )
    }
    .unwrap()
}

/// A 2D `RGBA8Unorm` texture over `buffer`'s bytes, which a kernel reads.
fn texture_over(buffer: &Buffer) -> Texture {
    let (width, height, row) = LINEAR;
    let mut descriptor =
        TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, width, height, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::shared();
    buffer.new_texture(&descriptor, 0, row).unwrap()
}

/// A private `format` texture of `side` texels a side with `usage`.
fn texture(device: &Device, format: PixelFormat, side: u64, usage: TextureUsage) -> Texture {
    let mut descriptor = TextureDescriptor::texture_2d(format, side, side, false);
    descriptor.usage = usage;
    descriptor.resource_options = ResourceOptions::private();
    device.new_texture(&descriptor).unwrap()
}

/// Draw the fixture's two triangles with `state` into a new target, their positions and colors
/// read from buffers over no-copy memory whose deallocators set `freed`, and wait; the buffers.
fn draw(device: &Device, state: &RenderPipelineState, freed: &[Arc<AtomicBool>; 2]) -> [Buffer; 2] {
    let target = texture(
        device,
        PixelFormat::RGBA8Unorm,
        SIDE,
        TextureUsage::RENDER_TARGET,
    );
    let [freed_positions, freed_colors] = freed;
    let positions = no_copy_buffer(device, freed_positions);
    let colors = no_copy_buffer(device, freed_colors);
    let corners: [[f32; 2]; 6] = [
        [-1.0, 1.0],
        [1.0, 1.0],
        [-1.0, -1.0],
        [1.0, 1.0],
        [1.0, -1.0],
        [-1.0, -1.0],
    ];
    // SAFETY: the buffers' memory is the test's, mapped, `NO_COPY` bytes long, and no command
    // uses it yet.
    unsafe {
        let at = positions.contents().unwrap().cast::<[f32; 2]>().as_ptr();
        for (i, corner) in corners.iter().enumerate() {
            at.add(i).write(*corner);
        }
        let at = colors.contents().unwrap().cast::<[f32; 4]>().as_ptr();
        for i in 0..corners.len() {
            at.add(i).write(DRAWN);
        }
    }
    let mut pass = RenderPassDescriptor::new();
    let color = &mut pass.color_attachments[0];
    color.texture = Some(target);
    color.load_action = LoadAction::Clear;
    color.store_action = StoreAction::Store;
    color.clear_color = ClearColor::new(0.0, 0.0, 1.0, 1.0);
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
    encoder.set_render_pipeline_state(state).unwrap();
    encoder.set_vertex_buffer(Some(&positions), 0, 0).unwrap();
    encoder.set_vertex_buffer(Some(&colors), 0, 1).unwrap();
    encoder
        .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
        .unwrap();
    encoder.end_encoding().unwrap();
    finish(&command_buffer);
    [positions, colors]
}

/// Dispatch the fixture's `invert_texels` with `state` from a texture over no-copy memory whose
/// deallocator sets `freed` into a new texture, and wait; the no-copy buffer and its texture.
fn invert(
    device: &Device,
    state: &ComputePipelineState,
    freed: &Arc<AtomicBool>,
) -> (Buffer, Texture) {
    let buffer = no_copy_buffer(device, freed);
    let source = texture_over(&buffer);
    let (width, _, _) = LINEAR;
    let destination = texture(
        device,
        PixelFormat::RGBA8Unorm,
        width,
        TextureUsage::SHADER_WRITE,
    );
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(state).unwrap();
    encoder.set_texture(Some(&source), 0).unwrap();
    encoder.set_texture(Some(&destination), 1).unwrap();
    encoder
        .dispatch_threads(Size::new(width, width, 1), Size::new(8, 8, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    finish(&command_buffer);
    (buffer, source)
}

/// Wait until `freed` is set: a deallocator runs once its buffer's last handle goes.
fn freed_soon(freed: &AtomicBool) -> bool {
    let deadline = Instant::now() + WAIT;
    while !freed.load(Ordering::SeqCst) && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(5));
    }
    freed.load(Ordering::SeqCst)
}

/// What `after` made past `before`: shader modules, `VkPipeline`s, and variants taken back.
fn made(before: &WorkCounts, after: &WorkCounts) -> (u64, u64, u64) {
    (
        after.shader_modules_made - before.shader_modules_made,
        after.variant_pipelines_made - before.variant_pipelines_made,
        after.variants_taken_back - before.variants_taken_back,
    )
}

/// A render state released and made again takes back its variant: no shader module and no
/// `VkPipeline` made the second time. The no-copy buffers the first draw read are freed with
/// their last handles while that variant is kept, as Metal frees them: a kept variant holds no
/// resource. The device's teardown then destroys what it kept.
#[test]
fn a_render_state_made_again_takes_back_its_released_variant() {
    on_device(|device, _| {
        let freed = [
            Arc::new(AtomicBool::new(false)),
            Arc::new(AtomicBool::new(false)),
        ];
        let first = device.work_counts();
        let state = render_state(device, PixelFormat::RGBA8Unorm);
        let buffers = draw(device, &state, &freed);
        let (modules, pipelines, _) = made(&first, &device.work_counts());
        assert!(
            modules > 0 && pipelines > 0,
            "the first state made its objects"
        );
        // The state goes first, its variant kept; then the buffers' last handles.
        drop(state);
        drop(buffers);
        assert!(
            freed.iter().all(|f| freed_soon(f)),
            "the draw's no-copy buffers were freed with the state's variant kept"
        );
        let before = device.work_counts();
        let freed = [
            Arc::new(AtomicBool::new(false)),
            Arc::new(AtomicBool::new(false)),
        ];
        drop(draw(
            device,
            &render_state(device, PixelFormat::RGBA8Unorm),
            &freed,
        ));
        let (modules, pipelines, taken) = made(&before, &device.work_counts());
        assert_eq!((modules, pipelines), (0, 0), "made again, it made nothing");
        assert!(taken > 0, "made again, it took its variant back");
    });
}

/// A compute state released and made again takes back its variant; the texture over no-copy
/// memory its dispatch read is freed with its last handles while the variant is kept.
#[test]
fn a_compute_state_made_again_takes_back_its_released_variant() {
    on_device(|device, _| {
        let freed = Arc::new(AtomicBool::new(false));
        let state = kernel_state(device, "invert_texels");
        let held = invert(device, &state, &freed);
        // The state goes first, its variant kept; then the texture's and buffer's last handles.
        drop(state);
        drop(held);
        assert!(
            freed_soon(&freed),
            "the dispatch's no-copy buffer was freed with the state's variant kept"
        );
        let before = device.work_counts();
        let freed = Arc::new(AtomicBool::new(false));
        drop(invert(
            device,
            &kernel_state(device, "invert_texels"),
            &freed,
        ));
        let (modules, pipelines, taken) = made(&before, &device.work_counts());
        assert_eq!((modules, pipelines), (0, 0), "made again, it made nothing");
        assert!(taken > 0, "made again, it took its variant back");
    });
}

/// A state of another compiled pipeline takes back nothing: a render state of another target
/// format, and another kernel, each make their own objects.
#[test]
fn a_state_of_another_pipeline_takes_back_nothing() {
    on_device(|device, _| {
        let freed = [
            Arc::new(AtomicBool::new(false)),
            Arc::new(AtomicBool::new(false)),
        ];
        drop(draw(
            device,
            &render_state(device, PixelFormat::RGBA8Unorm),
            &freed,
        ));
        drop(kernel_state(device, "invert_texels"));
        let before = device.work_counts();
        drop(render_state(device, PixelFormat::BGRA8Unorm));
        drop(kernel_state(device, "scale_and_offset"));
        let after = device.work_counts();
        let (modules, _, taken) = made(&before, &after);
        // Each took modules of its stages: made, or the device's of the same words (the render
        // state's vertex function's, which the kept variant still holds).
        let shared = after.shader_modules_shared - before.shader_modules_shared;
        assert!(
            modules + shared >= 3,
            "each took its stages' modules ({modules} made, {shared} shared)"
        );
        assert!(modules >= 1, "the other kernel made its module");
        assert_eq!(taken, 0, "neither took a variant back");
    });
}

/// Past the bound the least recently released go first: of [`MANY`] kernel states, each its own
/// compiled pipeline, made and released in turn, the first made again makes its module and the
/// last made again takes its variant back.
#[test]
fn the_least_recently_released_variants_go_first() {
    on_device(|device, _| {
        let library = fixture_library(device, "engine/unspecialized_stage_functions.ll");
        for n in 0..MANY {
            drop(guarded_state(device, &library, n));
        }
        let before = device.work_counts();
        drop(guarded_state(device, &library, 0));
        let (modules, _, taken) = made(&before, &device.work_counts());
        assert!(modules > 0 && taken == 0, "the first went past the bound");
        let before = device.work_counts();
        drop(guarded_state(device, &library, MANY - 1));
        let (modules, _, taken) = made(&before, &device.work_counts());
        assert!(modules == 0 && taken > 0, "the last was kept");
    });
}

/// Draw the fixture's two triangles with `state` into a new shared `format` target of [`SIDE`]
/// texels a side, cleared to blue, and wait: the target's bytes.
fn drawn(device: &Device, state: &RenderPipelineState, format: PixelFormat) -> Vec<u8> {
    let mut descriptor = TextureDescriptor::texture_2d(format, SIDE, SIDE, false);
    descriptor.usage = TextureUsage::RENDER_TARGET;
    descriptor.resource_options = ResourceOptions::shared();
    let target = device.new_texture(&descriptor).unwrap();
    let corners: [[f32; 2]; 6] = [
        [-1.0, 1.0],
        [1.0, 1.0],
        [-1.0, -1.0],
        [1.0, 1.0],
        [1.0, -1.0],
        [-1.0, -1.0],
    ];
    let positions: Vec<u8> = corners
        .iter()
        .flatten()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let colors: Vec<u8> = (0..corners.len())
        .flat_map(|_| DRAWN)
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let positions = device
        .new_buffer_with_bytes(&positions, ResourceOptions::shared())
        .unwrap();
    let colors = device
        .new_buffer_with_bytes(&colors, ResourceOptions::shared())
        .unwrap();
    let mut pass = RenderPassDescriptor::new();
    let color = &mut pass.color_attachments[0];
    color.texture = Some(target.clone());
    color.load_action = LoadAction::Clear;
    color.store_action = StoreAction::Store;
    color.clear_color = ClearColor::new(0.0, 0.0, 1.0, 1.0);
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
    encoder.set_render_pipeline_state(state).unwrap();
    encoder.set_vertex_buffer(Some(&positions), 0, 0).unwrap();
    encoder.set_vertex_buffer(Some(&colors), 0, 1).unwrap();
    encoder
        .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
        .unwrap();
    encoder.end_encoding().unwrap();
    finish(&command_buffer);
    let mut bytes = vec![0u8; (4 * SIDE * SIDE) as usize];
    target
        .get_bytes(
            &mut bytes,
            4 * SIDE,
            0,
            metal2vulkan_runtime::Region::new_2d(0, 0, SIDE, SIDE),
            0,
            0,
        )
        .unwrap();
    bytes
}

/// Live render states of the same functions to targets of other formats share the device's
/// module of each stage whose SPIR-V is the same: a state made beside another makes only the
/// modules of words no live variant has, and each draws what it draws alone, on a device of its
/// own.
#[test]
fn states_of_the_same_functions_share_their_modules() {
    let alone = |format: PixelFormat| {
        let mut got = Vec::new();
        on_device(|device, _| {
            let state = render_state(device, format);
            got = drawn(device, &state, format);
        });
        got
    };
    let want = [
        alone(PixelFormat::RGBA8Unorm),
        alone(PixelFormat::BGRA8Unorm),
    ];
    on_device(|device, _| {
        let before = device.work_counts();
        let rgba = render_state(device, PixelFormat::RGBA8Unorm);
        let first = drawn(device, &rgba, PixelFormat::RGBA8Unorm);
        let between = device.work_counts();
        let bgra = render_state(device, PixelFormat::BGRA8Unorm);
        let second = drawn(device, &bgra, PixelFormat::BGRA8Unorm);
        let after = device.work_counts();
        assert_eq!(first, want[0], "RGBA8Unorm, beside the other");
        assert_eq!(second, want[1], "BGRA8Unorm, beside the other");
        let made_first = between.shader_modules_made - before.shader_modules_made;
        let made_second = after.shader_modules_made - between.shader_modules_made;
        let shared_second = after.shader_modules_shared - between.shader_modules_shared;
        assert!(made_first >= 2, "the first state made both stages' modules");
        assert!(
            shared_second >= 1,
            "the second took the vertex function's module ({made_second} made, {shared_second} \
             shared)"
        );
        assert!(
            made_second < made_first,
            "the second made fewer modules than the first ({made_second} of {made_first})"
        );
        drop((rgba, bgra));
    });
}
