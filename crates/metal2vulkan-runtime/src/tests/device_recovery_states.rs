//! What a device's recovery leaves of the objects that hold no Vulkan object: a depth/stencil
//! state, a library, a function, an address pool, a binary archive, a dynamic library and a
//! rasterization rate map made before it serve after it, as Metal's objects go on, and hold
//! nothing of the lost generation; their devices are generation handles, which keep meaning their
//! generation.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs, on the kind of device
//! the cfg selects (docs/VALIDATION.md). No GPU fault: the loss is marked (`Device::recover_for_test`).

use super::device_recovery::{device, outcome, Scratch};
use super::device_recovery_carry::{fill, live_vk_devices, read, BYTES};
use super::encoder_holds::library;
use crate::{
    AddressPlacement, BinaryArchiveDescriptor, BufferContents, BufferDescriptor,
    CommandBufferStatus, CpuCacheMode, Device, DynamicLibrary, HazardTrackingMode, LoadAction,
    PipelineOptions, RenderPassDescriptor, ResourceOptions, StorageMode, StoreAction,
    TextureDescriptor,
};
use metal2vulkan::bind::{RateMapDescriptor, RateMapLayerDescriptor};
use metal2vulkan::metal::state::DepthStencilDescriptor;
use metal2vulkan::metal::{CompareFunction, PixelFormat, TextureUsage};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use metal2vulkan_devtools::llvm_assemble_text;
use metal2vulkan_devtools::metallib::{metallib_linked, wrap, MtlbFunction, MtlbLinkage};
use std::path::Path;

/// A depth/stencil state made before a recovery is set in a render encoder of the next
/// generation, whose command buffer completes, and the lost generation goes while the state
/// lives.
#[test]
fn a_depth_stencil_state_goes_on_past_a_recovery() {
    let scratch = Scratch::new();
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let mut descriptor = DepthStencilDescriptor::default();
    descriptor.depth_compare_function = CompareFunction::LessEqual;
    let state = device.new_depth_stencil_state(&descriptor).unwrap();
    device.recover_for_test().unwrap();
    let mut target = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 16, 16, false);
    target.usage = TextureUsage::RENDER_TARGET;
    target.resource_options = ResourceOptions::private();
    let color = device.new_texture(&target).unwrap();
    let mut depth = TextureDescriptor::texture_2d(PixelFormat::Depth32Float, 16, 16, false);
    depth.usage = TextureUsage::RENDER_TARGET;
    depth.resource_options = ResourceOptions::private();
    let depth = device.new_texture(&depth).unwrap();
    let mut pass = RenderPassDescriptor::new();
    let attachment = &mut pass.color_attachments[0];
    attachment.texture = Some(color);
    attachment.load_action = LoadAction::Clear;
    attachment.store_action = StoreAction::Store;
    pass.depth_attachment.texture = Some(depth);
    pass.depth_attachment.load_action = LoadAction::Clear;
    pass.depth_attachment.store_action = StoreAction::DontCare;
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
    encoder.set_depth_stencil_state(&state).unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    assert_eq!(
        outcome(&command_buffer),
        (CommandBufferStatus::Completed, None)
    );
    assert_eq!(state.descriptor(), &descriptor);
    assert_eq!(live_vk_devices(&device, 1), 1);
}

/// A rasterization rate map made before a recovery serves after it: a render pass over it of the
/// next generation encodes and its command buffer completes, its parameter data copies into a
/// buffer of the next generation and a reset compiles; its device is the next generation's, and
/// the lost generation goes while the map lives.
#[test]
fn a_rasterization_rate_map_goes_on_past_a_recovery() {
    let scratch = Scratch::new();
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let mut descriptor = RateMapDescriptor::new([64, 64]);
    descriptor.layers = vec![RateMapLayerDescriptor::new(vec![1.0, 0.5], vec![1.0, 0.5])];
    let map = device.new_rasterization_rate_map(&descriptor).unwrap();
    device.recover_for_test().unwrap();
    let size = map.physical_size_for_layer(0);
    let mut target =
        TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, size.width, size.height, false);
    target.usage = TextureUsage::RENDER_TARGET;
    target.resource_options = ResourceOptions::private();
    let color = device.new_texture(&target).unwrap();
    let mut pass = RenderPassDescriptor::new();
    let attachment = &mut pass.color_attachments[0];
    attachment.texture = Some(color);
    attachment.load_action = LoadAction::Clear;
    attachment.store_action = StoreAction::Store;
    pass.rasterization_rate_map = Some(map.clone());
    let command_buffer = queue.command_buffer().unwrap();
    let encoder = command_buffer.render_command_encoder(&pass).unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    assert_eq!(
        outcome(&command_buffer),
        (CommandBufferStatus::Completed, None)
    );
    let parameters = map.parameter_buffer_size_and_align().size;
    let buffer = device
        .new_buffer(parameters, ResourceOptions::shared())
        .unwrap();
    map.copy_parameter_data_to_buffer(&buffer, 0).unwrap();
    map.reset_using_descriptor(&descriptor).unwrap();
    assert!(map.device().generation_serves());
    assert_eq!(live_vk_devices(&device, 1), 1);
}

/// A library and a function made before a recovery serve after it: the function made before
/// and one made from the library after each make a compute pipeline state of the next
/// generation, set in a compute encoder whose command buffer completes, and the lost generation
/// goes while they live.
#[test]
fn a_library_and_its_functions_go_on_past_a_recovery() {
    let scratch = Scratch::new();
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let library = library(&device, "examples/runtime_examples.ll");
    let before = library.new_function_with_name("scale_and_offset").unwrap();
    device.recover_for_test().unwrap();
    let after = library.new_function_with_name("invert_texels").unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut compute = command_buffer.compute_command_encoder().unwrap();
    for function in [&before, &after] {
        let state = device
            .new_compute_pipeline_state_with_function(function, &PipelineOptions::default())
            .unwrap();
        compute.set_compute_pipeline_state(&state).unwrap();
    }
    compute.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    assert_eq!(
        outcome(&command_buffer),
        (CommandBufferStatus::Completed, None)
    );
    assert_eq!(live_vk_devices(&device, 1), 1);
}

/// An address pool reserved before a recovery takes buffers in its slots after it: a `Shared`, a
/// `Managed` and a `Private` buffer, each at the pool's address plus its slot's offset, filled and
/// read by the next generation's GPU work, beside the `Private` buffer the recovery carried in
/// its first slot; and the lost generation goes while the pool lives.
#[test]
fn an_address_pool_takes_buffers_past_a_recovery() {
    let scratch = Scratch::new();
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let pool = device
        .reserve_address_pool(AddressPlacement::Any, 4 * BYTES)
        .unwrap();
    let in_slot = |offset: u64, options: ResourceOptions| {
        let descriptor = BufferDescriptor::new(BYTES, options, BufferContents::Zero);
        device.new_buffer_with_descriptor(descriptor.with_pool_slot(&pool, offset))
    };
    let before = in_slot(0, ResourceOptions::private()).unwrap();
    fill(&queue, &before, 1);
    device.recover_for_test().unwrap();
    let managed = ResourceOptions::new(
        StorageMode::Managed,
        CpuCacheMode::DefaultCache,
        HazardTrackingMode::Default,
    );
    let kinds = [
        ResourceOptions::shared(),
        managed,
        ResourceOptions::private(),
    ];
    for (slot, options) in (1..).zip(kinds) {
        let offset = slot * BYTES;
        let buffer = in_slot(offset, options).unwrap_or_else(|error| {
            panic!("a {options:?} buffer in the pool's slot at {offset}: {error:?}")
        });
        assert!(buffer.device().generation_serves());
        assert_eq!(buffer.gpu_address().0, pool.gpu_address().0 + offset);
        let byte = 2 + slot as u8;
        fill(&queue, &buffer, byte);
        assert_eq!(
            read(&device, &queue, &buffer).unwrap(),
            vec![byte; BYTES as usize]
        );
    }
    assert!(before.device().generation_serves());
    assert_eq!(before.gpu_address(), pool.gpu_address());
    assert_eq!(
        read(&device, &queue, &before).unwrap(),
        vec![0; BYTES as usize]
    );
    assert_eq!(live_vk_devices(&device, 1), 1);
}

/// The `VkDevice` a handle's raw handles answer.
fn vk_device_of(device: &Device) -> ash::vk::Device {
    // SAFETY: only the handle is read, while the device lives.
    unsafe { device.raw() }.device.handle()
}

/// The device of each object that holds no Vulkan object is a generation handle, which keeps
/// meaning its generation past a recovery: a command pool made through a fence's device's raw
/// handles before the recovery is destroyed through that handle's raw handles after it, inside
/// its `with_loader_word`, as the lost `VkDevice` they answer still, which the handle keeps alive.
/// The objects' devices after the recovery are the next generation's, and the lost `VkDevice`
/// goes with the handles taken of it.
#[test]
fn an_objects_device_keeps_meaning_its_generation_past_a_recovery() {
    let scratch = Scratch::new();
    let device = device(&scratch.0);
    let fence = device.new_fence();
    let state = device
        .new_depth_stencil_state(&DepthStencilDescriptor::default())
        .unwrap();
    let library = library(&device, "examples/runtime_examples.ll");
    let function = library.new_function_with_name("scale_and_offset").unwrap();
    let pool = device
        .reserve_address_pool(AddressPlacement::Any, BYTES)
        .unwrap();
    let map = device
        .new_rasterization_rate_map(&RateMapDescriptor::new([64, 64]))
        .unwrap();
    let devices = || {
        [
            fence.device(),
            state.device(),
            library.device(),
            function.device(),
            pool.device(),
            map.device(),
        ]
    };
    let taken = devices();
    let lost = vk_device_of(&taken[0]);
    for made_on in &taken {
        assert!(made_on.generation_serves());
        assert_eq!(vk_device_of(made_on), lost);
    }
    let made_on = &taken[0];
    let info = ash::vk::CommandPoolCreateInfo::default();
    let command_pool = made_on.with_loader_word(|| {
        // SAFETY: the info is valid; the pool is the test's, destroyed below.
        unsafe { made_on.raw().device.create_command_pool(&info, None) }
    });
    let command_pool = command_pool.unwrap();
    device.recover_for_test().unwrap();
    // A root handle's raw handles answer the new `VkDevice` from now on.
    let serving = vk_device_of(&device);
    assert_ne!(serving, lost);
    for made_on in &taken {
        assert!(!made_on.generation_serves());
        assert_eq!(vk_device_of(made_on), lost);
    }
    made_on.with_loader_word(|| {
        // SAFETY: the pool was made on the `VkDevice` these handles answer, which the handle
        // keeps alive.
        unsafe {
            made_on
                .raw()
                .device
                .destroy_command_pool(command_pool, None)
        }
    });
    for now in devices() {
        assert!(now.generation_serves());
        assert_eq!(vk_device_of(&now), serving);
    }
    assert_eq!(live_vk_devices(&device, 2), 2);
    drop(taken);
    assert_eq!(live_vk_devices(&device, 1), 1);
}

/// The module of [`dynamic_library`]'s one function, with the data layout and target of the
/// AIR the tests' fixtures hold.
const DYLIB_MODULE: &str = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-\
i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-\
v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"
target triple = \"air64_v28-apple-macosx26.0.0\"

define float @_Z9dyl_scalef(float %x) {
  %r = fmul float %x, 3.000000e+00
  ret float %r
}

!air.version = !{!90}
!air.language_version = !{!91}
!90 = !{i32 2, i32 8, i32 0}
!91 = !{!\"Metal\", i32 4, i32 0, i32 0}
";

/// A dynamic library exporting one function, made on `device`, its install name a file in `dir`.
fn dynamic_library(device: &Device, dir: &Path) -> DynamicLibrary {
    let module = llvm_assemble_text(DYLIB_MODULE).unwrap();
    let name = dir.join("libdyl.metallib").to_string_lossy().into_owned();
    let function = MtlbFunction {
        name: "_Z9dyl_scalef",
        type_code: 3,
        bitcode: wrap(&module),
        constants: Vec::new(),
        public: Vec::new(),
        tags: Vec::new(),
    };
    let linkage = MtlbLinkage {
        dynamic: true,
        install_name: Some(&name),
        dependencies: Vec::new(),
        imports: Vec::new(),
    };
    let library = device
        .new_library_with_data(&metallib_linked(&[function], &linkage))
        .unwrap();
    device.new_dynamic_library(&library).unwrap()
}

/// A binary archive and a dynamic library made before a recovery serve after it: the archive
/// takes an add on the next generation, and a creation there that lists it makes its state; the
/// dynamic library keeps its label. Neither holds the lost generation, which goes while they
/// live.
#[test]
fn an_archive_and_a_dynamic_library_go_on_past_a_recovery() {
    let scratch = Scratch::new();
    let device = device(&scratch.0);
    let function = library(&device, "examples/runtime_examples.ll")
        .new_function_with_name("scale_and_offset")
        .unwrap();
    let descriptor =
        ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    let archive = device
        .new_binary_archive(&BinaryArchiveDescriptor::new(None))
        .unwrap();
    archive.add_compute_pipeline_functions(&descriptor).unwrap();
    let dynamic = dynamic_library(&device, &scratch.0);
    dynamic.set_label("dynamic");
    device.recover_for_test().unwrap();
    archive.add_compute_pipeline_functions(&descriptor).unwrap();
    let options = PipelineOptions {
        binary_archives: vec![archive.clone()],
        ..PipelineOptions::default()
    };
    let state = device
        .new_compute_pipeline_state(&descriptor, &options)
        .unwrap();
    assert!(state.device().generation_serves());
    assert_eq!(dynamic.label().as_deref(), Some("dynamic"));
    assert_eq!(live_vk_devices(&device, 1), 1);
}
