//! What a device's recovery leaves of the objects that hold no Vulkan object: a depth/stencil
//! state, a library, a function and an address pool made before it serve after it, as Metal's
//! objects go on, and hold nothing of the lost generation.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs, on the kind of device
//! the cfg selects (docs/VALIDATION.md). No GPU fault: the loss is marked (`Device::recover_for_test`).

use super::device_recovery::{device, outcome, Scratch};
use super::device_recovery_carry::{fill, live_vk_devices, read, BYTES};
use super::encoder_holds::library;
use crate::{
    AddressPlacement, BufferContents, BufferDescriptor, CommandBufferStatus, CpuCacheMode,
    HazardTrackingMode, LoadAction, PipelineOptions, RenderPassDescriptor, ResourceOptions,
    StorageMode, StoreAction, TextureDescriptor,
};
use metal2vulkan::metal::state::DepthStencilDescriptor;
use metal2vulkan::metal::{CompareFunction, PixelFormat, TextureUsage};

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
