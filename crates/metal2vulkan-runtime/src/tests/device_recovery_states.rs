//! What a device's recovery leaves of the objects that hold no Vulkan object: a depth/stencil
//! state, a library and a function made before it serve after it, as Metal's objects go on, and
//! hold nothing of the lost generation.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs, on the kind of device
//! the cfg selects (docs/VALIDATION.md). No GPU fault: the loss is marked (`Device::recover_for_test`).

use super::device_recovery::{device, outcome, Scratch};
use super::device_recovery_carry::live_vk_devices;
use super::encoder_holds::library;
use crate::{
    CommandBufferStatus, LoadAction, PipelineOptions, RenderPassDescriptor, ResourceOptions,
    StoreAction, TextureDescriptor,
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
