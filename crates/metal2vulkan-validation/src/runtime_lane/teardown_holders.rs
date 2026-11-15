//! The device's teardown against every holder of its objects, on the device the cfg selects, under
//! the validation layer: the program's last device handle goes before, between or after the objects
//! that hold the device, in every order, and the process neither crashes nor hears from the layer
//! (a destruction after the device's, or a leak at it).
//!
//! The holders: live variants that share the device's shader modules (two render states of the
//! same vertex function, one of them with an immutable sampler), a kept instance (a released
//! state's objects, which share those modules too), a variant taken back from the kept ones, and
//! an uncommitted command buffer whose draw holds the only handle of its state's variant, and
//! live and kept variants sharing the device's immutable sampler set, a host step's share of a
//! state's set, an uncommitted command buffer holding a shared set, and a draw's pipeline handed
//! to the device's pool, maybe still being made. On lavapipe the earlier rule (each module
//! destroyed itself through its own copy of the device's functions) crashed whenever a variant held
//! the last device handle: the teardown destroyed the device and the instance, the loader unloaded
//! the driver, and then the variant's modules called into it (SIGSEGV 10 of 10, `pass_start_reads`
//! and `extended_range_blends` included).
//!
//! The pipelines are the owned fixture `validation/fixtures/public/examples/runtime_examples.ll`.

use super::indirect_host_steps;
use super::{layer_reported_nothing, validated_device_with, Messages, ScratchCache, WAIT};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::metal::{
    IndirectCommandType, PixelFormat, PrimitiveType, RenderStages, ResourceUsage, TextureUsage,
};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, ComputePipelineDescriptor, RenderPipelineDescriptor, StageSource,
};
use metal2vulkan_devtools::llvm_assemble;
use metal2vulkan_runtime::{
    AddressModeMap, ClearColor, CommandBuffer, CommandBufferStatus, ComputePipelineState, Device,
    EventRef, IndirectCommandBufferDescriptor, Library, LoadAction, PipelineOptions, Region,
    RenderPassDescriptor, RenderPipelineState, ResourceOptions, StoreAction, Texture,
    TextureDescriptor,
};
use std::any::Any;

/// The owned fixture's library on `device`.
fn library(device: &Device) -> Library {
    let ll = metal2vulkan_devtools::fixtures_dir().join("examples/runtime_examples.ll");
    let bitcode = llvm_assemble(&ll).unwrap();
    device.new_library_with_air(&bitcode).unwrap()
}

/// The fixture's render state of `colored_vertex` and `fragment`, one target of `format`.
fn render_state(device: &Device, fragment: &str, format: PixelFormat) -> RenderPipelineState {
    let library = library(device);
    let vertex = library.new_function_with_name("colored_vertex").unwrap();
    let fragment = library.new_function_with_name(fragment).unwrap();
    let mut descriptor =
        RenderPipelineDescriptor::new(StageSource::Function(vertex.function_descriptor()));
    descriptor.fragment = Some(StageSource::Function(fragment.function_descriptor()));
    descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(format));
    device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// The fixture's compute state of kernel `name`.
fn kernel_state(device: &Device, name: &str) -> ComputePipelineState {
    let function = library(device).new_function_with_name(name).unwrap();
    let descriptor =
        ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    device
        .new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// An uncommitted command buffer that encoded one draw of the fixture's two triangles with
/// `state` into an `RGBA8Unorm` target.
fn encoded_draw(device: &Device, state: &RenderPipelineState) -> CommandBuffer {
    encoded_draw_as(device, state, PrimitiveType::Triangle, None)
}

/// A shader-readable 8x8 `RGBA8Unorm` texture whose address-mode map is `map`.
fn mapped_texture(device: &Device, map: AddressModeMap) -> Texture {
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 8, 8, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    descriptor.address_mode_map = map;
    device.new_texture(&descriptor).unwrap()
}

/// [`encoded_draw`]'s command buffer, drawing as `primitive` and binding `texture` at fragment
/// texture 0 where one is given.
fn encoded_draw_as(
    device: &Device,
    state: &RenderPipelineState,
    primitive: PrimitiveType,
    texture: Option<&Texture>,
) -> CommandBuffer {
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 8, 8, false);
    descriptor.usage = TextureUsage::RENDER_TARGET;
    descriptor.resource_options = ResourceOptions::private();
    let target = device.new_texture(&descriptor).unwrap();
    let positions: Vec<u8> = [[-1.0f32, 1.0], [1.0, 1.0], [-1.0, -1.0]]
        .iter()
        .flatten()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let colors: Vec<u8> = (0..3)
        .flat_map(|_| [1.0f32, 0.2, 0.0, 1.0])
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
    if let Some(texture) = texture {
        encoder.set_fragment_texture(Some(texture), 0).unwrap();
    }
    encoder.draw_primitives(primitive, 0, 3, 1, 0).unwrap();
    encoder.end_encoding().unwrap();
    command_buffer
}

/// Two holders, as the drops take them.
fn held(first: impl Any, second: impl Any) -> Vec<Box<dyn Any>> {
    let first: Box<dyn Any> = Box::new(first);
    let second: Box<dyn Any> = Box::new(second);
    vec![first, second]
}

/// Every order of `0..n`.
fn orders(n: usize) -> Vec<Vec<usize>> {
    if n == 0 {
        return vec![Vec::new()];
    }
    let mut all = Vec::new();
    for rest in orders(n - 1) {
        for at in 0..=rest.len() {
            let mut order = rest.clone();
            order.insert(at, n - 1);
            all.push(order);
        }
    }
    all
}

/// For every order of the holders `make` makes on a fresh device and that device's last handle
/// (the holder after them), each on a device of its own: drop them in that order, wait for the
/// device's teardown to end, wherever it runs, and the validation layer reports nothing. Each
/// device's shader cache is the test's, so translation is paid once.
fn in_every_order(holders: usize, make: impl Fn(&Device) -> Vec<Box<dyn Any>>) {
    let cache = ScratchCache::new().unwrap();
    let all = orders(holders + 1);
    assert_eq!(all.len(), (1..=holders + 1).product::<usize>());
    for order in all {
        let messages = Messages::default();
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |_| {}).unwrap();
        let mut held = make(&device);
        assert_eq!(held.len(), holders, "the holders made");
        held.push(Box::new(device));
        let mut ranked: Vec<(usize, Box<dyn Any>)> = held
            .into_iter()
            .enumerate()
            .map(|(index, holder)| {
                let rank = order.iter().position(|&o| o == index).unwrap();
                (rank, holder)
            })
            .collect();
        ranked.sort_by_key(|(rank, _)| *rank);
        for (_, holder) in ranked {
            drop(holder);
        }
        // The teardown may run after the last drop, on the device's reaper once a making still
        // running ends: its messages are read, and nothing is left running as the test ends.
        assert!(
            messages.unshared_within(WAIT),
            "dropped in order {order:?}: the device's teardown did not end"
        );
        layer_reported_nothing(
            &messages,
            &format!("dropped in order {order:?} (the device is {holders})"),
        );
    }
}

/// Live variants sharing the device's modules, one with an immutable sampler, and a kept
/// instance's objects sharing them too, against the device's last handle, in every order.
#[test]
fn variants_sharing_modules_and_a_kept_instance_outlive_the_device_in_every_order() {
    in_every_order(2, |device| {
        // A kept instance at the teardown: a state of the same functions to another format,
        // released, its objects kept with their shares of the device's modules.
        drop(render_state(
            device,
            "colored_fragment",
            PixelFormat::BGRA8Unorm,
        ));
        let before = device.work_counts();
        let colored = render_state(device, "colored_fragment", PixelFormat::RGBA8Unorm);
        let sampled = render_state(device, "sampled_fragment", PixelFormat::RGBA8Unorm);
        let after = device.work_counts();
        assert!(
            after.shader_modules_shared > before.shader_modules_shared,
            "the live states share the device's modules"
        );
        assert_eq!(
            after.variants_taken_back, before.variants_taken_back,
            "the kept instance stays kept"
        );
        held(colored, sampled)
    });
}

/// A variant taken back from the kept ones, and an uncommitted command buffer whose draw holds
/// the only handle of its state's variant, against the device's last handle, in every order.
#[test]
fn a_taken_back_variant_and_a_command_buffer_s_variant_outlive_the_device_in_every_order() {
    in_every_order(2, |device| {
        drop(kernel_state(device, "invert_texels"));
        let before = device.work_counts();
        let taken_back = kernel_state(device, "invert_texels");
        assert!(
            device.work_counts().variants_taken_back > before.variants_taken_back,
            "made again, the kernel's state took its variant back"
        );
        let state = render_state(device, "colored_fragment", PixelFormat::RGBA8Unorm);
        let command_buffer = encoded_draw(device, &state);
        drop(state);
        held(taken_back, command_buffer)
    });
}

/// Two live states whose layouts bake the same immutable sampler share the device's set of it,
/// and a kept instance's objects (a released state of the same functions) share it too: the
/// set is destroyed by the device's map alone, at a settle or the teardown, whichever of the
/// holders and the device's last handle goes last. The counts show the sharing: the released
/// `RGBA16Float` state made the one set, the two live states took it (none made, none
/// destroyed, so the kept instance held it meanwhile), and neither took that instance back.
#[test]
fn states_sharing_an_immutable_sampler_set_outlive_the_device_in_every_order() {
    in_every_order(2, |device| {
        let before = device.work_counts();
        drop(render_state(
            device,
            "sampled_fragment",
            PixelFormat::RGBA16Float,
        ));
        let released = device.work_counts();
        let one = render_state(device, "sampled_fragment", PixelFormat::RGBA8Unorm);
        let other = render_state(device, "sampled_fragment", PixelFormat::BGRA8Unorm);
        let after = device.work_counts();
        assert_eq!(
            released.sampler_sets_made - before.sampler_sets_made,
            1,
            "the RGBA16Float state made the device's set"
        );
        assert_eq!(
            (
                after.sampler_sets_made - released.sampler_sets_made,
                after.sampler_sets_shared - released.sampler_sets_shared,
            ),
            (0, 2),
            "the two live states share the device's one set"
        );
        assert_eq!(
            after.sampler_sets_destroyed, before.sampler_sets_destroyed,
            "the set stayed while the kept instance held it"
        );
        assert_eq!(
            after.variants_taken_back, before.variants_taken_back,
            "the RGBA16Float instance stays kept"
        );
        held(one, other)
    });
}

/// A host step's draw of a sampled state made after its command buffer's commit, as Metal
/// allows one: the owned fixture `examples/icb_argument_sampling.ll`, whose fragment function
/// samples the texture its argument buffer holds through a constexpr sampler. Metal refuses a state
/// that supports indirect command buffers whose functions use a direct texture or sampler argument,
/// and the runtime refuses a mapped texture a function sampling handles can
/// reach (`MappedSampling::Handles`), so no address-mode map reaches the step's variant: its
/// immutable sampler is the state's, and the step shares the set the state's creation made on the
/// program's thread, making none of its own (`WorkCounts::host_step_sampler_sets_made` stays). The
/// command buffer's records are the GPU's to write (a blit copied one), so its execution is a host
/// step's; it waits for an event, and once the commit returned the test makes the state, encodes
/// command 0 with it on the CPU, and signals. Each pixel of the 8x8 target samples its own texel's
/// centre of the 8x8 source, so the draw writes the source's texels. The state, the completed
/// command buffer, the target and the source's texels.
pub(super) fn host_step_s_shared_set(
    device: &Device,
) -> (RenderPipelineState, CommandBuffer, Texture, Vec<u8>) {
    let mut commands = IndirectCommandBufferDescriptor::new();
    commands.command_types = IndirectCommandType::DRAW;
    commands.max_vertex_buffer_bind_count = 2;
    commands.max_fragment_buffer_bind_count = 1;
    let icb = device
        .new_indirect_command_buffer(&commands, 2, ResourceOptions::shared())
        .unwrap();
    let queue = device.new_command_queue().unwrap();
    let copying = queue.command_buffer().unwrap();
    let mut blit = copying.blit_command_encoder().unwrap();
    blit.copy_indirect_command_buffer(&icb, 0..1, &icb, 1)
        .unwrap();
    blit.end_encoding().unwrap();
    indirect_host_steps::complete(&copying);
    let positions: Vec<u8> = [[-1.0f32, -1.0], [3.0, -1.0], [-1.0, 3.0]]
        .iter()
        .flatten()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let positions = device
        .new_buffer_with_bytes(&positions, ResourceOptions::shared())
        .unwrap();
    let colors = device
        .new_buffer_with_bytes(&[0u8; 3 * 16], ResourceOptions::shared())
        .unwrap();
    let texels: Vec<u8> = (0..8u8)
        .flat_map(|y| (0..8u8).flat_map(move |x| [x * 32, y * 32, (x + y) * 16, 255]))
        .collect();
    let source = mapped_texture(device, AddressModeMap::IDENTITY);
    let region = Region::new_2d(0, 0, 8, 8);
    source
        .replace_region(region, 0, 0, &texels, 8 * 4, 8 * 8 * 4)
        .unwrap();
    let arguments = device
        .new_buffer_with_bytes(
            &source.gpu_resource_id().0.to_le_bytes(),
            ResourceOptions::shared(),
        )
        .unwrap();
    let mut target = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 8, 8, false);
    target.usage = TextureUsage::RENDER_TARGET;
    target.resource_options = ResourceOptions::shared();
    let target = device.new_texture(&target).unwrap();
    let mut pass = RenderPassDescriptor::new();
    let color = &mut pass.color_attachments[0];
    color.texture = Some(target.clone());
    color.load_action = LoadAction::Clear;
    color.store_action = StoreAction::Store;
    let event = device.new_shared_event().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    command_buffer
        .encode_wait_for_event(EventRef::from(&event), 1)
        .unwrap();
    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
    for buffer in [&positions, &colors] {
        encoder
            .use_resource(
                buffer.as_resource(),
                ResourceUsage::READ,
                RenderStages::VERTEX,
            )
            .unwrap();
    }
    encoder
        .use_resource(
            arguments.as_resource(),
            ResourceUsage::READ,
            RenderStages::FRAGMENT,
        )
        .unwrap();
    encoder
        .use_resource(
            source.as_resource(),
            ResourceUsage::READ,
            RenderStages::FRAGMENT,
        )
        .unwrap();
    encoder.execute_commands_in_buffer(&icb, 0..1).unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    // After the commit: the state, and command 0 drawing with it.
    let descriptor = shared_set_descriptor(device);
    let before = device.work_counts();
    let state = device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap();
    let created = device.work_counts();
    assert_eq!(
        created.sampler_sets_made - before.sampler_sets_made,
        1,
        "the state's creation made the set of its constexpr sampler"
    );
    let command = icb.indirect_render_command_at(0).unwrap();
    command.set_render_pipeline_state(&state).unwrap();
    command.set_vertex_buffer(Some(&positions), 0, 0).unwrap();
    command.set_vertex_buffer(Some(&colors), 0, 1).unwrap();
    command.set_fragment_buffer(Some(&arguments), 0, 0).unwrap();
    command
        .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
        .unwrap();
    event.set_signaled_value(1);
    indirect_host_steps::wait(&command_buffer);
    let after = device.work_counts();
    assert_eq!(
        (
            after.sampler_sets_made - created.sampler_sets_made,
            after.host_step_sampler_sets_made - created.host_step_sampler_sets_made,
        ),
        (0, 0),
        "the host step shared the state's set"
    );
    (state, command_buffer, target, texels)
}

/// The descriptor of [`host_step_s_shared_set`]'s state: the fixture's `colored_vertex` and
/// `argument_sampled_fragment` into an `RGBA8Unorm` color, for indirect command buffers.
pub(super) fn shared_set_descriptor(device: &Device) -> RenderPipelineDescriptor {
    let ll = metal2vulkan_devtools::fixtures_dir().join("examples/icb_argument_sampling.ll");
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
    let mut descriptor = RenderPipelineDescriptor::new(function("colored_vertex"));
    descriptor.fragment = Some(function("argument_sampled_fragment"));
    descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    descriptor.support_indirect_command_buffers = true;
    descriptor
}

/// A host step's draw of a sampled state made after the commit ([`host_step_s_shared_set`]),
/// sharing the set the state's creation made, against the device's last handle: the state and
/// the completed command buffer that ran the step, in every order.
#[test]
fn a_host_step_s_shared_sampler_set_outlives_the_device_in_every_order() {
    in_every_order(2, |device| {
        let (state, command_buffer, _, _) = host_step_s_shared_set(device);
        held(state, command_buffer)
    });
}

/// An uncommitted command buffer whose draw holds the only handle of a sampled state's variant,
/// and so a share of the device's sampler set, beside a live state that shares the set, against
/// the device's last handle, in every order. The draw's texture is a plain one: the variant is
/// the state's own, made at its creation.
#[test]
fn a_command_buffer_holding_a_shared_sampler_set_outlives_the_device_in_every_order() {
    in_every_order(2, |device| {
        let before = device.work_counts();
        let state = render_state(device, "sampled_fragment", PixelFormat::RGBA8Unorm);
        let live = render_state(device, "sampled_fragment", PixelFormat::BGRA8Unorm);
        let texture = mapped_texture(device, AddressModeMap::IDENTITY);
        let command_buffer =
            encoded_draw_as(device, &state, PrimitiveType::Triangle, Some(&texture));
        drop(state);
        let after = device.work_counts();
        assert_eq!(
            (
                after.sampler_sets_made - before.sampler_sets_made,
                after.sampler_sets_shared - before.sampler_sets_shared,
            ),
            (1, 1),
            "the two states share one set"
        );
        assert_eq!(command_buffer.status(), CommandBufferStatus::NotEnqueued);
        held(command_buffer, live)
    });
}

/// A draw whose `VkPipeline` its state's creation did not make (a line strip, of another class
/// than the triangle list the creation predicted; a triangle strip shares the list's pipeline
/// where the topology is dynamic), handed to the device's pool and maybe still being made when the
/// command buffer, the state and the device's last handle go, in every order: the state keeps what
/// the pool's piece holds, its variant and so the device, and its release stops the piece not
/// started and hands what a running one gives back to the device's reaper, so the device goes on
/// the thread that lets go of its last holder, never on the pool's. The test waits for the
/// teardown: an exit with a making still in flight is the open case, which this test no longer
/// reaches.
#[test]
fn a_draw_s_making_in_the_pool_outlives_the_device_in_every_order() {
    in_every_order(2, |device| {
        let state = render_state(device, "colored_fragment", PixelFormat::RGBA8Unorm);
        let before = device.work_counts();
        let command_buffer = encoded_draw_as(device, &state, PrimitiveType::LineStrip, None);
        assert_eq!(
            device.work_counts().pipelines_pooled - before.pipelines_pooled,
            1,
            "the line strip's pipeline went to the pool"
        );
        held(command_buffer, state)
    });
}

/// A command buffer committed and completed before its state and the device go: its draw's
/// variant, released as it completed, goes with the state, whichever of the three goes last.
#[test]
fn a_completed_command_buffer_s_state_outlives_the_device_in_every_order() {
    in_every_order(2, |device| {
        let state = render_state(device, "colored_fragment", PixelFormat::RGBA8Unorm);
        let command_buffer = encoded_draw(device, &state);
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        held(state, command_buffer)
    });
}
