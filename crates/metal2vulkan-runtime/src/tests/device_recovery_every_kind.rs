//! Every kind of object a program makes and keeps, made before a device's recovery, serves after
//! it, as Metal's objects go on (`device/carry.rs`): each is used on the next generation (encoded
//! with, bound, reset, written) by command buffers that complete, and the lost generation goes
//! while they all live. The kinds: a command queue, a `Private` buffer, a heap and a buffer
//! placed in it, a texture, an event, a shared event, a fence, a sampler state, a depth/stencil
//! state, a library and a function, a compute and a drawing pipeline state, an address pool, a
//! rasterization rate map, a binary archive, a dynamic library, an indirect command buffer of
//! dispatches and one of draws, an argument encoder and a visible function table.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs, on the kind of device
//! the cfg selects (docs/VALIDATION.md). No GPU fault: the loss is marked (`Device::recover_for_test`).
//! The functions are the owned fixtures `examples/runtime_examples.ll`, `engine/tob_renders.ll`,
//! `engine/visible_table_caller.ll` and `engine/visible_table_functions.ll` (`llvm-as`).

use super::device_recovery::{heard_device, outcome, recovery_notices, Heard, Scratch};
use super::device_recovery_carry::{fill, live_vk_devices, read, BYTES};
use super::device_recovery_live_table::{
    dispatched, encode_command, execute, indirect_state, WORDS,
};
use super::device_recovery_pipelines::{draw, drawing_library, drawing_state, drawn};
use super::device_recovery_states::dynamic_library;
use super::encoder_holds::library;
use crate::{
    AddressPlacement, BinaryArchiveDescriptor, CommandBufferStatus, CommandQueue,
    ComputePipelineState, Device, EventRef, HeapDescriptor, HeapType,
    IndirectCommandBufferDescriptor, LoadAction, PipelineOptions, RenderPassDescriptor,
    ResourceOptions, Size, StoreAction, TextureDescriptor, VisibleFunctionTable,
    VisibleFunctionTableDescriptor,
};
use metal2vulkan::bind::{
    ArgumentDescriptor, RateMapDescriptor, RateMapLayerDescriptor, SamplerDescriptor,
};
use metal2vulkan::metal::state::DepthStencilDescriptor;
use metal2vulkan::metal::{
    CompareFunction, DataType, IndirectCommandType, PixelFormat, TextureUsage,
};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, LinkedFunctions, StageSource};

/// The threads a table's dispatch runs, each calling entry `tid % 5` of its table.
const THREADS: u32 = 40;

/// `visible_table_dispatch` linking `table_add`, `table_mul`, `table_sub` and `table_half`, and
/// a table of 5 entries of it calling `table_add` at 0 and `table_mul` at 1.
fn table_state(device: &Device) -> (ComputePipelineState, VisibleFunctionTable) {
    let caller = library(device, "engine/visible_table_caller.ll");
    let functions = library(device, "engine/visible_table_functions.ll");
    let linked = ["table_add", "table_mul", "table_sub", "table_half"]
        .iter()
        .fold(LinkedFunctions::new(), |linked, name| {
            linked.with_function(
                functions
                    .new_function_with_name(name)
                    .unwrap()
                    .linked_function(),
            )
        });
    let kernel = caller
        .new_function_with_name("visible_table_dispatch")
        .unwrap();
    let state = device
        .new_compute_pipeline_state(
            &ComputePipelineDescriptor::new(StageSource::Function(
                kernel.function_descriptor().with_linked(linked),
            )),
            &PipelineOptions::default(),
        )
        .unwrap();
    let table = state
        .new_visible_function_table(&VisibleFunctionTableDescriptor::new(5))
        .unwrap();
    let handle = |name: &str| {
        let function = functions.new_function_with_name(name).unwrap();
        state.function_handle(&function).unwrap().unwrap()
    };
    let (add, mul) = (handle("table_add"), handle("table_mul"));
    table.set_functions(&[Some(&add), Some(&mul)], 0).unwrap();
    (state, table)
}

/// `state` over [`THREADS`] threads with `table` at buffer 1, as `queue`'s GPU work runs it,
/// completed: the words it wrote at buffer 0.
fn table_words(
    device: &Device,
    queue: &CommandQueue,
    state: &ComputePipelineState,
    table: &VisibleFunctionTable,
) -> Vec<u32> {
    let out = device.new_buffer(BYTES, ResourceOptions::shared()).unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(state).unwrap();
    encoder.set_buffer(Some(&out), 0, 0).unwrap();
    encoder
        .set_visible_function_tables(&[Some(table)], 1)
        .unwrap();
    encoder
        .dispatch_threads(
            Size::new(THREADS.into(), 1, 1),
            Size::new(THREADS.into(), 1, 1),
        )
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    assert_eq!(
        outcome(&command_buffer),
        (CommandBufferStatus::Completed, None)
    );
    let contents = out.contents().unwrap();
    // SAFETY: a `Shared` buffer of `BYTES` bytes, which no work in flight writes.
    let bytes = unsafe { std::slice::from_raw_parts(contents.as_ptr(), BYTES as usize) };
    (bytes.as_chunks::<4>().0.iter())
        .take(THREADS as usize)
        .map(|word| u32::from_le_bytes(*word))
        .collect()
}

/// What [`table_words`] reads back from a table calling `table_add` at 0 and `table_mul` at 1,
/// and `table_sub` at 3 where `sub`; its other entries null (`0xfffffffd`).
fn called(sub: bool) -> Vec<u32> {
    (0..THREADS)
        .map(|tid| match tid % 5 {
            0 => tid + 3,
            1 => tid * 3,
            3 if sub => tid.wrapping_sub(3),
            _ => 0xffff_fffd,
        })
        .collect()
}

/// One of each kind of object a program makes and keeps, made before a recovery (the module
/// docs), serves after it: each is used by the next generation's command buffers, which
/// complete, and does what it did before. The indirect command buffer of dispatches runs its
/// command as encoded before where the recovery kept its records (they lie in the runtime's own
/// pages), else runs nothing, every command reset as its notice says; either way a blit's reset
/// of its commands and of the draws' buffer's, then a command encoded again, run. The visible
/// function table's entries call the functions they called, and one set after. The argument
/// encoder writes a buffer made before into an argument buffer made after. The recovery's notice
/// counts the indirect command buffers and the table; the lost `VkDevice` goes.
#[test]
fn every_kind_of_object_made_before_a_recovery_serves_after_it() {
    let scratch = Scratch::new();
    let heard = Heard::default();
    let device = heard_device(&scratch.0, &heard);
    let queue = device.new_command_queue().unwrap();
    let private = || {
        device
            .new_buffer(BYTES, ResourceOptions::private())
            .unwrap()
    };
    let buffer = private();
    fill(&queue, &buffer, 7);
    let heap = device
        .new_heap(&HeapDescriptor::new(
            HeapType::Automatic,
            1 << 20,
            ResourceOptions::private(),
        ))
        .unwrap();
    let placed = heap.new_buffer(BYTES, ResourceOptions::private()).unwrap();
    let mut target = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 64, 64, false);
    target.usage = TextureUsage::RENDER_TARGET;
    target.resource_options = ResourceOptions::private();
    let texture = device.new_texture(&target).unwrap();
    let event = device.new_event().unwrap();
    let shared_event = device.new_shared_event().unwrap();
    let fence = device.new_fence();
    let sampler = device.new_sampler_state(&SamplerDescriptor::new()).unwrap();
    let mut depth = DepthStencilDescriptor::default();
    depth.depth_compare_function = CompareFunction::LessEqual;
    let depth_stencil = device.new_depth_stencil_state(&depth).unwrap();
    let examples = library(&device, "examples/runtime_examples.ll");
    let function = examples.new_function_with_name("scale_and_offset").unwrap();
    let compute = indirect_state(&device);
    let drawing = drawing_state(&device, &drawing_library(&device), None, "drawn");
    let pool = device
        .reserve_address_pool(AddressPlacement::Any, 2 * BYTES)
        .unwrap();
    let mut rates = RateMapDescriptor::new([64, 64]);
    rates.layers = vec![RateMapLayerDescriptor::new(vec![1.0, 0.5], vec![1.0, 0.5])];
    let map = device.new_rasterization_rate_map(&rates).unwrap();
    let described =
        ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    let archive = device
        .new_binary_archive(&BinaryArchiveDescriptor::new(None))
        .unwrap();
    archive.add_compute_pipeline_functions(&described).unwrap();
    let dynamic = dynamic_library(&device, &scratch.0);
    let mut dispatches = IndirectCommandBufferDescriptor::new();
    dispatches.command_types = IndirectCommandType::CONCURRENT_DISPATCH_THREADS;
    dispatches.max_kernel_buffer_bind_count = 3;
    let icb = device
        .new_indirect_command_buffer(&dispatches, 1, ResourceOptions::shared())
        .unwrap();
    let (input, output, scale) = (private(), private(), private());
    let buffers = [&input, &output, &scale];
    encode_command(&icb, &compute, buffers);
    assert_eq!(execute(&device, &queue, &icb, buffers), dispatched());
    let mut draws = IndirectCommandBufferDescriptor::new();
    draws.command_types = IndirectCommandType::DRAW;
    draws.max_vertex_buffer_bind_count = 1;
    let draw_icb = device
        .new_indirect_command_buffer(&draws, 2, ResourceOptions::shared())
        .unwrap();
    let mut arguments = device
        .new_argument_encoder_with_arguments(&[ArgumentDescriptor::new(DataType::Pointer, 0)])
        .unwrap();
    let (table_state, table) = table_state(&device);
    assert_eq!(
        table_words(&device, &queue, &table_state, &table),
        called(false)
    );

    let from = heard.lock().unwrap().len();
    device.recover_for_test().unwrap();
    let said = recovery_notices(&heard, from, "the device");
    assert_eq!(said.len(), 1, "{said:?}");
    assert!(
        said[0].contains(", and 2 indirect command buffers, ")
            && said[0].contains(", and 1 visible function tables with their entries"),
        "{said:?}"
    );

    // The buffer (its contents lost: it reads zero), the heap's buffer and one made in the heap
    // after, the address pool's slot, through the queue made before.
    assert_eq!(
        read(&device, &queue, &buffer).unwrap(),
        vec![0; BYTES as usize]
    );
    let in_heap = heap.new_buffer(BYTES, ResourceOptions::private()).unwrap();
    for (byte, buffer) in [(3, &placed), (4, &in_heap)] {
        fill(&queue, buffer, byte);
        assert_eq!(
            read(&device, &queue, buffer).unwrap(),
            vec![byte; BYTES as usize]
        );
    }
    let slot = device
        .new_buffer_with_descriptor(
            crate::BufferDescriptor::new(
                BYTES,
                ResourceOptions::private(),
                crate::BufferContents::Zero,
            )
            .with_pool_slot(&pool, 0),
        )
        .unwrap();
    assert_eq!(slot.gpu_address(), pool.gpu_address());

    // The compute state made before, and one made of the function made before, the sampler
    // state bound, the fence updated and waited for, the events signalled.
    let made_after = device
        .new_compute_pipeline_state(&described, &PipelineOptions::default())
        .unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    for state in [&compute, &made_after] {
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(state).unwrap();
        encoder.set_buffer(Some(&input), 0, 0).unwrap();
        encoder.set_buffer(Some(&output), 0, 1).unwrap();
        encoder.set_buffer(Some(&scale), 0, 2).unwrap();
        encoder.set_sampler_state(Some(&sampler), 0).unwrap();
        encoder
            .dispatch_threads(Size::new(WORDS, 1, 1), Size::new(32, 1, 1))
            .unwrap();
        encoder.update_fence(&fence).unwrap();
        encoder.end_encoding().unwrap();
    }
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.wait_for_fence(&fence).unwrap();
    blit.end_encoding().unwrap();
    (command_buffer.encode_signal_event(EventRef::from(&event), 1)).unwrap();
    (command_buffer.encode_signal_event(EventRef::from(&shared_event), 2)).unwrap();
    command_buffer.commit().unwrap();
    assert_eq!(
        outcome(&command_buffer),
        (CommandBufferStatus::Completed, None)
    );
    assert!(shared_event.wait_until_signaled_value(2, super::device_recovery::WAIT));

    // The drawing state; the depth/stencil state in a pass over the texture made before; a pass
    // over the rate map, its target of the map's physical size.
    assert_eq!(draw(&device, &queue, &drawing).unwrap(), drawn());
    let mut depth_target = TextureDescriptor::texture_2d(PixelFormat::Depth32Float, 64, 64, false);
    depth_target.usage = TextureUsage::RENDER_TARGET;
    depth_target.resource_options = ResourceOptions::private();
    let depth_target = device.new_texture(&depth_target).unwrap();
    let size = map.physical_size_for_layer(0);
    let mut mapped =
        TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, size.width, size.height, false);
    mapped.usage = TextureUsage::RENDER_TARGET;
    mapped.resource_options = ResourceOptions::private();
    let mapped = device.new_texture(&mapped).unwrap();
    for over_map in [false, true] {
        let mut pass = RenderPassDescriptor::new();
        let attachment = &mut pass.color_attachments[0];
        attachment.texture = Some(match over_map {
            false => texture.clone(),
            true => mapped.clone(),
        });
        attachment.load_action = LoadAction::Clear;
        attachment.store_action = StoreAction::Store;
        if over_map {
            pass.rasterization_rate_map = Some(map.clone());
        } else {
            pass.depth_attachment.texture = Some(depth_target.clone());
            pass.depth_attachment.load_action = LoadAction::Clear;
            pass.depth_attachment.store_action = StoreAction::DontCare;
        }
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        if !over_map {
            encoder.set_depth_stencil_state(&depth_stencil).unwrap();
        }
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        assert_eq!(
            outcome(&command_buffer),
            (CommandBufferStatus::Completed, None)
        );
    }

    // The archive takes an add, a creation listing it makes its state; the dynamic library
    // keeps its install name.
    archive.add_compute_pipeline_functions(&described).unwrap();
    let options = PipelineOptions {
        binary_archives: vec![archive.clone()],
        ..PipelineOptions::default()
    };
    device
        .new_compute_pipeline_state(&described, &options)
        .unwrap();
    assert!(dynamic.install_name().ends_with("libdyl.metallib"));

    // The indirect command buffers: the command as the recovery left it, then each buffer's
    // commands reset by a blit, and the command encoded again, run.
    let command = icb.indirect_compute_command_at(0).unwrap();
    let kept = command.command_type().unwrap() == IndirectCommandType::CONCURRENT_DISPATCH_THREADS;
    let ran = execute(&device, &queue, &icb, buffers);
    match kept {
        true => assert_eq!(ran, dispatched(), "the command kept"),
        false => assert!(
            ran.iter().all(|word| word.to_bits() == u32::MAX),
            "a reset command ran: {ran:?}"
        ),
    }
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.reset_commands_in_buffer(&icb, 0..1).unwrap();
    blit.reset_commands_in_buffer(&draw_icb, 0..2).unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    assert_eq!(
        outcome(&command_buffer),
        (CommandBufferStatus::Completed, None)
    );
    assert!(icb.device().generation_serves());
    encode_command(&icb, &compute, buffers);
    assert_eq!(execute(&device, &queue, &icb, buffers), dispatched());

    // The argument encoder writes the buffer made before into an argument buffer made after.
    let argument_buffer = device.new_buffer(BYTES, ResourceOptions::shared()).unwrap();
    arguments
        .set_argument_buffer(Some(&argument_buffer), 0)
        .unwrap();
    arguments.set_buffer(Some(&buffer), 0, 0).unwrap();
    let contents = argument_buffer.contents().unwrap();
    // SAFETY: a `Shared` buffer of `BYTES` bytes, which no work reads or writes.
    let word = unsafe { contents.as_ptr().cast::<u64>().read_unaligned() };
    assert_eq!(word, buffer.gpu_address().0);
    assert!(arguments.device().generation_serves());

    // The table calls what it called, and an entry set after.
    assert_eq!(
        table_words(&device, &queue, &table_state, &table),
        called(false)
    );
    let functions = library(&device, "engine/visible_table_functions.ll");
    let sub = functions.new_function_with_name("table_sub").unwrap();
    let sub = table_state.function_handle(&sub).unwrap().unwrap();
    table.set_function(Some(&sub), 3).unwrap();
    assert_eq!(
        table_words(&device, &queue, &table_state, &table),
        called(true)
    );

    // Nothing holds the lost `VkDevice`, while every object lives (to the end of the test).
    assert_eq!(live_vk_devices(&device, 1), 1, "{said:?}");
}
