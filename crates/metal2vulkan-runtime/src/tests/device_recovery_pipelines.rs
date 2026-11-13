//! What a device's recovery carries of the program's pipeline states (`pipeline/carry.rs`,
//! `pipeline/render/carry.rs`): a compute or drawing state made before it is the same state on the
//! new `VkDevice`, its base variant made again there from what its creation compiled, and its
//! dispatches or draws there run it; and how long a recovery takes with many live states, each of
//! its own pipeline.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs, on the kind of device
//! the cfg selects (docs/VALIDATION.md). No GPU fault: the loss is marked (`Device::recover_for_test`).

use super::device_recovery::{heard_device, outcome, recovery_notices, Heard, Scratch};
use super::device_recovery_carry::live_vk_devices;
use super::encoder_holds::library;
use super::encoder_holds::{bytes_of, pass, words_of};
use crate::{
    BlitOption, CommandBufferStatus, CommandQueue, ComputePipelineState, Device, Error, Function,
    Library, Origin, PipelineOptions, RenderPipelineState, ResourceOptions, Size,
    TextureDescriptor,
};
use metal2vulkan::metal::{BlendFactor, PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, ComputePipelineDescriptor, RenderPipelineDescriptor, StageSource,
};
use std::time::{Duration, Instant};

/// The threads each dispatch runs: one float each.
const COUNT: u64 = 256;

/// The live states of each kind the timed recovery carries.
const STATES: u32 = 200;

/// The side of the target each draw fills.
const SIDE: u64 = 16;

/// The color each draw writes.
const COLOR: [f32; 4] = [0.2, 0.4, 0.6, 1.0];

/// [`COLOR`] as an `RGBA8Unorm` texel's word.
const PACKED: u32 = u32::from_le_bytes([51, 102, 153, 255]);

/// The blend factors a drawing state of the timed recovery takes, by pairs: each with one source
/// factor of the dual-source-free ones.
const FACTORS: [BlendFactor; 15] = [
    BlendFactor::Zero,
    BlendFactor::One,
    BlendFactor::SourceColor,
    BlendFactor::OneMinusSourceColor,
    BlendFactor::SourceAlpha,
    BlendFactor::OneMinusSourceAlpha,
    BlendFactor::DestinationColor,
    BlendFactor::OneMinusDestinationColor,
    BlendFactor::DestinationAlpha,
    BlendFactor::OneMinusDestinationAlpha,
    BlendFactor::SourceAlphaSaturated,
    BlendFactor::BlendColor,
    BlendFactor::OneMinusBlendColor,
    BlendFactor::BlendAlpha,
    BlendFactor::OneMinusBlendAlpha,
];

/// `state`, `scale_and_offset`, dispatched over [`COUNT`] floats `i` with a scale of 2, as
/// `queue`'s GPU work runs it, completed: what it wrote, `3 i` each. Refused as the encoding
/// refuses the state.
fn run(
    device: &Device,
    queue: &CommandQueue,
    state: &ComputePipelineState,
) -> Result<Vec<f32>, Error> {
    let input: Vec<u8> = (0..COUNT).flat_map(|i| (i as f32).to_le_bytes()).collect();
    let input = device.new_buffer_with_bytes(&input, ResourceOptions::shared())?;
    let output = device.new_buffer(COUNT * 4, ResourceOptions::shared())?;
    let scale = device.new_buffer_with_bytes(&2.0f32.to_le_bytes(), ResourceOptions::shared())?;
    let command_buffer = queue.command_buffer()?;
    let mut encoder = command_buffer.compute_command_encoder()?;
    encoder.set_compute_pipeline_state(state)?;
    encoder.set_buffer(Some(&input), 0, 0)?;
    encoder.set_buffer(Some(&output), 0, 1)?;
    encoder.set_buffer(Some(&scale), 0, 2)?;
    encoder.dispatch_threads(Size::new(COUNT, 1, 1), Size::new(32, 1, 1))?;
    encoder.end_encoding()?;
    command_buffer.commit()?;
    assert_eq!(
        outcome(&command_buffer),
        (CommandBufferStatus::Completed, None)
    );
    let pointer = output.contents().unwrap();
    // SAFETY: the buffer's bytes, which the completed work wrote and nothing else uses.
    let bytes = unsafe { std::slice::from_raw_parts(pointer.as_ptr(), (COUNT * 4) as usize) };
    Ok((bytes.as_chunks::<4>().0.iter())
        .map(|word| f32::from_le_bytes(*word))
        .collect())
}

/// What [`run`] reads back.
fn expected() -> Vec<f32> {
    (0..COUNT).map(|i| 3.0 * i as f32).collect()
}

/// `scale_and_offset` of the runtime's example fixture, on `device`.
fn kernel(device: &Device) -> Function {
    library(device, "examples/runtime_examples.ll")
        .new_function_with_name("scale_and_offset")
        .unwrap()
}

/// A state of `function`, labelled `label`: of at most `most` threads a threadgroup (a field of
/// its pipeline's key) where `most` is given, else one whose dispatches take 32 threads a
/// threadgroup, which its creation predicts. Metal refuses a request that states both and
/// differs.
fn state(
    device: &Device,
    function: &Function,
    most: Option<u32>,
    label: &str,
) -> ComputePipelineState {
    let mut descriptor =
        ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    match most {
        Some(most) => descriptor.max_total_threads_per_threadgroup = Some(most),
        None => descriptor.required_threads_per_threadgroup = Some([32, 1, 1]),
    }
    let options = PipelineOptions {
        label: Some(label.to_owned()),
        ..PipelineOptions::default()
    };
    device
        .new_compute_pipeline_state(&descriptor, &options)
        .unwrap()
}

/// A compute state made before a recovery is the same state after it (its `uniqueIdentifier` and
/// label, a borrow of which taken before it reads the same), on the next generation, as the
/// recovery's notice counts; its dispatches there run it and write what they wrote before; and
/// the lost generation goes at its rest, its program there let go of while the state lives.
#[test]
fn a_recovery_carries_a_compute_state() {
    let scratch = Scratch::new();
    let heard = Heard::default();
    let device = heard_device(&scratch.0, &heard);
    let queue = device.new_command_queue().unwrap();
    let state = state(&device, &kernel(&device), None, "scaled");
    assert_eq!(run(&device, &queue, &state).unwrap(), expected());
    let identity = (state.unique_identifier(), state.label().map(str::to_owned));
    let label = state.label();
    let from = heard.lock().unwrap().len();
    device.recover_for_test().unwrap();
    let said = recovery_notices(&heard, from, "the device");
    assert_eq!(said.len(), 1, "{said:?}");
    assert!(
        said[0].contains(
            ", and 1 pipeline states made again (their predicted pipelines queued), 0 more whose \
             making there failed"
        ),
        "{said:?}"
    );
    assert!(state.device().generation_serves());
    assert_eq!(
        (state.unique_identifier(), state.label().map(str::to_owned)),
        identity
    );
    assert_eq!(run(&device, &queue, &state).unwrap(), expected());
    // The lost `VkDevice` went at its rest, the state's program there let go of.
    assert_eq!(live_vk_devices(&device, 1), 1);
    assert_eq!(device.work_counts().lost_programs_released, 1);
    assert_eq!(label, Some("scaled"));
    assert_eq!(run(&device, &queue, &state).unwrap(), expected());
}

/// A drawing state of `tob_rect` and `tob_solid` of `library` into `RGBA8Unorm`, blending by
/// `factors` (source, destination) where given, labelled `label`.
fn drawing_state(
    device: &Device,
    library: &Library,
    factors: Option<(BlendFactor, BlendFactor)>,
    label: &str,
) -> RenderPipelineState {
    let vertex = library.new_function_with_name("tob_rect").unwrap();
    let fragment = library.new_function_with_name("tob_solid").unwrap();
    let mut descriptor =
        RenderPipelineDescriptor::new(StageSource::Function(vertex.function_descriptor()));
    descriptor.fragment = Some(StageSource::Function(fragment.function_descriptor()));
    let mut color = ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm);
    if let Some((source, destination)) = factors {
        color.blending_enabled = true;
        color.source_rgb_blend_factor = source;
        color.destination_rgb_blend_factor = destination;
    }
    descriptor.color_attachments[0] = Some(color);
    let options = PipelineOptions {
        label: Some(label.to_owned()),
        ..PipelineOptions::default()
    };
    device
        .new_render_pipeline_state(&descriptor, &options)
        .unwrap()
}

/// The library of `tob_rect` and `tob_solid`, on `device`.
fn drawing_library(device: &Device) -> Library {
    library(device, "engine/tob_renders.ll")
}

/// `state` drawing a rectangle over a [`SIDE`]-square target in [`COLOR`], as `queue`'s GPU work
/// runs it, completed: the target's texels' words. Refused as the encoding refuses the state.
fn draw(
    device: &Device,
    queue: &CommandQueue,
    state: &RenderPipelineState,
) -> Result<Vec<u32>, Error> {
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
    descriptor.usage = TextureUsage::RENDER_TARGET;
    descriptor.resource_options = ResourceOptions::private();
    let target = device.new_texture(&descriptor)?;
    let back = device.new_buffer(SIDE * SIDE * 4, ResourceOptions::shared())?;
    let command_buffer = queue.command_buffer()?;
    let mut encoder = command_buffer.render_command_encoder(&pass(&target))?;
    encoder.set_render_pipeline_state(state)?;
    encoder.set_vertex_bytes(&bytes_of(&[-1.0f32, -1.0, 1.0, 1.0], f32::to_le_bytes), 0)?;
    encoder.set_fragment_bytes(&bytes_of(&COLOR, f32::to_le_bytes), 0)?;
    encoder.draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)?;
    encoder.end_encoding()?;
    let mut blit = command_buffer.blit_command_encoder()?;
    blit.copy_from_texture_to_buffer(
        &target,
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(SIDE, SIDE, 1),
        &back,
        0,
        SIDE * 4,
        SIDE * SIDE * 4,
        BlitOption::None,
    )?;
    blit.end_encoding()?;
    command_buffer.commit()?;
    assert_eq!(
        outcome(&command_buffer),
        (CommandBufferStatus::Completed, None)
    );
    Ok(words_of(&back))
}

/// What [`draw`] reads back.
fn drawn() -> Vec<u32> {
    vec![PACKED; (SIDE * SIDE) as usize]
}

/// A drawing state made before a recovery is the same state after it (its `uniqueIdentifier` and
/// label, a borrow of which taken before it reads the same), on the next generation, as the
/// recovery's notice counts; its draws there draw what they drew before; and the lost generation
/// goes at its rest, its program there let go of while the state lives.
#[test]
fn a_recovery_carries_a_drawing_state() {
    let scratch = Scratch::new();
    let heard = Heard::default();
    let device = heard_device(&scratch.0, &heard);
    let queue = device.new_command_queue().unwrap();
    let state = drawing_state(&device, &drawing_library(&device), None, "drawn");
    assert_eq!(draw(&device, &queue, &state).unwrap(), drawn());
    let identity = (state.unique_identifier(), state.label().map(str::to_owned));
    let label = state.label();
    let from = heard.lock().unwrap().len();
    device.recover_for_test().unwrap();
    let said = recovery_notices(&heard, from, "the device");
    assert_eq!(said.len(), 1, "{said:?}");
    assert!(
        said[0].contains(
            ", and 1 pipeline states made again (their predicted pipelines queued), 0 more whose \
             making there failed"
        ),
        "{said:?}"
    );
    assert!(state.device().generation_serves());
    assert_eq!(
        (state.unique_identifier(), state.label().map(str::to_owned)),
        identity
    );
    assert_eq!(draw(&device, &queue, &state).unwrap(), drawn());
    // The lost `VkDevice` went at its rest, the state's program there let go of.
    assert_eq!(live_vk_devices(&device, 1), 1);
    assert_eq!(device.work_counts().lost_programs_released, 1);
    assert_eq!(label, Some("drawn"));
    assert_eq!(draw(&device, &queue, &state).unwrap(), drawn());
}

/// How long `device`'s recovery takes, `recover_for_test` returning once the new `VkDevice`
/// serves with what it carried; its notice, from `heard` past `from`, counts `states` states made
/// again and none failed.
fn recovery_time(device: &Device, heard: &Heard, states: u32) -> Duration {
    let from = heard.lock().unwrap().len();
    let began = Instant::now();
    device.recover_for_test().unwrap();
    let took = began.elapsed();
    let said = recovery_notices(heard, from, "the device");
    let counted = format!(
        ", and {states} pipeline states made again (their predicted pipelines queued), 0 more \
         whose making there failed"
    );
    assert!(states == 0 || said[0].contains(&counted), "{said:?}");
    took
}

/// A recovery's wall time with no live state, with [`STATES`] live compute states of as many
/// pipelines, and with [`STATES`] live drawing states of as many pipelines, each state made
/// again (a drawing state's predicted `VkPipeline` queued; the compute states, of no stated
/// threadgroup size, predict none), logged as `RECOVERY-WALL`; and a state of each kind made
/// again runs there.
#[test]
fn a_recovery_with_many_live_states_is_timed() {
    let scratch = Scratch::new();
    let heard = Heard::default();
    let device = heard_device(&scratch.0, &heard);
    let queue = device.new_command_queue().unwrap();
    let bare = recovery_time(&device, &heard, 0);
    let function = kernel(&device);
    let states: Vec<_> = (0..STATES)
        .map(|most| state(&device, &function, Some(32 + most), "timed"))
        .collect();
    let compute = recovery_time(&device, &heard, STATES);
    assert_eq!(run(&device, &queue, &states[0]).unwrap(), expected());
    drop(states);
    let library = drawing_library(&device);
    let pairs = FACTORS
        .iter()
        .flat_map(|&s| FACTORS.iter().map(move |&d| (s, d)));
    let states: Vec<_> = pairs
        .take(STATES as usize)
        .map(|pair| drawing_state(&device, &library, Some(pair), "timed"))
        .collect();
    let drawing = recovery_time(&device, &heard, STATES);
    eprintln!(
        "RECOVERY-WALL: {}: no live state {bare:?}, {STATES} live compute states {compute:?}, \
         {STATES} live drawing states {drawing:?}",
        device.name()
    );
    // The state blending `One` and `Zero`: what it draws is the color.
    assert_eq!(draw(&device, &queue, &states[15]).unwrap(), drawn());
}
