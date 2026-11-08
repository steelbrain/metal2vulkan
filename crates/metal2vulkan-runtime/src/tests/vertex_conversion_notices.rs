//! A render state announces what the device's fixed-function vertex input converts for its vertex
//! descriptor, at a draw that binds it, while one translation of its vertex function serves every
//! format: a normalized attribute's `device-vertex-conversion` (none on a device whose profile
//! takes `DriverDeviation::MetalNormalizedVertexFetch`, whose conversion is Metal's: MoltenVK on
//! the Apple M3 Max's macOS 27 identity), a narrow float's read as `float` `device-nan-encoding`,
//! and a 32-bit float attribute's neither. The translator decides the notes where it links the
//! state's pipeline (`pipeline::link::vertex::fixed_function_notes`) and puts them on its vertex
//! stage, whose module names no part of the format; so the states differ in their notes and not
//! in their function's compile key (`WorkCounts::functions_translated`).
//!
//! Each draw fetches one record of `0x7f` bytes and stores the attribute's four words, printed:
//! `UChar4Normalized` reads 127/255 (Metal's `0x3EFEFEFF`; a driver that multiplies by the
//! reciprocal answers `0x3EFEFF00`), and `Half4` reads the `half` NaN `0x7F7F` (Metal's
//! `0x7FC00000`; the device's widening may keep the payload).
//!
//! A device test: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`), on the kind
//! of device the cfg selects. The function is the owned fixture
//! `validation/fixtures/public/engine/dsx_int_attributes.ll`'s `int_attribute_to_float4`
//! (`llvm-as`).

use super::blit_holds::{Scratch, WAIT};
use super::encoder_holds::{library, words_of};
use crate::{
    CommandBufferStatus, Degradation, DegradeNotice, Device, DeviceKind, DeviceOptions,
    EncoderCall, Library, MessageSink, PipelineOptions, RenderPassDescriptor, RenderPipelineState,
    ResourceOptions, ShaderCacheOptions,
};
use metal2vulkan::device::RenderLoweringPolicy;
use metal2vulkan::error::DriverDeviation;
use metal2vulkan::metal::{PrimitiveType, VertexFormat};
use metal2vulkan::pipeline::{
    RenderPipelineDescriptor, StageSource, VertexAttributeDescriptor, VertexBufferLayoutDescriptor,
    VertexDescriptor,
};
use std::path::Path;
use std::sync::{Arc, Mutex};

/// The kinds the states' vertex input may announce, each announced at every occurrence.
const NOTES: [Degradation; 2] = [
    Degradation::DeviceVertexConversion,
    Degradation::DeviceNanEncoding,
];

/// The notices a device announced.
type Heard = Arc<Mutex<Vec<DegradeNotice>>>;

/// A device of the kind the cfg selects, with an empty shader cache of its own, that announces
/// [`NOTES`] at every occurrence, its notices into `heard`.
fn device(cache: &Path, heard: &Heard) -> Device {
    device_lowering(cache, heard, RenderLoweringPolicy::AtCall)
}

/// [`device`], lowering render calls as `lowering` says.
fn device_lowering(cache: &Path, heard: &Heard, lowering: RenderLoweringPolicy) -> Device {
    let heard = Arc::clone(heard);
    let options = DeviceOptions {
        kinds: match cfg!(m2v_cpu_icd) {
            true => vec![DeviceKind::Cpu],
            false => vec![
                DeviceKind::Discrete,
                DeviceKind::Integrated,
                DeviceKind::Virtual,
            ],
        },
        shader_cache: ShaderCacheOptions {
            directory: Some(cache.to_owned()),
            ..ShaderCacheOptions::default()
        },
        messages: Some(MessageSink::new(move |message| {
            if let Some(notice) = &message.degradation {
                heard.lock().unwrap().push(notice.clone());
            }
        })),
        announce_each: NOTES.into_iter().collect(),
        ..DeviceOptions::default()
    };
    let mut options = options;
    options.policy.render_lowering = lowering;
    Device::system_default_with(&options).unwrap()
}

/// A state of `int_attribute_to_float4`, which stores its `float4` attribute 0 to buffer 0 at
/// its vertex: rasterization disabled, attribute 0 of `format` at the start of layout 1's
/// 16-byte records.
fn state(device: &Device, library: &Library, format: VertexFormat) -> RenderPipelineState {
    let function = library
        .new_function_with_name("int_attribute_to_float4")
        .unwrap();
    let mut descriptor =
        RenderPipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    descriptor.rasterization_enabled = false;
    let mut vertex = VertexDescriptor::new();
    vertex
        .attributes
        .insert(0, VertexAttributeDescriptor::new(format, 0, 1));
    vertex
        .layouts
        .insert(1, VertexBufferLayoutDescriptor::new(Some(16)));
    descriptor.vertex_descriptor = Some(vertex);
    device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// One point of `state` in a pass without attachments, its record `0x7f` bytes, committed and
/// waited for: the words it stored, and the kinds of [`NOTES`] the draw announced, sorted.
fn draw(
    device: &Device,
    state: &RenderPipelineState,
    heard: &Heard,
) -> (Vec<u32>, Vec<Degradation>) {
    heard.lock().unwrap().clear();
    let out = device.new_buffer(16, ResourceOptions::default()).unwrap();
    let mut pass = RenderPassDescriptor::new();
    pass.render_target_width = 1;
    pass.render_target_height = 1;
    pass.default_raster_sample_count = 1;
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
    encoder.set_render_pipeline_state(state).unwrap();
    encoder.set_vertex_bytes(&[0x7f; 16], 1).unwrap();
    encoder.set_vertex_buffer(Some(&out), 0, 0).unwrap();
    encoder
        .draw_primitives(PrimitiveType::Point, 0, 1, 1, 0)
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    let mut kinds: Vec<Degradation> = heard
        .lock()
        .unwrap()
        .iter()
        .map(|notice| notice.degradation)
        .filter(|kind| NOTES.contains(kind))
        .collect();
    kinds.sort();
    kinds.dedup();
    (words_of(&out), kinds)
}

/// States over one vertex function whose vertex descriptors differ only in the format of the
/// attribute fixed-function input fetches translate the function once, and each announces at its
/// draw exactly the conversions its format takes on the device: a normalized format's but where
/// the device's conversion is measured to be Metal's.
#[test]
fn a_state_announces_its_formats_conversions_from_one_translation() {
    let scratch = Scratch::new("vertex-conversion-notices");
    let heard = Heard::default();
    let device = device(&scratch.0, &heard);
    let library = library(&device, "engine/dsx_int_attributes.ll");
    let measured = device
        .compile_device()
        .profile()
        .takes_driver_deviation(DriverDeviation::MetalNormalizedVertexFetch);
    eprintln!(
        "VERTEX-CONVERSION: {} takes the measured normalized conversion: {measured}",
        device.name()
    );
    let normalized = match measured {
        true => vec![],
        false => vec![Degradation::DeviceVertexConversion],
    };
    let cases = [
        (VertexFormat::UChar4Normalized, normalized.clone()),
        (VertexFormat::Half4, vec![Degradation::DeviceNanEncoding]),
        (VertexFormat::Float4, vec![]),
        (VertexFormat::Short2Normalized, normalized),
    ];
    // The first creation translates the function; every later one must find it compiled.
    let first = state(&device, &library, cases[0].0);
    let before = device.work_counts();
    assert!(
        before.functions_translated > 0,
        "the first creation must translate the vertex function at all, or the zero below is \
         vacuous"
    );
    let mut states = vec![first];
    for (format, _) in &cases[1..] {
        states.push(state(&device, &library, *format));
    }
    let after = device.work_counts();
    assert_eq!(
        after.functions_translated - before.functions_translated,
        0,
        "states differing only in a fixed-function attribute's format must share the vertex \
         function's translation: its notes are the pipeline's, not the compile key's"
    );
    assert_eq!(
        after.functions_shared - before.functions_shared,
        cases.len() as u64 - 1,
        "each later creation asks for the vertex function once and finds it compiled"
    );
    for ((format, want), state) in cases.iter().zip(&states) {
        let (words, kinds) = draw(&device, state, &heard);
        eprintln!("VERTEX-CONVERSION: {format:?} stored {words:08x?}, announced {kinds:?}");
        assert_eq!(&kinds, want, "{format:?}");
    }
}

/// A bind's notice made once and announced at each later bind of the same variant (the
/// variant's `half` attribute's `device-nan-encoding`, asked for at every occurrence) names, at
/// each, the command buffer and the logged call of that bind, under `Worker`: a second command
/// buffer's draw, a call later in its encoder than the first's, is named as its own though its
/// notice was made at the first's bind.
#[test]
fn a_bind_s_notice_names_its_own_command_buffer_and_call() {
    let scratch = Scratch::new("vertex-conversion-binds");
    let heard = Heard::default();
    let device = device_lowering(&scratch.0, &heard, RenderLoweringPolicy::Worker);
    let library = library(&device, "engine/dsx_int_attributes.ll");
    let state = state(&device, &library, VertexFormat::Half4);
    let out = device.new_buffer(16, ResourceOptions::default()).unwrap();
    let mut pass = RenderPassDescriptor::new();
    pass.render_target_width = 1;
    pass.render_target_height = 1;
    pass.default_raster_sample_count = 1;
    let queue = device.new_command_queue().unwrap();
    // The draw of each command buffer, after `extra` repeated binds of buffer 0.
    let drawn = |extra: usize| {
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encoder.set_render_pipeline_state(&state).unwrap();
        encoder.set_vertex_bytes(&[0x7f; 16], 1).unwrap();
        for _ in 0..=extra {
            encoder.set_vertex_buffer(Some(&out), 0, 0).unwrap();
        }
        encoder
            .draw_primitives(PrimitiveType::Point, 0, 1, 1, 0)
            .unwrap();
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        let call = EncoderCall {
            selector: "drawPrimitives:vertexStart:vertexCount:instanceCount:baseInstance:",
            index: 3 + extra as u32,
        };
        (command_buffer, call)
    };
    let first = drawn(0);
    let made = device.work_counts().bound_notices_made;
    let second = drawn(1);
    assert_eq!(
        device.work_counts().bound_notices_made,
        made,
        "the second bind made its notices again: the naming below proves nothing"
    );
    let mut named: Vec<_> = (heard.lock().unwrap().iter())
        .filter(|notice| notice.degradation == Degradation::DeviceNanEncoding)
        .map(|notice| (notice.command_buffer, notice.call))
        .collect();
    // A bind may take a variant's pipeline twice; each names its own draw.
    named.dedup();
    let expected: Vec<_> = [&first, &second]
        .iter()
        .map(|(command_buffer, call)| (Some(command_buffer.object_id()), Some(*call)))
        .collect();
    assert_eq!(named, expected);
}
