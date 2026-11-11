//! A vertex attribute fetched in fixed function at elements past `2^32` bytes of a small buffer's
//! bind offset (the runtime's `render/vertex_reach.rs`), on the device the cfg selects, under the
//! validation layer. Metal reads each such element as zero (an Apple M3 Max under macOS 27.0.1,
//! the conformance case `vertex-fetch-past-4gib-small-buffer`); a driver whose fixed-function
//! vertex input takes an element's displacement modulo `2^32`
//! (`RuntimeDriverDefect::VertexFetchWrapsPast4GiB`: lavapipe) reads the bytes at its low 32 bits.
//!
//! The owned fixture `vertex_fetch_reach` writes `(v, vid + iid)` at `out[(vid + iid) & 3]` from
//! one `uint` attribute at buffer 1, so a draw of four vertices of one instance, or of one vertex
//! of four instances, writes four slots of `out` from any start; buffer 1 holds `0x10 + w` at word
//! `w`, 64 bytes, under a layout of stride 4 (per vertex or per instance), which every device
//! here fetches in fixed function.
//!
//! - **The draw's own values.** A vertex start, a 16-bit draw's base vertex and a base instance
//!   past `2^30` read zero under the default policy, and on a device with the defect the draw
//!   pulls the layout (`WorkCounts::far_vertex_fetch_pulled_draws`).
//! - **GPU memory's reach.** 32-bit indices past `2^30`, and an indirect draw's vertex start past
//!   it: under `VertexReachPolicy::Bounded` (the runtime's default) a device with the defect reads
//!   the wrapped words and announces `vertex-fetch-may-wrap` ([`MAY_WRAP`]) once, and every other
//!   device reads zero, announcing nothing: the defect's row measured on each device. Under
//!   `VertexReachPolicy::Exact` (`DevicePolicy::exact_answers()`, the lane's own) every device
//!   reads zero, and one with the defect pulls the layout.
//! - **Its cost** (`far_reach_costs`, ignored: a measurement, not a check): a draw of 32-bit
//!   indices fetched in fixed function and pulled, the pipelines each policy makes for a state
//!   drawn with and without indices, and each pass's time.

use super::{
    buffer_with, layer_reported_nothing_at_all, read_buffer, validated_device_with, Messages,
    ScratchCache, WAIT,
};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::device::VertexReachPolicy;
use metal2vulkan::error::RuntimeDriverDefect;
use metal2vulkan::metal::{IndexType, PrimitiveType, VertexFormat, VertexStepFunction};
use metal2vulkan::pipeline::{
    RenderPipelineDescriptor, StageSource, VertexAttributeDescriptor, VertexBufferLayoutDescriptor,
    VertexDescriptor,
};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    Buffer, CommandBufferStatus, CommandQueue, Degradation, Device, PipelineOptions,
    RenderCommandEncoder, RenderPassDescriptor, RenderPipelineState,
};
use std::time::Instant;

/// The kind a draw whose indices or arguments state its reach announces on a device with the
/// defect under `VertexReachPolicy::Bounded`
/// ([`gpu_memory_s_far_elements_wrap_where_the_device_s_fetch_wraps`]).
pub(super) const MAY_WRAP: [Degradation; 1] = [Degradation::VertexFetchMayWrap];

/// The first element past `2^30`: a stride of 4 places it at `2^32`.
const FAR: u32 = 1 << 30;

/// The vertices, or instances, of each draw.
const FETCHES: u32 = 4;

/// The attribute buffer's words: `0x10 + w` at word `w`.
const WORDS: u32 = 16;

/// The device under `policy`, the lane's validation layer reporting to `messages`.
fn device_under(messages: &Messages, cache: &ScratchCache, policy: VertexReachPolicy) -> Device {
    validated_device_with(TEST_DEVICE, messages, cache.path(), |options| {
        options.policy.vertex_reach = policy;
    })
    .unwrap()
}

/// Whether `device`'s driver takes a fixed-function element's displacement modulo `2^32`.
fn wraps(device: &Device) -> bool {
    (device.properties().driver_defects).contains(&RuntimeDriverDefect::VertexFetchWrapsPast4GiB)
}

/// The fixture's pipeline, rasterization disabled: its attribute a `uint` at offset 0 of buffer 1,
/// whose layout's stride is 4, stepped by `step`.
fn pipeline(device: &Device, step: VertexStepFunction) -> RenderPipelineState {
    let bitcode = llvm_assemble(&engine_dir().join("vertex_fetch_reach.ll")).unwrap();
    let library = device.new_library_with_air(&bitcode).unwrap();
    let function = library
        .new_function_with_name("vertex_fetch_reach")
        .unwrap();
    let mut descriptor =
        RenderPipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    descriptor.rasterization_enabled = false;
    let mut vertex = VertexDescriptor::new();
    let mut layout = VertexBufferLayoutDescriptor::new(Some(4));
    layout.step_function = step;
    vertex.layouts.insert(1, layout);
    vertex
        .attributes
        .insert(0, VertexAttributeDescriptor::new(VertexFormat::UInt, 0, 1));
    descriptor.vertex_descriptor = Some(vertex);
    device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// The words the attribute of each of `out`'s four slots read after one draw of `pipeline`, made
/// by `draw` over the attribute buffer bound at buffer 1, in a 1x1 pass without attachments.
fn drawn(
    queue: &CommandQueue,
    pipeline: &RenderPipelineState,
    draw: impl FnOnce(&mut RenderCommandEncoder),
) -> Vec<u32> {
    let device = queue.device();
    let words: Vec<u8> = (0..WORDS).flat_map(|w| (0x10 + w).to_le_bytes()).collect();
    let attributes = buffer_with(device, &words).unwrap();
    let out = buffer_with(device, &[0xffu8; FETCHES as usize * 8]).unwrap();
    let mut pass = RenderPassDescriptor::new();
    pass.render_target_width = 1;
    pass.render_target_height = 1;
    pass.default_raster_sample_count = 1;
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
    encoder.set_render_pipeline_state(pipeline).unwrap();
    encoder.set_vertex_buffer(Some(&out), 0, 0).unwrap();
    encoder.set_vertex_buffer(Some(&attributes), 0, 1).unwrap();
    draw(&mut encoder);
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    let read = read_buffer(&out, 0, u64::from(FETCHES) * 8).unwrap();
    // Each slot's first word: the attribute it read.
    (read.as_chunks::<8>().0.iter())
        .map(|slot| u32::from_le_bytes([slot[0], slot[1], slot[2], slot[3]]))
        .collect()
}

/// A buffer of `indices`, of `index_type`.
fn indices(device: &Device, index_type: IndexType, indices: &[u32]) -> Buffer {
    let bytes: Vec<u8> = match index_type {
        IndexType::UInt16 => (indices.iter())
            .flat_map(|&i| (i as u16).to_le_bytes())
            .collect(),
        _ => indices.iter().flat_map(|i| i.to_le_bytes()).collect(),
    };
    buffer_with(device, &bytes).unwrap()
}

/// The words a draw reads where each of its four elements reads zero.
const ZEROS: [u32; 4] = [0; 4];

/// The words a draw of elements 0 to 3 reads: words 0 to 3.
const CONTROL: [u32; 4] = [0x10, 0x11, 0x12, 0x13];

/// The kinds of the degrade notices `messages` took since it last took them, of [`MAY_WRAP`]'s
/// kind alone: a device may announce others at its first buffer or draw (a `Shared` buffer in
/// system memory).
fn may_wrap_notices(messages: &Messages) -> Vec<Degradation> {
    (messages.take_notices().iter())
        .filter_map(|message| message.degradation.as_ref())
        .map(|notice| notice.degradation)
        .filter(|kind| MAY_WRAP.contains(kind))
        .collect()
}

/// The words a draw of elements `FAR` to `FAR + 3` at stride 4 reads where the fetch wraps:
/// words 0 to 3, by the slot each element's ids sum to.
const WRAPPED: [u32; 4] = [0x10, 0x11, 0x12, 0x13];

/// Under either policy, a far element the draw's own values state reads zero on every device: a
/// vertex start, a 16-bit draw's base vertex, a base instance past `2^30`; a device with the
/// defect pulls the layout for each draw, and no other does. The controls read the buffer.
#[test]
fn a_draw_s_own_far_elements_read_zero_on_every_device() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    for policy in [VertexReachPolicy::Bounded, VertexReachPolicy::Exact] {
        let device = device_under(&messages, &cache, policy);
        let queue = device.new_command_queue().unwrap();
        let per_vertex = pipeline(&device, VertexStepFunction::PerVertex);
        let per_instance = pipeline(&device, VertexStepFunction::PerInstance);
        let short = indices(&device, IndexType::UInt16, &[0, 1, 2, 3]);
        let point = PrimitiveType::Point;
        let pulled = || device.work_counts().far_vertex_fetch_pulled_draws;
        let before = pulled();
        let control = drawn(&queue, &per_vertex, |e| {
            e.draw_primitives(point, 0, FETCHES, 1, 0).unwrap()
        });
        assert_eq!(control, CONTROL, "{policy:?}: the control");
        assert_eq!(pulled(), before, "{policy:?}: the control pulls nothing");
        let started = drawn(&queue, &per_vertex, |e| {
            e.draw_primitives(point, FAR, FETCHES, 1, 0).unwrap()
        });
        assert_eq!(started, ZEROS, "{policy:?}: a vertex start of 2^30");
        let based = drawn(&queue, &per_vertex, |e| {
            let (count, ty) = (FETCHES, IndexType::UInt16);
            e.draw_indexed_primitives(point, count, ty, &short, 0, 1, FAR as i32, 0)
                .unwrap()
        });
        assert_eq!(
            based, ZEROS,
            "{policy:?}: a 16-bit draw's base vertex of 2^30"
        );
        let instanced = drawn(&queue, &per_instance, |e| {
            e.draw_primitives(point, 0, 1, FETCHES, FAR).unwrap()
        });
        assert_eq!(instanced, ZEROS, "{policy:?}: a base instance of 2^30");
        let expected = if wraps(&device) { 3 } else { 0 };
        assert_eq!(pulled() - before, expected, "{policy:?}: {}", device.name());
        assert_eq!(may_wrap_notices(&messages), Vec::new(), "{policy:?}");
    }
    layer_reported_nothing_at_all(&messages);
}

/// 32-bit indices past `2^30`, and an indirect draw's vertex start past it, under
/// `VertexReachPolicy::Bounded`: a device with the defect reads each element's wrapped word and
/// announces [`MAY_WRAP`] once; every other device reads zero and announces nothing. Under
/// `VertexReachPolicy::Exact` every device reads zero, and one with the defect pulls the layout.
#[test]
fn gpu_memory_s_far_elements_wrap_where_the_device_s_fetch_wraps() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    for policy in [VertexReachPolicy::Bounded, VertexReachPolicy::Exact] {
        let device = device_under(&messages, &cache, policy);
        let queue = device.new_command_queue().unwrap();
        let per_vertex = pipeline(&device, VertexStepFunction::PerVertex);
        let far: Vec<u32> = (FAR..FAR + FETCHES).collect();
        let long = indices(&device, IndexType::UInt32, &far);
        let arguments: Vec<u8> = [FETCHES, 1, FAR, 0]
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect();
        let arguments = buffer_with(&device, &arguments).unwrap();
        let point = PrimitiveType::Point;
        let pulled = || device.work_counts().far_vertex_fetch_pulled_draws;
        let before = pulled();
        let indexed = drawn(&queue, &per_vertex, |e| {
            let (count, ty) = (FETCHES, IndexType::UInt32);
            e.draw_indexed_primitives(point, count, ty, &long, 0, 1, 0, 0)
                .unwrap()
        });
        let indirect = drawn(&queue, &per_vertex, |e| {
            e.draw_primitives_indirect(point, &arguments, 0).unwrap()
        });
        let wrapped = wraps(&device) && policy == VertexReachPolicy::Bounded;
        let expected = if wrapped { WRAPPED } else { ZEROS };
        let name = device.name().to_owned();
        assert_eq!(
            indexed, expected,
            "{policy:?}, {name}: 32-bit indices of 2^30"
        );
        assert_eq!(
            indirect, expected,
            "{policy:?}, {name}: an indirect vertex start"
        );
        let exact = wraps(&device) && policy == VertexReachPolicy::Exact;
        assert_eq!(pulled() - before, if exact { 2 } else { 0 }, "{policy:?}");
        let expected = if wrapped {
            MAY_WRAP.to_vec()
        } else {
            Vec::new()
        };
        assert_eq!(may_wrap_notices(&messages), expected, "{policy:?}, {name}");
    }
    layer_reported_nothing_at_all(&messages);
}

/// The vertices of each draw of [`far_reach_costs`]: 2^20 on a GPU.
#[cfg(m2v_gpu)]
const COST_VERTICES: u32 = 1 << 20;

/// The vertices of each draw of [`far_reach_costs`] on a CPU ICD, which shades them on the CPU.
#[cfg(not(m2v_gpu))]
const COST_VERTICES: u32 = 1 << 16;

/// What pulling a draw of 32-bit indices costs over fixing its fetch in function, without the
/// validation layer: under each policy, the pipelines the device makes for a state drawn first
/// without indices and then with 32-bit indices (`WorkCounts::variant_pipelines_made`), and the
/// passes of one such indexed draw of [`COST_VERTICES`] points (the owned fixture
/// `ob3_vertex_input`, each vertex writing the attribute it read), timed over 20 passes after 3
/// warm ones by the host's wall time from the encoder's making to the wait's return
/// (`FAR-REACH-COST`: the median and the fastest). On a device without the defect both policies
/// fetch in fixed function. Measures; asserts only that each pass completes.
#[test]
#[ignore = "a measurement, not a check: run by name with --run-ignored all"]
fn far_reach_costs() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    for policy in [VertexReachPolicy::Bounded, VertexReachPolicy::Exact] {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
            options.layers.clear();
            options.policy.vertex_reach = policy;
        })
        .unwrap();
        let bitcode = llvm_assemble(&engine_dir().join("ob3_vertex_input.ll")).unwrap();
        let library = device.new_library_with_air(&bitcode).unwrap();
        let function = library.new_function_with_name("ob3_vertex_input").unwrap();
        let mut descriptor =
            RenderPipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
        descriptor.rasterization_enabled = false;
        let mut vertex = VertexDescriptor::new();
        vertex
            .layouts
            .insert(0, VertexBufferLayoutDescriptor::new(Some(4)));
        vertex
            .attributes
            .insert(0, VertexAttributeDescriptor::new(VertexFormat::UInt, 0, 0));
        descriptor.vertex_descriptor = Some(vertex);
        let made = || device.work_counts().variant_pipelines_made;
        let before = made();
        let state = device
            .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
            .unwrap();
        let words: Vec<u8> = (0..COST_VERTICES).flat_map(|w| w.to_le_bytes()).collect();
        let attributes = buffer_with(&device, &words).unwrap();
        let out = buffer_with(&device, &vec![0u8; COST_VERTICES as usize * 4]).unwrap();
        let all: Vec<u32> = (0..COST_VERTICES).collect();
        let long = indices(&device, IndexType::UInt32, &all);
        let queue = device.new_command_queue().unwrap();
        let mut pass = RenderPassDescriptor::new();
        pass.render_target_width = 1;
        pass.render_target_height = 1;
        pass.default_raster_sample_count = 1;
        let run = |indexed: bool| {
            let began = Instant::now();
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
            encoder.set_render_pipeline_state(&state).unwrap();
            encoder.set_vertex_buffer(Some(&attributes), 0, 0).unwrap();
            encoder.set_vertex_buffer(Some(&out), 0, 1).unwrap();
            let point = PrimitiveType::Point;
            match indexed {
                true => encoder.draw_indexed_primitives(
                    point,
                    COST_VERTICES,
                    IndexType::UInt32,
                    &long,
                    0,
                    1,
                    0,
                    0,
                ),
                false => encoder.draw_primitives(point, 0, COST_VERTICES, 1, 0),
            }
            .unwrap();
            encoder.end_encoding().unwrap();
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
            began.elapsed().as_secs_f64() * 1e3
        };
        run(false);
        let unindexed = made() - before;
        let mut walls: Vec<f64> = (0..23).map(|_| run(true)).skip(3).collect();
        let pipelines = made() - before;
        walls.sort_by(f64::total_cmp);
        eprintln!(
            "FAR-REACH-COST {} {policy:?}: {unindexed} pipeline(s) made for the draw without \
             indices, {pipelines} once a draw of 32-bit indices followed; a pass of {COST_VERTICES} \
             points of 32-bit indices: median {:.3} ms, fastest {:.3} ms (host wall)",
            device.name(),
            walls[walls.len() / 2],
            walls[0]
        );
        messages.take_notices();
    }
}
