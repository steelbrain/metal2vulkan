//! A host step's `host-replayed-indirect-commands` notice names every rule of the GPU form
//! (`render/executions/gpu.rs`) that keeps its pass from it, not only the first
//! (`render/executions/cause.rs`, `DeferredPass::rules_past`); and each render pass over a
//! rasterization-rate map that a host step records without a probe is counted by why
//! (`WorkCounts::rate_map_steps_unfanned`, `rate_map_steps_nothing_to_probe`,
//! `rate_map_steps_probing_none`).
//!
//! An indirect command buffer whose records the GPU may write, of an indexed triangle and an empty
//! command, inheriting the encoder's pipeline state, is executed in a rate-mapped pass: alone, its
//! notice names the rate map and nothing past it, the rest of the GPU form's rules kept; with a
//! visibility result mode set, its notice names that rule first and the rate map past it. An
//! execution of the empty command alone probes no draw, counted so: read and probing none, or,
//! through its GPU route, with no command to probe.
//!
//! A device test: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`), on the kind
//! of device the cfg selects. The functions are `tob_triangle` and `tob_solid` of the owned
//! fixture `validation/fixtures/public/engine/tob_renders.ll` (`llvm-as`).

use super::blit_holds::{options, Scratch, WAIT};
use super::encoder_holds::library;
use crate::write_routes::GpuWriter;
use crate::{
    Buffer, CommandBufferStatus, CommandQueue, Degradation, DegradeNotice, Device,
    IndirectCommandBuffer, IndirectCommandBufferDescriptor, LoadAction, MessageSink,
    PipelineOptions, RasterizationRateMap, RenderPassDescriptor, RenderPipelineState,
    ResourceOptions, StoreAction, TextureDescriptor, VisibilityResultMode, WorkCounts,
};
use metal2vulkan::bind::{RateMapDescriptor, RateMapLayerDescriptor};
use metal2vulkan::device::VkFeature;
use metal2vulkan::metal::{
    IndexType, IndirectCommandType, PixelFormat, PrimitiveType, TextureUsage,
};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use std::sync::{Arc, Mutex};

/// The map's screen side.
const SCREEN: u32 = 64;

/// The rules the notices name, as `render/executions/gpu.rs` words them.
const RATE_MAP: &str = "the pass has a rasterization rate map";
const VISIBILITY: &str = "the encoder counts visibility results";
/// What a notice that names rules past its first says before them (`cause::named`).
const PAST: &str = "past that rule";

/// What the sink heard: each `host-replayed-indirect-commands` notice.
type Heard = Arc<Mutex<Vec<DegradeNotice>>>;

/// The device, its queue, the pipeline, the triangle's buffers, the indirect command buffer and
/// the map.
struct Workload {
    device: Device,
    queue: CommandQueue,
    pipeline: RenderPipelineState,
    _drawing: [Buffer; 3],
    icb: IndirectCommandBuffer,
    map: RasterizationRateMap,
    visibility: Buffer,
}

impl Workload {
    /// The workload on a device of the kind the cfg selects, its shader cache in `scratch`, its
    /// host-step notices into `heard`.
    fn new(scratch: &Scratch, heard: &Heard) -> Workload {
        let mut options = options(&scratch.0);
        let into = Arc::clone(heard);
        options.messages = Some(MessageSink::new(move |message| {
            if let Some(notice) = &message.degradation {
                if notice.degradation == Degradation::HostReplayedIndirectCommands {
                    into.lock().unwrap().push(notice.clone());
                }
            }
        }));
        let device = Device::system_default_with(&options).unwrap();
        let renders = library(&device, "engine/tob_renders.ll");
        let function = |name| {
            StageSource::Function(
                renders
                    .new_function_with_name(name)
                    .unwrap()
                    .function_descriptor(),
            )
        };
        let mut descriptor = RenderPipelineDescriptor::new(function("tob_triangle"));
        descriptor.fragment = Some(function("tob_solid"));
        descriptor.color_attachments[0] =
            Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
        descriptor.support_indirect_command_buffers = true;
        let pipeline = device
            .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
            .unwrap();
        let shared = ResourceOptions::shared();
        let buffer = |bytes: &[u8]| device.new_buffer_with_bytes(bytes, shared).unwrap();
        let corners: Vec<u8> = [[-0.8f32, -0.8], [0.8, -0.8], [0.0, 0.8]]
            .iter()
            .flatten()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let indices: Vec<u8> = [0u32, 1, 2].iter().flat_map(|i| i.to_le_bytes()).collect();
        let color: Vec<u8> = [0.25f32, 0.5, 0.75, 1.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let drawing = [buffer(&corners), buffer(&indices), buffer(&color)];
        let mut icb = IndirectCommandBufferDescriptor::new();
        icb.command_types = IndirectCommandType::DRAW_INDEXED;
        icb.inherit_pipeline_state = true;
        icb.inherit_buffers = false;
        icb.max_vertex_buffer_bind_count = 1;
        icb.max_fragment_buffer_bind_count = 1;
        let icb = device.new_indirect_command_buffer(&icb, 2, shared).unwrap();
        // Command 0 draws the triangle; command 1 stays empty.
        let command = icb.indirect_render_command_at(0).unwrap();
        command.set_vertex_buffer(Some(&drawing[0]), 0, 0).unwrap();
        command
            .set_fragment_buffer(Some(&drawing[2]), 0, 0)
            .unwrap();
        command
            .draw_indexed_primitives(
                PrimitiveType::Triangle,
                3,
                IndexType::UInt32,
                &drawing[1],
                0,
                1,
                0,
                0,
            )
            .unwrap();
        // The GPU may write the records: the host step records the pass.
        icb.written_by_gpu(GpuWriter::Reset);
        drop(icb.gpu_write());
        let mut map = RateMapDescriptor::new([SCREEN, SCREEN]);
        map.layers = vec![RateMapLayerDescriptor::new(
            vec![0.5, 1.0, 0.5],
            vec![0.5, 1.0, 0.5],
        )];
        let map = device.new_rasterization_rate_map(&map).unwrap();
        Workload {
            queue: device.new_command_queue().unwrap(),
            pipeline,
            _drawing: drawing,
            icb,
            map,
            visibility: buffer(&[0; 8]),
            device,
        }
    }

    /// One pass over the map executing `range` of the buffer's commands under the encoder's
    /// pipeline state, counting visibility results where `counted`: the device's counts before and
    /// after.
    fn frame(&self, range: std::ops::Range<u64>, counted: bool) -> (WorkCounts, WorkCounts) {
        let size = self.map.physical_size_for_layer(0);
        let mut d =
            TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, size.width, size.height, false);
        d.usage = TextureUsage::RENDER_TARGET;
        let target = self.device.new_texture(&d).unwrap();
        let before = self.device.work_counts();
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut pass = RenderPassDescriptor::new();
        let color = &mut pass.color_attachments[0];
        color.texture = Some(target);
        color.load_action = LoadAction::Clear;
        color.store_action = StoreAction::Store;
        pass.rasterization_rate_map = Some(self.map.clone());
        pass.visibility_result_buffer = Some(self.visibility.clone());
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encoder.set_render_pipeline_state(&self.pipeline).unwrap();
        if counted {
            (encoder.set_visibility_result_mode(VisibilityResultMode::Boolean, 0)).unwrap();
        }
        encoder
            .execute_commands_in_buffer(&self.icb, range)
            .unwrap();
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        (before, self.device.work_counts())
    }
}

/// Whether `device` fans a rate-mapped pass's triangles out at a host step
/// (`render/fan_out.rs`'s `fans_out`, under the default policy).
fn fans_out(device: &Device) -> bool {
    let features = &device.compile_device().profile().enabled().features;
    [
        VkFeature::GeometryShader,
        VkFeature::ShaderClipDistance,
        VkFeature::VertexPipelineStoresAndAtomics,
    ]
    .iter()
    .all(|feature| features.contains(feature))
}

/// The probes and the rate-mapped passes recorded unfanned, with nothing to probe, and probing
/// none, between two counts.
fn probe_counts(before: &WorkCounts, after: &WorkCounts) -> [u64; 4] {
    [
        after.rate_map_probes - before.rate_map_probes,
        after.rate_map_steps_unfanned - before.rate_map_steps_unfanned,
        after.rate_map_steps_nothing_to_probe - before.rate_map_steps_nothing_to_probe,
        after.rate_map_steps_probing_none - before.rate_map_steps_probing_none,
    ]
}

/// The notice of a pass whose only broken rule is its rate map names that rule alone; one whose
/// visibility result mode breaks a rule before it names that rule, then the rate map past it.
/// Each host-stepped pass over the map is counted once: probed, or why not.
#[test]
fn a_host_step_notice_names_every_rule_its_pass_breaks() {
    let scratch = Scratch::new("host-step-rules");
    let heard = Heard::default();
    let workload = Workload::new(&scratch, &heard);
    let fans = fans_out(&workload.device);
    // Probed (where the device fans out) or unfanned.
    let one = match fans {
        true => [1, 0, 0, 0],
        false => [0, 1, 0, 0],
    };
    let (before, after) = workload.frame(0..1, false);
    assert_eq!(probe_counts(&before, &after), one, "the triangle alone");
    let (before, after) = workload.frame(0..1, true);
    assert_eq!(probe_counts(&before, &after), one, "counting visibility");
    // The empty command alone: read, and no draw probed; through its GPU route, where it takes
    // one (`render/executions/gpu_route.rs`), no command that draws; or unfanned.
    let (before, after) = workload.frame(1..2, false);
    let routed = after.rate_map_gpu_routes - before.rate_map_gpu_routes;
    let nothing = match (fans, routed) {
        (true, 0) => [0, 0, 0, 1],
        (true, _) => [0, 0, 1, 0],
        (false, _) => [0, 1, 0, 0],
    };
    assert_eq!(probe_counts(&before, &after), nothing, "the empty command");

    let heard = heard.lock().unwrap();
    let reasons: Vec<&str> = heard.iter().map(|notice| notice.reason.as_str()).collect();
    eprintln!("HOST-STEP-RULES {reasons:#?}");
    // One notice per first rule: the rate map's, then the visibility mode's.
    assert_eq!(reasons.len(), 2, "{reasons:#?}");
    let map = reasons[0];
    assert!(
        map.contains(&format!("as {RATE_MAP}")) && !map.contains(PAST),
        "the rate map's notice names a rule past it: {map}"
    );
    let counted = reasons[1];
    let past = counted.find(PAST);
    let (first, rest) = counted.split_at(past.unwrap_or(counted.len()));
    assert!(
        first.contains(&format!("as {VISIBILITY}")) && rest.contains(RATE_MAP),
        "the visibility mode's notice names the rate map past it: {counted}"
    );
    assert!(
        !rest.contains("refused"),
        "a rule past the rate map's refused: {counted}"
    );
}
