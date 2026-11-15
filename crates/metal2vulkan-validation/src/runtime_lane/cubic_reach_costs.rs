//! What the cubic reach test costs a function that samples through an argument buffer's sampler, on
//! the device the cfg selects, without the validation layer: report only, to weigh the always-on
//! test against its alternatives. Each sample through an argument buffer's sampler tests the
//! sampler's table entry for a cubic filter and, in a branch a run that names no cubic sampler
//! never takes, ORs a bit into the command buffer's `ExecutionStatus` word. The branch is a store
//! in the function: a fragment function holding one may lose the device's early depth test or
//! hidden-surface removal although it never runs, so the fragment cases draw layers the depth test
//! or a later opaque layer hides.
//!
//! Every case's texture is a 256x256 `RGBA8Unorm` one sampled linearly, its texels noise; each
//! case runs with no cubic sampler live:
//!
//! - **The kernel.** A synthetic kernel sampling the texture 16 times per thread through its
//!   argument buffer's sampler, over 2^20 threads, summing the samples, dispatched 16 times.
//! - **The fragment layers.** `tail_argument_samplers`' `argument_sampler_fragment` (one sample
//!   through its argument buffer's sampler) drawn by `coverage_loop`'s `cov_cover_uv` (the
//!   fullscreen triangle at depth 0) into a 2048x2048 `RGBA8Unorm` target: once; 16 times over a
//!   `Depth32Float` attachment cleared to 1.0 under `Less` with writes (the first layer hides the
//!   other 15); and 16 times without depth (each layer hides the one before).
//! - **The control.** The same layers of `tob_renders`' `tob_sample` (one sample through a slot's
//!   texture and sampler, no store) drawn by `rt6_resolves`' `rt6_resolve_full`.
//!
//! Each case times one command buffer [`RUNS`] times after one untimed run: its GPU start to end
//! and the host's wall time from its commit until its wait returns. Printed per case: the
//! medians and the fastest runs. Run at two commits (before and after the test) on one device,
//! the cases compare the test's cost; within one run, the fragment layers' ratio to one layer,
//! beside the control's, shows whether the device still hides the hidden layers.
//!
//! Ignored by default: a measurement, not a check. Run it by name with `--run-ignored all`.

use super::{buffer_with, validated_device_with, Messages, ScratchCache, WAIT};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::metal::state::DepthStencilDescriptor;
use metal2vulkan::metal::{
    CompareFunction, PixelFormat, PrimitiveType, RenderStages, ResourceUsage, SamplerMinMagFilter,
    TextureUsage,
};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, ComputePipelineDescriptor, RenderPipelineDescriptor, StageSource,
};
use metal2vulkan_devtools::{llvm_assemble, llvm_assemble_text};
use metal2vulkan_runtime::{
    Buffer, CommandBuffer, CommandQueue, ComputePipelineState, DepthStencilState, Device, Library,
    LoadAction, PipelineOptions, Region, RenderPassDescriptor, RenderPipelineState,
    ResourceOptions, SamplerState, Size, StoreAction, Texture, TextureDescriptor,
};
use std::time::Instant;

/// The timed runs of each case.
const RUNS: usize = 9;

/// The sampled texture's side.
const SIDE: u64 = 256;

/// The fragment target's side.
const TARGET: u64 = 2048;

/// The layers a hiding case draws.
const LAYERS: usize = 16;

/// The kernel's threads and samples per thread, and its dispatches per command buffer (enough
/// work that the command buffer's latency on the host is a small part of its wall time).
const THREADS: u64 = 1 << 20;
const SAMPLES: usize = 16;
const DISPATCHES: usize = 16;

/// The engine fixture `stem`'s library on `device`.
fn library(device: &Device, stem: &str) -> Library {
    let ll = metal2vulkan_devtools::engine_dir().join(format!("{stem}.ll"));
    device
        .new_library_with_air(&llvm_assemble(&ll).unwrap())
        .unwrap()
}

/// Synthetic AIR written for this test: kernel `k` samples the `texture2d<float>` its argument
/// buffer (buffer 2, `{ texture2d<float> t; sampler s; }`) holds through the sampler beside it,
/// [`SAMPLES`] times at level 0, at thread `gid`'s point of a 1024-wide grid over the unit square
/// moved by `j / 128` and `j / 256` for sample `j`, and writes the samples' sum to buffer 1's lane
/// `gid`.
fn kernel() -> String {
    let mut body = String::new();
    for j in 0..SAMPLES {
        let (dx, dy) = (j as f64 / 128.0, j as f64 / 256.0);
        body += &format!(
            "  %uv{j} = fadd <2 x float> %base, <float {dx:?}, float {dy:?}>\n  %r{j} = call {{ <4 x float>, i8 }} @air.sample_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %s, <2 x float> %uv{j}, i1 true, <2 x i32> zeroinitializer, i1 true, float 0.0, float 0.0, i32 0)\n  %v{j} = extractvalue {{ <4 x float>, i8 }} %r{j}, 0\n"
        );
        let sum = match j {
            0 => "zeroinitializer".to_owned(),
            _ => format!("%a{}", j - 1),
        };
        body += &format!("  %a{j} = fadd <4 x float> {sum}, %v{j}\n");
    }
    let last = SAMPLES - 1;
    format!(
        r#"%struct.M = type {{ ptr addrspace(1), ptr addrspace(2) }}

define void @k(ptr addrspace(1) %out, ptr addrspace(2) %m, i32 %gid) {{
  %i = zext i32 %gid to i64
  %xi = and i32 %gid, 1023
  %yi = lshr i32 %gid, 10
  %xf = call float @air.convert.f.f32.u.i32(i32 %xi)
  %yf = call float @air.convert.f.f32.u.i32(i32 %yi)
  %x = fmul float %xf, 0.0009765625
  %y = fmul float %yf, 0.0009765625
  %b0 = insertelement <2 x float> undef, float %x, i64 0
  %base = insertelement <2 x float> %b0, float %y, i64 1
  %pt = getelementptr inbounds %struct.M, ptr addrspace(2) %m, i64 0, i32 0
  %t = load ptr addrspace(1), ptr addrspace(2) %pt, align 8
  %ps = getelementptr inbounds %struct.M, ptr addrspace(2) %m, i64 0, i32 1
  %s = load ptr addrspace(2), ptr addrspace(2) %ps, align 8
{body}  %q = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i64 %i
  store <4 x float> %a{last}, ptr addrspace(1) %q, align 16
  ret void
}}

declare float @air.convert.f.f32.u.i32(i32)
declare {{ <4 x float>, i8 }} @air.sample_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)

!air.kernel = !{{!0}}
!air.version = !{{!20}}
!air.language_version = !{{!21}}
!0 = !{{ptr @k, !1, !2}}
!1 = !{{}}
!2 = !{{!4, !5, !7}}
!4 = !{{i32 0, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"out"}}
!5 = !{{i32 1, !"air.indirect_buffer", !"air.buffer_size", i32 16, !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !8, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"M", !"air.arg_name", !"m"}}
!8 = !{{i32 0, i32 8, i32 0, !"texture2d<float, sample>", !"t", !"air.indirect_argument", !9, i32 8, i32 8, i32 0, !"sampler", !"s", !"air.indirect_argument", !10}}
!9 = !{{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"t"}}
!10 = !{{i32 1, !"air.sampler", !"air.location_index", i32 1, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"s"}}
!7 = !{{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}}
!20 = !{{i32 2, i32 8, i32 0}}
!21 = !{{!"Metal", i32 4, i32 0, i32 0}}
"#
    )
}

/// The pipelines, textures and state the cases use, made on a device the caller holds.
struct Kit {
    queue: CommandQueue,
    kernel: ComputePipelineState,
    /// `argument_sampler_fragment` drawn by `cov_cover_uv`, with and without depth.
    argument: RenderPipelineState,
    argument_depth: RenderPipelineState,
    /// `tob_sample` drawn by `rt6_resolve_full`, with and without depth.
    slot: RenderPipelineState,
    slot_depth: RenderPipelineState,
    less: DepthStencilState,
    sampled: Texture,
    sampler: SamplerState,
    /// The argument buffer: the texture's and the sampler's resource IDs.
    argument_buffer: Buffer,
    /// `rt6_resolve_full`'s depth.
    z: Buffer,
    target: Texture,
    depth: Texture,
    out: Buffer,
}

impl Kit {
    fn new(device: &Device) -> Kit {
        let samplers = library(device, "tail_argument_samplers");
        let coverage = library(device, "coverage_loop");
        let resolves = library(device, "rt6_resolves");
        let renders = library(device, "tob_renders");
        let function = |library: &Library, name: &str| {
            StageSource::Function(
                library
                    .new_function_with_name(name)
                    .unwrap()
                    .function_descriptor(),
            )
        };
        let pipeline = |vertex: StageSource, fragment: StageSource, depth: bool| {
            let mut descriptor = RenderPipelineDescriptor::new(vertex);
            descriptor.fragment = Some(fragment);
            descriptor.color_attachments[0] =
                Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
            if depth {
                descriptor.depth_format = Some(PixelFormat::Depth32Float);
            }
            device
                .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
                .unwrap()
        };
        let argument = |depth| {
            pipeline(
                function(&coverage, "cov_cover_uv"),
                function(&samplers, "argument_sampler_fragment"),
                depth,
            )
        };
        let slot = |depth| {
            pipeline(
                function(&resolves, "rt6_resolve_full"),
                function(&renders, "tob_sample"),
                depth,
            )
        };
        let bitcode = llvm_assemble_text(&kernel()).unwrap();
        let k = device
            .new_library_with_air(&bitcode)
            .unwrap()
            .new_function_with_name("k")
            .unwrap();
        let kernel = device
            .new_compute_pipeline_state(
                &ComputePipelineDescriptor::new(StageSource::Function(k.function_descriptor())),
                &PipelineOptions::default(),
            )
            .unwrap();
        let mut less = DepthStencilDescriptor::default();
        less.depth_compare_function = CompareFunction::Less;
        less.depth_write_enabled = true;
        let mut descriptor =
            TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
        descriptor.usage = TextureUsage::SHADER_READ;
        descriptor.resource_options = ResourceOptions::shared();
        let sampled = device.new_texture(&descriptor).unwrap();
        let mut noise = 0x6332_u64;
        let texels: Vec<u8> = (0..SIDE * SIDE * 4)
            .map(|_| {
                noise = noise
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                (noise >> 56) as u8
            })
            .collect();
        sampled
            .replace_region(Region::new_2d(0, 0, SIDE, SIDE), 0, 0, &texels, SIDE * 4, 0)
            .unwrap();
        let mut linear = SamplerDescriptor::new();
        linear.min_filter = SamplerMinMagFilter::Linear;
        linear.mag_filter = SamplerMinMagFilter::Linear;
        let sampler = device.new_sampler_state(&linear).unwrap();
        let ids = [sampled.gpu_resource_id().0, sampler.gpu_resource_id().0];
        let words: Vec<u8> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
        let argument_buffer = buffer_with(device, &words).unwrap();
        let attachment = |format: PixelFormat| {
            let mut d = TextureDescriptor::texture_2d(format, TARGET, TARGET, false);
            d.usage = TextureUsage::RENDER_TARGET;
            d.resource_options = ResourceOptions::private();
            device.new_texture(&d).unwrap()
        };
        Kit {
            queue: device.new_command_queue().unwrap(),
            kernel,
            argument: argument(false),
            argument_depth: argument(true),
            slot: slot(false),
            slot_depth: slot(true),
            less: device.new_depth_stencil_state(&less).unwrap(),
            argument_buffer,
            z: buffer_with(device, &0f32.to_le_bytes()).unwrap(),
            target: attachment(PixelFormat::RGBA8Unorm),
            depth: attachment(PixelFormat::Depth32Float),
            out: device
                .new_buffer(THREADS * 16, ResourceOptions::private())
                .unwrap(),
            sampled,
            sampler,
        }
    }
}

/// What one case encodes into its command buffer.
type Encode<'a> = dyn Fn(&Kit, &CommandBuffer) + 'a;

/// The kernel over [`THREADS`] threads, [`DISPATCHES`] times.
fn dispatch() -> Box<Encode<'static>> {
    Box::new(|kit: &Kit, command_buffer: &CommandBuffer| {
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(&kit.kernel).unwrap();
        encoder
            .use_resource(kit.sampled.as_resource(), ResourceUsage::READ)
            .unwrap();
        encoder.set_buffer(Some(&kit.out), 0, 1).unwrap();
        encoder
            .set_buffer(Some(&kit.argument_buffer), 0, 2)
            .unwrap();
        for _ in 0..DISPATCHES {
            encoder
                .dispatch_threads(Size::new(THREADS, 1, 1), Size::new(256, 1, 1))
                .unwrap();
        }
        encoder.end_encoding().unwrap();
    })
}

/// One pass into the target drawing `layers` fullscreen triangles: through the argument
/// buffer's sampler, else (`slot`) through a slot's, over the depth attachment where `depth`.
fn layers(slot: bool, depth: bool, layers: usize) -> Box<Encode<'static>> {
    Box::new(move |kit: &Kit, command_buffer: &CommandBuffer| {
        let mut descriptor = RenderPassDescriptor::new();
        let target = &mut descriptor.color_attachments[0];
        target.texture = Some(kit.target.clone());
        target.load_action = LoadAction::Clear;
        target.store_action = StoreAction::Store;
        if depth {
            descriptor.depth_attachment.texture = Some(kit.depth.clone());
            descriptor.depth_attachment.load_action = LoadAction::Clear;
            descriptor.depth_attachment.clear_depth = 1.0;
            descriptor.depth_attachment.store_action = StoreAction::DontCare;
        }
        let mut encoder = command_buffer.render_command_encoder(&descriptor).unwrap();
        let pipeline = match (slot, depth) {
            (false, false) => &kit.argument,
            (false, true) => &kit.argument_depth,
            (true, false) => &kit.slot,
            (true, true) => &kit.slot_depth,
        };
        encoder.set_render_pipeline_state(pipeline).unwrap();
        if depth {
            encoder.set_depth_stencil_state(&kit.less).unwrap();
        }
        if slot {
            encoder.set_vertex_buffer(Some(&kit.z), 0, 0).unwrap();
            encoder.set_fragment_texture(Some(&kit.sampled), 0).unwrap();
            encoder
                .set_fragment_sampler_state(Some(&kit.sampler), 0)
                .unwrap();
        } else {
            encoder
                .use_resource(
                    kit.sampled.as_resource(),
                    ResourceUsage::READ,
                    RenderStages::FRAGMENT,
                )
                .unwrap();
            encoder
                .set_fragment_buffer(Some(&kit.argument_buffer), 0, 0)
                .unwrap();
        }
        for _ in 0..layers {
            encoder
                .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
                .unwrap();
        }
        encoder.end_encoding().unwrap();
    })
}

/// One timed run's GPU and wall times, in microseconds.
struct Times {
    gpu: f64,
    wall: f64,
}

/// The times of one command buffer of `encode`.
fn run(kit: &Kit, encode: &Encode<'_>) -> Times {
    let command_buffer = kit.queue.command_buffer().unwrap();
    encode(kit, &command_buffer);
    let committed = Instant::now();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    let wall = committed.elapsed().as_secs_f64() * 1e6;
    assert_eq!(command_buffer.error(), None);
    let start = command_buffer.gpu_start_time().unwrap();
    let end = command_buffer.gpu_end_time().unwrap();
    Times {
        gpu: (end - start) * 1e6,
        wall,
    }
}

/// The median and the smallest of `values`.
fn median_and_least(mut values: Vec<f64>) -> (f64, f64) {
    values.sort_by(f64::total_cmp);
    (values[values.len() / 2], values[0])
}

/// Every case's times, printed (the module docs).
#[test]
#[ignore = "a measurement, not a check: run by name with --run-ignored all"]
fn cubic_reach_costs() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
        options.layers.clear();
    })
    .unwrap();
    messages.take_notices();
    let name = device.name().to_owned();
    let kit = Kit::new(&device);
    let cases: Vec<(String, Box<Encode<'_>>)> = vec![
        (
            format!(
                "kernel: {SAMPLES} argument samples x {THREADS} threads x {DISPATCHES} dispatches"
            ),
            dispatch(),
        ),
        ("argument fragment: 1 layer".into(), layers(false, false, 1)),
        (
            format!("argument fragment: {LAYERS} layers under depth Less"),
            layers(false, true, LAYERS),
        ),
        (
            format!("argument fragment: {LAYERS} opaque layers, no depth"),
            layers(false, false, LAYERS),
        ),
        ("slot fragment: 1 layer".into(), layers(true, false, 1)),
        (
            format!("slot fragment: {LAYERS} layers under depth Less"),
            layers(true, true, LAYERS),
        ),
        (
            format!("slot fragment: {LAYERS} opaque layers, no depth"),
            layers(true, false, LAYERS),
        ),
    ];
    for (case, encode) in &cases {
        let _ = run(&kit, encode.as_ref());
        let runs: Vec<Times> = (0..RUNS).map(|_| run(&kit, encode.as_ref())).collect();
        let (gpu, gpu_least) = median_and_least(runs.iter().map(|r| r.gpu).collect());
        let (wall, wall_least) = median_and_least(runs.iter().map(|r| r.wall).collect());
        eprintln!(
            "CUBIC-REACH-COST on {name}: {case}: GPU median {gpu:.1} us, fastest \
             {gpu_least:.1} us; wall median {wall:.1} us, fastest {wall_least:.1} us"
        );
    }
}
