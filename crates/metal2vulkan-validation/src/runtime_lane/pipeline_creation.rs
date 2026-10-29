//! What making a render state's `VkPipeline`s costs on the device the cfg selects, without the
//! validation layer: report only, in release builds, before and after a change to when and how
//! concurrently the runtime makes them.
//!
//! The pipeline is shaped as the consumer's slow first draws are (macOS guests' compositing
//! fragments, never their bytes): a fragment function of our own that samples twelve
//! `texture2d<half>` slots through two runtime samplers, [`TAPS`] times each, so its translation
//! specializes on each sampler's word (`SamplerState`) and reads each slot's bound format class
//! (`TexelFormat`) from a push field, as theirs do. A draw binds every slot to one texture of a
//! variant's format ([`VARIANTS`], four format classes); the four share one `VkPipeline`, where
//! each was its own when the class was specialized. Every run's functions take a fresh constant
//! (`nonce`), so no translation, Mesa or Metal cache answers for another run; a timing run also
//! turns Metal's and Mesa's disk caches off. Printed as `PPC-CREATE` lines:
//!
//! - **serial**: one thread makes the state, then one command buffer of one pass draws the four
//!   variants: each draw call's time, `end_encoding`, `commit`, the wait for completion, and each
//!   `VkPipeline`'s creation as the execution record states it.
//! - **threads**: the state made once, then four threads each draw one variant in a command
//!   buffer of its own, starting together: each thread's draw call, and the wall time until the
//!   last completed (the driver's own concurrency for pipelines of one shader module).
//! - **modules**: four threads each make a state of its own nonce (four shader modules, so four
//!   libraries on MoltenVK) and draw one variant: the same times.
//!
//! Ignored by default: a measurement, not a check. Run it by name with `--run-ignored all`.
//!
//! probes beside it, on the same fragment, printed as `PCC-` lines:
//!
//! - [`pipelines_per_format_class`]: one state's draws of [`FAMILY`], three formats of one
//!   family and one of another, in one command buffer: the `VkPipeline`s made (the makings the
//!   draws handed to the pool, and the distinct plans the draws executed).
//! - [`pool_against_serial`]: [`CLASSES`]' ten formats, each its own `VkPipeline`, drawn by a
//!   fresh state's one command buffer (the pool makes them as the draws go on), against a fresh
//!   state's ten command buffers, each committed and waited for before the next is encoded (one
//!   making at a time): each one's wall time from the first encoding to the last completion.
//! - [`state_makings_scale_with_the_threads`]: fresh states made and drawn on one thread and on as
//!   many threads at once, for counts up to the host's hardware threads, printed as `MKS` lines:
//!   whether makings run side by side.
//! - [`pipelines_of_this_build`]: [`CLASSES`]' ten draws by a state of this test binary's nonce,
//!   which another process of the same binary makes alike: run twice, in two processes, with
//!   Metal's shader cache on, the second's times are those of a second boot of one build.
//!
//! The check beside it: draws whose `VkPipeline`s the runtime makes on its pool while the draws go
//! on render, and count in a visibility query, exactly what the same draws do once those pipelines
//! are made, and their command buffer runs before one committed after it on its queue
//! ([`draws_made_on_the_pool_draw_as_made_ones`]).

use super::executions::KERNEL;
use super::{
    buffer_with, layer_reported_nothing_at_all, pipeline_state, public_case, read_buffer,
    validated_device, validated_device_with, Messages, ScratchCache, WAIT,
};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::metal::{PixelFormat, PrimitiveType, SamplerMinMagFilter, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::llvm_assemble_text;
use metal2vulkan_runtime::{
    CommandBuffer, CommandBufferStatus, CommandQueue, Device, ExecutedPipeline, LoadAction,
    PipelineOptions, Region, RenderPassDescriptor, RenderPipelineState, ResourceOptions,
    SamplerState, StoreAction, Texture, TextureDescriptor, Viewport, VisibilityResultMode,
};
use std::fmt::Write as _;
use std::sync::{Arc, Barrier, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// The fragment function's texture slots and samplers.
const SLOTS: usize = 12;
const SAMPLERS: usize = 2;
/// The samples each slot takes, per run size.
const TAPS: [usize; 2] = [2, 8];
/// The variants' formats: four `TexelFormat` classes whose `half` conversions differ on every
/// device (32-bit float is left out: not every device filters it; the half-float family is too:
/// where the device samples through Metal's own texture unit it converts as the 8-bit unsigned
/// normalized one).
const VARIANTS: [PixelFormat; 4] = [
    PixelFormat::RGBA8Unorm,
    PixelFormat::RGBA16Snorm,
    PixelFormat::RGBA16Unorm,
    PixelFormat::RGBA8Snorm,
];
/// The target's side, `RGBA16Float`.
const SIDE: u64 = 64;

/// LLVM's spelling of a `half` constant.
fn half(x: f32) -> String {
    format!("0xH{:04X}", half_bits(x))
}

/// The `half` nearest `x` (the probe's weights are exact halves).
fn half_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mantissa = bits & 0x7f_ffff;
    assert!((1..31).contains(&exponent), "a normal half weight");
    sign | ((exponent as u16) << 10) | ((mantissa >> 13) as u16)
}

/// LLVM's spelling of a `float` constant: the double of the float's value, in hex.
fn float(x: f32) -> String {
    format!("0x{:016X}", f64::from(x).to_bits())
}

/// The AIR of the probe's vertex function `ppc_vertex` (a fullscreen triangle with its
/// coordinates as `uv`) and fragment function `ppc_fragment`, whose sample offsets carry `nonce`.
fn air(nonce: u32, taps: usize) -> String {
    let mut s = String::new();
    s.push_str(
        "define <{ <4 x float>, <2 x float> }> @ppc_vertex(i32 noundef %vid) {
  %a = shl i32 %vid, 1
  %b = and i32 %a, 2
  %c = and i32 %vid, 2
  %u = tail call fast float @air.convert.f.f32.u.i32(i32 %b)
  %v = tail call fast float @air.convert.f.f32.u.i32(i32 %c)
  %x = fmul fast float %u, 2.000000e+00
  %x1 = fadd fast float %x, -1.000000e+00
  %y = fmul fast float %v, 2.000000e+00
  %y1 = fadd fast float %y, -1.000000e+00
  %p0 = insertelement <4 x float> <float poison, float poison, float 0.000000e+00, float 1.000000e+00>, float %x1, i64 0
  %p1 = insertelement <4 x float> %p0, float %y1, i64 1
  %t0 = insertelement <2 x float> poison, float %u, i64 0
  %t1 = insertelement <2 x float> %t0, float %v, i64 1
  %o0 = insertvalue <{ <4 x float>, <2 x float> }> undef, <4 x float> %p1, 0
  %o1 = insertvalue <{ <4 x float>, <2 x float> }> %o0, <2 x float> %t1, 1
  ret <{ <4 x float>, <2 x float> }> %o1
}

",
    );
    let mut args = vec![
        "<4 x float> %position".to_owned(),
        "<2 x float> %uv".to_owned(),
    ];
    args.extend((0..SLOTS).map(|t| format!("ptr addrspace(1) %t{t}")));
    args.extend((0..SAMPLERS).map(|j| format!("ptr addrspace(2) %s{j}")));
    writeln!(s, "define <4 x half> @ppc_fragment({}) {{", args.join(", ")).unwrap();
    let mut acc = "zeroinitializer".to_owned();
    for slot in 0..SLOTS {
        let mut sum = "zeroinitializer".to_owned();
        for tap in 0..taps {
            let n = format!("{slot}_{tap}");
            // A multiple of 2^-20 below 2^-8, exact beside the offsets: each of 4096 nonces
            // in a row is its own constant (a nonce scaled whole was lost to `f32`'s rounding at
            // the clock's magnitude, so consecutive states shared their functions).
            let jitter = (nonce % 4096) as f32 * 2f32.powi(-20);
            let ox = 0.0013 * (tap + 1) as f32 + 0.0007 * slot as f32 + jitter;
            let oy = 0.0011 * (tap + 1) as f32 - 0.0005 * slot as f32 + jitter;
            let w = half(1.0 / (tap + 2) as f32);
            let sampler = slot % SAMPLERS;
            writeln!(
                s,
                "  %c{n} = fadd fast <2 x float> %uv, <float {}, float {}>
  %r{n} = tail call {{ <4 x half>, i8 }} @air.sample_texture_2d.v4f16(ptr addrspace(1) %t{slot}, ptr addrspace(2) %s{sampler}, <2 x float> %c{n}, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %v{n} = extractvalue {{ <4 x half>, i8 }} %r{n}, 0
  %m{n} = fmul fast <4 x half> %v{n}, <half {w}, half {w}, half {w}, half {w}>
  %q{n} = fadd fast <4 x half> %m{n}, {sum}",
                float(ox),
                float(oy)
            )
            .unwrap();
            sum = format!("%q{n}");
        }
        writeln!(
            s,
            "  %h{slot} = fmul fast <4 x half> {sum}, {sum}
  %k{slot} = fadd fast <4 x half> %h{slot}, {acc}"
        )
        .unwrap();
        acc = format!("%k{slot}");
    }
    writeln!(s, "  ret <4 x half> {acc}\n}}\n").unwrap();
    s.push_str(
        "declare float @air.convert.f.f32.u.i32(i32) local_unnamed_addr
declare { <4 x half>, i8 } @air.sample_texture_2d.v4f16(ptr addrspace(1) readonly captures(none), ptr addrspace(2) readonly captures(none), <2 x float>, i1, <2 x i32>, i1, float, float, i32) local_unnamed_addr #0

attributes #0 = { convergent mustprogress nofree nounwind willreturn memory(read) }

!air.vertex = !{!0}
!air.fragment = !{!6}
!air.compile_options = !{!90, !91, !92}
!air.version = !{!93}
!air.language_version = !{!94}
!0 = !{ptr @ppc_vertex, !1, !4}
!1 = !{!2, !3}
!2 = !{!\"air.position\", !\"air.arg_type_name\", !\"float4\", !\"air.arg_name\", !\"position\"}
!3 = !{!\"air.vertex_output\", !\"generated(2uvDv2_f)\", !\"air.arg_type_name\", !\"float2\", !\"air.arg_name\", !\"uv\"}
!4 = !{!5}
!5 = !{i32 0, !\"air.vertex_id\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"vid\"}
!6 = !{ptr @ppc_fragment, !7, !9}
!7 = !{!8}
!8 = !{!\"air.render_target\", i32 0, i32 0, !\"air.arg_type_name\", !\"half4\", !\"air.arg_name\", !\"color\"}
",
    );
    let ids: Vec<String> = (10..12 + SLOTS + SAMPLERS)
        .map(|i| format!("!{i}"))
        .collect();
    writeln!(s, "!9 = !{{{}}}", ids.join(", ")).unwrap();
    s.push_str(
        "!10 = !{i32 0, !\"air.position\", !\"air.center\", !\"air.no_perspective\", !\"air.arg_type_name\", !\"float4\", !\"air.arg_name\", !\"position\", !\"air.arg_unused\"}
!11 = !{i32 1, !\"air.fragment_input\", !\"generated(2uvDv2_f)\", !\"air.center\", !\"air.perspective\", !\"air.arg_type_name\", !\"float2\", !\"air.arg_name\", !\"uv\"}
",
    );
    for t in 0..SLOTS {
        writeln!(
            s,
            "!{} = !{{i32 {}, !\"air.texture\", !\"air.location_index\", i32 {t}, i32 1, !\"air.sample\", !\"air.arg_type_name\", !\"texture2d<half, sample>\", !\"air.arg_name\", !\"t{t}\"}}",
            12 + t,
            2 + t
        )
        .unwrap();
    }
    for j in 0..SAMPLERS {
        writeln!(
            s,
            "!{} = !{{i32 {}, !\"air.sampler\", !\"air.location_index\", i32 {j}, i32 1, !\"air.arg_type_name\", !\"sampler\", !\"air.arg_name\", !\"s{j}\"}}",
            12 + SLOTS + j,
            2 + SLOTS + j
        )
        .unwrap();
    }
    s.push_str(
        "!90 = !{!\"air.compile.denorms_disable\"}
!91 = !{!\"air.compile.fast_math_enable\"}
!92 = !{!\"air.compile.framebuffer_fetch_enable\"}
!93 = !{i32 2, i32 8, i32 0}
!94 = !{!\"Metal\", i32 4, i32 0, i32 0}
",
    );
    s
}

/// Milliseconds of `d`.
fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

/// A list of milliseconds, printed.
fn list(values: &[f64]) -> String {
    let each: Vec<String> = values.iter().map(|v| format!("{v:.0}")).collect();
    format!("[{}]", each.join(","))
}

/// The probe's state descriptor on `device`: its functions of `nonce`, each slot sampled `taps`
/// times, into one `RGBA16Float` target.
fn descriptor(device: &Device, nonce: u32, taps: usize) -> RenderPipelineDescriptor {
    let bytes = llvm_assemble_text(&air(nonce, taps)).unwrap();
    let library = device.new_library_with_air(&bytes).unwrap();
    let function = |name: &str| {
        StageSource::Function(
            library
                .new_function_with_name(name)
                .unwrap()
                .function_descriptor(),
        )
    };
    let mut descriptor = RenderPipelineDescriptor::new(function("ppc_vertex"));
    descriptor.fragment = Some(function("ppc_fragment"));
    descriptor.color_attachments[0] =
        Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA16Float));
    descriptor
}

/// The device, its queue, a texture of each variant's format, the samplers and the target.
struct Probe {
    device: Device,
    queue: CommandQueue,
    textures: Vec<Texture>,
    samplers: Vec<SamplerState>,
    target: Texture,
    /// A nonce no earlier process used: the clock's microseconds.
    base: u32,
    next: std::sync::atomic::AtomicU32,
}

impl Probe {
    fn new(device: Device) -> Probe {
        let queue = device.new_command_queue().unwrap();
        let texture = |format, side, usage| {
            let mut d = TextureDescriptor::texture_2d(format, side, side, false);
            d.usage = usage;
            d.resource_options = ResourceOptions::private();
            device.new_texture(&d).unwrap()
        };
        let textures = VARIANTS
            .iter()
            .map(|&format| texture(format, 4, TextureUsage::SHADER_READ))
            .collect();
        let target = texture(
            PixelFormat::RGBA16Float,
            SIDE,
            TextureUsage::RENDER_TARGET.union(TextureUsage::SHADER_READ),
        );
        let mut sampler = SamplerDescriptor::new();
        sampler.min_filter = SamplerMinMagFilter::Linear;
        sampler.mag_filter = SamplerMinMagFilter::Linear;
        let samplers = (0..SAMPLERS)
            .map(|_| device.new_sampler_state(&sampler).unwrap())
            .collect();
        let base = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_micros() as u32;
        Probe {
            device,
            queue,
            textures,
            samplers,
            target,
            base,
            next: std::sync::atomic::AtomicU32::new(0),
        }
    }

    /// A 4x4 private texture of `format` the fragment samples.
    fn sampled(&self, format: PixelFormat) -> Texture {
        let mut d = TextureDescriptor::texture_2d(format, 4, 4, false);
        d.usage = TextureUsage::SHADER_READ;
        d.resource_options = ResourceOptions::private();
        self.device.new_texture(&d).unwrap()
    }

    /// A state of a fresh nonce's functions with `taps` samples a slot, and the time making it
    /// took (the translation, its modules and layouts).
    fn state(&self, taps: usize) -> (RenderPipelineState, Duration) {
        let nonce = self
            .base
            .wrapping_add(self.next.fetch_add(1, std::sync::atomic::Ordering::SeqCst));
        self.state_of(nonce, taps)
    }

    /// A state of `nonce`'s functions with `taps` samples a slot, and the time making it took.
    fn state_of(&self, nonce: u32, taps: usize) -> (RenderPipelineState, Duration) {
        let start = Instant::now();
        let descriptor = descriptor(&self.device, nonce, taps);
        let state = self
            .device
            .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
            .unwrap();
        (state, start.elapsed())
    }

    /// One command buffer of one pass drawing `variants` with `state`: each draw call's time,
    /// the command buffer, and `end_encoding`'s time.
    fn encode(
        &self,
        state: &RenderPipelineState,
        variants: &[usize],
    ) -> (Vec<f64>, CommandBuffer, f64) {
        let textures: Vec<&Texture> = variants.iter().map(|&v| &self.textures[v]).collect();
        self.draw(state, &textures)
    }

    /// One command buffer of one pass drawing `state` once per texture of `textures`, bound to
    /// every slot: each draw call's time, the command buffer, and `end_encoding`'s time.
    fn draw(
        &self,
        state: &RenderPipelineState,
        textures: &[&Texture],
    ) -> (Vec<f64>, CommandBuffer, f64) {
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut pass = RenderPassDescriptor::new();
        let color = &mut pass.color_attachments[0];
        color.texture = Some(self.target.clone());
        color.load_action = LoadAction::Clear;
        color.store_action = StoreAction::Store;
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encoder.set_render_pipeline_state(state).unwrap();
        for (index, sampler) in self.samplers.iter().enumerate() {
            encoder
                .set_fragment_sampler_state(Some(sampler), index as u32)
                .unwrap();
        }
        let mut draws = Vec::new();
        for &texture in textures {
            for slot in 0..SLOTS {
                encoder
                    .set_fragment_texture(Some(texture), slot as u32)
                    .unwrap();
            }
            let start = Instant::now();
            encoder
                .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
                .unwrap();
            draws.push(ms(start.elapsed()));
        }
        let start = Instant::now();
        encoder.end_encoding().unwrap();
        (draws, command_buffer, ms(start.elapsed()))
    }
}

/// Commit `command_buffer` and wait for it: the commit's and the wait's times, and the driver's
/// time making each `VkPipeline` its draws recorded.
fn run(command_buffer: &CommandBuffer) -> (f64, f64, Vec<f64>) {
    let start = Instant::now();
    command_buffer.commit().unwrap();
    let commit = ms(start.elapsed());
    let start = Instant::now();
    command_buffer.wait_until_completed_for(WAIT * 10).unwrap();
    let wait = ms(start.elapsed());
    assert_eq!(
        command_buffer.status(),
        CommandBufferStatus::Completed,
        "{:?}",
        command_buffer.error()
    );
    let created = command_buffer
        .executions()
        .iter()
        .flat_map(|e| {
            e.pipeline_creation()
                .iter()
                .map(|&d| ms(d))
                .collect::<Vec<_>>()
        })
        .collect();
    (commit, wait, created)
}

/// The probe's lines (the module docs).
#[test]
#[ignore = "a measurement, not a check: run by name with --run-ignored all"]
fn pipeline_creation_costs() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
        options.layers.clear();
    })
    .unwrap();
    let name = device.name().to_owned();
    let maximize = std::env::var("MVK_CONFIG_SHOULD_MAXIMIZE_CONCURRENT_COMPILATION")
        .unwrap_or_else(|_| "unset".into());
    println!(
        "PPC-CREATE {name}: {SLOTS} half slots, {SAMPLERS} samplers, variants {VARIANTS:?}, \
         MVK maximize concurrent compilation {maximize}"
    );
    let probe = Arc::new(Probe::new(device));
    let variants: Vec<usize> = (0..VARIANTS.len()).collect();
    for taps in TAPS {
        // One thread, one command buffer of the four variants' draws.
        let total = Instant::now();
        let (state, made) = probe.state(taps);
        let (draws, command_buffer, end) = probe.encode(&state, &variants);
        let (commit, wait, created) = run(&command_buffer);
        println!(
            "PPC-CREATE serial taps={taps} state_ms={:.0} draw_ms={} end_ms={end:.1} \
             commit_ms={commit:.1} wait_ms={wait:.0} created_ms={} total_ms={:.0}",
            ms(made),
            list(&draws),
            list(&created),
            ms(total.elapsed())
        );
        // One state, four threads drawing a variant each.
        let (state, made) = probe.state(taps);
        let start = Arc::new(Barrier::new(variants.len()));
        let total = Instant::now();
        let lines: Vec<String> = std::thread::scope(|scope| {
            let threads: Vec<_> = variants
                .iter()
                .map(|&variant| {
                    let (probe, state, start) = (&probe, state.clone(), Arc::clone(&start));
                    scope.spawn(move || {
                        start.wait();
                        let (draws, command_buffer, _) = probe.encode(&state, &[variant]);
                        let (_, _, created) = run(&command_buffer);
                        format!(
                            "v{variant}:draw={} created={}",
                            list(&draws),
                            list(&created)
                        )
                    })
                })
                .collect();
            threads.into_iter().map(|t| t.join().unwrap()).collect()
        });
        println!(
            "PPC-CREATE threads taps={taps} state_ms={:.0} wall_ms={:.0} {}",
            ms(made),
            ms(total.elapsed()),
            lines.join(" ")
        );
        // Four threads, each a state of its own nonce (its own shader modules) and a variant.
        let start = Arc::new(Barrier::new(variants.len()));
        let total = Instant::now();
        let lines: Vec<String> = std::thread::scope(|scope| {
            let threads: Vec<_> = variants
                .iter()
                .map(|&variant| {
                    let (probe, start) = (&probe, Arc::clone(&start));
                    scope.spawn(move || {
                        start.wait();
                        let (state, made) = probe.state(taps);
                        let (draws, command_buffer, _) = probe.encode(&state, &[variant]);
                        let (_, _, created) = run(&command_buffer);
                        format!(
                            "v{variant}:state={:.0} draw={} created={}",
                            ms(made),
                            list(&draws),
                            list(&created)
                        )
                    })
                })
                .collect();
            threads.into_iter().map(|t| t.join().unwrap()).collect()
        });
        println!(
            "PPC-CREATE modules taps={taps} wall_ms={:.0} {}",
            ms(total.elapsed()),
            lines.join(" ")
        );
    }
}

/// Three formats of one `TexelFormat` family (8-bit unsigned normalized, summed exactly) with
/// different channels, and one of another family (`half`), which converts as the first where the
/// device samples through Metal's own texture unit.
const FAMILY: [PixelFormat; 4] = [
    PixelFormat::RG8Unorm,
    PixelFormat::RGBA8Unorm,
    PixelFormat::A8Unorm,
    PixelFormat::RGBA16Float,
];

/// Ten formats whose `TexelFormat` classes differ in bits the fragment's `half` conversion
/// reads (the family, the code width, or the packed channels), so each is its own `VkPipeline`
/// where the device does not sample through Metal's own texture unit; where it does, the 8-bit
/// unsigned normalized, half-float, packed and small float ones share one.
const CLASSES: [PixelFormat; 10] = [
    PixelFormat::RGBA8Unorm,
    PixelFormat::RGBA16Unorm,
    PixelFormat::RGBA8Snorm,
    PixelFormat::RGBA16Snorm,
    PixelFormat::RGBA16Float,
    PixelFormat::B5G6R5Unorm,
    PixelFormat::A1BGR5Unorm,
    PixelFormat::ABGR4Unorm,
    PixelFormat::RGB10A2Unorm,
    PixelFormat::RG11B10Float,
];

/// The samples a slot takes in probes: [`TAPS`]' larger.
const PCC_TAPS: usize = 8;

/// The probe on the device the cfg selects, without the validation layer, and its scratch cache
/// (a fresh one: no translation is cached from another process).
fn measuring() -> (Probe, ScratchCache) {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
        options.layers.clear();
    })
    .unwrap();
    (Probe::new(device), cache)
}

/// The distinct plans (variant and specialization) `command_buffer`'s draws executed: one
/// `VkPipeline` each, its state, pass and fixed state being one.
fn distinct_plans(probe: &Probe, command_buffer: &CommandBuffer) -> usize {
    let device = probe.device.compile_device();
    let plans: std::collections::BTreeSet<String> = command_buffer
        .executions()
        .iter()
        .map(|e| match e.pipeline() {
            ExecutedPipeline::Render(p) => {
                let plan = p.plan(device, e.shape()).unwrap();
                format!("{:?} {:?}", plan.variant, plan.specialization)
            }
            ExecutedPipeline::Compute(_) => panic!("the probe draws only"),
        })
        .collect();
    plans.len()
}

/// The `VkPipeline`s one state's draws of [`FAMILY`] make, one command buffer of four draws: the
/// makings handed to the pool and the distinct plans executed. Three formats of one family whose
/// channels the fragment never reads apart make one `VkPipeline`.
#[test]
#[ignore = "a measurement, not a check: run by name with --run-ignored all"]
fn pipelines_per_format_class() {
    let (probe, _cache) = measuring();
    let textures: Vec<Texture> = FAMILY.iter().map(|&f| probe.sampled(f)).collect();
    let textures: Vec<&Texture> = textures.iter().collect();
    let (state, made) = probe.state(PCC_TAPS);
    let before = probe.device.work_counts().pipelines_pooled;
    let total = Instant::now();
    let (draws, command_buffer, end) = probe.draw(&state, &textures);
    let pooled = probe.device.work_counts().pipelines_pooled - before;
    let (commit, wait, created) = run(&command_buffer);
    println!(
        "PCC-COUNT {}: formats {FAMILY:?}, taps {PCC_TAPS}: draws={} pipelines_pooled={pooled} \
         distinct_plans={} state_ms={:.0} draw_ms={} end_ms={end:.1} commit_ms={commit:.1} \
         wait_ms={wait:.0} created_ms={} total_ms={:.0}",
        probe.device.name(),
        draws.len(),
        distinct_plans(&probe, &command_buffer),
        ms(made),
        list(&draws),
        list(&created),
        ms(total.elapsed())
    );
}

/// Item (b): [`CLASSES`]' ten `VkPipeline`s made by the pool as one command buffer's draws go on,
/// against one at a time (each draw's command buffer committed and waited for before the next is
/// encoded), each by a fresh state: the wall time from the first encoding to the last completion,
/// twice in alternating order.
#[test]
#[ignore = "a measurement, not a check: run by name with --run-ignored all"]
fn pool_against_serial() {
    let (probe, _cache) = measuring();
    let textures: Vec<Texture> = CLASSES.iter().map(|&f| probe.sampled(f)).collect();
    let textures: Vec<&Texture> = textures.iter().collect();
    println!(
        "PCC-POOL {}: formats {CLASSES:?}, taps {PCC_TAPS}",
        probe.device.name()
    );
    for round in 0..2 {
        let (state, made) = probe.state(PCC_TAPS);
        let before = probe.device.work_counts().pipelines_pooled;
        let total = Instant::now();
        let (draws, command_buffer, _) = probe.draw(&state, &textures);
        let pooled = probe.device.work_counts().pipelines_pooled - before;
        let (_, wait, created) = run(&command_buffer);
        println!(
            "PCC-POOL round={round} pooled: state_ms={:.0} wall_ms={:.0} pipelines_pooled={pooled} \
             distinct_plans={} draw_ms={} wait_ms={wait:.0} created_ms={}",
            ms(made),
            ms(total.elapsed()),
            distinct_plans(&probe, &command_buffer),
            list(&draws),
            list(&created)
        );
        let (state, made) = probe.state(PCC_TAPS);
        let total = Instant::now();
        let mut each = Vec::new();
        let mut created = Vec::new();
        for &texture in &textures {
            let start = Instant::now();
            let (_, command_buffer, _) = probe.draw(&state, &[texture]);
            let (_, _, made) = run(&command_buffer);
            each.push(ms(start.elapsed()));
            created.extend(made);
        }
        println!(
            "PCC-POOL round={round} serial: state_ms={:.0} wall_ms={:.0} each_ms={} created_ms={}",
            ms(made),
            ms(total.elapsed()),
            list(&each),
            list(&created)
        );
    }
}

/// The counts of threads [`state_makings_scale_with_the_threads`] makes states on at once, past
/// which it takes the host's hardware threads alone.
const SCALE: [usize; 6] = [1, 2, 4, 8, 16, 32];

/// How making distinct states scales with the threads making them at once. After one state made
/// and drawn untimed (the device's first uses), for each count of [`SCALE`] below the host's
/// hardware threads and for that count itself: that many states of fresh nonces (each its own
/// translation and shader modules) made and drawn one after another on one thread, each state
/// drawing one variant in a command buffer of its own, waited for; then as many threads starting
/// together, each making a fresh state and drawing it so. Printed as `MKS` lines: both wall times
/// from the first making to the last completion, their ratio, and each thread's state creation,
/// draw call and `VkPipeline` creation (as the execution record states it), sorted. Where the
/// makings run side by side, the threads' wall time grows far slower than the count up to the
/// budget's turns (`pipeline/budget.rs`).
#[test]
#[ignore = "a measurement, not a check: run by name with --run-ignored all"]
fn state_makings_scale_with_the_threads() {
    let (probe, _cache) = measuring();
    let host = std::thread::available_parallelism().map_or(1, |n| n.get());
    println!(
        "MKS {}: {host} hardware threads, {SLOTS} half slots, taps {PCC_TAPS}",
        probe.device.name()
    );
    let (state, _) = probe.state(PCC_TAPS);
    let (_, command_buffer, _) = probe.encode(&state, &[0]);
    run(&command_buffer);
    // A fresh state drawn once: its creation, its draw call, and its `VkPipeline`s' creations.
    let make = |variant: usize| {
        let (state, made) = probe.state(PCC_TAPS);
        let (draws, command_buffer, _) = probe.encode(&state, &[variant % VARIANTS.len()]);
        let (_, _, created) = run(&command_buffer);
        (ms(made), draws, created)
    };
    let sorted = |mut values: Vec<f64>| {
        values.sort_by(f64::total_cmp);
        list(&values)
    };
    let mut counts: Vec<usize> = SCALE.into_iter().filter(|&n| n < host).collect();
    counts.push(host);
    for n in counts {
        let total = Instant::now();
        let serial: Vec<_> = (0..n).map(&make).collect();
        let serial_wall = ms(total.elapsed());
        let start = Barrier::new(n);
        let total = Instant::now();
        let threads: Vec<_> = std::thread::scope(|scope| {
            let threads: Vec<_> = (0..n)
                .map(|variant| {
                    let (make, start) = (&make, &start);
                    scope.spawn(move || {
                        start.wait();
                        make(variant)
                    })
                })
                .collect();
            threads.into_iter().map(|t| t.join().unwrap()).collect()
        });
        let wall = ms(total.elapsed());
        let column = |made: &[(f64, Vec<f64>, Vec<f64>)], at: usize| {
            sorted(
                made.iter()
                    .flat_map(|(state, draws, created)| match at {
                        0 => vec![*state],
                        1 => draws.clone(),
                        _ => created.clone(),
                    })
                    .collect(),
            )
        };
        println!(
            "MKS n={n} serial_wall_ms={serial_wall:.0} threads_wall_ms={wall:.0} \
             speedup={:.2} serial_state_ms={} threads_state_ms={} threads_draw_ms={} \
             serial_created_ms={} threads_created_ms={}",
            serial_wall / wall,
            column(&serial, 0),
            column(&threads, 0),
            column(&threads, 1),
            column(&serial, 2),
            column(&threads, 2),
        );
    }
}

/// A nonce of this test binary: a hash of its path (which names a run's target directory)
/// and its modification time, so two processes of one build make the same functions and no other
/// build's.
fn build_nonce() -> u32 {
    use std::hash::{Hash, Hasher};
    let exe = std::env::current_exe().unwrap();
    let modified = std::fs::metadata(&exe)
        .unwrap()
        .modified()
        .unwrap()
        .duration_since(UNIX_EPOCH)
        .unwrap();
    // `DefaultHasher::new` hashes alike in every process.
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    (exe, modified).hash(&mut hasher);
    hasher.finish() as u32
}

/// Item 2: [`CLASSES`]' ten draws by a state of [`build_nonce`]'s functions, one command buffer.
/// Run twice, in two processes, with Metal's shader cache on: the first pays every compile, and the
/// second what Metal's disk cache leaves of it for one build's pipelines.
#[test]
#[ignore = "a measurement, not a check: run by name with --run-ignored all"]
fn pipelines_of_this_build() {
    let (probe, _cache) = measuring();
    let textures: Vec<Texture> = CLASSES.iter().map(|&f| probe.sampled(f)).collect();
    let textures: Vec<&Texture> = textures.iter().collect();
    let nonce = build_nonce();
    let (state, made) = probe.state_of(nonce, PCC_TAPS);
    let total = Instant::now();
    let (_, command_buffer, _) = probe.draw(&state, &textures);
    let (_, wait, created) = run(&command_buffer);
    println!(
        "PCC-PROC {}: nonce {nonce:#x}, formats {CLASSES:?}, taps {PCC_TAPS}: state_ms={:.0} \
         wall_ms={:.0} distinct_plans={} wait_ms={wait:.0} created_ms={} created_sum_ms={:.0}",
        probe.device.name(),
        ms(made),
        ms(total.elapsed()),
        distinct_plans(&probe, &command_buffer),
        list(&created),
        created.iter().sum::<f64>()
    );
}

/// The bytes of a 4x4 texture of `format` whose texels are finite and distinct, filled with
/// `replace_region`.
fn filled(device: &Device, format: PixelFormat) -> Texture {
    let mut descriptor = TextureDescriptor::texture_2d(format, 4, 4, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    let texture = device.new_texture(&descriptor).unwrap();
    let wide = matches!(
        format,
        PixelFormat::RGBA16Float | PixelFormat::RGBA16Unorm | PixelFormat::RGBA16Snorm
    );
    let bytes: Vec<u8> = (0..64u16)
        .flat_map(|component| match (wide, format) {
            // Halves from 0.0625 up, by sixteenths of a unit.
            (true, PixelFormat::RGBA16Float) => half_bits(0.0625 * f32::from(component % 16 + 1))
                .to_le_bytes()
                .to_vec(),
            (true, _) => (component * 997).to_le_bytes().to_vec(),
            (false, _) => vec![(component * 37 % 251) as u8 & 0x7f],
        })
        .collect();
    let row = if wide { 32 } else { 16 };
    texture
        .replace_region(Region::new_2d(0, 0, 4, 4), 0, 0, &bytes, row, 0)
        .unwrap();
    texture
}

/// One command buffer of one pass drawing `state` binding each of `textures` into a quadrant of
/// `target` of its own, then the first quadrant's two first vertices as a line, counted in a
/// `Counting` visibility query at offset 0 of `results`.
fn quadrants(
    queue: &CommandQueue,
    (state, samplers, textures): (&RenderPipelineState, &[SamplerState], &[Texture]),
    (target, results): (&Texture, &metal2vulkan_runtime::Buffer),
) -> CommandBuffer {
    let command_buffer = queue.command_buffer().unwrap();
    let mut pass = RenderPassDescriptor::new();
    let color = &mut pass.color_attachments[0];
    color.texture = Some(target.clone());
    color.load_action = LoadAction::Clear;
    color.store_action = StoreAction::Store;
    pass.visibility_result_buffer = Some(results.clone());
    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
    encoder.set_render_pipeline_state(state).unwrap();
    for (index, sampler) in samplers.iter().enumerate() {
        encoder
            .set_fragment_sampler_state(Some(sampler), index as u32)
            .unwrap();
    }
    encoder
        .set_visibility_result_mode(VisibilityResultMode::Counting, 0)
        .unwrap();
    let half = (SIDE / 2) as f64;
    for (variant, texture) in textures.iter().enumerate() {
        for slot in 0..SLOTS {
            encoder
                .set_fragment_texture(Some(texture), slot as u32)
                .unwrap();
        }
        let (x, y) = ((variant % 2) as f64 * half, (variant / 2) as f64 * half);
        encoder
            .set_viewport(Viewport::new(x, y, half, half, 0.0, 1.0))
            .unwrap();
        encoder
            .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
            .unwrap();
    }
    // A key no creation predicts: the line class's topology.
    encoder
        .set_viewport(Viewport::new(0.0, 0.0, half, half, 0.0, 1.0))
        .unwrap();
    encoder
        .draw_primitives(PrimitiveType::Line, 0, 2, 1, 0)
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer
}

/// Draws whose `VkPipeline`s the runtime makes on its pool while the draws go on draw as draws of
/// pipelines made already: one state drawn binding each of four formats into a quadrant each, then
/// as a line, under a `Counting` visibility query. The format draws take the `VkPipeline` the
/// state's creation made, on every device: each format's class reaches the function as a push
/// field, and none keys apart. The line, whose key no creation predicts, hands its making to the
/// pool; none is made at a draw; the command buffer completes, and a command buffer committed
/// after it on its queue completes after it. The same draws again,
/// their pipelines made, hand nothing to the pool, and render the same bytes and count the same
/// samples. The validation layer reports nothing.
#[test]
fn draws_made_on_the_pool_draw_as_made_ones() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
    let probe = Probe::new(device.clone());
    let (state, _) = probe.state(1);
    let textures: Vec<Texture> = VARIANTS
        .iter()
        .map(|&format| filled(&device, format))
        .collect();
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA16Float, SIDE, SIDE, false);
    descriptor.usage = TextureUsage::RENDER_TARGET;
    let draw = |pooled: u64| {
        let target = device.new_texture(&descriptor).unwrap();
        let results = buffer_with(&device, &[0xee; 8]).unwrap();
        let before = device.work_counts().pipelines_pooled;
        let command_buffer = quadrants(
            &probe.queue,
            (&state, &probe.samplers, &textures),
            (&target, &results),
        );
        assert_eq!(
            device.work_counts().pipelines_pooled - before,
            pooled,
            "the draws' makings handed to the pool"
        );
        // A command buffer after it on its queue, which needs no pipeline.
        let after = probe.queue.command_buffer().unwrap();
        let scratch = buffer_with(&device, &[0; 16]).unwrap();
        let mut blit = after.blit_command_encoder().unwrap();
        blit.fill_buffer(&scratch, 0..16, 7).unwrap();
        blit.end_encoding().unwrap();
        let order = Arc::new(Mutex::new(Vec::new()));
        for (name, buffer) in [("draws", &command_buffer), ("after", &after)] {
            let order = Arc::clone(&order);
            buffer
                .add_scheduled_handler(Box::new(move |_: &CommandBuffer| {
                    order.lock().unwrap().push(name);
                }))
                .unwrap();
        }
        command_buffer.commit().unwrap();
        after.commit().unwrap();
        after.wait_until_completed_for(WAIT * 10).unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        for buffer in [&command_buffer, &after] {
            assert_eq!(
                buffer.status(),
                CommandBufferStatus::Completed,
                "{:?} {:?}",
                buffer.error(),
                buffer.failed_pipeline()
            );
        }
        // Submitted in commit order: the queue's context runs the scheduled handlers in the
        // order the scheduler submitted the command buffers.
        let start = Instant::now();
        while order.lock().unwrap().len() < 2 && start.elapsed() < WAIT {
            std::thread::sleep(Duration::from_millis(1));
        }
        assert_eq!(*order.lock().unwrap(), ["draws", "after"]);
        let created = command_buffer
            .executions()
            .iter()
            .map(|e| e.pipeline_creation().len())
            .sum::<usize>();
        assert_eq!(created, VARIANTS.len() + 1, "each draw's creation recorded");
        let mut pixels = vec![0u8; (SIDE * SIDE * 8) as usize];
        target
            .get_bytes(
                &mut pixels,
                SIDE * 8,
                0,
                Region::new_2d(0, 0, SIDE, SIDE),
                0,
                0,
            )
            .unwrap();
        let counted = u64::from_le_bytes(read_buffer(&results, 0, 8).unwrap().try_into().unwrap());
        (pixels, counted)
    };
    // The bound format's class is a push field. On a device that samples through Metal's own
    // texture unit, a slot's `half` texels are the device's `float` texel rounded to nearest even
    // (`SpecKey::PlainHalfTexels` 1) for every class but a YCbCr field on the grid `8192`
    // (`TexelFormat::plain_on_metal_sampler`), which no variant is: the draws through the probe's
    // linear samplers of normalized coordinates key as the state's creation predicted (no texture
    // bound and Metal's default sampler: its pipeline was made at the creation). Elsewhere every
    // slot's `half` texel is the device's rounding of its own `float` texel
    // (`PrecisionNote::HalfTexelConversion`, whatever is bound), which neither the class nor the
    // samplers' word specializes. The draws take the creation's pipeline on every device; the line
    // alone hands its making to the pool.
    let pooled = draw(1);
    let made = draw(0);
    assert!(pooled.1 > 0, "the draws counted samples");
    assert_eq!(pooled.1, made.1, "the same samples counted");
    assert!(pooled.0 == made.0, "the same bytes rendered");
    layer_reported_nothing_at_all(&messages);
}

/// Pipeline states compare by identity, as Metal's objects do, so a program finds the state a
/// failed draw's pipeline names (`CommandBuffer::failed_pipeline`) among its own: a render state
/// equals its clones and no state made from the same descriptor, and so does a compute state.
#[test]
fn pipeline_states_equal_only_their_own_clones() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
    let descriptor = descriptor(&device, 1, TAPS[0]);
    let make = || {
        device
            .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
            .unwrap()
    };
    let (first, second) = (make(), make());
    assert!(first == first.clone());
    assert!(
        first != second,
        "two states of one descriptor are two states"
    );
    let case = public_case(KERNEL).unwrap();
    let (first, second) = (
        pipeline_state(&device, &case).unwrap(),
        pipeline_state(&device, &case).unwrap(),
    );
    assert!(first == first.clone());
    assert!(first != second, "two states of one kernel are two states");
    layer_reported_nothing_at_all(&messages);
}
