//! Device tests of step 12 on the device the cfg selects: Metal's device queries as the runtime
//! answers them, printed per venue, texture creation agreeing with the texture queries' answers,
//! and the Apple-only calls answered typed on a live command buffer and its compute and blit
//! encoders.

use super::{layer_reported_nothing_at_all, validated_device, Messages, ScratchCache};
use crate::gpu::TEST_DEVICE;
use metal2vulkan::device::{VkFeature, VkProperty};
use metal2vulkan::error::{Capability, Feature, Unsupported};
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};
use metal2vulkan::vk::SampleCountFlags;
use metal2vulkan_runtime::{
    CommandBufferStatus, CopyProblem, Device, Error, Invalid, Nil, Problem, Query, ResourceOptions,
    RuntimeGap, Size, Texture, TextureDescriptor, TextureProblem, DEFAULT_PROTECTION_OPTIONS,
};
use std::time::Duration;

/// The longest a command buffer of these tests may take.
const WAIT: Duration = Duration::from_secs(60);

/// The sample counts a Vulkan image can have, 1 to 64: the counts texture creation is asked at.
const SAMPLE_COUNTS: [u64; 7] = [1, 2, 4, 8, 16, 32, 64];

/// Metal's integer color formats, which `supportsTextureSampleCount:` leaves out.
const INTEGER_FORMATS: [PixelFormat; 19] = [
    PixelFormat::R8Uint,
    PixelFormat::R8Sint,
    PixelFormat::R16Uint,
    PixelFormat::R16Sint,
    PixelFormat::RG8Uint,
    PixelFormat::RG8Sint,
    PixelFormat::R32Uint,
    PixelFormat::R32Sint,
    PixelFormat::RG16Uint,
    PixelFormat::RG16Sint,
    PixelFormat::RGBA8Uint,
    PixelFormat::RGBA8Sint,
    PixelFormat::RGB10A2Uint,
    PixelFormat::RG32Uint,
    PixelFormat::RG32Sint,
    PixelFormat::RGBA16Uint,
    PixelFormat::RGBA16Sint,
    PixelFormat::RGBA32Uint,
    PixelFormat::RGBA32Sint,
];

/// One answer of a query: its argument (none for a query without one) and the answer, or its
/// refusal (`NotImplemented` naming the gap the query waits on).
type Answer = (Option<u64>, Result<String, Error>);

/// Each gated query's answers on `device`: the sample counts 0 to 64, the amplification counts 0
/// to 2, Metal's largest (8) and one above it, the rate-map layer counts 0 to 3, and the linear
/// alignment of `RGBA8Unorm` (the device's own query).
fn query_answers(device: &Device) -> Vec<(Query, Vec<Answer>)> {
    let p = device.properties();
    fn one<T: std::fmt::Debug>(answer: Result<T, Error>) -> Vec<Answer> {
        vec![(None, answer.map(|value| format!("{value:?}")))]
    }
    fn gapped<T: std::fmt::Debug>(answer: Result<T, RuntimeGap>) -> Vec<Answer> {
        one(answer.map_err(Error::NotImplemented))
    }
    fn each(counts: &[u64], answer: impl Fn(u64) -> Result<bool, RuntimeGap>) -> Vec<Answer> {
        counts
            .iter()
            .map(|&count| {
                let answer = answer(count).map(|value| value.to_string());
                (Some(count), answer.map_err(Error::NotImplemented))
            })
            .collect()
    }
    Query::ALL
        .into_iter()
        .map(|query| {
            let answers = match query {
                Query::RasterOrderGroups => gapped(p.are_raster_order_groups_supported()),
                Query::Depth24Stencil8 => gapped(p.is_depth24_stencil8_pixel_format_supported()),
                Query::ProgrammableSamplePositions => {
                    gapped(p.are_programmable_sample_positions_supported())
                }
                Query::TextureSampleCount => each(&[0, 1, 2, 3, 4, 8, 16, 32, 64], |count| {
                    p.supports_texture_sample_count(count)
                }),
                Query::VertexAmplificationCount => each(&[0, 1, 2, 8, 9], |count| {
                    p.supports_vertex_amplification_count(count)
                }),
                Query::RasterizationRateMapLayerCount => each(&[0, 1, 2, 3], |count| {
                    p.supports_rasterization_rate_map_with_layer_count(count)
                }),
                Query::MinimumLinearTextureAlignment => one(device
                    .minimum_linear_texture_alignment_for_pixel_format(PixelFormat::RGBA8Unorm)),
                Query::SamplerReductionMode => gapped(p.supports_sampler_reduction_mode()),
                Query::PullModelInterpolation => gapped(p.supports_pull_model_interpolation()),
                other => panic!("{other:?}: a query this test does not ask"),
            };
            (query, answers)
        })
        .collect()
}

/// The runtime device answers Metal's public device queries from their sources: its identity
/// agrees with the description it was created from, and every property equals the compile
/// layer's answer for that description. A query whose runtime piece is not built answers its gap
/// (`families.rs` tests the GPU families). Every answer is printed, with the device's values
/// behind the gaps from their sources, as the venue's record. Texture creation agrees with the
/// texture queries' answers (`textures_are_made_as_the_texture_queries_answer`).
///
/// On a live command buffer and compute encoder, the Apple-only calls answer typed: protection
/// options at Metal's default are accepted and any other refused, every command-predication call
/// is refused, and the compression reinterpretation flush is a no-op. The refusals record
/// nothing: the encoder still ends, and the command buffer commits and completes. Once everything
/// is dropped, the validation layer reported nothing.
#[test]
fn a_runtime_device_answers_its_queries_and_the_apple_only_calls() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let p = device.properties();
        let d = &p.description;
        let metal = device.compile_device().metal_properties();

        assert_eq!(p.name, device.name());
        assert_eq!(p.vendor_id, d.vendor_id);
        assert_eq!(p.device_id, d.device_id);
        assert_eq!(p.api_version, d.api_version);
        // The compile layer's answer, at most three quarters of the device's largest device-local
        // heap, to 16 KiB (`DeviceProperties::max_buffer_length`).
        println!(
            "CAPABILITIES {}: max_buffer_length {} of the compile layer's {}",
            p.name, p.max_buffer_length, metal.max_buffer_length
        );
        assert!(
            p.max_buffer_length == metal.max_buffer_length
                || (p.max_buffer_length < metal.max_buffer_length
                    && p.max_buffer_length.is_multiple_of(16 << 10)),
            "max_buffer_length {} against the compile layer's {}",
            p.max_buffer_length,
            metal.max_buffer_length
        );
        let [width, height, depth] = metal.max_threads_per_threadgroup.map(u64::from);
        assert_eq!(
            p.max_threads_per_threadgroup,
            Size::new(width, height, depth)
        );
        assert_eq!(
            p.max_threadgroup_memory_length,
            u64::from(metal.max_threadgroup_memory_length)
        );
        assert_eq!(p.argument_buffers_support, metal.argument_buffers_tier);
        assert_eq!(p.read_write_texture_support, metal.read_write_texture_tier);
        assert_eq!(
            p.supports_32bit_float_filtering,
            metal.supports_32bit_float_filtering
        );
        assert!(p.max_viewports >= 1, "{}", p.max_viewports);

        eprintln!(
            "runtime properties: {} ({:?}), vendor {:#06x}, device {:#06x}, Vulkan {}.{}",
            p.name, p.kind, p.vendor_id, p.device_id, p.api_version.major, p.api_version.minor
        );
        eprintln!(
            "  maxBufferLength {}, maxThreadsPerThreadgroup {:?}, maxThreadgroupMemoryLength {}",
            p.max_buffer_length, p.max_threads_per_threadgroup, p.max_threadgroup_memory_length
        );
        eprintln!(
            "  argumentBuffersSupport {:?}, readWriteTextureSupport {:?}, \
             supports32BitFloatFiltering {}, maxViewports {}",
            p.argument_buffers_support,
            p.read_write_texture_support,
            p.supports_32bit_float_filtering,
            p.max_viewports
        );
        eprintln!(
            "  behind the gaps: pixel interlock {}, multiview {} of {} views",
            d.has(VkFeature::FragmentShaderPixelInterlock),
            d.has(VkFeature::Multiview),
            d.limits.max_multiview_view_count
        );
        for (query, answers) in query_answers(&device) {
            eprintln!("  {}: {answers:?}", query.selector());
            for (argument, answer) in answers {
                match query.gap() {
                    Some(gap) => assert_eq!(
                        answer,
                        Err(Error::NotImplemented(gap)),
                        "{query:?} {argument:?}"
                    ),
                    None => assert!(answer.is_ok(), "{query:?} {argument:?}: {answer:?}"),
                }
            }
        }
        let buffer = device.new_buffer(16, ResourceOptions::shared()).unwrap();
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        // Options set and set back before any encoder is made under them: nothing of them stays
        // (`protection_options.rs` has what they do).
        command_buffer.set_protection_options(1).unwrap();
        command_buffer
            .set_protection_options(DEFAULT_PROTECTION_OPTIONS)
            .unwrap();
        // Unbalanced records answer NO, as Metal's do; their range holds no dispatch, so the
        // DoWhile after them is killed from its start: its If's End answers NO, its own YES.
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder.encode_start_else().unwrap();
        assert_eq!(encoder.encode_end_if(), Ok(false));
        assert_eq!(encoder.encode_end_while(), Ok(false));
        assert_eq!(
            encoder.encode_end_do_while(Some(&buffer), 8, 3, 1),
            Ok(false)
        );
        encoder.encode_start_while(None, 0, 3, 1).unwrap();
        encoder.encode_start_do_while().unwrap();
        encoder.encode_start_if(Some(&buffer), 0, 3, 1).unwrap();
        encoder.encode_start_else().unwrap();
        assert_eq!(encoder.encode_end_if(), Ok(false));
        assert_eq!(
            encoder.encode_end_do_while(Some(&buffer), 0, 3, 1),
            Ok(true)
        );
        encoder.insert_compressed_texture_reinterpretation_flush();
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        assert_eq!(command_buffer.error(), None);
    }
    layer_reported_nothing_at_all(&messages);
}

/// A private 4² texture of `pf` with `samples` samples, a render target read by shaders: 2D for
/// one sample, else 2D multisample.
fn target(device: &Device, pf: PixelFormat, samples: u64) -> Result<Texture, Error> {
    let mut descriptor = TextureDescriptor::texture_2d(pf, 4, 4, false);
    if samples > 1 {
        descriptor.texture_type = TextureType::Type2DMultisample;
    }
    descriptor.sample_count = samples;
    descriptor.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::private();
    device.new_texture(&descriptor)
}

/// The sample counts some Apple GPU's textures take: a count neither these nor the device take is a
/// descriptor Metal stops (`Invalid`, `Texture(SampleCount)`), checked before anything the device
/// lacks.
const APPLE_SAMPLE_COUNTS: [u64; 3] = [1, 2, 4];

/// What a texture of `samples` samples a device answers `answer` for gets: made where the
/// device takes the count; refused as a count the device lacks (`Unsupported`, the texture sample
/// count) where an Apple GPU takes it; else stopped as Metal stops it (`Invalid`,
/// `Texture(SampleCount)`).
fn made_or_count_refused(made: &Result<Texture, Error>, samples: u64, answer: bool) -> bool {
    match made {
        Ok(_) => true,
        Err(Error::Unsupported(u)) => {
            !answer
                && APPLE_SAMPLE_COUNTS.contains(&samples)
                && u.feature == Feature::TextureSampleCount
        }
        Err(Error::Invalid(Invalid { problem, .. })) => {
            !answer
                && !APPLE_SAMPLE_COUNTS.contains(&samples)
                && *problem == Problem::Texture(TextureProblem::SampleCount)
        }
        Err(_) => false,
    }
}

/// Texture creation agrees with the texture queries' answers on the device, every render target
/// private, 4², and read by shaders:
/// - `isDepth24Stencil8PixelFormatSupported`: no, on every venue, as on an Apple GPU, and a
///   `Depth24UnormStencil8` target is stopped as Metal on one stops a texture of a format it
///   describes none of (`Invalid`, `Texture(NotAFormat)`), whatever the Vulkan driver's
///   `D24_UNORM_S8_UINT`.
/// - `supportsTextureSampleCount:`: at each Vulkan sample count, an `RGBA8Unorm` and a
///   `Depth32Float` target are both made exactly where it answers yes; where it answers no, one
///   of them is refused as a count the device lacks (`Unsupported`, the texture sample count)
///   where an Apple GPU takes the count (1, 2 or 4), and both are stopped as Metal stops a count
///   no Apple GPU takes (`Invalid`, `Texture(SampleCount)`) elsewhere.
/// - Integer formats are outside that answer. At each Vulkan sample count the descriptor takes
///   (one the device's answer or an Apple GPU takes), a target of each of Metal's integer color
///   formats is made, or refused (`Unsupported`, the texture sample count, naming
///   `framebufferIntegerColorSampleCounts`) where its format lacks the count; at any other count
///   every one is stopped (`Invalid`, `Texture(SampleCount)`). Every count in
///   `framebufferIntegerColorSampleCounts`, the counts the device guarantees every integer
///   format, is made; a format may take more. The refusals are printed: 64 samples, which no
///   venue takes, is stopped on every venue.
///
/// The Apple-only texture calls answer as on a Mac without the features: a descriptor's rotation
/// and a texture's protection options are Metal's defaults, a rotation off the default is
/// refused, and on a blit encoder invalidating a compressed texture, whole or one slice and level,
/// is a no-op, a slice or level past the texture refused (`Invalid`). The command buffer
/// completes, and once everything is dropped the validation layer reported nothing.
#[test]
fn textures_are_made_as_the_texture_queries_answer() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let p = device.properties();
        eprintln!("texture queries on {}", p.name);

        let d24s8 = p.is_depth24_stencil8_pixel_format_supported().unwrap();
        let made = target(&device, PixelFormat::Depth24UnormStencil8, 1);
        eprintln!(
            "  isDepth24Stencil8PixelFormatSupported {d24s8}: Depth24UnormStencil8 {:?}",
            made.as_ref().map(|_| "made")
        );
        assert!(!d24s8, "isDepth24Stencil8PixelFormatSupported answers no");
        match made {
            Err(Error::Invalid(Invalid {
                problem: Problem::Texture(TextureProblem::NotAFormat),
                ..
            })) => {}
            other => panic!("Depth24UnormStencil8: {other:?}"),
        }

        let mut answered = Vec::new();
        for samples in SAMPLE_COUNTS {
            let answer = p.supports_texture_sample_count(samples).unwrap();
            let color = target(&device, PixelFormat::RGBA8Unorm, samples);
            let depth = target(&device, PixelFormat::Depth32Float, samples);
            assert!(
                made_or_count_refused(&color, samples, answer),
                "{samples} (answer {answer}): {color:?}"
            );
            assert!(
                made_or_count_refused(&depth, samples, answer),
                "{samples} (answer {answer}): {depth:?}"
            );
            assert_eq!(
                color.is_ok() && depth.is_ok(),
                answer,
                "{samples}: RGBA8Unorm {:?}, Depth32Float {:?}",
                color.err(),
                depth.err()
            );
            if answer {
                answered.push(samples);
            }
        }
        eprintln!("  supportsTextureSampleCount: {answered:?}");
        assert_eq!(answered.first(), Some(&1));

        let guaranteed = p.description.limits.framebuffer_integer_color_sample_counts;
        eprintln!("  framebufferIntegerColorSampleCounts {guaranteed:?}");
        let mut stopped_counts = Vec::new();
        for samples in SAMPLE_COUNTS {
            let bit = u32::try_from(samples).unwrap();
            // A count neither the device nor an Apple GPU takes: every format stopped first.
            let taken = answered.contains(&samples) || APPLE_SAMPLE_COUNTS.contains(&samples);
            if !taken {
                let stop = Problem::Texture(TextureProblem::SampleCount);
                for pf in INTEGER_FORMATS {
                    let made = target(&device, pf, samples);
                    assert!(
                        matches!(&made, Err(Error::Invalid(Invalid { problem, .. })) if *problem == stop),
                        "{pf:?} at {samples}: {made:?}"
                    );
                }
                eprintln!(
                    "  integer targets at {samples}: every one stopped (no Apple GPU takes it)"
                );
                stopped_counts.push(samples);
                continue;
            }
            let refused: Vec<PixelFormat> = INTEGER_FORMATS
                .into_iter()
                .filter(|&pf| match target(&device, pf, samples) {
                    Ok(_) => false,
                    Err(refused) => {
                        let property = VkProperty::FramebufferIntegerColorSampleCount(bit);
                        assert_eq!(
                            refused,
                            Error::Unsupported(Unsupported::needs(
                                Feature::TextureSampleCount,
                                Capability::Property(property),
                            )),
                            "{pf:?} at {samples}"
                        );
                        true
                    }
                })
                .collect();
            if guaranteed.contains(SampleCountFlags::from_raw(bit)) {
                assert!(refused.is_empty(), "{samples}: {refused:?}");
            }
            eprintln!(
                "  integer targets at {samples}: {} made, refused {refused:?}",
                INTEGER_FORMATS.len() - refused.len()
            );
        }
        assert!(
            stopped_counts.contains(&64),
            "stopped at {stopped_counts:?}"
        );

        // A texture's rotation is the runtime's: without framebufferOnly it is refused as Metal's
        // descriptor validation asserts, and the device's creator makes no framebuffer-only
        // texture.
        let mut rotated = TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, 4, 4, false);
        rotated.rotation = 1;
        let refused = |descriptor: &TextureDescriptor, problem: TextureProblem| {
            let made = device.new_texture(descriptor);
            assert!(
                matches!(&made, Err(Error::Invalid(Invalid { problem: p, .. }))
                    if *p == Problem::Texture(problem)),
                "{made:?}"
            );
        };
        refused(&rotated, TextureProblem::RotationWithoutFramebufferOnly);
        rotated.framebuffer_only = true;
        refused(&rotated, TextureProblem::FramebufferOnly);
        let texture = target(&device, PixelFormat::RGBA8Unorm, 1).unwrap();
        assert_eq!(texture.protection_options(), DEFAULT_PROTECTION_OPTIONS);
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.invalidate_compressed_texture(&texture).unwrap();
        blit.invalidate_compressed_texture_slice(&texture, 0, 0)
            .unwrap();
        let past = Error::Invalid(Invalid::new(
            "invalidate_compressed_texture_slice",
            Problem::Copy(CopyProblem::Region),
        ));
        assert_eq!(
            blit.invalidate_compressed_texture_slice(&texture, 1, 0),
            Err(past.clone())
        );
        assert_eq!(
            blit.invalidate_compressed_texture_slice(&texture, 0, 1),
            Err(past)
        );
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        assert_eq!(command_buffer.error(), None);
    }
    layer_reported_nothing_at_all(&messages);
}

/// The device's own numbers, printed per venue (`GPU-CORES`): `gpuCoreCount` is the count its
/// driver reports (`DeviceProperties::gpu_core_count`) or none, never Apple's GPU's; each private
/// property with no answer gives none, and they are announced once, naming them, however often
/// the properties are read.
/// A buffer one byte past `max_buffer_length` is nil, as plain Metal answers
/// (`newBufferWithLength:options:` past `maxBufferLength`), in every storage mode, and the
/// longest Metal's rule allows is its compile layer's answer or three quarters of the device's
/// largest device-local heap, to 16 KiB (`DeviceProperties::max_buffer_length`), printed as
/// `MAX-BUFFER-LENGTH`. `Shared` and `Private` storage are refused alike.
#[test]
fn a_buffer_past_max_buffer_length_is_nil() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let cap = device.properties().max_buffer_length;
        let limit = device.compile_device().metal_properties().max_buffer_length;
        println!(
            "MAX-BUFFER-LENGTH {}: {cap} ({:.3} GiB) of the compile layer's {limit}",
            device.name(),
            cap as f64 / f64::from(1u32 << 30)
        );
        for options in [ResourceOptions::shared(), ResourceOptions::private()] {
            let past = device.new_buffer(cap + 1, options);
            assert_eq!(
                past.err(),
                Some(Error::Nil(Nil::new(
                    "new_buffer",
                    Problem::BufferLength { length: cap + 1 }
                ))),
                "{options:?}"
            );
        }
    }
    layer_reported_nothing_at_all(&messages);
}

/// A `Private` buffer of `max_buffer_length` bytes is made where the device's heap bounds the
/// length (below the compile layer's answer, the RTX 5080 Laptop GPU's), and one byte more is nil.
/// It takes three quarters of the device's memory, so it runs only by name.
#[test]
#[ignore = "allocates three quarters of the device's memory: run by name with --run-ignored all"]
fn a_buffer_of_max_buffer_length_is_made_where_the_heap_bounds_it() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let cap = device.properties().max_buffer_length;
        let limit = device.compile_device().metal_properties().max_buffer_length;
        if cap == limit {
            println!(
                "SKIPPED {}: no heap bounds max_buffer_length ({cap})",
                device.name()
            );
            return;
        }
        let made = device.new_buffer(cap, ResourceOptions::private());
        println!(
            "MAX-BUFFER-LENGTH {}: a Private buffer of {cap} bytes made: {}",
            device.name(),
            made.is_ok()
        );
        let made = made.unwrap_or_else(|e| panic!("a buffer of {cap} bytes: {e}"));
        assert_eq!(made.length(), cap);
        drop(made);
        assert!(matches!(
            device.new_buffer(cap + 1, ResourceOptions::private()),
            Err(Error::Nil(_))
        ));
    }
    layer_reported_nothing_at_all(&messages);
}

#[test]
fn the_device_reports_its_own_core_count() {
    use metal2vulkan_runtime::{Degradation, PrivateCount};
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
    drop(messages.take_notices());
    let p = device.properties();
    let private = device.private_properties();
    let cores = private.count(PrivateCount::GpuCoreCount);
    eprintln!(
        "GPU-CORES {} (vendor {:#06x}): gpu_core_count {:?}; private gpuCoreCount {:?} (Metal's on \
         Apple's GPU {}); no answer: {}",
        p.name,
        p.vendor_id,
        p.gpu_core_count,
        cores.value(),
        cores.metal(),
        private.unknown().join(" ")
    );
    assert_eq!(cores.value(), p.gpu_core_count.map(u64::from));
    assert!(p.gpu_core_count != Some(0));
    let _ = device.private_properties();
    let unknown = super::degrade_notices::counts(&messages.take_notices())
        .get(&Degradation::PrivatePropertiesUnknown)
        .copied()
        .unwrap_or(0);
    assert_eq!(unknown, usize::from(!private.unknown().is_empty()));
}
