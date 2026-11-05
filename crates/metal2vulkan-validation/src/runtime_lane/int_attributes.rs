//! Unnormalized 8- and 16-bit integer vertex formats read into float attributes
//! (`dsx_int_attributes`), on the device the cfg selects, under the validation layer.
//!
//! Metal's answer, measured on an M4 (plain and under its debug layer alike): each component reads
//! as its integer value, zero- or sign-extended, as a float, the missing ones as `(0, 0, 0, 1)`; a
//! `half` attribute rounds the value to nearest even, 65520 and above to infinity; a 32-bit integer
//! format is refused at pipeline creation.
//!
//! - `int_attribute_to_float4` and `int_attribute_to_half4`, for each of the 16 formats at byte
//!   `16 i` of a 256-byte record (row `r`'s component `c` the width's value `(r + 3 c) % 8`, the
//!   probe's tables): per vertex, 8 points, which fixed-function input fetches in the format's
//!   `USCALED` or `SSCALED` form where the device can (the executed pipeline's vertex input says
//!   which; printed), and with the constant step, 8 draws of one point bound at each record, which
//!   the shader pulls and converts, or fixed-function input fetches in the scaled form by a zero
//!   divisor on a binding of stride 0 where the device has one (`metal::vertex::attribute_fetch`;
//!   the NVIDIA RTX 5080 Laptop does). Each row's `float4` is compared bit for bit, but a `half`
//!   one's where the stage rounds by the device's conversion (`native-half-conversion`, announced):
//!   within its bound (`half_within`), counted.
//! - `label_vertex` and `label_fragment`, a text label pipeline's shape (eight attributes, a
//!   `UShort2` among them read as `float2`; nine varyings of mixed interpolation), into a 4x4
//!   `BGRA8Unorm_sRGB` target blended source-over beside a `Depth32Float_Stencil8` attachment:
//!   every pixel is `(255, 0, 255, 255)` in memory order, the fragment's checks of each varying
//!   against Metal's values passed. Metal draws the same pixels. Where the pipeline's stages round
//!   a `half` by the device's conversion (`native-half-conversion`), the red channel's check of
//!   the `half` varying of 4099 (Metal's 4100) may fail: red is then 0, the other checks hold.
//! - A 32-bit integer format read into a float attribute refuses the pipeline,
//!   `InvalidDescriptor(VertexAttribute)`, where Metal refuses it too.

use super::degrade_notices::{counts, made_render_kinds};
use super::{
    buffer_with, half_within, layer_reported_nothing_at_all, read_buffer, validated_device,
    Messages, ScratchCache, WAIT,
};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::error::DescriptorError;
use metal2vulkan::metal::{
    BlendFactor, PixelFormat, PrimitiveType, TextureUsage, VertexFormat, VertexStepFunction,
};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource, VertexAttributeDescriptor,
    VertexBufferLayoutDescriptor, VertexDescriptor,
};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    ClearColor, CommandBufferStatus, CommandQueue, Degradation, Device, Error, ExecutedPipeline,
    Library, LoadAction, PipelineOptions, Region, RenderPassDescriptor, RenderPipelineState,
    ResourceOptions, StoreAction, TextureDescriptor,
};

/// The formats, each with its component bytes, count and signedness.
const FORMATS: [(VertexFormat, usize, usize, bool); 16] = [
    (VertexFormat::UChar, 1, 1, false),
    (VertexFormat::UChar2, 1, 2, false),
    (VertexFormat::UChar3, 1, 3, false),
    (VertexFormat::UChar4, 1, 4, false),
    (VertexFormat::Char, 1, 1, true),
    (VertexFormat::Char2, 1, 2, true),
    (VertexFormat::Char3, 1, 3, true),
    (VertexFormat::Char4, 1, 4, true),
    (VertexFormat::UShort, 2, 1, false),
    (VertexFormat::UShort2, 2, 2, false),
    (VertexFormat::UShort3, 2, 3, false),
    (VertexFormat::UShort4, 2, 4, false),
    (VertexFormat::Short, 2, 1, true),
    (VertexFormat::Short2, 2, 2, true),
    (VertexFormat::Short3, 2, 3, true),
    (VertexFormat::Short4, 2, 4, true),
];

/// The rows, and the record each holds every format in.
const ROWS: usize = 8;
const STRIDE: usize = 16 * FORMATS.len();

/// The probe's values of each width and signedness.
fn values(bytes: usize, signed: bool) -> [i64; 8] {
    match (bytes, signed) {
        (1, false) => [0, 1, 2, 127, 128, 200, 254, 255],
        (1, true) => [0, 1, -1, 127, -128, -2, 100, -100],
        (_, false) => [0, 1, 2049, 2051, 65504, 65519, 65520, 65535],
        (_, true) => [0, -1, 32767, -32768, 2049, -2051, 1000, -1000],
    }
}

/// The records: row `r`'s component `c` of format `i` is its value `(r + 3 c) % 8` at byte
/// `16 i + c * bytes`; the rest is 0xEE.
fn records() -> Vec<u8> {
    let mut out = vec![0xEE; STRIDE * ROWS];
    for r in 0..ROWS {
        for (i, &(_, bytes, count, signed)) in FORMATS.iter().enumerate() {
            for c in 0..count {
                let v = values(bytes, signed)[(r + 3 * c) % 8];
                let at = r * STRIDE + 16 * i + c * bytes;
                out[at..at + bytes].copy_from_slice(&v.to_le_bytes()[..bytes]);
            }
        }
    }
    out
}

/// Metal's `half` of integer `v`: rounded to nearest even at 11 significant bits, 65520 and
/// above to infinity.
fn metal_half(v: i64) -> f32 {
    let a = v.unsigned_abs();
    let magnitude = if a >= 65520 {
        f32::INFINITY
    } else {
        let bits = 64 - a.leading_zeros();
        let rounded = if bits <= 11 {
            a
        } else {
            let shift = bits - 11;
            let (q, rest, half) = (a >> shift, a & ((1 << shift) - 1), 1u64 << (shift - 1));
            let q = if rest > half || (rest == half && q & 1 == 1) {
                q + 1
            } else {
                q
            };
            q << shift
        };
        rounded as f32
    };
    if v < 0 {
        -magnitude
    } else {
        magnitude
    }
}

/// Metal's `float4` of row `r` of format `i` read into a `float4` (`half: false`) or a `half4`
/// attribute, as bits.
fn metal_row(i: usize, r: usize, half: bool) -> [u32; 4] {
    let (_, bytes, count, signed) = FORMATS[i];
    let mut out = [0.0f32, 0.0, 0.0, 1.0];
    for (c, o) in out.iter_mut().enumerate().take(count) {
        let v = values(bytes, signed)[(r + 3 * c) % 8];
        *o = if half { metal_half(v) } else { v as f32 };
    }
    out.map(f32::to_bits)
}

fn library(device: &Device) -> Library {
    let ll = engine_dir().join("dsx_int_attributes.ll");
    device
        .new_library_with_air(&llvm_assemble(&ll).unwrap())
        .unwrap()
}

/// The conversion pipeline of `function`, rasterization disabled, its attribute 0 of `format` at
/// `offset` in buffer 1's layout of `STRIDE` stepped by `step`.
fn conversion(
    device: &Device,
    library: &Library,
    function: &str,
    format: VertexFormat,
    offset: u32,
    step: VertexStepFunction,
) -> Result<RenderPipelineState, Error> {
    let function = library.new_function_with_name(function).unwrap();
    let mut descriptor =
        RenderPipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    descriptor.rasterization_enabled = false;
    let mut vertex = VertexDescriptor::new();
    let mut layout = VertexBufferLayoutDescriptor::new(Some(STRIDE as u32));
    layout.step_function = step;
    layout.step_rate = u32::from(step != VertexStepFunction::Constant);
    vertex.layouts.insert(1, layout);
    vertex
        .attributes
        .insert(0, VertexAttributeDescriptor::new(format, offset, 1));
    descriptor.vertex_descriptor = Some(vertex);
    device.new_render_pipeline_state(&descriptor, &PipelineOptions::default())
}

/// How the executed pipeline fetched attribute 0: its fixed-function format's name, or
/// "pulled".
fn form(pipeline: &ExecutedPipeline) -> String {
    let ExecutedPipeline::Render(p) = pipeline else {
        panic!("a draw executed no render pipeline");
    };
    match p.vertex_input.attributes.first() {
        Some(a) => a.format.name().unwrap_or("unnamed").to_owned(),
        None => {
            assert_eq!(p.vertex_input.pulled.len(), 1, "{:?}", p.vertex_input);
            "pulled".to_owned()
        }
    }
}

/// The rows each draw of `pipeline` wrote, in a 1x1 pass without attachments: per vertex one
/// draw of `ROWS` points, constant `ROWS` draws of one point bound at each record; and how the
/// first executed pipeline fetched the attribute.
fn drawn(
    queue: &CommandQueue,
    pipeline: &RenderPipelineState,
    records: &metal2vulkan_runtime::Buffer,
    constant: bool,
) -> (Vec<[u32; 4]>, String) {
    let device = queue.device();
    let out = buffer_with(device, &[0xCD; 16 * ROWS]).unwrap();
    let mut pass = RenderPassDescriptor::new();
    pass.render_target_width = 1;
    pass.render_target_height = 1;
    pass.default_raster_sample_count = 1;
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
    encoder.set_render_pipeline_state(pipeline).unwrap();
    if constant {
        for r in 0..ROWS {
            encoder
                .set_vertex_buffer(Some(&out), 16 * r as u64, 0)
                .unwrap();
            encoder
                .set_vertex_buffer(Some(records), (STRIDE * r) as u64, 1)
                .unwrap();
            encoder
                .draw_primitives(PrimitiveType::Point, 0, 1, 1, 0)
                .unwrap();
        }
    } else {
        encoder.set_vertex_buffer(Some(&out), 0, 0).unwrap();
        encoder.set_vertex_buffer(Some(records), 0, 1).unwrap();
        encoder
            .draw_primitives(PrimitiveType::Point, 0, ROWS as u32, 1, 0)
            .unwrap();
    }
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    let executions = command_buffer.executions();
    let form = form(executions[0].pipeline());
    let rows = read_buffer(&out, 0, 16 * ROWS as u64)
        .unwrap()
        .as_chunks::<16>()
        .0
        .iter()
        .map(|v| {
            let w = |i: usize| u32::from_le_bytes([v[i], v[i + 1], v[i + 2], v[i + 3]]);
            [w(0), w(4), w(8), w(12)]
        })
        .collect();
    (rows, form)
}

/// Every 8- and 16-bit integer format read into a `float4` and a `half4` attribute, per vertex
/// and with the constant step, reads Metal's values; each fetch is pulled or fixed function in the
/// format's scaled form, as the device fetches it.
#[test]
fn integer_formats_read_into_float_attributes_take_metal_s_values() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let library = library(&device);
        let queue = device.new_command_queue().unwrap();
        let records = buffer_with(&device, &records()).unwrap();
        let mut wrong = Vec::new();
        let mut native = 0usize;
        let mut fixed = 0;
        for (i, &(format, ..)) in FORMATS.iter().enumerate() {
            let own = metal2vulkan::metal::vertex::vk_vertex_format(format).unwrap();
            let scaled = own
                .name()
                .unwrap()
                .replace("_UINT", "_USCALED")
                .replace("_SINT", "_SSCALED");
            for (function, half) in [
                ("int_attribute_to_float4", false),
                ("int_attribute_to_half4", true),
            ] {
                for step in [VertexStepFunction::PerVertex, VertexStepFunction::Constant] {
                    let constant = step == VertexStepFunction::Constant;
                    let pipeline =
                        conversion(&device, &library, function, format, 16 * i as u32, step)
                            .unwrap_or_else(|e| panic!("{function} {format:?} {step:?}: {e}"));
                    let (rows, form) = drawn(&queue, &pipeline, &records, constant);
                    println!(
                        "INT-ATTR {}: {function} {format:?} {step:?}: {form}",
                        device.name()
                    );
                    if form != "pulled" {
                        assert_eq!(form, scaled, "{function} {format:?} {step:?}");
                        fixed += 1;
                    }
                    for (r, got) in rows.iter().enumerate() {
                        let want = metal_row(i, r, half);
                        let within = (0..4)
                            .all(|c| half_within(f32::from_bits(got[c]), f32::from_bits(want[c])));
                        if *got != want && half && within {
                            native += 1;
                            continue;
                        }
                        if *got != want {
                            wrong.push(format!(
                                "{function} {format:?} {step:?} row {r}: got {got:08x?}, Metal {want:08x?}"
                            ));
                        }
                    }
                }
            }
        }
        for line in &wrong {
            println!("INT-ATTR-DIFFERS {}: {line}", device.name());
        }
        println!(
            "INT-ATTR {}: {} rows checked, {} differ, {native} half rows the device's rounding; \
             fixed function in {fixed} of 64 pipelines",
            device.name(),
            FORMATS.len() * 2 * 2 * ROWS,
            wrong.len()
        );
        assert!(wrong.is_empty(), "{} rows differ", wrong.len());
        if native > 0 {
            let announced = counts(&messages.take_notices());
            assert!(
                announced.contains_key(&Degradation::NativeHalfConversion),
                "{announced:?}"
            );
        }
    }
    layer_reported_nothing_at_all(&messages);
}

/// A 32-bit integer format read into a float attribute refuses the pipeline, as on Metal.
#[test]
fn a_32_bit_integer_format_into_a_float_attribute_is_refused() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let library = library(&device);
        for format in [VertexFormat::UInt2, VertexFormat::Int4] {
            for function in ["int_attribute_to_float4", "int_attribute_to_half4"] {
                let made = conversion(
                    &device,
                    &library,
                    function,
                    format,
                    0,
                    VertexStepFunction::PerVertex,
                );
                assert!(
                    matches!(
                        &made,
                        Err(Error::Compile(
                            metal2vulkan::error::Error::InvalidDescriptor(
                                DescriptorError::VertexAttribute { attribute: 0 }
                            )
                        ))
                    ),
                    "{function} {format:?}: {:?}",
                    made.as_ref().map(|_| ())
                );
            }
        }
    }
    layer_reported_nothing_at_all(&messages);
}

/// The label pipeline, its `UShort2` attribute read as `float2`, draws Metal's pixels.
#[test]
fn the_label_pipeline_draws_metal_s_pixels() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let library = library(&device);
        let function = |name: &str| {
            StageSource::Function(
                library
                    .new_function_with_name(name)
                    .unwrap()
                    .function_descriptor(),
            )
        };
        let mut descriptor = RenderPipelineDescriptor::new(function("label_vertex"));
        descriptor.fragment = Some(function("label_fragment"));
        let mut vertex = VertexDescriptor::new();
        vertex
            .layouts
            .insert(0, VertexBufferLayoutDescriptor::new(Some(48)));
        use VertexFormat as V;
        for (a, format, offset) in [
            (0, V::Float2, 0),
            (1, V::UShort2Normalized, 8),
            (2, V::UShort2, 12),
            (3, V::UChar4Normalized, 16),
            (4, V::UChar4Normalized, 20),
            (5, V::UShort2Normalized, 24),
            (6, V::Float2, 32),
            (7, V::Float2, 40),
        ] {
            vertex
                .attributes
                .insert(a, VertexAttributeDescriptor::new(format, offset, 0));
        }
        descriptor.vertex_descriptor = Some(vertex);
        let mut color = ColorAttachmentDescriptor::new(PixelFormat::BGRA8UnormSRGB);
        color.blending_enabled = true;
        color.source_rgb_blend_factor = BlendFactor::SourceAlpha;
        color.destination_rgb_blend_factor = BlendFactor::OneMinusSourceAlpha;
        color.source_alpha_blend_factor = BlendFactor::One;
        color.destination_alpha_blend_factor = BlendFactor::OneMinusSourceAlpha;
        descriptor.color_attachments[0] = Some(color);
        descriptor.depth_format = Some(PixelFormat::Depth32FloatStencil8);
        descriptor.stencil_format = Some(PixelFormat::Depth32FloatStencil8);
        let pipeline = device
            .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
            .unwrap();
        let native_half =
            made_render_kinds(&device, &descriptor).contains(&Degradation::NativeHalfConversion);
        // The records: a corner per vertex of the strip; the rest alike in each.
        let mut records = Vec::new();
        for (x, y) in [(-2.0f32, -2.0f32), (2.0, -2.0), (-2.0, 2.0), (2.0, 2.0)] {
            let mut r = [0u8; 48];
            r[0..4].copy_from_slice(&x.to_le_bytes());
            r[4..8].copy_from_slice(&y.to_le_bytes());
            for (at, v) in [
                (8, 32768u16),
                (10, 65535),
                (12, 300),
                (14, 4099),
                (24, 0),
                (26, 13107),
            ] {
                r[at..at + 2].copy_from_slice(&v.to_le_bytes());
            }
            r[16..20].copy_from_slice(&[255, 128, 0, 64]);
            r[20..24].copy_from_slice(&[10, 20, 30, 255]);
            for (at, v) in [(32, 0.5f32), (36, 0.5), (40, 0.125), (44, -0.25)] {
                r[at..at + 4].copy_from_slice(&v.to_le_bytes());
            }
            records.extend_from_slice(&r);
        }
        let expected: Vec<u8> = [
            300.0f32,
            4099.0,
            (32768.0 / 65535.0) as f32,
            1.0,
            0.0,
            (13107.0 / 65535.0) as f32,
            0.125,
            -0.25,
            1.0,
            (128.0 / 255.0) as f32,
            0.0,
            (64.0 / 255.0) as f32,
            (10.0 / 255.0) as f32,
            (20.0 / 255.0) as f32,
            (30.0 / 255.0) as f32,
            1.0,
            300.0,
            4100.0,
            (32768.0 / 65535.0) as f32,
            1.0,
        ]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
        let records = buffer_with(&device, &records).unwrap();
        let expected = buffer_with(&device, &expected).unwrap();
        let mut target = TextureDescriptor::texture_2d(PixelFormat::BGRA8UnormSRGB, 4, 4, false);
        target.usage = TextureUsage::RENDER_TARGET;
        target.resource_options = ResourceOptions::shared();
        let target = device.new_texture(&target).unwrap();
        let mut depth =
            TextureDescriptor::texture_2d(PixelFormat::Depth32FloatStencil8, 4, 4, false);
        depth.usage = TextureUsage::RENDER_TARGET;
        depth.resource_options = ResourceOptions::private();
        let depth = device.new_texture(&depth).unwrap();
        let mut glyphs = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 1, 1, false);
        glyphs.usage = TextureUsage::SHADER_READ;
        glyphs.resource_options = ResourceOptions::shared();
        let glyphs = device.new_texture(&glyphs).unwrap();
        glyphs
            .replace_region(Region::new_2d(0, 0, 1, 1), 0, 0, &[255; 4], 4, 0)
            .unwrap();
        let sampler = device.new_sampler_state(&SamplerDescriptor::new()).unwrap();
        let mut pass = RenderPassDescriptor::new();
        let c = &mut pass.color_attachments[0];
        c.texture = Some(target.clone());
        c.load_action = LoadAction::Clear;
        c.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
        c.store_action = StoreAction::Store;
        pass.depth_attachment.texture = Some(depth.clone());
        pass.depth_attachment.load_action = LoadAction::Clear;
        pass.depth_attachment.clear_depth = 1.0;
        pass.depth_attachment.store_action = StoreAction::DontCare;
        pass.stencil_attachment.texture = Some(depth.clone());
        pass.stencil_attachment.load_action = LoadAction::Clear;
        pass.stencil_attachment.store_action = StoreAction::DontCare;
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encoder.set_render_pipeline_state(&pipeline).unwrap();
        encoder.set_vertex_buffer(Some(&records), 0, 0).unwrap();
        encoder.set_fragment_buffer(Some(&expected), 0, 0).unwrap();
        encoder.set_fragment_texture(Some(&glyphs), 0).unwrap();
        encoder
            .set_fragment_sampler_state(Some(&sampler), 0)
            .unwrap();
        encoder
            .draw_primitives(PrimitiveType::TriangleStrip, 0, 4, 1, 0)
            .unwrap();
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        let executions = command_buffer.executions();
        let ExecutedPipeline::Render(executed) = executions[0].pipeline() else {
            panic!("a draw executed no render pipeline");
        };
        let count = executed
            .vertex_input
            .attributes
            .iter()
            .find(|a| a.offset == 12)
            .map_or("pulled", |a| a.format.name().unwrap_or("unnamed"));
        println!(
            "INT-ATTR {}: the label pipeline's UShort2 attribute: {count}",
            device.name()
        );
        let mut pixels = vec![0u8; 64];
        target
            .get_bytes(&mut pixels, 16, 0, Region::new_2d(0, 0, 4, 4), 0, 0)
            .unwrap();
        println!(
            "INT-ATTR {}: label pixels {pixels:?} (the device's half rounding: {native_half})",
            device.name()
        );
        let held = pixels.as_chunks::<4>().0.iter().all(|p| match native_half {
            true => matches!(p, [255, 0, 0 | 255, 255]),
            false => *p == [255, 0, 255, 255],
        });
        assert!(held, "the label's pixels: {pixels:?}");
    }
    layer_reported_nothing_at_all(&messages);
}
