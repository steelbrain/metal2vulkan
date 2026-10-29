//! The device's fixed-function conversion of a normalized vertex attribute
//! (`DriverDeviation::MetalNormalizedVertexFetch`), on the device the cfg selects, under the
//! validation layer, at every code of every normalized format.
//!
//! Metal's answer, measured at every code of every normalized format on an Apple M3 Max under
//! macOS 27.0.1 (MSL of ours, per vertex and per instance; the committed recordings of
//! `vertex-format-float4`, `-half4` and `-instance-step` agree at their sites,
//! [`metal_s_recorded_rows_are_the_lane_s_rule`]): a code `c` of a `b`-bit field reads into a
//! `float` attribute as `c / (2^b - 1)` unsigned and `max(c / (2^(b-1) - 1), -1)` signed,
//! correctly rounded, but an `Int1010102Normalized` 10-bit field's, which is
//! `max(c * float(1 / 511), -1)` rounded to `float` (one `f32` ulp toward zero at 96 of the 1,024
//! codes); into a `half` attribute, the exact quotient rounded once to `half`. A missing
//! component reads `(0, 0, 0, 1)`'s.
//!
//! The owned fixture `dsx_int_attributes`' `int_attribute_to_float4` and `int_attribute_to_half4`
//! read attribute 0 and write it to `out[vertex_id]`. Each of the 19 normalized formats is laid
//! in records of 8 bytes, one element per code of its fields (256 for an 8-bit format, 65,536 for
//! a 16-bit one, 1,024 for a packed word), component `i` of element `v` holding code
//! `(v + i (n / 4 + 1)) mod n` (a packed word's 10-bit fields `(v + 257 i) mod 1024`, its 2-bit
//! field `v mod 4`), so every component takes every code. Per vertex, both entries, one draw of a
//! point per element. Per instance, the `float4` entry, one draw of one point per element at its
//! base instance and vertex, so the instance step fetches element `v` for `out[v]`: every element
//! of an 8-bit or packed format, and of a four-component 16-bit one the first `n / 4 + 1`, which
//! put every code in one of its components (a draw an element: the 16-bit formats of fewer
//! components are drawn per vertex alone). Every word is compared with Metal's rule computed
//! exactly on the CPU.
//!
//! Where the profile takes the deviation, an attribute the device fetches in fixed function reads
//! Metal's `float` at every code: the `float4` pipelines record no `DeviceVertexConversion` and
//! require `VkProperty::MetalNormalizedVertexFetch`, and every word is held to Metal's. A `half4`
//! pipeline of an 8- or 10-bit format records no note for its conversion either and requires the
//! property; a 16-bit format's records the note, its `float` quotient rounded to `half` again
//! (one `half` ulp from Metal's at codes 65455 and 65519 unsigned, 32759 and -32759 signed), and
//! there any word the stage's notes cover (that note, or the module's own `half` rounding,
//! `NativeHalfConversion` or `NativeHalfQuantize`) is held within one `half` ulp. Elsewhere every
//! normalized pipeline records the note, and the test prints how many words differ and by how
//! many `f32` ulps at most, holding none. A leg the projection pulls is the module's product of
//! each code and its maximum's reciprocal, not the device's conversion: it records the note on
//! every profile where that product misses Metal's at some code, and is not drawn.

use super::degrade_notices::counts;
use super::{
    buffer_with, half_within, layer_reported_nothing_at_all, read_buffer, validated_device,
    Messages, ScratchCache, WAIT,
};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::device::VkProperty;
use metal2vulkan::error::DriverDeviation;
use metal2vulkan::metal::{PrimitiveType, VertexFormat, VertexStepFunction};
use metal2vulkan::pipeline::{
    PrecisionNote, RenderPipelineDescriptor, StageSource, VertexAttributeDescriptor,
    VertexBufferLayoutDescriptor, VertexDescriptor,
};
use metal2vulkan::Budget;
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    Buffer, CommandBufferStatus, CommandQueue, Degradation, Device, Library, PipelineOptions,
    RenderPassDescriptor,
};

/// How a normalized format lays its fields.
#[derive(Clone, Copy, Debug)]
enum Coding {
    /// `count` fields of `bits` bits, in component order.
    Int {
        bits: u32,
        count: usize,
        signed: bool,
    },
    /// Four unsigned bytes in memory order b, g, r, a.
    Bgra,
    /// One word: 10-bit x, y and z from bit 0, a 2-bit w at bit 30.
    Packed { signed: bool },
}

/// Every normalized `MTLVertexFormat`, with its coding.
const FORMATS: [(VertexFormat, Coding); 19] = {
    use Coding::{Bgra, Int, Packed};
    use VertexFormat as V;
    const fn int(bits: u32, count: usize, signed: bool) -> Coding {
        Int {
            bits,
            count,
            signed,
        }
    }
    [
        (V::UCharNormalized, int(8, 1, false)),
        (V::UChar2Normalized, int(8, 2, false)),
        (V::UChar3Normalized, int(8, 3, false)),
        (V::UChar4Normalized, int(8, 4, false)),
        (V::CharNormalized, int(8, 1, true)),
        (V::Char2Normalized, int(8, 2, true)),
        (V::Char3Normalized, int(8, 3, true)),
        (V::Char4Normalized, int(8, 4, true)),
        (V::UShortNormalized, int(16, 1, false)),
        (V::UShort2Normalized, int(16, 2, false)),
        (V::UShort3Normalized, int(16, 3, false)),
        (V::UShort4Normalized, int(16, 4, false)),
        (V::ShortNormalized, int(16, 1, true)),
        (V::Short2Normalized, int(16, 2, true)),
        (V::Short3Normalized, int(16, 3, true)),
        (V::Short4Normalized, int(16, 4, true)),
        (V::UChar4NormalizedBGRA, Bgra),
        (V::Int1010102Normalized, Packed { signed: true }),
        (V::UInt1010102Normalized, Packed { signed: false }),
    ]
};

/// Bytes of one record, and of one row the function writes.
const STRIDE: usize = 8;
const ROW: usize = 16;

/// The fixture's entries: the `float4` one, and the `half4` one (`half: true`).
const ENTRIES: [(&str, bool); 2] = [
    ("int_attribute_to_float4", false),
    ("int_attribute_to_half4", true),
];

/// One field of a component: its code (sign-extended where signed), its maximum code, and
/// whether Metal reads it through the reciprocal product (`Int1010102Normalized`'s 10-bit
/// fields).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Field {
    code: i64,
    max: i64,
    reciprocal: bool,
}

impl Coding {
    /// The elements that put every code in every component.
    fn elements(self) -> usize {
        match self {
            Coding::Int { bits, .. } => 1 << bits,
            Coding::Bgra => 256,
            Coding::Packed { .. } => 1024,
        }
    }

    /// The legs the lane draws of the format, each an entry, a step and the elements it reads
    /// (the first so many): both entries per vertex over every element; the `float4` entry per
    /// instance, a draw an element, over every element of an 8-bit or packed format, and of a
    /// 16-bit one with four components over the first `n / 4 + 1`, which put every code in one
    /// of its components (the 16-bit formats of fewer components are drawn per vertex alone).
    fn legs(self) -> Vec<(&'static str, bool, VertexStepFunction, usize)> {
        let all = self.elements();
        let mut legs: Vec<_> = ENTRIES
            .iter()
            .map(|&(entry, half)| (entry, half, VertexStepFunction::PerVertex, all))
            .collect();
        let instanced = match self {
            Coding::Int {
                bits: 16, count: 4, ..
            } => Some(all / 4 + 1),
            Coding::Int { bits: 16, .. } => None,
            _ => Some(all),
        };
        if let Some(elements) = instanced {
            legs.push((
                ENTRIES[0].0,
                false,
                VertexStepFunction::PerInstance,
                elements,
            ));
        }
        legs
    }

    /// The raw fields of element `v`, in memory order, each with its width.
    fn raw(self, v: usize) -> Vec<(u32, u32)> {
        let shifted = |bits: u32, i: usize| {
            let n = 1usize << bits;
            ((v + i * (n / 4 + 1)) % n) as u32
        };
        match self {
            Coding::Int { bits, count, .. } => {
                (0..count).map(|i| (shifted(bits, i), bits)).collect()
            }
            Coding::Bgra => (0..4).map(|i| (shifted(8, i), 8)).collect(),
            Coding::Packed { .. } => vec![
                (v as u32 % 1024, 10),
                (((v + 257) % 1024) as u32, 10),
                (((v + 514) % 1024) as u32, 10),
                (v as u32 % 4, 2),
            ],
        }
    }

    /// Element `v`'s record.
    fn record(self, v: usize) -> [u8; STRIDE] {
        let mut out = [0u8; STRIDE];
        match self {
            Coding::Packed { .. } => {
                let mut word = 0u32;
                let mut at = 0;
                for (code, bits) in self.raw(v) {
                    word |= code << at;
                    at += bits;
                }
                out[..4].copy_from_slice(&word.to_le_bytes());
            }
            _ => {
                let mut at = 0;
                for (code, bits) in self.raw(v) {
                    let bytes = (bits / 8) as usize;
                    out[at..at + bytes].copy_from_slice(&code.to_le_bytes()[..bytes]);
                    at += bytes;
                }
            }
        }
        out
    }

    /// The fields element `v` gives the shader's x, y, z and w, `None` where the format has no
    /// such component.
    fn fields(self, v: usize) -> [Option<Field>; 4] {
        let signed = matches!(
            self,
            Coding::Int { signed: true, .. } | Coding::Packed { signed: true }
        );
        let field = |(raw, bits): (u32, u32)| {
            let code = match signed && raw >= 1 << (bits - 1) {
                true => i64::from(raw) - (1 << bits),
                false => i64::from(raw),
            };
            let max = match signed {
                true => (1 << (bits - 1)) - 1,
                false => (1 << bits) - 1,
            };
            Field {
                code,
                max,
                reciprocal: signed && bits == 10,
            }
        };
        let mut fields: Vec<Field> = self.raw(v).into_iter().map(field).collect();
        if matches!(self, Coding::Bgra) {
            fields.swap(0, 2);
        }
        let mut out = [None; 4];
        for (o, f) in out.iter_mut().zip(fields) {
            *o = Some(f);
        }
        out
    }

    /// Whether a field's `float` quotient, rounded to `half` again, can miss Metal's `half`:
    /// a 16-bit field's.
    fn rounds_twice(self) -> bool {
        matches!(self, Coding::Int { bits: 16, .. })
    }
}

/// The binary float of `bits` significand bits and least exponent `least` nearest `num / den`
/// (`den > 0`), ties to even, as an `f64`, which holds every `float` and `half` exactly.
fn nearest(num: i64, den: i64, bits: u32, least: i32) -> f64 {
    let (n, d) = (u128::from(num.unsigned_abs()), den as u128);
    if n == 0 {
        return 0.0;
    }
    let scaled = |e: i32| match e < 0 {
        true => (n << -e, d),
        false => (n, d << e),
    };
    let mut e = (128 - n.leading_zeros()) as i32 - (128 - d.leading_zeros()) as i32 - bits as i32;
    loop {
        let (a, b) = scaled(e);
        match a / b {
            q if q >= 1 << bits => e += 1,
            q if q < 1 << (bits - 1) => e -= 1,
            _ => break,
        }
    }
    let e = e.max(least);
    let (a, b) = scaled(e);
    let (mut q, r) = (a / b, a % b);
    if 2 * r > b || (2 * r == b && q & 1 == 1) {
        q += 1;
    }
    let magnitude = q as f64 * 2f64.powi(e);
    match num < 0 {
        true => -magnitude,
        false => magnitude,
    }
}

/// Metal's `float` of `field`, as bits.
fn metal_float(field: Field) -> u32 {
    let code = field.code.max(-field.max);
    let value = match field.reciprocal {
        // Exactly as Metal computes it: the `f32` product rounds to nearest even.
        true => ((field.code as f32) * (1.0f32 / field.max as f32)).max(-1.0),
        false => nearest(code, field.max, 24, -149) as f32,
    };
    value.to_bits()
}

/// Metal's `half` of `field`, the exact quotient rounded once, widened to `float` bits.
fn metal_half(field: Field) -> u32 {
    (nearest(field.code.max(-field.max), field.max, 11, -24) as f32).to_bits()
}

/// Metal's row of element `v` of `coding` read into a `float4` (`half: false`) or `half4`
/// attribute, as bits.
fn metal_row(coding: Coding, v: usize, half: bool) -> [u32; 4] {
    let defaults = [0.0f32, 0.0, 0.0, 1.0].map(f32::to_bits);
    let fields = coding.fields(v);
    std::array::from_fn(|i| match fields[i] {
        Some(f) if half => metal_half(f),
        Some(f) => metal_float(f),
        None => defaults[i],
    })
}

/// The distance in `f32` ulps of two finite floats' bits.
fn ulps(a: u32, b: u32) -> u64 {
    let ordered = |x: u32| match x >> 31 {
        1 => -i64::from(x & 0x7fff_ffff),
        _ => i64::from(x),
    };
    ordered(a).abs_diff(ordered(b))
}

fn library(device: &Device) -> Library {
    let ll = engine_dir().join("dsx_int_attributes.ll");
    device
        .new_library_with_air(&llvm_assemble(&ll).unwrap())
        .unwrap()
}

/// The descriptor of `function`'s pipeline, rasterization disabled, its attribute 0 of `format`
/// at the start of buffer 1's records stepped by `step`.
fn descriptor(
    library: &Library,
    function: &str,
    format: VertexFormat,
    step: VertexStepFunction,
) -> RenderPipelineDescriptor {
    let function = library.new_function_with_name(function).unwrap();
    let mut descriptor =
        RenderPipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    descriptor.rasterization_enabled = false;
    let mut vertex = VertexDescriptor::new();
    let mut layout = VertexBufferLayoutDescriptor::new(Some(STRIDE as u32));
    layout.step_function = step;
    layout.step_rate = 1;
    vertex.layouts.insert(1, layout);
    vertex
        .attributes
        .insert(0, VertexAttributeDescriptor::new(format, 0, 1));
    descriptor.vertex_descriptor = Some(vertex);
    descriptor
}

/// The rows `descriptor`'s pipeline writes for `elements` records: per vertex one draw of a point
/// per element, per instance one draw of one point per element at its base vertex and instance.
fn drawn(
    queue: &CommandQueue,
    descriptor: &RenderPipelineDescriptor,
    records: &Buffer,
    elements: usize,
    step: VertexStepFunction,
) -> Vec<[u32; 4]> {
    let device = queue.device();
    let pipeline = device
        .new_render_pipeline_state(descriptor, &PipelineOptions::default())
        .unwrap();
    let out = buffer_with(device, &vec![0xCD; ROW * elements]).unwrap();
    let mut pass = RenderPassDescriptor::new();
    pass.render_target_width = 1;
    pass.render_target_height = 1;
    pass.default_raster_sample_count = 1;
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
    encoder.set_render_pipeline_state(&pipeline).unwrap();
    encoder.set_vertex_buffer(Some(&out), 0, 0).unwrap();
    encoder.set_vertex_buffer(Some(records), 0, 1).unwrap();
    let n = elements as u32;
    match step {
        VertexStepFunction::PerInstance => {
            for v in 0..n {
                encoder
                    .draw_primitives(PrimitiveType::Point, v, 1, 1, v)
                    .unwrap();
            }
        }
        _ => encoder
            .draw_primitives(PrimitiveType::Point, 0, n, 1, 0)
            .unwrap(),
    }
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    read_buffer(&out, 0, (ROW * elements) as u64)
        .unwrap()
        .as_chunks::<16>()
        .0
        .iter()
        .map(|r| {
            let w = |i: usize| u32::from_le_bytes([r[i], r[i + 1], r[i + 2], r[i + 3]]);
            [w(0), w(4), w(8), w(12)]
        })
        .collect()
}

/// Every code of every normalized format, read into a `float4` and a `half4` attribute per vertex
/// and into a `float4` one per instance ([`Coding::legs`]), is Metal's where the profile takes
/// `MetalNormalizedVertexFetch` and the pipeline records no note for it, and within the notes it
/// records there otherwise (module docs); elsewhere it is printed.
#[test]
fn every_normalized_code_reads_metal_s_value() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let name = device.name().to_owned();
        let metal = device
            .compile_device()
            .profile()
            .takes_driver_deviation(DriverDeviation::MetalNormalizedVertexFetch);
        println!("NORM-FETCH {name}: the profile takes MetalNormalizedVertexFetch: {metal}");
        let library = library(&device);
        let queue = device.new_command_queue().unwrap();
        let mut wrong = Vec::new();
        let (mut words, mut differ, mut fixed, mut legs, mut noted) = (0usize, 0usize, 0, 0, false);
        for (format, coding) in FORMATS {
            let all = coding.elements();
            let bytes: Vec<u8> = (0..all).flat_map(|v| coding.record(v)).collect();
            let records = buffer_with(&device, &bytes).unwrap();
            for (function, half, step, elements) in coding.legs() {
                legs += 1;
                let what = format!("{function} {format:?} {step:?}");
                let d = descriptor(&library, function, format, step);
                let compiled = device
                    .compile_device()
                    .compile_render_pipeline(&d, &Budget::standard())
                    .unwrap_or_else(|e| panic!("{what}: {e}"));
                let notes = &compiled.stages[0].precision_notes;
                let requires = compiled
                    .requirements
                    .properties
                    .contains(&VkProperty::MetalNormalizedVertexFetch);
                let pulled = !compiled.vertex_input.pulled.is_empty();
                let conversion = notes.contains(&PrecisionNote::DeviceVertexConversion);
                if pulled {
                    // The module's decode, its product of each code and the maximum's
                    // reciprocal: noted on every profile where it misses Metal's at some code
                    // (all but `Int1010102Normalized` read as `float`; a 16-bit format read as
                    // `half`), and no conversion of the device's to measure.
                    let misses = match half {
                        true => coding.rounds_twice(),
                        false => !matches!(coding, Coding::Packed { signed: true }),
                    };
                    println!("NORM-FETCH {name}: {what}: pulled, not measured");
                    assert!(conversion == misses && !requires, "{what}: {notes:?}");
                    continue;
                }
                fixed += 1;
                // The pipeline relies on the measured conversion exactly where it records
                // no note for it.
                let relies = metal && !(half && coding.rounds_twice());
                assert_eq!(
                    (conversion, requires),
                    (!relies, relies),
                    "{what}: {notes:?}, {:?}",
                    compiled.requirements.properties
                );
                noted |= conversion;
                // A word the stage's notes cover is held within one `half` ulp.
                let half_noted = half
                    && (conversion
                        || notes.contains(&PrecisionNote::NativeHalfConversion)
                        || notes.contains(&PrecisionNote::NativeHalfQuantize));
                let rows = drawn(&queue, &d, &records, elements, step);
                let (mut off, mut most, mut first) = (0usize, 0u64, None);
                for (v, got) in rows.iter().enumerate() {
                    let want = metal_row(coding, v, half);
                    for c in 0..4 {
                        if got[c] == want[c] {
                            continue;
                        }
                        off += 1;
                        most = most.max(ulps(got[c], want[c]));
                        first.get_or_insert((v, c, got[c], want[c]));
                        let within = half_within(f32::from_bits(got[c]), f32::from_bits(want[c]));
                        // Off the measured profiles the device's answer is printed, not held.
                        let held = metal && (!half_noted || !within);
                        if held {
                            wrong.push(format!(
                                "{what} element {v} component {c}: got {:#010x}, Metal's \
                                     {:#010x}",
                                got[c], want[c]
                            ));
                        }
                    }
                }
                words += 4 * elements;
                differ += off;
                println!(
                    "NORM-FETCH {name}: {what}: {} words, {off} differ{}, notes {notes:?}",
                    4 * elements,
                    match first {
                        Some((v, c, got, want)) => format!(
                            " (at most {most} f32 ulps; first element {v} component {c}: \
                                 {got:#010x}, Metal's {want:#010x})"
                        ),
                        None => String::new(),
                    }
                );
            }
        }
        for line in wrong.iter().take(32) {
            println!("NORM-FETCH-DIFFERS {name}: {line}");
        }
        println!(
            "NORM-FETCH {name}: {words} words of {fixed} fixed-function pipelines, {differ} differ \
             from Metal's, {} outside the stage's notes",
            wrong.len()
        );
        assert!(
            !metal || fixed == legs,
            "a profile that takes the deviation fetches every format in fixed function here, or \
             the lane measures less than it claims: {fixed} of {legs}"
        );
        assert!(
            wrong.is_empty(),
            "{} words differ from Metal's",
            wrong.len()
        );
        let announced = counts(&messages.take_notices());
        assert_eq!(
            announced.contains_key(&Degradation::DeviceVertexConversion),
            noted,
            "the draws announce the conversion exactly where a pipeline notes it: {announced:?}"
        );
    }
    layer_reported_nothing_at_all(&messages);
}

/// The rows Metal's recording of `case` (the Apple M3 Max, macOS 27.0.1) holds at `site`, as
/// bits.
fn recorded_rows(case: &str, site: &str) -> Vec<[u32; 4]> {
    use base64::Engine as _;
    let path = format!(
        "{}/../metal2vulkan-conformance/recordings/{case}.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let recording: serde_json::Value = serde_json::from_str(&text).unwrap();
    let answer = recording["answers"]
        .as_array()
        .unwrap()
        .iter()
        .find(|a| a["key"]["chip"] == "apple-m3-max" && a["key"]["os_version"] == "27.0.1")
        .unwrap_or_else(|| panic!("{case}: no answer of the M3 Max under macOS 27.0.1"));
    let root = &answer["outcomes"][0]["transcript"]["root"];
    let node = root["children"]
        .as_array()
        .unwrap()
        .iter()
        .find(|n| n["name"] == site)
        .unwrap_or_else(|| panic!("{case}: no site {site}"));
    let rows = node["children"]
        .as_array()
        .unwrap()
        .iter()
        .find(|n| n["name"] == "rows")
        .unwrap_or_else(|| panic!("{case} {site}: no rows"));
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(rows["observed"]["value"]["bytes"].as_str().unwrap())
        .unwrap();
    bytes
        .as_chunks::<16>()
        .0
        .iter()
        .map(|r| {
            let w = |i: usize| u32::from_le_bytes([r[i], r[i + 1], r[i + 2], r[i + 3]]);
            [w(0), w(4), w(8), w(12)]
        })
        .collect()
}

/// The lane's rule is Metal's at every word the committed recordings of `vertex-format-float4`,
/// `-half4` and `-instance-step` hold for a normalized format: each recorded component is the
/// rule's value of the code nearest it, and a component the format lacks the default's.
#[test]
fn metal_s_recorded_rows_are_the_lane_s_rule() {
    let mut checked = 0;
    for (case, half) in [
        ("vertex-format-float4", false),
        ("vertex-format-half4", true),
        ("vertex-format-instance-step", false),
    ] {
        for (format, coding) in FORMATS {
            let rows = recorded_rows(case, format.name());
            assert!(!rows.is_empty(), "{case} {format:?}");
            // The component's field shape, from the format's element 0.
            let shapes = coding.fields(0);
            for (r, row) in rows.iter().enumerate() {
                for (c, &bits) in row.iter().enumerate() {
                    let what = format!("{case} {format:?} row {r} component {c}: {bits:#010x}");
                    let Some(shape) = shapes[c] else {
                        let default = [0.0f32, 0.0, 0.0, 1.0][c].to_bits();
                        assert_eq!(bits, default, "{what}");
                        continue;
                    };
                    let value = f64::from(f32::from_bits(bits));
                    let code = (value * shape.max as f64).round() as i64;
                    let field = Field { code, ..shape };
                    let rule = match half {
                        true => metal_half(field),
                        false => metal_float(field),
                    };
                    assert_eq!(rule, bits, "{what}: the rule's {rule:#010x} of code {code}");
                    checked += 1;
                }
            }
        }
    }
    println!("NORM-FETCH: {checked} recorded components are the lane's rule");
    assert!(checked > 0);
}

/// The CPU rule against values worked by hand: 127/255 is `0x3EFEFEFF` (lavapipe's reciprocal
/// product reads `0x3EFEFF00`), the most negative codes read -1, an
/// `Int1010102Normalized` field reads -260 as `0xBF024120` where the correctly rounded quotient
/// is `0xBF024121`, and code 65519 of 65535 reads `half` `0x3BFF` (rounded once) where its
/// `float` `0x3F7FF000` rounds again to `0x3C00`.
#[test]
fn the_rule_answers_the_worked_values() {
    let field = |code, max, reciprocal| Field {
        code,
        max,
        reciprocal,
    };
    assert_eq!(metal_float(field(127, 255, false)), 0x3EFE_FEFF);
    assert_eq!(metal_float(field(-128, 127, false)), (-1.0f32).to_bits());
    assert_eq!(metal_float(field(-127, 127, false)), (-1.0f32).to_bits());
    assert_eq!(metal_float(field(-512, 511, true)), (-1.0f32).to_bits());
    assert_eq!(metal_float(field(-260, 511, true)), 0xBF02_4120);
    assert_eq!(metal_float(field(-260, 511, false)), 0xBF02_4121);
    assert_eq!(metal_float(field(65519, 65535, false)), 0x3F7F_F000);
    assert_eq!(metal_half(field(65519, 65535, false)), 0x3F7F_E000);
    assert_eq!(metal_float(field(0, 65535, false)), 0);
    assert_eq!(metal_float(field(65535, 65535, false)), 1.0f32.to_bits());
}
