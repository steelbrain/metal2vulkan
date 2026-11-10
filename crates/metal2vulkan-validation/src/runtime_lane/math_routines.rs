//! `frexp` and `ilogb` on the device the cfg selects, under the validation layer.
//!
//! Apple's compiler emits them as calls of runtime routines its backend defines at pipeline
//! creation; the compile layer replaces each call by the routine's answer in plain AIR. The owned
//! fixture `kernel_frexp_ilogb` has a kernel for `float` and one for `half`: each thread reads one
//! input and writes eight words, the scalar forms and lanes of the 4-lane forms beside other
//! lanes' constants. Metal's answer, as an M4 gives it from the fixture's own AIR and as
//! `tests::engine_metal_routines` holds it over a wider table: with `u` the `float` bits (a `half`
//! widened exactly first) and `E` the biased exponent, a zero or `float` denormal answers `(+0, 0)`
//! and `INT_MIN`; an infinity `(x, 0)` and `INT_MAX`; a NaN `(x, 0)` and `INT_MIN`; anything else
//! `((u & 0x807fffff) | 0x3f000000, E - 126)` and `E - 127`.

use super::driver_deviations::{half_nan, stage_notes};
use super::{
    at_plan, buffer_with, layer_reported_nothing_at_all, read_buffer, validated_device_with,
    Messages, ScratchCache, WAIT,
};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::pipeline::{ComputePipelineDescriptor, PrecisionNote, StageSource};
use metal2vulkan_devtools::llvm_assemble;
use metal2vulkan_runtime::{
    CommandBufferStatus, ComputePipelineState, Device, PipelineOptions, Size,
};
use std::path::Path;

/// The `float` inputs: zeros, normals, denormals, the extremes, infinities, quiet and signaling
/// NaNs with payloads.
const FLOATS: [u32; 20] = [
    0x0000_0000,
    0x8000_0000,
    0x3f80_0000,
    0xbf80_0000,
    0x3f40_0000,
    0x4060_0000,
    0x0000_0001,
    0x807f_ffff,
    0x0040_0000,
    0x0080_0000,
    0x80c0_0000,
    0x7f7f_ffff,
    0x7f80_0000,
    0xff80_0000,
    0x7fc0_0001,
    0xffc1_2345,
    0x7f80_0001,
    0x42f6_e979,
    0x7149_f2ca,
    0x0d80_0000,
];

/// The `half` inputs (the low 16 bits of each word): zeros, normals, denormals, the extremes,
/// infinities and NaNs.
const HALVES: [u32; 18] = [
    0x0000, 0x8000, 0x3c00, 0xbc00, 0x0001, 0x8001, 0x03ff, 0x0200, 0x0400, 0x7bff, 0x7c00, 0xfc00,
    0x7e01, 0xfe55, 0x7c01, 0x3555, 0xc8f0, 0x5640,
];

/// Metal's `frexp` and `ilogb` of the `float` of bits `u`: the mantissa's bits, the exponent, and
/// `ilogb`.
fn float_answer(u: u32) -> (u32, i32, i32) {
    let biased = ((u >> 23) & 0xff) as i32;
    match biased {
        0 => (0, 0, i32::MIN),
        0xff if u & 0x7f_ffff == 0 => (u, 0, i32::MAX),
        0xff => (u, 0, i32::MIN),
        _ => ((u & 0x807f_ffff) | 0x3f00_0000, biased - 126, biased - 127),
    }
}

/// The `float` bits of the `half` of bits `h`, exactly (a denormal normalized).
fn widened(h: u16) -> u32 {
    let sign = u32::from(h & 0x8000) << 16;
    let exponent = u32::from((h >> 10) & 0x1f);
    let mantissa = u32::from(h & 0x3ff);
    match exponent {
        0 if mantissa == 0 => sign,
        0 => {
            // `mantissa * 2^-24`, its leading bit at `top`.
            let top = 31 - mantissa.leading_zeros();
            sign | ((top + 103) << 23) | ((mantissa << (23 - top)) & 0x7f_ffff)
        }
        0x1f => sign | 0x7f80_0000 | (mantissa << 13),
        _ => sign | ((exponent + 112) << 23) | (mantissa << 13),
    }
}

/// Metal's `frexp` and `ilogb` of the `half` of bits `h`: the mantissa's 16 bits, the exponent,
/// and `ilogb`.
fn half_answer(h: u16) -> (u32, i32, i32) {
    let (m, e, l) = float_answer(widened(h));
    let mantissa = match m {
        0 => 0,
        // An infinity or a NaN answers `x`.
        _ if (m >> 23) & 0xff == 0xff => u32::from(h),
        // A mantissa in [0.5, 1): `float` exponent 126, `half` 14.
        _ => ((m >> 16) & 0x8000) | (14 << 10) | ((m & 0x7f_ffff) >> 13),
    };
    (mantissa, e, l)
}

/// The eight words a kernel writes for input `x`, its answers being `answer`'s and those of the
/// constants 0.25 and -7 (`quarter`'s mantissa bits). Word 7 also holds, at bit 16, the bit count
/// of `x` times 2^32 + 1, twice `x`'s own: the fixture's 64-bit operation.
fn words(x: u32, answer: fn(u32) -> (u32, i32, i32), quarter: u32) -> [u32; 8] {
    let (m, e, l) = answer(x);
    let wide = (2 * x.count_ones()) << 16;
    // frexp(-7) = (-0.875, 3), ilogb(-7) = 2.
    [
        m,
        e as u32,
        l as u32,
        l as u32,
        m,
        e as u32,
        l as u32,
        quarter ^ 3 ^ 2 ^ wide,
    ]
}

/// The fixture's kernel `name` on `device`.
fn pipeline(device: &Device, name: &str) -> ComputePipelineState {
    let library = device.new_library_with_air(&fixture()).unwrap();
    let function = library.new_function_with_name(name).unwrap();
    let descriptor =
        ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    device
        .new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// The owned fixture's bitcode.
fn fixture() -> Vec<u8> {
    let ll =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/public/engine/kernel_frexp_ilogb.ll");
    llvm_assemble(&ll).unwrap()
}

/// Whether kernel `name`'s stage leaves a NaN's bits to the device on `device`: it records
/// `PrecisionNote::NativeHalfConversion` (the device's conversion, which may quiet a signalling
/// NaN) or `PrecisionNote::NativeNanEncoding`.
fn leaves_nan_bits(device: &Device, name: &str) -> bool {
    let notes = stage_notes(device, &fixture(), name);
    notes.contains(&PrecisionNote::NativeHalfConversion)
        || notes.contains(&PrecisionNote::NativeNanEncoding)
}

/// Whether the word `got` answers Metal's `want`: the same, or where `nan_bits` a NaN in both
/// (of `half` bits in the low half where `half`, else of `float` bits).
fn word_holds(got: u32, want: u32, half: bool, nan_bits: bool) -> bool {
    let nan = |w: u32| match half {
        true => w >> 16 == 0 && half_nan(w as u16),
        false => f32::from_bits(w).is_nan(),
    };
    got == want || (nan_bits && nan(got) && nan(want))
}

/// The words kernel `name` writes for `inputs`, one thread each.
fn dispatched(device: &Device, name: &str, inputs: &[u32]) -> Vec<u32> {
    let pipeline = pipeline(device, name);
    let bytes: Vec<u8> = inputs.iter().flat_map(|x| x.to_le_bytes()).collect();
    let a = buffer_with(device, &bytes).unwrap();
    let o = buffer_with(device, &vec![0xee; 32 * inputs.len()]).unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(&pipeline).unwrap();
    encoder.set_buffer(Some(&a), 0, 0).unwrap();
    encoder.set_buffer(Some(&o), 0, 1).unwrap();
    let n = inputs.len() as u64;
    encoder
        .dispatch_threads(Size::new(n, 1, 1), Size::new(n, 1, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    read_buffer(&o, 0, 32 * n)
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| u32::from_le_bytes(*w))
        .collect()
}

/// Each input's eight words are Metal's on every device, for `float` and `half`, but a NaN
/// fraction's bits, the device's where the kernel's stage leaves them to it ([`leaves_nan_bits`]).
#[test]
fn frexp_and_ilogb_answer_as_metals_routines() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), at_plan).unwrap();
        let name = device.name().to_owned();
        for (kernel, inputs, answer, quarter) in [
            (
                "frexp_ilogb_float",
                &FLOATS[..],
                float_answer as fn(u32) -> (u32, i32, i32),
                0x3f00_0000,
            ),
            (
                "frexp_ilogb_half",
                &HALVES[..],
                (|x| half_answer(x as u16)) as fn(u32) -> (u32, i32, i32),
                0x3800,
            ),
        ] {
            let read = dispatched(&device, kernel, inputs);
            let half = kernel == "frexp_ilogb_half";
            let nan_bits = leaves_nan_bits(&device, kernel);
            for (i, &x) in inputs.iter().enumerate() {
                let (got, want) = (&read[8 * i..8 * i + 8], words(x, answer, quarter));
                let held = (got.iter().zip(&want)).all(|(&g, &w)| word_holds(g, w, half, nan_bits));
                assert!(
                    held,
                    "{kernel} of {x:#010x} on {name}: {got:?}, Metal's {want:?}"
                );
            }
            eprintln!(
                "math routines {kernel} on {name}: Metal's answer for {} inputs",
                inputs.len()
            );
        }
    }
    layer_reported_nothing_at_all(&messages);
}
