//! The device's forms of Metal's maths, through the runtime, on the device the cfg selects, under
//! the validation layer (`candidate::runtime_case::validated_options`, whose policy changes no
//! translation: the device's forms are the only ones). The maths functions, divisions and fused
//! multiply-adds (`PrecisionNote::NativeTranscendentals`, `NativeDivision`, `NativeFma`) and
//! `float` `dot`, `mix` and `fast::sqrt` (`NativeDot`, `NativeMix`, and `NativeTranscendentals`
//! for `fast::sqrt`) are the device's on every device whose functions are not Metal's own (a
//! `half` `dot` or `mix` records its note on every device), and the NaN
//! and minimum and maximum (`NativeNanEncoding`, `NativeMinMax`) on every device not measured to
//! answer Metal's.
//! The stage records exactly the kinds of the forms it takes (where the device's answer is not
//! Metal's own), the pipeline state announces the kinds its stage records, and every thread's
//! answers stay within the bounds the kinds record.
//!
//! The kernel reads ten `float`s a thread and writes fourteen: `sin` and `cos` of an angle in
//! `[-π, π]`, `exp2` of an exponent in `[-10, 10]`, `log2` of a value in `[2^-20, 2^20]`, a
//! precise division (its divisor up to `2^127`, past the `2^126` Vulkan's bound covers, where
//! the translation scales both operands), a `float` `fma`, a `half` `fma`, `fmin`, `fmax` and a
//! product of an operand that is a NaN (a payload of its own) on every fourth thread, a two-lane
//! `dot`, a `mix`, a `fast::sqrt` and a `saturate` of the operand that may be a NaN. The
//! references are computed in `f64` on the host, where each operation of these operands is exact
//! or rounded once below `f32`'s precision.

use super::degrade_notices::{counts, on_device_with};
use super::{buffer_with, read_buffer, Messages, WAIT};
use crate::air_library::AirLibrary;
use metal2vulkan::cache::VariantKey;
use metal2vulkan::device::{FloatControl, FloatWidth, VkProperty};
use metal2vulkan::error::DriverDeviation;
use metal2vulkan::library::Library;
use metal2vulkan::pipeline::{ComputePipelineDescriptor, FunctionDescriptor, PrecisionNote};
use metal2vulkan::pipeline::{Stage, StageSource};
use metal2vulkan::Budget;
use metal2vulkan_devtools::llvm_assemble_text;
use metal2vulkan_runtime::{CommandBufferStatus, Degradation, Device, PipelineOptions, Size};
use std::collections::BTreeSet;

/// Threads a threadgroup.
const THREADS: u64 = 64;

/// Threads in all.
const COUNT: u64 = 4096;

/// `float`s a thread reads.
const IN: usize = 10;

/// `float`s a thread writes.
const OUT: usize = 14;

/// The kernel `nm`: thread `i` reads `in[10i..10i+10]` (`s`, `e`, `l`, `a`, `b`, `c`, `x`, `y`,
/// `z`, `n`) and writes `out[14i..14i+14]`: `sin(s)`, `cos(s)`, `exp2(e)`, `log2(l)`, `a / b`,
/// `fma(a, b, c)`, `fma(half(x), half(y), half(z))`, `fmin(a, n)`, `fmax(n, c)`, `n * b`,
/// `dot(float2(c, s), float2(e, c))`, `mix(s, c, x)`, `fast::sqrt(l)` and `saturate(n)`.
fn kernel() -> String {
    let loads: String = ["s", "e", "l", "a", "b", "c", "x", "y", "z", "n"]
        .iter()
        .enumerate()
        .map(|(k, name)| {
            format!(
                "  %in_at_{name} = add i64 %in_base, {k}\n  %in_ptr_{name} = getelementptr inbounds float, ptr addrspace(1) %in, i64 %in_at_{name}\n  %{name} = load float, ptr addrspace(1) %in_ptr_{name}, align 4\n"
            )
        })
        .collect();
    let stores: String = (0..OUT)
        .map(|k| {
            format!(
                "  %out_at_{k} = add i64 %out_base, {k}\n  %out_ptr_{k} = getelementptr inbounds float, ptr addrspace(1) %out, i64 %out_at_{k}\n  store float %r{k}, ptr addrspace(1) %out_ptr_{k}, align 4\n"
            )
        })
        .collect();
    format!(
        "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"
target triple = \"air64_v28-apple-macosx26.0.0\"

define void @nm(ptr addrspace(1) noundef \"air-buffer-no-alias\" %in, ptr addrspace(1) noundef \"air-buffer-no-alias\" %out, i32 noundef %gid) {{
entry:
  %g = zext i32 %gid to i64
  %in_base = mul i64 %g, {IN}
  %out_base = mul i64 %g, {OUT}
{loads}  %r0 = tail call float @air.sin.f32(float %s)
  %r1 = tail call float @air.cos.f32(float %s)
  %r2 = tail call float @air.exp2.f32(float %e)
  %r3 = tail call float @air.log2.f32(float %l)
  %r4 = fdiv float %a, %b
  %r5 = tail call float @air.fma.f32(float %a, float %b, float %c)
  %hx = fptrunc float %x to half
  %hy = fptrunc float %y to half
  %hz = fptrunc float %z to half
  %hr = tail call half @air.fma.f16(half %hx, half %hy, half %hz)
  %r6 = fpext half %hr to float
  %r7 = tail call float @air.fmin.f32(float %a, float %n)
  %r8 = tail call float @air.fmax.f32(float %n, float %c)
  %r9 = fmul float %n, %b
  %du0 = insertelement <2 x float> undef, float %c, i32 0
  %du = insertelement <2 x float> %du0, float %s, i32 1
  %dv0 = insertelement <2 x float> undef, float %e, i32 0
  %dv = insertelement <2 x float> %dv0, float %c, i32 1
  %r10 = tail call float @air.dot.v2f32(<2 x float> %du, <2 x float> %dv)
  %r11 = tail call float @air.mix.f32(float %s, float %c, float %x)
  %r12 = tail call fast float @air.fast_sqrt.f32(float %l)
  %r13 = tail call float @air.saturate.f32(float %n)
{stores}  ret void
}}

declare float @air.sin.f32(float)
declare float @air.cos.f32(float)
declare float @air.exp2.f32(float)
declare float @air.log2.f32(float)
declare float @air.fma.f32(float, float, float)
declare half @air.fma.f16(half, half, half)
declare float @air.fmin.f32(float, float)
declare float @air.fmax.f32(float, float)
declare float @air.dot.v2f32(<2 x float>, <2 x float>)
declare float @air.mix.f32(float, float, float)
declare float @air.fast_sqrt.f32(float)
declare float @air.saturate.f32(float)

!air.kernel = !{{!0}}
!air.version = !{{!6}}
!air.language_version = !{{!7}}
!0 = !{{ptr @nm, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4, !5}}
!3 = !{{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"float\", !\"air.arg_name\", !\"in\"}}
!4 = !{{i32 1, !\"air.buffer\", !\"air.location_index\", i32 1, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"float\", !\"air.arg_name\", !\"out\"}}
!5 = !{{i32 2, !\"air.thread_position_in_grid\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"gid\"}}
!6 = !{{i32 2, i32 8, i32 0}}
!7 = !{{!\"Metal\", i32 4, i32 0, i32 0}}
"
    )
}

/// A deterministic stream of `u64`s (xorshift64*).
struct Stream(u64);

impl Stream {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// A value in `[0, 1)`.
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// A sign.
    fn sign(&mut self) -> f64 {
        if self.next() & 1 == 0 {
            1.0
        } else {
            -1.0
        }
    }

    /// A `float` of magnitude in `[2^lo, 2^hi)`, of either sign.
    fn magnitude(&mut self, lo: f64, hi: f64) -> f32 {
        (self.sign() * (lo + (hi - lo) * self.unit()).exp2()) as f32
    }

    /// A `half` value of magnitude in `[0.5, 2)`, of either sign, as a `float`.
    fn half(&mut self) -> f32 {
        let m = 1024 + (self.next() % 1024) as i32;
        let e = if self.next() & 1 == 0 { -11 } else { -10 };
        (self.sign() * m as f64 * (e as f64).exp2()) as f32
    }
}

/// A quiet NaN with a payload of its own, which no device answers as its canonical NaN.
const PAYLOAD_NAN: u32 = 0x7fc0_1234;

/// Each thread's ten operands.
fn operands() -> Vec<[f32; IN]> {
    let mut r = Stream(0x9E37_79B9_7F4A_7C15);
    (0..COUNT)
        .map(|i| {
            let s = (std::f64::consts::PI * (2.0 * r.unit() - 1.0)) as f32;
            let e = (20.0 * r.unit() - 10.0) as f32;
            let l = r.magnitude(-20.0, 20.0).abs();
            // Every 16th divisor lies past 2^126, its dividend kept so the quotient is normal.
            let (a, b) = if i % 16 == 0 {
                (r.magnitude(100.0, 120.0), r.magnitude(126.0, 127.9))
            } else {
                let a = (r.sign() * (0.5 + 1.5 * r.unit())) as f32;
                (a, (r.sign() * (0.5 + 1.5 * r.unit())) as f32)
            };
            let c = (r.sign() * (0.5 + 1.5 * r.unit())) as f32;
            let n = match i % 4 {
                0 => f32::from_bits(PAYLOAD_NAN),
                _ => (r.sign() * (0.5 + 1.5 * r.unit())) as f32,
            };
            [s, e, l, a, b, c, r.half(), r.half(), r.half(), n]
        })
        .collect()
}

/// One unit in the last place of a `float` of magnitude `x`'s binade (normal `x`).
fn ulp(x: f64) -> f64 {
    (x.abs().log2().floor() - 23.0).exp2()
}

/// One unit in the last place of a `half` of magnitude `x`'s binade, `2^-24` below the normals.
fn half_ulp(x: f64) -> f64 {
    (x.abs().log2().floor() - 10.0).exp2().max(2f64.powi(-24))
}

/// How far `got` is from `want` past `bound` (none within it), each word's check named. A `want`
/// past the `float` range (a row's huge `fma` operands, the division's) is that infinity, which
/// every rounding answers.
fn excess(what: &str, got: f32, want: f64, bound: f64) -> Option<String> {
    let error = match want as f32 {
        overflow if overflow.is_infinite() => match got == overflow {
            true => 0.0,
            false => f64::INFINITY,
        },
        _ => (got as f64 - want).abs(),
    };
    (got.is_nan() || error > bound)
        .then(|| format!("{what}: {got:e} for {want:e}, bound {bound:e}"))
}

/// The recorded bounds (`PrecisionNote::NativeTranscendentals`, `NativeDivision`, `NativeFma`,
/// `NativeMinMax`, `NativeNanEncoding`, `NativeDot`, `NativeMix`) a thread's results break,
/// given its operands `x`: the minimum and maximum of a number and a NaN the number (as Metal
/// answers), of two numbers the lesser or greater exactly, a product of a NaN some NaN (where
/// Metal answers its canonical `0x7fc00000`), of two numbers correctly rounded, a two-lane `dot`
/// within `5 * 2^-24 * sum |a_i * b_i|`, a `mix` within `2^-21 * (|x * (1 - t)| + |y * t|)`, and
/// `fast::sqrt` within Vulkan's `1 / InverseSqrt`, 4.5 ulp, and `saturate` (Metal's
/// `clamp(x, +0, 1)`) of a NaN zero and of a number that number clamped exactly. A `half` fused
/// multiply-add is within one `half` ulp, or, where the stage rounds `half` natively
/// (`native_half`: `NativeHalfQuantize` or `NativeHalfConversion`), the zero of its sign below
/// `2^-14`.
fn broken(x: &[f32; IN], r: &[f32; OUT], native_half: bool) -> Vec<String> {
    let [s, e, l, a, b, c, hx, hy, hz, n] = x.map(f64::from);
    let products = (c * e).abs() + (s * c).abs();
    let blend = (s * (1.0 - hx)).abs() + (c * hx).abs();
    let log = l.log2();
    let product = a * b;
    let fused = a.mul_add(b, c);
    let half_fused = hx * hy + hz;
    [
        excess("sin", r[0], s.sin(), 2f64.powi(-11)),
        excess("cos", r[1], s.cos(), 2f64.powi(-11)),
        excess(
            "exp2",
            r[2],
            e.exp2(),
            (3.0 + 2.0 * e.abs()) * ulp(e.exp2()),
        ),
        excess(
            "log2",
            r[3],
            log,
            if (0.5..=2.0).contains(&l) {
                2f64.powi(-21)
            } else {
                3.0 * ulp(log)
            },
        ),
        excess("a / b", r[4], a / b, 2.5 * ulp(a / b)),
        excess("fma", r[5], fused, 0.5 * ulp(product) + 0.5 * ulp(fused)),
        match native_half && half_fused.abs() < 2f64.powi(-14) && r[6] == 0.0 {
            true => None,
            false => excess("half fma", r[6], half_fused, half_ulp(half_fused)),
        },
        excess("fmin", r[7], if n.is_nan() { a } else { a.min(n) }, 0.0),
        excess("fmax", r[8], if n.is_nan() { c } else { c.max(n) }, 0.0),
        match n.is_nan() {
            true => (!r[9].is_nan()).then(|| format!("n * b: {:e} for a NaN", r[9])),
            false => excess("n * b", r[9], n * b, 0.5 * ulp(n * b)),
        },
        excess("dot", r[10], c * e + s * c, 5.0 * 2f64.powi(-24) * products),
        excess("mix", r[11], s + hx * (c - s), 2f64.powi(-21) * blend),
        excess("fast::sqrt", r[12], l.sqrt(), 4.5 * ulp(l.sqrt())),
        excess(
            "saturate",
            r[13],
            if n.is_nan() { 0.0 } else { n.clamp(0.0, 1.0) },
            0.0,
        ),
    ]
    .into_iter()
    .flatten()
    .collect()
}

/// The notes `nm`'s stage records on `device`'s compile device.
fn stage_notes(device: &Device, bitcode: &[u8]) -> Vec<PrecisionNote> {
    let compile = device.compile_device();
    let budget = Budget::standard();
    let library = Library::from_air(bitcode).unwrap();
    let function = FunctionDescriptor::new(library.function("nm").unwrap());
    let inputs = VariantKey::new().lowering_inputs(Stage::Kernel);
    let compiled = compile
        .compile_function(&function, &inputs, &budget)
        .unwrap();
    let descriptor = ComputePipelineDescriptor::new(StageSource::Compiled(compiled));
    let stage = compile
        .compile_compute_pipeline(&descriptor, &budget)
        .unwrap()
        .stage;
    stage.precision_notes.to_vec()
}

/// The native kinds of [`stage_notes`].
fn stage_kinds(notes: &[PrecisionNote]) -> BTreeSet<Degradation> {
    notes
        .iter()
        .copied()
        .filter(|note| NATIVE.contains(note))
        .map(Degradation::from_precision_note)
        .collect()
}

/// The notes of native device maths.
const NATIVE: [PrecisionNote; 7] = [
    PrecisionNote::NativeTranscendentals,
    PrecisionNote::NativeDivision,
    PrecisionNote::NativeFma,
    PrecisionNote::NativeNanEncoding,
    PrecisionNote::NativeMinMax,
    PrecisionNote::NativeDot,
    PrecisionNote::NativeMix,
];

/// `x` rounded to a `half` to nearest even (a normal `half`'s magnitude).
fn to_half(x: f64) -> f64 {
    let ulp = half_ulp(x);
    (x / ulp).round_ties_even() * ulp
}

/// On a device whose `float` functions are not Metal's own, the stage records exactly the kinds of
/// the forms it takes: the maths functions `NativeTranscendentals` and `NativeDivision` (and
/// `NativeFma` where an `fma` takes the device's `Fma`), `NativeDot`, `NativeMix` and
/// `NativeTranscendentals` (`fast::sqrt`), and `NativeMinMax` where the device is not measured to
/// answer Metal's operand; and `NativeNanEncoding` wherever the device is not measured to answer
/// Metal's canonical NaN (on one measured to, it records the note where the kernel computes another
/// of Metal's NaNs). The kernel's `a / b`, an `fdiv` without `arcp`, is Metal's precise division,
/// which is the device's `OpFDiv` on every device (`lower::arith::float_binop_notes`): its
/// `NativeDivision` is recorded on every device. The pipeline state announces each native kind of
/// its stage once, and every thread's answers are within the bounds their kinds record. On
/// MoltenVK, whose functions and minimum and maximum are Metal's own, the stage records none of
/// their kinds but the precise division's and announces that one alone. The device announces
/// `float-tininess-after-rounding` as it is made, exactly off MoltenVK.
#[test]
fn native_maths_answers_within_its_recorded_bounds() {
    let bitcode = llvm_assemble_text(&kernel()).unwrap();
    on_device_with(
        |_| {},
        |device, messages| answers_under(device, messages, &bitcode),
    );
}

/// [`native_maths_answers_within_its_recorded_bounds`] on `device`.
fn answers_under(device: &Device, messages: &Messages, bitcode: &[u8]) {
    let name = device.name().to_string();
    let notes = stage_notes(device, bitcode);
    let expected = stage_kinds(&notes);
    // A `half` the device rounds natively may be the zero of its sign below `2^-14`.
    let native_half = notes.iter().any(|note| {
        matches!(
            note,
            PrecisionNote::NativeHalfQuantize | PrecisionNote::NativeHalfConversion
        )
    });
    let compile = device.compile_device();
    let takes = |d| compile.profile().takes_driver_deviation(d);
    let metals = takes(DriverDeviation::MetalFloatFunctions);
    eprintln!("{name}: Metal's own functions {metals}; native kinds {expected:?}");
    // The constructor checked the creation notices against `creation_kinds`: the device's
    // multiply and fused multiply-add, announced as it was made.
    assert_eq!(
        super::creation_kinds(device).contains(&Degradation::FloatTininessAfterRounding),
        !metals,
        "{name}: float-tininess-after-rounding"
    );
    for (note, native) in [
        (PrecisionNote::NativeTranscendentals, !metals),
        // The precise division (`a / b` without `arcp`) on every device.
        (PrecisionNote::NativeDivision, true),
        (PrecisionNote::NativeDot, !metals),
        (PrecisionNote::NativeMix, !metals),
        (
            PrecisionNote::NativeMinMax,
            !metals && !takes(DriverDeviation::MetalMinMax),
        ),
    ] {
        let kind = Degradation::from_precision_note(note);
        assert_eq!(expected.contains(&kind), native, "{name}: {kind:?}");
    }
    if !takes(DriverDeviation::MetalCanonicalNans) {
        let kind = Degradation::from_precision_note(PrecisionNote::NativeNanEncoding);
        assert!(expected.contains(&kind), "{name}: {kind:?}");
    }

    let library = device.new_library_with_air(bitcode).unwrap();
    let function = library.new_function_with_name("nm").unwrap();
    let descriptor =
        ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    let state = device
        .new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap();
    let announced: BTreeSet<Degradation> = counts(&messages.take_notices())
        .into_iter()
        .filter(|(kind, n)| {
            let native = kind
                .precision_note()
                .is_some_and(|note| NATIVE.contains(&note));
            assert!(!native || *n == 1, "{name}: {kind:?} announced {n} times");
            native
        })
        .map(|(kind, _)| kind)
        .collect();
    assert_eq!(announced, expected, "{name}: the announced native kinds");
    if metals {
        // The device's functions are Metal's own (MoltenVK): no native maths to bound;
        // `driver_deviations` checks its answers against Metal's.
        return;
    }

    let operands = operands();
    let bytes: Vec<u8> = operands
        .iter()
        .flatten()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let input = buffer_with(device, &bytes).unwrap();
    let output = buffer_with(device, &vec![0xAB; COUNT as usize * OUT * 4]).unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(&state).unwrap();
    encoder.set_buffer(Some(&input), 0, 0).unwrap();
    encoder.set_buffer(Some(&output), 0, 1).unwrap();
    encoder
        .dispatch_threadgroups(Size::new(COUNT / THREADS, 1, 1), Size::new(THREADS, 1, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    let results: Vec<f32> = read_buffer(&output, 0, COUNT * OUT as u64 * 4)
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| f32::from_le_bytes([w[0], w[1], w[2], w[3]]))
        .collect();
    let mut failures = Vec::new();
    let mut correctly_rounded = 0;
    for (i, (x, r)) in operands
        .iter()
        .zip(results.as_chunks::<OUT>().0.iter())
        .enumerate()
    {
        failures.extend(
            broken(x, r, native_half)
                .into_iter()
                .map(|f| format!("thread {i}: {f}")),
        );
        let [_, _, _, a, b, ..] = x.map(f64::from);
        correctly_rounded += u32::from(r[4] == (a / b) as f32);
    }
    eprintln!(
        "{name}: {} answers break a bound; {correctly_rounded} of {COUNT} quotients correctly \
         rounded",
        failures.len()
    );
    assert!(
        failures.is_empty(),
        "{name}: {:#?}",
        &failures[..failures.len().min(24)]
    );
}

/// The kernel `hr`: thread `i` reads `in[2i]` and `in[2i + 1]` (`a`, `b`) and writes
/// `out[2i] = half(a) + half(b)` and `out[2i + 1] = half(a)`, each widened to `float`.
fn half_kernel() -> String {
    "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"
target triple = \"air64_v28-apple-macosx26.0.0\"

define void @hr(ptr addrspace(1) noundef \"air-buffer-no-alias\" %in, ptr addrspace(1) noundef \"air-buffer-no-alias\" %out, i32 noundef %gid) {
entry:
  %g = zext i32 %gid to i64
  %i0 = mul i64 %g, 2
  %i1 = add i64 %i0, 1
  %pa = getelementptr inbounds float, ptr addrspace(1) %in, i64 %i0
  %pb = getelementptr inbounds float, ptr addrspace(1) %in, i64 %i1
  %a = load float, ptr addrspace(1) %pa, align 4
  %b = load float, ptr addrspace(1) %pb, align 4
  %ha = fptrunc float %a to half
  %hb = fptrunc float %b to half
  %hs = fadd half %ha, %hb
  %s = fpext half %hs to float
  %n = fpext half %ha to float
  %ps = getelementptr inbounds float, ptr addrspace(1) %out, i64 %i0
  %pn = getelementptr inbounds float, ptr addrspace(1) %out, i64 %i1
  store float %s, ptr addrspace(1) %ps, align 4
  store float %n, ptr addrspace(1) %pn, align 4
  ret void
}

!air.kernel = !{!0}
!air.version = !{!6}
!air.language_version = !{!7}
!0 = !{ptr @hr, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"float\", !\"air.arg_name\", !\"in\"}
!4 = !{i32 1, !\"air.buffer\", !\"air.location_index\", i32 1, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"float\", !\"air.arg_name\", !\"out\"}
!5 = !{i32 2, !\"air.thread_position_in_grid\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"gid\"}
!6 = !{i32 2, i32 8, i32 0}
!7 = !{!\"Metal\", i32 4, i32 0, i32 0}
"
    .to_owned()
}

/// Whether `got` is within the native half rounding's recorded bound of Metal's `half` of `x`: one
/// unit in the last place of the `half`, 65504 for an infinity, or a zero for an `x` below `2^-14`
/// (the quantize tests the value it rounds, so Metal's `2^-14` from just below may be a zero too).
fn half_within(got: f32, x: f64) -> bool {
    let want = metal_half(x);
    (want.is_finite() && (f64::from(got) - want).abs() <= half_ulp(want))
        || (want.is_infinite() && f64::from(got) == 65504f64.copysign(want))
        || (x.abs() < 2f64.powi(-14) && got == 0.0)
}

/// Whether `got` is within the bound ([`half_within`]) of the `half` sum of `a` and `b` each first
/// rounded to `half` within it: the bound is each rounding's, so a sum of operands the device
/// rounded an ulp off, or flushed, is the device's sum of those.
fn half_within_sum(got: f32, a: f64, b: f64) -> bool {
    let near = |x: f64| -> Vec<f64> {
        let want = metal_half(x);
        if want.is_infinite() {
            return vec![want, 65504f64.copysign(want)];
        }
        let ulp = half_ulp(want);
        let mut v = vec![want, want - ulp, want + ulp];
        if x.abs() < 2f64.powi(-14) {
            v.push(0f64.copysign(x));
        }
        v
    };
    near(a)
        .iter()
        .any(|ha| near(b).iter().any(|hb| half_within(got, ha + hb)))
}

/// Metal's `half(float)` of `x`: to nearest even on the `half` grid, half denormals kept,
/// infinity from 65520 up.
fn metal_half(x: f64) -> f64 {
    if x.abs() >= 65520.0 {
        return f64::INFINITY.copysign(x);
    }
    to_half(x).copysign(x)
}

/// Each thread's two operands: `half` values and the midpoints between neighbours (ties), in
/// every binade from the half denormals (`2^-24`) to the last (whose last midpoint, 65520, is
/// infinity to Metal), of either sign.
fn half_operands() -> Vec<[f32; 2]> {
    let mut r = Stream(0xD1B5_4A32_D192_ED03);
    let mut one = || {
        let e = -24 + (r.next() % 40) as i32;
        let (first, step) = match e < -14 {
            true => (0.0, 2f64.powi(-24)),
            false => (1024.0, 2f64.powi(e - 10)),
        };
        let tie = if r.next() & 1 == 0 { 0.0 } else { 0.5 };
        let k = first + (r.next() % 1024) as f64 + tie;
        (r.sign() * k * step) as f32
    };
    (0..COUNT).map(|_| [one(), one()]).collect()
}

/// On a device whose float controls do not make its conversion to 16 bits Metal's, the stage
/// records `NativeHalfConversion` (or `NativeHalfQuantize` without `shaderFloat16`, whose narrowing
/// and sum are the bare quantize) and the pipeline state announces it, and each
/// `half` narrowing and sum is within the bound the note records: one unit in the last place of the
/// `half`, 65504 where Metal answers infinity from 65520 up on a device that rounds toward zero,
/// and the zero of its sign below `2^-14` on one that flushes 16-bit denormals (or whose quantize
/// does), the sum's of operands each so rounded ([`half_within_sum`]); or the `float`
/// computation itself, each rounding the device dropped left out: NVIDIA's (RTX 3090) and
/// lavapipe's (Mesa 26.2.4 with LLVM 23.1.1; 2526 of 8192 sums Metal's) compilers drop the
/// conversion pair, so `half(a)` is `a` and the sum `a + b` rounded to `float` once (or of the
/// operands rounded, or rounded itself), exactly. CI's lavapipe (Mesa 26.0.8 with LLVM 21.1.8)
/// rounds here (8192 of 8192). A device whose float controls make its conversion Metal's (the
/// module declares `RoundingModeRTE`, `DenormPreserve` and `SignedZeroInfNanPreserve` for 16
/// bits: ANV and RADV) records neither and answers Metal's exactly, and so does one whose
/// conversion is measured to be Metal's (MoltenVK on the Apple M3 Max's macOS 27 identity,
/// `DriverDeviation::MetalHalfConversion`: the module requires its property), whose `half` sum
/// is its own 16-bit addition.
#[test]
fn half_rounding_answers_within_its_recorded_bound() {
    let bitcode = llvm_assemble_text(&half_kernel()).unwrap();
    let kinds = [
        PrecisionNote::NativeHalfConversion,
        PrecisionNote::NativeHalfQuantize,
    ];
    on_device_with(
        |_| {},
        |device, messages| {
            let name = format!("{} half rounding", device.name());
            // A module rounding under the float controls that make the device's conversion
            // Metal's declares `DenormPreserve` for 16 bits, and so requires its property.
            let required = {
                let library = Library::from_air(&bitcode).unwrap();
                let function = FunctionDescriptor::new(library.function("hr").unwrap());
                let inputs = VariantKey::new().lowering_inputs(Stage::Kernel);
                device
                    .compile_device()
                    .compile_function(&function, &inputs, &Budget::standard())
                    .unwrap()
                    .requirements()
                    .properties
                    .clone()
            };
            let controlled = required.contains(&VkProperty::FloatControl(
                FloatControl::DenormPreserve,
                FloatWidth::F16,
            ));
            let library = device.new_library_with_air(&bitcode).unwrap();
            let function = library.new_function_with_name("hr").unwrap();
            let descriptor = ComputePipelineDescriptor::new(StageSource::Function(
                function.function_descriptor(),
            ));
            let state = device
                .new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
                .unwrap();
            let announced: BTreeSet<PrecisionNote> = counts(&messages.take_notices())
                .into_keys()
                .filter_map(|kind| kind.precision_note())
                .filter(|note| kinds.contains(note))
                .collect();
            // A rounding by the bare quantize is the device's: only the float controls, or a
            // conversion measured to be Metal's, answer Metal's.
            let measured = required.contains(&VkProperty::MetalHalfConversion);
            let metal_s = controlled || measured;
            eprintln!(
                "{name}: float controls {controlled}; measured conversion {measured}; announced \
                 {announced:?}"
            );
            assert_eq!(announced.len(), usize::from(!metal_s), "{name}");
            let operands = half_operands();
            let bytes: Vec<u8> = operands
                .iter()
                .flatten()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            let input = buffer_with(device, &bytes).unwrap();
            let output = buffer_with(device, &vec![0xAB; COUNT as usize * 2 * 4]).unwrap();
            let queue = device.new_command_queue().unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer.compute_command_encoder().unwrap();
            encoder.set_compute_pipeline_state(&state).unwrap();
            encoder.set_buffer(Some(&input), 0, 0).unwrap();
            encoder.set_buffer(Some(&output), 0, 1).unwrap();
            encoder
                .dispatch_threadgroups(Size::new(COUNT / THREADS, 1, 1), Size::new(THREADS, 1, 1))
                .unwrap();
            encoder.end_encoding().unwrap();
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
            let results: Vec<f32> = read_buffer(&output, 0, COUNT * 2 * 4)
                .unwrap()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|w| f32::from_le_bytes([w[0], w[1], w[2], w[3]]))
                .collect();
            let mut failures = Vec::new();
            let mut exact = 0;
            for (i, (&[a, b], r)) in operands
                .iter()
                .zip(results.as_chunks::<2>().0.iter())
                .enumerate()
            {
                let ha = metal_half(f64::from(a));
                let sum = metal_half(ha + metal_half(f64::from(b)));
                // The `float` computation with the device's dropped roundings left out.
                let hb = metal_half(f64::from(b));
                let unrounded_sum = [
                    a + b,
                    ha as f32 + hb as f32,
                    metal_half(f64::from(a + b)) as f32,
                ];
                for (what, got, want, unrounded) in [
                    ("half(a) + half(b)", r[0], sum, &unrounded_sum[..]),
                    ("half(a)", r[1], ha, &[a][..]),
                ] {
                    let same = got.to_bits() == (want as f32).to_bits();
                    exact += u32::from(same);
                    let rounded = match what {
                        "half(a)" => half_within(got, f64::from(a)),
                        _ => half_within_sum(got, f64::from(a), f64::from(b)),
                    };
                    let within =
                        same || rounded || unrounded.iter().any(|u| u.to_bits() == got.to_bits());
                    if !((!metal_s && within) || same) {
                        failures.push(format!(
                            "thread {i}: {what} of {a:e}, {b:e}: {got:e} for Metal's {want:e}"
                        ));
                    }
                }
            }
            eprintln!("{name}: {exact} of {} answers Metal's exactly", 2 * COUNT);
            assert!(
                failures.is_empty(),
                "{name}: {:#?}",
                &failures[..failures.len().min(24)]
            );
        },
    );
}
