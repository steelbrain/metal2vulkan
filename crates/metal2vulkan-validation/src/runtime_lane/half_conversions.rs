//! A `half` a kernel rounds from a `float` it holds, on the device the cfg selects, under the
//! validation layer: an AIR `fptrunc` of a loaded `float`, of a sum and a product of two and of a
//! fused multiply-add (`DriverDeviation::MetalHalfConversion`), and a `half` `tanh` of a `half`,
//! which the module computes in `float` and rounds.
//!
//! What plain Metal answers (our probe on the Apple M3 Max under macOS 27.0.1, with and without
//! fast math, over these inputs and more): `half(x)` of the `float` the function holds, to nearest
//! even on the `half` grid, `half` denormals kept, infinity from 65520 up; the `float` sum,
//! product and fused multiply-add are IEEE's, rounded to nearest even. Its `half` `tanh` is its own
//! function, which answers 7 of the 65,536 `half`s otherwise than the nearest even `half` of its
//! `float` `tanh`.
//!
//! The module rounds each `half` through the device's conversion of the `float` to 16 bits. Where
//! the profile takes the deviation (MoltenVK on the Apple M3 Max's macOS 27 identity) the
//! conversion kernels record no `PrecisionNote::NativeHalfConversion` and every lane is Metal's
//! word; the `tanh` kernel records it there, as a `half` function computed in `float`. Under the
//! float controls (ANV, RADV) the conversion is Metal's and records no note, but where the driver
//! fuses the sum, the product or the fused multiply-add into it and rounds the exact result once
//! (RADV, `DriverWorkaround::HalfConversionFusedWithArithmetic`), which records it. Elsewhere a
//! kernel that rounds through the conversion records the note. A kernel that records it may answer
//! the device's lanes, which the test prints as `HALF-CONVERSIONS` lines.

use super::degrade_notices::on_device;
use super::driver_deviations::stage_notes;
use super::{buffer_with, read_buffer, WAIT};
use crate::air_library::AirLibrary;
use metal2vulkan::device::VkProperty;
use metal2vulkan::pipeline::{ComputePipelineDescriptor, PrecisionNote, StageSource};
use metal2vulkan_devtools::llvm_assemble_text;
use metal2vulkan_runtime::{CommandBufferStatus, Device, PipelineOptions, Size};

/// A kernel of the lane: its entry, how many `float` operands it reads per lane, and its body
/// between the loads of `%a`, `%b` and `%c` and the store of the `half` `%r`.
struct Kernel {
    entry: &'static str,
    operands: u64,
    body: &'static str,
}

/// The kernels. Each converts the `float` the function holds, but `tanh16`, whose `half` operand
/// is `%a` rounded.
const KERNELS: [Kernel; 5] = [
    Kernel {
        entry: "load",
        operands: 1,
        body: "  %r = fptrunc float %a to half\n",
    },
    Kernel {
        entry: "sum",
        operands: 2,
        body: "  %s = fadd float %a, %b\n  %r = fptrunc float %s to half\n",
    },
    Kernel {
        entry: "product",
        operands: 2,
        body: "  %s = fmul float %a, %b\n  %r = fptrunc float %s to half\n",
    },
    Kernel {
        entry: "fma",
        operands: 3,
        body: "  %s = call float @air.fma.f32(float %a, float %b, float %c)\n  %r = fptrunc float %s to half\n",
    },
    Kernel {
        entry: "tanh16",
        operands: 1,
        body: "  %h = fptrunc float %a to half\n  %r = call half @air.tanh.f16(half %h)\n",
    },
];

/// The lane's AIR: every kernel of [`KERNELS`], buffer 0 its lanes' `float` operands, buffer 1
/// its `half` results.
fn library() -> String {
    let mut defs = String::new();
    let mut metas = String::new();
    let mut refs = Vec::new();
    for (k, kernel) in KERNELS.iter().enumerate() {
        let mut loads = format!(
            "  %i = zext i32 %gid to i64\n  %base = mul i64 %i, {}\n",
            kernel.operands
        );
        for (o, name) in ["a", "b", "c"]
            .iter()
            .enumerate()
            .take(kernel.operands as usize)
        {
            loads.push_str(&format!(
                "  %j{name} = add i64 %base, {o}\n  %p{name} = getelementptr inbounds float, ptr addrspace(1) %in, i64 %j{name}\n  %{name} = load float, ptr addrspace(1) %p{name}, align 4\n"
            ));
        }
        defs.push_str(&format!(
            "define void @{e}(ptr addrspace(1) %in, ptr addrspace(1) %out, i32 %gid) {{\nentry:\n{loads}{body}  %o = getelementptr inbounds half, ptr addrspace(1) %out, i64 %i\n  store half %r, ptr addrspace(1) %o, align 2\n  ret void\n}}\n\n",
            e = kernel.entry,
            body = kernel.body,
        ));
        let n = 10 + 4 * k;
        refs.push(format!("!{n}"));
        metas.push_str(&format!(
            "!{n} = !{{ptr @{e}, !1, !{a}}}\n!{a} = !{{!{b}, !{c}, !2}}\n!{b} = !{{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"float\", !\"air.arg_name\", !\"in\"}}\n!{c} = !{{i32 1, !\"air.buffer\", !\"air.location_index\", i32 1, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 2, !\"air.arg_type_align_size\", i32 2, !\"air.arg_type_name\", !\"half\", !\"air.arg_name\", !\"out\"}}\n",
            e = kernel.entry,
            a = n + 1,
            b = n + 2,
            c = n + 3,
        ));
    }
    format!(
        "; Owned synthetic kernels.\nsource_filename = \"half_conversions.metal\"\n\n{defs}declare float @air.fma.f32(float, float, float)\ndeclare half @air.tanh.f16(half)\n\n!air.kernel = !{{{}}}\n!air.version = !{{!3}}\n!air.language_version = !{{!4}}\n!1 = !{{}}\n!2 = !{{i32 2, !\"air.thread_position_in_grid\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"gid\"}}\n!3 = !{{i32 2, i32 8, i32 0}}\n!4 = !{{!\"Metal\", i32 4, i32 0, i32 0}}\n{metas}",
        refs.join(", ")
    )
}

/// The value of half `h` (finite).
fn half_value(h: u16) -> f64 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = i32::from((h >> 10) & 0x1f);
    let mant = f64::from(h & 0x3ff);
    sign * match exp {
        0 => mant * 2f64.powi(-24),
        e => (1024.0 + mant) * 2f64.powi(e - 25),
    }
}

/// The `float`s the conversions round: every finite `half`'s value, the `float` nearest each
/// midpoint between two adjacent finite `half`s and the `float`s one ulp either side of it, of
/// both signs, and the overflow edge.
fn tie_floats() -> Vec<f32> {
    let mut v = Vec::new();
    for h in 0..0x7c00u16 {
        let x = half_value(h);
        v.push(x as f32);
        if h < 0x7bff {
            let m = ((x + half_value(h + 1)) / 2.0) as f32;
            let b = m.to_bits();
            v.extend([m, f32::from_bits(b - 1), f32::from_bits(b + 1)]);
        }
    }
    v.extend([65504.0, 65519.0, 65519.996, 65520.0, 65536.0, 1.0e5]);
    let negated: Vec<f32> = v.iter().map(|x| -x).collect();
    v.extend(negated);
    v
}

/// Each kernel's lanes: its operands, with no `float` denormal among them or among the `float`
/// result (Metal reads and flushes them as zero, `PrecisionNote::DeviceDenormals`): a loaded
/// `float`, a tie with a small addend, a tie with a factor a few ulps from one, and a fused
/// multiply-add of a tie by one with a small addend.
fn operands(kernel: &Kernel, ties: &[f32]) -> Vec<[f32; 3]> {
    let normal = |x: f32| x == 0.0 || x.abs() >= f32::MIN_POSITIVE;
    ties.iter()
        .enumerate()
        .map(|(n, &t)| {
            let k = (n % 7) as i32 - 3;
            let small = t * 2f32.powi(-20) * k as f32;
            let factor = 1.0 + k as f32 * f32::EPSILON;
            match kernel.entry {
                "sum" => [t, small, 0.0],
                "product" => [t, factor, 0.0],
                "fma" => [t, 1.0, small],
                _ => [t, 0.0, 0.0],
            }
        })
        .filter(|o| o.iter().all(|&x| normal(x)) && normal(result(kernel.entry, o)))
        .collect()
}

/// The `float` a kernel converts: IEEE's operation, rounded to nearest even.
fn result(kernel: &str, o: &[f32; 3]) -> f32 {
    match kernel {
        "sum" => o[0] + o[1],
        "product" => o[0] * o[1],
        "fma" => o[0].mul_add(o[1], o[2]),
        _ => o[0],
    }
}

/// Metal's `half(x)`: to nearest even on the `half` grid, `half` denormals kept, infinity from
/// 65520 up (`x` not a NaN).
pub(super) fn half_of(x: f32) -> u16 {
    let sign = if x.is_sign_negative() { 0x8000 } else { 0 };
    let a = f64::from(x.abs());
    if a >= 65520.0 {
        return sign | 0x7c00;
    }
    if a < 2f64.powi(-14) {
        return sign | (a * 2f64.powi(24)).round_ties_even() as u16;
    }
    let e = a.log2().floor() as i32;
    let e = if 2f64.powi(e) > a { e - 1 } else { e };
    let m = (a / 2f64.powi(e - 10)).round_ties_even() as u32;
    let (e, m) = if m == 2048 { (e + 1, 1024) } else { (e, m) };
    if e > 15 {
        return sign | 0x7c00;
    }
    sign | ((e + 15) as u16) << 10 | (m - 1024) as u16
}

/// `kernel`'s `half` words on `device` for `lanes`.
fn run(device: &Device, kernel: &Kernel, air: &[u8], lanes: &[[f32; 3]]) -> Vec<u16> {
    let library = device.new_library_with_air(air).unwrap();
    let function = library.new_function_with_name(kernel.entry).unwrap();
    let pipeline = device
        .new_compute_pipeline_state(
            &ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor())),
            &PipelineOptions::default(),
        )
        .unwrap();
    let bytes: Vec<u8> = lanes
        .iter()
        .flat_map(|o| o[..kernel.operands as usize].to_vec())
        .flat_map(f32::to_le_bytes)
        .collect();
    let n = lanes.len() as u64;
    let input = buffer_with(device, &bytes).unwrap();
    let out = buffer_with(device, &vec![0x5a; (n * 2) as usize]).unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(&pipeline).unwrap();
    encoder.set_buffer(Some(&input), 0, 0).unwrap();
    encoder.set_buffer(Some(&out), 0, 1).unwrap();
    encoder
        .dispatch_threads(Size::new(n, 1, 1), Size::new(64, 1, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    let got = read_buffer(&out, 0, n * 2).unwrap();
    got.as_chunks::<2>()
        .0
        .iter()
        .map(|b| u16::from_le_bytes(*b))
        .collect()
}

/// Every conversion kernel answers Metal's `half` on every lane and records no
/// `PrecisionNote::NativeHalfConversion` where the profile takes
/// `DriverDeviation::MetalHalfConversion` (`VkProperty::MetalHalfConversion`), and the `tanh`
/// kernel records it there; elsewhere a kernel's lanes may be the device's where it records the
/// note. Every finite `half`'s value and the `float`s at and beside each midpoint between two,
/// both signs (254,000 lanes or so a kernel; the module docs).
#[test]
fn half_conversions_are_metals_where_measured_and_noted_elsewhere() {
    let air = llvm_assemble_text(&library()).unwrap();
    let ties = tie_floats();
    on_device(|device, _| {
        let measured = device
            .compile_device()
            .profile()
            .enabled()
            .properties
            .contains(&VkProperty::MetalHalfConversion);
        let mut report = Vec::new();
        for kernel in &KERNELS {
            let notes = stage_notes(device, &air, kernel.entry);
            let noted = notes.contains(&PrecisionNote::NativeHalfConversion);
            if kernel.entry == "tanh16" {
                // A `half` function computed in `float`: noted where the conversion is measured.
                assert!(!measured || noted, "tanh16: {notes:?}");
                continue;
            }
            assert!(
                !measured || !noted,
                "{}: NativeHalfConversion recorded where the profile takes MetalHalfConversion",
                kernel.entry
            );
            let lanes = operands(kernel, &ties);
            let got = run(device, kernel, &air, &lanes);
            let mut differ = 0;
            for (o, &g) in lanes.iter().zip(&got) {
                let want = half_of(result(kernel.entry, o));
                if g == want {
                    continue;
                }
                assert!(
                    noted,
                    "{}: operands {o:?} answered {g:#06x}, Metal's {want:#06x} (notes {notes:?})",
                    kernel.entry
                );
                differ += 1;
            }
            report.push(format!(
                "HALF-CONVERSIONS {} on {}: {differ} of {} lanes the device's, note {noted}",
                kernel.entry,
                device.name(),
                lanes.len()
            ));
        }
        for line in &report {
            println!("{line}");
        }
    });
}
