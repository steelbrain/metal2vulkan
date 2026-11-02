//! The packed normalized formats' conversions measured to be Metal's where the profile takes
//! `DriverDeviation::MetalPackConversions`, on the device the cfg selects, under the validation
//! layer: `unpack_unorm565_to_float` and `pack_float_to_unorm565`, and in `half` the unpacks of
//! `unorm4x8`, `snorm4x8`, `unorm10a2` and `unorm565` and the packs of those and of `unorm2x16`
//! and `snorm2x16`.
//!
//! What plain Metal answers (our probe on the Apple M3 Max under macOS 27.0.1, with and without
//! fast math, over these inputs and more): a `half` unpack is each field's exact quotient
//! `c / (2^N - 1)` (signed: `c / (2^(N-1) - 1)`, at least `-1`) rounded to `half` once; a `float`
//! `rgb565` unpack is the code times the `float` nearest `1 / (2^N - 1)`, rounded to `float`; a
//! pack from `half` rounds the exact product of the channel clamped to `[0, 1]` (`[-1, 1]`
//! signed) and `2^N - 1` (`2^(N-1) - 1`) to nearest even; a `float` `rgb565` pack rounds that
//! product rounded to `float` first, to nearest even; a NaN channel packs as code 0.
//!
//! Where the profile takes the deviation (MoltenVK on the Apple M3 Max's macOS 27 identity), no
//! kernel records `PrecisionNote::DevicePackConversion` and every lane is Metal's. Elsewhere each
//! records the note and its lanes are the device's, which the test prints as `PACK-CONVERSIONS`
//! lines.

use super::degrade_notices::on_device;
use super::driver_deviations::stage_notes;
use super::{buffer_with, read_buffer, WAIT};
use crate::air_library::AirLibrary;
use metal2vulkan::device::VkProperty;
use metal2vulkan::pipeline::{ComputePipelineDescriptor, PrecisionNote, StageSource};
use metal2vulkan_devtools::llvm_assemble_text;
use metal2vulkan_runtime::{CommandBufferStatus, Device, PipelineOptions, Size};

/// The kind of a kernel's input word.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Input {
    /// A packed word: every code of every field.
    Packed,
    /// A `half`, broadcast to every channel: every bit pattern.
    Half,
    /// A `float`, broadcast to every channel ([`float_words`]).
    Float,
}

/// A kernel of the lane: its entry, the AIR format name, its lanes, whether it packs, its input
/// and whether its floats are `half`.
struct Kernel {
    entry: &'static str,
    format: &'static str,
    lanes: u32,
    packs: bool,
    input: Input,
}

impl Kernel {
    /// Whether the kernel's floats are `half`.
    fn half(&self) -> bool {
        match self.input {
            Input::Half => true,
            Input::Float => false,
            Input::Packed => self.entry.starts_with("uh_"),
        }
    }

    /// The packed integer's width: 16 for `rgb565`, else 32.
    fn width(&self) -> u32 {
        match self.format {
            "unorm.rgb565" => 16,
            _ => 32,
        }
    }

    /// The format's fields: (offset, width, signed), in channel order.
    fn fields(&self) -> &'static [(u32, u32, bool)] {
        match self.format {
            "unorm4x8" => &[(0, 8, false), (8, 8, false), (16, 8, false), (24, 8, false)],
            "snorm4x8" => &[(0, 8, true), (8, 8, true), (16, 8, true), (24, 8, true)],
            "unorm2x16" => &[(0, 16, false), (16, 16, false)],
            "snorm2x16" => &[(0, 16, true), (16, 16, true)],
            "unorm.rgb10a2" => &[
                (0, 10, false),
                (10, 10, false),
                (20, 10, false),
                (30, 2, false),
            ],
            "unorm.rgb565" => &[(0, 5, false), (5, 6, false), (11, 5, false)],
            f => unreachable!("{f}"),
        }
    }
}

/// The kernels: every form `DriverDeviation::MetalPackConversions` covers.
const KERNELS: [Kernel; 12] = [
    unpack("u565f", "unorm.rgb565", 3),
    unpack("uh_unorm4x8", "unorm4x8", 4),
    unpack("uh_snorm4x8", "snorm4x8", 4),
    unpack("uh_10a2", "unorm.rgb10a2", 4),
    unpack("uh_565", "unorm.rgb565", 3),
    pack("ph_unorm4x8", "unorm4x8", 4, Input::Half),
    pack("ph_snorm4x8", "snorm4x8", 4, Input::Half),
    pack("ph_unorm2x16", "unorm2x16", 2, Input::Half),
    pack("ph_snorm2x16", "snorm2x16", 2, Input::Half),
    pack("ph_10a2", "unorm.rgb10a2", 4, Input::Half),
    pack("ph_565", "unorm.rgb565", 3, Input::Half),
    pack("p565f", "unorm.rgb565", 3, Input::Float),
];

const fn unpack(entry: &'static str, format: &'static str, lanes: u32) -> Kernel {
    Kernel {
        entry,
        format,
        lanes,
        packs: false,
        input: Input::Packed,
    }
}

const fn pack(entry: &'static str, format: &'static str, lanes: u32, input: Input) -> Kernel {
    Kernel {
        entry,
        format,
        lanes,
        packs: true,
        input,
    }
}

/// The lane's AIR: every kernel of [`KERNELS`], buffer 0 its input words (`uint`, or `half` for
/// a `half` pack), buffer 1 its results: an unpack's four lanes per input (`float` or `half`, the
/// unused lanes left), a pack's `uint`.
fn library() -> String {
    let mut defs = String::new();
    let mut decls = String::new();
    let mut metas = String::new();
    let mut refs = Vec::new();
    for (k, kernel) in KERNELS.iter().enumerate() {
        let (fty, fname, fsize) = match kernel.half() {
            true => ("half", "f16", 2),
            false => ("float", "f32", 4),
        };
        let lanes = kernel.lanes;
        let vty = format!("<{lanes} x {fty}>");
        let iw = format!("i{}", kernel.width());
        let call = format!(
            "air.{}.{}.v{lanes}{fname}",
            if kernel.packs { "pack" } else { "unpack" },
            kernel.format
        );
        let (inty, insize) = match kernel.input {
            Input::Half => ("half", 2),
            Input::Float => ("float", 4),
            Input::Packed => ("i32", 4),
        };
        let outsize = match kernel.packs {
            true => 4,
            false => fsize,
        };
        let mut body = format!(
            "  %i = zext i32 %gid to i64\n  %p = getelementptr inbounds {inty}, ptr addrspace(1) %in, i64 %i\n  %x = load {inty}, ptr addrspace(1) %p, align {insize}\n"
        );
        if kernel.packs {
            decls.push_str(&format!("declare {iw} @{call}({vty})\n"));
            let mut prev = "undef".to_string();
            for l in 0..lanes {
                body.push_str(&format!(
                    "  %v{l} = insertelement {vty} {prev}, {fty} %x, i32 {l}\n"
                ));
                prev = format!("%v{l}");
            }
            body.push_str(&format!("  %w = call {iw} @{call}({vty} {prev})\n"));
            body.push_str(match kernel.width() {
                16 => "  %r = zext i16 %w to i32\n",
                _ => "  %r = add i32 %w, 0\n",
            });
            body.push_str(
                "  %o = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %i\n  store i32 %r, ptr addrspace(1) %o, align 4\n",
            );
        } else {
            decls.push_str(&format!("declare {vty} @{call}({iw})\n"));
            let w = match kernel.width() {
                16 => {
                    body.push_str("  %x16 = trunc i32 %x to i16\n");
                    "%x16"
                }
                _ => "%x",
            };
            body.push_str(&format!("  %v = call {vty} @{call}({iw} {w})\n"));
            body.push_str("  %b = shl i64 %i, 2\n");
            for l in 0..lanes {
                body.push_str(&format!(
                    "  %e{l} = extractelement {vty} %v, i32 {l}\n  %j{l} = or i64 %b, {l}\n  %o{l} = getelementptr inbounds {fty}, ptr addrspace(1) %out, i64 %j{l}\n  store {fty} %e{l}, ptr addrspace(1) %o{l}, align {fsize}\n"
                ));
            }
        }
        defs.push_str(&format!(
            "define void @{e}(ptr addrspace(1) %in, ptr addrspace(1) %out, i32 %gid) {{\nentry:\n{body}  ret void\n}}\n\n",
            e = kernel.entry,
        ));
        let n = 10 + 4 * k;
        refs.push(format!("!{n}"));
        let inname = match kernel.input {
            Input::Half => "half",
            Input::Float => "float",
            Input::Packed => "uint",
        };
        let outname = match kernel.packs {
            true => "uint",
            false => fty,
        };
        metas.push_str(&format!(
            "!{n} = !{{ptr @{e}, !1, !{a}}}\n!{a} = !{{!{b}, !{c}, !2}}\n!{b} = !{{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 {insize}, !\"air.arg_type_align_size\", i32 {insize}, !\"air.arg_type_name\", !\"{inname}\", !\"air.arg_name\", !\"in\"}}\n!{c} = !{{i32 1, !\"air.buffer\", !\"air.location_index\", i32 1, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 {outsize}, !\"air.arg_type_align_size\", i32 {outsize}, !\"air.arg_type_name\", !\"{outname}\", !\"air.arg_name\", !\"out\"}}\n",
            e = kernel.entry,
            a = n + 1,
            b = n + 2,
            c = n + 3,
        ));
    }
    format!(
        "; Owned synthetic kernels.\nsource_filename = \"pack_conversions.metal\"\n\n{defs}{decls}\n!air.kernel = !{{{}}}\n!air.version = !{{!3}}\n!air.language_version = !{{!4}}\n!1 = !{{}}\n!2 = !{{i32 2, !\"air.thread_position_in_grid\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"gid\"}}\n!3 = !{{i32 2, i32 8, i32 0}}\n!4 = !{{!\"Metal\", i32 4, i32 0, i32 0}}\n{metas}",
        refs.join(", ")
    )
}

/// The packed words the unpacks read: every 16-bit value `i` spread as `i | i << 16`, which
/// gives every byte, every 10-bit field and every `rgb565` word every code.
fn packed_words() -> Vec<u32> {
    (0..=u32::from(u16::MAX)).map(|i| i | i << 16).collect()
}

/// The `float` words the `float` pack reads: every `half` value widened exactly, the 33 `float`s
/// around each tie of the 5- and 6-bit scales' products, NaNs and the specials.
fn float_words() -> Vec<u32> {
    let mut words: Vec<u32> = (0..=u16::MAX).map(widened).collect();
    for scale in [31u32, 63] {
        for k in 0..scale {
            let tie = ((f64::from(k) + 0.5) / f64::from(scale)) as f32;
            for d in -16i32..=16 {
                words.push(tie.to_bits().wrapping_add_signed(d));
            }
        }
    }
    words.extend([
        0x7fc0_0000,
        0xffc0_0001,
        0x7f80_0001,
        0x7f80_0000,
        0xff80_0000,
        0x0000_0001,
        0x8000_0001,
        0x3f80_0001,
        0x3f7f_ffff,
    ]);
    while words.len() % 64 != 0 {
        words.push(0x3f80_0000);
    }
    words
}

/// The exact `f32` bits of half `h`.
fn widened(h: u16) -> u32 {
    let sign = u32::from(h >> 15) << 31;
    let exp = u32::from((h >> 10) & 0x1f);
    let mant = u32::from(h & 0x3ff);
    match exp {
        0x1f => sign | 0x7f80_0000 | mant << 13,
        0 => sign | (mant as f32 * 2f32.powi(-24)).to_bits(),
        e => sign | (e + 112) << 23 | mant << 13,
    }
}

/// The IEEE binary16 bits of `x` (zero, or a normal `half` value in `[-1, 1]`) rounded to
/// nearest even once.
fn half_bits(x: f64) -> u16 {
    if x == 0.0 {
        return 0;
    }
    let sign = if x < 0.0 { 0x8000 } else { 0 };
    let a = x.abs();
    let e = ((a.to_bits() >> 52) & 0x7ff) as i32 - 1023;
    assert!(e >= -14, "{x} is not a normal half");
    let n = (a * 2f64.powi(10 - e)).round_ties_even() as i32;
    let (e, n) = if n == 2048 { (e + 1, 1024) } else { (e, n) };
    sign | (((e + 15) << 10) | (n - 1024)) as u16
}

/// A field's code of `x` (a NaN code 0): `x` clamped to `[0, 1]` (signed: `[-1, 1]`) times the
/// scale, rounded to nearest even, masked to the width; `product` rounds the product first.
fn code(x: f64, width: u32, signed: bool, product: impl Fn(f64) -> f64) -> u32 {
    if x.is_nan() {
        return 0;
    }
    let (lo, scale) = match signed {
        true => (-1.0, f64::from((1u32 << (width - 1)) - 1)),
        false => (0.0, f64::from((1u32 << width) - 1)),
    };
    let c = product(x.clamp(lo, 1.0) * scale).round_ties_even() as i64;
    (c & ((1i64 << width) - 1)) as u32
}

/// Metal's words for input word `x` of `kernel` (the module docs): an unpack's lanes, a pack's
/// one word.
fn metal(kernel: &Kernel, x: u32) -> Vec<u32> {
    let fields = kernel.fields();
    match (kernel.packs, kernel.input) {
        (false, _) => fields
            .iter()
            .map(|&(at, width, signed)| {
                let raw = (x >> at) & ((1u32 << width) - 1);
                match (kernel.half(), signed) {
                    (false, _) => {
                        // `float` `rgb565` alone: the code times the `float` reciprocal.
                        let scale = ((1u32 << width) - 1) as f32;
                        (raw as f32 * (1.0 / scale)).to_bits()
                    }
                    (true, false) => {
                        u32::from(half_bits(f64::from(raw) / f64::from((1u32 << width) - 1)))
                    }
                    (true, true) => {
                        let c = i64::from(raw) - i64::from(raw >> (width - 1)) * (1i64 << width);
                        let q = c as f64 / f64::from((1u32 << (width - 1)) - 1);
                        u32::from(half_bits(q.max(-1.0)))
                    }
                }
            })
            .collect(),
        (true, Input::Half) => {
            let h = f64::from(f32::from_bits(widened(x as u16)));
            vec![fields
                .iter()
                .map(|&(at, width, signed)| code(h, width, signed, |p| p) << at)
                .fold(0, |a, b| a | b)]
        }
        (true, _) => {
            let f = f64::from(f32::from_bits(x));
            vec![fields
                .iter()
                .map(|&(at, width, signed)| code(f, width, signed, |p| f64::from(p as f32)) << at)
                .fold(0, |a, b| a | b)]
        }
    }
}

/// `kernel`'s words on `device` for `input`: an unpack's four lanes per input (the unused ones
/// left as filled), a pack's one.
fn run(device: &Device, kernel: &Kernel, air: &[u8], input: &[u32]) -> Vec<u32> {
    let library = device.new_library_with_air(air).unwrap();
    let function = library.new_function_with_name(kernel.entry).unwrap();
    let pipeline = device
        .new_compute_pipeline_state(
            &ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor())),
            &PipelineOptions::default(),
        )
        .unwrap();
    let bytes: Vec<u8> = match kernel.input {
        Input::Half => input
            .iter()
            .flat_map(|&w| (w as u16).to_le_bytes())
            .collect(),
        _ => input.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let (per, size): (u64, u64) = match (kernel.packs, kernel.half()) {
        (true, _) => (1, 4),
        (false, true) => (4, 2),
        (false, false) => (4, 4),
    };
    let n = input.len() as u64;
    let input = buffer_with(device, &bytes).unwrap();
    let out = buffer_with(device, &vec![0x5a; (n * per * size) as usize]).unwrap();
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
    let got = read_buffer(&out, 0, n * per * size).unwrap();
    match size {
        2 => got
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| u32::from(u16::from_le_bytes(*b)))
            .collect(),
        _ => got
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| u32::from_le_bytes(*b))
            .collect(),
    }
}

/// Every kernel answers Metal's words on every lane, and records no
/// `PrecisionNote::DevicePackConversion`, where the profile takes
/// `DriverDeviation::MetalPackConversions` (`VkProperty::MetalPackConversions`); elsewhere each
/// records the note. Every code of every field, every `half` bit pattern, and 68,672 `float`
/// words (the module docs).
#[test]
fn measured_pack_conversions_are_metals_where_measured_and_noted_elsewhere() {
    let air = llvm_assemble_text(&library()).unwrap();
    let packed = packed_words();
    let halves: Vec<u32> = (0..=u32::from(u16::MAX)).collect();
    let floats = float_words();
    on_device(|device, _| {
        let measured = device
            .compile_device()
            .profile()
            .enabled()
            .properties
            .contains(&VkProperty::MetalPackConversions);
        let mut report = Vec::new();
        for kernel in &KERNELS {
            let notes = stage_notes(device, &air, kernel.entry);
            let noted = notes.contains(&PrecisionNote::DevicePackConversion);
            assert_eq!(noted, !measured, "{}: {notes:?}", kernel.entry);
            let input = match kernel.input {
                Input::Packed => &packed,
                Input::Half => &halves,
                Input::Float => &floats,
            };
            let got = run(device, kernel, &air, input);
            let per = got.len() / input.len();
            let mut differ = 0;
            for (i, &x) in input.iter().enumerate() {
                let want = metal(kernel, x);
                let lanes = &got[i * per..i * per + want.len()];
                if lanes == want.as_slice() {
                    continue;
                }
                assert!(
                    noted,
                    "{}: input {x:#x} answered {lanes:x?}, Metal's {want:x?}",
                    kernel.entry
                );
                differ += 1;
            }
            report.push(format!(
                "PACK-CONVERSIONS {} on {}: {differ} of {} inputs the device's, note {noted}",
                kernel.entry,
                device.name(),
                input.len()
            ));
        }
        for line in &report {
            println!("{line}");
        }
    });
}
