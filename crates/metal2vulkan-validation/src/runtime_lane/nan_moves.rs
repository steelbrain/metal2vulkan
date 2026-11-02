//! The NaNs a kernel moves rather than computes, on the device the cfg selects, under the
//! validation layer: a `half` widened to `float` from its bits, a `half` moved through a branch's
//! merge and to a later block and stored, a `float` negated and its magnitude taken, and a `half`'s
//! magnitude taken (`DriverDeviation::MetalNanMoves`). A merge's `half` the module holds in a
//! function's `float` variable, which the driver's compiler keeps in the registers.
//!
//! What plain Metal answers (our probe on the Apple M3 Max under macOS 27.0.1, with and without
//! fast math, the same AIR as this lane's): a `half` widened to `float` is its exact value, every
//! NaN `0x7fc00000`; a `half` moved and stored keeps its bits, a signalling or negative NaN's
//! included; a `float` `fneg` and `fabs` flip and clear the sign of every word, a NaN's payload
//! kept; a `half` `fabs` clears the sign, every NaN `0x7e00`. A `half` `fneg` answers every NaN as
//! `0x7e00` too, which a `float` read of it widens to `0x7fc00000`.
//!
//! The module holds each `half` widened to `float` and moves its NaN by the device's widening,
//! narrowing, negation and magnitude. Where the profile takes the deviation (MoltenVK on the Apple
//! M3 Max's macOS 27 identity), each kernel but the `half` `fneg`'s records no
//! `PrecisionNote::NativeNanEncoding` and every lane is Metal's word. Elsewhere a kernel that
//! converts records the note and its NaN lanes are the device's, which the test prints as
//! `NAN-MOVES` lines; its other
//! lanes are Metal's but a `float` denormal the device flushes where the stage records
//! `PrecisionNote::DeviceDenormals`. The `half` `fneg` read as `float` (`neg16w`) records the note
//! on every profile: the device negates the held `float`, whose NaN keeps its flipped sign unless
//! the driver's compiler moves the negation onto the `half` before the widening, as MoltenVK's did
//! here (every lane Metal's), which nothing states.

use super::degrade_notices::on_device;
use super::driver_deviations::stage_notes;
use super::{buffer_with, read_buffer, WAIT};
use crate::air_library::AirLibrary;
use metal2vulkan::device::VkProperty;
use metal2vulkan::pipeline::{ComputePipelineDescriptor, PrecisionNote, StageSource};
use metal2vulkan_devtools::llvm_assemble_text;
use metal2vulkan_runtime::{CommandBufferStatus, Device, PipelineOptions, Size};

/// A kernel of the lane: its entry, whether it reads `half` words (else `float` ones), whether it
/// writes `half` words, and its body between the load of `%x` and the store of `%r`.
struct Kernel {
    entry: &'static str,
    reads_half: bool,
    writes_half: bool,
    body: &'static str,
}

/// The kernels. `phi16` and `phi16x2` (a `half2`'s merge) read their second operand at `gid ^ 1`,
/// and they and `late16` branch on bit 1 of `gid`, so the `half` crosses a block before its store.
const KERNELS: [Kernel; 8] = [
    Kernel {
        entry: "widen16",
        reads_half: true,
        writes_half: false,
        body: "  %r = fpext half %x to float\n",
    },
    Kernel {
        entry: "phi16",
        reads_half: true,
        writes_half: true,
        body: "  %g1 = xor i32 %gid, 1\n  %i1 = zext i32 %g1 to i64\n  %q = getelementptr inbounds half, ptr addrspace(1) %in, i64 %i1\n  %y = load half, ptr addrspace(1) %q, align 2\n  %c = and i32 %gid, 2\n  %z = icmp eq i32 %c, 0\n  br i1 %z, label %left, label %right\nleft:\n  br label %join\nright:\n  br label %join\njoin:\n  %r = phi half [ %y, %left ], [ %x, %right ]\n",
    },
    Kernel {
        entry: "phi16x2",
        reads_half: true,
        writes_half: true,
        body: "  %g1 = xor i32 %gid, 1\n  %i1 = zext i32 %g1 to i64\n  %q = getelementptr inbounds half, ptr addrspace(1) %in, i64 %i1\n  %y = load half, ptr addrspace(1) %q, align 2\n  %va = insertelement <2 x half> undef, half %x, i32 0\n  %vb = insertelement <2 x half> %va, half %y, i32 1\n  %wa = insertelement <2 x half> undef, half %y, i32 0\n  %wb = insertelement <2 x half> %wa, half %x, i32 1\n  %c = and i32 %gid, 2\n  %z = icmp eq i32 %c, 0\n  br i1 %z, label %left, label %right\nleft:\n  br label %join\nright:\n  br label %join\njoin:\n  %v = phi <2 x half> [ %wb, %left ], [ %vb, %right ]\n  %r = extractelement <2 x half> %v, i32 0\n",
    },
    Kernel {
        entry: "late16",
        reads_half: true,
        writes_half: true,
        body: "  %c = and i32 %gid, 2\n  %z = icmp eq i32 %c, 0\n  br i1 %z, label %left, label %join\nleft:\n  br label %join\njoin:\n  %r = bitcast half %x to half\n",
    },
    Kernel {
        entry: "abs16",
        reads_half: true,
        writes_half: true,
        body: "  %r = call half @air.fabs.f16(half %x)\n",
    },
    Kernel {
        entry: "neg16w",
        reads_half: true,
        writes_half: false,
        body: "  %n = fneg half %x\n  %r = fpext half %n to float\n",
    },
    Kernel {
        entry: "neg32",
        reads_half: false,
        writes_half: false,
        body: "  %r = fneg float %x\n",
    },
    Kernel {
        entry: "abs32",
        reads_half: false,
        writes_half: false,
        body: "  %r = call float @air.fabs.f32(float %x)\n",
    },
];

/// The lane's AIR: every kernel of [`KERNELS`], buffer 0 its input words, buffer 1 its results.
fn library() -> String {
    let mut defs = String::new();
    let mut metas = String::new();
    let mut refs = Vec::new();
    for (k, kernel) in KERNELS.iter().enumerate() {
        let (inty, insize) = match kernel.reads_half {
            true => ("half", 2),
            false => ("float", 4),
        };
        let (outty, outsize) = match kernel.writes_half {
            true => ("half", 2),
            false => ("float", 4),
        };
        defs.push_str(&format!(
            "define void @{e}(ptr addrspace(1) %in, ptr addrspace(1) %out, i32 %gid) {{\nentry:\n  %i = zext i32 %gid to i64\n  %p = getelementptr inbounds {inty}, ptr addrspace(1) %in, i64 %i\n  %x = load {inty}, ptr addrspace(1) %p, align {insize}\n{body}  %o = getelementptr inbounds {outty}, ptr addrspace(1) %out, i64 %i\n  store {outty} %r, ptr addrspace(1) %o, align {outsize}\n  ret void\n}}\n\n",
            e = kernel.entry,
            body = kernel.body,
        ));
        let n = 10 + 4 * k;
        refs.push(format!("!{n}"));
        metas.push_str(&format!(
            "!{n} = !{{ptr @{e}, !1, !{a}}}\n!{a} = !{{!{b}, !{c}, !2}}\n!{b} = !{{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 {insize}, !\"air.arg_type_align_size\", i32 {insize}, !\"air.arg_type_name\", !\"{inty}\", !\"air.arg_name\", !\"in\"}}\n!{c} = !{{i32 1, !\"air.buffer\", !\"air.location_index\", i32 1, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 {outsize}, !\"air.arg_type_align_size\", i32 {outsize}, !\"air.arg_type_name\", !\"{outty}\", !\"air.arg_name\", !\"out\"}}\n",
            e = kernel.entry,
            a = n + 1,
            b = n + 2,
            c = n + 3,
        ));
    }
    format!(
        "; Owned synthetic kernels.\nsource_filename = \"nan_moves.metal\"\n\n{defs}declare half @air.fabs.f16(half)\ndeclare float @air.fabs.f32(float)\n\n!air.kernel = !{{{}}}\n!air.version = !{{!3}}\n!air.language_version = !{{!4}}\n!1 = !{{}}\n!2 = !{{i32 2, !\"air.thread_position_in_grid\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"gid\"}}\n!3 = !{{i32 2, i32 8, i32 0}}\n!4 = !{{!\"Metal\", i32 4, i32 0, i32 0}}\n{metas}",
        refs.join(", ")
    )
}

/// The `float` words the `float` kernels read: every `half` value widened exactly, NaNs of both
/// signs over every top-12 payload bits and four low patterns, and the specials.
fn float_words() -> Vec<u32> {
    let mut words: Vec<u32> = (0..=u16::MAX).map(widened).collect();
    for sign in [0, 0x8000_0000] {
        for high in 0..4096u32 {
            for low in [0, 1, 0x400, 0x7ff] {
                let payload = match high << 11 | low {
                    0 => 1 << 22,
                    p => p,
                };
                words.push(sign | 0x7f80_0000 | payload);
            }
        }
    }
    words.extend([
        0,
        0x8000_0000,
        1,
        0x8000_0001,
        0x007f_ffff,
        0x0080_0000,
        0x7f7f_ffff,
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

/// Whether half `h` is a NaN.
fn half_nan(h: u32) -> bool {
    h & 0x7c00 == 0x7c00 && h & 0x3ff != 0
}

/// Whether `float` word `w` is a NaN.
fn float_nan(w: u32) -> bool {
    w & 0x7f80_0000 == 0x7f80_0000 && w & 0x007f_ffff != 0
}

/// Metal's word for input word `x` of `kernel` (the module docs).
fn metal(kernel: &str, x: u32) -> u32 {
    match kernel {
        "widen16" if half_nan(x) => 0x7fc0_0000,
        "widen16" => widened(x as u16),
        "phi16" | "phi16x2" | "late16" => x,
        "abs16" if half_nan(x) => 0x7e00,
        "abs16" => x & 0x7fff,
        "neg16w" if half_nan(x) => 0x7fc0_0000,
        "neg16w" => widened(x as u16 ^ 0x8000),
        "neg32" => x ^ 0x8000_0000,
        "abs32" => x & 0x7fff_ffff,
        _ => unreachable!("{kernel}"),
    }
}

/// The input word lane `i` of `kernel` stores: a merge's lanes whose bit 1 is clear take their
/// neighbour's.
fn source(kernel: &str, i: usize) -> usize {
    match kernel {
        "phi16" | "phi16x2" if i & 2 == 0 => i ^ 1,
        _ => i,
    }
}

/// `kernel`'s words on `device` for `input`.
fn run(device: &Device, kernel: &Kernel, air: &[u8], input: &[u32]) -> Vec<u32> {
    let library = device.new_library_with_air(air).unwrap();
    let function = library.new_function_with_name(kernel.entry).unwrap();
    let pipeline = device
        .new_compute_pipeline_state(
            &ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor())),
            &PipelineOptions::default(),
        )
        .unwrap();
    let bytes: Vec<u8> = match kernel.reads_half {
        true => input
            .iter()
            .flat_map(|&w| (w as u16).to_le_bytes())
            .collect(),
        false => input.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let out_size: u64 = if kernel.writes_half { 2 } else { 4 };
    let n = input.len() as u64;
    let input = buffer_with(device, &bytes).unwrap();
    let out = buffer_with(device, &vec![0x5a; (n * out_size) as usize]).unwrap();
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
    let got = read_buffer(&out, 0, n * out_size).unwrap();
    match kernel.writes_half {
        true => got
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| u32::from(u16::from_le_bytes(*b)))
            .collect(),
        false => got
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| u32::from_le_bytes(*b))
            .collect(),
    }
}

/// Every kernel answers Metal's word on every lane where its stage records no
/// `PrecisionNote::NativeNanEncoding`, and its NaN lanes are the device's where it records the
/// note; where the profile takes `DriverDeviation::MetalNanMoves` (`VkProperty::MetalNanMoves`)
/// no stage records it but the `half` `fneg` read as `float`, which records it on every profile.
/// Every `half` bit pattern, and 98,368 `float` words (the module docs).
#[test]
fn moved_nans_are_metals_where_measured_and_noted_elsewhere() {
    let air = llvm_assemble_text(&library()).unwrap();
    let halves: Vec<u32> = (0..=u32::from(u16::MAX)).collect();
    let floats = float_words();
    on_device(|device, _| {
        let moves = device
            .compile_device()
            .profile()
            .enabled()
            .properties
            .contains(&VkProperty::MetalNanMoves);
        let mut report = Vec::new();
        for kernel in &KERNELS {
            let notes = stage_notes(device, &air, kernel.entry);
            let noted = notes.contains(&PrecisionNote::NativeNanEncoding);
            let flushes = notes.contains(&PrecisionNote::DeviceDenormals);
            // The `half` `fneg` is noted on every profile; nothing else where the moves are
            // measured.
            let always = kernel.entry == "neg16w";
            assert!(!always || noted, "{}: {notes:?}", kernel.entry);
            assert!(
                !moves || noted == always,
                "{}: NativeNanEncoding recorded {noted} where the profile takes MetalNanMoves",
                kernel.entry
            );
            let input = if kernel.reads_half { &halves } else { &floats };
            let got = run(device, kernel, &air, input);
            let nan = |x: u32| match kernel.reads_half {
                true => half_nan(x),
                false => float_nan(x),
            };
            let denormal =
                |x: u32| !kernel.reads_half && x & 0x7f80_0000 == 0 && x & 0x7f_ffff != 0;
            let mut nan_differ = 0;
            for (i, &g) in got.iter().enumerate() {
                let x = input[source(kernel.entry, i)];
                let want = metal(kernel.entry, x);
                if g == want {
                    continue;
                }
                if nan(x) && noted {
                    nan_differ += 1;
                    continue;
                }
                assert!(
                    flushes && denormal(x),
                    "{}: input {x:#x} answered {g:#x}, Metal's {want:#x} (notes {notes:?})",
                    kernel.entry
                );
            }
            report.push(format!(
                "NAN-MOVES {} on {}: {nan_differ} NaN lanes of {} the device's, note {noted}",
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
