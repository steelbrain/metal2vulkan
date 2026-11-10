//! What a kernel stores when it writes a `float` to a normalized storage texture, against Metal's
//! measured conversion, on the device the cfg selects, under the validation layer.
//!
//! Metal stores a kernel's `float` write through `access::write` and `access::read_write` as a
//! fragment function's write to a color attachment of the format (`color_writes`): a unorm channel
//! of `b` bits `x` clamped to `[0, 1]` (a NaN 0) times `2^b - 1` exactly, rounded to nearest with a
//! tie up; a snorm channel likewise on `[-1, 1]`, a tie away from zero. A packed 16-bit format's
//! channel (`B5G6R5Unorm`, `A1BGR5Unorm`, `ABGR4Unorm`, `BGR5A1Unorm`) takes the same exact
//! product, not the `half` a color attachment rounds through first, and its 1-bit channel stores 1
//! only above 0.5 (an Apple M4 under macOS 27.0.1: every code's value and every
//! tie between codes with two `f32` steps either side, per channel, of the 8- and 16-bit unorm and
//! snorm, the 10:10:10:2 and the packed 16-bit formats, both accesses alike).
//!
//! Each case writes `color_writes`' probe values for the format (every channel of a texel the same
//! value) through `half_storage_writes`' kernels (`access::write`, `access::read_write`, a texture
//! buffer and an argument buffer's texture), reads the texels back and compares every channel with
//! Metal's code. A case whose format the runtime refuses for storage is printed and skipped. A
//! channel that differs is counted, with examples, on a `NORM-STORAGE-WRITE` line. The kernel
//! writes the value as computed and the device converts it: the lane holds a channel to Metal's
//! code where the device is measured to store it (`DriverDeviation::MetalNormStorageWrites`, but
//! for a 1-bit channel, which RADV rounds up at one half), the measurement of the deviation on
//! that device; elsewhere the module records the device's conversion
//! (`PrecisionNote::DeviceColorConversion`) and the lane prints the channels that differ without
//! holding them.
//!
//! [`kernel_writes_store_each_channel_in_its_field`] writes texels whose channels differ into the
//! 10:10:10:2 formats, whose fields Metal orders red first (`RGB10A2Unorm`) and blue first
//! (`BGR10A2Unorm`, which NVIDIA and lavapipe store through an `RGB10A2Unorm` view whose red and
//! blue the shader exchanges), and holds each field to its channel's code.

use super::at_plan;
use super::color_writes::{field, metal, round_product, Channel, Output, CASES};
use super::degrade_notices::on_device_with;
use super::half_storage_writes::{written, written_texels, Form};
use super::layer_reported_nothing;
use metal2vulkan::error::DriverDeviation;
use metal2vulkan::metal::PixelFormat;

/// Metal's stored code of `x` in `channel` (the module docs); `None` for a channel of another
/// kind (sRGB, half-float and small-float channels have lanes of their own).
fn stored_code(channel: Channel, x: f32) -> Option<u64> {
    match channel {
        Channel::Unorm(_) | Channel::Snorm(_) => Some(metal(channel, x, false)),
        Channel::HalfUnorm(1) => Some(u64::from(x > 0.5)),
        Channel::HalfUnorm(bits) => {
            let unit = if x >= 0.0 { x.min(1.0) } else { 0.0 };
            Some(u64::from(round_product(unit, (1 << bits) - 1)))
        }
        _ => None,
    }
}

/// Every form into every normalized format of `color_writes`' cases, printed against Metal's
/// conversion and held to it where the device is measured to store it (the module docs). The cases
/// share one device (they read no notice), and each case's validation messages fail it by name.
#[test]
fn kernel_writes_store_metals_normalized_codes() {
    let mut differ = Vec::new();
    let normalized = CASES.iter().filter(|c| {
        c.output == Output::Float
            && c.channels.iter().all(|&(_, channel)| {
                matches!(
                    channel,
                    Channel::Unorm(_) | Channel::Snorm(_) | Channel::HalfUnorm(_)
                )
            })
    });
    on_device_with(at_plan, |device, messages| {
        let name = device.name().to_owned();
        let measured = device
            .compile_device()
            .profile()
            .takes_driver_deviation(DriverDeviation::MetalNormStorageWrites);
        for case in normalized {
            let values = (case.values)();
            for form in Form::ALL {
                let label = format!("{:?} by {}", case.pf, form.kernel());
                let stored = written(device, form, case.pf, case.bytes, &values);
                layer_reported_nothing(messages, &label);
                let Some(stored) = stored else {
                    eprintln!("NORM-STORAGE-WRITE {label} on {name}: refused");
                    continue;
                };
                let (mut wrong, mut held, mut examples) = (0usize, 0usize, Vec::new());
                for (i, &x) in values.iter().enumerate() {
                    let texel = &stored[i * case.bytes..(i + 1) * case.bytes];
                    for &(offset, channel) in case.channels {
                        let Some(want) = stored_code(channel, x) else {
                            continue;
                        };
                        let got = field(texel, offset, channel.bits());
                        if got == want {
                            continue;
                        }
                        wrong += 1;
                        if measured && !matches!(channel, Channel::HalfUnorm(1)) {
                            held += 1;
                        }
                        if examples.len() < 8 {
                            examples.push(format!(
                                "{x:e} ({:#010x}) at bit {offset}: {got:#x}, Metal's {want:#x}",
                                x.to_bits()
                            ));
                        }
                    }
                }
                let channels = values.len() * case.channels.len();
                eprintln!(
                    "NORM-STORAGE-WRITE {label} on {name}: {wrong} of {channels} channels differ \
                     from Metal's ({held} held to it): {}",
                    examples.join("; ")
                );
                if held > 0 {
                    differ.push(format!("{label}: {held} of {channels}"));
                }
            }
        }
    });
    assert!(
        differ.is_empty(),
        "kernel normalized writes not Metal's: {differ:#?}"
    );
}

/// Each 10:10:10:2 format's fields by Metal's channel (red, green, blue, alpha): the bit each
/// starts at, as Metal's clear of either format packs it (`runtime::clear_value`'s measured
/// texels: `BGR10A2Unorm`'s blue in the low ten bits).
const FIELDS: [(PixelFormat, [u32; 4]); 2] = [
    (PixelFormat::RGB10A2Unorm, [0, 10, 20, 30]),
    (PixelFormat::BGR10A2Unorm, [20, 10, 0, 30]),
];

/// Every form's write of texels whose red, green, blue and alpha are distinct codes' values stores
/// each channel's code in its own field of both 10:10:10:2 formats. The cases share one device
/// (they read no notice), and each case's validation messages fail it by name.
#[test]
fn kernel_writes_store_each_channel_in_its_field() {
    let texels: Vec<[f32; 4]> = (0..1024u32)
        .map(|i| {
            let code = |k: u32| (i * k + k) % 1024;
            [
                code(7) as f32 / 1023.0,
                code(13) as f32 / 1023.0,
                code(29) as f32 / 1023.0,
                (i % 4) as f32 / 3.0,
            ]
        })
        .collect();
    let mut differ = Vec::new();
    on_device_with(at_plan, |device, messages| {
        let name = device.name().to_owned();
        for (pf, fields) in FIELDS {
            for form in Form::ALL {
                let label = format!("{pf:?} by {}", form.kernel());
                let stored = written_texels(device, form, pf, 4, &texels);
                layer_reported_nothing(messages, &label);
                let Some(stored) = stored else {
                    eprintln!("NORM-STORAGE-FIELDS {label} on {name}: refused");
                    continue;
                };
                let (mut wrong, mut examples) = (0usize, Vec::new());
                for (texel, values) in stored.as_chunks::<4>().0.iter().zip(&texels) {
                    for (c, (&x, &offset)) in values.iter().zip(&fields).enumerate() {
                        let channel = Channel::Unorm(if c == 3 { 2 } else { 10 });
                        let want = stored_code(channel, x).expect("a unorm channel");
                        let got = field(texel, offset, channel.bits());
                        if got != want {
                            wrong += 1;
                            if examples.len() < 8 {
                                examples.push(format!(
                                    "channel {c} {x:e} at bit {offset}: {got:#x}, Metal's {want:#x}"
                                ));
                            }
                        }
                    }
                }
                eprintln!(
                    "NORM-STORAGE-FIELDS {label} on {name}: {wrong} of {} channels differ from \
                     Metal's: {}",
                    4 * texels.len(),
                    examples.join("; ")
                );
                if wrong > 0 {
                    differ.push(format!("{label}: {wrong}"));
                }
            }
        }
    });
    assert!(differ.is_empty(), "fields not Metal's: {differ:#?}");
}
