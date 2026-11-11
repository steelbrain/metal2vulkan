//! What a kernel stores when it writes a `float` to a half-float storage texture (`R16Float`,
//! `RG16Float`, `RGBA16Float`), against Metal's measured conversion, on the device the cfg
//! selects, under the validation layer.
//!
//! Metal rounds each channel toward zero, stores a finite value past 65504 as 65504, keeps an
//! infinity, and stores every NaN as `0x7e00` (an M4 under macOS 27.0.1: 380,972
//! of 380,972 non-NaN channels per format, every NaN of 16 payloads and both signs; Shared and
//! Private, plain and under the debug layer, alike): [`half_toward_zero`], the fragment write's
//! rule.
//!
//! The kernels are the owned fixture `hsw_storage_writes`'s, each writing buffer 0's `float4` at
//! its thread's texel: `hsw_write` through `access::write`, `hsw_read_write` through
//! `access::read_write`, `hsw_buffer` through a texture buffer, and `hsw_held` through an argument
//! buffer's texture. Each case writes the values of `color_writes`' half-float probe (every finite
//! half and every tie between neighbouring halves with one `f32` step either side, of either
//! sign) and its specials (zeros, infinities, past the largest, denormals, NaNs of four
//! payloads), every channel of a texel the same value, reads the texels back and compares every
//! channel with Metal's code. The NaNs a device stores are printed per input; a NaN channel
//! stored as another NaN is counted apart, not as a failure, where the device announces
//! `device-nan-encoding` (the recorded near-equivalent).
//!
//! The product writes as computed, the device's conversion rounding every value and storing every
//! NaN. A case is held to Metal's rounding where the device is measured to make it
//! (`DriverDeviation::MetalHalfStorageWrites`: RADV, ANV, NVIDIA's measured GPUs, lavapipe from
//! Mesa 26.0.8 through 26.2.4 and MoltenVK), the measurement of the deviation on that device;
//! elsewhere the module records the device's conversion (`PrecisionNote::DeviceColorConversion`),
//! and the lane prints the channels that differ without holding them. A NaN is the device's,
//! announced as `device-nan-encoding` where the profile does not take `MetalHalfStorageNans` (ANV
//! stores `0xfe00`).

use super::color_writes::{
    half, half_toward_zero, specials, unsigned_float, unsigned_float_toward_zero,
};
use super::degrade_notices::{on_device, on_device_with};
use super::{at_plan, buffer_with, layer_reported_nothing, read_buffer, Messages, WAIT};
use crate::air_library::AirLibrary;
use metal2vulkan::error::DriverDeviation;
use metal2vulkan::metal::{PixelFormat, ResourceUsage, TextureUsage};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    CommandBufferStatus, ComputePipelineState, Degradation, Device, PipelineOptions, Region,
    ResourceOptions, Size, Texture, TextureDescriptor,
};
use std::collections::BTreeMap;

/// A 2D texture's row of texels.
const WIDTH: u64 = 1024;

/// The texels of one texture-buffer dispatch: Vulkan's least `maxTexelBufferElements`.
const BUFFER_TEXELS: usize = 65536;

/// How a case's kernel reaches the texture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Form {
    Write,
    ReadWrite,
    Buffer,
    Held,
}

impl Form {
    pub(super) const ALL: [Form; 4] = [Form::Write, Form::ReadWrite, Form::Buffer, Form::Held];

    pub(super) fn kernel(self) -> &'static str {
        match self {
            Form::Write => "hsw_write",
            Form::ReadWrite => "hsw_read_write",
            Form::Buffer => "hsw_buffer",
            Form::Held => "hsw_held",
        }
    }
}

/// The formats and their channels.
const FORMATS: [(PixelFormat, usize); 3] = [
    (PixelFormat::R16Float, 1),
    (PixelFormat::RG16Float, 2),
    (PixelFormat::RGBA16Float, 4),
];

fn state(device: &Device, kernel: &str) -> ComputePipelineState {
    let bitcode = llvm_assemble(&engine_dir().join("hsw_storage_writes.ll")).unwrap();
    let library = device.new_library_with_air(&bitcode).unwrap();
    let function = library.new_function_with_name(kernel).unwrap();
    device
        .new_compute_pipeline_state(
            &ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor())),
            &PipelineOptions::default(),
        )
        .unwrap()
}

/// `values` as `float4`s, padded with zeros to `texels`.
fn texel_values(values: &[[f32; 4]], texels: usize) -> Vec<u8> {
    (0..texels)
        .flat_map(|i| values.get(i).copied().unwrap_or([0.0; 4]))
        .flat_map(f32::to_le_bytes)
        .collect()
}

/// Run `kernel`'s `state` over `groups` threadgroups of `per_group` threads with `values` at
/// buffer 0, `width` at buffer 1 and `texture` bound as `form` binds it, and wait for it; the
/// dispatch's refusal where the device cannot bind the texture so (its format's storage use).
fn dispatch(
    device: &Device,
    (state, form): (&ComputePipelineState, Form),
    texture: &Texture,
    values: &[u8],
    (groups, per_group): (Size, Size),
) -> Result<(), String> {
    let values = buffer_with(device, values).unwrap();
    let width = buffer_with(device, &(WIDTH as u32).to_le_bytes()).unwrap();
    let held = buffer_with(device, &texture.gpu_resource_id().0.to_le_bytes()).unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(state).unwrap();
    encoder.set_buffer(Some(&values), 0, 0).unwrap();
    match form {
        Form::Held => {
            encoder.set_buffer(Some(&width), 0, 1).unwrap();
            encoder.set_buffer(Some(&held), 0, 2).unwrap();
            encoder
                .use_resource(texture.as_resource(), ResourceUsage::WRITE)
                .unwrap();
        }
        Form::Buffer => encoder.set_texture(Some(texture), 0).unwrap(),
        Form::Write | Form::ReadWrite => {
            encoder.set_buffer(Some(&width), 0, 1).unwrap();
            encoder.set_texture(Some(texture), 0).unwrap();
        }
    }
    encoder
        .dispatch_threadgroups(groups, per_group)
        .map_err(|e| e.to_string())?;
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    assert_eq!(command_buffer.error(), None);
    Ok(())
}

/// The bytes `form`'s kernel stores for `values` (one per texel, in each of its lanes) in a
/// texture of `pf`, `bytes` a texel; `None` where the runtime refuses the texture or texture
/// buffer of `pf` (its error printed).
pub(super) fn written(
    device: &Device,
    form: Form,
    pf: PixelFormat,
    bytes: usize,
    values: &[f32],
) -> Option<Vec<u8>> {
    let texels: Vec<[f32; 4]> = values.iter().map(|&x| [x; 4]).collect();
    written_texels(device, form, pf, bytes, &texels)
}

/// The bytes `form`'s kernel stores for `values` (one `float4` per texel) in a texture of `pf`,
/// `bytes` a texel; `None` where the runtime refuses the texture or texture buffer of `pf` (its
/// error printed).
pub(super) fn written_texels(
    device: &Device,
    form: Form,
    pf: PixelFormat,
    bytes: usize,
    values: &[[f32; 4]],
) -> Option<Vec<u8>> {
    let state = state(device, form.kernel());
    let usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
    if form == Form::Buffer {
        let mut stored = Vec::with_capacity(values.len() * bytes);
        for chunk in values.chunks(BUFFER_TEXELS) {
            let texels = chunk.len().div_ceil(64) * 64;
            let row = (texels * bytes) as u64;
            let backing = device.new_buffer(row, ResourceOptions::shared()).unwrap();
            let d = TextureDescriptor::texture_buffer(
                pf,
                texels as u64,
                ResourceOptions::shared(),
                usage,
            );
            let texture = match backing.new_texture(&d, 0, row) {
                Ok(texture) => texture,
                Err(e) => {
                    eprintln!("a texture buffer of {pf:?}: refused, {e}");
                    return None;
                }
            };
            let groups = Size::new(texels as u64 / 64, 1, 1);
            if let Err(e) = dispatch(
                device,
                (&state, form),
                &texture,
                &texel_values(chunk, texels),
                (groups, Size::new(64, 1, 1)),
            ) {
                eprintln!("a texture buffer of {pf:?}: dispatch refused, {e}");
                return None;
            }
            stored.extend(read_buffer(&backing, 0, (chunk.len() * bytes) as u64).unwrap());
        }
        return Some(stored);
    }
    let height = (values.len() as u64).div_ceil(WIDTH).div_ceil(16) * 16;
    let mut d = TextureDescriptor::texture_2d(pf, WIDTH, height, false);
    d.usage = usage;
    d.resource_options = ResourceOptions::shared();
    let texture = match device.new_texture(&d) {
        Ok(texture) => texture,
        Err(e) => {
            eprintln!("a storage texture of {pf:?}: refused, {e}");
            return None;
        }
    };
    let groups = Size::new(WIDTH / 16, height / 16, 1);
    if let Err(e) = dispatch(
        device,
        (&state, form),
        &texture,
        &texel_values(values, (WIDTH * height) as usize),
        (groups, Size::new(16, 16, 1)),
    ) {
        eprintln!("a storage texture of {pf:?}: dispatch refused, {e}");
        return None;
    }
    let row = WIDTH * bytes as u64;
    let mut stored = vec![0u8; (row * height) as usize];
    texture
        .get_bytes(
            &mut stored,
            row,
            0,
            Region::new_2d(0, 0, WIDTH, height),
            0,
            0,
        )
        .unwrap();
    stored.truncate(values.len() * bytes);
    Some(stored)
}

/// Whether half bits `h` are a NaN.
fn nan(h: u16) -> bool {
    h & 0x7c00 == 0x7c00 && h & 0x3ff != 0
}

/// Every form into every format, printed against Metal's conversion and held to it where the
/// device is measured to make it (the module docs). Every case
/// runs on one device, which announces `DeviceNanEncoding` at each dispatch that binds a variant
/// taking it (a device announces a kind once unless its options ask for each occurrence), so
/// each case reads its own notice as on a device of its own; its validation messages fail it by
/// name.
#[test]
fn kernel_writes_store_metals_halves() {
    let values = [half(), specials()].concat();
    let mut differ = Vec::new();
    on_device_with(
        |options| {
            options.announce_each.insert(Degradation::DeviceNanEncoding);
            at_plan(options)
        },
        |device, messages| {
            for form in Form::ALL {
                for (pf, channels) in FORMATS {
                    if let Some(d) = judge(device, messages, (form, pf, channels), &values) {
                        differ.push(d);
                    }
                    layer_reported_nothing(messages, &format!("{pf:?} by {}", form.kernel()));
                }
            }
        },
    );
    assert!(
        differ.is_empty(),
        "kernel half writes not Metal's: {differ:#?}"
    );
}

/// `form`'s write of `values` into `pf` (`channels` a texel) on `device`, printed against
/// Metal's conversion: `None` where every channel is Metal's (a NaN stored as another NaN
/// counted apart where the case announced `device-nan-encoding`), else the case and its count.
fn judge(
    device: &Device,
    messages: &Messages,
    (form, pf, channels): (Form, PixelFormat, usize),
    values: &[f32],
) -> Option<String> {
    let name = device.name().to_owned();
    let label = format!("{pf:?} by {}", form.kernel());
    messages.take_notices();
    let stored = written(device, form, pf, 2 * channels, values).expect("a half-float texture");
    let noted = messages.take_notices().iter().any(|m| {
        m.degradation
            .as_ref()
            .is_some_and(|n| n.degradation == Degradation::DeviceNanEncoding)
    });
    let (mut wrong, mut away, mut examples) = (0usize, 0usize, Vec::new());
    let (mut nans, mut stores) = (0usize, BTreeMap::new());
    for (i, &x) in values.iter().enumerate() {
        for c in 0..channels {
            let at = (i * channels + c) * 2;
            let got = u16::from_le_bytes([stored[at], stored[at + 1]]);
            let want = half_toward_zero(x);
            if x.is_nan() {
                stores.entry(x.to_bits()).or_insert(got);
            }
            if got == want {
                continue;
            }
            if noted && nan(got) && nan(want) {
                nans += 1;
                continue;
            }
            wrong += 1;
            // The device's code one step further from zero: rounded to nearest.
            away += usize::from(got == want.wrapping_add(1) && !x.is_nan());
            if examples.len() < 6 {
                examples.push(format!(
                    "{x:e} ({:#010x}): {got:#06x}, Metal's {want:#06x}",
                    x.to_bits()
                ));
            }
        }
    }
    let stores = stores
        .iter()
        .map(|(x, h)| format!("{x:#010x} as {h:#06x}"))
        .collect::<Vec<_>>()
        .join(", ");
    eprintln!("kernel half write {label} on {name}: NaNs stored {stores}");
    if nans > 0 {
        eprintln!(
            "kernel half write {label} on {name}: {nans} NaN channels store another NaN \
             (DeviceNanEncoding announced)"
        );
    }
    let channels_written = values.len() * channels;
    if wrong == 0 {
        eprintln!(
            "kernel half write {label} on {name}: Metal's, 0 of {channels_written} channels \
             differ"
        );
        return None;
    }
    eprintln!(
        "kernel half write {label} on {name}: {wrong} of {channels_written} channels differ, \
         {away} of them one step away from zero: {}",
        examples.join("; ")
    );
    let measured = device
        .compile_device()
        .profile()
        .takes_driver_deviation(DriverDeviation::MetalHalfStorageWrites);
    measured.then(|| format!("{label}: {wrong} of {channels_written}"))
}

/// `RG11B10Float`'s fields: each one's offset in the texel and fraction bits.
const SMALL_FLOAT_FIELDS: [(u32, u32); 3] = [(0, 6), (11, 6), (22, 5)];

/// Whether unsigned small-float code `code` of `mantissa` fraction bits is a NaN.
fn small_nan(code: u32, mantissa: u32) -> bool {
    code >> mantissa == 0x1f && code & ((1 << mantissa) - 1) != 0
}

/// A kernel's `float` write to an `RG11B10Float` storage texture through every form, against
/// Metal's conversion: every field rounded toward zero, a negative value 0, a finite value past
/// the largest the largest, an infinity kept, a NaN `0x7e0` or `0x3f0` (our probe and the
/// M3 Max's m3pkg-twr, alike: the kernel's, a fragment function's and a color attachment's
/// writes at the default rounding mode, byte for byte). The values are every finite code's and
/// every tie between neighbouring codes, with one `f32` step either side, of both field widths,
/// their negations, and the specials. A NaN field stored as another NaN counts apart where the
/// case announced `device-nan-encoding`.
#[test]
fn kernel_writes_store_metals_small_floats() {
    let values = [unsigned_float(6), unsigned_float(5), specials()].concat();
    let mut differ = Vec::new();
    for form in Form::ALL {
        on_device(|device, messages| {
            let name = device.name().to_owned();
            let label = format!("RG11B10Float by {}", form.kernel());
            messages.take_notices();
            let Some(stored) = written(device, form, PixelFormat::RG11B10Float, 4, &values) else {
                return;
            };
            let noted = messages.take_notices().iter().any(|m| {
                m.degradation
                    .as_ref()
                    .is_some_and(|n| n.degradation == Degradation::DeviceNanEncoding)
            });
            let (mut wrong, mut nearest, mut nans, mut examples) = (0, 0, 0, Vec::new());
            let mut stores = BTreeMap::new();
            for (i, &x) in values.iter().enumerate() {
                let texel = u32::from_le_bytes(stored[i * 4..i * 4 + 4].try_into().unwrap());
                for (offset, mantissa) in SMALL_FLOAT_FIELDS {
                    let bits = mantissa + 5;
                    let got = (texel >> offset) & ((1 << bits) - 1);
                    let want = unsigned_float_toward_zero(x, mantissa) as u32;
                    if x.is_nan() {
                        stores.entry((x.to_bits(), offset)).or_insert(got);
                    }
                    if got == want {
                        continue;
                    }
                    if noted && small_nan(got, mantissa) && small_nan(want, mantissa) {
                        nans += 1;
                        continue;
                    }
                    wrong += 1;
                    // The device's code one step further from zero: rounded to nearest.
                    nearest += usize::from(got == want + 1 && !x.is_nan());
                    if examples.len() < 6 {
                        examples.push(format!(
                            "{x:e} ({:#010x}) field {offset}: {got:#x}, Metal's {want:#x}",
                            x.to_bits()
                        ));
                    }
                }
            }
            let stores = stores
                .iter()
                .map(|((x, offset), c)| format!("{x:#010x} field {offset} as {c:#x}"))
                .collect::<Vec<_>>()
                .join(", ");
            eprintln!("kernel small-float write {label} on {name}: NaNs stored {stores}");
            if nans > 0 {
                eprintln!(
                    "kernel small-float write {label} on {name}: {nans} NaN fields store another \
                     NaN (DeviceNanEncoding announced)"
                );
            }
            let fields = values.len() * SMALL_FLOAT_FIELDS.len();
            if wrong == 0 {
                eprintln!(
                    "kernel small-float write {label} on {name}: Metal's, 0 of {fields} fields \
                     differ"
                );
                return;
            }
            eprintln!(
                "kernel small-float write {label} on {name}: {wrong} of {fields} fields differ, \
                 {nearest} of them one step away from zero: {}",
                examples.join("; ")
            );
            differ.push(format!("{label}: {wrong} of {fields}"));
        });
    }
    assert!(
        differ.is_empty(),
        "kernel small-float writes not Metal's: {differ:#?}"
    );
}

/// Every NaN class of a `float` that a kernel writes to an `RG11B10Float` storage texture, through
/// every form: each of the 8,192 patterns of a NaN's top 13 payload bits, its 10 low bits clear and
/// set (pattern 0 with them set alone, which is no infinity), of both signs, every channel of a
/// texel the same NaN. Metal stores each field as its canonical NaN (`0x7e0`, `0x3f0`), whatever
/// the NaN's sign and payload (`DriverDeviation::MetalSmallFloatNans`'s measurement on the M3
/// Max). On a device that takes that deviation every field is Metal's and nothing announces
/// `device-nan-encoding`: the deviation's gate on each driver it lists. Elsewhere the fields
/// stored as another code are counted and printed, and where any is, the case announces
/// `device-nan-encoding`.
#[test]
fn every_nan_a_kernel_writes_to_a_small_float_texture_is_metal_s_where_measured() {
    let mut values = Vec::new();
    for sign in [0, 0x8000_0000u32] {
        for top in 0..1u32 << 13 {
            for low in [0, 0x3ff] {
                if top << 10 | low != 0 {
                    values.push(f32::from_bits(sign | 0x7f80_0000 | top << 10 | low));
                }
            }
        }
    }
    let mut failed = Vec::new();
    for form in Form::ALL {
        on_device(|device, messages| {
            let name = device.name().to_owned();
            let label = format!("RG11B10Float by {}", form.kernel());
            messages.take_notices();
            let Some(stored) = written(device, form, PixelFormat::RG11B10Float, 4, &values) else {
                return;
            };
            let noted = messages.take_notices().iter().any(|m| {
                m.degradation
                    .as_ref()
                    .is_some_and(|n| n.degradation == Degradation::DeviceNanEncoding)
            });
            let measured = device
                .compile_device()
                .profile()
                .takes_driver_deviation(DriverDeviation::MetalSmallFloatNans);
            // Each code stored in place of the field's canonical NaN, and how many times.
            let mut other = BTreeMap::new();
            for (i, x) in values.iter().enumerate() {
                assert!(x.is_nan());
                let texel = u32::from_le_bytes(stored[i * 4..i * 4 + 4].try_into().unwrap());
                for (offset, mantissa) in SMALL_FLOAT_FIELDS {
                    let got = (texel >> offset) & ((1 << (mantissa + 5)) - 1);
                    if got != (0x1f << mantissa | 1 << (mantissa - 1)) {
                        *other.entry((offset, got)).or_insert(0usize) += 1;
                    }
                }
            }
            let differ: usize = other.values().sum();
            let fields = values.len() * SMALL_FLOAT_FIELDS.len();
            eprintln!(
                "every small-float NaN {label} on {name}: {differ} of {fields} NaN fields not \
                 Metal's {other:x?}; MetalSmallFloatNans {measured}, device-nan-encoding {noted}"
            );
            if (measured && (differ > 0 || noted)) || (differ > 0 && !noted) {
                failed.push(format!(
                    "{label}: {differ} of {fields}, measured {measured}, announced {noted}"
                ));
            }
        });
    }
    assert!(
        failed.is_empty(),
        "small-float NaN writes against MetalSmallFloatNans: {failed:#?}"
    );
}
