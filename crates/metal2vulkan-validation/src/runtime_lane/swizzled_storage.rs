//! Swizzled textures read and written through storage slots on the device the cfg selects, under
//! the validation layer.
//!
//! What plain Metal does (an M4 under macOS 27.0.1, plain twice alike; misuse,
//! which Metal's validation stops: "writes texture whose usage (0x11) doesn't specify
//! MTLTextureUsageShaderWrite"): a 2 x 2 texture of `ShaderRead | ShaderWrite |
//! PixelFormatView` usage viewed with a swizzle (the view's usage then `ShaderRead |
//! PixelFormatView`), or made with one (`Unknown` usage, reported the same), every texel
//! `(10, 20, 30, 40)` in bytes (`RG8Unorm` `(10, 20)`, `R32Float` 7.5, `R32Uint` 1000):
//!
//! - an `access::read` or `read_write` read returns the texel swizzled: each channel the source
//!   its swizzle names, `Zero` 0 and `One` 1, a format's missing green and blue 0 and its missing
//!   alpha 1;
//! - an `access::write` or `read_write` write of `v` stores channel `c` from the first lane whose
//!   source is `c`, else lane `c`;
//! - `atomic_fetch_add` adds to red and returns the texel unswizzled.
//!
//! The kernels are the owned fixture `rrp_swizzled_storage`'s: `rrp_sw_read_f` (`access::read`)
//! and `rrp_sw_rwread_f` and `rrp_sw_rwread_u` (`read_write`) store texel (0, 0) to buffer 0;
//! `rrp_sw_write_f`, `rrp_sw_rwwrite_f` and `rrp_sw_write_u` write buffer 1's value to each
//! thread's texel of a 2 x 2 grid; `rrp_sw_atomic_u` adds 5 to texel (0, 0) and stores what it
//! held. An 8-bit normalized read is compared bit for bit with its code `c` as Metal reads it,
//! `c / 255` correctly rounded (no channel may sit an ulp from it).

use super::degrade_notices::counts;
use super::degrade_notices::on_device_with;
use super::pixel_samplers::metal_sampler;
use super::{at_plan, buffer_with, converted_holds, read_buffer, WAIT};
use crate::air_library::AirLibrary;
use metal2vulkan::metal::{
    PixelFormat, TextureSwizzle, TextureSwizzleChannels, TextureType, TextureUsage,
};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    Buffer, CommandBufferStatus, ComputePipelineState, Degradation, Device, PipelineOptions,
    Region, ResourceOptions, Size, Texture, TextureDescriptor,
};

/// A format the probe measured.
#[derive(Clone, Copy, Debug)]
enum Format {
    Rgba8,
    Rg8,
    R32Float,
    R32Uint,
}

impl Format {
    fn pixel_format(self) -> PixelFormat {
        match self {
            Format::Rgba8 => PixelFormat::RGBA8Unorm,
            Format::Rg8 => PixelFormat::RG8Unorm,
            Format::R32Float => PixelFormat::R32Float,
            Format::R32Uint => PixelFormat::R32Uint,
        }
    }

    /// Bytes a texel.
    fn bytes(self) -> usize {
        match self {
            Format::Rg8 => 2,
            _ => 4,
        }
    }

    /// The channels the format stores.
    fn channels(self) -> usize {
        match self {
            Format::Rgba8 => 4,
            Format::Rg8 => 2,
            _ => 1,
        }
    }

    /// The probe's texel, as its bytes.
    fn texel_bytes(self) -> Vec<u8> {
        match self {
            Format::Rgba8 => vec![10, 20, 30, 40],
            Format::Rg8 => vec![10, 20],
            Format::R32Float => 7.5f32.to_le_bytes().to_vec(),
            Format::R32Uint => 1000u32.to_le_bytes().to_vec(),
        }
    }

    /// The probe's texel as Metal reads it unswizzled: an 8-bit normalized format's codes (`One`
    /// 255), a float's bits (`One` 1.0), an integer (`One` 1), missing channels 0 and alpha one.
    fn texel(self) -> ([u32; 4], u32) {
        match self {
            Format::Rgba8 => ([10, 20, 30, 40], 255),
            Format::Rg8 => ([10, 20, 0, 255], 255),
            Format::R32Float => {
                let one = 1.0f32.to_bits();
                ([7.5f32.to_bits(), 0, 0, one], one)
            }
            Format::R32Uint => ([1000, 0, 0, 1], 1),
        }
    }
}

/// `name`'s swizzle: each letter a channel's source, `0` and `1` the constants.
fn swizzle(name: &str) -> TextureSwizzleChannels {
    let v: Vec<TextureSwizzle> = name
        .chars()
        .map(|c| match c {
            'R' => TextureSwizzle::Red,
            'G' => TextureSwizzle::Green,
            'B' => TextureSwizzle::Blue,
            'A' => TextureSwizzle::Alpha,
            '0' => TextureSwizzle::Zero,
            _ => TextureSwizzle::One,
        })
        .collect();
    TextureSwizzleChannels::new(v[0], v[1], v[2], v[3])
}

/// The probe's cases: a format, a swizzle, and whether the texture is made with it (`Unknown`
/// usage) rather than viewed through it.
fn cases() -> Vec<(Format, &'static str, bool)> {
    let mut out = Vec::new();
    for s in [
        "RGBA", "GBAR", "BGRA", "ABGR", "GRBA", "RRRR", "R01A", "0000", "1111", "RGB1", "RGBR",
        "0GBA",
    ] {
        out.push((Format::Rgba8, s, false));
    }
    for s in ["GR01", "GRBA", "RRRR", "0R01"] {
        out.push((Format::Rg8, s, false));
    }
    for s in ["GRBA", "RRRR", "ARGB"] {
        out.push((Format::R32Float, s, false));
        out.push((Format::R32Uint, s, false));
    }
    out.push((Format::R32Uint, "1RG0", false));
    for s in ["GBAR", "RRRR"] {
        out.push((Format::Rgba8, s, true));
    }
    out
}

/// Metal's read of `texel` through `s` (`one` the constant's value).
fn read_rule(s: TextureSwizzleChannels, texel: [u32; 4], one: u32) -> [u32; 4] {
    [s.red, s.green, s.blue, s.alpha].map(|source| match source.as_raw() {
        0 => 0,
        1 => one,
        raw => texel[(raw - 2) as usize],
    })
}

/// The channels Metal stores for a write of `v` through `s`: channel `c` from the first lane
/// whose source is `c`, else lane `c`.
fn write_rule(s: TextureSwizzleChannels, v: [u32; 4]) -> [u32; 4] {
    let sources = [s.red, s.green, s.blue, s.alpha].map(|t| t.as_raw());
    std::array::from_fn(
        |c| match sources.iter().position(|&raw| raw == c as u32 + 2) {
            Some(i) => v[i],
            None => v[c],
        },
    )
}

fn state(device: &Device, kernel: &str) -> ComputePipelineState {
    let bitcode = llvm_assemble(&engine_dir().join("rrp_swizzled_storage.ll")).unwrap();
    let library = device.new_library_with_air(&bitcode).unwrap();
    let function = library.new_function_with_name(kernel).unwrap();
    device
        .new_compute_pipeline_state(
            &ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor())),
            &PipelineOptions::default(),
        )
        .unwrap()
}

/// The case's texture, filled with the probe's texel, and the texture its kernels bind: a view
/// through `name`'s swizzle, or the texture made with it (`made`).
fn textures(device: &Device, format: Format, name: &str, made: bool) -> (Texture, Texture) {
    let pf = format.pixel_format();
    let mut d = TextureDescriptor::texture_2d(pf, 2, 2, false);
    d.resource_options = ResourceOptions::shared();
    if made {
        d.usage = TextureUsage::empty();
        d.swizzle = swizzle(name);
    } else {
        d.usage = TextureUsage::SHADER_READ
            | TextureUsage::SHADER_WRITE
            | TextureUsage::PIXEL_FORMAT_VIEW;
        if matches!(format, Format::R32Uint) {
            d.usage |= TextureUsage::SHADER_ATOMIC;
        }
    }
    let texture = device.new_texture(&d).unwrap();
    let bound = match made {
        true => texture.clone(),
        false => texture
            .new_texture_view_with_swizzle(pf, TextureType::Type2D, 0..1, 0..1, swizzle(name))
            .unwrap(),
    };
    (texture, bound)
}

/// Fill every texel of `texture` with the probe's texel.
fn reset(texture: &Texture, format: Format) {
    let bytes = format.texel_bytes().repeat(4);
    let row = 2 * format.bytes() as u64;
    texture
        .replace_region(Region::new_2d(0, 0, 2, 2), 0, 0, &bytes, row, 0)
        .unwrap();
}

/// The texels of `texture`, each as its channels' words (8-bit codes, or a word).
fn texels(texture: &Texture, format: Format) -> Vec<Vec<u32>> {
    let n = format.bytes();
    let mut bytes = vec![0xee; 4 * n];
    texture
        .get_bytes(
            &mut bytes,
            2 * n as u64,
            0,
            Region::new_2d(0, 0, 2, 2),
            0,
            0,
        )
        .unwrap();
    bytes
        .chunks_exact(n)
        .map(|t| match format {
            Format::Rgba8 | Format::Rg8 => t.iter().map(|&b| u32::from(b)).collect(),
            _ => vec![u32::from_le_bytes([t[0], t[1], t[2], t[3]])],
        })
        .collect()
}

/// Dispatch `kernel` over `grid` threads (one threadgroup) with `texture` at slot 0 and
/// `buffers` at their indices, and wait for it.
fn dispatch(
    device: &Device,
    kernel: &str,
    texture: &Texture,
    buffers: &[(u32, &Buffer)],
    grid: u64,
) {
    let state = state(device, kernel);
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(&state).unwrap();
    encoder.set_texture(Some(texture), 0).unwrap();
    for &(index, buffer) in buffers {
        encoder.set_buffer(Some(buffer), 0, index).unwrap();
    }
    encoder
        .dispatch_threads(Size::new(grid, grid, 1), Size::new(grid, grid, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(
        command_buffer.status(),
        CommandBufferStatus::Completed,
        "{kernel}"
    );
    assert_eq!(command_buffer.error(), None, "{kernel}");
}

/// The four words of `buffer`'s first 16 bytes.
fn words(buffer: &Buffer) -> [u32; 4] {
    let b = read_buffer(buffer, 0, 16).unwrap();
    std::array::from_fn(|i| {
        u32::from_le_bytes([b[4 * i], b[4 * i + 1], b[4 * i + 2], b[4 * i + 3]])
    })
}

/// Each case's reads return the texel swizzled, through an `access::read` binding (a sampled
/// view, which takes the swizzle) and a `read_write` one (a storage view the function swizzles).
/// An 8-bit normalized channel's code `c` reads `c / 255` correctly rounded where the device samples
/// through Metal's own texture unit; elsewhere the device's conversion of it, announced
/// `device-texel-conversion`, within its bound (`converted_holds`).
#[test]
fn reads_of_a_swizzled_texture_are_swizzled() {
    on_device_with(at_plan, |device, messages| {
        let exact = metal_sampler(device);
        let mut normalized = false;
        for (format, name, made) in cases() {
            let what = format!("{format:?} {name} made {made}");
            let (texture, bound) = textures(device, format, name, made);
            let (texel, one) = format.texel();
            let want = read_rule(swizzle(name), texel, one);
            let kernels: &[&str] = match format {
                Format::R32Uint => &["rrp_sw_rwread_u"],
                _ => &["rrp_sw_read_f", "rrp_sw_rwread_f"],
            };
            for &kernel in kernels {
                reset(&texture, format);
                let out = buffer_with(device, &[0xab; 16]).unwrap();
                dispatch(device, kernel, &bound, &[(0, &out)], 1);
                let got = words(&out);
                match format {
                    // An 8-bit normalized channel's code `c` reads `c / 255`, correctly rounded.
                    Format::Rgba8 | Format::Rg8 => {
                        normalized = true;
                        let want = want.map(|c| c as f32 / 255.0);
                        let holds =
                            (0..4).all(|c| converted_holds(f32::from_bits(got[c]), want[c], exact));
                        assert!(holds, "{what} {kernel}: {got:08x?}, Metal's {want:?}");
                    }
                    _ => assert_eq!(got, want, "{what} {kernel}"),
                }
            }
        }
        let announced = counts(&messages.take_notices());
        assert!(
            exact || !normalized || announced.contains_key(&Degradation::DeviceTexelConversion),
            "{announced:?}"
        );
    });
}

/// Each case's writes, `access::write` and `read_write`, store the value with the swizzle undone:
/// channel `c` from the first lane whose source is `c`, else lane `c`.
fn writes_of(device: &Device, format: Format, name: &str, made: bool) {
    let what = format!("{format:?} {name} made {made}");
    let (texture, bound) = textures(device, format, name, made);
    let values: &[(&str, [u32; 4])] = match format {
        Format::Rgba8 | Format::Rg8 => &[
            ("rrp_sw_write_f", [0x10, 0x20, 0x30, 0x40]),
            ("rrp_sw_rwwrite_f", [0x50, 0x60, 0x70, 0x80]),
        ],
        Format::R32Float => &[
            (
                "rrp_sw_write_f",
                [0x3fa0_0000, 0x4020_0000, 0x4070_0000, 0x40a0_0000],
            ),
            (
                "rrp_sw_rwwrite_f",
                [0x40c8_0000, 0x40f0_0000, 0x410c_0000, 0x4120_0000],
            ),
        ],
        Format::R32Uint => &[("rrp_sw_write_u", [11, 22, 33, 44])],
    };
    for &(kernel, v) in values {
        reset(&texture, format);
        // An 8-bit normalized value is its code over 255, which every device stores as the code.
        let lanes: Vec<u8> = v
            .iter()
            .flat_map(|&x| match format {
                Format::Rgba8 | Format::Rg8 => (x as f32 / 255.0).to_le_bytes(),
                _ => x.to_le_bytes(),
            })
            .collect();
        let value = buffer_with(device, &lanes).unwrap();
        dispatch(device, kernel, &bound, &[(1, &value)], 2);
        let want = write_rule(swizzle(name), v)[..format.channels()].to_vec();
        for (n, got) in texels(&texture, format).into_iter().enumerate() {
            assert_eq!(got, want, "{what} {kernel} texel {n}");
        }
    }
}

#[test]
fn writes_through_a_swizzled_texture_undo_the_swizzle() {
    on_device_with(at_plan, |device, _| {
        for (format, name, made) in cases() {
            writes_of(device, format, name, made);
        }
    });
}

/// An atomic through a swizzled `R32Uint` view adds to red and returns the texel unswizzled.
#[test]
fn an_atomic_through_a_swizzled_view_ignores_the_swizzle() {
    on_device_with(at_plan, |device, _| {
        for name in ["RRRR", "GRBA", "ARGB", "1RG0"] {
            let (texture, bound) = textures(device, Format::R32Uint, name, false);
            reset(&texture, Format::R32Uint);
            let out = buffer_with(device, &[0xab; 16]).unwrap();
            dispatch(device, "rrp_sw_atomic_u", &bound, &[(0, &out)], 1);
            assert_eq!(words(&out), [1000, 0, 0, 1], "{name}");
            let red: Vec<u32> = texels(&texture, Format::R32Uint)
                .into_iter()
                .map(|t| t[0])
                .collect();
            assert_eq!(red, [1005, 1000, 1000, 1000], "{name}");
        }
    });
}
