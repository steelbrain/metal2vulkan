//! Apple's RGBX textures (`RGBX8Unorm` through `RGBX32Float`, Apple SPI) on the device the cfg
//! selects, under the validation layer, against Metal's measured answers (an M4 under macOS 27.0.1,
//! our own probes).
//!
//! An RGBX texture holds its RGBA twin's bytes (`metal::format::x_channel`). Every read sees
//! alpha 1: a kernel's read, sample and fetch, whatever the X field holds. A kernel's write
//! stores the green field's bits in the X field, in every texel it writes. A fill packs the X
//! field as Metal does (`clear_value.rs`: zero, or the alpha of a float format). Until the
//! runtime rewrites the X fields Metal's render passes and mipmaps store, both are refused,
//! typed: a pass with an RGBX color attachment, with `RenderTarget` usage or without
//! (`NotImplemented(Format)`), and `generate_mipmaps` of more than one level
//! (`NotImplemented(MipmapGeneration)`).
//!
//! Each case runs on the twin too, over the same bytes and values: the RGBX texture's reads are
//! the twin's red, green and blue and alpha 1, and its stored bytes the twin's with the X field
//! the green field. The kernels are the owned fixture `fpv_rgbx_textures`'s, each at its
//! thread's texel of a 16 by 16 texture.

use super::degrade_notices::{counts, on_device_with};
use super::{at_plan, buffer_with, read_buffer, WAIT};
use crate::air_library::AirLibrary;
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::error::{Feature, Needs};
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, PrecisionNote, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    ClearColor, CommandBufferStatus, ComputePipelineState, Degradation, Device, DeviceOptions,
    Error, LoadAction, PipelineOptions, Region, RenderPassDescriptor, ResourceOptions, Size,
    StoreAction, Texture, TextureDescriptor,
};

/// A texture's width and height.
const SIDE: u64 = 16;

/// The texels of a texture.
const TEXELS: usize = (SIDE * SIDE) as usize;

/// What a format's texels read as in a kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Read {
    Float,
    Uint,
    Int,
}

/// Each RGBX format, its RGBA twin, its texel's bytes, its channel's bits and how it reads.
const FORMATS: [(PixelFormat, PixelFormat, usize, u32, Read); 11] = {
    use PixelFormat as P;
    [
        (P::RGBX8Unorm, P::RGBA8Unorm, 4, 8, Read::Float),
        (P::RGBX8UnormSRGB, P::RGBA8UnormSRGB, 4, 8, Read::Float),
        (P::RGBX8Snorm, P::RGBA8Snorm, 4, 8, Read::Float),
        (P::RGBX8Uint, P::RGBA8Uint, 4, 8, Read::Uint),
        (P::RGBX8Sint, P::RGBA8Sint, 4, 8, Read::Int),
        (P::RGBX16Uint, P::RGBA16Uint, 8, 16, Read::Uint),
        (P::RGBX16Sint, P::RGBA16Sint, 8, 16, Read::Int),
        (P::RGBX16Float, P::RGBA16Float, 8, 16, Read::Float),
        (P::RGBX32Uint, P::RGBA32Uint, 16, 32, Read::Uint),
        (P::RGBX32Sint, P::RGBA32Sint, 16, 32, Read::Int),
        (P::RGBX32Float, P::RGBA32Float, 16, 32, Read::Float),
    ]
};

/// A deterministic stream of random words (xorshift64*).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| (self.next() >> 32) as u8).collect()
    }
}

/// `fpv_rgbx_textures`' kernel `kernel`, compiled for `device`.
pub(super) fn state(device: &Device, kernel: &str) -> ComputePipelineState {
    let bitcode = llvm_assemble(&engine_dir().join("fpv_rgbx_textures.ll")).unwrap();
    let library = device.new_library_with_air(&bitcode).unwrap();
    let function = library.new_function_with_name(kernel).unwrap();
    device
        .new_compute_pipeline_state(
            &ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor())),
            &PipelineOptions::default(),
        )
        .unwrap()
}

/// A shared 16 by 16 texture of `pf` that kernels read and write, holding `bytes`.
fn texture(device: &Device, pf: PixelFormat, bytes: &[u8]) -> Texture {
    let mut d = TextureDescriptor::texture_2d(pf, SIDE, SIDE, false);
    d.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
    d.resource_options = ResourceOptions::shared();
    let texture = device.new_texture(&d).unwrap();
    let row = bytes.len() as u64 / SIDE;
    texture
        .replace_region(Region::new_2d(0, 0, SIDE, SIDE), 0, 0, bytes, row, 0)
        .unwrap();
    texture
}

/// `texture`'s bytes, `bytes` a texel.
fn bytes_of(texture: &Texture, bytes: usize) -> Vec<u8> {
    let row = SIDE * bytes as u64;
    let mut out = vec![0u8; TEXELS * bytes];
    texture
        .get_bytes(&mut out, row, 0, Region::new_2d(0, 0, SIDE, SIDE), 0, 0)
        .unwrap();
    out
}

/// Run `kernel` over every texel of `texture` with `data` at buffer 0 (and a nearest sampler
/// where `sampled`), and return buffer 0's bytes.
pub(super) fn run(
    device: &Device,
    kernel: &str,
    texture: &Texture,
    data: &[u8],
    sampled: bool,
) -> Vec<u8> {
    let state = state(device, kernel);
    let data = buffer_with(device, data).unwrap();
    let width = buffer_with(device, &(SIDE as u32).to_le_bytes()).unwrap();
    let sampler = device.new_sampler_state(&SamplerDescriptor::new()).unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(&state).unwrap();
    encoder.set_buffer(Some(&data), 0, 0).unwrap();
    encoder.set_buffer(Some(&width), 0, 1).unwrap();
    encoder.set_texture(Some(texture), 0).unwrap();
    if sampled {
        encoder.set_sampler_state(Some(&sampler), 0).unwrap();
    }
    encoder
        .dispatch_threadgroups(Size::new(1, 1, 1), Size::new(SIDE, SIDE, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    assert_eq!(command_buffer.error(), None);
    read_buffer(&data, 0, (TEXELS * 16) as u64).unwrap()
}

/// The kernels that read a texture of `read`, and whether each samples.
fn readers(read: Read) -> &'static [(&'static str, bool)] {
    match read {
        Read::Float => &[
            ("fpv_rgbx_fetch", false),
            ("fpv_rgbx_sample", true),
            ("fpv_rgbx_read_write", false),
        ],
        Read::Uint => &[("fpv_rgbx_read_write_uint", false)],
        Read::Int => &[("fpv_rgbx_read_write_int", false)],
    }
}

/// The kernel that writes a texture of `read`.
fn writer(read: Read) -> &'static str {
    match read {
        Read::Float => "fpv_rgbx_write",
        Read::Uint => "fpv_rgbx_write_uint",
        Read::Int => "fpv_rgbx_write_int",
    }
}

/// The 4-byte lanes of `bytes`.
fn words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
        .collect()
}

/// The bits a kernel reads as alpha 1.
fn one(read: Read) -> u32 {
    match read {
        Read::Float => 1.0f32.to_bits(),
        Read::Uint | Read::Int => 1,
    }
}

/// Every read of random bytes in every RGBX format: its texels' bytes as replaced, and each
/// kernel's red, green and blue the twin's, its alpha 1.
#[test]
fn rgbx_reads_take_alpha_one() {
    on_device_with(at_plan, |device, _| {
        let name = device.name().to_owned();
        let mut differ = Vec::new();
        for (i, (pf, twin, bytes, _, read)) in FORMATS.into_iter().enumerate() {
            let data = Rng(0x9e37_79b9_7f4a_7c15 ^ i as u64).bytes(TEXELS * bytes);
            let rgbx = texture(device, pf, &data);
            let rgba = texture(device, twin, &data);
            if bytes_of(&rgbx, bytes) != data {
                differ.push(format!("{pf:?}: get_bytes is not the replaced bytes"));
            }
            for &(kernel, sampled) in readers(read) {
                let zeros = vec![0u8; TEXELS * 16];
                let got = words(&run(device, kernel, &rgbx, &zeros, sampled));
                let want = words(&run(device, kernel, &rgba, &zeros, sampled));
                let wrong = (0..TEXELS)
                    .filter(|&t| {
                        got[4 * t..4 * t + 3] != want[4 * t..4 * t + 3]
                            || got[4 * t + 3] != one(read)
                    })
                    .count();
                eprintln!("RGBX read {pf:?} by {kernel} on {name}: {wrong} of {TEXELS} differ");
                if wrong > 0 {
                    differ.push(format!("{pf:?} by {kernel}: {wrong} of {TEXELS}"));
                }
            }
        }
        assert!(differ.is_empty(), "RGBX reads not Metal's: {differ:#?}");
    });
}

/// `n` random values a kernel writes into a texture of `read` whose channels have `bits` bits,
/// as `float4`, `uint4` or `int4` bytes: floats in [-2, 2), integers in the channel's range.
fn values(read: Read, bits: u32, seed: u64) -> Vec<u8> {
    let mut rng = Rng(seed);
    (0..TEXELS * 4)
        .flat_map(|_| {
            let r = rng.next();
            let word = match read {
                Read::Float => ((r >> 40) as f32 / (1u64 << 24) as f32 * 4.0 - 2.0).to_bits(),
                Read::Uint => match bits {
                    32 => (r >> 32) as u32,
                    _ => (r >> 32) as u32 & ((1 << bits) - 1),
                },
                Read::Int => {
                    let w = (r >> 32) as u32;
                    match bits {
                        32 => w,
                        _ => (((w << (32 - bits)) as i32) >> (32 - bits)) as u32,
                    }
                }
            };
            word.to_le_bytes()
        })
        .collect()
}

/// Every kernel write of random values into every RGBX format over bytes of `0xA5`: the twin's
/// bytes, the X field of each texel the green field. An sRGB format's codes are the MSL transfer
/// by the device's `Pow` in both writes, each within one code of Metal's
/// (`PrecisionNote::DevicePackConversion`, which the runtime announces), so the two lie within
/// [`SRGB_WRITES_APART`] codes of each other in each field (two of the 256 texels a code apart on
/// the Apple M3 Max through MoltenVK 1.4.2, where the two modules' `Pow` answer otherwise), and
/// the X field is the texel's own green exactly.
#[test]
fn rgbx_writes_store_green_as_x() {
    on_device_with(at_plan, |device, messages| {
        let name = device.name().to_owned();
        let mut differ = Vec::new();
        // The sRGB texels within the bound but not the twin's bytes.
        let mut apart = 0;
        for (i, (pf, twin, bytes, bits, read)) in FORMATS.into_iter().enumerate() {
            let fill = vec![0xA5u8; TEXELS * bytes];
            let rgbx = texture(device, pf, &fill);
            let rgba = texture(device, twin, &fill);
            let values = values(read, bits, 0x51ed_270b_27a3_2d4c ^ i as u64);
            run(device, writer(read), &rgbx, &values, false);
            run(device, writer(read), &rgba, &values, false);
            let got = bytes_of(&rgbx, bytes);
            let mut want = bytes_of(&rgba, bytes);
            let field = bytes / 4;
            for texel in want.chunks_exact_mut(bytes) {
                let green = texel[field..2 * field].to_vec();
                texel[3 * field..].copy_from_slice(&green);
            }
            let srgb = twin == PixelFormat::RGBA8UnormSRGB;
            let within = |g: &[u8], w: &[u8]| {
                srgb && g[3] == g[1]
                    && g.iter()
                        .zip(w)
                        .all(|(a, b)| a.abs_diff(*b) <= SRGB_WRITES_APART)
            };
            let (mut wrong, mut near) = (0, 0);
            for (g, w) in got.chunks_exact(bytes).zip(want.chunks_exact(bytes)) {
                match (g == w, within(g, w)) {
                    (true, _) => {}
                    (false, true) => near += 1,
                    (false, false) => wrong += 1,
                }
            }
            apart += near;
            eprintln!(
                "RGBX write {pf:?} on {name}: {wrong} of {TEXELS} texels differ from the twin's \
                 with X = green, {near} more within the device's sRGB encode"
            );
            if wrong > 0 {
                differ.push(format!("{pf:?}: {wrong} of {TEXELS}"));
            }
        }
        assert!(differ.is_empty(), "RGBX writes not Metal's: {differ:#?}");
        if apart > 0 {
            let note = Degradation::from_precision_note(PrecisionNote::DevicePackConversion);
            let announced = counts(&messages.take_notices());
            assert!(
                announced.contains_key(&note),
                "{apart} sRGB texels within the device's encode, unannounced: {announced:?}"
            );
        }
    });
}

/// How many codes apart two writes of the same `float` into an 8-bit sRGB texture may store a
/// field: each the device's encode, within one code of Metal's
/// (`PrecisionNote::DevicePackConversion`'s bound).
const SRGB_WRITES_APART: u8 = 2;

/// Metal's color fills of every RGBX format: three colors' texel bytes, and a
/// byte fill's pattern cut at the texel. A plain color fill of a format the device renders is the
/// device's clear of the color (`blit_fill.rs`): where its fill announces
/// `device-color-conversion`, asked for at each fill, a texel may stand a code from Metal's in a
/// field (`color_fills::fields_apart`), the X field among them.
#[test]
fn rgbx_fills_store_metals_bytes() {
    let colors = [
        ClearColor::new(0.25, 0.5, 0.75, 1.0),
        ClearColor::new(0.1, 0.9, 0.3, 0.5),
        ClearColor::new(1.5, -0.5, 7.0, 0.0),
    ];
    let zero8 = "0000000000000000";
    let zero16 = "00000000000000000000000000000000";
    let cases: [(PixelFormat, [&str; 3]); 11] = {
        use PixelFormat as P;
        [
            (P::RGBX8Unorm, ["4080bf00", "1ae54d00", "ff00ff00"]),
            (P::RGBX8UnormSRGB, ["89bce100", "59f39500", "ff00ff00"]),
            (P::RGBX8Snorm, ["20405f00", "0d722600", "7fc07f00"]),
            (P::RGBX8Uint, ["00000000", "00000000", "01000700"]),
            (P::RGBX8Sint, ["00000000", "00000000", "01000700"]),
            (P::RGBX16Uint, [zero8, zero8, "0100000007000000"]),
            (P::RGBX16Sint, [zero8, zero8, "0100000007000000"]),
            (
                P::RGBX16Float,
                ["00340038003a003c", "662e333bcd340038", "003e00b800470000"],
            ),
            (
                P::RGBX32Uint,
                [zero16, zero16, "01000000000000000700000000000000"],
            ),
            (
                P::RGBX32Sint,
                [zero16, zero16, "01000000000000000700000000000000"],
            ),
            (
                P::RGBX32Float,
                [
                    "0000803e0000003f0000403f0000803f",
                    "cdcccc3d6666663f9a99993e0000003f",
                    "0000c03f000000bf0000e04000000000",
                ],
            ),
        ]
    };
    let hex = |bytes: &[u8]| bytes.iter().map(|b| format!("{b:02x}")).collect::<String>();
    let each = |options: &mut DeviceOptions| {
        options.announce_each = [Degradation::DeviceColorConversion].into_iter().collect();
    };
    on_device_with(
        |options| {
            each(options);
            at_plan(options)
        },
        |device, messages| {
            let mut differ = Vec::new();
            for (pf, wants) in cases {
                let bytes = FORMATS.iter().find(|f| f.0 == pf).unwrap().2;
                let texture = texture(device, pf, &vec![0xA5u8; TEXELS * bytes]);
                let pattern: Vec<u8> = (0..16u8).map(|k| 0x80 + 7 * k).collect();
                let fills = [None, Some(colors[0]), Some(colors[1]), Some(colors[2])];
                let wants = [
                    hex(&pattern[..bytes]),
                    wants[0].into(),
                    wants[1].into(),
                    wants[2].into(),
                ];
                for (color, want) in fills.into_iter().zip(wants) {
                    let queue = device.new_command_queue().unwrap();
                    let command_buffer = queue.command_buffer().unwrap();
                    let mut blit = command_buffer.blit_command_encoder().unwrap();
                    let region = Region::new_2d(0, 0, SIDE, SIDE);
                    match color {
                        None => blit.fill_texture(&texture, 0, 0, region, &pattern),
                        Some(c) => blit.fill_texture_color(&texture, 0, 0, region, c, None),
                    }
                    .unwrap();
                    let converted = super::degrade_notices::counts(&messages.take_notices())
                        .contains_key(&Degradation::DeviceColorConversion);
                    blit.end_encoding().unwrap();
                    command_buffer.commit().unwrap();
                    command_buffer.wait_until_completed_for(WAIT).unwrap();
                    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
                    let got = bytes_of(&texture, bytes);
                    let metal: Vec<u8> = (0..want.len())
                        .step_by(2)
                        .map(|i| u8::from_str_radix(&want[i..i + 2], 16).unwrap())
                        .collect();
                    let wrong = got
                        .chunks_exact(bytes)
                        .filter(|&texel| {
                            let near = || {
                                super::color_fills::fields_apart(pf, texel, &metal)
                                    .is_some_and(|apart| !apart.nans)
                            };
                            texel != metal && !(converted && near())
                        })
                        .count();
                    if wrong > 0 {
                        differ.push(format!("{pf:?} {color:?}: {wrong} texels not {want}"));
                    }
                }
            }
            assert!(differ.is_empty(), "RGBX fills not Metal's: {differ:#?}");
        },
    );
}

/// A pass rendering into an RGBX texture, with `RenderTarget` usage or without, and the mipmaps
/// of one of two levels: refused `NotImplemented` until the runtime stores the X fields Metal
/// stores there.
#[test]
fn rgbx_passes_and_mipmaps_are_refused() {
    on_device_with(at_plan, |device, _| {
        for (pf, ..) in FORMATS {
            for usage in [
                TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ,
                TextureUsage::SHADER_READ,
            ] {
                let mut d = TextureDescriptor::texture_2d(pf, SIDE, SIDE, false);
                d.usage = usage;
                d.resource_options = ResourceOptions::private();
                let target = device.new_texture(&d).unwrap();
                let mut pass = RenderPassDescriptor::new();
                let color = &mut pass.color_attachments[0];
                color.texture = Some(target);
                color.load_action = LoadAction::Clear;
                color.store_action = StoreAction::Store;
                color.clear_color = ClearColor::new(0.25, 0.5, 0.75, 1.0);
                let queue = device.new_command_queue().unwrap();
                let command_buffer = queue.command_buffer().unwrap();
                match command_buffer.render_command_encoder(&pass) {
                    Err(Error::Unsupported(u)) => {
                        assert_eq!(
                            (u.feature, u.needs),
                            (Feature::Format, Needs::NotImplemented),
                            "{pf:?} {usage:?}"
                        );
                    }
                    Err(other) => panic!("{pf:?} {usage:?}: a pass refused as {other}"),
                    Ok(_) => panic!("{pf:?} {usage:?}: a pass taken"),
                }
            }
            let mut d = TextureDescriptor::texture_2d(pf, SIDE, SIDE, true);
            d.mipmap_level_count = 2;
            d.usage = TextureUsage::SHADER_READ;
            d.resource_options = ResourceOptions::private();
            let texture = device.new_texture(&d).unwrap();
            let queue = device.new_command_queue().unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            match blit.generate_mipmaps(&texture) {
                Err(Error::Unsupported(u)) => assert_eq!(
                    (u.feature, u.needs),
                    (Feature::MipmapGeneration, Needs::NotImplemented),
                    "{pf:?}"
                ),
                Err(other) => panic!("{pf:?}: mipmaps refused as {other}"),
                Ok(()) => panic!("{pf:?}: mipmaps taken"),
            }
            blit.end_encoding().unwrap();
        }
    });
}

/// A `half` sample of a 32-bit float RGBX texture and its twin, shared and private, through the
/// runtime sampler at slot 0 (`rgbx_sample_half`, a sample's `half4`), nearest and linear, clamped
/// to the edge and to an opaque white border, each lane at its own texel's center, where a linear
/// filter weighs that texel alone: each lane against Metal's answer, the texel's `float` channel
/// rounded to nearest even (finite channels within `half`'s range; `DriverDeviation::MetalHalfConversion`
/// measures a read, sample and gather of every `float` word of `RGBA32Float`), with an RGBX alpha
/// of 1 (`HALF-SAMPLE` lines name the notices). On a profile that takes that deviation, where the
/// device samples through Metal's own texture unit, the dispatch announces no
/// `half-texel-conversion`.
#[test]
fn rgbx_half_samples_are_metals() {
    use metal2vulkan::device::DriverDeviation;
    use metal2vulkan::metal::{
        SamplerAddressMode as Mode, SamplerBorderColor, SamplerMinMagFilter,
    };
    use PixelFormat as P;
    let mut differ = Vec::new();
    for (i, pf) in [P::RGBX32Float, P::RGBA32Float].into_iter().enumerate() {
        for options in [ResourceOptions::shared(), ResourceOptions::private()] {
            on_device_with(at_plan, |device, messages| {
                let mut rng = Rng(0x5a3c_e11f ^ i as u64);
                let texels: Vec<[f32; 4]> = (0..TEXELS)
                    .map(|_| {
                        std::array::from_fn(|_| {
                            (rng.next() >> 40) as f32 / (1u64 << 24) as f32 * 120_000.0 - 60_000.0
                        })
                    })
                    .collect();
                let data: Vec<u8> = texels
                    .iter()
                    .flat_map(|t| t.iter().flat_map(|v| v.to_le_bytes()))
                    .collect();
                let mut d = TextureDescriptor::texture_2d(pf, SIDE, SIDE, false);
                d.usage = TextureUsage::SHADER_READ;
                d.resource_options = options;
                let texture = device.new_texture(&d).unwrap();
                let source = buffer_with(device, &data).unwrap();
                let queue = device.new_command_queue().unwrap();
                let fill = queue.command_buffer().unwrap();
                let mut blit = fill.blit_command_encoder().unwrap();
                blit.copy_from_buffer_to_texture(
                    &source,
                    0,
                    SIDE * 16,
                    0,
                    Size::new(SIDE, SIDE, 1),
                    &texture,
                    0,
                    0,
                    metal2vulkan_runtime::Origin::new(0, 0, 0),
                    metal2vulkan_runtime::BlitOption::None,
                )
                .unwrap();
                blit.end_encoding().unwrap();
                fill.commit().unwrap();
                fill.wait_until_completed_for(WAIT).unwrap();
                messages.take_notices();
                let bitcode = llvm_assemble(&engine_dir().join("rgbx_sample_half.ll")).unwrap();
                let library = device.new_library_with_air(&bitcode).unwrap();
                let function = library.new_function_with_name("rgbx_sample_half").unwrap();
                let state = device
                    .new_compute_pipeline_state(
                        &ComputePipelineDescriptor::new(StageSource::Function(
                            function.function_descriptor(),
                        )),
                        &PipelineOptions::default(),
                    )
                    .unwrap();
                let centers: Vec<u8> = (0..TEXELS as u64)
                    .flat_map(|t| {
                        let side = SIDE as f32;
                        [
                            ((t % SIDE) as f32 + 0.5) / side,
                            ((t / SIDE) as f32 + 0.5) / side,
                        ]
                        .into_iter()
                        .flat_map(f32::to_le_bytes)
                    })
                    .collect();
                let centers = buffer_with(device, &centers).unwrap();
                let want = |t: usize, c: usize| match (c, pf) {
                    (3, P::RGBX32Float) => 1.0,
                    _ => super::extended_range_targets::half_value(
                        super::extended_range_targets::half_bits(texels[t][c]),
                    ),
                };
                let mut announced = std::collections::BTreeMap::new();
                for filter in [SamplerMinMagFilter::Nearest, SamplerMinMagFilter::Linear] {
                    for mode in [Mode::ClampToEdge, Mode::ClampToBorderColor] {
                        let mut s = SamplerDescriptor::new();
                        (s.min_filter, s.mag_filter) = (filter, filter);
                        (s.s_address_mode, s.t_address_mode) = (mode, mode);
                        s.border_color = SamplerBorderColor::OpaqueWhite;
                        let sampler = device.new_sampler_state(&s).unwrap();
                        let out = buffer_with(device, &vec![0u8; TEXELS * 8]).unwrap();
                        let command_buffer = queue.command_buffer().unwrap();
                        let mut encoder = command_buffer.compute_command_encoder().unwrap();
                        encoder.set_compute_pipeline_state(&state).unwrap();
                        encoder.set_texture(Some(&texture), 0).unwrap();
                        encoder.set_sampler_state(Some(&sampler), 0).unwrap();
                        encoder.set_buffer(Some(&out), 0, 0).unwrap();
                        encoder.set_buffer(Some(&centers), 0, 1).unwrap();
                        encoder
                            .dispatch_threads(Size::new(SIDE, SIDE, 1), Size::new(SIDE, SIDE, 1))
                            .unwrap();
                        encoder.end_encoding().unwrap();
                        command_buffer.commit().unwrap();
                        command_buffer.wait_until_completed_for(WAIT).unwrap();
                        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
                        let notices = messages.take_notices();
                        for notice in &notices {
                            eprintln!(
                                "HALF-SAMPLE {pf:?} {:#x} {filter:?} {mode:?} notice: {}",
                                options.raw(),
                                notice.text
                            );
                        }
                        announced.extend(counts(&notices));
                        let got = read_buffer(&out, 0, (TEXELS * 8) as u64).unwrap();
                        let mut wrong = 0;
                        for t in 0..TEXELS {
                            let lanes: [f32; 4] = std::array::from_fn(|c| {
                                let at = 8 * t + 2 * c;
                                super::extended_range_targets::half_value(u16::from_le_bytes([
                                    got[at],
                                    got[at + 1],
                                ]))
                            });
                            if (0..4).any(|c| lanes[c].to_bits() != want(t, c).to_bits()) {
                                if wrong < 4 {
                                    eprintln!(
                                        "HALF-SAMPLE {pf:?} {:#x} {filter:?} {mode:?} texel {t}: \
                                         {lanes:?}, Metal's {:?}",
                                        options.raw(),
                                        [want(t, 0), want(t, 1), want(t, 2), want(t, 3)]
                                    );
                                }
                                wrong += 1;
                            }
                        }
                        eprintln!(
                            "HALF-SAMPLE {pf:?} options {:#x} {filter:?} {mode:?} on {}: {wrong} \
                             of {TEXELS} lanes differ",
                            options.raw(),
                            device.name()
                        );
                        if wrong > 0 {
                            differ.push(format!(
                                "{pf:?} {:#x} {filter:?} {mode:?}: {wrong}",
                                options.raw()
                            ));
                        }
                    }
                }
                let profile = device.compile_device().profile();
                let metals = profile.takes_driver_deviation(DriverDeviation::MetalSampler)
                    && profile.takes_driver_deviation(DriverDeviation::MetalHalfConversion);
                let conversion =
                    Degradation::from_precision_note(PrecisionNote::HalfTexelConversion);
                eprintln!(
                    "HALF-SAMPLE {pf:?} options {:#x} on {}: the dispatches announced {announced:?}",
                    options.raw(),
                    device.name()
                );
                if metals && announced.contains_key(&conversion) {
                    differ.push(format!(
                        "{pf:?} {:#x}: announced {conversion:?}",
                        options.raw()
                    ));
                }
            });
        }
    }
    assert!(differ.is_empty(), "half samples not Metal's: {differ:#?}");
}
