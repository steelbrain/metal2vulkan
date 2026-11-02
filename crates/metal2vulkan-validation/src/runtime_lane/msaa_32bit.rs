//! Device tests of `supports32BitMSAA` (`Device::supports_32bit_msaa`) on the device the cfg
//! selects, under the validation layer, against our probe `msaa_32bit_formats.m` (an Apple M3 Max
//! under macOS 27.0.1, which answers yes; plain Metal, and its API validation where named):
//!
//! - **What the answer promises is made.** Metal makes a 2D multisample `RenderTarget |
//!   ShaderRead` texture of each of the nine formats (`R32Uint`, `R32Sint`, `RG32Uint`,
//!   `RG32Sint`, `RGBA32Uint`, `RGBA32Sint`, `R32Float`, `RG32Float`, `RGBA32Float`) at 2 and 4
//!   samples, and stops one at 8 ("sampleCount (8) is not supported by device"). Where the device
//!   answers yes, every one is made at every count its textures take above 1, 4 among them, and
//!   each format's capabilities list `Msaa` (and a float one's `Resolve`); where it answers no, 4
//!   is not taken or one of them is refused as a count its format lacks.
//! - **An integer target holds every bit drawn.** Metal reads back, through `texture2d_ms<uint>`,
//!   every channel a draw at sample rate wrote into each 32-bit integer format at 2 and 4 samples,
//!   bit for bit (256 channels at 4: edge values `0x7fffffff`, `0x80000000`, `0xffffffff`,
//!   `0x01000001` among them), with and without its API validation. The runtime's `R32Uint` and
//!   `RGBA32Uint` targets are drawn the same way (the owned fixture `rt6_samples`'s fill) and read
//!   back the same way, at each count made.
//! - **A float target resolves as Metal's.** An `RG32Float` 4-sample target drawn so and resolved
//!   by its pass's `StoreAndMultisampleResolve` holds its samples as written, and resolves to
//!   Metal's texels ([`METAL_RG32FLOAT_4X`]): the samples summed in order in `float` from sample 0,
//!   each input, partial sum and the result times `1/4` a denormal flushed to the zero of its
//!   sign, and a NaN anywhere the canonical `0x7fc00000` (every channel of `R32Float`, `RG32Float`
//!   and `RGBA32Float` at 2 and 4, under both resolving store actions). Exact on a device whose
//!   average is measured to be Metal's (`DriverFact::ColorResolveAverageIsMetals`), and nothing
//!   announced; elsewhere the differences are printed and `device-rounded-resolve` announced.
//! - **An integer target resolves nowhere.** Metal's validation stops an `RGBA32Uint` resolve
//!   ("PixelFormat MTLPixelFormatRGBA32Uint cannot be a MSAA resolve target"), and plain Metal
//!   writes values no sample holds: refused (`Invalid`, `ResolveFormat`).

use super::degrade_notices::counts;
use super::renders::{blit_read, run};
use super::{layer_reported_nothing_at_all, validated_device, Messages, ScratchCache};
use crate::candidate::runtime_samples::{encode as samples_of, SampleFill};
use crate::case::{TextureFormat as CaseFormat, TextureType as CaseType};
use crate::executor_contract::multisample::{ReadbackView, SampleRead};
use crate::gpu::TEST_DEVICE;
use crate::literal::texture_layout;
use metal2vulkan::device::DriverFact;
use metal2vulkan::error::Feature;
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};
use metal2vulkan_runtime::{
    Attachment, Degradation, Device, Error, FormatCapability, Invalid, Problem,
    RenderPassDescriptor, RenderPassProblem, ResourceOptions, StoreAction, Texture,
    TextureDescriptor,
};

/// The formats of `supports32BitMSAA`: the six integer ones a multisample texture takes, then the
/// three float ones that resolve.
const FORMATS: [PixelFormat; 9] = [
    PixelFormat::R32Uint,
    PixelFormat::R32Sint,
    PixelFormat::RG32Uint,
    PixelFormat::RG32Sint,
    PixelFormat::RGBA32Uint,
    PixelFormat::RGBA32Sint,
    PixelFormat::R32Float,
    PixelFormat::RG32Float,
    PixelFormat::RGBA32Float,
];

/// The probe's targets are 4².
const SIDE: u32 = 4;

/// A private 4² 2D multisample texture of `pf` of `samples`, a render target read by shaders
/// (one sample: 2D), as the probe makes.
fn target(device: &Device, pf: PixelFormat, samples: u32) -> Result<Texture, Error> {
    let mut d = TextureDescriptor::texture_2d(pf, SIDE.into(), SIDE.into(), false);
    if samples > 1 {
        d.texture_type = TextureType::Type2DMultisample;
    }
    d.sample_count = samples.into();
    d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    d.resource_options = ResourceOptions::private();
    device.new_texture(&d)
}

/// The sample counts above 1 the device's textures take (`supportsTextureSampleCount:`).
fn multisample_counts(device: &Device) -> Vec<u32> {
    let p = device.properties();
    [2u32, 4, 8, 16, 32, 64]
        .into_iter()
        .filter(|&n| p.supports_texture_sample_count(n.into()) == Ok(true))
        .collect()
}

/// The probe's integer channel `i`: eight edge values, then a hash.
fn word(i: u32) -> u32 {
    const EDGE: [u32; 8] = [
        0,
        1,
        0x7fff_ffff,
        0x8000_0000,
        0xffff_ffff,
        0x00ff_ffff,
        0x0100_0001,
        0xdead_beef,
    ];
    match EDGE.get(i as usize) {
        Some(&edge) => edge,
        None => i.wrapping_mul(0x9E37_79B9).wrapping_add(0x7F4A_7C15),
    }
}

/// The probe's float samples of pixels 0 to 7, `[sample][red, green]` bits: exact sums; 1, 1e8,
/// 1, -1e8, whose sum's order shows; 0.1 to 0.4; NaNs with payloads; infinities; a denormal
/// input and two normals whose average is a denormal; signed zeros; `FLT_MAX` twice.
const SPECIAL: [[[u32; 2]; 4]; 8] = [
    [
        [0x3f80_0000, 0x3f00_0000],
        [0x4000_0000, 0x3e80_0000],
        [0x4040_0000, 0x3e00_0000],
        [0x40a0_0000, 0x3d80_0000],
    ],
    [
        [0x3f80_0000, 0xccbe_bc20],
        [0x4cbe_bc20, 0x3f80_0000],
        [0x3f80_0000, 0x4cbe_bc20],
        [0xccbe_bc20, 0x3f80_0000],
    ],
    [
        [0x3dcc_cccd, 0xbdcc_cccd],
        [0x3e4c_cccd, 0xbe4c_cccd],
        [0x3e99_999a, 0xbe99_999a],
        [0x3ecc_cccd, 0xbecc_cccd],
    ],
    [
        [0x7fc1_2345, 0x3f80_0000],
        [0x3f80_0000, 0x4000_0000],
        [0x4000_0000, 0x4040_0000],
        [0x4040_0000, 0xffc5_4321],
    ],
    [
        [0x7f80_0000, 0x7f80_0000],
        [0x3f80_0000, 0xff80_0000],
        [0x3f80_0000, 0x0000_0000],
        [0x3f80_0000, 0x0000_0000],
    ],
    [
        [0x0001_16c2, 0x00d9_c7dd],
        [0x0000_0000, 0x00d9_c7dd],
        [0x0000_0000, 0x0000_0000],
        [0x0000_0000, 0x0000_0000],
    ],
    [
        [0x8000_0000, 0x0000_0000],
        [0x8000_0000, 0x8000_0000],
        [0x8000_0000, 0x0000_0000],
        [0x8000_0000, 0x8000_0000],
    ],
    [
        [0x7f7f_ffff, 0x7f7f_ffff],
        [0x7f7f_ffff, 0xff7f_ffff],
        [0x0000_0000, 0x7f7f_ffff],
        [0x0000_0000, 0xff7f_ffff],
    ],
];

/// The probe's float channel `c` (red 0, green 1) of pixel `p`'s sample `s`, as bits: pixels 0
/// to 7 [`SPECIAL`], then a 24-bit mantissa centred on 0 scaled by `2^e`, `e` in -4..11.
fn float_bits(p: u32, s: u32, c: u32) -> u32 {
    if let Some(pixel) = SPECIAL.get(p as usize) {
        return pixel[s as usize][c as usize];
    }
    let x = ((p * 4 + s) * 2 + c)
        .wrapping_mul(0x9E37_79B9)
        .wrapping_add(0x7F4A_7C15);
    let e = (x & 15) as i32 - 4;
    let scale = f32::from_bits(((127 + e) as u32) << 23);
    (((x >> 8) as f32 / 16_777_216.0 - 0.5) * scale).to_bits()
}

/// Metal's `RG32Float` texels of the probe's 4-sample fill resolved (`float-resolve RG32Float 4`,
/// `store` and `resolve` alike), `[red, green]` bits per pixel.
const METAL_RG32FLOAT_4X: [[u32; 2]; 16] = [
    [0x4030_0000, 0x3e70_0000],
    [0x0000_0000, 0x3e80_0000],
    [0x3e80_0000, 0xbe80_0000],
    [0x7fc0_0000, 0x7fc0_0000],
    [0x7f80_0000, 0x7fc0_0000],
    [0x0000_0000, 0x0000_0000],
    [0x8000_0000, 0x0000_0000],
    [0x7f80_0000, 0x0000_0000],
    [0x40fb_1667, 0x422d_51ca],
    [0xc293_c014, 0xc184_1696],
    [0x4067_03a5, 0x416c_2b0b],
    [0xc311_3979, 0xc11f_c1c3],
    [0xbf90_4b14, 0xc15b_f100],
    [0xc358_92ea, 0xc19b_ab2c],
    [0xc077_4eb2, 0xc229_4348],
    [0x4360_13a5, 0x420c_4544],
];

/// Little-endian bytes of `words`.
fn le_bytes(words: impl IntoIterator<Item = u32>) -> Vec<u8> {
    words.into_iter().flat_map(u32::to_le_bytes).collect()
}

/// The little-endian words of `bytes`.
fn le_words(bytes: &[u8]) -> Vec<u32> {
    let (words, rest) = bytes.as_chunks::<4>();
    assert!(rest.is_empty(), "{} bytes", bytes.len());
    words.iter().map(|&word| u32::from_le_bytes(word)).collect()
}

/// The device's answer agrees with what it makes: where it answers yes, a target of each of the
/// nine formats at every count its textures take above 1 (4 among them), each format's
/// capabilities listing `Msaa`, and a float format's `Resolve`; where it answers no, 4 not taken
/// or a target refused as a count its format lacks (`Unsupported`, `TextureSampleCount`). Printed
/// per venue (`MSAA32`).
#[test]
fn the_answer_is_what_the_device_makes() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let answer = device.supports_32bit_msaa();
        let counts = multisample_counts(&device);
        let mut refused = Vec::new();
        for pf in FORMATS {
            for &samples in &counts {
                match target(&device, pf, samples) {
                    Ok(texture) => assert_eq!(u64::from(samples), texture.sample_count()),
                    Err(Error::Unsupported(u)) if u.feature == Feature::TextureSampleCount => {
                        refused.push(format!("{pf:?} at {samples}: {u:?}"));
                    }
                    Err(error) => panic!("{pf:?} at {samples}: {error}"),
                }
            }
        }
        println!(
            "MSAA32 on {}: supports32BitMSAA {answer}; texture counts above 1 {counts:?}; \
             refused {refused:?}",
            device.name()
        );
        if answer {
            assert!(counts.contains(&4), "{counts:?}");
            assert!(refused.is_empty(), "{refused:?}");
            for pf in FORMATS {
                let capabilities = device.pixel_format_capabilities(pf);
                assert!(capabilities.contains(FormatCapability::Msaa), "{pf:?}");
                let float = matches!(
                    pf,
                    PixelFormat::R32Float | PixelFormat::RG32Float | PixelFormat::RGBA32Float
                );
                assert_eq!(
                    capabilities.contains(FormatCapability::Resolve),
                    float,
                    "{pf:?}"
                );
            }
        } else {
            assert!(!counts.contains(&4) || !refused.is_empty(), "{counts:?}");
        }
        // The M3 Max answers yes, and so does MoltenVK on it.
        if device.name().contains("Apple") {
            assert!(answer, "{}", device.name());
        }
    }
    layer_reported_nothing_at_all(&messages);
}

/// `R32Uint` and `RGBA32Uint` targets, at each count above 1 the device makes them at, drawn at
/// sample rate with the probe's words (four a sample, a format keeping its own channels), hold
/// every bit drawn, read back through `texture2d_ms<uint>`, as Metal's do.
#[test]
fn a_32bit_integer_target_holds_every_bit_drawn() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let queue = device.new_command_queue().unwrap();
        let mut drawn = 0;
        for (pf, fill, view, channels) in [
            (
                PixelFormat::R32Uint,
                CaseFormat::R32Uint,
                ReadbackView::R32Uint,
                1,
            ),
            (
                PixelFormat::RGBA32Uint,
                CaseFormat::Rgba32Uint,
                ReadbackView::Rgba32Uint,
                4,
            ),
        ] {
            for samples in multisample_counts(&device) {
                let label = format!("{pf:?} {samples}x");
                let texture = match target(&device, pf, samples) {
                    Ok(texture) => texture,
                    Err(error) => {
                        println!("MSAA32-INT {label} on {}: {error}", device.name());
                        assert!(!device.supports_32bit_msaa(), "{label}: {error}");
                        continue;
                    }
                };
                let written = le_bytes(
                    (0..SIDE * SIDE * samples)
                        .flat_map(|i| (0..channels).map(move |c| word(i * 4 + c))),
                );
                let layout =
                    texture_layout(CaseType::D2Multisample, [SIDE, SIDE, 1], samples).unwrap();
                let command_buffer = queue.command_buffer().unwrap();
                SampleFill::new(&device, &texture, fill, layout, &written)
                    .unwrap()
                    .encode(&command_buffer)
                    .unwrap();
                let read = samples_of(
                    &device,
                    &command_buffer,
                    &texture,
                    SampleRead::Color(view),
                    [SIDE, SIDE],
                    samples,
                )
                .unwrap();
                run(&command_buffer);
                let stored = read.bytes().unwrap();
                let differ = le_words(&stored)
                    .into_iter()
                    .zip(le_words(&written))
                    .filter(|(s, w)| s != w)
                    .count();
                println!(
                    "MSAA32-INT {label} on {}: {differ} of {} channels differ from those drawn",
                    device.name(),
                    written.len() / 4
                );
                assert_eq!(stored, written, "{label}");
                drawn += 1;
            }
        }
        if device.supports_32bit_msaa() {
            assert!(drawn >= 2, "{drawn} targets drawn");
        }
    }
    layer_reported_nothing_at_all(&messages);
}

/// An `RG32Float` 4-sample target drawn at sample rate with the probe's values and resolved by its
/// pass's `StoreAndMultisampleResolve` keeps its samples as written (a NaN may be stored as
/// another NaN, printed) and resolves to Metal's texels: exactly, with nothing announced, where
/// the device's average is measured to be Metal's; elsewhere each difference is printed, and
/// `device-rounded-resolve` is announced. Then an `RGBA32Uint` target's resolve is refused, as
/// Metal stops it.
#[test]
fn a_32bit_float_target_resolves_as_metal() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let name = device.name().to_owned();
        let queue = device.new_command_queue().unwrap();
        let samples = 4;
        match (
            target(&device, PixelFormat::RG32Float, samples),
            target(&device, PixelFormat::RG32Float, 1),
        ) {
            (Ok(msaa), Ok(resolved)) => {
                let written = le_bytes((0..SIDE * SIDE).flat_map(|p| {
                    (0..samples).flat_map(move |s| (0..2).map(move |c| float_bits(p, s, c)))
                }));
                let layout =
                    texture_layout(CaseType::D2Multisample, [SIDE, SIDE, 1], samples).unwrap();
                let command_buffer = queue.command_buffer().unwrap();
                SampleFill::new(&device, &msaa, CaseFormat::Rg32Float, layout, &written)
                    .unwrap()
                    .encode_resolving(&command_buffer, &resolved)
                    .unwrap();
                let read = samples_of(
                    &device,
                    &command_buffer,
                    &msaa,
                    SampleRead::Color(ReadbackView::Rg32Uint),
                    [SIDE, SIDE],
                    samples,
                )
                .unwrap();
                run(&command_buffer);
                let stored = read.bytes().unwrap();
                let got = blit_read(&device, &resolved, 8);
                let notices = counts(&messages.take_notices());
                let rounded = notices.contains_key(&Degradation::DeviceRoundedResolve);
                let nan = |bits: u32| f32::from_bits(bits).is_nan();
                for (s, w) in le_words(&stored).into_iter().zip(le_words(&written)) {
                    assert!(
                        s == w || (nan(s) && nan(w)),
                        "stored {s:08x}, drawn {w:08x}"
                    );
                }
                let differ: Vec<String> = le_words(&got)
                    .into_iter()
                    .zip(METAL_RG32FLOAT_4X.iter().flatten())
                    .enumerate()
                    .filter(|&(_, (g, &m))| g != m)
                    .map(|(at, (g, m))| {
                        format!(
                            "pixel {} channel {}: {g:08x}, Metal {m:08x}",
                            at / 2,
                            at % 2
                        )
                    })
                    .collect();
                let metals = device
                    .properties()
                    .driver_facts
                    .contains(&DriverFact::ColorResolveAverageIsMetals);
                println!(
                    "MSAA32-RESOLVE RG32Float 4x on {name}: {} of 32 channels as Metal's{}; \
                     device-rounded-resolve {}",
                    32 - differ.len(),
                    if differ.is_empty() {
                        String::new()
                    } else {
                        format!(", apart: {differ:?}")
                    },
                    if rounded {
                        "announced"
                    } else {
                        "not announced"
                    }
                );
                if metals {
                    assert!(differ.is_empty(), "{differ:?}");
                    assert!(!rounded, "a resolve measured to be Metal's announced");
                } else {
                    assert!(rounded, "a device-rounded resolve unannounced");
                }
            }
            (msaa, single) => {
                println!(
                    "MSAA32-RESOLVE RG32Float 4x on {name}: no target ({:?}, {:?})",
                    msaa.err(),
                    single.err()
                );
                assert!(!device.supports_32bit_msaa());
            }
        }
        // `int-resolve RGBA32Uint 4`, which Metal's validation stops.
        if let (Ok(msaa), Ok(single)) = (
            target(&device, PixelFormat::RGBA32Uint, samples),
            target(&device, PixelFormat::RGBA32Uint, 1),
        ) {
            let mut pass = RenderPassDescriptor::new();
            let attachment = &mut pass.color_attachments[0];
            attachment.texture = Some(msaa);
            attachment.resolve_texture = Some(single);
            attachment.store_action = StoreAction::MultisampleResolve;
            let opened = queue
                .command_buffer()
                .unwrap()
                .render_command_encoder(&pass)
                .map(drop);
            assert_eq!(
                opened,
                Err(Error::Invalid(Invalid::new(
                    "render_command_encoder",
                    Problem::RenderPass(RenderPassProblem::ResolveFormat(Attachment::Color(0))),
                )))
            );
        }
    }
    layer_reported_nothing_at_all(&messages);
}
