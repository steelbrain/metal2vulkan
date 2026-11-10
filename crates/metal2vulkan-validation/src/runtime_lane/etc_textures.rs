//! ETC2 and EAC textures on the device the cfg selects, under the validation layer, against
//! Metal's decode (`metal2vulkan::bind::EtcDecode`, matched against an Apple M3 Max's reads of
//! random blocks by its own tests).
//!
//! On a device without the formats ([`holds_etc`] false: desktop GPUs and lavapipe) the texture
//! holds its blocks in a carrier and its reads go through a decoded twin, announced
//! `carried-decoded-twin` once and, for an 11-bit format, `decoded-texel-precision` once. Every
//! format's texels read as Metal's after every kind of write ([`etc_texels_read_as_metal_s`]): an
//! 8-bit format's exactly (where the device does not sample through Metal's own texture unit,
//! within its conversion of the codes, announced `device-texel-conversion`: `converted_holds`; an
//! sRGB form's colour there the device's decode of the code, `srgb_decode_holds`), an 11-bit
//! format's within half a 16-bit step; a new texture's zero blocks, then blocks a CPU copy wrote,
//! a blit from a buffer, a fill of one block and a copy from another texture, each read by a
//! command buffer after the write's, each decoded into the twin (a blit's by a GPU decode pass, a
//! CPU copy's by one or on the CPU, counted in `WorkCounts`), and an 8-bit format's view as its
//! other colour space, which reads the same codes. The blocks each wrote come back unchanged from
//! `get_bytes` and from a blit into a buffer, on every device. A view of an 11-bit format as its
//! other signedness is refused there (`Feature::TextureView`). A device with the formats reads
//! through its own decode (not compared), announcing neither kind.

use super::astc_textures::{blit, download, read_whole, reader, texture, upload};
use super::degrade_notices::counts;
use super::degrade_notices::on_device_with;
use super::{at_plan, converted_holds, pixel_samplers::metal_sampler, srgb_decode_holds};
use metal2vulkan::bind::{EtcDecode, EtcTexels, PvrtcDecode};
use metal2vulkan::device::FormatUsage;
use metal2vulkan::metal::format::{srgb_raw_twin, vk_format, FormatEmulation};
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use metal2vulkan_runtime::{Degradation, Device, Error, Origin, Region, Size, Texture};

/// The ten ETC2 and EAC formats.
const FORMATS: [PixelFormat; 10] = [
    PixelFormat::ETC2RGB8,
    PixelFormat::ETC2RGB8SRGB,
    PixelFormat::ETC2RGB8A1,
    PixelFormat::ETC2RGB8A1SRGB,
    PixelFormat::EACRGBA8,
    PixelFormat::EACRGBA8SRGB,
    PixelFormat::EACR11Unorm,
    PixelFormat::EACR11Snorm,
    PixelFormat::EACRG11Unorm,
    PixelFormat::EACRG11Snorm,
];

/// The side, in texels, of every texture the lane reads (`astc_textures`'s): 4 x 4 blocks.
const SIDE: u32 = 16;

/// Whether `device` holds a texture of `pf` in its own format, decoding it its own way, rather
/// than in a carrier with a decoded twin.
fn holds_etc(device: &Device, pf: PixelFormat) -> bool {
    let usage = [FormatUsage::SampledImage].into();
    let mapping = vk_format(pf, &usage, device.compile_device().profile()).unwrap();
    mapping.emulation != FormatEmulation::Decoded
}

/// `count` blocks of `bytes` each of a seeded xorshift: random bits, which decode through every
/// mode.
fn seeded_blocks(seed: u64, count: usize, bytes: usize) -> Vec<u8> {
    let mut state = seed | 1;
    let mut out = Vec::with_capacity(count * bytes + 8);
    while out.len() < count * bytes {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        out.extend_from_slice(&state.wrapping_mul(0x2545_f491_4f6c_dd1d).to_le_bytes());
    }
    out.truncate(count * bytes);
    out
}

/// The `float` Metal reads of channel `c` of a `SIDE` x `SIDE` level of `pf`'s `blocks`, each
/// texel's four: an 8-bit code over 255, an sRGB form's colour through Metal's twelve-bit table
/// (the table PVRTC's and BC's sRGB forms read through), an 11-bit value over 2047 (signed 1023),
/// and an `R11` or `RG11` format's missing channels 0, alpha 1.
fn metal_texels(pf: PixelFormat, blocks: &[u8]) -> Vec<[f32; 4]> {
    let decode = EtcDecode::of(pf).unwrap();
    let size = [SIDE, SIDE, 1];
    let texels = decode.decode(EtcTexels::Float, blocks, size);
    if decode.eleven_bit() {
        let channels = decode.texel_bytes(EtcTexels::Float) as usize / 4;
        return (texels.chunks_exact(4 * channels))
            .map(|texel| {
                let value =
                    |k: usize| f32::from_le_bytes(texel[4 * k..4 * k + 4].try_into().unwrap());
                let green = if channels == 2 { value(1) } else { 0.0 };
                [value(0), green, 0.0, 1.0]
            })
            .collect();
    }
    let table = match srgb_raw_twin(pf).is_some() {
        true => PvrtcDecode::of(PixelFormat::PVRTCRGBA4BPPSRGB).unwrap(),
        false => PvrtcDecode::of(PixelFormat::PVRTCRGBA4BPP).unwrap(),
    };
    (texels.as_chunks::<4>().0.iter())
        .map(|codes| std::array::from_fn(|c| table.metal_value(c, codes[c])))
        .collect()
}

/// The most a compact 11-bit texel is off Metal's value: half a 16-bit step, with room for `f32`.
fn compact_bound(pf: PixelFormat) -> f32 {
    match pf {
        PixelFormat::EACR11Snorm | PixelFormat::EACRG11Snorm => 0.5 / 32767.0 + 1e-6,
        _ => 0.5 / 65535.0 + 1e-6,
    }
}

/// The other colour space's form of an 8-bit ETC2 or EAC format, or an 11-bit format's other
/// signedness.
fn other_form(pf: PixelFormat) -> PixelFormat {
    use PixelFormat as P;
    match pf {
        P::ETC2RGB8 => P::ETC2RGB8SRGB,
        P::ETC2RGB8SRGB => P::ETC2RGB8,
        P::ETC2RGB8A1 => P::ETC2RGB8A1SRGB,
        P::ETC2RGB8A1SRGB => P::ETC2RGB8A1,
        P::EACRGBA8 => P::EACRGBA8SRGB,
        P::EACRGBA8SRGB => P::EACRGBA8,
        P::EACR11Unorm => P::EACR11Snorm,
        P::EACR11Snorm => P::EACR11Unorm,
        P::EACRG11Unorm => P::EACRG11Snorm,
        _ => P::EACRG11Unorm,
    }
}

/// Every ETC2 and EAC format's texels read as Metal's on a device without the formats (the module
/// docs): a new texture's, then after a CPU copy, a blit from a buffer, a fill and a copy from
/// another texture, the blocks of each coming back unchanged.
#[test]
fn etc_texels_read_as_metal_s() {
    on_device_with(at_plan, |device, messages| {
        let name = device.name().to_owned();
        let state = reader(device);
        let mut wrong = Vec::new();
        let mut undecoded = Vec::new();
        let (mut twinned, mut eleven) = (false, false);
        // The GPU decode passes and CPU decodes so far.
        let decodes = |device: &Device| {
            let counts = device.work_counts();
            (counts.block_decode_passes, counts.cpu_block_decodes)
        };
        for (f, pf) in FORMATS.into_iter().enumerate() {
            let decode = EtcDecode::of(pf).unwrap();
            let native = holds_etc(device, pf);
            twinned |= !native;
            eleven |= !native && decode.eleven_bit();
            let bytes = decode.block_bytes() as usize;
            let row = u64::from(SIDE / 4) * bytes as u64;
            let count = (SIDE / 4 * SIDE / 4) as usize;
            let side = Size::new(SIDE.into(), SIDE.into(), 1);
            let whole = Region::new_2d(0, 0, SIDE.into(), SIDE.into());
            let check = |wrong: &mut Vec<String>,
                         what: &str,
                         texture: &Texture,
                         read_as: PixelFormat,
                         blocks: &[u8]| {
                let mut back = vec![0; blocks.len()];
                texture.get_bytes(&mut back, row, 0, whole, 0, 0).unwrap();
                if back != blocks {
                    wrong.push(format!("{pf:?} {what}: get_bytes changed the blocks"));
                }
                let length = blocks.len() as u64;
                if download(device, texture, (0, 0), side, (row, 0), length) != blocks {
                    wrong.push(format!(
                        "{pf:?} {what}: a blit into a buffer changed the blocks"
                    ));
                }
                if native {
                    return;
                }
                let got = read_whole(device, &state, texture);
                let want = metal_texels(read_as, blocks);
                let exact = !decode.eleven_bit();
                for (t, (got, want)) in got.iter().zip(&want).enumerate() {
                    for c in 0..4 {
                        let srgb = c < 3 && srgb_raw_twin(read_as).is_some();
                        let fits = match exact {
                            true if srgb => {
                                srgb_decode_holds(got[c], want[c], metal_sampler(device))
                            }
                            true => converted_holds(got[c], want[c], metal_sampler(device)),
                            false => (got[c] - want[c]).abs() <= compact_bound(pf),
                        };
                        if !fits && wrong.len() < 24 {
                            wrong.push(format!(
                                "{pf:?} {what} read as {read_as:?}, texel {t} channel {c}: {} \
                                 against Metal's {}",
                                got[c], want[c]
                            ));
                        }
                    }
                }
            };
            let texture = texture(device, pf, TextureUsage::SHADER_READ);
            check(&mut wrong, "fresh", &texture, pf, &vec![0; count * bytes]);
            let seed = 0xe7c2_0000 + 0x10 * f as u64;
            let written = seeded_blocks(seed, count, bytes);
            let before = decodes(device);
            texture
                .replace_region(whole, 0, 0, &written, row, 0)
                .unwrap();
            check(&mut wrong, "CPU copy", &texture, pf, &written);
            // A CPU copy's decode: a GPU pass where it goes through a submission, else the CPU's.
            let after = decodes(device);
            if !native && after == before {
                undecoded.push(format!("{pf:?} CPU copy: neither a pass nor a CPU decode"));
            }
            let blitted = seeded_blocks(seed + 1, count, bytes);
            let before = decodes(device);
            upload(device, &texture, (0, 0), side, (row, 0), &blitted);
            check(&mut wrong, "blit", &texture, pf, &blitted);
            // An 8-bit format's view in its other colour space reads the same codes; an 11-bit
            // one's as its other signedness is refused where the texture has a twin.
            let other = other_form(pf);
            match texture.new_texture_view(other) {
                // A device with the formats views them as it does.
                _ if native && decode.eleven_bit() => {}
                Ok(view) if !decode.eleven_bit() => {
                    check(&mut wrong, "blit", &view, other, &blitted)
                }
                Err(Error::Unsupported(_)) if decode.eleven_bit() => {}
                made => wrong.push(format!("{pf:?} viewed as {other:?}: {made:?}")),
            }
            let one = seeded_blocks(seed + 2, 1, bytes);
            blit(device, |encoder| {
                let blocks = Region::new_2d(0, 0, u64::from(SIDE / 4), u64::from(SIDE / 4));
                encoder.fill_texture(&texture, 0, 0, blocks, &one).unwrap();
            });
            check(&mut wrong, "fill", &texture, pf, &one.repeat(count));
            let copy = self::texture(device, pf, TextureUsage::SHADER_READ);
            let source = seeded_blocks(seed + 3, count, bytes);
            texture
                .replace_region(whole, 0, 0, &source, row, 0)
                .unwrap();
            let blits = decodes(device);
            blit(device, |encoder| {
                let at = Origin::new(0, 0, 0);
                encoder
                    .copy_from_texture_to_texture(&texture, 0, 0, at, side, &copy, 0, 0, at)
                    .unwrap();
            });
            check(&mut wrong, "texture copy", &copy, pf, &source);
            // Every blit's decode is a GPU pass: the upload's and the fill's, then the copy's.
            let after = decodes(device);
            if !native && (blits.0 < before.0 + 2 || after.0 == blits.0) {
                undecoded.push(format!(
                    "{pf:?} blits: GPU decode passes {before:?} -> {blits:?} -> {after:?}"
                ));
            }
        }
        assert!(wrong.is_empty(), "on {name}:\n{}", wrong.join("\n"));
        assert!(undecoded.is_empty(), "on {name}:\n{}", undecoded.join("\n"));
        let counts = counts(&messages.take_notices());
        let announced = |kind| counts.get(&kind).copied().unwrap_or(0);
        assert_eq!(
            (
                announced(Degradation::CarriedDecodedTwin),
                announced(Degradation::DecodedTexelPrecision)
            ),
            (usize::from(twinned), usize::from(eleven)),
            "on {name}"
        );
        assert_eq!(
            announced(Degradation::DeviceTexelConversion),
            usize::from(!metal_sampler(device)),
            "the device's conversion of the codes, on {name}"
        );
    });
}
