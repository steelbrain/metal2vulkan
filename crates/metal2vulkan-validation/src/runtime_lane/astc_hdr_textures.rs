//! ASTC HDR textures on the device the cfg selects, under the validation layer, against Metal's
//! decode (`metal2vulkan::bind::AstcDecode`'s HDR profile, matched bit for bit against an Apple
//! M3 Max by its own tests and the conformance suite's recordings).
//!
//! Metal makes a texture of every `_HDR` format, of `ShaderWrite` and `RenderTarget` usage too
//! (the probe `astc_hdr_textures`), and so does the runtime, natively or in a carrier:
//!
//! - Where the device holds the format's `SFLOAT` block ([`decodes_astc`]) its reads are the
//!   device's own decode, which through Metal's own texture unit (MoltenVK) is Metal's: compared
//!   there, every bit. A view in the LDR form reads the LDR decode of the same blocks.
//! - Elsewhere (lavapipe, the desktop GPUs) its blocks lie in a carrier and its reads go through a
//!   decoded twin of Metal's halves, exact: announced `carried-decoded-twin`, never
//!   `decoded-texel-precision`. A view in the LDR or sRGB form is another decode, refused.
//!
//! Every format's texels read and sample (nearest) as Metal's ([`astc_hdr_texels_read_as_metal_s`]):
//! a new texture's zero blocks NaN, then blocks a CPU copy wrote, a blit from a buffer, a fill of
//! an HDR void-extent block and a copy from another texture, each decoded into a twin (a blit's by
//! a GPU decode pass; a CPU copy's by a pass or on the CPU, counted in `WorkCounts`). The blocks
//! each wrote come back unchanged from `get_bytes` and a blit into a buffer. Apple's 4x2 and 8x4
//! footprints, which no Vulkan format holds, lie in a carrier on every device. A pass rendering into
//! one and a kernel's write of one, which Metal's validation stops (plain Metal's pass stores
//! nothing, its write leaves bytes no format defines), are refused, typed
//! ([`astc_hdr_textures_of_every_usage_are_made`]).

use super::astc_textures::{
    blit, block_count, blocks_of, decodes_astc, download, metal_texels, read_whole, reader,
    row_bytes, texels_of, texture, upload, SIDE,
};
use super::degrade_notices::{counts, on_device};
use super::pixel_samplers::metal_sampler;
use super::{buffer_with, read_buffer, WAIT};
use crate::air_library::AirLibrary;
use metal2vulkan::bind::{AstcDecode, SamplerDescriptor};
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    ClearColor, CommandBufferStatus, ComputePipelineState, Degradation, Device, Error, LoadAction,
    Origin, PipelineOptions, Region, RenderPassDescriptor, SamplerState, Size, StoreAction,
    Texture,
};

/// Every ASTC HDR format.
const FORMATS: [PixelFormat; 16] = [
    PixelFormat::ASTC4x4HDR,
    PixelFormat::ASTC4x2HDR,
    PixelFormat::ASTC5x4HDR,
    PixelFormat::ASTC5x5HDR,
    PixelFormat::ASTC6x5HDR,
    PixelFormat::ASTC6x6HDR,
    PixelFormat::ASTC8x4HDR,
    PixelFormat::ASTC8x5HDR,
    PixelFormat::ASTC8x6HDR,
    PixelFormat::ASTC8x8HDR,
    PixelFormat::ASTC10x5HDR,
    PixelFormat::ASTC10x6HDR,
    PixelFormat::ASTC10x8HDR,
    PixelFormat::ASTC10x10HDR,
    PixelFormat::ASTC12x10HDR,
    PixelFormat::ASTC12x12HDR,
];

/// An HDR void-extent block of the halves 2, -0.5, 65504 and 1: every texel that color.
const HDR_VOID_EXTENT: [u8; 16] = [
    0xfc, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x40, 0x00, 0xb8, 0xff, 0x7b, 0x00, 0x3c,
];

/// `count` 16-byte blocks of a seeded xorshift: random bits, of which most are error blocks and
/// the rest decode through every mode, LDR and HDR, every fourth an HDR void-extent block of
/// random halves.
fn seeded_blocks(seed: u64, count: usize) -> Vec<u8> {
    let mut state = seed | 1;
    let mut next = move || {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        state.wrapping_mul(0x2545_f491_4f6c_dd1d)
    };
    let mut blocks = Vec::with_capacity(16 * count);
    for index in 0..count {
        let mut block = [0; 16];
        block[..8].copy_from_slice(&next().to_le_bytes());
        block[8..].copy_from_slice(&next().to_le_bytes());
        if index % 4 == 3 {
            block[..8].copy_from_slice(&HDR_VOID_EXTENT[..8]);
        }
        blocks.extend_from_slice(&block);
    }
    blocks
}

/// The format scout's `scout_sample_f` on `device`, which samples each texel of a 16 x 16 texture
/// a quarter and three quarters of the way into it.
fn sampler_reader(device: &Device) -> ComputePipelineState {
    let bitcode = llvm_assemble(&engine_dir().join("format_scout.ll")).unwrap();
    let library = device.new_library_with_air(&bitcode).unwrap();
    let function = library.new_function_with_name("scout_sample_f").unwrap();
    device
        .new_compute_pipeline_state(
            &ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor())),
            &PipelineOptions::default(),
        )
        .unwrap()
}

/// The `float4` texels of `texture` (`SIDE` x `SIDE`) as `state` ([`sampler_reader`]) samples
/// them through `sampler` (nearest: each texel's own), in one command buffer, waited for.
fn sample_whole(
    device: &Device,
    state: &ComputePipelineState,
    sampler: &SamplerState,
    texture: &Texture,
) -> Vec<[f32; 4]> {
    let texels = u64::from(SIDE * SIDE);
    let out = buffer_with(device, &vec![0; 16 * texels as usize]).unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(state).unwrap();
    encoder.set_texture(Some(texture), 0).unwrap();
    encoder.set_sampler_state(Some(sampler), 0).unwrap();
    encoder.set_buffer(Some(&out), 0, 0).unwrap();
    let side = u64::from(SIDE);
    encoder
        .dispatch_threads(Size::new(side, side, 1), Size::new(side, side, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    texels_of(&read_buffer(&out, 0, 16 * texels).unwrap())
}

/// Whether `got` is Metal's `want`, bit for bit, any NaN for a NaN.
fn same(got: f32, want: f32) -> bool {
    got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan())
}

/// The first `limit` texels of `got` that are not Metal's `want`, named for `what`.
fn differences(what: &str, got: &[[f32; 4]], want: &[[f32; 4]], limit: usize) -> Vec<String> {
    (got.iter().zip(want).enumerate())
        .flat_map(|(t, (got, want))| (0..4).map(move |c| (t, c, got[c], want[c])))
        .filter(|&(_, _, got, want)| !same(got, want))
        .take(limit)
        .map(|(t, c, got, want)| {
            format!("{what} texel {t} channel {c}: {got} against Metal's {want}")
        })
        .collect()
}

/// Every ASTC HDR format's texels read and sample as Metal's (the module docs): a new texture's,
/// then after a CPU copy, a blit from a buffer, a fill and a copy from another texture, the blocks
/// of each coming back unchanged.
#[test]
fn astc_hdr_texels_read_as_metal_s() {
    on_device(|device, messages| {
        let name = device.name().to_owned();
        let state = reader(device);
        let sampling = sampler_reader(device);
        let sampler = device.new_sampler_state(&SamplerDescriptor::new()).unwrap();
        let metal = metal_sampler(device);
        let mut wrong = Vec::new();
        let mut undecoded = Vec::new();
        let mut paths = (0, 0);
        // The GPU decode passes and CPU decodes so far.
        let decodes = |device: &Device| {
            let counts = device.work_counts();
            (counts.block_decode_passes, counts.cpu_block_decodes)
        };
        for (f, pf) in FORMATS.into_iter().enumerate() {
            let decode = AstcDecode::of(pf).unwrap();
            let native = decodes_astc(device, pf);
            match native {
                true => paths.0 += 1,
                false => paths.1 += 1,
            }
            let row = row_bytes(decode);
            let side = Size::new(SIDE.into(), SIDE.into(), 1);
            let whole = Region::new_2d(0, 0, SIDE.into(), SIDE.into());
            let mut check = |what: &str, texture: &Texture, blocks: &[u8]| {
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
                // A device's own decode is compared only through Metal's own texture unit.
                if native && !metal {
                    return;
                }
                let want = metal_texels(decode, blocks);
                let read = read_whole(device, &state, texture);
                wrong.extend(differences(&format!("{pf:?} {what} read"), &read, &want, 8));
                let sampled = sample_whole(device, &sampling, &sampler, texture);
                let at = format!("{pf:?} {what} sample");
                wrong.extend(differences(&at, &sampled, &want, 8));
            };
            let usage = TextureUsage::SHADER_READ;
            let texture = texture(device, pf, usage);
            // A new texture's zero blocks are error blocks: NaN.
            let count = block_count(decode);
            check("fresh", &texture, &vec![0; count * 16]);
            let seed = 0x4d72_0000 + 0x10 * f as u64;
            let written = seeded_blocks(seed, count);
            let before = decodes(device);
            texture
                .replace_region(whole, 0, 0, &written, row, 0)
                .unwrap();
            check("CPU copy", &texture, &written);
            // A CPU copy's decode: a GPU pass where it goes through a submission, else the CPU's.
            let after = decodes(device);
            if !native && after == before {
                undecoded.push(format!("{pf:?} CPU copy: neither a pass nor a CPU decode"));
            }
            let blitted = seeded_blocks(seed + 1, count);
            let before = decodes(device);
            upload(device, &texture, (0, 0), side, (row, 0), &blitted);
            check("blit", &texture, &blitted);
            let [across, down] = blocks_of(decode, SIDE.into(), SIDE.into());
            blit(device, |encoder| {
                let blocks = Region::new_2d(0, 0, across, down);
                encoder
                    .fill_texture(&texture, 0, 0, blocks, &HDR_VOID_EXTENT)
                    .unwrap();
            });
            check("fill", &texture, &HDR_VOID_EXTENT.repeat(count));
            let copy = self::texture(device, pf, usage);
            let source = seeded_blocks(seed + 2, count);
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
            check("texture copy", &copy, &source);
            // Every blit's decode is a GPU pass: the upload's and the fill's, then the copy's.
            let after = decodes(device);
            if !native && (blits.0 < before.0 + 2 || after.0 == blits.0) {
                undecoded.push(format!(
                    "{pf:?} blits: GPU decode passes {before:?} -> {blits:?} -> {after:?}"
                ));
            }
        }
        eprintln!(
            "ASTCHDR {name}: {} formats in their own block, {} in a carrier; Metal's sampler {metal}",
            paths.0, paths.1
        );
        assert!(wrong.is_empty(), "on {name}:\n{}", wrong.join("\n"));
        assert!(undecoded.is_empty(), "on {name}:\n{}", undecoded.join("\n"));
        let counts = counts(&messages.take_notices());
        let announced = |kind| counts.get(&kind).copied().unwrap_or(0);
        let carried = usize::from(paths.1 > 0);
        assert_eq!(
            announced(Degradation::CarriedDecodedTwin),
            carried,
            "on {name}"
        );
        assert_eq!(
            announced(Degradation::DecodedTexelPrecision),
            0,
            "on {name}"
        );
    });
}

/// A texture of `ShaderWrite` and `RenderTarget` usage too is made, as Metal makes it; a pass
/// rendering into it and a kernel writing it are refused, typed, where Metal's validation stops
/// each. A view in its LDR form reads the LDR decode of its blocks where the device holds
/// both blocks, and is refused where its blocks lie in a carrier (another decode).
#[test]
fn astc_hdr_textures_of_every_usage_are_made() {
    on_device(|device, messages| {
        let name = device.name().to_owned();
        let pf = PixelFormat::ASTC8x8HDR;
        let usage = TextureUsage::SHADER_READ
            | TextureUsage::SHADER_WRITE
            | TextureUsage::RENDER_TARGET
            | TextureUsage::PIXEL_FORMAT_VIEW;
        let target = texture(device, pf, usage);
        let mut pass = RenderPassDescriptor::new();
        let color = &mut pass.color_attachments[0];
        color.texture = Some(target.clone());
        color.load_action = LoadAction::Clear;
        color.store_action = StoreAction::Store;
        color.clear_color = ClearColor::new(0.5, 4.0, 8.0, 1.0);
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let encoder = command_buffer.render_command_encoder(&pass).map(drop);
        assert!(
            matches!(encoder, Err(Error::Unsupported(_))),
            "on {name}: {encoder:?}"
        );
        // A kernel's write of it, which Metal's validation stops too (plain Metal leaves bytes
        // no format defines).
        let bitcode = llvm_assemble(&engine_dir().join("format_scout.ll")).unwrap();
        let library = device.new_library_with_air(&bitcode).unwrap();
        let function = library.new_function_with_name("scout_write_f").unwrap();
        let writer = device
            .new_compute_pipeline_state(
                &ComputePipelineDescriptor::new(StageSource::Function(
                    function.function_descriptor(),
                )),
                &PipelineOptions::default(),
            )
            .unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(&writer).unwrap();
        let side = u64::from(SIDE);
        let written = encoder.set_texture(Some(&target), 0).and_then(|()| {
            encoder.dispatch_threads(Size::new(side, side, 1), Size::new(side, side, 1))
        });
        // The device lacks the format as a storage image: its binding is refused.
        assert!(
            matches!(written, Err(Error::Bind(_) | Error::Unsupported(_))),
            "on {name}: {written:?}"
        );
        let decode = AstcDecode::of(pf).unwrap();
        let blocks = seeded_blocks(0x4d72_ffff, block_count(decode));
        let whole = Region::new_2d(0, 0, SIDE.into(), SIDE.into());
        target
            .replace_region(whole, 0, 0, &blocks, row_bytes(decode), 0)
            .unwrap();
        let view = target.new_texture_view(PixelFormat::ASTC8x8LDR);
        if !decodes_astc(device, pf) {
            assert!(
                matches!(view, Err(Error::Unsupported(_))),
                "on {name}: {view:?}"
            );
            let counts = counts(&messages.take_notices());
            assert_eq!(counts.get(&Degradation::CarriedDecodedTwin), Some(&1));
            return;
        }
        let view = view.unwrap();
        if metal_sampler(device) {
            let ldr = AstcDecode::of(PixelFormat::ASTC8x8LDR).unwrap();
            let got = read_whole(device, &reader(device), &view);
            let want = metal_texels(ldr, &blocks);
            let wrong = differences("LDR view read", &got, &want, 8);
            assert!(wrong.is_empty(), "on {name}:\n{}", wrong.join("\n"));
        }
    });
}
