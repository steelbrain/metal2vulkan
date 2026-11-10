//! 4:2:2 textures (`GBGR422`, `BGRG422`; the runtime's `chroma_pairs.rs`) on the device the cfg
//! selects, under the validation layer, against what our probes measured on an M4 under macOS
//! 27.0.1 (each case plain twice and under Metal's API validation):
//!
//! - Either format makes a 16 x 16 2D texture, `Private` and `Shared`, of `ShaderRead`,
//!   `ShaderWrite` or `RenderTarget` usage. A mipmapped one, and a view of one in any other
//!   format, Metal refuses, plain too (`Invalid`).
//! - `replaceRegion` and `getBytes`, blits between a buffer and the texture, and copies between
//!   two such textures and into `RG8Unorm` and `R16Uint` move its 2-byte texels unchanged,
//!   whatever the region's x and width, the buffer's offset and its `bytesPerRow` (Metal's
//!   validation asks for 2 x 1 blocks of 4 bytes; plain Metal copies the texels).
//! - A copy into such a texture from a texture of another format is refused typed
//!   (`Unsupported`, `Feature::Format`): plain Metal writes `0xff` over part of each row.
//! - A kernel that only writes the texture completes and leaves it as it was, as plain Metal's
//!   `write()` does (its validation stops the binding).
//! - A kernel reads and samples Metal's texels (`metal2vulkan::bind::ChromaDecode`, matched
//!   against an Apple M3 Max's reads by its own tests; where the device does not sample through
//!   Metal's own texture unit, within its conversion of the codes, announced
//!   `device-texel-conversion`: `converted_holds`) through the texture's decoded twin,
//!   announced `carried-decoded-twin` once: a new texture's opaque black, then after a CPU copy
//!   of the whole texture and of part of two rows, a blit from a buffer of the whole texture and
//!   of part of a row, and a copy from another texture, each decoded into the twin in whole rows
//!   (a blit's by a GPU decode pass, a CPU copy's by one or on the CPU, counted in `WorkCounts`;
//!   [`four_two_two_texels_read_as_metal_s`]). Where the device converts the format itself with
//!   Metal's reconstruction (`WorkCounts::chroma_device_twins`: MoltenVK), the twin holds the
//!   device's own conversion of the pairs, its red and blue within a code of Metal's and its
//!   green Metal's, every decode a GPU pass, announced `decoded-texel-precision` once; elsewhere
//!   never.
//! - A render pipeline state of the format, which Metal's own validation aborts plain too, and a
//!   render pass into the texture, whose clear plain Metal stores in the format's encoding (not
//!   built), are refused typed (`Unsupported`, `Feature::Format`).

use super::astc_textures::{read_whole, reader};
use super::degrade_notices::on_device_with;
use super::degrade_notices::{counts, on_device};
use super::pixel_samplers::metal_sampler;
use super::{at_plan, buffer_with, converted_holds, read_buffer, WAIT};
use crate::air_library::AirLibrary;
use metal2vulkan::bind::ChromaDecode;
use metal2vulkan::error::Feature;
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, ComputePipelineDescriptor, RenderPipelineDescriptor, StageSource,
};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    BlitOption, Buffer, CommandBufferStatus, ComputePipelineState, Degradation, Device, Error,
    LoadAction, Origin, PipelineOptions, Region, RenderPassDescriptor, ResourceOptions, Size,
    StoreAction, Texture, TextureDescriptor,
};

/// The two 4:2:2 formats.
const PAIRS: [PixelFormat; 2] = [PixelFormat::GBGR422, PixelFormat::BGRG422];

/// The textures' side, and a row's bytes.
const SIDE: u64 = 16;
const ROW: u64 = 2 * SIDE;

/// `n` bytes from `base`, byte `i` `base + i` (wrapping).
fn seq(n: u64, base: u8) -> Vec<u8> {
    (0..n).map(|i| base.wrapping_add(i as u8)).collect()
}

/// A 2D texture of `pf`, [`SIDE`] square, of `usage` and `levels`, shared or private.
fn descriptor(pf: PixelFormat, usage: TextureUsage, shared: bool) -> TextureDescriptor {
    let mut d = TextureDescriptor::texture_2d(pf, SIDE, SIDE, false);
    d.usage = usage;
    d.resource_options = if shared {
        ResourceOptions::shared()
    } else {
        ResourceOptions::private()
    };
    d
}

fn texture(device: &Device, pf: PixelFormat, usage: TextureUsage, shared: bool) -> Texture {
    device
        .new_texture(&descriptor(pf, usage, shared))
        .unwrap_or_else(|e| panic!("{pf:?} {usage:?} shared {shared}: {e:?}"))
}

/// Run `record` in one blit encoder of a new command buffer, which completes.
fn blit(
    device: &Device,
    record: impl FnOnce(&mut metal2vulkan_runtime::BlitCommandEncoder) -> Result<(), Error>,
) -> Result<(), Error> {
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.blit_command_encoder().unwrap();
    let recorded = record(&mut encoder);
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    recorded
}

/// Fill the whole of `t` from `bytes` at [`ROW`] a row by a blit.
fn fill(device: &Device, t: &Texture, bytes: &[u8]) {
    let source = buffer_with(device, bytes).unwrap();
    blit(device, |e| {
        e.copy_from_buffer_to_texture(
            &source,
            0,
            ROW,
            ROW * SIDE,
            Size::new(SIDE, SIDE, 1),
            t,
            0,
            0,
            Origin::new(0, 0, 0),
            BlitOption::None,
        )
    })
    .unwrap();
}

/// The whole of `t`'s bytes, by a blit at [`ROW`] a row.
fn bytes_of(device: &Device, t: &Texture) -> Vec<u8> {
    let out: Buffer = buffer_with(device, &vec![0xee; (ROW * SIDE) as usize]).unwrap();
    blit(device, |e| {
        e.copy_from_texture_to_buffer(
            t,
            0,
            0,
            Origin::new(0, 0, 0),
            Size::new(SIDE, SIDE, 1),
            &out,
            0,
            ROW,
            ROW * SIDE,
            BlitOption::None,
        )
    })
    .unwrap();
    read_buffer(&out, 0, ROW * SIDE).unwrap()
}

/// `whole` with the bytes of `texels` texels from x `x` on rows `0..rows` replaced by `put`'s
/// consecutive rows of `pitch` bytes from `offset`: the bytes plain Metal's copy of a region
/// leaves.
fn placed(
    whole: &[u8],
    put: &[u8],
    (x, texels, rows): (u64, u64, u64),
    offset: u64,
    pitch: u64,
) -> Vec<u8> {
    let mut out = whole.to_vec();
    for row in 0..rows {
        let at = (row * ROW + 2 * x) as usize;
        let from = (offset + row * pitch) as usize;
        let n = (2 * texels) as usize;
        out[at..at + n].copy_from_slice(&put[from..from + n]);
    }
    out
}

/// Probe 8's creation: each format, `Private` and `Shared`, of each usage, made; a mipmapped
/// one and a view in another format refused as Metal's validation refuses them (probe 11
/// `mips_*`, `view_*`: Metal asserts plain too).
#[test]
fn four_two_two_textures_are_made_as_metal_makes_them() {
    on_device(|device, _| {
        for pf in PAIRS {
            for shared in [false, true] {
                for usage in [
                    TextureUsage::SHADER_READ,
                    TextureUsage::SHADER_WRITE,
                    TextureUsage::RENDER_TARGET,
                ] {
                    let t = texture(device, pf, usage, shared);
                    assert_eq!(t.pixel_format(), pf);
                }
            }
            let mut mipmapped = descriptor(pf, TextureUsage::SHADER_READ, false);
            mipmapped.mipmap_level_count = 5;
            let made = device.new_texture(&mipmapped);
            assert!(matches!(made, Err(Error::Invalid(_))), "{pf:?}: {made:?}");
            let t = texture(
                device,
                pf,
                TextureUsage::SHADER_READ | TextureUsage::PIXEL_FORMAT_VIEW,
                false,
            );
            let other = if pf == PixelFormat::GBGR422 {
                PixelFormat::BGRG422
            } else {
                PixelFormat::GBGR422
            };
            for view in [PixelFormat::RG8Unorm, PixelFormat::R16Uint, other] {
                let made = t.new_texture_view(view);
                assert!(
                    matches!(made, Err(Error::Invalid(_))),
                    "{pf:?} as {view:?}: {made:?}"
                );
            }
        }
    });
}

/// Probes 8 and 11's copies: whole textures, regions off the texel pairs, a source offset of 2,
/// pitches of 34 and 30 bytes, the CPU's and blits', and copies between textures, each moving
/// Metal's bytes.
#[test]
fn four_two_two_copies_move_the_texels_bytes() {
    on_device_with(at_plan, |device, _| {
        let pattern = seq(ROW * SIDE, 0);
        let ee = vec![0xee; (ROW * SIDE) as usize];
        for pf in PAIRS {
            // Whole: a blit in and out, and the CPU's copies.
            let private = texture(device, pf, TextureUsage::SHADER_READ, false);
            fill(device, &private, &pattern);
            assert_eq!(bytes_of(device, &private), pattern, "{pf:?} blit");
            let shared = texture(device, pf, TextureUsage::SHADER_READ, true);
            let whole = Region::new_2d(0, 0, SIDE, SIDE);
            shared
                .replace_region(whole, 0, 0, &pattern, ROW, 0)
                .unwrap();
            let mut got = vec![0; pattern.len()];
            shared.get_bytes(&mut got, ROW, 0, whole, 0, 0).unwrap();
            assert_eq!(got, pattern, "{pf:?} CPU");
            // Blits in of regions off the pairs, an offset of 2, pitches of 34 and 30.
            let source = seq(4096, 0x40);
            let source_buffer = buffer_with(device, &source).unwrap();
            for (x, w, h, offset, pitch) in [
                (1, 2, 1, 0, ROW),
                (2, 3, 1, 0, ROW),
                (3, 1, 1, 0, ROW),
                (14, 2, 2, 0, ROW),
                (0, SIDE, 2, 2, ROW),
                (0, SIDE, 4, 0, 34),
                (0, SIDE, 2, 0, 30),
            ] {
                let what =
                    format!("{pf:?} blit in x {x} w {w} h {h} offset {offset} pitch {pitch}");
                let t = texture(device, pf, TextureUsage::SHADER_READ, false);
                fill(device, &t, &ee);
                blit(device, |e| {
                    e.copy_from_buffer_to_texture(
                        &source_buffer,
                        offset,
                        pitch,
                        pitch * h,
                        Size::new(w, h, 1),
                        &t,
                        0,
                        0,
                        Origin::new(x, 0, 0),
                        BlitOption::None,
                    )
                })
                .unwrap_or_else(|e| panic!("{what}: {e:?}"));
                let want = placed(&ee, &source, (x, w, h), offset, pitch);
                assert_eq!(bytes_of(device, &t), want, "{what}");
            }
            // Blits out of such regions.
            for (x, w, offset, pitch) in [
                (1, 2, 0, ROW),
                (2, 3, 0, ROW),
                (3, 1, 0, ROW),
                (0, SIDE, 2, ROW),
                (0, SIDE, 0, 34),
                (2, 2, 0, 4),
            ] {
                let what = format!("{pf:?} blit out x {x} w {w} offset {offset} pitch {pitch}");
                let out = buffer_with(device, &[0xee; 1024]).unwrap();
                blit(device, |e| {
                    e.copy_from_texture_to_buffer(
                        &private,
                        0,
                        0,
                        Origin::new(x, 0, 0),
                        Size::new(w, 2, 1),
                        &out,
                        offset,
                        pitch,
                        pitch * 2,
                        BlitOption::None,
                    )
                })
                .unwrap_or_else(|e| panic!("{what}: {e:?}"));
                let mut want = vec![0xee; 1024];
                for row in 0..2u64 {
                    let to = (offset + row * pitch) as usize;
                    let from = (row * ROW + 2 * x) as usize;
                    let n = (2 * w) as usize;
                    want[to..to + n].copy_from_slice(&pattern[from..from + n]);
                }
                assert_eq!(read_buffer(&out, 0, 1024).unwrap(), want, "{what}");
            }
            // The CPU's copies of such regions and pitches.
            let put = seq(4096, 0xa0);
            for (x, w, h, pitch) in [(1, 2, 1, 4), (2, 3, 1, 6), (3, 1, 2, 2), (0, SIDE, 2, 34)] {
                let what = format!("{pf:?} replaceRegion x {x} w {w} h {h} pitch {pitch}");
                shared.replace_region(whole, 0, 0, &ee, ROW, 0).unwrap();
                shared
                    .replace_region(Region::new_2d(x, 0, w, h), 0, 0, &put, pitch, 0)
                    .unwrap_or_else(|e| panic!("{what}: {e:?}"));
                let mut got = vec![0; ee.len()];
                shared.get_bytes(&mut got, ROW, 0, whole, 0, 0).unwrap();
                assert_eq!(got, placed(&ee, &put, (x, w, h), 0, pitch), "{what}");
            }
            shared
                .replace_region(whole, 0, 0, &pattern, ROW, 0)
                .unwrap();
            for (x, w, pitch) in [(1, 2, 4), (2, 3, 6), (3, 1, 2), (0, SIDE, 34)] {
                let what = format!("{pf:?} getBytes x {x} w {w} pitch {pitch}");
                let mut got = vec![0xee; 128];
                shared
                    .get_bytes(&mut got, pitch, 0, Region::new_2d(x, 0, w, 2), 0, 0)
                    .unwrap_or_else(|e| panic!("{what}: {e:?}"));
                let mut want = vec![0xee; 128];
                for row in 0..2u64 {
                    let to = (row * pitch) as usize;
                    let from = (row * ROW + 2 * x) as usize;
                    let n = (2 * w) as usize;
                    want[to..to + n].copy_from_slice(&pattern[from..from + n]);
                }
                assert_eq!(got, want, "{what}");
            }
            // Between textures: off the pairs, into the other order, `RG8Unorm` and `R16Uint`.
            for (sx, dx, w) in [(1, 4, 2), (0, 3, 2), (2, 6, 3)] {
                let what = format!("{pf:?} texture copy {sx} -> {dx} w {w}");
                let d = texture(device, pf, TextureUsage::SHADER_READ, false);
                fill(device, &d, &ee);
                blit(device, |e| {
                    e.copy_from_texture_to_texture(
                        &private,
                        0,
                        0,
                        Origin::new(sx, 0, 0),
                        Size::new(w, 2, 1),
                        &d,
                        0,
                        0,
                        Origin::new(dx, 0, 0),
                    )
                })
                .unwrap_or_else(|e| panic!("{what}: {e:?}"));
                let mut want = ee.clone();
                for row in 0..2u64 {
                    let to = (row * ROW + 2 * dx) as usize;
                    let from = (row * ROW + 2 * sx) as usize;
                    let n = (2 * w) as usize;
                    want[to..to + n].copy_from_slice(&pattern[from..from + n]);
                }
                assert_eq!(bytes_of(device, &d), want, "{what}");
            }
            for other in [
                PixelFormat::GBGR422,
                PixelFormat::BGRG422,
                PixelFormat::RG8Unorm,
                PixelFormat::R16Uint,
            ] {
                let what = format!("{pf:?} into {other:?}");
                let d = texture(device, other, TextureUsage::SHADER_READ, false);
                blit(device, |e| {
                    e.copy_from_texture_to_texture(
                        &private,
                        0,
                        0,
                        Origin::new(0, 0, 0),
                        Size::new(SIDE, SIDE, 1),
                        &d,
                        0,
                        0,
                        Origin::new(0, 0, 0),
                    )
                })
                .unwrap_or_else(|e| panic!("{what}: {e:?}"));
                assert_eq!(bytes_of(device, &d), pattern, "{what}");
            }
            // Into it from another format: refused typed.
            let rg8 = texture(
                device,
                PixelFormat::RG8Unorm,
                TextureUsage::SHADER_READ,
                false,
            );
            let into = blit(device, |e| {
                e.copy_from_texture_to_texture(
                    &rg8,
                    0,
                    0,
                    Origin::new(0, 0, 0),
                    Size::new(SIDE, SIDE, 1),
                    &private,
                    0,
                    0,
                    Origin::new(0, 0, 0),
                )
            });
            assert!(
                matches!(&into, Err(Error::Unsupported(u)) if u.feature == Feature::Format),
                "RG8Unorm into {pf:?}: {into:?}"
            );
        }
    });
}

/// A compute state of the owned fixture `engine/{stem}`'s `entry`.
fn state(device: &Device, stem: &str, entry: &str) -> ComputePipelineState {
    let bitcode = llvm_assemble(&engine_dir().join(format!("{stem}.ll"))).unwrap();
    let library = device.new_library_with_air(&bitcode).unwrap();
    let function = library.new_function_with_name(entry).unwrap();
    device
        .new_compute_pipeline_state(
            &ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor())),
            &PipelineOptions::default(),
        )
        .unwrap()
}

/// Probe 11's `write_filled_*` (and probe 8's `write_*`): `storage_texture_write`, which writes
/// `(0.25, 0.5, 0.75, 1)` at each thread's position, over a filled 4:2:2 texture of `ShaderRead
/// | ShaderWrite` usage completes and leaves its bytes as they were; over an `RGBA8Unorm` one,
/// the control, it writes them (its tie within one unit of Metal's). `sampling`'s
/// `sample_pixel`, whose linear sample at texel (0, 0)'s centre reads that texel, reads Metal's
/// texel there (within one ulp: the device's filter converts the twin's codes).
#[test]
fn a_four_two_two_texture_written_by_a_kernel_keeps_its_texels() {
    on_device_with(at_plan, |device, _| {
        let write = state(device, "storage_texture_write", "storage_texture_write");
        let read = state(device, "sampling", "sample_pixel");
        let queue = device.new_command_queue().unwrap();
        let usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
        let pattern = seq(ROW * SIDE, 0);
        let dispatch = |t: &Texture| {
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer.compute_command_encoder().unwrap();
            encoder.set_compute_pipeline_state(&write).unwrap();
            encoder.set_texture(Some(t), 0).unwrap();
            let grid = Size::new(SIDE, SIDE, 1);
            encoder
                .dispatch_threads(grid, Size::new(SIDE, 1, 1))
                .unwrap();
            encoder.end_encoding().unwrap();
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        };
        for pf in PAIRS {
            let t = texture(device, pf, usage, false);
            fill(device, &t, &pattern);
            dispatch(&t);
            assert_eq!(bytes_of(device, &t), pattern, "{pf:?}: written");
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer.compute_command_encoder().unwrap();
            encoder.set_compute_pipeline_state(&read).unwrap();
            encoder.set_texture(Some(&t), 0).unwrap();
            let out = buffer_with(device, &[0; 1024]).unwrap();
            encoder.set_buffer(Some(&out), 0, 0).unwrap();
            let one = Size::new(1, 1, 1);
            encoder.dispatch_threads(one, one).unwrap();
            encoder.end_encoding().unwrap();
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
            let got = &metal_texels(pf, &pattern)[0];
            let read = read_buffer(&out, 0, 16).unwrap();
            for (c, want) in got.iter().enumerate() {
                let value = f32::from_le_bytes(read[4 * c..4 * c + 4].try_into().unwrap());
                assert!(
                    (value - want).abs() <= f32::EPSILON,
                    "{pf:?}: sampled channel {c}: {value} against Metal's {want}"
                );
            }
        }
        let control = texture(device, PixelFormat::RGBA8Unorm, usage, false);
        dispatch(&control);
        let mut bytes = vec![0; (4 * SIDE * SIDE) as usize];
        let readback = buffer_with(device, &bytes).unwrap();
        blit(device, |e| {
            e.copy_from_texture_to_buffer(
                &control,
                0,
                0,
                Origin::new(0, 0, 0),
                Size::new(SIDE, SIDE, 1),
                &readback,
                0,
                4 * SIDE,
                4 * SIDE * SIDE,
                BlitOption::None,
            )
        })
        .unwrap();
        bytes = read_buffer(&readback, 0, 4 * SIDE * SIDE).unwrap();
        // Metal's texel is (0x40, 0x80, 0xbf, 0xff). Green, 0.5, is a tie (127.5), and the device
        // converts a kernel's write to a normalized storage texture: the RTX 5080 Laptop GPU
        // stores 0x7f (storage writes). Within one unit of Metal's there, every other channel
        // exactly.
        assert_eq!(
            [bytes[0], bytes[2], bytes[3]],
            [0x40, 0xbf, 0xff],
            "the control is written"
        );
        assert!(
            bytes[1].abs_diff(0x80) <= 1,
            "the control's tie is written: {:#x}",
            bytes[1]
        );
    });
}

/// Metal's `float` texels of a [`SIDE`]-square 4:2:2 texture of `pf` holding `bytes`
/// (`ChromaDecode`): each code over 255.
fn metal_texels(pf: PixelFormat, bytes: &[u8]) -> Vec<[f32; 4]> {
    let decode = ChromaDecode::of(pf).unwrap();
    let codes = decode.decode(bytes, [SIDE as u32, SIDE as u32, 1]);
    (codes.as_chunks::<4>().0.iter())
        .map(|texel| texel.map(|code| f32::from(code) / 255.0))
        .collect()
}

/// Each format's texels read exactly as Metal's after every kind of write (the module docs): a
/// `Shared` texture's after a CPU copy of the whole texture and of 4 x 2 texels off the pairs'
/// rows, whose red and blue change beside the region too, then after a blit of the whole texture
/// and of 6 x 1 texels; a `Private` texture's after a copy from the `Shared` one. Each read is a
/// command buffer after the write. The CPU copies' decodes are a GPU pass or the CPU's, the
/// blits' GPU passes; the twin announces `carried-decoded-twin` once, and `cpu-block-decode` once
/// where a CPU copy decoded on the CPU.
#[test]
fn four_two_two_texels_read_as_metal_s() {
    on_device_with(at_plan, |device, messages| {
        let name = device.name().to_owned();
        let state = reader(device);
        let exact = metal_sampler(device);
        let usage = TextureUsage::SHADER_READ;
        let decodes = |device: &Device| {
            let counts = device.work_counts();
            (counts.block_decode_passes, counts.cpu_block_decodes)
        };
        let mut wrong = Vec::new();
        // The device's own conversion of the pairs (the module docs): red and blue within a
        // code of Metal's, each the device's nearest code.
        let converts = |device: &Device| device.work_counts().chroma_device_twins > 0;
        let check =
            |wrong: &mut Vec<String>, what: &str, pf: PixelFormat, t: &Texture, bytes: &[u8]| {
                assert_eq!(bytes_of(device, t), bytes, "{pf:?} {what}: its bytes");
                let got = read_whole(device, &state, t);
                let want = metal_texels(pf, bytes);
                let device_conversion = converts(device);
                for (at, (got, want)) in got.iter().zip(&want).enumerate() {
                    for c in 0..4 {
                        let holds = match c {
                            0 | 2 if device_conversion => {
                                (got[c] - want[c]).abs() <= 1.0 / 255.0 + 1e-6
                            }
                            _ => converted_holds(got[c], want[c], exact),
                        };
                        if !holds && wrong.len() < 24 {
                            wrong.push(format!(
                                "{pf:?} {what}, texel {at} channel {c}: {} against Metal's {}",
                                got[c], want[c]
                            ));
                        }
                    }
                }
            };
        let whole = Region::new_2d(0, 0, SIDE, SIDE);
        for (f, pf) in PAIRS.into_iter().enumerate() {
            let t = texture(device, pf, usage, true);
            check(&mut wrong, "fresh", pf, &t, &vec![0; (ROW * SIDE) as usize]);
            let written = seq(ROW * SIDE, 0x31 + 0x40 * f as u8);
            let before = decodes(device);
            t.replace_region(whole, 0, 0, &written, ROW, 0).unwrap();
            check(&mut wrong, "CPU copy", pf, &t, &written);
            // A part of two rows off the pairs: 4 texels from x 6 on rows 3 and 4.
            let patch = seq(16, 0xa0);
            let region = Region::new_2d(6, 3, 4, 2);
            t.replace_region(region, 0, 0, &patch, 8, 0).unwrap();
            let mut expected = written.clone();
            for row in 0..2 {
                let at = ((3 + row) * ROW + 2 * 6) as usize;
                expected[at..at + 8].copy_from_slice(&patch[8 * row as usize..][..8]);
            }
            check(
                &mut wrong,
                "CPU copy of part of two rows",
                pf,
                &t,
                &expected,
            );
            if decodes(device) == before {
                wrong.push(format!(
                    "{pf:?} CPU copies: neither a pass nor a CPU decode"
                ));
            }
            let blitted = seq(ROW * SIDE, 0x17 + 0x40 * f as u8);
            let before = decodes(device);
            fill(device, &t, &blitted);
            check(&mut wrong, "blit", pf, &t, &blitted);
            // 6 texels from x 2 on row 7.
            let row = seq(12, 0xc8);
            let source = buffer_with(device, &row).unwrap();
            blit(device, |e| {
                e.copy_from_buffer_to_texture(
                    &source,
                    0,
                    12,
                    12,
                    Size::new(6, 1, 1),
                    &t,
                    0,
                    0,
                    Origin::new(2, 7, 0),
                    BlitOption::None,
                )
            })
            .unwrap();
            let mut expected = blitted.clone();
            let at = (7 * ROW + 2 * 2) as usize;
            expected[at..at + 12].copy_from_slice(&row);
            check(&mut wrong, "blit of part of a row", pf, &t, &expected);
            let private = texture(device, pf, usage, false);
            blit(device, |e| {
                let at = Origin::new(0, 0, 0);
                let side = Size::new(SIDE, SIDE, 1);
                e.copy_from_texture_to_texture(&t, 0, 0, at, side, &private, 0, 0, at)
            })
            .unwrap();
            check(&mut wrong, "texture copy", pf, &private, &expected);
            let after = decodes(device);
            if after.0 < before.0 + 3 {
                wrong.push(format!(
                    "{pf:?} blits: GPU decode passes {before:?} -> {after:?}"
                ));
            }
        }
        assert!(wrong.is_empty(), "on {name}:\n{}", wrong.join("\n"));
        let counts = counts(&messages.take_notices());
        let announced = |kind| counts.get(&kind).copied().unwrap_or(0);
        assert_eq!(
            (
                announced(Degradation::CarriedDecodedTwin),
                announced(Degradation::DecodedTexelPrecision)
            ),
            (1, usize::from(converts(device))),
            "on {name}"
        );
        if converts(device) {
            assert_eq!(
                device.work_counts().cpu_block_decodes,
                0,
                "every decode a GPU pass where the device converts the format, on {name}"
            );
        }
        assert_eq!(
            announced(Degradation::DeviceTexelConversion),
            usize::from(!exact),
            "the device's conversion of the codes, on {name}"
        );
        assert_eq!(
            announced(Degradation::CpuBlockDecode),
            usize::from(device.work_counts().cpu_block_decodes > 0),
            "the CPU copies' decodes on the CPU, on {name}"
        );
    });
}

/// Probe 8's `pipeline_*` (Metal's validation aborts a render pipeline of the format, plain
/// too) and `pass_*` (a pass into the texture, whose clear plain Metal stores in the format's
/// encoding, probe 11's `pass_clear_*`; its validation stops it, "not color renderable"): both
/// refused typed, the pass as the runtime's gap.
#[test]
fn four_two_two_render_targets_are_refused_typed() {
    on_device_with(at_plan, |device, _| {
        let bitcode = llvm_assemble(&engine_dir().join("rt6_restart.ll")).unwrap();
        let library = device.new_library_with_air(&bitcode).unwrap();
        let function = |name| {
            library
                .new_function_with_name(name)
                .unwrap()
                .function_descriptor()
        };
        for pf in PAIRS {
            let mut descriptor = RenderPipelineDescriptor::new(StageSource::Function(function(
                "rt6_restart_vertex",
            )));
            descriptor.fragment = Some(StageSource::Function(function("rt6_restart_count")));
            descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(pf));
            let made = device.new_render_pipeline_state(&descriptor, &PipelineOptions::default());
            assert!(
                matches!(&made, Err(Error::Unsupported(u)) if u.feature == Feature::Format),
                "{pf:?}: pipeline: {:?}",
                made.err()
            );
            for usage in [
                TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ,
                TextureUsage::SHADER_READ,
            ] {
                let t = texture(device, pf, usage, false);
                let mut pass = RenderPassDescriptor::new();
                pass.color_attachments[0].texture = Some(t);
                pass.color_attachments[0].load_action = LoadAction::Clear;
                pass.color_attachments[0].store_action = StoreAction::Store;
                let queue = device.new_command_queue().unwrap();
                let command_buffer = queue.command_buffer().unwrap();
                let encoder = command_buffer.render_command_encoder(&pass);
                assert!(
                    matches!(&encoder, Err(Error::Unsupported(u)) if u.feature == Feature::Format),
                    "{pf:?} {usage:?}: pass: {:?}",
                    encoder.err()
                );
            }
        }
    });
}
