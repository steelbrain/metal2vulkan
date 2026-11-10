//! PVRTC textures on the device the cfg selects, under the validation layer, against what our probe
//! measured on an M4 (macOS 27.0.1), alike plain twice and the row-linear copies
//! alike under Metal's API validation:
//!
//! - Every PVRTC format makes a texture, `Private` and `Shared`, of levels down to 1 x 1, and a
//!   shareable one opens over the same blocks; where the device has no format for it, the
//!   runtime holds the blocks in a carrier (`pvrtc.rs`), a 2-bit texture's half as wide.
//! - With `MTLBlitOptionRowLinearPVRTC` and a `bytesPerRow`, copies between buffers and the
//!   texture move the blocks row by row: whole levels, a region of whole blocks, from an offset,
//!   rows padded apart, and each level of a full chain, the levels under 4 x 4 one block each.
//!   A copy between two such textures moves the blocks. Without the option and with a nonzero
//!   `bytesPerRow`, which Metal's validation stops, plain Metal copies as with it (on an M4, macOS
//!   27.0.1, and the reims-vgpu consumer's run on an M3 Max, macOS 26.7).
//! - The option on another format is ignored, as plain Metal ignores it (`RGBA8Unorm` and
//!   `BC1_RGBA`; Metal's validation stops it on `BC1_RGBA`).
//! - At a `bytesPerRow` of 0, with the option or without it (Metal's validation stops both), the
//!   blocks lie in the region's own Morton order, the x bit lowest, whole levels and regions of
//!   whole blocks alike, a position no block takes neither read nor written (on
//!   an M4, macOS 27.0.1; the reims-vgpu consumer's report of a whole 16 x 16 level on an M3 Max,
//!   macOS 26.7).
//! - A copy at a `bytesPerRow` of 0 of a region off its blocks copies the blocks it is rounded
//!   to, its origin down and its size up, in their own Morton order.
//! - `replaceRegion` and `getBytes` of a whole level take its blocks in plain Metal's CPU order,
//!   Z order with the y bit lowest, at a `bytesPerRow` of 0, of the level's row, and of any value
//!   above it: plain Metal moves the level's bytes contiguously and ignores the pitch (measured on
//!   this M3 Max under macOS 27.0.1, plain Metal with no validation layer, both bit depths, two
//!   runs byte-identical; `runtime/src/texture.rs` states the rule).
//! - The runtime's gap, `NotImplemented(PvrtcBlockOrder)`: `replaceRegion` and `getBytes` of a
//!   region other than a whole level, which plain Metal does not place by the region at all -- it
//!   ignores the origin, reads blocks outside the region, and for an 8x7-block region wrote 224
//!   bytes past the 448 the region accounts for -- and of a whole level at a nonzero `bytesPerRow`
//!   below its row, where the AGX driver asserts and the call moves nothing.
//! - Where the device has no format for it, its reads go through a decoded twin of its codes,
//!   announced `carried-decoded-twin` once: every format's texels read exactly as Metal's
//!   (`metal2vulkan::bind::PvrtcDecode`, matched against an Apple M3 Max by its own tests; where
//!   the device does not sample through Metal's own texture unit, within its conversion of the
//!   codes, announced `device-texel-conversion`: `converted_holds`, an sRGB form's colour the
//!   device's decode of its code, announced `srgb-texel-decode`: `srgb_decode_holds`), and so
//!   do its view's as the format's other colour space, after a CPU copy, a row-linear blit and a
//!   copy from another texture, each decoded into the twin (a blit's by a GPU decode pass, a CPU
//!   copy's by one or on the CPU, counted in `WorkCounts`; [`pvrtc_texels_read_as_metal_s`]).
//! - A texture held for copies alone (a 4:2:2 texture in a heap over the caller's host pages, which
//!   keeps no decoded twin, as a PVRTC texture there keeps none), made resident for a kernel that
//!   reads texture handles from memory, takes the neutral's place there, and the dispatch announces
//!   it once ([`UNREADABLE`]); a kernel that reads no handles announces nothing.

use super::astc_textures::{read_whole, reader};
use super::case_making_kinds;
use super::degrade_notices::{assert_made_once, counts, shared_in_system_memory};
use super::host_memory::Mapping;
use super::no_copy_heaps::{no_copy_heap, shared, HEAP};
use super::pixel_samplers::metal_sampler;
use super::{
    at_plan, bound_buffers, converted_holds, encode_case, layer_reported_nothing_at_all,
    pipeline_state, public_case, read_buffer, validated_device, CaseResources, Messages,
    PublicCase, ScratchCache, WAIT,
};
use crate::candidate::engine::ComputeOutput;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::bind::PvrtcDecode;
use metal2vulkan::device::FormatUsage;
use metal2vulkan::metal::format::{vk_format, FormatEmulation};
use metal2vulkan::metal::{PixelFormat, ResourceUsage, TextureUsage};
use metal2vulkan_runtime::{
    BlitCommandEncoder, BlitOption, Buffer, CommandBufferStatus, CommandQueue, CpuCacheMode,
    Degradation, Device, Error, HazardTrackingMode, HeapType, Origin, Region, ResourceOptions,
    RuntimeGap, Size, StorageMode, Texture, TextureDescriptor,
};

/// The 4-bit PVRTC formats, which the runtime holds where the device has no format for them.
const FOUR_BIT: [PixelFormat; 4] = [
    PixelFormat::PVRTCRGB4BPP,
    PixelFormat::PVRTCRGB4BPPSRGB,
    PixelFormat::PVRTCRGBA4BPP,
    PixelFormat::PVRTCRGBA4BPPSRGB,
];

/// The 2-bit PVRTC formats, of 8 x 4-texel blocks: a device with their Vulkan format (MoltenVK's
/// `VK_IMG_format_pvrtc`, the RGBA ones) keeps them in it, and the runtime holds every other in
/// a carrier image half as wide.
const TWO_BIT: [PixelFormat; 4] = [
    PixelFormat::PVRTCRGB2BPP,
    PixelFormat::PVRTCRGB2BPPSRGB,
    PixelFormat::PVRTCRGBA2BPP,
    PixelFormat::PVRTCRGBA2BPPSRGB,
];

/// A public case whose kernel reads texture handles from memory: it samples an unsigned texture
/// through a sampler, both held in its argument buffer (`resource_ids.rs`'s).
const HANDLES: &str =
    "sample-an-unsigned-texture-through-a-sampler-held-beside-it-in-an-argument-buffer";

/// A public case whose kernel reads buffers alone.
const BUFFERS: &str = "store-through-a-selected-base-and-test-a-nullable-pointer";

/// The kind [`a_texture_held_for_copies_a_kernel_s_handles_reach_is_announced_once`] takes.
pub(super) const UNREADABLE: [Degradation; 1] = [Degradation::UnreadableHandleTexture];

/// The option a PVRTC copy takes.
const ROWS: BlitOption = BlitOption::RowLinearPVRTC;

/// `n` bytes of the probe's blocks: each 8-byte block's first byte its index, the others `tag`
/// plus their place in the block.
fn blocks(n: usize, tag: u8) -> Vec<u8> {
    (0..n)
        .map(|i| {
            if i % 8 == 0 {
                (i / 8) as u8
            } else {
                tag + (i % 8) as u8
            }
        })
        .collect()
}

/// The device, its queue, and the validation layer's messages.
struct Lane {
    device: Device,
    queue: CommandQueue,
    messages: Messages,
}

/// Run `test` on a validated device of the lane, then fail on any validation message.
fn on_device(test: impl FnOnce(&Lane)) {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let queue = device.new_command_queue().unwrap();
        test(&Lane {
            device,
            queue,
            messages: messages.clone(),
        });
    }
    layer_reported_nothing_at_all(&messages);
}

impl Lane {
    /// A square 2D texture of `pf`, `side` texels, `levels` levels, in `storage`, read by shaders.
    fn texture(&self, pf: PixelFormat, side: u64, levels: u64, storage: StorageMode) -> Texture {
        self.device
            .new_texture(&descriptor(pf, side, levels, storage))
            .unwrap()
    }

    /// A Shared buffer holding `bytes`.
    fn buffer(&self, bytes: &[u8]) -> Buffer {
        let buffer = self
            .device
            .new_buffer(bytes.len() as u64, ResourceOptions::shared())
            .unwrap();
        // SAFETY: the buffer holds `bytes.len()` bytes at its contents, and no command buffer
        // uses it yet.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                buffer.contents().unwrap().as_ptr(),
                bytes.len(),
            );
        }
        buffer
    }

    /// The output of `case`'s dispatch in one command buffer whose encoder made `resident`
    /// resident for reading first, where given.
    fn dispatch(&self, case: &PublicCase, resident: Option<&Texture>) -> Vec<u8> {
        let pipeline = pipeline_state(&self.device, case).unwrap();
        let resources = CaseResources::new(&self.device, case).unwrap();
        let buffers = bound_buffers(&self.device, &case.compute, &resources).unwrap();
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        if let Some(texture) = resident {
            encoder
                .use_resource(texture.as_resource(), ResourceUsage::READ)
                .unwrap();
        }
        encode_case(&mut encoder, &pipeline, &case.compute, &buffers, &resources).unwrap();
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(
            command_buffer.status(),
            CommandBufferStatus::Completed,
            "{:?}",
            command_buffer.error()
        );
        let ComputeOutput::Buffer {
            binding,
            offset,
            length,
        } = case.compute.output
        else {
            panic!("{}: a buffer output", case.case.name);
        };
        read_buffer(&buffers[&binding], offset, length).unwrap()
    }

    /// Run `record` in one blit encoder; answer the bytes of each buffer it returns.
    fn run(
        &self,
        record: impl FnOnce(&mut BlitCommandEncoder) -> Result<Vec<Buffer>, Error>,
    ) -> Result<Vec<Vec<u8>>, Error> {
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        let recorded = record(&mut blit);
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        Ok(recorded?
            .into_iter()
            .map(|buffer| {
                let n = buffer.length() as usize;
                // SAFETY: the command buffer completed; the buffer holds `n` bytes.
                unsafe { std::slice::from_raw_parts(buffer.contents().unwrap().as_ptr(), n) }
                    .to_vec()
            })
            .collect())
    }

    /// Copy `bytes` into `at`.
    fn put(
        &self,
        blit: &mut BlitCommandEncoder,
        texture: &Texture,
        bytes: &[u8],
        at: At,
    ) -> Result<(), Error> {
        blit.copy_from_buffer_to_texture(
            &self.buffer(bytes),
            at.offset,
            at.bytes_per_row,
            at.bytes_per_image,
            at.region.size,
            texture,
            0,
            at.level,
            at.region.origin,
            at.option,
        )
    }

    /// `at` into a new buffer of `n` 0xee bytes (`at.offset` 0).
    fn get(
        &self,
        blit: &mut BlitCommandEncoder,
        texture: &Texture,
        n: usize,
        at: At,
    ) -> Result<Buffer, Error> {
        let buffer = self.buffer(&vec![0xee; n]);
        blit.copy_from_texture_to_buffer(
            texture,
            0,
            at.level,
            at.region.origin,
            at.region.size,
            &buffer,
            0,
            at.bytes_per_row,
            n as u64,
            at.option,
        )?;
        Ok(buffer)
    }
}

/// Where a copy's blocks lie: `region` of `level`, and the buffer's rows from `offset`,
/// `bytes_per_row` apart, copied with `option`.
#[derive(Clone, Copy, Debug)]
struct At {
    region: Region,
    level: u64,
    offset: u64,
    bytes_per_row: u64,
    /// The buffer's bytes a copy in reads, `bytesPerImage`.
    bytes_per_image: u64,
    option: BlitOption,
}

impl At {
    /// `region` of level 0, rows of 4-texel-high blocks `bytes_per_row` apart from the buffer's
    /// start, row by row.
    fn rows(region: Region, bytes_per_row: u64) -> At {
        At {
            region,
            level: 0,
            offset: 0,
            bytes_per_row,
            bytes_per_image: bytes_per_row * region.size.height.div_ceil(4),
            option: ROWS,
        }
    }

    /// The same, `bytes_per_image` apart.
    fn image(self, bytes_per_image: u64) -> At {
        At {
            bytes_per_image,
            ..self
        }
    }

    /// The same at `level`.
    fn level(self, level: u64) -> At {
        At { level, ..self }
    }

    /// The same from `offset`.
    fn from(self, offset: u64) -> At {
        At { offset, ..self }
    }

    /// The same with `option`.
    fn option(self, option: BlitOption) -> At {
        At { option, ..self }
    }
}

/// A square 2D texture's descriptor.
fn descriptor(pf: PixelFormat, side: u64, levels: u64, storage: StorageMode) -> TextureDescriptor {
    let mut d = TextureDescriptor::texture_2d(pf, side, side, false);
    d.mipmap_level_count = levels;
    d.usage = TextureUsage::SHADER_READ;
    d.resource_options = ResourceOptions::new(
        storage,
        CpuCacheMode::DefaultCache,
        HazardTrackingMode::Default,
    );
    d
}

/// The whole of a square level `side` texels on a side.
fn whole(side: u64) -> Region {
    Region::new_2d(0, 0, side, side)
}

/// Probe 7's row-linear copies (`pv_rl_*`, `pv_levels_rgba4_16`, `pv_t2t_rgba4_16`), on every
/// 4-bit format, `Private` and `Shared`.
#[test]
fn pvrtc_textures_keep_their_blocks_through_row_linear_copies() {
    on_device(|lane| {
        for pf in FOUR_BIT {
            for storage in [StorageMode::Private, StorageMode::Shared] {
                let what = format!("{pf:?} {storage:?}");
                // A whole level in and out, 16 and 32 texels a side (4 x 4 and 8 x 8 blocks).
                for side in [16u64, 32] {
                    let n = (side / 4 * side / 4 * 8) as usize;
                    let bytes = blocks(n, 0xb0);
                    let texture = lane.texture(pf, side, 1, storage);
                    let out = lane
                        .run(|blit| {
                            let row = side / 4 * 8;
                            lane.put(blit, &texture, &bytes, At::rows(whole(side), row))?;
                            Ok(vec![lane.get(
                                blit,
                                &texture,
                                n,
                                At::rows(whole(side), row),
                            )?])
                        })
                        .unwrap();
                    assert_eq!(out[0], bytes, "{what} {side}");
                }
                // 2 x 2 blocks at texel (8, 4) over a fill of 0x11, then the whole level and the
                // region out.
                let texture = lane.texture(pf, 32, 1, storage);
                let at = Region::new_2d(8, 4, 8, 8);
                let out = lane
                    .run(|blit| {
                        lane.put(blit, &texture, &[0x11; 512], At::rows(whole(32), 64))?;
                        lane.put(blit, &texture, &blocks(32, 0xd0), At::rows(at, 16))?;
                        Ok(vec![
                            lane.get(blit, &texture, 512, At::rows(whole(32), 64))?,
                            lane.get(blit, &texture, 32, At::rows(at, 16))?,
                        ])
                    })
                    .unwrap();
                let mut expected = vec![0x11; 512];
                let region = blocks(32, 0xd0);
                expected[64 + 16..64 + 32].copy_from_slice(&region[..16]);
                expected[128 + 16..128 + 32].copy_from_slice(&region[16..]);
                assert_eq!(out[0], expected, "{what}: the region in");
                assert_eq!(out[1], region, "{what}: the region out");
                // From offset 8 of the buffer; rows 48 bytes apart, the 16 bytes between them
                // left alone on the way out.
                let texture = lane.texture(pf, 16, 1, storage);
                let source = blocks(256, 0xb0);
                let padded = blocks(192, 0xb0);
                let out = lane
                    .run(|blit| {
                        lane.put(blit, &texture, &source, At::rows(whole(16), 32).from(8))?;
                        let shifted = lane.get(blit, &texture, 128, At::rows(whole(16), 32))?;
                        lane.put(blit, &texture, &padded, At::rows(whole(16), 48))?;
                        let padded_out = lane.get(blit, &texture, 192, At::rows(whole(16), 48))?;
                        Ok(vec![shifted, padded_out])
                    })
                    .unwrap();
                assert_eq!(out[0], source[8..136], "{what}: from offset 8");
                let mut expected = padded.clone();
                for row in 0..4 {
                    expected[row * 48 + 32..row * 48 + 48].fill(0xee);
                }
                assert_eq!(out[1], expected, "{what}: padded rows");
                // Each level of a 16 x 16 chain: 4 x 4, 2 x 2, then one block a level.
                let texture = lane.texture(pf, 16, 5, storage);
                let levels: Vec<(u64, usize, Vec<u8>)> = (0..5u64)
                    .map(|level| {
                        let side: u64 = 16 >> level;
                        let row = side.div_ceil(4) as usize * 8;
                        let n = row * side.div_ceil(4) as usize;
                        (side, row, blocks(n, 0x10 * (level as u8 + 1)))
                    })
                    .collect();
                let out = lane
                    .run(|blit| {
                        for (level, (side, row, bytes)) in levels.iter().enumerate() {
                            let at = At::rows(whole(*side), *row as u64).level(level as u64);
                            lane.put(blit, &texture, bytes, at)?;
                        }
                        levels
                            .iter()
                            .enumerate()
                            .map(|(level, (side, row, bytes))| {
                                let at = At::rows(whole(*side), *row as u64).level(level as u64);
                                lane.get(blit, &texture, bytes.len(), at)
                            })
                            .collect()
                    })
                    .unwrap();
                for (level, (_, _, bytes)) in levels.iter().enumerate() {
                    assert_eq!(&out[level], bytes, "{what}: level {level}");
                }
                // A texture's blocks into another's.
                let (a, b) = (
                    lane.texture(pf, 16, 1, storage),
                    lane.texture(pf, 16, 1, StorageMode::Private),
                );
                let bytes = blocks(128, 0xb0);
                let out = lane
                    .run(|blit| {
                        lane.put(blit, &a, &bytes, At::rows(whole(16), 32))?;
                        blit.copy_from_texture_to_texture(
                            &a,
                            0,
                            0,
                            Origin::new(0, 0, 0),
                            Size::new(16, 16, 1),
                            &b,
                            0,
                            0,
                            Origin::new(0, 0, 0),
                        )?;
                        Ok(vec![lane.get(blit, &b, 128, At::rows(whole(16), 32))?])
                    })
                    .unwrap();
                assert_eq!(out[0], bytes, "{what}: texture to texture");
            }
        }
    });
}

/// Probe 7's row-linear copies on the 2-bit formats, whose 8 x 4-texel blocks a device without
/// their format holds in a carrier image half as wide, `Private` and `Shared`: whole 16- and
/// 32-texel levels (2 x 4 and 4 x 8 blocks) in and out; 2 x 2 blocks at texel (8, 4) of a
/// 32-texel level over a fill of 0x11, then the whole level and the region out; each level of a
/// 16-texel chain (2 x 4 blocks, 1 x 2, then one block a level); and a texture's blocks into
/// another's.
#[test]
fn two_bit_pvrtc_textures_keep_their_blocks_through_row_linear_copies() {
    on_device(|lane| {
        for pf in TWO_BIT {
            for storage in [StorageMode::Private, StorageMode::Shared] {
                let what = format!("{pf:?} {storage:?}");
                for side in [16u64, 32] {
                    let row = side / 8 * 8;
                    let n = (row * side / 4) as usize;
                    let bytes = blocks(n, 0xb0);
                    let texture = lane.texture(pf, side, 1, storage);
                    let out = lane
                        .run(|blit| {
                            lane.put(blit, &texture, &bytes, At::rows(whole(side), row))?;
                            Ok(vec![lane.get(
                                blit,
                                &texture,
                                n,
                                At::rows(whole(side), row),
                            )?])
                        })
                        .unwrap();
                    assert_eq!(out[0], bytes, "{what} {side}");
                }
                // 32 x 32 texels: 4 x 8 blocks, 32 bytes a row; the region's blocks are (1, 1),
                // (2, 1), (1, 2) and (2, 2).
                let texture = lane.texture(pf, 32, 1, storage);
                let at = Region::new_2d(8, 4, 16, 8);
                let out = lane
                    .run(|blit| {
                        lane.put(blit, &texture, &[0x11; 256], At::rows(whole(32), 32))?;
                        lane.put(blit, &texture, &blocks(32, 0xd0), At::rows(at, 16))?;
                        Ok(vec![
                            lane.get(blit, &texture, 256, At::rows(whole(32), 32))?,
                            lane.get(blit, &texture, 32, At::rows(at, 16))?,
                        ])
                    })
                    .unwrap();
                let mut expected = vec![0x11; 256];
                let region = blocks(32, 0xd0);
                expected[32 + 8..32 + 24].copy_from_slice(&region[..16]);
                expected[64 + 8..64 + 24].copy_from_slice(&region[16..]);
                assert_eq!(out[0], expected, "{what}: the region in");
                assert_eq!(out[1], region, "{what}: the region out");
                // Each level of a 16 x 16 chain.
                let texture = lane.texture(pf, 16, 5, storage);
                let levels: Vec<(u64, u64, Vec<u8>)> = (0..5u64)
                    .map(|level| {
                        let side: u64 = 16 >> level;
                        let row = side.div_ceil(8) * 8;
                        let n = (row * side.div_ceil(4)) as usize;
                        (side, row, blocks(n, 0x10 * (level as u8 + 1)))
                    })
                    .collect();
                let out = lane
                    .run(|blit| {
                        for (level, (side, row, bytes)) in levels.iter().enumerate() {
                            let at = At::rows(whole(*side), *row).level(level as u64);
                            lane.put(blit, &texture, bytes, at)?;
                        }
                        levels
                            .iter()
                            .enumerate()
                            .map(|(level, (side, row, bytes))| {
                                let at = At::rows(whole(*side), *row).level(level as u64);
                                lane.get(blit, &texture, bytes.len(), at)
                            })
                            .collect()
                    })
                    .unwrap();
                for (level, (_, _, bytes)) in levels.iter().enumerate() {
                    assert_eq!(&out[level], bytes, "{what}: level {level}");
                }
                // A texture's blocks into another's.
                let (a, b) = (
                    lane.texture(pf, 16, 1, storage),
                    lane.texture(pf, 16, 1, StorageMode::Private),
                );
                let bytes = blocks(64, 0xb0);
                let out = lane
                    .run(|blit| {
                        lane.put(blit, &a, &bytes, At::rows(whole(16), 16))?;
                        blit.copy_from_texture_to_texture(
                            &a,
                            0,
                            0,
                            Origin::new(0, 0, 0),
                            Size::new(16, 16, 1),
                            &b,
                            0,
                            0,
                            Origin::new(0, 0, 0),
                        )?;
                        Ok(vec![lane.get(blit, &b, 64, At::rows(whole(16), 16))?])
                    })
                    .unwrap();
                assert_eq!(out[0], bytes, "{what}: texture to texture");
            }
        }
    });
}

/// Our probe's copies without the option at a nonzero `bytesPerRow` (an M4 under macOS
/// 27.0.1; the reims-vgpu consumer's run on an M3 Max under macOS 26.7): they move the blocks row
/// by row as the option's copies do, in both directions. Whole 16- and 32-texel levels of the 4-bit
/// formats and of the 2-bit ones, on the way in and out and against the option's copies; then, for
/// the 4-bit formats, rows 48 bytes apart (the 16 bytes past each row's blocks left alone on the
/// way out) and 2 x 2 blocks at texel (8, 4) of a 32-texel level.
#[test]
fn a_pvrtc_copy_without_the_option_moves_rows_at_a_nonzero_pitch() {
    on_device(|lane| {
        let none = BlitOption::None;
        for pf in FOUR_BIT.into_iter().chain(TWO_BIT) {
            let width = if TWO_BIT.contains(&pf) { 8 } else { 4 };
            for side in [16u64, 32] {
                let what = format!("{pf:?} {side}");
                let made = lane
                    .device
                    .new_texture(&descriptor(pf, side, 1, StorageMode::Private));
                let texture = made.unwrap_or_else(|error| panic!("{what}: {error:?}"));
                let row = side / width * 8;
                let n = (row * side / 4) as usize;
                let (bytes, again) = (blocks(n, 0xb0), blocks(n, 0xc0));
                let rows = At::rows(whole(side), row);
                let out = lane
                    .run(|blit| {
                        lane.put(blit, &texture, &bytes, rows.option(none))?;
                        let with = lane.get(blit, &texture, n, rows)?;
                        let without = lane.get(blit, &texture, n, rows.option(none))?;
                        lane.put(blit, &texture, &again, rows)?;
                        let after = lane.get(blit, &texture, n, rows.option(none))?;
                        Ok(vec![with, without, after])
                    })
                    .unwrap();
                assert_eq!(out[0], bytes, "{what}: in without the option");
                assert_eq!(out[1], bytes, "{what}: out without the option");
                assert_eq!(out[2], again, "{what}: in with the option, out without it");
            }
        }
        for pf in FOUR_BIT {
            let what = format!("{pf:?}");
            let texture = lane.texture(pf, 16, 1, StorageMode::Private);
            let padded = blocks(192, 0xb0);
            let out = lane
                .run(|blit| {
                    let at = At::rows(whole(16), 48).option(none);
                    lane.put(blit, &texture, &padded, at)?;
                    Ok(vec![
                        lane.get(blit, &texture, 192, at)?,
                        lane.get(blit, &texture, 128, At::rows(whole(16), 32))?,
                    ])
                })
                .unwrap();
            let mut expected = padded.clone();
            for row in 0..4 {
                expected[row * 48 + 32..row * 48 + 48].fill(0xee);
            }
            assert_eq!(out[0], expected, "{what}: padded rows");
            let packed: Vec<u8> = (0..4)
                .flat_map(|row| padded[row * 48..row * 48 + 32].to_vec())
                .collect();
            assert_eq!(out[1], packed, "{what}: padded rows, out row by row");
            let texture = lane.texture(pf, 32, 1, StorageMode::Private);
            let at = Region::new_2d(8, 4, 8, 8);
            let out = lane
                .run(|blit| {
                    lane.put(blit, &texture, &[0x11; 512], At::rows(whole(32), 64))?;
                    lane.put(
                        blit,
                        &texture,
                        &blocks(32, 0xd0),
                        At::rows(at, 16).option(none),
                    )?;
                    Ok(vec![
                        lane.get(blit, &texture, 512, At::rows(whole(32), 64))?,
                        lane.get(blit, &texture, 32, At::rows(at, 16).option(none))?,
                    ])
                })
                .unwrap();
            let mut expected = vec![0x11; 512];
            let region = blocks(32, 0xd0);
            expected[64 + 16..64 + 32].copy_from_slice(&region[..16]);
            expected[128 + 16..128 + 32].copy_from_slice(&region[16..]);
            assert_eq!(out[0], expected, "{what}: the region in");
            assert_eq!(out[1], region, "{what}: the region out");
        }
    });
}

/// Probe 7's `pv_shared_texture`: a shareable PVRTC texture takes blocks, and a texture opened
/// from its handle gives them back.
#[test]
fn a_shared_pvrtc_texture_opens_over_its_blocks() {
    on_device(|lane| {
        for pf in FOUR_BIT {
            let texture = lane
                .device
                .new_shared_texture(&descriptor(pf, 16, 1, StorageMode::Private))
                .unwrap();
            let bytes = blocks(128, 0xb0);
            lane.run(|blit| {
                lane.put(blit, &texture, &bytes, At::rows(whole(16), 32))?;
                Ok(Vec::new())
            })
            .unwrap();
            let handle = texture.new_shared_texture_handle().unwrap();
            let opened = lane
                .device
                .new_shared_texture_with_handle(&handle, Default::default())
                .unwrap();
            assert_eq!(opened.pixel_format(), pf);
            let out = lane
                .run(|blit| {
                    Ok(vec![lane.get(
                        blit,
                        &opened,
                        128,
                        At::rows(whole(16), 32),
                    )?])
                })
                .unwrap();
            assert_eq!(out[0], bytes, "{pf:?}");
        }
    });
}

/// Probe 7's `pv_opt_*`: the option on another format is ignored, as plain Metal ignores it.
#[test]
fn the_row_linear_option_on_another_format_is_ignored() {
    on_device(|lane| {
        for (pf, side, row, n) in [
            (PixelFormat::RGBA8Unorm, 4u64, 16u64, 64usize),
            (PixelFormat::BC1RGBA, 16, 32, 128),
        ] {
            let bytes = blocks(n, 0xb0);
            let texture = lane.texture(pf, side, 1, StorageMode::Private);
            let at = At::rows(whole(side), row).image(n as u64);
            let out = lane
                .run(|blit| {
                    lane.put(blit, &texture, &bytes, at)?;
                    Ok(vec![lane.get(
                        blit,
                        &texture,
                        n,
                        at.option(BlitOption::None),
                    )?])
                })
                .unwrap();
            assert_eq!(out[0], bytes, "{pf:?}");
        }
    });
}

/// The position at which plain Metal lays block (`x`, `y`) of a region `blocks` wide and high out
/// at a `bytesPerRow` of 0: the region's own Morton order, the x bit lowest, each axis as many
/// bits as its block count needs, the larger axis's extra bits on top.
fn morton(x: u64, y: u64, blocks: [u64; 2]) -> u64 {
    let bits = |n: u64| (0..64).find(|&b| 1u64 << b >= n).unwrap();
    let (nx, ny) = (bits(blocks[0]), bits(blocks[1]));
    let mut out = Vec::new();
    for i in 0..nx.max(ny) {
        if i < nx {
            out.push((x >> i) & 1);
        }
        if i < ny {
            out.push((y >> i) & 1);
        }
    }
    out.iter().enumerate().map(|(b, &v)| v << b).sum()
}

/// Our probe's copies at a `bytesPerRow` of 0 (an M4 under macOS 27.0.1; the reims-vgpu
/// consumer's report of a whole 16 x 16 4-bit level on an M3 Max under macOS 26.7): with the option
/// and without it, a region's blocks lie in its own Morton order from the buffer's offset. Into the
/// texture, over a level of 0x11 bytes, from a buffer whose blocks carry their index, the blocks
/// outside the region kept; out of it, from a level whose blocks carry theirs, into a buffer of
/// 0xee bytes, the positions no block takes kept. Whole levels and the probe's regions (off the
/// order's quadrants, and 3 blocks wide), 16 and 32 texels square, the 4-bit formats and the 2-bit
/// ones (8 x 4-texel blocks).
#[test]
fn a_pvrtc_copy_at_a_pitch_of_0_lays_blocks_in_metals_block_order() {
    on_device(|lane| {
        let formats = [PixelFormat::PVRTCRGBA4BPP, PixelFormat::PVRTCRGB4BPPSRGB];
        for pf in formats.into_iter().chain(TWO_BIT) {
            let two_bit = TWO_BIT.contains(&pf);
            let width = if two_bit { 8 } else { 4 };
            for side in [16u64, 32] {
                let what = format!("{pf:?} {side}");
                let made = lane
                    .device
                    .new_texture(&descriptor(pf, side, 1, StorageMode::Private));
                let texture = made.unwrap_or_else(|error| panic!("{what}: {error:?}"));
                // The probe's regions, in blocks: the first block and the blocks wide and high.
                let regions: &[([u64; 2], [u64; 2])] = match (two_bit, side) {
                    (false, 16) => &[
                        ([0, 0], [4, 4]),
                        ([1, 1], [2, 2]),
                        ([2, 0], [2, 2]),
                        ([0, 1], [4, 2]),
                    ],
                    (false, _) => &[
                        ([0, 0], [8, 8]),
                        ([1, 1], [2, 2]),
                        ([4, 4], [4, 4]),
                        ([2, 1], [3, 2]),
                        ([1, 2], [4, 4]),
                    ],
                    (true, 16) => &[([0, 0], [2, 4]), ([1, 1], [1, 2]), ([0, 2], [2, 2])],
                    (true, _) => &[([0, 0], [4, 8]), ([1, 1], [2, 2]), ([2, 4], [2, 4])],
                };
                for &([x0, y0], blocks_of) in regions {
                    let at =
                        Region::new_2d(x0 * width, y0 * 4, blocks_of[0] * width, blocks_of[1] * 4);
                    let what = format!("{what} {blocks_of:?} at {x0},{y0}");
                    lays_blocks_in_metals_order(
                        lane,
                        &texture,
                        [side, width],
                        at,
                        [x0, y0],
                        blocks_of,
                        &what,
                    );
                }
            }
        }
    });
}

/// A copy at a `bytesPerRow` of 0 of the texels `at` of `texture` (`side` texels square, blocks
/// `width` texels wide), into the texture and out of it, with the option and without it, moves the
/// blocks `blocks_of` wide and high from block `[x0, y0]` in their own Morton order, as Metal's.
fn lays_blocks_in_metals_order(
    lane: &Lane,
    texture: &Texture,
    [side, width]: [u64; 2],
    at: Region,
    [x0, y0]: [u64; 2],
    blocks_of: [u64; 2],
    what: &str,
) {
    let (across, down) = (side / width, side / 4);
    let row = across * 8;
    let n = (row * down) as usize;
    // Each block of the region: its level block's row-linear byte and its block-order byte.
    let placed: Vec<(usize, usize)> = (0..blocks_of[1])
        .flat_map(|y| (0..blocks_of[0]).map(move |x| (x, y)))
        .map(|(x, y)| {
            let level = ((y0 + y) * across + x0 + x) as usize * 8;
            (level, morton(x, y, blocks_of) as usize * 8)
        })
        .collect();
    for option in [BlitOption::None, ROWS] {
        let what = format!("{what} {option:?}");
        let zero = At::rows(at, 0).option(option);
        let source = blocks(2 * n, 0xb0);
        let level = blocks(n, 0xc0);
        let out = lane
            .run(|blit| {
                let rows = At::rows(whole(side), row);
                lane.put(blit, texture, &vec![0x11; n], rows)?;
                lane.put(blit, texture, &source, zero)?;
                let into = lane.get(blit, texture, n, rows)?;
                lane.put(blit, texture, &level, rows)?;
                Ok(vec![into, lane.get(blit, texture, 2 * n, zero)?])
            })
            .unwrap();
        let mut into = vec![0x11; n];
        let mut back = vec![0xee; 2 * n];
        for &(l, z) in &placed {
            into[l..l + 8].copy_from_slice(&source[z..z + 8]);
            back[z..z + 8].copy_from_slice(&level[l..l + 8]);
        }
        assert_eq!(out[0], into, "{what}: into the texture");
        assert_eq!(out[1], back, "{what}: out of the texture");
    }
}

/// One of our probe's copies of a region off its blocks: the format, the texture's side
/// in texels, the region's texel origin and size, and the blocks Metal moved, the first and the
/// count across and down.
struct OffBlocks {
    format: PixelFormat,
    side: u64,
    texels: [u64; 4],
    first: [u64; 2],
    blocks: [u64; 2],
}

impl OffBlocks {
    const fn new(
        format: PixelFormat,
        side: u64,
        texels: [u64; 4],
        first: [u64; 2],
        blocks: [u64; 2],
    ) -> Self {
        OffBlocks {
            format,
            side,
            texels,
            first,
            blocks,
        }
    }
}

/// Our probe's copies at a `bytesPerRow` of 0 of regions off their blocks (an M4 under
/// macOS 27.0.1; its validation stops them): the region is the blocks it is rounded to, its
/// origin down to its block and its size up to whole blocks, each on its own, in their own Morton
/// order, with the option and without it, into the texture and out of it. The probe's regions:
/// an origin or a size off the block, both, one ending at the level's edge, one texel, and a
/// 3 x 3-block region whose order leaves positions no block takes; 4-bit 16 and 32 texels square,
/// and 2-bit (8 x 4-texel blocks).
#[test]
fn a_pvrtc_copy_at_a_pitch_of_0_of_a_region_off_its_blocks_copies_the_blocks_it_is_rounded_to() {
    const FOUR: PixelFormat = PixelFormat::PVRTCRGBA4BPP;
    const TWO: PixelFormat = PixelFormat::PVRTCRGBA2BPP;
    let cases = [
        OffBlocks::new(FOUR, 16, [2, 0, 8, 4], [0, 0], [2, 1]),
        OffBlocks::new(FOUR, 16, [0, 0, 6, 8], [0, 0], [2, 2]),
        OffBlocks::new(FOUR, 16, [4, 4, 6, 6], [1, 1], [2, 2]),
        OffBlocks::new(FOUR, 16, [1, 1, 4, 4], [0, 0], [1, 1]),
        OffBlocks::new(FOUR, 16, [2, 2, 14, 14], [0, 0], [4, 4]),
        OffBlocks::new(FOUR, 16, [0, 0, 1, 1], [0, 0], [1, 1]),
        OffBlocks::new(FOUR, 32, [4, 4, 10, 12], [1, 1], [3, 3]),
        OffBlocks::new(FOUR, 32, [6, 0, 8, 8], [1, 0], [2, 2]),
        OffBlocks::new(TWO, 16, [4, 0, 8, 8], [0, 0], [1, 2]),
        OffBlocks::new(TWO, 16, [0, 2, 16, 8], [0, 0], [2, 2]),
        OffBlocks::new(TWO, 32, [0, 0, 12, 8], [0, 0], [2, 2]),
    ];
    on_device(|lane| {
        for case in &cases {
            let (pf, side, [x, y, w, h]) = (case.format, case.side, case.texels);
            let width = if TWO_BIT.contains(&pf) { 8 } else { 4 };
            let texture = lane
                .device
                .new_texture(&descriptor(pf, side, 1, StorageMode::Private))
                .unwrap();
            let what = format!("{pf:?} {side} texels {x},{y} {w}x{h}");
            lays_blocks_in_metals_order(
                lane,
                &texture,
                [side, width],
                Region::new_2d(x, y, w, h),
                case.first,
                case.blocks,
                &what,
            );
        }
    });
}

/// The position plain Metal's CPU copies give block (`x`, `y`) of a whole level `blocks` across
/// and down: Z order with the y bit lowest, the larger axis's extra bits on top ([`morton`] with
/// the axes swapped).
fn cpu_order(x: u64, y: u64, blocks: [u64; 2]) -> u64 {
    morton(y, x, [blocks[1], blocks[0]])
}

/// Our probe's CPU copies of whole levels (an M4 under macOS 27.0.1; valid usage at a
/// `bytesPerRow` of 0, plain and validated alike; at the level's row too, which validation stops):
/// `replaceRegion` takes a whole level's blocks in Metal's CPU order, Z order with the y bit
/// lowest, and `getBytes` gives them so. Into the level, from bytes whose blocks carry their
/// position, read back row by row; out of it, from a level filled row by row with blocks carrying
/// their index. Every level of a 16 x 16 chain (4 x 4 blocks down to one block a level) and a 32 x
/// 32 level, 4-bit formats and the 2-bit ones (8 x 4-texel blocks).
#[test]
fn a_pvrtc_texture_s_cpu_copies_of_a_whole_level_take_its_blocks_in_metals_cpu_order() {
    on_device(|lane| {
        let formats = [PixelFormat::PVRTCRGBA4BPP, PixelFormat::PVRTCRGB4BPPSRGB];
        for pf in formats.into_iter().chain(TWO_BIT) {
            let width = if TWO_BIT.contains(&pf) { 8 } else { 4 };
            for (side, levels) in [(16, 5), (32, 1)] {
                let texture = lane.texture(pf, side, levels, StorageMode::Shared);
                for level in 0..levels {
                    let texels = (side >> level).max(1);
                    let blocks_of = [texels.div_ceil(width), texels.div_ceil(4)];
                    let row = blocks_of[0] * 8;
                    let n = (row * blocks_of[1]) as usize;
                    let rows = At::rows(whole(texels), row).level(level);
                    // Each block: its level block's row-major byte and its CPU-order byte.
                    let placed: Vec<(usize, usize)> = (0..blocks_of[1])
                        .flat_map(|y| (0..blocks_of[0]).map(move |x| (x, y)))
                        .map(|(x, y)| {
                            let at = (y * blocks_of[0] + x) as usize * 8;
                            (at, cpu_order(x, y, blocks_of) as usize * 8)
                        })
                        .collect();
                    for pitch in [0, row] {
                        let what = format!("{pf:?} {side} level {level} at a pitch of {pitch}");
                        let host = blocks(n, 0xb0);
                        texture
                            .replace_region(whole(texels), level, 0, &host, pitch, 0)
                            .unwrap_or_else(|error| panic!("{what}: {error:?}"));
                        let into = lane
                            .run(|blit| Ok(vec![lane.get(blit, &texture, n, rows)?]))
                            .unwrap();
                        let filled = blocks(n, 0xc0);
                        lane.run(|blit| {
                            lane.put(blit, &texture, &filled, rows)?;
                            Ok(Vec::new())
                        })
                        .unwrap();
                        let mut out = vec![0xee; n];
                        texture
                            .get_bytes(&mut out, pitch, 0, whole(texels), level, 0)
                            .unwrap_or_else(|error| panic!("{what}: {error:?}"));
                        let mut want_into = vec![0; n];
                        let mut want_out = vec![0; n];
                        for &(l, z) in &placed {
                            want_into[l..l + 8].copy_from_slice(&host[z..z + 8]);
                            want_out[z..z + 8].copy_from_slice(&filled[l..l + 8]);
                        }
                        assert_eq!(into[0], want_into, "{what}: replace_region");
                        assert_eq!(out, want_out, "{what}: get_bytes");
                    }
                }
            }
        }
    });
}

/// Whether `result` is the runtime's block-order gap.
fn gap<T: std::fmt::Debug>(result: Result<T, Error>, what: &str) {
    assert!(
        matches!(
            result,
            Err(Error::NotImplemented(RuntimeGap::PvrtcBlockOrder))
        ),
        "{what}: {result:?}"
    );
}

/// The runtime's block-order gap is refused `NotImplemented(PvrtcBlockOrder)`, recording
/// nothing: `replaceRegion` and `getBytes` of a region other than a whole level, which plain
/// Metal does not place by the region at all (the module docs), and of a whole level at a nonzero
/// `bytesPerRow` below the level's row, where the driver asserts and the call moves nothing. A
/// whole level at a padded pitch is taken, and
/// [`a_padded_whole_level_moves_the_bytes_a_tight_one_moves`] holds it to the tight answer.
#[test]
fn pvrtc_block_order_is_refused_typed() {
    on_device(|lane| {
        for pf in FOUR_BIT.into_iter().chain(TWO_BIT) {
            let texture = lane.texture(pf, 32, 1, StorageMode::Shared);
            let row = if TWO_BIT.contains(&pf) { 32 } else { 64 };
            let bytes = blocks(2048, 0xb0);
            let mut out = vec![0; 2048];
            for (region, pitch) in [
                (Region::new_2d(8, 4, 16, 8), 0),
                (Region::new_2d(0, 0, 16, 32), 0),
                // A whole level at a nonzero pitch below its row: the driver asserts there and
                // the call moves nothing, so the runtime refuses it rather than emulate that.
                (Region::new_2d(0, 0, 32, 32), row / 2),
            ] {
                let what = format!("{pf:?} {region:?} at a pitch of {pitch}");
                gap(
                    texture.replace_region(region, 0, 0, &bytes, pitch, 0),
                    &what,
                );
                gap(texture.get_bytes(&mut out, pitch, 0, region, 0, 0), &what);
            }
        }
    });
}

/// A whole level at a `bytesPerRow` at or above the level's row moves exactly the bytes a pitch
/// of 0 moves: plain Metal ignores the pitch there and moves the level's bytes contiguously, so a
/// padded pitch reads and writes its padding bytes as block data.
///
/// Measured on this M3 Max (macOS 27.0.1, plain Metal with no validation layer, a 32x32 texture
/// of both bit depths, two runs byte-identical): at the level's row, row + 8, twice the row,
/// twice + 8 and seven times it, every byte `getBytes` wrote and every byte `replaceRegion`
/// consumed was the byte a pitch of 0 moves. Below the level's row and nonzero the AGX driver
/// logs `bytes_per_row >= used_bytes_per_row` and the call moves nothing, which
/// [`pvrtc_block_order_is_refused_typed`] holds refused.
///
/// This is the whole of the check: a padded pitch must not merely be accepted, it must move the
/// tight answer's bytes. The runtime already did -- `host_copy` substitutes the level's own row
/// for the caller's value before planning a PVRTC copy -- so only the guard was wrong.
#[test]
fn a_padded_whole_level_moves_the_bytes_a_tight_one_moves() {
    on_device(|lane| {
        for pf in FOUR_BIT.into_iter().chain(TWO_BIT) {
            let row = if TWO_BIT.contains(&pf) { 32 } else { 64 };
            // Both bit depths have blocks 4 texels tall, so a 32x32 level is 8 rows of
            // blocks: 512 bytes of a 4-bit level, 256 of a 2-bit one.
            let level = row * 8;
            let whole = Region::new_2d(0, 0, 32, 32);
            let source = seeded(0x9e37_79b9, level as usize);
            // The tight answer: written and read back at a pitch of 0.
            let tight = lane.texture(pf, 32, 1, StorageMode::Shared);
            tight
                .replace_region(whole, 0, 0, &source, 0, 0)
                .unwrap_or_else(|e| panic!("{pf:?} tight write: {e}"));
            let mut want = vec![0xcd; level as usize];
            tight
                .get_bytes(&mut want, 0, 0, whole, 0, 0)
                .unwrap_or_else(|e| panic!("{pf:?} tight read: {e}"));
            for pitch in [row, row + 8, 2 * row, 2 * row + 8, 7 * row] {
                let what = format!("{pf:?} whole level at a pitch of {pitch} (row {row})");
                let texture = lane.texture(pf, 32, 1, StorageMode::Shared);
                // The padded write consumes the level's bytes contiguously, padding and all, so
                // the same source bytes must leave the level holding the tight answer.
                texture
                    .replace_region(whole, 0, 0, &source, pitch, 0)
                    .unwrap_or_else(|e| panic!("{what}: write {e}"));
                let mut got = vec![0xcd; level as usize];
                texture
                    .get_bytes(&mut got, pitch, 0, whole, 0, 0)
                    .unwrap_or_else(|e| panic!("{what}: read {e}"));
                assert_eq!(got, want, "{what}: the bytes a tight copy moves");
            }
        }
    });
}

/// Whether `device` holds a texture of `pf` in its own PVRTC format, decoding it its own way,
/// rather than in a carrier with a decoded twin.
fn holds_pvrtc(device: &Device, pf: PixelFormat) -> bool {
    let usage = [FormatUsage::SampledImage].into();
    vk_format(pf, &usage, device.compile_device().profile())
        .is_ok_and(|m| m.emulation == FormatEmulation::Direct)
}

/// `n` bytes of a seeded xorshift: random blocks of every mode.
fn seeded(seed: u64, n: usize) -> Vec<u8> {
    let mut x = seed | 1;
    let mut out = Vec::with_capacity(n + 8);
    while out.len() < n {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        out.extend_from_slice(&x.wrapping_mul(0x2545_f491_4f6c_dd1d).to_le_bytes());
    }
    out.truncate(n);
    out
}

/// The form of a PVRTC format in the other colour space, its sRGB or linear twin.
fn other_space(pf: PixelFormat) -> PixelFormat {
    use PixelFormat as P;
    match pf {
        P::PVRTCRGB2BPP => P::PVRTCRGB2BPPSRGB,
        P::PVRTCRGB2BPPSRGB => P::PVRTCRGB2BPP,
        P::PVRTCRGB4BPP => P::PVRTCRGB4BPPSRGB,
        P::PVRTCRGB4BPPSRGB => P::PVRTCRGB4BPP,
        P::PVRTCRGBA2BPP => P::PVRTCRGBA2BPPSRGB,
        P::PVRTCRGBA2BPPSRGB => P::PVRTCRGBA2BPP,
        P::PVRTCRGBA4BPP => P::PVRTCRGBA4BPPSRGB,
        P::PVRTCRGBA4BPPSRGB => P::PVRTCRGBA4BPP,
        other => panic!("{other:?} is no PVRTC format"),
    }
}

/// Every PVRTC format's texels read exactly as Metal's on a device without the format (the module
/// docs), through the texture and through its view as the other colour space: a new texture's
/// zero blocks, then a whole level's blocks a CPU copy wrote (in Metal's CPU order), a row-linear
/// blit from a buffer, and a copy from another texture, each read by a command buffer after the
/// write's. A device with the format reads through its own decode (not compared).
#[test]
fn pvrtc_texels_read_as_metal_s() {
    super::degrade_notices::on_device_with(at_plan, |device, messages| {
        let name = device.name().to_owned();
        let state = reader(device);
        let exact = metal_sampler(device);
        let queue = device.new_command_queue().unwrap();
        let lane = Lane {
            device: device.clone(),
            queue,
            messages: messages.clone(),
        };
        let mut wrong = Vec::new();
        let mut undecoded = Vec::new();
        let mut twinned = false;
        // The GPU decode passes and CPU decodes so far.
        let decodes = |device: &Device| {
            let counts = device.work_counts();
            (counts.block_decode_passes, counts.cpu_block_decodes)
        };
        for (f, pf) in FOUR_BIT.into_iter().chain(TWO_BIT).enumerate() {
            let native = holds_pvrtc(device, pf);
            twinned |= !native;
            let decode = PvrtcDecode::of(pf).unwrap();
            let [bw, bh] = decode.block_extent().map(u64::from);
            let blocks = [16 / bw, 16 / bh];
            let bytes = (blocks[0] * blocks[1] * 8) as usize;
            let mut check = |what: &str, texture: &Texture, rows: &[u8]| {
                if native {
                    return;
                }
                let codes = decode.decode(rows, [16, 16, 1]);
                let view = texture.new_texture_view(other_space(pf)).unwrap();
                for (read, through) in [(texture, pf), (&view, other_space(pf))] {
                    let as_read = PvrtcDecode::of(through).unwrap();
                    let got = read_whole(device, &state, read);
                    for (t, got) in got.iter().enumerate() {
                        for (c, &got) in got.iter().enumerate() {
                            let want = as_read.metal_value(c, codes[4 * t + c]);
                            let holds = match as_read.srgb && c < 3 {
                                true => super::srgb_decode_holds(got, want, exact),
                                false => converted_holds(got, want, exact),
                            };
                            if !holds && wrong.len() < 24 {
                                wrong.push(format!(
                                    "{pf:?} {what} read as {through:?}, texel {t} channel {c}: \
                                     {got} against Metal's {want}"
                                ));
                            }
                        }
                    }
                }
            };
            let texture = lane.texture(pf, 16, 1, StorageMode::Shared);
            check("fresh", &texture, &vec![0; bytes]);
            // A CPU copy of the whole level, its blocks in Metal's CPU order.
            let rows = seeded(0x9e37_0000 + 0x10 * f as u64, bytes);
            let mut cpu = vec![0; bytes];
            for (i, block) in rows.as_chunks::<8>().0.iter().enumerate() {
                let (x, y) = (i as u64 % blocks[0], i as u64 / blocks[0]);
                let at = cpu_order(x, y, blocks) as usize * 8;
                cpu[at..at + 8].copy_from_slice(block);
            }
            let before = decodes(device);
            texture.replace_region(whole(16), 0, 0, &cpu, 0, 0).unwrap();
            check("CPU copy", &texture, &rows);
            if !native && decodes(device) == before {
                undecoded.push(format!("{pf:?} CPU copy: neither a pass nor a CPU decode"));
            }
            // A row-linear blit from a buffer, then a copy into another texture.
            let blitted = seeded(0x9e37_0001 + 0x10 * f as u64, bytes);
            let before = decodes(device);
            let row = blocks[0] * 8;
            lane.run(|blit| {
                lane.put(blit, &texture, &blitted, At::rows(whole(16), row))?;
                Ok(Vec::new())
            })
            .unwrap();
            check("blit", &texture, &blitted);
            let copy = lane.texture(pf, 16, 1, StorageMode::Private);
            lane.run(|blit| {
                let at = Origin::new(0, 0, 0);
                let size = Size::new(16, 16, 1);
                blit.copy_from_texture_to_texture(&texture, 0, 0, at, size, &copy, 0, 0, at)?;
                Ok(Vec::new())
            })
            .unwrap();
            check("texture copy", &copy, &blitted);
            // Each blit's decode is a GPU pass.
            let after = decodes(device);
            if !native && after.0 < before.0 + 2 {
                undecoded.push(format!(
                    "{pf:?} blits: GPU decode passes {before:?} -> {after:?}"
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
            (usize::from(twinned), 0),
            "on {name}"
        );
        assert_eq!(
            announced(Degradation::DeviceTexelConversion),
            usize::from(!exact),
            "the device's conversion of the codes, on {name}"
        );
    });
}

/// A 4:2:2 texture in a heap over the caller's host pages, whose texels a carrier holds for copies
/// alone (it keeps no decoded twin, as a PVRTC texture's blocks there keep none), made resident
/// for a kernel that reads no texture handles changes nothing and announces nothing. Made resident
/// for a kernel that reads texture handles from memory, it changes nothing the kernel reads
/// through its own handle (its element is the neutral texture), and the dispatch announces
/// [`Degradation::UnreadableHandleTexture`] once over two command buffers. The kernel samples an
/// unsigned texture and the 4:2:2 texture's class is a float one (`RGBX8Unorm`'s), so the element
/// is the neutral by class as well: [`Degradation::CrossClassTextureNeutral`], also once. Nothing
/// where the device imports no host memory.
#[test]
fn a_texture_held_for_copies_a_kernel_s_handles_reach_is_announced_once() {
    on_device(|lane| {
        let pages = Mapping::anonymous(HEAP);
        // SAFETY: the mapping holds `HEAP` writable bytes and outlives the heap, dropped after it.
        let made =
            unsafe { no_copy_heap(&lane.device, pages.pointer(), HeapType::Placement, None) };
        let Some(heap) = made else {
            return;
        };
        let mut held = descriptor(PixelFormat::GBGR422, 16, 1, StorageMode::Shared);
        held.resource_options = shared();
        let pvrtc = heap.new_texture_at_offset(&held, 0).unwrap();
        // What the heap and its texture announce is theirs.
        drop(lane.messages.take_notices());
        // Beside the dispatches' own kind, what their steps force: the case's pipeline state's
        // making, and the `Shared` buffers' system memory.
        let forced = |case| {
            let mut kinds = case_making_kinds(&lane.device, case);
            if shared_in_system_memory(&lane.device) {
                kinds.insert(Degradation::SharedInSystemMemory);
            }
            kinds
        };
        let buffers = public_case(BUFFERS).unwrap();
        let alone = lane.dispatch(&buffers, None);
        assert_eq!(lane.dispatch(&buffers, Some(&pvrtc)), alone, "{BUFFERS}");
        assert_made_once(&lane.messages, &[], &forced(&buffers));
        let handles = public_case(HANDLES).unwrap();
        let alone = lane.dispatch(&handles, None);
        for _ in 0..2 {
            assert_eq!(lane.dispatch(&handles, Some(&pvrtc)), alone, "{HANDLES}");
        }
        let mut kinds = forced(&handles);
        kinds.insert(Degradation::CrossClassTextureNeutral);
        assert_made_once(&lane.messages, &UNREADABLE, &kinds);
    });
}
