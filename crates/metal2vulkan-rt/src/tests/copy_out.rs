//! The regions of a copy from a texture to bytes (`crate::copy_out`), against Vulkan's addressing
//! ("Buffer and Image Addressing"), and on every device the copies a driver was measured to place
//! wrong as one region.

use crate::copy_out::{regions, row_power};
use crate::texture::CopyUnit;
use ash::vk;
use std::collections::BTreeSet;

/// The bytes a copy of `region` of texel blocks `unit`'s accesses, by Vulkan's addressing: each
/// texel's `[texelOffset, texelOffset + blockSize)`, a layer's like a slice of depth's. A block's
/// texels share their bytes, so its first texel stands for it.
fn accessed(unit: &CopyUnit, region: &vk::BufferImageCopy) -> BTreeSet<u64> {
    let [bw, bh] = unit.block;
    let bytes = u64::from(unit.bytes);
    let e = region.image_extent;
    let row = u64::from(region.buffer_row_length.max(e.width).div_ceil(bw)) * bytes;
    let slice = u64::from(region.buffer_image_height.max(e.height).div_ceil(bh)) * row;
    let slices = e.depth.max(region.image_subresource.layer_count);
    let mut set = BTreeSet::new();
    for z in 0..slices {
        for y in (0..e.height).step_by(bh as usize) {
            for x in (0..e.width).step_by(bw as usize) {
                let at = region.buffer_offset
                    + u64::from(x / bw) * bytes
                    + u64::from(y / bh) * row
                    + u64::from(z) * slice;
                set.extend(at..at + bytes);
            }
        }
    }
    set
}

fn unit(bytes: u32, block: u32) -> CopyUnit {
    CopyUnit {
        aspect: vk::ImageAspectFlags::COLOR,
        bytes,
        block: [block, block],
    }
}

fn region(offset: u64, length: u32, [x, width]: [u32; 2], height: u32) -> vk::BufferImageCopy {
    vk::BufferImageCopy {
        buffer_offset: offset,
        buffer_row_length: length,
        buffer_image_height: 0,
        image_subresource: vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        },
        image_offset: vk::Offset3D {
            x: x as i32,
            y: 0,
            z: 0,
        },
        image_extent: vk::Extent3D {
            width,
            height,
            depth: 1,
        },
    }
}

/// Every region of 1 to 8 blocks of 1, 2, 4, 8 and 16 bytes (single texels and 4x4 blocks, the
/// last block column whole or cut), rows 0 to 4 blocks past it or packed, at offsets of 0 to 3
/// blocks, one image or three of depth or of layers, their images packed or a block row apart:
/// a copy splits exactly where its row stride is not a multiple of M, the largest power
/// of two up to 16 dividing its row's bytes; its parts access together exactly its bytes and none
/// twice, keep its row length and image height, cover its texels in order, and each has a stride
/// that is a multiple of its own M.
#[test]
fn a_copy_out_splits_where_its_stride_is_no_multiple_of_its_rows_power_and_keeps_its_bytes() {
    let mut splits = 0;
    for (bytes, block) in [(1, 1), (2, 1), (4, 1), (8, 1), (16, 1), (8, 4), (16, 4)] {
        let u = unit(bytes, block);
        for blocks in 1..=8u32 {
            for cut in [0, 1] {
                // A cut last column (a level's edge) is narrower than its block.
                if cut == 1 && block == 1 {
                    continue;
                }
                let width = blocks * block - cut;
                for extra in [None, Some(0), Some(1), Some(2), Some(3), Some(4)] {
                    let length = extra.map_or(0, |extra| (blocks + extra) * block);
                    for (at, (depth, layers, tall)) in (0..4u64).flat_map(|at| {
                        [(1, 1, None), (3, 1, None), (3, 1, Some(1)), (1, 3, Some(1))]
                            .map(|images| (at, images))
                    }) {
                        let mut r = region(at * u64::from(bytes), length, [8, width], 2 * block);
                        r.image_extent.depth = depth;
                        r.image_subresource.layer_count = layers;
                        r.buffer_image_height = tall.map_or(0, |tall| (2 + tall) * block);
                        let parts = regions(&u, r);
                        let parts = parts.as_slice();
                        let row = u64::from(blocks) * u64::from(bytes);
                        let stride =
                            u64::from(length.max(width).div_ceil(block)) * u64::from(bytes);
                        let split = !stride.is_multiple_of(row_power(row));
                        let case = format!(
                            "{bytes} bytes, {blocks} blocks, width {width}, length {length}, at \
                             {at}, depth {depth}, layers {layers}, image height {}",
                            r.buffer_image_height
                        );
                        assert_eq!(parts.len(), if split { 2 } else { 1 }, "{case}");
                        let whole = accessed(&u, &r);
                        let mut union = BTreeSet::new();
                        for part in parts {
                            let mine = accessed(&u, part);
                            assert!(union.is_disjoint(&mine), "{case}: parts overlap");
                            union.extend(mine);
                            assert_eq!(part.buffer_image_height, r.buffer_image_height, "{case}");
                            assert_eq!(part.image_extent.depth, depth, "{case}");
                            assert_eq!(part.image_subresource.layer_count, layers, "{case}");
                            let part_blocks = part.image_extent.width.div_ceil(block);
                            let part_row = u64::from(part_blocks) * u64::from(bytes);
                            let part_stride = u64::from(
                                part.buffer_row_length
                                    .max(part.image_extent.width)
                                    .div_ceil(block),
                            ) * u64::from(bytes);
                            assert_eq!(part_stride, stride, "{case}: the stride is kept");
                            if split {
                                assert!(part_stride.is_multiple_of(row_power(part_row)), "{case}");
                            }
                        }
                        assert_eq!(union, whole, "{case}: the parts' bytes are the copy's");
                        if split {
                            splits += 1;
                            let [first, last] = [parts[0], parts[1]];
                            assert_eq!(first.image_offset.x, 8, "{case}");
                            assert_eq!(
                                last.image_offset.x as u32,
                                8 + first.image_extent.width,
                                "{case}"
                            );
                            assert_eq!(
                                first.image_extent.width + last.image_extent.width,
                                width,
                                "{case}"
                            );
                            assert_eq!(first.buffer_row_length, length, "{case}");
                        }
                    }
                }
            }
        }
    }
    assert!(splits > 0);
}

/// A depth or stencil aspect's copy is one region whatever its stride: its offsets are multiples
/// of 4, which a last part's need not be.
#[test]
fn a_depth_or_stencil_copy_out_is_one_region() {
    for (name, aspect, bytes) in [
        ("depth", vk::ImageAspectFlags::DEPTH, 2),
        ("depth", vk::ImageAspectFlags::DEPTH, 4),
        ("stencil", vk::ImageAspectFlags::STENCIL, 1),
    ] {
        let u = CopyUnit {
            aspect,
            bytes,
            block: [1, 1],
        };
        let mut r = region(0, 3, [0, 2], 2);
        r.image_subresource.aspect_mask = aspect;
        assert_eq!(
            regions(&u, r).as_slice().len(),
            1,
            "{name} of {bytes} bytes"
        );
    }
}

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::copy_out::{row_power, SPLIT};
    use crate::tests::command::on_device::{each_device, finished};
    use crate::{
        BufferLayout, Device, IndexClass, MemoryClass, PixelFormat, Region, StorageMode,
        TextureDescriptor, TexturePlace, TextureType, TextureUsage,
    };
    use std::sync::atomic::Ordering;

    /// A 2-slice texture of `bx` x `bx` blocks of `dim` texels and `block` bytes, copied out of
    /// each slice to a buffer of 0xcd at an offset `rem` past a multiple of 32, rows `extra`
    /// blocks past their blocks; the bytes compared with the texture's, the padding kept. Returns
    /// whether the copy is of the class a single region was measured wrong for.
    fn copy_out(
        index: u32,
        name: &str,
        device: &Device,
        (pf, block, dim): (PixelFormat, u64, u32),
        bx: u64,
        extra: u64,
        rem: u64,
    ) -> bool {
        let w = bx as u32 * dim;
        let mut d = TextureDescriptor::new(TextureType::Type2DArray, pf, w, w);
        d.array_length = 2;
        d.usage = TextureUsage::SHADER_READ;
        d.storage_mode = StorageMode::Shared;
        let texture = device.new_texture(&d, Some(index)).expect("a texture");
        let row = (bx + extra) * block;
        let tight = |s: u32| -> Vec<u8> {
            let mut bytes = Vec::new();
            for y in 0..bx {
                for x in 0..bx {
                    for k in 0..block {
                        bytes.push(match k {
                            0 => s as u8,
                            1 => y as u8,
                            2 => x as u8,
                            _ => 0xa0 | k as u8,
                        });
                    }
                }
            }
            bytes
        };
        let mut places = Vec::new();
        let mut at = 0u64;
        for s in 0..2u32 {
            at = (at + 5 * block).div_ceil(32) * 32 + rem;
            places.push((s, at));
            at += row * bx;
        }
        let mut want = vec![0xcdu8; at as usize];
        for &(s, offset) in &places {
            let bytes = tight(s);
            for y in 0..bx {
                let from = (y * bx * block) as usize;
                let to = (offset + y * row) as usize;
                want[to..to + (bx * block) as usize]
                    .copy_from_slice(&bytes[from..from + (bx * block) as usize]);
            }
        }
        let blk = device
            .take_block(MemoryClass::Shared, at.div_ceil(4096) * 4096)
            .expect("a Shared block");
        let buffer = device
            .new_buffer(blk.id, 0, at.div_ceil(4) * 4, None)
            .expect("a buffer");
        let contents = buffer.contents().expect("contents");
        // SAFETY: the buffer is mapped, `at` bytes long at least, and no work uses it yet.
        unsafe { std::ptr::write_bytes(contents.as_ptr(), 0xcd, at as usize) };
        let region = Region {
            origin: [0; 3],
            size: [w, w, 1],
        };
        for &(s, _) in &places {
            (texture.replace_region(region, 0, s, &tight(s), bx * block, bx * block * bx))
                .expect("replace_region");
        }
        let queue = device.new_command_queue(4).expect("a queue");
        let command_buffer = queue.command_buffer().expect("a command buffer");
        let mut encoder = command_buffer.blit_command_encoder().expect("an encoder");
        for &(s, offset) in &places {
            let place = TexturePlace {
                slice: s,
                level: 0,
                origin: [0; 3],
            };
            let layout = BufferLayout {
                offset,
                bytes_per_row: row,
                bytes_per_image: row * bx,
            };
            (encoder.copy_from_texture_to_buffer(&texture, place, [w, w, 1], &buffer, layout))
                .expect("a copy out");
        }
        encoder.end_encoding();
        command_buffer.commit().expect("a commit");
        finished(name, &command_buffer);
        // SAFETY: the copy completed and its writes reach the host.
        let ours = unsafe { std::slice::from_raw_parts(contents.as_ptr(), at as usize) }.to_vec();
        let first = (0..at as usize).find(|&i| ours[i] != want[i]);
        assert_eq!(
            first, None,
            "{name}: {pf:?} {bx} blocks wide, rows of {row} bytes, offset {rem} past 32: the first \
             wrong byte"
        );
        let power = row_power(bx * block);
        drop((buffer, texture, queue, command_buffer));
        device.release_block(blk).expect("a release");
        places
            .iter()
            .any(|&(_, offset)| offset.is_multiple_of(power))
            && !row.is_multiple_of(power)
    }

    /// Copies out of a texture to bytes land each row at its stride, the padding between rows
    /// untouched, for 1-, 2-, 4-, 8- and 16-byte elements and 8-byte blocks, 2 to 6 blocks wide,
    /// rows packed or 1 to 2 blocks past, at offsets on and off twice the element: the cells
    /// where the UHD 630's driver (101.2145) wrote a single region's rows
    /// floor(stride / M) × M bytes apart. Those copies are split (`crate::copy_out`), which the
    /// split count shows.
    #[test]
    fn every_device_copies_a_textures_rows_out_at_their_stride() {
        each_device("copy_out_rows", |name, _, device| {
            let grant = (device.grant_indices(IndexClass::Texture, 1, 256)).expect("indices");
            let (mut index, mut class) = (0, 0);
            let before = SPLIT.load(Ordering::SeqCst);
            for format in [
                (PixelFormat::R8Uint, 1, 1),
                (PixelFormat::RG8Uint, 2, 1),
                (PixelFormat::R32Uint, 4, 1),
                (PixelFormat::RGBA8Unorm, 4, 1),
                (PixelFormat::RG32Uint, 8, 1),
                (PixelFormat::RGBA16Float, 8, 1),
                (PixelFormat::BC1RGBA, 8, 4),
                (PixelFormat::BC4RUnorm, 8, 4),
                (PixelFormat::RGBA32Uint, 16, 1),
            ] {
                for bx in [2, 3, 4, 6] {
                    for extra in [0, 1, 2] {
                        for rem in [0, format.1] {
                            index += 1;
                            class += copy_out(index, name, &device, format, bx, extra, rem) as u32;
                        }
                    }
                }
            }
            let split = SPLIT.load(Ordering::SeqCst) - before;
            println!(
                "{name}: {index} copies of 2 slices, {class} of the class, {split} regions split"
            );
            assert!(
                class > 0 && split >= u64::from(class),
                "{name}: the class's copies split"
            );
            (device.release_indices(grant)).expect("a release");
        });
    }
}
