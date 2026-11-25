//! A compressed region of whole blocks that ends past its level's edge inside the level's last
//! block, the one an edge of no whole blocks rounds up to: the one 4 x 4 block of a level smaller
//! than a block.
//!
//! - Its place stops at the level's edge, between textures and between a buffer and a texture,
//!   either way, and the copy command's extent with it: the blocks are the size's. A size past
//!   that block (8 x 4 at a level of 4 x 2), an origin at the edge, and a size of no whole blocks
//!   past the edge are `Refusal::TextureRegion`.
//! - Under a device cfg, on every device of the cfg's kind that makes BC1 or BC3 textures: at
//!   levels 9, 10 and 11 of a 2048 x 1024 texture of 12 levels, a 4 x 4 copy from a Managed
//!   texture into a Private one (a guest's), from a buffer, into a buffer, by `replace_region` and
//!   by `get_bytes` copies the level's one block whole, every byte of levels 8 to 11 (or of the
//!   bytes read) plain Metal's: probe `compressed_copies_of_small_levels.m` in the conformance
//!   crate, an Apple M3 Max, macOS 27.0.1, job m3-max-73696, with the validation layer off.

use crate::blit::{buffer_copy, texture_copy, BufferLayout, Lies, Side, TexturePlace};
use crate::error::{Error, Gap, Refusal};
use crate::texture::{CopyUnit, Placed};
use crate::{PixelFormat, TextureDescriptor, TextureType};
use ash::vk::{self, Handle};

/// A BC3 2048 x 1024 texture of 12 levels: level 8 is 8 x 4 texels, two blocks, and levels 9,
/// 10 and 11 are 4 x 2, 2 x 1 and 1 x 1, one block each.
fn chain() -> TextureDescriptor {
    let mut d = TextureDescriptor::new(TextureType::Type2D, PixelFormat::BC3RGBA, 2048, 1024);
    d.mipmap_level_count = 12;
    d
}

fn side(d: &TextureDescriptor) -> Side<'_> {
    Side {
        descriptor: d,
        format: vk::Format::BC3_UNORM_BLOCK,
        unit: Some(CopyUnit {
            aspect: vk::ImageAspectFlags::COLOR,
            bytes: 16,
            block: [4, 4],
        }),
        base: [0, 0],
        rows: None,
        bound: None,
    }
}

fn at(level: u32, origin: [u32; 3]) -> TexturePlace {
    TexturePlace {
        slice: 0,
        level,
        origin,
    }
}

fn placed(level: u32, origin: [u32; 3], size: [u32; 3]) -> Placed {
    Placed {
        level,
        layer: 0,
        origin,
        size,
    }
}

fn refused<T>(result: Result<Option<T>, Error>) -> Refusal {
    match result {
        Err(Error::Refused(refusal)) => refusal,
        Err(other) => panic!("not a refusal: {other}"),
        Ok(_) => panic!("not a refusal: it was checked"),
    }
}

/// A Shared buffer of 256 bytes, a region at 0 in it, rows of blocks `bytes_per_row` apart.
fn lies(bytes_per_row: u64) -> Lies {
    Lies {
        buffer: vk::Buffer::from_raw(1),
        base: 0,
        len: 256,
        layout: BufferLayout {
            offset: 0,
            bytes_per_row,
            bytes_per_image: 0,
        },
    }
}

#[test]
fn whole_blocks_past_a_level_s_edge_in_its_last_block_place_to_the_edge() {
    let d = chain();
    let s = side(&d);
    // The block at levels 9, 10 and 11, between textures: each side's place stops at the edge.
    for (level, edge) in [(9, [4, 2, 1]), (10, [2, 1, 1]), (11, [1, 1, 1])] {
        let copy = texture_copy(
            &s,
            at(level, [0; 3]),
            [4, 4, 1],
            &s,
            at(level, [0; 3]),
            false,
        )
        .expect("a copy")
        .expect("a block");
        assert_eq!(copy.from, placed(level, [0; 3], edge), "level {level}");
        assert_eq!(copy.to, placed(level, [0; 3], edge), "level {level}");
        let region = copy.regions().collect::<Vec<_>>();
        assert_eq!(region.len(), 1);
        let e = region[0].extent;
        assert_eq!(
            [e.width, e.height, e.depth],
            edge,
            "level {level}: the command's extent"
        );
        // Between a buffer and a texture, either way: one block's row, 16 bytes.
        for into_image in [true, false] {
            let copy = buffer_copy(&s, at(level, [0; 3]), [4, 4, 1], lies(16), into_image)
                .expect("a copy")
                .expect("a block");
            assert_eq!(copy.placed, placed(level, [0; 3], edge), "level {level}");
        }
    }
    // Level 8 is two blocks: its second, whole, is not rounded.
    let copy = texture_copy(&s, at(8, [4, 0, 0]), [4, 4, 1], &s, at(8, [4, 0, 0]), false)
        .expect("a copy")
        .expect("a block");
    assert_eq!(copy.from, placed(8, [4, 0, 0], [4, 4, 1]));
    // Past the block the edge rounds up to, from an origin at the edge, and a size of no whole
    // blocks past it.
    for (level, origin, size) in [
        (9, [0; 3], [8, 4, 1]),
        (9, [0, 0, 0], [4, 8, 1]),
        (9, [4, 0, 0], [4, 4, 1]),
        (9, [0; 3], [4, 3, 1]),
        (10, [0; 3], [3, 1, 1]),
    ] {
        assert!(
            matches!(
                refused(texture_copy(
                    &s,
                    at(level, origin),
                    size,
                    &s,
                    at(level, origin),
                    false
                )),
                Refusal::TextureRegion { .. }
            ),
            "level {level} {origin:?} {size:?}"
        );
        assert!(
            matches!(
                refused(buffer_copy(&s, at(level, origin), size, lies(32), true)),
                Refusal::TextureRegion { .. }
            ),
            "level {level} {origin:?} {size:?}"
        );
    }
}

#[test]
fn a_copy_between_levels_whose_places_stop_at_different_edges_is_a_gap() {
    let d = chain();
    let s = side(&d);
    let gap = Gap::TextureCopy {
        source: PixelFormat::BC3RGBA,
        destination: PixelFormat::BC3RGBA,
    };
    // The block of level 9 (4 x 2) into the second block of level 8 (8 x 4), and back; level 10
    // (2 x 1) the same.
    for (from, to) in [
        (at(9, [0; 3]), at(8, [4, 0, 0])),
        (at(8, [4, 0, 0]), at(9, [0; 3])),
        (at(10, [0; 3]), at(8, [4, 0, 0])),
        (at(8, [4, 0, 0]), at(10, [0; 3])),
    ] {
        match texture_copy(&s, from, [4, 4, 1], &s, to, false) {
            Err(Error::Unsupported(found)) => assert_eq!(found, gap, "{from:?} {to:?}"),
            Err(other) => panic!("{from:?} {to:?}: not a gap: {other}"),
            Ok(_) => panic!("{from:?} {to:?}: copied"),
        }
    }
    // Two levels whose places stop at one edge copy: level 9 of one chain into level 9 of
    // another.
    let other = chain();
    let o = side(&other);
    let copy = texture_copy(&s, at(9, [0; 3]), [4, 4, 1], &o, at(9, [0; 3]), false)
        .expect("a copy")
        .expect("a block");
    assert_eq!(copy.to, placed(9, [0; 3], [4, 2, 1]));
}

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::command::on_device::{each_device, finished};
    use crate::{
        BlitCommandEncoder, Block, Buffer, BufferLayout, CommandBuffer, CommandQueue, Device,
        Error, Gap, HostBytes, MemoryClass, PixelFormat, Region, StorageMode, Texture,
        TextureDescriptor, TexturePlace, TextureType, TextureUsage,
    };

    /// The levels each case fills and reads back: 8 (8 x 4 texels), 9 (4 x 2), 10 (2 x 1) and 11
    /// (1 x 1).
    const LEVELS: std::ops::RangeInclusive<u32> = 8..=11;

    /// A level's extent in texels.
    fn extent(level: u32) -> [u32; 3] {
        [(2048 >> level).max(1), (1024 >> level).max(1), 1]
    }

    /// A level's blocks across and down.
    fn blocks(level: u32) -> [usize; 2] {
        let [w, h, _] = extent(level);
        [w.div_ceil(4) as usize, h.div_ceil(4) as usize]
    }

    /// A level's bytes in rows of blocks, packed: byte `k` the probe's source byte, or `fill`.
    fn level_bytes(level: u32, unit: usize, fill: Option<u8>) -> Vec<u8> {
        let [across, down] = blocks(level);
        (0..across * down * unit)
            .map(|k| fill.unwrap_or(((0x11 * level as usize + 3 * k + 1) & 0xff) as u8))
            .collect()
    }

    /// FNV-1a 64 of `bytes`.
    fn fnv(bytes: &[u8]) -> u64 {
        bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, &byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
        })
    }

    fn whole(level: u32) -> Region {
        Region {
            origin: [0; 3],
            size: extent(level),
        }
    }

    fn place(level: u32) -> TexturePlace {
        TexturePlace {
            slice: 0,
            level,
            origin: [0; 3],
        }
    }

    /// One format's run: its textures, a Shared block for its buffers, and the queue.
    struct Run<'a> {
        name: &'a str,
        device: &'a Device,
        queue: &'a CommandQueue,
        block: &'a Block,
        format: PixelFormat,
        unit: usize,
    }

    impl Run<'_> {
        fn texture(&self, storage_mode: StorageMode) -> Texture {
            let mut d = TextureDescriptor::new(TextureType::Type2D, self.format, 2048, 1024);
            d.mipmap_level_count = 12;
            d.usage = TextureUsage::SHADER_READ;
            d.storage_mode = storage_mode;
            (self.device.new_texture(&d, None))
                .unwrap_or_else(|error| panic!("{}: {error}", self.name))
        }

        /// A buffer of `bytes` at `at` in the block.
        fn buffer(&self, at: u64, bytes: &[u8]) -> Buffer {
            let len = bytes.len() as u64;
            let buffer = (self.device.new_buffer(self.block.id, at, len, None))
                .unwrap_or_else(|error| panic!("{}: {error}", self.name));
            let contents = buffer.contents().expect("a Shared buffer");
            // SAFETY: the range lies in the buffer, which is alive, and no GPU work uses it.
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr(), bytes.len())
            };
            buffer
        }

        /// A command buffer recorded by `blit`, run to completion with no error.
        fn run(&self, blit: impl FnOnce(&mut BlitCommandEncoder)) {
            let command_buffer: CommandBuffer = self.queue.command_buffer().expect("a buffer");
            let mut encoder = command_buffer.blit_command_encoder().expect("an encoder");
            blit(&mut encoder);
            encoder.end_encoding();
            command_buffer.commit().expect("a commit");
            finished(self.name, &command_buffer);
            assert_eq!(command_buffer.error(), None, "{}", self.name);
        }

        /// Levels 8 to 11 of `texture` filled with `fill`, or the source's bytes: by
        /// `replace_region` where it is Managed, else by a copy from a buffer.
        fn fill(&self, texture: &Texture, fill: Option<u8>) {
            if texture.descriptor().storage_mode == StorageMode::Managed {
                for level in LEVELS {
                    let bytes = level_bytes(level, self.unit, fill);
                    let row = (blocks(level)[0] * self.unit) as u64;
                    (texture.replace_region(
                        whole(level),
                        level,
                        0,
                        HostBytes::from(&bytes[..]),
                        row,
                        0,
                    ))
                    .unwrap_or_else(|error| panic!("{}: {error}", self.name));
                }
                return;
            }
            let staging: Vec<_> = LEVELS
                .map(|level| {
                    (
                        level,
                        self.buffer(
                            1024 + 256 * u64::from(level - 8),
                            &level_bytes(level, self.unit, fill),
                        ),
                    )
                })
                .collect();
            self.run(|encoder| {
                for (level, buffer) in &staging {
                    let layout = BufferLayout {
                        offset: 0,
                        bytes_per_row: (blocks(*level)[0] * self.unit) as u64,
                        bytes_per_image: 0,
                    };
                    (encoder.copy_from_buffer(
                        buffer,
                        layout,
                        extent(*level),
                        texture,
                        place(*level),
                    ))
                    .unwrap_or_else(|error| panic!("{}: {error}", self.name));
                }
            });
        }

        /// Levels 8 to 11 of `texture`, by a copy of each whole into one buffer, packed.
        fn levels(&self, texture: &Texture) -> Vec<u8> {
            let lens: Vec<usize> = LEVELS
                .map(|level| blocks(level).iter().product::<usize>() * self.unit)
                .collect();
            let out = self.buffer(2048, &vec![0xbb; lens.iter().sum()]);
            self.run(|encoder| {
                let mut offset = 0;
                for (level, len) in LEVELS.zip(&lens) {
                    let row = (blocks(level)[0] * self.unit) as u64;
                    let layout = BufferLayout {
                        offset,
                        bytes_per_row: row,
                        bytes_per_image: *len as u64,
                    };
                    (encoder.copy_from_texture_to_buffer(
                        texture,
                        place(level),
                        extent(level),
                        &out,
                        layout,
                    ))
                    .unwrap_or_else(|error| panic!("{}: {error}", self.name));
                    offset += *len as u64;
                }
            });
            read(&out)
        }
    }

    /// A Shared buffer's bytes, as the host reads them.
    fn read(buffer: &Buffer) -> Vec<u8> {
        let contents = buffer.contents().expect("a Shared buffer");
        // SAFETY: the buffer is alive, `length` bytes long, and no GPU work uses it.
        unsafe { std::slice::from_raw_parts(contents.as_ptr(), buffer.length() as usize) }.to_vec()
    }

    /// Plain Metal's FNV-1a 64 of each case's read-back: probe
    /// `compressed_copies_of_small_levels.m` in the conformance crate, an Apple M3 Max, macOS
    /// 27.0.1, job m3-max-73696, with the validation layer off: levels 8 to 11 of a texture
    /// destination, or a buffer destination's 64 bytes.
    const METAL: [(&str, u64); 30] = [
        ("bc1-mp-9-block", 0xdf21_4fe1_8bfe_662d),
        ("bc1-b2t-9-block", 0xdf21_4fe1_8bfe_662d),
        ("bc1-t2b-9-block", 0xeb66_b13e_7d84_37ad),
        ("bc1-replace-9-block", 0xdf21_4fe1_8bfe_662d),
        ("bc1-get-9-block", 0xeb66_b13e_7d84_37ad),
        ("bc1-mp-10-block", 0xfda6_5c9a_502d_a7a5),
        ("bc1-b2t-10-block", 0xfda6_5c9a_502d_a7a5),
        ("bc1-t2b-10-block", 0xe466_c8da_c77e_3825),
        ("bc1-replace-10-block", 0xfda6_5c9a_502d_a7a5),
        ("bc1-get-10-block", 0xe466_c8da_c77e_3825),
        ("bc1-mp-11-block", 0x3145_d998_df90_8f15),
        ("bc1-b2t-11-block", 0x3145_d998_df90_8f15),
        ("bc1-t2b-11-block", 0xe2d8_ac95_6a0c_98d5),
        ("bc1-replace-11-block", 0x3145_d998_df90_8f15),
        ("bc1-get-11-block", 0xe2d8_ac95_6a0c_98d5),
        ("bc3-mp-9-block", 0xcd21_5888_fb21_8a95),
        ("bc3-b2t-9-block", 0xcd21_5888_fb21_8a95),
        ("bc3-t2b-9-block", 0xb5b6_a640_cc0c_0ed5),
        ("bc3-replace-9-block", 0xcd21_5888_fb21_8a95),
        ("bc3-get-9-block", 0xb5b6_a640_cc0c_0ed5),
        ("bc3-mp-10-block", 0x716f_1b3f_1cb3_9e25),
        ("bc3-b2t-10-block", 0x716f_1b3f_1cb3_9e25),
        ("bc3-t2b-10-block", 0xe802_82a8_6672_47e5),
        ("bc3-replace-10-block", 0x716f_1b3f_1cb3_9e25),
        ("bc3-get-10-block", 0xe802_82a8_6672_47e5),
        ("bc3-mp-11-block", 0x0a3e_663a_b6db_29e5),
        ("bc3-b2t-11-block", 0x0a3e_663a_b6db_29e5),
        ("bc3-t2b-11-block", 0x7e40_11a7_6428_e2e5),
        ("bc3-replace-11-block", 0x0a3e_663a_b6db_29e5),
        ("bc3-get-11-block", 0x7e40_11a7_6428_e2e5),
    ];

    fn metal(case: &str) -> u64 {
        (METAL.iter().find(|(name, _)| *name == case))
            .unwrap_or_else(|| panic!("no hash for {case}"))
            .1
    }

    /// Plain Metal's bytes and the model's: `bytes` are `model`, and their hash the probe's.
    fn check(name: &str, case: &str, bytes: Vec<u8>, model: Vec<u8>) {
        assert_eq!(bytes, model, "{name}: {case}: every byte");
        assert_eq!(
            fnv(&bytes),
            metal(case),
            "{name}: {case}: plain Metal's bytes"
        );
    }

    /// What a case leaves: levels 8 to 11 of a texture destination of 0xdd, `level` the source's.
    fn landed(level: u32, unit: usize) -> Vec<u8> {
        LEVELS
            .flat_map(|l| level_bytes(l, unit, (l != level).then_some(0xdd)))
            .collect()
    }

    /// What a case leaves in a buffer destination of 64 bytes of 0xee: `level`'s one block first.
    fn read_out(level: u32, unit: usize) -> Vec<u8> {
        let mut bytes = vec![0xee; 64];
        bytes[..unit].copy_from_slice(&level_bytes(level, unit, None));
        bytes
    }

    #[test]
    fn every_device_copies_a_level_smaller_than_a_block_at_the_block_s_size() {
        each_device("small_levels", |name, _, device| {
            let queue = device.new_command_queue(4).expect("a queue");
            let block = (device.take_block(MemoryClass::Shared, 4096)).expect("a Shared block");
            for (format, unit, label) in [
                (PixelFormat::BC1RGBA, 8, "bc1"),
                (PixelFormat::BC3RGBA, 16, "bc3"),
            ] {
                let mut d = TextureDescriptor::new(TextureType::Type2D, format, 2048, 1024);
                d.mipmap_level_count = 12;
                match device.new_texture(&d, None) {
                    Ok(_) => {}
                    Err(Error::Unsupported(Gap::TextureFormat { .. })) => {
                        println!("{name}: no {format:?} texture on this device");
                        continue;
                    }
                    Err(error) => panic!("{name}: {format:?}: {error}"),
                }
                let run = Run {
                    name,
                    device: &device,
                    queue: &queue,
                    block: &block,
                    format,
                    unit,
                };
                let size = [4, 4, 1];
                for level in 9..=11 {
                    let case = |kind: &str| format!("{label}-{kind}-{level}-block");
                    // The guest's copy: a Managed texture's level into a Private one's.
                    let from = run.texture(StorageMode::Managed);
                    run.fill(&from, None);
                    let to = run.texture(StorageMode::Private);
                    run.fill(&to, Some(0xdd));
                    run.run(|encoder| {
                        (encoder.copy_from_texture(&from, place(level), size, &to, place(level)))
                            .unwrap_or_else(|error| panic!("{name}: {}: {error}", case("mp")))
                    });
                    check(name, &case("mp"), run.levels(&to), landed(level, unit));
                    // A buffer's block into a Private texture.
                    let to = run.texture(StorageMode::Private);
                    run.fill(&to, Some(0xdd));
                    let bytes = run.buffer(0, &level_bytes(level, unit, None));
                    let layout = BufferLayout {
                        offset: 0,
                        bytes_per_row: unit as u64,
                        bytes_per_image: 0,
                    };
                    run.run(|encoder| {
                        (encoder.copy_from_buffer(&bytes, layout, size, &to, place(level)))
                            .unwrap_or_else(|error| panic!("{name}: {}: {error}", case("b2t")))
                    });
                    check(name, &case("b2t"), run.levels(&to), landed(level, unit));
                    // A Private texture's block into a buffer of 0xee.
                    let from = run.texture(StorageMode::Private);
                    run.fill(&from, None);
                    let out = run.buffer(256, &[0xee; 64]);
                    let layout = BufferLayout {
                        offset: 0,
                        bytes_per_row: unit as u64,
                        bytes_per_image: unit as u64,
                    };
                    run.run(|encoder| {
                        (encoder.copy_from_texture_to_buffer(
                            &from,
                            place(level),
                            size,
                            &out,
                            layout,
                        ))
                        .unwrap_or_else(|error| panic!("{name}: {}: {error}", case("t2b")))
                    });
                    check(name, &case("t2b"), read(&out), read_out(level, unit));
                    // replaceRegion of the block into a Managed texture of 0xdd.
                    let to = run.texture(StorageMode::Managed);
                    run.fill(&to, Some(0xdd));
                    let block_bytes = level_bytes(level, unit, None);
                    let region = Region {
                        origin: [0; 3],
                        size,
                    };
                    let row = unit as u64;
                    (to.replace_region(
                        region,
                        level,
                        0,
                        HostBytes::from(&block_bytes[..]),
                        row,
                        0,
                    ))
                    .unwrap_or_else(|error| panic!("{name}: {}: {error}", case("replace")));
                    check(name, &case("replace"), run.levels(&to), landed(level, unit));
                    // getBytes of the block of a Managed texture into bytes of 0xee.
                    let from = run.texture(StorageMode::Managed);
                    run.fill(&from, None);
                    let mut got = vec![0xee; 64];
                    (from.get_bytes(&mut got, row, 0, region, level, 0))
                        .unwrap_or_else(|error| panic!("{name}: {}: {error}", case("get")));
                    check(name, &case("get"), got, read_out(level, unit));
                    println!("{name}: {label} level {level}: the 4 x 4 block copied by each path");
                }
            }
            device.release_block(block).expect("the block back");
        });
    }
}
