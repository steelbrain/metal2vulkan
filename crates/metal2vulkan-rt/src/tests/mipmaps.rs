//! `generateMipmapsForTexture:` on the blit encoder.
//!
//! - Apple defines mipmaps of colour-renderable, filterable formats alone: a depth or stencil,
//!   compressed, YCbCr or integer format is not one (`mipmapped`).
//! - Under a device cfg, on every device of the cfg's kind: each level below the first of an
//!   8 by 8 `RGBA8Unorm` texture of four levels is, in every channel, within one code of the box
//!   of the four texels it halves in the level above as the device left them (the static note
//!   `BlitCommandEncoder::GENERATE_MIPMAPS_NOTE`: the filter is the device's), its first level
//!   kept; the chain is one submission and its four barriers, besides the segment's end, its
//!   first level's bytes uploaded and read back before it. A texture of one level records
//!   nothing, and one of an integer format is `Refusal::MipmapFormat`.

use crate::blit::mipmapped;
use crate::PixelFormat;

#[test]
fn mipmaps_are_of_colour_renderable_filterable_formats() {
    for (pf, expected) in [
        (PixelFormat::RGBA8Unorm, true),
        (PixelFormat::RGBA8UnormSRGB, true),
        (PixelFormat::BGRA8Unorm, true),
        (PixelFormat::RGBA16Float, true),
        (PixelFormat::R32Float, true),
        (PixelFormat::RGBA8Uint, false),
        (PixelFormat::R32Sint, false),
        (PixelFormat::BC1RGBA, false),
        (PixelFormat::Depth32Float, false),
        (PixelFormat::Stencil8, false),
    ] {
        assert_eq!(mipmapped(pf), expected, "{pf:?}");
    }
}

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::blit::on_device::settled;
    use crate::tests::command::on_device::{each_device, finished};
    use crate::{
        BlitCommandEncoder, Device, Error, PixelFormat, Refusal, Region, StorageMode, Texture,
        TextureDescriptor, TextureType, TextureUsage,
    };

    /// The texture's side, and its levels.
    const SIDE: u32 = 8;
    const LEVELS: u32 = 4;

    /// A Shared 2D texture of `pf`, `SIDE` square, of `levels` levels.
    fn texture(name: &str, device: &Device, pf: PixelFormat, levels: u32) -> Texture {
        let mut d = TextureDescriptor::new(TextureType::Type2D, pf, SIDE, SIDE);
        d.mipmap_level_count = levels;
        d.usage = TextureUsage::RENDER_TARGET;
        d.storage_mode = StorageMode::Shared;
        (device.new_texture(&d, None)).unwrap_or_else(|error| panic!("{name}: {error}"))
    }

    /// Level `level` of an `RGBA8Unorm` texture, row by row.
    fn level_bytes(name: &str, texture: &Texture, level: u32) -> Vec<u8> {
        let side = (SIDE >> level).max(1);
        let mut bytes = vec![0u8; (side * side * 4) as usize];
        let whole = Region {
            origin: [0; 3],
            size: [side, side, 1],
        };
        let row = u64::from(side) * 4;
        (texture.get_bytes(&mut bytes, row, row * u64::from(side), whole, level, 0))
            .unwrap_or_else(|error| panic!("{name}: level {level}: {error}"));
        bytes
    }

    /// One blit encoder's `generate_mipmaps` of `texture`, committed and finished.
    fn generate(name: &str, device: &Device, texture: &Texture) -> Result<(), Error> {
        let queue = device.new_command_queue(1).expect("a queue");
        let command_buffer = queue.command_buffer().expect("a command buffer");
        let mut encoder: BlitCommandEncoder =
            (command_buffer.blit_command_encoder()).expect("an encoder");
        let generated = encoder.generate_mipmaps(texture);
        encoder.end_encoding();
        command_buffer.commit().expect("a commit");
        finished(name, &command_buffer);
        assert_eq!(command_buffer.error(), None, "{name}");
        generated
    }

    #[test]
    fn every_device_makes_each_level_a_box_of_the_one_above() {
        each_device("mipmaps", |name, _, device| {
            let mipmapped = texture(name, &device, PixelFormat::RGBA8Unorm, LEVELS);
            // Texel (x, y)'s channel c, a pattern no box of four texels makes even.
            let first: Vec<u8> = (0..SIDE * SIDE * 4)
                .map(|i| {
                    let (texel, c) = (i / 4, i % 4);
                    ((texel % SIDE) * 37 + (texel / SIDE) * 59 + c * 83) as u8
                })
                .collect();
            let whole = Region {
                origin: [0; 3],
                size: [SIDE, SIDE, 1],
            };
            let row = u64::from(SIDE) * 4;
            (mipmapped.replace_region(
                whole,
                0,
                0,
                crate::HostBytes::from(&first[..]),
                row,
                row * u64::from(SIDE),
            ))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
            // Where the bytes are staged, they wait for the device's upload, which goes ahead of
            // the next submission; reading them back submits any such upload and waits for it, so
            // the chain's command buffer is measured alone, its upload's barrier outside it.
            assert_eq!(
                level_bytes(name, &mipmapped, 0),
                first,
                "{name}: level 0 uploaded"
            );
            let counters = device.counters();
            settled(&device);
            let before = counters.metrics();
            let closings = counters.closings();
            (generate(name, &device, &mipmapped)).unwrap_or_else(|e| panic!("{name}: {e}"));
            let after = counters.metrics();
            // The chain's last barrier holds one global memory barrier beside its image barriers.
            let closed = counters.closings();
            assert_eq!(
                (closed.0 - closings.0, closed.1 - closings.1),
                (1, 1),
                "{name}: the chain's closing barrier and its global memory barrier"
            );
            assert_eq!(
                (
                    after.submissions - before.submissions,
                    after.barriers - before.barriers
                ),
                (1, u64::from(LEVELS) + 1),
                "{name}: submissions, barriers (the chain's and the segment's end)"
            );
            assert_eq!(level_bytes(name, &mipmapped, 0), first, "{name}: level 0");
            let mut worst = 0.0f32;
            for level in 1..LEVELS {
                let above = level_bytes(name, &mipmapped, level - 1);
                let got = level_bytes(name, &mipmapped, level);
                let (side, wide) = ((SIDE >> level) as usize, (SIDE >> (level - 1)) as usize);
                for y in 0..side {
                    for x in 0..side {
                        for c in 0..4 {
                            let at = |x: usize, y: usize| f32::from(above[(y * wide + x) * 4 + c]);
                            let (x2, y2) = (2 * x, 2 * y);
                            let boxed =
                                (at(x2, y2) + at(x2 + 1, y2) + at(x2, y2 + 1) + at(x2 + 1, y2 + 1))
                                    / 4.0;
                            let texel = f32::from(got[(y * side + x) * 4 + c]);
                            let off = (texel - boxed).abs();
                            worst = worst.max(off);
                            assert!(
                                off <= 1.0,
                                "{name}: level {level} texel ({x}, {y}) channel {c}: {texel} \
                                 against the box {boxed}"
                            );
                        }
                    }
                }
            }

            // One level: nothing to make, nothing recorded but the command buffer's own.
            let one = texture(name, &device, PixelFormat::RGBA8Unorm, 1);
            settled(&device);
            let before = counters.metrics();
            (generate(name, &device, &one)).unwrap_or_else(|e| panic!("{name}: {e}"));
            let after = counters.metrics();
            assert_eq!(after.barriers, before.barriers, "{name}: one level");

            // An integer format: Apple defines no mipmaps of it.
            let integer = texture(name, &device, PixelFormat::RGBA8Uint, LEVELS);
            match generate(name, &device, &integer) {
                Err(Error::Refused(Refusal::MipmapFormat {
                    pixel_format: PixelFormat::RGBA8Uint,
                })) => {}
                other => panic!("{name}: an integer format's mipmaps: {other:?}"),
            }
            drop((mipmapped, one, integer));
            println!(
                "{name}: each of {} levels within {worst} of a code of the box above",
                LEVELS - 1
            );
        });
    }
}
