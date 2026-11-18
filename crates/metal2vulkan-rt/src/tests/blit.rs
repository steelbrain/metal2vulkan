//! Blit copies between textures.
//!
//! - The rules: formats whose units have the same bytes copy into each other; units of other
//!   bytes, and other sample counts, are `Refusal::TextureCopyMismatch`; what one copy command in
//!   the floor does not make is `Gap::TextureCopy`; a level, slice or region outside a texture or
//!   off its block grid is `Refusal::TextureRegion`; a size of 0 on an axis a texture lacks copies
//!   one texel there, and on another copies nothing; a 3D texture's image copies to and from a 2D
//!   slice.
//! - Under a device cfg, on every device of the cfg's kind: the bytes of a level of a 2D texture
//!   land in a slice of a 2D array of another format of one unit's bytes, a cube's face in an
//!   image of a 3D texture, an image of a 3D texture in a 2D texture, a slice of an array in
//!   another slice of it, and a block-compressed region in another texture where the device makes
//!   one, everything around each region kept. One copy alone is one submission, two barriers and
//!   the segment's end, and copies no byte of the runtime's own; a barrier an encoder owed is paid
//!   by the copy's first. Every gauge comes back to zero with the device.

use crate::blit::{texture_copy, ImageCopy, Side, TexturePlace};
use crate::error::{Error, Gap, Refusal};
use crate::texture::{CopyUnit, Placed};
use crate::{PixelFormat, TextureDescriptor, TextureType};
use ash::vk;

/// A side of a copy: a texture of `d` held in `format`, of `unit`.
fn side(d: &TextureDescriptor, format: vk::Format, unit: Option<CopyUnit>) -> Side<'_> {
    Side {
        descriptor: d,
        format,
        unit,
    }
}

fn color(bytes: u32) -> Option<CopyUnit> {
    Some(CopyUnit {
        aspect: vk::ImageAspectFlags::COLOR,
        bytes,
        block: [1, 1],
    })
}

fn descriptor(
    texture_type: TextureType,
    format: PixelFormat,
    extent: [u32; 3],
) -> TextureDescriptor {
    let mut d = TextureDescriptor::new(texture_type, format, extent[0], extent[1]);
    d.depth = extent[2];
    d
}

fn at(slice: u32, level: u32, origin: [u32; 3]) -> TexturePlace {
    TexturePlace {
        slice,
        level,
        origin,
    }
}

fn placed(level: u32, layer: u32, offset: [i32; 3], extent: [u32; 3]) -> Placed {
    Placed {
        level,
        layer,
        offset: vk::Offset3D {
            x: offset[0],
            y: offset[1],
            z: offset[2],
        },
        extent: vk::Extent3D {
            width: extent[0],
            height: extent[1],
            depth: extent[2],
        },
    }
}

fn refused<T: std::fmt::Debug>(result: Result<T, Error>) -> Refusal {
    match result {
        Err(Error::Refused(refusal)) => refusal,
        other => panic!("not a refusal: {other:?}"),
    }
}

fn gap<T: std::fmt::Debug>(result: Result<T, Error>) -> Gap {
    match result {
        Err(Error::Unsupported(gap)) => gap,
        other => panic!("not a gap: {other:?}"),
    }
}

#[test]
fn a_copy_takes_formats_of_one_units_bytes_and_places_both_regions() {
    let mut rgba = descriptor(TextureType::Type2D, PixelFormat::RGBA8Unorm, [13, 7, 1]);
    rgba.mipmap_level_count = 3;
    let mut words = descriptor(TextureType::Type2DArray, PixelFormat::R32Uint, [8, 8, 1]);
    words.array_length = 3;
    let source = side(&rgba, vk::Format::R8G8B8A8_UNORM, color(4));
    let destination = side(&words, vk::Format::R32_UINT, color(4));
    let copy = texture_copy(
        &source,
        at(0, 1, [1, 0, 0]),
        [4, 3, 1],
        &destination,
        at(2, 0, [2, 4, 0]),
        false,
    )
    .expect("a copy");
    assert_eq!(
        copy,
        Some(ImageCopy {
            from: placed(1, 0, [1, 0, 0], [4, 3, 1]),
            to: placed(0, 2, [2, 4, 0], [4, 3, 1]),
            aspect: vk::ImageAspectFlags::COLOR,
        })
    );
    // A depth of 0 on 2D textures copies one image; a width of 0 copies nothing.
    let flat = texture_copy(
        &source,
        at(0, 0, [0; 3]),
        [2, 2, 0],
        &destination,
        at(0, 0, [0; 3]),
        false,
    )
    .expect("a copy")
    .expect("one image");
    assert_eq!(flat.from.extent.depth, 1);
    assert_eq!(
        texture_copy(
            &source,
            at(0, 0, [0; 3]),
            [0, 2, 1],
            &destination,
            at(0, 0, [0; 3]),
            false
        )
        .expect("nothing"),
        None
    );
    // A level, a slice or a region past either texture.
    for (from, to, size) in [
        (at(0, 3, [0; 3]), at(0, 0, [0; 3]), [1, 1, 1]),
        (at(0, 0, [0; 3]), at(3, 0, [0; 3]), [1, 1, 1]),
        (at(0, 1, [3, 0, 0]), at(0, 0, [0; 3]), [4, 1, 1]),
        (at(0, 0, [0; 3]), at(0, 0, [6, 0, 0]), [4, 1, 1]),
    ] {
        assert!(
            matches!(
                refused(texture_copy(&source, from, size, &destination, to, false)),
                Refusal::TextureRegion { .. }
            ),
            "{from:?} {to:?} {size:?}"
        );
    }
}

#[test]
fn a_3d_image_copies_to_and_from_a_2d_slice_and_a_cube_face_is_its_slice() {
    let volume = descriptor(TextureType::Type3D, PixelFormat::R8Unorm, [4, 4, 3]);
    let plane = descriptor(TextureType::Type2D, PixelFormat::R8Unorm, [4, 4, 1]);
    let cube = descriptor(TextureType::Cube, PixelFormat::R8Unorm, [4, 4, 1]);
    let (v, p, c) = (
        side(&volume, vk::Format::R8_UNORM, color(1)),
        side(&plane, vk::Format::R8_UNORM, color(1)),
        side(&cube, vk::Format::R8_UNORM, color(1)),
    );
    let copy = texture_copy(
        &v,
        at(0, 0, [1, 1, 2]),
        [2, 2, 1],
        &p,
        at(0, 0, [2, 0, 0]),
        false,
    )
    .expect("a copy")
    .expect("one image");
    assert_eq!(copy.from, placed(0, 0, [1, 1, 2], [2, 2, 1]));
    assert_eq!(copy.to, placed(0, 0, [2, 0, 0], [2, 2, 1]));
    let copy = texture_copy(
        &c,
        at(4, 0, [0; 3]),
        [4, 4, 1],
        &v,
        at(0, 0, [0, 0, 1]),
        false,
    )
    .expect("a copy")
    .expect("a face");
    assert_eq!(copy.from, placed(0, 4, [0; 3], [4, 4, 1]));
    assert_eq!(copy.to, placed(0, 0, [0, 0, 1], [4, 4, 1]));
    // Two images of a 3D texture do not fit one 2D slice.
    assert!(matches!(
        refused(texture_copy(
            &v,
            at(0, 0, [0; 3]),
            [4, 4, 2],
            &p,
            at(0, 0, [0; 3]),
            false
        )),
        Refusal::TextureRegion { .. }
    ));
    // One texture's two slices copy; one slice into itself is not built.
    let mut array = descriptor(TextureType::Type2DArray, PixelFormat::R8Unorm, [4, 4, 1]);
    array.array_length = 2;
    let a = side(&array, vk::Format::R8_UNORM, color(1));
    assert!(texture_copy(
        &a,
        at(0, 0, [0; 3]),
        [2, 2, 1],
        &a,
        at(1, 0, [2, 2, 0]),
        true
    )
    .expect("a copy")
    .is_some());
    assert_eq!(
        gap(texture_copy(
            &a,
            at(0, 0, [0; 3]),
            [2, 2, 1],
            &a,
            at(0, 0, [2, 2, 0]),
            true
        )),
        Gap::TextureCopy {
            source: PixelFormat::R8Unorm,
            destination: PixelFormat::R8Unorm
        }
    );
}

#[test]
fn what_metal_leaves_undefined_is_refused_and_what_one_command_does_not_make_is_a_gap() {
    let d = |format, extent| descriptor(TextureType::Type2D, format, extent);
    let rgba = d(PixelFormat::RGBA8Unorm, [8, 8, 1]);
    let rg = d(PixelFormat::RG8Unorm, [8, 8, 1]);
    let bc1 = d(PixelFormat::BC1RGBA, [8, 8, 1]);
    let rg32 = d(PixelFormat::RG32Uint, [8, 8, 1]);
    let depth = d(PixelFormat::Depth32Float, [8, 8, 1]);
    let r32 = d(PixelFormat::R32Float, [8, 8, 1]);
    let alpha = d(PixelFormat::A8Unorm, [8, 8, 1]);
    let r8 = d(PixelFormat::R8Unorm, [8, 8, 1]);
    let line = descriptor(TextureType::Type1D, PixelFormat::R8Unorm, [8, 1, 1]);
    let mut msaa = descriptor(
        TextureType::Type2DMultisample,
        PixelFormat::RGBA8Unorm,
        [8, 8, 1],
    );
    msaa.sample_count = 4;
    let whole = |source: &Side<'_>, destination: &Side<'_>| {
        texture_copy(
            source,
            at(0, 0, [0; 3]),
            [8, 1, 1],
            destination,
            at(0, 0, [0; 3]),
            false,
        )
    };
    let rgba = side(&rgba, vk::Format::R8G8B8A8_UNORM, color(4));
    let bc1 = side(
        &bc1,
        vk::Format::BC1_RGBA_UNORM_BLOCK,
        Some(CopyUnit {
            aspect: vk::ImageAspectFlags::COLOR,
            bytes: 8,
            block: [4, 4],
        }),
    );
    let depth = side(
        &depth,
        vk::Format::D32_SFLOAT,
        Some(CopyUnit {
            aspect: vk::ImageAspectFlags::DEPTH,
            bytes: 4,
            block: [1, 1],
        }),
    );
    let mismatch = |source, destination| Refusal::TextureCopyMismatch {
        source,
        destination,
    };
    let not_built = |source, destination| Gap::TextureCopy {
        source,
        destination,
    };
    assert_eq!(
        refused(whole(&rgba, &side(&rg, vk::Format::R8G8_UNORM, color(2)))),
        mismatch(PixelFormat::RGBA8Unorm, PixelFormat::RG8Unorm)
    );
    assert_eq!(
        refused(whole(&side(&msaa, vk::Format::R8G8B8A8_UNORM, None), &rgba)),
        mismatch(PixelFormat::RGBA8Unorm, PixelFormat::RGBA8Unorm)
    );
    let msaa = side(&msaa, vk::Format::R8G8B8A8_UNORM, None);
    assert_eq!(
        gap(whole(&msaa, &msaa)),
        not_built(PixelFormat::RGBA8Unorm, PixelFormat::RGBA8Unorm)
    );
    assert_eq!(
        gap(whole(&bc1, &side(&rg32, vk::Format::R32G32_UINT, color(8)))),
        not_built(PixelFormat::BC1RGBA, PixelFormat::RG32Uint)
    );
    assert_eq!(
        gap(whole(&depth, &side(&r32, vk::Format::R32_SFLOAT, color(4)))),
        not_built(PixelFormat::Depth32Float, PixelFormat::R32Float)
    );
    assert_eq!(
        gap(whole(
            &side(&alpha, vk::Format::A8_UNORM_KHR, color(1)),
            &side(&r8, vk::Format::R8_UNORM, color(1))
        )),
        not_built(PixelFormat::A8Unorm, PixelFormat::R8Unorm)
    );
    // `A8Unorm` held in `R8_UNORM` copies as its byte.
    assert!(whole(
        &side(&alpha, vk::Format::R8_UNORM, color(1)),
        &side(&r8, vk::Format::R8_UNORM, color(1))
    )
    .expect("a copy")
    .is_some());
    assert_eq!(
        gap(whole(
            &side(&line, vk::Format::R8_UNORM, color(1)),
            &side(&r8, vk::Format::R8_UNORM, color(1))
        )),
        not_built(PixelFormat::R8Unorm, PixelFormat::R8Unorm)
    );
    // A compressed region is whole blocks or reaches the level's edge on both sides.
    assert!(texture_copy(
        &bc1,
        at(0, 0, [4, 0, 0]),
        [4, 4, 1],
        &bc1,
        at(0, 0, [0, 4, 0]),
        false
    )
    .expect("a copy")
    .is_some());
    assert!(matches!(
        refused(texture_copy(
            &bc1,
            at(0, 0, [0; 3]),
            [2, 4, 1],
            &bc1,
            at(0, 0, [0; 3]),
            false
        )),
        Refusal::TextureRegion { .. }
    ));
    let mut chain = descriptor(TextureType::Type2D, PixelFormat::BC1RGBA, [8, 8, 1]);
    chain.mipmap_level_count = 4;
    let chain = side(&chain, vk::Format::BC1_RGBA_UNORM_BLOCK, bc1.unit);
    // Level 2 is 2 x 2 texels, inside one block: its edge, but not the destination's.
    assert!(texture_copy(
        &chain,
        at(0, 2, [0; 3]),
        [2, 2, 1],
        &chain,
        at(0, 3, [0; 3]),
        false
    )
    .is_err());
    assert!(matches!(
        refused(texture_copy(
            &chain,
            at(0, 2, [0; 3]),
            [2, 2, 1],
            &bc1,
            at(0, 0, [0; 3]),
            false
        )),
        Refusal::TextureRegion { .. }
    ));
}

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::command::on_device::{each_device, finished};
    use crate::{
        CommandBuffer, Device, DispatchType, Error, Gap, IndexClass, PixelFormat, Refusal, Region,
        StorageMode, Texture, TextureDescriptor, TexturePlace, TextureType, TextureUsage,
    };

    /// A byte pattern of `len` bytes, different at every offset of one call and between calls.
    fn pattern(len: usize, seed: u8) -> Vec<u8> {
        (0..len)
            .map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed))
            .collect()
    }

    /// One level and slice of a texture, whole, in elements of `unit` bytes covering `block`
    /// texels square.
    struct Level<'a> {
        texture: &'a Texture,
        slice: u32,
        level: u32,
        unit: usize,
        block: u32,
    }

    impl Level<'_> {
        fn extent(&self) -> [u32; 3] {
            let d = self.texture.descriptor();
            [d.width, d.height, d.depth].map(|e| (e >> self.level).max(1))
        }

        /// Elements across, rows and images.
        fn grid(&self) -> [usize; 3] {
            let [w, h, d] = self.extent();
            [
                w.div_ceil(self.block) as usize,
                h.div_ceil(self.block) as usize,
                d as usize,
            ]
        }

        fn layout(&self) -> (Region, u64, u64, usize) {
            let [across, rows, images] = self.grid();
            let row = across * self.unit;
            let region = Region {
                origin: [0; 3],
                size: self.extent(),
            };
            (region, row as u64, (row * rows) as u64, row * rows * images)
        }

        /// Writes a pattern of `seed` over the whole level and slice, and returns it.
        fn write(&self, seed: u8) -> Vec<u8> {
            let (region, row, image, len) = self.layout();
            let bytes = pattern(len, seed);
            self.texture
                .replace_region(region, self.level, self.slice, &bytes, row, image)
                .expect("replace_region");
            bytes
        }

        fn read(&self) -> Vec<u8> {
            let (region, row, image, len) = self.layout();
            let mut bytes = vec![0; len];
            self.texture
                .get_bytes(&mut bytes, row, image, region, self.level, self.slice)
                .expect("get_bytes");
            bytes
        }

        /// The place of element `at` in Metal's terms.
        fn place(&self, at: [usize; 3]) -> TexturePlace {
            let b = self.block as usize;
            TexturePlace {
                slice: self.slice,
                level: self.level,
                origin: [(at[0] * b) as u32, (at[1] * b) as u32, at[2] as u32],
            }
        }
    }

    /// One copy of `size` elements, from `at` in `from`, whose bytes are `source`, to `to_at` in
    /// `to`, whose bytes are `before`: the bytes `to` must hold after it.
    struct Expect<'a> {
        from: &'a Level<'a>,
        source: &'a [u8],
        at: [usize; 3],
        to: &'a Level<'a>,
        before: Vec<u8>,
        to_at: [usize; 3],
        size: [usize; 3],
    }

    impl Expect<'_> {
        fn record(&self, name: &str, command_buffer: &CommandBuffer) {
            let mut encoder = command_buffer.blit_command_encoder().expect("an encoder");
            let b = self.from.block as usize;
            let texels = [self.size[0] * b, self.size[1] * b, self.size[2]].map(|n| n as u32);
            // A compressed level smaller than a block copies the texels it has.
            let extent = self.from.extent();
            let texels = std::array::from_fn(|axis| texels[axis].min(extent[axis]));
            encoder
                .copy_from_texture(
                    self.from.texture,
                    self.from.place(self.at),
                    texels,
                    self.to.texture,
                    self.to.place(self.to_at),
                )
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            encoder.end_encoding();
        }

        /// The destination's bytes after the copy.
        fn after(&self) -> Vec<u8> {
            let (from, to) = (self.from.grid(), self.to.grid());
            let unit = self.from.unit;
            let mut bytes = self.before.clone();
            for z in 0..self.size[2] {
                for y in 0..self.size[1] {
                    let element = |grid: [usize; 3], at: [usize; 3]| {
                        ((at[2] + z) * grid[1] + at[1] + y) * grid[0] + at[0]
                    };
                    let (s, d) = (
                        element(from, self.at) * unit,
                        element(to, self.to_at) * unit,
                    );
                    let len = self.size[0] * unit;
                    bytes[d..d + len].copy_from_slice(&self.source[s..s + len]);
                }
            }
            bytes
        }
    }

    fn texture(
        device: &Device,
        texture_type: TextureType,
        format: PixelFormat,
        extent: [u32; 3],
        usage: TextureUsage,
        index: u32,
    ) -> Result<Texture, Error> {
        let mut d = TextureDescriptor::new(texture_type, format, extent[0], extent[1]);
        d.depth = extent[2];
        d.usage = usage;
        d.storage_mode = StorageMode::Shared;
        if matches!(texture_type, TextureType::Type2DArray) {
            d.array_length = 3;
        }
        if texture_type == TextureType::Type2D && format == PixelFormat::RGBA8Unorm {
            d.mipmap_level_count = 3;
        }
        device.new_texture(&d, Some(index))
    }

    /// Bytes land where each copy names, everything around them kept; one copy alone costs one
    /// submission and three barriers and copies nothing of the runtime's own; an owed barrier is
    /// paid by the copy's first.
    #[test]
    fn every_device_copies_between_textures() {
        each_device("blit_textures", |name, _, device| {
            device
                .grant_indices(IndexClass::Texture, 4096, 16)
                .expect("a grant");
            device
                .grant_indices(IndexClass::StorageTexture, 1, 16)
                .expect("a grant");
            let read = TextureUsage::SHADER_READ;
            let write = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
            let make = |texture_type, format, extent, usage, index| {
                texture(&device, texture_type, format, extent, usage, index)
                    .unwrap_or_else(|error| panic!("{name}: {format:?}: {error}"))
            };
            let rgba = make(
                TextureType::Type2D,
                PixelFormat::RGBA8Unorm,
                [13, 7, 1],
                read,
                4096,
            );
            let words = make(
                TextureType::Type2DArray,
                PixelFormat::R32Uint,
                [8, 8, 1],
                write,
                1,
            );
            let cube = make(
                TextureType::Cube,
                PixelFormat::R8Unorm,
                [4, 4, 1],
                read,
                4097,
            );
            let volume = make(
                TextureType::Type3D,
                PixelFormat::R8Unorm,
                [4, 4, 3],
                read,
                4098,
            );
            let plane = make(
                TextureType::Type2D,
                PixelFormat::R8Unorm,
                [4, 4, 1],
                read,
                4099,
            );
            let level = |texture, slice, level, unit| Level {
                texture,
                slice,
                level,
                unit,
                block: 1,
            };
            let (rgba1, words2) = (level(&rgba, 0, 1, 4), level(&words, 2, 0, 4));
            let (words0, words1) = (level(&words, 0, 0, 4), level(&words, 1, 0, 4));
            let (face, image) = (level(&cube, 4, 0, 1), level(&volume, 0, 0, 1));
            let flat = level(&plane, 0, 0, 1);
            let sources = [rgba1.write(1), face.write(2), words0.write(3)];
            let befores = [
                words2.write(4),
                image.write(5),
                flat.write(6),
                words1.write(7),
            ];
            let queue = device.new_command_queue(4).expect("a queue");
            // The textures' uploads go with a command buffer of their own.
            let first = queue.command_buffer().expect("a command buffer");
            first.commit().expect("a commit");
            finished(name, &first);

            // A level of a 2D texture into a slice of an array of another format.
            let into_array = Expect {
                from: &rgba1,
                source: &sources[0],
                at: [1, 0, 0],
                to: &words2,
                before: befores[0].clone(),
                to_at: [2, 4, 0],
                size: [4, 3, 1],
            };
            let counters = device.counters();
            let before = counters.metrics();
            let command_buffer = queue.command_buffer().expect("a command buffer");
            into_array.record(name, &command_buffer);
            command_buffer.commit().expect("a commit");
            finished(name, &command_buffer);
            assert_eq!(command_buffer.error(), None, "{name}");
            let after = counters.metrics();
            assert_eq!(
                (
                    after.submissions - before.submissions,
                    after.barriers - before.barriers,
                    after.bytes_copied - before.bytes_copied,
                    after.passes - before.passes,
                ),
                (1, 3, 0, 1),
                "{name}: submissions, barriers, bytes copied, passes"
            );
            assert_eq!(words2.read(), into_array.after(), "{name}: into the array");

            // A cube's face into a 3D image, a 3D image into a 2D texture, and a slice of an
            // array into another of it, in one encoder; then a barrier an encoder owed, paid by
            // a copy's first.
            let into_volume = Expect {
                from: &face,
                source: &sources[1],
                at: [0, 0, 0],
                to: &image,
                before: befores[1].clone(),
                to_at: [0, 0, 1],
                size: [4, 4, 1],
            };
            let volume_after = into_volume.after();
            let into_plane = Expect {
                from: &image,
                source: &volume_after,
                at: [1, 1, 1],
                to: &flat,
                before: befores[2].clone(),
                to_at: [2, 0, 0],
                size: [2, 2, 1],
            };
            let across = Expect {
                from: &words0,
                source: &sources[2],
                at: [0, 0, 0],
                to: &words1,
                before: befores[3].clone(),
                to_at: [3, 3, 0],
                size: [5, 5, 1],
            };
            let command_buffer = queue.command_buffer().expect("a command buffer");
            for copy in [&into_volume, &into_plane, &across] {
                copy.record(name, &command_buffer);
            }
            command_buffer.commit().expect("a commit");
            finished(name, &command_buffer);
            assert_eq!(command_buffer.error(), None, "{name}");
            assert_eq!(image.read(), volume_after, "{name}: into the volume");
            assert_eq!(flat.read(), into_plane.after(), "{name}: into the plane");
            assert_eq!(words1.read(), across.after(), "{name}: across slices");

            let before = counters.metrics();
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut compute = command_buffer
                .compute_command_encoder(DispatchType::Serial)
                .expect("an encoder");
            compute.memory_barrier();
            assert!(matches!(
                command_buffer.blit_command_encoder(),
                Err(Error::Refused(Refusal::EncoderOpen))
            ));
            compute.end_encoding();
            into_array.record(name, &command_buffer);
            command_buffer.commit().expect("a commit");
            finished(name, &command_buffer);
            let after = counters.metrics();
            assert_eq!(
                after.barriers - before.barriers,
                3,
                "{name}: the owed barrier"
            );

            // A block-compressed region, where the device makes the format.
            let bc = |index| {
                texture(
                    &device,
                    TextureType::Type2D,
                    PixelFormat::BC1RGBA,
                    [8, 8, 1],
                    read,
                    index,
                )
            };
            match (bc(4100), bc(4101)) {
                (Ok(a), Ok(b)) => {
                    let (from, to) = (
                        Level {
                            texture: &a,
                            slice: 0,
                            level: 0,
                            unit: 8,
                            block: 4,
                        },
                        Level {
                            texture: &b,
                            slice: 0,
                            level: 0,
                            unit: 8,
                            block: 4,
                        },
                    );
                    let source = from.write(8);
                    let blocks = Expect {
                        from: &from,
                        source: &source,
                        at: [1, 0, 0],
                        to: &to,
                        before: to.write(9),
                        to_at: [0, 1, 0],
                        size: [1, 1, 1],
                    };
                    let command_buffer = queue.command_buffer().expect("a command buffer");
                    blocks.record(name, &command_buffer);
                    command_buffer.commit().expect("a commit");
                    finished(name, &command_buffer);
                    assert_eq!(to.read(), blocks.after(), "{name}: BC1 blocks");
                }
                (Err(Error::Unsupported(Gap::TextureFormat { .. })), _) => {
                    println!("{name}: no BC1 texture on this device")
                }
                (a, b) => panic!("{name}: BC1: {:?} {:?}", a.err(), b.err()),
            }

            // A slice past the array is refused, and the encoder goes on.
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = command_buffer.blit_command_encoder().expect("an encoder");
            let past = TexturePlace {
                slice: 3,
                level: 0,
                origin: [0; 3],
            };
            assert!(matches!(
                encoder.copy_from_texture(&words, past, [1, 1, 1], &plane, TexturePlace::default()),
                Err(Error::Refused(Refusal::TextureRegion {
                    level: 0,
                    slice: 3
                }))
            ));
            encoder.end_encoding();
            command_buffer.commit().expect("a commit");
            finished(name, &command_buffer);
            assert!(matches!(
                command_buffer.blit_command_encoder(),
                Err(Error::Refused(Refusal::Committed))
            ));
        });
    }
}
