//! Texture views, under a device cfg on every device of the cfg's kind: a view of every level, a
//! level subrange, a view of a view, a 2D view of one slice of a 2D array and a slice subrange of
//! the array each report Metal's extent, levels and slices, and a blit copy out of each reads the
//! parent's level and slice under the view's first, its bytes plain Metal's (probe
//! `texture_view_rules.m` in the conformance crate, an Apple M3 Max, macOS 27.0.1, job
//! m3-max-43677). A view costs no submission, no block and no copied byte. What Metal's validation
//! stops is `Refusal::TextureViewRange` or `Refusal::TextureViewFormat`, and each kind of view
//! the runtime does not build `Gap::TextureView` naming it. Every gauge comes back to zero with the
//! device, the views' image views among them. A view of a Managed texture writes the texture's own
//! named bytes.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::blit::on_device::settled;
    use crate::tests::command::on_device::{each_device, finished, within};
    use crate::{
        Buffer, BufferLayout, BytesPlace, CommandQueue, Device, Error, Gap, IndexClass,
        MemoryClass, PixelFormat, Refusal, Region, StorageMode, Texture, TextureDescriptor,
        TexturePlace, TextureSwizzle, TextureSwizzleChannels, TextureType, TextureUsage,
        TextureViewKind,
    };

    /// FNV-1a 64 of `bytes`.
    fn fnv(bytes: &[u8]) -> u64 {
        bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, &byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
        })
    }

    /// An 8x8 `RGBA8Unorm` Shared texture of `ShaderRead` usage at `index`: 2D of 3 levels, or a
    /// 2D array of 3 slices and 2 levels, each level and slice holding the probe's bytes,
    /// `i * 31 + 21 + 7 * level + 50 * slice`.
    fn parent(device: &Device, array: bool, index: u32) -> Texture {
        let texture_type = match array {
            true => TextureType::Type2DArray,
            false => TextureType::Type2D,
        };
        let mut d = TextureDescriptor::new(texture_type, PixelFormat::RGBA8Unorm, 8, 8);
        d.array_length = if array { 3 } else { 1 };
        d.mipmap_level_count = if array { 2 } else { 3 };
        d.usage = TextureUsage::SHADER_READ;
        d.storage_mode = StorageMode::Shared;
        let texture = device.new_texture(&d, Some(index)).expect("a texture");
        for level in 0..d.mipmap_level_count {
            for slice in 0..d.array_length {
                let side = 8 >> level;
                let bytes: Vec<u8> = (0..side * side * 4)
                    .map(|i| (i * 31 + 21 + 7 * level + 50 * slice) as u8)
                    .collect();
                let region = Region {
                    origin: [0; 3],
                    size: [side, side, 1],
                };
                let (row, image) = (u64::from(side) * 4, u64::from(side * side) * 4);
                (texture.replace_region(region, level, slice, &bytes, row, image))
                    .expect("the bytes");
            }
        }
        texture
    }

    /// The bytes of level 0 and slice 0 of `view`, copied by a blit into `out`, a Shared buffer
    /// long enough, rows tight.
    fn bytes(queue: &CommandQueue, view: &Texture, out: &Buffer) -> Vec<u8> {
        let d = view.descriptor();
        let len = u64::from(d.width * d.height) * 4;
        let command_buffer = queue.command_buffer().expect("a command buffer");
        let mut encoder = command_buffer.blit_command_encoder().expect("an encoder");
        let layout = BufferLayout {
            offset: 0,
            bytes_per_row: u64::from(d.width) * 4,
            bytes_per_image: 0,
        };
        let size = [d.width, d.height, 1];
        (encoder.copy_from_texture_to_buffer(view, TexturePlace::default(), size, out, layout))
            .expect("a copy");
        encoder.end_encoding();
        command_buffer.commit().expect("a commit");
        finished("the copy", &command_buffer);
        assert_eq!(command_buffer.error(), None);
        let contents = out.contents().expect("a Shared buffer");
        // SAFETY: the copy into the buffer completed, and its writes reach the host.
        unsafe { std::slice::from_raw_parts(contents.as_ptr(), len as usize) }.to_vec()
    }

    #[test]
    fn every_device_views_levels_and_slices_of_a_texture() {
        each_device("texture_views", |name, _, device| {
            device
                .grant_indices(IndexClass::Texture, 4096, 32)
                .expect("a grant");
            let flat = parent(&device, false, 4096);
            let array = parent(&device, true, 4097);
            let block = (device.take_block(MemoryClass::Shared, 256)).expect("a Shared block");
            let out = (device.new_buffer(block.id, 0, 256, Some(0))).expect("a buffer");
            let queue = device.new_command_queue(1).expect("a queue");
            let read = |view: &Texture| bytes(&queue, view, &out);
            use TextureType::{Type2D, Type2DArray};
            // The probe's cases: the parent, the view's type, levels and slices; what the view
            // reports (width, levels, array length); its level 0's bytes' FNV-1a 64.
            let cases = [
                (
                    "whole",
                    &flat,
                    Type2D,
                    0..3,
                    0..1,
                    [8, 3, 1],
                    0x1bad_e16c_f57f_1725,
                ),
                (
                    "levels",
                    &flat,
                    Type2D,
                    1..3,
                    0..1,
                    [4, 2, 1],
                    0x88f8_2fdd_3017_a3a5,
                ),
                (
                    "level-one",
                    &flat,
                    Type2D,
                    2..3,
                    0..1,
                    [2, 1, 1],
                    0x5d01_ec06_0898_5ec5,
                ),
                (
                    "array-2d",
                    &array,
                    Type2D,
                    0..2,
                    0..1,
                    [8, 2, 1],
                    0x1bad_e16c_f57f_1725,
                ),
                (
                    "array-2d-last",
                    &array,
                    Type2D,
                    1..2,
                    2..3,
                    [4, 1, 1],
                    0xee6d_72d7_23b9_c425,
                ),
                (
                    "array-slices",
                    &array,
                    Type2DArray,
                    0..2,
                    1..3,
                    [8, 2, 2],
                    0xe16c_7fad_7cbe_2725,
                ),
            ];
            let counters = device.counters();
            let mut views = Vec::new();
            for (at, (case, parent, texture_type, levels, slices, reports, hash)) in
                (4100..).zip(cases)
            {
                settled(&device);
                let before = counters.metrics();
                let view = (parent.new_texture_view(
                    PixelFormat::RGBA8Unorm,
                    texture_type,
                    levels,
                    slices,
                    Some(at),
                ))
                .unwrap_or_else(|error| panic!("{name}: {case}: {error}"));
                let after = counters.metrics();
                assert_eq!(
                    (
                        after.submissions - before.submissions,
                        after.blocks_made - before.blocks_made,
                        after.bytes_copied - before.bytes_copied,
                    ),
                    (0, 0, 0),
                    "{name}: {case}: a view's submissions, blocks and bytes copied"
                );
                let d = view.descriptor();
                assert_eq!(
                    (
                        d.texture_type,
                        [d.width, d.mipmap_level_count, d.array_length]
                    ),
                    (texture_type, reports),
                    "{name}: {case}: what the view reports"
                );
                assert_eq!(d.height, d.width, "{name}: {case}");
                assert_eq!(fnv(&read(&view)), hash, "{name}: {case}: the bytes");
                views.push(view);
            }

            // A view of the levels view: the parent's level 2, as level-one reads it.
            let of_view = (views[1].new_texture_view(
                PixelFormat::RGBA8Unorm,
                Type2D,
                1..2,
                0..1,
                Some(4110),
            ))
            .unwrap_or_else(|error| panic!("{name}: a view of a view: {error}"));
            assert_eq!(
                fnv(&read(&of_view)),
                0x5d01_ec06_0898_5ec5,
                "{name}: a view of a view"
            );

            // What Metal's validation stops, and the kinds not built.
            let view = |parent: &Texture, format, texture_type, levels, slices| {
                parent.new_texture_view(format, texture_type, levels, slices, Some(4111))
            };
            let rgba = PixelFormat::RGBA8Unorm;
            for (case, result) in [
                ("levels-past", view(&flat, rgba, Type2D, 2..4, 0..1)),
                ("levels-empty", view(&flat, rgba, Type2D, 1..1, 0..1)),
                ("slices-past", view(&array, rgba, Type2D, 0..1, 3..4)),
            ] {
                assert!(
                    matches!(
                        result,
                        Err(Error::Refused(Refusal::TextureViewRange { .. }))
                    ),
                    "{name}: {case}"
                );
            }
            assert!(
                matches!(
                    view(&flat, PixelFormat::R32Uint, Type2D, 0..3, 0..1),
                    Err(Error::Refused(Refusal::TextureViewFormat { .. }))
                ),
                "{name}: another format without PixelFormatView usage"
            );
            let mut d = TextureDescriptor::new(Type2D, rgba, 8, 8);
            d.usage = TextureUsage::SHADER_READ | TextureUsage::PIXEL_FORMAT_VIEW;
            let formats = device.new_texture(&d, Some(4112)).expect("a texture");
            d.usage = TextureUsage::SHADER_READ;
            d.swizzle = TextureSwizzleChannels {
                red: TextureSwizzle::Blue,
                blue: TextureSwizzle::Red,
                ..TextureSwizzleChannels::DEFAULT
            };
            let swizzled = device.new_texture(&d, Some(4113)).expect("a texture");
            let mut d = TextureDescriptor::new(TextureType::Cube, rgba, 4, 4);
            d.usage = TextureUsage::SHADER_READ;
            let cube = device.new_texture(&d, Some(4114)).expect("a texture");
            for (case, result, kind) in [
                (
                    "another format",
                    view(&formats, PixelFormat::R32Uint, Type2D, 0..1, 0..1),
                    TextureViewKind::PixelFormat,
                ),
                (
                    "a swizzled texture",
                    view(&swizzled, rgba, Type2D, 0..1, 0..1),
                    TextureViewKind::Swizzle,
                ),
                (
                    "a 2D array of a 2D texture",
                    view(&flat, rgba, Type2DArray, 0..3, 0..1),
                    TextureViewKind::TextureType,
                ),
                (
                    "a 2D view of two slices",
                    view(&array, rgba, Type2D, 0..1, 0..2),
                    TextureViewKind::TextureType,
                ),
                (
                    "three faces of a cube",
                    view(&cube, rgba, TextureType::Cube, 0..1, 0..3),
                    TextureViewKind::Slices,
                ),
            ] {
                match result {
                    Err(Error::Unsupported(Gap::TextureView { kind: got })) if got == kind => {}
                    Err(error) => panic!("{name}: {case}: {error}"),
                    Ok(_) => panic!("{name}: {case}: made"),
                }
            }

            // The views outlive their parents' handles, and every gauge falls with the last.
            drop((flat, array));
            assert_eq!(fnv(&read(&of_view)), 0x5d01_ec06_0898_5ec5);
            drop((views, of_view, formats, swizzled, cube));
            within("the textures freed", || counters.gauges().textures == 0);
        });
    }

    #[test]
    fn every_device_views_a_managed_texture_s_named_bytes() {
        each_device("managed texture views", |name, _, device| {
            let block = (device.take_block(MemoryClass::Shared, 1)).expect("a Shared block");
            let named = (device.new_buffer(block.id, 0, 4096, Some(0))).expect("a buffer");
            // A 4x4 2D array of 2 slices and 2 levels, each level and slice's rows 64 bytes
            // apart, 512 bytes after the one before.
            let places: Vec<BytesPlace> = (0..4)
                .map(|at| BytesPlace {
                    offset: at * 512,
                    bytes_per_row: 64,
                    bytes_per_image: 0,
                })
                .collect();
            let mut d =
                TextureDescriptor::new(TextureType::Type2DArray, PixelFormat::RGBA8Unorm, 4, 4);
            d.array_length = 2;
            d.mipmap_level_count = 2;
            d.usage = TextureUsage::RENDER_TARGET;
            let texture = (device.new_managed_texture(&d, None, &named, &places))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let view = (texture.new_texture_view(
                PixelFormat::RGBA8Unorm,
                TextureType::Type2D,
                1..2,
                1..2,
                None,
            ))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
            let region = Region {
                origin: [0; 3],
                size: [2, 2, 1],
            };
            let written: Vec<u8> = (0..16).map(|i| i * 13 + 5).collect();
            (view.replace_region(region, 0, 0, &written, 8, 0))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            // The texture's level 1 and slice 1 reads what the view wrote, from the same bytes.
            let mut read = vec![0; 16];
            (texture.get_bytes(&mut read, 8, 0, region, 1, 1))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            assert_eq!(read, written, "{name}: the texture reads the view's write");
            let contents = named.contents().expect("a Shared buffer");
            // SAFETY: no GPU work writes the buffer.
            let row = unsafe { std::slice::from_raw_parts(contents.as_ptr().add(3 * 512), 8) };
            assert_eq!(
                row,
                &written[..8],
                "{name}: the write lies in the place of level 1, slice 1"
            );
        });
    }
}
