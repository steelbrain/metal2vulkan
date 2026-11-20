//! Texture views, under a device cfg on every device of the cfg's kind: a view of every level, a
//! level subrange, a view of a view, a 2D view of one slice of a 2D array and a slice subrange of
//! the array each report Metal's extent, levels and slices, and a blit copy out of each reads the
//! parent's level and slice under the view's first, its bytes plain Metal's (probe
//! `texture_view_rules.m` in the conformance crate, an Apple M3 Max, macOS 27.0.1, job
//! m3-max-43677). A 2D texture as a 2D array, and a cube's faces as Metal's casts view them (a 2D
//! array of some faces or of all six, a 2D texture of one), each report Metal's type, extent,
//! levels and array length, and a blit copy out of each reads the parent's level and slice under
//! its first. A view costs no
//! submission, no block and no copied byte. What Metal's validation stops is
//! `Refusal::TextureViewRange`, `Refusal::TextureViewSlices` or `Refusal::TextureViewFormat`, and
//! each kind of view the runtime does not build `Gap::TextureView` naming it. A view with a swizzle holds Metal's
//! combination of it over its parent's, in its descriptor and its resource ID's word, over a
//! texture of any usage. A view of another pixel format Metal views the texture as (its sRGB
//! sibling, or with `PixelFormatView` usage a same-size colour format) reads its parent's bytes,
//! and one of a format Metal does not view it as is `Refusal::TextureViewFormat`. Without that
//! usage, a view the image's own Vulkan format holds is made as plain Metal makes it (an `A8Unorm`
//! view of an `R8Unorm` texture, which a draw samples as its parent's red byte in alpha), and one
//! that needs another Vulkan format is `Refusal::TextureViewFormat`; a texture of
//! Metal's unknown usage (0) takes every usage, `PixelFormatView`'s views among them. Every gauge
//! comes back to zero with the device, the views' image views among them. A view of a Managed
//! texture writes the texture's own named bytes. A command buffer keeps a view it uses after its
//! parent, as Metal's retains every object its encoders use: released while the command buffer
//! is pending, the view keeps its image views and its element of set 1 until it retires. A texture
//! or view of a block-compressed pixel format carries the static note of its decode
//! (`PrecisionNote::CompressedTexelDecode`), one of another format none.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use std::thread;
    use std::time::Duration;

    use crate::tests::blit::on_device::settled;
    use crate::tests::command::on_device::{each_device, finished, shared_event, within};
    use crate::tests::draw::on_device::{
        descriptor, examples, pass, read, run, target, vertices, Buffers, CONSTEXPR, SIDE,
    };
    use crate::{
        Buffer, BufferLayout, BytesPlace, CommandQueue, Device, Error, Gap, IndexClass,
        MemoryClass, PixelFormat, PrecisionNote, PrimitiveType, Refusal, Region, StorageMode,
        Texture, TextureDescriptor, TexturePlace, TextureSwizzle, TextureSwizzleChannels,
        TextureType, TextureUsage, TextureViewKind,
    };
    use metal2vulkan::metal::resource_word::TextureWord;

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
                (texture.replace_region(
                    region,
                    level,
                    slice,
                    crate::HostBytes::from(&bytes[..]),
                    row,
                    image,
                ))
                .expect("the bytes");
            }
        }
        texture
    }

    /// The bytes of level 0 and slice 0 of `view`, of `texel` bytes a texel, copied by a blit
    /// into `out`, a Shared buffer long enough, rows tight.
    fn bytes(queue: &CommandQueue, view: &Texture, out: &Buffer, texel: u32) -> Vec<u8> {
        let d = view.descriptor();
        let len = u64::from(d.width * d.height * texel);
        let command_buffer = queue.command_buffer().expect("a command buffer");
        let mut encoder = command_buffer.blit_command_encoder().expect("an encoder");
        let layout = BufferLayout {
            offset: 0,
            bytes_per_row: u64::from(d.width * texel),
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

    /// The bytes of level `level` and face `face` of the `RGBA8Unorm` cube of [`cube_parent`].
    fn cube_bytes(level: u32, face: u32) -> Vec<u8> {
        let side = 8 >> level;
        (0..side * side * 4)
            .map(|i| (i * 13 + 5 + 11 * level + 29 * face) as u8)
            .collect()
    }

    /// An 8x8 `RGBA8Unorm` Shared cube of 2 levels and `ShaderRead` usage at `index`, each level
    /// and face holding [`cube_bytes`].
    fn cube_parent(device: &Device, index: u32) -> Texture {
        let mut d = TextureDescriptor::new(TextureType::Cube, PixelFormat::RGBA8Unorm, 8, 8);
        d.mipmap_level_count = 2;
        d.usage = TextureUsage::SHADER_READ;
        d.storage_mode = StorageMode::Shared;
        let texture = device.new_texture(&d, Some(index)).expect("a texture");
        for level in 0..2 {
            for face in 0..6 {
                let side = 8 >> level;
                let region = Region {
                    origin: [0; 3],
                    size: [side, side, 1],
                };
                let (row, image) = (u64::from(side) * 4, u64::from(side * side) * 4);
                let bytes = cube_bytes(level, face);
                (texture.replace_region(
                    region,
                    level,
                    face,
                    crate::HostBytes::from(&bytes[..]),
                    row,
                    image,
                ))
                .expect("the bytes");
            }
        }
        texture
    }

    /// Metal's casts of a cube over the same image (Apple's
    /// `makeTextureView(pixelFormat:textureType:levels:slices:)`), each over a level and face
    /// range inside the cube: a 2D array of some faces, of every level and face, and a 2D texture
    /// of one face. Each reports Metal's type, extent, levels and array length, and a blit copy
    /// out of its level 0 and slice 0 reads the cube's level and face under them. A view costs no
    /// submission, block or copied byte. A view as a cube array is `Gap::CubeArrayTexture`, as a
    /// cube array texture is.
    #[test]
    fn every_device_views_a_cube_s_faces_as_metal_casts_them() {
        each_device("texture_views_of_cubes", |name, _, device| {
            device
                .grant_indices(IndexClass::Texture, 4096, 16)
                .expect("a grant");
            let cube = cube_parent(&device, 4096);
            let block = (device.take_block(MemoryClass::Shared, 256)).expect("a Shared block");
            let out = (device.new_buffer(block.id, 0, 256, Some(0))).expect("a buffer");
            let queue = device.new_command_queue(1).expect("a queue");
            use TextureType::{CubeArray, Type2D, Type2DArray};
            // The view's type, levels and faces; what the view reports (width, levels, array
            // length); the cube's level and face its level 0 and slice 0 are.
            let cases = [
                ("faces", Type2DArray, 1..2, 2..5, [4, 1, 3], [1, 2]),
                ("every-face", Type2DArray, 0..2, 0..6, [8, 2, 6], [0, 0]),
                ("face", Type2D, 1..2, 4..5, [4, 1, 1], [1, 4]),
            ];
            let counters = device.counters();
            for (at, (case, texture_type, levels, faces, reports, under)) in (4100..).zip(cases) {
                settled(&device);
                let before = counters.metrics();
                let view = (cube.new_texture_view(
                    PixelFormat::RGBA8Unorm,
                    texture_type,
                    levels,
                    faces,
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
                let [level, face] = under;
                assert_eq!(
                    bytes(&queue, &view, &out, 4),
                    cube_bytes(level, face),
                    "{name}: {case}: the bytes"
                );
            }
            let as_array =
                cube.new_texture_view(PixelFormat::RGBA8Unorm, CubeArray, 0..1, 0..6, Some(4110));
            assert!(
                matches!(as_array, Err(Error::Unsupported(Gap::CubeArrayTexture))),
                "{name}: a cube array view of a cube"
            );
        });
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
            let read = |view: &Texture| bytes(&queue, view, &out, 4);
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
                (
                    "2d-as-array",
                    &flat,
                    Type2DArray,
                    0..3,
                    0..1,
                    [8, 3, 1],
                    0x1bad_e16c_f57f_1725,
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
            // Pixel-format views, each of its parent's bytes: the sRGB sibling of a texture made
            // without `PixelFormatView` usage, and an `R32Uint` view of one made with it; a format
            // of another size, or a compressed one, Metal refuses even so.
            let mut d = TextureDescriptor::new(Type2D, rgba, 8, 8);
            d.usage = TextureUsage::SHADER_READ | TextureUsage::PIXEL_FORMAT_VIEW;
            d.storage_mode = StorageMode::Shared;
            let formats = device.new_texture(&d, Some(4112)).expect("a texture");
            let whole = Region {
                origin: [0; 3],
                size: [8, 8, 1],
            };
            let texels: Vec<u8> = (0..256).map(|i| (i * 7 + 3) as u8).collect();
            (formats.replace_region(whole, 0, 0, crate::HostBytes::from(&texels[..]), 32, 256))
                .expect("the bytes");
            for (case, parent, format) in [
                ("an sRGB sibling", &flat, PixelFormat::RGBA8UnormSRGB),
                ("a same-size format", &formats, PixelFormat::R32Uint),
            ] {
                let levels = 0..parent.descriptor().mipmap_level_count;
                let made = (view(parent, format, Type2D, levels, 0..1))
                    .unwrap_or_else(|error| panic!("{name}: {case}: {error}"));
                assert_eq!(made.descriptor().pixel_format, format, "{name}: {case}");
                assert_eq!(read(&made), read(parent), "{name}: {case}: the bytes");
            }
            for format in [PixelFormat::R16Float, PixelFormat::BC1RGBA] {
                assert!(
                    matches!(
                        view(&formats, format, Type2D, 0..1, 0..1),
                        Err(Error::Refused(Refusal::TextureViewFormat { .. }))
                    ),
                    "{name}: {format:?} of an RGBA8Unorm texture"
                );
            }
            let mut d = TextureDescriptor::new(TextureType::Cube, rgba, 4, 4);
            d.usage = TextureUsage::SHADER_READ;
            let cube = device.new_texture(&d, Some(4114)).expect("a texture");
            // A non-square 2D array, whose image is not cube-compatible (a square one's cube view
            // is `tests::cube_views`').
            let mut d = TextureDescriptor::new(Type2DArray, rgba, 4, 2);
            d.array_length = 6;
            d.usage = TextureUsage::SHADER_READ;
            let six = device.new_texture(&d, Some(4118)).expect("a texture");
            // A slice count the view's type does not take, as Metal's validation stops it.
            for (case, result) in [
                (
                    "a 2D view of two slices",
                    view(&array, rgba, Type2D, 0..1, 0..2),
                ),
                (
                    "three faces of a cube",
                    view(&cube, rgba, TextureType::Cube, 0..1, 0..3),
                ),
            ] {
                assert!(
                    matches!(
                        result,
                        Err(Error::Refused(Refusal::TextureViewSlices { .. }))
                    ),
                    "{name}: {case}"
                );
            }
            for (case, result, kind) in [
                (
                    "a cube of a non-square 2D array",
                    view(&six, rgba, TextureType::Cube, 0..1, 0..6),
                    TextureViewKind::CubeOfArray,
                ),
                (
                    "a 3D view of a cube",
                    view(&cube, rgba, TextureType::Type3D, 0..1, 0..1),
                    TextureViewKind::TextureType,
                ),
            ] {
                match result {
                    Err(Error::Unsupported(Gap::TextureView { kind: got })) if got == kind => {}
                    Err(error) => panic!("{name}: {case}: {error}"),
                    Ok(_) => panic!("{name}: {case}: made"),
                }
            }

            // A view's swizzle over its parent's is Metal's combination in its resource ID's word,
            // and its descriptor holds the swizzle it was made with, as Metal's getter answers; a
            // view without one keeps the parent's in both; and a texture of shader-write and
            // render-target usage, whose own descriptor Metal refuses a swizzle, takes a swizzled
            // view.
            use TextureSwizzle::{Alpha, Blue, Green, One, Red, Zero};
            let channels =
                |[red, green, blue, alpha]: [TextureSwizzle; 4]| TextureSwizzleChannels {
                    red,
                    green,
                    blue,
                    alpha,
                };
            let mut d = TextureDescriptor::new(Type2D, rgba, 8, 8);
            d.usage = TextureUsage::SHADER_READ;
            d.swizzle = channels([Red, One, One, Green]);
            let apple = device.new_texture(&d, Some(4115)).expect("a texture");
            d.usage = TextureUsage::SHADER_READ
                | TextureUsage::SHADER_WRITE
                | TextureUsage::RENDER_TARGET;
            d.swizzle = TextureSwizzleChannels::DEFAULT;
            let written = device.new_texture(&d, Some(4116)).expect("a texture");
            let swizzled_view = |parent: &Texture, swizzle| {
                parent.new_texture_view_with_swizzle(rgba, Type2D, 0..1, 0..1, swizzle, Some(4117))
            };
            for (case, made, answered, reads) in [
                (
                    "Apple's example",
                    swizzled_view(&apple, channels([Red, Green, Alpha, Blue])),
                    [Red, Green, Alpha, Blue],
                    [Red, One, Green, One],
                ),
                (
                    "a view without a swizzle",
                    view(&apple, rgba, Type2D, 0..1, 0..1),
                    [Red, One, One, Green],
                    [Red, One, One, Green],
                ),
                (
                    "a written texture's",
                    swizzled_view(&written, channels([Blue, Zero, Red, One])),
                    [Blue, Zero, Red, One],
                    [Blue, Zero, Red, One],
                ),
            ] {
                let made = made.unwrap_or_else(|error| panic!("{name}: {case}: {error}"));
                let answered = channels(answered);
                assert_eq!(made.descriptor().swizzle, answered, "{name}: {case}");
                let word = (TextureWord::of(rgba, channels(reads))).expect("a word").0;
                assert_eq!(
                    made.gpu_resource_id().word,
                    word,
                    "{name}: {case}: the word"
                );
            }

            // The views outlive their parents' handles, and every gauge falls with the last.
            drop((flat, array));
            assert_eq!(fnv(&read(&of_view)), 0x5d01_ec06_0898_5ec5);
            drop((views, of_view, formats, cube, six, apple, written));
            within("the textures freed", || counters.gauges().textures == 0);
        });
    }

    /// A texture of Metal's unknown usage (0) takes every usage, `PixelFormatView`'s among them,
    /// so Metal views it as any colour format of its texel's bytes: an `R8Unorm` and an `RG8Unorm`
    /// texture of that usage, made with an index ([`Device::new_texture`]) and for views alone
    /// ([`Device::new_texture_for_views`]), each viewed as the integer format of its size
    /// (`R8Uint`, `RG8Uint`). Bytes written through the texture read back through the view, and
    /// bytes written through the view read back through the texture.
    #[test]
    fn every_device_views_a_texture_of_unknown_usage_as_a_same_size_format() {
        each_device("unknown_usage_views", |name, _, device| {
            device
                .grant_indices(IndexClass::Texture, 4096, 32)
                .expect("a grant");
            let block = (device.take_block(MemoryClass::Shared, 256)).expect("a Shared block");
            let out = (device.new_buffer(block.id, 0, 256, Some(0))).expect("a buffer");
            let queue = device.new_command_queue(1).expect("a queue");
            let mut indices = 4096..4128;
            let whole = Region {
                origin: [0; 3],
                size: [8, 8, 1],
            };
            for (format, view_format, texel) in [
                (PixelFormat::R8Unorm, PixelFormat::R8Uint, 1u32),
                (PixelFormat::RG8Unorm, PixelFormat::RG8Uint, 2),
            ] {
                for for_views in [false, true] {
                    let case = format!("{format:?} as {view_format:?} (for views: {for_views})");
                    let mut d = TextureDescriptor::new(TextureType::Type2D, format, 8, 8);
                    d.usage = TextureUsage::empty();
                    d.storage_mode = StorageMode::Shared;
                    let texture = match for_views {
                        true => device.new_texture_for_views(&d),
                        false => device.new_texture(&d, indices.next()),
                    };
                    let texture = texture.unwrap_or_else(|error| panic!("{name}: {case}: {error}"));
                    let view = (texture.new_texture_view(
                        view_format,
                        TextureType::Type2D,
                        0..1,
                        0..1,
                        indices.next(),
                    ))
                    .unwrap_or_else(|error| panic!("{name}: {case}: the view: {error}"));
                    assert_eq!(
                        view.descriptor().pixel_format,
                        view_format,
                        "{name}: {case}"
                    );
                    let row = u64::from(8 * texel);
                    for (seed, written_through, read_through) in
                        [(11u32, &texture, &view), (5, &view, &texture)]
                    {
                        let written: Vec<u8> =
                            (0..64 * texel).map(|i| (i * 37 + seed) as u8).collect();
                        let host = crate::HostBytes::from(&written[..]);
                        (written_through.replace_region(whole, 0, 0, host, row, 0))
                            .unwrap_or_else(|error| panic!("{name}: {case}: {error}"));
                        assert_eq!(
                            bytes(&queue, read_through, &out, texel),
                            written,
                            "{name}: {case}: read through the other"
                        );
                    }
                }
            }
        });
    }

    /// An `A8Unorm` view of an `R8Unorm` texture of `ShaderRead` usage alone, which Metal's
    /// validation stops and plain Metal makes (probe `texture_view_rules.m`, case `r8-a8`, an
    /// Apple M3 Max under macOS 27.0.1, job m3-max-59523), is made in the image's own `R8_UNORM`:
    /// each pixel of a draw sampling it reads its parent's red byte as alpha, `(0, 0, 0, r)`. An
    /// `R8Uint` view of the same texture, which another Vulkan format holds, stays
    /// `Refusal::TextureViewFormat`.
    #[test]
    fn every_device_samples_an_a8_view_of_an_r8_texture_as_its_red_byte_in_alpha() {
        let Some(library) = examples() else {
            return;
        };
        let d = descriptor(&library, "sampled_fragment", PixelFormat::RGBA8Unorm);
        each_device("a8_view_of_r8", |name, _, device| {
            for (class, first) in [(IndexClass::Texture, 4096), (IndexClass::Sampler, 1)] {
                (device.grant_indices(class, first, 4))
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
            }
            let state = (device.new_render_pipeline_state(&d, |_| Ok(vec![CONSTEXPR])))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let mut sd =
                TextureDescriptor::new(TextureType::Type2D, PixelFormat::R8Unorm, SIDE, SIDE);
            sd.usage = TextureUsage::SHADER_READ;
            sd.storage_mode = StorageMode::Shared;
            let parent = (device.new_texture(&sd, Some(4096)))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let red: Vec<u8> = (0..SIDE * SIDE)
                .map(|i| match i {
                    0 => 0,
                    i if i == SIDE * SIDE - 1 => 255,
                    i => (i * 37 + 11) as u8,
                })
                .collect();
            let whole = Region {
                origin: [0; 3],
                size: [SIDE, SIDE, 1],
            };
            let row = u64::from(SIDE);
            (parent.replace_region(whole, 0, 0, crate::HostBytes::from(&red[..]), row, 0))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let view = |format| {
                parent.new_texture_view(format, TextureType::Type2D, 0..1, 0..1, Some(4097))
            };
            assert!(
                matches!(
                    view(PixelFormat::R8Uint),
                    Err(Error::Refused(Refusal::TextureViewFormat { .. }))
                ),
                "{name}: an R8Uint view without PixelFormatView usage"
            );
            let alpha = (view(PixelFormat::A8Unorm))
                .unwrap_or_else(|error| panic!("{name}: the A8Unorm view: {error}"));
            assert_eq!(
                alpha.descriptor().pixel_format,
                PixelFormat::A8Unorm,
                "{name}"
            );
            let mut buffers = Buffers::new(&device);
            let (positions, colors) = vertices(&device, &mut buffers);
            let color = target(
                &device,
                PixelFormat::RGBA8Unorm,
                TextureUsage::RENDER_TARGET,
            );
            let queue = device.new_command_queue(4).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = (command_buffer.render_command_encoder(&pass(&color, None)))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let draw = |result: Result<(), Error>| result.unwrap_or_else(|e| panic!("{name}: {e}"));
            draw(encoder.set_render_pipeline_state(&state));
            draw(encoder.set_vertex_buffer(Some(&positions), 0, 0));
            draw(encoder.set_vertex_buffer(Some(&colors), 0, 1));
            draw(encoder.set_fragment_texture(Some(&alpha), 0));
            draw(encoder.draw_primitives(PrimitiveType::Triangle, 3, 3, 1, 0));
            encoder.end_encoding();
            run(name, &command_buffer);
            let expected: Vec<u8> = red.iter().flat_map(|&r| [0, 0, 0, r]).collect();
            assert_eq!(read(&color, 4), expected, "{name}: each pixel (0, 0, 0, r)");
            println!("{name}: an A8Unorm view of an R8Unorm texture read its red bytes as alpha");
        });
    }

    /// A texture of a block-compressed pixel format answers the static note of its decode, and so
    /// does its view in its sRGB sibling, decided by the format alone; an uncompressed texture and
    /// its view answer none. Each compressed format counts where the device makes it, and each
    /// device makes at least one.
    #[test]
    fn every_device_notes_the_decode_of_a_block_compressed_texture_and_its_views() {
        each_device("compressed_decode_note", |name, _, device| {
            let _grant = (device.grant_indices(IndexClass::Texture, 4096, 32)).expect("a grant");
            let next = std::cell::Cell::new(4096);
            let index = || {
                next.set(next.get() + 1);
                Some(next.get() - 1)
            };
            let note = Some(PrecisionNote::CompressedTexelDecode);
            let made = |pixel_format| {
                let mut d = TextureDescriptor::new(TextureType::Type2D, pixel_format, 8, 8);
                d.usage = TextureUsage::SHADER_READ;
                device.new_texture(&d, index())
            };
            let view = |texture: &Texture, pixel_format| {
                (texture.new_texture_view(pixel_format, TextureType::Type2D, 0..1, 0..1, index()))
                    .unwrap_or_else(|error| panic!("{name}: a {pixel_format:?} view: {error}"))
            };
            let compressed = [
                (PixelFormat::BC1RGBA, PixelFormat::BC1RGBASRGB),
                (PixelFormat::BC7RGBAUnorm, PixelFormat::BC7RGBAUnormSRGB),
                (PixelFormat::ETC2RGB8, PixelFormat::ETC2RGB8SRGB),
                (PixelFormat::ASTC4x4LDR, PixelFormat::ASTC4x4SRGB),
            ];
            let mut noted = Vec::new();
            for (pixel_format, srgb) in compressed {
                let texture = match made(pixel_format) {
                    Ok(texture) => texture,
                    Err(Error::Unsupported(Gap::TextureFormat { .. })) => continue,
                    Err(error) => panic!("{name}: a {pixel_format:?} texture: {error}"),
                };
                assert_eq!(texture.precision_note(), note, "{name}: {pixel_format:?}");
                assert_eq!(
                    view(&texture, srgb).precision_note(),
                    note,
                    "{name}: {srgb:?}"
                );
                noted.push(pixel_format);
            }
            assert!(
                !noted.is_empty(),
                "{name}: no block-compressed texture made"
            );
            eprintln!("{name}: the decode note on {noted:?} and their sRGB views");
            let plain = (made(PixelFormat::RGBA8Unorm))
                .unwrap_or_else(|error| panic!("{name}: an RGBA8Unorm texture: {error}"));
            assert_eq!(plain.precision_note(), None, "{name}: RGBA8Unorm");
            let plain_view = view(&plain, PixelFormat::RGBA8UnormSRGB);
            assert_eq!(plain_view.precision_note(), None, "{name}: RGBA8UnormSRGB");
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
            // A colour render target takes an index, and so does its view.
            (device.grant_indices(IndexClass::Texture, 4096, 2)).expect("a grant");
            let texture = (device.new_managed_texture(&d, Some(4096), &named, &places))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let view = (texture.new_texture_view(
                PixelFormat::RGBA8Unorm,
                TextureType::Type2D,
                1..2,
                1..2,
                Some(4097),
            ))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
            let region = Region {
                origin: [0; 3],
                size: [2, 2, 1],
            };
            let written: Vec<u8> = (0..16).map(|i| i * 13 + 5).collect();
            (view.replace_region(region, 0, 0, crate::HostBytes::from(&written[..]), 8, 0))
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

    #[test]
    fn every_device_keeps_a_view_bound_after_its_parent_until_its_command_buffer_retires() {
        each_device("view_retention", |name, _, device| {
            device
                .grant_indices(IndexClass::Texture, 4096, 32)
                .expect("a grant");
            let flat = parent(&device, false, 4096);
            let view = (flat.new_texture_view(
                PixelFormat::RGBA8Unorm,
                TextureType::Type2D,
                1..3,
                0..1,
                Some(4100),
            ))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
            let block = (device.take_block(MemoryClass::Shared, 256)).expect("a Shared block");
            let out = (device.new_buffer(block.id, 0, 256, Some(0))).expect("a buffer");
            let queue = device.new_command_queue(1).expect("a queue");
            let f = shared_event(&device);
            let held = queue.command_buffer().expect("a command buffer");
            let mut encoder = held.blit_command_encoder().expect("an encoder");
            let layout = BufferLayout {
                offset: 0,
                bytes_per_row: 16,
                bytes_per_image: 0,
            };
            // The parent first, then its view, which shares its hazard id.
            for texture in [&flat, &view] {
                (encoder.copy_from_texture_to_buffer(
                    texture,
                    TexturePlace::default(),
                    [4, 4, 1],
                    &out,
                    layout,
                ))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            }
            encoder.end_encoding();
            held.encode_wait_for_event(f.as_event(), 1).expect("a wait");
            held.commit().expect("a commit");
            let counters = device.counters();
            let before = counters.gauges();
            let kept = |gauges: crate::Gauges| (gauges.textures, gauges.descriptor_slots);
            drop(view);
            thread::sleep(Duration::from_millis(50));
            assert_eq!(
                kept(counters.gauges()),
                kept(before),
                "{name}: the released view's image views and element of set 1 while its command \
                 buffer is pending"
            );
            f.set_signaled_value(1);
            finished(name, &held);
            assert_eq!(held.error(), None, "{name}");
            within("the view's going", || {
                kept(counters.gauges()) == (before.textures - 1, before.descriptor_slots - 1)
            });
        });
    }
}
