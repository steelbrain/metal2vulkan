//! Device tests of texture views of another component layout on a device that reinterprets no
//! image's components, on the device the cfg selects, under the validation layer.
//!
//! The device is one the test makes and the runtime adopts (`presenting::Adopted`), as a
//! consumer's would be: where the driver has `VK_KHR_portability_subset` (MoltenVK), the subset
//! is enabled without `imageViewFormatReinterpretation`, so the runtime makes no Vulkan view of
//! another component layout there unless its consumer says it enabled the subset's feature
//! (`RawDevice::view_format_reinterpretation`). What a shadow of such a view rests on is measured
//! here: a copy between two images of formats of one copy unit keeps every bit the source holds,
//! whatever the formats' components mean (NaN payloads, denormal halves, the snorm minimum,
//! packed floats, compressed blocks). The textures take `PixelFormatView` usage, as a texture
//! Metal views in another layout and its shadow do; what each holds of the bytes it was given is
//! printed apart, since a format's own storage may keep fewer bits than the copy moves.

use super::buffer_texture_blits::linear_rows_differ;
use super::degrade_notices::{assert_made_once, forced, shared_in_system_memory};
use super::presenting::Adopted;
use super::{at_plan, layer_reported_nothing, validated_device_with, Messages, ScratchCache};
use crate::gpu::TEST_DEVICE;
use metal2vulkan::error::Feature;
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};
use metal2vulkan_runtime::{
    BlitCommandEncoder, BlitOption, Buffer, ClearColor, CommandQueue, CpuCacheMode, Degradation,
    Device, Error, HazardTrackingMode, LoadAction, Origin, Region, RenderPassDescriptor,
    ResourceOptions, Size, StorageMode, StoreAction, Texture, TextureDescriptor,
};
use std::time::Duration;

const WAIT: Duration = Duration::from_secs(60);

/// The textures' side, in texels: 16, four blocks of a compressed format.
const SIDE: u64 = 16;

/// Pairs of formats of one copy unit whose components differ, each copied both ways: a 4-byte
/// texel as words, floats and the 8- and 16-bit normalized and float layouts, the packed formats,
/// 2-, 8- and 16-byte texels, and a compressed family's two members.
const PAIRS: [(PixelFormat, PixelFormat); 13] = [
    (PixelFormat::RGBA8Unorm, PixelFormat::R32Uint),
    (PixelFormat::RGBA8Snorm, PixelFormat::R32Float),
    (PixelFormat::RG16Snorm, PixelFormat::R32Uint),
    (PixelFormat::RG16Float, PixelFormat::R32Float),
    (PixelFormat::RGB10A2Unorm, PixelFormat::R32Uint),
    (PixelFormat::RG11B10Float, PixelFormat::R32Uint),
    (PixelFormat::RGB9E5Float, PixelFormat::R32Float),
    (PixelFormat::R16Float, PixelFormat::R16Uint),
    (PixelFormat::B5G6R5Unorm, PixelFormat::RG8Snorm),
    (PixelFormat::RGBA16Float, PixelFormat::RG32Uint),
    (PixelFormat::RGBA16Snorm, PixelFormat::RG32Float),
    (PixelFormat::RGBA32Float, PixelFormat::RGBA32Uint),
    (PixelFormat::BC4RUnorm, PixelFormat::BC4RSnorm),
];

/// Words whose bits a conversion through the formats' components would change: float NaNs of
/// every width with payloads and signs, infinities, denormals and negative zeros, the snorm
/// minimums of 8 and 16 bits, and the packed floats' NaNs and denormals.
const SPECIAL: [u32; 16] = [
    0x7f80_0001, // f32 signalling NaN
    0xffc0_1234, // f32 quiet NaN, negative, with a payload
    0x0000_0001, // f32 denormal
    0x8000_0000, // f32 negative zero
    0x7c01_fe00, // f16 signalling NaN, f16 negative quiet NaN
    0x0001_8001, // f16 denormals of both signs
    0x8000_8000, // snorm16 minimums
    0x8080_8080, // snorm8 minimums
    0x7fff_ffff, // f32 NaN, all payload bits
    0xffff_ffff, // every bit
    0x07ff_fc01, // RG11B10: an 11-bit NaN and a 10-bit NaN
    0x0000_0401, // RG11B10: denormals
    0xf800_0001, // RGB9E5: the largest exponent, a small mantissa
    0x7c00_7c00, // f16 infinities
    0x0080_0000, // f32 smallest normal
    0xc000_0003, // RGB10A2 and packed fields at their edges
];

/// `n` bytes: the special words first, repeated, then a fixed pseudo-random sequence.
fn pattern(n: usize, seed: u32) -> Vec<u8> {
    let mut bytes: Vec<u8> = SPECIAL
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .cycle()
        .take(n / 2)
        .collect();
    let mut state = seed.wrapping_mul(0x9e37_79b9) | 1;
    while bytes.len() < n {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        bytes.extend(state.to_le_bytes());
    }
    bytes.truncate(n);
    bytes
}

/// The bytes of one row of `SIDE` texels of `format`, and of the whole level: a compressed
/// format's rows are rows of 4-texel blocks.
fn layout(format: PixelFormat) -> (u64, u64) {
    match format {
        PixelFormat::BC4RUnorm | PixelFormat::BC4RSnorm => {
            let row = SIDE / 4 * 8;
            (row, row * SIDE / 4)
        }
        _ => {
            let row = SIDE * texel_bytes(format);
            (row, row * SIDE)
        }
    }
}

fn texel_bytes(format: PixelFormat) -> u64 {
    use PixelFormat as P;
    match format {
        P::R16Float | P::R16Uint | P::B5G6R5Unorm | P::RG8Snorm => 2,
        P::RGBA16Float | P::RG32Uint | P::RGBA16Snorm | P::RG32Float => 8,
        P::RGBA32Float | P::RGBA32Uint => 16,
        _ => 4,
    }
}

/// A private `SIDE`-square 2D texture of `format` that Metal can view in another layout.
fn texture(device: &Device, format: PixelFormat) -> Result<Texture, Error> {
    let mut descriptor = TextureDescriptor::texture_2d(format, SIDE, SIDE, false);
    descriptor.usage = TextureUsage::SHADER_READ | TextureUsage::PIXEL_FORMAT_VIEW;
    descriptor.resource_options = ResourceOptions::new(
        StorageMode::Private,
        CpuCacheMode::DefaultCache,
        HazardTrackingMode::Default,
    );
    device.new_texture(&descriptor)
}

fn shared(device: &Device, bytes: &[u8]) -> Buffer {
    let buffer = device
        .new_buffer(bytes.len() as u64, ResourceOptions::shared())
        .unwrap();
    let contents = buffer.contents().expect("a host-visible buffer");
    // SAFETY: the buffer holds `bytes.len()` bytes at `contents`, alive with the buffer, and no
    // command buffer uses them yet.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr(), bytes.len()) };
    buffer
}

fn contents(buffer: &Buffer) -> Vec<u8> {
    let contents = buffer.contents().expect("a host-visible buffer");
    // SAFETY: as in `shared`; the command buffer that wrote them completed.
    unsafe { std::slice::from_raw_parts(contents.as_ptr(), buffer.length() as usize) }.to_vec()
}

/// How many bytes differ and the first differing unit, expected then got; `None` where none does.
type Differs = Option<(usize, String)>;

/// How `got` differs from `expected`, in units of `unit` bytes.
fn difference(expected: &[u8], got: &[u8], unit: usize) -> Differs {
    let differing = expected.iter().zip(got).filter(|(a, b)| a != b).count();
    (differing > 0).then(|| {
        let first = expected
            .chunks(unit)
            .zip(got.chunks(unit))
            .position(|(a, b)| a != b)
            .unwrap_or(0);
        let at = first * unit..(first + 1) * unit;
        (
            differing,
            format!(
                "unit {first}: {:02x?} -> {:02x?}",
                &expected[at.clone()],
                &got[at]
            ),
        )
    })
}

/// What `from`'s texture held of the bytes it was given, and what a copy of it into `to`'s kept
/// of what it held: each `None` where every byte arrived, else how many of the level's bytes
/// differ and the first differing unit, expected then got. `Err` names a format the device does
/// not hold.
fn copied(
    device: &Device,
    queue: &CommandQueue,
    from: PixelFormat,
    to: PixelFormat,
    seed: u32,
) -> Result<(Differs, Differs), String> {
    let (row, level) = layout(from);
    assert_eq!(
        layout(to),
        (row, level),
        "{from:?} and {to:?} are of one copy unit"
    );
    let source = texture(device, from).map_err(|error| format!("{from:?}: {error}"))?;
    let destination = texture(device, to).map_err(|error| format!("{to:?}: {error}"))?;
    let bytes = pattern(level as usize, seed);
    let upload = shared(device, &bytes);
    let held = shared(device, &vec![0xee; level as usize]);
    let back = shared(device, &vec![0xee; level as usize]);
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    let size = Size::new(SIDE, SIDE, 1);
    let origin = Origin::new(0, 0, 0);
    let none = BlitOption::None;
    blit.copy_from_buffer_to_texture(&upload, 0, row, level, size, &source, 0, 0, origin, none)
        .unwrap();
    blit.copy_from_texture_to_buffer(&source, 0, 0, origin, size, &held, 0, row, level, none)
        .unwrap();
    blit.copy_from_texture_to_texture(&source, 0, 0, origin, size, &destination, 0, 0, origin)
        .unwrap();
    blit.copy_from_texture_to_buffer(&destination, 0, 0, origin, size, &back, 0, row, level, none)
        .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    let unit = match from {
        PixelFormat::BC4RUnorm | PixelFormat::BC4RSnorm => 8,
        _ => texel_bytes(from) as usize,
    };
    let held = contents(&held);
    Ok((
        difference(&bytes, &held, unit),
        difference(&held, &contents(&back), unit),
    ))
}

/// A copy between two images of formats of one copy unit keeps every bit the source holds, both
/// ways, for each of [`PAIRS`], on a device that reinterprets no image's components where its
/// portability subset could (the measurements of view reinterpretation rest on). Each copy is
/// printed (`copy FROM -> TO: exact`, or how many bytes differ and the first unit that does), and
/// so is what the source held of the bytes it was given where it held less (`texture FROM holds:
/// ...`, the format's storage, not the copy); a format the device does not hold is printed and
/// skipped. The device's answer to a view of another layout is printed too: made, a shadow of the
/// texture's texels where the subset is enabled (`texture/view_shadow.rs`).
#[test]
fn copies_between_formats_of_one_unit_keep_every_bit() {
    let cache = ScratchCache::new().unwrap();
    let adopted = Adopted::open(false).unwrap();
    let mut differing = Vec::new();
    {
        let device = adopted.runtime(cache.path());
        let queue = device.new_command_queue().unwrap();
        let mut viewed = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
        viewed.usage = TextureUsage::SHADER_READ | TextureUsage::PIXEL_FORMAT_VIEW;
        viewed.resource_options = ResourceOptions::private();
        let viewed = device.new_texture(&viewed).unwrap();
        match viewed.new_texture_view(PixelFormat::R32Uint) {
            Ok(_) => println!("view RGBA8Unorm as R32Uint: made"),
            Err(Error::Unsupported(gap)) => {
                assert_eq!(gap.feature, Feature::ViewFormatReinterpretation);
                println!("view RGBA8Unorm as R32Uint: refused ({gap:?})");
            }
            Err(error) => panic!("view RGBA8Unorm as R32Uint: {error}"),
        }
        for (seed, (a, b)) in PAIRS.into_iter().enumerate() {
            for (from, to) in [(a, b), (b, a)] {
                match copied(&device, &queue, from, to, seed as u32 + 1) {
                    Ok((held, copy)) => {
                        if let Some((count, first)) = held {
                            println!("texture {from:?} holds: {count} bytes differ, {first}");
                        }
                        match copy {
                            None => println!("copy {from:?} -> {to:?}: exact"),
                            Some((count, first)) => {
                                println!("copy {from:?} -> {to:?}: {count} bytes differ, {first}");
                                differing.push((from, to, count, first));
                            }
                        }
                    }
                    Err(unavailable) => {
                        println!("copy {from:?} -> {to:?}: unavailable, {unavailable}")
                    }
                }
            }
        }
    }
    let reported = adopted.close();
    assert!(
        reported.is_empty(),
        "the validation layer reported: {reported:#?}"
    );
    assert!(differing.is_empty(), "copies changed bits: {differing:#?}");
}

/// What `device` answers to an `RGBA8Unorm` texture's view as `R32Uint`, cleared to 0x11223344
/// as a render pass's attachment (a Vulkan view of the image in the view's format): `Some` of the
/// texture's bytes after the pass, or `None` where the view is refused for
/// `Feature::ViewFormatReinterpretation`. Any other answer fails.
fn cleared_through_a_uint_view(device: &Device) -> Option<Vec<u8>> {
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
    descriptor.usage = TextureUsage::RENDER_TARGET | TextureUsage::PIXEL_FORMAT_VIEW;
    descriptor.resource_options = ResourceOptions::private();
    let texture = device.new_texture(&descriptor).unwrap();
    let view = match texture.new_texture_view(PixelFormat::R32Uint) {
        Ok(view) => view,
        Err(Error::Unsupported(gap)) if gap.feature == Feature::ViewFormatReinterpretation => {
            return None
        }
        Err(error) => panic!("new_texture_view: {error}"),
    };
    let level = (SIDE * SIDE * 4) as usize;
    let back = shared(device, &vec![0xee; level]);
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut pass = RenderPassDescriptor::new();
    let attachment = &mut pass.color_attachments[0];
    attachment.texture = Some(view);
    attachment.load_action = LoadAction::Clear;
    attachment.store_action = StoreAction::Store;
    attachment.clear_color = ClearColor::new(f64::from(0x1122_3344u32), 0.0, 0.0, 0.0);
    command_buffer
        .render_command_encoder(&pass)
        .unwrap()
        .end_encoding()
        .unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    let size = Size::new(SIDE, SIDE, 1);
    let origin = Origin::new(0, 0, 0);
    let (row, image) = (SIDE * 4, SIDE * SIDE * 4);
    blit.copy_from_texture_to_buffer(
        &texture,
        0,
        0,
        origin,
        size,
        &back,
        0,
        row,
        image,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    Some(contents(&back))
}

/// An adopted device makes a view of another component layout and stores its words' bytes
/// through it, whether or not its consumer enabled the portability subset's
/// `imageViewFormatReinterpretation` and says so (`RawDevice::view_format_reinterpretation`): an
/// `R32Uint` view of an `RGBA8Unorm` texture cleared to 0x11223344 leaves `44 33 22 11` in every
/// texel, as Metal's view does (`view_attachments::a_uint_view_stores_its_words`). Where the
/// device has the subset and its consumer does not say so, the view is a shadow of the texture's
/// texels (`texture/view_shadow.rs`); elsewhere a Vulkan view of the image.
#[test]
fn an_adopted_device_views_another_layout_whether_its_consumer_enabled_it_or_not() {
    let cache = ScratchCache::new().unwrap();
    let word = 0x1122_3344u32.to_le_bytes();
    let expected: Vec<u8> = word
        .iter()
        .copied()
        .cycle()
        .take((SIDE * SIDE * 4) as usize)
        .collect();
    for reinterpret in [true, false] {
        let adopted = if reinterpret {
            Adopted::open_reinterpreting()
        } else {
            Adopted::open(false)
        }
        .unwrap();
        let (subset, reinterprets) = (adopted.subset(), adopted.reinterprets());
        let answer = {
            let device = adopted.runtime(cache.path());
            let answer = cleared_through_a_uint_view(&device);
            println!(
                "view on an adopted {} (subset: {subset}, reinterpretation enabled: \
                 {reinterprets}): {}",
                device.name(),
                match (&answer, subset && !reinterprets) {
                    (None, _) => "refused",
                    (Some(_), true) => "made, shadowed",
                    (Some(_), false) => "made",
                }
            );
            answer
        };
        assert_eq!(
            answer.as_ref(),
            Some(&expected),
            "the view stores its words' bytes"
        );
        let reported = adopted.close();
        assert!(
            reported.is_empty(),
            "the validation layer reported: {reported:#?}"
        );
    }
}

/// The kinds a texture view of another component layout takes where the device makes no Vulkan
/// view of an image in another layout: the copies of the view's shadow. Every venue takes it
/// under `DeviceOptions::withhold_view_reinterpretation`.
pub(super) const VIEW_SHADOWS: [Degradation; 1] = [Degradation::ShadowedTextureView];

/// A `SIDE`-square 2D `RGBA8Unorm` texture that Metal can view in another layout, of
/// `storage` and `levels` and, where `slices` is above 1, a 2D array.
fn viewed(device: &Device, storage: StorageMode, levels: u64, slices: u64) -> Texture {
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
    if slices > 1 {
        descriptor.texture_type = TextureType::Type2DArray;
        descriptor.array_length = slices;
    }
    descriptor.mipmap_level_count = levels;
    descriptor.usage = TextureUsage::SHADER_READ | TextureUsage::PIXEL_FORMAT_VIEW;
    descriptor.resource_options = ResourceOptions::new(
        storage,
        CpuCacheMode::DefaultCache,
        HazardTrackingMode::Default,
    );
    device.new_texture(&descriptor).unwrap()
}

/// Run `encode` in one blit encoder of one command buffer of `queue`, and wait for it.
fn blit(queue: &CommandQueue, encode: impl FnOnce(&mut BlitCommandEncoder)) {
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.blit_command_encoder().unwrap();
    encode(&mut encoder);
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
}

/// Copy `rows` rows of `texture`'s `level` and `slice` from `buffer`, the level's rows packed,
/// or (`into_texture` false) to it.
fn rows(
    encoder: &mut BlitCommandEncoder,
    texture: &Texture,
    (slice, level): (u64, u64),
    rows: std::ops::Range<u64>,
    buffer: &Buffer,
    into_texture: bool,
) {
    let width = (texture.width() >> level).max(1);
    let row = width * 4;
    let size = Size::new(width, rows.end - rows.start, 1);
    let origin = Origin::new(0, rows.start, 0);
    let (offset, image) = (rows.start * row, (rows.end - rows.start) * row);
    let none = BlitOption::None;
    if into_texture {
        encoder
            .copy_from_buffer_to_texture(
                buffer, offset, row, image, size, texture, slice, level, origin, none,
            )
            .unwrap();
    } else {
        encoder
            .copy_from_texture_to_buffer(
                texture, slice, level, origin, size, buffer, offset, row, image, none,
            )
            .unwrap();
    }
}

/// Through one blit encoder, then across encoders, an `R32Uint` view of an `RGBA8Unorm` texture
/// reads the texture's writes and the texture reads the view's, half a level at a time: a
/// shadow's copies in and back go between the commands as the texture's own uses order them.
fn blits_through_a_view(device: &Device) {
    let queue = device.new_command_queue().unwrap();
    let texture = viewed(device, StorageMode::Private, 1, 1);
    let view = texture.new_texture_view(PixelFormat::R32Uint).unwrap();
    let level = (SIDE * SIDE * 4) as usize;
    let half = level / 2;
    let [p1, p2, p3, p4] = [1, 2, 3, 4].map(|seed| pattern(level, seed));
    let given = [&p1, &p2, &p3, &p4].map(|bytes| shared(device, bytes));
    let read = [(); 4].map(|_| shared(device, &vec![0xee; level]));
    let top = 0..SIDE / 2;
    let bottom = SIDE / 2..SIDE;
    blit(&queue, |encoder| {
        rows(encoder, &texture, (0, 0), 0..SIDE, &given[0], true);
        rows(encoder, &view, (0, 0), 0..SIDE, &read[0], false);
        rows(encoder, &view, (0, 0), top.clone(), &given[1], true);
        rows(encoder, &texture, (0, 0), 0..SIDE, &read[1], false);
        rows(encoder, &texture, (0, 0), bottom.clone(), &given[2], true);
        rows(encoder, &view, (0, 0), 0..SIDE, &read[2], false);
    });
    blit(&queue, |encoder| {
        rows(encoder, &view, (0, 0), 0..SIDE, &given[3], true)
    });
    blit(&queue, |encoder| {
        rows(encoder, &texture, (0, 0), 0..SIDE, &read[3], false)
    });
    let joined = |a: &[u8], b: &[u8]| [&a[..half], &b[half..]].concat();
    assert_eq!(contents(&read[0]), p1, "the view reads the texture's write");
    assert_eq!(
        contents(&read[1]),
        joined(&p2, &p1),
        "the texture reads the view's"
    );
    assert_eq!(
        contents(&read[2]),
        joined(&p2, &p3),
        "the view reads the texture's again"
    );
    assert_eq!(contents(&read[3]), p4, "a later encoder reads the view's");
}

/// A view of one level and one slice of a 2-level, 2-slice array writes those texels alone and
/// reads them: the shadow's copies move the subresources the view presents.
fn subresources_through_a_view(device: &Device) {
    let queue = device.new_command_queue().unwrap();
    let texture = viewed(device, StorageMode::Private, 2, 2);
    let view = texture
        .new_texture_view_with_levels_slices(PixelFormat::R32Uint, TextureType::Type2D, 1..2, 1..2)
        .unwrap();
    let parts = [(0, 0), (0, 1), (1, 0), (1, 1)];
    let bytes = |(_, level): (u64, u64)| ((SIDE >> level) * (SIDE >> level) * 4) as usize;
    let given: Vec<Vec<u8>> = parts
        .iter()
        .zip(10..)
        .map(|(&part, seed)| pattern(bytes(part), seed))
        .collect();
    let written = pattern(bytes((1, 1)), 20);
    let buffers: Vec<Buffer> = given.iter().map(|b| shared(device, b)).collect();
    let through = shared(device, &written);
    let read_view = shared(device, &vec![0xee; bytes((1, 1))]);
    let read: Vec<Buffer> = parts
        .iter()
        .map(|&part| shared(device, &vec![0xee; bytes(part)]))
        .collect();
    let rows_of = |(_, level): (u64, u64)| 0..SIDE >> level;
    blit(&queue, |encoder| {
        for (&part, buffer) in parts.iter().zip(&buffers) {
            rows(encoder, &texture, part, rows_of(part), buffer, true);
        }
        rows(encoder, &view, (0, 0), rows_of((0, 1)), &read_view, false);
        rows(encoder, &view, (0, 0), rows_of((0, 1)), &through, true);
        for (&part, buffer) in parts.iter().zip(&read) {
            rows(encoder, &texture, part, rows_of(part), buffer, false);
        }
    });
    assert_eq!(
        contents(&read_view),
        given[3],
        "the view reads slice 1, level 1"
    );
    for (i, (&part, buffer)) in parts.iter().zip(&read).enumerate() {
        let expected = if part == (1, 1) { &written } else { &given[i] };
        assert_eq!(&contents(buffer), expected, "slice and level {part:?}");
    }
}

/// An `R32Uint` view of an `RGBA8Unorm` texture over a buffer's bytes, its rows apart, reads and
/// writes those bytes (plan (a): a texture over the same bytes in the view's format).
fn buffer_texture_view(device: &Device) {
    let queue = device.new_command_queue().unwrap();
    // Rows 16 bytes apart past their texels: Metal's linear alignment is 16 bytes for every
    // format a buffer backs (`minimumLinearTextureAlignmentForPixelFormat:`).
    let row = SIDE * 4 + 16;
    let length = row * SIDE;
    let before = pattern(length as usize, 30);
    let buffer = shared(device, &before);
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
    descriptor.usage = TextureUsage::SHADER_READ | TextureUsage::PIXEL_FORMAT_VIEW;
    descriptor.resource_options = buffer.resource_options();
    let texture = buffer.new_texture(&descriptor, 0, row).unwrap();
    let view = texture.new_texture_view(PixelFormat::R32Uint).unwrap();
    let packed = (SIDE * 4) as usize;
    let level = packed * SIDE as usize;
    let written = pattern(level, 31);
    let given = shared(device, &written);
    let read = shared(device, &vec![0xee; level]);
    blit(&queue, |encoder| {
        rows(encoder, &view, (0, 0), 0..SIDE, &read, false);
        rows(encoder, &view, (0, 0), 0..SIDE, &given, true);
    });
    let rows_of = |bytes: &[u8]| -> Vec<u8> {
        bytes
            .chunks(row as usize)
            .flat_map(|r| r[..packed].to_vec())
            .collect()
    };
    assert_eq!(
        contents(&read),
        rows_of(&before),
        "the view reads the buffer's bytes"
    );
    let after = contents(&buffer);
    assert_eq!(rows_of(&after), written, "the view writes them");
    let gaps = |bytes: &[u8]| -> Vec<u8> {
        bytes
            .chunks(row as usize)
            .flat_map(|r| r[packed..].to_vec())
            .collect()
    };
    assert_eq!(
        gaps(&after),
        gaps(&before),
        "and not the bytes between rows"
    );
}

/// Through an `R32Uint` view of a `Shared` `RGBA8Unorm` texture, `replace_region` writes the
/// texture's texels and `get_bytes` reads them.
fn host_access_through_a_view(device: &Device) {
    let texture = viewed(device, StorageMode::Shared, 1, 1);
    let view = texture.new_texture_view(PixelFormat::R32Uint).unwrap();
    let row = SIDE * 4;
    let level = (row * SIDE) as usize;
    let region = Region::new_2d(0, 0, SIDE, SIDE);
    let given = pattern(level, 40);
    view.replace_region(region, 0, 0, &given, row, 0).unwrap();
    let mut through_texture = vec![0xee; level];
    texture
        .get_bytes(&mut through_texture, row, 0, region, 0, 0)
        .unwrap();
    assert_eq!(through_texture, given, "the texture reads the view's write");
    let again = pattern(level, 41);
    texture
        .replace_region(region, 0, 0, &again, row, 0)
        .unwrap();
    let mut through_view = vec![0xee; level];
    view.get_bytes(&mut through_view, row, 0, region, 0, 0)
        .unwrap();
    assert_eq!(through_view, again, "the view reads the texture's write");
}

/// A texture view of another component layout answers as a Vulkan view of the image in the
/// view's format does, where the device makes none and the view is a shadow of the image's
/// texels (`DeviceOptions::withhold_view_reinterpretation`, `texture/view_shadow.rs`), and where
/// it makes one: an `R32Uint` view of an `RGBA8Unorm` texture cleared as an attachment stores its
/// words' bytes; blits through the view and the texture read each other's writes, within one
/// encoder and across encoders; a view of one level and slice moves those texels alone; a view of
/// a texture over a buffer reads and writes the buffer's bytes; and `replace_region` and
/// `get_bytes` through the view reach the texture's texels. The shadow's copies are announced
/// once, and only where they are made.
#[test]
fn a_shadowed_view_answers_as_a_view_of_the_image_does() {
    let expected: Vec<u8> = 0x1122_3344u32
        .to_le_bytes()
        .into_iter()
        .cycle()
        .take((SIDE * SIDE * 4) as usize)
        .collect();
    for withheld in [true, false] {
        let cache = ScratchCache::new().unwrap();
        let messages = Messages::default();
        {
            let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
                options.withhold_view_reinterpretation = withheld;
                at_plan(options)
            })
            .unwrap();
            assert_eq!(
                cleared_through_a_uint_view(&device).as_ref(),
                Some(&expected),
                "withheld {withheld}: the view stores its words' bytes"
            );
            blits_through_a_view(&device);
            subresources_through_a_view(&device);
            buffer_texture_view(&device);
            host_access_through_a_view(&device);
            // The view's shadow where withheld, nothing of it where the device views the image;
            // and what the steps force: the texture over a buffer's shadow where no linear image
            // holds its texels alone, and the `Shared` buffers' system memory.
            let kinds: &[Degradation] = if withheld { &VIEW_SHADOWS } else { &[] };
            let row = SIDE * 4 + 16;
            let side = SIDE as u32;
            let first = forced(&[
                (
                    Degradation::ShadowedTexture,
                    linear_rows_differ(&device, PixelFormat::RGBA8Unorm, [side, side], 4, row),
                ),
                (
                    Degradation::SharedInSystemMemory,
                    shared_in_system_memory(&device),
                ),
            ]);
            assert_made_once(&messages, kinds, &first);
            println!(
                "views of another layout on {} (withheld: {withheld}): as the image's",
                device.name()
            );
        }
        layer_reported_nothing(&messages, &format!("withheld {withheld}"));
    }
}
