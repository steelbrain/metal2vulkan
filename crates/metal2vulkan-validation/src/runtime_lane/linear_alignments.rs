//! Device tests of `minimumLinearTextureAlignmentForPixelFormat:`, the device's answer for each
//! pixel format (the runtime's `texture/linear_alignment.rs`), on the device the cfg selects, under
//! the validation layer:
//!
//! - [`a_texture_at_the_answered_alignment_is_in_place`]: on a device of
//!   `HostPageTexturePolicy::InPlace` whose check passed for textures that read over pages
//!   imported by their pointer, a `ShaderRead` texture of each of [`FORMATS`] at each of
//!   [`WIDTHS`], [`ROWS`] rows tall, over a no-copy buffer at an offset of the answer and rows of
//!   its row's bytes rounded up to the answer apart, is an image in place (`vk_image` answers it,
//!   counted `host_textures_in_place`), and a blit out of it reads the caller's texels. Printed
//!   (`LINEAR-ALIGNMENT`): every format's answer, by answer, the device's own
//!   (`deviceLinearTextureAlignmentBytes`), and how long the answers took.
//! - [`a_row_under_the_answer_is_made`]: an `RGBA8Unorm` texture of rows of 64 bytes at an
//!   offset of 64, over a no-copy buffer and over a buffer of the runtime's, is made on every
//!   device, as Metal makes it: a shadow (counted `shadowed_textures`) where the answer is more
//!   than 64, and its blit out reads its texels.
//! - [`a_device_of_the_same_driver_reads_the_answer_again`]: the first query probes every format
//!   of the driver (`WorkCounts::linear_alignments_probed`) and no later one asks anything; a
//!   device of the same driver and cache reads the one record and probes nothing
//!   (`linear_alignments_recalled`), answering every format alike. Printed: the first query's
//!   time with no record and with it.

use super::host_memory::{counting, no_copy, not_importable, Mapping, GUEST_PAGE};
use super::{layer_reported_nothing_at_all, read_buffer, validated_device_with};
use super::{Messages, ScratchCache, WAIT};
use crate::gpu::TEST_DEVICE;
use metal2vulkan::device::HostPageTexturePolicy;
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use metal2vulkan_runtime::{
    BlitOption, Buffer, Device, Origin, PrivateCount, ResourceOptions, Size, Texture,
    TextureDescriptor,
};
use std::collections::BTreeMap;
use std::time::{Duration, Instant};

/// The formats the in-place test makes textures of, with their texel's bytes: each texel size a
/// buffer backs.
const FORMATS: [(PixelFormat, u64); 9] = [
    (PixelFormat::R8Unorm, 1),
    (PixelFormat::RG8Unorm, 2),
    (PixelFormat::R16Float, 2),
    (PixelFormat::RGBA8Unorm, 4),
    (PixelFormat::BGRA8Unorm, 4),
    (PixelFormat::R32Float, 4),
    (PixelFormat::RGBA16Float, 8),
    (PixelFormat::RG32Float, 8),
    (PixelFormat::RGBA32Float, 16),
];

/// The widths the in-place test makes textures at, in texels: none of them the runtime's probe's.
const WIDTHS: [u64; 8] = [2, 5, 16, 33, 128, 200, 640, 1307];

/// The rows of each texture: no multiple of 4.
const ROWS: u64 = 5;

/// The longest the device's check of textures over host pages may take to answer.
const CHECK_WAIT: Duration = Duration::from_secs(120);

/// A texture over a no-copy buffer of fresh anonymous pages, and the buffer and pages it lies in.
struct Over {
    texture: Texture,
    buffer: Buffer,
    mapping: Mapping,
}

/// A `ShaderRead` texture of `pf`, `w` x `h` texels, at `offset` with rows
/// `bpr` apart in a no-copy buffer of anonymous pages holding [`counting`] bytes, with room for 4
/// rows past its last (lavapipe's linear image takes its height rounded up to 4, and an image in
/// place lies inside the buffer's memory); `None` where the device imports no host pages.
fn over_pages(
    device: &Device,
    pf: PixelFormat,
    (w, h): (u64, u64),
    (offset, bpr): (u64, u64),
) -> Option<Over> {
    let end = offset + bpr * (h + 4);
    let len = (end as usize).next_multiple_of(GUEST_PAGE);
    let mapping = Mapping::anonymous(len);
    mapping.write(0, &counting(len, 3));
    // SAFETY: the mapping outlives the buffer and the texture (`Over` drops them first).
    let made = unsafe {
        no_copy(
            device,
            mapping.pointer(),
            len,
            ResourceOptions::default(),
            None,
        )
    };
    let buffer = match made {
        Ok(buffer) => buffer,
        Err(error) if not_importable(&error) => return None,
        Err(error) => panic!("new_buffer_with_bytes_no_copy: {error:?}"),
    };
    let mut d = TextureDescriptor::texture_2d(pf, w, h, false);
    d.usage = TextureUsage::SHADER_READ;
    d.resource_options = buffer.resource_options();
    let texture = buffer.new_texture(&d, offset, bpr).unwrap();
    Some(Over {
        texture,
        buffer,
        mapping,
    })
}

/// The texels of `texture` (`w` x `h`, of `texel` bytes), read by a blit into a buffer, packed.
fn blitted(device: &Device, texture: &Texture, (w, h): (u64, u64), texel: u64) -> Vec<u8> {
    let queue = device.new_command_queue().unwrap();
    let length = w * texel * h;
    let out = device
        .new_buffer(length, ResourceOptions::shared())
        .unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_texture_to_buffer(
        texture,
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(w, h, 1),
        &out,
        0,
        w * texel,
        length,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    read_buffer(&out, 0, length).unwrap()
}

/// The texels of rows `bpr` apart from `offset` in `bytes`, `w` x `h` of `texel` bytes, packed.
fn packed(bytes: &[u8], (offset, bpr): (u64, u64), (w, h): (u64, u64), texel: u64) -> Vec<u8> {
    (0..h)
        .flat_map(|y| {
            let at = (offset + y * bpr) as usize;
            bytes[at..at + (w * texel) as usize].to_vec()
        })
        .collect()
}

/// Run `test` on a lane device of `policy` over the cache at `cache`, then fail on any validation
/// message, leaks included.
fn on_device_of(
    policy: HostPageTexturePolicy,
    cache: &ScratchCache,
    test: impl FnOnce(&Device, &Messages),
) {
    let messages = Messages::default();
    {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
            options.policy.host_page_textures = policy;
        })
        .unwrap();
        test(&device, &messages);
    }
    layer_reported_nothing_at_all(&messages);
}

/// Every format's answer and the device's own, printed, each at least 16 and a power of two: a
/// texture at the answer is in place, over host pages imported by their pointer, wherever the
/// device's check passed for textures that read there, and its blit out reads its texels.
#[test]
fn a_texture_at_the_answered_alignment_is_in_place() {
    let cache = ScratchCache::new().unwrap();
    on_device_of(HostPageTexturePolicy::InPlace, &cache, |device, _| {
        let started = Instant::now();
        let mut by_answer: BTreeMap<u64, Vec<&str>> = BTreeMap::new();
        for &pf in PixelFormat::ALL {
            if let Ok(answer) = device.minimum_linear_texture_alignment_for_pixel_format(pf) {
                assert!(answer.is_power_of_two() && answer >= 16, "{pf:?}: {answer}");
                by_answer.entry(answer).or_default().push(pf.name());
            }
        }
        let formats_took = started.elapsed();
        let own = device
            .private_properties()
            .count(PrivateCount::DeviceLinearTextureAlignmentBytes)
            .value();
        let most = by_answer.keys().max().copied();
        assert_eq!(own, most, "the device's own is the largest of the formats'");
        for (answer, formats) in &by_answer {
            eprintln!(
                "LINEAR-ALIGNMENT on {}: {answer} for {} formats: {}",
                device.name(),
                formats.len(),
                formats.join(" ")
            );
        }
        let counts = device.work_counts();
        eprintln!(
            "LINEAR-ALIGNMENT on {}: the device's own {own:?}; every format's in {:.1} ms, {} \
             probed, {} recalled",
            device.name(),
            formats_took.as_secs_f64() * 1e3,
            counts.linear_alignments_probed,
            counts.linear_alignments_recalled
        );
        assert!(device.settle_in_place_checks(CHECK_WAIT));
        let report = device.in_place_check_report();
        let passed = report.iter().any(|line| line == "Pointer: Passed");
        eprintln!(
            "LINEAR-ALIGNMENT on {}: the in-place check {report:?}",
            device.name()
        );
        let mut in_place = 0;
        for (pf, texel) in FORMATS {
            let answer = device
                .minimum_linear_texture_alignment_for_pixel_format(pf)
                .unwrap();
            for w in WIDTHS {
                let at = (answer, (w * texel).next_multiple_of(answer));
                let before = device.work_counts();
                let Some(over) = over_pages(device, pf, (w, ROWS), at) else {
                    eprintln!(
                        "LINEAR-ALIGNMENT on {}: the device imports no host memory",
                        device.name()
                    );
                    assert_eq!(answer, 16, "{pf:?}: Metal's where no image is in place");
                    return;
                };
                let what = format!("{} {w}x{ROWS} at {at:?}", pf.name());
                // SAFETY: the handle is only compared, while the texture lives.
                let aliased = unsafe { over.texture.vk_image() }.is_some();
                let counted =
                    device.work_counts().host_textures_in_place - before.host_textures_in_place;
                assert_eq!(counted, u64::from(aliased), "{what}: its form not counted");
                if passed {
                    assert!(aliased, "{what}: a shadow at the answered alignment");
                }
                in_place += u64::from(aliased);
                let pages = over.mapping.read(0..over.buffer.length() as usize);
                let ours = blitted(device, &over.texture, (w, ROWS), texel);
                assert!(
                    ours == packed(&pages, at, (w, ROWS), texel),
                    "{what}: the blit out read other bytes than the caller's texels"
                );
            }
        }
        eprintln!(
            "LINEAR-ALIGNMENT on {}: {in_place} of {} textures at the answer in place",
            device.name(),
            FORMATS.len() * WIDTHS.len()
        );
    });
}

/// A texture of rows of 64 bytes at an offset of 64 (`RGBA8Unorm`, 16 texels wide), a multiple
/// of Metal's 16 and under the device's answer where it is larger, is made over a no-copy buffer
/// and over a buffer of the runtime's, as a shadow where the answer is more than 64, and reads its
/// texels.
#[test]
fn a_row_under_the_answer_is_made() {
    let cache = ScratchCache::new().unwrap();
    on_device_of(HostPageTexturePolicy::InPlace, &cache, |device, _| {
        let (pf, texel, w) = (PixelFormat::RGBA8Unorm, 4, 16);
        let at = (64, 64);
        let answer = device
            .minimum_linear_texture_alignment_for_pixel_format(pf)
            .unwrap();
        assert!(device.settle_in_place_checks(CHECK_WAIT));
        // Over host pages.
        let before = device.work_counts();
        if let Some(over) = over_pages(device, pf, (w, ROWS), at) {
            let shadowed = device.work_counts().shadowed_textures - before.shadowed_textures;
            // SAFETY: the handle is only compared, while the texture lives.
            let aliased = unsafe { over.texture.vk_image() }.is_some();
            eprintln!(
                "LINEAR-ALIGNMENT on {}: 64-byte rows over host pages, the answer {answer}: in \
                 place {aliased}, shadows {shadowed}",
                device.name()
            );
            assert_eq!(shadowed, u64::from(!aliased));
            if answer > 64 {
                assert!(
                    !aliased,
                    "rows of 64 bytes in place under an answer of {answer}"
                );
            }
            let pages = over.mapping.read(0..over.buffer.length() as usize);
            let ours = blitted(device, &over.texture, (w, ROWS), texel);
            assert!(ours == packed(&pages, at, (w, ROWS), texel));
        }
        // Over a buffer of the runtime's.
        let len = at.0 + at.1 * ROWS;
        let bytes = counting(len as usize, 5);
        let buffer = device
            .new_buffer_with_bytes(&bytes, ResourceOptions::shared())
            .unwrap();
        let mut d = TextureDescriptor::texture_2d(pf, w, ROWS, false);
        d.usage = TextureUsage::SHADER_READ;
        d.resource_options = buffer.resource_options();
        let before = device.work_counts();
        let texture = buffer.new_texture(&d, at.0, at.1).unwrap();
        let shadowed = device.work_counts().shadowed_textures - before.shadowed_textures;
        // SAFETY: the handle is only compared, while the texture lives.
        let aliased = unsafe { texture.vk_image() }.is_some();
        eprintln!(
            "LINEAR-ALIGNMENT on {}: 64-byte rows over a buffer of the runtime's: aliased \
             {aliased}, shadows {shadowed}",
            device.name()
        );
        if answer > 64 {
            assert_eq!((aliased, shadowed), (false, 1));
        }
        let ours = blitted(device, &texture, (w, ROWS), texel);
        assert!(ours == packed(&bytes, at, (w, ROWS), texel));
    });
}

/// Every format's answer on `device`, and the formats it probed and read from a record meanwhile.
fn every_answer(device: &Device) -> (BTreeMap<u32, u64>, u64, u64) {
    let before = device.work_counts();
    let answers = (PixelFormat::ALL.iter())
        .filter_map(|&pf| {
            let answer = device.minimum_linear_texture_alignment_for_pixel_format(pf);
            answer.ok().map(|answer| (pf.as_raw(), answer))
        })
        .collect();
    let after = device.work_counts();
    let probed = after.linear_alignments_probed - before.linear_alignments_probed;
    let recalled = after.linear_alignments_recalled - before.linear_alignments_recalled;
    (answers, probed, recalled)
}

/// The first query of a format (`BGRA8Unorm`) probes every format a buffer backs, once, and no
/// later query of any asks the driver anything; a device of the same driver over the same cache
/// reads the first device's one record at its first query, probes nothing, and answers every
/// format alike. Printed (`LINEAR-ALIGNMENT`): how long that first query took with no record and
/// with the record, which a program pays once as it starts.
#[test]
fn a_device_of_the_same_driver_reads_the_answer_again() {
    let cache = ScratchCache::new().unwrap();
    let mut first = BTreeMap::new();
    for record in [false, true] {
        on_device_of(HostPageTexturePolicy::InPlace, &cache, |device, _| {
            let before = device.work_counts();
            let started = Instant::now();
            let answer = device.minimum_linear_texture_alignment_for_pixel_format(BGRA8);
            let took = started.elapsed();
            let after = device.work_counts();
            let probed = after.linear_alignments_probed - before.linear_alignments_probed;
            let recalled = after.linear_alignments_recalled - before.linear_alignments_recalled;
            let (answers, more_probed, more_recalled) = every_answer(device);
            eprintln!(
                "LINEAR-ALIGNMENT on {}: the first query (BGRA8Unorm {answer:?}) with the record \
                 {record} took {:.2} ms: {probed} probed, {recalled} recalled; every other query \
                 {more_probed} probed, {more_recalled} recalled",
                device.name(),
                took.as_secs_f64() * 1e3
            );
            assert_eq!(answer.ok(), answers.get(&BGRA8.as_raw()).copied());
            assert_eq!(
                (more_probed, more_recalled),
                (0, 0),
                "worked out at the first query"
            );
            let formats = answers.len() as u64;
            match record {
                false => {
                    assert_eq!((probed, recalled), (formats, 0), "every format probed");
                    first = answers;
                }
                true => {
                    assert_eq!((probed, recalled), (0, formats), "the record read");
                    assert_eq!(answers, first, "the record's answers");
                }
            }
        });
    }
}

/// The format the record test asks first.
const BGRA8: PixelFormat = PixelFormat::BGRA8Unorm;
