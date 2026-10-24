//! `getBytes` and `replaceRegion` of a `Shared` texture with the caller's bytes laid out packed and
//! pitched, of a 2D texture's whole level and part of it and of a 3D texture's images: each row
//! is read or written where its layout puts it; `getBytes` leaves the bytes between rows and
//! images and past the last row as they were, and `replaceRegion` writes no texel outside the
//! region. On whichever path the device copies through (on the host, `texture/host_copy.rs`, which
//! reads and writes a packed layout in place; on its staging queue, `texture/staging_copy.rs`; or
//! through a submission), which the tests print.
//!
//! And a two-plane extended-range texture's copies, whose texels hold a colour word the call moves
//! beside an alpha word it keeps (`two_planes.rs`), on the path every other texture's copies take:
//! where the device copies texels on the host they read the texels, lay the words over them and
//! write them back there, waiting for no submission, as Metal's calls of such a texture wait for
//! no GPU work.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use super::blit_holds::{device, Scratch};
use crate::{Device, Region, ResourceOptions, Texture, TextureDescriptor};
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};

/// The byte a caller's bytes hold where no row lands.
const UNTOUCHED: u8 = 0xab;

/// Each case: the texture's size, the region, the caller's bytes per row and per image, and the
/// length of the caller's bytes.
const CASES: [([u64; 3], Region, u64, u64, usize); 7] = [
    // Packed, whole: exactly its bytes, and with bytes past them.
    ([7, 5, 1], region([0, 0, 0], [7, 5, 1]), 28, 0, 140),
    ([7, 5, 1], region([0, 0, 0], [7, 5, 1]), 28, 0, 160),
    // Pitched rows.
    ([7, 5, 1], region([0, 0, 0], [7, 5, 1]), 36, 0, 36 * 4 + 28),
    // Part of the level, packed and pitched.
    ([7, 5, 1], region([2, 1, 0], [3, 2, 1]), 12, 0, 24),
    ([7, 5, 1], region([2, 1, 0], [3, 2, 1]), 20, 0, 32),
    // A 3D texture's images, packed and apart.
    ([4, 3, 2], region([0, 0, 0], [4, 3, 2]), 16, 48, 96),
    ([4, 3, 2], region([0, 0, 0], [4, 3, 2]), 16, 64, 112),
];

/// The region of `size` at `origin`.
const fn region(origin: [u64; 3], size: [u64; 3]) -> Region {
    Region {
        origin: crate::Origin {
            x: origin[0],
            y: origin[1],
            z: origin[2],
        },
        size: crate::Size {
            width: size[0],
            height: size[1],
            depth: size[2],
        },
    }
}

/// A `Shared` `RGBA8Unorm` texture of `width` x `height` x `depth` (3D where `depth` is above 1),
/// each byte of its texels its own (`pattern(_, 0)`), written whole.
fn texture(device: &Device, [width, height, depth]: [u64; 3]) -> Texture {
    let mut descriptor =
        TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, width, height, false);
    if depth > 1 {
        descriptor.texture_type = TextureType::Type3D;
        descriptor.depth = depth;
    }
    descriptor.usage = TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::shared();
    let texture = device.new_texture(&descriptor).unwrap();
    let bytes = pattern([width, height, depth], 0);
    let region = Region::new_3d(0, 0, 0, width, height, depth);
    let row = 4 * width;
    (texture.replace_region(region, 0, 0, &bytes, row, row * height)).unwrap();
    texture
}

/// A `Shared` two-plane extended-range texture of `width` x `height`, read and written by a
/// shader as the cases that sample one make it (`two_planes.rs`): its colour words are what a CPU
/// copy of it moves, 4 bytes a texel, and its alpha words lie in the image beside them.
fn two_plane(device: &Device, [width, height]: [u64; 2]) -> Texture {
    let format = PixelFormat::RGB10A82PXR10;
    let mut descriptor = TextureDescriptor::texture_2d(format, width, height, false);
    descriptor.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
    descriptor.resource_options = ResourceOptions::shared();
    device.new_texture(&descriptor).unwrap()
}

/// Texels of a texture of `size`, packed: byte `i` is `i * 7 + 1 + seed` modulo 251, so no two
/// bytes of a row or of neighbouring rows are alike, and two seeds' bytes differ everywhere.
fn pattern([width, height, depth]: [u64; 3], seed: u64) -> Vec<u8> {
    (0..4 * width * height * depth)
        .map(|i| ((i * 7 + 1 + seed) % 251) as u8)
        .collect()
}

/// The packed offset in a texture of `size` of the first texel of row `y` of image `z` of
/// `region`, and the offset in the caller's bytes laid out `bytes_per_row` and `bytes_per_image`
/// apart; each row is `4 * region.size.width` bytes.
fn rows(
    size: [u64; 3],
    region: Region,
    (bytes_per_row, bytes_per_image): (u64, u64),
) -> impl Iterator<Item = (usize, usize)> {
    (0..region.size.depth).flat_map(move |z| {
        (0..region.size.height).map(move |y| {
            let texel =
                ((region.origin.z + z) * size[1] + region.origin.y + y) * size[0] + region.origin.x;
            (
                4 * texel as usize,
                (z * bytes_per_image + y * bytes_per_row) as usize,
            )
        })
    })
}

/// The reads and writes the device made on the host and on its staging queue, and the internal
/// submissions it waited for, while `run` ran.
fn paths(device: &Device, run: impl FnOnce()) -> [u64; 3] {
    let before = device.work_counts();
    run();
    let after = device.work_counts();
    [
        after.host_texel_copies - before.host_texel_copies,
        after.staging_submissions - before.staging_submissions,
        after.internal_submissions - before.internal_submissions,
    ]
}

/// Each layout of each region reads its rows where the layout puts them, and nothing else.
#[test]
fn get_bytes_reads_each_layout_its_rows_alone() {
    let scratch = Scratch::new("get-bytes-layouts");
    let device = device(&scratch.0);
    let mut counted = [0; 3];
    for (size, region, bytes_per_row, bytes_per_image, len) in CASES {
        let texture = texture(&device, size);
        let mut bytes = vec![UNTOUCHED; len];
        let [host, staged, waited] = paths(&device, || {
            texture
                .get_bytes(&mut bytes, bytes_per_row, bytes_per_image, region, 0, 0)
                .unwrap()
        });
        counted = [counted[0] + host, counted[1] + staged, counted[2] + waited];
        let texels = pattern(size, 0);
        let mut wanted = vec![UNTOUCHED; len];
        let row = 4 * region.size.width as usize;
        for (texel, at) in rows(size, region, (bytes_per_row, bytes_per_image)) {
            wanted[at..at + row].copy_from_slice(&texels[texel..texel + row]);
        }
        assert_eq!(
            bytes, wanted,
            "{size:?} {region:?}, {bytes_per_row} and {bytes_per_image} apart"
        );
    }
    let [host, staged, waited] = counted;
    eprintln!(
        "{}: {host} of {} reads on the host, {staged} on the staging queue, \
         {waited} internal submissions waited for",
        device.name(),
        CASES.len()
    );
}

/// Each layout of each region writes its rows where the layout puts them into the region, and no
/// other texel: the texture read back whole holds the caller's rows in the region and its own
/// texels elsewhere.
#[test]
fn replace_region_writes_each_layout_its_rows_alone() {
    let scratch = Scratch::new("replace-region-layouts");
    let device = device(&scratch.0);
    let mut counted = [0; 3];
    for (size, region, bytes_per_row, bytes_per_image, len) in CASES {
        let texture = texture(&device, size);
        let written = pattern(size, 100);
        let layout = (bytes_per_row, bytes_per_image);
        let mut bytes = vec![UNTOUCHED; len];
        let row = 4 * region.size.width as usize;
        let mut wanted = pattern(size, 0);
        for (texel, at) in rows(size, region, layout) {
            bytes[at..at + row].copy_from_slice(&written[texel..texel + row]);
            wanted[texel..texel + row].copy_from_slice(&written[texel..texel + row]);
        }
        let [host, staged, waited] = paths(&device, || {
            texture
                .replace_region(region, 0, 0, &bytes, bytes_per_row, bytes_per_image)
                .unwrap()
        });
        counted = [counted[0] + host, counted[1] + staged, counted[2] + waited];
        let mut back = vec![0; wanted.len()];
        let whole = Region::new_3d(0, 0, 0, size[0], size[1], size[2]);
        let row = 4 * size[0];
        (texture.get_bytes(&mut back, row, row * size[1], whole, 0, 0)).unwrap();
        assert_eq!(
            back, wanted,
            "{size:?} {region:?}, {bytes_per_row} and {bytes_per_image} apart"
        );
    }
    let [host, staged, waited] = counted;
    eprintln!(
        "{}: {host} of {} writes on the host, {staged} on the staging queue, \
         {waited} internal submissions waited for",
        device.name(),
        CASES.len()
    );
}

/// A two-plane extended-range texture's CPU copies take the path every other texture's take:
/// where the device copies a texture's texels on the host, `replaceRegion` reads the region's
/// texels there, lays its colour words over their first words and writes them back, and
/// `getBytes` reads them there -- no submission, so no wait, as Metal's calls of such a texture
/// wait for no GPU work. Both read each other's bytes: the words a `getBytes` sees are the words
/// the `replaceRegion` wrote, and a region's write leaves every word outside it as it was.
#[test]
fn two_plane_cpu_copies_wait_for_no_submission_where_a_textures_do() {
    let scratch = Scratch::new("two-plane-cpu-copies");
    let device = device(&scratch.0);
    // Whether this device copies a texture's texels on the host at all, which its two-plane
    // textures copy on too (both are `Shared` color images of its own memory).
    let plain = texture(&device, [7, 5, 1]);
    let [plain_host, _, plain_waited] = paths(&device, || {
        let whole = Region::new_3d(0, 0, 0, 7, 5, 1);
        (plain.replace_region(whole, 0, 0, &pattern([7, 5, 1], 3), 28, 140)).unwrap()
    });
    let size = [16, 16, 1];
    let texture = two_plane(&device, [size[0], size[1]]);
    let whole = Region::new_3d(0, 0, 0, size[0], size[1], size[2]);
    let (row, image) = (4 * size[0], 4 * size[0] * size[1]);
    // The whole level's colour words.
    let written = pattern(size, 7);
    let [host, staged, waited] = paths(&device, || {
        (texture.replace_region(whole, 0, 0, &written, row, image)).unwrap()
    });
    let mut back = vec![UNTOUCHED; written.len()];
    let [read_host, read_staged, read_waited] = paths(&device, || {
        (texture.get_bytes(&mut back, row, image, whole, 0, 0)).unwrap()
    });
    assert_eq!(back, written, "the whole level's colour words");
    // Part of the level, the words outside it as they were.
    let part = region([2, 1, 0], [3, 2, 1]);
    let (part_row, words) = (4 * part.size.width, pattern(size, 100));
    let mut bytes = vec![UNTOUCHED; (part_row * part.size.height) as usize];
    let mut wanted = written.clone();
    for (texel, at) in rows(size, part, (part_row, 0)) {
        let span = part_row as usize;
        bytes[at..at + span].copy_from_slice(&words[texel..texel + span]);
        wanted[texel..texel + span].copy_from_slice(&words[texel..texel + span]);
    }
    (texture.replace_region(part, 0, 0, &bytes, part_row, 0)).unwrap();
    let mut back = vec![UNTOUCHED; wanted.len()];
    (texture.get_bytes(&mut back, row, image, whole, 0, 0)).unwrap();
    assert_eq!(back, wanted, "{part:?} alone");
    eprintln!(
        "{}: a two-plane texture's write took {host} copies on the host, {staged} on the staging \
         queue and {waited} internal submissions, its read {read_host}, {read_staged} and \
         {read_waited}; a plain texture's write took {plain_host} on the host and {plain_waited} \
         internal submissions",
        device.name()
    );
    // A device that copies a plain texture's texels on the host copies a two-plane texture's
    // there too, and waits for nothing: the call read the texels and wrote them back through two
    // submissions it waited for before.
    if plain_host > 0 {
        assert!(
            host > 0 && read_host > 0,
            "a two-plane texture's CPU copies left the host path a plain texture's take: \
             either the device has no `HOST_IMAGE_TRANSFER` for the image's `R32G32_UINT`, or \
             the copies regressed onto a submission"
        );
        assert_eq!(
            [waited, read_waited],
            [0, 0],
            "a two-plane texture's CPU copies waited for an internal submission"
        );
    }
}
