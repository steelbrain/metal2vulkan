//! A shadowed texture over a `Private` buffer the device made (an IOSurface plane's, rows padded
//! so that no image holds them on any device) keeps its image's standing across command buffers
//! (`shadow_standing.rs`): a later command buffer's first copy in moves nothing where nothing
//! written since met the truth or the image, and every write that did reaches what the texture
//! reads, and every write through the texture reaches the buffer.
//!
//! Each leg checks both halves: the bytes read are the buffer's as every write left them, and
//! the counts say which copies in moved nothing (`WorkCounts::shadow_copies_in_skipped_by_standing`).
//! A standing that never held leaves the bytes right and the counts wrong; one that held over a
//! write leaves the counts right and the bytes wrong.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. Blits alone: no function.

use super::blit_holds::{device, Scratch, WAIT};
use crate::{
    BlitOption, Buffer, CommandQueue, Device, IoSurfacePlane, Origin, ResourceOptions, Size,
    Texture, TextureDescriptor, WorkCounts,
};
use metal2vulkan::metal::{PixelFormat, TextureUsage};

/// The plane: `WIDTH` texels a row, `HEIGHT` rows, rows of `PITCH` bytes, 64 past each row's
/// texels.
const WIDTH: u64 = 64;
const HEIGHT: u64 = 16;
const PITCH: u64 = WIDTH * 4 + 64;
const LENGTH: u64 = PITCH * HEIGHT;

/// The plane's texture over a `Private` buffer, and a `Shared` buffer its reads land in.
struct Plane {
    device: Device,
    queue: CommandQueue,
    buffer: Buffer,
    texture: Texture,
    read: Buffer,
}

impl Plane {
    fn new(device: Device) -> Plane {
        let queue = device.new_command_queue().unwrap();
        let buffer = (device.new_buffer(LENGTH, ResourceOptions::private())).unwrap();
        let mut descriptor =
            TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, WIDTH, HEIGHT, false);
        descriptor.usage = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
        let plane = IoSurfacePlane::new(0, PITCH, true);
        let texture = buffer.new_texture_with_iosurface_plane(&descriptor, plane);
        let texture = texture.unwrap();
        assert!(texture.shadow().is_some(), "padded rows make a shadow");
        let read = (device.new_buffer(LENGTH, ResourceOptions::shared())).unwrap();
        let plane = Plane {
            device,
            queue,
            buffer,
            texture,
            read,
        };
        plane.fill(0..LENGTH, 0x11, &plane.queue);
        plane
    }

    /// A fill of the buffer's `bytes` with `value` in a command buffer of its own on `queue`.
    fn fill(&self, bytes: std::ops::Range<u64>, value: u8, queue: &CommandQueue) {
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.fill_buffer(&self.buffer, bytes, value).unwrap();
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
    }

    /// The texture's texels as a command buffer of its own reads them, row after row, and the
    /// counts it moved.
    fn texels(&self) -> (Vec<u8>, Stood) {
        let before = self.device.work_counts();
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        let size = Size::new(WIDTH, HEIGHT, 1);
        let at = Origin::new(0, 0, 0);
        let row = WIDTH * 4;
        let none = BlitOption::None;
        (blit.copy_from_texture_to_buffer(
            &self.texture,
            0,
            0,
            at,
            size,
            &self.read,
            0,
            row,
            0,
            none,
        ))
        .unwrap();
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        let bytes = self.read_back((WIDTH * 4 * HEIGHT) as usize);
        (bytes, stood(&before, &self.device.work_counts()))
    }

    /// The buffer's own bytes, as a copy into the read buffer moves them.
    fn bytes(&self) -> Vec<u8> {
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        (blit.copy_from_buffer_to_buffer(&self.buffer, 0, &self.read, 0, LENGTH)).unwrap();
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        self.read_back(LENGTH as usize)
    }

    fn read_back(&self, len: usize) -> Vec<u8> {
        let bytes = self.read.contents().unwrap().as_ptr();
        // SAFETY: the read buffer's first `len` bytes, which no command writes now.
        unsafe { std::slice::from_raw_parts(bytes, len) }.to_vec()
    }

    /// A write of `value` into every texel through the texture, from a `Shared` buffer.
    fn write_through(&self, value: u8) {
        let row = WIDTH * 4;
        let source = (self
            .device
            .new_buffer(row * HEIGHT, ResourceOptions::shared()))
        .unwrap();
        let bytes = source.contents().unwrap().as_ptr();
        // SAFETY: the source's bytes, which no command uses yet.
        unsafe { std::ptr::write_bytes(bytes, value, (row * HEIGHT) as usize) };
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        let size = Size::new(WIDTH, HEIGHT, 1);
        let at = Origin::new(0, 0, 0);
        let none = BlitOption::None;
        (blit.copy_from_buffer_to_texture(&source, 0, row, 0, size, &self.texture, 0, 0, at, none))
            .unwrap();
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
    }
}

/// The standing's counts between two snapshots of a device's.
struct Stood {
    against: u64,
    skipped: u64,
    texels: u64,
}

fn stood(before: &WorkCounts, after: &WorkCounts) -> Stood {
    Stood {
        against: after.shadow_copies_in_against_standing - before.shadow_copies_in_against_standing,
        skipped: after.shadow_copies_in_skipped_by_standing
            - before.shadow_copies_in_skipped_by_standing,
        texels: after.shadow_texels_skipped_by_standing - before.shadow_texels_skipped_by_standing,
    }
}

/// Every texel's bytes `value`.
fn all(bytes: &[u8], value: u8) -> bool {
    bytes.iter().all(|&byte| byte == value)
}

/// A second command buffer's read of the texture moves nothing in; a fill of the buffer by
/// another queue's command buffer since lets go of the standing, and the read after it takes the
/// fill; a write through the texture reaches the buffer's bytes, and the read after it holds it.
#[test]
fn a_private_plane_s_image_stands_until_a_command_writes_the_buffer_or_the_texture() {
    let scratch = Scratch::new("shadow-standing");
    let plane = Plane::new(device(&scratch.0));
    let (first, counts) = plane.texels();
    assert!(all(&first, 0x11), "the first read takes the buffer's bytes");
    assert_eq!(counts.against, 1);
    assert_eq!(counts.skipped, 0, "nothing stood yet");
    let (second, counts) = plane.texels();
    assert!(all(&second, 0x11));
    assert_eq!(counts.skipped, 1, "the image stood");
    assert_eq!(counts.texels, WIDTH * HEIGHT, "every texel");
    // A host's transfer queue fills the buffer: the next read copies in again.
    let transfer = plane.device.new_command_queue().unwrap();
    plane.fill(PITCH..3 * PITCH, 0x5A, &transfer);
    let (filled, counts) = plane.texels();
    let row = (WIDTH * 4) as usize;
    assert!(
        all(&filled[row..3 * row], 0x5A),
        "the fill reached the texture"
    );
    assert!(all(&filled[..row], 0x11) && all(&filled[3 * row..], 0x11));
    assert_eq!(counts.skipped, 0, "the fill let go");
    let (_, counts) = plane.texels();
    assert_eq!(counts.skipped, 1, "and it stood again");
    // A write through the texture reaches the buffer, leaving the bytes between rows.
    plane.write_through(0xC3);
    let bytes = plane.bytes();
    for y in 0..HEIGHT as usize {
        let at = y * PITCH as usize;
        assert!(all(&bytes[at..at + row], 0xC3), "row {y}'s texels");
        let gap = if (1..3).contains(&y) { 0x5A } else { 0x11 };
        assert!(
            all(&bytes[at + row..at + PITCH as usize], gap),
            "row {y}'s gap"
        );
    }
    let (written, counts) = plane.texels();
    assert!(all(&written, 0xC3), "the read holds the write");
    assert_eq!(counts.skipped, 1, "the write's end left the image standing");
}

/// A command buffer that fills the buffer and then reads the texture in a later encoder copies
/// in, the fill included; so does every command buffer once the buffer's `VkBuffer` was handed
/// out, which the caller's own work may write.
#[test]
fn a_private_plane_copies_in_after_its_own_command_buffer_s_write_and_once_handed_out() {
    let scratch = Scratch::new("shadow-standing-own");
    let plane = Plane::new(device(&scratch.0));
    let _ = plane.texels();
    let (_, counts) = plane.texels();
    assert_eq!(counts.skipped, 1, "the image stood");
    let before = plane.device.work_counts();
    let command_buffer = plane.queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.fill_buffer(&plane.buffer, 0..PITCH, 0x77).unwrap();
    blit.end_encoding().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    let size = Size::new(WIDTH, HEIGHT, 1);
    let at = Origin::new(0, 0, 0);
    let row = WIDTH * 4;
    let none = BlitOption::None;
    (blit.copy_from_texture_to_buffer(
        &plane.texture,
        0,
        0,
        at,
        size,
        &plane.read,
        0,
        row,
        0,
        none,
    ))
    .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    let counts = stood(&before, &plane.device.work_counts());
    let read = plane.read_back((row * HEIGHT) as usize);
    assert!(
        all(&read[..row as usize], 0x77),
        "the read took its own command buffer's fill"
    );
    assert!(all(&read[row as usize..], 0x11));
    assert_eq!(counts.skipped, 0);
    // SAFETY: the handle is only read; no work of the caller's uses it.
    let _ = unsafe { plane.buffer.vk_buffer() };
    let _ = plane.texels();
    let (_, counts) = plane.texels();
    assert_eq!(counts.against, 0, "handed out, it never stands");
}
