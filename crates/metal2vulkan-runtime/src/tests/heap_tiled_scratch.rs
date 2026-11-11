//! A heap texture held as a shadow of Metal's tiles (`texture/heap_tiled.rs`) moves its elements
//! through one staging buffer that also holds its tight elements and its shapes, made unzeroed:
//! no move reads a byte of it that no move wrote first (`heap_tiled.rs`'s `Scratch`). Here every
//! byte of that buffer is set to a poison word before each use, and none of it reaches the
//! texture or the heap's bytes:
//!
//! - **The first use** (a blit out of the texture) reads the texels the CPU wrote into the heap's
//!   bytes.
//! - **A copy into every texel** (no copy in, the copy back alone) leaves the texels it wrote,
//!   and no poison in the heap's bytes between the elements Metal's tiles leave (a 100 x 70
//!   texture in 64 x 64 tiles), nor past the texture.
//! - **A copy into a part** (copy in, keep, merge back) leaves the part's texels and the others
//!   as they were, and no poison in the heap's bytes.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. A device that holds such a texture
//! otherwise skips them, saying so. Printed (`TILED-SCRATCH`): the buffer poisoned and each step.

use super::blit_holds::{device, Host, Scratch, BYTES, WAIT};
use crate::shadow::{ShadowSource, ShadowTarget};
use crate::{
    BlitOption, Buffer, CommandQueue, HeapDescriptor, HeapType, Origin, Region, ResourceOptions,
    Size, Texture, TextureDescriptor,
};
use metal2vulkan::metal::{PixelFormat, TextureUsage};

/// The texture's sides: neither a whole number of Metal's 64 x 64 tiles of 4-byte texels, so its
/// level's tiles hold bytes no element lies in.
const WIDTH: u64 = 100;
const HEIGHT: u64 = 70;

/// Where the texture lies in the heap.
const OFFSET: u64 = 64 << 10;

/// The byte every byte of the staging buffer is set to before each use.
const POISON: u8 = 0xA5;

/// The word every word of the heap's bytes holds before the texture is made.
const FILL: u32 = 0x3C3C_C3C3;

/// The part a partial copy writes: `PART` texels square from (`AT`, `AT`).
const AT: u64 = 8;
const PART: u64 = 16;

/// `length` bytes that name their place and `seed`, none of them [`POISON`] four in a row.
fn pattern(seed: u8, length: u64) -> Vec<u8> {
    (0..length)
        .map(|i| (i as u8) ^ (i >> 8) as u8 ^ seed)
        .collect()
}

/// Run one blit encoder `encode` records, committed and waited for.
fn blit(queue: &CommandQueue, encode: impl FnOnce(&mut crate::BlitCommandEncoder)) {
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    encode(&mut blit);
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
}

/// Set every byte of `staging` to [`POISON`].
fn poison(queue: &CommandQueue, staging: &Buffer) {
    blit(queue, |blit| {
        blit.fill_buffer(staging, 0..staging.length(), POISON)
            .unwrap();
    });
}

/// The texture's texels as a blit reads them out of its image.
fn blitted(device: &crate::Device, queue: &CommandQueue, texture: &Texture) -> Vec<u8> {
    let out = device
        .new_buffer(WIDTH * HEIGHT * 4, ResourceOptions::shared())
        .unwrap();
    blit(queue, |blit| {
        blit.copy_from_texture_to_buffer(
            texture,
            0,
            0,
            Origin::new(0, 0, 0),
            Size::new(WIDTH, HEIGHT, 1),
            &out,
            0,
            WIDTH * 4,
            WIDTH * HEIGHT * 4,
            BlitOption::None,
        )
        .unwrap();
    });
    let contents = out.contents().unwrap();
    // SAFETY: a `Shared` buffer of that many bytes, which no command writes any more.
    unsafe { std::slice::from_raw_parts(contents.as_ptr(), (WIDTH * HEIGHT * 4) as usize) }.to_vec()
}

/// The texture's texels as `getBytes` reads them out of the heap's bytes.
fn texels(texture: &Texture) -> Vec<u8> {
    let mut read = vec![0u8; (WIDTH * HEIGHT * 4) as usize];
    let region = Region::new_2d(0, 0, WIDTH, HEIGHT);
    texture
        .get_bytes(&mut read, WIDTH * 4, WIDTH * HEIGHT * 4, region, 0, 0)
        .unwrap();
    read
}

/// A blit of `side` texels square of `source` into the texture at (`at`, `at`).
fn copy_in(queue: &CommandQueue, (source, texture): (&Buffer, &Texture), at: u64, side: [u64; 2]) {
    blit(queue, |blit| {
        blit.copy_from_buffer_to_texture(
            source,
            0,
            side[0] * 4,
            side[0] * side[1] * 4,
            Size::new(side[0], side[1], 1),
            texture,
            0,
            0,
            Origin::new(at, at, 0),
            BlitOption::None,
        )
        .unwrap();
    });
}

/// How many words of the texture's span of the heap's bytes hold the poison, and whether every
/// word outside the span holds [`FILL`] still.
fn poisoned_words(host: &Host, span: u64) -> (usize, bool) {
    let words = host.words();
    let (first, last) = ((OFFSET / 4) as usize, ((OFFSET + span) / 4) as usize);
    let word = u32::from_ne_bytes([POISON; 4]);
    let poisoned = words[first..last].iter().filter(|&&w| w == word).count();
    let outside = words[..first]
        .iter()
        .chain(&words[last..])
        .all(|&w| w == FILL);
    (poisoned, outside)
}

#[test]
fn no_byte_of_a_tiled_textures_unzeroed_staging_reaches_the_texture_or_the_heap() {
    let scratch = Scratch::new("heap-tiled-scratch");
    let host = Host::new();
    host.fill(FILL);
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let descriptor =
        HeapDescriptor::new(HeapType::Placement, BYTES as u64, ResourceOptions::shared());
    // SAFETY: the host memory outlives the heap and its texture (dropped after the device), and
    // only their commands and this test's reads use it.
    let heap =
        match unsafe { device.new_heap_with_bytes_no_copy(host.memory(), &descriptor, None, None) }
        {
            Ok(heap) => heap,
            Err(error) => {
                eprintln!("skipped: no heap over host pages here: {error}");
                return;
            }
        };
    let mut surface = TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, WIDTH, HEIGHT, false);
    surface.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    surface.resource_options = ResourceOptions::shared();
    let span = device.heap_texture_size_and_align(&surface).unwrap().size;
    let texture = heap.new_texture_at_offset(&surface, OFFSET).unwrap();
    let staging = match texture.shadow().map(|copies| &copies.source) {
        Some(ShadowSource::Bytes {
            target: ShadowTarget::Tiled { staging, .. },
            ..
        }) => staging.buffer.clone(),
        _ => {
            eprintln!("skipped: the device holds the texture otherwise than in Metal's tiles");
            return;
        }
    };
    let elements = WIDTH * HEIGHT * 4;
    eprintln!(
        "TILED-SCRATCH: a {WIDTH}x{HEIGHT} texture of {span} bytes in Metal's tiles ({elements} \
         of its texels), its staging buffer {} bytes, poisoned with {POISON:#x}",
        staging.length()
    );
    assert!(
        span > elements,
        "Metal's tiles hold bytes no element lies in"
    );

    // The first use reads what the CPU wrote, through the poisoned buffer.
    let written = pattern(1, elements);
    let region = Region::new_2d(0, 0, WIDTH, HEIGHT);
    (texture.replace_region(region, 0, 0, &written, WIDTH * 4, elements)).unwrap();
    poison(&queue, &staging);
    let read = blitted(&device, &queue, &texture);
    let wrong = (read.iter().zip(&written)).filter(|(a, b)| a != b).count();
    eprintln!("TILED-SCRATCH: first use: {wrong} of {elements} bytes read otherwise than written");
    assert_eq!(read, written, "the first use's texels");

    // A copy into every texel: the copy back alone.
    let whole = pattern(2, elements);
    let whole_source = device
        .new_buffer_with_bytes(&whole, ResourceOptions::shared())
        .unwrap();
    poison(&queue, &staging);
    copy_in(&queue, (&whole_source, &texture), 0, [WIDTH, HEIGHT]);
    let (poisoned, outside) = poisoned_words(&host, span);
    eprintln!("TILED-SCRATCH: every texel: {poisoned} poisoned words in the heap's bytes");
    assert_eq!(
        texels(&texture),
        whole,
        "the texels after a copy into every texel"
    );
    assert_eq!(
        poisoned, 0,
        "no poison in the heap's bytes after the copy back"
    );
    assert!(outside, "every heap byte outside the texture as it was");

    // A copy into a part: copy in, keep, merge back.
    let part = pattern(3, PART * PART * 4);
    let part_source = device
        .new_buffer_with_bytes(&part, ResourceOptions::shared())
        .unwrap();
    poison(&queue, &staging);
    copy_in(&queue, (&part_source, &texture), AT, [PART, PART]);
    let mut expected = whole;
    for y in 0..PART {
        let at = (((AT + y) * WIDTH + AT) * 4) as usize;
        let from = (y * PART * 4) as usize;
        expected[at..at + (PART * 4) as usize]
            .copy_from_slice(&part[from..from + (PART * 4) as usize]);
    }
    let (poisoned, outside) = poisoned_words(&host, span);
    eprintln!("TILED-SCRATCH: a part: {poisoned} poisoned words in the heap's bytes");
    assert_eq!(
        texels(&texture),
        expected,
        "the texels after a copy into a part"
    );
    assert_eq!(
        blitted(&device, &queue, &texture),
        expected,
        "the image after a copy into a part"
    );
    assert_eq!(poisoned, 0, "no poison in the heap's bytes after the merge");
    assert!(outside, "every heap byte outside the texture as it was");
    drop(texture);
    drop(heap);
    drop(device);
}
