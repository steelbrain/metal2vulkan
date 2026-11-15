//! A shadowed texture read from two encoders of one command buffer: the second encoder reads the
//! image the first left holding the truth and copies nothing in (`shadow::ShadowCarry`).
//!
//! Metal shows an encoder the bytes as of its command buffer's commit (`shadow.rs`'s copy-in
//! rule), so the second encoder's copy in would move texels the image holds already. This checks
//! both halves of leaving it out on a device: the texels the second encoder read are the buffer's,
//! and the counters say its copy in was left out. A carry that stopped firing leaves the texel
//! comparison passing and the counts wrong; one that fires over an image Vulkan never ordered or
//! left in another layout leaves the counts right and the texels wrong.
//!
//! The texture is [`WIDTH`]x[`HEIGHT`] over a `Shared` buffer, rows of [`ROW`] bytes (not a
//! multiple of 256, so a shadow wherever the device lays no image's rows so), its texels' channels
//! 0 or 255 ([`texel`]) so that `invert_texels`' writes are exact (Vulkan converts 0.0 and 1.0 to
//! 0 and 255 exactly). Two compute encoders of one command buffer each dispatch `invert_texels`
//! over every texel, reading the shadowed texture and writing a target texture of its own: both
//! targets must hold every texel inverted. A binding bounds nothing it reads, so the first copy in
//! moves every texel, and the second encoder's would move them again.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The function is the owned fixture
//! `validation/fixtures/public/examples/runtime_examples.ll` (`llvm-as`).

use super::blit_holds::{device, Scratch, WAIT};
use crate::{
    CommandBufferStatus, ComputePipelineState, Device, Library, PipelineOptions, Region,
    ResourceOptions, Size, Texture, TextureDescriptor,
};
use metal2vulkan::metal::{PixelFormat, TextureUsage};

/// The texture: `WIDTH` texels a row, `HEIGHT` rows, rows of `ROW` bytes.
const WIDTH: u64 = 68;
const HEIGHT: u64 = 5;
const ROW: u64 = 352;
/// The bytes between the rows, which no copy moves.
const GAP: u8 = 0xee;

/// Texel `(x, y)`'s bytes, memory order: byte `j` is 255 where bit `j` of `x + 7 y` is set, else
/// 0, so every texel's channels are exactly representable both ways.
fn texel(x: u64, y: u64) -> [u8; 4] {
    let v = x + 7 * y;
    [0, 1, 2, 3].map(|j| if (v >> j) & 1 == 1 { 255 } else { 0 })
}

/// What `invert_texels` writes from texel `(x, y)`: its three colour channels inverted and its
/// alpha kept, memory order (alpha is byte 3 of `RGBA8Unorm`).
fn inverted(x: u64, y: u64) -> [u8; 4] {
    let t = texel(x, y);
    [255 - t[0], 255 - t[1], 255 - t[2], t[3]]
}

/// The `invert_texels` pipeline of the owned fixture.
fn invert(device: &Device) -> ComputePipelineState {
    let ll = metal2vulkan_devtools::fixtures_dir().join("examples/runtime_examples.ll");
    let bitcode = metal2vulkan_devtools::llvm_assemble(&ll).unwrap();
    let library = metal2vulkan::library::Library::from_air(&bitcode).unwrap();
    let library: Library = device.new_library_with_compile_library(library);
    let function = library.new_function_with_name("invert_texels").unwrap();
    device
        .new_compute_pipeline_state_with_function(&function, &PipelineOptions::default())
        .unwrap()
}

/// A `WIDTH`x`HEIGHT` `Shared` target texture a dispatch writes and the host reads.
fn target(device: &Device) -> Texture {
    let mut descriptor =
        TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, WIDTH, HEIGHT, false);
    descriptor.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
    descriptor.resource_options = ResourceOptions::shared();
    device.new_texture(&descriptor).unwrap()
}

/// `texture`'s texels, row after row.
fn texels_of(texture: &Texture) -> Vec<u8> {
    let mut bytes = vec![0u8; (WIDTH * HEIGHT * 4) as usize];
    texture
        .get_bytes(
            &mut bytes,
            WIDTH * 4,
            0,
            Region::new_2d(0, 0, WIDTH, HEIGHT),
            0,
            0,
        )
        .unwrap();
    bytes
}

/// The second encoder of a command buffer reads the shadowed texture's image as the first left it,
/// taking no copy in (the module docs).
#[test]
fn a_second_encoder_reads_the_image_the_first_left_and_copies_nothing_in() {
    let scratch = Scratch::new("shadow-carried-read");
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let span = (ROW * HEIGHT) as usize;
    let buffer = device
        .new_buffer(ROW * HEIGHT, ResourceOptions::shared())
        .unwrap();
    let bytes = buffer.contents().unwrap().as_ptr();
    let mut laid = vec![GAP; span];
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let at = (y * ROW + x * 4) as usize;
            laid[at..at + 4].copy_from_slice(&texel(x, y));
        }
    }
    // SAFETY: the buffer's bytes, which no command uses now.
    unsafe { std::ptr::copy_nonoverlapping(laid.as_ptr(), bytes, span) };

    // Before the texture: its shadow is made as the texture is (`WorkCounts::shadowed_textures`),
    // so a snapshot after it would read no shadow made and say nothing about whether this leg
    // reaches the carry at all.
    let counts = device.work_counts();
    let mut descriptor =
        TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, WIDTH, HEIGHT, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    descriptor.resource_options = buffer.resource_options();
    let source = buffer.new_texture(&descriptor, 0, ROW).unwrap();
    let first = target(&device);
    let second = target(&device);
    let invert = invert(&device);
    let grid = Size::new(WIDTH, HEIGHT, 1);
    let group = Size::new(4, 1, 1);

    let command_buffer = queue.command_buffer().unwrap();
    for into in [&first, &second] {
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(&invert).unwrap();
        encoder.set_texture(Some(&source), 0).unwrap();
        encoder.set_texture(Some(into), 1).unwrap();
        encoder.dispatch_threads(grid, group).unwrap();
        encoder.end_encoding().unwrap();
    }
    command_buffer.commit().unwrap();
    assert!(
        command_buffer.wait_until_completed_for(WAIT).is_ok(),
        "the command buffer never completed"
    );
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    let after = device.work_counts();

    // Both encoders read the buffer's texels, the second through the image the first left.
    let mut want = vec![0u8; (WIDTH * HEIGHT * 4) as usize];
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let at = ((y * WIDTH + x) * 4) as usize;
            want[at..at + 4].copy_from_slice(&inverted(x, y));
        }
    }
    for (which, into) in [("first", &first), ("second", &second)] {
        let read = texels_of(into);
        let differing: Vec<usize> = (0..read.len()).filter(|&i| read[i] != want[i]).collect();
        assert!(
            differing.is_empty(),
            "the {which} encoder read {} wrong bytes, the first at {:?}",
            differing.len(),
            &differing[..differing.len().min(16)],
        );
    }

    // The texels the whole command buffer copied in, and the texels of its first copies in that
    // an earlier encoder of it had copied in too.
    let shadowed = after.shadowed_textures - counts.shadowed_textures;
    let copied_in = after.shadow_texels_copied_in - counts.shadow_texels_copied_in;
    let carried =
        after.shadow_texels_first_copy_in_carried - counts.shadow_texels_first_copy_in_carried;
    let again =
        after.shadow_texels_first_copied_in_again - counts.shadow_texels_first_copied_in_again;
    let back = after.shadow_texels_copied_back - counts.shadow_texels_copied_back;
    eprintln!(
        "SHADOW-CARRIED-READ: {shadowed} shadow made, {copied_in} texels copied in, \
         {carried} carried, {again} copied in again, {back} copied back"
    );
    // The device lays an image's rows as the buffer's, so the texture is its own bytes and
    // nothing is copied: the texels checked above are the whole of this leg, and it says nothing
    // about the carry. Fail rather than pass quietly, so a stack this leg cannot reach reads as
    // that and not as the carry proved.
    assert!(
        shadowed > 0,
        "this device lays the texture's rows as the buffer's, so it took no shadow and the carry \
         was never reached: the leg proves nothing here"
    );
    assert_eq!(shadowed, 1, "one shadow for the source texture");
    // The first encoder moved every texel in; the second's first use found its command buffer
    // already holding the image and moved none, which is the count the carry is sized by.
    assert_eq!(
        copied_in,
        WIDTH * HEIGHT,
        "the texture's texels copied in once for the command buffer, not once per encoder"
    );
    assert_eq!(
        carried,
        WIDTH * HEIGHT,
        "the second encoder's copy in was not the one the carry took off"
    );
    // The texture is a mapped buffer's and far under 2^20 texels, so none of the three figures
    // that divide a large IOSurface plane's first copies in moves for it.
    let large = [
        after.shadow_texels_first_copy_in_carried_large_surfaces
            - counts.shadow_texels_first_copy_in_carried_large_surfaces,
        after.shadow_texels_first_copied_in_large_surfaces_stood_earlier
            - counts.shadow_texels_first_copied_in_large_surfaces_stood_earlier,
        after.shadow_texels_first_copied_in_large_surfaces_unstood
            - counts.shadow_texels_first_copied_in_large_surfaces_unstood,
    ];
    assert_eq!(
        large,
        [0, 0, 0],
        "a small mapped buffer's texture counted large"
    );
    assert_eq!(again, 0, "an encoder copied in what an earlier one had");
    assert_eq!(back, 0, "a texture only read was copied back");
}
