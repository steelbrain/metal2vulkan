//! A texture that renders nothing over a buffer whose bytes only the device's ordered commands
//! write (`Buffer::gpu_bytes_ordered`: a `Private` buffer, a `Managed` one's own on a device
//! without unified memory), its rows lying with the buffer's bytes between them
//! (`linear_plan::Between::Kept`):
//!
//! - **Over a `Private` buffer.** An `RGBA8Unorm` texture of 101x48 texels (404 bytes a row) at 416,
//!   448 and 512 bytes a row (the next multiples of 16, 64 and 256, at which drivers lay linear
//!   rows: MoltenVK, llvmpipe and ANV, RADV and NVIDIA), read alone or read and written by shaders,
//!   is the device's linear image, made `PREINITIALIZED`, over the buffer's memory where the device
//!   lays its rows at the pitch: counted (`WorkCounts::kept_padding_aliases`), no shadow
//!   (`shadowed_textures`), answering `NotHostPages`; else a shadow whose answer names its cause,
//!   never `BytesBetweenRows`. At least one pitch takes the image. A blit reads the texels the
//!   buffer holds, and once that blit took the image's first move, the buffer read back holds
//!   every byte it was given, the bytes between the rows among them.
//! - **At the device's linear pitch.** At the pitch `Device::linear_texture_bytes_per_row` answers
//!   for the texture, it is in place over a `Private` buffer; at each other of those pitches, a
//!   shadow.
//! - **A `Managed` texture changed in one row.** On a device without unified memory, over a
//!   `Managed` no-copy buffer that is two copies (`managed_copy.rs`), at the first pitch the
//!   device lays its linear rows at: in place, as over a `Private` buffer. The CPU writes one row
//!   of the caller's pages and `did_modify_range` names it
//!   (`WorkCounts::managed_ranges_modified`, `managed_bytes_modified`); the next blit of the
//!   texture reads that row. A blit writes another row into the texture, with no shadow texel
//!   copied in, and `synchronize_resource` copies its span out (`managed_resources_synchronized`,
//!   `managed_bytes_synchronized`): the caller's pages hold both rows, and every byte between the
//!   rows as given. The same texture over a `Shared` buffer, which the CPU writes at any time, is
//!   a shadow (`BytesBetweenRows`), whose copy of the texels in before a kernel reads them is
//!   counted (`shadow_texels_copied_in`): what the `Managed` texture paid before its route.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. Printed (`KEPT-PADDING`): each pitch's
//! answer and route, and the counts.

use super::blit_holds::{device, Host, Scratch, BYTES, WAIT};
use super::encoder_holds::library;
use crate::{
    BlitCommandEncoder, BlitOption, Buffer, CommandQueue, CpuCacheMode, Device, HazardTrackingMode,
    HostBacking, Origin, PipelineOptions, Region, ResourceOptions, ResourceRef, Size, StorageMode,
    Texture, TextureDescriptor, TexturePlacement, TextureShadowReason,
};
use metal2vulkan::metal::{PixelFormat, TextureUsage};

/// The texture's texels a row, and its rows.
const WIDTH: u64 = 101;
const HEIGHT: u64 = 48;

/// The bytes of a row's texels.
const ROW: u64 = WIDTH * 4;

/// The pitches past its 404-byte rows the texture is made at.
const PITCHES: [u64; 3] = [416, 448, 512];

/// `Managed` storage.
fn managed() -> ResourceOptions {
    ResourceOptions::new(
        StorageMode::Managed,
        CpuCacheMode::DefaultCache,
        HazardTrackingMode::Default,
    )
}

/// `length` bytes that name their place and `seed`, none of them zero.
fn named(length: u64, seed: u64) -> Vec<u8> {
    (0..length)
        .map(|i| ((i * 13 + i / 251 + seed) % 251) as u8 + 1)
        .collect()
}

/// The texture's descriptor, of `options` and `usage`.
fn descriptor(options: ResourceOptions, usage: TextureUsage) -> TextureDescriptor {
    let mut descriptor =
        TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, WIDTH, HEIGHT, false);
    descriptor.usage = usage;
    descriptor.resource_options = options;
    descriptor
}

/// One command buffer of `queue` of the blit commands `encode` encodes, completed.
fn run(queue: &CommandQueue, encode: impl FnOnce(&mut BlitCommandEncoder)) {
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    encode(&mut blit);
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
}

/// The first `length` bytes the CPU maps of `buffer` (a two-copy buffer's: the caller's pages).
fn mapped(buffer: &Buffer, length: u64) -> Vec<u8> {
    let at = buffer.contents().expect("a buffer the CPU maps");
    // SAFETY: the buffer's mapped bytes, at least `length` of them, which no command uses now.
    unsafe { std::slice::from_raw_parts(at.as_ptr(), length as usize) }.to_vec()
}

/// Write `bytes` from byte `at` of the bytes the CPU maps of `buffer`.
fn write(buffer: &Buffer, at: u64, bytes: &[u8]) {
    assert!(at + bytes.len() as u64 <= buffer.length());
    let to = buffer.contents().expect("a buffer the CPU maps");
    // SAFETY: the buffer's mapped bytes hold `at + bytes.len()` bytes (above), which no command
    // uses now.
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), to.as_ptr().add(at as usize), bytes.len())
    };
}

/// `buffer`'s first `length` bytes, through a blit into a `Shared` buffer.
fn read_back(device: &Device, queue: &CommandQueue, buffer: &Buffer, length: u64) -> Vec<u8> {
    let read = device
        .new_buffer(length, ResourceOptions::shared())
        .unwrap();
    run(queue, |blit| {
        blit.copy_from_buffer_to_buffer(buffer, 0, &read, 0, length)
            .unwrap();
    });
    mapped(&read, length)
}

/// The texels of `texture` as a blit copies them into a buffer of rows one after another.
fn blitted(device: &Device, queue: &CommandQueue, texture: &Texture) -> Vec<u8> {
    let read = device
        .new_buffer(ROW * HEIGHT, ResourceOptions::shared())
        .unwrap();
    run(queue, |blit| {
        blit.copy_from_texture_to_buffer(
            texture,
            0,
            0,
            Origin::new(0, 0, 0),
            Size::new(WIDTH, HEIGHT, 1),
            &read,
            0,
            ROW,
            ROW * HEIGHT,
            BlitOption::None,
        )
        .unwrap();
    });
    mapped(&read, ROW * HEIGHT)
}

/// The texels of `texture` as a kernel (`invert_texels`) reads them, one thread a texel, and
/// writes them inverted (alpha kept) into a texture of its own, rows one after another.
fn inverted(device: &Device, queue: &CommandQueue, texture: &Texture) -> Vec<u8> {
    let examples = library(device, "examples/runtime_examples.ll");
    let kernel = examples.new_function_with_name("invert_texels").unwrap();
    let options = PipelineOptions::default();
    let invert = (device.new_compute_pipeline_state_with_function(&kernel, &options)).unwrap();
    let mut plain = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, WIDTH, HEIGHT, false);
    plain.usage = TextureUsage::SHADER_WRITE;
    let written = device.new_texture(&plain).unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(&invert).unwrap();
    encoder.set_texture(Some(texture), 0).unwrap();
    encoder.set_texture(Some(&written), 1).unwrap();
    (encoder.dispatch_threads(Size::new(WIDTH, HEIGHT, 1), Size::new(1, 1, 1))).unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    let mut bytes = vec![0; (ROW * HEIGHT) as usize];
    let region = Region::new_2d(0, 0, WIDTH, HEIGHT);
    (written.get_bytes(&mut bytes, ROW, 0, region, 0, 0)).unwrap();
    bytes
}

/// The texels `bytes` holds at `pitch`, rows one after another.
fn texels_of(bytes: &[u8], pitch: u64) -> Vec<u8> {
    (0..HEIGHT)
        .flat_map(|y| {
            let at = (y * pitch) as usize;
            bytes[at..at + ROW as usize].iter().copied()
        })
        .collect()
}

#[test]
fn a_texture_over_a_private_buffer_holds_its_rows_padding_in_place_and_keeps_it() {
    let scratch = Scratch::new("kept-padding-private");
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let mut in_place = Vec::new();
    for usage in [
        TextureUsage::SHADER_READ,
        TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE,
    ] {
        for pitch in PITCHES {
            let what = format!("{usage:?} at {pitch}");
            let length = pitch * HEIGHT;
            let written = named(length, pitch);
            let staging = device
                .new_buffer_with_bytes(&written, ResourceOptions::shared())
                .unwrap();
            let buffer = device
                .new_buffer(length, ResourceOptions::private())
                .unwrap();
            assert!(buffer.gpu_bytes_ordered(), "{what}: a Private buffer");
            run(&queue, |blit| {
                blit.copy_from_buffer_to_buffer(&staging, 0, &buffer, 0, length)
                    .unwrap();
            });
            let d = descriptor(ResourceOptions::private(), usage);
            let answer = buffer.texture_placement(&d, 0, pitch).unwrap();
            let before = device.work_counts();
            let texture = buffer.new_texture(&d, 0, pitch).unwrap();
            let after = device.work_counts();
            let aliases = texture.aliases_buffer();
            assert_eq!(aliases, texture.shadow().is_none(), "{what}");
            assert_eq!(
                answer == TexturePlacement::NotHostPages,
                aliases,
                "{what}: the answer {answer:?} is the making's"
            );
            assert_ne!(
                answer,
                TexturePlacement::Shadow(TextureShadowReason::BytesBetweenRows),
                "{what}: its padding is no cause"
            );
            assert_eq!(
                after.kept_padding_aliases - before.kept_padding_aliases,
                u64::from(aliases),
                "{what}: counted as an image over kept padding"
            );
            assert_eq!(
                after.shadowed_textures - before.shadowed_textures,
                u64::from(!aliases),
                "{what}: counted as a shadow"
            );
            assert!(
                blitted(&device, &queue, &texture) == texels_of(&written, pitch),
                "{what}: a blit reads the buffer's texels"
            );
            let back = read_back(&device, &queue, &buffer, length);
            let changed = (0..length as usize)
                .filter(|&i| back[i] != written[i])
                .count();
            assert_eq!(changed, 0, "{what}: bytes of the buffer changed");
            if aliases {
                in_place.push(what.clone());
            }
            eprintln!(
                "KEPT-PADDING on {}: {what} over a Private buffer: {answer:?}, in place {aliases}",
                device.name()
            );
        }
    }
    assert!(!in_place.is_empty(), "no pitch took the linear image");
}

#[test]
fn a_texture_over_a_private_buffer_at_the_device_s_linear_pitch_is_in_place() {
    let scratch = Scratch::new("kept-padding-pitch");
    let device = device(&scratch.0);
    for usage in [
        TextureUsage::SHADER_READ,
        TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE,
    ] {
        let d = descriptor(ResourceOptions::private(), usage);
        let Some(pitch) = device.linear_texture_bytes_per_row(&d).unwrap() else {
            panic!("{usage:?}: the device makes no linear image of the texture");
        };
        assert!(pitch >= ROW, "{usage:?}: rows of {pitch} bytes");
        let buffer = device
            .new_buffer(pitch * HEIGHT, ResourceOptions::private())
            .unwrap();
        let texture = buffer.new_texture(&d, 0, pitch).unwrap();
        assert!(texture.aliases_buffer(), "{usage:?}: in place at {pitch}");
        // At every other padded pitch it is a shadow: only the image at the device's own pitch
        // holds the bytes between the rows.
        for other in PITCHES.into_iter().filter(|&other| other != pitch) {
            let buffer = device
                .new_buffer(other * HEIGHT, ResourceOptions::private())
                .unwrap();
            let texture = buffer.new_texture(&d, 0, other).unwrap();
            assert!(!texture.aliases_buffer(), "{usage:?}: in place at {other}");
        }
        eprintln!(
            "KEPT-PADDING on {}: {usage:?}: the device's linear rows lie at {pitch} bytes",
            device.name()
        );
    }
}

#[test]
fn a_managed_texture_changed_in_one_row_reads_it_in_place_where_a_shared_one_copies_its_shadow() {
    let host = Host::new();
    let scratch = Scratch::new("kept-padding-managed");
    let device = device(&scratch.0);
    if device.properties().has_unified_memory {
        eprintln!(
            "KEPT-PADDING on {}: skipped, a Managed buffer is one copy on unified memory",
            device.name()
        );
        return;
    }
    let queue = device.new_command_queue().unwrap();
    let usage = TextureUsage::SHADER_READ;
    // The first pitch at which the texture over a two-copy buffer is in place.
    let mut found = None;
    for pitch in PITCHES {
        let length = pitch * HEIGHT;
        assert!(length as usize <= BYTES);
        // SAFETY: the host memory outlives the buffer, dropped before it, and only it uses them.
        let buffer =
            unsafe { device.new_buffer_with_bytes_no_copy(host.memory(), length, managed(), None) }
                .unwrap();
        if buffer.host_backing() != Some(HostBacking::DeviceCopy) {
            eprintln!(
                "KEPT-PADDING on {}: skipped, the Managed no-copy buffer is no two copies ({:?})",
                device.name(),
                buffer.host_backing()
            );
            return;
        }
        let texture = buffer
            .new_texture(&descriptor(managed(), usage), 0, pitch)
            .unwrap();
        eprintln!(
            "KEPT-PADDING on {}: Managed at {pitch}: in place {}",
            device.name(),
            texture.aliases_buffer()
        );
        if texture.aliases_buffer() {
            found = Some(pitch);
            break;
        }
    }
    let Some(pitch) = found else {
        panic!("no pitch took the linear image over a Managed buffer");
    };
    let length = pitch * HEIGHT;
    let written = named(length, 7);
    let words: Vec<u32> = written
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&word| u32::from_le_bytes(word))
        .collect();
    host.set(0, &words);
    // SAFETY: as above.
    let buffer =
        unsafe { device.new_buffer_with_bytes_no_copy(host.memory(), length, managed(), None) }
            .unwrap();
    let made = device.work_counts();
    let texture = buffer
        .new_texture(&descriptor(managed(), usage), 0, pitch)
        .unwrap();
    let after = device.work_counts();
    assert!(texture.aliases_buffer(), "in place at {pitch}");
    assert_eq!(after.kept_padding_aliases - made.kept_padding_aliases, 1);
    assert!(blitted(&device, &queue, &texture) == texels_of(&written, pitch));

    // The CPU writes row 7's texels; `did_modify_range` names them.
    let mut expected = written.clone();
    let row = (7 * pitch) as usize..(7 * pitch + ROW) as usize;
    expected[row.clone()].copy_from_slice(&named(ROW, 99));
    write(&buffer, row.start as u64, &expected[row.clone()]);
    let before = device.work_counts();
    buffer
        .did_modify_range(row.start as u64..row.end as u64)
        .unwrap();
    let read = blitted(&device, &queue, &texture);
    let after = device.work_counts();
    assert!(read == texels_of(&expected, pitch), "the blit reads row 7");
    let asked = (
        after.managed_ranges_modified - before.managed_ranges_modified,
        after.managed_bytes_modified - before.managed_bytes_modified,
    );
    assert_eq!(asked, (1, ROW), "the range asked");
    let copied_in = after.managed_bytes_copied_in - before.managed_bytes_copied_in;
    let shadow_in = after.shadow_texels_copied_in - before.shadow_texels_copied_in;
    assert_eq!(shadow_in, 0, "no shadow texel copied in");

    // A blit writes row 20 into the texture; `synchronize_resource` copies its span out.
    let row20 = named(ROW, 55);
    let source = device
        .new_buffer_with_bytes(&row20, ResourceOptions::shared())
        .unwrap();
    let synced = device.work_counts();
    run(&queue, |blit| {
        blit.copy_from_buffer_to_texture(
            &source,
            0,
            ROW,
            ROW,
            Size::new(WIDTH, 1, 1),
            &texture,
            0,
            0,
            Origin::new(0, 20, 0),
            BlitOption::None,
        )
        .unwrap();
        blit.synchronize_resource(ResourceRef::Texture(&texture))
            .unwrap();
    });
    let done = device.work_counts();
    let at = (20 * pitch) as usize;
    expected[at..at + ROW as usize].copy_from_slice(&row20);
    let pages = mapped(&buffer, length);
    let changed = (0..length as usize)
        .filter(|&i| pages[i] != expected[i])
        .count();
    assert_eq!(
        changed, 0,
        "bytes of the caller's pages otherwise than expected"
    );
    let synchronized = (
        done.managed_resources_synchronized - synced.managed_resources_synchronized,
        done.managed_bytes_synchronized - synced.managed_bytes_synchronized,
    );
    let span = (HEIGHT - 1) * pitch + ROW;
    assert_eq!(synchronized, (1, span), "the span asked");
    let copied_out = done.managed_bytes_copied_out - synced.managed_bytes_copied_out;
    let written_in = done.shadow_texels_copied_in - synced.shadow_texels_copied_in;
    assert_eq!(
        written_in, 0,
        "no shadow texel copied in for the blit's row"
    );
    drop((texture, buffer));

    // The same over a `Shared` buffer: a shadow, which copies its texels in at the blit; but one
    // in a block of the runtime's own pages takes the route over fresh pages
    // (`memory.rs`'s `Holds::PageBuffers`).
    let shared = device
        .new_buffer(length, ResourceOptions::shared())
        .unwrap();
    if shared.in_page_block() {
        eprintln!(
            "KEPT-PADDING on {}: Managed at {pitch} in place: row 7 asked {asked:?}, copied in \
             {copied_in} bytes, shadow texels copied in {shadow_in}; synchronize_resource asked \
             {synchronized:?}, copied out {copied_out} bytes; the Shared buffer lies in a block \
             of the runtime's own pages",
            device.name()
        );
        return;
    }
    write(&shared, 0, &written);
    let d = descriptor(ResourceOptions::shared(), usage);
    assert_eq!(
        shared.texture_placement(&d, 0, pitch).unwrap(),
        TexturePlacement::Shadow(TextureShadowReason::BytesBetweenRows)
    );
    let shadowed = shared.new_texture(&d, 0, pitch).unwrap();
    assert!(shadowed.shadow().is_some());
    write(&shared, row.start as u64, &expected[row]);
    // A blit reads and writes a texture over a buffer at the buffer's bytes; a kernel reads its
    // shadow, which takes the texels in first.
    let before = device.work_counts();
    let inverted = inverted(&device, &queue, &shadowed);
    let after = device.work_counts();
    let shared_in = after.shadow_texels_copied_in - before.shadow_texels_copied_in;
    assert!(shared_in > 0, "the shadow copies its texels in");
    let row7: Vec<u8> = named(ROW, 99)
        .as_chunks::<4>()
        .0
        .iter()
        .flat_map(|&[r, g, b, a]| [255 - r, 255 - g, 255 - b, a])
        .collect();
    // Within 1: the device's own rounding of the kernel's float answer to unorm.
    let off = (inverted[7 * ROW as usize..8 * ROW as usize].iter())
        .zip(&row7)
        .filter(|(read, wanted)| read.abs_diff(**wanted) > 1)
        .count();
    assert_eq!(off, 0, "bytes of row 7 the kernel reads otherwise");
    eprintln!(
        "KEPT-PADDING on {}: Managed at {pitch} in place: row 7 asked {asked:?}, copied in \
         {copied_in} bytes, shadow texels copied in {shadow_in}; synchronize_resource asked \
         {synchronized:?}, copied out {copied_out} bytes; the Shared texture's shadow copied in \
         {shared_in} texels",
        device.name()
    );
}
