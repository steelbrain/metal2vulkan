//! A texture that renders nothing over a buffer whose bytes only the device's ordered commands
//! write (`Buffer::gpu_bytes_ordered`: a `Private` buffer, a `Managed` one's own on a device
//! without unified memory), its rows lying with the buffer's bytes between them
//! (`linear_plan::Between::Kept`):
//!
//! - **Over a `Private` buffer.** An `RGBA8Unorm` texture of 101x48 texels (404 bytes a row) at 416,
//!   448 and 512 bytes a row (the next multiples of 16, 64 and 256, at which drivers lay linear
//!   rows: MoltenVK, llvmpipe and ANV, RADV and NVIDIA), read alone or read and written by shaders,
//!   is in place over the buffer's memory: the device's linear image, made `PREINITIALIZED`, where
//!   the device lays its rows at the pitch, else the `DRM_FORMAT_MOD_LINEAR` image at the pitch
//!   where the device takes it, whose first move the GPU keeps the bytes through
//!   (`WorkCounts::first_moves_kept`, `first_move_bytes_kept`). In place: counted
//!   (`kept_padding_aliases`), no shadow (`shadowed_textures`), answering `NotHostPages`; else a
//!   shadow whose answer names its cause, never `BytesBetweenRows`. At least one pitch takes an
//!   image. A blit reads the texels the buffer holds, and once that blit took the image's first
//!   move, the buffer read back holds every byte it was given, the bytes between the rows among
//!   them.
//! - **At the device's linear pitch.** At the pitch `Device::linear_texture_bytes_per_row` answers
//!   for the texture, it is in place over a `Private` buffer with no keep; at each other of those
//!   pitches, a shadow, or in place where its first move is kept.
//! - **At a pitch the driver takes only explicitly.** A `BGRA8Unorm` texture of 1760 texels a row
//!   at 7168 bytes a row (7040 its texels'), and one of 96 at 512 (384), both multiples of the
//!   128 bytes NVIDIA aligns linear rows to, neither its own linear pitch: in place on NVIDIA, its
//!   first move kept once, of at least the texels' span (the image's memory size); the texels and
//!   every byte between the rows read back as given. Elsewhere printed, as the route depends on
//!   the driver's answer.
//! - **A `Managed` texture changed in one row.** On a device without unified memory, over a
//!   `Managed` no-copy buffer that is two copies (`managed_copy.rs`), at the first of those
//!   pitches that takes an image: in place, as over a `Private` buffer. The CPU writes one row
//!   of the caller's pages and `did_modify_range` names it
//!   (`WorkCounts::managed_ranges_modified`, `managed_bytes_modified`); the next blit of the
//!   texture reads that row. A blit writes another row into the texture, with no shadow texel
//!   copied in, and `synchronize_resource` copies its span out (`managed_resources_synchronized`,
//!   `managed_bytes_synchronized`): the caller's pages hold both rows, and every byte between the
//!   rows as given. The same texture over a `Shared` buffer, which the CPU writes at any time, is
//!   a shadow (`BytesBetweenRows`), whose copy of the texels in before a kernel reads them is
//!   counted (`shadow_texels_copied_in`): what the `Managed` texture paid before its route.
//!
//! - **In a block of the runtime's buffers allocated exportable.** On a device whose in-place
//!   check answered, so that its buffers lie where they lie for good (in a block of the runtime's
//!   buffers allocated exportable and never exported where the device lays them there,
//!   `memory.rs`'s "Buffers in exportable blocks": the 780M under RADV; else a plain block), under
//!   the validation layer, which reports nothing: a `BGRA8Unorm` texture of 65x8 texels (260 bytes
//!   a row) at the device's own linear pitch over a `Private` buffer. One that only reads is in
//!   place, counted among the images over kept padding, its image in an exportable block the
//!   device's linear image made `UNDEFINED` for the block's handle type, whose first move keeps
//!   the bytes it is bound to (`first_moves_kept`); one that renders too, in place where the
//!   device's check of renders over its pooled memory passed for texels that lie as these do, a
//!   shadow where it passed for neither kind of rows. After a first use (a blit's read, a render
//!   pass's clear, a render pass's load) the buffer holds every byte between the rows as given,
//!   and its texels as given or cleared. And in the reims-vgpu consumer's order: one that renders over a
//!   buffer that declares its padding unobserved (`BufferDescriptor::padding_unobserved`), both
//!   let go, then one over a buffer made as any other, counted as over the latter alone.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. Printed (`KEPT-PADDING`,
//! `KEPT-PADDING-BLOCKS`): each pitch's answer and route, and the counts.

use super::blit_holds::{device, Host, Scratch, BYTES, WAIT};
use super::encoder_holds::library;
use super::layer_claim::Watched;
use crate::in_place::{Access, PagesClass, RowEnds, Verdict};
use crate::memory::BoundMemory;
use crate::{
    BlitCommandEncoder, BlitOption, Buffer, BufferContents, BufferDescriptor, ClearColor,
    CommandQueue, CpuCacheMode, Device, HazardTrackingMode, HostBacking, LoadAction, Origin,
    PipelineOptions, Region, RenderPassDescriptor, ResourceOptions, ResourceRef, Size, StorageMode,
    StoreAction, Texture, TextureDescriptor, TexturePlacement, TextureShadowReason, WorkCounts,
};
use metal2vulkan::device::HostPageTexturePolicy;
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

/// The texels of `texture` (of 4 bytes a texel) as a blit copies them into a buffer of rows one
/// after another.
fn blitted(device: &Device, queue: &CommandQueue, texture: &Texture) -> Vec<u8> {
    let (width, height) = (texture.width(), texture.height());
    let row = width * 4;
    let read = device
        .new_buffer(row * height, ResourceOptions::shared())
        .unwrap();
    run(queue, |blit| {
        blit.copy_from_texture_to_buffer(
            texture,
            0,
            0,
            Origin::new(0, 0, 0),
            Size::new(width, height, 1),
            &read,
            0,
            row,
            row * height,
            BlitOption::None,
        )
        .unwrap();
    });
    mapped(&read, row * height)
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
    rows_of(bytes, pitch, ROW, HEIGHT)
}

/// The `height` rows of `row` bytes `bytes` holds at `pitch`, one after another.
fn rows_of(bytes: &[u8], pitch: u64, row: u64, height: u64) -> Vec<u8> {
    (0..height)
        .flat_map(|y| {
            let at = (y * pitch) as usize;
            bytes[at..at + row as usize].iter().copied()
        })
        .collect()
}

/// The keeps a first move made between `before` and `after`, and their bytes.
fn kept(before: &WorkCounts, after: &WorkCounts) -> (u64, u64) {
    (
        after.first_moves_kept - before.first_moves_kept,
        after.first_move_bytes_kept - before.first_move_bytes_kept,
    )
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
            let own = device.linear_texture_bytes_per_row(&d).unwrap();
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
            let moved = device.work_counts();
            assert!(
                blitted(&device, &queue, &texture) == texels_of(&written, pitch),
                "{what}: a blit reads the buffer's texels"
            );
            let keeps = kept(&moved, &device.work_counts());
            // The device's own linear image takes no keep; the explicit-pitch one takes one.
            let keep = aliases && own != Some(pitch);
            assert_eq!(
                keeps.0,
                u64::from(keep),
                "{what}: first moves kept {keeps:?}"
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
                "KEPT-PADDING on {}: {what} over a Private buffer: {answer:?}, in place {aliases}, \
                 the device's own pitch {own:?}, kept {keeps:?}",
                device.name()
            );
        }
    }
    assert!(!in_place.is_empty(), "no pitch took an image");
}

#[test]
fn a_texture_over_a_private_buffer_at_the_device_s_linear_pitch_is_in_place() {
    let scratch = Scratch::new("kept-padding-pitch");
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
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
        let moved = device.work_counts();
        blitted(&device, &queue, &texture);
        let keeps = kept(&moved, &device.work_counts());
        assert_eq!(keeps.0, 0, "{usage:?}: at {pitch}, kept {keeps:?}");
        // At every other padded pitch it is a shadow, or in place where the device takes the
        // pitch explicitly: only the image at the device's own pitch holds the bytes between the
        // rows through its first move, and the other's first move keeps them.
        let mut others = Vec::new();
        for other in PITCHES.into_iter().filter(|&other| other != pitch) {
            let buffer = device
                .new_buffer(other * HEIGHT, ResourceOptions::private())
                .unwrap();
            let texture = buffer.new_texture(&d, 0, other).unwrap();
            let moved = device.work_counts();
            blitted(&device, &queue, &texture);
            let keeps = kept(&moved, &device.work_counts());
            let aliases = texture.aliases_buffer();
            assert_eq!(
                keeps.0,
                u64::from(aliases),
                "{usage:?}: at {other}, kept {keeps:?}"
            );
            others.push((other, aliases));
        }
        eprintln!(
            "KEPT-PADDING on {}: {usage:?}: the device's linear rows lie at {pitch} bytes; \
             in place, kept, at the others {others:?}",
            device.name()
        );
    }
}

#[test]
fn a_texture_over_a_private_buffer_at_a_pitch_the_driver_takes_only_explicitly_is_kept_in_place() {
    // Texels a row, bytes a row, rows.
    const SHAPES: [(u64, u64, u64); 2] = [(1760, 7168, 16), (96, 512, 64)];
    let scratch = Scratch::new("kept-padding-explicit");
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let nvidia = device.properties().vendor_id == 0x10de;
    for (width, pitch, height) in SHAPES {
        let row = width * 4;
        let what = format!("{width} texels at {pitch} ({row})");
        let length = pitch * height;
        let written = named(length, width);
        let staging = device
            .new_buffer_with_bytes(&written, ResourceOptions::shared())
            .unwrap();
        let buffer = device
            .new_buffer(length, ResourceOptions::private())
            .unwrap();
        run(&queue, |blit| {
            blit.copy_from_buffer_to_buffer(&staging, 0, &buffer, 0, length)
                .unwrap();
        });
        let mut d = TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, width, height, false);
        d.usage = TextureUsage::SHADER_READ;
        d.resource_options = ResourceOptions::private();
        let own = device.linear_texture_bytes_per_row(&d).unwrap();
        let answer = buffer.texture_placement(&d, 0, pitch).unwrap();
        let texture = buffer.new_texture(&d, 0, pitch).unwrap();
        let aliases = texture.aliases_buffer();
        assert_eq!(
            aliases,
            answer == TexturePlacement::NotHostPages,
            "{what}: {answer:?}"
        );
        let moved = device.work_counts();
        assert!(
            blitted(&device, &queue, &texture) == rows_of(&written, pitch, row, height),
            "{what}: a blit reads the buffer's texels"
        );
        let keeps = kept(&moved, &device.work_counts());
        let back = read_back(&device, &queue, &buffer, length);
        let changed = (0..length as usize)
            .filter(|&i| back[i] != written[i])
            .count();
        assert_eq!(changed, 0, "{what}: bytes of the buffer changed");
        eprintln!(
            "KEPT-PADDING on {}: {what} over a Private buffer: {answer:?}, in place {aliases}, the \
             device's own pitch {own:?}, kept {keeps:?}",
            device.name()
        );
        if aliases && own != Some(pitch) {
            let span = (height - 1) * pitch + row;
            assert_eq!(keeps.0, 1, "{what}: one first move kept");
            assert!(
                keeps.1 >= span,
                "{what}: kept {} bytes, the texels' span {span}",
                keeps.1
            );
        }
        if nvidia {
            assert!(aliases, "{what}: in place on NVIDIA");
            assert_ne!(own, Some(pitch), "{what}: not NVIDIA's own pitch");
        }
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

/// What [`OnBlocks`]'s tests print as.
const BLOCKS: &str = "KEPT-PADDING-BLOCKS";

/// The texels a row of [`OnBlocks`]'s textures (`BGRA8Unorm`, 260 bytes a row: a row 4 bytes past
/// the 256 bytes RADV aligns linear rows to), and their rows.
const WIDE: u64 = 65;
const HIGH: u64 = 8;

/// A device of the kind the cfg selects under the validation layer, its in-place check answered,
/// so that its buffers lie where they lie for good, and what the tests over it read of it.
struct OnBlocks {
    device: Device,
    watched: Watched,
    _scratch: Scratch,
    name: String,
    /// The device's buffers lie in blocks of the runtime's buffers allocated exportable.
    blocks: bool,
    /// Its check's verdicts of renders over its pooled memory, for texels in lines and on them.
    renders: [Verdict; 2],
    /// The rows' pitch: the device's own linear pitch for the texture, past its 260 bytes.
    pitch: u64,
}

impl OnBlocks {
    /// The device, its scratch named `leg`.
    fn new(leg: &str) -> OnBlocks {
        let watched = Watched::default();
        let scratch = Scratch::new(leg);
        let device = super::pending_rebind::device(&watched, &scratch.0);
        assert!(device.settle_in_place_checks(WAIT), "{BLOCKS}: the check");
        let name = device.name().to_owned();
        let exportable = (crate::device::lock(&device.shared().memory).fd_buffer_types).is_ok();
        let blocks = exportable && device.fd_blocks_taken();
        let renders = RowEnds::ALL
            .map(|rows| device.in_place_verdict(PagesClass::Pooled, Access::Renders, rows));
        let read = OnBlocks::descriptor(TextureUsage::SHADER_READ);
        let Some(pitch) = device.linear_texture_bytes_per_row(&read).unwrap() else {
            panic!("{BLOCKS} on {name}: the device makes no linear image of the texture");
        };
        assert!(
            pitch > WIDE * 4,
            "{BLOCKS} on {name}: rows of {pitch} bytes hold no padding"
        );
        eprintln!(
            "{BLOCKS} on {name}: rows {pitch} bytes apart; the device's buffers in exportable \
             blocks {blocks}; renders over pooled memory {renders:?} (in lines, on lines)"
        );
        OnBlocks {
            device,
            watched,
            _scratch: scratch,
            name,
            blocks,
            renders,
            pitch,
        }
    }

    /// The texture's descriptor, `Private`, of `usage`.
    fn descriptor(usage: TextureUsage) -> TextureDescriptor {
        let mut d = TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, WIDE, HIGH, false);
        d.usage = usage;
        d.resource_options = ResourceOptions::private();
        d
    }

    /// Whether the device's check of renders over its pooled memory passed for both kinds of
    /// rows, under a policy that takes images in place: a texture that renders over kept padding
    /// is then in place.
    fn renders_kept(&self) -> bool {
        let policy = self.device.compile_device().policy().host_page_textures;
        policy == HostPageTexturePolicy::InPlace && self.renders == [Verdict::Passed; 2]
    }

    /// Whether it passed for neither: such a texture is then a shadow.
    fn renders_refused(&self) -> bool {
        !self.renders.contains(&Verdict::Passed)
    }
}

/// How the first use of a texture of
/// `a_texture_over_a_private_buffer_in_a_block_of_its_own_kind_keeps_its_rows_padding` moves its
/// image.
#[derive(Clone, Copy, Debug)]
enum FirstUse {
    /// A blit reads its texels.
    Blit,
    /// A render pass clears and stores it.
    Clear,
    /// A render pass loads and stores it.
    Load,
}

#[test]
fn a_texture_over_a_private_buffer_in_a_block_of_its_own_kind_keeps_its_rows_padding() {
    const CLEAR: [u8; 4] = [0, 255, 0, 255];
    let on = OnBlocks::new("kept-padding-blocks");
    let (device, name, pitch) = (&on.device, &on.name, on.pitch);
    let queue = device.new_command_queue().unwrap();
    let read = TextureUsage::SHADER_READ;
    let rendered = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
    let row = WIDE * 4;
    let length = pitch * HIGH;
    for (usage, first) in [
        (read, FirstUse::Blit),
        (rendered, FirstUse::Clear),
        (rendered, FirstUse::Load),
    ] {
        let what = format!("{first:?}");
        let written = named(length, pitch + first as u64);
        let staging = device
            .new_buffer_with_bytes(&written, ResourceOptions::shared())
            .unwrap();
        let buffer = device
            .new_buffer(length, ResourceOptions::private())
            .unwrap();
        run(&queue, |blit| {
            blit.copy_from_buffer_to_buffer(&staging, 0, &buffer, 0, length)
                .unwrap();
        });
        let bound = buffer.bound_memory();
        let in_block = matches!(bound, BoundMemory::Blocks);
        assert_eq!(
            in_block, on.blocks,
            "{BLOCKS} on {name}: {what}: the buffer lies in {bound:?}"
        );
        let d = OnBlocks::descriptor(usage);
        let answer = buffer.texture_placement(&d, 0, pitch).unwrap();
        let before = device.work_counts();
        let texture = buffer.new_texture(&d, 0, pitch).unwrap();
        let made = device.work_counts();
        let aliases = texture.aliases_buffer();
        let counted = (
            made.kept_padding_aliases - before.kept_padding_aliases,
            made.shadowed_textures - before.shadowed_textures,
        );
        eprintln!(
            "{BLOCKS} on {name}: {what}: over {bound:?}: {answer:?}, in place {aliases}, counted \
             (kept, shadowed) {counted:?}"
        );
        assert_eq!(
            answer == TexturePlacement::NotHostPages,
            aliases,
            "{BLOCKS} on {name}: {what}: the answer {answer:?} is the making's"
        );
        assert_eq!(
            counted,
            (u64::from(aliases), u64::from(!aliases)),
            "{BLOCKS} on {name}: {what}: counted (kept, shadowed)"
        );
        match usage == read {
            true => assert!(aliases, "{BLOCKS} on {name}: {what}: {answer:?}"),
            false if on.renders_kept() => {
                assert!(aliases, "{BLOCKS} on {name}: {what}: {answer:?}")
            }
            false if on.renders_refused() => assert!(!aliases, "{BLOCKS} on {name}: {what}"),
            false => {}
        }
        let expected = match first {
            FirstUse::Blit => {
                assert!(
                    blitted(device, &queue, &texture) == rows_of(&written, pitch, row, HIGH),
                    "{BLOCKS} on {name}: {what}: a blit reads the buffer's texels"
                );
                written.clone()
            }
            FirstUse::Clear | FirstUse::Load => {
                let mut pass = RenderPassDescriptor::new();
                let color = &mut pass.color_attachments[0];
                color.texture = Some(texture.clone());
                color.store_action = StoreAction::Store;
                color.load_action = match first {
                    FirstUse::Clear => LoadAction::Clear,
                    _ => LoadAction::Load,
                };
                color.clear_color = ClearColor::new(0.0, 1.0, 0.0, 1.0);
                let command_buffer = queue.command_buffer().unwrap();
                (command_buffer.render_command_encoder(&pass).unwrap())
                    .end_encoding()
                    .unwrap();
                command_buffer.commit().unwrap();
                command_buffer.wait_until_completed_for(WAIT).unwrap();
                assert_eq!(command_buffer.error(), None, "{BLOCKS} on {name}: {what}");
                let mut expected = written.clone();
                if let FirstUse::Clear = first {
                    for y in 0..HIGH {
                        let at = (y * pitch) as usize;
                        for texel in expected[at..at + row as usize].chunks_mut(4) {
                            texel.copy_from_slice(&CLEAR);
                        }
                    }
                }
                expected
            }
        };
        let keeps = kept(&made, &device.work_counts());
        eprintln!("{BLOCKS} on {name}: {what}: first moves kept {keeps:?}");
        // The device's own linear image over a plain block starts `PREINITIALIZED` and takes no
        // keep; made for an exportable block's handle type, it starts `UNDEFINED` and takes one.
        assert_eq!(
            keeps.0,
            u64::from(aliases && in_block),
            "{BLOCKS} on {name}: {what}: first moves kept {keeps:?}"
        );
        let back = read_back(device, &queue, &buffer, length);
        let between = (0..length as usize)
            .filter(|&i| (i as u64 % pitch) >= row && back[i] != written[i])
            .count();
        assert_eq!(
            between, 0,
            "{BLOCKS} on {name}: {what}: bytes between the rows changed"
        );
        let texels = (0..length as usize)
            .filter(|&i| (i as u64 % pitch) < row && back[i] != expected[i])
            .count();
        assert_eq!(texels, 0, "{BLOCKS} on {name}: {what}: texels read back");
    }
    on.watched.layer_reported_nothing(BLOCKS);
}

#[test]
fn a_texture_that_renders_over_a_plain_private_buffer_after_one_over_an_unobserved_one_is_kept() {
    let on = OnBlocks::new("kept-padding-after-unobserved");
    let (device, name, pitch) = (&on.device, &on.name, on.pitch);
    let length = pitch * HIGH;
    let d = OnBlocks::descriptor(TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET);
    // The counts a texture over `buffer` moves, the texture and the buffer let go after.
    let texture_over = |buffer: Buffer| {
        let before = device.work_counts();
        let texture = buffer.new_texture(&d, 0, pitch).unwrap();
        let after = device.work_counts();
        drop(texture);
        (
            after.unobserved_padding_aliases - before.unobserved_padding_aliases,
            after.shadowed_textures - before.shadowed_textures,
            after.kept_padding_aliases - before.kept_padding_aliases,
        )
    };
    let unobserved =
        BufferDescriptor::new(length, ResourceOptions::private(), BufferContents::Zero)
            .with_padding_unobserved();
    let declared = texture_over(device.new_buffer_with_descriptor(unobserved).unwrap());
    let plain = device
        .new_buffer(length, ResourceOptions::private())
        .unwrap();
    let bound = plain.bound_memory();
    let plain = texture_over(plain);
    eprintln!(
        "{BLOCKS} on {name}: after a buffer that declares its padding unobserved {declared:?}, \
         over one made as any other, in {bound:?}, {plain:?} (unobserved, shadowed, kept)"
    );
    if on.renders_kept() {
        assert_eq!(plain, (0, 0, 1), "{BLOCKS} on {name}: renders passed");
    }
    if on.renders_refused() {
        assert_eq!(plain, (0, 1, 0), "{BLOCKS} on {name}: renders refused");
    }
    on.watched.layer_reported_nothing(BLOCKS);
}
