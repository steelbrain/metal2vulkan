//! The over-copies by cause: each counted copy of a texture over a buffer counts in the `Metrics`
//! pair of the cause the texture's making decided, and in no other. Device-free, each cause's copy
//! moves its own pair alone. Under a device cfg, on every device of the cfg's kind, four RGBA8Unorm
//! textures over a buffer, each read once by a blit in a command buffer of its own: one whose rows
//! lie one linear rounding past the device's own pitch, one a texel off its linear image's memory
//! alignment, a render target of 3 rows whose subresource the device may advertise past them, and
//! one over host memory's import (where the device imports host memory). A copied one moves the
//! totals by its one copy in, its texels' bytes, and its cause's pair alike, every other cause's
//! pair still; one in place moves nothing. On lavapipe each is copied for the cause it was made
//! for: pitch, alignment, render reach and host memory. lavapipe's linear tiling has its optimal
//! tiling's features for every color format (Mesa's `lvp_formats.c`), so no texture here is
//! copied for a format or usage.

use crate::counters::{Counters, OverCause};
use crate::Metrics;

/// Every cause.
const CAUSES: [OverCause; 12] = [
    OverCause::HostMemory,
    OverCause::DmaBuf,
    OverCause::DmaBufs,
    OverCause::Dedicated,
    OverCause::Surface,
    OverCause::Spilled,
    OverCause::Linear,
    OverCause::LinearFeatures,
    OverCause::Pitch,
    OverCause::RenderReach,
    OverCause::Alignment,
    OverCause::ImageMemory,
];

/// `cause`'s pair of `metrics`: its copies and their bytes.
fn pair(metrics: &Metrics, cause: OverCause) -> (u64, u64) {
    let m = metrics;
    match cause {
        OverCause::HostMemory => (m.over_copies_host_memory, m.over_copy_bytes_host_memory),
        OverCause::DmaBuf => (m.over_copies_dma_buf, m.over_copy_bytes_dma_buf),
        OverCause::DmaBufs => (m.over_copies_dma_bufs, m.over_copy_bytes_dma_bufs),
        OverCause::Dedicated => (m.over_copies_dedicated, m.over_copy_bytes_dedicated),
        OverCause::Surface => (m.over_copies_surface, m.over_copy_bytes_surface),
        OverCause::Spilled => (m.over_copies_spilled, m.over_copy_bytes_spilled),
        OverCause::Linear => (m.over_copies_linear, m.over_copy_bytes_linear),
        OverCause::LinearFeatures => (
            m.over_copies_linear_features,
            m.over_copy_bytes_linear_features,
        ),
        OverCause::Pitch => (m.over_copies_pitch, m.over_copy_bytes_pitch),
        OverCause::RenderReach => (m.over_copies_render_reach, m.over_copy_bytes_render_reach),
        OverCause::Alignment => (m.over_copies_alignment, m.over_copy_bytes_alignment),
        OverCause::ImageMemory => (m.over_copies_image_memory, m.over_copy_bytes_image_memory),
    }
}

/// Each cause's pair moved from `before` to `after`, in [`CAUSES`] order.
fn moved(before: &Metrics, after: &Metrics) -> Vec<(OverCause, (u64, u64))> {
    CAUSES
        .iter()
        .map(|&cause| {
            let (from, to) = (pair(before, cause), pair(after, cause));
            (cause, (to.0 - from.0, to.1 - from.1))
        })
        .collect()
}

/// One copy of each cause, of a cause's own byte count, moves that cause's pair alone, and the
/// totals by all of them.
#[test]
fn each_cause_counts_in_its_own_pair_alone() {
    for (at, &cause) in CAUSES.iter().enumerate() {
        let counters = Counters::new();
        let before = counters.metrics();
        let bytes = 64 * (at as u64 + 1);
        counters.over_copied(bytes, false, cause);
        let after = counters.metrics();
        for (other, pair) in moved(&before, &after) {
            let expected = match other == cause {
                true => (1, bytes),
                false => (0, 0),
            };
            assert_eq!(pair, expected, "{cause:?}'s copy in {other:?}'s pair");
        }
        assert_eq!(
            (after.over_copies, after.over_copy_bytes, after.bytes_copied),
            (1, bytes, bytes),
            "{cause:?}"
        );
        assert_eq!(after.over_copies_imported, 0, "{cause:?}");
    }
}

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use super::moved;
    use crate::counters::OverCause;
    use crate::tests::command::on_device::{each_device, finished};
    use crate::tests::import::on_device::{free, host_memory, releasing, wait_until};
    use crate::texture::TextureObject;
    use crate::{
        Buffer, BufferLayout, CommandQueue, Device, IndexClass, MemoryClass, PixelFormat, Texture,
        TextureDescriptor, TexturePlace, TextureType, TextureUsage,
    };
    use std::sync::atomic::Ordering;

    const WIDTH: u32 = 16;
    const ROW: u64 = WIDTH as u64 * 4;
    /// The buffers' offset in their blocks: a multiple of every linear image's memory alignment.
    const AT: u64 = 65_536;

    /// A texture of `height` rows of `usage` over `buffer` at `offset`, rows `pitch` apart, at
    /// texture index `index`.
    fn over(
        name: &str,
        buffer: &Buffer,
        (height, usage): (u32, TextureUsage),
        index: u32,
        (offset, pitch): (u64, u64),
    ) -> Texture {
        let mut d =
            TextureDescriptor::new(TextureType::Type2D, PixelFormat::RGBA8Unorm, WIDTH, height);
        d.usage = usage;
        (buffer.new_texture(&d, Some(index), offset, pitch))
            .unwrap_or_else(|error| panic!("{name}: a texture over a buffer: {error}"))
    }

    /// Reads `texture` (`height` rows) by a blit into a Shared buffer, in a command buffer of its
    /// own, and checks the copies it counted: a copied texture moves the totals by its one copy
    /// in, of its texels' bytes, and its cause's pair (and the imported pair, where `imported`)
    /// alike, every other cause's pair still; one in place moves nothing. Returns the texture's
    /// cause, `None` in place.
    fn read_once(
        (name, what): (&str, &str),
        device: &Device,
        queue: &CommandQueue,
        texture: &Texture,
        (height, imported): (u32, bool),
    ) -> Option<OverCause> {
        let cause = texture.copied_over().map(|(_, copy)| copy.cause);
        let bytes = ROW * u64::from(height);
        let out_block = (device.take_block(MemoryClass::Shared, bytes))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        let out = (device.new_buffer(out_block.id, 0, bytes, None)).expect("a buffer");
        let counters = device.counters();
        let before = counters.metrics();
        let command_buffer = queue.command_buffer().expect("a command buffer");
        let mut encoder = command_buffer.blit_command_encoder().expect("an encoder");
        let layout = BufferLayout {
            offset: 0,
            bytes_per_row: ROW,
            bytes_per_image: 0,
        };
        (encoder.copy_from_texture_to_buffer(
            texture,
            TexturePlace::default(),
            [WIDTH, height, 1],
            &out,
            layout,
        ))
        .unwrap_or_else(|error| panic!("{name}: {what}: {error}"));
        encoder.end_encoding();
        command_buffer.commit().expect("a commit");
        finished(name, &command_buffer);
        assert_eq!(command_buffer.error(), None, "{name}: {what}");
        let after = counters.metrics();
        let (copies, copy_bytes) = match cause {
            Some(_) => (1, bytes),
            None => (0, 0),
        };
        assert_eq!(
            (
                after.over_copies - before.over_copies,
                after.over_copy_bytes - before.over_copy_bytes,
            ),
            (copies, copy_bytes),
            "{name}: {what} ({cause:?}): the totals"
        );
        let imported_moved = (
            after.over_copies_imported - before.over_copies_imported,
            after.over_copy_bytes_imported - before.over_copy_bytes_imported,
        );
        let imported_expected = match imported {
            true => (copies, copy_bytes),
            false => (0, 0),
        };
        assert_eq!(
            imported_moved, imported_expected,
            "{name}: {what}: the imported pair"
        );
        for (other, pair) in moved(&before, &after) {
            let expected = match Some(other) == cause {
                true => (copies, copy_bytes),
                false => (0, 0),
            };
            assert_eq!(
                pair, expected,
                "{name}: {what} ({cause:?}): {other:?}'s pair"
            );
        }
        drop((command_buffer, out));
        (device.release_block(out_block)).unwrap_or_else(|error| panic!("{name}: {error}"));
        cause
    }

    #[test]
    fn every_device_counts_each_textures_copies_under_its_cause() {
        each_device("over causes", |name, _, device| {
            let answer = (device.minimum_linear_texture_alignment(PixelFormat::RGBA8Unorm))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let pitch = ROW.next_multiple_of(answer);
            let queue = device.new_command_queue(4).expect("a queue");
            let first = IndexClass::Texture.range().start;
            (device.grant_indices(IndexClass::Texture, first, 4)).expect("a grant");
            let lavapipe = cfg!(m2v_cpu_icd);
            let mut found = Vec::new();

            // Rows past the device's pitch, and a place a texel off the linear image's memory
            // alignment: each over a Shared buffer.
            let height = 4;
            let len = u64::from(height) * (pitch + answer) + 4;
            let block = (device.take_block(MemoryClass::Shared, AT + len))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let buffer = (device.new_buffer(block.id, AT, len, None))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let usage = TextureUsage::SHADER_READ;
            for (what, index, place, made_for) in [
                ("pitch", first, (0, pitch + answer), OverCause::Pitch),
                ("alignment", first + 1, (4, pitch), OverCause::Alignment),
            ] {
                let texture = over(name, &buffer, (height, usage), index, place);
                let cause = read_once((name, what), &device, &queue, &texture, (height, false));
                if lavapipe {
                    assert_eq!(cause, Some(made_for), "{name}: {what}");
                }
                found.push((what, cause));
            }

            // A render target of 3 rows over a buffer of its rows alone.
            let height = 3;
            let rows = u64::from(height) * pitch;
            let target_block = (device.take_block(MemoryClass::Shared, AT + rows))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let target_buffer = (device.new_buffer(target_block.id, AT, rows, None))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let usage = TextureUsage::RENDER_TARGET;
            let target = over(name, &target_buffer, (height, usage), first + 2, (0, pitch));
            let cause = read_once(
                (name, "render reach"),
                &device,
                &queue,
                &target,
                (height, false),
            );
            if lavapipe {
                assert_eq!(cause, Some(OverCause::RenderReach), "{name}: render reach");
            }
            found.push(("render reach", cause));

            // Over host memory's import, where the device imports it.
            if let Some((_, alignment)) = device.core().imports.host.as_ref() {
                let height = 4;
                let len = (u64::from(height) * pitch).next_multiple_of(*alignment);
                let (pointer, mapping) = host_memory(len, *alignment);
                let (release, released) = releasing();
                // SAFETY: the memory is the test's, allocated until it frees it after the
                // import's release.
                let (import, _) = unsafe { device.import_host_memory(pointer, len, release) }
                    .unwrap_or_else(|error| panic!("{name}: an import: {error}"));
                let imported = (device.new_buffer(import.id, 0, len, None)).expect("a buffer");
                let usage = TextureUsage::SHADER_READ;
                let texture = over(name, &imported, (height, usage), first + 3, (0, pitch));
                let cause = read_once(
                    (name, "host memory"),
                    &device,
                    &queue,
                    &texture,
                    (height, true),
                );
                assert_eq!(cause, Some(OverCause::HostMemory), "{name}: host memory");
                found.push(("host memory", cause));
                drop((texture, imported, import));
                wait_until("the release", || released.load(Ordering::SeqCst));
                free(mapping);
            }
            println!("over causes: {name}: {found:?}");
        });
    }
}
