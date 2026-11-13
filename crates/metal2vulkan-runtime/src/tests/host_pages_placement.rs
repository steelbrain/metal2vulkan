//! `Device::host_pages_texture_placement` answers, before any no-copy buffer of the pages is made,
//! what `Buffer::texture_placement_with_iosurface_plane` answers once one is (both through
//! `texture/route.rs`'s `Device::placement_over`): the same answer wherever the steps before an
//! import or an image decide it, and elsewhere `InPlace`, where the buffer's answer is the image
//! in place or a refusal only an image or the import decides (`InPlaceRefusal::Image`, `Pitch`,
//! `Placement`, `Source`, `DmaBuf`).
//!
//! Over a spread of shapes: texels of 1, 4 and 16 bytes; rows at their own width and padded past
//! it; rows of 8 bytes (a plane under 16 bytes a row, `linear_plan.rs`); at the pages' first byte
//! and past a page; `ShaderRead` and `Unknown` usage; and a plane past the buffer's length,
//! refused alike. By pointer, on the device the cfg selects once its check settled, and under
//! `HostPageTexturePolicy::Shadow`; as dma-bufs on Linux, on a device made without
//! `VK_EXT_external_memory_host` (`tests/no_pointer_import.rs`), where `/dev/udmabuf` opens. Each
//! pre-import answer makes no `VkImage` on the asking thread
//! (`ObjectChurn::vk_images_made_this_thread`).
//!
//! A device test: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`), on the kind
//! of device the cfg selects. Printed (`HOST-PAGES-PLACEMENT`): each shape's two answers.

use super::blit_holds::{device, over_host, shadow_device, Host, Scratch, BYTES};
use crate::{
    Buffer, Device, Error, HostBacking, InPlaceRefusal, IoSurfacePlane, ResourceOptions,
    TextureDescriptor, TexturePlacement, TextureShadowReason,
};
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use std::time::Duration;

/// The longest the device's check may take here.
const SETTLE: Duration = Duration::from_secs(120);

/// A shape: its format, width and height, the plane's `bytesPerRow`, and its offset from the
/// pages' first byte (`None`: the last 4096 bytes of `length`, so its texels run past it, which
/// both calls refuse alike).
type Shape = (PixelFormat, u64, u64, u64, Option<u64>);

/// The shapes asked, each with `ShaderRead` and `Unknown` usage.
const SHAPES: [Shape; 8] = [
    (PixelFormat::R8Unorm, 8, 8, 8, Some(0)),
    (PixelFormat::R8Unorm, 64, 32, 64, Some(0)),
    (PixelFormat::R8Unorm, 60, 32, 64, Some(4096)),
    (PixelFormat::RGBA8Unorm, 64, 32, 256, Some(0)),
    (PixelFormat::RGBA8Unorm, 61, 32, 256, Some(4096 + 256)),
    (PixelFormat::RGBA32Float, 32, 16, 512, Some(0)),
    (PixelFormat::RGBA32Float, 30, 16, 512, Some(4096)),
    (PixelFormat::RGBA8Unorm, 64, 64, 256, None),
];

/// What the buffer's answer may be where the pre-import answer is `InPlace`: the image in place,
/// or a refusal only an image or the import decides.
fn import_decides(after: TexturePlacement) -> bool {
    use InPlaceRefusal as R;
    match after {
        TexturePlacement::InPlace => true,
        TexturePlacement::Shadow(TextureShadowReason::Refused(refusal)) => {
            matches!(
                refusal,
                R::Image | R::Pitch | R::Placement | R::Source | R::DmaBuf
            )
        }
        _ => false,
    }
}

/// Ask every shape of [`SHAPES`] of `device` before (`host_pages_texture_placement`, `backing`,
/// from `start`) and after (`buffer`, of the same `length` bytes over the same pages) the buffer,
/// `label` naming the run: each pre-import answer made no image, and equals the buffer's (a
/// refusal included) where it is not `InPlace`. The pre-import answers.
fn ask_each(
    device: &Device,
    label: &str,
    (backing, buffer): (HostBacking, &Buffer),
    (start, length): (u64, u64),
) -> Vec<Result<TexturePlacement, Error>> {
    let mut answers = Vec::new();
    for (format, width, height, bytes_per_row, offset) in SHAPES {
        let offset = offset.unwrap_or(length - 4096);
        for usage in [TextureUsage::SHADER_READ, TextureUsage::UNKNOWN] {
            let mut descriptor = TextureDescriptor::texture_2d(format, width, height, false);
            descriptor.usage = usage;
            descriptor.resource_options = ResourceOptions::default();
            let plane = IoSurfacePlane::new(offset, bytes_per_row, false);
            let made = || device.work_counts().object_churn.vk_images_made_this_thread;
            let images = made();
            let before =
                device.host_pages_texture_placement(backing, &descriptor, plane, start, length);
            let images = made() - images;
            let after = buffer.texture_placement_with_iosurface_plane(&descriptor, plane);
            println!(
                "HOST-PAGES-PLACEMENT {} {label}: {format:?} {width}x{height} at {offset}, {bytes_per_row} \
                 a row, {usage:?}: before the buffer {before:?}, after {after:?}",
                device.name()
            );
            assert_eq!(images, 0, "{label}: the pre-import answer made an image");
            match (&before, &after) {
                (Ok(TexturePlacement::InPlace), Ok(after)) => assert!(
                    import_decides(*after),
                    "{label}: {format:?} {width}x{height} at {offset}: InPlace before, {after:?} after"
                ),
                _ => assert_eq!(
                    before, after,
                    "{label}: {format:?} {width}x{height} at {offset}, {usage:?}"
                ),
            }
            answers.push(before);
        }
    }
    answers
}

#[test]
fn before_the_buffer_the_placement_is_the_buffer_s_where_no_import_decides() {
    let scratch = Scratch::new("host-pages-placement");
    let host = Host::new();
    let start = host.words().as_ptr() as u64;
    let length = BYTES as u64;
    for shadow in [false, true] {
        let device = match shadow {
            false => device(&scratch.0),
            true => shadow_device(&scratch.0),
        };
        if !shadow {
            assert!(device.settle_in_place_checks(SETTLE), "the check settled");
        }
        let runs = Arc::new(AtomicUsize::new(0));
        let Some(buffer) = over_host(&device, &host, &runs) else {
            println!(
                "HOST-PAGES-PLACEMENT {}: no no-copy buffer made",
                device.name()
            );
            return;
        };
        if buffer.host_backing() != Some(HostBacking::ImportedByPointer) {
            println!(
                "HOST-PAGES-PLACEMENT {}: the pages are {:?}, not imported by their pointer",
                device.name(),
                buffer.host_backing()
            );
            return;
        }
        let label = match shadow {
            false => "by pointer",
            true => "by pointer, every texture a shadow",
        };
        let pages = (HostBacking::ImportedByPointer, &buffer);
        let answers = ask_each(&device, label, pages, (start, length));
        assert!(
            answers.iter().any(Result::is_err),
            "{label}: no shape was refused"
        );
        if shadow {
            let policy = Ok(TexturePlacement::Shadow(TextureShadowReason::Policy));
            assert!(
                answers.contains(&policy),
                "{label}: the policy answered nothing"
            );
        }
        // Another backing is refused.
        let descriptor = TextureDescriptor::texture_2d(PixelFormat::R8Unorm, 4, 4, false);
        let plane = IoSurfacePlane::new(0, 64, false);
        let copied = HostBacking::Copied;
        assert!(device
            .host_pages_texture_placement(copied, &descriptor, plane, start, length)
            .is_err());
        drop(buffer);
    }
}

#[cfg(target_os = "linux")]
#[test]
fn before_the_buffer_the_placement_over_dma_bufs_is_the_buffer_s_where_no_import_decides() {
    use super::host_residency::{host_page, udmabuf_openable, Ram};
    use super::no_pointer_import;
    use crate::HostMemory;
    const TEST: &str =
        "before_the_buffer_the_placement_over_dma_bufs_is_the_buffer_s_where_no_import_decides";
    if !udmabuf_openable(TEST) {
        return;
    }
    let scratch = Scratch::new("host-pages-placement-dma-buf");
    let device = no_pointer_import::device(&scratch.0, None);
    assert!(device.settle_in_place_checks(SETTLE), "the check settled");
    let page = host_page();
    let pages = 64;
    let ram = Ram::new(pages * page, 0, pages * page);
    let calls = Arc::new(AtomicUsize::new(0));
    let length = (pages * page) as u64;
    let start = ram.pointer(0).as_ptr() as u64;
    // SAFETY: the RAM outlives the buffer, and the source makes dma-bufs of its pages.
    let made = unsafe {
        device.new_buffer_with_bytes_no_copy(
            HostMemory::with_dma_buf(ram.pointer(0), ram.source(&calls)),
            length,
            ResourceOptions::default(),
            None,
        )
    };
    let Ok(buffer) = made else {
        println!("{TEST}: on {}: no buffer over dma-bufs made", device.name());
        return;
    };
    if buffer.host_backing() != Some(HostBacking::ImportedAsDmaBuf) {
        println!(
            "{TEST}: on {}: the pages are {:?}, not imported as dma-bufs",
            device.name(),
            buffer.host_backing()
        );
        return;
    }
    let dma_buf = (HostBacking::ImportedAsDmaBuf, &buffer);
    let answers = ask_each(&device, "as dma-bufs", dma_buf, (start, length));
    assert!(
        answers.iter().any(Result::is_err),
        "as dma-bufs: no shape was refused"
    );
    drop(buffer);
}
