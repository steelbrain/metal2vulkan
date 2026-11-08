//! A render target over host pages in place (`texture/in_place_image.rs`) takes the storage usage
//! a snapshot's fetch of it in the pixel interlock reads through (`render/snapshot.rs`) only where
//! the device's check passed for a shader's writes over such pages of texels of its width (those
//! of 1 byte checked apart from those of 4 and 8): with that usage a shader writes its texels in
//! place. Its image's plan says which usages it took.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The device's shader cache is a fresh
//! directory, so its check runs.

use super::blit_holds::{device, over_host, Host, Scratch};
use crate::framebuffer_fetch::through_storage;
use crate::in_place::{Access, Padding};
use crate::TextureDescriptor;
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use metal2vulkan::vk::ImageUsageFlags;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use std::time::Duration;

/// The textures' side, in texels.
const SIDE: u64 = 64;

#[test]
fn a_render_target_in_place_takes_the_storage_usage_only_where_writes_passed() {
    let host = Host::new();
    let scratch = Scratch::new("in-place-storage");
    let device = device(&scratch.0);
    let runs = Arc::new(AtomicUsize::new(0));
    let Some(buffer) = over_host(&device, &host, &runs) else {
        return;
    };
    let settled = device.settle_in_place_checks(Duration::from_secs(120));
    let report = device.in_place_check_report();
    // Whether the check passed for `access` over the buffer's pages, for texels of rows of `row`
    // bytes one after another from its start (`in_place::RowEnds`).
    let passed = |access, row: u64| {
        use crate::in_place::{PagesClass, RowEnds, Verdict};
        let layout = crate::linear_plan::BufferLayout {
            offset: 0,
            row_pitch: row,
            slice_pitch: row * SIDE,
            row_bytes: row,
            rows: SIDE,
            slices: 1,
        };
        let start = buffer.contents().map_or(0, |at| at.as_ptr().addr() as u64);
        let checks = &device.shared().in_place;
        let rows = RowEnds::of(start, &layout, checks.line());
        checks.verdict(PagesClass::Pointer, access, rows) == Verdict::Passed
    };
    let fetches = through_storage(device.compile_device().profile());
    for (format, texel) in [
        (PixelFormat::RGBA8Unorm, 4),
        (PixelFormat::BGRA8Unorm, 4),
        (PixelFormat::RGBA16Float, 8),
        (PixelFormat::R8Unorm, 1),
    ] {
        let mut descriptor = TextureDescriptor::texture_2d(format, SIDE, SIDE, false);
        descriptor.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
        let texture = buffer.new_texture(&descriptor, 0, SIDE * texel).unwrap();
        let in_place = texture.shadow().is_none() && texture.linear_image().unwrap();
        let usage = texture.image_plan().unwrap().usage;
        let storage = usage.contains(ImageUsageFlags::STORAGE);
        let feedback = usage.contains(ImageUsageFlags::ATTACHMENT_FEEDBACK_LOOP_EXT);
        println!(
            "IN-PLACE-STORAGE on {}: {format:?}: settled {settled}, fetched through storage \
             {fetches}, in place {in_place}, feedback loop usage {feedback}, storage usage \
             {storage}; {report:?}",
            device.name()
        );
        if in_place {
            assert!(
                passed(Access::Renders, SIDE * texel),
                "{format:?}: in place where the check did not pass for render targets"
            );
            let written = Access::written(texel, Padding::Kept);
            let writes = written.is_some_and(|written| passed(written, SIDE * texel));
            assert!(
                !storage || (writes && feedback && fetches),
                "{format:?}: the storage usage where the check did not pass for a shader's \
                 writes, or with no fetch to read through it"
            );
        }
    }
}
