//! The cost of a consumer's present past its import gate (`copied.rs`): each frame makes a new
//! no-copy buffer over a surface's pages, its import refused, and a texture of the surface's plane
//! over it, through which the display reads the frame, as the consumer does at every present.
//! Two shapes, 64x64 and 3840x2160 `BGRA8Unorm`, each two ways: beside a buffer of the surface's
//! that lives across the frames (the guest's render target, written each frame through its own
//! texture), and alone (the frame's buffer the only one over the pages, written through its own
//! texture). Each prints the copies made a frame, the median CPU time of the buffer's and the
//! texture's making, and of the whole frame (making, encoding, commit and the wait), and the
//! bytes copied in and merged back a frame (`COPIED-PRESENT`). Ignored: run it in release,
//! `cargo test --release -p metal2vulkan-runtime --lib copied_present_bench -- --ignored
//! --nocapture` under `--cfg m2v_gpu`.

use crate::{
    BlitOption, Buffer, BufferContents, BufferDescriptor, Device, DeviceKind, DeviceOptions,
    HostMemory, IoSurfacePlane, Origin, ResourceOptions, Size, Texture, TextureDescriptor,
};
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use std::alloc::Layout;
use std::ptr::NonNull;
use std::sync::Arc;
use std::time::{Duration, Instant};

const WAIT: Duration = Duration::from_secs(60);
/// The region each frame writes and reads: the frame's mark, not the frame's whole cost.
const MARK: u64 = 64;

/// A surface's pages: `width` x `height` texels of 4 bytes, packed rows.
struct Pages {
    pointer: NonNull<u8>,
    layout: Layout,
}

impl Pages {
    fn new(len: usize) -> Pages {
        let layout = Layout::from_size_align(len, 1 << 16).unwrap();
        // SAFETY: the layout's size is not zero.
        let pointer = NonNull::new(unsafe { std::alloc::alloc_zeroed(layout) }).unwrap();
        Pages { pointer, layout }
    }
}

impl Drop for Pages {
    fn drop(&mut self) {
        // SAFETY: allocated with this layout; dropped after every buffer over it.
        unsafe { std::alloc::dealloc(self.pointer.as_ptr(), self.layout) };
    }
}

/// A no-copy buffer of `len` bytes over `pages`, its import refused.
fn refused(device: &Device, pages: &Pages, len: u64) -> Buffer {
    // SAFETY: the pages outlive every buffer over them (each bench drops its buffers first).
    let memory = unsafe { HostMemory::new(pages.pointer) }.with_import_gate(Arc::new(|_| false));
    let contents = BufferContents::NoCopy {
        memory,
        deallocator: None,
        keep_alive: None,
    };
    let options = ResourceOptions::default();
    (device.new_buffer_with_descriptor(BufferDescriptor::new(len, options, contents))).unwrap()
}

fn plane(buffer: &Buffer, width: u64, height: u64) -> Texture {
    let mut d = TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, width, height, false);
    d.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE | TextureUsage::RENDER_TARGET;
    d.resource_options = buffer.resource_options();
    let plane = IoSurfacePlane::new(0, width * 4, false);
    buffer.new_texture_with_iosurface_plane(&d, plane).unwrap()
}

fn median(mut times: Vec<f64>) -> f64 {
    times.sort_by(f64::total_cmp);
    times[times.len() / 2]
}

/// Run `frames` presents of a `width` x `height` surface, beside a buffer of its own that lives
/// across them where `beside`.
fn present(device: &Device, (width, height): (u64, u64), beside: bool, frames: usize) {
    let len = width * height * 4;
    let pages = Pages::new(len as usize);
    let queue = device.new_command_queue().unwrap();
    let mark = (MARK * MARK * 4) as usize;
    let source = device
        .new_buffer(mark as u64, ResourceOptions::shared())
        .unwrap();
    let out = device
        .new_buffer(mark as u64, ResourceOptions::shared())
        .unwrap();
    let region = (Origin::new(0, 0, 0), Size::new(MARK, MARK, 1));
    let target = beside.then(|| {
        let buffer = refused(device, &pages, len);
        let texture = plane(&buffer, width, height);
        (buffer, texture)
    });
    let before = device.work_counts();
    let (mut makes, mut totals) = (Vec::new(), Vec::new());
    for _ in 0..frames {
        let start = Instant::now();
        let buffer = refused(device, &pages, len);
        let texture = plane(&buffer, width, height);
        let made = start.elapsed();
        let written = target.as_ref().map_or(&texture, |(_, texture)| texture);
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        let (origin, size) = region;
        let row = MARK * 4;
        let all = row * MARK;
        (blit.copy_from_buffer_to_texture(
            &source,
            0,
            row,
            all,
            size,
            written,
            0,
            0,
            origin,
            BlitOption::None,
        ))
        .unwrap();
        blit.end_encoding().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        (blit.copy_from_texture_to_buffer(
            &texture,
            0,
            0,
            origin,
            size,
            &out,
            0,
            row,
            all,
            BlitOption::None,
        ))
        .unwrap();
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
        drop((texture, buffer));
        makes.push(made.as_secs_f64() * 1e3);
        totals.push(start.elapsed().as_secs_f64() * 1e3);
    }
    let after = device.work_counts();
    let per = |a: u64, b: u64| (a - b) as f64 / frames as f64;
    eprintln!(
        "COPIED-PRESENT {width}x{height} {}: {frames} frames; copies made a frame {:.2}; make \
         median {:.3} ms; frame median {:.3} ms; copied in {:.0} B a frame; merged back {:.0} B a \
         frame",
        if beside {
            "beside a live buffer"
        } else {
            "alone"
        },
        per(
            after.copied_host_buffers_made,
            before.copied_host_buffers_made
        ),
        median(makes),
        median(totals),
        per(after.copied_copy_in_bytes, before.copied_copy_in_bytes),
        per(
            after.copied_merge_back_bytes,
            before.copied_merge_back_bytes
        ),
    );
    drop(target);
}

/// The presents of the module docs, on a device of the cfg's kind.
#[test]
#[ignore = "a release-mode benchmark"]
fn copied_present_bench() {
    let options = DeviceOptions {
        kinds: match cfg!(m2v_cpu_icd) {
            true => vec![DeviceKind::Cpu],
            false => vec![
                DeviceKind::Discrete,
                DeviceKind::Integrated,
                DeviceKind::Virtual,
            ],
        },
        ..DeviceOptions::default()
    };
    let device = Device::system_default_with(&options).unwrap();
    eprintln!("COPIED-PRESENT on {}", device.name());
    for (shape, frames) in [((64, 64), 300), ((3840, 2160), 40)] {
        for beside in [true, false] {
            present(&device, shape, beside, frames);
        }
    }
}
