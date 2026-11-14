//! Read twins on a device (`read_twin.rs`): a no-copy buffer of Metal's private option bit 24,
//! where the system maps no read-only view of its pages, read in place by the commands that
//! cannot write it, every write still dropped.
//!
//! - **Made where it can be.** A buffer over pages the device imports takes a twin; one over pages
//!   the process maps read-only, or whose import the caller's gate refuses, takes none (its import
//!   would be a copy); nor does any under a policy that contains no access.
//! - **Blits read in place.** A screen-sized buffer (a 3648 x 2038 BGRA image's bytes, about
//!   30 MB) copied into a texture and into a buffer is read through its twin: the copies hold the
//!   pages' bytes, a CPU write to the pages reaches the next command buffer with nothing copied in,
//!   and the commits compare none of the buffer's copy (`copied_copy_ins_compared`,
//!   `copied_copy_in_bytes_compared`, `copied_compare_threads`).
//! - **A write dropped, read back, then gone.** A fill of the buffer and a copy out of it in one
//!   command buffer: the copy reads the fill back, through the buffer's copy, which that commit
//!   compares; the next command buffer reads the pages, in place again, and the pages never took
//!   the fill.
//! - **Every store kind leaves the pages.** A blit's fill and copy into the buffer, a kernel's
//!   store into it at a slot, and a kernel's store through its address loaded from memory: the
//!   pages keep their bytes and a later command buffer reads them in place.
//! - **The guard, and the negative control.** A fill of the twin itself is refused by the
//!   encoder, its command buffer failed and the pages kept; with the guard off on the test's
//!   thread the same fill changes the pages: the twin is the pages in place, and what keeps them
//!   is that no write reaches it.
//!
//! Device tests: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`). A device
//! that imports no host memory, and a system that maps read-only views (macOS, whose buffers take
//! no twin), say so and pass.

use crate::{
    Buffer, CommandBufferStatus, Device, DeviceKind, DeviceOptions, HostMemory, ImportGate,
    ImportRequest, Origin, ResourceOptions, Size, TextureDescriptor, WorkCounts,
};
use metal2vulkan::device::{ContainmentPolicy, DevicePolicy, GuestIsolation};
use metal2vulkan::metal::PixelFormat;
use std::alloc::Layout;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};
use std::time::Duration;

const WAIT: Duration = Duration::from_secs(120);
/// The screen-sized image a window server reads every frame: 3648 x 2038 BGRA texels.
const WIDTH: u64 = 3648;
const HEIGHT: u64 = 2038;
const ROW: u64 = WIDTH * 4;
/// The image's bytes, rounded up to whole pages of 64 KiB.
const SCREEN: usize = ((ROW * HEIGHT) as usize).next_multiple_of(1 << 16);
/// A small buffer's bytes.
const SMALL: usize = 64 * 1024;

/// Host memory of whole pages in a pattern of its own, freed when dropped (after the device).
struct Pages {
    pointer: NonNull<u8>,
    layout: Layout,
}

impl Pages {
    fn new(length: usize, seed: u8) -> Pages {
        let layout = Layout::from_size_align(length, 1 << 16).unwrap();
        // SAFETY: the layout's size is not zero.
        let pointer = NonNull::new(unsafe { std::alloc::alloc(layout) }).unwrap();
        let pages = Pages { pointer, layout };
        pages.write(0, &pattern(length, seed));
        pages
    }

    fn bytes(&self) -> Vec<u8> {
        // SAFETY: the allocation's bytes, which no command writes now.
        unsafe { std::slice::from_raw_parts(self.pointer.as_ptr(), self.layout.size()) }.to_vec()
    }

    fn write(&self, at: usize, bytes: &[u8]) {
        assert!(at + bytes.len() <= self.layout.size());
        // SAFETY: the allocation's bytes, which no command uses now.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                self.pointer.as_ptr().add(at),
                bytes.len(),
            )
        };
    }

    /// A no-copy buffer of Metal's private option bit 24 over the pages, under `gate`.
    fn private_bit_buffer(&self, device: &Device, gate: Option<ImportGate>) -> Buffer {
        // SAFETY: the pages outlive the buffer and the device (dropped after both).
        let memory = unsafe { HostMemory::new(self.pointer) };
        let memory = match gate {
            Some(gate) => memory.with_import_gate(gate),
            None => memory,
        };
        // SAFETY: as above.
        unsafe {
            device.new_buffer_with_bytes_no_copy(
                memory,
                self.layout.size() as u64,
                private_bit(),
                None,
            )
        }
        .expect("a buffer of bit 24 over the pages")
    }
}

impl Drop for Pages {
    fn drop(&mut self) {
        // SAFETY: allocated with this layout in `new`.
        unsafe { std::alloc::dealloc(self.pointer.as_ptr(), self.layout) };
    }
}

/// `length` bytes of a pattern of `seed`'s, none of them zero.
fn pattern(length: usize, seed: u8) -> Vec<u8> {
    (0..length)
        .map(|at| (at as u8).wrapping_mul(31).wrapping_add(seed) | 1)
        .collect()
}

/// `Shared` with Metal's private option bit 24.
fn private_bit() -> ResourceOptions {
    ResourceOptions::from_raw(ResourceOptions::shared().raw() | ResourceOptions::PRIVATE_BIT)
        .unwrap()
}

/// The test's device, under `policy` where given; `None` where read twins cannot be made here
/// (said).
fn device(policy: Option<DevicePolicy>) -> Option<Device> {
    if crate::read_only_view::VIEWS_DROP_WRITES {
        eprintln!("READ-TWIN: bit 24 maps a read-only view here; nothing to test");
        return None;
    }
    let mut options = DeviceOptions {
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
    if let Some(policy) = policy {
        options.policy = policy;
    }
    let device = Device::system_default_with(&options).unwrap();
    if device.shared().host_import.is_err() {
        eprintln!("READ-TWIN: the device imports no host memory here; nothing to test");
        return None;
    }
    // Its in-place check's own imports land in no count compared below: settled first.
    let exportable = crate::device::lock(&device.shared().memory)
        .fd_buffer_types
        .is_ok();
    if exportable {
        assert!(device.settle_in_place_checks(Duration::from_secs(120)));
    }
    Some(device)
}

/// Commit one command buffer whose blit encoder `encode` fills, wait for it, and answer its
/// status.
fn blit(
    device: &Device,
    encode: impl FnOnce(&mut crate::BlitCommandEncoder),
) -> CommandBufferStatus {
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.blit_command_encoder().unwrap();
    encode(&mut encoder);
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    command_buffer.status()
}

/// A buffer of the device's own of `length` bytes.
fn own(device: &Device, length: usize) -> Buffer {
    device
        .new_buffer(length as u64, ResourceOptions::shared())
        .unwrap()
}

/// `buffer`'s first `length` bytes as the CPU reads them.
fn contents(buffer: &Buffer, length: usize) -> Vec<u8> {
    let at = buffer.contents().unwrap();
    // SAFETY: the buffer's bytes, which no command writes now.
    unsafe { std::slice::from_raw_parts(at.as_ptr(), length) }.to_vec()
}

/// `buffer`'s bytes as a later command buffer's copy out of it reads them.
fn gpu_bytes(device: &Device, buffer: &Buffer, length: usize) -> Vec<u8> {
    let out = own(device, length);
    let status = blit(device, |encoder| {
        encoder
            .copy_from_buffer_to_buffer(buffer, 0, &out, 0, length as u64)
            .unwrap();
    });
    assert_eq!(status, CommandBufferStatus::Completed);
    contents(&out, length)
}

/// What the commits' copies in did between two counts: the copies compared, their bytes, and
/// the threads the comparisons started.
fn compared(before: &WorkCounts, after: &WorkCounts) -> [u64; 3] {
    [
        after.copied_copy_ins_compared - before.copied_copy_ins_compared,
        after.copied_copy_in_bytes_compared - before.copied_copy_in_bytes_compared,
        after.copied_compare_threads - before.copied_compare_threads,
    ]
}

#[test]
fn a_screen_sized_private_bit_buffer_is_read_in_place_by_blits() {
    let pages = Pages::new(SCREEN, 3);
    let Some(device) = device(None) else {
        return;
    };
    let before = device.work_counts();
    let buffer = pages.private_bit_buffer(&device, None);
    let after = device.work_counts();
    assert_eq!(
        after.copied_read_only_memory - before.copied_read_only_memory,
        1
    );
    assert_eq!(
        after.read_twins_made - before.read_twins_made,
        1,
        "a twin made"
    );
    assert!(buffer.read_twin().is_some());

    // A copy into a texture, the window server's read, and one into a buffer, in one command
    // buffer: both through the twin, the commit comparing nothing.
    let mut descriptor =
        TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, WIDTH, HEIGHT, false);
    descriptor.resource_options = ResourceOptions::private();
    let texture = device.new_texture(&descriptor).unwrap();
    let out = own(&device, SCREEN);
    let before = device.work_counts();
    let status = blit(&device, |encoder| {
        encoder
            .copy_from_buffer_to_texture(
                &buffer,
                0,
                ROW,
                0,
                Size::new(WIDTH, HEIGHT, 1),
                &texture,
                0,
                0,
                Origin::new(0, 0, 0),
                crate::BlitOption::None,
            )
            .unwrap();
        encoder
            .copy_from_buffer_to_buffer(&buffer, 0, &out, 0, SCREEN as u64)
            .unwrap();
    });
    let after = device.work_counts();
    assert_eq!(status, CommandBufferStatus::Completed);
    assert_eq!(after.read_twin_blit_reads - before.read_twin_blit_reads, 2);
    assert_eq!(compared(&before, &after), [0, 0, 0], "nothing compared");
    eprintln!(
        "READ-TWIN: two blits out of a {SCREEN}-byte buffer of bit 24: copies in compared {:?} \
         (copies, bytes, threads), unused {}",
        compared(&before, &after),
        after.copied_copy_ins_unused - before.copied_copy_ins_unused
    );
    assert!(
        contents(&out, SCREEN) == pages.bytes(),
        "the buffer copy holds the pages"
    );
    let texels = own(&device, (ROW * HEIGHT) as usize);
    let status = blit(&device, |encoder| {
        encoder
            .copy_from_texture_to_buffer(
                &texture,
                0,
                0,
                Origin::new(0, 0, 0),
                Size::new(WIDTH, HEIGHT, 1),
                &texels,
                0,
                ROW,
                0,
                crate::BlitOption::None,
            )
            .unwrap();
    });
    assert_eq!(status, CommandBufferStatus::Completed);
    assert!(
        contents(&texels, (ROW * HEIGHT) as usize) == pages.bytes()[..(ROW * HEIGHT) as usize],
        "the texture holds the pages' texels"
    );

    // A CPU write reaches the next command buffer in place, nothing copied in.
    pages.write(1 << 20, &[0x7e; 4096]);
    let before = device.work_counts();
    assert!(
        gpu_bytes(&device, &buffer, SCREEN) == pages.bytes(),
        "the CPU's write read"
    );
    let after = device.work_counts();
    assert_eq!(compared(&before, &after), [0, 0, 0]);
    assert_eq!(after.copied_copy_in_bytes, before.copied_copy_in_bytes);

    // A fill and a copy out in one command buffer: the copy reads the fill back, through the
    // buffer's copy, which the commit compares; the pages never take it.
    let kept = pages.bytes();
    let back = own(&device, SMALL);
    let before = device.work_counts();
    let status = blit(&device, |encoder| {
        encoder.fill_buffer(&buffer, 0..SMALL as u64, 0x5a).unwrap();
        encoder
            .copy_from_buffer_to_buffer(&buffer, 0, &back, 0, SMALL as u64)
            .unwrap();
    });
    let after = device.work_counts();
    assert_eq!(status, CommandBufferStatus::Completed);
    assert!(
        contents(&back, SMALL) == vec![0x5a; SMALL],
        "the same command buffer reads the fill back"
    );
    assert_eq!(
        after.read_twin_reads_after_writes - before.read_twin_reads_after_writes,
        1
    );
    assert_eq!(after.read_twin_blit_reads, before.read_twin_blit_reads);
    let [copies, bytes, threads] = compared(&before, &after);
    eprintln!(
        "READ-TWIN: a fill and a copy out of it in one command buffer: copies in compared \
         {copies} copies, {bytes} bytes, {threads} threads"
    );
    assert_eq!(copies, 1, "the copy route compares the buffer's copy");
    assert!(pages.bytes() == kept, "the pages never took the fill");
    // The next command buffer reads the pages, in place again.
    let before = device.work_counts();
    assert!(
        gpu_bytes(&device, &buffer, SCREEN) == kept,
        "the next command buffer reads the pages"
    );
    let after = device.work_counts();
    assert_eq!(after.read_twin_blit_reads - before.read_twin_blit_reads, 1);
    assert_eq!(compared(&before, &after), [0, 0, 0]);
    assert_eq!(after.read_twin_writes_refused, 0);
    drop((buffer, texture, out, texels, back));
}

#[test]
fn every_store_into_a_private_bit_buffer_leaves_its_pages() {
    let pages = Pages::new(SMALL, 5);
    let Some(device) = device(None) else {
        return;
    };
    let buffer = pages.private_bit_buffer(&device, None);
    assert!(buffer.read_twin().is_some(), "a twin made");
    let kept = pages.bytes();
    // A command buffer's use of it, through its twin: the device's live buffers list the buffer,
    // reached unnamed from here.
    assert!(gpu_bytes(&device, &buffer, SMALL) == kept);
    // A blit's fill, and its copy into the buffer.
    let source = own(&device, SMALL);
    // SAFETY: the buffer's bytes, which no command uses now.
    unsafe { std::ptr::write_bytes(source.contents().unwrap().as_ptr(), 0x33, SMALL) };
    let status = blit(&device, |encoder| {
        encoder.fill_buffer(&buffer, 256..4096, 0xc4).unwrap();
    });
    assert_eq!(status, CommandBufferStatus::Completed);
    assert!(pages.bytes() == kept, "a fill leaves the pages");
    let status = blit(&device, |encoder| {
        encoder
            .copy_from_buffer_to_buffer(&source, 0, &buffer, 0, SMALL as u64)
            .unwrap();
    });
    assert_eq!(status, CommandBufferStatus::Completed);
    assert!(
        pages.bytes() == kept,
        "a copy into the buffer leaves the pages"
    );
    // A kernel's store into it at a slot (`out`), and through its address loaded from memory.
    let function = super::encoder_holds::library(&device, "engine/ob2_loaded_pointer.ll")
        .new_function_with_name("ob2_loaded_pointer")
        .unwrap();
    let state = device
        .new_compute_pipeline_state_with_function(&function, &crate::PipelineOptions::default())
        .unwrap();
    let indices: Vec<u8> = (0..16u32).flat_map(u32::to_le_bytes).collect();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(&state).unwrap();
    encoder.set_bytes(&indices, 0).unwrap();
    encoder
        .set_bytes(&buffer.gpu_address().0.to_le_bytes(), 1)
        .unwrap();
    encoder.set_buffer(Some(&buffer), 0, 2).unwrap();
    encoder
        .dispatch_threads(Size::new(8, 1, 1), Size::new(8, 1, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    assert!(pages.bytes() == kept, "a kernel's stores leave the pages");
    // A later command buffer reads the pages in place.
    let before = device.work_counts();
    assert!(
        gpu_bytes(&device, &buffer, SMALL) == kept,
        "a later command buffer reads the pages"
    );
    let after = device.work_counts();
    assert_eq!(after.read_twin_blit_reads - before.read_twin_blit_reads, 1);
    assert_eq!(after.read_twin_writes_refused, 0);
    drop((buffer, source, state, queue));
}

#[test]
fn a_write_of_the_twin_itself_is_refused_and_would_reach_the_pages() {
    let pages = Pages::new(SMALL, 7);
    let Some(device) = device(None) else {
        return;
    };
    let buffer = pages.private_bit_buffer(&device, None);
    let twin = buffer.read_twin().expect("a twin made");
    let kept = pages.bytes();
    // The guard: the fill is refused, its command buffer failed, the pages kept.
    let before = device.work_counts();
    let status = blit(&device, |encoder| {
        encoder.fill_buffer(&twin, 0..4096, 0xe1).unwrap();
    });
    let after = device.work_counts();
    assert_eq!(
        status,
        CommandBufferStatus::Error,
        "the command buffer fails"
    );
    assert_eq!(
        after.read_twin_writes_refused - before.read_twin_writes_refused,
        1
    );
    assert!(pages.bytes() == kept, "the guard keeps the pages");
    // The negative control: with the guard off on this thread, the same fill reaches the pages,
    // which the twin is in place.
    crate::read_twin::UNGUARDED.with(|unguarded| unguarded.set(true));
    let status = blit(&device, |encoder| {
        encoder.fill_buffer(&twin, 0..4096, 0xe1).unwrap();
    });
    crate::read_twin::UNGUARDED.with(|unguarded| unguarded.set(false));
    assert_eq!(status, CommandBufferStatus::Completed);
    assert!(
        pages.bytes()[..4096] == [0xe1; 4096],
        "unguarded, a write through the twin changes the pages"
    );
    drop((twin, buffer));
}

#[test]
fn no_twin_where_the_pages_import_would_be_a_copy() {
    let pages = Pages::new(SMALL, 9);
    let Some(device) = device(None) else {
        return;
    };
    // The caller's gate refuses the import: the twin's would be a copy.
    let asked: Arc<Mutex<Vec<ImportRequest>>> = Arc::default();
    let gate: ImportGate = {
        let asked = Arc::clone(&asked);
        Arc::new(move |request| {
            asked.lock().unwrap().push(request);
            false
        })
    };
    let before = device.work_counts();
    let buffer = pages.private_bit_buffer(&device, Some(gate));
    let after = device.work_counts();
    assert_eq!(after.read_twins_made, before.read_twins_made, "no twin");
    assert_eq!(
        after.read_twins_unimported - before.read_twins_unimported,
        1
    );
    assert!(buffer.read_twin().is_none());
    // Its copy serves its reads, and drops its writes, as before.
    let kept = pages.bytes();
    let status = blit(&device, |encoder| {
        encoder.fill_buffer(&buffer, 0..4096, 0x11).unwrap();
    });
    assert_eq!(status, CommandBufferStatus::Completed);
    assert!(gpu_bytes(&device, &buffer, SMALL) == kept);
    assert!(pages.bytes() == kept);
    drop(buffer);
    // Pages the process maps read-only: their import would be a copy too.
    #[cfg(target_os = "linux")]
    {
        let host = super::copied_device::ReadOnlyHost::new(31);
        let calls: Arc<std::sync::atomic::AtomicUsize> = Arc::default();
        let gate: ImportGate = Arc::new(|_| true);
        let before = device.work_counts();
        // SAFETY: the view outlives the buffer and the device (dropped after both).
        let buffer = unsafe {
            device.new_buffer_with_bytes_no_copy(
                host.memory(gate, &calls),
                host.length(),
                private_bit(),
                None,
            )
        }
        .unwrap();
        let after = device.work_counts();
        assert_eq!(after.read_twins_made, before.read_twins_made, "no twin");
        assert_eq!(
            after.read_twins_unimported - before.read_twins_unimported,
            1
        );
        assert!(buffer.read_twin().is_none());
        drop(buffer);
    }
}

#[test]
fn no_twin_where_the_policy_contains_no_access() {
    let pages = Pages::new(SMALL, 11);
    let mut policy = DevicePolicy::UNTRUSTED;
    policy.containment = ContainmentPolicy::UnsafeOff;
    policy.isolation = GuestIsolation::Off;
    let Some(device) = device(Some(policy)) else {
        return;
    };
    let before = device.work_counts();
    let buffer = pages.private_bit_buffer(&device, None);
    let after = device.work_counts();
    assert_eq!(after.read_twins_made, before.read_twins_made, "no twin");
    assert_eq!(
        after.read_twins_uncontained - before.read_twins_uncontained,
        1
    );
    assert!(buffer.read_twin().is_none());
    let before = device.work_counts();
    assert!(
        gpu_bytes(&device, &buffer, SMALL) == pages.bytes(),
        "read through its copy"
    );
    let after = device.work_counts();
    assert_eq!(after.read_twin_blit_reads, before.read_twin_blit_reads);
    assert_eq!(compared(&before, &after)[0], 1, "its copy compared");
    drop(buffer);
}
