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
//!   device the same fill changes the pages: the twin is the pages in place, and what keeps them
//!   is that no write reaches it.
//! - **A texture over it, read in place.** A buffer's texture takes a read twin of its own, which
//!   a kernel that only reads the texture binds: its texels are the pages', and the commit
//!   compares none of the buffer's copy. After a fill of the buffer in the same command buffer
//!   the kernel reads the texture itself, the fill read back, and the next command buffer reads
//!   the pages in place again. So for the screen-sized buffer's texture at a 3648-texel pitch,
//!   where the device's linear image of it lies within the buffer's bytes; where it runs past
//!   them (lavapipe's takes two rows more), neither the texture nor a twin lies in place, and the
//!   kernel reads the twin's shadow of the pages, as below.
//! - **A texture no image lies over, read through a shadow of its twin.** With RADV's rule for a
//!   `DRM_FORMAT_MOD_LINEAR` plane forced on the test's device (rows and a first byte at
//!   multiples of 256, `DeviceShared::linear_pitch_rule_forced`), a window server's shapes (48 x
//!   48 at rows of 192 bytes; 128 x 128 at rows of 512 bytes from 32 bytes in; 64 x 64 at rows of
//!   272 bytes, 16 past its texels) lie in place over
//!   neither the buffer's copy nor its twin's pages: the texture is a shadow of the copy, and its
//!   twin, kept, a shadow of the pages (`read_twin_texture_shadows`). A draw that samples it and a
//!   kernel that reads it bind the twin's shadow, copied in from the pages: the texels are the
//!   pages', a CPU write reaches the next command buffer, and no commit compares the copy; after
//!   a fill of the buffer in the same command buffer they read the texture itself, the fill read
//!   back, the pages kept. Over pages whose memory came with a log of their writes, the twin's
//!   shadow copies in every row at the first command buffer, nothing at the next where the log
//!   reports no write (`dirty_copies_in_skipped`, `dirty_copy_in_bytes_copied`), and the rows of a
//!   told write alone.
//! - **A texture over it, sampled in place by a draw.** A small buffer's texture sampled by a
//!   draw reads through its twin, the pass's pixels those of a draw sampling a texture of the
//!   device's own holding the same texels, and the commit compares nothing; after a fill of the
//!   buffer in the same command buffer the draw samples the texture itself, the fill read back.
//! - **A buffer slot, read in place.** A kernel that only reads the buffer at a slot (and
//!   reaches no memory through a root table) binds its twin there: it reads the pages, the commit
//!   comparing nothing; after a fill of the buffer in the same command buffer it reads the copy,
//!   the fill read back. Bound at the slot the kernel writes, the buffer itself takes the writes,
//!   dropped; bound there without the proof the twin's write is refused, and with the guard off
//!   too it changes the pages.
//! - **The slots' proof, and its negative control.** A kernel that writes the texture binds the
//!   texture: its writes are dropped and the pages kept. Bound at that slot without the proof
//!   (on the test's device alone), the twin's write is refused by the guard, and with the guard
//!   off too it changes the pages.
//!
//! Device tests: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`). A device
//! that imports no host memory, and a system that maps read-only views (macOS, whose buffers take
//! no twin), say so and pass.

use super::dirty_pages_device::Log;
use crate::{
    Buffer, CommandBufferStatus, ComputePipelineState, Device, DeviceKind, DeviceOptions,
    HostMemory, ImportGate, ImportRequest, MessageSink, Origin, Region, ResourceOptions, Size,
    Texture, TextureDescriptor, WorkCounts,
};
use metal2vulkan::device::{ContainmentPolicy, DevicePolicy, GuestIsolation};
use metal2vulkan::metal::{PixelFormat, PrimitiveType, TextureUsage};
use std::alloc::Layout;
use std::ptr::NonNull;
use std::sync::atomic::Ordering;
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

    /// A no-copy buffer of Metal's private option bit 24 over the pages, whose writes `log` logs
    /// (`HostMemory::with_dirty_pages`, `dirty_pages.rs`).
    fn logged_private_bit_buffer(&self, device: &Device, log: &Arc<Log>) -> Buffer {
        // SAFETY: the pages outlive the buffer and the device (dropped after both).
        let memory = unsafe { HostMemory::new(self.pointer) }.with_dirty_pages(Arc::clone(log));
        // SAFETY: as above.
        unsafe {
            device.new_buffer_with_bytes_no_copy(
                memory,
                self.layout.size() as u64,
                private_bit(),
                None,
            )
        }
        .expect("a logged buffer of bit 24 over the pages")
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
    device_with(policy, None)
}

/// [`device`], its messages delivered to `messages` where given.
fn device_with(policy: Option<DevicePolicy>, messages: Option<MessageSink>) -> Option<Device> {
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
        messages,
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

/// A command buffer whose kernel reaches a buffer of bit 24 only through its address loaded from
/// memory compares the buffer's copy for that reach alone (`copied_unnamed_reach_copy_ins`, and
/// its bytes); a kernel may store through an address, so no read twin could serve
/// that reach (`copied_unnamed_reach_twin_copy_ins` unchanged), and its stores leave the pages.
#[test]
fn a_kernel_reaching_a_private_bit_buffer_unnamed_compares_its_copy_for_that_alone() {
    let pages = Pages::new(SMALL, 53);
    let Some(device) = device(None) else {
        return;
    };
    let buffer = pages.private_bit_buffer(&device, None);
    assert!(buffer.read_twin().is_some(), "a twin made");
    // A command buffer's use of it: the device's live buffers list it, reached unnamed from here.
    assert!(gpu_bytes(&device, &buffer, SMALL) == pages.bytes());
    // A CPU write, so the copy in compares bytes whatever its pages' log reports.
    pages.write(SMALL / 2, &[0x5a; 16]);
    let kept = pages.bytes();
    let function = super::encoder_holds::library(&device, "engine/ob2_loaded_pointer.ll")
        .new_function_with_name("ob2_loaded_pointer")
        .unwrap();
    let state = device
        .new_compute_pipeline_state_with_function(&function, &crate::PipelineOptions::default())
        .unwrap();
    let indices: Vec<u8> = (0..16u32).flat_map(u32::to_le_bytes).collect();
    let out = own(&device, SMALL);
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(&state).unwrap();
    encoder.set_bytes(&indices, 0).unwrap();
    encoder
        .set_bytes(&buffer.gpu_address().0.to_le_bytes(), 1)
        .unwrap();
    encoder.set_buffer(Some(&out), 0, 2).unwrap();
    encoder
        .dispatch_threads(Size::new(8, 1, 1), Size::new(8, 1, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    let before = device.work_counts();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    let after = device.work_counts();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    let counted = [
        after.copied_unnamed_reach_copy_ins - before.copied_unnamed_reach_copy_ins,
        after.copied_unnamed_reach_bytes_compared - before.copied_unnamed_reach_bytes_compared,
        after.copied_unnamed_reach_twin_copy_ins - before.copied_unnamed_reach_twin_copy_ins,
        after.copied_unnamed_reach_twin_bytes_compared
            - before.copied_unnamed_reach_twin_bytes_compared,
    ];
    eprintln!(
        "READ-TWIN: a kernel reaching a buffer of bit 24 unnamed: copies in for that reach alone \
         [copy ins, bytes, twinned copy ins, twinned bytes] {counted:?}"
    );
    assert_eq!([counted[0], counted[2], counted[3]], [1, 0, 0]);
    assert!((1..=SMALL as u64).contains(&counted[1]), "{counted:?}");
    assert!(pages.bytes() == kept, "a kernel's stores leave the pages");
    drop((buffer, out, state, queue));
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
    // The negative control: with the guard off on this device, the same fill reaches the pages,
    // which the twin is in place.
    (device.shared().read_twin_unguarded).store(true, Ordering::Relaxed);
    let status = blit(&device, |encoder| {
        encoder.fill_buffer(&twin, 0..4096, 0xe1).unwrap();
    });
    (device.shared().read_twin_unguarded).store(false, Ordering::Relaxed);
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

/// Where the textures made between `before` and `after` lie: their counts in place, as
/// shadows, and why.
fn placed(before: &WorkCounts, after: &WorkCounts) -> String {
    let delta = |of: fn(&WorkCounts) -> u64| of(after) - of(before);
    format!(
        "twins {} (not in place {}, shadows {}), in place over host pages {}, over copies {}, \
         pitched {}, shadowed {}: pending {} failed {} unchecked {} refused rows {} image {} \
         pitch {} placement {} source {}",
        delta(|c| c.read_twin_textures_made),
        delta(|c| c.read_twin_textures_not_in_place),
        delta(|c| c.read_twin_texture_shadows),
        delta(|c| c.host_textures_in_place),
        delta(|c| c.copied_textures_in_place),
        delta(|c| c.pitched_host_images),
        delta(|c| c.shadowed_textures),
        delta(|c| c.host_texture_shadows_check_pending),
        delta(|c| c.host_texture_shadows_check_failed),
        delta(|c| c.host_texture_shadows_unchecked),
        delta(|c| c.host_texture_shadows_refused_rows),
        delta(|c| c.host_texture_shadows_refused_image),
        delta(|c| c.host_texture_shadows_refused_pitch),
        delta(|c| c.host_texture_shadows_refused_placement),
        delta(|c| c.host_texture_shadows_refused_source),
    )
}

/// The kernel that writes each texel of its texture 0 into its texture 1 inverted, its alpha
/// kept (`examples/runtime_examples.ll`'s `invert_texels`).
fn invert(device: &Device) -> ComputePipelineState {
    let function = super::encoder_holds::library(device, "examples/runtime_examples.ll")
        .new_function_with_name("invert_texels")
        .unwrap();
    device
        .new_compute_pipeline_state_with_function(&function, &crate::PipelineOptions::default())
        .unwrap()
}

/// The side of the square of texels a kernel inverts.
const SIDE: u64 = 64;

/// Commit one command buffer that runs `state` over the first `SIDE` x `SIDE` texels with
/// `source` at texture 0 and `destination` at texture 1, after a blit fill of `fill`'s buffer and
/// bytes where given, wait for it, and answer its status.
fn invert_texels(
    device: &Device,
    state: &ComputePipelineState,
    (source, destination): (&Texture, &Texture),
    fill: Option<(&Buffer, std::ops::Range<u64>, u8)>,
) -> CommandBufferStatus {
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    if let Some((buffer, bytes, value)) = fill {
        let mut encoder = command_buffer.blit_command_encoder().unwrap();
        encoder.fill_buffer(buffer, bytes, value).unwrap();
        encoder.end_encoding().unwrap();
    }
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(state).unwrap();
    encoder.set_texture(Some(source), 0).unwrap();
    encoder.set_texture(Some(destination), 1).unwrap();
    encoder
        .dispatch_threads(Size::new(SIDE, SIDE, 1), Size::new(8, 8, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    command_buffer.status()
}

/// A `Shared` texture of `format` (of 4-byte texels) of the device's own, `SIDE` texels square,
/// that kernels read and write.
fn square(device: &Device, format: PixelFormat) -> Texture {
    let mut descriptor = TextureDescriptor::texture_2d(format, SIDE, SIDE, false);
    descriptor.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
    descriptor.resource_options = ResourceOptions::shared();
    device.new_texture(&descriptor).unwrap()
}

/// The first `SIDE` x `SIDE` texels of `texture` as the CPU reads them, rows tight.
fn square_texels(texture: &Texture) -> Vec<u8> {
    let mut bytes = vec![0; (SIDE * SIDE * 4) as usize];
    let region = Region::new_2d(0, 0, SIDE, SIDE);
    texture
        .get_bytes(&mut bytes, SIDE * 4, 0, region, 0, 0)
        .unwrap();
    bytes
}

/// The first `SIDE` x `SIDE` texels of `bytes` (`BGRA8Unorm` or `RGBA8Unorm`), rows `row` bytes
/// apart, as `invert_texels` writes them: each colour channel inverted, its alpha kept.
fn inverted(bytes: &[u8], row: u64) -> Vec<u8> {
    let mut out = Vec::with_capacity((SIDE * SIDE * 4) as usize);
    for y in 0..SIDE {
        let start = (y * row) as usize;
        for texel in bytes[start..start + (SIDE * 4) as usize].chunks(4) {
            out.extend([255 - texel[0], 255 - texel[1], 255 - texel[2], texel[3]]);
        }
    }
    out
}

/// A texture of `format` (of 4-byte texels), `width` x `height`, of `usage`, over `buffer`'s
/// bytes from its first, rows tight.
fn texture_over(
    buffer: &Buffer,
    (format, width, height): (PixelFormat, u64, u64),
    usage: TextureUsage,
) -> Texture {
    let mut descriptor = TextureDescriptor::texture_2d(format, width, height, false);
    descriptor.usage = usage;
    descriptor.resource_options = ResourceOptions::shared();
    buffer.new_texture(&descriptor, 0, width * 4).unwrap()
}

/// The screen's texture (`WIDTH` x `HEIGHT` `BGRA8Unorm`, rows `ROW` bytes apart, `ShaderRead`)
/// over a buffer of bit 24 over `pages`, and whether it took a read twin that serves: one where
/// the device lays its linear image in the buffer's bytes, or where it does not (lavapipe's image
/// takes two rows more) and the texture is a shadow of its copy, a shadow of the twin's pages.
/// Where neither (the texture in place over its copy, its twin not over the pages), the texture
/// is read through itself, its texels the pages'.
fn screen_texture(device: &Device, pages: &Pages) -> (Buffer, Texture, bool) {
    let buffer = pages.private_bit_buffer(device, None);
    assert!(buffer.read_twin().is_some(), "a twin made");
    let before = device.work_counts();
    let shape = (PixelFormat::BGRA8Unorm, WIDTH, HEIGHT);
    let texture = texture_over(&buffer, shape, TextureUsage::SHADER_READ);
    let after = device.work_counts();
    eprintln!(
        "READ-TWIN: the screen's texture: {}",
        placed(&before, &after)
    );
    let twinned = after.read_twin_textures_made > before.read_twin_textures_made
        || after.read_twin_texture_shadows > before.read_twin_texture_shadows;
    if !twinned {
        let refused = after.host_texture_shadows_refused_placement
            - before.host_texture_shadows_refused_placement;
        assert_eq!(
            after.read_twin_textures_not_in_place - before.read_twin_textures_not_in_place,
            1
        );
        assert!(refused > 0, "the twin's image would not lie in the bytes");
    }
    (buffer, texture, twinned)
}

#[test]
fn a_screen_sized_private_bit_buffer_s_texture_is_read_in_place_by_a_kernel() {
    let pages = Pages::new(SCREEN, 13);
    let Some(device) = device(None) else {
        return;
    };
    let (buffer, texture, twinned) = screen_texture(&device, &pages);
    if !twinned {
        let state = invert(&device);
        let out = square(&device, PixelFormat::BGRA8Unorm);
        let before = device.work_counts();
        let status = invert_texels(&device, &state, (&texture, &out), None);
        let after = device.work_counts();
        assert_eq!(status, CommandBufferStatus::Completed);
        assert_eq!(after.read_twin_slot_reads, before.read_twin_slot_reads);
        assert!(
            square_texels(&out) == inverted(&pages.bytes(), ROW),
            "the pages' texels"
        );
        return;
    }
    read_by_a_kernel(&device, (&pages, &buffer), &texture, (0, ROW));
    drop((texture, buffer));
}

#[test]
fn a_screen_sized_private_bit_buffer_s_texture_is_sampled_in_place_by_a_draw() {
    let pages = Pages::new(SCREEN, 23);
    let Some(device) = device(None) else {
        return;
    };
    let (buffer, texture, twinned) = screen_texture(&device, &pages);
    let shape = (WIDTH, HEIGHT, ROW);
    if !twinned {
        let target = draw_target(&device);
        let own = reference(&device, shape, &pages.bytes());
        let expected = sampled(&device, &own, &target);
        let before = device.work_counts();
        let drawn = sampled(&device, &texture, &target);
        let after = device.work_counts();
        assert_eq!(after.read_twin_slot_reads, before.read_twin_slot_reads);
        assert!(drawn == expected, "the pages' texels sampled");
        return;
    }
    sampled_in_place(
        &device,
        (&pages, &buffer),
        &texture,
        (0, WIDTH, HEIGHT, ROW),
    );
    drop((texture, buffer));
}

#[test]
fn a_private_bit_buffer_s_texture_is_read_in_place_by_a_kernel() {
    let pages = Pages::new(SMALL, 37);
    let Some(device) = device(None) else {
        return;
    };
    let buffer = pages.private_bit_buffer(&device, None);
    let before = device.work_counts();
    let shape = (PixelFormat::BGRA8Unorm, SIDE, SIDE);
    let texture = texture_over(&buffer, shape, TextureUsage::SHADER_READ);
    let after = device.work_counts();
    assert_eq!(
        after.read_twin_textures_made - before.read_twin_textures_made,
        1,
        "the texture's twin made, in place: {}",
        placed(&before, &after)
    );
    read_by_a_kernel(&device, (&pages, &buffer), &texture, (0, SIDE * 4));
    drop((texture, buffer));
}

/// The shadow log (`shadow_log.rs`), on: a texture over a buffer of bit 24 at a pitch no driver
/// measured lays rows at (a 256-byte row at 272 bytes, `tests/texture_placement_device.rs`) takes
/// no twin in place, and a line names the texture and why; where the texture itself is a shadow,
/// the first dispatch that reads it names it, its slot, and that its function fetches it. Each
/// line once per texture, and at least one of them reached on every device.
#[test]
fn the_shadow_log_names_a_twin_not_in_place_and_a_dispatch_s_first_read() {
    crate::shadow_log::turn_on();
    let lines = Arc::new(Mutex::new(Vec::new()));
    let heard = Arc::clone(&lines);
    let sink = MessageSink::new(move |message| {
        if message.id_name == "metal2vulkan-shadows" {
            heard.lock().unwrap().push(message.text.clone());
        }
    });
    let pages = Pages::new(SMALL, 41);
    let Some(device) = device_with(None, Some(sink)) else {
        return;
    };
    let buffer = pages.private_bit_buffer(&device, None);
    let pitch = SIDE * 4 + 16;
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, SIDE, SIDE, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::shared();
    let before = device.work_counts();
    let texture = buffer.new_texture(&descriptor, 0, pitch).unwrap();
    let after = device.work_counts();
    let not_in_place =
        after.read_twin_textures_not_in_place - before.read_twin_textures_not_in_place;
    let shadowed = texture.shadow().is_some();
    // A shadow whose rows a kernel reads in place (`texture/texel_rows.rs`) binds them, not the
    // shadow.
    let kind = match after.texel_rows_textures > before.texel_rows_textures {
        true => "its rows in place, not its shadow",
        false => "a shadow of its truth",
    };
    let state = invert(&device);
    let out = square(&device, PixelFormat::BGRA8Unorm);
    eprintln!(
        "SHADOW-LOG: made: {}; rows {}",
        placed(&before, &after),
        after.texel_rows_textures - before.texel_rows_textures
    );
    for _ in 0..2 {
        let before = device.work_counts();
        let status = invert_texels(&device, &state, (&texture, &out), None);
        let after = device.work_counts();
        assert_eq!(status, CommandBufferStatus::Completed);
        eprintln!(
            "SHADOW-LOG: a dispatch: twin slot reads {}, compared {:?}, shadow copies {}, texels \
             copied in {}",
            after.read_twin_slot_reads - before.read_twin_slot_reads,
            compared(&before, &after),
            after.shadow_copies - before.shadow_copies,
            after.shadow_texels_copied_in - before.shadow_texels_copied_in,
        );
    }
    assert!(
        square_texels(&out) == inverted(&pages.bytes(), pitch),
        "the pages' texels"
    );
    let lines = lines.lock().unwrap().clone();
    let named = |what: &str| -> Vec<String> {
        (lines.iter())
            .filter(|line| line.contains(what))
            .cloned()
            .collect()
    };
    let texture_named = format!(
        "BGRA8Unorm texture of {SIDE}x{SIDE} and 1 levels, object {}",
        texture.object_id().get()
    );
    let twin = named(&format!("the read twin of the {texture_named}, "));
    let dispatch = named(&format!(
        "the first dispatch that binds the {texture_named}, through {kind}, at Kernel texture 0: \
         its function fetches it"
    ));
    eprintln!(
        "SHADOW-LOG: twins not in place {not_in_place}, the texture a shadow {shadowed}: {}",
        placed(&before, &after)
    );
    for line in twin.iter().chain(&dispatch) {
        eprintln!("SHADOW-LOG: {line}");
    }
    assert_eq!(
        twin.len() as u64,
        not_in_place,
        "a line per twin not in place"
    );
    if let Some(line) = twin.first() {
        assert!(line.contains(&format!(
            "at offset 0 with rows of {pitch} bytes, is not in place"
        )));
        if line.contains("its making made a shadow over the twin") {
            let made = named(&format!(
                "make_twin_texture made a BGRA8Unorm texture of {SIDE}x{SIDE}"
            ));
            assert_eq!(made.len(), 1, "the twin's shadow names its own cause");
        }
    }
    assert_eq!(
        dispatch.len(),
        usize::from(shadowed),
        "the first dispatch alone"
    );
    assert!(
        not_in_place > 0 || shadowed,
        "the shape reached neither line on this device"
    );
    drop((texture, buffer));
}

/// A texture of `shape` (`BGRA8Unorm`, `ShaderRead`, `width` x `height` at `offset` with rows
/// `row` bytes apart) over `buffer`, of bit 24, under RADV's rule for a `DRM_FORMAT_MOD_LINEAR`
/// plane, forced on the test's device (`DeviceShared::linear_pitch_rule_forced`, 256): no image of
/// it lies over the buffer's copy or its twin's pages, so the texture is a shadow of the copy and
/// its twin, kept, a shadow of the pages (`WorkCounts::read_twin_texture_shadows`).
fn refused_twin(
    device: &Device,
    buffer: &Buffer,
    (offset, width, height, row): (u64, u64, u64, u64),
) -> Texture {
    (device.shared().linear_pitch_rule_forced).store(256, Ordering::Relaxed);
    assert!(buffer.read_twin().is_some(), "a twin made");
    let mut descriptor =
        TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, width, height, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::shared();
    let before = device.work_counts();
    let texture = buffer.new_texture(&descriptor, offset, row).unwrap();
    let after = device.work_counts();
    eprintln!(
        "READ-TWIN: a {width}x{height} texture at offset {offset}, rows {row} bytes apart, under \
         the rule: {}",
        placed(&before, &after)
    );
    let delta = |of: fn(&WorkCounts) -> u64| of(&after) - of(&before);
    assert!(
        texture.shadow().is_some(),
        "the texture a shadow of its copy"
    );
    assert_eq!(
        [
            delta(|c| c.read_twin_textures_made),
            delta(|c| c.read_twin_textures_not_in_place),
            delta(|c| c.read_twin_texture_shadows),
        ],
        [0, 1, 1],
        "its twin not in place, kept as a shadow of the pages: (made, not in place, shadows)"
    );
    texture
}

/// The shapes of a window server's textures over buffers of bit 24 that RADV lays no image of: 48
/// x 48 at rows of 192 bytes, and 128 x 128 at rows of 512 bytes from 32 bytes in; and 64 x 64 at
/// rows of 272 bytes, 16 past the texels.
const REFUSED: [(u64, u64, u64, u64); 3] =
    [(0, 48, 48, 192), (32, 128, 128, 512), (0, 64, 64, 272)];

#[test]
fn a_private_bit_buffer_s_texture_no_image_lies_over_is_sampled_through_a_shadow_of_its_twin() {
    let pages = [43, 47, 61].map(|seed| Pages::new(2 * SMALL, seed));
    let Some(device) = device(None) else {
        return;
    };
    for (pages, shape) in pages.iter().zip(REFUSED) {
        let buffer = pages.private_bit_buffer(&device, None);
        let texture = refused_twin(&device, &buffer, shape);
        sampled_in_place(&device, (pages, &buffer), &texture, shape);
        drop((texture, buffer));
    }
}

#[test]
fn a_private_bit_buffer_s_texture_no_image_lies_over_is_read_through_a_shadow_of_its_twin() {
    let pages = [53, 67].map(|seed| Pages::new(2 * SMALL, seed));
    let Some(device) = device(None) else {
        return;
    };
    for (pages, shape) in pages.iter().zip([REFUSED[1], REFUSED[2]]) {
        let buffer = pages.private_bit_buffer(&device, None);
        let texture = refused_twin(&device, &buffer, shape);
        read_by_a_kernel(&device, (pages, &buffer), &texture, (shape.0, shape.3));
        drop((texture, buffer));
    }
}

/// The row pitch of the driver's own linear `B8G8R8A8_UNORM` image `width` texels wide and
/// `SIDE` tall, of sampled and transfer usage (`vkGetImageSubresourceLayout`), made and destroyed
/// unbound; `None` where the driver makes none.
fn linear_rows(device: &Device, width: u32) -> Option<u64> {
    // SAFETY: the handles are used while the device lives, for an image the test destroys.
    let handles = unsafe { device.raw() };
    let info = ash::vk::ImageCreateInfo::default()
        .image_type(ash::vk::ImageType::TYPE_2D)
        .format(ash::vk::Format::B8G8R8A8_UNORM)
        .extent(ash::vk::Extent3D {
            width,
            height: SIDE as u32,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(ash::vk::SampleCountFlags::TYPE_1)
        .tiling(ash::vk::ImageTiling::LINEAR)
        .usage(
            ash::vk::ImageUsageFlags::SAMPLED
                | ash::vk::ImageUsageFlags::TRANSFER_SRC
                | ash::vk::ImageUsageFlags::TRANSFER_DST,
        )
        .sharing_mode(ash::vk::SharingMode::EXCLUSIVE)
        .initial_layout(ash::vk::ImageLayout::UNDEFINED);
    let color = ash::vk::ImageSubresource {
        aspect_mask: ash::vk::ImageAspectFlags::COLOR,
        mip_level: 0,
        array_layer: 0,
    };
    // SAFETY: the device is live and `info` valid; the image is destroyed after the query.
    let image = unsafe { handles.device.create_image(&info, None) }.ok()?;
    // SAFETY: a live linear image of a color format, destroyed with no memory bound.
    unsafe {
        let layout = handles.device.get_image_subresource_layout(image, color);
        handles.device.destroy_image(image, None);
        Some(layout.row_pitch)
    }
}

/// The pitches [`textures_at_padded_rows_read_their_bytes`] lays rows at: tight (`SIDE` texels of
/// 4 bytes) and padded past that by 16 to 768 bytes.
const PADDED: [u64; 6] = [256, 272, 288, 320, 384, 1024];

/// Textures over plain buffers' bytes (no bit 24, so no twin), 64 x 64 `BGRA8Unorm`, at each of
/// [`PADDED`]'s pitches, over a no-copy buffer's host pages and over a buffer of the device's own
/// memory, read by a kernel and copied out by a blit: each reads the texels its bytes hold,
/// whether it lies in place over them (by the driver's own linear image or a
/// `DRM_FORMAT_MOD_LINEAR` one at Metal's pitch: `pitched_host_images`, `pooled_pitched_images`)
/// or reads its shadow. On a driver whose shaders read an explicit plane's rows at another pitch
/// where they lie a number of bytes apart that is no multiple of 32
/// (`RuntimeDriverDefect::ExplicitPlanePitchMisread`: NVIDIA's, which reads rows 272 bytes apart
/// at 256), no texture at such a pitch lies in such an image. Printed (`PADDED-ROWS`): the pitch
/// each read's texels lie at, if any multiple of 16 bytes from 256 to 1024.
#[test]
fn textures_at_padded_rows_read_their_bytes() {
    padded_rows(false);
}

/// [`textures_at_padded_rows_read_their_bytes`] on a device that takes its driver for one whose
/// shaders read an explicit plane's rows at another pitch where they lie a number of bytes apart
/// that is no multiple of 32 (`DeviceShared::explicit_plane_pitch_misread_forced`): the texture
/// at rows 272 bytes apart lies in no `DRM_FORMAT_MOD_LINEAR` image at that pitch, and reads its
/// texels; on lavapipe, which lays such an image at each pitch, it is the one the rule changes.
#[test]
fn textures_at_padded_rows_read_their_bytes_where_the_driver_misreads_the_pitch() {
    padded_rows(true);
}

/// [`textures_at_padded_rows_read_their_bytes`], on a device whose driver is taken for one that
/// misreads an explicit plane's pitch where `forced`.
fn padded_rows(forced: bool) {
    // Bytes of no period a pitch here divides (`pattern`'s repeats every 256 bytes).
    let scattered: Vec<u8> = (0..SMALL as u32)
        .map(|at| (at.wrapping_mul(2_654_435_761) >> 24) as u8 | 1)
        .collect();
    let pages = PADDED.map(|_| Pages::new(SMALL, 0));
    pages.iter().for_each(|pages| pages.write(0, &scattered));
    let Some(device) = device(None) else {
        return;
    };
    (device.shared().explicit_plane_pitch_misread_forced).store(forced, Ordering::Relaxed);
    let defect = metal2vulkan::error::RuntimeDriverDefect::ExplicitPlanePitchMisread;
    let misread = forced || device.properties().driver_defects.contains(&defect);
    let state = invert(&device);
    let out = square(&device, PixelFormat::BGRA8Unorm);
    let copy = square(&device, PixelFormat::BGRA8Unorm);
    let queue = device.new_command_queue().unwrap();
    // The forced rule is not the driver's: no alignment probe runs under it, as its answers are
    // kept per driver (`texture/linear_alignment.rs`).
    if !forced {
        eprintln!(
            "PADDED-ROWS: minimumLinearTextureAlignmentForPixelFormat(BGRA8Unorm) {}; the \
             driver's linear image's rows for widths of 1, 3, 17, 64, 65, 68, 72, 80, 96 and 100 \
             texels: {:?}",
            (device.minimum_linear_texture_alignment_for_pixel_format(PixelFormat::BGRA8Unorm))
                .unwrap(),
            [1, 3, 17, 64, 65, 68, 72, 80, 96, 100].map(|width| linear_rows(&device, width))
        );
    }
    let mut wrong = Vec::new();
    for (pages, pitch) in pages.iter().zip(PADDED) {
        let bytes = pages.bytes();
        // SAFETY: the pages outlive the buffer and the device (dropped after both).
        let memory = unsafe { HostMemory::new(pages.pointer) };
        // SAFETY: as above.
        let host = unsafe {
            device.new_buffer_with_bytes_no_copy(
                memory,
                SMALL as u64,
                ResourceOptions::shared(),
                None,
            )
        }
        .unwrap();
        let own = (device.new_buffer_with_bytes(&bytes, ResourceOptions::shared())).unwrap();
        for (what, buffer) in [("host pages", &host), ("own memory", &own)] {
            let mut descriptor =
                TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, SIDE, SIDE, false);
            descriptor.usage = TextureUsage::SHADER_READ;
            descriptor.resource_options = ResourceOptions::shared();
            let before = device.work_counts();
            let texture = buffer.new_texture(&descriptor, 0, pitch).unwrap();
            let after = device.work_counts();
            let status = invert_texels(&device, &state, (&texture, &out), None);
            assert_eq!(status, CommandBufferStatus::Completed);
            super::merge_scratch::blit(&queue, |encoder| {
                let size = Size::new(SIDE, SIDE, 1);
                let origin = Origin::new(0, 0, 0);
                (encoder.copy_from_texture_to_texture(
                    &texture, 0, 0, origin, size, &copy, 0, 0, origin,
                ))
                .unwrap()
            });
            let lies = |texels: &[u8], invert: bool| {
                (SIDE * 4..=1024).step_by(16).find(|&at| {
                    let at_pitch = inverted(&bytes, at);
                    let at_pitch = if invert {
                        at_pitch
                    } else {
                        inverted(&at_pitch, SIDE * 4)
                    };
                    texels == at_pitch
                })
            };
            let [kernel, blit] = [
                lies(&square_texels(&out), true),
                lies(&square_texels(&copy), false),
            ];
            let pitched = (after.pitched_host_images - before.pitched_host_images)
                + (after.pooled_pitched_images - before.pooled_pitched_images);
            eprintln!(
                "PADDED-ROWS: over {what}, rows of {pitch} bytes (misread pitches {misread}): a \
                 kernel reads the rows at {kernel:?}, a blit at {blit:?}: {}; pooled pitched {}",
                placed(&before, &after),
                after.pooled_pitched_images - before.pooled_pitched_images
            );
            if kernel != Some(pitch) || blit != Some(pitch) {
                wrong.push((what, pitch, kernel, blit));
            }
            if misread && !pitch.is_multiple_of(32) {
                assert_eq!(pitched, 0, "over {what}, rows of {pitch} bytes: pitched");
            }
        }
        drop((host, own));
    }
    assert!(wrong.is_empty(), "read at other pitches: {wrong:?}");
}

/// The gated copies in of shadows over logged pages between two counts: gated, whole, skipped,
/// the bytes a whole copy would move, and those moved (`dirty_pages.rs`).
fn gated(before: &WorkCounts, after: &WorkCounts) -> [u64; 5] {
    [
        after.dirty_copies_in_gated - before.dirty_copies_in_gated,
        after.dirty_copies_in_whole - before.dirty_copies_in_whole,
        after.dirty_copies_in_skipped - before.dirty_copies_in_skipped,
        after.dirty_copy_in_bytes_whole - before.dirty_copy_in_bytes_whole,
        after.dirty_copy_in_bytes_copied - before.dirty_copy_in_bytes_copied,
    ]
}

/// Over pages whose memory came with a log of the writes to them (`HostMemory::with_dirty_pages`),
/// the twin's pages carry the log too: a kernel that reads the texture through the twin's shadow
/// copies every row in at the first command buffer, nothing at the next where the log reports no
/// write, and only the rows it reports after a CPU write it was told of; no commit compares.
#[test]
fn a_shadow_of_a_twin_over_logged_pages_copies_in_only_what_the_log_reports() {
    let pages = Pages::new(2 * SMALL, 59);
    let log = Arc::new(Log::default());
    let Some(device) = device(None) else {
        return;
    };
    let buffer = pages.logged_private_bit_buffer(&device, &log);
    let (offset, width, height, row) = REFUSED[1];
    let texture = refused_twin(&device, &buffer, (offset, width, height, row));
    let state = invert(&device);
    let out = square(&device, PixelFormat::BGRA8Unorm);
    let texels = |bytes: Vec<u8>| bytes[offset as usize..].to_vec();
    let frame = |what: &str| {
        let before = device.work_counts();
        let status = invert_texels(&device, &state, (&texture, &out), None);
        let after = device.work_counts();
        assert_eq!(status, CommandBufferStatus::Completed);
        assert_eq!(after.read_twin_slot_reads - before.read_twin_slot_reads, 1);
        assert_eq!(
            compared(&before, &after),
            [0, 0, 0],
            "{what}: nothing compared"
        );
        assert!(
            square_texels(&out) == inverted(&texels(pages.bytes()), row),
            "{what}: the pages' texels"
        );
        let moved = gated(&before, &after);
        eprintln!(
            "READ-TWIN: over logged pages, {what}: (gated, whole, skipped, bytes whole, copied) \
             {moved:?}"
        );
        moved
    };
    let first = frame("the first command buffer");
    assert_eq!(first[0], 1, "the first copy in is gated");
    assert!(first[4] > 0 && first[4] == first[3], "every row");
    let unchanged = frame("no write since");
    assert_eq!(unchanged[2], 1, "the copy in skipped");
    assert_eq!(unchanged[4], 0, "nothing moved");
    // Row 10's first texels, written and told.
    let at = offset + 10 * row;
    pages.write(at as usize, &[0x42; 64]);
    log.told(at..at + 64, 0);
    let told = frame("a told write");
    assert!(told[4] > 0 && told[4] < told[3], "the told rows alone");
    drop((texture, buffer));
}

/// A kernel reads `texture`, over `buffer`'s bytes (of `pages`) from `offset` in rows `row` bytes
/// apart, which took a read twin: through its twin, the commit comparing nothing; a CPU write of
/// the pages read in place; after a fill of the buffer in the same command buffer, through the
/// texture itself, the fill read back; and the next command buffer through its twin again.
fn read_by_a_kernel(
    device: &Device,
    (pages, buffer): (&Pages, &Buffer),
    texture: &Texture,
    (offset, row): (u64, u64),
) {
    let state = invert(device);
    let out = square(device, PixelFormat::BGRA8Unorm);
    let texels = |bytes: Vec<u8>| bytes[offset as usize..].to_vec();
    let before = device.work_counts();
    let status = invert_texels(device, &state, (texture, &out), None);
    let after = device.work_counts();
    assert_eq!(status, CommandBufferStatus::Completed);
    assert_eq!(after.read_twin_slot_reads - before.read_twin_slot_reads, 1);
    assert_eq!(compared(&before, &after), [0, 0, 0], "nothing compared");
    eprintln!(
        "READ-TWIN: a kernel reading a texture over a buffer of bit 24, rows {row} bytes apart: \
         copies in compared {:?} (copies, bytes, threads)",
        compared(&before, &after)
    );
    assert!(
        square_texels(&out) == inverted(&texels(pages.bytes()), row),
        "the pages' texels"
    );
    // A CPU write of the pages reaches the next command buffer's read in place.
    pages.write(offset as usize, &[0x42; 64]);
    let status = invert_texels(device, &state, (texture, &out), None);
    assert_eq!(status, CommandBufferStatus::Completed);
    assert!(
        square_texels(&out) == inverted(&texels(pages.bytes()), row),
        "the CPU's write read"
    );
    // A fill of the buffer, then the kernel, in one command buffer: it reads the texture itself,
    // the fill read back, and the commit compares the copy.
    let kept = pages.bytes();
    let before = device.work_counts();
    let fill = Some((buffer, 0..offset + row * SIDE, 0x5a));
    let status = invert_texels(device, &state, (texture, &out), fill);
    let after = device.work_counts();
    assert_eq!(status, CommandBufferStatus::Completed);
    assert_eq!(after.read_twin_slot_reads, before.read_twin_slot_reads);
    assert_eq!(
        after.read_twin_reads_after_writes - before.read_twin_reads_after_writes,
        1
    );
    assert_eq!(compared(&before, &after)[0], 1, "its copy compared");
    assert!(
        square_texels(&out) == inverted(&vec![0x5a; (row * SIDE) as usize], row),
        "the fill read back"
    );
    assert!(pages.bytes() == kept, "the fill never reaches the pages");
    // The next command buffer reads the pages in place again.
    let before = device.work_counts();
    let status = invert_texels(device, &state, (texture, &out), None);
    let after = device.work_counts();
    assert_eq!(status, CommandBufferStatus::Completed);
    assert_eq!(after.read_twin_slot_reads - before.read_twin_slot_reads, 1);
    assert_eq!(compared(&before, &after), [0, 0, 0]);
    assert!(
        square_texels(&out) == inverted(&texels(kept), row),
        "the pages read again"
    );
    assert_eq!(after.read_twin_writes_refused, 0);
}

#[test]
fn a_texture_twin_bound_without_the_slots_proof_is_refused_and_would_reach_the_pages() {
    let pages = Pages::new(SMALL, 17);
    let Some(device) = device(None) else {
        return;
    };
    let buffer = pages.private_bit_buffer(&device, None);
    let usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
    let before = device.work_counts();
    let texture = texture_over(&buffer, (PixelFormat::RGBA8Unorm, SIDE, SIDE), usage);
    let after = device.work_counts();
    assert_eq!(
        after.read_twin_textures_made - before.read_twin_textures_made,
        1,
        "the texture's twin made, in place"
    );
    let state = invert(&device);
    let source = square(&device, PixelFormat::RGBA8Unorm);
    let texels = pattern((SIDE * SIDE * 4) as usize, 0x70);
    let region = Region::new_2d(0, 0, SIDE, SIDE);
    source
        .replace_region(region, 0, 0, &texels, SIDE * 4, 0)
        .unwrap();
    let kept = pages.bytes();
    // The proof: a kernel that writes the texture binds the texture itself, its writes dropped.
    let before = device.work_counts();
    let status = invert_texels(&device, &state, (&source, &texture), None);
    let after = device.work_counts();
    assert_eq!(status, CommandBufferStatus::Completed);
    assert_eq!(after.read_twin_slot_reads, before.read_twin_slot_reads);
    assert!(pages.bytes() == kept, "a kernel's writes leave the pages");
    // Bound without the proof, the twin's write is refused by the guard.
    (device.shared().read_twin_unproven).store(true, Ordering::Relaxed);
    let before = device.work_counts();
    let status = invert_texels(&device, &state, (&source, &texture), None);
    let after = device.work_counts();
    let seen = (
        status,
        after.read_twin_slot_reads - before.read_twin_slot_reads,
        after.read_twin_writes_refused - before.read_twin_writes_refused,
    );
    eprintln!("READ-TWIN: a texture twin bound unproven: (status, slot reads, refused) {seen:?}");
    assert_eq!(
        seen,
        (CommandBufferStatus::Error, 1, 1),
        "the command buffer fails: (status, twin slot reads, writes refused)"
    );
    assert!(pages.bytes() == kept, "the guard keeps the pages");
    // The negative control: with the guard off too, the kernel's writes reach the pages.
    (device.shared().read_twin_unguarded).store(true, Ordering::Relaxed);
    let status = invert_texels(&device, &state, (&source, &texture), None);
    (device.shared().read_twin_unguarded).store(false, Ordering::Relaxed);
    (device.shared().read_twin_unproven).store(false, Ordering::Relaxed);
    assert_eq!(status, CommandBufferStatus::Completed);
    assert!(
        pages.bytes()[..(SIDE * SIDE * 4) as usize] == inverted(&texels, SIDE * 4)[..],
        "the pages hold the kernel's texels"
    );
    drop((texture, source, state, buffer));
}

/// One command buffer that clears `target` and draws a triangle over it sampling `source`
/// (`examples/runtime_examples.ll`'s `sampled_fragment`), after a blit fill of `fill`'s buffer
/// and bytes where given, to completion: its status.
fn draw_sampled(
    device: &Device,
    (source, target): (&Texture, &Texture),
    fill: Option<(&Buffer, std::ops::Range<u64>, u8)>,
) -> CommandBufferStatus {
    let state = super::encoder_holds::render_state(
        device,
        "examples/runtime_examples.ll",
        "colored_vertex",
        "sampled_fragment",
    );
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    if let Some((buffer, bytes, value)) = fill {
        let mut encoder = command_buffer.blit_command_encoder().unwrap();
        encoder.fill_buffer(buffer, bytes, value).unwrap();
        encoder.end_encoding().unwrap();
    }
    let pass = super::encoder_holds::pass(target);
    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
    encoder.set_render_pipeline_state(&state).unwrap();
    let positions = [-1.0f32, -1.0, 3.0, -1.0, -1.0, 3.0];
    let positions = super::encoder_holds::bytes_of(&positions, f32::to_le_bytes);
    let colors = super::encoder_holds::bytes_of(&[1.0f32; 12], f32::to_le_bytes);
    encoder.set_vertex_bytes(&positions, 0).unwrap();
    encoder.set_vertex_bytes(&colors, 1).unwrap();
    encoder.set_fragment_texture(Some(source), 0).unwrap();
    encoder
        .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    command_buffer.status()
}

/// `target`'s `SIDE` x `SIDE` texels after a draw sampling `source` (`draw_sampled`, no fill).
fn sampled(device: &Device, source: &Texture, target: &Texture) -> Vec<u8> {
    let status = draw_sampled(device, (source, target), None);
    assert_eq!(status, CommandBufferStatus::Completed);
    square_texels(target)
}

/// The `SIDE` x `SIDE` target a draw samples a texture into.
fn draw_target(device: &Device) -> Texture {
    let mut plain = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
    plain.usage = TextureUsage::RENDER_TARGET;
    plain.resource_options = ResourceOptions::shared();
    device.new_texture(&plain).unwrap()
}

/// A `BGRA8Unorm` texture of the device's own, `width` x `height`, holding `bytes`'s texels in
/// rows `row` bytes apart: what a draw samples there is what it must sample through a twin over
/// the same bytes.
fn reference(device: &Device, (width, height, row): (u64, u64, u64), bytes: &[u8]) -> Texture {
    let mut descriptor =
        TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, width, height, false);
    descriptor.resource_options = ResourceOptions::shared();
    let texture = device.new_texture(&descriptor).unwrap();
    let region = Region::new_2d(0, 0, width, height);
    let texels = &bytes[..(row * height) as usize];
    (texture.replace_region(region, 0, 0, texels, row, 0)).unwrap();
    texture
}

/// A draw samples `texture` (`BGRA8Unorm`, `width` x `height`, rows `row` bytes apart over
/// `buffer`'s bytes from `offset`, of `pages`), which took a read twin: through its twin, the
/// commit comparing nothing; after a fill of the buffer in the same command buffer, through the
/// texture itself, the fill read back; and the next command buffer through its twin again.
fn sampled_in_place(
    device: &Device,
    (pages, buffer): (&Pages, &Buffer),
    texture: &Texture,
    (offset, width, height, row): (u64, u64, u64, u64),
) {
    let target = draw_target(device);
    let own = reference(
        device,
        (width, height, row),
        &pages.bytes()[offset as usize..],
    );
    let expected = sampled(device, &own, &target);
    let before = device.work_counts();
    let drawn = sampled(device, texture, &target);
    let after = device.work_counts();
    assert_eq!(after.read_twin_slot_reads - before.read_twin_slot_reads, 1);
    assert_eq!(compared(&before, &after), [0, 0, 0], "nothing compared");
    eprintln!(
        "READ-TWIN: a draw sampling a {width}x{height} texture over a buffer of bit 24: copies \
         in compared {:?} (copies, bytes, threads)",
        compared(&before, &after)
    );
    assert!(drawn == expected, "the pages' texels sampled");
    // A fill of the buffer, then the draw, in one command buffer: it samples the texture itself,
    // the fill read back.
    let bytes = row * height;
    let filled = vec![0x5a; bytes as usize];
    let own = reference(device, (width, height, row), &filled);
    let expected_filled = sampled(device, &own, &target);
    let kept = pages.bytes();
    let before = device.work_counts();
    let fill = Some((buffer, 0..offset + bytes, 0x5a));
    let status = draw_sampled(device, (texture, &target), fill);
    let after = device.work_counts();
    assert_eq!(status, CommandBufferStatus::Completed);
    assert_eq!(after.read_twin_slot_reads, before.read_twin_slot_reads);
    assert_eq!(
        after.read_twin_reads_after_writes - before.read_twin_reads_after_writes,
        1
    );
    assert!(
        square_texels(&target) == expected_filled,
        "the fill read back"
    );
    assert!(pages.bytes() == kept, "the fill never reaches the pages");
    assert!(
        sampled(device, texture, &target) == expected,
        "the next command buffer samples the pages"
    );
    assert_eq!(device.work_counts().read_twin_writes_refused, 0);
}

#[test]
fn a_private_bit_buffer_s_texture_is_sampled_in_place_by_a_draw() {
    let pages = Pages::new(SMALL, 19);
    let Some(device) = device(None) else {
        return;
    };
    let buffer = pages.private_bit_buffer(&device, None);
    let before = device.work_counts();
    let shape = (PixelFormat::BGRA8Unorm, SIDE, SIDE);
    let texture = texture_over(&buffer, shape, TextureUsage::SHADER_READ);
    let after = device.work_counts();
    assert_eq!(
        after.read_twin_textures_made - before.read_twin_textures_made,
        1,
        "the texture's twin made, in place"
    );
    sampled_in_place(
        &device,
        (&pages, &buffer),
        &texture,
        (0, SIDE, SIDE, SIDE * 4),
    );
    drop((texture, buffer));
}

/// Run `step` on a thread of its own, failing the test where it has not returned within a
/// minute: a step that waits for a lock it holds itself never returns.
fn returns(what: &str, step: impl FnOnce() + Send + 'static) {
    let (done, finished) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        step();
        let _ = done.send(());
    });
    let waited = finished.recv_timeout(Duration::from_secs(60));
    assert!(waited.is_ok(), "{what} never returned");
}

/// A rebind of a buffer of bit 24 that took a read twin, and a detach of one, each return and let
/// the twin go: the twin's going forgets its key among the device's rebinds, whose lock the move
/// holds, so it goes only once the move let go of that lock (`buffer::LeftTwins`).
#[test]
fn a_rebind_or_a_detach_of_a_buffer_with_a_twin_lets_the_twin_go() {
    let (pages, other, third) = (
        Pages::new(SMALL, 41),
        Pages::new(SMALL, 43),
        Pages::new(SMALL, 47),
    );
    let Some(device) = device(None) else {
        return;
    };
    let buffer = pages.private_bit_buffer(&device, None);
    assert!(buffer.read_twin().is_some(), "a twin made");
    let rebound = buffer.clone();
    let to = other.pointer.as_ptr() as usize;
    returns("the rebind", move || {
        let to = NonNull::new(to as *mut u8).unwrap();
        // SAFETY: the other pages outlive the buffer and the device (dropped after both), and no
        // other thread encodes or commits.
        let memory = unsafe { HostMemory::new(to) };
        // SAFETY: as above.
        unsafe { rebound.rebind_bytes_no_copy(memory, None, None) }.unwrap();
    });
    assert!(
        buffer.read_twin().is_none(),
        "the twin goes with the old pages"
    );
    assert_eq!(buffer.contents(), Some(other.pointer));
    let detached = third.private_bit_buffer(&device, None);
    assert!(detached.read_twin().is_some(), "a twin made");
    let detaching = detached.clone();
    returns("the detach", move || {
        detaching.detach_backing().unwrap();
    });
    assert!(
        detached.read_twin().is_none(),
        "the twin goes with the pages"
    );
    drop((buffer, detached));
}

/// The words a kernel reads at a slot: `count` floats, each a multiple of a quarter.
fn quarters(count: usize) -> Vec<u8> {
    (0..count)
        .flat_map(|at| (at as f32 * 0.25).to_le_bytes())
        .collect()
}

/// What `scale_and_offset` writes for `input`'s floats, scaled by 2: `input[i] * 2 + i`.
fn scaled(input: &[u8]) -> Vec<u8> {
    (input.as_chunks::<4>().0.iter().enumerate())
        .flat_map(|(at, word)| (f32::from_le_bytes(*word) * 2.0 + at as f32).to_le_bytes())
        .collect()
}

/// How many floats a kernel scales.
const FLOATS: usize = 1024;

/// Commit one command buffer that runs `examples/runtime_examples.ll`'s `scale_and_offset` over
/// `FLOATS` threads, reading `input` at buffer 0 and writing `output` at buffer 1, scaled by 2,
/// after a blit fill of `fill`'s buffer and bytes where given, wait for it, and answer its status.
fn scale(
    device: &Device,
    (input, output): (&Buffer, &Buffer),
    fill: Option<(&Buffer, std::ops::Range<u64>, u8)>,
) -> CommandBufferStatus {
    let function = super::encoder_holds::library(device, "examples/runtime_examples.ll")
        .new_function_with_name("scale_and_offset")
        .unwrap();
    let state = device
        .new_compute_pipeline_state_with_function(&function, &crate::PipelineOptions::default())
        .unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    if let Some((buffer, bytes, value)) = fill {
        let mut encoder = command_buffer.blit_command_encoder().unwrap();
        encoder.fill_buffer(buffer, bytes, value).unwrap();
        encoder.end_encoding().unwrap();
    }
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(&state).unwrap();
    encoder.set_buffer(Some(input), 0, 0).unwrap();
    encoder.set_buffer(Some(output), 0, 1).unwrap();
    encoder.set_bytes(&2.0f32.to_le_bytes(), 2).unwrap();
    encoder
        .dispatch_threads(Size::new(FLOATS as u64, 1, 1), Size::new(64, 1, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    command_buffer.status()
}

#[test]
fn a_kernel_reading_a_private_bit_buffer_at_a_slot_reads_its_pages_in_place() {
    let pages = Pages::new(SMALL, 23);
    pages.write(0, &quarters(FLOATS));
    let Some(device) = device(None) else {
        return;
    };
    let buffer = pages.private_bit_buffer(&device, None);
    assert!(buffer.read_twin().is_some(), "a twin made");
    let output = own(&device, FLOATS * 4);
    // Read through its twin: the pages' floats, the copy compared nowhere.
    let before = device.work_counts();
    let status = scale(&device, (&buffer, &output), None);
    let after = device.work_counts();
    assert_eq!(status, CommandBufferStatus::Completed);
    assert_eq!(after.read_twin_slot_reads - before.read_twin_slot_reads, 1);
    assert_eq!(compared(&before, &after), [0, 0, 0], "nothing compared");
    eprintln!(
        "READ-TWIN: a kernel reading a buffer of bit 24 at a slot: copies in compared {:?} \
         (copies, bytes, threads)",
        compared(&before, &after)
    );
    let expected = scaled(&pages.bytes()[..FLOATS * 4]);
    assert!(
        contents(&output, FLOATS * 4) == expected,
        "the pages' floats"
    );
    // A fill of the buffer, then the kernel, in one command buffer: it reads the copy, the fill
    // read back, and the commit compares the copy.
    let kept = pages.bytes();
    let before = device.work_counts();
    let fill = Some((&buffer, 0..(FLOATS * 4) as u64, 0x40));
    let status = scale(&device, (&buffer, &output), fill);
    let after = device.work_counts();
    assert_eq!(status, CommandBufferStatus::Completed);
    assert_eq!(after.read_twin_slot_reads, before.read_twin_slot_reads);
    assert_eq!(
        after.read_twin_reads_after_writes - before.read_twin_reads_after_writes,
        1
    );
    assert_eq!(compared(&before, &after)[0], 1, "its copy compared");
    let filled = vec![0x40; FLOATS * 4];
    assert!(
        contents(&output, FLOATS * 4) == scaled(&filled),
        "the fill read back"
    );
    assert!(pages.bytes() == kept, "the fill never reaches the pages");
    // The next command buffer reads the pages in place again.
    let status = scale(&device, (&buffer, &output), None);
    assert_eq!(status, CommandBufferStatus::Completed);
    assert!(
        contents(&output, FLOATS * 4) == expected,
        "the pages read again"
    );
    assert_eq!(device.work_counts().read_twin_writes_refused, 0);
    drop((output, buffer));
}

#[test]
fn a_buffer_twin_bound_without_the_slots_proof_is_refused_and_would_reach_the_pages() {
    let pages = Pages::new(SMALL, 29);
    let Some(device) = device(None) else {
        return;
    };
    let buffer = pages.private_bit_buffer(&device, None);
    assert!(buffer.read_twin().is_some(), "a twin made");
    let input = own(&device, FLOATS * 4);
    let floats = quarters(FLOATS);
    // SAFETY: the buffer's bytes, which no command uses now.
    unsafe {
        let at = input.contents().unwrap().as_ptr();
        std::ptr::copy_nonoverlapping(floats.as_ptr(), at, floats.len());
    }
    let kept = pages.bytes();
    // The proof: a kernel that writes the buffer at a slot binds the buffer itself, its writes
    // dropped.
    let before = device.work_counts();
    let status = scale(&device, (&input, &buffer), None);
    let after = device.work_counts();
    assert_eq!(status, CommandBufferStatus::Completed);
    assert_eq!(after.read_twin_slot_reads, before.read_twin_slot_reads);
    assert!(pages.bytes() == kept, "a kernel's writes leave the pages");
    // Bound without the proof, the twin's write is refused by the guard.
    (device.shared().read_twin_unproven).store(true, Ordering::Relaxed);
    let before = device.work_counts();
    let status = scale(&device, (&input, &buffer), None);
    let after = device.work_counts();
    assert_eq!(
        status,
        CommandBufferStatus::Error,
        "the command buffer fails"
    );
    assert_eq!(after.read_twin_slot_reads - before.read_twin_slot_reads, 1);
    assert_eq!(
        after.read_twin_writes_refused - before.read_twin_writes_refused,
        1
    );
    assert!(pages.bytes() == kept, "the guard keeps the pages");
    // The negative control: with the guard off too, the kernel's writes reach the pages.
    (device.shared().read_twin_unguarded).store(true, Ordering::Relaxed);
    let status = scale(&device, (&input, &buffer), None);
    (device.shared().read_twin_unguarded).store(false, Ordering::Relaxed);
    (device.shared().read_twin_unproven).store(false, Ordering::Relaxed);
    assert_eq!(status, CommandBufferStatus::Completed);
    assert!(
        pages.bytes()[..FLOATS * 4] == scaled(&floats)[..],
        "bound without the proof and unguarded, a kernel's writes reach the pages"
    );
    drop((input, buffer));
}
