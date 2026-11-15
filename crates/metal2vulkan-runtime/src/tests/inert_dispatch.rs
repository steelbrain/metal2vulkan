//! A dispatch of an inert kernel (`ComputePipelineState::inert`: its module holds no loop and
//! writes nothing but its threads' own and its threadgroups' memory) records no Vulkan dispatch
//! (`compute::record`, `WorkCounts::dispatches_skipped_inert`), and what orders around it still
//! orders:
//!
//! - **Skipped.** A kernel that does nothing, dispatched in threadgroups of one thread, as threads
//!   and indirectly, and a kernel that writes only threadgroup memory: each dispatch is counted.
//! - **Recorded.** A kernel that stores to a device buffer writes its words, and one that loops
//!   though it writes only threadgroup memory runs: neither is counted.
//! - **Fences.** An encoder whose only dispatch is skipped waits for one fence and updates another,
//!   and a later encoder that waits for the second reads what an earlier one wrote, over buffers
//!   whose hazards the runtime does not track.
//! - **Events.** A command buffer whose only dispatch is skipped waits for a shared event's value
//!   and signals the next, on a queue of its own, and a third command buffer that waits for that
//!   value reads what the first wrote; the three are committed in reverse order.
//! - **Uses.** The command buffer uses what a skipped dispatch binds, as Metal's does. A `Managed`
//!   buffer the CPU wrote and named with `did_modify_range` is copied in by the command buffer of
//!   the skipped dispatch that binds it, not by the next one that reads it, which copies nothing
//!   in and waits for no fill-queue submission. A `Shared` no-copy buffer detached before a
//!   skipped dispatch binds it fails that command buffer (`DetachedBacking`).
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The module is LLVM text written here
//! (`llvm-as`).

use super::blit_holds::{device, plan_device, Scratch, WAIT};
use super::opencl_mode::{words, words_buffer};
use crate::{
    Buffer, CommandBufferError, CommandBufferStatus, ComputePipelineState, CpuCacheMode, Device,
    EventRef, HazardTrackingMode, HostMemory, PipelineOptions, ResourceOptions, Size, StorageMode,
};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use std::ptr::NonNull;

/// Five kernels of buffers `a` (index 0) and `b` (index 1) and the thread's grid position:
/// `empty` does nothing; `tg_only` stores to threadgroup memory; `loop_tg` stores `a[0]` times to
/// threadgroup memory; `fill` stores `gid + 1` at `b[gid]`; `twice` stores `2 * a[gid]` at
/// `b[gid]`.
const KERNELS: &str = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\n\
target triple = \"air64_v28-apple-macosx26.0.0\"\n\n\
@tg = internal addrspace(3) global [64 x i32] undef, align 4\n\n\
define void @empty(ptr addrspace(1) noundef %a, ptr addrspace(1) noundef %b, i32 noundef %gid) {\n\
entry:\n\
  ret void\n\
}\n\n\
define void @tg_only(ptr addrspace(1) noundef %a, ptr addrspace(1) noundef %b, i32 noundef %gid) {\n\
entry:\n\
  %i = and i32 %gid, 63\n\
  %iw = zext i32 %i to i64\n\
  %p = getelementptr inbounds [64 x i32], ptr addrspace(3) @tg, i64 0, i64 %iw\n\
  store i32 %gid, ptr addrspace(3) %p, align 4\n\
  ret void\n\
}\n\n\
define void @loop_tg(ptr addrspace(1) noundef %a, ptr addrspace(1) noundef %b, i32 noundef %gid) {\n\
entry:\n\
  %n = load i32, ptr addrspace(1) %a, align 4\n\
  %none = icmp eq i32 %n, 0\n\
  br i1 %none, label %done, label %body\n\
body:\n\
  %i = phi i32 [ 0, %entry ], [ %i.next, %body ]\n\
  %k = and i32 %i, 63\n\
  %kw = zext i32 %k to i64\n\
  %p = getelementptr inbounds [64 x i32], ptr addrspace(3) @tg, i64 0, i64 %kw\n\
  store i32 %i, ptr addrspace(3) %p, align 4\n\
  %i.next = add nuw i32 %i, 1\n\
  %more = icmp ult i32 %i.next, %n\n\
  br i1 %more, label %body, label %done\n\
done:\n\
  ret void\n\
}\n\n\
define void @fill(ptr addrspace(1) noundef %a, ptr addrspace(1) noundef %b, i32 noundef %gid) {\n\
entry:\n\
  %v = add i32 %gid, 1\n\
  %gw = zext i32 %gid to i64\n\
  %p = getelementptr inbounds i32, ptr addrspace(1) %b, i64 %gw\n\
  store i32 %v, ptr addrspace(1) %p, align 4\n\
  ret void\n\
}\n\n\
define void @twice(ptr addrspace(1) noundef %a, ptr addrspace(1) noundef %b, i32 noundef %gid) {\n\
entry:\n\
  %gw = zext i32 %gid to i64\n\
  %q = getelementptr inbounds i32, ptr addrspace(1) %a, i64 %gw\n\
  %x = load i32, ptr addrspace(1) %q, align 4\n\
  %v = shl i32 %x, 1\n\
  %p = getelementptr inbounds i32, ptr addrspace(1) %b, i64 %gw\n\
  store i32 %v, ptr addrspace(1) %p, align 4\n\
  ret void\n\
}\n\n\
!llvm.module.flags = !{!0, !1}\n!air.kernel = !{!20, !21, !22, !23, !24}\n!air.version = !{!8}\n!air.language_version = !{!9}\n\
!0 = !{i32 7, !\"air.max_device_buffers\", i32 31}\n\
!1 = !{i32 7, !\"air.max_threadgroup_buffers\", i32 31}\n\
!3 = !{}\n!4 = !{!5, !6, !7}\n\
!5 = !{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"a\"}\n\
!6 = !{i32 1, !\"air.buffer\", !\"air.location_index\", i32 1, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"b\"}\n\
!7 = !{i32 2, !\"air.thread_position_in_grid\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"gid\"}\n\
!8 = !{i32 2, i32 8, i32 0}\n!9 = !{!\"Metal\", i32 4, i32 0, i32 0}\n\
!20 = !{ptr @empty, !3, !4}\n!21 = !{ptr @tg_only, !3, !4}\n!22 = !{ptr @loop_tg, !3, !4}\n\
!23 = !{ptr @fill, !3, !4}\n!24 = !{ptr @twice, !3, !4}\n";

/// The words the tests write and read.
const N: u32 = 1024;

/// The state of kernel `name` of [`KERNELS`].
fn state(device: &Device, name: &str) -> ComputePipelineState {
    let air = metal2vulkan_devtools::llvm_assemble_text(KERNELS).unwrap();
    let library = metal2vulkan::library::Library::from_air(&air).unwrap();
    let function = device
        .new_library_with_compile_library(library)
        .new_function_with_name(name)
        .unwrap();
    let descriptor =
        ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    device
        .new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// A shared buffer of `N` zero words whose hazards the runtime does not track.
fn untracked(device: &Device) -> Buffer {
    let options =
        ResourceOptions::shared().with_hazard_tracking_mode(HazardTrackingMode::Untracked);
    device
        .new_buffer_with_bytes(&vec![0; 4 * N as usize], options)
        .unwrap()
}

/// `2 * (i + 1)` for every `i < N`: what `twice` stores from what `fill` stored.
fn doubled_fill() -> Vec<u32> {
    (0..N).map(|i| 2 * (i + 1)).collect()
}

/// The device's count of skipped dispatches.
fn skipped(device: &Device) -> u64 {
    device.work_counts().dispatches_skipped_inert
}

/// The inert kernels' dispatches, direct in threadgroups of one thread, as threads and indirect,
/// are counted skipped; the device store's and the loop's are not, and the store writes its words.
#[test]
fn an_inert_kernel_records_no_dispatch_and_a_writing_or_looping_one_does() {
    let scratch = Scratch::new("inert-dispatch");
    let device = device(&scratch.0);
    let (empty, tg_only, loop_tg, fill) = (
        state(&device, "empty"),
        state(&device, "tg_only"),
        state(&device, "loop_tg"),
        state(&device, "fill"),
    );
    let count = words_buffer(&device, &[5]);
    let out = words_buffer(&device, &[0; N as usize]);
    let arguments = words_buffer(&device, &[4, 1, 1]);
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_buffer(Some(&count), 0, 0).unwrap();
    encoder.set_buffer(Some(&out), 0, 1).unwrap();
    let one = Size::new(1, 1, 1);
    let before = skipped(&device);
    encoder.set_compute_pipeline_state(&empty).unwrap();
    encoder
        .dispatch_threadgroups(Size::new(4096, 1, 1), one)
        .unwrap();
    encoder
        .dispatch_threads(Size::new(1000, 1, 1), Size::new(64, 1, 1))
        .unwrap();
    encoder
        .dispatch_threadgroups_indirect(&arguments, 0, one)
        .unwrap();
    assert_eq!(
        skipped(&device) - before,
        3,
        "the empty kernel's dispatches"
    );
    encoder.set_compute_pipeline_state(&tg_only).unwrap();
    encoder
        .dispatch_threadgroups(Size::new(4, 1, 1), Size::new(64, 1, 1))
        .unwrap();
    assert_eq!(skipped(&device) - before, 4, "the threadgroup store's");
    encoder.set_compute_pipeline_state(&loop_tg).unwrap();
    encoder
        .dispatch_threadgroups(one, Size::new(64, 1, 1))
        .unwrap();
    encoder.set_compute_pipeline_state(&fill).unwrap();
    encoder
        .dispatch_threads(Size::new(u64::from(N), 1, 1), Size::new(64, 1, 1))
        .unwrap();
    assert_eq!(skipped(&device) - before, 4, "the loop's and the store's");
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    assert_eq!(words(&out), (1..=N).collect::<Vec<_>>());
}

/// An encoder whose only dispatch is skipped waits for the first encoder's fence and updates its
/// own, which the third waits for: the third reads what the first wrote, over untracked buffers.
#[test]
fn fences_around_a_skipped_dispatch_order_a_later_read_of_an_earlier_write() {
    let scratch = Scratch::new("inert-fences");
    let device = device(&scratch.0);
    let (empty, fill, twice) = (
        state(&device, "empty"),
        state(&device, "fill"),
        state(&device, "twice"),
    );
    let (x, y) = (untracked(&device), untracked(&device));
    let (written, passed) = (device.new_fence(), device.new_fence());
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let grid = Size::new(u64::from(N), 1, 1);
    let group = Size::new(64, 1, 1);
    let before = skipped(&device);
    let mut first = command_buffer.compute_command_encoder().unwrap();
    first.set_compute_pipeline_state(&fill).unwrap();
    first.set_buffer(Some(&x), 0, 1).unwrap();
    first.dispatch_threads(grid, group).unwrap();
    first.update_fence(&written).unwrap();
    first.end_encoding().unwrap();
    let mut skipping = command_buffer.compute_command_encoder().unwrap();
    skipping.wait_for_fence(&written).unwrap();
    skipping.set_compute_pipeline_state(&empty).unwrap();
    skipping
        .dispatch_threadgroups(Size::new(u64::from(N), 1, 1), Size::new(1, 1, 1))
        .unwrap();
    skipping.update_fence(&passed).unwrap();
    skipping.end_encoding().unwrap();
    let mut last = command_buffer.compute_command_encoder().unwrap();
    last.wait_for_fence(&passed).unwrap();
    last.set_compute_pipeline_state(&twice).unwrap();
    last.set_buffer(Some(&x), 0, 0).unwrap();
    last.set_buffer(Some(&y), 0, 1).unwrap();
    last.dispatch_threads(grid, group).unwrap();
    last.end_encoding().unwrap();
    assert_eq!(skipped(&device) - before, 1);
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    assert_eq!(words(&y), doubled_fill());
}

/// A command buffer whose only dispatch is skipped waits for a shared event's 1 and signals 2, on a
/// queue of its own; a third, waiting for 2, reads what the first wrote before it signalled 1.
/// Committed last first, so only the event orders them.
#[test]
fn events_around_a_skipped_dispatch_order_a_later_read_of_an_earlier_write() {
    let scratch = Scratch::new("inert-events");
    let device = plan_device(&scratch.0);
    let (empty, fill, twice) = (
        state(&device, "empty"),
        state(&device, "fill"),
        state(&device, "twice"),
    );
    let (x, y) = (untracked(&device), untracked(&device));
    let event = device.new_shared_event().unwrap();
    let queues: Vec<_> = (0..3)
        .map(|_| device.new_command_queue().unwrap())
        .collect();
    let grid = Size::new(u64::from(N), 1, 1);
    let group = Size::new(64, 1, 1);
    let before = skipped(&device);
    let writer = queues[0].command_buffer().unwrap();
    let mut encoder = writer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(&fill).unwrap();
    encoder.set_buffer(Some(&x), 0, 1).unwrap();
    encoder.dispatch_threads(grid, group).unwrap();
    encoder.end_encoding().unwrap();
    writer
        .encode_signal_event(EventRef::Shared(&event), 1)
        .unwrap();
    let skipping = queues[1].command_buffer().unwrap();
    skipping
        .encode_wait_for_event(EventRef::Shared(&event), 1)
        .unwrap();
    let mut encoder = skipping.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(&empty).unwrap();
    encoder
        .dispatch_threadgroups(Size::new(u64::from(N), 1, 1), Size::new(1, 1, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    skipping
        .encode_signal_event(EventRef::Shared(&event), 2)
        .unwrap();
    let reader = queues[2].command_buffer().unwrap();
    reader
        .encode_wait_for_event(EventRef::Shared(&event), 2)
        .unwrap();
    let mut encoder = reader.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(&twice).unwrap();
    encoder.set_buffer(Some(&x), 0, 0).unwrap();
    encoder.set_buffer(Some(&y), 0, 1).unwrap();
    encoder.dispatch_threads(grid, group).unwrap();
    encoder.end_encoding().unwrap();
    assert_eq!(skipped(&device) - before, 1);
    for command_buffer in [&reader, &skipping, &writer] {
        command_buffer.commit().unwrap();
    }
    for command_buffer in [&writer, &skipping, &reader] {
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    }
    assert_eq!(words(&y), doubled_fill());
}

/// The bytes of the uses tests' no-copy buffers: past the fill queue's least
/// (`device::staging::FILL_QUEUE_LEAST`), so a device with a fill queue copies the `Managed` one in
/// there.
const PAGES: u64 = 2 << 20;

/// Anonymous private pages of the test's own, holding `byte` in each, unmapped when dropped:
/// declared before the device, so dropped after every buffer over them.
struct Pages(NonNull<u8>);

impl Pages {
    fn new(byte: u8) -> Pages {
        // SAFETY: a fresh private anonymous mapping, unmapped in `drop`.
        let at = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                PAGES as usize,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANON,
                -1,
                0,
            )
        };
        assert_ne!(at, libc::MAP_FAILED);
        let pages = Pages(NonNull::new(at.cast()).unwrap());
        pages.write(byte);
        pages
    }

    /// Write `byte` over every byte, as the CPU, while no command buffer uses them.
    fn write(&self, byte: u8) {
        // SAFETY: the mapping's bytes, which no command buffer uses meanwhile.
        unsafe { std::ptr::write_bytes(self.0.as_ptr(), byte, PAGES as usize) };
    }

    /// A no-copy buffer of `storage` over them whole.
    fn buffer(&self, device: &Device, storage: StorageMode) -> Buffer {
        let options = ResourceOptions::new(
            storage,
            CpuCacheMode::DefaultCache,
            HazardTrackingMode::Default,
        );
        // SAFETY: the pages outlive every buffer over them (declared before the device).
        unsafe {
            device.new_buffer_with_bytes_no_copy(HostMemory::new(self.0), PAGES, options, None)
        }
        .unwrap()
    }
}

impl Drop for Pages {
    fn drop(&mut self) {
        // SAFETY: the mapping `new` made, which no buffer lies in any more.
        unsafe { libc::munmap(self.0.as_ptr().cast(), PAGES as usize) };
    }
}

/// A command buffer of `queue` whose one encoder dispatches `state` over `a` (index 0) and `b`
/// (index 1), `grid` threads in threadgroups of `group`, committed and waited for: its status and
/// error.
fn dispatched(
    queue: &crate::CommandQueue,
    state: &ComputePipelineState,
    (a, b): (&Buffer, &Buffer),
    (grid, group): (u64, u64),
) -> (CommandBufferStatus, Option<CommandBufferError>) {
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(state).unwrap();
    encoder.set_buffer(Some(a), 0, 0).unwrap();
    encoder.set_buffer(Some(b), 0, 1).unwrap();
    encoder
        .dispatch_threads(Size::new(grid, 1, 1), Size::new(group, 1, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    (command_buffer.status(), command_buffer.error())
}

/// A `Managed` no-copy buffer the CPU wrote and named whole with `did_modify_range`, bound by a
/// skipped dispatch in one command buffer and read by `twice` in the next: on a device without
/// unified memory, the first command buffer's commit takes the range
/// (`WorkCounts::managed_bytes_copied_in` counts its bytes there) and the second copies nothing
/// in and waits for no fill-queue submission. `twice` reads the CPU's words on every device.
#[test]
fn a_skipped_dispatch_s_command_buffer_copies_in_the_managed_buffers_it_binds() {
    let pages = Pages::new(0x11);
    let scratch = Scratch::new("inert-copies-in");
    let device = device(&scratch.0);
    let (empty, twice) = (state(&device, "empty"), state(&device, "twice"));
    let managed = pages.buffer(&device, StorageMode::Managed);
    let out = words_buffer(&device, &[0; N as usize]);
    let queue = device.new_command_queue().unwrap();
    let two_copies = !device.properties().has_unified_memory;
    // The buffer's making copies its pages in with the next submission: this one.
    let read = dispatched(&queue, &twice, (&managed, &out), (u64::from(N), 64));
    assert_eq!(read, (CommandBufferStatus::Completed, None));
    assert_eq!(words(&out), vec![0x2222_2222; N as usize]);
    pages.write(0x21);
    managed.did_modify_range(0..PAGES).unwrap();
    let (before, skipped_before) = (device.work_counts(), skipped(&device));
    let bound = dispatched(&queue, &empty, (&managed, &out), (1, 1));
    assert_eq!(bound, (CommandBufferStatus::Completed, None));
    let between = device.work_counts();
    assert_eq!(
        skipped(&device) - skipped_before,
        1,
        "the empty kernel's dispatch"
    );
    let read = dispatched(&queue, &twice, (&managed, &out), (u64::from(N), 64));
    assert_eq!(read, (CommandBufferStatus::Completed, None));
    let after = device.work_counts();
    assert_eq!(words(&out), vec![0x4242_4242; N as usize]);
    let copied = |from: &crate::WorkCounts, to: &crate::WorkCounts| {
        to.managed_bytes_copied_in - from.managed_bytes_copied_in
    };
    let waits = |from: &crate::WorkCounts, to: &crate::WorkCounts| {
        to.fill_queue_waits - from.fill_queue_waits
    };
    eprintln!(
        "INERT-COPIES-IN {}: two copies {two_copies}; copied in by the skipped dispatch's command \
         buffer {}, by the reader's {}; fill-queue submissions {} and {}, waits {} and {}",
        device.name(),
        copied(&before, &between),
        copied(&between, &after),
        between.fill_queue_submissions - before.fill_queue_submissions,
        after.fill_queue_submissions - between.fill_queue_submissions,
        waits(&before, &between),
        waits(&between, &after),
    );
    if two_copies {
        assert_eq!(
            (copied(&before, &between), copied(&between, &after)),
            (PAGES, 0),
            "the bytes copied in by the skipped dispatch's command buffer and by the reader's"
        );
        assert_eq!(waits(&between, &after), 0, "the reader's fill-queue waits");
    }
}

/// A `Shared` no-copy buffer detached before a skipped dispatch binds it: the dispatch's command
/// buffer fails as one whose recorded command names the buffer does (`DetachedBacking`), as
/// Metal's command buffer uses what its commands bind.
#[test]
fn a_skipped_dispatch_over_a_detached_buffer_fails_its_command_buffer() {
    let pages = Pages::new(0x11);
    let scratch = Scratch::new("inert-detached");
    let device = device(&scratch.0);
    let empty = state(&device, "empty");
    let shared = pages.buffer(&device, StorageMode::Shared);
    let out = words_buffer(&device, &[0; N as usize]);
    let queue = device.new_command_queue().unwrap();
    assert_eq!(shared.detach_backing(), Ok(true));
    let outcome = dispatched(&queue, &empty, (&shared, &out), (1, 1));
    assert_eq!(
        outcome,
        (
            CommandBufferStatus::Error,
            Some(CommandBufferError::DetachedBacking)
        )
    );
}
