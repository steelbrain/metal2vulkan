//! Device tests of the runtime's buffers over host memory, from bytes, and their labels and
//! purgeable state on the device the cfg selects, under the validation layer. Each test fails on
//! any validation message, so a leaked import or buffer fails it too.
//!
//! The host memory is the consumer's (reims-vgpu): guest RAM is one shared memory
//! object, and the buffers wrap a second, 16-KiB-aligned mapping of some of its pages
//! (`gpu_task/memory.rs`'s `no_copy_buffer`, with a deallocator that unmaps it; `display/efi.rs`'s
//! framebuffer, with none), with resource options 0. The CPU writes through the guest RAM
//! mapping, the GPU through the buffer, and each reads what the other wrote. The answers expected
//! are Metal's, as measured on an Apple M3 Max and an M4.

#[cfg(any(m2v_cpu_icd, m2v_gpu_hazards))]
use super::layer_reported_nothing;
use super::{
    at_plan, buffer_with, encode_case, pipeline_state, public_case, read_buffer,
    validated_device_with, CaseResources, Messages, ScratchCache,
};
use crate::candidate::engine::ComputeOutput;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::error::Feature;
use metal2vulkan::metal::{PixelFormat, TextureUsage};
#[cfg(any(m2v_cpu_icd, m2v_gpu_hazards))]
use metal2vulkan::vk::DriverId;
use metal2vulkan_runtime::DeviceOptions;
use metal2vulkan_runtime::{
    BlitOption, Buffer, BufferContents, BufferDescriptor, CommandBuffer, CommandBufferStatus,
    CommandQueue, CopyProblem, CpuCacheMode, Deallocator, DebugMessage, Device, Error, EventRef,
    HazardTrackingMode, HostBacking, HostMemory, Invalid, Nil, Origin, Problem, PurgeableState,
    Region, ResourceOptions, Size, StorageMode, TextureDescriptor, WorkCounts,
};
#[cfg(any(m2v_cpu_icd, m2v_gpu_hazards))]
use metal2vulkan_runtime::{CommandBufferError, LayerSetting, LayerSettingValue};
use std::collections::BTreeMap;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// The longest a command buffer of these tests may take.
const WAIT: Duration = Duration::from_secs(60);

/// The consumer's guest page, and the alignment of its mappings.
pub(super) const GUEST_PAGE: usize = 16 << 10;

fn invalid(call: &'static str, problem: Problem) -> Error {
    Error::Invalid(Invalid::new(call, problem))
}

/// Whether `error` is the typed refusal of host memory this device cannot import exactly.
pub(super) fn not_importable(error: &Error) -> bool {
    matches!(error, Error::Unsupported(u) if u.feature == Feature::HostMemory)
}

fn options(
    storage: StorageMode,
    cache: CpuCacheMode,
    hazard: HazardTrackingMode,
) -> ResourceOptions {
    ResourceOptions::new(storage, cache, hazard)
}

/// The host's page size.
pub(super) fn host_page() -> usize {
    // SAFETY: `sysconf` reads a system constant.
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    usize::try_from(page).expect("a page size")
}

/// One shared memory object of `len` bytes: the consumer's guest RAM.
pub(super) struct GuestRam {
    fd: libc::c_int,
    len: usize,
    /// The guest RAM's own mapping, through which the CPU writes and reads.
    view: Mapping,
}

impl GuestRam {
    pub(super) fn new(len: usize) -> GuestRam {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let name = std::ffi::CString::new(format!(
            "/m2v-rt4-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ))
        .unwrap();
        // SAFETY: a new shared memory object, unlinked at once: only the descriptor names it.
        let fd = unsafe {
            let fd = libc::shm_open(
                name.as_ptr(),
                libc::O_RDWR | libc::O_CREAT | libc::O_EXCL,
                0o600 as libc::c_uint,
            );
            assert!(fd >= 0, "shm_open: {}", std::io::Error::last_os_error());
            libc::shm_unlink(name.as_ptr());
            assert_eq!(libc::ftruncate(fd, len as libc::off_t), 0, "ftruncate");
            fd
        };
        let view = Mapping::of(fd, 0, len, GUEST_PAGE, libc::PROT_READ | libc::PROT_WRITE);
        GuestRam { fd, len, view }
    }

    /// A second mapping of `len` bytes from `offset`, aligned to the guest page, as the consumer
    /// maps a task's pages.
    pub(super) fn map(&self, offset: usize, len: usize, protection: libc::c_int) -> Mapping {
        assert!(offset + len <= self.len);
        Mapping::of(self.fd, offset, len, GUEST_PAGE, protection)
    }

    /// A copy of the guest RAM's bytes `range`.
    pub(super) fn read(&self, range: std::ops::Range<usize>) -> Vec<u8> {
        self.view.read(range)
    }

    /// Write `bytes` into the guest RAM at `offset`.
    pub(super) fn write(&self, offset: usize, bytes: &[u8]) {
        self.view.write(offset, bytes);
    }
}

impl Drop for GuestRam {
    fn drop(&mut self) {
        // SAFETY: the descriptor is this object's; its mappings keep the memory alive.
        unsafe { libc::close(self.fd) };
    }
}

/// A mapping at an address aligned to `align`, unmapped when dropped.
pub(super) struct Mapping {
    /// The reservation the mapping sits in, unmapped whole.
    reservation: *mut libc::c_void,
    reserved: usize,
    at: NonNull<u8>,
    len: usize,
}

// SAFETY: the mapping is plain memory of this process; who writes it when is each test's to order.
unsafe impl Send for Mapping {}

impl Mapping {
    /// `len` bytes of `fd` from `offset`, shared, at an address aligned to `align`.
    fn of(
        fd: libc::c_int,
        offset: usize,
        len: usize,
        align: usize,
        protection: libc::c_int,
    ) -> Mapping {
        let reserved = len + align;
        // SAFETY: a fresh reservation, then a fixed mapping inside it.
        unsafe {
            let reservation = libc::mmap(
                std::ptr::null_mut(),
                reserved,
                libc::PROT_NONE,
                libc::MAP_PRIVATE | libc::MAP_ANON,
                -1,
                0,
            );
            assert_ne!(reservation, libc::MAP_FAILED, "mmap reservation");
            let start = (reservation as usize).next_multiple_of(align);
            let at = libc::mmap(
                start as *mut libc::c_void,
                len,
                protection,
                libc::MAP_SHARED | libc::MAP_FIXED,
                fd,
                offset as libc::off_t,
            );
            assert_eq!(at as usize, start, "mmap of the object");
            Mapping {
                reservation,
                reserved,
                at: NonNull::new(at.cast()).unwrap(),
                len,
            }
        }
    }

    /// `len` fresh private anonymous bytes (what allocators return) at a page-aligned address.
    pub(super) fn anonymous(len: usize) -> Mapping {
        Mapping::anonymous_with(len, libc::MAP_PRIVATE)
    }

    /// `len` fresh shared anonymous bytes (`MAP_SHARED | MAP_ANON`) at a page-aligned address.
    fn shared_anonymous(len: usize) -> Mapping {
        Mapping::anonymous_with(len, libc::MAP_SHARED)
    }

    /// `len` fresh anonymous bytes, mapped with `sharing` (`MAP_PRIVATE` or `MAP_SHARED`).
    fn anonymous_with(len: usize, sharing: libc::c_int) -> Mapping {
        // SAFETY: a fresh mapping.
        let at = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                sharing | libc::MAP_ANON,
                -1,
                0,
            )
        };
        assert_ne!(at, libc::MAP_FAILED, "mmap");
        Mapping {
            reservation: at,
            reserved: len,
            at: NonNull::new(at.cast()).unwrap(),
            len,
        }
    }

    pub(super) fn pointer(&self) -> NonNull<u8> {
        self.at
    }

    /// A copy of the mapping's bytes `range`.
    pub(super) fn read(&self, range: std::ops::Range<usize>) -> Vec<u8> {
        assert!(range.start <= range.end && range.end <= self.len);
        // SAFETY: the range lies in the mapping; each test orders the CPU's and the GPU's
        // accesses.
        unsafe {
            std::slice::from_raw_parts(self.at.as_ptr().add(range.start), range.len()).to_vec()
        }
    }

    /// Write `bytes` into the mapping at `offset`.
    pub(super) fn write(&self, offset: usize, bytes: &[u8]) {
        assert!(offset + bytes.len() <= self.len);
        // SAFETY: as in `read`.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.at.as_ptr().add(offset), bytes.len())
        };
    }

    /// Set every byte of the mapping to `value`.
    pub(super) fn fill(&self, value: u8) {
        self.write(0, &vec![value; self.len]);
    }

    /// Map `len` bytes of `fd` from `offset` in place of the mapping's pages, at the same address
    /// (`MAP_FIXED`): the consumer's unmap and remap of a guest page at one host address.
    fn remap(&self, fd: libc::c_int, offset: usize) {
        // SAFETY: the mapping's own range.
        unsafe { map_in_place(self.at, self.len, fd, offset) };
    }

    /// Make the mapping read-only.
    pub(super) fn protect_read_only(&self) {
        // SAFETY: the mapping is this object's.
        let done = unsafe { libc::mprotect(self.at.as_ptr().cast(), self.len, libc::PROT_READ) };
        assert_eq!(done, 0, "mprotect");
    }
}

/// Map `len` bytes of `fd` from `offset` at `at`, in place of the pages there (`MAP_FIXED`).
///
/// # Safety
///
/// `at..at + len` is a mapping of this process that nothing else uses as the pages there.
unsafe fn map_in_place(at: NonNull<u8>, len: usize, fd: libc::c_int, offset: usize) {
    // SAFETY: the caller's contract; the range is replaced by a shared mapping of the object.
    let mapped = unsafe {
        libc::mmap(
            at.as_ptr().cast(),
            len,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_SHARED | libc::MAP_FIXED,
            fd,
            offset as libc::off_t,
        )
    };
    assert_eq!(
        mapped,
        at.as_ptr().cast(),
        "mmap in place: {}",
        std::io::Error::last_os_error()
    );
}

/// Map `len` fresh private anonymous bytes at `at`, in place of the pages there (`MAP_FIXED`).
///
/// # Safety
///
/// As [`map_in_place`].
unsafe fn anonymous_in_place(at: NonNull<u8>, len: usize) {
    // SAFETY: the caller's contract; the range is replaced by fresh private pages.
    let mapped = unsafe {
        libc::mmap(
            at.as_ptr().cast(),
            len,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANON | libc::MAP_FIXED,
            -1,
            0,
        )
    };
    assert_eq!(
        mapped,
        at.as_ptr().cast(),
        "mmap in place: {}",
        std::io::Error::last_os_error()
    );
}

impl Drop for Mapping {
    fn drop(&mut self) {
        // SAFETY: the reservation is this object's, and no buffer uses it any more (the tests
        // drop each mapping only after its buffers, or from a buffer's deallocator).
        unsafe { libc::munmap(self.reservation, self.reserved) };
    }
}

/// What a deallocator saw: how often it ran, with which arguments, on which thread, and whether
/// it was dropped without running.
#[derive(Default)]
pub(super) struct Seen {
    pub(super) runs: AtomicUsize,
    pub(super) dropped: AtomicUsize,
    pub(super) args: Mutex<Option<(usize, u64)>>,
    pub(super) thread: Mutex<Option<std::thread::ThreadId>>,
    /// The first 16 bytes of the memory when it ran.
    pub(super) bytes: Mutex<Vec<u8>>,
}

/// Counts its drop into `seen` (a deallocator dropped unrun).
struct DropCount(Arc<Seen>);

impl Drop for DropCount {
    fn drop(&mut self) {
        self.0.dropped.fetch_add(1, Ordering::SeqCst);
    }
}

/// A deallocator that records into `seen` and then drops `keep` (a mapping it releases).
pub(super) fn recording<T: Send + 'static>(seen: &Arc<Seen>, keep: T) -> Deallocator {
    let seen = Arc::clone(seen);
    let count = DropCount(Arc::clone(&seen));
    Box::new(move |pointer: NonNull<u8>, length: u64| {
        // SAFETY: the deallocator runs before the memory is released (by `keep`, below).
        let first =
            unsafe { std::slice::from_raw_parts(pointer.as_ptr(), 16.min(length as usize)) };
        *seen.bytes.lock().unwrap() = first.to_vec();
        *seen.args.lock().unwrap() = Some((pointer.as_ptr() as usize, length));
        *seen.thread.lock().unwrap() = Some(std::thread::current().id());
        seen.runs.fetch_add(1, Ordering::SeqCst);
        drop(keep);
        std::mem::forget(count);
    })
}

/// A no-copy buffer over `len` bytes at `pointer`.
///
/// # Safety
///
/// [`HostMemory::new`]'s contract.
pub(super) unsafe fn no_copy(
    device: &Device,
    pointer: NonNull<u8>,
    len: usize,
    options: ResourceOptions,
    deallocator: Option<Deallocator>,
) -> Result<Buffer, Error> {
    // SAFETY: the caller's contract.
    unsafe {
        device.new_buffer_with_bytes_no_copy(
            HostMemory::new(pointer),
            len as u64,
            options,
            deallocator,
        )
    }
}

/// Commit `command_buffer`, wait for it, and check it completed without error.
pub(super) fn run(command_buffer: &CommandBuffer) {
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
}

/// `buffer`'s bytes `range`, read back by the GPU into a new Shared buffer.
pub(super) fn gpu_read(
    queue: &CommandQueue,
    buffer: &Buffer,
    range: std::ops::Range<u64>,
) -> Vec<u8> {
    let device = queue.device();
    let length = range.end - range.start;
    let copy = device
        .new_buffer(length, ResourceOptions::shared())
        .unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_buffer_to_buffer(buffer, range.start, &copy, 0, length)
        .unwrap();
    blit.end_encoding().unwrap();
    run(&command_buffer);
    read_buffer(&copy, 0, length).unwrap()
}

/// Fill `range` of `buffer` with `value` on the GPU.
pub(super) fn gpu_fill(
    queue: &CommandQueue,
    buffer: &Buffer,
    range: std::ops::Range<u64>,
    value: u8,
) {
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.fill_buffer(buffer, range, value).unwrap();
    blit.end_encoding().unwrap();
    run(&command_buffer);
}

/// Bytes `i % 251` from `start`.
pub(super) fn counting(len: usize, start: usize) -> Vec<u8> {
    (start..start + len).map(|i| (i % 251) as u8).collect()
}

/// Run `test` on a validated device, then check that once everything is dropped (the device
/// last) the layer reported nothing.
pub(super) fn on_device(test: impl FnOnce(&Device)) {
    on_device_with(|_| {}, test)
}

/// [`on_device`], `adjust` applied to the options of each device it makes (`at_plan` gives them
/// their queue plan).
pub(super) fn on_device_with(adjust: fn(&mut DeviceOptions), test: impl FnOnce(&Device)) {
    let messages = on_device_reporting_with(adjust, test);
    assert!(
        messages.is_empty(),
        "the validation layer reported: {messages:#?}"
    );
}

/// Run `test` on a validated device; the failing messages reported once everything is dropped
/// (the device last).
fn on_device_reporting(test: impl FnOnce(&Device)) -> Vec<DebugMessage> {
    on_device_reporting_with(|_| {}, test)
}

/// [`on_device_reporting`], `adjust` applied to the options of each device it makes (`at_plan` gives them
/// their queue plan).
fn on_device_reporting_with(
    adjust: fn(&mut DeviceOptions),
    test: impl FnOnce(&Device),
) -> Vec<DebugMessage> {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), adjust).unwrap();
        test(&device);
    }
    messages.take()
}

/// The consumer's task memory (`gpu_task/memory.rs`, `no_copy_buffer`): a 16-KiB-aligned mapping
/// of two guest pages, options 0, and a deallocator that releases the mapping. A public kernel
/// case runs over two such buffers (its input written by the CPU through guest RAM, its output by
/// the kernel through the buffer's device address) and matches the case's committed Metal output
/// read through guest RAM; blits read the CPU's bytes and write the GPU's. The deallocator runs
/// once, at the buffers' last release, with the pointer and length, and only then is the mapping
/// unmapped. Guest RAM is shared memory: a driver that does not import shared memory refuses the
/// buffers typed (as `host_memory_kinds_are_imported_or_refused_typed` shows), and each refused
/// call drops its deallocator unrun.
#[test]
fn the_consumer_s_task_memory_takes_gpu_and_cpu_writes() {
    on_device(|device| {
        let case =
            public_case("saturating-halving-and-wide-product-integers-at-their-edges").unwrap();
        let pipeline = pipeline_state(device, &case).unwrap();
        let queue = device.new_command_queue().unwrap();
        let ram = GuestRam::new(8 * GUEST_PAGE);
        let seen: BTreeMap<u32, Arc<Seen>> = case
            .compute
            .buffers
            .keys()
            .map(|&index| (index, Arc::default()))
            .collect();
        let mut pointers = BTreeMap::new();
        let made: Result<BTreeMap<u32, Buffer>, Error> = case
            .compute
            .buffers
            .iter()
            .enumerate()
            .map(|(slot, (&index, bytes))| {
                // Two guest pages per buffer, from guest page 1 + 2 * slot.
                let offset = (1 + 2 * slot) * GUEST_PAGE;
                ram.write(offset, bytes);
                let mapping = ram.map(offset, 2 * GUEST_PAGE, libc::PROT_READ | libc::PROT_WRITE);
                let pointer = mapping.pointer();
                pointers.insert(index, (pointer, offset));
                let deallocator = recording(&seen[&index], mapping);
                // SAFETY: the mapping stays mapped until the deallocator, which owns it, runs.
                let buffer = unsafe {
                    no_copy(
                        device,
                        pointer,
                        2 * GUEST_PAGE,
                        ResourceOptions::default(),
                        Some(deallocator),
                    )
                }?;
                assert_eq!(buffer.contents(), Some(pointer));
                assert_eq!(buffer.length(), 2 * GUEST_PAGE as u64);
                assert_eq!(buffer.gpu_address().0 % GUEST_PAGE as u64, 0);
                Ok((index, buffer))
            })
            .collect();
        let buffers = match made {
            Ok(buffers) => buffers,
            Err(error) => {
                assert!(not_importable(&error), "{error:?}");
                for seen in seen.values() {
                    assert_eq!(seen.runs.load(Ordering::SeqCst), 0);
                }
                eprintln!("runtime consumer task memory: refused (shared memory)");
                return;
            }
        };
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        let resources = CaseResources::new(device, &case).unwrap();
        encode_case(&mut encoder, &pipeline, &case.compute, &buffers, &resources).unwrap();
        encoder.end_encoding().unwrap();
        run(&command_buffer);
        let metal2vulkan_output = match case.compute.output {
            ComputeOutput::Buffer {
                binding,
                offset,
                length,
            } => {
                let start = pointers[&binding].1 + offset as usize;
                ram.read(start..start + length as usize)
            }
            ref other => panic!("an output {other:?}"),
        };
        assert!(
            metal2vulkan_output == case.metal_output,
            "the output differs from Metal's:\n runtime {:02x?}\n metal   {:02x?}",
            metal2vulkan_output,
            case.metal_output
        );

        // Blits: the GPU reads the CPU's bytes of the second page and writes a fill the CPU
        // reads through guest RAM.
        let (&index, buffer) = buffers.iter().next().unwrap();
        let offset = pointers[&index].1;
        let page = GUEST_PAGE as u64;
        let written = counting(GUEST_PAGE, 7);
        ram.write(offset + GUEST_PAGE, &written);
        assert_eq!(gpu_read(&queue, buffer, page..2 * page), written);
        gpu_fill(&queue, buffer, page + 3..page + 4099, 0xa5);
        let ram_bytes = ram.read(offset + GUEST_PAGE..offset + 2 * GUEST_PAGE);
        assert_eq!(&ram_bytes[..3], &written[..3]);
        assert!(ram_bytes[3..4099].iter().all(|&byte| byte == 0xa5));
        assert_eq!(&ram_bytes[4099..], &written[4099..]);

        // A clone keeps the buffer; the last release runs the deallocator once, here.
        let clone = buffer.clone();
        drop(buffers);
        let first = &seen[&index];
        assert_eq!(
            first.runs.load(Ordering::SeqCst),
            0,
            "a clone holds the buffer"
        );
        drop(clone);
        for (index, seen) in &seen {
            assert_eq!(seen.runs.load(Ordering::SeqCst), 1, "buffer {index}");
            assert_eq!(seen.dropped.load(Ordering::SeqCst), 0, "buffer {index}");
            let (pointer, offset) = pointers[index];
            assert_eq!(
                *seen.args.lock().unwrap(),
                Some((pointer.as_ptr() as usize, 2 * GUEST_PAGE as u64))
            );
            assert_eq!(
                *seen.thread.lock().unwrap(),
                Some(std::thread::current().id())
            );
            assert_eq!(*seen.bytes.lock().unwrap(), ram.read(offset..offset + 16));
        }
    });
}

/// Metal's answers for a buffer a pending command buffer holds (`dealloc_held`): with the program's
/// handle dropped, the deallocator has not run while the command buffer waits for an event; it runs
/// once, after the command buffer completed and its completed handlers ran, on a thread of the
/// runtime, with the pointer and the length, and sees the GPU's writes. Here it has run by the time
/// `wait_until_completed` returns, as on the M3 (the M4 raced it).
#[test]
fn a_waiting_command_buffer_holds_its_buffer_until_it_completed() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        // Private memory, which every venue imports (RADV refuses guest RAM's shared memory).
        let length = 2 * GUEST_PAGE;
        let mapping = Mapping::anonymous(length);
        mapping.write(0, &counting(length, 5));
        let pointer = mapping.pointer();
        let seen = Arc::new(Seen::default());
        // SAFETY: the deallocator releases the mapping, after the buffer's last use.
        let buffer = unsafe {
            no_copy(
                device,
                pointer,
                length,
                ResourceOptions::default(),
                Some(recording(&seen, mapping)),
            )
        }
        .unwrap();
        let event = device.new_shared_event().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        command_buffer
            .encode_wait_for_event(EventRef::from(&event), 1)
            .unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.fill_buffer(&buffer, 0..16, 0x6b).unwrap();
        blit.end_encoding().unwrap();
        // How often the deallocator had run when the completed handler ran.
        let in_handler = Arc::new(AtomicUsize::new(usize::MAX));
        {
            let (seen, in_handler) = (Arc::clone(&seen), Arc::clone(&in_handler));
            command_buffer
                .add_completed_handler(Box::new(move |_: &CommandBuffer| {
                    in_handler.store(seen.runs.load(Ordering::SeqCst), Ordering::SeqCst);
                }))
                .unwrap();
        }
        command_buffer.commit().unwrap();
        drop(buffer);
        std::thread::sleep(Duration::from_millis(100));
        assert_eq!(command_buffer.status(), CommandBufferStatus::Committed);
        assert_eq!(
            seen.runs.load(Ordering::SeqCst),
            0,
            "the deallocator ran while its command buffer waited"
        );
        event.set_signaled_value(1);
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
        assert_eq!(in_handler.load(Ordering::SeqCst), 0);
        assert_eq!(seen.runs.load(Ordering::SeqCst), 1);
        assert_eq!(seen.dropped.load(Ordering::SeqCst), 0);
        assert_eq!(
            *seen.args.lock().unwrap(),
            Some((pointer.as_ptr() as usize, length as u64))
        );
        assert_ne!(
            *seen.thread.lock().unwrap(),
            Some(std::thread::current().id())
        );
        assert_eq!(*seen.bytes.lock().unwrap(), [0x6b; 16]);
    });
}

/// A deallocator may use the command buffer that held its buffer: it reads the status, waits
/// for it (which returns at once, as Metal's does) and drops its handle on the runtime's thread,
/// and none of it blocks, nor does the program's wait.
#[test]
fn a_deallocator_may_touch_the_command_buffer_that_held_its_buffer() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let memory = Mapping::anonymous(GUEST_PAGE);
        let pointer = memory.pointer();
        let command_buffer = queue.command_buffer().unwrap();
        let (sender, receiver) = std::sync::mpsc::channel();
        let deallocator: Deallocator = {
            let command_buffer = command_buffer.clone();
            Box::new(move |_: NonNull<u8>, _: u64| {
                let status = command_buffer.status();
                // Metal's deallocator returns from its wait for completion here.
                let waited = (
                    command_buffer.wait_until_completed(),
                    command_buffer.wait_until_completed_for(WAIT),
                );
                drop(command_buffer);
                drop(memory);
                sender.send((status, waited)).unwrap();
            })
        };
        // SAFETY: the deallocator releases the mapping, after the buffer's last use.
        let buffer = unsafe {
            no_copy(
                device,
                pointer,
                GUEST_PAGE,
                ResourceOptions::default(),
                Some(deallocator),
            )
        }
        .unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.fill_buffer(&buffer, 0..64, 0x2d).unwrap();
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        drop(buffer);
        let (status, waited) = receiver
            .recv_timeout(WAIT)
            .expect("the deallocator ran and returned");
        assert_eq!(status, CommandBufferStatus::Completed);
        assert_eq!(waited, (Ok(()), Ok(())));
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
    });
}

/// The deallocator of a buffer an encoder named, with its command buffer's every reference to the
/// buffer let go only after it completed, still runs as that command buffer releases what it
/// retained, where its wait for the command buffer returns at once, as Metal's does. Here the
/// device's thread that submits holds on past the completion, deterministically: the command
/// buffer `filled` is submitted in one pass with `holding`, whose submission handler returns only
/// once `after`, committed behind `filled` on its queue, ran its completed handler, after
/// `filled`'s release. (`opener`, on `holding`'s queue, holds that thread until all three are
/// committed, so the three go in that one pass.) A blit encoder logs its calls while the device has
/// a buffer over host pages (`replay.rs`), and its log was let go there, with the buffer's last
/// handle, after the completion: the deallocator then ran on that thread, whose own hold on the
/// command buffer kept the deallocator's wait from ever returning. `after` needs no GPU work, so
/// where `filled` completed before the pass took it, its completion turn is that thread's own
/// (`Completion::turn_owed`), which the thread makes before it runs `holding`'s handler; made
/// after it, the handler waited for a turn only its return let go.
///
/// The two command queues run on one `VkQueue`, and, where the device has another, on two: a
/// device of one `VkQueue` per family shares it between them, and one of many gives them two
/// (the lane's devices keep each queue on the `VkQueue` the round robin gave it,
/// `QueueSharingPolicy::PerQueue`), so each arrangement is made here on every device that has
/// it ([`queue_where`]).
#[test]
fn a_deallocator_returns_whatever_thread_submits_past_its_completion() {
    on_device_with(at_plan, |device| {
        let first = device.new_command_queue().unwrap();
        let on = first.vk_queue_sharing().0;
        let shared = queue_where(device, on, true).expect("a queue on the first one's VkQueue");
        deallocator_past_its_completion(device, &first, &shared, "one VkQueue");
        match queue_where(device, on, false) {
            Some(apart) => deallocator_past_its_completion(device, &first, &apart, "two VkQueues"),
            None => eprintln!(
                "deallocator past its completion: {} has one VkQueue",
                device.name()
            ),
        }
    });
}

/// A new command queue of `device` on the `VkQueue` `vk_queue` (`shared`), or on another: the
/// first such of up to 64 made one after another, which the round robin spreads over the
/// device's `VkQueue`s (`CommandQueue::vk_queue_sharing`). `None` where none was: a device of one
/// `VkQueue`, for another.
fn queue_where(device: &Device, vk_queue: usize, shared: bool) -> Option<CommandQueue> {
    (0..64)
        .map(|_| device.new_command_queue().unwrap())
        .find(|queue| (queue.vk_queue_sharing().0 == vk_queue) == shared)
}

/// [`a_deallocator_returns_whatever_thread_submits_past_its_completion`] on `first` and `second`,
/// the queues of the `arrangement` named.
fn deallocator_past_its_completion(
    device: &Device,
    first: &CommandQueue,
    second: &CommandQueue,
    arrangement: &str,
) {
    let queues = (first.vk_queue_sharing().0, second.vk_queue_sharing().0);
    let before = device.work_counts();
    let opener = second.command_buffer().unwrap();
    let (opened, open) = std::sync::mpsc::channel::<()>();
    let (release, released) = std::sync::mpsc::channel::<()>();
    opener
        .add_submission_handler(Box::new(move |_: &CommandBuffer| {
            opened.send(()).unwrap();
            released.recv_timeout(WAIT).unwrap();
        }))
        .unwrap();
    opener.commit().unwrap();
    open.recv_timeout(WAIT)
        .expect("the opener's submission handler ran");
    let memory = Mapping::anonymous(GUEST_PAGE);
    let pointer = memory.pointer();
    let filled = first.command_buffer().unwrap();
    // The thread the deallocator began on, once it began.
    let began: Arc<Mutex<Option<String>>> = Arc::default();
    let (sender, receiver) = std::sync::mpsc::channel();
    let deallocator: Deallocator = {
        let (filled, began) = (filled.clone(), Arc::clone(&began));
        Box::new(move |_: NonNull<u8>, _: u64| {
            let thread = std::thread::current().name().map(str::to_owned);
            *began.lock().unwrap() = Some(thread.unwrap_or_else(|| "(unnamed)".to_owned()));
            let status = filled.status();
            let waited = (
                filled.wait_until_completed(),
                filled.wait_until_completed_for(WAIT),
            );
            drop(filled);
            drop(memory);
            sender.send((status, waited)).unwrap();
        })
    };
    // SAFETY: the deallocator releases the mapping, after the buffer's last use.
    let buffer = unsafe {
        no_copy(
            device,
            pointer,
            GUEST_PAGE,
            ResourceOptions::default(),
            Some(deallocator),
        )
    }
    .unwrap();
    let mut blit = filled.blit_command_encoder().unwrap();
    blit.fill_buffer(&buffer, 0..64, 0x2d).unwrap();
    blit.end_encoding().unwrap();
    filled.commit().unwrap();
    drop(buffer);
    let after = first.command_buffer().unwrap();
    let (ran, after_ran) = std::sync::mpsc::channel::<()>();
    after
        .add_completed_handler(Box::new(move |_: &CommandBuffer| {
            ran.send(()).unwrap();
        }))
        .unwrap();
    after.commit().unwrap();
    let holding = second.command_buffer().unwrap();
    // What the device did, for a failure's report.
    let report = {
        let arrangement = arrangement.to_owned();
        let (filled, later, began) = (filled.clone(), after.clone(), Arc::clone(&began));
        move |device: &Device| {
            format!(
                "on {arrangement} {queues:?} of {}: the filled command buffer is {:?}, the later \
                 one {:?}; the deallocator began on {:?}; {}",
                device.name(),
                filled.status(),
                later.status(),
                began.lock().unwrap(),
                submitted_since(&before, &device.work_counts())
            )
        }
    };
    let report_there = report.clone();
    holding
        .add_submission_handler(Box::new(move |taken: &CommandBuffer| {
            if after_ran.recv_timeout(WAIT).is_err() {
                panic!(
                    "the later command buffer's completed handler never ran {}",
                    report_there(taken.device())
                );
            }
        }))
        .unwrap();
    holding.commit().unwrap();
    release.send(()).unwrap();
    let Ok((status, waited)) = receiver.recv_timeout(WAIT) else {
        panic!("the deallocator never returned {}", report(device));
    };
    assert_eq!(status, CommandBufferStatus::Completed, "{arrangement}");
    assert_eq!(waited, (Ok(()), Ok(())), "{arrangement}");
    for command_buffer in [&opener, &filled, &after, &holding] {
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
    }
    eprintln!(
        "deallocator past its completion: on {arrangement} {queues:?} of {}",
        device.name()
    );
}

/// What the device's scheduler did from `before` to `after`: its submissions, by whom, and what
/// held its heads back, for a failure's report.
fn submitted_since(before: &WorkCounts, after: &WorkCounts) -> String {
    let since = |field: fn(&WorkCounts) -> u64| field(after) - field(before);
    format!(
        "since the opener's commit, {} kernel submissions ({} batches at commit, {} grouped, {} \
         groups kept) in {} passes; heads held for host steps {}, checks {}, earlier commits {}, \
         event values {}, at their gate {}",
        since(|counts| counts.queue_submissions),
        since(|counts| counts.batches_submitted_at_commit),
        since(|counts| counts.batches_grouped),
        since(|counts| counts.groups_kept_pending),
        since(|counts| counts.scheduler_passes),
        since(|counts| counts.heads_held_for_host_steps),
        since(|counts| counts.heads_held_for_checks),
        since(|counts| counts.heads_held_for_earlier_commits),
        since(|counts| counts.heads_held_for_event_values),
        since(|counts| counts.heads_waiting_at_their_gate),
    )
}

/// The consumer's EFI framebuffer (`display/efi.rs`): a 16-KiB-aligned mapping of the
/// framebuffer, a length of the rows times the stride rounded up to a guest page, options 0, and
/// no deallocator. The GPU reads the pixels the guest wrote; the buffer's release runs nothing,
/// and the mapping, released after it, stays the consumer's. Guest RAM is shared memory, which a
/// driver that does not import it refuses typed.
#[test]
fn the_consumer_s_framebuffer_takes_a_buffer_without_a_deallocator() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        // 1024 x 768 BGRA8 at a stride of 4096 bytes: 3 MiB, a whole number of guest pages.
        let (stride, height) = (4096usize, 768usize);
        let length = (height * stride).next_multiple_of(GUEST_PAGE);
        let ram = GuestRam::new(GUEST_PAGE + length);
        let pixels = counting(length, 11);
        ram.write(GUEST_PAGE, &pixels);
        let framebuffer = ram.map(GUEST_PAGE, length, libc::PROT_READ | libc::PROT_WRITE);
        // SAFETY: the mapping outlives the buffer (it is dropped after it, below).
        let buffer = unsafe {
            no_copy(
                device,
                framebuffer.pointer(),
                length,
                ResourceOptions::default(),
                None,
            )
        };
        let buffer = match buffer {
            Ok(buffer) => buffer,
            Err(error) => {
                assert!(not_importable(&error), "{error:?}");
                eprintln!("runtime consumer framebuffer: refused (shared memory)");
                return;
            }
        };
        assert_eq!(buffer.contents(), Some(framebuffer.pointer()));
        let last_row = (height as u64 - 1) * stride as u64;
        assert_eq!(
            gpu_read(&queue, &buffer, last_row..last_row + stride as u64),
            pixels[last_row as usize..last_row as usize + stride]
        );
        assert_eq!(gpu_read(&queue, &buffer, 0..64), pixels[..64]);
        drop(buffer);
        drop(framebuffer);
    });
}

/// A no-copy buffer reports what Metal's reports: its pointer as `contents`, its length, the
/// resolved hazard tracking mode in its options (0 reports Tracked, 0x200), and the storage and
/// cache modes it was made with; Managed, write-combined and untracked buffers are made too, and
/// every one answers `NonVolatile` to every purgeable state and keeps its bytes.
#[test]
fn no_copy_buffers_report_what_metal_reports() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let memory = Mapping::anonymous(4 * GUEST_PAGE);
        let at = |page: usize| {
            NonNull::new(memory.pointer().as_ptr().wrapping_add(page * GUEST_PAGE)).unwrap()
        };
        use CpuCacheMode::{DefaultCache, WriteCombined};
        use HazardTrackingMode::{Default as Resolved, Tracked, Untracked};
        use StorageMode::{Managed, Shared};
        for (page, made, reported) in [
            (0, options(Shared, DefaultCache, Resolved), 0x200),
            (1, options(Managed, DefaultCache, Resolved), 0x210),
            (2, options(Shared, WriteCombined, Resolved), 0x201),
            (3, options(Shared, DefaultCache, Untracked), 0x100),
        ] {
            memory.write(page * GUEST_PAGE, &[page as u8 + 1; 16]);
            // SAFETY: the mapping outlives the buffer.
            let buffer = unsafe { no_copy(device, at(page), GUEST_PAGE, made, None) }.unwrap();
            assert_eq!(buffer.contents(), Some(at(page)));
            assert_eq!(buffer.length(), GUEST_PAGE as u64);
            assert_eq!(buffer.resource_options().raw(), reported, "{made:?}");
            assert_eq!(buffer.storage_mode(), made.storage_mode().unwrap());
            assert_eq!(buffer.cpu_cache_mode(), made.cpu_cache_mode().unwrap());
            let tracked = if made.hazard_tracking_mode().unwrap() == Untracked {
                Untracked
            } else {
                Tracked
            };
            assert_eq!(buffer.hazard_tracking_mode(), tracked);
            for state in [
                PurgeableState::KeepCurrent,
                PurgeableState::Volatile,
                PurgeableState::Empty,
                PurgeableState::KeepCurrent,
                PurgeableState::NonVolatile,
            ] {
                assert_eq!(
                    buffer.set_purgeable_state(state),
                    Ok(PurgeableState::NonVolatile)
                );
            }
            assert_eq!(gpu_read(&queue, &buffer, 0..16), [page as u8 + 1; 16]);
            buffer.set_label("a guest buffer");
            assert_eq!(buffer.label().as_deref(), Some("a guest buffer"));
        }
    });
}

/// Ranges Metal takes that do not start or end at a page (Metal makes every one and covers the
/// whole pages they touch). The runtime imports those pages and binds the buffer below its bytes,
/// at the buffer's memory alignment (and 16 bytes), so every one is made on every device; the
/// buffer's bytes start past its `VkBuffer`'s where the pointer misses that alignment
/// (`host_bytes.rs` uses such buffers every way). A buffer keeps its pointer as `contents` and the
/// pointer's offset in the page in `gpu_address`, and the GPU reads and writes its bytes exactly,
/// through a mapping the rest of whose pages the CPU still sees untouched. Metal's offset-1 shape
/// is checked for its creation, address and `contents` only (Metal's own uint accesses there are
/// garbage).
#[test]
fn ranges_inside_pages_import_the_pages_they_touch() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let page = host_page();
        let memory = Mapping::anonymous(4 * page.max(GUEST_PAGE));
        let mut answers = Vec::new();
        for (offset, length) in [
            (0, 100),
            (0, page + 4096),
            (4096, 4096),
            (256, 4096),
            (16, 4096),
            (16, 100),
            (4, 4096),
            (page - 8, 16),
            (1, 4096),
        ] {
            memory.fill(0x3c);
            let pointer = NonNull::new(memory.pointer().as_ptr().wrapping_add(offset)).unwrap();
            // SAFETY: the mapping outlives the buffer.
            let made =
                unsafe { no_copy(device, pointer, length, ResourceOptions::default(), None) };
            let buffer = made.unwrap_or_else(|error| panic!("offset {offset}: {error:?}"));
            answers.push(format!("{offset}+{length}: made"));
            assert_eq!(buffer.contents(), Some(pointer), "offset {offset}");
            assert_eq!(buffer.length(), length as u64);
            assert_eq!(
                buffer.gpu_address().0 % page as u64,
                (offset % page) as u64,
                "offset {offset}"
            );
            if offset % 4 != 0 {
                continue;
            }
            let bytes = counting(length, offset);
            memory.write(offset, &bytes);
            assert_eq!(
                gpu_read(&queue, &buffer, 0..length as u64),
                bytes,
                "offset {offset}"
            );
            gpu_fill(&queue, &buffer, 0..length as u64, 0xe1);
            let all = memory.read(0..memory.len);
            assert!(all[offset..offset + length]
                .iter()
                .all(|&byte| byte == 0xe1));
            assert!(
                all[..offset].iter().all(|&byte| byte == 0x3c),
                "offset {offset}"
            );
            assert!(
                all[offset + length..].iter().all(|&byte| byte == 0x3c),
                "offset {offset}"
            );
        }
        eprintln!("runtime ranges inside pages: {}", answers.join(", "));
    });
}

/// Two no-copy buffers over the same pages (Metal makes both): each sees what the GPU wrote
/// through the other. The second lies in the first's import, at its own Metal address; the first's
/// deallocator runs as the first goes, the second living on in the import.
#[test]
fn two_buffers_over_the_same_pages_share_them() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let memory = Mapping::anonymous(GUEST_PAGE);
        let seen = Arc::new(Seen::default());
        // SAFETY: the mapping outlives both buffers.
        let first = unsafe {
            no_copy(
                device,
                memory.pointer(),
                GUEST_PAGE,
                ResourceOptions::default(),
                Some(recording(&seen, ())),
            )
        }
        .unwrap();
        // SAFETY: as above.
        let second = unsafe {
            no_copy(
                device,
                memory.pointer(),
                GUEST_PAGE,
                ResourceOptions::default(),
                None,
            )
        }
        .unwrap();
        assert_ne!(first.gpu_address(), second.gpu_address());
        gpu_fill(&queue, &first, 0..64, 0x71);
        assert_eq!(gpu_read(&queue, &second, 0..64), [0x71; 64]);
        assert!(memory.read(0..64).iter().all(|&byte| byte == 0x71));
        drop(first);
        assert_eq!(
            seen.runs.load(Ordering::SeqCst),
            1,
            "the first's deallocator"
        );
        gpu_fill(&queue, &second, 64..128, 0x5a);
        assert!(memory.read(64..128).iter().all(|&byte| byte == 0x5a));
        assert_eq!(gpu_read(&queue, &second, 0..64), [0x71; 64]);
        eprintln!("runtime same pages twice: both made");
    });
}

/// Where a no-copy buffer made over a host address after the pages there were replaced writes,
/// while a buffer made over the same address before still lives: the consumer's `[blocks]` setup
/// (its suite's `interplay_memlive_unmap_retarget_map` and `memory_gap_table_rewrite`), which
/// keeps one host address for a guest address across the guest's unmap and remap. A over guest
/// page P1 at address X, used by the GPU; X remapped to guest page P2 in place; C made over X
/// (after `host_range_remapped` over X when `report`), then a fill of C's bytes 0x30..0x40.
/// Metal wires the pages at X when it makes C, so the fill lands in P2. Answers where it landed:
/// in P1, P2, or neither, and what a read of C's bytes gives (P2's, or P1's); `None`, printed,
/// where the device imports no shared memory (refused typed, as
/// `host_memory_kinds_are_imported_or_refused_typed` shows).
fn after_a_remap(device: &Device, report: bool) -> Option<(&'static str, bool)> {
    let queue = device.new_command_queue().unwrap();
    let ram = GuestRam::new(2 * GUEST_PAGE);
    ram.write(0, &[0x11; GUEST_PAGE]);
    ram.write(GUEST_PAGE, &counting(GUEST_PAGE, 5));
    let task = ram.map(0, GUEST_PAGE, libc::PROT_READ | libc::PROT_WRITE);
    let at = task.pointer();
    // SAFETY: the mapping outlives both buffers (they drop first).
    let a = match unsafe { no_copy(device, at, GUEST_PAGE, ResourceOptions::default(), None) } {
        Ok(a) => a,
        Err(error) => {
            assert!(not_importable(&error), "{error:?}");
            eprintln!(
                "REMAP {} reported={report}: refused (shared memory)",
                device.name()
            );
            return None;
        }
    };
    // A in use.
    assert_eq!(gpu_read(&queue, &a, 0..16), [0x11; 16]);
    task.remap(ram.fd, GUEST_PAGE);
    assert_eq!(task.read(0..16), counting(16, 5));
    if report {
        device.host_range_remapped(at, GUEST_PAGE as u64);
    }
    // SAFETY: as above.
    let c = unsafe { no_copy(device, at, GUEST_PAGE, ResourceOptions::default(), None) }.unwrap();
    gpu_fill(&queue, &c, 0x30..0x40, 0xc3);
    let p1 = ram.read(0x30..0x40) == [0xc3; 16];
    let p2 = ram.read(GUEST_PAGE + 0x30..GUEST_PAGE + 0x40) == [0xc3; 16];
    let landed = match (p1, p2) {
        (false, true) => "the new pages",
        (true, false) => "the old pages",
        (true, true) => "both",
        (false, false) => "neither",
    };
    let mut expected = counting(GUEST_PAGE, 5);
    expected[0x30..0x40].fill(0xc3);
    let reads_new = gpu_read(&queue, &c, 0..GUEST_PAGE as u64) == expected;
    eprintln!(
        "REMAP {} reported={report}: C's fill landed in {landed}; a read of C gives the new \
         pages' bytes: {reads_new}",
        device.name()
    );
    drop((a, c));
    drop(task);
    Some((landed, reads_new))
}

/// After the caller reports the remap (`Device::host_range_remapped`), a no-copy buffer made
/// over the address lies over the new pages, as Metal's does, though a buffer made over the same
/// bytes before still lives.
#[test]
fn a_buffer_made_after_a_reported_remap_lies_over_the_new_pages() {
    on_device(|device| {
        if let Some(answer) = after_a_remap(device, true) {
            assert_eq!(answer, ("the new pages", true));
        }
    });
}

/// Without the report, a no-copy buffer made over exactly the bytes a live one was made over lies
/// in that one's import, whose pages are the device's to keep or follow after a remap: printed, not
/// checked (`REMAP ... reported=false`).
#[test]
fn a_buffer_made_after_an_unreported_remap_shares_the_live_import() {
    on_device(|device| {
        after_a_remap(device, false);
    });
}

/// Which pages a no-copy buffer uses when the caller maps other pages at its address between its
/// making and its first GPU use: guest page P1 mapped at address X, B made over X, X remapped in
/// place to guest page P2 of the same shared memory object (`map_in_place`), and only then B's
/// first use, a read and a fill of its bytes 0x30..0x40. Metal keeps the pages at the making, and
/// so does the runtime on macOS, whose import is MoltenVK's `newBufferWithBytesNoCopy:` in the
/// making call: checked there. Elsewhere the driver decides, and the remap breaks
/// `HostMemory::new`'s contract: printed only (`REMAP-BEFORE-USE`; lavapipe
/// follows the address, its imports reading through the pointer). Where the device imports no
/// shared memory, the refusal is printed (typed, as
/// `host_memory_kinds_are_imported_or_refused_typed` shows).
#[test]
fn a_no_copy_buffer_uses_the_pages_mapped_when_it_was_made() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let ram = GuestRam::new(2 * GUEST_PAGE);
        ram.write(0, &[0x11; GUEST_PAGE]);
        ram.write(GUEST_PAGE, &counting(GUEST_PAGE, 5));
        let task = ram.map(0, GUEST_PAGE, libc::PROT_READ | libc::PROT_WRITE);
        // SAFETY: the mapping outlives the buffer (it drops first), and stays mapped through the
        // remap below.
        let made = unsafe {
            no_copy(
                device,
                task.pointer(),
                GUEST_PAGE,
                ResourceOptions::default(),
                None,
            )
        };
        let b = match made {
            Ok(b) => b,
            Err(error) => {
                assert!(not_importable(&error), "{error:?}");
                eprintln!(
                    "REMAP-BEFORE-USE {}: refused (shared memory)",
                    device.name()
                );
                return;
            }
        };
        task.remap(ram.fd, GUEST_PAGE);
        assert_eq!(task.read(0..16), counting(16, 5));
        let read = gpu_read(&queue, &b, 0..16);
        gpu_fill(&queue, &b, 0x30..0x40, 0xc3);
        let p1 = ram.read(0x30..0x40) == [0xc3; 16];
        let p2 = ram.read(GUEST_PAGE + 0x30..GUEST_PAGE + 0x40) == [0xc3; 16];
        eprintln!(
            "REMAP-BEFORE-USE {}: a read of B gives the pages at its making: {}; its fill landed \
             in them: {p1}, in the pages mapped after: {p2}",
            device.name(),
            read == [0x11; 16]
        );
        if cfg!(target_os = "macos") {
            assert_eq!(
                read, [0x11; 16],
                "B read other pages than those at its making"
            );
            assert!(
                p1 && !p2,
                "B's fill landed in other pages than those at its making"
            );
        }
        drop(b);
        drop(task);
    });
}

/// The consumer reports a remap from a no-copy buffer's deallocator, where it unmaps
/// (`Device::host_range_remapped` called inside it): the runtime holds no lock the call takes
/// while it runs a deallocator, whether the buffer's last release runs it on the program's thread
/// or a command buffer's completion does (`held`). A over fresh private pages at address X and B
/// over the same bytes (lying in A's import) are used; A's deallocator maps new private pages at
/// X, writes them, and reports it; C, made over X after, writes the new pages. Private anonymous
/// memory, which every venue imports (the consumer's shared guest RAM is refused typed where a
/// driver imports none, `a_buffer_made_after_a_reported_remap_lies_over_the_new_pages`).
fn a_remap_reported_from_a_deallocator(device: &Device, held: bool) {
    let queue = device.new_command_queue().unwrap();
    let task = Mapping::anonymous(GUEST_PAGE);
    task.fill(0x11);
    let at = task.pointer();
    let (sender, receiver) = std::sync::mpsc::channel();
    let deallocator: Deallocator = {
        let device = device.clone();
        Box::new(move |pointer: NonNull<u8>, length: u64| {
            // SAFETY: the task mapping's pages, which A no longer uses; B lies in its import.
            unsafe { anonymous_in_place(pointer, length as usize) };
            let fresh = counting(length as usize, 5);
            // SAFETY: the new pages, mapped writable just above.
            unsafe { std::ptr::copy_nonoverlapping(fresh.as_ptr(), pointer.as_ptr(), fresh.len()) };
            device.host_range_remapped(pointer, length);
            sender.send(std::thread::current().id()).unwrap();
        })
    };
    // SAFETY: the mapping outlives the buffers (they drop first); A's deallocator replaces its
    // pages in place, after A's last use.
    let a = unsafe {
        no_copy(
            device,
            at,
            GUEST_PAGE,
            ResourceOptions::default(),
            Some(deallocator),
        )
    }
    .unwrap();
    // SAFETY: as above.
    let b = unsafe { no_copy(device, at, GUEST_PAGE, ResourceOptions::default(), None) }.unwrap();
    gpu_fill(&queue, &a, 0..16, 0x22);
    assert_eq!(gpu_read(&queue, &b, 0..16), [0x22; 16]);
    let command_buffer = held.then(|| {
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.fill_buffer(&a, 16..32, 0x33).unwrap();
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer
    });
    drop(a);
    let thread = receiver
        .recv_timeout(WAIT)
        .expect("A's deallocator reported the remap and returned");
    if let Some(command_buffer) = command_buffer {
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
    }
    drop(b);
    assert_eq!(task.read(0..GUEST_PAGE), counting(GUEST_PAGE, 5));
    // SAFETY: as above.
    let c = unsafe { no_copy(device, at, GUEST_PAGE, ResourceOptions::default(), None) }.unwrap();
    gpu_fill(&queue, &c, 0x30..0x40, 0xc3);
    let mut expected = counting(GUEST_PAGE, 5);
    expected[0x30..0x40].fill(0xc3);
    assert_eq!(task.read(0..GUEST_PAGE), expected);
    assert_eq!(gpu_read(&queue, &c, 0..GUEST_PAGE as u64), expected);
    eprintln!(
        "REMAP-DEALLOCATOR {} held={held}: reported on the program's thread: {}; C wrote the \
         new pages",
        device.name(),
        thread == std::thread::current().id()
    );
    drop(c);
    drop(task);
}

/// A remap reported from the deallocator the buffer's last release runs on the program's thread.
#[test]
fn a_deallocator_run_by_a_release_may_report_a_remap() {
    on_device(|device| a_remap_reported_from_a_deallocator(device, false));
}

/// A remap reported from the deallocator a command buffer's completion runs.
#[test]
fn a_deallocator_run_by_a_completion_may_report_a_remap() {
    on_device(|device| a_remap_reported_from_a_deallocator(device, true));
}

/// Remaps reported from any thread while others make, use and release no-copy buffers: two
/// threads report, over and over, each worker's page, the 8 GiB from the GiB the lowest page
/// lies in, and the whole address space, while four workers each make two buffers over their own page
/// (the second lying in the first's import, or importing anew after a report), have the GPU
/// read the CPU's bytes through the second and write through the first, and release both, the
/// first's deallocator reporting its own bytes. No page changed, so every answer is the one
/// without reports, and nothing blocks.
#[test]
fn remaps_reported_from_any_thread_change_no_answer() {
    const WORKERS: usize = 4;
    const ROUNDS: usize = 24;
    on_device_with(at_plan, |device| {
        let pages: Vec<Mapping> = (0..WORKERS)
            .map(|_| Mapping::anonymous(GUEST_PAGE))
            .collect();
        let addresses: Vec<usize> = pages
            .iter()
            .map(|page| page.pointer().as_ptr() as usize)
            .collect();
        let lowest = addresses.iter().min().copied().unwrap();
        let reservation = (lowest & !((1 << 30) - 1), 8u64 << 30);
        let reports = Arc::new(AtomicUsize::new(0));
        let name = device.name();
        let done = std::sync::atomic::AtomicBool::new(false);
        let (sender, receiver) = std::sync::mpsc::channel();
        let device = device.clone();
        std::thread::spawn(move || {
            std::thread::scope(|scope| {
                let workers: Vec<_> = pages
                    .into_iter()
                    .enumerate()
                    .map(|(worker, page)| {
                        let (device, reports) = (&device, &reports);
                        scope.spawn(move || {
                            let queue = device.new_command_queue().unwrap();
                            for round in 0..ROUNDS {
                                let value = ((worker * ROUNDS + round) as u8) | 1;
                                page.write(0, &[value; 64]);
                                let deallocator: Deallocator = {
                                    let (device, reports) = (device.clone(), Arc::clone(reports));
                                    Box::new(move |pointer: NonNull<u8>, length: u64| {
                                        device.host_range_remapped(pointer, length);
                                        reports.fetch_add(1, Ordering::SeqCst);
                                    })
                                };
                                let options = ResourceOptions::default();
                                // SAFETY: the page outlives both buffers.
                                let first = unsafe {
                                    no_copy(
                                        device,
                                        page.pointer(),
                                        GUEST_PAGE,
                                        options,
                                        Some(deallocator),
                                    )
                                }
                                .unwrap();
                                // SAFETY: as above.
                                let second = unsafe {
                                    no_copy(device, page.pointer(), GUEST_PAGE, options, None)
                                }
                                .unwrap();
                                assert_eq!(gpu_read(&queue, &second, 0..64), [value; 64]);
                                gpu_fill(&queue, &first, 64..128, !value);
                                assert_eq!(page.read(64..128), [!value; 64]);
                                drop((first, second));
                            }
                        })
                    })
                    .collect();
                for _ in 0..2 {
                    let (device, done, addresses) = (&device, &done, &addresses);
                    scope.spawn(move || {
                        let report = |at: usize, length: u64| {
                            device.host_range_remapped(NonNull::new(at as *mut u8).unwrap(), length)
                        };
                        while !done.load(Ordering::SeqCst) {
                            for &at in addresses {
                                report(at, GUEST_PAGE as u64);
                            }
                            report(reservation.0, reservation.1);
                            report(1, u64::MAX - 1);
                            std::thread::yield_now();
                        }
                    });
                }
                for worker in workers {
                    worker.join().unwrap();
                }
                done.store(true, Ordering::SeqCst);
            });
            // The first buffers' deallocators run once nothing holds their pages.
            let deadline = std::time::Instant::now() + WAIT;
            while reports.load(Ordering::SeqCst) < WORKERS * ROUNDS
                && std::time::Instant::now() < deadline
            {
                std::thread::sleep(Duration::from_millis(5));
            }
            assert_eq!(reports.load(Ordering::SeqCst), WORKERS * ROUNDS);
            drop(device);
            sender.send(()).unwrap();
        });
        receiver
            .recv_timeout(4 * WAIT)
            .expect("every thread finished, none blocked");
        eprintln!(
            "REMAP-THREADS {}: {WORKERS} workers of {ROUNDS} rounds and two reporters: every \
             answer as without reports",
            name
        );
    });
}

/// A copy out of a no-copy buffer reads past its length to the end of its host pages: the
/// caller's bytes there, the current ones, as plain Metal reads them (an M4 under macOS 27.0.1;
/// its API validation stops them). The consumer's transfer buffer: 0x3f bytes at the start of a
/// page-aligned mapping (and at 256 bytes into it), options 0, paged into a texture and copied into
/// a buffer. A copy past the pages is refused (`Invalid`), as past any other buffer's length.
#[test]
fn copies_out_of_a_no_copy_buffer_read_its_pages_past_its_length() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let page = host_page();
        let memory = Mapping::anonymous(2 * page.max(GUEST_PAGE));
        let pattern = |seed: usize| -> Vec<u8> {
            (0..memory.len)
                .map(|i| (i * 7 + seed) as u8)
                .collect::<Vec<u8>>()
        };
        let mut texture = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 8, 8, false);
        texture.usage = TextureUsage::SHADER_READ;
        texture.resource_options = ResourceOptions::shared();
        let texture = device.new_texture(&texture).unwrap();
        let paged = |buffer: &Buffer, offset: u64| {
            let command_buffer = queue.command_buffer().unwrap();
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            let size = Size::new(8, 8, 1);
            let copied = blit.copy_from_buffer_to_texture(
                buffer,
                offset,
                32,
                0,
                size,
                &texture,
                0,
                0,
                Origin::new(0, 0, 0),
                BlitOption::None,
            );
            blit.end_encoding().unwrap();
            copied?;
            run(&command_buffer);
            let mut bytes = vec![0; 256];
            texture.get_bytes(&mut bytes, 32, 0, Region::new_2d(0, 0, 8, 8), 0, 0)?;
            Ok::<_, Error>(bytes)
        };
        for offset in [0, 256] {
            let pointer = NonNull::new(memory.pointer().as_ptr().wrapping_add(offset)).unwrap();
            // SAFETY: the mapping outlives the buffer.
            let made = unsafe { no_copy(device, pointer, 0x3f, ResourceOptions::default(), None) };
            let buffer = match made {
                Ok(buffer) => buffer,
                Err(error) if not_importable(&error) => {
                    eprintln!("runtime no-copy reads past the length at {offset}: {error}");
                    continue;
                }
                Err(error) => panic!("offset {offset}: {error:?}"),
            };
            // The rest of the host page from the pointer.
            let reach = (page - offset) as u64;
            let bytes = pattern(3);
            memory.write(0, &bytes);
            assert_eq!(
                gpu_read(&queue, &buffer, 0..reach),
                bytes[offset..page],
                "offset {offset}"
            );
            assert_eq!(
                paged(&buffer, 0x20).unwrap(),
                bytes[offset + 0x20..offset + 0x120],
                "offset {offset}"
            );
            // The bytes the copy reads are the current ones.
            let later = pattern(0x5d);
            memory.write(0, &later);
            assert_eq!(
                paged(&buffer, reach - 256).unwrap(),
                later[page - 256..page],
                "offset {offset}"
            );
            // Past the page.
            let copy = device
                .new_buffer(reach + 4, ResourceOptions::shared())
                .unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            assert_eq!(
                blit.copy_from_buffer_to_buffer(&buffer, 0, &copy, 0, reach + 4),
                Err(invalid(
                    "copy_from_buffer_to_buffer",
                    Problem::RangeOutOfBounds {
                        offset: 0,
                        size: reach + 4,
                        length: 0x3f,
                    }
                ))
            );
            blit.end_encoding().unwrap();
            assert_eq!(
                paged(&buffer, reach - 252),
                Err(invalid(
                    "copy_from_buffer_to_texture",
                    Problem::Copy(CopyProblem::BufferRange)
                ))
            );
        }
    });
}

/// Which kinds of host memory this device imports. Private anonymous memory (what allocators
/// return) is made on every venue, and the GPU reads and writes it. Shared memory, anonymous or a
/// second mapping of a POSIX shared memory object (the consumer's guest RAM), is made where the
/// driver imports it and refused typed where it does not; Metal takes every kind.
#[test]
fn host_memory_kinds_are_imported_or_refused_typed() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let ram = GuestRam::new(2 * GUEST_PAGE);
        let mut answers = Vec::new();
        for (kind, memory) in [
            ("private anonymous", Mapping::anonymous(GUEST_PAGE)),
            ("shared anonymous", Mapping::shared_anonymous(GUEST_PAGE)),
            (
                "shared memory object",
                ram.map(GUEST_PAGE, GUEST_PAGE, libc::PROT_READ | libc::PROT_WRITE),
            ),
        ] {
            let bytes = counting(GUEST_PAGE, kind.len());
            memory.write(0, &bytes);
            // SAFETY: the mapping outlives the buffer.
            let made = unsafe {
                no_copy(
                    device,
                    memory.pointer(),
                    GUEST_PAGE,
                    ResourceOptions::default(),
                    None,
                )
            };
            match made {
                Ok(buffer) => {
                    assert_eq!(
                        gpu_read(&queue, &buffer, 0..GUEST_PAGE as u64),
                        bytes,
                        "{kind}"
                    );
                    gpu_fill(&queue, &buffer, 0..64, 0x93);
                    assert!(
                        memory.read(0..64).iter().all(|&byte| byte == 0x93),
                        "{kind}"
                    );
                    answers.push(format!("{kind}: made"));
                }
                Err(error) => {
                    assert_ne!(kind, "private anonymous", "{error:?}");
                    assert!(not_importable(&error), "{kind}: {error:?}");
                    answers.push(format!("{kind}: refused"));
                }
            }
        }
        eprintln!("runtime host memory kinds: {}", answers.join(", "));
    });
}

/// A read-only page (Metal makes the buffer, its GPU reads it, and its GPU's writes to it are
/// dropped). On Linux the runtime makes the buffer over a copy of it whose GPU writes are dropped
/// (`WorkCounts::copied_unwritable_pages`), as it does every page the process cannot write: Linux
/// drivers pin imported pages for writing (measured at 83a57ba2: RADV refused the page, NVIDIA
/// failed it as out of device memory, ANV made it and lost the device at the GPU's read; lavapipe
/// read it). Elsewhere (MoltenVK, which hands the range to Metal) it is imported. Either way the
/// GPU reads what the CPU wrote before the page became read-only; on Linux the GPU's fill of the
/// copy leaves the page as it was.
#[test]
fn a_read_only_page_is_a_copy_on_linux_and_read_elsewhere() {
    on_device(|device| {
        let memory = Mapping::anonymous(GUEST_PAGE);
        let bytes = counting(GUEST_PAGE, 3);
        memory.write(0, &bytes);
        memory.protect_read_only();
        // SAFETY: the mapping outlives the buffer, and the GPU writes only its copy (Linux).
        let made = unsafe {
            no_copy(
                device,
                memory.pointer(),
                GUEST_PAGE,
                ResourceOptions::default(),
                None,
            )
        };
        let queue = device.new_command_queue().unwrap();
        let buffer = made.unwrap();
        assert_eq!(gpu_read(&queue, &buffer, 0..GUEST_PAGE as u64), bytes);
        if cfg!(target_os = "linux") {
            assert_eq!(buffer.host_backing(), Some(HostBacking::Copied));
            assert_eq!(
                device.work_counts().copied_unwritable_pages,
                1,
                "one copy, counted"
            );
            gpu_fill(&queue, &buffer, 0..64, 0x5a);
            assert_eq!(
                memory.read(0..GUEST_PAGE),
                bytes,
                "the GPU's write was dropped"
            );
            eprintln!("runtime read-only page: a copy, read, the GPU's write dropped");
        } else {
            eprintln!("runtime read-only page: made, read");
        }
    });
}

/// Pages whose write permission the caller vouches for (`HostMemory::assume_writable`): a buffer
/// over them asks the kernel about none of their mappings (`WorkCounts::host_mappings_queried`),
/// where a buffer over the same kind of pages without the vouching asks about each on Linux (one
/// mapping here) and nothing elsewhere; both are made, and the GPU reads and writes their bytes.
#[test]
fn vouched_pages_are_imported_with_no_mapping_asked() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let queried = || device.work_counts().host_mappings_queried;
        for vouched in [false, true] {
            let memory = Mapping::anonymous(GUEST_PAGE);
            let bytes = counting(GUEST_PAGE, if vouched { 7 } else { 8 });
            memory.write(0, &bytes);
            let before = queried();
            // SAFETY: the mapping outlives the buffer, writable while it lives.
            let buffer = unsafe {
                let host = HostMemory::new(memory.pointer());
                let host = if vouched {
                    host.assume_writable()
                } else {
                    host
                };
                device.new_buffer_with_bytes_no_copy(
                    host,
                    GUEST_PAGE as u64,
                    ResourceOptions::default(),
                    None,
                )
            }
            .unwrap();
            let asked = queried() - before;
            let expected = u64::from(cfg!(target_os = "linux") && !vouched);
            assert_eq!(asked, expected, "vouched {vouched}: mappings asked");
            assert_eq!(gpu_read(&queue, &buffer, 0..GUEST_PAGE as u64), bytes);
            gpu_fill(&queue, &buffer, 0..64, 0x6b);
            // SAFETY: the mapping lives, and no command writes it now.
            let written = unsafe { std::slice::from_raw_parts(memory.pointer().as_ptr(), 64) };
            assert_eq!(written, [0x6b; 64], "vouched {vouched}: the GPU's write");
            eprintln!("runtime vouched {vouched}: {asked} mappings asked");
        }
    });
}

/// Under the program's own MoltenVK setting that loses the device after a GPU failure
/// (`MVK_CONFIG_RESUME_LOST_DEVICE` off, which wins over the runtime's), a command buffer that
/// fills a buffer and then writes into read-only host pages, and command buffers of fills
/// committed beside it, are released as soon as each finished: the device's queues were drained
/// before they finished (`WorkCounts::queue_drains`), so MoltenVK's completed handlers that give
/// each fill's scratch buffer back to its command pool's pool (`vkCmdUpdateBuffer`, the fill's
/// form there) ran before the command pools went. Before, a wait for the failed command buffer's
/// fence returned as the first of its Metal command buffer's handlers marked the device lost,
/// the command pools went, and a later handler aborted the process (`mutex lock failed`): in
/// one run of `a_gpu_fault_fails_its_command_buffer_and_the_queue_runs_on` on each of two Apple
/// M4s. On MoltenVK alone, as that test. A GPU hazard (`m2v_gpu_hazards`): it faults the GPU on
/// purpose three times, and Metal's recovery fails other processes' command buffers in flight.
#[cfg(any(m2v_cpu_icd, m2v_gpu_hazards))]
#[test]
fn a_lost_device_s_command_buffers_let_go_of_their_pools_after_the_driver_s_completion() {
    /// Devices lost one after another, each by one fault.
    const ROUNDS: usize = 3;
    /// Command buffers of fills committed beside each faulting one.
    const BESIDE: usize = 8;
    for round in 0..ROUNDS {
        let cache = ScratchCache::new().unwrap();
        let messages = Messages::default();
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
            options.layer_settings.push(LayerSetting::new(
                c"MoltenVK".to_owned(),
                c"MVK_CONFIG_RESUME_LOST_DEVICE".to_owned(),
                LayerSettingValue::Bool(false),
            ));
        })
        .unwrap();
        if device.properties().description.driver_id != Some(DriverId::MOLTENVK) {
            eprintln!("runtime lost device's pools: not MoltenVK, not run");
            return;
        }
        let memory = Mapping::anonymous(GUEST_PAGE);
        memory.write(0, &counting(GUEST_PAGE, 7));
        memory.protect_read_only();
        // SAFETY: the mapping outlives the buffer; the GPU's write into it is the case.
        let read_only = unsafe {
            no_copy(
                &device,
                memory.pointer(),
                GUEST_PAGE,
                ResourceOptions::default(),
                None,
            )
        }
        .unwrap();
        let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 64, 16, false);
        descriptor.usage = TextureUsage::SHADER_READ;
        let texture = device.new_texture(&descriptor).unwrap();
        texture
            .replace_region(
                Region::new_2d(0, 0, 64, 16),
                0,
                0,
                &counting(64 * 16 * 4, 1),
                256,
                0,
            )
            .unwrap();
        let filled: Vec<Buffer> = (0..=BESIDE)
            .map(|_| device.new_buffer(256, ResourceOptions::shared()).unwrap())
            .collect();
        let queue = device.new_command_queue().unwrap();
        let before = device.work_counts();
        let faulting = queue.command_buffer().unwrap();
        let mut blit = faulting.blit_command_encoder().unwrap();
        blit.fill_buffer(&filled[0], 0..256, 0x5a).unwrap();
        blit.copy_from_texture_to_buffer(
            &texture,
            0,
            0,
            Origin::new(0, 0, 0),
            Size::new(64, 16, 1),
            &read_only,
            0,
            256,
            4096,
            BlitOption::None,
        )
        .unwrap();
        blit.end_encoding().unwrap();
        faulting.commit().unwrap();
        // Committed before or after the loss: refused at its commit, or failed with it.
        let beside: Vec<CommandBuffer> = filled[1..]
            .iter()
            .filter_map(|buffer| {
                let command_buffer = queue.command_buffer().ok()?;
                let mut fill = command_buffer.blit_command_encoder().ok()?;
                fill.fill_buffer(buffer, 0..256, 0xa5).ok()?;
                fill.end_encoding().ok()?;
                command_buffer.commit().ok()?;
                Some(command_buffer)
            })
            .collect();
        faulting.wait_until_completed_for(WAIT).unwrap();
        for command_buffer in &beside {
            command_buffer.wait_until_completed_for(WAIT).unwrap();
        }
        assert_eq!(faulting.status(), CommandBufferStatus::Error);
        // The device recovers from the loss (`device/line.rs`): the oldest incomplete command
        // buffer fails as the hang.
        assert_eq!(faulting.error(), Some(CommandBufferError::GpuHang));
        let drains = device.work_counts().queue_drains - before.queue_drains;
        eprintln!(
            "LOST-DEVICE-POOLS round {round}: {} committed beside the fault, {drains} drains",
            beside.len()
        );
        assert!(drains >= 1, "round {round}: the queues were not drained");
        // The command buffers' pools go now; a handler of MoltenVK's left to run would abort the
        // process in the moments after.
        drop((faulting, beside));
        std::thread::sleep(Duration::from_millis(200));
        let reported = messages.take();
        assert!(
            reported
                .iter()
                .all(|message| message.text.contains("execution failed (code")),
            "round {round}: {reported:#?}"
        );
        drop((read_only, texture, filled, queue, device));
        drop(memory);
        layer_reported_nothing(&messages, &format!("round {round}, after the teardown"));
    }
}

/// A GPU write into read-only host pages fails its command buffer, and the queue runs on, as
/// Metal answers (an M4 with macOS 27.0.1: `Internal`, 1; the consumer's M3 Max with 26.7:
/// `PageFault`, 3): MoltenVK reports the failed Metal command buffer, the runtime relays its code
/// to the command buffer whose batch it was, and the next command buffer completes
/// (`device/relay.rs`). A command buffer committed while the failing one runs finishes with its own
/// answer: completed, or, as Metal's victim of the recovery, `Execution(1)` or `PageFault`, never
/// the failed one's error. Under the program's own MoltenVK setting that loses the device after a
/// failure (`MVK_CONFIG_RESUME_LOST_DEVICE` off, which wins over the runtime's), MoltenVK marks the
/// device lost before it reports, so no batch's fence can be read: the device recovers from the
/// loss (`device/line.rs`), the failed command buffer fails as the hang (`GpuHang`) and one
/// committed beside it as its victim (`GpuRecoveryVictim`) or completed, and the next buffer and
/// command buffer are made on the new `VkDevice` and complete;
/// no runtime thread reaches the lost device outside a section of the exit
/// guard on the way (a debug build once ended the process there). MoltenVK's reports of the failure
/// are the only messages. On MoltenVK alone: on Linux the buffer is a copy whose GPU writes are
/// dropped, which no write faults (`a_read_only_page_is_a_copy_on_linux_and_read_elsewhere`). A GPU hazard (`m2v_gpu_hazards`): it faults the GPU on purpose twice, and
/// Metal's recovery fails other processes' command buffers in flight.
#[cfg(any(m2v_cpu_icd, m2v_gpu_hazards))]
#[test]
fn a_gpu_fault_fails_its_command_buffer_and_the_queue_runs_on() {
    for resume in [None, Some(false)] {
        let cache = ScratchCache::new().unwrap();
        let messages = Messages::default();
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
            if let Some(resume) = resume {
                options.layer_settings.push(LayerSetting::new(
                    c"MoltenVK".to_owned(),
                    c"MVK_CONFIG_RESUME_LOST_DEVICE".to_owned(),
                    LayerSettingValue::Bool(resume),
                ));
            }
        })
        .unwrap();
        if device.properties().description.driver_id != Some(DriverId::MOLTENVK) {
            eprintln!("runtime GPU fault: not MoltenVK, not run");
            return;
        }
        let memory = Mapping::anonymous(GUEST_PAGE);
        memory.write(0, &counting(GUEST_PAGE, 7));
        memory.protect_read_only();
        // SAFETY: the mapping outlives the buffer; the GPU's write into it is the case.
        let read_only = unsafe {
            no_copy(
                &device,
                memory.pointer(),
                GUEST_PAGE,
                ResourceOptions::default(),
                None,
            )
        }
        .unwrap();
        let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 64, 16, false);
        descriptor.usage = TextureUsage::SHADER_READ;
        let texture = device.new_texture(&descriptor).unwrap();
        texture
            .replace_region(
                Region::new_2d(0, 0, 64, 16),
                0,
                0,
                &counting(64 * 16 * 4, 1),
                256,
                0,
            )
            .unwrap();
        let queue = device.new_command_queue().unwrap();
        let faulting = queue.command_buffer().unwrap();
        let mut blit = faulting.blit_command_encoder().unwrap();
        blit.copy_from_texture_to_buffer(
            &texture,
            0,
            0,
            Origin::new(0, 0, 0),
            Size::new(64, 16, 1),
            &read_only,
            0,
            256,
            4096,
            BlitOption::None,
        )
        .unwrap();
        blit.end_encoding().unwrap();
        faulting.commit().unwrap();
        // A command buffer committed while the failing one runs: Metal's victim of the recovery,
        // or completed, with an answer of its own either way.
        let beside = device.new_buffer(256, ResourceOptions::shared()).unwrap();
        let victim = queue.command_buffer().unwrap();
        let mut fill = victim.blit_command_encoder().unwrap();
        fill.fill_buffer(&beside, 0..256, 0x5a).unwrap();
        fill.end_encoding().unwrap();
        let committed = victim.commit();
        faulting.wait_until_completed_for(WAIT).unwrap();
        if committed.is_ok() {
            victim.wait_until_completed_for(WAIT).unwrap();
        }
        let failed = faulting.error();
        let (beside_status, beside_error) = (victim.status(), victim.error());
        eprintln!(
            "runtime GPU fault, resume {resume:?}: {failed:?}; committed beside it: \
             {beside_status:?} {beside_error:?}"
        );
        assert_eq!(faulting.status(), CommandBufferStatus::Error, "{resume:?}");
        if resume.is_none() {
            assert_eq!(committed, Ok(()));
            assert!(
                matches!(
                    failed,
                    Some(CommandBufferError::PageFault | CommandBufferError::Execution(1))
                ),
                "{resume:?}: {failed:?}"
            );
            match beside_status {
                CommandBufferStatus::Completed => {
                    assert_eq!(read_buffer(&beside, 0, 256).unwrap(), vec![0x5a; 256]);
                }
                _ => assert!(
                    matches!(
                        (beside_status, beside_error),
                        (
                            CommandBufferStatus::Error,
                            Some(CommandBufferError::PageFault | CommandBufferError::Execution(1))
                        )
                    ),
                    "{beside_status:?} {beside_error:?}"
                ),
            }
        } else {
            assert_eq!(failed, Some(CommandBufferError::GpuHang), "{resume:?}");
            // Committed before the loss: failed with it as the hang's victim, or completed.
            assert_eq!(committed, Ok(()));
            assert!(
                matches!(
                    (beside_status, beside_error),
                    (CommandBufferStatus::Completed, None)
                        | (
                            CommandBufferStatus::Error,
                            Some(CommandBufferError::GpuRecoveryVictim)
                        )
                ),
                "{beside_status:?} {beside_error:?}"
            );
        }
        // The next command buffer, committed once the failed one completed, over a buffer made
        // then: after a loss, on the device's new `VkDevice`.
        let after = device.new_buffer(256, ResourceOptions::shared());
        let next = after.clone().and_then(|after| {
            let next = queue.command_buffer()?;
            let mut blit = next.blit_command_encoder()?;
            blit.fill_buffer(&after, 0..256, 0xd4)?;
            blit.end_encoding()?;
            next.commit()?;
            let _ = next.wait_until_completed_for(WAIT);
            Ok((next.status(), next.error()))
        });
        eprintln!("runtime GPU fault, resume {resume:?}, the next: {next:?}");
        assert_eq!(next, Ok((CommandBufferStatus::Completed, None)));
        assert_eq!(
            read_buffer(after.as_ref().unwrap(), 0, 256).unwrap(),
            vec![0xd4; 256]
        );
        let reported = messages.take();
        assert!(
            !reported.is_empty()
                && reported
                    .iter()
                    .all(|message| message.text.contains("execution failed (code")),
            "{resume:?}: {reported:#?}"
        );
        drop((
            faulting, victim, read_only, texture, after, beside, queue, device,
        ));
        drop(memory);
        layer_reported_nothing(&messages, &format!("{resume:?}, after the teardown"));
    }
}

/// Memory Metal makes no buffer over is refused by the call that makes the buffer, as Metal
/// answers nil there (pages mapped without access, pages not mapped, and a mapped page followed by
/// one that is not), never at a later use: typed, `Unsupported` (`HostMemory`) or `OutOfMemory` (on
/// Linux the runtime's check of the process's mappings, on macOS the import's answer), the
/// deallocator dropped unrun. Nothing reaches the GPU. Each answer is printed
/// (`REFUSED-AT-MAKING`). The driver may report each import it failed, as MoltenVK reports its
/// failed `vkAllocateMemory`; nothing else is reported. A mapped page followed by one the test
/// unmapped is made again over a fresh reservation where another thread's mapping took the
/// unmapped page before the making looked (the page seen mapped once a buffer was made), so the
/// making is held to memory whose page is unmapped as it looks. The copied host buffer's own
/// mapping was such a one until it read the caller's pages before making its memory (7 of 20
/// runs on the M3 Max at d65be37964, m3-max-5626).
#[test]
fn memory_metal_makes_no_buffer_over_is_refused_when_it_is_made() {
    // The half-mapped case's tries made again, each one more failed import the driver may report.
    let tries = AtomicUsize::new(0);
    let reported = on_device_reporting(|device| {
        let len = 2 * GUEST_PAGE;
        let reserve = || {
            // SAFETY: a fresh reservation of this test's, without access.
            let reserved = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    len,
                    libc::PROT_NONE,
                    libc::MAP_PRIVATE | libc::MAP_ANON,
                    -1,
                    0,
                )
            };
            assert_ne!(reserved, libc::MAP_FAILED, "mmap");
            NonNull::new(reserved.cast::<u8>()).unwrap()
        };
        let at = reserve();
        let seen = Arc::<Seen>::default();
        let mut answers = Vec::new();
        let refused = |what: &str, length: usize| {
            // SAFETY: deliberately not memory the contract asks for: the call refuses it before
            // any use, and the buffer is never made.
            let made = unsafe {
                no_copy(
                    device,
                    at,
                    length,
                    ResourceOptions::default(),
                    Some(recording(&seen, ())),
                )
            };
            let error = made
                .err()
                .unwrap_or_else(|| panic!("{what}: a buffer was made"));
            assert!(
                not_importable(&error) || matches!(error, Error::OutOfMemory(_)),
                "{what}: {error:?}"
            );
            format!("{what}: {error:?}")
        };
        answers.push(refused("pages mapped without access", GUEST_PAGE));
        // A mapped page and then one not mapped, in a reservation of their own: its second page
        // goes, its first is mapped readable and writable. Another thread's mapping (the
        // driver's, or one of the runtime's threads') may land in the page just unmapped before
        // the making looks at it: then every byte is mapped, and the making rightly makes a
        // buffer. Such a try, its page seen mapped again, is made again over a fresh reservation,
        // so the case holds the making to memory whose second page is unmapped as it looks.
        let mut retried = 0;
        loop {
            let half = reserve();
            // SAFETY: the reservation's own pages: its second goes, its first is mapped readable
            // and writable.
            unsafe {
                let second = half.as_ptr().add(GUEST_PAGE).cast();
                assert_eq!(libc::munmap(second, GUEST_PAGE), 0, "munmap");
                anonymous_in_place(half, GUEST_PAGE);
            }
            let what = "a mapped page and then one not mapped";
            let tried = Arc::<Seen>::default();
            // SAFETY: as `refused`'s.
            let made = unsafe {
                no_copy(
                    device,
                    half,
                    len,
                    ResourceOptions::default(),
                    Some(recording(&tried, ())),
                )
            };
            // SAFETY: a query of the second page's mapping, which reads and writes nothing.
            let second_mapped = unsafe {
                libc::msync(
                    half.as_ptr().add(GUEST_PAGE).cast(),
                    GUEST_PAGE,
                    libc::MS_ASYNC,
                )
            } == 0;
            match made {
                Err(error) => {
                    assert!(
                        not_importable(&error) || matches!(error, Error::OutOfMemory(_)),
                        "{what}: {error:?}"
                    );
                    assert_eq!(tried.runs.load(Ordering::SeqCst), 0, "{what}: ran");
                    assert_eq!(tried.dropped.load(Ordering::SeqCst), 1, "{what}: kept");
                    answers.push(format!("{what}: {error:?} ({retried} tries made again)"));
                    tries.store(retried, Ordering::SeqCst);
                    // SAFETY: the first page goes too, and with it the reservation.
                    assert_eq!(unsafe { libc::munmap(half.as_ptr().cast(), GUEST_PAGE) }, 0);
                    break;
                }
                Ok(buffer) => {
                    assert!(
                        second_mapped,
                        "{what}: a buffer was made, its second page unmapped still"
                    );
                    assert!(
                        retried < 8,
                        "{what}: another mapping took the page {retried} times"
                    );
                    drop(buffer);
                    retried += 1;
                    // SAFETY: the try's first page goes; the mapping that took the second is
                    // another's, left as it is.
                    assert_eq!(unsafe { libc::munmap(half.as_ptr().cast(), GUEST_PAGE) }, 0);
                }
            }
        }
        // SAFETY: the first reservation's pages go.
        assert_eq!(unsafe { libc::munmap(at.as_ptr().cast(), len) }, 0);
        answers.push(refused("pages not mapped", GUEST_PAGE));
        assert_eq!(seen.runs.load(Ordering::SeqCst), 0);
        assert_eq!(seen.dropped.load(Ordering::SeqCst), 2);
        eprintln!(
            "REFUSED-AT-MAKING {}: {}",
            device.name(),
            answers.join("; ")
        );
    });
    let (failed_imports, other): (Vec<_>, Vec<_>) = reported.into_iter().partition(|message| {
        message.id_name == "mvk-error"
            && message
                .text
                .starts_with("VK_ERROR_OUT_OF_DEVICE_MEMORY: vkAllocateMemory()")
    });
    assert!(
        other.is_empty(),
        "the validation layer reported: {other:#?}"
    );
    assert!(
        failed_imports.len() <= 3 + tries.load(Ordering::SeqCst),
        "more failed imports reported than made: {failed_imports:#?}"
    );
}

/// Each refusal drops the deallocator without running it, as Metal never runs it for a buffer it
/// did not make: what Metal stops (`Invalid`: Private and Memoryless storage), and what plain
/// Metal answers nil for (`Nil`: length 0, and its descriptor form, and a length past
/// `max_buffer_length`). (The private option bit the consumer's detached buffers
/// send, 0x1000000, is taken: `degrade_notices`'
/// `the_private_option_bit_drops_the_gpu_s_writes_to_no_copy_memory`.)
#[test]
fn refused_no_copy_buffers_never_run_their_deallocator() {
    on_device(|device| {
        let memory = Mapping::anonymous(GUEST_PAGE);
        let seen = Arc::<Seen>::default();
        let call = "new_buffer_with_bytes_no_copy";
        let max = device.properties().max_buffer_length;
        let nil = |problem| Error::Nil(Nil::new(call, problem));
        for (length, made, refusal) in [
            (
                0,
                ResourceOptions::default(),
                nil(Problem::BufferLength { length: 0 }),
            ),
            (
                GUEST_PAGE as u64,
                ResourceOptions::private(),
                invalid(call, Problem::StorageMode),
            ),
            (
                GUEST_PAGE as u64,
                options(
                    StorageMode::Memoryless,
                    CpuCacheMode::DefaultCache,
                    HazardTrackingMode::Default,
                ),
                invalid(call, Problem::StorageMode),
            ),
            (
                max + 1,
                ResourceOptions::default(),
                nil(Problem::BufferLength { length: max + 1 }),
            ),
        ] {
            // SAFETY: the mapping outlives the call, which makes nothing.
            let refused = unsafe {
                device.new_buffer_with_bytes_no_copy(
                    HostMemory::new(memory.pointer()),
                    length,
                    made,
                    Some(recording(&seen, ())),
                )
            };
            assert_eq!(refused.unwrap_err(), refusal);
        }
        // SAFETY: as above.
        let descriptor = BufferDescriptor::new(
            0,
            ResourceOptions::default(),
            BufferContents::NoCopy {
                memory: unsafe { HostMemory::new(memory.pointer()) },
                deallocator: Some(recording(&seen, ())),
                keep_alive: None,
            },
        );
        assert_eq!(
            device.new_buffer_with_descriptor(descriptor).unwrap_err(),
            Error::Nil(Nil::new(
                "new_buffer_with_descriptor",
                Problem::BufferLength { length: 0 }
            ))
        );
        assert_eq!(seen.runs.load(Ordering::SeqCst), 0);
        assert_eq!(seen.dropped.load(Ordering::SeqCst), 5);
    });
}

/// `new_buffer_with_bytes` and the descriptor's `Copy` hold a copy of the bytes in every storage
/// mode (a Private one read back by the GPU), and later writes to the source do not reach it; a
/// descriptor's `Zero` contents read zero; bytes of another length than the descriptor's, and no
/// bytes, are refused.
#[test]
fn buffers_from_bytes_hold_a_copy() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let bytes = counting(1000, 5);
        for made in [
            ResourceOptions::shared(),
            ResourceOptions::private(),
            options(
                StorageMode::Managed,
                CpuCacheMode::DefaultCache,
                HazardTrackingMode::Default,
            ),
        ] {
            let mut source = bytes.clone();
            let buffer = device.new_buffer_with_bytes(&source, made).unwrap();
            let described = device
                .new_buffer_with_descriptor(BufferDescriptor::new(
                    1000,
                    made,
                    BufferContents::Copy(&source),
                ))
                .unwrap();
            // The buffers hold copies: the source's later bytes do not reach them.
            source.fill(0);
            assert_eq!(buffer.length(), 1000);
            assert_eq!(gpu_read(&queue, &buffer, 0..1000), bytes, "{made:?}");
            assert_eq!(gpu_read(&queue, &described, 0..1000), bytes, "{made:?}");
        }
        let source = counting(64, 0);
        let buffer = device
            .new_buffer_with_bytes(&source, ResourceOptions::shared())
            .unwrap();
        drop(source);
        assert_eq!(read_buffer(&buffer, 0, 64).unwrap(), counting(64, 0));
        let zero = device
            .new_buffer_with_descriptor(BufferDescriptor::new(
                256,
                ResourceOptions::shared(),
                BufferContents::Zero,
            ))
            .unwrap();
        assert_eq!(read_buffer(&zero, 0, 256).unwrap(), [0; 256]);
        assert_eq!(
            device
                .new_buffer_with_descriptor(BufferDescriptor::new(
                    10,
                    ResourceOptions::shared(),
                    BufferContents::Copy(&[1, 2, 3]),
                ))
                .unwrap_err(),
            invalid("new_buffer_with_descriptor", Problem::LengthMismatch)
        );
        assert_eq!(
            device
                .new_buffer_with_bytes(&[], ResourceOptions::shared())
                .unwrap_err(),
            Error::Nil(Nil::new(
                "new_buffer_with_bytes",
                Problem::BufferLength { length: 0 }
            ))
        );
    });
}

/// Purgeable state as Metal answers it for buffers the runtime allocates: a buffer of 16 KiB or
/// more keeps its state, and `Empty` discards a Shared or Managed buffer's bytes, which read zero
/// after, and keeps a Private buffer's; a volatile buffer keeps its bytes (the runtime never
/// purges). A shorter buffer answers `NonVolatile` to every request and keeps its bytes, as Metal's
/// suballocated buffers did. Labels are kept.
#[test]
fn purgeable_state_answers_as_metal_and_empty_discards() {
    on_device(|device| {
        use PurgeableState::*;
        const LENGTH: u64 = 16384;
        const N: usize = LENGTH as usize;
        // Shorter than 16 KiB, and a multiple of 4 a blit copies.
        const SHORT: u64 = LENGTH - 4;
        let queue = device.new_command_queue().unwrap();
        for (made, emptied) in [
            (ResourceOptions::shared(), 0),
            (
                options(
                    StorageMode::Managed,
                    CpuCacheMode::DefaultCache,
                    HazardTrackingMode::Default,
                ),
                0,
            ),
            (ResourceOptions::private(), 0xab),
        ] {
            let buffer = device.new_buffer_with_bytes(&[0xab; N], made).unwrap();
            assert_eq!(buffer.label(), None);
            buffer.set_label("purgeable");
            assert_eq!(buffer.label().as_deref(), Some("purgeable"));
            assert_eq!(buffer.set_purgeable_state(KeepCurrent), Ok(NonVolatile));
            assert_eq!(buffer.set_purgeable_state(Volatile), Ok(NonVolatile));
            assert_eq!(buffer.set_purgeable_state(KeepCurrent), Ok(Volatile));
            assert_eq!(gpu_read(&queue, &buffer, 0..LENGTH), [0xab; N], "{made:?}");
            assert_eq!(buffer.set_purgeable_state(Empty), Ok(Volatile));
            assert_eq!(buffer.set_purgeable_state(KeepCurrent), Ok(Empty));
            assert_eq!(
                gpu_read(&queue, &buffer, 0..LENGTH),
                [emptied; N],
                "{made:?}"
            );
            assert_eq!(buffer.set_purgeable_state(NonVolatile), Ok(Empty));
            assert_eq!(buffer.set_purgeable_state(KeepCurrent), Ok(NonVolatile));

            let short = device
                .new_buffer_with_bytes(&[0xab; SHORT as usize], made)
                .unwrap();
            for state in [KeepCurrent, Volatile, KeepCurrent, Empty, KeepCurrent] {
                assert_eq!(
                    short.set_purgeable_state(state),
                    Ok(NonVolatile),
                    "{made:?}"
                );
            }
            assert_eq!(
                gpu_read(&queue, &short, 0..SHORT),
                [0xab; SHORT as usize],
                "{made:?}"
            );
        }
        let shared = buffer_with(device, &[0x5c; N]).unwrap();
        shared.set_purgeable_state(Empty).unwrap();
        assert_eq!(read_buffer(&shared, 0, LENGTH).unwrap(), [0; N]);
    });
}

/// What making a no-copy buffer over many GiB costs, now that its pages are imported when it is
/// made (reims-vgpu's guest RAM, 16 GiB of untouched memory; Metal makes it in under 5 ms and pays
/// 24 to 88 ms per GiB at its first use). At 1, 4, 8 and 16 GiB of fresh private anonymous memory,
/// those the device's `max_buffer_length` allows, three buffers made and dropped in turn: the times
/// are printed (`NOCOPY` lines: the best and the first creation, and the first use), a report, not
/// a bound, and the first command using the first buffer writes where it should. macOS only, where
/// the runtime once deferred the import: Linux drivers pin every
/// imported page when the buffer is made, as they always did. A GPU hazard (`m2v_gpu_hazards`):
/// its buffers import up to 16 GiB of host memory, as far as `max_buffer_length`.
#[cfg(all(target_os = "macos", any(m2v_cpu_icd, m2v_gpu_hazards)))]
#[test]
fn a_no_copy_buffer_s_pages_are_imported_when_it_is_made() {
    use std::time::Instant;
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let max = device.properties().max_buffer_length;
        for gib in [1u64, 4, 8, 16] {
            let len = gib << 30;
            if len > max {
                println!(
                    "NOCOPY {}: {gib} GiB past max_buffer_length {max}",
                    device.name()
                );
                continue;
            }
            let mapping = Mapping::anonymous(len as usize);
            let (mut cold, mut best, mut first_use) =
                (Duration::ZERO, Duration::MAX, Duration::ZERO);
            for round in 0..3 {
                let start = Instant::now();
                // SAFETY: the mapping is fresh private memory this test owns, unmapped after the
                // buffer is dropped.
                let buffer = unsafe {
                    no_copy(
                        device,
                        mapping.pointer(),
                        len as usize,
                        ResourceOptions::shared(),
                        None,
                    )
                }
                .unwrap();
                let took = start.elapsed();
                best = best.min(took);
                if round == 0 {
                    cold = took;
                    let start = Instant::now();
                    gpu_fill(&queue, &buffer, len - 4096..len - 4080, 0x5a);
                    first_use = start.elapsed();
                    let contents = buffer.contents().unwrap().as_ptr();
                    // SAFETY: the buffer's bytes are the mapping's, which the GPU no longer
                    // writes.
                    let written = unsafe {
                        std::slice::from_raw_parts(contents.add(len as usize - 4097), 18)
                    };
                    assert_eq!(written[0], 0, "{gib} GiB");
                    assert_eq!(&written[1..17], &[0x5a; 16], "{gib} GiB");
                    assert_eq!(written[17], 0, "{gib} GiB");
                }
                drop(buffer);
            }
            println!(
                "NOCOPY {}: {gib} GiB made in {:.3} ms (first {:.3} ms), first use {:.3} ms",
                device.name(),
                best.as_secs_f64() * 1e3,
                cold.as_secs_f64() * 1e3,
                first_use.as_secs_f64() * 1e3
            );
        }
    });
}
