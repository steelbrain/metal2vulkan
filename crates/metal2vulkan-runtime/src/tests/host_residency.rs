//! The residency of no-copy buffers' dma-buf imports (`host_residency.rs`), on the device the cfg
//! selects.
//!
//! The memory is guest RAM as QEMU's `memory-backend-memfd,share=on` makes it: a `memfd` sealed
//! against growing and shrinking, mapped shared, and its pages made into dma-bufs through
//! `/dev/udmabuf`, as a consumer's source makes them. Where the device refuses the memory's
//! pointer and imports the dma-bufs (RADV), it keeps a descriptor of each and releases the import
//! while idle. The tests release imports at once through the crate's hook
//! (`Device::release_idle_host_imports`), except one that waits for the device's residency
//! thread, and check:
//! - the counts that show the release and the import again ran (`WorkCounts::host_imports_released`,
//!   `host_imports_reimported`, `host_imports_resident`, `host_import_fds_kept`,
//!   `dma_buf_memories_live`);
//! - the bytes a blit reads and writes after a release, the CPU's writes made while released among
//!   them, and that the source is never called again;
//! - the pages a release keeps across a remap of the buffer's addresses;
//! - that a buffer whose Metal address was handed out stays imported;
//! - that a buffer past the descriptors the device keeps (capped through
//!   `Device::cap_kept_host_import_fds`) is made and stays imported, counted while it lives
//!   (`host_imports_kept_for_fds`).
//!
//! A device that imports the pointer (NVIDIA, lavapipe, ANV) runs none of it: each test says so
//! and passes, and `tests/pointer_residency.rs` checks the residency of pointer imports. Where
//! `/dev/udmabuf` cannot be opened each test says so and passes without running.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on Linux, on the kind of device the cfg selects.

use crate::{
    Buffer, CommandQueue, Device, DeviceKind, DeviceOptions, DmaBufChunk, DmaBufRequest,
    DmaBufSource, HostMemory, ResourceOptions, ShaderCacheOptions, WorkCounts,
};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// The longest a command buffer may take.
const WAIT: Duration = Duration::from_secs(60);

/// The longest a test waits for a completed command buffer to let go of an import, or for the
/// residency thread to release one.
const SETTLE: Duration = Duration::from_secs(20);

/// How long the device leaves an import idle before its residency thread releases it the first
/// time (`host_residency::IDLE`), less the residency clock's millisecond rounding.
const FIRST_WINDOW: Duration = Duration::from_millis(1990);

/// The udmabuf device.
const UDMABUF: &str = "/dev/udmabuf";

/// `struct udmabuf_create` of Linux's `include/uapi/linux/udmabuf.h`.
#[repr(C)]
struct Create {
    memfd: u32,
    flags: u32,
    offset: u64,
    size: u64,
}

const _: () = assert!(size_of::<Create>() == 24);

/// `UDMABUF_FLAGS_CLOEXEC`.
const UDMABUF_FLAGS_CLOEXEC: u32 = 0x01;

/// `UDMABUF_CREATE`, `_IOW('u', 0x42, struct udmabuf_create)` in the generic ioctl encoding.
const UDMABUF_CREATE: u64 =
    (1 << 30) | ((size_of::<Create>() as u64) << 16) | ((b'u' as u64) << 8) | 0x42;

/// The host's page size.
pub(super) fn host_page() -> usize {
    // SAFETY: `sysconf` reads a system constant.
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    usize::try_from(page).expect("a page size")
}

/// Whether this process may open `/dev/udmabuf`; where not, `test` says so.
fn udmabuf_openable(test: &str) -> bool {
    let opened = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(UDMABUF);
    if let Err(error) = &opened {
        eprintln!("{test}: {UDMABUF} cannot be opened ({error}): not run");
    }
    opened.is_ok()
}

/// A dma-buf of `memfd`'s `size` bytes from `offset`.
pub(super) fn udmabuf(memfd: &OwnedFd, offset: u64, size: u64) -> std::io::Result<OwnedFd> {
    let device = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(UDMABUF)?;
    let create = Create {
        memfd: memfd.as_raw_fd() as u32,
        flags: UDMABUF_FLAGS_CLOEXEC,
        offset,
        size,
    };
    // SAFETY: the ioctl reads `create`, a valid `udmabuf_create`.
    let fd = unsafe { libc::ioctl(device.as_raw_fd(), UDMABUF_CREATE as _, &create) };
    if fd < 0 {
        return Err(std::io::Error::last_os_error());
    }
    // SAFETY: a new descriptor the ioctl returned, owned at once.
    Ok(unsafe { OwnedFd::from_raw_fd(fd) })
}

/// A scratch directory for the device's shader cache, removed when dropped.
pub(super) struct Scratch(pub(super) PathBuf);

impl Scratch {
    pub(super) fn new() -> Scratch {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "m2v-host-residency-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&path).unwrap();
        Scratch(path)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The device the cfg selects: a CPU device under `m2v_cpu_icd`, else a GPU, its shader cache
/// in `cache`.
pub(super) fn device(cache: &Path) -> Device {
    let options = DeviceOptions {
        kinds: match cfg!(m2v_cpu_icd) {
            true => vec![DeviceKind::Cpu],
            false => vec![
                DeviceKind::Discrete,
                DeviceKind::Integrated,
                DeviceKind::Virtual,
            ],
        },
        shader_cache: ShaderCacheOptions {
            directory: Some(cache.to_owned()),
            ..ShaderCacheOptions::default()
        },
        ..DeviceOptions::default()
    };
    Device::system_default_with(&options).unwrap()
}

/// Run `test` on the device the cfg selects, where `/dev/udmabuf` opens.
fn on_device(name: &str, test: impl FnOnce(&Device)) {
    if !udmabuf_openable(name) {
        return;
    }
    let scratch = Scratch::new();
    let device = device(&scratch.0);
    test(&device);
}

/// Guest RAM: a `memfd` sealed against growing and shrinking, and a view of `len` of its bytes, a
/// shared mapping from a file offset that [`Ram::remap`] changes.
pub(super) struct Ram {
    fd: OwnedFd,
    at: NonNull<u8>,
    len: usize,
    /// The file offset the view maps now.
    file_at: Arc<AtomicUsize>,
}

impl Ram {
    /// A file of `file_len` bytes, its view of `len` bytes from `file_at`.
    pub(super) fn new(file_len: usize, file_at: usize, len: usize) -> Ram {
        // SAFETY: a new anonymous file, owned at once, then sized and sealed.
        let fd = unsafe {
            let fd = libc::memfd_create(
                c"m2v-host-residency".as_ptr(),
                libc::MFD_CLOEXEC | libc::MFD_ALLOW_SEALING,
            );
            assert!(fd >= 0, "memfd_create: {}", std::io::Error::last_os_error());
            OwnedFd::from_raw_fd(fd)
        };
        // SAFETY: the descriptor is this object's.
        unsafe {
            assert_eq!(libc::ftruncate(fd.as_raw_fd(), file_len as libc::off_t), 0);
            let seals = libc::F_SEAL_GROW | libc::F_SEAL_SHRINK | libc::F_SEAL_SEAL;
            assert_eq!(libc::fcntl(fd.as_raw_fd(), libc::F_ADD_SEALS, seals), 0);
        }
        // SAFETY: a new shared mapping of the file.
        let at = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd.as_raw_fd(),
                file_at as libc::off_t,
            )
        };
        assert_ne!(at, libc::MAP_FAILED, "mmap of the view");
        Ram {
            fd,
            at: NonNull::new(at.cast()).unwrap(),
            len,
            file_at: Arc::new(AtomicUsize::new(file_at)),
        }
    }

    /// The view's address `offset` bytes in.
    pub(super) fn pointer(&self, offset: usize) -> NonNull<u8> {
        assert!(offset < self.len);
        // SAFETY: inside the view.
        unsafe { NonNull::new_unchecked(self.at.as_ptr().add(offset)) }
    }

    /// Map the file from `file_at` over the view, in place of the pages there (`mmap` with
    /// `MAP_FIXED`), as a guest remaps the RAM at those addresses.
    pub(super) fn remap(&self, file_at: usize) {
        // SAFETY: the view's own range, mapped again.
        let mapped = unsafe {
            libc::mmap(
                self.at.as_ptr().cast(),
                self.len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED | libc::MAP_FIXED,
                self.fd.as_raw_fd(),
                file_at as libc::off_t,
            )
        };
        assert_eq!(mapped, self.at.as_ptr().cast(), "mmap over the view");
        self.file_at.store(file_at, Ordering::Release);
    }

    /// Write `bytes` into the file at `offset`, not through the view.
    pub(super) fn write_file(&self, offset: usize, bytes: &[u8]) {
        // SAFETY: `bytes` is readable for its length.
        let written = unsafe {
            libc::pwrite(
                self.fd.as_raw_fd(),
                bytes.as_ptr().cast(),
                bytes.len(),
                offset as libc::off_t,
            )
        };
        assert_eq!(written, bytes.len() as isize, "pwrite");
    }

    /// The file's `len` bytes from `offset`, not through the view.
    pub(super) fn read_file(&self, offset: usize, len: usize) -> Vec<u8> {
        let mut bytes = vec![0; len];
        // SAFETY: `bytes` is writable for its length.
        let read = unsafe {
            libc::pread(
                self.fd.as_raw_fd(),
                bytes.as_mut_ptr().cast(),
                len,
                offset as libc::off_t,
            )
        };
        assert_eq!(read, len as isize, "pread");
        bytes
    }

    /// The view's bytes.
    pub(super) fn view(&self) -> Vec<u8> {
        // SAFETY: the view's bytes; each test orders the CPU's and the GPU's accesses.
        unsafe { std::slice::from_raw_parts(self.at.as_ptr(), self.len).to_vec() }
    }

    /// A source of one dma-buf of the file pages behind each request's pages, at the file offset
    /// the view maps when it is called, counting its calls in `calls`.
    fn source(&self, calls: &Arc<AtomicUsize>) -> DmaBufSource {
        let fd = Arc::new(self.fd.try_clone().unwrap());
        let base = self.at.as_ptr() as usize;
        let file_at = Arc::clone(&self.file_at);
        let calls = Arc::clone(calls);
        Arc::new(move |request: DmaBufRequest| {
            calls.fetch_add(1, Ordering::AcqRel);
            let at = (request.start.as_ptr() as usize)
                .checked_sub(base)
                .ok_or_else(|| std::io::Error::other("pages before the view"))?;
            let offset = (file_at.load(Ordering::Acquire) + at) as u64;
            Ok(vec![DmaBufChunk {
                fd: udmabuf(&fd, offset, request.length)?,
                length: request.length,
            }])
        })
    }
}

impl Drop for Ram {
    fn drop(&mut self) {
        // SAFETY: the view is this object's, and no buffer uses it any more (each test drops its
        // buffers first).
        unsafe { libc::munmap(self.at.as_ptr().cast(), self.len) };
    }
}

/// Bytes `i % 251` from `start`.
pub(super) fn counting(len: usize, start: usize) -> Vec<u8> {
    (start..start + len).map(|i| (i % 251) as u8).collect()
}

/// A no-copy buffer of `length` bytes over `ram`'s view from `offset`, its source counting its
/// calls in `calls`: the buffer, and the dma-buf chunks the device imported for it (0 where it
/// imported the pointer, which the test then says).
fn no_copy(
    test: &str,
    device: &Device,
    ram: &Ram,
    (offset, length): (usize, usize),
    calls: &Arc<AtomicUsize>,
) -> (Buffer, u64) {
    let before = device.work_counts().dma_buf_imports;
    // SAFETY: each test's view outlives its buffers, and the source makes dma-bufs of its pages.
    let buffer = unsafe {
        device.new_buffer_with_bytes_no_copy(
            HostMemory::with_dma_buf(ram.pointer(offset), ram.source(calls)),
            length as u64,
            ResourceOptions::default(),
            None,
        )
    }
    .unwrap();
    let chunks = device.work_counts().dma_buf_imports - before;
    if chunks == 0 {
        eprintln!(
            "{test}: on {}: the pointer imported (`tests/pointer_residency.rs`)",
            device.name()
        );
    }
    (buffer, chunks)
}

/// Commit `command_buffer` and wait for it.
fn run(command_buffer: &crate::CommandBuffer) {
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
}

/// `buffer`'s first `length` bytes, read back by a blit, and how long the blit's call took to
/// encode (with an import again, where the device released the buffer's).
pub(super) fn gpu_read(
    queue: &CommandQueue,
    buffer: &Buffer,
    length: usize,
) -> (Vec<u8>, Duration) {
    let device = queue.device();
    let copy = device
        .new_buffer(length as u64, ResourceOptions::shared())
        .unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    let start = Instant::now();
    blit.copy_from_buffer_to_buffer(buffer, 0, &copy, 0, length as u64)
        .unwrap();
    let took = start.elapsed();
    blit.end_encoding().unwrap();
    run(&command_buffer);
    let at = copy.contents().unwrap();
    // SAFETY: the copy's own bytes, which its completed command buffer wrote.
    let bytes = unsafe { std::slice::from_raw_parts(at.as_ptr(), length).to_vec() };
    (bytes, took)
}

/// `buffer`'s first `length` bytes, read back by a blit.
pub(super) fn read(queue: &CommandQueue, buffer: &Buffer, length: usize) -> Vec<u8> {
    gpu_read(queue, buffer, length).0
}

/// Fill `range` of `buffer` with `value` by a blit.
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

/// Release `count` idle imports through the hook, retried until the command buffers that read
/// them let go (each at its completion, on the device's handler thread), within [`SETTLE`].
pub(super) fn release(test: &str, device: &Device, count: usize) {
    let start = Instant::now();
    let mut released = 0;
    while released < count {
        released += device.release_idle_host_imports();
        assert!(
            start.elapsed() < SETTLE,
            "{test}: {released} of {count} imports released within {SETTLE:?}"
        );
        std::thread::yield_now();
    }
    assert_eq!(released, count, "{test}: imports released");
}

/// The device's counts once `settled` holds of them, within [`SETTLE`].
pub(super) fn counts_once(
    test: &str,
    device: &Device,
    what: &str,
    settled: impl Fn(&WorkCounts) -> bool,
) -> WorkCounts {
    let start = Instant::now();
    loop {
        let counts = device.work_counts();
        if settled(&counts) {
            return counts;
        }
        assert!(start.elapsed() < SETTLE, "{test}: {what}: {counts:?}");
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// The residency counts of `counts`: released, imported again, resident, descriptors kept.
pub(super) fn residency(counts: &WorkCounts) -> (u64, u64, u64, u64) {
    (
        counts.host_imports_released,
        counts.host_imports_reimported,
        counts.host_imports_resident,
        counts.host_import_fds_kept,
    )
}

/// A no-copy buffer over 4 pages, where the device imports dma-bufs of them:
/// - read by a blit, its import released: its chunks' memories leave the device's live dma-buf
///   memories, the dma-bufs' descriptors kept; and released no further while released;
/// - the CPU's writes made while it is released, through `contents` and through the file, read
///   by the next blit, which imports it again from the descriptors kept, without calling the
///   source;
/// - released again, a GPU fill lands in the pages;
/// - released again, its Metal address handed out imports it again at once and for good,
///   closing the descriptors kept, and the hook releases it no more.
///
/// Where the device imports the pointer, the hook releases nothing and the counts stay 0 unless the
/// driver's pointer imports follow their address (`tests/pointer_residency.rs`).
#[test]
fn an_idle_dma_buf_import_is_released_and_imported_again_at_its_next_blit() {
    const TEST: &str = "dma-buf residency";
    on_device(TEST, |device| {
        let page = host_page();
        let len = 4 * page;
        let queue = device.new_command_queue().unwrap();
        let ram = Ram::new(len, 0, len);
        let first = counting(len, 7);
        ram.write_file(0, &first);
        let calls = Arc::new(AtomicUsize::new(0));
        let (buffer, chunks) = no_copy(TEST, device, &ram, (0, len), &calls);
        assert!(read(&queue, &buffer, len) == first, "{TEST}");
        if chunks == 0 {
            if !super::pointer_residency::follows(device) {
                assert_eq!(device.release_idle_host_imports(), 0, "{TEST}");
                assert_eq!(residency(&device.work_counts()), (0, 0, 0, 0), "{TEST}");
            }
            return;
        }
        let imported = device.work_counts();
        assert_eq!(residency(&imported), (0, 0, 1, chunks), "{TEST}");
        let live = imported.dma_buf_memories_live;

        release(TEST, device, 1);
        let released = device.work_counts();
        assert_eq!(residency(&released), (1, 0, 0, chunks), "{TEST}: released");
        assert_eq!(released.dma_buf_memories_live, live - chunks, "{TEST}");
        assert_eq!(
            device.release_idle_host_imports(),
            0,
            "{TEST}: released twice"
        );

        // The CPU writes while it is released: half through `contents`, half through the file.
        let second = counting(len, 59);
        let contents = buffer.contents().unwrap();
        // SAFETY: the buffer's own bytes, which no command reads now.
        unsafe { std::ptr::copy_nonoverlapping(second.as_ptr(), contents.as_ptr(), len / 2) };
        ram.write_file(len / 2, &second[len / 2..]);
        let (bytes, importing) = gpu_read(&queue, &buffer, len);
        assert!(bytes == second, "{TEST}: the bytes written while released");
        let (bytes, imported) = gpu_read(&queue, &buffer, len);
        assert!(bytes == second, "{TEST}");
        let again = device.work_counts();
        assert_eq!(
            residency(&again),
            (1, 1, 1, chunks),
            "{TEST}: imported again"
        );
        assert_eq!(again.dma_buf_memories_live, live, "{TEST}");
        assert_eq!(
            calls.load(Ordering::Acquire),
            1,
            "{TEST}: the source's calls"
        );

        // A GPU write after a release lands in the pages.
        release(TEST, device, 1);
        gpu_fill(&queue, &buffer, page as u64..3 * page as u64, 0x6d);
        let mut filled = second.clone();
        filled[page..3 * page].fill(0x6d);
        assert!(ram.read_file(0, len) == filled, "{TEST}: the GPU's fill");
        let counts = device.work_counts();
        assert_eq!(residency(&counts), (2, 2, 1, chunks), "{TEST}: filled");

        // Its Metal address handed out keeps it imported for good, from the moment it is.
        release(TEST, device, 1);
        let address = buffer.gpu_address();
        assert_ne!(address.0, 0, "{TEST}");
        let counts = device.work_counts();
        assert_eq!(residency(&counts), (3, 3, 1, 0), "{TEST}: exposed");
        assert_eq!(counts.dma_buf_memories_live, live, "{TEST}");
        assert!(read(&queue, &buffer, len) == filled, "{TEST}");
        assert_eq!(device.release_idle_host_imports(), 0, "{TEST}: exposed");
        assert_eq!(
            calls.load(Ordering::Acquire),
            1,
            "{TEST}: the source's calls"
        );
        drop(buffer);
        let gone = counts_once(TEST, device, "the import gone", |counts| {
            counts.host_imports_resident == 0 && counts.dma_buf_memories_live == live - chunks
        });
        assert_eq!(residency(&gone), (3, 3, 0, 0), "{TEST}: gone");
        eprintln!(
            "{TEST}: on {}: {chunks} chunks released 3 times, imported again 3 times; a copy's \
             call took {importing:?} importing them again, {imported:?} imported",
            device.name()
        );
    });
}

/// A no-copy buffer over a view of the file's pages 0 and 1, where the device imports dma-bufs
/// of them, read by a blit and its import released; then the view's addresses remapped to the
/// file's pages 4 and 5 and the device told (`Device::host_range_remapped`). The next blit of the
/// buffer reads pages 0 and 1, as Metal's buffer keeps the pages wired at its making: it is
/// imported again from the dma-bufs kept, not from the source, which would make dma-bufs of the
/// new pages. A buffer made after the remap over the same bytes reads pages 4 and 5, which the
/// CPU's view shows.
#[test]
fn a_released_import_comes_back_over_its_own_pages_after_a_remap() {
    const TEST: &str = "dma-buf residency remap";
    on_device(TEST, |device| {
        let page = host_page();
        let len = 2 * page;
        let queue = device.new_command_queue().unwrap();
        let ram = Ram::new(8 * page, 0, len);
        let old = counting(len, 13);
        let new = counting(len, 131);
        ram.write_file(0, &old);
        ram.write_file(4 * page, &new);
        let calls = Arc::new(AtomicUsize::new(0));
        let (buffer, chunks) = no_copy(TEST, device, &ram, (0, len), &calls);
        assert!(read(&queue, &buffer, len) == old, "{TEST}");
        if chunks == 0 {
            return;
        }
        release(TEST, device, 1);

        ram.remap(4 * page);
        device.host_range_remapped(ram.pointer(0), len as u64);
        assert!(ram.view() == new, "{TEST}: the CPU's view");
        assert!(
            read(&queue, &buffer, len) == old,
            "{TEST}: the buffer made before the remap"
        );
        let counts = device.work_counts();
        assert_eq!(
            residency(&counts),
            (1, 1, 1, chunks),
            "{TEST}: imported again"
        );
        assert_eq!(
            calls.load(Ordering::Acquire),
            1,
            "{TEST}: the source's calls"
        );

        let later_calls = Arc::new(AtomicUsize::new(0));
        let (later, _) = no_copy(TEST, device, &ram, (0, len), &later_calls);
        assert!(
            read(&queue, &later, len) == new,
            "{TEST}: a buffer made after the remap"
        );
        assert_eq!(later_calls.load(Ordering::Acquire), 1, "{TEST}");
        drop(later);
        drop(buffer);
        eprintln!(
            "{TEST}: on {}: the released import read its own pages after the remap",
            device.name()
        );
    });
}

/// A no-copy buffer over a page, where the device imports dma-bufs of it, read by a blit, then
/// left idle: the device's residency thread releases it no sooner than its first window after
/// that blit, with no call to the hook. The next blit imports it again, which doubles the
/// window: left idle again, the thread releases it no sooner than twice that after the blit.
#[test]
fn the_residency_thread_releases_an_idle_import_after_its_window() {
    const TEST: &str = "dma-buf residency thread";
    on_device(TEST, |device| {
        let page = host_page();
        let queue = device.new_command_queue().unwrap();
        let ram = Ram::new(page, 0, page);
        let bytes = counting(page, 3);
        ram.write_file(0, &bytes);
        let calls = Arc::new(AtomicUsize::new(0));
        let (buffer, chunks) = no_copy(TEST, device, &ram, (0, page), &calls);
        if chunks == 0 {
            assert!(read(&queue, &buffer, page) == bytes, "{TEST}");
            return;
        }
        for (round, window) in [(1, FIRST_WINDOW), (2, 2 * FIRST_WINDOW)] {
            let start = Instant::now();
            assert!(read(&queue, &buffer, page) == bytes, "{TEST}");
            let counts = counts_once(TEST, device, "released by the thread", |counts| {
                counts.host_imports_released == round
            });
            let after = start.elapsed();
            assert!(after >= window, "{TEST}: released after {after:?}");
            assert_eq!(residency(&counts), (round, round - 1, 0, chunks), "{TEST}");
            eprintln!(
                "{TEST}: on {}: released by the thread {after:?} after a blit",
                device.name()
            );
        }
        drop(buffer);
    });
}

/// With the device's kept descriptors capped at one (`Device::cap_kept_host_import_fds`), no-copy
/// buffers over the file's pages 0 and 1, of one dma-buf each where the device imports dma-bufs:
/// the first keeps its descriptor; the second, past the cap, is made and imported as any other and
/// keeps none, so it stays imported, counted (`WorkCounts::host_imports_kept_for_fds`): the hook
/// releases the first alone, blits read both buffers' bytes, and the second is never imported
/// again. Once the first is dropped its descriptor is closed, and a third buffer keeps one again.
/// Once the second is dropped, the count falls back.
#[test]
fn a_buffer_past_the_kept_descriptors_stays_imported() {
    const TEST: &str = "dma-buf residency past the cap";
    on_device(TEST, |device| {
        device.cap_kept_host_import_fds(1);
        let page = host_page();
        let queue = device.new_command_queue().unwrap();
        let ram = Ram::new(2 * page, 0, 2 * page);
        let bytes = counting(2 * page, 17);
        ram.write_file(0, &bytes);
        let (page_0, page_1) = (&bytes[..page], &bytes[page..]);
        let calls = Arc::new(AtomicUsize::new(0));
        let (kept, chunks) = no_copy(TEST, device, &ram, (0, page), &calls);
        if chunks == 0 {
            assert!(read(&queue, &kept, page) == page_0, "{TEST}");
            return;
        }
        assert_eq!(chunks, 1, "{TEST}: one dma-buf a page");
        assert_eq!(
            device.work_counts().host_imports_kept_for_fds,
            0,
            "{TEST}: the first kept"
        );
        let (past, _) = no_copy(TEST, device, &ram, (page, page), &calls);
        let counts = device.work_counts();
        assert_eq!(
            residency(&counts),
            (0, 0, 1, 1),
            "{TEST}: the second past the cap"
        );
        assert_eq!(
            counts.host_imports_kept_for_fds, 1,
            "{TEST}: the second kept imported for the cap"
        );
        assert!(read(&queue, &kept, page) == page_0, "{TEST}");
        assert!(read(&queue, &past, page) == page_1, "{TEST}");

        release(TEST, device, 1);
        assert_eq!(
            device.release_idle_host_imports(),
            0,
            "{TEST}: past the cap"
        );
        assert!(read(&queue, &past, page) == page_1, "{TEST}");
        assert!(read(&queue, &kept, page) == page_0, "{TEST}");
        let counts = device.work_counts();
        assert_eq!(residency(&counts), (1, 1, 1, 1), "{TEST}: the first again");

        drop(kept);
        counts_once(TEST, device, "the first's descriptor closed", |counts| {
            counts.host_import_fds_kept == 0
        });
        let (third, _) = no_copy(TEST, device, &ram, (0, page), &calls);
        let counts = device.work_counts();
        assert_eq!(residency(&counts), (1, 1, 1, 1), "{TEST}: the third kept");
        assert_eq!(
            counts.host_imports_kept_for_fds, 1,
            "{TEST}: the second still kept imported"
        );
        assert!(read(&queue, &third, page) == page_0, "{TEST}");
        drop(third);
        drop(past);
        counts_once(TEST, device, "the second's import freed", |counts| {
            counts.host_imports_kept_for_fds == 0
        });
        eprintln!(
            "{TEST}: on {}: the buffer past the cap stayed imported",
            device.name()
        );
    });
}
