//! Device tests of no-copy imports whose pointer the device refuses, made through dma-bufs of
//! their pages (`HostMemory::with_dma_buf`), on the device the cfg selects, under the validation
//! layer (each test fails on any message the layer reports).
//!
//! The memory is guest RAM as QEMU's `memory-backend-memfd,share=on` makes it: a `memfd` sealed
//! against growing and shrinking (`F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL`) and mapped shared,
//! with views that map pieces of the file contiguously, as a consumer gathers a guest object's
//! pages. The CPU writes and reads through the file's own whole mapping, the GPU through the
//! buffer over a view. Each test makes its dma-bufs itself, `udmabuf`s of the file pieces behind
//! the pages the runtime asks for (`/dev/udmabuf`, `UDMABUF_CREATE_LIST`), split where the request
//! allows to keep each within the host's udmabuf limits, which the tests read and never change,
//! and padded, where several chunks are made, with a scratch page of the file's that no view maps,
//! as a consumer does: the runtime runs no udmabuf code.
//!
//! RADV refuses the pointer of such memory and imports the chunks, which the runtime counts
//! (`WorkCounts::dma_buf_imports`, and `dma_buf_sparse_buffers` for a buffer of several). A
//! texture over a no-copy buffer of them is a shadow of its bytes, as over any host pages, and
//! asks the source for nothing; a texture placed in a no-copy heap over them is an image of
//! `DRM_FORMAT_MOD_LINEAR` over a dma-buf of its own pages, which it asks the source for too
//! (`dma_buf_images`), or its shadow where it makes none (`dma_buf_image_fallbacks`). NVIDIA and
//! lavapipe import the pointer, never call the source and count nothing. A driver not measured
//! has its answers printed. Where `/dev/udmabuf` cannot be opened each test says so and passes
//! without running.

use super::degrade_notices;
use super::first_use::{read_mark, release, scheduled, Busy, ENDED};
use super::first_use_writes::CHECK_WAIT;
use super::host_memory::READ_ONLY_PAGES;
use super::host_memory::{
    counting, gpu_fill, gpu_read, host_page, not_importable, on_device, run, Mapping,
};
use super::in_place_textures::refused;
use super::keep_alives::eventually;
use super::no_copy_heaps::{
    blit_into, blit_texels, encode_texels_in, encode_texels_out, shared, staged,
};
use super::ycbcr_in_place::{reads_in_place, sampler, texels, ycbcr_work, Kernels, Surface};
use super::{
    layer_reported_nothing, layer_reported_nothing_at_all, read_buffer, validated_device,
    validated_device_with, Messages, ScratchCache,
};
use crate::gpu::TEST_DEVICE;
use metal2vulkan::bind::{YCbCrMatrix, YCbCrRange};
use metal2vulkan::device::HostPageTexturePolicy;
use metal2vulkan::metal::{PixelFormat, SamplerMinMagFilter, TextureUsage};
use metal2vulkan::vk::DriverId;
use metal2vulkan_runtime::{
    BlitOption, Buffer, BufferContents, BufferDescriptor, CommandBuffer, CommandQueue,
    DebugMessage, Degradation, Device, DmaBufChunk, DmaBufRequest, DmaBufSource, Error, EventRef,
    Heap, HeapDescriptor, HeapType, HostMemory, KeepAlive, Origin, Problem, ResourceOptions,
    RuntimeGap, Size, TextureDescriptor,
};
use std::os::fd::{AsFd, AsRawFd, BorrowedFd, FromRawFd, OwnedFd};
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// `struct udmabuf_create_item` of Linux's `include/uapi/linux/udmabuf.h`.
#[repr(C)]
struct CreateItem {
    memfd: u32,
    pad: u32,
    offset: u64,
    size: u64,
}

/// `struct udmabuf_create_list` of the same header, its items following it.
#[repr(C)]
struct CreateList {
    flags: u32,
    count: u32,
}

const _: () = assert!(size_of::<CreateItem>() == 24 && size_of::<CreateList>() == 8);

/// `UDMABUF_FLAGS_CLOEXEC`.
const UDMABUF_FLAGS_CLOEXEC: u32 = 0x01;

/// `UDMABUF_CREATE_LIST`, `_IOW('u', 0x43, struct udmabuf_create_list)` in the generic ioctl
/// encoding.
const UDMABUF_CREATE_LIST: u64 =
    (1 << 30) | ((size_of::<CreateList>() as u64) << 16) | ((b'u' as u64) << 8) | 0x43;

/// The udmabuf device.
const UDMABUF: &str = "/dev/udmabuf";

/// Where the kernel states udmabuf's limits.
const PARAMETERS: &str = "/sys/module/udmabuf/parameters";

/// udmabuf's limits on this host, read (never changed): the most items one list takes
/// (`list_limit`, 1024 by default) and the most bytes one dma-buf holds (`size_limit_mb`, 64 MiB
/// by default). The defaults where the module states none.
#[derive(Clone, Copy, Debug)]
struct Limits {
    items: usize,
    bytes: u64,
}

impl Limits {
    fn of_host() -> Limits {
        let read = |name: &str| -> Option<u64> {
            std::fs::read_to_string(format!("{PARAMETERS}/{name}"))
                .ok()?
                .trim()
                .parse()
                .ok()
        };
        Limits {
            items: read("list_limit").map_or(1024, |items| items as usize),
            bytes: read("size_limit_mb").unwrap_or(64) << 20,
        }
    }
}

/// A dma-buf of the pieces `(offset, size)` of `memfd`, in order (`UDMABUF_CREATE_LIST`).
fn udmabuf(memfd: BorrowedFd<'_>, pieces: &[(u64, u64)]) -> std::io::Result<OwnedFd> {
    let device = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(UDMABUF)?;
    let count = u32::try_from(pieces.len()).map_err(std::io::Error::other)?;
    let memfd = u32::try_from(memfd.as_raw_fd()).map_err(std::io::Error::other)?;
    // The list's head, then its items, in words aligned for both.
    let mut words = vec![0u64; 1 + 3 * pieces.len()];
    // SAFETY: `words` holds the head and `count` items, each at an 8-byte-aligned word.
    unsafe {
        let head = words.as_mut_ptr().cast::<CreateList>();
        head.write(CreateList {
            flags: UDMABUF_FLAGS_CLOEXEC,
            count,
        });
        let items = words.as_mut_ptr().add(1).cast::<CreateItem>();
        for (index, &(offset, size)) in pieces.iter().enumerate() {
            items.add(index).write(CreateItem {
                memfd,
                pad: 0,
                offset,
                size,
            });
        }
    }
    // SAFETY: the list lives for the call, and the kernel reads its head and `count` items.
    let fd = unsafe { libc::ioctl(device.as_raw_fd(), UDMABUF_CREATE_LIST as _, words.as_ptr()) };
    if fd < 0 {
        return Err(std::io::Error::last_os_error());
    }
    // SAFETY: the ioctl returned a new descriptor, this caller's.
    Ok(unsafe { OwnedFd::from_raw_fd(fd) })
}

/// Whether this process can make a udmabuf; printed, as the test's skip, where it cannot.
pub(super) fn udmabuf_openable(test: &str) -> bool {
    match std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(UDMABUF)
    {
        Ok(_) => true,
        Err(error) => {
            eprintln!("{test}: SKIPPED: {UDMABUF} cannot be opened ({error})");
            false
        }
    }
}

/// Guest RAM: a sealed `memfd` of `len` bytes and one scratch page past them, which no view maps
/// and which pads a source's last chunk where it makes several; and the file's own whole shared
/// mapping of the `len` bytes.
pub(super) struct Memfd {
    fd: OwnedFd,
    len: usize,
    pub(super) whole: View,
}

/// What a test's source saw: each request, with the number of chunks it answered (0 for an
/// `Err`).
pub(super) type Calls = Arc<Mutex<Vec<(DmaBufRequest, usize)>>>;

impl Memfd {
    pub(super) fn new(len: usize) -> Memfd {
        // SAFETY: a new anonymous file, owned at once.
        let fd = unsafe {
            let fd = libc::memfd_create(
                c"m2v-guest-ram".as_ptr(),
                libc::MFD_CLOEXEC | libc::MFD_ALLOW_SEALING,
            );
            assert!(fd >= 0, "memfd_create: {}", std::io::Error::last_os_error());
            OwnedFd::from_raw_fd(fd)
        };
        let file = len + host_page();
        // SAFETY: the descriptor is this object's.
        unsafe {
            assert_eq!(libc::ftruncate(fd.as_raw_fd(), file as libc::off_t), 0);
            let seals = libc::F_SEAL_GROW | libc::F_SEAL_SHRINK | libc::F_SEAL_SEAL;
            assert_eq!(libc::fcntl(fd.as_raw_fd(), libc::F_ADD_SEALS, seals), 0);
        }
        let whole = View::new(fd.as_fd(), &[(0, len)]);
        Memfd { fd, len, whole }
    }

    /// A view of the file's `pieces` `(offset, size)`, mapped contiguously in that order.
    pub(super) fn view(&self, pieces: &[(usize, usize)]) -> View {
        assert!(pieces
            .iter()
            .all(|&(offset, size)| offset + size <= self.len));
        View::new(self.fd.as_fd(), pieces)
    }

    /// A source of dma-bufs of `view`'s pages, as a consumer makes them: the file pieces behind
    /// the request, split at multiples of its granule into as few chunks as keep each within the
    /// host's udmabuf limits (an `Err` where one granule alone exceeds them), the last of several
    /// padded with the scratch page. It records each request in `calls`.
    pub(super) fn source(&self, view: &View, calls: &Calls) -> DmaBufSource {
        self.source_split(view, calls, false)
    }

    /// As [`Memfd::source`], but one chunk per granule wherever the request allows several.
    fn granules(&self, view: &View, calls: &Calls) -> DmaBufSource {
        self.source_split(view, calls, true)
    }

    /// [`Memfd::source`], or [`Memfd::granules`] where `per_granule`.
    fn source_split(&self, view: &View, calls: &Calls, per_granule: bool) -> DmaBufSource {
        let fd = Arc::new(self.fd.try_clone().unwrap());
        let base = view.at.as_ptr() as usize;
        let pieces = view.pieces.clone();
        let calls = Arc::clone(calls);
        let scratch = self.len as u64;
        let limits = Limits::of_host();
        Arc::new(move |request: DmaBufRequest| {
            let made = (|| {
                let at = (request.start.as_ptr() as usize)
                    .checked_sub(base)
                    .ok_or_else(|| std::io::Error::other("pages before the view"))?;
                let limits = match per_granule {
                    true => Limits {
                        bytes: limits.bytes.min(request.granule),
                        ..limits
                    },
                    false => limits,
                };
                let plan = chunk_plan(&pieces, at, &request, limits, scratch)?;
                plan.into_iter()
                    .map(|(items, length)| {
                        Ok(DmaBufChunk {
                            fd: udmabuf(fd.as_fd(), &items)?,
                            length,
                        })
                    })
                    .collect::<std::io::Result<Vec<DmaBufChunk>>>()
            })();
            let chunks = made.as_ref().map_or(0, Vec::len);
            calls.lock().unwrap().push((request, chunks));
            made
        })
    }
}

/// One chunk a consumer plans: its file pieces `(offset, size)` and its length.
type PlannedChunk = (Vec<(u64, u64)>, u64);

/// The chunks a consumer answers `request` with over a view of `pieces` whose first byte the
/// request starts `at`: each its file pieces and its length, split only at multiples of the
/// request's granule, as few as keep each within `limits`; where there are several, the last is
/// padded to a multiple of the granule with the page at file offset `scratch`, repeated. An `Err`
/// where one granule's pages alone exceed the limits, or lie past the view.
fn chunk_plan(
    pieces: &[(usize, usize)],
    at: usize,
    request: &DmaBufRequest,
    limits: Limits,
    scratch: u64,
) -> std::io::Result<Vec<PlannedChunk>> {
    let (length, granule) = (request.length, request.granule.max(1));
    let mut chunks: Vec<PlannedChunk> = Vec::new();
    let mut current: PlannedChunk = (Vec::new(), 0);
    let mut from = 0u64;
    let too_big = |unit: u64, items: usize| {
        std::io::Error::other(format!(
            "a granule of {unit} bytes in {items} pieces exceeds udmabuf's limits {limits:?}"
        ))
    };
    while from < length {
        let unit = granule.min(length - from);
        let items = file_pieces(pieces, at + from as usize, unit as usize)
            .ok_or_else(|| std::io::Error::other("pages past the view"))?;
        let mut joined = current.0.clone();
        for &(offset, size) in &items {
            match joined.last_mut() {
                Some((last, last_size)) if *last + *last_size == offset => *last_size += size,
                _ => joined.push((offset, size)),
            }
        }
        if joined.len() <= limits.items && current.1 + unit <= limits.bytes {
            current = (joined, current.1 + unit);
        } else if current.0.is_empty() || items.len() > limits.items || unit > limits.bytes {
            return Err(too_big(unit, items.len()));
        } else {
            chunks.push(std::mem::replace(&mut current, (items, unit)));
        }
        from += unit;
    }
    chunks.push(current);
    if chunks.len() > 1 {
        let page = host_page() as u64;
        let last = chunks.last_mut().expect("several");
        while !last.1.is_multiple_of(granule) {
            last.0.push((scratch, page));
            last.1 += page;
        }
        if last.0.len() > limits.items || last.1 > limits.bytes {
            return Err(too_big(last.1, last.0.len()));
        }
    }
    Ok(chunks)
}

/// The file pieces behind `len` bytes from `at` of a view of `pieces` (in view order), adjacent
/// ones merged; `None` past the view.
fn file_pieces(pieces: &[(usize, usize)], at: usize, len: usize) -> Option<Vec<(u64, u64)>> {
    let mut out: Vec<(u64, u64)> = Vec::new();
    let (mut from, end) = (at, at.checked_add(len)?);
    let mut piece_start = 0;
    for &(offset, size) in pieces {
        let piece_end = piece_start + size;
        if from < piece_end && from < end {
            let take = piece_end.min(end) - from;
            let file = (offset + from - piece_start) as u64;
            match out.last_mut() {
                Some((last, last_size)) if *last + *last_size == file => *last_size += take as u64,
                _ => out.push((file, take as u64)),
            }
            from += take;
        }
        piece_start = piece_end;
    }
    (from == end).then_some(out)
}

/// Pieces of a file mapped shared and contiguously at one address, unmapped when dropped.
pub(super) struct View {
    at: NonNull<u8>,
    len: usize,
    pieces: Vec<(usize, usize)>,
}

// SAFETY: the view is plain memory of this process; each test orders the CPU's and the GPU's
// accesses.
unsafe impl Send for View {}

impl View {
    fn new(fd: BorrowedFd<'_>, pieces: &[(usize, usize)]) -> View {
        let len: usize = pieces.iter().map(|&(_, size)| size).sum();
        // SAFETY: a fresh reservation, then fixed shared mappings of the file inside it.
        unsafe {
            let reservation = libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_NONE,
                libc::MAP_PRIVATE | libc::MAP_ANON,
                -1,
                0,
            );
            assert_ne!(reservation, libc::MAP_FAILED, "mmap reservation");
            let mut at = 0;
            for &(offset, size) in pieces {
                let wanted = reservation.cast::<u8>().add(at).cast();
                let mapped = libc::mmap(
                    wanted,
                    size,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_SHARED | libc::MAP_FIXED,
                    fd.as_raw_fd(),
                    offset as libc::off_t,
                );
                assert_eq!(
                    mapped,
                    wanted,
                    "mmap of a piece: {}",
                    std::io::Error::last_os_error()
                );
                at += size;
            }
            View {
                at: NonNull::new(reservation.cast()).unwrap(),
                len,
                pieces: pieces.to_vec(),
            }
        }
    }

    /// The view's address `offset` bytes in.
    pub(super) fn pointer(&self, offset: usize) -> NonNull<u8> {
        assert!(offset < self.len);
        // SAFETY: inside the view.
        unsafe { NonNull::new_unchecked(self.at.as_ptr().add(offset)) }
    }

    /// A copy of the view's bytes `range`.
    pub(super) fn read(&self, range: std::ops::Range<usize>) -> Vec<u8> {
        assert!(range.start <= range.end && range.end <= self.len);
        // SAFETY: the range lies in the view; each test orders the accesses.
        unsafe {
            std::slice::from_raw_parts(self.at.as_ptr().add(range.start), range.len()).to_vec()
        }
    }

    /// Map the whole view read-only.
    fn protect_read_only(&self) {
        // SAFETY: the mappings are this view's.
        let done = unsafe { libc::mprotect(self.at.as_ptr().cast(), self.len, libc::PROT_READ) };
        assert_eq!(done, 0, "mprotect: {}", std::io::Error::last_os_error());
    }

    /// Write `bytes` into the view at `offset`.
    pub(super) fn write(&self, offset: usize, bytes: &[u8]) {
        assert!(offset + bytes.len() <= self.len);
        // SAFETY: as in `read`.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.at.as_ptr().add(offset), bytes.len())
        };
    }
}

impl Drop for View {
    fn drop(&mut self) {
        // SAFETY: the mappings are this view's, and no buffer uses them any more (each test drops
        // its buffers first).
        unsafe { libc::munmap(self.at.as_ptr().cast(), self.len) };
    }
}

/// What `device` does with a memfd's pages: `Some(true)` where it refuses their pointer and
/// imports dma-bufs (RADV), `Some(false)` where it imports the pointer (NVIDIA, lavapipe); `None`
/// for a driver not measured, whose answers are printed only.
fn imports_dma_buf(device: &Device) -> Option<bool> {
    let driver = device.properties().description.driver_id;
    if driver == Some(DriverId::MESA_RADV) {
        Some(true)
    } else if driver == Some(DriverId::NVIDIA_PROPRIETARY)
        || driver == Some(DriverId::MESA_LLVMPIPE)
    {
        Some(false)
    } else {
        None
    }
}

/// The device's counts of chunks imported and of sparse buffers assembled so far.
fn dma_buf_counts(device: &Device) -> (u64, u64) {
    let counts = device.work_counts();
    (counts.dma_buf_imports, counts.dma_buf_sparse_buffers)
}

/// Whether `error` refuses dma-bufs that break the source's contract (`Invalid`, `DmaBuf`).
fn misuse(error: &Error) -> bool {
    matches!(error, Error::Invalid(invalid) if invalid.problem == Problem::DmaBuf)
}

/// Check what making a no-copy object over a memfd view with a source did, against the device's
/// expected answer: where it imports dma-bufs, the source asked once for `pages` (the first byte
/// and length of the pages) at a granule that is their length or a multiple of the host page
/// below it, and each chunk it answered imported and counted, a buffer of several counted as
/// assembled, and `chunks` of them where given; else no call and no count. `before` is the
/// counts before the call. The chunks answered.
fn check_route(
    test: &str,
    device: &Device,
    calls: &Calls,
    pages: (NonNull<u8>, u64),
    before: (u64, u64),
    chunks: Option<usize>,
) -> usize {
    let calls = calls.lock().unwrap().clone();
    let (imports, sparse) = dma_buf_counts(device);
    let counted = (imports - before.0, sparse - before.1);
    let answered = calls.first().map_or(0, |&(_, chunks)| chunks);
    match imports_dma_buf(device) {
        Some(true) => {
            assert_eq!(calls.len(), 1, "{test}: the source's calls: {calls:?}");
            let (request, _) = calls[0];
            assert_eq!(
                (request.start, request.length),
                pages,
                "{test}: {request:?}"
            );
            let page = host_page() as u64;
            assert!(
                request.granule == request.length
                    || (request.granule < request.length && request.granule.is_multiple_of(page)),
                "{test}: {request:?}"
            );
            if let Some(chunks) = chunks {
                assert_eq!(answered, chunks, "{test}: chunks answered");
            }
            let assembled = u64::from(answered > 1);
            assert_eq!(counted, (answered as u64, assembled), "{test}: counted");
        }
        Some(false) => {
            assert!(calls.is_empty(), "{test}: the source was called: {calls:?}");
            assert_eq!(counted, (0, 0), "{test}: counted");
        }
        None => {}
    }
    let name = device.name();
    eprintln!("{test}: on {name}: chunks and sparse buffers {counted:?}, source calls {calls:?}");
    answered
}

/// A no-copy buffer over `length` bytes of `view` from `offset`, with `source`.
pub(super) fn no_copy(
    device: &Device,
    view: &View,
    offset: usize,
    length: usize,
    source: DmaBufSource,
) -> Result<Buffer, Error> {
    // SAFETY: each test's view outlives its buffers, and the source makes dma-bufs of its pages.
    unsafe {
        device.new_buffer_with_bytes_no_copy(
            HostMemory::with_dma_buf(view.pointer(offset), source),
            length as u64,
            ResourceOptions::default(),
            None,
        )
    }
}

/// A no-copy buffer over a scattered memfd view, two discontiguous file pieces mapped contiguously
/// in the other order (file pages 5 and 6, then 1 and 2), as a consumer gathers a guest object, at
/// a pointer and a length that are not page-aligned: where the device refuses the pointer (RADV),
/// the source is called once with the pages the range touches (no import-alignment rounding), its
/// one chunk's list holding both pieces in view order, imported and counted; elsewhere the pointer
/// is imported and the source dropped uncalled. Either way `contents` is the pointer, the GPU reads
/// the CPU's writes through the file's own mapping, a GPU fill across the seam lands at the right
/// file offsets on either side, and the CPU's later writes reach the GPU, with no map, flush or
/// invalidate.
#[test]
fn a_scattered_view_is_imported_across_its_pieces() {
    const TEST: &str = "dma-buf scattered view";
    if !udmabuf_openable(TEST) {
        return;
    }
    on_device(|device| {
        let page = host_page();
        let queue = device.new_command_queue().unwrap();
        let ram = Memfd::new(8 * page);
        let pieces = [(5 * page, 2 * page), (page, 2 * page)];
        assert_eq!(
            file_pieces(&pieces, 0, 4 * page),
            Some(vec![
                (5 * page as u64, 2 * page as u64),
                (page as u64, 2 * page as u64)
            ])
        );
        let view = ram.view(&pieces);
        // The CPU writes the view's bytes `bytes` through the file's own mapping.
        let write = |bytes: &[u8]| {
            ram.whole.write(5 * page, &bytes[..2 * page]);
            ram.whole.write(page, &bytes[2 * page..]);
        };
        let earlier = counting(4 * page, 3);
        write(&earlier);
        // From 100 bytes into the view to 64 bytes before its end.
        let (offset, length) = (100, 4 * page - 164);
        let calls = Calls::default();
        let before = dma_buf_counts(device);
        let buffer = no_copy(device, &view, offset, length, ram.source(&view, &calls)).unwrap();
        let pages = (view.at, 4 * page as u64);
        check_route(TEST, device, &calls, pages, before, Some(1));
        assert_eq!(buffer.contents(), Some(view.pointer(offset)));
        let read = gpu_read(&queue, &buffer, 0..length as u64);
        assert!(
            read == earlier[offset..offset + length],
            "{TEST}: the GPU's read"
        );
        // A fill of 100 bytes either side of the seam.
        let seam = 2 * page;
        let fill = (seam - offset - 100) as u64..(seam - offset + 100) as u64;
        gpu_fill(&queue, &buffer, fill, 0xc3);
        let end_of_first = ram.whole.read(7 * page - 100..7 * page);
        let start_of_second = ram.whole.read(page..page + 100);
        assert!(end_of_first.iter().all(|&byte| byte == 0xc3));
        assert!(start_of_second.iter().all(|&byte| byte == 0xc3));
        assert_eq!(
            ram.whole.read(7 * page - 101..7 * page - 100),
            [earlier[seam - 101]]
        );
        assert_eq!(
            ram.whole.read(page + 100..page + 101),
            [earlier[seam + 100]]
        );
        assert_eq!(view.read(seam - 100..seam + 100), vec![0xc3; 200]);
        // And the CPU's later writes reach the GPU without a flush.
        let later = counting(4 * page, 97);
        write(&later);
        let read = gpu_read(&queue, &buffer, 0..length as u64);
        assert!(
            read == later[offset..offset + length],
            "{TEST}: the CPU's later bytes"
        );
        drop(buffer);
    });
}

/// `runs` single pages of a new memfd, every other page of the file so that no two are adjacent,
/// mapped as one view, each page's bytes counting from its run: the memfd, the view and its bytes.
fn every_other_page(runs: usize) -> (Memfd, View, Vec<u8>) {
    let page = host_page();
    let ram = Memfd::new(2 * runs * page);
    let pieces: Vec<(usize, usize)> = (0..runs).map(|run| (2 * run * page, page)).collect();
    let view = ram.view(&pieces);
    let mut bytes = Vec::with_capacity(runs * page);
    for run in 0..runs {
        let run_bytes = counting(page, run);
        ram.whole.write(2 * run * page, &run_bytes);
        bytes.extend(run_bytes);
    }
    (ram, view, bytes)
}

/// [`every_other_page`]'s `runs` pages; a no-copy buffer over all but the view's first 64 and
/// last 64 bytes, made through `source`, read and filled by the GPU and checked at every page's
/// file offset, and the page after each run left alone. The making's refusal, where it is refused.
fn check_runs(test: &str, device: &Device, runs: usize) -> Result<(), Error> {
    let page = host_page();
    let queue = device.new_command_queue().unwrap();
    let (ram, view, expected) = every_other_page(runs);
    let (offset, length) = (64, runs * page - 128);
    let calls = Calls::default();
    let before = dma_buf_counts(device);
    let buffer = no_copy(device, &view, offset, length, ram.source(&view, &calls))?;
    check_route(
        test,
        device,
        &calls,
        (view.at, (runs * page) as u64),
        before,
        None,
    );
    assert!(
        gpu_read(&queue, &buffer, 0..length as u64) == expected[offset..offset + length],
        "{test}: the GPU's read"
    );
    gpu_fill(&queue, &buffer, 0..length as u64, 0x6e);
    for run in 0..runs {
        let file = ram.whole.read(2 * run * page..2 * run * page + page);
        let (from, to) = match run {
            0 => (offset, page),
            _ if run == runs - 1 => (0, page - 64),
            _ => (0, page),
        };
        assert!(
            file[from..to].iter().all(|&byte| byte == 0x6e),
            "{test}: run {run}"
        );
        assert_eq!(
            file[..from],
            expected[run * page..run * page + from],
            "{test}: run {run}"
        );
        let skipped = ram
            .whole
            .read((2 * run + 1) * page..(2 * run + 1) * page + 1);
        assert_eq!(skipped, [0], "{test}: the page after run {run}");
    }
    drop(buffer);
    Ok(())
}

/// A view of more discontiguous runs than one udmabuf takes on this host (its `list_limit`, 1024
/// by default, read and never changed): `list_limit + 76` single pages. Where the device refuses
/// the pointer and binds sparse buffers of dma-bufs (RADV), the runtime asks for chunks at its
/// granule, the source answers two (the second padded with its scratch page to the granule), and
/// the buffer is assembled from them, counted; the GPU and the CPU see each other's bytes at
/// every page's file offset. Elsewhere the pointer is imported, the source uncalled.
#[test]
fn a_view_past_one_udmabuf_s_limits_is_assembled_from_chunks() {
    const TEST: &str = "dma-buf many runs";
    if !udmabuf_openable(TEST) {
        return;
    }
    on_device(|device| {
        let runs = Limits::of_host().items + 76;
        match (imports_dma_buf(device), check_runs(TEST, device, runs)) {
            (_, Ok(())) => {}
            (None, Err(error)) => eprintln!("{TEST}: on {}: {error:?}", device.name()),
            (Some(_), Err(error)) => panic!("{TEST}: {error:?}"),
        }
    });
}

/// A `YCBCR8_420_2P` texture over a no-copy buffer of more discontiguous pages than one udmabuf
/// takes (`list_limit + 76` single pages, every other page of the file, each row of both planes a
/// page of its own). Where the device refuses the pointer and binds sparse buffers of dma-bufs
/// (RADV), the buffer is assembled from chunks, counted, and the planes cross from one chunk into
/// the next; where its memory is the host's (an integrated GPU, a CPU device), a kernel's samples
/// read the planes in place across them (`WorkCounts::ycbcr_planes_bound`), converting nothing.
/// Every nearest sample at a texel's centre is Metal's texel of the codes, on every device.
#[test]
fn a_ycbcr_texture_over_a_buffer_of_chunks_reads_its_planes_in_place() {
    const TEST: &str = "dma-buf ycbcr planes";
    if !udmabuf_openable(TEST) {
        return;
    }
    on_device(|device| {
        let page = host_page();
        let runs = Limits::of_host().items + 76;
        // Two luma rows a chroma row, every row its own page.
        let rows = 2 * (runs / 3);
        let surface = Surface {
            format: PixelFormat::YCBCR84202P,
            ostype: *b"420v",
            matrix: None,
            conversion: (YCbCrRange::Video, YCbCrMatrix::Bt601),
            size: [512, rows as u64],
            row: page as u64,
            chroma_at: (rows * page) as u64,
        };
        assert!(surface.length() as usize <= runs * page);
        let ram = Memfd::new(2 * runs * page);
        let pieces: Vec<(usize, usize)> = (0..runs).map(|run| (2 * run * page, page)).collect();
        let view = ram.view(&pieces);
        let bytes = surface.bytes(11);
        for (run, piece) in bytes.chunks(page).enumerate() {
            ram.whole.write(2 * run * page, piece);
        }
        let calls = Calls::default();
        let before = dma_buf_counts(device);
        let buffer = match no_copy(device, &view, 0, runs * page, ram.source(&view, &calls)) {
            Ok(buffer) => buffer,
            Err(error) if imports_dma_buf(device).is_none() => {
                eprintln!("{TEST}: on {}: {error:?}", device.name());
                return;
            }
            Err(error) => panic!("{TEST}: {error:?}"),
        };
        let (imports, sparse) = dma_buf_counts(device);
        let chunks = calls.lock().unwrap().iter().map(|(_, n)| n).sum::<usize>();
        eprintln!(
            "{TEST} on {}: {} imports, {} sparse buffers, {chunks} chunks",
            device.name(),
            imports - before.0,
            sparse - before.1
        );
        if imports_dma_buf(device) == Some(true) {
            assert!(chunks >= 2, "{TEST}: the source's chunks");
            assert_eq!(sparse - before.1, 1, "{TEST}: a sparse buffer");
        }
        let in_place = reads_in_place(device);
        let texture = surface.texture(&buffer, 0).unwrap();
        let kernels = Kernels::new(device);
        let nearest = sampler(device, SamplerMinMagFilter::Nearest);
        let counted = device.work_counts();
        let got = texels(
            device,
            &kernels.sample,
            (&texture, surface.size),
            Some(&nearest),
        );
        let work = ycbcr_work(&counted, &device.work_counts());
        eprintln!(
            "{TEST} on {}: in place {in_place}, work {work:?}",
            device.name()
        );
        let model = surface.model(&bytes);
        let wrong = got.iter().zip(&model).filter(|(a, b)| a != b).count();
        assert_eq!(wrong, 0, "{TEST}: texels not Metal's");
        if in_place {
            assert_eq!(work[2..], [1, 0, 0], "{TEST}: the planes read in place");
        }
        drop(texture);
        drop(buffer);
    });
}

/// [`every_other_page`]'s 64 pages, each its own mapping, as a consumer's view of a surface's
/// scattered pages: a no-copy buffer over them all asks the kernel about each of the 64 mappings
/// for their write permission before its import (`WorkCounts::host_mappings_queried`), refused or
/// not. A second buffer over the same bytes, made while the first lives, lies in its import and
/// asks about none, its source uncalled, and the GPU reads the CPU's bytes through it. A buffer
/// over the first half of the bytes, not the same range, makes its own import and asks about its
/// 32 mappings.
#[test]
fn a_buffer_that_lies_in_a_live_import_asks_about_no_mapping() {
    const TEST: &str = "dma-buf joined import";
    const RUNS: usize = 64;
    if !udmabuf_openable(TEST) {
        return;
    }
    on_device(|device| {
        let page = host_page();
        let queue = device.new_command_queue().unwrap();
        let (ram, view, expected) = every_other_page(RUNS);
        let length = RUNS * page;
        let queried = || device.work_counts().host_mappings_queried;
        let before = queried();
        let calls = Calls::default();
        let first = no_copy(device, &view, 0, length, ram.source(&view, &calls));
        let asked = queried() - before;
        assert_eq!(
            asked, RUNS as u64,
            "{TEST}: the first buffer's mappings asked"
        );
        let first = match (imports_dma_buf(device), first) {
            (_, Ok(first)) => first,
            (None, Err(error)) => {
                eprintln!("{TEST}: on {}: {error:?}", device.name());
                return;
            }
            (Some(_), Err(error)) => panic!("{TEST}: {error:?}"),
        };
        let before = queried();
        let joined = Calls::default();
        let second = no_copy(device, &view, 0, length, ram.source(&view, &joined)).unwrap();
        assert_eq!(
            queried() - before,
            0,
            "{TEST}: the second buffer's mappings asked"
        );
        let joined = joined.lock().unwrap().clone();
        assert!(joined.is_empty(), "{TEST}: the second's source: {joined:?}");
        assert!(
            gpu_read(&queue, &second, 0..length as u64) == expected,
            "{TEST}: the GPU's read through the second buffer"
        );
        let before = queried();
        let half = no_copy(
            device,
            &view,
            0,
            length / 2,
            ram.source(&view, &Calls::default()),
        );
        assert_eq!(
            queried() - before,
            RUNS as u64 / 2,
            "{TEST}: the half's mappings asked ({:?})",
            half.as_ref().err()
        );
        eprintln!(
            "{TEST}: on {}: {RUNS} mappings asked for the first buffer, none for the second",
            device.name()
        );
        drop(half);
        drop(second);
        drop(first);
    });
}

/// Memfd pages a view maps read-only, with a source: the process cannot write them, so no pointer
/// of them is imported (Linux drivers pin imported pages for writing), and the buffer is made from
/// the source alone on a device that imports dma-bufs, whatever it does with a pointer of such
/// memory (`ReadOnlyPagesWritten`, announced once for two such buffers over different pages). The
/// GPU reads the CPU's bytes through it, and its writes reach the file's pages, where the CPU
/// reads them through the file's own mapping (where Metal's leave the pages as they were). The
/// same pages with no source are refused typed, as before, and so is the source's memory on a
/// device that imports no dma-buf.
#[test]
fn read_only_pages_are_made_from_the_source() {
    const TEST: &str = "dma-buf read-only pages";
    if !udmabuf_openable(TEST) {
        return;
    }
    degrade_notices::on_device(|device, messages| {
        let page = host_page();
        let queue = device.new_command_queue().unwrap();
        let ram = Memfd::new(6 * page);
        let bytes = counting(6 * page, 9);
        ram.whole.write(0, &bytes);
        // Whether the device's route for these pages is a dma-buf import. A device that imports
        // the pointer instead (NVIDIA, lavapipe) has no working import for pages the process maps
        // read-only, which it refuses the pointer of: NVIDIA's 615.71.09 answers
        // `ERROR_OUT_OF_DEVICE_MEMORY` with the whole card free. Its answer is printed, not held
        // to a class, since it is the driver's.
        let route = imports_dma_buf(device) == Some(true);
        let read_only = |notices: &[DebugMessage]| {
            let kind = |m: &&DebugMessage| {
                (m.degradation.as_ref()).is_some_and(|n| READ_ONLY_PAGES.contains(&n.degradation))
            };
            notices.iter().filter(kind).count()
        };
        let mut made = Vec::new();
        for (pieces, at) in [([(page, 2 * page)], page), ([(4 * page, page)], 4 * page)] {
            let view = ram.view(&pieces);
            view.protect_read_only();
            let length = pieces[0].1;
            // SAFETY: the view outlives the buffer, and nothing is made of it.
            let plain = unsafe {
                device.new_buffer_with_bytes_no_copy(
                    HostMemory::new(view.pointer(0)),
                    length as u64,
                    ResourceOptions::default(),
                    None,
                )
            };
            let error = plain.unwrap_err();
            assert!(not_importable(&error), "{TEST}: no source: {error:?}");
            let calls = Calls::default();
            let before = dma_buf_counts(device);
            let buffer = no_copy(device, &view, 0, length, ram.source(&view, &calls));
            let buffer = match (route, buffer) {
                (true, Ok(buffer)) => buffer,
                (true, made) => panic!("{TEST}: the dma-buf route: {:?}", made.err()),
                (false, made) => {
                    eprintln!(
                        "{TEST}: on {}: the pages' route is not a dma-buf import: {:?}",
                        device.name(),
                        made.err()
                    );
                    return;
                }
            };
            let calls = calls.lock().unwrap().clone();
            assert_eq!(calls.len(), 1, "{TEST}: the source's calls: {calls:?}");
            assert!(
                dma_buf_counts(device).0 > before.0,
                "{TEST}: no chunk imported"
            );
            assert!(
                gpu_read(&queue, &buffer, 0..length as u64) == bytes[at..at + length],
                "{TEST}: the GPU's read"
            );
            gpu_fill(&queue, &buffer, 0..64, 0x5a);
            assert_eq!(
                ram.whole.read(at..at + 64),
                [0x5a; 64],
                "{TEST}: the GPU's write in the file's pages"
            );
            made.push((buffer, view));
        }
        assert_eq!(read_only(&messages.take_notices()), 1, "{TEST}: announced");
        eprintln!(
            "{TEST}: on {}: made from the source, the GPU's writes in the pages",
            device.name()
        );
        drop(made);
    });
}

/// A keep-alive that records, as it drops, the device's live dma-buf imports
/// (`WorkCounts::dma_buf_memories_live`).
struct LiveAtDrop {
    device: Device,
    seen: Arc<Mutex<Option<u64>>>,
}

impl Drop for LiveAtDrop {
    fn drop(&mut self) {
        let live = self.device.work_counts().dma_buf_memories_live;
        *self.seen.lock().unwrap() = Some(live);
    }
}

/// A no-copy buffer's keep-alive goes once the imports of its pages are freed, as
/// `HostMemory`'s contract has it (the pages must stay mapped until then): a buffer over two
/// pages the process maps read-only, made from the source's dma-bufs on a device whose route for
/// such pages is a dma-buf import ([`imports_dma_buf`]), with a keep-alive and no deallocator,
/// released; as the keep-alive drops, every import the buffer made is freed (the live dma-buf
/// imports are back to their count before it).
#[test]
fn a_keep_alive_goes_after_the_imports_of_its_pages() {
    const TEST: &str = "dma-buf keep-alive order";
    if !udmabuf_openable(TEST) {
        return;
    }
    degrade_notices::on_device(|device, _| {
        // As `read_only_pages_are_made_from_the_source`: only a device whose route for these
        // pages is a dma-buf import has imports to order the keep-alive against.
        if imports_dma_buf(device) != Some(true) {
            eprintln!(
                "{TEST}: on {}: the pages' route is not a dma-buf import",
                device.name()
            );
            return;
        }
        let page = host_page();
        let ram = Memfd::new(4 * page);
        let view = ram.view(&[(page, 2 * page)]);
        view.protect_read_only();
        let live = || device.work_counts().dma_buf_memories_live;
        let before = live();
        let seen = Arc::new(Mutex::new(None));
        let keep_alive: KeepAlive = Box::new(LiveAtDrop {
            device: device.clone(),
            seen: Arc::clone(&seen),
        });
        let calls = Calls::default();
        let source = ram.source(&view, &calls);
        // SAFETY: the view outlives the buffer and its keep-alive, and the source makes dma-bufs
        // of its pages.
        let memory = unsafe { HostMemory::with_dma_buf(view.pointer(0), source) };
        let contents = BufferContents::NoCopy {
            memory,
            deallocator: None,
            keep_alive: Some(keep_alive),
        };
        let length = (2 * page) as u64;
        let descriptor = BufferDescriptor::new(length, ResourceOptions::default(), contents);
        let buffer = device.new_buffer_with_descriptor(descriptor).unwrap();
        let imported = live() - before;
        assert!(imported > 0, "{TEST}: no import made: {calls:?}");
        drop(buffer);
        assert!(
            eventually(|| seen.lock().unwrap().is_some()),
            "{TEST}: the keep-alive never went"
        );
        let at_drop = seen.lock().unwrap().unwrap();
        assert_eq!(
            at_drop, before,
            "{TEST}: dma-buf imports live as the keep-alive went ({imported} made)"
        );
        eprintln!(
            "{TEST}: on {}: {imported} imports freed before the keep-alive went",
            device.name()
        );
    });
}

/// The refusals: memfd memory with no source is refused typed where the device refuses its
/// pointer (RADV), as before; a source that fails keeps that refusal, called once; chunks that do
/// not cover the pages, and a chunk whose descriptor is no dma-buf (the memfd itself), are the
/// caller's misuse (`Invalid`, `DmaBuf`), their descriptors closed. Nothing is counted for any of
/// them. Where the device imports the pointer, each is made and no source is called.
#[test]
fn shared_memory_is_refused_without_a_working_source() {
    const TEST: &str = "dma-buf refusals";
    on_device(|device| {
        let page = host_page();
        let ram = Memfd::new(4 * page);
        let view = ram.view(&[(page, 2 * page)]);
        let before = dma_buf_counts(device);
        let refuses = imports_dma_buf(device);
        let check =
            |what: &str, made: Result<Buffer, Error>, expected: fn(&Error) -> bool| match made {
                Ok(_) => assert_ne!(refuses, Some(true), "{TEST}: {what} was made"),
                Err(error) => {
                    assert_ne!(refuses, Some(false), "{TEST}: {what}: {error:?}");
                    assert!(expected(&error), "{TEST}: {what}: {error:?}");
                }
            };
        // SAFETY: the view outlives the buffer.
        let plain = unsafe {
            device.new_buffer_with_bytes_no_copy(
                HostMemory::new(view.pointer(0)),
                2 * page as u64,
                ResourceOptions::default(),
                None,
            )
        };
        check("no source", plain, not_importable);
        let calls = Arc::new(Mutex::new(0usize));
        let counted = Arc::clone(&calls);
        let failing: DmaBufSource = Arc::new(move |_| {
            *counted.lock().unwrap() += 1;
            Err(std::io::Error::other("no dma-buf"))
        });
        let failed = no_copy(device, &view, 0, 2 * page, failing);
        check("a failing source", failed, not_importable);
        let expected_calls = usize::from(refuses == Some(true));
        if refuses.is_some() {
            assert_eq!(*calls.lock().unwrap(), expected_calls, "{TEST}: calls");
        }
        if udmabuf_openable(TEST) {
            // One page of the two asked for.
            let fd = Arc::new(ram.fd.try_clone().unwrap());
            let short: DmaBufSource = Arc::new(move |_| {
                Ok(vec![DmaBufChunk {
                    fd: udmabuf(fd.as_fd(), &[(page as u64, page as u64)])?,
                    length: page as u64,
                }])
            });
            let made = no_copy(device, &view, 0, 2 * page, short);
            check("a chunk short of the pages", made, misuse);
        }
        // The memfd itself, which is no dma-buf.
        let fd = Arc::new(ram.fd.try_clone().unwrap());
        let file: DmaBufSource = Arc::new(move |request| {
            Ok(vec![DmaBufChunk {
                fd: fd.try_clone()?,
                length: request.length,
            }])
        });
        let made = no_copy(device, &view, 0, 2 * page, file);
        check("a chunk that is no dma-buf", made, misuse);
        assert_eq!(dma_buf_counts(device), before, "{TEST}: counted");
    });
}

/// The source runs with no lock of the runtime's held: one that makes a buffer of the device's
/// own and a no-copy buffer of private memory before it answers, for a no-copy buffer, a no-copy
/// heap and a texture placed in it (its image over a dma-buf of its own pages), and for a rebind
/// of a buffer onto memfd memory, returns, and each is made over its chunks (RADV) or over the
/// pointer, the source uncalled (elsewhere). A texture over the no-copy buffer, a shadow of its
/// bytes, asks it nothing. The rebound buffer's bytes are then the view's.
#[test]
fn the_source_runs_with_no_runtime_lock_held() {
    const TEST: &str = "dma-buf source calls the runtime";
    if !udmabuf_openable(TEST) {
        return;
    }
    on_device(|device| {
        let page = host_page();
        let queue = device.new_command_queue().unwrap();
        let ram = Memfd::new(8 * page);
        let view = ram.view(&[(4 * page, 2 * page)]);
        let reentrant = |ram: &Memfd, view: &View, calls: &Calls| -> DmaBufSource {
            let inner = ram.source(view, calls);
            let device = device.clone();
            Arc::new(move |request| {
                let own = device.new_buffer(4096, ResourceOptions::default()).unwrap();
                let private = Mapping::anonymous(page);
                // SAFETY: the mapping outlives the buffer, dropped first.
                let over = unsafe {
                    device.new_buffer_with_bytes_no_copy(
                        HostMemory::new(private.pointer()),
                        page as u64,
                        ResourceOptions::default(),
                        None,
                    )
                };
                drop((own, over.unwrap()));
                drop(private);
                inner(request)
            })
        };
        let calls = Calls::default();
        let before = dma_buf_counts(device);
        let source = reentrant(&ram, &view, &calls);
        let buffer = no_copy(device, &view, 0, 2 * page, source).unwrap();
        check_route(
            TEST,
            device,
            &calls,
            (view.at, 2 * page as u64),
            before,
            Some(1),
        );
        // A texture over it, a shadow, asks the source nothing.
        let counted = image_counts(device);
        let asked = calls.lock().unwrap().len();
        let texture = buffer
            .new_texture(&read_only(PixelFormat::BGRA8Unorm, 64, 8), 0, 256)
            .unwrap();
        check_images(TEST, device, counted, 0, 0);
        assert_eq!(
            calls.lock().unwrap().len(),
            asked,
            "{TEST}: the texture's calls"
        );
        drop(texture);
        drop(buffer);
        // A heap over other pages, and a texture placed in it: the heap's call, and the texture
        // image's (RADV).
        let heap_size = 1 << 20;
        let heap_ram = Memfd::new(heap_size);
        let heap_view = heap_ram.view(&[(0, heap_size)]);
        let calls = Calls::default();
        let before = dma_buf_counts(device);
        let source = reentrant(&heap_ram, &heap_view, &calls);
        let descriptor = HeapDescriptor::new(HeapType::Placement, heap_size as u64, shared());
        // SAFETY: the view outlives the heap, and the source makes dma-bufs of its pages.
        let heap = unsafe {
            device.new_heap_with_bytes_no_copy(
                HostMemory::with_dma_buf(heap_view.pointer(0), source),
                &descriptor,
                None,
                None,
            )
        }
        .unwrap();
        let pages = (heap_view.at, heap_size as u64);
        check_route(TEST, device, &calls, pages, before, None);
        let mut placed = read_only(PixelFormat::RGBA8Unorm, 64, 64);
        placed.resource_options = shared();
        let counted = image_counts(device);
        let texture = heap.new_texture_at_offset(&placed, 0).unwrap();
        check_images(TEST, device, counted, 1, 0);
        drop(texture);
        drop(heap);
        // A rebind of a buffer over private memory onto the view.
        let private = Mapping::anonymous(2 * page);
        // SAFETY: the mapping outlives the buffer.
        let buffer = unsafe {
            device.new_buffer_with_bytes_no_copy(
                HostMemory::new(private.pointer()),
                2 * page as u64,
                ResourceOptions::default(),
                None,
            )
        }
        .unwrap();
        let written = counting(2 * page, 41);
        ram.whole.write(4 * page, &written);
        let calls = Calls::default();
        let before = dma_buf_counts(device);
        // SAFETY: the view outlives the buffer, and no other thread encodes or commits.
        unsafe {
            buffer.rebind_bytes_no_copy(
                HostMemory::with_dma_buf(view.pointer(0), reentrant(&ram, &view, &calls)),
                None,
                None,
            )
        }
        .unwrap();
        check_route(
            TEST,
            device,
            &calls,
            (view.at, 2 * page as u64),
            before,
            Some(1),
        );
        assert_eq!(buffer.contents(), Some(view.pointer(0)));
        assert_eq!(gpu_read(&queue, &buffer, 0..2 * page as u64), written);
        drop(buffer);
        drop(private);
    });
}

/// How long a test waits for a command buffer.
const WAIT: Duration = Duration::from_secs(60);

/// Commit a command buffer on `queue` that waits for `event` to reach 1 and then copies
/// `length` bytes into a new shared buffer with `copy`; run `release` once it is committed, then
/// signal the event and wait: the command buffer's completion, on one of the device's handler
/// threads, lets go of what it held last. The bytes copied.
fn copied_after_release(
    queue: &CommandQueue,
    length: u64,
    copy: impl FnOnce(&mut metal2vulkan_runtime::BlitCommandEncoder, &Buffer),
    release: impl FnOnce(),
) -> Vec<u8> {
    let device = queue.device();
    let event = device.new_shared_event().unwrap();
    let out = device
        .new_buffer(length, ResourceOptions::shared())
        .unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    command_buffer
        .encode_wait_for_event(EventRef::from(&event), 1)
        .unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    copy(&mut blit, &out);
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    release();
    event.set_signaled_value(1);
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    read_buffer(&out, 0, length).unwrap()
}

/// Where a source went: the thread that dropped it, once it went.
type DroppedOn = Arc<Mutex<Option<std::thread::Thread>>>;

/// Notes, as it goes, the thread it goes on.
struct DropMark(DroppedOn);

impl Drop for DropMark {
    fn drop(&mut self) {
        *self.0.lock().unwrap() = Some(std::thread::current());
    }
}

/// `source`, noting in `on` the thread that drops it: the runtime drops a source with the import
/// it retained it with.
fn marked(source: DmaBufSource, on: &DroppedOn) -> DmaBufSource {
    let mark = DropMark(Arc::clone(on));
    Arc::new(move |request| {
        let _ = &mark;
        source(request)
    })
}

/// Where the device imports dma-bufs, wait for the source noted in `on` to go, and check it went
/// on a thread of the runtime's, not this test's: the import `what` was freed as a command buffer
/// that held it last completed.
fn went_off_thread(test: &str, device: &Device, on: &DroppedOn, what: &str) {
    if imports_dma_buf(device) != Some(true) {
        return;
    }
    let deadline = Instant::now() + WAIT;
    let thread = loop {
        if let Some(thread) = on.lock().unwrap().clone() {
            break thread;
        }
        assert!(Instant::now() < deadline, "{test}: {what}: never freed");
        std::thread::sleep(Duration::from_millis(1));
    };
    assert_ne!(
        thread.id(),
        std::thread::current().id(),
        "{test}: {what}: freed on the test's thread"
    );
    eprintln!(
        "{test}: {what}: freed on the runtime's thread {:?}",
        thread.name()
    );
}

/// What a command buffer holds of dma-buf pages goes on a thread of the runtime's as it completes,
/// inside an exit-guard section (the runtime's debug builds end the process on a Vulkan call
/// outside one): a 64 x 64 `BGRA8Unorm` texture over a no-copy buffer of memfd memory, rows 256
/// bytes apart, released with the buffer while a blit out of the texture waits on an event (the
/// consumer's display-path surface textures); a buffer assembled from several chunks, released
/// so; and a buffer's old pages, rebound away while a command buffer first submitted before the
/// rebind, which copies out of them, waits on the GPU behind the busy work (one that waits on an
/// event is first submitted after the rebind, and reaches the new pages). Each blit reads the bytes the CPU
/// wrote, and each source, retained with its import, goes on a thread other than the test's.
/// Where the device imports the pointer the same holds of its import, and the source goes uncalled
/// as the buffer is made.
#[test]
fn dma_buf_pages_go_on_a_handler_thread_as_a_command_buffer_completes() {
    const TEST: &str = "dma-buf pages released at completion";
    if !udmabuf_openable(TEST) {
        return;
    }
    on_device(|device| {
        let page = host_page();
        let queue = device.new_command_queue().unwrap();
        // A texture over the buffer, both released before the blit runs.
        let length = 64 * 256;
        let ram = Memfd::new(length + 2 * page);
        let view = ram.view(&[(page, length)]);
        let written = counting(length, 7);
        ram.whole.write(page, &written);
        let calls = Calls::default();
        let before = dma_buf_counts(device);
        let dropped = DroppedOn::default();
        let source = marked(ram.source(&view, &calls), &dropped);
        let buffer = no_copy(device, &view, 0, length, source).unwrap();
        check_route(
            TEST,
            device,
            &calls,
            (view.at, length as u64),
            before,
            Some(1),
        );
        let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, 64, 64, false);
        descriptor.usage = TextureUsage::SHADER_READ;
        let texture = buffer.new_texture(&descriptor, 0, 256).unwrap();
        let held = Mutex::new(Some((texture, buffer)));
        let read = copied_after_release(
            &queue,
            length as u64,
            |blit, out| {
                let guard = held.lock().unwrap();
                let (texture, _) = guard.as_ref().unwrap();
                blit.copy_from_texture_to_buffer(
                    texture,
                    0,
                    0,
                    Origin::new(0, 0, 0),
                    Size::new(64, 64, 1),
                    out,
                    0,
                    256,
                    length as u64,
                    BlitOption::None,
                )
                .unwrap();
            },
            || drop(held.lock().unwrap().take()),
        );
        assert!(read == written, "{TEST}: the texture's texels");
        went_off_thread(TEST, device, &dropped, "the texture's pages");
        // A buffer of several chunks, released before the copy runs.
        let runs = Limits::of_host().items + 76;
        let (ram, view, expected) = every_other_page(runs);
        let length = runs * page;
        let calls = Calls::default();
        let before = dma_buf_counts(device);
        let dropped = DroppedOn::default();
        let source = marked(ram.source(&view, &calls), &dropped);
        let buffer = no_copy(device, &view, 0, length, source).unwrap();
        check_route(TEST, device, &calls, (view.at, length as u64), before, None);
        let held = Mutex::new(Some(buffer));
        let read = copied_after_release(
            &queue,
            length as u64,
            |blit, out| {
                let guard = held.lock().unwrap();
                let buffer = guard.as_ref().unwrap();
                blit.copy_from_buffer_to_buffer(buffer, 0, out, 0, length as u64)
                    .unwrap();
            },
            || drop(held.lock().unwrap().take()),
        );
        assert!(read == expected, "{TEST}: the assembled buffer's bytes");
        went_off_thread(TEST, device, &dropped, "the assembled buffer's chunks");
        // A buffer's old pages, rebound away while a command buffer that copies out of them,
        // first submitted before the rebind, waits on the GPU behind the busy work.
        let busy = Busy::new(device, &queue);
        let length = 2 * page;
        let ram = Memfd::new(8 * page);
        let old = ram.view(&[(page, length)]);
        let new = ram.view(&[(4 * page, length)]);
        let written = counting(length, 3);
        ram.whole.write(page, &written);
        let calls = Calls::default();
        let before = dma_buf_counts(device);
        let dropped = DroppedOn::default();
        let source = marked(ram.source(&old, &calls), &dropped);
        let buffer = no_copy(device, &old, 0, length, source).unwrap();
        check_route(
            TEST,
            device,
            &calls,
            (old.at, length as u64),
            before,
            Some(1),
        );
        let out = device
            .new_buffer(length as u64, ResourceOptions::shared())
            .unwrap();
        let (busy_work, words) = busy.commit(device, &queue);
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_buffer_to_buffer(&buffer, 0, &out, 0, length as u64)
            .unwrap();
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        scheduled(&command_buffer);
        // SAFETY: both views outlive the buffer, and no other thread encodes or commits.
        unsafe {
            buffer.rebind_bytes_no_copy(
                HostMemory::with_dma_buf(new.pointer(0), ram.source(&new, &Calls::default())),
                None,
                None,
            )
        }
        .unwrap();
        assert_ne!(
            read_mark(&words),
            ENDED,
            "{TEST}: the busy work ended before the rebind"
        );
        if imports_dma_buf(device) == Some(true) {
            assert!(
                dropped.lock().unwrap().is_none(),
                "{TEST}: the old pages were freed while a command buffer in flight held them"
            );
        }
        release(&words);
        busy_work.wait_until_completed_for(WAIT).unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(busy_work.error(), None);
        assert_eq!(command_buffer.error(), None);
        assert_eq!(read_mark(&words), ENDED, "{TEST}: the busy work's end mark");
        let read = read_buffer(&out, 0, length as u64).unwrap();
        assert!(read == written, "{TEST}: the old pages' bytes");
        went_off_thread(TEST, device, &dropped, "the old pages");
        drop(buffer);
    });
}

/// A no-copy buffer assembled from several chunks (a sparse buffer, RADV) made while the busy
/// work's command buffers spin on the device's first `VkQueue` until the test releases them: the
/// call returns while the work still spins (its binding waits for no GPU work in flight), and once
/// it is released the GPU reads the buffer's bytes. The busy work gives up after about 2 s, so a
/// call that waits for it fails the test rather than hangs it. Elsewhere the pointer is imported,
/// which waits for nothing either.
#[test]
fn a_sparse_buffer_s_binding_waits_for_no_gpu_work() {
    const TEST: &str = "dma-buf binding behind GPU work";
    if !udmabuf_openable(TEST) {
        return;
    }
    on_device(|device| {
        // The first queue made on a new device submits to its first `VkQueue`.
        let queue = device.new_command_queue().unwrap();
        let busy = Busy::new(device, &queue);
        // Past RADV's largest sparse granule measured (64 KiB), so a split at it is several.
        let span = 3 << 16;
        let ram = Memfd::new(span);
        let view = ram.view(&[(0, span)]);
        let written = counting(span, 23);
        ram.whole.write(0, &written);
        let calls = Calls::default();
        let before = dma_buf_counts(device);
        let (busy_work, mark) = busy.commit(device, &queue);
        busy_work.scheduled();
        assert_eq!(read_mark(&mark), 0, "{TEST}: the busy work ended first");
        let at = Instant::now();
        let made = no_copy(device, &view, 0, span, ram.granules(&view, &calls));
        let took = at.elapsed();
        let spinning = read_mark(&mark) == 0;
        release(&mark);
        busy_work.wait_until_completed_for(WAIT).unwrap();
        let buffer = made.unwrap();
        eprintln!("{TEST}: on {}: the call took {took:?}", device.name());
        assert!(spinning, "{TEST}: the call returned after the busy work");
        let chunks = check_route(TEST, device, &calls, (view.at, span as u64), before, None);
        if imports_dma_buf(device) == Some(true) {
            assert!(chunks > 1, "{TEST}: {chunks} chunks answered");
        }
        assert!(
            gpu_read(&queue, &buffer, 0..span as u64) == written,
            "{TEST}: the GPU's read"
        );
        drop(buffer);
    });
}

/// The device's counts of heap textures' images made over dma-bufs of their own pages, and of
/// heap textures over dma-buf pages that took their shadow instead.
fn image_counts(device: &Device) -> (u64, u64) {
    let counts = device.work_counts();
    (counts.dma_buf_images, counts.dma_buf_image_fallbacks)
}

/// Check what making textures over dma-buf pages counted since `before`: `images` images and
/// `fallbacks` fallbacks where the device imports dma-bufs (RADV, whose list holds
/// `VK_EXT_image_drm_format_modifier` and `VK_EXT_queue_family_foreign`); none where it imports
/// the pointer (NVIDIA, lavapipe), whose heap textures are images bound to the pointer's import.
fn check_images(test: &str, device: &Device, before: (u64, u64), images: u64, fallbacks: u64) {
    let now = image_counts(device);
    let counted = (now.0 - before.0, now.1 - before.1);
    match imports_dma_buf(device) {
        Some(true) => assert_eq!(counted, (images, fallbacks), "{test}: images, fallbacks"),
        Some(false) => assert_eq!(counted, (0, 0), "{test}: counted"),
        None => eprintln!("{test}: on {}: counted {counted:?}", device.name()),
    }
}

/// The `height` rows of `row` bytes at `offset` of `bytes`, `pitch` apart, packed.
fn packed(bytes: &[u8], offset: usize, pitch: usize, row: usize, height: usize) -> Vec<u8> {
    (0..height)
        .flat_map(|r| &bytes[offset + r * pitch..offset + r * pitch + row])
        .copied()
        .collect()
}

/// A `width` x `height` 2D texture of `pf` that shaders only read.
fn read_only(pf: PixelFormat, width: u64, height: u64) -> TextureDescriptor {
    let mut descriptor = TextureDescriptor::texture_2d(pf, width, height, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    descriptor
}

/// Check the source's requests `asked`, made as a texture whose `texels` are `(pitch, row, rows)`
/// (rows `pitch` bytes apart, `row` bytes each) was made at `offset` of a view starting `at`:
/// where the device made its image over a dma-buf of its own pages (`image`, a heap texture on
/// RADV), one request, for the page its texels start in and the pages from there its image's
/// memory covers, in one chunk (its granule its length), one chunk answered; none where it took
/// its shadow, or where the device imports the pointer.
fn check_texture_request(
    test: &str,
    device: &Device,
    asked: &[(DmaBufRequest, usize)],
    at: NonNull<u8>,
    offset: usize,
    texels: (usize, usize, usize),
    image: bool,
) {
    let page = host_page();
    let (pitch, row, rows) = texels;
    match (imports_dma_buf(device), image) {
        (Some(true), true) => {
            assert_eq!(asked.len(), 1, "{test}: the source's calls: {asked:?}");
            let (request, chunks) = asked[0];
            let first = offset / page * page;
            let needed = (offset % page + pitch * (rows - 1) + row) as u64;
            // SAFETY: inside the view.
            let start = unsafe { NonNull::new_unchecked(at.as_ptr().add(first)) };
            assert_eq!(request.start, start, "{test}: {request:?}");
            assert!(
                request.length >= needed
                    && request.length.is_multiple_of(page as u64)
                    && request.length < needed + page as u64,
                "{test}: {request:?} for {needed} bytes"
            );
            assert_eq!(request.granule, request.length, "{test}: {request:?}");
            assert_eq!(chunks, 1, "{test}: chunks answered");
        }
        (Some(_), _) => assert!(asked.is_empty(), "{test}: the source's calls: {asked:?}"),
        (None, _) => eprintln!("{test}: on {}: the source's calls {asked:?}", device.name()),
    }
}

/// Under `HostPageTexturePolicy::InPlace`, once the device's check answered (printed), a texture
/// that only reads over a no-copy buffer of memfd memory takes the image in place over a dma-buf
/// of its own pages where the source makes one, and its shadow where it makes none: over one run
/// of the file (64x16 `RGBA8Unorm`, rows 256 bytes apart), one dma-buf; over more discontiguous
/// runs than one udmabuf takes (a `page / 4`-wide `RGBA8Unorm` texture of a row a page, over
/// `list_limit + 76` single pages, the guest surface whose pages are scattered), the source makes
/// no one dma-buf, and the shadow is counted as the source's refusal
/// (`host_texture_shadows_refused_source`). An image in place is counted by its pages' class
/// (`host_textures_in_place`): over a dma-buf of its own pages (`dma_buf_textures_in_place`)
/// where the device imports dma-bufs (RADV), and over the pointer's import where the device
/// imports the pointer (NVIDIA, lavapipe), whose buffer and texture never call the source. Where
/// the device imports dma-bufs and its check passed for them, the first is in place and the
/// second the source's refusal; where it imports the pointer and its check passed for that
/// class, each is in place over the pointer's import or a shadow its refusal counts (as
/// `in_place_textures.rs` holds any texture over host pages); elsewhere the forms are printed.
/// Either way a blit reads the CPU's bytes and the CPU's later bytes. Printed
/// (`DMA-BUF-IN-PLACE`): each texture's form, its class and the shadow's counted reasons.
#[test]
fn sampled_textures_over_dma_buf_pages_are_in_place_where_their_pages_make_one_dma_buf() {
    const TEST: &str = "DMA-BUF-IN-PLACE";
    if !udmabuf_openable(TEST) {
        return;
    }
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
            options.policy.host_page_textures = HostPageTexturePolicy::InPlace;
        })
        .unwrap();
        assert!(device.settle_in_place_checks(CHECK_WAIT));
        let report = device.in_place_check_report();
        eprintln!("{TEST} on {}: the check: {report:?}", device.name());
        let passed = |class: &str| {
            report
                .iter()
                .any(|line| *line == format!("{class}: Passed"))
        };
        let imports = imports_dma_buf(&device);
        let queue = device.new_command_queue().unwrap();
        let page = host_page();
        let runs = Limits::of_host().items + 76;
        // What, the memfd and its view, the texture's width, height and pitch, and whether the
        // source makes one dma-buf of its pages.
        let one_run = {
            let ram = Memfd::new(10 * page);
            let view = ram.view(&[(2 * page, 8 * page)]);
            ram.whole.write(2 * page, &counting(8 * page, 13));
            ("one run", ram, view, (64u64, 16u64, 256usize), true)
        };
        let scattered = {
            let (ram, view, _) = every_other_page(runs);
            let width = (page / 4) as u64;
            (
                "scattered runs",
                ram,
                view,
                (width, runs as u64, page),
                false,
            )
        };
        for (what, ram, view, (width, height, pitch), one) in [one_run, scattered] {
            let calls = Calls::default();
            let buffer = no_copy(&device, &view, 0, view.len, ram.source(&view, &calls)).unwrap();
            let before = device.work_counts();
            let texture = buffer
                .new_texture(
                    &read_only(PixelFormat::RGBA8Unorm, width, height),
                    0,
                    pitch as u64,
                )
                .unwrap();
            let after = device.work_counts();
            // SAFETY: the handle is only compared, while the texture lives.
            let aliased = unsafe { texture.vk_image() }.is_some();
            let in_place = after.host_textures_in_place - before.host_textures_in_place;
            let over_dma_buf = after.dma_buf_textures_in_place - before.dma_buf_textures_in_place;
            let source = after.host_texture_shadows_refused_source
                - before.host_texture_shadows_refused_source;
            let pitched = after.host_texture_shadows_refused_pitch
                - before.host_texture_shadows_refused_pitch;
            let other = after.host_texture_shadows_refused_dma_buf
                - before.host_texture_shadows_refused_dma_buf;
            eprintln!(
                "{TEST} on {}: {what}, {width}x{height} rows {pitch} bytes apart: {} (in place \
                 {in_place}, of them over a dma-buf {over_dma_buf}; refused for the source \
                 {source}, the pitch {pitched}, other dma-buf reasons {other}), the source's \
                 calls {:?}",
                device.name(),
                if aliased { "in place" } else { "shadowed" },
                calls.lock().unwrap(),
            );
            assert_eq!(in_place, u64::from(aliased), "{TEST}: {what}: in place");
            match imports {
                Some(true) => {
                    assert_eq!(over_dma_buf, in_place, "{TEST}: {what}: over a dma-buf");
                    if passed("DmaBuf") {
                        assert_eq!((aliased, source), (one, u64::from(!one)), "{TEST}: {what}");
                    }
                }
                Some(false) => {
                    assert_eq!(over_dma_buf, 0, "{TEST}: {what}: over a dma-buf");
                    let calls = calls.lock().unwrap();
                    assert!(
                        calls.is_empty(),
                        "{TEST}: {what}: the source's calls {calls:?}"
                    );
                    if passed("Pointer") {
                        assert_eq!(
                            in_place + refused(&before, &after),
                            1,
                            "{TEST}: {what}: in place, or a shadow its refusal counts"
                        );
                    }
                }
                None => {}
            }
            let (row, rows) = (4 * width as usize, height as usize);
            let size = (width, height);
            let bytes = view.read(0..pitch * rows);
            assert!(
                blit_texels(&queue, &texture, size, 4) == packed(&bytes, 0, pitch, row, rows),
                "{TEST}: {what}: the CPU's bytes"
            );
            let later = counting(pitch * rows, 41);
            view.write(0, &later);
            assert!(
                blit_texels(&queue, &texture, size, 4) == packed(&later, 0, pitch, row, rows),
                "{TEST}: {what}: the CPU's later bytes"
            );
            drop(texture);
            drop(buffer);
            drop(view);
            drop(ram);
        }
        messages.take_notices();
    }
    layer_reported_nothing_at_all(&messages);
}

/// Textures over a no-copy buffer of memfd memory (the consumer's display-path surfaces), rows 256
/// bytes apart, are shadows of the buffer's bytes on every device (`shadowed-texture`), as over any
/// host pages: none asks the source, none counts as an image or a fallback, and making one asks
/// the driver no format's modifier features (`WorkCounts::modifier_feature_queries`). Over three
/// pages, a page into the buffer; over one page, whose rows of 240 bytes of texels have 16 of the
/// file's between them; 256 bytes into a page; 64 bytes into a page, of 64-byte rows; and one a
/// shader writes. Each reads the bytes the CPU wrote; the CPU's later writes, with no call between,
/// are what the next command buffer reads; a blit's write into it lands in the file's pages at the
/// texels' places, the padding past each row keeping the CPU's bytes. The one over three pages,
/// `BGRA8Unorm` made without `PixelFormatView` usage, is read and written through a view in
/// `RGBA8Unorm` (`texture/view_shadow.rs`), whose write the texture then reads.
#[test]
fn a_texture_over_dma_buf_pages_is_a_shadow_of_them() {
    const TEST: &str = "dma-buf texture";
    if !udmabuf_openable(TEST) {
        return;
    }
    on_device(|device| {
        let page = host_page();
        let queue = device.new_command_queue().unwrap();
        let length = 8 * page;
        let ram = Memfd::new(10 * page);
        let pages = ram.view(&[(2 * page, length)]);
        let at = pages.at;
        ram.whole.write(2 * page, &counting(length, 13));
        let calls = Calls::default();
        let before = dma_buf_counts(device);
        let buffer = no_copy(device, &pages, 0, length, ram.source(&pages, &calls)).unwrap();
        check_route(TEST, device, &calls, (at, length as u64), before, Some(1));
        let pitch = 256;
        // What, its descriptor, offset, and whether it is used through a view.
        let writable = |pf, width, height| {
            let mut descriptor = read_only(pf, width, height);
            descriptor.usage |= TextureUsage::SHADER_WRITE;
            descriptor
        };
        let cases = [
            (
                "one page",
                read_only(PixelFormat::RGBA8Unorm, 60, 16),
                0,
                false,
            ),
            (
                "three pages",
                read_only(PixelFormat::BGRA8Unorm, 64, 40),
                page,
                true,
            ),
            (
                "256 bytes in",
                read_only(PixelFormat::RGBA8Unorm, 64, 20),
                5 * page + 256,
                false,
            ),
            (
                "64 bytes in",
                read_only(PixelFormat::RGBA8Unorm, 16, 8),
                7 * page + 64,
                false,
            ),
            (
                "shader write",
                writable(PixelFormat::RGBA8Unorm, 64, 16),
                0,
                false,
            ),
        ];
        for (k, case) in cases.into_iter().enumerate() {
            let (what, descriptor, offset, view) = case;
            let test = format!("{TEST}: {what}");
            let size = (descriptor.width, descriptor.height);
            let (row, rows) = (4 * size.0 as usize, size.1 as usize);
            let asked_before = calls.lock().unwrap().len();
            let counted = image_counts(device);
            let earlier = counting(pitch * rows, 20 + k);
            ram.whole.write(2 * page + offset, &earlier);
            let queries = device.work_counts().modifier_feature_queries;
            let texture = buffer
                .new_texture(&descriptor, offset as u64, pitch as u64)
                .unwrap();
            assert_eq!(
                device.work_counts().modifier_feature_queries,
                queries,
                "{test}: the texture asked the driver's modifier features"
            );
            // SAFETY: the handle is only compared, while the texture lives.
            let bound = unsafe { texture.vk_image() }.is_some();
            assert!(!bound, "{test}: an image bound to host pages");
            check_images(&test, device, counted, 0, 0);
            let asked = calls.lock().unwrap()[asked_before..].to_vec();
            let texels = (pitch, row, rows);
            check_texture_request(&test, device, &asked, at, offset, texels, false);
            let target = match view {
                true => {
                    let counted = image_counts(device);
                    let viewed = texture.new_texture_view(PixelFormat::RGBA8Unorm).unwrap();
                    check_images(&test, device, counted, 0, 0);
                    viewed
                }
                false => texture.clone(),
            };
            assert!(
                blit_texels(&queue, &target, size, 4) == packed(&earlier, 0, pitch, row, rows),
                "{test}: the CPU's bytes"
            );
            // The CPU's later bytes, with no call between.
            let later = counting(pitch * rows, 60 + k);
            ram.whole.write(2 * page + offset, &later);
            assert!(
                blit_texels(&queue, &target, size, 4) == packed(&later, 0, pitch, row, rows),
                "{test}: the CPU's later bytes"
            );
            // A blit's write lands at the texels' places; the padding keeps the CPU's bytes.
            let written = counting(row * rows, 100 + k);
            blit_into(&queue, &target, size, 4, &written);
            assert!(
                blit_texels(&queue, &texture, size, 4) == written,
                "{test}: the texture reads the write"
            );
            let file = ram
                .whole
                .read(2 * page + offset..2 * page + offset + pitch * rows);
            for r in 0..rows {
                let at = r * pitch;
                assert!(
                    file[at..at + row] == written[r * row..(r + 1) * row],
                    "{test}: row {r} of the blit's texels"
                );
                assert!(
                    file[at + row..at + pitch] == later[at + row..at + pitch],
                    "{test}: row {r}'s padding"
                );
            }
            drop((target, texture));
        }
        drop(buffer);
    });
}

/// A texture over a no-copy buffer of dma-buf pages, a shadow of the buffer's bytes, is ordered
/// with the buffer's other uses (its copy in reads the buffer's bytes, its copy back writes them):
/// in one command buffer, a blit's write of the buffer is what the next encoder's copy out of the
/// texture reads, and a copy into the texture is what the next encoder's copy out of the buffer
/// reads; and so across two command buffers on one queue. Nothing counts as an image or a
/// fallback. Where the device imports the pointer the same holds.
#[test]
fn a_buffer_and_its_texture_over_dma_buf_pages_order_their_writes() {
    const TEST: &str = "dma-buf texture and buffer";
    if !udmabuf_openable(TEST) {
        return;
    }
    on_device(|device| {
        let page = host_page();
        let queue = device.new_command_queue().unwrap();
        let length = 4 * page;
        let ram = Memfd::new(6 * page);
        let view = ram.view(&[(page, length)]);
        ram.whole.write(page, &counting(length, 5));
        let calls = Calls::default();
        let buffer = no_copy(device, &view, 0, length, ram.source(&view, &calls)).unwrap();
        let counted = image_counts(device);
        let size = (64, 16);
        let bytes = (4 * size.0 * size.1) as usize;
        let texture = buffer
            .new_texture(&read_only(PixelFormat::RGBA8Unorm, size.0, size.1), 0, 256)
            .unwrap();
        check_images(TEST, device, counted, 0, 0);
        let out = device
            .new_buffer(bytes as u64, ResourceOptions::shared())
            .unwrap();
        let copy = |command_buffer: &CommandBuffer, from: &Buffer, to: &Buffer| {
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            blit.copy_from_buffer_to_buffer(from, 0, to, 0, bytes as u64)
                .unwrap();
            blit.end_encoding().unwrap();
        };
        // A write through one, then a read through the other: in one command buffer, and in two
        // committed in order.
        let mut texels = Vec::new();
        for (two, into_texture, salt) in [
            (false, false, 5),
            (false, true, 77),
            (true, false, 101),
            (true, true, 151),
        ] {
            texels = counting(bytes, salt);
            let from = staged(device, &texels);
            let first = queue.command_buffer().unwrap();
            match into_texture {
                true => encode_texels_in(&first, &from, size, 4, &texture),
                false => copy(&first, &from, &buffer),
            }
            let second = two.then(|| {
                first.commit().unwrap();
                queue.command_buffer().unwrap()
            });
            let reader = second.as_ref().unwrap_or(&first);
            match into_texture {
                true => copy(reader, &buffer, &out),
                false => encode_texels_out(reader, &texture, size, 4, &out),
            }
            run(reader);
            assert_eq!(first.error(), None);
            assert!(
                read_buffer(&out, 0, bytes as u64).unwrap() == texels,
                "{TEST}: two command buffers {two}, written into the texture {into_texture}"
            );
        }
        assert!(
            ram.whole.read(page..page + bytes) == texels,
            "{TEST}: the file's pages"
        );
        drop(texture);
        drop(buffer);
    });
}

/// Where the source fails for a heap texture's pages (it answers the heap's request, then none),
/// the texture placed in a no-copy heap over dma-buf pages takes its shadow of the heap's bytes,
/// counted as a fallback and announced once (`shadowed-texture`, naming the dma-buf pages and the
/// source's failure); the source asked once for it. A blit's write through it reads back. Where
/// the device imports the pointer, nothing is asked or counted.
#[test]
fn a_heap_texture_over_dma_buf_pages_takes_its_shadow_where_the_source_fails() {
    const TEST: &str = "dma-buf heap texture fallback";
    if !udmabuf_openable(TEST) {
        return;
    }
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let page = host_page();
        let size = 1 << 20;
        let queue = device.new_command_queue().unwrap();
        let ram = Memfd::new(size + 2 * page);
        let view = ram.view(&[(2 * page, size)]);
        ram.whole.write(2 * page, &counting(size, 17));
        let calls = Calls::default();
        let inner = ram.source(&view, &calls);
        let answered = Arc::new(Mutex::new(0usize));
        let asked = Arc::clone(&answered);
        let once: DmaBufSource = Arc::new(move |request| {
            let mut asked = asked.lock().unwrap();
            *asked += 1;
            match *asked {
                1 => inner(request),
                _ => Err(std::io::Error::other("no dma-buf for a texture")),
            }
        });
        let descriptor = HeapDescriptor::new(HeapType::Placement, size as u64, shared());
        // SAFETY: the view outlives the heap, and the source makes dma-bufs of its pages.
        let heap = unsafe {
            device.new_heap_with_bytes_no_copy(
                HostMemory::with_dma_buf(view.pointer(0), once),
                &descriptor,
                None,
                None,
            )
        }
        .unwrap();
        let extent = (64, 64);
        let mut texture_descriptor = read_only(PixelFormat::RGBA8Unorm, extent.0, extent.1);
        texture_descriptor.resource_options = shared();
        let answer = device
            .heap_texture_size_and_align(&texture_descriptor)
            .unwrap();
        let offset = answer.align.max(page as u64) * 3;
        let counted = image_counts(&device);
        let texture = heap
            .new_texture_at_offset(&texture_descriptor, offset)
            .unwrap();
        check_images(TEST, &device, counted, 0, 1);
        let notices: Vec<_> = messages
            .take_notices()
            .into_iter()
            .filter_map(|message| message.degradation)
            .filter(|notice| notice.degradation == Degradation::ShadowedTexture)
            .collect();
        match imports_dma_buf(&device) {
            Some(true) => {
                assert_eq!(*answered.lock().unwrap(), 2, "{TEST}: the source's calls");
                assert_eq!(notices.len(), 1, "{TEST}: {notices:?}");
                let reason = &notices[0].reason;
                assert!(
                    reason.contains("dma-buf") && reason.contains("no dma-buf for a texture"),
                    "{TEST}: {reason}"
                );
                eprintln!("{TEST}: on {}: shadowed-texture: {reason}", device.name());
            }
            Some(false) => assert_eq!(*answered.lock().unwrap(), 0, "{TEST}: called"),
            None => eprintln!("{TEST}: on {}: notices {notices:?}", device.name()),
        }
        let bytes = (4 * extent.0 * extent.1) as usize;
        let texels = counting(bytes, 61);
        blit_into(&queue, &texture, extent, 4, &texels);
        assert!(
            blit_texels(&queue, &texture, extent, 4) == texels,
            "{TEST}: the blit's write reads back"
        );
        drop((texture, heap));
    }
    layer_reported_nothing(&messages, TEST);
}

/// A placement heap of `view`'s length over its pages of `ram`, with `ram`'s source noting in
/// `calls`.
fn heap_over(device: &Device, ram: &Memfd, view: &View, calls: &Calls) -> Heap {
    let descriptor = HeapDescriptor::new(HeapType::Placement, view.len as u64, shared());
    // SAFETY: the view outlives the heap, and the source makes dma-bufs of its pages.
    unsafe {
        device.new_heap_with_bytes_no_copy(
            HostMemory::with_dma_buf(view.pointer(0), ram.source(view, calls)),
            &descriptor,
            None,
            None,
        )
    }
    .unwrap()
}

/// A no-copy heap over a memfd view (whose pages its size covers exactly): where the device
/// refuses the pointer (RADV), the heap's bytes are the source's chunks, and a buffer placed in it
/// reads the CPU's bytes at its offset and its GPU fill lands at the file's offsets. The heap
/// places a texture Metal may keep losslessly compressed (a 64 x 64 `RGBA8Unorm` texture only
/// read, `heap-texture-lossless-layout`), which a heap over pages imported by their pointer binds
/// as a linear image to its memory: where the heap's pages are imported as dma-bufs (RADV), the
/// texture is an image of the driver's `DRM_FORMAT_MOD_LINEAR` layout at its offset, over a
/// dma-buf of its own pages, counted, the source asked once for it; a blit's write into it reads
/// back, and lands in the heap's pages at the texture's offset (rows 256 bytes apart, the driver's
/// pitch for the image). Elsewhere the pointer is imported, nothing is counted or asked, and the
/// write reads back too.
#[test]
fn a_no_copy_heap_s_texture_over_dma_buf_pages_is_an_image_over_its_own_pages() {
    const TEST: &str = "dma-buf heap texture";
    if !udmabuf_openable(TEST) {
        return;
    }
    on_device(|device| {
        let page = host_page();
        let size = 1 << 20;
        let queue = device.new_command_queue().unwrap();
        let ram = Memfd::new(size + 2 * page);
        let view = ram.view(&[(2 * page, size)]);
        let held = counting(size, 29);
        ram.whole.write(2 * page, &held);
        let calls = Calls::default();
        let before = dma_buf_counts(device);
        let heap = heap_over(device, &ram, &view, &calls);
        check_route(TEST, device, &calls, (view.at, size as u64), before, None);
        let placed = 2 * page as u64;
        let buffer = heap
            .new_buffer_at_offset(placed, shared(), page as u64)
            .unwrap();
        assert_eq!(buffer.contents(), Some(view.pointer(page)));
        assert_eq!(gpu_read(&queue, &buffer, 0..placed), held[page..3 * page]);
        gpu_fill(&queue, &buffer, 0..placed, 0x3c);
        let file = ram.whole.read(3 * page..5 * page);
        assert!(file.iter().all(|&byte| byte == 0x3c), "{TEST}: the fill");
        drop(buffer);
        let extent = (64, 64);
        let mut descriptor = read_only(PixelFormat::RGBA8Unorm, extent.0, extent.1);
        descriptor.resource_options = shared();
        let answer = device.heap_texture_size_and_align(&descriptor).unwrap();
        let offset = answer.align.max(page as u64) * 3;
        let asked_before = calls.lock().unwrap().len();
        let counted = image_counts(device);
        let texture = heap.new_texture_at_offset(&descriptor, offset).unwrap();
        check_images(TEST, device, counted, 1, 0);
        let asked = calls.lock().unwrap()[asked_before..].to_vec();
        let texels = (256, 256, 64);
        check_texture_request(TEST, device, &asked, view.at, offset as usize, texels, true);
        let bytes = (4 * extent.0 * extent.1) as usize;
        let written = counting(bytes, 71);
        blit_into(&queue, &texture, extent, 4, &written);
        assert!(
            blit_texels(&queue, &texture, extent, 4) == written,
            "{TEST}: the blit's write reads back"
        );
        let at = 2 * page + offset as usize;
        let file = ram.whole.read(at..at + bytes);
        match imports_dma_buf(device) {
            Some(true) => assert!(file == written, "{TEST}: the heap's pages"),
            _ => eprintln!(
                "{TEST}: on {}: the heap's pages hold the texels row by row: {}",
                device.name(),
                file == written
            ),
        }
        drop(texture);
        drop(heap);
    });
}

/// A rebind of bytes over dma-buf pages to other such pages: a no-copy buffer's texture, a shadow
/// of the bytes, copies the new pages' texels, the new source asked once for the bytes alone,
/// nothing counted; a no-copy heap's texture is made again as an image over a dma-buf of its own
/// new pages (RADV), counted, the new source asked once for the bytes and once for the texture.
/// Each then reads the new pages' bytes, and a blit's write through it lands there, the old pages
/// keeping theirs. Elsewhere the textures are made again as before, nothing counted or asked.
#[test]
fn a_rebind_makes_textures_over_dma_buf_pages_again_over_the_new_pages() {
    const TEST: &str = "dma-buf texture rebind";
    if !udmabuf_openable(TEST) {
        return;
    }
    on_device(|device| {
        let page = host_page();
        let queue = device.new_command_queue().unwrap();
        let radv = imports_dma_buf(device) == Some(true);
        // A no-copy buffer's texture.
        let length = 2 * page;
        let ram = Memfd::new(8 * page);
        let old = ram.view(&[(page, length)]);
        let new = ram.view(&[(5 * page, length)]);
        let (before, after) = (counting(length, 3), counting(length, 101));
        ram.whole.write(page, &before);
        ram.whole.write(5 * page, &after);
        let source = ram.source(&old, &Calls::default());
        let buffer = no_copy(device, &old, 0, length, source).unwrap();
        let size = (64, 32);
        let bytes = (4 * size.0 * size.1) as usize;
        let texture = buffer
            .new_texture(&read_only(PixelFormat::RGBA8Unorm, size.0, size.1), 0, 256)
            .unwrap();
        assert!(blit_texels(&queue, &texture, size, 4) == before[..bytes]);
        let calls = Calls::default();
        let counted = image_counts(device);
        // SAFETY: both views outlive the buffer, and no other thread encodes or commits.
        unsafe {
            buffer.rebind_bytes_no_copy(
                HostMemory::with_dma_buf(new.pointer(0), ram.source(&new, &calls)),
                None,
                None,
            )
        }
        .unwrap();
        check_images(TEST, device, counted, 0, 0);
        let asked = calls.lock().unwrap().len();
        assert_eq!(asked, usize::from(radv), "{TEST}: the new source's calls");
        assert!(
            blit_texels(&queue, &texture, size, 4) == after[..bytes],
            "{TEST}: the texture reads the new pages"
        );
        let texels = counting(bytes, 55);
        blit_into(&queue, &texture, size, 4, &texels);
        assert!(
            ram.whole.read(5 * page..5 * page + bytes) == texels,
            "{TEST}: the new pages"
        );
        assert!(
            ram.whole.read(page..page + length) == before,
            "{TEST}: the old pages"
        );
        drop((texture, buffer));
        // A no-copy heap's texture.
        let heap_size = 1 << 20;
        let ram = Memfd::new(2 * heap_size);
        let old = ram.view(&[(0, heap_size)]);
        let new = ram.view(&[(heap_size, heap_size)]);
        let heap = heap_over(device, &ram, &old, &Calls::default());
        let extent = (64, 64);
        let mut descriptor = read_only(PixelFormat::RGBA8Unorm, extent.0, extent.1);
        descriptor.resource_options = shared();
        let texture = heap.new_texture_at_offset(&descriptor, 0).unwrap();
        let heap_bytes = (4 * extent.0 * extent.1) as usize;
        let written = counting(heap_bytes, 7);
        blit_into(&queue, &texture, extent, 4, &written);
        let calls = Calls::default();
        let counted = image_counts(device);
        // SAFETY: both views outlive the heap, and no other thread encodes or commits.
        unsafe {
            heap.rebind_bytes_no_copy(
                HostMemory::with_dma_buf(new.pointer(0), ram.source(&new, &calls)),
                None,
                None,
            )
        }
        .unwrap();
        check_images(TEST, device, counted, 1, 0);
        let asked = calls.lock().unwrap().len();
        let expected = if radv { 2 } else { 0 };
        assert_eq!(asked, expected, "{TEST}: the heap's new source's calls");
        let texels = counting(heap_bytes, 131);
        blit_into(&queue, &texture, extent, 4, &texels);
        assert!(
            blit_texels(&queue, &texture, extent, 4) == texels,
            "{TEST}: the heap's texture over the new pages"
        );
        if radv {
            assert!(
                ram.whole.read(heap_size..heap_size + heap_bytes) == texels,
                "{TEST}: the heap's new pages"
            );
            assert!(
                ram.whole.read(0..heap_bytes) == written,
                "{TEST}: the heap's old pages"
            );
        }
        drop(texture);
        drop(heap);
    });
}

/// A rebind between pages a source answers in one chunk and pages it answers a chunk per granule
/// (a sparse buffer, RADV), both ways, of a no-copy buffer 80 bytes into its page with a texture
/// over it whose rows are 240 bytes apart: the texture is a shadow of the bytes, as every texture
/// over a no-copy buffer's pages, its copies planned at the bytes' offset in their `VkBuffer`. Either buffer holds the bytes at their offset in their page, so each rebind is made
/// and the texture reads each new pages' texels. Elsewhere the pointer is imported each time. Then
/// a rebind onto anonymous memory, whose pointer every device imports: made, the texture reading
/// its texels, or refused (`NotImplemented`, `RebindTextures`) where that import holds the bytes
/// elsewhere in its `VkBuffer`; never made with the texels misplaced.
#[test]
fn a_rebind_between_one_chunk_and_several_keeps_a_texture_s_texels() {
    const TEST: &str = "dma-buf rebind between chunkings";
    if !udmabuf_openable(TEST) {
        return;
    }
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        // Past RADV's largest sparse granule measured (64 KiB), so a split at it is several.
        let span = 3 << 16;
        let ram = Memfd::new(3 * span);
        let views: Vec<View> = (0..3).map(|at| ram.view(&[(at * span, span)])).collect();
        let fills: Vec<Vec<u8>> = (0..3)
            .map(|at| {
                let fill = counting(span, 11 + 40 * at);
                ram.whole.write(at * span, &fill);
                fill
            })
            .collect();
        let (offset, length) = (80, span - 160);
        let size = (60, 64);
        let pitch = 4 * size.0;
        let bytes = (pitch * size.1) as usize;
        let source = ram.source(&views[0], &Calls::default());
        let buffer = no_copy(device, &views[0], offset, length, source).unwrap();
        let counted = image_counts(device);
        let descriptor = read_only(PixelFormat::RGBA8Unorm, size.0, size.1);
        let texture = buffer.new_texture(&descriptor, 0, pitch).unwrap();
        check_images(TEST, device, counted, 0, 0);
        assert!(blit_texels(&queue, &texture, size, 4) == fills[0][offset..offset + bytes]);
        for (at, several) in [(1, true), (2, false)] {
            let calls = Calls::default();
            let source = match several {
                true => ram.granules(&views[at], &calls),
                false => ram.source(&views[at], &calls),
            };
            let before = dma_buf_counts(device);
            // SAFETY: the views outlive the buffer, and no other thread encodes or commits.
            unsafe {
                buffer.rebind_bytes_no_copy(
                    HostMemory::with_dma_buf(views[at].pointer(offset), source),
                    None,
                    None,
                )
            }
            .unwrap();
            let assembled = dma_buf_counts(device).1 - before.1;
            if imports_dma_buf(device) == Some(true) {
                assert_eq!(assembled, u64::from(several), "{TEST}: view {at}");
            }
            assert!(
                blit_texels(&queue, &texture, size, 4) == fills[at][offset..offset + bytes],
                "{TEST}: the texture over view {at}"
            );
        }
        let anonymous = Mapping::anonymous(span);
        let fill = counting(span, 151);
        anonymous.write(0, &fill);
        let pointer = NonNull::new(anonymous.pointer().as_ptr().wrapping_add(offset)).unwrap();
        // SAFETY: the mapping outlives the buffer, and no other thread encodes or commits.
        let rebound = unsafe { buffer.rebind_bytes_no_copy(HostMemory::new(pointer), None, None) };
        match rebound {
            Ok(()) => assert!(
                blit_texels(&queue, &texture, size, 4) == fill[offset..offset + bytes],
                "{TEST}: the texture over anonymous memory"
            ),
            Err(Error::NotImplemented(RuntimeGap::RebindTextures)) => {}
            Err(error) => panic!("{TEST}: the rebind onto anonymous memory: {error:?}"),
        }
        eprintln!(
            "{TEST}: on {}: the rebind onto anonymous memory: {rebound:?}",
            device.name()
        );
        drop((texture, buffer));
    });
}
