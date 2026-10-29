//! The comparisons that keep a copied host buffer coherent with the caller's pages (`copied.rs`),
//! on plain memory: a commit's copy in writes the bytes the CPU changed alone, and a completion's
//! merge back writes the bytes the GPU changed alone, keeping a byte the CPU changed since. And
//! the copy's first read of the pages: where the kernel's read fails, pages mapped readable are
//! copied all the same and others refused; on Linux, a mapping of a dma-buf (`VM_PFNMAP`), which
//! the kernel's read refuses, is copied (where `/dev/udmabuf` opens).

use crate::copied::{read_caller, sync, CopiedBuffers, NotCopied};
use std::cell::Cell;
use std::ptr::NonNull;
use std::sync::Arc;

/// `bytes` as a pointer the comparison reads or writes.
fn at(bytes: &mut [u8]) -> NonNull<u8> {
    NonNull::new(bytes.as_mut_ptr()).unwrap()
}

#[test]
fn copied_copy_in_writes_the_bytes_the_cpu_changed_alone() {
    let mut snapshot: Vec<u8> = (0..37).collect();
    let mut pages = snapshot.clone();
    // The GPU changed byte 1 of the copy, not yet merged back.
    let mut copy = snapshot.clone();
    copy[1] = 0xEE;
    // The CPU changed byte 2, in the same word, and byte 36, in the last, short span.
    pages[2] = 0xAA;
    pages[36] = 0xBB;
    // SAFETY: three buffers of the snapshot's length.
    let written = unsafe { sync(at(&mut pages), at(&mut copy), &mut snapshot, false) };
    assert_eq!(written, 2);
    assert_eq!(copy[1], 0xEE, "the GPU's byte in the same word stays");
    assert_eq!((copy[2], copy[36]), (0xAA, 0xBB), "the CPU's bytes are in");
    assert_eq!((snapshot[2], snapshot[36]), (0xAA, 0xBB));
    assert_eq!(snapshot[1], 1, "the GPU's byte is not agreed yet");
}

#[test]
fn copied_merge_back_keeps_a_byte_the_cpu_changed_since() {
    let mut snapshot: Vec<u8> = (0..16).collect();
    let mut pages = snapshot.clone();
    let mut copy = snapshot.clone();
    // The GPU wrote bytes 3, 4 and 9; the CPU wrote byte 4 too, and byte 5, since the commit.
    copy[3] = 0x33;
    copy[4] = 0x44;
    copy[9] = 0x99;
    pages[4] = 0xC4;
    pages[5] = 0xC5;
    // SAFETY: three buffers of the snapshot's length.
    let written = unsafe { sync(at(&mut copy), at(&mut pages), &mut snapshot, true) };
    assert_eq!(written, 2, "bytes 3 and 9");
    assert_eq!(
        (pages[3], pages[9]),
        (0x33, 0x99),
        "the GPU's bytes are back"
    );
    assert_eq!((pages[4], pages[5]), (0xC4, 0xC5), "the CPU's bytes stay");
    // The next copy in carries the CPU's byte 4 (and 5) into the copy.
    // SAFETY: as above.
    let written = unsafe { sync(at(&mut pages), at(&mut copy), &mut snapshot, false) };
    assert_eq!(written, 2);
    assert_eq!((copy[4], copy[5]), (0xC4, 0xC5));
    assert_eq!(snapshot, pages, "they agree");
    assert_eq!(copy, pages);
}

/// Where the kernel's read of the pages fails (a mapping it cannot pin), pages the process maps
/// readable are read with plain loads and copied; where a page is not mapped readable, the copy
/// is refused and nothing written; where the kernel reads them, no mapping is asked about.
#[test]
fn copied_pages_the_kernel_cannot_read_are_copied_where_mapped_readable() {
    let mut pages: Vec<u8> = (0..4099_u32).map(|i| (i * 7 + 3) as u8).collect();
    let length = pages.len();
    let start = pages.as_ptr() as u64;
    let buffers = Arc::new(CopiedBuffers::default());
    let asked = Cell::new(None);
    let mut copy = vec![0_u8; length];
    let read_readable = |from: NonNull<u8>, into: &mut [u8]| {
        let readable = |start, end| {
            asked.set(Some((start, end)));
            true
        };
        // SAFETY: plain memory, readable.
        unsafe { read_caller(from, into, |_, _| false, readable) }
    };
    // SAFETY: the copy and the pages are the test's, of `length` bytes, live past the answer.
    let copied =
        unsafe { buffers.add_reading(at(&mut pages), at(&mut copy), length, read_readable) };
    assert!(copied.is_ok(), "copied");
    assert_eq!(copy, pages, "the copy holds the pages' bytes");
    assert_eq!(asked.get(), Some((start, start + length as u64)), "asked");
    let mut refused_copy = vec![0_u8; length];
    let read_unreadable = |from: NonNull<u8>, into: &mut [u8]| {
        // SAFETY: nothing is read.
        unsafe { read_caller(from, into, |_, _| false, |_, _| false) }
    };
    // SAFETY: as above; nothing reads the pages.
    let refused = unsafe {
        buffers.add_reading(
            at(&mut pages),
            at(&mut refused_copy),
            length,
            read_unreadable,
        )
    };
    assert_eq!(refused.err(), Some(NotCopied::Unreadable));
    assert!(refused_copy.iter().all(|&byte| byte == 0), "nothing copied");
    let mut into = vec![0_u8; length];
    asked.set(None);
    let kernel = |from: NonNull<u8>, into: &mut [u8]| {
        // SAFETY: the pages are `into.len()` bytes of plain memory.
        unsafe { std::ptr::copy_nonoverlapping(from.as_ptr(), into.as_mut_ptr(), into.len()) };
        true
    };
    // SAFETY: as above.
    let read = unsafe {
        read_caller(at(&mut pages), &mut into, kernel, |start, end| {
            asked.set(Some((start, end)));
            false
        })
    };
    assert!(read);
    assert_eq!(
        (into, asked.get()),
        (pages, None),
        "the kernel's read alone"
    );
}

/// A copy of pages of a dma-buf mapping (udmabuf's, `VM_PFNMAP`), which the kernel's read cannot
/// pin and fails: the copy is made, and holds their bytes. The test says which read took them
/// (`--no-capture`). Not run where `/dev/udmabuf` does not open, which the test says.
#[cfg(target_os = "linux")]
#[test]
fn copied_pages_of_a_dma_buf_mapping_are_copied() {
    use crate::copied::read_pages;
    // SAFETY: `sysconf` reads a system constant.
    let page = usize::try_from(unsafe { libc::sysconf(libc::_SC_PAGESIZE) }).unwrap();
    let length = 16 * page;
    let pattern: Vec<u8> = (0..length).map(|i| (i % 251) as u8 ^ 0x5A).collect();
    let Some(mapping) = DmaBufMapping::new(&pattern) else {
        return;
    };
    // The path the test is for: the kernel cannot read the mapping's pages.
    let mut probe = vec![0_u8; page];
    match read_pages(mapping.pages, &mut probe) {
        true => println!("the kernel reads this udmabuf mapping: copied through the kernel"),
        false => println!("the kernel cannot read this udmabuf mapping: copied with plain loads"),
    }
    let buffers = Arc::new(CopiedBuffers::default());
    let mut copy = vec![0_u8; length];
    // SAFETY: the copy is the test's; the mapping is `length` bytes, readable, live past the
    // answer.
    let copied = unsafe { buffers.add(mapping.pages, at(&mut copy), length) };
    assert!(copied.is_ok(), "copied");
    assert!(copy == pattern, "the copy holds the dma-buf's bytes");
}

/// A shared mapping of a dma-buf (`/dev/udmabuf`) of a `memfd`'s pages, as a consumer maps guest
/// RAM: `VM_PFNMAP`, which the kernel cannot pin. Unmapped when dropped.
#[cfg(target_os = "linux")]
pub(super) struct DmaBufMapping {
    /// The mapping's first byte.
    pub(super) pages: NonNull<u8>,
    length: usize,
    _dma_buf: std::os::fd::OwnedFd,
    _memfd: std::os::fd::OwnedFd,
}

#[cfg(target_os = "linux")]
impl DmaBufMapping {
    /// A mapping of a dma-buf of a new `memfd` holding `bytes` (whole pages), or `None` where
    /// `/dev/udmabuf` does not open, which it says.
    pub(super) fn new(bytes: &[u8]) -> Option<DmaBufMapping> {
        use std::io::Write;
        use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
        /// `struct udmabuf_create` of Linux's `include/uapi/linux/udmabuf.h`.
        #[repr(C)]
        struct Create {
            memfd: u32,
            flags: u32,
            offset: u64,
            size: u64,
        }
        /// `UDMABUF_CREATE`, `_IOW('u', 0x42, struct udmabuf_create)` in the generic ioctl
        /// encoding.
        const UDMABUF_CREATE: u64 =
            (1 << 30) | ((size_of::<Create>() as u64) << 16) | ((b'u' as u64) << 8) | 0x42;
        /// `UDMABUF_FLAGS_CLOEXEC`.
        const UDMABUF_FLAGS_CLOEXEC: u32 = 0x01;
        let device = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open("/dev/udmabuf");
        let device = match device {
            Ok(device) => device,
            Err(error) => {
                println!("/dev/udmabuf cannot be opened ({error}): not run");
                return None;
            }
        };
        let length = bytes.len();
        // A memfd sealed against shrinking, as udmabuf requires, holding the bytes.
        // SAFETY: a NUL-terminated name.
        let memfd = unsafe {
            libc::memfd_create(
                c"m2v-copied-udmabuf".as_ptr(),
                libc::MFD_CLOEXEC | libc::MFD_ALLOW_SEALING,
            )
        };
        assert!(memfd >= 0, "memfd: {}", std::io::Error::last_os_error());
        // SAFETY: a new descriptor, owned at once.
        let memfd = unsafe { OwnedFd::from_raw_fd(memfd) };
        std::fs::File::from(memfd.try_clone().unwrap())
            .write_all(bytes)
            .unwrap();
        // SAFETY: seals the memfd made above.
        let sealed =
            unsafe { libc::fcntl(memfd.as_raw_fd(), libc::F_ADD_SEALS, libc::F_SEAL_SHRINK) };
        assert_eq!(sealed, 0, "seal: {}", std::io::Error::last_os_error());
        let create = Create {
            memfd: memfd.as_raw_fd() as u32,
            flags: UDMABUF_FLAGS_CLOEXEC,
            offset: 0,
            size: length as u64,
        };
        // SAFETY: the ioctl reads `create`, a valid `udmabuf_create`.
        let dma_buf = unsafe { libc::ioctl(device.as_raw_fd(), UDMABUF_CREATE as _, &create) };
        assert!(dma_buf >= 0, "udmabuf: {}", std::io::Error::last_os_error());
        // SAFETY: a new descriptor the ioctl returned, owned at once.
        let dma_buf = unsafe { OwnedFd::from_raw_fd(dma_buf) };
        // SAFETY: a new shared mapping of the dma-buf, unmapped by the answer's drop.
        let mapped = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                length,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                dma_buf.as_raw_fd(),
                0,
            )
        };
        assert_ne!(
            mapped,
            libc::MAP_FAILED,
            "{}",
            std::io::Error::last_os_error()
        );
        Some(DmaBufMapping {
            pages: NonNull::new(mapped.cast()).unwrap(),
            length,
            _dma_buf: dma_buf,
            _memfd: memfd,
        })
    }
}

#[cfg(target_os = "linux")]
impl Drop for DmaBufMapping {
    fn drop(&mut self) {
        // SAFETY: the mapping made by `new`, which nothing uses now.
        let unmapped = unsafe { libc::munmap(self.pages.as_ptr().cast(), self.length) };
        assert_eq!(unmapped, 0);
    }
}
