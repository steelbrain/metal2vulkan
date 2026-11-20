//! Host pages a case hands the device as a no-copy buffer's bytes, and the read-only state of them
//! whose GPU writes Metal drops.
//!
//! Every other buffer in this suite is the device's own memory. A case here needs the other kind:
//! memory **the program owns**, handed to Metal by
//! `newBufferWithBytesNoCopy:length:options:deallocator:`, because that is the only shape in which
//! a Metal program can say "this memory is read-only" — Metal's private `MTLResourceOptions`
//! bit 24, which the old runtime named
//! `ResourceOptions::PRIVATE_BIT`. With the
//! bit, the GPU's writes to the memory go nowhere, and matching that exactly on every driver is
//! the owner's standing rule.
//!
//! # Why a case states bytes and a pattern rather than an address
//!
//! Nothing host-specific may reach a transcript, and a host pointer is the most host-specific
//! value there is. So a case states [`HostBufferSpec`]: how many bytes, the byte they all start
//! as, and whether the memory is read-only. The side maps them, fills them and answers a handle.
//! The case then reads the bytes back through
//! [`Side::read_buffer`](crate::case::Side::read_buffer), which is the CPU's own view of the
//! pages, and through a copy the GPU makes into an ordinary buffer, which is the GPU's.
//!
//! # Why the size is a multiple of 16 KiB
//!
//! A no-copy buffer's pointer and length are the host's pages, and the host page is 16 KiB on
//! Apple silicon and 4 KiB on the x86-64 and arm64 Linux venues. A case that stated "four pages"
//! would hand Metal 64 KiB here and lavapipe 16 KiB, and the two legs would then be comparing
//! read-backs of different lengths. So a case states **bytes**, [`PAGE_MULTIPLE`] holds the
//! largest page in play, and a spec whose size is not a multiple of it is refused before any
//! device sees it.
//!
//! # The CPU write is out of scope
//!
//! A CPU write through a read-only mapping ends the process (measured on an M3 Max under
//! macOS 27.0.1). That is Metal's own behaviour, it is not something this project changes, and
//! nothing here writes through such a mapping: [`HostPages::bytes`] reads, and the only write is
//! the pattern the pages are filled with before any read-only state exists.

use std::ptr::NonNull;

/// The page size every venue's own page divides: 16 KiB, Apple silicon's.
///
/// A [`HostBufferSpec`]'s size is a multiple of this, so one case's bytes are the same number of
/// bytes on every venue however large the host's page is.
pub const PAGE_MULTIPLE: u64 = 16 * 1024;

/// What a case asks for when it wants a buffer over its own host pages.
///
/// The read-only flag is Metal's private `MTLResourceOptions` bit 24, and it is the whole subject
/// of the family that uses this: with it, Metal takes the GPU's writes to the memory nowhere.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostBufferSpec {
    /// How many bytes the buffer spans, a multiple of [`PAGE_MULTIPLE`].
    pub bytes: u64,
    /// The byte every one of those bytes holds before anything else runs.
    ///
    /// A case states a distinctive one, never zero: against zeroed pages a write that wrongly
    /// wrote zeros and a write that was dropped read alike, which is the one reading this family
    /// exists to tell apart.
    pub pattern: u8,
    /// Metal's private `MTLResourceOptions` bit 24: the memory takes the GPU's writes nowhere.
    ///
    /// A case states both values of this, because "the write was dropped" and "my shader wrote
    /// nothing" are indistinguishable from the read-only run alone. The writable run is the
    /// control that makes the read-only run a measurement.
    pub read_only: bool,
}

impl HostBufferSpec {
    /// `bytes` of writable host memory, every byte `pattern`.
    pub const fn writable(bytes: u64, pattern: u8) -> Self {
        Self {
            bytes,
            pattern,
            read_only: false,
        }
    }

    /// The same memory made read-only: `MTLResourceOptions` bit 24.
    #[must_use]
    pub const fn read_only(mut self) -> Self {
        self.read_only = true;
        self
    }

    /// The spec's own bytes for an input fingerprint: the size, the pattern and the flag.
    ///
    /// No address and no handle, so a recording keyed on this is as portable as the rest of a
    /// case.
    pub fn words(&self) -> Vec<u8> {
        let mut words = self.bytes.to_le_bytes().to_vec();
        words.push(self.pattern);
        words.push(u8::from(self.read_only));
        words
    }

    /// Why the spec cannot be honoured on any host, where that is so: a size of zero, or one that
    /// is not a multiple of [`PAGE_MULTIPLE`].
    ///
    /// Decided here rather than per side, so both legs refuse the same spec for the same reason
    /// and a case cannot ask one venue a question the other could not be asked.
    pub fn unmappable(&self) -> Option<String> {
        if self.bytes == 0 {
            return Some("a no-copy buffer of no bytes".to_owned());
        }
        if !self.bytes.is_multiple_of(PAGE_MULTIPLE) {
            return Some(format!(
                "{} bytes is not a multiple of the {PAGE_MULTIPLE}-byte page every venue's own \
                 page divides",
                self.bytes
            ));
        }
        None
    }
}

/// Pages this process maps, which a side hands the device as a no-copy buffer's bytes.
///
/// It owns the mapping and unmaps it when it is dropped, so a side keeps one alive for as long as
/// the buffer over it: the field holding these is **declared last** in each side's struct, so
/// every resource pool drops before it does.
#[derive(Debug)]
pub struct HostPages {
    pointer: NonNull<u8>,
    length: usize,
}

// SAFETY: the mapping is this value's alone, and unmapping it is this value's alone.
unsafe impl Send for HostPages {}
// SAFETY: as above; a shared value answers its pointer and reads its bytes.
unsafe impl Sync for HostPages {}

impl HostPages {
    /// `length` bytes of private anonymous memory, every byte `pattern`: `None` where the system
    /// maps none.
    ///
    /// `mmap` (`VirtualAlloc` on Windows) rather than an allocation, because a no-copy buffer's
    /// pointer and length have to be the host's own pages — page-aligned and whole pages — which
    /// no allocator promises.
    pub fn of(length: usize, pattern: u8) -> Option<Self> {
        if length == 0 {
            return None;
        }
        let pointer = map(length)?;
        // SAFETY: the mapping is this value's own, just made writable, and `length` bytes long.
        unsafe { std::ptr::write_bytes(pointer.as_ptr(), pattern, length) };
        Some(Self { pointer, length })
    }

    /// Where the pages start: the pointer a no-copy buffer is made over.
    pub fn pointer(&self) -> NonNull<u8> {
        self.pointer
    }

    /// How many bytes they are.
    pub fn length(&self) -> usize {
        self.length
    }

    /// The pages as the CPU reads them now.
    ///
    /// A read, never a write: writing through a read-only mapping ends the process, and whether
    /// the GPU's write reached these bytes is exactly what a case asks.
    pub fn bytes(&self) -> &[u8] {
        // SAFETY: the mapping is this value's own, readable, and `length` bytes long. Reading it
        // while the GPU writes it is the data race `contents` is; a case reads it after it has
        // waited for every command buffer that could write it.
        unsafe { std::slice::from_raw_parts(self.pointer.as_ptr(), self.length) }
    }
}

impl Drop for HostPages {
    fn drop(&mut self) {
        #[cfg(unix)]
        // SAFETY: the mapping is this value's, and nothing reads it any more: the buffer over it
        // was dropped first, because the field holding these is declared last in its side.
        unsafe {
            libc::munmap(self.pointer.as_ptr().cast(), self.length);
        }
        #[cfg(windows)]
        // SAFETY: as above; a release frees the allocation whole, from its base, with a size of 0.
        unsafe {
            use windows_sys::Win32::System::Memory::{VirtualFree, MEM_RELEASE};
            VirtualFree(self.pointer.as_ptr().cast(), 0, MEM_RELEASE);
        }
    }
}

/// `length` bytes of new private anonymous pages, read-write, at an address the kernel picks;
/// `None` where the system maps none.
#[cfg(unix)]
fn map(length: usize) -> Option<NonNull<u8>> {
    // SAFETY: a new private anonymous mapping, which reads no memory of ours.
    let mapped = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            length,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANON,
            -1,
            0,
        )
    };
    if mapped == libc::MAP_FAILED {
        return None;
    }
    NonNull::new(mapped.cast::<u8>())
}

/// `length` bytes of new private pages, read-write, at an address the system picks: a committed
/// allocation of their own (`VirtualAlloc`); `None` where the system makes none.
#[cfg(windows)]
fn map(length: usize) -> Option<NonNull<u8>> {
    use windows_sys::Win32::System::Memory::{
        VirtualAlloc, MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE,
    };
    // SAFETY: a new allocation, which reads no memory of ours.
    let mapped = unsafe {
        VirtualAlloc(
            std::ptr::null(),
            length,
            MEM_RESERVE | MEM_COMMIT,
            PAGE_READWRITE,
        )
    };
    NonNull::new(mapped.cast::<u8>())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A spec's words hold its three facts and nothing else, and two specs have the same words
    /// exactly when they are equal.
    #[test]
    fn a_spec_fingerprints_as_its_three_facts() {
        let spec = HostBufferSpec::writable(PAGE_MULTIPLE * 4, 0xab);
        assert_eq!(spec.words().len(), 8 + 1 + 1);
        for moved in [
            HostBufferSpec::writable(PAGE_MULTIPLE * 8, 0xab),
            HostBufferSpec::writable(PAGE_MULTIPLE * 4, 0xac),
            spec.read_only(),
        ] {
            assert_ne!(spec, moved, "{moved:?}");
            assert_ne!(spec.words(), moved.words(), "{moved:?}");
        }
    }

    /// A size that is not a whole number of the largest page in play is refused before any device
    /// sees it, and a multiple of it is not.
    ///
    /// The failure this guards is a case reading 64 KiB on Metal and 16 KiB on a 4 KiB-page venue
    /// and comparing the two.
    #[test]
    fn a_size_that_is_not_a_whole_page_everywhere_is_refused() {
        assert!(HostBufferSpec::writable(0, 0xab).unmappable().is_some());
        for bytes in [1, 4096, PAGE_MULTIPLE - 1, PAGE_MULTIPLE + 1] {
            assert!(
                HostBufferSpec::writable(bytes, 0xab).unmappable().is_some(),
                "{bytes} bytes is not refused, so a case could hand two venues different lengths"
            );
        }
        for bytes in [PAGE_MULTIPLE, PAGE_MULTIPLE * 2, PAGE_MULTIPLE * 64] {
            assert_eq!(
                HostBufferSpec::writable(bytes, 0xab).unmappable(),
                None,
                "{bytes} bytes is refused, which would refuse a buffer every venue can make"
            );
        }
    }

    /// Mapped pages hold the pattern they were asked for, every byte of them, and the mapping is
    /// as long as it was asked for.
    #[test]
    fn mapped_pages_hold_their_pattern() {
        let length = usize::try_from(PAGE_MULTIPLE * 2).expect("a usize on every venue");
        let pages = HostPages::of(length, 0xab).expect("the system maps 32 KiB");
        assert_eq!(pages.length(), length);
        assert_eq!(pages.bytes().len(), length);
        assert!(pages.bytes().iter().all(|byte| *byte == 0xab));
        assert!(HostPages::of(0, 0xab).is_none());
    }
}
