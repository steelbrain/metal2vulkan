//! Whether a device imports a pointer to shared memory ([`Device::imports_shared_host_pointers`]),
//! found by importing one page of it.
//!
//! Drivers differ there, and nothing the device reports says which way: `VK_EXT_external_memory_
//! host` answers the memory types a pointer may take, and RADV answers them for a pointer whose
//! allocation it then refuses. RADV imports a pointer through amdgpu's userptr, which the kernel
//! takes only over anonymous private pages: it refused anonymous shared memory and every mapping
//! of a file, `memfd`s among them, typed. NVIDIA and lavapipe imported the same memory. So the
//! device is asked by doing it, once: one page of a `memfd` mapped shared, read and write (on
//! Linux, as QEMU's `memory-backend-memfd,share=on` guest RAM is; anonymous shared memory on other
//! Unix systems), imported as a no-copy buffer's pages are, and released at once.

use super::error::{Error, MemoryKind};
use super::host_memory::{import_window, refused};
use super::memory::MemoryClass;
use super::Device;
use std::ptr::NonNull;

impl Device {
    /// Whether the device imports a pointer to shared memory, as a no-copy buffer's import takes
    /// it (`VK_EXT_external_memory_host`): a runtime extension on Unix systems, with no Metal
    /// counterpart (Metal takes any page). Found once per device by importing one page of a
    /// `memfd` mapped shared, read and write (anonymous shared memory off Linux), which is
    /// released at once, and answered from that after; the import is counted as a no-copy
    /// buffer's is. False for a device that imports no host memory.
    ///
    /// Where it is false, a no-copy buffer over shared memory is made only from the caller's
    /// dma-buf source ([`HostMemory::with_dma_buf`](crate::HostMemory::with_dma_buf)), and a
    /// source is what such memory needs, whatever its pages' protection: pages the process maps
    /// read-only go to the source without their pointer's import being tried (a dma-buf pins
    /// nothing of the process's mapping), so a caller need not make a writable second mapping of
    /// them for the import. Measured: MoltenVK (an M3 Max) and lavapipe answer true, RADV (a
    /// Radeon 780M and an RX 7900 XTX, Mesa 26.2.4) false; NVIDIA's driver imported such memory
    /// where the no-copy buffers were measured, though this answer was not asked there.
    ///
    /// Fails only where the answer cannot be found: out of host memory for the page, or the
    /// device lost. The device's refusal of the pointer is no failure but the answer, false.
    pub fn imports_shared_host_pointers(&self) -> Result<bool, Error> {
        self.current().imports_shared_host_pointers_now()
    }

    /// [`Device::imports_shared_host_pointers`] on the generation that serves (`device/line.rs`).
    pub(crate) fn imports_shared_host_pointers_now(&self) -> Result<bool, Error> {
        let shared = self.shared();
        if let Some(&answer) = shared.shared_pointer_imports.get() {
            return Ok(answer);
        }
        let Ok(importer) = shared.host_import.as_ref() else {
            return Ok(false);
        };
        if self.is_lost() {
            return Err(self.lost_refusal());
        }
        let unit = importer.page.max(importer.alignment);
        let memory = SharedPages::new(2 * unit).ok_or(Error::OutOfMemory(MemoryKind::Host))?;
        // The first whole unit of the pages: the import's window is exactly it.
        let at = (memory.at.as_ptr() as u64).next_multiple_of(unit);
        let pointer = NonNull::new(at as *mut u8).ok_or(Error::OutOfMemory(MemoryKind::Host))?;
        let window = import_window(at, unit, importer.page, importer.alignment)
            .map_err(|_| Error::OutOfMemory(MemoryKind::Host))?;
        // SAFETY: the window's pages are this call's mapping, which stays mapped until the
        // placement is freed below.
        let imported = unsafe {
            self.import_pointer(importer, pointer, unit, &window, MemoryClass::Shared, false)
        };
        let answer = match imported {
            Ok((buffer, placement)) => {
                self.discard_import(buffer, placement);
                true
            }
            Err(error) if refused(&error) => false,
            Err(error) => return Err(error),
        };
        drop(memory);
        Ok(*shared.shared_pointer_imports.get_or_init(|| answer))
    }
}

/// Pages of shared memory mapped read and write, unmapped when it drops: of a `memfd` on Linux,
/// anonymous elsewhere.
struct SharedPages {
    at: NonNull<u8>,
    length: usize,
}

impl SharedPages {
    /// `length` bytes of them; `None` where they cannot be made.
    fn new(length: u64) -> Option<SharedPages> {
        let length = usize::try_from(length).ok()?;
        let (flags, fd) = SharedPages::backing(length)?;
        // SAFETY: a fresh mapping of `fd`'s first `length` bytes (or of anonymous memory, for -1),
        // which the kernel places.
        let at = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                length,
                libc::PROT_READ | libc::PROT_WRITE,
                flags,
                fd.as_ref().map_or(-1, std::os::fd::AsRawFd::as_raw_fd),
                0,
            )
        };
        // The mapping keeps the file; its descriptor closes here.
        drop(fd);
        if at == libc::MAP_FAILED {
            return None;
        }
        Some(SharedPages {
            at: NonNull::new(at.cast())?,
            length,
        })
    }

    /// The mapping's flags and file: a `memfd` of `length` bytes on Linux.
    #[cfg(target_os = "linux")]
    fn backing(length: usize) -> Option<(libc::c_int, Option<std::os::fd::OwnedFd>)> {
        use std::os::fd::FromRawFd;
        // SAFETY: a name and flags the call takes.
        let fd = unsafe {
            libc::memfd_create(c"metal2vulkan-shared-pointer".as_ptr(), libc::MFD_CLOEXEC)
        };
        if fd < 0 {
            return None;
        }
        // SAFETY: `fd` was just made and is owned here alone.
        let fd = unsafe { std::os::fd::OwnedFd::from_raw_fd(fd) };
        // SAFETY: a live descriptor of a `memfd` this call made.
        let sized =
            unsafe { libc::ftruncate(std::os::fd::AsRawFd::as_raw_fd(&fd), length as libc::off_t) };
        (sized == 0).then_some((libc::MAP_SHARED, Some(fd)))
    }

    /// The mapping's flags: anonymous shared memory off Linux.
    #[cfg(not(target_os = "linux"))]
    fn backing(_length: usize) -> Option<(libc::c_int, Option<std::os::fd::OwnedFd>)> {
        Some((libc::MAP_SHARED | libc::MAP_ANON, None))
    }
}

impl Drop for SharedPages {
    fn drop(&mut self) {
        // SAFETY: this object's own mapping, which nothing imports any more.
        unsafe { libc::munmap(self.at.as_ptr().cast(), self.length) };
    }
}
