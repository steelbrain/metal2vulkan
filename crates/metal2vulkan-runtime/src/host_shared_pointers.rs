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
//! Unix systems; on Windows a view of a section of the paging file, as a VMM's guest RAM made by
//! `CreateFileMapping` is), imported as a no-copy buffer's pages are, and released at once.

use super::error::{Error, MemoryKind};
use super::host_memory::{import_window, refused};
use super::memory::MemoryClass;
use super::Device;
use std::ptr::NonNull;

impl Device {
    /// Whether the device imports a pointer to shared memory, as a no-copy buffer's import takes
    /// it (`VK_EXT_external_memory_host`): a runtime extension, with no Metal counterpart (Metal
    /// takes any page). Found once per device by importing one page of a `memfd` mapped shared,
    /// read and write (anonymous shared memory elsewhere on Unix, a view of a section of the
    /// paging file on Windows), which is released at once, and answered from that after; the
    /// import is counted as a no-copy buffer's is. False for a device that imports no host
    /// pointer (one without `VK_EXT_external_memory_host`).
    ///
    /// Where it is false, a no-copy buffer over shared memory is imported only from the caller's
    /// dma-buf source ([`HostMemory::with_dma_buf`](crate::HostMemory::with_dma_buf)), and without
    /// one that works is a copy of the pages, announced (`copied-host-pages`,
    /// `WorkCounts::copied_host_buffers_device_refused`), as is any memory without one on a
    /// device that imports no pointer. A source is what such memory needs to be imported,
    /// whatever its pages' protection: pages the process maps read-only go to the source without
    /// their pointer's import being tried (a dma-buf pins nothing of the process's mapping), so a
    /// caller need not make a writable second mapping of them for the import. Measured: MoltenVK
    /// (an M3 Max) and lavapipe answer true, RADV (a Radeon 780M and an RX 7900 XTX, Mesa 26.2.4)
    /// false; NVIDIA's driver imported such memory where the no-copy buffers were measured,
    /// though this answer was not asked there.
    ///
    /// Fails only where the answer cannot be found: out of host memory for the page, or the
    /// device lost. The device's refusal of the pointer is no failure but the answer, false.
    pub fn imports_shared_host_pointers(&self) -> Result<bool, Error> {
        self.at_serving(|device| device.imports_shared_host_pointers_now())
    }

    /// [`Device::imports_shared_host_pointers`] on the generation that serves (`device/line.rs`).
    pub(crate) fn imports_shared_host_pointers_now(&self) -> Result<bool, Error> {
        let shared = self.shared();
        if let Some(&answer) = shared.shared_pointer_imports.get() {
            return Ok(answer);
        }
        let Some((importer, pointers)) = (shared.host_import.as_ref().ok())
            .and_then(|importer| Some((importer, importer.pointers()?)))
        else {
            return Ok(false);
        };
        if self.is_lost() {
            return Err(self.lost_refusal());
        }
        let unit = pointers.unit();
        let memory = SharedPages::new(2 * unit).ok_or(Error::OutOfMemory(MemoryKind::Host))?;
        // The first whole unit of the pages: the import's window is exactly it.
        let at = (memory.at.as_ptr() as u64).next_multiple_of(unit);
        let pointer = NonNull::new(at as *mut u8).ok_or(Error::OutOfMemory(MemoryKind::Host))?;
        let window = import_window(at, unit, importer.page, pointers.alignment)
            .map_err(|_| Error::OutOfMemory(MemoryKind::Host))?;
        // SAFETY: the window's pages are this call's mapping, which stays mapped until the
        // placement is freed below.
        let imported = unsafe {
            self.import_pointer(pointers, pointer, unit, &window, MemoryClass::Shared, false)
        };
        let answer = match imported {
            Ok((buffer, placement)) => {
                self.discard_import(buffer.raw, placement);
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
/// anonymous elsewhere on Unix, a view of a section of the paging file on Windows.
struct SharedPages {
    at: NonNull<u8>,
    #[cfg(unix)]
    length: usize,
}

impl SharedPages {
    /// `length` bytes of them; `None` where they cannot be made.
    #[cfg(unix)]
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

    /// The mapping's flags: anonymous shared memory on the other unix hosts.
    #[cfg(all(unix, not(target_os = "linux")))]
    fn backing(_length: usize) -> Option<(libc::c_int, Option<std::os::fd::OwnedFd>)> {
        Some((libc::MAP_SHARED | libc::MAP_ANON, None))
    }

    /// `length` bytes of them on Windows: a view, read and write, of a section of `length` bytes
    /// the paging file backs, committed (`CreateFileMappingW` of no file, `SEC_COMMIT`), as a VMM's
    /// guest RAM made so is; `None` where they cannot be made. The view keeps the section; its
    /// handle closes here.
    #[cfg(windows)]
    fn new(length: u64) -> Option<SharedPages> {
        use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MapViewOfFile, FILE_MAP_ALL_ACCESS, PAGE_READWRITE, SEC_COMMIT,
        };
        let bytes = usize::try_from(length).ok()?;
        let (high, low) = ((length >> 32) as u32, length as u32);
        // SAFETY: a new unnamed section of the paging file, with the default security.
        let section = unsafe {
            CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE | SEC_COMMIT,
                high,
                low,
                std::ptr::null(),
            )
        };
        if section.is_null() {
            return None;
        }
        // SAFETY: a view of the whole of the section just made, read and write.
        let view = unsafe { MapViewOfFile(section, FILE_MAP_ALL_ACCESS, 0, 0, bytes) };
        // SAFETY: the section's handle, made above and closed once; the view keeps the section.
        unsafe { CloseHandle(section) };
        Some(SharedPages {
            at: NonNull::new(view.Value.cast())?,
        })
    }
}

impl Drop for SharedPages {
    fn drop(&mut self) {
        #[cfg(unix)]
        // SAFETY: this object's own mapping, which nothing imports any more.
        unsafe {
            libc::munmap(self.at.as_ptr().cast(), self.length)
        };
        #[cfg(windows)]
        // SAFETY: this object's own view, which nothing imports any more.
        unsafe {
            use windows_sys::Win32::System::Memory::{UnmapViewOfFile, MEMORY_MAPPED_VIEW_ADDRESS};
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                Value: self.at.as_ptr().cast(),
            })
        };
    }
}
