//! What the device tests ask of the host system, in each system's form: fresh pages (private
//! anonymous memory, and shared memory as a VMM's guest RAM is: a `memfd` mapped shared on Linux,
//! anonymous shared memory on the other unix hosts, a view of a section of the paging file on
//! Windows), their release and protection, and CPU times.

use std::ptr::NonNull;
use std::time::Duration;

/// `length` fresh bytes of private anonymous memory, read and write, at a page boundary (on
/// Windows at the allocation granularity, 64 KiB): their first byte. [`release`] frees them.
pub(super) fn private(length: usize) -> NonNull<u8> {
    #[cfg(unix)]
    // SAFETY: a fresh private anonymous mapping, at an address the kernel chooses.
    let start = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            length,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANON,
            -1,
            0,
        )
    };
    #[cfg(unix)]
    assert_ne!(
        start,
        libc::MAP_FAILED,
        "{}",
        std::io::Error::last_os_error()
    );
    #[cfg(windows)]
    // SAFETY: a fresh allocation of its own, at an address the system chooses.
    let start = unsafe {
        use windows_sys::Win32::System::Memory::{
            VirtualAlloc, MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE,
        };
        VirtualAlloc(
            std::ptr::null(),
            length,
            MEM_RESERVE | MEM_COMMIT,
            PAGE_READWRITE,
        )
    };
    NonNull::new(start.cast()).expect("fresh private pages")
}

/// `length` fresh bytes of shared memory, read and write, as a VMM's guest RAM is: a `memfd`
/// mapped shared on Linux, anonymous shared memory on the other unix hosts, and on Windows a view
/// of a section of `length` bytes the paging file backs, committed (`CreateFileMappingW` of no
/// file, `SEC_COMMIT`): their first byte. [`release`] frees them.
pub(super) fn shared(length: usize) -> NonNull<u8> {
    #[cfg(target_os = "linux")]
    let (flags, fd) = {
        // SAFETY: a NUL-terminated name and flags the call takes.
        let fd = unsafe { libc::memfd_create(c"m2v-test-shared".as_ptr(), libc::MFD_CLOEXEC) };
        assert!(fd >= 0, "{}", std::io::Error::last_os_error());
        // SAFETY: the descriptor just made, closed below once mapped.
        assert_eq!(unsafe { libc::ftruncate(fd, length as libc::off_t) }, 0);
        (libc::MAP_SHARED, fd)
    };
    #[cfg(all(unix, not(target_os = "linux")))]
    let (flags, fd) = (libc::MAP_SHARED | libc::MAP_ANON, -1);
    #[cfg(unix)]
    // SAFETY: a fresh shared mapping of the descriptor's bytes (or of anonymous memory).
    let start = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            length,
            libc::PROT_READ | libc::PROT_WRITE,
            flags,
            fd,
            0,
        )
    };
    #[cfg(unix)]
    {
        assert_ne!(
            start,
            libc::MAP_FAILED,
            "{}",
            std::io::Error::last_os_error()
        );
        if fd >= 0 {
            // SAFETY: the descriptor made above; the mapping keeps the file.
            unsafe { libc::close(fd) };
        }
    }
    #[cfg(windows)]
    let start = {
        use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MapViewOfFile, FILE_MAP_ALL_ACCESS, PAGE_READWRITE, SEC_COMMIT,
        };
        let bytes = length as u64;
        // SAFETY: a new unnamed section of the paging file, with the default security.
        let section = unsafe {
            CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE | SEC_COMMIT,
                (bytes >> 32) as u32,
                bytes as u32,
                std::ptr::null(),
            )
        };
        assert!(!section.is_null(), "{}", std::io::Error::last_os_error());
        // SAFETY: a view of the whole of the section just made, read and write.
        let view = unsafe { MapViewOfFile(section, FILE_MAP_ALL_ACCESS, 0, 0, length) };
        // SAFETY: the section's handle, made above and closed once; the view keeps the section.
        unsafe { CloseHandle(section) };
        view.Value
    };
    NonNull::new(start.cast()).expect("fresh shared pages")
}

/// Free the `length` bytes at `start` that [`private`] or [`shared`] made, whole; nothing may use
/// them any more.
pub(super) fn release(start: NonNull<u8>, length: usize) {
    #[cfg(unix)]
    // SAFETY: the caller's: a mapping of the test's own, which nothing uses any more.
    unsafe {
        assert_eq!(libc::munmap(start.as_ptr().cast(), length), 0);
    }
    #[cfg(windows)]
    {
        use windows_sys::Win32::System::Memory::{
            UnmapViewOfFile, VirtualFree, VirtualQuery, MEMORY_BASIC_INFORMATION,
            MEMORY_MAPPED_VIEW_ADDRESS, MEM_MAPPED, MEM_RELEASE,
        };
        let _ = length;
        let mut info = MEMORY_BASIC_INFORMATION::default();
        let size = std::mem::size_of::<MEMORY_BASIC_INFORMATION>();
        // SAFETY: a query of this process's own address space, written into `info`.
        let written = unsafe { VirtualQuery(start.as_ptr().cast(), &mut info, size) };
        assert_ne!(written, 0, "{}", std::io::Error::last_os_error());
        // SAFETY: the caller's: a view or an allocation of the test's own, from its base, which
        // nothing uses any more.
        let freed = unsafe {
            match info.Type == MEM_MAPPED {
                true => UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: start.as_ptr().cast(),
                }),
                false => VirtualFree(start.as_ptr().cast(), 0, MEM_RELEASE),
            }
        };
        assert_ne!(freed, 0, "{}", std::io::Error::last_os_error());
    }
}

/// The host's page size: `sysconf(_SC_PAGESIZE)` on unix hosts, `GetSystemInfo`'s `dwPageSize`
/// on Windows.
pub(super) fn page() -> u64 {
    #[cfg(unix)]
    // SAFETY: a name the call takes.
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) as u64 };
    #[cfg(windows)]
    let page = {
        use windows_sys::Win32::System::SystemInformation::{GetSystemInfo, SYSTEM_INFO};
        // SAFETY: an all-zero `SYSTEM_INFO` is valid, and the call fills it.
        let mut info: SYSTEM_INFO = unsafe { std::mem::zeroed() };
        // SAFETY: as above.
        unsafe { GetSystemInfo(&mut info) };
        u64::from(info.dwPageSize)
    };
    page
}

/// What the process may do with pages ([`protect`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Access {
    /// Nothing: a read or a write faults.
    None,
    /// Read them only.
    Read,
    /// Read and write them.
    ReadWrite,
}

/// Let the process do `access` with the `length` bytes at `start` from now on.
pub(super) fn protect(start: NonNull<u8>, length: usize, access: Access) {
    #[cfg(unix)]
    {
        let protection = match access {
            Access::None => libc::PROT_NONE,
            Access::Read => libc::PROT_READ,
            Access::ReadWrite => libc::PROT_READ | libc::PROT_WRITE,
        };
        // SAFETY: the caller's: pages of the test's own, whole.
        let done = unsafe { libc::mprotect(start.as_ptr().cast(), length, protection) };
        assert_eq!(done, 0, "{}", std::io::Error::last_os_error());
    }
    #[cfg(windows)]
    {
        use windows_sys::Win32::System::Memory::{
            VirtualProtect, PAGE_NOACCESS, PAGE_READONLY, PAGE_READWRITE,
        };
        let protection = match access {
            Access::None => PAGE_NOACCESS,
            Access::Read => PAGE_READONLY,
            Access::ReadWrite => PAGE_READWRITE,
        };
        let mut old = 0;
        // SAFETY: the caller's: pages of the test's own, whole, and `old` for the call to fill.
        let done = unsafe { VirtualProtect(start.as_ptr().cast(), length, protection, &mut old) };
        assert_ne!(done, 0, "{}", std::io::Error::last_os_error());
    }
}

/// The calling thread's CPU time so far: `CLOCK_THREAD_CPUTIME_ID` on unix hosts, its kernel and
/// user times on Windows (`GetThreadTimes`).
pub(super) fn thread_cpu() -> Duration {
    #[cfg(unix)]
    {
        let mut now = libc::timespec {
            tv_sec: 0,
            tv_nsec: 0,
        };
        // SAFETY: a clock every unix host the tests run on has, into a value of the call's type.
        unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut now) };
        Duration::new(now.tv_sec as u64, now.tv_nsec as u32)
    }
    #[cfg(windows)]
    {
        Duration::from_nanos(crate::this_thread::cpu_nanos().expect("the thread's times"))
    }
}

/// The process's CPU time so far, its user and system (kernel) time: `getrusage` on unix hosts,
/// `GetProcessTimes` on Windows.
pub(super) fn process_cpu() -> Duration {
    #[cfg(unix)]
    {
        // SAFETY: an all-zero `rusage` is valid, and `getrusage` writes the struct it is given.
        let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
        // SAFETY: as above.
        unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };
        let time =
            |time: libc::timeval| Duration::new(time.tv_sec as u64, time.tv_usec as u32 * 1000);
        time(usage.ru_utime) + time(usage.ru_stime)
    }
    #[cfg(windows)]
    {
        use windows_sys::Win32::Foundation::FILETIME;
        use windows_sys::Win32::System::Threading::{GetCurrentProcess, GetProcessTimes};
        let (mut created, mut exited) = (FILETIME::default(), FILETIME::default());
        let (mut kernel, mut user) = (FILETIME::default(), FILETIME::default());
        // SAFETY: the pseudo-handle of this process, and four valid `FILETIME`s to fill.
        let read = unsafe {
            GetProcessTimes(
                GetCurrentProcess(),
                &mut created,
                &mut exited,
                &mut kernel,
                &mut user,
            )
        };
        assert_ne!(read, 0, "{}", std::io::Error::last_os_error());
        // FILETIME counts 100 ns units.
        let ticks =
            |time: FILETIME| (u64::from(time.dwHighDateTime) << 32) | u64::from(time.dwLowDateTime);
        Duration::from_nanos((ticks(kernel) + ticks(user)) * 100)
    }
}
