//! Windows memory facts used by no-copy buffer imports.
//! The host's page size and allocation granularity are different: Vulkan host-pointer
//! imports need the former, while a fresh VirtualAlloc reservation uses the latter.

#[repr(C)]
struct SystemInfo {
    architecture: u16,
    reserved: u16,
    page_size: u32,
    minimum_address: usize,
    maximum_address: usize,
    processor_mask: usize,
    processor_count: u32,
    processor_type: u32,
    allocation_granularity: u32,
    processor_level: u16,
    processor_revision: u16,
}

#[link(name = "kernel32")]
unsafe extern "system" {
    fn GetSystemInfo(info: *mut SystemInfo);
}

/// The Windows page size used to validate a caller's Metal no-copy buffer; no allocation.
pub(super) fn host_page() -> Option<u64> {
    let mut info = std::mem::MaybeUninit::<SystemInfo>::uninit();
    // SAFETY: SystemInfo has the Win32 SYSTEM_INFO layout and is writable for the call.
    unsafe { GetSystemInfo(info.as_mut_ptr()) };
    // SAFETY: GetSystemInfo always initializes the complete structure.
    let info = unsafe { info.assume_init() };
    Some(u64::from(info.page_size))
}

#[cfg(test)]
mod tests {
    #[test]
    fn page_size_is_not_allocation_granularity() {
        assert_eq!(super::host_page(), Some(4096));
    }
}

/// The Win64 MEMORY_BASIC_INFORMATION layout, including its partition identifier.
#[repr(C)]
struct MemoryInfo {
    base: usize,
    allocation_base: usize,
    allocation_protection: u32,
    partition: u16,
    length: usize,
    state: u32,
    protection: u32,
    kind: u32,
}

#[link(name = "kernel32")]
unsafe extern "system" {
    fn VirtualQuery(address: *const std::ffi::c_void, info: *mut MemoryInfo, size: usize) -> usize;
    fn VirtualAlloc(
        base: *mut std::ffi::c_void,
        size: usize,
        kind: u32,
        protect: u32,
    ) -> *mut std::ffi::c_void;
    fn VirtualFree(base: *mut std::ffi::c_void, size: usize, kind: u32) -> i32;
}

/// Whether every byte has the requested CPU access, as VirtualQuery reports now.
/// A reservation, gap, guard page or inaccessible page is not valid import/copy memory.
pub(super) fn permitted(start: u64, end: u64, write: bool) -> bool {
    permitted_counted(start, end, write, &mut 0)
}

/// The same query with an exact region count for the runtime's import diagnostics.
pub(super) fn permitted_counted(start: u64, end: u64, write: bool, asked: &mut u64) -> bool {
    let mut at = start;
    while at < end {
        let mut info = std::mem::MaybeUninit::<MemoryInfo>::uninit();
        // SAFETY: VirtualQuery only inspects the address; it writes the complete local structure.
        *asked += 1;
        let size = unsafe {
            VirtualQuery(
                at as *const _,
                info.as_mut_ptr(),
                std::mem::size_of::<MemoryInfo>(),
            )
        };
        if size != std::mem::size_of::<MemoryInfo>() {
            return false;
        }
        // SAFETY: the query returned the full structure's size.
        let info = unsafe { info.assume_init() };
        if info.state != 0x1000 || info.protection & 0x100 != 0 {
            return false;
        }
        let access = info.protection & 0xff;
        let allowed = if write {
            matches!(access, 0x04 | 0x08 | 0x40 | 0x80)
        } else {
            matches!(access, 0x02 | 0x04 | 0x08 | 0x20 | 0x40 | 0x80)
        };
        if !allowed {
            return false;
        }
        at = (info.base + info.length) as u64;
    }
    true
}

/// Fresh read/write zero pages owned by the runtime, aligned by Windows VirtualAlloc.
pub(super) fn map_pages(length: u64) -> Option<std::ptr::NonNull<u8>> {
    let length = usize::try_from(length).ok()?;
    // SAFETY: the kernel chooses a new private allocation; no existing memory is replaced.
    std::ptr::NonNull::new(
        unsafe { VirtualAlloc(std::ptr::null_mut(), length, 0x3000, 0x04) }.cast(),
    )
}

/// Release a whole allocation from map_pages after its last Vulkan import is freed.
pub(super) fn unmap_pages(pointer: std::ptr::NonNull<u8>) {
    // SAFETY: the caller owns this allocation and no remaining GPU or CPU access reaches it.
    let released = unsafe { VirtualFree(pointer.as_ptr().cast(), 0, 0x8000) };
    assert_ne!(
        released,
        0,
        "VirtualFree failed: {}",
        std::io::Error::last_os_error()
    );
}

#[cfg(test)]
mod memory_tests {
    use super::*;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn VirtualProtect(
            base: *mut std::ffi::c_void,
            size: usize,
            access: u32,
            old: *mut u32,
        ) -> i32;
    }

    /// The copied-host-buffer path must write back writable pages and reject a read-only page.
    #[test]
    fn permissions_follow_the_actual_windows_mapping() {
        let pointer = map_pages(8192).unwrap();
        let first = pointer.as_ptr() as u64;
        assert!(permitted(first, first + 8192, true));
        let mut old = 0;
        // SAFETY: the test exclusively owns this second page and the out parameter.
        assert_ne!(
            unsafe { VirtualProtect((first + 4096) as *mut _, 4096, 2, &mut old) },
            0
        );
        assert!(permitted(first, first + 8192, false));
        assert!(!permitted(first, first + 8192, true));
        assert!(permitted(first, first + 4096, true));
        unmap_pages(pointer);
        assert!(!permitted(first, first + 8192, false));
    }
}
