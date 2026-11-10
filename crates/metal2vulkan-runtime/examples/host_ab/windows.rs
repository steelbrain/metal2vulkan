//! Windows CPU clocks and anonymous pages for the same host A/B workloads.

use std::ffi::c_void;
use std::time::Duration;

/// FILETIME stores CPU usage in 100 ns ticks, split into two DWORDs.
#[repr(C)]
#[derive(Default)]
struct FileTime {
    low: u32,
    high: u32,
}

#[link(name = "kernel32")]
unsafe extern "system" {
    fn GetCurrentProcess() -> *mut c_void;
    fn GetCurrentThread() -> *mut c_void;
    fn GetProcessTimes(
        handle: *mut c_void,
        created: *mut FileTime,
        exited: *mut FileTime,
        kernel: *mut FileTime,
        user: *mut FileTime,
    ) -> i32;
    fn GetThreadTimes(
        handle: *mut c_void,
        created: *mut FileTime,
        exited: *mut FileTime,
        kernel: *mut FileTime,
        user: *mut FileTime,
    ) -> i32;
    fn VirtualAlloc(address: *mut c_void, size: usize, kind: u32, protection: u32) -> *mut c_void;
    fn VirtualFree(address: *mut c_void, size: usize, kind: u32) -> i32;
}

/// Read user plus kernel CPU time; wall time would hide scheduling delays.
fn cpu_time(thread: bool) -> Duration {
    let (mut created, mut exited, mut kernel, mut user) = (
        FileTime::default(),
        FileTime::default(),
        FileTime::default(),
        FileTime::default(),
    );
    // SAFETY: pseudo-handles need no closing; all four outputs are initialized,
    // correctly sized FILETIMEs that remain live for the synchronous call.
    let ok = unsafe {
        if thread {
            GetThreadTimes(
                GetCurrentThread(),
                &mut created,
                &mut exited,
                &mut kernel,
                &mut user,
            )
        } else {
            GetProcessTimes(
                GetCurrentProcess(),
                &mut created,
                &mut exited,
                &mut kernel,
                &mut user,
            )
        }
    };
    assert_ne!(ok, 0, "CPU clock: {}", std::io::Error::last_os_error());
    let ticks = |value: FileTime| u64::from(value.high) << 32 | u64::from(value.low);
    let total = ticks(kernel) + ticks(user);
    Duration::new(total / 10_000_000, (total % 10_000_000) as u32 * 100)
}

pub fn process_cpu() -> Duration {
    cpu_time(false)
}
pub fn thread_cpu() -> Duration {
    cpu_time(true)
}

/// Reserve and commit zero-filled pages. The returned region is page-aligned,
/// remains valid until `release`, and can back a no-copy GPU allocation.
pub fn allocate(size: usize) -> *mut u8 {
    // SAFETY: null requests a new address range; MEM_RESERVE | MEM_COMMIT with
    // PAGE_READWRITE commits exactly this process's anonymous pages.
    let pointer = unsafe { VirtualAlloc(std::ptr::null_mut(), size, 0x3000, 0x04) };
    assert!(
        !pointer.is_null(),
        "VirtualAlloc: {}",
        std::io::Error::last_os_error()
    );
    pointer.cast()
}

/// # Safety
/// `pointer` must be an allocation returned by `allocate`, with no live users.
pub unsafe fn release(pointer: *mut u8) {
    // SAFETY: the caller owns the original base address. MEM_RELEASE requires
    // size zero and releases the entire reservation, including committed pages.
    let ok = unsafe { VirtualFree(pointer.cast(), 0, 0x8000) };
    assert_ne!(ok, 0, "VirtualFree: {}", std::io::Error::last_os_error());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anonymous_pages_are_zeroed_writable_and_releasable() {
        let pointer = allocate(8192);
        // SAFETY: both pages belong to this test until the release below.
        unsafe {
            assert!(std::slice::from_raw_parts(pointer, 8192)
                .iter()
                .all(|byte| *byte == 0));
            pointer.add(8191).write(0x37);
            assert_eq!(pointer.add(8191).read(), 0x37);
            release(pointer);
        }
    }

    #[test]
    fn cpu_clocks_are_monotonic() {
        let before = (process_cpu(), thread_cpu());
        for value in 0..100_000 {
            std::hint::black_box(value);
        }
        assert!(process_cpu() >= before.0);
        assert!(thread_cpu() >= before.1);
    }
}
