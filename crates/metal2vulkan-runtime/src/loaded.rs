//! How many shared objects the process has loaded. The exit guard registers its hook again at a
//! device made after a load: a driver or layer loaded again (the Vulkan loader unloads them with
//! the last instance) makes its statics again, after the hook's earlier registrations.

/// The shared objects the process has loaded since it started, unloaded ones included: the C
/// library's count (`dlpi_adds`). `None` where it keeps none.
#[cfg(any(target_os = "linux", target_os = "android"))]
pub(crate) fn count() -> Option<u64> {
    use std::ffi::{c_int, c_void};
    /// Read the count from the first object, and stop.
    unsafe extern "C" fn first(
        info: *mut libc::dl_phdr_info,
        size: libc::size_t,
        data: *mut c_void,
    ) -> c_int {
        let covered =
            std::mem::offset_of!(libc::dl_phdr_info, dlpi_adds) + std::mem::size_of::<u64>();
        if size >= covered {
            // SAFETY: the C library passes an object of `size` bytes, which covers the field,
            // and `data` is the caller's `Option<u64>`.
            unsafe { *data.cast::<Option<u64>>() = Some((*info).dlpi_adds) };
        }
        1
    }
    let mut adds: Option<u64> = None;
    // SAFETY: the callback only reads its object and writes `adds`, which outlives the call.
    unsafe { libc::dl_iterate_phdr(Some(first), (&raw mut adds).cast()) };
    adds.map(|adds| adds + forced())
}

/// The images the process has loaded since this was first asked, unloaded ones included (dyld
/// calls the callback once for each image loaded when it is registered, and for each one loaded
/// later).
#[cfg(target_vendor = "apple")]
pub(crate) fn count() -> Option<u64> {
    use std::ffi::c_void;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Once;
    static ADDS: AtomicU64 = AtomicU64::new(0);
    static WATCH: Once = Once::new();
    /// The image's header and slide, unread.
    extern "C" fn added(_: *const c_void, _: isize) {
        ADDS.fetch_add(1, Ordering::SeqCst);
    }
    unsafe extern "C" {
        fn _dyld_register_func_for_add_image(func: extern "C" fn(*const c_void, isize));
    }
    // SAFETY: the callback only counts; dyld keeps it for the process's life.
    WATCH.call_once(|| unsafe { _dyld_register_func_for_add_image(added) });
    Some(ADDS.load(Ordering::SeqCst) + forced())
}

/// The loads tests forced ([`force`]).
#[cfg(test)]
static FORCED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Tests: count one load more, as a shared object loaded now would.
#[cfg(test)]
pub(crate) fn force() {
    FORCED.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
}

/// The loads tests forced so far ([`force`]), counted with the host's.
#[cfg(all(
    test,
    any(target_os = "linux", target_os = "android", target_vendor = "apple")
))]
pub(crate) fn forced() -> u64 {
    FORCED.load(std::sync::atomic::Ordering::SeqCst)
}

/// No load is forced outside tests.
#[cfg(all(
    not(test),
    any(target_os = "linux", target_os = "android", target_vendor = "apple")
))]
const fn forced() -> u64 {
    0
}

/// No count on other hosts: the exit guard registers its hook at every device there.
#[cfg(not(any(target_os = "linux", target_os = "android", target_vendor = "apple")))]
pub(crate) fn count() -> Option<u64> {
    None
}
