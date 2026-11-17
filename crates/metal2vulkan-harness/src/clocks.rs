//! The process's CPU clock.

/// The whole process's CPU time, every thread of it, user and system, in nanoseconds; `None`
/// where the host does not answer it (and on hosts other than unix, whose process times count in
/// scheduler ticks, too coarse for one command buffer).
#[cfg(unix)]
pub fn process_cpu_nanos() -> Option<u64> {
    let mut when = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `clock_gettime` writes the one `timespec` it is given and reads nothing else.
    let answered = unsafe { libc::clock_gettime(libc::CLOCK_PROCESS_CPUTIME_ID, &mut when) };
    if answered != 0 {
        return None;
    }
    let seconds = u64::try_from(when.tv_sec).ok()?;
    let nanos = u64::try_from(when.tv_nsec).ok()?;
    seconds.checked_mul(1_000_000_000)?.checked_add(nanos)
}

/// See the unix form: no host clock fine enough here.
#[cfg(not(unix))]
pub fn process_cpu_nanos() -> Option<u64> {
    None
}
