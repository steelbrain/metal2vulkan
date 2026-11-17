//! The user-space instructions the calling thread retires, read from the kernel's performance
//! counters: what the matrix's `instructions` records (CPU instructions per draw and per dispatch,
//! docs/VALIDATION.md, "The matrix") are measured with.
//!
//! [`InstructionCounter`] counts one thread, in user space alone. That is the form an unprivileged
//! process may open at `perf_event_paranoid` 2 (the lab's Linux nodes all run at 2): a counter
//! that also counts the kernel is refused there.
//!
//! A hybrid x86 CPU has one performance-monitoring unit per core type (`cpu_core`, `cpu_atom`, and
//! on some parts `cpu_lowpower`), and the generic hardware event counts on one of them only: a
//! thread that ran on another core type loses the instructions it retired there (an i9-13900H read
//! 16.0 million of a 25.6-million-instruction loop moved over its CPUs; a Core Ultra 9 275HX read
//! 0 of one that stayed on its efficiency cores). So where the kernel lists such units, the
//! counter opens the event on each and sums them.
//!
//! Only Linux offers the counters (read here on its little-endian targets): elsewhere
//! [`InstructionCounter::new`] answers [`io::ErrorKind::Unsupported`], and a test records no
//! `instructions` line there.

use std::io;
use std::marker::PhantomData;

/// User-space instructions retired by the thread that made it, summed over every core type.
///
/// It counts the thread that called [`InstructionCounter::new`], so it is neither `Send` nor
/// `Sync`. Each [`InstructionCounter::measure`] counts its own work alone.
pub struct InstructionCounter {
    #[cfg(all(target_os = "linux", target_endian = "little"))]
    events: Vec<std::fs::File>,
    #[cfg(not(all(target_os = "linux", target_endian = "little")))]
    never: std::convert::Infallible,
    _thread: PhantomData<*const ()>,
}

impl InstructionCounter {
    /// Opens the counters for the calling thread. An error is the kernel's refusal (for one,
    /// `perf_event_paranoid` above 2, or no hardware counters in a virtual machine), or
    /// [`io::ErrorKind::Unsupported`] off Linux.
    pub fn new() -> io::Result<Self> {
        #[cfg(all(target_os = "linux", target_endian = "little"))]
        {
            let events = linux::core_events()?
                .into_iter()
                .map(linux::open)
                .collect::<io::Result<Vec<_>>>()?;
            Ok(Self {
                events,
                _thread: PhantomData,
            })
        }
        #[cfg(not(all(target_os = "linux", target_endian = "little")))]
        {
            Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "user-space instruction counters are read through Linux's perf_event_open alone",
            ))
        }
    }

    /// Runs `work` on this thread and returns its result with the user-space instructions the
    /// thread retired meanwhile. The count includes a few instructions of the counters' own
    /// switching, the same in every call.
    pub fn measure<R>(&mut self, work: impl FnOnce() -> R) -> io::Result<(R, u64)> {
        #[cfg(all(target_os = "linux", target_endian = "little"))]
        {
            for event in &self.events {
                linux::control(event, linux::IOC_RESET)?;
            }
            for event in &self.events {
                linux::control(event, linux::IOC_ENABLE)?;
            }
            let result = work();
            for event in &self.events {
                linux::control(event, linux::IOC_DISABLE)?;
            }
            let mut retired = 0;
            for event in &self.events {
                retired += linux::read(event)?;
            }
            Ok((result, retired))
        }
        #[cfg(not(all(target_os = "linux", target_endian = "little")))]
        {
            let _ = work;
            match self.never {}
        }
    }
}

#[cfg(all(target_os = "linux", target_endian = "little"))]
mod linux {
    use std::fs::{self, File};
    use std::io::{self, Read};
    use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};

    /// `PERF_TYPE_HARDWARE` and its `PERF_COUNT_HW_INSTRUCTIONS`.
    const TYPE_HARDWARE: u32 = 0;
    const HW_INSTRUCTIONS: u64 = 1;
    /// `perf_event_attr`'s bit fields `disabled`, `exclude_kernel` and `exclude_hv`, in a
    /// little-endian target's bit order.
    const DISABLED: u64 = 1 << 0;
    const EXCLUDE_KERNEL: u64 = 1 << 5;
    const EXCLUDE_HV: u64 = 1 << 6;
    /// `PERF_FLAG_FD_CLOEXEC`: the descriptor does not pass to a child the test starts.
    const FLAG_FD_CLOEXEC: libc::c_ulong = 1 << 3;
    /// `PERF_EVENT_IOC_ENABLE`, `_DISABLE` and `_RESET` (`_IO('$', n)`).
    pub(super) const IOC_ENABLE: libc::c_ulong = 0x2400;
    pub(super) const IOC_DISABLE: libc::c_ulong = 0x2401;
    pub(super) const IOC_RESET: libc::c_ulong = 0x2403;
    /// Where the kernel lists its performance-monitoring units, each with its `type` number.
    const UNITS: &str = "/sys/bus/event_source/devices";

    /// `struct perf_event_attr` as the kernel first defined it (`PERF_ATTR_SIZE_VER0`), which
    /// every later kernel still takes: its `type`, `size`, `config` and bit-field word, at their
    /// byte offsets, everything else zero.
    const ATTR_SIZE: usize = 64;
    const ATTR_TYPE: usize = 0;
    const ATTR_SIZE_FIELD: usize = 4;
    const ATTR_CONFIG: usize = 8;
    const ATTR_FLAGS: usize = 40;

    /// The instructions event's `config` for each core type: on a hybrid x86 CPU (units named
    /// `cpu_*`), the event on that unit (its `type` in bits 32 to 63); otherwise the generic one.
    pub(super) fn core_events() -> io::Result<Vec<u64>> {
        let mut events = Vec::new();
        if let Ok(units) = fs::read_dir(UNITS) {
            for unit in units {
                let unit = unit?;
                if !unit.file_name().to_string_lossy().starts_with("cpu_") {
                    continue;
                }
                let kind = fs::read_to_string(unit.path().join("type"))?;
                let kind: u64 = kind.trim().parse().map_err(|error| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("{}/type: {error}", unit.path().display()),
                    )
                })?;
                events.push((kind << 32) | HW_INSTRUCTIONS);
            }
        }
        if events.is_empty() {
            events.push(HW_INSTRUCTIONS);
        }
        Ok(events)
    }

    /// The calling thread's user-space instructions event of `config`, disabled until enabled.
    pub(super) fn open(config: u64) -> io::Result<File> {
        let mut attr = [0u8; ATTR_SIZE];
        attr[ATTR_TYPE..ATTR_TYPE + 4].copy_from_slice(&TYPE_HARDWARE.to_ne_bytes());
        attr[ATTR_SIZE_FIELD..ATTR_SIZE_FIELD + 4]
            .copy_from_slice(&(ATTR_SIZE as u32).to_ne_bytes());
        attr[ATTR_CONFIG..ATTR_CONFIG + 8].copy_from_slice(&config.to_ne_bytes());
        attr[ATTR_FLAGS..ATTR_FLAGS + 8]
            .copy_from_slice(&(DISABLED | EXCLUDE_KERNEL | EXCLUDE_HV).to_ne_bytes());
        let (thread, any_cpu, no_group): (libc::pid_t, libc::c_int, libc::c_int) = (0, -1, -1);
        // SAFETY: `attr` is a valid `perf_event_attr` of the size it states and outlives the
        // call; pid 0 and cpu -1 name the calling thread on any CPU, with no group.
        let fd = unsafe {
            libc::syscall(
                libc::SYS_perf_event_open,
                attr.as_ptr(),
                thread,
                any_cpu,
                no_group,
                FLAG_FD_CLOEXEC,
            )
        };
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: the kernel just returned this descriptor, and nothing else owns it.
        Ok(File::from(unsafe {
            OwnedFd::from_raw_fd(fd as libc::c_int)
        }))
    }

    /// One `ioctl` without an argument on `event`.
    pub(super) fn control(event: &File, request: libc::c_ulong) -> io::Result<()> {
        // SAFETY: an event descriptor this module opened, and a request that takes no argument.
        if unsafe { libc::ioctl(event.as_raw_fd(), request as _, 0) } < 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }

    /// The count of `event` (read format 0: the value alone).
    pub(super) fn read(mut event: &File) -> io::Result<u64> {
        let mut value = [0; 8];
        event.read_exact(&mut value)?;
        Ok(u64::from_ne_bytes(value))
    }
}

#[cfg(all(test, target_os = "linux", target_arch = "x86_64"))]
mod tests {
    use super::InstructionCounter;
    use std::path::Path;

    /// Iterations of the fixed loop on each CPU.
    const ITERATIONS: u64 = 200_000;

    /// Whether this host lets an unprivileged process count its own user-space instructions: a
    /// hardware unit the kernel lists, and `perf_event_paranoid` at 2 or below.
    fn counters_offered() -> bool {
        let paranoid = std::fs::read_to_string("/proc/sys/kernel/perf_event_paranoid")
            .ok()
            .and_then(|level| level.trim().parse::<i32>().ok());
        let units = Path::new("/sys/bus/event_source/devices");
        let unit = units.join("cpu").exists()
            || std::fs::read_dir(units).is_ok_and(|entries| {
                entries
                    .flatten()
                    .any(|entry| entry.file_name().to_string_lossy().starts_with("cpu_"))
            });
        unit && paranoid.is_some_and(|level| level <= 2)
    }

    /// The CPUs this thread may run on.
    fn allowed_cpus() -> Vec<usize> {
        // SAFETY: a zeroed `cpu_set_t` is an empty set, filled in by the call.
        let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
        // SAFETY: `set` is a valid `cpu_set_t` of the size passed.
        let got = unsafe { libc::sched_getaffinity(0, std::mem::size_of_val(&set), &mut set) };
        assert_eq!(
            got,
            0,
            "sched_getaffinity: {}",
            std::io::Error::last_os_error()
        );
        (0..libc::CPU_SETSIZE as usize)
            // SAFETY: `cpu` is below `CPU_SETSIZE`.
            .filter(|&cpu| unsafe { libc::CPU_ISSET(cpu, &set) })
            .collect()
    }

    /// Pins this thread to `cpus`.
    fn pin(cpus: &[usize]) {
        // SAFETY: a zeroed `cpu_set_t` is an empty set.
        let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
        for &cpu in cpus {
            // SAFETY: `cpu` came from `allowed_cpus`, so it is below `CPU_SETSIZE`.
            unsafe { libc::CPU_SET(cpu, &mut set) };
        }
        // SAFETY: `set` is a valid `cpu_set_t` of the size passed.
        let got = unsafe { libc::sched_setaffinity(0, std::mem::size_of_val(&set), &set) };
        assert_eq!(
            got,
            0,
            "sched_setaffinity: {}",
            std::io::Error::last_os_error()
        );
    }

    /// Exactly two instructions per iteration: a decrement and a branch.
    fn fixed_loop(iterations: u64) {
        // SAFETY: the loop touches only its own register.
        unsafe {
            std::arch::asm!(
                "2:",
                "dec {n}",
                "jnz 2b",
                n = inout(reg) iterations => _,
                options(nomem, nostack),
            );
        }
    }

    /// The fixed loop counts its two instructions per iteration although it runs on every CPU
    /// this thread may use in turn, every core type of a hybrid CPU among them: a counter that
    /// counted one core type alone would come up short by the others' share. The record it prints
    /// follows the counter itself across commits in the matrix.
    #[test]
    fn a_fixed_loop_counts_its_instructions_on_every_core_type() {
        if !counters_offered() {
            let refusal = InstructionCounter::new().err();
            assert!(
                refusal.is_some(),
                "the host offers no counters, yet a counter opened"
            );
            println!("INSTRUCTIONS-UNAVAILABLE: {}", refusal.unwrap());
            return;
        }
        let cpus = allowed_cpus();
        let mut counter = InstructionCounter::new().expect("the host offers counters");
        let run = |counter: &mut InstructionCounter| {
            counter
                .measure(|| {
                    for &cpu in &cpus {
                        pin(&[cpu]);
                        fixed_loop(ITERATIONS);
                    }
                })
                .expect("reading the counters")
                .1
        };
        // The first round loads what the later ones reuse.
        run(&mut counter);
        let counts = [run(&mut counter), run(&mut counter)];
        pin(&cpus);
        let expected = 2 * ITERATIONS * cpus.len() as u64;
        for count in counts {
            // The pinning between the loops adds its own instructions, a few hundred per CPU.
            assert!(
                (expected..expected + expected / 200).contains(&count),
                "{count} user-space instructions counted for {expected} on {} CPUs",
                cpus.len()
            );
        }
        println!(
            "M2V-MATRIX instructions instruction_counter/per_1000_iterations {}",
            counts[0] * 1000 / (ITERATIONS * cpus.len() as u64)
        );
    }
}
