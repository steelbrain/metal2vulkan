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
//! [`ProcessInstructions`] counts the whole process in user space: the thread that opens it and
//! every thread started after, through one inherited counter per core type. That is what the
//! harness records per timed run, so work a runtime moves onto its own threads still counts.
//! [`ThreadInstructions`] counts named threads alone, by their OS ids: the harness's own thread and
//! the threads a runtime says it started, apart from the threads a driver starts for itself.
//!
//! [`ProcessEvent`] counts the whole process in user space as [`ProcessInstructions`] does, for a
//! raw event of one unit named the way `perf` names it: what the runtime's lock budget counts
//! locked instructions with.
//!
//! Only Linux offers the counters (read here on its little-endian targets): elsewhere
//! [`InstructionCounter::new`], [`ProcessInstructions::new`], [`ThreadInstructions::new`] and
//! [`ProcessEvent::new`] answer [`io::ErrorKind::Unsupported`], and nothing records an
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

/// User-space instructions retired by the whole process since [`ProcessInstructions::new`]: the
/// thread that opened it and every thread started after it, summed over every core type.
///
/// A thread started before the open does not count, so a program opens it before it starts its
/// threads (a device's, its driver's). Two [`ProcessInstructions::read`]s bracket a span of work
/// in the whole process, whichever thread did it.
pub struct ProcessInstructions {
    #[cfg(all(target_os = "linux", target_endian = "little"))]
    events: Vec<std::fs::File>,
    #[cfg(not(all(target_os = "linux", target_endian = "little")))]
    never: std::convert::Infallible,
}

impl ProcessInstructions {
    /// Opens the counters, counting from now. An error is the kernel's refusal (for one,
    /// `perf_event_paranoid` above 2, or no hardware counters in a virtual machine), or
    /// [`io::ErrorKind::Unsupported`] off Linux.
    pub fn new() -> io::Result<Self> {
        #[cfg(all(target_os = "linux", target_endian = "little"))]
        {
            let events = linux::core_events()?
                .into_iter()
                .map(linux::open_inherited)
                .collect::<io::Result<Vec<_>>>()?;
            Ok(Self { events })
        }
        #[cfg(not(all(target_os = "linux", target_endian = "little")))]
        {
            Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "user-space instruction counters are read through Linux's perf_event_open alone",
            ))
        }
    }

    /// The user-space instructions the process retired since the open: its threads still running
    /// and those that ended.
    pub fn read(&self) -> io::Result<u64> {
        #[cfg(all(target_os = "linux", target_endian = "little"))]
        {
            let mut retired = 0;
            for event in &self.events {
                retired += linux::read(event)?;
            }
            Ok(retired)
        }
        #[cfg(not(all(target_os = "linux", target_endian = "little")))]
        {
            match self.never {}
        }
    }
}

/// User-space instructions retired by the threads of this process whose OS ids were given to
/// [`ThreadInstructions::new`], each alone (not the threads it starts), summed over every core
/// type, since the open.
pub struct ThreadInstructions {
    #[cfg(all(target_os = "linux", target_endian = "little"))]
    events: Vec<std::fs::File>,
    #[cfg(not(all(target_os = "linux", target_endian = "little")))]
    never: std::convert::Infallible,
}

impl ThreadInstructions {
    /// Opens the counters on threads `ids` (Linux thread ids, as [`this_thread`] gives), counting
    /// from now. A thread that already ended is skipped: it retires nothing more. An error is the
    /// kernel's refusal, or [`io::ErrorKind::Unsupported`] off Linux.
    pub fn new(ids: &[u64]) -> io::Result<Self> {
        #[cfg(all(target_os = "linux", target_endian = "little"))]
        {
            let configs = linux::core_events()?;
            let mut events = Vec::with_capacity(ids.len() * configs.len());
            for &id in ids {
                let thread = libc::pid_t::try_from(id).map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidInput, format!("thread id {id}"))
                })?;
                for &config in &configs {
                    match linux::open_thread(thread, config) {
                        Ok(event) => events.push(event),
                        Err(error) if error.raw_os_error() == Some(libc::ESRCH) => break,
                        Err(error) => return Err(error),
                    }
                }
            }
            Ok(Self { events })
        }
        #[cfg(not(all(target_os = "linux", target_endian = "little")))]
        {
            let _ = ids;
            Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "user-space instruction counters are read through Linux's perf_event_open alone",
            ))
        }
    }

    /// The user-space instructions the threads retired since the open.
    pub fn read(&self) -> io::Result<u64> {
        #[cfg(all(target_os = "linux", target_endian = "little"))]
        {
            let mut retired = 0;
            for event in &self.events {
                retired += linux::read(event)?;
            }
            Ok(retired)
        }
        #[cfg(not(all(target_os = "linux", target_endian = "little")))]
        {
            match self.never {}
        }
    }
}

/// A raw event the whole process retires in user space, on one performance-monitoring unit: the
/// thread that opens it and every thread started after it, as [`ProcessInstructions`] counts
/// instructions.
///
/// [`ProcessEvent::new`] takes the event as `perf` names a raw one: `<unit>/r<hex>`, the raw
/// `config` `<hex>` on the unit the kernel lists as `<unit>` (`cpu_core/r21d0`: an Intel
/// performance core's `MEM_INST_RETIRED.LOCK_LOADS`, one per locked instruction), or `r<hex>` on
/// the unit `cpu` (`r1e25`: a Zen 4 core's locked instructions, event 0x25 with unit mask 0x1e). A
/// trailing `/`, `/u` or `:u` changes nothing: the count is user space's alone. The unit of one
/// core type of a hybrid CPU counts on that type's CPUs alone ([`ProcessEvent::cpus`]), so a
/// program pins itself to them ([`pin_this_thread`]) before it opens the counter and starts its
/// threads.
pub struct ProcessEvent {
    #[cfg(all(target_os = "linux", target_endian = "little"))]
    event: std::fs::File,
    #[cfg(not(all(target_os = "linux", target_endian = "little")))]
    never: std::convert::Infallible,
    cpus: Vec<usize>,
}

impl ProcessEvent {
    /// Opens the event `name`, counting from now. An error is a name of another form
    /// ([`io::ErrorKind::InvalidInput`]), a unit the kernel does not list, the kernel's refusal, or
    /// [`io::ErrorKind::Unsupported`] off Linux.
    pub fn new(name: &str) -> io::Result<Self> {
        let (unit, config) = raw_event(name)?;
        #[cfg(all(target_os = "linux", target_endian = "little"))]
        {
            let (kind, cpus) = linux::unit(unit)?;
            let event = linux::open_raw_inherited(kind, config)?;
            Ok(Self { event, cpus })
        }
        #[cfg(not(all(target_os = "linux", target_endian = "little")))]
        {
            let _ = (unit, config);
            Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "user-space event counters are read through Linux's perf_event_open alone",
            ))
        }
    }

    /// The events the process retired since the open: its threads still running and those that
    /// ended.
    pub fn read(&self) -> io::Result<u64> {
        #[cfg(all(target_os = "linux", target_endian = "little"))]
        {
            linux::read(&self.event)
        }
        #[cfg(not(all(target_os = "linux", target_endian = "little")))]
        {
            match self.never {}
        }
    }

    /// The CPUs the unit counts on, where the kernel lists them (a hybrid CPU's core type); empty
    /// where it counts on every CPU.
    pub fn cpus(&self) -> &[usize] {
        &self.cpus
    }
}

/// The unit and raw `config` of an event named `<unit>/r<hex>` or `r<hex>` (on the unit `cpu`),
/// with an optional trailing `/`, `/u` or `:u`.
fn raw_event(name: &str) -> io::Result<(&str, u64)> {
    let invalid = || {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("{name:?} is not a raw event (`<unit>/r<hex>` or `r<hex>`)"),
        )
    };
    let bare = ["/u", ":u", "/"]
        .iter()
        .find_map(|suffix| name.strip_suffix(suffix))
        .unwrap_or(name);
    let (unit, event) = match bare.split_once('/') {
        Some((unit, event)) if !unit.is_empty() => (unit, event),
        Some(_) => return Err(invalid()),
        None => ("cpu", bare),
    };
    let hex = event.strip_prefix('r').ok_or_else(invalid)?;
    let config = u64::from_str_radix(hex, 16).map_err(|_| invalid())?;
    Ok((unit, config))
}

/// The CPUs of a kernel CPU list (`0-11,16`), as Linux lists a unit's CPUs: read only where the
/// counters are Linux's.
#[cfg(all(target_os = "linux", target_endian = "little"))]
fn cpu_list(text: &str) -> io::Result<Vec<usize>> {
    let invalid = || {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{text:?} is not a CPU list"),
        )
    };
    let mut cpus = Vec::new();
    for part in text.trim().split(',').filter(|part| !part.is_empty()) {
        let (first, last) = part.split_once('-').unwrap_or((part, part));
        let first: usize = first.parse().map_err(|_| invalid())?;
        let last: usize = last.parse().map_err(|_| invalid())?;
        if last < first {
            return Err(invalid());
        }
        cpus.extend(first..=last);
    }
    Ok(cpus)
}

/// Pins the calling thread to `cpus`, and so every thread it starts after. An error is the
/// kernel's refusal, or [`io::ErrorKind::Unsupported`] off Linux.
pub fn pin_this_thread(cpus: &[usize]) -> io::Result<()> {
    #[cfg(target_os = "linux")]
    {
        // SAFETY: a zeroed `cpu_set_t` is an empty set.
        let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
        for &cpu in cpus {
            if cpu >= libc::CPU_SETSIZE as usize {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("CPU {cpu} is past the set's size"),
                ));
            }
            // SAFETY: `cpu` is below `CPU_SETSIZE`.
            unsafe { libc::CPU_SET(cpu, &mut set) };
        }
        // SAFETY: `set` is a valid `cpu_set_t` of the size passed.
        if unsafe { libc::sched_setaffinity(0, std::mem::size_of_val(&set), &set) } != 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = cpus;
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "threads are pinned through Linux's sched_setaffinity alone",
        ))
    }
}

/// The calling thread's Linux thread id (`gettid`), which [`ThreadInstructions::new`] takes; `None`
/// off Linux.
pub fn this_thread() -> Option<u64> {
    #[cfg(all(target_os = "linux", target_endian = "little"))]
    {
        // SAFETY: `gettid` takes no argument and cannot fail.
        u64::try_from(unsafe { libc::syscall(libc::SYS_gettid) }).ok()
    }
    #[cfg(not(all(target_os = "linux", target_endian = "little")))]
    {
        None
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
    /// `perf_event_attr`'s bit fields `disabled`, `inherit`, `exclude_kernel` and `exclude_hv`, in
    /// a little-endian target's bit order.
    const DISABLED: u64 = 1 << 0;
    const INHERIT: u64 = 1 << 1;
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

    /// The `type` number and the CPUs (empty: every CPU) of the unit the kernel lists as `unit`.
    pub(super) fn unit(unit: &str) -> io::Result<(u32, Vec<usize>)> {
        let dir = std::path::Path::new(UNITS).join(unit);
        let kind = fs::read_to_string(dir.join("type"))
            .map_err(|error| io::Error::new(error.kind(), format!("{}: {error}", dir.display())))?;
        let kind = kind.trim().parse().map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{}/type: {error}", dir.display()),
            )
        })?;
        let cpus = match fs::read_to_string(dir.join("cpus")) {
            Ok(list) => super::cpu_list(&list)?,
            Err(error) if error.kind() == io::ErrorKind::NotFound => Vec::new(),
            Err(error) => return Err(error),
        };
        Ok((kind, cpus))
    }

    /// The raw user-space event `config` of the unit of `type` `kind`, of the calling thread and
    /// every thread it starts after, counting from now.
    pub(super) fn open_raw_inherited(kind: u32, config: u64) -> io::Result<File> {
        open_on(0, kind, config, INHERIT | EXCLUDE_KERNEL | EXCLUDE_HV)
    }

    /// The calling thread's user-space instructions event of `config`, disabled until enabled.
    pub(super) fn open(config: u64) -> io::Result<File> {
        open_with(config, DISABLED | EXCLUDE_KERNEL | EXCLUDE_HV)
    }

    /// The user-space instructions event of `config` of the calling thread and every thread it
    /// starts after, counting from now.
    pub(super) fn open_inherited(config: u64) -> io::Result<File> {
        open_with(config, INHERIT | EXCLUDE_KERNEL | EXCLUDE_HV)
    }

    /// Thread `thread`'s own user-space instructions event of `config`, counting from now.
    pub(super) fn open_thread(thread: libc::pid_t, config: u64) -> io::Result<File> {
        open_on(thread, TYPE_HARDWARE, config, EXCLUDE_KERNEL | EXCLUDE_HV)
    }

    /// The calling thread's user-space instructions event of `config` with the bit fields `flags`.
    fn open_with(config: u64, flags: u64) -> io::Result<File> {
        open_on(0, TYPE_HARDWARE, config, flags)
    }

    /// Thread `thread`'s (0: the calling thread's) event of `config` on the unit of `type` `kind`,
    /// with the bit fields `flags`.
    fn open_on(thread: libc::pid_t, kind: u32, config: u64, flags: u64) -> io::Result<File> {
        let mut attr = [0u8; ATTR_SIZE];
        attr[ATTR_TYPE..ATTR_TYPE + 4].copy_from_slice(&kind.to_ne_bytes());
        attr[ATTR_SIZE_FIELD..ATTR_SIZE_FIELD + 4]
            .copy_from_slice(&(ATTR_SIZE as u32).to_ne_bytes());
        attr[ATTR_CONFIG..ATTR_CONFIG + 8].copy_from_slice(&config.to_ne_bytes());
        attr[ATTR_FLAGS..ATTR_FLAGS + 8].copy_from_slice(&flags.to_ne_bytes());
        let (any_cpu, no_group): (libc::c_int, libc::c_int) = (-1, -1);
        // SAFETY: `attr` is a valid `perf_event_attr` of the size it states and outlives the
        // call; `thread` names a thread (0 the calling one) and cpu -1 any CPU, with no group.
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

#[cfg(test)]
mod names {
    use super::raw_event;

    /// A raw event's name gives its unit and config, `cpu` where it names none; a trailing `/`,
    /// `/u` or `:u` is taken; anything else is refused.
    #[test]
    fn a_raw_event_name_gives_its_unit_and_config() {
        for (name, want) in [
            ("cpu_core/r21d0", ("cpu_core", 0x21d0)),
            ("cpu_core/r21d0/", ("cpu_core", 0x21d0)),
            ("cpu_core/r21d0/u", ("cpu_core", 0x21d0)),
            ("r1e25", ("cpu", 0x1e25)),
            ("r1e25:u", ("cpu", 0x1e25)),
        ] {
            assert_eq!(raw_event(name).expect(name), want, "{name}");
        }
        for name in [
            "",
            "r",
            "1e25",
            "/r1e25",
            "cpu_core/mem_inst_retired.lock_loads/u",
            "rxyz",
        ] {
            assert!(raw_event(name).is_err(), "{name:?} was taken");
        }
    }

    /// A kernel CPU list gives its CPUs in order; a list of another form is refused.
    #[cfg(all(target_os = "linux", target_endian = "little"))]
    #[test]
    fn a_cpu_list_gives_its_cpus() {
        use super::cpu_list;
        assert_eq!(
            cpu_list("0-3,8,10-11\n").expect("a list"),
            [0, 1, 2, 3, 8, 10, 11]
        );
        assert_eq!(cpu_list("").expect("an empty list"), Vec::<usize>::new());
        for list in ["3-1", "a", "0-"] {
            assert!(cpu_list(list).is_err(), "{list:?} was taken");
        }
    }
}

#[cfg(all(test, target_os = "linux", target_arch = "x86_64"))]
mod tests {
    use super::{
        pin_this_thread, this_thread, InstructionCounter, ProcessEvent, ProcessInstructions,
        ThreadInstructions,
    };
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
        pin_this_thread(cpus).unwrap_or_else(|error| panic!("sched_setaffinity: {error}"));
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

    /// The thread counter counts the threads it was given and no other: a thread's fixed loop on
    /// every CPU it may use is in the count of a counter opened on its id, the opening thread's own
    /// loop is not.
    #[test]
    fn the_thread_counter_counts_the_threads_named_and_no_other() {
        if !counters_offered() {
            assert!(
                ThreadInstructions::new(&[this_thread().expect("Linux")]).is_err(),
                "the host offers no counters, yet a counter opened"
            );
            return;
        }
        let cpus = allowed_cpus();
        let expected = 2 * ITERATIONS * cpus.len() as u64;
        let worker_loops = ITERATIONS * cpus.len() as u64;
        let (id_tx, id_rx) = std::sync::mpsc::channel();
        let (go_tx, go_rx) = std::sync::mpsc::channel::<()>();
        let worker = std::thread::spawn(move || {
            id_tx.send(this_thread().expect("Linux")).expect("the id");
            go_rx.recv().expect("the go");
            for &cpu in &cpus {
                pin(&[cpu]);
                fixed_loop(ITERATIONS);
            }
        });
        let id = id_rx.recv().expect("the worker's id");
        let counter = ThreadInstructions::new(&[id]).expect("the host offers counters");
        let before = counter.read().expect("reading the counters");
        // The opening thread's own loop, as long as the worker's, which the counter is not on.
        fixed_loop(worker_loops);
        go_tx.send(()).expect("the go");
        worker.join().expect("the loop's thread");
        let counted = counter.read().expect("reading the counters") - before;
        // The wait for the go, the pinning and the thread's end add their own instructions.
        assert!(
            (expected..expected + expected / 20).contains(&counted),
            "{counted} user-space instructions counted for {expected}"
        );
    }

    /// A raw event counts as `perf` names it: `rc0`, the instructions retired on Intel's and on
    /// AMD's x86 cores alike (Intel's architectural event 0xc0, AMD's PMCx0C0), on the performance
    /// cores' unit where the CPU has one, counts a thread's fixed loop, started after it opened, on
    /// every CPU of that unit.
    #[test]
    fn a_raw_event_counts_a_thread_started_after_it_opened() {
        if !counters_offered() {
            assert!(
                ProcessEvent::new("rc0").is_err(),
                "the host offers no counters, yet a counter opened"
            );
            return;
        }
        let hybrid = Path::new("/sys/bus/event_source/devices/cpu_core").exists();
        let name = if hybrid { "cpu_core/rc0/u" } else { "rc0" };
        let allowed = allowed_cpus();
        let cpus = match hybrid {
            true => {
                let unit = std::fs::read_to_string("/sys/bus/event_source/devices/cpu_core/cpus")
                    .expect("the performance cores");
                let unit = super::cpu_list(&unit).expect("a CPU list");
                (allowed.iter().copied())
                    .filter(|cpu| unit.contains(cpu))
                    .collect::<Vec<_>>()
            }
            false => allowed.clone(),
        };
        pin(&cpus);
        let event = ProcessEvent::new(name).expect("the host offers counters");
        if hybrid {
            assert!(!event.cpus().is_empty(), "{name}: the unit's CPUs");
        }
        let expected = 2 * ITERATIONS * cpus.len() as u64;
        let before = event.read().expect("reading the counter");
        let worker = std::thread::spawn(move || {
            for &cpu in &cpus {
                pin(&[cpu]);
                fixed_loop(ITERATIONS);
            }
        });
        worker.join().expect("the loop's thread");
        let counted = event.read().expect("reading the counter") - before;
        pin(&allowed);
        // Starting and ending the thread and the pinning add their own instructions.
        assert!(
            (expected..expected + expected / 20).contains(&counted),
            "{name}: {counted} counted for {expected} instructions"
        );
    }

    /// The process counter counts a thread started after it opened: the fixed loop on every CPU
    /// that thread may use, on a thread of its own, is in the count once the thread ended.
    #[test]
    fn the_process_counter_counts_a_thread_started_after_it_opened() {
        if !counters_offered() {
            assert!(
                ProcessInstructions::new().is_err(),
                "the host offers no counters, yet a counter opened"
            );
            return;
        }
        let process = ProcessInstructions::new().expect("the host offers counters");
        let cpus = allowed_cpus();
        let expected = 2 * ITERATIONS * cpus.len() as u64;
        let before = process.read().expect("reading the counters");
        let worker = std::thread::spawn(move || {
            for &cpu in &cpus {
                pin(&[cpu]);
                fixed_loop(ITERATIONS);
            }
        });
        worker.join().expect("the loop's thread");
        let counted = process.read().expect("reading the counters") - before;
        // Starting and ending the thread and the pinning add their own instructions.
        assert!(
            (expected..expected + expected / 20).contains(&counted),
            "{counted} user-space instructions counted for {expected}"
        );
    }
}
