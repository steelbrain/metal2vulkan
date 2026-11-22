//! Process-group supervision for validation workers. Policy belongs to the caller.
//!
//! It began as a copy of the product's `src/process.rs`, which 0.2.0 deleted with the old engine's
//! CLI: the engine is in-process only, so the killable worker boundary is validation tooling
//! and the consumer's out-of-process recipe (`docs/HOWTO.md` section 5), not a product
//! API. It charges a group member only the memory that is its own (`process/linux.rs`).

#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "linux")]
use linux::member_resident_bytes;

#[cfg(any(target_os = "linux", all(test, target_os = "macos")))]
use std::fs;
use std::process::{Child, Command, ExitStatus};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

#[derive(Default)]
pub struct ProcessControl<'a> {
    pub deadline: Option<Instant>,
    pub memory_limit_bytes: Option<u64>,
    pub cancelled: Option<&'a AtomicBool>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub enum ProcessFailure {
    DeadlineExceeded,
    Cancelled,
    MemoryLimitExceeded { limit: u64, measured: u64 },
    Monitoring(String),
}

impl std::fmt::Display for ProcessFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DeadlineExceeded => f.write_str("worker deadline exceeded"),
            Self::Cancelled => f.write_str("worker cancelled"),
            Self::MemoryLimitExceeded { limit, measured } => write!(f, "worker exceeded {limit} byte process-group resident-memory budget (measured {measured} bytes)"),
            Self::Monitoring(error) => write!(f, "worker monitoring: {error}"),
        }
    }
}

impl std::error::Error for ProcessFailure {}

/// Why [`MonitoredChild::wait`] stopped a child, with the group's peak sampled RSS up to the
/// stop: a stopped attempt is measured like a finished one.
#[derive(Debug)]
pub struct ProcessStop {
    pub failure: ProcessFailure,
    /// Maximum sampled process-group RSS before the stop when memory monitoring was enabled.
    pub peak_resident_bytes: Option<u64>,
}

impl std::fmt::Display for ProcessStop {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.failure.fmt(f)
    }
}

impl std::error::Error for ProcessStop {}

impl ProcessControl<'_> {
    pub fn check(&self) -> Result<(), ProcessFailure> {
        if self
            .cancelled
            .is_some_and(|flag| flag.load(Ordering::Relaxed))
        {
            return Err(ProcessFailure::Cancelled);
        }
        if self
            .deadline
            .is_some_and(|deadline| Instant::now() >= deadline)
        {
            return Err(ProcessFailure::DeadlineExceeded);
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct ProcessExit {
    pub status: ExitStatus,
    /// The larger of the process group's peak sampled RSS (with memory monitoring) and the
    /// kernel's peak resident size of the leader at its reap (`ru_maxrss` from `wait4`) where that
    /// figure is the leader's own (`own_kernel_peak`). The sampler sums every member but samples
    /// only while the leader runs, every 1 ms until a sample finds memory of the group's own and
    /// every 10 ms after, and charges a leader still in its spawner's memory nothing; the kernel
    /// counts the leader alone (or the largest child it reaped), never the group's sum, so the
    /// sampled total stays the bound the memory limit enforces. `None` when neither measured memory
    /// of the worker's own.
    pub peak_resident_bytes: Option<u64>,
}

/// Owns a dedicated process group and kills remaining descendants on every exit path.
pub struct MonitoredChild {
    child: Child,
    state: ChildState,
}

/// Where a [`MonitoredChild`]'s leader is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ChildState {
    /// Not yet reaped here: `Child` alone may wait for it (it may have already).
    Running,
    /// Reaped here by [`try_reap`], which `Child` does not know: it must never wait for that pid
    /// again, which may already name another process. The group may still hold descendants.
    Reaped,
    /// The group was killed and the leader waited for.
    Stopped,
}

impl MonitoredChild {
    /// Start `command` in its own process group. The child inherits this process's private
    /// corpus root ([`crate::source::PRIVATE_CORPUS_ROOT_ENV`]), so a worker reads the private
    /// data its parent reads.
    #[cfg(unix)]
    pub fn spawn(command: &mut Command) -> std::io::Result<Self> {
        configure_process_group(command);
        match crate::source::private_corpus_root() {
            Some(root) => command.env(crate::source::PRIVATE_CORPUS_ROOT_ENV, root),
            None => command.env_remove(crate::source::PRIVATE_CORPUS_ROOT_ENV),
        };
        Ok(Self {
            child: command.spawn()?,
            state: ChildState::Running,
        })
    }

    #[cfg(not(unix))]
    pub fn spawn(_command: &mut Command) -> std::io::Result<Self> {
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "process-group supervision is unsupported on this platform",
        ))
    }

    pub fn id(&self) -> u32 {
        self.child.id()
    }

    pub fn wait(mut self, control: &ProcessControl<'_>) -> Result<ProcessExit, ProcessStop> {
        let mut peak = None::<u64>;
        let result = (|| {
            #[cfg(not(any(target_os = "macos", target_os = "linux")))]
            if control.memory_limit_bytes.is_some() {
                return Err(ProcessFailure::Monitoring(
                    "process-group memory monitoring is unsupported on this platform".into(),
                ));
            }
            loop {
                control.check()?;
                match try_reap(&mut self.child) {
                    Ok(Some(reaped)) => {
                        if reaped.kernel_peak_bytes.is_some() {
                            self.state = ChildState::Reaped;
                        }
                        let kernel = reaped.kernel_peak_bytes.and_then(own_kernel_peak);
                        return Ok(ProcessExit {
                            status: reaped.status,
                            // `None` orders below every `Some`.
                            peak_resident_bytes: own_memory(peak).max(kernel),
                        });
                    }
                    Ok(None) => {}
                    Err(error) => return Err(ProcessFailure::Monitoring(error.to_string())),
                }
                if let Some(limit) = control.memory_limit_bytes {
                    if let Some(resident) =
                        worker_resident_bytes(self.id()).map_err(ProcessFailure::Monitoring)?
                    {
                        peak = Some(peak.unwrap_or(0).max(resident));
                        if resident > limit {
                            return Err(ProcessFailure::MemoryLimitExceeded {
                                limit,
                                measured: resident,
                            });
                        }
                    }
                }
                // Until the leader runs its own image a sample charges it nothing, so a worker that
                // execs and exits inside one 10 ms tick would leave no sample of its own: sample
                // every 1 ms until one finds memory of the group's own.
                let tick = if control.memory_limit_bytes.is_some() && own_memory(peak).is_none() {
                    1
                } else {
                    10
                };
                std::thread::sleep(Duration::from_millis(tick));
            }
        })();
        self.stop();
        result
            .and_then(|result| {
                control.check()?;
                Ok(result)
            })
            .map_err(|failure| ProcessStop {
                failure,
                peak_resident_bytes: own_memory(peak),
            })
    }

    fn stop(&mut self) {
        match self.state {
            ChildState::Running => terminate_child(&mut self.child),
            ChildState::Reaped => kill_group(self.child.id()),
            ChildState::Stopped => {}
        }
        self.state = ChildState::Stopped;
    }
}

impl Drop for MonitoredChild {
    fn drop(&mut self) {
        self.stop();
    }
}

// Refuse an incomplete snapshot rather than silently undercounting a large group.
#[cfg(any(target_os = "macos", target_os = "linux"))]
const MAX_WORKER_GROUP_MEMBERS: usize = 1024;

/// The group's resident bytes: the sum of what each member holds on its own, or `None` when no
/// member was left to sample.
#[cfg(any(target_os = "macos", target_os = "linux"))]
fn worker_resident_bytes(group: u32) -> Result<Option<u64>, String> {
    let mut resident = None::<u64>;
    for pid in worker_group_members(group)? {
        if let Some(bytes) = member_resident_bytes(group, pid)? {
            resident = Some(
                resident
                    .unwrap_or(0)
                    .checked_add(bytes)
                    .ok_or("worker process-group resident sum overflow")?,
            );
        }
    }
    Ok(resident)
}

/// What member `pid` of `group` holds resident, or `None` once it is gone or has left the group.
/// macOS's `posix_spawn` creates a process and its image in one call, so a spawned member never
/// runs in its parent's memory, the Linux case.
#[cfg(target_os = "macos")]
fn member_resident_bytes(group: u32, pid: u32) -> Result<Option<u64>, String> {
    // Recheck membership because a process can exit between enumeration and sampling.
    if process_group(pid)? != Some(group) {
        return Ok(None);
    }
    process_resident_bytes(pid)
}

#[cfg(any(target_os = "macos", target_os = "linux"))]
fn process_group(pid: u32) -> Result<Option<u32>, String> {
    let group = unsafe { libc::getpgid(pid as libc::pid_t) };
    if group >= 0 {
        return Ok(Some(group as u32));
    }
    let error = std::io::Error::last_os_error();
    if error.raw_os_error() == Some(libc::ESRCH) {
        return Ok(None);
    }
    Err(format!("read worker process group for {pid}: {error}"))
}

#[cfg(target_os = "macos")]
fn worker_group_members(group: u32) -> Result<Vec<u32>, String> {
    let mut pids = [0 as libc::pid_t; MAX_WORKER_GROUP_MEMBERS];
    let capacity = std::mem::size_of_val(&pids) as libc::c_int;
    // PROC_PGRP_ONLY from the platform's proc_info.h; libc exposes the function.
    let bytes = unsafe {
        *libc::__error() = 0;
        libc::proc_listpids(2, group, pids.as_mut_ptr().cast(), capacity)
    };
    let error = std::io::Error::last_os_error();
    if bytes < 0 || (bytes == 0 && error.raw_os_error() != Some(0)) {
        if error.raw_os_error() == Some(libc::ESRCH) {
            return Ok(Vec::new());
        }
        return Err(format!("enumerate worker process group: {error}"));
    }
    if bytes >= capacity {
        return Err("worker process-group memory snapshot exceeds member capacity".into());
    }
    if !(bytes as usize).is_multiple_of(std::mem::size_of::<libc::pid_t>()) {
        return Err("worker process-group memory snapshot has an invalid byte count".into());
    }
    Ok(pids[..bytes as usize / std::mem::size_of::<libc::pid_t>()]
        .iter()
        .copied()
        .filter(|pid| *pid > 0)
        .map(|pid| pid as u32)
        .collect())
}

#[cfg(target_os = "linux")]
fn worker_group_members(group: u32) -> Result<Vec<u32>, String> {
    let mut members = Vec::new();
    for entry in fs::read_dir("/proc").map_err(|error| format!("enumerate processes: {error}"))? {
        let entry = entry.map_err(|error| format!("enumerate process: {error}"))?;
        let Some(pid) = entry
            .file_name()
            .to_str()
            .and_then(|name| name.parse::<libc::pid_t>().ok())
        else {
            continue;
        };
        if pid <= 0 || process_group(pid as u32)? != Some(group) {
            continue;
        }
        if members.len() == MAX_WORKER_GROUP_MEMBERS {
            return Err("worker process-group memory snapshot exceeds member capacity".into());
        }
        members.push(pid as u32);
    }
    Ok(members)
}

#[cfg(target_os = "macos")]
fn process_resident_bytes(pid: u32) -> Result<Option<u64>, String> {
    let mut usage = std::mem::MaybeUninit::<libc::rusage_info_v2>::zeroed();
    let result = unsafe {
        libc::proc_pid_rusage(
            pid as libc::c_int,
            libc::RUSAGE_INFO_V2,
            usage.as_mut_ptr().cast::<libc::rusage_info_t>(),
        )
    };
    if result == 0 {
        Ok(Some(unsafe { usage.assume_init() }.ri_resident_size))
    } else if std::io::Error::last_os_error().raw_os_error() == Some(libc::ESRCH) {
        Ok(None)
    } else {
        Err(format!(
            "read translation worker resident memory: {}",
            std::io::Error::last_os_error()
        ))
    }
}

#[cfg(target_os = "linux")]
fn process_resident_bytes(pid: u32) -> Result<Option<u64>, String> {
    let path = format!("/proc/{pid}/status");
    let status = match fs::read_to_string(&path) {
        Ok(status) => status,
        Err(error) if process_gone(&error) => return Ok(None),
        Err(error) => return Err(format!("read {path}: {error}")),
    };
    Ok(status_bytes(&status, "VmRSS:"))
}

/// Whether reading a group member's `/proc/<pid>/status` failed because the member is gone: its
/// directory was removed before the open (`ENOENT`), or the member was reaped between the open
/// and the read, which the kernel reports as `ESRCH`. A worker reaps its own children (a tool it
/// ran), so either can happen while the group is sampled; the member no longer counts.
#[cfg(any(target_os = "linux", all(test, unix)))]
fn process_gone(error: &std::io::Error) -> bool {
    error.kind() == std::io::ErrorKind::NotFound || error.raw_os_error() == Some(libc::ESRCH)
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn worker_resident_bytes(_pid: u32) -> Result<Option<u64>, String> {
    Ok(None)
}

/// This process's resident bytes now, as its workers' are read; `None` where RSS monitoring is
/// unsupported or the read failed. A supervisor reports its own memory beside its workers'.
pub fn own_resident_bytes() -> Option<u64> {
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    {
        process_resident_bytes(std::process::id()).ok().flatten()
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        None
    }
}

/// The most this process has held resident since its image was installed; `None` where it cannot
/// be read. A worker states it in its report (`worker::report`): the parent's sampler misses a
/// worker that execs and exits between two samples, and on Linux the kernel's figure at the reap
/// may be the parent's own (`own_kernel_peak`), but the worker knows its own high water. On Linux
/// it is the image's own `VmHWM`, not `getrusage`'s `ru_maxrss`, which holds the spawner's high
/// water folded in at the exec; on macOS it is `ru_maxrss`, which is the process's own
/// (`member_resident_bytes`).
pub fn own_peak_resident_bytes() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        own_high_water_bytes()
    }
    #[cfg(target_os = "macos")]
    {
        let mut usage = std::mem::MaybeUninit::<libc::rusage>::zeroed();
        // SAFETY: `usage` is valid for the write, which a success makes.
        let read = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
        // SAFETY: `getrusage` filled `usage` when it succeeded.
        (read == 0).then(|| max_rss_bytes(unsafe { usage.assume_init() }.ru_maxrss))
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        None
    }
}

/// A leader [`try_reap`] found exited.
struct Reaped {
    status: ExitStatus,
    /// The kernel's peak resident bytes of the leader (`ru_maxrss`, which also covers the
    /// children it reaped, each alone), when this call reaped it; `None` when `Child` had already
    /// waited for it.
    kernel_peak_bytes: Option<u64>,
}

/// Reap `child` if it has exited, without blocking, with the kernel's peak resident size of it:
/// `wait4` returns its `rusage` with the status, which `Child::try_wait` drops. A child `Child`
/// already waited for keeps the status `Child` stored.
#[cfg(unix)]
fn try_reap(child: &mut Child) -> std::io::Result<Option<Reaped>> {
    use std::os::unix::process::ExitStatusExt as _;
    let pid = libc::pid_t::try_from(child.id())
        .map_err(|_| std::io::Error::other("a child pid out of range"))?;
    loop {
        let mut status = 0;
        let mut usage = std::mem::MaybeUninit::<libc::rusage>::zeroed();
        // SAFETY: `status` and `usage` are valid for writes, and `pid` is this child's.
        let reaped = unsafe { libc::wait4(pid, &mut status, libc::WNOHANG, usage.as_mut_ptr()) };
        if reaped == pid {
            // SAFETY: `wait4` filled `usage` when it reaped the child.
            let max_rss = unsafe { usage.assume_init() }.ru_maxrss;
            return Ok(Some(Reaped {
                status: ExitStatus::from_raw(status),
                kernel_peak_bytes: Some(max_rss_bytes(max_rss)),
            }));
        }
        if reaped == 0 {
            return Ok(None);
        }
        let error = std::io::Error::last_os_error();
        match error.raw_os_error() {
            Some(libc::EINTR) => continue,
            Some(libc::ECHILD) => {
                return child.try_wait().map(|status| {
                    status.map(|status| Reaped {
                        status,
                        kernel_peak_bytes: None,
                    })
                })
            }
            _ => return Err(error),
        }
    }
}

#[cfg(not(unix))]
fn try_reap(child: &mut Child) -> std::io::Result<Option<Reaped>> {
    child.try_wait().map(|status| {
        status.map(|status| Reaped {
            status,
            kernel_peak_bytes: None,
        })
    })
}

/// A sampled peak that found memory of the group's own: a sample of only a leader still in its
/// spawner's memory is charged 0, which measures nothing of the worker.
fn own_memory(peak: Option<u64>) -> Option<u64> {
    peak.filter(|bytes| *bytes > 0)
}

/// The kernel's peak for a leader reaped here, when it is the leader's own. On Linux a spawn runs
/// in its spawner's memory until its exec, and the exec folds that memory's high-water mark into
/// the child's `ru_maxrss` (`exec_mmap` calls `setmax_mm_hiwater_rss` on the old memory): the
/// kernel states the larger of the supervisor's high water at the exec and the worker's own peak,
/// so a trivial `sleep` under a 600-MiB supervisor read 613 MiB. The supervisor's high water
/// (`VmHWM`, read at the reap) has only grown since the exec, so a figure above it is the worker's
/// own; one at or below it may be the supervisor's and is not used, and the sampled peak stands
/// alone. (The kernel records a high water before an unmap lowers the resident size, though not
/// before reclaim does, so a supervisor whose pages were reclaimed between the exec and the reap
/// is the one case this can misread.) macOS's `posix_spawn` starts the child in its own memory
/// (`member_resident_bytes`), so there the figure is the worker's.
fn own_kernel_peak(kernel_peak_bytes: u64) -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let supervisor = own_high_water_bytes()?;
        (kernel_peak_bytes > supervisor).then_some(kernel_peak_bytes)
    }
    #[cfg(not(target_os = "linux"))]
    {
        Some(kernel_peak_bytes)
    }
}

/// This process's resident high-water mark (`VmHWM`), or `None` when it cannot be read.
#[cfg(target_os = "linux")]
fn own_high_water_bytes() -> Option<u64> {
    let status = fs::read_to_string("/proc/self/status").ok()?;
    status_bytes(&status, "VmHWM:")
}

/// A `/proc/<pid>/status` field stated in kB, in bytes.
#[cfg(target_os = "linux")]
fn status_bytes(status: &str, field: &str) -> Option<u64> {
    status
        .lines()
        .find_map(|line| line.strip_prefix(field))
        .and_then(|value| value.split_whitespace().next())
        .and_then(|value| value.parse::<u64>().ok())
        .map(|kib| kib * 1024)
}

/// `ru_maxrss` in bytes: macOS states it in bytes, Linux and the BSDs in KiB.
#[cfg(unix)]
fn max_rss_bytes(max_rss: libc::c_long) -> u64 {
    let value = u64::try_from(max_rss).unwrap_or(0);
    if cfg!(target_os = "macos") {
        value
    } else {
        value.saturating_mul(1024)
    }
}

/// Kill what is left of the group of a leader [`try_reap`] reaped: descendants that outlived it,
/// as [`terminate_child`] does once the leader is gone. Nothing waits for the leader again.
#[cfg(unix)]
fn kill_group(leader: u32) {
    let group = -(leader as i32);
    // SAFETY: `kill` takes no pointers.
    unsafe {
        let _ = libc::kill(group, libc::SIGTERM);
        let _ = libc::kill(group, libc::SIGKILL);
    }
}

#[cfg(not(unix))]
fn kill_group(_leader: u32) {}

#[cfg(unix)]
fn configure_process_group(command: &mut Command) {
    use std::os::unix::process::CommandExt as _;
    command.process_group(0);
}

#[cfg(unix)]
fn terminate_child(child: &mut Child) {
    let group = -(child.id() as i32);
    unsafe {
        let _ = libc::kill(group, libc::SIGTERM);
    }
    let grace_started = Instant::now();
    loop {
        match child.try_wait() {
            // A descendant can outlive the leader and ignore SIGTERM.
            Ok(Some(_)) => break,
            Ok(None) if grace_started.elapsed() < Duration::from_millis(100) => {
                std::thread::sleep(Duration::from_millis(10));
            }
            Ok(None) | Err(_) => break,
        }
    }
    unsafe {
        let _ = libc::kill(group, libc::SIGKILL);
    }
    let _ = child.wait();
}

#[cfg(not(unix))]
fn terminate_child(child: &mut Child) {
    let _ = child.kill();
    let _ = child.wait();
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Stdio;

    /// A member reaped while the group is sampled is gone, not a monitoring failure: on Linux the
    /// `/proc/<pid>/status` a sampler opened before the reap reads `ESRCH`.
    #[cfg(unix)]
    #[test]
    fn a_member_reaped_while_sampled_is_gone() {
        assert!(process_gone(&std::io::Error::from_raw_os_error(
            libc::ENOENT
        )));
        assert!(process_gone(&std::io::Error::from_raw_os_error(
            libc::ESRCH
        )));
        assert!(!process_gone(&std::io::Error::from_raw_os_error(
            libc::EACCES
        )));
        #[cfg(target_os = "linux")]
        {
            use std::io::Read;
            let mut child = Command::new("sleep").arg("60").spawn().unwrap();
            let opened = fs::File::open(format!("/proc/{}/status", child.id()));
            child.kill().unwrap();
            child.wait().unwrap();
            let error = opened
                .unwrap()
                .read_to_string(&mut String::new())
                .unwrap_err();
            assert!(process_gone(&error), "{error}");
        }
    }

    #[cfg(unix)]
    #[test]
    fn caller_controls_cancellation_and_unlimited_execution() {
        let mut command = Command::new("sh");
        command.args(["-c", "exit 7"]);
        let result = MonitoredChild::spawn(&mut command)
            .unwrap()
            .wait(&ProcessControl::default())
            .unwrap();
        assert_eq!(result.status.code(), Some(7));
        // Unmonitored, the kernel's figure at the reap is reported where it is the worker's own:
        // on macOS always, on Linux only above this supervisor's high water, which a shell that
        // exits at once stays under (`own_kernel_peak`).
        if cfg!(target_os = "linux") {
            assert_eq!(result.peak_resident_bytes, None);
        } else {
            assert!(result.peak_resident_bytes.is_some_and(|peak| peak > 0));
        }

        let cancelled = AtomicBool::new(false);
        let mut command = Command::new("sh");
        command.args(["-c", "exec sleep 60"]);
        let worker = MonitoredChild::spawn(&mut command).unwrap();
        std::thread::scope(|scope| {
            scope.spawn(|| {
                std::thread::sleep(Duration::from_millis(20));
                cancelled.store(true, Ordering::Relaxed);
            });
            assert!(matches!(
                worker.wait(&ProcessControl {
                    cancelled: Some(&cancelled),
                    ..Default::default()
                }),
                Err(ProcessStop {
                    failure: ProcessFailure::Cancelled,
                    peak_resident_bytes: None
                })
            ));
        });
    }

    #[cfg(any(target_os = "macos", target_os = "linux"))]
    #[test]
    fn worker_memory_fixture() {
        let Ok(marker) = std::env::var("M2V_TEST_MEMORY_MARKER") else {
            return;
        };
        if std::env::var_os("M2V_TEST_MEMORY_CHILD").is_some() {
            let mib = if std::env::var_os("M2V_TEST_MEMORY_OVER_BUDGET").is_some() {
                256
            } else {
                64
            };
            let mut allocation = vec![0u8; mib * 1024 * 1024];
            for page in allocation.chunks_mut(4096) {
                page[0] = 1;
            }
            std::hint::black_box(&allocation);
            fs::write(marker, std::process::id().to_string()).unwrap();
            std::thread::sleep(Duration::from_secs(60));
            std::hint::black_box(allocation);
        } else {
            let count = if std::env::var_os("M2V_TEST_MEMORY_OVER_BUDGET").is_some() {
                2
            } else {
                1
            };
            let children: Vec<_> = (0..count)
                .map(|_| {
                    Command::new(std::env::current_exe().unwrap())
                        .args(["--exact", "worker::process::tests::worker_memory_fixture"])
                        .env("M2V_TEST_MEMORY_CHILD", "1")
                        .spawn()
                        .unwrap()
                })
                .collect();
            for mut child in children {
                child.wait().unwrap();
            }
        }
    }

    #[cfg(any(target_os = "macos", target_os = "linux"))]
    #[test]
    fn worker_memory_includes_external_descendants() {
        let scratch = crate::ScratchDir::new("process-test").unwrap();
        let marker = scratch.path().join("ready");
        let mut command = Command::new(std::env::current_exe().unwrap());
        command
            .args(["--exact", "worker::process::tests::worker_memory_fixture"])
            .env("M2V_TEST_MEMORY_MARKER", &marker)
            .env_remove("M2V_TEST_MEMORY_CHILD")
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        configure_process_group(&mut command);
        let mut worker = command.spawn().unwrap();
        let measured = (|| -> Result<(u64, u64), String> {
            let started = Instant::now();
            while !marker.exists() {
                if started.elapsed() > Duration::from_secs(5) {
                    return Err("memory fixture did not become ready".into());
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            let members = worker_group_members(worker.id())?;
            if members.len() != 2 || !members.contains(&worker.id()) {
                return Err(format!("unexpected fixture process group: {members:?}"));
            }
            let leader = process_resident_bytes(worker.id())?.ok_or("leader disappeared")?;
            let group = worker_resident_bytes(worker.id())?.ok_or("group disappeared")?;
            Ok((leader, group))
        })();
        terminate_child(&mut worker);
        let (leader, group) = measured.unwrap();
        assert!(
            group > leader + 32 * 1024 * 1024,
            "leader={leader}, group={group}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn completed_worker_cannot_bypass_an_expired_deadline() {
        let mut command = Command::new(std::env::current_exe().unwrap());
        command
            .arg("--list")
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        configure_process_group(&mut command);
        let mut worker = command.spawn().unwrap();
        assert!(worker.wait().unwrap().success());
        let control = ProcessControl {
            deadline: Some(Instant::now()),
            ..Default::default()
        };
        assert!(matches!(
            MonitoredChild {
                child: worker,
                state: ChildState::Running
            }
            .wait(&control),
            Err(ProcessStop {
                failure: ProcessFailure::DeadlineExceeded,
                ..
            })
        ));
    }

    #[cfg(any(target_os = "macos", target_os = "linux"))]
    #[test]
    fn worker_memory_watchdog_enforces_aggregate_limit() {
        let scratch = crate::ScratchDir::new("process-test").unwrap();
        let mut command = Command::new(std::env::current_exe().unwrap());
        command
            .args(["--exact", "worker::process::tests::worker_memory_fixture"])
            .env("M2V_TEST_MEMORY_MARKER", scratch.path().join("ready"))
            .env("M2V_TEST_MEMORY_OVER_BUDGET", "1")
            .env_remove("M2V_TEST_MEMORY_CHILD")
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        configure_process_group(&mut command);
        let worker = command.spawn().unwrap();
        let control = ProcessControl {
            deadline: Some(Instant::now() + Duration::from_secs(20)),
            memory_limit_bytes: Some(500 * 1024 * 1024),
            cancelled: None,
        };
        let error = MonitoredChild {
            child: worker,
            state: ChildState::Running,
        }
        .wait(&control)
        .unwrap_err();
        assert!(
            matches!(error.failure, ProcessFailure::MemoryLimitExceeded { .. }),
            "{error}"
        );
        // The stop carries the peak it sampled, at least the breaching sample.
        let ProcessFailure::MemoryLimitExceeded { measured, .. } = error.failure else {
            unreachable!()
        };
        assert_eq!(error.peak_resident_bytes, Some(measured));
    }

    #[cfg(unix)]
    #[test]
    fn worker_shutdown_kills_descendants_after_the_leader_exits() {
        use std::io::Read;
        use std::os::fd::OwnedFd;
        use std::os::unix::net::UnixStream;

        for leader in ["wait", "exit 0"] {
            let (mut output, writer) = UnixStream::pair().unwrap();
            output
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            let mut command = Command::new("sh");
            command.args([
                "-c",
                &format!("(trap '' TERM; printf ready; exec sleep 60) & {leader}"),
            ]);
            command.stdout(Stdio::from(OwnedFd::from(writer)));
            let mut worker = MonitoredChild::spawn(&mut command).unwrap();
            // Command owns another copy of stdout until it is dropped.
            drop(command);
            let group = -(worker.id() as i32);
            let mut ready = [0; 5];
            let handshake = output.read_exact(&mut ready);
            if leader == "exit 0" {
                worker.child.wait().unwrap();
            }
            drop(worker);
            let eof = output.read(&mut [0]);
            // Clean up even when the assertion would expose a surviving descendant.
            unsafe {
                libc::kill(group, libc::SIGKILL);
            }
            handshake.unwrap();
            assert_eq!(&ready, b"ready");
            assert!(
                matches!(eof, Ok(0)),
                "descendant retained stdout after {leader}: {eof:?}"
            );
        }
    }

    /// A leader [`MonitoredChild::wait`] reaped itself (`ChildState::Reaped`) still has the
    /// descendants that outlive it killed, by group, without `Child` waiting for its pid again.
    #[cfg(unix)]
    #[test]
    fn a_reaped_leaders_descendants_are_killed() {
        use std::io::Read;
        use std::os::fd::OwnedFd;
        use std::os::unix::net::UnixStream;

        let (mut output, writer) = UnixStream::pair().unwrap();
        output
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        let mut command = Command::new("sh");
        command.args(["-c", "(trap '' TERM; printf ready; exec sleep 60) & exit 0"]);
        command.stdout(Stdio::from(OwnedFd::from(writer)));
        let worker = MonitoredChild::spawn(&mut command).unwrap();
        drop(command);
        let group = -(worker.id() as i32);
        let mut ready = [0; 5];
        let handshake = output.read_exact(&mut ready);
        let exit = worker.wait(&ProcessControl::default());
        let eof = output.read(&mut [0]);
        // Clean up even when the assertion would expose a surviving descendant.
        unsafe {
            libc::kill(group, libc::SIGKILL);
        }
        handshake.unwrap();
        assert_eq!(&ready, b"ready");
        let exit = exit.unwrap();
        assert!(exit.status.success());
        // Unmonitored, as in `caller_controls_cancellation_and_unlimited_execution`.
        if cfg!(target_os = "linux") {
            assert_eq!(exit.peak_resident_bytes, None);
        } else {
            assert!(exit.peak_resident_bytes.is_some_and(|peak| peak > 0));
        }
        assert!(matches!(eof, Ok(0)), "descendant retained stdout: {eof:?}");
    }

    /// Run as a worker by [`a_worker_that_exits_at_once_reports_its_peak`]: touch 256 MiB and
    /// exit at once.
    #[test]
    fn fast_exit_memory_fixture() {
        if std::env::var_os("M2V_TEST_FAST_EXIT_CHILD").is_none() {
            return;
        }
        let mut allocation = vec![0u8; 256 * 1024 * 1024];
        for page in allocation.chunks_mut(4096) {
            page[0] = 1;
        }
        std::hint::black_box(&allocation);
    }

    /// A monitored worker that exits at once still reports a peak of its own. One that touches
    /// 256 MiB, far above this test's own high water, reports at least that, whether or not a
    /// sample landed while it ran: the kernel's figure at the reap is its own on both platforms.
    /// On macOS even `true` reports the kernel's figure. On Linux `true` can exec and exit inside
    /// the sampler's first 1 ms tick, and its figure is this supervisor's high water, not its own
    /// (`own_kernel_peak`), so it may report none; a figure of the supervisor's never stands in.
    #[cfg(unix)]
    #[test]
    fn a_worker_that_exits_at_once_reports_its_peak() {
        let control = ProcessControl {
            memory_limit_bytes: Some(500 * 1024 * 1024),
            ..Default::default()
        };
        let mut command = Command::new("true");
        let exit = MonitoredChild::spawn(&mut command)
            .unwrap()
            .wait(&control)
            .unwrap();
        assert!(exit.status.success());
        if cfg!(target_os = "macos") {
            assert!(
                exit.peak_resident_bytes.is_some_and(|peak| peak > 0),
                "{exit:?}"
            );
        }

        let mut command = Command::new(std::env::current_exe().unwrap());
        command
            .args([
                "--exact",
                "worker::process::tests::fast_exit_memory_fixture",
            ])
            .env("M2V_TEST_FAST_EXIT_CHILD", "1")
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        let exit = MonitoredChild::spawn(&mut command)
            .unwrap()
            .wait(&control)
            .unwrap();
        assert!(exit.status.success());
        assert!(
            exit.peak_resident_bytes
                .is_some_and(|peak| peak >= 256 * 1024 * 1024),
            "{exit:?}"
        );
    }

    /// Run as a worker by [`a_workers_own_high_water_is_its_images_alone`]: write this
    /// process's [`own_peak_resident_bytes`] to the file the variable names.
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    #[test]
    fn own_peak_fixture() {
        let Some(path) = std::env::var_os("M2V_TEST_OWN_PEAK_FILE") else {
            return;
        };
        let peak = own_peak_resident_bytes().unwrap_or(0);
        fs::write(path, peak.to_string()).unwrap();
    }

    /// A worker's own high water ([`own_peak_resident_bytes`]) is its image's alone: spawned by a
    /// supervisor that held 256 MiB, which on Linux the exec folds into the worker's `ru_maxrss`
    /// (`own_kernel_peak`), it states less than that, and more than nothing.
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    #[test]
    fn a_workers_own_high_water_is_its_images_alone() {
        const HELD: u64 = 256 * 1024 * 1024;
        let mut held = vec![0u8; HELD as usize];
        for page in held.chunks_mut(4096) {
            page[0] = 1;
        }
        std::hint::black_box(&held);
        let scratch = crate::ScratchDir::new("process-test").unwrap();
        let path = scratch.path().join("peak");
        let status = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "worker::process::tests::own_peak_fixture"])
            .env("M2V_TEST_OWN_PEAK_FILE", &path)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .unwrap();
        drop(held);
        assert!(status.success());
        let stated: u64 = fs::read_to_string(&path).unwrap().parse().unwrap();
        assert!(stated > 0 && stated < HELD, "{stated}");
    }

    /// On Linux the kernel's figure for a reaped worker counts only above this process's own
    /// high water: at or below it, it may be this process's memory folded in at the exec.
    #[cfg(target_os = "linux")]
    #[test]
    fn a_kernel_figure_counts_only_above_the_supervisors_high_water() {
        let high_water = own_high_water_bytes().unwrap();
        assert!(high_water > 0);
        assert_eq!(own_kernel_peak(1), None);
        let above = high_water + (1 << 30);
        assert_eq!(own_kernel_peak(above), Some(above));
    }
}
