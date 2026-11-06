//! How long after the GPU ends a command buffer its completion is seen: the plain driver's floor
//! and the runtime's, on the device the cfg selects. A measurement, not a check: ignored by
//! default, printed as `COMPLETION-LATENCY` lines. Run it by name with `--run-ignored all`, in a
//! release build (`--release`).
//!
//! The shape is a consumer's that waits for every completion: [`RUNS`] submissions, each a fill
//! of [`BYTES`] bytes, made [`GAP`] apart after [`WARM`] untimed, so the device is idle between
//! them. Every time is on the host's clock (`gpu_time::host_nanos`), and a GPU's end is its
//! bottom-of-pipe timestamp, placed there by the device's calibration as `gpu_time.rs` places a
//! command buffer's (`GpuClock::place`, with a pair taken at most `REFRESH` before).
//!
//! - **The plain driver** (`raw_completion_latency`): a Vulkan device of the test's own, made
//!   through the loader with no runtime in the process. One command buffer, recorded once: a
//!   timestamp at the top of the pipe, the fill, a barrier to the host, a timestamp at the bottom;
//!   its queries reset on the host before each submission, which signals a timeline semaphore's
//!   next value (and a fence where the wait is a fence's). Waited for seven ways, each printed
//!   apart: `vkWaitSemaphores` on the submitting thread once `vkQueueSubmit` returned
//!   (`timeline`); `vkWaitForFences` there (`fence`); `poll` of a sync file exported from the
//!   fence (`sync_fd`, Linux, where the driver exports one); `vkWaitSemaphores` on a thread asked
//!   [`ARM_LEAD`] before the submission, so it is in its wait as the submission is made (`armed`,
//!   the floor of a thread that learns of the completion apart from the submitting one); on a
//!   thread asleep on a condition variable that
//!   the submitting thread wakes once `vkQueueSubmit` returned, as the runtime's queue waiter
//!   wakes for a watched batch (`woken`); and `vkGetSemaphoreCounterValue` read in a loop on the
//!   submitting thread, the first moment the host can see the value with no sleep (`spin`, a
//!   floor, not a wait anyone ships); and `vkWaitForFences` for the fill and barrier alone, with
//!   no timestamps (`fence untimed`), timed from `vkQueueSubmit` only: where the runtime's
//!   command buffers write no timestamps (MoltenVK on Apple's GPUs answers GPU times from the
//!   host, `gpu_time.rs`) and a timestamp costs the driver a pass of its own, its time from the
//!   submission is the one the runtime's from the commit compares with.
//! - **The runtime** (`runtime_completion_latency`): a device of the runtime's, one queue, each
//!   command buffer a blit encoder's `fill_buffer` with a completion-seen handler and a completed
//!   handler that note the host time, committed with `commit` (the consumer's shape), and in turn
//!   with them one with the completion-seen handler alone, so committed, and one so committed with
//!   `commit_and_submit`; none waited for. From the GPU end (`gpu_end_time`, on a device that
//!   calibrates its timestamps) to the completion-seen handler; and the runtime's steps between,
//!   from its trace (`command_buffer::completion::trace`): on the commit's way to the
//!   `VkQueue`, the commit posted to the scheduler, its thread woken by the commit and waking,
//!   the commit taken in by a pass, the driver's submission called and returned; then the batch
//!   watched, the driver's wait entered and returned, the completion turn's `seen` (the command
//!   buffer's completion time) and the handlers.
//!
//! - **From commit to submission** (`runtime_commit_to_submission`): a device of the runtime's
//!   under `RenderLoweringPolicy::Worker`, each command buffer carrying a submission check that
//!   passes and a submission handler that notes the host time and its thread, committed with
//!   `commit_and_submit` [`GAP`] apart, [`SUBMISSION_RUNS`] of each shape in a row: a fill alone
//!   on its queue; a fill on one queue right after another queue's head was committed whose check
//!   waits for a shared event's value that a third queue's long fill signals, so that head waits
//!   for the signal to land; and a render pass whose end the device's lowering thread has not
//!   lowered at the commit. From the commit call to the submission handler, with the threads the
//!   handlers ran on and the scheduler's work counts over each shape.
//!
//! Each line also counts the waits entered after the GPU end: there, the time to the wait's return
//! holds the wake of the thread that waits, not the driver's wake alone.
//!
//! The plain driver's latency less the runtime's is the runtime's own.
//!
//! Device tests: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`).

use super::encoder_holds::pass;
use crate::command_buffer::completion::trace::{self, Traced};
use crate::gpu_time::{self, host_nanos, GpuClock, Reading, REFRESH};
use crate::{
    Buffer, CommandBuffer, CommandQueue, Device, DeviceKind, DeviceOptions, EventRef,
    ResourceOptions, ShaderCacheOptions, Submission, TextureDescriptor, WorkCounts,
};
use ash::vk;
use metal2vulkan::device::RenderLoweringPolicy;
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::{CStr, CString};
use std::path::PathBuf;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// The timed submissions of each shape, and the untimed ones before them.
const RUNS: usize = 2000;
const WARM: usize = 20;

/// The time from one submission to the next.
const GAP: Duration = Duration::from_millis(2);

/// The bytes each fill writes.
const BYTES: u64 = 4096;

/// How long before its submission the `armed` waiter is asked to wait: longer than its wake takes
/// (up to about a millisecond at the 99th percentile on the RTX 5080 laptop's host).
const ARM_LEAD: Duration = Duration::from_millis(1);

/// The longest any wait may take, in nanoseconds.
const TIMEOUT: u64 = 5_000_000_000;

/// The device kinds the cfg selects, preferred first.
fn kinds() -> Vec<DeviceKind> {
    match cfg!(m2v_cpu_icd) {
        true => vec![DeviceKind::Cpu],
        false => vec![
            DeviceKind::Discrete,
            DeviceKind::Integrated,
            DeviceKind::Virtual,
        ],
    }
}

/// The physical device type of `kind`.
fn physical_type(kind: DeviceKind) -> vk::PhysicalDeviceType {
    match kind {
        DeviceKind::Discrete => vk::PhysicalDeviceType::DISCRETE_GPU,
        DeviceKind::Integrated => vk::PhysicalDeviceType::INTEGRATED_GPU,
        DeviceKind::Virtual => vk::PhysicalDeviceType::VIRTUAL_GPU,
        DeviceKind::Cpu => vk::PhysicalDeviceType::CPU,
        DeviceKind::Other => vk::PhysicalDeviceType::OTHER,
    }
}

/// The 50th, 90th and 99th percentiles of `values`, nearest rank.
fn percentiles(mut values: Vec<f64>) -> [f64; 3] {
    assert!(!values.is_empty(), "no samples");
    values.sort_by(f64::total_cmp);
    let at = |p: usize| values[((values.len() * p).div_ceil(100)).clamp(1, values.len()) - 1];
    [at(50), at(90), at(99)]
}

/// `values`' percentiles as the lines print them, in microseconds.
fn line(values: Vec<f64>) -> String {
    let [p50, p90, p99] = percentiles(values);
    format!("p50 {p50:.1} p90 {p90:.1} p99 {p99:.1}")
}

/// Microseconds from `from` to `to`, host nanoseconds, signed.
fn us(from: u64, to: u64) -> f64 {
    (i128::from(to) - i128::from(from)) as f64 / 1e3
}

/// Sleep until `at`.
fn sleep_until(at: Instant) {
    let now = Instant::now();
    if at > now {
        std::thread::sleep(at - now);
    }
}

/// How the plain driver's submission is waited for (the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RawWait {
    Timeline,
    Fence,
    SyncFd,
    Armed,
    Woken,
    Spin,
    Untimed,
}

impl RawWait {
    fn name(self) -> &'static str {
        match self {
            RawWait::Timeline => "timeline",
            RawWait::Fence => "fence",
            RawWait::SyncFd => "sync_fd",
            RawWait::Armed => "armed",
            RawWait::Woken => "woken",
            RawWait::Spin => "spin",
            RawWait::Untimed => "fence untimed",
        }
    }
}

/// One plain submission's host times, in nanoseconds: `vkQueueSubmit` called and returned, the
/// wait entered and returned, and the GPU's start and end placed on the host's clock.
#[derive(Clone, Copy, Debug)]
struct RawSample {
    submit: u64,
    submitted: u64,
    entered: u64,
    returned: u64,
    /// `None` for the command buffer that writes no timestamps (`fence untimed`).
    gpu_start: Option<u64>,
    gpu_end: Option<u64>,
}

/// A waiter thread's request: wait for the timeline's value (`armed` and `woken`), answered with
/// the moments its wait was entered and returned.
struct Waiter {
    /// Dropped first, which ends the thread.
    ask: Option<mpsc::Sender<u64>>,
    answer: mpsc::Receiver<(u64, u64)>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Waiter {
    fn new(device: ash::Device, timeline: vk::Semaphore) -> Waiter {
        let (ask, asked) = mpsc::channel::<u64>();
        let (answered, answer) = mpsc::channel();
        let thread = std::thread::spawn(move || {
            for value in asked {
                let entered = host_nanos();
                let semaphores = [timeline];
                let values = [value];
                let info = vk::SemaphoreWaitInfo::default()
                    .semaphores(&semaphores)
                    .values(&values);
                // SAFETY: the semaphore is the device's, live until the waiter is dropped.
                unsafe { device.wait_semaphores(&info, TIMEOUT) }.unwrap();
                let returned = host_nanos();
                if answered.send((entered, returned)).is_err() {
                    return;
                }
            }
        });
        Waiter {
            ask: Some(ask),
            answer,
            thread: Some(thread),
        }
    }
}

impl Waiter {
    fn ask(&self) -> &mpsc::Sender<u64> {
        self.ask.as_ref().unwrap()
    }
}

impl Drop for Waiter {
    fn drop(&mut self) {
        drop(self.ask.take());
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// A plain Vulkan device of the test's own, and what one fill's submission needs.
struct Raw {
    _entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,
    name: String,
    queue: vk::Queue,
    pool: vk::CommandPool,
    command: vk::CommandBuffer,
    /// The fill alone, with no timestamps: a timestamp costs MoltenVK a pass of its own.
    untimed: vk::CommandBuffer,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    queries: vk::QueryPool,
    timeline: vk::Semaphore,
    /// The timeline's last value signaled.
    value: u64,
    fence: vk::Fence,
    /// A fence the driver exports sync files of, and the entry points, where it does (Linux).
    exported: Option<(vk::Fence, ash::khr::external_fence_fd::Device)>,
    clock: GpuClock,
    /// The latest calibration reading and when it was taken.
    latest: Option<(Instant, Reading)>,
}

impl Raw {
    fn new() -> Raw {
        let entry = crate::device::create::open_loader(None).unwrap();
        // SAFETY: a global command of a loaded entry.
        let listed = unsafe { entry.enumerate_instance_extension_properties(None) }.unwrap();
        let portability = listed.iter().any(|extension| {
            extension.extension_name_as_c_str() == Ok(ash::khr::portability_enumeration::NAME)
        });
        let mut extensions: Vec<*const std::ffi::c_char> = Vec::new();
        if portability {
            extensions.push(ash::khr::portability_enumeration::NAME.as_ptr());
        }
        let application = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_2);
        let info = vk::InstanceCreateInfo::default()
            .application_info(&application)
            .enabled_extension_names(&extensions)
            .flags(match portability {
                true => vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR,
                false => vk::InstanceCreateFlags::empty(),
            });
        // SAFETY: `info` and what it points at live for the call.
        let instance = unsafe { entry.create_instance(&info, None) }.unwrap();
        // SAFETY: the instance is live.
        let physicals = unsafe { instance.enumerate_physical_devices() }.unwrap();
        // SAFETY: each is the instance's.
        let device_type = |physical: vk::PhysicalDevice| unsafe {
            instance
                .get_physical_device_properties(physical)
                .device_type
        };
        let physical = kinds()
            .into_iter()
            .find_map(|kind| {
                physicals
                    .iter()
                    .copied()
                    .find(|&physical| device_type(physical) == physical_type(kind))
            })
            .expect("a device of the kinds the cfg selects");
        // SAFETY: the instance's.
        let properties = unsafe { instance.get_physical_device_properties(physical) };
        let name = properties
            .device_name_as_c_str()
            .map_or("?".to_owned(), |name| name.to_string_lossy().into_owned());
        // SAFETY: the instance's.
        let families = unsafe { instance.get_physical_device_queue_family_properties(physical) };
        let family = families
            .iter()
            .position(|family| {
                family.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                    && family.timestamp_valid_bits > 0
            })
            .expect("a graphics queue family that writes timestamps") as u32;
        let valid_bits = families[family as usize].timestamp_valid_bits;
        // SAFETY: the instance's.
        let listed = unsafe { instance.enumerate_device_extension_properties(physical) }.unwrap();
        let names: BTreeSet<CString> = listed
            .iter()
            .filter_map(|extension| extension.extension_name_as_c_str().ok())
            .map(CStr::to_owned)
            .collect();
        let calibrated = gpu_time::Extension::among(&names);
        let mut enabled: Vec<&CStr> = calibrated
            .map(gpu_time::Extension::name)
            .into_iter()
            .collect();
        let portability_subset = c"VK_KHR_portability_subset";
        if names.contains(portability_subset) {
            enabled.push(portability_subset);
        }
        let exports =
            cfg!(target_os = "linux") && names.contains(ash::khr::external_fence_fd::NAME) && {
                let info = vk::PhysicalDeviceExternalFenceInfo::default()
                    .handle_type(vk::ExternalFenceHandleTypeFlags::SYNC_FD);
                let mut external = vk::ExternalFenceProperties::default();
                // SAFETY: the instance's; Vulkan 1.1's command.
                unsafe {
                    instance.get_physical_device_external_fence_properties(
                        physical,
                        &info,
                        &mut external,
                    )
                };
                external
                    .external_fence_features
                    .contains(vk::ExternalFenceFeatureFlags::EXPORTABLE)
            };
        if exports {
            enabled.push(ash::khr::external_fence_fd::NAME);
        }
        let enabled: Vec<*const std::ffi::c_char> = enabled.iter().map(|n| n.as_ptr()).collect();
        let priorities = [1.0];
        let queues = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(family)
            .queue_priorities(&priorities)];
        let mut features = vk::PhysicalDeviceVulkan12Features::default()
            .timeline_semaphore(true)
            .host_query_reset(true);
        let info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queues)
            .enabled_extension_names(&enabled)
            .push_next(&mut features);
        // SAFETY: `info` and what it points at live for the call.
        let device = unsafe { instance.create_device(physical, &info, None) }.unwrap();
        // SAFETY: the device is live and made of `physical` with the extension enabled, through
        // the loader.
        let calibration =
            unsafe { gpu_time::calibrated(&entry, &instance, physical, &device, calibrated) };
        let clock = GpuClock::new(valid_bits, properties.limits.timestamp_period, calibration)
            .expect("the family writes timestamps");
        assert!(
            clock.calibration.is_some(),
            "{name}: no calibrated timestamps, so no GPU end on the host's clock"
        );
        // SAFETY: the family and index were created.
        let queue = unsafe { device.get_device_queue(family, 0) };
        // SAFETY: the device is live; every info is valid.
        unsafe {
            let pool = device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default().queue_family_index(family),
                    None,
                )
                .unwrap();
            let commands = device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(2),
                )
                .unwrap();
            let (command, untimed) = (commands[0], commands[1]);
            let buffer = device
                .create_buffer(
                    &vk::BufferCreateInfo::default()
                        .size(BYTES)
                        .usage(vk::BufferUsageFlags::TRANSFER_DST),
                    None,
                )
                .unwrap();
            let requirements = device.get_buffer_memory_requirements(buffer);
            let memory_properties = instance.get_physical_device_memory_properties(physical);
            let types =
                &memory_properties.memory_types[..memory_properties.memory_type_count as usize];
            let index = (0..types.len())
                .find(|&index| {
                    requirements.memory_type_bits & (1 << index) != 0
                        && types[index]
                            .property_flags
                            .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
                })
                .or_else(|| {
                    (0..types.len())
                        .find(|&index| requirements.memory_type_bits & (1 << index) != 0)
                })
                .unwrap() as u32;
            let memory = device
                .allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(requirements.size)
                        .memory_type_index(index),
                    None,
                )
                .unwrap();
            device.bind_buffer_memory(buffer, memory, 0).unwrap();
            let queries = device
                .create_query_pool(
                    &vk::QueryPoolCreateInfo::default()
                        .query_type(vk::QueryType::TIMESTAMP)
                        .query_count(2),
                    None,
                )
                .unwrap();
            let mut typed = vk::SemaphoreTypeCreateInfo::default()
                .semaphore_type(vk::SemaphoreType::TIMELINE)
                .initial_value(0);
            let timeline = device
                .create_semaphore(
                    &vk::SemaphoreCreateInfo::default().push_next(&mut typed),
                    None,
                )
                .unwrap();
            let fence = device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .unwrap();
            let exported = exports.then(|| {
                let mut export = vk::ExportFenceCreateInfo::default()
                    .handle_types(vk::ExternalFenceHandleTypeFlags::SYNC_FD);
                let fence = device
                    .create_fence(&vk::FenceCreateInfo::default().push_next(&mut export), None)
                    .unwrap();
                (
                    fence,
                    ash::khr::external_fence_fd::Device::new(&instance, &device),
                )
            });
            // One command buffer for every run: the fill between two timestamps.
            device
                .begin_command_buffer(command, &vk::CommandBufferBeginInfo::default())
                .unwrap();
            device.cmd_write_timestamp(command, vk::PipelineStageFlags::TOP_OF_PIPE, queries, 0);
            let to_host = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ);
            let fill = |command| {
                device.cmd_fill_buffer(command, buffer, 0, BYTES, 0x5a5a_5a5a);
                device.cmd_pipeline_barrier(
                    command,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::HOST,
                    vk::DependencyFlags::empty(),
                    &[to_host],
                    &[],
                    &[],
                );
            };
            fill(command);
            device.cmd_write_timestamp(command, vk::PipelineStageFlags::BOTTOM_OF_PIPE, queries, 1);
            device.end_command_buffer(command).unwrap();
            device
                .begin_command_buffer(untimed, &vk::CommandBufferBeginInfo::default())
                .unwrap();
            fill(untimed);
            device.end_command_buffer(untimed).unwrap();
            Raw {
                _entry: entry,
                instance,
                device,
                name,
                queue,
                pool,
                command,
                untimed,
                buffer,
                memory,
                queries,
                timeline,
                value: 0,
                fence,
                exported,
                clock,
                latest: None,
            }
        }
    }

    /// The device's calibration reading, taken again where the last is [`REFRESH`] old, as the
    /// runtime's (`Timing::pair`).
    fn reading(&mut self) -> Reading {
        if let Some((at, reading)) = self.latest {
            if at.elapsed() < REFRESH {
                return reading;
            }
        }
        let (reading, _) = self.clock.pair().expect("a calibration reading");
        self.latest = Some((Instant::now(), reading));
        reading
    }

    /// One submission of the fill, waited for as `wait` says (through `waiter` for `armed` and
    /// `woken`), its host times returned.
    fn run(&mut self, wait: RawWait, waiter: Option<&Waiter>) -> RawSample {
        let device = &self.device;
        // SAFETY: the queries are in no pending submission (the last one was waited for), and
        // the device enables `hostQueryReset`.
        unsafe { device.reset_query_pool(self.queries, 0, 2) };
        self.value += 1;
        let value = self.value;
        if wait == RawWait::Armed {
            // In its wait before the submission is made.
            waiter.unwrap().ask().send(value).unwrap();
            std::thread::sleep(ARM_LEAD);
        }
        let fence = match wait {
            RawWait::Fence | RawWait::Untimed => self.fence,
            RawWait::SyncFd => self.exported.as_ref().unwrap().0,
            _ => vk::Fence::null(),
        };
        let commands = [match wait {
            RawWait::Untimed => self.untimed,
            _ => self.command,
        }];
        let semaphores = [self.timeline];
        let values = [value];
        let mut timeline =
            vk::TimelineSemaphoreSubmitInfo::default().signal_semaphore_values(&values);
        let submit_info = vk::SubmitInfo::default()
            .command_buffers(&commands)
            .signal_semaphores(&semaphores)
            .push_next(&mut timeline);
        let submit = host_nanos();
        // SAFETY: the command buffer is executable and not pending; the fence unsignaled.
        unsafe { device.queue_submit(self.queue, &[submit_info], fence) }.unwrap();
        let submitted = host_nanos();
        let wait_value = |device: &ash::Device| {
            let info = vk::SemaphoreWaitInfo::default()
                .semaphores(&semaphores)
                .values(&values);
            // SAFETY: the semaphore is the device's.
            unsafe { device.wait_semaphores(&info, TIMEOUT) }.unwrap();
        };
        let (entered, returned) = match wait {
            RawWait::Timeline => {
                let entered = host_nanos();
                wait_value(device);
                (entered, host_nanos())
            }
            RawWait::Fence | RawWait::Untimed => {
                let entered = host_nanos();
                // SAFETY: the fence is the device's, in the submission just made.
                unsafe { device.wait_for_fences(&[fence], true, TIMEOUT) }.unwrap();
                let returned = host_nanos();
                // SAFETY: signaled, its submission completed.
                unsafe { device.reset_fences(&[fence]) }.unwrap();
                (entered, returned)
            }
            RawWait::SyncFd => self.poll_sync_fd(fence),
            RawWait::Armed => waiter.unwrap().answer.recv().unwrap(),
            RawWait::Woken => {
                waiter.unwrap().ask().send(value).unwrap();
                waiter.unwrap().answer.recv().unwrap()
            }
            RawWait::Spin => {
                let entered = host_nanos();
                loop {
                    // SAFETY: the semaphore is the device's.
                    let read = unsafe { device.get_semaphore_counter_value(self.timeline) };
                    if read.unwrap() >= value {
                        break;
                    }
                }
                (entered, host_nanos())
            }
        };
        // Every wait above but the timeline's own orders the host after the signal through a
        // wait for it: this one returns at once.
        wait_value(device);
        if wait == RawWait::Untimed {
            return RawSample {
                submit,
                submitted,
                entered,
                returned,
                gpu_start: None,
                gpu_end: None,
            };
        }
        let mut stamps = [0u64; 2];
        // SAFETY: both queries were written by the completed submission.
        unsafe {
            device.get_query_pool_results(
                self.queries,
                0,
                &mut stamps,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            )
        }
        .unwrap();
        let reading = self.reading();
        RawSample {
            submit,
            submitted,
            entered,
            returned,
            gpu_start: Some(self.clock.place(reading.pair, stamps[0])),
            gpu_end: Some(self.clock.place(reading.pair, stamps[1])),
        }
    }

    /// Wait for the exported fence's submission by `poll` on a sync file of it: the moments the
    /// export was asked for and the poll returned.
    #[cfg(target_os = "linux")]
    fn poll_sync_fd(&self, fence: vk::Fence) -> (u64, u64) {
        let (_, external) = self.exported.as_ref().unwrap();
        let entered = host_nanos();
        let info = vk::FenceGetFdInfoKHR::default()
            .fence(fence)
            .handle_type(vk::ExternalFenceHandleTypeFlags::SYNC_FD);
        // SAFETY: the fence was made exportable as a sync file and has a signal pending; the
        // export resets it (copy transference).
        let fd = unsafe { external.get_fence_fd(&info) }.unwrap();
        if fd < 0 {
            // Signaled already.
            return (entered, host_nanos());
        }
        let mut polled = libc::pollfd {
            fd,
            events: libc::POLLIN,
            revents: 0,
        };
        // SAFETY: one valid pollfd.
        let ready = unsafe { libc::poll(&mut polled, 1, (TIMEOUT / 1_000_000) as i32) };
        let returned = host_nanos();
        assert_eq!(ready, 1, "the sync file signaled");
        // SAFETY: the file is the test's own.
        unsafe { libc::close(fd) };
        (entered, returned)
    }

    #[cfg(not(target_os = "linux"))]
    fn poll_sync_fd(&self, _: vk::Fence) -> (u64, u64) {
        unreachable!("no sync files here")
    }

    /// The ways to wait this device offers.
    fn waits(&self) -> Vec<RawWait> {
        [
            RawWait::Timeline,
            RawWait::Fence,
            RawWait::SyncFd,
            RawWait::Armed,
            RawWait::Woken,
            RawWait::Spin,
            RawWait::Untimed,
        ]
        .into_iter()
        .filter(|&wait| wait != RawWait::SyncFd || self.exported.is_some())
        .collect()
    }
}

impl Drop for Raw {
    fn drop(&mut self) {
        // SAFETY: nothing is pending (every submission was waited for); each object is the
        // device's, the device the instance's.
        unsafe {
            self.device.device_wait_idle().unwrap();
            if let Some((fence, _)) = self.exported.take() {
                self.device.destroy_fence(fence, None);
            }
            self.device.destroy_fence(self.fence, None);
            self.device.destroy_semaphore(self.timeline, None);
            self.device.destroy_query_pool(self.queries, None);
            self.device.destroy_buffer(self.buffer, None);
            self.device.free_memory(self.memory, None);
            self.device.destroy_command_pool(self.pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

/// The plain driver's latency from a submission's GPU end to each wait's return (the module
/// docs).
#[test]
#[ignore = "a measurement, not a check: run by name with --run-ignored all"]
fn raw_completion_latency() {
    let mut raw = Raw::new();
    eprintln!(
        "COMPLETION-LATENCY raw ({}): {RUNS} fills of {BYTES} bytes {GAP:?} apart; sync_fd {}",
        raw.name,
        match raw.exported.is_some() {
            true => "exported",
            false => "not offered",
        }
    );
    for wait in raw.waits() {
        let waiter = matches!(wait, RawWait::Armed | RawWait::Woken)
            .then(|| Waiter::new(raw.device.clone(), raw.timeline));
        let mut samples = Vec::with_capacity(RUNS);
        let mut next = Instant::now();
        for run in 0..WARM + RUNS {
            next += GAP;
            let sample = raw.run(wait, waiter.as_ref());
            if run >= WARM {
                samples.push(sample);
            }
            sleep_until(next);
        }
        drop(waiter);
        let of = |f: fn(&RawSample) -> f64| line(samples.iter().map(f).collect());
        let gpu = |f: fn(&RawSample) -> Option<f64>| match samples[0].gpu_end {
            Some(_) => line(samples.iter().filter_map(f).collect()),
            None => "-".to_owned(),
        };
        let late = (samples.iter())
            .filter(|s| s.gpu_end.is_some_and(|end| s.entered > end))
            .count();
        eprintln!(
            "COMPLETION-LATENCY raw {} ({}): gpu end -> wait returned {} us; wait entered -> \
             gpu end {} us ({late} of {} entered after it); submit returned -> gpu end {} us; \
             submit -> gpu start {} us; submit -> wait returned {} us; vkQueueSubmit {} us",
            wait.name(),
            raw.name,
            gpu(|s| Some(us(s.gpu_end?, s.returned))),
            gpu(|s| Some(us(s.entered, s.gpu_end?))),
            samples.len(),
            gpu(|s| Some(us(s.submitted, s.gpu_end?))),
            gpu(|s| Some(us(s.submit, s.gpu_start?))),
            of(|s| us(s.submit, s.returned)),
            of(|s| us(s.submit, s.submitted)),
        );
    }
}

/// A scratch directory for the runtime device's shader cache, removed when dropped.
struct Scratch(PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// One of the runtime's command buffers as its handlers saw it: host nanoseconds of its commit
/// call and the call's return, its completion-seen handler and its completed handler.
struct Committed {
    command_buffer: CommandBuffer,
    commit: u64,
    committed: u64,
    seen: Arc<Mutex<Option<u64>>>,
    completed: Arc<Mutex<Option<u64>>>,
}

/// The runtime's latency from a command buffer's GPU end to its completion seen, and its steps
/// between (the module docs).
#[test]
#[ignore = "a measurement, not a check: run by name with --run-ignored all"]
fn runtime_completion_latency() {
    let scratch = Scratch(
        std::env::temp_dir().join(format!("m2v-completion-latency-{}", std::process::id())),
    );
    std::fs::create_dir_all(&scratch.0).unwrap();
    let options = DeviceOptions {
        kinds: kinds(),
        shader_cache: ShaderCacheOptions {
            directory: Some(scratch.0.clone()),
            ..ShaderCacheOptions::default()
        },
        ..DeviceOptions::default()
    };
    let device = Device::system_default_with(&options).unwrap();
    let name = device.name().to_owned();
    let timing = &device.shared().timing;
    let calibrated = !timing.answers_from_host()
        && timing
            .clock()
            .is_some_and(|clock| clock.calibration.is_some());
    let queue = device.new_command_queue().unwrap();
    let buffer = device
        .new_buffer(BYTES, ResourceOptions::private())
        .unwrap();
    let fill = |value: u8| {
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.fill_buffer(&buffer, 0..BYTES, value).unwrap();
        blit.end_encoding().unwrap();
        command_buffer
    };
    let mut next = Instant::now();
    for run in 0..WARM {
        next += GAP;
        let command_buffer = fill(run as u8);
        command_buffer.commit().unwrap();
        command_buffer
            .wait_until_completed_for(Duration::from_secs(5))
            .unwrap();
        sleep_until(next);
    }
    let committed = commit_timed(&fill, &mut next);
    let times = times_of(calibrated, &committed);
    for (index, shape) in SHAPES.iter().enumerate() {
        let times: Vec<&Times> = (times.iter()).skip(index).step_by(SHAPES.len()).collect();
        print_runtime(&name, shape.name, calibrated, &times);
    }
}

/// One shape of the runtime's command buffers (the module docs).
struct Shape {
    name: &'static str,
    /// It has a completed handler besides its completion-seen handler.
    completed_handler: bool,
    /// It is committed with `commit_and_submit`, else with `commit`.
    and_submit: bool,
}

/// The shapes, committed in turn.
const SHAPES: [Shape; 3] = [
    Shape {
        name: "seen and completed handlers",
        completed_handler: true,
        and_submit: false,
    },
    Shape {
        name: "seen handler alone",
        completed_handler: false,
        and_submit: false,
    },
    Shape {
        name: "seen handler alone, commit_and_submit",
        completed_handler: false,
        and_submit: true,
    },
];

/// [`RUNS`] fills `fill` makes of each of the [`SHAPES`], in turn, so the shapes meet the same
/// host; committed [`GAP`] apart from `next` on and waited for at the end, with the runtime's
/// steps traced (`trace`).
fn commit_timed(
    fill: &dyn Fn(u8) -> CommandBuffer,
    next: &mut Instant,
) -> (Vec<Committed>, Vec<Traced>) {
    trace::start();
    let mut committed = Vec::with_capacity(SHAPES.len() * RUNS);
    for run in 0..SHAPES.len() * RUNS {
        let shape = &SHAPES[run % SHAPES.len()];
        *next += GAP;
        let command_buffer = fill(run as u8);
        let seen = Arc::new(Mutex::new(None));
        let completed = Arc::new(Mutex::new(None));
        let noted = Arc::clone(&seen);
        command_buffer
            .add_completion_seen_handler(Box::new(move |_: &CommandBuffer| {
                *noted.lock().unwrap() = Some(host_nanos());
            }))
            .unwrap();
        if shape.completed_handler {
            let noted = Arc::clone(&completed);
            command_buffer
                .add_completed_handler(Box::new(move |_: &CommandBuffer| {
                    *noted.lock().unwrap() = Some(host_nanos());
                }))
                .unwrap();
        }
        let commit = host_nanos();
        match shape.and_submit {
            true => command_buffer.commit_and_submit().unwrap(),
            false => command_buffer.commit().unwrap(),
        }
        let returned = host_nanos();
        committed.push(Committed {
            command_buffer,
            commit,
            committed: returned,
            seen,
            completed,
        });
        sleep_until(*next);
    }
    for each in &committed {
        each.command_buffer
            .wait_until_completed_for(Duration::from_secs(5))
            .unwrap();
    }
    (committed, trace::stop())
}

/// One command buffer's host times, nanoseconds, from its commit to its handlers, and its GPU
/// end where the device calibrates its timestamps.
struct Times {
    commit: u64,
    /// The commit call returned.
    committed: u64,
    /// The steps on the commit's way to its submission (`trace::submitting`), in order, between
    /// its commit call and its batch's watch.
    submitting: Vec<(&'static str, u64)>,
    watched: u64,
    entered: u64,
    returned: u64,
    turn: u64,
    /// The turn's steps of `finish` (`trace::step`), in order.
    turn_steps: Vec<(&'static str, u64)>,
    seen: u64,
    completed: Option<u64>,
    gpu_end: Option<u64>,
}

/// The times of `committed`'s command buffers, from their handlers and the steps traced meanwhile.
fn times_of(calibrated: bool, (committed, steps): &(Vec<Committed>, Vec<Traced>)) -> Vec<Times> {
    // Each turn's batch, and the steps of `finish` that follow it, up to the next turn's.
    let mut finished: Vec<(usize, u64, u64, Vec<(&'static str, u64)>)> = Vec::new();
    for step in steps {
        match *step {
            Traced::Finished { queue, value, seen } => {
                finished.push((queue, value, seen, Vec::new()));
            }
            Traced::Step { what, at } => {
                if let Some((.., turn)) = finished.last_mut() {
                    turn.push((what, at));
                }
            }
            _ => {}
        }
    }
    assert_eq!(
        finished.len(),
        committed.len(),
        "one finished batch ends each timed command buffer"
    );
    // Each command buffer's steps: the latest wait for a value up to its own that returned before
    // its turn, and its batch's watch.
    let waited = |queue: usize, value: u64, seen: u64| {
        steps.iter().rev().find_map(|step| match *step {
            Traced::Waited {
                queue: q,
                value: v,
                entered,
                returned,
            } if q == queue && v <= value && returned <= seen => Some((entered, returned)),
            _ => None,
        })
    };
    let watched = |queue: usize, value: u64| {
        steps.iter().find_map(|step| match *step {
            Traced::Watched {
                queue: q,
                value: v,
                at,
            } if q == queue && v == value => Some(at),
            _ => None,
        })
    };
    committed
        .iter()
        .zip(&finished)
        .map(|(each, (queue, value, turn, turn_steps))| {
            let (queue, value, turn) = (*queue, *value, *turn);
            let (entered, returned) = waited(queue, value, turn).expect("a wait before the turn");
            let gpu_end = calibrated
                .then(|| (each.command_buffer.gpu_end_time().unwrap() * 1e9).round() as u64);
            let watched = watched(queue, value).expect("its batch watched");
            let submitting = (steps.iter())
                .filter_map(|step| match *step {
                    Traced::Submitting { what, at } if (each.commit..=watched).contains(&at) => {
                        Some((what, at))
                    }
                    _ => None,
                })
                .collect();
            Times {
                commit: each.commit,
                committed: each.committed,
                submitting,
                watched,
                entered,
                returned,
                turn,
                turn_steps: turn_steps.clone(),
                seen: (each.seen.lock().unwrap()).expect("its completion-seen handler ran"),
                completed: *each.completed.lock().unwrap(),
                gpu_end,
            }
        })
        .collect()
}

/// Print one shape's lines, of `times`.
fn print_runtime(name: &str, shape: &str, calibrated: bool, times: &[&Times]) {
    let of = |f: &dyn Fn(&Times) -> f64| line(times.iter().map(|t| f(t)).collect());
    let ends = match calibrated {
        true => format!(
            "gpu end -> seen handler {} us; gpu end -> wait returned {} us; wait entered -> gpu \
             end {} us ({} of {} entered after it); ",
            of(&|t| us(t.gpu_end.unwrap(), t.seen)),
            of(&|t| us(t.gpu_end.unwrap(), t.returned)),
            of(&|t| us(t.entered, t.gpu_end.unwrap())),
            (times.iter())
                .filter(|t| t.entered > t.gpu_end.unwrap())
                .count(),
            times.len(),
        ),
        false => "no calibrated GPU end; ".to_owned(),
    };
    let completed = match times[0].completed {
        Some(_) => format!(
            "seen handler -> completed handler {} us; ",
            line(
                (times.iter())
                    .map(|t| us(t.seen, t.completed.expect("its completed handler ran")))
                    .collect()
            )
        ),
        None => String::new(),
    };
    eprintln!(
        "COMPLETION-LATENCY runtime, {shape} ({name}): {} fills of {BYTES} bytes {GAP:?} apart, \
         between the other shapes': {ends}commit -> commit returned {} us; commit -> watched {} us; \
         watched -> wait entered {} us; wait returned -> turn {} us; turn -> seen handler {} us; \
         {completed}commit -> seen handler {} us",
        times.len(),
        of(&|t| us(t.commit, t.committed)),
        of(&|t| us(t.commit, t.watched)),
        of(&|t| us(t.watched, t.entered)),
        of(&|t| us(t.returned, t.turn)),
        of(&|t| us(t.turn, t.seen)),
        of(&|t| us(t.commit, t.seen)),
    );
    // The turn's steps, each from the one before it (the turn first) and the last to the
    // completion-seen handler: the first of each name after the turn's batch, which a turn of
    // batches that end no command buffer may follow before the next one's. Each over the turns
    // that marked both its ends.
    const NAMES: [&str; 3] = ["events reached", "fences settled", "times placed"];
    let at = |t: &Times, name: &str| {
        (t.turn_steps.iter())
            .find(|&&(what, _)| what == name)
            .map(|&(_, at)| at)
    };
    let mut split = String::new();
    for step in 0..=NAMES.len() {
        let from = |t: &Times| match step.checked_sub(1) {
            None => Some(t.turn),
            Some(s) => at(t, NAMES[s]),
        };
        let to = |t: &Times| match NAMES.get(step) {
            None => Some(t.seen),
            Some(name) => at(t, name),
        };
        let (from_name, to_name) = (
            step.checked_sub(1).map_or("turn", |s| NAMES[s]),
            NAMES.get(step).copied().unwrap_or("seen handler"),
        );
        let spans: Vec<f64> = (times.iter())
            .filter_map(|t| Some(us(from(t)?, to(t)?)))
            .collect();
        match spans.is_empty() {
            true => split.push_str(&format!("; {from_name} -> {to_name} none")),
            false => split.push_str(&format!(
                "; {from_name} -> {to_name} {} us ({} turns)",
                line(spans.clone()),
                spans.len()
            )),
        }
    }
    eprintln!("COMPLETION-LATENCY runtime turn, {shape} ({name}){split}");
    // The commit's way to its submission, each step from the one before it that the command
    // buffer marked (the commit call first), and the last to its batch's watch: the first of each
    // name after the commit call. Each over the command buffers that marked it.
    const WAY: [&str; 6] = [
        "posted",
        "woke the thread",
        "thread woke",
        "taken in",
        "submit called",
        "submit returned",
    ];
    let mut way = String::new();
    for step in 0..=WAY.len() {
        let to_name = WAY.get(step).copied().unwrap_or("watched");
        let spans: Vec<f64> = (times.iter())
            .filter_map(|t| {
                let at = |name: &str| {
                    (t.submitting.iter())
                        .find(|&&(what, _)| what == name)
                        .map(|&(_, at)| at)
                };
                let to = match WAY.get(step) {
                    None => t.watched,
                    Some(name) => at(name)?,
                };
                // The latest step before it that this command buffer marked, else its commit.
                let from = (WAY[..step].iter().rev())
                    .find_map(|name| at(name))
                    .unwrap_or(t.commit);
                Some(us(from, to))
            })
            .collect();
        match spans.is_empty() {
            true => way.push_str(&format!("; -> {to_name} none")),
            false => way.push_str(&format!(
                "; -> {to_name} {} us ({} commits)",
                line(spans.clone()),
                spans.len()
            )),
        }
    }
    eprintln!("COMPLETION-LATENCY runtime commit, {shape} ({name}){way}");
}

/// The timed command buffers of each of [`runtime_commit_to_submission`]'s shapes, after
/// [`WARM`] untimed.
const SUBMISSION_RUNS: usize = 500;

/// The bytes of the long fill whose signal the held head's check waits for to land: long enough
/// that the head still waits as the next commit is made.
const LONG_BYTES: u64 = 64 << 20;

/// The shapes of [`runtime_commit_to_submission`] (the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Way {
    /// A fill alone on its queue.
    Alone,
    /// A fill committed right after another queue's head whose check waits for a signal to land.
    BehindALandingCheck,
    /// A render pass whose end the lowering thread has not lowered at the commit.
    RenderPass,
}

/// The shapes, each with its line's name.
const WAYS: [(Way, &str); 3] = [
    (Way::Alone, "a checked fill alone"),
    (
        Way::BehindALandingCheck,
        "a checked fill, another queue's head waiting for its check's signal to land",
    ),
    (
        Way::RenderPass,
        "a checked render pass, its end with the lowering thread",
    ),
];

/// One timed command buffer: host nanoseconds of its commit call and the call's return, and its
/// submission handler's run, with that handler's thread.
type Submitted = (u64, u64, Arc<Mutex<Option<(u64, Option<String>)>>>);

/// The runtime's time from `commit_and_submit` to a command buffer's submission, seen by its
/// submission handler, over the shapes of [`WAYS`] (the module docs).
#[test]
#[ignore = "a measurement, not a check: run by name with --run-ignored all"]
fn runtime_commit_to_submission() {
    let scratch = Scratch(
        std::env::temp_dir().join(format!("m2v-commit-to-submission-{}", std::process::id())),
    );
    std::fs::create_dir_all(&scratch.0).unwrap();
    let mut options = DeviceOptions {
        kinds: kinds(),
        shader_cache: ShaderCacheOptions {
            directory: Some(scratch.0.clone()),
            ..ShaderCacheOptions::default()
        },
        ..DeviceOptions::default()
    };
    options.policy.render_lowering = RenderLoweringPolicy::Worker;
    let device = Device::system_default_with(&options).unwrap();
    let name = device.name().to_owned();
    let queues: Vec<CommandQueue> = (0..3)
        .map(|_| device.new_command_queue().unwrap())
        .collect();
    let private = |bytes| {
        device
            .new_buffer(bytes, ResourceOptions::private())
            .unwrap()
    };
    let (small, other, long) = (private(BYTES), private(BYTES), private(LONG_BYTES));
    let landing = device.new_shared_event().unwrap();
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 256, 256, false);
    descriptor.usage = TextureUsage::RENDER_TARGET;
    descriptor.resource_options = ResourceOptions::private();
    let target = device.new_texture(&descriptor).unwrap();
    let committing = std::thread::current().name().map(str::to_owned);
    let mut signal = 0;
    for (way, shape) in WAYS {
        let before = device.work_counts();
        let mut timed: Vec<Submitted> = Vec::with_capacity(SUBMISSION_RUNS);
        let mut held = Vec::new();
        let mut next = Instant::now();
        for run in 0..WARM + SUBMISSION_RUNS {
            next += GAP;
            if way == Way::BehindALandingCheck {
                signal += 1;
                let signaller = filled(&queues[2], &long, run as u8);
                signaller
                    .encode_signal_event(EventRef::from(&landing), signal)
                    .unwrap();
                signaller.commit_and_submit().unwrap();
                let waiter = queues[1].command_buffer().unwrap();
                checked(&waiter);
                waiter
                    .encode_wait_for_event(EventRef::from(&landing), signal)
                    .unwrap();
                fill_into(&waiter, &other, run as u8);
                waiter.commit_and_submit().unwrap();
                held.extend([signaller, waiter]);
            }
            let command_buffer = queues[0].command_buffer().unwrap();
            checked(&command_buffer);
            let handled = Arc::new(Mutex::new(None));
            let noted = Arc::clone(&handled);
            command_buffer
                .add_submission_handler(Box::new(move |_: &CommandBuffer| {
                    let thread = std::thread::current().name().map(str::to_owned);
                    *noted.lock().unwrap() = Some((host_nanos(), thread));
                }))
                .unwrap();
            match way {
                Way::RenderPass => {
                    let encoder = command_buffer
                        .render_command_encoder(&pass(&target))
                        .unwrap();
                    encoder.end_encoding().unwrap();
                }
                Way::Alone | Way::BehindALandingCheck => {
                    fill_into(&command_buffer, &small, run as u8);
                }
            }
            let commit = host_nanos();
            command_buffer.commit_and_submit().unwrap();
            let returned = host_nanos();
            if run >= WARM {
                timed.push((commit, returned, handled));
            }
            held.push(command_buffer);
            sleep_until(next);
        }
        for command_buffer in &held {
            command_buffer
                .wait_until_completed_for(Duration::from_secs(5))
                .unwrap();
        }
        let after = device.work_counts();
        print_submissions(
            &name,
            shape,
            committing.as_deref(),
            &timed,
            (&before, &after),
        );
    }
}

/// A command buffer of `queue` that fills `target` whole with `value`.
fn filled(queue: &CommandQueue, target: &Buffer, value: u8) -> CommandBuffer {
    let command_buffer = queue.command_buffer().unwrap();
    fill_into(&command_buffer, target, value);
    command_buffer
}

/// Encode into `command_buffer` a fill of `target` whole with `value`.
fn fill_into(command_buffer: &CommandBuffer, target: &Buffer, value: u8) {
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.fill_buffer(target, 0..target.length(), value).unwrap();
    blit.end_encoding().unwrap();
}

/// Give `command_buffer` a submission check that passes it.
fn checked(command_buffer: &CommandBuffer) {
    command_buffer
        .set_submission_check(Arc::new(|_: &Submission<'_>| None))
        .unwrap();
}

/// Print one shape's line of [`runtime_commit_to_submission`]: from the commit call to its return
/// and to the submission handler, the threads the handlers ran on (`committing` the thread that
/// committed), and what the scheduler's work counts moved by over the shape.
fn print_submissions(
    name: &str,
    shape: &str,
    committing: Option<&str>,
    timed: &[Submitted],
    (before, after): (&WorkCounts, &WorkCounts),
) {
    let handled: Vec<(u64, u64, u64, Option<String>)> = (timed.iter())
        .map(|(commit, returned, handled)| {
            let (at, thread) =
                (handled.lock().unwrap().clone()).expect("its submission handler ran");
            (*commit, *returned, at, thread)
        })
        .collect();
    let mut threads: BTreeMap<String, usize> = BTreeMap::new();
    for (.., thread) in &handled {
        let thread = match thread.as_deref() {
            Some(thread) if Some(thread) == committing => "the committing thread",
            Some(thread) => thread,
            None => "an unnamed thread",
        };
        *threads.entry(thread.to_owned()).or_default() += 1;
    }
    let threads: Vec<String> = (threads.iter())
        .map(|(thread, count)| format!("{thread} {count}"))
        .collect();
    let moved = |field: fn(&WorkCounts) -> u64| field(after) - field(before);
    let counts = [
        ("commits_handed_off", moved(|c| c.commits_handed_off)),
        (
            "scheduler_passes_at_commit",
            moved(|c| c.scheduler_passes_at_commit),
        ),
        (
            "batches_submitted_at_commit",
            moved(|c| c.batches_submitted_at_commit),
        ),
        (
            "batches_checked_at_commit",
            moved(|c| c.batches_checked_at_commit),
        ),
        (
            "scheduler_wakes_commit",
            moved(|c| c.scheduler_wakes_commit),
        ),
        ("scheduler_wakes_host", moved(|c| c.scheduler_wakes_host)),
        (
            "scheduler_wakes_completion",
            moved(|c| c.scheduler_wakes_completion),
        ),
        ("scheduler_passes", moved(|c| c.scheduler_passes)),
        ("heads_held_for_checks", moved(|c| c.heads_held_for_checks)),
        (
            "heads_awaiting_check_signals",
            moved(|c| c.heads_awaiting_check_signals),
        ),
        (
            "heads_held_for_earlier_commits",
            moved(|c| c.heads_held_for_earlier_commits),
        ),
        (
            "batches_held_past_a_look",
            moved(|c| c.batches_held_past_a_look),
        ),
        ("batch_hold_nanos", moved(|c| c.batch_hold_nanos)),
        (
            "render_log_commits_by_thread",
            moved(|c| c.render_log_commits_by_thread),
        ),
        (
            "render_log_commits_by_program",
            moved(|c| c.render_log_commits_by_program),
        ),
    ];
    let counts: Vec<String> = (counts.iter())
        .map(|(field, moved)| format!("{field} {moved}"))
        .collect();
    eprintln!(
        "COMPLETION-LATENCY submission, {shape} ({name}): {} commit_and_submit {GAP:?} apart: \
         commit -> commit returned {} us; commit -> submission handler {} us; handlers on {}; {}",
        handled.len(),
        line(
            (handled.iter())
                .map(|(commit, returned, ..)| us(*commit, *returned))
                .collect()
        ),
        line(
            (handled.iter())
                .map(|(commit, _, at, _)| us(*commit, *at))
                .collect()
        ),
        threads.join(", "),
        counts.join(", "),
    );
}
