//! The times of a `Shared` texture's CPU copies, `get_bytes` and `replace_region`, of
//! `BGRA8Unorm` textures of three sizes ([`SIZES`]: 256 x 256, 1920 x 1080 and 4096 x 4096), and
//! of a read through the GPU beside them: a blit of the texture into a `Shared` buffer, waited
//! for, then the host's copy out of that buffer's memory. Printed per size as a `CPU-COPY-TIMES`
//! line: the path the texture's copies take (on the host, from memory the host caches or not; on
//! the staging queue, `device/staging.rs`; through submissions), the best and median time of
//! each call, the blit's time from its commit to its completion and the host copy's.
//!
//! At 1920 x 1080, `get_bytes` is timed again while about [`BUSY`] of fills of a `Private`
//! buffer run on the GPU, committed on the program's command queue before the reads: each read
//! that returned while they ran is in the line's distribution under load.
//!
//! Each read is checked texel for texel against what the GPU or the host wrote. Times are a
//! release build's where the run is one.
//!
//! Asserted: a device where no memory the host caches may hold a texture's image
//! (`memory::host_cached_placement`: a discrete GPU with no device-local type the host caches)
//! and that lists a queue family to stage on beside its own ([`has_a_family_to_stage_on`]: an RX
//! 7900 XTX's compute family) copies its textures on the staging queue, never on the host
//! through its mapping of uncached memory (`device/staging.rs`, `device/create.rs`'s
//! `queue_plan`).
//!
//! After a runtime suspend: where Linux lists the device's PCI function with runtime power
//! management on (`power/control` `auto`), a measurement run by name waits, the device idle, for
//! the function to suspend (`power/runtime_status` `suspended`), then times each step of a
//! sequence: `replace_region` then `get_bytes` of a `Shared` texture made and copied before the
//! suspend, `get_bytes` then `replace_region`, and a command buffer of the program's queue (a fill:
//! its commit, then its wait) before them both; then both copies again, awake. A 1920 x 1080
//! texture in a pooled block, and a 4096 x 4096 one in a block of its own (past half a block,
//! `memory.rs`'s `BLOCK_BYTES`), each sequence twice, twelve suspends in all. Printed as `CPU-COPY-SUSPEND`
//! lines, with the path the copies took, the function's status after each step and the staging
//! queue's submissions the steps made (`WorkCounts::staging_flushes`).
//!
//! A device test: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`), on the kind
//! of device the cfg selects.

use crate::memory::MemoryClass;
use crate::{
    BlitOption, Buffer, CommandBuffer, CommandBufferStatus, CommandQueue, Device, DeviceKind,
    DeviceOptions, Origin, Region, ResourceOptions, Size, Texture, TextureDescriptor,
};
use ash::vk;
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// The textures' sizes, and the times each call is made at that size.
const SIZES: [(u64, u64, usize); 3] = [(256, 256, 20), (1920, 1080, 5), (4096, 4096, 3)];

/// The longest a command buffer here may take.
const WAIT: Duration = Duration::from_secs(60);

/// How long the GPU work the loaded reads run beside lasts, about.
const BUSY: Duration = Duration::from_millis(1500);

/// The bytes of the `Private` buffer the busy work fills.
const TARGET: u64 = 256 << 20;

/// The fills timed alone to size the busy work, and the most it makes.
const TIMED: u32 = 4;
const MOST_FILLS: u32 = 4096;

/// The most reads made while the busy work runs.
const LOADED_READS: usize = 20;

/// The device the cfg selects: a CPU device under `m2v_cpu_icd`, else a GPU.
fn device() -> Device {
    let options = DeviceOptions {
        kinds: match cfg!(m2v_cpu_icd) {
            true => vec![DeviceKind::Cpu],
            false => vec![
                DeviceKind::Discrete,
                DeviceKind::Integrated,
                DeviceKind::Virtual,
            ],
        },
        ..DeviceOptions::default()
    };
    Device::system_default_with(&options).unwrap()
}

/// Whether `device` lists a queue family its textures' CPU copies may stage on beside its own, as
/// Vulkan itself reports the physical device (so a device that took no staging queue is caught):
/// one whose copies take any texel, of transfers alone (sparse binding and protected memory
/// aside: a copy engine), or of graphics or compute work, other than the first family that does
/// both, on a device whose families are not all equivalent to that one (MoltenVK's are, and a
/// device takes queues of all of them).
fn has_a_family_to_stage_on(device: &Device) -> bool {
    // SAFETY: the handles are used only here, while the device lives, to query the physical
    // device.
    let raw = unsafe { device.raw() };
    // SAFETY: a physical device of the instance.
    let families = unsafe {
        raw.instance
            .get_physical_device_queue_family_properties(raw.physical_device)
    };
    let unit = vk::Extent3D {
        width: 1,
        height: 1,
        depth: 1,
    };
    let work = vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE;
    let Some(first) = (families.iter())
        .position(|family| family.queue_count > 0 && family.queue_flags.contains(work))
    else {
        return false;
    };
    let primary = families[first];
    let equivalent = |family: &vk::QueueFamilyProperties| {
        family.queue_flags == primary.queue_flags
            && family.queue_count == primary.queue_count
            && family.timestamp_valid_bits == primary.timestamp_valid_bits
            && family.min_image_transfer_granularity == primary.min_image_transfer_granularity
    };
    if families.iter().all(equivalent) {
        return false;
    }
    let besides =
        vk::QueueFlags::TRANSFER | vk::QueueFlags::SPARSE_BINDING | vk::QueueFlags::PROTECTED;
    (families.iter().enumerate()).any(|(index, family)| {
        let flags = family.queue_flags;
        let copy_engine = flags.contains(vk::QueueFlags::TRANSFER) && (flags & !besides).is_empty();
        index != first
            && family.queue_count > 0
            && family.min_image_transfer_granularity == unit
            && (copy_engine || flags.intersects(work))
    })
}

/// `len` bytes of a pattern of `salt`.
fn pattern(len: usize, salt: usize) -> Vec<u8> {
    (0..len).map(|i| ((i * 7 + salt) % 251) as u8).collect()
}

/// A `Shared` buffer of `len` bytes, `bytes` at its start where given.
fn shared_buffer(device: &Device, len: usize, bytes: Option<&[u8]>) -> Buffer {
    let buffer = device
        .new_buffer(len as u64, ResourceOptions::shared())
        .unwrap();
    if let Some(bytes) = bytes {
        let contents = buffer.contents().unwrap();
        // SAFETY: the buffer's `len` bytes, which no command uses yet.
        let into = unsafe { std::slice::from_raw_parts_mut(contents.as_ptr(), len) };
        into.copy_from_slice(bytes);
    }
    buffer
}

/// Each time in `times`, in milliseconds: the best and the median.
fn best_and_median(times: &mut [f64]) -> (f64, f64) {
    times.sort_by(f64::total_cmp);
    (times[0], times[times.len() / 2])
}

/// The time `f` took, in milliseconds.
fn millis(f: impl FnOnce()) -> f64 {
    let at = Instant::now();
    f();
    at.elapsed().as_secs_f64() * 1000.0
}

/// A command buffer of `queue`'s of `fills` fills of `target`, committed.
fn fills(queue: &CommandQueue, target: &Buffer, fills: u32) -> CommandBuffer {
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    for value in 0..fills {
        blit.fill_buffer(target, 0..TARGET, value as u8).unwrap();
    }
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer
}

/// `texture`'s `region` read by `get_bytes` (rows `row` bytes apart) while about [`BUSY`] of
/// fills run on `queue`, each read checked against `expected`, until the fills completed or
/// [`LOADED_READS`] were made: the times of those that returned while the fills ran, in
/// milliseconds, and how many reads were made.
fn loaded_reads(
    device: &Device,
    queue: &CommandQueue,
    texture: &Texture,
    region: Region,
    row: u64,
    expected: &[u8],
) -> (Vec<f64>, usize) {
    let target = device
        .new_buffer(TARGET, ResourceOptions::private())
        .unwrap();
    let timed = fills(queue, &target, TIMED);
    let at = Instant::now();
    timed.wait_until_completed_for(WAIT).unwrap();
    let each = at.elapsed() / TIMED;
    let count = (BUSY.as_secs_f64() / each.as_secs_f64().max(1e-6)).ceil() as u32;
    let busy = fills(queue, &target, count.clamp(TIMED, MOST_FILLS));
    let mut read = vec![0xee; expected.len()];
    let mut times = Vec::new();
    let mut made = 0;
    while made < LOADED_READS && busy.status() != CommandBufferStatus::Completed {
        read.fill(0xee);
        let took = millis(|| {
            (texture.get_bytes(&mut read, row, 0, region, 0, 0)).unwrap();
        });
        made += 1;
        assert!(read == expected, "get_bytes under load reads the texels");
        if busy.status() != CommandBufferStatus::Completed {
            times.push(took);
        }
    }
    busy.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(busy.error(), None);
    (times, made)
}

/// Commit a blit of `texture`'s level 0 into `buffer` (rows `row` bytes apart), or of `buffer`
/// into `texture` where `into_texture`, on `queue`, and wait for it.
fn blit(queue: &CommandQueue, texture: &Texture, buffer: &Buffer, row: u64, into_texture: bool) {
    let size = Size::new(texture.width(), texture.height(), 1);
    let origin = Origin::new(0, 0, 0);
    let image = row * texture.height();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.blit_command_encoder().unwrap();
    match into_texture {
        true => encoder.copy_from_buffer_to_texture(
            buffer,
            0,
            row,
            image,
            size,
            texture,
            0,
            0,
            origin,
            BlitOption::None,
        ),
        false => encoder.copy_from_texture_to_buffer(
            texture,
            0,
            0,
            origin,
            size,
            buffer,
            0,
            row,
            image,
            BlitOption::None,
        ),
    }
    .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
}

/// Every size of [`SIZES`]: a `Shared` `BGRA8Unorm` texture a render pass may draw into, as a
/// frame target is, written by a blit and read by `get_bytes`, read again through a blit into a
/// `Shared` buffer and the host's copy out of it, then written by `replace_region` and read by
/// `get_bytes`, at 1920 x 1080 also while GPU work runs. Prints the `CPU-COPY-TIMES` lines (the
/// module docs).
#[test]
fn a_shared_texture_s_cpu_copies_take_the_times_printed() {
    let device = device();
    let name = device.name().to_owned();
    let queue = device.new_command_queue().unwrap();
    let cached = crate::device::lock(&device.shared().memory).has_host_cached(MemoryClass::Private);
    let stages = !cached && has_a_family_to_stage_on(&device);
    let mut paths = Vec::new();
    for (width, height, calls) in SIZES {
        let row = 4 * width;
        let len = (row * height) as usize;
        let mut descriptor =
            TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, width, height, false);
        descriptor.usage = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
        descriptor.resource_options = ResourceOptions::shared();
        let before = device.work_counts();
        let texture = device.new_texture(&descriptor).unwrap();
        let made = device.work_counts();
        let hosted = made.images_started_on_the_host > before.images_started_on_the_host;
        let staged = made.staging_moves_queued > before.staging_moves_queued;
        let uncached = made.uncached_host_images > before.uncached_host_images;
        let path = match (hosted, staged) {
            (true, _) if uncached => "copied on the host from uncached memory",
            (true, _) => "copied on the host",
            (false, true) => "copied on the staging queue",
            (false, false) => "copied through submissions",
        };
        paths.push((width, height, path));
        let region = Region::new_2d(0, 0, width, height);
        // The GPU writes it, and `get_bytes` reads that.
        let drawn = pattern(len, 11);
        let source = shared_buffer(&device, len, Some(&drawn));
        blit(&queue, &texture, &source, row, true);
        let mut read = vec![0xee; len];
        let mut reads: Vec<f64> = (0..calls)
            .map(|_| {
                read.fill(0xee);
                let took = millis(|| {
                    (texture.get_bytes(&mut read, row, 0, region, 0, 0)).unwrap();
                });
                assert!(
                    read == drawn,
                    "{name} {width}x{height}: get_bytes reads the blit's texels"
                );
                took
            })
            .collect();
        // The same texels through the GPU: a blit into a `Shared` buffer, waited for, then the
        // host's copy out of its memory.
        let out = shared_buffer(&device, len, None);
        let mut blits = Vec::with_capacity(calls);
        let mut copies = Vec::with_capacity(calls);
        for _ in 0..calls {
            blits.push(millis(|| blit(&queue, &texture, &out, row, false)));
            let contents = out.contents().unwrap();
            read.fill(0xee);
            copies.push(millis(|| {
                // SAFETY: the buffer's `len` bytes; the blit that wrote them completed, and no
                // command uses them now.
                let bytes = unsafe { std::slice::from_raw_parts(contents.as_ptr(), len) };
                read.copy_from_slice(bytes);
            }));
            assert!(
                read == drawn,
                "{name} {width}x{height}: the blit into a buffer copies the texture's texels"
            );
        }
        // The host writes it, and `get_bytes` reads that.
        let written = pattern(len, 12);
        let mut writes: Vec<f64> = (0..calls)
            .map(|_| {
                millis(|| {
                    (texture.replace_region(region, 0, 0, &written, row, 0)).unwrap();
                })
            })
            .collect();
        read.fill(0xee);
        (texture.get_bytes(&mut read, row, 0, region, 0, 0)).unwrap();
        assert!(
            read == written,
            "{name} {width}x{height}: get_bytes reads replace_region's texels"
        );
        let under_load = match width == 1920 {
            true => {
                let (mut times, made) =
                    loaded_reads(&device, &queue, &texture, region, row, &written);
                match times.is_empty() {
                    true => format!(
                        "; with {BUSY:?} of fills running, none of {made} reads returned while \
                         they ran"
                    ),
                    false => {
                        let returned = times.len();
                        let (best, median) = best_and_median(&mut times);
                        let worst = times[returned - 1];
                        format!(
                            "; with {BUSY:?} of fills running, get_bytes best {best:.2} ms, \
                             median {median:.2} ms, worst {worst:.2} ms of the {returned} of \
                             {made} reads that returned while they ran"
                        )
                    }
                }
            }
            false => String::new(),
        };
        let (read_best, read_median) = best_and_median(&mut reads);
        let (blit_best, _) = best_and_median(&mut blits);
        let (copy_best, _) = best_and_median(&mut copies);
        let (write_best, write_median) = best_and_median(&mut writes);
        eprintln!(
            "CPU-COPY-TIMES {name} {width}x{height} BGRA8Unorm Shared, {path}: get_bytes best \
             {read_best:.2} ms, median {read_median:.2} ms of {calls}; through a blit into a \
             Shared buffer: the blit {blit_best:.2} ms, the host's copy {copy_best:.2} ms (best \
             of {calls}); replace_region best {write_best:.2} ms, median {write_median:.2} ms of \
             {calls}{under_load}"
        );
    }
    if stages {
        for (width, height, path) in paths {
            assert_eq!(
                path, "copied on the staging queue",
                "{name} {width}x{height}: a device with no memory the host caches for a texture's \
                 image and a queue family to stage on copies its textures on the staging queue"
            );
        }
    }
}

/// The sizes the measurement after a runtime suspend copies: a texture in a pooled block, and one
/// in a block of its own.
const SUSPENDED_SIZES: [(u64, u64); 2] = [(1920, 1080), (4096, 4096)];

/// How long past its autosuspend delay the measurement waits for the device to suspend.
const SUSPEND_MARGIN: Duration = Duration::from_secs(20);

/// The `power` directory Linux lists for `device`'s PCI function, where the device names its bus
/// address (`DeviceProperties::pci_bus`) and the directory lists a runtime status.
fn power_directory(device: &Device) -> Option<PathBuf> {
    let (domain, bus, slot, function) = device.properties().pci_bus?;
    let power = PathBuf::from(format!(
        "/sys/bus/pci/devices/{domain:04x}:{bus:02x}:{slot:02x}.{function:x}/power"
    ));
    runtime_status(&power).map(|_| power)
}

/// The function's runtime power status: `active`, `suspending`, `suspended`, `resuming`.
fn runtime_status(power: &Path) -> Option<String> {
    let status = std::fs::read_to_string(power.join("runtime_status")).ok()?;
    Some(status.trim().to_owned())
}

/// How long the device stayed idle before its function suspended, polling every 10 ms up to
/// `most`; `None` where it did not.
fn idle_until_suspended(power: &Path, most: Duration) -> Option<Duration> {
    let at = Instant::now();
    while at.elapsed() < most {
        if runtime_status(power).as_deref() == Some("suspended") {
            return Some(at.elapsed());
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    None
}

/// What a measurement after a runtime suspend does first, and after it.
#[derive(Clone, Copy, Debug)]
enum Step {
    /// The texture's `replace_region` of new texels.
    Write,
    /// The texture's `get_bytes`, checked against the texels last written.
    Read,
    /// A command buffer of the program's queue that fills a small `Private` buffer, committed and
    /// waited for: the device's own first submission, no texture copy. Timed as the commit, then
    /// the wait.
    Fill,
}

/// The steps after each suspend, in order: each first call, and the program's command buffer
/// first, ahead of the texture's copies.
const SEQUENCES: [&[Step]; 3] = [
    &[Step::Write, Step::Read],
    &[Step::Read, Step::Write],
    &[Step::Fill, Step::Read, Step::Write],
];

/// For each size of [`SUSPENDED_SIZES`], a `Shared` `BGRA8Unorm` texture a render pass may draw
/// into, made, written and read while the device is awake; then each sequence of
/// [`SEQUENCES`] twice, each after the device idled until its PCI function suspended, every step
/// timed, then a write and a read again, awake. Prints the `CPU-COPY-SUSPEND` lines (the module
/// docs). Each read is checked against the texels written.
#[test]
#[ignore = "a measurement: run alone on its device by name"]
fn a_shared_texture_s_first_cpu_copies_after_a_runtime_suspend_take_the_times_printed() {
    let device = device();
    let name = device.name().to_owned();
    let Some(power) = power_directory(&device) else {
        eprintln!("CPU-COPY-SUSPEND {name}: no PCI function's runtime power status listed");
        return;
    };
    let read_power = |file: &str| {
        let text = std::fs::read_to_string(power.join(file)).unwrap_or_default();
        text.trim().to_owned()
    };
    let control = read_power("control");
    let delay = read_power("autosuspend_delay_ms");
    eprintln!(
        "CPU-COPY-SUSPEND {name}: {} control {control}, autosuspend delay {delay} ms, status {}",
        power.display(),
        read_power("runtime_status"),
    );
    if control != "auto" {
        eprintln!("CPU-COPY-SUSPEND {name}: runtime power management is off");
        return;
    }
    let most = Duration::from_millis(delay.parse().unwrap_or(0)) + SUSPEND_MARGIN;
    let queue = device.new_command_queue().unwrap();
    let small = device.new_buffer(256, ResourceOptions::private()).unwrap();
    // The commit's time, then the wait's, in milliseconds.
    let fill = || {
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.fill_buffer(&small, 0..256, 7).unwrap();
        blit.end_encoding().unwrap();
        let committed = millis(|| command_buffer.commit().unwrap());
        let completed = millis(|| {
            command_buffer.wait_until_completed_for(WAIT).unwrap();
        });
        assert_eq!(command_buffer.error(), None);
        (committed, completed)
    };
    fill();
    for (width, height) in SUSPENDED_SIZES {
        let row = 4 * width;
        let len = (row * height) as usize;
        let mut descriptor =
            TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, width, height, false);
        descriptor.usage = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
        descriptor.resource_options = ResourceOptions::shared();
        let before = device.work_counts();
        let texture = device.new_texture(&descriptor).unwrap();
        let made = device.work_counts();
        let path = match (
            made.images_started_on_the_host > before.images_started_on_the_host,
            made.staging_moves_queued > before.staging_moves_queued,
        ) {
            (true, _) if made.uncached_host_images > before.uncached_host_images => {
                "copied on the host from uncached memory"
            }
            (true, _) => "copied on the host",
            (false, true) => "copied on the staging queue",
            (false, false) => "copied through submissions",
        };
        let region = Region::new_2d(0, 0, width, height);
        let mut read = vec![0xee; len];
        let mut salt = 20;
        let mut written = pattern(len, salt);
        // Awake: the texture's first copies, so the times below are the suspend's alone.
        (texture.replace_region(region, 0, 0, &written, row, 0)).unwrap();
        (texture.get_bytes(&mut read, row, 0, region, 0, 0)).unwrap();
        assert!(read == written, "{name} {width}x{height}: awake");
        let mut step = |step: Step, read: &mut [u8], written: &mut Vec<u8>| match step {
            Step::Write => {
                salt += 1;
                let bytes = pattern(len, salt);
                let took = millis(|| {
                    (texture.replace_region(region, 0, 0, &bytes, row, 0)).unwrap();
                });
                *written = bytes;
                format!("replace_region {took:.2} ms")
            }
            Step::Read => {
                read.fill(0xee);
                let took = millis(|| (texture.get_bytes(read, row, 0, region, 0, 0)).unwrap());
                assert!(*read == written[..], "{name} {width}x{height}: a read");
                format!("get_bytes {took:.2} ms")
            }
            Step::Fill => {
                let (committed, completed) = fill();
                format!("a command buffer's commit {committed:.2} ms, its wait {completed:.2} ms")
            }
        };
        for sequence in SEQUENCES.iter().chain(&SEQUENCES) {
            let Some(idle) = idle_until_suspended(&power, most) else {
                eprintln!(
                    "CPU-COPY-SUSPEND {name}: the device did not suspend within {most:?} \
                     (status {})",
                    read_power("runtime_status")
                );
                return;
            };
            let counts = device.work_counts();
            let mut steps = Vec::new();
            for &each in *sequence {
                let took = step(each, &mut read, &mut written);
                let status = runtime_status(&power).unwrap_or_default();
                steps.push(format!("{took} ({status})"));
            }
            let flushed = device.work_counts().staging_flushes - counts.staging_flushes;
            let wrote_again = step(Step::Write, &mut read, &mut written);
            let got_again = step(Step::Read, &mut read, &mut written);
            eprintln!(
                "CPU-COPY-SUSPEND {name} {width}x{height} BGRA8Unorm Shared, {path}: suspended \
                 after {:.2} s idle; then {}; {flushed} staging submission(s); awake again: \
                 {wrote_again}, {got_again}",
                idle.as_secs_f64(),
                steps.join(", "),
            );
        }
    }
}
