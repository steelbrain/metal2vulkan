//! Not a consumer's example: what a device's `VkQueue`s cost the driver, by how many it takes, on
//! the first discrete GPU (else integrated, virtual or CPU device) the loader lists. The runtime's
//! queue plan (`src/device/create.rs`'s `queue_plan`) takes up to 16 queues of the graphics family
//! (its round robin and two for presenting) and two of a family of transfers alone (the staging
//! and fill queues), where the device has them; `DeviceOptions::queue_limits` takes fewer.
//!
//! ```sh
//! cargo run --release -p metal2vulkan-runtime --example queue_plan_cost -- \
//!     [--rounds R] [--hold H] [--runtime] [--exhaust PROCESSES] [--exhaust-only] \
//!     [--fills N,M,...]
//! ```
//!
//! Each shape is a count of graphics-family queues and of copy-family queues (none where the
//! device has no family of transfers alone): 1 + 0, 1 + 2 (the least plan's), 2 + 2, 4 + 2,
//! 8 + 2 and the graphics family's count + 2 (the plan's, 16 + 2 on NVIDIA). It prints
//! `QUEUE-PLAN-COST` lines:
//!
//! - **make**: `R` rounds (5 by default), each making and destroying one device of each shape in
//!   turn, through Vulkan alone (no extension, no feature): `vkCreateDevice` and
//!   `vkDestroyDevice` wall times, median, least and most per shape, and the slope per queue.
//! - **hold**: `H` devices of each shape held at once (4 by default), with the GPU's memory in use
//!   before and while they live, as `nvidia-smi` reports it where it runs (framebuffer and BAR1
//!   used, MiB), and that over `H` per device.
//! - **runtime** (`--runtime`): the runtime's own device of each queue limit (round robin 1, 2, 4,
//!   8 and the plan's, presenting 0 or 2), `Device::system_default_with` and its drop, after one
//!   device of each not counted: what the queues add to a device the runtime makes.
//! - **exhaust** (`--exhaust P`, opt-in: it runs the GPU out of what its devices need, which the
//!   driver logs, such as NVIDIA's `NV_ERR_NO_MEMORY` from `kchannelAllocMem`): for the plan's
//!   shape and the least plan's, up to `P` processes, started one after another, each making
//!   devices of the shape until the driver refuses one or it holds 6, all held until the last
//!   process is done: how many devices, and so `VkQueue`s, the GPU held at once, and each
//!   refusal's `VkResult`. While they are held, one more process makes the runtime's own device
//!   (`Device::system_default_with`, the plan): whether it came up (on its least plan where the
//!   driver refused the plan, which it announces) or what it answered.
//!   `--exhaust-only` skips the make and hold phases.
//! - **fills** (`--fills N,M,...`): for each count, that many devices of the least plan's shape
//!   held at once, then the runtime's device made beside them, as above: what the runtime's
//!   device making answers where the plan's `vkCreateDevice` still fits but little more does.
//! - **rooms** (`--rooms SPEC,...`, opt-in as the exhaust phase is): for each `SPEC`, `K+N` (or
//!   `N`, K of 0), K devices of the plan's shape and N of the least plan's held at once, then one
//!   more process asks for a device of every graphics-family count from the plan's down to one
//!   (the copy family's queues kept), each made and destroyed at once where the driver makes it:
//!   which counts the driver accepts beside what is held, and how long each `vkCreateDevice`
//!   took, refused or made.
//! - **overlap** (`--overlap K,...`): for the plan's shape, two between (round robins of 7 and 3
//!   with the two presenting) and the least plan's (16 + 2, 9 + 2, 5 + 2 and 1 + 2 on NVIDIA), K
//!   processes each making one device at the same moment, `R` rounds (`--rounds`, 3 by default):
//!   each `vkCreateDevice`'s time, median, least and most, and the time until all K were made, as
//!   when every device of a GPU is made again at once (a device lost).
//! - **runtime rooms** (`--runtime-rooms N,...`, opt-in as the exhaust phase is): as the rooms
//!   phase, beside N of the runtime's own devices (as the runtime fill phase makes them) instead
//!   of bare ones: a runtime device holds more of what the driver gives than a bare device of its
//!   queues.
//! - **runtime fill** (`--runtime-fill P`, opt-in as the exhaust phase is): up to `P` processes,
//!   started one after another, each making the runtime's own devices (the plan, each of its own
//!   options, so not one device returned again) until one is refused or it holds 6, all held
//!   until the last is done: each device's make time and the `VkQueue`s it took (the plan's, or
//!   what its `least-queue-plan` notice says), as the GPU fills.

use ash::vk;
use std::io::{BufRead, BufReader, Read, Write};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

/// The most devices one process of the exhaust phase makes: NVIDIA's driver refused one
/// process's eighth device on an RTX 2080 and an RTX 3090 whatever its queues.
const PER_PROCESS: usize = 6;

/// A shape: queues of the graphics family, and of the copy family.
type Shape = (u32, u32);

struct Gpu {
    _entry: ash::Entry,
    instance: ash::Instance,
    physical: vk::PhysicalDevice,
    name: String,
    driver: String,
    graphics: u32,
    graphics_count: u32,
    copy: Option<(u32, u32)>,
}

impl Gpu {
    fn new() -> Gpu {
        // SAFETY: loading the Vulkan loader runs its initialization; nothing else is assumed.
        let entry = unsafe { ash::Entry::load() }.expect("the Vulkan loader");
        let application = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_2);
        let info = vk::InstanceCreateInfo::default().application_info(&application);
        // SAFETY: `info` and what it points at live for the call.
        let instance = unsafe { entry.create_instance(&info, None) }.expect("an instance");
        // SAFETY: the instance is live.
        let physicals = unsafe { instance.enumerate_physical_devices() }.expect("devices");
        let types = [
            vk::PhysicalDeviceType::DISCRETE_GPU,
            vk::PhysicalDeviceType::INTEGRATED_GPU,
            vk::PhysicalDeviceType::VIRTUAL_GPU,
            vk::PhysicalDeviceType::CPU,
        ];
        // SAFETY: each is the instance's.
        let kind = |physical: vk::PhysicalDevice| unsafe {
            instance
                .get_physical_device_properties(physical)
                .device_type
        };
        let physical = (types.iter())
            .find_map(|&wanted| physicals.iter().copied().find(|&p| kind(p) == wanted))
            .expect("a device");
        let mut driver = vk::PhysicalDeviceDriverProperties::default();
        let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut driver);
        // SAFETY: the instance's; Vulkan 1.1's command.
        unsafe { instance.get_physical_device_properties2(physical, &mut properties) };
        let name = (properties.properties.device_name_as_c_str())
            .map_or("?".to_owned(), |name| name.to_string_lossy().into_owned());
        let driver = (driver.driver_info_as_c_str())
            .map_or("?".to_owned(), |info| info.to_string_lossy().into_owned());
        // SAFETY: the instance's.
        let families = unsafe { instance.get_physical_device_queue_family_properties(physical) };
        let wanted = vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE;
        let graphics = (families.iter())
            .position(|family| family.queue_flags.contains(wanted) && family.queue_count > 0)
            .expect("a family of graphics and compute work");
        let besides =
            vk::QueueFlags::TRANSFER | vk::QueueFlags::SPARSE_BINDING | vk::QueueFlags::PROTECTED;
        let copy = (families.iter().enumerate()).find(|(index, family)| {
            *index != graphics
                && family.queue_count > 0
                && family.queue_flags.contains(vk::QueueFlags::TRANSFER)
                && (family.queue_flags & !besides).is_empty()
        });
        Gpu {
            _entry: entry,
            name,
            driver,
            graphics: graphics as u32,
            graphics_count: families[graphics].queue_count,
            copy: copy.map(|(index, family)| (index as u32, family.queue_count)),
            instance,
            physical,
        }
    }

    /// The shapes the phases take (the module docs), the plan's last.
    fn shapes(&self) -> Vec<Shape> {
        let copies = self.copy.map_or(0, |(_, count)| count.min(2));
        let mut shapes = vec![(1, 0)];
        for graphics in [1, 2, 4, 8] {
            if graphics <= self.graphics_count && !shapes.contains(&(graphics, copies)) {
                shapes.push((graphics, copies));
            }
        }
        let plan = (self.graphics_count.min(16), copies);
        if !shapes.contains(&plan) {
            shapes.push(plan);
        }
        shapes
    }

    /// The least plan's shape and the plan's (the module docs).
    fn least_and_plan(&self) -> [Shape; 2] {
        let copies = self.copy.map_or(0, |(_, count)| count.min(2));
        [(1, copies), (self.graphics_count.min(16), copies)]
    }

    /// A device of `shape`, and how long `vkCreateDevice` took; the driver's answer where it
    /// refused it.
    fn make(&self, (graphics, copies): Shape) -> Result<(ash::Device, Duration), vk::Result> {
        let priorities = vec![1.0f32; graphics.max(copies) as usize];
        let mut infos = vec![vk::DeviceQueueCreateInfo::default()
            .queue_family_index(self.graphics)
            .queue_priorities(&priorities[..graphics as usize])];
        if let (Some((family, _)), true) = (self.copy, copies > 0) {
            infos.push(
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(family)
                    .queue_priorities(&priorities[..copies as usize]),
            );
        }
        let info = vk::DeviceCreateInfo::default().queue_create_infos(&infos);
        let start = Instant::now();
        // SAFETY: `info` and what it points at live for the call; `physical` is the instance's.
        let device = unsafe { self.instance.create_device(self.physical, &info, None) }?;
        Ok((device, start.elapsed()))
    }
}

/// Destroy `device`, and how long it took.
fn destroy(device: ash::Device) -> Duration {
    let start = Instant::now();
    // SAFETY: nothing of the device is in use: it made nothing.
    unsafe { device.destroy_device(None) };
    start.elapsed()
}

fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1e3
}

/// The median, least and most of `times`, in milliseconds.
fn spread(times: &mut [f64]) -> (f64, f64, f64) {
    times.sort_by(f64::total_cmp);
    (times[times.len() / 2], times[0], times[times.len() - 1])
}

/// The least-squares slope of `points` (x, y).
fn slope(points: &[(f64, f64)]) -> f64 {
    let n = points.len() as f64;
    let (sx, sy) = points
        .iter()
        .fold((0.0, 0.0), |(sx, sy), (x, y)| (sx + x, sy + y));
    let (mx, my) = (sx / n, sy / n);
    let (num, den) = points.iter().fold((0.0, 0.0), |(num, den), (x, y)| {
        (num + (x - mx) * (y - my), den + (x - mx) * (x - mx))
    });
    num / den
}

/// The GPU's framebuffer and BAR1 memory in use, MiB, as `nvidia-smi` reports them for its first
/// GPU; `None` where it does not run.
fn smi_memory() -> Option<(u64, u64)> {
    let output = Command::new("nvidia-smi")
        .args(["-i", "0", "-q", "-d", "MEMORY"])
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&output.stdout);
    let mut section = "";
    let (mut fb, mut bar1) = (None, None);
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.ends_with("Memory Usage") {
            section = trimmed;
        }
        if let Some(value) = trimmed.strip_prefix("Used") {
            let mib = value
                .trim_start_matches([' ', ':'])
                .trim_end_matches(" MiB")
                .parse::<u64>()
                .ok();
            match section {
                "FB Memory Usage" => fb = mib,
                "BAR1 Memory Usage" => bar1 = mib,
                _ => {}
            }
        }
    }
    Some((fb?, bar1?))
}

fn queues((graphics, copies): Shape) -> u32 {
    graphics + copies
}

fn make_phase(gpu: &Gpu, rounds: usize) {
    let shapes = gpu.shapes();
    let mut made: Vec<Vec<f64>> = vec![Vec::new(); shapes.len()];
    let mut dropped: Vec<Vec<f64>> = vec![Vec::new(); shapes.len()];
    // One device of each first, not counted: the driver's first device loads what later ones
    // share.
    for &shape in &shapes {
        if let Ok((device, _)) = gpu.make(shape) {
            destroy(device);
        }
    }
    for _ in 0..rounds {
        for (at, &shape) in shapes.iter().enumerate() {
            match gpu.make(shape) {
                Ok((device, took)) => {
                    made[at].push(ms(took));
                    dropped[at].push(ms(destroy(device)));
                }
                Err(result) => println!("QUEUE-PLAN-COST make {shape:?} refused: {result:?}"),
            }
        }
    }
    let mut made_points = Vec::new();
    let mut dropped_points = Vec::new();
    for (at, &shape) in shapes.iter().enumerate() {
        if made[at].is_empty() {
            continue;
        }
        let (m50, mlo, mhi) = spread(&mut made[at]);
        let (d50, dlo, dhi) = spread(&mut dropped[at]);
        let count = queues(shape);
        made_points.push((count as f64, m50));
        dropped_points.push((count as f64, d50));
        println!(
            "QUEUE-PLAN-COST make {}+{} = {count} queues: create median {m50:.1} ms ({mlo:.1} to \
             {mhi:.1}), destroy median {d50:.1} ms ({dlo:.1} to {dhi:.1}), {} rounds",
            shape.0,
            shape.1,
            made[at].len()
        );
    }
    if made_points.len() > 1 {
        println!(
            "QUEUE-PLAN-COST make slope: create {:.2} ms a queue, destroy {:.2} ms a queue",
            slope(&made_points),
            slope(&dropped_points)
        );
    }
}

fn hold_phase(gpu: &Gpu, hold: usize) {
    let settle = || std::thread::sleep(Duration::from_millis(500));
    settle();
    let Some(base) = smi_memory() else {
        println!("QUEUE-PLAN-COST hold: no nvidia-smi, no memory read");
        return;
    };
    println!(
        "QUEUE-PLAN-COST hold base: fb {} MiB, bar1 {} MiB",
        base.0, base.1
    );
    let mut points = Vec::new();
    for shape in gpu.shapes() {
        let mut devices = Vec::new();
        for _ in 0..hold {
            match gpu.make(shape) {
                Ok((device, _)) => devices.push(device),
                Err(result) => {
                    println!("QUEUE-PLAN-COST hold {shape:?} refused: {result:?}");
                    break;
                }
            }
        }
        settle();
        let held = smi_memory().unwrap_or(base);
        let count = devices.len();
        for device in devices {
            destroy(device);
        }
        settle();
        let after = smi_memory().unwrap_or(base);
        let per = |used: u64, base: u64| (used as f64 - base as f64) / count.max(1) as f64;
        println!(
            "QUEUE-PLAN-COST hold {}+{} = {} queues x {count} devices: fb {} MiB, bar1 {} MiB \
             (after: fb {}, bar1 {}): per device fb {:.1} MiB, bar1 {:.1} MiB",
            shape.0,
            shape.1,
            queues(shape),
            held.0,
            held.1,
            after.0,
            after.1,
            per(held.0, base.0),
            per(held.1, base.1)
        );
        points.push((
            queues(shape) as f64,
            per(held.0, base.0),
            per(held.1, base.1),
        ));
    }
    if points.len() > 1 {
        let fb: Vec<(f64, f64)> = points.iter().map(|&(q, fb, _)| (q, fb)).collect();
        let bar1: Vec<(f64, f64)> = points.iter().map(|&(q, _, bar1)| (q, bar1)).collect();
        println!(
            "QUEUE-PLAN-COST hold slope: fb {:.3} MiB a queue, bar1 {:.3} MiB a queue",
            slope(&fb),
            slope(&bar1)
        );
    }
}

fn runtime_phase() {
    use metal2vulkan_runtime::{Device, DeviceKind, DeviceOptions, QueueLimits};
    let cache = std::env::temp_dir().join(format!("queue-plan-cost-{}", std::process::id()));
    let limits: Vec<Option<QueueLimits>> = [1, 2, 4, 8]
        .iter()
        .flat_map(|&round_robin| {
            [0, 2].map(|presenting| {
                Some(QueueLimits {
                    round_robin,
                    presenting,
                })
            })
        })
        .chain([None])
        .collect();
    let options = |queue_limits: Option<QueueLimits>| {
        let mut options = DeviceOptions::default();
        options.kinds = vec![DeviceKind::Discrete, DeviceKind::Integrated];
        options.shader_cache =
            metal2vulkan_runtime::ShaderCacheOptions::in_directory(cache.clone());
        options.queue_limits = queue_limits;
        options
    };
    let mut made: Vec<Vec<f64>> = vec![Vec::new(); limits.len()];
    let mut dropped: Vec<Vec<f64>> = vec![Vec::new(); limits.len()];
    for round in 0..4 {
        for (at, &limit) in limits.iter().enumerate() {
            let start = Instant::now();
            let device = Device::system_default_with(&options(limit)).expect("a device");
            let took = start.elapsed();
            let start = Instant::now();
            drop(device);
            let gone = start.elapsed();
            if round > 0 {
                made[at].push(ms(took));
                dropped[at].push(ms(gone));
            }
        }
    }
    for (at, limit) in limits.iter().enumerate() {
        let (m50, mlo, mhi) = spread(&mut made[at]);
        let (d50, dlo, dhi) = spread(&mut dropped[at]);
        let name = match limit {
            Some(limit) => format!(
                "round robin {} presenting {}",
                limit.round_robin, limit.presenting
            ),
            None => "the plan".to_owned(),
        };
        println!(
            "QUEUE-PLAN-COST runtime {name}: make median {m50:.1} ms ({mlo:.1} to {mhi:.1}), \
             drop median {d50:.1} ms ({dlo:.1} to {dhi:.1}), {} rounds",
            made[at].len()
        );
    }
    let _ = std::fs::remove_dir_all(&cache);
}

/// A child of the exhaust phase: make devices of `shape` until refused or `most` (at most
/// [`PER_PROCESS`]), say how many (`HOLDING n`), hold them until stdin closes.
fn child(gpu: &Gpu, shape: Shape, most: usize) {
    let mut devices = Vec::new();
    let mut refusal = None;
    while devices.len() < most.min(PER_PROCESS) {
        match gpu.make(shape) {
            Ok((device, _)) => devices.push(device),
            Err(result) => {
                refusal = Some(result);
                break;
            }
        }
    }
    println!("HOLDING {} {refusal:?}", devices.len());
    std::io::stdout().flush().ok();
    let mut rest = Vec::new();
    std::io::stdin().read_to_end(&mut rest).ok();
    for device in devices {
        destroy(device);
    }
}

fn exhaust_phase(gpu: &Gpu, processes: usize) {
    for shape in gpu.least_and_plan() {
        held_beside_runtime(shape, processes, usize::MAX);
    }
}

/// The fill phase (`--fills N,M,...`): for each count, that many devices of the least plan's
/// shape held at once (or as many as the driver gives), then the runtime's own device made
/// beside them, as the exhaust phase makes it: where the plan's `vkCreateDevice` still fits but
/// little else does.
fn fill_phase(gpu: &Gpu, fills: &[usize]) {
    for &fill in fills {
        held_beside_runtime(
            gpu.least_and_plan()[0],
            fill.div_ceil(PER_PROCESS) + 1,
            fill,
        );
    }
}

/// Devices held by child processes: the children, holding theirs until their stdin closes; how
/// many devices; each child's first line, and each child's count, `r(...)` naming a refusal.
struct Held {
    children: Vec<Child>,
    devices: usize,
    lines: Vec<String>,
    counts: Vec<String>,
}

impl Held {
    /// Up to `processes` processes, started one after another, each making devices of `shape`
    /// until refused or 6, `most` in all.
    fn of(shape: Shape, processes: usize, most: usize) -> Held {
        let (graphics, copies) = (shape.0.to_string(), shape.1.to_string());
        Held::spawned(processes, most, |left| {
            vec!["--child".into(), graphics.clone(), copies.clone(), left]
        })
    }

    /// Up to `processes` processes, started one after another, each making the runtime's own
    /// devices until one is refused or it holds 6, `most` in all (`--runtime-fill-child`).
    fn runtime(processes: usize, most: usize) -> Held {
        Held::spawned(processes, most, |left| {
            vec!["--runtime-fill-child".into(), left]
        })
    }

    /// Up to `processes` children run with `args` of the devices left to make, started one after
    /// another until `most` are held or one makes none; each says how many it holds in its first
    /// line's second word.
    fn spawned(processes: usize, most: usize, args: impl Fn(String) -> Vec<String>) -> Held {
        let exe = std::env::current_exe().expect("this program");
        let mut held = Held {
            children: Vec::new(),
            devices: 0,
            lines: Vec::new(),
            counts: Vec::new(),
        };
        for _ in 0..processes {
            if held.devices >= most {
                break;
            }
            let mut child = Command::new(&exe)
                .args(args((most - held.devices).to_string()))
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .spawn()
                .expect("a child");
            let stdout = child.stdout.take().expect("its stdout");
            let mut line = String::new();
            BufReader::new(stdout).read_line(&mut line).ok();
            let made: usize = (line.split_whitespace().nth(1))
                .and_then(|n| n.parse().ok())
                .unwrap_or(0);
            // The refusal's `VkResult`, where the child's line names one (`Some(ERROR_...)`).
            let refusal = (line.split_once("Some("))
                .and_then(|(_, rest)| rest.split_once(')'))
                .map(|(code, _)| code.to_owned());
            held.counts.push(match refusal {
                Some(code) => format!("{made}r({code})"),
                None => made.to_string(),
            });
            held.lines.push(line.trim().to_owned());
            held.devices += made;
            held.children.push(child);
            if made == 0 {
                break;
            }
        }
        held
    }

    /// Close each child's stdin and wait for it: its devices destroyed.
    fn release(self) {
        for mut child in self.children {
            drop(child.stdin.take());
            child.wait().ok();
        }
    }
}

/// The rooms phase (`--rooms SPEC,...`, the module docs).
fn rooms_phase(gpu: &Gpu, specs: &[(usize, usize)]) {
    let [least, plan] = gpu.least_and_plan();
    let exe = std::env::current_exe().expect("this program");
    for &(plans, leasts) in specs {
        let first = Held::of(plan, plans.div_ceil(PER_PROCESS) + 1, plans);
        let second = Held::of(least, leasts.div_ceil(PER_PROCESS) + 1, leasts);
        let vk_queues =
            first.devices * queues(plan) as usize + second.devices * queues(least) as usize;
        let probe = Command::new(&exe)
            .arg("--probe-child")
            .output()
            .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_owned())
            .unwrap_or_else(|error| format!("not run: {error}"));
        println!(
            "QUEUE-PLAN-COST room beside {} of {}+{} [{}] and {} of {}+{} [{}] ({vk_queues} \
             VkQueues held): {probe}",
            first.devices,
            plan.0,
            plan.1,
            first.counts.join(" "),
            second.devices,
            least.0,
            least.1,
            second.counts.join(" ")
        );
        second.release();
        first.release();
        std::thread::sleep(Duration::from_secs(2));
    }
}

/// The rooms phase's probe: a device of each graphics-family count from the plan's down to one,
/// the copy family's queues kept, each destroyed at once where made; one line, the most queues
/// the driver made, each attempt (`g+c:ok|ERROR_...:ms`), and the made attempts' create time a
/// queue.
fn probe_child(gpu: &Gpu) {
    let [(_, copies), (most, _)] = gpu.least_and_plan();
    let mut attempts = Vec::new();
    let mut made = Vec::new();
    let mut largest = None;
    for graphics in (1..=most).rev() {
        let shape = (graphics, copies);
        let start = Instant::now();
        let answer = gpu.make(shape);
        let took = ms(start.elapsed());
        let what = match answer {
            Ok((device, _)) => {
                destroy(device);
                made.push((queues(shape) as f64, took));
                largest.get_or_insert(queues(shape));
                "ok".to_owned()
            }
            Err(result) => format!("{result:?}"),
        };
        attempts.push(format!("{graphics}+{copies}:{what}:{took:.1}"));
    }
    let per = match made.len() > 1 {
        true => format!("{:.2}", slope(&made)),
        false => "-".to_owned(),
    };
    println!(
        "most made {largest:?} queues; attempts {}; made create {per} ms a queue",
        attempts.join(" ")
    );
}

/// The runtime fill phase (`--runtime-fill P`, the module docs).
fn runtime_fill_phase(processes: usize) {
    let held = Held::runtime(processes, usize::MAX);
    let mut devices = 0;
    for (at, line) in held.lines.iter().enumerate() {
        let made: usize = (line.split_whitespace().nth(1))
            .and_then(|n| n.parse().ok())
            .unwrap_or(0);
        devices += made;
        println!("QUEUE-PLAN-COST runtime fill process {at} ({devices} devices held): {line}");
    }
    held.release();
}

/// The runtime rooms phase (`--runtime-rooms N,...`, the module docs).
fn runtime_rooms_phase(counts: &[usize]) {
    let exe = std::env::current_exe().expect("this program");
    for &count in counts {
        let held = Held::runtime(count.div_ceil(PER_PROCESS) + 1, count);
        let probe = Command::new(&exe)
            .arg("--probe-child")
            .output()
            .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_owned())
            .unwrap_or_else(|error| format!("not run: {error}"));
        println!(
            "QUEUE-PLAN-COST runtime room beside {} runtime devices [{}]: {probe}",
            held.devices,
            held.lines.join(" || ")
        );
        held.release();
        std::thread::sleep(Duration::from_secs(2));
    }
}

/// The runtime fill phase's child: the runtime's own devices (the plan), each of its own options
/// (a shader cache directory of its own, so not the device the last call made), until one is
/// refused or it holds `most` (at most [`PER_PROCESS`]), held until stdin closes; one line,
/// `RTFILL n`, then each device's make time and what it took (`plan`, or its `least-queue-plan`
/// notice's reason), and the refusal.
fn runtime_fill_child(most: usize) {
    use metal2vulkan_runtime::{Degradation, Device, DeviceKind, DeviceOptions, MessageSink};
    use std::sync::{Arc, Mutex};
    let cache = std::env::temp_dir().join(format!("queue-plan-cost-{}", std::process::id()));
    let heard: Arc<Mutex<Option<String>>> = Arc::default();
    let mut options = DeviceOptions::default();
    options.kinds = vec![DeviceKind::Discrete, DeviceKind::Integrated];
    let sink = Arc::clone(&heard);
    options.messages = Some(MessageSink::new(move |message| {
        if let Some(notice) = &message.degradation {
            if notice.degradation == Degradation::LeastQueuePlan {
                *sink.lock().unwrap() = Some(notice.reason.clone());
            }
        }
    }));
    let mut devices = Vec::new();
    let mut each = Vec::new();
    let mut refusal = None;
    while devices.len() < most.min(PER_PROCESS) {
        *heard.lock().unwrap() = None;
        let directory = cache.join(devices.len().to_string());
        options.shader_cache = metal2vulkan_runtime::ShaderCacheOptions::in_directory(directory);
        let start = Instant::now();
        let made = Device::system_default_with(&options);
        let took = ms(start.elapsed());
        match made {
            Ok(device) => {
                let what = heard.lock().unwrap().take();
                each.push(format!("{took:.0}ms:{}", what.as_deref().unwrap_or("plan")));
                devices.push(device);
            }
            Err(error) => {
                refusal = Some(format!("{took:.0}ms:{error}"));
                break;
            }
        }
    }
    println!(
        "RTFILL {} [{}] refused {refusal:?}",
        devices.len(),
        each.join(" | ")
    );
    std::io::stdout().flush().ok();
    let mut rest = Vec::new();
    std::io::stdin().read_to_end(&mut rest).ok();
    drop(devices);
    let _ = std::fs::remove_dir_all(&cache);
}

/// Up to `processes` processes, started one after another, each making devices of `shape` until
/// refused or 6, `most` in all, held while one more makes the runtime's own device; printed.
fn held_beside_runtime(shape: Shape, processes: usize, most: usize) {
    let exe = std::env::current_exe().expect("this program");
    let held = Held::of(shape, processes, most);
    let memory = smi_memory();
    let runtime = Command::new(&exe)
        .arg("--runtime-child")
        // Its notices (the least plan's among them) go to this one's stderr.
        .stderr(Stdio::inherit())
        .output()
        .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_owned())
        .unwrap_or_else(|error| format!("not run: {error}"));
    let (devices, counts) = (held.devices, held.counts.join(" "));
    let processes = held.children.len();
    held.release();
    println!(
        "QUEUE-PLAN-COST exhaust {}+{} = {} queues: {devices} devices ({} VkQueues) held at \
         once by {processes} processes [{counts}] (r: refused), memory then {memory:?}; then \
         the runtime's device: {runtime}",
        shape.0,
        shape.1,
        queues(shape),
        devices * queues(shape) as usize,
    );
    std::thread::sleep(Duration::from_secs(2));
}

/// A child of the exhaust phase made while the others hold their devices: the runtime's own device
/// of the plan, and what it answered (`RUNTIME made` or `RUNTIME refused: ...`). Where it took
/// its least plan it says so on stderr (`metal2vulkan-degrade-least-queue-plan`).
fn runtime_child() {
    use metal2vulkan_runtime::{Device, DeviceKind, DeviceOptions};
    let cache = std::env::temp_dir().join(format!("queue-plan-cost-{}", std::process::id()));
    let mut options = DeviceOptions::default();
    options.kinds = vec![DeviceKind::Discrete, DeviceKind::Integrated];
    options.shader_cache = metal2vulkan_runtime::ShaderCacheOptions::in_directory(cache.clone());
    match Device::system_default_with(&options) {
        Ok(device) => {
            println!("RUNTIME made");
            drop(device);
        }
        Err(error) => println!("RUNTIME refused: {error:?}"),
    }
    let _ = std::fs::remove_dir_all(&cache);
}

/// The overlap phase (`--overlap K,...`, the module docs): for each of its shapes and each count,
/// `rounds` rounds of that many processes making one device each at once.
fn overlap_phase(gpu: &Gpu, counts: &[usize], rounds: usize) {
    let exe = std::env::current_exe().expect("this program");
    let [(_, copies), (most, _)] = gpu.least_and_plan();
    // The plan's (16 graphics-family queues on NVIDIA: 14 round robin and 2 presenting), the
    // round robins of 7 and 3 with presenting's two, and the least plan's one.
    let mut shapes: Vec<Shape> = vec![(most, copies)];
    for graphics in [9, 5, 1] {
        if graphics < most {
            shapes.push((graphics, copies));
        }
    }
    for shape in shapes {
        for &count in counts {
            let mut each = Vec::new();
            let mut walls = Vec::new();
            let mut refused = Vec::new();
            for _ in 0..rounds {
                let mut children: Vec<(Child, BufReader<std::process::ChildStdout>)> = (0..count)
                    .map(|_| {
                        let mut child = Command::new(&exe)
                            .args([
                                "--overlap-child",
                                &shape.0.to_string(),
                                &shape.1.to_string(),
                            ])
                            .stdin(Stdio::piped())
                            .stdout(Stdio::piped())
                            .spawn()
                            .expect("a child");
                        let mut stdout = BufReader::new(child.stdout.take().expect("stdout"));
                        let mut ready = String::new();
                        stdout.read_line(&mut ready).ok();
                        (child, stdout)
                    })
                    .collect();
                let start = Instant::now();
                for (child, _) in &mut children {
                    let stdin = child.stdin.as_mut().expect("stdin");
                    stdin.write_all(b"go\n").ok();
                    stdin.flush().ok();
                }
                for (_, stdout) in &mut children {
                    let mut line = String::new();
                    stdout.read_line(&mut line).ok();
                    match line.split_whitespace().collect::<Vec<_>>()[..] {
                        ["MADE", took] => each.push(took.parse::<f64>().unwrap_or(f64::NAN)),
                        _ => refused.push(line.trim().to_owned()),
                    }
                }
                walls.push(ms(start.elapsed()));
                // Destroyed once all are made, so no destroy overlaps a making.
                for (mut child, _) in children {
                    drop(child.stdin.take());
                    child.wait().ok();
                }
            }
            if each.is_empty() {
                println!(
                    "QUEUE-PLAN-COST overlap {}+{} x{count}: none made {refused:?}",
                    shape.0, shape.1
                );
                continue;
            }
            let (m50, mlo, mhi) = spread(&mut each);
            let (w50, _, whi) = spread(&mut walls);
            println!(
                "QUEUE-PLAN-COST overlap {}+{} = {} queues x{count} at once: create median \
                 {m50:.1} ms ({mlo:.1} to {mhi:.1}), all made median {w50:.1} ms (most {whi:.1}), \
                 {rounds} rounds, refused {refused:?}",
                shape.0,
                shape.1,
                queues(shape)
            );
        }
    }
}

/// The overlap phase's child: says `READY` once its instance is made, makes a device of `shape`
/// when a line arrives on stdin, says `MADE ms` (or the refusal) and destroys it once stdin closes.
fn overlap_child(gpu: &Gpu, shape: Shape) {
    println!("READY");
    std::io::stdout().flush().ok();
    let mut go = String::new();
    std::io::stdin().read_line(&mut go).ok();
    match gpu.make(shape) {
        Ok((device, took)) => {
            println!("MADE {:.1}", ms(took));
            std::io::stdout().flush().ok();
            let mut rest = Vec::new();
            std::io::stdin().read_to_end(&mut rest).ok();
            destroy(device);
        }
        Err(result) => println!("REFUSED {result:?}"),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("--runtime-child") {
        runtime_child();
        return;
    }
    if args.first().map(String::as_str) == Some("--runtime-fill-child") {
        let most = (args.get(1)).map_or(PER_PROCESS, |most| most.parse().unwrap());
        runtime_fill_child(most);
        return;
    }
    let gpu = Gpu::new();
    if args.first().map(String::as_str) == Some("--child") {
        let shape = (args[1].parse().unwrap(), args[2].parse().unwrap());
        let most = (args.get(3)).map_or(PER_PROCESS, |most| most.parse().unwrap());
        child(&gpu, shape, most);
        return;
    }
    if args.first().map(String::as_str) == Some("--probe-child") {
        probe_child(&gpu);
        return;
    }
    if args.first().map(String::as_str) == Some("--overlap-child") {
        overlap_child(&gpu, (args[1].parse().unwrap(), args[2].parse().unwrap()));
        return;
    }
    let value = |flag: &str| {
        (args.iter().position(|arg| arg == flag))
            .and_then(|at| args.get(at + 1))
            .and_then(|value| value.parse::<usize>().ok())
    };
    println!(
        "QUEUE-PLAN-COST device {} ({}), graphics family {} of {} queues, copy family {:?}",
        gpu.name, gpu.driver, gpu.graphics, gpu.graphics_count, gpu.copy
    );
    if !args.iter().any(|arg| arg == "--exhaust-only") {
        make_phase(&gpu, value("--rounds").unwrap_or(5));
        hold_phase(&gpu, value("--hold").unwrap_or(4));
    }
    if args.iter().any(|arg| arg == "--runtime") {
        runtime_phase();
    }
    if let Some(processes) = value("--exhaust") {
        exhaust_phase(&gpu, processes);
    }
    let fills = (args.iter().position(|arg| arg == "--fills"))
        .and_then(|at| args.get(at + 1))
        .map(|list| {
            list.split(',')
                .filter_map(|n| n.parse().ok())
                .collect::<Vec<usize>>()
        });
    if let Some(fills) = fills {
        fill_phase(&gpu, &fills);
    }
    // Each `K+N` or `N`: K devices of the plan's shape and N of the least plan's.
    let rooms = (args.iter().position(|arg| arg == "--rooms"))
        .and_then(|at| args.get(at + 1))
        .map(|list| {
            (list.split(','))
                .filter_map(|spec| match spec.split_once('+') {
                    Some((plans, leasts)) => Some((plans.parse().ok()?, leasts.parse().ok()?)),
                    None => Some((0, spec.parse().ok()?)),
                })
                .collect::<Vec<(usize, usize)>>()
        });
    if let Some(rooms) = rooms {
        rooms_phase(&gpu, &rooms);
    }
    let overlap = (args.iter().position(|arg| arg == "--overlap"))
        .and_then(|at| args.get(at + 1))
        .map(|list| {
            (list.split(','))
                .filter_map(|n| n.parse().ok())
                .collect::<Vec<usize>>()
        });
    if let Some(counts) = overlap {
        overlap_phase(&gpu, &counts, value("--rounds").unwrap_or(3));
    }
    let runtime_rooms = (args.iter().position(|arg| arg == "--runtime-rooms"))
        .and_then(|at| args.get(at + 1))
        .map(|list| {
            (list.split(','))
                .filter_map(|n| n.parse().ok())
                .collect::<Vec<usize>>()
        });
    if let Some(counts) = runtime_rooms {
        runtime_rooms_phase(&counts);
    }
    if let Some(processes) = value("--runtime-fill") {
        runtime_fill_phase(processes);
    }
}
