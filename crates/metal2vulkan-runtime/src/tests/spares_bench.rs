//! What a command pool's command buffers cost, as the device's spares make, free and trim them
//! (`spares.rs`): on the device's own queue family, a pool like the runtime's
//! (`VK_COMMAND_POOL_CREATE_TRANSIENT_BIT`, `VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT`)
//! allocates [`MADE`] primaries one at a time, each begun, recording a barrier and ended, as an
//! encoder's would; then frees all but [`KEPT`] of them as give-backs do, [`FREED_AT_ONCE`] in
//! one `vkFreeCommandBuffers`, each followed by the pool's trim (`vkTrimCommandPool`), and does it
//! again [`ROUNDS`] times, its command buffers allocated anew each round.
//!
//! Printed (`SPARES-BENCH`), per round: each call's time (`vkAllocateCommandBuffers`'s median and
//! slowest, the begins', the frees' and the trims'), and the process's memory before the makings,
//! once they were made, and after the frees and trims: where its DRM file reports it
//! (`/proc/self/fdinfo`, `drm-total-*` and `drm-resident-*`, summed over the device files the
//! process holds), and on macOS its physical footprint (`proc_pid_rusage`'s `ri_phys_footprint`,
//! which counts the GPU memory the process owns). Ignored: run it alone, by name, under
//! `--cfg m2v_gpu` (or `m2v_cpu_icd`):
//! `cargo nextest run -p metal2vulkan-runtime --lib --run-ignored all spares_bench --no-capture`.

use crate::spares::FREED_AT_ONCE;
use crate::tests::freed_memory::device;
use ash::vk;
use std::collections::BTreeMap;
use std::time::{Duration, Instant};

/// The command buffers each round makes.
const MADE: usize = 256;

/// The command buffers each round keeps as it frees the rest.
const KEPT: usize = 4;

/// The rounds.
const ROUNDS: usize = 6;

/// The process's memory by key, in KiB: its DRM memory (`drm-total-gtt`, `drm-resident-system0`,
/// ...) summed over its DRM files, and on macOS its physical footprint (`phys-footprint`); empty
/// where neither is reported.
fn process_memory() -> BTreeMap<String, u64> {
    let mut memory = BTreeMap::new();
    #[cfg(target_os = "macos")]
    {
        let mut info = std::mem::MaybeUninit::<libc::rusage_info_v4>::zeroed();
        // SAFETY: `info` is as large as the flavor's answer, and the pid is this process's.
        let read = unsafe {
            libc::proc_pid_rusage(
                std::process::id() as libc::c_int,
                libc::RUSAGE_INFO_V4,
                info.as_mut_ptr().cast(),
            )
        };
        if read == 0 {
            // SAFETY: the call filled it.
            let info = unsafe { info.assume_init() };
            memory.insert("phys-footprint".to_owned(), info.ri_phys_footprint / 1024);
        }
    }
    let Ok(entries) = std::fs::read_dir("/proc/self/fdinfo") else {
        return memory;
    };
    for entry in entries.flatten() {
        let Ok(text) = std::fs::read_to_string(entry.path()) else {
            continue;
        };
        if !text.contains("drm-driver:") {
            continue;
        }
        for line in text.lines() {
            let Some((key, value)) = line.split_once(':') else {
                continue;
            };
            if !(key.starts_with("drm-total-") || key.starts_with("drm-resident-")) {
                continue;
            }
            let mut words = value.split_whitespace();
            let (Some(number), unit) = (words.next(), words.next()) else {
                continue;
            };
            let Ok(number) = number.parse::<u64>() else {
                continue;
            };
            let kib = match unit {
                Some("MiB") => number * 1024,
                Some("KiB") => number,
                Some("GiB") => number * 1024 * 1024,
                _ => number / 1024,
            };
            *memory.entry(key.to_owned()).or_default() += kib;
        }
    }
    memory
}

/// The keys of `memory` that are nonzero, with their KiB.
fn said(memory: &BTreeMap<String, u64>) -> String {
    let nonzero: Vec<String> = (memory.iter())
        .filter(|(_, kib)| **kib > 0)
        .map(|(key, kib)| format!("{key} {kib} KiB"))
        .collect();
    match nonzero.is_empty() {
        true => "not reported".to_owned(),
        false => nonzero.join(", "),
    }
}

/// The median and slowest of `times`, in microseconds.
fn spread(times: &mut [Duration]) -> String {
    times.sort_unstable();
    let us = |time: Duration| time.as_secs_f64() * 1e6;
    format!(
        "median {:.1} us, slowest {:.1} us",
        us(times[times.len() / 2]),
        us(times[times.len() - 1])
    )
}

#[test]
#[ignore = "a measurement: run alone on its device by name"]
fn spares_bench() {
    let scratch = std::env::temp_dir().join(format!("m2v-spares-bench-{}", std::process::id()));
    std::fs::create_dir_all(&scratch).unwrap();
    let device = device(&scratch);
    let shared = device.shared();
    let vk = shared.vk();
    let info = vk::CommandPoolCreateInfo::default()
        .flags(
            vk::CommandPoolCreateFlags::TRANSIENT
                | vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
        )
        .queue_family_index(shared.queue_family);
    // SAFETY: the device is live and `info` valid.
    let pool = unsafe { vk.create_command_pool(&info, None) }.unwrap();
    let mut kept: Vec<vk::CommandBuffer> = Vec::new();
    for round in 0..ROUNDS {
        let before = process_memory();
        let (mut allocations, mut begins) = (Vec::new(), Vec::new());
        let mut made = Vec::new();
        while kept.len() + made.len() < MADE {
            let allocate = vk::CommandBufferAllocateInfo::default()
                .command_pool(pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let start = Instant::now();
            // SAFETY: the pool is this thread's alone.
            let buffer = unsafe { vk.allocate_command_buffers(&allocate) }.unwrap()[0];
            allocations.push(start.elapsed());
            made.push(buffer);
        }
        for &buffer in kept.iter().chain(&made) {
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
            let start = Instant::now();
            // SAFETY: the command buffer is the pool's, never pending, and its begin resets it
            // (the pool's reset flag); the barrier names no resource.
            unsafe {
                vk.begin_command_buffer(buffer, &begin).unwrap();
                vk.cmd_pipeline_barrier(
                    buffer,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[barrier],
                    &[],
                    &[],
                );
                vk.end_command_buffer(buffer).unwrap();
            }
            begins.push(start.elapsed());
        }
        let full = process_memory();
        // The first round keeps the last few it made, as a pool's recent uses keep theirs.
        if kept.is_empty() {
            kept = made.split_off(made.len() - KEPT.min(made.len()));
        }
        let (mut frees, mut trims) = (Vec::new(), Vec::new());
        for some in made.chunks(FREED_AT_ONCE) {
            let start = Instant::now();
            // SAFETY: none of them was submitted, and the pool is this thread's alone.
            unsafe { vk.free_command_buffers(pool, some) };
            frees.push(start.elapsed());
            let start = Instant::now();
            // SAFETY: the pool is this thread's alone.
            unsafe { vk.trim_command_pool(pool, vk::CommandPoolTrimFlags::empty()) };
            trims.push(start.elapsed());
        }
        let after = process_memory();
        println!(
            "SPARES-BENCH {} round {round}: {} allocated, {}; begin, barrier and end of {}, {}; \
             {} vkFreeCommandBuffers of up to {FREED_AT_ONCE}, {}; their vkTrimCommandPool, {}; \
             DRM memory before: {}; made: {}; freed and trimmed: {}",
            device.name(),
            made.len(),
            spread(&mut allocations),
            begins.len(),
            spread(&mut begins),
            frees.len(),
            spread(&mut frees),
            spread(&mut trims),
            said(&before),
            said(&full),
            said(&after),
        );
    }
    // SAFETY: none of its command buffers is pending; the pool is this thread's alone.
    unsafe {
        vk.free_command_buffers(pool, &kept);
        vk.destroy_command_pool(pool, None);
    }
    drop(device);
    let _ = std::fs::remove_dir_all(&scratch);
}
