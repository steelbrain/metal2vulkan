//! Devices of one GPU made one after another in one process, each with a command queue, all kept
//! alive, until one fails to be made or [`MOST`] are live: how many the driver makes, and what each
//! takes of the GPU's memory heaps (the runtime's live blocks in each) and, where `nvidia-smi`
//! answers, of the GPU's BAR (its BAR1 use). Each device has a shader cache and a message sink of
//! its own, so a device of its own (`create.rs` groups the devices it makes by their options and
//! sink), as each process of a host's guests makes one. Printed as `MANY-DEVICES` lines.
//!
//! The driver refuses devices of one process past a count of its own (seven on an RTX 2080 and an
//! RTX 3090, NVIDIA 620.70.06, the eighth with two `VkQueue`s as with 17), so a second test makes
//! one device, with a command queue, in a process of its own and holds it for [`HELD`]: a run
//! starts it in several processes at once, as a host's guests are, each printing what it took as
//! `ONE-DEVICE` lines.
//!
//! A GPU hazard: it makes devices up to the driver's limit (docs/VALIDATION.md), so on a GPU it
//! compiles only under `m2v_gpu_hazards`; a CPU ICD has no BAR, and it does not compile there.

use super::blit_holds::Scratch;
use crate::device::lock;
use crate::{Device, DeviceKind, DeviceOptions, Error, MessageSink, ShaderCacheOptions};
use std::path::Path;
use std::process::Command;
use std::time::{Duration, Instant};

/// How long [`a_device_is_held_beside_other_processes_devices`] holds its device.
const HELD: Duration = Duration::from_secs(60);

/// The most devices the run keeps live: past the 16 of the plan's queues an RTX 3090 made
/// beside the BAR it shares with them.
const MOST: usize = 24;

/// The BAR1 MiB in use of the first GPU `nvidia-smi` lists, where it answers.
fn bar1_used_mib() -> Option<u64> {
    let output = Command::new("nvidia-smi")
        .args(["-q", "-d", "MEMORY"])
        .output()
        .ok()?;
    let text = String::from_utf8(output.stdout).ok()?;
    let mut lines = text
        .lines()
        .skip_while(|line| !line.contains("BAR1 Memory Usage"));
    let used = lines.find(|line| line.trim_start().starts_with("Used"))?;
    let value = used.split(':').nth(1)?.trim();
    value.trim_end_matches("MiB").trim().parse().ok()
}

/// A device of the GPU with its shader cache in `cache` and a message sink of its own, which
/// prints its notices.
fn own_device(cache: &Path, at: usize) -> Result<Device, Error> {
    let options = DeviceOptions {
        kinds: vec![DeviceKind::Discrete, DeviceKind::Integrated],
        shader_cache: ShaderCacheOptions {
            directory: Some(cache.to_owned()),
            ..ShaderCacheOptions::default()
        },
        messages: Some(MessageSink::new(move |message| {
            if let Some(notice) = &message.degradation {
                println!("MANY-DEVICES device={at} notice {notice:?}");
            }
        })),
        ..DeviceOptions::default()
    };
    Device::system_default_with(&options)
}

/// What `device` holds of its memory now: each heap's size and live bytes, and each pool's pooled
/// blocks, in MiB.
fn holdings(device: &Device) -> String {
    let memory = lock(&device.shared().memory);
    let mib = |bytes: u64| bytes as f64 / f64::from(1 << 20);
    let heaps: Vec<String> = (memory.heaps_live().iter())
        .map(|&(size, live)| format!("{:.0}/{:.1}", mib(size), mib(live)))
        .collect();
    let pools: Vec<String> = (memory.pooled_by_pool().iter())
        .map(|((holds, type_index), sizes)| {
            let sizes: Vec<String> = sizes
                .iter()
                .map(|&size| format!("{:.0}", mib(size)))
                .collect();
            format!("{holds:?}@{type_index}:[{}]", sizes.join(","))
        })
        .collect();
    format!(
        "heaps_mib(size/live)=[{}] pooled_mib=[{}]",
        heaps.join(" "),
        pools.join(" ")
    )
}

#[test]
fn devices_are_made_until_the_driver_refuses_one() {
    let scratch = Scratch::new("many-devices");
    let before = bar1_used_mib();
    println!("MANY-DEVICES start bar1_used_mib={before:?}");
    let mut live = Vec::new();
    let mut refused = None;
    for at in 0..MOST {
        let cache = scratch.0.join(format!("device-{at}"));
        std::fs::create_dir_all(&cache).unwrap();
        let started = Instant::now();
        let made = match own_device(&cache, at) {
            Ok(made) => made,
            Err(error) => {
                refused = Some(at);
                println!(
                    "MANY-DEVICES device={at} refused after {:.1} ms: {error} ({error:?})",
                    started.elapsed().as_secs_f64() * 1e3
                );
                break;
            }
        };
        let opened = started.elapsed();
        let queue = made.new_command_queue();
        let queues = made.shared().queues.len();
        println!(
            "MANY-DEVICES device={at} made_ms={:.1} vk_queues={queues} queue_made={} \
             bar1_used_mib={:?} {}",
            opened.as_secs_f64() * 1e3,
            queue.is_ok(),
            bar1_used_mib(),
            holdings(&made)
        );
        live.push((made, queue));
    }
    println!(
        "MANY-DEVICES end live={} refused_at={refused:?} bar1_used_mib={:?}",
        live.len(),
        bar1_used_mib()
    );
    assert!(!live.is_empty(), "no device was made");
}

#[test]
fn a_device_is_held_beside_other_processes_devices() {
    let scratch = Scratch::new("one-device");
    let before = bar1_used_mib();
    let started = Instant::now();
    let made = match own_device(&scratch.0, 0) {
        Ok(made) => made,
        Err(error) => {
            println!(
                "ONE-DEVICE refused after {:.1} ms: {error} ({error:?}) bar1_used_mib={before:?}",
                started.elapsed().as_secs_f64() * 1e3
            );
            return;
        }
    };
    let opened = started.elapsed();
    let queue = made.new_command_queue();
    println!(
        "ONE-DEVICE made_ms={:.1} vk_queues={} queue_made={} bar1_used_mib(before,after)=({before:?},{:?}) {}",
        opened.as_secs_f64() * 1e3,
        made.shared().queues.len(),
        queue.is_ok(),
        bar1_used_mib(),
        holdings(&made)
    );
    std::thread::sleep(HELD);
    drop(queue);
}
