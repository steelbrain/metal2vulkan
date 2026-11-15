//! What new buffers of 16 KiB or more cost to make: a batch of [`COUNT`] buffers whose lengths
//! spread geometrically from 16 KiB + 1 to 4 MiB, `Shared` and `Private`, made, their first fills
//! run by one empty command buffer committed and waited for (a `Private` buffer's zeros are a
//! device fill its device's next submission records first, `texture/first_move.rs`; a `Shared`
//! one's are cleared on the host as it is made), and dropped, [`ROUNDS`] times. Printed as
//! `BUFFER-MAKING` lines: each round's making and fills, in milliseconds, the bytes the device
//! allocated (`WorkCounts::device_memory_allocated`) and the host cleared
//! (`new_buffer_bytes_cleared`), and the batch's lengths and `VkMemoryRequirements.size` summed;
//! the first round makes every buffer in new memory, the later ones take parked buffers
//! (`recycle.rs`). And `BUFFER-REQUIREMENTS` lines: a 16385-byte buffer's `VkBuffer`'s memory
//! requirements, and a plain `VkBuffer`'s of 16400 and of 32768 bytes on the same device. Run it
//! on two trees to compare them. Ignored: run it in release under the GPU safety cfg.
//!
//! Device tests: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`), on the kind
//! of device the cfg selects.

use super::blit_holds::{device, Scratch};
use crate::{Buffer, Device, ResourceOptions, WorkCounts};
use ash::vk;
use std::time::{Duration, Instant};

const WAIT: Duration = Duration::from_secs(120);
/// The buffers a batch makes, and the rounds.
const COUNT: u32 = 1000;
const ROUNDS: usize = 5;
/// The shortest and longest length of a batch.
const SHORTEST: u64 = (16 << 10) + 1;
const LONGEST: u64 = 4 << 20;

/// The batch's lengths: geometric from [`SHORTEST`] to [`LONGEST`].
fn lengths() -> Vec<u64> {
    let ratio = (LONGEST as f64 / SHORTEST as f64).powf(1.0 / f64::from(COUNT - 1));
    (0..COUNT)
        .map(|at| ((SHORTEST as f64) * ratio.powi(at as i32)).round() as u64)
        .map(|length| length.clamp(SHORTEST, LONGEST))
        .collect()
}

/// `buffer`'s `VkBuffer`'s memory requirements.
fn requirements(device: &Device, buffer: &Buffer) -> vk::MemoryRequirements {
    let (raw, _) = buffer.at(0);
    // SAFETY: `raw` is the live buffer's `VkBuffer`, of this device.
    unsafe { device.shared().vk().get_buffer_memory_requirements(raw) }
}

/// A plain `VkBuffer` of `size` bytes' memory requirements, of storage and transfer usage.
fn plain_requirements(device: &Device, size: u64) -> vk::MemoryRequirements {
    let raw = device.shared().vk();
    let usage = vk::BufferUsageFlags::STORAGE_BUFFER
        | vk::BufferUsageFlags::TRANSFER_SRC
        | vk::BufferUsageFlags::TRANSFER_DST;
    let info = vk::BufferCreateInfo::default().size(size).usage(usage);
    // SAFETY: the device is live and `info` valid.
    let buffer = unsafe { crate::creation::buffer(raw, &info) }.unwrap();
    // SAFETY: `buffer` is a live buffer of the device, unbound and unused.
    let requirements = unsafe { raw.get_buffer_memory_requirements(buffer) };
    // SAFETY: as above.
    unsafe { raw.destroy_buffer(buffer, None) };
    requirements
}

/// One round of `options`: the batch made, its fills run, then dropped.
fn round(device: &Device, options: ResourceOptions, lengths: &[u64]) -> String {
    let queue = device.new_command_queue().unwrap();
    let before = device.work_counts();
    let started = Instant::now();
    let batch: Vec<Buffer> = (lengths.iter())
        .map(|&length| device.new_buffer(length, options).unwrap())
        .collect();
    let made = started.elapsed();
    let started = Instant::now();
    let command_buffer = queue.command_buffer().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    let filled = started.elapsed();
    let after = device.work_counts();
    let required: u64 = (batch.iter())
        .map(|buffer| requirements(device, buffer).size)
        .sum();
    let lengths: u64 = batch.iter().map(Buffer::length).sum();
    drop(batch);
    let delta = |of: fn(&WorkCounts) -> u64| of(&after) - of(&before);
    format!(
        "made {:.2} ms, fills {:.2} ms, allocated {} bytes, host cleared {} bytes, lengths {} \
         bytes, required {} bytes",
        made.as_secs_f64() * 1e3,
        filled.as_secs_f64() * 1e3,
        delta(|counts| counts.device_memory_allocated),
        delta(|counts| counts.new_buffer_bytes_cleared),
        lengths,
        required,
    )
}

#[test]
#[ignore = "a release benchmark: run with --ignored --nocapture"]
fn new_buffers_of_16_kib_or_more() {
    let scratch = Scratch::new("buffer-making-bench");
    let device = device(&scratch.0);
    let name = device.name();
    for (storage, options) in [
        ("Shared", ResourceOptions::shared()),
        ("Private", ResourceOptions::private()),
    ] {
        let buffer = device.new_buffer(SHORTEST, options).unwrap();
        let made = requirements(&device, &buffer);
        println!(
            "BUFFER-REQUIREMENTS on {name}: a {SHORTEST}-byte {storage} buffer's VkBuffer: size \
             {}, alignment {}",
            made.size, made.alignment
        );
    }
    for size in [SHORTEST.next_multiple_of(16), 32768] {
        let plain = plain_requirements(&device, size);
        println!(
            "BUFFER-REQUIREMENTS on {name}: a plain VkBuffer of {size} bytes: size {}, \
             alignment {}",
            plain.size, plain.alignment
        );
    }
    let lengths = lengths();
    for (storage, options) in [
        ("Shared", ResourceOptions::shared()),
        ("Private", ResourceOptions::private()),
    ] {
        for at in 0..ROUNDS {
            let line = round(&device, options, &lengths);
            println!("BUFFER-MAKING on {name}: {COUNT} {storage} buffers, round {at}: {line}");
        }
    }
}
