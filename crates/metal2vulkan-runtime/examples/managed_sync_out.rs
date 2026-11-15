//! Not a consumer's example: what `synchronize_resource` of `Managed` buffers over the caller's
//! pages costs a command buffer on a device without unified memory, where each such buffer is two
//! copies (the caller's pages and the GPU's own) and a synchronize copies the GPU's copy out to the
//! pages. Two shapes a guest workload showed:
//!
//! - **Synchronizes of 40 buffers a command buffer**, 4,399,104 bytes each (about 176 MB a command
//!   buffer, 800 synchronizes over 20 command buffers), after a blit encoder that fills `written`
//!   of them whole; the rest the CPU wrote and named with `did_modify_range` before the command
//!   buffer (rows `all`, `mixed`), or nobody wrote (row `none`).
//! - **Synchronizes of two 335,872-byte buffers nobody wrote**, after a 1 MiB fill of a `Private`
//!   buffer (row `small`), against the fill alone (row `fill`).
//!
//! ```sh
//! cargo run --release -p metal2vulkan-runtime --example managed_sync_out
//! ```
//!
//! After each command buffer the CPU checks a sample of every buffer's pages. It prints one line
//! a row, each value a command buffer's (medians of the times, means of the counts over the timed
//! ones):
//!
//! ```text
//! ROW <row> gpu_ms <GPU time> wall_ms <commit to completion> copied_mb <MB copied out>
//!     subs <vkQueueSubmit calls> infos <VkSubmitInfos> vkcbs <VkCommandBuffers begun>
//!     (encoders, continuations, batch work, batch barriers, internal) fillq <fill-queue
//!     submissions> commit_us <commit's own time> topped_up <managed_synchronizes_topped_up> <device>
//! ```
//!
//! `topped_up` reads `-` where the device's counts have no such field.

use metal2vulkan_runtime::{
    Buffer, CommandBufferStatus, CommandQueue, CpuCacheMode, Device, HazardTrackingMode,
    HostMemory, ResourceOptions, ResourceRef, StorageMode, WorkCounts,
};
use std::ptr::NonNull;
use std::time::{Duration, Instant};

/// The example's result: every runtime refusal is a typed `metal2vulkan_runtime::Error`.
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// The longest the example waits for a command buffer.
const WAIT: Duration = Duration::from_secs(60);

/// The bytes of each of the 40 buffers: whole pages of any host.
const LARGE: u64 = 1074 * 4096;
/// The bytes of each of the two small buffers.
const SMALL: u64 = 82 * 4096;
/// The bytes of the small rows' fill.
const FILL: u64 = 1 << 20;
/// The command buffers run before the timed ones.
const WARM: usize = 3;
/// The first bytes of every buffer's pages.
const FIRST: u8 = 0x11;

/// Anonymous private pages of the example's own, unmapped when dropped, after the buffer over them.
struct Pages(NonNull<u8>, u64);

impl Pages {
    fn new(bytes: u64) -> Result<Pages> {
        #[cfg(unix)]
        // SAFETY: a new anonymous mapping, which nothing else uses.
        let pointer = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                bytes as usize,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        #[cfg(windows)]
        // SAFETY: a new allocation of its own, at an address the system picks.
        let pointer = unsafe {
            windows_sys::Win32::System::Memory::VirtualAlloc(
                std::ptr::null(),
                bytes as usize,
                windows_sys::Win32::System::Memory::MEM_RESERVE
                    | windows_sys::Win32::System::Memory::MEM_COMMIT,
                windows_sys::Win32::System::Memory::PAGE_READWRITE,
            )
        };
        #[cfg(unix)]
        if pointer == libc::MAP_FAILED {
            return Err("mmap failed".into());
        }
        let pointer = NonNull::new(pointer.cast::<u8>()).ok_or("mmap gave null")?;
        let pages = Pages(pointer, bytes);
        pages.write(FIRST);
        Ok(pages)
    }

    /// Write `value` over every byte, as the CPU.
    fn write(&self, value: u8) {
        // SAFETY: the mapping's bytes, which no command buffer uses while the CPU writes them.
        unsafe { std::ptr::write_bytes(self.0.as_ptr(), value, self.1 as usize) };
    }

    /// The byte at `at`, as the CPU reads it.
    fn byte(&self, at: u64) -> u8 {
        // SAFETY: inside the mapping, which no command buffer writes while the CPU reads it.
        unsafe { self.0.as_ptr().add(at as usize).read_volatile() }
    }

    /// Check a sample of its bytes hold `want`.
    fn check(&self, want: u8, what: &str) -> Result<()> {
        for at in (0..self.1).step_by(65_537).chain([self.1 - 1]) {
            let byte = self.byte(at);
            if byte != want {
                return Err(format!("{what}: byte {at} is {byte:#x}, not {want:#x}").into());
            }
        }
        Ok(())
    }
}

impl Drop for Pages {
    fn drop(&mut self) {
        // SAFETY: the mapping `new` made; every buffer over it went before it.
        #[cfg(unix)]
        unsafe {
            libc::munmap(self.0.as_ptr().cast(), self.1 as usize)
        };
        #[cfg(windows)]
        // SAFETY: as above; a release frees the allocation whole, from its base.
        unsafe {
            windows_sys::Win32::System::Memory::VirtualFree(
                self.0.as_ptr().cast(),
                0,
                windows_sys::Win32::System::Memory::MEM_RELEASE,
            )
        };
    }
}

/// `count` buffers of `bytes` each, each over pages of its own.
struct Buffers {
    buffers: Vec<Buffer>,
    // After the buffers: dropped once they went.
    pages: Vec<Pages>,
}

impl Buffers {
    fn new(device: &Device, count: usize, bytes: u64) -> Result<Buffers> {
        let options = ResourceOptions::new(
            StorageMode::Managed,
            CpuCacheMode::DefaultCache,
            HazardTrackingMode::Default,
        );
        let mut buffers = Vec::with_capacity(count);
        let mut pages = Vec::with_capacity(count);
        for _ in 0..count {
            let host = Pages::new(bytes)?;
            // SAFETY: the pages outlive the buffer (dropped after it, `Buffers`), and only its
            // commands and this example's CPU accesses use them.
            let buffer = unsafe {
                device.new_buffer_with_bytes_no_copy(HostMemory::new(host.0), bytes, options, None)
            }?;
            buffers.push(buffer);
            pages.push(host);
        }
        Ok(Buffers { buffers, pages })
    }
}

/// One row's command buffers: what each does.
#[derive(Clone, Copy)]
enum Shape {
    /// 40 large buffers: the GPU fills `written` of them, the CPU writes the rest where
    /// `modified`, and a blit synchronizes all 40.
    Large { written: usize, modified: bool },
    /// The 1 MiB fill, then a blit synchronizing the two small buffers where `synchronized`.
    Small { synchronized: bool },
}

/// One command buffer of `shape`, of round `round`: its GPU time and its wall time from the commit
/// to its completion, in seconds.
fn run(
    queue: &CommandQueue,
    shape: Shape,
    buffers: &Buffers,
    target: &Buffer,
    round: u8,
) -> Result<(f64, f64)> {
    let command_buffer = queue.command_buffer()?;
    match shape {
        Shape::Large { written, modified } => {
            if modified {
                for (buffer, pages) in buffers.buffers.iter().zip(&buffers.pages).skip(written) {
                    pages.write(round);
                    buffer.did_modify_range(0..LARGE)?;
                }
            }
            if written > 0 {
                let mut fill = command_buffer.blit_command_encoder()?;
                for buffer in &buffers.buffers[..written] {
                    fill.fill_buffer(buffer, 0..LARGE, round)?;
                }
                fill.end_encoding()?;
            }
            let mut sync = command_buffer.blit_command_encoder()?;
            for buffer in &buffers.buffers {
                sync.synchronize_resource(ResourceRef::Buffer(buffer))?;
            }
            sync.end_encoding()?;
        }
        Shape::Small { synchronized } => {
            let mut fill = command_buffer.blit_command_encoder()?;
            fill.fill_buffer(target, 0..FILL, round)?;
            fill.end_encoding()?;
            if synchronized {
                let mut sync = command_buffer.blit_command_encoder()?;
                for buffer in &buffers.buffers {
                    sync.synchronize_resource(ResourceRef::Buffer(buffer))?;
                }
                sync.end_encoding()?;
            }
        }
    }
    let began = Instant::now();
    command_buffer.commit()?;
    command_buffer
        .wait_until_completed_for(WAIT)
        .map_err(|_| format!("a command buffer did not complete within {WAIT:?}"))?;
    let wall = began.elapsed().as_secs_f64();
    if command_buffer.status() != CommandBufferStatus::Completed {
        let error = command_buffer.error();
        return Err(format!(
            "a command buffer ended {:?}: {error:?}",
            command_buffer.status()
        )
        .into());
    }
    let gpu = command_buffer.gpu_end_time()? - command_buffer.gpu_start_time()?;
    match shape {
        Shape::Large { written, modified } => {
            for (index, pages) in buffers.pages.iter().enumerate() {
                let want = match index < written || modified {
                    true => round,
                    false => FIRST,
                };
                pages.check(want, &format!("round {round}, buffer {index}"))?;
            }
        }
        Shape::Small { .. } => {
            for (index, pages) in buffers.pages.iter().enumerate() {
                pages.check(FIRST, &format!("round {round}, small buffer {index}"))?;
            }
        }
    }
    Ok((gpu, wall))
}

/// A field of `counts` by name, read from their `Debug` form: `None` where they have none.
fn field(counts: &WorkCounts, name: &str) -> Option<u64> {
    let text = format!("{counts:?}");
    let at = text.find(&format!(" {name}: "))? + name.len() + 3;
    let digits: String = text[at..]
        .chars()
        .take_while(char::is_ascii_digit)
        .collect();
    digits.parse().ok()
}

/// The median of `values`, sorted here.
fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn main() -> Result<()> {
    let device = Device::system_default()?;
    let unified = device.properties().has_unified_memory;
    println!("device: {} (unified memory: {unified})", device.name());
    let queue = device.new_command_queue()?;
    let private = ResourceOptions::new(
        StorageMode::Private,
        CpuCacheMode::DefaultCache,
        HazardTrackingMode::Default,
    );
    let target = device.new_buffer(FILL, private)?;
    let rows = [
        (
            "all",
            Shape::Large {
                written: 40,
                modified: false,
            },
            20,
        ),
        (
            "mixed",
            Shape::Large {
                written: 30,
                modified: true,
            },
            20,
        ),
        (
            "none",
            Shape::Large {
                written: 0,
                modified: false,
            },
            20,
        ),
        ("small", Shape::Small { synchronized: true }, 200),
        (
            "fill",
            Shape::Small {
                synchronized: false,
            },
            200,
        ),
    ];
    for (name, shape, timed) in rows {
        let buffers = match shape {
            Shape::Large { .. } => Buffers::new(&device, 40, LARGE)?,
            Shape::Small { .. } => Buffers::new(&device, 2, SMALL)?,
        };
        let mut round = 0u8;
        let mut next = || {
            round = round.wrapping_add(1).max(FIRST + 1);
            round
        };
        for _ in 0..WARM {
            run(&queue, shape, &buffers, &target, next())?;
        }
        let before = device.work_counts();
        let mut gpu = Vec::with_capacity(timed);
        let mut wall = Vec::with_capacity(timed);
        for _ in 0..timed {
            let (seconds, elapsed) = run(&queue, shape, &buffers, &target, next())?;
            gpu.push(seconds * 1e3);
            wall.push(elapsed * 1e3);
        }
        let after = device.work_counts();
        let each =
            |pick: fn(&WorkCounts) -> u64| (pick(&after) - pick(&before)) as f64 / timed as f64;
        let topped_up = match (
            field(&after, "managed_synchronizes_topped_up"),
            field(&before, "managed_synchronizes_topped_up"),
        ) {
            (Some(after), Some(before)) => (after - before).to_string(),
            _ => "-".to_string(),
        };
        println!(
            "ROW {name} gpu_ms {:.3} wall_ms {:.3} copied_mb {:.2} subs {:.2} infos {:.2} \
             vkcbs {:.2} ({:.2}, {:.2}, {:.2}, {:.2}, {:.2}) fillq {:.2} commit_us {:.1} \
             topped_up {topped_up} {}",
            median(&mut gpu),
            median(&mut wall),
            each(|counts| counts.managed_bytes_copied_out) / 1e6,
            each(|counts| counts.queue_submissions),
            each(|counts| counts.submit_infos),
            each(|counts| counts.command_buffers_begun),
            each(|counts| counts.command_buffers_begun_encoders),
            each(|counts| counts.command_buffers_begun_continuations),
            each(|counts| counts.command_buffers_begun_batch_work),
            each(|counts| counts.command_buffers_begun_batch_barriers),
            each(|counts| counts.command_buffers_begun_internal),
            each(|counts| counts.fill_queue_submissions),
            each(|counts| counts.commit_nanos) / 1e3,
            device.name()
        );
    }
    Ok(())
}
