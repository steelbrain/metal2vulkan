//! Not a consumer's example: what `synchronize_resource` of `Managed` buffers over the caller's
//! pages costs a command buffer on a device without unified memory, where each such buffer is two
//! copies (the caller's pages and the GPU's own) and a synchronize copies the GPU's copy out to the
//! pages. The shape a guest workload showed: 40 buffers of 4,399,104 bytes each, all synchronized
//! by each command buffer (about 176 MB), of which an earlier encoder of the same command buffer
//! wrote some.
//!
//! ```sh
//! cargo run --release -p metal2vulkan-runtime --example managed_sync_out
//! ```
//!
//! Each command buffer has two blit encoders: the first fills `written` of the buffers whole
//! (with a value of its round), the second synchronizes all 40. After each, the CPU checks a
//! sample of every buffer's pages: a written buffer's hold its round's value, the rest their
//! first bytes. It prints one line a row:
//!
//! ```text
//! SYNC-OUT <written>/40 <GPU ms, median> <min> <max> <wall ms, median> <MB copied out a command buffer> <the device>
//! ```
//!
//! The wall time runs from the commit to the completion. The bytes copied out are the device's
//! count (`WorkCounts::managed_bytes_copied_out`) over the timed command buffers.

use metal2vulkan_runtime::{
    Buffer, CommandBufferStatus, CommandQueue, CpuCacheMode, Device, HazardTrackingMode,
    HostMemory, ResourceOptions, ResourceRef, StorageMode,
};
use std::ptr::NonNull;
use std::time::{Duration, Instant};

/// The example's result: every runtime refusal is a typed `metal2vulkan_runtime::Error`.
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// The longest the example waits for a command buffer.
const WAIT: Duration = Duration::from_secs(60);

/// The buffers each command buffer synchronizes.
const BUFFERS: usize = 40;
/// The bytes of each buffer: whole pages of any host.
const BYTES: u64 = 1074 * 4096;
/// The command buffers run before the timed ones.
const WARM: usize = 3;
/// The command buffers timed.
const TIMED: usize = 20;
/// The first bytes of every buffer's pages.
const FIRST: u8 = 0x11;

/// Anonymous private pages of the example's own, unmapped when dropped, after the buffer over them.
struct Pages(NonNull<u8>);

impl Pages {
    fn new() -> Result<Pages> {
        // SAFETY: a new anonymous mapping, which nothing else uses.
        let pointer = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                BYTES as usize,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if pointer == libc::MAP_FAILED {
            return Err("mmap failed".into());
        }
        let pointer = NonNull::new(pointer.cast::<u8>()).ok_or("mmap gave null")?;
        // SAFETY: the new mapping's bytes, the example's alone.
        unsafe { std::ptr::write_bytes(pointer.as_ptr(), FIRST, BYTES as usize) };
        Ok(Pages(pointer))
    }

    /// The byte at `at`, as the CPU reads it.
    fn byte(&self, at: u64) -> u8 {
        // SAFETY: inside the mapping, which no command buffer writes while the CPU reads it.
        unsafe { self.0.as_ptr().add(at as usize).read_volatile() }
    }
}

impl Drop for Pages {
    fn drop(&mut self) {
        // SAFETY: the mapping `new` made; every buffer over it went before it.
        unsafe { libc::munmap(self.0.as_ptr().cast(), BYTES as usize) };
    }
}

/// The buffers, each over pages of its own.
struct Buffers {
    buffers: Vec<Buffer>,
    // After the buffers: dropped once they went.
    pages: Vec<Pages>,
}

impl Buffers {
    fn new(device: &Device) -> Result<Buffers> {
        let options = ResourceOptions::new(
            StorageMode::Managed,
            CpuCacheMode::DefaultCache,
            HazardTrackingMode::Default,
        );
        let mut buffers = Vec::with_capacity(BUFFERS);
        let mut pages = Vec::with_capacity(BUFFERS);
        for _ in 0..BUFFERS {
            let host = Pages::new()?;
            // SAFETY: the pages outlive the buffer (dropped after it, `Buffers`), and only its
            // commands and this example's CPU reads use them.
            let buffer = unsafe {
                device.new_buffer_with_bytes_no_copy(HostMemory::new(host.0), BYTES, options, None)
            }?;
            buffers.push(buffer);
            pages.push(host);
        }
        Ok(Buffers { buffers, pages })
    }

    /// Check a sample of each buffer's pages after round `round` of `written` buffers written.
    fn check(&self, written: usize, round: u8) -> Result<()> {
        for (index, pages) in self.pages.iter().enumerate() {
            let want = match index < written {
                true => round,
                false => FIRST,
            };
            for at in (0..BYTES).step_by(65_537).chain([BYTES - 1]) {
                let byte = pages.byte(at);
                if byte != want {
                    return Err(format!(
                        "round {round}, {written} written: buffer {index} byte {at} is {byte:#x}, \
                         not {want:#x}"
                    )
                    .into());
                }
            }
        }
        Ok(())
    }
}

/// One command buffer: `written` of the buffers filled with `round`, then all synchronized. Its GPU
/// time and its wall time from the commit, in seconds.
fn run(queue: &CommandQueue, buffers: &Buffers, written: usize, round: u8) -> Result<(f64, f64)> {
    let command_buffer = queue.command_buffer()?;
    if written > 0 {
        let mut fill = command_buffer.blit_command_encoder()?;
        for buffer in &buffers.buffers[..written] {
            fill.fill_buffer(buffer, 0..BYTES, round)?;
        }
        fill.end_encoding()?;
    }
    let mut sync = command_buffer.blit_command_encoder()?;
    for buffer in &buffers.buffers {
        sync.synchronize_resource(ResourceRef::Buffer(buffer))?;
    }
    sync.end_encoding()?;
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
    Ok((gpu, wall))
}

fn main() -> Result<()> {
    let device = Device::system_default()?;
    let unified = device.properties().has_unified_memory;
    println!("device: {} (unified memory: {unified})", device.name());
    let queue = device.new_command_queue()?;
    for written in [BUFFERS, 4, 0] {
        let buffers = Buffers::new(&device)?;
        let mut round = 0u8;
        let mut next = || {
            round = round.wrapping_add(1).max(FIRST + 1);
            round
        };
        for _ in 0..WARM {
            let round = next();
            run(&queue, &buffers, written, round)?;
            buffers.check(written, round)?;
        }
        let copied = device.work_counts().managed_bytes_copied_out;
        let mut gpu = Vec::with_capacity(TIMED);
        let mut wall = Vec::with_capacity(TIMED);
        for _ in 0..TIMED {
            let round = next();
            let (seconds, elapsed) = run(&queue, &buffers, written, round)?;
            buffers.check(written, round)?;
            gpu.push(seconds * 1e3);
            wall.push(elapsed * 1e3);
        }
        let copied = (device.work_counts().managed_bytes_copied_out - copied) / TIMED as u64;
        gpu.sort_by(f64::total_cmp);
        wall.sort_by(f64::total_cmp);
        println!(
            "SYNC-OUT {written}/{BUFFERS} {:.3} {:.3} {:.3} {:.3} {:.1} {}",
            gpu[TIMED / 2],
            gpu[0],
            gpu[TIMED - 1],
            wall[TIMED / 2],
            copied as f64 / 1e6,
            device.name()
        );
    }
    Ok(())
}
