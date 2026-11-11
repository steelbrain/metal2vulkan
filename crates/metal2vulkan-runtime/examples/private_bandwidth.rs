//! Not a consumer's example: the GPU's bandwidth through two 256 MiB buffers, made as a program
//! makes them (`new_buffer`, `new_buffer_with_descriptor` at an address it picks or at a pinned
//! one, or `new_buffer_with_bytes_no_copy` over pages of the example's own, or over the bytes of a
//! `Shared` no-copy buffer that is a copy of them), each pair after the
//! pairs before it in the same device went, so that a pair may take the memory an earlier one
//! left. A kernel reads one buffer and writes the other, one dispatch a command buffer, committed
//! and waited for, with no CPU access.
//!
//! ```sh
//! METAL2VULKAN_LOG_MEMORY=1 cargo run --release -p metal2vulkan-runtime --example private_bandwidth
//! ```
//!
//! Each sequence runs on a device of its own (the last one's dropped first), each pair made after
//! the device idled for [`PAUSE`], filled once with zero by a blit, then run [`PASSES`] times in
//! each of [`ROUNDS`] rounds back to back; the last [`TIMED`] of a round are timed. It prints one
//! line a round:
//!
//! ```text
//! BANDWIDTH <sequence> <pair> round <n> wall <GB/s> gpu <GB/s> made: <memory counts> <the device>
//! ```
//!
//! GB/s is read plus written over the median command buffer: `wall` from commit to its wait's end,
//! `gpu` from its GPU start and end times. A `PASSES` line after it gives each command buffer's
//! commit time since the Unix epoch and its two times, in milliseconds (`at:wall/gpu`), and
//! `COUNTS` lines every `WorkCounts` field the first timed pass and the last one moved. The memory
//! counts are those the two buffers' making moved: `vkAllocateMemory` and `vkBindBufferMemory`
//! calls, live dedicated blocks, live empty blocks, overflows into system memory; then the first
//! buffer's `host_backing`. With `METAL2VULKAN_LOG_MEMORY=1` the memory log names the block and
//! memory type each buffer took.
//!
//! The kernel is `scale_and_offset` of the owned fixture (`common::library`), with a scale of 1:
//! 4 bytes read and 4 written a thread.

mod common;

use metal2vulkan_runtime::{
    AddressPlacement, Buffer, BufferContents, BufferDescriptor, CommandQueue, ComputePipelineState,
    Device, GpuAddress, HostMemory, PipelineOptions, ResourceOptions, Size, StorageMode,
    WorkCounts,
};
use std::ptr::NonNull;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// The bytes of each buffer.
const BYTES: u64 = 256 << 20;
/// The floats each buffer holds.
const COUNT: u64 = BYTES / 4;
/// The command buffers a pair runs.
const PASSES: usize = 10;
/// The last of them timed.
const TIMED: usize = 8;
/// The rounds of [`PASSES`] a pair runs, back to back.
const ROUNDS: usize = 2;
/// The device's idle time before each pair is made: as long as a guest's stages wait.
const PAUSE: Duration = Duration::from_secs(3);
/// The first pinned address, and the distance between a pair's two pins: past every range the
/// runtime picks for two 256 MiB buffers at the space's start.
const PIN_BASE: u64 = 1 << 40;
const PIN_STEP: u64 = 1 << 30;

/// How a pair's buffers are made.
#[derive(Clone, Copy, Debug)]
enum Made {
    /// `newBufferWithLength:options:`.
    New,
    /// `newBufferWithDescriptor:` at an address the runtime picks.
    Descriptor,
    /// `newBufferWithDescriptor:` at a pinned address.
    Pinned,
    /// `newBufferWithBytesNoCopy:length:options:deallocator:` over pages of the example's own.
    NoCopy,
    /// `newBufferWithLength:options:`, with a `Shared` no-copy buffer over pages of the example's
    /// own beside each, which the kernel never uses and which goes with it: as a guest's `Managed`
    /// buffer is made with its CPU copy over the guest's pages.
    Beside,
    /// `newBufferWithBytesNoCopy:length:options:deallocator:` over exactly the bytes of a `Shared`
    /// no-copy buffer beside it, which the kernel never uses, whose pages' import the caller's
    /// gate refused, so that it is a copy of them (`HostMemory::with_import_gate`, which refuses
    /// the pair's buffer's import too): as a host that refuses a second import of pages makes a
    /// guest's buffers.
    OverCopy,
}

/// The storage mode's bits of a resource's options.
const STORAGE_MASK: u64 = 0xf0;

/// The resource option bit no Metal header names that programs pass with their buffers' options
/// (`ResourceOptions::BIT_20`), as a guest's calls carry it.
const BIT_20: u64 = ResourceOptions::BIT_20;

/// A pair: its buffers' storage mode, how they are made, and the option bits passed beside the
/// storage mode.
#[derive(Clone, Copy, Debug)]
struct Pair(StorageMode, Made, u64);

impl Pair {
    fn name(self) -> String {
        format!("{:?}-{:?}-{:#x}", self.0, self.1, self.options().raw())
    }

    fn options(self) -> ResourceOptions {
        let modes = ResourceOptions::new(self.0, Default::default(), Default::default());
        ResourceOptions::from_raw(modes.raw() | self.2).expect("options a buffer takes")
    }
}

/// Anonymous private pages of the example's own, unmapped when dropped, after the buffer over them.
struct Pages(NonNull<u8>);

impl Pages {
    fn new() -> common::Result<Pages> {
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
        Ok(Pages(NonNull::new(pointer.cast()).ok_or("mmap gave null")?))
    }
}

impl Drop for Pages {
    fn drop(&mut self) {
        // SAFETY: the mapping `new` made; every buffer over it went before it.
        unsafe { libc::munmap(self.0.as_ptr().cast(), BYTES as usize) };
    }
}

/// The sequences, each on a device of its own: a pair first in a device, and after pairs of each
/// storage mode went.
const SEQUENCES: [(&str, &[Pair]); 7] = [
    (
        "first",
        &[
            Pair(StorageMode::Private, Made::New, 0),
            Pair(StorageMode::Private, Made::Descriptor, 0),
            Pair(StorageMode::Private, Made::Pinned, 0),
        ],
    ),
    (
        "after-managed",
        &[
            Pair(StorageMode::Managed, Made::New, 0),
            Pair(StorageMode::Private, Made::New, 0),
            Pair(StorageMode::Private, Made::Pinned, 0),
        ],
    ),
    (
        "after-shared",
        &[
            Pair(StorageMode::Shared, Made::New, 0),
            Pair(StorageMode::Private, Made::New, 0),
            Pair(StorageMode::Private, Made::Pinned, 0),
        ],
    ),
    (
        "shared-managed-private",
        &[
            Pair(StorageMode::Shared, Made::Pinned, 0),
            Pair(StorageMode::Managed, Made::Pinned, 0),
            Pair(StorageMode::Private, Made::Pinned, 0),
        ],
    ),
    // As a guest's buffers come: `Shared` and `Managed` ones over its pages with bit 20, as its
    // no-copy calls pass it, then `Private` ones at its addresses, of its plain options.
    (
        "no-copy-then-private",
        &[
            Pair(StorageMode::Shared, Made::NoCopy, BIT_20),
            Pair(StorageMode::Managed, Made::NoCopy, BIT_20),
            Pair(StorageMode::Private, Made::Pinned, 0),
            Pair(StorageMode::Private, Made::Pinned, BIT_20),
        ],
    ),
    // As a guest's Metal buffers come: `Shared` ones over its pages, `Managed` ones with their CPU
    // copies over its pages beside them, then `Private` ones at its addresses.
    (
        "guest-order",
        &[
            Pair(StorageMode::Shared, Made::NoCopy, 0),
            Pair(StorageMode::Managed, Made::Beside, 0),
            Pair(StorageMode::Private, Made::Pinned, 0),
        ],
    ),
    // No-copy buffers over the bytes of `Shared` ones that are copies of them, and over pages no
    // other buffer is made over.
    (
        "over-copy",
        &[
            Pair(StorageMode::Managed, Made::OverCopy, 0),
            Pair(StorageMode::Managed, Made::NoCopy, 0),
            Pair(StorageMode::Shared, Made::OverCopy, 0),
        ],
    ),
];

/// A `Shared` no-copy buffer of `options`' other modes over new pages of the example's own, which
/// it pushes to `pages`.
fn no_copy(
    device: &Device,
    options: ResourceOptions,
    pages: &mut Vec<Pages>,
) -> common::Result<Buffer> {
    let host = Pages::new()?;
    // SAFETY: the pages outlive the buffer (`run_pair` drops them after it), and only its
    // commands use them.
    let buffer = unsafe {
        device.new_buffer_with_bytes_no_copy(HostMemory::new(host.0), BYTES, options, None)
    }?;
    pages.push(host);
    Ok(buffer)
}

/// A no-copy buffer of `options` over the bytes of `host`, whose import a gate refuses.
///
/// # Safety
///
/// The pages outlive the buffer.
unsafe fn refused(
    device: &Device,
    options: ResourceOptions,
    host: &Pages,
) -> common::Result<Buffer> {
    // SAFETY: the caller's contract, and only the example's commands use the pages.
    let memory = unsafe { HostMemory::new(host.0) }.with_import_gate(Arc::new(|_| false));
    // SAFETY: as above.
    Ok(unsafe { device.new_buffer_with_bytes_no_copy(memory, BYTES, options, None) }?)
}

/// One buffer of `pair`, the `index`th of its two, over pages it pushes to `pages` where it is a
/// no-copy one, and the buffers beside it pushed to `beside`.
fn make(
    device: &Device,
    pair: Pair,
    index: u64,
    (pages, beside): (&mut Vec<Pages>, &mut Vec<Buffer>),
) -> common::Result<Buffer> {
    let options = pair.options();
    let buffer = match pair.1 {
        Made::NoCopy => no_copy(device, options, pages)?,
        Made::New => device.new_buffer(BYTES, options)?,
        Made::Beside => {
            let buffer = device.new_buffer(BYTES, options)?;
            let shared = ResourceOptions::from_raw(
                (options.raw() & !STORAGE_MASK) | ResourceOptions::shared().raw(),
            )?;
            beside.push(no_copy(device, shared, pages)?);
            buffer
        }
        Made::OverCopy => {
            let host = Pages::new()?;
            // SAFETY: the pages outlive both buffers (`run_pair` drops them after them).
            let copy = unsafe { refused(device, ResourceOptions::shared(), &host) }?;
            // SAFETY: as above.
            let buffer = unsafe { refused(device, options, &host) }?;
            beside.push(copy);
            pages.push(host);
            buffer
        }
        Made::Descriptor => device.new_buffer_with_descriptor(BufferDescriptor::new(
            BYTES,
            options,
            BufferContents::Zero,
        ))?,
        Made::Pinned => {
            let at = GpuAddress(PIN_BASE + index * PIN_STEP);
            let descriptor = BufferDescriptor::new(BYTES, options, BufferContents::Zero)
                .with_placement(AddressPlacement::At(at));
            device.new_buffer_with_descriptor(descriptor)?
        }
    };
    Ok(buffer)
}

/// The memory counts the making moved, from `before` to `after` (the module docs).
fn moved(before: &WorkCounts, after: &WorkCounts) -> String {
    let (b, a) = (&before.object_churn, &after.object_churn);
    let delta = |after: u64, before: u64| after as i64 - before as i64;
    format!(
        "allocations {} bindings {} dedicated {:+} empty {:+} overflows {}",
        delta(a.vk_memory_allocations, b.vk_memory_allocations),
        delta(a.vk_buffer_bindings, b.vk_buffer_bindings),
        delta(
            after.dedicated_memory_blocks_live,
            before.dedicated_memory_blocks_live
        ),
        delta(
            after.empty_memory_blocks_live,
            before.empty_memory_blocks_live
        ),
        delta(
            after.device_memory_overflows,
            before.device_memory_overflows
        ),
    )
}

/// Every `WorkCounts` field that moved from `before` to `after`, as `after` names it: the lines
/// of the two counts' debug forms that differ.
fn changed(before: &WorkCounts, after: &WorkCounts) -> String {
    let (before, after) = (format!("{before:#?}"), format!("{after:#?}"));
    let lines: Vec<&str> = (before.lines().zip(after.lines()))
        .filter(|(was, now)| was != now)
        .map(|(_, now)| now.trim().trim_end_matches(','))
        .collect();
    lines.join(" ")
}

/// One command buffer of the kernel from `input` to `output`: its commit's time since the Unix
/// epoch, in milliseconds, and its wall time from commit to its wait's end and its GPU time, in
/// seconds.
fn pass(
    queue: &CommandQueue,
    state: &ComputePipelineState,
    input: &Buffer,
    output: &Buffer,
) -> common::Result<(u128, f64, f64)> {
    let command_buffer = queue.command_buffer()?;
    let mut encoder = command_buffer.compute_command_encoder()?;
    encoder.set_compute_pipeline_state(state)?;
    encoder.set_buffer(Some(input), 0, 0)?;
    encoder.set_buffer(Some(output), 0, 1)?;
    encoder.set_bytes(&1.0f32.to_le_bytes(), 2)?;
    encoder.dispatch_threads(Size::new(COUNT, 1, 1), Size::new(256, 1, 1))?;
    encoder.end_encoding()?;
    let at = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis();
    let start = Instant::now();
    common::run(&command_buffer)?;
    let wall = start.elapsed().as_secs_f64();
    let gpu = command_buffer.gpu_end_time()? - command_buffer.gpu_start_time()?;
    Ok((at, wall, gpu))
}

/// The floats at the start of an output the check reads.
const CHECKED: u64 = 1024;

/// Check that `output` holds the kernel's answer for an input of zeros, `output[i] = i`, in its
/// first [`CHECKED`] floats, copied by a blit into a `Shared` buffer the host reads.
fn check(
    device: &Device,
    queue: &CommandQueue,
    output: &Buffer,
    pair: &Pair,
) -> common::Result<()> {
    let read = device.new_buffer(CHECKED * 4, ResourceOptions::shared())?;
    let command_buffer = queue.command_buffer()?;
    let mut blit = command_buffer.blit_command_encoder()?;
    blit.copy_from_buffer_to_buffer(output, 0, &read, 0, CHECKED * 4)?;
    blit.end_encoding()?;
    common::run(&command_buffer)?;
    let bytes = common::bytes(&read)?;
    for (i, value) in bytes.as_chunks::<4>().0.iter().enumerate() {
        let value = f32::from_le_bytes(*value);
        if value != i as f32 {
            return Err(format!("{}: output[{i}] is {value}, not {i}", pair.name()).into());
        }
    }
    Ok(())
}

/// The median of `seconds` as GB/s, read plus written.
fn rate(mut seconds: Vec<f64>) -> f64 {
    seconds.sort_by(f64::total_cmp);
    2.0 * BYTES as f64 / seconds[seconds.len() / 2] / 1e9
}

/// Run `pair` on `device`: made, filled, then timed.
fn run_pair(
    device: &Device,
    (queue, state): (&CommandQueue, &ComputePipelineState),
    sequence: &str,
    pair: Pair,
) -> common::Result<()> {
    // Before the buffers: dropped after them.
    let (mut pages, mut beside) = (Vec::new(), Vec::new());
    let before = device.work_counts();
    let input = make(device, pair, 0, (&mut pages, &mut beside))?;
    let output = make(device, pair, 1, (&mut pages, &mut beside))?;
    let made = moved(&before, &device.work_counts());
    let made = format!("{made} backing {:?}", input.host_backing());
    let command_buffer = queue.command_buffer()?;
    let mut blit = command_buffer.blit_command_encoder()?;
    blit.fill_buffer(&input, 0..BYTES, 0)?;
    blit.fill_buffer(&output, 0..BYTES, 0)?;
    blit.end_encoding()?;
    common::run(&command_buffer)?;
    for round in 0..ROUNDS {
        let (mut walls, mut gpus, mut passes) = (Vec::new(), Vec::new(), Vec::new());
        let mut counts = Vec::new();
        for index in 0..PASSES {
            let before = device.work_counts();
            let (at, wall, gpu) = pass(queue, state, &input, &output)?;
            passes.push(format!("{at}:{:.2}/{:.2}", wall * 1e3, gpu * 1e3));
            if index >= PASSES - TIMED {
                walls.push(wall);
                gpus.push(gpu);
            }
            if index == PASSES - TIMED || index == PASSES - 1 {
                counts.push((index, changed(&before, &device.work_counts())));
            }
        }
        check(device, queue, &output, &pair)?;
        let name = pair.name();
        for (index, moved) in counts {
            println!("COUNTS {sequence} {name} round {round} pass {index}: {moved}");
        }
        println!(
            "BANDWIDTH {sequence} {name} round {round} wall {:.1} gpu {:.1} made: {made} {}",
            rate(walls),
            rate(gpus),
            device.name()
        );
        println!(
            "PASSES {sequence} {name} round {round} at-ms:wall/gpu-ms: {}",
            passes.join(" ")
        );
    }
    Ok(())
}

fn main() -> common::Result<()> {
    for (sequence, pairs) in SEQUENCES {
        // A device of its own: the last sequence's went with its last handle.
        let device = Device::system_default()?;
        let unified = device.properties().has_unified_memory;
        println!(
            "sequence {sequence}: {} (unified memory: {unified})",
            device.name()
        );
        let library = common::library(&device)?;
        let function = library.new_function_with_name("scale_and_offset")?;
        let state = device
            .new_compute_pipeline_state_with_function(&function, &PipelineOptions::default())?;
        let queue = device.new_command_queue()?;
        for &pair in pairs {
            std::thread::sleep(PAUSE);
            run_pair(&device, (&queue, &state), sequence, pair)?;
        }
    }
    Ok(())
}
