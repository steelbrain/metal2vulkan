//! Not a consumer's example: the GPU's bandwidth through two 256 MiB buffers of the runtime's own
//! memory, made as a program makes them (`new_buffer`, or `new_buffer_with_descriptor` at an
//! address it picks or at a pinned one), each pair after the pairs before it in the same device
//! went, so that a pair may take the memory an earlier one left. A kernel reads one buffer and
//! writes the other, one dispatch a command buffer, committed and waited for, with no CPU access.
//!
//! ```sh
//! METAL2VULKAN_LOG_MEMORY=1 cargo run --release -p metal2vulkan-runtime --example private_bandwidth
//! ```
//!
//! Each sequence runs on a device of its own (the last one's dropped first), each pair made, filled
//! once with zero by a blit, then run [`PASSES`] times; the last [`TIMED`] are timed. It prints one
//! line a pair:
//!
//! ```text
//! BANDWIDTH <sequence> <pair> wall <GB/s> gpu <GB/s> made: <memory counts> <the device>
//! ```
//!
//! GB/s is read plus written over the median command buffer: `wall` from commit to its wait's end,
//! `gpu` from its GPU start and end times. The memory counts are those the two buffers' making
//! moved: `vkAllocateMemory` and `vkBindBufferMemory` calls, live dedicated blocks, live empty
//! blocks, overflows into system memory. With `METAL2VULKAN_LOG_MEMORY=1` the memory log names
//! the block and memory type each buffer took.
//!
//! The kernel is `scale_and_offset` of the owned fixture (`common::library`), with a scale of 1:
//! 4 bytes read and 4 written a thread.

mod common;

use metal2vulkan_runtime::{
    AddressPlacement, Buffer, BufferContents, BufferDescriptor, CommandQueue, ComputePipelineState,
    Device, GpuAddress, PipelineOptions, ResourceOptions, Size, StorageMode, WorkCounts,
};
use std::time::Instant;

/// The bytes of each buffer.
const BYTES: u64 = 256 << 20;
/// The floats each buffer holds.
const COUNT: u64 = BYTES / 4;
/// The command buffers a pair runs.
const PASSES: usize = 10;
/// The last of them timed.
const TIMED: usize = 8;
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
}

/// A pair: its buffers' storage mode and how they are made.
#[derive(Clone, Copy, Debug)]
struct Pair(StorageMode, Made);

impl Pair {
    fn name(self) -> String {
        format!("{:?}-{:?}", self.0, self.1)
    }
}

/// The sequences, each on a device of its own: a pair first in a device, and after pairs of each
/// storage mode went.
const SEQUENCES: [(&str, &[Pair]); 4] = [
    (
        "first",
        &[
            Pair(StorageMode::Private, Made::New),
            Pair(StorageMode::Private, Made::Descriptor),
            Pair(StorageMode::Private, Made::Pinned),
        ],
    ),
    (
        "after-managed",
        &[
            Pair(StorageMode::Managed, Made::New),
            Pair(StorageMode::Private, Made::New),
            Pair(StorageMode::Private, Made::Pinned),
        ],
    ),
    (
        "after-shared",
        &[
            Pair(StorageMode::Shared, Made::New),
            Pair(StorageMode::Private, Made::New),
            Pair(StorageMode::Private, Made::Pinned),
        ],
    ),
    (
        "shared-managed-private",
        &[
            Pair(StorageMode::Shared, Made::Pinned),
            Pair(StorageMode::Managed, Made::Pinned),
            Pair(StorageMode::Private, Made::Pinned),
        ],
    ),
];

/// One buffer of `pair`, the `index`th of its two.
fn make(device: &Device, pair: Pair, index: u64) -> common::Result<Buffer> {
    let options = ResourceOptions::new(pair.0, Default::default(), Default::default());
    let buffer = match pair.1 {
        Made::New => device.new_buffer(BYTES, options)?,
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

/// One command buffer of the kernel from `input` to `output`: its wall time from commit to its
/// wait's end and its GPU time, in seconds.
fn pass(
    queue: &CommandQueue,
    state: &ComputePipelineState,
    input: &Buffer,
    output: &Buffer,
) -> common::Result<(f64, f64)> {
    let command_buffer = queue.command_buffer()?;
    let mut encoder = command_buffer.compute_command_encoder()?;
    encoder.set_compute_pipeline_state(state)?;
    encoder.set_buffer(Some(input), 0, 0)?;
    encoder.set_buffer(Some(output), 0, 1)?;
    encoder.set_bytes(&1.0f32.to_le_bytes(), 2)?;
    encoder.dispatch_threads(Size::new(COUNT, 1, 1), Size::new(256, 1, 1))?;
    encoder.end_encoding()?;
    let start = Instant::now();
    common::run(&command_buffer)?;
    let wall = start.elapsed().as_secs_f64();
    let gpu = command_buffer.gpu_end_time()? - command_buffer.gpu_start_time()?;
    Ok((wall, gpu))
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
    let before = device.work_counts();
    let input = make(device, pair, 0)?;
    let output = make(device, pair, 1)?;
    let made = moved(&before, &device.work_counts());
    let command_buffer = queue.command_buffer()?;
    let mut blit = command_buffer.blit_command_encoder()?;
    blit.fill_buffer(&input, 0..BYTES, 0)?;
    blit.fill_buffer(&output, 0..BYTES, 0)?;
    blit.end_encoding()?;
    common::run(&command_buffer)?;
    let (mut walls, mut gpus) = (Vec::new(), Vec::new());
    for index in 0..PASSES {
        let (wall, gpu) = pass(queue, state, &input, &output)?;
        if index >= PASSES - TIMED {
            walls.push(wall);
            gpus.push(gpu);
        }
    }
    check(device, queue, &output, &pair)?;
    println!(
        "BANDWIDTH {sequence} {} wall {:.1} gpu {:.1} made: {made} {}",
        pair.name(),
        rate(walls),
        rate(gpus),
        device.name()
    );
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
            run_pair(&device, (&queue, &state), sequence, pair)?;
        }
    }
    Ok(())
}
