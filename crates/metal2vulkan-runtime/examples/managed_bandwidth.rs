//! Not a consumer's example: the GPU's bandwidth through buffers of each storage mode and placement,
//! a kernel reading one 256 MiB buffer and writing another of the same kind, in GB/s read plus
//! written. A `Managed` buffer over the caller's pages (`new_buffer_with_bytes_no_copy`) is the
//! subject: on a device without unified memory Metal keeps it as two copies, the caller's pages and
//! the GPU's own, so the kernel runs at the device memory's speed; a `Shared` one over the caller's
//! pages is the control, read across the bus where the device has no unified memory, as on a Mac
//! with a discrete GPU.
//!
//! ```sh
//! cargo run --release -p metal2vulkan-runtime --example managed_bandwidth
//! ```
//!
//! Each row checks the kernel's answer (`output[i] = 2 * input[i]` for `input[i] = i`), reading a
//! `Managed` output after a blit's `synchronize_resource`; the `Managed` no-copy rows also write the
//! input again on the CPU, tell the buffer (`did_modify_range`) and check the next command buffer
//! reads it. It prints one line a row:
//!
//! ```text
//! BANDWIDTH <row> <GB/s, the median of the timed command buffers> <min> <max> <the device>
//! ```
//!
//! The kernel is `scale_and_offset` of the owned fixture (`common::library`), with a scale of 1:
//! 4 bytes read and 4 written a thread.

mod common;

use metal2vulkan_runtime::{
    Buffer, CommandQueue, ComputePipelineState, CpuCacheMode, Device, HazardTrackingMode,
    HostMemory, PipelineOptions, ResourceOptions, ResourceRef, Size, StorageMode,
};
use std::ptr::NonNull;

/// The bytes of each buffer.
const BYTES: u64 = 256 << 20;
/// The floats each buffer holds.
const COUNT: u64 = BYTES / 4;
/// The command buffers run before the timed ones.
const WARM: usize = 2;
/// The command buffers timed.
const TIMED: usize = 10;

/// How a row makes its two buffers.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Made {
    /// `newBufferWithBytesNoCopy:` over pages of the example's own.
    NoCopy(StorageMode),
    /// `newBufferWithLength:`.
    Own(StorageMode),
}

impl Made {
    fn name(self) -> String {
        match self {
            Made::NoCopy(storage) => format!("{storage:?}-no-copy"),
            Made::Own(storage) => format!("{storage:?}"),
        }
    }

    fn storage(self) -> StorageMode {
        match self {
            Made::NoCopy(storage) | Made::Own(storage) => storage,
        }
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

/// The floats at `pointer`, `COUNT` of them.
///
/// # Safety
///
/// `pointer` holds `BYTES` bytes that nothing else reads or writes meanwhile.
unsafe fn floats<'a>(pointer: NonNull<u8>) -> &'a mut [f32] {
    // SAFETY: the caller's contract; mmap and Metal's buffers are page-aligned.
    unsafe { std::slice::from_raw_parts_mut(pointer.as_ptr().cast(), COUNT as usize) }
}

/// A row's two buffers, the pages under no-copy ones, and how they were made.
struct Row {
    made: Made,
    input: Buffer,
    output: Buffer,
    // After the buffers: dropped once they went.
    _pages: Vec<Pages>,
}

impl Row {
    fn new(device: &Device, made: Made) -> common::Result<Row> {
        let options = ResourceOptions::new(
            made.storage(),
            CpuCacheMode::DefaultCache,
            HazardTrackingMode::Default,
        );
        let mut pages = Vec::new();
        let mut make = |first: bool| -> common::Result<Buffer> {
            match made {
                Made::NoCopy(_) => {
                    let host = Pages::new()?;
                    if first {
                        // The input's bytes, written before the buffer is made: its first GPU use
                        // reads them.
                        // SAFETY: the pages are new and the example's alone.
                        let values = unsafe { floats(host.0) };
                        for (i, value) in values.iter_mut().enumerate() {
                            *value = i as f32;
                        }
                    }
                    // SAFETY: the pages outlive the buffer (dropped after it, `Row`), and only its
                    // commands and this example's CPU reads and writes use them.
                    let buffer = unsafe {
                        device.new_buffer_with_bytes_no_copy(
                            HostMemory::new(host.0),
                            BYTES,
                            options,
                            None,
                        )
                    }?;
                    pages.push(host);
                    Ok(buffer)
                }
                Made::Own(storage) => {
                    let buffer = device.new_buffer(BYTES, options)?;
                    if first && storage != StorageMode::Private {
                        let contents = buffer.contents().ok_or("no contents")?;
                        // SAFETY: the buffer's bytes, which no command uses yet.
                        let values = unsafe { floats(contents) };
                        for (i, value) in values.iter_mut().enumerate() {
                            *value = i as f32;
                        }
                        buffer.did_modify_range(0..BYTES)?;
                    }
                    Ok(buffer)
                }
            }
        };
        let input = make(true)?;
        let output = make(false)?;
        Ok(Row {
            made,
            input,
            output,
            _pages: pages,
        })
    }
}

/// One command buffer of the kernel over `row`'s buffers, synchronizing a `Managed` output when
/// `synchronize`: its GPU time in seconds.
fn run(
    queue: &CommandQueue,
    state: &ComputePipelineState,
    row: &Row,
    synchronize: bool,
) -> common::Result<f64> {
    let command_buffer = queue.command_buffer()?;
    let mut encoder = command_buffer.compute_command_encoder()?;
    encoder.set_compute_pipeline_state(state)?;
    encoder.set_buffer(Some(&row.input), 0, 0)?;
    encoder.set_buffer(Some(&row.output), 0, 1)?;
    encoder.set_bytes(&1.0f32.to_le_bytes(), 2)?;
    encoder.dispatch_threads(Size::new(COUNT, 1, 1), Size::new(256, 1, 1))?;
    encoder.end_encoding()?;
    if synchronize && row.made.storage() == StorageMode::Managed {
        let mut blit = command_buffer.blit_command_encoder()?;
        blit.synchronize_resource(ResourceRef::Buffer(&row.output))?;
        blit.end_encoding()?;
    }
    common::run(&command_buffer)?;
    Ok(command_buffer.gpu_end_time()? - command_buffer.gpu_start_time()?)
}

/// Check `row`'s output holds `2 * (i + shift)` at a sample of its floats; nothing for a `Private`
/// output, which the CPU cannot read.
fn check(row: &Row, shift: f32) -> common::Result<()> {
    let Some(contents) = row.output.contents() else {
        return Ok(());
    };
    // SAFETY: the output's bytes, which no command buffer writes now.
    let values = unsafe { floats(contents) };
    for i in (0..COUNT as usize)
        .step_by(4099)
        .chain([COUNT as usize - 1])
    {
        let want = 2.0 * (i as f32 + shift);
        if values[i] != want {
            return Err(format!(
                "{}: output[{i}] is {}, not {want}",
                row.made.name(),
                values[i]
            )
            .into());
        }
    }
    Ok(())
}

fn main() -> common::Result<()> {
    let device = Device::system_default()?;
    let unified = device.properties().has_unified_memory;
    println!("device: {} (unified memory: {unified})", device.name());
    let library = common::library(&device)?;
    let function = library.new_function_with_name("scale_and_offset")?;
    let state =
        device.new_compute_pipeline_state_with_function(&function, &PipelineOptions::default())?;
    let queue = device.new_command_queue()?;
    let rows = [
        Made::NoCopy(StorageMode::Shared),
        Made::NoCopy(StorageMode::Managed),
        Made::Own(StorageMode::Shared),
        Made::Own(StorageMode::Managed),
        Made::Own(StorageMode::Private),
    ];
    for made in rows {
        let row = Row::new(&device, made)?;
        // The first command buffer reads the input's first contents.
        run(&queue, &state, &row, true)?;
        check(&row, 0.0)?;
        if made == Made::NoCopy(StorageMode::Managed) {
            // A CPU write the buffer is told of reaches the next command buffer.
            let contents = row.input.contents().ok_or("no contents")?;
            // SAFETY: the input's bytes, which no command buffer uses now.
            let values = unsafe { floats(contents) };
            for value in values.iter_mut() {
                *value += 1.0;
            }
            row.input.did_modify_range(0..BYTES)?;
            run(&queue, &state, &row, true)?;
            check(&row, 1.0)?;
        }
        for _ in 0..WARM {
            run(&queue, &state, &row, false)?;
        }
        let mut rates = Vec::with_capacity(TIMED);
        for _ in 0..TIMED {
            let seconds = run(&queue, &state, &row, false)?;
            rates.push(2.0 * BYTES as f64 / seconds / 1e9);
        }
        rates.sort_by(f64::total_cmp);
        println!(
            "BANDWIDTH {} {:.1} {:.1} {:.1} {}",
            made.name(),
            rates[TIMED / 2],
            rates[0],
            rates[TIMED - 1],
            device.name()
        );
    }
    Ok(())
}
