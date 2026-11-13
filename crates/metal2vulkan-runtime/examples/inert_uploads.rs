//! Not a consumer's example: which command buffer copies a `Managed` buffer's CPU writes in, on a
//! device without unified memory, when the first command buffer to bind the buffer only
//! dispatches a kernel that does nothing (an empty kernel with buffer arguments it never reads).
//! A program uses such a command buffer to upload ahead of timed work. The example also measures
//! what the next command buffer, the one that reads the buffer, costs. Three rows, each a shape
//! of a guest workload's:
//!
//! - **`batch`**: 40 `Managed` no-copy buffers of 8 MiB, which the CPU writes and names with
//!   `did_modify_range`. Command buffer `A` dispatches the empty kernel once for each buffer, and
//!   command buffer `B` dispatches a kernel that reads 4 KiB of each.
//! - **`frames`**: 33 such buffers of 614,400 bytes, written and named the same way. `A` binds
//!   each, then 11 frames each read three of them, then fill 1 MiB of a `Private` buffer and
//!   synchronize two 335,872-byte `Managed` buffers that nobody wrote.
//! - **`inert`**: command buffers of 42 dispatches of the empty kernel over `Private` buffers alone.
//!
//! ```sh
//! cargo run --release -p metal2vulkan-runtime --example inert_uploads
//! ```
//!
//! The kernels are LLVM text written here, assembled by `llvm-as` from `PATH`. The CPU checks
//! every word `B` and the frames read. Each line is the median over the timed command buffers of
//! the row's kind (`A`, `B`, `frame`, `inert`), and each count is a mean per command buffer:
//!
//! ```text
//! ROW <row> <kind> gpu_ms <GPU time> wall_ms <commit to completion> encode_us <encoding, per
//!     dispatch> copied_in_mb <MB copied in> fillq <fill-queue submissions> fillq_waits <batches
//!     that waited for one> vkcbs <VkCommandBuffers begun> subs <vkQueueSubmit calls> skipped
//!     <dispatches_skipped_inert> <device>
//! ```
//!
//! `skipped` reads `-` where the device's counts have no such field.

use metal2vulkan_runtime::{
    Buffer, CommandBuffer, CommandBufferStatus, CommandQueue, ComputePipelineState, CpuCacheMode,
    Device, HazardTrackingMode, HostMemory, PipelineOptions, ResourceOptions, ResourceRef, Size,
    StorageMode, WorkCounts,
};
use std::process::Command;
use std::ptr::NonNull;
use std::time::{Duration, Instant};

/// The example's result: every runtime refusal is a typed `metal2vulkan_runtime::Error`.
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// The longest the example waits for a command buffer.
const WAIT: Duration = Duration::from_secs(60);

/// Two kernels of buffers `a` (index 0) and `b` (index 1) and the thread's grid position:
/// `empty` does nothing; `twice` stores `2 * a[gid]` at `b[gid]`.
const KERNELS: &str = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\n\
target triple = \"air64_v28-apple-macosx26.0.0\"\n\n\
define void @empty(ptr addrspace(1) noundef %a, ptr addrspace(1) noundef %b, i32 noundef %gid) {\n\
entry:\n\
  ret void\n\
}\n\n\
define void @twice(ptr addrspace(1) noundef %a, ptr addrspace(1) noundef %b, i32 noundef %gid) {\n\
entry:\n\
  %gw = zext i32 %gid to i64\n\
  %q = getelementptr inbounds i32, ptr addrspace(1) %a, i64 %gw\n\
  %x = load i32, ptr addrspace(1) %q, align 4\n\
  %v = shl i32 %x, 1\n\
  %p = getelementptr inbounds i32, ptr addrspace(1) %b, i64 %gw\n\
  store i32 %v, ptr addrspace(1) %p, align 4\n\
  ret void\n\
}\n\n\
!llvm.module.flags = !{!0, !1}\n!air.kernel = !{!20, !21}\n!air.version = !{!8}\n!air.language_version = !{!9}\n\
!0 = !{i32 7, !\"air.max_device_buffers\", i32 31}\n\
!1 = !{i32 7, !\"air.max_threadgroup_buffers\", i32 31}\n\
!3 = !{}\n!4 = !{!5, !6, !7}\n\
!5 = !{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"a\"}\n\
!6 = !{i32 1, !\"air.buffer\", !\"air.location_index\", i32 1, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"b\"}\n\
!7 = !{i32 2, !\"air.thread_position_in_grid\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"gid\"}\n\
!8 = !{i32 2, i32 8, i32 0}\n!9 = !{!\"Metal\", i32 4, i32 0, i32 0}\n\
!20 = !{ptr @empty, !3, !4}\n!21 = !{ptr @twice, !3, !4}\n";

/// The words `twice` reads of each buffer it reads: 4 KiB.
const WORDS: u64 = 1024;
/// The bytes of the small buffers the frames synchronize.
const SMALL: u64 = 82 * 4096;
/// The bytes of the frames' fill.
const FILL: u64 = 1 << 20;

/// Anonymous private pages of the example's own, unmapped when dropped, after the buffer over them.
struct Pages(NonNull<u8>, u64);

impl Pages {
    fn new(bytes: u64) -> Result<Pages> {
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
        if pointer == libc::MAP_FAILED {
            return Err("mmap failed".into());
        }
        let pointer = NonNull::new(pointer.cast::<u8>()).ok_or("mmap gave null")?;
        Ok(Pages(pointer, bytes))
    }

    /// Write `value` over every byte, as the CPU.
    fn write(&self, value: u8) {
        // SAFETY: the mapping's bytes, which no command buffer uses while the CPU writes them.
        unsafe { std::ptr::write_bytes(self.0.as_ptr(), value, self.1 as usize) };
    }
}

impl Drop for Pages {
    fn drop(&mut self) {
        // SAFETY: the mapping `new` made; every buffer over it went before it.
        unsafe { libc::munmap(self.0.as_ptr().cast(), self.1 as usize) };
    }
}

/// `count` `Managed` no-copy buffers of `bytes` each, each over pages of its own.
struct Managed {
    buffers: Vec<Buffer>,
    // After the buffers: dropped once they went.
    pages: Vec<Pages>,
}

impl Managed {
    fn new(device: &Device, count: usize, bytes: u64) -> Result<Managed> {
        let options = ResourceOptions::new(
            StorageMode::Managed,
            CpuCacheMode::DefaultCache,
            HazardTrackingMode::Default,
        );
        let mut buffers = Vec::with_capacity(count);
        let mut pages = Vec::with_capacity(count);
        for _ in 0..count {
            let host = Pages::new(bytes)?;
            // SAFETY: the pages outlive the buffer (dropped after it, `Managed`), and only its
            // commands and this example's CPU accesses use them.
            let buffer = unsafe {
                device.new_buffer_with_bytes_no_copy(HostMemory::new(host.0), bytes, options, None)
            }?;
            buffers.push(buffer);
            pages.push(host);
        }
        Ok(Managed { buffers, pages })
    }

    /// The CPU writes `value` over every buffer and names it whole with `did_modify_range`.
    fn write(&self, value: u8) -> Result<()> {
        for (buffer, pages) in self.buffers.iter().zip(&self.pages) {
            pages.write(value);
            buffer.did_modify_range(0..pages.1)?;
        }
        Ok(())
    }
}

/// The state of kernel `name` of [`KERNELS`], assembled by `llvm-as`.
fn state(device: &Device, name: &str) -> Result<ComputePipelineState> {
    let mut child = Command::new("llvm-as")
        .args(["-o", "-", "-"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|error| format!("cannot run llvm-as (put LLVM's bin on PATH): {error}"))?;
    {
        use std::io::Write;
        let mut stdin = child.stdin.take().ok_or("llvm-as has no stdin")?;
        stdin.write_all(KERNELS.as_bytes())?;
    }
    let output = child.wait_with_output()?;
    if !output.status.success() {
        return Err(format!("llvm-as: {}", String::from_utf8_lossy(&output.stderr)).into());
    }
    let library = metal2vulkan::library::Library::from_air(&output.stdout)?;
    let function = device
        .new_library_with_compile_library(library)
        .new_function_with_name(name)?;
    Ok(device.new_compute_pipeline_state_with_function(&function, &PipelineOptions::default())?)
}

/// Commit `command_buffer` and wait for it: its GPU time and its wall time from the commit to its
/// completion, in milliseconds.
fn run(command_buffer: &CommandBuffer) -> Result<(f64, f64)> {
    let began = Instant::now();
    command_buffer.commit()?;
    command_buffer
        .wait_until_completed_for(WAIT)
        .map_err(|_| format!("a command buffer did not complete within {WAIT:?}"))?;
    let wall = began.elapsed().as_secs_f64() * 1e3;
    if command_buffer.status() != CommandBufferStatus::Completed {
        let error = command_buffer.error();
        return Err(format!(
            "a command buffer ended {:?}: {error:?}",
            command_buffer.status()
        )
        .into());
    }
    let gpu = (command_buffer.gpu_end_time()? - command_buffer.gpu_start_time()?) * 1e3;
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

/// One kind of command buffer's samples and counts, over the command buffers it ran.
#[derive(Default)]
struct Kind {
    gpu: Vec<f64>,
    wall: Vec<f64>,
    encode: Vec<f64>,
    copied_in: u64,
    fillq: u64,
    fillq_waits: u64,
    vkcbs: u64,
    subs: u64,
    skipped: Option<u64>,
}

impl Kind {
    /// Add one command buffer: its times, its encoding's time per dispatch in microseconds, and
    /// the device's counts before its encoding and after its completion.
    fn add(
        &mut self,
        (gpu, wall): (f64, f64),
        encode: f64,
        before: &WorkCounts,
        after: &WorkCounts,
    ) {
        self.gpu.push(gpu);
        self.wall.push(wall);
        self.encode.push(encode);
        self.copied_in += after.managed_bytes_copied_in - before.managed_bytes_copied_in;
        self.fillq += after.fill_queue_submissions - before.fill_queue_submissions;
        self.fillq_waits += after.fill_queue_waits - before.fill_queue_waits;
        self.vkcbs += after.command_buffers_begun - before.command_buffers_begun;
        self.subs += after.queue_submissions - before.queue_submissions;
        let skipped = |counts| field(counts, "dispatches_skipped_inert");
        if let (Some(after), Some(before)) = (skipped(after), skipped(before)) {
            *self.skipped.get_or_insert(0) += after - before;
        }
    }

    fn print(&mut self, row: &str, kind: &str, device: &str) {
        let n = self.gpu.len() as f64;
        let median = |values: &mut Vec<f64>| {
            values.sort_by(f64::total_cmp);
            values[values.len() / 2]
        };
        let skipped = match self.skipped {
            Some(skipped) => format!("{:.2}", skipped as f64 / n),
            None => "-".to_string(),
        };
        println!(
            "ROW {row} {kind} gpu_ms {:.3} wall_ms {:.3} encode_us {:.2} copied_in_mb {:.2} \
             fillq {:.2} fillq_waits {:.2} vkcbs {:.2} subs {:.2} skipped {skipped} {device}",
            median(&mut self.gpu),
            median(&mut self.wall),
            median(&mut self.encode),
            self.copied_in as f64 / n / 1e6,
            self.fillq as f64 / n,
            self.fillq_waits as f64 / n,
            self.vkcbs as f64 / n,
            self.subs as f64 / n,
        );
    }
}

/// The kernels and the queue every row uses.
struct Rig {
    device: Device,
    queue: CommandQueue,
    empty: ComputePipelineState,
    twice: ComputePipelineState,
}

impl Rig {
    /// A command buffer that dispatches `empty` once for each of `buffers` (bound at index 0 and
    /// 1), one thread each: its encoding's time per dispatch, in microseconds.
    fn binds(&self, buffers: &[Buffer]) -> Result<(CommandBuffer, f64)> {
        let command_buffer = self.queue.command_buffer()?;
        let began = Instant::now();
        let mut encoder = command_buffer.compute_command_encoder()?;
        encoder.set_compute_pipeline_state(&self.empty)?;
        for buffer in buffers {
            encoder.set_buffer(Some(buffer), 0, 0)?;
            encoder.set_buffer(Some(buffer), 0, 1)?;
            encoder.dispatch_threadgroups(Size::new(1, 1, 1), Size::new(1, 1, 1))?;
        }
        encoder.end_encoding()?;
        let encode = began.elapsed().as_secs_f64() * 1e6 / buffers.len() as f64;
        Ok((command_buffer, encode))
    }

    /// Encode, into `command_buffer`, `twice` over the first [`WORDS`] words of each of
    /// `buffers`, into `out` at 4 KiB apart.
    fn reads(
        &self,
        command_buffer: &CommandBuffer,
        buffers: &[Buffer],
        out: &Buffer,
    ) -> Result<()> {
        let mut encoder = command_buffer.compute_command_encoder()?;
        encoder.set_compute_pipeline_state(&self.twice)?;
        for (n, buffer) in buffers.iter().enumerate() {
            encoder.set_buffer(Some(buffer), 0, 0)?;
            encoder.set_buffer(Some(out), n as u64 * WORDS * 4, 1)?;
            encoder.dispatch_threads(Size::new(WORDS, 1, 1), Size::new(64, 1, 1))?;
        }
        encoder.end_encoding()?;
        Ok(())
    }
}

/// Check that `out`'s first `count` blocks of [`WORDS`] words hold twice the word of bytes
/// `value`.
fn check(out: &Buffer, count: usize, value: u8, what: &str) -> Result<()> {
    let contents = out.contents().ok_or("the output has no contents")?;
    let want = (u32::from_ne_bytes([value; 4])).wrapping_shl(1);
    for at in 0..count * WORDS as usize {
        // SAFETY: `out` is `Shared`, at least `count` blocks long, and no command buffer
        // writes it while the CPU reads it.
        let word = unsafe { contents.as_ptr().cast::<u32>().add(at).read_volatile() };
        if word != want {
            return Err(format!("{what}: word {at} is {word:#x}, not {want:#x}").into());
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let device = Device::system_default()?;
    let unified = device.properties().has_unified_memory;
    let name = device.name().to_string();
    println!("device: {name} (unified memory: {unified})");
    let rig = Rig {
        queue: device.new_command_queue()?,
        empty: state(&device, "empty")?,
        twice: state(&device, "twice")?,
        device,
    };
    let device = &rig.device;
    let private = ResourceOptions::new(
        StorageMode::Private,
        CpuCacheMode::DefaultCache,
        HazardTrackingMode::Default,
    );
    let out = device.new_buffer(40 * WORDS * 4, ResourceOptions::shared())?;
    let mut round = 0x20u8;
    let mut next = || {
        round = round.wrapping_add(1).max(0x21);
        round
    };

    // `inert`: the empty kernel's own cost, over buffers with nothing to copy in.
    let plain: Vec<Buffer> = (0..42)
        .map(|_| device.new_buffer(64 * 1024, private))
        .collect::<std::result::Result<_, _>>()?;
    let mut inert = Kind::default();
    for timed in [false; 3].into_iter().chain([true; 20]) {
        let before = device.work_counts();
        let (command_buffer, encode) = rig.binds(&plain)?;
        let times = run(&command_buffer)?;
        if timed {
            inert.add(times, encode, &before, &device.work_counts());
        }
    }
    inert.print("inert", "inert", &name);

    // `frames`: 33 buffers uploaded by `A`, read three a frame.
    let uploads = Managed::new(device, 33, 150 * 4096)?;
    let small = Managed::new(device, 2, SMALL)?;
    let fill = device.new_buffer(FILL, private)?;
    let (mut a, mut frame) = (Kind::default(), Kind::default());
    for timed in [false, true, true, true, true] {
        let value = next();
        uploads.write(value)?;
        let before = device.work_counts();
        let (command_buffer, encode) = rig.binds(&uploads.buffers)?;
        let times = run(&command_buffer)?;
        if timed {
            a.add(times, encode, &before, &device.work_counts());
        }
        for three in uploads.buffers.chunks(3) {
            let before = device.work_counts();
            let command_buffer = rig.queue.command_buffer()?;
            let began = Instant::now();
            rig.reads(&command_buffer, three, &out)?;
            let mut blit = command_buffer.blit_command_encoder()?;
            blit.fill_buffer(&fill, 0..FILL, value)?;
            for buffer in &small.buffers {
                blit.synchronize_resource(ResourceRef::Buffer(buffer))?;
            }
            blit.end_encoding()?;
            let encode = began.elapsed().as_secs_f64() * 1e6 / three.len() as f64;
            let times = run(&command_buffer)?;
            check(&out, three.len(), value, "a frame")?;
            if timed {
                frame.add(times, encode, &before, &device.work_counts());
            }
        }
    }
    a.print("frames", "A", &name);
    frame.print("frames", "frame", &name);

    // `batch`: 40 buffers of 8 MiB uploaded by `A`, read by `B`.
    let uploads = Managed::new(device, 40, 8 << 20)?;
    let (mut a, mut b) = (Kind::default(), Kind::default());
    for timed in [false, false, true, true, true, true, true, true] {
        let value = next();
        uploads.write(value)?;
        let before = device.work_counts();
        let (command_buffer, encode) = rig.binds(&uploads.buffers)?;
        let times = run(&command_buffer)?;
        if timed {
            a.add(times, encode, &before, &device.work_counts());
        }
        let before = device.work_counts();
        let command_buffer = rig.queue.command_buffer()?;
        let began = Instant::now();
        rig.reads(&command_buffer, &uploads.buffers, &out)?;
        let encode = began.elapsed().as_secs_f64() * 1e6 / uploads.buffers.len() as f64;
        let times = run(&command_buffer)?;
        check(&out, uploads.buffers.len(), value, "B")?;
        if timed {
            b.add(times, encode, &before, &device.work_counts());
        }
    }
    a.print("batch", "A", &name);
    b.print("batch", "B", &name);
    Ok(())
}
