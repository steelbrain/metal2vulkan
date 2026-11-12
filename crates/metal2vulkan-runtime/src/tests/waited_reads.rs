//! Reads of host pages by a command buffer committed before the CPU writes them, whose first
//! command waits for an event the host signals once the write is made. Metal runs the command
//! buffer's commands after the wait, so they read the written bytes (`shadow.rs`, **Copy in**: a
//! command buffer committed before a CPU write read the written bytes).
//!
//! Three homes of the pages: a plain import, an import whose memory came with a log of the writes
//! to them (`HostMemory::with_dirty_pages`, `dirty_pages.rs`), and a copy of them where the
//! import gate refused the import (`copied.rs`). Over each, a buffer and a 64x64 `RGBA8Unorm`
//! texture over its first bytes, read and written through (a shadow on every device; over a copy,
//! a shadow or an image over the copy). Each round the CPU writes one pattern and a command buffer
//! blits the buffer's bytes and the texture's texels out (so the shadow and the copy hold it);
//! then the waiting command buffer, the same blits after a wait for the event, is committed, the
//! CPU writes a second pattern (told to the log), and the host signals: both blits must read the
//! second.
//!
//! Over a copy, each waiting command buffer copies in again as its batch is submitted after the
//! wait (`command_buffer/copies_again.rs`, `WorkCounts::copied_copy_ins_after_waits`): one a
//! round. The same rounds over a copy with no wait, the second pattern written before the second
//! command buffer's commit, copy in again never: only at each commit.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use super::blit_holds::{device, Host, Scratch, BYTES, WAIT};
use super::dirty_pages_device::Log;
use crate::{
    BlitOption, Buffer, CommandBuffer, CommandBufferStatus, CommandQueue, Device, Error, EventRef,
    Origin, ResourceOptions, Size, Texture, TextureDescriptor, WorkCounts,
};
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use std::sync::Arc;

/// The texture's side, its rows' pitch, and the bytes each read takes.
const SIDE: u64 = 64;
const ROW: u64 = 256;
const SPAN: u64 = SIDE * ROW;
const ROUNDS: u64 = 3;

/// Where the pages lie for the device.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Home {
    /// An import of them.
    Import,
    /// An import of them, their memory logged ([`Log`]).
    Logged,
    /// A copy of them, the gate having refused their import.
    Copied,
}

/// The buffer over `host` at `home`, logged by `log` where it says; `None`, printed, where the
/// device makes no buffer over host memory.
fn over(device: &Device, host: &Host, home: Home, log: &Arc<Log>) -> Option<Buffer> {
    // SAFETY: the memory outlives the buffer (dropped after the device), and only the buffer's
    // commands and the test use it while it lives.
    let memory = unsafe { host.memory() };
    let memory = match home {
        Home::Import => memory,
        Home::Logged => memory.with_dirty_pages(Arc::clone(log)),
        Home::Copied => memory.with_import_gate(Arc::new(|_| false)),
    };
    // SAFETY: as above.
    let made = unsafe {
        device.new_buffer_with_bytes_no_copy(memory, BYTES as u64, ResourceOptions::default(), None)
    };
    match made {
        Ok(buffer) => Some(buffer),
        Err(Error::Unsupported(gap)) => {
            eprintln!("WAITED-READS: no buffer over host memory: {gap:?}");
            None
        }
        Err(error) => panic!("new_buffer_with_bytes_no_copy: {error:?}"),
    }
}

/// A command buffer of `queue`'s that, after a wait for `wait`'s event to reach its value where
/// one is given, blits the buffer's first [`SPAN`] bytes into `out` and the texture's texels after
/// them; committed.
fn reads(
    queue: &CommandQueue,
    (buffer, texture): (&Buffer, &Texture),
    out: &Buffer,
    wait: Option<(EventRef<'_>, u64)>,
) -> CommandBuffer {
    let command_buffer = queue.command_buffer().unwrap();
    if let Some((event, value)) = wait {
        (command_buffer.encode_wait_for_event(event, value)).unwrap();
    }
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    (blit.copy_from_buffer_to_buffer(buffer, 0, out, 0, SPAN)).unwrap();
    blit.copy_from_texture_to_buffer(
        texture,
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(SIDE, SIDE, 1),
        out,
        SPAN,
        ROW,
        SPAN,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer
}

/// The first byte of `out`'s two reads that is not `value`: which read, its offset and its byte.
fn first_wrong(out: &Buffer, value: u8) -> Option<(&'static str, usize, u8)> {
    let contents = out.contents().unwrap();
    // SAFETY: the buffer's first `2 * SPAN` bytes, which no command buffer writes now.
    let bytes = unsafe { std::slice::from_raw_parts(contents.as_ptr(), 2 * SPAN as usize) };
    let at = bytes.iter().position(|&byte| byte != value)?;
    let read = match at < SPAN as usize {
        true => "the buffer",
        false => "the texture",
    };
    Some((read, at % SPAN as usize, bytes[at]))
}

/// What the rounds moved: (copied copies in again after waits, copied bytes copied in, logged
/// copies in gated, logged copies in whole).
fn moved(before: &WorkCounts, after: &WorkCounts) -> [u64; 4] {
    [
        after.copied_copy_ins_after_waits - before.copied_copy_ins_after_waits,
        after.copied_copy_in_bytes - before.copied_copy_in_bytes,
        after.dirty_copies_in_gated - before.dirty_copies_in_gated,
        after.dirty_copies_in_whole - before.dirty_copies_in_whole,
    ]
}

/// The rounds over pages at `home`, the second read waiting for the host's signal where `waits`.
fn run(case: &str, home: Home, waits: bool) {
    // Declared before the device, so dropped after it.
    let host = Host::new();
    let scratch = Scratch::new(&format!("waited-reads-{case}"));
    let device = device(&scratch.0);
    let log = Arc::new(Log::default());
    let Some(buffer) = over(&device, &host, home, &log) else {
        return;
    };
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
    descriptor.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
    let texture = buffer.new_texture(&descriptor, 0, ROW).unwrap();
    let out = (device.new_buffer(2 * SPAN, ResourceOptions::default())).unwrap();
    let queue = device.new_command_queue().unwrap();
    let event = device.new_shared_event().unwrap();
    let write = |value: u8| {
        host.fill(u32::from_ne_bytes([value; 4]));
        log.told(0..BYTES as u64, 0);
    };
    let finish = |command_buffer: CommandBuffer| {
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
    };
    let mut failures = Vec::new();
    let before = device.work_counts();
    for round in 0..ROUNDS {
        let (first, second) = (0x11 + round as u8, 0x81 + round as u8);
        write(first);
        finish(reads(&queue, (&buffer, &texture), &out, None));
        if let Some(wrong) = first_wrong(&out, first) {
            failures.push(format!("round {round}: the read before: {wrong:x?}"));
        }
        let read = match waits {
            true => {
                let value = round + 1;
                let wait = Some((EventRef::Shared(&event), value));
                let read = reads(&queue, (&buffer, &texture), &out, wait);
                // It waits for the host's signal, which comes once the write is made.
                assert_ne!(read.status(), CommandBufferStatus::Completed);
                write(second);
                event.set_signaled_value(value);
                read
            }
            false => {
                write(second);
                reads(&queue, (&buffer, &texture), &out, None)
            }
        };
        finish(read);
        if let Some(wrong) = first_wrong(&out, second) {
            failures.push(format!("round {round}: the read after: {wrong:x?}"));
        }
    }
    let moved = moved(&before, &device.work_counts());
    eprintln!(
        "WAITED-READS {case} on {}: {} wrong reads of {}; (copied copies in again after waits, \
         copied bytes copied in, logged copies in gated, whole) {moved:?} over {} commits; first \
         {:?}",
        device.name(),
        failures.len(),
        2 * ROUNDS,
        2 * ROUNDS,
        failures.first()
    );
    assert!(failures.is_empty(), "{case}: {failures:?}");
    let again = match (home, waits) {
        (Home::Copied, true) => ROUNDS,
        _ => 0,
    };
    assert_eq!(moved[0], again, "{case}: copies in again after waits");
}

#[test]
fn an_imported_read_after_a_wait_takes_the_bytes_written_before_the_signal() {
    run("import", Home::Import, true);
}

#[test]
fn a_logged_read_after_a_wait_takes_the_bytes_written_before_the_signal() {
    run("logged", Home::Logged, true);
}

#[test]
fn a_copied_read_after_a_wait_takes_the_bytes_written_before_the_signal() {
    run("copied", Home::Copied, true);
}

#[test]
fn a_copied_read_with_no_wait_takes_the_bytes_written_before_its_commit() {
    run("copied-no-wait", Home::Copied, false);
}
