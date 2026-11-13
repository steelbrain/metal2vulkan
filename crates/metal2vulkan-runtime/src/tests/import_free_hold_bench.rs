//! What a `vkFreeMemory` of an import of host pages holds of the device's submissions, and what
//! pacing the frees changes: a thread commits a one-word fill on a queue of its own every
//! [`PING`] and waits for it, while the bench's own thread makes imports and then frees them in
//! a run, through the runtime's two doors (`creation::allocate_memory`, `release::object`). The
//! kinds: [`PIECES`] imports of 512 KiB of anonymous pages by their pointer
//! (`VK_EXT_external_memory_host`, the submission checks' import tax's shape), four of 64 MiB (a
//! buffer in fresh pages, `fresh_pages.rs`), [`PIECES`] dma-bufs of 512 KiB of a sealed `memfd`'s
//! pages (udmabuf, where the device imports dma-bufs and `/dev/udmabuf` opens), each imported as
//! a no-copy buffer's is (`memory::import_dma_buf`: for a buffer made for dma-bufs, dedicated to it
//! where the driver takes no import dedicated to nothing), and, beside them, [`PIECES`] plain
//! host-visible allocations of 512 KiB and four of 64 MiB (what a buffer not in fresh pages
//! allocates). Each kind runs [`ROUNDS`] rounds of each order, alternating: its frees in turn, one
//! right after another, and paced, each after a wait as long as the frees before it took, less the
//! time since the last ended (`step_aside::pace`, the runtime's own threads' rule). Printed (`IMPORT-FREE-HOLD`), per kind: the makes' spread;
//! per kind and order: the frees' spread, the runs' lengths, and the spread of the pings whose
//! commit to completion overlapped a run, beside those that overlapped no make or free.
//!
//! The completion leg (`import_free_hold_bench_completed`): the same pinger beside imports a
//! completed command buffer's release frees (`command_buffer/mod.rs`'s release step). Each round
//! makes [`PIECES`] no-copy buffers of 512 KiB, or four of 64 MiB, over anonymous pages, each
//! with a deallocator that notes when it ran; records a copy of a word out of each into a
//! `Private` buffer, commits, and lets go of the buffers while the command buffer is in flight,
//! so its completion lets go of their pages last and frees their imports. Printed
//! (`IMPORT-FREE-HOLD completed`), per size: the frees' runs (the first deallocator to the last),
//! how long after `wait_until_completed` returned the last deallocator ran, the spread of the
//! pings whose commit to completion overlapped a round's commit to its last deallocator, beside
//! those that overlapped none, and the free thread's counts over the leg.
//!
//! Ignored: run them in release under `--cfg m2v_gpu`, `cargo test --release -p
//! metal2vulkan-runtime --lib import_free_hold_bench -- --ignored --nocapture`.

use super::host_residency::{udmabuf, udmabuf_openable};
use crate::{Buffer, Device, DeviceKind, DeviceOptions, HostMemory, ResourceOptions};
use ash::vk;
use std::ops::Range;
use std::os::fd::{AsFd, AsRawFd, FromRawFd, OwnedFd};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const WAIT: Duration = Duration::from_secs(60);
/// The imports of a kind of 512 KiB, as many as the import tax makes.
const PIECES: usize = 32;
/// The rounds of each order, per kind.
const ROUNDS: usize = 8;
/// How long a round's imports stay made before their frees.
const HOLD: Duration = Duration::from_millis(2);
/// The time between two rounds, so that each one's effect on the pings stands apart.
const GAP: Duration = Duration::from_millis(50);
/// How often the pinger commits.
const PING: Duration = Duration::from_micros(250);

/// Where a kind's memory comes from.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Source {
    /// Anonymous pages, imported by their pointer.
    Pointer,
    /// Dma-bufs of a sealed `memfd`'s pages.
    DmaBuf,
    /// The device's own host-visible memory.
    Plain,
}

/// A kind of memory the bench makes and frees: where it comes from, each one's bytes, and how
/// many a round makes.
#[derive(Clone, Copy)]
struct Kind {
    source: Source,
    bytes: u64,
    count: usize,
}

impl std::fmt::Display for Kind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let source = match self.source {
            Source::Pointer => "pointer imports",
            Source::DmaBuf => "dma-buf imports",
            Source::Plain => "plain allocations",
        };
        write!(f, "{} {source} of {} KiB", self.count, self.bytes >> 10)
    }
}

/// How a round frees.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Order {
    InTurn,
    Paced,
}

impl std::fmt::Display for Order {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Order::InTurn => "in turn",
            Order::Paced => "paced",
        })
    }
}

/// What the bench measured of one kind: each make's span, and per order each free's span and
/// each run's.
struct Spans {
    kind: Kind,
    makes: Vec<Range<Instant>>,
    frees: Vec<(Order, Range<Instant>)>,
    runs: Vec<(Order, Range<Instant>)>,
}

/// Sets its flag as it goes, so the pinger stops however the bench ends.
struct Stops<'a>(&'a AtomicBool);

impl Drop for Stops<'_> {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}

/// A milliseconds summary of `values`: median, 99th percentile and slowest, and how many took
/// 1 ms and 5 ms or more.
fn spread(mut values: Vec<Duration>) -> String {
    if values.is_empty() {
        return "none".to_owned();
    }
    values.sort();
    let ms = |at: usize| values[at.min(values.len() - 1)].as_secs_f64() * 1e3;
    let over = |least: u128| {
        values
            .iter()
            .filter(|value| value.as_millis() >= least)
            .count()
    };
    format!(
        "{} of them: median {:.3} ms, p99 {:.3} ms, slowest {:.3} ms, {} of 1 ms or more, {} of \
         5 ms or more",
        values.len(),
        ms(values.len() / 2),
        ms(values.len() * 99 / 100),
        ms(values.len() - 1),
        over(1),
        over(5)
    )
}

fn overlaps(a: &Range<Instant>, b: &Range<Instant>) -> bool {
    a.start < b.end && b.start < a.end
}

fn lengths<'a>(spans: impl Iterator<Item = &'a Range<Instant>>) -> Vec<Duration> {
    spans.map(|span| span.end - span.start).collect()
}

/// The first of the memory types `bits` allows that is host-visible and coherent.
fn coherent(properties: &vk::PhysicalDeviceMemoryProperties, bits: u32) -> Option<u32> {
    let types = &properties.memory_types[..properties.memory_type_count as usize];
    let wanted = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
    (0..types.len() as u32).find(|&index| {
        bits & (1 << index) != 0 && types[index as usize].property_flags.contains(wanted)
    })
}

/// Anonymous pages of `len` bytes from an address aligned to `align`, written so that each is
/// in place before its import pins it; unmapped as this goes.
struct Pages {
    mapped: NonNull<u8>,
    mapped_len: usize,
    at: NonNull<u8>,
}

impl Pages {
    fn new(len: usize, align: usize) -> Pages {
        let mapped_len = len + align;
        // SAFETY: a new private anonymous mapping.
        let mapped = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                mapped_len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        assert_ne!(mapped, libc::MAP_FAILED, "mmap");
        let mapped = NonNull::new(mapped.cast::<u8>()).unwrap();
        let skip = mapped.as_ptr().align_offset(align);
        // SAFETY: `skip` is under `align`, so the `len` bytes from it lie inside the mapping.
        let at = unsafe { NonNull::new_unchecked(mapped.as_ptr().add(skip)) };
        // SAFETY: inside the mapping, readable and writable.
        unsafe { std::ptr::write_bytes(at.as_ptr(), 0x5a, len) };
        Pages {
            mapped,
            mapped_len,
            at,
        }
    }

    fn at(&self, offset: usize) -> NonNull<u8> {
        // SAFETY: the caller's offsets lie inside the pages.
        unsafe { NonNull::new_unchecked(self.at.as_ptr().add(offset)) }
    }
}

impl Drop for Pages {
    fn drop(&mut self) {
        // SAFETY: the mapping is this object's, and no import over it lives.
        unsafe { libc::munmap(self.mapped.as_ptr().cast(), self.mapped_len) };
    }
}

/// A sealed `memfd` of `len` bytes.
fn memfd(len: usize) -> OwnedFd {
    // SAFETY: a new anonymous file, owned at once, then sized and sealed.
    unsafe {
        let fd = libc::memfd_create(
            c"m2v-import-free-hold".as_ptr(),
            libc::MFD_CLOEXEC | libc::MFD_ALLOW_SEALING,
        );
        assert!(fd >= 0, "memfd_create: {}", std::io::Error::last_os_error());
        let fd = OwnedFd::from_raw_fd(fd);
        assert_eq!(libc::ftruncate(fd.as_raw_fd(), len as libc::off_t), 0);
        let seals = libc::F_SEAL_GROW | libc::F_SEAL_SHRINK | libc::F_SEAL_SEAL;
        assert_eq!(libc::fcntl(fd.as_raw_fd(), libc::F_ADD_SEALS, seals), 0);
        fd
    }
}

/// One the bench made: its memory, and the buffer made for a dma-buf's import, destroyed before
/// the memory is freed.
struct Made {
    memory: vk::DeviceMemory,
    buffer: Option<vk::Buffer>,
}

/// What one round makes its memory of.
enum Backing {
    Pointer(Pages),
    DmaBuf(OwnedFd),
    Plain(u32),
}

/// The device's memory types.
fn memory_properties(device: &Device) -> vk::PhysicalDeviceMemoryProperties {
    // SAFETY: the bench reads the instance's own query of its physical device.
    let handles = unsafe { device.raw() };
    // SAFETY: the physical device is the instance's.
    unsafe {
        handles
            .instance
            .get_physical_device_memory_properties(handles.physical_device)
    }
}

/// Make one of `kind`, the `index`th of its round, from `backing`, of a type `properties` lists:
/// its memory, or why the device refused it.
fn make(
    device: &Device,
    properties: &vk::PhysicalDeviceMemoryProperties,
    (kind, backing, index): (Kind, &Backing, usize),
) -> Result<Made, String> {
    let shared = device.shared();
    let raw = shared.vk();
    let importer = (shared.host_import.as_ref()).map_err(|refusal| refusal.to_string())?;
    let offset = index as u64 * kind.bytes;
    let info = vk::MemoryAllocateInfo::default().allocation_size(kind.bytes);
    match backing {
        Backing::Pointer(pages) => {
            let pointers = importer.pointers().ok_or("no pointer import")?;
            let at = pages.at(offset as usize);
            // SAFETY: the pages are aligned to the import's unit (`Pages::new`).
            let bits = unsafe { pointers.memory_types(raw, at) }.map_err(|e| e.to_string())?;
            let type_index = coherent(properties, bits).ok_or("no coherent type")?;
            let mut import = vk::ImportMemoryHostPointerInfoEXT::default()
                .handle_type(vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT)
                .host_pointer(at.as_ptr().cast());
            let info = info.memory_type_index(type_index).push_next(&mut import);
            // SAFETY: the pages stay mapped until every import of them is freed.
            let memory = unsafe { crate::creation::allocate_memory(raw, &info) };
            let memory = memory.map_err(|e| e.to_string())?;
            Ok(Made {
                memory,
                buffer: None,
            })
        }
        Backing::DmaBuf(memfd) => {
            let fd = udmabuf(memfd, offset, kind.bytes).map_err(|e| e.to_string())?;
            // SAFETY: the device is the importer's.
            let bits =
                unsafe { importer.dma_buf_types(raw, fd.as_fd()) }.map_err(|e| e.to_string())?;
            let mut external = vk::ExternalMemoryBufferCreateInfo::default()
                .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
            let buffer_info = shared.sharing().buffer(
                vk::BufferCreateInfo::default()
                    .size(kind.bytes)
                    .usage(crate::buffer::usage(shared.device_address))
                    .push_next(&mut external),
            );
            // SAFETY: the device is live and the info valid; the buffer goes with the import.
            let buffer = unsafe { crate::creation::buffer(raw, &buffer_info) };
            let buffer = buffer.map_err(|e| e.to_string())?;
            // SAFETY: the buffer is live.
            let requirements = unsafe { raw.get_buffer_memory_requirements(buffer) };
            let allowed = (
                bits & requirements.memory_type_bits,
                crate::memory::MemoryClass::Shared,
            );
            let purpose = crate::memory::DmaBufFor::Buffer(buffer);
            let imported = crate::memory::import_dma_buf(
                &shared.memory,
                raw,
                fd,
                kind.bytes,
                allowed,
                purpose,
            );
            match imported {
                Ok((memory, _, _)) => Ok(Made {
                    memory,
                    buffer: Some(buffer),
                }),
                Err(failure) => {
                    // SAFETY: nothing is bound to the buffer, and no command uses it.
                    unsafe { crate::release::object(raw, buffer) };
                    Err(format!("{failure:?}"))
                }
            }
        }
        Backing::Plain(type_index) => {
            let info = info.memory_type_index(*type_index);
            // SAFETY: a plain allocation.
            let memory = unsafe { crate::creation::allocate_memory(raw, &info) };
            let memory = memory.map_err(|e| e.to_string())?;
            Ok(Made {
                memory,
                buffer: None,
            })
        }
    }
}

/// What a round of `kind` makes its memory of; `None`, said why, where the device or the host
/// has none.
fn backing(device: &Device, kind: Kind) -> Option<Backing> {
    let shared = device.shared();
    let importer = shared.host_import.as_ref().ok();
    let len = kind.bytes as usize * kind.count;
    match kind.source {
        Source::Pointer => {
            let Some(pointers) = importer.and_then(|importer| importer.pointers()) else {
                eprintln!("IMPORT-FREE-HOLD {kind}: the device imports no pointer: not run");
                return None;
            };
            Some(Backing::Pointer(Pages::new(len, pointers.unit() as usize)))
        }
        Source::DmaBuf => {
            if !importer.is_some_and(|importer| importer.imports_dma_bufs()) {
                eprintln!("IMPORT-FREE-HOLD {kind}: the device imports no dma-buf: not run");
                return None;
            }
            udmabuf_openable("import_free_hold_bench").then(|| Backing::DmaBuf(memfd(len)))
        }
        Source::Plain => coherent(&memory_properties(device), u32::MAX).map(Backing::Plain),
    }
}

/// Destroy `one`'s buffer and free its memory: the free's span.
fn free(raw: &ash::Device, one: Made) -> Range<Instant> {
    // SAFETY: nothing uses the buffer or the memory.
    unsafe {
        if let Some(buffer) = one.buffer {
            crate::release::object(raw, buffer);
        }
        let started = Instant::now();
        crate::release::object(raw, one.memory);
        started..Instant::now()
    }
}

/// Run [`ROUNDS`] rounds of each order of `kind` on `device`, [`GAP`] apart; `None` where it
/// cannot be made.
fn measure(device: &Device, kind: Kind) -> Option<Spans> {
    let backing = backing(device, kind)?;
    let properties = memory_properties(device);
    let raw = device.shared().vk();
    let mut spans = Spans {
        kind,
        makes: Vec::new(),
        frees: Vec::new(),
        runs: Vec::new(),
    };
    for round in 0..2 * ROUNDS {
        let order = if round % 2 == 0 {
            Order::InTurn
        } else {
            Order::Paced
        };
        let mut made = Vec::new();
        for index in 0..kind.count {
            let started = Instant::now();
            match make(device, &properties, (kind, &backing, index)) {
                Ok(one) => made.push(one),
                Err(why) => {
                    eprintln!("IMPORT-FREE-HOLD {kind}: refused ({why}): not run");
                    for one in made {
                        free(raw, one);
                    }
                    return None;
                }
            }
            spans.makes.push(started..Instant::now());
        }
        std::thread::sleep(HOLD);
        // What the makes owe is not the frees' (`step_aside::pace`).
        let _ = crate::step_aside::pace(|| true);
        let began = Instant::now();
        for one in made {
            if order == Order::Paced {
                let _ = crate::step_aside::pace(|| false);
            }
            spans.frees.push((order, free(raw, one)));
        }
        spans.runs.push((order, began..Instant::now()));
        std::thread::sleep(GAP);
    }
    Some(spans)
}

/// The bench's device: the first GPU, its submission checks settled.
fn bench_device() -> Device {
    let options = DeviceOptions {
        kinds: vec![
            DeviceKind::Discrete,
            DeviceKind::Integrated,
            DeviceKind::Virtual,
        ],
        ..DeviceOptions::default()
    };
    let device = Device::system_default_with(&options).unwrap();
    // The submission checks' own imports and frees, out of the way first.
    let _settled = device.settle_submit_checks(WAIT);
    device
}

/// Commit a one-word fill on a queue of `device`'s own every [`PING`] and wait for it, until
/// `done`: each one's commit to completion.
fn ping(device: &Device, done: &AtomicBool) -> Vec<Range<Instant>> {
    let queue = device.new_command_queue().unwrap();
    let word = device.new_buffer(4096, ResourceOptions::private()).unwrap();
    let mut pings = Vec::new();
    while !done.load(Ordering::Acquire) {
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.fill_buffer(&word, 0..4, 2).unwrap();
        blit.end_encoding().unwrap();
        let committed = Instant::now();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        pings.push(committed..Instant::now());
        std::thread::sleep(PING);
    }
    pings
}

/// The device's name, vendor and driver version, as the bench prints them.
fn named(device: &Device) -> String {
    // SAFETY: the bench reads the instance's own query of its physical device.
    let handles = unsafe { device.raw() };
    // SAFETY: the physical device is the instance's.
    let properties = unsafe {
        handles
            .instance
            .get_physical_device_properties(handles.physical_device)
    };
    format!(
        "{} (vendor {:#06x}, driver version {:#x})",
        device.name(),
        properties.vendor_id,
        properties.driver_version
    )
}

#[test]
#[ignore = "a release benchmark: run with --ignored --nocapture"]
fn import_free_hold_bench() {
    let device = bench_device();
    // The bench's frees are owed as the runtime's own threads' are.
    crate::step_aside::mark_a_paced_thread();
    let done = AtomicBool::new(false);
    let kinds = [
        (Source::Pointer, 512 << 10, PIECES),
        (Source::Pointer, 64 << 20, 4),
        (Source::DmaBuf, 512 << 10, PIECES),
        (Source::Plain, 512 << 10, PIECES),
        (Source::Plain, 64 << 20, 4),
    ]
    .map(|(source, bytes, count)| Kind {
        source,
        bytes,
        count,
    });
    let (pings, measured) = std::thread::scope(|scope| {
        let stops = Stops(&done);
        let pinger = scope.spawn(|| ping(&device, &done));
        // The pinger's queue and its first pings before the first make.
        std::thread::sleep(Duration::from_millis(500));
        let measured: Vec<Spans> = kinds
            .into_iter()
            .filter_map(|kind| measure(&device, kind))
            .collect();
        drop(stops);
        (pinger.join().unwrap(), measured)
    });
    eprintln!("IMPORT-FREE-HOLD device: {}", named(&device));
    let quiet = pings.iter().filter(|ping| {
        !(measured.iter())
            .flat_map(|spans| {
                spans
                    .makes
                    .iter()
                    .chain(spans.frees.iter().map(|(_, span)| span))
            })
            .any(|span| overlaps(ping, span))
    });
    let quiet = lengths(quiet);
    eprintln!(
        "IMPORT-FREE-HOLD pings beside no make or free: {}",
        spread(quiet)
    );
    for spans in &measured {
        let kind = spans.kind;
        eprintln!(
            "IMPORT-FREE-HOLD {kind}: makes {}",
            spread(lengths(spans.makes.iter()))
        );
        for order in [Order::InTurn, Order::Paced] {
            let of = |list: &[(Order, Range<Instant>)]| -> Vec<Range<Instant>> {
                (list.iter())
                    .filter(|(each, _)| *each == order)
                    .map(|(_, span)| span.clone())
                    .collect()
            };
            let (frees, runs) = (of(&spans.frees), of(&spans.runs));
            let beside = pings
                .iter()
                .filter(|ping| runs.iter().any(|run| overlaps(ping, run)));
            eprintln!(
                "IMPORT-FREE-HOLD {kind}, {order}: frees {}",
                spread(lengths(frees.iter()))
            );
            eprintln!(
                "IMPORT-FREE-HOLD {kind}, {order}: runs {}",
                spread(lengths(runs.iter()))
            );
            eprintln!(
                "IMPORT-FREE-HOLD {kind}, {order}: pings beside a run {}",
                spread(lengths(beside))
            );
        }
    }
}

/// What the completion leg measured of one size: each round's commit to its last deallocator,
/// its frees' run (the first deallocator to the last), and how long after the wait returned its
/// last deallocator ran (zero where before).
struct Completed {
    bytes: usize,
    count: usize,
    windows: Vec<Range<Instant>>,
    runs: Vec<Duration>,
    after_wait: Vec<Duration>,
}

/// [`PIECES`] no-copy buffers of 512 KiB, or four of 64 MiB, over anonymous pages, each with a
/// deallocator that notes when it ran into `ran`; `None`, printed, where the device makes none.
fn no_copy_buffers(
    device: &Device,
    pages: &Pages,
    (bytes, count): (usize, usize),
    ran: &Arc<Mutex<Vec<Instant>>>,
) -> Option<Vec<Buffer>> {
    let mut buffers = Vec::with_capacity(count);
    for index in 0..count {
        let ran = Arc::clone(ran);
        let deallocator: crate::Deallocator = Box::new(move |_: NonNull<u8>, _: u64| {
            ran.lock().unwrap().push(Instant::now());
        });
        // SAFETY: the pages outlive every buffer over them (`Pages` drops after the leg waits for
        // every deallocator), and only these buffers' commands use them.
        let made = unsafe {
            device.new_buffer_with_bytes_no_copy(
                HostMemory::new(pages.at(index * bytes)),
                bytes as u64,
                ResourceOptions::default(),
                Some(deallocator),
            )
        };
        match made {
            Ok(buffer) => buffers.push(buffer),
            Err(error) => {
                eprintln!("IMPORT-FREE-HOLD completed: no no-copy buffer ({error:?}): not run");
                return None;
            }
        }
    }
    Some(buffers)
}

/// [`ROUNDS`] rounds of the completion leg (the module docs) of `count` buffers of `bytes` on
/// `device`, [`GAP`] apart; `None` where it cannot make them.
fn completed_rounds(device: &Device, (bytes, count): (usize, usize)) -> Option<Completed> {
    let pages = Pages::new(bytes * count, 64 << 10);
    let queue = device.new_command_queue().unwrap();
    let target = device.new_buffer(4096, ResourceOptions::private()).unwrap();
    let mut completed = Completed {
        bytes,
        count,
        windows: Vec::new(),
        runs: Vec::new(),
        after_wait: Vec::new(),
    };
    for _ in 0..ROUNDS {
        let ran = Arc::new(Mutex::new(Vec::new()));
        let buffers = no_copy_buffers(device, &pages, (bytes, count), &ran)?;
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        for (index, buffer) in buffers.iter().enumerate() {
            let at = 4 * (index as u64 % 1024);
            blit.copy_from_buffer_to_buffer(buffer, 0, &target, at, 4)
                .unwrap();
        }
        blit.end_encoding().unwrap();
        let committed = Instant::now();
        command_buffer.commit().unwrap();
        // The command buffer in flight holds the pages last: its completion frees the imports.
        drop(buffers);
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        let waited = Instant::now();
        let deadline = waited + WAIT;
        while ran.lock().unwrap().len() < count {
            assert!(Instant::now() < deadline, "a deallocator never ran");
            std::thread::sleep(Duration::from_micros(100));
        }
        let mut ran = ran.lock().unwrap().clone();
        ran.sort();
        let (first, last) = (ran[0], ran[ran.len() - 1]);
        completed.windows.push(committed..last);
        completed.runs.push(last - first);
        completed
            .after_wait
            .push(last.saturating_duration_since(waited));
        std::thread::sleep(GAP);
    }
    Some(completed)
}

#[test]
#[ignore = "a release benchmark: run with --ignored --nocapture"]
fn import_free_hold_bench_completed() {
    let device = bench_device();
    let done = AtomicBool::new(false);
    let before = device.work_counts();
    let (pings, measured) = std::thread::scope(|scope| {
        let stops = Stops(&done);
        let pinger = scope.spawn(|| ping(&device, &done));
        // The pinger's queue and its first pings before the first make.
        std::thread::sleep(Duration::from_millis(500));
        let measured: Vec<Completed> = [(512 << 10, PIECES), (64 << 20, 4)]
            .into_iter()
            .filter_map(|size| completed_rounds(&device, size))
            .collect();
        drop(stops);
        (pinger.join().unwrap(), measured)
    });
    let after = device.work_counts();
    eprintln!("IMPORT-FREE-HOLD completed device: {}", named(&device));
    let quiet = pings.iter().filter(|ping| {
        !(measured.iter())
            .flat_map(|completed| completed.windows.iter())
            .any(|window| overlaps(ping, window))
    });
    eprintln!(
        "IMPORT-FREE-HOLD completed: pings beside no round {}",
        spread(lengths(quiet))
    );
    for completed in &measured {
        let size = format!(
            "{} no-copy buffers of {} KiB",
            completed.count,
            completed.bytes >> 10
        );
        let beside = pings
            .iter()
            .filter(|ping| (completed.windows.iter()).any(|window| overlaps(ping, window)));
        eprintln!(
            "IMPORT-FREE-HOLD completed {size}: frees' runs {}",
            spread(completed.runs.clone())
        );
        eprintln!(
            "IMPORT-FREE-HOLD completed {size}: last deallocator after the wait returned {}",
            spread(completed.after_wait.clone())
        );
        eprintln!(
            "IMPORT-FREE-HOLD completed {size}: pings beside a round {}",
            spread(lengths(beside))
        );
    }
    eprintln!(
        "IMPORT-FREE-HOLD completed counts: memory_frees_posted {}, memory_frees_paced {} \
         ({:.3} ms waited)",
        after.memory_frees_posted - before.memory_frees_posted,
        after.memory_frees_paced - before.memory_frees_paced,
        (after.memory_frees_paced_nanos - before.memory_frees_paced_nanos) as f64 / 1e6
    );
}
