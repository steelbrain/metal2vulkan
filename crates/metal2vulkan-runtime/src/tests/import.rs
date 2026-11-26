//! Imports: the two type rules, the dma-buf rule's refusal on a device whose Shared type is not
//! host-coherent (Meteor Lake's types under Mesa's Intel driver and i915), and under a device cfg,
//! on every device of the cfg's kind, that
//! host memory is imported into the first host-visible, host-coherent type its query offers, that
//! a GPU write reaches it and its release runs only once its memory is freed, that a texture over
//! a buffer in host memory's import is read and written by blits through the counted copy, that an
//! ordinary Shared allocation succeeds after an import of a shared mapping of a file (which RADV
//! refuses at the import), that an import counts against the device's allocations, and that a
//! dma-buf is imported where its first offered type is host-visible and host-coherent and refused
//! at the query, with nothing imported, everywhere else; and, on every device the floor opens
//! through the refused device's memory view, that a dma-buf is refused for the Shared type's
//! coherence, alone and several behind one buffer, with nothing imported, while host memory of
//! the same length takes the host-memory rule's answer.

use crate::import::host_type;
#[cfg(unix)]
use crate::{ImportHandle, Limit};

/// The dma-buf rule takes the first type the fd's query offers, only where it is host-visible,
/// host-coherent and listed by the buffer's requirements; never a later one.
#[cfg(unix)]
#[test]
fn a_dma_buf_takes_only_the_first_type_offered() {
    use crate::import::dma_buf_type;
    // Coherent types 1, 2, 3, 4 and 5; the buffer lists 0 to 5.
    let coherent = 0b11_1110;
    let listed = 0b11_1111;
    let cases = [
        // The first offered (0) is not coherent: refused, though 4 is offered and coherent.
        (0x11, None),
        (0x7f, None),
        // The first offered (5) is coherent and listed.
        (0x420, Some(5)),
        (0x20, Some(5)),
        // The first offered (1) is coherent but not listed by a buffer that lists 0 and 2.
        (0x06, None),
        (0, None),
    ];
    for (offered, expected) in cases {
        let listed = if offered == 0x06 { 0b101 } else { listed };
        assert_eq!(
            dma_buf_type(ImportHandle::DmaBuf, true, offered, listed, coherent).ok(),
            expected,
            "offered {offered:#x}"
        );
    }
}

/// On a device whose Shared type is not host-coherent, the dma-buf rule refuses whatever the fd's
/// query offers, and the host-memory rule still takes the pointer's type. Meteor Lake's types
/// under Mesa's Intel driver and the i915 kernel driver (src/intel/vulkan/i915/anv_device.c, a GPU
/// without the shared last-level cache): 0 device-local, host-visible and host-coherent
/// (write-combined), 1 device-local, host-visible and host-cached, not coherent. The floor's
/// choice puts Shared in type 1, so the device holds a coherence state, and the fd's offer of
/// type 0, which the rule took before, is refused; the pointer's offer of type 0 is taken. On the
/// Iris Xe's types (i915 with the last-level cache: 0 device-local, 1 that and host-visible and
/// host-coherent, 2 that and host-cached), Shared is coherent, no state is held, and the fd's offer
/// of type 1 is taken as before.
#[cfg(unix)]
#[test]
fn a_dma_buf_is_refused_where_the_shared_type_is_not_host_coherent() {
    use crate::coherence::Coherence;
    use crate::floor::choose;
    use crate::import::dma_buf_type;
    use crate::tests::floor::memory;
    use ash::vk::MemoryPropertyFlags as F;
    let local = F::DEVICE_LOCAL;
    let coherent_type = F::HOST_VISIBLE | F::HOST_COHERENT;
    // The rule's `coherent` bits: the host-visible, host-coherent types.
    let coherent_bits = |memory: &ash::vk::PhysicalDeviceMemoryProperties| {
        (0u32..)
            .zip(memory.memory_types_as_slice())
            .filter(|(_, kind)| kind.property_flags.contains(coherent_type))
            .fold(0, |bits, (index, _)| bits | 1 << index)
    };
    let meteor_lake = memory(&[
        local | coherent_type,
        local | F::HOST_VISIBLE | F::HOST_CACHED,
    ]);
    let types = choose(&meteor_lake, u32::MAX).expect("Meteor Lake's types");
    assert_eq!(types.shared, 1);
    let held = Coherence::of(&meteor_lake, types, 64).is_some();
    assert!(
        held,
        "a coherence state for a Shared type that is not coherent"
    );
    let coherent = coherent_bits(&meteor_lake);
    assert_eq!(coherent, 0b01);
    for offered in [0b01, 0b11, 0b10] {
        assert_eq!(
            dma_buf_type(ImportHandle::DmaBuf, !held, offered, 0b11, coherent),
            Err(Limit::ImportCoherence {
                handle: ImportHandle::DmaBuf
            }),
            "offered {offered:#x}"
        );
        assert_eq!(
            dma_buf_type(ImportHandle::DmaBufs, !held, offered, 0b11, coherent),
            Err(Limit::ImportCoherence {
                handle: ImportHandle::DmaBufs
            }),
            "offered {offered:#x}"
        );
    }
    // The rule before the state keyed it took the fd's offer of type 0.
    assert_eq!(
        dma_buf_type(ImportHandle::DmaBuf, true, 0b01, 0b11, coherent),
        Ok(0)
    );
    // The pointer's offer of type 0 is taken.
    assert_eq!(host_type(0b01, 0b11, coherent), Some(0));

    let iris = memory(&[
        local,
        local | coherent_type,
        local | coherent_type | F::HOST_CACHED,
    ]);
    let types = choose(&iris, u32::MAX).expect("the Iris Xe's types");
    assert_eq!(types.shared, 2);
    let held = Coherence::of(&iris, types, 64).is_some();
    assert!(!held, "no coherence state for a coherent Shared type");
    let coherent = coherent_bits(&iris);
    assert_eq!(
        dma_buf_type(ImportHandle::DmaBuf, !held, 0b110, 0b111, coherent),
        Ok(1)
    );
    assert_eq!(
        dma_buf_type(ImportHandle::DmaBuf, !held, 0b001, 0b111, coherent),
        Err(Limit::ImportType {
            handle: ImportHandle::DmaBuf,
            offered: 0b001,
            listed: 0b111
        })
    );
}

/// The host-memory rule takes the first type offered that is host-visible, host-coherent and
/// listed by the buffer's requirements.
#[test]
fn host_memory_takes_the_first_coherent_listed_type_offered() {
    // Offered 3 and 4, both coherent: 3.
    assert_eq!(host_type(0x18, 0x3b, 0b1_1110), Some(3));
    // Offered 0 to 6, 0 not coherent: 1.
    assert_eq!(host_type(0x7f, 0x07, 0b110), Some(1));
    // Offered 5 alone, coherent and listed.
    assert_eq!(host_type(0x20, 0x7ad, 0b10_0000), Some(5));
    // Nothing offered that is coherent and listed.
    assert_eq!(host_type(0x01, 0xff, 0b1110), None);
    assert_eq!(host_type(0x08, 0x07, 0b1000), None);
    assert_eq!(host_type(0, 0xff, 0xff), None);
}

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
pub(crate) mod on_device {
    use crate::memory::Release;
    use crate::tests::command::on_device::{each_device, finished, within, Busy};
    use crate::tests::devices_of_the_cfgs_kind;
    use crate::texture::TextureObject;
    use crate::{
        BufferLayout, CommandBufferError, Device, DeviceLoss, DeviceOptions, Error, ImportHandle,
        IndexClass, Limit, MemoryClass, Metrics, PixelFormat, Region, StorageMode,
        TextureDescriptor, TexturePlace, TextureType, TextureUsage,
    };
    use ash::vk;
    use std::alloc::Layout;
    use std::ptr::NonNull;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::thread;
    use std::time::{Duration, Instant};

    /// The runtime's own allocations besides the pool's, which the block-maker thread makes as
    /// its first work: the zero source, the parameter ring's first block and the staging region.
    const INTERNALS: u32 = 3;

    /// Waits up to 120 s for `done` on `device`. A timeout's message says what the device did
    /// during the wait: its submissions, the command buffers that completed with GPU times,
    /// whether one completed in the wait's last 10 s, and the releases still waiting. So a stall
    /// tells the GPU's completions stopping apart from a release that did not follow completions
    /// that came. It ends with what the completion side holds then (`Done::stall`): the command
    /// buffers in flight, whether the oldest's progress value is signalled, and where the
    /// completion thread waits, so a stall also tells work the GPU did not finish apart from work
    /// it finished that nothing collected; and with what the submit side holds (`Work::stall`),
    /// for work that never reached the GPU.
    pub(crate) fn wait_until(device: &Device, what: &str, mut done: impl FnMut() -> bool) {
        let counters = device.counters();
        let completed = |metrics: &Metrics| metrics.gpu_run.iter().sum::<u64>();
        let start = Instant::now();
        let deadline = start + Duration::from_secs(120);
        let first = counters.metrics();
        let (mut seen, mut sampled, mut last) = (completed(&first), start, None);
        while !done() {
            let now = Instant::now();
            // Read every 100 ms, not at each poll: the read takes the sites' locks.
            if now >= sampled + Duration::from_millis(100) || now >= deadline {
                sampled = now;
                let count = completed(&counters.metrics());
                if count != seen {
                    (seen, last) = (count, Some(now));
                }
            }
            if now >= deadline {
                let core = device.core();
                let (completing, submitting) = (core.done.stall(&core.device), core.work.stall());
                let (metrics, gauges) = (counters.metrics(), counters.gauges());
                let recent = last.is_some_and(|at| now - at <= Duration::from_secs(10));
                let at = last.map_or_else(
                    || "none".to_owned(),
                    |at| format!("{:.1} s into it", (at - start).as_secs_f64()),
                );
                panic!(
                    "timed out waiting for {what}: in the {} s of the wait, {} submissions, {} \
                     command buffers completed with GPU times ({} recorded untimed), the last \
                     completion seen {at}, one in its last 10 s: {recent}; releases_waiting {}, \
                     imports {}; {completing}; {submitting}",
                    (now - start).as_secs(),
                    metrics.submissions - first.submissions,
                    completed(&metrics) - completed(&first),
                    metrics.untimed_command_buffers - first.untimed_command_buffers,
                    gauges.releases_waiting,
                    gauges.imports,
                );
            }
            thread::sleep(Duration::from_millis(1));
        }
    }

    /// `len` bytes of host memory of the test's own at a multiple of `alignment`, each byte its
    /// offset times 7, and the allocation to free.
    pub(crate) fn host_memory(len: u64, alignment: u64) -> (NonNull<u8>, (NonNull<u8>, Layout)) {
        let layout = Layout::from_size_align(len as usize, alignment as usize).expect("a layout");
        // SAFETY: the layout's size is not zero.
        let pointer = NonNull::new(unsafe { std::alloc::alloc(layout) }).expect("an allocation");
        for at in 0..len as usize {
            // SAFETY: `at` is inside the allocation.
            unsafe { pointer.add(at).write((at * 7) as u8) };
        }
        (pointer, (pointer, layout))
    }

    pub(crate) fn free((pointer, layout): (NonNull<u8>, Layout)) {
        // SAFETY: an allocation of `layout` the test made, which nothing uses any more.
        unsafe { std::alloc::dealloc(pointer.as_ptr(), layout) };
    }

    /// A release that sets a flag, and the flag.
    pub(crate) fn releasing() -> (Option<Release>, Arc<AtomicBool>) {
        let ran = Arc::new(AtomicBool::new(false));
        let set = Arc::clone(&ran);
        let release: Release = Box::new(move || set.store(true, Ordering::SeqCst));
        (Some(release), ran)
    }

    /// The readings a timed-out `wait_until` ends with say what each side holds: a committed
    /// command buffer that waits on an event nothing signalled yet is the submit side's, held for
    /// the event's value, and no command buffer is in flight; once the event is signalled and the
    /// command buffer completed, the submit side holds none.
    #[test]
    fn every_device_s_stall_readings_name_a_command_buffer_held_for_an_event() {
        each_device("stall_readings", |name, _, device| {
            let core = device.core();
            let event = device.new_shared_event().expect("a shared event");
            let block = (device.take_block(MemoryClass::Shared, 4096)).expect("a Shared block");
            let buffer = (device.new_buffer(block.id, 0, 4096, None)).expect("a buffer");
            let queue = device.new_command_queue(1).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            (command_buffer.encode_wait_for_event(event.as_event(), 1)).expect("a wait");
            let mut encoder = command_buffer.blit_command_encoder().expect("an encoder");
            encoder.fill_buffer(&buffer, 0..4, 7).expect("a fill");
            encoder.end_encoding();
            command_buffer.commit().expect("a commit");
            let held = "1 heads held for an event's value";
            let start = Instant::now();
            let submitting = loop {
                let submitting = core.work.stall();
                if submitting.contains(held) || start.elapsed() > Duration::from_secs(10) {
                    break submitting;
                }
                thread::sleep(Duration::from_millis(1));
            };
            assert!(submitting.contains(held), "{name}: {submitting}");
            assert!(submitting.ends_with("; not lost"), "{name}: {submitting}");
            let completing = core.done.stall(&core.device);
            assert!(
                completing.starts_with("in flight 0 command buffers"),
                "{name}: {completing}"
            );
            event.set_signaled_value(1);
            finished(name, &command_buffer);
            let after = core.work.stall();
            assert!(
                after.contains(" 0 command buffers not wholly submitted"),
                "{name}: {after}"
            );
            println!("stall readings: {name}: {submitting}; {completing}");
        });
    }

    /// A loss while an import's release waits does not keep the release from running: host
    /// memory imported into a Shared block that a committed command buffer fills, the block
    /// dropped, so its release waits for that command buffer. Held on an event nothing signals
    /// (the submit side's), the command buffer ends at the loss DeviceLost not in flight; behind
    /// busy work (on the GPU), DeviceLost in flight once the device is idle, which the injected
    /// loss reaches when the busy work is let go, as a lost device's work ends. Either way the
    /// release runs. Each case has a device of its own, as a lost device stays lost. The loss is
    /// the consumer's report (`Device::report_loss`), which takes the runtime's own loss path
    /// with no GPU fault.
    #[test]
    fn every_device_runs_a_waiting_release_after_a_loss() {
        let loss = DeviceLoss {
            call: "vkQueuePresentKHR",
            result: vk::Result::ERROR_DEVICE_LOST,
            fault: None,
        };
        for in_flight in [false, true] {
            each_device("release_after_loss", |name, _, device| {
                let Some(alignment) = (device.core().imports.host.as_ref()).map(|(_, at)| *at)
                else {
                    println!("release after a loss: {name}: imports no host memory; not run");
                    return;
                };
                let len = (64u64 << 10).next_multiple_of(alignment);
                let (pointer, mapping) = host_memory(len, alignment);
                let (release, released) = releasing();
                // SAFETY: the memory is the test's, mapped until it frees it after the release.
                let (block, _) = unsafe { device.import_host_memory(pointer, len, release) }
                    .unwrap_or_else(|error| panic!("{name}: an import: {error}"));
                let buffer = (device.new_buffer(block.id, 0, len, None)).expect("a buffer");
                let event = device.new_shared_event().expect("a shared event");
                let busy = in_flight.then(|| Busy::new(&device));
                let queue = device.new_command_queue(4).expect("a queue");
                let filled = queue.command_buffer().expect("a command buffer");
                if !in_flight {
                    (filled.encode_wait_for_event(event.as_event(), 1)).expect("a wait");
                }
                let mut encoder = filled.blit_command_encoder().expect("an encoder");
                encoder.fill_buffer(&buffer, 0..len, 0x5a).expect("a fill");
                encoder.end_encoding();
                let before = device.counters().metrics().submissions;
                filled.commit().expect("a commit");
                if in_flight {
                    within("the fill's submission", || {
                        device.counters().metrics().submissions > before
                    });
                }
                drop((buffer, block));
                thread::sleep(Duration::from_millis(20));
                let waits = !released.load(Ordering::SeqCst);
                assert!(waits, "{name}: in flight {in_flight}: the release waits");
                device.report_loss(loss.clone());
                if let Some(busy) = &busy {
                    busy.open();
                }
                wait_until(&device, "the release after a loss", || {
                    released.load(Ordering::SeqCst)
                });
                finished(name, &filled);
                assert_eq!(
                    filled.error(),
                    Some(CommandBufferError::DeviceLost { in_flight }),
                    "{name}"
                );
                drop((busy, filled, queue, event, device));
                free(mapping);
                println!("release after a loss: {name}: in flight {in_flight}: the release ran");
            });
        }
    }

    /// A loss while an import's release waits for other work does not keep the release from
    /// running: host memory imported into a Shared block nothing uses, given back while a fill of
    /// another block is in flight behind busy work, so its release waits for the fill's
    /// submission to complete. At the loss, once the busy work is let go, the device's idle wait
    /// covers the fill: the fill ends DeviceLost in flight, and what waited for its progress value
    /// goes, the release among it. The completion side's mark of that value is printed.
    #[test]
    fn every_device_runs_a_release_waiting_on_other_work_after_a_loss() {
        let loss = DeviceLoss {
            call: "vkQueuePresentKHR",
            result: vk::Result::ERROR_DEVICE_LOST,
            fault: None,
        };
        each_device("release_behind_other_work", |name, _, device| {
            let Some(alignment) = (device.core().imports.host.as_ref()).map(|(_, at)| *at) else {
                println!("release after a loss: {name}: imports no host memory; not run");
                return;
            };
            let other = (device.take_block(MemoryClass::Shared, 4096)).expect("a Shared block");
            let target = (device.new_buffer(other.id, 0, 4096, None)).expect("a buffer");
            let busy = Busy::new(&device);
            let queue = device.new_command_queue(4).expect("a queue");
            let filled = queue.command_buffer().expect("a command buffer");
            let mut encoder = filled.blit_command_encoder().expect("an encoder");
            encoder.fill_buffer(&target, 0..4096, 0x5a).expect("a fill");
            encoder.end_encoding();
            let before = device.counters().metrics().submissions;
            filled.commit().expect("a commit");
            within("the fill's submission", || {
                device.counters().metrics().submissions > before
            });
            let len = (64u64 << 10).next_multiple_of(alignment);
            let (pointer, mapping) = host_memory(len, alignment);
            let (release, released) = releasing();
            // SAFETY: the memory is the test's, mapped until it frees it after the release.
            let (block, _) = unsafe { device.import_host_memory(pointer, len, release) }
                .unwrap_or_else(|error| panic!("{name}: an import: {error}"));
            drop(block);
            thread::sleep(Duration::from_millis(20));
            let waits = !released.load(Ordering::SeqCst);
            assert!(waits, "{name}: the release waits for the fill");
            let marked = device.core().done.completed();
            device.report_loss(loss.clone());
            busy.open();
            wait_until(
                &device,
                "the release behind other work after a loss",
                || released.load(Ordering::SeqCst),
            );
            finished(name, &filled);
            assert_eq!(
                filled.error(),
                Some(CommandBufferError::DeviceLost { in_flight: true }),
                "{name}"
            );
            println!(
                "release after a loss: {name}: behind other work: the completion side's mark {} \
                 before the loss, {} after; the release ran",
                marked,
                device.core().done.completed()
            );
            drop((busy, filled, queue, target, other, device));
            free(mapping);
        });
    }

    /// Host memory is imported into a Shared block whose contents are the memory: a fill of its
    /// second half by the GPU reaches the host, the first half keeps the host's bytes, and the
    /// release runs once the last handle and buffer are gone and the memory is freed, with the
    /// gauges back. A device that does not import host memory refuses it typed.
    #[test]
    fn every_device_imports_host_memory_and_releases_it_after_the_free() {
        each_device("import_host_memory", |name, _, device| {
            let counters = device.counters();
            let Some((_, alignment)) = device.core().imports.host.as_ref() else {
                let (pointer, mapping) = host_memory(64 << 10, 4096);
                let (release, _) = releasing();
                // SAFETY: the memory is the test's, mapped until it unmaps it below.
                match unsafe { device.import_host_memory(pointer, 64 << 10, release) } {
                    Err(Error::Limit(Limit::NoImport {
                        handle: ImportHandle::HostMemory,
                    })) => {}
                    other => panic!("{name}: {other:?}"),
                }
                free(mapping);
                println!("import: {name}: imports no host memory, refused typed");
                return;
            };
            let alignment = *alignment;
            let len = (64u64 << 10).next_multiple_of(alignment);
            let (pointer, mapping) = host_memory(len, alignment);
            let before = counters.gauges();
            let (release, released) = releasing();
            // SAFETY: the memory is the test's, mapped until it unmaps it after the release.
            let (block, memory_type) = unsafe { device.import_host_memory(pointer, len, release) }
                .unwrap_or_else(|error| panic!("{name}: an import: {error}"));
            assert_eq!(
                (block.class, block.contents, block.len, block.offset),
                (MemoryClass::Shared, Some(pointer), len, 0),
                "{name}"
            );
            let live = counters.gauges();
            assert_eq!(
                (live.imports, live.blocks),
                (1, before.blocks + 1),
                "{name}"
            );
            let buffer = (device.new_buffer(block.id, 0, len, None)).expect("a buffer");
            assert_eq!(buffer.contents(), Some(pointer), "{name}");
            let queue = device.new_command_queue(4).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = command_buffer.blit_command_encoder().expect("an encoder");
            encoder
                .fill_buffer(&buffer, len / 2..len, 0x5a)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            encoder.end_encoding();
            command_buffer.commit().expect("a commit");
            finished(name, &command_buffer);
            assert_eq!(command_buffer.error(), None, "{name}");
            for at in 0..len as usize {
                // SAFETY: `at` is inside the imported memory, which the test still maps.
                let byte = unsafe { pointer.add(at).read_volatile() };
                let expected = if at < len as usize / 2 {
                    (at * 7) as u8
                } else {
                    0x5a
                };
                assert_eq!(byte, expected, "{name}: byte {at}");
            }
            drop((command_buffer, queue, buffer));
            assert!(
                !released.load(Ordering::SeqCst),
                "{name}: released while held"
            );
            let device_local = block.device_local;
            let held = counters.gauges().blocks;
            drop(block);
            wait_until(&device, "the release", || released.load(Ordering::SeqCst));
            let after = counters.gauges();
            assert_eq!((after.imports, after.blocks), (0, held - 1), "{name}");
            free(mapping);
            println!(
                "import: {name}: host memory imported into type {memory_type} (device-local \
                 {device_local}), written by the GPU, released after its free"
            );
        });
    }

    /// A Managed buffer over imported host memory whose bytes are all named for the next commit
    /// (`new_managed_buffer` with `copied` its length, as `newBufferWithBytesNoCopy:` of Managed
    /// memory makes one) lets the import go as its last handle and the block's go, with no commit
    /// after them: on a device that keeps a device copy, the ranges named hold the buffer weakly,
    /// so nothing keeps its blocks, and the import's release runs within 10 s, the imports gauge
    /// back to 0. Metal runs a no-copy buffer's deallocator as it deallocates the buffer. On a
    /// device that keeps one copy nothing is named, and the same holds. A device that does not
    /// import host memory has nothing to show.
    #[test]
    fn every_device_releases_an_import_under_a_managed_buffer_dropped_before_any_commit() {
        each_device("import_managed_drop", |name, _, device| {
            let counters = device.counters();
            let Some((_, alignment)) = device.core().imports.host.as_ref() else {
                println!("import: {name}: imports no host memory");
                return;
            };
            let alignment = *alignment;
            let len = (64u64 << 10).next_multiple_of(alignment);
            let (pointer, mapping) = host_memory(len, alignment);
            let before = counters.gauges();
            let (release, released) = releasing();
            // SAFETY: the memory is the test's, mapped until it unmaps it after the release.
            let (block, _) = unsafe { device.import_host_memory(pointer, len, release) }
                .unwrap_or_else(|error| panic!("{name}: an import: {error}"));
            let two = !device.has_unified_memory();
            let copy = two
                .then(|| (device.take_block(MemoryClass::Private, len)).expect("a Private block"));
            let place = copy.as_ref().map(|copy| (copy.id, 0));
            let buffer = (device.new_managed_buffer((block.id, 0), place, len, len))
                .unwrap_or_else(|error| panic!("{name}: a Managed buffer: {error}"));
            assert_eq!(counters.gauges().imports, 1, "{name}");
            drop((buffer, block));
            let deadline = Instant::now() + Duration::from_secs(10);
            while !released.load(Ordering::SeqCst) {
                assert!(
                    Instant::now() < deadline,
                    "{name}: the import's release did not run within 10 s of the buffer's drop \
                     (device copy {two})"
                );
                thread::sleep(Duration::from_millis(1));
            }
            // The blocks gauge is not compared: the block-maker thread may make an allocation
            // ahead of the Private take meanwhile.
            let after = counters.gauges();
            assert_eq!(
                (after.imports, after.buffers),
                (0, before.buffers),
                "{name}: the import gone with the buffer"
            );
            drop(copy);
            assert_eq!(counters.metrics().managed_bytes_captured, 0, "{name}");
            free(mapping);
            println!(
                "import: {name}: a Managed buffer over host memory (device copy {two}) dropped \
                 with its bytes named and no commit, the import released"
            );
        });
    }

    /// A texture over a buffer in an imported block of host memory, read and written by shaders,
    /// is the counted copy on every device: a blit's read of it into a Shared buffer finds the
    /// host's rows, a blit's write to it reaches the host's memory, and the copies between the
    /// buffer and its image count in `Metrics::over_copies` and, being over an imported block, in
    /// `Metrics::over_copies_imported`, each its texels' bytes.
    #[test]
    fn every_device_makes_a_texture_over_an_imported_buffer() {
        each_device("import_texture", |name, _, device| {
            const WIDTH: u32 = 64;
            const HEIGHT: u32 = 8;
            let Some((_, alignment)) = device.core().imports.host.as_ref() else {
                println!("import: {name}: imports no host memory");
                return;
            };
            let alignment = *alignment;
            let row = u64::from(WIDTH) * 4;
            let rows = u64::from(HEIGHT);
            let answer = (device.minimum_linear_texture_alignment(PixelFormat::RGBA8Unorm))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let pitch = row.next_multiple_of(answer);
            let len = (pitch * rows).next_multiple_of(alignment);
            let (pointer, mapping) = host_memory(len, alignment);
            let (release, released) = releasing();
            // SAFETY: the memory is the test's, mapped until it unmaps it after the release.
            let (block, memory_type) = unsafe { device.import_host_memory(pointer, len, release) }
                .unwrap_or_else(|error| panic!("{name}: an import: {error}"));
            let buffer = (device.new_buffer(block.id, 0, len, None)).expect("a buffer");
            let first = IndexClass::Texture.range().start;
            (device.grant_indices(IndexClass::Texture, first, 1)).expect("a grant");
            let mut d =
                TextureDescriptor::new(TextureType::Type2D, PixelFormat::RGBA8Unorm, WIDTH, HEIGHT);
            d.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
            let texture = (buffer.new_texture(&d, Some(first), 0, pitch))
                .unwrap_or_else(|error| panic!("{name}: a texture over the import: {error}"));
            assert_eq!(
                texture.descriptor().storage_mode,
                StorageMode::Shared,
                "{name}"
            );
            assert!(texture.copied_over().is_some(), "{name}: copied");
            let copies = || device.counters().metrics().over_copies;
            let start = device.counters().metrics();
            let before = copies();
            // The host's rows, as the image's texels.
            let host = |at: u64| {
                // SAFETY: `at` is inside the imported memory, which the test still maps.
                unsafe { pointer.add(at as usize).read_volatile() }
            };
            let packed = |byte: &dyn Fn(u64) -> u8| -> Vec<u8> {
                (0..rows)
                    .flat_map(|y| (0..row).map(move |x| (y, x)))
                    .map(|(y, x)| byte(y * pitch + x))
                    .collect()
            };
            let queue = device.new_command_queue(4).expect("a queue");

            // Read: a blit of the image into a Shared buffer finds the host's rows.
            let out_block = (device.take_block(MemoryClass::Shared, row * rows))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let out = (device.new_buffer(out_block.id, 0, row * rows, Some(0))).expect("a buffer");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = command_buffer.blit_command_encoder().expect("an encoder");
            let layout = BufferLayout {
                offset: 0,
                bytes_per_row: row,
                bytes_per_image: 0,
            };
            encoder
                .copy_from_texture_to_buffer(
                    &texture,
                    TexturePlace::default(),
                    [WIDTH, HEIGHT, 1],
                    &out,
                    layout,
                )
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            encoder.end_encoding();
            command_buffer.commit().expect("a commit");
            finished(name, &command_buffer);
            assert_eq!(command_buffer.error(), None, "{name}");
            let contents = out.contents().expect("Shared");
            // SAFETY: the buffer's bytes, which no command writes any more.
            let read =
                unsafe { std::slice::from_raw_parts(contents.as_ptr(), (row * rows) as usize) };
            let written: Vec<u8> = (0..rows)
                .flat_map(|y| (0..row).map(move |x| ((y * pitch + x) * 7) as u8))
                .collect();
            assert!(
                read == written.as_slice() && packed(&host) == written,
                "{name}: the image read the host's bytes"
            );
            let copied_in = copies();
            assert!(copied_in > before, "{name}: the bytes copied in, counted");

            // Write: a blit from a plain texture reaches the host's memory.
            let mut plain_d = d;
            plain_d.usage = TextureUsage::RENDER_TARGET;
            plain_d.storage_mode = StorageMode::Shared;
            let plain = (device.new_texture_for_views(&plain_d))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let source: Vec<u8> = (0..row * rows).map(|i| (i % 241) as u8 ^ 0x3c).collect();
            let whole = Region {
                origin: [0; 3],
                size: [WIDTH, HEIGHT, 1],
            };
            (plain.replace_region(whole, 0, 0, crate::HostBytes::from(&source[..]), row, 0))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = command_buffer.blit_command_encoder().expect("an encoder");
            encoder
                .copy_from_texture(
                    &plain,
                    TexturePlace::default(),
                    [WIDTH, HEIGHT, 1],
                    &texture,
                    TexturePlace::default(),
                )
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            encoder.end_encoding();
            command_buffer.commit().expect("a commit");
            finished(name, &command_buffer);
            assert_eq!(command_buffer.error(), None, "{name}");
            assert!(
                packed(&host) == source,
                "{name}: the write reached the host"
            );
            assert!(
                copies() > copied_in,
                "{name}: the image copied back, counted"
            );
            let counted = copies() - before;
            // Each copy is over the imported block: counted on the imported side too, each of
            // the texture's texels.
            let end = device.counters().metrics();
            let imported = end.over_copies_imported - start.over_copies_imported;
            let imported_bytes = end.over_copy_bytes_imported - start.over_copy_bytes_imported;
            assert_eq!(imported, counted, "{name}: imported copies");
            assert_eq!(
                (imported_bytes, end.over_copy_bytes - start.over_copy_bytes),
                (counted * row * rows, counted * row * rows),
                "{name}: imported bytes"
            );
            drop((
                command_buffer,
                queue,
                texture,
                plain,
                out,
                out_block,
                buffer,
                block,
            ));
            wait_until(&device, "the release", || released.load(Ordering::SeqCst));
            free(mapping);
            println!(
                "import: {name}: a texture over imported host memory (type {memory_type}), \
                 copied ({counted} copies counted), read and written"
            );
        });
    }

    /// After an import of 64 KiB of a shared mapping of a file (the shape of a guest's RAM, which
    /// RADV refuses at the import), whatever its answer, an ordinary Shared allocation succeeds:
    /// a take of 32 MiB, past the pool's 16 MiB allocations, which makes its own. A refused
    /// import leaves the gauges as they were.
    #[cfg(unix)]
    #[test]
    fn every_device_allocates_shared_memory_after_an_import_of_a_shared_file_mapping() {
        each_device("import_shared_file", |name, _, device| {
            let counters = device.counters();
            let alignment = match device.core().imports.host.as_ref() {
                Some((_, alignment)) => *alignment,
                None => {
                    println!("import: {name}: imports no host memory");
                    return;
                }
            };
            let len = (64u64 << 10).next_multiple_of(alignment);
            let file = tempfile(len);
            // SAFETY: a new shared mapping of the test's file, which nothing else uses.
            let base = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    len as usize,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_SHARED,
                    std::os::fd::AsRawFd::as_raw_fd(&file),
                    0,
                )
            };
            assert_ne!(base, libc::MAP_FAILED, "{name}: a shared mapping");
            let pointer = NonNull::new(base.cast::<u8>()).expect("a mapping's address");
            let before = counters.gauges();
            let (release, released) = releasing();
            // SAFETY: the memory is the test's, mapped until it unmaps it below.
            let answer = unsafe { device.import_host_memory(pointer, len, release) };
            let answered = match answer {
                Ok((block, _)) => {
                    drop(block);
                    wait_until(&device, "the release", || released.load(Ordering::SeqCst));
                    "imported".to_string()
                }
                Err(Error::Vulkan { call, result }) => {
                    assert_eq!(counters.gauges(), before, "{name}: a refused import");
                    format!("refused at {call} ({result:?})")
                }
                Err(Error::Limit(Limit::ImportType { offered, .. })) => {
                    assert_eq!(counters.gauges(), before, "{name}: a refused import");
                    format!("refused at the query (offered {offered:#x})")
                }
                Err(other) => panic!("{name}: {other}"),
            };
            let blocks = counters.gauges().blocks;
            let shared = (device.take_block(MemoryClass::Shared, 32 << 20))
                .unwrap_or_else(|error| panic!("{name}: a Shared allocation after it: {error}"));
            assert_eq!(
                counters.gauges().blocks,
                blocks + 1,
                "{name}: its own allocation"
            );
            let contents = shared.contents.expect("a Shared block's contents");
            // SAFETY: the block's mapping spans it.
            unsafe { contents.as_ptr().write_volatile(1) };
            drop(shared);
            // SAFETY: the test's mapping, which nothing uses any more.
            assert_eq!(unsafe { libc::munmap(base, len as usize) }, 0);
            println!(
                "import: {name}: a shared file mapping {answered}; a Shared allocation after it"
            );
        });
    }

    /// A file of `len` bytes in the OS temp dir, removed from the directory at once.
    #[cfg(unix)]
    fn tempfile(len: u64) -> std::fs::File {
        let path = std::env::temp_dir().join(format!("m2v-import-{}", std::process::id()));
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)
            .expect("a scratch file");
        std::fs::remove_file(&path).expect("the scratch file's removal");
        file.set_len(len).expect("the scratch file's length");
        file
    }

    /// An import counts against the device's `maxMemoryAllocationCount`: with the count answered
    /// as one past the device's allocations once its block-maker thread made its internals, one
    /// import is made, and the next is `Limit::MemoryAllocations` before the driver is called, its
    /// release dropped unrun.
    #[test]
    fn every_device_counts_imports_against_its_allocations() {
        const MAX: u32 = INTERNALS + 3;
        let instance = crate::tests::validation::instance(&[]);
        let devices = devices_of_the_cfgs_kind(&instance);
        assert!(!devices.is_empty(), "no device of this cfg's kind");
        for (name, physical) in &devices {
            let device = Device::open(physical, DeviceOptions::default(), |limits| {
                limits.max_memory_allocation_count = MAX;
            })
            .unwrap_or_else(|error| panic!("{name}: {error}"));
            let counters = device.counters();
            let Some((_, alignment)) = device.core().imports.host.as_ref() else {
                println!("import: {name}: imports no host memory");
                continue;
            };
            let alignment = *alignment;
            device.core().pool.await_internals();
            assert_eq!(
                counters.gauges().blocks,
                u64::from(MAX - 1),
                "{name}: the first allocation of each class and the internals"
            );
            let len = (64u64 << 10).next_multiple_of(alignment);
            let (first, first_mapping) = host_memory(len, alignment);
            let (second, second_mapping) = host_memory(len, alignment);
            let (release, first_released) = releasing();
            // SAFETY: the memory is the test's, mapped until it unmaps it after the release.
            let (block, _) = unsafe { device.import_host_memory(first, len, release) }
                .unwrap_or_else(|error| panic!("{name}: the first import: {error}"));
            assert_eq!(counters.gauges().blocks, u64::from(MAX), "{name}");
            let (release, second_released) = releasing();
            // SAFETY: as above.
            match unsafe { device.import_host_memory(second, len, release) } {
                Err(Error::Limit(Limit::MemoryAllocations { max: MAX })) => {}
                other => panic!("{name}: an import past the count: {other:?}"),
            }
            assert_eq!(counters.gauges().blocks, u64::from(MAX), "{name}");
            drop(block);
            wait_until(&device, "the release", || {
                first_released.load(Ordering::SeqCst)
            });
            assert!(!second_released.load(Ordering::SeqCst), "{name}");
            drop(device);
            free(first_mapping);
            free(second_mapping);
            println!("import: {name}: an import past the count refused");
        }
    }

    /// A dma-buf of the first `len` bytes of a memfd of that length, through `/dev/udmabuf`, and
    /// the memfd; `None` where the kernel offers no `/dev/udmabuf` to this process.
    #[cfg(target_os = "linux")]
    fn dma_buf(len: u64) -> Option<(std::os::fd::OwnedFd, std::os::fd::OwnedFd)> {
        let memfd = memfd(len);
        Some((udmabuf(&memfd, 0, len)?, memfd))
    }

    /// A memfd of `len` bytes, sealed against shrinking, as `/dev/udmabuf` takes one.
    #[cfg(target_os = "linux")]
    pub(crate) fn memfd(len: u64) -> std::os::fd::OwnedFd {
        use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
        // SAFETY: a new memfd, owned at once.
        let memfd = unsafe {
            let fd = libc::memfd_create(
                c"m2v-import".as_ptr(),
                libc::MFD_ALLOW_SEALING | libc::MFD_CLOEXEC,
            );
            assert!(fd >= 0, "a memfd");
            OwnedFd::from_raw_fd(fd)
        };
        // SAFETY: the memfd is ours.
        unsafe {
            assert_eq!(libc::ftruncate(memfd.as_raw_fd(), len as libc::off_t), 0);
            assert_eq!(
                libc::fcntl(memfd.as_raw_fd(), libc::F_ADD_SEALS, libc::F_SEAL_SHRINK),
                0
            );
        }
        memfd
    }

    /// A dma-buf of the `len` bytes at `offset` of `memfd`, through `/dev/udmabuf`; `None` where
    /// the kernel offers no `/dev/udmabuf` to this process.
    #[cfg(target_os = "linux")]
    pub(crate) fn udmabuf(
        memfd: &std::os::fd::OwnedFd,
        offset: u64,
        len: u64,
    ) -> Option<std::os::fd::OwnedFd> {
        use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
        #[repr(C)]
        struct Create {
            memfd: u32,
            flags: u32,
            offset: u64,
            size: u64,
        }
        // `_IOW('u', 0x42, struct udmabuf_create)`.
        const UDMABUF_CREATE: libc::c_ulong = 0x4018_7542;
        const UDMABUF_FLAGS_CLOEXEC: u32 = 1;
        let device = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open("/dev/udmabuf")
            .ok()?;
        let create = Create {
            memfd: memfd.as_raw_fd() as u32,
            flags: UDMABUF_FLAGS_CLOEXEC,
            offset,
            size: len,
        };
        // SAFETY: the ioctl reads `create`, which lives through it.
        let fd = unsafe {
            libc::ioctl(
                device.as_raw_fd(),
                UDMABUF_CREATE,
                std::ptr::from_ref(&create),
            )
        };
        assert!(fd >= 0, "a dma-buf: {}", std::io::Error::last_os_error());
        // SAFETY: the ioctl returned a new fd.
        Some(unsafe { OwnedFd::from_raw_fd(fd) })
    }

    /// Through the refused device's memory view (`each_device_under_the_view`), whose Shared type
    /// is not host-coherent, so the device holds a coherence state: a dma-buf is refused with
    /// `Limit::ImportCoherence` before any allocation, alone (`import_dma_buf`) and three behind
    /// one buffer (`import_dma_bufs`, where the device offers the granule), with nothing imported
    /// and no release run; and host memory of the same length, by its pointer
    /// (`import_host_memory`), takes the host-memory rule's answer, which the state does not key:
    /// imported into a type host-visible and host-coherent through the view, or refused with
    /// `Limit::ImportType` where the pointer's query offers only types the view reads as not
    /// coherent (the Radeon 780M under RADV offers its cached type alone). Each device prints the
    /// fd's offer and what the rule answers it without the state: a type there is one the rule
    /// took before it was keyed by the state. A device that imports no dma-buf refuses it with
    /// `Limit::NoImport`.
    #[cfg(target_os = "linux")]
    #[test]
    fn every_device_under_the_view_refuses_a_dma_buf_and_keeps_the_pointer_rule() {
        use crate::import::{dma_buf_type, host_type};
        use crate::tests::command::on_device::each_device_under_the_view;
        use std::os::fd::AsRawFd;
        const LEN: u64 = 64 << 10;
        let dma_buf_ext = vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT;
        each_device_under_the_view("import_coherence", |name, _, device, memory| {
            assert!(
                device.core().coherence.is_some(),
                "{name}: a coherence state"
            );
            let Some((fd, _memfd)) = dma_buf(LEN) else {
                println!("import_coherence: {name}: no /dev/udmabuf here; not run");
                return;
            };
            let counters = device.counters();
            let before = counters.gauges();
            let Some(fds) = device.core().imports.dma_buf.as_ref() else {
                match device.import_dma_buf(fd, LEN, None) {
                    Err(Error::Limit(Limit::NoImport {
                        handle: ImportHandle::DmaBuf,
                    })) => {}
                    other => panic!("{name}: {other:?}"),
                }
                println!("import_coherence: {name}: imports no dma-buf, refused typed");
                return;
            };
            let mut properties = vk::MemoryFdPropertiesKHR::default();
            // SAFETY: the extensions are enabled; `fd` is open, and the query takes no ownership.
            unsafe { fds.get_memory_fd_properties(dma_buf_ext, fd.as_raw_fd(), &mut properties) }
                .unwrap_or_else(|error| panic!("{name}: the fd's query: {error}"));
            let offered = properties.memory_type_bits;
            let listed = crate::memory::import_requirements(device.core(), LEN, dma_buf_ext)
                .memory_type_bits;
            let coherent = device.core().imports.coherent;
            let unkeyed = dma_buf_type(ImportHandle::DmaBuf, true, offered, listed, coherent);
            let (release, released) = releasing();
            match device.import_dma_buf(fd, LEN, release) {
                Err(Error::Limit(Limit::ImportCoherence {
                    handle: ImportHandle::DmaBuf,
                })) => {}
                other => panic!("{name}: one dma-buf: {other:?}"),
            }
            assert_eq!(counters.gauges(), before, "{name}: nothing imported");
            assert!(!released.load(Ordering::SeqCst), "{name}: no release run");
            let several = match device.dma_buf_granule() {
                Some(granule) => {
                    let memfd = memfd(3 * granule);
                    let chunks = (0..3)
                        .map(|at| udmabuf(&memfd, at * granule, granule).map(|fd| (fd, granule)))
                        .collect::<Option<Vec<_>>>()
                        .expect("three dma-bufs");
                    let (release, released) = releasing();
                    match device.import_dma_bufs(chunks, release) {
                        Err(Error::Limit(Limit::ImportCoherence {
                            handle: ImportHandle::DmaBufs,
                        })) => {}
                        other => panic!("{name}: three dma-bufs: {other:?}"),
                    }
                    assert_eq!(counters.gauges(), before, "{name}: nothing imported");
                    assert!(!released.load(Ordering::SeqCst), "{name}: no release run");
                    format!("three dma-bufs of {granule} bytes refused alike")
                }
                None => "no granule, so no several dma-bufs".to_string(),
            };
            println!(
                "import_coherence: {name}: the fd's query offers {offered:#x} (the buffer lists \
                 {listed:#x}), which the rule without the state answers {unkeyed:?}; one \
                 dma-buf refused for the Shared type's coherence, nothing imported; {several}"
            );
            let Some(&(_, alignment)) = device.core().imports.host.as_ref() else {
                println!("import_coherence: {name}: imports no host memory");
                return;
            };
            let len = LEN.next_multiple_of(alignment);
            let (pointer, mapping) = host_memory(len, alignment);
            let (release, host_released) = releasing();
            // SAFETY: the memory is the test's, mapped until it unmaps it below.
            match unsafe { device.import_host_memory(pointer, len, release) } {
                Ok((block, memory_type)) => {
                    let flags = memory.memory_types[memory_type as usize].property_flags;
                    let wanted = vk::MemoryPropertyFlags::HOST_VISIBLE
                        | vk::MemoryPropertyFlags::HOST_COHERENT;
                    assert!(
                        flags.contains(wanted),
                        "{name}: type {memory_type}, flags {:#x}",
                        flags.as_raw()
                    );
                    drop(block);
                    wait_until(&device, "the release", || {
                        host_released.load(Ordering::SeqCst)
                    });
                    println!(
                        "import_coherence: {name}: host memory of {len} bytes imported into \
                         type {memory_type}"
                    );
                }
                Err(Error::Limit(Limit::ImportType {
                    handle: ImportHandle::HostMemory,
                    offered,
                    listed,
                })) => {
                    assert_eq!(host_type(offered, listed, coherent), None, "{name}");
                    assert_eq!(counters.gauges(), before, "{name}: nothing imported");
                    println!(
                        "import_coherence: {name}: host memory refused by the host-memory rule: \
                         the pointer's query offers {offered:#x}, none host-coherent through \
                         the view"
                    );
                }
                Err(error) => panic!("{name}: host memory: {error}"),
            }
            free(mapping);
        });
    }

    /// A dma-buf is imported where the first type its fd's query offers is host-visible and
    /// host-coherent, and a GPU fill reaches its pages; elsewhere it is refused at the query with
    /// nothing imported, and an ordinary Shared allocation and a host-memory import still
    /// succeed after it. A device that does not import a dma-buf refuses it typed, and so does a
    /// device whose Shared type is not host-coherent (under `m2v_shared_view`).
    #[cfg(target_os = "linux")]
    #[test]
    fn every_device_imports_a_dma_buf_only_into_its_first_offered_type() {
        use crate::import::dma_buf_type;
        each_device("import_dma_buf", |name, _, device| {
            const LEN: u64 = 64 << 10;
            let Some((fd, memfd)) = dma_buf(LEN) else {
                println!("import: {name}: no /dev/udmabuf here; the dma-buf import not run");
                return;
            };
            let counters = device.counters();
            let before = counters.gauges();
            if device.core().imports.dma_buf.is_none() {
                match device.import_dma_buf(fd, LEN, None) {
                    Err(Error::Limit(Limit::NoImport {
                        handle: ImportHandle::DmaBuf,
                    })) => {}
                    other => panic!("{name}: {other:?}"),
                }
                println!("import: {name}: imports no dma-buf, refused typed");
                return;
            }
            let (release, released) = releasing();
            match device.import_dma_buf(fd, LEN, release) {
                Ok((block, memory_type)) => {
                    assert_eq!(
                        (block.class, block.contents, block.len, block.offset),
                        (MemoryClass::Shared, None, LEN, 0),
                        "{name}"
                    );
                    let buffer = (device.new_buffer(block.id, 0, LEN, None)).expect("a buffer");
                    let queue = device.new_command_queue(4).expect("a queue");
                    let command_buffer = queue.command_buffer().expect("a command buffer");
                    let mut encoder = command_buffer.blit_command_encoder().expect("an encoder");
                    encoder
                        .fill_buffer(&buffer, 0..LEN, 0x3c)
                        .unwrap_or_else(|error| panic!("{name}: {error}"));
                    encoder.end_encoding();
                    command_buffer.commit().expect("a commit");
                    finished(name, &command_buffer);
                    assert_eq!(command_buffer.error(), None, "{name}");
                    let mut pages = vec![0u8; LEN as usize];
                    let read = std::os::unix::fs::FileExt::read_exact_at(
                        &std::fs::File::from(memfd),
                        &mut pages,
                        0,
                    );
                    read.expect("the memfd's bytes");
                    assert!(pages.iter().all(|&byte| byte == 0x3c), "{name}: the pages");
                    drop((command_buffer, queue, buffer, block));
                    wait_until(&device, "the release", || released.load(Ordering::SeqCst));
                    assert_eq!(counters.gauges().imports, 0, "{name}");
                    println!(
                        "import: {name}: a dma-buf imported into its first offered type, \
                         {memory_type}"
                    );
                }
                Err(Error::Limit(Limit::ImportCoherence {
                    handle: ImportHandle::DmaBuf,
                })) => {
                    assert!(device.core().coherence.is_some(), "{name}");
                    assert_eq!(counters.gauges(), before, "{name}: nothing imported");
                    assert!(!released.load(Ordering::SeqCst), "{name}");
                    println!("import: {name}: a dma-buf refused for the Shared type's coherence");
                }
                Err(Error::Limit(Limit::ImportType {
                    handle: ImportHandle::DmaBuf,
                    offered,
                    listed,
                })) => {
                    let coherent = device.core().imports.coherent;
                    let handle = ImportHandle::DmaBuf;
                    let rule = dma_buf_type(handle, true, offered, listed, coherent);
                    assert!(rule.is_err(), "{name}");
                    assert!(device.core().coherence.is_none(), "{name}");
                    assert_eq!(counters.gauges(), before, "{name}: nothing imported");
                    assert!(!released.load(Ordering::SeqCst), "{name}");
                    let shared = (device.take_block(MemoryClass::Shared, 32 << 20))
                        .unwrap_or_else(|error| panic!("{name}: a Shared allocation: {error}"));
                    drop(shared);
                    if let Some((_, alignment)) = device.core().imports.host.as_ref() {
                        let len = LEN.next_multiple_of(*alignment);
                        let (pointer, mapping) = host_memory(len, *alignment);
                        let (release, host_released) = releasing();
                        // SAFETY: the memory is the test's, mapped until it unmaps it below.
                        let (block, _) =
                            unsafe { device.import_host_memory(pointer, len, release) }
                                .unwrap_or_else(|error| panic!("{name}: host memory: {error}"));
                        drop(block);
                        wait_until(&device, "the release", || {
                            host_released.load(Ordering::SeqCst)
                        });
                        free(mapping);
                    }
                    println!(
                        "import: {name}: a dma-buf refused at the query (offered {offered:#x}, \
                         listed {listed:#x}); a Shared allocation and host memory after it"
                    );
                }
                Err(other) => panic!("{name}: {other}"),
            }
        });
    }
}
