//! The time of [`crate::Device::import_host_memory`], under a device cfg on every device of the
//! cfg's kind that imports host memory, and of `Device::import_dma_bufs` (Linux) on every one
//! that binds dma-bufs sparse: the whole call, and each Vulkan call and lock it makes
//! (`import.rs`, `memory::import`, `memory::sparse_buffer`), made by hand in its order beside it,
//! with the runtime's queue idle and with another thread committing small work all along.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::device::lock;
    #[cfg(target_os = "linux")]
    use crate::import::dma_buf_type;
    use crate::memory::import_requirements;
    use crate::tests::command::on_device::{finished, on_each_device};
    #[cfg(target_os = "linux")]
    use crate::tests::import::on_device::{memfd, udmabuf};
    use crate::tests::import::on_device::{releasing, wait_until};
    use crate::{Device, MemoryClass};
    use ash::vk;
    use std::alloc::Layout;
    use std::collections::VecDeque;
    #[cfg(target_os = "linux")]
    use std::os::fd::OwnedFd;
    use std::ptr::NonNull;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::thread;
    use std::time::Instant;

    /// The call's steps, in its order: the pointer's query, the buffer's requirements, the
    /// buffer, the import, the bind, the pool's lock for the block's numbers, the buffer's
    /// address, and the device's lock of its claims. It takes no queue and binds nothing sparse.
    const PARTS: [&str; 8] = [
        "vkGetMemoryHostPointerPropertiesEXT",
        "vkGetDeviceBufferMemoryRequirements",
        "vkCreateBuffer",
        "vkAllocateMemory (the host pointer's import)",
        "vkBindBufferMemory",
        "the pool's lock (the block's numbers)",
        "vkGetBufferDeviceAddress",
        "the claims' lock",
    ];

    /// The imports timed each way, of each length, in each arm.
    const REPS: usize = 40;

    /// `len` bytes of host memory of the test's own at a multiple of `alignment`, every page
    /// written: from the allocator, or (`shared`) a shared mapping of a memfd, as a virtual
    /// machine's RAM is mapped. Freed as it drops.
    struct Pages {
        pointer: NonNull<u8>,
        len: usize,
        /// The allocator's layout; `None` for a mapping.
        layout: Option<Layout>,
    }

    impl Pages {
        fn new(len: u64, alignment: u64, shared: bool) -> Pages {
            let (pointer, layout) = if shared {
                (map_shared(len), None)
            } else {
                let layout =
                    Layout::from_size_align(len as usize, alignment as usize).expect("a layout");
                // SAFETY: the layout's size is not zero.
                let pointer =
                    NonNull::new(unsafe { std::alloc::alloc(layout) }).expect("an allocation");
                (pointer, Some(layout))
            };
            for at in (0..len as usize).step_by(4096) {
                // SAFETY: `at` is inside the memory.
                unsafe { pointer.add(at).write(at as u8) };
            }
            Pages {
                pointer,
                len: len as usize,
                layout,
            }
        }
    }

    impl Drop for Pages {
        fn drop(&mut self) {
            match self.layout {
                // SAFETY: allocated with `layout`, and nothing uses it any more.
                Some(layout) => unsafe { std::alloc::dealloc(self.pointer.as_ptr(), layout) },
                None => {
                    // SAFETY: mapped by `map_shared` at `len` bytes, and nothing uses it any more.
                    let unmapped = unsafe { libc::munmap(self.pointer.as_ptr().cast(), self.len) };
                    assert_eq!(unmapped, 0, "the mapping's end");
                }
            }
        }
    }

    /// A shared mapping of a new memfd of `len` bytes.
    #[cfg(target_os = "linux")]
    fn map_shared(len: u64) -> NonNull<u8> {
        use std::os::fd::AsRawFd;
        let memfd = memfd(len);
        // SAFETY: a new mapping of the memfd's `len` bytes, which outlives the fd.
        let at = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len as usize,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                memfd.as_raw_fd(),
                0,
            )
        };
        assert_ne!(at, libc::MAP_FAILED, "a mapping");
        NonNull::new(at.cast()).expect("a mapping")
    }

    /// Off Linux the test maps no memfd: its kinds there are the allocator's alone.
    #[cfg(not(target_os = "linux"))]
    fn map_shared(_: u64) -> NonNull<u8> {
        unreachable!("a memfd off Linux")
    }

    /// The memory kinds timed: the allocator's, and on Linux a shared memfd mapping.
    fn kinds() -> &'static [(&'static str, bool)] {
        if cfg!(target_os = "linux") {
            &[("anonymous", false), ("a shared memfd mapping", true)]
        } else {
            &[("anonymous", false)]
        }
    }

    /// The whole call's time over `len` bytes, in nanoseconds, and the memory type it took; the
    /// block dropped and its release waited for outside the time. Err where the call refuses
    /// the memory, which is the test's again at once.
    fn whole(
        device: &Device,
        len: u64,
        alignment: u64,
        shared: bool,
    ) -> Result<(u128, u32), crate::Error> {
        let pages = Pages::new(len, alignment, shared);
        let (release, released) = releasing();
        let start = Instant::now();
        // SAFETY: the memory is the test's, mapped until it frees it after the release.
        let made = unsafe { device.import_host_memory(pages.pointer, len, release) };
        let took = start.elapsed().as_nanos();
        let (block, index) = made?;
        drop(block);
        wait_until("the release", || released.load(Ordering::SeqCst));
        drop(pages);
        Ok((took, index))
    }

    /// Each of the call's steps' time over `len` bytes, in nanoseconds, made by hand into memory
    /// type `index`; what they made is destroyed outside the times.
    fn parts(
        device: &Device,
        len: u64,
        alignment: u64,
        shared: bool,
        index: u32,
    ) -> [u128; PARTS.len()] {
        let core = device.core();
        let ash = &core.device;
        let (host, _) = (core.imports.host.as_ref()).expect("host memory imports");
        let handle = vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT;
        let pages = Pages::new(len, alignment, shared);
        let pointer = pages.pointer;
        let mut times = [0u128; PARTS.len()];
        let mut properties = vk::MemoryHostPointerPropertiesEXT::default();
        let start = Instant::now();
        // SAFETY: the extension is enabled; the pointer is the test's host memory.
        unsafe {
            (host.fp().get_memory_host_pointer_properties_ext)(
                host.device(),
                handle,
                pointer.as_ptr().cast_const().cast(),
                &mut properties,
            )
        }
        .result()
        .expect("the pointer's query");
        times[0] = start.elapsed().as_nanos();
        let start = Instant::now();
        let needs = import_requirements(core, len, handle);
        times[1] = start.elapsed().as_nanos();
        assert_ne!(
            properties.memory_type_bits & needs.memory_type_bits & (1 << index),
            0,
            "the call's type is offered and listed"
        );
        let mut external = vk::ExternalMemoryBufferCreateInfo::default().handle_types(handle);
        let info = vk::BufferCreateInfo::default()
            .size(len)
            .usage(crate::floor::BUFFER_USAGE)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .push_next(&mut external);
        let start = Instant::now();
        // SAFETY: the call's own create info: `len` is at most `maxBufferSize`, and the device's
        // buffers import host memory.
        let buffer = unsafe { ash.create_buffer(&info, None) }.expect("a buffer");
        times[2] = start.elapsed().as_nanos();
        let mut import = vk::ImportMemoryHostPointerInfoEXT::default()
            .handle_type(handle)
            .host_pointer(pointer.as_ptr().cast());
        let mut flags =
            vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        let allocate = vk::MemoryAllocateInfo::default()
            .allocation_size(len)
            .memory_type_index(index)
            .push_next(&mut import)
            .push_next(&mut flags);
        let start = Instant::now();
        // SAFETY: an import of the test's memory, at the device's alignment, into a type its
        // query offered and its buffers list.
        let memory = unsafe { ash.allocate_memory(&allocate, None) }.expect("an import");
        times[3] = start.elapsed().as_nanos();
        let start = Instant::now();
        // SAFETY: the memory is of a type the buffer lists, of its size, and bound to nothing.
        unsafe { ash.bind_buffer_memory(buffer, memory, 0) }.expect("a bind");
        times[4] = start.elapsed().as_nanos();
        let start = Instant::now();
        let _numbers = core.pool.own_numbers().expect("the block's numbers");
        times[5] = start.elapsed().as_nanos();
        let address = vk::BufferDeviceAddressInfo::default().buffer(buffer);
        let start = Instant::now();
        // SAFETY: the buffer has device-address usage and is bound.
        let address = unsafe { ash.get_buffer_device_address(&address) };
        times[6] = start.elapsed().as_nanos();
        let start = Instant::now();
        drop(lock(&core.claims));
        times[7] = start.elapsed().as_nanos();
        assert_ne!(address, 0, "a bound buffer's address");
        // SAFETY: nothing uses the buffer or the memory.
        unsafe {
            ash.destroy_buffer(buffer, None);
            ash.free_memory(memory, None);
        }
        // Its import is freed: nothing uses it.
        drop(pages);
        times
    }

    /// Commits a 4-byte fill on a queue of its own, again and again with at most 8 in flight,
    /// until `stop`: the runtime's submit thread busy beside the imports. The commits made.
    fn busy(device: Device, stop: Arc<AtomicBool>) -> thread::JoinHandle<u64> {
        thread::spawn(move || {
            let block = (device.take_block(MemoryClass::Shared, 4096)).expect("a Shared block");
            let buffer = (device.new_buffer(block.id, 0, 4096, None)).expect("a buffer");
            let queue = device.new_command_queue(16).expect("a queue");
            let mut flight = VecDeque::new();
            let mut commits = 0;
            while !stop.load(Ordering::Relaxed) {
                let command_buffer = queue.command_buffer().expect("a command buffer");
                let mut encoder = command_buffer.blit_command_encoder().expect("an encoder");
                encoder.fill_buffer(&buffer, 0..4, 7).expect("a fill");
                encoder.end_encoding();
                command_buffer.commit().expect("a commit");
                commits += 1;
                flight.push_back(command_buffer);
                if flight.len() > 8 {
                    let first = flight.pop_front().expect("one in flight");
                    finished("the busy thread", &first);
                }
            }
            for command_buffer in &flight {
                finished("the busy thread", command_buffer);
            }
            commits
        })
    }

    /// The median and the longest of `nanos`, in milliseconds.
    fn shown(nanos: &[u128]) -> String {
        let mut sorted = nanos.to_vec();
        sorted.sort_unstable();
        let ms = |nanos: u128| nanos as f64 / 1e6;
        format!(
            "median {:.3} ms, max {:.3} ms",
            ms(sorted[sorted.len() / 2]),
            ms(sorted[sorted.len() - 1])
        )
    }

    /// The steps of `Device::import_dma_bufs`, in its order: the sparse buffer with its memory
    /// requirements, each fd's query, the pool's lock for the block's numbers, each chunk's
    /// import, the wait for the runtime queue's lock, the sparse bind under it, and the device's
    /// lock of its claims. The rest of the call (the sparse space's reservation, the block's
    /// record) is the whole less these.
    #[cfg(target_os = "linux")]
    const DMA_PARTS: [&str; 7] = [
        "the sparse buffer (vkCreateBuffer, vkGetBufferMemoryRequirements)",
        "the fds' queries (vkGetMemoryFdPropertiesKHR, one a chunk)",
        "the pool's lock (the block's numbers)",
        "the chunks' imports (vkAllocateMemory, one a chunk)",
        "the wait for the runtime queue's lock",
        "the bind (vkQueueBindSparse)",
        "the claims' lock",
    ];

    /// Dma-bufs of `memfd`'s consecutive ranges of `lens`, with their lengths.
    #[cfg(target_os = "linux")]
    fn dma_chunks(memfd: &OwnedFd, lens: &[u64]) -> Vec<(OwnedFd, u64)> {
        let mut at = 0;
        (lens.iter())
            .map(|&len| {
                let fd = udmabuf(memfd, at, len).expect("a dma-buf");
                at += len;
                (fd, len)
            })
            .collect()
    }

    /// The whole call's time over dma-bufs of `memfd`'s ranges of `lens`, in nanoseconds; the
    /// dma-bufs made before it, and the block dropped and its release waited for after it.
    #[cfg(target_os = "linux")]
    fn dma_whole(device: &Device, memfd: &OwnedFd, lens: &[u64]) -> u128 {
        let chunks = dma_chunks(memfd, lens);
        let (release, released) = releasing();
        let start = Instant::now();
        let made = device.import_dma_bufs(chunks, release);
        let took = start.elapsed().as_nanos();
        let (block, _) = made.unwrap_or_else(|error| panic!("an import: {error}"));
        drop(block);
        wait_until("the release", || released.load(Ordering::SeqCst));
        took
    }

    /// Each of the call's steps' time over dma-bufs of `memfd`'s ranges of `lens`, in
    /// nanoseconds, made by hand: the bind signals a timeline of the test's own, waited for
    /// before what the steps made is destroyed, all outside the times.
    #[cfg(target_os = "linux")]
    fn dma_parts(device: &Device, memfd: &OwnedFd, lens: &[u64]) -> [u128; DMA_PARTS.len()] {
        use std::os::fd::{AsRawFd, IntoRawFd};
        let core = device.core();
        let ash = &core.device;
        let fds = (core.imports.dma_buf.as_ref()).expect("dma-buf imports");
        let handle = vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT;
        let chunks = dma_chunks(memfd, lens);
        let len: u64 = lens.iter().sum();
        let timeline = crate::event::timeline(ash).expect("a timeline");
        let mut times = [0u128; DMA_PARTS.len()];
        let start = Instant::now();
        // SAFETY: the device binds dma-bufs sparse (it has a granule), and `len` is a few
        // granules, under its `maxBufferSize`.
        let (buffer, needs) =
            unsafe { crate::memory::sparse_buffer(core, len) }.expect("a sparse buffer");
        times[0] = start.elapsed().as_nanos();
        let start = Instant::now();
        let mut index = None;
        for (fd, _) in &chunks {
            let mut properties = vk::MemoryFdPropertiesKHR::default();
            // SAFETY: the extensions are enabled; `fd` is open, and the query takes no
            // ownership.
            unsafe { fds.get_memory_fd_properties(handle, fd.as_raw_fd(), &mut properties) }
                .expect("the fd's query");
            let listed = index.map_or(needs.memory_type_bits, |first: u32| {
                needs.memory_type_bits & (1 << first)
            });
            let taken = dma_buf_type(properties.memory_type_bits, listed, core.imports.coherent);
            index = Some(taken.expect("the call's type"));
        }
        times[1] = start.elapsed().as_nanos();
        let index = index.expect("a chunk");
        let start = Instant::now();
        let _numbers = core.pool.own_numbers().expect("the block's numbers");
        times[2] = start.elapsed().as_nanos();
        let start = Instant::now();
        let mut made = Vec::with_capacity(chunks.len());
        for (fd, chunk) in chunks {
            let mut import = vk::ImportMemoryFdInfoKHR::default()
                .handle_type(handle)
                .fd(fd.into_raw_fd());
            let mut flags = vk::MemoryAllocateFlagsInfo::default()
                .flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
            let info = vk::MemoryAllocateInfo::default()
                .allocation_size(chunk)
                .memory_type_index(index)
                .push_next(&mut import)
                .push_next(&mut flags);
            // SAFETY: an import of the dma-buf into a type its query offered and the sparse
            // buffer takes; the import owns the fd.
            let memory = unsafe { crate::memory::import_chunk(core, &info) }.expect("an import");
            made.push((memory, chunk));
        }
        times[3] = start.elapsed().as_nanos();
        let mut offset = 0;
        let binds: Vec<vk::SparseMemoryBind> = (made.iter())
            .map(|&(memory, size)| {
                let bind = vk::SparseMemoryBind::default()
                    .resource_offset(offset)
                    .size(size)
                    .memory(memory);
                offset += size;
                bind
            })
            .collect();
        let buffer_binds = [vk::SparseBufferMemoryBindInfo::default()
            .buffer(buffer)
            .binds(&binds)];
        let signals = [timeline];
        let values = [1u64];
        let mut timeline_info =
            vk::TimelineSemaphoreSubmitInfo::default().signal_semaphore_values(&values);
        let info = vk::BindSparseInfo::default()
            .buffer_binds(&buffer_binds)
            .signal_semaphores(&signals)
            .push_next(&mut timeline_info);
        let start = Instant::now();
        let queue = lock(&core.queues.runtime);
        times[4] = start.elapsed().as_nanos();
        let start = Instant::now();
        // SAFETY: the runtime's queue binds sparse memory and is held; each bind lies in the new
        // buffer at a multiple of its alignment, from the first byte of an import of its size,
        // of a type the buffer takes; the timeline is new, at 0.
        unsafe { ash.queue_bind_sparse(*queue, &[info], vk::Fence::null()) }.expect("the bind");
        times[5] = start.elapsed().as_nanos();
        drop(queue);
        let start = Instant::now();
        drop(lock(&core.claims));
        times[6] = start.elapsed().as_nanos();
        let semaphores = [timeline];
        let wait = vk::SemaphoreWaitInfo::default()
            .semaphores(&semaphores)
            .values(&values);
        // SAFETY: the timeline is live, and the bind signals it.
        unsafe { ash.wait_semaphores(&wait, 10_000_000_000) }.expect("the bind's end");
        // SAFETY: the bind has ended, and nothing uses the buffer, its imports or the timeline;
        // once the buffer is gone nothing is bound to the imports.
        unsafe {
            ash.destroy_buffer(buffer, None);
            for (memory, bytes) in made {
                crate::memory::free_chunk(core, memory, bytes);
            }
            ash.destroy_semaphore(timeline, None);
        }
        times
    }

    /// The median of `nanos`.
    #[cfg(target_os = "linux")]
    fn median(nanos: &[u128]) -> u128 {
        let mut sorted = nanos.to_vec();
        sorted.sort_unstable();
        sorted[sorted.len() / 2]
    }

    /// `Device::import_dma_bufs` of 8 dma-bufs over 29 MiB of one memfd, each a whole number of
    /// granules, whole and by step, each `REPS` times, with the runtime's queue idle and then
    /// beside a thread committing small fills all along: each one's median and longest, and the
    /// whole's median less the steps' summed median, printed. On an instance without the
    /// validation layer, whose own cost would be the count.
    #[cfg(target_os = "linux")]
    #[test]
    fn every_device_times_a_dma_buf_import_by_part() {
        let instance = crate::tests::validation::bare(&[]);
        on_each_device("import_dma_bufs_time", &instance, |name, _, device| {
            let Some(granule) = device.dma_buf_granule() else {
                println!("dma-buf import time: {name}: binds no dma-buf sparse; not timed");
                return;
            };
            let chunk = ((29u64 << 20) / 8).next_multiple_of(granule);
            let lens = [chunk; 8];
            let memfd = memfd(chunk * 8);
            if udmabuf(&memfd, 0, granule).is_none() {
                println!("dma-buf import time: {name}: no /dev/udmabuf here; not timed");
                return;
            }
            for arm in ["idle", "busy"] {
                let stop = Arc::new(AtomicBool::new(false));
                let worker = (arm == "busy").then(|| busy(device.clone(), Arc::clone(&stop)));
                let mut wholes = Vec::with_capacity(REPS);
                let mut by_part = vec![Vec::with_capacity(REPS); DMA_PARTS.len()];
                for _ in 0..REPS {
                    wholes.push(dma_whole(&device, &memfd, &lens));
                    let steps = dma_parts(&device, &memfd, &lens);
                    for (times, took) in by_part.iter_mut().zip(steps) {
                        times.push(took);
                    }
                }
                stop.store(true, Ordering::Relaxed);
                let commits = worker.map_or(0, |worker| worker.join().expect("the busy thread"));
                let sums: Vec<u128> = (0..REPS)
                    .map(|rep| by_part.iter().map(|times| times[rep]).sum())
                    .collect();
                let rest = median(&wholes).saturating_sub(median(&sums));
                println!(
                    "dma-buf import time: {name}: 8 chunks of {chunk} bytes, granule {granule}, \
                     {arm} ({commits} commits beside): the whole call {}; its steps by hand \
                     summed {}; the rest (the whole's median less the steps') {:.3} ms",
                    shown(&wholes),
                    shown(&sums),
                    rest as f64 / 1e6
                );
                for (part, times) in DMA_PARTS.iter().zip(&by_part) {
                    println!(
                        "dma-buf import time: {name}: {arm}: {part} {}",
                        shown(times)
                    );
                }
            }
        });
    }

    /// `Device::import_host_memory` of 12 KiB, 16 KiB and 29 MiB of host memory of each kind
    /// (anonymous, and on Linux a shared memfd mapping, unless the device refuses it), whole and by
    /// step, each `REPS` times, with the runtime's queue idle and then beside a thread committing
    /// small fills all along: each one's median and longest, printed. On an instance without the
    /// validation layer, whose own cost would be the count.
    #[test]
    fn every_device_times_a_host_memory_import_by_part() {
        let instance = crate::tests::validation::bare(&[]);
        on_each_device("import_host_memory_time", &instance, |name, _, device| {
            let Some(alignment) = (device.core().imports.host.as_ref()).map(|(_, at)| *at) else {
                println!("import time: {name}: imports no host memory; not timed");
                return;
            };
            for &(kind, shared) in kinds() {
                if let Err(error) = whole(&device, 29 << 20, alignment, shared) {
                    println!("import time: {name}: {kind}: refused: {error}; not timed");
                    continue;
                }
                for len in [12u64 << 10, 16 << 10, 29 << 20] {
                    let len = len.next_multiple_of(alignment);
                    for arm in ["idle", "busy"] {
                        let stop = Arc::new(AtomicBool::new(false));
                        let worker =
                            (arm == "busy").then(|| busy(device.clone(), Arc::clone(&stop)));
                        let mut wholes = Vec::with_capacity(REPS);
                        let mut by_part = vec![Vec::with_capacity(REPS); PARTS.len()];
                        for _ in 0..REPS {
                            let (took, index) = whole(&device, len, alignment, shared)
                                .unwrap_or_else(|error| panic!("an import: {error}"));
                            wholes.push(took);
                            let steps = parts(&device, len, alignment, shared, index);
                            for (times, took) in by_part.iter_mut().zip(steps) {
                                times.push(took);
                            }
                        }
                        stop.store(true, Ordering::Relaxed);
                        let commits =
                            worker.map_or(0, |worker| worker.join().expect("the busy thread"));
                        let sums: Vec<u128> = (0..REPS)
                            .map(|rep| by_part.iter().map(|times| times[rep]).sum())
                            .collect();
                        println!(
                            "import time: {name}: {kind}, {len} bytes, {arm} ({commits} commits \
                             beside): the whole call {}; its steps by hand summed {}",
                            shown(&wholes),
                            shown(&sums)
                        );
                        for (part, times) in PARTS.iter().zip(&by_part) {
                            println!(
                                "import time: {name}: {kind}, {len} bytes, {arm}: {part} {}",
                                shown(times)
                            );
                        }
                    }
                }
            }
        });
    }
}
