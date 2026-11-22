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

    /// The kinds of host memory timed.
    #[derive(Clone, Copy, PartialEq)]
    enum Kind {
        /// The allocator's.
        Anonymous,
        /// One shared mapping of a memfd, as a virtual machine's RAM is mapped (Linux).
        Shared,
        /// The same bytes as one-page shared mappings of a memfd of twice their length, each at a
        /// page of the memfd of its own scattered over it: as many mappings as pages (Linux).
        Scattered,
        /// As a virtual machine's RAM is viewed (Linux): a memfd of twice the length, every page
        /// written first and sealed against growing, shrinking and further seals, and one-page
        /// runs of its scattered pages each mapped shared and fixed over an anonymous private
        /// reservation made without a reserve.
        Guest,
    }

    /// The memory kinds timed, with their names: the allocator's, and on Linux one shared memfd
    /// mapping, one-page mappings of scattered pages of a memfd, and a guest RAM view.
    fn kinds() -> &'static [(&'static str, Kind)] {
        if cfg!(target_os = "linux") {
            &[
                ("anonymous", Kind::Anonymous),
                ("one shared memfd mapping", Kind::Shared),
                ("one-page memfd mappings, scattered", Kind::Scattered),
                (
                    "a guest RAM view: one-page runs over a reservation",
                    Kind::Guest,
                ),
            ]
        } else {
            &[("anonymous", Kind::Anonymous)]
        }
    }

    /// `len` bytes of host memory of the test's own of `kind`, at a multiple of `alignment`,
    /// every page written. Freed as it drops.
    struct Pages {
        pointer: NonNull<u8>,
        len: usize,
        /// The allocator's layout; `None` for mappings.
        layout: Option<Layout>,
    }

    impl Pages {
        fn new(len: u64, alignment: u64, kind: Kind) -> Pages {
            let (pointer, layout) = if kind == Kind::Anonymous {
                let layout =
                    Layout::from_size_align(len as usize, alignment as usize).expect("a layout");
                // SAFETY: the layout's size is not zero.
                let pointer =
                    NonNull::new(unsafe { std::alloc::alloc(layout) }).expect("an allocation");
                (pointer, Some(layout))
            } else {
                (map_shared(len, kind), None)
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
                None => unmap_shared(self.pointer, self.len),
            }
        }
    }

    /// The end of `len` bytes of mappings `map_shared` made at `pointer`, which nothing uses any
    /// more.
    #[cfg(target_os = "linux")]
    fn unmap_shared(pointer: NonNull<u8>, len: usize) {
        // SAFETY: mapped by `map_shared` over `len` bytes, and nothing uses it any more.
        let unmapped = unsafe { libc::munmap(pointer.as_ptr().cast(), len) };
        assert_eq!(unmapped, 0, "the mappings' end");
    }

    /// Off Linux the test maps no memfd, so it ends no mapping.
    #[cfg(not(target_os = "linux"))]
    fn unmap_shared(_: NonNull<u8>, _: usize) {
        unreachable!("a memfd off Linux")
    }

    /// `len` bytes of shared mappings of a new memfd, by `kind` ([`Kind`]): one mapping of a memfd
    /// of `len` bytes; or one mapping a page, the pages of a memfd of twice `len` taken at a stride
    /// prime to its page count, so no two neighbouring mappings merge, over a shared mapping of
    /// the memfd's start (`Scattered`) or over an anonymous reservation of a memfd written and
    /// sealed first (`Guest`).
    #[cfg(target_os = "linux")]
    fn map_shared(len: u64, kind: Kind) -> NonNull<u8> {
        use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
        const PAGE: u64 = 4096;
        const STRIDE: u64 = 7919;
        let file = if kind == Kind::Shared { len } else { 2 * len };
        let memfd = if kind == Kind::Guest {
            // SAFETY: a new memfd, owned at once.
            let memfd = unsafe {
                let fd = libc::memfd_create(
                    c"m2v-guest-ram".as_ptr(),
                    libc::MFD_ALLOW_SEALING | libc::MFD_CLOEXEC,
                );
                assert!(fd >= 0, "a memfd");
                OwnedFd::from_raw_fd(fd)
            };
            // SAFETY: the memfd is ours.
            assert_eq!(
                unsafe { libc::ftruncate(memfd.as_raw_fd(), file as libc::off_t) },
                0
            );
            memfd
        } else {
            memfd(file)
        };
        let map = |at: *mut libc::c_void, bytes: u64, flags: libc::c_int, offset: u64| {
            // SAFETY: a mapping of the memfd's bytes at `offset`, inside its length, which
            // outlives the fd; where `at` is not null, it is inside the test's own reservation.
            let mapped = unsafe {
                libc::mmap(
                    at,
                    bytes as usize,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_SHARED | flags,
                    memfd.as_raw_fd(),
                    offset as libc::off_t,
                )
            };
            assert_ne!(mapped, libc::MAP_FAILED, "a mapping");
            mapped
        };
        if kind == Kind::Shared {
            return NonNull::new(map(std::ptr::null_mut(), len, 0, 0).cast()).expect("a mapping");
        }
        let base = if kind == Kind::Guest {
            // Every page of the file written first, then the file sealed.
            let whole = map(std::ptr::null_mut(), file, 0, 0);
            for at in (0..file).step_by(PAGE as usize) {
                // SAFETY: `at` is inside the whole file's mapping.
                unsafe { whole.cast::<u8>().add(at as usize).write(1) };
            }
            // SAFETY: the mapping above, of `file` bytes, which nothing uses any more.
            assert_eq!(unsafe { libc::munmap(whole, file as usize) }, 0);
            let seals = libc::F_SEAL_GROW | libc::F_SEAL_SHRINK | libc::F_SEAL_SEAL;
            // SAFETY: the memfd is ours, made with sealing allowed.
            let sealed = unsafe { libc::fcntl(memfd.as_raw_fd(), libc::F_ADD_SEALS, seals) };
            assert_eq!(sealed, 0, "the seals");
            // SAFETY: a new anonymous private reservation, made without a reserve.
            let reserved = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    len as usize,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
                    -1,
                    0,
                )
            };
            assert_ne!(reserved, libc::MAP_FAILED, "a reservation");
            reserved
        } else {
            map(std::ptr::null_mut(), len, 0, 0)
        };
        let pages = len / PAGE;
        assert_ne!(
            (2 * pages) % STRIDE,
            0,
            "the stride is prime to the memfd's pages"
        );
        for page in 0..pages {
            let from = (page * STRIDE) % (2 * pages);
            // SAFETY: the page lies inside `base`'s `len` bytes, which the test mapped above.
            let at = unsafe { base.cast::<u8>().add((page * PAGE) as usize) };
            map(at.cast(), PAGE, libc::MAP_FIXED, from * PAGE);
        }
        NonNull::new(base.cast()).expect("a mapping")
    }

    /// Off Linux the test maps no memfd: its kinds there are the allocator's alone.
    #[cfg(not(target_os = "linux"))]
    fn map_shared(_: u64, _: Kind) -> NonNull<u8> {
        unreachable!("a memfd off Linux")
    }

    /// The whole call's time over `len` bytes, in nanoseconds, and the memory type it took; the
    /// block dropped and its release waited for outside the time. Err where the call refuses
    /// the memory, which is the test's again at once.
    fn whole(
        device: &Device,
        len: u64,
        alignment: u64,
        kind: Kind,
    ) -> Result<(u128, u32), crate::Error> {
        let pages = Pages::new(len, alignment, kind);
        let (release, released) = releasing();
        let start = Instant::now();
        // SAFETY: the memory is the test's, mapped until it frees it after the release.
        let made = unsafe { device.import_host_memory(pages.pointer, len, release) };
        let took = start.elapsed().as_nanos();
        let (block, index) = made?;
        drop(block);
        wait_until(&device, "the release", || released.load(Ordering::SeqCst));
        drop(pages);
        Ok((took, index))
    }

    /// Each of the call's steps' time over `len` bytes, in nanoseconds, made by hand into memory
    /// type `index`; what they made is destroyed outside the times.
    fn parts(
        device: &Device,
        len: u64,
        alignment: u64,
        kind: Kind,
        index: u32,
    ) -> [u128; PARTS.len()] {
        let core = device.core();
        let ash = &core.device;
        let (host, _) = (core.imports.host.as_ref()).expect("host memory imports");
        let handle = vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT;
        let pages = Pages::new(len, alignment, kind);
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

    /// Commits a fill of 256 MiB of device-local memory on a queue of its own, again and again
    /// with up to 4 in flight, until `stop`: real GPU work backed up on the runtime's queue, so
    /// that its submissions wait on the GPU. Work queued behind the backlog waits at most four
    /// fills, which stays inside the 10-second waits below on a CPU device under load too; 64 in
    /// flight did not (16 GiB of fills on lavapipe). The commits made.
    #[cfg(target_os = "linux")]
    fn backlog(device: Device, stop: Arc<AtomicBool>) -> thread::JoinHandle<u64> {
        thread::spawn(move || {
            const LEN: u64 = 256 << 20;
            let block = (device.take_block(MemoryClass::Private, LEN)).expect("a Private block");
            let buffer = (device.new_buffer(block.id, 0, LEN, None)).expect("a buffer");
            let queue = device.new_command_queue(128).expect("a queue");
            let mut flight = VecDeque::new();
            let mut commits = 0;
            while !stop.load(Ordering::Relaxed) {
                let command_buffer = queue.command_buffer().expect("a command buffer");
                let mut encoder = command_buffer.blit_command_encoder().expect("an encoder");
                encoder.fill_buffer(&buffer, 0..LEN, 7).expect("a fill");
                encoder.end_encoding();
                command_buffer.commit().expect("a commit");
                commits += 1;
                flight.push_back(command_buffer);
                if flight.len() > 4 {
                    let first = flight.pop_front().expect("one in flight");
                    finished("the backlog thread", &first);
                }
            }
            for command_buffer in &flight {
                finished("the backlog thread", command_buffer);
            }
            commits
        })
    }

    /// The process's threads' OS ids and names, for a trace's lines: `/proc/self/task`.
    #[cfg(target_os = "linux")]
    fn thread_names() -> String {
        let mut names = Vec::new();
        for entry in std::fs::read_dir("/proc/self/task")
            .into_iter()
            .flatten()
            .flatten()
        {
            let comm = std::fs::read_to_string(entry.path().join("comm")).unwrap_or_default();
            names.push(format!(
                "{}={}",
                entry.file_name().to_string_lossy(),
                comm.trim()
            ));
        }
        names.sort();
        names.join(" ")
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

    /// Dma-bufs of `memfd`'s consecutive ranges of `lens`, with their lengths; or (`scattered`)
    /// each of its own one-page runs of `memfd`, a memfd of twice their sum, at a stride prime to
    /// its page count, through one `UDMABUF_CREATE_LIST` a dma-buf.
    #[cfg(target_os = "linux")]
    fn dma_chunks(memfd: &OwnedFd, lens: &[u64], scattered: bool) -> Vec<(OwnedFd, u64)> {
        const PAGE: u64 = 4096;
        const STRIDE: u64 = 7919;
        let pages = 2 * lens.iter().sum::<u64>() / PAGE;
        let mut at = 0;
        (lens.iter())
            .map(|&len| {
                let fd = if scattered {
                    let runs: Vec<(u64, u64)> = (at / PAGE..(at + len) / PAGE)
                        .map(|page| ((page * STRIDE) % pages * PAGE, PAGE))
                        .collect();
                    udmabuf_runs(memfd, &runs).expect("a dma-buf of runs")
                } else {
                    udmabuf(memfd, at, len).expect("a dma-buf")
                };
                at += len;
                (fd, len)
            })
            .collect()
    }

    /// A dma-buf of `memfd`'s `runs` (offset, length) in order, through `/dev/udmabuf`'s
    /// `UDMABUF_CREATE_LIST`; `None` where the kernel offers no `/dev/udmabuf` or no list.
    #[cfg(target_os = "linux")]
    fn udmabuf_runs(memfd: &OwnedFd, runs: &[(u64, u64)]) -> Option<OwnedFd> {
        use std::os::fd::{AsRawFd, FromRawFd};
        // `_IOW('u', 0x43, struct udmabuf_create_list)`, the list's 8-byte head.
        const UDMABUF_CREATE_LIST: libc::c_ulong = 0x4008_7543;
        const UDMABUF_FLAGS_CLOEXEC: u64 = 1;
        let device = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open("/dev/udmabuf")
            .ok()?;
        // The head (flags, count) and then each item (memfd, padding, offset, size), in u64s
        // laid out as the kernel's structs on a little-endian host.
        let mut words = vec![UDMABUF_FLAGS_CLOEXEC | ((runs.len() as u64) << 32)];
        for &(offset, size) in runs {
            words.extend([memfd.as_raw_fd() as u32 as u64, offset, size]);
        }
        // SAFETY: the ioctl reads the head and its `count` items, which `words` holds through it.
        let fd = unsafe { libc::ioctl(device.as_raw_fd(), UDMABUF_CREATE_LIST, words.as_ptr()) };
        // SAFETY: a non-negative return is a new fd.
        (fd >= 0).then(|| unsafe { OwnedFd::from_raw_fd(fd) })
    }

    /// The whole call's time over `chunks` (dma-bufs and their lengths), in nanoseconds; the
    /// block dropped and its release waited for after it.
    #[cfg(target_os = "linux")]
    fn dma_whole(device: &Device, chunks: Vec<(OwnedFd, u64)>) -> u128 {
        let (release, released) = releasing();
        let start = Instant::now();
        let made = device.import_dma_bufs(chunks, release);
        let took = start.elapsed().as_nanos();
        let (block, _) = made.unwrap_or_else(|error| panic!("an import: {error}"));
        drop(block);
        wait_until(&device, "the release", || released.load(Ordering::SeqCst));
        took
    }

    /// Each of the call's steps' time over `chunks` (dma-bufs and their lengths), in
    /// nanoseconds, made by hand: the bind signals a timeline of the test's own, waited for
    /// before what the steps made is destroyed, all outside the times.
    #[cfg(target_os = "linux")]
    fn dma_parts(device: &Device, chunks: Vec<(OwnedFd, u64)>) -> [u128; DMA_PARTS.len()] {
        use std::os::fd::{AsRawFd, IntoRawFd};
        let core = device.core();
        let ash = &core.device;
        let fds = (core.imports.dma_buf.as_ref()).expect("dma-buf imports");
        let handle = vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT;
        let len: u64 = chunks.iter().map(|&(_, len)| len).sum();
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

    /// `Device::import_dma_bufs` of 8 dma-bufs over 29 MiB of one memfd, each one run of a whole
    /// number of granules, and of 8 dma-bufs of 64 scattered one-page runs each, whole and by
    /// step, each `REPS` times, with the runtime's queue idle and then
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
            let runs = (64u64 << 12).next_multiple_of(granule);
            let probe = memfd(granule);
            if udmabuf(&probe, 0, granule).is_none() {
                println!("dma-buf import time: {name}: no /dev/udmabuf here; not timed");
                return;
            }
            let shapes = [
                (
                    format!("8 chunks of {chunk} bytes, one run each"),
                    chunk,
                    false,
                ),
                (
                    format!("8 chunks of {runs} bytes, one-page runs each, scattered"),
                    runs,
                    true,
                ),
            ];
            for (shape, chunk, scattered) in shapes {
                let lens = [chunk; 8];
                let memfd = memfd(if scattered { 16 * chunk } else { 8 * chunk });
                if scattered && udmabuf_runs(&memfd, &[(0, 4096)]).is_none() {
                    println!(
                        "dma-buf import time: {name}: {shape}: no UDMABUF_CREATE_LIST; not timed"
                    );
                    continue;
                }
                for arm in ["idle", "busy"] {
                    let stop = Arc::new(AtomicBool::new(false));
                    let worker = (arm == "busy").then(|| busy(device.clone(), Arc::clone(&stop)));
                    let mut wholes = Vec::with_capacity(REPS);
                    let mut by_part = vec![Vec::with_capacity(REPS); DMA_PARTS.len()];
                    for _ in 0..REPS {
                        wholes.push(dma_whole(&device, dma_chunks(&memfd, &lens, scattered)));
                        let steps = dma_parts(&device, dma_chunks(&memfd, &lens, scattered));
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
                    let rest = median(&wholes).saturating_sub(median(&sums));
                    println!(
                        "dma-buf import time: {name}: {shape}, granule {granule}, \
                     {arm} ({commits} commits beside): the whole call {}; its steps by hand \
                     summed {}; the rest (the whole's median less the steps') {:.3} ms",
                        shown(&wholes),
                        shown(&sums),
                        rest as f64 / 1e6
                    );
                    for (part, times) in DMA_PARTS.iter().zip(&by_part) {
                        println!(
                            "dma-buf import time: {name}: {shape}, {arm}: {part} {}",
                            shown(times)
                        );
                    }
                }
            }
        });
    }

    /// `count` dma-bufs of `chunk` bytes each over `memfd`, a memfd of twice their sum, each
    /// made by one `UDMABUF_CREATE_LIST` of `runs` runs: one or two pages each (as many of two as
    /// the chunk's pages need past `runs`), each at a two-page slot of the memfd taken at a stride
    /// prime to its slot count; one run a dma-buf is its own consecutive range. With each
    /// create's time, in nanoseconds.
    #[cfg(target_os = "linux")]
    fn view_chunks(
        memfd: &OwnedFd,
        count: u64,
        chunk: u64,
        runs: u64,
    ) -> (Vec<(OwnedFd, u64)>, Vec<u128>) {
        const PAGE: u64 = 4096;
        const STRIDE: u64 = 7919;
        let pages = chunk / PAGE;
        assert!(
            runs == 1 || (runs <= pages && pages <= 2 * runs),
            "one run, or runs of one or two pages, fill the chunk"
        );
        let slots = count * pages;
        assert_ne!(
            slots % STRIDE,
            0,
            "the stride is prime to the memfd's slots"
        );
        let mut made = Vec::new();
        let mut creates = Vec::new();
        for c in 0..count {
            let list: Vec<(u64, u64)> = if runs == 1 {
                vec![(c * chunk, chunk)]
            } else {
                (0..runs)
                    .map(|r| {
                        let slot = ((c * runs + r) * STRIDE) % slots;
                        let len = if r < pages - runs { 2 * PAGE } else { PAGE };
                        (slot * 2 * PAGE, len)
                    })
                    .collect()
            };
            let start = Instant::now();
            let fd = udmabuf_runs(memfd, &list).expect("a dma-buf of runs");
            creates.push(start.elapsed().as_nanos());
            made.push((fd, chunk));
        }
        (made, creates)
    }

    /// `Device::import_dma_bufs` of a guest RAM view's shape: 8 MiB in 2 dma-bufs, each made by
    /// one `UDMABUF_CREATE_LIST` of 1,013 scattered runs of a memfd with every page written first,
    /// against the same 8 MiB in 2 dma-bufs of one run each: whole and by step, `REPS` times with
    /// the runtime's queue idle and then backlogged with fills of 256 MiB ([`backlog`]), and each
    /// create's time apart. On an instance without the
    /// validation layer, whose own cost would be the count.
    #[cfg(target_os = "linux")]
    #[test]
    fn every_device_times_a_guest_view_dma_buf_import() {
        use std::os::fd::AsRawFd;
        let instance = crate::tests::validation::bare(&[]);
        on_each_device("import_view_dma_bufs_time", &instance, |name, _, device| {
            let Some(granule) = device.dma_buf_granule() else {
                println!("view dma-buf import time: {name}: binds no dma-buf sparse; not timed");
                return;
            };
            let chunk = (4u64 << 20).next_multiple_of(granule);
            let file = 4 * chunk;
            let memfd = memfd(file);
            if udmabuf_runs(&memfd, &[(0, 4096)]).is_none() {
                println!("view dma-buf import time: {name}: no UDMABUF_CREATE_LIST; not timed");
                return;
            }
            // Every page of the memfd written first.
            // SAFETY: a new shared mapping of the memfd's `file` bytes, unmapped below.
            let whole = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    file as usize,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_SHARED,
                    memfd.as_raw_fd(),
                    0,
                )
            };
            assert_ne!(whole, libc::MAP_FAILED, "a mapping");
            for at in (0..file).step_by(4096) {
                // SAFETY: `at` is inside the mapping.
                unsafe { whole.cast::<u8>().add(at as usize).write(1) };
            }
            // SAFETY: the mapping above, which nothing uses any more.
            assert_eq!(unsafe { libc::munmap(whole, file as usize) }, 0);
            for queue in ["idle", "backlogged"] {
                let stop = Arc::new(AtomicBool::new(false));
                let worker =
                    (queue == "backlogged").then(|| backlog(device.clone(), Arc::clone(&stop)));
                if worker.is_some() {
                    // The backlog builds before the first import.
                    thread::sleep(std::time::Duration::from_millis(500));
                    println!(
                        "view dma-buf import time: {name}: threads {}",
                        thread_names()
                    );
                }
                for runs in [1013u64, 1] {
                    let mut wholes = Vec::with_capacity(REPS);
                    let mut creates = Vec::with_capacity(4 * REPS);
                    let mut by_part = vec![Vec::with_capacity(REPS); DMA_PARTS.len()];
                    for _ in 0..REPS {
                        let (chunks, made) = view_chunks(&memfd, 2, chunk, runs);
                        creates.extend(made);
                        wholes.push(dma_whole(&device, chunks));
                        let (chunks, made) = view_chunks(&memfd, 2, chunk, runs);
                        creates.extend(made);
                        let steps = dma_parts(&device, chunks);
                        for (times, took) in by_part.iter_mut().zip(steps) {
                            times.push(took);
                        }
                    }
                    let sums: Vec<u128> = (0..REPS)
                        .map(|rep| by_part.iter().map(|times| times[rep]).sum())
                        .collect();
                    let rest = median(&wholes).saturating_sub(median(&sums));
                    let shape = format!("2 dma-bufs of {chunk} bytes, {runs} runs each, {queue}");
                    println!(
                        "view dma-buf import time: {name}: {shape}, granule {granule}: the whole \
                     call {}; its steps by hand summed {}; the rest (the whole's median less the \
                     steps') {:.3} ms; each UDMABUF_CREATE_LIST (outside the call) {}",
                        shown(&wholes),
                        shown(&sums),
                        rest as f64 / 1e6,
                        shown(&creates)
                    );
                    for (part, times) in DMA_PARTS.iter().zip(&by_part) {
                        println!(
                            "view dma-buf import time: {name}: {shape}: {part} {}",
                            shown(times)
                        );
                    }
                }
                stop.store(true, Ordering::Relaxed);
                if let Some(worker) = worker {
                    let commits = worker.join().expect("the backlog thread");
                    println!(
                    "view dma-buf import time: {name}: the backlog committed {commits} fills of \
                     256 MiB; the device's submissions so far {}",
                    device.counters().metrics().submissions
                );
                }
            }
        });
    }

    /// `Device::import_host_memory` of 12 KiB, 16 KiB, 17 MiB and 29 MiB of host memory of each
    /// kind (anonymous; on Linux one shared memfd mapping, and one-page mappings of scattered pages
    /// of a memfd; a kind the device refuses is printed and not timed), whole and by
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
            for &(named, kind) in kinds() {
                if let Err(error) = whole(&device, 17 << 20, alignment, kind) {
                    println!("import time: {name}: {named}: refused: {error}; not timed");
                    continue;
                }
                for len in [12u64 << 10, 16 << 10, 17 << 20, 29 << 20] {
                    let len = len.next_multiple_of(alignment);
                    for arm in ["idle", "busy"] {
                        let stop = Arc::new(AtomicBool::new(false));
                        let worker =
                            (arm == "busy").then(|| busy(device.clone(), Arc::clone(&stop)));
                        let mut wholes = Vec::with_capacity(REPS);
                        let mut by_part = vec![Vec::with_capacity(REPS); PARTS.len()];
                        for _ in 0..REPS {
                            let (took, index) = whole(&device, len, alignment, kind)
                                .unwrap_or_else(|error| panic!("an import: {error}"));
                            wholes.push(took);
                            let steps = parts(&device, len, alignment, kind, index);
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
                            "import time: {name}: {named}, {len} bytes, {arm} ({commits} commits \
                             beside): the whole call {}; its steps by hand summed {}",
                            shown(&wholes),
                            shown(&sums)
                        );
                        for (part, times) in PARTS.iter().zip(&by_part) {
                            println!(
                                "import time: {name}: {named}, {len} bytes, {arm}: {part} {}",
                                shown(times)
                            );
                        }
                    }
                }
            }
        });
    }
}
