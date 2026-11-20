//! The time of [`crate::Device::import_host_memory`], under a device cfg on every device of the
//! cfg's kind that imports host memory: the whole call, and each Vulkan call and lock it makes
//! (`import.rs`, `memory::import`), made by hand in its order beside it, with the runtime's queue
//! idle and with another thread committing small work all along.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::device::lock;
    use crate::memory::import_requirements;
    use crate::tests::command::on_device::{finished, on_each_device};
    use crate::tests::import::on_device::{releasing, wait_until};
    use crate::{Device, MemoryClass};
    use ash::vk;
    use std::alloc::Layout;
    use std::collections::VecDeque;
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
    /// written, and its layout.
    fn pages(len: u64, alignment: u64) -> (NonNull<u8>, Layout) {
        let layout = Layout::from_size_align(len as usize, alignment as usize).expect("a layout");
        // SAFETY: the layout's size is not zero.
        let pointer = NonNull::new(unsafe { std::alloc::alloc(layout) }).expect("an allocation");
        for at in (0..len as usize).step_by(4096) {
            // SAFETY: `at` is inside the allocation.
            unsafe { pointer.add(at).write(at as u8) };
        }
        (pointer, layout)
    }

    /// The whole call's time over `len` bytes, in nanoseconds, and the memory type it took; the
    /// block dropped and its release waited for outside the time.
    fn whole(device: &Device, len: u64, alignment: u64) -> (u128, u32) {
        let (pointer, layout) = pages(len, alignment);
        let (release, released) = releasing();
        let start = Instant::now();
        // SAFETY: the memory is the test's, mapped until it frees it after the release.
        let made = unsafe { device.import_host_memory(pointer, len, release) };
        let took = start.elapsed().as_nanos();
        let (block, index) = made.unwrap_or_else(|error| panic!("an import: {error}"));
        drop(block);
        wait_until("the release", || released.load(Ordering::SeqCst));
        // SAFETY: allocated with `layout` above, and the release ran: nothing uses it.
        unsafe { std::alloc::dealloc(pointer.as_ptr(), layout) };
        (took, index)
    }

    /// Each of the call's steps' time over `len` bytes, in nanoseconds, made by hand into memory
    /// type `index`; what they made is destroyed outside the times.
    fn parts(device: &Device, len: u64, alignment: u64, index: u32) -> [u128; PARTS.len()] {
        let core = device.core();
        let ash = &core.device;
        let (host, _) = (core.imports.host.as_ref()).expect("host memory imports");
        let handle = vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT;
        let (pointer, layout) = pages(len, alignment);
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
        // SAFETY: allocated with `layout` above, and its import is freed: nothing uses it.
        unsafe { std::alloc::dealloc(pointer.as_ptr(), layout) };
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

    /// `Device::import_host_memory` of 12 KiB, 16 KiB and 29 MiB of host memory, whole and by
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
            for len in [12u64 << 10, 16 << 10, 29 << 20] {
                let len = len.next_multiple_of(alignment);
                for arm in ["idle", "busy"] {
                    let stop = Arc::new(AtomicBool::new(false));
                    let worker = (arm == "busy").then(|| busy(device.clone(), Arc::clone(&stop)));
                    let mut wholes = Vec::with_capacity(REPS);
                    let mut by_part = vec![Vec::with_capacity(REPS); PARTS.len()];
                    for _ in 0..REPS {
                        let (took, index) = whole(&device, len, alignment);
                        wholes.push(took);
                        let steps = parts(&device, len, alignment, index);
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
                        "import time: {name}: {len} bytes, {arm} ({commits} commits beside): the \
                         whole call {}; its steps by hand summed {}",
                        shown(&wholes),
                        shown(&sums)
                    );
                    for (part, times) in PARTS.iter().zip(&by_part) {
                        println!(
                            "import time: {name}: {len} bytes, {arm}: {part} {}",
                            shown(times)
                        );
                    }
                }
            }
        });
    }
}
