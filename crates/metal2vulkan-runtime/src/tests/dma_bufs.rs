//! Imports over several dma-bufs ([`crate::Device::import_dma_bufs`]): the granule rule, and on
//! every Linux device of the cfg's kind, that a block over three dma-bufs of one memfd is one
//! range, read and written exact across its chunks' boundaries by a kernel through the block's
//! GPU addresses and by blits through its buffer, the bytes landing in the memfd, its imports
//! counted and its release run once the last is freed; that a length off the granule is refused
//! with nothing imported; that a device without the granule refuses the call typed; and the time
//! a submission takes beside live imports, of a block over several dma-bufs and of one dma-buf
//! each.

#[cfg(unix)]
use crate::import::granule;
#[cfg(unix)]
use ash::vk;

#[cfg(unix)]
fn needs(alignment: u64, memory_type_bits: u32) -> vk::MemoryRequirements {
    vk::MemoryRequirements {
        size: alignment,
        alignment,
        memory_type_bits,
    }
}

/// The granule is the sparse buffer's alignment, rounded up to the host's page, and only where
/// the sparse buffer lists every coherent type the plain one lists.
#[cfg(unix)]
#[test]
fn the_granule_is_the_sparse_alignment_where_the_sparse_buffer_takes_every_coherent_type() {
    // RADV's 780M: both list 0x1eaad; the coherent types are 3, 4, 5, 6 and 15 among them.
    let coherent = 0b1000_0000_0111_1000;
    let radv = (needs(16, 0x1eaad), needs(0x10000, 0x1eaad));
    assert_eq!(granule(&radv.0, &radv.1, coherent, 4096), Some(0x10000));
    // lavapipe: one type, 4 KiB alignment.
    let lavapipe = (needs(64, 1), needs(0x1000, 1));
    assert_eq!(granule(&lavapipe.0, &lavapipe.1, 1, 4096), Some(0x1000));
    // An alignment below the page rounds up to it; one above a page that is no multiple of it
    // has no granule.
    assert_eq!(granule(&needs(16, 1), &needs(256, 1), 1, 4096), Some(4096));
    assert_eq!(granule(&needs(16, 1), &needs(6144, 1), 1, 4096), None);
    // A sparse buffer that does not list a coherent type the plain one lists: none, though it
    // lists every other.
    assert_eq!(
        granule(&needs(16, 0b111), &needs(0x10000, 0b101), 0b010, 4096),
        None
    );
    // A non-coherent type the sparse buffer does not list leaves the granule.
    assert_eq!(
        granule(&needs(16, 0b111), &needs(0x10000, 0b011), 0b010, 4096),
        Some(0x10000)
    );
}

#[cfg(all(target_os = "linux", any(m2v_gpu, m2v_cpu_icd)))]
mod on_device {
    use crate::buffer::BufferObject;
    use crate::tests::command::on_device::{each_device, finished, on_each_device};
    use crate::tests::compute::on_device::{add, WORDS};
    use crate::tests::import::on_device::{memfd, releasing, udmabuf, wait_until};
    use crate::{
        Buffer, CommandQueue, Device, DispatchType, Error, ImportHandle, Limit, MemoryClass,
    };
    use ash::vk;
    use std::os::fd::OwnedFd;
    use std::os::unix::fs::FileExt;
    use std::sync::atomic::Ordering;
    use std::time::Instant;

    /// The memfd's word at word index `w` before the GPU writes.
    fn word(w: u64) -> u32 {
        (w as u32).wrapping_mul(0x9e37_79b1) ^ 0x5a5a_5a5a
    }

    /// The blit's source word at word index `w` of its 512 bytes.
    fn blit_word(w: u64) -> u32 {
        0xb17_0000 + w as u32
    }

    fn words_of(bytes: &[u8]) -> Vec<u32> {
        (bytes.as_chunks::<4>().0.iter())
            .map(|&word| u32::from_le_bytes(word))
            .collect()
    }

    fn bytes_of(words: impl Iterator<Item = u32>) -> Vec<u8> {
        words.flat_map(u32::to_le_bytes).collect()
    }

    /// A Shared buffer of `len` bytes of a block of its own.
    fn shared(device: &Device, len: u64) -> (crate::Block, Buffer) {
        let block = (device.take_block(MemoryClass::Shared, len)).expect("a Shared block");
        let buffer = (device.new_buffer(block.id, 0, len, None)).expect("a buffer");
        (block, buffer)
    }

    fn read_bytes(buffer: &Buffer) -> Vec<u8> {
        let pointer = buffer.contents().expect("a Shared buffer");
        // SAFETY: the buffer is Shared and alive, and its GPU work completed.
        unsafe { std::slice::from_raw_parts(pointer.as_ptr(), buffer.length() as usize) }.to_vec()
    }

    fn write_bytes(buffer: &Buffer, bytes: &[u8]) {
        let pointer = buffer.contents().expect("a Shared buffer");
        assert_eq!(bytes.len() as u64, buffer.length());
        // SAFETY: the buffer is Shared, alive and `bytes`' length, and no GPU work uses it.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), pointer.as_ptr(), bytes.len()) };
    }

    fn run(name: &str, command_buffer: &crate::CommandBuffer) {
        command_buffer.commit().expect("a commit");
        finished(name, command_buffer);
        assert_eq!(command_buffer.error(), None, "{name}");
    }

    /// Dma-bufs of `memfd`'s consecutive ranges of `lens`, with their lengths; `None` where the
    /// kernel offers no `/dev/udmabuf`.
    fn chunks(memfd: &OwnedFd, lens: &[u64]) -> Option<Vec<(OwnedFd, u64)>> {
        let mut at = 0;
        let mut chunks = Vec::with_capacity(lens.len());
        for &len in lens {
            chunks.push((udmabuf(memfd, at, len)?, len));
            at += len;
        }
        Some(chunks)
    }

    /// The compute round and the blit round over the block's buffer of `total` bytes, whose
    /// chunks meet at `first` and `second`: `add` reads 256 bytes across `first` through their
    /// GPU addresses and writes 256 across `second`; then a blit reads 512 bytes across `second`
    /// through the buffer and writes 512 across `first`. Each against the bytes the memfd holds
    /// after, which are all checked.
    fn rounds(
        name: &str,
        device: &Device,
        queue: &CommandQueue,
        buffer: &Buffer,
        memfd: &OwnedFd,
        (first, second, total): (u64, u64, u64),
    ) {
        let half = (WORDS * 4 / 2) as u64;
        let (a_at, o_at) = (first - half, second - half);
        let state = add(device);
        let command_buffer = queue.command_buffer().expect("a command buffer");
        let mut encoder =
            (command_buffer.compute_command_encoder(DispatchType::Serial)).expect("an encoder");
        encoder.set_compute_pipeline_state(&state).expect("a state");
        encoder.set_buffer(Some(buffer), a_at, 0).expect("a");
        let b = bytes_of((0..WORDS as u32).map(|i| 1000 + i));
        encoder.set_bytes(&b, 1).expect("b");
        encoder.set_buffer(Some(buffer), o_at, 2).expect("o");
        (encoder.dispatch_threads([WORDS as u32, 1, 1], [16, 1, 1])).expect("a dispatch");
        encoder.end_encoding();
        run(name, &command_buffer);

        let (_read_block, read) = shared(device, 512);
        let (_source_block, source) = shared(device, 512);
        write_bytes(&source, &bytes_of((0..128).map(blit_word)));
        let command_buffer = queue.command_buffer().expect("a command buffer");
        let mut encoder = command_buffer.blit_command_encoder().expect("an encoder");
        (encoder.copy_from_buffer_to_buffer(buffer, second - 256, &read, 0, 512))
            .unwrap_or_else(|error| panic!("{name}: the read: {error}"));
        (encoder.copy_from_buffer_to_buffer(&source, 0, buffer, first - 256, 512))
            .unwrap_or_else(|error| panic!("{name}: the write: {error}"));
        encoder.end_encoding();
        run(name, &command_buffer);

        let mut expected: Vec<u32> = (0..total / 4).map(word).collect();
        for i in 0..WORDS as u64 {
            let o = ((o_at / 4) + i) as usize;
            expected[o] = word(a_at / 4 + i).wrapping_add(1000 + i as u32);
        }
        let read_from = ((second - 256) / 4) as usize;
        assert_eq!(
            words_of(&read_bytes(&read)),
            expected[read_from..read_from + 128],
            "{name}: the blit read across the second boundary, the kernel's write in it"
        );
        let write_at = ((first - 256) / 4) as usize;
        for w in 0..128 {
            expected[write_at + w] = blit_word(w as u64);
        }
        let mut pages = vec![0u8; total as usize];
        (memfd.try_clone().map(std::fs::File::from))
            .and_then(|file| file.read_exact_at(&mut pages, 0))
            .expect("the memfd's bytes");
        let got = words_of(&pages);
        let wrong = (0..got.len()).find(|&w| got[w] != expected[w]);
        assert_eq!(
            wrong.map(|w| (w * 4, got[w], expected[w])),
            None,
            "{name}: the memfd's bytes after the kernel and the blits"
        );
    }

    /// Where the device offers the granule, three dma-bufs of one memfd, of one, two and one
    /// granule, are one block: Shared, with no contents, its buffer at its GPU address, three
    /// imports counted, one bind. Its first command buffer, the bind's first submission, reads
    /// and writes across both boundaries exact ([`rounds`]); dropped, the release runs once the
    /// imports are freed and the gauges are back. A length off the granule and no dma-buf at all
    /// are refused with nothing imported. A device without the granule refuses the call typed.
    #[test]
    fn every_device_imports_dma_bufs_behind_one_buffer_or_refuses_typed() {
        each_device("import_dma_bufs", |name, _, device| {
            let counters = device.counters();
            let Some(granule) = device.dma_buf_granule() else {
                let memfd = memfd(64 << 10);
                let Some(chunks) = chunks(&memfd, &[64 << 10]) else {
                    println!("dma_bufs: {name}: no /dev/udmabuf here; not run");
                    return;
                };
                match device.import_dma_bufs(chunks, None) {
                    Err(Error::Limit(Limit::NoImport {
                        handle: ImportHandle::DmaBufs,
                    })) => {}
                    other => panic!("{name}: {other:?}"),
                }
                println!("dma_bufs: {name}: no granule, refused typed");
                return;
            };
            let lens = [granule, 2 * granule, granule];
            let total: u64 = lens.iter().sum();
            let memfd = memfd(total);
            let initial = bytes_of((0..total / 4).map(word));
            memfd
                .try_clone()
                .map(std::fs::File::from)
                .and_then(|file| file.write_all_at(&initial, 0))
                .expect("the memfd's first bytes");
            let Some(parts) = chunks(&memfd, &lens) else {
                println!("dma_bufs: {name}: no /dev/udmabuf here; not run");
                return;
            };

            // The refusals first, with nothing imported.
            let before = counters.gauges();
            match device.import_dma_bufs(Vec::new(), None) {
                Err(Error::Limit(Limit::ImportAlignment {
                    handle: ImportHandle::DmaBufs,
                    len: 0,
                    alignment,
                })) => assert_eq!(alignment, granule, "{name}"),
                other => panic!("{name}: no dma-buf: {other:?}"),
            }
            let off = (chunks(&memfd, &[2 * granule]).expect("a dma-buf"))
                .into_iter()
                .map(|(fd, len)| (fd, len - granule / 2))
                .collect();
            match device.import_dma_bufs(off, None) {
                Err(Error::Limit(Limit::ImportAlignment {
                    handle: ImportHandle::DmaBufs,
                    len,
                    alignment,
                })) => assert_eq!((len, alignment), (2 * granule - granule / 2, granule)),
                other => panic!("{name}: a length off the granule: {other:?}"),
            }
            assert_eq!(counters.gauges(), before, "{name}: nothing imported");

            let binds = counters.metrics().sparse_binds;
            let (release, released) = releasing();
            let (block, memory_type) = (device.import_dma_bufs(parts, release))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            assert_eq!(
                (block.class, block.contents, block.len, block.offset),
                (MemoryClass::Shared, None, total, 0),
                "{name}"
            );
            let gauges = counters.gauges();
            assert_eq!(
                (gauges.imports, gauges.blocks),
                (before.imports + 3, before.blocks + 3),
                "{name}: each dma-buf one import"
            );
            assert_eq!(
                counters.metrics().sparse_binds,
                binds + 1,
                "{name}: one bind"
            );
            let buffer = (device.new_buffer(block.id, 0, total, None)).expect("a buffer");
            assert_eq!(buffer.gpu_address(), block.gpu_address, "{name}");
            let queue = device.new_command_queue(4).expect("a queue");
            rounds(
                name,
                &device,
                &queue,
                &buffer,
                &memfd,
                (granule, 3 * granule, total),
            );
            drop((queue, buffer, block));
            wait_until(&device, "the release", || released.load(Ordering::SeqCst));
            // Each import falls before the release runs; the pool's own allocations may have
            // moved with the test's Shared blocks.
            assert_eq!(
                counters.gauges().imports,
                before.imports,
                "{name}: every import freed before the release"
            );
            println!(
                "dma_bufs: {name}: three dma-bufs of {granule}-byte granules behind one buffer of \
                 type {memory_type}, exact across both boundaries by address and by blit"
            );
        });
    }

    /// The times of `submits` calls of `vkQueueSubmit2` on the runtime's queue, each of one
    /// command buffer, waited for after the call and outside the time, in nanoseconds.
    fn submit_times(device: &Device, command: vk::CommandBuffer, submits: usize) -> Vec<u128> {
        let raw = device.raw();
        let ash = raw.device;
        // SAFETY: the device is live.
        let fence =
            (unsafe { ash.create_fence(&vk::FenceCreateInfo::default(), None) }).expect("a fence");
        let mut nanos: Vec<u128> = Vec::with_capacity(submits);
        for _ in 0..submits {
            let commands = [vk::CommandBufferSubmitInfo::default().command_buffer(command)];
            let submit = [vk::SubmitInfo2::default().command_buffer_infos(&commands)];
            let took = device.with_runtime_queue(|queue| {
                let start = Instant::now();
                // SAFETY: the command buffer is executable and not pending (the fence waited
                // for its last submission); the queue is held.
                unsafe { ash.queue_submit2(queue, &submit, fence) }.expect("a submit");
                start.elapsed().as_nanos()
            });
            nanos.push(took);
            // SAFETY: the fence was submitted.
            unsafe {
                ash.wait_for_fences(&[fence], true, u64::MAX)
                    .expect("a wait");
                ash.reset_fences(&[fence]).expect("a reset");
            }
        }
        // SAFETY: nothing waits on the fence.
        unsafe { ash.destroy_fence(fence, None) };
        nanos
    }

    /// The median of `nanos` in microseconds, and the median with its spread as the test prints
    /// it: the 5th and 95th percentiles, the longest and the count.
    fn spread(nanos: &[u128]) -> (f64, String) {
        let mut sorted = nanos.to_vec();
        sorted.sort_unstable();
        let at = |percent: usize| sorted[(sorted.len() - 1) * percent / 100] as f64 / 1000.0;
        let median = at(50);
        let shown = format!(
            "{median:.2} us (p5 {:.2}, p95 {:.2}, max {:.2}, {} submits)",
            at(5),
            at(95),
            at(100),
            sorted.len()
        );
        (median, shown)
    }

    /// The time of a `vkQueueSubmit2` of one command buffer (empty, and one of a 4-byte fill),
    /// beside no import, beside 8 and 40 live dma-bufs of one block over several, and beside 8
    /// and 40 live blocks of one dma-buf each, in three rounds of every arm: each arm's median
    /// over its rounds' submissions with their spread, and the difference from none per live
    /// import, printed. The runtime adds nothing to its own submissions for a live block over
    /// several dma-bufs; what a driver pays for each live import of memory, it pays for each
    /// chunk. On an instance without the validation layer, whose own cost would be the count.
    #[test]
    fn every_device_times_submissions_beside_live_dma_bufs() {
        const COUNTS: [usize; 2] = [8, 40];
        const SUBMITS: usize = 1000;
        const ROUNDS: usize = 3;
        let instance = crate::tests::validation::bare(&[]);
        on_each_device("dma_bufs_submit_cost", &instance, |name, _, device| {
            let Some(granule) = device.dma_buf_granule() else {
                println!("dma_bufs: {name}: no granule; not timed");
                return;
            };
            let most = COUNTS[COUNTS.len() - 1];
            let memfd = memfd(granule * most as u64);
            let lens = vec![granule; most];
            if chunks(&memfd, &lens[..1]).is_none() {
                println!("dma_bufs: {name}: no /dev/udmabuf here; not timed");
                return;
            }
            let (_target_block, target) = shared(&device, 4096);
            let (vk_target, offset) = target.device_range();
            let raw = device.raw();
            let ash = raw.device;
            let pool_info =
                vk::CommandPoolCreateInfo::default().queue_family_index(raw.queue_family_index);
            // SAFETY: the family is the device's.
            let pool = unsafe { ash.create_command_pool(&pool_info, None) }.expect("a pool");
            let allocate = vk::CommandBufferAllocateInfo::default()
                .command_pool(pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(2);
            // SAFETY: the pool is live.
            let commands = unsafe { ash.allocate_command_buffers(&allocate) }.expect("commands");
            // SAFETY: the command buffers are new; the target is held until the pool goes.
            unsafe {
                for &command in &commands {
                    ash.begin_command_buffer(command, &vk::CommandBufferBeginInfo::default())
                        .expect("a begin");
                }
                ash.cmd_fill_buffer(commands[1], vk_target, offset, 4, 7);
                for &command in &commands {
                    ash.end_command_buffer(command).expect("an end");
                }
            }
            let shapes = ["empty", "a fill"];
            // [shape][arm] -> every round's times: arm 0 beside no import, then for the count at
            // `at`, arm 1 + 2 * at beside one block over that many dma-bufs and arm 2 + 2 * at
            // beside that many one-dma-buf blocks.
            let mut times = vec![vec![Vec::new(); 1 + 2 * COUNTS.len()]; shapes.len()];
            let mut one_each = device.core().imports.dma_buf.is_some();
            for _ in 0..ROUNDS {
                for (shape, &command) in commands.iter().enumerate() {
                    times[shape][0].extend(submit_times(&device, command, SUBMITS));
                }
                for (at, &count) in COUNTS.iter().enumerate() {
                    let parts = chunks(&memfd, &lens[..count]).expect("the dma-bufs");
                    let (block, _) = (device.import_dma_bufs(parts, None))
                        .unwrap_or_else(|error| panic!("{name}: {error}"));
                    for (shape, &command) in commands.iter().enumerate() {
                        times[shape][1 + 2 * at].extend(submit_times(&device, command, SUBMITS));
                    }
                    drop(block);
                    wait_until(&device, "the chunks' frees", || {
                        device.counters().gauges().imports == 0
                    });
                    let blocks: Vec<_> = (0..count)
                        .filter(|_| one_each)
                        .filter_map(|chunk| {
                            let start = chunk as u64 * granule;
                            let fd = udmabuf(&memfd, start, granule).expect("a dma-buf");
                            device.import_dma_buf(fd, granule, None).ok()
                        })
                        .collect();
                    one_each &= blocks.len() == count;
                    if one_each {
                        for (shape, &command) in commands.iter().enumerate() {
                            times[shape][2 + 2 * at]
                                .extend(submit_times(&device, command, SUBMITS));
                        }
                    }
                    drop(blocks);
                    wait_until(&device, "the imports' frees", || {
                        device.counters().gauges().imports == 0
                    });
                }
            }
            // SAFETY: nothing is pending.
            unsafe { ash.destroy_command_pool(pool, None) };
            for (shape, arms) in shapes.iter().zip(&times) {
                let (none, shown) = spread(&arms[0]);
                let mut line =
                    format!("dma_bufs: {name}: a submit of {shape} took {shown} beside no import");
                for (at, &count) in COUNTS.iter().enumerate() {
                    let per = |time: f64| (time - none) / count as f64;
                    let (several, shown) = spread(&arms[1 + 2 * at]);
                    line += &format!(
                        "; {shown} beside one block of {count} dma-bufs ({:.3} us per dma-buf)",
                        per(several)
                    );
                    let each = &arms[2 + 2 * at];
                    line += &match each.is_empty() {
                        true => format!(
                            "; no {count} one-dma-buf blocks (the device takes no such dma-buf alone)"
                        ),
                        false => {
                            let (each, shown) = spread(each);
                            format!(
                                "; {shown} beside {count} one-dma-buf blocks ({:.3} us per import)",
                                per(each)
                            )
                        }
                    };
                }
                println!("{line}");
            }
        });
    }
}
