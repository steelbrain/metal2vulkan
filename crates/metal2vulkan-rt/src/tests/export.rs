//! Exports, under a device cfg, on every device of the cfg's kind: each handle the device exports
//! gives a Private block of its own, counted live in `Gauges::exports` until its memory is freed,
//! which rt writes and releases (`BlitCommandEncoder::release_to_external`) twice, with no acquire
//! between, and whose handle a second device of the same physical device imports, reading rt's
//! second bytes through it and writing the whole block, which rt acquires
//! (`Block::external_writes_done`) and reads; an acquire or a release of a block that is not
//! exported is refused; each handle the device does not export is
//! refused typed; an exported block holds no image (a texture
//! over a buffer in it is the counted copy, and a heap texture in it is refused); and submissions
//! beside 64 live exports, whose cost the harness's own row `export_submit_cost` times.
//!
//! The second device stands in for the other API (a video decoder): a device of another instance
//! cannot run under the validation layer, whose log is one file per test process
//! (`crate::tests::validation`).

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::buffer::BufferObject;
    use crate::tests::command::on_device::{each_device, finished, within};
    use crate::tests::devices_of_the_cfgs_kind;
    use crate::texture::TextureObject;
    use crate::{
        BufferLayout, Device, Error, Export, ExportHandle, Gap, HeapRange, IndexClass, Limit,
        MemoryClass, PixelFormat, Refusal, TextureDescriptor, TexturePlace, TextureType,
        TextureUsage,
    };
    use ash::vk;

    /// The length of each export the tests make.
    const LEN: u64 = 1 << 20;

    /// Whether `device` enabled `name`.
    fn enabled(device: &Device, name: &std::ffi::CStr) -> bool {
        device.extensions().iter().any(|n| n.as_c_str() == name)
    }

    /// A barrier of the whole of `buffer` that moves it from queue family `from` to `to`: a
    /// release where `to` is another API's, after every command's writes, and an acquire where
    /// `from` is, before every command's reads and writes. Each side's other masks are ignored
    /// (the spec's queue family ownership transfer), and are none.
    fn transfer(
        buffer: vk::Buffer,
        from: u32,
        to: u32,
        release: bool,
    ) -> vk::BufferMemoryBarrier2<'static> {
        let all = vk::PipelineStageFlags2::ALL_COMMANDS;
        let barrier = vk::BufferMemoryBarrier2::default()
            .src_queue_family_index(from)
            .dst_queue_family_index(to)
            .buffer(buffer)
            .offset(0)
            .size(vk::WHOLE_SIZE);
        match release {
            true => barrier
                .src_stage_mask(all)
                .src_access_mask(vk::AccessFlags2::MEMORY_WRITE),
            false => barrier
                .dst_stage_mask(all)
                .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE),
        }
    }

    /// Records `barrier` into `command`.
    ///
    /// # Safety
    ///
    /// `command` is recording on `device`, and the barrier's buffer is live.
    unsafe fn record(
        device: &ash::Device,
        command: vk::CommandBuffer,
        barrier: vk::BufferMemoryBarrier2<'_>,
    ) {
        let barriers = [barrier];
        let dependency = vk::DependencyInfo::default().buffer_memory_barriers(&barriers);
        // SAFETY: the caller's.
        unsafe { device.cmd_pipeline_barrier2(command, &dependency) };
    }

    /// The other API's buffer: `export`'s memory imported into `device` as a buffer of `len`
    /// bytes made as the exporting block's buffer is, bound to an allocation dedicated to it, as
    /// an import of an opaque handle must be (`VUID-VkMemoryDedicatedAllocateInfo-buffer-01879`).
    fn import(
        device: &Device,
        handle: ExportHandle,
        export: Export,
        len: u64,
    ) -> (vk::Buffer, vk::DeviceMemory) {
        let raw = device.raw();
        let ash = raw.device;
        let mut external = vk::ExternalMemoryBufferCreateInfo::default().handle_types(handle.vk());
        let info = vk::BufferCreateInfo::default()
            .size(len)
            .usage(crate::floor::BUFFER_USAGE)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .push_next(&mut external);
        // SAFETY: the device's buffers import the handle, as they export it.
        let buffer = unsafe { ash.create_buffer(&info, None) }.expect("the import's buffer");
        // SAFETY: the buffer is live.
        let needs = unsafe { ash.get_buffer_memory_requirements(buffer) };
        if handle != ExportHandle::DmaBuf {
            assert_eq!(
                needs.size, export.size,
                "the import's buffer is the export's"
            );
        }
        let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().buffer(buffer);
        let mut flags =
            vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        #[cfg(unix)]
        let memory = {
            use std::os::fd::{AsRawFd, IntoRawFd};
            let types = match handle {
                ExportHandle::DmaBuf => {
                    let fds = ash::khr::external_memory_fd::Device::new(raw.instance, ash);
                    let mut properties = vk::MemoryFdPropertiesKHR::default();
                    // SAFETY: the fd is a dma-buf the device exported.
                    unsafe {
                        fds.get_memory_fd_properties(
                            handle.vk(),
                            export.handle.as_raw_fd(),
                            &mut properties,
                        )
                    }
                    .expect("the dma-buf's memory types");
                    properties.memory_type_bits
                }
                _ => 1 << export.memory_type,
            };
            let types = types & needs.memory_type_bits;
            let index = match types & (1 << export.memory_type) {
                0 => types.trailing_zeros(),
                _ => export.memory_type,
            };
            assert!(index < 32, "the import takes a type the buffer lists");
            let fd = export.handle.into_raw_fd();
            let mut import = vk::ImportMemoryFdInfoKHR::default()
                .handle_type(handle.vk())
                .fd(fd);
            let info = vk::MemoryAllocateInfo::default()
                .allocation_size(needs.size)
                .memory_type_index(index)
                .push_next(&mut import)
                .push_next(&mut dedicated)
                .push_next(&mut flags);
            // SAFETY: an import of the fd, of the size and a type the buffer takes, dedicated to
            // it; a successful import owns the fd.
            unsafe { ash.allocate_memory(&info, None) }.expect("the import")
        };
        #[cfg(windows)]
        let memory = {
            use std::os::windows::io::AsRawHandle;
            let mut import = vk::ImportMemoryWin32HandleInfoKHR::default()
                .handle_type(handle.vk())
                .handle(export.handle.as_raw_handle() as vk::HANDLE);
            let info = vk::MemoryAllocateInfo::default()
                .allocation_size(needs.size)
                .memory_type_index(export.memory_type)
                .push_next(&mut import)
                .push_next(&mut dedicated)
                .push_next(&mut flags);
            // SAFETY: an import of the NT handle, of the export's size and type, dedicated to the
            // buffer; the import does not own the handle, which `export` closes as it drops.
            unsafe { ash.allocate_memory(&info, None) }.expect("the import")
        };
        // SAFETY: the memory is the buffer's dedicated allocation.
        unsafe { ash.bind_buffer_memory(buffer, memory, 0) }.expect("the import's binding");
        (buffer, memory)
    }

    /// Each handle the device exports gives a Private block of its own, counted live in
    /// `Gauges::exports`, that rt fills and releases to the other API's queue family in one
    /// command buffer (`BlitCommandEncoder::release_to_external`, counted), then fills and
    /// releases again in a second with no acquire between, as Vulkan allows; a second device of
    /// the same physical device imports it and reads rt's second bytes from it once both have
    /// completed; the second device then fills the whole block and releases it, and rt's acquire
    /// (`Block::external_writes_done`, counted) lets a blit read its bytes. The gauge is back
    /// once the block is given back. An acquire of a taken block, and a release of a buffer in
    /// one, are `Refusal::NotExported`. Each handle it does not export is `Limit::NoExport`, and
    /// a dma-buf on a device without the foreign queue family `Limit::ExportType`.
    #[test]
    fn every_device_exports_each_offered_handle_and_refuses_the_rest() {
        each_device("export", |name, instance, device| {
            let counters = device.counters();
            let foreign = enabled(&device, ash::ext::queue_family_foreign::NAME);
            let taken = (device.take_block(MemoryClass::Private, LEN))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            match taken.external_writes_done() {
                Err(Error::Refused(Refusal::NotExported { block })) if block == taken.id => {}
                other => panic!("{name}: an acquire of a taken block: {other:?}"),
            }
            let in_taken = (device.new_buffer(taken.id, 0, LEN, None)).expect("a buffer");
            let queue = device.new_command_queue(4).expect("a queue");
            let unused = queue.command_buffer().expect("a command buffer");
            let mut encoder = unused.blit_command_encoder().expect("an encoder");
            match encoder.release_to_external(&in_taken) {
                Err(Error::Refused(Refusal::NotExported { block })) if block == taken.id => {}
                other => panic!("{name}: a release of a buffer in a taken block: {other:?}"),
            }
            encoder.end_encoding();
            drop((unused, queue, in_taken, taken));
            let mut exported = 0;
            for handle in ExportHandle::ALL {
                if !device.core().exports.offers(handle) || (handle.foreign() && !foreign) {
                    match device.new_exported_block(handle, LEN) {
                        Err(Error::Limit(Limit::NoExport { handle: refused }))
                            if refused == handle && !device.core().exports.offers(handle) => {}
                        Err(Error::Limit(Limit::ExportType {
                            handle: refused,
                            listed: None,
                        })) if refused == handle => {}
                        other => panic!("{name}: {handle:?}: {other:?}"),
                    }
                    println!("export: {name}: {handle:?} not exported, refused typed");
                    continue;
                }
                let before = counters.gauges();
                let (block, export) = (device.new_exported_block(handle, LEN))
                    .unwrap_or_else(|error| panic!("{name}: {handle:?}: {error}"));
                assert_eq!(
                    (block.class, block.len, block.offset, block.contents),
                    (MemoryClass::Private, LEN, 0, None),
                    "{name}: {handle:?}"
                );
                assert!(export.size >= LEN, "{name}: {handle:?}: {}", export.size);
                let live = counters.gauges();
                assert_eq!(
                    (live.exports, live.blocks),
                    (before.exports + 1, before.blocks + 1),
                    "{name}: {handle:?}"
                );

                // rt writes the block and releases it to the other API, twice: the second write
                // takes no acquire after the first release, and the other API reads its bytes.
                let buffer = (device.new_buffer(block.id, 0, LEN, None)).expect("a buffer");
                let queue = device.new_command_queue(4).expect("a queue");
                let releases = counters.metrics().external_releases;
                for (low, high) in [(0x11, 0x22), (0x5a, 0xa5)] {
                    let command_buffer = queue.command_buffer().expect("a command buffer");
                    let mut encoder = command_buffer.blit_command_encoder().expect("an encoder");
                    (encoder.fill_buffer(&buffer, 0..LEN / 2, low))
                        .unwrap_or_else(|error| panic!("{name}: {error}"));
                    (encoder.fill_buffer(&buffer, LEN / 2..LEN, high))
                        .unwrap_or_else(|error| panic!("{name}: {error}"));
                    (encoder.release_to_external(&buffer))
                        .unwrap_or_else(|error| panic!("{name}: {handle:?}: {error}"));
                    encoder.end_encoding();
                    command_buffer.commit().expect("a commit");
                    finished(name, &command_buffer);
                    assert_eq!(command_buffer.error(), None, "{name}");
                }
                assert_eq!(
                    counters.metrics().external_releases,
                    releases + 2,
                    "{name}: the releases, counted"
                );
                let external = handle.family();

                // The other API imports the handle, acquires the memory and reads rt's bytes.
                let physical = (devices_of_the_cfgs_kind(instance).into_iter())
                    .map(|(_, physical)| physical)
                    .find(|physical| physical.raw() == device.raw().physical_device)
                    .expect("the device's physical device");
                let other = Device::new(&physical, Default::default())
                    .unwrap_or_else(|error| panic!("{name}: the second device: {error}"));
                let (imported, memory) = import(&other, handle, export, LEN);
                let out_block = (other.take_block(MemoryClass::Shared, LEN))
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
                let out = (other.new_buffer(out_block.id, 0, LEN, None)).expect("a buffer");
                let (out_buffer, out_offset) = out.device_range();
                let other_family = other.raw().queue_family_index;
                let other_queue = other.new_command_queue(4).expect("a queue");
                let read = other_queue.command_buffer().expect("a command buffer");
                read.record_for_test(&[&out], |raw, command| {
                    let copy = [vk::BufferCopy {
                        src_offset: 0,
                        dst_offset: out_offset,
                        size: LEN,
                    }];
                    // SAFETY: the command buffer is recording; the import lives until the test
                    // frees it after the wait, and the Shared buffer is held.
                    unsafe {
                        record(
                            raw,
                            command,
                            transfer(imported, external, other_family, false),
                        );
                        raw.cmd_copy_buffer(command, imported, out_buffer, &copy);
                    }
                })
                .expect("the read");
                read.commit().expect("a commit");
                finished(name, &read);
                assert_eq!(read.error(), None, "{name}");
                let contents = out.contents().expect("Shared");
                // SAFETY: the buffer's bytes, which no command writes any more.
                let bytes = unsafe { std::slice::from_raw_parts(contents.as_ptr(), LEN as usize) };
                let wrong = (bytes.iter().enumerate())
                    .find(|&(at, &byte)| byte != if (at as u64) < LEN / 2 { 0x5a } else { 0xa5 });
                assert_eq!(
                    wrong, None,
                    "{name}: {handle:?}: the import read rt's bytes"
                );

                // The other API writes the whole block and releases it; rt acquires it and a
                // blit reads the other API's bytes.
                let write = other_queue.command_buffer().expect("a command buffer");
                write
                    .record_for_test(&[], |raw, command| {
                        // SAFETY: the command buffer is recording; the import lives until the
                        // test frees it after the wait.
                        unsafe {
                            raw.cmd_fill_buffer(command, imported, 0, LEN, 0x3c3c_3c3c);
                            record(
                                raw,
                                command,
                                transfer(imported, other_family, external, true),
                            );
                        }
                    })
                    .expect("the write");
                write.commit().expect("a commit");
                finished(name, &write);
                assert_eq!(write.error(), None, "{name}");
                let acquires = counters.metrics().external_acquires;
                (block.external_writes_done())
                    .unwrap_or_else(|error| panic!("{name}: {handle:?}: the acquire: {error}"));
                assert_eq!(
                    counters.metrics().external_acquires,
                    acquires + 1,
                    "{name}: the acquire, counted"
                );
                let mine_block = (device.take_block(MemoryClass::Shared, LEN))
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
                let mine = (device.new_buffer(mine_block.id, 0, LEN, None)).expect("a buffer");
                let back = queue.command_buffer().expect("a command buffer");
                let mut encoder = back.blit_command_encoder().expect("an encoder");
                (encoder.copy_from_buffer_to_buffer(&buffer, 0, &mine, 0, LEN))
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
                encoder.end_encoding();
                back.commit().expect("a commit");
                finished(name, &back);
                assert_eq!(back.error(), None, "{name}");
                let contents = mine.contents().expect("Shared");
                // SAFETY: the buffer's bytes, which no command writes any more.
                let bytes = unsafe { std::slice::from_raw_parts(contents.as_ptr(), LEN as usize) };
                let wrong = (bytes.iter().enumerate()).find(|&(_, &byte)| byte != 0x3c);
                assert_eq!(
                    wrong, None,
                    "{name}: {handle:?}: rt read the other API's bytes"
                );
                drop((
                    back,
                    mine,
                    mine_block,
                    write,
                    read,
                    other_queue,
                    out,
                    out_block,
                ));
                // SAFETY: nothing uses the import any more.
                unsafe {
                    other.raw().device.destroy_buffer(imported, None);
                    other.raw().device.free_memory(memory, None);
                }
                drop(other);

                drop((queue, buffer, block));
                within("the export's free", || {
                    counters.gauges().exports == before.exports
                });
                assert_eq!(
                    counters.gauges().blocks,
                    before.blocks,
                    "{name}: {handle:?}"
                );
                exported += 1;
                println!(
                    "release: {name}: {handle:?} written and released twice, no acquire between, \
                     the second write read by a second device"
                );
                println!(
                    "export: {name}: {handle:?} exported ({} bytes, type {}), imported by a second \
                     device, read back, written there, acquired, read by rt, freed",
                    LEN, device.core().memory_type(MemoryClass::Private)
                );
            }
            println!("M2V-MATRIX metric export/handles_exported {exported}");
        });
    }

    /// No image is made in an exported block's memory, which is dedicated to its buffer
    /// (`VUID-vkBindImageMemory-memory-10926`): a texture over a buffer in it is the counted copy,
    /// and a heap texture in it is `Gap::HeapTexture`.
    #[test]
    fn an_exported_block_holds_no_image() {
        each_device("export_image", |name, _, device| {
            let Some(handle) = (ExportHandle::ALL.into_iter())
                .find(|&handle| device.new_exported_block(handle, LEN).is_ok())
            else {
                println!("export: {name}: exports no handle");
                return;
            };
            let (block, _export) = (device.new_exported_block(handle, LEN))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let d = {
                let mut d =
                    TextureDescriptor::new(TextureType::Type2D, PixelFormat::RGBA8Unorm, 64, 8);
                d.usage = TextureUsage::SHADER_READ;
                d.storage_mode = crate::StorageMode::Private;
                d
            };
            let heap = HeapRange {
                block: block.id,
                offset: 0,
                len: LEN,
            };
            match device.new_heap_texture(&d, None, heap, 0) {
                Err(Error::Unsupported(Gap::HeapTexture { .. })) => {}
                Err(error) => panic!("{name}: a heap texture in an export: {error}"),
                Ok(_) => panic!("{name}: a heap texture made in an export"),
            }
            let pitch = (device.minimum_linear_texture_alignment(PixelFormat::RGBA8Unorm))
                .unwrap_or_else(|error| panic!("{name}: {error}"))
                .max(256);
            let buffer = (device.new_buffer(block.id, 0, LEN, None)).expect("a buffer");
            let first = IndexClass::Texture.range().start;
            (device.grant_indices(IndexClass::Texture, first, 1)).expect("a grant");
            let texture = (buffer.new_texture(&d, Some(first), 0, pitch))
                .unwrap_or_else(|error| panic!("{name}: a texture over the export: {error}"));
            assert!(texture.copied_over().is_some(), "{name}: copied");
            // A read of the texture copies the buffer in, counted.
            let before = device.counters().metrics().over_copies;
            let out_block = (device.take_block(MemoryClass::Shared, 64 * 4 * 8))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let out = (device.new_buffer(out_block.id, 0, 64 * 4 * 8, Some(0))).expect("a buffer");
            let queue = device.new_command_queue(4).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = command_buffer.blit_command_encoder().expect("an encoder");
            let layout = BufferLayout {
                offset: 0,
                bytes_per_row: 64 * 4,
                bytes_per_image: 0,
            };
            (encoder.copy_from_texture_to_buffer(
                &texture,
                TexturePlace::default(),
                [64, 8, 1],
                &out,
                layout,
            ))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
            encoder.end_encoding();
            command_buffer.commit().expect("a commit");
            finished(name, &command_buffer);
            assert_eq!(command_buffer.error(), None, "{name}");
            let copies = device.counters().metrics().over_copies - before;
            assert!(copies > 0, "{name}: the copy in, counted");
            println!("M2V-MATRIX metric export_image/over_copies {copies}");
        });
    }

    /// Submissions beside 64 live exports (the consumer's cap of surface caches): a small fill
    /// submitted on the presenting queue, waited for by a fence, with no export and with 64 live,
    /// in rounds that take turns, the exports counted live in `Gauges::exports` and freed between
    /// rounds. What the exports cost each submission is the harness's own row
    /// `export_submit_cost`, timed alone without the validation layer.
    #[test]
    fn every_device_submits_beside_64_live_exports() {
        const EXPORTS: usize = 64;
        const SUBMITS: usize = 4;
        const ROUNDS: usize = 2;
        each_device("export_submits", |name, _, device| {
            let Some(handle) = (ExportHandle::ALL.into_iter()).find(|&handle| {
                device.core().exports.offers(handle)
                    && (!handle.foreign() || enabled(&device, ash::ext::queue_family_foreign::NAME))
            }) else {
                println!("export: {name}: exports no handle");
                return;
            };
            let raw = device.raw();
            let ash = raw.device;
            let target = (device.take_block(MemoryClass::Private, 1 << 16))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let target = (device.new_buffer(target.id, 0, 1 << 16, None)).expect("a buffer");
            let (vk_target, offset) = target.device_range();
            let pool_info =
                vk::CommandPoolCreateInfo::default().queue_family_index(raw.queue_family_index);
            // SAFETY: the family is the device's.
            let pool = unsafe { ash.create_command_pool(&pool_info, None) }.expect("a pool");
            let allocate = vk::CommandBufferAllocateInfo::default()
                .command_pool(pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            // SAFETY: the pool is live.
            let command = unsafe { ash.allocate_command_buffers(&allocate) }.expect("a command")[0];
            // SAFETY: the command buffer is new; the buffer is held until the pool goes.
            unsafe {
                ash.begin_command_buffer(command, &vk::CommandBufferBeginInfo::default())
                    .expect("a begin");
                ash.cmd_fill_buffer(command, vk_target, offset, 256, 7);
                ash.end_command_buffer(command).expect("an end");
            }
            // SAFETY: the device is live.
            let fence = unsafe { ash.create_fence(&vk::FenceCreateInfo::default(), None) }
                .expect("a fence");
            let submits = || {
                for _ in 0..SUBMITS {
                    let commands = [vk::CommandBufferSubmitInfo::default().command_buffer(command)];
                    let submit = [vk::SubmitInfo2::default().command_buffer_infos(&commands)];
                    device.with_present_queue(|queue| {
                        // SAFETY: the command buffer is executable and not pending (the fence
                        // waited for its last submission); the queue is held.
                        unsafe { ash.queue_submit2(queue, &submit, fence) }.expect("a submit")
                    });
                    // SAFETY: the fence was submitted.
                    unsafe {
                        ash.wait_for_fences(&[fence], true, u64::MAX)
                            .expect("a wait");
                        ash.reset_fences(&[fence]).expect("a reset");
                    }
                }
            };
            for _ in 0..ROUNDS {
                submits();
                let exports: Vec<_> = (0..EXPORTS)
                    .map(|_| (device.new_exported_block(handle, 1 << 16)).expect("an export"))
                    .collect();
                assert_eq!(device.counters().gauges().exports, EXPORTS as u64, "{name}");
                submits();
                drop(exports);
                within("the exports' frees", || {
                    device.counters().gauges().exports == 0
                });
            }
            // SAFETY: nothing is pending.
            unsafe {
                ash.destroy_fence(fence, None);
                ash.destroy_command_pool(pool, None);
            }
            drop(target);
            println!(
                "export: {name}: {handle:?}: {ROUNDS} rounds of {SUBMITS} submits with no export \
                 and with {EXPORTS} live"
            );
        });
    }
}
