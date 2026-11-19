//! Exported surfaces, under a device cfg, on every device of the cfg's kind: for each handle the
//! device exports and each picture (1080p 8-bit: R8 and RG8 at half size; 4K 16-bit: R16 and
//! RG16 at half size), the product's VA-API row (a dma-buf at 256-byte pitches and starts, each
//! plane holding the coded picture's rows: 8 and 4 rows below the 1080p planes, none below the
//! 4K ones), its NVIDIA row (an opaque handle at a pitch multiple of 4, each plane directly below
//! the one before) and a copy-form row (an opaque handle at a pitch multiple of 4096, which no
//! driver's fixed pitch of these pictures meets). Each surface's layout is checked against the
//! rule (plane 0 at 0, one pitch, exact starts); a second device of the same physical device
//! imports the handle and writes three pictures, each acquired by rt and read through its plane
//! textures by a blit into a Shared buffer whose bytes the host compares (a consumer's CPU read of
//! a decoded picture); rt copies a picture into the planes and releases them, and the second
//! device reads it through the handle (the encode round); rt clears each plane in a render pass,
//! reads it, and releases it to the second device's read (the render round). Then the buffer
//! round: the surface's buffer, made at its first call and counted in the surface gauges (none in
//! the copy form, whose buffer is its block's), reads two pictures the second device wrote (the
//! first through the buffer's own acquire at its make, the second through the next
//! `external_writes_done`), and a negative control of the layout (the same bytes read one byte
//! off) finds them wrong; a picture blitted into the buffer is read through the planes, and
//! released, through the handle. On the 1080p picture, where `llvm-as` assembles the examples'
//! library, a kernel writes a picture through the buffer and the encoder's next kernel reads
//! plane 0 through a view: its texels exact, and in place one barrier more than a control whose
//! first kernel writes a buffer of its own, the ordering a resource outside the surface's unit
//! would not take. The memory type the export names, with its flags, the make's wall time per
//! surface, and whether the device's buffers import a dma-buf are printed.
//!
//! The second device stands in for the other API, as in the export tests: a device of another
//! instance cannot run under the validation layer.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::buffer::BufferObject;
    use crate::tests::command::on_device::{each_device, finished, within};
    use crate::tests::devices_of_the_cfgs_kind;
    use crate::{
        BufferLayout, ClearColor, CommandQueue, ComputePipelineDescriptor, ComputePipelineState,
        Device, DispatchType, ExportHandle, FunctionDescriptor, IndexClass, Library, LoadAction,
        MemoryClass, PixelFormat, PlaneLayout, PlaneSpec, RenderPassDescriptor, StageSource,
        Surface, SurfaceCopy, SurfaceExport, SurfaceForm, SurfaceSpec, TextureDescriptor,
        TexturePlace, TextureType, TextureUsage,
    };
    use ash::vk;
    use metal2vulkan_devtools::{fixtures_dir, llvm_assemble, AssembleError};
    use std::time::Instant;

    /// A plane of a picture: its format, size and texel bytes.
    #[derive(Clone, Copy)]
    struct Plane {
        format: PixelFormat,
        width: u32,
        height: u32,
        texel: u64,
    }

    impl Plane {
        fn row(&self) -> u64 {
            u64::from(self.width) * self.texel
        }
    }

    const PICTURES: [(&str, [Plane; 2]); 2] = [
        (
            "1080p 8-bit",
            [
                Plane {
                    format: PixelFormat::R8Unorm,
                    width: 1920,
                    height: 1080,
                    texel: 1,
                },
                Plane {
                    format: PixelFormat::RG8Unorm,
                    width: 960,
                    height: 540,
                    texel: 2,
                },
            ],
        ),
        (
            "4K 16-bit",
            [
                Plane {
                    format: PixelFormat::R16Unorm,
                    width: 3840,
                    height: 2160,
                    texel: 2,
                },
                Plane {
                    format: PixelFormat::RG16Unorm,
                    width: 1920,
                    height: 1080,
                    texel: 4,
                },
            ],
        ),
    ];

    /// A row of the test: its name, handle and multiples, and whether each plane holds the coded
    /// picture's rows (the luma's height rounded up to 16, the chroma half that), as a VA-API
    /// decoder writes them, or only the picture's.
    struct Row {
        name: &'static str,
        handle: ExportHandle,
        pitch_multiple: u64,
        start_multiple: u64,
        coded: bool,
    }

    impl Row {
        /// Each plane's rows below it, of picture `planes`.
        fn rows_below(&self, planes: &[Plane; 2]) -> [u32; 2] {
            if !self.coded {
                return [0, 0];
            }
            let rows = planes[0].height.next_multiple_of(16);
            [rows - planes[0].height, rows / 2 - planes[1].height]
        }
    }

    /// The opaque handle of this host.
    const OPAQUE: ExportHandle = match cfg!(windows) {
        true => ExportHandle::OpaqueWin32,
        false => ExportHandle::OpaqueFd,
    };

    const ROWS: [Row; 3] = [
        Row {
            name: "VA-API",
            handle: ExportHandle::DmaBuf,
            pitch_multiple: 256,
            start_multiple: 256,
            coded: true,
        },
        Row {
            name: "NVIDIA",
            handle: OPAQUE,
            pitch_multiple: 4,
            start_multiple: 1,
            coded: false,
        },
        Row {
            name: "copy-form",
            handle: OPAQUE,
            pitch_multiple: 4096,
            start_multiple: 1,
            coded: false,
        },
    ];

    /// The byte at row `y`, byte `b` of plane `i` of picture `seed`.
    fn byte(seed: u32, i: usize, y: u32, b: u64) -> u8 {
        (seed as u64 * 37 + i as u64 * 101 + u64::from(y) * 7 + b * 3) as u8
    }

    /// Writes picture `seed` into `out` at the surface's layout.
    fn lay_picture(seed: u32, planes: &[Plane; 2], layout: &[PlaneLayout], out: &mut [u8]) {
        for (i, (plane, layout)) in planes.iter().zip(layout).enumerate() {
            for y in 0..plane.height {
                let at = (layout.offset + u64::from(y) * layout.row_pitch) as usize;
                for b in 0..plane.row() {
                    out[at + b as usize] = byte(seed, i, y, b);
                }
            }
        }
    }

    /// The first byte of `bytes`, laid as `offsets` and `pitch` say, that is not picture `seed`'s.
    fn wrong(
        seed: u32,
        planes: &[Plane; 2],
        (offsets, pitch): ([u64; 2], [u64; 2]),
        bytes: &[u8],
    ) -> Option<(usize, u32, u64)> {
        for (i, plane) in planes.iter().enumerate() {
            for y in 0..plane.height {
                let at = (offsets[i] + u64::from(y) * pitch[i]) as usize;
                for b in 0..plane.row() {
                    if bytes[at + b as usize] != byte(seed, i, y, b) {
                        return Some((i, y, b));
                    }
                }
            }
        }
        None
    }

    /// A Shared buffer of `len` bytes on `device`, and its bytes.
    fn shared(device: &Device, len: u64) -> (crate::Block, crate::Buffer) {
        let block = device
            .take_block(MemoryClass::Shared, len)
            .expect("a block");
        let buffer = device.new_buffer(block.id, 0, len, None).expect("a buffer");
        (block, buffer)
    }

    /// `buffer`'s bytes.
    fn bytes_of(buffer: &crate::Buffer) -> &[u8] {
        let contents = buffer.contents().expect("Shared");
        // SAFETY: the buffer's bytes, which no pending command reaches.
        unsafe { std::slice::from_raw_parts(contents.as_ptr(), buffer.length() as usize) }
    }

    /// Runs `f` on `buffer`'s bytes to write them.
    fn fill(buffer: &crate::Buffer, f: impl FnOnce(&mut [u8])) {
        let contents = buffer.contents().expect("Shared");
        // SAFETY: the buffer's bytes, which no pending command reaches and nothing else borrows.
        f(unsafe { std::slice::from_raw_parts_mut(contents.as_ptr(), buffer.length() as usize) })
    }

    /// The other API's buffer over the surface's memory.
    struct Imported {
        buffer: vk::Buffer,
        memory: vk::DeviceMemory,
        len: u64,
    }

    /// The surface's handle imported into `other` as a buffer of the export's size, bound to an
    /// allocation dedicated to it.
    fn import(other: &Device, handle: ExportHandle, export: SurfaceExport) -> Imported {
        let raw = other.raw();
        let ash = raw.device;
        let len = export.export.size;
        let mut external = vk::ExternalMemoryBufferCreateInfo::default().handle_types(handle.vk());
        let info = vk::BufferCreateInfo::default()
            .size(len)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .push_next(&mut external);
        // SAFETY: the device's buffers import the handle, as they export it.
        let buffer = unsafe { ash.create_buffer(&info, None) }.expect("the import's buffer");
        // SAFETY: the buffer is live.
        let needs = unsafe { ash.get_buffer_memory_requirements(buffer) };
        let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().buffer(buffer);
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
                            export.export.handle.as_raw_fd(),
                            &mut properties,
                        )
                    }
                    .expect("the dma-buf's memory types");
                    properties.memory_type_bits
                }
                _ => 1 << export.export.memory_type,
            };
            let types = types & needs.memory_type_bits;
            let index = match types & (1 << export.export.memory_type) {
                0 => types.trailing_zeros(),
                _ => export.export.memory_type,
            };
            assert!(index < 32, "the import takes a type the buffer lists");
            let fd = export.export.handle.into_raw_fd();
            let mut import = vk::ImportMemoryFdInfoKHR::default()
                .handle_type(handle.vk())
                .fd(fd);
            let info = vk::MemoryAllocateInfo::default()
                .allocation_size(len.max(needs.size))
                .memory_type_index(index)
                .push_next(&mut import)
                .push_next(&mut dedicated);
            // SAFETY: an import of the fd, dedicated to the buffer; a successful import owns it.
            unsafe { ash.allocate_memory(&info, None) }.expect("the import")
        };
        #[cfg(windows)]
        let memory = {
            use std::os::windows::io::AsRawHandle;
            let mut import = vk::ImportMemoryWin32HandleInfoKHR::default()
                .handle_type(handle.vk())
                .handle(export.export.handle.as_raw_handle() as vk::HANDLE);
            let info = vk::MemoryAllocateInfo::default()
                .allocation_size(len.max(needs.size))
                .memory_type_index(export.export.memory_type)
                .push_next(&mut import)
                .push_next(&mut dedicated);
            // SAFETY: an import of the NT handle, dedicated to the buffer.
            unsafe { ash.allocate_memory(&info, None) }.expect("the import")
        };
        // SAFETY: the memory is the buffer's dedicated allocation.
        unsafe { ash.bind_buffer_memory(buffer, memory, 0) }.expect("the import's binding");
        Imported {
            buffer,
            memory,
            len,
        }
    }

    /// A barrier of the whole import from queue family `from` to `to`.
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

    /// The other API's side: its device, queue and import.
    struct Other {
        device: Device,
        queue: CommandQueue,
        import: Imported,
        external: u32,
    }

    impl Other {
        /// Writes `bytes`'s whole length into the import, then releases it.
        fn write(&self, name: &str, bytes: &crate::Buffer) {
            let (from, offset) = bytes.device_range();
            let family = self.device.raw().queue_family_index;
            let (import, len, external) = (self.import.buffer, self.import.len, self.external);
            let command_buffer = self.queue.command_buffer().expect("a command buffer");
            command_buffer
                .record_for_test(&[bytes], |raw, command| {
                    let copy = [vk::BufferCopy {
                        src_offset: offset,
                        dst_offset: 0,
                        size: len,
                    }];
                    let release = [transfer(import, family, external, true)];
                    // SAFETY: the command buffer is recording; the import outlives the wait.
                    unsafe {
                        raw.cmd_copy_buffer(command, from, import, &copy);
                        raw.cmd_pipeline_barrier2(
                            command,
                            &vk::DependencyInfo::default().buffer_memory_barriers(&release),
                        );
                    }
                })
                .expect("the write");
            command_buffer.commit().expect("a commit");
            finished(name, &command_buffer);
            assert_eq!(command_buffer.error(), None, "{name}");
        }

        /// Acquires the import and reads its whole length into `bytes`.
        fn read(&self, name: &str, bytes: &crate::Buffer) {
            let (to, offset) = bytes.device_range();
            let family = self.device.raw().queue_family_index;
            let (import, len, external) = (self.import.buffer, self.import.len, self.external);
            let command_buffer = self.queue.command_buffer().expect("a command buffer");
            command_buffer
                .record_for_test(&[bytes], |raw, command| {
                    let acquire = [transfer(import, external, family, false)];
                    let copy = [vk::BufferCopy {
                        src_offset: 0,
                        dst_offset: offset,
                        size: len,
                    }];
                    // SAFETY: the command buffer is recording; the import outlives the wait.
                    unsafe {
                        raw.cmd_pipeline_barrier2(
                            command,
                            &vk::DependencyInfo::default().buffer_memory_barriers(&acquire),
                        );
                        raw.cmd_copy_buffer(command, import, to, &copy);
                    }
                })
                .expect("the read");
            command_buffer.commit().expect("a commit");
            finished(name, &command_buffer);
            assert_eq!(command_buffer.error(), None, "{name}");
        }
    }

    impl Drop for Other {
        fn drop(&mut self) {
            let ash = self.device.raw().device;
            // SAFETY: every command buffer that used the import has completed.
            unsafe {
                ash.destroy_buffer(self.import.buffer, None);
                ash.free_memory(self.import.memory, None);
            }
        }
    }

    /// The tightly packed bytes of each plane's texels, as rt's blit reads them: each plane's
    /// offset in a buffer and its pitch.
    fn packed(planes: &[Plane; 2]) -> ([u64; 2], [u64; 2], u64) {
        let sizes = planes.map(|p| p.row() * u64::from(p.height));
        ([0, sizes[0]], planes.map(|p| p.row()), sizes[0] + sizes[1])
    }

    /// rt reads each plane through its texture into `out`, packed.
    fn rt_read(
        name: &str,
        queue: &CommandQueue,
        surface: &Surface,
        planes: &[Plane; 2],
        out: &crate::Buffer,
    ) {
        let (offsets, pitches, _) = packed(planes);
        let command_buffer = queue.command_buffer().expect("a command buffer");
        let mut encoder = command_buffer.blit_command_encoder().expect("an encoder");
        for (i, (texture, plane)) in surface.planes().iter().zip(planes).enumerate() {
            let layout = BufferLayout {
                offset: offsets[i],
                bytes_per_row: pitches[i],
                bytes_per_image: 0,
            };
            (encoder.copy_from_texture_to_buffer(
                texture,
                TexturePlace::default(),
                [plane.width, plane.height, 1],
                out,
                layout,
            ))
            .unwrap_or_else(|error| panic!("{name}: the read: {error}"));
        }
        encoder.end_encoding();
        command_buffer.commit().expect("a commit");
        finished(name, &command_buffer);
        assert_eq!(command_buffer.error(), None, "{name}");
    }

    /// Checks the surface's layout against the rule.
    fn check_layout(name: &str, row: &Row, planes: &[Plane; 2], export: &SurfaceExport) {
        let [p0, p1] = [export.planes[0], export.planes[1]];
        assert_eq!(p0.offset, 0, "{name}: plane 0 at offset 0");
        assert_eq!(p0.row_pitch, p1.row_pitch, "{name}: one pitch");
        let pitch = p0.row_pitch;
        assert!(
            pitch.is_multiple_of(row.pitch_multiple),
            "{name}: pitch {pitch}"
        );
        assert!(
            pitch >= planes[0].row().max(planes[1].row()),
            "{name}: pitch {pitch}"
        );
        let rows_below = row.rows_below(planes);
        let below = pitch * u64::from(planes[0].height + rows_below[0]);
        let start = below.next_multiple_of(row.start_multiple.max(planes[1].texel));
        assert_eq!(p1.offset, start, "{name}: plane 1's exact start");
        let end = p1.offset + pitch * u64::from(planes[1].height + rows_below[1]);
        assert!(
            export.export.size >= end,
            "{name}: the size holds the planes"
        );
    }

    /// For each handle the device exports and each picture, each row's surface: its layout, three
    /// pictures written by the other device and read by rt, the encode round and the render
    /// round, each byte compared; the form and the make's wall time printed.
    #[test]
    fn every_device_carries_each_form_of_surface() {
        each_device("surface", |name, instance, device| {
            let counters = device.counters();
            let physical = (devices_of_the_cfgs_kind(instance).into_iter())
                .map(|(_, physical)| physical)
                .find(|physical| physical.raw() == device.raw().physical_device)
                .expect("the device's physical device");
            let queue = device.new_command_queue(4).expect("a queue");
            let raw = device.raw();
            // SAFETY: the physical device is the instance's.
            let kinds = unsafe {
                (raw.instance).get_physical_device_memory_properties(raw.physical_device)
            };
            let mut in_place_pitch = [None; 2];
            #[cfg(unix)]
            println!(
                "surface: {name}: dma-buf buffer import offered: {}",
                device.core().imports.dma_buf.is_some()
            );
            let kernels = kernels(name, &device);
            for row in &ROWS {
                let handle = row.handle;
                let foreign = device
                    .extensions()
                    .iter()
                    .any(|n| n.as_c_str() == ash::ext::queue_family_foreign::NAME);
                if !device.core().exports.offers(handle) || (handle.foreign() && !foreign) {
                    println!("surface: {name}: {} {handle:?} not exported", row.name);
                    continue;
                }
                for (at, (picture, planes)) in PICTURES.iter().enumerate() {
                    let label = format!("{name}: {} {handle:?} {picture}", row.name);
                    let specs = [0, 1].map(|i| {
                        let p = planes[i];
                        let mut descriptor = TextureDescriptor::new(
                            TextureType::Type2D,
                            p.format,
                            p.width,
                            p.height,
                        );
                        // Read by shaders through views alone: no index of its own.
                        descriptor.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
                        PlaneSpec {
                            descriptor,
                            index: None,
                            rows_below: row.rows_below(planes)[i],
                        }
                    });
                    let spec = SurfaceSpec {
                        handle,
                        pitch_multiple: row.pitch_multiple,
                        start_multiple: row.start_multiple,
                        planes: &specs,
                    };
                    let before = counters.metrics();
                    let began = Instant::now();
                    let (surface, export) = (device.new_exported_surface(&spec))
                        .unwrap_or_else(|error| panic!("{label}: {error}"));
                    let made = began.elapsed();
                    let form = export.form;
                    let after = counters.metrics();
                    match form {
                        SurfaceForm::InPlace => {
                            assert_eq!(
                                after.surfaces_in_place,
                                before.surfaces_in_place + 1,
                                "{label}"
                            );
                        }
                        SurfaceForm::Copy(_) => {
                            assert_eq!(
                                after.surfaces_copied,
                                before.surfaces_copied + 1,
                                "{label}"
                            );
                        }
                    }
                    let index = export.export.memory_type;
                    let flags =
                        flag_names(kinds.memory_types_as_slice()[index as usize].property_flags);
                    println!(
                        "surface: {label}: form {form:?}, pitch {}, plane 1 at {}, size {}, memory type {index} ({flags}), made in {:.3} ms",
                        export.planes[0].row_pitch,
                        export.planes[1].offset,
                        export.export.size,
                        made.as_secs_f64() * 1e3
                    );
                    check_layout(&label, row, planes, &export);
                    if row.name == "NVIDIA" && form == SurfaceForm::InPlace {
                        in_place_pitch[at] = Some(export.planes[0].row_pitch);
                    }
                    if row.name == "copy-form" {
                        match in_place_pitch[at] {
                            Some(pitch) if !pitch.is_multiple_of(row.pitch_multiple) => {
                                assert!(
                                    matches!(form, SurfaceForm::Copy(SurfaceCopy::Pitch { plane: 0 })),
                                    "{label}: the driver's pitch {pitch} meets no 4096 multiple: {form:?}"
                                );
                            }
                            _ => {}
                        }
                    }
                    let layout = export.planes.clone();
                    let total = export.export.size;
                    let other = Device::new(&physical, Default::default())
                        .unwrap_or_else(|error| panic!("{label}: the second device: {error}"));
                    let other_queue = other.new_command_queue(4).expect("a queue");
                    let imported = import(&other, handle, export);
                    let other = Other {
                        device: other,
                        queue: other_queue,
                        import: imported,
                        external: handle.family(),
                    };
                    let (_theirs_block, theirs) = shared(&other.device, total);
                    let (_mine_block, mine) = shared(&device, packed(planes).2);
                    let (offsets, pitches, _) = packed(planes);
                    let at_layout = (
                        [layout[0].offset, layout[1].offset],
                        [layout[0].row_pitch; 2],
                    );

                    // Three pictures, each written by the other API and read by rt.
                    let acquires = counters.metrics().external_acquires;
                    for seed in 1..=3 {
                        fill(&theirs, |out| lay_picture(seed, planes, &layout, out));
                        other.write(&label, &theirs);
                        (surface.external_writes_done())
                            .unwrap_or_else(|error| panic!("{label}: the acquire: {error}"));
                        rt_read(&label, &queue, &surface, planes, &mine);
                        assert_eq!(
                            wrong(seed, planes, (offsets, pitches), bytes_of(&mine)),
                            None,
                            "{label}: picture {seed} read by rt"
                        );
                    }
                    assert_eq!(
                        counters.metrics().external_acquires,
                        acquires + 3,
                        "{label}"
                    );

                    // The encode round: rt copies a picture into the planes and releases them.
                    let (_source_block, source) = shared(&device, packed(planes).2);
                    fill(&source, |out| {
                        for (i, plane) in planes.iter().enumerate() {
                            for y in 0..plane.height {
                                for b in 0..plane.row() {
                                    let at = offsets[i] + u64::from(y) * pitches[i] + b;
                                    out[at as usize] = byte(9, i, y, b);
                                }
                            }
                        }
                    });
                    let releases = counters.metrics().external_releases;
                    let command_buffer = queue.command_buffer().expect("a command buffer");
                    let mut encoder = command_buffer.blit_command_encoder().expect("an encoder");
                    for (i, (texture, plane)) in surface.planes().iter().zip(planes).enumerate() {
                        let source_layout = BufferLayout {
                            offset: offsets[i],
                            bytes_per_row: pitches[i],
                            bytes_per_image: 0,
                        };
                        (encoder.copy_from_buffer(
                            &source,
                            source_layout,
                            [plane.width, plane.height, 1],
                            texture,
                            TexturePlace::default(),
                        ))
                        .unwrap_or_else(|error| panic!("{label}: the encode copy: {error}"));
                    }
                    (encoder.release_surface_to_external(&surface))
                        .unwrap_or_else(|error| panic!("{label}: the release: {error}"));
                    encoder.end_encoding();
                    command_buffer.commit().expect("a commit");
                    finished(&label, &command_buffer);
                    assert_eq!(command_buffer.error(), None, "{label}");
                    assert_eq!(
                        counters.metrics().external_releases,
                        releases + 1,
                        "{label}"
                    );
                    other.read(&label, &theirs);
                    assert_eq!(
                        wrong(9, planes, at_layout, bytes_of(&theirs)),
                        None,
                        "{label}: the encode round read through the handle"
                    );

                    // The render round: rt clears each plane in a render pass, reads it, and
                    // releases it to the other API's read.
                    let value = |plane: &Plane, i: usize| match plane.texel
                        / u64::from(plane_channels(plane))
                    {
                        1 => [51.0 / 255.0, 204.0 / 255.0][i % 2],
                        _ => [13107.0 / 65535.0, 52428.0 / 65535.0][i % 2],
                    };
                    let command_buffer = queue.command_buffer().expect("a command buffer");
                    for (i, (texture, plane)) in surface.planes().iter().zip(planes).enumerate() {
                        let mut pass = RenderPassDescriptor::default();
                        let c = &mut pass.color_attachments[0];
                        c.texture = Some(texture.clone());
                        c.load_action = LoadAction::Clear;
                        let v = value(plane, i);
                        c.clear_color = ClearColor {
                            red: v,
                            green: v,
                            blue: 0.0,
                            alpha: 1.0,
                        };
                        let encoder = (command_buffer.render_command_encoder(&pass))
                            .unwrap_or_else(|error| panic!("{label}: the pass: {error}"));
                        encoder.end_encoding();
                    }
                    command_buffer.commit().expect("a commit");
                    finished(&label, &command_buffer);
                    assert_eq!(command_buffer.error(), None, "{label}");
                    let cleared = |i: usize, plane: &Plane| -> Vec<u8> {
                        let v = value(plane, i);
                        let channel: Vec<u8> = match plane.texel / u64::from(plane_channels(plane))
                        {
                            1 => vec![(v * 255.0).round() as u8],
                            _ => ((v * 65535.0).round() as u16).to_le_bytes().to_vec(),
                        };
                        channel.repeat(plane_channels(plane) as usize)
                    };
                    rt_read(&label, &queue, &surface, planes, &mine);
                    let ours = bytes_of(&mine);
                    for (i, plane) in planes.iter().enumerate() {
                        let texel = cleared(i, plane);
                        let size = (plane.row() * u64::from(plane.height)) as usize;
                        let span = &ours[offsets[i] as usize..offsets[i] as usize + size];
                        assert!(
                            span.chunks(texel.len()).all(|t| t == texel.as_slice()),
                            "{label}: plane {i} rendered, read by rt"
                        );
                    }
                    let command_buffer = queue.command_buffer().expect("a command buffer");
                    let mut encoder = command_buffer.blit_command_encoder().expect("an encoder");
                    (encoder.release_surface_to_external(&surface))
                        .unwrap_or_else(|error| panic!("{label}: the release: {error}"));
                    encoder.end_encoding();
                    command_buffer.commit().expect("a commit");
                    finished(&label, &command_buffer);
                    other.read(&label, &theirs);
                    let read = bytes_of(&theirs);
                    for (i, plane) in planes.iter().enumerate() {
                        let texel = cleared(i, plane);
                        for y in 0..plane.height {
                            let at =
                                (layout[i].offset + u64::from(y) * layout[i].row_pitch) as usize;
                            let span = &read[at..at + plane.row() as usize];
                            assert!(
                                span.chunks(texel.len()).all(|t| t == texel.as_slice()),
                                "{label}: plane {i} row {y} rendered, read through the handle"
                            );
                        }
                    }
                    let in_place = form == SurfaceForm::InPlace;
                    let round = Round {
                        label: &label,
                        device: &device,
                        queue: &queue,
                        surface: &surface,
                        in_place,
                        planes,
                        layout: &layout,
                    };
                    let buffer = buffer_round(&round, &other);
                    if let (0, Some(kernels)) = (at, &kernels) {
                        compute_round(&round, &buffer, kernels);
                    }
                    drop(buffer);
                    println!(
                        "surface: {label}: three pictures, the encode round, the render round and the buffer round exact"
                    );
                    drop(other);
                    drop(surface);
                }
            }
            drop(queue);
            within("the surfaces' frees", || {
                let gauges = counters.gauges();
                gauges.surface_imports == 0 && gauges.exports == 0 && gauges.surface_buffers == 0
            });
        });
    }

    /// A surface of a row and a picture, as the buffer round and the compute round take it.
    #[derive(Clone, Copy)]
    struct Round<'a> {
        label: &'a str,
        device: &'a Device,
        queue: &'a CommandQueue,
        surface: &'a Surface,
        /// The surface took the form in place.
        in_place: bool,
        planes: &'a [Plane; 2],
        layout: &'a [PlaneLayout],
    }

    /// The buffer round on the surface, which the other API holds released: its buffer reads two
    /// pictures the other API wrote, one off its layout is wrong, and a picture blitted into it
    /// is read through the planes and through the handle. The buffer.
    fn buffer_round(round: &Round<'_>, other: &Other) -> crate::Buffer {
        let Round {
            label,
            device,
            queue,
            surface,
            in_place,
            planes,
            layout,
        } = *round;
        let counters = device.counters();
        let at_layout = (
            [layout[0].offset, layout[1].offset],
            [layout[0].row_pitch; 2],
        );
        let (_theirs_block, theirs) = shared(&other.device, other.import.len);
        let (_mine_block, mine) = shared(device, packed(planes).2);
        let (offsets, pitches, _) = packed(planes);

        // Picture 4, written before the buffer's make: its own acquire reads it.
        fill(&theirs, |out| lay_picture(4, planes, layout, out));
        other.write(label, &theirs);
        (surface.external_writes_done())
            .unwrap_or_else(|error| panic!("{label}: the acquire: {error}"));
        let (gauges, acquires) = (counters.gauges(), counters.metrics().external_acquires);
        let buffer =
            (surface.buffer()).unwrap_or_else(|error| panic!("{label}: the buffer: {error}"));
        // In place the export's size; in the copy form the block's length, which holds the planes.
        let total = buffer.length();
        let end = layout[1].offset + layout[1].row_pitch * u64::from(planes[1].height);
        assert!(
            total <= other.import.len && total >= end,
            "{label}: the buffer's length {total}"
        );
        if in_place {
            assert_eq!(
                total, other.import.len,
                "{label}: the buffer spans the export"
            );
        }
        let (_whole_block, whole) = shared(device, total);
        let read_whole = |buffer: &crate::Buffer| {
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = command_buffer.blit_command_encoder().expect("an encoder");
            (encoder.copy_from_buffer_to_buffer(buffer, 0, &whole, 0, total))
                .unwrap_or_else(|error| panic!("{label}: the buffer's read: {error}"));
            encoder.end_encoding();
            command_buffer.commit().expect("a commit");
            finished(label, &command_buffer);
            assert_eq!(command_buffer.error(), None, "{label}");
        };
        let made = counters.gauges();
        assert_eq!(
            made.surface_buffers,
            gauges.surface_buffers + u64::from(in_place),
            "{label}: the buffer in place counted"
        );
        #[cfg(unix)]
        assert_eq!(
            made.surface_imports,
            gauges.surface_imports
                + u64::from(in_place && surface.handle() == ExportHandle::DmaBuf),
            "{label}: a dma-buf's buffer's import counted"
        );
        assert_eq!(
            counters.metrics().external_acquires,
            acquires + u64::from(in_place),
            "{label}: the buffer's acquire at its make"
        );
        assert!(
            buffer.contents().is_none(),
            "{label}: the buffer is Private"
        );
        let again =
            (surface.buffer()).unwrap_or_else(|error| panic!("{label}: the buffer: {error}"));
        assert_eq!(again.id(), buffer.id(), "{label}: one buffer");
        assert_eq!(
            counters.gauges().surface_buffers,
            made.surface_buffers,
            "{label}: made once"
        );
        drop(again);
        read_whole(&buffer);
        assert_eq!(
            wrong(4, planes, at_layout, bytes_of(&whole)),
            None,
            "{label}: picture 4 read through the buffer"
        );
        // The negative control of the layout: the same bytes one byte off are wrong.
        let off = ([at_layout.0[0] + 1, at_layout.0[1]], at_layout.1);
        assert!(
            wrong(4, planes, off, bytes_of(&whole)).is_some(),
            "{label}: the check finds a wrong offset"
        );

        // Picture 5, acquired with the planes.
        fill(&theirs, |out| lay_picture(5, planes, layout, out));
        other.write(label, &theirs);
        (surface.external_writes_done())
            .unwrap_or_else(|error| panic!("{label}: the acquire: {error}"));
        read_whole(&buffer);
        assert_eq!(
            wrong(5, planes, at_layout, bytes_of(&whole)),
            None,
            "{label}: picture 5 read through the buffer"
        );

        // Picture 6, blitted into the buffer, read through the planes in the same encoder, then
        // released and read through the handle.
        fill(&whole, |out| lay_picture(6, planes, layout, out));
        let command_buffer = queue.command_buffer().expect("a command buffer");
        let mut encoder = command_buffer.blit_command_encoder().expect("an encoder");
        (encoder.copy_from_buffer_to_buffer(&whole, 0, &buffer, 0, total))
            .unwrap_or_else(|error| panic!("{label}: the buffer's write: {error}"));
        for (i, (texture, plane)) in surface.planes().iter().zip(planes).enumerate() {
            let at = BufferLayout {
                offset: offsets[i],
                bytes_per_row: pitches[i],
                bytes_per_image: 0,
            };
            (encoder.copy_from_texture_to_buffer(
                texture,
                TexturePlace::default(),
                [plane.width, plane.height, 1],
                &mine,
                at,
            ))
            .unwrap_or_else(|error| panic!("{label}: the planes' read: {error}"));
        }
        (encoder.release_surface_to_external(surface))
            .unwrap_or_else(|error| panic!("{label}: the release: {error}"));
        encoder.end_encoding();
        command_buffer.commit().expect("a commit");
        finished(label, &command_buffer);
        assert_eq!(command_buffer.error(), None, "{label}");
        assert_eq!(
            wrong(6, planes, (offsets, pitches), bytes_of(&mine)),
            None,
            "{label}: picture 6 written through the buffer, read through the planes"
        );
        other.read(label, &theirs);
        assert_eq!(
            wrong(6, planes, at_layout, bytes_of(&theirs)),
            None,
            "{label}: picture 6 written through the buffer, read through the handle"
        );
        println!("surface: {label}: the buffer round exact (in place: {in_place})");
        buffer
    }

    /// The examples' kernels: `add` (`o = a + b`) and `invert_texels`, or None where `llvm-as` is
    /// missing.
    fn kernels(
        name: &str,
        device: &Device,
    ) -> Option<(ComputePipelineState, ComputePipelineState)> {
        let ll = fixtures_dir().join("examples/runtime_examples.ll");
        let library = match llvm_assemble(&ll) {
            Ok(bitcode) => Library::from_air(&bitcode).expect("the library"),
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: the surface's compute round: {why}");
                return None;
            }
            Err(error) => panic!("{}: {error:?}", ll.display()),
        };
        let function = library.function("invert_texels").expect("the function");
        let descriptor = ComputePipelineDescriptor::new(StageSource::Function(
            FunctionDescriptor::new(function),
        ));
        let invert = (device
            .new_compute_pipeline_state(&descriptor, crate::tests::no_constexpr_samplers))
        .unwrap_or_else(|error| panic!("{name}: {error}"));
        Some((crate::tests::compute::on_device::add(device), invert))
    }

    /// The compute round on the 1080p picture: `add` writes picture 7 through the surface's
    /// buffer, and the encoder's next dispatch reads plane 0 through a view (`invert_texels`, into
    /// a texture of its own): every texel the inverse of picture 7's luma, and the planes read
    /// picture 7. In place, the pair takes one barrier more than a control whose `add` writes a
    /// buffer of its own, the ordering of the buffer's write before the plane's read.
    fn compute_round(
        round: &Round<'_>,
        buffer: &crate::Buffer,
        (add, invert): &(ComputePipelineState, ComputePipelineState),
    ) {
        let Round {
            label,
            device,
            queue,
            surface,
            in_place,
            planes,
            layout,
        } = *round;
        let counters = device.counters();
        // The surface is the runtime's again (the other API wrote nothing since its release).
        (surface.external_writes_done())
            .unwrap_or_else(|error| panic!("{label}: the acquire: {error}"));
        let total = buffer.length();
        let words = u32::try_from(total / 4).expect("words");
        let (_zeros_block, zeros) = shared(device, total);
        let (_pattern_block, pattern) = shared(device, total);
        let (_elsewhere_block, elsewhere) = shared(device, total);
        fill(&zeros, |out| out.fill(0));
        fill(&pattern, |out| {
            out.fill(0);
            lay_picture(7, planes, layout, out);
        });
        let first = IndexClass::Texture.range().start;
        let _grant = (device.grant_indices(IndexClass::Texture, first, 2)).expect("a grant");
        let [width, height] = [planes[0].width, planes[0].height];
        let view = (surface.planes()[0].new_texture_view(
            planes[0].format,
            TextureType::Type2D,
            0..1,
            0..1,
            Some(first),
        ))
        .unwrap_or_else(|error| panic!("{label}: the plane's view: {error}"));
        let mut d =
            TextureDescriptor::new(TextureType::Type2D, PixelFormat::RGBA8Unorm, width, height);
        d.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
        let out = (device.new_texture(&d, Some(first + 1)))
            .unwrap_or_else(|error| panic!("{label}: the output: {error}"));
        let pair = |target: &crate::Buffer| -> u64 {
            let before = counters.metrics().barriers;
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder =
                (command_buffer.compute_command_encoder(DispatchType::Serial)).expect("an encoder");
            let done = |result: Result<(), crate::Error>| {
                result.unwrap_or_else(|error| panic!("{label}: {error}"))
            };
            done(encoder.set_compute_pipeline_state(add));
            done(encoder.set_buffer(Some(&zeros), 0, 0));
            done(encoder.set_buffer(Some(&pattern), 0, 1));
            done(encoder.set_buffer(Some(target), 0, 2));
            done(encoder.dispatch_threads([words, 1, 1], [64, 1, 1]));
            done(encoder.set_compute_pipeline_state(invert));
            done(encoder.set_texture(Some(&view), 0));
            done(encoder.set_texture(Some(&out), 1));
            done(encoder.dispatch_threads([width, height, 1], [8, 8, 1]));
            encoder.end_encoding();
            command_buffer.commit().expect("a commit");
            finished(label, &command_buffer);
            assert_eq!(command_buffer.error(), None, "{label}");
            counters.metrics().barriers - before
        };
        // The first control takes the textures' and the acquire's uploads; the second is the one
        // compared, after the ordered pair's texels are read.
        pair(&elsewhere);
        let ordered = pair(buffer);
        let (_texels_block, texels) = shared(device, u64::from(width) * u64::from(height) * 4);
        let command_buffer = queue.command_buffer().expect("a command buffer");
        let mut encoder = command_buffer.blit_command_encoder().expect("an encoder");
        let at = BufferLayout {
            offset: 0,
            bytes_per_row: u64::from(width) * 4,
            bytes_per_image: 0,
        };
        (encoder.copy_from_texture_to_buffer(
            &out,
            TexturePlace::default(),
            [width, height, 1],
            &texels,
            at,
        ))
        .unwrap_or_else(|error| panic!("{label}: the output's read: {error}"));
        encoder.end_encoding();
        command_buffer.commit().expect("a commit");
        finished(label, &command_buffer);
        assert_eq!(command_buffer.error(), None, "{label}");
        let bytes = bytes_of(&texels);
        for y in 0..height {
            for x in 0..width {
                let at = ((y * width + x) * 4) as usize;
                let expected = [255 - byte(7, 0, y, u64::from(x)), 255, 255, 255];
                assert_eq!(
                    bytes[at..at + 4],
                    expected,
                    "{label}: texel ({x}, {y}) of the plane a kernel read after a kernel wrote the buffer"
                );
            }
        }
        let (_mine_block, mine) = shared(device, packed(planes).2);
        rt_read(label, queue, surface, planes, &mine);
        let (offsets, pitches, _) = packed(planes);
        assert_eq!(
            wrong(7, planes, (offsets, pitches), bytes_of(&mine)),
            None,
            "{label}: picture 7 written by a kernel through the buffer, read through the planes"
        );
        let control = pair(&elsewhere);
        if in_place {
            assert_eq!(
                ordered,
                control + 1,
                "{label}: one barrier orders the buffer's write before the plane's read"
            );
        }
        println!(
            "surface: {label}: the compute round exact, barriers {ordered} against the control's {control}"
        );
        drop((view, out));
    }

    /// The channels of a plane's format.
    fn plane_channels(plane: &Plane) -> u32 {
        match plane.format {
            PixelFormat::RG8Unorm | PixelFormat::RG16Unorm => 2,
            _ => 1,
        }
    }

    /// A memory type's flags by name, and the bits of any other.
    fn flag_names(flags: vk::MemoryPropertyFlags) -> String {
        use vk::MemoryPropertyFlags as M;
        let named = [
            (M::DEVICE_LOCAL, "DEVICE_LOCAL"),
            (M::HOST_VISIBLE, "HOST_VISIBLE"),
            (M::HOST_COHERENT, "HOST_COHERENT"),
            (M::HOST_CACHED, "HOST_CACHED"),
            (M::LAZILY_ALLOCATED, "LAZILY_ALLOCATED"),
            (M::PROTECTED, "PROTECTED"),
        ];
        let mut names: Vec<String> = (named.iter())
            .filter(|(flag, _)| flags.contains(*flag))
            .map(|(_, name)| (*name).to_owned())
            .collect();
        let rest = (named.iter()).fold(flags.as_raw(), |rest, (flag, _)| rest & !flag.as_raw());
        if rest != 0 {
            names.push(format!("{rest:#x}"));
        }
        match names.is_empty() {
            true => "none".to_owned(),
            false => names.join(" | "),
        }
    }
}

/// Each copy-form reason the layout's verdicts give, by construction from synthetic answers: the
/// ones no lab device produces (`Tiling`, `Dedicated`, `MemoryType`) among them.
mod verdicts {
    use crate::surface::{opaque_pitch, opaque_starts, tiling_verdict, type_of, PlaneShape};
    use crate::SurfaceCopy;
    use ash::vk::{ExternalMemoryFeatureFlags as F, ImageUsageFlags as U};

    const IN_PLACE: U = U::from_raw(
        U::SAMPLED.as_raw()
            | U::COLOR_ATTACHMENT.as_raw()
            | U::TRANSFER_SRC.as_raw()
            | U::TRANSFER_DST.as_raw(),
    );

    const SHAPE: PlaneShape = PlaneShape {
        row_pitch: 2048,
        offset: 0,
        size: 2048 * 1080,
        alignment: 256,
        requires: false,
    };

    #[test]
    fn a_tiling_without_a_usage_or_the_handle_takes_the_copy_form() {
        let both = F::EXPORTABLE | F::IMPORTABLE;
        let tiling = Err(SurfaceCopy::Tiling { plane: 1 });
        assert_eq!(
            tiling_verdict(1, IN_PLACE & !U::COLOR_ATTACHMENT, both),
            tiling
        );
        assert_eq!(tiling_verdict(1, IN_PLACE, F::EXPORTABLE), tiling);
        assert_eq!(tiling_verdict(1, IN_PLACE, F::IMPORTABLE), tiling);
        assert_eq!(
            tiling_verdict(1, IN_PLACE, both | F::DEDICATED_ONLY),
            Err(SurfaceCopy::Dedicated { plane: 1 })
        );
        assert_eq!(tiling_verdict(1, IN_PLACE | U::STORAGE, both), Ok(()));
    }

    #[test]
    fn an_opaque_layout_takes_the_copy_form_where_its_images_do_not_hold_it() {
        let required = PlaneShape {
            requires: true,
            ..SHAPE
        };
        assert_eq!(
            opaque_pitch(&[SHAPE, required], 4),
            Err(SurfaceCopy::Dedicated { plane: 1 })
        );
        let other = PlaneShape {
            row_pitch: 1920,
            ..SHAPE
        };
        assert_eq!(
            opaque_pitch(&[SHAPE, other], 4),
            Err(SurfaceCopy::Pitch { plane: 1 })
        );
        assert_eq!(
            opaque_pitch(&[SHAPE, SHAPE], 4096),
            Err(SurfaceCopy::Pitch { plane: 0 })
        );
        assert_eq!(opaque_pitch(&[SHAPE, SHAPE], 256), Ok(2048));

        let below = 2048 * 1080;
        assert_eq!(opaque_starts(&[SHAPE, SHAPE], &[0, below]), Ok(()));
        let offset = PlaneShape {
            offset: 64,
            ..SHAPE
        };
        assert_eq!(
            opaque_starts(&[SHAPE, offset], &[0, below]),
            Err(SurfaceCopy::Start { plane: 1 })
        );
        let aligned = PlaneShape {
            alignment: 1 << 16,
            ..SHAPE
        };
        assert_eq!(
            opaque_starts(&[SHAPE, aligned], &[0, below]),
            Err(SurfaceCopy::Start { plane: 1 })
        );
        let padded = PlaneShape {
            size: below + 256,
            ..SHAPE
        };
        assert_eq!(
            opaque_starts(&[padded, SHAPE], &[0, below]),
            Err(SurfaceCopy::Start { plane: 1 })
        );
    }

    #[test]
    fn planes_of_no_common_memory_type_take_the_copy_form() {
        assert_eq!(type_of(&[0b0011, 0b1100], 0, 0b1111), None);
        assert_eq!(type_of(&[0b0111, 0b0110], 2, 0b0001), Some(2));
        assert_eq!(type_of(&[0b0110, 0b1110], 0, 0b0100), Some(2));
        assert_eq!(type_of(&[0b1010, 0b1110], 0, 0b0001), Some(1));
    }
}
