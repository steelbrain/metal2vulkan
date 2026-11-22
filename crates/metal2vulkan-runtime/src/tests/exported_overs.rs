//! A texture over a buffer in an exported block, under a device cfg, on every device of the cfg's
//! kind: copied in once after each event that changes its bytes (`crate::over::ExportedOvers`),
//! counted in `Metrics::over_copies` and `Metrics::over_copy_bytes`. After a fill of the buffer and
//! the block's acquire, four submissions that read the texture copy it in once; a fill between
//! makes one copy more, and a zero fill of the block's range one more; reads with no event between
//! copy nothing. Each read finds the bytes of the last event. The block's handle states no address
//! (`BlockInfo::gpu_address` is `None`), so `Metrics::exported_blocks_addressed` reads 0.
//!
//! A write through an address owes such a texture its copy in only once an address of its block
//! went out. The texture, read by three submissions that each then write through an address into
//! another buffer (`store_through`), is copied in by none of them, and the count of blocks whose
//! address went out reads 0; once the buffer the texture is over answers `gpu_address`, it reads
//! 1, and each submission after the next copies the texture in again. A plane of an exported
//! surface in the copy form, read by four such submissions, is copied in once, and the count reads
//! 0; once its buffer's `gpu_address` is read, it reads 1, and each submission after the next
//! copies the plane in again.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::command::on_device::{each_device, finished};
    use crate::{
        Block, Buffer, BufferLayout, CommandQueue, ComputePipelineDescriptor, ComputePipelineState,
        Device, DispatchType, ExportHandle, FunctionDescriptor, IndexClass, Library, MemoryClass,
        PixelFormat, PlaneSpec, Resource, ResourceUsage, StageSource, StorageMode, SurfaceForm,
        SurfaceSpec, Texture, TextureDescriptor, TexturePlace, TextureType, TextureUsage,
    };
    use metal2vulkan_devtools::{llvm_assemble_text, AssembleError};

    const WIDTH: u32 = 64;
    const HEIGHT: u32 = 8;
    const ROW: u64 = WIDTH as u64 * 4;
    const LEN: u64 = 1 << 16;
    /// The read layout of a texture of `WIDTH` by `HEIGHT` texels.
    const LAYOUT: BufferLayout = BufferLayout {
        offset: 0,
        bytes_per_row: ROW,
        bytes_per_image: 0,
    };

    /// The opaque handle of this host.
    const OPAQUE: ExportHandle = match cfg!(windows) {
        true => ExportHandle::OpaqueWin32,
        false => ExportHandle::OpaqueFd,
    };

    /// A kernel that loads a pointer from buffer 0, an argument buffer of one pointer member, and
    /// stores each thread's position at word `gid` past it: a write through an address outside
    /// its bound slots.
    const STORE_THROUGH: &str = r#"; Owned synthetic kernel.
source_filename = "store_through.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

%struct.Held = type { ptr addrspace(1) }

define void @store_through(ptr addrspace(2) noundef readonly align 8 dereferenceable(8) %held, i32 noundef %gid) {
entry:
  %at = getelementptr inbounds %struct.Held, ptr addrspace(2) %held, i64 0, i32 0
  %p = load ptr addrspace(1), ptr addrspace(2) %at, align 8
  %w = zext i32 %gid to i64
  %q = getelementptr inbounds i32, ptr addrspace(1) %p, i64 %w
  store i32 %gid, ptr addrspace(1) %q, align 4
  ret void
}

!air.kernel = !{!0}
!air.version = !{!5}
!air.language_version = !{!6}
!0 = !{ptr @store_through, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.indirect_buffer", !"air.buffer_size", i32 8, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !7, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"Held", !"air.arg_name", !"held"}
!4 = !{i32 1, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}
!5 = !{i32 2, i32 8, i32 0}
!6 = !{!"Metal", i32 4, i32 0, i32 0}
!7 = !{i32 0, i32 8, i32 0, !"uint", !"p", !"air.indirect_argument", !8}
!8 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"p"}
"#;

    /// Runs `encode` in a command buffer of its own on `queue`, and waits for it.
    fn run(name: &str, queue: &CommandQueue, encode: impl FnOnce(&mut crate::BlitCommandEncoder)) {
        let command_buffer = queue.command_buffer().expect("a command buffer");
        let mut encoder = command_buffer.blit_command_encoder().expect("an encoder");
        encode(&mut encoder);
        encoder.end_encoding();
        command_buffer.commit().expect("a commit");
        finished(name, &command_buffer);
        assert_eq!(command_buffer.error(), None, "{name}");
    }

    /// `count` submissions that each copy the texture into `out`, every byte of which then is
    /// `byte`: the copies in they made, and their bytes.
    fn reads(
        name: &str,
        (device, queue): (&Device, &CommandQueue),
        (texture, out): (&Texture, &Buffer),
        count: usize,
        byte: u8,
    ) -> (u64, u64) {
        let before = device.counters().metrics();
        let contents = out.contents().expect("Shared");
        for read in 0..count {
            // SAFETY: no pending command reaches the buffer's bytes.
            unsafe { contents.as_ptr().write_bytes(0xee, out.length() as usize) };
            let layout = BufferLayout {
                offset: 0,
                bytes_per_row: ROW,
                bytes_per_image: 0,
            };
            run(name, queue, |encoder| {
                (encoder.copy_from_texture_to_buffer(
                    texture,
                    TexturePlace::default(),
                    [WIDTH, HEIGHT, 1],
                    out,
                    layout,
                ))
                .unwrap_or_else(|error| panic!("{name}: the read: {error}"));
            });
            // SAFETY: the read has completed; nothing else reaches the bytes.
            let bytes =
                unsafe { std::slice::from_raw_parts(contents.as_ptr(), out.length() as usize) };
            let wrong = bytes.iter().position(|&b| b != byte);
            assert_eq!(wrong, None, "{name}: read {read}: byte {byte:#x}");
        }
        let after = device.counters().metrics();
        (
            after.over_copies - before.over_copies,
            after.over_copy_bytes - before.over_copy_bytes,
        )
    }

    /// A dispatch of `store_through` whose argument buffer `held` holds `target`'s address.
    struct Writer<'a> {
        state: &'a ComputePipelineState,
        held: &'a Buffer,
        target: &'a Buffer,
    }

    /// `count` submissions that each copy the texture into `out`, every byte of which then is
    /// `byte`, and then write through an address into the writer's target, every word of which
    /// then is its thread's position: the copies in each made.
    fn written_reads(
        name: &str,
        (device, queue): (&Device, &CommandQueue),
        (texture, out): (&Texture, &Buffer),
        writer: &Writer<'_>,
        count: usize,
        byte: u8,
    ) -> Vec<u64> {
        let contents = out.contents().expect("Shared");
        let words = writer.target.contents().expect("Shared");
        let threads = (writer.target.length() / 4) as u32;
        (0..count)
            .map(|read| {
                let before = device.counters().metrics().over_copies;
                // SAFETY: no pending command reaches either buffer's bytes.
                unsafe {
                    contents.as_ptr().write_bytes(0xee, out.length() as usize);
                    words
                        .as_ptr()
                        .write_bytes(0xff, writer.target.length() as usize);
                }
                let command_buffer = queue.command_buffer().expect("a command buffer");
                let mut blit = command_buffer.blit_command_encoder().expect("an encoder");
                (blit.copy_from_texture_to_buffer(
                    texture,
                    TexturePlace::default(),
                    [WIDTH, HEIGHT, 1],
                    out,
                    LAYOUT,
                ))
                .unwrap_or_else(|error| panic!("{name}: the read: {error}"));
                blit.end_encoding();
                let mut compute = (command_buffer.compute_command_encoder(DispatchType::Serial))
                    .expect("an encoder");
                (compute.set_compute_pipeline_state(writer.state)).expect("a state");
                (compute.set_buffer(Some(writer.held), 0, 0)).expect("the argument buffer");
                (compute.use_resource(Resource::Buffer(writer.target), ResourceUsage::WRITE))
                    .expect("the target");
                (compute.dispatch_threadgroups([1, 1, 1], [threads, 1, 1]))
                    .unwrap_or_else(|error| panic!("{name}: the write: {error}"));
                compute.end_encoding();
                command_buffer.commit().expect("a commit");
                finished(name, &command_buffer);
                assert_eq!(command_buffer.error(), None, "{name}");
                // SAFETY: the submission has completed; nothing else reaches the bytes.
                let bytes =
                    unsafe { std::slice::from_raw_parts(contents.as_ptr(), out.length() as usize) };
                let wrong = bytes.iter().position(|&b| b != byte);
                assert_eq!(wrong, None, "{name}: read {read}: byte {byte:#x}");
                // SAFETY: as above; the target holds `threads` words, 4-byte aligned.
                let written = unsafe {
                    std::slice::from_raw_parts(words.as_ptr().cast::<u32>(), threads as usize)
                };
                let wrong = (0..threads).position(|at| written[at as usize] != at);
                assert_eq!(wrong, None, "{name}: write {read} through the address");
                device.counters().metrics().over_copies - before
            })
            .collect()
    }

    /// `store_through`'s state, and the block of the two Shared buffers it reaches: the argument
    /// buffer, which holds the target's address, and the target.
    fn store_through(
        name: &str,
        device: &Device,
        library: &Library,
    ) -> (ComputePipelineState, Block, Buffer, Buffer) {
        let function = library.function("store_through").expect("the kernel");
        let kernel = ComputePipelineDescriptor::new(StageSource::Function(
            FunctionDescriptor::new(function),
        ));
        let state = (device
            .new_compute_pipeline_state(&kernel, crate::tests::no_constexpr_samplers))
        .unwrap_or_else(|error| panic!("{name}: store_through: {error}"));
        let interface = &state.inner.compiled.whole.interface;
        assert!(
            interface.writes_unbound,
            "{name}: store_through writes through an address"
        );
        let block = (device.take_block(MemoryClass::Shared, 2 * ROW))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        let target = (device.new_buffer(block.id, 0, ROW, Some(0))).expect("a buffer");
        let held = (device.new_buffer(block.id, ROW, 8, Some(0))).expect("a buffer");
        let address = target.gpu_address().to_le_bytes();
        let contents = held.contents().expect("Shared");
        // SAFETY: the buffer holds 8 bytes, and no command reaches them.
        unsafe { std::ptr::copy_nonoverlapping(address.as_ptr(), contents.as_ptr(), 8) };
        (state, block, held, target)
    }

    /// A plane of an exported surface in the copy form, over a block whose address no call has
    /// handed out, read by submissions that each then write through an address into another
    /// buffer: the copies in each made before and after its buffer's `gpu_address` is read, and
    /// the surface's form; `None` where the device makes no such surface.
    fn through_addresses(
        name: &str,
        at: (&Device, &CommandQueue),
        out: &Buffer,
        writer: &Writer<'_>,
    ) -> Option<(Vec<u64>, Vec<u64>, SurfaceForm)> {
        let device = at.0;
        let mut d =
            TextureDescriptor::new(TextureType::Type2D, PixelFormat::RGBA8Unorm, WIDTH, HEIGHT);
        d.usage = TextureUsage::SHADER_READ;
        d.storage_mode = StorageMode::Private;
        let planes = [PlaneSpec {
            descriptor: d,
            index: None,
            rows_below: 0,
        }];
        // A pitch multiple no driver's pitch of a 256-byte row meets: the copy form.
        let spec = SurfaceSpec {
            handle: OPAQUE,
            pitch_multiple: 4096,
            start_multiple: 1,
            planes: &planes,
        };
        let addressed = || device.counters().metrics().exported_blocks_addressed;
        let made = addressed();
        let (surface, export) = match device.new_exported_surface(&spec) {
            Ok(made) => made,
            Err(error) => {
                println!("exported_overs: {name}: no {OPAQUE:?} surface: {error}");
                return None;
            }
        };
        if export.form == SurfaceForm::InPlace {
            println!("exported_overs: {name}: the {OPAQUE:?} surface is in place");
            return None;
        }
        let buffer = (surface.buffer()).unwrap_or_else(|error| panic!("{name}: {error}"));
        run(name, at.1, |encoder| {
            (encoder.fill_buffer(&buffer, 0..buffer.length(), 0x3c))
                .unwrap_or_else(|error| panic!("{name}: the fill: {error}"));
        });
        let over = (&surface.planes()[0], out);
        let before = written_reads(name, at, over, writer, 4, 0x3c);
        assert_eq!(before, [1, 0, 0, 0], "{name}: no address of the block out");
        assert_eq!(addressed() - made, 0, "{name}: no address of the block out");
        assert_ne!(
            buffer.gpu_address(),
            0,
            "{name}: the surface buffer's address"
        );
        assert_eq!(
            addressed() - made,
            1,
            "{name}: the surface buffer's address read"
        );
        // The submission before the read owes nothing: no address of the block was out then.
        let after = written_reads(name, at, over, writer, 3, 0x3c);
        assert_eq!(after, [0, 1, 1], "{name}: the block's address out");
        Some((before, after, export.form))
    }

    #[test]
    fn an_exported_blocks_texture_is_copied_in_once_a_change() {
        let library = match llvm_assemble_text(STORE_THROUGH) {
            Ok(bitcode) => Some(Library::from_air(&bitcode).expect("the library")),
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: the writes through an address: {why}");
                None
            }
            Err(error) => panic!("store_through: {error:?}"),
        };
        each_device("exported_overs", |name, _, device| {
            let Some(handle) = (ExportHandle::ALL.into_iter())
                .find(|&handle| device.new_exported_block(handle, LEN).is_ok())
            else {
                println!("exported_overs: {name}: exports no handle");
                return;
            };
            let addressed = || device.counters().metrics().exported_blocks_addressed;
            let made = addressed();
            let (block, _export) = (device.new_exported_block(handle, LEN))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            assert_eq!(
                block.gpu_address, None,
                "{name}: the block's handle states no address"
            );
            assert_eq!(
                addressed() - made,
                0,
                "{name}: the make hands out no address"
            );
            let buffer = (device.new_buffer(block.id, 0, LEN, None)).expect("a buffer");
            let pitch = (device.minimum_linear_texture_alignment(PixelFormat::RGBA8Unorm))
                .unwrap_or_else(|error| panic!("{name}: {error}"))
                .max(ROW);
            let mut d =
                TextureDescriptor::new(TextureType::Type2D, PixelFormat::RGBA8Unorm, WIDTH, HEIGHT);
            d.usage = TextureUsage::SHADER_READ;
            d.storage_mode = StorageMode::Private;
            let first = IndexClass::Texture.range().start;
            let _grant = (device.grant_indices(IndexClass::Texture, first, 1)).expect("a grant");
            let texture = (buffer.new_texture(&d, Some(first), 0, pitch))
                .unwrap_or_else(|error| panic!("{name}: a texture over the export: {error}"));
            let out_block = (device.take_block(MemoryClass::Shared, ROW * u64::from(HEIGHT)))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let out = (device.new_buffer(out_block.id, 0, ROW * u64::from(HEIGHT), Some(0)))
                .expect("a buffer");
            let queue = device.new_command_queue(4).expect("a queue");
            let plane = ROW * u64::from(HEIGHT);
            let fill = |value: u8| {
                run(name, &queue, |encoder| {
                    (encoder.fill_buffer(&buffer, 0..LEN, value))
                        .unwrap_or_else(|error| panic!("{name}: the fill: {error}"));
                });
            };
            let at = (&device, &queue);
            let over = (&texture, &out);
            fill(0x5a);
            (block.external_writes_done()).unwrap_or_else(|error| panic!("{name}: {error}"));
            let acquired = reads(name, at, over, 4, 0x5a);
            assert_eq!(acquired, (1, plane), "{name}: an acquire, four reads");
            fill(0xb7);
            let filled = reads(name, at, over, 3, 0xb7);
            assert_eq!(filled, (1, plane), "{name}: a fill, three reads");
            (device.zero_block_range(block.id, 0, LEN))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let zeroed = reads(name, at, over, 2, 0);
            assert_eq!(zeroed, (1, plane), "{name}: a zero fill, two reads");
            let unchanged = reads(name, at, over, 2, 0);
            assert_eq!(unchanged, (0, 0), "{name}: two reads, no event");
            println!(
                "exported_overs: {name}: {handle:?}: copies in after an acquire, a fill, a zero \
                 fill and no event: {} {} {} {}",
                acquired.0, filled.0, zeroed.0, unchanged.0
            );
            let Some(library) = &library else {
                return;
            };
            let (state, _block, held, target) = store_through(name, &device, library);
            let writer = Writer {
                state: &state,
                held: &held,
                target: &target,
            };
            let before = written_reads(name, at, over, &writer, 3, 0);
            assert_eq!(before, [0, 0, 0], "{name}: no address of the block out");
            assert_eq!(addressed() - made, 0, "{name}: no address of the block out");
            assert_ne!(buffer.gpu_address(), 0, "{name}: the buffer's address");
            assert_eq!(addressed() - made, 1, "{name}: the buffer's address read");
            // The submission before the read owes nothing: no address of the block was out then.
            let after = written_reads(name, at, over, &writer, 3, 0);
            assert_eq!(after, [0, 1, 1], "{name}: the block's address out");
            println!(
                "exported_overs: {name}: {handle:?} block: copies in a submission that writes \
                 through an address, the block's address not out: {before:?}, then out: \
                 {after:?}; blocks addressed 0, then 1"
            );
            if let Some((before, after, form)) = through_addresses(name, at, &out, &writer) {
                println!(
                    "exported_overs: {name}: {OPAQUE:?} surface, {form:?}: copies in a \
                     submission that writes through an address, the block's address not out: \
                     {before:?}, then out: {after:?}; blocks addressed 0, then 1"
                );
            }
        });
    }
}
