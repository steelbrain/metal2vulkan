//! Texture buffers (`MTLTextureTypeTextureBuffer`), under a device cfg on every device of the cfg's
//! kind, each a buffer view of a Shared buffer's bytes ([`crate::Buffer::new_texture`]):
//!
//! - the owned fixture `ob1_texel_buffer` reads four `R32Uint` texels of one texture buffer, at a
//!   nonzero offset in its buffer, through the uniform texel-buffer array, and writes four of
//!   another through the storage array: in bounds it reads the buffer's bytes and writes them,
//!   and past the width it reads zero and writes nothing, as the Metal Shading Language defines
//!   (a read out of bounds returns zero, a write out of bounds is ignored); the bytes around both
//!   keep theirs, and the host reads the written texels through the texture and the buffer alike;
//! - an owned kernel adds to an `R32Uint` texture buffer's texels through an atomic and loads them
//!   again (`access::read_write`), answering the texels the buffer held and leaving the sums there;
//! - what Metal refuses (an offset off the device's alignment, a `bytesPerRow` below the row,
//!   texels past the buffer, a descriptor of more than one row, level, slice or sample, a swizzle
//!   with writes) and what the runtime does not build (a texture buffer made other than over a
//!   buffer, a swizzled one, a view of one, a blit copy of one, a format held through a component
//!   mapping) are each refused typed.
//!
//! The device's gauges read zero once each test's device is gone (`each_device`), every texture
//! buffer's view freed with it. `llvm-as` assembles the functions; without it the tests skip
//! visibly.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::command::on_device::{each_device, finished};
    use crate::{
        BlitOption, Buffer, BufferLayout, ComputePipelineDescriptor, ComputePipelineState, Device,
        DispatchType, Error, FunctionDescriptor, Gap, IndexClass, Library, MemoryClass,
        PixelFormat, Refusal, Region, StageSource, StorageMode, Texture, TextureDescriptor,
        TexturePlace, TextureSwizzle, TextureSwizzleChannels, TextureType, TextureUsage,
        TextureViewKind,
    };
    use metal2vulkan_devtools::{engine_dir, llvm_assemble, llvm_assemble_text, AssembleError};

    /// The bytes of the Shared block every buffer of a test lies in.
    const BLOCK: u64 = 16 << 10;
    /// The buffer the texture buffers lie over: its place in the block and its length.
    const DATA: (u64, u64) = (0, 4096);
    /// What the bytes around a texture buffer hold, before and after.
    const SENTINEL: u32 = 0x5a5a_5a5a;

    /// `bitcode`'s library, or None where `llvm-as` is missing.
    fn library(name: &str, bitcode: Result<Vec<u8>, AssembleError>) -> Option<Library> {
        match bitcode {
            Ok(bitcode) => Some(Library::from_air(&bitcode).expect("the library")),
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                None
            }
            Err(error) => panic!("{name}: {error:?}"),
        }
    }

    /// The kernel `name`'s state.
    fn state(device: &Device, library: &Library, name: &str) -> ComputePipelineState {
        let function = library.function(name).expect("the kernel");
        let descriptor = ComputePipelineDescriptor::new(StageSource::Function(
            FunctionDescriptor::new(function),
        ));
        (device.new_compute_pipeline_state(&descriptor, |samplers| Ok(vec![0; samplers.len()])))
            .unwrap_or_else(|error| panic!("{name}: {error}"))
    }

    /// A texture buffer's descriptor: `width` texels of `pixel_format`, of `usage`.
    fn descriptor(pixel_format: PixelFormat, width: u32, usage: TextureUsage) -> TextureDescriptor {
        let mut d = TextureDescriptor::new(TextureType::TextureBuffer, pixel_format, width, 1);
        d.usage = usage;
        d
    }

    /// The words of a Shared buffer, as the host reads them.
    fn words(buffer: &Buffer) -> Vec<u32> {
        let contents = buffer.contents().expect("a Shared buffer");
        let len = usize::try_from(buffer.length() / 4).expect("a length");
        // SAFETY: the buffer is alive, Shared, of `len` words, and no work in flight writes it.
        unsafe { std::slice::from_raw_parts(contents.cast::<u32>().as_ptr(), len) }.to_vec()
    }

    /// Writes `values` to a Shared buffer's words from word `first`.
    fn set_words(buffer: &Buffer, first: usize, values: &[u32]) {
        let contents = buffer.contents().expect("a Shared buffer");
        let len = usize::try_from(buffer.length() / 4).expect("a length");
        assert!(first + values.len() <= len, "words past the buffer");
        // SAFETY: the buffer is alive, Shared, and holds the words; no work in flight reaches it.
        let words = unsafe { std::slice::from_raw_parts_mut(contents.cast::<u32>().as_ptr(), len) };
        words[first..first + values.len()].copy_from_slice(values);
    }

    /// One dispatch of `state` over `threads` threads in one threadgroup, with `textures` and
    /// `buffers` at their slots, committed and finished.
    fn dispatch(
        device: &Device,
        state: &ComputePipelineState,
        textures: &[&Texture],
        buffers: &[&Buffer],
        threads: u32,
    ) {
        let queue = device.new_command_queue(1).expect("a queue");
        let command_buffer = queue.command_buffer().expect("a command buffer");
        let mut encoder =
            (command_buffer.compute_command_encoder(DispatchType::Serial)).expect("an encoder");
        encoder
            .set_compute_pipeline_state(state)
            .expect("the state");
        for (slot, texture) in (0..).zip(textures) {
            encoder.set_texture(Some(texture), slot).expect("a texture");
        }
        for (slot, buffer) in (0..).zip(buffers) {
            encoder.set_buffer(Some(buffer), 0, slot).expect("a buffer");
        }
        encoder
            .dispatch_threadgroups([1, 1, 1], [threads, 1, 1])
            .expect("the dispatch");
        encoder.end_encoding();
        command_buffer.commit().expect("a commit");
        finished("texture buffers", &command_buffer);
        assert_eq!(command_buffer.error(), None);
    }

    #[test]
    fn every_device_reads_and_writes_texture_buffers_over_a_buffer_s_bytes() {
        let name = "ob1_texel_buffer.ll";
        let Some(library) = library(name, llvm_assemble(&engine_dir().join(name))) else {
            return;
        };
        each_device("texture buffers", |name, _, device| {
            let _grant = (device.grant_indices(IndexClass::Texture, 1, 2))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let block = (device.take_block(MemoryClass::Shared, BLOCK)).expect("a block");
            let place =
                |(at, len)| (device.new_buffer(block.id, at, len, Some(0))).expect("a buffer");
            let data = place(DATA);
            let idx = place((4096, 64));
            let out = place((8192, 32));
            let alignment = (device.minimum_texture_buffer_alignment(PixelFormat::R32Uint))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            assert!((1..=256).contains(&alignment), "{name}: {alignment}");
            // The read one at a nonzero multiple of the alignment, the written one at 2048.
            let (src_at, dst_at) = (4 * alignment, 2048);
            let words_of = |at: u64| usize::try_from(at / 4).expect("an offset");
            let (src, dst) = (words_of(src_at), words_of(dst_at));
            set_words(&data, 0, &[SENTINEL; 1024]);
            set_words(&data, src, &[11, 22, 33, 44]);
            set_words(&data, dst, &[0; 4]);
            let threads: Vec<u32> = (0..8).collect();
            set_words(&idx, 0, &threads);
            set_words(&idx, 8, &threads);
            set_words(&out, 0, &[SENTINEL; 8]);
            let read = (data.new_texture(
                &descriptor(PixelFormat::R32Uint, 4, TextureUsage::SHADER_READ),
                Some(1),
                src_at,
                16,
            ))
            .unwrap_or_else(|error| panic!("{name}: the read texture buffer: {error}"));
            let written = (data.new_texture(
                &descriptor(PixelFormat::R32Uint, 4, TextureUsage::SHADER_WRITE),
                Some(2),
                dst_at,
                16,
            ))
            .unwrap_or_else(|error| panic!("{name}: the written texture buffer: {error}"));
            assert_eq!(read.descriptor().storage_mode, StorageMode::Shared);
            assert_eq!(read.gpu_resource_id().index, 1);
            let kernel = state(&device, &library, "ob1_texel_buffer");
            dispatch(&device, &kernel, &[&read, &written], &[&idx, &out], 8);
            // In bounds, the buffer's texels; past the width, zero.
            assert_eq!(words(&out)[..8], [11, 22, 33, 44, 0, 0, 0, 0], "{name}");
            let bytes = words(&data);
            let marked: Vec<u32> = (0..4).map(|t| 0xa000_0000 | t).collect();
            assert_eq!(bytes[dst..dst + 4], marked, "{name}");
            // Nothing past either texture buffer's texels was read into them or written.
            assert_eq!(bytes[dst + 4..dst + 8], [SENTINEL; 4], "{name}");
            assert_eq!(bytes[src + 4..src + 8], [SENTINEL; 4], "{name}");
            assert_eq!(bytes[src..src + 4], [11, 22, 33, 44], "{name}");
            // The host reads the written texels through the texture as in the buffer.
            let mut texels = [0u8; 16];
            let region = Region {
                origin: [0; 3],
                size: [4, 1, 1],
            };
            (written.get_bytes(&mut texels, 16, 0, region, 0, 0))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let first = u32::from_ne_bytes(texels[..4].try_into().expect("a word"));
            assert_eq!(first, 0xa000_0000, "{name}");
            println!("{name}: texture buffers read and written at offsets {src_at} and {dst_at}");
        });
    }

    /// `tb_atomic(ab, out, t)`: thread `t` adds 3 to texel `t` of `ab`, a `texture_buffer<uint,
    /// access::read_write>`, writing the texel it held to `out[t]`, then loads the texel to
    /// `out[8 + t]`.
    const ATOMIC_KERNEL: &str = r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

define void @tb_atomic(ptr addrspace(1) %ab, ptr addrspace(1) noundef "air-buffer-no-alias" %out, i32 noundef %t) {
entry:
  %old = tail call <4 x i32> @air.atomic_fetch_add_explicit_texture_buffer_1d.u.v4i32(ptr addrspace(1) %ab, i32 %t, <4 x i32> <i32 3, i32 3, i32 3, i32 3>, i32 0, i32 3)
  %o = extractelement <4 x i32> %old, i64 0
  %p = getelementptr inbounds i32, ptr addrspace(1) %out, i32 %t
  store i32 %o, ptr addrspace(1) %p, align 4
  %now = tail call <4 x i32> @air.atomic_load_explicit_texture_buffer_1d.u.v4i32(ptr addrspace(1) %ab, i32 %t, i32 0, i32 3)
  %n = extractelement <4 x i32> %now, i64 0
  %t8 = add i32 %t, 8
  %q = getelementptr inbounds i32, ptr addrspace(1) %out, i32 %t8
  store i32 %n, ptr addrspace(1) %q, align 4
  ret void
}

declare <4 x i32> @air.atomic_fetch_add_explicit_texture_buffer_1d.u.v4i32(ptr addrspace(1), i32, <4 x i32>, i32, i32)
declare <4 x i32> @air.atomic_load_explicit_texture_buffer_1d.u.v4i32(ptr addrspace(1), i32, i32, i32)

!air.kernel = !{!0}
!air.version = !{!5}
!air.language_version = !{!6}
!0 = !{ptr @tb_atomic, !1, !2}
!1 = !{}
!2 = !{!3, !4, !7}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.arg_type_name", !"texture_buffer<uint, read_write>", !"air.arg_name", !"ab"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!5 = !{i32 2, i32 8, i32 0}
!6 = !{!"Metal", i32 4, i32 0, i32 0}
!7 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"t"}
"#;

    #[test]
    fn every_device_updates_a_texture_buffer_s_texels_by_an_atomic() {
        let name = "tb_atomic";
        let Some(library) = library(name, llvm_assemble_text(ATOMIC_KERNEL)) else {
            return;
        };
        each_device("texture buffer atomics", |name, _, device| {
            let _grant = (device.grant_indices(IndexClass::Texture, 1, 1))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let block = (device.take_block(MemoryClass::Shared, BLOCK)).expect("a block");
            let place =
                |(at, len)| (device.new_buffer(block.id, at, len, Some(0))).expect("a buffer");
            let data = place(DATA);
            let out = place((8192, 64));
            set_words(&data, 0, &[10, 20, 30, 40]);
            let usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
            let texels =
                (data.new_texture(&descriptor(PixelFormat::R32Uint, 4, usage), Some(1), 0, 16))
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
            let kernel = state(&device, &library, "tb_atomic");
            dispatch(&device, &kernel, &[&texels], &[&out], 4);
            let answers = words(&out);
            assert_eq!(answers[..4], [10, 20, 30, 40], "{name}: the texels held");
            assert_eq!(
                answers[8..12],
                [13, 23, 33, 43],
                "{name}: the texels loaded"
            );
            assert_eq!(
                words(&data)[..4],
                [13, 23, 33, 43],
                "{name}: the buffer's bytes"
            );
            println!("{name}: a texture buffer's atomics update its buffer's bytes");
        });
    }

    #[test]
    fn every_device_refuses_what_metal_refuses_of_a_texture_buffer_and_what_is_not_built() {
        each_device("texture buffer refusals", |name, _, device| {
            let _grant = (device.grant_indices(IndexClass::Texture, 1, 2))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let block = (device.take_block(MemoryClass::Shared, BLOCK)).expect("a block");
            let data = (device.new_buffer(block.id, DATA.0, DATA.1, Some(0))).expect("a buffer");
            let read = TextureUsage::SHADER_READ;
            let d = descriptor(PixelFormat::R32Uint, 4, read);
            let alignment = (device.minimum_texture_buffer_alignment(PixelFormat::R32Uint))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let refused = |result: Result<Texture, Error>| match result {
                Ok(_) => panic!("{name}: made"),
                Err(error) => error,
            };
            // Made other than over a buffer.
            assert!(matches!(
                refused(device.new_texture(&d, Some(1))),
                Error::Refused(Refusal::TextureBufferNotOverBuffer)
            ));
            // An offset off the alignment, where the device has one above a byte.
            if alignment > 1 {
                assert!(matches!(
                    refused(data.new_texture(&d, Some(1), alignment + 1, 16)),
                    Error::Refused(Refusal::TextureBufferAlignment {
                        pixel_format: PixelFormat::R32Uint,
                        alignment: a,
                        ..
                    }) if a == alignment
                ));
            }
            // A row's bytes below its texels', and texels past the buffer.
            for (width, bytes_per_row) in [(4, 15), (2048, 8192)] {
                let d = descriptor(PixelFormat::R32Uint, width, read);
                assert!(matches!(
                    refused(data.new_texture(&d, Some(1), 0, bytes_per_row)),
                    Error::Refused(Refusal::BufferTextureLayout { .. })
                ));
            }
            // More than one row, level, slice or sample.
            let mut tall = d;
            tall.height = 2;
            assert!(matches!(
                refused(data.new_texture(&tall, Some(1), 0, 16)),
                Error::Refused(Refusal::TextureExtentForType)
            ));
            let mut levels = d;
            levels.mipmap_level_count = 2;
            assert!(matches!(
                refused(data.new_texture(&levels, Some(1), 0, 16)),
                Error::Refused(Refusal::TextureLevelCount { levels: 2, max: 1 })
            ));
            // A swizzle: refused with writes as Metal refuses it, not built with reads.
            let mut swizzled = d;
            swizzled.swizzle = TextureSwizzleChannels {
                red: TextureSwizzle::Green,
                ..TextureSwizzleChannels::DEFAULT
            };
            assert!(matches!(
                refused(data.new_texture(&swizzled, Some(1), 0, 16)),
                Error::Unsupported(Gap::TextureBufferSwizzle {
                    pixel_format: PixelFormat::R32Uint
                })
            ));
            swizzled.usage = TextureUsage::SHADER_WRITE;
            assert!(matches!(
                refused(data.new_texture(&swizzled, Some(1), 0, 16)),
                Error::Refused(Refusal::TextureSwizzleWithWrites)
            ));
            // A format held through a component mapping.
            assert!(matches!(
                device.minimum_texture_buffer_alignment(PixelFormat::A8Unorm),
                Err(Error::Unsupported(Gap::TextureFormat {
                    pixel_format: PixelFormat::A8Unorm
                }))
            ));
            // A view of one, and a blit copy of one.
            let texels = (data.new_texture(&d, Some(1), 0, 16))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            assert!(matches!(
                texels.new_texture_view(
                    PixelFormat::R32Uint,
                    TextureType::TextureBuffer,
                    0..1,
                    0..1,
                    Some(2)
                ),
                Err(Error::Unsupported(Gap::TextureView {
                    kind: TextureViewKind::TextureBuffer
                }))
            ));
            let queue = device.new_command_queue(1).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut blit = command_buffer
                .blit_command_encoder()
                .expect("a blit encoder");
            let place = TexturePlace {
                slice: 0,
                level: 0,
                origin: [0; 3],
            };
            assert!(matches!(
                blit.copy_from_texture(&texels, place, [4, 1, 1], &texels, place),
                Err(Error::Unsupported(Gap::TextureCopy { .. }))
            ));
            let layout = BufferLayout {
                offset: 2048,
                bytes_per_row: 16,
                bytes_per_image: 0,
            };
            assert!(matches!(
                blit.copy_from_buffer_with_options(
                    &data,
                    layout,
                    [4, 1, 1],
                    &texels,
                    place,
                    BlitOption::None
                ),
                Err(Error::Unsupported(Gap::TextureBytes {
                    pixel_format: PixelFormat::R32Uint
                }))
            ));
            blit.end_encoding();
            drop((command_buffer, queue, texels));
            println!(
                "{name}: texture buffers refused as Metal refuses them, alignment {alignment}"
            );
        });
    }
}
