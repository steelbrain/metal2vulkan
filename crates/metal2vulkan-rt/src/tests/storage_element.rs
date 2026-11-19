//! A texture with no storage element in kernels, under a device cfg on every device of the cfg's
//! kind, through the owned fixture `rrp_swizzled_storage` over a 2 by 2 `RGBA8Unorm_sRGB` texture
//! of shader-read and shader-write usage, a format Vulkan requires no device to store:
//!
//! - an `access::write` slot and an `access::read_write` slot bound to it are
//!   `Gap::StorageElement`, found at the dispatch, exactly where the device's optimal tiling holds
//!   neither the format's storage nor its storage writes without a format, and dispatch where it
//!   holds both;
//! - an `access::read` slot bound to it dispatches on every device.
//!
//! And a texture an argument buffer reaches by its resource ID, through an owned kernel that writes
//! an argument buffer's `texture2d<float, write>` and one that reads an argument buffer's
//! `texture2d<float, read>`: for 2 by 2 `RGBA8Unorm_sRGB`, `BGRA8Unorm_sRGB` and `A8Unorm`
//! textures and an `RGBA8Unorm_sRGB` view of an `RGBA8Unorm` one, each of shader-read and
//! shader-write usage, and on every device:
//!
//! - after `use_resource` with a write usage, the writing kernel's dispatch is
//!   `Gap::StorageElement` with no slot and the texture's format, exactly where the texture has
//!   no storage element, and dispatches where it has one; the reading kernel's dispatch in the
//!   same encoder runs either way;
//! - the same texture placed in a heap and reached after `use_heap` alone is not refused.
//!
//! `llvm-as` assembles the kernels; without it the tests skip visibly.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::command::on_device::{each_device, finished, Arena};
    use crate::{
        Buffer, ComputeCommandEncoder, ComputePipelineDescriptor, ComputePipelineState, Device,
        DispatchType, Error, FunctionDescriptor, Gap, HeapRange, IndexClass, Library, MemoryClass,
        PixelFormat, Resource, ResourceUsage, StageSource, StorageMode, Texture, TextureDescriptor,
        TextureType, TextureUsage,
    };
    use ash::vk;
    use metal2vulkan_devtools::{engine_dir, llvm_assemble, llvm_assemble_text, AssembleError};

    /// The fixture's library, or None where `llvm-as` is missing.
    fn library() -> Option<Library> {
        match llvm_assemble(&engine_dir().join("rrp_swizzled_storage.ll")) {
            Ok(bitcode) => Some(Library::from_air(&bitcode).expect("the library")),
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                None
            }
            Err(error) => panic!("rrp_swizzled_storage.ll: {error:?}"),
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

    /// One dispatch of `state` over a 2 by 2 grid with `texture` at slot 0 and `buffer` at slots 0
    /// and 1, committed and finished: the encoder's answer.
    fn dispatch(
        device: &Device,
        state: &ComputePipelineState,
        texture: &Texture,
        buffer: &crate::Buffer,
    ) -> Result<(), Error> {
        let queue = device.new_command_queue(1).expect("a queue");
        let command_buffer = queue.command_buffer().expect("a command buffer");
        let mut encoder =
            (command_buffer.compute_command_encoder(DispatchType::Serial)).expect("an encoder");
        encoder
            .set_compute_pipeline_state(state)
            .expect("the state");
        encoder.set_texture(Some(texture), 0).expect("a texture");
        for index in [0, 1] {
            encoder
                .set_buffer(Some(buffer), 0, index)
                .expect("a buffer");
        }
        let dispatched = encoder.dispatch_threadgroups([1, 1, 1], [2, 2, 1]);
        encoder.end_encoding();
        command_buffer.commit().expect("a commit");
        finished("storage elements", &command_buffer);
        assert_eq!(command_buffer.error(), None);
        dispatched
    }

    #[test]
    fn every_device_refuses_a_stored_slot_bound_to_a_texture_with_no_storage_element() {
        let Some(library) = library() else {
            return;
        };
        each_device("storage elements", |name, _, device| {
            let (_, optimal) = device.core().format_features(vk::Format::R8G8B8A8_SRGB);
            let stores = optimal.contains(
                vk::FormatFeatureFlags2::STORAGE_IMAGE
                    | vk::FormatFeatureFlags2::STORAGE_WRITE_WITHOUT_FORMAT,
            );
            let grant = (device.grant_indices(IndexClass::Texture, 1, 1))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let mut d =
                TextureDescriptor::new(TextureType::Type2D, PixelFormat::RGBA8UnormSRGB, 2, 2);
            d.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
            d.storage_mode = StorageMode::Shared;
            let texture =
                (device.new_texture(&d, Some(1))).unwrap_or_else(|error| panic!("{name}: {error}"));
            let mut arena = Arena::new(&device);
            let out = arena.buffer();
            for kernel in ["rrp_sw_write_f", "rrp_sw_rwread_f"] {
                let stored = state(&device, &library, kernel);
                match (stores, dispatch(&device, &stored, &texture, &out)) {
                    (true, Ok(())) => {}
                    (
                        false,
                        Err(Error::Unsupported(Gap::StorageElement {
                            slot: Some(0),
                            pixel_format: PixelFormat::RGBA8UnormSRGB,
                        })),
                    ) => {}
                    (stores, other) => {
                        panic!(
                            "{name}: {kernel}, the device storing the format {stores}: {other:?}"
                        )
                    }
                }
            }
            let read = state(&device, &library, "rrp_sw_read_f");
            (dispatch(&device, &read, &texture, &out)).unwrap_or_else(|e| panic!("{name}: {e}"));
            drop(texture);
            (device.release_indices(grant)).unwrap_or_else(|error| panic!("{name}: {error}"));
            println!("{name}: the device stores 8-bit sRGB {stores}; stored slots answered so");
        });
    }

    /// Two kernels over the argument buffer at buffer 0, whose one member is a texture's 8-byte
    /// resource ID: `argument_write` writes `(1, 0.5, 0.25, 1)` to the member's
    /// `texture2d<float, write>` at each thread's position; `argument_read` reads texel `(0, 0)`
    /// of the member's `texture2d<float, read>` and stores its red bits in word 0 of `out`
    /// (buffer 1).
    const ARGUMENT_KERNELS: &str = r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

%struct.T = type { ptr addrspace(1) }
%struct.R = type { %struct.T }

define void @argument_write(ptr addrspace(2) %r, <2 x i32> %gid) {
entry:
  %pt = getelementptr inbounds %struct.R, ptr addrspace(2) %r, i64 0, i32 0, i32 0
  %t = load ptr addrspace(1), ptr addrspace(2) %pt, align 8
  tail call void @air.write_texture_2d.v4f32(ptr addrspace(1) %t, <2 x i32> %gid, <4 x float> <float 1.0, float 0.5, float 0.25, float 1.0>, i32 0, i32 2)
  ret void
}

define void @argument_read(ptr addrspace(2) %r, ptr addrspace(1) noundef "air-buffer-no-alias" %out) {
entry:
  %pt = getelementptr inbounds %struct.R, ptr addrspace(2) %r, i64 0, i32 0, i32 0
  %t = load ptr addrspace(1), ptr addrspace(2) %pt, align 8
  %s = tail call ptr addrspace(2) @air.get_read_sampler()
  %r0 = tail call { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %s, <2 x i32> zeroinitializer, <2 x i32> zeroinitializer, i32 0, i32 1)
  %v = extractvalue { <4 x float>, i8 } %r0, 0
  %red = extractelement <4 x float> %v, i64 0
  %rb = bitcast float %red to i32
  store i32 %rb, ptr addrspace(1) %out, align 4
  ret void
}

declare void @air.write_texture_2d.v4f32(ptr addrspace(1), <2 x i32>, <4 x float>, i32, i32)
declare ptr addrspace(2) @air.get_read_sampler()
declare { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32)

!air.kernel = !{!0, !9}
!air.version = !{!7}
!air.language_version = !{!8}
!0 = !{ptr @argument_write, !1, !2}
!1 = !{}
!2 = !{!3, !5}
!3 = !{i32 0, !"air.indirect_buffer", !"air.buffer_size", i32 8, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !4, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"W", !"air.arg_name", !"r"}
!4 = !{i32 0, i32 8, i32 0, !"texture2d<float, write>", !"t", !"air.indirect_argument", !6}
!5 = !{i32 1, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"gid"}
!6 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<float, write>", !"air.arg_name", !"t"}
!7 = !{i32 2, i32 8, i32 0}
!8 = !{!"Metal", i32 4, i32 0, i32 0}
!9 = !{ptr @argument_read, !1, !10}
!10 = !{!11, !14}
!11 = !{i32 0, !"air.indirect_buffer", !"air.buffer_size", i32 8, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !12, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"R", !"air.arg_name", !"r"}
!12 = !{i32 0, i32 8, i32 0, !"texture2d<float, read>", !"t", !"air.indirect_argument", !13}
!13 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"texture2d<float, read>", !"air.arg_name", !"t"}
!14 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
"#;

    /// One compute encoder `encode` fills, committed and finished: what `encode` answered.
    fn encoded(
        device: &Device,
        encode: impl FnOnce(&mut ComputeCommandEncoder) -> Result<(), Error>,
    ) -> Result<(), Error> {
        let queue = device.new_command_queue(1).expect("a queue");
        let command_buffer = queue.command_buffer().expect("a command buffer");
        let mut encoder =
            (command_buffer.compute_command_encoder(DispatchType::Serial)).expect("an encoder");
        let answered = encode(&mut encoder);
        encoder.end_encoding();
        command_buffer.commit().expect("a commit");
        finished("storage elements", &command_buffer);
        assert_eq!(command_buffer.error(), None);
        answered
    }

    /// Writes `texture`'s resource ID over the start of the Shared buffer `argument`.
    fn place_id(argument: &Buffer, texture: &Texture) {
        let id = texture.gpu_resource_id();
        let bytes = (u64::from(id.index) | u64::from(id.word) << 32).to_le_bytes();
        let contents = argument.contents().expect("a Shared buffer");
        // SAFETY: the 8 bytes lie in the buffer, which is alive, and no GPU work uses it.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr(), bytes.len()) };
    }

    #[test]
    fn every_device_refuses_a_declared_write_of_a_texture_with_no_storage_element() {
        let library = match llvm_assemble_text(ARGUMENT_KERNELS) {
            Ok(bitcode) => Library::from_air(&bitcode).expect("the library"),
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                return;
            }
            Err(error) => panic!("the argument kernels: {error:?}"),
        };
        each_device("declared storage elements", |name, _, device| {
            let grant = (device.grant_indices(IndexClass::Texture, 1, 16))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let write = state(&device, &library, "argument_write");
            let read = state(&device, &library, "argument_read");
            let descriptor = |pixel_format| {
                let mut d = TextureDescriptor::new(TextureType::Type2D, pixel_format, 2, 2);
                d.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
                d.storage_mode = StorageMode::Private;
                d
            };
            let mut index = 1..17;
            let mut next = || index.next().expect("an index of the grant");
            let mut cases = Vec::new();
            for format in [
                PixelFormat::RGBA8UnormSRGB,
                PixelFormat::BGRA8UnormSRGB,
                PixelFormat::A8Unorm,
            ] {
                let texture = (device.new_texture(&descriptor(format), Some(next())))
                    .unwrap_or_else(|error| panic!("{name}: {format:?}: {error}"));
                cases.push((format!("{format:?}"), texture));
            }
            let parent = (device.new_texture(&descriptor(PixelFormat::RGBA8Unorm), None))
                .unwrap_or_else(|error| panic!("{name}: RGBA8Unorm: {error}"));
            let view = (parent.new_texture_view(
                PixelFormat::RGBA8UnormSRGB,
                TextureType::Type2D,
                0..1,
                0..1,
                Some(next()),
            ))
            .unwrap_or_else(|error| panic!("{name}: the sRGB view: {error}"));
            cases.push((
                "an RGBA8UnormSRGB view of an RGBA8Unorm texture".into(),
                view,
            ));
            let mut arena = Arena::new(&device);
            let (argument, out) = (arena.buffer(), arena.buffer());
            let mut refused = Vec::new();
            for (case, texture) in &cases {
                place_id(&argument, texture);
                let unstored = texture.unstored();
                let answered = encoded(&device, |encoder| {
                    encoder.use_resource(Resource::Texture(texture), ResourceUsage::WRITE)?;
                    encoder.set_buffer(Some(&argument), 0, 0)?;
                    encoder.set_buffer(Some(&out), 0, 1)?;
                    encoder.set_compute_pipeline_state(&read)?;
                    (encoder.dispatch_threadgroups([1, 1, 1], [1, 1, 1])).unwrap_or_else(|error| {
                        panic!("{name}: {case}: the reading kernel: {error}")
                    });
                    encoder.set_compute_pipeline_state(&write)?;
                    encoder.dispatch_threadgroups([1, 1, 1], [2, 2, 1])
                });
                match (unstored, answered) {
                    (None, Ok(())) => {}
                    (
                        Some(format),
                        Err(Error::Unsupported(Gap::StorageElement {
                            slot: None,
                            pixel_format,
                        })),
                    ) if pixel_format == format => refused.push(case.clone()),
                    (unstored, other) => {
                        panic!("{name}: {case}, with no storage element {unstored:?}: {other:?}")
                    }
                }
            }

            // A heap's textures, declared by `use_heap` alone.
            let block = (device.take_block(MemoryClass::Private, MemoryClass::Private.block_len()))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let heap = HeapRange {
                block: block.id,
                offset: 0,
                len: MemoryClass::Private.block_len(),
            };
            let mut offset = 0;
            for format in [
                PixelFormat::RGBA8UnormSRGB,
                PixelFormat::BGRA8UnormSRGB,
                PixelFormat::A8Unorm,
            ] {
                let d = descriptor(format);
                let fit = (device.heap_texture_size_and_align(&d))
                    .unwrap_or_else(|error| panic!("{name}: {format:?}: {error}"));
                offset = u64::next_multiple_of(offset, fit.align);
                let texture = (device.new_heap_texture(&d, Some(next()), heap, offset))
                    .unwrap_or_else(|error| panic!("{name}: {format:?} in a heap: {error}"));
                offset += fit.size;
                place_id(&argument, &texture);
                (encoded(&device, |encoder| {
                    encoder.use_heap(block.id)?;
                    encoder.set_buffer(Some(&argument), 0, 0)?;
                    encoder.set_compute_pipeline_state(&write)?;
                    encoder.dispatch_threadgroups([1, 1, 1], [2, 2, 1])
                }))
                .unwrap_or_else(|error| panic!("{name}: {format:?} in a used heap: {error}"));
            }
            drop((cases, parent));
            (device.release_block(block)).unwrap_or_else(|error| panic!("{name}: {error}"));
            (device.release_indices(grant)).unwrap_or_else(|error| panic!("{name}: {error}"));
            println!(
                "{name}: a declared write refused for {refused:?}; every other case and every \
                 used heap dispatched"
            );
        });
    }
}
