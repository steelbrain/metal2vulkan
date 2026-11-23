//! Storage textures written at a level a function reads at run time (MSL `write(color, coord,
//! lod)`), under a device cfg on every device of the cfg's kind, through an owned kernel that
//! writes `(x, y, lod, 1)` at texel `(x, y)` of level `lod` of a `texture2d<float, write>`, `lod`
//! read from its buffer:
//!
//! - over an 8 by 8 `RGBA32Float` texture of four levels, every level filled first, dispatches at
//!   levels 0, 1 and 2, each over its level's texels, write exactly the 64, 16 and 4 texels of
//!   those levels, and level 3, which no dispatch writes, keeps its fill: what Metal does on an
//!   Intel Mac (a Radeon Pro 5300M and a UHD 630 under macOS 26.7.1).
//!
//! `llvm-as` assembles the kernel; without it the test skips visibly.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::command::on_device::{each_device, finished};
    use crate::{
        ComputePipelineDescriptor, ComputePipelineState, Device, DispatchType, FunctionDescriptor,
        IndexClass, Library, PixelFormat, Region, StageSource, StorageMode, Texture,
        TextureDescriptor, TextureType, TextureUsage,
    };
    use metal2vulkan_devtools::{llvm_assemble_text, AssembleError};

    /// `level_write`: a kernel over `texture2d<float, write> t [[texture(0)]]` and `constant uint
    /// &lod [[buffer(0)]]` whose thread `(x, y)` writes `(x, y, lod, 1)` at texel `(x, y)` of
    /// level `lod`.
    const LEVEL_WRITE: &str = r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

define void @level_write(ptr addrspace(1) %t, ptr addrspace(2) noundef "air-buffer-no-alias" %lod, <2 x i32> noundef %gid) {
entry:
  %l = load i32, ptr addrspace(2) %lod, align 4
  %x = extractelement <2 x i32> %gid, i64 0
  %y = extractelement <2 x i32> %gid, i64 1
  %fx = uitofp i32 %x to float
  %fy = uitofp i32 %y to float
  %fl = uitofp i32 %l to float
  %v0 = insertelement <4 x float> <float poison, float poison, float poison, float 1.0>, float %fx, i64 0
  %v1 = insertelement <4 x float> %v0, float %fy, i64 1
  %v = insertelement <4 x float> %v1, float %fl, i64 2
  tail call void @air.write_texture_2d.v4f32(ptr addrspace(1) %t, <2 x i32> %gid, <4 x float> %v, i32 %l, i32 2)
  ret void
}

declare void @air.write_texture_2d.v4f32(ptr addrspace(1), <2 x i32>, <4 x float>, i32, i32)

!air.kernel = !{!0}
!air.version = !{!7}
!air.language_version = !{!8}
!0 = !{ptr @level_write, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<float, write>", !"air.arg_name", !"t"}
!4 = !{i32 1, !"air.buffer", !"air.buffer_size", i32 4, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"lod"}
!5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"gid"}
!7 = !{i32 2, i32 8, i32 0}
!8 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

    /// The texture's side, and its levels.
    const SIDE: u32 = 8;
    const LEVELS: u32 = 4;

    /// The library of [`LEVEL_WRITE`], or None where `llvm-as` is missing.
    fn library() -> Option<Library> {
        match llvm_assemble_text(LEVEL_WRITE) {
            Ok(bitcode) => Some(Library::from_air(&bitcode).expect("the library")),
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                None
            }
            Err(error) => panic!("level_write: {error:?}"),
        }
    }

    /// The kernel's state.
    fn state(name: &str, device: &Device, library: &Library) -> ComputePipelineState {
        let function = library.function("level_write").expect("the kernel");
        let descriptor = ComputePipelineDescriptor::new(StageSource::Function(
            FunctionDescriptor::new(function),
        ));
        (device.new_compute_pipeline_state(&descriptor, |samplers| Ok(vec![0; samplers.len()])))
            .unwrap_or_else(|error| panic!("{name}: {error}"))
    }

    /// The side of level `level`.
    fn side(level: u32) -> u32 {
        (SIDE >> level).max(1)
    }

    /// The fill of texel `(x, y)` of level `level`: no texel the kernel writes holds it.
    fn fill(level: u32, x: u32, y: u32) -> [f32; 4] {
        [-1.0 - level as f32, x as f32, y as f32, 0.5]
    }

    /// Level `level` of the `RGBA32Float` texture, texel by texel, row by row.
    fn level_texels(name: &str, texture: &Texture, level: u32) -> Vec<[f32; 4]> {
        let side = side(level);
        let mut bytes = vec![0u8; (side * side * 16) as usize];
        let region = Region {
            origin: [0; 3],
            size: [side, side, 1],
        };
        let row = u64::from(side) * 16;
        (texture.get_bytes(&mut bytes, row, row * u64::from(side), region, level, 0))
            .unwrap_or_else(|error| panic!("{name}: level {level}: {error}"));
        let words: Vec<f32> = (bytes.as_chunks::<4>().0.iter())
            .map(|w| f32::from_le_bytes(*w))
            .collect();
        words.as_chunks::<4>().0.to_vec()
    }

    /// An 8 by 8 `RGBA32Float` texture of four levels at `index`, of shader-read and shader-write
    /// usage, each level holding its [`fill`].
    fn filled(name: &str, device: &Device, index: u32) -> Texture {
        let mut d =
            TextureDescriptor::new(TextureType::Type2D, PixelFormat::RGBA32Float, SIDE, SIDE);
        d.mipmap_level_count = LEVELS;
        d.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
        d.storage_mode = StorageMode::Shared;
        let texture =
            (device.new_texture(&d, Some(index))).unwrap_or_else(|error| panic!("{name}: {error}"));
        for level in 0..LEVELS {
            let side = side(level);
            let bytes: Vec<u8> = (0..side * side)
                .flat_map(|i| fill(level, i % side, i / side))
                .flat_map(f32::to_le_bytes)
                .collect();
            let region = Region {
                origin: [0; 3],
                size: [side, side, 1],
            };
            let row = u64::from(side) * 16;
            (texture.replace_region(
                region,
                level,
                0,
                crate::HostBytes::from(&bytes[..]),
                row,
                row * u64::from(side),
            ))
            .unwrap_or_else(|error| panic!("{name}: level {level}: {error}"));
        }
        texture
    }

    #[test]
    fn every_device_writes_the_levels_a_kernel_reads_from_a_buffer() {
        let Some(library) = library() else {
            return;
        };
        each_device("storage_levels", |name, _, device| {
            let state = state(name, &device, &library);
            let _grant = (device.grant_indices(IndexClass::Texture, 1, 1))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let texture = filled(name, &device, 1);
            let queue = device.new_command_queue(1).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = (command_buffer.compute_command_encoder(DispatchType::Serial))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let call = |result: Result<(), crate::Error>| {
                result.unwrap_or_else(|error| panic!("{name}: {error}"))
            };
            call(encoder.set_compute_pipeline_state(&state));
            call(encoder.set_texture(Some(&texture), 0));
            for level in 0..3 {
                call(encoder.set_bytes(&level.to_le_bytes(), 0));
                call(encoder.dispatch_threads([side(level), side(level), 1], [1, 1, 1]));
            }
            encoder.end_encoding();
            command_buffer.commit().expect("a commit");
            finished(name, &command_buffer);
            assert_eq!(command_buffer.error(), None, "{name}");
            for level in 0..LEVELS {
                let side = side(level);
                let want: Vec<[f32; 4]> = (0..side * side)
                    .map(|i| {
                        let (x, y) = (i % side, i / side);
                        match level {
                            0..3 => [x as f32, y as f32, level as f32, 1.0],
                            _ => fill(level, x, y),
                        }
                    })
                    .collect();
                assert_eq!(
                    level_texels(name, &texture, level),
                    want,
                    "{name}: level {level}"
                );
            }
        });
    }
}
