//! Storage textures written at a level a function reads at run time (MSL `write(color, coord,
//! lod)`), under a device cfg on every device of the cfg's kind, through owned functions that
//! write at level `lod` of a `texture2d<float, write>`, `lod` read from their buffer: a kernel that
//! writes `(x, y, lod, 1)` at texel `(x, y)`, and a fragment function that writes `(1, 2, lod, 1)`
//! at texel `(0, 0)`.
//!
//! - Over an 8 by 8 `RGBA32Float` texture of four levels, every level filled first, dispatches at
//!   levels 0, 1 and 2, each over its level's texels, write exactly the 64, 16 and 4 texels of
//!   those levels, and level 3, which no dispatch writes, keeps its fill: what Metal does on an
//!   Intel Mac (a Radeon Pro 5300M and a UHD 630 under macOS 26.7.1).
//! - Dispatches at levels 4, 9 and `u32::MAX`, past the texture's count, and at levels 0 and 1 of
//!   nil write nothing: every level keeps its fill.
//! - A texture bound again makes no view (`Metrics::image_view_creates`) and takes no run
//!   (`Gauges::level_elements`): the first bind of each of two textures makes its four views and
//!   takes one run, and twenty binds after them make none.
//! - Draws at levels 1 and 2, then 5, through the fragment function write texel `(0, 0)` of levels
//!   1 and 2 and nothing else.
//! - 4,097 textures of two levels, each bound to the kernel in one command buffer: the 4,097th's
//!   dispatch is `Limit::LevelElements { max: 4096 }`, holding no run; once the first texture is
//!   released and its command buffer has retired, the 4,097th's dispatch runs. Every run is back
//!   once the device is gone ([`each_device`]'s gauges).
//! - `Metrics::level_field_commands` counts each dispatch and draw that pushes a texture's level
//!   fields: one dispatch of the writing kernel, then one of a kernel that stores
//!   `get_num_mip_levels` of a `texture2d<float, write>` (which answers 4), each one more; and the
//!   two draws at levels 1 and 2 above, and the one at 5, three more.
//!
//! `llvm-as` assembles the functions; without it each test skips visibly.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::error::{Error, Limit};
    use crate::tests::command::on_device::{each_device, finished, within, Arena};
    use crate::tests::compute::on_device::words;
    use crate::tests::draw::on_device::{examples, pass, run, state as render_state, target};
    use crate::tests::draw::on_device::{vertices, Buffers};
    use crate::{
        ColorAttachmentDescriptor, ComputeCommandEncoder, ComputePipelineDescriptor,
        ComputePipelineState, Device, DispatchType, FunctionDescriptor, IndexClass, Library,
        PixelFormat, PrimitiveType, Region, RenderPipelineDescriptor, StageSource, StorageMode,
        Texture, TextureDescriptor, TextureType, TextureUsage,
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

    /// `level_write_fragment`: a fragment function over `texture2d<float, write> t
    /// [[texture(0)]]` and `constant uint &lod [[buffer(0)]]` that writes `(1, 2, lod, 1)` at texel
    /// `(0, 0)` of level `lod` and returns opaque green.
    const LEVEL_WRITE_FRAGMENT: &str = r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

define <4 x float> @level_write_fragment(ptr addrspace(1) %t, ptr addrspace(2) noundef "air-buffer-no-alias" %lod) {
entry:
  %l = load i32, ptr addrspace(2) %lod, align 4
  %fl = uitofp i32 %l to float
  %v = insertelement <4 x float> <float 1.0, float 2.0, float poison, float 1.0>, float %fl, i64 2
  tail call void @air.write_texture_2d.v4f32(ptr addrspace(1) %t, <2 x i32> zeroinitializer, <4 x float> %v, i32 %l, i32 2)
  ret <4 x float> <float 0.0, float 1.0, float 0.0, float 1.0>
}

declare void @air.write_texture_2d.v4f32(ptr addrspace(1), <2 x i32>, <4 x float>, i32, i32)

!air.fragment = !{!0}
!air.version = !{!7}
!air.language_version = !{!8}
!0 = !{ptr @level_write_fragment, !1, !3}
!1 = !{!2}
!2 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4", !"air.arg_name", !"color"}
!3 = !{!4, !5}
!4 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<float, write>", !"air.arg_name", !"t"}
!5 = !{i32 1, !"air.buffer", !"air.buffer_size", i32 4, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"lod"}
!7 = !{i32 2, i32 8, i32 0}
!8 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

    /// `level_count`: a kernel over `texture2d<float, write> t [[texture(0)]]` and `device uint
    /// &out [[buffer(1)]]` that stores `t.get_num_mip_levels()` in `out`.
    const LEVEL_COUNT: &str = r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

define void @level_count(ptr addrspace(1) %t, ptr addrspace(1) noundef "air-buffer-no-alias" %out) {
entry:
  %n = tail call i32 @air.get_num_mip_levels_texture_2d(ptr addrspace(1) %t)
  store i32 %n, ptr addrspace(1) %out, align 4
  ret void
}

declare i32 @air.get_num_mip_levels_texture_2d(ptr addrspace(1))

!air.kernel = !{!0}
!air.version = !{!7}
!air.language_version = !{!8}
!0 = !{ptr @level_count, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<float, write>", !"air.arg_name", !"t"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!7 = !{i32 2, i32 8, i32 0}
!8 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

    /// The texture's side, and its levels.
    const SIDE: u32 = 8;
    const LEVELS: u32 = 4;

    /// The library of `text`, or None where `llvm-as` is missing.
    fn library(text: &str) -> Option<Library> {
        match llvm_assemble_text(text) {
            Ok(bitcode) => Some(Library::from_air(&bitcode).expect("the library")),
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                None
            }
            Err(error) => panic!("an owned function: {error:?}"),
        }
    }

    /// The state of `library`'s kernel `kernel`.
    fn kernel_state(
        name: &str,
        device: &Device,
        library: &Library,
        kernel: &str,
    ) -> ComputePipelineState {
        let function = library.function(kernel).expect("the kernel");
        let descriptor = ComputePipelineDescriptor::new(StageSource::Function(
            FunctionDescriptor::new(function),
        ));
        (device.new_compute_pipeline_state(&descriptor, |samplers| Ok(vec![0; samplers.len()])))
            .unwrap_or_else(|error| panic!("{name}: {error}"))
    }

    /// The writing kernel's state.
    fn state(name: &str, device: &Device, library: &Library) -> ComputePipelineState {
        kernel_state(name, device, library, "level_write")
    }

    /// The side of level `level`.
    fn side(level: u32) -> u32 {
        (SIDE >> level).max(1)
    }

    /// The fill of texel `(x, y)` of level `level`: no texel the functions write holds it.
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

    /// Asserts that each level of `texture` holds `want(level, x, y)` at each texel.
    fn holds(name: &str, texture: &Texture, want: impl Fn(u32, u32, u32) -> [f32; 4]) {
        for level in 0..LEVELS {
            let side = side(level);
            let want: Vec<[f32; 4]> = (0..side * side)
                .map(|i| want(level, i % side, i / side))
                .collect();
            assert_eq!(
                level_texels(name, texture, level),
                want,
                "{name}: level {level}"
            );
        }
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

    /// Binds `texture` (None: nil) to the kernel's slot and dispatches `grid` threads at level
    /// `level`.
    fn dispatch(
        encoder: &mut ComputeCommandEncoder,
        texture: Option<&Texture>,
        level: u32,
        grid: u32,
    ) -> Result<(), Error> {
        encoder.set_texture(texture, 0)?;
        encoder.set_bytes(&level.to_le_bytes(), 0)?;
        encoder.dispatch_threads([grid, grid, 1], [1, 1, 1])
    }

    #[test]
    fn every_device_writes_the_levels_a_kernel_reads_from_a_buffer() {
        let Some(library) = library(LEVEL_WRITE) else {
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
            for level in 0..3u32 {
                call(encoder.set_bytes(&level.to_le_bytes(), 0));
                call(encoder.dispatch_threads([side(level), side(level), 1], [1, 1, 1]));
            }
            encoder.end_encoding();
            command_buffer.commit().expect("a commit");
            finished(name, &command_buffer);
            assert_eq!(command_buffer.error(), None, "{name}");
            holds(name, &texture, |level, x, y| match level {
                0..3 => [x as f32, y as f32, level as f32, 1.0],
                _ => fill(level, x, y),
            });
        });
    }

    #[test]
    fn every_device_counts_the_dispatches_that_push_level_fields() {
        let (Some(writes), Some(counts)) = (library(LEVEL_WRITE), library(LEVEL_COUNT)) else {
            return;
        };
        each_device("storage_levels_counted", |name, _, device| {
            let counters = device.counters();
            let pushed = || counters.metrics().level_field_commands;
            let writes = state(name, &device, &writes);
            let counts = kernel_state(name, &device, &counts, "level_count");
            let _grant = (device.grant_indices(IndexClass::Texture, 1, 1))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let texture = filled(name, &device, 1);
            // The arena keeps the buffer's block.
            let mut arena = Arena::new(&device);
            let out = arena.buffer();
            let queue = device.new_command_queue(1).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let call = |result: Result<(), Error>| {
                result.unwrap_or_else(|error| panic!("{name}: {error}"))
            };
            let before = pushed();
            let mut encoder = (command_buffer.compute_command_encoder(DispatchType::Serial))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            call(encoder.set_compute_pipeline_state(&writes));
            call(dispatch(&mut encoder, Some(&texture), 1, side(1)));
            encoder.end_encoding();
            assert_eq!(pushed() - before, 1, "{name}: the level write");
            let mut encoder = (command_buffer.compute_command_encoder(DispatchType::Serial))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            call(encoder.set_compute_pipeline_state(&counts));
            call(encoder.set_texture(Some(&texture), 0));
            call(encoder.set_buffer(Some(&out), 0, 1));
            call(encoder.dispatch_threads([1, 1, 1], [1, 1, 1]));
            encoder.end_encoding();
            assert_eq!(pushed() - before, 2, "{name}: the level count");
            command_buffer.commit().expect("a commit");
            finished(name, &command_buffer);
            assert_eq!(command_buffer.error(), None, "{name}");
            assert_eq!(words(&out)[0], LEVELS, "{name}: get_num_mip_levels");
        });
    }

    #[test]
    fn every_device_drops_a_write_past_the_level_count_and_at_nil() {
        let Some(library) = library(LEVEL_WRITE) else {
            return;
        };
        each_device("storage_levels_dropped", |name, _, device| {
            let state = state(name, &device, &library);
            let _grant = (device.grant_indices(IndexClass::Texture, 1, 1))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let texture = filled(name, &device, 1);
            let queue = device.new_command_queue(1).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = (command_buffer.compute_command_encoder(DispatchType::Serial))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            (encoder.set_compute_pipeline_state(&state))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            for level in [LEVELS, 9, u32::MAX] {
                (dispatch(&mut encoder, Some(&texture), level, SIDE))
                    .unwrap_or_else(|error| panic!("{name}: level {level}: {error}"));
            }
            for level in [0, 1] {
                (dispatch(&mut encoder, None, level, SIDE))
                    .unwrap_or_else(|error| panic!("{name}: nil at level {level}: {error}"));
            }
            encoder.end_encoding();
            command_buffer.commit().expect("a commit");
            finished(name, &command_buffer);
            assert_eq!(command_buffer.error(), None, "{name}");
            holds(name, &texture, fill);
        });
    }

    #[test]
    fn every_device_binds_a_texture_again_with_no_view_made() {
        let Some(library) = library(LEVEL_WRITE) else {
            return;
        };
        each_device("storage_levels_rebind", |name, _, device| {
            let counters = device.counters();
            let views = || counters.metrics().image_view_creates.count;
            let runs = || counters.gauges().level_elements;
            let state = state(name, &device, &library);
            let _grant = (device.grant_indices(IndexClass::Texture, 1, 2))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let (a, b) = (filled(name, &device, 1), filled(name, &device, 2));
            let made = views();
            let queue = device.new_command_queue(1).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = (command_buffer.compute_command_encoder(DispatchType::Serial))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            (encoder.set_compute_pipeline_state(&state))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let mut bind = |texture, level| {
                (dispatch(&mut encoder, Some(texture), level, side(level)))
                    .unwrap_or_else(|error| panic!("{name}: level {level}: {error}"))
            };
            bind(&a, 1);
            assert_eq!(views() - made, u64::from(LEVELS), "{name}: a's first bind");
            assert_eq!(runs(), 1, "{name}: a's first bind");
            bind(&b, 2);
            assert_eq!(
                views() - made,
                2 * u64::from(LEVELS),
                "{name}: b's first bind"
            );
            assert_eq!(runs(), 2, "{name}: b's first bind");
            for _ in 0..10 {
                bind(&a, 1);
                bind(&b, 2);
            }
            assert_eq!(
                views() - made,
                2 * u64::from(LEVELS),
                "{name}: the binds again"
            );
            assert_eq!(runs(), 2, "{name}: the binds again");
            encoder.end_encoding();
            command_buffer.commit().expect("a commit");
            finished(name, &command_buffer);
            assert_eq!(command_buffer.error(), None, "{name}");
            let written = |at: u32| {
                move |level: u32, x: u32, y: u32| match level == at {
                    true => [x as f32, y as f32, level as f32, 1.0],
                    false => fill(level, x, y),
                }
            };
            holds(name, &a, written(1));
            holds(name, &b, written(2));
            drop(command_buffer);
            drop((a, b));
            within("the runs' return", || runs() == 0);
        });
    }

    #[test]
    fn every_device_writes_a_level_a_fragment_function_reads_from_a_buffer() {
        let (Some(examples), Some(library)) = (examples(), library(LEVEL_WRITE_FRAGMENT)) else {
            return;
        };
        let mut descriptor = RenderPipelineDescriptor::new(StageSource::Function(
            FunctionDescriptor::new(examples.function("colored_vertex").expect("the function")),
        ));
        descriptor.fragment = Some(StageSource::Function(FunctionDescriptor::new(
            (library.function("level_write_fragment")).expect("the function"),
        )));
        descriptor.color_attachments[0] =
            Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
        each_device("storage_levels_draw", |name, _, device| {
            let counters = device.counters();
            let pushed = || counters.metrics().level_field_commands;
            let state = render_state(name, &device, &descriptor);
            let _grant = (device.grant_indices(IndexClass::Texture, 1, 1))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let texture = filled(name, &device, 1);
            let mut buffers = Buffers::new(&device);
            let (positions, colors) = vertices(&device, &mut buffers);
            let color = target(
                &device,
                PixelFormat::RGBA8Unorm,
                TextureUsage::RENDER_TARGET,
            );
            let queue = device.new_command_queue(1).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = (command_buffer.render_command_encoder(&pass(&color, None)))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let call = |result: Result<(), Error>| {
                result.unwrap_or_else(|error| panic!("{name}: {error}"))
            };
            call(encoder.set_render_pipeline_state(&state));
            call(encoder.set_vertex_buffer(Some(&positions), 0, 0));
            call(encoder.set_vertex_buffer(Some(&colors), 0, 1));
            call(encoder.set_fragment_texture(Some(&texture), 0));
            let before = pushed();
            for level in [1u32, 2, 5] {
                call(encoder.set_fragment_bytes(&level.to_le_bytes(), 0));
                call(encoder.draw_primitives(PrimitiveType::Triangle, 3, 3, 1, 0));
            }
            encoder.end_encoding();
            assert_eq!(pushed() - before, 3, "{name}: the draws");
            run(name, &command_buffer);
            holds(name, &texture, |level, x, y| match (level, x, y) {
                (1 | 2, 0, 0) => [1.0, 2.0, level as f32, 1.0],
                _ => fill(level, x, y),
            });
        });
    }

    #[test]
    fn every_device_fills_the_level_elements_past_their_capacity() {
        let Some(library) = library(LEVEL_WRITE) else {
            return;
        };
        each_device("storage_levels_full", |name, _, device| {
            let counters = device.counters();
            let runs = || counters.gauges().level_elements;
            let state = state(name, &device, &library);
            let _grant = (device.grant_indices(IndexClass::Texture, 1, 4097))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let mut d = TextureDescriptor::new(TextureType::Type2D, PixelFormat::RGBA8Unorm, 2, 2);
            d.mipmap_level_count = 2;
            d.usage = TextureUsage::SHADER_WRITE;
            d.storage_mode = StorageMode::Private;
            let mut textures: Vec<Texture> = (1..=4097)
                .map(|index| {
                    (device.new_texture(&d, Some(index)))
                        .unwrap_or_else(|error| panic!("{name}: texture {index}: {error}"))
                })
                .collect();
            let queue = device.new_command_queue(2).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = (command_buffer.compute_command_encoder(DispatchType::Concurrent))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            (encoder.set_compute_pipeline_state(&state))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            for (at, texture) in textures[..4096].iter().enumerate() {
                (dispatch(&mut encoder, Some(texture), 1, 1))
                    .unwrap_or_else(|error| panic!("{name}: texture {}: {error}", at + 1));
            }
            assert_eq!(runs(), 4096, "{name}");
            match dispatch(&mut encoder, Some(&textures[4096]), 1, 1) {
                Err(Error::Limit(limit)) => {
                    assert_eq!(limit, Limit::LevelElements { max: 4096 }, "{name}")
                }
                other => panic!("{name}: the 4,097th texture's dispatch: {other:?}"),
            }
            assert_eq!(runs(), 4096, "{name}: the refused dispatch holds no run");
            encoder.end_encoding();
            command_buffer.commit().expect("a commit");
            finished(name, &command_buffer);
            assert_eq!(command_buffer.error(), None, "{name}");
            drop(command_buffer);
            drop(textures.remove(0));
            within("the first texture's run", || runs() == 4095);
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = (command_buffer.compute_command_encoder(DispatchType::Concurrent))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            (encoder.set_compute_pipeline_state(&state))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            (dispatch(&mut encoder, Some(&textures[4095]), 1, 1))
                .unwrap_or_else(|error| panic!("{name}: the 4,097th texture, again: {error}"));
            assert_eq!(runs(), 4096, "{name}");
            encoder.end_encoding();
            command_buffer.commit().expect("a commit");
            finished(name, &command_buffer);
            assert_eq!(command_buffer.error(), None, "{name}");
            drop(command_buffer);
            drop(textures);
            within("every run's return", || runs() == 0);
        });
    }
}
