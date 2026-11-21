//! A cube view of six slices of a square 2D array, under a device cfg on every device of the
//! cfg's kind. A square 2D array of six slices or more is made cube-compatible where the device's
//! image format query takes the flag, so Metal's cast of six of its slices as a cube
//! (`newTextureViewWithPixelFormat:textureType:levels:slices:`) is a cube view of its image: the
//! owned fixture `tvc_samples`'s `tvc_cube` samples every texel centre of every face of a cube view
//! of slices 1 to 6 of an 8-slice `RGBA32Float` array, and each sample is the texel of the slice
//! under the face (face `f` is slice `1 + f`), by the face table Metal and Vulkan share. A cube
//! view of a non-square 2D array is `Gap::TextureView` naming `TextureViewKind::CubeOfArray`.
//!
//! The flag costs no memory: square 2D arrays of six slices or more, of a format of each kind
//! the runtime makes (colour, mutable colour, depth, block-compressed) and several extents and
//! level counts, have the memory requirements with the flag that they have without it, wherever
//! the device makes both.
//!
//! A cube array texture (`imageCubeArray`, in the floor) is sampled as one, and Metal's cast of its
//! first six faces as a cube is a cube view of its image: the owned fixture `unk_cube_grad_lod`
//! samples the cube view and the array's second cube at two directions on each face, and over a
//! cube array of two cubes whose slice `s` holds `s` in every texel, each lane reads its face from
//! the view and six past it from the array.
//!
//! `llvm-as` assembles the fixtures; without it the sampling tests skip visibly.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::command::on_device::{each_device, finished};
    use crate::tests::draw::on_device::{floats, Buffers};
    use crate::{
        ComputePipelineDescriptor, DispatchType, Error, FunctionDescriptor, Gap, IndexClass,
        Library, NumberedSampler, PixelFormat, Region, StageSource, StorageMode, TextureDescriptor,
        TextureType, TextureUsage, TextureViewKind,
    };
    use ash::vk;
    use metal2vulkan::device::VkFeature;
    use metal2vulkan_devtools::{engine_dir, llvm_assemble, AssembleError};

    /// The array's side, and its slices.
    const SIDE: u32 = 4;
    const SLICES: u32 = 8;
    /// The array's slice under the cube view's first face.
    const FIRST: u32 = 1;
    /// The constexpr sampler's index the test gives `tvc_cube`.
    const CONSTEXPR: u32 = 1;

    /// The engine fixture `file`'s library, or None where `llvm-as` is missing.
    fn library(file: &str) -> Option<Library> {
        match llvm_assemble(&engine_dir().join(file)) {
            Ok(bitcode) => Some(Library::from_air(&bitcode).expect("the library")),
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                None
            }
            Err(error) => panic!("{file}: {error:?}"),
        }
    }

    /// The array's texel `(x, y)` of slice `slice`: each channel names it.
    fn texel(slice: u32, x: u32, y: u32) -> [f32; 4] {
        [
            slice as f32,
            x as f32,
            y as f32,
            (1000 + slice * 16 + y * SIDE + x) as f32,
        ]
    }

    #[test]
    fn every_device_samples_a_cube_view_of_six_slices_of_a_2d_array_as_its_faces() {
        let Some(library) = library("tvc_samples.ll") else {
            return;
        };
        each_device("cube view of a 2D array", |name, _, device| {
            let function = library.function("tvc_cube").expect("the kernel");
            let descriptor = ComputePipelineDescriptor::new(StageSource::Function(
                FunctionDescriptor::new(function),
            ));
            let _samplers = (device.grant_indices(IndexClass::Sampler, CONSTEXPR, 1))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let _textures = (device.grant_indices(IndexClass::Texture, 1, 3))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let state = (device.new_compute_pipeline_state(&descriptor, |samplers| {
                assert!(
                    matches!(samplers, [NumberedSampler::Constexpr(_)]),
                    "{name}: {samplers:?}"
                );
                Ok(vec![CONSTEXPR])
            }))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
            let mut d = TextureDescriptor::new(
                TextureType::Type2DArray,
                PixelFormat::RGBA32Float,
                SIDE,
                SIDE,
            );
            d.array_length = SLICES;
            d.usage = TextureUsage::SHADER_READ;
            d.storage_mode = StorageMode::Shared;
            let array = (device.new_texture(&d, Some(1)))
                .unwrap_or_else(|error| panic!("{name}: the array: {error}"));
            for slice in 0..SLICES {
                let texels: Vec<f32> = (0..SIDE)
                    .flat_map(|y| (0..SIDE).flat_map(move |x| texel(slice, x, y)))
                    .collect();
                let bytes = floats(&texels);
                let region = Region {
                    origin: [0; 3],
                    size: [SIDE, SIDE, 1],
                };
                let row = u64::from(SIDE * 16);
                (array.replace_region(
                    region,
                    0,
                    slice,
                    crate::HostBytes::from(&bytes[..]),
                    row,
                    row * u64::from(SIDE),
                ))
                .unwrap_or_else(|error| panic!("{name}: slice {slice}: {error}"));
            }
            let cube = (array.new_texture_view(
                PixelFormat::RGBA32Float,
                TextureType::Cube,
                0..1,
                FIRST..FIRST + 6,
                Some(2),
            ))
            .unwrap_or_else(|error| panic!("{name}: the cube view: {error}"));
            let view = cube.descriptor();
            assert_eq!(
                (
                    view.texture_type,
                    view.width,
                    view.height,
                    view.array_length
                ),
                (TextureType::Cube, SIDE, SIDE, 1),
                "{name}: the view's descriptor"
            );
            // Every texel centre of every face, at level 0: (level, face, x, y).
            let points: Vec<[u32; 4]> = (0..6)
                .flat_map(|face| {
                    (0..SIDE).flat_map(move |y| (0..SIDE).map(move |x| [0, face, x, y]))
                })
                .collect();
            let words: Vec<u8> = (points.iter().flatten())
                .flat_map(|word| word.to_le_bytes())
                .collect();
            let mut buffers = Buffers::new(&device);
            let at = buffers.of(&device, &words);
            let out = buffers.of(&device, &vec![0; points.len() * 16]);
            let queue = device.new_command_queue(1).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder =
                (command_buffer.compute_command_encoder(DispatchType::Serial)).expect("an encoder");
            encoder
                .set_compute_pipeline_state(&state)
                .expect("the state");
            encoder.set_texture(Some(&cube), 0).expect("the cube");
            encoder.set_buffer(Some(&at), 0, 0).expect("at");
            encoder.set_buffer(Some(&out), 0, 1).expect("out");
            let groups = (points.len() as u32).div_ceil(32);
            (encoder.dispatch_threadgroups([groups, 1, 1], [32, 1, 1]))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            encoder.end_encoding();
            command_buffer.commit().expect("a commit");
            finished(name, &command_buffer);
            assert_eq!(command_buffer.error(), None, "{name}");
            let contents = out.contents().expect("a Shared buffer");
            // SAFETY: the dispatch that wrote the buffer completed, and its writes reach the host.
            let samples = unsafe {
                std::slice::from_raw_parts(contents.as_ptr().cast::<[f32; 4]>(), points.len())
            };
            for (&[_, face, x, y], sample) in points.iter().zip(samples) {
                assert_eq!(
                    *sample,
                    texel(FIRST + face, x, y),
                    "{name}: face {face}, texel ({x}, {y})"
                );
            }
            // A non-square 2D array's image is not cube-compatible.
            let mut d = TextureDescriptor::new(
                TextureType::Type2DArray,
                PixelFormat::RGBA32Float,
                SIDE,
                2 * SIDE,
            );
            d.array_length = 6;
            d.usage = TextureUsage::SHADER_READ;
            let wide = (device.new_texture(&d, Some(3)))
                .unwrap_or_else(|error| panic!("{name}: the non-square array: {error}"));
            match wide.new_texture_view(
                PixelFormat::RGBA32Float,
                TextureType::Cube,
                0..1,
                0..6,
                None,
            ) {
                Err(Error::Unsupported(Gap::TextureView {
                    kind: TextureViewKind::CubeOfArray,
                })) => {}
                Err(error) => panic!("{name}: a cube view of a non-square array: {error}"),
                Ok(_) => panic!("{name}: a cube view of a non-square array was made"),
            }
            drop((cube, array, command_buffer));
            println!(
                "{name}: a cube view of slices {FIRST} to {} of a {SIDE} x {SIDE} 2D array of \
                 {SLICES} slices sampled {} texel centres, each the slice's texel under its \
                 face; a non-square array's cube view is CubeOfArray",
                FIRST + 5,
                points.len()
            );
        });
    }

    /// The two directions `unk_cube_grad_lod` samples on each face, `in[face]` and `in[6 + face]`
    /// (faces +x, -x, +y, -y, +z, -z), each with the face's axis its major one.
    const DIRECTIONS: [[f32; 3]; 12] = [
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.25, -0.5],
        [-1.0, -0.5, 0.25],
        [0.25, 1.0, -0.5],
        [-0.5, -1.0, 0.25],
        [0.25, -0.5, 1.0],
        [-0.5, 0.25, -1.0],
    ];

    #[test]
    fn every_device_samples_a_cube_array_and_a_cube_view_of_its_first_cube() {
        let Some(library) = library("unk_cube_grad_lod.ll") else {
            return;
        };
        each_device("cube array", |name, _, device| {
            let enabled = &device.core().translator.profile().enabled().features;
            assert!(
                enabled.contains(&VkFeature::ImageCubeArray),
                "{name}: imageCubeArray, which the floor requires"
            );
            let function = library.function("unk_cube_grad_lod").expect("the kernel");
            let descriptor = ComputePipelineDescriptor::new(StageSource::Function(
                FunctionDescriptor::new(function),
            ));
            let _samplers = (device.grant_indices(IndexClass::Sampler, CONSTEXPR, 1))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let _textures = (device.grant_indices(IndexClass::Texture, 1, 2))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let state = (device.new_compute_pipeline_state(&descriptor, |samplers| {
                assert!(
                    matches!(samplers, [NumberedSampler::Constexpr(_)]),
                    "{name}: {samplers:?}"
                );
                Ok(vec![CONSTEXPR])
            }))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
            let mut d = TextureDescriptor::new(
                TextureType::CubeArray,
                PixelFormat::RGBA32Float,
                SIDE,
                SIDE,
            );
            d.array_length = 2;
            d.usage = TextureUsage::SHADER_READ;
            d.storage_mode = StorageMode::Shared;
            let cubes = (device.new_texture(&d, Some(1)))
                .unwrap_or_else(|error| panic!("{name}: the cube array: {error}"));
            let region = Region {
                origin: [0; 3],
                size: [SIDE, SIDE, 1],
            };
            let row = u64::from(SIDE * 16);
            for slice in 0..12 {
                let texels: Vec<f32> = (0..SIDE * SIDE)
                    .flat_map(|_| [slice as f32, 0.0, 0.0, 1.0])
                    .collect();
                let bytes = floats(&texels);
                (cubes.replace_region(
                    region,
                    0,
                    slice,
                    crate::HostBytes::from(&bytes[..]),
                    row,
                    row * u64::from(SIDE),
                ))
                .unwrap_or_else(|error| panic!("{name}: slice {slice}: {error}"));
            }
            let cube = (cubes.new_texture_view(
                PixelFormat::RGBA32Float,
                TextureType::Cube,
                0..1,
                0..6,
                Some(2),
            ))
            .unwrap_or_else(|error| panic!("{name}: the cube view: {error}"));
            assert_eq!(
                (
                    cubes.descriptor().array_length,
                    cube.descriptor().texture_type
                ),
                (2, TextureType::Cube),
                "{name}: the descriptors"
            );
            // The directions, then the six gradient axes, then the gradient's magnitude: 0, so
            // every lane samples level 0.
            let mut input: Vec<f32> = (DIRECTIONS.iter())
                .flat_map(|&[x, y, z]| [x, y, z, 0.0])
                .collect();
            for axis in 0..6 {
                let mut g = [0.0; 4];
                g[axis / 2] = if axis % 2 == 0 { 1.0 } else { -1.0 };
                input.extend(g);
            }
            input.extend([0.0; 4]);
            const LANES: usize = 72;
            let mut buffers = Buffers::new(&device);
            let at = buffers.of(&device, &floats(&input));
            let out = buffers.of(&device, &vec![0; LANES * 16]);
            let queue = device.new_command_queue(1).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder =
                (command_buffer.compute_command_encoder(DispatchType::Serial)).expect("an encoder");
            encoder
                .set_compute_pipeline_state(&state)
                .expect("the state");
            encoder.set_texture(Some(&cube), 0).expect("the cube view");
            encoder
                .set_texture(Some(&cubes), 1)
                .expect("the cube array");
            encoder.set_buffer(Some(&at), 0, 0).expect("in");
            encoder.set_buffer(Some(&out), 0, 1).expect("out");
            (encoder.dispatch_threadgroups([3, 1, 1], [24, 1, 1]))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            encoder.end_encoding();
            command_buffer.commit().expect("a commit");
            finished(name, &command_buffer);
            assert_eq!(command_buffer.error(), None, "{name}");
            let contents = out.contents().expect("a Shared buffer");
            // SAFETY: the dispatch that wrote the buffer completed, and its writes reach the host.
            let samples =
                unsafe { std::slice::from_raw_parts(contents.as_ptr().cast::<[f32; 4]>(), LANES) };
            for (lane, sample) in samples.iter().enumerate() {
                let face = (lane % 6) as f32;
                assert_eq!(
                    *sample,
                    [face, 6.0 + face, face, 6.0 + face],
                    "{name}: lane {lane}: the view's face, then the second cube's"
                );
            }
            drop((cube, cubes, command_buffer));
            println!(
                "{name}: imageCubeArray enabled; {LANES} lanes sampled a cube array of two \
                 {SIDE} x {SIDE} cubes at its second cube and a cube view of its first, each \
                 the face's slice"
            );
        });
    }

    /// Each format kind's image: its name, format, usage and the formats a mutable one lists.
    type Kind = (
        &'static str,
        vk::Format,
        vk::ImageUsageFlags,
        &'static [vk::Format],
    );

    /// The format kinds compared: colour, mutable colour, depth and block-compressed.
    fn kinds() -> [Kind; 6] {
        use vk::ImageUsageFlags as U;
        let copies = U::TRANSFER_SRC | U::TRANSFER_DST;
        let color = U::SAMPLED | copies | U::COLOR_ATTACHMENT;
        let depth = U::SAMPLED | copies | U::DEPTH_STENCIL_ATTACHMENT;
        const SRGB: &[vk::Format] = &[vk::Format::R8G8B8A8_UNORM, vk::Format::R8G8B8A8_SRGB];
        [
            ("RGBA8Unorm", vk::Format::R8G8B8A8_UNORM, color, &[]),
            (
                "RGBA8Unorm mutable to sRGB",
                vk::Format::R8G8B8A8_UNORM,
                color,
                SRGB,
            ),
            ("RGBA16Float", vk::Format::R16G16B16A16_SFLOAT, color, &[]),
            ("R32Float", vk::Format::R32_SFLOAT, color, &[]),
            ("Depth32Float", vk::Format::D32_SFLOAT, depth, &[]),
            (
                "BC1RGBA",
                vk::Format::BC1_RGBA_UNORM_BLOCK,
                U::SAMPLED | copies,
                &[],
            ),
        ]
    }

    /// The extents, slices and levels compared: a square 2D array each.
    const SHAPES: [(u32, u32, u32); 4] = [(1, 6, 1), (64, 6, 7), (100, 7, 7), (256, 12, 9)];

    #[test]
    fn every_device_gives_a_square_2d_array_the_same_memory_with_or_without_cube_compatibility() {
        each_device("cube-compatible memory", |name, _, device| {
            let core = device.core();
            let (mut compared, mut skipped) = (0, Vec::new());
            for (kind, format, usage, list) in kinds() {
                for (side, layers, levels) in SHAPES {
                    let base = match list.is_empty() {
                        true => vk::ImageCreateFlags::empty(),
                        false => vk::ImageCreateFlags::MUTABLE_FORMAT,
                    };
                    let mut needs = Vec::new();
                    for flags in [base, base | vk::ImageCreateFlags::CUBE_COMPATIBLE] {
                        let mut query_list =
                            vk::ImageFormatListCreateInfo::default().view_formats(list);
                        let mut query = vk::PhysicalDeviceImageFormatInfo2::default()
                            .format(format)
                            .ty(vk::ImageType::TYPE_2D)
                            .tiling(vk::ImageTiling::OPTIMAL)
                            .usage(usage)
                            .flags(flags);
                        if !list.is_empty() {
                            query = query.push_next(&mut query_list);
                        }
                        let Some(properties) = core.image_format_properties(&query) else {
                            break;
                        };
                        if properties.max_array_layers < layers
                            || properties.max_extent.width < side
                        {
                            break;
                        }
                        let mut info_list =
                            vk::ImageFormatListCreateInfo::default().view_formats(list);
                        let mut info = vk::ImageCreateInfo::default()
                            .flags(flags)
                            .image_type(vk::ImageType::TYPE_2D)
                            .format(format)
                            .extent(vk::Extent3D {
                                width: side,
                                height: side,
                                depth: 1,
                            })
                            .mip_levels(levels)
                            .array_layers(layers)
                            .samples(vk::SampleCountFlags::TYPE_1)
                            .tiling(vk::ImageTiling::OPTIMAL)
                            .usage(usage)
                            .sharing_mode(vk::SharingMode::EXCLUSIVE)
                            .initial_layout(vk::ImageLayout::UNDEFINED);
                        if !list.is_empty() {
                            info = info.push_next(&mut info_list);
                        }
                        let mut dedicated = vk::MemoryDedicatedRequirements::default();
                        let mut got = vk::MemoryRequirements2::default().push_next(&mut dedicated);
                        let request =
                            vk::DeviceImageMemoryRequirements::default().create_info(&info);
                        // SAFETY: the device is live, and the image format query takes the info.
                        unsafe {
                            core.device
                                .get_device_image_memory_requirements(&request, &mut got)
                        };
                        let r = got.memory_requirements;
                        needs.push((
                            r.size,
                            r.alignment,
                            r.memory_type_bits,
                            dedicated.prefers_dedicated_allocation,
                            dedicated.requires_dedicated_allocation,
                        ));
                    }
                    match needs.as_slice() {
                        [without, with] => {
                            assert_eq!(
                                without, with,
                                "{name}: {kind}, {side} x {side}, {layers} slices, {levels} \
                                 levels: (size, alignment, memory types, prefers and requires \
                                 a dedicated allocation) without and with the flag"
                            );
                            compared += 1;
                        }
                        _ => skipped.push(format!("{kind} {side}x{side}x{layers}")),
                    }
                }
            }
            assert!(compared > 0, "{name}: no image compared");
            println!(
                "{name}: {compared} square 2D arrays take the same memory with and without \
                 cube compatibility; not made both ways: {skipped:?}"
            );
        });
    }
}
