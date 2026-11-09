//! The engine compute candidate with textures, texture buffers, texture arrays, and samplers
//! on a device: the owned `texture_ops`, `sampling`, and `msf_mip_levels` kernels and the public
//! `texture_array_width` kernel, planned by the binder and executed on the executor's own context
//! under `VK_LAYER_KHRONOS_validation`, against expectations derived from their MSL. `G-linux`'s
//! `m2v_cpu_icd` step runs these on lavapipe. Every kernel is loop-free.

use super::engine_compute::run;
use crate::candidate::engine::{resources, ComputeCase, ComputeOutput};
use crate::case::mips::level_dimensions;
use crate::case::{
    ResourceRole, SamplerAddressMode, SamplerFilter, SamplerMipFilter, SamplerResource,
    TextureFormat, TextureType,
};
use crate::literal::{LiteralTexture, LiteralTextureArray};
use metal2vulkan::bind::Dispatch;
use metal2vulkan::library::FunctionConstantValues;
use metal2vulkan_devtools::{engine_dir, fixtures_dir, llvm_assemble};
use std::collections::BTreeMap;

fn texture(
    binding: u32,
    texture_type: TextureType,
    format: TextureFormat,
    dimensions: [u32; 3],
    bytes: Vec<u8>,
) -> LiteralTexture {
    LiteralTexture {
        binding,
        role: ResourceRole::InOut,
        texture_type,
        format,
        dimensions,
        sample_count: 1,
        bytes,
        sparse: None,
        mip_levels: 1,
    }
}

fn floats(values: &[f32]) -> Vec<u8> {
    values.iter().copied().flat_map(f32::to_le_bytes).collect()
}

fn words(values: &[u32]) -> Vec<u8> {
    values.iter().copied().flat_map(u32::to_le_bytes).collect()
}

/// A kernel case over `textures` (and one output buffer when `out` is non-empty) dispatching
/// `threads` one thread per group.
fn kernel(
    entry: &str,
    textures: Vec<LiteralTexture>,
    out: Vec<u8>,
    threads: [u32; 3],
    output: ComputeOutput,
) -> ComputeCase {
    ComputeCase {
        out_of_bounds: Vec::new(),
        creation_bound: crate::budgets::PIPELINE_CREATION_CEILING,
        entry: entry.into(),
        constants: FunctionConstantValues::defaults(),
        buffers: if out.is_empty() {
            BTreeMap::new()
        } else {
            BTreeMap::from([(0, out)])
        },
        argument_buffer_buffers: BTreeMap::new(),
        argument_buffer_textures: BTreeMap::new(),
        argument_buffer_samplers: BTreeMap::new(),
        rate_maps: BTreeMap::new(),
        argument_buffer_rate_maps: BTreeMap::new(),
        objects: std::collections::BTreeMap::new(),
        command_buffers: std::collections::BTreeMap::new(),
        textures: textures
            .into_iter()
            .map(|texture| (texture.binding, texture))
            .collect(),
        samplers: BTreeMap::new(),
        function_tables: BTreeMap::new(),
        acceleration_structures: BTreeMap::new(),
        threadgroup_lengths: BTreeMap::new(),
        imageblock: None,
        tile: None,
        stage_input: None,
        stage_in_region: None,
        attribute_strides: Default::default(),
        dispatch: Dispatch::Threads {
            threads,
            threads_per_group: [1, 1, 1],
        },
        output,
    }
}

fn buffer_output(length: u64) -> ComputeOutput {
    ComputeOutput::Buffer {
        binding: 0,
        offset: 0,
        length,
    }
}

fn fixture(name: &str) -> Vec<u8> {
    llvm_assemble(&engine_dir().join(name)).expect("llvm-as on the device lane")
}

/// Each element of a texture array is the texture bound at its own slot: the public
/// `texture_array_width` kernel (`array<texture2d<float>, 2> [[texture(0)]]`) writes the width of
/// element 1, which is 7 texels wide where element 0 is 2.
#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn a_texture_array_element_is_the_texture_at_its_slot() {
    let element = |width: u32| {
        texture(
            0,
            TextureType::D2,
            TextureFormat::R32Float,
            [width, 1, 1],
            vec![0; 4 * width as usize],
        )
    };
    let mut case = kernel(
        "texture_array_width",
        Vec::new(),
        vec![0xab; 4],
        [1, 1, 1],
        buffer_output(4),
    );
    case.textures = resources::bound_textures(
        &[],
        &[LiteralTextureArray {
            binding: 0,
            elements: vec![element(2), element(7)],
            overrides_texture_at_base: false,
        }],
    )
    .unwrap();
    let bitcode = llvm_assemble(&fixtures_dir().join("kernel_texture_array_width.ll"))
        .expect("llvm-as on the device lane");
    let run = run(&case, &bitcode).unwrap();
    assert_eq!(run.output, words(&[7]));
}

/// A uniform texel buffer read and a storage texel buffer written and observed.
#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn texel_buffers_read_one_view_and_write_the_other() {
    let source = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let case = kernel(
        "texel_buffers",
        vec![
            texture(
                0,
                TextureType::Buffer,
                TextureFormat::Rgba32Float,
                [2, 1, 1],
                floats(&source),
            ),
            texture(
                1,
                TextureType::Buffer,
                TextureFormat::Rgba32Float,
                [2, 1, 1],
                vec![0xab; 32],
            ),
        ],
        Vec::new(),
        [2, 1, 1],
        ComputeOutput::Texture {
            binding: 1,
            origin: [0, 0, 0],
            dimensions: [2, 1, 1],
        },
    );
    let run = run(&case, &fixture("texture_ops.ll")).unwrap();
    let doubled: Vec<f32> = source.iter().map(|value| value * 2.0).collect();
    assert_eq!(run.output, floats(&doubled));
}

/// The public out-of-bounds texture-buffer case (`ob1_texel_buffer`): reads and writes past two
/// 4-texel `r32_uint` texture buffers. Each is host-visible buffer memory, which the candidate
/// allocates inside a guard of the case's reach, so the census counts the guard bytes each run
/// changed: none, and the written buffer's own texels are the in-bounds writes.
#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn a_texture_buffer_written_past_its_end_is_guarded() {
    use crate::case::out_of_bounds::{ContainmentForm, OutOfBoundsResource, OutOfBoundsTarget};
    let idx = [0u32, 1, 2, 3, 4, 5, 64, 1000, 3, 2, 1, 0, 4, 8, 100, 4096];
    let mut case = kernel(
        "ob1_texel_buffer",
        vec![
            texture(
                0,
                TextureType::Buffer,
                TextureFormat::R32Uint,
                [4, 1, 1],
                words(&[0xF100_0001, 0xF100_0002, 0xF100_0003, 0xF100_0004]),
            ),
            texture(
                1,
                TextureType::Buffer,
                TextureFormat::R32Uint,
                [4, 1, 1],
                vec![0xab; 16],
            ),
        ],
        Vec::new(),
        [8, 1, 1],
        ComputeOutput::Texture {
            binding: 1,
            origin: [0, 0, 0],
            dimensions: [4, 1, 1],
        },
    );
    case.buffers = BTreeMap::from([(0, words(&idx)), (1, vec![0xab; 32])]);
    case.out_of_bounds = [(0, 4000), (1, 16384)]
        .into_iter()
        .map(|(binding, reach_after)| OutOfBoundsTarget {
            resource: OutOfBoundsResource::Texture { binding },
            form: ContainmentForm::TexelBuffer,
            reach_before: 0,
            reach_after,
        })
        .collect();
    let run = run(&case, &fixture("ob1_texel_buffer.ll")).unwrap();
    assert_eq!(
        run.output,
        words(&[0xA000_0003, 0xA000_0002, 0xA000_0001, 0xA000_0000])
    );
    let targets = run
        .census
        .out_of_bounds
        .expect("an out-of-bounds case's census records its targets")
        .targets;
    assert_eq!(targets.len(), 2);
    for target in targets {
        assert_eq!(target.escaped, Some(0), "{target:?}");
        assert!(target.past_end.is_some(), "{target:?}");
    }
}

/// A `read_write` texture read, written, fenced, and read again, observed as a texture region.
#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn a_read_write_texture_sees_its_own_fenced_write() {
    let initial = [10u32, 20, 30, 40, 1, 1, 1, 1];
    let case = kernel(
        "read_write_fence",
        vec![texture(
            0,
            TextureType::D2,
            TextureFormat::Rgba32Uint,
            [2, 1, 1],
            words(&initial),
        )],
        Vec::new(),
        [2, 1, 1],
        ComputeOutput::Texture {
            binding: 0,
            origin: [0, 0, 0],
            dimensions: [2, 1, 1],
        },
    );
    let run = run(&case, &fixture("texture_ops.ll")).unwrap();
    let expected: Vec<u32> = initial
        .iter()
        .zip([1, 2, 3, 4].iter().cycle())
        .map(|(value, add)| (value + add) * 2)
        .collect();
    assert_eq!(run.output, words(&expected));
}

/// Cube and cube-array reads by face and layer, and their size queries.
#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn cube_and_cube_array_faces_read_the_authored_layers() {
    let face = |layer: u32| [layer as f32, layer as f32 + 0.5, 0.0, 1.0];
    let faces = |count: u32| -> Vec<f32> { (0..count).flat_map(face).collect() };
    let case = kernel(
        "cube_faces",
        vec![
            texture(
                0,
                TextureType::Cube,
                TextureFormat::Rgba32Float,
                [1, 1, 6],
                floats(&faces(6)),
            ),
            texture(
                1,
                TextureType::CubeArray,
                TextureFormat::Rgba32Float,
                [1, 1, 12],
                floats(&faces(12)),
            ),
        ],
        vec![0xab; 48],
        [1, 1, 1],
        buffer_output(48),
    );
    let run = run(&case, &fixture("texture_ops.ll")).unwrap();
    // Face 3 of the cube; face 4 of the second cube (layer 10); width, cubes, levels.
    let mut expected = face(3).to_vec();
    expected.extend(face(10));
    expected.extend([1.0, 2.0, 1.0, 0.0]);
    assert_eq!(run.output, floats(&expected));
}

/// A mipmapped texture of `texture_type`, `format`, and level 0 extent `dimensions` with
/// `levels` levels, every texel of level `level` and layer `layer` holding `value(level, layer)`
/// (`case::mips`'s layout: level 0 first, each level's layers in order).
fn mipmapped(
    texture_type: TextureType,
    format: TextureFormat,
    dimensions: [u32; 3],
    levels: u32,
    value: impl Fn(u32, u32) -> Vec<u8>,
) -> LiteralTexture {
    let mut bytes = Vec::new();
    for level in 0..levels {
        let extent = level_dimensions(texture_type, dimensions, level);
        let layout = crate::literal::texture_layout(texture_type, extent, 1).unwrap();
        let texels = layout.width * layout.height * layout.depth;
        for layer in 0..layout.array_layers {
            bytes.extend(value(level, layer).repeat(texels as usize));
        }
    }
    LiteralTexture {
        role: ResourceRole::Input,
        mip_levels: levels,
        ..texture(0, texture_type, format, dimensions, bytes)
    }
}

/// The first channel of what `msf_mip_levels`'s kernel `entry` samples through its nearest
/// sampler with a nearest mip filter at each level of `texture` (thread `i` at `level(i)`), on
/// this executor's own context; the kernel writes `stride` `float4`s a thread.
fn nearest_at_each_level(entry: &str, texture: LiteralTexture, stride: usize) -> Vec<f32> {
    let levels = texture.mip_levels;
    let length = 16 * stride * levels as usize;
    let mut case = kernel(
        entry,
        vec![texture],
        Vec::new(),
        [levels, 1, 1],
        ComputeOutput::Buffer {
            binding: 1,
            offset: 0,
            length: length as u64,
        },
    );
    let at: Vec<f32> = (0..levels).map(|level| level as f32).collect();
    case.buffers = BTreeMap::from([(0, floats(&at)), (1, vec![0xab; length])]);
    let run = run(&case, &fixture("msf_mip_levels.ll")).unwrap();
    run.output
        .chunks(16 * stride)
        .map(|thread| f32::from_le_bytes(thread[..4].try_into().unwrap()))
        .collect()
}

/// Every level of a mipmapped texture is the image's own level, uploaded from its own bytes:
/// `msf_mip_levels`'s kernels sample a 2D, a 3D, a cube, and slice 1 of a 2D array texture at
/// each level, nearest, and read the value that level holds (a 2D array's and a cube's in its
/// layer). The `r8_unorm` 2D texture's 1-byte level 2 starts past padding: its 3x3 level 1 ends
/// off the 4-byte alignment a copy's offset needs.
#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn each_level_of_a_mipmapped_texture_holds_its_own_texels() {
    let level_and_layer = |level: u32, layer: u32| level as f32 + 100.0 * layer as f32;
    let r32 = |level, layer| level_and_layer(level, layer).to_le_bytes().to_vec();
    for (entry, texture_type, dimensions, levels, layer, stride) in [
        ("msf_lvl2d", TextureType::D2, [8, 8, 1], 4, 0, 2),
        ("msf_lvl3d", TextureType::D3, [4, 4, 4], 3, 0, 1),
        ("msf_lvlcube", TextureType::Cube, [4, 4, 6], 3, 0, 1),
        ("msf_lvlarray", TextureType::D2Array, [4, 4, 3], 3, 1, 1),
    ] {
        let texture = mipmapped(
            texture_type,
            TextureFormat::R32Float,
            dimensions,
            levels,
            r32,
        );
        let expected: Vec<f32> = (0..levels).map(|l| level_and_layer(l, layer)).collect();
        assert_eq!(
            nearest_at_each_level(entry, texture, stride),
            expected,
            "{entry}"
        );
    }
    let code = |level: u32| 40 * level as u8 + 7;
    let texture = mipmapped(
        TextureType::D2,
        TextureFormat::R8Unorm,
        [6, 6, 1],
        3,
        |l, _| vec![code(l)],
    );
    let codes: Vec<u8> = nearest_at_each_level("msf_lvl2d", texture, 2)
        .into_iter()
        .map(|value| (value * 255.0).round() as u8)
        .collect();
    assert_eq!(codes, (0..3).map(code).collect::<Vec<_>>());
}

/// A depth texture read and a multisampled integer texture read of one sample, which this
/// executor initializes per sample.
#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn depth_and_multisampled_textures_read_their_authored_texels() {
    let samples: Vec<u8> = (0..4i32)
        .flat_map(|sample| [sample, 7 * sample, -sample, 1])
        .flat_map(i32::to_le_bytes)
        .collect();
    let mut multisampled = texture(
        1,
        TextureType::D2Multisample,
        TextureFormat::Rgba32Sint,
        [1, 1, 1],
        samples,
    );
    multisampled.sample_count = 4;
    let case = kernel(
        "depth_ms",
        vec![
            texture(
                0,
                TextureType::D2,
                TextureFormat::Depth32Float,
                [1, 1, 1],
                floats(&[0.375]),
            ),
            multisampled,
        ],
        vec![0xab; 8],
        [1, 1, 1],
        buffer_output(8),
    );
    let run = run(&case, &fixture("texture_ops.ll")).unwrap();
    // Sample 2's `.y`.
    assert_eq!(run.output, floats(&[0.375, 14.0]));
}

/// A depth texture whose texel lies outside `[0, 1]` (2.0, which Metal's `Depth32Float` texel
/// keeps): where the opened device enabled `VK_EXT_depth_range_unrestricted`, the run reads it as
/// authored; elsewhere the executor neither copies it (Vulkan takes no such copied depth,
/// `VUID-vkCmdCopyBufferToImage-pRegions-07931`) nor clamps it, and refuses the case typed
/// before any image is made (`ToolingRequirement::DepthOutsideUnitRange`).
#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn a_depth_outside_the_unit_range_runs_only_where_the_device_takes_it() {
    use crate::check::admission::{ExecutorError, NotSupported};
    use crate::requirement::ToolingRequirement;
    let samples: Vec<u8> = (0..4i32)
        .flat_map(|sample| [sample, 7 * sample, -sample, 1])
        .flat_map(i32::to_le_bytes)
        .collect();
    let mut multisampled = texture(
        1,
        TextureType::D2Multisample,
        TextureFormat::Rgba32Sint,
        [1, 1, 1],
        samples,
    );
    multisampled.sample_count = 4;
    let case = kernel(
        "depth_ms",
        vec![
            texture(
                0,
                TextureType::D2,
                TextureFormat::Depth32Float,
                [1, 1, 1],
                floats(&[2.0]),
            ),
            multisampled,
        ],
        vec![0xab; 8],
        [1, 1, 1],
        buffer_output(8),
    );
    let unrestricted = crate::candidate::platform::opened_worker_device(crate::gpu::TEST_DEVICE)
        .unwrap_or_else(|error| panic!("open the lane's device: {error:?}"))
        .device()
        .profile()
        .enabled()
        .extensions
        .contains(&metal2vulkan::device::VkExtension::ExtDepthRangeUnrestricted);
    let ran = run(&case, &fixture("texture_ops.ll"));
    eprintln!(
        "DEPTH-RANGE-CASE unrestricted {unrestricted}: {}",
        match &ran {
            Ok(run) => format!("ran, output {:02x?}", run.output),
            Err(error) => error.to_string(),
        }
    );
    match (ran, unrestricted) {
        (Ok(run), true) => assert_eq!(run.output, floats(&[2.0, 14.0])),
        (Err(ExecutorError::NotSupported(NotSupported::Requirements(requirements))), false) => {
            assert_eq!(
                requirements,
                [ToolingRequirement::DepthOutsideUnitRange].into()
            )
        }
        (other, unrestricted) => panic!(
            "a depth of 2.0 runs where the device enabled the extension ({unrestricted}) and is \
             refused as depth_outside_unit_range elsewhere: {other:?}"
        ),
    }
}

/// A sample count the device's image format properties do not allow is refused typed before any
/// image is created, never created as invalid usage (`VUID-VkImageCreateInfo-samples-02258`):
/// the CPU ICDs the gate runs allow no 2-sample images (lavapipe allows 1, 4, and 8).
#[cfg(all(m2v_cpu_icd, any(target_os = "linux", target_os = "macos")))]
#[test]
fn a_sample_count_the_device_lacks_is_refused_before_any_image() {
    let samples: Vec<u8> = (0..2i32)
        .flat_map(|sample| [sample, 7 * sample, -sample, 1])
        .flat_map(i32::to_le_bytes)
        .collect();
    let mut multisampled = texture(
        1,
        TextureType::D2Multisample,
        TextureFormat::Rgba32Sint,
        [1, 1, 1],
        samples,
    );
    multisampled.sample_count = 2;
    let case = kernel(
        "depth_ms",
        vec![
            texture(
                0,
                TextureType::D2,
                TextureFormat::Depth32Float,
                [1, 1, 1],
                floats(&[0.375]),
            ),
            multisampled,
        ],
        vec![0xab; 8],
        [1, 1, 1],
        buffer_output(8),
    );
    match run(&case, &fixture("texture_ops.ll")) {
        Err(crate::check::admission::ExecutorError::NotSupported(
            crate::check::admission::NotSupported::NotEnabled(missing),
        )) => assert!(
            missing
                .iter()
                .any(|m| m.contains("texture 1") && m.contains("with 2 sample(s)")),
            "{missing:?}"
        ),
        other => panic!("a 2-sample texture must be refused typed: {other:?}"),
    }
}

/// A runtime sampler bound with a sampled texture, at an explicit level past the texture's one
/// level: nearest, clamped, normalized coordinates select texel `gid`.
#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn a_runtime_sampler_samples_the_bound_texture() {
    let texels: Vec<f32> = (0..4)
        .flat_map(|texel| [texel as f32, 0.25, 0.5, 1.0])
        .collect();
    let mut case = kernel(
        "sample_level",
        vec![texture(
            0,
            TextureType::D2,
            TextureFormat::Rgba32Float,
            [4, 1, 1],
            floats(&texels),
        )],
        vec![0xab; 64],
        [4, 1, 1],
        buffer_output(64),
    );
    case.samplers.insert(
        0,
        SamplerResource {
            binding: 0,
            address_mode: SamplerAddressMode::ClampToEdge,
            min_filter: SamplerFilter::Nearest,
            mag_filter: SamplerFilter::Nearest,
            mip_filter: SamplerMipFilter::NotMipmapped,
            normalized_coordinates: true,
        },
    );
    let run = run(&case, &fixture("sampling.ll")).unwrap();
    assert_eq!(run.output, floats(&texels));
}
