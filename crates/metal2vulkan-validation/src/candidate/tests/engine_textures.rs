//! The engine compute candidate with textures, texture buffers, texture arrays, and samplers,
//! bound or in an argument buffer, and with buffers in an argument buffer, on a device: the owned
//! `texture_ops`, `sampling`, `msf_mip_levels`, `selected_member_texture`,
//! `argument_sampler_sample`, `ob2_loaded_pointer` and `depth_fill_read` kernels and the public
//! `texture_array_width` kernel, planned by the binder and executed on the executor's own context
//! under `VK_LAYER_KHRONOS_validation`, against expectations derived from their MSL; the cases
//! the runtime makes run through it too on Linux ([`runs`]). `G-linux`'s `m2v_cpu_icd` step runs
//! these on lavapipe. Every kernel is loop-free.

use super::engine_compute::run;
use crate::candidate::engine::{resources, ComputeCase, ComputeOutput, ComputeRun};
use crate::case::mips::level_dimensions;
use crate::case::{
    ArgumentBufferSamplerResource, ResourceRole, SamplerAddressMode, SamplerFilter,
    SamplerMipFilter, SamplerResource, TextureFormat, TextureType,
};
use crate::check::admission::ExecutorError;
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
        textures: textures
            .into_iter()
            .map(|texture| (texture.binding, texture))
            .collect(),
        samplers: BTreeMap::new(),
        threadgroup_lengths: BTreeMap::new(),
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

/// A run of a case.
type Run = fn(&ComputeCase, &[u8]) -> Result<ComputeRun, ExecutorError>;

/// The runs a case the runtime makes takes: on the executor's own context, and on Linux through
/// the runtime (`engine::runs_through_runtime`), each named.
#[cfg(target_os = "linux")]
fn runs() -> [(&'static str, Run); 2] {
    [
        ("own context", run),
        ("runtime", super::engine_compute::run_through_runtime),
    ]
}

/// The runs a case the runtime makes takes: on the executor's own context alone, off Linux.
#[cfg(not(target_os = "linux"))]
fn runs() -> [(&'static str, Run); 1] {
    [("own context", run)]
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
    for (name, run) in runs() {
        let run = run(&case, &bitcode).unwrap_or_else(|error| panic!("{name}: {error}"));
        assert_eq!(run.output, words(&[7]), "{name}");
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
    let expected: Vec<u32> = initial
        .iter()
        .zip([1, 2, 3, 4].iter().cycle())
        .map(|(value, add)| (value + add) * 2)
        .collect();
    let bitcode = fixture("texture_ops.ll");
    for (name, run) in runs() {
        let run = run(&case, &bitcode).unwrap_or_else(|error| panic!("{name}: {error}"));
        assert_eq!(run.output, words(&expected), "{name}");
    }
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
/// executor initializes per sample, and so does a run through the runtime, by a draw that writes
/// each sample through an unsigned-integer view (`platform::runtime_multisample`).
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
    let bitcode = fixture("texture_ops.ll");
    for (name, run) in runs() {
        let run = run(&case, &bitcode).unwrap_or_else(|error| panic!("{name}: {error}"));
        // Sample 2's `.y`.
        assert_eq!(run.output, floats(&[0.375, 14.0]), "{name}");
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
    let bitcode = fixture("sampling.ll");
    for (name, run) in runs() {
        let run = run(&case, &bitcode).unwrap_or_else(|error| panic!("{name}: {error}"));
        assert_eq!(run.output, floats(&texels), "{name}");
    }
}

/// Two textures of an argument buffer, one chosen per thread: the owned `selected_member_texture`
/// kernel reads texel `t >> 1` of `second` for an odd `t` and of `first` for an even one, and
/// writes its red channel times the owner's `scale` (2) plus the chosen texture's width (4 or 8).
/// Each texture is the one whose `gpuResourceID` its owner holds at its field.
#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn an_argument_buffer_texture_is_the_texture_at_its_field() {
    let halves = |reds: &[u16]| -> Vec<u8> {
        reds.iter()
            .flat_map(|&red| [red, 0, 0, 0x3c00])
            .flat_map(u16::to_le_bytes)
            .collect()
    };
    let member = |width: u32, reds: &[u16]| LiteralTexture {
        role: ResourceRole::Input,
        ..texture(
            0,
            TextureType::D2,
            TextureFormat::Rgba16Float,
            [width, 1, 1],
            halves(reds),
        )
    };
    let mut case = kernel(
        "selected_member_texture",
        Vec::new(),
        Vec::new(),
        [8, 1, 1],
        ComputeOutput::Buffer {
            binding: 1,
            offset: 0,
            length: 32,
        },
    );
    let mut material = 2.0f32.to_le_bytes().to_vec();
    material.resize(24, 0);
    case.buffers = BTreeMap::from([(0, material), (1, vec![0xab; 32])]);
    case.dispatch = Dispatch::Threads {
        threads: [8, 1, 1],
        threads_per_group: [8, 1, 1],
    };
    // first: red 1, 2, 3, 4; second: red 10 to 17.
    case.argument_buffer_textures = BTreeMap::from([
        ((0, 8), member(4, &[0x3c00, 0x4000, 0x4200, 0x4400])),
        (
            (0, 16),
            member(
                8,
                &[
                    0x4900, 0x4980, 0x4a00, 0x4a80, 0x4b00, 0x4b80, 0x4c00, 0x4c40,
                ],
            ),
        ),
    ]);
    let bitcode = fixture("selected_member_texture.ll");
    for (name, run) in runs() {
        let run = run(&case, &bitcode).unwrap_or_else(|error| panic!("{name}: {error}"));
        let expected = [6.0, 28.0, 8.0, 30.0, 10.0, 32.0, 12.0, 34.0];
        assert_eq!(run.output, floats(&expected), "{name}");
    }
}

/// A sampler of an argument buffer beside the texture it samples: the owned
/// `argument_sampler_sample` kernel samples the member texture at `(0.25g + 0.125, 0.25g +
/// 0.125)` through the member sampler, nearest and normalized, so thread `g` reads texel
/// `(g, g)` of the 4x4 texture, whose red is `x + 4y`: the executor's own context and, on Linux,
/// the runtime, which samples through the element of the sampler arrays the member's resource ID
/// names.
#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn an_argument_buffer_sampler_samples_its_member_texture() {
    let texels: Vec<f32> = (0..16)
        .flat_map(|texel| [texel as f32, 0.0, 0.0, 1.0])
        .collect();
    let mut case = kernel(
        "argument_sampler_sample",
        Vec::new(),
        Vec::new(),
        [4, 1, 1],
        ComputeOutput::Buffer {
            binding: 1,
            offset: 0,
            length: 64,
        },
    );
    case.buffers = BTreeMap::from([(0, vec![0; 16]), (1, vec![0xab; 64])]);
    case.argument_buffer_textures = BTreeMap::from([(
        (0, 0),
        LiteralTexture {
            role: ResourceRole::Input,
            ..texture(
                0,
                TextureType::D2,
                TextureFormat::Rgba32Float,
                [4, 4, 1],
                floats(&texels),
            )
        },
    )]);
    case.argument_buffer_samplers = BTreeMap::from([(
        (0, 8),
        ArgumentBufferSamplerResource {
            buffer_binding: 0,
            field_offset: 8,
            address_mode: SamplerAddressMode::ClampToEdge,
            min_filter: SamplerFilter::Nearest,
            mag_filter: SamplerFilter::Nearest,
            mip_filter: SamplerMipFilter::NotMipmapped,
            normalized_coordinates: true,
            compare_function: None,
        },
    )]);
    let bitcode = fixture("tail_argument_samplers.ll");
    let expected: Vec<f32> = (0..4)
        .flat_map(|g| [5.0 * g as f32, 0.0, 0.0, 1.0])
        .collect();
    for (name, run) in runs() {
        let run = run(&case, &bitcode).unwrap_or_else(|error| panic!("{name}: {error}"));
        assert_eq!(run.output, floats(&expected), "{name}");
    }
}

/// A buffer of an argument buffer, loaded and stored through: the owned `ob2_loaded_pointer`
/// kernel loads word `idx[t]` of the buffer whose address its holder holds into `out[t]`, and
/// stores `0xB0000000 | t` at word `idx[8 + t]` of it. Every index is inside the buffer's 16
/// words: the loads read words 0 to 7, the stores write words 8 to 15. The loads are observed in
/// `out`, the stores in the member buffer itself.
#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn an_argument_buffer_buffer_is_the_buffer_at_its_field() {
    let indices: Vec<u32> = (0..16).collect();
    let pointee: Vec<u32> = (100..116).collect();
    let mut case = kernel(
        "ob2_loaded_pointer",
        Vec::new(),
        Vec::new(),
        [8, 1, 1],
        ComputeOutput::None,
    );
    case.buffers = BTreeMap::from([(0, words(&indices)), (1, vec![0; 8]), (2, vec![0xab; 32])]);
    case.dispatch = Dispatch::Threads {
        threads: [8, 1, 1],
        threads_per_group: [8, 1, 1],
    };
    case.argument_buffer_buffers = BTreeMap::from([((1, 0), words(&pointee))]);
    let bitcode = fixture("ob2_loaded_pointer.ll");
    let stored: Vec<u32> = (100..108).chain((0..8).map(|t| 0xB000_0000 | t)).collect();
    let observations = [
        (
            ComputeOutput::Buffer {
                binding: 2,
                offset: 0,
                length: 32,
            },
            words(&pointee[..8]),
        ),
        (
            ComputeOutput::ArgumentBufferBuffer {
                owner: 1,
                field_offset: 0,
                offset: 0,
                length: 64,
            },
            words(&stored),
        ),
    ];
    for (output, expected) in observations {
        let case = ComputeCase {
            output,
            ..case.clone()
        };
        for (name, run) in runs() {
            let run = run(&case, &bitcode).unwrap_or_else(|error| panic!("{name}: {error}"));
            assert_eq!(run.output, expected, "{name}: {output:?}");
        }
    }
}

/// A 32-bit float depth texture, which the runtime takes no upload of (`Gap::FloatDepthUpload`):
/// through the runtime the run fills it by a draw of the owned `depth_fill` pair, and on both runs
/// `depth_fill_read` reads every depth back as authored, row 1 below row 0. A depth outside
/// `[0, 1]`, which a draw would clamp, is refused through the runtime as the requirement the case
/// checker raises from the case's literals (`ToolingRequirement::DepthOutsideUnitRange`).
#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn a_float_depth_texture_is_filled_by_a_draw_and_reads_back_as_authored() {
    let depths = [
        0.0f32,
        0.25,
        0.5,
        1.0,
        0.123_456_79,
        0.75,
        1.0e-3,
        0.999_999_9,
    ];
    let case = |values: &[f32]| {
        let depth = LiteralTexture {
            role: ResourceRole::Input,
            ..texture(
                0,
                TextureType::D2,
                TextureFormat::Depth32Float,
                [4, 2, 1],
                floats(values),
            )
        };
        kernel(
            "depth_fill_read",
            vec![depth],
            vec![0xab; 32],
            [8, 1, 1],
            buffer_output(32),
        )
    };
    let bitcode = fixture("depth_fill.ll");
    for (name, run) in runs() {
        let ran = run(&case(&depths), &bitcode).unwrap_or_else(|error| panic!("{name}: {error}"));
        assert_eq!(ran.output, floats(&depths), "{name}");
    }
    #[cfg(target_os = "linux")]
    {
        use crate::check::admission::NotSupported;
        use crate::requirement::ToolingRequirement;
        let mut outside = depths;
        outside[5] = 2.0;
        match super::engine_compute::run_through_runtime(&case(&outside), &bitcode) {
            Err(ExecutorError::NotSupported(NotSupported::Requirements(requirements))) => {
                assert_eq!(
                    requirements,
                    [ToolingRequirement::DepthOutsideUnitRange].into()
                )
            }
            other => panic!(
                "a depth of 2.0 through the runtime is refused as depth_outside_unit_range: \
                 {other:?}"
            ),
        }
    }
}
