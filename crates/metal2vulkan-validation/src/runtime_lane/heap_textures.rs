//! Textures in automatic and placement heaps on the device the cfg selects, under the validation
//! layer:
//!
//! - **Answers.** `heap_texture_size_and_align` answers each probe shape the same each time, a
//!   power-of-two alignment and a size that holds the texels; `HEAPTEX` lines print it beside
//!   Metal's own answer (an M4 on macOS 27.0.1 and an M3 Max on 26.7 answer alike). On MoltenVK,
//!   whose image requirement is Metal's, the two are equal, but for a depth or stencil texture with
//!   `allowGPUOptimizedContents` off. A texture with the property off takes the image of one of
//!   `PixelFormatView` usage, on every device the answer of that texture (for a depth or stencil
//!   format, at least that answer: every answer holds Metal's where it is measured, and Metal keeps
//!   such a texture of `PixelFormatView` usage compressed), which MoltenVK makes a
//!   `PixelFormatView` Metal texture where it is a color one: Metal's answer for the texture with
//!   the property off. A depth or stencil texture's metadata no Vulkan request drops there, so the
//!   device's image of it is the property-on texture's; the answer is still Metal's own for the
//!   descriptor as given, the property-off number, where a shadow of the heap's bytes holds the
//!   texture to it (16384 for the single-sample 64x64 `Depth32Float` one, where Metal answers
//!   16512 with the property on).
//! - **Placement.** A texture fits at every multiple of the answered alignment of a placement
//!   heap, its `heap_offset` that offset; an offset off the alignment, bytes past the heap's end,
//!   another storage or CPU cache mode and the other heap type's call are refused `Invalid`, as
//!   Metal's validation stops them. An automatic heap places first fit, its
//!   `used_size` the sum of its textures' answered sizes, and takes a range back when its
//!   texture is made aliasable or goes.
//! - **Aliasing.** A heap's buffer and texture over the same bytes alias them, as Metal's do: a
//!   texture write changes the buffer's bytes, and a write through the buffer
//!   changes every texel the texture reads, ordered by the heap's tracking within one command
//!   buffer. The bytes' layout is each device's own (Metal's is Apple's tiled or compressed
//!   layout), so the test asserts the change, not the bytes.

use super::degrade_notices::counts;
use super::heap_layouts::HEAP_LAYOUTS;
use super::heap_metal_sizes::STACKED_NIL_ALIGNMENT;
use super::host_memory::{gpu_fill, gpu_read, on_device, run};
use super::presenting::Adopted;
use super::textures::{no_texture_alone, pattern, texel_bytes};
use super::{
    bound_buffers, buffer_with, encode_case, layer_reported_nothing_at_all, pipeline_state,
    public_case, read_buffer, run_case, texture_descriptor, validated_device,
    validated_device_with, CaseResources, Messages, PublicCase, ScratchCache,
};
use crate::air_library::AirLibrary;
use crate::candidate::engine::ComputeOutput;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::device::{DriverFact, HeapCompressionPolicy};
use metal2vulkan::error::{Feature, LinearNeed, Needs};
use metal2vulkan::metal::format::{copies_only, footprint, is_depth_stencil, Footprint};
use metal2vulkan::metal::{
    IndirectCommandType, PixelFormat, PrimitiveType, RenderStages, ResourceUsage, TextureType,
    TextureUsage,
};
use metal2vulkan::pipeline::{RenderPipelineDescriptor, StageSource};
use metal2vulkan::vk::{DriverId, NumericFormat};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    BlitOption, Buffer, CommandQueue, CpuCacheMode, Degradation, Device, Error, HazardTrackingMode,
    Heap, HeapDescriptor, HeapProblem, HeapType, IndirectCommandBufferDescriptor, Invalid,
    LoadAction, Nil, Origin, PipelineOptions, Problem, Region, RenderPassDescriptor,
    ResourceOptions, RuntimeGap, Size, SizeAndAlign, StorageMode, StoreAction, Texture,
    TextureDescriptor, TextureProblem,
};
use std::collections::{BTreeMap, BTreeSet};

fn invalid(call: &'static str, problem: Problem) -> Error {
    Error::Invalid(Invalid::new(call, problem))
}

/// What `call` answers where plain Metal answers nil.
fn nil(call: &'static str, problem: Problem) -> Error {
    Error::Nil(Nil::new(call, problem))
}

/// A probe shape and Metal's answer for it, `Private`.
struct Shape {
    name: &'static str,
    texture_type: TextureType,
    pixel_format: PixelFormat,
    extent: [u64; 3],
    array_length: u64,
    levels: u64,
    samples: u64,
    usage: TextureUsage,
    /// `allowGPUOptimizedContents`.
    optimized: bool,
    /// Metal's `(size, align)`.
    metal: (u64, u64),
}

impl Shape {
    fn descriptor(&self, options: ResourceOptions) -> TextureDescriptor {
        let [width, height, depth] = self.extent;
        let mut descriptor = TextureDescriptor::default();
        descriptor.texture_type = self.texture_type;
        descriptor.pixel_format = self.pixel_format;
        (descriptor.width, descriptor.height, descriptor.depth) = (width, height, depth);
        descriptor.mipmap_level_count = self.levels;
        descriptor.sample_count = self.samples;
        descriptor.array_length = self.array_length;
        descriptor.usage = self.usage;
        descriptor.resource_options = options;
        descriptor.allow_gpu_optimized_contents = self.optimized;
        descriptor
    }

    /// The bytes level 0 of one slice holds at least.
    fn texel_bytes(&self) -> u64 {
        let texel = match self.pixel_format {
            PixelFormat::R8Unorm => 1,
            PixelFormat::RG8Unorm => 2,
            _ => 4,
        };
        let [width, height, depth] = self.extent;
        width * height * depth * texel * self.samples
    }
}

const READ: TextureUsage = TextureUsage::SHADER_READ;

/// Metal's answers for the probe shapes (the M4 and the M3 Max alike).
fn shapes() -> Vec<Shape> {
    let plain = |name, pixel_format, side: u64, levels, usage, optimized, metal| Shape {
        name,
        texture_type: TextureType::Type2D,
        pixel_format,
        extent: [side, side, 1],
        array_length: 1,
        levels,
        samples: 1,
        usage,
        optimized,
        metal,
    };
    let write = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
    let target = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
    let rgba = PixelFormat::RGBA8Unorm;
    vec![
        plain("rgba8-16", rgba, 16, 1, READ, true, (1152, 128)),
        plain("rgba8-16-write", rgba, 16, 1, write, true, (1024, 128)),
        plain("rgba8-16-target", rgba, 16, 1, target, true, (1152, 128)),
        plain(
            "rgba8-16-unoptimized",
            rgba,
            16,
            1,
            READ,
            false,
            (1024, 128),
        ),
        plain("rgba8-17", rgba, 17, 1, READ, true, (4224, 128)),
        plain("rgba8-64", rgba, 64, 1, READ, true, (16512, 128)),
        plain("rgba8-64-write", rgba, 64, 1, write, true, (16384, 128)),
        plain("rgba8-100", rgba, 100, 1, READ, true, (66048, 128)),
        plain("rgba8-256-mips", rgba, 256, 9, READ, true, (352768, 128)),
        plain(
            "r8-64",
            PixelFormat::R8Unorm,
            64,
            1,
            READ,
            true,
            (4224, 128),
        ),
        plain(
            "rg8-64",
            PixelFormat::RG8Unorm,
            64,
            1,
            READ,
            true,
            (8320, 128),
        ),
        Shape {
            name: "rgba8-1d-128",
            texture_type: TextureType::Type1D,
            extent: [128, 1, 1],
            ..plain("", rgba, 1, 1, READ, true, (512, 128))
        },
        Shape {
            name: "rgba8-array3-33",
            texture_type: TextureType::Type2DArray,
            extent: [33, 33, 1],
            array_length: 3,
            ..plain("", rgba, 1, 1, READ, true, (49536, 16384))
        },
        Shape {
            name: "rgba8-cube-33",
            texture_type: TextureType::Cube,
            extent: [33, 33, 1],
            ..plain("", rgba, 1, 1, READ, true, (99072, 16384))
        },
        Shape {
            name: "rgba8-3d-33",
            texture_type: TextureType::Type3D,
            extent: [33, 33, 33],
            ..plain("", rgba, 1, 1, READ, true, (544896, 16384))
        },
        Shape {
            name: "rgba8-ms4-33",
            texture_type: TextureType::Type2DMultisample,
            extent: [33, 33, 1],
            samples: 4,
            usage: TextureUsage::RENDER_TARGET,
            ..plain("", rgba, 1, 1, READ, true, (66048, 128))
        },
        // The property off (an M3 Max and an M4 alike).
        plain("rgba8-256", rgba, 256, 1, READ, true, (264192, 128)),
        plain(
            "rgba8-256-unoptimized",
            rgba,
            256,
            1,
            READ,
            false,
            (262144, 128),
        ),
        Shape {
            name: "bgra8-512x128-target",
            extent: [512, 128, 1],
            ..plain(
                "",
                PixelFormat::BGRA8Unorm,
                1,
                1,
                target,
                true,
                (264192, 128),
            )
        },
        Shape {
            name: "bgra8-512x128-target-unoptimized",
            extent: [512, 128, 1],
            ..plain(
                "",
                PixelFormat::BGRA8Unorm,
                1,
                1,
                target,
                false,
                (262144, 128),
            )
        },
        Shape {
            name: "rgba8-ms4-256",
            texture_type: TextureType::Type2DMultisample,
            samples: 4,
            usage: TextureUsage::RENDER_TARGET,
            ..plain("", rgba, 256, 1, READ, true, (1056768, 128))
        },
        Shape {
            name: "rgba8-ms4-256-unoptimized",
            texture_type: TextureType::Type2DMultisample,
            samples: 4,
            usage: TextureUsage::RENDER_TARGET,
            ..plain("", rgba, 256, 1, READ, false, (1048576, 128))
        },
        plain(
            "depth32-64-target",
            PixelFormat::Depth32Float,
            64,
            1,
            target,
            true,
            (16512, 128),
        ),
        plain(
            "depth32-64-target-unoptimized",
            PixelFormat::Depth32Float,
            64,
            1,
            target,
            false,
            (16384, 128),
        ),
    ]
}

/// A `side`-square `RGBA8Unorm` texture's descriptor, read by shaders, with `options`.
fn rgba8(side: u64, options: ResourceOptions) -> TextureDescriptor {
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, side, side, false);
    descriptor.resource_options = options;
    descriptor
}

/// A heap of `heap_type`, `size` bytes, and `options`.
fn heap(device: &Device, heap_type: HeapType, size: u64, options: ResourceOptions) -> Heap {
    device
        .new_heap(&HeapDescriptor::new(heap_type, size, options))
        .unwrap()
}

/// The kinds a heap texture's size answer announces where it is not Metal's: more bytes or a
/// coarser alignment than Metal's modelled answer, which its device's image needs; a depth or
/// stencil texture whose driver keeps it GPU-optimized and no shadow of Metal's tiles serves; a
/// shape whose Metal answer the runtime does not model.
/// The kind a heap texture Metal tiles announces where no image bound in the heap's memory lays
/// out Metal's tiles and a shadow of them serves (`AliasRoute::Tiled`, `texture/heap.rs`): its
/// bytes there are Metal's tiles, so it is counted for no layout.
const TILE_SHADOW: [Degradation; 1] = [Degradation::ShadowedTexture];

const SIZE_NOTICES: [Degradation; 3] = [
    Degradation::HeapTextureLargerThanMetal,
    Degradation::OptimizedDepthStencilTexture,
    Degradation::UnmeasuredHeapTextureSize,
];

/// The answers for each probe shape, printed beside Metal's, each answered alike twice, a
/// power-of-two alignment and a size holding the texels, with `allowGPUOptimizedContents` off the
/// answer for the shape of `PixelFormatView` usage too: Metal's own on MoltenVK (a depth texture's
/// a shadow of Metal's tiles there, `heap_depth_textures_keep_their_texels_in_metal_s_tiles`). On
/// every device
/// each answer is Metal's, announcing nothing, or more bytes or a coarser alignment than Metal's
/// with one of [`SIZE_NOTICES`] announced for it (the device announces each occurrence). Then the
/// placement and automatic heaps' rules.
#[test]
fn heap_textures_answer_and_place_as_metal_s_do() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
            options.announce_each = SIZE_NOTICES.into();
        })
        .unwrap();
        let device = &device;
        let announced =
            || -> BTreeSet<Degradation> { counts(&messages.take_notices()).into_keys().collect() };
        let moltenvk = device.properties().description.driver_id == Some(DriverId::MOLTENVK);
        let mut differ = Vec::new();
        let shapes = shapes();
        for shape in &shapes {
            let descriptor = shape.descriptor(ResourceOptions::private());
            let answer = match device.heap_texture_size_and_align(&descriptor) {
                Ok(answer) => answer,
                Err(error @ (Error::Unsupported(_) | Error::NotImplemented(_))) => {
                    eprintln!("HEAPTEX {}: refused {error:?}", shape.name);
                    let kinds = announced();
                    assert!(
                        kinds.is_empty(),
                        "{}: refused, announced {kinds:?}",
                        shape.name
                    );
                    continue;
                }
                Err(error) => panic!("{}: {error:?}", shape.name),
            };
            let kinds = announced();
            let again = device.heap_texture_size_and_align(&descriptor).unwrap();
            assert_eq!(answer, again, "{}: one answer", shape.name);
            assert_eq!(announced(), kinds, "{}: one answer's notices", shape.name);
            if !shape.optimized {
                let mut viewed = descriptor;
                viewed.allow_gpu_optimized_contents = true;
                viewed.usage |= TextureUsage::PIXEL_FORMAT_VIEW;
                let viewed = device.heap_texture_size_and_align(&viewed).unwrap();
                if is_depth_stencil(shape.pixel_format) {
                    assert!(
                        viewed.size >= answer.size && viewed.align >= answer.align,
                        "{}: a PixelFormatView texture's {viewed:?} holds {answer:?}",
                        shape.name
                    );
                } else {
                    assert_eq!(
                        answer, viewed,
                        "{}: a PixelFormatView texture's",
                        shape.name
                    );
                }
                // Its image is made, in a heap of the answered size and on the device.
                let options = ResourceOptions::private();
                let placement = heap(device, HeapType::Placement, answer.size, options);
                let placed = placement.new_texture_at_offset(&descriptor, 0).unwrap();
                let made = device.new_texture(&descriptor).unwrap();
                for texture in [&placed, &made] {
                    assert!(!texture.allow_gpu_optimized_contents(), "{}", shape.name);
                    assert_eq!(texture.usage(), shape.usage, "{}", shape.name);
                }
                // The view's answer announces only a size's kinds, and the textures' makings
                // the heap texture's route where the device's driver facts force one: an image
                // the driver may compress one of `ALIASES`; a depth or stencil texture's image
                // the driver lays apart from its memory a shadow of the heap's bytes
                // (`shadowed-texture`, as the heap depth lane measures on MoltenVK) or, where no
                // shadow serves it, `heap-depth-stencil-apart`; and one of [`HEAP_LAYOUTS`] for a
                // texture whose bytes in the heap the driver is not measured to lay out as Metal's
                // (a multisample texture, a depth or stencil one, any on a driver without
                // `DriverFact::ColorImagesLieAsMetalHeapTextures`), or for such a texture no image
                // bound in the heap's memory serves, a shadow of Metal's tiles there
                // (`shadowed-texture`, `texture/heap_tiled.rs`; the NVIDIA RTX 5080 Laptop's). A
                // 32-bit float depth texture whose image a shadow fills from the heap's bytes, on
                // a device that may store a depth outside `[0, 1]` copied there clamped (no
                // `VK_EXT_depth_range_unrestricted`, not measured to keep one: ANV),
                // `restricted-depth-upload`.
                let facts = &device.properties().driver_facts;
                let apart = is_depth_stencil(shape.pixel_format)
                    && facts.contains(&DriverFact::DepthStencilImagesLieApart);
                let clamps = !device.enables_extension(ash::ext::depth_range_unrestricted::NAME)
                    && !facts.contains(&DriverFact::DepthImageKeepsValuesOutsideUnitRange);
                let float_depth = matches!(
                    shape.pixel_format,
                    PixelFormat::Depth32Float | PixelFormat::Depth32FloatStencil8
                );
                let upload = [Degradation::RestrictedDepthUpload];
                let as_metal = shape.samples == 1
                    && !is_depth_stencil(shape.pixel_format)
                    && facts.contains(&DriverFact::ColorImagesLieAsMetalHeapTextures);
                let routes = facts
                    .contains(&DriverFact::CompressedImagesHideAliasedWrites)
                    .then_some(&ALIASES[..])
                    .into_iter()
                    .chain(apart.then_some(&DEPTH_STENCIL_ROUTES[..]))
                    .chain((!as_metal).then_some(&HEAP_LAYOUTS[..]))
                    .chain((!as_metal).then_some(&TILE_SHADOW[..]))
                    .chain((clamps && float_depth).then_some(&upload[..]))
                    .flatten();
                let allowed: BTreeSet<Degradation> =
                    SIZE_NOTICES.iter().chain(routes).copied().collect();
                let more = announced();
                assert!(
                    more.is_subset(&allowed),
                    "{}: announced {more:?}, beyond {allowed:?}",
                    shape.name
                );
            }
            assert!(answer.align.is_power_of_two(), "{}", shape.name);
            assert!(answer.size >= shape.texel_bytes(), "{}", shape.name);
            let SizeAndAlign { size, align, .. } = answer;
            eprintln!(
                "HEAPTEX {}: size {size} align {align}; Metal's size {} align {}",
                shape.name, shape.metal.0, shape.metal.1
            );
            // MoltenVK makes a `PixelFormatView` Metal texture of a color texture's image with
            // `allowGPUOptimizedContents` off, laid out as Metal lays out the texture. A depth or
            // stencil texture's it makes optimized (no Vulkan request reaches the property), so
            // such a texture is a shadow of Metal's tiles at Metal's answer.
            if (size, align) != shape.metal {
                differ.push(shape.name);
            }
            // Metal's answer for the shape announces nothing; any other is past it, with its
            // reason announced.
            if (size, align) == shape.metal {
                assert!(
                    kinds.is_empty(),
                    "{}: Metal's answer, announced {kinds:?}",
                    shape.name
                );
            } else {
                assert!(
                    size >= shape.metal.0 && align >= shape.metal.1,
                    "{}: size {size} align {align} below Metal's {:?}",
                    shape.name,
                    shape.metal
                );
                assert!(
                    kinds.len() == 1 && SIZE_NOTICES.iter().any(|kind| kinds.contains(kind)),
                    "{}: size {size} align {align} past Metal's {:?}, announced {kinds:?}",
                    shape.name,
                    shape.metal
                );
                eprintln!(
                    "HEAPTEX-PAST {}: size {size} align {align}, Metal's {:?}: {kinds:?}",
                    shape.name, shape.metal
                );
            }
        }
        if moltenvk {
            assert!(
                differ.is_empty(),
                "MoltenVK answers otherwise than Metal: {differ:?}"
            );
        }
        placement_rules(device);
        automatic_rules(device);
    }
    layer_reported_nothing_at_all(&messages);
}

/// A placement heap's textures: at every multiple of the alignment, refused off it and past the
/// end (Metal's nil), with other options (Metal's stop), and through the
/// automatic heap's call.
fn placement_rules(device: &Device) {
    const CALL: &str = "new_texture_at_offset";
    let options = ResourceOptions::private();
    let heap = heap(device, HeapType::Placement, 1 << 20, options);
    let descriptor = rgba8(64, options);
    let SizeAndAlign { size, align, .. } = device.heap_texture_size_and_align(&descriptor).unwrap();
    let last = (heap.size() - size) / align * align;
    let mut placed = Vec::new();
    for offset in [0, align, 3 * align, last] {
        let texture = heap.new_texture_at_offset(&descriptor, offset).unwrap();
        assert_eq!(texture.heap_offset(), offset);
        assert_eq!(texture.heap(), Some(&heap));
        assert_eq!(texture.storage_mode(), StorageMode::Private);
        placed.push(texture);
    }
    assert_eq!(heap.used_size(), 0, "a placement heap's used size is 0");
    if align > 1 {
        let offset = align / 2;
        assert_eq!(
            heap.new_texture_at_offset(&descriptor, offset).err(),
            Some(nil(CALL, Problem::Heap(HeapProblem::Offset { offset })))
        );
    }
    let past = last + align;
    assert_eq!(
        heap.new_texture_at_offset(&descriptor, past).err(),
        Some(nil(
            CALL,
            Problem::Heap(HeapProblem::PastEnd {
                offset: past,
                length: size,
                heap_size: heap.size(),
            })
        ))
    );
    let shared = rgba8(64, ResourceOptions::shared());
    assert_eq!(
        heap.new_texture_at_offset(&shared, 0).err(),
        Some(invalid(
            CALL,
            Problem::Heap(HeapProblem::StorageMode {
                heap: StorageMode::Private,
                resource: StorageMode::Shared,
            })
        ))
    );
    let combined = rgba8(
        64,
        ResourceOptions::new(
            StorageMode::Private,
            CpuCacheMode::WriteCombined,
            HazardTrackingMode::Default,
        ),
    );
    assert_eq!(
        heap.new_texture_at_offset(&combined, 0).err(),
        Some(invalid(
            CALL,
            Problem::Heap(HeapProblem::CpuCacheMode {
                heap: CpuCacheMode::DefaultCache,
                resource: CpuCacheMode::WriteCombined,
            })
        ))
    );
    assert_eq!(
        heap.new_texture(&descriptor).err(),
        Some(invalid(
            "new_texture",
            Problem::Heap(HeapProblem::HeapType {
                heap_type: HeapType::Placement,
            })
        ))
    );
    // A placement heap's textures are aliasable from the start, as Metal's, and `make_aliasable`
    // changes nothing.
    assert!(placed[0].is_aliasable() && placed[1].is_aliasable());
    assert_eq!(placed[0].make_aliasable(), Ok(()));
    assert!(placed[0].is_aliasable() && placed[1].is_aliasable());
}

/// An automatic heap's textures: first fit at multiples of the alignment, `heap_offset` 0 as
/// Metal answers, `used_size` the sum of the answered sizes, and a range back when a texture is
/// made aliasable or goes.
fn automatic_rules(device: &Device) {
    let options = ResourceOptions::private();
    let heap = heap(device, HeapType::Automatic, 1 << 20, options);
    let small = rgba8(16, options);
    let large = rgba8(64, options);
    let small_size = device.heap_texture_size_and_align(&small).unwrap().size;
    let SizeAndAlign {
        size: large_size,
        align: large_align,
        ..
    } = device.heap_texture_size_and_align(&large).unwrap();
    let first = heap.new_texture(&small).unwrap();
    let second = heap.new_texture(&large).unwrap();
    assert_eq!(first.heap_offset(), 0, "an automatic heap's offset is 0");
    assert_eq!(second.heap_offset(), 0);
    assert_eq!(heap.used_size(), small_size + large_size);
    second.make_aliasable().unwrap();
    assert!(second.is_aliasable());
    assert_eq!(heap.used_size(), small_size);
    drop(first);
    assert_eq!(heap.used_size(), 0);
    assert_eq!(
        heap.new_texture_at_offset(&small, 0).err(),
        Some(invalid(
            "new_texture_at_offset",
            Problem::Heap(HeapProblem::HeapType {
                heap_type: HeapType::Automatic,
            })
        ))
    );
    // Outside a heap, the call changes nothing, as plain Metal.
    let not_in_heap = device.new_texture(&small).unwrap();
    assert_eq!(not_in_heap.make_aliasable(), Ok(()));
    assert!(!not_in_heap.is_aliasable());
    // A heap full of textures has no room for another: first fit at multiples of the alignment
    // puts one every `stride` bytes, the last one's bytes inside the heap (Metal's rule over the
    // device's answered size and alignment, which may leave more than a texture's bytes free).
    let mut filled = Vec::new();
    loop {
        match heap.new_texture(&large) {
            Ok(texture) => filled.push(texture),
            Err(Error::OutOfMemory(_)) => break,
            Err(other) => panic!("filling an automatic heap: {other:?}"),
        }
    }
    assert_eq!(heap.used_size(), filled.len() as u64 * large_size);
    let stride = large_size.next_multiple_of(large_align);
    assert_eq!(filled.len() as u64, (heap.size() - large_size) / stride + 1);
}

/// A `side`-square `RGBA8Unorm` texture's texels `(x, y, 0xAB, 0xCD)`, rows packed.
fn texel_pattern(side: u64) -> Vec<u8> {
    (0..side * side)
        .flat_map(|i| [(i % side) as u8, (i / side) as u8, 0xAB, 0xCD])
        .collect()
}

/// A Shared buffer holding `bytes`.
fn staged(device: &Device, bytes: &[u8]) -> Buffer {
    let buffer = device
        .new_buffer(bytes.len() as u64, ResourceOptions::shared())
        .unwrap();
    let contents = buffer.contents().unwrap();
    // SAFETY: the buffer holds `bytes.len()` bytes, and no command uses them yet.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr(), bytes.len()) };
    buffer
}

/// The texels of a `side`-square texture, read back through a blit.
fn texels(queue: &CommandQueue, texture: &Texture, side: u64) -> Vec<u8> {
    let device = queue.device();
    let bytes = side * side * 4;
    let copy = device.new_buffer(bytes, ResourceOptions::shared()).unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_texture_to_buffer(
        texture,
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(side, side, 1),
        &copy,
        0,
        side * 4,
        bytes,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    run(&command_buffer);
    // SAFETY: the buffer holds `bytes` bytes the completed copy wrote.
    unsafe { std::slice::from_raw_parts(copy.contents().unwrap().as_ptr(), bytes as usize) }
        .to_vec()
}

/// The aliasing of a placement heap's texture of `descriptor` (`RGBA8Unorm`, square) at offset 0
/// and a buffer of its answered size over the same bytes, `what` naming it: texels
/// blitted into the texture's first level read back unchanged and change some of the buffer's
/// bytes, which this asserts; then bytes copied into the buffer, and the first level's texels read
/// back, in one command buffer ordered by the heap's tracking. Gives how many of those texels
/// the buffer's write left unchanged; `None`, printed, where the heap refuses the texture for its
/// tiling (`tiling_refusal`).
fn alias_round_trip(
    queue: &CommandQueue,
    heap: &Heap,
    descriptor: &TextureDescriptor,
    what: &str,
) -> Option<usize> {
    let device = queue.device();
    let (side, options) = (descriptor.width, descriptor.resource_options);
    let texture = match heap.new_texture_at_offset(descriptor, 0) {
        Ok(texture) => texture,
        Err(error) if tiling_refusal(&error).is_some() => {
            eprintln!("HEAPTEX alias {what}: refused {error:?}");
            return None;
        }
        Err(error) => panic!("{what}: {error:?}"),
    };
    let size = device.heap_texture_size_and_align(descriptor).unwrap().size;
    let buffer = heap.new_buffer_at_offset(size, options, 0).unwrap();
    gpu_fill_zero(queue, &buffer, size);
    // Texels into the texture: the buffer's bytes change.
    let pattern = texel_pattern(side);
    let source = staged(device, &pattern);
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_buffer_to_texture(
        &source,
        0,
        side * 4,
        side * side * 4,
        Size::new(side, side, 1),
        &texture,
        0,
        0,
        Origin::new(0, 0, 0),
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    run(&command_buffer);
    assert_eq!(texels(queue, &texture, side), pattern, "{what}");
    let bytes = gpu_read(queue, &buffer, 0..size);
    let changed = bytes.iter().filter(|&&byte| byte != 0).count();
    assert!(
        changed > 0,
        "{what}: the texture's write left the buffer zero"
    );
    // Bytes through the buffer, then the texture's texels, in one command buffer.
    let fill: Vec<u8> = (0..size as usize)
        .map(|i| (0x11 * (i % 4 + 1) + i / 4 % 7) as u8)
        .collect();
    let fill = staged(device, &fill);
    let read = device
        .new_buffer(side * side * 4, ResourceOptions::shared())
        .unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_buffer_to_buffer(&fill, 0, &buffer, 0, size)
        .unwrap();
    blit.end_encoding().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_texture_to_buffer(
        &texture,
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(side, side, 1),
        &read,
        0,
        side * 4,
        side * side * 4,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    run(&command_buffer);
    let after = gpu_read(queue, &read, 0..side * side * 4);
    let unchanged = after
        .as_chunks::<4>()
        .0
        .iter()
        .zip(pattern.as_chunks::<4>().0.iter())
        .filter(|(after, before)| after == before)
        .count();
    eprintln!(
        "HEAPTEX alias {what}: size {size}, {changed} buffer bytes changed by the texture's \
         write, {unchanged} texels unchanged by the buffer's"
    );
    Some(unchanged)
}

/// A heap's buffer and texture over the same bytes alias them, as Metal's do: in
/// a tracked `Shared` and a tracked `Private` placement heap, for `RGBA8Unorm` textures of 16,
/// 64 and 1024 texels a side of storage, render-target and sampled-only usage, texels blitted into
/// the texture change the buffer's bytes and read back unchanged; bytes copied into the buffer then
/// change every texel the texture reads, the copy and the read encoded in one command buffer and
/// ordered by the heap's tracking.
///
/// Where the device's driver may keep a compressed image's texels apart from its memory
/// (`DriverFact::CompressedImagesHideAliasedWrites`: ANV, whose texture's write changed 234 of
/// the buffer's 1024 bytes and the buffer's write none of its 256 texels), the texture takes
/// host-transfer usage or a linear image: the device announces one of `heap-texture-host-transfer`
/// and `heap-texture-linear` once, never `heap-texture-hides-aliased-writes`, whose texels would
/// fail the test. Elsewhere none of the three is announced. `HEAPTEX alias` lines print the counts.
#[test]
fn a_buffer_write_shows_in_an_overlapping_texture() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    // Whether the driver hides such writes, and whether a texture was made in a heap.
    let (hides, made) = {
        let device = &validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let hides = device
            .properties()
            .driver_facts
            .contains(&DriverFact::CompressedImagesHideAliasedWrites);
        let mut made = false;
        let queue = device.new_command_queue().unwrap();
        for storage in [StorageMode::Shared, StorageMode::Private] {
            let options = ResourceOptions::new(
                storage,
                CpuCacheMode::DefaultCache,
                HazardTrackingMode::Tracked,
            );
            // Storage usage often keeps a driver's image uncompressed, so render-target and
            // sampled-only usage, and a side past the small sizes a compressor may skip, are
            // measured too.
            let heap = heap(device, HeapType::Placement, 1 << 23, options);
            for side in [16, 64, 1024] {
                for usage in [
                    TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE,
                    TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET,
                    TextureUsage::SHADER_READ,
                ] {
                    let mut descriptor = rgba8(side, options);
                    descriptor.usage = usage;
                    let what = format!("{storage:?} {side}x{side} {usage:?}");
                    let Some(unchanged) = alias_round_trip(&queue, &heap, &descriptor, &what)
                    else {
                        continue;
                    };
                    made = true;
                    assert_eq!(unchanged, 0, "{what}: texels the buffer's write left alone");
                }
            }
        }
        (hides, made)
    };
    let notices = counts(&messages.take_notices());
    let [transfer, linear, hidden] = ALIASES.map(|kind| notices.get(&kind).copied().unwrap_or(0));
    eprintln!(
        "HEAPTEX alias: {transfer} heap-texture-host-transfer, {linear} heap-texture-linear, \
         {hidden} heap-texture-hides-aliased-writes notices"
    );
    assert_eq!(hidden, 0);
    assert_eq!(transfer + linear, usize::from(hides && made));
    layer_reported_nothing_at_all(&messages);
}

/// Under the default `HeapCompressionPolicy::Compressed`, where the device's driver may keep a
/// compressed image's texels apart from its memory
/// (`DriverFact::CompressedImagesHideAliasedWrites`: ANV), a heap's `RGBA8Unorm` texture that Metal
/// keeps losslessly compressed (render-target and sampled-only usage, 64 and 1024 texels a side)
/// takes the driver's compressed image where that image fits Metal's answer, announced once
/// (`heap-texture-compressed`); its texels the buffer's write may leave alone, as Metal's
/// compressed texture shows such a write otherwise than as written: printed (`HEAPTEX compressed`
/// lines), not asserted. Where the compressed image does not fit (ANV on an Iris Xe aligns it to
/// 64 KiB, where Metal answers 128), the texture lies in place as an image that fits
/// (`WorkCounts::heap_textures_refit`: host-transfer usage or a linear image). That one, and one
/// Metal keeps uncompressed (storage usage, or 8 texels a side), which takes the uncompressed image
/// it takes under `HeapCompressionPolicy::Uncompressed`, show the buffer's write in every texel
/// they read ([`alias_round_trip`]), announced `heap-texture-host-transfer` or
/// `heap-texture-linear`, each once. Elsewhere no route is announced.
#[test]
fn a_heap_texture_metal_compresses_keeps_the_driver_s_compression_by_default() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    let (hides, compressed_made, uncompressed_made, refit) = {
        let device = &validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
            options.policy.heap_compression = HeapCompressionPolicy::Compressed;
        })
        .unwrap();
        let hides = device
            .properties()
            .driver_facts
            .contains(&DriverFact::CompressedImagesHideAliasedWrites);
        let (mut compressed_made, mut uncompressed_made, mut refit) = (false, false, false);
        let queue = device.new_command_queue().unwrap();
        for storage in [StorageMode::Shared, StorageMode::Private] {
            let options = tracked(storage);
            let heap = heap(device, HeapType::Placement, 1 << 23, options);
            for (side, usage, compressed) in [
                (
                    64,
                    TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET,
                    true,
                ),
                (1024, TextureUsage::SHADER_READ, true),
                (
                    64,
                    TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE,
                    false,
                ),
                (
                    8,
                    TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET,
                    false,
                ),
            ] {
                let mut descriptor = rgba8(side, options);
                descriptor.usage = usage;
                let what = format!("{storage:?} {side}x{side} {usage:?}");
                let before = device.work_counts().heap_textures_refit;
                let Some(unchanged) = alias_round_trip(&queue, &heap, &descriptor, &what) else {
                    continue;
                };
                let fitted = device.work_counts().heap_textures_refit > before;
                refit |= fitted;
                if compressed && !fitted {
                    compressed_made = true;
                    eprintln!("HEAPTEX compressed {what}: {unchanged} texels unchanged");
                } else {
                    uncompressed_made |= !compressed;
                    eprintln!("HEAPTEX uncompressed {what}: in place of a shadow {fitted}");
                    assert_eq!(unchanged, 0, "{what}: texels the buffer's write left alone");
                }
            }
        }
        (hides, compressed_made, uncompressed_made, refit)
    };
    let notices = counts(&messages.take_notices());
    let count = |kind: Degradation| notices.get(&kind).copied().unwrap_or(0);
    let [transfer, linear, hidden] = ALIASES.map(count);
    let [compressed] = COMPRESSED.map(count);
    eprintln!(
        "HEAPTEX default: {compressed} heap-texture-compressed, {transfer} \
         heap-texture-host-transfer, {linear} heap-texture-linear, {hidden} \
         heap-texture-hides-aliased-writes notices"
    );
    assert_eq!(compressed, usize::from(hides && compressed_made));
    assert_eq!(hidden, 0);
    assert!(transfer <= 1 && linear <= 1, "a route announced twice");
    let routed = (hides && uncompressed_made) || refit;
    assert_eq!(transfer + linear > 0, routed);
    layer_reported_nothing_at_all(&messages);
}

/// What a 2048x2048 `RGBA8Unorm` render target in a `Private` placement heap costs the GPU under
/// each `HeapCompressionPolicy`, without the validation layer: a pass that clears it and stores,
/// and a blit copying it into another such texture, each its own command buffer, medians and the
/// fastest of 40 after 8 untimed. On a driver whose compressed images hide aliased writes (ANV)
/// the two policies give the texture its compressed and its uncompressed image; elsewhere the
/// same image. Ignored by default: a measurement, not a check.
#[test]
#[ignore = "a measurement, not a check: run by name with --run-ignored all"]
fn heap_compression_costs() {
    const SIDE: u64 = 2048;
    for policy in [
        HeapCompressionPolicy::Compressed,
        HeapCompressionPolicy::Uncompressed,
    ] {
        let messages = Messages::default();
        let cache = ScratchCache::new().unwrap();
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
            options.layers.clear();
            options.policy.heap_compression = policy;
        })
        .unwrap();
        let options = ResourceOptions::private();
        let mut descriptor = rgba8(SIDE, options);
        descriptor.usage = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
        let size = device
            .heap_texture_size_and_align(&descriptor)
            .unwrap()
            .size;
        let heap = heap(&device, HeapType::Placement, 2 * size, options);
        let target = heap.new_texture_at_offset(&descriptor, 0).unwrap();
        let copy = heap.new_texture_at_offset(&descriptor, size).unwrap();
        let notices = counts(&messages.take_notices());
        let queue = device.new_command_queue().unwrap();
        let time = |encode: &dyn Fn(&metal2vulkan_runtime::CommandBuffer)| {
            let mut gpu = Vec::new();
            for run in 0..48 {
                let command_buffer = queue.command_buffer().unwrap();
                encode(&command_buffer);
                command_buffer.commit().unwrap();
                command_buffer
                    .wait_until_completed_for(super::WAIT)
                    .unwrap();
                assert_eq!(command_buffer.error(), None);
                let start = command_buffer.gpu_start_time().unwrap();
                let end = command_buffer.gpu_end_time().unwrap();
                if run >= 8 {
                    gpu.push((end - start) * 1e6);
                }
            }
            gpu.sort_by(f64::total_cmp);
            (gpu[gpu.len() / 2], gpu[0])
        };
        let clear = time(&|command_buffer| {
            let mut pass = RenderPassDescriptor::new();
            let color = &mut pass.color_attachments[0];
            color.texture = Some(target.clone());
            color.load_action = LoadAction::Clear;
            color.store_action = StoreAction::Store;
            let encoder = command_buffer.render_command_encoder(&pass).unwrap();
            encoder.end_encoding().unwrap();
        });
        let blit = time(&|command_buffer| {
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            blit.copy_texture(&target, &copy).unwrap();
            blit.end_encoding().unwrap();
        });
        eprintln!(
            "HEAP-COMPRESSION on {} ({policy:?}, notices {notices:?}): clear median {:.1} us, \
             fastest {:.1} us; copy median {:.1} us, fastest {:.1} us",
            device.name(),
            clear.0,
            clear.1,
            blit.0,
            blit.1
        );
    }
}

/// An adopted device whose consumer enabled `hostImageCopy` and says so
/// (`RawDevice::host_image_copy`) gives its heaps' colour textures Metal's aliasing
/// ([`alias_round_trip`]), a mipmapped one included: on ANV, whose compressed images hide writes
/// through an alias, their images take host-transfer usage, as on a device the runtime creates. The
/// same device adopted without saying so takes a linear image where one serves, else the optimal
/// image, the recorded near-equivalent (`heap-texture-hides-aliased-writes`), whose texels the
/// buffer's write may leave alone: printed (`HEAPTEX adopted` lines), not asserted. An adopted
/// device announces on stderr, so the routes are not counted here.
#[test]
fn an_adopted_device_stating_host_image_copy_aliases_its_heap_textures() {
    let cache = ScratchCache::new().unwrap();
    for states in [true, false] {
        let adopted = if states {
            Adopted::open_host_copying()
        } else {
            Adopted::open(false)
        }
        .unwrap();
        let copies = adopted.copies_host_images();
        {
            let device = adopted.runtime(cache.path());
            let hides = device
                .properties()
                .driver_facts
                .contains(&DriverFact::CompressedImagesHideAliasedWrites);
            if states && hides {
                assert!(
                    copies,
                    "{}: compressed images hide aliased writes, and the device has no \
                     hostImageCopy",
                    device.name()
                );
            }
            let queue = device.new_command_queue().unwrap();
            for storage in [StorageMode::Shared, StorageMode::Private] {
                let options = tracked(storage);
                let heap = heap(&device, HeapType::Placement, 1 << 20, options);
                for mipmapped in [false, true] {
                    let mut descriptor =
                        TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 16, 16, mipmapped);
                    descriptor.resource_options = options;
                    descriptor.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
                    let what = format!(
                        "adopted {} (hostImageCopy enabled: {copies}) {storage:?} 16x16 of {} \
                         levels",
                        device.name(),
                        descriptor.mipmap_level_count
                    );
                    let Some(unchanged) = alias_round_trip(&queue, &heap, &descriptor, &what)
                    else {
                        continue;
                    };
                    println!("HEAPTEX {what}: {unchanged} texels unchanged by the buffer's write");
                    if copies || !hides {
                        assert_eq!(unchanged, 0, "{what}: texels the buffer's write left alone");
                    }
                }
            }
        }
        let reported = adopted.close();
        assert!(
            reported.is_empty(),
            "the validation layer reported: {reported:#?}"
        );
    }
}

/// The kind a heap's depth or stencil texture laid apart from the heap announces.
pub(super) const APART: [Degradation; 1] = [Degradation::HeapDepthStencilApart];

/// The kinds a heap's depth or stencil texture announces where the driver lays such images apart
/// from their memory (`DriverFact::DepthStencilImagesLieApart`): its image a shadow of the heap's
/// bytes (one sample, its texels inside its bytes; a texture Metal keeps losslessly compressed, a
/// shadow of its uncompressed tiles), else laid apart.
const DEPTH_STENCIL_ROUTES: [Degradation; 3] = [
    Degradation::ShadowedTexture,
    Degradation::HeapTextureLosslessLayout,
    Degradation::HeapDepthStencilApart,
];

/// The kind a depth or stencil texture without GPU-optimized contents announces where the driver
/// keeps its layout optimized.
pub(super) const OPTIMIZED: [Degradation; 1] = [Degradation::OptimizedDepthStencilTexture];

/// A depth texture without GPU-optimized contents, sized and made in a `Private` placement heap,
/// where the driver keeps depth and stencil images GPU-optimized
/// (`DriverFact::DepthStencilImagesStayGpuOptimized`: MoltenVK on Apple's GPUs), answers Metal's
/// own number for the descriptor as given: a 4-sample `Depth32Float` one 65536 bytes for 64x64,
/// where Metal answers 66048 with the property on, a single-sample one 16384 where it answers
/// 16512, and a 4-sample `Depth16Unorm` one 32768 where it answers 33280. None of them announces
/// `optimized-depth-stencil-texture`, which is the mark of the other route: the notice is
/// announced for a heap depth texture that is no shadow of its texels or samples in Metal's tiles
/// there (`heap_tiled.rs`), whose answer is then the optimized image's instead of Metal's, so an
/// empty count says every texture here is such a shadow. Made on the device first, none announces
/// it there either: nothing shows a device's own texture's layout; nor do a color texture without
/// the property, and a depth texture with it.
#[test]
fn a_heap_depth_texture_without_optimized_contents_answers_metal_s_footprint() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    let optimized = {
        let device = &validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let optimized = device
            .properties()
            .driver_facts
            .contains(&DriverFact::DepthStencilImagesStayGpuOptimized);
        let options = ResourceOptions::private();
        let target = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
        let make = |pixel_format, allow, samples| {
            let mut descriptor = TextureDescriptor::texture_2d(pixel_format, 64, 64, false);
            if samples > 1 {
                descriptor.texture_type = TextureType::Type2DMultisample;
                descriptor.sample_count = samples;
            }
            descriptor.usage = target;
            descriptor.resource_options = options;
            descriptor.allow_gpu_optimized_contents = allow;
            let answer = device.heap_texture_size_and_align(&descriptor).unwrap();
            let placement = heap(device, HeapType::Placement, answer.size, options);
            placement.new_texture_at_offset(&descriptor, 0).unwrap();
            device.new_texture(&descriptor).unwrap();
            answer
        };
        make(PixelFormat::RGBA8Unorm, false, 1);
        let with = make(PixelFormat::Depth32Float, true, 1);
        let single = make(PixelFormat::Depth32Float, false, 1);
        let with_samples = make(PixelFormat::Depth32Float, true, 4);
        let without = make(PixelFormat::Depth32Float, false, 4);
        assert!(!counts(&messages.take_notices()).contains_key(&OPTIMIZED[0]));
        let mut own = TextureDescriptor::texture_2d(PixelFormat::Depth16Unorm, 64, 64, false);
        own.texture_type = TextureType::Type2DMultisample;
        own.sample_count = 4;
        own.usage = target;
        own.resource_options = options;
        own.allow_gpu_optimized_contents = false;
        device.new_texture(&own).unwrap();
        assert!(
            !counts(&messages.take_notices()).contains_key(&OPTIMIZED[0]),
            "a device's own depth texture without optimized contents announced"
        );
        let short_with = make(PixelFormat::Depth16Unorm, true, 4);
        let short = make(PixelFormat::Depth16Unorm, false, 4);
        eprintln!(
            "HEAPTEX optimized depth: {} bytes with, {} without, one sample {} with, {} without; \
             16-bit {} with, {} without (fact {optimized})",
            with_samples.size, without.size, with.size, single.size, short_with.size, short.size
        );
        if optimized {
            assert_eq!((without.size, without.align), (65536, 128));
            assert_eq!((single.size, single.align), (16384, 128));
            assert_eq!((short.size, short.align), (32768, 128));
            assert_eq!((short_with.size, short_with.align), (33280, 128));
        }
        optimized
    };
    let notices = counts(&messages.take_notices());
    let announced = notices.get(&OPTIMIZED[0]).copied().unwrap_or(0);
    assert_eq!(
        announced, 0,
        "a heap depth texture that is a shadow of Metal's tiles announced (fact {optimized})"
    );
    layer_reported_nothing_at_all(&messages);
}

/// A mipmapped `Depth32Float_Stencil8` texture without GPU-optimized contents, sized and made in a
/// `Private` placement heap where the driver keeps depth and stencil images GPU-optimized
/// (`DriverFact::DepthStencilImagesStayGpuOptimized`: MoltenVK on Apple's GPUs), announces
/// `optimized-depth-stencil-texture`: a shadow of Metal's tiles moves one level of **each aspect**
/// per level of the texture, and the kernel's shape holds 16 of them
/// (`bind::TiledTexelsModule::MAX_LEVELS`, through `HeapTiledCopies::fit`), so a two-aspect texture
/// of 9 levels asks 18 and no shadow serves it. Its image is then a shadow of the heap's bytes in
/// rows rather than in Metal's tiles, announced `heap-texture-untiled-layout` beside this kind.
///
/// What the heap **answers** is Metal's model for the descriptor as given: 448256 bytes, the same
/// as the 8-level row's, where the same shape with `allowGPUOptimizedContents` on answers 451200,
/// the larger of the two. All three are asserted, so the row holds the kind's cost to what it
/// says: the answer is Metal's own for the descriptor, not the property-on texture's. What the
/// heap's bytes lose instead is the layout, which `heap-texture-untiled-layout` and
/// `shadowed-texture` carry.
///
/// Three controls bind that reason, each the announcing shape with one thing changed: **8 levels**,
/// 16 moves and the most the shape holds, announces nothing; a one-aspect `Depth32Float` texture of
/// the same 9 levels, 9 moves, announces nothing; and the 9-level texture with
/// `allowGPUOptimizedContents` **on** announces nothing, the property being half of what the notice
/// asks. Nor does the announcing shape as a device's own texture: nothing shows its layout there.
/// So an announcement here is the level bound and not the format, the mipmapping or the heap.
///
/// Measured on an Apple M3 Max under macOS 27.0.1 with MoltenVK 1.4.2 by a sweep of 11088 depth and
/// stencil shapes with the property off (six depth and stencil formats, nine texture types, seven
/// extents, up to five level counts, four usages and two storage modes), of which the device
/// answers 3880: 220 announce the kind. So it is reachable on this venue, and the announcing rows
/// are of two shapes -- a mipmapped two-aspect texture as here, and a small multisample depth or
/// stencil texture (4x4 and 37x5 at 2 and 4 samples, of shader-read usage, in a `Private` heap),
/// whose own reason is unaudited. This test takes the first, whose bound is a constant.
#[test]
fn a_heap_two_aspect_texture_of_more_levels_than_the_moves_hold_keeps_its_optimized_layout() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    {
        // Each occurrence announced, so a count is the number of sites reached and not one.
        let device = &validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
            options.announce_each = OPTIMIZED.into();
        })
        .unwrap();
        let optimized = device
            .properties()
            .driver_facts
            .contains(&DriverFact::DepthStencilImagesStayGpuOptimized);
        let options = ResourceOptions::private();
        let kind = || {
            counts(&messages.take_notices())
                .get(&OPTIMIZED[0])
                .copied()
                .unwrap_or(0)
        };
        // One row: `size` and `align` answered for the descriptor, and the kind's count at each of
        // three sites in turn -- the size query, a placement at offset 0 of a heap of exactly that
        // size, and the device's own texture -- so a count names the site that reached it.
        let row = |pixel_format, levels, allow| {
            let mut descriptor = TextureDescriptor::texture_2d(pixel_format, 256, 256, true);
            descriptor.mipmap_level_count = levels;
            descriptor.usage = TextureUsage::SHADER_READ;
            descriptor.resource_options = options;
            descriptor.allow_gpu_optimized_contents = allow;
            let _ = messages.take_notices();
            let answer = device.heap_texture_size_and_align(&descriptor).unwrap();
            let at_query = kind();
            let placement = heap(device, HeapType::Placement, answer.size, options);
            let placed = placement.new_texture_at_offset(&descriptor, 0).unwrap();
            let at_placement = kind();
            // The descriptor read back off the texture, so a row cannot be of another shape.
            assert_eq!(placed.mipmap_level_count(), levels, "{pixel_format:?}");
            assert_eq!(placed.pixel_format(), pixel_format);
            assert_eq!(placed.allow_gpu_optimized_contents(), allow, "{levels}");
            let own = device.new_texture(&descriptor).unwrap();
            let at_the_device = kind();
            assert_eq!(own.mipmap_level_count(), levels, "{pixel_format:?}");
            (answer, [at_query, at_placement, at_the_device])
        };
        let ds = PixelFormat::Depth32FloatStencil8;
        let (nine, nine_counts) = row(ds, 9, false);
        let (eight, eight_counts) = row(ds, 8, false);
        let (depth, depth_counts) = row(PixelFormat::Depth32Float, 9, false);
        let (with, with_counts) = row(ds, 9, true);
        eprintln!(
            "HEAPTEX optimized levels (query, placement, the device's own): 9 levels {nine:?} \
             {nine_counts:?}; 8 levels {eight:?} {eight_counts:?}; one aspect {depth:?} \
             {depth_counts:?}; the property on {with:?} {with_counts:?} (fact {optimized})"
        );
        if optimized {
            // The size query and the placement each announce once; the device's own texture, whose
            // layout nothing shows, does not.
            assert_eq!(
                nine_counts,
                [1, 1, 0],
                "the 9-level two-aspect texture's sites"
            );
            assert_eq!(eight_counts, [0, 0, 0], "8 levels announced");
            assert_eq!(depth_counts, [0, 0, 0], "one aspect of 9 levels announced");
            assert_eq!(with_counts, [0, 0, 0], "the property on announced");
            // Metal's model for the descriptor as given, not the property-on texture's.
            assert_eq!((nine.size, nine.align), (448256, 128));
            assert_eq!(eight, nine, "8 levels answered another chain than 9");
            assert_eq!((with.size, with.align), (451200, 128));
            assert_eq!((depth.size, depth.align), (349824, 128));
        } else {
            assert_eq!(nine_counts, [0, 0, 0], "announced without the fact");
        }
    }
    layer_reported_nothing_at_all(&messages);
}

/// The kinds a heap's colour texture announces on a driver whose compressed images hide writes
/// through an alias, by the route its image takes: host-transfer usage, a linear image, or the
/// optimal image, a near-equivalent.
pub(super) const ALIASES: [Degradation; 3] = [
    Degradation::HeapTextureHostTransfer,
    Degradation::HeapTextureLinear,
    Degradation::HeapTextureHidesAliasedWrites,
];

/// The kind a heap's colour texture Metal compresses announces on such a driver, where it keeps
/// the driver's compression under the default.
pub(super) const COMPRESSED: [Degradation; 1] = [Degradation::HeapTextureCompressed];

/// The usages the `Shared` heap sweep makes each format's texture with.
const SWEPT_USAGES: [(&str, TextureUsage); 4] = [
    ("read", TextureUsage::SHADER_READ),
    (
        "view",
        TextureUsage::SHADER_READ.union(TextureUsage::PIXEL_FORMAT_VIEW),
    ),
    (
        "write",
        TextureUsage::SHADER_READ.union(TextureUsage::SHADER_WRITE),
    ),
    (
        "render",
        TextureUsage::SHADER_READ.union(TextureUsage::RENDER_TARGET),
    ),
];

/// Whether `error` refuses a heap texture for what the device's linear images lack.
fn tiling_refusal(error: &Error) -> Option<LinearNeed> {
    match error {
        Error::Unsupported(u) => match u.feature {
            Feature::HeapTextureTiling { missing } => Some(missing),
            _ => None,
        },
        _ => None,
    }
}

/// Every pixel format's 16x16 2D texture of a `Shared` placement heap, read, viewed as other pixel
/// formats, written or rendered to, is made or refused typed, the query and the call alike: where
/// the device's optimal images take no memory a `Shared` heap lies in (the RTX 3090's), the texture
/// is a linear image in the heap's memory, refused `HeapTextureTiling` only for what the device's
/// linear images lack, which a `Private` heap's texture never is. Each uncompressed color format's
/// texture made that is not floating point takes bytes through `replace_region` and gives them back
/// through `get_bytes`, and again through a second `get_bytes`: a copy the driver makes after the
/// first read's submission must not change them (MoltenVK 1.4.1 copied a texture into its heap's
/// memory, which is the texture's own, once a command buffer with a barrier asking for host reads
/// completed; `image_barrier.rs` in the runtime). A texture of several levels, a cube, a 3D texture
/// and an array are made or refused alike. `HEAPTEX shared` lines print each refusal for linear
/// tiling and the counts, the second reads that differ among them, and the submissions on the
/// device's staging queue (a copy engine's, where texture copies and first moves run on a family
/// of transfers alone, `device/staging.rs` in the runtime). No colour texture is refused
/// `NotImplemented(HeapTextureMemory)`: one whose image needs extended usage, such as a written
/// sRGB texture, is checked format by format. `Unspecialized` takes Metal's answers: 0 bytes from
/// the query, at 128 for a 2D texture, a cube, a 3D texture or an array (`STACKED_NIL_ALIGNMENT`),
/// and no texture from the call
/// (`TextureProblem::Unspecialized`, Metal's nil; an M4 under macOS 27.0.1, every
/// usage). A format Metal makes no texture of from a descriptor alone, such as a two-plane YCbCr
/// one, takes Metal's answers too: 0 bytes from the query, at 16 for a two-plane YCbCr 2D texture
/// and 128 for another, and nil from the call (`Error::Nil`, `TextureProblem::NoTextureOfFormat`).
/// `R10Unorm_PACKED` and `RG10Unorm_PACKED` take the query's 0 bytes at 16 and the **gap's**
/// refusal from the call: plain Metal lays out no image storage for either, which is the 0 bytes,
/// and a placement heap of its makes a texture of either at offset 0 all the same
/// (`Footprint::NoTexture`), so a nil here would claim a refusal Metal does not make.
/// A format whose heap footprint its own structure decides and of which this crate holds no image
/// (`metal::format::footprint`) answers that footprint, non-zero, and refuses the call alone
/// `NotImplemented(Format)`: on an Apple M3 Max under macOS 27.0.1 (26A434) plain Metal answers
/// the footprint **and** makes the texture, so the refusal is this crate's open gap and the
/// `HEAPTEX shared no image` lines name every format and descriptor it covers. Every such format
/// answers in all four swept usages, counted from that table, so one that stops answering fails
/// rather than passing as a query and a call that both refuse. A footprint format held for copies
/// alone (`metal::format::copies_only`) is made instead, in all four, counted from that table too. An ASTC texture on a device that
/// holds it in a carrier with a decoded twin (lavapipe) answers Metal's model of its blocks and is
/// refused its making, `NotImplemented(DecodedHeapTexture)`: the twin would decode what the heap's
/// bytes held. `HEAPTEX shared decoded` lines name each.
#[test]
fn shared_heap_textures_are_made_or_refused_for_linear_tiling() {
    on_device(|device| {
        let options = ResourceOptions::new(
            StorageMode::Shared,
            CpuCacheMode::DefaultCache,
            HazardTrackingMode::Tracked,
        );
        let private = ResourceOptions::new(
            StorageMode::Private,
            CpuCacheMode::DefaultCache,
            HazardTrackingMode::Tracked,
        );
        let heap = heap(device, HeapType::Placement, 4 << 20, options);
        let unspecialized = nil(
            "new_texture_at_offset",
            Problem::Texture(TextureProblem::Unspecialized),
        );
        let no_texture = nil(
            "new_texture_at_offset",
            Problem::Texture(TextureProblem::NoTextureOfFormat),
        );
        let staged = device.work_counts().staging_submissions;
        let (mut made, mut round_trips, mut refused) = (0usize, 0usize, Vec::new());
        let mut copies = 0usize;
        let mut changed = Vec::new();
        let mut changed_again = Vec::new();
        let mut gaps = Vec::new();
        let mut no_image = Vec::new();
        let mut decoded = Vec::new();
        let mut check = |what: String, mut descriptor: TextureDescriptor| {
            descriptor.resource_options = options;
            let query = device.heap_texture_size_and_align(&descriptor);
            let call = heap.new_texture_at_offset(&descriptor, 0);
            match (query, call) {
                (Ok(_), Ok(texture)) => {
                    made += 1;
                    copies += usize::from(copies_only(descriptor.pixel_format));
                    let Some(bytes) = texel_bytes(descriptor.pixel_format) else {
                        return;
                    };
                    if descriptor.texture_type != TextureType::Type2D
                        || floating(descriptor.pixel_format)
                    {
                        return;
                    }
                    let row = 16 * bytes;
                    let region = Region::new_2d(0, 0, 16, 16);
                    let src = pattern(row * 16, descriptor.pixel_format.as_raw() as u8);
                    texture
                        .replace_region(region, 0, 0, &src, row as u64, 0)
                        .unwrap_or_else(|e| panic!("{what}: {e:?}"));
                    let mut back = vec![0xeeu8; row * 16];
                    texture
                        .get_bytes(&mut back, row as u64, 0, region, 0, 0)
                        .unwrap_or_else(|e| panic!("{what}: {e:?}"));
                    let mut again = vec![0xeeu8; row * 16];
                    texture
                        .get_bytes(&mut again, row as u64, 0, region, 0, 0)
                        .unwrap_or_else(|e| panic!("{what}: {e:?}"));
                    if again != src {
                        changed_again.push(what.clone());
                    }
                    if back != src {
                        changed.push(what);
                    }
                    round_trips += 1;
                }
                // Metal's nil, and its query's 0 bytes at the type's alignment.
                (Ok(answer), Err(error)) if error == unspecialized => {
                    let align = match descriptor.texture_type {
                        TextureType::Type1D | TextureType::Type2D => 128,
                        TextureType::TextureBuffer => 16,
                        _ => STACKED_NIL_ALIGNMENT,
                    };
                    assert_eq!(
                        (answer.size, answer.align, descriptor.pixel_format),
                        (0, align, PixelFormat::Unspecialized),
                        "{what}"
                    )
                }
                // Metal's nil for a format it makes no texture of from a descriptor alone, and
                // its query's 0 bytes at 16 where the format's bytes are a surface's rows rather
                // than Metal's tiles, 128 where they are tiles.
                (Ok(answer), Err(error)) if error == no_texture => {
                    let pf = descriptor.pixel_format;
                    // The two-plane YCbCr formats, `RGBA16Float_XR`, `AnyDepth` and `AnyStencil`
                    // (`no_texture_alone`). The two packed 10-bit formats are **not** here: a
                    // placement heap of plain Metal's answers one of them a texture of 0
                    // allocated bytes at offset 0, where the device's creator and an automatic
                    // heap answer nil, so this creator's refusal of them is this crate's own gap
                    // and reaches the arm below.
                    assert!(no_texture_alone(pf), "{what}: {error:?}");
                    let rows = metal2vulkan::metal::format::ycbcr_format(pf).is_some();
                    let align = if rows { 16 } else { 128 };
                    assert_eq!((answer.size, answer.align), (0, align), "{what}")
                }
                // Metal answers the footprint **and** makes the texture, measured for every
                // format of the family; this crate answers that footprint from the format's own
                // structure, which needs no image, and holds no image to make the texture in. So
                // the making alone stays refused `NotImplemented(Format)` -- the typed refusal
                // that records the gap -- and the answer is the one the query is asked for.
                (Ok(answer), Err(Error::Unsupported(ref u)))
                    if (&u.feature, &u.needs) == (&Feature::Format, &Needs::NotImplemented) =>
                {
                    let pf = descriptor.pixel_format;
                    let Some(shape) = footprint(pf) else {
                        panic!("{what}: answered {answer:?} with no footprint of its own")
                    };
                    // A packed 10-bit format is the one whose footprint is `NoTexture`, 0 bytes
                    // at 16: Metal lays out no image storage for it anywhere, which is why the
                    // query counts nothing, and a placement heap makes a texture of it over the
                    // caller's offset all the same (`Footprint::NoTexture`'s own docs). So its 0-byte
                    // answer stands beside the refusal here, where every other format of this arm
                    // answers a footprint of its own.
                    if shape == Footprint::NoTexture {
                        assert_eq!(
                            (answer.size, answer.align),
                            (0, 16),
                            "{what}: answered {answer:?}"
                        );
                    } else {
                        assert_ne!(
                            shape,
                            Footprint::NoTexture,
                            "{what}: a format Metal makes no texture of answered {answer:?}"
                        );
                        assert!(answer.size > 0, "{what}: answered {answer:?}");
                    }
                    no_image.push(format!("{what}: {answer:?}"));
                }
                // An ASTC texture on a device that holds it in a carrier with a decoded twin, which
                // would decode what the heap's bytes held: the query answers Metal's model of its
                // blocks and the making is refused, the runtime's gap.
                (Ok(answer), Err(Error::NotImplemented(RuntimeGap::DecodedHeapTexture))) => {
                    let pf = descriptor.pixel_format;
                    assert!(
                        !super::astc_textures::decodes_astc(device, pf),
                        "{what}: refused as a decoded heap texture on a device that decodes it"
                    );
                    let metal = device.metal_heap_texture_size_and_align(&descriptor);
                    assert_eq!(metal, Ok(Some(answer)), "{what}: Metal's model");
                    decoded.push(format!("{what}: {answer:?}"));
                }
                (Ok(answer), Err(error)) => panic!("{what}: answered {answer:?}, made {error:?}"),
                (Err(error), Ok(_)) => panic!("{what}: refused {error:?} by the query alone"),
                (Err(query), Err(call)) => {
                    let colour = !is_depth_stencil(descriptor.pixel_format);
                    if colour && call == Error::NotImplemented(RuntimeGap::HeapTextureMemory) {
                        gaps.push(format!("{what}: {call:?}"));
                    }
                    let Some(missing) = tiling_refusal(&query) else {
                        return;
                    };
                    assert_eq!(query, call, "{what}");
                    let mut privately = descriptor;
                    privately.resource_options = private;
                    let answer = device.heap_texture_size_and_align(&privately);
                    assert!(
                        answer.as_ref().err().and_then(tiling_refusal).is_none(),
                        "{what}: a Private heap's texture refused {answer:?}"
                    );
                    refused.push(format!("{what}: {missing:?}"));
                }
            }
        };
        for &pf in PixelFormat::ALL {
            for (label, usage) in SWEPT_USAGES {
                let mut descriptor = TextureDescriptor::texture_2d(pf, 16, 16, false);
                descriptor.usage = usage;
                check(format!("{pf:?} {label}"), descriptor);
            }
        }
        let rgba8 = PixelFormat::RGBA8Unorm;
        let mut mipmapped = TextureDescriptor::texture_2d(rgba8, 16, 16, true);
        mipmapped.usage = TextureUsage::SHADER_READ;
        check("RGBA8Unorm with 5 levels".to_owned(), mipmapped);
        let mut cube = TextureDescriptor::texture_cube(rgba8, 16, false);
        cube.usage = TextureUsage::SHADER_READ;
        check("RGBA8Unorm cube".to_owned(), cube);
        let mut volume = TextureDescriptor::texture_2d(rgba8, 16, 16, false);
        volume.texture_type = TextureType::Type3D;
        volume.depth = 4;
        volume.usage = TextureUsage::SHADER_READ;
        check("RGBA8Unorm 3D".to_owned(), volume);
        let mut array = TextureDescriptor::texture_2d(rgba8, 16, 16, false);
        array.texture_type = TextureType::Type2DArray;
        array.array_length = 4;
        array.usage = TextureUsage::SHADER_READ;
        check("RGBA8Unorm array of 4".to_owned(), array);
        for line in &refused {
            eprintln!("HEAPTEX shared refused {line}");
        }
        for line in &no_image {
            eprintln!("HEAPTEX shared no image {line}");
        }
        for line in &decoded {
            eprintln!("HEAPTEX shared decoded {line}");
        }
        eprintln!(
            "HEAPTEX shared on {}: {made} made ({round_trips} round trips, {} second reads \
             differ, {} staging queue submissions), {} refused for linear tiling, {} answered a \
             footprint with no image of the format",
            device.name(),
            changed_again.len(),
            device.work_counts().staging_submissions - staged,
            refused.len(),
            no_image.len()
        );
        assert!(made > 0, "no Shared heap texture was made");
        assert!(
            changed.is_empty(),
            "bytes changed by the round trip: {changed:?}"
        );
        assert!(
            changed_again.is_empty(),
            "bytes changed between the round trip and a second read: {changed_again:?}"
        );
        assert!(
            gaps.is_empty(),
            "colour textures refused as gaps: {gaps:#?}"
        );
        // Every format whose footprint its own structure decides answered that footprint in all
        // four swept usages: one held for copies alone (`copies_only`) with its texture made, every
        // other with no image of the format, the packed 10-bit pair's `NoTexture` 0 bytes at 16
        // among them (the arm above). The counts come from the product's own tables rather than
        // numbers here, so a footprint that stops answering, or a held format whose making stops,
        // fails this instead of passing quietly as a query and a call that both refuse.
        let held = PixelFormat::ALL
            .iter()
            .filter(|&&pf| copies_only(pf))
            .count();
        let answering = PixelFormat::ALL
            .iter()
            .filter(|&&pf| footprint(pf).is_some() && !copies_only(pf))
            .count();
        assert_eq!(
            copies,
            held * SWEPT_USAGES.len(),
            "textures made of the formats held for copies alone"
        );
        assert_eq!(
            no_image.len(),
            answering * SWEPT_USAGES.len(),
            "footprints answered with no image of the format"
        );
    });
}

/// Whether `pf`'s texels are floating point, whose bytes a round trip need not keep: a NaN's
/// payload is the device's to canonicalize (MoltenVK's `RG11B10Float` in a heap does, at
/// `8b98df40`).
fn floating(pf: PixelFormat) -> bool {
    metal2vulkan::metal::format::direct_format(pf)
        .and_then(|format| format.block())
        .and_then(|block| block.color)
        .is_some_and(|color| matches!(color, NumericFormat::Sfloat | NumericFormat::Ufloat))
}

/// Zero `size` bytes of `buffer` with a blit.
fn gpu_fill_zero(queue: &CommandQueue, buffer: &Buffer, size: u64) {
    super::host_memory::gpu_fill(queue, buffer, 0..size, 0);
}

/// Reads four texture buffers of 256 texels each into its output buffer.
const FOUR_TEXEL_READS: &str = "half-conversion-by-texture-buffer-reads";

/// Writes eight `rgba8_uint` texels through a texture buffer (texture 1, its output).
const TEXEL_WRITER: &str = "write-four-bytes-into-a-storage-texel-buffer";

/// Tracked options of `storage`, a heap's and its resources'.
fn tracked(storage: StorageMode) -> ResourceOptions {
    ResourceOptions::new(
        storage,
        CpuCacheMode::DefaultCache,
        HazardTrackingMode::Tracked,
    )
}

/// A texture buffer of a heap takes Metal's answer, its row of texels at a multiple of 16 bytes (a
/// Metal heap probe, an M3 Max and an M4 alike, plain and validated): a placement heap places it at
/// every multiple of 16 with its bytes inside the heap, and refuses an offset off it and bytes past
/// the end as Metal's validation stops them (`Invalid`); an automatic heap places it first fit,
/// answers `heap_offset` 0, counts its row in `used_size` (400 and 400 and a 16-byte buffer, 816,
/// as Metal counted) and takes the range back when it is made aliasable or goes. The texture
/// answers its heap, in a `Shared` and a `Private` heap.
#[test]
fn heap_texture_buffers_answer_and_place_as_metal_s_do() {
    const CALL: &str = "new_texture_at_offset";
    on_device(|device| {
        let usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
        for storage in [StorageMode::Shared, StorageMode::Private] {
            let options = tracked(storage);
            let formats = [
                (PixelFormat::R8Unorm, 1),
                (PixelFormat::RGBA8Unorm, 4),
                (PixelFormat::R32Uint, 4),
                (PixelFormat::RGBA32Float, 16),
            ];
            for (pixel_format, texel) in formats {
                for width in [1, 3, 100, 4096] {
                    let descriptor =
                        TextureDescriptor::texture_buffer(pixel_format, width, options, usage);
                    let answer = device.heap_texture_size_and_align(&descriptor).unwrap();
                    assert_eq!(
                        (answer.size, answer.align),
                        (width * texel, 16),
                        "{storage:?} {pixel_format:?} x {width}"
                    );
                }
            }
            let placement = heap(device, HeapType::Placement, 1 << 16, options);
            let descriptor =
                TextureDescriptor::texture_buffer(PixelFormat::R32Uint, 64, options, usage);
            let size = 256;
            let last = placement.size() - size;
            let mut placed = Vec::new();
            for offset in [0, 16, 64, 256, 4096 + 16, last] {
                let texture = placement
                    .new_texture_at_offset(&descriptor, offset)
                    .unwrap();
                assert_eq!(texture.heap_offset(), offset);
                assert_eq!(texture.heap(), Some(&placement));
                assert_eq!(texture.texture_type(), TextureType::TextureBuffer);
                assert_eq!(texture.storage_mode(), storage);
                placed.push(texture);
            }
            assert_eq!(placement.used_size(), 0);
            assert_eq!(
                placement.new_texture_at_offset(&descriptor, 4).err(),
                Some(nil(CALL, Problem::Heap(HeapProblem::Offset { offset: 4 })))
            );
            let past = last + 16;
            assert_eq!(
                placement.new_texture_at_offset(&descriptor, past).err(),
                Some(nil(
                    CALL,
                    Problem::Heap(HeapProblem::PastEnd {
                        offset: past,
                        length: size,
                        heap_size: placement.size(),
                    })
                ))
            );
            assert!(placed[0].is_aliasable() && placed[1].is_aliasable());
            placed[0].make_aliasable().unwrap();
            assert!(placed[0].is_aliasable() && placed[1].is_aliasable());

            let automatic = heap(device, HeapType::Automatic, 1 << 20, options);
            let descriptor =
                TextureDescriptor::texture_buffer(PixelFormat::RGBA8Unorm, 100, options, usage);
            let first = automatic.new_texture(&descriptor).unwrap();
            assert_eq!(first.heap_offset(), 0, "an automatic heap's offset is 0");
            assert_eq!(first.heap(), Some(&automatic));
            assert_eq!(automatic.used_size(), 400);
            let second = automatic.new_texture(&descriptor).unwrap();
            let buffer = automatic.new_buffer(16, options).unwrap();
            assert_eq!(automatic.used_size(), 816);
            first.make_aliasable().unwrap();
            assert_eq!(automatic.used_size(), 416);
            let third = automatic.new_texture(&descriptor).unwrap();
            assert_eq!(automatic.used_size(), 816);
            drop((first, second, third, buffer));
            assert_eq!(automatic.used_size(), 0);
        }
    });
}

/// Write `bytes` into `buffer` from its first byte with a blit from a staged copy.
fn gpu_write(queue: &CommandQueue, buffer: &Buffer, bytes: &[u8]) {
    let source = staged(queue.device(), bytes);
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_buffer_to_buffer(&source, 0, buffer, 0, bytes.len() as u64)
        .unwrap();
    blit.end_encoding().unwrap();
    run(&command_buffer);
}

/// `case`'s one dispatch over `resources`, and its buffers after it.
fn dispatch_case(
    device: &Device,
    case: &PublicCase,
    resources: &CaseResources,
) -> BTreeMap<u32, Buffer> {
    let buffers = bound_buffers(device, &case.compute, resources).unwrap();
    let pipeline = pipeline_state(device, case).unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encode_case(&mut encoder, &pipeline, &case.compute, &buffers, resources).unwrap();
    encoder.end_encoding().unwrap();
    run(&command_buffer);
    buffers
}

/// `case`'s texture at `index`, a texture buffer of `heap` at `offset`.
fn heap_texture_buffer(case: &PublicCase, index: u32, heap: &Heap, offset: u64) -> Texture {
    let literal = &case.compute.textures[&index];
    let mut descriptor = texture_descriptor(literal, case.texture_usage[&index]).unwrap();
    descriptor.resource_options = heap.resource_options();
    heap.new_texture_at_offset(&descriptor, offset).unwrap()
}

/// A texture buffer of a heap is the heap's bytes from its offset, in order (a Metal heap probe on
/// the M3 Max and an M4: a kernel read what a buffer over the bytes held, and its writes showed in
/// the buffer there and nowhere else): the four texture buffers `FOUR_TEXEL_READS` reads, packed at
/// 16-byte steps of a placement heap whose bytes a buffer over the heap wrote, give its own run's
/// output; and the texture buffer `TEXEL_WRITER` writes, at byte 48, writes there what its own run
/// writes and leaves the bytes around it; in a `Shared` and a `Private` heap.
#[test]
fn a_heap_texture_buffer_is_the_heap_s_bytes() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let reader = public_case(FOUR_TEXEL_READS).unwrap();
        let read = run_case(device, &reader).unwrap().output;
        let writer = public_case(TEXEL_WRITER).unwrap();
        let written = run_case(device, &writer).unwrap().output;
        for storage in [StorageMode::Shared, StorageMode::Private] {
            let options = tracked(storage);
            let placement = heap(device, HeapType::Placement, 1 << 16, options);
            let bytes = placement
                .new_buffer_at_offset(placement.size(), options, 0)
                .unwrap();
            let mut image = vec![0x5a_u8; placement.size() as usize];
            let mut at = 16;
            let mut placed = Vec::new();
            for (&index, literal) in &reader.compute.textures {
                let length = literal.bytes.len();
                image[at..at + length].copy_from_slice(&literal.bytes);
                placed.push((index, at as u64));
                at = (at + length).next_multiple_of(16);
            }
            gpu_write(&queue, &bytes, &image);
            let mut resources = CaseResources::new(device, &reader).unwrap();
            for &(index, offset) in &placed {
                let texture = heap_texture_buffer(&reader, index, &placement, offset);
                resources.textures.insert(index, texture);
            }
            let buffers = dispatch_case(device, &reader, &resources);
            let ComputeOutput::Buffer {
                binding,
                offset,
                length,
            } = reader.compute.output
            else {
                panic!("{FOUR_TEXEL_READS}: a buffer output");
            };
            let output = read_buffer(&buffers[&binding], offset, length).unwrap();
            assert_eq!(output, read, "{storage:?}: the texels at {placed:?}");

            // The writer's texture buffer at byte 48, between bytes of 0x5a.
            let literal = &writer.compute.textures[&1];
            let (start, end) = (48, 48 + literal.bytes.len());
            let mut image = vec![0x5a_u8; 4096];
            image[start..end].copy_from_slice(&literal.bytes);
            gpu_write(&queue, &bytes, &image);
            let mut resources = CaseResources::new(device, &writer).unwrap();
            let texture = heap_texture_buffer(&writer, 1, &placement, start as u64);
            resources.textures.insert(1, texture);
            dispatch_case(device, &writer, &resources);
            let after = gpu_read(&queue, &bytes, 0..4096);
            assert_eq!(
                after[start..end],
                written[..],
                "{storage:?}: the texels written"
            );
            let around = after[..start].iter().chain(&after[end..]);
            assert!(
                around.copied().all(|byte| byte == 0x5a),
                "{storage:?}: a byte around the texture buffer changed"
            );
        }
    });
}

/// Two `Depth32Float` textures a shader reads, side by side in a placement heap, keep their texels
/// (a Metal heap probe, an M3 Max and an M4 alike: each read back as uploaded, then a buffer filled
/// over the first one's answered bytes changed every texel of it and none of the second's). Each
/// reads back what a texture outside a heap of the same upload reads (the device's depth image may
/// store a value outside `[0, 1]` clamped or a denormal as zero, announced), and the comparisons
/// of both compare as a texture's outside a heap does; in a `Shared` and a `Private` heap.
/// `HEAPTEX depth` lines print the answer and the device's notices.
///
/// Where the device's driver lays depth and stencil images apart from their memory
/// (`DriverFact::DepthStencilImagesLieApart`: MoltenVK, which makes such an image's Metal texture
/// from the device, never from the memory's heap; an M4, MoltenVK 1.4.1, where
/// the fill changed none), each texture's image is a shadow of the heap's bytes: the fill changes
/// every texel of the first, Metal's answer, and the device announces `shadowed-texture` once,
/// never `heap-depth-stencil-apart`. Where no memory a `Shared` heap lies in takes the device's
/// optimal depth image and no linear one serves (the RTX 3090), the `Shared` heap's texture is a
/// shadow of the heap's bytes too: the fill changes every texel of the first, announced
/// `shadowed-texture` once. Elsewhere neither is announced.
#[test]
fn heap_depth_textures_keep_their_bytes() {
    use super::depth_compares::{pipeline, run as compare, SIDE, VALUES};
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    // Whether the driver lays the textures apart, whether one was made in a heap, and whether
    // the fill changed every texel of the first in each storage made.
    let (apart, made, all_changed) = {
        let device = &validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let apart = device
            .properties()
            .driver_facts
            .contains(&DriverFact::DepthStencilImagesLieApart);
        let mut made = false;
        let mut all_changed = Vec::new();
        let queue = device.new_command_queue().unwrap();
        let side = u64::from(SIDE);
        let first: Vec<u8> = VALUES.iter().flat_map(|v| v.to_le_bytes()).collect();
        let second: Vec<u8> = VALUES.iter().rev().flat_map(|v| v.to_le_bytes()).collect();
        // A sample_compare at each texel's centre against references in and outside [0, 1].
        let step = 1.0 / SIDE as f32;
        let jobs: Vec<[f32; 4]> = (0..SIDE * SIDE)
            .flat_map(|i| {
                let (x, y) = ((i % SIDE) as f32, (i / SIDE) as f32);
                [-0.5, 0.25, 0.5, 0.75, 1.5, 1000.0]
                    .map(|r| [(x + 0.5) * step, (y + 0.5) * step, r, 0.0])
            })
            .collect();
        let state = pipeline(device, "msf_cmp_sample2d", false);
        let plain = |bytes: &[u8]| {
            let mut descriptor =
                TextureDescriptor::texture_2d(PixelFormat::Depth32Float, side, side, false);
            descriptor.resource_options = ResourceOptions::private();
            let texture = device.new_texture(&descriptor).unwrap();
            upload(&queue, &[(&texture, bytes)], side);
            let compares = compare(device, (&texture, None), &state, &jobs, 1);
            (texels(&queue, &texture, side), compares)
        };
        let ((first_kept, first_compares), (second_kept, second_compares)) =
            (plain(&first), plain(&second));
        for storage in [StorageMode::Shared, StorageMode::Private] {
            let options = tracked(storage);
            let mut descriptor =
                TextureDescriptor::texture_2d(PixelFormat::Depth32Float, side, side, false);
            descriptor.resource_options = options;
            descriptor.usage = TextureUsage::SHADER_READ;
            let what = format!("{storage:?}");
            let answer = match device.heap_texture_size_and_align(&descriptor) {
                Ok(answer) => answer,
                Err(error) if tiling_refusal(&error).is_some() => {
                    eprintln!("HEAPTEX depth {what}: refused {error:?}");
                    continue;
                }
                Err(error) => panic!("{what}: {error:?}"),
            };
            let SizeAndAlign { size, align, .. } = answer;
            let placement = heap(device, HeapType::Placement, 1 << 20, options);
            let next = size.next_multiple_of(align);
            let a = placement.new_texture_at_offset(&descriptor, 0).unwrap();
            let b = placement.new_texture_at_offset(&descriptor, next).unwrap();
            made = true;
            // The second first: the first's upload lands after it.
            upload(&queue, &[(&b, &second), (&a, &first)], side);
            assert_eq!(texels(&queue, &a, side), first_kept, "{what}: the first");
            assert_eq!(texels(&queue, &b, side), second_kept, "{what}: the second");
            let compared = compare(device, (&a, None), &state, &jobs, 1);
            assert_eq!(bits(&compared), bits(&first_compares), "{what}: the first");
            let compared = compare(device, (&b, None), &state, &jobs, 1);
            assert_eq!(
                bits(&compared),
                bits(&second_compares),
                "{what}: the second"
            );
            // A buffer over the first's bytes, filled: the second keeps its texels.
            let over = placement.new_buffer_at_offset(size, options, 0).unwrap();
            gpu_fill(&queue, &over, 0..size, 0x3e);
            assert_eq!(
                texels(&queue, &b, side),
                second_kept,
                "{what}: after the fill"
            );
            let compared = compare(device, (&b, None), &state, &jobs, 1);
            assert_eq!(
                bits(&compared),
                bits(&second_compares),
                "{what}: after the fill"
            );
            let changed = texels(&queue, &a, side)
                .as_chunks::<4>()
                .0
                .iter()
                .zip(first_kept.as_chunks::<4>().0.iter())
                .filter(|(after, before)| after != before)
                .count();
            eprintln!(
                "HEAPTEX depth {what}: size {size} align {align}; the fill changed {changed} of \
                 the first's {} texels",
                side * side
            );
            all_changed.push((storage, changed == (side * side) as usize));
            if apart {
                assert_eq!(
                    changed,
                    (side * side) as usize,
                    "{what}: a shadow of the heap's bytes kept texels the fill covered"
                );
            } else {
                assert!(
                    changed > 0,
                    "{what}: the fill over the first changed none of its texels"
                );
            }
        }
        (apart, made, all_changed)
    };
    let notices = counts(&messages.take_notices());
    let count = |kind| notices.get(&kind).copied().unwrap_or(0);
    let aparts = count(APART[0]);
    let shadowed = count(Degradation::ShadowedTexture);
    eprintln!("HEAPTEX depth: {aparts} heap-depth-stencil-apart, {shadowed} shadowed-texture");
    assert_eq!(aparts, 0);
    // A shadow of a `Shared` heap's bytes where its memory takes no optimal depth image: every
    // texel the fill covered changed, as in a shadow where the driver lays the image apart.
    let shared_shadow =
        !apart && shadowed == 1 && all_changed.contains(&(StorageMode::Shared, true));
    if shared_shadow {
        eprintln!("HEAPTEX depth: the Shared heap's texture is a shadow of the heap's bytes");
    }
    assert_eq!(shadowed, usize::from((apart && made) || shared_shadow));
    layer_reported_nothing_at_all(&messages);
}

/// A heap's single-sample depth and stencil textures keep their texels where Metal's tiles place
/// them in the heap's bytes (`heap_layouts::tiled_offset`), at Metal's answer, where the driver
/// lays such images apart from their memory or keeps them GPU-optimized
/// (`DriverFact::DepthStencilImagesLieApart`, `DriverFact::DepthStencilImagesStayGpuOptimized`:
/// MoltenVK): a Metal probe on the M3 Max under macOS 27.0.1 found every texel of `Depth16Unorm`,
/// `Depth32Float` and `Stencil8` textures of 16x16, 37x5, 64x64, 100x70 and 300x200 with
/// `allowGPUOptimizedContents` off there, uploaded by a blit naming each texel; and each of a
/// `Depth32Float_Stencil8` texture's in two planes, its depth as a `Depth32Float` texture's at its
/// first byte and its stencil as a `Stencil8` texture's from the `Depth32Float` answer rounded up
/// to 16 KiB (one level and mipmapped, one slice and arrays). Each such texture here, in a
/// `Shared` placement heap, each aspect uploaded so, lies in a buffer over its bytes where the
/// tiles place it, and bytes a blit writes there through the buffer read back through a blit as
/// those texels. With the property on, a 64x64 `Depth32Float` texture (which Metal keeps
/// losslessly compressed, 16512 bytes) holds its uncompressed tiles there, announced
/// `heap-texture-lossless-layout` once. Elsewhere `HEAPTEX tiles` lines print whether they lie so;
/// but an `X32_Stencil8` texture's stencil lies as a `Stencil8` texture's on every device (the
/// probe alike: 8x8 to 300x200, both ways), its image never fitting the `Stencil8` answer.
#[test]
fn heap_depth_textures_keep_their_texels_in_metal_s_tiles() {
    use super::heap_layouts::tiled_offset;
    use BlitOption::{DepthFromDepthStencil as DEPTH, None as WHOLE, StencilFromDepthStencil};
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    let tiled = {
        let device = &validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let facts = &device.properties().driver_facts;
        let tiled = facts.contains(&DriverFact::DepthStencilImagesLieApart)
            || facts.contains(&DriverFact::DepthStencilImagesStayGpuOptimized);
        let queue = device.new_command_queue().unwrap();
        let options = tracked(StorageMode::Shared);
        let ds = PixelFormat::Depth32FloatStencil8;
        // Each shape's format, extent, `allowGPUOptimizedContents` and Metal's size (every
        // alignment 128), measured, and its aspects: each one's element bytes, blit option and
        // first byte from the texture's.
        type Aspects = &'static [(u64, BlitOption, u64)];
        let shapes: [(PixelFormat, [u64; 2], bool, u64, Aspects); 14] = [
            (
                PixelFormat::Depth16Unorm,
                [37, 5],
                false,
                0x400,
                &[(2, WHOLE, 0)],
            ),
            (
                PixelFormat::Depth16Unorm,
                [100, 70],
                false,
                0x8000,
                &[(2, WHOLE, 0)],
            ),
            (
                PixelFormat::Depth32Float,
                [37, 5],
                false,
                0x800,
                &[(4, WHOLE, 0)],
            ),
            (
                PixelFormat::Depth32Float,
                [64, 64],
                false,
                0x4000,
                &[(4, WHOLE, 0)],
            ),
            (
                PixelFormat::Depth32Float,
                [100, 70],
                false,
                0x10000,
                &[(4, WHOLE, 0)],
            ),
            (
                PixelFormat::Stencil8,
                [37, 5],
                false,
                0x200,
                &[(1, WHOLE, 0)],
            ),
            (
                PixelFormat::Stencil8,
                [100, 70],
                false,
                0x4000,
                &[(1, WHOLE, 0)],
            ),
            (
                PixelFormat::Depth32Float,
                [64, 64],
                true,
                0x4080,
                &[(4, WHOLE, 0)],
            ),
            (
                ds,
                [37, 5],
                false,
                0x4200,
                &[(4, DEPTH, 0), (1, StencilFromDepthStencil, 0x4000)],
            ),
            (
                ds,
                [64, 64],
                false,
                0x5000,
                &[(4, DEPTH, 0), (1, StencilFromDepthStencil, 0x4000)],
            ),
            (
                ds,
                [100, 70],
                false,
                0x14000,
                &[(4, DEPTH, 0), (1, StencilFromDepthStencil, 0x10000)],
            ),
            (
                PixelFormat::X32Stencil8,
                [37, 5],
                false,
                0x200,
                &[(1, WHOLE, 0)],
            ),
            (
                PixelFormat::X32Stencil8,
                [100, 70],
                false,
                0x4000,
                &[(1, WHOLE, 0)],
            ),
            (
                PixelFormat::X32Stencil8,
                [300, 200],
                false,
                0x18000,
                &[(1, WHOLE, 0)],
            ),
        ];
        for (format, [w, h], allow, metal, aspects) in shapes {
            let what = format!("{format:?} {w}x{h} allow {allow}");
            let mut descriptor = TextureDescriptor::texture_2d(format, w, h, false);
            descriptor.resource_options = options;
            // A stencil view's texture is read, its stencil blitted.
            let view = format == PixelFormat::X32Stencil8;
            descriptor.usage = match view {
                true => TextureUsage::SHADER_READ,
                false => TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET,
            };
            descriptor.allow_gpu_optimized_contents = allow;
            let answer = device.heap_texture_size_and_align(&descriptor).unwrap();
            let heap = heap(device, HeapType::Placement, 1 << 20, options);
            let texture = heap.new_texture_at_offset(&descriptor, 0).unwrap();
            let over = heap.new_buffer_at_offset(answer.size, options, 0).unwrap();
            let count = w * h;
            let (mut off, mut read_off) = (0, 0);
            for &(unit, option, first) in aspects {
                // Texel `i`'s element, naming it: a depth of (i + 1) / 2^23 (exact), a 16-bit
                // unorm of i + 1, a stencil byte of i + 1.
                let element = |i: u64| -> Vec<u8> {
                    match unit {
                        4 => ((i + 1) as f32 / (1u32 << 23) as f32)
                            .to_le_bytes()
                            .to_vec(),
                        2 => ((i + 1) as u16).to_le_bytes().to_vec(),
                        _ => vec![(i + 1) as u8],
                    }
                };
                let named: Vec<u8> = (0..count).flat_map(element).collect();
                let row = w * unit;
                let blit = |into: bool, buffer: &Buffer| {
                    let command_buffer = queue.command_buffer().unwrap();
                    let mut blit = command_buffer.blit_command_encoder().unwrap();
                    let (origin, size) = (Origin::new(0, 0, 0), Size::new(w, h, 1));
                    match into {
                        true => blit.copy_from_buffer_to_texture(
                            buffer,
                            0,
                            row,
                            row * h,
                            size,
                            &texture,
                            0,
                            0,
                            origin,
                            option,
                        ),
                        false => blit.copy_from_texture_to_buffer(
                            &texture,
                            0,
                            0,
                            origin,
                            size,
                            buffer,
                            0,
                            row,
                            row * h,
                            option,
                        ),
                    }
                    .unwrap();
                    blit.end_encoding().unwrap();
                    run(&command_buffer);
                };
                gpu_fill(&queue, &over, 0..answer.size, 0xee);
                blit(true, &staged(device, &named));
                let bytes = gpu_read(&queue, &over, 0..answer.size);
                let at = |i: u64| (first + tiled_offset(unit, [w, h], i % w, i / w)) as usize;
                off += (0..count)
                    .filter(|&i| bytes.get(at(i)..at(i) + unit as usize) != Some(&element(i)[..]))
                    .count();
                // Bytes written through the buffer where the tiles place each texel, naming it
                // backwards, read back through a blit.
                let mut backwards = bytes;
                for i in 0..count {
                    let at = at(i);
                    if let Some(bytes) = backwards.get_mut(at..at + unit as usize) {
                        bytes.copy_from_slice(&element(count - 1 - i));
                    }
                }
                let source = staged(device, &backwards);
                let command_buffer = queue.command_buffer().unwrap();
                let mut copy = command_buffer.blit_command_encoder().unwrap();
                copy.copy_from_buffer_to_buffer(&source, 0, &over, 0, answer.size)
                    .unwrap();
                copy.end_encoding().unwrap();
                run(&command_buffer);
                let read = staged(device, &vec![0; (row * h) as usize]);
                blit(false, &read);
                let read = read_buffer(&read, 0, row * h).unwrap();
                read_off += (0..count)
                    .filter(|&i| {
                        let at = (i * unit) as usize;
                        read[at..at + unit as usize] != element(count - 1 - i)[..]
                    })
                    .count();
            }
            let texels = count * aspects.len() as u64;
            eprintln!(
                "HEAPTEX tiles {what}: size {} align {}, Metal's {metal} 128; {off} of {texels} \
                 texels off Metal's tiles, {read_off} read otherwise from them",
                answer.size, answer.align
            );
            if tiled {
                assert_eq!((answer.size, answer.align), (metal, 128), "{what}");
            }
            // A stencil view's image never fits a `Stencil8` texture's answer: on every device a
            // shadow of Metal's tiles holds it.
            if tiled || view {
                assert_eq!((off, read_off), (0, 0), "{what}: texels off Metal's tiles");
            }
        }
        tiled
    };
    let notices = counts(&messages.take_notices());
    let count = |kind| notices.get(&kind).copied().unwrap_or(0);
    eprintln!("HEAPTEX tiles: {notices:?}");
    if tiled {
        assert_eq!(count(Degradation::HeapTextureLosslessLayout), 1);
        assert_eq!(count(Degradation::ShadowedTexture), 1);
        assert_eq!(count(OPTIMIZED[0]), 0);
        assert_eq!(count(APART[0]), 0);
    }
    layer_reported_nothing_at_all(&messages);
}

/// A CPU write of a heap's `Depth32Float` texture's bytes made while its command buffer runs,
/// after the shadow's copy in read them and before the keep of a later command that writes
/// another texel through the texture, is kept, as Metal keeps a write of bytes no command
/// touches (`shadow.rs`, **Merged back**), and the command's write is merged. The command buffer:
/// a blit reads one texel (the copy in), into the spin's spare bytes, which orders the spin
/// ([`Busy::spin_in`]) after it; then a blit encoder whose first copy reads the word the spin
/// publishes, so the keep before its next copy, which writes texel `(3, 1)`, runs after the
/// spin. The CPU writes texel `(3, 0)` where Metal's tiles place it, in a buffer over the heap's
/// bytes (`heap_layouts::tiled_offset`), once the spin raised its ticket, and then releases the
/// spin. The CPU's bytes are kept, and a blit reads the blit's texel; where the texture is a
/// shadow of Metal's tiles, the heap's bytes hold that texel where they place it and every other
/// texel as it was (one in place lays its texels out as its driver does). The texture's image holds the copied texels between, so a keep of the heap's bytes as
/// it runs would hold the CPU's write, and the merge would undo it with the image's value; the
/// keep holds the bytes the copy in read. A `HEAPTEX keep after copy in` line prints whether the
/// write fell between: after the spin began (its ticket seen, where the device shows the host a
/// running kernel's store, or the command buffer's GPU start time 2 ms before it), and before the
/// spin ended, by its release or by its cap of 250 ms of GPU time (ANV on the Iris Xe showed the
/// host no ticket and the spin no release, and its cap ended it); elsewhere the writes are kept
/// all the same. Where the device makes the
/// texture a shadow of Metal's tiles (`WorkCounts::heap_textures_tiled`), the merge is counted
/// (`WorkCounts::shadow_merge_dispatches_planned`), and its keep not as one right after its copy
/// in (`WorkCounts::shadow_keeps_from_staging`): the first encoder's is the only copy in.
#[test]
fn a_heap_depth_texture_keeps_a_cpu_write_made_after_its_copy_in() {
    use super::first_use::{release, Busy};
    use super::heap_layouts::tiled_offset;
    use metal2vulkan_runtime::host_time;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::time::{Duration, Instant};
    // The word at `index` of a Shared buffer, read atomically while the GPU may write it.
    let word = |buffer: &Buffer, index: usize| {
        let contents = buffer.contents().expect("a Shared buffer");
        // SAFETY: a 4-aligned word inside the buffer, which the host only reads.
        let word = unsafe { AtomicU32::from_ptr(contents.as_ptr().cast::<u32>().add(index)) };
        word.load(Ordering::SeqCst)
    };
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    {
        let device = &validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let queue = device.new_command_queue().unwrap();
        let busy = Busy::new(device, &queue);
        let options = tracked(StorageMode::Shared);
        let [w, h] = [37, 5];
        let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::Depth32Float, w, h, false);
        descriptor.resource_options = options;
        descriptor.usage = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
        descriptor.allow_gpu_optimized_contents = false;
        let answer = device.heap_texture_size_and_align(&descriptor).unwrap();
        let heap = heap(device, HeapType::Placement, 1 << 20, options);
        let made = device.work_counts();
        let texture = heap.new_texture_at_offset(&descriptor, 0).unwrap();
        let tiled = device.work_counts().heap_textures_tiled > made.heap_textures_tiled;
        let over = heap.new_buffer_at_offset(answer.size, options, 0).unwrap();
        // Texel `i`'s depth names it: (i + 1) / 2^23, exact and inside [0, 1].
        let depth = |i: u64| ((i + 1) as f32 / (1u32 << 23) as f32).to_le_bytes();
        let at = |i: u64| tiled_offset(4, [w, h], i % w, i / w) as usize;
        let mut bytes = vec![0; answer.size as usize];
        for i in 0..w * h {
            bytes[at(i)..at(i) + 4].copy_from_slice(&depth(i));
        }
        let source = staged(device, &bytes);
        let command_buffer = queue.command_buffer().unwrap();
        let mut copy = command_buffer.blit_command_encoder().unwrap();
        copy.copy_from_buffer_to_buffer(&source, 0, &over, 0, answer.size)
            .unwrap();
        copy.end_encoding().unwrap();
        run(&command_buffer);
        let (cpu_texel, gpu_texel) = (3, w + 3);
        let (cpu, written) = (0.5f32.to_le_bytes(), 0.25f32.to_le_bytes());
        let value = staged(device, &written);
        let words = staged(device, &[0; 16]);
        let sink = staged(device, &[0; 4]);
        let counted = device.work_counts();
        let command_buffer = queue.command_buffer().unwrap();
        let one = Size::new(1, 1, 1);
        let mut read = command_buffer.blit_command_encoder().unwrap();
        read.copy_from_texture_to_buffer(
            &texture,
            0,
            0,
            Origin::new(0, 0, 0),
            one,
            &words,
            8,
            4,
            4,
            BlitOption::None,
        )
        .unwrap();
        read.end_encoding().unwrap();
        let ticket = busy.spin_in(device, &command_buffer, &words);
        let mut write = command_buffer.blit_command_encoder().unwrap();
        write
            .copy_from_buffer_to_buffer(&words, 4, &sink, 0, 4)
            .unwrap();
        write
            .copy_from_buffer_to_texture(
                &value,
                0,
                4,
                4,
                one,
                &texture,
                0,
                0,
                Origin::new(gpu_texel % w, gpu_texel / w, 0),
                BlitOption::None,
            )
            .unwrap();
        write.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        // The spin's ticket raised, where the device shows the host a running kernel's store;
        // else 20 ms on, which the GPU's start of the command buffer is checked against below.
        let deadline = Instant::now() + Duration::from_millis(20);
        let seen = loop {
            match word(&ticket, 0) == 2 {
                true => break true,
                false if Instant::now() >= deadline => break false,
                false => std::thread::sleep(Duration::from_micros(50)),
            }
        };
        let written_at = host_time();
        let contents = over.contents().expect("a Shared heap's buffer");
        // SAFETY: the texel's four bytes inside the buffer, which no command of the running
        // command buffer touches.
        unsafe {
            let texel = contents.as_ptr().add(at(cpu_texel));
            std::ptr::copy_nonoverlapping(cpu.as_ptr(), texel, 4);
        }
        release(&words);
        command_buffer
            .wait_until_completed_for(super::WAIT)
            .unwrap();
        assert_eq!(command_buffer.error(), None);
        // Between the copy in and the keep: after the spin started (its ticket seen, or the
        // command buffer begun on the GPU 2 ms before), and before the spin ended, by its release
        // or by its cap, 250 ms of the GPU's time at the fastest rate its calibration measured
        // (`Busy::new`), from its start, which runs after the command buffer's.
        let started = command_buffer.gpu_start_time().unwrap_or(f64::MAX);
        let begun = seen || started + 0.002 < written_at;
        let released = word(&words, 1) == 5;
        let before_cap = written_at < started + 0.2;
        let after = device.work_counts();
        let merges =
            after.shadow_merge_dispatches_planned - counted.shadow_merge_dispatches_planned;
        let staged_keeps = after.shadow_keeps_from_staging - counted.shadow_keeps_from_staging;
        eprintln!(
            "HEAPTEX keep after copy in: tiled {tiled}, the CPU's write between the copy in and \
             the keep {} (the ticket seen {seen}, the GPU began {:.1} ms before the write, the \
             spin released {released}), {merges} merge dispatches, {staged_keeps} keeps from \
             staging right after a copy in",
            begun && (released || before_cap),
            (written_at - started) * 1e3
        );
        if tiled {
            assert!(merges > 0, "the merge");
            assert_eq!(staged_keeps, 0, "a keep apart from its copy in");
        }
        let bytes = gpu_read(&queue, &over, 0..answer.size);
        let texel = |i: u64| &bytes[at(i)..at(i) + 4];
        assert_eq!(texel(cpu_texel), cpu, "the CPU's write since the copy in");
        let read = staged(device, &[0; 4]);
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        let origin = Origin::new(gpu_texel % w, gpu_texel / w, 0);
        blit.copy_from_texture_to_buffer(
            &texture,
            0,
            0,
            origin,
            one,
            &read,
            0,
            4,
            4,
            BlitOption::None,
        )
        .unwrap();
        blit.end_encoding().unwrap();
        run(&command_buffer);
        assert_eq!(
            read_buffer(&read, 0, 4).unwrap(),
            written,
            "the blit's write"
        );
        // A texture in place lays its texels out as its driver does: where Metal's tiles hold
        // them, each lies where they place it.
        if tiled {
            assert_eq!(
                texel(gpu_texel),
                written,
                "the blit's write in the heap's bytes"
            );
            let off: Vec<u64> = (0..w * h)
                .filter(|&i| i != cpu_texel && i != gpu_texel && texel(i) != depth(i))
                .collect();
            assert!(off.is_empty(), "texels no write touched: {off:?}");
        }
    }
    layer_reported_nothing_at_all(&messages);
}

/// A heap's `Depth32Float` texture whose bytes hold depths outside `[0, 1]` (1.5, -0.5, 2.0, a
/// NaN, an infinity and `0xeeeeeeee`, beside `-0.0` and depths inside) takes them into its image
/// with no copy outside that range, under the validation layer's GPU-assisted check of buffer to
/// image copies (`gpuav_buffer_copies`, which reports
/// `VUID-vkCmdCopyBufferToImage-pRegions-07931` for each such value on a device without
/// `VK_EXT_depth_range_unrestricted`). Where the device would store such a depth clamped
/// (`ImageFacts::depth_clamps`: no such extension enabled, and no
/// `DriverFact::DepthImageKeepsValuesOutsideUnitRange`), its shadow's copy in clamps each into
/// `[0, 1]`, a NaN to 1.0 and `-0.0` kept, and a blit of the texture into a buffer reads the
/// clamped values; elsewhere it reads the bytes as they are. A `-0.0` reads as the device stores
/// it, with its sign or as `0.0` (ANV's depth images keep no zero's sign). Both where the texture is a shadow of
/// Metal's tiles, as on ANV and MoltenVK; one in place (lavapipe) lays its texels out as its
/// driver does and copies nothing in. The texture is a `Shared` placement
/// heap's 37x5 one, its bytes written through a buffer over them where Metal's tiles place each
/// texel (`heap_layouts::tiled_offset`). `HEAPTEX depth range` prints whether it was a shadow of
/// Metal's tiles (`WorkCounts::heap_textures_tiled`), the route that copies it in, and the
/// layer's other messages; of those, only the layer's warnings about the settings the check
/// takes pass (`VALIDATION-SETTINGS`, `WARNING-Setting-Limit-Adjusted`).
#[test]
fn a_heap_depth_texture_s_copy_in_takes_no_depth_outside_0_1() {
    use super::heap_layouts::tiled_offset;
    use super::VALIDATION_LAYER;
    use metal2vulkan_runtime::{LayerSetting, LayerSettingValue, MessageSeverity};
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    // Texel `i`'s bits: a depth outside [0, 1] or a NaN, else one inside naming it.
    let bits = |i: u64| match i % 8 {
        0 => 1.5f32.to_bits(),
        1 => (-0.5f32).to_bits(),
        2 => 0x7fc0_0000,
        3 => f32::INFINITY.to_bits(),
        4 => ((i + 1) as f32 / (1u32 << 23) as f32).to_bits(),
        5 => (-0.0f32).to_bits(),
        6 => 2.0f32.to_bits(),
        _ => 0xeeee_eeee,
    };
    // The value the copy in leaves of `bits` where it clamps.
    let clamped = |bits: u32| -> u32 {
        let one = 1.0f32.to_bits();
        match bits {
            _ if bits & 0x7fff_ffff > 0x7f80_0000 => one,
            0x8000_0000 => bits,
            _ if bits > 0x8000_0000 => 0,
            _ if bits > one => one,
            _ => bits,
        }
    };
    let (clamps, tiled, read) = {
        let device = &validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
            let on = |name: &std::ffi::CStr| {
                let value = LayerSettingValue::Bool(true);
                LayerSetting::new(VALIDATION_LAYER.to_owned(), name.to_owned(), value)
            };
            options.layer_settings.extend([
                on(c"gpuav_enable"),
                on(c"gpuav_buffer_copies"),
                LayerSetting::new(
                    VALIDATION_LAYER.to_owned(),
                    c"gpuav_shader_instrumentation".to_owned(),
                    LayerSettingValue::Bool(false),
                ),
            ]);
        })
        .unwrap();
        let facts = &device.properties().driver_facts;
        let clamps = !device.enables_extension(ash::ext::depth_range_unrestricted::NAME)
            && !facts.contains(&DriverFact::DepthImageKeepsValuesOutsideUnitRange);
        let queue = device.new_command_queue().unwrap();
        let options = tracked(StorageMode::Shared);
        let [w, h] = [37, 5];
        let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::Depth32Float, w, h, false);
        descriptor.resource_options = options;
        descriptor.usage = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
        descriptor.allow_gpu_optimized_contents = false;
        let answer = device.heap_texture_size_and_align(&descriptor).unwrap();
        let heap = heap(device, HeapType::Placement, 1 << 20, options);
        let made = device.work_counts();
        let texture = heap.new_texture_at_offset(&descriptor, 0).unwrap();
        let tiled = device.work_counts().heap_textures_tiled > made.heap_textures_tiled;
        let over = heap.new_buffer_at_offset(answer.size, options, 0).unwrap();
        let mut bytes = vec![0; answer.size as usize];
        for i in 0..w * h {
            let at = tiled_offset(4, [w, h], i % w, i / w) as usize;
            bytes[at..at + 4].copy_from_slice(&bits(i).to_le_bytes());
        }
        let source = staged(device, &bytes);
        let command_buffer = queue.command_buffer().unwrap();
        let mut copy = command_buffer.blit_command_encoder().unwrap();
        copy.copy_from_buffer_to_buffer(&source, 0, &over, 0, answer.size)
            .unwrap();
        copy.end_encoding().unwrap();
        run(&command_buffer);
        let row = 4 * w;
        let into = staged(device, &vec![0; (row * h) as usize]);
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_texture_to_buffer(
            &texture,
            0,
            0,
            Origin::new(0, 0, 0),
            Size::new(w, h, 1),
            &into,
            0,
            row,
            row * h,
            BlitOption::None,
        )
        .unwrap();
        blit.end_encoding().unwrap();
        run(&command_buffer);
        (clamps, tiled, read_buffer(&into, 0, row * h).unwrap())
    };
    let reported = messages.take();
    let (range, other): (Vec<_>, Vec<_>) = (reported.iter())
        .partition(|message| message.id_name.contains("07931") || message.text.contains("07931"));
    eprintln!(
        "HEAPTEX depth range: clamps {clamps}, tiled {tiled}; {} copies outside [0, 1] reported, \
         and {} other messages: {other:#?}",
        range.len(),
        other.len()
    );
    assert!(
        range.is_empty(),
        "a copy of a depth outside [0, 1]: {range:#?}"
    );
    // The layer's warnings about the settings the GPU-assisted check takes aside (that it runs
    // beside the core checks, and the descriptor limits it lowers for its own), nothing else.
    let settings = ["VALIDATION-SETTINGS", "WARNING-Setting-Limit-Adjusted"];
    let unexpected: Vec<_> = (other.iter())
        .filter(|message| {
            let setting = settings.contains(&message.id_name.as_str());
            !(message.severity == MessageSeverity::Warning && setting)
        })
        .collect();
    assert!(unexpected.is_empty(), "reported: {unexpected:#?}");
    layer_reported_nothing_at_all(&messages);
    // A texture in place lays its texels out as its driver does, and copies nothing in: only a
    // shadow of Metal's tiles reads each texel where they place it.
    let off: Vec<(u64, u32, u32)> = (0..37 * 5)
        .filter(|_| tiled)
        .filter_map(|i| {
            let at = (4 * i) as usize;
            let got = u32::from_le_bytes(read[at..at + 4].try_into().unwrap());
            let want = match clamps {
                true => clamped(bits(i)),
                false => bits(i),
            };
            // A zero's sign is the device's to keep or drop.
            let zero = (want | got) & 0x7fff_ffff == 0;
            (got != want && !zero).then_some((i, bits(i), got))
        })
        .collect();
    assert!(
        off.is_empty(),
        "texels read otherwise (texel, bits, read): {off:x?}"
    );
}

/// One aspect of one slice of a multisample texture a lane moves ([`sample_round_trip`]): its
/// element bytes, its blit option, the slice, and its plane's first byte in that slice from the
/// texture's.
pub(super) type SampleAspect = (u64, BlitOption, u64, u64);

/// Each sample of `texture` (`[w, h]` pixels of `samples` samples), each of its `aspects` uploaded
/// by a blit naming it (a depth of `(i + 1) / 2^23` for sample `i`, exact; a 16-bit unorm of `i +
/// 1`; a stencil byte of `i + 1`; a colour element of `i + 1`'s low bytes), found in the heap's
/// bytes, which `read` reads from the texture's first, where Metal's tiles place it: the plane's
/// first byte, then pixel `(x, y)`'s element of all its samples (`tiled_offset` of `samples · B`
/// bytes), sample `s` at `s · B`, as the Metal probes on the M3 Max under macOS 27.0.1 found every
/// sample (`texture/heap_tiled.rs`); and bytes `write` writes there, naming each sample backwards,
/// read back by a blit as those samples. Metal's blits lay a pixel's samples one after another in
/// the buffer. The samples off Metal's tiles each way; `None` where the runtime refuses such a
/// blit (`NotImplemented(MultisampledBufferCopy)`: a 16-bit depth on a device whose multisampled
/// copies go by draws).
pub(super) fn sample_round_trip(
    queue: &CommandQueue,
    texture: &Texture,
    ([w, h], samples): ([u64; 2], u64),
    aspects: &[SampleAspect],
    read: &dyn Fn() -> Vec<u8>,
    write: &dyn Fn(&[u8]),
) -> Option<(usize, usize)> {
    use super::heap_layouts::tiled_offset;
    let device = texture.device();
    let count = w * h * samples;
    let (mut off, mut read_off) = (0, 0);
    for &(unit, option, slice, first) in aspects {
        let element = |i: u64| -> Vec<u8> {
            match (unit, option) {
                (4, BlitOption::None | BlitOption::DepthFromDepthStencil)
                    if is_depth_stencil(texture.pixel_format()) =>
                {
                    ((i + 1) as f32 / (1u32 << 23) as f32)
                        .to_le_bytes()
                        .to_vec()
                }
                _ => (i + 1)
                    .to_le_bytes()
                    .iter()
                    .copied()
                    .cycle()
                    .take(unit as usize)
                    .collect(),
            }
        };
        let named: Vec<u8> = (0..count).flat_map(element).collect();
        let row = w * samples * unit;
        let blit = |into: bool, buffer: &Buffer| {
            let command_buffer = queue.command_buffer().unwrap();
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            let (origin, size) = (Origin::new(0, 0, 0), Size::new(w, h, 1));
            let copied = match into {
                true => blit.copy_from_buffer_to_texture(
                    buffer,
                    0,
                    row,
                    row * h,
                    size,
                    texture,
                    slice,
                    0,
                    origin,
                    option,
                ),
                false => blit.copy_from_texture_to_buffer(
                    texture,
                    slice,
                    0,
                    origin,
                    size,
                    buffer,
                    0,
                    row,
                    row * h,
                    option,
                ),
            };
            blit.end_encoding().unwrap();
            run(&command_buffer);
            match copied {
                Ok(()) => true,
                Err(Error::NotImplemented(RuntimeGap::MultisampledBufferCopy)) => false,
                Err(error) => panic!("a multisampled blit: {error:?}"),
            }
        };
        if !blit(true, &staged(device, &named)) {
            return None;
        }
        let bytes = read();
        // Sample `i` is sample `i % samples` of pixel `i / samples`, in rows of pixels.
        let at = |i: u64| {
            let (pixel, s) = (i / samples, i % samples);
            let (x, y) = (pixel % w, pixel / w);
            (first + tiled_offset(samples * unit, [w, h], x, y) + s * unit) as usize
        };
        off += (0..count)
            .filter(|&i| bytes.get(at(i)..at(i) + unit as usize) != Some(&element(i)[..]))
            .count();
        let mut backwards = bytes;
        for i in 0..count {
            let at = at(i);
            if let Some(bytes) = backwards.get_mut(at..at + unit as usize) {
                bytes.copy_from_slice(&element(count - 1 - i));
            }
        }
        write(&backwards);
        let read_back = staged(device, &vec![0; (row * h) as usize]);
        if !blit(false, &read_back) {
            return None;
        }
        let read_back = read_buffer(&read_back, 0, row * h).unwrap();
        read_off += (0..count)
            .filter(|&i| {
                let at = (i * unit) as usize;
                read_back[at..at + unit as usize] != element(count - 1 - i)[..]
            })
            .count();
    }
    Some((off, read_off))
}

/// A heap's multisample depth, stencil and colour textures keep their samples where Metal's tiles
/// place them in a `Shared` placement heap's bytes ([`sample_round_trip`]) where the heap's memory
/// takes no image of them: a shadow of their samples in Metal's tiles there, counted
/// (`WorkCounts::heap_textures_tiled`), at Metal's answer (MoltenVK: no optimal depth or stencil
/// image in host-visible memory), a multisample array's slices each its chain from the one before
/// (an Apple M3 Max under macOS 27.0.1: every sample of 2- and 3-slice `Depth32Float` and
/// `Stencil8` arrays drawn where the rule names). Every texture counted so, on every device, lies
/// so both ways; on a driver that lays depth and stencil images apart or keeps them GPU-optimized,
/// each uncompressed one of a 32-bit float depth or a stencil is counted, and a 64 x 64
/// `Depth32Float` one with `allowGPUOptimizedContents` on (which Metal keeps compressed) holds its
/// uncompressed tiles there, announced `heap-texture-lossless-layout` once. A `Depth16Unorm` one,
/// whose samples the draws there do not move, and a colour one may lie otherwise, and a shape at a
/// sample count the device takes not (lavapipe: 2 for depth) is left out: `HEAPTEX samples` lines
/// print which.
#[test]
fn heap_multisample_textures_keep_their_samples_in_metal_s_tiles() {
    use BlitOption::{DepthFromDepthStencil as DEPTH, None as WHOLE, StencilFromDepthStencil};
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    let tiled = {
        let device = &validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let facts = &device.properties().driver_facts;
        let tiled = facts.contains(&DriverFact::DepthStencilImagesLieApart)
            || facts.contains(&DriverFact::DepthStencilImagesStayGpuOptimized);
        // A device that may store a depth outside `[0, 1]` copied into a depth image clamped.
        let clamps = !device.enables_extension(ash::ext::depth_range_unrestricted::NAME)
            && !facts.contains(&DriverFact::DepthImageKeepsValuesOutsideUnitRange);
        let queue = device.new_command_queue().unwrap();
        let options = tracked(StorageMode::Shared);
        let ds = PixelFormat::Depth32FloatStencil8;
        // Each shape's format, extent, samples, slices, `allowGPUOptimizedContents` and Metal's
        // size and alignment, measured, and its aspects of each slice: a slice's chain lies at
        // its chain's bytes from the one before, at multiples of 16 KiB from a chain of 16 KiB on.
        type Shape = (
            PixelFormat,
            [u64; 2],
            u64,
            u64,
            bool,
            (u64, u64),
            Vec<SampleAspect>,
        );
        let one = |unit, option| vec![(unit, option, 0, 0)];
        let slices = |unit, option, count, stride| {
            (0..count)
                .map(|slice| (unit, option, slice, slice * stride))
                .collect()
        };
        let (d32, d16, s8) = (
            PixelFormat::Depth32Float,
            PixelFormat::Depth16Unorm,
            PixelFormat::Stencil8,
        );
        let shapes: [Shape; 12] = [
            (d32, [37, 5], 4, 1, false, (0x2000, 128), one(4, WHOLE)),
            (d32, [100, 70], 2, 1, false, (0x18000, 128), one(4, WHOLE)),
            (d32, [64, 64], 4, 1, true, (0x10200, 128), one(4, WHOLE)),
            (
                d32,
                [64, 64],
                2,
                2,
                false,
                (0x10000, 0x4000),
                slices(4, WHOLE, 2, 0x8000),
            ),
            (
                d32,
                [37, 5],
                4,
                3,
                false,
                (0x6000, 128),
                slices(4, WHOLE, 3, 0x2000),
            ),
            (d16, [64, 64], 4, 1, false, (0x8000, 128), one(2, WHOLE)),
            (s8, [37, 5], 4, 1, false, (0x800, 128), one(1, WHOLE)),
            (s8, [100, 70], 2, 1, false, (0x8000, 128), one(1, WHOLE)),
            (
                s8,
                [100, 70],
                4,
                2,
                false,
                (0x20000, 0x4000),
                slices(1, WHOLE, 2, 0x10000),
            ),
            (
                ds,
                [64, 64],
                4,
                1,
                false,
                (0x14000, 128),
                vec![(4, DEPTH, 0, 0), (1, StencilFromDepthStencil, 0, 0x10000)],
            ),
            (
                PixelFormat::RGBA8Unorm,
                [33, 33],
                4,
                1,
                false,
                (0x10000, 128),
                one(4, WHOLE),
            ),
            (
                PixelFormat::R16Float,
                [37, 5],
                2,
                1,
                false,
                (0x800, 128),
                one(2, WHOLE),
            ),
        ];
        for (format, [w, h], samples, layers, allow, metal, aspects) in shapes {
            let what = format!("{format:?} {w}x{h} x{samples} {layers} slices allow {allow}");
            let mut descriptor = TextureDescriptor::texture_2d(format, w, h, false);
            descriptor.texture_type = match layers {
                1 => TextureType::Type2DMultisample,
                _ => TextureType::Type2DMultisampleArray,
            };
            descriptor.array_length = layers;
            descriptor.sample_count = samples;
            descriptor.resource_options = options;
            descriptor.usage = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
            descriptor.allow_gpu_optimized_contents = allow;
            let answer = match device.heap_texture_size_and_align(&descriptor) {
                Err(Error::Unsupported(u)) if u.feature == Feature::TextureSampleCount => {
                    eprintln!("HEAPTEX samples {what}: the device takes no such sample count");
                    continue;
                }
                answer => answer.unwrap(),
            };
            let heap = heap(device, HeapType::Placement, 1 << 20, options);
            let before = device.work_counts().heap_textures_tiled;
            let texture = heap.new_texture_at_offset(&descriptor, 0).unwrap();
            let counted = device.work_counts().heap_textures_tiled - before == 1;
            let float_depth = format == d32 || format == ds;
            if counted && float_depth {
                // Its shadow's copies in hand its image whatever the heap's bytes hold.
                let upload = announced(&messages, Degradation::RestrictedDepthUpload);
                assert_eq!(
                    upload, clamps,
                    "{what}: restricted-depth-upload as it is made"
                );
            }
            let over = heap.new_buffer_at_offset(answer.size, options, 0).unwrap();
            gpu_fill(&queue, &over, 0..answer.size, 0xee);
            let read = || gpu_read(&queue, &over, 0..answer.size);
            let write = |bytes: &[u8]| {
                let source = staged(device, bytes);
                let command_buffer = queue.command_buffer().unwrap();
                let mut copy = command_buffer.blit_command_encoder().unwrap();
                copy.copy_from_buffer_to_buffer(&source, 0, &over, 0, answer.size)
                    .unwrap();
                copy.end_encoding().unwrap();
                run(&command_buffer);
            };
            let shape = ([w, h], samples);
            let staged = device.work_counts().shadow_keeps_from_staging;
            let Some((off, read_off)) =
                sample_round_trip(&queue, &texture, shape, &aspects, &read, &write)
            else {
                eprintln!("HEAPTEX samples {what}: counted tiled {counted}; its blits refused");
                assert!(!counted, "{what}: tiled, but its blits refused");
                continue;
            };
            let staged = device.work_counts().shadow_keeps_from_staging - staged;
            let total = w * h * samples * aspects.len() as u64;
            eprintln!(
                "HEAPTEX samples {what}: size {} align {}, Metal's {metal:?}; counted tiled \
                 {counted}; {off} of {total} samples off Metal's tiles, {read_off} read otherwise \
                 from them; {staged} keeps from staging",
                answer.size, answer.align
            );
            if counted {
                assert_eq!((answer.size, answer.align), metal, "{what}");
                assert_eq!((off, read_off), (0, 0), "{what}: samples off Metal's tiles");
                // A blit into one slice or aspect of a depth leaves texels its merge keeps: the
                // heap's bytes its copy in took, not its image's, which the copy in may change.
                if float_depth && aspects.len() > 1 {
                    assert!(staged > 0, "{what}: a depth's keep of its image");
                }
            }
            let moved = matches!(
                format,
                PixelFormat::Depth32Float
                    | PixelFormat::Stencil8
                    | PixelFormat::Depth32FloatStencil8
            );
            if tiled && moved {
                assert_eq!((answer.size, answer.align), metal, "{what}");
                assert_eq!((off, read_off), (0, 0), "{what}: samples off Metal's tiles");
                assert_eq!(counted, !allow, "{what}: counted tiled");
            }
        }
        tiled
    };
    let notices = counts(&messages.take_notices());
    let count = |kind| notices.get(&kind).copied().unwrap_or(0);
    eprintln!("HEAPTEX samples: {notices:?}");
    if tiled {
        assert_eq!(count(Degradation::HeapTextureLosslessLayout), 1);
        assert_eq!(count(APART[0]), 0);
    }
    layer_reported_nothing_at_all(&messages);
}

/// Whether `messages` took or holds a notice of `kind`, left pending.
fn announced(messages: &Messages, kind: Degradation) -> bool {
    let pending = messages.take_notices();
    messages.restore_notices(pending.clone());
    messages.taken_kinds().contains(&kind)
        || (pending.iter())
            .filter_map(|message| message.degradation.as_ref())
            .any(|notice| notice.degradation == kind)
}

/// A heap's colour textures that Metal keeps uncompressed, in a `Private` placement heap over the
/// device's memory, lie where Metal's tiles place each texel (`heap_layouts::tiled_offset`) on a
/// driver whose colour images lie as Metal's heap textures
/// (`DriverFact::ColorImagesLieAsMetalHeapTextures`: MoltenVK 1.4.2 on the M3 Max), and the device
/// announces no heap layout there; elsewhere the first announces `heap-texture-untiled-layout`
/// and each is counted, but one no image bound in the heap's memory serves, a shadow of Metal's
/// tiles there (`shadowed-texture`, announced: the NVIDIA RTX 5080 Laptop's). On every device a
/// texture counted for no layout lies in Metal's tiles.
/// Kept uncompressed by `ShaderRead | ShaderWrite` usage: `RGBA8Unorm` of 64 x 64 and 37 x 5,
/// `R8Unorm` of 100 x 70 and `RG16Float` of 300 x 200; by `allowGPUOptimizedContents` off:
/// `RGBA8Unorm` of 64 x 64 of `ShaderRead` usage and of `ShaderRead | RenderTarget`, and of
/// `ShaderRead` usage `ARGB8Unorm_sRGB`, `BGRA8Unorm_sRGB` and `RGBA8Unorm_sRGB` of 64 x 64, a
/// 2-slice `ARGB8Unorm_sRGB` array of 64 x 64 and `RGBA16Float` of 4 x 4 (a consumer's report: on
/// MoltenVK the `ARGB8Unorm_sRGB` texture's optimal image takes 16512 bytes, past Metal's 16384,
/// where its linear image fits but lays its rows out otherwise than Metal's tiles); by a side
/// shorter than 16: `RGBA8Unorm` of 8 x 8 and `RGBA16Float` of 1 x 1 of `ShaderRead` usage. Each
/// slice is written by a blit naming each texel and found through a buffer over the heap, then
/// the heap's bytes are written there through the buffer, naming each texel otherwise, and read
/// back through a blit as those texels; `HEAPTEX device-tiles` lines print the texels off the
/// tiles each way.
#[test]
fn device_heap_textures_lie_in_metal_s_tiles() {
    use super::heap_layouts::tiled_offset;
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    let untiled = Degradation::HeapTextureUntiledLayout;
    let (as_metal, counted) = {
        let device = &validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let as_metal = device
            .properties()
            .driver_facts
            .contains(&DriverFact::ColorImagesLieAsMetalHeapTextures);
        let before = device.degraded_objects(untiled).unwrap();
        let queue = device.new_command_queue().unwrap();
        let options = tracked(StorageMode::Private);
        let (read, written) = (TextureUsage::SHADER_READ, TextureUsage::SHADER_WRITE);
        let target = read | TextureUsage::RENDER_TARGET;
        let (rgba8, srgb) = (PixelFormat::RGBA8Unorm, PixelFormat::ARGB8UnormSRGB);
        let shapes = [
            (rgba8, 4, [64, 64], 1, read | written, true),
            (rgba8, 4, [37, 5], 1, read | written, true),
            (PixelFormat::R8Unorm, 1, [100, 70], 1, read | written, true),
            (
                PixelFormat::RG16Float,
                4,
                [300, 200],
                1,
                read | written,
                true,
            ),
            (rgba8, 4, [64, 64], 1, read, false),
            (rgba8, 4, [64, 64], 1, target, false),
            (rgba8, 4, [8, 8], 1, read, true),
            (srgb, 4, [64, 64], 1, read, false),
            (PixelFormat::BGRA8UnormSRGB, 4, [64, 64], 1, read, false),
            (PixelFormat::RGBA8UnormSRGB, 4, [64, 64], 1, read, false),
            (srgb, 4, [64, 64], 2, read, false),
            (PixelFormat::RGBA16Float, 8, [4, 4], 1, read, false),
            (PixelFormat::RGBA16Float, 8, [1, 1], 1, read, true),
        ];
        let mut kept = Vec::new();
        for (format, unit, [w, h], slices, usage, optimized) in shapes {
            let what = format!("{format:?} {w}x{h} x{slices} {usage:?} optimized {optimized}");
            let mut descriptor = TextureDescriptor::texture_2d(format, w, h, false);
            if slices > 1 {
                descriptor.texture_type = TextureType::Type2DArray;
                descriptor.array_length = slices;
            }
            descriptor.resource_options = options;
            descriptor.usage = usage;
            descriptor.allow_gpu_optimized_contents = optimized;
            let answer = device.heap_texture_size_and_align(&descriptor).unwrap();
            // Each slice's chain lies at its stride from the one before.
            let stride = answer.size / slices;
            let heap = heap(device, HeapType::Placement, 1 << 20, options);
            let made_before = device.degraded_objects(untiled).unwrap();
            let texture = heap.new_texture_at_offset(&descriptor, 0).unwrap();
            let announced = device.degraded_objects(untiled).unwrap() > made_before;
            let over = heap.new_buffer_at_offset(answer.size, options, 0).unwrap();
            gpu_fill(&queue, &over, 0..answer.size, 0);
            let count = w * h;
            // Texel `i` of `slice`'s element, naming both: `unit` bytes of a number, the second
            // way's counted down from the first's.
            let element = |slice: u64, i: u64, back: bool| {
                let n = slice * count + i + 1;
                let n = if back { 0xffff_ffff_ffff - n } else { n };
                n.to_le_bytes()[..unit as usize].to_vec()
            };
            let at = |slice: u64, i: u64| {
                (slice * stride + tiled_offset(unit, [w, h], i % w, i / w)) as usize
            };
            let row = w * unit;
            let size = Size::new(w, h, 1);
            let origin = Origin::new(0, 0, 0);
            let (mut off, mut read_off) = (0, 0);
            for slice in 0..slices {
                let named: Vec<u8> = (0..count).flat_map(|i| element(slice, i, false)).collect();
                let source = staged(device, &named);
                let command_buffer = queue.command_buffer().unwrap();
                let mut blit = command_buffer.blit_command_encoder().unwrap();
                blit.copy_from_buffer_to_texture(
                    &source,
                    0,
                    row,
                    row * h,
                    size,
                    &texture,
                    slice,
                    0,
                    origin,
                    BlitOption::None,
                )
                .unwrap();
                blit.end_encoding().unwrap();
                run(&command_buffer);
            }
            let mut bytes = gpu_read(&queue, &over, 0..answer.size);
            for slice in 0..slices {
                off += (0..count)
                    .filter(|&i| {
                        let at = at(slice, i);
                        bytes.get(at..at + unit as usize) != Some(&element(slice, i, false)[..])
                    })
                    .count();
            }
            // The other way: the heap's bytes written through the buffer, read through the
            // texture.
            for slice in 0..slices {
                for i in 0..count {
                    let at = at(slice, i);
                    if let Some(bytes) = bytes.get_mut(at..at + unit as usize) {
                        bytes.copy_from_slice(&element(slice, i, true));
                    }
                }
            }
            let source = staged(device, &bytes);
            let command_buffer = queue.command_buffer().unwrap();
            let mut copy = command_buffer.blit_command_encoder().unwrap();
            copy.copy_from_buffer_to_buffer(&source, 0, &over, 0, answer.size)
                .unwrap();
            copy.end_encoding().unwrap();
            run(&command_buffer);
            for slice in 0..slices {
                let back = staged(device, &vec![0; (row * h) as usize]);
                let command_buffer = queue.command_buffer().unwrap();
                let mut blit = command_buffer.blit_command_encoder().unwrap();
                blit.copy_from_texture_to_buffer(
                    &texture,
                    slice,
                    0,
                    origin,
                    size,
                    &back,
                    0,
                    row,
                    row * h,
                    BlitOption::None,
                )
                .unwrap();
                blit.end_encoding().unwrap();
                run(&command_buffer);
                let back = read_buffer(&back, 0, row * h).unwrap();
                read_off += (0..count)
                    .filter(|&i| {
                        let at = (i * unit) as usize;
                        back[at..at + unit as usize] != element(slice, i, true)[..]
                    })
                    .count();
            }
            let total = count * slices;
            eprintln!(
                "HEAPTEX device-tiles {what}: size {}, of {total} texels {off} off Metal's tiles, \
                 {read_off} read otherwise from them (fact {as_metal}, announced {announced})",
                answer.size
            );
            if as_metal {
                assert!(!announced, "{what}: counted untiled");
            }
            if !announced {
                assert_eq!((off, read_off), (0, 0), "{what}: texels off Metal's tiles");
            }
            kept.push((texture, over, heap));
        }
        let counted = device.degraded_objects(untiled).unwrap() - before;
        drop(kept);
        (as_metal, counted)
    };
    // The textures not counted on a driver without the fact are shadows of Metal's tiles.
    let tiled = 13 - counted;
    let notices = counts(&messages.take_notices());
    let layouts: Vec<_> = (notices.iter())
        .filter(|(kind, _)| HEAP_LAYOUTS.contains(kind))
        .collect();
    eprintln!("HEAPTEX device-tiles: {layouts:?}, {counted} counted, {tiled} not");
    match as_metal {
        true => assert_eq!((layouts.len(), counted), (0, 0)),
        false => {
            let first = (counted > 0).then_some((&untiled, &1));
            assert_eq!(layouts, first.into_iter().collect::<Vec<_>>());
            let shadowed = notices.contains_key(&TILE_SHADOW[0]);
            assert!(
                shadowed || tiled == 0,
                "shadows of Metal's tiles: {notices:?}"
            );
        }
    }
    layer_reported_nothing_at_all(&messages);
}

/// One aspect of a `side`-square depth or stencil texture, read back through a blit with
/// `option`: `texel` bytes each.
fn aspect_texels(
    queue: &CommandQueue,
    texture: &Texture,
    side: u64,
    texel: u64,
    option: BlitOption,
) -> Vec<u8> {
    let bytes = side * side * texel;
    let copy = queue
        .device()
        .new_buffer(bytes, ResourceOptions::shared())
        .unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_texture_to_buffer(
        texture,
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(side, side, 1),
        &copy,
        0,
        side * texel,
        bytes,
        option,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    run(&command_buffer);
    // SAFETY: the buffer holds `bytes` bytes the completed copy wrote.
    unsafe { std::slice::from_raw_parts(copy.contents().unwrap().as_ptr(), bytes as usize) }
        .to_vec()
}

/// A heap's depth and stencil textures a render pass clears and stores, in a tracked `Private`
/// placement heap: a `Depth32Float`, a `Stencil8` and a `Depth32Float_Stencil8` texture, each
/// under a buffer over its answered bytes filled with zeros, cleared by a pass (depth 0.25,
/// stencil 0x5a) that stores them, then the buffer filled with 0x3e. Where the device's driver
/// lays depth and stencil images apart from their memory (`DriverFact::DepthStencilImagesLieApart`:
/// MoltenVK), each texture is a shadow of the heap's bytes, in Metal's tiles: the pass's clear
/// lands in the depth plane, from the texture's first byte, and in the stencil plane, from the
/// texture's first byte or, for `Depth32Float_Stencil8`, from the `Depth32Float` answer rounded up
/// to 16 KiB, and the fill changes every texel of each aspect read back by a blit (D32_S8's two
/// regions). Elsewhere the texture aliases the heap's memory in the driver's layout,
/// which Metal's answer for a pass's clear is not measured against, so `HEAPTEX attach` lines print
/// the counts only.
#[test]
fn a_heap_depth_attachment_s_clear_reaches_the_heap_s_bytes() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let options = tracked(StorageMode::Private);
        let shadows = device
            .properties()
            .driver_facts
            .contains(&DriverFact::DepthStencilImagesLieApart);
        let side = 16u64;
        let texels = (side * side) as usize;
        let formats = [
            PixelFormat::Depth32Float,
            PixelFormat::Stencil8,
            PixelFormat::Depth32FloatStencil8,
        ];
        for format in formats {
            let mut descriptor = TextureDescriptor::texture_2d(format, side, side, false);
            descriptor.resource_options = options;
            descriptor.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
            let what = format!("{format:?}");
            let size = match device.heap_texture_size_and_align(&descriptor) {
                Ok(answer) => answer.size,
                Err(error) => {
                    eprintln!("HEAPTEX attach {what}: refused {error:?}");
                    continue;
                }
            };
            let heap = heap(device, HeapType::Placement, 1 << 20, options);
            let texture = heap.new_texture_at_offset(&descriptor, 0).unwrap();
            let buffer = heap.new_buffer_at_offset(size, options, 0).unwrap();
            gpu_fill_zero(&queue, &buffer, size);
            let depth = format != PixelFormat::Stencil8;
            let stencil = format != PixelFormat::Depth32Float;
            let mut pass = RenderPassDescriptor::new();
            if depth {
                let attachment = &mut pass.depth_attachment;
                attachment.texture = Some(texture.clone());
                attachment.load_action = LoadAction::Clear;
                attachment.store_action = StoreAction::Store;
                attachment.clear_depth = 0.25;
            }
            if stencil {
                let attachment = &mut pass.stencil_attachment;
                attachment.texture = Some(texture.clone());
                attachment.load_action = LoadAction::Clear;
                attachment.store_action = StoreAction::Store;
                attachment.clear_stencil = 0x5a;
            }
            let command_buffer = queue.command_buffer().unwrap();
            command_buffer
                .render_command_encoder(&pass)
                .unwrap()
                .end_encoding()
                .unwrap();
            run(&command_buffer);
            let bytes = gpu_read(&queue, &buffer, 0..size);
            let changed = bytes.iter().filter(|&&byte| byte != 0).count();
            // The aspects' options of a combined texture.
            let (depth_option, stencil_option) = if depth && stencil {
                (
                    BlitOption::DepthFromDepthStencil,
                    BlitOption::StencilFromDepthStencil,
                )
            } else {
                (BlitOption::None, BlitOption::None)
            };
            gpu_fill(&queue, &buffer, 0..size, 0x3e);
            let kept_depth = if depth {
                let read = aspect_texels(&queue, &texture, side, 4, depth_option);
                read.as_chunks::<4>()
                    .0
                    .iter()
                    .filter(|texel| **texel == 0.25f32.to_le_bytes())
                    .count()
            } else {
                0
            };
            let kept_stencil = if stencil {
                let read = aspect_texels(&queue, &texture, side, 1, stencil_option);
                read.iter().filter(|&&texel| texel == 0x5a).count()
            } else {
                0
            };
            eprintln!(
                "HEAPTEX attach {what}: size {size}; the clear changed {changed} bytes; after the \
                 fill {kept_depth} depth and {kept_stencil} stencil texels kept the clear"
            );
            if shadows {
                let mut at = 0;
                if depth {
                    let plane = &bytes[..texels * 4];
                    assert!(
                        plane
                            .as_chunks::<4>()
                            .0
                            .iter()
                            .all(|t| *t == 0.25f32.to_le_bytes()),
                        "{what}: the depth plane does not hold the clear"
                    );
                    at = texels * 4;
                }
                if stencil {
                    // The stencil plane past the depth plane's answer, rounded up to 16 KiB.
                    if depth {
                        let mut plane = descriptor;
                        plane.pixel_format = PixelFormat::Depth32Float;
                        let answer = device.heap_texture_size_and_align(&plane).unwrap();
                        at = answer.size.next_multiple_of(16 << 10) as usize;
                    }
                    assert!(
                        bytes[at..at + texels].iter().all(|&t| t == 0x5a),
                        "{what}: the stencil plane does not hold the clear"
                    );
                }
                assert_eq!(
                    (kept_depth, kept_stencil),
                    (0, 0),
                    "{what}: texels the fill covered"
                );
            }
        }
    });
}

/// A heap's 4-sample 16x16 `Depth32Float` and `Depth32Float_Stencil8` textures a render pass
/// clears, loads, stores and resolves, in a tracked `Shared` placement heap. Where the heap's memory
/// takes no image of them, each is a shadow of its samples in Metal's tiles, counted tiled
/// (MoltenVK): a pass that clears them (depth 0.25, stencil 0x5a) and stores puts the clear in
/// every sample's bytes of each plane (the depth plane from the texture's first byte, the stencil
/// plane past the depth plane's answer rounded up to 16 KiB); bytes then written there (depth 0.5,
/// stencil 0x33) a pass that loads the depth and resolves it (sample 0, Metal's default) reads as
/// written; and a pass that loads and stores both, drawing nothing, leaves every byte of the
/// texture's. Every such texture is so where the driver lays depth and stencil images apart from
/// their memory or keeps them GPU-optimized. Elsewhere `HEAPTEX ms-attach` lines print what the
/// passes left.
#[test]
fn a_heap_multisample_depth_attachment_s_passes_reach_the_heap_s_bytes() {
    on_device(|device| {
        if device.properties().supports_texture_sample_count(4) != Ok(true) {
            eprintln!("HEAPTEX ms-attach: the device takes no 4-sample texture");
            return;
        }
        let queue = device.new_command_queue().unwrap();
        let options = tracked(StorageMode::Shared);
        let side = 16u64;
        let samples = (side * side * 4) as usize;
        for format in [PixelFormat::Depth32Float, PixelFormat::Depth32FloatStencil8] {
            let what = format!("{format:?}");
            let mut descriptor = TextureDescriptor::texture_2d(format, side, side, false);
            descriptor.texture_type = TextureType::Type2DMultisample;
            descriptor.sample_count = 4;
            descriptor.resource_options = options;
            descriptor.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
            descriptor.allow_gpu_optimized_contents = false;
            let size = device
                .heap_texture_size_and_align(&descriptor)
                .unwrap()
                .size;
            let heap = heap(device, HeapType::Placement, 1 << 20, options);
            let before = device.work_counts().heap_textures_tiled;
            let texture = heap.new_texture_at_offset(&descriptor, 0).unwrap();
            let tiled = device.work_counts().heap_textures_tiled - before == 1;
            let buffer = heap.new_buffer_at_offset(size, options, 0).unwrap();
            gpu_fill_zero(&queue, &buffer, size);
            let stencil = format == PixelFormat::Depth32FloatStencil8;
            // The stencil plane past the depth plane's answer, rounded up to 16 KiB.
            let stencil_at = (samples * 4).next_multiple_of(16 << 10);
            let pass = |load: LoadAction, store: StoreAction, resolve: Option<&Texture>| {
                let mut pass = RenderPassDescriptor::new();
                let attachment = &mut pass.depth_attachment;
                attachment.texture = Some(texture.clone());
                attachment.load_action = load;
                attachment.store_action = store;
                attachment.clear_depth = 0.25;
                attachment.resolve_texture = resolve.cloned();
                if stencil {
                    let attachment = &mut pass.stencil_attachment;
                    attachment.texture = Some(texture.clone());
                    attachment.load_action = load;
                    attachment.store_action = match resolve {
                        Some(_) => StoreAction::DontCare,
                        None => store,
                    };
                    attachment.clear_stencil = 0x5a;
                }
                let command_buffer = queue.command_buffer().unwrap();
                command_buffer
                    .render_command_encoder(&pass)
                    .unwrap()
                    .end_encoding()
                    .unwrap();
                run(&command_buffer);
            };
            pass(LoadAction::Clear, StoreAction::Store, None);
            let bytes = gpu_read(&queue, &buffer, 0..size);
            let cleared_depth = bytes[..samples * 4]
                .as_chunks::<4>()
                .0
                .iter()
                .filter(|sample| **sample == 0.25f32.to_le_bytes())
                .count();
            let cleared_stencil = match stencil {
                true => bytes[stencil_at..stencil_at + samples]
                    .iter()
                    .filter(|&&sample| sample == 0x5a)
                    .count(),
                false => 0,
            };
            // Each sample written through the buffer, then resolved.
            let mut written = bytes;
            for sample in written[..samples * 4].as_chunks_mut::<4>().0 {
                *sample = 0.5f32.to_le_bytes();
            }
            if stencil {
                written[stencil_at..stencil_at + samples].fill(0x33);
            }
            let source = staged(device, &written);
            let command_buffer = queue.command_buffer().unwrap();
            let mut copy = command_buffer.blit_command_encoder().unwrap();
            copy.copy_from_buffer_to_buffer(&source, 0, &buffer, 0, size)
                .unwrap();
            copy.end_encoding().unwrap();
            run(&command_buffer);
            let mut single = TextureDescriptor::texture_2d(format, side, side, false);
            single.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
            single.resource_options = ResourceOptions::private();
            let resolved = device.new_texture(&single).unwrap();
            pass(
                LoadAction::Load,
                StoreAction::MultisampleResolve,
                Some(&resolved),
            );
            let option = match stencil {
                true => BlitOption::DepthFromDepthStencil,
                false => BlitOption::None,
            };
            let read = aspect_texels(&queue, &resolved, side, 4, option);
            let resolved_written = read
                .as_chunks::<4>()
                .0
                .iter()
                .filter(|texel| **texel == 0.5f32.to_le_bytes())
                .count();
            // A pass that loads and stores, drawing nothing, leaves every byte.
            let before = gpu_read(&queue, &buffer, 0..size);
            pass(LoadAction::Load, StoreAction::Store, None);
            let after = gpu_read(&queue, &buffer, 0..size);
            let changed = (before.iter().zip(&after))
                .filter(|(before, after)| before != after)
                .count();
            eprintln!(
                "HEAPTEX ms-attach {what}: tiled {tiled}; the clear in {cleared_depth} depth and \
                 {cleared_stencil} stencil samples of {samples}; {resolved_written} resolved \
                 texels read the bytes written; a load and store changed {changed} bytes"
            );
            if tiled {
                assert_eq!(cleared_depth, samples, "{what}: the depth plane's clear");
                if stencil {
                    assert_eq!(
                        cleared_stencil, samples,
                        "{what}: the stencil plane's clear"
                    );
                }
                assert_eq!(resolved_written, side as usize * side as usize, "{what}");
                assert_eq!(changed, 0, "{what}: a load and store changed bytes");
            }
            // A driver that lays depth images apart or keeps them GPU-optimized holds none in
            // the heap's memory.
            let facts = &device.properties().driver_facts;
            if facts.contains(&DriverFact::DepthStencilImagesLieApart)
                || facts.contains(&DriverFact::DepthStencilImagesStayGpuOptimized)
            {
                assert!(
                    tiled,
                    "{what}: not tiled where the driver lays depth images apart"
                );
            }
        }
    });
}

/// A render pass that executes indirect render commands keeps its heap depth attachment. A
/// `Private` placement heap's 16x16 `Depth32Float` texture is the depth attachment, cleared to 0.25
/// and stored, of a pass whose one command draws the owned `icb_render_replay_template` fixture's
/// vertex function (rasterization disabled: vertex `vid` writes `params[0] * (vid + 1) + params[1]`
/// to word `vid` of `out`), with the heap resident. The pass runs on every device and its command
/// draws; where the texture is a shadow of the heap's bytes (MoltenVK), its copies go before the
/// pass and as the encoder ends, and the clear reaches the heap's bytes, as for a pass of the
/// encoder's own draws. The runtime once refused such a pass there.
#[test]
fn a_pass_executing_render_commands_keeps_its_heap_depth_attachment() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let options = tracked(StorageMode::Private);
        let shadows = device
            .properties()
            .driver_facts
            .contains(&DriverFact::DepthStencilImagesLieApart);
        let side = 16u64;
        let texels = (side * side) as usize;
        let mut descriptor =
            TextureDescriptor::texture_2d(PixelFormat::Depth32Float, side, side, false);
        descriptor.resource_options = options;
        descriptor.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
        let size = match device.heap_texture_size_and_align(&descriptor) {
            Ok(answer) => answer.size,
            Err(error) => {
                eprintln!("HEAPTEX icb: refused {error:?}");
                return;
            }
        };
        let heap = heap(device, HeapType::Placement, 1 << 20, options);
        let texture = heap.new_texture_at_offset(&descriptor, 0).unwrap();
        let bytes = heap.new_buffer_at_offset(size, options, 0).unwrap();
        gpu_fill_zero(&queue, &bytes, size);
        let bitcode = llvm_assemble(&engine_dir().join("icb_render_replay_template.ll")).unwrap();
        let library = device.new_library_with_air(&bitcode).unwrap();
        let function = library
            .new_function_with_name("icb_render_replay_template")
            .unwrap();
        let mut pipeline =
            RenderPipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
        pipeline.rasterization_enabled = false;
        pipeline.support_indirect_command_buffers = true;
        pipeline.depth_format = Some(PixelFormat::Depth32Float);
        let state = device
            .new_render_pipeline_state(&pipeline, &PipelineOptions::default())
            .unwrap();
        let params: Vec<u8> = [3u32, 11].iter().flat_map(|w| w.to_le_bytes()).collect();
        let params = buffer_with(device, &params).unwrap();
        let out = buffer_with(device, &[0; 64]).unwrap();
        let mut commands = IndirectCommandBufferDescriptor::new();
        commands.command_types = IndirectCommandType::DRAW;
        commands.max_vertex_buffer_bind_count = 2;
        commands.max_fragment_buffer_bind_count = 1;
        let icb = device
            .new_indirect_command_buffer(&commands, 1, ResourceOptions::shared())
            .unwrap();
        let command = icb.indirect_render_command_at(0).unwrap();
        command.set_render_pipeline_state(&state).unwrap();
        command.set_vertex_buffer(Some(&params), 0, 0).unwrap();
        command.set_vertex_buffer(Some(&out), 0, 1).unwrap();
        command
            .draw_primitives(PrimitiveType::Point, 0, 4, 1, 0)
            .unwrap();
        let mut pass = RenderPassDescriptor::new();
        let attachment = &mut pass.depth_attachment;
        attachment.texture = Some(texture.clone());
        attachment.load_action = LoadAction::Clear;
        attachment.store_action = StoreAction::Store;
        attachment.clear_depth = 0.25;
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        let resident = [(&params, ResourceUsage::READ), (&out, ResourceUsage::WRITE)];
        for (buffer, usage) in resident {
            encoder
                .use_resource(buffer.as_resource(), usage, RenderStages::VERTEX)
                .unwrap();
        }
        encoder.use_heap(&heap, RenderStages::VERTEX).unwrap();
        encoder.execute_commands_in_buffer(&icb, 0..1).unwrap();
        encoder.end_encoding().unwrap();
        run(&command_buffer);
        let drawn: Vec<u32> = read_buffer(&out, 0, 16)
            .unwrap()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
            .collect();
        let held = gpu_read(&queue, &bytes, 0..size);
        let cleared = held[..texels * 4]
            .as_chunks::<4>()
            .0
            .iter()
            .filter(|texel| **texel == 0.25f32.to_le_bytes())
            .count();
        eprintln!(
            "HEAPTEX icb: the command drew {drawn:?}; {cleared} of the heap's first {texels} \
             texels hold the clear"
        );
        assert_eq!(drawn, [14, 17, 20, 23], "the command's draw");
        if shadows {
            assert_eq!(cleared, texels, "the clear in the heap's bytes");
        }
    });
}

/// What a heap shadow's copies cost a render pass: a 1024x1024 `Depth32Float` texture, made outside
/// a heap and in a `Private` placement heap, as the depth attachment of 9 passes each in a command
/// buffer of its own, which clear it and do not store it (a frame's depth buffer), clear and store
/// it, or load and store it. Where the heap's texture is a shadow of its bytes (MoltenVK), a pass
/// that loads it copies its texels in before it, and one that stores it copies them back after it;
/// one that clears the whole texture takes no copy in. Elsewhere the two are alike. `HEAPTEX cost`
/// lines print each one's median GPU time per pass; nothing is asserted.
#[test]
fn a_heap_depth_attachment_s_copies_cost() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let options = tracked(StorageMode::Private);
        let side = 1024;
        let mut descriptor =
            TextureDescriptor::texture_2d(PixelFormat::Depth32Float, side, side, false);
        descriptor.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
        descriptor.resource_options = options;
        let plain = device.new_texture(&descriptor).unwrap();
        let size = match device.heap_texture_size_and_align(&descriptor) {
            Ok(answer) => answer.size,
            Err(error) => {
                eprintln!("HEAPTEX cost: refused {error:?}");
                return;
            }
        };
        let heap = heap(device, HeapType::Placement, size, options);
        let placed = heap.new_texture_at_offset(&descriptor, 0).unwrap();
        let actions = [
            (
                "clears and does not store",
                LoadAction::Clear,
                StoreAction::DontCare,
            ),
            ("clears and stores", LoadAction::Clear, StoreAction::Store),
            ("loads and stores", LoadAction::Load, StoreAction::Store),
        ];
        let made = [("plain", &plain), ("heap", &placed)];
        for ((how, load, store), (what, texture)) in actions
            .into_iter()
            .flat_map(|action| made.map(|made| (action, made)))
        {
            let mut times = Vec::new();
            for _ in 0..9 {
                let mut pass = RenderPassDescriptor::new();
                let attachment = &mut pass.depth_attachment;
                attachment.texture = Some(texture.clone());
                attachment.load_action = load;
                attachment.store_action = store;
                let command_buffer = queue.command_buffer().unwrap();
                command_buffer
                    .render_command_encoder(&pass)
                    .unwrap()
                    .end_encoding()
                    .unwrap();
                run(&command_buffer);
                let (start, end) = match (
                    command_buffer.gpu_start_time(),
                    command_buffer.gpu_end_time(),
                ) {
                    (Ok(start), Ok(end)) => (start, end),
                    (start, end) => {
                        eprintln!("HEAPTEX cost: no GPU times ({start:?}, {end:?})");
                        return;
                    }
                };
                times.push(end - start);
            }
            times.sort_by(f64::total_cmp);
            eprintln!(
                "HEAPTEX cost {what}: a {side}x{side} Depth32Float pass that {how}, median \
                 {:.1} us on the GPU (of {:.1} to {:.1})",
                times[4] * 1e6,
                times[0] * 1e6,
                times[8] * 1e6
            );
        }
    });
}

/// The bits of each `float`.
fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

/// Upload each `(texture, bytes)` (a `side`-square texture of 4-byte texels) with a blit, in order,
/// in one command buffer.
fn upload(queue: &CommandQueue, textures: &[(&Texture, &[u8])], side: u64) {
    let sources: Vec<Buffer> = textures
        .iter()
        .map(|(_, bytes)| staged(queue.device(), bytes))
        .collect();
    let command_buffer = queue.command_buffer().unwrap();
    for ((texture, _), source) in textures.iter().zip(&sources) {
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_buffer_to_texture(
            source,
            0,
            side * 4,
            side * side * 4,
            Size::new(side, side, 1),
            texture,
            0,
            0,
            Origin::new(0, 0, 0),
            BlitOption::None,
        )
        .unwrap();
        blit.end_encoding().unwrap();
    }
    run(&command_buffer);
}
