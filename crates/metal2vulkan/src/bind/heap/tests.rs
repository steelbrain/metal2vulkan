use super::*;
use crate::bind::state::{SamplerDescriptor, StageTable, TextureFacts, UsedResource};
use crate::bind::state::{StageShape, UsedClass, UsedShape};
use crate::metal::{PixelFormat, RenderStages, TextureUsage};
use crate::pipeline::sym::{SampledType, StorageFormatClass};
use crate::spirv::testing::committed_profiles;

fn class(
    dim: ImageDim,
    arrayed: bool,
    ms: bool,
    ty: SampledType,
    access: ImageAccess,
) -> ImageHandleClass {
    ImageHandleClass {
        dim,
        arrayed,
        multisampled: ms,
        sampled_type: ty,
        depth_compare: false,
        access,
        format: if access == ImageAccess::Storage {
            StorageFormatClass::Unknown
        } else {
            StorageFormatClass::NotStorage
        },
    }
}

fn shape(ty: TextureType, pf: PixelFormat, samples: u32, usage: TextureUsage) -> TextureShape {
    TextureShape::Texture {
        address_modes: crate::metal::sampler::AddressModeMap::IDENTITY,
        offset_base_axes: 0,
        ycbcr_in_place: false,
        ycbcr_planes_sampler: None,
        texel_rows: false,
        texture_type: ty,
        pixel_format: pf,
        sample_count: samples,
        usage,
        mipmap_level_count: 1,
        sparse: false,
        swizzle: crate::metal::TextureSwizzleChannels::DEFAULT,
    }
}

const READ: TextureUsage = TextureUsage::SHADER_READ;
const WRITE: TextureUsage = TextureUsage::SHADER_WRITE;

/// A member fits a class by its texture type's dimension, arrayedness, and multisampling, its
/// sample count, its numeric type, and the usage the class's views need; a storage class never
/// holds a depth or stencil format, and no class holds a texture buffer.
#[test]
fn fit_is_shape_samples_numeric_type_usage_and_aspect() {
    use ImageAccess::{Sampled, Storage};
    use SampledType::{Float, Uint};
    use TextureType as T;
    let f2d = class(ImageDim::D2, false, false, Float, Sampled);
    let rgba = PixelFormat::RGBA8Unorm;
    assert_eq!(fit(&shape(T::Type2D, rgba, 1, READ), &f2d), Fit::Holds);
    assert_eq!(
        fit(&shape(T::Type2D, PixelFormat::Depth32Float, 1, READ), &f2d),
        Fit::Holds
    );
    assert_eq!(
        fit(&shape(T::Type2D, PixelFormat::R32Uint, 1, READ), &f2d),
        Fit::NumericType
    );
    assert_eq!(
        fit(
            &shape(T::Type2D, rgba, 1, TextureUsage::RENDER_TARGET),
            &f2d
        ),
        Fit::No
    );
    assert_eq!(fit(&shape(T::Type2DArray, rgba, 1, READ), &f2d), Fit::No);
    // A format the table holds no representation of fits no class: the heap does not know which
    // of its textures the function reads, so a resident one it cannot view is the neutral, and
    // nothing is refused.
    let unheld = PixelFormat::RGB8Unorm;
    assert_eq!(fit(&shape(T::Type2D, unheld, 1, READ), &f2d), Fit::No);
    assert_eq!(
        fit(
            &shape(T::Type2D, unheld, 1, READ),
            &class(ImageDim::D2, false, false, Uint, Sampled)
        ),
        Fit::No
    );
    assert_eq!(fit(&shape(T::Type2D, rgba, 4, READ), &f2d), Fit::No);
    assert_eq!(fit(&shape(T::Cube, rgba, 1, READ), &f2d), Fit::No);
    assert_eq!(
        fit(
            &TextureShape::Buffer {
                pixel_format: rgba,
                swizzle: crate::metal::TextureSwizzleChannels::DEFAULT,
                parts: 1,
            },
            &f2d
        ),
        Fit::No
    );
    let ms = class(ImageDim::D2, false, true, Float, Sampled);
    assert_eq!(
        fit(&shape(T::Type2DMultisample, rgba, 4, READ), &ms),
        Fit::Holds
    );
    assert_eq!(
        fit(&shape(T::Type2DMultisample, rgba, 1, READ), &ms),
        Fit::No
    );
    let u2d = class(ImageDim::D2, false, false, Uint, Sampled);
    assert_eq!(
        fit(&shape(T::Type2D, PixelFormat::R32Uint, 1, READ), &u2d),
        Fit::Holds
    );
    let s2d = class(ImageDim::D2, false, false, Float, Storage);
    assert_eq!(fit(&shape(T::Type2D, rgba, 1, WRITE), &s2d), Fit::Holds);
    assert_eq!(fit(&shape(T::Type2D, rgba, 1, READ), &s2d), Fit::No);
    // Apple's private `BlockWritesOnly` is a usage Metal's validation takes a write through, as
    // `ShaderWrite` (CoreAnimation's blur textures are `0x10005`).
    let blocked = TextureUsage::BLOCK_WRITES_ONLY;
    for usage in [blocked, READ | TextureUsage::RENDER_TARGET | blocked] {
        assert_eq!(fit(&shape(T::Type2D, rgba, 1, usage), &s2d), Fit::Holds);
        assert_eq!(fit(&shape(T::Type2D, rgba, 1, usage), &f2d), {
            if usage.contains(READ) {
                Fit::Holds
            } else {
                Fit::No
            }
        });
    }
    assert_eq!(
        fit(&shape(T::Type2D, PixelFormat::Depth32Float, 1, WRITE), &s2d),
        Fit::No
    );
    assert_eq!(
        fit(&shape(T::Type2D, PixelFormat::R32Uint, 1, WRITE), &s2d),
        Fit::NumericType
    );
    // A depth-comparison class holds exactly the depth formats of its shape, as a compared slot
    // takes; the float class of the shape holds them too (a read without comparison).
    let d2d = ImageHandleClass {
        depth_compare: true,
        ..f2d
    };
    for pf in [
        PixelFormat::Depth32Float,
        PixelFormat::Depth16Unorm,
        PixelFormat::Depth32FloatStencil8,
    ] {
        assert_eq!(
            fit(&shape(T::Type2D, pf, 1, READ), &d2d),
            Fit::Holds,
            "{pf:?}"
        );
    }
    assert_eq!(fit(&shape(T::Type2D, rgba, 1, READ), &d2d), Fit::No);
    assert_eq!(
        fit(
            &shape(T::Type2DArray, PixelFormat::Depth32Float, 1, READ),
            &d2d
        ),
        Fit::No
    );
    let dcube = ImageHandleClass {
        depth_compare: true,
        ..class(ImageDim::Cube, false, false, Float, Sampled)
    };
    assert_eq!(
        fit(&shape(T::Cube, PixelFormat::Depth32Float, 1, READ), &dcube),
        Fit::Holds
    );
}

fn used_shape(t: TextureShape, stages: RenderStages) -> UsedShape {
    let mut u = UsedResource::new(
        ResourceClass::Texture(TextureFacts::new(
            TextureType::Type2D,
            PixelFormat::RGBA8Unorm,
        )),
        0,
    );
    u.stages = stages;
    let mut state = BindState::default();
    state.used_resources.push(u);
    let mut s = state.shape().used.remove(&0).expect("used");
    s.class = UsedClass::Texture(t);
    s
}

/// Members are the used image textures by token that reach a stage with a table, then the bound
/// image textures of those stages by slot; buffers, heaps, and texture buffers are not members.
#[test]
fn members_are_used_then_bound_image_textures_reaching_a_table() {
    let t2d = shape(TextureType::Type2D, PixelFormat::RGBA8Unorm, 1, READ);
    let buffer = TextureShape::Buffer {
        pixel_format: PixelFormat::R32Float,
        swizzle: crate::metal::TextureSwizzleChannels::DEFAULT,
        parts: 1,
    };
    let mut shape_ = BindShape::default();
    shape_
        .used
        .insert(9, used_shape(t2d, RenderStages::FRAGMENT));
    shape_.used.insert(2, used_shape(t2d, RenderStages::VERTEX));
    shape_
        .used
        .insert(3, used_shape(buffer, RenderStages::VERTEX));
    let mut heap_used = used_shape(t2d, RenderStages::VERTEX);
    heap_used.class = UsedClass::Heap;
    shape_.used.insert(4, heap_used);
    for stage in [Stage::Vertex, Stage::Fragment] {
        let mut s = StageShape::default();
        s.textures.insert(5, t2d);
        s.textures.insert(1, buffer);
        shape_.stages.insert(stage, s);
    }
    let reach = |s: RenderStages| {
        let mut out = Vec::new();
        if s.contains(RenderStages::VERTEX) {
            out.push(Stage::Vertex);
        }
        if s.contains(RenderStages::FRAGMENT) {
            out.push(Stage::Fragment);
        }
        out
    };
    // Only the vertex stage has a table: the fragment-only token and slot are not members.
    let m = members(&shape_, &[Stage::Vertex], reach);
    let texture = |stage, index| SlotRef {
        stage,
        class: SlotClass::Texture,
        index,
    };
    assert_eq!(
        m.iter()
            .map(|m| (m.source, m.stages.clone()))
            .collect::<Vec<_>>(),
        vec![
            (HeapSource::Used(2), vec![Stage::Vertex]),
            (
                HeapSource::Bound(texture(Stage::Vertex, 5)),
                vec![Stage::Vertex]
            ),
        ]
    );
    let m = members(&shape_, &[Stage::Vertex, Stage::Fragment], reach);
    assert_eq!(
        m.iter().map(|m| m.source).collect::<Vec<_>>(),
        vec![
            HeapSource::Used(2),
            HeapSource::Used(9),
            HeapSource::Bound(texture(Stage::Vertex, 5)),
            HeapSource::Bound(texture(Stage::Fragment, 5)),
        ]
    );
}

/// A table's bytes: the members' nonzero ids (the lowest element of a repeated id), sorted, after
/// leading null entries up to the planned length; `lookup_host` resolves each to its element.
/// Over its planned length, a token with two ids, and a state of another shape are internal.
#[test]
fn table_bytes_are_sorted_deduplicated_and_padded() {
    let t2d = shape(TextureType::Type2D, PixelFormat::RGBA8Unorm, 1, READ);
    let ids = [
        Some(30u64),
        Some(10),
        None,
        Some(0),
        Some(30),
        Some(u64::MAX),
    ];
    let mut state = BindState::default();
    let mut members = Vec::new();
    for (i, id) in ids.iter().enumerate() {
        let token = i as u32 + 1;
        let mut u = UsedResource::new(
            ResourceClass::Texture(TextureFacts::new(
                TextureType::Type2D,
                PixelFormat::RGBA8Unorm,
            )),
            token,
        );
        u.resource_id = *id;
        state.used_resources.push(u);
        members.push(HeapMember {
            source: HeapSource::Used(token),
            shape: t2d,
            stages: vec![Stage::Kernel],
            unreadable: false,
        });
    }
    let planned = PlannedHandleTable {
        set: 0,
        binding: 4,
        stage: Stage::Kernel,
        entries: 7,
    };
    let out = handle_table_bytes(&members, &[planned], &state).expect("bytes");
    assert_eq!((out[0].set, out[0].binding), (0, 4));
    let words: Vec<u32> = out[0]
        .bytes
        .chunks(4)
        .map(|w| u32::from_le_bytes(w.try_into().expect("word")))
        .collect();
    let entries: Vec<(u64, u32)> = words
        .chunks(4)
        .map(|e| (u64::from(e[0]) | (u64::from(e[1]) << 32), e[2]))
        .collect();
    assert_eq!(
        entries,
        vec![
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (10, 2),
            (30, 1),
            (u64::MAX, 6)
        ]
    );
    for (id, e) in [(10, 2), (30, 1), (u64::MAX, 6), (0, 0), (20, 0), (31, 0)] {
        assert_eq!(lookup_host(&entries, id), e, "id {id}");
    }
    // Too short a plan for the state's ids.
    let short = PlannedHandleTable {
        entries: 3,
        ..planned
    };
    assert!(matches!(
        handle_table_bytes(&members, &[short], &state),
        Err(BindError::Internal(_))
    ));
    // One token listed twice with two ids.
    let mut twice = state.clone();
    let mut again = twice.used_resources[0];
    again.resource_id = Some(99);
    twice.used_resources.push(again);
    assert!(matches!(
        handle_table_bytes(&members, &[planned], &twice),
        Err(BindError::Internal(_))
    ));
    // A member whose facts differ from the plan's.
    let mut other = state.clone();
    other.used_resources[1].class =
        ResourceClass::Texture(TextureFacts::new(TextureType::Type2D, PixelFormat::R32Uint));
    assert!(matches!(
        handle_table_bytes(&members, &[planned], &other),
        Err(BindError::Internal(_))
    ));
    // A bound member: its slot's id.
    let slot = SlotRef {
        stage: Stage::Kernel,
        class: SlotClass::Texture,
        index: 2,
    };
    let mut k = StageTable::default();
    k.textures.insert(
        2,
        BoundTexture::Texture {
            facts: TextureFacts::new(TextureType::Type2D, PixelFormat::RGBA8Unorm),
            resource_id: Some(7),
        },
    );
    let mut bound = BindState::default();
    bound.stages.insert(Stage::Kernel, k);
    let m = [HeapMember {
        source: HeapSource::Bound(slot),
        shape: t2d,
        stages: vec![Stage::Kernel],
        unreadable: false,
    }];
    let two = PlannedHandleTable {
        entries: 2,
        ..planned
    };
    let out = handle_table_bytes(&m, &[two], &bound).expect("bytes");
    assert_eq!(&out[0].bytes[16..28], &[7, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]);
}

/// Under `HeapPlacement::ResourceId` each member and listed argument sampler is at the element
/// of its resource ID: the tables map each ID to itself. Under `Planned` (the default) the tables
/// are the planned ones.
#[test]
fn resource_id_placement_puts_members_and_samplers_at_their_ids() {
    let t2d = shape(TextureType::Type2D, PixelFormat::RGBA8Unorm, 1, READ);
    let mut state = BindState::default();
    let mut template = ResourceTemplate::default();
    for (token, id) in [(1, Some(5u64)), (2, Some(3)), (3, None)] {
        let mut u = UsedResource::new(
            ResourceClass::Texture(TextureFacts::new(
                TextureType::Type2D,
                PixelFormat::RGBA8Unorm,
            )),
            token,
        );
        u.resource_id = id;
        state.used_resources.push(u);
        template.heap_members.push(HeapMember {
            source: HeapSource::Used(token),
            shape: t2d,
            stages: vec![Stage::Kernel],
            unreadable: false,
        });
    }
    for id in [7, 2] {
        state
            .argument_samplers
            .push(crate::bind::ArgumentSampler::new(
                SamplerDescriptor::new(),
                id,
            ));
    }
    let planned = PlannedHandleTable {
        set: 0,
        binding: 4,
        stage: Stage::Kernel,
        entries: 4,
    };
    let entries = |state: &BindState| -> Vec<(u64, u32)> {
        let out = handle_table_bytes(&template.heap_members, &[planned], state).expect("bytes");
        out[0]
            .bytes
            .chunks(16)
            .map(|e| {
                let word = |k: usize| u32::from_le_bytes(e[4 * k..4 * k + 4].try_into().unwrap());
                (u64::from(word(0)) | (u64::from(word(1)) << 32), word(2))
            })
            .collect()
    };
    // Planned: elements 1 and 2 hold ids 5 and 3.
    assert_eq!(entries(&state), vec![(0, 0), (0, 0), (3, 2), (5, 1)]);

    state.heap_placement = HeapPlacement::ResourceId;
    assert_eq!(entries(&state), vec![(0, 0), (0, 0), (3, 3), (5, 5)]);
    let samplers = PlannedHandleTable {
        entries: 3,
        ..planned
    };
    let shape = ArgumentSamplersShape::of(&state.argument_samplers);
    let out = sampler_table_bytes(&shape, &[samplers], &state).expect("sampler table");
    let ids: Vec<(u32, u32)> = out[0]
        .bytes
        .chunks(16)
        .map(|e| {
            let word = |k: usize| u32::from_le_bytes(e[4 * k..4 * k + 4].try_into().unwrap());
            (word(0), word(2))
        })
        .collect();
    assert_eq!(ids, vec![(0, 0), (2, 2), (7, 7)]);
}

/// Under `HeapPlacement::ResourceId` a listed argument sampler that names an element is at that
/// element, wherever its resource ID lies: its table entry maps the ID there, under `Planned` too.
#[test]
fn resource_id_placement_puts_a_sampler_at_its_element() {
    let mut state = BindState {
        heap_placement: HeapPlacement::ResourceId,
        ..BindState::default()
    };
    for (id, element) in [(16380, 1), (2, 2)] {
        state.argument_samplers.push(crate::bind::ArgumentSampler {
            element: Some(element),
            ..crate::bind::ArgumentSampler::new(SamplerDescriptor::new(), id)
        });
    }
    let planned = PlannedHandleTable {
        set: 0,
        binding: 4,
        stage: Stage::Kernel,
        entries: 3,
    };
    let table = |state: &BindState| -> Vec<(u64, u32)> {
        let shape = ArgumentSamplersShape::of(&state.argument_samplers);
        let out = sampler_table_bytes(&shape, &[planned], state).expect("sampler table");
        out[0]
            .bytes
            .chunks(16)
            .map(|e| {
                let word = |k: usize| u32::from_le_bytes(e[4 * k..4 * k + 4].try_into().unwrap());
                (u64::from(word(0)) | (u64::from(word(1)) << 32), word(2))
            })
            .collect()
    };
    assert_eq!(table(&state), vec![(0, 0), (2, 2), (16380, 1)]);
    state.heap_placement = HeapPlacement::Planned;
    assert_eq!(table(&state), vec![(0, 0), (2, 2), (16380, 1)]);
}

/// A storage member is viewed through its direct format when its storage choice is `Identity`
/// without an alias, the view the class's formatless declaration serves, or a half-float format's
/// rounding, which a heap write takes from its handle's class. One whose choice is an alias a heap
/// access converts by (`storage_conversions`: an sRGB format through its UNORM twin where the
/// profile stores no sRGB format, a blue-first one through its RGBA-order twin where it stores
/// only that) is viewed through the alias, which the formatless declaration must serve and the
/// image lists among its view formats, while a swizzle that loses its class is refused; another
/// alias (`RGB9E5Float`'s word) is a typed refusal. An `A8Unorm` member held in red
/// (`StorageImageClass::AlphaInRed`) is viewed through its `R8_UNORM` image, whose texels a heap
/// access converts where the handle carries `A8Unorm`'s class: unswizzled, or swizzled to read as
/// unswizzled; a swizzle that moves its alpha carries another class and is refused. Each occurs on
/// the committed profiles; lavapipe views `RGBA8Unorm_sRGB`, `BGRA8Unorm_sRGB` and `BGR10A2Unorm`
/// members through their aliases, and the M3 Max's MoltenVK, which stores them, directly.
#[test]
fn a_storage_member_is_viewed_through_the_alias_a_heap_access_converts() {
    let s2d = class(
        ImageDim::D2,
        false,
        false,
        SampledType::Float,
        ImageAccess::Storage,
    );
    let (mut direct, mut refused, mut in_red, mut converted) = (0, 0, 0, 0);
    let mut lavapipe = BTreeSet::new();
    for (stem, p) in committed_profiles().expect("profiles") {
        let planner = HeapPlanner {
            profile: &p,
            members: Vec::new(),
            classes: vec![s2d],
            tables: Vec::new(),
            defaulted: BTreeSet::new(),
            layouts: Vec::new(),
            opaque_black: false,
            filters: Vec::new(),
            sampling: BTreeSet::new(),
            write_rounding: BTreeMap::new(),
            cross_neutral: BTreeMap::new(),
            storage_reads: BTreeMap::new(),
            sampler_tables: Vec::new(),
            samplers: None,
            directories: Vec::new(),
            snapshots: Vec::new(),
            addressed_tables: false,
            function_tables: None,
            planned: Vec::new(),
            planned_samplers: Vec::new(),
            planned_directories: Vec::new(),
            diagnostics: Vec::new(),
        };
        let conversions = storage_conversions(&p);
        for pf in [
            PixelFormat::RGBA8Unorm,
            PixelFormat::BGRA8Unorm,
            PixelFormat::RGBA8UnormSRGB,
            PixelFormat::BGRA8UnormSRGB,
            PixelFormat::BGR10A2Unorm,
            PixelFormat::RGB9E5Float,
            PixelFormat::R16Float,
            PixelFormat::RGBA16Float,
            PixelFormat::A8Unorm,
        ] {
            let m = shape(TextureType::Type2D, pf, 1, WRITE);
            let got = planner.view(&m, &s2d, vk::ImageLayout::GENERAL);
            let half = matches!(pf, PixelFormat::R16Float | PixelFormat::RGBA16Float);
            let use_ = heap_storage_use(&p);
            use crate::metal::TextureSwizzle as S;
            let swizzled = |swizzle: (S, S, S, S)| {
                let mut swizzled = m;
                if let TextureShape::Texture { swizzle: s, .. } = &mut swizzled {
                    *s = crate::metal::TextureSwizzleChannels::new(
                        swizzle.0, swizzle.1, swizzle.2, swizzle.3,
                    );
                }
                planner.view(&swizzled, &s2d, vk::ImageLayout::GENERAL)
            };
            let not_implemented = Err(BindError::Unsupported(Unsupported::not_implemented(
                Feature::ArgumentBufferTextures,
            )));
            if stem == "moltenvk-m3"
                && !matches!(pf, PixelFormat::RGB9E5Float | PixelFormat::A8Unorm)
                && !half
            {
                assert!(
                    got.as_ref()
                        .is_ok_and(|v| Some(v.format) == crate::metal::format::direct_format(pf)),
                    "{stem} {pf:?}: viewed directly"
                );
            }
            match storage_choice(pf, use_, &p) {
                Ok(c) if half => {
                    let v = got.unwrap_or_else(|e| panic!("{stem} {pf:?} ({:?}): {e}", c.class));
                    assert_eq!(Some(v.format), crate::metal::format::direct_format(pf));
                }
                Ok(c) if c.class == StorageImageClass::Identity && c.alias.is_none() => {
                    let v = got.unwrap_or_else(|e| panic!("{stem} {pf:?}: {e}"));
                    assert_eq!(Some(v.format), crate::metal::format::direct_format(pf));
                    assert_eq!(v.layout, vk::ImageLayout::GENERAL);
                    direct += 1;
                }
                Ok(c) if c.class == StorageImageClass::AlphaInRed => {
                    let v = got.unwrap_or_else(|e| panic!("{stem} {pf:?}: {e}"));
                    assert_eq!(v.format, vk::Format::R8_UNORM, "{stem}");
                    for (swizzle, kept) in [
                        ((S::Zero, S::Zero, S::Zero, S::Alpha), true),
                        ((S::Alpha, S::Zero, S::Zero, S::One), false),
                    ] {
                        let got = swizzled(swizzle);
                        match kept {
                            true => assert_eq!(
                                got.map(|v| v.format),
                                Ok(vk::Format::R8_UNORM),
                                "{stem} {swizzle:?}"
                            ),
                            false => assert_eq!(got, not_implemented, "{stem} {swizzle:?}"),
                        }
                    }
                    in_red += 1;
                }
                Ok(c)
                    if c.alias.is_some()
                        && conversions.contains(&StorageConversion {
                            index: crate::metal::texel_format::heap_class_index(
                                pf,
                                crate::metal::TextureSwizzleChannels::DEFAULT,
                            ),
                            red_blue: crate::metal::format::red_blue_exchanged(pf),
                            class: c.class,
                        }) =>
                {
                    match without_format_gaps(c.vk_format, use_, &p).is_empty() {
                        true => {
                            let v = got.unwrap_or_else(|e| panic!("{stem} {pf:?}: {e}"));
                            assert_eq!(v.format, c.vk_format, "{stem} {pf:?}");
                            assert!(crate::metal::format::storage_view_formats(pf, &p)
                                .contains(&v.format));
                            assert_eq!(
                                swizzled((S::Blue, S::Green, S::Red, S::Alpha)),
                                not_implemented,
                                "{stem} {pf:?}: a swizzle that loses the class"
                            );
                            if stem == "lavapipe" {
                                lavapipe.insert(pf);
                            }
                            converted += 1;
                        }
                        false => assert!(got.is_err(), "{stem} {pf:?}"),
                    }
                }
                Ok(_) => {
                    assert_eq!(got, not_implemented, "{stem} {pf:?}");
                    refused += 1;
                }
                Err(_) => assert!(got.is_err(), "{stem} {pf:?}"),
            }
        }
    }
    assert!(
        direct > 0 && refused > 0 && in_red > 0 && converted > 0,
        "direct {direct}, refused {refused}, in red {in_red}, converted {converted}"
    );
    assert_eq!(
        lavapipe,
        BTreeSet::from([
            PixelFormat::RGBA8UnormSRGB,
            PixelFormat::BGRA8UnormSRGB,
            PixelFormat::BGR10A2Unorm,
        ])
    );
}

/// On lavapipe a heap's storage conversions are the alias a slot takes for each of the three
/// formats `read-write-tier-held` sweeps beyond every tier, keyed apart from their red-first and
/// UNORM kin by class and red-blue order; on the M3 Max's MoltenVK, which stores each format
/// itself, there are none, so no module there converts or notes a conversion.
#[test]
fn a_heap_converts_by_class_and_red_blue_order() {
    let key = |pf| {
        (
            crate::metal::texel_format::heap_class_index(
                pf,
                crate::metal::TextureSwizzleChannels::DEFAULT,
            ),
            crate::metal::format::red_blue_exchanged(pf),
        )
    };
    for (stem, p) in committed_profiles().expect("profiles") {
        let got: BTreeMap<(u32, bool), StorageImageClass> = storage_conversions(&p)
            .into_iter()
            .map(|c| ((c.index, c.red_blue), c.class))
            .collect();
        match stem.as_str() {
            "lavapipe" => {
                assert_eq!(
                    got.get(&key(PixelFormat::RGBA8UnormSRGB)),
                    Some(&StorageImageClass::SrgbEncode)
                );
                assert_eq!(
                    got.get(&key(PixelFormat::BGRA8UnormSRGB)),
                    Some(&StorageImageClass::SrgbEncode)
                );
                assert_eq!(
                    got.get(&key(PixelFormat::BGR10A2Unorm)),
                    Some(&StorageImageClass::SwizzleBgra)
                );
                for pf in [
                    PixelFormat::RGBA8Unorm,
                    PixelFormat::BGRA8Unorm,
                    PixelFormat::RGB10A2Unorm,
                ] {
                    assert_eq!(got.get(&key(pf)), None, "{pf:?} is stored as it is");
                }
                assert_ne!(
                    key(PixelFormat::BGR10A2Unorm),
                    key(PixelFormat::RGB10A2Unorm)
                );
            }
            "moltenvk-m3" => assert!(got.is_empty(), "{got:?}"),
            _ => {}
        }
    }
}

/// One layout per bound texture: a slot whose own write or heap element is a storage image is
/// `GENERAL` in every write naming it (plain, combined, heap), and nothing else changes: a slot
/// only sampled, a used member, and a neutral keep theirs.
#[test]
fn a_storage_write_makes_every_write_of_its_slot_general() {
    let s = |index| SlotRef {
        stage: Stage::Kernel,
        class: SlotClass::Texture,
        index,
    };
    let ro = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
    let general = vk::ImageLayout::GENERAL;
    let view = |layout| ImageViewDesc {
        view_type: vk::ImageViewType::TYPE_2D,
        format: vk::Format::R8G8B8A8_UNORM,
        aspect: vk::ImageAspectFlags::COLOR,
        layout,
        components: vk::ComponentMapping::IDENTITY,
        level: None,
        shader_swizzle: false,
    };
    let sampled = HandleClass::Image(class(
        ImageDim::D2,
        false,
        false,
        SampledType::Float,
        ImageAccess::Sampled,
    ));
    let storage = HandleClass::Image(class(
        ImageDim::D2,
        false,
        false,
        SampledType::Float,
        ImageAccess::Storage,
    ));
    let write = |set, binding, array_element, ty, source| DescriptorWrite {
        set,
        binding,
        array_element,
        ty,
        source,
    };
    let entry = |class, element, source, layout| WriteSource::HeapEntry {
        class,
        element,
        source,
        view: view(layout),
    };
    let HandleClass::Image(storage_image) = storage else {
        unreachable!()
    };
    let unbound = WriteSource::Neutral(neutral(&storage_image).expect("neutral"));
    let (si, sm) = (
        vk::DescriptorType::STORAGE_IMAGE,
        vk::DescriptorType::SAMPLED_IMAGE,
    );
    let mut writes = vec![
        // Slot 0: a storage slot whose texture the sampled heap holds.
        write(
            0,
            0,
            0,
            si,
            WriteSource::Texture {
                slot: s(0),
                view: view(general),
            },
        ),
        write(1, 0, 1, sm, entry(sampled, 1, HeapSource::Bound(s(0)), ro)),
        // Slot 1: a sampled and a combined slot whose texture a storage heap array holds.
        write(
            0,
            1,
            0,
            sm,
            WriteSource::Texture {
                slot: s(1),
                view: view(ro),
            },
        ),
        write(
            0,
            2,
            0,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            WriteSource::CombinedImageSampler {
                image: ImageSource::Texture {
                    slot: s(1),
                    view: view(ro),
                },
                sampler: crate::bind::plan::SamplerSource::Immutable,
            },
        ),
        write(
            1,
            0,
            2,
            sm,
            entry(sampled, 2, HeapSource::Bound(s(1)), general),
        ),
        write(
            1,
            1,
            2,
            si,
            entry(storage, 2, HeapSource::Bound(s(1)), general),
        ),
        // Slot 2 only sampled, a used member, and a neutral: unchanged.
        write(
            0,
            3,
            0,
            sm,
            WriteSource::Texture {
                slot: s(2),
                view: view(ro),
            },
        ),
        write(1, 0, 3, sm, entry(sampled, 3, HeapSource::Used(7), ro)),
        write(1, 1, 1, si, unbound),
    ];
    let before = writes.clone();
    one_layout_per_slot(&mut writes);
    let layouts: Vec<Option<vk::ImageLayout>> = writes
        .iter_mut()
        .map(|w| match &mut w.source {
            WriteSource::HeapEntry { view, .. } => Some(view.layout),
            other => slot_view(other).map(|(_, v)| v.layout),
        })
        .collect();
    assert_eq!(
        layouts,
        vec![
            Some(general),
            Some(general),
            Some(general),
            Some(general),
            Some(general),
            Some(general),
            Some(ro),
            Some(ro),
            None,
        ]
    );
    // Only layouts change.
    for (a, b) in writes.iter_mut().zip(&before) {
        if let Some((_, v)) = slot_view(&mut a.source) {
            v.layout = vk::ImageLayout::UNDEFINED;
        }
        let mut b = b.clone();
        if let Some((_, v)) = slot_view(&mut b.source) {
            v.layout = vk::ImageLayout::UNDEFINED;
        }
        assert_eq!(*a, b);
    }
}

/// A heap member is checked at one level: the bind state's texture of more levels is that member
/// (a heap array views level 0 alone), and one of another format is not.
#[test]
fn a_member_s_level_count_is_not_the_plan_s() {
    let mut facts = TextureFacts::new(TextureType::Type2D, PixelFormat::RGBA8Unorm);
    facts.usage = READ;
    facts.mipmap_level_count = 3;
    let mut used = UsedResource::new(ResourceClass::Texture(facts), 7);
    used.resource_id = Some(42);
    let mut state = BindState::default();
    state.used_resources.push(used);
    let by_token = UsedByToken::Gathered(BTreeMap::from([(
        7,
        state.used_resources.iter().collect::<Vec<_>>(),
    )]));
    let member = |shape: TextureShape| HeapMember {
        source: HeapSource::Used(7),
        shape,
        stages: vec![Stage::Kernel],
        unreadable: false,
    };
    let planned = facts.shape().at_one_level();
    assert_ne!(planned, facts.shape());
    assert_eq!(
        member_id(&member(planned), &by_token, &state).unwrap(),
        Some(42)
    );
    assert_eq!(
        member_id(&member(facts.shape()), &by_token, &state).unwrap(),
        Some(42)
    );
    let other = shape(TextureType::Type2D, PixelFormat::R32Float, 1, READ);
    assert!(member_id(&member(other), &by_token, &state).is_err());
}

/// A texture of a form a heap reads (`ExtendedRange::HEAP_READ`: unswizzled, of one sample, not
/// sparse) is held raw: the unsigned integer class of its shape views it as a slot's read does
/// (its raw format), and the float class holds its neutral (`Fit::Raw`). A plan's float array
/// writes the neutral, or refuses it where a stage that enters it samples heap textures.
#[test]
fn a_form_a_heap_reads_is_held_raw_in_the_unsigned_class() {
    use crate::metal::format::ExtendedRange;
    use ImageAccess::Sampled;
    use SampledType::{Float, Sint, Uint};
    fn refused<T>() -> Result<T, BindError> {
        Err(BindError::Unsupported(Unsupported::not_implemented(
            Feature::ArgumentBufferTextures,
        )))
    }
    let f2d = class(ImageDim::D2, false, false, Float, Sampled);
    let u2d = class(ImageDim::D2, false, false, Uint, Sampled);
    let i2d = class(ImageDim::D2, false, false, Sint, Sampled);
    let layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
    let read: Vec<PixelFormat> = PixelFormat::ALL
        .iter()
        .copied()
        .filter(|&pf| {
            ExtendedRange::of(pf)
                .and_then(ExtendedRange::heap_read)
                .is_some()
        })
        .collect();
    assert_eq!(read.len(), ExtendedRange::HEAP_READ.len());
    let profiles = committed_profiles().expect("profiles");
    for (stem, p) in &profiles {
        for &pf in &read {
            let s = shape(TextureType::Type2D, pf, 1, READ);
            assert_eq!(fit(&s, &f2d), Fit::Raw, "{pf:?}");
            assert_eq!(fit(&s, &u2d), Fit::Holds, "{pf:?}");
            assert_eq!(fit(&s, &i2d), Fit::NumericType, "{pf:?}");
            let slot = crate::bind::plan::resources::sampled_view(p, pf, &u2d)
                .unwrap_or_else(|e| panic!("{stem} {pf:?}: {e}"));
            assert_eq!(slot.format, ExtendedRange::of(pf).unwrap().raw_format());
        }
    }

    // A plan's float array holds the neutral, and its unsigned one the raw view; a stage that
    // samples heap textures refuses the float array.
    let (_, p) = &profiles[0];
    let pf = read[0];
    let planner = |classes: Vec<ImageHandleClass>, sampling: BTreeSet<Stage>| HeapPlanner {
        profile: p,
        members: vec![HeapMember {
            source: HeapSource::Used(1),
            shape: shape(TextureType::Type2D, pf, 1, READ),
            stages: vec![Stage::Kernel],
            unreadable: false,
        }],
        classes,
        tables: Vec::new(),
        defaulted: BTreeSet::new(),
        layouts: vec![layout],
        opaque_black: false,
        filters: Vec::new(),
        sampling,
        write_rounding: BTreeMap::new(),
        cross_neutral: BTreeMap::new(),
        storage_reads: BTreeMap::new(),
        sampler_tables: Vec::new(),
        samplers: None,
        directories: Vec::new(),
        snapshots: Vec::new(),
        addressed_tables: false,
        function_tables: None,
        planned: Vec::new(),
        planned_samplers: Vec::new(),
        planned_directories: Vec::new(),
        diagnostics: Vec::new(),
    };
    let binding = |c: ImageHandleClass| DescriptorBinding {
        binding: 0,
        ty: vk::DescriptorType::SAMPLED_IMAGE,
        count: 16,
        stages: vk::ShaderStageFlags::COMPUTE,
        source: BindingSource::Heap(HandleClass::Image(c)),
        immutable_sampler: None,
        flags: vk::DescriptorBindingFlags::empty(),
        variables: 1,
        immutable_samplers: Vec::new(),
    };
    let mut reads = planner(vec![f2d, u2d], BTreeSet::new());
    let float = reads.array(1, &binding(f2d)).expect("float array");
    assert!(
        matches!(float[1].source, WriteSource::Neutral(_)),
        "{:?}",
        float[1]
    );
    let uint = reads.array(1, &binding(u2d)).expect("unsigned array");
    assert!(
        matches!(uint[1].source, WriteSource::HeapEntry { view, .. }
            if view.format == ExtendedRange::of(pf).unwrap().raw_format()),
        "{:?}",
        uint[1]
    );
    let mut samples = planner(vec![f2d, u2d], BTreeSet::from([Stage::Kernel]));
    assert_eq!(samples.array(1, &binding(f2d)), refused());
    let mut other = planner(vec![f2d, u2d], BTreeSet::from([Stage::Fragment]));
    assert!(
        other.array(1, &binding(f2d)).is_ok(),
        "another stage samples"
    );
}

/// A storage class array refuses a member that a stage of `RuntimeFact::HeapWriteRounding` reaches
/// where the stage's mode-2 write to it is unbuilt (`RG11B10Float`, `RGB9E5Float`, and a
/// half-float texture where the stage writes `float` texels), naming
/// `PrivateRenderField::TextureWriteRounding`; any other member, and one only another stage
/// reaches, is written as before.
#[test]
fn a_heap_member_a_mode_two_write_would_round_is_refused() {
    use crate::error::PrivateRenderField;
    let s2d = class(
        ImageDim::D2,
        false,
        false,
        SampledType::Float,
        ImageAccess::Storage,
    );
    let refused = Err(BindError::Unsupported(Unsupported::not_implemented(
        Feature::PrivateRenderState(PrivateRenderField::TextureWriteRounding),
    )));
    let binding = DescriptorBinding {
        binding: 0,
        ty: vk::DescriptorType::STORAGE_IMAGE,
        count: 16,
        stages: vk::ShaderStageFlags::FRAGMENT,
        source: BindingSource::Heap(HandleClass::Image(s2d)),
        immutable_sampler: None,
        flags: vk::DescriptorBindingFlags::empty(),
        variables: 1,
        immutable_samplers: Vec::new(),
    };
    let mut checked = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        for pf in [
            PixelFormat::RGBA8Unorm,
            PixelFormat::R32Float,
            PixelFormat::RG16Float,
            PixelFormat::RGBA16Float,
            PixelFormat::RG11B10Float,
            PixelFormat::RGB9E5Float,
        ] {
            let member = shape(TextureType::Type2D, pf, 1, READ | WRITE);
            if fit(&member, &s2d) != Fit::Holds {
                continue;
            }
            for (stage, halves) in [
                (Stage::Fragment, true),
                (Stage::Fragment, false),
                (Stage::Vertex, true),
            ] {
                let mut planner = HeapPlanner {
                    profile: &p,
                    members: vec![HeapMember {
                        source: HeapSource::Used(1),
                        shape: member,
                        stages: vec![Stage::Fragment],
                        unreadable: false,
                    }],
                    classes: vec![s2d],
                    tables: Vec::new(),
                    defaulted: BTreeSet::new(),
                    layouts: vec![vk::ImageLayout::GENERAL],
                    opaque_black: false,
                    filters: Vec::new(),
                    sampling: BTreeSet::new(),
                    write_rounding: BTreeMap::from([(stage, halves)]),
                    cross_neutral: BTreeMap::new(),
                    storage_reads: BTreeMap::new(),
                    sampler_tables: Vec::new(),
                    samplers: None,
                    directories: Vec::new(),
                    snapshots: Vec::new(),
                    addressed_tables: false,
                    function_tables: None,
                    planned: Vec::new(),
                    planned_samplers: Vec::new(),
                    planned_directories: Vec::new(),
                    diagnostics: Vec::new(),
                };
                let what = format!("{stem} {pf:?} {stage:?} halves {halves}");
                let rounds = stage == Stage::Fragment
                    && crate::metal::texel_format::mode_two_rounds(pf, halves);
                let got = planner.array(1, &binding);
                match rounds {
                    true => assert_eq!(got, refused, "{what}"),
                    false => assert_ne!(got, refused, "{what}"),
                }
                checked += usize::from(rounds);
            }
        }
    }
    assert!(checked > 0, "no profile holds a format the check refuses");
}

/// A state listed in token order (`BindState::used_in_token_order`) is searched, not gathered by
/// token, and gives the same tables as the same state listed so without the flag: among a buffer
/// per token around them, with a member listed twice under one id, and the same internal errors
/// for a member listed with two ids and a member not listed.
#[test]
fn a_state_in_token_order_finds_its_members_by_search() {
    let t2d = shape(TextureType::Type2D, PixelFormat::RGBA8Unorm, 1, READ);
    let texture = || {
        ResourceClass::Texture(TextureFacts::new(
            TextureType::Type2D,
            PixelFormat::RGBA8Unorm,
        ))
    };
    let mut state = BindState::default();
    let mut template = ResourceTemplate::default();
    for token in 0..200u32 {
        let member = token % 50 == 7;
        let class = match member {
            true => texture(),
            false => ResourceClass::Buffer,
        };
        let mut u = UsedResource::new(class, token);
        u.resource_id = member.then_some(u64::from(token) + 1);
        state.used_resources.push(u);
        if token == 57 {
            state.used_resources.push(u);
        }
        if member {
            template.heap_members.push(HeapMember {
                source: HeapSource::Used(token),
                shape: t2d,
                stages: vec![Stage::Kernel],
                unreadable: false,
            });
        }
    }
    state.heap_placement = HeapPlacement::ResourceId;
    let planned = PlannedHandleTable {
        set: 0,
        binding: 4,
        stage: Stage::Kernel,
        entries: 8,
    };
    let mut ordered = state.clone();
    ordered.used_in_token_order = true;
    assert!(matches!(UsedByToken::of(&ordered), UsedByToken::Listed(..)));
    assert!(matches!(UsedByToken::of(&state), UsedByToken::Gathered(_)));
    let tables = |state: &BindState| handle_table_bytes(&template.heap_members, &[planned], state);
    assert_eq!(
        tables(&ordered).expect("ordered"),
        tables(&state).expect("any")
    );
    // Member 107 listed again with another id, beside its first entry.
    for state in [&state, &ordered] {
        let mut twice = state.clone();
        let at = 1 + twice
            .used_resources
            .iter()
            .position(|u| u.token == 107)
            .unwrap();
        let mut again = twice.used_resources[at - 1];
        again.resource_id = Some(999);
        twice.used_resources.insert(at, again);
        assert!(matches!(tables(&twice), Err(BindError::Internal(_))));
        let mut missing = state.clone();
        missing.used_resources.retain(|u| u.token != 157);
        assert!(matches!(tables(&missing), Err(BindError::Internal(_))));
    }
}

/// A state listed in token order finds its heap members' entries searching forward from where
/// the last member's ended, the plan's members ascending by token (`members`): 4096 textures,
/// each beside a buffer, compare a few entries a member (an upper bound of 8), not the logarithm
/// of the 8192 listed twice over, answering as the same state in no order does.
#[test]
fn a_state_in_token_order_finds_its_members_searching_forward() {
    const MEMBERS: u32 = 4096;
    let t2d = shape(TextureType::Type2D, PixelFormat::RGBA8Unorm, 1, READ);
    let mut state = BindState::default();
    let mut template = ResourceTemplate::default();
    for member in 0..MEMBERS {
        let token = 2 * member + 1;
        state
            .used_resources
            .push(UsedResource::new(ResourceClass::Buffer, token - 1));
        let class = ResourceClass::Texture(TextureFacts::new(
            TextureType::Type2D,
            PixelFormat::RGBA8Unorm,
        ));
        let mut texture = UsedResource::new(class, token);
        texture.resource_id = Some(u64::from(token) + 1);
        state.used_resources.push(texture);
        template.heap_members.push(HeapMember {
            source: HeapSource::Used(token),
            shape: t2d,
            stages: vec![Stage::Kernel],
            unreadable: false,
        });
    }
    state.heap_placement = HeapPlacement::ResourceId;
    let planned = PlannedHandleTable {
        set: 0,
        binding: 4,
        stage: Stage::Kernel,
        entries: 2 * MEMBERS + 8,
    };
    let mut ordered = state.clone();
    ordered.used_in_token_order = true;
    let looked_at = |work: &dyn Fn()| {
        LISTED_LOOKED_AT.with(|looked| looked.set(0));
        work();
        LISTED_LOOKED_AT.with(Cell::get)
    };
    let tables = |state: &BindState| handle_table_bytes(&template.heap_members, &[planned], state);
    let bound = 8 * u64::from(MEMBERS);
    let looked = looked_at(&|| assert_eq!(tables(&ordered).unwrap(), tables(&state).unwrap()));
    assert!(looked <= bound, "the tables' lookups compared {looked}");
}

/// A texture buffer fits only a texel-buffer class: the one of its numeric type holds it, where its
/// usage has the class's access (its facts' usage; a slot's bound texture buffer states none), and
/// one of another numeric type fits it in everything but that; no image class fits it.
#[test]
fn a_texture_buffer_is_held_by_the_texel_buffer_class_of_its_numeric_type() {
    use ImageAccess::{Sampled, Storage};
    use SampledType::{Float, Uint};
    let fb = class(ImageDim::Buffer, false, false, Float, Sampled);
    let ub = class(ImageDim::Buffer, false, false, Uint, Sampled);
    let fsb = class(ImageDim::Buffer, false, false, Float, Storage);
    let f2d = class(ImageDim::D2, false, false, Float, Sampled);
    let rgba = PixelFormat::RGBA32Float;
    let bound = TextureShape::Buffer {
        pixel_format: rgba,
        swizzle: crate::metal::TextureSwizzleChannels::DEFAULT,
        parts: 1,
    };
    let read = shape(TextureType::TextureBuffer, rgba, 1, READ);
    let write = shape(TextureType::TextureBuffer, rgba, 1, WRITE);
    for s in [&bound, &read] {
        assert_eq!(fit(s, &fb), Fit::Holds);
        assert_eq!(fit(s, &ub), Fit::NumericType);
        assert_eq!(fit(s, &f2d), Fit::No);
    }
    assert_eq!(fit(&bound, &fsb), Fit::Holds);
    assert_eq!(fit(&read, &fsb), Fit::No);
    assert_eq!(fit(&write, &fsb), Fit::Holds);
    assert_eq!(fit(&write, &fb), Fit::No);
    assert_eq!(
        fit(&shape(TextureType::Type2D, rgba, 1, READ), &fb),
        Fit::No
    );
}
