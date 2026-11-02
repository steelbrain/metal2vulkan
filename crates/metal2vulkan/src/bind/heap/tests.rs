use super::*;
use crate::bind::neutral::NeutralResource;
use crate::bind::state::{SamplerDescriptor, StageTable, TextureFacts, UsedResource};
use crate::bind::state::{StageShape, UsedClass, UsedShape};
use crate::metal::sampler::SamplerUse;
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
        texture_type: ty,
        pixel_format: pf,
        sample_count: samples,
        usage,
        mipmap_level_count: 1,
        sparse: false,
        swizzle: crate::metal::TextureSwizzleChannels::DEFAULT,
    }
}

/// The committed `moltenvk-m3` profile.
fn m3() -> DeviceProfile {
    crate::pipeline::testing::profile("moltenvk-m3", crate::device::DevicePolicy::default())
        .expect("profile")
}

const READ: TextureUsage = TextureUsage::SHADER_READ;
const WRITE: TextureUsage = TextureUsage::SHADER_WRITE;

/// On every committed profile the heap classes are distinct and sorted, every image class has
/// a neutral of exactly its type (an empty heap element can be written), the four sampler
/// classes (one per use: comparison and integer texel) close the list, and cube arrays and
/// multisampled storage images follow the profile.
#[test]
fn heap_classes_are_the_declarable_images_and_every_sampler_use() {
    for (stem, p) in committed_profiles().expect("profiles") {
        let classes = heap_classes(&p);
        assert!(classes.windows(2).all(|w| w[0] < w[1]), "{stem}");
        for compare in [false, true] {
            for integer in [false, true] {
                let class = HandleClass::Sampler { compare, integer };
                assert!(classes.contains(&class));
                assert_eq!(class.sampler_use(), Some(SamplerUse::new(compare, integer)));
                assert_eq!(class.sampler_use().map(HandleClass::sampler), Some(class));
            }
        }
        let images = &p.fields().images;
        for c in &classes {
            let HandleClass::Image(i) = c else { continue };
            assert!(
                NeutralResource::for_image(i).is_some(),
                "{stem}: {i:?} has no neutral"
            );
            if i.dim == ImageDim::Cube && i.arrayed {
                assert!(
                    images.cube_array,
                    "{stem}: cube array without imageCubeArray"
                );
            }
            if i.multisampled && i.access == ImageAccess::Storage {
                assert!(images.storage_multisample, "{stem}: {i:?}");
            }
        }
        let n_images = classes.len() - 4;
        assert_eq!(n_images, image_classes(&p).len(), "{stem}: duplicates");
    }
}

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
    }];
    let two = PlannedHandleTable {
        entries: 2,
        ..planned
    };
    let out = handle_table_bytes(&m, &[two], &bound).expect("bytes");
    assert_eq!(&out[0].bytes[16..28], &[7, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]);
}

/// Under `HeapPlacement::ResourceId` each member and listed argument sampler is at the element
/// of its resource ID: the tables map each ID to itself, and `HeapElements` sends each heap write
/// there (a member without an ID, and a sampler element past the list, take no write); writes
/// outside the heap set and the neutral element keep theirs. An ID at or past the capacity is
/// `HeapTooLarge`. Under `Planned` (the default) every write keeps its element and the tables
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
    let write = |set: u32, ty: vk::DescriptorType, array_element: u32| DescriptorWrite {
        set,
        binding: 0,
        array_element,
        ty,
        source: WriteSource::Neutral(NeutralResource::Sampler { compare: false }),
    };
    let (image, sampler) = (
        vk::DescriptorType::SAMPLED_IMAGE,
        vk::DescriptorType::SAMPLER,
    );

    // Planned: elements 1 and 2 hold ids 5 and 3, and every write keeps its element.
    assert_eq!(entries(&state), vec![(0, 0), (0, 0), (3, 2), (5, 1)]);
    let planned_elements = heap_elements(&template, &state, 16).expect("planned");
    for (set, ty, e) in [(1, image, 2), (1, sampler, 9), (0, image, 4)] {
        assert_eq!(planned_elements.element(&write(set, ty, e)), Some(e));
    }

    state.heap_placement = HeapPlacement::ResourceId;
    assert_eq!(entries(&state), vec![(0, 0), (0, 0), (3, 3), (5, 5)]);
    let elements = heap_elements(&template, &state, 16).expect("by id");
    let placed = |set, ty, e| elements.element(&write(set, ty, e));
    assert_eq!(placed(1, image, 0), Some(0), "the neutral element");
    assert_eq!(placed(1, image, 1), Some(5));
    assert_eq!(placed(1, image, 2), Some(3));
    assert_eq!(placed(1, image, 3), None, "a member without an id");
    assert_eq!(placed(1, sampler, 1), Some(7));
    assert_eq!(placed(1, sampler, 2), Some(2));
    assert_eq!(placed(1, sampler, 3), None, "past the listed samplers");
    assert_eq!(placed(0, image, 3), Some(3), "outside the heap set");
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

    // An id at the capacity.
    assert!(matches!(
        heap_elements(&template, &state, 5),
        Err(BindError::HeapTooLarge { count: 6, limit: 5 })
    ));
}

/// Under `HeapPlacement::ResourceId` a listed argument sampler that names an element is at that
/// element, wherever its resource ID lies: its table entry maps the ID there, and its writes go
/// there, so an ID past the capacity is held. Its element at or past the capacity is
/// `HeapTooLarge`; under `Planned` the element is ignored.
#[test]
fn resource_id_placement_puts_a_sampler_at_its_element() {
    let template = ResourceTemplate::default();
    let mut state = BindState {
        heap_placement: HeapPlacement::ResourceId,
        ..BindState::default()
    };
    for (id, element) in [(16380, 1), (2, 2)] {
        state.argument_samplers.push(
            crate::bind::ArgumentSampler::new(SamplerDescriptor::new(), id).at_element(element),
        );
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
    let write = |array_element: u32| DescriptorWrite {
        set: 1,
        binding: 0,
        array_element,
        ty: vk::DescriptorType::SAMPLER,
        source: WriteSource::Neutral(NeutralResource::Sampler { compare: false }),
    };
    let elements = heap_elements(&template, &state, 16).expect("by element");
    assert_eq!(
        elements.element(&write(1)),
        Some(1),
        "the state at ID 16380"
    );
    assert_eq!(elements.element(&write(2)), Some(2));
    assert!(matches!(
        heap_elements(&template, &state, 2),
        Err(BindError::HeapTooLarge { count: 3, limit: 2 })
    ));
    state.heap_placement = HeapPlacement::Planned;
    assert_eq!(table(&state), vec![(0, 0), (2, 2), (16380, 1)]);
    let planned_elements = heap_elements(&template, &state, 2).expect("planned");
    assert_eq!(planned_elements.element(&write(2)), Some(2));
}

/// A storage member is viewed through its direct format when its storage choice is `Identity`
/// without an alias, the view the class's formatless declaration serves, or a half-float format's
/// rounding, which a heap write takes from its handle's class; one needing a conversion view (bgra8
/// or sRGB where the profile cannot store them directly) is a typed refusal. Both occur on the
/// committed profiles, and every profile views a half-float member directly. An `A8Unorm` member
/// held in red (`StorageImageClass::AlphaInRed`) is viewed through its `R8_UNORM` image, whose
/// texels a heap access converts where the handle carries `A8Unorm`'s class: unswizzled, or
/// swizzled to read as unswizzled; a swizzle that moves its alpha carries another class and is
/// refused.
#[test]
fn a_storage_member_needing_a_conversion_view_is_refused() {
    let s2d = class(
        ImageDim::D2,
        false,
        false,
        SampledType::Float,
        ImageAccess::Storage,
    );
    let (mut direct, mut refused, mut in_red) = (0, 0, 0);
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
            sampler_tables: Vec::new(),
            samplers: None,
            directories: Vec::new(),
            snapshots: Vec::new(),
            addressed_tables: false,
            function_tables: None,
            zero_bytes: 0,
            planned: Vec::new(),
            planned_samplers: Vec::new(),
            planned_directories: Vec::new(),
            diagnostics: Vec::new(),
        };
        for pf in [
            PixelFormat::RGBA8Unorm,
            PixelFormat::BGRA8Unorm,
            PixelFormat::RGBA8UnormSRGB,
            PixelFormat::R16Float,
            PixelFormat::RGBA16Float,
            PixelFormat::A8Unorm,
        ] {
            let m = shape(TextureType::Type2D, pf, 1, WRITE);
            let got = planner.view(&m, &s2d, vk::ImageLayout::GENERAL);
            let half = matches!(pf, PixelFormat::R16Float | PixelFormat::RGBA16Float);
            match storage_choice(pf, heap_storage_use(&p), &p) {
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
                    use crate::metal::TextureSwizzle as S;
                    for (swizzle, kept) in [
                        ((S::Zero, S::Zero, S::Zero, S::Alpha), true),
                        ((S::Alpha, S::Zero, S::Zero, S::One), false),
                    ] {
                        let mut swizzled = m;
                        if let TextureShape::Texture { swizzle: s, .. } = &mut swizzled {
                            *s = crate::metal::TextureSwizzleChannels::new(
                                swizzle.0, swizzle.1, swizzle.2, swizzle.3,
                            );
                        }
                        let got = planner.view(&swizzled, &s2d, vk::ImageLayout::GENERAL);
                        match kept {
                            true => assert_eq!(
                                got.map(|v| v.format),
                                Ok(vk::Format::R8_UNORM),
                                "{stem} {swizzle:?}"
                            ),
                            false => assert_eq!(
                                got,
                                Err(BindError::Unsupported(Unsupported::not_implemented(
                                    Feature::ArgumentBufferTextures
                                ))),
                                "{stem} {swizzle:?}"
                            ),
                        }
                    }
                    in_red += 1;
                }
                Ok(_) => {
                    assert_eq!(
                        got,
                        Err(BindError::Unsupported(Unsupported::not_implemented(
                            Feature::ArgumentBufferTextures
                        ))),
                        "{stem} {pf:?}"
                    );
                    refused += 1;
                }
                Err(_) => assert!(got.is_err(), "{stem} {pf:?}"),
            }
        }
    }
    assert!(
        direct > 0 && refused > 0 && in_red > 0,
        "direct {direct}, refused {refused}, in red {in_red}"
    );
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

/// A plan's sampler tables list again any states that fit them: as many as the planned entries
/// hold beside the null entry, each of a shape the plan was made for or one its checks admit
/// ([`sampler_tables_admit_the_shapes_the_plan_s_checks_pass`]), written as the plan writes
/// them; a list past the entries is not the plan's (`None`), and a plan without sampler tables
/// relists nothing.
#[test]
fn sampler_tables_relist_states_of_the_plans_shapes() {
    let linear = {
        let mut d = SamplerDescriptor::new();
        d.min_filter = crate::metal::SamplerMinMagFilter::Linear;
        d
    };
    let nearest = SamplerDescriptor::new();
    let compare = {
        let mut d = SamplerDescriptor::new();
        d.compare_function = crate::metal::CompareFunction::Less;
        d
    };
    let listed = |descs: &[SamplerDescriptor]| -> Vec<crate::bind::ArgumentSampler> {
        descs
            .iter()
            .enumerate()
            .map(|(i, &d)| {
                crate::bind::ArgumentSampler::new(d, i as u64 + 1).at_element(i as u32 + 1)
            })
            .collect()
    };
    let planned = PlannedHandleTable {
        set: 0,
        binding: 4,
        stage: Stage::Kernel,
        entries: 4,
    };
    let encoded = listed(&[linear, nearest]);
    let template = ResourceTemplate {
        sampler_tables: vec![planned],
        argument_samplers: Some(ArgumentSamplersShape::of(&encoded)),
        sampler_admission: SamplerAdmission {
            classes: vec![HandleClass::Sampler {
                compare: false,
                integer: false,
            }],
            cubic: true,
            opaque_black: false,
        },
        ..ResourceTemplate::default()
    };
    let profile = m3();
    let relisted_sampler_tables = |template: &ResourceTemplate,
                                   samplers: &[crate::bind::ArgumentSampler],
                                   placement: HeapPlacement| {
        relisted_sampler_tables(template, &profile, samplers, placement)
    };
    let placement = HeapPlacement::ResourceId;
    let state = |samplers: Vec<crate::bind::ArgumentSampler>| BindState {
        heap_placement: placement,
        argument_samplers: samplers,
        ..BindState::default()
    };
    // The encoded list, and one more of a planned shape: the tables the plan writes for them.
    for samplers in [encoded.clone(), listed(&[linear, nearest, linear])] {
        let relisted = relisted_sampler_tables(&template, &samplers, placement)
            .expect("relisted")
            .expect("held");
        let written =
            listed_sampler_tables(&[planned], &state(samplers).argument_samplers, placement)
                .expect("written");
        assert_eq!(relisted, written);
    }
    let three = relisted_sampler_tables(&template, &listed(&[linear, nearest, linear]), placement)
        .expect("relisted")
        .expect("held");
    let ids: Vec<(u32, u32)> = three[0]
        .bytes
        .chunks(16)
        .map(|e| {
            let word = |k: usize| u32::from_le_bytes(e[4 * k..4 * k + 4].try_into().unwrap());
            (word(0), word(2))
        })
        .collect();
    assert_eq!(ids, vec![(0, 0), (1, 1), (2, 2), (3, 3)]);
    // Past the entries: not the plan's.
    let past = listed(&[linear, nearest, linear, nearest]);
    assert_eq!(
        relisted_sampler_tables(&template, &past, placement).expect("past"),
        None
    );
    // A shape the plan was not made for, which its checks pass: listed as the plan writes it.
    let shaped = listed(&[linear, compare]);
    assert_eq!(
        relisted_sampler_tables(&template, &shaped, placement).expect("shaped"),
        Some(listed_sampler_tables(&[planned], &shaped, placement).expect("written"))
    );
    // A plan without sampler tables.
    let bare = ResourceTemplate::default();
    assert_eq!(
        relisted_sampler_tables(&bare, &encoded, placement).expect("bare"),
        None
    );
}

/// A cubic state made after a command was encoded, beside stages that all test their argument
/// samplers for a cubic filter, is relisted with `SamplerWord::CUBIC` in its flags, which the
/// stage's test reads, whether the plan's list held a cubic state or not. Beside a stage that does
/// not test (a fragment stage's table) a new cubic shape is not the plan's (`None`: the runtime
/// keeps the states listed before, announcing `unlisted-argument-sampler`), as the plan itself
/// refuses a listed one there.
#[test]
fn a_late_cubic_sampler_is_listed_with_its_cubic_bit_beside_tested_stages() {
    use crate::metal::sampler::SamplerWord;
    let linear = {
        let mut d = SamplerDescriptor::new();
        d.min_filter = crate::metal::SamplerMinMagFilter::Linear;
        d
    };
    let cubic = {
        let mut d = SamplerDescriptor::new();
        d.min_filter = crate::metal::SamplerMinMagFilter::Cubic;
        d.mag_filter = crate::metal::SamplerMinMagFilter::Cubic;
        d
    };
    let listed = |descs: &[SamplerDescriptor]| -> Vec<crate::bind::ArgumentSampler> {
        descs
            .iter()
            .enumerate()
            .map(|(i, &d)| {
                crate::bind::ArgumentSampler::new(d, i as u64 + 1).at_element(i as u32 + 1)
            })
            .collect()
    };
    let planned = PlannedHandleTable {
        set: 0,
        binding: 4,
        stage: Stage::Kernel,
        entries: 4,
    };
    let template = |encoded: &[crate::bind::ArgumentSampler], tested: bool| ResourceTemplate {
        sampler_tables: vec![planned],
        argument_samplers: Some(ArgumentSamplersShape::of(encoded)),
        sampler_admission: SamplerAdmission {
            classes: vec![HandleClass::Sampler {
                compare: false,
                integer: false,
            }],
            cubic: tested,
            opaque_black: false,
        },
        ..ResourceTemplate::default()
    };
    let profile = m3();
    let placement = HeapPlacement::ResourceId;
    let flags = |tables: Vec<crate::bind::instantiate::DescriptorBytes>| -> Vec<(u32, u32)> {
        tables[0]
            .bytes
            .chunks(16)
            .map(|e| {
                let word = |k: usize| u32::from_le_bytes(e[4 * k..4 * k + 4].try_into().unwrap());
                (word(0), word(3))
            })
            .collect()
    };
    let cubic_word = SamplerWord::of(&cubic).0;
    assert_eq!(cubic_word & SamplerWord::CUBIC, SamplerWord::CUBIC);
    let linear_word = SamplerWord::of(&linear).0;
    // Encoded without a cubic state, beside tested stages: a late one is listed, its flags cubic.
    let plain = template(&listed(&[linear]), true);
    let tables = relisted_sampler_tables(&plain, &profile, &listed(&[linear, cubic]), placement)
        .expect("late")
        .expect("held");
    assert_eq!(
        flags(tables),
        vec![(0, 0), (0, 0), (1, linear_word), (2, cubic_word)]
    );
    // Beside a stage that does not test: not the plan's.
    let untested = template(&listed(&[linear]), false);
    assert_eq!(
        relisted_sampler_tables(&untested, &profile, &listed(&[linear, cubic]), placement)
            .expect("late"),
        None
    );
    // Encoded with one: a late one of its shape is relisted, its flags cubic.
    let with = template(&listed(&[linear, cubic]), true);
    let tables =
        relisted_sampler_tables(&with, &profile, &listed(&[linear, cubic, cubic]), placement)
            .expect("late")
            .expect("held");
    assert_eq!(
        flags(tables),
        vec![(0, 0), (1, linear_word), (2, cubic_word), (3, cubic_word)]
    );
}

/// A sampler state of a shape the plan was not made for is held as a command buffer commits where
/// the plan's checks of a listed shape pass: `check_sampler` for every use of its sampler arrays
/// (invalid LOD clamps refuse; an anisotropic state passes beside a cube's explicit gradients,
/// whose use's sampler is isotropic, and, on a device without `samplerAnisotropy`, there alone),
/// and an opaque-black border only where the members were checked against one. A plan whose arrays
/// sample nothing (a function that only names the states) holds any shape.
#[test]
fn sampler_tables_admit_the_shapes_the_plan_s_checks_pass() {
    let profile = m3();
    let floor =
        crate::pipeline::testing::profile("vk12-floor", crate::device::DevicePolicy::default())
            .expect("profile");
    let placement = HeapPlacement::ResourceId;
    let planned = PlannedHandleTable {
        set: 0,
        binding: 4,
        stage: Stage::Kernel,
        entries: 4,
    };
    let encoded = [crate::bind::ArgumentSampler::new(SamplerDescriptor::new(), 1).at_element(1)];
    let template = |classes: Vec<HandleClass>, opaque_black: bool| ResourceTemplate {
        sampler_tables: vec![planned],
        argument_samplers: Some(ArgumentSamplersShape::of(&encoded)),
        sampler_admission: SamplerAdmission {
            classes,
            cubic: true,
            opaque_black,
        },
        ..ResourceTemplate::default()
    };
    let class = HandleClass::Sampler {
        compare: false,
        integer: false,
    };
    let held_on =
        |profile: &DeviceProfile, template: &ResourceTemplate, late: SamplerDescriptor| -> bool {
            let mut samplers = encoded.to_vec();
            samplers.push(crate::bind::ArgumentSampler::new(late, 2).at_element(2));
            assert!(!template
                .argument_samplers
                .as_ref()
                .expect("shape")
                .shapes
                .contains(&late.shape()));
            relisted_sampler_tables(template, profile, &samplers, placement)
                .expect("relisted")
                .is_some()
        };
    let held = |template: &ResourceTemplate, late: SamplerDescriptor| -> bool {
        held_on(&profile, template, late)
    };
    let lod = {
        let mut d = SamplerDescriptor::new();
        d.lod_min_clamp = 2.0;
        d.lod_max_clamp = 1.0;
        d
    };
    let anisotropic = {
        let mut d = SamplerDescriptor::new();
        d.min_filter = crate::metal::SamplerMinMagFilter::Linear;
        d.mag_filter = crate::metal::SamplerMinMagFilter::Linear;
        d.mip_filter = crate::metal::SamplerMipFilter::Linear;
        d.max_anisotropy = 4;
        d
    };
    let black = {
        let mut d = SamplerDescriptor::new();
        d.s_address_mode = crate::metal::SamplerAddressMode::ClampToBorderColor;
        d.border_color = crate::metal::SamplerBorderColor::OpaqueBlack;
        d
    };
    assert!(anisotropic.shape().anisotropic);
    assert!(black.shape().opaque_black_border);
    let sampled = template(vec![class], false);
    assert!(!held(&sampled, lod), "invalid LOD clamps");
    assert!(held(&sampled, anisotropic), "anisotropic");
    assert!(!held(&sampled, black), "members unchecked for the border");
    // Without `samplerAnisotropy` the plan's check refuses an anisotropic state.
    assert!(
        !floor
            .fields()
            .enabled
            .features
            .contains(&crate::device::requirements::VkFeature::SamplerAnisotropy),
        "vk12-floor states samplerAnisotropy"
    );
    assert!(
        !held_on(&floor, &sampled, anisotropic),
        "anisotropic without samplerAnisotropy"
    );
    let checked = template(vec![class], true);
    assert!(held(&checked, black), "members checked for the border");
    let naming = template(Vec::new(), false);
    for late in [lod, anisotropic, black] {
        assert!(held(&naming, late), "{late:?}: nothing sampled");
    }
}

/// A plan keyed by `BindShape::for_plan` states a heap member at one level; the bind state's
/// texture of more levels is that member (a heap array views level 0 alone), and one of another
/// format is not.
#[test]
fn a_member_s_level_count_is_not_the_plan_s() {
    let mut facts = TextureFacts::new(TextureType::Type2D, PixelFormat::RGBA8Unorm);
    facts.usage = READ;
    facts.mipmap_level_count = 3;
    let mut used = UsedResource::new(ResourceClass::Texture(facts), 7);
    used.resource_id = Some(42);
    let mut state = BindState::default();
    state.used_resources.push(used);
    let by_token: UsedByToken<'_> =
        BTreeMap::from([(7, state.used_resources.iter().collect::<Vec<_>>())]);
    let member = |shape: TextureShape| HeapMember {
        source: HeapSource::Used(7),
        shape,
        stages: vec![Stage::Kernel],
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
/// (its raw format), the float class holds its neutral (`Fit::Raw`), and `heap_element` answers
/// both so: the device-wide set's elements, which a consumer writes for every live texture, hold
/// the raw view and the neutral. A plan's float array writes the neutral, or refuses it where a
/// stage that enters it samples heap textures. No class views any other extended-range texture (an
/// `_sRGB` twin, a two-plane form, a swizzled or multisampled one), as before.
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
    let others: Vec<PixelFormat> = PixelFormat::ALL
        .iter()
        .copied()
        .filter(|&pf| ExtendedRange::of(pf).is_some_and(|f| f.heap_read().is_none()))
        .collect();
    assert!(!others.is_empty());
    let profiles = committed_profiles().expect("profiles");
    for (stem, p) in &profiles {
        let image = |c: ImageHandleClass| HandleClass::Image(c);
        for &pf in &read {
            let s = shape(TextureType::Type2D, pf, 1, READ);
            assert_eq!(fit(&s, &f2d), Fit::Raw, "{pf:?}");
            assert_eq!(fit(&s, &u2d), Fit::Holds, "{pf:?}");
            assert_eq!(fit(&s, &i2d), Fit::NumericType, "{pf:?}");
            let facts = TextureFacts::new(TextureType::Type2D, pf);
            let mut slot = crate::bind::plan::resources::sampled_view(p, pf, &u2d)
                .unwrap_or_else(|e| panic!("{stem} {pf:?}: {e}"));
            slot.layout = layout;
            assert_eq!(
                heap_element(p, &image(u2d), &facts, layout),
                Ok(HeapElement::View(slot)),
                "{stem} {pf:?}"
            );
            assert_eq!(slot.format, ExtendedRange::of(pf).unwrap().raw_format());
            assert_eq!(
                heap_element(p, &image(f2d), &facts, layout),
                Ok(HeapElement::Neutral),
                "{stem} {pf:?}"
            );
            assert_eq!(
                heap_element(p, &image(i2d), &facts, layout),
                Ok(HeapElement::Neutral)
            );
            // Swizzled or multisampled: viewed by no class, as before.
            let mut swizzled = facts;
            swizzled.swizzle.red = crate::metal::TextureSwizzle::Zero;
            for c in [f2d, u2d] {
                let got = heap_element(p, &image(c), &swizzled, layout);
                assert!(!matches!(got, Ok(HeapElement::View(_))), "{stem} {pf:?}");
            }
            let mut ms = TextureFacts::new(TextureType::Type2DMultisample, pf);
            ms.sample_count = 4;
            let ms_class = |ty| class(ImageDim::D2, false, true, ty, Sampled);
            for ty in [Float, Uint] {
                let got = heap_element(p, &image(ms_class(ty)), &ms, layout);
                assert!(!matches!(got, Ok(HeapElement::View(_))), "{stem} {pf:?}");
            }
        }
        for &pf in &others {
            let facts = TextureFacts::new(TextureType::Type2D, pf);
            for c in [f2d, u2d] {
                let got = heap_element(p, &image(c), &facts, layout);
                assert!(!matches!(got, Ok(HeapElement::View(_))), "{stem} {pf:?}");
            }
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
        sampler_tables: Vec::new(),
        samplers: None,
        directories: Vec::new(),
        snapshots: Vec::new(),
        addressed_tables: false,
        function_tables: None,
        zero_bytes: 0,
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
                    sampler_tables: Vec::new(),
                    samplers: None,
                    directories: Vec::new(),
                    snapshots: Vec::new(),
                    addressed_tables: false,
                    function_tables: None,
                    zero_bytes: 0,
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
