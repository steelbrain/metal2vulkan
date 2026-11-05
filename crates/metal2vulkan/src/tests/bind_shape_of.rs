//! `BindState::has_shape` answers `state.shape() == *shape` without making the shape: over states
//! that each differ from a base in one field of the shape (and some in fields the shape leaves
//! out, or in two ways that give one shape), for every pair of them.

use crate::bind::{
    ArgumentFunctionTable, ArgumentSampler, BindState, BoundBuffer, BoundSampler, BoundTexture,
    CompensatedViewport, FragmentTests, ImageblockSize, ResourceClass, SamplerDescriptor,
    StageTable, TessellationFactorBuffer, TessellationFactors, TextureFacts,
    TileThreadgroupArgument, TileThreadgroupMemory, UsedResource, VertexAmplification,
};
use crate::metal::{
    CompareFunction, PixelFormat, RenderStages, ResourceUsage, SamplerMinMagFilter, TextureType,
};
use crate::pipeline::lower_inputs::ViewportCompensation;
use crate::pipeline::sym::Stage;

/// A buffer of `length` bytes bound at `offset`.
fn buffer(length: u64, offset: u64) -> BoundBuffer {
    BoundBuffer::Buffer {
        length,
        offset,
        gpu_address: 0x1000,
        device_address: None,
        attribute_stride: None,
    }
}

/// A 2D texture of `format`.
fn texture(format: PixelFormat) -> BoundTexture {
    BoundTexture::Texture {
        facts: TextureFacts::new(TextureType::Type2D, format),
        resource_id: Some(7),
    }
}

/// A sampler that filters linearly where `linear`, and compares where `compares`.
fn sampler(linear: bool, compares: bool) -> SamplerDescriptor {
    let mut desc = SamplerDescriptor::new();
    if linear {
        desc.min_filter = SamplerMinMagFilter::Linear;
        desc.mag_filter = SamplerMinMagFilter::Linear;
    }
    if compares {
        desc.compare_function = CompareFunction::Less;
    }
    desc
}

/// The base: a vertex buffer, a fragment texture and sampler, and one argument sampler.
fn base() -> BindState {
    let mut state = BindState::default();
    let mut vertex = StageTable::default();
    vertex.buffers.insert(0, buffer(256, 0));
    state.stages.insert(Stage::Vertex, vertex);
    let mut fragment = StageTable::default();
    fragment
        .textures
        .insert(0, texture(PixelFormat::RGBA8Unorm));
    fragment
        .samplers
        .insert(0, BoundSampler::new(sampler(false, false)));
    state.stages.insert(Stage::Fragment, fragment);
    state
        .argument_samplers
        .push(ArgumentSampler::new(sampler(false, false), 1));
    state
}

/// The base changed by `change`.
fn with(change: impl FnOnce(&mut BindState)) -> BindState {
    let mut state = base();
    change(&mut state);
    state
}

/// `state`'s vertex table.
fn vertex(state: &mut BindState) -> &mut StageTable {
    state.stages.get_mut(&Stage::Vertex).unwrap()
}

/// `state`'s fragment table.
fn fragment(state: &mut BindState) -> &mut StageTable {
    state.stages.get_mut(&Stage::Fragment).unwrap()
}

/// Each state.
fn states() -> Vec<BindState> {
    let used = |token, usage| {
        let mut used = UsedResource::new(ResourceClass::Buffer, token);
        used.usage = usage;
        used.stages = RenderStages::FRAGMENT;
        used
    };
    vec![
        BindState::default(),
        base(),
        // Of the base's shape: another offset of the same alignment, another length of the
        // same class, an empty stage table, a per-draw value, and a function table (the tables'
        // shape is their capacity and byte classes, which one small table leaves at none's).
        with(|s| {
            vertex(s).buffers.insert(0, buffer(255, 256));
        }),
        with(|s| {
            s.stages.insert(Stage::Mesh, StageTable::default());
        }),
        with(|s| s.viewport_count = 4),
        with(|s| {
            s.function_tables.push(ArgumentFunctionTable::new(5, 4));
        }),
        // A buffer of another length class, offset alignment or class, or at another slot.
        with(|s| {
            vertex(s).buffers.insert(0, buffer(1 << 20, 0));
        }),
        with(|s| {
            vertex(s).buffers.insert(0, buffer(256, 2));
        }),
        // A view base alignment no slot tells apart, or a slot that tells apart none, or one
        // that tells it apart at an offset it does not divide, all of the base's shape; one that
        // tells it apart where it divides the offset, of another.
        with(|s| s.view_base_alignment = 64),
        with(|s| vertex(s).at_bind_views = 1),
        with(|s| {
            s.view_base_alignment = 64;
            vertex(s).at_bind_views = 1;
            vertex(s).buffers.insert(0, buffer(256, 16));
        }),
        with(|s| {
            s.view_base_alignment = 64;
            vertex(s).at_bind_views = 1;
        }),
        with(|s| {
            vertex(s).buffers.insert(
                0,
                BoundBuffer::Bytes {
                    len: 16,
                    attribute_stride: None,
                },
            );
        }),
        with(|s| {
            let moved = vertex(s).buffers.remove(&0).unwrap();
            vertex(s).buffers.insert(1, moved);
        }),
        with(|s| {
            vertex(s).buffers.insert(1, buffer(256, 0));
        }),
        with(|s| {
            vertex(s).buffers.clear();
        }),
        // A texture of another format, a sampler of another shape, a threadgroup length.
        with(|s| {
            fragment(s)
                .textures
                .insert(0, texture(PixelFormat::BGRA8Unorm));
        }),
        with(|s| {
            fragment(s)
                .samplers
                .insert(0, BoundSampler::new(sampler(true, false)));
        }),
        with(|s| {
            fragment(s).threadgroup_lengths.insert(0, 64);
        }),
        // Used resources, a token listed twice (its usages unioned) and its union listed once.
        with(|s| s.used_resources.push(used(3, ResourceUsage::READ))),
        with(|s| {
            s.used_resources.push(used(3, ResourceUsage::READ));
            s.used_resources.push(used(3, ResourceUsage::WRITE));
        }),
        with(|s| {
            s.used_resources
                .push(used(3, ResourceUsage::READ | ResourceUsage::WRITE))
        }),
        // Argument samplers: another of the same shape (same class), of another shape, none,
        // and enough to take the next class.
        with(|s| {
            s.argument_samplers
                .push(ArgumentSampler::new(sampler(false, false), 2))
        }),
        with(|s| {
            s.argument_samplers
                .push(ArgumentSampler::new(sampler(false, true), 2))
        }),
        with(|s| {
            s.argument_samplers[0] = ArgumentSampler::new(sampler(false, true), 1);
        }),
        with(|s| s.argument_samplers.clear()),
        with(|s| {
            for id in 2..20 {
                s.argument_samplers
                    .push(ArgumentSampler::new(sampler(false, false), id));
            }
        }),
        // Past the samplers answered without making their shape, the same shape and another.
        with(|s| {
            for id in 2..80 {
                s.argument_samplers
                    .push(ArgumentSampler::new(sampler(false, false), id));
            }
        }),
        with(|s| {
            for id in 2..80 {
                s.argument_samplers
                    .push(ArgumentSampler::new(sampler(id == 79, false), id));
            }
        }),
        // Enough function tables to take the next class.
        with(|s| {
            for id in 0..16 {
                s.function_tables
                    .push(ArgumentFunctionTable::new(5 + id, 4));
            }
        }),
        with(|s| s.vertex_amplification = VertexAmplification::new(2, Vec::new())),
        with(|s| s.imageblock_size = ImageblockSize::new(32, 32)),
        with(|s| s.render_target_array_length = 4),
        with(|s| {
            let buffer = TessellationFactorBuffer::new(64, 0, 0);
            s.tessellation_factors = TessellationFactors::new(Some(buffer), 1.0);
        }),
        with(|s| s.render_area = [64, 64]),
        with(|s| {
            let mut memory = TileThreadgroupMemory::new(256);
            memory
                .arguments
                .insert(0, TileThreadgroupArgument::new(64, 0));
            s.tile_threadgroup = memory;
        }),
        with(|s| {
            s.viewport_compensation = Some(CompensatedViewport {
                compensation: ViewportCompensation {
                    depth: true,
                    ..ViewportCompensation::default()
                },
                ..CompensatedViewport::default()
            });
        }),
        with(|s| {
            s.zeroed_colors.insert(1, PixelFormat::RGBA8Unorm);
        }),
        with(|s| {
            s.dropped_colors.insert(1);
        }),
        with(|s| s.kept_imageblock_bytes = Some(8)),
        with(|s| s.depth_clamp_range = Some([0.0, 1.0])),
        with(|s| {
            s.interlocked_colors.insert(1);
        }),
        with(|s| {
            s.extended_shadows.insert(0);
        }),
        with(|s| s.fragment_tests = FragmentTests::Pass),
        with(|s| s.quad_primitive_ids = true),
        with(|s| s.private_list = true),
    ]
}

#[test]
fn a_state_has_exactly_the_shape_it_makes() {
    let states = states();
    let shapes: Vec<_> = states.iter().map(BindState::shape).collect();
    let mut same = 0;
    for (at, state) in states.iter().enumerate() {
        assert!(state.has_shape(&shapes[at]), "state {at}: {state:?}");
        for (of, shape) in shapes.iter().enumerate() {
            let equal = shapes[at] == *shape;
            same += usize::from(equal && at != of);
            assert_eq!(
                state.has_shape(shape),
                equal,
                "state {at} against the shape of state {of}"
            );
        }
    }
    // The states of the base's shape (the base, the four its comment names, the three of a view
    // base alignment told apart nowhere it divides an offset, and the base with another argument
    // sampler of the same shape) and the used resources' two of one shape: each against the
    // others of its shape.
    assert_eq!(same, 9 * 8 + 2, "pairs of one shape");
}

/// A buffer's bind offset cycling through the multiples of 16 up to 256 at a view base alignment
/// of 64 keeps one shape where its slot tells no alignment past 4 apart (vertex data, or a view
/// a runtime left out), and takes two where the slot tells it apart (a uniform view: 64 divides
/// the offset or not): none past those two, and none new on a second cycle.
#[test]
fn a_slot_s_offsets_moving_among_multiples_of_16_keep_their_shapes() {
    for (views, shapes) in [(0, 1), (1, 2)] {
        let mut seen = Vec::new();
        for cycle in 0..2 {
            for offset in (16..=256).step_by(16) {
                let state = with(|s| {
                    s.view_base_alignment = 64;
                    vertex(s).at_bind_views = views;
                    vertex(s).buffers.insert(0, buffer(4096, offset));
                });
                let shape = state.shape();
                if !seen.contains(&shape) {
                    assert_eq!(
                        cycle, 0,
                        "a new shape at offset {offset} on the second cycle"
                    );
                    seen.push(shape);
                }
            }
        }
        assert_eq!(seen.len(), shapes, "slots telling apart {views:#b}");
    }
}

/// A state that lists used resources is found among shapes that each match it in every other
/// field, and differ in what it lists but the last, with its used resources' shape made once
/// (`position_of_shape`), as allocating as one comparison (`has_shape`) is: compared one by one,
/// each comparison makes it again.
#[test]
fn a_state_s_shape_is_found_among_many_with_its_used_resources_made_once() {
    use super::constant_blob_footprint::allocations_during;
    const SHAPES: u32 = 4;
    let listing = |tokens: std::ops::Range<u32>| {
        let mut state = base();
        for token in tokens {
            let mut used = UsedResource::new(ResourceClass::Buffer, token);
            used.usage = ResourceUsage::READ;
            used.stages = RenderStages::FRAGMENT;
            state.used_resources.push(used);
        }
        state
    };
    // Each lists 16 resources; the state lists the last's.
    let shapes: Vec<_> = (0..SHAPES)
        .map(|at| listing(at * 16..at * 16 + 16).shape())
        .collect();
    let state = listing((SHAPES - 1) * 16..SHAPES * 16);
    let last = SHAPES as usize - 1;

    let (found, one_by_one) =
        allocations_during(|| shapes.iter().position(|shape| state.has_shape(shape)));
    assert_eq!(found, Some(last));
    let (found, at_once) = allocations_during(|| state.position_of_shape(&shapes));
    assert_eq!(found, Some(last));
    let (has, one) = allocations_during(|| state.has_shape(&shapes[last]));
    assert!(has);
    assert!(one > 0, "a used resources' shape is made");
    assert_eq!(
        at_once, one,
        "found among {SHAPES}: as allocating as one comparison"
    );
    assert_eq!(one_by_one, one * u64::from(SHAPES), "one by one: one each");
    // A state that lists none makes nothing; nor does one whose shape differs in another field.
    let none = base();
    let (found, made) = allocations_during(|| none.position_of_shape(&shapes));
    assert_eq!((found, made), (None, 0), "a state that lists none");
    let mut other = listing(0..16);
    other.render_target_array_length = 2;
    let (found, made) = allocations_during(|| other.position_of_shape(&shapes));
    assert_eq!((found, made), (None, 0), "a state of another field");
}

/// A state that lists its used resources in token order (`BindState::used_in_token_order`)
/// compares them with a shape's in place, making nothing: over listings that differ in a token, a
/// usage, a stage, a class, a token listed twice (whose entries union, the first's class kept) or
/// one listed more or less, each answers as the same listing in no order does, against the shape
/// of every other, and is found among them all without allocating.
#[test]
fn a_state_listed_in_token_order_compares_its_used_resources_in_place() {
    use super::constant_blob_footprint::allocations_during;
    let entry = |class, token, usage, stages| {
        let mut used = UsedResource::new(class, token);
        used.usage = usage;
        used.stages = stages;
        used
    };
    let buffer = ResourceClass::Buffer;
    let read = |token| entry(buffer, token, ResourceUsage::READ, RenderStages::FRAGMENT);
    let listing = |change: &dyn Fn(&mut Vec<UsedResource>)| {
        let mut state = base();
        state.used_resources = (0..16).map(read).collect();
        change(&mut state.used_resources);
        state
    };
    let facts = |format| ResourceClass::Texture(TextureFacts::new(TextureType::Type2D, format));
    let unorm = facts(PixelFormat::RGBA8Unorm);
    let states = [
        listing(&|_| {}),
        listing(&|used| used[15].token = 40),
        listing(&|used| used[5].usage = ResourceUsage::WRITE),
        listing(&|used| used[5].stages = RenderStages::VERTEX),
        listing(&|used| used[7].class = unorm),
        listing(&|used| used[7].class = facts(PixelFormat::R32Float)),
        listing(&|used| used.insert(8, read(8))),
        listing(&|used| {
            used.insert(
                9,
                entry(buffer, 8, ResourceUsage::WRITE, RenderStages::VERTEX),
            )
        }),
        listing(&|used| {
            used.insert(
                9,
                entry(unorm, 8, ResourceUsage::WRITE, RenderStages::VERTEX),
            )
        }),
        listing(&|used| {
            used[8].usage = ResourceUsage::READ | ResourceUsage::WRITE;
            used[8].stages = RenderStages::VERTEX | RenderStages::FRAGMENT;
        }),
        listing(&|used| {
            used.remove(10);
        }),
        listing(&|used| used.push(read(16))),
        listing(&|used| used.truncate(1)),
    ];
    let shapes: Vec<_> = states.iter().map(BindState::shape).collect();
    let mut same = 0;
    for (at, state) in states.iter().enumerate() {
        let mut ordered = state.clone();
        ordered.used_in_token_order = true;
        for (of, shape) in shapes.iter().enumerate() {
            let equal = shapes[at] == *shape;
            same += usize::from(equal && at != of);
            let (has, made) = allocations_during(|| ordered.has_shape(shape));
            assert_eq!(has, equal, "state {at} in token order, shape {of}");
            assert_eq!(
                has,
                state.has_shape(shape),
                "state {at} in no order, shape {of}"
            );
            assert_eq!(made, 0, "state {at} in token order, shape {of}: made");
        }
        let (found, made) = allocations_during(|| ordered.position_of_shape(&shapes));
        assert_eq!(found, state.position_of_shape(&shapes), "state {at}: found");
        assert_eq!(made, 0, "state {at}: found among them all, made");
    }
    // A token listed twice with the same usage and stages is the listing's shape (one pair); one
    // listed again with a vertex write, as a buffer or as a texture, is the one listed once that
    // reads and writes it in both stages (three of one shape).
    assert_eq!(same, 2 + 3 * 2, "pairs of one shape");
}
