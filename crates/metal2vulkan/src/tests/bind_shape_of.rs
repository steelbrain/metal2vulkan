//! `BindState::has_shape` answers `state.shape() == *shape` without making the shape: over states
//! that each differ from a base in one field of the shape (and some in fields the shape leaves
//! out, or in two ways that give one shape), for every pair of them. `BoundBuffer::same_shape`
//! answers as two bindings' shapes compare, over every pair of a set of bindings.

use crate::bind::{
    ArgumentFunctionTable, ArgumentSampler, BindState, BoundBuffer, BoundSampler, BoundTexture,
    BufferRanges, FragmentTests, ImageblockSize, ResourceClass, SamplerDescriptor, StageTable,
    TessellationFactorBuffer, TessellationFactors, TextureFacts, TileThreadgroupArgument,
    TileThreadgroupMemory, UsedResource, VertexAmplification,
};
use crate::metal::{
    CompareFunction, PixelFormat, RenderStages, ResourceUsage, SamplerMinMagFilter, TextureType,
};
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

/// Buffer range limits of `uniform` and `storage` bytes.
fn ranges(uniform: u32, storage: u32) -> BufferRanges {
    BufferRanges { uniform, storage }
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
        // Buffer range limits that tell the buffer's class apart, of the base's shape; limits of
        // 4 GiB, which tell apart no class of either length, of one other shape twice.
        with(|s| vertex(s).buffer_ranges = ranges(255, 255)),
        with(|s| vertex(s).buffer_ranges = ranges(u32::MAX, u32::MAX)),
        with(|s| {
            vertex(s).buffer_ranges = ranges(u32::MAX, u32::MAX);
            vertex(s).buffers.insert(0, buffer(1 << 20, 0));
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
    // The states of the base's shape (the base, the four its comment names, the one of buffer
    // range limits that tell its class apart, the three of a view base alignment told apart
    // nowhere it divides an offset, and the base with another argument sampler of the same
    // shape), the two of 4 GiB limits and the used resources' two of one shape: each against the
    // others of its shape.
    assert_eq!(same, 10 * 9 + 2 + 2, "pairs of one shape");
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

/// `BoundBuffer::same_shape` answers `a.shape(x, r) == b.shape(x, r)` for every pair of bindings
/// over lengths of one length class and of others, bind offsets of one alignment class and of
/// others, with and without a device address and a stride, and inline bytes, at view base
/// alignments that tell no offset class past the vertex fetch's apart and one that does, under
/// buffer range limits that tell every length class apart, some, and few.
#[test]
fn a_buffer_binding_has_the_shape_of_another_exactly_where_their_shapes_compare_equal() {
    let mut bindings = Vec::new();
    for length in [1, 16, 17, 4096, u64::MAX] {
        for offset in [0, 2, 4, 6, 64, 96] {
            for device_address in [None, Some(0x2000)] {
                for attribute_stride in [None, Some(4), Some(6)] {
                    bindings.push(BoundBuffer::Buffer {
                        length,
                        offset,
                        gpu_address: 0x1000,
                        device_address,
                        attribute_stride,
                    });
                }
            }
        }
    }
    for length in [1 << 16, (1 << 16) + 1, 1 << 27, 1 << 33] {
        bindings.push(buffer(length, 0));
    }
    for len in [4, 8] {
        for attribute_stride in [None, Some(4)] {
            bindings.push(BoundBuffer::Bytes {
                len,
                attribute_stride,
            });
        }
    }
    let (mut alike, mut apart) = (0, 0);
    let limits = [
        ranges(0, 0),
        ranges(1 << 16, 1 << 27),
        ranges(u32::MAX, u32::MAX),
    ];
    for (alignment, r) in [0, 4, 64].into_iter().flat_map(|a| limits.map(|r| (a, r))) {
        for (at, binding) in bindings.iter().enumerate() {
            for (of, other) in bindings.iter().enumerate() {
                let equal = binding.shape(alignment, r) == other.shape(alignment, r);
                assert_eq!(
                    binding.same_shape(other, alignment, r),
                    equal,
                    "binding {at} against binding {of} at alignment {alignment} under {r:?}"
                );
                alike += usize::from(equal && at != of);
                apart += usize::from(!equal);
            }
        }
    }
    assert!(alike > 0 && apart > 0, "{alike} pairs alike, {apart} apart");
}
