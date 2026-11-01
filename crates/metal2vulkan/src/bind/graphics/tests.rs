//! Draw mapping, vertex-buffer windows, bind rules, and draw uses on synthetic templates. Render
//! plans of compiled pipelines are tested end to end in `src/tests/engine_graphics_binder.rs`.

use super::*;
use crate::bind::plan::{PlannedBuffer, ResourceTemplate};
use crate::bind::state::{length_class, StageTable};
use crate::pipeline::interface::DescriptorView;
use std::cell::RefCell;

thread_local! {
    /// The commands every [`instantiate`] of a test writes over, from one call to the next.
    static KEPT: RefCell<DrawCommands> = RefCell::new(DrawCommands::default());
}

/// The commands of `draw` ([`super::instantiate`]) written over no draw's, which must be what
/// writing them over those of every draw of the test before it gives (and its error the same).
fn instantiate(
    t: &GraphicsTemplate,
    state: &BindState,
    draw: &DrawCall,
) -> Result<DrawCommands, BindError> {
    let mut fresh = DrawCommands::default();
    let made = super::instantiate(t, state, draw, &mut fresh).map(|()| fresh);
    KEPT.with(|kept| {
        let mut kept = kept.borrow_mut();
        let over = super::instantiate(t, state, draw, &mut kept).map(|()| kept.clone());
        assert_eq!(over, made, "the commands written over those before");
    });
    made
}

/// The draw's own resources ([`super::draw_uses`]), in a list of their own.
fn draw_uses(
    draw: &DrawCall,
    overrun: Option<IndexOverrun>,
    factors: Option<&FactorTemplate>,
) -> Vec<ResourceUse> {
    let mut uses = Vec::new();
    super::draw_uses(&mut uses, draw, overrun, factors);
    uses
}

/// Every Metal draw maps to the Vulkan draw of the same arguments, and the base-vertex push field
/// reports Vulkan's `BaseVertex` (first vertex, or the vertex offset of an indexed draw) and
/// `BaseInstance`; indirect draws report zero (their arguments stay on the GPU).
#[test]
fn draws_map_to_the_vulkan_draw_of_the_same_arguments() {
    let (d, base) = vk_draw(
        &DrawCall::Primitives {
            vertex_start: 7,
            vertex_count: 3,
            instance_count: 4,
            base_instance: 2,
        },
        None,
    );
    assert_eq!(
        d,
        vk::DrawCommand::Draw {
            vertex_count: 3,
            instance_count: 4,
            first_vertex: 7,
            first_instance: 2
        }
    );
    assert_eq!(base, [7, 2]);
    let (d, base) = vk_draw(
        &DrawCall::Indexed {
            index_type: vk::IndexType::UINT32,
            index_buffer: IndexBufferRef::new(1, 48),
            index_count: 9,
            index_offset: 12,
            instance_count: 1,
            base_vertex: -3,
            base_instance: 5,
        },
        None,
    );
    assert_eq!(
        d,
        vk::DrawCommand::DrawIndexed {
            index_type: vk::IndexType::UINT32,
            index_buffer_offset: 12,
            index_count: 9,
            instance_count: 1,
            first_index: 0,
            vertex_offset: -3,
            first_instance: 5
        }
    );
    assert_eq!(base, [(-3i32) as u32, 5]);
    let (d, base) = vk_draw(
        &DrawCall::Indirect {
            indirect: IndirectRef::new(2, 40),
            indexed: Some((vk::IndexType::UINT16, IndexBufferRef::new(3, 64), 8)),
        },
        None,
    );
    assert_eq!(
        d,
        vk::DrawCommand::DrawIndexedIndirect {
            index_type: vk::IndexType::UINT16,
            index_buffer_offset: 8,
            offset: 40,
            draw_count: 1,
            stride: 20
        }
    );
    assert_eq!(base, [0, 0]);
}

/// A render plan names only vertex and fragment stages; a tessellation pipeline's vertex
/// function is its evaluation stage, and its synthesized vertex and control stages are none, as
/// are the geometry stages of barycentric weights read as varyings and a band's control and
/// evaluation stages.
#[test]
fn only_vertex_and_fragment_stages_are_render_stages() {
    use vk::ShaderStage as S;
    let stages = |modules: &[S], tessellated| {
        metal_stages(modules.iter().copied(), (tessellated, false), false)
    };
    let tessellation = [
        S::VERTEX,
        S::TESSELLATION_CONTROL,
        S::TESSELLATION_EVALUATION,
        S::FRAGMENT,
    ];
    assert_eq!(
        metal_stages(tessellation.iter().copied(), (false, true), false),
        Ok(vec![
            (Stage::Vertex, S::VERTEX),
            (Stage::Fragment, S::FRAGMENT)
        ])
    );
    assert_eq!(
        stages(&[S::VERTEX, S::FRAGMENT], false),
        Ok(vec![
            (Stage::Vertex, S::VERTEX),
            (Stage::Fragment, S::FRAGMENT)
        ])
    );
    assert_eq!(
        stages(
            &[
                S::VERTEX,
                S::TESSELLATION_CONTROL,
                S::TESSELLATION_EVALUATION,
                S::FRAGMENT
            ],
            true
        ),
        Ok(vec![
            (Stage::Vertex, S::TESSELLATION_EVALUATION),
            (Stage::Fragment, S::FRAGMENT)
        ])
    );
    assert_eq!(
        stages(
            &[
                S::VERTEX,
                S::GEOMETRY,
                S::GEOMETRY,
                S::GEOMETRY,
                S::FRAGMENT
            ],
            false
        ),
        Ok(vec![
            (Stage::Vertex, S::VERTEX),
            (Stage::Fragment, S::FRAGMENT)
        ])
    );
    for (modules, tessellated) in [
        (&[S::COMPUTE][..], false),
        (&[S::TESSELLATION_EVALUATION][..], false),
        (&[S::TESSELLATION_CONTROL][..], false),
        (&[S::COMPUTE][..], true),
    ] {
        assert!(matches!(
            stages(modules, tessellated),
            Err(BindError::Internal(_))
        ));
    }
    assert_eq!(
        pipeline_stage(Stage::Fragment),
        vk::PipelineStageFlags2::FRAGMENT_SHADER
    );
    assert_eq!(
        tessellated_pipeline_stage(Stage::Vertex),
        vk::PipelineStageFlags2::TESSELLATION_EVALUATION_SHADER
    );
    assert_eq!(
        tessellated_pipeline_stage(Stage::Fragment),
        vk::PipelineStageFlags2::FRAGMENT_SHADER
    );
}

fn vslot(index: u32) -> SlotRef {
    SlotRef {
        stage: Stage::Vertex,
        class: SlotClass::Buffer,
        index,
    }
}

/// A per-vertex buffer of static stride 24 whose attributes read bytes `[0, 24)` at alignment 4
/// (binding 0, slot 1), and a per-instance buffer of per-bind stride, divisor 2, reading bytes
/// `[4, 8)` at alignment 2 (binding 1, slot 2).
fn vertex_buffers() -> Vec<PlannedVertexBuffer> {
    vec![
        PlannedVertexBuffer {
            binding: 0,
            slot: vslot(1),
            stride: VertexStride::Static(24),
            input_rate: vk::VertexInputRate::VERTEX,
            divisor: 1,
            alignment: 4,
            reads: (0, 24),
        },
        PlannedVertexBuffer {
            binding: 1,
            slot: vslot(2),
            stride: VertexStride::PerBind,
            input_rate: vk::VertexInputRate::INSTANCE,
            divisor: 2,
            alignment: 2,
            reads: (4, 8),
        },
    ]
}

fn rules(portability_alignment: Option<u32>, dynamic: bool) -> StrideRules {
    StrideRules {
        max: 2048,
        portability_alignment,
        dynamic,
        within_stride: false,
    }
}

fn primitives(vertex_start: u32, vertex_count: u32, instances: (u32, u32)) -> DrawCall {
    DrawCall::Primitives {
        vertex_start,
        vertex_count,
        instance_count: instances.1,
        base_instance: instances.0,
    }
}

fn indexed(index_count: u32, instances: (u32, u32)) -> DrawCall {
    DrawCall::Indexed {
        index_type: vk::IndexType::UINT16,
        index_buffer: IndexBufferRef::new(7, 4096),
        index_count,
        index_offset: 64,
        instance_count: instances.1,
        base_vertex: 5,
        base_instance: instances.0,
    }
}

/// A fixed-function buffer's window is what the draw's vertex (per-vertex) or instance
/// (per-instance, Vulkan's `firstInstance + i / divisor`) range fetches; the vertices of an
/// indexed draw and everything of an indirect draw are GPU-side (`Whole`); a draw of nothing
/// fetches nothing; a window past `i64` is `Whole`.
#[test]
fn vertex_windows_follow_the_vertex_and_instance_ranges() {
    let [per_vertex, per_instance] = vertex_buffers()[..] else {
        panic!("two buffers")
    };
    let bounded = |start, len| Window::Bounded { start, len };
    let empty = bounded(0, 0);
    // Vertices 2..=4 at stride 24: bytes 48..120.
    assert_eq!(
        fetch_window(&per_vertex, 24, &primitives(2, 3, (0, 1))),
        bounded(48, 72)
    );
    // Instances 1..=3 (five instances from 1, divisor 2) at stride 8, bytes 4..8 of each.
    assert_eq!(
        fetch_window(&per_instance, 8, &primitives(2, 3, (1, 5))),
        bounded(12, 20)
    );
    assert_eq!(
        fetch_window(&per_instance, 8, &indexed(6, (1, 5))),
        bounded(12, 20)
    );
    let mut zero = per_instance;
    zero.divisor = 0;
    assert_eq!(
        fetch_window(&zero, 8, &primitives(0, 3, (3, 9))),
        bounded(28, 4)
    );
    // A zero stride reads one element for every vertex.
    assert_eq!(
        fetch_window(&per_vertex, 0, &primitives(9, 100, (0, 1))),
        bounded(0, 24)
    );
    assert_eq!(
        fetch_window(&per_vertex, 24, &indexed(6, (0, 1))),
        Window::Whole
    );
    let indirect = DrawCall::Indirect {
        indirect: IndirectRef::new(1, 0),
        indexed: None,
    };
    assert_eq!(fetch_window(&per_vertex, 24, &indirect), Window::Whole);
    assert_eq!(fetch_window(&per_instance, 8, &indirect), Window::Whole);
    for draw in [
        primitives(2, 0, (0, 1)),
        primitives(2, 3, (0, 0)),
        indexed(0, (0, 1)),
        indexed(6, (0, 0)),
    ] {
        assert_eq!(fetch_window(&per_vertex, 24, &draw), empty, "{draw:?}");
        assert_eq!(fetch_window(&per_instance, 8, &draw), empty, "{draw:?}");
    }
    assert_eq!(
        fetch_window(&per_vertex, 1 << 31, &primitives(1 << 31, 1, (0, 1))),
        bounded(1 << 62, 24)
    );
    assert_eq!(
        fetch_window(&per_vertex, u32::MAX, &primitives(u32::MAX, 1, (0, 1))),
        Window::Whole,
        "past i64"
    );
    // A start inside `i64` whose end is past it (`start + len` would overflow) is `Whole`.
    assert_eq!(
        fetch_window(&per_vertex, 1 << 31, &primitives(1 << 31, u32::MAX, (0, 1))),
        Window::Whole,
        "end past i64"
    );
    // The hull of two windows holds both; an empty one adds nothing; `Whole` absorbs.
    assert_eq!(hull(bounded(0, 16), bounded(48, 72)), bounded(0, 120));
    assert_eq!(hull(bounded(-8, 4), bounded(0, 4)), bounded(-8, 12));
    assert_eq!(hull(empty, bounded(48, 72)), bounded(48, 72));
    assert_eq!(hull(bounded(48, 72), empty), bounded(48, 72));
    assert_eq!(hull(bounded(0, 16), Window::Whole), Window::Whole);
}

/// Fixed-function input fetches a vertex buffer when its bind offset is a multiple of the
/// attributes' alignment and a per-bind stride is at most `maxVertexInputBindingStride`, a
/// multiple of the attributes' alignment and of the portability alignment (and at least it), and
/// as dynamic state 0 or covering an element's attributes, and covering them (0 too) where the
/// portability subset withholds `vertexAttributeAccessBeyondStride`; otherwise the plan pulls the
/// layout.
#[test]
fn fixed_function_fetches_only_what_vulkan_can_bind() {
    let vb = vertex_buffers()[1];
    let per_bind = |rules: StrideRules, stride| fetchable(&vb, &rules, 4, Some(stride));
    assert!(per_bind(rules(None, true), 8));
    assert!(per_bind(rules(None, true), 0));
    assert!(
        !per_bind(rules(None, true), 6),
        "does not cover bytes [4, 8)"
    );
    assert!(per_bind(rules(None, false), 6));
    assert!(!per_bind(rules(None, false), 3), "not a multiple of 2");
    assert!(per_bind(rules(Some(4), true), 12));
    assert!(!per_bind(rules(Some(4), true), 10));
    assert!(!per_bind(rules(Some(4), false), 0));
    assert!(!per_bind(rules(Some(1), true), 0));
    assert!(per_bind(rules(None, true), 2048));
    assert!(!per_bind(rules(None, true), 2050), "past the limit");
    // Without `vertexAttributeAccessBeyondStride` a stride must cover bytes [4, 8) (a zero one
    // already breaks the portability subset's stride alignment).
    let within = StrideRules {
        within_stride: true,
        ..rules(Some(1), false)
    };
    assert!(per_bind(within, 8));
    assert!(!per_bind(within, 6), "does not cover bytes [4, 8)");
    assert!(per_bind(rules(Some(1), false), 6));
    // The bind offset's alignment class against the attributes' alignment (2 here, 4 for the
    // first buffer); a static stride is link's.
    assert!(fetchable(&vb, &rules(None, true), 2, None));
    assert!(!fetchable(&vb, &rules(None, true), 1, None));
    let first = vertex_buffers()[0];
    assert!(fetchable(&first, &rules(None, true), 4, None));
    assert!(!fetchable(&first, &rules(None, true), 2, None));
    // Classed up to 4, and the view base alignment where one is told apart and divides it.
    let offsets = [0, 1, 2, 3, 4, 6, 8, 12, 64, 192, 1 << 40, (1 << 40) + 2];
    assert_eq!(
        offsets.map(|o| offset_alignment(o, 0)),
        [4, 1, 2, 1, 4, 2, 4, 4, 4, 4, 4, 2]
    );
    assert_eq!(
        offsets.map(|o| offset_alignment(o, 64)),
        [64, 1, 2, 1, 4, 2, 4, 4, 64, 64, 64, 2]
    );
}

/// An attribute's alignment is its component size (the whole word of a packed format) and the
/// binding's element bytes run from the smallest offset to the largest end.
#[test]
fn attribute_reads_take_the_widest_alignment_and_the_element_hull() {
    let attr = |binding, format, offset| VertexAttribute {
        location: 0,
        binding,
        format,
        offset,
    };
    let attributes = [
        attr(0, vk::Format::R16G16_SFLOAT, 4),
        attr(0, vk::Format::R8G8B8A8_UNORM, 1),
        attr(1, vk::Format::R32G32B32_SFLOAT, 0),
        attr(0, vk::Format::A2B10G10R10_UNORM_PACK32, 12),
    ];
    assert_eq!(attribute_reads(&attributes, 0), Ok((4, (1, 16))));
    assert_eq!(attribute_reads(&attributes, 1), Ok((4, (0, 12))));
    assert!(matches!(
        attribute_reads(&attributes, 2),
        Err(BindError::Internal(_))
    ));
    assert!(matches!(
        attribute_reads(&[attr(0, vk::Format::BC1_RGB_UNORM_BLOCK, 0)], 0),
        Err(BindError::Internal(_))
    ));
}

/// A draw lists its index buffer (the direct draw's indices from the index offset, the rest of
/// the buffer for an indirect one) and its indirect arguments as uses.
#[test]
fn draws_list_their_index_and_indirect_buffers() {
    let index = |range| ResourceUse {
        resource: UseTarget::IndexBuffer,
        access: AccessSet::READ,
        class: AccessClass::Index,
        stages: vk::PipelineStageFlags2::INDEX_INPUT,
        range,
    };
    let args = |len| ResourceUse {
        resource: UseTarget::Indirect,
        access: AccessSet::READ,
        class: AccessClass::Indirect,
        stages: vk::PipelineStageFlags2::DRAW_INDIRECT,
        range: Some(ByteRange { offset: 0, len }),
    };
    assert!(draw_uses(&primitives(0, 3, (0, 1)), None, None).is_empty());
    assert_eq!(
        draw_uses(&indexed(6, (0, 1)), None, None),
        vec![index(Some(ByteRange { offset: 0, len: 12 }))]
    );
    let mut wide = indexed(6, (0, 1));
    if let DrawCall::Indexed { index_type, .. } = &mut wide {
        *index_type = vk::IndexType::UINT32;
    }
    assert_eq!(
        draw_uses(&wide, None, None),
        vec![index(Some(ByteRange { offset: 0, len: 24 }))]
    );
    assert_eq!(
        draw_uses(
            &DrawCall::Indirect {
                indirect: IndirectRef::new(1, 32),
                indexed: None,
            },
            None,
            None
        ),
        vec![args(16)]
    );
    assert_eq!(
        draw_uses(
            &DrawCall::Indirect {
                indirect: IndirectRef::new(1, 32),
                indexed: Some((vk::IndexType::UINT32, IndexBufferRef::new(2, 64), 8)),
            },
            None,
            None
        ),
        vec![index(None), args(20)]
    );
}

/// A direct indexed draw of `index_count` indices of `index_type` from `index_offset` in an
/// index buffer of `length` bytes.
fn indexed_in(
    index_type: vk::IndexType,
    length: u64,
    index_offset: u64,
    index_count: u32,
) -> DrawCall {
    DrawCall::Indexed {
        index_type,
        index_buffer: IndexBufferRef::new(7, length),
        index_count,
        index_offset,
        instance_count: 2,
        base_vertex: -5,
        base_instance: 1,
    }
}

/// The index positions past the buffer are those whose bytes do not lie wholly inside it, counted
/// from the index offset in each index size; a draw to the buffer's last byte has none. They are
/// drawn where the device bounds the fetch (`robustBufferAccess2`) and the first index lies inside
/// the buffer, and clamped otherwise; other draws and sizeless index types have none.
#[test]
fn index_positions_past_the_buffer_are_counted_from_the_index_offset() {
    use vk::IndexType as T;
    let robust = |past_end| Some(IndexOverrun::Robust { past_end });
    let clamped = |past_end| Some(IndexOverrun::Clamped { past_end });
    for (draw, past) in [
        // 12 halves from byte 64 of 88 bytes end at its last byte.
        (indexed_in(T::UINT16, 88, 64, 12), None),
        (indexed_in(T::UINT16, 88, 64, 15), Some(3)),
        // Metal's fetch reads bytes up to the length rounded up to 4 (`fetch_end`): the half at
        // byte 88 of 89 bytes reads byte 89 too, and the one at 90 is past the end.
        (indexed_in(T::UINT16, 89, 64, 14), None),
        (indexed_in(T::UINT16, 89, 64, 15), Some(1)),
        (indexed_in(T::UINT32, 18, 0, 5), None),
        (indexed_in(T::UINT32, 18, 0, 8), Some(3)),
        (indexed_in(T::UINT32, 66, 64, 3), Some(2)),
        (indexed_in(T::UINT8, 10, 3, 9), None),
        (indexed_in(T::UINT8, 10, 3, 11), Some(2)),
        // A draw of no index reads none, at its offset or past the end (its bind is
        // `a_draw_of_no_index_binds_inside_the_buffer`'s).
        (indexed_in(T::UINT32, 64, 32, 0), None),
        (indexed_in(T::UINT32, 64, 64, 0), None),
        (indexed_in(T::UINT32, 64, 4096, 0), None),
        // The largest buffer holds every index a `u32` count can name.
        (indexed_in(T::UINT32, u64::MAX, u64::MAX - 8, 2), None),
        (
            indexed_in(T::UINT32, u64::MAX, u64::MAX - 8, u32::MAX),
            Some(u32::MAX - 2),
        ),
    ] {
        assert_eq!(
            index_overrun(false, &draw),
            past.and_then(clamped),
            "{draw:?}"
        );
        assert_eq!(
            index_overrun(true, &draw),
            past.and_then(robust),
            "{draw:?}"
        );
    }
    // No index inside the buffer: an offset at or past its end cannot be bound, so it is clamped
    // with the feature too. The rounding to 4 bytes applies only from an offset before the
    // length.
    for draw in [
        indexed_in(T::UINT32, 64, 64, 3),
        indexed_in(T::UINT32, 64, 4096, 3),
        indexed_in(T::UINT16, 66, 66, 3),
    ] {
        assert_eq!(index_overrun(true, &draw), clamped(3), "{draw:?}");
        assert_eq!(index_overrun(false, &draw), clamped(3), "{draw:?}");
    }
    let sizeless = indexed_in(T::NONE_KHR, 4, 0, 64);
    let indirect = DrawCall::Indirect {
        indirect: IndirectRef::new(1, 0),
        indexed: Some((T::UINT32, IndexBufferRef::new(2, 4), 0)),
    };
    for draw in [sizeless, indirect, primitives(0, 3, (0, 1))] {
        assert_eq!(index_overrun(false, &draw), None, "{draw:?}");
    }
}

/// A clamped draw draws its indices inside the buffer from the same offset, and one with none binds
/// the buffer at offset 0 and draws no index; a robust draw is the Metal draw's. Uses follow: a
/// clamped draw reads the indices it draws, a robust one the buffer from the index offset.
#[test]
fn an_overrunning_draw_draws_and_reads_as_its_overrun_states() {
    let draw = indexed_in(vk::IndexType::UINT16, 89, 64, 15);
    let vk_indexed = |index_buffer_offset, index_count| vk::DrawCommand::DrawIndexed {
        index_type: vk::IndexType::UINT16,
        index_buffer_offset,
        index_count,
        instance_count: 2,
        first_index: 0,
        vertex_offset: -5,
        first_instance: 1,
    };
    let index = |len| ResourceUse {
        resource: UseTarget::IndexBuffer,
        access: AccessSet::READ,
        class: AccessClass::Index,
        stages: vk::PipelineStageFlags2::INDEX_INPUT,
        range: Some(ByteRange { offset: 0, len }),
    };
    // 14 halves lie before byte 92, the length rounded up to 4 (`fetch_end`).
    let clamped = Some(IndexOverrun::Clamped { past_end: 1 });
    let robust = Some(IndexOverrun::Robust { past_end: 1 });
    assert_eq!(index_overrun(false, &draw), clamped);
    assert_eq!(index_overrun(true, &draw), robust);
    assert_eq!(
        vk_draw(&draw, clamped),
        (vk_indexed(64, 14), [(-5i32) as u32, 1])
    );
    assert_eq!(draw_uses(&draw, clamped, None), vec![index(28)]);
    assert_eq!(vk_draw(&draw, robust).0, vk_indexed(64, 15));
    assert_eq!(draw_uses(&draw, robust, None), vec![index(28)]);
    assert_eq!(vk_draw(&draw, None).0, vk_indexed(64, 15));
    assert_eq!(draw_uses(&draw, None, None), vec![index(30)]);
    let past = indexed_in(vk::IndexType::UINT16, 64, 64, 4);
    let none = Some(IndexOverrun::Clamped { past_end: 4 });
    assert_eq!(vk_draw(&past, none).0, vk_indexed(0, 0));
    assert_eq!(draw_uses(&past, none, None), vec![index(0)]);
}

/// End to end on a synthetic template: instantiation records the overrun and draws it as the
/// device's index robustness allows, and leaves a draw inside the buffer unchanged.
#[test]
fn instantiation_bounds_a_draw_by_its_index_buffer() {
    let template = |robust_index_fetch| GraphicsTemplate {
        buffers: Vec::new(),
        stages: Vec::new(),
        vertex_buffers: Vec::new(),
        pulled_strides: Vec::new(),
        resources: ResourceTemplate::default(),
        vertex_strides: rules(None, false),
        multiview_instance_limit: None,
        other_layer_variant: false,
        amplifications: Vec::new(),
        tessellation: None,
        indirect_first_instance_missing: false,
        robust_index_fetch,
        shadow_stencil: None,
        mesh: None,
        shader_tests: None,
        deindexed: None,
        private_list: false,
        storage_kept: 0,
        line_band: None,
    };
    let state = BindState::default();
    let past = indexed_in(vk::IndexType::UINT32, 40, 8, 10);
    let inside = indexed_in(vk::IndexType::UINT32, 48, 8, 10);
    let count = |cmds: &DrawCommands| match cmds.draw {
        vk::DrawCommand::DrawIndexed { index_count, .. } => index_count,
        _ => panic!("an indexed draw: {:?}", cmds.draw),
    };
    for robust in [false, true] {
        let t = template(robust);
        let cmds = instantiate(&t, &state, &inside).expect("inside");
        assert_eq!((count(&cmds), cmds.index_overrun), (10, None));
        let cmds = instantiate(&t, &state, &past).expect("past");
        let expected = if robust {
            (10, Some(IndexOverrun::Robust { past_end: 2 }))
        } else {
            (8, Some(IndexOverrun::Clamped { past_end: 2 }))
        };
        assert_eq!((count(&cmds), cmds.index_overrun), expected);
        assert!(!cmds.indirect_needs_arguments);
    }
}

/// A draw of no index binds the buffer at offset 0 where its own offset is at or past the buffer's
/// length, which may lie past the `VkBuffer` (`VUID-vkCmdBindIndexBuffer-offset-08782`), for every
/// index type and with or without `robustBufferAccess2`; it records no overrun. One whose offset
/// lies inside the buffer keeps it.
#[test]
fn a_draw_of_no_index_binds_inside_the_buffer() {
    let template = |robust_index_fetch| GraphicsTemplate {
        buffers: Vec::new(),
        stages: Vec::new(),
        vertex_buffers: Vec::new(),
        pulled_strides: Vec::new(),
        resources: ResourceTemplate::default(),
        vertex_strides: rules(None, false),
        multiview_instance_limit: None,
        other_layer_variant: false,
        amplifications: Vec::new(),
        tessellation: None,
        indirect_first_instance_missing: false,
        robust_index_fetch,
        shadow_stencil: None,
        mesh: None,
        shader_tests: None,
        deindexed: None,
        private_list: false,
        storage_kept: 0,
        line_band: None,
    };
    let state = BindState::default();
    let bound = |cmds: &DrawCommands| match cmds.draw {
        vk::DrawCommand::DrawIndexed {
            index_buffer_offset,
            index_count,
            ..
        } => (index_buffer_offset, index_count),
        _ => panic!("an indexed draw: {:?}", cmds.draw),
    };
    for ty in [
        vk::IndexType::UINT8,
        vk::IndexType::UINT16,
        vk::IndexType::UINT32,
        vk::IndexType::NONE_KHR,
    ] {
        for robust in [false, true] {
            let t = template(robust);
            for (offset, at) in [
                (32, 32),
                (60, 60),
                (64, 0),
                (66, 0),
                (4096, 0),
                (u64::MAX, 0),
            ] {
                let draw = indexed_in(ty, 64, offset, 0);
                let cmds = instantiate(&t, &state, &draw).expect("draw");
                assert_eq!(bound(&cmds), (at, 0), "{ty:?} {robust} {offset}");
                assert_eq!(cmds.index_overrun, None, "{ty:?} {robust} {offset}");
                assert_eq!(vk_draw(&draw, None).0, cmds.draw);
            }
        }
    }
}

/// An indexed indirect draw, whose counts are in GPU memory, asks for its arguments on the CPU
/// where nothing bounds its index fetch: on a device without `robustBufferAccess2`, or with an
/// index offset at or past the buffer's length. Elsewhere it stays GPU-side, and a draw without
/// indices is unaffected.
#[test]
fn an_unbounded_indexed_indirect_draw_reads_its_arguments() {
    let indirect = |indexed| DrawCall::Indirect {
        indirect: IndirectRef::new(1, 0),
        indexed,
    };
    let at = |offset| Some((vk::IndexType::UINT32, IndexBufferRef::new(2, 64), offset));
    for (robust, draw, unbounded) in [
        (false, indirect(at(0)), true),
        (false, indirect(at(60)), true),
        (true, indirect(at(0)), false),
        (true, indirect(at(60)), false),
        (true, indirect(at(64)), true),
        (true, indirect(at(4096)), true),
        (false, indirect(None), false),
        (true, indirect(None), false),
        (false, indexed_in(vk::IndexType::UINT32, 64, 0, 99), false),
    ] {
        assert_eq!(
            indirect_indices_unbounded(robust, &draw),
            unbounded,
            "{robust} {draw:?}"
        );
    }
    let template = |robust_index_fetch, indirect_first_instance_missing| GraphicsTemplate {
        buffers: Vec::new(),
        stages: Vec::new(),
        vertex_buffers: Vec::new(),
        pulled_strides: Vec::new(),
        resources: ResourceTemplate::default(),
        vertex_strides: rules(None, false),
        multiview_instance_limit: None,
        other_layer_variant: false,
        amplifications: Vec::new(),
        tessellation: None,
        indirect_first_instance_missing,
        robust_index_fetch,
        shadow_stencil: None,
        mesh: None,
        shader_tests: None,
        deindexed: None,
        private_list: false,
        storage_kept: 0,
        line_band: None,
    };
    let state = BindState::default();
    for robust in [false, true] {
        let t = template(robust, false);
        let cmds = instantiate(&t, &state, &indirect(at(8))).expect("indexed");
        assert_eq!(cmds.indirect_needs_arguments, !robust);
        // The index bound is the only reason: the GPU pass may bound it instead.
        assert_eq!(cmds.index_bounds_pass, !robust);
        assert_eq!(cmds.index_overrun, None);
        let cmds = instantiate(&t, &state, &indirect(at(64))).expect("indexed past");
        assert!(cmds.indirect_needs_arguments && cmds.index_bounds_pass);
        let cmds = instantiate(&t, &state, &indirect(None)).expect("indirect");
        assert!(!cmds.indirect_needs_arguments && !cmds.index_bounds_pass);
        // Another reason asks for the CPU read too: the pass does not stand in for it.
        let t = template(robust, true);
        let cmds = instantiate(&t, &state, &indirect(at(8))).expect("indexed");
        assert!(cmds.indirect_needs_arguments && !cmds.index_bounds_pass);
    }
}

/// Instantiation binds each vertex buffer at its offset (which must be a multiple of the
/// attributes' alignment), checks and passes per-bind strides, and gives each buffer the window
/// its draw fetches, joined with the window of a buffer argument at the same index.
#[test]
fn draws_bind_vertex_buffers_at_aligned_offsets_with_their_windows() {
    let template = |rules| GraphicsTemplate {
        buffers: vec![PlannedBuffer {
            slot: vslot(1),
            window: Window::Bounded { start: 0, len: 16 },
            form: BufferForm::Descriptor(DescriptorView::Storage),
            access: AccessSet::READ,
            binding: Some((0, 0)),
            alignment: 16,
            max_range: 1 << 27,
            bound: BoundKind::Buffer {
                device_address: false,
                length_class: length_class(4096),
            },
            disjoint: false,
        }],
        stages: Vec::new(),
        vertex_buffers: vertex_buffers(),
        pulled_strides: Vec::new(),
        resources: ResourceTemplate::default(),
        vertex_strides: rules,
        multiview_instance_limit: None,
        other_layer_variant: false,
        amplifications: Vec::new(),
        tessellation: None,
        indirect_first_instance_missing: false,
        robust_index_fetch: false,
        shadow_stencil: None,
        mesh: None,
        shader_tests: None,
        deindexed: None,
        private_list: false,
        storage_kept: 0,
        line_band: None,
    };
    let state = |offsets: (u64, u64), stride| {
        let buffer = |offset, attribute_stride| BoundBuffer::Buffer {
            length: 4096,
            offset,
            gpu_address: 0x1_0000,
            device_address: None,
            attribute_stride,
        };
        let mut t = StageTable::default();
        t.buffers.insert(1, buffer(offsets.0, None));
        t.buffers.insert(2, buffer(offsets.1, Some(stride)));
        let mut s = BindState::default();
        s.stages.insert(Stage::Vertex, t);
        s
    };
    let t = template(rules(None, true));
    let draw = primitives(2, 3, (1, 5));
    let cmds = instantiate(&t, &state((64, 6), 8), &draw).expect("draw");
    assert_eq!(cmds.buffer_offsets, vec![(0, 64), (1, 6)]);
    // Dynamic strides list every binding's, the static one included (`pStrides`).
    assert_eq!(cmds.vertex_strides, vec![(0, 24), (1, 8)]);
    let window = |slot| {
        cmds.extents
            .iter()
            .find(|e| e.slot == slot)
            .map(|e| e.window)
    };
    assert_eq!(
        window(vslot(1)),
        Some(Window::Bounded { start: 0, len: 120 })
    );
    assert_eq!(
        window(vslot(2)),
        Some(Window::Bounded { start: 12, len: 20 })
    );
    assert!(cmds.uses.is_empty());
    let cmds = instantiate(&t, &state((64, 6), 8), &indexed(6, (1, 5))).expect("indexed");
    assert_eq!(
        cmds.extents
            .iter()
            .find(|e| e.slot == vslot(1))
            .map(|e| e.window),
        Some(Window::Whole)
    );
    assert_eq!(cmds.uses.len(), 1);
    // Offsets and strides fixed function cannot fetch are the plan's pulled variant: a draw of this
    // plan with them is an internal error.
    let pulled = |r: Result<DrawCommands, BindError>| matches!(r, Err(BindError::Internal(_)));
    assert!(pulled(instantiate(&t, &state((66, 6), 8), &draw)));
    assert!(pulled(instantiate(&t, &state((64, 7), 8), &draw)));
    assert!(pulled(instantiate(&t, &state((64, 6), 6), &draw)));
    let t = template(rules(None, false));
    let cmds = instantiate(&t, &state((64, 6), 6), &draw).expect("static strides");
    assert_eq!(
        cmds.vertex_strides,
        vec![(1, 6)],
        "only the per-bind stride"
    );
    let t = template(rules(Some(4), false));
    assert!(pulled(instantiate(&t, &state((64, 6), 6), &draw)));
    assert!(instantiate(&t, &state((64, 6), 12), &draw).is_ok());
}

/// Inline vertex bytes at a per-bind layout read at their bind's stride, 0 without one, as plain
/// Metal reads them (`setVertexBytes:length:atIndex:` read vertex 0 for every
/// vertex, the stride form at 0, 4 and 8 its stride); at `MTLAttributeStrideStatic`
/// (`u32::MAX`), which an M4 read as element 0 and zeros past it, they are refused
/// `MissingVertexStride`; and a stride fixed function cannot fetch is the plan's pulled variant.
#[test]
fn inline_vertex_bytes_bind_the_strides_a_dynamic_layout_reads() {
    let mut template = GraphicsTemplate {
        buffers: Vec::new(),
        stages: Vec::new(),
        vertex_buffers: vertex_buffers(),
        pulled_strides: Vec::new(),
        resources: ResourceTemplate::default(),
        vertex_strides: rules(None, true),
        multiview_instance_limit: None,
        other_layer_variant: false,
        amplifications: Vec::new(),
        tessellation: None,
        indirect_first_instance_missing: false,
        robust_index_fetch: false,
        shadow_stencil: None,
        mesh: None,
        shader_tests: None,
        deindexed: None,
        private_list: false,
        storage_kept: 0,
        line_band: None,
    };
    let state = |attribute_stride| {
        let mut t = StageTable::default();
        t.buffers.insert(
            1,
            BoundBuffer::Bytes {
                len: 256,
                attribute_stride: None,
            },
        );
        t.buffers.insert(
            2,
            BoundBuffer::Bytes {
                len: 256,
                attribute_stride,
            },
        );
        let mut s = BindState::default();
        s.stages.insert(Stage::Vertex, t);
        s
    };
    let draw = primitives(2, 3, (1, 5));
    for (stride, read) in [(None, 0), (Some(0), 0), (Some(8), 8), (Some(16), 16)] {
        let cmds = instantiate(&template, &state(stride), &draw).expect("draw");
        assert_eq!(cmds.buffer_offsets, vec![(0, 0), (1, 0)], "{stride:?}");
        // Dynamic strides list every binding's, the static one included (`pStrides`).
        assert_eq!(cmds.vertex_strides, vec![(0, 24), (1, read)], "{stride:?}");
    }
    assert_eq!(
        instantiate(&template, &state(Some(u32::MAX)), &draw).err(),
        Some(BindError::MissingVertexStride { slot: vslot(2) })
    );
    // A stride fixed function cannot fetch (past the attributes' alignment, or below the bytes
    // an element reads) is the plan's pulled variant: a draw of this plan with it is an internal
    // error.
    let pulled = |r: Result<DrawCommands, BindError>| matches!(r, Err(BindError::Internal(_)));
    assert!(pulled(instantiate(&template, &state(Some(7)), &draw)));
    assert!(pulled(instantiate(&template, &state(Some(6)), &draw)));
    template.vertex_strides = rules(None, false);
    let cmds = instantiate(&template, &state(Some(6)), &draw).expect("static strides");
    assert_eq!(
        cmds.vertex_strides,
        vec![(1, 6)],
        "only the per-bind stride"
    );
}

fn factor_template(step: metal::TessellationFactorStepFunction, bound: bool) -> FactorTemplate {
    FactorTemplate {
        step,
        patch_bytes: 12,
        alignment: 16,
        max_range: 1 << 20,
        control_points: 4,
        patch_indices: Some((0, 4)),
        control_point_indices: None,
        site: FactorSite::Control {
            push_offset: 20,
            binding: bound.then_some((0, 3)),
        },
    }
}

fn patches(
    patch_start: u32,
    patch_count: u32,
    instance_count: u32,
    base_instance: u32,
) -> DrawCall {
    DrawCall::Patches {
        control_points: 4,
        patch_start,
        patch_count,
        patch_index: None,
        instance_count,
        base_instance,
        control_point_index: None,
    }
}

fn factor_state(offset: u64, instance_stride: u64, scale: f32) -> BindState {
    BindState {
        tessellation_factors: crate::bind::state::TessellationFactors::new(
            Some(crate::bind::state::TessellationFactorBuffer::new(
                4096,
                offset,
                instance_stride,
            )),
            scale,
        ),
        ..BindState::default()
    }
}

/// A patch draw is one vertex per patch from the patch start, with the draw's instances; its
/// indirect form reads `MTLDrawPatchIndirectArguments` as a `VkDrawIndirectCommand`.
#[test]
fn patch_draws_draw_one_vertex_per_patch() {
    let (d, base) = vk_draw(&patches(5, 9, 3, 2), None);
    assert_eq!(
        d,
        vk::DrawCommand::Draw {
            vertex_count: 9,
            instance_count: 3,
            first_vertex: 5,
            first_instance: 2
        }
    );
    assert_eq!(base, [5, 2]);
    let indirect = DrawCall::PatchesIndirect {
        control_points: 4,
        patch_index: None,
        indirect: IndirectRef::new(1, 48),
        control_point_index: None,
    };
    assert_eq!(
        vk_draw(&indirect, None).0,
        vk::DrawCommand::DrawIndirect {
            offset: 48,
            draw_count: 1,
            stride: 16
        }
    );
    assert_eq!(draw_uses(&patches(0, 1, 1, 0), None, None), Vec::new());
    assert_eq!(draw_uses(&indirect, None, None).len(), 1);
}

/// A patch draw's patch index buffer: bound over the indices of the draw's patches (an indirect
/// draw's from its offset to the buffer's end), its residual offset and 1 pushed after the factors'
/// words, and read by the control stage; an offset that is not a multiple of 4 is refused, as
/// Metal's validation refuses it.
#[test]
fn patch_draws_bind_their_patch_index_buffers() {
    use metal::TessellationFactorStepFunction as S;
    let t = factor_template(S::PerPatch, false);
    let indexed = |offset| DrawCall::Patches {
        control_points: 4,
        patch_start: 5,
        patch_count: 3,
        patch_index: Some((IndexBufferRef::new(2, 4096), offset)),
        instance_count: 1,
        base_instance: 0,
        control_point_index: None,
    };
    let (push, ranges) = patch_draw(Some(&t), &BindState::default(), &indexed(36))
        .expect("draw")
        .expect("a patch draw");
    // Indices at bytes [36 + 20, 36 + 32) of the buffer: bound from 48 (16-aligned) to 80.
    let [range] = ranges[..] else {
        panic!("{ranges:?}: one bound patch index buffer")
    };
    assert_eq!((range.set, range.binding), (0, 4));
    assert_eq!((range.range.offset, range.range.range), (48, 32));
    let word = |k: usize| u32::from_le_bytes(push.bytes[4 * k..4 * k + 4].try_into().unwrap());
    assert_eq!(push.bytes.len(), 24);
    assert_eq!((word(3), word(4)), (36u32.wrapping_sub(48), 1));
    assert_eq!(
        draw_uses(&indexed(36), None, Some(&t)),
        [ResourceUse {
            resource: UseTarget::PatchIndices,
            access: AccessSet::READ,
            class: AccessClass::Storage,
            stages: vk::PipelineStageFlags2::TESSELLATION_CONTROL_SHADER,
            range: Some(ByteRange {
                offset: 20,
                len: 12
            }),
        }]
    );
    let indirect = DrawCall::PatchesIndirect {
        control_points: 4,
        patch_index: Some((IndexBufferRef::new(2, 100), 40)),
        indirect: IndirectRef::new(1, 0),
        control_point_index: None,
    };
    let (push, ranges) = patch_draw(Some(&t), &BindState::default(), &indirect)
        .expect("draw")
        .expect("a patch draw");
    let [range] = ranges[..] else {
        panic!("{ranges:?}: one bound patch index buffer")
    };
    // From 32 (the offset aligned down) to the padded end, 112.
    assert_eq!((range.range.offset, range.range.range), (32, 80));
    assert_eq!(push.bytes[12..], [8, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]);
    let uses = draw_uses(&indirect, None, Some(&t));
    assert_eq!(uses.len(), 2);
    assert_eq!(
        (uses[0].resource, uses[0].range),
        (UseTarget::PatchIndices, None)
    );
    // An emulated pipeline's vertex stage reads them, where its function reads what they index,
    // and the scan reads the arguments.
    let mut emulated = t;
    emulated.site = FactorSite::Scan {
        kernel: crate::pipeline::graphics::EmulatedTessellation {
            domain: crate::pipeline::interface::PatchDomain::Quad,
            control_points: 4,
            descriptor: Default::default(),
        }
        .scan_kernel(),
        factors: false,
    };
    let uses = draw_uses(&indirect, None, Some(&emulated));
    assert_eq!(
        uses.iter()
            .map(|u| (u.resource, u.stages))
            .collect::<Vec<_>>(),
        [
            (
                UseTarget::PatchIndices,
                vk::PipelineStageFlags2::VERTEX_SHADER
            ),
            (UseTarget::Indirect, vk::PipelineStageFlags2::COMPUTE_SHADER)
        ]
    );
    emulated.patch_indices = None;
    let uses = draw_uses(&indirect, None, Some(&emulated));
    assert_eq!(
        uses.iter().map(|u| u.resource).collect::<Vec<_>>(),
        [UseTarget::Indirect]
    );
    assert_eq!(
        patch_draw(Some(&t), &BindState::default(), &indexed(2)),
        Err(BindError::InvalidPatchIndexBuffer { offset: 2 })
    );
}

/// A pipeline with a control-point index type draws indexed patches only, one without it plain
/// ones only, as Metal's validation requires. An indexed draw binds its control-point index buffer
/// over the indices of its patches' control points when it has no patch index buffer (from the
/// offset to the buffer's end otherwise, and for an indirect draw), pushes its residual offset
/// after the patch index words, and is read by the control stage; an offset that is not a multiple
/// of the index size is refused, as Metal's validation refuses it.
#[test]
fn indexed_patch_draws_bind_their_control_point_index_buffers() {
    use metal::TessellationFactorStepFunction as S;
    let mut t = factor_template(S::PerPatch, false);
    t.control_point_indices = Some((Some((0, 5)), 2));
    let indexed = |offset, patch_index| DrawCall::Patches {
        control_points: 4,
        patch_start: 5,
        patch_count: 3,
        patch_index,
        instance_count: 1,
        base_instance: 0,
        control_point_index: Some((IndexBufferRef::new(3, 4096), offset)),
    };
    let (push, ranges) = patch_draw(Some(&t), &BindState::default(), &indexed(38, None))
        .expect("draw")
        .expect("a patch draw");
    // Indices at bytes [38 + 40, 38 + 64) of the buffer: bound from 64 (16-aligned) to 112.
    let [range] = ranges[..] else {
        panic!("{ranges:?}: one bound control-point index buffer")
    };
    assert_eq!((range.set, range.binding), (0, 5));
    assert_eq!((range.range.offset, range.range.range), (64, 48));
    let word = |k: usize| u32::from_le_bytes(push.bytes[4 * k..4 * k + 4].try_into().unwrap());
    assert_eq!(push.bytes.len(), 24);
    assert_eq!((word(4), word(5)), (0, 38u32.wrapping_sub(64)));
    assert_eq!(
        draw_uses(&indexed(38, None), None, Some(&t)),
        [ResourceUse {
            resource: UseTarget::ControlPointIndices,
            access: AccessSet::READ,
            class: AccessClass::Storage,
            stages: vk::PipelineStageFlags2::TESSELLATION_CONTROL_SHADER,
            range: Some(ByteRange {
                offset: 40,
                len: 24
            }),
        }]
    );
    // With a patch index buffer, every index from the offset: from 32 to the padded end.
    let with_patch_index = indexed(38, Some((IndexBufferRef::new(2, 4096), 0)));
    let (_, ranges) = patch_draw(Some(&t), &BindState::default(), &with_patch_index)
        .expect("draw")
        .expect("a patch draw");
    assert_eq!(ranges.len(), 2);
    assert_eq!((ranges[1].set, ranges[1].binding), (0, 5));
    assert_eq!(
        (ranges[1].range.offset, ranges[1].range.range),
        (32, 4096 - 32)
    );
    let uses = draw_uses(&with_patch_index, None, Some(&t));
    assert_eq!(
        uses.iter()
            .map(|u| (u.resource, u.range))
            .collect::<Vec<_>>(),
        [
            (
                UseTarget::PatchIndices,
                Some(ByteRange {
                    offset: 20,
                    len: 12
                })
            ),
            (UseTarget::ControlPointIndices, None)
        ]
    );
    let indirect = DrawCall::PatchesIndirect {
        control_points: 4,
        patch_index: None,
        indirect: IndirectRef::new(1, 0),
        control_point_index: Some((IndexBufferRef::new(3, 100), 6)),
    };
    let (push, ranges) = patch_draw(Some(&t), &BindState::default(), &indirect)
        .expect("draw")
        .expect("a patch draw");
    assert_eq!((ranges[0].range.offset, ranges[0].range.range), (0, 112));
    assert_eq!(push.bytes[20..], [6, 0, 0, 0]);
    let uses = draw_uses(&indirect, None, Some(&t));
    assert_eq!(
        uses.iter().map(|u| u.resource).collect::<Vec<_>>(),
        [UseTarget::ControlPointIndices, UseTarget::Indirect]
    );
    assert_eq!(uses[0].range, None);
    assert_eq!(
        patch_draw(Some(&t), &BindState::default(), &indexed(3, None)),
        Err(BindError::InvalidControlPointIndexBuffer { offset: 3 })
    );
    // Indexing that differs from the pipeline's.
    assert_eq!(
        patch_draw(Some(&t), &BindState::default(), &patches(0, 1, 1, 0)),
        Err(BindError::ControlPointIndexing { indexed: false })
    );
    let plain = factor_template(S::PerPatch, false);
    assert_eq!(
        patch_draw(Some(&plain), &BindState::default(), &indexed(0, None)),
        Err(BindError::ControlPointIndexing { indexed: true })
    );
}

/// The factor window of each step function: the draw's patches, its instances (from the base
/// instance), both, or the one struct; empty without patches or instances; whole when indirect.
#[test]
fn each_step_function_reads_its_factor_window() {
    use metal::TessellationFactorStepFunction as S;
    let window = |step, draw: &DrawCall| factor_window(&factor_template(step, true), 64, draw);
    let draw = patches(2, 3, 4, 1);
    assert_eq!(
        window(S::Constant, &draw),
        Window::Bounded { start: 0, len: 12 }
    );
    assert_eq!(
        window(S::PerPatch, &draw),
        Window::Bounded { start: 24, len: 36 }
    );
    assert_eq!(
        window(S::PerInstance, &draw),
        Window::Bounded {
            start: 64,
            len: 3 * 64 + 12
        }
    );
    assert_eq!(
        window(S::PerPatchAndPerInstance, &draw),
        Window::Bounded {
            start: 24 + 64,
            len: 2 * 12 + 3 * 64 + 12
        }
    );
    assert_eq!(
        window(S::PerPatch, &patches(2, 0, 4, 1)),
        Window::Bounded { start: 0, len: 0 }
    );
    let indirect = DrawCall::PatchesIndirect {
        control_points: 4,
        patch_index: None,
        indirect: IndirectRef::new(1, 0),
        control_point_index: None,
    };
    assert_eq!(window(S::PerPatch, &indirect), Window::Whole);
}

/// A patch draw pushes the factor buffer's residual bind offset, its instance stride rounded down
/// to a multiple of 4, and the scale rounded to `half`, and binds the factors it reads; without a
/// buffer it pushes zeros over the zero buffer. Draws Metal does not make with the pipeline are
/// refused.
#[test]
fn patch_draws_push_and_bind_their_factors() {
    use metal::TessellationFactorStepFunction as S;
    let t = factor_template(S::PerPatch, true);
    let (push, range) = patch_draw(Some(&t), &factor_state(40, 64, 1.3), &patches(2, 3, 1, 0))
        .expect("draw")
        .expect("a patch draw");
    // Factors at bytes [40 + 24, 40 + 60) of the buffer: bound from 64 (16-aligned) to 112.
    let [range] = range[..] else {
        panic!("{range:?}: one bound factor buffer")
    };
    assert_eq!((range.set, range.binding), (0, 3));
    assert_eq!((range.range.offset, range.range.range), (64, 48));
    let half = f16_bits_to_f32(f32_to_f16_bits(1.3));
    assert_ne!(half, 1.3);
    let mut bytes = (40u32.wrapping_sub(64)).to_le_bytes().to_vec();
    bytes.extend_from_slice(&64u32.to_le_bytes());
    bytes.extend_from_slice(&half.to_bits().to_le_bytes());
    bytes.extend_from_slice(&[0; 12]);
    assert_eq!(
        push,
        PushConstantWrite {
            stages: vk::ShaderStageFlags::TESSELLATION_CONTROL,
            offset: 20,
            bytes,
        }
    );
    // An offset 2 past a multiple of 4, which the control stage reads from that byte: factors at
    // bytes [42 + 24, 42 + 60) bound from 64 to 112, the residual 42 - 64. A stride 2 past a
    // multiple of 4 steps the instances by the multiple below, as Metal does: 66 pushes 64.
    let (push, range) = patch_draw(Some(&t), &factor_state(42, 66, 1.0), &patches(2, 3, 1, 0))
        .expect("draw")
        .expect("a patch draw");
    let [range] = range[..] else {
        panic!("{range:?}: one bound factor buffer")
    };
    assert_eq!((range.range.offset, range.range.range), (64, 48));
    let mut bytes = (42u32.wrapping_sub(64)).to_le_bytes().to_vec();
    bytes.extend_from_slice(&64u32.to_le_bytes());
    bytes.extend_from_slice(&1f32.to_bits().to_le_bytes());
    bytes.extend_from_slice(&[0; 12]);
    assert_eq!(push.bytes, bytes);
    // Per instance, the rounded stride bounds the factors read: instances 1 and 2 at 40 + 64 and
    // 40 + 128, [104, 180) bound from 96 to 192.
    let per_instance = factor_template(S::PerInstance, true);
    let (push, range) = patch_draw(
        Some(&per_instance),
        &factor_state(40, 66, 1.0),
        &patches(0, 1, 2, 1),
    )
    .expect("draw")
    .expect("a patch draw");
    let [range] = range[..] else {
        panic!("{range:?}: one bound factor buffer")
    };
    assert_eq!((range.range.offset, range.range.range), (96, 96));
    assert_eq!(push.bytes[4..8], 64u32.to_le_bytes());
    // No buffer: zeros over the zero buffer.
    let unbound = factor_template(S::PerPatch, false);
    let (push, range) = patch_draw(Some(&unbound), &BindState::default(), &patches(0, 1, 1, 0))
        .expect("draw")
        .expect("a patch draw");
    assert_eq!(range, []);
    let mut bytes = vec![0; 8];
    bytes.extend_from_slice(&1f32.to_bits().to_le_bytes());
    bytes.extend_from_slice(&[0; 12]);
    assert_eq!(push.bytes, bytes);
    // A state whose shape is not the plan's.
    assert!(matches!(
        patch_draw(
            Some(&unbound),
            &factor_state(0, 0, 1.0),
            &patches(0, 1, 1, 0)
        ),
        Err(BindError::Internal(_))
    ));
    // Primitives with a tessellation pipeline, patches with any other.
    let primitives = DrawCall::Primitives {
        vertex_start: 0,
        vertex_count: 3,
        instance_count: 1,
        base_instance: 0,
    };
    assert_eq!(
        patch_draw(Some(&t), &BindState::default(), &primitives),
        Err(BindError::InvalidDrawKind { patches: false })
    );
    assert_eq!(
        patch_draw(None, &BindState::default(), &patches(0, 1, 1, 0)),
        Err(BindError::InvalidDrawKind { patches: true })
    );
    assert_eq!(
        patch_draw(None, &BindState::default(), &primitives),
        Ok(None)
    );
    // Another control-point count, unaligned factors, too many factors.
    let mut three = patches(0, 1, 1, 0);
    if let DrawCall::Patches { control_points, .. } = &mut three {
        *control_points = 3;
    }
    assert_eq!(
        patch_draw(Some(&t), &factor_state(0, 0, 1.0), &three),
        Err(BindError::PatchControlPoints {
            given: 3,
            expected: 4
        })
    );
    // An odd offset is read from the byte below, an odd stride rounded down to a multiple of 4,
    // as plain Metal reads them (one of 2 past a multiple of 4 is read from that byte).
    for step in [S::PerPatch, S::PerInstance] {
        let t = factor_template(step, true);
        for (offset, stride, read) in [
            (1, 0, (0, 0)),
            (0, 3, (0, 0)),
            (3, 6, (2, 4)),
            (7, 15, (6, 12)),
        ] {
            assert_eq!(
                patch_draw(
                    Some(&t),
                    &factor_state(offset, stride, 1.0),
                    &patches(0, 1, 2, 0)
                ),
                patch_draw(
                    Some(&t),
                    &factor_state(read.0, read.1, 1.0),
                    &patches(0, 1, 2, 0)
                ),
                "{step:?}: offset {offset}, stride {stride}"
            );
        }
    }
    let mut large = factor_state(0, 0, 1.0);
    if let Some(b) = &mut large.tessellation_factors.buffer {
        b.length = 1 << 24;
    }
    assert!(matches!(
        patch_draw(Some(&t), &large, &patches(0, 1 << 20, 1, 0)),
        Err(BindError::TessellationFactorRange { limit, .. }) if limit == 1 << 20
    ));
    // A scale that is not a positive `half` once rounded: 0, negative, or NaN (2^-25 is a tie,
    // which rounds to 0, its even neighbour).
    for scale in [
        0.0,
        -0.0,
        -1.0,
        f32::NAN,
        f32::NEG_INFINITY,
        2f32.powi(-25),
        2f32.powi(-30),
        -1e-5,
    ] {
        assert_eq!(
            patch_draw(Some(&t), &factor_state(0, 0, scale), &patches(0, 1, 1, 0)),
            Err(BindError::InvalidTessellationFactorScale {
                bits: scale.to_bits()
            }),
            "{scale}"
        );
    }
    // Every positive `half` is pushed as rounded, subnormal ones included, and an infinite one as
    // 2^32, which takes every positive factor past the maximum level.
    for (scale, pushed) in [
        (2f32.powi(-24), 2f32.powi(-24)),
        (1.5 * 2f32.powi(-25), 2f32.powi(-24)),
        (1e-6, f32::from_bits(0x3588_0000)),
        (f32::from_bits(0x3880_0000), f32::from_bits(0x3880_0000)),
        (1.3, f32::from_bits(0x3fa6_6000)),
        (2.0005, 2.0),
        (65504.0, 65504.0),
        (65520.0, 2f32.powi(32)),
        (1e6, 2f32.powi(32)),
        (f32::INFINITY, 2f32.powi(32)),
    ] {
        let (push, _) = patch_draw(Some(&t), &factor_state(0, 0, scale), &patches(0, 1, 1, 0))
            .unwrap_or_else(|e| panic!("{scale}: {e:?}"))
            .expect("a patch draw");
        let word = u32::from_le_bytes(push.bytes[8..12].try_into().unwrap());
        assert_eq!(f32::from_bits(word), pushed, "{scale}");
    }
}

/// A mesh template of the RADV 780M's mesh limits (256 threads a threadgroup, 65535 threadgroups
/// an axis, `2^22` in all), whose mesh stage reads its grid from a `DispatchRegion` at offset 0
/// and whose fragment stage decides `Presence` of buffer 0.
fn mesh_template(flat: bool, multiples: Vec<u32>) -> GraphicsTemplate {
    use crate::bind::plan::{GroupLimits, GroupStage, MeshTemplate};
    use crate::pipeline::artifact::SpecConstant;
    use crate::pipeline::interface::PlacedField;
    let spec = |id, key| SpecConstant {
        id,
        key,
        size: 4,
        default: 1,
        unread_bits: 0,
    };
    let fragment_buffer = SlotRef {
        stage: Stage::Fragment,
        class: SlotClass::Buffer,
        index: 0,
    };
    let mut layout = vec![
        spec(0, SpecKey::LocalSize(Axis::X)),
        spec(1, SpecKey::LocalSize(Axis::Y)),
        spec(2, SpecKey::LocalSize(Axis::Z)),
        spec(3, SpecKey::MultiRegion),
        spec(4, SpecKey::Presence(fragment_buffer)),
    ];
    if flat {
        layout.push(spec(5, SpecKey::FlatLocalSize));
    }
    GraphicsTemplate {
        buffers: Vec::new(),
        stages: vec![StageTemplate {
            stage: Stage::Mesh,
            flags: vk::ShaderStageFlags::MESH_EXT,
            fields: vec![PlacedField {
                field: PushField::DispatchRegion,
                loc: PushLoc::Push { offset: 0 },
                size: 32,
            }],
            push_range: Some((0, 32)),
            table: None,
            table_size: 0,
            root_first: None,
        }],
        vertex_buffers: Vec::new(),
        pulled_strides: Vec::new(),
        resources: ResourceTemplate::default(),
        vertex_strides: rules(None, false),
        multiview_instance_limit: None,
        other_layer_variant: false,
        amplifications: Vec::new(),
        tessellation: None,
        indirect_first_instance_missing: false,
        robust_index_fetch: true,
        shadow_stencil: None,
        mesh: Some(MeshTemplate {
            spec: layout,
            fixed: vec![(SpecKey::Presence(fragment_buffer), 1)],
            multiples,
            mesh: GroupLimits {
                size: [256; 3],
                invocations: 256,
                count: [65535; 3],
                total: Some(1 << 22),
                stage: GroupStage::Mesh,
            },
            object: None,
            folded: false,
            records: None,
        }),
        shader_tests: None,
        deindexed: None,
        private_list: false,
        storage_kept: 0,
        line_band: None,
    }
}

/// [`mesh_template`]'s pipeline with an object function, of RADV's task limits (128 threads a
/// threadgroup): its object stage reads its grid from a `DispatchRegion` at offset 0 under its
/// own keys (constants 5 to 8), its mesh stage from one at 32.
fn object_template() -> GraphicsTemplate {
    use crate::bind::plan::GroupLimits;
    use crate::pipeline::artifact::SpecConstant;
    use crate::pipeline::interface::PlacedField;
    let mut t = mesh_template(false, Vec::new());
    let region = |offset| {
        vec![PlacedField {
            field: PushField::DispatchRegion,
            loc: PushLoc::Push { offset },
            size: 32,
        }]
    };
    t.stages[0].fields = region(32);
    t.stages[0].push_range = Some((32, 32));
    t.stages.insert(
        0,
        StageTemplate {
            stage: Stage::Object,
            flags: vk::ShaderStageFlags::TASK_EXT,
            fields: region(0),
            push_range: Some((0, 32)),
            table: None,
            table_size: 0,
            root_first: None,
        },
    );
    let m = t.mesh.as_mut().expect("a mesh template");
    for (id, key) in [
        (5, SpecKey::ObjectLocalSize(Axis::X)),
        (6, SpecKey::ObjectLocalSize(Axis::Y)),
        (7, SpecKey::ObjectLocalSize(Axis::Z)),
        (8, SpecKey::ObjectMultiRegion),
    ] {
        m.spec.push(SpecConstant {
            id,
            key,
            size: 4,
            default: 1,
            unread_bits: 0,
        });
    }
    m.object = Some(GroupLimits {
        size: [128; 3],
        invocations: 128,
        count: [65535; 3],
        total: Some(1 << 22),
        stage: crate::bind::plan::GroupStage::Task,
    });
    t
}

/// A draw of a pipeline with an object function launches object threadgroups, made with each
/// stage's threadgroup size under its own keys and writing each stage's `DispatchRegion` (the
/// object stage's grid; the mesh stage's grid is the object function's to set); a `drawMeshThreads`
/// whose threads fill whole object threadgroups launches them, and one that leaves a partial object
/// threadgroup is not implemented; an object threadgroup Metal draws nothing for launches nothing;
/// and each stage's threadgroup, and the object grid, are checked against the device's limits for
/// its stage.
#[test]
fn object_draws_specialize_each_stage_within_its_limits() {
    let state = BindState::default();
    let t = object_template();
    let groups = |groups, lo, lm| DrawCall::MeshThreadgroups {
        groups,
        threads_per_object_group: lo,
        threads_per_mesh_group: lm,
    };
    let threads = |threads, lo| DrawCall::MeshThreads {
        threads,
        threads_per_object_group: lo,
        threads_per_mesh_group: [2, 1, 1],
    };
    let tasks = |x, y, z| vk::DrawCommand::DrawMeshTasks {
        group_count_x: x,
        group_count_y: y,
        group_count_z: z,
    };
    let values = |key: &SpecializationKey| -> Vec<(u32, u32)> {
        key.entries
            .iter()
            .map(|e| {
                let at = e.offset as usize;
                let bytes = key.data[at..at + 4].try_into().expect("4 bytes");
                (e.constant_id, u32::from_le_bytes(bytes))
            })
            .collect()
    };
    let cmds = instantiate(&t, &state, &groups([3, 1, 1], [4, 2, 1], [2, 1, 1])).expect("groups");
    assert_eq!(cmds.draw, tasks(3, 1, 1));
    assert_eq!(
        values(cmds.specialization.as_ref().expect("a mesh draw's key")),
        [
            (0, 2),
            (1, 1),
            (2, 1),
            (3, 0),
            (4, 1),
            (5, 4),
            (6, 2),
            (7, 1),
            (8, 0)
        ]
    );
    let region = |grid, l| region_bytes([0; 3], grid, l).expect("region").to_vec();
    assert_eq!(
        cmds.push_constants,
        [
            PushConstantWrite {
                stages: vk::ShaderStageFlags::TASK_EXT,
                offset: 0,
                bytes: region([12, 2, 1], [4, 2, 1]),
            },
            PushConstantWrite {
                stages: vk::ShaderStageFlags::MESH_EXT,
                offset: 32,
                bytes: region([0; 3], [2, 1, 1]),
            },
        ]
    );
    // `drawMeshThreads`: whole object threadgroups only (a probe measured partial ones).
    let cmds = instantiate(&t, &state, &threads([8, 4, 1], [4, 2, 1])).expect("whole");
    assert_eq!(cmds.draw, tasks(2, 2, 1));
    for partial in [threads([6, 1, 1], [4, 1, 1]), threads([5, 3, 1], [4, 2, 1])] {
        assert_eq!(
            instantiate(&t, &state, &partial).err(),
            Some(not_implemented(Feature::MeshShading)),
            "{partial:?}"
        );
    }
    // Metal draws nothing: an object threadgroup of no thread or past 1024, as a mesh one.
    for draw in [
        groups([3, 1, 1], [0, 1, 1], [2, 1, 1]),
        groups([3, 1, 1], [1025, 1, 1], [2, 1, 1]),
        groups([3, 1, 1], [4, 1, 1], [0, 1, 1]),
        threads([8, 1, 1], [0, 1, 1]),
    ] {
        let cmds = instantiate(&t, &state, &draw).expect("nothing");
        assert_eq!(cmds.draw, tasks(0, 0, 0), "{draw:?}");
    }
    let limit = |limit, required, available| {
        Some(BindError::Unsupported(Unsupported::needs(
            Feature::MeshShading,
            Capability::Limit {
                limit,
                required,
                available,
            },
        )))
    };
    for (draw, refusal) in [
        (
            groups([1; 3], [256, 1, 1], [1; 3]),
            limit(VkLimit::MaxTaskWorkGroupSize(0), 256, 128),
        ),
        (
            groups([1; 3], [16, 16, 1], [1; 3]),
            limit(VkLimit::MaxTaskWorkGroupInvocations, 256, 128),
        ),
        (
            groups([1; 3], [1; 3], [512, 1, 1]),
            limit(VkLimit::MaxMeshWorkGroupSize(0), 512, 256),
        ),
        (
            groups([1, 1, 65536], [1; 3], [1; 3]),
            limit(VkLimit::MaxTaskWorkGroupCount(2), 65536, 65535),
        ),
        (
            groups([2048, 2049, 1], [1; 3], [1; 3]),
            limit(VkLimit::MaxTaskWorkGroupTotalCount, 2048 * 2049, 1 << 22),
        ),
    ] {
        assert_eq!(instantiate(&t, &state, &draw).err(), refusal, "{draw:?}");
    }
    // Past what Metal runs: nothing drawn, and the command buffer discarded, ahead of the task
    // limits.
    for counts in [[(1 << 29) + 1, 1, 1], [65535, 65535, 1]] {
        let cmds = instantiate(&t, &state, &groups(counts, [1; 3], [2, 1, 1])).expect("discarded");
        assert_eq!(cmds.draw, tasks(0, 0, 0), "{counts:?}");
        assert!(cmds.discards_command_buffer, "{counts:?}");
    }
    // Threads per grid wrapped to 32 bits: the threadgroups they fill, rounded up (Metal's), none
    // where they wrap to 0, held to the task limits; the region reports them, read (`MultiRegion`
    // 1) where they leave a partial threadgroup.
    for (counts, lo, runs, reported, multi_region) in [
        ([33_554_433, 1, 1], [128, 1, 1], [1, 1, 1], [128, 1, 1], 0),
        ([44_739_243, 1, 1], [96, 1, 1], [1, 1, 1], [32, 1, 1], 1),
        ([33_554_432, 1, 1], [128, 1, 1], [0, 0, 0], [0, 1, 1], 0),
    ] {
        let cmds = instantiate(&t, &state, &groups(counts, lo, [2, 1, 1])).expect("wraps");
        let [x, y, z] = runs;
        assert_eq!(cmds.draw, tasks(x, y, z), "{counts:?}");
        if runs[0] == 0 {
            continue;
        }
        let key = cmds.specialization.as_ref().expect("a mesh draw's key");
        assert!(values(key).contains(&(8, multi_region)), "{counts:?}");
        assert_eq!(
            cmds.push_constants[0].bytes,
            region(reported, lo),
            "{counts:?}"
        );
    }
    // Indirect: its object threadgroups' counts, through the object form of the arguments pass;
    // an object threadgroup Metal draws nothing for reads none.
    let indirect = |lo| DrawCall::MeshThreadgroupsIndirect {
        indirect: IndirectRef::new(1, 12),
        threads_per_object_group: lo,
        threads_per_mesh_group: [32, 1, 1],
    };
    let cmds = instantiate(&t, &state, &indirect([4, 1, 1])).expect("indirect");
    assert!(matches!(
        cmds.draw,
        vk::DrawCommand::DrawMeshTasksIndirect { offset: 12, .. }
    ));
    assert!(!cmds.indirect_needs_arguments && cmds.uses.len() == 1);
    assert_eq!(cmds.mesh_arguments, Some(MeshArgumentsForm::Object));
    let cmds = instantiate(&t, &state, &indirect([0, 1, 1])).expect("indirect nothing");
    assert_eq!(cmds.draw, tasks(0, 0, 0));
    assert!(cmds.uses.is_empty() && cmds.mesh_arguments.is_none());
}

/// A folded object function's draws (`MeshTemplate::folded`): a direct draw runs its object grid
/// in the pieces of its fold within the task limits, as `vkCmdDrawMeshTasksEXT` of its one piece or
/// `vkCmdDrawMeshTasksIndirectEXT` of its pieces' uploaded commands, its object stage reading the
/// fold's record at its `ReplayRegionAddress` hole and every grid builtin from it (`MultiRegion`
/// 1); a grid past what runs draws nothing and says so; threads per grid wrap as Metal's; and an
/// indirect draw is `vkCmdDrawMeshTasksIndirectCountEXT` of the folded arguments pass's pieces.
#[test]
fn folded_object_draws_run_their_grid_in_pieces() {
    use crate::bind::mesh_arguments::{fold_object_grid, MeshArguments, MeshGridLimits};
    use crate::pipeline::interface::PlacedField;
    let state = BindState::default();
    let mut t = object_template();
    t.stages[0].fields = vec![PlacedField {
        field: PushField::ReplayRegionAddress,
        loc: PushLoc::Push { offset: 0 },
        size: 8,
    }];
    t.stages[0].push_range = Some((0, 8));
    t.mesh.as_mut().expect("a mesh template").folded = true;
    let task = MeshGridLimits {
        axes: [65535; 3],
        total: 1 << 22,
    };
    let groups = |groups, lo| DrawCall::MeshThreadgroups {
        groups,
        threads_per_object_group: lo,
        threads_per_mesh_group: [2, 1, 1],
    };
    let tasks = |[x, y, z]: [u32; 3]| vk::DrawCommand::DrawMeshTasks {
        group_count_x: x,
        group_count_y: y,
        group_count_z: z,
    };
    let hole = PushHole::ReplayRegionAddress {
        stages: vk::ShaderStageFlags::TASK_EXT,
        offset: 0,
    };
    // One piece, a grid past the task stage's axis among them: no refusal.
    for (counts, lo, dispatch) in [
        ([3, 1, 1], [4, 2, 1], [3, 1, 1]),
        ([1, 1, 65536], [1; 3], [65535, 2, 1]),
        ([131_070, 2, 1], [1; 3], [65535, 4, 1]),
    ] {
        let cmds = instantiate(&t, &state, &groups(counts, lo)).expect("one piece");
        let fold = fold_object_grid(counts, lo, task, true).expect("fold");
        assert_eq!(fold.dispatch, dispatch, "{counts:?}");
        assert_eq!(cmds.draw, tasks(dispatch), "{counts:?}");
        assert_eq!(cmds.object_fold, Some(fold), "{counts:?}");
        assert!(cmds.push_holes.contains(&hole), "{counts:?}");
        assert!(
            cmds.mesh_arguments.is_none() && cmds.uses.is_empty(),
            "{counts:?}"
        );
        let key = cmds.specialization.as_ref().expect("a mesh draw's key");
        let multi_region = key.entries.iter().find(|e| e.constant_id == 8).map(|e| {
            let at = e.offset as usize;
            u32::from_le_bytes(key.data[at..at + 4].try_into().expect("4 bytes"))
        });
        assert_eq!(multi_region, Some(1), "{counts:?}");
    }
    // Pieces: the uploaded commands, at their offset in the upload.
    let cmds = instantiate(&t, &state, &groups([70_000, 100, 1], [1; 3])).expect("pieces");
    assert_eq!(
        cmds.draw,
        vk::DrawCommand::DrawMeshTasksIndirect {
            offset: u64::from(MeshArguments::OBJECT_COMMANDS),
            draw_count: 2,
            stride: 12,
        }
    );
    let fold = cmds.object_fold.expect("a fold");
    assert_eq!(
        (fold.dispatch, fold.pieces, fold.past),
        ([65535, 64, 1], 2, false)
    );
    assert_eq!(fold.bytes().len(), 36 + 2 * 12);
    assert!(cmds.mesh_arguments.is_none());
    // 2^29 threadgroups run, in 129 pieces of these limits; more, which Metal's GPU discards,
    // draw nothing and discard the command buffer.
    let cmds = instantiate(&t, &state, &groups([1 << 29, 1, 1], [1; 3])).expect("2^29");
    let fold = cmds.object_fold.expect("a fold");
    assert_eq!((fold.pieces, fold.discarded), (129, false));
    assert!(!cmds.discards_command_buffer);
    for counts in [[65535, 65535, 1], [(1 << 29) + 1, 1, 1]] {
        let cmds = instantiate(&t, &state, &groups(counts, [1; 3])).expect("discarded");
        assert_eq!(cmds.draw, tasks([0; 3]), "{counts:?}");
        assert!(
            cmds.object_fold.is_some_and(|f| f.discarded && !f.past),
            "{counts:?}"
        );
        assert!(cmds.discards_command_buffer, "{counts:?}");
    }
    // Threads per grid wrapped to 32 bits: a direct draw runs the threadgroups they fill, rounded
    // up (Metal's), none where they wrap to 0, and the record reports them.
    let cmds = instantiate(&t, &state, &groups([33_554_433, 1, 1], [128, 1, 1])).expect("wraps");
    assert_eq!(cmds.draw, tasks([1, 1, 1]));
    let fold = cmds.object_fold.expect("a fold");
    assert_eq!(fold.record[..3], [1, 1, 1]);
    assert_eq!(fold.record[4..7], [128, 1, 1]);
    // Nothing Metal draws: no fold past anything.
    for (counts, lo) in [([3, 0, 1], [4, 1, 1]), ([1 << 26, 1, 1], [128, 1, 1])] {
        let cmds = instantiate(&t, &state, &groups(counts, lo)).expect("nothing");
        assert_eq!(cmds.draw, tasks([0; 3]), "{counts:?}");
        assert!(
            cmds.object_fold.is_some_and(|f| !f.past && f.pieces == 0),
            "{counts:?}"
        );
    }
    // Indirect: the folded arguments pass's pieces.
    let indirect = DrawCall::MeshThreadgroupsIndirect {
        indirect: IndirectRef::new(1, 12),
        threads_per_object_group: [4, 1, 1],
        threads_per_mesh_group: [32, 1, 1],
    };
    let cmds = instantiate(&t, &state, &indirect).expect("indirect");
    assert_eq!(
        cmds.draw,
        vk::DrawCommand::DrawMeshTasksIndirectCount {
            offset: u64::from(MeshArguments::OBJECT_COMMANDS),
            count_offset: u64::from(MeshArguments::OBJECT_COUNT),
            max_draw_count: MeshArguments::OBJECT_PIECES,
            stride: 12,
        }
    );
    assert_eq!(cmds.mesh_arguments, Some(MeshArgumentsForm::FoldedObject));
    assert!(cmds.object_fold.is_none() && cmds.push_holes.contains(&hole));
    assert!(!cmds.indirect_needs_arguments && cmds.uses.len() == 1);
}

/// A mesh draw is `vkCmdDrawMeshTasksEXT` of its threadgroups, made with its threadgroup's size as
/// its specialization (`MultiRegion` 0 beside the plan's constants) and writing its grid in
/// `DispatchRegion`; `drawMeshThreads` launches the whole threadgroups its threads fill, and writes
/// their threads as its grid where the threads leave a partial threadgroup (Metal's builtins report
/// them, a probe); what Metal draws nothing for launches nothing, a grid with an axis past 65,535
/// or of more than 1,048,575 threadgroups among it; the device's mesh limits refuse typed; an
/// indirect draw's arguments pass reads 12 bytes of arguments in the compute stage; and a mesh
/// draw of any other pipeline, or another draw of a mesh pipeline, is `InvalidMeshDraw`.
#[test]
fn mesh_draws_specialize_their_threadgroup_within_the_device() {
    let state = BindState::default();
    let t = mesh_template(false, Vec::new());
    let mesh = |groups, l| DrawCall::MeshThreadgroups {
        groups,
        threads_per_object_group: [1; 3],
        threads_per_mesh_group: l,
    };
    let threads = |threads, l| DrawCall::MeshThreads {
        threads,
        threads_per_object_group: [1; 3],
        threads_per_mesh_group: l,
    };
    let tasks = |x, y, z| vk::DrawCommand::DrawMeshTasks {
        group_count_x: x,
        group_count_y: y,
        group_count_z: z,
    };
    // `(constant id, value)` of a key.
    let values = |key: &SpecializationKey| -> Vec<(u32, u32)> {
        key.entries
            .iter()
            .map(|e| {
                let at = e.offset as usize;
                let bytes = key.data[at..at + 4].try_into().expect("4 bytes");
                (e.constant_id, u32::from_le_bytes(bytes))
            })
            .collect()
    };
    let cmds = instantiate(&t, &state, &mesh([3, 2, 1], [4, 2, 1])).expect("threadgroups");
    assert_eq!(cmds.draw, tasks(3, 2, 1));
    assert_eq!(
        values(cmds.specialization.as_ref().expect("a mesh draw's key")),
        [(0, 4), (1, 2), (2, 1), (3, 0), (4, 1)]
    );
    assert_eq!(
        cmds.push_constants,
        [PushConstantWrite {
            stages: vk::ShaderStageFlags::MESH_EXT,
            offset: 0,
            bytes: region_bytes([0; 3], [12, 4, 1], [4, 2, 1])
                .expect("region")
                .to_vec(),
        }]
    );
    assert!(cmds.uses.is_empty() && !cmds.indirect_needs_arguments);
    // `drawMeshThreads`: floor(t / m) threadgroups (20 by 16: one), whose threads are the grid,
    // whether or not the function reads it (10 by 3 in 4 by 2: an 8 by 2 grid, a probe).
    let mut grid_free = mesh_template(false, Vec::new());
    if let Some(m) = grid_free.mesh.as_mut() {
        m.spec.retain(|c| c.key != SpecKey::MultiRegion);
    }
    let cmds = instantiate(&grid_free, &state, &threads([20, 1, 1], [16, 1, 1])).expect("threads");
    assert_eq!(cmds.draw, tasks(1, 1, 1));
    let cmds = instantiate(&t, &state, &threads([32, 4, 1], [16, 2, 1])).expect("whole");
    assert_eq!(cmds.draw, tasks(2, 2, 1));
    let cmds = instantiate(&t, &state, &threads([10, 3, 1], [4, 2, 1])).expect("partial");
    assert_eq!(cmds.draw, tasks(2, 1, 1));
    assert_eq!(
        cmds.push_constants[0].bytes,
        region_bytes([0; 3], [8, 2, 1], [4, 2, 1])
            .expect("region")
            .to_vec()
    );
    // Metal draws nothing: no thread, past 1024 threads, no threadgroup, a grid past Metal's
    // (`bind::mesh_arguments`). Past the device's 256 threads but within Metal's 1024 is the
    // device's limit.
    for (draw, what) in [
        (mesh([3, 1, 1], [0, 1, 1]), "no thread"),
        (mesh([3, 1, 1], [1025, 1, 1]), "past 1024"),
        (mesh([3, 0, 1], [512, 1, 1]), "no threadgroup"),
        (threads([15, 1, 1], [16, 1, 1]), "no whole threadgroup"),
        (mesh([1, 65536, 1], [1; 3]), "65,536 along y"),
        (mesh([65535, 65535, 1], [1; 3]), "past Metal's total"),
    ] {
        let cmds = instantiate(&t, &state, &draw).expect(what);
        let vk::DrawCommand::DrawMeshTasks {
            group_count_x,
            group_count_y,
            group_count_z,
        } = cmds.draw
        else {
            panic!("{what}: {:?}", cmds.draw);
        };
        assert!(
            [group_count_x, group_count_y, group_count_z].contains(&0),
            "{what}: {:?}",
            cmds.draw
        );
        assert!(cmds.specialization.is_some(), "{what}");
    }
    let limit = |limit, required, available| {
        Some(BindError::Unsupported(Unsupported::needs(
            Feature::MeshShading,
            Capability::Limit {
                limit,
                required,
                available,
            },
        )))
    };
    for (draw, refusal) in [
        (
            mesh([1; 3], [512, 1, 1]),
            limit(VkLimit::MaxMeshWorkGroupSize(0), 512, 256),
        ),
        (
            mesh([1; 3], [1, 1, 300]),
            limit(VkLimit::MaxMeshWorkGroupSize(2), 300, 256),
        ),
        (
            mesh([1; 3], [16, 32, 1]),
            limit(VkLimit::MaxMeshWorkGroupInvocations, 512, 256),
        ),
    ] {
        assert_eq!(instantiate(&t, &state, &draw).err(), refusal, "{draw:?}");
    }
    // A flattened stage: its threads as one axis, whole subgroups of its tier only.
    let flat = mesh_template(true, vec![32]);
    let cmds = instantiate(&flat, &state, &mesh([1; 3], [8, 8, 1])).expect("flat");
    assert!(values(cmds.specialization.as_ref().expect("key")).contains(&(5, 64)));
    assert_eq!(
        instantiate(&flat, &state, &mesh([1; 3], [16, 17, 1])).err(),
        limit(VkLimit::MaxMeshWorkGroupSize(0), 272, 256)
    );
    assert_eq!(
        instantiate(&flat, &state, &mesh([1; 3], [6, 1, 1])).err(),
        Some(not_implemented(Feature::MeshShading))
    );
    // A draw that launches nothing takes a threadgroup the stage can be made with.
    let cmds = instantiate(&flat, &state, &mesh([1; 3], [0, 1, 1])).expect("nothing");
    assert!(values(cmds.specialization.as_ref().expect("key")).contains(&(5, 32)));
    // A grid Metal draws nothing for launches nothing, past the device's limits too.
    for groups in [[65536, 1, 1], [1024, 1024, 1], [1, 1, 70000]] {
        let cmds = instantiate(&t, &state, &mesh(groups, [32, 1, 1])).expect("nothing");
        assert_eq!(cmds.draw, tasks(0, 0, 0), "{groups:?}");
    }
    // Indirect: 12 bytes of arguments, read by the mesh form of the arguments pass; at an offset
    // that is not a multiple of 4, nothing.
    let indirect = |offset| DrawCall::MeshThreadgroupsIndirect {
        indirect: IndirectRef::new(1, offset),
        threads_per_object_group: [1; 3],
        threads_per_mesh_group: [32, 1, 1],
    };
    let cmds = instantiate(&t, &state, &indirect(12)).expect("indirect");
    assert_eq!(
        cmds.draw,
        vk::DrawCommand::DrawMeshTasksIndirect {
            offset: 12,
            draw_count: 1,
            stride: 12,
        }
    );
    assert!(!cmds.indirect_needs_arguments && !cmds.index_bounds_pass);
    assert_eq!(cmds.mesh_arguments, Some(MeshArgumentsForm::Mesh));
    assert_eq!(
        cmds.uses,
        [ResourceUse {
            resource: UseTarget::Indirect,
            access: AccessSet::READ,
            class: AccessClass::Storage,
            stages: vk::PipelineStageFlags2::COMPUTE_SHADER,
            range: Some(ByteRange { offset: 0, len: 12 }),
        }]
    );
    let cmds = instantiate(&t, &state, &indirect(2)).expect("misaligned");
    assert_eq!(cmds.draw, tasks(0, 0, 0));
    assert!(cmds.uses.is_empty() && cmds.mesh_arguments.is_none());
    // A draw of primitives with a mesh pipeline, and a mesh draw with another.
    assert_eq!(
        instantiate(&t, &state, &primitives(0, 3, (0, 1))).err(),
        Some(BindError::InvalidMeshDraw { mesh: false })
    );
    let mut other = mesh_template(false, Vec::new());
    other.mesh = None;
    other.stages.clear();
    assert_eq!(
        instantiate(&other, &state, &mesh([1; 3], [1; 3])).err(),
        Some(BindError::InvalidMeshDraw { mesh: true })
    );
    assert_eq!(
        instantiate(&other, &state, &primitives(0, 3, (0, 1))).map(|c| c.specialization),
        Ok(None)
    );
}

/// `PushField::ShaderStencilState`'s words: each face's word in its documented bits (the
/// shader-side stencil's fields, the depth-failure operation in bits 25 to 27, masks cut to 8
/// bits), the front face's first, then the references' low 8 bits (front in bits 0 to 7, back in
/// bits 8 to 15), then zero.
#[test]
fn shader_stencil_words_pack_each_face_then_the_references() {
    use metal::{CompareFunction, StencilOperation};
    let mut state = BindState::default();
    let front = &mut state.depth_stencil.front_face_stencil;
    front.stencil_compare_function = CompareFunction::Equal;
    front.stencil_failure_operation = StencilOperation::Zero;
    front.depth_failure_operation = StencilOperation::DecrementWrap;
    front.depth_stencil_pass_operation = StencilOperation::Replace;
    front.read_mask = 0x0F;
    front.write_mask = 0x1F0;
    let back = &mut state.depth_stencil.back_face_stencil;
    back.stencil_compare_function = CompareFunction::Greater;
    back.depth_failure_operation = StencilOperation::Invert;
    state.stencil_references = [0x1A5, 0x3C];
    let words: Vec<u32> = shader_stencil_words(&state)
        .chunks(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let field = |w: u32, offset: u32, width: u32| (w >> offset) & ((1 << width) - 1);
    let [f, b, references, zero] = words[..] else {
        panic!("{words:?}");
    };
    // compare, stencil failure, pass, read mask, write mask, depth failure
    let fields = |w| {
        [
            field(w, 0, 3),
            field(w, 3, 3),
            field(w, 6, 3),
            field(w, 9, 8),
            field(w, 17, 8),
            field(w, 25, 3),
        ]
    };
    assert_eq!(fields(f), [2, 1, 2, 0x0F, 0xF0, 7]);
    assert_eq!(fields(b), [4, 0, 0, 0xFF, 0xFF, 5]);
    assert_eq!([f >> 28, b >> 28], [0, 0], "bits 28 to 31 stay clear");
    assert_eq!(references, 0xA5 | 0x3C << 8);
    assert_eq!(zero, 0);
}
