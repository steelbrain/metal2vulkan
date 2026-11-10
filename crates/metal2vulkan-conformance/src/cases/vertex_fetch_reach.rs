//! A vertex attribute at a layout's fixed stride, fetched at elements whose displacement from the
//! buffer's start lies past `2^32` bytes, over a **small** buffer.
//!
//! Metal places an attribute's element `e` at `e · stride + offset` from the bind offset
//! (`render/vertex_reach.rs` in the runtime has the measurement in a buffer past 4 GiB, and
//! `attribute-stride-vertex-buffer` the same at the strides an encoder binds). A fixed stride is a
//! few bytes, so such an element is one a draw names past `2^30` or so: a vertex start, a base
//! vertex, an index, or a base instance. Over a buffer of 64 bytes each of those elements lies past
//! the buffer, while a fetch that computed the displacement in 32 bits would wrap back inside it.
//! The case asks what Metal reads there, through both of our fetches: the device's own
//! fixed-function input (a stride of 4 or 12, aligned for a 32-bit format) and the pulled one (a
//! stride of 6, which fixed-function input cannot align a 32-bit format to).
//!
//! # The draws
//!
//! One draw a segment of `out`, four vertices of one instance or one vertex of four instances:
//!
//! | layout | draw | elements | their 32-bit displacements |
//! |---|---|---|---|
//! | per vertex, stride 4 | vertex start 0 (the control) | 0 to 3 | the same |
//! | per vertex, stride 4 | vertex start `2^30` | `2^30` to `2^30 + 3` | 0, 4, 8, 12 |
//! | per vertex, stride 4 | indices 0 to 3, base vertex `2^30` | the same | the same |
//! | per vertex, stride 4 | indices `2^30` to `2^30 + 3` | the same | the same |
//! | per vertex, stride 12 | vertex start 0 (the control) | 0 to 3 | the same |
//! | per vertex, stride 12 | vertex start [`WRAP_12`] | from `WRAP_12` | 8, 20, 32, 44 |
//! | per vertex, stride 6 | vertex start 0 (the control) | 0 to 3 | the same |
//! | per vertex, stride 6 | vertex start [`WRAP_6`] | from `WRAP_6` | 2, 8, 14, 20 |
//! | per instance, stride 4, 12, 6 | base instance 0 (the controls) | 0 to 3 | the same |
//! | per instance, stride 4 | base instance `2^30` | `2^30` to `2^30 + 3` | 0, 4, 8, 12 |
//! | per instance, stride 12 | base instance [`WRAP_12`] | from `WRAP_12` | 8, 20, 32, 44 |
//! | per instance, stride 6 | base instance [`WRAP_6`] | from `WRAP_6` | 2, 8, 14, 20 |
//!
//! A 64-bit displacement puts every far element past the buffer; a 32-bit one reads inside it,
//! at a word a 4-byte stride never starts at where the stride is 6.
//!
//! # How a case is read
//!
//! The fixture `vertex_fetch_reach` writes `(v, vid + iid)` at `out[(vid + iid) & 3]` (each id with
//! its base, as Metal counts them), and the attribute buffer holds `0x10 + w` at word `w`, so each
//! segment's words name the bytes its draw fetched. `out` is made holding `UNWRITTEN` and read
//! whole.
//!
//! # Safety
//!
//! Rasterization is disabled and each draw is four points of a loop-free vertex function, so
//! nothing is rasterized and nothing loops. The fetches past the buffer are the measurement; every
//! other byte any draw reads lies inside the attribute or index buffer, and every write lies inside
//! its segment of `out` (`every_write_lies_inside_out`).

use super::indirect_commands::{
    first_refused_piece, fixture_inputs, program_node, shaders_of, UNWRITTEN,
};
use crate::allowance::Allowance;
use crate::case::{call, made, Abort, Case, Failure, Handle, Input, MetalSource, Side};
use crate::fixture::Fixture;
use crate::recording::ShaderRef;
use crate::render::{
    RenderPass, RenderPipelineSpec, RenderStep, RenderWork, StageFunction, VertexAttribute,
    VertexBufferLayout, VertexLayout, VertexStride,
};
use crate::transcript::{ByteLayout, Element, Node, Observed, Sites};
use metal2vulkan::metal::{IndexType, PrimitiveType, VertexFormat, VertexStepFunction};

/// The owned fixture: a rasterization-disabled vertex function that fetches one `uint`
/// `[[stage_in]]` attribute and writes it with its ids' sum at the sum's low two bits.
const FIXTURE: &str = "vertex_fetch_reach";

/// The vertex buffer index the attribute's layout reads; 0 is the function's own `out`.
const ATTRIBUTES_INDEX: u32 = 1;

/// The words the attribute buffer holds: 64 bytes.
const WORDS: u32 = 16;

/// The first element past `2^30`: a stride of 4 places it at `2^32`.
const FAR: u32 = 1 << 30;

/// The first element a stride of 12 places past `2^32`: at `2^32 + 8`.
const WRAP_12: u32 = 357_913_942;

/// The first element a stride of 6 places past `2^32`: at `2^32 + 2`.
const WRAP_6: u32 = 715_827_883;

/// Vertices, or instances, each draw fetches.
const FETCHES: u32 = 4;

/// Bytes one draw's segment of `out` holds: a `uint2` a slot.
const SEGMENT_BYTES: u64 = FETCHES as u64 * 8;

/// The fixed strides of the layouts, per vertex and per instance alike.
const STRIDES: [u32; 3] = [4, 12, 6];

/// The indices the indexed draws read: 0 to 3, then `FAR` to `FAR + 3`, 32-bit.
const INDICES: [u32; 8] = [0, 1, 2, 3, FAR, FAR + 1, FAR + 2, FAR + 3];

/// One `UInt` attribute at offset 0 of buffer [`ATTRIBUTES_INDEX`].
const ATTRIBUTES: [VertexAttribute; 1] = [VertexAttribute {
    index: 0,
    format: VertexFormat::UInt,
    offset: 0,
    buffer: ATTRIBUTES_INDEX,
}];

/// The layout of buffer [`ATTRIBUTES_INDEX`] at `stride`, stepped per vertex or per instance.
const fn layout(stride: u32, step: VertexStepFunction) -> [VertexBufferLayout; 1] {
    [VertexBufferLayout {
        index: ATTRIBUTES_INDEX,
        stride: VertexStride::Bytes(stride),
        step,
        rate: 1,
    }]
}

/// The six layouts, in [`pipelines`]' order: per vertex at each stride, then per instance.
const LAYOUTS: [[VertexBufferLayout; 1]; 6] = [
    layout(STRIDES[0], VertexStepFunction::PerVertex),
    layout(STRIDES[1], VertexStepFunction::PerVertex),
    layout(STRIDES[2], VertexStepFunction::PerVertex),
    layout(STRIDES[0], VertexStepFunction::PerInstance),
    layout(STRIDES[1], VertexStepFunction::PerInstance),
    layout(STRIDES[2], VertexStepFunction::PerInstance),
];

/// The pipelines: the fixture's vertex function, rasterization disabled, under each layout.
fn pipelines() -> Vec<RenderPipelineSpec<'static>> {
    LAYOUTS
        .iter()
        .map(|layouts| {
            let mut spec = RenderPipelineSpec::of(StageFunction::named(FIXTURE), None, &[]);
            spec.rasterization = false;
            spec.vertex_layout = Some(VertexLayout::new(&ATTRIBUTES, layouts));
            spec
        })
        .collect()
}

/// The attribute buffer's bytes: `0x10 + w` at word `w`.
fn attribute_bytes() -> Vec<u8> {
    (0..WORDS).flat_map(|w| (0x10 + w).to_le_bytes()).collect()
}

/// The index buffer's bytes: [`INDICES`].
fn index_bytes() -> Vec<u8> {
    INDICES.iter().flat_map(|i| i.to_le_bytes()).collect()
}

/// One draw: the pipeline (an index of [`pipelines`]) and the draw step.
struct Draw {
    pipeline: usize,
    step: RenderStep,
}

/// Four vertices of pipeline `pipeline` from vertex `first`.
fn vertices(pipeline: usize, first: u32) -> Draw {
    Draw {
        pipeline,
        step: RenderStep::Draw {
            primitive: PrimitiveType::Point,
            first,
            count: FETCHES,
            instances: 1,
        },
    }
}

/// Four indices of `indices` from index `from`, plus `base_vertex`, one instance.
fn indexed(indices: Handle, from: u64, base_vertex: i32) -> Draw {
    Draw {
        pipeline: 0,
        step: RenderStep::DrawIndexed {
            primitive: PrimitiveType::Point,
            count: FETCHES,
            index_type: IndexType::UInt32,
            indices,
            offset: from * 4,
            instances: 1,
            base_vertex,
            base_instance: 0,
        },
    }
}

/// Four instances of pipeline `pipeline` from `base_instance`, of index 0's one vertex.
fn instances(pipeline: usize, indices: Handle, base_instance: u32) -> Draw {
    Draw {
        pipeline,
        step: RenderStep::DrawIndexed {
            primitive: PrimitiveType::Point,
            count: 1,
            index_type: IndexType::UInt32,
            indices,
            offset: 0,
            instances: FETCHES,
            base_vertex: 0,
            base_instance,
        },
    }
}

/// The draws, in the module's table's order.
fn draws(indices: Handle) -> Vec<Draw> {
    vec![
        vertices(0, 0),
        vertices(0, FAR),
        indexed(indices, 0, FAR as i32),
        indexed(indices, 4, 0),
        vertices(1, 0),
        vertices(1, WRAP_12),
        vertices(2, 0),
        vertices(2, WRAP_6),
        instances(3, indices, 0),
        instances(4, indices, 0),
        instances(5, indices, 0),
        instances(3, indices, FAR),
        instances(4, indices, WRAP_12),
        instances(5, indices, WRAP_6),
    ]
}

/// Bytes `out` holds: one segment a draw.
fn out_bytes() -> u64 {
    SEGMENT_BYTES * draws(Handle(0)).len() as u64
}

/// The pass: no attachment at a 1 by 1 extent, the attributes bound once, and per draw its
/// pipeline, `out` bound at its segment, and the draw.
fn pass(pipelines: &[Handle], out: Handle, attributes: Handle, indices: Handle) -> RenderPass {
    let mut steps = vec![RenderStep::VertexBuffer {
        buffer: attributes,
        offset: 0,
        index: ATTRIBUTES_INDEX,
    }];
    for (at, draw) in draws(indices).into_iter().enumerate() {
        steps.push(RenderStep::Pipeline(pipelines[draw.pipeline]));
        steps.push(RenderStep::VertexBuffer {
            buffer: out,
            offset: at as u64 * SEGMENT_BYTES,
            index: 0,
        });
        steps.push(draw.step);
    }
    RenderPass::sized([1, 1]).drawing(steps)
}

/// The case: the draws of the module's table.
pub struct VertexFetchReach;

impl Case for VertexFetchReach {
    fn name(&self) -> &'static str {
        "vertex-fetch-past-4gib-small-buffer"
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // The answer is read back through a `Shared` buffer, so a device without unified memory
        // announces `shared-in-system-memory`; the declaration covers the notices site alone.
        vec![crate::replay::shared_memory_notice()]
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        shaders_of(FIXTURE, FIXTURE)
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let mut inputs = fixture_inputs(FIXTURE)?;
        for (at, spec) in pipelines_named().into_iter().zip(pipelines()) {
            inputs.push(Input::bytes(at, &spec.words()));
        }
        let pipelines: Vec<Handle> = (0..LAYOUTS.len() as u32).map(Handle).collect();
        let first = LAYOUTS.len() as u32;
        inputs.push(Input::bytes("buffer:attributes", &attribute_bytes()));
        inputs.push(Input::bytes("buffer:indices", &index_bytes()));
        inputs.push(Input::bytes(
            "buffer:out",
            &vec![UNWRITTEN; out_bytes() as usize],
        ));
        inputs.push(Input::bytes(
            "pass",
            &pass(
                &pipelines,
                Handle(first + 2),
                Handle(first),
                Handle(first + 1),
            )
            .words(),
        ));
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let loaded = Fixture::new(FIXTURE).loaded().map_err(|why| Abort {
            at: "library".to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        })?;
        let library = made(sites, "library", side.library(&loaded.source()))?;
        let mut pipelines = Vec::new();
        for (at, spec) in pipelines_named().into_iter().zip(self::pipelines()) {
            pipelines.push(made(sites, at, side.render_pipeline(library, &spec))?);
        }
        let attributes = made(sites, "buffer:attributes", side.buffer(&attribute_bytes()))?;
        let indices = made(sites, "buffer:indices", side.buffer(&index_bytes()))?;
        let out = made(
            sites,
            "buffer:out",
            side.buffer(&vec![UNWRITTEN; out_bytes() as usize]),
        )?;
        let work = vec![RenderWork::Pass(pass(&pipelines, out, attributes, indices))];
        let outcome = call(sites, "render", side.render(&work), |outcome| {
            outcome.status.clone()
        })?;
        sites.push(program_node(&work, &outcome));
        if let Some((at, failure)) = first_refused_piece(&work, &outcome) {
            return Err(Abort { at, failure });
        }
        let read = side
            .read_buffer(out, 0, out_bytes() as usize)
            .map_err(|failure| Abort {
                at: "out".to_owned(),
                failure,
            })?;
        sites.push(Node::leaf(
            "out",
            Observed::bytes(read, ByteLayout::buffer(0, out_bytes(), Element::U32)),
        ));
        Ok(())
    }
}

/// The sites of the six pipelines, in [`pipelines`]' order.
fn pipelines_named() -> [&'static str; 6] {
    [
        "pipeline:vertex-4",
        "pipeline:vertex-12",
        "pipeline:vertex-6",
        "pipeline:instance-4",
        "pipeline:instance-12",
        "pipeline:instance-6",
    ]
}

/// The family's case.
pub fn all() -> Vec<Box<dyn Case>> {
    vec![Box::new(VertexFetchReach)]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every draw writes its own segment and nothing past `out`: slot `s` of draw `at` is the
    /// eight bytes at `at · SEGMENT_BYTES + 8 s`, and the fixture writes slots 0 to 3 alone.
    #[test]
    fn every_write_lies_inside_out() {
        let draws = draws(Handle(0)).len() as u64;
        let last = (draws - 1) * SEGMENT_BYTES + 8 * u64::from(FETCHES - 1) + 8;
        assert_eq!(last, out_bytes());
    }

    /// Each far element's 64-bit displacement lies past the buffer and its 32-bit one inside it,
    /// at the bytes the module's table names; every control element lies inside.
    #[test]
    fn the_elements_reach_where_the_table_says() {
        let bytes = u64::from(WORDS) * 4;
        for (stride, first, wrapped) in [
            (4u64, FAR, [0u64, 4, 8, 12]),
            (12, WRAP_12, [8, 20, 32, 44]),
            (6, WRAP_6, [2, 8, 14, 20]),
        ] {
            for k in 0..u64::from(FETCHES) {
                let control = k * stride;
                assert!(control + 4 <= bytes, "stride {stride} control {k}");
                let far = (u64::from(first) + k) * stride;
                assert!(far >= 1 << 32, "stride {stride} element {k}");
                assert_eq!(far % (1 << 32), wrapped[k as usize], "stride {stride}");
                assert!(wrapped[k as usize] + 4 <= bytes);
            }
            // The element before the first lies inside 2^32: the first is the first past it.
            assert!(
                (u64::from(first) - 1) * stride + 4 <= 1 << 32,
                "stride {stride}"
            );
        }
        assert_eq!(STRIDES, [4, 12, 6]);
    }

    /// A draw's indexed form reads only the indices the buffer holds.
    #[test]
    fn the_indexed_draws_read_inside_the_indices() {
        for draw in draws(Handle(0)) {
            if let RenderStep::DrawIndexed { count, offset, .. } = draw.step {
                assert!(offset + 4 * u64::from(count) <= index_bytes().len() as u64);
            }
        }
    }
}
