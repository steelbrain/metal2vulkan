//! Metal's constant vertex step across vertices, instances, a vertex start and a base instance.
//!
//! `MTLVertexStepFunctionConstant` fetches a layout's first element for every vertex of every
//! instance. The other cases that set it (`vertex-format-constant-step`, `integer-vertex-attributes`)
//! draw one instance from vertex 0, so the element they name is the first under any step that
//! counts from the draw's own start. This case names elements no other step could reach the first
//! of: two vertices of two instances, from vertex 0 and base instance 0 (the control), from base
//! instance 1, from vertex start `2^30`, and from base instance `2^30`.
//!
//! # How a case is read
//!
//! The fixture `vertex_fetch_reach` writes `(v, vid + iid)` at `out[(vid + iid) & 3]` (each id with
//! its base, as Metal counts them), and the attribute buffer, a layout at the constant step of
//! stride 4, holds `0x10 + w` at word `w`, so each slot names the element its vertex fetched. Each
//! draw writes its own segment of `out`, which is made holding `UNWRITTEN` and read whole. Two of a
//! draw's four invocations share a slot (vertex 1 of instance 0 and vertex 0 of instance 1 have one
//! sum), and they write the same words exactly where both fetched the same element.
//!
//! | draw | vertices | instances | slots written |
//! |---|---|---|---|
//! | from vertex 0 | 0, 1 | 0, 1 | 0, 1, 2 |
//! | indices 1 and 2, base instance 1 | 1, 2 | 1, 2 | 2, 3, 0 |
//! | from vertex `2^30` | `2^30`, `2^30 + 1` | 0, 1 | 0, 1, 2 |
//! | indices 0 and 1, base instance `2^30` | 0, 1 | `2^30`, `2^30 + 1` | 0, 1, 2 |
//!
//! At stride 4 a per-vertex or per-instance fetch of element `2^30` lies past the 64-byte buffer; the
//! constant step reads its first word whatever the draw names.
//!
//! # Safety
//!
//! Rasterization is disabled and each draw is four points of a loop-free vertex function, so
//! nothing is rasterized and nothing loops. Every write lies inside its draw's segment of `out`
//! (`every_write_lies_inside_out`), and the constant step reads the attribute buffer's first word.

use super::indirect_commands::{
    first_refused_piece, fixture_inputs, program_node, shaders_of, UNWRITTEN,
};
use crate::allowance::Allowance;
use crate::case::{call, made, Abort, Case, Failure, Handle, Input, MetalSource, Side};
use crate::fixture::Fixture;
use crate::recording::ShaderRef;
use crate::render::{
    RenderPass, RenderPipelineSpec, RenderStep, RenderWork, StageFunction, VertexAttribute,
    VertexBufferLayout, VertexLayout,
};
use crate::transcript::{ByteLayout, Element, Node, Observed, Sites};
use metal2vulkan::metal::{IndexType, PrimitiveType, VertexFormat};

/// The owned fixture: a rasterization-disabled vertex function that fetches one `uint`
/// `[[stage_in]]` attribute and writes it with its ids' sum at the sum's low two bits.
const FIXTURE: &str = "vertex_fetch_reach";

/// The vertex buffer index the attribute's layout reads; 0 is the function's own `out`.
const ATTRIBUTES_INDEX: u32 = 1;

/// The words the attribute buffer holds: 64 bytes.
const WORDS: u32 = 16;

/// The layout's stride: one word an element.
const STRIDE: u32 = 4;

/// A vertex start or base instance past `2^30`: element `2^30` lies at `2^32` at [`STRIDE`].
const FAR: u32 = 1 << 30;

/// Vertices each draw names, and instances.
const VERTICES: u32 = 2;
const INSTANCES: u32 = 2;

/// The slots of `out` one draw's sums can reach: the fixture writes `(vid + iid) & 3`.
const SLOTS: u64 = 4;

/// Bytes one draw's segment of `out` holds: a `uint2` a slot.
const SEGMENT_BYTES: u64 = SLOTS * 8;

/// The indices the indexed draws read, 32-bit.
const INDICES: [u32; 4] = [0, 1, 2, 3];

/// One `UInt` attribute at offset 0 of buffer [`ATTRIBUTES_INDEX`].
const ATTRIBUTES: [VertexAttribute; 1] = [VertexAttribute {
    index: 0,
    format: VertexFormat::UInt,
    offset: 0,
    buffer: ATTRIBUTES_INDEX,
}];

/// Buffer [`ATTRIBUTES_INDEX`] at the constant step, of stride [`STRIDE`].
const LAYOUTS: [VertexBufferLayout; 1] = [VertexBufferLayout::constant(ATTRIBUTES_INDEX, STRIDE)];

/// The pipeline: the fixture's vertex function, rasterization disabled, under [`LAYOUTS`].
fn pipeline() -> RenderPipelineSpec<'static> {
    let mut spec = RenderPipelineSpec::of(StageFunction::named(FIXTURE), None, &[]);
    spec.rasterization = false;
    spec.vertex_layout = Some(VertexLayout::new(&ATTRIBUTES, &LAYOUTS));
    spec
}

/// The attribute buffer's bytes: `0x10 + w` at word `w`.
fn attribute_bytes() -> Vec<u8> {
    (0..WORDS).flat_map(|w| (0x10 + w).to_le_bytes()).collect()
}

/// The index buffer's bytes: [`INDICES`].
fn index_bytes() -> Vec<u8> {
    INDICES.iter().flat_map(|i| i.to_le_bytes()).collect()
}

/// [`VERTICES`] vertices of [`INSTANCES`] instances from vertex `first`, base instance 0.
fn from_vertex(first: u32) -> RenderStep {
    RenderStep::Draw {
        primitive: PrimitiveType::Point,
        first,
        count: VERTICES,
        instances: INSTANCES,
    }
}

/// [`VERTICES`] indices of `indices` from index `from`, [`INSTANCES`] instances from
/// `base_instance`.
fn from_instance(indices: Handle, from: u64, base_instance: u32) -> RenderStep {
    RenderStep::DrawIndexed {
        primitive: PrimitiveType::Point,
        count: VERTICES,
        index_type: IndexType::UInt32,
        indices,
        offset: from * 4,
        instances: INSTANCES,
        base_vertex: 0,
        base_instance,
    }
}

/// The draws, in the module's table's order.
fn draws(indices: Handle) -> Vec<RenderStep> {
    vec![
        from_vertex(0),
        from_instance(indices, 1, 1),
        from_vertex(FAR),
        from_instance(indices, 0, FAR),
    ]
}

/// Bytes `out` holds: one segment a draw.
fn out_bytes() -> u64 {
    SEGMENT_BYTES * draws(Handle(0)).len() as u64
}

/// The pass: no attachment at a 1 by 1 extent, the pipeline and the attributes bound once, and per
/// draw `out` bound at its segment, and the draw.
fn pass(pipeline: Handle, out: Handle, attributes: Handle, indices: Handle) -> RenderPass {
    let mut steps = vec![
        RenderStep::Pipeline(pipeline),
        RenderStep::VertexBuffer {
            buffer: attributes,
            offset: 0,
            index: ATTRIBUTES_INDEX,
        },
    ];
    for (at, draw) in draws(indices).into_iter().enumerate() {
        steps.push(RenderStep::VertexBuffer {
            buffer: out,
            offset: at as u64 * SEGMENT_BYTES,
            index: 0,
        });
        steps.push(draw);
    }
    RenderPass::sized([1, 1]).drawing(steps)
}

/// The case: the draws of the module's table.
pub struct VertexConstantStep;

impl Case for VertexConstantStep {
    fn name(&self) -> &'static str {
        "vertex-constant-step-instances"
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
        inputs.push(Input::bytes("pipeline", &pipeline().words()));
        inputs.push(Input::bytes("buffer:attributes", &attribute_bytes()));
        inputs.push(Input::bytes("buffer:indices", &index_bytes()));
        inputs.push(Input::bytes(
            "buffer:out",
            &vec![UNWRITTEN; out_bytes() as usize],
        ));
        inputs.push(Input::bytes(
            "pass",
            &pass(Handle(0), Handle(3), Handle(1), Handle(2)).words(),
        ));
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let loaded = Fixture::new(FIXTURE).loaded().map_err(|why| Abort {
            at: "library".to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        })?;
        let library = made(sites, "library", side.library(&loaded.source()))?;
        let pipeline = made(
            sites,
            "pipeline",
            side.render_pipeline(library, &pipeline()),
        )?;
        let attributes = made(sites, "buffer:attributes", side.buffer(&attribute_bytes()))?;
        let indices = made(sites, "buffer:indices", side.buffer(&index_bytes()))?;
        let out = made(
            sites,
            "buffer:out",
            side.buffer(&vec![UNWRITTEN; out_bytes() as usize]),
        )?;
        let work = vec![RenderWork::Pass(pass(pipeline, out, attributes, indices))];
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

/// The family's case.
pub fn all() -> Vec<Box<dyn Case>> {
    vec![Box::new(VertexConstantStep)]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcript::Value;

    /// Every draw writes its own segment and nothing past `out`: the fixture writes slots 0 to 3
    /// of the segment it is bound at, and the last segment ends at `out`'s end.
    #[test]
    fn every_write_lies_inside_out() {
        let draws = draws(Handle(0)).len() as u64;
        assert_eq!((draws - 1) * SEGMENT_BYTES + 8 * SLOTS, out_bytes());
    }

    /// The far draws name elements a per-vertex or per-instance step would fetch past the
    /// attribute buffer, while the constant step's element, the first, lies inside it.
    #[test]
    fn the_far_draws_name_elements_past_the_buffer() {
        let bytes = u64::from(WORDS) * 4;
        assert!(u64::from(FAR) * u64::from(STRIDE) >= bytes);
        assert!(u64::from(STRIDE) <= bytes);
    }

    /// A draw's indexed form reads only the indices the buffer holds.
    #[test]
    fn the_indexed_draws_read_inside_the_indices() {
        for draw in draws(Handle(0)) {
            if let RenderStep::DrawIndexed { count, offset, .. } = draw {
                assert!(offset + 4 * u64::from(count) <= index_bytes().len() as u64);
            }
        }
    }

    /// Metal's answer: every slot a draw's sums reach holds the attribute buffer's first word with
    /// its sum, and the one slot they do not reach is `UNWRITTEN`, in every draw.
    #[test]
    fn metal_fetched_the_first_element_at_every_vertex_and_instance() {
        let recording = crate::recording::read(VertexConstantStep.name()).expect("the recording");
        let answer = recording.answer().expect("an answer of the oracle");
        let transcript = answer.deterministic().expect("one outcome");
        let Some(Value::Bytes(words)) = transcript.root.child("out").map(|n| &n.observed.value)
        else {
            panic!("the recording holds no read-back of out");
        };
        let word = |at: usize| u32::from_le_bytes(words[4 * at..4 * at + 4].try_into().unwrap());
        let unwritten = u32::from_le_bytes([UNWRITTEN; 4]);
        for (draw, (vertex, instance)) in
            [(0, 0), (1, 1), (FAR, 0), (0, FAR)].into_iter().enumerate()
        {
            let mut want = [[unwritten; 2]; SLOTS as usize];
            for v in vertex..vertex + VERTICES {
                for i in instance..instance + INSTANCES {
                    let sum = v + i;
                    want[(sum & 3) as usize] = [0x10, sum];
                }
            }
            let base = draw * SLOTS as usize * 2;
            let got: Vec<[u32; 2]> = (0..SLOTS as usize)
                .map(|slot| [word(base + 2 * slot), word(base + 2 * slot + 1)])
                .collect();
            assert_eq!(got, want, "draw {draw}");
        }
    }
}
