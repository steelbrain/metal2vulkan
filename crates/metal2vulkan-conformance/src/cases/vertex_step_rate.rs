//! Metal's per-instance vertex step at a rate above 1, from base instance 0 and from a base
//! instance past it.
//!
//! `MTLVertexStepFunctionPerInstance` at `stepRate` `r` fetches a new element once every `r`
//! instances. The other cases that step per instance do so at rate 1, where the element is the
//! instance's own under any count; at a rate above 1 the count's origin shows. This case names it:
//! one vertex of four instances at rates 2 and 3, each from base instance 0 (the controls) and
//! from base instance 5.
//!
//! # How a case is read
//!
//! The fixture `vertex_fetch_reach` writes `(v, vid + iid)` at `out[(vid + iid) & 3]` (each id with
//! its base, as Metal counts them), and the attribute buffer, a layout stepped per instance of
//! stride 4, holds `0x10 + w` at word `w`, so each slot names the element its instance fetched.
//! Every draw is one index, 0, at base vertex 0, so `vid` is 0 and the four instances of a draw
//! write four distinct slots of its own segment of `out`, which is made holding `UNWRITTEN` and
//! read whole.
//!
//! | rate | base instance | instances | the elements of a count from the base | from instance 0 |
//! |---|---|---|---|---|
//! | 2 | 0 | 0 to 3 | 0, 0, 1, 1 | the same |
//! | 2 | 5 | 5 to 8 | 5, 5, 6, 6 | 2, 3, 3, 4 |
//! | 3 | 0 | 0 to 3 | 0, 0, 0, 1 | the same |
//! | 3 | 5 | 5 to 8 | 5, 5, 5, 6 | 1, 2, 2, 2 |
//!
//! A count from the base instance is Vulkan's divisor
//! (`firstInstance + (instanceIndex - firstInstance) / divisor`); one from instance 0 divides the
//! instance index whole.
//!
//! # Safety
//!
//! Rasterization is disabled and each draw is four instances of one point of a loop-free vertex
//! function, so nothing is rasterized and nothing loops. Every write lies inside its draw's segment
//! of `out` (`every_write_lies_inside_out`), and every element either count names lies inside the
//! attribute buffer (`every_element_lies_inside_the_attributes`).

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

/// The layout's stride: one word an element.
const STRIDE: u32 = 4;

/// The step rates, one pipeline each.
const RATES: [u32; 2] = [2, 3];

/// The base instances each pipeline draws from.
const BASES: [u32; 2] = [0, 5];

/// Instances each draw names.
const INSTANCES: u32 = 4;

/// The slots of `out` one draw's sums can reach: the fixture writes `(vid + iid) & 3`.
const SLOTS: u64 = 4;

/// Bytes one draw's segment of `out` holds: a `uint2` a slot.
const SEGMENT_BYTES: u64 = SLOTS * 8;

/// The one index every draw reads, 32-bit.
const INDICES: [u32; 1] = [0];

/// One `UInt` attribute at offset 0 of buffer [`ATTRIBUTES_INDEX`].
const ATTRIBUTES: [VertexAttribute; 1] = [VertexAttribute {
    index: 0,
    format: VertexFormat::UInt,
    offset: 0,
    buffer: ATTRIBUTES_INDEX,
}];

/// Buffer [`ATTRIBUTES_INDEX`] stepped per instance at `rate`, of stride [`STRIDE`].
const fn layout(rate: u32) -> [VertexBufferLayout; 1] {
    [VertexBufferLayout {
        index: ATTRIBUTES_INDEX,
        stride: VertexStride::Bytes(STRIDE),
        step: VertexStepFunction::PerInstance,
        rate,
    }]
}

/// The layouts, in [`RATES`]' order.
const LAYOUTS: [[VertexBufferLayout; 1]; 2] = [layout(RATES[0]), layout(RATES[1])];

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

/// The draws, in the module's table's order: each rate, from each base instance.
fn draws() -> Vec<(u32, u32)> {
    RATES
        .iter()
        .flat_map(|&rate| BASES.iter().map(move |&base| (rate, base)))
        .collect()
}

/// Bytes `out` holds: one segment a draw.
fn out_bytes() -> u64 {
    SEGMENT_BYTES * draws().len() as u64
}

/// The pass: no attachment at a 1 by 1 extent, the attributes bound once, and per draw its rate's
/// pipeline, `out` bound at its segment, and one index of [`INSTANCES`] instances from its base
/// instance.
fn pass(pipelines: &[Handle], out: Handle, attributes: Handle, indices: Handle) -> RenderPass {
    let mut steps = vec![RenderStep::VertexBuffer {
        buffer: attributes,
        offset: 0,
        index: ATTRIBUTES_INDEX,
    }];
    for (at, (rate, base_instance)) in draws().into_iter().enumerate() {
        let pipeline = RATES.iter().position(|&r| r == rate).expect("a rate");
        steps.push(RenderStep::Pipeline(pipelines[pipeline]));
        steps.push(RenderStep::VertexBuffer {
            buffer: out,
            offset: at as u64 * SEGMENT_BYTES,
            index: 0,
        });
        steps.push(RenderStep::DrawIndexed {
            primitive: PrimitiveType::Point,
            count: 1,
            index_type: IndexType::UInt32,
            indices,
            offset: 0,
            instances: INSTANCES,
            base_vertex: 0,
            base_instance,
        });
    }
    RenderPass::sized([1, 1]).drawing(steps)
}

/// The case: the draws of the module's table.
pub struct VertexStepRate;

impl Case for VertexStepRate {
    fn name(&self) -> &'static str {
        "vertex-step-rate-instances"
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
        for (rate, spec) in RATES.iter().zip(pipelines()) {
            inputs.push(Input::bytes(format!("pipeline:rate-{rate}"), &spec.words()));
        }
        inputs.push(Input::bytes("buffer:attributes", &attribute_bytes()));
        inputs.push(Input::bytes("buffer:indices", &index_bytes()));
        inputs.push(Input::bytes(
            "buffer:out",
            &vec![UNWRITTEN; out_bytes() as usize],
        ));
        inputs.push(Input::bytes(
            "pass",
            &pass(&[Handle(0), Handle(1)], Handle(4), Handle(2), Handle(3)).words(),
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
        for (rate, spec) in RATES.iter().zip(self::pipelines()) {
            let at = format!("pipeline:rate-{rate}");
            pipelines.push(made(sites, &at, side.render_pipeline(library, &spec))?);
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

/// The family's case.
pub fn all() -> Vec<Box<dyn Case>> {
    vec![Box::new(VertexStepRate)]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcript::Value;

    /// Every draw writes its own segment and nothing past `out`: the fixture writes slots 0 to 3
    /// of the segment it is bound at, and the last segment ends at `out`'s end.
    #[test]
    fn every_write_lies_inside_out() {
        let draws = draws().len() as u64;
        assert_eq!((draws - 1) * SEGMENT_BYTES + 8 * SLOTS, out_bytes());
    }

    /// The elements a count from the base instance and a count from instance 0 name lie inside
    /// the attribute buffer, and the two counts differ at every draw from base instance 5.
    #[test]
    fn every_element_lies_inside_the_attributes() {
        for (rate, base) in draws() {
            let from_base: Vec<u32> = (base..base + INSTANCES)
                .map(|i| base + (i - base) / rate)
                .collect();
            let whole: Vec<u32> = (base..base + INSTANCES).map(|i| i / rate).collect();
            for element in from_base.iter().chain(&whole) {
                assert!(element * STRIDE + 4 <= WORDS * 4, "element {element}");
            }
            assert_eq!(from_base == whole, base == 0, "rate {rate} base {base}");
        }
    }

    /// Metal's answer: every instance fetched element `base + (iid - base) / rate`, its count
    /// starting at the draw's base instance, in every draw.
    #[test]
    fn metal_counted_the_step_from_the_base_instance() {
        let recording = crate::recording::read(VertexStepRate.name()).expect("the recording");
        let answer = recording.answer().expect("an answer of the oracle");
        let transcript = answer.deterministic().expect("one outcome");
        let Some(Value::Bytes(words)) = transcript.root.child("out").map(|n| &n.observed.value)
        else {
            panic!("the recording holds no read-back of out");
        };
        let word = |at: usize| u32::from_le_bytes(words[4 * at..4 * at + 4].try_into().unwrap());
        for (draw, (rate, base)) in draws().into_iter().enumerate() {
            let mut want = [[0u32; 2]; SLOTS as usize];
            for i in base..base + INSTANCES {
                want[(i & 3) as usize] = [0x10 + base + (i - base) / rate, i];
            }
            let at = draw * SLOTS as usize * 2;
            let got: Vec<[u32; 2]> = (0..SLOTS as usize)
                .map(|slot| [word(at + 2 * slot), word(at + 2 * slot + 1)])
                .collect();
            assert_eq!(got, want, "rate {rate} from base instance {base}");
        }
    }
}
