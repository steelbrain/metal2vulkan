//! The render encoder's `setVertexBuffer:offset:attributeStride:atIndex:` and
//! `setVertexBytes:length:attributeStride:atIndex:` at the strides whose displacement leaves 32
//! bits.
//!
//! A vertex descriptor's buffer layout of `MTLBufferLayoutStrideDynamic` reads its buffer at the
//! stride the encoder binds it with. Metal places vertex `v`'s element at `v · stride` from the
//! bind offset in 64 bits (`render/vertex_reach.rs` in the runtime has the measurement in a buffer
//! past 4 GiB). These cases ask what that means over a **small** binding, where the element of
//! every vertex past the first lies past the binding for each of the large strides below, and
//! where a fetch that computed the displacement in 32 bits instead would wrap and could land back
//! inside it. `attribute-stride-vertex-buffer` binds a buffer, and `attribute-stride-vertex-bytes`
//! the same bytes inline, copied by each call.
//!
//! # The strides
//!
//! One draw of `VERTICES` points a stride, each writing its own segment of `out`:
//!
//! | stride | 64-bit displacement of vertices 1, 2, 3 | the same in 32 bits |
//! |---|---|---|
//! | `4` | 4, 8, 12 (the control: every element inside) | the same |
//! | `MTLAttributeStrideStatic` (`u64::MAX`) | past the buffer, if Metal takes the low 32 bits as the compute encoder does | past the buffer |
//! | `2^32 - 8` | past the buffer | `2^32 - 8`, `2^32 - 16`, `2^32 - 24`: past it too |
//! | `2^31 + 4` | past the buffer | `2^31 + 4`, **8**, `2^31 + 12`: vertex 2 lands on element 2 |
//!
//! The constant's row is the open question, not a prediction: the compute encoder holds it as
//! `2^32 - 1` (`attribute_strides`), and a render encoder's table holds no stride for it. The last
//! row is the one a 32-bit displacement reads wrongly, at vertex 2 alone.
//!
//! # How a case is read
//!
//! The fixture's `icb_render_stride_template` writes `(v, vid)` at `out[vid]`, and the attribute
//! buffer holds `0x10 + e` at element `e`, so each segment's words name the elements its stride
//! fetched, whether a buffer or inline bytes hold them. `out` is made holding `UNWRITTEN` and read
//! whole.
//!
//! # Safety
//!
//! Rasterization is disabled and each draw is four points of a loop-free vertex function, so
//! nothing is rasterized and nothing loops. The fetches past the buffer are the measurement; every
//! other byte any draw reads lies inside the attribute binding, and every write lies inside `out`
//! (`every_write_lies_inside_out`).

use super::indirect_commands::{
    first_refused_piece, fixture_inputs, program_node, shaders_of, UNWRITTEN,
};
use crate::allowance::Allowance;
use crate::case::{call, made, Abort, Case, Failure, Handle, Input, MetalSource, Side};
use crate::compute_pass::ATTRIBUTE_STRIDE_STATIC;
use crate::fixture::Fixture;
use crate::recording::ShaderRef;
use crate::render::{
    RenderPass, RenderPipelineSpec, RenderStep, RenderWork, StageFunction, VertexAttribute,
    VertexBufferLayout, VertexLayout,
};
use crate::transcript::{ByteLayout, Element, Node, Observed, Sites};
use metal2vulkan::metal::{PrimitiveType, VertexFormat};

/// The owned fixture: a rasterization-disabled vertex function that fetches one `uint`
/// `[[stage_in]]` attribute and writes it with its vertex id.
const FIXTURE: &str = "icb_render_stride_template";

/// The vertex buffer index the attribute's layout reads; 0 is the function's own `out`.
const ATTRIBUTES_INDEX: u32 = 1;

/// The words the attribute buffer holds.
const ELEMENTS: u32 = 16;

/// Vertices each draw fetches.
const VERTICES: u32 = 4;

/// Bytes one draw's segment of `out` holds: a `uint2` a vertex.
const SEGMENT_BYTES: u64 = VERTICES as u64 * 8;

/// The strides the draws bind, one draw each (the module's table).
const STRIDES: [u64; 4] = [4, ATTRIBUTE_STRIDE_STATIC, (1 << 32) - 8, (1 << 31) + 4];

/// Bytes `out` holds: one segment a draw.
const OUT_BYTES: u64 = SEGMENT_BYTES * STRIDES.len() as u64;

/// One `UInt` attribute at offset 0 of buffer [`ATTRIBUTES_INDEX`], under a layout of
/// `MTLBufferLayoutStrideDynamic`.
const ATTRIBUTES: [VertexAttribute; 1] = [VertexAttribute {
    index: 0,
    format: VertexFormat::UInt,
    offset: 0,
    buffer: ATTRIBUTES_INDEX,
}];
const LAYOUTS: [VertexBufferLayout; 1] = [VertexBufferLayout::dynamic_stride(ATTRIBUTES_INDEX)];

/// The pipeline: the fixture's vertex function, rasterization disabled, under the dynamic layout.
fn pipeline_spec() -> RenderPipelineSpec<'static> {
    let mut spec = RenderPipelineSpec::of(StageFunction::named(FIXTURE), None, &[]);
    spec.rasterization = false;
    spec.vertex_layout = Some(VertexLayout::new(&ATTRIBUTES, &LAYOUTS));
    spec
}

/// The attribute buffer's bytes: `0x10 + e` at element `e`.
fn attribute_bytes() -> Vec<u8> {
    (0..ELEMENTS)
        .flat_map(|e| (0x10 + e).to_le_bytes())
        .collect()
}

/// Where a case's draws read the attributes.
#[derive(Clone, Copy)]
enum Binding {
    /// A buffer of [`attribute_bytes`] the side made, bound by
    /// `setVertexBuffer:offset:attributeStride:atIndex:`.
    Buffer(Handle),
    /// [`attribute_bytes`] copied inline by `setVertexBytes:length:attributeStride:atIndex:`.
    Bytes,
}

impl Binding {
    /// The step that binds the attributes at `stride`.
    fn step(self, stride: u64) -> RenderStep {
        match self {
            Binding::Buffer(buffer) => RenderStep::VertexBufferWithStride {
                buffer,
                offset: 0,
                stride,
                index: ATTRIBUTES_INDEX,
            },
            Binding::Bytes => RenderStep::VertexBytesWithStride {
                bytes: attribute_bytes(),
                stride,
                index: ATTRIBUTES_INDEX,
            },
        }
    }
}

/// The pass: no attachment at a 1 by 1 extent, and per stride `out` bound at its segment, the
/// attributes at the stride, and a draw of [`VERTICES`] points.
fn pass(pipeline: Handle, out: Handle, attributes: Binding) -> RenderPass {
    let mut steps = vec![RenderStep::Pipeline(pipeline)];
    for (at, stride) in STRIDES.into_iter().enumerate() {
        steps.push(RenderStep::VertexBuffer {
            buffer: out,
            offset: at as u64 * SEGMENT_BYTES,
            index: 0,
        });
        steps.push(attributes.step(stride));
        steps.push(RenderStep::Draw {
            primitive: PrimitiveType::Point,
            first: 0,
            count: VERTICES,
            instances: 1,
        });
    }
    RenderPass::sized([1, 1]).drawing(steps)
}

/// A case: the attributes in a buffer, or inline when `inline`.
pub struct VertexAttributeStrides {
    inline: bool,
}

impl Case for VertexAttributeStrides {
    fn name(&self) -> &'static str {
        match self.inline {
            false => "attribute-stride-vertex-buffer",
            true => "attribute-stride-vertex-bytes",
        }
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
        inputs.push(Input::bytes("render-pipeline", &pipeline_spec().words()));
        let attributes = match self.inline {
            false => {
                inputs.push(Input::bytes("buffer:attributes", &attribute_bytes()));
                Binding::Buffer(Handle(2))
            }
            true => Binding::Bytes,
        };
        inputs.push(Input::bytes(
            "buffer:out",
            &vec![UNWRITTEN; OUT_BYTES as usize],
        ));
        inputs.push(Input::bytes(
            "pass",
            &pass(Handle(0), Handle(1), attributes).words(),
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
            side.render_pipeline(library, &pipeline_spec()),
        )?;
        let attributes = match self.inline {
            false => Binding::Buffer(made(
                sites,
                "buffer:attributes",
                side.buffer(&attribute_bytes()),
            )?),
            true => Binding::Bytes,
        };
        let out = made(
            sites,
            "buffer:out",
            side.buffer(&vec![UNWRITTEN; OUT_BYTES as usize]),
        )?;
        let work = vec![RenderWork::Pass(pass(pipeline, out, attributes))];
        let outcome = call(sites, "render", side.render(&work), |outcome| {
            outcome.status.clone()
        })?;
        sites.push(program_node(&work, &outcome));
        if let Some((at, failure)) = first_refused_piece(&work, &outcome) {
            return Err(Abort { at, failure });
        }
        let read = side
            .read_buffer(out, 0, OUT_BYTES as usize)
            .map_err(|failure| Abort {
                at: "out".to_owned(),
                failure,
            })?;
        sites.push(Node::leaf(
            "out",
            Observed::bytes(read, ByteLayout::buffer(0, OUT_BYTES, Element::U32)),
        ));
        Ok(())
    }
}

/// The family's cases: the attributes in a buffer, then inline.
pub fn all() -> Vec<Box<dyn Case>> {
    vec![
        Box::new(VertexAttributeStrides { inline: false }),
        Box::new(VertexAttributeStrides { inline: true }),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every draw writes its own segment and nothing past `out`: vertex `v` of draw `at` writes
    /// the eight bytes at `at · SEGMENT_BYTES + 8 v`.
    #[test]
    fn every_write_lies_inside_out() {
        for at in 0..STRIDES.len() as u64 {
            let last = at * SEGMENT_BYTES + 8 * u64::from(VERTICES - 1) + 8;
            assert!(last <= OUT_BYTES, "draw {at} ends at {last}");
        }
    }

    /// The control stride reads inside the buffer, and the last stride's 32-bit displacement of
    /// vertex 2 is element 2 while its 64-bit one is past the buffer: the row the module's table
    /// says a 32-bit fetch reads wrongly.
    #[test]
    fn the_strides_reach_where_the_table_says() {
        let bytes = u64::from(ELEMENTS) * 4;
        assert!(u64::from(VERTICES - 1) * STRIDES[0] + 4 <= bytes);
        let wide = STRIDES[3];
        assert!(2 * wide > bytes);
        assert_eq!((2 * wide) % (1 << 32), 8);
        for stride in &STRIDES[1..] {
            assert!(*stride > bytes, "{stride}");
        }
    }
}
