//! `MTLPrimitiveType` crossed with `MTLIndexType`: whether the index value `0xffff` ends a
//! primitive or names a vertex, which neither the primitive type nor the index type decides alone.
//!
//! `render_draw_forms` put every one of the five primitive types and both index types to Metal, and
//! `topology_classes` measured the pipeline's `inputPrimitiveTopology` class beside the type the
//! draw states. None of them crosses the primitive type with anything: each of its cases states one
//! index type a draw and asks what the draw covered, and the covered set of a list or a strip over
//! ordinary indices is the primitive type's answer whatever width the indices were written at.
//!
//! One thing is not. **Metal restarts a strip at the all-ones index of its index type**, and draws
//! that index as a vertex for a list or for points — which the runtime models with Vulkan's
//! `primitiveRestartEnable`, set for strips only, on a reading taken from Metal probes and from no
//! case of this suite. That reading is a function of both axes at once:
//!
//! - the **value** that ends a primitive is the index type's own maximum, `0xffff` for `UInt16` and
//!   `0xffffffff` for `UInt32`, so `0xffff` is a sentinel under one index type and the plain vertex
//!   65535 under the other;
//! - and **only a strip** has a primitive for it to end: a point, a line list and a triangle list
//!   draw the same index as a vertex.
//!
//! So the one index value `0xffff`, in one index run, over the five primitive types and the two
//! index types, is ten cells in which the answer is the pair's. The two strips under `UInt16` are
//! the only cells where the value ends a primitive, and the eight others are the control that says
//! so.
//!
//! # What this case does not ask, and why
//!
//! The all-ones index of `UInt32` is **not** crossed with a list here. A side that did not restart
//! would fetch vertex `0xffffffff` of the positions buffer, which no buffer can hold, so the cell
//! cannot be authored without a draw whose own answer is an out-of-range read. The pair is asked at
//! `0xffff` instead, which is in range under both index types by construction, and that is the cell
//! the cross needs: it is the index type, and nothing else, that decides whether the value is a
//! sentinel at all.
//!
//! # The observable
//!
//! The owned fixture `icb_merged_draws`, whose fragment answer **is** the draw's own parameters:
//! `icb_merged_vertex` places vertex `vid` at `xy[vid]` and passes `uint4(mark, vid, iid, base)`
//! flat to `icb_merged_fragment`, which returns it into an `RGBA32Uint` target. A texel is that
//! `uint4` stored as it is, so a restart is a **pixel that is still the clear** and a vertex read is
//! a pixel carrying the index that covered it. There is no rounding rule and no tolerance.
//!
//! The positions buffer holds 65536 vertices so that index `0xffff` names a position the case
//! states, which is what makes the `UInt32` leg of every cell a draw of a real vertex rather than a
//! read past the buffer (`tests::every_index_names_a_position_the_case_states`).
//!
//! # What the recorded answers settle
//!
//! Metal's own, from the committed recording on an Apple M3 Max under macOS 27.0.1. Pixels covered,
//! and the vertex ids among them, for the one run `[0, 1, 0xffff, 2, 3, 4]`:
//!
//! | primitive type | `UInt16` | `UInt32` |
//! |---|---|---|
//! | `Point` | 6 pixels, ids 0 1 2 3 4 **65535** | the same 6 pixels |
//! | `Line` | 11 pixels, ids including **65535** | the same 11 pixels |
//! | `LineStrip` | 14 pixels, **no id 65535** | 21 pixels, ids including 65535 |
//! | `Triangle` | 18 pixels | the same 18 pixels |
//! | `TriangleStrip` | 6 pixels, one triangle | 25 pixels, four triangles |
//!
//! So **Metal restarts a strip at the all-ones index of its index type and draws that index as a
//! vertex for points and for lists**, measured rather than probed: `0xffff` ends a primitive in
//! exactly the two cells where the primitive type is a strip **and** the index type is `UInt16`,
//! and in no other cell of the ten. The two strips' `UInt16` answers carry no fragment of vertex
//! 65535 at all, which is what says the index was never fetched, and their `UInt32` answers carry
//! it, which is what says the value itself is no sentinel. The runtime's reading — Vulkan's
//! `primitiveRestartEnable`, set for strips only — is Metal's, and
//! `tests::the_index_type_decides_only_for_a_strip` holds exactly that shape against the
//! recording.
//!
//! # Nothing here can fault a GPU
//!
//! One pass a cell, one draw a pass, six indices a draw, over an 8 by 8 target, no loop in the
//! fixture. Every index any draw reads is 0 to 4 or 65535, each of which names one of the five
//! positions the case states inside a buffer of 65536, and every one of those positions lies inside
//! the target.

use crate::allowance::Allowance;
use crate::case::{call, made, Abort, Case, Handle, Input, MetalSource, Side};
use crate::recording::ShaderRef;
use crate::render::{ColorState, PassColor, RenderPass, RenderStep, RenderWork};
use crate::transcript::{Node, Observed, Sites, Value};
use metal2vulkan::metal::{IndexType, PrimitiveType};

use super::render_draw_forms::{
    fixture_inputs, layout, library, pipeline, plane, positions_buffer, shader_refs, target_spec,
    FORMAT, TEXEL,
};

/// The target every cell draws into: 8 by 8, which holds the five positions the case states with
/// room between them.
const SIZE: [u32; 2] = [8, 8];

/// Vertices the positions buffer holds, which is one more than the largest index any draw reads:
/// the `UInt16` all-ones index **is** a vertex for a list and for points, so the buffer has to hold
/// it or that cell would be a read past the buffer rather than a question about the index type.
const VERTICES: u32 = 0x1_0000;

/// The index the cross is asked at: `UInt16`'s all-ones value, and the plain vertex 65535 under
/// `UInt32`.
const SENTINEL: u32 = 0xffff;

/// The indices every cell's draw reads, in order, with [`SENTINEL`] in the middle so that a strip
/// that ends there has a primitive on each side of it.
///
/// Six indices, and no three of the positions they name are collinear, so every triangle any
/// primitive type assembles from them covers pixels: a cell whose answer is the clear is a restart
/// and not a degenerate triangle.
const RUN: [u32; 6] = [0, 1, SENTINEL, 2, 3, 4];

/// Where the vertices the run names lie, as pixel centres of [`SIZE`].
///
/// Index 65535 is one of them, which is what the whole buffer is sized for; every other vertex of
/// the buffer is the first position, and no draw reads one.
const NAMED: [(u32, [f32; 2]); 6] = [
    (0, [0.5, 0.5]),
    (1, [6.5, 1.5]),
    (2, [1.5, 6.5]),
    (3, [6.5, 5.5]),
    (4, [3.5, 3.5]),
    (SENTINEL, [0.5, 4.5]),
];

/// How far apart the cells' index runs lie: 32 bytes holds six `UInt32` indices and is a multiple
/// of both index sizes, so one buffer serves both widths at one stride.
const RUN_STRIDE: u64 = 32;

/// Bytes a cell's own `constant` bindings lie apart, which is the granularity every render case of
/// the suite binds a `constant` buffer at.
const BIND_STRIDE: u64 = 16;

/// Every `MTLPrimitiveType`, named one by one in the SDK table's own declaration order, and every
/// `MTLIndexType` beside them: a cross's text has to state the values it varies, and
/// `tests::the_cells_are_every_primitive_type_under_every_index_type` holds both lists against the
/// generated ones.
const PRIMITIVES: [PrimitiveType; 5] = [
    PrimitiveType::Point,
    PrimitiveType::Line,
    PrimitiveType::LineStrip,
    PrimitiveType::Triangle,
    PrimitiveType::TriangleStrip,
];

/// Both index types, in the SDK table's own order.
const INDEX_TYPES: [IndexType; 2] = [IndexType::UInt16, IndexType::UInt32];

/// Every cell of the cross, in the order the passes are encoded: each primitive type under each
/// index type, all at the one index run.
fn cells() -> Vec<(PrimitiveType, IndexType)> {
    let mut cells = Vec::new();
    for primitive in PRIMITIVES {
        for index_type in INDEX_TYPES {
            cells.push((primitive, index_type));
        }
    }
    cells
}

/// The positions buffer: [`VERTICES`] vertices, the six [`NAMED`] ones where the case states and
/// every other at the first position, which no draw reads.
fn positions() -> Vec<[f32; 2]> {
    let mut at = vec![NAMED[0].1; VERTICES as usize];
    for (index, position) in NAMED {
        at[index as usize] = position;
    }
    at
}

/// The marks buffer: one word a cell at [`BIND_STRIDE`], cell `c`'s mark `c + 1`, so the clear's 0
/// is a pixel no draw wrote.
fn marks() -> Vec<u8> {
    let all = cells();
    let mut bytes = vec![0u8; all.len() * BIND_STRIDE as usize];
    for at in 0..all.len() {
        let word = at * BIND_STRIDE as usize;
        bytes[word..word + 4].copy_from_slice(&(at as u32 + 1).to_le_bytes());
    }
    bytes
}

/// The frames buffer: one `float4` a cell, `(width, height, 0, 0)`. Every cell draws one instance,
/// so no cell shifts a row and the third channel is 0.
fn frames() -> Vec<u8> {
    let mut bytes = Vec::new();
    for _ in cells() {
        for channel in [SIZE[0] as f32, SIZE[1] as f32, 0.0, 0.0] {
            bytes.extend(channel.to_le_bytes());
        }
    }
    bytes
}

/// The index buffer: each cell's own block at its own [`RUN_STRIDE`], the one [`RUN`] written at
/// the width the cell's index type names.
///
/// The two widths hold the **same six numbers**, which is the point: the cell's answer parts on
/// what the index type makes of `0xffff` and not on which indices the case chose.
fn indices() -> Vec<u8> {
    let all = cells();
    let mut bytes = vec![0u8; all.len() * RUN_STRIDE as usize];
    for (at, (_, index_type)) in all.iter().enumerate() {
        let block = at * RUN_STRIDE as usize;
        for (which, index) in RUN.iter().enumerate() {
            match index_type {
                IndexType::UInt32 => {
                    let byte = block + 4 * which;
                    bytes[byte..byte + 4].copy_from_slice(&index.to_le_bytes());
                }
                _ => {
                    let byte = block + 2 * which;
                    bytes[byte..byte + 2].copy_from_slice(&(*index as u16).to_le_bytes());
                }
            }
        }
    }
    bytes
}

/// The site a cell's answer hangs at: the primitive type's enumerator, holding the index type's,
/// so the transcript names both values of the pair by name beside the answer.
fn site(primitive: PrimitiveType, index_type: IndexType) -> (&'static str, &'static str) {
    (primitive.name(), index_type.name())
}

/// `MTLPrimitiveType` crossed with `MTLIndexType` at the index `UInt16` ends a primitive on.
pub struct RestartIndices;

impl RestartIndices {
    const NAME: &'static str = "primitive-type-restart-indices";

    /// The steps one cell's pass holds: the pipeline, the positions, the cell's mark and frame, and
    /// its indexed draw of the one run.
    fn steps(at: usize, cell: (PrimitiveType, IndexType), held: &Bound) -> Vec<RenderStep> {
        vec![
            RenderStep::Pipeline(held.pipeline),
            RenderStep::VertexBuffer {
                buffer: held.positions,
                offset: 0,
                index: 2,
            },
            RenderStep::VertexBuffer {
                buffer: held.marks,
                offset: at as u64 * BIND_STRIDE,
                index: 0,
            },
            RenderStep::VertexBuffer {
                buffer: held.frames,
                offset: at as u64 * BIND_STRIDE,
                index: 1,
            },
            RenderStep::DrawIndexedPlain {
                primitive: cell.0,
                count: RUN.len() as u32,
                index_type: cell.1,
                indices: held.indices,
                offset: at as u64 * RUN_STRIDE,
            },
        ]
    }
}

/// What one cell's pass binds.
struct Bound {
    pipeline: Handle,
    positions: Handle,
    marks: Handle,
    frames: Handle,
    indices: Handle,
}

impl Case for RestartIndices {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // Nothing is loosened and nothing is declared: every draw here is the direct indexed form,
        // which reaches no bounds pass and announces nothing.
        Vec::new()
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        shader_refs()
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let mut inputs = fixture_inputs()?;
        let colors = [Some(ColorState::new(FORMAT))];
        inputs.push(Input::bytes("render-pipeline", &pipeline(&colors).words()));
        inputs.push(Input::bytes("target", &target_spec(SIZE).words()));
        inputs.push(Input::bytes(
            "buffer:positions",
            &positions_buffer(&positions()),
        ));
        inputs.push(Input::bytes("buffer:marks", &marks()));
        inputs.push(Input::bytes("buffer:frames", &frames()));
        inputs.push(Input::bytes("buffer:indices", &indices()));
        // The cells' own table, so a cell whose primitive type or index type moved moves the key.
        let mut table = Vec::new();
        for (primitive, index_type) in cells() {
            table.extend(primitive.as_raw().to_le_bytes());
            table.extend(index_type.as_raw().to_le_bytes());
        }
        inputs.push(Input::bytes("cells", &table));
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let library = library(side, sites)?;
        let colors = [Some(ColorState::new(FORMAT))];
        let held = Bound {
            pipeline: made(
                sites,
                "render-pipeline",
                side.render_pipeline(library, &pipeline(&colors)),
            )?,
            positions: made(
                sites,
                "buffer:positions",
                side.buffer(&positions_buffer(&positions())),
            )?,
            marks: made(sites, "buffer:marks", side.buffer(&marks()))?,
            frames: made(sites, "buffer:frames", side.buffer(&frames()))?,
            indices: made(sites, "buffer:indices", side.buffer(&indices()))?,
        };
        let target = made(sites, "target", side.target(&target_spec(SIZE)))?;
        let all = cells();
        let read_back = made(
            sites,
            "buffer:read-back",
            side.zeroed_buffer(all.len() * plane(SIZE) as usize),
        )?;

        // One pass a cell, each clearing the one target and each read back into its own plane of
        // the read-back buffer: a cell's answer is the whole target, and the clear between them is
        // what makes a restart readable as a pixel nobody wrote.
        let mut work = Vec::new();
        for (at, cell) in all.iter().enumerate() {
            work.push(RenderWork::Pass(
                RenderPass::clearing(vec![PassColor::cleared(target, [0.0; 4])])
                    .drawing(Self::steps(at, *cell, &held)),
            ));
            work.push(RenderWork::ReadTexture {
                texture: target,
                into: read_back,
                offset: at as u64 * u64::from(plane(SIZE)),
                row_pitch: TEXEL * SIZE[0],
                size: SIZE,
            });
        }
        let outcome = call(sites, "render", side.render(&work), |outcome| {
            outcome.status.clone()
        })?;

        let mut pieces = Node::branch("pieces", Vec::new());
        for (at, (primitive, index_type)) in all.iter().enumerate() {
            let (outer, inner) = site(*primitive, *index_type);
            for (which, what) in ["pass", "read"].iter().enumerate() {
                let refused = outcome.refused.get(2 * at + which).and_then(Option::as_ref);
                pieces = pieces.with(Node::leaf(
                    format!("{outer}/{inner}/{what}"),
                    Observed::value(match refused {
                        Some(failure) => failure.recorded(),
                        None => Value::ok(Value::Absent),
                    }),
                ));
            }
        }
        sites.push(pieces);
        if let Some(failure) = outcome.refused.iter().flatten().next() {
            return Err(Abort {
                at: "pieces".to_owned(),
                failure: failure.clone(),
            });
        }

        let bytes = call(
            sites,
            "read-back",
            side.read_buffer(read_back, 0, all.len() * plane(SIZE) as usize),
            |_| Value::Absent,
        )?;
        let mut groups: Vec<(&'static str, Vec<Node>)> = Vec::new();
        for (at, (primitive, index_type)) in all.iter().enumerate() {
            let (outer, inner) = site(*primitive, *index_type);
            let first = at * plane(SIZE) as usize;
            let held = bytes[first..first + plane(SIZE) as usize].to_vec();
            let node = Node::branch(inner, Vec::new())
                .with(Node::leaf("texels", Observed::bytes(held, layout(SIZE))));
            match groups.last_mut() {
                Some((name, nodes)) if *name == outer => nodes.push(node),
                _ => groups.push((outer, vec![node])),
            }
        }
        for (outer, nodes) in groups {
            sites.push(Node::branch(outer, nodes));
        }
        Ok(())
    }
}

/// Every case of the family.
pub fn boxed() -> Vec<Box<dyn Case>> {
    vec![Box::new(RestartIndices)]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every index any draw of the case reads names a position the case states, inside a buffer
    /// that holds it. An index past the buffer would be a read of a `device const` pointer out of
    /// range, which is not a question this suite may ask.
    #[test]
    fn every_index_names_a_position_the_case_states() {
        for index in RUN {
            assert!(
                NAMED.iter().any(|(named, _)| *named == index),
                "index {index} is one the case states"
            );
            assert!(index < VERTICES, "index {index} is inside the buffer");
        }
    }

    /// Every stated position lies inside the target, so no draw's pixels fall outside the plane the
    /// case compares.
    #[test]
    fn every_position_lies_inside_the_target() {
        for (index, [x, y]) in NAMED {
            assert!(
                x > 0.0 && x < SIZE[0] as f32 && y > 0.0 && y < SIZE[1] as f32,
                "vertex {index} at ({x}, {y}) is inside the target"
            );
        }
    }

    /// No three stated positions are collinear, so every triangle any primitive type assembles
    /// from the run covers at least one pixel and an empty answer is a restart.
    #[test]
    fn no_three_positions_are_collinear() {
        let all: Vec<[f32; 2]> = NAMED.iter().map(|(_, at)| *at).collect();
        for a in 0..all.len() {
            for b in a + 1..all.len() {
                for c in b + 1..all.len() {
                    let (p, q, r) = (all[a], all[b], all[c]);
                    let area = (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]);
                    assert!(area.abs() > 0.5, "{p:?}, {q:?} and {r:?} are not collinear");
                }
            }
        }
    }

    /// The texels the recording of the cell `(primitive, index_type)` holds.
    fn recorded(primitive: PrimitiveType, index_type: IndexType) -> Vec<u8> {
        let recording = crate::recording::read(RestartIndices::NAME).expect("the recording");
        let (outer, inner) = site(primitive, index_type);
        for answer in &recording.answers {
            for outcome in &answer.outcomes {
                let mut node = &outcome.transcript.root;
                for name in [outer, inner, "texels"] {
                    node = node.child(name).expect("the cell's leaf");
                }
                if let Value::Bytes(bytes) = &node.observed.value {
                    return bytes.clone();
                }
            }
        }
        panic!("the recorded texels of {outer}/{inner}");
    }

    /// Bytes one texel of [`FORMAT`] takes, as a chunk length.
    const TEXEL_BYTES: usize = TEXEL as usize;

    /// Texels of `bytes` that are not the clear.
    fn covered(bytes: &[u8]) -> usize {
        let (texels, _) = bytes.as_chunks::<TEXEL_BYTES>();
        texels
            .iter()
            .filter(|texel| texel.iter().any(|byte| *byte != 0))
            .count()
    }

    /// What `bytes` says about each pixel, with the cell's own mark left out: `None` for a pixel
    /// that is still the clear, and the vertex, instance and base-instance channels of the
    /// fragment that covered it otherwise.
    ///
    /// The mark is the cell's number and differs between two cells by construction — it is what
    /// says the plane was written by its own draw — so two cells are compared on what they covered
    /// and on which vertex covered it, which is the whole of the answer here.
    fn shape(bytes: &[u8]) -> Vec<Option<[u32; 3]>> {
        let (texels, _) = bytes.as_chunks::<TEXEL_BYTES>();
        texels
            .iter()
            .map(|texel| {
                let word = |at: usize| {
                    u32::from_le_bytes(texel[4 * at..4 * at + 4].try_into().expect("a word"))
                };
                match word(0) {
                    0 => None,
                    _ => Some([word(1), word(2), word(3)]),
                }
            })
            .collect()
    }

    /// **Only a strip under `UInt16` reads `0xffff` as the end of a primitive.** The two strips
    /// answer different covered sets under the two index types, and the three others answer the
    /// same set under both: neither axis alone decides a cell, which is what makes this a crossing
    /// and not two sweeps.
    #[test]
    fn the_index_type_decides_only_for_a_strip() {
        for primitive in PRIMITIVES {
            let narrow = shape(&recorded(primitive, IndexType::UInt16));
            let wide = shape(&recorded(primitive, IndexType::UInt32));
            match primitive {
                PrimitiveType::LineStrip | PrimitiveType::TriangleStrip => assert_ne!(
                    narrow,
                    wide,
                    "{} under the two index types",
                    primitive.name()
                ),
                _ => assert_eq!(
                    narrow,
                    wide,
                    "{} under the two index types",
                    primitive.name()
                ),
            }
        }
    }

    /// The cells are every `MTLPrimitiveType` under every `MTLIndexType`, so a value the SDK grows
    /// is a failure here rather than a cell quietly missing from the cross.
    #[test]
    fn the_cells_are_every_primitive_type_under_every_index_type() {
        assert_eq!(PRIMITIVES.to_vec(), PrimitiveType::ALL.to_vec());
        assert_eq!(INDEX_TYPES.to_vec(), IndexType::ALL.to_vec());
    }

    /// Every cell covered a pixel, so no cell's answer is empty and the comparison of a strip
    /// against a list is a comparison of two sets and not of one set against nothing.
    #[test]
    fn every_cell_covered_a_pixel() {
        for (primitive, index_type) in cells() {
            let bytes = recorded(primitive, index_type);
            assert!(
                covered(&bytes) > 0,
                "{} under {} covered a pixel",
                primitive.name(),
                index_type.name()
            );
        }
    }
}
