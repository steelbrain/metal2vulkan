//! Every draw entry point of `MTLRenderCommandEncoder` that is not a patch or a mesh draw, over
//! **every value of `MTLPrimitiveType`** and both values of `MTLIndexType`.
//!
//! The suite reached two of Metal's five primitive types before this family -- `Point` and
//! `Triangle`, counted over every case module's sources -- and one of the two index types. A
//! primitive type is not a detail of a draw: it is what decides which pixels the draw's vertices
//! cover and which vertex each covered pixel takes its flat values from, so three of the five were
//! a family of Metal behaviour the suite said nothing about. The same held of the draw calls: of
//! the eight non-patch, non-mesh draw selectors, three were reached and five were not, and one of
//! those five was in the step vocabulary with no case encoding it.
//!
//! | case | what it draws, and over what |
//! |---|---|
//! | `render-draw-forms-points` | `MTLPrimitiveTypePoint` through all **eight** draw selectors, sweeping the vertex start, the instance count, the base vertex (0, positive and negative), the base instance and the index-buffer offset |
//! | `render-draw-forms-triangles` | `MTLPrimitiveTypeTriangle` and `MTLPrimitiveTypeTriangleStrip` over the same quad, through seven of the selectors |
//! | `render-draw-forms-lines` | `MTLPrimitiveTypeLine` and `MTLPrimitiveTypeLineStrip` over horizontal segments, through five of the selectors |
//! | `render-draw-forms-empty` | the documented **zero** of every count, through each selector that takes one, between two draws that write |
//!
//! # What makes each parameter visible
//!
//! The owned fixture `icb_merged_draws` is the one the suite already had whose fragment answer
//! **is** the draw's own parameters: `icb_merged_vertex` places vertex `vid` of instance `iid` at
//! `xy[vid]` moved down `iid * frame.z` pixel rows and passes `uint4(mark, vid, iid, base)` flat
//! to `icb_merged_fragment`, which returns it. The target is `RGBA32Uint`, so a texel **is** that
//! `uint4`: no rounding rule, no interpolation and no tolerance anywhere in the comparison.
//!
//! So each parameter under test reaches a channel or a coordinate of its own:
//!
//! - **`vertexStart`** and the **index values** reach `vid`, which is both the channel and, through
//!   `xy[vid]`, the pixel. A draw that read the wrong vertex writes in the wrong place **and**
//!   records the index it read.
//! - **`instanceCount`** reaches the row: each instance of a draw whose frame shifts by a row lands
//!   a row lower, so `n` instances write `n` rows and `n - 1` write one fewer.
//! - **`baseVertex`** reaches `vid` through the index, which is why this family states it at all
//!   three signs: 0 (the five selectors that name none), positive, and **negative**, each with
//!   indices chosen so a side that dropped it would write into another draw's band.
//! - **`baseInstance`** reaches the `base` channel outright.
//! - **`indexBufferOffset`** and the indirect `indexStart` reach which indices a draw reads, and
//!   each draw that states one carries a **spare index** its honoured form never reads: the pixel
//!   that index would have covered is in the expectation as clear, so dropping the offset is a
//!   pixel rather than an opinion.
//! - **`MTLPrimitiveType`** reaches the covered **set**: the same four corner positions are one
//!   point each under `Point`, two triangles filling a quad under `Triangle` and `TriangleStrip`,
//!   and segments under `Line` and `LineStrip`. `the_primitive_types_cover_different_pixels` holds
//!   that apart with no device asked.
//!
//! What a value **cannot** be distinguished from, said rather than left implied: the two triangle
//! types draw the identical pair of triangles here, by construction, so they differ in this family
//! only in the `vid` channel -- which vertex provokes each triangle. That is the whole of what
//! separates a list from a strip when the vertices are laid out to produce the same shape, and it
//! is a stronger reading than two different shapes would be, because a difference in the covered
//! set could be a difference in the geometry rather than in the assembly.
//!
//! # Metal's short selectors are its own defaults, and that is the claim
//!
//! Five of the eight selectors name fewer parameters than the longest of their family. The runtime
//! has **one** entry point per family rather than five, so our leg reaches `DrawPlain` and
//! `DrawInstanced` through `draw_primitives` with the defaults written out, and the two indexed
//! short forms through `draw_indexed_primitives` the same way. The comparison is therefore a
//! reading of **Metal's documented defaults**: a short selector that drew other than one instance
//! from base vertex 0 and base instance 0 differs here, in a channel or a pixel. It is not a claim
//! that our API has five selectors, and this family does not make one.
//!
//! # Nothing here can fault a GPU
//!
//! One pass a case, at most ten draws, at most 24 primitives a draw, over a 16 by 16 or 16 by 20
//! target, no loop in the fixture. Every index a draw can read names a vertex of the case's own
//! positions array, which `every_index_names_a_position_the_case_states` holds with no device
//! asked -- an index past it would be an out-of-bounds read of a `device const` pointer. Every
//! vertex position lies inside the target.

use crate::allowance::Allowance;
use crate::case::{call, made, Abort, Case, Failure, Handle, Input, MetalSource, Side};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::render::{
    ColorState, PassColor, Plane, ReadAt, RenderPass, RenderPipelineSpec, RenderStep, RenderWork,
    TargetSpec,
};
use crate::transcript::{
    Aspect, ByteLayout, Digest, Element, Gaps, Node, Observed, Region, Sites, Value,
};
use metal2vulkan::metal::{IndexType, PixelFormat, PrimitiveType};
use metal2vulkan_runtime::Degradation;

/// The target's format: four `uint`s a texel, which is the `uint4` the fragment function returns,
/// stored as it is. The whole expectation is integer arithmetic.
pub(super) const FORMAT: PixelFormat = PixelFormat::RGBA32Uint;

/// Bytes one texel of [`FORMAT`] takes.
pub(super) const TEXEL: u32 = 16;

/// Every target here is this many pixels wide, which is also how many vertices a row of the points
/// case's position grid holds.
const WIDTH: u32 = 16;

/// The owned fixture both stages come from.
const FIXTURE: &str = "icb_merged_draws";

/// The vertex stage: vertex `vid` of instance `iid` at `xy[vid]` moved down `iid * frame.z` rows.
const VERTEX: &str = "icb_merged_vertex";

/// The fragment stage: `uint4(mark, vid, iid, base)`.
const FRAGMENT: &str = "icb_merged_fragment";

/// The entry points a case's shader reference names.
const ENTRIES: &str = "icb_merged_vertex+icb_merged_fragment";

/// How far apart the per-draw bindings lie in the marks and frames buffers.
///
/// Sixteen rather than the four a `uint` takes: a `constant` binding's offset is bound at this
/// granularity by every render case of the suite, and a `float4` frame needs it anyway.
const BIND_STRIDE: u64 = 16;

/// How far apart the index runs and the indirect argument blocks lie in their buffers.
///
/// A multiple of 16 is a multiple of both index sizes and of 4, so one constant serves
/// `indexBufferOffset` for `UInt16` and `UInt32` alike and the indirect offsets as well.
const RUN_STRIDE: u64 = 16;

/// How many runs record mode takes a case of this module over.
///
/// Eight rather than the suite's three, for the reason `render_state` asks for eight: nothing here
/// is ordered by anything a device may reorder -- one pass, one queue, no visibility counter, no
/// sample or tile order, and within a draw Metal orders the primitives -- so the runs are expected
/// to fold into one outcome, and asking for more makes the recording say so on evidence.
const RUNS: u32 = 8;

/// The notice the two cases that make an **indexed indirect** draw declare, and the lines case
/// does not.
///
/// MoltenVK does not enable `robustBufferAccess2` and its driver is not measured to read index 0
/// past the bound range, so an indexed indirect draw's index fetch is nothing Vulkan bounds: the
/// GPU reads `indexCount` out of the argument buffer, and a count the index buffer cannot hold
/// would read past it. The runtime therefore bounds the indices on the GPU **before the pass**
/// (`metal2vulkan_runtime`'s `index_bounds.rs`) and announces the cost, which is work Metal does
/// not do. The declaration loosens no value -- it says the announcement is this case's -- so the
/// case still fails on any texel that moves.
///
/// Measured on an Apple M3 Max (macOS 27.0.1, build 26A434, MoltenVK, API 1.2): both cases
/// announce `metal2vulkan-degrade-indirect-index-bounds-pass` once, and the runtime's own counts
/// say which path fired -- `bounded_run_draws 1` and `cmd_draw_indexed_indirect 2`, the two
/// indirect draws the degradation's own text says one becomes. Every texel of both cases, and
/// every other field of their transcripts, compared exactly.
///
/// `render-draw-forms-lines` is the control, and it is a case rather than an argument: its
/// indexed draw is the eight-argument direct form, so it reaches no bounds pass, moves neither
/// count, declares nothing and replays **exact** on the same venue in the same run.
/// The degrade notices **every** case of the family may announce, whatever draw form it uses. A
/// list rather than a device query, because a case never branches on the device: it runs and reads
/// what came back, and a venue that announces a kind not here fails the case until the kind is
/// added.
///
/// Each is a declaration over the notices site and loosens no value: an `RGBA32Uint` texel is the
/// `uint4` the fragment function returned, so every channel of every case still compares exactly.
const PASS_NOTICES: &[Degradation] = &[
    // The vertex stage's own compiler rewrote its arithmetic as it made the pipeline, on a device
    // whose float arithmetic is not Metal's.
    Degradation::DeviceArithmetic,
    // The vertex function writes `[[point_size]]`, which a device without `shaderPointSize`'s
    // guarantee takes as its own point size rather than Metal's.
    Degradation::DevicePointSize,
];

/// The allowances every case keeps: one declaration per kind of [`PASS_NOTICES`].
fn pass_notices() -> Vec<Allowance> {
    let mut allowances: Vec<Allowance> = PASS_NOTICES
        .iter()
        .map(|kind| crate::replay::notice_allowance(kind.code()))
        .collect();
    // The positions, the indices, the indirect arguments and the read-back are all `Shared`
    // ([`crate::replay::shared_memory_notice`]).
    allowances.push(crate::replay::shared_memory_notice());
    allowances
}

fn bounds_pass_notice() -> Vec<Allowance> {
    let mut allowances = pass_notices();
    allowances.push(crate::replay::notice_allowance(
        Degradation::IndirectIndexBoundsPass.code(),
    ));
    allowances
}

/// Every case of the family, in the order the report lists them.
pub fn all() -> Vec<Box<dyn Case>> {
    vec![
        Box::new(Points),
        Box::new(Triangles),
        Box::new(Lines),
        Box::new(Empty),
    ]
}

// ------------------------------------------------------------------------------------------------
// The vocabulary each case states its draws in

/// How a draw reaches the encoder, with the parameters that are its own.
///
/// One variant a Metal selector, which is the point: a case states its table in these, the steps
/// are derived from it, and the expectation is derived from the **same** table, so a draw whose
/// parameters moved moves both at once.
#[derive(Clone, Debug)]
enum Form {
    /// `drawPrimitives:vertexStart:vertexCount:`.
    Plain { first: u32, count: u32 },
    /// `drawPrimitives:vertexStart:vertexCount:instanceCount:`.
    Instanced {
        first: u32,
        count: u32,
        instances: u32,
    },
    /// `drawPrimitives:vertexStart:vertexCount:instanceCount:baseInstance:`, whose base instance
    /// the vocabulary holds at 0.
    Based {
        first: u32,
        count: u32,
        instances: u32,
    },
    /// `drawPrimitives:indirectBuffer:indirectBufferOffset:`, whose four arguments are
    /// `vertexCount`, `instanceCount`, `vertexStart` and `baseInstance`.
    Indirect { arguments: [u32; 4] },
    /// `drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:`.
    IndexedPlain {
        index_type: IndexType,
        indices: Vec<u32>,
    },
    /// `drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:instanceCount:`.
    IndexedInstanced {
        index_type: IndexType,
        indices: Vec<u32>,
        instances: u32,
    },
    /// `drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:instanceCount:
    /// baseVertex:baseInstance:`.
    IndexedBased {
        index_type: IndexType,
        /// The indices as the buffer holds them, before the base vertex is added.
        indices: Vec<u32>,
        instances: u32,
        base_vertex: i32,
        base_instance: u32,
    },
    /// `drawIndexedPrimitives:indexType:indexBuffer:indexBufferOffset:indirectBuffer:
    /// indirectBufferOffset:`, whose five arguments are `indexCount`, `instanceCount`,
    /// `indexStart`, `baseVertex` and `baseInstance`.
    IndexedIndirect {
        index_type: IndexType,
        /// The whole run the encoder's `indexBufferOffset` points at; the arguments' `indexStart`
        /// says which of them the draw reads.
        indices: Vec<u32>,
        arguments: [u32; 5],
    },
}

impl Form {
    /// The indices the form's own run holds, where it has one.
    fn run(&self) -> Option<&[u32]> {
        match self {
            Form::IndexedPlain { indices, .. }
            | Form::IndexedInstanced { indices, .. }
            | Form::IndexedBased { indices, .. }
            | Form::IndexedIndirect { indices, .. } => Some(indices),
            _ => None,
        }
    }

    /// The step the form is encoded as, given where its own run and argument block lie.
    fn step(
        &self,
        primitive: PrimitiveType,
        indices: Handle,
        arguments: Handle,
        at: u64,
    ) -> RenderStep {
        let run = at * RUN_STRIDE;
        match self {
            Form::Plain { first, count } => RenderStep::DrawPlain {
                primitive,
                first: *first,
                count: *count,
            },
            Form::Instanced {
                first,
                count,
                instances,
            } => RenderStep::DrawInstanced {
                primitive,
                first: *first,
                count: *count,
                instances: *instances,
            },
            Form::Based {
                first,
                count,
                instances,
            } => RenderStep::Draw {
                primitive,
                first: *first,
                count: *count,
                instances: *instances,
            },
            Form::Indirect { .. } => RenderStep::DrawIndirect {
                primitive,
                arguments,
                offset: run,
            },
            Form::IndexedPlain {
                index_type,
                indices: run_indices,
            } => RenderStep::DrawIndexedPlain {
                primitive,
                count: run_indices.len() as u32,
                index_type: *index_type,
                indices,
                offset: run,
            },
            Form::IndexedInstanced {
                index_type,
                indices: run_indices,
                instances,
            } => RenderStep::DrawIndexedInstanced {
                primitive,
                count: run_indices.len() as u32,
                index_type: *index_type,
                indices,
                offset: run,
                instances: *instances,
            },
            Form::IndexedBased {
                index_type,
                indices: run_indices,
                instances,
                base_vertex,
                base_instance,
            } => RenderStep::DrawIndexed {
                primitive,
                count: run_indices.len() as u32,
                index_type: *index_type,
                indices,
                offset: run,
                instances: *instances,
                base_vertex: *base_vertex,
                base_instance: *base_instance,
            },
            Form::IndexedIndirect { index_type, .. } => RenderStep::DrawIndexedIndirect {
                primitive,
                index_type: *index_type,
                indices,
                index_offset: run,
                arguments,
                offset: run,
            },
        }
    }
}

/// One draw of a case: its mark, its primitive type, how far an instance shifts, and the form.
#[derive(Clone, Debug)]
struct Row {
    /// The word the draw's own pixels carry in channel 0. Marks start at 1, so the clear's 0 is
    /// a pixel no draw wrote.
    mark: u32,
    primitive: PrimitiveType,
    /// `frame.z`: how many pixel rows one instance shifts the draw down.
    ///
    /// A draw whose base instance is not 0 states **0** here, so where its instance ids start is a
    /// question about a channel rather than about which pixels it covered. Metal's `instance_id`
    /// may or may not include the base instance, and this family does not assume either
    /// ([`iids_either_convention`]).
    shift: u32,
    form: Form,
}

// ------------------------------------------------------------------------------------------------
// The buffers a case feeds its run

/// The marks buffer: `rows.len()` words at [`BIND_STRIDE`], word `d` the mark of row `d`.
fn marks_buffer(rows: &[Row]) -> Vec<u8> {
    let mut bytes = vec![0u8; rows.len() * BIND_STRIDE as usize];
    for (at, row) in rows.iter().enumerate() {
        let word = at * BIND_STRIDE as usize;
        bytes[word..word + 4].copy_from_slice(&row.mark.to_le_bytes());
    }
    bytes
}

/// The frames buffer: one `float4` a row, `(width, height, shift, 0)`, so a row reaches its own
/// instance shift through `setVertexBuffer:offset:atIndex:` alone.
fn frames_buffer(rows: &[Row], size: [u32; 2]) -> Vec<u8> {
    let mut bytes = Vec::new();
    for row in rows {
        for channel in [size[0] as f32, size[1] as f32, row.shift as f32, 0.0] {
            bytes.extend(channel.to_le_bytes());
        }
    }
    bytes
}

/// The positions buffer: `xy` as a case lays it out, two `float`s a vertex.
pub(super) fn positions_buffer(positions: &[[f32; 2]]) -> Vec<u8> {
    positions
        .iter()
        .flat_map(|at| {
            at.iter()
                .flat_map(|axis| axis.to_le_bytes())
                .collect::<Vec<u8>>()
        })
        .collect()
}

/// The index buffer: each row's own run at its own [`RUN_STRIDE`] block, written in the width its
/// row's index type names, so one buffer serves `UInt16` and `UInt32` runs together.
fn indices_buffer(rows: &[Row]) -> Vec<u8> {
    // A block a row, whether or not the row is an indexed draw, so a run's offset is its
    // row's index times the stride and nothing has to be numbered twice.
    let mut bytes = vec![0u8; rows.len() * RUN_STRIDE as usize];
    for (at, row) in rows.iter().enumerate() {
        let Some(run) = row.form.run() else { continue };
        let wide = matches!(
            &row.form,
            Form::IndexedPlain {
                index_type: IndexType::UInt32,
                ..
            } | Form::IndexedInstanced {
                index_type: IndexType::UInt32,
                ..
            } | Form::IndexedBased {
                index_type: IndexType::UInt32,
                ..
            } | Form::IndexedIndirect {
                index_type: IndexType::UInt32,
                ..
            }
        );
        let block = at as u64 * RUN_STRIDE;
        for (which, index) in run.iter().enumerate() {
            let word = block as usize + which * if wide { 4 } else { 2 };
            match wide {
                true => bytes[word..word + 4].copy_from_slice(&index.to_le_bytes()),
                false => bytes[word..word + 2].copy_from_slice(&(*index as u16).to_le_bytes()),
            }
        }
    }
    bytes
}

/// The indirect argument buffer: each indirect row's own block at its own [`RUN_STRIDE`].
fn arguments_buffer(rows: &[Row]) -> Vec<u8> {
    // A block a row, as the index buffer holds, and two strides wide because an indexed
    // indirect draw's five arguments take twenty bytes.
    let mut bytes = vec![0u8; rows.len() * 2 * RUN_STRIDE as usize];
    for (at, row) in rows.iter().enumerate() {
        let block = at * RUN_STRIDE as usize;
        let words: Vec<u32> = match &row.form {
            Form::Indirect { arguments } => arguments.to_vec(),
            Form::IndexedIndirect { arguments, .. } => arguments.to_vec(),
            _ => continue,
        };
        for (which, word) in words.iter().enumerate() {
            let at = block + 4 * which;
            bytes[at..at + 4].copy_from_slice(&word.to_le_bytes());
        }
    }
    bytes
}

// ------------------------------------------------------------------------------------------------
// The shape every case shares

/// The target a case draws into: a private render target of [`FORMAT`], render-target usage alone.
pub(super) fn target_spec(size: [u32; 2]) -> TargetSpec {
    TargetSpec::plain(FORMAT, size)
}

/// Bytes the whole of a `size` level 0 takes.
pub(super) fn plane(size: [u32; 2]) -> u32 {
    TEXEL * size[0] * size[1]
}

/// The layout a target's level 0 is read under: one row of the layout a row of texels, one element
/// a channel, and the bytes between the rows compared.
pub(super) fn layout(size: [u32; 2]) -> ByteLayout {
    let row_pitch = TEXEL * size[0];
    ByteLayout {
        region: Region::Texture {
            level: 0,
            slice: 0,
            aspect: Aspect::Color,
            sample: 0,
            origin: [0, 0, 0],
            size: [size[0], size[1], 1],
        },
        element: Element::U32,
        row_offset: 0,
        row_elements: 4 * size[0],
        row_pitch,
        rows: size[1],
        image_pitch: row_pitch * size[1],
        images: 1,
        gaps: Gaps::Compared,
    }
}

/// The shader reference both texts of the one fixture give.
pub(super) fn shader_refs() -> Result<Vec<ShaderRef>, String> {
    let fixture = Fixture::new(FIXTURE);
    [RUNTIME_TEXT, METAL_TEXT]
        .into_iter()
        .map(|extension| {
            Ok(ShaderRef {
                fixture: format!("{FIXTURE}{extension}"),
                entry: ENTRIES.to_owned(),
                source_sha256: crate::sha256_hex(fixture.text(extension)?.as_bytes()),
            })
        })
        .collect()
}

/// The two inputs the fixture's own texts are.
pub(super) fn fixture_inputs() -> Result<Vec<Input>, String> {
    let fixture = Fixture::new(FIXTURE);
    Ok(vec![
        Input {
            name: "air".to_owned(),
            digest: Digest::sha256(fixture.text(RUNTIME_TEXT)?.as_bytes()),
        },
        Input {
            name: "air-typed".to_owned(),
            digest: Digest::sha256(fixture.text(METAL_TEXT)?.as_bytes()),
        },
    ])
}

/// The library of the module's one fixture.
pub(super) fn library(side: &mut dyn Side, sites: &mut Sites) -> Result<Handle, Abort> {
    let loaded = Fixture::new(FIXTURE).loaded().map_err(|why| Abort {
        at: "library".to_owned(),
        failure: Failure::named("FixtureNotRead", why),
    })?;
    made(sites, "library", side.library(&loaded.source()))
}

/// The one pipeline every case makes.
pub(super) fn pipeline(colors: &[Option<ColorState>]) -> RenderPipelineSpec<'_> {
    RenderPipelineSpec::new(VERTEX, FRAGMENT, colors)
}

/// The steps one pass holds: the positions bound once, then each row's mark, its frame and its
/// draw.
fn steps(
    state: Handle,
    marks: Handle,
    frames: Handle,
    positions: Handle,
    indices: Handle,
    arguments: Handle,
    rows: &[Row],
) -> Vec<RenderStep> {
    let mut steps = vec![
        RenderStep::Pipeline(state),
        RenderStep::VertexBuffer {
            buffer: positions,
            offset: 0,
            index: 2,
        },
    ];
    for (at, row) in rows.iter().enumerate() {
        steps.push(RenderStep::VertexBuffer {
            buffer: marks,
            offset: at as u64 * BIND_STRIDE,
            index: 0,
        });
        steps.push(RenderStep::VertexBuffer {
            buffer: frames,
            offset: at as u64 * BIND_STRIDE,
            index: 1,
        });
        steps.push(row.form.step(row.primitive, indices, arguments, at as u64));
    }
    steps
}

/// The piece that copies the target's level 0 into the buffer the host reads.
fn read_target(texture: Handle, into: Handle, size: [u32; 2]) -> RenderWork {
    RenderWork::ReadTexture {
        texture,
        into,
        offset: 0,
        row_pitch: TEXEL * size[0],
        size,
        plane: Plane::Whole,
        at: ReadAt::BASE,
    }
}

/// The inputs every case of the family states, in one place: the fixture, the pipeline, the target,
/// the clear, and every buffer the draws read.
fn case_inputs(rows: &[Row], positions: &[[f32; 2]], size: [u32; 2]) -> Result<Vec<Input>, String> {
    let mut inputs = fixture_inputs()?;
    let colors = [Some(ColorState::new(FORMAT))];
    inputs.push(Input::bytes("render-pipeline", &pipeline(&colors).words()));
    inputs.push(Input::bytes("target", &target_spec(size).words()));
    inputs.push(Input::bytes("buffer:marks", &marks_buffer(rows)));
    inputs.push(Input::bytes("buffer:frames", &frames_buffer(rows, size)));
    inputs.push(Input::bytes(
        "buffer:positions",
        &positions_buffer(positions),
    ));
    inputs.push(Input::bytes("buffer:indices", &indices_buffer(rows)));
    inputs.push(Input::bytes("buffer:arguments", &arguments_buffer(rows)));
    // The steps' own parameters reach the fingerprint through the pass's words, so the draws are
    // covered there. This is the expectation's other half: a row's primitive type and its instance
    // shift decide the map the module test reads, so a row that moved is a change to the inputs.
    let mut table = Vec::new();
    for row in rows {
        table.extend(row.mark.to_le_bytes());
        table.extend(row.primitive.as_raw().to_le_bytes());
        table.extend(row.shift.to_le_bytes());
    }
    inputs.push(Input::bytes("draws", &table));
    Ok(inputs)
}

/// Run a case: make everything it needs, encode its one pass, read the target back and emit it.
fn drawn(
    side: &mut dyn Side,
    sites: &mut Sites,
    rows: &[Row],
    positions: &[[f32; 2]],
    size: [u32; 2],
) -> Result<(), Abort> {
    let library = library(side, sites)?;
    let colors = [Some(ColorState::new(FORMAT))];
    let state = made(
        sites,
        "render-pipeline",
        side.render_pipeline(library, &pipeline(&colors)),
    )?;
    let target = made(sites, "target", side.target(&target_spec(size)))?;
    let marks = made(sites, "buffer:marks", side.buffer(&marks_buffer(rows)))?;
    let frames = made(
        sites,
        "buffer:frames",
        side.buffer(&frames_buffer(rows, size)),
    )?;
    let held = made(
        sites,
        "buffer:positions",
        side.buffer(&positions_buffer(positions)),
    )?;
    let indices = made(sites, "buffer:indices", side.buffer(&indices_buffer(rows)))?;
    let arguments = made(
        sites,
        "buffer:arguments",
        side.buffer(&arguments_buffer(rows)),
    )?;
    let read_back = made(
        sites,
        "buffer:read-back",
        side.zeroed_buffer(plane(size) as usize),
    )?;
    let work = vec![
        RenderWork::Pass(
            RenderPass::clearing(vec![PassColor::cleared(target, [0.0; 4])])
                .drawing(steps(state, marks, frames, held, indices, arguments, rows)),
        ),
        read_target(target, read_back, size),
    ];
    let outcome = call(sites, "render", side.render(&work), |outcome| {
        outcome.status.clone()
    })?;
    let names = ["pass", "read"];
    let mut pieces = Node::branch("pieces", Vec::new());
    for (at, what) in names.iter().enumerate() {
        pieces = pieces.with(Node::leaf(
            *what,
            Observed::value(match outcome.refused.get(at).and_then(Option::as_ref) {
                Some(failure) => failure.recorded(),
                None => Value::ok(Value::Absent),
            }),
        ));
    }
    sites.push(pieces);
    if let Some((at, failure)) = names
        .iter()
        .enumerate()
        .find_map(|(at, what)| outcome.refused.get(at)?.as_ref().map(|f| (*what, f)))
    {
        return Err(Abort {
            at: format!("pieces/{at}"),
            failure: failure.clone(),
        });
    }
    let bytes = call(
        sites,
        "read-back",
        side.read_buffer(read_back, 0, plane(size) as usize),
        |_| Value::Absent,
    )?;
    sites.push(
        Node::branch("drawn", Vec::new())
            .with(Node::leaf("texels", Observed::bytes(bytes, layout(size)))),
    );
    Ok(())
}

// ------------------------------------------------------------------------------------------------
// render-draw-forms-points

/// The points case's target: 16 by 16, one pixel a vertex of a 16 by 16 position grid.
const POINT_SIZE: [u32; 2] = [WIDTH, 16];

/// The points case's positions: vertex `k` at the centre of pixel `(k % 16, k / 16)`, so a vertex
/// id **is** a pixel and reading the wrong index is a pixel in the wrong place.
fn point_positions() -> Vec<[f32; 2]> {
    (0..POINT_SIZE[0] * POINT_SIZE[1])
        .map(|k| {
            [
                (k % POINT_SIZE[0]) as f32 + 0.5,
                (k / POINT_SIZE[0]) as f32 + 0.5,
            ]
        })
        .collect()
}

/// `MTLPrimitiveTypePoint` through all eight draw selectors.
pub struct Points;

/// The points case's nine draws, in the order the pass encodes them.
///
/// Rows 0 to 14 of the target are a draw's; **row 7 and row 15 are nobody's**, and holding the
/// clear is half of what the map says.
fn point_rows() -> Vec<Row> {
    let at = |column: u32, row: u32| row * POINT_SIZE[0] + column;
    vec![
        // The shortest selector there is, at vertex start 0: six points along row 0.
        Row {
            mark: 1,
            primitive: PrimitiveType::Point,
            shift: 1,
            form: Form::Plain {
                first: at(0, 0),
                count: 6,
            },
        },
        // Two instances, which is two rows: the instance count is the second row.
        Row {
            mark: 2,
            primitive: PrimitiveType::Point,
            shift: 1,
            form: Form::Instanced {
                first: at(0, 1),
                count: 5,
                instances: 2,
            },
        },
        // Three instances through the form that also names a base instance, which the vocabulary
        // holds at 0: three rows, and the `base` channel says 0.
        Row {
            mark: 3,
            primitive: PrimitiveType::Point,
            shift: 1,
            form: Form::Based {
                first: at(0, 3),
                count: 4,
                instances: 3,
            },
        },
        // An indirect draw whose buffer names a base instance of 7. Its frame shifts nothing, so
        // where its instance ids start is a channel rather than a row.
        Row {
            mark: 4,
            primitive: PrimitiveType::Point,
            shift: 0,
            form: Form::Indirect {
                arguments: [4, 1, at(0, 6), 7],
            },
        },
        // The shortest indexed selector, at index-buffer offset 0, over `UInt16` indices in an
        // order no vertex range could produce.
        Row {
            mark: 5,
            primitive: PrimitiveType::Point,
            shift: 1,
            form: Form::IndexedPlain {
                index_type: IndexType::UInt16,
                indices: vec![at(3, 8), at(1, 8), at(4, 8), at(0, 8), at(2, 8)],
            },
        },
        // `UInt32` indices, two instances: the other index type and the other row.
        Row {
            mark: 6,
            primitive: PrimitiveType::Point,
            shift: 1,
            form: Form::IndexedInstanced {
                index_type: IndexType::UInt32,
                indices: vec![at(5, 9), at(2, 9), at(7, 9), at(0, 9)],
                instances: 2,
            },
        },
        // A **positive** base vertex: the run's indices are 0, 2, 3 and 5, so a side that dropped
        // the base vertex would write row 0 -- over mark 1's own pixels.
        Row {
            mark: 7,
            primitive: PrimitiveType::Point,
            shift: 1,
            form: Form::IndexedBased {
                index_type: IndexType::UInt16,
                indices: vec![0, 2, 3, 5],
                instances: 2,
                base_vertex: at(0, 11) as i32,
                base_instance: 0,
            },
        },
        // A **negative** base vertex beside a base instance of 5: the indices name row 14 and the
        // base vertex takes them back a row, so a side that dropped it would write row 14 -- over
        // the row the next draw owns.
        Row {
            mark: 8,
            primitive: PrimitiveType::Point,
            shift: 0,
            form: Form::IndexedBased {
                index_type: IndexType::UInt16,
                indices: vec![at(1, 14), at(3, 14), at(6, 14)],
                instances: 1,
                base_vertex: -(POINT_SIZE[0] as i32),
                base_instance: 5,
            },
        },
        // An indexed indirect draw from index 1 of its run: index 0 is the spare, and the pixel it
        // names stays clear exactly while `indexStart` is honoured.
        Row {
            mark: 9,
            primitive: PrimitiveType::Point,
            shift: 0,
            form: Form::IndexedIndirect {
                index_type: IndexType::UInt16,
                indices: vec![at(15, 14), at(0, 14), at(2, 14), at(7, 14)],
                arguments: [3, 1, 1, 0, 9],
            },
        },
    ]
}

impl Case for Points {
    fn name(&self) -> &'static str {
        "render-draw-forms-points"
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn record_runs(&self) -> u32 {
        RUNS
    }

    fn allowances(&self) -> Vec<Allowance> {
        // No value is loosened. An `RGBA32Uint` texel is the `uint4` the fragment function
        // returned: there is no conversion, no interpolation and no arithmetic of the shader's in
        // the path, so every channel is an exact answer or it is a bug. What is declared is the one
        // notice this case's own indexed indirect draw provokes ([`bounds_pass_notice`]).
        bounds_pass_notice()
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        shader_refs()
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        case_inputs(&point_rows(), &point_positions(), POINT_SIZE)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        drawn(side, sites, &point_rows(), &point_positions(), POINT_SIZE)
    }
}

// ------------------------------------------------------------------------------------------------
// render-draw-forms-triangles

/// The triangles case's target: 16 by 20, ten bands of two rows.
const TRIANGLE_SIZE: [u32; 2] = [WIDTH, 20];

/// How many position entries one band of the triangles case holds.
const BAND_VERTICES: u32 = 12;

/// The triangles case's positions: ten bands of twelve entries, the first six the vertex list of
/// two triangles filling the band's quad and the next four the strip of the **same two triangles**.
///
/// The quad of band `b` is the pixel rectangle `[0, 16) x [2b, 2b + 2)`, and its corners are stated
/// at pixel **boundaries** rather than centres, so the top-left fill rule decides every pixel and
/// no pixel centre lies on the shared diagonal: it crosses row `2b` at `x = 12` and row `2b + 1` at
/// `x = 4`, and a centre is at a half-integer.
fn triangle_positions() -> Vec<[f32; 2]> {
    let mut positions = Vec::new();
    for band in 0..TRIANGLE_SIZE[1] / 2 {
        let top = (2 * band) as f32;
        let bottom = (2 * band + 2) as f32;
        let right = TRIANGLE_SIZE[0] as f32;
        // The list: triangle (0, top), (right, top), (0, bottom), then triangle (right, top),
        // (0, bottom), (right, bottom).
        positions.extend([
            [0.0, top],
            [right, top],
            [0.0, bottom],
            [right, top],
            [0.0, bottom],
            [right, bottom],
        ]);
        // The strip: the same two triangles as (v0, v1, v2) and (v1, v2, v3).
        positions.extend([[0.0, top], [right, top], [0.0, bottom], [right, bottom]]);
        // The band's two spare entries, which no draw of this case reads: they keep a band's stride
        // at twelve so a band's own start is arithmetic.
        positions.extend([[0.0, top], [right, bottom]]);
    }
    positions
}

/// `MTLPrimitiveTypeTriangle` and `MTLPrimitiveTypeTriangleStrip` over the same quad.
pub struct Triangles;

/// The triangles case's seven draws. Bands 0 to 8 are a draw's; **band 9 is nobody's**.
fn triangle_rows() -> Vec<Row> {
    let list = |band: u32| band * BAND_VERTICES;
    let strip = |band: u32| band * BAND_VERTICES + 6;
    vec![
        Row {
            mark: 1,
            primitive: PrimitiveType::Triangle,
            shift: 2,
            form: Form::Plain {
                first: list(0),
                count: 6,
            },
        },
        Row {
            mark: 2,
            primitive: PrimitiveType::Triangle,
            shift: 2,
            form: Form::Instanced {
                first: list(1),
                count: 6,
                instances: 2,
            },
        },
        // The strip over two instances: four vertices a band where the list takes six.
        Row {
            mark: 3,
            primitive: PrimitiveType::TriangleStrip,
            shift: 2,
            form: Form::Based {
                first: strip(3),
                count: 4,
                instances: 2,
            },
        },
        // An indirect list draw with a base instance of 6 and no instance shift.
        Row {
            mark: 4,
            primitive: PrimitiveType::Triangle,
            shift: 0,
            form: Form::Indirect {
                arguments: [6, 1, list(5), 6],
            },
        },
        // A `UInt16` indexed list at offset 0, its indices in the list's own order.
        Row {
            mark: 5,
            primitive: PrimitiveType::Triangle,
            shift: 2,
            form: Form::IndexedPlain {
                index_type: IndexType::UInt16,
                indices: (list(6)..list(6) + 6).collect(),
            },
        },
        // A `UInt32` indexed strip with a positive base vertex: the run's indices are 0 to 3, so a
        // side that dropped the base vertex would draw band 0 over mark 1's quad.
        Row {
            mark: 6,
            primitive: PrimitiveType::TriangleStrip,
            shift: 0,
            form: Form::IndexedBased {
                index_type: IndexType::UInt32,
                indices: vec![0, 1, 2, 3],
                instances: 1,
                base_vertex: strip(7) as i32,
                base_instance: 3,
            },
        },
        // An indexed indirect strip from index 1 of its run. Index 0 repeats index 1, so dropping
        // `indexStart` leaves the quad's lower-right half **clear**: the first triangle becomes
        // degenerate and the second covers the upper-left half alone.
        Row {
            mark: 7,
            primitive: PrimitiveType::TriangleStrip,
            shift: 0,
            form: Form::IndexedIndirect {
                index_type: IndexType::UInt16,
                indices: vec![strip(8), strip(8), strip(8) + 1, strip(8) + 2, strip(8) + 3],
                arguments: [4, 1, 1, 0, 11],
            },
        },
    ]
}

impl Case for Triangles {
    fn name(&self) -> &'static str {
        "render-draw-forms-triangles"
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn record_runs(&self) -> u32 {
        RUNS
    }

    fn allowances(&self) -> Vec<Allowance> {
        // No value loosened, for the reason the points case states, and the same one notice: this
        // case makes an indexed indirect draw too ([`bounds_pass_notice`]).
        bounds_pass_notice()
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        shader_refs()
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        case_inputs(&triangle_rows(), &triangle_positions(), TRIANGLE_SIZE)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        drawn(
            side,
            sites,
            &triangle_rows(),
            &triangle_positions(),
            TRIANGLE_SIZE,
        )
    }
}

// ------------------------------------------------------------------------------------------------
// render-draw-forms-lines

/// The lines case's target: 16 by 16, one row a band.
const LINE_SIZE: [u32; 2] = [WIDTH, 16];

/// How many position entries one band of the lines case holds.
const LINE_VERTICES: u32 = 8;

/// Where a band's line-list segments begin and end, in pixel columns.
const LIST_SEGMENTS: [[u32; 2]; 2] = [[2, 7], [9, 13]];

/// Where a band's line-strip vertices lie, in pixel columns.
const STRIP_COLUMNS: [u32; 4] = [1, 5, 10, 14];

/// The lines case's positions: sixteen bands of eight entries, the first four the two segments of
/// a line list and the next four a strip of three segments, every endpoint at a pixel **centre** on
/// the band's own row.
///
/// Horizontal segments at pixel centres on purpose: of every shape a line can take it is the one
/// whose covered pixels are least a question of the diamond-exit rule, so what this case asks is
/// which draw selectors reach a line draw at all and which vertex provokes each segment, with the
/// coverage rule itself contributing as little as a line case can.
fn line_positions() -> Vec<[f32; 2]> {
    let mut positions = Vec::new();
    for band in 0..LINE_SIZE[1] {
        let y = band as f32 + 0.5;
        for [from, to] in LIST_SEGMENTS {
            positions.push([from as f32 + 0.5, y]);
            positions.push([to as f32 + 0.5, y]);
        }
        for column in STRIP_COLUMNS {
            positions.push([column as f32 + 0.5, y]);
        }
    }
    positions
}

/// `MTLPrimitiveTypeLine` and `MTLPrimitiveTypeLineStrip` over horizontal segments.
pub struct Lines;

/// The lines case's five draws. Bands 0 to 9 are a draw's; **bands 10 to 15 are nobody's**.
fn line_rows() -> Vec<Row> {
    let list = |band: u32| band * LINE_VERTICES;
    let strip = |band: u32| band * LINE_VERTICES + 4;
    vec![
        Row {
            mark: 1,
            primitive: PrimitiveType::Line,
            shift: 1,
            form: Form::Plain {
                first: list(0),
                count: 4,
            },
        },
        Row {
            mark: 2,
            primitive: PrimitiveType::Line,
            shift: 1,
            form: Form::Instanced {
                first: list(1),
                count: 4,
                instances: 2,
            },
        },
        Row {
            mark: 3,
            primitive: PrimitiveType::LineStrip,
            shift: 1,
            form: Form::Based {
                first: strip(3),
                count: 4,
                instances: 2,
            },
        },
        // A `UInt16` indexed line list at offset 0, its two segments in the reverse of the buffer's
        // order, which no vertex range could produce.
        Row {
            mark: 4,
            primitive: PrimitiveType::Line,
            shift: 1,
            form: Form::IndexedPlain {
                index_type: IndexType::UInt16,
                indices: vec![list(6) + 2, list(6) + 3, list(6), list(6) + 1],
            },
        },
        // A `UInt32` indexed strip with a positive base vertex and a base instance of 4.
        Row {
            mark: 5,
            primitive: PrimitiveType::LineStrip,
            shift: 0,
            form: Form::IndexedBased {
                index_type: IndexType::UInt32,
                indices: vec![0, 1, 2, 3],
                instances: 1,
                base_vertex: strip(8) as i32,
                base_instance: 4,
            },
        },
    ]
}

impl Case for Lines {
    fn name(&self) -> &'static str {
        "render-draw-forms-lines"
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn record_runs(&self) -> u32 {
        RUNS
    }

    fn allowances(&self) -> Vec<Allowance> {
        // Nothing loosened, for the reason the points case states, and no bounds-pass notice:
        // this case's indexed draw is the eight-argument direct form, so it reaches no bounds
        // pass. A line's covered set is a rasterization rule rather than arithmetic, so the module
        // test holds this case to what any rule implies and the comparison holds both legs to the
        // same bytes: a coverage rule that differs is a finding, not a tolerance.
        pass_notices()
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        shader_refs()
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        case_inputs(&line_rows(), &line_positions(), LINE_SIZE)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        drawn(side, sites, &line_rows(), &line_positions(), LINE_SIZE)
    }
}

// ------------------------------------------------------------------------------------------------
// render-draw-forms-empty

/// A draw of **no vertices and no indices** through every selector that can take one, between two
/// draws that write.
///
/// The documented boundary of every count these selectors carry. Plain Metal draws nothing for a
/// draw of no vertices, indices or instances (`metal2vulkan_runtime`'s `render.rs` module docs,
/// measured on an M4), which is a **defined** answer rather than an undefined one: the band stays
/// the pass's clear. So it compares exactly, which is the whole reason the boundary is worth a case.
///
/// A target that is all clear would prove nothing, so marks 1 and 10 write the first and last rows
/// through the shortest selector there is: the pass ran, a draw of this pipeline over these
/// bindings does write, and the eight draws between them wrote nothing. Mark 9's indirect draw
/// names a **real** run of three indices into row 9, and its arguments say `indexCount` 0 -- so row
/// 9 staying clear is a reading of the count rather than of an empty buffer.
///
/// **What this case does not sweep, and why.** An `instanceCount` of 0 is the other half of the
/// boundary and is not here: plain Metal draws nothing for it too, but Metal's own API validation
/// layer **stops** it (the same module docs, the probe case `zero_instances`), so the suite's
/// validated oracle would record an abort of its child rather than a fact. Pairing the two is a
/// `rate_maps`-shaped case that declares a plain-against-validated disagreement, and whether the
/// suite should hold one here is a decision above this family.
const EMPTY_SIZE: [u32; 2] = POINT_SIZE;

/// A draw of no vertices or no indices through every selector that can take one.
pub struct Empty;

/// The empty case's ten draws: two that write, and eight of no vertices or no indices between them.
fn empty_rows() -> Vec<Row> {
    let at = |column: u32, row: u32| row * EMPTY_SIZE[0] + column;
    let point = |mark: u32, form: Form| Row {
        mark,
        primitive: PrimitiveType::Point,
        shift: 1,
        form,
    };
    vec![
        // The positive control ahead of the empty draws.
        point(
            1,
            Form::Plain {
                first: at(0, 0),
                count: 6,
            },
        ),
        point(
            2,
            Form::Plain {
                first: at(0, 2),
                count: 0,
            },
        ),
        point(
            3,
            Form::Instanced {
                first: at(0, 3),
                count: 0,
                instances: 3,
            },
        ),
        point(
            4,
            Form::Based {
                first: at(0, 4),
                count: 0,
                instances: 2,
            },
        ),
        // An indirect draw whose `vertexCount` is 0 and whose `vertexStart` names row 5.
        point(
            5,
            Form::Indirect {
                arguments: [0, 1, at(0, 5), 0],
            },
        ),
        point(
            6,
            Form::IndexedPlain {
                index_type: IndexType::UInt16,
                indices: Vec::new(),
            },
        ),
        point(
            7,
            Form::IndexedInstanced {
                index_type: IndexType::UInt32,
                indices: Vec::new(),
                instances: 2,
            },
        ),
        point(
            8,
            Form::IndexedBased {
                index_type: IndexType::UInt16,
                indices: Vec::new(),
                instances: 1,
                base_vertex: at(0, 8) as i32,
                base_instance: 4,
            },
        ),
        // The run is three real indices into row 9 and the arguments say `indexCount` 0, so row 9
        // staying clear reads the count rather than an empty buffer.
        point(
            9,
            Form::IndexedIndirect {
                index_type: IndexType::UInt16,
                indices: vec![at(0, 9), at(2, 9), at(4, 9)],
                arguments: [0, 1, 0, 0, 0],
            },
        ),
        // The positive control after them.
        point(
            10,
            Form::Plain {
                first: at(0, 15),
                count: 6,
            },
        ),
    ]
}

impl Case for Empty {
    fn name(&self) -> &'static str {
        "render-draw-forms-empty"
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn record_runs(&self) -> u32 {
        RUNS
    }

    fn allowances(&self) -> Vec<Allowance> {
        // This case makes an indexed indirect draw, so it declares the same one notice the points
        // and triangles cases do ([`bounds_pass_notice`]) -- the runtime bounds the indices before
        // the pass whatever the count turns out to be. No value is loosened.
        bounds_pass_notice()
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        shader_refs()
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        case_inputs(&empty_rows(), &point_positions(), EMPTY_SIZE)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        drawn(side, sites, &empty_rows(), &point_positions(), EMPTY_SIZE)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::allowance::{Justification, Loosening};
    use crate::transcript::Transcript;
    use std::collections::{BTreeMap, BTreeSet};

    // The three readings below are the expectation's half of the draw table: what a form
    // resolves to, said in Rust beside what the encoder is told. They live here rather than on
    // `Form` because nothing a case runs needs them -- the steps carry the parameters, and
    // these derive what the parameters mean.

    /// The vertex ids the draw resolves to, in the order the primitives take them.
    fn vertices(form: &Form) -> Vec<u32> {
        match form {
            Form::Plain { first, count } => (*first..first + count).collect(),
            Form::Instanced { first, count, .. } | Form::Based { first, count, .. } => {
                (*first..first + count).collect()
            }
            Form::Indirect { arguments } => {
                let [count, _, first, _] = *arguments;
                (first..first + count).collect()
            }
            Form::IndexedPlain { indices, .. } | Form::IndexedInstanced { indices, .. } => {
                indices.clone()
            }
            Form::IndexedBased {
                indices,
                base_vertex,
                ..
            } => indices
                .iter()
                .map(|index| (*index as i64 + i64::from(*base_vertex)) as u32)
                .collect(),
            Form::IndexedIndirect {
                indices, arguments, ..
            } => {
                let [count, _, start, base, _] = *arguments;
                let base = base as i32;
                (start..start + count)
                    .map(|at| (indices[at as usize] as i64 + i64::from(base)) as u32)
                    .collect()
            }
        }
    }

    /// How many instances the draw runs.
    fn instances(form: &Form) -> u32 {
        match form {
            Form::Plain { .. } | Form::IndexedPlain { .. } => 1,
            Form::Instanced { instances, .. }
            | Form::Based { instances, .. }
            | Form::IndexedInstanced { instances, .. }
            | Form::IndexedBased { instances, .. } => *instances,
            Form::Indirect { arguments } => arguments[1],
            Form::IndexedIndirect { arguments, .. } => arguments[1],
        }
    }

    /// The `baseInstance` the draw passes, which the fragment function answers as its `base`
    /// channel.
    fn base_instance(form: &Form) -> u32 {
        match form {
            Form::Plain { .. }
            | Form::Instanced { .. }
            | Form::Based { .. }
            | Form::IndexedPlain { .. }
            | Form::IndexedInstanced { .. } => 0,
            Form::IndexedBased { base_instance, .. } => *base_instance,
            Form::Indirect { arguments } => arguments[3],
            Form::IndexedIndirect { arguments, .. } => arguments[4],
        }
    }

    /// What one texel is expected to hold: the four channels, in the fragment function's order.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct Texel {
        mark: u32,
        vid: u32,
        iid: u32,
        base: u32,
    }

    /// The texel a pixel no draw covered holds: the pass's clear, which is zero in every channel.
    const CLEAR: Texel = Texel {
        mark: 0,
        vid: 0,
        iid: 0,
        base: 0,
    };

    /// The one outcome of `case`'s recorded Metal answer, or `None` where nothing has recorded it.
    fn only_outcome(case: &str) -> Option<Transcript> {
        let recording = crate::recording::read(case).ok()?;
        let answer = recording.answers.first()?;
        assert_eq!(
            answer.outcomes.len(),
            1,
            "{case}: {} outcomes over {} runs, so a draw's effect moved between runs",
            answer.outcomes.len(),
            answer.runs
        );
        Some(answer.outcomes[0].transcript.clone())
    }

    /// The target's recorded texels of `case`.
    fn recorded(case: &str, size: [u32; 2]) -> Option<Vec<u8>> {
        let transcript = only_outcome(case)?;
        let node = transcript
            .root
            .child("drawn")
            .and_then(|node| node.child("texels"))
            .expect("no drawn/texels site");
        let Value::Bytes(bytes) = &node.observed.value else {
            panic!("drawn/texels holds no bytes");
        };
        assert_eq!(bytes.len(), plane(size) as usize);
        Some(bytes.clone())
    }

    /// The texel at `(x, y)` of a target's bytes.
    fn texel(bytes: &[u8], x: u32, y: u32, size: [u32; 2]) -> Texel {
        let at = ((y * size[0] + x) * TEXEL) as usize;
        let word = |which: usize| {
            let from = at + 4 * which;
            u32::from_le_bytes([
                bytes[from],
                bytes[from + 1],
                bytes[from + 2],
                bytes[from + 3],
            ])
        };
        Texel {
            mark: word(0),
            vid: word(1),
            iid: word(2),
            base: word(3),
        }
    }

    /// The instance ids a draw of `instances` instances from `base` runs, under each of the two
    /// conventions Metal may hold: `instance_id` including the base instance, or not.
    ///
    /// The family does not assume either. Where a draw's base instance is 0 the two agree, and
    /// where it is not, the draw's frame shifts nothing, so the convention reaches the `iid`
    /// channel and never the pixel.
    fn iids_either_convention(instances: u32, base: u32, including: bool) -> Vec<u32> {
        let from = if including { base } else { 0 };
        (from..from + instances).collect()
    }

    /// What the points case's target holds, from the draw table and nothing Metal was asked.
    ///
    /// Later draws overwrite earlier ones, as the pass encodes them, which is why this is built in
    /// order rather than decided per pixel.
    fn points_expected(including: bool) -> BTreeMap<(u32, u32), Texel> {
        let mut map = BTreeMap::new();
        let positions = point_positions();
        for row in point_rows() {
            let base = base_instance(&row.form);
            for iid in iids_either_convention(instances(&row.form), base, including) {
                // The row's own shift is what turns an instance into a row, and the shift a draw
                // with a base instance states is 0, so the instance ids below never move a pixel.
                let down = row.shift * if row.shift == 0 { 0 } else { iid };
                for vid in vertices(&row.form) {
                    let at = positions[vid as usize];
                    let (x, y) = (at[0] as u32, at[1] as u32 + down);
                    map.insert(
                        (x, y),
                        Texel {
                            mark: row.mark,
                            vid,
                            iid,
                            base,
                        },
                    );
                }
            }
        }
        map
    }

    /// **Every index any draw of the family can read names a vertex the case states.**
    ///
    /// An index past the positions array is an out-of-bounds read of a `device const float2 *`,
    /// which is undefined behaviour on the GPU and is not a thing a case may ask. Held with no
    /// device asked, over the resolved vertex ids rather than the raw indices, so a base vertex is
    /// in the claim.
    #[test]
    fn every_index_names_a_position_the_case_states() {
        // The check sees a vertex id past the array, so a green run means the cases' indices are
        // inside rather than that the check cannot fire.
        let past = Form::Plain {
            first: 1_000,
            count: 1,
        };
        assert!(vertices(&past).iter().any(|vid| *vid >= 256));
        for (rows, positions, size) in [
            (point_rows(), point_positions(), POINT_SIZE),
            (triangle_rows(), triangle_positions(), TRIANGLE_SIZE),
            (line_rows(), line_positions(), LINE_SIZE),
            (empty_rows(), point_positions(), EMPTY_SIZE),
        ] {
            for row in &rows {
                for vid in vertices(&row.form) {
                    assert!(
                        (vid as usize) < positions.len(),
                        "mark {} reads vertex {vid} of {} positions",
                        row.mark,
                        positions.len()
                    );
                }
                // A row's spare index is read by no honoured form, but a side that dropped
                // `indexStart` would read it, so it has to be inside the array too.
                if let Some(run) = row.form.run() {
                    for index in run {
                        let resolved = match &row.form {
                            Form::IndexedBased { base_vertex, .. } => {
                                *index as i64 + i64::from(*base_vertex)
                            }
                            _ => *index as i64,
                        };
                        assert!(
                            resolved >= 0 && (resolved as usize) < positions.len(),
                            "mark {} holds index {index}, which resolves to {resolved}",
                            row.mark
                        );
                    }
                }
            }
            // Every position lies inside the target, which is what keeps a draw's pixels on it.
            for at in &positions {
                assert!(at[0] >= 0.0 && at[0] <= size[0] as f64 as f32, "{at:?}");
                assert!(at[1] >= 0.0 && at[1] <= size[1] as f64 as f32, "{at:?}");
            }
        }
    }

    /// **The five primitive types cover different pixels**, which is what makes sweeping them
    /// worth more than naming them.
    ///
    /// Stated over the cases' own geometry with no device asked: a point draw of `n` vertices
    /// covers `n` pixels, the two triangle types each fill a whole 16 by 2 band from the same four
    /// corners, and a line draw covers neither. A sweep whose values were indistinguishable by
    /// construction would fail here.
    #[test]
    fn the_primitive_types_cover_different_pixels() {
        // Points: one pixel a vertex, and the first draw's six are six pixels of row 0.
        let points = points_expected(true);
        let first: Vec<(u32, u32)> = points
            .iter()
            .filter(|(_, held)| held.mark == 1)
            .map(|(at, _)| *at)
            .collect();
        assert_eq!(first, vec![(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)]);
        // Triangles: the two triangle types take the same four corners of a band, so the shape
        // they fill is one shape and only the provoking vertex can tell them apart. Held over the
        // positions rather than over a recording.
        let positions = triangle_positions();
        for band in 0..TRIANGLE_SIZE[1] / 2 {
            let list = (band * BAND_VERTICES) as usize;
            let strip = list + 6;
            let corners: BTreeSet<[u32; 2]> = positions[list..list + 6]
                .iter()
                .map(|at| [at[0] as u32, at[1] as u32])
                .collect();
            let strip_corners: BTreeSet<[u32; 2]> = positions[strip..strip + 4]
                .iter()
                .map(|at| [at[0] as u32, at[1] as u32])
                .collect();
            assert_eq!(corners, strip_corners, "band {band}");
            assert_eq!(corners.len(), 4);
        }
        // The list's two triangles and the strip's two are the same two triangles, which is the
        // claim the `vid` channel then separates.
        let list = 0usize;
        let strip = 6usize;
        let triangle = |at: usize, which: [usize; 3]| -> BTreeSet<[u32; 2]> {
            which
                .iter()
                .map(|offset| {
                    let point = positions[at + offset];
                    [point[0] as u32, point[1] as u32]
                })
                .collect()
        };
        assert_eq!(triangle(list, [0, 1, 2]), triangle(strip, [0, 1, 2]));
        assert_eq!(triangle(list, [3, 4, 5]), triangle(strip, [1, 2, 3]));
        // Lines: the endpoints are pixel centres on one row, so a segment's pixels are a subset of
        // its own row and a band is one row tall -- which no triangle band of this family is.
        let line = line_positions();
        for band in 0..LINE_SIZE[1] {
            for which in 0..LINE_VERTICES {
                let at = line[(band * LINE_VERTICES + which) as usize];
                assert_eq!(at[1], band as f32 + 0.5, "band {band} vertex {which}");
                assert_eq!(at[0].fract(), 0.5);
            }
        }
        // The three coverages are three different shapes, which is the claim this test's name
        // makes and the part a sweep of indistinguishable values would fail.
        //
        // A **point** draw covers one pixel a vertex, so its covered count is its vertex count. A
        // **triangle** draw of the same six vertices fills a 16 by 2 band: 32 pixels, which
        // `metal_filled_each_triangle_band_as_its_assembly_says` holds as the 16 and 16 the shared
        // diagonal splits them into, and 32 is not 6. A **line** draw's band is one row tall where
        // a triangle band is two, so no line draw can cover a triangle band and no triangle draw
        // can cover a line band.
        let point_draw = &point_rows()[0];
        assert_eq!(vertices(&point_draw.form).len(), 6);
        assert_eq!(
            points
                .values()
                .filter(|held| held.mark == point_draw.mark)
                .count(),
            vertices(&point_draw.form).len(),
            "a point draw covers one pixel a vertex"
        );
        let triangle_band = TRIANGLE_SIZE[0] * 2;
        assert_eq!(triangle_band, 32);
        assert_ne!(triangle_band as usize, vertices(&point_draw.form).len());
        // A line band is one row and a triangle band is two, read off the two cases' own layouts.
        let line_band_rows = line_rows()[0].shift;
        let triangle_band_rows = triangle_rows()[0].shift;
        assert_eq!(line_band_rows, 1);
        assert_eq!(triangle_band_rows, 2);
        assert_ne!(line_band_rows, triangle_band_rows);
    }

    /// **The points map names every draw apart, and the rows nobody drew hold the clear.**
    ///
    /// Over the rule rather than a recording, so it holds with no device and fails the moment a
    /// row's parameters are changed into another row's.
    #[test]
    fn the_points_map_names_every_draw_apart() {
        let rows = point_rows();
        let marks: BTreeSet<u32> = rows.iter().map(|row| row.mark).collect();
        assert_eq!(marks.len(), rows.len(), "two draws share a mark");
        assert!(!marks.contains(&CLEAR.mark), "a draw's mark is the clear's");
        for including in [false, true] {
            let map = points_expected(including);
            // Rows 7 and 15 are nobody's, whichever convention holds.
            for y in [7, 15] {
                for x in 0..POINT_SIZE[0] {
                    assert!(
                        !map.contains_key(&(x, y)),
                        "({x}, {y}) is covered and should be the clear"
                    );
                }
            }
            // Each draw covers the rows the table says and no other.
            let mut rows_of: BTreeMap<u32, BTreeSet<u32>> = BTreeMap::new();
            for ((_, y), held) in &map {
                rows_of.entry(held.mark).or_default().insert(*y);
            }
            assert_eq!(rows_of[&1], BTreeSet::from([0]));
            assert_eq!(rows_of[&2], BTreeSet::from([1, 2]));
            assert_eq!(rows_of[&3], BTreeSet::from([3, 4, 5]));
            assert_eq!(rows_of[&4], BTreeSet::from([6]));
            assert_eq!(rows_of[&5], BTreeSet::from([8]));
            assert_eq!(rows_of[&6], BTreeSet::from([9, 10]));
            assert_eq!(rows_of[&7], BTreeSet::from([11, 12]));
            assert_eq!(rows_of[&8], BTreeSet::from([13]));
            assert_eq!(rows_of[&9], BTreeSet::from([14]));
            // The base instances reach the `base` channel and nothing else.
            let base_of = |mark: u32| {
                map.iter()
                    .find(|(_, held)| held.mark == mark)
                    .map(|(_, held)| held.base)
                    .unwrap()
            };
            assert_eq!(base_of(4), 7);
            assert_eq!(base_of(8), 5);
            assert_eq!(base_of(9), 9);
            for mark in [1, 2, 3, 5, 6, 7] {
                assert_eq!(base_of(mark), 0, "mark {mark}");
            }
            // Mark 9's spare index names a pixel no honoured draw covers: dropping `indexStart`
            // would cover it, so its staying clear is the reading of the parameter.
            assert!(!map.contains_key(&(15, 14)));
        }
        // The two conventions differ exactly where a base instance is not 0, which is what makes
        // reading the convention from the recording a question with an answer.
        let (excluding, including) = (points_expected(false), points_expected(true));
        assert_ne!(excluding, including);
        for (at, held) in &including {
            if held.base == 0 {
                assert_eq!(excluding[at], *held, "{at:?} moved with no base instance");
            }
        }
    }

    /// **Metal's recorded points target holds, in every texel, what the draw table says.**
    ///
    /// The instance-id convention is read from the recording rather than assumed, as
    /// [`iids_either_convention`] says, and the reading is then held over every draw that has a
    /// base instance: a convention that moved between draws is the finding.
    #[test]
    fn metal_drew_each_point_form_as_its_parameters_say() {
        let Some(bytes) = recorded(Points.name(), POINT_SIZE) else {
            println!("render-draw-forms-points: no recording in the tree yet");
            return;
        };
        // Mark 4's draw is one instance from base instance 7, so its `iid` channel is 7 under the
        // including convention and 0 under the other.
        let probe = texel(&bytes, 0, 6, POINT_SIZE);
        assert_eq!(probe.mark, 4, "(0, 6) is not mark 4's pixel");
        let including = match probe.iid {
            0 => false,
            7 => true,
            other => panic!("mark 4's instance id is {other}, which is neither 0 nor its base 7"),
        };
        let map = points_expected(including);
        let mut differ = Vec::new();
        for y in 0..POINT_SIZE[1] {
            for x in 0..POINT_SIZE[0] {
                let got = texel(&bytes, x, y, POINT_SIZE);
                let want = map.get(&(x, y)).copied().unwrap_or(CLEAR);
                if got != want {
                    differ.push(format!(
                        "({x}, {y}): Metal {got:?}, the table says {want:?}"
                    ));
                }
            }
        }
        for line in differ.iter().take(20) {
            println!("POINTS-DIFFERS: {line}");
        }
        assert!(
            differ.is_empty(),
            "{} of {} texels differ from what the draws' own parameters say (Metal's instance id \
             {} the base instance)",
            differ.len(),
            POINT_SIZE[0] * POINT_SIZE[1],
            match including {
                true => "includes",
                false => "excludes",
            }
        );
    }

    /// **Metal's recorded triangles target fills each draw's band and leaves band 9 clear**, and
    /// the two halves of every band carry the two provoking vertices the list or the strip implies.
    #[test]
    fn metal_filled_each_triangle_band_as_its_assembly_says() {
        let Some(bytes) = recorded(Triangles.name(), TRIANGLE_SIZE) else {
            println!("render-draw-forms-triangles: no recording in the tree yet");
            return;
        };
        let rows = triangle_rows();
        // Which bands each draw is expected in, from its own table entry.
        let mut bands: BTreeMap<u32, BTreeSet<u32>> = BTreeMap::new();
        let positions = triangle_positions();
        for row in &rows {
            let base = base_instance(&row.form);
            // A draw with a base instance states no shift, so its band is its vertices' own
            // whichever instance-id convention holds.
            for iid in iids_either_convention(instances(&row.form), base, false) {
                let top = vertices(&row.form)
                    .iter()
                    .map(|vid| positions[*vid as usize][1] as u32)
                    .min()
                    .unwrap();
                bands
                    .entry(row.mark)
                    .or_default()
                    .insert(top / 2 + if row.shift == 0 { 0 } else { iid });
            }
        }
        // Every pixel of a draw's band carries that draw's mark, and the band's 32 pixels split
        // into exactly two provoking vertices.
        for row in &rows {
            for band in &bands[&row.mark] {
                let mut vids: BTreeMap<u32, usize> = BTreeMap::new();
                for y in 2 * band..2 * band + 2 {
                    for x in 0..TRIANGLE_SIZE[0] {
                        let got = texel(&bytes, x, y, TRIANGLE_SIZE);
                        assert_eq!(
                            got.mark, row.mark,
                            "({x}, {y}) of band {band} holds mark {} and not {}",
                            got.mark, row.mark
                        );
                        assert_eq!(got.base, base_instance(&row.form));
                        *vids.entry(got.vid).or_default() += 1;
                    }
                }
                assert_eq!(
                    vids.len(),
                    2,
                    "band {band} of mark {} holds {} provoking vertices: {vids:?}",
                    row.mark,
                    vids.len()
                );
                // The diagonal crosses row 0 of the band at x = 12 and row 1 at x = 4, so the
                // upper-left triangle takes 12 + 4 = 16 pixels and the other the remaining 16.
                let counts: Vec<usize> = vids.values().copied().collect();
                assert_eq!(counts, vec![16, 16], "band {band} of mark {}", row.mark);
            }
        }
        // Band 9 is nobody's.
        for y in 18..TRIANGLE_SIZE[1] {
            for x in 0..TRIANGLE_SIZE[0] {
                assert_eq!(
                    texel(&bytes, x, y, TRIANGLE_SIZE),
                    CLEAR,
                    "({x}, {y}) is outside every band and is not the clear"
                );
            }
        }
        // A list and a strip over the same quad differ in which vertex provokes each triangle: the
        // list's are its 0th and 3rd, the strip's its 0th and 1st. Read as the **gap** between the
        // two vids a band holds, which is 3 for a list and 1 for a strip.
        for row in &rows {
            let band = *bands[&row.mark].iter().next().unwrap();
            let mut vids: BTreeSet<u32> = BTreeSet::new();
            for y in 2 * band..2 * band + 2 {
                for x in 0..TRIANGLE_SIZE[0] {
                    vids.insert(texel(&bytes, x, y, TRIANGLE_SIZE).vid);
                }
            }
            let mut held = vids.into_iter();
            let (low, high) = (held.next().unwrap(), held.next().unwrap());
            let want = match row.primitive {
                PrimitiveType::Triangle => 3,
                _ => 1,
            };
            assert_eq!(
                high - low,
                want,
                "mark {} is a {:?} and its band's provoking vertices are {low} and {high}",
                row.mark,
                row.primitive
            );
        }
    }

    /// **Metal's recorded lines target holds each draw's segments on its own row**, and the bands
    /// nobody drew hold the clear.
    ///
    /// What a line covers along its own row is a rasterization rule rather than arithmetic, so what
    /// this holds is what any rule implies: a covered pixel lies on its draw's row and between its
    /// segment's endpoints, each segment covers at least one pixel, and no pixel outside a draw's
    /// band was touched. The covered counts are printed rather than asserted, because asserting a
    /// number here would be adopting whatever this run measured as the rule.
    #[test]
    fn metal_drew_each_line_form_on_its_own_row() {
        let Some(bytes) = recorded(Lines.name(), LINE_SIZE) else {
            println!("render-draw-forms-lines: no recording in the tree yet");
            return;
        };
        let positions = line_positions();
        let mut covered: BTreeMap<u32, BTreeSet<(u32, u32)>> = BTreeMap::new();
        for y in 0..LINE_SIZE[1] {
            for x in 0..LINE_SIZE[0] {
                let got = texel(&bytes, x, y, LINE_SIZE);
                if got == CLEAR {
                    continue;
                }
                covered.entry(got.mark).or_default().insert((x, y));
            }
        }
        for row in line_rows() {
            let pixels = covered
                .get(&row.mark)
                .unwrap_or_else(|| panic!("mark {} covered nothing", row.mark));
            // Every vertex the draw reads, by row and column, so the claim is over the draw's own
            // geometry and not over a band index.
            let span: Vec<[f32; 2]> = vertices(&row.form)
                .iter()
                .map(|vid| positions[*vid as usize])
                .collect();
            let instances = instances(&row.form);
            let rows: BTreeSet<u32> = (0..instances)
                .flat_map(|iid| {
                    let down = if row.shift == 0 { 0 } else { row.shift * iid };
                    span.iter().map(move |at| at[1] as u32 + down)
                })
                .collect();
            let columns: BTreeSet<u32> = span.iter().map(|at| at[0] as u32).collect();
            let (low, high) = (
                *columns.iter().next().unwrap(),
                *columns.iter().next_back().unwrap(),
            );
            for (x, y) in pixels {
                assert!(
                    rows.contains(y),
                    "mark {} covered ({x}, {y}), which is on no row of its own",
                    row.mark
                );
                assert!(
                    *x >= low && *x <= high,
                    "mark {} covered ({x}, {y}), outside its columns {low} to {high}",
                    row.mark
                );
            }
            // Each instance of the draw covered something, which is what says the instance count
            // reached a line draw.
            for y in &rows {
                assert!(
                    pixels.iter().any(|(_, at)| at == y),
                    "mark {} covered nothing on row {y}",
                    row.mark
                );
            }
            let counts: Vec<usize> = rows
                .iter()
                .map(|y| pixels.iter().filter(|(_, at)| at == y).count())
                .collect();
            println!(
                "LINES: mark {} ({:?}) covered {counts:?} pixel(s) on rows {rows:?}",
                row.mark, row.primitive
            );
        }
        // The bands nobody drew hold the clear.
        for y in 10..LINE_SIZE[1] {
            for x in 0..LINE_SIZE[0] {
                assert_eq!(
                    texel(&bytes, x, y, LINE_SIZE),
                    CLEAR,
                    "({x}, {y}) is outside every band and is not the clear"
                );
            }
        }
    }

    /// **Metal's recorded empty-draw target holds the two rows that write and nothing else.**
    ///
    /// The two controls are what makes this readable: a target of all clear would pass with a pass
    /// that never ran, and this one fails unless rows 0 and 15 hold their draws' own pixels.
    #[test]
    fn metal_drew_nothing_for_every_empty_form() {
        let Some(bytes) = recorded(Empty.name(), EMPTY_SIZE) else {
            println!("render-draw-forms-empty: no recording in the tree yet");
            return;
        };
        let rows = empty_rows();
        let writing: Vec<&Row> = rows
            .iter()
            .filter(|row| !vertices(&row.form).is_empty())
            .collect();
        assert_eq!(writing.len(), 2, "the case holds two draws that write");
        let positions = point_positions();
        let mut want: BTreeMap<(u32, u32), Texel> = BTreeMap::new();
        for row in &writing {
            for vid in vertices(&row.form) {
                let at = positions[vid as usize];
                want.insert(
                    (at[0] as u32, at[1] as u32),
                    Texel {
                        mark: row.mark,
                        vid,
                        iid: 0,
                        base: 0,
                    },
                );
            }
        }
        // The two controls are on the first and the last row, so an empty draw that wrote anywhere
        // between them is caught.
        assert_eq!(
            want.keys().map(|(_, y)| *y).collect::<BTreeSet<u32>>(),
            BTreeSet::from([0, EMPTY_SIZE[1] - 1])
        );
        let mut differ = Vec::new();
        for y in 0..EMPTY_SIZE[1] {
            for x in 0..EMPTY_SIZE[0] {
                let got = texel(&bytes, x, y, EMPTY_SIZE);
                let expected = want.get(&(x, y)).copied().unwrap_or(CLEAR);
                if got != expected {
                    differ.push(format!("({x}, {y}): Metal {got:?}, want {expected:?}"));
                }
            }
        }
        for line in differ.iter().take(20) {
            println!("EMPTY-DIFFERS: {line}");
        }
        assert!(
            differ.is_empty(),
            "{} texel(s) are not what two writing draws and eight empty ones say",
            differ.len()
        );
    }

    /// **The family reaches every value of `MTLPrimitiveType` and both of `MTLIndexType`**, which
    /// is the whole of what it set out to sweep, and the ratio is stated here rather than in prose.
    #[test]
    fn the_family_sweeps_both_enumerations_whole() {
        let mut primitives = BTreeSet::new();
        let mut index_types = BTreeSet::new();
        let mut calls = BTreeSet::new();
        for rows in [point_rows(), triangle_rows(), line_rows(), empty_rows()] {
            for row in rows {
                primitives.insert(row.primitive.as_raw());
                let step = row.form.step(row.primitive, Handle(0), Handle(1), 0);
                calls.insert(step.call());
                match &row.form {
                    Form::IndexedPlain { index_type, .. }
                    | Form::IndexedInstanced { index_type, .. }
                    | Form::IndexedBased { index_type, .. }
                    | Form::IndexedIndirect { index_type, .. } => {
                        index_types.insert(index_type.as_raw());
                    }
                    _ => {}
                }
            }
        }
        // Five of five: Point, Line, LineStrip, Triangle and TriangleStrip.
        assert_eq!(primitives, BTreeSet::from([0, 1, 2, 3, 4]));
        // Two of two: UInt16 and UInt32.
        assert_eq!(index_types, BTreeSet::from([0, 1]));
        // Eight of eight non-patch, non-mesh draw selectors.
        assert_eq!(calls.len(), 8, "the family reaches {calls:?}");
    }

    /// Whether `case` makes **no** indexed indirect draw, read off its own table rather than off
    /// its name: a case that grew one would start announcing the bounds pass, and this is what
    /// makes the declaration follow the draws.
    fn line_rows_hold_no_indexed_indirect(case: &str) -> bool {
        let rows = match case {
            "render-draw-forms-points" => point_rows(),
            "render-draw-forms-triangles" => triangle_rows(),
            "render-draw-forms-lines" => line_rows(),
            "render-draw-forms-empty" => empty_rows(),
            other => panic!("{other} is not a case of this family"),
        };
        !rows
            .iter()
            .any(|row| matches!(row.form, Form::IndexedIndirect { .. }))
    }

    /// **Every case of the family names itself once**, and the names are the ones the recordings
    /// are keyed by.
    #[test]
    fn the_family_names_four_cases_apart() {
        let names: Vec<&str> = all().iter().map(|case| case.name()).collect();
        let distinct: BTreeSet<&str> = names.iter().copied().collect();
        assert_eq!(distinct.len(), names.len(), "two cases share a name");
        assert_eq!(names.len(), 4);
        for case in all() {
            assert_eq!(case.record_runs(), RUNS);
            assert!(matches!(case.metal_source(), MetalSource::Run));
            // No case loosens a value: every channel it reads is a `uint` the fragment function
            // returned, so every allowance any case names sits over the notices site.
            //
            // Every case declares the family's own list ([`PASS_NOTICES`]). On top of it, the
            // notice an **indexed indirect** draw provokes is declared by exactly the cases that
            // make one -- which is the falsification of that declaration rather than a
            // restatement of it, because the lines case reaches the same code with the direct
            // indexed form and announces nothing.
            let declared = case.allowances();
            for allowance in &declared {
                assert!(allowance.path.starts_with("case/notices"), "{allowance:?}");
                assert_eq!(allowance.loosening, Loosening::Unchecked, "{allowance:?}");
            }
            let bounds_pass =
                Justification::Degradation(Degradation::IndirectIndexBoundsPass.code().to_owned());
            let names_bounds_pass = declared
                .iter()
                .any(|allowance| allowance.justified_by == bounds_pass);
            let indirect = line_rows_hold_no_indexed_indirect(case.name());
            assert_eq!(
                names_bounds_pass,
                !indirect,
                "{} and its bounds-pass declaration disagree",
                case.name()
            );
            assert_eq!(
                declared.len(),
                // The family's own list, the notice a device's own memory model provokes, and the
                // bounds-pass notice where the draw is an indexed indirect one.
                PASS_NOTICES.len() + 1 + usize::from(!indirect),
                "{} declares the wrong number of notices",
                case.name()
            );
            assert!(case.inputs().is_ok());
            assert!(case.shaders().is_ok());
        }
    }
}
