//! What an indirect command buffer's **crossings** answer: the descriptor's inheritance crossed
//! with the state a command sets for itself, a reset range crossed with the range an execution
//! runs, a pass's own scissor crossed with the draws a command makes, a resident set's declared
//! usage crossed with what the commands actually do to the resource, and the two patch draws no
//! case had reached.
//!
//! [`super::indirect_commands`] sweeps the calls one at a time: each of its rows encodes a buffer
//! whose descriptor leaves nothing to the parent encoder, executes every command it holds, and
//! reads the words back. That is the per-value axis, and it is blind by construction to every
//! question of the form *which of two states wins*. Three of those questions decide whether an
//! indirect command buffer works at all:
//!
//! - a buffer that **inherits** a pipeline state or its buffers still lets a command set them, and
//!   the execution has to bind the right one;
//! - `resetWithRange:` names a range and `executeCommandsInBuffer:withRange:` names another, and a
//!   command is blank only where the first covers it and run only where the second does;
//! - the parent encoder's own render state — a scissor, a viewport — is what an indirect draw
//!   rasterizes under, because a render command may not set one
//!   ([`crate::indirect_commands`]: the setters end Metal's process on this oracle).
//!
//! Each row below is a crossing rather than a value, so each states **two** settings and reads an
//! answer that only their pair can produce. A row whose cells all answer the same words would be
//! reporting that the two settings are independent, and this module says where that happened
//! rather than leaving the row in: the one crossing measured independent is named in
//! [`RESIDENCY_USAGE`]'s own documentation.
//!
//! | Row | The pair it crosses |
//! |---|---|
//! | [`PATCH_DRAWS`] | the two patch-draw command types, each with its own factors, patch start and instance count |
//! | [`INHERITED_STATE`] | `inheritBuffers` and `inheritPipelineState` against the bindings and the pipeline a command sets for itself |
//! | [`RESET_RANGES`] | `resetWithRange:` against the range an execution runs, and against a command re-encoded after the reset |
//! | [`PASS_SCISSOR`] | the pass's own `setScissorRect:` against the two halves an indirect draw rasterizes |
//! | [`RESIDENCY_USAGE`] | `MTLResourceUsage` as declared against what the commands read and write |
//!
//! # Why none of this can hang or fault
//!
//! Every row is bounded and small: at most six commands a buffer, the largest grid four threads,
//! the largest draw six vertices of two instances, the largest patch draw two patches of one
//! control point at a factor of two, and the one rasterized target 8 by 8. No fixture holds a
//! loop. The three shapes [`super::indirect_commands`] names as dangerous are avoided the same way
//! here, and this module adds two of its own, both on a measurement rather than on caution:
//!
//! - **A command that sets a pipeline on a buffer that inherits one** raises `Invalid` in the
//!   runtime and **ends plain Metal's process** (`metal2vulkan_runtime`'s
//!   `IndirectComputeCommand::set_compute_pipeline_state`, measured). So no row crosses
//!   `inheritPipelineState` with a command's own `setRenderPipelineState:`: the cell cannot be
//!   recorded at all, and [`INHERITED_STATE`] names it in its own documentation instead.
//! - **An inherited pipeline state made without `supportIndirectCommandBuffers`** faults the GPU
//!   at the executing encoder, so the parent pipeline [`INHERITED_STATE`] binds is made with the
//!   flag exactly as the commands' own are.
//!
//! [`BIND_LIMITS`] sets a buffer at an index at and past its descriptor's maximum, which is the
//! one row here whose answer is a **refusal of ours**: plain Metal keeps such a binding for its
//! own getters (`metal2vulkan_runtime`'s `IndirectCommandBuffer::set_buffer`, measured on an
//! M3 Max) where we raise `Invalid`. That row **executes nothing**, so no over-long binding ever
//! reaches a GPU, and the gap it holds visible is listed in `known-gaps.tsv`.

use crate::allowance::Allowance;
use crate::case::STAGE_VERTEX;
use crate::case::{call, made, Abort, Case, Failure, Handle, Input, MetalSource, Side};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::indirect_commands::{
    ComputeCommandStep, EncodedCommand, IcbDescriptor, RenderCommandStep,
};
use crate::recording::ShaderRef;
use crate::render::{
    ColorState, PassColor, RenderPass, RenderPipelineSpec, RenderStep, RenderWork, ScissorBox,
    StageFunction, TargetSpec, TessellationState, VertexLayout,
};
use crate::transcript::{
    Aspect, ByteLayout, Digest, Element, Gaps, Node, Observed, Region, Sites, Value,
};
use metal2vulkan::metal::{
    IndirectCommandType, PixelFormat, PrimitiveType, TessellationControlPointIndexType,
    TessellationPartitionMode,
};

/// The byte every output buffer is filled with before a row runs, so a word no command wrote reads
/// as this rather than as zero: which is what lets a row say that a reset command, a command
/// outside the executed range, or a command whose bindings the execution replaced wrote nothing.
const UNWRITTEN: u8 = 0xee;

/// The parameter pairs a command binds, one pair a command, each at its own 16-byte offset so a
/// command that bound the wrong offset writes another command's words.
const PARAMS: [[u32; 2]; 6] = [
    [3, 100],
    [5, 200],
    [7, 300],
    [11, 400],
    [13, 500],
    [17, 600],
];

/// Bytes between one command's parameter pair and the next.
const PARAM_STRIDE: u64 = 16;

/// The parameter buffer's bytes: [`PARAMS`] at [`PARAM_STRIDE`].
fn params_bytes() -> Vec<u8> {
    let mut out = vec![0u8; PARAMS.len() * PARAM_STRIDE as usize];
    for (at, pair) in PARAMS.iter().enumerate() {
        let start = at * PARAM_STRIDE as usize;
        for (word, value) in pair.iter().enumerate() {
            out[start + word * 4..start + word * 4 + 4].copy_from_slice(&value.to_le_bytes());
        }
    }
    out
}

/// The layout a row's output buffer is read under: the whole buffer as 32-bit words, so a diff
/// names the word that moved.
fn buffer_layout(length: u64) -> ByteLayout {
    ByteLayout::buffer(0, length, Element::U32)
}

/// The sites one buffer's encoding emits: a branch a command, holding what each of its steps
/// answered, named by the step's own Metal selector so a refusal reads as the call we refused.
fn encoding_node(
    name: &str,
    commands: &[EncodedCommand],
    refused: &[Vec<Option<Failure>>],
) -> Node {
    let mut node = Node::branch(name, Vec::new());
    for (at, command) in commands.iter().enumerate() {
        let mut steps = Node::branch(format!("command/{at}"), Vec::new());
        let answers = refused.get(at).map(Vec::as_slice).unwrap_or_default();
        for (step, answer) in answers.iter().enumerate() {
            steps = steps.with(Node::leaf(
                format!("{step}-{}", command.step_call(step)),
                Observed::value(match answer {
                    Some(failure) => failure.recorded(),
                    None => Value::ok(Value::Absent),
                }),
            ));
        }
        node = node.with(steps);
    }
    node
}

/// The first refusal `refused` holds, as the site it happened at and the failure.
fn first_refusal(
    at_name: &str,
    commands: &[EncodedCommand],
    refused: &[Vec<Option<Failure>>],
) -> Option<(String, Failure)> {
    for (at, answers) in refused.iter().enumerate() {
        for (step, answer) in answers.iter().enumerate() {
            if let Some(failure) = answer {
                let name = commands.get(at).map_or("call", |c| c.step_call(step));
                return Some((
                    format!("{at_name}/command/{at}/{step}-{name}"),
                    failure.clone(),
                ));
            }
        }
    }
    None
}

/// The sites one `Side::render` program's answer emits.
fn program_node(work: &[RenderWork], outcome: &crate::case::PassOutcome) -> Node {
    let mut calls = Node::branch("calls", Vec::new());
    for (at, refused) in outcome.refused.iter().enumerate() {
        let name = work.get(at).map_or("call", RenderWork::call);
        calls = calls.with(Node::leaf(
            format!("{at}-{name}"),
            Observed::value(match refused {
                Some(failure) => failure.recorded(),
                None => Value::ok(Value::Absent),
            }),
        ));
    }
    Node::branch("program", Vec::new())
        .with(calls)
        .with(Node::leaf(
            "status",
            Observed::value(Value::ok(outcome.status.clone())),
        ))
        .with(Node::leaf(
            "command-buffer-error",
            Observed::value(match &outcome.error {
                Some(error) => Value::Variant {
                    name: "Some".to_owned(),
                    payload: Box::new(error.clone()),
                },
                None => Value::unit_variant("None"),
            }),
        ))
}

/// The first piece of a program that refused, as the site and the failure.
fn first_refused_piece(
    work: &[RenderWork],
    outcome: &crate::case::PassOutcome,
) -> Option<(String, Failure)> {
    outcome
        .refused
        .iter()
        .enumerate()
        .find_map(|(at, refused)| refused.as_ref().map(|failure| (at, failure.clone())))
        .map(|(at, failure)| {
            let name = work.get(at).map_or("call", RenderWork::call);
            (format!("program/calls/{at}-{name}"), failure)
        })
}

/// Both committed texts of `fixture`, as a case's shader references.
fn shaders_of(fixture: &'static str, entry: &str) -> Result<Vec<ShaderRef>, String> {
    let held = Fixture::new(fixture);
    [RUNTIME_TEXT, METAL_TEXT]
        .into_iter()
        .map(|extension| {
            Ok(ShaderRef {
                fixture: format!("{fixture}{extension}"),
                entry: entry.to_owned(),
                source_sha256: crate::sha256_hex(held.text(extension)?.as_bytes()),
            })
        })
        .collect()
}

/// Both committed texts of `fixture`, as a case's inputs.
fn fixture_inputs(fixture: &'static str) -> Result<Vec<Input>, String> {
    let held = Fixture::new(fixture);
    Ok(vec![
        Input {
            name: "air".to_owned(),
            digest: Digest::sha256(held.text(RUNTIME_TEXT)?.as_bytes()),
        },
        Input {
            name: "air-typed".to_owned(),
            digest: Digest::sha256(held.text(METAL_TEXT)?.as_bytes()),
        },
    ])
}

/// A descriptor's and its commands' own words, as a case's inputs, under `name` so a row that
/// states several buffers feeds each of them to its fingerprint.
fn commands_inputs(
    name: &str,
    descriptor: &IcbDescriptor,
    commands: &[EncodedCommand],
) -> Vec<Input> {
    let mut words = Vec::new();
    for command in commands {
        words.extend(command.words());
        words.push(b'\n');
    }
    vec![
        Input::bytes(format!("{name}:descriptor"), &descriptor.words()),
        Input::bytes(format!("{name}:commands"), &words),
    ]
}

/// Read `length` bytes of `buffer` from `offset` and emit them at `at` as 32-bit words.
fn read_words(
    side: &mut dyn Side,
    sites: &mut Sites,
    at: &str,
    buffer: Handle,
    offset: u64,
    length: u64,
) -> Result<(), Abort> {
    let read = side
        .read_buffer(buffer, offset, length as usize)
        .map_err(|failure| Abort {
            at: at.to_owned(),
            failure,
        })?;
    sites.push(Node::leaf(at, Observed::bytes(read, buffer_layout(length))));
    Ok(())
}

// --------------------------------------------------------------------------------------------
// The patch draws

/// The name of the row whose commands draw patches, the two `MTLIndirectCommandType` values no
/// case had put to Metal.
pub const PATCH_DRAWS: &str = "indirect-render-commands-patch-draws";

/// The owned fixture the patch commands draw: a post-tessellation vertex function of triangle
/// patches of **one** control point and no stage input, which marks every domain point it is
/// called with at that point's cell of its own patch's and instance's 5-by-5 grid.
///
/// One control point and no stage input is what makes the row cheap and exact: the pipeline needs
/// no vertex attribute, so nothing but the patch draw itself decides what the marks hold, and a
/// mark is idempotent so the answer is the tessellator's point set whatever the invocation count
/// and order.
const PATCH_FIXTURE: &str = "icb_render_patch_template";

/// Cells one grid of the fixture's marks holds: a 5-by-5 grid.
const GRID_CELLS: u64 = 25;

/// Grids the fixture's marks hold: `min(instance, 7) * 8 + min(patch, 7)`, so 64.
const GRIDS: u64 = 64;

/// Bytes the whole marks buffer takes.
const MARKS_BYTES: u64 = GRIDS * GRID_CELLS * 4;

/// Words of the marks buffer a row reads back: the first two instances' first two patches, which
/// is every grid either command reaches. Grid `i * 8 + p` lies at word `(i * 8 + p) * 25`, so
/// reading the first 10 grids covers instances 0 and 1 of patches 0 and 1 and nothing a command
/// could reach is left out.
const MARKS_READ_GRIDS: u64 = 10;

/// The patch count the plain patch draw states, and the one the indexed draw states.
const PLAIN_PATCHES: u32 = 2;

/// The maximum tessellation factor the row's pipelines clamp to. 4 is above every factor the row
/// states, so no factor of the row is clamped and each one's level is its own number.
const PATCH_MAX_FACTOR: u32 = 4;

/// The two factor records the row's commands read, as `half` edge and inside factors: a record of
/// all-ones, whose level is 1 and marks a triangle patch's three corners (cells 0, 4 and 20), and
/// a record of all-twos, whose level is 2 and marks those three, the three edge midpoints (cells
/// 2, 10 and 12) and the centre (cell 6).
///
/// Two levels rather than one is what makes the two commands' answers differ in **shape** and not
/// only in place: a command that read the other's factor record marks three cells where seven
/// belong, or seven where three do.
const PATCH_FACTORS: [[f32; 4]; 2] = [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]];

/// Bytes one `half` factor record of a triangle patch takes: three edges and one inside factor.
const FACTOR_BYTES: u64 = 8;

/// The factor buffer's bytes: [`PATCH_FACTORS`] as `half` records, one after the other.
fn patch_factor_bytes() -> Vec<u8> {
    let mut bytes = Vec::new();
    for record in PATCH_FACTORS {
        for factor in record {
            bytes.extend(half_bits(factor).to_le_bytes());
        }
    }
    bytes
}

/// `value` as the bits of an IEEE `half`, which is the one factor format Metal has
/// (`MTLTessellationFactorFormatHalf`). Only the row's own small whole numbers reach it.
fn half_bits(value: f32) -> u16 {
    assert!(
        value.is_finite() && value > 0.0,
        "a factor of this row is finite and positive: {value}"
    );
    let bits = value.to_bits();
    let exponent = ((bits >> 23) & 0xff) as i32 - 127;
    let mantissa = bits & 0x007f_ffff;
    assert!(
        (-14..=15).contains(&exponent),
        "a factor of this row is a normal half: {value}"
    );
    assert_eq!(
        mantissa & 0x0000_1fff,
        0,
        "a factor of this row is exact as a half: {value}"
    );
    (((exponent + 15) as u16) << 10) | ((mantissa >> 13) as u16)
}

/// The control-point indices the indexed patch draw reads: one `uint16` a control point, the
/// identity, which is what the fixture's one control point per patch leaves it.
///
/// The fixture declares no `patch_control_point`, so **no index of this buffer reaches the
/// function**, and the row says so rather than pretending the permutation matters: what it asks of
/// `MTLIndirectCommandTypeDrawIndexedPatches` is that the command encodes, that Metal records it as
/// an indexed patch draw, and that the execution draws its patches and instances. Which control
/// point an index names is [`super::tessellation_sweeps`]'s question, and that family answers it
/// over a fixture with a stage input.
fn control_point_index_bytes() -> Vec<u8> {
    let mut bytes = Vec::new();
    for index in 0u16..PLAIN_PATCHES as u16 + 2 {
        bytes.extend(index.to_le_bytes());
    }
    bytes
}

/// The row's descriptor: both patch-draw types, one vertex buffer (the fixture's `marks`), and
/// nothing inherited, so each command carries its own pipeline, its own binding, its own factors
/// and its own patch range.
fn patch_descriptor() -> IcbDescriptor {
    IcbDescriptor {
        command_types: IndirectCommandType::DRAW_PATCHES
            .union(IndirectCommandType::DRAW_INDEXED_PATCHES),
        max_vertex_buffer_bind_count: 1,
        ..IcbDescriptor::new()
    }
}

/// The pipeline a patch command holds, under `index_type`: the fixture's post-tessellation vertex
/// function, integer partitioning, rasterization disabled, and `supportIndirectCommandBuffers`.
fn patch_pipeline(index_type: TessellationControlPointIndexType) -> RenderPipelineSpec<'static> {
    RenderPipelineSpec::tessellating(
        StageFunction::named(PATCH_FIXTURE),
        VertexLayout::new(&[], &[]),
        TessellationState::new()
            .partitioned(TessellationPartitionMode::Integer)
            .clamped_to(PATCH_MAX_FACTOR)
            .indexed_by(index_type),
    )
    .for_indirect_commands()
}

/// The row that draws patches from an indirect command buffer.
///
/// # What the two command types answer
///
/// Command 0 is `MTLIndirectCommandTypeDrawPatches`: two patches from patch 0, one instance, the
/// level-1 factor record. Command 1 is `MTLIndirectCommandTypeDrawIndexedPatches`: one patch from
/// patch **1**, **two** instances, the level-2 record and a control-point index buffer. So the two
/// differ in the command type, the patch start, the patch count, the instance count and the
/// factors at once, and their marks lie in grids no other command of the row reaches: command 0
/// marks grids 0 and 1 (instance 0, patches 0 and 1) and command 1 marks grids 1 and 9 (instance
/// 0 and 1, patch 1).
///
/// Grid 1 is reached by **both**, on purpose and without ambiguity: a mark is idempotent and
/// command 0's level-1 cells are a subset of command 1's level-2 cells there, so grid 1 holds the
/// union either way the GPU orders the two. Grid 0 is command 0's alone and grid 9 is command 1's
/// alone, which is what separates them: a command that drew the other's patch range marks a grid
/// the other owns, and a command that read the other's factors marks the wrong count of cells in
/// its own.
pub struct PatchDraws;

impl PatchDraws {
    /// The commands the row encodes.
    fn commands(
        &self,
        plain_pipeline: Handle,
        indexed_pipeline: Handle,
        marks: Handle,
        factors: Handle,
        control_points: Handle,
    ) -> Vec<EncodedCommand> {
        vec![
            EncodedCommand::Render(vec![
                RenderCommandStep::Pipeline(plain_pipeline),
                RenderCommandStep::VertexBuffer {
                    buffer: marks,
                    offset: 0,
                    index: 0,
                },
                RenderCommandStep::DrawPatches {
                    control_points: 1,
                    first: 0,
                    count: PLAIN_PATCHES,
                    patch_indices: None,
                    patch_index_offset: 0,
                    instances: 1,
                    base_instance: 0,
                    factors,
                    factor_offset: 0,
                    factor_instance_stride: 0,
                },
            ]),
            EncodedCommand::Render(vec![
                RenderCommandStep::Pipeline(indexed_pipeline),
                RenderCommandStep::VertexBuffer {
                    buffer: marks,
                    offset: 0,
                    index: 0,
                },
                RenderCommandStep::DrawIndexedPatches {
                    control_points: 1,
                    first: 1,
                    count: 1,
                    patch_indices: None,
                    patch_index_offset: 0,
                    control_point_indices: control_points,
                    control_point_index_offset: 0,
                    instances: 2,
                    base_instance: 0,
                    factors,
                    factor_offset: FACTOR_BYTES,
                    factor_instance_stride: 0,
                },
            ]),
        ]
    }
}

impl Case for PatchDraws {
    fn name(&self) -> &'static str {
        PATCH_DRAWS
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        tessellation_notices()
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        shaders_of(PATCH_FIXTURE, PATCH_FIXTURE)
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let mut inputs = fixture_inputs(PATCH_FIXTURE)?;
        let commands = self.commands(Handle(0), Handle(1), Handle(2), Handle(3), Handle(4));
        inputs.extend(commands_inputs("icb", &patch_descriptor(), &commands));
        inputs.push(Input::bytes(
            "pipeline:plain",
            &patch_pipeline(TessellationControlPointIndexType::None).words(),
        ));
        inputs.push(Input::bytes(
            "pipeline:indexed",
            &patch_pipeline(TessellationControlPointIndexType::UInt16).words(),
        ));
        inputs.push(Input::bytes("buffer:factors", &patch_factor_bytes()));
        inputs.push(Input::bytes(
            "buffer:control-point-indices",
            &control_point_index_bytes(),
        ));
        inputs.push(Input::bytes(
            "buffer:marks",
            &vec![0u8; MARKS_BYTES as usize],
        ));
        inputs.push(Input::bytes("pass", b"no attachment, 1x1 extent"));
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let fixture = Fixture::new(PATCH_FIXTURE);
        let loaded = fixture.loaded().map_err(|why| Abort {
            at: "library".to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        })?;
        let library = made(sites, "library", side.library(&loaded.source()))?;
        let plain = made(
            sites,
            "pipeline:plain",
            side.render_pipeline(
                library,
                &patch_pipeline(TessellationControlPointIndexType::None),
            ),
        )?;
        let indexed = made(
            sites,
            "pipeline:indexed",
            side.render_pipeline(
                library,
                &patch_pipeline(TessellationControlPointIndexType::UInt16),
            ),
        )?;
        let factors = made(sites, "buffer:factors", side.buffer(&patch_factor_bytes()))?;
        let control_points = made(
            sites,
            "buffer:control-point-indices",
            side.buffer(&control_point_index_bytes()),
        )?;
        let marks = made(
            sites,
            "buffer:marks",
            side.buffer(&vec![0u8; MARKS_BYTES as usize]),
        )?;

        let commands = self.commands(plain, indexed, marks, factors, control_points);
        let descriptor = patch_descriptor();
        let encoded = made(
            sites,
            "command-buffer",
            side.indirect_command_buffer(&descriptor, &commands),
        )?;
        sites.push(encoding_node("encoded", &commands, &encoded.refused));
        if let Some((at, failure)) = first_refusal("encoded", &commands, &encoded.refused) {
            return Err(Abort { at, failure });
        }
        call(
            sites,
            "size",
            side.indirect_commands_size(encoded.commands),
            |size| Value::Uint(*size),
        )?;

        let steps = vec![
            RenderStep::UseBuffer {
                buffer: factors,
                write: false,
                stages: STAGE_VERTEX,
            },
            RenderStep::UseBuffer {
                buffer: control_points,
                write: false,
                stages: STAGE_VERTEX,
            },
            RenderStep::UseBuffer {
                buffer: marks,
                write: true,
                stages: STAGE_VERTEX,
            },
            RenderStep::ExecuteCommands {
                commands: encoded.commands,
                first: 0,
                count: commands.len() as u64,
            },
        ];
        let work = vec![RenderWork::Pass(RenderPass::sized([1, 1]).drawing(steps))];
        let outcome = call(sites, "render", side.render(&work), |outcome| {
            outcome.status.clone()
        })?;
        sites.push(program_node(&work, &outcome));
        if let Some((at, failure)) = first_refused_piece(&work, &outcome) {
            return Err(Abort { at, failure });
        }
        read_words(
            side,
            sites,
            "marks",
            marks,
            0,
            MARKS_READ_GRIDS * GRID_CELLS * 4,
        )
    }
}

/// The notices a tessellating pipeline of ours announces, which every row that draws a patch
/// names.
///
/// Taken from [`super::tessellation_sweeps`], whose own documentation says why each is named:
/// `tessellation-winding` and `tessellation-factor-edge-cases` are announced unconditionally by
/// the control stage, and `tessellation-triangle-order` and `emulated-tessellation-draws` by the
/// venues whose tessellator is not Apple's or which have no `tessellationShader` at all. None can
/// move a mark, because a mark is a set member and not a position in a triangle stream; they are
/// named because a notice our run announces that no case names is a divergence.
fn tessellation_notices() -> Vec<Allowance> {
    [
        metal2vulkan_runtime::Degradation::TessellationWinding,
        metal2vulkan_runtime::Degradation::TessellationFactorEdgeCases,
        metal2vulkan_runtime::Degradation::TessellationTriangleOrder,
        metal2vulkan_runtime::Degradation::EmulatedTessellationDraws,
    ]
    .into_iter()
    .map(|notice| crate::replay::notice_allowance(notice.code()))
    .collect()
}

// --------------------------------------------------------------------------------------------
// Inheritance crossed with the state a command sets for itself

/// The name of the row that crosses `inheritBuffers` and `inheritPipelineState` with the bindings
/// and the pipeline a command sets for itself.
pub const INHERITED_STATE: &str = "indirect-render-commands-inherited-state";

/// The owned fixture whose rasterization-disabled vertex function writes
/// `params[0] * (vid + 1) + params[1] + 1000 * iid` to `out[16 * iid + vid]`, with `params` at
/// vertex buffer 0 and `out` at vertex buffer 1. A binding that came from the wrong side reads as
/// another segment's words, or as none at all.
const RENDER_FIXTURE: &str = "icb_render_replay_template";

/// Words one segment of the inheritance row's output buffer holds: 16, because the fixture indexes
/// `out[16 * iid + vid]`, so one instance's whole row of slots lies inside one segment and a draw
/// of the instance after it cannot leave the segment it was bound at.
const SEGMENT_WORDS: u64 = 16;

/// Bytes one segment takes.
const SEGMENT_BYTES: u64 = SEGMENT_WORDS * 4;

/// Segments the inheritance row's output buffer holds: six the commands bind for themselves, one
/// the parent encoder binds, and a tail no draw of the row can reach.
const INHERIT_SEGMENTS: u64 = 8;

/// Bytes the inheritance row's output buffer takes.
const INHERIT_BYTES: u64 = INHERIT_SEGMENTS * SEGMENT_BYTES;

/// The segment the parent encoder binds, which every command of an inheriting buffer writes.
const PARENT_SEGMENT: u64 = 6;

/// The parameter pair the parent encoder binds, whose words no command binds for itself, so a word
/// written from the parent's `params` cannot be mistaken for one written from a command's.
const PARENT_PARAMS: u64 = 3;

/// Which of the three buffers the row states a leg is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Inheritance {
    /// `inheritPipelineState` and `inheritBuffers` both off: every command's own pipeline and own
    /// bindings, which is the control the other two are read against.
    Own,
    /// `inheritBuffers` on, with each command still setting the bindings it would have used.
    Buffers,
    /// `inheritPipelineState` on, so no command sets a pipeline and each still binds its own
    /// buffers.
    Pipeline,
}

impl Inheritance {
    /// The leg's own site name.
    fn site(self) -> &'static str {
        match self {
            Inheritance::Own => "own-state",
            Inheritance::Buffers => "inherit-buffers",
            Inheritance::Pipeline => "inherit-pipeline-state",
        }
    }

    /// The leg's descriptor: draws of primitives, two vertex buffers, and the one field the leg is
    /// about.
    fn descriptor(self) -> IcbDescriptor {
        IcbDescriptor {
            command_types: IndirectCommandType::DRAW,
            max_vertex_buffer_bind_count: 2,
            inherit_buffers: self == Inheritance::Buffers,
            inherit_pipeline_state: self == Inheritance::Pipeline,
            ..IcbDescriptor::new()
        }
    }

    /// The two segments the leg's commands bind for themselves.
    fn segments(self) -> [u64; 2] {
        match self {
            Inheritance::Own => [0, 1],
            Inheritance::Buffers => [2, 3],
            Inheritance::Pipeline => [4, 5],
        }
    }
}

/// The two draws every leg's commands make, in order, each writing a different range of slots of
/// whichever segment the execution bound.
///
/// Three points from vertex 0 and two from vertex **4**: so one command writes slots 0 to 2 and the
/// other slots 4 and 5, and the two are still told apart where they write the **same** segment,
/// which is what an inheriting buffer's execution makes them do. Without that the inheriting leg's
/// two commands would be one answer written twice.
const INHERIT_DRAWS: [RenderCommandStep; 2] = [
    RenderCommandStep::Draw {
        primitive: PrimitiveType::Point,
        first: 0,
        count: 3,
        instances: 1,
        base_instance: 0,
    },
    RenderCommandStep::Draw {
        primitive: PrimitiveType::Point,
        first: 4,
        count: 2,
        instances: 1,
        base_instance: 0,
    },
];

/// The pipeline every command of the row holds and the parent encoder binds: the
/// rasterization-disabled vertex function alone, with `supportIndirectCommandBuffers` — which the
/// **parent's** must carry too, because an inherited state made without it faults Metal's GPU at
/// the executing encoder.
fn inherit_pipeline_spec() -> RenderPipelineSpec<'static> {
    let mut spec = RenderPipelineSpec::of(StageFunction::named(RENDER_FIXTURE), None, &[]);
    spec.rasterization = false;
    spec.support_indirect_command_buffers = true;
    spec
}

/// The commands one leg encodes.
fn inherit_commands(
    leg: Inheritance,
    pipeline: Handle,
    params: Handle,
    out: Handle,
) -> Vec<EncodedCommand> {
    let segments = leg.segments();
    INHERIT_DRAWS
        .iter()
        .enumerate()
        .map(|(at, draw)| {
            let mut steps = Vec::new();
            // A command of a buffer that inherits its pipeline state must set none: the setter
            // raises `Invalid` in the runtime and ends plain Metal's process, so the cell that
            // crossed the two cannot be recorded and is named in the row's documentation.
            if leg != Inheritance::Pipeline {
                steps.push(RenderCommandStep::Pipeline(pipeline));
            }
            steps.push(RenderCommandStep::VertexBuffer {
                buffer: params,
                offset: at as u64 * PARAM_STRIDE,
                index: 0,
            });
            steps.push(RenderCommandStep::VertexBuffer {
                buffer: out,
                offset: segments[at] * SEGMENT_BYTES,
                index: 1,
            });
            steps.push(draw.clone());
            EncodedCommand::Render(steps)
        })
        .collect()
}

/// The row that crosses the descriptor's inheritance with the state a command sets for itself.
///
/// # The three cells, and the one that cannot be recorded
///
/// Three buffers hold the **same two commands** — same pipeline, same parameter offsets, same
/// draws — and differ only in one field of their descriptors. The parent encoder binds a pipeline,
/// a `params` window no command binds ([`PARENT_PARAMS`]) and the segment no command binds
/// ([`PARENT_SEGMENT`]), and re-binds all three before each execution, because Metal leaves a
/// render encoder's state undefined after an indirect execution.
///
/// - **`own-state`** is the control: nothing inherited. Each command's own segment holds its own
///   draw's words, from its own `params`.
/// - **`inherit-buffers`** keeps the same two `setVertexBuffer:` calls. Metal **records** them —
///   its getters answer them — and the execution binds the *encoder's* buffers instead, so the two
///   commands write the parent's segment and their own two segments stay untouched. That is the
///   whole crossing: the per-value axis sets `inheritBuffers` on a buffer it never executes
///   ([`super::indirect_commands::LEFT_RENDER_STATES`]) and reads nothing downstream of it, and a
///   side that let a command's own binding win would pass every row of that axis.
/// - **`inherit-pipeline-state`** sets no pipeline on either command and binds its own buffers, so
///   its answer is `own-state`'s words in its own two segments — produced by a pipeline that was
///   never written into the buffer. A side that did not take the parent's pipeline at execution
///   would write nothing there.
///
/// The fourth cell of the pair — **a command that sets a pipeline on a buffer that inherits one** —
/// is not here and must not be. The runtime raises `Invalid`
/// (`IndirectCommandBufferProblem::InheritedPipelineState`) and plain Metal **ends its process**,
/// which would end the recorder; the measurement is the runtime's own
/// (`IndirectComputeCommand::set_compute_pipeline_state`'s doc comment, "both end Metal's
/// process").
pub struct InheritedState;

/// Every leg of [`InheritedState`], in the order the row executes them.
const INHERIT_LEGS: [Inheritance; 3] = [
    Inheritance::Own,
    Inheritance::Buffers,
    Inheritance::Pipeline,
];

impl Case for InheritedState {
    fn name(&self) -> &'static str {
        INHERITED_STATE
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        Vec::new()
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        shaders_of(RENDER_FIXTURE, RENDER_FIXTURE)
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let mut inputs = fixture_inputs(RENDER_FIXTURE)?;
        for leg in INHERIT_LEGS {
            let commands = inherit_commands(leg, Handle(0), Handle(1), Handle(2));
            inputs.extend(commands_inputs(leg.site(), &leg.descriptor(), &commands));
        }
        inputs.push(Input::bytes(
            "render-pipeline",
            &inherit_pipeline_spec().words(),
        ));
        inputs.push(Input::bytes("buffer:params", &params_bytes()));
        inputs.push(Input::bytes(
            "buffer:out",
            &vec![UNWRITTEN; INHERIT_BYTES as usize],
        ));
        inputs.push(Input::bytes(
            "parent",
            format!(
                "pipeline, params {PARENT_PARAMS}, out segment {PARENT_SEGMENT}, re-bound before \
                 each execution"
            )
            .as_bytes(),
        ));
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let fixture = Fixture::new(RENDER_FIXTURE);
        let loaded = fixture.loaded().map_err(|why| Abort {
            at: "library".to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        })?;
        let library = made(sites, "library", side.library(&loaded.source()))?;
        let pipeline = made(
            sites,
            "pipeline",
            side.render_pipeline(library, &inherit_pipeline_spec()),
        )?;
        let params = made(sites, "buffer:params", side.buffer(&params_bytes()))?;
        let out = made(
            sites,
            "buffer:out",
            side.buffer(&vec![UNWRITTEN; INHERIT_BYTES as usize]),
        )?;

        let mut buffers = Vec::new();
        for leg in INHERIT_LEGS {
            let commands = inherit_commands(leg, pipeline, params, out);
            let descriptor = leg.descriptor();
            let encoded = made(
                sites,
                &format!("{}/command-buffer", leg.site()),
                side.indirect_command_buffer(&descriptor, &commands),
            )?;
            sites.push(encoding_node(
                &format!("{}/encoded", leg.site()),
                &commands,
                &encoded.refused,
            ));
            if let Some((at, failure)) = first_refusal(
                &format!("{}/encoded", leg.site()),
                &commands,
                &encoded.refused,
            ) {
                return Err(Abort { at, failure });
            }
            call(
                sites,
                &format!("{}/size", leg.site()),
                side.indirect_commands_size(encoded.commands),
                |size| Value::Uint(*size),
            )?;
            buffers.push((leg, encoded.commands, commands.len() as u64));
        }

        let mut steps = vec![
            RenderStep::UseBuffer {
                buffer: params,
                write: false,
                stages: STAGE_VERTEX,
            },
            RenderStep::UseBuffer {
                buffer: out,
                write: true,
                stages: STAGE_VERTEX,
            },
        ];
        for (_, commands, count) in &buffers {
            // Metal leaves a render encoder's state undefined after an indirect execution, so the
            // parent's three bindings are stated again before each one rather than once.
            steps.push(RenderStep::Pipeline(pipeline));
            steps.push(RenderStep::VertexBuffer {
                buffer: params,
                offset: PARENT_PARAMS * PARAM_STRIDE,
                index: 0,
            });
            steps.push(RenderStep::VertexBuffer {
                buffer: out,
                offset: PARENT_SEGMENT * SEGMENT_BYTES,
                index: 1,
            });
            steps.push(RenderStep::ExecuteCommands {
                commands: *commands,
                first: 0,
                count: *count,
            });
        }
        let work = vec![RenderWork::Pass(RenderPass::sized([1, 1]).drawing(steps))];
        let outcome = call(sites, "render", side.render(&work), |outcome| {
            outcome.status.clone()
        })?;
        sites.push(program_node(&work, &outcome));
        if let Some((at, failure)) = first_refused_piece(&work, &outcome) {
            return Err(Abort { at, failure });
        }
        for leg in INHERIT_LEGS {
            for segment in leg.segments() {
                read_words(
                    side,
                    sites,
                    &format!("{}/segment-{segment}", leg.site()),
                    out,
                    segment * SEGMENT_BYTES,
                    SEGMENT_BYTES,
                )?;
            }
        }
        read_words(
            side,
            sites,
            "parent-segment",
            out,
            PARENT_SEGMENT * SEGMENT_BYTES,
            SEGMENT_BYTES,
        )?;
        read_words(
            side,
            sites,
            "tail",
            out,
            (INHERIT_SEGMENTS - 1) * SEGMENT_BYTES,
            SEGMENT_BYTES,
        )
    }
}

// --------------------------------------------------------------------------------------------
// A reset range crossed with an execution range

/// The name of the row that crosses `resetWithRange:`, a command's own `reset`, and the range an
/// execution runs.
pub const RESET_RANGES: &str = "indirect-compute-commands-reset-and-ranges";

/// The owned fixture an executed compute command dispatches: each thread writes
/// `params[0] * (tid + 1) + params[1]` to `out[tid]` through threadgroup memory, so a command's
/// grid and both its bindings are readable off one word.
const COMPUTE_FIXTURE: &str = "icb_replay_template";

/// Words one compute command's output segment holds: eight, which is above the largest grid any
/// command of the row dispatches, so a command that wrote past its own grid reads inside its own
/// segment rather than in the next command's.
const RESET_SEGMENT_WORDS: u64 = 8;

/// Bytes one compute command's output segment takes.
const RESET_SEGMENT_BYTES: u64 = RESET_SEGMENT_WORDS * 4;

/// Segments one leg of the row owns: four the four commands bind, one the re-encoded command 1
/// binds, and a tail no dispatch reaches.
const RESET_SEGMENTS: u64 = 6;

/// Bytes one leg's output region takes.
const RESET_LEG_BYTES: u64 = RESET_SEGMENTS * RESET_SEGMENT_BYTES;

/// Commands each leg of the row encodes.
const RESET_COMMANDS: u64 = 4;

/// The threadgroup length every compute command sets at index 0, in bytes: four words, at or above
/// the largest threadgroup any command dispatches and a multiple of 16, as Metal's encoder
/// requires of a threadgroup length.
const THREADGROUP_BYTES: u32 = 16;

/// The segment the re-encoded command 1 binds, which no other command of a leg reaches, so its
/// words say the re-encode landed and the original command's segment says the `reset` took.
const REENCODED_SEGMENT: u64 = 4;

/// The parameter pair the re-encoded command 1 binds, whose words no other command binds.
const REENCODED_PARAMS: u64 = 4;

/// Which leg of [`ResetRanges`] a buffer is: what it resets, and what range it executes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ResetLeg {
    /// `resetWithRange:(1, 2)`, then the **whole** buffer executed. The reset's two commands are
    /// blank and the two outside it run.
    SubRangeWholeExecution,
    /// `resetWithRange:(2, 1)`, command 1 reset by its own `reset` and re-encoded differently in
    /// the same list, then the range `1..3` executed: the one leg where all three of the reset
    /// calls, the re-encode and the execution range meet.
    ReEncodedSubRange,
    /// No reset, and a range of **no** commands executed from the middle of the buffer.
    EmptyRange,
    /// No reset, and the range whose end is the buffer's own command count.
    TailRange,
}

impl ResetLeg {
    /// The leg's own site name.
    fn site(self) -> &'static str {
        match self {
            ResetLeg::SubRangeWholeExecution => "reset-sub-range-whole-execution",
            ResetLeg::ReEncodedSubRange => "reset-re-encoded-sub-range",
            ResetLeg::EmptyRange => "empty-range",
            ResetLeg::TailRange => "range-ending-at-the-command-count",
        }
    }

    /// The leg's index, which is the output region it owns.
    fn at(self) -> u64 {
        match self {
            ResetLeg::SubRangeWholeExecution => 0,
            ResetLeg::ReEncodedSubRange => 1,
            ResetLeg::EmptyRange => 2,
            ResetLeg::TailRange => 3,
        }
    }

    /// The `resetWithRange:` the leg makes on the host, as `(first, count)`, or `None`.
    fn reset(self) -> Option<(u64, u64)> {
        match self {
            ResetLeg::SubRangeWholeExecution => Some((1, 2)),
            ResetLeg::ReEncodedSubRange => Some((2, 1)),
            ResetLeg::EmptyRange | ResetLeg::TailRange => None,
        }
    }

    /// The range the leg executes, as `(first, count)`.
    fn executed(self) -> (u64, u64) {
        match self {
            ResetLeg::SubRangeWholeExecution => (0, RESET_COMMANDS),
            ResetLeg::ReEncodedSubRange => (1, 2),
            ResetLeg::EmptyRange => (2, 0),
            ResetLeg::TailRange => (RESET_COMMANDS - 1, 1),
        }
    }

    /// Whether the leg re-encodes command 1 after its own `reset`.
    fn re_encodes(self) -> bool {
        self == ResetLeg::ReEncodedSubRange
    }
}

/// Every leg of [`ResetRanges`].
const RESET_LEGS: [ResetLeg; 4] = [
    ResetLeg::SubRangeWholeExecution,
    ResetLeg::ReEncodedSubRange,
    ResetLeg::EmptyRange,
    ResetLeg::TailRange,
];

/// The grids the four commands of a leg dispatch, in order: four different counts of words, so a
/// command that ran another's grid writes a different count of words at a different place.
const RESET_GRIDS: [[u32; 3]; RESET_COMMANDS as usize] =
    [[4, 1, 1], [3, 1, 1], [2, 1, 1], [1, 1, 1]];

/// The row's descriptor: dispatches of threadgroups, two kernel buffers, one threadgroup length,
/// nothing inherited.
fn reset_descriptor() -> IcbDescriptor {
    IcbDescriptor {
        command_types: IndirectCommandType::CONCURRENT_DISPATCH,
        max_kernel_buffer_bind_count: 2,
        max_kernel_threadgroup_memory_bind_count: 1,
        ..IcbDescriptor::new()
    }
}

/// The commands one leg encodes, over the parameter buffer and the output buffer.
fn reset_commands(
    leg: ResetLeg,
    pipeline: Handle,
    params: Handle,
    out: Handle,
) -> Vec<EncodedCommand> {
    let base = leg.at() * RESET_LEG_BYTES;
    let bind = |at: u64, segment: u64, grid: [u32; 3]| {
        vec![
            ComputeCommandStep::Pipeline(pipeline),
            ComputeCommandStep::KernelBuffer {
                buffer: params,
                offset: at * PARAM_STRIDE,
                index: 0,
            },
            ComputeCommandStep::KernelBuffer {
                buffer: out,
                offset: base + segment * RESET_SEGMENT_BYTES,
                index: 1,
            },
            ComputeCommandStep::ThreadgroupMemoryLength {
                length: THREADGROUP_BYTES,
                index: 0,
            },
            ComputeCommandStep::DispatchThreadgroups {
                groups: [1, 1, 1],
                threads: grid,
            },
        ]
    };
    (0..RESET_COMMANDS)
        .map(|at| {
            let mut steps = bind(at, at, RESET_GRIDS[at as usize]);
            if leg.re_encodes() && at == 1 {
                // The command is taken back after it was encoded and then encoded again, at
                // another parameter window, another segment and another grid: so its answer says
                // which of the two encodings the execution ran, and a side that kept the first
                // writes the segment the `reset` was supposed to blank.
                steps.push(ComputeCommandStep::Reset);
                steps.extend(bind(REENCODED_PARAMS, REENCODED_SEGMENT, [2, 1, 1]));
            }
            EncodedCommand::Compute(steps)
        })
        .collect()
}

/// The row that crosses the two calls that blank a command with the range an execution runs.
///
/// # The four legs, and what only their pairs can see
///
/// Four buffers hold the **same four commands**, each dispatching its own grid into its own
/// segment of its own leg's region, and differ only in what they reset and what range they run:
///
/// - **`reset-sub-range-whole-execution`**: `resetWithRange:(1, 2)` and then every command run.
///   Segments 0 and 3 hold their commands' words and segments 1 and 2 are untouched. This is the
///   cell the per-value axis cannot reach: [`super::indirect_commands::COMPUTE_RESET`] resets a
///   range too, and executes the whole buffer as well, so a side that widened a reset range to the
///   whole buffer would blank everything there and still be reporting "reset works".
/// - **`reset-re-encoded-sub-range`**: `resetWithRange:(2, 1)`, command 1 taken back by its own
///   `reset` and encoded again at another window, another segment and another grid, and the range
///   `1..3` run. Only the re-encoded segment holds words: command 0 and command 3 are outside the
///   range, command 2 is blank, and command 1's first encoding was taken back.
/// - **`empty-range`**: a range of **no** commands from the middle of the buffer. Nothing is
///   written anywhere, which is the one answer a side that treated a zero length as "all of them"
///   cannot give.
/// - **`range-ending-at-the-command-count`**: the last command alone, the range whose end is the
///   buffer's own size.
///
/// The falsification these four were written for is one **only a pair** can see: widen
/// `resetWithRange:` to start at command 0. `reset-sub-range-whole-execution` loses segment 0 and
/// `reset-re-encoded-sub-range` does **not move at all** — its execution range never reaches
/// command 0 — so which leg sees the bug is decided by the execution range and by nothing else.
pub struct ResetRanges;

impl Case for ResetRanges {
    fn name(&self) -> &'static str {
        RESET_RANGES
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        Vec::new()
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        shaders_of(COMPUTE_FIXTURE, COMPUTE_FIXTURE)
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let mut inputs = fixture_inputs(COMPUTE_FIXTURE)?;
        for leg in RESET_LEGS {
            let commands = reset_commands(leg, Handle(0), Handle(1), Handle(2));
            inputs.extend(commands_inputs(leg.site(), &reset_descriptor(), &commands));
            let (first, count) = leg.executed();
            inputs.push(Input::bytes(
                format!("{}:ranges", leg.site()),
                format!(
                    "reset {:?}, executed {first}..{}",
                    leg.reset(),
                    first + count
                )
                .as_bytes(),
            ));
        }
        inputs.push(Input::bytes("buffer:params", &params_bytes()));
        inputs.push(Input::bytes(
            "buffer:out",
            &vec![UNWRITTEN; (RESET_LEGS.len() as u64 * RESET_LEG_BYTES) as usize],
        ));
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let fixture = Fixture::new(COMPUTE_FIXTURE);
        let loaded = fixture.loaded().map_err(|why| Abort {
            at: "library".to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        })?;
        let library = made(sites, "library", side.library(&loaded.source()))?;
        let pipeline = made(
            sites,
            "pipeline",
            side.compute_pipeline_for_commands(library, COMPUTE_FIXTURE, None),
        )?;
        let params = made(sites, "buffer:params", side.buffer(&params_bytes()))?;
        let bytes = (RESET_LEGS.len() as u64 * RESET_LEG_BYTES) as usize;
        let out = made(sites, "buffer:out", side.buffer(&vec![UNWRITTEN; bytes]))?;

        let mut work = Vec::new();
        for leg in RESET_LEGS {
            let commands = reset_commands(leg, pipeline, params, out);
            let encoded = made(
                sites,
                &format!("{}/command-buffer", leg.site()),
                side.indirect_command_buffer(&reset_descriptor(), &commands),
            )?;
            sites.push(encoding_node(
                &format!("{}/encoded", leg.site()),
                &commands,
                &encoded.refused,
            ));
            if let Some((at, failure)) = first_refusal(
                &format!("{}/encoded", leg.site()),
                &commands,
                &encoded.refused,
            ) {
                return Err(Abort { at, failure });
            }
            if let Some((first, count)) = leg.reset() {
                call(
                    sites,
                    &format!("{}/reset-with-range", leg.site()),
                    side.reset_indirect_commands(encoded.commands, first, count),
                    |_| Value::Absent,
                )?;
            }
            let (first, count) = leg.executed();
            work.push(RenderWork::ComputeCommands {
                commands: encoded.commands,
                first,
                count,
                used: vec![(params, false), (out, true)],
            });
        }
        let outcome = call(sites, "render", side.render(&work), |outcome| {
            outcome.status.clone()
        })?;
        sites.push(program_node(&work, &outcome));
        if let Some((at, failure)) = first_refused_piece(&work, &outcome) {
            return Err(Abort { at, failure });
        }
        for leg in RESET_LEGS {
            read_words(
                side,
                sites,
                &format!("{}/out", leg.site()),
                out,
                leg.at() * RESET_LEG_BYTES,
                RESET_LEG_BYTES,
            )?;
        }
        Ok(())
    }
}

// --------------------------------------------------------------------------------------------
// The pass's own render state crossed with the draws a command makes

/// The name of the row that crosses the pass's own scissor and viewport with the two halves an
/// indirect render command rasterizes.
pub const PASS_SCISSOR: &str = "indirect-render-commands-under-the-pass-state";

/// The owned fixture whose `tob_rect` draws a rectangle from a buffer's two corners and whose
/// `tob_solid` returns a colour from a buffer.
const DRAW_FIXTURE: &str = "tob_renders";

/// The target's side in texels, and its format: a format whose codes survive a store unchanged, so
/// every texel the row compares is exact.
const DRAWN_SIDE: u32 = 8;
const DRAWN_FORMAT: PixelFormat = PixelFormat::RGBA8Unorm;

/// Bytes one texel of [`DRAWN_FORMAT`] takes.
const DRAWN_TEXEL: u32 = 4;

/// The clear the pass loads its target with: a code neither command's colour holds, so a texel no
/// command reached reads as this.
const DRAWN_CLEAR: u8 = 7;

/// The two rectangles the commands draw, as `tob_rect` takes them: the two clip-space corners
/// `(x, y)` and `(z, w)`. The left half and the right half, so which command drew which column is
/// unambiguous.
const RECTS: [[f32; 4]; 2] = [[-1.0, -1.0, 0.0, 1.0], [0.0, -1.0, 1.0, 1.0]];

/// The two colours the commands' fragment buffers hold, each channel a whole code over 255 so the
/// store is exact, and every channel of each different from the other's and from the clear.
const COLORS: [[u8; 4]; 2] = [[16, 64, 112, 200], [160, 208, 224, 240]];

/// Bytes between one command's `float4` and the next, in both the rectangle and the colour buffer.
const FLOAT4_STRIDE: u64 = 16;

/// The rectangle buffer's bytes: [`RECTS`], one `float4` each.
fn rect_bytes() -> Vec<u8> {
    RECTS
        .iter()
        .flat_map(|rect| {
            rect.iter()
                .flat_map(|edge| edge.to_le_bytes())
                .collect::<Vec<u8>>()
        })
        .collect()
}

/// The colour buffer's bytes: [`COLORS`], one `float4` each, each channel its code over 255.
fn color_bytes() -> Vec<u8> {
    COLORS
        .iter()
        .flat_map(|color| {
            color
                .iter()
                .flat_map(|code| (f32::from(*code) / 255.0).to_le_bytes())
                .collect::<Vec<u8>>()
        })
        .collect()
}

/// The clear colour, as the pass takes it.
fn drawn_clear() -> [f64; 4] {
    [f64::from(DRAWN_CLEAR) / 255.0; 4]
}

/// The scissor the first leg's pass sets: four columns from column 2, the full height. It crosses
/// **both** commands — the left half is columns 0 to 3 and the right half columns 4 to 7 — so one
/// rectangle is enough to say that each command was clipped and that neither was dropped.
const PASS_SCISSOR_BOX: ScissorBox = ScissorBox::new(2, 0, 4, DRAWN_SIDE as u64);

/// The viewport the second leg's pass sets: the full width, four rows from row 2. The two
/// rectangles span the whole of clip space, so the viewport is what decides which rows they cover
/// and the answer names the rows rather than the columns the scissor names.
fn pass_viewport() -> crate::render::ViewportRect {
    crate::render::ViewportRect::new([0.0, 2.0], [f64::from(DRAWN_SIDE), 4.0])
}

/// The row's descriptor: draws of primitives, one vertex buffer and one fragment buffer, nothing
/// inherited.
fn drawn_descriptor() -> IcbDescriptor {
    IcbDescriptor {
        command_types: IndirectCommandType::DRAW,
        max_vertex_buffer_bind_count: 1,
        max_fragment_buffer_bind_count: 1,
        ..IcbDescriptor::new()
    }
}

/// The one colour attachment the row's pipeline carries.
const DRAWN_COLOR: [Option<ColorState>; 1] = [Some(ColorState::new(DRAWN_FORMAT))];

/// The pipeline the row's commands hold: `tob_rect` and `tob_solid` over one attachment, with
/// `supportIndirectCommandBuffers`.
fn drawn_pipeline_spec() -> RenderPipelineSpec<'static> {
    RenderPipelineSpec::new("tob_rect", "tob_solid", &DRAWN_COLOR).for_indirect_commands()
}

/// The two commands the row encodes: one a rectangle, each binding its own `float4` of the two
/// buffers, so a command that bound the wrong offset draws the other's rectangle or colour.
///
/// Neither holds a barrier, unlike [`super::indirect_commands::RENDER_DRAWN`]'s: the two
/// rectangles cover disjoint columns, so the order is immaterial, and leaving the barrier out keeps
/// this row's answer free of the `render-pass-split` notice that row announces — which is what lets
/// the pass state here be compared with nothing loosened.
fn drawn_commands(pipeline: Handle, rects: Handle, colors: Handle) -> Vec<EncodedCommand> {
    (0..RECTS.len() as u64)
        .map(|at| {
            EncodedCommand::Render(vec![
                RenderCommandStep::Pipeline(pipeline),
                RenderCommandStep::VertexBuffer {
                    buffer: rects,
                    offset: at * FLOAT4_STRIDE,
                    index: 0,
                },
                RenderCommandStep::FragmentBuffer {
                    buffer: colors,
                    offset: at * FLOAT4_STRIDE,
                    index: 0,
                },
                RenderCommandStep::Draw {
                    primitive: PrimitiveType::Triangle,
                    first: 0,
                    count: 6,
                    instances: 1,
                    base_instance: 0,
                },
            ])
        })
        .collect()
}

/// The layout a target is read under: one row of the layout a row of texels, one element a
/// channel, so a diff names the pixel and the channel.
fn drawn_layout() -> ByteLayout {
    let row_pitch = DRAWN_TEXEL * DRAWN_SIDE;
    ByteLayout {
        region: Region::Texture {
            level: 0,
            slice: 0,
            aspect: Aspect::Color,
            sample: 0,
            origin: [0, 0, 0],
            size: [DRAWN_SIDE, DRAWN_SIDE, 1],
        },
        element: Element::U8,
        row_offset: 0,
        row_elements: row_pitch,
        row_pitch,
        rows: DRAWN_SIDE,
        image_pitch: row_pitch * DRAWN_SIDE,
        images: 1,
        gaps: Gaps::Compared,
    }
}

/// Which of the two pass states a leg sets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PassState {
    /// `setScissorRect:` of [`PASS_SCISSOR_BOX`], which clips the columns.
    Scissor,
    /// `setViewports:count:` of one [`pass_viewport`], which moves the rows.
    Viewport,
}

impl PassState {
    fn site(self) -> &'static str {
        match self {
            PassState::Scissor => "pass-scissor",
            PassState::Viewport => "pass-viewport",
        }
    }

    /// The step the pass states before it executes the commands.
    fn step(self) -> RenderStep {
        match self {
            PassState::Scissor => RenderStep::Scissor(PASS_SCISSOR_BOX),
            PassState::Viewport => RenderStep::Viewport(pass_viewport()),
        }
    }
}

/// Both legs of [`PassStateCommands`].
const PASS_STATES: [PassState; 2] = [PassState::Scissor, PassState::Viewport];

/// The row that crosses the pass's own render state with the draws an indirect command makes.
///
/// # Why it is a crossing and not a value
///
/// A render command **may not set a scissor, a viewport or any other render state**: every one of
/// those setters raises `APINotSupported` and ends Metal's process on this oracle
/// ([`crate::indirect_commands`]), so the vocabulary has no step for them. The consequence is
/// exactly the question this row asks: an indirect draw's scissor and viewport can only be the
/// **parent encoder's**, and an implementation that reset them when it took the commands' own
/// state, or that ran the draws outside its own pass instance, would answer a whole target where
/// a clipped one belongs.
///
/// Nothing on the per-value axis reaches it. `setScissorRect:` and `setViewport:` are compared by
/// [`super::render_state`] over the encoder's own draws, and an indirect draw is compared by
/// [`super::indirect_commands::RENDER_DRAWN`] under a pass that sets neither. This row is their
/// product: the same two commands as that row's, under a pass that sets one.
///
/// Both legs read every texel of their own 8-by-8 target:
///
/// - **`pass-scissor`** sets four columns from column 2. Columns 2 and 3 take command 0's colour,
///   columns 4 and 5 command 1's, and columns 0, 1, 6 and 7 stay at the clear. One rectangle
///   clipping both commands is what says neither was dropped and neither escaped.
/// - **`pass-viewport`** sets the full width and four rows from row 2. The rectangles span the
///   whole of clip space, so rows 2 to 5 hold the two colours in their own halves and rows 0, 1, 6
///   and 7 stay at the clear — a different axis of the same question, and one a scissor cannot
///   answer, since a viewport *maps* the draw where a scissor *discards* it.
pub struct PassStateCommands;

impl Case for PassStateCommands {
    fn name(&self) -> &'static str {
        PASS_SCISSOR
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // A normalized store our side may round in the shader where the device's own conversion is
        // not Metal's, which is what `RENDER_DRAWN` measured on lavapipe and not on the M3 Max.
        // Named because a notice our run announces that no case names is a divergence; it loosens
        // no texel, and every texel of this row is still compared exactly.
        vec![crate::replay::notice_allowance(
            metal2vulkan_runtime::Degradation::ShaderColorConversion.code(),
        )]
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        shaders_of(DRAW_FIXTURE, "tob_solid")
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let mut inputs = fixture_inputs(DRAW_FIXTURE)?;
        let commands = drawn_commands(Handle(0), Handle(1), Handle(2));
        inputs.extend(commands_inputs("icb", &drawn_descriptor(), &commands));
        inputs.push(Input::bytes(
            "render-pipeline",
            &drawn_pipeline_spec().words(),
        ));
        inputs.push(Input::bytes(
            "target",
            &TargetSpec::plain(DRAWN_FORMAT, [DRAWN_SIDE, DRAWN_SIDE]).words(),
        ));
        inputs.push(Input::bytes("buffer:rects", &rect_bytes()));
        inputs.push(Input::bytes("buffer:colors", &color_bytes()));
        inputs.push(Input::bytes(
            "clear",
            format!("{DRAWN_CLEAR} in every channel").as_bytes(),
        ));
        inputs.push(Input::bytes("pass-scissor", &PASS_SCISSOR_BOX.words()));
        inputs.push(Input::bytes("pass-viewport", &pass_viewport().words()));
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let fixture = Fixture::new(DRAW_FIXTURE);
        let loaded = fixture.loaded().map_err(|why| Abort {
            at: "library".to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        })?;
        let library = made(sites, "library", side.library(&loaded.source()))?;
        let pipeline = made(
            sites,
            "pipeline",
            side.render_pipeline(library, &drawn_pipeline_spec()),
        )?;
        let rects = made(sites, "buffer:rects", side.buffer(&rect_bytes()))?;
        let colors = made(sites, "buffer:colors", side.buffer(&color_bytes()))?;
        let texels = (DRAWN_SIDE * DRAWN_SIDE * DRAWN_TEXEL) as usize;

        let commands = drawn_commands(pipeline, rects, colors);
        let encoded = made(
            sites,
            "command-buffer",
            side.indirect_command_buffer(&drawn_descriptor(), &commands),
        )?;
        sites.push(encoding_node("encoded", &commands, &encoded.refused));
        if let Some((at, failure)) = first_refusal("encoded", &commands, &encoded.refused) {
            return Err(Abort { at, failure });
        }

        let mut targets = Vec::new();
        let mut reads = Vec::new();
        let mut work = Vec::new();
        for state in PASS_STATES {
            let target = made(
                sites,
                &format!("{}/target", state.site()),
                side.target(&TargetSpec::plain(DRAWN_FORMAT, [DRAWN_SIDE, DRAWN_SIDE])),
            )?;
            let read_back = made(
                sites,
                &format!("{}/buffer:read-back", state.site()),
                side.buffer(&vec![0; texels]),
            )?;
            targets.push(target);
            reads.push(read_back);
            let steps = vec![
                RenderStep::UseBuffer {
                    buffer: rects,
                    write: false,
                    stages: STAGE_VERTEX,
                },
                RenderStep::UseBuffer {
                    buffer: colors,
                    write: false,
                    stages: crate::case::STAGE_FRAGMENT,
                },
                state.step(),
                RenderStep::ExecuteCommands {
                    commands: encoded.commands,
                    first: 0,
                    count: commands.len() as u64,
                },
            ];
            work.push(RenderWork::Pass(
                RenderPass::clearing(vec![PassColor::cleared(target, drawn_clear())])
                    .drawing(steps),
            ));
            work.push(RenderWork::ReadTexture {
                texture: target,
                into: read_back,
                offset: 0,
                row_pitch: DRAWN_TEXEL * DRAWN_SIDE,
                size: [DRAWN_SIDE, DRAWN_SIDE],
            });
        }
        let outcome = call(sites, "render", side.render(&work), |outcome| {
            outcome.status.clone()
        })?;
        sites.push(program_node(&work, &outcome));
        if let Some((at, failure)) = first_refused_piece(&work, &outcome) {
            return Err(Abort { at, failure });
        }
        for (state, read_back) in PASS_STATES.into_iter().zip(reads) {
            let read = side
                .read_buffer(read_back, 0, texels)
                .map_err(|failure| Abort {
                    at: format!("{}/target", state.site()),
                    failure,
                })?;
            sites.push(Node::leaf(
                format!("{}/target-texels", state.site()),
                Observed::bytes(read, drawn_layout()),
            ));
        }
        Ok(())
    }
}

// --------------------------------------------------------------------------------------------
// The resident set's declared usage crossed with what the commands do

/// The name of the row that crosses `MTLResourceUsage` as declared with what an indirect command's
/// kernel reads and writes.
pub const RESIDENCY_USAGE: &str = "indirect-compute-commands-residency-usage";

/// Commands the residency row encodes.
const USAGE_COMMANDS: u64 = 3;

/// Which way round a leg declares the two buffers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Declared {
    /// `MTLResourceUsageRead` for the buffer the kernels read and
    /// `MTLResourceUsageRead | MTLResourceUsageWrite` for the one they write: the declaration a
    /// program would make, and the control the other leg is read against.
    Matching,
    /// The two the other way round: the written buffer declared **read-only**, and the read buffer
    /// declared writable.
    Crossed,
}

impl Declared {
    fn site(self) -> &'static str {
        match self {
            Declared::Matching => "usage-as-used",
            Declared::Crossed => "usage-crossed",
        }
    }

    /// The leg's index, which is the output region it owns.
    fn at(self) -> u64 {
        match self {
            Declared::Matching => 0,
            Declared::Crossed => 1,
        }
    }

    /// `(params, out)` as the leg declares them, each `true` for
    /// `MTLResourceUsageRead | MTLResourceUsageWrite`.
    fn usage(self) -> (bool, bool) {
        match self {
            Declared::Matching => (false, true),
            Declared::Crossed => (true, false),
        }
    }
}

/// Both legs of [`ResidencyUsage`].
const DECLARATIONS: [Declared; 2] = [Declared::Matching, Declared::Crossed];

/// Bytes one leg's output region takes: one segment a command and a tail no dispatch reaches.
const USAGE_LEG_BYTES: u64 = (USAGE_COMMANDS + 1) * RESET_SEGMENT_BYTES;

/// The commands one leg encodes: three dispatches, each over its own parameter window and its own
/// segment of its own leg's region.
fn usage_commands(
    leg: Declared,
    pipeline: Handle,
    params: Handle,
    out: Handle,
) -> Vec<EncodedCommand> {
    let base = leg.at() * USAGE_LEG_BYTES;
    (0..USAGE_COMMANDS)
        .map(|at| {
            EncodedCommand::Compute(vec![
                ComputeCommandStep::Pipeline(pipeline),
                ComputeCommandStep::KernelBuffer {
                    buffer: params,
                    offset: at * PARAM_STRIDE,
                    index: 0,
                },
                ComputeCommandStep::KernelBuffer {
                    buffer: out,
                    offset: base + at * RESET_SEGMENT_BYTES,
                    index: 1,
                },
                ComputeCommandStep::ThreadgroupMemoryLength {
                    length: THREADGROUP_BYTES,
                    index: 0,
                },
                ComputeCommandStep::DispatchThreadgroups {
                    groups: [1, 1, 1],
                    threads: RESET_GRIDS[at as usize],
                },
            ])
        })
        .collect()
}

/// The row that crosses the usage a resident resource is declared under with what the commands
/// actually do to it.
///
/// # Why both ways round
///
/// A command's buffer bindings are addresses, so Metal tracks nothing for them and the executing
/// encoder has to say which resources the commands' functions reach (`useResource:usage:`). What
/// it *also* says is a usage, and the question no per-value row asks is whether that usage is a
/// **promise Metal keeps us to** or a hint: an `MTLResourceUsageRead` declaration over a buffer the
/// commands write is the shape a program gets wrong, and the owner's standing ruling is that a
/// read-only GPU write is **dropped, never refused**, on every driver.
///
/// Two legs hold the same three dispatches over their own regions, and differ only in the
/// declaration:
///
/// - **`usage-as-used`** declares `Read` for the parameter buffer and `Read | Write` for the output
///   buffer: the matching declaration, and the control.
/// - **`usage-crossed`** declares them the other way round — the written buffer **read-only** and
///   the read buffer writable — and its commands write it anyway.
///
/// The two legs' words being equal is the **result** this row reports, not a vacuity: it says the
/// declared usage and the access performed are independent on this oracle, and that our side
/// neither drops the write nor refuses the declaration. It is held by execution rather than by
/// discipline: patching our leg to drop a write whose resource was declared read-only parts
/// `usage-crossed` from `usage-as-used` and moves nothing else in the suite.
pub struct ResidencyUsage;

impl Case for ResidencyUsage {
    fn name(&self) -> &'static str {
        RESIDENCY_USAGE
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        Vec::new()
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        shaders_of(COMPUTE_FIXTURE, COMPUTE_FIXTURE)
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let mut inputs = fixture_inputs(COMPUTE_FIXTURE)?;
        for leg in DECLARATIONS {
            let commands = usage_commands(leg, Handle(0), Handle(1), Handle(2));
            inputs.extend(commands_inputs(leg.site(), &reset_descriptor(), &commands));
            let (params, out) = leg.usage();
            inputs.push(Input::bytes(
                format!("{}:declared", leg.site()),
                format!("params write {params}, out write {out}").as_bytes(),
            ));
        }
        inputs.push(Input::bytes("buffer:params", &params_bytes()));
        inputs.push(Input::bytes(
            "buffer:out",
            &vec![UNWRITTEN; (DECLARATIONS.len() as u64 * USAGE_LEG_BYTES) as usize],
        ));
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let fixture = Fixture::new(COMPUTE_FIXTURE);
        let loaded = fixture.loaded().map_err(|why| Abort {
            at: "library".to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        })?;
        let library = made(sites, "library", side.library(&loaded.source()))?;
        let pipeline = made(
            sites,
            "pipeline",
            side.compute_pipeline_for_commands(library, COMPUTE_FIXTURE, None),
        )?;
        let params = made(sites, "buffer:params", side.buffer(&params_bytes()))?;
        let bytes = (DECLARATIONS.len() as u64 * USAGE_LEG_BYTES) as usize;
        let out = made(sites, "buffer:out", side.buffer(&vec![UNWRITTEN; bytes]))?;

        let mut work = Vec::new();
        for leg in DECLARATIONS {
            let commands = usage_commands(leg, pipeline, params, out);
            let encoded = made(
                sites,
                &format!("{}/command-buffer", leg.site()),
                side.indirect_command_buffer(&reset_descriptor(), &commands),
            )?;
            sites.push(encoding_node(
                &format!("{}/encoded", leg.site()),
                &commands,
                &encoded.refused,
            ));
            if let Some((at, failure)) = first_refusal(
                &format!("{}/encoded", leg.site()),
                &commands,
                &encoded.refused,
            ) {
                return Err(Abort { at, failure });
            }
            let (params_write, out_write) = leg.usage();
            work.push(RenderWork::ComputeCommands {
                commands: encoded.commands,
                first: 0,
                count: commands.len() as u64,
                used: vec![(params, params_write), (out, out_write)],
            });
        }
        let outcome = call(sites, "render", side.render(&work), |outcome| {
            outcome.status.clone()
        })?;
        sites.push(program_node(&work, &outcome));
        if let Some((at, failure)) = first_refused_piece(&work, &outcome) {
            return Err(Abort { at, failure });
        }
        for leg in DECLARATIONS {
            read_words(
                side,
                sites,
                &format!("{}/out", leg.site()),
                out,
                leg.at() * USAGE_LEG_BYTES,
                USAGE_LEG_BYTES,
            )?;
        }
        Ok(())
    }
}

// --------------------------------------------------------------------------------------------
// The declared bind-count maximum crossed with the index a command binds

/// The name of the row that binds a kernel buffer at an index at and past the descriptor's
/// declared maximum.
pub const BIND_LIMITS: &str = "indirect-commands-bind-count-limit";

/// The maximum the row's descriptor declares, and so the first index that is not below it.
const DECLARED_BINDS: u32 = 2;

/// The indices past the declared maximum the row binds, each on a command of its own: the first
/// index at the maximum, and one two past it.
///
/// Both are small on purpose. A binding past the slot array lands in the **same command's** record
/// on plain Metal ("plain Metal keeps it for its private getters", `metal2vulkan_runtime`'s
/// `IndirectCommandBuffer::set_buffer`), and the row **executes nothing**, so no over-long binding
/// ever reaches a GPU — which is why the indices are 2 and 4 and not something that would reach
/// another command's record or another page.
const OVER_INDICES: [u32; 2] = [DECLARED_BINDS, DECLARED_BINDS + 2];

/// The row's descriptor: dispatches of threadgroups and [`DECLARED_BINDS`] kernel buffers.
fn bind_limit_descriptor() -> IcbDescriptor {
    IcbDescriptor {
        command_types: IndirectCommandType::CONCURRENT_DISPATCH,
        max_kernel_buffer_bind_count: DECLARED_BINDS,
        max_kernel_threadgroup_memory_bind_count: 1,
        ..IcbDescriptor::new()
    }
}

/// The commands the row encodes: one that binds every index below the maximum, and one for each
/// index at or past it.
fn bind_limit_commands(pipeline: Handle, params: Handle) -> Vec<EncodedCommand> {
    let mut commands = vec![EncodedCommand::Compute(
        (0..DECLARED_BINDS)
            .map(|index| ComputeCommandStep::KernelBuffer {
                buffer: params,
                offset: 0,
                index,
            })
            .chain([ComputeCommandStep::Pipeline(pipeline)])
            .collect(),
    )];
    for index in OVER_INDICES {
        commands.push(EncodedCommand::Compute(vec![
            ComputeCommandStep::Pipeline(pipeline),
            ComputeCommandStep::KernelBuffer {
                buffer: params,
                offset: 0,
                index,
            },
        ]));
    }
    commands
}

/// The row that crosses the descriptor's declared bind-count maximum with the index a command
/// binds at.
///
/// # What it measures, and why it executes nothing
///
/// `maxKernelBufferBindCount` is the one descriptor field this suite sets everywhere and asks
/// nothing of: every row of [`super::indirect_commands`] declares exactly the count its commands
/// bind, so no recording says what happens at the boundary. Three commands here do:
///
/// - command 0 binds every index **below** the maximum, which must be taken;
/// - command 1 binds the index **at** it;
/// - command 2 binds one two **past** it.
///
/// Our side raises `Invalid` (`Problem::IndexOutOfRange`) for both of the last two. Plain Metal
/// takes them and keeps the binding for its own getters, which is the runtime's own measurement on
/// an M3 Max, so the row's verdict is a **refusal of ours** and `known-gaps.tsv` holds it with that
/// measurement beside it. It is listed there rather than written around because, by the owner's
/// standing rule, what Metal allows we must allow; whether the right fix is to keep the binding in
/// an overflow the way the runtime already does past slot 31 is the todo behind the row.
///
/// The row **makes the buffer and never executes it**. There is no pass, no dispatch and no
/// read-back: what it compares is the encoding's own answers and the buffer's `size`, so a binding
/// that landed somewhere plain Metal does not define is never handed to a GPU. It is also why the
/// two indices are small ([`OVER_INDICES`]).
pub struct BindLimits;

impl Case for BindLimits {
    fn name(&self) -> &'static str {
        BIND_LIMITS
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        Vec::new()
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        shaders_of(COMPUTE_FIXTURE, COMPUTE_FIXTURE)
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let mut inputs = fixture_inputs(COMPUTE_FIXTURE)?;
        let commands = bind_limit_commands(Handle(0), Handle(1));
        inputs.extend(commands_inputs("icb", &bind_limit_descriptor(), &commands));
        inputs.push(Input::bytes("buffer:params", &params_bytes()));
        inputs.push(Input::bytes("executed", b"nothing"));
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let fixture = Fixture::new(COMPUTE_FIXTURE);
        let loaded = fixture.loaded().map_err(|why| Abort {
            at: "library".to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        })?;
        let library = made(sites, "library", side.library(&loaded.source()))?;
        let pipeline = made(
            sites,
            "pipeline",
            side.compute_pipeline_for_commands(library, COMPUTE_FIXTURE, None),
        )?;
        let params = made(sites, "buffer:params", side.buffer(&params_bytes()))?;

        let commands = bind_limit_commands(pipeline, params);
        let encoded = made(
            sites,
            "command-buffer",
            side.indirect_command_buffer(&bind_limit_descriptor(), &commands),
        )?;
        // Every step's own answer is kept rather than aborted on: the whole point of the row is
        // which of the three commands each side took, so a refusal at command 1 must not hide what
        // command 2 answered.
        sites.push(encoding_node("encoded", &commands, &encoded.refused));
        call(
            sites,
            "size",
            side.indirect_commands_size(encoded.commands),
            |size| Value::Uint(*size),
        )?;
        Ok(())
    }
}

/// Every row of this module.
pub fn boxed() -> Vec<Box<dyn Case>> {
    vec![
        Box::new(PatchDraws),
        Box::new(InheritedState),
        Box::new(ResetRanges),
        Box::new(PassStateCommands),
        Box::new(ResidencyUsage),
        Box::new(BindLimits),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// Every row's name is distinct, since a name is what a recording is keyed by.
    #[test]
    fn every_row_has_its_own_name() {
        let names: Vec<&str> = boxed().iter().map(|case| case.name()).collect();
        let unique: BTreeSet<&str> = names.iter().copied().collect();
        assert_eq!(
            unique.len(),
            names.len(),
            "a row's name is repeated: {names:?}"
        );
    }

    /// No two legs of the inheritance row bind the same output segment, and none of them binds the
    /// parent's: the row's whole answer rests on each cell owning its own words, and a segment
    /// shared between two legs would compare one leg's answer twice.
    #[test]
    fn the_inheritance_legs_own_disjoint_segments() {
        let mut seen: BTreeSet<u64> = BTreeSet::new();
        for leg in INHERIT_LEGS {
            for segment in leg.segments() {
                assert!(
                    seen.insert(segment),
                    "segment {segment} is bound by two legs"
                );
                assert_ne!(
                    segment, PARENT_SEGMENT,
                    "a leg binds the segment the parent encoder binds"
                );
                assert!(
                    segment < INHERIT_SEGMENTS - 1,
                    "a leg binds the tail segment no draw may reach"
                );
            }
        }
    }

    /// No two legs of the reset row and no two of the residency row write into one region, and the
    /// re-encoded command's segment is one no original command binds.
    #[test]
    fn the_reset_and_usage_legs_own_disjoint_regions() {
        let legs: BTreeSet<u64> = RESET_LEGS.into_iter().map(ResetLeg::at).collect();
        assert_eq!(
            legs.len(),
            RESET_LEGS.len(),
            "two reset legs share a region"
        );
        let usage: BTreeSet<u64> = DECLARATIONS.into_iter().map(Declared::at).collect();
        assert_eq!(
            usage.len(),
            DECLARATIONS.len(),
            "two residency legs share a region"
        );
        const {
            assert!(
                REENCODED_SEGMENT >= RESET_COMMANDS,
                "the re-encoded command binds a segment an original command binds"
            );
            assert!(
                REENCODED_SEGMENT < RESET_SEGMENTS,
                "the re-encoded command binds past its leg's region"
            );
        }
    }

    /// Every parameter window a command of this module binds is inside the parameter buffer, and
    /// the parent's window and the re-encoded command's are windows no other command binds.
    #[test]
    fn every_parameter_window_is_its_own_and_inside_the_buffer() {
        for window in [PARENT_PARAMS, REENCODED_PARAMS] {
            assert!(
                (window as usize) < PARAMS.len(),
                "window {window} is past the parameter buffer"
            );
        }
        // The commands bind windows 0..RESET_COMMANDS; the parent's and the re-encode's lie above.
        const {
            assert!(PARENT_PARAMS >= RESET_COMMANDS - 1);
            assert!(REENCODED_PARAMS >= RESET_COMMANDS);
            assert!(PARENT_PARAMS != REENCODED_PARAMS);
        }
    }

    /// Each reset leg's executed range lies inside the commands it encodes, and the four legs state
    /// four different `(reset, executed)` pairs — which is what makes each one a cell rather than a
    /// repeat.
    #[test]
    fn the_four_reset_legs_are_four_different_pairs() {
        let mut pairs = BTreeSet::new();
        for leg in RESET_LEGS {
            let (first, count) = leg.executed();
            assert!(
                first + count <= RESET_COMMANDS,
                "{} runs past its own commands",
                leg.site()
            );
            if let Some((at, held)) = leg.reset() {
                assert!(
                    at + held <= RESET_COMMANDS,
                    "{} resets past its own commands",
                    leg.site()
                );
            }
            assert!(
                pairs.insert((leg.reset(), leg.executed())),
                "{} states a pair another leg already states",
                leg.site()
            );
        }
    }

    /// The scissor and the viewport the pass row sets each lie inside its target and each leave
    /// texels on **both** sides of themselves, so a leg whose state was dropped reads as a whole
    /// target rather than as the same one.
    #[test]
    fn the_pass_state_clips_strictly_inside_the_target() {
        let side = u64::from(DRAWN_SIDE);
        const {
            assert!(PASS_SCISSOR_BOX.x > 0, "the scissor touches the left edge");
            assert!(
                PASS_SCISSOR_BOX.x + PASS_SCISSOR_BOX.width < DRAWN_SIDE as u64,
                "the scissor touches the right edge"
            );
        }
        let viewport = pass_viewport();
        assert!(
            viewport.origin[1] > 0.0,
            "the viewport touches the top edge"
        );
        assert!(
            viewport.origin[1] + viewport.size[1] < side as f64,
            "the viewport touches the bottom edge"
        );
        // The scissor crosses both commands' halves, so one rectangle clips both draws.
        let half = side / 2;
        assert!(
            PASS_SCISSOR_BOX.x < half,
            "the scissor misses the left half"
        );
        assert!(
            PASS_SCISSOR_BOX.x + PASS_SCISSOR_BOX.width > half,
            "the scissor misses the right half"
        );
    }

    /// The bind-limit row binds at **and** past its declared maximum, and every index it states is
    /// small enough to stay inside its own command's record.
    #[test]
    fn the_bind_limit_row_states_the_boundary_and_one_past_it() {
        const {
            assert!(
                OVER_INDICES[0] == DECLARED_BINDS,
                "the row does not state the index at the maximum"
            );
            assert!(
                OVER_INDICES[1] > DECLARED_BINDS,
                "the row does not state an index past the maximum"
            );
        }
        for index in OVER_INDICES {
            assert!(
                index < 8,
                "index {index} is far enough past the slots to reach another record"
            );
        }
    }
}
