//! What an indirect command buffer's CPU-encoded commands **ran**: compute dispatches, draws of
//! primitives, and the two calls that put a command back as a fresh buffer's.
//!
//! [`super::mesh_commands`] is the one family that reached an indirect command buffer before this
//! one, and it reaches exactly one corner of it: a buffer of **mesh** draws, two commands, a
//! pipeline state and object and mesh buffers. Everything else `MTLIndirectCommandBuffer` offers
//! was untouched — the whole compute command protocol, every ordinary draw, the resets, the
//! barriers, the attribute strides, the stage-in region, and fourteen of the nineteen properties of
//! the descriptor.
//!
//! So each row here encodes a buffer by hand, **executes it on the GPU**, and compares the words
//! the commands' own functions wrote with Metal's. The functions are the three owned `icb_*`
//! templates the runtime's own lane uses, which is what makes the answer readable: each writes a
//! word derived from its bindings, its grid position and its command's own parameters, so a
//! command that bound the wrong offset, ran the wrong grid or did not run at all reads as a
//! different word rather than as an absence.
//!
//! | Row | What it executes |
//! |---|---|
//! | [`COMPUTE_EXECUTED`] | three compute commands: both dispatch kinds, kernel buffers at offsets, threadgroup lengths, a barrier and a cleared barrier |
//! | [`COMPUTE_RESET`] | the same three, with one command `reset` after its dispatch and one put back by the buffer's `resetWithRange:` |
//! | [`COMPUTE_STAGE_IN`] | one compute command that fetches a `[[stage_in]]` attribute at an attribute stride of its own, over a stage-in region it sets |
//! | [`RENDER_EXECUTED`] | three render commands into a buffer: a draw, a draw of a base instance from a vertex start, and an indexed draw of a base vertex |
//! | [`RENDER_DRAWN`] | two render commands that **rasterize**, each with a vertex and a fragment buffer of its own, into one 8 by 8 target, one of them behind a barrier |
//! | [`LEFT_RENDER_STATES`] | nine descriptors, one property each, whose buffers are made and never executed |
//!
//! # What the rows announce, and on which venue
//!
//! Measured on this Apple M3 Max through MoltenVK 1.4.2 and on lavapipe, replaying each row
//! against its recording: **one** row announces notices, and five announce nothing at all.
//!
//! - [`RENDER_DRAWN`] announces `render-pass-split` on both venues, for the barrier its command 1
//!   keeps: the runtime draws an indirect render command with a barrier in a render pass instance
//!   of its own. On lavapipe (LLVM 21.1.8) it announces `shader-color-conversion` beside it, for
//!   the normalized store our side rounds in the shader where the device's own conversion is not
//!   Metal's; the M3 Max announces that one nowhere. Every texel of the row compared exactly on
//!   both venues, so the notices are the whole difference the row has and each declaration loosens
//!   no value.
//! - The five other rows announced nothing on either venue, and **declare nothing**. A declaration
//!   no run was measured provoking is a declaration nobody checked, so none of them carries one,
//!   however plausible the note would be for its shape.
//!
//! Metal's own API validation is the other axis, and it disagrees with plain Metal twice here —
//! which is the whole reason the recording keeps it:
//!
//! - [`RENDER_DRAWN`]: `setBarrier` and `clearBarrier` on a buffer without mesh draws are
//!   reported ("requires DrawMeshThreadgroups or DrawMeshThreads") while plain Metal takes them
//!   and orders the draws.
//! - [`LEFT_RENDER_STATES`]: a descriptor that leaves a render state to its commands is reported
//!   while plain Metal makes the buffer.
//!
//! # Why none of this can hang or fault
//!
//! Every row is bounded by construction and the bounds are small: the largest grid is eight
//! threads, the largest draw six vertices of one instance, the one target is 8 by 8, and no
//! function any row runs holds a loop. Three shapes the device could be hurt by are named and
//! avoided, each on a measurement rather than on caution:
//!
//! - **A render command's own render-state setters** (`setCullMode:`, `setDepthStencilState:` and
//!   the four beside them) raise `APINotSupported` and **end Metal's process** on a device that
//!   does not claim the `Apple10` family, which this oracle does not. The vocabulary has no step
//!   for them at all ([`crate::indirect_commands`]), so no row can reach one. What [`LEFT_RENDER_STATES`]
//!   asks is the *descriptor* half, which plain Metal answers: it makes a buffer that leaves those
//!   states to its commands, and the row observes that it did without ever setting one.
//! - **An inherited pipeline state made without `supportIndirectCommandBuffers`** faults Metal's
//!   GPU at the executing encoder. Every state a command here holds is made by
//!   [`Side::compute_pipeline_for_commands`] or by a [`RenderPipelineSpec::for_indirect_commands`]
//!   pipeline, both of which set the flag, and no row inherits a state.
//! - **A draw a buffer's command types do not hold.** An indexed draw on a buffer of
//!   `MTLIndirectCommandTypeDraw` alone faults plain Metal's GPU, and a draw of primitives on a
//!   buffer of mesh draws alone ends its process. Every row states exactly the types its own
//!   commands use.
//!
//! [`Side::compute_pipeline_for_commands`]: crate::case::Side::compute_pipeline_for_commands
//! [`RenderPipelineSpec::for_indirect_commands`]: crate::render::RenderPipelineSpec::for_indirect_commands
//!
//! Two calls of the surface are reached by rows of their own rather than by the three above.
//! [`COMPUTE_IMAGEBLOCK`] sets each command's own imageblock extent, which its kernel reports
//! back, so a command whose extent was dropped answers the other command's.
//! [`RENDER_VERTEX_STRIDE`] binds one vertex buffer at two different attribute strides under a
//! layout of `MTLBufferLayoutStrideDynamic`, and a side that dropped the per-command stride reads
//! the buffer at 0 — the stride Metal reads a buffer bound without one at — so every vertex of
//! both commands fetches the same element. [`MESH_OBJECT_LENGTHS`] sets each mesh-draw command's
//! own object threadgroup memory lengths, which `icb_object_threadgroup`'s object function reads
//! back by **aliasing**: no builtin reports a length, so a shorter argument 0 letting one array's
//! writes reach the next is the only way a shader can observe one.
//!
//! # What is left, and why
//!
//! - **`drawPatches:…` and `drawIndexedPatches:…`** need a tessellation pipeline, which no side
//!   of this suite builds.
//! - **`gpuResourceID`** and the rows a kernel encodes are [`super::mesh_commands`]' gap, named
//!   there.

use crate::allowance::{Allowance, Justification, Loosening};
use crate::case::STAGE_OBJECT;
use crate::case::{
    call, made, Abort, Case, Failure, Handle, Input, MeshFunctions, MetalSource, Side,
};
use crate::case::{StageInAttribute, StageInLayout, StageInput, STAGE_FRAGMENT, STAGE_VERTEX};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::indirect_commands::{
    ComputeCommandStep, EncodedCommand, IcbDescriptor, RenderCommandStep,
};
use crate::recording::ShaderRef;
use crate::render::{
    ColorState, PassColor, Plane, ReadAt, RenderPass, RenderPipelineSpec, RenderStep, RenderWork,
    StageFunction, TargetSpec, VertexAttribute, VertexBufferLayout, VertexLayout,
};
use crate::transcript::{
    Aspect, ByteLayout, Digest, Element, Gaps, Node, Observed, Region, Sites, Value,
};
use metal2vulkan::metal::{
    AttributeFormat, IndexType, IndirectCommandType, PixelFormat, PrimitiveType, StepFunction,
    VertexFormat,
};

/// The owned fixture whose kernel writes `params[0] * (tid + 1) + params[1]` to `out[tid]`
/// through threadgroup memory, which is what makes a dispatch's grid and both its buffer bindings
/// readable off one word.
const COMPUTE_FIXTURE: &str = "icb_replay_template";

/// The owned fixture whose rasterization-disabled vertex function writes
/// `params[0] * (vid + 1) + params[1] + 1000 * iid` to `out[16 * iid + vid]`, so a draw's vertex
/// start, instance count and base instance are each readable off the words.
const RENDER_FIXTURE: &str = "icb_render_replay_template";

/// The owned fixture whose kernel reports its `[[stage_in]]` attribute beside the stage-in
/// region's origin and size.
const STAGE_IN_FIXTURE: &str = "icb_stage_in_template";

/// The owned fixture whose `tob_rect` draws a rectangle from a buffer's two corners and whose
/// `tob_solid` returns a colour from a buffer: the one pair here that rasterizes, so a render
/// command's **fragment** buffer has something to be read by.
const DRAW_FIXTURE: &str = "tob_renders";

/// The byte every output buffer is filled with before a row runs, so a word no command wrote
/// reads as this rather than as zero — which is what lets a row say that a reset command, or a
/// command outside the executed range, ran nothing.
const UNWRITTEN: u8 = 0xee;

/// The parameter pairs a row's commands bind, one pair a command, each at its own 16-byte offset
/// so a command that bound the wrong offset writes another command's words.
///
/// Four, because the render row has four commands; the compute rows have three and bind the first
/// three pairs.
const PARAMS: [[u32; 2]; 4] = [[3, 100], [5, 200], [7, 300], [11, 400]];

/// Bytes between one command's parameter pair and the next.
const PARAM_STRIDE: u64 = 16;

/// The threadgroup length every compute command sets at index 0, in bytes: four words, which is at
/// or above the largest threadgroup any row dispatches and a multiple of 16, as Metal's encoder
/// requires of a threadgroup length.
const THREADGROUP_BYTES: u32 = 16;

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
///
/// The names are unique among siblings by construction: a command's branch is its index, and a
/// step's leaf is its index and its call.
fn encoding_node(commands: &[EncodedCommand], refused: &[Vec<Option<Failure>>]) -> Node {
    let mut node = Node::branch("encoded", Vec::new());
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
    commands: &[EncodedCommand],
    refused: &[Vec<Option<Failure>>],
) -> Option<(String, Failure)> {
    for (at, answers) in refused.iter().enumerate() {
        for (step, answer) in answers.iter().enumerate() {
            if let Some(failure) = answer {
                let name = commands.get(at).map_or("call", |c| c.step_call(step));
                return Some((
                    format!("encoded/command/{at}/{step}-{name}"),
                    failure.clone(),
                ));
            }
        }
    }
    None
}

/// The sites one `Side::render` program's answer emits: what each piece of the command buffer
/// answered, the status, and the error.
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

/// The descriptor's and the commands' own words, as a case's inputs: what the case feeds its run
/// beside the shader.
fn commands_inputs(descriptor: &IcbDescriptor, commands: &[EncodedCommand]) -> Vec<Input> {
    let mut words = Vec::new();
    for command in commands {
        words.extend(command.words());
        words.push(b'\n');
    }
    vec![
        Input::bytes("icb-descriptor", &descriptor.words()),
        Input::bytes("commands", &words),
    ]
}

// --------------------------------------------------------------------------------------------
// The compute rows

/// The name of the row that executes three compute commands.
pub const COMPUTE_EXECUTED: &str = "indirect-compute-commands-executed";

/// The name of the row that resets one command and one range of them.
pub const COMPUTE_RESET: &str = "indirect-compute-commands-reset";

/// Words one compute command's output segment holds.
const SEGMENT_WORDS: u64 = 8;

/// Bytes one compute command's output segment holds.
const SEGMENT_BYTES: u64 = SEGMENT_WORDS * 4;

/// Words the compute rows' whole output buffer holds: three segments and a tail no command
/// reaches, so a command that wrote past its own segment reads as a word in the tail.
const COMPUTE_WORDS: u64 = SEGMENT_WORDS * 4;

/// What the three compute commands dispatch, in order.
///
/// Command 0 dispatches threadgroups, command 1 the threads form, command 2 threadgroups again,
/// and each writes its own output segment, so the three cannot be told apart by accident: a
/// command that ran another's grid writes a different count of words at a different place.
const COMPUTE_DISPATCHES: [ComputeCommandStep; 3] = [
    ComputeCommandStep::DispatchThreadgroups {
        groups: [2, 1, 1],
        threads: [4, 1, 1],
    },
    ComputeCommandStep::DispatchThreads {
        threads: [6, 1, 1],
        per_group: [3, 1, 1],
    },
    ComputeCommandStep::DispatchThreadgroups {
        groups: [1, 1, 1],
        threads: [2, 1, 1],
    },
];

/// The compute rows' descriptor: both dispatch kinds, two kernel buffers, one threadgroup length,
/// and nothing inherited, so every command carries its own state and bindings.
fn compute_descriptor() -> IcbDescriptor {
    IcbDescriptor {
        command_types: IndirectCommandType::CONCURRENT_DISPATCH
            .union(IndirectCommandType::CONCURRENT_DISPATCH_THREADS),
        max_kernel_buffer_bind_count: 2,
        max_kernel_threadgroup_memory_bind_count: 1,
        ..IcbDescriptor::new()
    }
}

/// Which of the two compute rows a case is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ComputeRow {
    /// Every command runs.
    Executed,
    /// Command 1 is `reset` after its dispatch, and command 2 by the buffer's `resetWithRange:`.
    Reset,
}

/// A row that encodes compute commands and executes them.
pub struct ComputeCommands {
    row: ComputeRow,
}

/// Both compute rows.
fn compute_rows() -> Vec<ComputeCommands> {
    [ComputeRow::Executed, ComputeRow::Reset]
        .into_iter()
        .map(|row| ComputeCommands { row })
        .collect()
}

impl ComputeCommands {
    /// The commands the row encodes, over the parameter buffer `params` and the output buffer
    /// `out`.
    fn commands(&self, pipeline: Handle, params: Handle, out: Handle) -> Vec<EncodedCommand> {
        let mut commands = Vec::new();
        for (at, dispatch) in COMPUTE_DISPATCHES.iter().enumerate() {
            let mut steps = vec![
                ComputeCommandStep::Pipeline(pipeline),
                ComputeCommandStep::KernelBuffer {
                    buffer: params,
                    offset: at as u64 * PARAM_STRIDE,
                    index: 0,
                },
                ComputeCommandStep::KernelBuffer {
                    buffer: out,
                    offset: at as u64 * SEGMENT_BYTES,
                    index: 1,
                },
                ComputeCommandStep::ThreadgroupMemoryLength {
                    length: THREADGROUP_BYTES,
                    index: 0,
                },
            ];
            // Command 1 waits for the commands before it; command 2 asks for a barrier and then
            // takes it back, so both calls are made and the words say the execution still ran.
            match at {
                1 => steps.push(ComputeCommandStep::Barrier),
                2 => {
                    steps.push(ComputeCommandStep::Barrier);
                    steps.push(ComputeCommandStep::ClearBarrier);
                }
                _ => {}
            }
            steps.push(dispatch.clone());
            // The reset row takes command 1's dispatch back after encoding it, so the command
            // holds no dispatch and the execution runs nothing for it.
            if self.row == ComputeRow::Reset && at == 1 {
                steps.push(ComputeCommandStep::Reset);
            }
            commands.push(EncodedCommand::Compute(steps));
        }
        commands
    }
}

impl Case for ComputeCommands {
    fn name(&self) -> &'static str {
        match self.row {
            ComputeRow::Executed => COMPUTE_EXECUTED,
            ComputeRow::Reset => COMPUTE_RESET,
        }
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // Every row reads its answer back through a `Shared` buffer, so on a device without
        // unified memory the runtime announces `shared-in-system-memory`
        // (`crate::replay::shared_memory_notice` holds the whole reason). It declares the
        // notices site alone, so every value of this family still compares against Metal's
        // exactly.
        //
        // Measured on the RTX 5080 Laptop (615.71.09): it fired for
        // the 5 rows of the family the card could run, and each was divergent on that field and
        // nothing else (replay-nvidia.log). On MoltenVK, where
        // the chip's GPU shares the CPU's memory, it cannot fire at all, which is why no
        // recording of this suite declares it.
        vec![crate::replay::shared_memory_notice()]
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        shaders_of(COMPUTE_FIXTURE, COMPUTE_FIXTURE)
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let mut inputs = fixture_inputs(COMPUTE_FIXTURE)?;
        let commands = self.commands(Handle(0), Handle(1), Handle(2));
        inputs.extend(commands_inputs(&compute_descriptor(), &commands));
        inputs.push(Input::bytes("buffer:params", &params_bytes()));
        inputs.push(Input::bytes(
            "buffer:out",
            &vec![UNWRITTEN; COMPUTE_WORDS as usize * 4],
        ));
        if self.row == ComputeRow::Reset {
            inputs.push(Input::bytes("reset-range", b"2..3"));
        }
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
        let unwritten = vec![UNWRITTEN; COMPUTE_WORDS as usize * 4];
        let out = made(sites, "buffer:out", side.buffer(&unwritten))?;

        let commands = self.commands(pipeline, params, out);
        let descriptor = compute_descriptor();
        let encoded = made(
            sites,
            "command-buffer",
            side.indirect_command_buffer(&descriptor, &commands),
        )?;
        sites.push(encoding_node(&commands, &encoded.refused));
        if let Some((at, failure)) = first_refusal(&commands, &encoded.refused) {
            return Err(Abort { at, failure });
        }
        call(
            sites,
            "size",
            side.indirect_commands_size(encoded.commands),
            |size| Value::Uint(*size),
        )?;
        if self.row == ComputeRow::Reset {
            // The buffer's own reset, over command 2 alone, so the row asks both calls that put a
            // command back: the command's `reset` above and this range.
            call(
                sites,
                "reset-with-range",
                side.reset_indirect_commands(encoded.commands, 2, 1),
                |_| Value::Absent,
            )?;
        }

        let work = vec![RenderWork::ComputeCommands {
            commands: encoded.commands,
            first: 0,
            count: commands.len() as u64,
            used: vec![(params, false), (out, true)],
        }];
        let outcome = call(sites, "render", side.render(&work), |outcome| {
            outcome.status.clone()
        })?;
        sites.push(program_node(&work, &outcome));
        if let Some((at, failure)) = first_refused_piece(&work, &outcome) {
            return Err(Abort { at, failure });
        }
        let read = side
            .read_buffer(out, 0, COMPUTE_WORDS as usize * 4)
            .map_err(|failure| Abort {
                at: "out".to_owned(),
                failure,
            })?;
        sites.push(Node::leaf(
            "out",
            Observed::bytes(read, buffer_layout(COMPUTE_WORDS * 4)),
        ));
        Ok(())
    }
}

// --------------------------------------------------------------------------------------------
// The row whose commands set their own imageblock extent

/// The name of the row whose compute commands each set their own imageblock extent.
pub const COMPUTE_IMAGEBLOCK: &str = "indirect-compute-commands-imageblock-extent";

/// The owned fixture the row runs: the kernel whose explicit imageblock each thread stages a mark
/// in, and which reports its neighbour's mark beside the imageblock's own extent.
const IMAGEBLOCK_FIXTURE: &str = "icb_imageblock_template";

/// The threadgroup every command dispatches: two by two threads, so four cells of the imageblock
/// are written and each thread's right-hand neighbour is another thread of its own threadgroup.
///
/// The same for both commands on purpose. The imageblock's extent is the only thing that moves
/// between them, so the words that differ are the extent and nothing else.
const IMAGEBLOCK_THREADS: [u32; 3] = [2, 2, 1];

/// The imageblock extents the row's commands set, one command each: the threadgroup's own extent,
/// and one larger than it.
///
/// The larger one is what tells the call from a side that ignored it: the threadgroup still writes
/// its four cells, so the mark every thread reads back is unchanged, and the only words that move
/// are `get_width()` and `get_height()`. A command whose extent was dropped reports the other
/// command's.
const IMAGEBLOCK_EXTENTS: [[u32; 2]; 2] = [[2, 2], [4, 4]];

/// Words one command's output segment holds: the kernel writes a `uint4` a thread, over the
/// [`IMAGEBLOCK_THREADS`] grid of one threadgroup.
const IMAGEBLOCK_SEGMENT_WORDS: u64 = (IMAGEBLOCK_THREADS[0] * IMAGEBLOCK_THREADS[1]) as u64 * 4;

/// Bytes one command's output segment holds.
const IMAGEBLOCK_SEGMENT_BYTES: u64 = IMAGEBLOCK_SEGMENT_WORDS * 4;

/// Words the row's whole output buffer holds: one segment a command.
const IMAGEBLOCK_WORDS: u64 = IMAGEBLOCK_SEGMENT_WORDS * IMAGEBLOCK_EXTENTS.len() as u64;

/// The row's descriptor: dispatches of threadgroups, one kernel buffer, and nothing inherited.
///
/// No threadgroup-memory bind count: an explicit imageblock is not a threadgroup buffer bound at
/// an index, it is the extent `setImageblockWidth:height:` sets, so the descriptor states none.
fn imageblock_descriptor() -> IcbDescriptor {
    IcbDescriptor {
        command_types: IndirectCommandType::CONCURRENT_DISPATCH,
        max_kernel_buffer_bind_count: 1,
        ..IcbDescriptor::new()
    }
}

/// The commands the row encodes: one an extent, each writing its own segment of `out`.
fn imageblock_commands(pipeline: Handle, out: Handle) -> Vec<EncodedCommand> {
    IMAGEBLOCK_EXTENTS
        .into_iter()
        .enumerate()
        .map(|(at, [width, height])| {
            EncodedCommand::Compute(vec![
                ComputeCommandStep::Pipeline(pipeline),
                ComputeCommandStep::KernelBuffer {
                    buffer: out,
                    offset: at as u64 * IMAGEBLOCK_SEGMENT_BYTES,
                    index: 0,
                },
                ComputeCommandStep::ImageblockSize { width, height },
                ComputeCommandStep::DispatchThreadgroups {
                    groups: [1, 1, 1],
                    threads: IMAGEBLOCK_THREADS,
                },
            ])
        })
        .collect()
}

/// The row whose compute commands each set their own imageblock extent.
pub struct ImageblockCommands;

impl Case for ImageblockCommands {
    fn name(&self) -> &'static str {
        COMPUTE_IMAGEBLOCK
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // As every row of the family: the answer is read back through a `Shared` buffer, so a
        // device without unified memory announces `shared-in-system-memory` and the declaration
        // covers the notices site alone, leaving every value compared exactly.
        vec![crate::replay::shared_memory_notice()]
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        shaders_of(IMAGEBLOCK_FIXTURE, IMAGEBLOCK_FIXTURE)
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let mut inputs = fixture_inputs(IMAGEBLOCK_FIXTURE)?;
        let commands = imageblock_commands(Handle(0), Handle(1));
        inputs.extend(commands_inputs(&imageblock_descriptor(), &commands));
        inputs.push(Input::bytes(
            "buffer:out",
            &vec![UNWRITTEN; IMAGEBLOCK_WORDS as usize * 4],
        ));
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let fixture = Fixture::new(IMAGEBLOCK_FIXTURE);
        let loaded = fixture.loaded().map_err(|why| Abort {
            at: "library".to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        })?;
        let library = made(sites, "library", side.library(&loaded.source()))?;
        let pipeline = made(
            sites,
            "pipeline",
            side.compute_pipeline_for_commands(library, IMAGEBLOCK_FIXTURE, None),
        )?;
        let unwritten = vec![UNWRITTEN; IMAGEBLOCK_WORDS as usize * 4];
        let out = made(sites, "buffer:out", side.buffer(&unwritten))?;

        let commands = imageblock_commands(pipeline, out);
        let descriptor = imageblock_descriptor();
        let encoded = made(
            sites,
            "command-buffer",
            side.indirect_command_buffer(&descriptor, &commands),
        )?;
        sites.push(encoding_node(&commands, &encoded.refused));
        if let Some((at, failure)) = first_refusal(&commands, &encoded.refused) {
            return Err(Abort { at, failure });
        }
        call(
            sites,
            "size",
            side.indirect_commands_size(encoded.commands),
            |size| Value::Uint(*size),
        )?;
        let work = vec![RenderWork::ComputeCommands {
            commands: encoded.commands,
            first: 0,
            count: commands.len() as u64,
            used: vec![(out, true)],
        }];
        let outcome = call(sites, "render", side.render(&work), |outcome| {
            outcome.status.clone()
        })?;
        sites.push(program_node(&work, &outcome));
        if let Some((at, failure)) = first_refused_piece(&work, &outcome) {
            return Err(Abort { at, failure });
        }
        let read = side
            .read_buffer(out, 0, IMAGEBLOCK_WORDS as usize * 4)
            .map_err(|failure| Abort {
                at: "out".to_owned(),
                failure,
            })?;
        sites.push(Node::leaf(
            "out",
            Observed::bytes(read, buffer_layout(IMAGEBLOCK_WORDS * 4)),
        ));
        Ok(())
    }
}

// --------------------------------------------------------------------------------------------
// The stage-in row

/// The name of the row whose command fetches a `[[stage_in]]` attribute at a stride of its own.
pub const COMPUTE_STAGE_IN: &str = "indirect-compute-commands-stage-in-stride";

/// The Metal buffer index the stage-in attribute's layout reads: 0 is the kernel's own `out`.
const STAGE_IN_BUFFER: u32 = 1;

/// The Metal buffer index an indexed stage-in step would read its indices from. No row takes an
/// indexed step, and the descriptor still has to name one.
const STAGE_IN_INDEX_BUFFER: u32 = 2;

/// The attribute stride the command binds its stage-in buffer at: two words an element, so the
/// element's own word and the filler after it are told apart and a stride Metal dropped reads as
/// another element's value.
const ATTRIBUTE_STRIDE: u64 = 8;

/// The stage-in region the command sets: three elements from element 2, which is where the
/// kernel's `grid_origin` and `grid_size` come from and which slot of `out` each thread writes.
const STAGE_IN_ORIGIN: [u32; 3] = [2, 0, 0];
const STAGE_IN_SIZE: [u32; 3] = [3, 1, 1];

/// The attribute values the stage-in buffer holds, one every [`ATTRIBUTE_STRIDE`] bytes, with a
/// filler word between them a correct fetch never reads.
///
/// Five elements, which is one more than the last the stage-in region reaches, so **no thread
/// fetches past the buffer**. The first recording of this row was taken with three, and Metal
/// answered 0 for the two elements past the end — an out-of-bounds read whose value Metal leaves
/// undefined, which is not an answer a case may compare.
/// [`tests::the_attribute_buffer_covers_the_region`] holds the margin.
const ATTRIBUTES: [u32; 10] = [0xa0, 0, 0xa1, 0, 0xa2, 0, 0xa3, 0, 0xa4, 0];

/// Words the stage-in row's output buffer holds: the kernel writes a `uint4` at
/// `origin.x + t`, so six slots cover the region and leave the rest unwritten.
const STAGE_IN_WORDS: u64 = 6 * 4;

/// The stage-in buffer's bytes: [`ATTRIBUTES`] with a filler after each value.
fn attribute_bytes() -> Vec<u8> {
    let mut out = Vec::new();
    for (at, value) in ATTRIBUTES.iter().enumerate() {
        // The filler words are a pattern of their own, so a fetch that read one reads a word no
        // attribute holds.
        let word = if at % 2 == 0 { *value } else { 0x5555_5555 };
        out.extend(word.to_le_bytes());
    }
    out
}

/// The row's stage-input descriptor: one `UInt` attribute at offset 0 of buffer
/// [`STAGE_IN_BUFFER`], whose layout's stride is `MTLBufferLayoutStrideDynamic` — the one stride
/// a command's `setKernelBuffer:offset:attributeStride:atIndex:` is for.
fn stage_input() -> StageInput {
    StageInput::new(
        vec![StageInAttribute {
            index: 0,
            format: AttributeFormat::UInt,
            offset: 0,
            buffer_index: STAGE_IN_BUFFER,
        }],
        vec![StageInLayout {
            buffer_index: STAGE_IN_BUFFER,
            stride: None,
            step_function: StepFunction::ThreadPositionInGridX,
            step_rate: 1,
        }],
        STAGE_IN_INDEX_BUFFER,
    )
}

/// The row's descriptor: dispatches of threads, two kernel buffers, and dynamic attribute
/// strides, which is what lets a command bind a stride at all.
fn stage_in_descriptor() -> IcbDescriptor {
    IcbDescriptor {
        command_types: IndirectCommandType::CONCURRENT_DISPATCH_THREADS,
        max_kernel_buffer_bind_count: 2,
        support_dynamic_attribute_stride: true,
        ..IcbDescriptor::new()
    }
}

/// The one command the row encodes.
fn stage_in_commands(pipeline: Handle, out: Handle, attributes: Handle) -> Vec<EncodedCommand> {
    vec![EncodedCommand::Compute(vec![
        ComputeCommandStep::Pipeline(pipeline),
        ComputeCommandStep::KernelBuffer {
            buffer: out,
            offset: 0,
            index: 0,
        },
        ComputeCommandStep::KernelBufferWithStride {
            buffer: attributes,
            offset: 0,
            stride: ATTRIBUTE_STRIDE,
            index: STAGE_IN_BUFFER,
        },
        ComputeCommandStep::StageInRegion {
            origin: STAGE_IN_ORIGIN,
            size: STAGE_IN_SIZE,
        },
        ComputeCommandStep::DispatchThreads {
            threads: STAGE_IN_SIZE,
            per_group: [1, 1, 1],
        },
    ])]
}

/// The row whose command fetches a stage-in attribute at its own stride.
pub struct StageInCommand;

impl Case for StageInCommand {
    fn name(&self) -> &'static str {
        COMPUTE_STAGE_IN
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // Every row reads its answer back through a `Shared` buffer, so on a device without
        // unified memory the runtime announces `shared-in-system-memory`
        // (`crate::replay::shared_memory_notice` holds the whole reason). It declares the
        // notices site alone, so every value of this family still compares against Metal's
        // exactly.
        //
        // Measured on the RTX 5080 Laptop (615.71.09): it fired for
        // the 5 rows of the family the card could run, and each was divergent on that field and
        // nothing else (replay-nvidia.log). On MoltenVK, where
        // the chip's GPU shares the CPU's memory, it cannot fire at all, which is why no
        // recording of this suite declares it.
        vec![crate::replay::shared_memory_notice()]
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        shaders_of(STAGE_IN_FIXTURE, STAGE_IN_FIXTURE)
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let mut inputs = fixture_inputs(STAGE_IN_FIXTURE)?;
        let commands = stage_in_commands(Handle(0), Handle(1), Handle(2));
        inputs.extend(commands_inputs(&stage_in_descriptor(), &commands));
        inputs.push(Input::bytes("buffer:attributes", &attribute_bytes()));
        inputs.push(Input::bytes(
            "buffer:out",
            &vec![UNWRITTEN; STAGE_IN_WORDS as usize * 4],
        ));
        inputs.push(Input::bytes(
            "stage-input",
            format!(
                "UInt at 0 of buffer {STAGE_IN_BUFFER}, stride dynamic, step \
                 ThreadPositionInGridX rate 1"
            )
            .as_bytes(),
        ));
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let fixture = Fixture::new(STAGE_IN_FIXTURE);
        let loaded = fixture.loaded().map_err(|why| Abort {
            at: "library".to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        })?;
        let library = made(sites, "library", side.library(&loaded.source()))?;
        let pipeline = made(
            sites,
            "pipeline",
            side.compute_pipeline_for_commands(library, STAGE_IN_FIXTURE, Some(&stage_input())),
        )?;
        let attributes = made(sites, "buffer:attributes", side.buffer(&attribute_bytes()))?;
        let unwritten = vec![UNWRITTEN; STAGE_IN_WORDS as usize * 4];
        let out = made(sites, "buffer:out", side.buffer(&unwritten))?;

        let commands = stage_in_commands(pipeline, out, attributes);
        let descriptor = stage_in_descriptor();
        let encoded = made(
            sites,
            "command-buffer",
            side.indirect_command_buffer(&descriptor, &commands),
        )?;
        sites.push(encoding_node(&commands, &encoded.refused));
        if let Some((at, failure)) = first_refusal(&commands, &encoded.refused) {
            return Err(Abort { at, failure });
        }
        call(
            sites,
            "size",
            side.indirect_commands_size(encoded.commands),
            |size| Value::Uint(*size),
        )?;
        let work = vec![RenderWork::ComputeCommands {
            commands: encoded.commands,
            first: 0,
            count: 1,
            used: vec![(attributes, false), (out, true)],
        }];
        let outcome = call(sites, "render", side.render(&work), |outcome| {
            outcome.status.clone()
        })?;
        sites.push(program_node(&work, &outcome));
        if let Some((at, failure)) = first_refused_piece(&work, &outcome) {
            return Err(Abort { at, failure });
        }
        let read = side
            .read_buffer(out, 0, STAGE_IN_WORDS as usize * 4)
            .map_err(|failure| Abort {
                at: "out".to_owned(),
                failure,
            })?;
        sites.push(Node::leaf(
            "out",
            Observed::bytes(read, buffer_layout(STAGE_IN_WORDS * 4)),
        ));
        Ok(())
    }
}

// --------------------------------------------------------------------------------------------
// The render row that writes through a buffer

/// The name of the row that executes three render commands whose vertex function writes a buffer.
pub const RENDER_EXECUTED: &str = "indirect-render-commands-executed";

/// Words one render command's output segment holds: the function writes `out[16 * iid + vid]`,
/// and the row's largest `iid` is 1, so 32 words cover every slot any command of it reaches.
const RENDER_SEGMENT_WORDS: u64 = 32;

/// Bytes one render command's output segment holds.
const RENDER_SEGMENT_BYTES: u64 = RENDER_SEGMENT_WORDS * 4;

/// How many commands the render row encodes: three that draw and a fourth that is `reset` after
/// its draw is encoded, so the execution runs nothing for it and its segment stays unwritten.
const RENDER_COMMANDS: usize = 4;

/// Words the render row's whole output buffer holds: one segment a command.
const RENDER_WORDS: u64 = RENDER_SEGMENT_WORDS * RENDER_COMMANDS as u64;

/// The indices the third render command's indexed draw reads, as 16-bit words: a permutation, so
/// a draw that ignored the index buffer writes other slots.
const INDICES: [u16; 4] = [3, 1, 2, 0];

/// The base vertex the indexed draw adds to every index, so a draw that dropped it writes a
/// different set of slots.
const BASE_VERTEX: i32 = 1;

/// The base instance the second command's draw starts at, which the function multiplies by 1000
/// and which decides which 16 slots of its segment it writes.
const BASE_INSTANCE: u32 = 1;

/// The draws the render row's commands make, in order. The fourth is the same shape as the
/// first, and its command takes it back.
fn render_draws(indices: Handle) -> [RenderCommandStep; RENDER_COMMANDS] {
    [
        // Four points of one instance from vertex 0: slots 0 to 3 of its segment.
        RenderCommandStep::Draw {
            primitive: PrimitiveType::Point,
            first: 0,
            count: 4,
            instances: 1,
            base_instance: 0,
        },
        // Three points from vertex 2, of the instance after the first: slots 18 to 20.
        RenderCommandStep::Draw {
            primitive: PrimitiveType::Point,
            first: 2,
            count: 3,
            instances: 1,
            base_instance: BASE_INSTANCE,
        },
        // Four indexed points, each index raised by the base vertex: slots 1 to 4.
        RenderCommandStep::DrawIndexed {
            primitive: PrimitiveType::Point,
            count: INDICES.len() as u32,
            index_type: IndexType::UInt16,
            indices,
            offset: 0,
            instances: 1,
            base_vertex: BASE_VERTEX,
            base_instance: 0,
        },
        // Four points again, which the command's own `reset` takes back.
        RenderCommandStep::Draw {
            primitive: PrimitiveType::Point,
            first: 0,
            count: 4,
            instances: 1,
            base_instance: 0,
        },
    ]
}

/// The render row's descriptor: draws of primitives and indexed draws — both, because an indexed
/// draw on a buffer that holds only `Draw` faults plain Metal's GPU — two vertex buffers, and
/// nothing inherited.
fn render_descriptor() -> IcbDescriptor {
    IcbDescriptor {
        command_types: IndirectCommandType::DRAW.union(IndirectCommandType::DRAW_INDEXED),
        max_vertex_buffer_bind_count: 2,
        ..IcbDescriptor::new()
    }
}

/// The pipeline the render row's commands hold: the rasterization-disabled vertex function alone,
/// with `supportIndirectCommandBuffers`, which is the one shape Metal takes for a `vertex void`
/// function and the one a command may set.
fn render_pipeline_spec() -> RenderPipelineSpec<'static> {
    let mut spec = RenderPipelineSpec::of(StageFunction::named(RENDER_FIXTURE), None, &[]);
    spec.rasterization = false;
    spec.support_indirect_command_buffers = true;
    spec
}

/// The commands the render row encodes.
fn render_commands(
    pipeline: Handle,
    params: Handle,
    out: Handle,
    indices: Handle,
) -> Vec<EncodedCommand> {
    let mut commands = Vec::new();
    for (at, draw) in render_draws(indices).into_iter().enumerate() {
        let mut steps = vec![
            RenderCommandStep::Pipeline(pipeline),
            RenderCommandStep::VertexBuffer {
                buffer: params,
                offset: at as u64 * PARAM_STRIDE,
                index: 0,
            },
            RenderCommandStep::VertexBuffer {
                buffer: out,
                offset: at as u64 * RENDER_SEGMENT_BYTES,
                index: 1,
            },
            draw,
        ];
        // The last command's draw is taken back after it is encoded, so the command holds none and
        // the execution runs nothing for it.
        if at + 1 == RENDER_COMMANDS {
            steps.push(RenderCommandStep::Reset);
        }
        commands.push(EncodedCommand::Render(steps));
    }
    commands
}

/// The row that executes three render commands whose vertex function writes a buffer.
pub struct RenderCommands;

impl Case for RenderCommands {
    fn name(&self) -> &'static str {
        RENDER_EXECUTED
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // Every row reads its answer back through a `Shared` buffer, so on a device without
        // unified memory the runtime announces `shared-in-system-memory`
        // (`crate::replay::shared_memory_notice` holds the whole reason). It declares the
        // notices site alone, so every value of this family still compares against Metal's
        // exactly.
        //
        // Measured on the RTX 5080 Laptop (615.71.09): it fired for
        // the 5 rows of the family the card could run, and each was divergent on that field and
        // nothing else (replay-nvidia.log). On MoltenVK, where
        // the chip's GPU shares the CPU's memory, it cannot fire at all, which is why no
        // recording of this suite declares it.
        vec![crate::replay::shared_memory_notice()]
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        shaders_of(RENDER_FIXTURE, RENDER_FIXTURE)
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let mut inputs = fixture_inputs(RENDER_FIXTURE)?;
        let commands = render_commands(Handle(0), Handle(1), Handle(2), Handle(3));
        inputs.extend(commands_inputs(&render_descriptor(), &commands));
        inputs.push(Input::bytes(
            "render-pipeline",
            &render_pipeline_spec().words(),
        ));
        inputs.push(Input::bytes("buffer:params", &params_bytes()));
        inputs.push(Input::bytes("buffer:indices", &index_bytes()));
        inputs.push(Input::bytes(
            "buffer:out",
            &vec![UNWRITTEN; RENDER_WORDS as usize * 4],
        ));
        inputs.push(Input::bytes("pass", b"no attachment, 1x1 extent"));
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
            side.render_pipeline(library, &render_pipeline_spec()),
        )?;
        let params = made(sites, "buffer:params", side.buffer(&params_bytes()))?;
        let indices = made(sites, "buffer:indices", side.buffer(&index_bytes()))?;
        let unwritten = vec![UNWRITTEN; RENDER_WORDS as usize * 4];
        let out = made(sites, "buffer:out", side.buffer(&unwritten))?;

        let commands = render_commands(pipeline, params, out, indices);
        let descriptor = render_descriptor();
        let encoded = made(
            sites,
            "command-buffer",
            side.indirect_command_buffer(&descriptor, &commands),
        )?;
        sites.push(encoding_node(&commands, &encoded.refused));
        if let Some((at, failure)) = first_refusal(&commands, &encoded.refused) {
            return Err(Abort { at, failure });
        }
        call(
            sites,
            "size",
            side.indirect_commands_size(encoded.commands),
            |size| Value::Uint(*size),
        )?;

        // A pass of no attachment at its own extent: the function rasterizes nothing and writes
        // through a buffer, so there is no target for it to have and Metal takes the extent from
        // the descriptor.
        let steps = vec![
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
        let read = side
            .read_buffer(out, 0, RENDER_WORDS as usize * 4)
            .map_err(|failure| Abort {
                at: "out".to_owned(),
                failure,
            })?;
        sites.push(Node::leaf(
            "out",
            Observed::bytes(read, buffer_layout(RENDER_WORDS * 4)),
        ));
        Ok(())
    }
}

// --------------------------------------------------------------------------------------------
// The mesh row whose commands set their own object threadgroup memory lengths

/// The name of the row whose mesh-draw commands each set their own object threadgroup memory
/// lengths.
pub const MESH_OBJECT_LENGTHS: &str = "indirect-mesh-commands-object-threadgroup-lengths";

/// The owned fixture the row runs, which `runtime_lane/mesh_commands.rs` measured this shape on
/// first: an object function with threadgroup memory at arguments 0 and 1 and no size of its own
/// for either.
///
/// Its thread 0 writes `0xb` into `b[0]`, then `0xa` into `a[0..63]`, then `b[0]` into word 0 of
/// object buffer 0 and `a[63]` into word 1. So the **length** is what the words read back: where
/// argument 0 is 256 bytes, `a`'s 64 words stay inside it and `b[0]` is still `0xb`; where it is
/// 16 bytes, `a`'s writes reach `b`, which follows it, and `a[63]` — past both — reads 0. No
/// builtin reports a threadgroup memory length, so this aliasing is the only way a shader can
/// observe one at all.
const OBJECT_LENGTH_FIXTURE: &str = "icb_object_threadgroup";

/// The fixture's three stages.
const OBJECT_LENGTH_FUNCTIONS: MeshFunctions<'static> = MeshFunctions {
    object: Some("icb_object_threadgroup_object"),
    mesh: "icb_object_threadgroup_mesh",
    fragment: "icb_object_threadgroup_fragment",
    attachments: 1,
};

/// The lengths the row's two commands set at arguments 0 and 1, with the words each then writes.
///
/// The first keeps `a` off `b` and the second does not, so the two commands answer differently
/// with everything else about them equal: a side that set one command's lengths for both, or none
/// at all, cannot answer both rows. Taken from the lane's own measurement on this M3 Max under
/// macOS 27.0.1.
const OBJECT_LENGTHS: [[u32; 2]; 2] = [[256, 16], [16, 16]];

/// Words one command's slice of the object buffer holds: the function writes words 0 and 1 of its
/// own binding, and a third no command writes, so a command that wrote past its own slice reads
/// as a word in the next one.
const OBJECT_LENGTH_WORDS: u64 = 4;

/// Bytes one command's slice of the object buffer takes.
const OBJECT_LENGTH_BYTES: u64 = OBJECT_LENGTH_WORDS * 4;

/// Words the whole object buffer holds: one slice a command.
const OBJECT_BUFFER_WORDS: u64 = OBJECT_LENGTH_WORDS * OBJECT_LENGTHS.len() as u64;

/// The target the row's pass carries, and the texels it reads back.
///
/// The pass has one because a mesh pipeline's fragment stage writes one, not because anything is
/// drawn into it: the object function launches a grid of `(0, 0, 0)`, so **no** mesh threadgroup
/// runs and every texel must still read the clear. That is the row's second observation, and the
/// one that fails if a side launched a threadgroup the function did not ask for.
const OBJECT_LENGTH_SIDE: u32 = 4;

/// The draw both commands make: one object threadgroup of one thread, 32 mesh threads.
///
/// One object thread because the function's barriers lie where its thread 0 alone reaches them,
/// which both Metal's and Vulkan's rules allow only where every thread of the threadgroup reaches
/// them. The lane measured this M3 Max answering alike with 1 thread and with 32, and an NVIDIA
/// GPU waiting at them for good with 32, so the row states the one count that is defined
/// everywhere.
const OBJECT_LENGTH_DRAW: RenderCommandStep = RenderCommandStep::DrawMeshThreadgroups {
    groups: [1, 1, 1],
    object: [1, 1, 1],
    mesh: [32, 1, 1],
};

/// The row's descriptor: mesh-threadgroup draws, one object and one mesh buffer, and **two**
/// object threadgroup memory lengths, which is what lets a command set either.
fn object_length_descriptor() -> IcbDescriptor {
    IcbDescriptor {
        command_types: IndirectCommandType::DRAW_MESH_THREADGROUPS,
        max_object_buffer_bind_count: 1,
        max_mesh_buffer_bind_count: 1,
        max_object_threadgroup_memory_bind_count: 2,
        ..IcbDescriptor::new()
    }
}

/// The commands the row encodes: one a pair of lengths, each writing its own slice of `out`.
fn object_length_commands(pipeline: Handle, out: Handle) -> Vec<EncodedCommand> {
    OBJECT_LENGTHS
        .into_iter()
        .enumerate()
        .map(|(at, lengths)| {
            let mut steps = vec![
                RenderCommandStep::Pipeline(pipeline),
                RenderCommandStep::ObjectBuffer {
                    buffer: out,
                    offset: at as u64 * OBJECT_LENGTH_BYTES,
                    index: 0,
                },
            ];
            for (index, length) in lengths.into_iter().enumerate() {
                steps.push(RenderCommandStep::ObjectThreadgroupMemoryLength {
                    length,
                    index: index as u32,
                });
            }
            steps.push(OBJECT_LENGTH_DRAW.clone());
            EncodedCommand::Render(steps)
        })
        .collect()
}

/// The row whose mesh-draw commands each set their own object threadgroup memory lengths.
pub struct ObjectLengthCommands;

impl Case for ObjectLengthCommands {
    fn name(&self) -> &'static str {
        MESH_OBJECT_LENGTHS
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // A mesh pass's notices, plus the host step a device announces for commands it draws
        // itself, both named by kind in `mesh_draws`; and the shared-memory note every row of this
        // family carries, since the answer is read back through a `Shared` buffer. Each declares a
        // notices site alone, so every value still compares against Metal's exactly.
        let mut allowances = super::mesh_draws::mesh_notice_allowances();
        allowances.push(crate::replay::shared_memory_notice());
        allowances
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        shaders_of(OBJECT_LENGTH_FIXTURE, OBJECT_LENGTH_FUNCTIONS.mesh)
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let mut inputs = fixture_inputs(OBJECT_LENGTH_FIXTURE)?;
        let commands = object_length_commands(Handle(0), Handle(1));
        inputs.extend(commands_inputs(&object_length_descriptor(), &commands));
        inputs.push(Input::bytes(
            "mesh-pipeline",
            format!(
                "{:?},{},{},{}x {}",
                OBJECT_LENGTH_FUNCTIONS.object,
                OBJECT_LENGTH_FUNCTIONS.mesh,
                OBJECT_LENGTH_FUNCTIONS.fragment,
                OBJECT_LENGTH_FUNCTIONS.attachments,
                crate::case::TARGET_FORMAT
            )
            .as_bytes(),
        ));
        inputs.push(Input::bytes(
            "buffer:out",
            &vec![0u8; OBJECT_BUFFER_WORDS as usize * 4],
        ));
        inputs.push(Input::bytes(
            "render-targets",
            format!(
                "1x {} {OBJECT_LENGTH_SIDE}x{OBJECT_LENGTH_SIDE} cleared to zero",
                crate::case::TARGET_FORMAT
            )
            .as_bytes(),
        ));
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let fixture = Fixture::new(OBJECT_LENGTH_FIXTURE);
        let loaded = fixture.loaded().map_err(|why| Abort {
            at: "library".to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        })?;
        let library = made(sites, "library", side.library(&loaded.source()))?;
        let pipeline = made(
            sites,
            "pipeline",
            side.mesh_pipeline_for_commands(library, &OBJECT_LENGTH_FUNCTIONS),
        )?;
        let target = made(
            sites,
            "target",
            side.render_target(OBJECT_LENGTH_SIDE, OBJECT_LENGTH_SIDE),
        )?;
        let out = made(
            sites,
            "buffer:out",
            side.buffer(&vec![0u8; OBJECT_BUFFER_WORDS as usize * 4]),
        )?;
        let texels = (OBJECT_LENGTH_SIDE * OBJECT_LENGTH_SIDE * crate::case::TARGET_TEXEL) as usize;
        let read_back = made(sites, "buffer:read-back", side.buffer(&vec![0; texels]))?;

        let commands = object_length_commands(pipeline, out);
        let descriptor = object_length_descriptor();
        let encoded = made(
            sites,
            "command-buffer",
            side.indirect_command_buffer(&descriptor, &commands),
        )?;
        sites.push(encoding_node(&commands, &encoded.refused));
        if let Some((at, failure)) = first_refusal(&commands, &encoded.refused) {
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
                buffer: out,
                write: true,
                stages: STAGE_OBJECT,
            },
            RenderStep::ExecuteCommands {
                commands: encoded.commands,
                first: 0,
                count: commands.len() as u64,
            },
        ];
        let work = vec![
            RenderWork::Pass(
                RenderPass::clearing(vec![PassColor::cleared(target, [0.0; 4])]).drawing(steps),
            ),
            RenderWork::ReadTexture {
                texture: target,
                into: read_back,
                offset: 0,
                row_pitch: crate::case::TARGET_TEXEL * OBJECT_LENGTH_SIDE,
                size: [OBJECT_LENGTH_SIDE, OBJECT_LENGTH_SIDE],
                plane: Plane::Whole,
                at: ReadAt::BASE,
            },
        ];
        let outcome = call(sites, "render", side.render(&work), |outcome| {
            outcome.status.clone()
        })?;
        sites.push(program_node(&work, &outcome));
        if let Some((at, failure)) = first_refused_piece(&work, &outcome) {
            return Err(Abort { at, failure });
        }
        let read = side
            .read_buffer(out, 0, OBJECT_BUFFER_WORDS as usize * 4)
            .map_err(|failure| Abort {
                at: "out".to_owned(),
                failure,
            })?;
        sites.push(Node::leaf(
            "out",
            Observed::bytes(read, buffer_layout(OBJECT_BUFFER_WORDS * 4)),
        ));
        let drawn = side
            .read_buffer(read_back, 0, texels)
            .map_err(|failure| Abort {
                at: "target".to_owned(),
                failure,
            })?;
        // Its own name, not the made target's: two sites of one name under one parent is a pair
        // the comparison cannot settle, and it says so rather than guessing.
        sites.push(Node::leaf(
            "target-texels",
            Observed::bytes(drawn, buffer_layout(texels as u64)),
        ));
        Ok(())
    }
}

// --------------------------------------------------------------------------------------------
// The render row whose commands bind a vertex buffer at a stride of their own

/// The name of the row whose render commands bind one vertex buffer at two different attribute
/// strides.
pub const RENDER_VERTEX_STRIDE: &str = "indirect-render-commands-vertex-stride";

/// The owned fixture the row runs: the rasterization-disabled vertex function that fetches one
/// `[[stage_in]]` attribute and writes it with its vertex and instance ids.
const STRIDE_FIXTURE: &str = "icb_render_stride_template";

/// The Metal vertex buffer index the attribute's layout reads. 0 is the function's own `out`,
/// which is a plain buffer argument and no layout of the descriptor, so the two cannot be
/// confused.
const STRIDE_BUFFER: u32 = 1;

/// The attribute values the row's buffer holds, one word each with a filler word between them.
///
/// A filler a correct fetch never reads at [`WIDE_STRIDE`] and **always** reads at
/// [`PACKED_STRIDE`], which is what makes the two commands' answers differ by the stride alone.
const STRIDE_ATTRIBUTES: [u32; 8] = [
    0xb0,
    0x5555_5555,
    0xb1,
    0x5555_5555,
    0xb2,
    0x5555_5555,
    0xb3,
    0x5555_5555,
];

/// The stride the first command binds: two words an element, so each vertex fetches an attribute
/// and no filler.
const WIDE_STRIDE: u64 = 8;

/// The stride the second command binds: one word an element, so every other vertex fetches a
/// filler.
///
/// The same buffer at another stride is the whole row. A side that dropped the per-command stride
/// reads the buffer at 0 — which is what Metal reads a buffer bound without one at — so every
/// vertex of both commands would fetch `0xb0`, and both commands' answers would be wrong and
/// identical.
const PACKED_STRIDE: u64 = 4;

/// Vertices each command draws, which is how many elements it fetches.
const STRIDE_VERTICES: u32 = 4;

/// Words one command's output segment holds: the function writes a `uint2` at `out[vid]`, and
/// each command draws [`STRIDE_VERTICES`] vertices from vertex 0.
const STRIDE_SEGMENT_WORDS: u64 = STRIDE_VERTICES as u64 * 2;

/// Bytes one command's output segment holds.
const STRIDE_SEGMENT_BYTES: u64 = STRIDE_SEGMENT_WORDS * 4;

/// The strides the row's commands bind, one command each.
const STRIDE_COMMANDS: [u64; 2] = [WIDE_STRIDE, PACKED_STRIDE];

/// Words the row's whole output buffer holds: one segment a command.
const STRIDE_WORDS: u64 = STRIDE_SEGMENT_WORDS * STRIDE_COMMANDS.len() as u64;

/// The attribute buffer's bytes: [`STRIDE_ATTRIBUTES`] as 32-bit words.
fn stride_attribute_bytes() -> Vec<u8> {
    STRIDE_ATTRIBUTES
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect()
}

/// The row's vertex descriptor: one `UInt` attribute at offset 0 of buffer [`STRIDE_BUFFER`],
/// whose layout's stride is `MTLBufferLayoutStrideDynamic` — the one layout a command's
/// `setVertexBuffer:offset:attributeStride:atIndex:` is for.
const STRIDE_ATTRIBUTE_LIST: [VertexAttribute; 1] = [VertexAttribute {
    index: 0,
    format: VertexFormat::UInt,
    offset: 0,
    buffer: STRIDE_BUFFER,
}];
const STRIDE_LAYOUTS: [VertexBufferLayout; 1] = [VertexBufferLayout::dynamic_stride(STRIDE_BUFFER)];

/// The pipeline the row's commands hold: the rasterization-disabled vertex function with the
/// dynamic-stride vertex descriptor, and `supportIndirectCommandBuffers` so a command may set it.
fn stride_pipeline_spec() -> RenderPipelineSpec<'static> {
    let mut spec = RenderPipelineSpec::of(StageFunction::named(STRIDE_FIXTURE), None, &[]);
    spec.rasterization = false;
    spec.support_indirect_command_buffers = true;
    spec.vertex_layout = Some(VertexLayout::new(&STRIDE_ATTRIBUTE_LIST, &STRIDE_LAYOUTS));
    spec
}

/// The row's descriptor: draws of primitives, two vertex buffers, and dynamic attribute strides,
/// which is what lets a command bind a stride at all.
fn stride_descriptor() -> IcbDescriptor {
    IcbDescriptor {
        command_types: IndirectCommandType::DRAW,
        max_vertex_buffer_bind_count: 2,
        support_dynamic_attribute_stride: true,
        ..IcbDescriptor::new()
    }
}

/// The commands the row encodes: one a stride, each writing its own segment of `out`.
fn stride_commands(pipeline: Handle, out: Handle, attributes: Handle) -> Vec<EncodedCommand> {
    STRIDE_COMMANDS
        .into_iter()
        .enumerate()
        .map(|(at, stride)| {
            EncodedCommand::Render(vec![
                RenderCommandStep::Pipeline(pipeline),
                RenderCommandStep::VertexBuffer {
                    buffer: out,
                    offset: at as u64 * STRIDE_SEGMENT_BYTES,
                    index: 0,
                },
                RenderCommandStep::VertexBufferWithStride {
                    buffer: attributes,
                    offset: 0,
                    stride,
                    index: STRIDE_BUFFER,
                },
                RenderCommandStep::Draw {
                    primitive: PrimitiveType::Point,
                    first: 0,
                    count: STRIDE_VERTICES,
                    instances: 1,
                    base_instance: 0,
                },
            ])
        })
        .collect()
}

/// The row whose render commands bind one vertex buffer at two different attribute strides.
pub struct StrideCommands;

impl Case for StrideCommands {
    fn name(&self) -> &'static str {
        RENDER_VERTEX_STRIDE
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // As every row of the family: the answer is read back through a `Shared` buffer, so a
        // device without unified memory announces `shared-in-system-memory` and the declaration
        // covers the notices site alone, leaving every value compared exactly.
        vec![crate::replay::shared_memory_notice()]
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        shaders_of(STRIDE_FIXTURE, STRIDE_FIXTURE)
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let mut inputs = fixture_inputs(STRIDE_FIXTURE)?;
        let commands = stride_commands(Handle(0), Handle(1), Handle(2));
        inputs.extend(commands_inputs(&stride_descriptor(), &commands));
        inputs.push(Input::bytes(
            "render-pipeline",
            &stride_pipeline_spec().words(),
        ));
        inputs.push(Input::bytes("buffer:attributes", &stride_attribute_bytes()));
        inputs.push(Input::bytes(
            "buffer:out",
            &vec![UNWRITTEN; STRIDE_WORDS as usize * 4],
        ));
        inputs.push(Input::bytes("pass", b"no attachment, 1x1 extent"));
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let fixture = Fixture::new(STRIDE_FIXTURE);
        let loaded = fixture.loaded().map_err(|why| Abort {
            at: "library".to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        })?;
        let library = made(sites, "library", side.library(&loaded.source()))?;
        let pipeline = made(
            sites,
            "pipeline",
            side.render_pipeline(library, &stride_pipeline_spec()),
        )?;
        let attributes = made(
            sites,
            "buffer:attributes",
            side.buffer(&stride_attribute_bytes()),
        )?;
        let unwritten = vec![UNWRITTEN; STRIDE_WORDS as usize * 4];
        let out = made(sites, "buffer:out", side.buffer(&unwritten))?;

        let commands = stride_commands(pipeline, out, attributes);
        let descriptor = stride_descriptor();
        let encoded = made(
            sites,
            "command-buffer",
            side.indirect_command_buffer(&descriptor, &commands),
        )?;
        sites.push(encoding_node(&commands, &encoded.refused));
        if let Some((at, failure)) = first_refusal(&commands, &encoded.refused) {
            return Err(Abort { at, failure });
        }
        call(
            sites,
            "size",
            side.indirect_commands_size(encoded.commands),
            |size| Value::Uint(*size),
        )?;

        // A pass of no attachment at its own extent, as the family's other render row takes: the
        // function rasterizes nothing and writes through a buffer.
        let steps = vec![
            RenderStep::UseBuffer {
                buffer: attributes,
                write: false,
                stages: STAGE_VERTEX,
            },
            RenderStep::UseBuffer {
                buffer: out,
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
        let read = side
            .read_buffer(out, 0, STRIDE_WORDS as usize * 4)
            .map_err(|failure| Abort {
                at: "out".to_owned(),
                failure,
            })?;
        sites.push(Node::leaf(
            "out",
            Observed::bytes(read, buffer_layout(STRIDE_WORDS * 4)),
        ));
        Ok(())
    }
}

/// The index buffer's bytes: [`INDICES`] as 16-bit words.
fn index_bytes() -> Vec<u8> {
    INDICES.iter().flat_map(|at| at.to_le_bytes()).collect()
}

// --------------------------------------------------------------------------------------------
// The render row that rasterizes

/// The name of the row whose two render commands draw into a target.
pub const RENDER_DRAWN: &str = "indirect-render-commands-drawn";

/// The target's side in texels, and its format: a format whose codes survive a store unchanged,
/// so every texel the row compares is exact.
const DRAWN_SIDE: u32 = 8;
const DRAWN_FORMAT: PixelFormat = PixelFormat::RGBA8Unorm;

/// Bytes one texel of [`DRAWN_FORMAT`] takes.
const DRAWN_TEXEL: u32 = 4;

/// The clear the pass loads its target with: a code neither command's colour holds, so a command
/// that drew nothing reads as this.
const DRAWN_CLEAR: u8 = 7;

/// The two rectangles the commands draw, as `tob_rect` takes them: the two clip-space corners
/// `(x, y)` and `(z, w)`. The left half and the right half, so which command drew which column is
/// unambiguous.
const RECTS: [[f32; 4]; 2] = [[-1.0, -1.0, 0.0, 1.0], [0.0, -1.0, 1.0, 1.0]];

/// The two colours the commands' fragment buffers hold, each channel a whole code over 255 so the
/// store is exact, and every channel of each different from the other's and from the clear.
const COLORS: [[u8; 4]; 2] = [[16, 64, 112, 200], [160, 208, 224, 240]];

/// The draw both rasterizing commands make: two triangles of six vertices, one instance.
const RECT_DRAW: RenderCommandStep = RenderCommandStep::Draw {
    primitive: PrimitiveType::Triangle,
    first: 0,
    count: 6,
    instances: 1,
    base_instance: 0,
};

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

/// Bytes between one command's `float4` and the next, in both the rectangle and the colour
/// buffer.
const FLOAT4_STRIDE: u64 = 16;

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

/// The commands the row encodes: one a rectangle, each binding its own `float4` of the two
/// buffers, so a command that bound the wrong offset draws the other's rectangle or colour.
fn drawn_commands(pipeline: Handle, rects: Handle, colors: Handle) -> Vec<EncodedCommand> {
    (0..RECTS.len() as u64)
        .map(|at| {
            let mut steps = vec![
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
            ];
            // Command 0 asks for a barrier and takes it back; command 1 keeps one. Both calls are
            // made, and neither can change a texel: the two rectangles cover disjoint columns, so
            // what each command draws is determined whichever way the execution orders them or
            // splits its pass. A render command's barrier is the one call here Metal's own API
            // validation stops while plain Metal orders it, which is a fact the recording's
            // second oracle keeps.
            if at == 0 {
                steps.push(RenderCommandStep::Barrier);
                steps.push(RenderCommandStep::ClearBarrier);
            } else {
                steps.push(RenderCommandStep::Barrier);
            }
            steps.push(RECT_DRAW.clone());
            EncodedCommand::Render(steps)
        })
        .collect()
}

/// The layout the target is read under: one row of the layout a row of texels, one element a
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

/// The row whose two render commands rasterize into a target.
pub struct DrawnCommands;

/// The notices this family was measured announcing, and the row each was measured on.
///
/// Both are [`RENDER_DRAWN`]'s, one a venue, and in both cases **every texel of the row still
/// compared exactly** — the notice is the whole difference the row has, so each declaration names
/// a code and loosens no value.
///
/// - **`render-pass-split`**, announced on **both** venues: the runtime draws the barrier
///   command 1 of the row keeps in a render pass instance of its own
///   (`metal2vulkan_runtime`'s `render/executions.rs`).
/// - **`shader-color-conversion`**, announced on **lavapipe** (LLVM 21.1.8) and not on the M3 Max:
///   the row's `RGBA8Unorm` attachment is a normalized store our side rounds in the shader where
///   the device's own conversion is not Metal's.
///
/// Each venue's set is read off the report's **allowance tally**, not its per-case line: the line
/// prints one justification a site, so it names one code where two fired. Measured: the lavapipe
/// run reads "2 named, 2 fired", the M3 Max run "2 named, 1 fired, 1 never fired" naming
/// `shader-color-conversion`. An unfired allowance is reported rather than failed, which is what
/// lets one row carry both venues' codes.
///
/// The other five rows were measured announcing **nothing** on either venue, and **declare
/// nothing**: a declaration no run was measured provoking is a declaration nobody checked.
const DRAWN_NOTICES: [metal2vulkan_runtime::Degradation; 2] = [
    metal2vulkan_runtime::Degradation::RenderPassSplit,
    metal2vulkan_runtime::Degradation::ShaderColorConversion,
];

impl Case for DrawnCommands {
    fn name(&self) -> &'static str {
        RENDER_DRAWN
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // Every row reads its answer back through a `Shared` buffer, so on a device without
        // unified memory the runtime announces `shared-in-system-memory`
        // (`crate::replay::shared_memory_notice` holds the whole reason). It declares the
        // notices site alone, so every value of this family still compares against Metal's
        // exactly.
        //
        // Measured on the RTX 5080 Laptop (615.71.09): it fired for
        // the 5 rows of the family the card could run, and each was divergent on that field and
        // nothing else (replay-nvidia.log). On MoltenVK, where
        // the chip's GPU shares the CPU's memory, it cannot fire at all, which is why no
        // recording of this suite declares it.
        std::iter::once(crate::replay::shared_memory_notice())
            .chain(DRAWN_NOTICES.iter().map(|kind| {
                Allowance::new(
                    format!("case/{}/*", crate::replay::NOTICES_SITE),
                    Loosening::Unchecked,
                    Justification::notice(*kind),
                )
            }))
            .collect()
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        shaders_of(DRAW_FIXTURE, "tob_solid")
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let mut inputs = fixture_inputs(DRAW_FIXTURE)?;
        let commands = drawn_commands(Handle(0), Handle(1), Handle(2));
        inputs.extend(commands_inputs(&drawn_descriptor(), &commands));
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
        let target = made(
            sites,
            "target",
            side.target(&TargetSpec::plain(DRAWN_FORMAT, [DRAWN_SIDE, DRAWN_SIDE])),
        )?;
        let rects = made(sites, "buffer:rects", side.buffer(&rect_bytes()))?;
        let colors = made(sites, "buffer:colors", side.buffer(&color_bytes()))?;
        let texels = (DRAWN_SIDE * DRAWN_SIDE * DRAWN_TEXEL) as usize;
        let read_back = made(sites, "buffer:read-back", side.buffer(&vec![0; texels]))?;

        let commands = drawn_commands(pipeline, rects, colors);
        let descriptor = drawn_descriptor();
        let encoded = made(
            sites,
            "command-buffer",
            side.indirect_command_buffer(&descriptor, &commands),
        )?;
        sites.push(encoding_node(&commands, &encoded.refused));
        if let Some((at, failure)) = first_refusal(&commands, &encoded.refused) {
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
                buffer: rects,
                write: false,
                stages: STAGE_VERTEX,
            },
            RenderStep::UseBuffer {
                buffer: colors,
                write: false,
                stages: STAGE_FRAGMENT,
            },
            RenderStep::ExecuteCommands {
                commands: encoded.commands,
                first: 0,
                count: commands.len() as u64,
            },
        ];
        let work = vec![
            RenderWork::Pass(
                RenderPass::clearing(vec![PassColor::cleared(target, drawn_clear())])
                    .drawing(steps),
            ),
            RenderWork::ReadTexture {
                texture: target,
                into: read_back,
                offset: 0,
                row_pitch: DRAWN_TEXEL * DRAWN_SIDE,
                size: [DRAWN_SIDE, DRAWN_SIDE],
                plane: Plane::Whole,
                at: ReadAt::BASE,
            },
        ];
        let outcome = call(sites, "render", side.render(&work), |outcome| {
            outcome.status.clone()
        })?;
        sites.push(program_node(&work, &outcome));
        if let Some((at, failure)) = first_refused_piece(&work, &outcome) {
            return Err(Abort { at, failure });
        }
        let read = side
            .read_buffer(read_back, 0, texels)
            .map_err(|failure| Abort {
                at: "target".to_owned(),
                failure,
            })?;
        sites.push(Node::leaf(
            "target-texels",
            Observed::bytes(read, drawn_layout()),
        ));
        Ok(())
    }
}

// --------------------------------------------------------------------------------------------
// The descriptor row

/// The name of the row that makes a buffer for each descriptor property no other row sets.
pub const LEFT_RENDER_STATES: &str = "indirect-command-buffer-descriptor-properties";

/// How many commands each descriptor of the row is made for.
const DESCRIPTOR_COMMANDS: u64 = 2;

/// One subject of the descriptor row: the site's name, and what it changes from Metal's defaults.
struct Property {
    name: &'static str,
    change: fn(&mut IcbDescriptor),
}

/// Every property the row asks about, each on a descriptor of its own.
///
/// The six `inherit*` ones are the feature whose **setters** end Metal's process on this oracle.
/// What plain Metal does with the descriptor is another matter and is what this row measures: it
/// makes the buffer. Metal's own API validation stops that creation, which the recording's second
/// oracle keeps beside the behaviour.
const PROPERTIES: &[Property] = &[
    Property {
        name: "inherit-depth-stencil-state",
        change: |d| d.inherit_depth_stencil_state = false,
    },
    Property {
        name: "inherit-depth-bias",
        change: |d| d.inherit_depth_bias = false,
    },
    Property {
        name: "inherit-depth-clip-mode",
        change: |d| d.inherit_depth_clip_mode = false,
    },
    Property {
        name: "inherit-cull-mode",
        change: |d| d.inherit_cull_mode = false,
    },
    Property {
        name: "inherit-front-facing-winding",
        change: |d| d.inherit_front_facing_winding = false,
    },
    Property {
        name: "inherit-triangle-fill-mode",
        change: |d| d.inherit_triangle_fill_mode = false,
    },
    Property {
        name: "support-color-attachment-mapping",
        change: |d| d.support_color_attachment_mapping = true,
    },
    Property {
        name: "max-object-threadgroup-memory-bind-count",
        change: |d| {
            d.command_types = IndirectCommandType::DRAW_MESH_THREADGROUPS;
            d.max_object_buffer_bind_count = 1;
            d.max_object_threadgroup_memory_bind_count = 2;
        },
    },
    Property {
        name: "max-kernel-threadgroup-memory-bind-count",
        change: |d| {
            d.command_types = IndirectCommandType::CONCURRENT_DISPATCH;
            d.max_kernel_buffer_bind_count = 1;
            d.max_kernel_threadgroup_memory_bind_count = 3;
        },
    },
    Property {
        name: "inherit-pipeline-state-and-buffers",
        change: |d| {
            d.inherit_pipeline_state = true;
            d.inherit_buffers = true;
            d.max_vertex_buffer_bind_count = 0;
        },
    },
];

/// The descriptor the row starts each property from: draws of primitives and one vertex buffer, so
/// a buffer of render commands is what each property is asked of unless it says otherwise.
fn property_base() -> IcbDescriptor {
    IcbDescriptor {
        command_types: IndirectCommandType::DRAW,
        max_vertex_buffer_bind_count: 1,
        ..IcbDescriptor::new()
    }
}

/// The descriptor property `at` asks about.
fn property_descriptor(at: usize) -> IcbDescriptor {
    let mut descriptor = property_base();
    if let Some(property) = PROPERTIES.get(at) {
        (property.change)(&mut descriptor);
    }
    descriptor
}

/// The row that makes one buffer for each descriptor property, and executes none of them.
pub struct DescriptorProperties;

impl Case for DescriptorProperties {
    fn name(&self) -> &'static str {
        LEFT_RENDER_STATES
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // Nothing: the case reads a descriptor's own properties back, never a buffer, so the
        // notice a card without unified memory announces for a `Shared` buffer cannot reach it.
        // Measured exact on the RTX 5080 Laptop (615.71.09) and on llvmpipe, which is why it
        // declares nothing where the other five rows of the family declare the venue's notice.
        Vec::new()
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        Ok(Vec::new())
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let mut inputs = Vec::new();
        for (at, property) in PROPERTIES.iter().enumerate() {
            inputs.push(Input::bytes(
                format!("descriptor:{}", property.name),
                &property_descriptor(at).words(),
            ));
        }
        inputs.push(Input::bytes(
            "commands",
            format!("{DESCRIPTOR_COMMANDS} commands, none encoded and none executed").as_bytes(),
        ));
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        // A buffer of no command at all, so nothing here can be executed even by accident: the
        // case makes each buffer, reads its size, and stops.
        let none: [EncodedCommand; 0] = [];
        for (at, property) in PROPERTIES.iter().enumerate() {
            let descriptor = property_descriptor(at);
            let mut node = Node::branch(property.name, Vec::new());
            match side.indirect_command_buffer(&descriptor, &none) {
                Ok(made) => {
                    node = node.with(Node::leaf(
                        "made",
                        Observed::value(Value::ok(Value::Absent)),
                    ));
                    let size = side
                        .indirect_commands_size(made.commands)
                        .map(|size| Value::ok(Value::Uint(size)))
                        .unwrap_or_else(|failure| failure.recorded());
                    node = node.with(Node::leaf("size", Observed::value(size)));
                    sites.push(node);
                }
                Err(failure) => {
                    node = node.with(Node::leaf("made", Observed::value(failure.recorded())));
                    if let Some(needs) = &failure.lacks {
                        sites.device_gap(&format!("{}/made", property.name), needs.clone());
                    }
                    sites.push(node);
                    return Err(Abort {
                        at: format!("{}/made", property.name),
                        failure,
                    });
                }
            }
        }
        Ok(())
    }
}

/// Every row of the family, as the case set holds them, in the order the report lists them.
pub fn boxed() -> Vec<Box<dyn Case>> {
    let mut cases: Vec<Box<dyn Case>> = Vec::new();
    for case in compute_rows() {
        cases.push(Box::new(case));
    }
    cases.push(Box::new(ImageblockCommands));
    cases.push(Box::new(StageInCommand));
    cases.push(Box::new(RenderCommands));
    cases.push(Box::new(StrideCommands));
    cases.push(Box::new(ObjectLengthCommands));
    cases.push(Box::new(DrawnCommands));
    cases.push(Box::new(DescriptorProperties));
    cases
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every row's name is distinct, since a name is what a recording is keyed by.
    #[test]
    fn every_row_has_its_own_name() {
        let names: Vec<&str> = boxed().iter().map(|case| case.name()).collect();
        let unique: std::collections::BTreeSet<&str> = names.iter().copied().collect();
        assert_eq!(names.len(), unique.len(), "a name is repeated: {names:?}");
    }

    /// No step of any row is one of the six render-state setters, which end Metal's process on
    /// this oracle. The vocabulary has no variant for them, so this is a guard on the *vocabulary*
    /// rather than on the rows: it fails the day one is added, which is the moment somebody has to
    /// read the measurement behind the omission.
    #[test]
    fn no_render_state_setter_is_in_the_vocabulary() {
        let source = include_str!("../indirect_commands.rs");
        for call in [
            "setDepthStencilState",
            "setDepthBias",
            "setDepthClipMode",
            "setCullMode",
            "setFrontFacingWinding",
            "setTriangleFillMode",
        ] {
            assert!(
                !source.contains(&format!("=> \"{call}\"")),
                "`{call}` has a step: it raises APINotSupported and ends Metal's process on a \
                 device that does not claim Apple10, which this suite's oracle does not"
            );
        }
    }

    /// Every command of every row sets its own pipeline state, or its buffer inherits one.
    ///
    /// An execution whose inherited state was made without `supportIndirectCommandBuffers` faults
    /// Metal's GPU at the encoder, and a command that sets none on a buffer that inherits none is
    /// the shape that reaches it. Checked over the rows' own commands rather than argued.
    #[test]
    fn every_executed_command_sets_a_pipeline_state() {
        let compute = compute_rows();
        let mut lists: Vec<Vec<EncodedCommand>> = compute
            .iter()
            .map(|row| row.commands(Handle(0), Handle(1), Handle(2)))
            .collect();
        lists.push(stage_in_commands(Handle(0), Handle(1), Handle(2)));
        lists.push(render_commands(Handle(0), Handle(1), Handle(2), Handle(3)));
        lists.push(drawn_commands(Handle(0), Handle(1), Handle(2)));
        for commands in &lists {
            for (at, command) in commands.iter().enumerate() {
                let sets = match command {
                    EncodedCommand::Compute(steps) => steps
                        .iter()
                        .any(|step| matches!(step, ComputeCommandStep::Pipeline(_))),
                    EncodedCommand::Render(steps) => steps
                        .iter()
                        .any(|step| matches!(step, RenderCommandStep::Pipeline(_))),
                };
                assert!(sets, "command {at} sets no pipeline state: {command:?}");
            }
        }
    }

    /// Every parameter offset a command binds lies inside the parameter buffer, so no command's
    /// kernel or vertex function reads past it. A read past the end is a value Metal leaves
    /// undefined, which is not an answer a case may compare.
    #[test]
    fn every_command_binds_a_parameter_pair_inside_the_buffer() {
        let bytes = params_bytes().len() as u64;
        let most = COMPUTE_DISPATCHES.len().max(RENDER_COMMANDS) as u64;
        assert!(
            most * PARAM_STRIDE <= bytes,
            "{most} commands at stride {PARAM_STRIDE} read past a buffer of {bytes} bytes"
        );
        assert!(
            most <= PARAMS.len() as u64,
            "{most} commands and {} parameter pairs",
            PARAMS.len()
        );
    }

    /// The render row's last command holds a `reset` after its draw, so the execution runs nothing
    /// for it and its segment of the output stays unwritten. That is the row's whole answer about
    /// `reset` on a render command, so a change that dropped the step would make the row say
    /// nothing about it.
    #[test]
    fn the_last_render_command_is_reset_after_its_draw() {
        let commands = render_commands(Handle(0), Handle(1), Handle(2), Handle(3));
        assert_eq!(commands.len(), RENDER_COMMANDS);
        let EncodedCommand::Render(steps) = &commands[RENDER_COMMANDS - 1] else {
            panic!("the last command is not a render command");
        };
        let last = steps.last().expect("a command of no step");
        assert_eq!(*last, RenderCommandStep::Reset, "the reset is not last");
        assert!(
            steps
                .iter()
                .any(|step| matches!(step, RenderCommandStep::Draw { .. })),
            "nothing is reset: the command holds no draw"
        );
        for (at, command) in commands.iter().enumerate().take(RENDER_COMMANDS - 1) {
            let EncodedCommand::Render(steps) = command else {
                panic!("command {at} is not a render command");
            };
            assert!(
                !steps.contains(&RenderCommandStep::Reset),
                "command {at} is reset too, so the row has fewer draws than it says"
            );
        }
    }

    /// The parameter buffer holds each command's pair at its own offset, which is what makes a
    /// command that bound the wrong offset write another's words.
    #[test]
    fn the_parameter_buffer_holds_each_pair_at_its_own_offset() {
        let bytes = params_bytes();
        for (at, pair) in PARAMS.iter().enumerate() {
            let start = at * PARAM_STRIDE as usize;
            for (word, value) in pair.iter().enumerate() {
                let found = u32::from_le_bytes(
                    bytes[start + word * 4..start + word * 4 + 4]
                        .try_into()
                        .expect("four bytes"),
                );
                assert_eq!(found, *value, "parameter {word} of command {at}");
            }
        }
    }

    /// No command of the compute rows writes outside its own segment, and the buffer has a tail
    /// no command reaches, so a write past a segment reads as a word in the tail.
    #[test]
    fn every_compute_segment_fits_with_a_tail_to_spare() {
        let largest = COMPUTE_DISPATCHES
            .iter()
            .map(|dispatch| match dispatch {
                ComputeCommandStep::DispatchThreadgroups { groups, threads } => {
                    u64::from(groups[0]) * u64::from(threads[0])
                }
                ComputeCommandStep::DispatchThreads { threads, .. } => u64::from(threads[0]),
                other => panic!("{other:?} is no dispatch"),
            })
            .max()
            .expect("three dispatches");
        assert!(
            largest <= SEGMENT_WORDS,
            "a dispatch of {largest} threads writes past its {SEGMENT_WORDS}-word segment"
        );
        assert!(
            COMPUTE_WORDS > SEGMENT_WORDS * COMPUTE_DISPATCHES.len() as u64,
            "the compute output buffer has no tail past its segments"
        );
    }

    /// Every threadgroup length the compute rows set covers the largest threadgroup the row
    /// dispatches, so no thread of any command stages its word outside the memory the command
    /// asked for.
    #[test]
    fn the_threadgroup_length_covers_every_threadgroup() {
        let widest = COMPUTE_DISPATCHES
            .iter()
            .map(|dispatch| match dispatch {
                ComputeCommandStep::DispatchThreadgroups { threads, .. } => threads[0],
                ComputeCommandStep::DispatchThreads { per_group, .. } => per_group[0],
                other => panic!("{other:?} is no dispatch"),
            })
            .max()
            .expect("three dispatches");
        assert!(
            THREADGROUP_BYTES >= widest * 4,
            "{THREADGROUP_BYTES} bytes hold fewer than the {widest} words a threadgroup stages"
        );
        assert_eq!(
            THREADGROUP_BYTES % 16,
            0,
            "Metal takes a threadgroup length in multiples of 16 bytes"
        );
    }

    /// Every slot the render row's draws write lies inside the segment the command bound, which is
    /// what keeps one command's draw out of another's words.
    #[test]
    fn every_render_draw_writes_inside_its_segment() {
        for (at, draw) in render_draws(Handle(0)).iter().enumerate() {
            let slots: Vec<u64> = match draw {
                RenderCommandStep::Draw {
                    first,
                    count,
                    instances,
                    base_instance,
                    ..
                } => (0..*instances)
                    .flat_map(|instance| {
                        let iid = u64::from(base_instance + instance);
                        (0..*count).map(move |step| 16 * iid + u64::from(first + step))
                    })
                    .collect(),
                RenderCommandStep::DrawIndexed {
                    base_vertex,
                    base_instance,
                    ..
                } => INDICES
                    .iter()
                    .map(|index| {
                        16 * u64::from(*base_instance)
                            + (i64::from(*index) + i64::from(*base_vertex)) as u64
                    })
                    .collect(),
                other => panic!("{other:?} is no draw"),
            };
            for slot in slots {
                assert!(
                    slot < RENDER_SEGMENT_WORDS,
                    "command {at} writes slot {slot} of a {RENDER_SEGMENT_WORDS}-word segment"
                );
            }
        }
    }

    /// The unwritten byte is a word no command of any row can write, so a slot that reads it was
    /// reached by nothing. The compute kernel's word is `params[0] * (tid + 1) + params[1]`, which
    /// for the row's parameters is far below it, and the render function's adds at most 1000 to
    /// one of the same.
    #[test]
    fn no_command_can_write_the_unwritten_word() {
        let unwritten = u32::from_le_bytes([UNWRITTEN; 4]);
        for pair in PARAMS {
            for tid in 0..SEGMENT_WORDS as u32 {
                let computed = pair[0] * (tid + 1) + pair[1];
                assert_ne!(computed, unwritten, "a dispatch writes the sentinel");
                assert_ne!(computed + 1000 * 2, unwritten, "a draw writes the sentinel");
            }
        }
    }

    /// Each rasterizing command's rectangle covers half the target's columns and the two do not
    /// overlap, so which command drew which column is a fact about the geometry.
    #[test]
    fn the_two_rectangles_split_the_target() {
        let [left, right] = RECTS;
        assert_eq!(left[2], right[0], "the rectangles leave a gap or overlap");
        assert_eq!(left[0], -1.0);
        assert_eq!(right[2], 1.0);
        assert_eq!(DRAWN_SIDE % 2, 0, "an odd side has no half");
    }

    /// Every colour the rasterizing row draws differs from the other and from the clear in every
    /// channel, so a texel of the wrong command's colour cannot read as the right one's.
    #[test]
    fn every_drawn_colour_is_distinct_in_every_channel() {
        let [first, second] = COLORS;
        for channel in 0..4 {
            assert_ne!(first[channel], second[channel], "channel {channel}");
            assert_ne!(
                first[channel], DRAWN_CLEAR,
                "channel {channel} is the clear"
            );
            assert_ne!(
                second[channel], DRAWN_CLEAR,
                "channel {channel} is the clear"
            );
        }
    }

    /// Every property of the descriptor row changes the base descriptor's words, so no subject of
    /// it is asking the same question as the base.
    #[test]
    fn every_descriptor_property_changes_the_descriptor() {
        let base = property_base().words();
        for (at, property) in PROPERTIES.iter().enumerate() {
            assert_ne!(
                property_descriptor(at).words(),
                base,
                "{} changes nothing",
                property.name
            );
        }
    }

    /// The descriptor row's sites are distinct among their siblings, which is what keeps a
    /// comparison from reading two subjects as one.
    #[test]
    fn every_descriptor_property_has_its_own_site() {
        let names: std::collections::BTreeSet<&str> =
            PROPERTIES.iter().map(|property| property.name).collect();
        assert_eq!(names.len(), PROPERTIES.len(), "a property's name repeats");
    }

    /// The stage-in row's attribute stride skips the filler word, so a fetch Metal made at the
    /// pipeline's own stride reads a different value than a fetch at the command's.
    #[test]
    fn the_attribute_stride_skips_the_filler() {
        assert_eq!(ATTRIBUTE_STRIDE, 8, "the stride is two words");
        let bytes = attribute_bytes();
        for element in 0..ATTRIBUTES.len() / 2 {
            let at = element * ATTRIBUTE_STRIDE as usize;
            let found = u32::from_le_bytes(bytes[at..at + 4].try_into().expect("four bytes"));
            assert_eq!(found, ATTRIBUTES[element * 2], "element {element}");
        }
    }

    /// Every attribute element the stage-in region reaches lies inside the buffer the command
    /// binds, so no thread fetches past it. A fetch past the end is a read Metal leaves undefined,
    /// and the row's first recording held two of them.
    #[test]
    fn the_attribute_buffer_covers_the_region() {
        let last = u64::from(STAGE_IN_ORIGIN[0] + STAGE_IN_SIZE[0] - 1);
        let needed = (last + 1) * ATTRIBUTE_STRIDE;
        assert!(
            attribute_bytes().len() as u64 >= needed,
            "element {last} at stride {ATTRIBUTE_STRIDE} needs {needed} bytes, and the buffer \
             holds {}",
            attribute_bytes().len()
        );
        let elements = attribute_bytes().len() as u64 / ATTRIBUTE_STRIDE;
        assert!(
            elements * 2 <= ATTRIBUTES.len() as u64,
            "an element lies past the values the buffer states"
        );
    }

    /// Every slot the stage-in command's threads write lies inside its output buffer.
    #[test]
    fn the_stage_in_region_fits_the_output() {
        let last = u64::from(STAGE_IN_ORIGIN[0] + STAGE_IN_SIZE[0] - 1);
        assert!(
            (last + 1) * 4 <= STAGE_IN_WORDS,
            "slot {last} of a uint4 lies past the {STAGE_IN_WORDS}-word buffer"
        );
    }
}
