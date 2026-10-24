//! The blit encoder's three indirect-command-buffer calls, each over a sweep of ranges.
//!
//! `MTLBlitCommandEncoder` has twenty-two implemented items and the suite reached nineteen of
//! them; the three left are the ones that take an `MTLIndirectCommandBuffer`:
//! `resetCommandsInBuffer:withRange:`, `copyIndirectCommandBuffer:sourceRange:destination:
//! destinationIndex:` and `optimizeIndirectCommandBuffer:withRange:`. Each is one case here, and
//! each case sweeps the same four ranges (`LEGS`).
//!
//! # What makes the answer readable
//!
//! Every case encodes a buffer of `COMMANDS` compute commands by hand, out of the owned
//! `icb_replay_template` fixture the runtime's own lane uses: command `c` binds the parameter pair
//! at `c · 16` and its output segment at `c · 32`, and its kernel writes
//! `params[0] · (tid + 1) + params[1]` into that segment's first four words. So the four commands
//! write four different sequences into four different places, and the buffer is made holding
//! `UNWRITTEN` — never zero — so a segment no command wrote reads as a byte the case put there.
//!
//! The case then encodes its blit call over one range, **executes the whole buffer**, and reads
//! the output. What each leg's site holds is therefore *which commands still ran*, which is the
//! only thing these three calls can be observed by:
//!
//! - a reset command runs nothing, so its segment stays `UNWRITTEN`;
//! - a copied command carries its **own bindings** with it, so a command copied from source index
//!   2 into destination index 1 writes segment **2** of the destination's output when the
//!   destination's command 1 runs. A copy that moved the index but not the state, or the state but
//!   not the parameters, writes a different segment or different words.
//!
//! The reset and the optimize execute the whole buffer; **the copy executes only the commands it
//! said it was writing** (`Call::executed_range`), and that is a correction a falsification
//! forced. Executing the whole destination made `destinationIndex` unobservable — a copied command
//! runs wherever it sits — and patching the runtime to ignore that argument left every byte of
//! this case unchanged. Executing `into_index` up instead means a copy that landed elsewhere runs
//! an unencoded command and writes nothing, and the patched runtime then parts the leg at once.
//!
//! # Depth, and what a swept range cannot tell apart
//!
//! The ranges are the boundaries of a count rather than an enum: **0 commands, 1 command, 1
//! command at a non-zero first index, and every command of the buffer**. The first index is swept
//! beside the count because the two arguments are one `NSRange` and a side that read only the
//! count would pass a sweep of counts alone.
//!
//! The ranges part the observation for two of the three calls: a reset of none leaves four
//! segments written and a reset of all four leaves none, and no two legs of either case hold the
//! same bytes. The copy's `destinationIndex` is swept beside them and is observable only because
//! of the executed range named below. **For `optimizeIndirectCommandBuffer:withRange:` they
//! cannot**, and the case says
//! so rather than implying otherwise: nothing Metal's optimize changes is observable by a getter
//! or by an execution, so all four of that case's legs hold the same four written segments *by
//! construction*. Its teeth are not that the legs differ but that they are all still **written**:
//! an implementation that took the call as a reset, or that dropped the commands in its range,
//! fails every leg at once. That is a real failure mode for a call whose correct answer is to do
//! nothing, and it is why the row exists; it is not a claim that four values were distinguished.
//!
//! # Why none of this can hang or fault
//!
//! Every command dispatches one threadgroup of `THREADS` threads, which is inside one simdgroup
//! and inside the smallest `maxTotalThreadsPerThreadgroup` any venue reports, and no grid is
//! derived from a device limit. The fixture holds no loop. Every command's bindings lie inside
//! their buffers (`every_command_binds_inside_its_buffers`), every swept range lies inside its
//! buffer's commands (`every_range_lies_inside_its_buffer`), and every pipeline state is made by
//! [`Side::compute_pipeline_for_commands`], which sets `supportIndirectCommandBuffers` — a state
//! made without it faults Metal's GPU at the executing encoder.
//!
//! **A range past a buffer's commands is left out on purpose.** Plain Metal takes one and Metal's
//! own validation stops neither; a copy that runs past its destination corrupts that buffer's
//! command 0. That is a corruption rather than an answer, our side refuses it, and asking a shared
//! GPU to execute a corrupted command buffer is not a comparison worth taking. The refusal is in
//! the runtime's own documented surface, not here.
//!
//! [`Side::compute_pipeline_for_commands`]: crate::case::Side::compute_pipeline_for_commands

use crate::allowance::Allowance;
use crate::case::{made, Abort, Case, Failure, Handle, Input, MetalSource, Side};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::indirect_commands::{ComputeCommandStep, EncodedCommand, IcbDescriptor};
use crate::recording::ShaderRef;
use crate::render::RenderWork;
use crate::transcript::{ByteLayout, Digest, Element, Node, Observed, Sites};
use metal2vulkan::metal::IndirectCommandType;
use metal2vulkan_runtime::Degradation;

/// The owned fixture whose kernel writes `params[0] * (tid + 1) + params[1]` to `out[tid]`.
const FIXTURE: &str = "icb_replay_template";

/// Commands every buffer of every case holds.
const COMMANDS: u64 = 4;

/// Threads each command dispatches, in one threadgroup: inside one simdgroup on every Apple GPU.
const THREADS: u32 = 4;

/// Words one command's output segment holds: twice what its dispatch writes, so a command that
/// wrote past its own grid shows inside its own segment.
const SEGMENT_WORDS: u64 = 8;

/// Bytes one command's output segment holds.
const SEGMENT_BYTES: u64 = SEGMENT_WORDS * 4;

/// Words one output buffer holds: a segment a command, and one more segment no command reaches, so
/// a command that wrote past the last segment shows in the tail.
const OUTPUT_WORDS: u64 = SEGMENT_WORDS * (COMMANDS + 1);

/// The byte every output buffer is made holding, so a word no command wrote reads as a value the
/// case put there rather than as zero — which is what lets a leg say a command ran nothing.
const UNWRITTEN: u8 = 0xee;

/// Bytes between one command's parameter pair and the next, which Metal's kernel-buffer offset
/// takes.
const PARAM_STRIDE: u64 = 16;

/// The parameter pairs the commands bind, one pair a command: no two commands' words coincide, so
/// a command that bound another's pair writes another's sequence.
const PARAMS: [[u32; 2]; COMMANDS as usize] = [[3, 100], [5, 200], [7, 300], [11, 400]];

/// The threadgroup length every command sets at index 0: four words, at or above the largest
/// threadgroup any command dispatches and a multiple of 16, as Metal's encoder requires.
const THREADGROUP_BYTES: u32 = 16;

/// One leg of a case's sweep: a range, and where a copy puts it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Leg {
    /// The leg's site name, which says the range in words.
    name: &'static str,
    /// The range's first command.
    first: u64,
    /// The commands in the range.
    count: u64,
    /// The destination command a copy's range lands at. Read by the copy case alone.
    into_index: u64,
}

/// The four ranges every case sweeps: none, one, one at a non-zero first index, and every command.
const LEGS: [Leg; 4] = [
    Leg {
        name: "range-none",
        first: 0,
        count: 0,
        into_index: 0,
    },
    Leg {
        name: "range-one-at-0",
        first: 0,
        count: 1,
        into_index: 0,
    },
    Leg {
        name: "range-one-at-2",
        first: 2,
        count: 1,
        into_index: 1,
    },
    Leg {
        name: "range-whole",
        first: 0,
        count: COMMANDS,
        into_index: 0,
    },
];

/// Which of the three calls a case is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Call {
    /// `resetCommandsInBuffer:withRange:`.
    Reset,
    /// `copyIndirectCommandBuffer:sourceRange:destination:destinationIndex:`.
    Copy,
    /// `optimizeIndirectCommandBuffer:withRange:`.
    Optimize,
}

impl Call {
    /// The case's name: its recording's file name, and its row in the report.
    const fn case(self) -> &'static str {
        match self {
            Call::Reset => "blit-reset-commands-in-buffer",
            Call::Copy => "blit-copy-indirect-command-buffer",
            Call::Optimize => "blit-optimize-indirect-command-buffer",
        }
    }

    /// Whether the call needs a second buffer of its own to land in.
    const fn has_destination(self) -> bool {
        matches!(self, Call::Copy)
    }

    /// Whether the call writes a buffer's records on the GPU, which makes the runtime replay a
    /// later execution of them through its host step and announce it once.
    ///
    /// The optimize call records nothing, so it alone does not.
    const fn replays_on_the_host(self) -> bool {
        matches!(self, Call::Reset | Call::Copy)
    }

    /// The piece of work the call encodes over `leg`, out of the leg's source buffer and, where
    /// the call has one, its destination.
    fn work(self, source: Handle, destination: Handle, leg: Leg) -> RenderWork {
        match self {
            Call::Reset => RenderWork::ResetCommands {
                commands: source,
                first: leg.first,
                count: leg.count,
            },
            Call::Copy => RenderWork::CopyCommands {
                from: source,
                first: leg.first,
                count: leg.count,
                into: destination,
                into_index: leg.into_index,
            },
            Call::Optimize => RenderWork::OptimizeCommands {
                commands: source,
                first: leg.first,
                count: leg.count,
            },
        }
    }

    /// Which of a leg's two buffers the case executes: the destination where the call has one,
    /// else the buffer it acted on.
    const fn executed(self, source: Handle, destination: Handle) -> Handle {
        if self.has_destination() {
            destination
        } else {
            source
        }
    }

    /// The commands the case executes after the call, which is **not** the whole buffer for the
    /// copy.
    ///
    /// The reset and the optimize act on the buffer they execute, so executing all of it is what
    /// says which commands survived. The copy does not: a copied command carries its own bindings,
    /// so executing the whole destination runs it wherever it landed and `destinationIndex`
    /// reaches nothing a site can read. A falsification showed exactly that — ignoring
    /// `destinationIndex` in the runtime left every byte of this case unchanged — so the copy
    /// executes the commands the call said it was writing, `into_index` up, and a copy that landed
    /// elsewhere runs an unencoded command there and writes nothing.
    ///
    /// A leg of no commands still executes one, so every leg states a range of the same shape;
    /// the command it runs is unencoded and writes nothing, which is that leg's answer anyway.
    const fn executed_range(self, leg: Leg) -> (u64, u64) {
        if self.has_destination() {
            (leg.into_index, if leg.count == 0 { 1 } else { leg.count })
        } else {
            (0, COMMANDS)
        }
    }
}

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

/// One output buffer's bytes before a leg runs.
fn output_bytes() -> Vec<u8> {
    vec![UNWRITTEN; OUTPUT_WORDS as usize * 4]
}

/// The descriptor every buffer of every case is made with.
///
/// One descriptor for a case's source and its destination: a copy between buffers of **different**
/// descriptors is garbled on plain Metal and its validation stops it nowhere, so no case here
/// states two.
fn descriptor() -> IcbDescriptor {
    IcbDescriptor {
        command_types: IndirectCommandType::CONCURRENT_DISPATCH,
        max_kernel_buffer_bind_count: 2,
        max_kernel_threadgroup_memory_bind_count: 1,
        ..IcbDescriptor::new()
    }
}

/// The [`COMMANDS`] commands a leg's source buffer holds, over the parameter buffer `params` and
/// the leg's own output buffer `out`.
fn encoded(pipeline: Handle, params: Handle, out: Handle) -> Vec<EncodedCommand> {
    (0..COMMANDS)
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
                    offset: at * SEGMENT_BYTES,
                    index: 1,
                },
                ComputeCommandStep::ThreadgroupMemoryLength {
                    length: THREADGROUP_BYTES,
                    index: 0,
                },
                ComputeCommandStep::DispatchThreadgroups {
                    groups: [1, 1, 1],
                    threads: [THREADS, 1, 1],
                },
            ])
        })
        .collect()
}

/// A destination buffer's [`COMMANDS`] commands, none of them encoded: a buffer of the case's own
/// descriptor whose commands are all as a fresh buffer's, so every word of its output is the
/// copy's doing and nothing else's.
fn unencoded() -> Vec<EncodedCommand> {
    (0..COMMANDS)
        .map(|_| EncodedCommand::Compute(Vec::new()))
        .collect()
}

/// A case of the family: one of the three calls, over every leg of `LEGS`.
pub struct BlitIndirectCommands {
    call: Call,
}

impl Case for BlitIndirectCommands {
    fn name(&self) -> &'static str {
        self.call.case()
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // Every byte of every leg's output compares exactly, on this venue and under this
        // declaration: the only site loosened is the set of notices, which Metal holds empty.
        //
        // Measured on this Apple M3 Max through MoltenVK 1.4.2: the reset and the copy cases
        // announce `host-replayed-indirect-commands`, because the runtime replays an execution of
        // commands the **GPU** wrote through its host step, and these two calls write a buffer's
        // records on the GPU as the command buffer runs. The optimize case records nothing on the
        // GPU, so it announces nothing and declares nothing — a declaration no run was measured
        // provoking is a declaration nobody checked.
        //
        // The allowance fails the case if the notice did not fire, so a call that stops replaying
        // on the host has to drop this line rather than carry a stale excuse.
        // Every row reads its answer back through a `Shared` buffer, so on a device without
        // unified memory the runtime announces `shared-in-system-memory`
        // (`crate::replay::shared_memory_notice` holds the whole reason). It declares the
        // notices site alone, so every value of this family still compares against Metal's
        // exactly.
        //
        // Measured on the RTX 5080 Laptop (615.71.09): it fired for
        // all 3 rows of the family, and each was divergent on that field and
        // nothing else (replay-nvidia.log). On MoltenVK, where
        // the chip's GPU shares the CPU's memory, it cannot fire at all, which is why no
        // recording of this suite declares it.
        let mut named = vec![crate::replay::shared_memory_notice()];
        if self.call.replays_on_the_host() {
            named.push(crate::replay::notice_allowance(
                Degradation::HostReplayedIndirectCommands.code(),
            ));
        }
        named
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let held = Fixture::new(FIXTURE);
        [RUNTIME_TEXT, METAL_TEXT]
            .into_iter()
            .map(|extension| {
                Ok(ShaderRef {
                    fixture: format!("{FIXTURE}{extension}"),
                    entry: FIXTURE.to_owned(),
                    source_sha256: crate::sha256_hex(held.text(extension)?.as_bytes()),
                })
            })
            .collect()
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let held = Fixture::new(FIXTURE);
        let commands = encoded(Handle(0), Handle(1), Handle(2));
        let mut words = Vec::new();
        for command in &commands {
            words.extend(command.words());
            words.push(b'\n');
        }
        // Every leg's own range, and where a copy puts it, so a recording cannot stay the answer
        // to a sweep whose ranges moved.
        let mut ranges = Vec::new();
        for leg in LEGS {
            ranges.extend(leg.name.as_bytes());
            ranges.push(b':');
            ranges.extend(leg.first.to_le_bytes());
            ranges.extend(leg.count.to_le_bytes());
            if self.call.has_destination() {
                ranges.extend(leg.into_index.to_le_bytes());
            }
            let (first, count) = self.call.executed_range(leg);
            ranges.extend(first.to_le_bytes());
            ranges.extend(count.to_le_bytes());
            ranges.push(b'\n');
        }
        Ok(vec![
            Input {
                name: "air".to_owned(),
                digest: Digest::sha256(held.text(RUNTIME_TEXT)?.as_bytes()),
            },
            Input {
                name: "air-typed".to_owned(),
                digest: Digest::sha256(held.text(METAL_TEXT)?.as_bytes()),
            },
            Input::bytes("icb-descriptor", &descriptor().words()),
            Input::bytes("commands", &words),
            Input::bytes("call", self.call.case().as_bytes()),
            Input::bytes("ranges", &ranges),
            Input::bytes("buffer:params", &params_bytes()),
            Input::bytes("buffer:out", &output_bytes()),
        ])
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let fixture = Fixture::new(FIXTURE);
        let loaded = fixture.loaded().map_err(|why| Abort {
            at: "library".to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        })?;
        let library = made(sites, "library", side.library(&loaded.source()))?;
        let pipeline = made(
            sites,
            "pipeline",
            side.compute_pipeline_for_commands(library, FIXTURE, None),
        )?;
        let params = made(sites, "buffer/params", side.buffer(&params_bytes()))?;

        // One leg at a time, each with its own buffers, so no leg can read a word another wrote.
        let mut outputs = Vec::new();
        let mut work = Vec::new();
        for leg in LEGS {
            let out = made(
                sites,
                &format!("{}/out", leg.name),
                side.buffer(&output_bytes()),
            )?;
            let commands = encoded(pipeline, params, out);
            let source = made(
                sites,
                &format!("{}/source", leg.name),
                side.indirect_command_buffer(&descriptor(), &commands),
            )?;
            let destination = if self.call.has_destination() {
                made(
                    sites,
                    &format!("{}/destination", leg.name),
                    side.indirect_command_buffer(&descriptor(), &unencoded()),
                )?
                .commands
            } else {
                source.commands
            };
            // A refusal at the encoding is the case's answer, named by the leg it happened in.
            for (at, answers) in source.refused.iter().enumerate() {
                for (step, answer) in answers.iter().enumerate() {
                    if let Some(failure) = answer {
                        return Err(Abort {
                            at: format!(
                                "{}/source/command/{at}/{step}-{}",
                                leg.name,
                                commands[at].step_call(step)
                            ),
                            failure: failure.clone(),
                        });
                    }
                }
            }
            work.push(self.call.work(source.commands, destination, leg));
            let (first, count) = self.call.executed_range(leg);
            work.push(RenderWork::ComputeCommands {
                commands: self.call.executed(source.commands, destination),
                first,
                count,
                used: vec![(params, false), (out, true)],
            });
            outputs.push(out);
        }

        let outcome = made(sites, "program", side.render(&work))?;
        let mut calls = Node::branch("calls", Vec::new());
        for (at, refused) in outcome.refused.iter().enumerate() {
            let name = work.get(at).map_or("call", RenderWork::call);
            calls = calls.with(Node::leaf(
                format!("{at}-{name}"),
                Observed::value(match refused {
                    Some(failure) => failure.recorded(),
                    None => crate::transcript::Value::ok(crate::transcript::Value::Absent),
                }),
            ));
        }
        sites.push(
            Node::branch("program-calls", Vec::new())
                .with(calls)
                .with(Node::leaf(
                    "status",
                    Observed::value(outcome.status.clone()),
                )),
        );
        if let Some((at, failure)) = outcome
            .refused
            .iter()
            .enumerate()
            .find_map(|(at, refused)| refused.as_ref().map(|failure| (at, failure)))
        {
            let name = work.get(at).map_or("call", RenderWork::call);
            return Err(Abort {
                at: format!("program-calls/calls/{at}-{name}"),
                failure: failure.clone(),
            });
        }

        // One site a leg, the whole output buffer as words, so a diff names the segment that
        // moved and the tail past the last segment compares too.
        for (leg, out) in LEGS.into_iter().zip(outputs) {
            let read = made(
                sites,
                &format!("read-back/{}", leg.name),
                side.read_buffer(out, 0, OUTPUT_WORDS as usize * 4),
            )?;
            sites.push(Node::leaf(
                leg.name,
                Observed::bytes(read, ByteLayout::buffer(0, OUTPUT_WORDS * 4, Element::U32)),
            ));
        }
        Ok(())
    }
}

/// Every case of the family.
pub fn all() -> Vec<Box<dyn Case>> {
    [Call::Reset, Call::Copy, Call::Optimize]
        .into_iter()
        .map(|call| Box::new(BlitIndirectCommands { call }) as Box<dyn Case>)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// Every leg's site name is its own, so no leg of a case silently loses its bytes to another:
    /// the transcript answers the first sibling of a name, and a duplicate would drop one whole
    /// range from the comparison.
    #[test]
    fn every_leg_names_itself() {
        let names: BTreeSet<&str> = LEGS.iter().map(|leg| leg.name).collect();
        assert_eq!(names.len(), LEGS.len(), "{names:?}");
        let cases: BTreeSet<&str> = [Call::Reset, Call::Copy, Call::Optimize]
            .iter()
            .map(|call| call.case())
            .collect();
        assert_eq!(cases.len(), 3, "{cases:?}");
    }

    /// Every swept range lies inside the buffer's commands, at both ends, and a copy's range lands
    /// inside the destination's too.
    ///
    /// A range past the commands is what the module leaves out on purpose, so this is the check
    /// that keeps a leg from being added back without that decision being revisited.
    #[test]
    fn every_range_lies_inside_its_buffer() {
        for leg in LEGS {
            assert!(
                leg.first + leg.count <= COMMANDS,
                "{}: {} commands at {} of a buffer of {COMMANDS}",
                leg.name,
                leg.count,
                leg.first
            );
            assert!(
                leg.into_index + leg.count <= COMMANDS,
                "{}: {} commands at {} of a destination of {COMMANDS}",
                leg.name,
                leg.count,
                leg.into_index
            );
        }
    }

    /// The sweep holds the four boundaries the module names: no commands, one, one at a non-zero
    /// first index, and every command of the buffer.
    #[test]
    fn the_sweep_covers_the_four_boundaries() {
        assert!(LEGS.iter().any(|leg| leg.count == 0), "no empty range");
        assert!(
            LEGS.iter().any(|leg| leg.count == 1 && leg.first == 0),
            "no range of one at the first command"
        );
        assert!(
            LEGS.iter().any(|leg| leg.count == 1 && leg.first > 0),
            "no range of one at a non-zero first index"
        );
        assert!(
            LEGS.iter().any(|leg| leg.count == COMMANDS),
            "no range of the whole buffer"
        );
    }

    /// Only the copy case states a destination, and its legs put their range somewhere a plain
    /// index sweep would not: one leg lands a source range at a **different** destination index,
    /// so a side that ignored `destinationIndex` writes another segment.
    #[test]
    fn only_the_copy_case_has_a_destination() {
        let with: Vec<&str> = [Call::Reset, Call::Copy, Call::Optimize]
            .into_iter()
            .filter(|call| call.has_destination())
            .map(Call::case)
            .collect();
        assert_eq!(with, [Call::Copy.case()]);
        assert!(
            LEGS.iter()
                .any(|leg| leg.count > 0 && leg.into_index != leg.first),
            "no leg copies a range to a different destination index"
        );
    }

    /// Every command binds inside its own buffers, and no command's output segment reaches the
    /// tail: the module's safety claim, computed from the commands rather than asserted beside
    /// them.
    #[test]
    fn every_command_binds_inside_its_buffers() {
        let commands = encoded(Handle(0), Handle(1), Handle(2));
        assert_eq!(commands.len(), COMMANDS as usize);
        for (at, command) in commands.iter().enumerate() {
            let EncodedCommand::Compute(steps) = command else {
                panic!("command {at} is not a compute command");
            };
            for step in steps {
                match step {
                    ComputeCommandStep::KernelBuffer { offset, index, .. } if *index == 0 => {
                        assert!(
                            offset + PARAM_STRIDE <= params_bytes().len() as u64,
                            "command {at} binds parameters at {offset}"
                        );
                    }
                    ComputeCommandStep::KernelBuffer { offset, index, .. } if *index == 1 => {
                        // The dispatch writes `THREADS` words from the bound offset, and the
                        // segment past the last is the tail nothing may reach.
                        let written = offset + u64::from(THREADS) * 4;
                        assert!(
                            written <= SEGMENT_WORDS * COMMANDS * 4,
                            "command {at} writes to {written}, past the last segment"
                        );
                    }
                    ComputeCommandStep::DispatchThreadgroups { groups, threads } => {
                        assert_eq!(*groups, [1, 1, 1], "command {at}");
                        assert_eq!(*threads, [THREADS, 1, 1], "command {at}");
                    }
                    _ => {}
                }
            }
        }
    }

    /// No two commands bind the same parameter pair, so a command that ran another's bindings
    /// writes another's sequence rather than the same words twice.
    #[test]
    fn no_two_commands_share_a_parameter_pair() {
        let pairs: BTreeSet<[u32; 2]> = PARAMS.iter().copied().collect();
        assert_eq!(pairs.len(), PARAMS.len(), "{pairs:?}");
        // A pair of zeros would make a command's words indistinguishable from the fixture writing
        // nothing but its own index.
        assert!(PARAMS.iter().all(|[a, b]| *a != 0 && *b != 0));
    }

    /// The copy executes the commands it wrote and nothing else, so `destinationIndex` reaches a
    /// site the comparison reads; the other two calls execute the whole buffer.
    ///
    /// This is the guard behind the falsification the module's prose records: with the whole
    /// destination executed, a runtime that ignored `destinationIndex` changed no byte of this
    /// case. A leg whose executed range stopped covering its own destination index would put the
    /// case back in that state, so the range is checked against the leg rather than trusted.
    #[test]
    fn the_copy_executes_only_where_it_wrote() {
        for leg in LEGS {
            let (first, count) = Call::Copy.executed_range(leg);
            assert_eq!(first, leg.into_index, "{}", leg.name);
            assert!(count > 0, "{}: an empty execution", leg.name);
            assert!(
                first + count <= COMMANDS,
                "{}: executes {count} commands at {first} of {COMMANDS}",
                leg.name
            );
            if leg.count > 0 {
                assert_eq!(count, leg.count, "{}", leg.name);
            }
            for other in [Call::Reset, Call::Optimize] {
                assert_eq!(
                    other.executed_range(leg),
                    (0, COMMANDS),
                    "{} does not execute the whole buffer",
                    other.case()
                );
            }
        }
        // The leg that lands a range at a destination index other than its source index is the
        // one that makes the argument observable, and its executed range has to be that index.
        let moved = LEGS
            .iter()
            .find(|leg| leg.count > 0 && leg.into_index != leg.first)
            .expect("a leg copies a range to a different destination index");
        assert_eq!(Call::Copy.executed_range(*moved).0, moved.into_index);
    }

    /// Each case's program is one blit call and one execution a leg, in that order: the blit has
    /// to be encoded before the execution it changes, or every leg would read the commands as they
    /// were encoded.
    #[test]
    fn each_leg_encodes_its_call_before_its_execution() {
        for call in [Call::Reset, Call::Copy, Call::Optimize] {
            let mut work = Vec::new();
            for leg in LEGS {
                work.push(call.work(Handle(0), Handle(1), leg));
                let (first, count) = call.executed_range(leg);
                work.push(RenderWork::ComputeCommands {
                    commands: call.executed(Handle(0), Handle(1)),
                    first,
                    count,
                    used: Vec::new(),
                });
            }
            assert_eq!(work.len(), LEGS.len() * 2);
            for pair in work.chunks(2) {
                assert_ne!(
                    pair[0].call(),
                    "executeCommandsInBuffer",
                    "{}: a leg executes before its blit call",
                    call.case()
                );
                assert_eq!(pair[1].call(), "executeCommandsInBuffer", "{}", call.case());
            }
        }
    }
}
