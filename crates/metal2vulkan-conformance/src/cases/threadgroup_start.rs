//! What threadgroup memory holds before the program writes it: one compute pass runs `tgz_reads`
//! (the owned fixture; its source says what each thread does), whose threads read their words of a
//! declared threadgroup array and of a `[[threadgroup(0)]]` region before any thread of their
//! threadgroup stores a nonzero pattern there, then read their neighbour's words back. The pass
//! sets the region's length once (`setThreadgroupMemoryLength:atIndex:`) and dispatches 2048
//! threadgroups of 256 threads, of 250 and of 30, twice each, into a 16-byte slot of one output
//! buffer per dispatch, so later threadgroups of a dispatch, and every threadgroup of the second
//! dispatch, run where earlier threadgroups stored their patterns. The dispatches and the answer
//! where threadgroup memory reads zero until the program writes it are
//! `metal2vulkan_devtools::threadgroup_start`'s; this module's test holds Metal's recorded answer
//! equal to it, and the new runtime's device test holds its own equal to it.
//!
//! Every output word is an atomic OR or a wrapping sum, the same in any order, so every byte
//! compares exactly. Nothing here can fault a GPU: one pass of six dispatches, every access inside
//! the threadgroup's own memory or the output's 16 bytes, loops bounded by the array's words.

use crate::allowance::Allowance;
use crate::case::{made, Abort, Case, Failure, Handle, Input, MetalSource, Side};
use crate::compute_pass::{ComputePass, ComputeStep, Dispatching};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::transcript::{ByteLayout, Digest, Element, Gaps, Node, Observed, Region, Sites};
use metal2vulkan_devtools::threadgroup_start::{DISPATCHES, FIXTURE, GROUPS, REGION_BYTES, SIZES};

/// The case.
pub struct ThreadgroupStart;

/// The case's name.
pub const NAME: &str = "compute-threadgroup-memory-reads-zero";

/// The bytes of one dispatch's output slot: four words.
const SLOT: u32 = 16;

/// The dispatches, in pass order: each size [`DISPATCHES`] times.
fn dispatches() -> impl Iterator<Item = u32> {
    SIZES
        .into_iter()
        .flat_map(|size| std::iter::repeat_n(size, DISPATCHES))
}

/// The output buffer's bytes, zero before the pass: one slot a dispatch.
fn output() -> Vec<u8> {
    vec![0; SLOT as usize * dispatches().count()]
}

fn fixture() -> Fixture {
    Fixture::new(FIXTURE)
}

/// The output's layout: a row of four words a dispatch, so a difference names the dispatch and
/// the word.
fn layout() -> ByteLayout {
    let rows = u32::try_from(dispatches().count()).expect("dispatches in a u32");
    ByteLayout {
        region: Region::Buffer {
            offset: 0,
            length: u64::from(SLOT * rows),
        },
        element: Element::U32,
        row_offset: 0,
        row_elements: SLOT / 4,
        row_pitch: SLOT,
        rows,
        image_pitch: SLOT * rows,
        images: 1,
        gaps: Gaps::Compared,
    }
}

/// The pass: the region's length once, then each dispatch with the output bound at its slot.
fn steps(pipeline: Handle, out: Handle) -> Vec<ComputeStep> {
    let mut steps = vec![
        ComputeStep::Pipeline(pipeline),
        ComputeStep::ThreadgroupMemory {
            length: REGION_BYTES,
            index: 0,
        },
    ];
    for (slot, size) in dispatches().enumerate() {
        steps.push(ComputeStep::Buffer {
            buffer: out,
            offset: u64::from(SLOT) * slot as u64,
            index: 0,
        });
        steps.push(ComputeStep::DispatchThreadgroups {
            groups: [GROUPS, 1, 1],
            threads: [size, 1, 1],
        });
    }
    steps
}

impl Case for ThreadgroupStart {
    fn name(&self) -> &'static str {
        NAME
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // Every byte compares exactly. The answer is read back through a `Shared` buffer, so a
        // device without unified memory announces `shared-in-system-memory`
        // (`crate::replay::shared_memory_notice` holds the whole reason).
        vec![crate::replay::shared_memory_notice()]
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let mut shaders = Vec::new();
        for extension in [RUNTIME_TEXT, METAL_TEXT] {
            shaders.push(ShaderRef {
                fixture: format!("{FIXTURE}{extension}"),
                entry: FIXTURE.to_owned(),
                source_sha256: crate::sha256_hex(fixture().text(extension)?.as_bytes()),
            });
        }
        Ok(shaders)
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let f = fixture();
        let steps = steps(Handle(0), Handle(0));
        Ok(vec![
            Input {
                name: "air".to_owned(),
                digest: Digest::sha256(f.text(RUNTIME_TEXT)?.as_bytes()),
            },
            Input {
                name: "air-typed".to_owned(),
                digest: Digest::sha256(f.text(METAL_TEXT)?.as_bytes()),
            },
            Input::bytes("buffer:out", &output()),
            // The pass, with its threadgroups, sizes and region length: a case re-pointed at
            // other dispatches is another question, so its recording cannot stand for this one.
            Input::bytes(
                "pass",
                &ComputePass {
                    dispatching: Dispatching::Serial,
                    steps: &steps,
                }
                .words(),
            ),
        ])
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let loaded = fixture().loaded().map_err(|why| Abort {
            at: "library".to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        })?;
        let library = made(sites, "library", side.library(&loaded.source()))?;
        let pipeline = made(sites, "pipeline", side.compute_pipeline(library, FIXTURE))?;
        let bytes = output();
        let out = made(sites, "buffer/out", side.buffer(&bytes))?;
        let steps = steps(pipeline, out);
        let outcome = made(
            sites,
            "pass",
            side.compute_pass(&ComputePass {
                dispatching: Dispatching::Serial,
                steps: &steps,
            }),
        )?;
        sites.push(outcome.node("outcome", &steps));
        if let Some((at, failure)) = outcome.first_refusal() {
            return Err(Abort {
                at: format!("outcome/step/{at}/{}", steps[at].call()),
                failure: failure.clone(),
            });
        }
        let read = made(
            sites,
            "read-back/out",
            side.read_buffer(out, 0, bytes.len()),
        )?;
        sites.push(Node::leaf("out", Observed::bytes(read, layout())));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcript::Value;
    use metal2vulkan_devtools::threadgroup_start::answer;

    /// Metal's recorded answer is every dispatch's
    /// `metal2vulkan_devtools::threadgroup_start::answer`: no word of either threadgroup object
    /// read nonzero before the program wrote it, in later threadgroups and in the second
    /// dispatch alike, and every pattern read back.
    #[test]
    fn metal_reads_zero_from_threadgroup_memory_until_the_program_writes_it() {
        let recording =
            crate::recording::read(NAME).unwrap_or_else(|e| panic!("{NAME}: no recording: {e}"));
        let want: Vec<u8> = dispatches()
            .flat_map(|size| answer(GROUPS, size))
            .flat_map(u32::to_le_bytes)
            .collect();
        let mut answers = 0;
        for metal in &recording.answers {
            for outcome in &metal.outcomes {
                answers += 1;
                let got = outcome
                    .transcript
                    .root
                    .child("out")
                    .map(|n| &n.observed.value);
                assert!(
                    matches!(got, Some(Value::Bytes(b)) if *b == want),
                    "{NAME}: Metal's words are {got:?}"
                );
            }
        }
        assert!(answers > 0, "{NAME}: no Metal answer");
    }
}
