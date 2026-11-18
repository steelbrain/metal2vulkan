//! A `uint` buffer bound off its argument's alignment: one compute pass runs `vwp_words` (the owned
//! fixture `vector_word_phases`) over `src` and `dst` bound at byte 2, where its `uint *`
//! arguments take a multiple of 4. Metal's API validation refuses that bind offset ("must be a
//! multiple of 4 but was set to 2", the `uint-2` bind of `probes/vector_access_placement.m`), and
//! the recording keeps those words as its validated fact. Plain Metal runs the pass, and the
//! recording keeps the pass's outcome: each step's answer, the command buffer's status and error.
//! The new runtime refuses the dispatch (`Refusal::BufferOffsetAlignment`), so there this case
//! is a refusal that Metal's own validation also makes.
//!
//! The case compares the outcome and reads no bytes back. Where plain Metal places each element of
//! a bind off its alignment is a question of its own (`probes/misaligned_buffer_binds.m`, and the
//! entry "A dispatch that reaches a buffer bound off its argument's alignment is refused" of
//! `docs/API.md` section 10), and this case is about the refusal. Nothing here can fault a GPU:
//! one dispatch of 64 threads, every access within 8 KiB buffers, which an Apple M3 Max runs at
//! addresses rounded down to a multiple of 4.

use crate::allowance::Allowance;
use crate::case::{made, Abort, Case, Handle, Input, MetalSource, Side};
use crate::compute_pass::{ComputePass, ComputeStep, Dispatching};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::transcript::{Digest, Sites};
use metal2vulkan_devtools::vector_word_phases::{inputs, FIXTURE, THREADS, WORDS};

/// The case.
pub struct BufferBindOffAlignment;

/// The case's name.
pub const NAME: &str = "buffer-bind-off-alignment";

/// The bind offset of both buffers: 2 bytes past a multiple of 4.
pub const OFFSET: u64 = 2;

/// The words Metal's API validation uses for the bind, which the recording's validated fact holds.
pub const VALIDATION_SAYS: &str = "must be a multiple of 4 but was set to 2";

fn fixture() -> Fixture {
    Fixture {
        name: FIXTURE,
        entry: WORDS,
    }
}

/// The pass: `vwp_words` with `src` and `dst` at [`OFFSET`], one threadgroup of [`THREADS`].
fn steps(words: Handle, src: Handle, dst: Handle) -> Vec<ComputeStep> {
    vec![
        ComputeStep::Pipeline(words),
        ComputeStep::Buffer {
            buffer: src,
            offset: OFFSET,
            index: 0,
        },
        ComputeStep::Buffer {
            buffer: dst,
            offset: OFFSET,
            index: 1,
        },
        ComputeStep::DispatchThreadgroups {
            groups: [1, 1, 1],
            threads: [THREADS, 1, 1],
        },
    ]
}

impl Case for BufferBindOffAlignment {
    fn name(&self) -> &'static str {
        NAME
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // The pass's buffers are `Shared`, so a device without unified memory announces
        // `shared-in-system-memory` (`crate::replay::shared_memory_notice` holds the whole reason).
        vec![crate::replay::shared_memory_notice()]
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let mut shaders = Vec::new();
        for extension in [RUNTIME_TEXT, METAL_TEXT] {
            shaders.push(ShaderRef {
                fixture: format!("{FIXTURE}{extension}"),
                entry: WORDS.to_owned(),
                source_sha256: crate::sha256_hex(fixture().text(extension)?.as_bytes()),
            });
        }
        Ok(shaders)
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let f = fixture();
        let i = inputs();
        let steps = steps(Handle(0), Handle(0), Handle(0));
        Ok(vec![
            Input {
                name: "air".to_owned(),
                digest: Digest::sha256(f.text(RUNTIME_TEXT)?.as_bytes()),
            },
            Input {
                name: "air-typed".to_owned(),
                digest: Digest::sha256(f.text(METAL_TEXT)?.as_bytes()),
            },
            Input::bytes("buffer:src", &i.src),
            Input::bytes("buffer:dst", &i.dst),
            // The pass, with its bind offsets: a case re-pointed at another offset is another
            // question, so its recording cannot stand for this one.
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
            failure: crate::case::Failure::named("FixtureNotRead", why),
        })?;
        let library = made(sites, "library", side.library(&loaded.source()))?;
        let words = made(sites, "pipeline", side.compute_pipeline(library, WORDS))?;
        let i = inputs();
        let src = made(sites, "buffer/src", side.buffer(&i.src))?;
        let dst = made(sites, "buffer/dst", side.buffer(&i.dst))?;
        let steps = steps(words, src, dst);
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
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recording::Validated;

    /// Metal's recording holds both facts: plain Metal ran the pass to completion, and its API
    /// validation stopped the bind with [`VALIDATION_SAYS`].
    #[test]
    fn metal_runs_the_bind_and_its_validation_refuses_it() {
        let recording =
            crate::recording::read(NAME).unwrap_or_else(|e| panic!("{NAME}: no recording: {e}"));
        assert!(!recording.answers.is_empty(), "{NAME}: no Metal answer");
        for metal in &recording.answers {
            assert!(
                matches!(&metal.validated, Validated::Stopped { message, .. }
                    if message.contains(VALIDATION_SAYS)),
                "{NAME}: the validated fact is {:?}",
                metal.validated
            );
            for outcome in &metal.outcomes {
                let status = outcome
                    .transcript
                    .root
                    .child("outcome")
                    .and_then(|node| node.child("status"))
                    .map(|node| format!("{:?}", node.observed.value));
                assert!(
                    status.as_deref().is_some_and(|s| s.contains("Completed")),
                    "{NAME}: plain Metal's status is {status:?}"
                );
            }
        }
    }
}
