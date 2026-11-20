//! `uint4` and `float4` accesses at each word phase past a buffer's bound base: where Metal reads
//! and writes a vector whose AIR states `align 16` at an address that is a multiple of 4 only.
//!
//! The kernels are the owned fixture `vector_word_phases` (its source says what each thread does).
//! One compute pass runs `vwp_words` over two `uint` buffers bound at the case's offset, 0, 4, 8
//! or 12 (a `uint` buffer binds at any multiple of 4), so its vector accesses through cast
//! pointers land 0, 4, 8 or 12 bytes past a multiple of 16, then `vwp_vec16` over two `float4`
//! buffers bound at multiples of 16. Every buffer's bytes are read back whole, the bytes before
//! the bind offset and past the kernels' regions included, so an access placed anywhere but its
//! own address shows as words that moved. The inputs and the bind offsets are
//! `metal2vulkan_devtools::vector_word_phases`'s, whose `answer` is every access at its own
//! address; this module's tests hold Metal's recorded answer equal to it, and the runtime's
//! device test holds its own equal to it.
//!
//! Every input word is a normal `float`, so the kernels' negations and moves are exact on every
//! device, and every byte compares exactly. Nothing here can fault a GPU: one pass of two
//! dispatches of 64 threads, each access inside its buffer, no loop.

use crate::allowance::Allowance;
use crate::case::{made, Abort, Case, Failure, Input, MetalSource, Side};
use crate::compute_pass::{ComputePass, ComputeStep, Dispatching};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::transcript::{ByteLayout, Digest, Element, Gaps, Node, Observed, Region, Sites};
use metal2vulkan_devtools::vector_word_phases::{
    inputs, vec16_offsets, FIXTURE, OFFSETS, THREADS, VEC16, WORDS,
};

/// The case: `vwp_words`'s buffers bound at `offset`.
pub struct VectorWordPhases {
    name: &'static str,
    offset: u64,
}

/// One case per word phase.
pub fn all() -> Vec<VectorWordPhases> {
    let names = [
        "vector-word-phases-0",
        "vector-word-phases-4",
        "vector-word-phases-8",
        "vector-word-phases-12",
    ];
    names
        .into_iter()
        .zip(OFFSETS)
        .map(|(name, offset)| VectorWordPhases { name, offset })
        .collect()
}

/// The buffers in the order the case makes them, and their bytes.
const BUFFERS: [&str; 4] = ["src", "dst", "a", "b"];

fn fixture(entry: &'static str) -> Fixture {
    Fixture {
        name: FIXTURE,
        entry,
    }
}

/// A buffer's bytes as one row of words, so a difference names the word that moved.
fn layout(length: usize) -> ByteLayout {
    let words = u32::try_from(length / 4).expect("words in a u32");
    ByteLayout {
        region: Region::Buffer {
            offset: 0,
            length: length as u64,
        },
        element: Element::U32,
        row_offset: 0,
        row_elements: words,
        row_pitch: 4 * words,
        rows: 1,
        image_pitch: 4 * words,
        images: 1,
        gaps: Gaps::Ignored,
    }
}

impl VectorWordPhases {
    /// The pass: `vwp_words` with `src` and `dst` at the case's offset, then `vwp_vec16` with `a`
    /// and `b` at multiples of 16, each one threadgroup of [`THREADS`].
    fn steps(
        &self,
        (words, vec16): (crate::case::Handle, crate::case::Handle),
        buffers: &[crate::case::Handle; 4],
    ) -> Vec<ComputeStep> {
        let (a_at, b_at) = vec16_offsets(self.offset);
        let dispatch = ComputeStep::DispatchThreadgroups {
            groups: [1, 1, 1],
            threads: [THREADS, 1, 1],
        };
        let bind = |buffer, offset, index| ComputeStep::Buffer {
            buffer,
            offset,
            index,
        };
        vec![
            ComputeStep::Pipeline(words),
            bind(buffers[0], self.offset, 0),
            bind(buffers[1], self.offset, 1),
            dispatch.clone(),
            ComputeStep::Pipeline(vec16),
            bind(buffers[2], a_at, 0),
            bind(buffers[3], b_at, 1),
            dispatch,
        ]
    }

    fn bytes() -> [Vec<u8>; 4] {
        let i = inputs();
        [i.src, i.dst, i.a, i.b]
    }
}

impl Case for VectorWordPhases {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // Every byte compares exactly. The answers are read back through `Shared` buffers, so a
        // device without unified memory announces `shared-in-system-memory`
        // (`crate::replay::shared_memory_notice` holds the whole reason).
        vec![crate::replay::shared_memory_notice()]
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let mut shaders = Vec::new();
        for entry in [WORDS, VEC16] {
            for extension in [RUNTIME_TEXT, METAL_TEXT] {
                shaders.push(ShaderRef {
                    fixture: format!("{FIXTURE}{extension}"),
                    entry: entry.to_owned(),
                    source_sha256: crate::sha256_hex(fixture(entry).text(extension)?.as_bytes()),
                });
            }
        }
        Ok(shaders)
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let f = fixture(WORDS);
        let mut inputs = vec![
            Input {
                name: "air".to_owned(),
                digest: Digest::sha256(f.text(RUNTIME_TEXT)?.as_bytes()),
            },
            Input {
                name: "air-typed".to_owned(),
                digest: Digest::sha256(f.text(METAL_TEXT)?.as_bytes()),
            },
        ];
        for (name, bytes) in BUFFERS.iter().zip(Self::bytes()) {
            inputs.push(Input::bytes(format!("buffer:{name}"), &bytes));
        }
        // The pass, with its bind offsets: a case re-pointed at another offset is another
        // question, so its recording cannot stand for this one.
        let handles = [crate::case::Handle(0); 4];
        let steps = self.steps((crate::case::Handle(0), crate::case::Handle(0)), &handles);
        inputs.push(Input::bytes(
            "pass",
            &ComputePass {
                dispatching: Dispatching::Serial,
                steps: &steps,
            }
            .words(),
        ));
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let loaded = fixture(WORDS).loaded().map_err(|why| Abort {
            at: "library".to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        })?;
        let library = made(sites, "library", side.library(&loaded.source()))?;
        let words = made(
            sites,
            "pipeline/words",
            side.compute_pipeline(library, WORDS),
        )?;
        let vec16 = made(
            sites,
            "pipeline/vec16",
            side.compute_pipeline(library, VEC16),
        )?;
        let bytes = Self::bytes();
        let mut buffers = [crate::case::Handle(0); 4];
        for ((held, name), b) in buffers.iter_mut().zip(BUFFERS).zip(&bytes) {
            *held = made(sites, &format!("buffer/{name}"), side.buffer(b))?;
        }
        let steps = self.steps((words, vec16), &buffers);
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
        for ((held, name), b) in buffers.iter().zip(BUFFERS).zip(&bytes) {
            let read = made(
                sites,
                &format!("read-back/{name}"),
                side.read_buffer(*held, 0, b.len()),
            )?;
            sites.push(Node::leaf(name, Observed::bytes(read, layout(b.len()))));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcript::Value;
    use metal2vulkan_devtools::vector_word_phases::answer;

    /// Metal's recorded answer at each word phase is every access at its own address: the
    /// buffers `metal2vulkan_devtools::vector_word_phases::answer` gives, byte for byte, which the
    /// product's interpreter test and the runtime's device test hold their own answers to.
    #[test]
    fn metal_places_each_vector_at_its_own_address() {
        for case in all() {
            let recording = crate::recording::read(case.name)
                .unwrap_or_else(|e| panic!("{}: no recording: {e}", case.name));
            let want = answer(case.offset);
            let want = [want.src, want.dst, want.a, want.b];
            let mut answers = 0;
            for metal in &recording.answers {
                for outcome in &metal.outcomes {
                    answers += 1;
                    for (name, want) in BUFFERS.iter().zip(&want) {
                        let got = outcome
                            .transcript
                            .root
                            .child(name)
                            .map(|n| &n.observed.value);
                        assert!(
                            matches!(got, Some(Value::Bytes(b)) if b == want),
                            "{}: Metal's {name} is not every access at its own address",
                            case.name
                        );
                    }
                }
            }
            assert!(answers > 0, "{}: no Metal answer", case.name);
        }
    }

    /// The four cases bind at the four word phases, under distinct names.
    #[test]
    fn the_cases_take_each_word_phase() {
        let cases = all();
        let offsets: Vec<u64> = cases.iter().map(|c| c.offset).collect();
        assert_eq!(offsets, [0, 4, 8, 12]);
        let names: std::collections::BTreeSet<&str> = cases.iter().map(|c| c.name).collect();
        assert_eq!(names.len(), cases.len());
    }
}
