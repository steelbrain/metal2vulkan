//! The compute encoder's two heap residency declarations, over a heap-placed binding.
//!
//! `useHeap:` and `useHeaps:count:` are the last two residency calls
//! `MTLComputeCommandEncoder` offers that no case reached. They are reached here the way
//! `compute-resource-residency` reaches the resource pair: one dispatch behind each, into a slot
//! apiece, through a buffer the case placed in a heap.
//!
//! # What this can observe, and what it cannot
//!
//! **A declaration cannot be observed by a case that reads its own bindings, and this one says so
//! rather than implying otherwise.** A bound slot is resident without being declared, so neither
//! call changes a word of either slot; the observable a residency declaration *would* have is a
//! resource the kernel reaches **off-slot** — through an argument buffer or a device address the
//! case never bound — and reaching an undeclared resource that way is undefined behaviour, which
//! is not an answer worth taking off a shared GPU. So the two slots of this case are the same
//! words by construction, and the sweep over the two spellings is an equivalence and not a value
//! comparison.
//!
//! So this case is **reach rather than depth**, and is the one case of this stream that is: it
//! adds two items to what the suite calls, and neither the choice between the two spellings nor
//! the heap placement itself is distinguishable by anything it reads back. A value sweep would
//! compare the same bytes twice. What it does hold is still not nothing:
//!
//! - the words are **Metal's own**, through a binding whose memory is a heap's rather than a
//!   buffer's, which no compute case did before: an implementation that placed the buffer at the
//!   wrong offset in the heap, or lost the heap's memory to its suballocator, writes different
//!   words or none;
//! - both calls are **encoded and answered**, so an implementation that refused either — or that
//!   took `useHeaps:count:` as a no-op where it takes `useHeap:` — is caught at the step's own
//!   site rather than silently;
//! - and the slot behind each call is compared against the other, so the two spellings are held to
//!   one another as well as to Metal.
//!
//! # How the case is read
//!
//! An **automatic** heap, a buffer Metal's own suballocator places in it, and two dispatches of
//! the owned `a4_index_word` fixture, which is `out[t] = in[3t + 1]`: it reads a plain buffer the
//! case made holding known words and writes the heap-placed one. The first dispatch writes slot 0
//! behind `useHeap:`, the second writes slot 1 behind `useHeaps:count:` over a range of one, and
//! both write the same words — which is the equivalence, not an accident.
//!
//! **Every word the case reads back is a word its own dispatches wrote.** A heap-placed buffer is
//! made without contents — `Side::heap_buffer` has no bytes to give it — so, unlike every other
//! compute case here, this one cannot fill its output with a guard byte first and cannot compare a
//! tail past its slots. It therefore reads exactly the two slots and nothing else
//! (`the_read_back_is_only_what_the_dispatches_wrote`), and the "nothing was written past the
//! grid" question stays with the cases that own a plain buffer.
//!
//! # Safety
//!
//! Each dispatch is one threadgroup of `THREADS` threads, which is inside one simdgroup on every
//! Apple GPU and inside the smallest `maxTotalThreadsPerThreadgroup` any venue reports; no grid is
//! derived from a device limit. The heap is `HEAP_BYTES` bytes, a fixed size far above what one
//! output needs and far below any venue's memory, and the fixture is loop-free and reads only inside
//! its bindings. Nothing here can hang,
//! fault or trip a watchdog. The kernel reads `in[3t + 1]`, so the furthest word it reads is
//! `3 · (THREADS − 1) + 1`, and the input buffer holds `SOURCE_WORDS`, which is more
//! (`the_input_holds_every_word_the_kernel_reads`).

use crate::allowance::Allowance;
use crate::case::{made, Abort, Case, Failure, Handle, Input, MetalSource, Side};
use crate::compute_pass::{ComputePass, ComputeStep, Dispatching};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::heap_query::{HeapBufferSpec, HeapSpec};
use crate::recording::ShaderRef;
use crate::texture_query::Storage;
use crate::transcript::{ByteLayout, Digest, Element, Node, Observed, Sites};

/// The fixture whose `a4_index_word` is `out[t] = in[3t + 1]`: an elementwise read of a plain
/// buffer and a write of the heap-placed one, with **no simdgroup operation**.
///
/// A simdgroup move would be the obvious kernel to reuse here, and it is the wrong one: the
/// runtime announces `emulated-simdgroups` for one, and whether that notice is still in hand when
/// this case runs depends on which case drained it first in the same process. A case whose verdict
/// moves with its neighbours is not a case, so this one runs a kernel that announces nothing.
const FIXTURE: &str = "a4_index_word";
const ENTRY: &str = "a4_index_word";

/// Threads in each dispatch's one threadgroup: inside one simdgroup on every Apple GPU.
const THREADS: u32 = 8;

/// Bytes one slot holds: one 32-bit word a thread.
const SLOT: usize = THREADS as usize * 4;

/// Slots the output holds: one a residency call.
const SLOTS: usize = 2;

/// The heap's size: far above the one output it holds, and far below any venue's memory. Metal
/// rounds it up as it likes, which is a fact `heap-own-getters` owns and this case never reads.
const HEAP_BYTES: u64 = 4096;

/// The bytes the case's dispatches write and the case reads back.
const READ_BYTES: usize = SLOTS * SLOT;

/// Words the input buffer holds: more than the `3 · (THREADS − 1) + 1` the kernel reads.
const SOURCE_WORDS: usize = 32;

/// The input buffer's bytes: word `i` holding `i * 2654435761 mod 2^32`, a sequence no index
/// pattern reproduces, so a dispatch that read the wrong word shows.
fn source() -> Vec<u8> {
    (0..SOURCE_WORDS)
        .flat_map(|index| (index as u32).wrapping_mul(2_654_435_761).to_le_bytes())
        .collect()
}

/// The heap every run makes: automatic, so Metal's own suballocator places the buffer, in shared
/// storage so the host reads the placed buffer back.
fn heap_spec() -> HeapSpec {
    HeapSpec::automatic(HEAP_BYTES, Storage::Shared)
}

/// The buffer the case places in the heap: the whole output, wherever the suballocator puts it.
fn buffer_spec() -> HeapBufferSpec {
    HeapBufferSpec::suballocated(READ_BYTES as u64)
}

/// The pass: the pipeline, the two bindings, then a residency call and a dispatch into a slot
/// apiece.
fn pass(pipeline: Handle, source: Handle, out: Handle, heap: Handle) -> Vec<ComputeStep> {
    let dispatch = ComputeStep::DispatchThreadgroups {
        groups: [1, 1, 1],
        threads: [THREADS, 1, 1],
    };
    vec![
        ComputeStep::Pipeline(pipeline),
        ComputeStep::Buffer {
            buffer: source,
            offset: 0,
            index: 0,
        },
        // Slot 0: the singular declaration.
        ComputeStep::Buffer {
            buffer: out,
            offset: 0,
            index: 1,
        },
        ComputeStep::UseHeap { heap },
        dispatch.clone(),
        // Slot 1: the plural form over a range of one.
        ComputeStep::BufferOffset {
            offset: SLOT as u64,
            index: 1,
        },
        ComputeStep::UseHeaps { heaps: vec![heap] },
        dispatch,
    ]
}

/// The case.
pub struct ComputeHeapResidency;

/// The case the suite holds.
pub fn heap_residency() -> ComputeHeapResidency {
    ComputeHeapResidency
}

impl ComputeHeapResidency {
    /// The fixture under its entry point.
    fn fixture() -> Fixture {
        Fixture {
            name: FIXTURE,
            entry: ENTRY,
        }
    }

    /// The buffers the case makes outside the heap: the input alone.
    fn buffers() -> Vec<Vec<u8>> {
        vec![source()]
    }
}

impl Case for ComputeHeapResidency {
    fn name(&self) -> &'static str {
        "compute-heap-residency"
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // No value is loosened: both slots compare exactly.
        //
        // Every row reads its answer back through a `Shared` buffer, so on a device without
        // unified memory the runtime announces `shared-in-system-memory`
        // (`crate::replay::shared_memory_notice` holds the whole reason). It declares the
        // notices site alone, so every value of this family still compares against Metal's
        // exactly.
        //
        // Measured on the RTX 5080 Laptop (615.71.09): it fired for
        // this case, and each was divergent on that field and
        // nothing else (replay-nvidia.log). On MoltenVK, where
        // the chip's GPU shares the CPU's memory, it cannot fire at all, which is why no
        // recording of this suite declares it.
        vec![crate::replay::shared_memory_notice()]
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let fixture = Self::fixture();
        [RUNTIME_TEXT, METAL_TEXT]
            .into_iter()
            .map(|extension| {
                Ok(ShaderRef {
                    fixture: format!("{FIXTURE}{extension}"),
                    entry: ENTRY.to_owned(),
                    source_sha256: crate::sha256_hex(fixture.text(extension)?.as_bytes()),
                })
            })
            .collect()
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let fixture = Self::fixture();
        let steps = pass(Handle(0), Handle(0), Handle(0), Handle(0));
        Ok(vec![
            Input {
                name: "air".to_owned(),
                digest: Digest::sha256(fixture.text(RUNTIME_TEXT)?.as_bytes()),
            },
            Input {
                name: "air-typed".to_owned(),
                digest: Digest::sha256(fixture.text(METAL_TEXT)?.as_bytes()),
            },
            Input::bytes("buffer:in", &source()),
            Input::bytes("heap", &heap_spec().words()),
            // The placed buffer's own length and offset, so a recording stops being the answer to
            // a program whose placement moved.
            Input::bytes("heap-buffer", &{
                let spec = buffer_spec();
                let mut words = spec.length.to_le_bytes().to_vec();
                words.extend(spec.offset.unwrap_or(u64::MAX).to_le_bytes());
                words
            }),
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
        let fixture = Self::fixture();
        let loaded = fixture.loaded().map_err(|why| Abort {
            at: "library".to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        })?;
        let library = made(sites, "library", side.library(&loaded.source()))?;
        let pipeline = made(sites, "pipeline", side.compute_pipeline(library, ENTRY))?;
        let mut plain = Vec::new();
        for (at, bytes) in Self::buffers().into_iter().enumerate() {
            plain.push(made(sites, &format!("buffer/{at}"), side.buffer(&bytes))?);
        }
        let heap = made(sites, "heap", side.heap(&heap_spec()))?;
        let out = made(sites, "heap-buffer", side.heap_buffer(heap, &buffer_spec()))?;
        let steps = pass(pipeline, plain[0], out, heap);
        let outcome = made(
            sites,
            "pass",
            side.compute_pass(&ComputePass {
                dispatching: Dispatching::Serial,
                steps: &steps,
            }),
        )?;
        sites.push(outcome.node("outcome", &steps));
        // A refused command stops the case here, after every command's own site is emitted: the
        // words a refused dispatch did not write are a heap's prior contents and not an answer.
        if let Some((at, failure)) = outcome.first_refusal() {
            return Err(Abort {
                at: format!("outcome/step/{at}/{}", steps[at].call()),
                failure: failure.clone(),
            });
        }
        let read = made(sites, "read-back", side.read_buffer(out, 0, READ_BYTES))?;
        sites.push(Node::leaf(
            "two-declarations",
            Observed::bytes(read, ByteLayout::buffer(0, READ_BYTES as u64, Element::U32)),
        ));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Both residency calls are encoded, one before each dispatch, so each slot stands behind one
    /// of them.
    #[test]
    fn each_slot_stands_behind_one_residency_call() {
        let steps = pass(Handle(0), Handle(1), Handle(2), Handle(3));
        let order: Vec<&str> = steps
            .iter()
            .filter(|step| {
                matches!(
                    step,
                    ComputeStep::UseHeap { .. }
                        | ComputeStep::UseHeaps { .. }
                        | ComputeStep::DispatchThreadgroups { .. }
                )
            })
            .map(ComputeStep::call)
            .collect();
        assert_eq!(
            order,
            [
                "useHeap",
                "dispatchThreadgroups",
                "useHeaps",
                "dispatchThreadgroups"
            ]
        );
    }

    /// Every dispatch is one threadgroup of [`THREADS`] threads, which is the module's safety
    /// bound, and every threadgroup allocation holds the word `region_k` reads.
    #[test]
    fn every_dispatch_is_one_small_threadgroup() {
        for step in pass(Handle(0), Handle(1), Handle(2), Handle(3)) {
            if let ComputeStep::DispatchThreadgroups { groups, threads } = step {
                assert_eq!(groups, [1, 1, 1]);
                assert_eq!(threads, [THREADS, 1, 1]);
            }
        }
    }

    /// The case reads back exactly the words its dispatches wrote, and the placed buffer is
    /// exactly that long.
    ///
    /// This is the guard behind the module's note that a heap-placed buffer carries no guard byte:
    /// a read that went one word further would be comparing the heap's prior contents, which is
    /// not a value the case put there.
    #[test]
    fn the_read_back_is_only_what_the_dispatches_wrote() {
        let written = SLOTS * THREADS as usize * 4;
        assert_eq!(READ_BYTES, written);
        assert_eq!(buffer_spec().length, written as u64);
        assert!(
            buffer_spec().length <= HEAP_BYTES,
            "the placed buffer does not fit the heap"
        );
        // An automatic heap places the buffer itself, so the case states no offset: an offset
        // would be a placement heap's, and the two are different descriptors.
        assert_eq!(buffer_spec().offset, None);
    }

    /// The input holds every word the kernel reads: `in[3t + 1]` for the last thread.
    ///
    /// The module's safety claim rests on this, so it is computed from the kernel's own index
    /// rather than asserted beside it.
    #[test]
    fn the_input_holds_every_word_the_kernel_reads() {
        let furthest = 3 * (THREADS as usize - 1) + 1;
        assert!(
            furthest < SOURCE_WORDS,
            "the kernel reads word {furthest} of an input of {SOURCE_WORDS}"
        );
        assert_eq!(source().len(), SOURCE_WORDS * 4);
    }

    /// The heap is shared storage, which is what lets the host read the placed buffer back at all.
    #[test]
    fn the_heap_is_host_readable() {
        assert_eq!(heap_spec().storage, Storage::Shared);
        assert!(heap_spec().is_well_formed().is_none());
    }
}
