//! A kernel's explicit imageblock at the sizes its encoder sets: one compute pass runs
//! `a4_imageblock_extent` (the owned fixture; its source says what each thread does) three times,
//! first before the encoder sets any imageblock size, as one thread, then after
//! `setImageblockWidth:height:` at 4 by 2 and again at 4 by 4, each as one threadgroup the
//! imageblock's size, into a slot of one output buffer each. Every thread stages its own cell,
//! reads its right-hand neighbour's after the imageblock barrier (wrapping at the imageblock's
//! width) and writes the extent the kernel's `get_width` and `get_height` answer, so a dispatch run
//! at a size other than the encoder's reads another extent and other neighbours. The first
//! dispatch is Metal's answer for a size never set: 1 by 1.
//!
//! The output buffer is made holding [`GUARD`] and compared whole, the bytes past each dispatch's
//! threads and the tail past the last slot included, so a dispatch that wrote more than its
//! threadgroup's words shows. Every word is an integer or the bits of a small integral float, so
//! every byte compares exactly. Nothing here can fault a GPU: one pass of three dispatches of one
//! threadgroup of at most 16 threads, loop-free, every access inside the threadgroup's imageblock
//! (a neighbour's column wraps at the imageblock's width and its row is the thread's own, inside a
//! threadgroup the imageblock's size) or inside its slot of the output.

use crate::allowance::Allowance;
use crate::case::{made, Abort, Case, Failure, Handle, Input, MetalSource, Side};
use crate::compute_pass::{ComputePass, ComputeStep, Dispatching};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::transcript::{ByteLayout, Digest, Element, Gaps, Node, Observed, Region, Sites};

/// The case.
pub struct ImageblockSizes;

/// The case's name.
pub const NAME: &str = "compute-imageblock-sizes";

/// The fixture, whose entry point is its own name.
const FIXTURE: &str = "a4_imageblock_extent";

/// The imageblock size each dispatch runs at, in pass order: `None` is a dispatch before the
/// encoder sets one.
const SIZES: [Option<[u32; 2]>; 3] = [None, Some([4, 2]), Some([4, 4])];

/// The byte every output buffer is made holding, so the bytes no dispatch wrote are a value the
/// case put there.
const GUARD: u8 = 0x7e;

/// Bytes of a `uint4`, the words one thread writes.
const CELL: u32 = 16;

/// Bytes of one dispatch's slot: a thread's `uint4` for each thread of the largest threadgroup.
const SLOT: u32 = CELL * 16;

/// Bytes of guard past the last slot.
const TAIL: u32 = 64;

/// The extent a dispatch runs at, which is its threadgroup's too: the size the encoder set, or
/// Metal's 1 by 1 where it set none.
fn extent(size: Option<[u32; 2]>) -> [u32; 2] {
    size.unwrap_or([1, 1])
}

/// The output buffer's bytes before the pass: guard bytes, a slot a dispatch and the tail.
fn output() -> Vec<u8> {
    vec![GUARD; (SLOT * SIZES.len() as u32 + TAIL) as usize]
}

fn fixture() -> Fixture {
    Fixture::new(FIXTURE)
}

/// The output's layout: a row a thread's `uint4`, so a difference names the thread and the word.
fn layout() -> ByteLayout {
    let length = output().len() as u32;
    ByteLayout {
        region: Region::Buffer {
            offset: 0,
            length: u64::from(length),
        },
        element: Element::U32,
        row_offset: 0,
        row_elements: CELL / 4,
        row_pitch: CELL,
        rows: length / CELL,
        image_pitch: length,
        images: 1,
        gaps: Gaps::Compared,
    }
}

/// The pass: each size the encoder sets, then the output bound at the dispatch's slot and one
/// threadgroup the imageblock's extent.
fn steps(pipeline: Handle, out: Handle) -> Vec<ComputeStep> {
    let mut steps = vec![ComputeStep::Pipeline(pipeline)];
    for (slot, size) in SIZES.into_iter().enumerate() {
        if let Some([width, height]) = size {
            steps.push(ComputeStep::ImageblockSize { width, height });
        }
        steps.push(ComputeStep::Buffer {
            buffer: out,
            offset: u64::from(SLOT) * slot as u64,
            index: 0,
        });
        let [width, height] = extent(size);
        steps.push(ComputeStep::DispatchThreadgroups {
            groups: [1, 1, 1],
            threads: [width, height, 1],
        });
    }
    steps
}

impl Case for ImageblockSizes {
    fn name(&self) -> &'static str {
        NAME
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // One declaration, which loosens no value: the kernel holds SIMD-group access pairs, which
        // the runtime's pipelines note on every device (`crate::replay::simdgroup_access_order_notice`;
        // announced on the RTX 3090, the Radeon 780M and the Iris Xe, jobs m3-max-70084 to 70086).
        vec![crate::replay::simdgroup_access_order_notice()]
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
            // The pass, with its sizes and threadgroups: a case re-pointed at other sizes is
            // another question, so its recording cannot stand for this one.
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

    /// The fixture's answer for the pass: in each dispatch's slot, thread `(x, y)` of an extent
    /// `W` by `H` writes, at `y * W + x`, its neighbour's staged value and position and the
    /// extent, `(bits(float((x + 1) % W) + 100 * y), ((x + 1) % W) | (y << 16), W, H)`; every
    /// other byte keeps its guard.
    fn answer() -> Vec<u8> {
        let mut bytes = output();
        for (slot, size) in SIZES.into_iter().enumerate() {
            let [width, height] = extent(size);
            let mut at = (SLOT as usize) * slot;
            for y in 0..height {
                for x in 0..width {
                    let next = (x + 1) % width;
                    let words = [
                        (next as f32 + 100.0 * y as f32).to_bits(),
                        next | (y << 16),
                        width,
                        height,
                    ];
                    for word in words {
                        bytes[at..at + 4].copy_from_slice(&word.to_le_bytes());
                        at += 4;
                    }
                }
            }
        }
        bytes
    }

    /// Metal's recorded answer is the fixture's at each extent: 1 by 1 for the dispatch before
    /// the encoder set a size, then each size it set, every byte past the threads' words still
    /// its guard.
    #[test]
    fn metal_runs_each_dispatch_at_the_imageblock_size_its_encoder_set() {
        let recording =
            crate::recording::read(NAME).unwrap_or_else(|e| panic!("{NAME}: no recording: {e}"));
        let want = answer();
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
                    "{NAME}: Metal's bytes are {got:?}"
                );
            }
        }
        assert!(answers > 0, "{NAME}: no Metal answer");
    }

    /// Every dispatch is one threadgroup of at most 16 threads, the imageblock's own extent, whose
    /// words fit its slot; every size the pass sets is one Metal takes.
    #[test]
    fn every_dispatch_fits_its_slot_and_its_imageblock() {
        for size in SIZES {
            let [width, height] = extent(size);
            assert!(width.is_power_of_two() && height.is_power_of_two());
            assert!(width * height * CELL <= SLOT, "{size:?}");
        }
        let steps = steps(Handle(0), Handle(0));
        let one_threadgroup = |step: &&ComputeStep| {
            matches!(
                step,
                ComputeStep::DispatchThreadgroups {
                    groups: [1, 1, 1],
                    ..
                }
            )
        };
        let dispatches = steps.iter().filter(one_threadgroup).count();
        assert_eq!(dispatches, SIZES.len());
    }
}
