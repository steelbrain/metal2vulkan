//! Compute threadgroups: what Metal's own threads of one threadgroup wrote, at Metal's sizes.
//!
//! Ported from the runtime lane `runtime_lane/compute_threadgroups.rs`. The owned fixture
//! `ttg_compute_simd_scratch` holds a 24 KiB `threadgroup` array and shuffles a `float4` one lane
//! down its simdgroup, so what it writes pins three Metal facts at once in each dispatch: what
//! `threads_per_threadgroup` answers, what `simd_shuffle_and_fill_down` fills the last lane of a
//! simdgroup from, and — in the largest row — that Metal holds a threadgroup of 1024 threads with
//! 24 KiB of threadgroup memory beside it.
//!
//! **Every row is one-dimensional, and that is a measured constraint rather than a preference.**
//! The fixture takes `uint size [[threads_per_threadgroup]]`, a **scalar**, and a scalar carries
//! the x component alone: a threadgroup of 8 by 4 gives `size == 8` while `t` runs to 31, so
//! `stage[size - 1u - t]` underflows and the kernel reads its `threadgroup` array far out of
//! bounds. Measured here on the oracle while porting this lane, by writing those rows and reading
//! what came back: the 8-by-4 row's words began at 6 (`size - 2 - t` for `size == 8`) and its tail
//! was uninitialised scratch, and Metal's own API validation named it —
//! `component 1: 4 must be <= 1 for size [[ threads_per_threadgroup ]]`. The rows were removed. The
//! lane never dispatches this fixture in more than one dimension, so the lane is not wrong; what is
//! missing is the fixture saying so, and a Metal fact about where
//! `thread_index_in_threadgroup` puts a thread of a two-dimensional threadgroup needs a kernel that
//! takes the builtin as a vector.
//!
//! **Metal's answer is run, not copied.** The lane never ran Metal: it computes its expectation
//! from a formula in Rust (this crate's own `tests::metals_words_are_the_lanes_formula` keeps that
//! formula, as the readable model, and checks the recorded words against it). So the words here are
//! the words Metal's threads wrote on the oracle, and the lane's formula is now evidence rather
//! than a comment.
//!
//! **The device's limit is pinned by a dispatch, not by a getter.** The lane reads
//! `maxTotalThreadsPerThreadgroup` off the state and sizes its dispatch from it, which cannot
//! work where there is no Metal: the number is the *device's* (Metal's 1024 on Apple's GPUs, 512
//! on a device that emulates 32-lane simdgroups in 32 KiB of shared memory), and a venue that
//! answers less is a smaller device rather than a divergence. So every row here states a fixed
//! threadgroup, Metal's own, and the row at Metal's 1024 is **last**: a venue that cannot hold it
//! refuses the dispatch typed, which the suite reads as the venue's honest gap
//! ([`Verdict::LacksFeature`](crate::verdict::Verdict::LacksFeature)), with every smaller row
//! already compared. The one getter a case does compare is
//! `staticThreadgroupMemoryLength`, which is the shader's own number and so the same wherever the
//! shader is compiled.
//!
//! **Not ported, and why.** The lane also dispatches a threadgroup of 2048 threads, past Metal's
//! 1024, and a threadgroup within Metal's limit but past the device's through
//! `dispatchThreadgroupsWithIndirectBuffer:`. Both are rows about an over-limit threadgroup, which
//! this round may not hand a GPU: Metal's API validation asserts on it and plain Metal leaves it
//! undefined. The over-limit direct form is [`past_metals_limit`], which carries the Apple M4
//! measurement this tree already holds and forbids re-recording; the indirect form needs a
//! dispatch call the suite's vocabulary does not have, and stays a lane.
//!
//! Nothing here can fault a GPU: one compute pass a row, one shared buffer, no texture, no loop,
//! and every threadgroup inside Metal's own 1024 threads.

use crate::case::{answered, call, made, Abort, Bindings, Case, Failure, Input, MetalSource, Side};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::replay::NOTICES_SITE;
use crate::transcript::{
    ByteLayout, Digest, Element, Gaps, Negative, Observed, Region, Sites, Transcript, Value,
};

/// The fixture both cases run.
const FIXTURE: &str = "ttg_compute_simd_scratch";

/// Metal's threads a threadgroup on Apple's GPUs, which the largest row dispatches.
const METAL_THREADS: u32 = 1024;

/// The bytes one thread's `float4` takes.
const THREAD_BYTES: u32 = 16;

/// The threadgroups every row dispatches, so each row records a second threadgroup whose
/// `threadgroup_position_in_grid` is 1.
const GROUPS: u32 = 2;

/// The byte the output buffer holds before a row runs, so a word no thread wrote is a recorded
/// value rather than whatever the allocator left.
const FILL: u8 = 0xAB;

/// One row: a threadgroup shape Metal runs.
struct Row {
    name: &'static str,
    /// The threadgroup, in threads along each axis.
    threads: [u32; 3],
}

/// The rows, **smallest first**, so a venue that cannot hold Metal's 1024-thread threadgroup has
/// every smaller row compared before it refuses the last one.
///
/// Each row's threads are a whole number of 32-lane simdgroups. A threadgroup of a partial
/// simdgroup is deliberately absent: the fixture's `simd_shuffle_and_fill_down` would read an
/// inactive lane there, which Metal leaves undefined, and a case whose answer Metal does not
/// define is a case that records noise.
const ROWS: &[Row] = &[
    // One simdgroup.
    Row {
        name: "t32",
        threads: [32, 1, 1],
    },
    // Two simdgroups: the shuffle's fill at the end of each.
    Row {
        name: "t64",
        threads: [64, 1, 1],
    },
    // Metal's own limit, with 24 KiB of threadgroup memory beside it. Last on purpose.
    Row {
        name: "t1024",
        threads: [METAL_THREADS, 1, 1],
    },
];

impl Row {
    /// The threads the threadgroup holds.
    fn size(&self) -> u32 {
        self.threads[0] * self.threads[1] * self.threads[2]
    }

    /// The output buffer's bytes: one `float4` a thread of every threadgroup.
    fn output_bytes(&self) -> usize {
        (GROUPS * self.size() * THREAD_BYTES) as usize
    }

    /// The layout the row's output buffer compares under: one row of the layout per threadgroup,
    /// read as the single-precision words the kernel stores, so a diff names the threadgroup, the
    /// thread and the channel that moved.
    fn layout(&self) -> ByteLayout {
        let pitch = self.size() * THREAD_BYTES;
        ByteLayout {
            region: Region::Buffer {
                offset: 0,
                length: self.output_bytes() as u64,
            },
            element: Element::F32,
            row_offset: 0,
            row_elements: self.size() * 4,
            row_pitch: pitch,
            rows: GROUPS,
            image_pitch: self.output_bytes() as u32,
            images: 1,
            gaps: Gaps::Compared,
        }
    }

    /// Everything the row sets that is not the buffer's bytes, as its input fingerprint: a row
    /// whose threadgroup moved is `Unrecorded` at that row rather than a quiet pass.
    fn parameters(&self) -> Vec<u8> {
        self.threads
            .iter()
            .chain(std::iter::once(&GROUPS))
            .flat_map(|axis| axis.to_le_bytes())
            .collect()
    }
}

/// The fixture, whose entry point is its own name.
fn fixture() -> Fixture {
    Fixture::new(FIXTURE)
}

/// The shaders both cases commit.
fn shaders_of() -> Result<Vec<ShaderRef>, String> {
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

/// The fixture's two texts as inputs.
fn fixture_inputs() -> Result<Vec<Input>, String> {
    Ok(vec![
        Input {
            name: "air".to_owned(),
            digest: Digest::sha256(fixture().text(RUNTIME_TEXT)?.as_bytes()),
        },
        Input {
            name: "air-typed".to_owned(),
            digest: Digest::sha256(fixture().text(METAL_TEXT)?.as_bytes()),
        },
    ])
}

/// The case at Metal's own threadgroup sizes.
pub struct SimdScratch;

/// The case at Metal's own threadgroup sizes.
pub fn simd_scratch() -> SimdScratch {
    SimdScratch
}

impl Case for SimdScratch {
    fn name(&self) -> &'static str {
        "compute-threadgroup-simd-scratch"
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<crate::allowance::Allowance> {
        // Every row reads its answer back through a `Shared` buffer, so on a device without
        // unified memory the runtime announces `shared-in-system-memory`
        // (`crate::replay::shared_memory_notice` holds the whole reason). It declares the notices
        // site alone: every value of this family still compares against Metal's exactly.
        //
        // Measured on the RTX 5080 Laptop (615.71.09), where it fired for all this case
        // and each row was divergent on that field and nothing else (replay-nvidia.log).
        vec![crate::replay::shared_memory_notice()]
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        shaders_of()
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let mut inputs = fixture_inputs()?;
        for row in ROWS {
            inputs.push(Input::bytes(format!("row:{}", row.name), &row.parameters()));
            inputs.push(Input::bytes(
                format!("buffer:{}", row.name),
                &vec![FILL; row.output_bytes()],
            ));
        }
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let loaded = fixture().loaded().map_err(|why| Abort {
            at: "library".to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        })?;
        let library = made(sites, "library", side.library(&loaded.source()))?;
        let pipeline = made(sites, "pipeline", side.compute_pipeline(library, FIXTURE))?;
        call(
            sites,
            "static-threadgroup-memory-length",
            side.static_threadgroup_memory_length(pipeline),
            |length| Value::Uint(*length),
        )?;
        for row in ROWS {
            let at = |site: &str| format!("{}/{site}", row.name);
            let output = made(
                sites,
                &at("buffer"),
                side.buffer(&vec![FILL; row.output_bytes()]),
            )?;
            let command_buffer = made(
                sites,
                &at("dispatch"),
                side.dispatch(
                    pipeline,
                    &Bindings::buffers(&[output]),
                    [GROUPS, 1, 1],
                    row.threads,
                ),
            )?;
            call(
                sites,
                &at("status"),
                side.wait(command_buffer),
                Clone::clone,
            )?;
            let bytes = call(
                sites,
                &at("read-back"),
                side.read_buffer(output, 0, row.output_bytes()),
                |_| Value::Absent,
            )?;
            sites.leaf(output_site(row.name), Observed::bytes(bytes, row.layout()));
        }
        Ok(())
    }
}

/// The over-limit case's carried measurement, and where it is written down.
const CARRIED_FROM: &str = "metal2vulkan_runtime::compute::ComputeCommandEncoder::threadgroup, \
                            whose Apple M4 measurement under macOS 27.0.1 found that plain Metal \
                            runs nothing for a threadgroup past its limit for the state";

/// Why record mode must never run the over-limit case on Metal.
const NEVER_RE_RECORD: &str = "a threadgroup of 2048 threads is past Metal's 1024: its API \
                               validation asserts on it and plain Metal leaves it undefined, so \
                               the suite does not hand a GPU the shape to find out again";

/// The chip the carried measurement was taken on.
const CARRIED_CHIP: &str = crate::recording::APPLE_M4;

/// The threads the over-limit case dispatches: twice Metal's limit, as the lane dispatches it.
const PAST_METAL: u32 = 2 * METAL_THREADS;

/// The runs the carried measurement stands on. The probe behind it reported one answer for each
/// over-limit shape it tried; the row records that one run rather than implying a sample.
const CARRIED_RUNS: u32 = 1;

/// The name of the case whose Metal answer is carried from the Apple M4.
pub const PAST_METALS_LIMIT: &str = "compute-threadgroup-past-metals-limit";

/// The case of a threadgroup past Metal's limit, whose Metal answer is carried.
pub struct PastMetalsLimit;

/// The case of a threadgroup past Metal's limit.
pub fn past_metals_limit() -> PastMetalsLimit {
    PastMetalsLimit
}

impl PastMetalsLimit {
    /// The buffer's bytes before the dispatch.
    fn before() -> Vec<u8> {
        vec![FILL; (PAST_METAL * THREAD_BYTES) as usize]
    }

    /// The transcript the carried measurement states.
    ///
    /// It holds the sites a run emits and nothing else — and it deliberately holds **no command
    /// buffer status**: what the probe measured is that the dispatch wrote nothing, and a status
    /// nobody wrote down would be the case inventing a second fact out of the first. The run below
    /// waits for the command buffer, because the read-back needs it to have completed, and does
    /// not record what the wait answered.
    pub fn carried(&self) -> Result<(u32, Transcript), String> {
        let mut sites = Sites::new();
        sites.push(crate::case::inputs_node(&self.inputs()?));
        sites.leaf("library", answered());
        sites.leaf("pipeline", answered());
        sites.leaf("buffer", answered());
        sites.leaf("dispatch", answered());
        sites.leaf("read-back", answered());
        sites.value("output", Value::Nothing(Negative::ChangedNothing));
        sites.notices(NOTICES_SITE, &[] as &[&str]);
        Ok((CARRIED_RUNS, sites.finish()))
    }
}

impl Case for PastMetalsLimit {
    fn name(&self) -> &'static str {
        PAST_METALS_LIMIT
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::carried_only(CARRIED_FROM, NEVER_RE_RECORD).measured_on(CARRIED_CHIP)
    }

    fn allowances(&self) -> Vec<crate::allowance::Allowance> {
        // Every row reads its answer back through a `Shared` buffer, so on a device without
        // unified memory the runtime announces `shared-in-system-memory`
        // (`crate::replay::shared_memory_notice` holds the whole reason). It declares the notices
        // site alone: every value of this family still compares against Metal's exactly.
        //
        // Measured on the RTX 5080 Laptop (615.71.09), where it fired for all this case
        // and each row was divergent on that field and nothing else (replay-nvidia.log).
        vec![crate::replay::shared_memory_notice()]
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        shaders_of()
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let mut inputs = fixture_inputs()?;
        inputs.push(Input::bytes("threads", PAST_METAL.to_le_bytes().as_ref()));
        inputs.push(Input::bytes("buffer:0", &Self::before()));
        Ok(inputs)
    }

    fn carried_transcript(&self) -> Result<Option<(u32, Transcript)>, String> {
        Ok(Some(self.carried()?))
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let loaded = fixture().loaded().map_err(|why| Abort {
            at: "library".to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        })?;
        let library = made(sites, "library", side.library(&loaded.source()))?;
        let pipeline = made(sites, "pipeline", side.compute_pipeline(library, FIXTURE))?;
        let before = Self::before();
        let output = made(sites, "buffer", side.buffer(&before))?;
        let command_buffer = made(
            sites,
            "dispatch",
            side.dispatch(
                pipeline,
                &Bindings::buffers(&[output]),
                [1, 1, 1],
                [PAST_METAL, 1, 1],
            ),
        )?;
        // Waited for, so the read-back below reads after the command buffer completed, and not
        // recorded: the carried measurement states what the dispatch wrote and nothing else.
        let _ = side.wait(command_buffer);
        let after = call(
            sites,
            "read-back",
            side.read_buffer(output, 0, before.len()),
            |_| Value::Absent,
        )?;
        // "Changed nothing" is the fact, stated as a fact rather than as bytes that happen to
        // match: a side that wrote anything emits its bytes instead, and the two differ.
        let observed = if after == before {
            Observed::value(Value::Nothing(Negative::ChangedNothing))
        } else {
            Observed::bytes(
                after,
                ByteLayout::buffer(0, before.len() as u64, Element::F32),
            )
        };
        sites.leaf("output", observed);
        Ok(())
    }
}

/// Every case of the family, in the order the report lists them.
pub fn all() -> Vec<Box<dyn Case>> {
    vec![Box::new(simd_scratch()), Box::new(past_metals_limit())]
}

/// The site a case emits the read-back of a row's output buffer at.
fn output_site(row: &str) -> String {
    format!("{row}/output")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// The lane's own model of what the kernel writes, kept exactly as
    /// `runtime_lane::compute_threadgroups::expected` states it: for threadgroup `g` of `size`
    /// threads at thread `t`, the staged value of thread `size - 1 - t` shuffled down one lane,
    /// lane 31 filled from lane 0's `float4(0)`.
    fn lanes_formula(size: u32, g: u32, t: u32) -> [f32; 4] {
        if t % 32 == 31 {
            [0.0; 4]
        } else {
            [(size - 2 - t) as f32, g as f32, 0.0, 1.0]
        }
    }

    /// The rows name distinct sites, since the comparison pairs the sides' sites by name.
    #[test]
    fn the_rows_name_distinct_sites() {
        let named: BTreeSet<&str> = ROWS.iter().map(|row| row.name).collect();
        assert_eq!(named.len(), ROWS.len());
    }

    /// Every row's threadgroup is one-dimensional, a whole number of 32-lane simdgroups and inside
    /// Metal's 1024, and its layout covers its output buffer exactly. Each clause stops the case
    /// recording something Metal does not define: a partial simdgroup makes the shuffle read an
    /// inactive lane, and a threadgroup of more than one dimension makes the fixture read its
    /// `threadgroup` array **out of bounds** (see the module's note on
    /// `[[threads_per_threadgroup]]`).
    #[test]
    fn every_rows_threadgroup_is_one_dimensional_whole_simdgroups_inside_metals_limit() {
        for row in ROWS {
            let size = row.size();
            assert_eq!(
                [row.threads[1], row.threads[2]],
                [1, 1],
                "{}: {:?} threads",
                row.name,
                row.threads
            );
            assert_eq!(size % 32, 0, "{}: {size} threads", row.name);
            assert!(size <= METAL_THREADS, "{}: {size} threads", row.name);
            assert_eq!(
                row.layout().extent(),
                row.output_bytes(),
                "{}: its layout reaches {} of {} bytes",
                row.name,
                row.layout().extent(),
                row.output_bytes()
            );
        }
    }

    /// Metal's 1024-thread row is the last, so a venue that cannot hold it refuses after every
    /// smaller row has been compared rather than before.
    #[test]
    fn metals_own_threadgroup_is_the_last_row() {
        let last = ROWS.last().expect("the family holds rows");
        assert_eq!(last.size(), METAL_THREADS);
        for row in &ROWS[..ROWS.len() - 1] {
            assert!(row.size() < METAL_THREADS, "{}", row.name);
        }
    }

    /// Each row's parameters digest differently, so the input fingerprint tells the rows apart.
    #[test]
    fn the_rows_parameters_are_distinct() {
        let digests: BTreeSet<String> = ROWS
            .iter()
            .map(|row| crate::sha256_hex(&row.parameters()))
            .collect();
        assert_eq!(digests.len(), ROWS.len());
    }

    /// **The cross-check the port owes the lane.** The words Metal wrote, as the committed
    /// recording holds them, are the words the lane's formula predicts, for every thread of every
    /// threadgroup of every row. The lane's expectation was never tied to Metal — it was computed
    /// in Rust and compared against our own runtime — and this is what ties it.
    #[test]
    fn metals_words_are_the_lanes_formula() {
        let case = simd_scratch();
        let recording = crate::recording::read(case.name()).expect("the committed recording");
        let answer = recording
            .answers
            .first()
            .expect("the recording holds an answer");
        let outcome = answer
            .outcomes
            .first()
            .expect("the answer holds an outcome");
        let mut checked = 0usize;
        for row in ROWS {
            let site = output_site(row.name);
            let node = outcome
                .transcript
                .root
                .child(&site)
                .unwrap_or_else(|| panic!("the recording holds no {site}"));
            let crate::transcript::Value::Bytes(bytes) = &node.observed.value else {
                panic!("{site}: the recording holds no bytes");
            };
            let layout = row.layout();
            for g in 0..GROUPS {
                for t in 0..row.size() {
                    let wanted = lanes_formula(row.size(), g, t);
                    for (channel, &word) in wanted.iter().enumerate() {
                        let at = layout.at(0, g, t * 4 + channel as u32);
                        let found =
                            f32::from_le_bytes(bytes[at..at + 4].try_into().expect("four bytes"));
                        assert_eq!(
                            found.to_bits(),
                            word.to_bits(),
                            "{site}: threadgroup {g} thread {t} channel {channel}"
                        );
                        checked += 1;
                    }
                }
            }
        }
        assert_eq!(
            checked,
            ROWS.iter()
                .map(|row| (GROUPS * row.size() * 4) as usize)
                .sum::<usize>()
        );
    }

    /// A side that answers every call the carried row makes and writes nothing, so the row can be
    /// run without a device: what it is for is checking that the carried transcript states exactly
    /// the sites a run emits.
    struct Unwriting {
        /// The next handle it answers; nothing compares one.
        next: u32,
        /// The bytes its one buffer holds, which no dispatch of it changes.
        held: Vec<u8>,
    }

    impl Unwriting {
        fn new() -> Self {
            Self {
                next: 0,
                held: Vec::new(),
            }
        }

        fn handle(&mut self) -> crate::case::Handle {
            self.next += 1;
            crate::case::Handle(self.next - 1)
        }
    }

    impl Side for Unwriting {
        fn venue(&self) -> crate::case::Venue {
            crate::case::Venue {
                device: "unwriting".to_owned(),
                driver: "none".to_owned(),
                api: "none".to_owned(),
                metal_level: "none".to_owned(),
            }
        }

        fn library(
            &mut self,
            _source: &crate::case::ShaderSource<'_>,
        ) -> crate::case::Answer<crate::case::Handle> {
            Ok(self.handle())
        }

        fn compute_pipeline(
            &mut self,
            _library: crate::case::Handle,
            _entry: &str,
        ) -> crate::case::Answer<crate::case::Handle> {
            Ok(self.handle())
        }

        fn buffer(&mut self, bytes: &[u8]) -> crate::case::Answer<crate::case::Handle> {
            self.held = bytes.to_vec();
            Ok(self.handle())
        }

        fn zeroed_buffer(&mut self, length: usize) -> crate::case::Answer<crate::case::Handle> {
            self.held = vec![0; length];
            Ok(self.handle())
        }

        fn texture_2d(
            &mut self,
            _format: metal2vulkan::metal::PixelFormat,
            _size: [u32; 2],
            _row_pitch: u32,
            _bytes: &[u8],
        ) -> crate::case::Answer<crate::case::Handle> {
            Err(Failure::unavailable("texture_2d"))
        }

        fn read_texture(
            &mut self,
            _texture: crate::case::Handle,
            _size: [u32; 2],
            _row_pitch: u32,
        ) -> crate::case::Answer<Vec<u8>> {
            Err(Failure::unavailable("read_texture"))
        }

        fn dispatch(
            &mut self,
            _pipeline: crate::case::Handle,
            _bindings: &Bindings<'_>,
            _threadgroups: [u32; 3],
            _threads: [u32; 3],
        ) -> crate::case::Answer<crate::case::Handle> {
            Ok(self.handle())
        }

        fn wait(&mut self, _command_buffer: crate::case::Handle) -> crate::case::Answer<Value> {
            Ok(Value::unit_variant("Completed"))
        }

        fn read_buffer(
            &mut self,
            _buffer: crate::case::Handle,
            offset: u64,
            length: usize,
        ) -> crate::case::Answer<Vec<u8>> {
            let at = offset as usize;
            Ok(self.held[at..at + length].to_vec())
        }

        fn announced(&mut self) -> Vec<String> {
            Vec::new()
        }

        fn compute_pipeline_with_stage_input(
            &mut self,
            _library: crate::case::Handle,
            _entry: &str,
            _stage_input: &crate::case::StageInput,
        ) -> crate::case::Answer<crate::case::Handle> {
            Err(Failure::unavailable("compute_pipeline_with_stage_input"))
        }

        fn dispatch_threads(
            &mut self,
            _pipeline: crate::case::Handle,
            _bindings: &Bindings<'_>,
            _stage_in: &[crate::case::StageInRegion],
            _grid: [u32; 3],
            _threads: [u32; 3],
        ) -> crate::case::Answer<crate::case::Handle> {
            Err(Failure::unavailable("dispatch_threads"))
        }
    }

    /// The carried row states exactly the sites its run emits, **checked by running it**: a
    /// carried transcript that named a site the run does not emit, or missed one it does, compares
    /// as a divergence at every replay, and reading the two by eye is how that was missed once.
    #[test]
    fn the_carried_rows_sites_are_the_runs_sites() {
        let case = past_metals_limit();
        let (runs, carried) = case.carried().expect("the carried transcript");
        assert_eq!(runs, CARRIED_RUNS);
        let mut side = Unwriting::new();
        let mut sites = Sites::new();
        sites.push(crate::case::inputs_node(
            &case.inputs().expect("the inputs"),
        ));
        case.run(&mut side, &mut sites).expect("the run answered");
        sites.notices(NOTICES_SITE, &[] as &[&str]);
        let names = |transcript: &Transcript| -> Vec<String> {
            transcript
                .root
                .children
                .iter()
                .map(|node| node.name.clone())
                .collect()
        };
        let ran = sites.finish();
        assert_eq!(names(&ran), names(&carried));
        // And a side that wrote nothing gives the carried row's own answer, so the two transcripts
        // are equal and not merely shaped alike.
        assert_eq!(ran, carried);
    }

    /// Record mode must refuse the carried row: it is the one row of this family whose shape the
    /// suite may not hand a GPU.
    #[test]
    fn record_mode_refuses_the_over_limit_row() {
        let forbidden = past_metals_limit()
            .metal_source()
            .forbidden()
            .expect("the row forbids re-recording");
        assert_eq!(forbidden, NEVER_RE_RECORD);
        assert!(simd_scratch().metal_source().forbidden().is_none());
    }
}
