//! **A resolving store, then a load of the same multisampled attachment**: whether the samples
//! `MTLStoreActionStoreAndMultisampleResolve` keeps are the samples the next pass loads.
//!
//! `pass_sequence_actions` crosses pass A's store with pass B's load over a single-sampled
//! attachment and leaves this crossing open: `MTLStoreActionMultisampleResolve` writes the resolve
//! texture and keeps nothing of the multisampled one, `MTLStoreActionStoreAndMultisampleResolve`
//! keeps both, and only a pass B that loads the multisampled texture parts them. A side that
//! resolved in the pass and dropped the samples (a resolve attachment beside a store of
//! `DONT_CARE`) answers every resolve right and every later load wrong, so no case that reads a
//! resolve alone can see it.
//!
//! | cell | pass A's store | what the cell shows |
//! |---|---|---|
//! | `MTLStoreActionStoreAndMultisampleResolve` | stores and resolves into resolve A | the question: pass B loads the samples pass A kept |
//! | `MTLStoreActionStore` | stores, no resolve | the control: the same load and the same reads after a plain store |
//!
//! # The pass pair
//!
//! Each cell runs one command buffer over its own 8 by 8 `RGBA16Float` attachment of four
//! samples.
//!
//! - **Pass A** clears the attachment to `CLEAR` and draws the fullscreen triangle into each
//!   2 by 2 cell of the grid under a sample mask of the cell's own index, so cell `i` holds white
//!   on the samples of `i`'s bits and the clear on the rest: every one of the sixteen masks, the
//!   empty one included, which no draw names.
//! - **Pass B** loads the attachment, draws into cell 0 alone under `B_MASK`, and stores and
//!   resolves into resolve B.
//! - **The reads**: a kernel reads every sample of the attachment after pass B, and two blits read
//!   resolve A (where pass A resolves) and resolve B.
//!
//! The sample mask decides which samples a draw writes, so no cell's answer rests on where the
//! device puts a sample in its pixel or on a rasterizer's coverage rule. Every value is exact in
//! `half`: the clear's channels are quarters, white is 1, and a resolve's average of four of them
//! is a multiple of a sixteenth, so no answer rests on a rounding either.
//!
//! # What each read separates
//!
//! - **The samples after pass B** hold pass A's pattern in cells 1 to 15 and pass B's in cell 0
//!   where pass A's store kept them. A side that dropped them answers what its memory held in
//!   cells 1 to 15 and pass B's draw over that in cell 0.
//! - **Resolve B** is the same question through pass B's own resolve, read by a blit, which is
//!   the shape a program that never reads a sample sees.
//! - **Resolve A**, in the first cell, is pass A's own resolve: each texel the average of its
//!   cell's four samples. It says the resolve in the pass is right, so a difference in resolve B
//!   is the load's and not the resolve's.
//! - **The control cell** reads the same samples and resolve B after a plain `Store`. A side
//!   whose load or kernel read is wrong fails both cells alike; a side that drops the samples of
//!   a resolving store fails the first alone.
//!
//! # What is not asked, and why
//!
//! - **`MTLStoreActionMultisampleResolve` before the load.** Metal keeps nothing of the
//!   multisampled texture after it, so a load after it reads contents Metal leaves undefined, and
//!   there is no answer to compare.
//! - **A sample count other than four.** Vulkan leaves 2 samples optional and lavapipe does not
//!   render them; four is the count every venue we run answers.
//!
//! # Why this case is safe
//!
//! Two command buffers, one a cell: two render passes over an 8 by 8 attachment, one dispatch of
//! 64 threads that each read four samples, and one or two blits. Sixteen draws of three vertices,
//! no loop in either stage, every scissor rectangle inside the attachment, every vertex inside the
//! clip volume, and no sample count past what the venue answers. A venue that does not render four
//! samples of the format refuses the target and the suite counts the case
//! [`LacksFeature`](crate::verdict::Verdict::LacksFeature).

use crate::allowance::Allowance;
use crate::case::{call, made, Abort, Case, Failure, Handle, Input, MetalSource, Side};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::render::{
    ColorState, Dispatch, PassColor, ReadAt, RenderPass, RenderPipelineSpec, RenderStep,
    RenderWork, ScissorBox, TargetSpec,
};
use crate::transcript::{
    Aspect, ByteLayout, Digest, Element, Gaps, Node, Observed, Region, Sites, Value,
};
use crate::values::StoreAction;
use metal2vulkan::metal::{PixelFormat, PrimitiveType};

/// The case's name: its recording's file name and its row in the report.
const NAME: &str = "resolve-store-then-load";

/// The owned fixture the resolve families share: the fullscreen triangle at vertex buffer 0's
/// depth, and the fragment whose sample mask is fragment buffer 0's.
const DRAWS: &str = "rt6_resolves";

/// `rt6_resolve_full`: `float4(4 * a - 1, z, 1)` at each of three vertex ids, where `z` is the
/// `float` vertex buffer 0 holds.
const VERTEX: &str = "rt6_resolve_full";

/// `rt6_resolve_masked`: white, and the `uint` fragment buffer 0 holds as `[[sample_mask]]`.
const FRAGMENT: &str = "rt6_resolve_masked";

/// The owned fixture whose kernel reads every sample of a 4-sample 8 by 8 texture.
const READS: &str = "wm_samples";

/// `wm_read_samples`: each thread of an 8 by 8 grid writes its texel's four samples, as `float4`,
/// at `(y * 8 + x) * 4 + sample` of buffer 1.
const READ_SAMPLES: &str = "wm_read_samples";

/// The side of the attachment and of each resolve texture, in texels: the side the kernel reads.
const SIDE: u32 = 8;

/// The side of one cell of the grid.
const CELL: u32 = 2;

/// Cells along one axis.
const GRID: u32 = SIDE / CELL;

/// The samples the case renders at, and the samples the kernel reads.
const SAMPLES: u32 = 4;

/// The attachment's format and the resolve textures': a float format, which Metal resolves, of
/// `half` channels, in which every value the case states is exact.
const FORMAT: PixelFormat = PixelFormat::RGBA16Float;

/// Bytes of one texel of [`FORMAT`].
const TEXEL: u32 = 8;

/// Bytes a resolve texture's read takes.
const RESOLVE_BYTES: u32 = TEXEL * SIDE * SIDE;

/// Bytes the kernel writes: a `float4` a sample.
const SAMPLE_BYTES: u32 = 16 * SAMPLES * SIDE * SIDE;

/// Pass A's clear: quarters, each exact in `half`, none of them white's 1 but alpha.
const CLEAR: [f64; 4] = [0.25, 0.5, 0.75, 1.0];

/// The depth the fullscreen triangle is drawn at, inside `[0, 1]`; the pass has no depth
/// attachment.
const Z: f32 = 0.5;

/// Pass B's sample mask over cell 0, whose samples pass A left at the clear: samples 1 and 2.
const B_MASK: u32 = 0b0110;

/// The draw one triangle takes.
const TRIANGLE: RenderStep = RenderStep::Draw {
    primitive: PrimitiveType::Triangle,
    first: 0,
    count: 3,
    instances: 1,
};

/// How many runs record mode takes the case over: two command buffers on one queue and nothing a
/// device may reorder, so the runs are expected to fold into one outcome, and an answer that moved
/// between runs is the finding.
const RUNS: u32 = 8;

/// Pass A's store actions, one a cell, the question first.
const STORES: [StoreAction; 2] = [StoreAction::StoreAndMultisampleResolve, StoreAction::Store];

/// The store action's enumerator name, which is also what the cell's sites hold.
fn store_name(store: StoreAction) -> &'static str {
    match store {
        StoreAction::StoreAndMultisampleResolve => "MTLStoreActionStoreAndMultisampleResolve",
        StoreAction::Store => "MTLStoreActionStore",
        _ => "MTLStoreActionUnknown",
    }
}

/// Grid cell `at`, laid out row major.
fn cell(at: u32) -> ScissorBox {
    let column = (at % GRID) * CELL;
    let row = (at / GRID) * CELL;
    ScissorBox::new(
        u64::from(column),
        u64::from(row),
        u64::from(CELL),
        u64::from(CELL),
    )
}

/// One masked draw of the fullscreen triangle into grid cell `at`.
fn masked_draw(pipeline: Handle, at: u32, mask: u32) -> Vec<RenderStep> {
    vec![
        RenderStep::Pipeline(pipeline),
        RenderStep::VertexBytes {
            bytes: Z.to_le_bytes().to_vec(),
            index: 0,
        },
        RenderStep::FragmentBytes {
            bytes: mask.to_le_bytes().to_vec(),
            index: 0,
        },
        RenderStep::Scissor(cell(at)),
        TRIANGLE,
    ]
}

/// The handles a cell's passes name. A side's handle numbers are its own and none reaches a
/// fingerprint, so the zeroed set states the same passes.
#[derive(Clone, Copy)]
struct Handles {
    target: Handle,
    resolve_a: Handle,
    resolve_b: Handle,
    pipeline: Handle,
}

impl Handles {
    /// The set a fingerprint is taken over: every handle zeroed.
    const fn fingerprint() -> Self {
        let zero = Handle(0);
        Self {
            target: zero,
            resolve_a: zero,
            resolve_b: zero,
            pipeline: zero,
        }
    }
}

/// Pass A: the clear, then a draw into each cell under its own index as the mask, stored by
/// `store` (into resolve A where it resolves).
fn pass_a(store: StoreAction, handles: Handles) -> RenderPass {
    let color = PassColor::cleared(handles.target, CLEAR);
    let color = match store {
        StoreAction::StoreAndMultisampleResolve => color.resolving(handles.resolve_a),
        store => color.stored(store),
    };
    let steps = (1..GRID * GRID)
        .flat_map(|at| masked_draw(handles.pipeline, at, at))
        .collect();
    RenderPass::clearing(vec![color]).drawing(steps)
}

/// Pass B: the load, one draw into cell 0 under [`B_MASK`], stored and resolved into resolve B.
fn pass_b(handles: Handles) -> RenderPass {
    let color = PassColor::loaded(handles.target).resolving(handles.resolve_b);
    RenderPass::clearing(vec![color]).drawing(masked_draw(handles.pipeline, 0, B_MASK))
}

/// The attachment: four samples, and shader-read so the kernel reads them.
fn target() -> TargetSpec {
    TargetSpec::read(FORMAT, [SIDE; 2]).multisampled(SAMPLES)
}

/// A single-sampled texture of the attachment's own format, which Metal requires of a resolve
/// texture.
fn resolve_target() -> TargetSpec {
    TargetSpec::plain(FORMAT, [SIDE; 2])
}

/// The one colour state the pipeline holds.
fn colors() -> [Option<ColorState>; 1] {
    [Some(ColorState::new(FORMAT))]
}

/// The pipeline: four samples, no depth.
fn pipeline_spec<'a>(states: &'a [Option<ColorState>]) -> RenderPipelineSpec<'a> {
    RenderPipelineSpec::new(VERTEX, FRAGMENT, states).multisampled(SAMPLES)
}

/// The layout the kernel's read is compared under: texels in rows, each texel's four samples in
/// order, each sample's four channels as `float`.
fn samples_layout() -> ByteLayout {
    let row_pitch = SAMPLE_BYTES / SIDE;
    ByteLayout {
        region: Region::Buffer {
            offset: 0,
            length: u64::from(SAMPLE_BYTES),
        },
        element: Element::F32,
        row_offset: 0,
        row_elements: row_pitch / 4,
        row_pitch,
        rows: SIDE,
        image_pitch: SAMPLE_BYTES,
        images: 1,
        gaps: Gaps::Compared,
    }
}

/// The layout a resolve texture's read is compared under: one `half` a channel.
fn resolve_layout() -> ByteLayout {
    let row_pitch = TEXEL * SIDE;
    ByteLayout {
        region: Region::Texture {
            level: 0,
            slice: 0,
            aspect: Aspect::Color,
            sample: 0,
            origin: [0, 0, 0],
            size: [SIDE, SIDE, 1],
        },
        element: Element::F16,
        row_offset: 0,
        row_elements: SIDE * 4,
        row_pitch,
        rows: SIDE,
        image_pitch: RESOLVE_BYTES,
        images: 1,
        gaps: Gaps::Compared,
    }
}

/// The shader references both texts of both fixtures give.
fn shader_refs() -> Result<Vec<ShaderRef>, String> {
    let mut refs = Vec::new();
    for (fixture, entry) in [
        (DRAWS, format!("{VERTEX}+{FRAGMENT}")),
        (READS, READ_SAMPLES.to_owned()),
    ] {
        let held = Fixture::new(fixture);
        for extension in [RUNTIME_TEXT, METAL_TEXT] {
            refs.push(ShaderRef {
                fixture: format!("{fixture}{extension}"),
                entry: entry.clone(),
                source_sha256: crate::sha256_hex(held.text(extension)?.as_bytes()),
            });
        }
    }
    Ok(refs)
}

/// A fixture that would not load, as the abort at `at`.
fn not_read(at: &str, why: String) -> Abort {
    Abort {
        at: at.to_owned(),
        failure: Failure::named("FixtureNotRead", why),
    }
}

/// The module's one case.
pub fn all() -> Vec<Box<dyn Case>> {
    vec![Box::new(ResolveThenLoad)]
}

/// A resolving store and a plain one, each followed by a pass that loads the samples.
pub struct ResolveThenLoad;

impl Case for ResolveThenLoad {
    fn name(&self) -> &'static str {
        NAME
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn record_runs(&self) -> u32 {
        RUNS
    }

    fn allowances(&self) -> Vec<Allowance> {
        // No value is loosened. The buffers the kernel and the blits write are `Shared`
        // ([`crate::replay::shared_memory_notice`]): the old runtime's side announced it on the
        // RTX 2080 (job m3-max-55760), and nothing else differed there.
        vec![crate::replay::shared_memory_notice()]
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        shader_refs()
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let states = colors();
        let mut inputs = Vec::new();
        for (fixture, name) in [(DRAWS, "draws"), (READS, "reads")] {
            let held = Fixture::new(fixture);
            inputs.push(Input {
                name: format!("air:{name}"),
                digest: Digest::sha256(held.text(RUNTIME_TEXT)?.as_bytes()),
            });
            inputs.push(Input {
                name: format!("air-typed:{name}"),
                digest: Digest::sha256(held.text(METAL_TEXT)?.as_bytes()),
            });
        }
        inputs.push(Input::bytes(
            "render-pipeline",
            &pipeline_spec(&states).words(),
        ));
        inputs.push(Input::bytes("compute-pipeline", READ_SAMPLES.as_bytes()));
        inputs.push(Input::bytes("target", &target().words()));
        inputs.push(Input::bytes("target:resolve", &resolve_target().words()));
        for store in STORES {
            let handles = Handles::fingerprint();
            inputs.push(Input::bytes(
                format!("pass-a:{}", store_name(store)),
                &pass_a(store, handles).words(),
            ));
        }
        inputs.push(Input::bytes(
            "pass-b",
            &pass_b(Handles::fingerprint()).words(),
        ));
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let draws = Fixture::new(DRAWS)
            .loaded()
            .map_err(|why| not_read("library:draws", why))?;
        let reads = Fixture::new(READS)
            .loaded()
            .map_err(|why| not_read("library:reads", why))?;
        let draw_library = made(sites, "library:draws", side.library(&draws.source()))?;
        let read_library = made(sites, "library:reads", side.library(&reads.source()))?;
        let states = colors();
        let pipeline = made(
            sites,
            "render-pipeline",
            side.render_pipeline(draw_library, &pipeline_spec(&states)),
        )?;
        let read = made(
            sites,
            "compute-pipeline",
            side.compute_pipeline(read_library, READ_SAMPLES),
        )?;

        for store in STORES {
            let name = store_name(store);
            let resolves = matches!(store, StoreAction::StoreAndMultisampleResolve);
            // Each cell its own attachment, resolve textures and buffers, so no cell's answer can
            // be another cell's leftovers.
            let handles = Handles {
                target: made(sites, &format!("target/{name}"), side.target(&target()))?,
                resolve_a: made(
                    sites,
                    &format!("target:resolve-a/{name}"),
                    side.target(&resolve_target()),
                )?,
                resolve_b: made(
                    sites,
                    &format!("target:resolve-b/{name}"),
                    side.target(&resolve_target()),
                )?,
                pipeline,
            };
            let samples = made(
                sites,
                &format!("buffer:samples/{name}"),
                side.zeroed_buffer(SAMPLE_BYTES as usize),
            )?;
            let resolved_a = made(
                sites,
                &format!("buffer:resolve-a/{name}"),
                side.zeroed_buffer(RESOLVE_BYTES as usize),
            )?;
            let resolved_b = made(
                sites,
                &format!("buffer:resolve-b/{name}"),
                side.zeroed_buffer(RESOLVE_BYTES as usize),
            )?;

            let read_resolve = |texture: Handle, into: Handle| RenderWork::ReadTexture {
                texture,
                into,
                offset: 0,
                row_pitch: TEXEL * SIDE,
                size: [SIDE; 2],
                plane: crate::render::Plane::Whole,
                at: ReadAt::BASE,
            };
            let mut work = vec![
                RenderWork::Pass(pass_a(store, handles)),
                RenderWork::Pass(pass_b(handles)),
                // The kernel binds its output at buffer 1; buffer 0 is bound to the same buffer
                // and not read.
                RenderWork::Dispatch(Dispatch {
                    pipeline: read,
                    buffers: vec![(samples, 0), (samples, 0)],
                    textures: vec![handles.target],
                    grid: [SIDE, SIDE, 1],
                    threads: [SIDE, SIDE, 1],
                }),
                read_resolve(handles.resolve_b, resolved_b),
            ];
            let mut names = vec!["pass-a", "pass-b", "read:samples", "read:resolve-b"];
            if resolves {
                work.push(read_resolve(handles.resolve_a, resolved_a));
                names.push("read:resolve-a");
            }

            let outcome = call(
                sites,
                &format!("render/{name}"),
                side.render(&work),
                |outcome| outcome.status.clone(),
            )?;
            let mut pieces = Node::branch(format!("pieces/{name}"), Vec::new());
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
            if let Some((what, failure)) = names
                .iter()
                .enumerate()
                .find_map(|(at, what)| outcome.refused.get(at)?.as_ref().map(|f| (what, f)))
            {
                return Err(Abort {
                    at: format!("pieces/{name}/{what}"),
                    failure: failure.clone(),
                });
            }

            let bytes = call(
                sites,
                &format!("read-back/samples/{name}"),
                side.read_buffer(samples, 0, SAMPLE_BYTES as usize),
                |_| Value::Absent,
            )?;
            sites.push(Node::leaf(
                format!("samples/{name}"),
                Observed::bytes(bytes, samples_layout()),
            ));
            let bytes = call(
                sites,
                &format!("read-back/resolve-b/{name}"),
                side.read_buffer(resolved_b, 0, RESOLVE_BYTES as usize),
                |_| Value::Absent,
            )?;
            sites.push(Node::leaf(
                format!("resolve-b/{name}"),
                Observed::bytes(bytes, resolve_layout()),
            ));
            if resolves {
                let bytes = call(
                    sites,
                    &format!("read-back/resolve-a/{name}"),
                    side.read_buffer(resolved_a, 0, RESOLVE_BYTES as usize),
                    |_| Value::Absent,
                )?;
                sites.push(Node::leaf(
                    format!("resolve-a/{name}"),
                    Observed::bytes(bytes, resolve_layout()),
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pass A's sixteen cells state every sample mask of four samples, the empty one by drawing
    /// nothing, and pass B's mask names samples cell 0 left at the clear.
    #[test]
    fn pass_a_states_every_mask_and_pass_b_draws_over_the_clear() {
        let masks: Vec<u32> = (1..GRID * GRID).collect();
        assert_eq!(masks, (1..16).collect::<Vec<u32>>());
        assert_ne!(B_MASK, 0);
        assert_eq!(B_MASK & !0b1111, 0);
    }

    /// Every value the case writes, and every average of four of them, is exact in `half`: the
    /// clear's channels and white are multiples of a quarter, and an average of four is a
    /// multiple of a sixteenth no larger than 1.
    #[test]
    fn every_value_and_every_average_is_exact_in_half() {
        for channel in CLEAR {
            for covered in 0..=4u32 {
                let average = (channel * f64::from(4 - covered) + f64::from(covered)) / 4.0;
                assert_eq!((average * 16.0).fract(), 0.0, "{channel} by {covered}");
                assert!(average <= 1.0);
            }
        }
    }

    /// The grid covers the attachment, and the attachment is the extent the kernel reads.
    #[test]
    fn the_grid_covers_the_side_the_kernel_reads() {
        assert_eq!(GRID * CELL, SIDE);
        assert_eq!(SIDE, 8, "wm_read_samples reads an 8 by 8 texture");
        assert_eq!(SAMPLES, 4, "wm_read_samples reads four samples");
        assert_eq!(
            samples_layout().row_pitch * samples_layout().rows,
            SAMPLE_BYTES
        );
        assert_eq!(
            resolve_layout().row_pitch * resolve_layout().rows,
            RESOLVE_BYTES
        );
    }
}
