//! What a sampler state bound with LOD clamps of its own reads
//! (`setSamplerState:lodMinClamp:lodMaxClamp:atIndex:`, `setFragmentSamplerState:...` and the
//! range forms): whether the bind's pair replaces the descriptor's clamps or narrows them, what a
//! pair whose lower bound is above its upper reads, and whether a plain bind after a clamped one
//! reads the descriptor's clamps again.
//!
//! **What Metal answered** on the M3 Max, in both cases alike: the bind's pair **replaces** the
//! descriptor's clamps, `(0, 4)` reading every level a state of `(1, 3)` never reads; a pair whose
//! lower bound is above its upper reads every sample at the upper bound's level, as the clamps
//! `(max, max)`; and a plain bind after a clamped one reads the descriptor's own clamps again.
//! Metal's validation layer refuses none of the pairs. `tests` hold these of the recordings.
//!
//! One sampler state in both cases: nearest filters, a nearest mip filter, clamped to the edge,
//! `lodMinClamp` 1 and `lodMaxClamp` 3. The texture is 16 by 16 `R32Float` with all five levels,
//! every texel of level `l` holding `l + 1`, so a sample names the level it read and nothing else.
//!
//! **`sampler-bind-lod-clamps-kernel`**: the owned fixture `unv_runtime_levels`' `unv_rpos2d`
//! samples through the runtime sampler at `[[sampler(0)]]` at an explicit level a lane reads,
//! levels 0 to 7 (5 to 7 past the texture's last). One compute pass: each row binds the state
//! (plainly, with a pair, or through the range form), binds the output at the row's eight words
//! and dispatches the eight lanes, so a row after another reads the binding the row made.
//!
//! **`sampler-bind-lod-clamps-fragment`**: `alv_levels`' `alv_rect` covers a 10 by 2 target, and
//! `mip_level_sampling`'s `mip_implicit_fragment` samples through the fragment sampler at
//! `[[sampler(0)]]` at the level its derivatives give: exactly `q` in the `q`-th quad along `x`,
//! levels 0 to 4. One render pass a row, which clears the target and draws once after its binds;
//! the two rows that bind twice draw after each bind, so the target holds the second draw's.
//!
//! Nothing here can fault a GPU: one pass or one draw a row, every texture read-only and inside
//! Metal's limits, no loop.

use crate::case::{
    call, drawn_or_refused, made, Abort, Case, Failure, Handle, Input, MetalSource, SamplerSpec,
    Side,
};
use crate::compute_pass::{ComputePass, ComputeStep, Dispatching};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::render::{
    ColorState, LodClamp, PassColor, Plane, ReadAt, RenderPass, RenderPipelineSpec, RenderStep,
    RenderWork, StageFunction, TargetSpec,
};
use crate::texture::{TextureRequest, Upload};
use crate::transcript::{
    Aspect, ByteLayout, Digest, Element, Gaps, Node, Observed, Region, Sites, Value,
};
use metal2vulkan::metal::{PixelFormat, PrimitiveType, SamplerMipFilter};

/// The kernel: a sample at `(p.x, p.y)` at `level(p.z)` through the runtime sampler at slot 0.
const KERNEL: Fixture = Fixture {
    name: "unv_runtime_levels",
    entry: "unv_rpos2d",
};

/// The vertex function: two triangles over the target.
const VERTEX: Fixture = Fixture {
    name: "alv_levels",
    entry: "alv_rect",
};

/// The fragment function: an implicit-level sample, level `q` in the `q`-th quad along `x`.
const FRAGMENT: Fixture = Fixture {
    name: "mip_level_sampling",
    entry: "mip_implicit_fragment",
};

/// The texture's side, which `mip_implicit_fragment`'s level rule is written for.
const SIDE: u32 = 16;

/// Its levels: 16, 8, 4, 2 and 1 texels a side.
const LEVELS: u32 = 5;

/// The state's own clamps.
const OWN: LodClamp = LodClamp::new(1.0, 3.0);

/// The levels the kernel's eight lanes sample at.
const KERNEL_LEVELS: [f32; 8] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];

/// The coordinate every lane samples at; every texel of a level reads alike.
const AT: [f32; 2] = [0.3, 0.6];

/// The byte every output word holds before a row runs.
const FILL: u8 = 0xab;

/// The fragment case's target: five quads along `x`, one along `y`.
const WIDTH: u32 = 10;
const HEIGHT: u32 = 2;

/// The target's format: what the fragment function returns, read back as its number.
const FORMAT: PixelFormat = PixelFormat::R32Float;

/// Bytes one row of the target takes.
const ROW_PITCH: u32 = WIDTH * 4;

/// Bytes one row's target takes, its slice of the read-back buffer.
const TARGET_BYTES: u32 = HEIGHT * ROW_PITCH;

/// The one colour attachment of the fragment case's pipeline.
const COLORS: [Option<ColorState>; 1] = [Some(ColorState::new(FORMAT))];

/// One binding a row makes.
#[derive(Clone, Copy, Debug)]
enum Bind {
    /// `setSamplerState:atIndex:`.
    Plain,
    /// `setSamplerState:lodMinClamp:lodMaxClamp:atIndex:`.
    Clamped(LodClamp),
    /// `setSamplerStates:lodMinClamps:lodMaxClamps:withRange:` of the one state at slot 0.
    Range(LodClamp),
}

/// A row: its name and the bindings it makes, in order. The kernel dispatches after the last, the
/// fragment case draws after each.
struct Row {
    name: &'static str,
    binds: &'static [Bind],
}

/// Every row. The pairs are the state's own, one wider on both sides, one inside it, one below and
/// one above it overlapping it, one wholly below and one wholly above it, one unbounded, two whose
/// lower bound is above the upper, one whose lower bound is below 0 and one wholly past the
/// texture's levels.
///
/// No pair holds a NaN: Metal documents no NaN clamp, and the M3 Max read `(NaN, 4)` as the
/// state's own clamps and `(0, NaN)` as levels up to 2 (job m3-max-70005), no rule a case can hold
/// a device to.
const ROWS: [Row; 17] = [
    Row {
        name: "plain",
        binds: &[Bind::Plain],
    },
    Row {
        name: "own",
        binds: &[Bind::Clamped(OWN)],
    },
    Row {
        name: "wider",
        binds: &[Bind::Clamped(LodClamp::new(0.0, 4.0))],
    },
    Row {
        name: "narrower",
        binds: &[Bind::Clamped(LodClamp::new(2.0, 2.0))],
    },
    Row {
        name: "lower",
        binds: &[Bind::Clamped(LodClamp::new(0.0, 2.0))],
    },
    Row {
        name: "upper",
        binds: &[Bind::Clamped(LodClamp::new(2.0, 4.0))],
    },
    Row {
        name: "wholly-below",
        binds: &[Bind::Clamped(LodClamp::new(0.0, 0.0))],
    },
    Row {
        name: "wholly-above",
        binds: &[Bind::Clamped(LodClamp::new(4.0, 4.0))],
    },
    Row {
        name: "unbounded",
        binds: &[Bind::Clamped(LodClamp::new(0.0, f32::MAX))],
    },
    Row {
        name: "min-above-max",
        binds: &[Bind::Clamped(LodClamp::new(3.0, 1.0))],
    },
    Row {
        name: "min-above-max-outside",
        binds: &[Bind::Clamped(LodClamp::new(4.0, 0.0))],
    },
    Row {
        name: "negative-lower",
        binds: &[Bind::Clamped(LodClamp::new(-2.0, 2.0))],
    },
    Row {
        name: "past-the-levels",
        binds: &[Bind::Clamped(LodClamp::new(6.0, 9.0))],
    },
    Row {
        name: "range-form",
        binds: &[Bind::Range(LodClamp::new(0.0, 4.0))],
    },
    Row {
        name: "range-form-own",
        binds: &[Bind::Range(OWN)],
    },
    Row {
        name: "plain-after-clamped",
        binds: &[Bind::Clamped(LodClamp::new(0.0, 4.0)), Bind::Plain],
    },
    Row {
        name: "clamped-after-plain",
        binds: &[Bind::Plain, Bind::Clamped(LodClamp::new(0.0, 4.0))],
    },
];

/// The state both cases make.
fn spec() -> SamplerSpec {
    let mut spec = SamplerSpec::new().mipped(SamplerMipFilter::Nearest);
    spec.lod_min_clamp = OWN.min;
    spec.lod_max_clamp = OWN.max;
    spec
}

/// The texels of level `level`: every one `level + 1`.
fn level_bytes(level: u32) -> Vec<u8> {
    let side = SIDE >> level;
    let texel = (level + 1) as f32;
    (0..side * side).flat_map(|_| texel.to_le_bytes()).collect()
}

/// Every level's texels, level 0 first.
fn chain_bytes() -> Vec<Vec<u8>> {
    (0..LEVELS).map(level_bytes).collect()
}

/// The uploads that load `chain`, one a level.
fn chain_uploads(chain: &[Vec<u8>]) -> Vec<Upload<'_>> {
    chain
        .iter()
        .enumerate()
        .map(|(level, bytes)| {
            let side = SIDE >> level as u32;
            Upload {
                level: level as u32,
                slice: 0,
                origin: [0, 0, 0],
                size: [side, side, 1],
                bytes,
                row_pitch: side * 4,
                rows: side,
                image_pitch: 0,
            }
        })
        .collect()
}

/// The kernel's parameter buffer: `(x, y, level, 0)` a lane.
fn kernel_params() -> Vec<u8> {
    KERNEL_LEVELS
        .iter()
        .flat_map(|&level| [AT[0], AT[1], level, 0.0])
        .flat_map(f32::to_le_bytes)
        .collect()
}

/// Bytes one kernel row writes: a `float` a lane.
const KERNEL_ROW: u32 = KERNEL_LEVELS.len() as u32 * 4;

/// Bytes the kernel case's output takes.
fn kernel_output() -> Vec<u8> {
    vec![FILL; ROWS.len() * KERNEL_ROW as usize]
}

/// The layout a kernel row's words are read under: one row a lane's eight levels.
fn kernel_layout() -> ByteLayout {
    ByteLayout {
        region: Region::Buffer {
            offset: 0,
            length: u64::from(KERNEL_ROW),
        },
        element: Element::F32,
        row_offset: 0,
        row_elements: KERNEL_LEVELS.len() as u32,
        row_pitch: KERNEL_ROW,
        rows: 1,
        image_pitch: KERNEL_ROW,
        images: 1,
        gaps: Gaps::Compared,
    }
}

/// The layout a fragment row's target is read under: a row of the layout a row of texels.
fn fragment_layout() -> ByteLayout {
    ByteLayout {
        region: Region::Texture {
            level: 0,
            slice: 0,
            aspect: Aspect::Color,
            sample: 0,
            origin: [0, 0, 0],
            size: [WIDTH, HEIGHT, 1],
        },
        element: Element::F32,
        row_offset: 0,
        row_elements: WIDTH,
        row_pitch: ROW_PITCH,
        rows: HEIGHT,
        image_pitch: TARGET_BYTES,
        images: 1,
        gaps: Gaps::Compared,
    }
}

/// The kernel case's compute steps for `binds`, the state being `sampler`.
fn compute_binds(sampler: Handle, binds: &[Bind]) -> Vec<ComputeStep> {
    binds
        .iter()
        .map(|bind| match *bind {
            Bind::Plain => ComputeStep::Sampler { sampler, index: 0 },
            Bind::Clamped(clamp) => ComputeStep::ClampedSampler {
                sampler,
                lod_min: clamp.min,
                lod_max: clamp.max,
                index: 0,
            },
            Bind::Range(clamp) => ComputeStep::ClampedSamplers {
                samplers: vec![(sampler, clamp.min, clamp.max)],
                first: 0,
            },
        })
        .collect()
}

/// The fragment case's render step for `bind`, the state being `sampler`.
fn render_bind(sampler: Handle, bind: Bind) -> RenderStep {
    match bind {
        Bind::Plain => RenderStep::FragmentSampler { sampler, index: 0 },
        Bind::Clamped(clamp) => RenderStep::FragmentSamplerClamped {
            sampler,
            clamp,
            index: 0,
        },
        Bind::Range(clamp) => RenderStep::FragmentSamplersClamped {
            samplers: vec![(sampler, clamp)],
            first: 0,
        },
    }
}

/// The draw of the two triangles `alv_rect` places.
const RECT_DRAW: RenderStep = RenderStep::Draw {
    primitive: PrimitiveType::Triangle,
    first: 0,
    count: 6,
    instances: 1,
};

/// The kernel case's pass: the pipeline, the texture and the parameters once, then each row's
/// binds, its output's words and its dispatch.
fn kernel_steps(handles: [Handle; 5]) -> Vec<ComputeStep> {
    let [pipeline, texture, sampler, params, out] = handles;
    let mut steps = vec![
        ComputeStep::Pipeline(pipeline),
        ComputeStep::Texture { texture, index: 0 },
        ComputeStep::Buffer {
            buffer: params,
            offset: 0,
            index: 0,
        },
    ];
    for (at, row) in ROWS.iter().enumerate() {
        steps.extend(compute_binds(sampler, row.binds));
        steps.push(ComputeStep::Buffer {
            buffer: out,
            offset: at as u64 * u64::from(KERNEL_ROW),
            index: 1,
        });
        steps.push(ComputeStep::DispatchThreadgroups {
            groups: [1, 1, 1],
            threads: [KERNEL_LEVELS.len() as u32, 1, 1],
        });
    }
    steps
}

/// The fragment case's pipeline: the vertex function of the library the call names, the fragment
/// function of `fragment`'s.
fn pipeline_spec(fragment: Handle) -> RenderPipelineSpec<'static> {
    RenderPipelineSpec::of(
        StageFunction::named(VERTEX.entry),
        Some(StageFunction::of(fragment, FRAGMENT.entry)),
        &COLORS,
    )
}

/// The steps of a fragment row's pass: the pipeline and the texture, then a draw after each bind.
fn fragment_steps(
    pipeline: Handle,
    texture: Handle,
    sampler: Handle,
    row: &Row,
) -> Vec<RenderStep> {
    let mut steps = vec![
        RenderStep::Pipeline(pipeline),
        RenderStep::FragmentTexture { texture, index: 0 },
    ];
    for bind in row.binds {
        steps.push(render_bind(sampler, *bind));
        steps.push(RECT_DRAW);
    }
    steps
}

/// The rows' names and pairs, as the input fingerprint covers them.
fn rows_words() -> Vec<u8> {
    let mut words = Vec::new();
    for row in &ROWS {
        words.extend(row.name.as_bytes());
        for bind in row.binds {
            let (kind, clamp) = match *bind {
                Bind::Plain => (0u8, LodClamp::new(0.0, 0.0)),
                Bind::Clamped(clamp) => (1, clamp),
                Bind::Range(clamp) => (2, clamp),
            };
            words.push(kind);
            words.extend(clamp.words());
        }
    }
    words
}

/// The fixtures' two texts, as a case's shaders and inputs name them.
fn shader_refs(fixtures: &[Fixture]) -> Result<Vec<ShaderRef>, String> {
    let mut shaders = Vec::new();
    for fixture in fixtures {
        for extension in [RUNTIME_TEXT, METAL_TEXT] {
            shaders.push(ShaderRef {
                fixture: format!("{}{extension}", fixture.name),
                entry: fixture.entry.to_owned(),
                source_sha256: crate::sha256_hex(fixture.text(extension)?.as_bytes()),
            });
        }
    }
    Ok(shaders)
}

/// The inputs both cases share: each fixture's texts and entry, the texture's levels, the state
/// and the rows.
fn shared_inputs(fixtures: &[Fixture]) -> Result<Vec<Input>, String> {
    let mut inputs = Vec::new();
    for fixture in fixtures {
        inputs.push(Input {
            name: format!("air:{}", fixture.name),
            digest: Digest::sha256(fixture.text(RUNTIME_TEXT)?.as_bytes()),
        });
        inputs.push(Input {
            name: format!("air-typed:{}", fixture.name),
            digest: Digest::sha256(fixture.text(METAL_TEXT)?.as_bytes()),
        });
        inputs.push(Input::bytes(
            format!("entry:{}", fixture.name),
            fixture.entry.as_bytes(),
        ));
    }
    for (level, bytes) in chain_bytes().iter().enumerate() {
        inputs.push(Input::bytes(format!("texture:level-{level}"), bytes));
    }
    inputs.push(Input::bytes("sampler", &spec().words()));
    inputs.push(Input::bytes("rows", &rows_words()));
    Ok(inputs)
}

/// The library of `fixture`, made on `side` under `site`.
fn library(
    side: &mut dyn Side,
    sites: &mut Sites,
    fixture: Fixture,
    site: &str,
) -> Result<Handle, Abort> {
    let loaded = fixture.loaded().map_err(|why| Abort {
        at: site.to_owned(),
        failure: Failure::named("FixtureNotRead", why),
    })?;
    made(sites, site, side.library(&loaded.source()))
}

/// The mipmapped texture, made on `side`.
fn texture(side: &mut dyn Side, sites: &mut Sites) -> Result<Handle, Abort> {
    let chain = chain_bytes();
    let uploads = chain_uploads(&chain);
    let request = TextureRequest::read_2d(FORMAT.as_raw(), SIDE, SIDE)
        .levels(LEVELS)
        .loaded(&uploads);
    made(sites, "texture", side.texture(&request))
}

/// The kernel case.
pub struct KernelBinds;

/// The fragment case.
pub struct FragmentBinds;

/// The kernel case's name.
pub const KERNEL_NAME: &str = "sampler-bind-lod-clamps-kernel";

/// The fragment case's name.
pub const FRAGMENT_NAME: &str = "sampler-bind-lod-clamps-fragment";

impl Case for KernelBinds {
    fn name(&self) -> &'static str {
        KERNEL_NAME
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        shader_refs(&[KERNEL])
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let mut inputs = shared_inputs(&[KERNEL])?;
        inputs.push(Input::bytes("buffer:params", &kernel_params()));
        inputs.push(Input::bytes("buffer:out", &kernel_output()));
        let steps = kernel_steps([Handle(0); 5]);
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
        let library = library(side, sites, KERNEL, "library")?;
        let pipeline = made(
            sites,
            "pipeline",
            side.compute_pipeline(library, KERNEL.entry),
        )?;
        let texture = texture(side, sites)?;
        let sampler = made(sites, "sampler", side.sampler(&spec()))?;
        let params = made(sites, "buffer/params", side.buffer(&kernel_params()))?;
        let output = kernel_output();
        let out = made(sites, "buffer/out", side.buffer(&output))?;
        let steps = kernel_steps([pipeline, texture, sampler, params, out]);
        let outcome = made(
            sites,
            "pass",
            side.compute_pass(&ComputePass {
                dispatching: Dispatching::Serial,
                steps: &steps,
            }),
        )?;
        sites.push(outcome.node("outcome", &steps));
        let bytes = call(
            sites,
            "read-back",
            side.read_buffer(out, 0, output.len()),
            |_| Value::Absent,
        )?;
        for (at, row) in ROWS.iter().enumerate() {
            let first = at * KERNEL_ROW as usize;
            let words = bytes[first..first + KERNEL_ROW as usize].to_vec();
            sites.push(Node::branch(row.name, Vec::new()).with(Node::leaf(
                "sampled",
                Observed::bytes(words, kernel_layout()),
            )));
        }
        Ok(())
    }
}

impl Case for FragmentBinds {
    fn name(&self) -> &'static str {
        FRAGMENT_NAME
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        shader_refs(&[VERTEX, FRAGMENT])
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let mut inputs = shared_inputs(&[VERTEX, FRAGMENT])?;
        inputs.push(Input::bytes("target", &target_spec().words()));
        // The fragment library is the second the run makes.
        inputs.push(Input::bytes("pipeline", &pipeline_spec(Handle(1)).words()));
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let vertex = library(side, sites, VERTEX, "library/vertex")?;
        let fragment = library(side, sites, FRAGMENT, "library/fragment")?;
        let pipeline = made(
            sites,
            "pipeline",
            side.render_pipeline(vertex, &pipeline_spec(fragment)),
        )?;
        let texture = texture(side, sites)?;
        let sampler = made(sites, "sampler", side.sampler(&spec()))?;
        let target = made(sites, "target", side.target(&target_spec()))?;
        let length = ROWS.len() * TARGET_BYTES as usize;
        let read_back = made(sites, "buffer/read-back", side.zeroed_buffer(length))?;

        let mut work = Vec::with_capacity(2 * ROWS.len());
        for (at, row) in ROWS.iter().enumerate() {
            work.push(RenderWork::Pass(
                RenderPass::clearing(vec![PassColor::cleared(target, [0.0; 4])])
                    .drawing(fragment_steps(pipeline, texture, sampler, row)),
            ));
            work.push(RenderWork::ReadTexture {
                texture: target,
                into: read_back,
                offset: at as u64 * u64::from(TARGET_BYTES),
                row_pitch: ROW_PITCH,
                size: [WIDTH, HEIGHT],
                plane: Plane::Whole,
                at: ReadAt::BASE,
            });
        }
        let outcome = call(sites, "render", side.render(&work), |outcome| {
            outcome.status.clone()
        })?;
        let bytes = call(
            sites,
            "read-back",
            side.read_buffer(read_back, 0, length),
            |_| Value::Absent,
        )?;
        for (at, row) in ROWS.iter().enumerate() {
            let first = at * TARGET_BYTES as usize;
            let texels = bytes[first..first + TARGET_BYTES as usize].to_vec();
            let node = Node::branch(row.name, Vec::new()).with(Node::leaf(
                "texels",
                Observed::bytes(texels, fragment_layout()),
            ));
            let refused = (outcome.refusal(2 * at)).or_else(|| outcome.refusal(2 * at + 1));
            sites.push(drawn_or_refused(node, refused));
        }
        Ok(())
    }
}

/// The fragment case's target: private, since the blit reads it back.
fn target_spec() -> TargetSpec {
    TargetSpec::read(FORMAT, [WIDTH, HEIGHT])
}

/// Both cases.
pub fn all() -> Vec<Box<dyn Case>> {
    vec![Box::new(KernelBinds), Box::new(FragmentBinds)]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The clamps a row's last binding samples with by the rule Metal's recordings show: a plain
    /// binding the state's own; a pair replaces them, and a pair whose lower bound is above its
    /// upper is the upper bound twice.
    fn clamps(row: &Row) -> (f32, f32) {
        match row.binds.last() {
            Some(Bind::Clamped(clamp) | Bind::Range(clamp)) if clamp.min > clamp.max => {
                (clamp.max, clamp.max)
            }
            Some(Bind::Clamped(clamp) | Bind::Range(clamp)) => (clamp.min, clamp.max),
            Some(Bind::Plain) | None => (OWN.min, OWN.max),
        }
    }

    /// What a sample at level `at` reads under `(lo, hi)`: the level clamped into them, then to the
    /// texture's last, as its texels' `level + 1`. Every level and bound here is whole, so a
    /// nearest mip filter reads the clamped level itself.
    fn reads(at: f32, (lo, hi): (f32, f32)) -> f32 {
        at.max(lo).min(hi).clamp(0.0, (LEVELS - 1) as f32) + 1.0
    }

    /// The `float`s of `bytes`.
    fn floats(bytes: &[u8]) -> Vec<f32> {
        let (words, _) = bytes.as_chunks::<4>();
        words.iter().map(|word| f32::from_le_bytes(*word)).collect()
    }

    /// The bytes every outcome of `case`'s recording holds at `row/leaf`.
    fn recorded(case: &str, row: &str, leaf: &str) -> Vec<Vec<u8>> {
        let recording =
            crate::recording::read(case).unwrap_or_else(|e| panic!("{case}: no recording: {e}"));
        let mut out = Vec::new();
        for answer in &recording.answers {
            for outcome in &answer.outcomes {
                let held = (outcome.transcript.root.child(row))
                    .and_then(|node| node.child(leaf))
                    .map(|node| &node.observed.value);
                match held {
                    Some(Value::Bytes(bytes)) => out.push(bytes.clone()),
                    other => panic!("{case}: {row}/{leaf} holds {other:?}"),
                }
            }
        }
        assert!(!out.is_empty(), "{case}: no Metal answer");
        out
    }

    /// Metal's kernel answer is the rule's at every lane of every row: a pair replaces the
    /// descriptor's clamps, and a lower bound above the upper reads at the upper.
    #[test]
    fn metal_replaces_the_clamps_at_a_kernel_bind() {
        for row in &ROWS {
            let want: Vec<f32> = KERNEL_LEVELS
                .iter()
                .map(|&at| reads(at, clamps(row)))
                .collect();
            for bytes in recorded(KERNEL_NAME, row.name, "sampled") {
                assert_eq!(floats(&bytes), want, "{}", row.name);
            }
        }
    }

    /// Metal's fragment answer is the same rule's at every pixel: quad `q` samples at level `q`.
    #[test]
    fn metal_replaces_the_clamps_at_a_fragment_bind() {
        for row in &ROWS {
            let line = (0..WIDTH).map(|x| reads((x / 2) as f32, clamps(row)));
            let want: Vec<f32> = line.clone().chain(line).collect();
            for bytes in recorded(FRAGMENT_NAME, row.name, "texels") {
                assert_eq!(floats(&bytes), want, "{}", row.name);
            }
        }
    }

    /// The rows tell replacing from narrowing: under narrowing, `(0, 4)` would read as the state's
    /// own `(1, 3)`. And the state's clamps are not the texture's, so a plain binding is a
    /// question too.
    #[test]
    fn the_rows_part_replacing_from_narrowing() {
        let own = KERNEL_LEVELS.map(|at| reads(at, (OWN.min, OWN.max)));
        let wider = KERNEL_LEVELS.map(|at| reads(at, (0.0, 4.0)));
        let unclamped = KERNEL_LEVELS.map(|at| reads(at, (0.0, f32::MAX)));
        assert_ne!(own, wider);
        assert_ne!(own, unclamped);
    }
}
