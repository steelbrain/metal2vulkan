//! A render encoder's **barriers and residency calls**: the memory barrier a later draw needs to
//! see an earlier draw's texture writes, and the `useResource` and `useHeap` calls that declare
//! what its draws may reach.
//!
//! | case | what it asks |
//! |---|---|
//! | `render-state-render-barriers` | whether both render memory barriers are answered between two draws of one pass, over a texture one of them reads and a texture the other writes, with the four `useResource` forms around them |
//! | `render-state-heap-residency` | whether the four `useHeap` forms are answered on an empty heap, and that the pass's own draw is untouched by them |
//!
//! **The barrier case does not read what the draw before it wrote, and that is a measurement
//! rather than a preference.** It was written that way first: one pass whose first draw's fragment
//! function wrote a texel of a scratch texture under every pixel, a
//! `memoryBarrierWithScope:MTLBarrierScopeTextures afterStages:MTLRenderStageFragment
//! beforeStages:MTLRenderStageFragment` and a `memoryBarrierWithResources:` over the same texture,
//! and a second draw that read it back. On an Apple M3 Max (macOS 27.0.1, build 26A434) **110 of
//! its 256 texels came back as zeros**, identically in all eight runs, in a region bounded by the
//! diagonal the rectangle's two triangles share. So the write was not made visible to the later
//! draw of the same pass, deterministically, and the suite will not hold either side to a figure
//! that is one GPU's tile schedule: committing it would have recorded that pattern as *the*
//! answer and asked every implementation to reproduce it.
//!
//! What the case does instead is two draws whose target regions and whose textures are disjoint —
//! one reads a texture the host filled, the other writes a texture nothing reads — with both
//! barriers between them. So its claim is that **Metal answers both barrier calls** and that the
//! two draws stored exactly what their own bindings say, in their own halves of the target. It is
//! **not** a claim that a barrier orders a read after a write inside one pass; nothing here could
//! show that, and the measurement above is why.
//!
//! **What the residency calls claim here is narrow, and the module says so rather than implying
//! more.** Every texture and buffer these passes reach is bound at a slot, which is the one way of
//! reaching a resource Metal needs no residency declaration for. So the claim is that **Metal
//! answers each of the eight calls** — `useResource:usage:` and `useResources:count:usage:` with
//! and without stages, `useHeap:` and `useHeaps:count:` with and without stages — and that the
//! pass's own texels are what they are without them. It is **not** a claim that residency makes a
//! resource reachable: no function here reads one through an address or a resource ID held in
//! memory, so nothing here could show that. A case that could needs a fragment function that reads
//! an argument buffer, and the fixtures hold none whose stage input a committed vertex function
//! produces; that is a packet of its own.
//!
//! The heap is a sparse heap holding nothing, because that is the one heap the suite's vocabulary
//! can make. A venue that cannot make one refuses the call typed, which is the venue's own gap and
//! not ours, and it costs this one case rather than the barrier case beside it — which is why the
//! two are not one case.
//!
//! Nothing here can fault a GPU: one pass a case, one or two draws of six vertices over a 16 by 16
//! or 4 by 4 target, no loop in the fixture, and a heap that holds no memory until a mapping takes
//! a tile, which nothing here asks for.

use crate::allowance::Allowance;
use crate::case::{
    call, made, Abort, Case, Failure, Handle, Input, MetalSource, Side, TextureRole, TextureSpec,
};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::render::{
    ColorState, PassColor, Plane, ReadAt, RenderPass, RenderPipelineSpec, RenderStep, RenderWork,
    ScissorBox, StepResource, TargetSpec,
};
use crate::sparse_query::SparseHeapSpec;
use crate::transcript::{
    Aspect, ByteLayout, Digest, Element, Gaps, Node, Observed, Region, Sites, Value,
};
use metal2vulkan::metal::{BarrierScope, PixelFormat, PrimitiveType, RenderStages, ResourceUsage};
use metal2vulkan_runtime::Degradation;

/// The format both targets and the scratch texture carry: a `float` stored as it is, so the whole
/// expectation is arithmetic.
const FORMAT: PixelFormat = PixelFormat::RGBA32Float;

/// Bytes one texel of [`FORMAT`] takes.
const TEXEL: u32 = 16;

/// The side of the barrier case's target and scratch texture, in texels.
const SIDE: u32 = 16;

/// The side of the heap case's target.
const SMALL: u32 = 4;

/// The clear a pass loads its target with, which no draw of either case can store.
const CLEAR: f32 = -7.5;

/// The owned fixture both cases draw with.
const FIXTURE: &str = "tob_renders";

/// `tob_rect`: a rectangle from a buffer's two clip-space corners, six vertices.
const RECT: &str = "tob_rect";

/// `tob_solid`: the `float4` a fragment buffer holds, unchanged.
const SOLID: &str = "tob_solid";

/// `tob_write`: writes `(x / 8, y / 8, 0.75, 1)` into texture 0's texel under the pixel and
/// returns opaque black.
const WRITE: &str = "tob_write";

/// `tob_read`: the texel of texture 0 under the pixel.
const READ: &str = "tob_read";

/// The entry points a case's shader reference names.
const ENTRIES: &str = "tob_rect+tob_solid+tob_write+tob_read";

/// The draw a rectangle takes: two triangles of six vertices, one instance.
const RECT_DRAW: RenderStep = RenderStep::Draw {
    primitive: PrimitiveType::Triangle,
    first: 0,
    count: 6,
    instances: 1,
};

/// The rectangle covering the whole target, which the heap case's one draw takes.
const WHOLE: [f32; 4] = [-1.0, -1.0, 1.0, 1.0];

/// The rectangle as the vertex function's `constant float4 &` binding takes it.
fn rect_bytes() -> Vec<u8> {
    WHOLE.iter().flat_map(|edge| edge.to_le_bytes()).collect()
}

/// The target's left half, in pixel columns: where the barrier case's reading draw lands.
const LEFT: ScissorBox = ScissorBox::new(0, 0, (SIDE / 2) as u64, SIDE as u64);

/// The target's right half: where its writing draw lands.
const RIGHT: ScissorBox = ScissorBox::new((SIDE / 2) as u64, 0, (SIDE / 2) as u64, SIDE as u64);

/// What the host writes into the texture the barrier case's first draw reads: values no draw of
/// either case can store, each texel its own.
fn source_texel(x: u32, y: u32) -> [f32; 4] {
    [32.0 + x as f32, 64.0 + y as f32, 0.5, 2.0]
}

/// That texture's bytes.
fn source_texels() -> Vec<u8> {
    (0..SIDE)
        .flat_map(|y| (0..SIDE).map(move |x| source_texel(x, y)))
        .flat_map(|texel| {
            texel
                .iter()
                .flat_map(|channel| channel.to_le_bytes())
                .collect::<Vec<u8>>()
        })
        .collect()
}

/// The texture the barrier case's reading draw reads: of `FORMAT` at the target's extent,
/// shader-read, filled by the host.
fn source_spec() -> TextureSpec {
    TextureSpec::new(FORMAT, [SIDE; 2], TEXEL * SIDE, TextureRole::Read)
}

/// The colour the heap case's draw stores, which is neither its clear, nor opaque black, nor any
/// texel the barrier case's textures hold.
const COLOR: [f32; 4] = [-0.5, 3.75, 0.5, 0.25];

/// The colour as the fragment function's binding takes it.
fn color_bytes() -> Vec<u8> {
    COLOR
        .iter()
        .flat_map(|channel| channel.to_le_bytes())
        .collect()
}

/// The texture the barrier case's second draw writes: of [`FORMAT`] at the target's extent,
/// shader-write alone, and nothing reads it. It is here because `useResource:usage:` with a write
/// usage needs a resource a function writes, and because the resource barrier needs a resource to
/// name beside the one the other draw reads.
fn scratch_spec() -> TextureSpec {
    TextureSpec::new(FORMAT, [SIDE; 2], TEXEL * SIDE, TextureRole::Write)
}

/// The words a texture's own descriptor fingerprints as.
///
/// A case usually fingerprints a texture by the texels it hands in, and the scratch texture is
/// handed none — nothing reads it — so its descriptor is what the fingerprint has to cover: the
/// format, the extent, the row the case states and the usage its role names.
fn texture_words(spec: TextureSpec) -> Vec<u8> {
    let mut bytes = spec.format.as_raw().to_le_bytes().to_vec();
    for axis in spec.size {
        bytes.extend(axis.to_le_bytes());
    }
    bytes.extend(spec.row_pitch.to_le_bytes());
    bytes.push(match spec.role {
        TextureRole::Read => 0,
        TextureRole::Write => 1,
        TextureRole::ReadWrite => 2,
        TextureRole::RenderTarget => 3,
    });
    bytes
}

/// The target a case draws into: private, of `MTLTextureUsageRenderTarget` alone.
fn target_spec(side: u32) -> TargetSpec {
    TargetSpec::plain(FORMAT, [side; 2])
}

/// Bytes the whole of a `side` by `side` level 0 takes.
fn plane(side: u32) -> u32 {
    TEXEL * side * side
}

/// The layout a target's whole level 0 is read under.
fn layout(side: u32) -> ByteLayout {
    let row_pitch = TEXEL * side;
    ByteLayout {
        region: Region::Texture {
            level: 0,
            slice: 0,
            aspect: Aspect::Color,
            sample: 0,
            origin: [0, 0, 0],
            size: [side, side, 1],
        },
        element: Element::F32,
        row_offset: 0,
        row_elements: 4 * side,
        row_pitch,
        rows: side,
        image_pitch: row_pitch * side,
        images: 1,
        gaps: Gaps::Compared,
    }
}

/// The piece that copies a target's level 0 into a buffer the host reads.
fn read_target(texture: Handle, into: Handle, side: u32) -> RenderWork {
    RenderWork::ReadTexture {
        texture,
        into,
        offset: 0,
        row_pitch: TEXEL * side,
        size: [side; 2],
        plane: Plane::Whole,
        at: ReadAt::BASE,
    }
}

/// The size the sparse heap takes, in bytes: one 16-KiB page, which is the descriptor's own page
/// size. A sparse heap holds no memory of its own until a mapping takes a tile, and nothing here
/// asks for one.
const HEAP_BYTES: u64 = 16 * 1024;

/// How many runs record mode takes a case of this module over, and why eight rather than three:
/// the barrier case is the one place in this packet where a wrong answer could be *intermittent*
/// rather than constant, because an unordered pair of draws may happen to come out ordered. Eight
/// runs that fold into one outcome are the evidence the recording then carries; three are too few
/// to call it.
const RUNS: u32 = 8;

/// The shader reference both texts of the one fixture give.
fn shader_refs() -> Result<Vec<ShaderRef>, String> {
    let fixture = Fixture::new(FIXTURE);
    [RUNTIME_TEXT, METAL_TEXT]
        .into_iter()
        .map(|extension| {
            Ok(ShaderRef {
                fixture: format!("{FIXTURE}{extension}"),
                entry: ENTRIES.to_owned(),
                source_sha256: crate::sha256_hex(fixture.text(extension)?.as_bytes()),
            })
        })
        .collect()
}

/// The two inputs the fixture's own texts are.
fn fixture_inputs() -> Result<Vec<Input>, String> {
    let fixture = Fixture::new(FIXTURE);
    Ok(vec![
        Input {
            name: "air".to_owned(),
            digest: Digest::sha256(fixture.text(RUNTIME_TEXT)?.as_bytes()),
        },
        Input {
            name: "air-typed".to_owned(),
            digest: Digest::sha256(fixture.text(METAL_TEXT)?.as_bytes()),
        },
    ])
}

/// The library of the module's one fixture.
fn library(side: &mut dyn Side, sites: &mut Sites) -> Result<Handle, Abort> {
    let loaded = Fixture::new(FIXTURE).loaded().map_err(|why| Abort {
        at: "library".to_owned(),
        failure: Failure::named("FixtureNotRead", why),
    })?;
    made(sites, "library", side.library(&loaded.source()))
}

/// Run `work` as one command buffer, emit what each piece answered, and read the target's texels.
fn drawn(
    side: &mut dyn Side,
    sites: &mut Sites,
    work: &[RenderWork],
    read_back: Handle,
    extent: u32,
) -> Result<(), Abort> {
    let outcome = call(sites, "render", side.render(work), |outcome| {
        outcome.status.clone()
    })?;
    let names = ["pass", "read"];
    let mut pieces = Node::branch("pieces", Vec::new());
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
    if let Some((at, failure)) = names
        .iter()
        .enumerate()
        .find_map(|(at, what)| outcome.refused.get(at)?.as_ref().map(|f| (*what, f)))
    {
        return Err(Abort {
            at: format!("pieces/{at}"),
            failure: failure.clone(),
        });
    }
    let bytes = call(
        sites,
        "read-back",
        side.read_buffer(read_back, 0, plane(extent) as usize),
        |_| Value::Absent,
    )?;
    sites.push(
        Node::branch("drawn", Vec::new())
            .with(Node::leaf("texels", Observed::bytes(bytes, layout(extent)))),
    );
    Ok(())
}

/// The colour state both pipelines of either case take.
const PLAIN: [Option<ColorState>; 1] = [Some(ColorState::new(FORMAT))];

/// Every case of the family, in the order the report lists them.
pub fn all() -> Vec<Box<dyn Case>> {
    vec![Box::new(RenderBarriers), Box::new(HeapResidency)]
}

// ------------------------------------------------------------------------------------------------
// render-state-render-barriers

/// The stages the barriers name on both sides: a fragment function's writes read by a fragment
/// function.
const FRAGMENT: RenderStages = RenderStages::FRAGMENT;

/// Both render memory barriers between two draws of one pass, with the four `useResource` forms
/// around them.
pub struct RenderBarriers;

impl RenderBarriers {
    /// The two pipelines: the one whose fragment function reads a texture, and the one whose
    /// writes one.
    fn pipelines(colors: &'static [Option<ColorState>]) -> [RenderPipelineSpec<'static>; 2] {
        [
            RenderPipelineSpec::new(RECT, READ, colors),
            RenderPipelineSpec::new(RECT, WRITE, colors),
        ]
    }

    /// The pass's steps: the reading draw over the target's left half, both barriers, and the
    /// writing draw over its right half.
    ///
    /// The two draws share no region of the target and no texture, so neither depends on the
    /// other and the pass's answer is what each draw's own bindings say. The barriers are between
    /// them because that is where a program puts them, and what the case holds them to is that
    /// Metal answered them and that neither draw moved.
    fn steps(
        reading: Handle,
        writing: Handle,
        rects: Handle,
        source: Handle,
        scratch: Handle,
    ) -> Vec<RenderStep> {
        let read_texture = StepResource::Texture(source);
        let written_texture = StepResource::Texture(scratch);
        let buffer = StepResource::Buffer(rects);
        vec![
            RenderStep::Pipeline(reading),
            RenderStep::VertexBuffer {
                buffer: rects,
                offset: 0,
                index: 0,
            },
            RenderStep::FragmentTexture {
                texture: source,
                index: 0,
            },
            // The four residency forms, over the texture one draw's fragment function reads, the
            // texture the other's writes, and the buffer the vertex function reads. Each is a call
            // Metal answers; none of them is what makes any of the three reachable, because all
            // three are bound at a slot.
            RenderStep::UseResourceInStages {
                resource: read_texture,
                usage: ResourceUsage::READ,
                stages: FRAGMENT,
            },
            RenderStep::UseResource {
                resource: written_texture,
                usage: ResourceUsage::WRITE,
            },
            RenderStep::UseResourcesInStages {
                resources: vec![read_texture, written_texture, buffer],
                usage: ResourceUsage::READ.union(ResourceUsage::WRITE),
                stages: RenderStages::VERTEX.union(RenderStages::FRAGMENT),
            },
            RenderStep::UseResources {
                resources: vec![read_texture, buffer],
                usage: ResourceUsage::READ,
            },
            RenderStep::Scissor(LEFT),
            RECT_DRAW,
            // The barriers: the scope form over every texture, and the resource form over the two
            // this pass names.
            RenderStep::BarrierScope {
                scope: BarrierScope::TEXTURES,
                after: FRAGMENT,
                before: FRAGMENT,
            },
            RenderStep::BarrierResources {
                resources: vec![read_texture, written_texture],
                after: FRAGMENT,
                before: FRAGMENT,
            },
            RenderStep::Pipeline(writing),
            RenderStep::FragmentTexture {
                texture: scratch,
                index: 0,
            },
            RenderStep::Scissor(RIGHT),
            RECT_DRAW,
        ]
    }
}

impl Case for RenderBarriers {
    fn name(&self) -> &'static str {
        "render-state-render-barriers"
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn record_runs(&self) -> u32 {
        RUNS
    }

    fn allowances(&self) -> Vec<Allowance> {
        // No value is loosened. A `RGBA32Float` texel is the `float4` the fragment function
        // returned, the fixture's arithmetic is a multiplication by an eighth of an integer, and a
        // texel read at integer coordinates takes no device's float bounds, so every texel here is
        // an exact answer or a bug.
        //
        // One notice is declared, and it is what the barrier calls themselves cost. **A Metal
        // memory barrier between draws of one pass orders the draws after it behind the draws
        // before it, which Vulkan gives across render pass instances**, so the runtime ends the
        // instance at the barrier and begins another (`metal2vulkan_runtime`'s `fence.rs`, counted
        // as that crate's `work_counts::Split::Barrier`). Metal does not split its pass, so the
        // runtime announces the difference in work.
        //
        // Measured on an Apple M3 Max (macOS 27.0.1, build 26A434), under the exact policy and
        // under the runtime's own defaults alike: our leg announces
        // `metal2vulkan-degrade-render-pass-split` once, its notice reading "memory barriers
        // between its draws end 1 of its Vulkan render pass instances", while every texel of both
        // halves of the target and every other field of the transcript compared exactly.
        //
        // One more is declared, and it is the venue's rather than this case's: the buffer the
        // pass reads its target back through is `Shared`, which a device with no unified memory
        // keeps in system memory because that is the only memory both sides can reach.
        [
            Degradation::RenderPassSplit,
            Degradation::SharedInSystemMemory,
        ]
        .iter()
        .map(|kind| crate::replay::notice_allowance(kind.code()))
        .collect()
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        shader_refs()
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let mut inputs = fixture_inputs()?;
        for (at, pipeline) in Self::pipelines(&PLAIN).iter().enumerate() {
            inputs.push(Input::bytes(
                format!("render-pipeline:{at}"),
                &pipeline.words(),
            ));
        }
        inputs.push(Input::bytes("target", &target_spec(SIDE).words()));
        inputs.push(Input::bytes("texture:source", &source_texels()));
        inputs.push(Input::bytes(
            "texture:scratch",
            &texture_words(scratch_spec()),
        ));
        inputs.push(Input::bytes("buffer:rects", &rect_bytes()));
        inputs.push(Input::bytes("clear", &CLEAR.to_le_bytes()));
        let halves: Vec<u8> = [LEFT, RIGHT].iter().flat_map(ScissorBox::words).collect();
        inputs.push(Input::bytes("halves", &halves));
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let library = library(side, sites)?;
        let mut states = Vec::new();
        for (at, pipeline) in Self::pipelines(&PLAIN).iter().enumerate() {
            states.push(made(
                sites,
                &format!("render-pipeline:{at}"),
                side.render_pipeline(library, pipeline),
            )?);
        }
        let target = made(sites, "target", side.target(&target_spec(SIDE)))?;
        let source = made(
            sites,
            "texture:source",
            side.texture_2d_in_role(&source_spec(), &source_texels()),
        )?;
        let scratch = made(
            sites,
            "texture:scratch",
            side.texture_2d_in_role(&scratch_spec(), &[]),
        )?;
        let rects = made(sites, "buffer:rects", side.buffer(&rect_bytes()))?;
        let read_back = made(
            sites,
            "buffer:read-back",
            side.zeroed_buffer(plane(SIDE) as usize),
        )?;
        let work = vec![
            RenderWork::Pass(
                RenderPass::clearing(vec![PassColor::cleared(target, [f64::from(CLEAR); 4])])
                    .drawing(Self::steps(states[0], states[1], rects, source, scratch)),
            ),
            read_target(target, read_back, SIDE),
        ];
        drawn(side, sites, &work, read_back, SIDE)
    }
}

// ------------------------------------------------------------------------------------------------
// render-state-heap-residency

/// Whether the four `useHeap` forms are answered on a heap that holds nothing, and that the pass's
/// own draw is untouched by them.
pub struct HeapResidency;

impl HeapResidency {
    /// The one pipeline: the rectangle and the solid colour.
    fn pipeline(colors: &'static [Option<ColorState>]) -> RenderPipelineSpec<'static> {
        RenderPipelineSpec::new(RECT, SOLID, colors)
    }

    /// The pass's steps.
    fn steps(state: Handle, rects: Handle, color: Handle, heap: Handle) -> Vec<RenderStep> {
        vec![
            RenderStep::Pipeline(state),
            RenderStep::VertexBuffer {
                buffer: rects,
                offset: 0,
                index: 0,
            },
            RenderStep::FragmentBuffer {
                buffer: color,
                offset: 0,
                index: 0,
            },
            RenderStep::UseHeap(heap),
            RenderStep::UseHeapInStages {
                heap,
                stages: FRAGMENT,
            },
            RenderStep::UseHeaps(vec![heap]),
            RenderStep::UseHeapsInStages {
                heaps: vec![heap],
                stages: RenderStages::VERTEX.union(RenderStages::FRAGMENT),
            },
            RECT_DRAW,
        ]
    }
}

impl Case for HeapResidency {
    fn name(&self) -> &'static str {
        "render-state-heap-residency"
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn record_runs(&self) -> u32 {
        RUNS
    }

    fn allowances(&self) -> Vec<Allowance> {
        // Nothing loosened, for the reason the barrier case states, and no pass-split notice:
        // this case encodes no barrier. What is declared is the one notice the venue's own memory
        // model provokes for the `Shared` read-back buffer.
        vec![crate::replay::notice_allowance(
            Degradation::SharedInSystemMemory.code(),
        )]
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        shader_refs()
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let mut inputs = fixture_inputs()?;
        inputs.push(Input::bytes(
            "render-pipeline",
            &Self::pipeline(&PLAIN).words(),
        ));
        inputs.push(Input::bytes("target", &target_spec(SMALL).words()));
        inputs.push(Input::bytes("buffer:rects", &rect_bytes()));
        inputs.push(Input::bytes("buffer:color", &color_bytes()));
        inputs.push(Input::bytes(
            "heap",
            &SparseHeapSpec::new(HEAP_BYTES).words(),
        ));
        inputs.push(Input::bytes("clear", &CLEAR.to_le_bytes()));
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let library = library(side, sites)?;
        let state = made(
            sites,
            "render-pipeline",
            side.render_pipeline(library, &Self::pipeline(&PLAIN)),
        )?;
        let target = made(sites, "target", side.target(&target_spec(SMALL)))?;
        let rects = made(sites, "buffer:rects", side.buffer(&rect_bytes()))?;
        let color = made(sites, "buffer:color", side.buffer(&color_bytes()))?;
        let heap = made(
            sites,
            "heap",
            side.sparse_heap(&SparseHeapSpec::new(HEAP_BYTES)),
        )?;
        let read_back = made(
            sites,
            "buffer:read-back",
            side.zeroed_buffer(plane(SMALL) as usize),
        )?;
        let work = vec![
            RenderWork::Pass(
                RenderPass::clearing(vec![PassColor::cleared(target, [f64::from(CLEAR); 4])])
                    .drawing(Self::steps(state, rects, color, heap)),
            ),
            read_target(target, read_back, SMALL),
        ];
        drawn(side, sites, &work, read_back, SMALL)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::allowance::Justification;
    use crate::transcript::Transcript;
    use std::collections::BTreeSet;

    /// The one outcome of `case`'s recorded Metal answer, or `None` where nothing recorded it.
    ///
    /// The assertion matters more here than anywhere else in the packet: a pair of draws a barrier
    /// failed to order may come out ordered by luck, so a case whose eight runs did **not** fold
    /// into one outcome is reporting exactly the non-determinism the barrier is there to remove.
    fn only_outcome(case: &str) -> Option<Transcript> {
        let recording = crate::recording::read(case).ok()?;
        let answer = recording.answers.first()?;
        assert_eq!(
            answer.outcomes.len(),
            1,
            "{case}: {} outcomes over {} runs, so the pass's answer moved between runs",
            answer.outcomes.len(),
            answer.runs
        );
        Some(answer.outcomes[0].transcript.clone())
    }

    /// The target's recorded texels of `case`.
    fn recorded(case: &str, side: u32) -> Option<Vec<u8>> {
        let transcript = only_outcome(case)?;
        let node = transcript
            .root
            .child("drawn")
            .and_then(|node| node.child("texels"))
            .expect("no drawn/texels site");
        let Value::Bytes(bytes) = &node.observed.value else {
            panic!("drawn/texels holds no bytes");
        };
        assert_eq!(bytes.len(), plane(side) as usize);
        Some(bytes.clone())
    }

    /// The `float4` at texel `(x, y)`.
    fn value(bytes: &[u8], x: u32, y: u32, side: u32) -> [f32; 4] {
        let at = ((y * side + x) * TEXEL) as usize;
        let mut channels = [0f32; 4];
        for (channel, held) in channels.iter_mut().enumerate() {
            let word = at + 4 * channel;
            *held = f32::from_le_bytes([
                bytes[word],
                bytes[word + 1],
                bytes[word + 2],
                bytes[word + 3],
            ]);
        }
        channels
    }

    /// What `tob_write` returns into the colour target: opaque black, which is the writing draw's
    /// own half of the answer.
    const OPAQUE_BLACK: [f32; 4] = [0.0, 0.0, 0.0, 1.0];

    /// What the barrier case says texel `(x, y)` of its target holds: the texture the reading
    /// draw read in the target's left half, and the opaque black the writing draw's fragment
    /// function returns in its right half.
    fn barriers_expected(x: u32, y: u32) -> [f32; 4] {
        match x < SIDE / 2 {
            true => source_texel(x, y),
            false => OPAQUE_BLACK,
        }
    }

    /// **Metal's recorded target is what each of the two draws stored in its own half**, so both
    /// barrier calls were answered between them and neither draw moved.
    #[test]
    fn metal_drew_both_halves_of_the_barrier_case_as_their_bindings_say() {
        let Some(bytes) = recorded(RenderBarriers.name(), SIDE) else {
            println!("render-state-render-barriers: no recording in the tree yet");
            return;
        };
        let mut differ = Vec::new();
        for y in 0..SIDE {
            for x in 0..SIDE {
                let (got, want) = (value(&bytes, x, y, SIDE), barriers_expected(x, y));
                if got != want {
                    differ.push(format!("({x}, {y}): Metal {got:?}, the map says {want:?}"));
                }
            }
        }
        for line in differ.iter().take(20) {
            println!("BARRIER-DIFFERS: {line}");
        }
        assert!(
            differ.is_empty(),
            "{} of {} texels are not what the two draws' own bindings say",
            differ.len(),
            SIDE * SIDE
        );
    }

    /// **The barrier case's two halves are values nothing else here leaves**, so it cannot pass on
    /// a target one of its draws never reached.
    ///
    /// This is the check the round's own vacuity incident earns: a draw that wrote nothing, and a
    /// pair of halves whose values cannot be told apart, both read as a pass. Here the left half
    /// is a texel no draw can compute, each one its own, and the right half is a constant neither
    /// the clear nor any of those texels.
    #[test]
    fn the_two_halves_are_values_nothing_else_leaves() {
        let read: BTreeSet<[u32; 4]> = (0..SIDE)
            .flat_map(|y| (0..SIDE).map(move |x| source_texel(x, y).map(f32::to_bits)))
            .collect();
        assert_eq!(
            read.len(),
            (SIDE * SIDE) as usize,
            "two texels answer alike"
        );
        for other in [[CLEAR; 4], OPAQUE_BLACK, COLOR, [0.0; 4]] {
            assert!(!read.contains(&other.map(f32::to_bits)), "{other:?}");
        }
        assert_ne!(OPAQUE_BLACK, [CLEAR; 4]);
        assert_ne!(OPAQUE_BLACK, COLOR);
        // The two halves are the whole target between them, so every texel is one draw's answer
        // and the clear appears nowhere -- which is what says both draws ran.
        assert_eq!(LEFT.width + RIGHT.width, u64::from(SIDE));
        assert_eq!(LEFT.x, 0);
        assert_eq!(RIGHT.x, LEFT.width);
        for rect in [LEFT, RIGHT] {
            assert_eq!(rect.y, 0);
            assert_eq!(rect.height, u64::from(SIDE));
            assert!(!rect.past([SIDE; 2]));
        }
        // The check sees a rectangle that reaches past the target, so a green run means these are
        // inside rather than that it cannot fire.
        assert!(ScissorBox::new(0, 0, u64::from(SIDE) + 1, 1).past([SIDE; 2]));
        // One texture is read-only and the other write-only, so neither draw can reach the
        // other's, which is what makes the two halves independent.
        assert_eq!(source_spec().role, TextureRole::Read);
        assert_eq!(scratch_spec().role, TextureRole::Write);
        assert_eq!(layout(SIDE).extent(), plane(SIDE) as usize);
        // The two pipelines are two pipelines.
        let [reading, writing] = RenderBarriers::pipelines(&PLAIN);
        assert_ne!(reading.words(), writing.words());
    }

    /// **The heap case's target is the colour its binding holds**, in every texel: the four
    /// `useHeap` forms changed nothing the pass observes.
    #[test]
    fn metal_drew_the_heap_case_as_its_binding_says() {
        let Some(bytes) = recorded(HeapResidency.name(), SMALL) else {
            println!("render-state-heap-residency: no recording in the tree yet");
            return;
        };
        for y in 0..SMALL {
            for x in 0..SMALL {
                assert_eq!(
                    value(&bytes, x, y, SMALL),
                    COLOR,
                    "({x}, {y}) is not the colour the fragment binding holds"
                );
            }
        }
        assert_ne!(COLOR, [CLEAR; 4]);
    }

    /// **Every case of the family names itself once**, under the settings the module states.
    #[test]
    fn the_family_names_two_cases_apart() {
        let names: Vec<&str> = all().iter().map(|case| case.name()).collect();
        let distinct: BTreeSet<&str> = names.iter().copied().collect();
        assert_eq!(distinct.len(), names.len(), "two cases share a name");
        assert_eq!(names.len(), 2);
        for case in all() {
            assert_eq!(case.record_runs(), RUNS);
            assert!(matches!(case.metal_source(), MetalSource::Run));
            // Both cases declare the notice the venue's own memory model provokes for the
            // `Shared` read-back buffer, and the barrier case declares the pass split besides. No
            // value of either is loosened, so every declaration sits at the notices site.
            let declared = case.allowances();
            for allowance in &declared {
                assert!(allowance.path.starts_with("case/notices"), "{allowance:?}");
            }
            let codes: Vec<&Justification> =
                declared.iter().map(|held| &held.justified_by).collect();
            let shared =
                Justification::Degradation(Degradation::SharedInSystemMemory.code().to_owned());
            let split = Justification::Degradation(Degradation::RenderPassSplit.code().to_owned());
            match case.name() {
                "render-state-render-barriers" => assert_eq!(codes, vec![&split, &shared]),
                _ => assert_eq!(codes, vec![&shared], "{}", case.name()),
            }
        }
        // The barriers name the stage whose writes the next draw reads, on both sides of each.
        assert_eq!(FRAGMENT, RenderStages::FRAGMENT);
        // The heap is one page of the descriptor's own page size, which holds no memory until a
        // mapping takes a tile.
        assert_eq!(HEAP_BYTES, 16 * 1024);
    }
}
