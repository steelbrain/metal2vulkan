//! The compute encoder's commands no other case encodes: the inline bytes, the plural setters, the
//! LOD-clamped sampler bindings, the dynamic threadgroup allocation, the residency declarations,
//! the two barriers, the threadgroup-grid dispatch, the indirect one, and a concurrent encoder.
//!
//! `MTLComputeCommandEncoder` has thirty-six implemented items and the suite reached four of them —
//! the pipeline, a buffer, a texture and `dispatchThreads:` — because every case before this one
//! ran a **fixed** pass ([`Side::dispatch`], [`Side::dispatch_threads`]). The eight cases here
//! state their commands instead ([`crate::compute_pass`]) and drive seventeen more.
//!
//! # How each case is read
//!
//! Every case writes its answers into **one** output buffer, in slots, and compares every byte of
//! it — the guard bytes past the last slot included. Most of them run the same dispatch twice
//! or four times, through a different set of binding calls each time, into a slot each:
//!
//! | case | what it holds |
//! |---|---|
//! | [`threadgroup_memory_and_grid`] | `setThreadgroupMemoryLength:atIndex:` at two lengths, `dispatchThreadgroups:threadsPerThreadgroup:`, and `setBufferOffset:atIndex:` moving the second dispatch's output |
//! | [`plural_buffer_binding`] | `setBuffers:offsets:withRange:` against `setBuffer:offset:atIndex:` |
//! | [`resource_residency`] | `useResource:usage:` and `useResources:count:usage:` |
//! | [`indirect_threadgroup_grid`] | `dispatchThreadgroupsWithIndirectBuffer:indirectBufferOffset:threadsPerThreadgroup:`, its grid read from a buffer at a nonzero offset |
//! | [`scope_barrier`] | a **concurrent** encoder, two chained dispatches and `memoryBarrierWithScope:` between them |
//! | [`resource_barrier`] | the same chain under `memoryBarrierWithResources:count:` |
//! | [`plural_texture_and_sampler_binding`] | `setTextures:withRange:`, `setSamplerStates:withRange:` and both LOD-clamped setters against the singular calls |
//! | [`inline_bytes`] | `setBytes:length:atIndex:` against a buffer bound at the same index |
//!
//! Two of these cases are **not** an equivalence of two binding calls, and they are the two whose
//! answer a wrong implementation cannot reproduce by accident. [`scope_barrier`] and
//! [`resource_barrier`] chain `a4_index_word` through itself on a **concurrent** encoder, where
//! Metal does not order the dispatches and the barrier is the only thing that makes the second
//! dispatch read what the first wrote. The case states the middle buffer's own words beside the
//! output's, so a run in which the barrier did nothing reads the buffer the case filled rather than
//! the first dispatch's answer.
//!
//! Where a case *is* an equivalence, the limit is written here rather than left implied: it holds
//! Metal to its own two spellings of one binding, and on our leg a difference between the slots
//! would be a bug of ours. What such a case cannot do is fail because a *value* is wrong — both
//! slots would be wrong together — and that is what the recording is for: the slots' bytes are
//! Metal's, and a sample or a sum that moved moves the case.
//!
//! # Safety
//!
//! Every dispatch is one threadgroup of at most 32 threads, which is a simdgroup on every Apple GPU
//! and inside the smallest `maxTotalThreadsPerThreadgroup` any venue reports; no grid is derived
//! from a device limit, and the indirect grid is three words the host wrote
//! (`every_dispatch_is_one_small_threadgroup`). The fixtures are loop-free and read only inside
//! their bindings: `region_k` writes `out[t]` and reads word 0 of its threadgroup allocation, which
//! every case allocates at least a word of (`every_threadgroup_allocation_holds_its_word`), and
//! `a4_index_word` reads `in[3t + 1]`, which every case's buffer holds
//! (`every_chained_dispatch_reads_inside_its_buffer`). Nothing here can hang, fault or trip a
//! watchdog.
//!
//! [`Side::dispatch`]: crate::case::Side::dispatch
//! [`Side::dispatch_threads`]: crate::case::Side::dispatch_threads

use crate::allowance::Allowance;
use crate::case::{made, Abort, Case, Failure, Input, MetalSource, Side, TextureRole, TextureSpec};
use crate::compute_pass::{
    ComputePass, ComputeStep, Dispatching, PassResource, SCOPE_BUFFERS, USAGE_READ, USAGE_WRITE,
};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::transcript::{ByteLayout, Digest, Element, Node, Observed, Sites};
use metal2vulkan::metal::{PixelFormat, SamplerMinMagFilter};

/// The byte every output buffer is made holding, so the bytes no dispatch wrote are a value the
/// case put there rather than whatever the allocator left.
const GUARD: u8 = 0x7e;

/// Threads in each dispatch's one threadgroup: one simdgroup on every Apple GPU, and inside the
/// smallest threadgroup limit any venue reports.
const THREADS: u32 = 32;

/// Bytes `region_k` writes for a threadgroup of [`THREADS`]: one 32-bit word a thread.
const SLOT: usize = THREADS as usize * 4;

/// Bytes of guard past the last slot of an output buffer, so a dispatch that wrote more words than
/// the case asked for shows.
const TAIL: usize = 64;

/// The fixture whose `region_k` writes one word a thread through a simdgroup move and the first
/// word of its `[[threadgroup(0)]]` allocation.
const REGION_FIXTURE: &str = "simd_uniform";
const REGION_ENTRY: &str = "region_k";

/// The fixture whose `a4_index_word` is `out[t] = in[3t + 1]`: the elementwise read-and-write a
/// chained pair of dispatches needs.
const INDEX_FIXTURE: &str = "a4_index_word";

/// The fixture whose `xr_sample_runtime` samples the texture at slot 0 through the sampler at slot
/// 0 at the coordinates a buffer holds.
const SAMPLE_FIXTURE: &str = "xr_samples";
const SAMPLE_ENTRY: &str = "xr_sample_runtime";

/// The two dynamic threadgroup lengths [`threadgroup_memory_and_grid`] allocates: a word and a
/// kibibyte and a bit. Both hold `region_k`'s one word, so what the case asks is whether Metal's
/// answer depends on the allocation's size, and both are far inside any device's threadgroup
/// memory.
const LENGTHS: [u32; 2] = [4, 1040];

/// Where the indirect grid's three words lie in their buffer: a nonzero, four-byte-aligned offset,
/// so a side that read the buffer from its start would dispatch a grid of guard bytes rather than
/// the case's.
const GRID_AT: u64 = 16;

/// The grid [`indirect_threadgroup_grid`] dispatches: one threadgroup.
const GRID: [u32; 3] = [1, 1, 1];

/// The threads the chained pair of dispatches runs: the writer's threadgroup, then the reader's.
///
/// The reader reads `mid[3t + 1]` for `t` under [`CHAIN_READ`], so the writer has to have written
/// at least `3 * (CHAIN_READ - 1) + 2` words of `mid`, which [`CHAIN_WRITE`] does.
const CHAIN_WRITE: u32 = 24;
const CHAIN_READ: u32 = 8;

/// Words the chain's input buffer holds: enough for the writer's own `in[3t + 1]`.
const CHAIN_IN: usize = 80;

/// The texels of the 4 by 4 `RGBA8Unorm` texture the sampling cases read, and the coordinates they
/// sample at.
const TEXTURE_SIDE: u32 = 4;

/// The coordinates [`plural_texture_and_sampler_binding`] samples at, as normalized `float2`s: one
/// per thread of its dispatch, inside the texture, none on a texel boundary.
const COORDS: [[f32; 2]; 4] = [[0.1, 0.2], [0.35, 0.6], [0.6, 0.35], [0.9, 0.8]];

/// Bytes `xr_sample_runtime` writes for [`COORDS`]: a `float4` each.
const SAMPLE_SLOT: usize = COORDS.len() * 16;

/// The texels of the sampled texture: byte `i` holding `i * 7 + 3`, so no two channels of a texel
/// are equal and a sample that read the wrong texel shows in every channel.
fn texels() -> Vec<u8> {
    (0..(TEXTURE_SIDE * TEXTURE_SIDE * 4) as usize)
        .map(|index| (index * 7 + 3) as u8)
        .collect()
}

/// The bytes [`COORDS`] is handed to a kernel as.
fn coord_bytes() -> Vec<u8> {
    COORDS
        .iter()
        .flat_map(|axes| axes.iter().flat_map(|axis| axis.to_le_bytes()))
        .collect()
}

/// `words` 32-bit words, word `i` holding `i * 2654435761 mod 2^32`: a sequence no index pattern
/// reproduces, so a dispatch that read the wrong word shows.
fn words(count: usize) -> Vec<u8> {
    (0..count)
        .flat_map(|index| (index as u32).wrapping_mul(2_654_435_761).to_le_bytes())
        .collect()
}

/// An output buffer of `slots` slots of `each` bytes, with [`TAIL`] of guard past the last.
fn output(slots: usize, each: usize) -> Vec<u8> {
    vec![GUARD; slots * each + TAIL]
}

/// A buffer's bytes as a site: every byte compared, guard included, as `Element::U32` so a diff
/// names the word that moved.
fn words_site(name: &str, bytes: Vec<u8>) -> Node {
    let length = bytes.len() as u64;
    Node::leaf(
        name,
        Observed::bytes(bytes, ByteLayout::buffer(0, length, Element::U32)),
    )
}

/// What a case asks of its side, beyond the pass itself.
///
/// A case states it rather than the module's `run` branching on the case's name: the shape of what
/// a compute pass needs is "these resources, then these commands", and each row says which.
struct Row {
    /// The case's name: its recording's file name, and its row in the report.
    case: &'static str,
    /// The fixture the pass's pipeline runs, and its entry point.
    fixture: Fixture,
    /// The pass this row encodes, built from the handles the row's resources came back as.
    pass: fn(&Made) -> (Dispatching, Vec<ComputeStep>),
    /// The resources the row makes, in the order [`Made`] holds them.
    resources: Resources,
    /// The site the row emits its first read-back at. A chained row emits `middle` beside it.
    site: &'static str,
}

impl Row {
    /// The sites the row emits from the buffers it read back.
    ///
    /// One shape for every row — the whole buffer as words — so the slots and the guard tail
    /// compare alike wherever a row puts them, and a row cannot pick a layout that stops short of
    /// its own guard.
    fn sites(&self, read: &[Vec<u8>]) -> Vec<Node> {
        let mut out = vec![words_site(
            self.site,
            read.first().cloned().unwrap_or_default(),
        )];
        if matches!(self.resources, Resources::Chain) {
            out.push(words_site(
                "middle",
                read.get(1).cloned().unwrap_or_default(),
            ));
        }
        out
    }

    /// The slots the row's site holds and the bytes of each, where the row runs one dispatch a
    /// binding call into a slot apiece. `None` for the rows that are not an equivalence of two
    /// spellings: the chained pair and the indirect grid.
    ///
    /// Read by the guard that holds Metal's slots to each other, which is the only thing that
    /// needs to know a row's slot shape: the pass itself states each offset outright.
    #[cfg(test)]
    fn slots(&self) -> Option<(usize, usize)> {
        match self.resources {
            Resources::Region { slots } if slots > 1 => Some((slots, SLOT)),
            Resources::Sampled { slots } if slots > 1 => Some((slots, SAMPLE_SLOT)),
            _ => None,
        }
    }
}

/// Which resources a row makes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Resources {
    /// One output buffer of `slots` slots of [`SLOT`] bytes: what a `region_k` row needs.
    Region { slots: usize },
    /// An output buffer of `slots` slots, and a buffer holding the indirect grid at [`GRID_AT`].
    RegionWithGrid { slots: usize },
    /// The chained pair's three buffers: the input words, the middle buffer and the output.
    Chain,
    /// The sampling row's texture, its sampler, its coordinates and an output of `slots` slots.
    Sampled { slots: usize },
}

/// The resources a row made, by the role the row's pass names them in.
struct Made {
    /// The pipeline the pass sets.
    pipeline: crate::case::Handle,
    /// The buffers the row made, in the order [`Resources`] names them.
    buffers: Vec<crate::case::Handle>,
    /// The textures the row made.
    textures: Vec<crate::case::Handle>,
    /// The sampler states the row made.
    samplers: Vec<crate::case::Handle>,
}

impl Resources {
    /// The bytes of each buffer this row makes, in order.
    fn buffers(self) -> Vec<Vec<u8>> {
        match self {
            Resources::Region { slots } => vec![output(slots, SLOT)],
            Resources::RegionWithGrid { slots } => {
                let mut grid = vec![GUARD; GRID_AT as usize];
                grid.extend(GRID.iter().flat_map(|axis| axis.to_le_bytes()));
                grid.extend([GUARD; 16]);
                vec![output(slots, SLOT), grid]
            }
            Resources::Chain => vec![
                words(CHAIN_IN),
                // The middle buffer is **made holding the guard**, not zeroed: the chain's answer
                // is that the reader read the writer's words, and a middle buffer of zeros could
                // not tell that from a barrier that did nothing.
                vec![GUARD; CHAIN_WRITE as usize * 4],
                output(1, CHAIN_READ as usize * 4),
            ],
            Resources::Sampled { slots } => {
                vec![coord_bytes(), output(slots, SAMPLE_SLOT)]
            }
        }
    }

    /// Which of this row's buffers the case reads back, and how many bytes of each.
    fn read_back(self) -> Vec<(usize, usize)> {
        match self {
            Resources::Region { slots } => vec![(0, output(slots, SLOT).len())],
            Resources::RegionWithGrid { slots } => vec![(0, output(slots, SLOT).len())],
            // Both the middle buffer and the output, because the middle buffer's words are what
            // say the first dispatch ran at all.
            Resources::Chain => vec![
                (2, output(1, CHAIN_READ as usize * 4).len()),
                (1, CHAIN_WRITE as usize * 4),
            ],
            Resources::Sampled { slots } => vec![(1, output(slots, SAMPLE_SLOT).len())],
        }
    }
}

/// A case of the family: one row.
pub struct ComputeEncoderCommands {
    row: &'static Row,
}

impl Case for ComputeEncoderCommands {
    fn name(&self) -> &'static str {
        self.row.case
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // No value is loosened. Every byte of every slot compares exactly, and no venue has been
        // measured announcing a notice of this family's own; a venue that does reads as
        // `UndeclaredNotice`, naming the code and the case, which is the loud form.
        //
        // Every row reads its answer back through a `Shared` buffer, so on a device without
        // unified memory the runtime announces `shared-in-system-memory`
        // (`crate::replay::shared_memory_notice` holds the whole reason). It declares the
        // notices site alone, so every value of this family still compares against Metal's
        // exactly.
        //
        // Measured on the RTX 5080 Laptop (615.71.09): it fired for
        // all 8 rows of the family, and each was divergent on that field and
        // nothing else (replay-nvidia.log). On MoltenVK, where
        // the chip's GPU shares the CPU's memory, it cannot fire at all, which is why no
        // recording of this suite declares it.
        vec![crate::replay::shared_memory_notice()]
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let fixture = self.row.fixture;
        let mut shaders = Vec::new();
        for extension in [RUNTIME_TEXT, METAL_TEXT] {
            shaders.push(ShaderRef {
                fixture: format!("{}{extension}", fixture.name),
                entry: fixture.entry.to_owned(),
                source_sha256: crate::sha256_hex(fixture.text(extension)?.as_bytes()),
            });
        }
        Ok(shaders)
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let fixture = self.row.fixture;
        let mut inputs = vec![
            Input {
                name: "air".to_owned(),
                digest: Digest::sha256(fixture.text(RUNTIME_TEXT)?.as_bytes()),
            },
            Input {
                name: "air-typed".to_owned(),
                digest: Digest::sha256(fixture.text(METAL_TEXT)?.as_bytes()),
            },
        ];
        for (at, bytes) in self.row.resources.buffers().into_iter().enumerate() {
            inputs.push(Input::bytes(format!("buffer:{at}"), &bytes));
        }
        if matches!(self.row.resources, Resources::Sampled { .. }) {
            inputs.push(Input::bytes("texels", &texels()));
            inputs.push(Input::bytes("sampler", &sampler_spec().words()));
        }
        // The pass itself: every call, in order, with its own numbers. A pass whose commands or
        // whose order moved is a change to the case's inputs, so a recording cannot stay the
        // answer to a program that is no longer this one.
        let made = Made {
            pipeline: crate::case::Handle(0),
            buffers: Vec::new(),
            textures: Vec::new(),
            samplers: Vec::new(),
        };
        let (dispatching, steps) = (self.row.pass)(&made);
        inputs.push(Input::bytes(
            "pass",
            &ComputePass {
                dispatching,
                steps: &steps,
            }
            .words(),
        ));
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let row = self.row;
        let loaded = row.fixture.loaded().map_err(|why| Abort {
            at: "library".to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        })?;
        let library = made(sites, "library", side.library(&loaded.source()))?;
        let pipeline = made(
            sites,
            "pipeline",
            side.compute_pipeline(library, row.fixture.entry),
        )?;
        let mut buffers = Vec::new();
        for (at, bytes) in row.resources.buffers().into_iter().enumerate() {
            buffers.push(made(sites, &format!("buffer/{at}"), side.buffer(&bytes))?);
        }
        let mut textures = Vec::new();
        let mut samplers = Vec::new();
        if matches!(row.resources, Resources::Sampled { .. }) {
            let spec = TextureSpec::new(
                PixelFormat::RGBA8Unorm,
                [TEXTURE_SIDE, TEXTURE_SIDE],
                TEXTURE_SIDE * 4,
                TextureRole::Read,
            );
            textures.push(made(
                sites,
                "texture",
                side.texture_2d_in_role(&spec, &texels()),
            )?);
            samplers.push(made(sites, "sampler", side.sampler(&sampler_spec()))?);
        }
        let held = Made {
            pipeline,
            buffers,
            textures,
            samplers,
        };
        let (dispatching, steps) = (row.pass)(&held);
        let outcome = made(
            sites,
            "pass",
            side.compute_pass(&ComputePass {
                dispatching,
                steps: &steps,
            }),
        )?;
        sites.push(outcome.node("outcome", &steps));
        // A refused command stops the case here, after every command's own site is emitted: the
        // bytes a refused dispatch did not write would otherwise read as divergences of their own,
        // each a row of the ledger for one bug.
        if let Some((at, failure)) = outcome.first_refusal() {
            return Err(Abort {
                at: format!("outcome/step/{at}/{}", steps[at].call()),
                failure: failure.clone(),
            });
        }
        let mut read = Vec::new();
        for (buffer, length) in row.resources.read_back() {
            read.push(made(
                sites,
                &format!("read-back/{buffer}"),
                side.read_buffer(held.buffers[buffer], 0, length),
            )?);
        }
        for node in row.sites(&read) {
            sites.push(node);
        }
        Ok(())
    }
}

/// The sampler the sampling cases bind: nearest, clamp to edge, normalized coordinates — Metal's
/// own defaults, so what the case varies is the **call** that binds it and nothing else.
fn sampler_spec() -> crate::case::SamplerSpec {
    crate::case::SamplerSpec::filtered(SamplerMinMagFilter::Nearest)
}

/// `region_k`'s one dispatch of [`THREADS`] threads in one threadgroup.
fn region_dispatch() -> ComputeStep {
    ComputeStep::DispatchThreadgroups {
        groups: [1, 1, 1],
        threads: [THREADS, 1, 1],
    }
}

/// Two dynamic threadgroup lengths, a threadgroup-grid dispatch each, and the second dispatch's
/// output moved by `setBufferOffset:atIndex:`.
static THREADGROUP_MEMORY: Row = Row {
    case: "compute-threadgroup-memory-and-grid",
    fixture: Fixture {
        name: REGION_FIXTURE,
        entry: REGION_ENTRY,
    },
    resources: Resources::Region { slots: 2 },
    pass: |made| {
        let out = made
            .buffers
            .first()
            .copied()
            .unwrap_or(crate::case::Handle(0));
        (
            Dispatching::Serial,
            vec![
                ComputeStep::Pipeline(made.pipeline),
                ComputeStep::Buffer {
                    buffer: out,
                    offset: 0,
                    index: 0,
                },
                ComputeStep::ThreadgroupMemory {
                    length: LENGTHS[0],
                    index: 0,
                },
                region_dispatch(),
                ComputeStep::BufferOffset {
                    offset: SLOT as u64,
                    index: 0,
                },
                ComputeStep::ThreadgroupMemory {
                    length: LENGTHS[1],
                    index: 0,
                },
                region_dispatch(),
            ],
        )
    },
    site: "two-lengths",
};

/// `setBuffers:offsets:withRange:` against `setBuffer:offset:atIndex:`.
static PLURAL_BUFFERS: Row = Row {
    case: "compute-plural-buffer-binding",
    fixture: Fixture {
        name: REGION_FIXTURE,
        entry: REGION_ENTRY,
    },
    resources: Resources::Region { slots: 2 },
    pass: |made| {
        let out = made
            .buffers
            .first()
            .copied()
            .unwrap_or(crate::case::Handle(0));
        (
            Dispatching::Serial,
            vec![
                ComputeStep::Pipeline(made.pipeline),
                ComputeStep::ThreadgroupMemory {
                    length: LENGTHS[0],
                    index: 0,
                },
                ComputeStep::Buffer {
                    buffer: out,
                    offset: 0,
                    index: 0,
                },
                region_dispatch(),
                ComputeStep::Buffers {
                    buffers: vec![(out, SLOT as u64)],
                    first: 0,
                },
                region_dispatch(),
            ],
        )
    },
    site: "singular-then-plural",
};

/// `useResource:usage:` and `useResources:count:usage:`, a dispatch behind each.
static RESIDENCY: Row = Row {
    case: "compute-resource-residency",
    fixture: Fixture {
        name: REGION_FIXTURE,
        entry: REGION_ENTRY,
    },
    resources: Resources::Region { slots: 2 },
    pass: |made| {
        let out = made
            .buffers
            .first()
            .copied()
            .unwrap_or(crate::case::Handle(0));
        (
            Dispatching::Serial,
            vec![
                ComputeStep::Pipeline(made.pipeline),
                ComputeStep::ThreadgroupMemory {
                    length: LENGTHS[0],
                    index: 0,
                },
                ComputeStep::Buffer {
                    buffer: out,
                    offset: 0,
                    index: 0,
                },
                ComputeStep::UseResource {
                    resource: PassResource::Buffer(out),
                    usage: USAGE_READ | USAGE_WRITE,
                },
                region_dispatch(),
                ComputeStep::BufferOffset {
                    offset: SLOT as u64,
                    index: 0,
                },
                ComputeStep::UseResources {
                    resources: vec![PassResource::Buffer(out)],
                    usage: USAGE_READ | USAGE_WRITE,
                },
                region_dispatch(),
            ],
        )
    },
    site: "declared-resident",
};

/// A threadgroup grid read from a buffer at a nonzero offset.
static INDIRECT_GRID: Row = Row {
    case: "compute-indirect-threadgroup-grid",
    fixture: Fixture {
        name: REGION_FIXTURE,
        entry: REGION_ENTRY,
    },
    resources: Resources::RegionWithGrid { slots: 1 },
    pass: |made| {
        let out = made
            .buffers
            .first()
            .copied()
            .unwrap_or(crate::case::Handle(0));
        let grid = made
            .buffers
            .get(1)
            .copied()
            .unwrap_or(crate::case::Handle(0));
        (
            Dispatching::Serial,
            vec![
                ComputeStep::Pipeline(made.pipeline),
                ComputeStep::ThreadgroupMemory {
                    length: LENGTHS[0],
                    index: 0,
                },
                ComputeStep::Buffer {
                    buffer: out,
                    offset: 0,
                    index: 0,
                },
                ComputeStep::DispatchThreadgroupsIndirect {
                    buffer: grid,
                    offset: GRID_AT,
                    threads: [THREADS, 1, 1],
                },
            ],
        )
    },
    site: "indirect-grid",
};

/// The chained pair of dispatches, with the barrier `barrier` between them, on a concurrent
/// encoder.
fn chain(made: &Made, barrier: ComputeStep) -> (Dispatching, Vec<ComputeStep>) {
    let handle = |at: usize| {
        made.buffers
            .get(at)
            .copied()
            .unwrap_or(crate::case::Handle(0))
    };
    let (source, middle, out) = (handle(0), handle(1), handle(2));
    (
        Dispatching::Concurrent,
        vec![
            ComputeStep::Pipeline(made.pipeline),
            ComputeStep::Buffer {
                buffer: source,
                offset: 0,
                index: 0,
            },
            ComputeStep::Buffer {
                buffer: middle,
                offset: 0,
                index: 1,
            },
            ComputeStep::DispatchThreadgroups {
                groups: [1, 1, 1],
                threads: [CHAIN_WRITE, 1, 1],
            },
            barrier,
            ComputeStep::Buffer {
                buffer: middle,
                offset: 0,
                index: 0,
            },
            ComputeStep::Buffer {
                buffer: out,
                offset: 0,
                index: 1,
            },
            ComputeStep::DispatchThreadgroups {
                groups: [1, 1, 1],
                threads: [CHAIN_READ, 1, 1],
            },
        ],
    )
}

/// `memoryBarrierWithScope:` between two chained dispatches of a concurrent encoder.
static SCOPE_BARRIER: Row = Row {
    case: "compute-scope-barrier",
    fixture: Fixture::new(INDEX_FIXTURE),
    resources: Resources::Chain,
    pass: |made| {
        chain(
            made,
            ComputeStep::BarrierScope {
                scope: SCOPE_BUFFERS,
            },
        )
    },
    site: "output",
};

/// `memoryBarrierWithResources:count:` between the same two dispatches.
static RESOURCE_BARRIER: Row = Row {
    case: "compute-resource-barrier",
    fixture: Fixture::new(INDEX_FIXTURE),
    resources: Resources::Chain,
    pass: |made| {
        let middle = made
            .buffers
            .get(1)
            .copied()
            .unwrap_or(crate::case::Handle(0));
        chain(
            made,
            ComputeStep::BarrierResources {
                resources: vec![PassResource::Buffer(middle)],
            },
        )
    },
    site: "output",
};

/// The four sampler and texture binding calls, a dispatch and a slot each.
static PLURAL_SAMPLERS: Row = Row {
    case: "compute-plural-texture-and-sampler-binding",
    fixture: Fixture {
        name: SAMPLE_FIXTURE,
        entry: SAMPLE_ENTRY,
    },
    resources: Resources::Sampled { slots: 4 },
    pass: |made| {
        let coords = made
            .buffers
            .first()
            .copied()
            .unwrap_or(crate::case::Handle(0));
        let out = made
            .buffers
            .get(1)
            .copied()
            .unwrap_or(crate::case::Handle(0));
        let texture = made
            .textures
            .first()
            .copied()
            .unwrap_or(crate::case::Handle(0));
        let sampler = made
            .samplers
            .first()
            .copied()
            .unwrap_or(crate::case::Handle(0));
        let dispatch = ComputeStep::DispatchThreadgroups {
            groups: [1, 1, 1],
            threads: [COORDS.len() as u32, 1, 1],
        };
        let mut steps = vec![
            ComputeStep::Pipeline(made.pipeline),
            ComputeStep::Buffer {
                buffer: coords,
                offset: 0,
                index: 0,
            },
            ComputeStep::Buffer {
                buffer: out,
                offset: 0,
                index: 1,
            },
            // Slot 0: the singular setters.
            ComputeStep::Texture { texture, index: 0 },
            ComputeStep::Sampler { sampler, index: 0 },
            dispatch.clone(),
            // Slot 1: the plural setters over a range of one.
            ComputeStep::BufferOffset {
                offset: SAMPLE_SLOT as u64,
                index: 1,
            },
            ComputeStep::Textures {
                textures: vec![texture],
                first: 0,
            },
            ComputeStep::Samplers {
                samplers: vec![sampler],
                first: 0,
            },
            dispatch.clone(),
            // Slot 2: the singular setter with the LOD clamps Metal's own binding has.
            ComputeStep::BufferOffset {
                offset: 2 * SAMPLE_SLOT as u64,
                index: 1,
            },
            ComputeStep::ClampedSampler {
                sampler,
                lod_min: LOD_CLAMPS.0,
                lod_max: LOD_CLAMPS.1,
                index: 0,
            },
            dispatch.clone(),
            // Slot 3: the plural form of the same.
            ComputeStep::BufferOffset {
                offset: 3 * SAMPLE_SLOT as u64,
                index: 1,
            },
            ComputeStep::ClampedSamplers {
                samplers: vec![(sampler, LOD_CLAMPS.0, LOD_CLAMPS.1)],
                first: 0,
            },
        ];
        steps.push(dispatch);
        (Dispatching::Serial, steps)
    },
    site: "four-bindings",
};

/// The LOD clamps the two clamped setters bind with.
///
/// **A one-level texture, so the clamps change no sample**, and the case says so rather than
/// implying otherwise: what it holds is that the two clamped setters bind the same state as the
/// plain ones, which is what makes their items exercised. Observing a clamp needs a mipmapped
/// texture and a sampler with a mip filter, and `SamplerSpec` states no mip filter at all — that
/// is a packet of its own, named in the module's own prose rather than left as a gap nobody wrote
/// down.
const LOD_CLAMPS: (f32, f32) = (0.0, 0.0);

/// `setBytes:length:atIndex:` against a buffer bound at the same index.
static INLINE_BYTES: Row = Row {
    case: "compute-inline-bytes",
    fixture: Fixture {
        name: SAMPLE_FIXTURE,
        entry: SAMPLE_ENTRY,
    },
    resources: Resources::Sampled { slots: 2 },
    pass: |made| {
        let coords = made
            .buffers
            .first()
            .copied()
            .unwrap_or(crate::case::Handle(0));
        let out = made
            .buffers
            .get(1)
            .copied()
            .unwrap_or(crate::case::Handle(0));
        let texture = made
            .textures
            .first()
            .copied()
            .unwrap_or(crate::case::Handle(0));
        let sampler = made
            .samplers
            .first()
            .copied()
            .unwrap_or(crate::case::Handle(0));
        let dispatch = ComputeStep::DispatchThreadgroups {
            groups: [1, 1, 1],
            threads: [COORDS.len() as u32, 1, 1],
        };
        (
            Dispatching::Serial,
            vec![
                ComputeStep::Pipeline(made.pipeline),
                ComputeStep::Texture { texture, index: 0 },
                ComputeStep::Sampler { sampler, index: 0 },
                ComputeStep::Buffer {
                    buffer: out,
                    offset: 0,
                    index: 1,
                },
                // Slot 0: the coordinates handed inline, with no buffer of the case's own behind
                // them.
                ComputeStep::Bytes {
                    bytes: coord_bytes(),
                    index: 0,
                },
                dispatch.clone(),
                // Slot 1: the same coordinates out of the buffer the case made.
                ComputeStep::BufferOffset {
                    offset: SAMPLE_SLOT as u64,
                    index: 1,
                },
                ComputeStep::Buffer {
                    buffer: coords,
                    offset: 0,
                    index: 0,
                },
                dispatch,
            ],
        )
    },
    site: "inline-then-buffer",
};

/// The case of two dynamic threadgroup lengths and a threadgroup-grid dispatch each.
pub fn threadgroup_memory_and_grid() -> ComputeEncoderCommands {
    ComputeEncoderCommands {
        row: &THREADGROUP_MEMORY,
    }
}

/// The case of the plural buffer setter.
pub fn plural_buffer_binding() -> ComputeEncoderCommands {
    ComputeEncoderCommands {
        row: &PLURAL_BUFFERS,
    }
}

/// The case of the two residency declarations.
pub fn resource_residency() -> ComputeEncoderCommands {
    ComputeEncoderCommands { row: &RESIDENCY }
}

/// The case of an indirect threadgroup grid.
pub fn indirect_threadgroup_grid() -> ComputeEncoderCommands {
    ComputeEncoderCommands {
        row: &INDIRECT_GRID,
    }
}

/// The case of a scope barrier between two chained dispatches of a concurrent encoder.
pub fn scope_barrier() -> ComputeEncoderCommands {
    ComputeEncoderCommands {
        row: &SCOPE_BARRIER,
    }
}

/// The case of a resource barrier between the same two dispatches.
pub fn resource_barrier() -> ComputeEncoderCommands {
    ComputeEncoderCommands {
        row: &RESOURCE_BARRIER,
    }
}

/// The case of the four texture and sampler binding calls.
pub fn plural_texture_and_sampler_binding() -> ComputeEncoderCommands {
    ComputeEncoderCommands {
        row: &PLURAL_SAMPLERS,
    }
}

/// The case of the inline bytes.
pub fn inline_bytes() -> ComputeEncoderCommands {
    ComputeEncoderCommands { row: &INLINE_BYTES }
}

/// Every case of the family, in the order the report lists them.
pub fn all() -> Vec<Box<dyn Case>> {
    vec![
        Box::new(threadgroup_memory_and_grid()),
        Box::new(plural_buffer_binding()),
        Box::new(resource_residency()),
        Box::new(indirect_threadgroup_grid()),
        Box::new(scope_barrier()),
        Box::new(resource_barrier()),
        Box::new(plural_texture_and_sampler_binding()),
        Box::new(inline_bytes()),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// Every row of the family, for the guards that read each one.
    const EVERY_ROW: [&Row; 8] = [
        &THREADGROUP_MEMORY,
        &PLURAL_BUFFERS,
        &RESIDENCY,
        &INDIRECT_GRID,
        &SCOPE_BARRIER,
        &RESOURCE_BARRIER,
        &PLURAL_SAMPLERS,
        &INLINE_BYTES,
    ];

    /// The handles a row's pass is built from, for a guard that reads the pass with no device.
    fn made() -> Made {
        Made {
            pipeline: crate::case::Handle(0),
            buffers: (0..4).map(crate::case::Handle).collect(),
            textures: vec![crate::case::Handle(0)],
            samplers: vec![crate::case::Handle(0)],
        }
    }

    /// The cases name distinct recordings, and `all` holds every row.
    #[test]
    fn the_cases_are_distinct_and_complete() {
        let named: BTreeSet<&str> = all().iter().map(|case| case.name()).collect();
        assert_eq!(named.len(), all().len());
        assert_eq!(named.len(), EVERY_ROW.len());
        let rows: BTreeSet<&str> = EVERY_ROW.iter().map(|row| row.case).collect();
        assert_eq!(rows, named);
    }

    /// Every case states its inputs and loosens no value: its one declaration is the notice a
    /// device without unified memory announces for the `Shared` buffer the case reads back
    /// through, at the notices site.
    #[test]
    fn every_case_states_its_inputs_and_loosens_nothing() {
        for case in all() {
            let inputs = case
                .inputs()
                .unwrap_or_else(|why| panic!("{}: {why}", case.name()));
            assert!(!inputs.is_empty(), "{}", case.name());
            // The venue's own memory notice and nothing else: no value of the family is loosened,
            // and the one declaration is a notice declaration at `case/notices/*`.
            let allowances = case.allowances();
            assert_eq!(allowances.len(), 1, "{}", case.name());
            assert_eq!(
                allowances[0].justified_by,
                crate::allowance::Justification::Degradation(
                    metal2vulkan_runtime::Degradation::SharedInSystemMemory
                        .code()
                        .to_owned()
                ),
                "{}",
                case.name()
            );
            assert_eq!(
                allowances[0].path,
                format!("case/{}/*", crate::replay::NOTICES_SITE),
                "{}",
                case.name()
            );
        }
    }

    /// Every dispatch of every case is **one** threadgroup of at most [`THREADS`] threads on one
    /// axis, which is a simdgroup on every Apple GPU and inside the smallest threadgroup limit a
    /// venue reports.
    ///
    /// This is the family's safety rule, held by a test rather than by care: no grid here is
    /// derived from a device limit, so no venue can be handed a threadgroup it cannot run, and no
    /// case can grow one by accident.
    #[test]
    fn every_dispatch_is_one_small_threadgroup() {
        for row in EVERY_ROW {
            let (_, steps) = (row.pass)(&made());
            let dispatches = steps.iter().filter(|step| step.dispatches()).count();
            assert!(dispatches > 0, "{}: no dispatch", row.case);
            for step in &steps {
                let (groups, threads) = match step {
                    ComputeStep::DispatchThreadgroups { groups, threads } => (*groups, *threads),
                    // The indirect grid's own three words are the case's, checked below.
                    ComputeStep::DispatchThreadgroupsIndirect { threads, .. } => (GRID, *threads),
                    _ => continue,
                };
                assert_eq!(groups, [1, 1, 1], "{}: {groups:?} threadgroups", row.case);
                assert_eq!(
                    [threads[1], threads[2]],
                    [1, 1],
                    "{}: {threads:?} threads",
                    row.case
                );
                assert!(
                    threads[0] > 0 && threads[0] <= THREADS,
                    "{}: {} threads a threadgroup",
                    row.case,
                    threads[0]
                );
            }
        }
    }

    /// The indirect grid's three words really are at the offset the dispatch reads, and they are
    /// the grid the guard above assumed.
    ///
    /// Without this the row would dispatch whatever lay at that offset: the buffer's other bytes
    /// are the guard, which as a grid would be 2,122,219,134 threadgroups.
    #[test]
    fn the_indirect_grid_lies_where_the_dispatch_reads_it() {
        let buffers = Resources::RegionWithGrid { slots: 1 }.buffers();
        let grid = &buffers[1];
        let at = GRID_AT as usize;
        assert_eq!(GRID_AT % 4, 0, "the offset is not four-byte aligned");
        assert!(at + 12 <= grid.len(), "the grid runs past its buffer");
        for (axis, want) in GRID.iter().enumerate() {
            let from = at + axis * 4;
            assert_eq!(
                u32::from_le_bytes(grid[from..from + 4].try_into().expect("four bytes")),
                *want
            );
        }
        // And the offset matters: the words before it are not the grid.
        assert_ne!(&grid[..12], &grid[at..at + 12]);
    }

    /// Every dynamic threadgroup allocation a case states holds at least `region_k`'s one word, so
    /// the kernel's read of word 0 is inside what the case allocated.
    #[test]
    fn every_threadgroup_allocation_holds_its_word() {
        for row in EVERY_ROW {
            let (_, steps) = (row.pass)(&made());
            for step in &steps {
                if let ComputeStep::ThreadgroupMemory { length, index } = step {
                    assert!(*length >= 4, "{}: {length} bytes", row.case);
                    assert_eq!(*index, 0, "{}: index {index}", row.case);
                }
            }
        }
        // And every `region_k` row really does allocate one: the kernel declares the array
        // dynamically, so a pass with no such command would read memory it was never given.
        for row in [
            &THREADGROUP_MEMORY,
            &PLURAL_BUFFERS,
            &RESIDENCY,
            &INDIRECT_GRID,
        ] {
            let (_, steps) = (row.pass)(&made());
            assert!(
                steps
                    .iter()
                    .any(|step| matches!(step, ComputeStep::ThreadgroupMemory { .. })),
                "{}: a region_k row with no threadgroup allocation",
                row.case
            );
        }
    }

    /// The chained pair reads inside its buffers: `a4_index_word` reads `in[3t + 1]`, so the
    /// writer's input holds `3 * (CHAIN_WRITE - 1) + 2` words and the reader's middle buffer holds
    /// `3 * (CHAIN_READ - 1) + 2`.
    #[test]
    fn every_chained_dispatch_reads_inside_its_buffer() {
        let buffers = Resources::Chain.buffers();
        let words_in = |at: usize| buffers[at].len() / 4;
        assert!(
            words_in(0) >= 3 * (CHAIN_WRITE as usize - 1) + 2,
            "the input holds {} words for {CHAIN_WRITE} threads",
            words_in(0)
        );
        assert!(
            words_in(1) >= CHAIN_WRITE as usize,
            "the middle buffer holds {} of {CHAIN_WRITE} written words",
            words_in(1)
        );
        assert!(
            words_in(1) >= 3 * (CHAIN_READ as usize - 1) + 2,
            "the middle buffer holds {} words for {CHAIN_READ} reading threads",
            words_in(1)
        );
        assert!(words_in(2) >= CHAIN_READ as usize);
    }

    /// Each row's pass fingerprints differently, so the input fingerprint tells the rows apart.
    ///
    /// The two barrier rows are the pair this matters most for: they differ in **one command**, and
    /// a fingerprint that missed it would let either recording stand for the other.
    #[test]
    fn the_rows_passes_are_distinct() {
        let digests: BTreeSet<String> = EVERY_ROW
            .iter()
            .map(|row| {
                let (dispatching, steps) = (row.pass)(&made());
                crate::sha256_hex(
                    &ComputePass {
                        dispatching,
                        steps: &steps,
                    }
                    .words(),
                )
            })
            .collect();
        assert_eq!(digests.len(), EVERY_ROW.len());
    }

    /// The two barrier rows run on a **concurrent** encoder and every other row on a serial one.
    ///
    /// The barrier rows' whole claim rests on it: on a serial encoder Metal orders the dispatches
    /// itself, and the barrier would be encoding a command that changes nothing.
    #[test]
    fn only_the_barrier_rows_dispatch_concurrently() {
        let concurrent: Vec<&str> = EVERY_ROW
            .iter()
            .filter(|row| (row.pass)(&made()).0 == Dispatching::Concurrent)
            .map(|row| row.case)
            .collect();
        assert_eq!(concurrent, [SCOPE_BARRIER.case, RESOURCE_BARRIER.case]);
        for row in [&SCOPE_BARRIER, &RESOURCE_BARRIER] {
            let (_, steps) = (row.pass)(&made());
            assert!(
                steps.iter().any(|step| matches!(
                    step,
                    ComputeStep::BarrierScope { .. } | ComputeStep::BarrierResources { .. }
                )),
                "{}: a concurrent chain with no barrier",
                row.case
            );
        }
    }

    /// Every row's read-back names a buffer the row made, and asks for no more bytes than it
    /// holds.
    #[test]
    fn every_read_back_lies_inside_its_buffer() {
        for row in EVERY_ROW {
            let buffers = row.resources.buffers();
            let reads = row.resources.read_back();
            assert!(!reads.is_empty(), "{}", row.case);
            for (at, length) in reads {
                assert!(at < buffers.len(), "{}: buffer {at}", row.case);
                assert!(
                    length <= buffers[at].len(),
                    "{}: {length} bytes of a buffer of {}",
                    row.case,
                    buffers[at].len()
                );
            }
        }
    }

    /// Every buffer offset a pass binds leaves the dispatch's own writes inside the buffer.
    ///
    /// A slot's worth of words past the offset has to fit, or the kernel would write past the
    /// buffer — which is the one way a case of this family could fault a device.
    #[test]
    fn every_bound_offset_leaves_the_writes_inside_the_buffer() {
        for row in EVERY_ROW {
            let buffers = row.resources.buffers();
            let (_, steps) = (row.pass)(&made());
            let slot = match row.resources {
                Resources::Sampled { .. } => SAMPLE_SLOT,
                _ => SLOT,
            };
            for step in &steps {
                let offset = match step {
                    ComputeStep::BufferOffset { offset, .. } => *offset,
                    ComputeStep::Buffers { buffers: bound, .. } => {
                        bound.iter().map(|(_, at)| *at).max().unwrap_or(0)
                    }
                    _ => continue,
                };
                let largest = buffers.iter().map(Vec::len).max().unwrap_or(0);
                assert!(
                    offset as usize + slot <= largest,
                    "{}: an offset of {offset} plus a slot of {slot} passes {largest} bytes",
                    row.case
                );
            }
        }
    }

    /// Every output buffer carries [`TAIL`] of guard past its last slot, so a dispatch that wrote
    /// more words than the case asked for shows as a guard byte that moved.
    #[test]
    fn every_output_buffer_carries_a_guard_tail() {
        // A tail of no bytes would make the guard a decoration: the last slot would end where
        // the buffer does, and a dispatch that wrote one word too many would write past it.
        assert_eq!(output(1, SLOT).len() - SLOT, TAIL);
        assert_ne!(TAIL, 0);
        for slots in 1..5 {
            for each in [SLOT, SAMPLE_SLOT] {
                let buffer = output(slots, each);
                assert_eq!(buffer.len(), slots * each + TAIL);
                assert!(buffer.iter().all(|byte| *byte == GUARD));
            }
        }
    }

    /// The bytes a committed recording holds at `site`.
    fn recorded(case: &dyn Case, site: &str) -> Vec<u8> {
        let recording = crate::recording::read(case.name())
            .unwrap_or_else(|why| panic!("{}: {why}", case.name()));
        let outcome = recording
            .answers
            .first()
            .expect("the recording holds an answer")
            .outcomes
            .first()
            .expect("the answer holds an outcome");
        let node = outcome
            .transcript
            .root
            .child(site)
            .unwrap_or_else(|| panic!("{}: the recording holds no {site}", case.name()));
        match &node.observed.value {
            crate::transcript::Value::Bytes(bytes) => bytes.clone(),
            other => panic!("{site}: the recording holds {}", other.kind()),
        }
    }

    /// `word`'s value at word `at` of `bytes`.
    fn word_at(bytes: &[u8], at: usize) -> u32 {
        u32::from_le_bytes(bytes[at * 4..at * 4 + 4].try_into().expect("four bytes"))
    }

    /// **Metal's two spellings of one binding wrote the same words.** For every row that runs the
    /// same dispatch once per binding call, each slot of the recording equals the first, the guard
    /// tail is intact, and the first slot is not still the guard.
    ///
    /// This is the cross-check the equivalence rows exist for, and the one that earns its place:
    /// the first recording of `compute-inline-bytes` failed it, and the cause was this suite's own
    /// Metal leg handing `setBytes:length:atIndex:` the address of a `Vec` rather than of its
    /// bytes. Without a test over the recording that read like a bug in the product.
    #[test]
    fn metals_slots_agree_for_every_equivalence_row() {
        let mut checked = 0usize;
        for case in all() {
            let row = EVERY_ROW
                .iter()
                .find(|row| row.case == case.name())
                .expect("every case is a row");
            let Some((slots, each)) = row.slots() else {
                continue;
            };
            let bytes = recorded(case.as_ref(), row.site);
            assert_eq!(bytes.len(), slots * each + TAIL, "{}", row.case);
            let first = &bytes[..each];
            assert!(
                first.iter().any(|byte| *byte != GUARD),
                "{}: the first slot still reads the guard, so the dispatch wrote nothing",
                row.case
            );
            for slot in 1..slots {
                assert_eq!(
                    &bytes[slot * each..(slot + 1) * each],
                    first,
                    "{}: slot {slot} differs from slot 0, so Metal's two spellings of one binding \
                     do not agree",
                    row.case
                );
            }
            assert_eq!(
                &bytes[slots * each..],
                vec![GUARD; TAIL].as_slice(),
                "{}: the guard past the last slot moved",
                row.case
            );
            checked += 1;
        }
        // Five rows are an equivalence of two or four binding calls; the chained pair and the
        // indirect grid are not, and are checked by their own tests below.
        assert_eq!(checked, 5);
    }

    /// **The chained pair read the writer's words, not the buffer the case filled.** For both
    /// barrier rows: the middle buffer holds `in[3t + 1]`, so the first dispatch ran, and the
    /// output holds `in[9t + 4]`, so the second read what the first wrote.
    ///
    /// Both halves matter. Without the middle buffer a run in which the barrier did nothing and
    /// the second dispatch read the case's own guard bytes could still look like an answer; the
    /// guard is `0x7e7e7e7e`, which `in[9t + 4]` never is.
    #[test]
    fn metals_chained_dispatches_read_the_writers_words() {
        let source = words(CHAIN_IN);
        for case in [scope_barrier(), resource_barrier()] {
            let middle = recorded(&case, "middle");
            let output = recorded(&case, "output");
            for t in 0..CHAIN_WRITE as usize {
                assert_eq!(
                    word_at(&middle, t),
                    word_at(&source, 3 * t + 1),
                    "{}: the writer's word {t}",
                    case.name()
                );
            }
            for t in 0..CHAIN_READ as usize {
                assert_eq!(
                    word_at(&output, t),
                    word_at(&source, 9 * t + 4),
                    "{}: the reader's word {t}",
                    case.name()
                );
            }
            assert_eq!(
                &output[CHAIN_READ as usize * 4..],
                vec![GUARD; TAIL].as_slice(),
                "{}: the guard past the output moved",
                case.name()
            );
        }
        // And the chain is not vacuous: the words it reads are not the guard the middle buffer was
        // made holding, so a barrier that did nothing could not produce this answer.
        let guard = u32::from_le_bytes([GUARD; 4]);
        for t in 0..CHAIN_READ as usize {
            assert_ne!(word_at(&source, 9 * t + 4), guard);
        }
    }

    /// The indirect dispatch ran the grid the buffer held: one threadgroup of [`THREADS`] threads
    /// wrote its slot, and the guard past it is intact.
    ///
    /// A side that read the grid from the buffer's start would have dispatched 0x7e7e7e7e
    /// threadgroups, so what this holds is that the offset was honoured and nothing beyond one
    /// threadgroup's words moved.
    #[test]
    fn metals_indirect_grid_ran_one_threadgroup() {
        let bytes = recorded(&indirect_threadgroup_grid(), INDIRECT_GRID.site);
        assert_eq!(bytes.len(), SLOT + TAIL);
        assert!(bytes[..SLOT].iter().any(|byte| *byte != GUARD));
        assert_eq!(&bytes[SLOT..], vec![GUARD; TAIL].as_slice());
    }

    /// Every row's site is a name no other row uses for a different thing, and the chained rows
    /// are the only ones that emit a second site.
    #[test]
    fn the_rows_name_their_sites() {
        for row in EVERY_ROW {
            let read: Vec<Vec<u8>> = vec![vec![0; 4], vec![0; 4]];
            let names: Vec<String> = row
                .sites(&read)
                .iter()
                .map(|node| node.name.to_string())
                .collect();
            let wanted: Vec<String> = match row.resources {
                Resources::Chain => vec![row.site.to_owned(), "middle".to_owned()],
                _ => vec![row.site.to_owned()],
            };
            assert_eq!(names, wanted, "{}", row.case);
        }
    }

    /// The coordinates the sampling rows use lie inside the texture and none of them falls on a
    /// texel boundary, where a nearest sample's answer depends on a rounding rule rather than on
    /// the texel.
    #[test]
    fn the_sampled_coordinates_avoid_the_texel_boundaries() {
        let side = TEXTURE_SIDE as f32;
        for axes in COORDS {
            for axis in axes {
                assert!((0.0..1.0).contains(&axis), "{axis} is outside the texture");
                let texel = axis * side;
                assert!(
                    (texel - texel.round()).abs() > 0.05,
                    "{axis} lands on the boundary between two texels"
                );
            }
        }
        assert_eq!(coord_bytes().len(), COORDS.len() * 8);
    }
}
