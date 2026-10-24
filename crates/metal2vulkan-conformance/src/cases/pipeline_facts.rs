//! What a pipeline **state** says about itself, compute and render.
//!
//! `MTLRenderPipelineState` was exercised at none of its fifteen items and
//! `MTLComputePipelineState` at one of fifteen, while the suite made hundreds of pipeline states:
//! every case set one and drew or dispatched with it, and no case asked one what it was. Three
//! `MTLRenderPipelineState` getters were compared — on a **tile** state, through
//! [`Side::tile_facts`] — and nothing on a drawing state at all.
//!
//! The getters are not all worth the same, and [`crate::pipeline_facts`] says which is which: a
//! `label` and a `supportIndirectCommandBuffers` are round trips through the descriptor, while
//! `staticThreadgroupMemoryLength`, `imageblockSampleLength` and
//! `imageblockMemoryLengthForDimensions:` are computed by our side from the function and the
//! attachment formats, by rules the runtime's own documentation says were measured on an Apple
//! M3 Max and an M4. This family is what holds those rules to Metal through a recording.
//!
//! [`Side::tile_facts`]: crate::case::Side::tile_facts
//!
//! # The shapes, and why each is here
//!
//! - **a compute state with nothing set** (`ob1_threadgroup`): the floor, and the one case that
//!   says a nil label comes back nil rather than empty;
//! - **a compute state with a label, indirect-command support and a required threadgroup size**:
//!   the three descriptor fields the state echoes, each a value distinct from its default, so one
//!   wired to another's would read wrong rather than pass;
//! - **a compute state whose kernel declares an explicit imageblock**
//!   (`a4_imageblock_extent`): the only shape whose
//!   `imageblockMemoryLengthForDimensions:` is anything but zero, asked at an extent whose three
//!   axes differ so a width read as a height is a different number;
//! - **a drawing state over one attachment**: `imageblockSampleLength` is that attachment's
//!   texel, `maxTotalThreadsPerThreadgroup` is 0 and `threadgroupSizeMatchesTileSize` is false,
//!   which is a drawing state's whole answer and not the device's limits;
//! - **a drawing state over two attachments of different formats, with a label and
//!   indirect-command support**: the sample length is the two texels packed, so a case where the
//!   packing rule has something to get wrong;
//! - **the same pipeline at four samples**: the memory length is the sample length times the
//!   extent times the raster sample count, and the sample count is the factor only this case
//!   moves.
//!
//! # Why this family is safe
//!
//! Each case makes one library and one pipeline state and reads its getters. Nothing is
//! allocated for the device, no command buffer is made and nothing is drawn or dispatched, so no
//! case here can fault the device or trip its watchdog. The pipeline creation is the one-time
//! translation cost the project's ceilings already cover.
//!
//! # Exactness
//!
//! Every field is a number, a bool or a name, and the family declares no tolerance and announces
//! nothing: no case touches a texture, a sampler or a float. The two getters that answer the
//! **device's** limits for a kernel are deliberately not read at all, for the reason
//! [`crate::pipeline_facts`] states.

use crate::case::{Abort, Case, Failure, Input, MetalSource, Side};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::pipeline_facts::ComputeStateSpec;
use crate::recording::ShaderRef;
use crate::render::{ColorState, RenderPipelineSpec};
use crate::transcript::{Digest, Node, Observed, Sites};
use metal2vulkan::metal::PixelFormat;

/// The site the library's make sits at.
const LIBRARY: &str = "library";

/// The site the state's make sits at.
const STATE: &str = "state";

/// The site the state's own answers sit at.
const FACTS: &str = "facts";

/// The imageblock extent every case asks `imageblockMemoryLengthForDimensions:` about.
///
/// Three different axes, so a side that read the width where the height belongs answers a
/// different number. The depth is 3 and Metal ignores it, which is part of the answer: a side that
/// multiplied by it would be off by three.
const IMAGEBLOCK: [u32; 3] = [8, 4, 3];

/// The fixture whose kernel takes a `threadgroup uint *`, which is the shape whose
/// `staticThreadgroupMemoryLength` Metal answers **zero** for: that memory is the argument's, sized
/// at the dispatch by `setThreadgroupMemoryLength:atIndex:`, and not an array the function declares.
/// All 13 sites of the committed recordings answer 0, so what the case compares here is that zero.
/// `function_bindings`' kernel.
const THREADGROUP_FIXTURE: &str = "a27_ob1_threadgroup";
const THREADGROUP_ENTRY: &str = "ob1_threadgroup";

/// The fixture whose kernel declares an explicit imageblock of a `float` and a `ushort2`, which is
/// the one shape whose imageblock memory length is not zero.
const IMAGEBLOCK_FIXTURE: &str = "a4_imageblock_extent";

/// The fixture whose vertex function is Metal's fullscreen triangle and whose fragment function
/// returns white: `color_write_masks`' own, so this family needs no fixture of its own. Its
/// `rt6_resolve_masked` writes two colour attachments, which is what the two-attachment case
/// takes.
const RENDER_FIXTURE: &str = "rt6_resolves";
const VERTEX: &str = "rt6_resolve_full";
const FRAGMENT: &str = "rt6_resolve_white";
const TWO_TARGETS: &str = "rt6_resolve_masked";

/// One colour attachment of `RGBA8Unorm`, whose texel is four bytes.
const ONE_COLOR: &[Option<ColorState>] = &[Some(ColorState::new(PixelFormat::RGBA8Unorm))];

/// Two colour attachments of **different** texel sizes — four bytes and eight — so the packed
/// sample length is a number neither format gives on its own.
const TWO_COLORS: &[Option<ColorState>] = &[
    Some(ColorState::new(PixelFormat::RGBA8Unorm)),
    Some(ColorState::new(PixelFormat::RGBA16Float)),
];

/// What one case makes and reads.
enum Subject {
    /// A compute pipeline state from a whole descriptor.
    Compute(ComputeStateSpec<'static>),
    /// A drawing render pipeline state from a whole descriptor.
    Render(RenderPipelineSpec<'static>),
}

/// A case over one pipeline state's own facts.
pub struct PipelineFactsCase {
    name: &'static str,
    fixture: &'static str,
    /// The entry point the case's input fingerprint names, which for a render case is its vertex
    /// function.
    entry: &'static str,
    subject: Subject,
}

/// Every case of the family.
pub fn all() -> Vec<PipelineFactsCase> {
    vec![
        // The floor: a kernel compiled from a descriptor with Metal's defaults. Its label is nil,
        // it supports no indirect command, it requires no threadgroup size, and its imageblock
        // memory length is zero because it declares no imageblock.
        PipelineFactsCase {
            name: "pipeline-facts-compute-plain",
            fixture: THREADGROUP_FIXTURE,
            entry: THREADGROUP_ENTRY,
            subject: Subject::Compute(ComputeStateSpec::new(THREADGROUP_ENTRY)),
        },
        // The three descriptor fields the state echoes, each away from its default: a label, the
        // indirect-command support, and a required threadgroup size whose three axes differ.
        PipelineFactsCase {
            name: "pipeline-facts-compute-described",
            fixture: THREADGROUP_FIXTURE,
            entry: THREADGROUP_ENTRY,
            subject: Subject::Compute(
                ComputeStateSpec::new(THREADGROUP_ENTRY)
                    .labelled("conformance-compute-state")
                    .for_indirect_commands()
                    .requiring([4, 2, 1]),
            ),
        },
        // The one kernel whose imageblock memory length is not zero.
        PipelineFactsCase {
            name: "pipeline-facts-compute-imageblock",
            fixture: IMAGEBLOCK_FIXTURE,
            entry: IMAGEBLOCK_FIXTURE,
            subject: Subject::Compute(
                ComputeStateSpec::new(IMAGEBLOCK_FIXTURE).labelled("conformance-imageblock-state"),
            ),
        },
        // A drawing state over one attachment, with Metal's defaults for the rest: the shape that
        // says what a drawing state answers for the three getters a tile state answers
        // differently.
        PipelineFactsCase {
            name: "pipeline-facts-render-plain",
            fixture: RENDER_FIXTURE,
            entry: VERTEX,
            subject: Subject::Render(RenderPipelineSpec::new(VERTEX, FRAGMENT, ONE_COLOR)),
        },
        // Two attachments of different texel sizes, with a label and indirect-command support: the
        // sample length is the two packed, which neither format gives alone.
        PipelineFactsCase {
            name: "pipeline-facts-render-two-attachments",
            fixture: RENDER_FIXTURE,
            entry: VERTEX,
            subject: Subject::Render(
                RenderPipelineSpec::new(VERTEX, TWO_TARGETS, TWO_COLORS)
                    .labelled("conformance-render-state")
                    .for_indirect_commands(),
            ),
        },
        // The same pipeline at four samples: the raster sample count is the factor the memory
        // length carries and the sample length does not, and this is the case that moves it.
        PipelineFactsCase {
            name: "pipeline-facts-render-multisampled",
            fixture: RENDER_FIXTURE,
            entry: VERTEX,
            subject: Subject::Render(
                RenderPipelineSpec::new(VERTEX, TWO_TARGETS, TWO_COLORS)
                    .labelled("conformance-render-state")
                    .for_indirect_commands()
                    .multisampled(4),
            ),
        },
    ]
}

impl PipelineFactsCase {
    fn loaded(&self) -> Fixture {
        Fixture {
            name: self.fixture,
            entry: self.entry,
        }
    }
}

impl Case for PipelineFactsCase {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let fixture = self.loaded();
        [RUNTIME_TEXT, METAL_TEXT]
            .into_iter()
            .map(|extension| {
                Ok(ShaderRef {
                    fixture: format!("{}{extension}", self.fixture),
                    entry: self.entry.to_owned(),
                    source_sha256: crate::sha256_hex(fixture.text(extension)?.as_bytes()),
                })
            })
            .collect()
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let fixture = self.loaded();
        let descriptor = match &self.subject {
            Subject::Compute(spec) => spec.words(),
            Subject::Render(spec) => spec.words(),
        };
        Ok(vec![
            Input {
                name: "air".to_owned(),
                digest: Digest::sha256(fixture.text(RUNTIME_TEXT)?.as_bytes()),
            },
            Input {
                name: "air-typed".to_owned(),
                digest: Digest::sha256(fixture.text(METAL_TEXT)?.as_bytes()),
            },
            // The descriptor is the whole of the rest of what the case feeds: every fact it reads
            // is a fact about the state this descriptor made.
            Input::bytes("descriptor", &descriptor),
            Input::bytes(
                "imageblock",
                &IMAGEBLOCK
                    .iter()
                    .flat_map(|axis| axis.to_le_bytes())
                    .collect::<Vec<u8>>(),
            ),
        ])
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let stop = |at: &str, why: String| Abort {
            at: at.to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        };
        let loaded = self.loaded().loaded().map_err(|why| stop(LIBRARY, why))?;
        let library = crate::case::made(sites, LIBRARY, side.library(&loaded.source()))?;
        // One `match` for the two kinds, each emitting its own facts under the same site name:
        // what a case means is "the state this descriptor made says this about itself", and the
        // two kinds of state answer different getters, so the two nodes hold different children
        // and a recording of one is never read against the other.
        match &self.subject {
            Subject::Compute(spec) => {
                let state = crate::case::made(
                    sites,
                    STATE,
                    side.compute_pipeline_with_descriptor(library, spec),
                )?;
                match side.compute_pipeline_facts(state, IMAGEBLOCK) {
                    Ok(facts) => sites.push(facts.node(FACTS)),
                    Err(failure) => return Err(refused(sites, failure)),
                }
            }
            Subject::Render(spec) => {
                let state = crate::case::made(sites, STATE, side.render_pipeline(library, spec))?;
                match side.render_pipeline_facts(state, IMAGEBLOCK) {
                    Ok(facts) => sites.push(facts.node(FACTS)),
                    Err(failure) => return Err(refused(sites, failure)),
                }
            }
        }
        Ok(())
    }
}

/// The abort a refused facts call is, with the typed error recorded at the facts site so it
/// compares as a refusal rather than as a missing site.
fn refused(sites: &mut Sites, failure: Failure) -> Abort {
    sites.push(Node::leaf(FACTS, Observed::value(failure.recorded())));
    Abort {
        at: FACTS.to_owned(),
        failure,
    }
}
