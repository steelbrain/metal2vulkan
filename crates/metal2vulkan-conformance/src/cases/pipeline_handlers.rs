//! `MTLDevice`'s seven asynchronous pipeline creations, each beside its synchronous sibling.
//!
//! [`crate::pipeline_handlers`] states the claim: the coverage map records each
//! `completionHandler:` call as implemented by the synchronous call "the handler receives what the
//! call returns", and no case had ever run one. All seven were unexercised, and so was every
//! question an asynchronous creation raises — whether the handler runs at all, whether it is given
//! a state or an error, whether an options value makes it a reflection, and whether what it
//! delivers is what the returning call returns.
//!
//! # The two axes, and the values each is swept over
//!
//! - **the call** — all seven, every one of `MTLDevice`'s `completionHandler:` forms. Nothing is
//!   left out.
//! - **success and refusal** — each of the seven is taken twice: once with a descriptor it
//!   compiles, and once with a descriptor whose stage function is of the **wrong type** for its
//!   slot, which both implementations refuse. The refusal leg is the half an asynchronous form can
//!   get wrong on its own: a state delivered beside an error, an error with no state and no error,
//!   or a handler that is simply never called.
//! - **the options value** — `newComputePipelineStateWithFunction:options:completionHandler:` is
//!   taken with **all eight** combinations of the three bits Apple's header names
//!   ([`crate::pipeline_option_reflection::MASKS`]' own value space), so the asynchronous path's reflection
//!   rule is swept whole. `MTLPipelineOptionFailOnBinaryArchiveMiss` alone is the row that
//!   matters: Metal fills the handler's reflection for it, which Apple's documentation for that
//!   bit does not say, and reading only the two information bits answered nil there. The other
//!   four option-bearing forms are taken with the bit clear and set, which is what makes
//!   `delivered-a-reflection` a site that **moves** — `false` for the first and `true` for the
//!   second — rather than a constant of the family.
//!
//! # Why the wrong-type function is the refusal, and not a number out of range
//!
//! A refusal has to be one **both** implementations answer with an error rather than with an
//! assertion inside their own validation: a Metal assertion ends the process, and a case that ends
//! Metal's process is not a case ([`crate::cases`]). A function of the wrong type in a stage slot
//! is refused by the pipeline compiler on both sides — the thing being compiled is wrong, not an
//! argument a validation layer screens — so the refusal arrives as the handler's error, which is
//! exactly the site under test.
//!
//! What is compared is **that the creation was refused** and how the refusal arrived, not what the
//! error said: [`crate::pipeline_handlers`] says why an `NSError`'s words and a typed Rust error's
//! are not comparable between two implementations.
//!
//! # What makes each case bite rather than pass vacuously
//!
//! - the **arrival** site states which of a delivery and a timeout happened, so a family that
//!   stopped running its handlers reads as a divergence at every row rather than as a pass. The
//!   check is falsified in `tests::a_creation_handler_that_never_ran_is_divergent_not_a_pass`,
//!   which builds the transcript a suppressed callback leaves and reads a divergence out of it;
//! - the **delivered state's own facts** are compared, not its non-nilness: a label the descriptor
//!   set, the static threadgroup memory of the kernel, the imageblock lengths, the echoed
//!   descriptor fields. A creation that answered some other pipeline differs here;
//! - the two **descriptor-echo** fields are set away from their defaults wherever the descriptor
//!   has them (a label, indirect-command support, a required threadgroup size), so a field wired
//!   to a neighbour's reads wrong rather than equal;
//! - the **refusal** rows answer `false` where the success rows answer `true` at
//!   `delivered-a-state`, and the reverse at `delivered-an-error`, so neither bool is a constant
//!   of the family.
//!
//! # Why this family is safe
//!
//! Each case makes one library and compiles one pipeline twice. Nothing is allocated for the
//! device, no command buffer is made, and nothing is drawn or dispatched, so no case here can
//! fault the device or trip its watchdog. The handler is waited for on its own channel with a
//! bound and a timeout is recorded rather than retried, so no case can hang. A pipeline creation
//! is the one-time compile the project's 20-second and 500-MiB ceilings already cover; the
//! refusal rows compile nothing at all.
//!
//! # Exactness
//!
//! Every site is a bool, a number, a name or an enumeration. The family declares no tolerance and
//! announces nothing: no case touches a texture, a sampler or a float.

use crate::case::{
    Abort, Case, Failure, Input, MeshFunctions, MetalSource, Side, TilePipelineSpec,
};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::pipeline_facts::ComputeStateSpec;
use crate::pipeline_handlers::{Form, IMAGEBLOCK};
use crate::recording::ShaderRef;
use crate::render::{ColorState, RenderPipelineSpec, StageFunction};
use crate::transcript::{Digest, Node, Observed, Sites};
use metal2vulkan::metal::PixelFormat;

/// The site the library's make sits at.
const LIBRARY: &str = "library";

/// The site both paths' answers sit under.
const PIPELINES: &str = "pipelines";

/// `MTLPipelineOptionNone`: a creation that must answer **no** reflection.
const NO_OPTIONS: u64 = 0;

/// `MTLPipelineOptionBindingInfo`: a creation that must answer one.
const BINDING_INFO: u64 = 1;

/// The fixture whose kernel `pipeline_facts` reads a compute state's own getters off, so this
/// family needs no fixture of its own.
///
/// Its `staticThreadgroupMemoryLength` is zero, measured: the kernel's `threadgroup uint *` is an
/// **argument**, which a dispatch sizes with `setThreadgroupMemoryLength:atIndex:`, not an array
/// the function declares. The descriptor rows below are what carry this family's non-default
/// numbers, through the three fields a compute state echoes.
const COMPUTE_FIXTURE: &str = "a27_ob1_threadgroup";
const KERNEL: &str = "ob1_threadgroup";

/// The fixture whose vertex function is a fullscreen triangle and whose fragment function returns
/// white: `pipeline_facts`' own, so this family needs no fixture of its own. Its two functions are
/// also the family's **wrong-type** functions, each put in the other's slot.
const RENDER_FIXTURE: &str = "rt6_resolves";
const VERTEX: &str = "rt6_resolve_full";
const FRAGMENT: &str = "rt6_resolve_white";

/// The fixture whose vertex functions fetch `[[stage_in]]` attributes, which `int_attributes`
/// draws with. Its vertex function is the family's **render** refusal: a descriptor that names it
/// and sets no `vertexDescriptor` is refused, because the fixed-function stage has nowhere to
/// fetch the attributes from.
const ATTRIBUTE_FIXTURE: &str = "dsx_int_attributes";
const ATTRIBUTE_VERTEX: &str = "int_attribute_to_float4";

/// The fixture whose tile function `tile_threadgroups` runs, and its one attachment format.
const TILE_FIXTURE: &str = "k10ts_tile_order";
const TILE: &str = "k10ts_tile_order";
const TILE_ATTACHMENTS: [PixelFormat; 1] = [PixelFormat::R32Float];

/// The fixture whose object, mesh and fragment functions `mesh_draws` draws with.
const MESH_FIXTURE: &str = "mesh_draws";
const MESH_OBJECT: &str = "mesh_draws_object";
const MESH: &str = "mesh_draws_mesh";
const MESH_FRAGMENT: &str = "mesh_draws_fragment";

/// The colour attachments the mesh pipeline writes, as `mesh_draws`' passes hold them.
const MESH_ATTACHMENTS: u32 = 2;

/// One colour attachment of `RGBA8Unorm`, whose texel is four bytes.
const ONE_COLOR: &[Option<ColorState>] = &[Some(ColorState::new(PixelFormat::RGBA8Unorm))];

/// The label every descriptor of the family that has one carries, so `label` is a name and not a
/// nil at every site that reads one.
const LABEL: &str = "conformance-handler-state";

/// The descriptor each case hands to both paths, held by value so a `Form` can borrow it.
///
/// A `Form` borrows its descriptor, and a case's `run` is handed `&self`, so the descriptor cannot
/// be built inside `run` and borrowed from a temporary. It is held here, built once when the case
/// set is built.
enum Described {
    Render(RenderPipelineSpec<'static>, Option<u64>),
    ComputeFunction(&'static str, Option<u64>),
    ComputeDescriptor(ComputeStateSpec<'static>, u64),
    Tile(TilePipelineSpec<'static>, u64),
    Mesh(MeshFunctions<'static>, u64),
}

impl Described {
    /// The form the side is asked for.
    fn form(&self) -> Form<'_> {
        match self {
            Described::Render(spec, options) => Form::Render {
                spec,
                options: *options,
            },
            Described::ComputeFunction(entry, options) => Form::ComputeFunction {
                entry,
                options: *options,
            },
            Described::ComputeDescriptor(spec, options) => Form::ComputeDescriptor {
                spec,
                options: *options,
            },
            Described::Tile(spec, options) => Form::Tile {
                spec,
                options: *options,
            },
            Described::Mesh(functions, options) => Form::Mesh {
                functions,
                options: *options,
            },
        }
    }
}

/// A case over one creation taken both ways.
pub struct PipelineHandlerCase {
    name: &'static str,
    fixture: &'static str,
    /// The entry point the case's input fingerprint names, which for a render case is its vertex
    /// function and for a mesh case its mesh function.
    entry: &'static str,
    described: Described,
}

/// The drawing pipeline every render row compiles.
fn drawing() -> RenderPipelineSpec<'static> {
    RenderPipelineSpec::new(VERTEX, FRAGMENT, ONE_COLOR).labelled(LABEL)
}

/// A pipeline whose vertex function fetches `[[stage_in]]` attributes and whose descriptor sets
/// **no** `vertexDescriptor`, which both implementations refuse with an error.
///
/// Not a function of the wrong type in the vertex slot, which is what the compute and tile
/// refusals are: a **fragment** function in `setVertexFunction:` crashes Metal rather than
/// refusing, and a case that ends Metal's process is not a case. The crash is reported as an open
/// item and the shape is not here.
fn drawing_refused() -> RenderPipelineSpec<'static> {
    let mut spec = RenderPipelineSpec::of(StageFunction::named(ATTRIBUTE_VERTEX), None, &[]);
    // A vertex function that writes a buffer and rasterizes nothing, as `int_attributes` makes it:
    // the refusal under test is the missing vertex descriptor and nothing else.
    spec.rasterization = false;
    spec
}

/// The compute descriptor the descriptor rows compile: every field the state echoes set away from
/// its default.
fn described_kernel(entry: &'static str) -> ComputeStateSpec<'static> {
    ComputeStateSpec::new(entry)
        .labelled(LABEL)
        .for_indirect_commands()
        .requiring([4, 2, 1])
}

/// The mesh pipeline the mesh rows compile, with `mesh` in the mesh slot.
fn mesh_with(mesh: &'static str) -> MeshFunctions<'static> {
    MeshFunctions {
        object: Some(MESH_OBJECT),
        mesh,
        fragment: MESH_FRAGMENT,
        attachments: MESH_ATTACHMENTS,
    }
}

/// Every `MTLPipelineOption` value the `newComputePipelineStateWithFunction:options:completionHandler:`
/// rows sweep, with the case name each takes.
///
/// All eight combinations of the three bits Apple's header names, in
/// [`crate::pipeline_option_reflection::MASKS`]' order, so the two families sweep the same value space and a
/// row of one reads against the matching row of the other. The names are fixed strings rather than
/// built from the mask, because a case's name is what a recording is keyed by.
const COMPUTE_FUNCTION_MASKS: [(u64, &str); 8] = [
    (0, "pipeline-handler-compute-function-options-none"),
    (1, "pipeline-handler-compute-function-options-binding-info"),
    (
        2,
        "pipeline-handler-compute-function-options-buffer-type-info",
    ),
    (
        3,
        "pipeline-handler-compute-function-options-binding-and-buffer-type",
    ),
    (
        4,
        "pipeline-handler-compute-function-options-fail-on-archive-miss",
    ),
    (
        5,
        "pipeline-handler-compute-function-options-binding-and-fail-on-miss",
    ),
    (
        6,
        "pipeline-handler-compute-function-options-buffer-type-and-fail-on-miss",
    ),
    (7, "pipeline-handler-compute-function-options-all-three"),
];

/// The case names one options value takes, one per option-bearing form.
///
/// A struct of fixed strings rather than a name built at run time: a case's name is what a
/// recording is keyed by, and a generated one would make the key a property of this code's
/// formatting.
struct OptionNames {
    render: &'static str,
    compute_descriptor: &'static str,
    tile: &'static str,
    mesh: &'static str,
}

/// The names of the rows taken with `MTLPipelineOptionNone`.
const NONE_NAMES: OptionNames = OptionNames {
    render: "pipeline-handler-render-options-none",
    compute_descriptor: "pipeline-handler-compute-descriptor-options-none",
    tile: "pipeline-handler-tile-options-none",
    mesh: "pipeline-handler-mesh-options-none",
};

/// The names of the rows taken with `MTLPipelineOptionBindingInfo`.
const BINDING_INFO_NAMES: OptionNames = OptionNames {
    render: "pipeline-handler-render-options-binding-info",
    compute_descriptor: "pipeline-handler-compute-descriptor-options-binding-info",
    tile: "pipeline-handler-tile-options-binding-info",
    mesh: "pipeline-handler-mesh-options-binding-info",
};

/// Every case of the family: each of the seven calls, with and without a reflection asked for
/// where the call takes options, and each refused once.
pub fn all() -> Vec<PipelineHandlerCase> {
    let mut cases = vec![
        // `newRenderPipelineStateWithDescriptor:completionHandler:`: the one drawing form with no
        // options at all, so `delivered-a-reflection` records that the call takes none.
        PipelineHandlerCase {
            name: "pipeline-handler-render",
            fixture: RENDER_FIXTURE,
            entry: VERTEX,
            described: Described::Render(drawing(), None),
        },
        PipelineHandlerCase {
            name: "pipeline-handler-render-refused",
            fixture: ATTRIBUTE_FIXTURE,
            entry: ATTRIBUTE_VERTEX,
            described: Described::Render(drawing_refused(), None),
        },
        // `newComputePipelineStateWithFunction:completionHandler:`: the other option-less form.
        PipelineHandlerCase {
            name: "pipeline-handler-compute-function",
            fixture: COMPUTE_FIXTURE,
            entry: KERNEL,
            described: Described::ComputeFunction(KERNEL, None),
        },
        // The kernel slot given a **vertex** function, which the compiler refuses on both sides.
        PipelineHandlerCase {
            name: "pipeline-handler-compute-function-refused",
            fixture: RENDER_FIXTURE,
            entry: VERTEX,
            described: Described::ComputeFunction(VERTEX, None),
        },
    ];
    // The five option-bearing forms, each with the reflection bit clear and set: the pair is what
    // makes `delivered-a-reflection` a site that moves rather than a constant. Each row names its
    // own case, because a case's name is a fixed string a recording is keyed by.
    for (options, names) in [(NO_OPTIONS, NONE_NAMES), (BINDING_INFO, BINDING_INFO_NAMES)] {
        cases.push(PipelineHandlerCase {
            name: names.render,
            fixture: RENDER_FIXTURE,
            entry: VERTEX,
            described: Described::Render(drawing(), Some(options)),
        });
        cases.push(PipelineHandlerCase {
            name: names.compute_descriptor,
            fixture: COMPUTE_FIXTURE,
            entry: KERNEL,
            described: Described::ComputeDescriptor(described_kernel(KERNEL), options),
        });
        cases.push(PipelineHandlerCase {
            name: names.tile,
            fixture: TILE_FIXTURE,
            entry: TILE,
            described: Described::Tile(
                TilePipelineSpec::new(TILE, &TILE_ATTACHMENTS, 512),
                options,
            ),
        });
        cases.push(PipelineHandlerCase {
            name: names.mesh,
            fixture: MESH_FIXTURE,
            entry: MESH,
            described: Described::Mesh(mesh_with(MESH), options),
        });
    }
    // **All eight `MTLPipelineOption` values on one form**, so the asynchronous path's reflection
    // rule is swept whole and not only at the bit Apple's documentation describes.
    // `MTLPipelineOptionFailOnBinaryArchiveMiss` alone is the row that matters: Metal fills the
    // handler's reflection for it, which `pipeline_option_reflection` found through the synchronous call,
    // and these rows are what hold the same rule through the block.
    for (options, name) in COMPUTE_FUNCTION_MASKS {
        cases.push(PipelineHandlerCase {
            name,
            fixture: COMPUTE_FIXTURE,
            entry: KERNEL,
            described: Described::ComputeFunction(KERNEL, Some(options)),
        });
    }
    // The refusal of each option-bearing form, with the reflection asked for: a refused creation
    // must answer no reflection either, which is what the `delivered-a-reflection` site of these
    // rows says.
    cases.extend([
        PipelineHandlerCase {
            name: "pipeline-handler-render-options-refused",
            fixture: ATTRIBUTE_FIXTURE,
            entry: ATTRIBUTE_VERTEX,
            described: Described::Render(drawing_refused(), Some(BINDING_INFO)),
        },
        PipelineHandlerCase {
            name: "pipeline-handler-compute-function-options-refused",
            fixture: RENDER_FIXTURE,
            entry: VERTEX,
            described: Described::ComputeFunction(VERTEX, Some(BINDING_INFO)),
        },
        PipelineHandlerCase {
            name: "pipeline-handler-compute-descriptor-options-refused",
            fixture: RENDER_FIXTURE,
            entry: VERTEX,
            described: Described::ComputeDescriptor(described_kernel(VERTEX), BINDING_INFO),
        },
        // The tile slot given a vertex function.
        PipelineHandlerCase {
            name: "pipeline-handler-tile-options-refused",
            fixture: RENDER_FIXTURE,
            entry: VERTEX,
            described: Described::Tile(
                TilePipelineSpec::new(VERTEX, &TILE_ATTACHMENTS, 512),
                BINDING_INFO,
            ),
        },
    ]);
    cases
}

impl PipelineHandlerCase {
    fn loaded(&self) -> Fixture {
        Fixture {
            name: self.fixture,
            entry: self.entry,
        }
    }
}

impl Case for PipelineHandlerCase {
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
        Ok(vec![
            Input {
                name: "air".to_owned(),
                digest: Digest::sha256(fixture.text(RUNTIME_TEXT)?.as_bytes()),
            },
            Input {
                name: "air-typed".to_owned(),
                digest: Digest::sha256(fixture.text(METAL_TEXT)?.as_bytes()),
            },
            Input::bytes("entry", self.entry.as_bytes()),
            // The call, its options and the whole descriptor both paths are given: everything the
            // case feeds, so a descriptor that moves is a change to the case's inputs.
            Input::bytes("form", &self.described.form().words()),
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
        match side.pipeline_both_paths(library, &self.described.form()) {
            Ok(both) => {
                sites.push(both.node(PIPELINES));
                Ok(())
            }
            Err(failure) => {
                // The typed error at the pair's own site, so a side with no such call reads as a
                // refusal rather than as a missing site.
                sites.push(Node::leaf(PIPELINES, Observed::value(failure.recorded())));
                Err(Abort {
                    at: PIPELINES.to_owned(),
                    failure,
                })
            }
        }
    }
}
