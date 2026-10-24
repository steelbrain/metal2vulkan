//! `MTLBinaryArchive`, `MTLBinaryArchiveDescriptor` and
//! `newBinaryArchiveWithDescriptor:error:`, and the one `MTLPipelineOption` bit whose purpose is
//! only observable inside them.
//!
//! # What the suite measured before this
//!
//! Nothing. `MTLBinaryArchive`'s items, its descriptor's two, and the device call that makes one
//! were reached by **no case**, which two other families said outright:
//! [`crate::compute_descriptors`] and [`crate::render_descriptors`] both name the archive list as
//! a descriptor field they leave empty because the container had no case at all.
//!
//! The gap mattered most at one bit. [`crate::cases::pipeline_options`] swept all eight values of
//! the defined `MTLPipelineOption` word — and swept them with **no archive listed**, where
//! `MTLPipelineOptionFailOnBinaryArchiveMiss` changes nothing by construction. That sweep's own
//! conclusion was that it had eight of eight of the enum and nothing of the bit's behaviour, and
//! that the archive family is where the behaviour lives. This family is that.
//!
//! # The four-cell table, as the oracle answered it
//!
//! Measured in a throwaway process on the oracle (Apple M3 Max, macOS 27.0.1) before any case was
//! written, over one archive holding one compute kernel and one render pair:
//!
//! | the bit | an archive listed | the archives hold the pipeline | compute | render |
//! |---|---|---|---|---|
//! | absent | one | yes | created | created |
//! | absent | one | no | created | created |
//! | present | one | yes | created | created |
//! | present | one | **no** | **refused** | **refused** |
//! | present | none | no | created | created |
//! | present | one empty | no | **refused** | — |
//!
//! So the bit is **not** vacuous here, and the pair that proves it is the second row against the
//! fourth: the same library, the same kernel, the same archive, one bit apart, and Metal answers
//! one way and then the other. The fifth row is the whole of what the earlier sweep could only
//! assert — with no archive listed the bit does nothing — now measured rather than quoted.
//!
//! # Which values of the sweep change the observed result, and which cannot
//!
//! - **The option bit changes it**, at the fourth row of the table above, for compute and for
//!   render. That is the falsifying pair, and the `binary-archive-fail-on-miss` case is what holds it.
//! - **Whether an archive is listed changes it**, at the fifth row: the same bit over the same
//!   missing key is refused with an archive listed and created without one.
//! - **Which archives are listed changes it**, at the sixth row: an archive that holds nothing
//!   misses a pipeline the first archive holds.
//! - **A second, empty archive beside the first cannot change it**, and is indistinguishable from
//!   the first alone **by construction**: a lookup that finds a key in any listed archive is a
//!   hit, so adding an archive that holds no key can only leave a hit a hit and a miss a miss.
//!   What the row still buys is the one way a side could get the list wrong — searching only the
//!   last archive, or only the first — which it catches at both ends of the pair.
//! - **A missing file and a file that is not an archive cannot be told apart** by what this
//!   family compares: Metal answers `MTLBinaryArchiveDomain` code 1 for both, and the family
//!   compares no error code, for the reason [`crate::archives`] measures. Both rows are still
//!   here, because the two are different inputs and a side that accepted either is a difference
//!   at the creation's own `refused` leaf.
//!
//! # Depth, and what is left out
//!
//! Of the four `add…PipelineFunctionsWithDescriptor:error:` calls Metal declares, the family
//! compares **four**: compute, render, tile and mesh, each of them twice over, since Metal's own
//! documentation says a descriptor added twice is "silently accepted" and nothing had held it to
//! that.
//!
//! Two further adders are **not** compared, and the reason is the call rather than the container:
//! `addLibraryWithDescriptor:error:` takes an `MTLStitchedLibraryDescriptor` and
//! `addFunctionWithDescriptor:library:error:` a visible or intersection function, and our runtime
//! implements neither function stitching nor a visible-function archive entry. A case for either
//! would compare a refusal of ours at a call this family is not about; both are named here and in
//! the report as reached by nothing.
//!
//! **`serializeToURL:` onto a directory is deliberately not a case.** Measured on the oracle:
//! Metal answered it `OK` and **replaced the directory with the archive file**, destroying the
//! probe's own scratch tree. [`crate::archives`] states why the vocabulary cannot express it, and
//! our side's divergence there — it answers `BinaryArchive(Write)` — is reported rather than cased.
//!
//! # Why this family is safe
//!
//! Every case makes libraries, archives and pipeline states and writes files inside one scratch
//! directory the side removes. No buffer or texture is allocated, no command buffer is made, and
//! nothing is dispatched or drawn, so no case here can fault the device or trip its watchdog. The
//! files are small — one archive and one metallib of one fixture — and no case writes outside the
//! scratch directory, which no step of the vocabulary can name a path outside of.
//!
//! # Exactness
//!
//! Every leaf is a bool, a name or a count. The family declares no tolerance and announces
//! nothing.

use crate::archives::{Added, Bytes, Created, Program, ProgramLibrary, ScratchFile, Source, Step};
use crate::case::{Abort, Case, Failure, Input, MetalSource, ShaderSource, Side};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::transcript::{Digest, Node, Observed, Sites};
use metal2vulkan::metal::PixelFormat;
use metal2vulkan_devtools::{fixtures_dir, llvm_assemble, AssembleError};

/// The site a program's observations sit under, and the site its refusal sits at.
const PROGRAM: &str = "program";

/// `MTLPipelineOptionFailOnBinaryArchiveMiss`, by Metal's own name for the bit.
const FAIL_ON_MISS: u64 = 1 << 2;

/// The compute fixture: [`crate::cases::pipeline_options`]' own, whose seven kernels give the
/// family a kernel to add and six it never adds, so a miss is the **same library** rather than a
/// second one.
const COMPUTE_FIXTURE: &str = "px_samplers";

/// The kernel every case adds, and the one no case adds: the hit and the miss.
const HELD: &str = "px_sample_argument";
const MISSED: &str = "px_const_nearest_edge";

/// Every entry point `px_samplers`' metallib names, with its metallib `TYPE` code (2, a kernel).
///
/// Stated because our side writes its own container and `xcrun metallib` writes Apple's: a
/// container naming fewer functions would answer a shorter `functionNames` than Metal's for a
/// reason that is this list rather than a difference.
const COMPUTE_ENTRIES: &[(&str, u8)] = &[
    ("px_sample_argument", 2),
    ("px_const_nearest_edge", 2),
    ("px_const_nearest_zero", 2),
    ("px_const_nearest_border", 2),
    ("px_const_linear_edge", 2),
    ("px_const_linear_zero", 2),
    ("px_const_linear_border", 2),
];

/// The render fixture, under the owned `reflection` fixtures:
/// [`crate::cases::texture_accesses`]' own, as [`crate::cases::pipeline_options`] takes it.
const RENDER_FIXTURE: &str = "texture_accesses";
const VERTEX: &str = "texture_accesses_vertex";
const FRAGMENT: &str = "texture_accesses_fragment";

/// The tile fixture and its function, which [`crate::cases::tile_pipeline_descriptors`] reads.
const TILE_FIXTURE: &str = "tile";
const TILE: &str = "reflection_tile";

/// The mesh fixture and its three stages, which [`crate::cases::mesh_pipeline_descriptors`] reads.
const MESH_FIXTURE: &str = "mesh_functions";
const OBJECT: &str = "reflection_object";
const MESH: &str = "reflection_mesh";
const MESH_FRAGMENT: &str = "reflection_mesh_fragment";

/// The colour attachment every descriptor here declares, which is
/// [`crate::cases::pipeline_options`]'.
const ATTACHMENT: PixelFormat = PixelFormat::RGBA8Unorm;

/// Which library a program's step index 0 and 1 are.
const COMPUTE_LIBRARY: usize = 0;
const RENDER_LIBRARY: usize = 1;

/// The whole round trip, end to end: the case the family exists for.
///
/// Create an empty archive, read its label before and after `setLabel:`, ask whether its device is
/// the case's, add a compute descriptor, add the **same** descriptor again, add a render
/// descriptor, write the archive to a file, ask whether the file is there and holds bytes, write it
/// again over itself, create a **second** archive from that file, read its label and its device,
/// create from it the pipeline it holds and the one it does not with the bit set, and serialise
/// the reloaded archive again.
const ROUND_TRIP: &[(&str, Step)] = &[
    ("new-archive", Step::NewArchive { from: None }),
    (
        "label-before",
        Step::ArchiveLabel {
            archive: 0,
            set: None,
        },
    ),
    (
        "label-after",
        Step::ArchiveLabel {
            archive: 0,
            set: Some("the case's archive"),
        },
    ),
    ("device", Step::ArchiveDevice { archive: 0 }),
    (
        "add-compute",
        Step::Add {
            archive: 0,
            what: Added::Compute {
                library: COMPUTE_LIBRARY,
                kernel: HELD,
            },
        },
    ),
    (
        "add-compute-again",
        Step::Add {
            archive: 0,
            what: Added::Compute {
                library: COMPUTE_LIBRARY,
                kernel: HELD,
            },
        },
    ),
    (
        "add-render",
        Step::Add {
            archive: 0,
            what: Added::Render {
                library: RENDER_LIBRARY,
                vertex: VERTEX,
                fragment: FRAGMENT,
                attachments: 1,
            },
        },
    ),
    (
        "serialize",
        Step::Serialize {
            archive: 0,
            to: ScratchFile::At("round-trip.metalar"),
        },
    ),
    (
        "file",
        Step::FileState {
            file: ScratchFile::At("round-trip.metalar"),
        },
    ),
    (
        "serialize-over-itself",
        Step::Serialize {
            archive: 0,
            to: ScratchFile::At("round-trip.metalar"),
        },
    ),
    (
        "reload",
        Step::NewArchive {
            from: Some(Source::File(ScratchFile::At("round-trip.metalar"))),
        },
    ),
    (
        "reloaded-label",
        Step::ArchiveLabel {
            archive: 1,
            set: None,
        },
    ),
    ("reloaded-device", Step::ArchiveDevice { archive: 1 }),
    (
        "reloaded-compute-hit",
        Step::Create {
            archives: &[1],
            options: FAIL_ON_MISS,
            what: Created::Compute {
                library: COMPUTE_LIBRARY,
                kernel: HELD,
            },
        },
    ),
    (
        "reloaded-compute-miss",
        Step::Create {
            archives: &[1],
            options: FAIL_ON_MISS,
            what: Created::Compute {
                library: COMPUTE_LIBRARY,
                kernel: MISSED,
            },
        },
    ),
    (
        "reloaded-render-hit",
        Step::Create {
            archives: &[1],
            options: FAIL_ON_MISS,
            what: Created::Render {
                library: RENDER_LIBRARY,
                vertex: VERTEX,
                fragment: FRAGMENT,
                attachments: 1,
            },
        },
    ),
    (
        "serialize-the-reloaded-archive",
        Step::Serialize {
            archive: 1,
            to: ScratchFile::At("again.metalar"),
        },
    ),
    (
        "again",
        Step::FileState {
            file: ScratchFile::At("again.metalar"),
        },
    ),
];

/// The four-cell table of the option bit, and the three rows beside it that make it a measurement
/// rather than four passes.
///
/// One program, so one archive's state is what every row reads: the bit absent and present, the
/// key held and missed, over compute and over render, and then the same with a second empty
/// archive listed, with only the empty archive listed, and with no archive listed at all.
const FOUR_CELL: &[(&str, Step)] = &[
    ("new-archive", Step::NewArchive { from: None }),
    (
        "add-compute",
        Step::Add {
            archive: 0,
            what: Added::Compute {
                library: COMPUTE_LIBRARY,
                kernel: HELD,
            },
        },
    ),
    (
        "add-render",
        Step::Add {
            archive: 0,
            what: Added::Render {
                library: RENDER_LIBRARY,
                vertex: VERTEX,
                fragment: FRAGMENT,
                attachments: 1,
            },
        },
    ),
    ("new-empty-archive", Step::NewArchive { from: None }),
    // The bit absent, over the archive that holds the key and over the key it does not hold.
    ("compute-absent-hit", compute(&[0], 0, HELD)),
    ("compute-absent-miss", compute(&[0], 0, MISSED)),
    // The bit present: the pair that falsifies the sweep.
    ("compute-present-hit", compute(&[0], FAIL_ON_MISS, HELD)),
    ("compute-present-miss", compute(&[0], FAIL_ON_MISS, MISSED)),
    // The same four with a second, empty archive listed after the first.
    ("compute-absent-hit-plus-empty", compute(&[0, 1], 0, HELD)),
    (
        "compute-absent-miss-plus-empty",
        compute(&[0, 1], 0, MISSED),
    ),
    (
        "compute-present-hit-plus-empty",
        compute(&[0, 1], FAIL_ON_MISS, HELD),
    ),
    (
        "compute-present-miss-plus-empty",
        compute(&[0, 1], FAIL_ON_MISS, MISSED),
    ),
    // Only the empty archive listed: the kernel the first archive holds is a miss here.
    ("compute-absent-hit-empty-only", compute(&[1], 0, HELD)),
    (
        "compute-present-hit-empty-only",
        compute(&[1], FAIL_ON_MISS, HELD),
    ),
    // No archive listed: what the earlier sweep could only assert.
    ("compute-absent-miss-no-archive", compute(&[], 0, MISSED)),
    (
        "compute-present-miss-no-archive",
        compute(&[], FAIL_ON_MISS, MISSED),
    ),
    // The render creation is a different selector of Metal's, so the table is run again on it
    // rather than inferred. The miss is the same two functions over a second colour attachment,
    // which both sides' archive keys cover, so every function is a hit and the pipeline is not.
    ("render-absent-hit", render(&[0], 0, 1)),
    ("render-absent-miss", render(&[0], 0, 2)),
    ("render-present-hit", render(&[0], FAIL_ON_MISS, 1)),
    ("render-present-miss", render(&[0], FAIL_ON_MISS, 2)),
    (
        "render-present-miss-plus-empty",
        render(&[0, 1], FAIL_ON_MISS, 2),
    ),
    (
        "render-present-hit-empty-only",
        render(&[1], FAIL_ON_MISS, 1),
    ),
    (
        "render-present-miss-no-archive",
        render(&[], FAIL_ON_MISS, 2),
    ),
];

/// The adders: each of the four Metal has, each twice, and the archive written afterwards.
///
/// Compute, render and tile in one program; the mesh adder has a case of its own, because our side
/// has no counterpart for it and a refusal ends a program at its step.
const ADDERS: &[(&str, Step)] = &[
    ("new-archive", Step::NewArchive { from: None }),
    (
        "add-compute",
        Step::Add {
            archive: 0,
            what: Added::Compute {
                library: COMPUTE_LIBRARY,
                kernel: HELD,
            },
        },
    ),
    (
        "add-compute-again",
        Step::Add {
            archive: 0,
            what: Added::Compute {
                library: COMPUTE_LIBRARY,
                kernel: HELD,
            },
        },
    ),
    (
        "add-render",
        Step::Add {
            archive: 0,
            what: Added::Render {
                library: RENDER_LIBRARY,
                vertex: VERTEX,
                fragment: FRAGMENT,
                attachments: 1,
            },
        },
    ),
    (
        "add-render-again",
        Step::Add {
            archive: 0,
            what: Added::Render {
                library: RENDER_LIBRARY,
                vertex: VERTEX,
                fragment: FRAGMENT,
                attachments: 1,
            },
        },
    ),
    (
        "add-tile",
        Step::Add {
            archive: 0,
            what: Added::Tile {
                library: 2,
                tile: TILE,
            },
        },
    ),
    (
        "add-tile-again",
        Step::Add {
            archive: 0,
            what: Added::Tile {
                library: 2,
                tile: TILE,
            },
        },
    ),
    (
        "serialize",
        Step::Serialize {
            archive: 0,
            to: ScratchFile::At("adders.metalar"),
        },
    ),
    (
        "file",
        Step::FileState {
            file: ScratchFile::At("adders.metalar"),
        },
    ),
];

/// The mesh adder, alone: `addMeshRenderPipelineFunctionsWithDescriptor:error:`.
const MESH_ADDER: &[(&str, Step)] = &[
    ("new-archive", Step::NewArchive { from: None }),
    (
        "add-mesh",
        Step::Add {
            archive: 0,
            what: Added::Mesh {
                library: 2,
                object: OBJECT,
                mesh: MESH,
                fragment: MESH_FRAGMENT,
            },
        },
    ),
    (
        "add-mesh-again",
        Step::Add {
            archive: 0,
            what: Added::Mesh {
                library: 2,
                object: OBJECT,
                mesh: MESH,
                fragment: MESH_FRAGMENT,
            },
        },
    ),
    (
        "serialize",
        Step::Serialize {
            archive: 0,
            to: ScratchFile::At("mesh.metalar"),
        },
    ),
    (
        "file",
        Step::FileState {
            file: ScratchFile::At("mesh.metalar"),
        },
    ),
];

/// What `serializeToURL:error:` answers where it cannot write, and what an archive nothing was
/// added to answers.
const SERIALIZING: &[(&str, Step)] = &[
    ("new-archive", Step::NewArchive { from: None }),
    // An archive nothing was added to: Metal's "Nothing to serialize".
    (
        "serialize-nothing-added",
        Step::Serialize {
            archive: 0,
            to: ScratchFile::At("empty.metalar"),
        },
    ),
    (
        "nothing-added-file",
        Step::FileState {
            file: ScratchFile::At("empty.metalar"),
        },
    ),
    (
        "add-compute",
        Step::Add {
            archive: 0,
            what: Added::Compute {
                library: COMPUTE_LIBRARY,
                kernel: HELD,
            },
        },
    ),
    // A path whose directory does not exist: Metal's "Invalid URL".
    (
        "serialize-into-a-missing-directory",
        Step::Serialize {
            archive: 0,
            to: ScratchFile::InAMissingDirectory("out.metalar"),
        },
    ),
    (
        "missing-directory-file",
        Step::FileState {
            file: ScratchFile::InAMissingDirectory("out.metalar"),
        },
    ),
    // A path that does exist, after the two that do not, so the program says the archive was
    // still serialisable.
    (
        "serialize-into-the-scratch-directory",
        Step::Serialize {
            archive: 0,
            to: ScratchFile::At("written.metalar"),
        },
    ),
    (
        "written-file",
        Step::FileState {
            file: ScratchFile::At("written.metalar"),
        },
    ),
];

/// What `newBinaryArchiveWithDescriptor:error:` answers for each kind of file its descriptor's
/// `url` can name.
///
/// Five sources, which are the five answers a reader would want of the call: a file that is not an
/// archive, an empty file, a path with nothing at it, a directory, and a **metallib**. The last is
/// not a mistake: Metal accepted one as an archive on the oracle, which our side's container
/// reader refuses, and the row is here because that is a refusal of something plain Metal does.
const SOURCES: &[(&str, Step)] = &[
    (
        "write-text",
        Step::Write {
            file: ScratchFile::At("not-an-archive.txt"),
            bytes: Bytes::Text("this is not an archive\n"),
        },
    ),
    (
        "from-a-text-file",
        Step::NewArchive {
            from: Some(Source::File(ScratchFile::At("not-an-archive.txt"))),
        },
    ),
    (
        "write-empty",
        Step::Write {
            file: ScratchFile::At("empty.bin"),
            bytes: Bytes::Empty,
        },
    ),
    (
        "from-an-empty-file",
        Step::NewArchive {
            from: Some(Source::File(ScratchFile::At("empty.bin"))),
        },
    ),
    (
        "from-a-missing-file",
        Step::NewArchive {
            from: Some(Source::File(ScratchFile::At("nothing-is-here.metalar"))),
        },
    ),
    (
        "from-a-directory",
        Step::NewArchive {
            from: Some(Source::TheScratchDirectory),
        },
    ),
    (
        "write-a-metallib",
        Step::Write {
            file: ScratchFile::At("library.metallib"),
            bytes: Bytes::Metallib {
                library: COMPUTE_LIBRARY,
            },
        },
    ),
    (
        "from-a-metallib",
        Step::NewArchive {
            from: Some(Source::File(ScratchFile::At("library.metallib"))),
        },
    ),
];

/// A compute creation step, which the four-cell table writes many of.
const fn compute(archives: &'static [usize], options: u64, kernel: &'static str) -> Step {
    Step::Create {
        archives,
        options,
        what: Created::Compute {
            library: COMPUTE_LIBRARY,
            kernel,
        },
    }
}

/// A render creation step over `attachments` colour attachments.
const fn render(archives: &'static [usize], options: u64, attachments: usize) -> Step {
    Step::Create {
        archives,
        options,
        what: Created::Render {
            library: RENDER_LIBRARY,
            vertex: VERTEX,
            fragment: FRAGMENT,
            attachments,
        },
    }
}

/// Which libraries a case's program needs, beyond the compute and render ones every case makes.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Third {
    /// None: the program names libraries 0 and 1 only.
    None,
    /// The tile fixture, as library 2.
    Tile,
    /// The mesh fixture, as library 2.
    Mesh,
}

/// A case over one archive program.
pub struct BinaryArchiveCase {
    name: &'static str,
    steps: &'static [(&'static str, Step)],
    third: Third,
}

/// Every case of the family.
pub fn all() -> Vec<BinaryArchiveCase> {
    vec![
        BinaryArchiveCase {
            name: "binary-archive-round-trip",
            steps: ROUND_TRIP,
            third: Third::None,
        },
        BinaryArchiveCase {
            name: "binary-archive-fail-on-miss",
            steps: FOUR_CELL,
            third: Third::None,
        },
        BinaryArchiveCase {
            name: "binary-archive-adders",
            steps: ADDERS,
            third: Third::Tile,
        },
        BinaryArchiveCase {
            name: "binary-archive-mesh-adder",
            steps: MESH_ADDER,
            third: Third::Mesh,
        },
        BinaryArchiveCase {
            name: "binary-archive-serializing",
            steps: SERIALIZING,
            third: Third::None,
        },
        BinaryArchiveCase {
            name: "binary-archive-sources",
            steps: SOURCES,
            third: Third::None,
        },
    ]
}

/// A fixture of the owned `reflection` directory, in the dialect `extension` names.
pub(crate) fn reflection_text(fixture: &str, extension: &str) -> Result<String, String> {
    let path = fixtures_dir()
        .join("reflection")
        .join(format!("{fixture}{extension}"));
    std::fs::read_to_string(&path).map_err(|error| format!("{}: {error}", path.display()))
}

/// The AIR bitcode the runtime parses for a fixture of the `reflection` directory.
pub(crate) fn reflection_bitcode(fixture: &str) -> Result<Vec<u8>, String> {
    let path = fixtures_dir()
        .join("reflection")
        .join(format!("{fixture}{RUNTIME_TEXT}"));
    llvm_assemble(&path).map_err(|error| match error {
        AssembleError::Missing(why) => format!("no llvm-as: {why}"),
        AssembleError::Rejected(why) => format!("llvm-as rejected {fixture}: {why}"),
    })
}

/// One of a case's shaders, in hand, from which a [`ShaderSource`] borrows.
pub(crate) struct Held {
    pub bitcode: Vec<u8>,
    pub air_text: String,
    pub entry: &'static str,
}

impl Held {
    /// The shader as a side takes it.
    pub fn source(&self) -> ShaderSource<'_> {
        ShaderSource {
            bitcode: &self.bitcode,
            air_text: &self.air_text,
            entry: self.entry,
        }
    }
}

impl BinaryArchiveCase {
    /// The fixture the case's third library is, where it has one.
    fn third(&self) -> Option<(&'static str, &'static str)> {
        match self.third {
            Third::None => None,
            Third::Tile => Some((TILE_FIXTURE, TILE)),
            Third::Mesh => Some((MESH_FIXTURE, OBJECT)),
        }
    }

    /// Every shader the case's program names, in library order: the directory it lies in under
    /// the public fixtures, the fixture itself, and the entry point the program runs.
    ///
    /// One statement for both [`Case::shaders`] and [`Case::inputs`], neither of which assembles
    /// anything: a case's inputs are the committed texts it names.
    fn named(&self) -> Vec<(&'static str, &'static str, &'static str)> {
        let mut named = vec![
            ("", COMPUTE_FIXTURE, HELD),
            ("reflection/", RENDER_FIXTURE, VERTEX),
        ];
        if let Some((fixture, entry)) = self.third() {
            named.push(("reflection/", fixture, entry));
        }
        named
    }

    /// The committed text of the fixture `fixture` of `directory`, in the dialect `extension`
    /// names.
    fn text(directory: &str, fixture: &'static str, extension: &str) -> Result<String, String> {
        match directory.is_empty() {
            true => Fixture::new(fixture).text(extension),
            false => reflection_text(fixture, extension),
        }
    }

    /// Every shader the case's program names, in library order.
    fn held(&self) -> Result<Vec<Held>, String> {
        let compute = Fixture::new(COMPUTE_FIXTURE);
        let mut held = vec![
            Held {
                bitcode: compute.bitcode()?,
                air_text: compute.text(METAL_TEXT)?,
                entry: HELD,
            },
            Held {
                bitcode: reflection_bitcode(RENDER_FIXTURE)?,
                air_text: reflection_text(RENDER_FIXTURE, METAL_TEXT)?,
                entry: VERTEX,
            },
        ];
        if let Some((fixture, entry)) = self.third() {
            held.push(Held {
                bitcode: reflection_bitcode(fixture)?,
                air_text: reflection_text(fixture, METAL_TEXT)?,
                entry,
            });
        }
        Ok(held)
    }
}

impl Case for BinaryArchiveCase {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let mut shaders = Vec::new();
        for (directory, fixture, entry) in self.named() {
            for extension in [RUNTIME_TEXT, METAL_TEXT] {
                let text = Self::text(directory, fixture, extension)?;
                shaders.push(ShaderRef {
                    fixture: format!("{directory}{fixture}{extension}"),
                    entry: entry.to_owned(),
                    source_sha256: crate::sha256_hex(text.as_bytes()),
                });
            }
        }
        Ok(shaders)
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let named = self.named();
        let mut inputs = Vec::new();
        // Each library's two committed texts, which are what the case states. Never the bitcode
        // `llvm-as` assembles from the `.ll`: that moves with the host's LLVM release, and a
        // fingerprint it was in could only ever match on the host that recorded the case.
        for (at, (directory, fixture, _)) in named.iter().enumerate() {
            for (name, extension) in [("air", RUNTIME_TEXT), ("air-typed", METAL_TEXT)] {
                inputs.push(Input {
                    name: format!("{name}-{at}"),
                    digest: Digest::sha256(Self::text(directory, fixture, extension)?.as_bytes()),
                });
            }
        }
        // The program is the case's whole question: a step that moves is a different question, and
        // a recording of one is no answer for another.
        let entries: Vec<&[(&str, u8)]> = (0..named.len())
            .map(|at| entries_for(at, self.third))
            .collect();
        inputs.push(Input::bytes(
            "program",
            &crate::archives::program_words(ATTACHMENT, &entries, self.steps),
        ));
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let held = self.held().map_err(|why| Abort {
            at: PROGRAM.to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        })?;
        let libraries = self.libraries(&held);
        match side.archives(&self.program(&libraries)) {
            Ok(observed) => {
                let mut node = Node::branch(PROGRAM, Vec::new());
                for ((site, _), observation) in self.steps.iter().zip(&observed) {
                    node = node.with(observation.node(*site));
                }
                sites.push(node);
                Ok(())
            }
            Err(failure) => {
                sites.push(Node::leaf(PROGRAM, Observed::value(failure.recorded())));
                Err(Abort {
                    at: PROGRAM.to_owned(),
                    failure,
                })
            }
        }
    }
}

impl BinaryArchiveCase {
    /// The libraries the case's program names, in library order.
    fn libraries<'a>(&self, held: &'a [Held]) -> Vec<ProgramLibrary<'a>> {
        held.iter()
            .enumerate()
            .map(|(at, shader)| ProgramLibrary {
                source: shader.source(),
                entries: entries_for(at, self.third),
            })
            .collect()
    }

    /// The program the case runs over those libraries.
    fn program<'a>(&self, libraries: &'a [ProgramLibrary<'a>]) -> Program<'a> {
        Program {
            libraries,
            attachment: ATTACHMENT,
            steps: self.steps,
        }
    }
}

/// The entry points of a case's library `at`, as its own metallib names them.
///
/// Only the compute library's are ever written into a container by this family, so the other two
/// name their own entries for completeness and no step reads them.
fn entries_for(at: usize, third: Third) -> &'static [(&'static str, u8)] {
    match (at, third) {
        (0, _) => COMPUTE_ENTRIES,
        (1, _) => &[(VERTEX, 0), (FRAGMENT, 1)],
        (_, Third::Tile) => &[(TILE, 2)],
        (_, Third::Mesh) => &[(OBJECT, 8), (MESH, 7), (MESH_FRAGMENT, 1)],
        (_, Third::None) => &[],
    }
}
