//! The vocabulary two container families share: `MTLBinaryArchive` and `MTLDynamicLibrary`, with
//! `newLibraryWithData:error:` beside them.
//!
//! # Why one vocabulary for both
//!
//! Neither container answers anything a single call can be compared at. An archive's whole
//! observable behaviour is a **sequence**: make one, add pipeline descriptors to it, write it to a
//! file, make a second one from that file, and create pipelines that the second archive holds and
//! does not hold. A dynamic library's is the same shape — make one, read its install name, write
//! it, load it back. So a case here is a short **program** ([`Program`]) of steps over a few
//! objects the side keeps by index, and what it compares is one observation a step
//! ([`Observation`]).
//!
//! Both families need the same four things: a scratch directory, a metallib of the case's shader,
//! a dynamic-library file of it, and a way to ask whether a file is there. One vocabulary is one
//! implementation of those per side rather than two.
//!
//! # What is compared, and what measurement says cannot be
//!
//! Every step's observation is a bool, a name, an enumerator or a count. **No error's text,
//! domain or code is compared**, and that is a measurement rather than a convention. Probed on the
//! oracle (Apple M3 Max, macOS 27.0.1) while this family was written, Metal answers an archive
//! miss from the **driver's** error domain, not from a Metal one:
//!
//! ```text
//! domain=AGXMetalG15X_M1 code=3
//! desc=Unable to find function "px_const_nearest_edge" (compute pipeline) in binary archives.
//! ```
//!
//! A domain named after one GPU family is not a field any other implementation could answer, so
//! the portable question is the one the option bit actually decides: **was the creation refused**.
//! The same holds for the file errors, which do come from `MTLBinaryArchiveDomain` and
//! `MTLDynamicLibraryDomain` but whose codes our side has no counterpart field for; the module's
//! own documentation records what was measured, and [`crate::pipeline_handlers`] states the same
//! choice for a pipeline creation's `NSError`.
//!
//! What that leaves compared is still sharp, because the programs are swept: a refusal is compared
//! at every step of every row, so a side that refuses one row Metal answers, or answers one row
//! Metal refuses, differs at that step and nowhere else.
//!
//! # One shape, whatever happened earlier
//!
//! A step whose object an earlier step failed to make still emits its own observation, in its own
//! shape, carrying the refused form of it. Both sides run the same program, so a side that made
//! the object where the other did not differs at the **creation's** step as well as at the
//! dependent one, which is what a reader wants: the cause is a site of its own.
//!
//! # What no step may do
//!
//! `serializeToURL:` **onto a directory** is not in the vocabulary, and the types will not express
//! it. Measured on the oracle: Metal answered that call `OK` and **replaced the directory with the
//! archive file**, which destroyed the probe's own scratch tree (and the first copy of the probe's
//! source with it). A step that can delete a directory has no place in a suite that runs
//! unattended, so [`Step::Serialize`] and [`Step::SerializeDynamic`] take a [`ScratchFile`], which
//! names a file and cannot name the directory, while a step that only **reads** a path takes a
//! [`Source`], which can. The divergence it would have measured — our side answers
//! `BinaryArchive(Write)` there — is reported rather than cased.

use crate::case::ShaderSource;
use crate::library_facts::LibraryFacts;
use crate::transcript::{Node, NodeName, Observed, Value};

/// A file a step names, inside the side's own scratch directory.
///
/// A name rather than a path: each side has its own scratch directory, and a program that named
/// an absolute path could not run on two hosts. Nothing a case writes can reach outside it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScratchFile {
    /// `<scratch>/<name>`.
    At(&'static str),
    /// `<scratch>/no-such-directory/<name>`: a path whose directory does not exist, which is
    /// what Metal answers "Invalid URL" for.
    InAMissingDirectory(&'static str),
}

impl ScratchFile {
    /// The words a case's input fingerprint covers.
    pub fn words(&self) -> Vec<u8> {
        match self {
            ScratchFile::At(name) => format!("at:{name}").into_bytes(),
            ScratchFile::InAMissingDirectory(name) => format!("missing:{name}").into_bytes(),
        }
    }
}

/// A path a step **reads**: a file, or the scratch directory itself.
///
/// Separate from [`ScratchFile`] so that no step which writes can name a directory: see the
/// module's own documentation for what Metal does to one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    /// A file of the scratch directory.
    File(ScratchFile),
    /// The scratch directory itself, which every creation that takes a URL is asked about.
    TheScratchDirectory,
}

impl Source {
    /// The words a case's input fingerprint covers.
    pub fn words(&self) -> Vec<u8> {
        match self {
            Source::File(file) => file.words(),
            Source::TheScratchDirectory => b"the-scratch-directory".to_vec(),
        }
    }
}

/// Bytes a step writes to a file, or hands to `newLibraryWithData:error:`.
///
/// A container is named by **what it is**, not by its bytes, because the two sides build it from
/// different dialects of the same shader: Metal's side packages the fixture's typed-pointer AIR
/// with `xcrun metallib`, and ours writes the container the devtools writer writes around the
/// bitcode `llvm-as` assembled. Each is a metallib of the same functions, which is what the
/// observations compare.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bytes {
    /// A metallib of the program's library `library`: an executable library.
    Metallib { library: usize },
    /// A dynamic library's file of the program's library `library`, whose install name is
    /// `install_name`.
    ///
    /// The install name is a string the case states, not a path either side chose, so both sides
    /// answer the same one. It names no real file on purpose: what a case here asks is what the
    /// name reads back as, never whether something loads it.
    DynamicLibraryFile {
        library: usize,
        install_name: &'static str,
    },
    /// Text that is no container at all.
    Text(&'static str),
    /// No bytes.
    Empty,
}

impl Bytes {
    /// The words a case's input fingerprint covers.
    pub fn words(&self) -> Vec<u8> {
        match self {
            Bytes::Metallib { library } => format!("metallib:{library}").into_bytes(),
            Bytes::DynamicLibraryFile {
                library,
                install_name,
            } => format!("dylib:{library}:{install_name}").into_bytes(),
            Bytes::Text(text) => format!("text:{text}").into_bytes(),
            Bytes::Empty => b"empty".to_vec(),
        }
    }
}

/// What an **adder** adds: one of the four `add…PipelineFunctionsWithDescriptor:error:` calls
/// `MTLBinaryArchive` has.
///
/// Four variants and four calls. The two further adders Metal declares —
/// `addLibraryWithDescriptor:error:`, which takes a stitched-library descriptor, and
/// `addFunctionWithDescriptor:library:error:`, which takes a visible or intersection function —
/// are not here: neither takes a pipeline descriptor, our runtime implements neither, and a step
/// for one would compare a refusal of ours against Metal's answer at a call this family is not
/// about. They are named as omissions in [`crate::cases::binary_archives`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Added {
    /// `addComputePipelineFunctionsWithDescriptor:error:`.
    Compute {
        library: usize,
        kernel: &'static str,
    },
    /// `addRenderPipelineFunctionsWithDescriptor:error:`, over `attachments` colour attachments
    /// of the program's format.
    Render {
        library: usize,
        vertex: &'static str,
        fragment: &'static str,
        attachments: usize,
    },
    /// `addTileRenderPipelineFunctionsWithDescriptor:error:`, over one colour attachment.
    Tile { library: usize, tile: &'static str },
    /// `addMeshRenderPipelineFunctionsWithDescriptor:error:`, over one colour attachment.
    Mesh {
        library: usize,
        object: &'static str,
        mesh: &'static str,
        fragment: &'static str,
    },
}

impl Added {
    /// The words a case's input fingerprint covers.
    pub fn words(&self) -> Vec<u8> {
        match self {
            Added::Compute { library, kernel } => format!("add-compute:{library}:{kernel}"),
            Added::Render {
                library,
                vertex,
                fragment,
                attachments,
            } => format!("add-render:{library}:{vertex}:{fragment}:{attachments}"),
            Added::Tile { library, tile } => format!("add-tile:{library}:{tile}"),
            Added::Mesh {
                library,
                object,
                mesh,
                fragment,
            } => format!("add-mesh:{library}:{object}:{mesh}:{fragment}"),
        }
        .into_bytes()
    }
}

/// What a **creation** creates: the two pipeline creations that take an option word and a list of
/// archives.
///
/// Compute and render, because those are the two selectors whose `MTLPipelineOption` the archive
/// bit is defined for and which our runtime looks archives up in for both. A tile or mesh state is
/// added to an archive here ([`Added`]) but not created from one: `MTLBinaryArchive`'s own
/// documentation calls a tile add a way to persist its code, and what this family holds to Metal
/// is the bit's effect at the two creations a consumer makes pipelines with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Created {
    /// `newComputePipelineStateWithDescriptor:options:reflection:error:`.
    Compute {
        library: usize,
        kernel: &'static str,
    },
    /// `newRenderPipelineStateWithDescriptor:options:reflection:error:`, over `attachments`
    /// colour attachments of the program's format.
    Render {
        library: usize,
        vertex: &'static str,
        fragment: &'static str,
        attachments: usize,
    },
}

impl Created {
    /// The words a case's input fingerprint covers.
    pub fn words(&self) -> Vec<u8> {
        match self {
            Created::Compute { library, kernel } => format!("compute:{library}:{kernel}"),
            Created::Render {
                library,
                vertex,
                fragment,
                attachments,
            } => format!("render:{library}:{vertex}:{fragment}:{attachments}"),
        }
        .into_bytes()
    }
}

/// Which library a `newDynamicLibrary:error:` step is given.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LibraryAt {
    /// One of the program's own libraries, by its index: an **executable** library, which Metal
    /// refuses as a dynamic one.
    Program(usize),
    /// The library the `n`th [`Step::NewLibraryWithData`] of the program made, counting from 0.
    FromData(usize),
}

impl LibraryAt {
    /// The words a case's input fingerprint covers.
    pub fn words(&self) -> Vec<u8> {
        match self {
            LibraryAt::Program(at) => format!("program-library:{at}"),
            LibraryAt::FromData(at) => format!("library-from-data:{at}"),
        }
        .into_bytes()
    }
}

/// Where a dynamic library comes from: the two calls `MTLDevice` makes one with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DynamicSource {
    /// `newDynamicLibrary:error:` of a library the program holds.
    Library(LibraryAt),
    /// `newDynamicLibraryWithURL:error:` of a path.
    Url(Source),
}

impl DynamicSource {
    /// The words a case's input fingerprint covers.
    pub fn words(&self) -> Vec<u8> {
        match self {
            DynamicSource::Library(at) => {
                let mut out = b"from-library:".to_vec();
                out.extend(at.words());
                out
            }
            DynamicSource::Url(source) => {
                let mut out = b"from-url:".to_vec();
                out.extend(source.words());
                out
            }
        }
    }
}

/// One step of a program.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Step {
    /// `newBinaryArchiveWithDescriptor:error:`, the descriptor's `url` set to `from` (`None`
    /// leaves it nil, which is Metal's empty archive). Each such step appends to the archives the
    /// program holds, in order from 0.
    NewArchive { from: Option<Source> },
    /// `label`, after `setLabel:` where `set` names one.
    ArchiveLabel {
        archive: usize,
        set: Option<&'static str>,
    },
    /// Whether the archive's `device` is the device the case's other objects were made on.
    ArchiveDevice { archive: usize },
    /// One of the four adders.
    Add { archive: usize, what: Added },
    /// `serializeToURL:error:`.
    Serialize { archive: usize, to: ScratchFile },
    /// A pipeline creation under the raw option word `options`, listing the archives `archives`
    /// names in that order (empty lists none).
    Create {
        archives: &'static [usize],
        options: u64,
        what: Created,
    },
    /// Whether `file` is there, and whether it holds bytes. Not a Metal call: it is how a case
    /// says what `serializeToURL:` left behind, which is the one half of serialising that both
    /// implementations can be asked about.
    FileState { file: ScratchFile },
    /// Write `bytes` to `file`, making no Metal call. A step rather than a field of the program so
    /// that a case states **when** the file appears: an archive created before its file is written
    /// is a different question from one created after.
    Write { file: ScratchFile, bytes: Bytes },
    /// `newLibraryWithData:error:` of `bytes`. Each such step appends to the libraries the program
    /// made from data, in order from 0 ([`LibraryAt::FromData`]).
    NewLibraryWithData { bytes: Bytes },
    /// A dynamic-library creation. Each such step appends to the dynamic libraries the program
    /// holds, in order from 0.
    NewDynamicLibrary { from: DynamicSource },
    /// `label` of a dynamic library, after `setLabel:` where `set` names one.
    DynamicLabel {
        dynamic: usize,
        set: Option<&'static str>,
    },
    /// Whether a dynamic library's `device` is the case's.
    DynamicDevice { dynamic: usize },
    /// `serializeToURL:error:` of a dynamic library.
    SerializeDynamic { dynamic: usize, to: ScratchFile },
}

impl Step {
    /// The words a case's input fingerprint covers: the step's whole question, so a case whose
    /// program moves fingerprints apart from the recording taken of the old one.
    pub fn words(&self) -> Vec<u8> {
        let mut out = Vec::new();
        match self {
            Step::NewArchive { from } => {
                out.extend(b"new-archive");
                match from {
                    Some(source) => out.extend(source.words()),
                    None => out.extend(b"nil-url"),
                }
            }
            Step::ArchiveLabel { archive, set } => {
                out.extend(format!("archive-label:{archive}:{}", set.unwrap_or("-")).into_bytes());
            }
            Step::ArchiveDevice { archive } => {
                out.extend(format!("archive-device:{archive}").into_bytes());
            }
            Step::Add { archive, what } => {
                out.extend(format!("add:{archive}:").into_bytes());
                out.extend(what.words());
            }
            Step::Serialize { archive, to } => {
                out.extend(format!("serialize:{archive}:").into_bytes());
                out.extend(to.words());
            }
            Step::Create {
                archives,
                options,
                what,
            } => {
                out.extend(b"create");
                for archive in *archives {
                    out.extend(archive.to_le_bytes());
                }
                out.extend(options.to_le_bytes());
                out.extend(what.words());
            }
            Step::FileState { file } => {
                out.extend(b"file-state");
                out.extend(file.words());
            }
            Step::Write { file, bytes } => {
                out.extend(b"write");
                out.extend(file.words());
                out.extend(bytes.words());
            }
            Step::NewLibraryWithData { bytes } => {
                out.extend(b"new-library-with-data");
                out.extend(bytes.words());
            }
            Step::NewDynamicLibrary { from } => {
                out.extend(b"new-dynamic-library");
                out.extend(from.words());
            }
            Step::DynamicLabel { dynamic, set } => {
                out.extend(format!("dynamic-label:{dynamic}:{}", set.unwrap_or("-")).into_bytes());
            }
            Step::DynamicDevice { dynamic } => {
                out.extend(format!("dynamic-device:{dynamic}").into_bytes());
            }
            Step::SerializeDynamic { dynamic, to } => {
                out.extend(format!("serialize-dynamic:{dynamic}:").into_bytes());
                out.extend(to.words());
            }
        }
        out
    }
}

/// One of the program's libraries, as both sides take it.
pub struct ProgramLibrary<'a> {
    /// The shader, in both dialects ([`ShaderSource`]).
    pub source: ShaderSource<'a>,
    /// Every entry point the library's own metallib names, with the metallib `TYPE` code of each
    /// (0 vertex, 1 fragment, 2 kernel or tile, 7 mesh, 8 object).
    ///
    /// The list is what our side's container writer has to name, because `xcrun metallib` names
    /// the whole module's and a container that named fewer would answer a shorter `functionNames`
    /// than Metal's for a reason that is the case's own construction rather than a difference. A
    /// case states them; `tests::archive_program_libraries_name_every_entry` holds the statement
    /// to the fixture's own text.
    pub entries: &'a [(&'static str, u8)],
}

/// A program a case runs, with the sites its steps are emitted at.
pub struct Program<'a> {
    /// The libraries the steps name by index.
    pub libraries: &'a [ProgramLibrary<'a>],
    /// The raw `MTLPixelFormat` every colour attachment a step declares is of.
    pub attachment: metal2vulkan::metal::PixelFormat,
    /// The steps, each with the transcript site its observation is emitted at.
    pub steps: &'a [(&'static str, Step)],
}

impl Program<'_> {
    /// The words a case's input fingerprint covers: the attachment format and every step.
    pub fn words(&self) -> Vec<u8> {
        let entries: Vec<&[(&str, u8)]> = self.libraries.iter().map(|it| it.entries).collect();
        program_words(self.attachment, &entries, self.steps)
    }
}

/// The words a case's input fingerprint covers for a program of these libraries' entry points,
/// this attachment and these steps: the program's **shape**, which holds no shader bytes.
///
/// A free function beside [`Program::words`] because a case states its input fingerprint before it
/// has assembled anything, and nothing of a program's shape lives in a [`ShaderSource`]. A
/// [`Program`] cannot be built without one, so a case that needed this through [`Program::words`]
/// would have to run `llvm-as` to state a fingerprint the assembler has no part in.
pub fn program_words(
    attachment: metal2vulkan::metal::PixelFormat,
    entries: &[&[(&str, u8)]],
    steps: &[(&'static str, Step)],
) -> Vec<u8> {
    let mut out = attachment.as_raw().to_le_bytes().to_vec();
    for library in entries {
        for (name, code) in *library {
            out.extend(name.as_bytes());
            out.push(*code);
        }
    }
    for (site, step) in steps {
        out.extend(site.as_bytes());
        out.extend(step.words());
    }
    out
}

/// What one step observed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Observation {
    /// A creation, an adder or a serialisation: whether the call was refused.
    ///
    /// A bool and not the error, for the reason the module's own documentation measures: Metal
    /// answers an archive miss from the driver's error domain.
    Made { refused: bool },
    /// A `label`: the string, or none for Metal's nil.
    Label(Option<String>),
    /// A bool a step asks outright, which is whether an object's `device` is the case's.
    Flag(bool),
    /// A file: whether it is there, and whether it holds bytes.
    FileState { exists: bool, nonempty: bool },
    /// A library a `newLibraryWithData:error:` step made: whether it was refused, and where it was
    /// not, the same four facts [`crate::library_facts`] compares of a library, in the same shape.
    Library {
        refused: bool,
        facts: Option<LibraryFacts>,
    },
    /// A dynamic library: whether its creation was refused, and its `installName` where it was
    /// not.
    Dynamic {
        refused: bool,
        install_name: Option<String>,
    },
}

impl Observation {
    /// The observation of a call that was refused, in the shape the step `step` emits.
    ///
    /// Both sides emit one shape for one step whatever happened, so a step whose object an earlier
    /// step failed to make still emits its own site. The match is over the step rather than over
    /// anything the side saw, which is what keeps the two sides' trees the same shape.
    pub fn refused_for(step: &Step) -> Self {
        match step {
            Step::NewArchive { .. }
            | Step::Add { .. }
            | Step::Serialize { .. }
            | Step::Create { .. }
            | Step::Write { .. }
            | Step::SerializeDynamic { .. } => Observation::Made { refused: true },
            Step::ArchiveLabel { .. } | Step::DynamicLabel { .. } => Observation::Label(None),
            Step::ArchiveDevice { .. } | Step::DynamicDevice { .. } => Observation::Flag(false),
            Step::FileState { .. } => Observation::FileState {
                exists: false,
                nonempty: false,
            },
            Step::NewLibraryWithData { .. } => Observation::Library {
                refused: true,
                facts: None,
            },
            Step::NewDynamicLibrary { .. } => Observation::Dynamic {
                refused: true,
                install_name: None,
            },
        }
    }

    /// The observation as the site a case emits it at, so both sides emit one shape.
    pub fn node(&self, name: impl Into<NodeName>) -> Node {
        match self {
            Observation::Made { refused } => Node::branch(name, Vec::new()).with(Node::leaf(
                "refused",
                Observed::value(Value::Bool(*refused)),
            )),
            Observation::Label(label) => Node::branch(name, Vec::new()).with(Node::leaf(
                "label",
                Observed::value(optional(label.as_deref())),
            )),
            Observation::Flag(flag) => Node::branch(name, Vec::new())
                .with(Node::leaf("device", Observed::value(Value::Bool(*flag)))),
            Observation::FileState { exists, nonempty } => Node::branch(name, Vec::new())
                .with(Node::leaf("exists", Observed::value(Value::Bool(*exists))))
                .with(Node::leaf(
                    "nonempty",
                    Observed::value(Value::Bool(*nonempty)),
                )),
            Observation::Library { refused, facts } => {
                let node = Node::branch(name, Vec::new()).with(Node::leaf(
                    "refused",
                    Observed::value(Value::Bool(*refused)),
                ));
                match facts {
                    Some(facts) => node.with(facts.node("library")),
                    None => node.with(Node::leaf(
                        "library",
                        Observed::value(Value::unit_variant("None")),
                    )),
                }
            }
            Observation::Dynamic {
                refused,
                install_name,
            } => Node::branch(name, Vec::new())
                .with(Node::leaf(
                    "refused",
                    Observed::value(Value::Bool(*refused)),
                ))
                .with(Node::leaf(
                    "install_name",
                    Observed::value(optional(install_name.as_deref())),
                )),
        }
    }
}

/// `Some(text)` or `None`, as [`crate::library_facts`] records an install name: a variant, so a
/// side that answered nil where the other answered a name differs at the variant rather than at a
/// string nobody can tell from an empty one.
fn optional(text: Option<&str>) -> Value {
    match text {
        Some(held) => Value::Variant {
            name: "Some".to_owned(),
            payload: Box::new(Value::Lines(vec![held.to_owned()])),
        },
        None => Value::unit_variant("None"),
    }
}

/// A scratch directory a side keeps its program's files in, removed when the side is dropped.
///
/// Shared by both sides rather than written twice: the Metal side builds its metallibs here too.
pub struct Scratch(pub std::path::PathBuf);

impl Scratch {
    /// A fresh directory under the OS temp directory, named for this process.
    pub fn new() -> Result<Self, String> {
        let path = std::env::temp_dir().join(format!(
            "metal2vulkan-conformance-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|since| since.as_nanos())
                .unwrap_or_default()
        ));
        std::fs::create_dir_all(&path).map_err(|error| format!("{}: {error}", path.display()))?;
        Ok(Self(path))
    }

    /// The path a [`ScratchFile`] names. The missing directory is never created, which is what
    /// makes a write to it the "Invalid URL" Metal answers.
    pub fn path(&self, file: ScratchFile) -> std::path::PathBuf {
        match file {
            ScratchFile::At(name) => self.0.join(name),
            ScratchFile::InAMissingDirectory(name) => self.0.join("no-such-directory").join(name),
        }
    }

    /// The path a [`Source`] names.
    pub fn source(&self, source: Source) -> std::path::PathBuf {
        match source {
            Source::File(file) => self.path(file),
            Source::TheScratchDirectory => self.0.clone(),
        }
    }

    /// Whether the file is there, and whether it holds bytes.
    pub fn state(&self, file: ScratchFile) -> Observation {
        match std::fs::metadata(self.path(file)) {
            Ok(metadata) => Observation::FileState {
                exists: true,
                nonempty: metadata.len() > 0,
            },
            Err(_) => Observation::FileState {
                exists: false,
                nonempty: false,
            },
        }
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The container a side writes for [`Bytes::Metallib`] and [`Bytes::DynamicLibraryFile`] out of
/// the AIR bitcode it parses, in the layout Apple's toolchain writes.
///
/// Our side's whole of both: the runtime takes a metallib's bytes
/// (`Device::new_library_with_data`) and has no packager of its own, so the suite's own writer
/// stands in for `xcrun metallib` and for `xcrun metal -dynamiclib`. Metal's side uses Apple's
/// tools instead, which is the same split every case already makes between the two AIR dialects.
pub fn container(
    bitcode: &[u8],
    entries: &[(&'static str, u8)],
    install_name: Option<&str>,
) -> Vec<u8> {
    let functions: Vec<_> = entries
        .iter()
        .map(
            |(name, code)| metal2vulkan_devtools::metallib::MtlbFunction {
                name,
                type_code: *code,
                bitcode: metal2vulkan_devtools::metallib::wrap(bitcode),
                constants: Vec::new(),
                public: Vec::new(),
                tags: Vec::new(),
            },
        )
        .collect();
    let linkage = metal2vulkan_devtools::metallib::MtlbLinkage {
        dynamic: install_name.is_some(),
        install_name,
        dependencies: Vec::new(),
        imports: Vec::new(),
    };
    metal2vulkan_devtools::metallib::metallib_linked(&functions, &linkage)
}
