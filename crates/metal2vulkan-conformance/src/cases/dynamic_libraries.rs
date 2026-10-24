//! `MTLDynamicLibrary`, `newDynamicLibrary:error:`, `newDynamicLibraryWithURL:error:` and
//! `newLibraryWithData:error:`.
//!
//! # What the suite measured before this
//!
//! Nothing of the container. `MTLDynamicLibrary`'s five items were reached by no case — which
//! [`crate::render_descriptors`] says outright of the preloaded-library lists it leaves empty —
//! and `newLibraryWithData:error:` was reached by none either: every library of every case is
//! loaded from a **file** (`newLibraryWithURL:`), so the one call that takes a library's bytes was
//! compared nowhere.
//!
//! # Where the bytes come from, since the two sides package differently
//!
//! Both calls take a container, and neither side can read the other's dialect of AIR. So each
//! side builds the container itself, out of the bitcode it parses: Metal's side with Apple's own
//! `xcrun metallib` and `xcrun metal -dynamiclib`, ours with the suite's writer
//! ([`crate::archives::container`]) around what `llvm-as` assembled. That is the same split every
//! case already makes between a fixture's two committed texts, and what the cases compare is what
//! the **library** then answers, never the bytes.
//!
//! The install name is a string the case states (`/metal2vulkan/conformance/libpx.metallib`), not
//! a path either side chose, so both sides answer the same one and nothing here depends on a file
//! being at it. A case that *loaded* a library by install name would need the file to be there;
//! none does, and that is named below.
//!
//! # Which values change the observed result, and which cannot
//!
//! - **A library's type changes `newDynamicLibrary:`.** The executable library every other case
//!   of the suite uses is refused; the dynamic container of the same shader is accepted. That is
//!   the falsifying pair, and it is the whole of what `MTLLibraryType` is for.
//! - **The install name changes what the library answers.** A metallib written with one answers
//!   that name at `installName`, under `MTLLibraryTypeDynamic`; the same shader's executable
//!   metallib answers nil, under `MTLLibraryTypeExecutable`. Two leaves of one tree move together,
//!   which is what makes the pair readable.
//! - **A URL's kind changes `newDynamicLibraryWithURL:`.** The serialised file loads; a missing
//!   path, a text file, a directory and an **executable** metallib do not.
//! - **A missing path and a directory cannot be told apart** by what this family compares: Metal
//!   answers `NSCocoaErrorDomain` 260 for one and 256 for the other, and the family compares no
//!   error code ([`crate::archives`] measures why). Both rows stay, because they are different
//!   inputs and a side that accepted either differs at the creation's own leaf.
//! - **A dynamic library's label is indistinguishable from an archive's by construction** — both
//!   are nil until set and the string after — and the row is still here, because it is the one
//!   `MTLDynamicLibrary` item that is neither the install name nor the device nor the file.
//!
//! # Depth, and what is left out
//!
//! **A dynamic library with no install name is not a case.** Metal's own toolchain cannot write
//! one: measured on the oracle, `xcrun metal -dynamiclib` with no `-install_name` writes the
//! output file's own basename as the install name, and the library it makes is accepted. So the
//! shape "a dynamic library whose `installName` is nil" cannot be put to Metal through the tools
//! the Metal leg has, and the refusal our runtime documents for it
//! (`Compile(InvalidDescriptor(DynamicLibrary))`) is reached at the same call by the
//! **executable** library's row, which is the same check failing on the other field.
//!
//! **Nothing here links a dynamic library into a pipeline.** `preloadedLibraries` and a render
//! pipeline's two stage lists are a pipeline-descriptor question, which
//! [`crate::compute_descriptors`] and [`crate::render_descriptors`] own, and the loading of an
//! install name at creation time needs a file at the name — which would make the case's answer
//! depend on a path. Both are named as reached by nothing.
//!
//! # Why this family is safe
//!
//! Each case parses containers, makes libraries and writes small files inside one scratch
//! directory the side removes. Nothing is allocated for the device, no command buffer is made and
//! nothing is dispatched. **No case asks a dynamic library for a function**:
//! `newFunctionWithName:` of a dynamic library's export stops at an assertion inside Metal (`TYPE`
//! 3, measured on an M3 Max and an M4), so the vocabulary has no step for it and
//! [`crate::archives`] reads the names off `functionNames` alone.
//!
//! # Exactness
//!
//! Every leaf is a bool, a name, an enumerator or a count. The family declares no tolerance and
//! announces nothing.

use crate::archives::{
    Bytes, DynamicSource, LibraryAt, Program, ProgramLibrary, ScratchFile, Source, Step,
};
use crate::case::{Abort, Case, Failure, Input, MetalSource, ShaderSource, Side};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::transcript::{Digest, Node, Observed, Sites};
use metal2vulkan::metal::PixelFormat;

/// The site a program's observations sit under, and the site its refusal sits at.
const PROGRAM: &str = "program";

/// The fixture both families read: one kernel, no function constant, so its dynamic container is
/// the smallest thing that can be one.
const FIXTURE: &str = "store_const";
const KERNEL: &str = "store_const";

/// Every entry point the fixture's metallib names, with its metallib `TYPE` code (2, a kernel).
const ENTRIES: &[(&str, u8)] = &[(KERNEL, 2)];

/// The install name every dynamic container here is written with.
///
/// A string the case states rather than a path a side chose, so both sides answer the same one. It
/// names no real file: what a case here asks is what the name reads back as.
const INSTALL_NAME: &str = "/metal2vulkan/conformance/libstore_const.metallib";

/// The format a program declares, which no step of this family uses: no case here makes a render
/// pipeline. Stated because [`Program`] names one.
const ATTACHMENT: PixelFormat = PixelFormat::RGBA8Unorm;

/// `newLibraryWithData:error:`, over the four kinds of bytes it can be given.
///
/// An executable container of the fixture, a dynamic container of the same shader, text, and no
/// bytes at all. Each row records the library's own four facts, in the shape
/// [`crate::library_facts`] compares them, so the type and the install name of the first two are
/// read side by side.
const FROM_DATA: &[(&str, Step)] = &[
    (
        "an-executable-container",
        Step::NewLibraryWithData {
            bytes: Bytes::Metallib { library: 0 },
        },
    ),
    (
        "a-dynamic-container",
        Step::NewLibraryWithData {
            bytes: Bytes::DynamicLibraryFile {
                library: 0,
                install_name: INSTALL_NAME,
            },
        },
    ),
    (
        "text",
        Step::NewLibraryWithData {
            bytes: Bytes::Text("not a metallib at all, just some bytes\n"),
        },
    ),
    (
        "no-bytes",
        Step::NewLibraryWithData {
            bytes: Bytes::Empty,
        },
    ),
];

/// `newDynamicLibrary:error:` of the two library types, and everything the dynamic one then
/// answers.
///
/// The executable library first, so the refusal and the acceptance sit beside each other in one
/// transcript: one shader, two containers, one bit of `MTLLibraryType` between them.
const FROM_LIBRARY: &[(&str, Step)] = &[
    (
        "from-an-executable-library",
        Step::NewDynamicLibrary {
            from: DynamicSource::Library(LibraryAt::Program(0)),
        },
    ),
    (
        "a-dynamic-container",
        Step::NewLibraryWithData {
            bytes: Bytes::DynamicLibraryFile {
                library: 0,
                install_name: INSTALL_NAME,
            },
        },
    ),
    (
        "from-a-dynamic-library",
        Step::NewDynamicLibrary {
            from: DynamicSource::Library(LibraryAt::FromData(0)),
        },
    ),
    (
        "label-before",
        Step::DynamicLabel {
            dynamic: 1,
            set: None,
        },
    ),
    (
        "label-after",
        Step::DynamicLabel {
            dynamic: 1,
            set: Some("the case's dynamic library"),
        },
    ),
    ("device", Step::DynamicDevice { dynamic: 1 }),
    (
        "serialize",
        Step::SerializeDynamic {
            dynamic: 1,
            to: ScratchFile::At("library.metallib"),
        },
    ),
    (
        "file",
        Step::FileState {
            file: ScratchFile::At("library.metallib"),
        },
    ),
    (
        "serialize-into-a-missing-directory",
        Step::SerializeDynamic {
            dynamic: 1,
            to: ScratchFile::InAMissingDirectory("library.metallib"),
        },
    ),
    (
        "missing-directory-file",
        Step::FileState {
            file: ScratchFile::InAMissingDirectory("library.metallib"),
        },
    ),
    // The file the serialisation wrote, loaded back: the round trip's other half.
    (
        "from-the-serialized-url",
        Step::NewDynamicLibrary {
            from: DynamicSource::Url(Source::File(ScratchFile::At("library.metallib"))),
        },
    ),
    (
        "reloaded-label",
        Step::DynamicLabel {
            dynamic: 2,
            set: None,
        },
    ),
    ("reloaded-device", Step::DynamicDevice { dynamic: 2 }),
];

/// `newDynamicLibraryWithURL:error:`, over every kind of path it can be given.
///
/// A dynamic container's file, a text file, an executable metallib, a path with nothing at it, and
/// a directory. The first is the one that loads, which is what makes the other four a measurement
/// rather than four refusals in a row.
const FROM_URL: &[(&str, Step)] = &[
    (
        "write-a-dynamic-container",
        Step::Write {
            file: ScratchFile::At("dynamic.metallib"),
            bytes: Bytes::DynamicLibraryFile {
                library: 0,
                install_name: INSTALL_NAME,
            },
        },
    ),
    (
        "from-a-dynamic-container",
        Step::NewDynamicLibrary {
            from: DynamicSource::Url(Source::File(ScratchFile::At("dynamic.metallib"))),
        },
    ),
    (
        "write-text",
        Step::Write {
            file: ScratchFile::At("not-a-library.txt"),
            bytes: Bytes::Text("this is not a library\n"),
        },
    ),
    (
        "from-a-text-file",
        Step::NewDynamicLibrary {
            from: DynamicSource::Url(Source::File(ScratchFile::At("not-a-library.txt"))),
        },
    ),
    (
        "write-an-executable-container",
        Step::Write {
            file: ScratchFile::At("executable.metallib"),
            bytes: Bytes::Metallib { library: 0 },
        },
    ),
    (
        "from-an-executable-container",
        Step::NewDynamicLibrary {
            from: DynamicSource::Url(Source::File(ScratchFile::At("executable.metallib"))),
        },
    ),
    (
        "from-a-missing-file",
        Step::NewDynamicLibrary {
            from: DynamicSource::Url(Source::File(ScratchFile::At("nothing-is-here.metallib"))),
        },
    ),
    (
        "from-a-directory",
        Step::NewDynamicLibrary {
            from: DynamicSource::Url(Source::TheScratchDirectory),
        },
    ),
];

/// A case over one program.
pub struct DynamicLibraryCase {
    name: &'static str,
    steps: &'static [(&'static str, Step)],
}

/// Every case of the family.
pub fn all() -> Vec<DynamicLibraryCase> {
    vec![
        DynamicLibraryCase {
            name: "library-from-data",
            steps: FROM_DATA,
        },
        DynamicLibraryCase {
            name: "dynamic-library-from-a-library",
            steps: FROM_LIBRARY,
        },
        DynamicLibraryCase {
            name: "dynamic-library-from-a-url",
            steps: FROM_URL,
        },
    ]
}

impl Case for DynamicLibraryCase {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let fixture = Fixture::new(FIXTURE);
        let mut shaders = Vec::new();
        for extension in [crate::fixture::RUNTIME_TEXT, METAL_TEXT] {
            let text = fixture.text(extension)?;
            shaders.push(ShaderRef {
                fixture: format!("{FIXTURE}{extension}"),
                entry: KERNEL.to_owned(),
                source_sha256: crate::sha256_hex(text.as_bytes()),
            });
        }
        Ok(shaders)
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let fixture = Fixture::new(FIXTURE);
        // The fixture's two committed texts, which are what the case states. Never the bitcode
        // `llvm-as` assembles from the `.ll`: that moves with the host's LLVM release, and a
        // fingerprint it was in could only ever match on the host that recorded the case.
        Ok(vec![
            Input {
                name: "air".to_owned(),
                digest: Digest::sha256(fixture.text(RUNTIME_TEXT)?.as_bytes()),
            },
            Input {
                name: "air-typed".to_owned(),
                digest: Digest::sha256(fixture.text(METAL_TEXT)?.as_bytes()),
            },
            Input::bytes(
                "program",
                &crate::archives::program_words(ATTACHMENT, &[ENTRIES], self.steps),
            ),
        ])
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let stop = |why: String| Abort {
            at: PROGRAM.to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        };
        let fixture = Fixture::new(FIXTURE);
        let bitcode = fixture.bitcode().map_err(stop)?;
        let air_text = fixture.text(METAL_TEXT).map_err(stop)?;
        let libraries = [ProgramLibrary {
            source: ShaderSource {
                bitcode: &bitcode,
                air_text: &air_text,
                entry: KERNEL,
            },
            entries: ENTRIES,
        }];
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

impl DynamicLibraryCase {
    /// The program the case runs over `libraries`.
    fn program<'a>(&self, libraries: &'a [ProgramLibrary<'a>]) -> Program<'a> {
        Program {
            libraries,
            attachment: ATTACHMENT,
            steps: self.steps,
        }
    }
}
