//! What a **library** says about itself, and the function a descriptor makes of it.
//!
//! `MTLLibrary` holds every function the suite reads — [`crate::cases::function_facts`] reads a
//! function's own getters, [`crate::cases::function_bindings`] its reflected arguments — and the
//! library itself was compared at two items of eleven: `newFunctionWithName:` and
//! `newFunctionWithName:constantValues:error:`, the two calls that hand a function over. Its four
//! own getters were compared nowhere, and `MTLFunctionDescriptor` was a container no case's Metal
//! leg touched at all.
//!
//! The getters are the first thing a consumer asks of a library it was handed: `functionNames` is
//! how it enumerates the entry points before it knows one, `type` is what separates an executable
//! library from a dynamic one (whose every `newFunctionWithName:` is refused), and `installName`
//! is what a dynamic library resolves against. Our side answers all three out of the metallib
//! container — its header's file type, its function list, its install-name record — so what these
//! cases hold to Metal is that reading.
//!
//! # The shapes, and why each is here
//!
//! - **a library of one function** (`a27_ob1_threadgroup`): the floor. `functionNames` is a
//!   one-name list, which is what says the list is the library's functions rather than a constant;
//! - **a library of several functions** (`px_samplers`): the shape that makes `functionNames` a
//!   list at all. Seven specific names and their count compare, so a side that answered one name,
//!   dropped one, or listed a helper function Metal does not is a readable difference rather than
//!   an equal one. The **order** is not compared, and
//!   [`crate::library_facts`] says what was measured: Metal's order is its own, our side's is the
//!   metallib container's, and the two disagree over the same bytes;
//! - **a function made by descriptor** (`newFunctionWithDescriptor:error:` naming
//!   `ob1_threadgroup`): the facts `newFunctionWithName:` gives, through the other call. The same
//!   `FunctionFacts` tree [`crate::cases::function_facts`] records, so the two are read against
//!   each other directly;
//! - **a descriptor carrying constant values** (`fc_vertex_input_to_output` with `has_uv` set):
//!   the descriptor's `constantValues` doing what
//!   `newFunctionWithName:constantValues:error:` does — the gated attribute present and the
//!   constants dictionary empty. `function-facts-specialized-by-index` is the same specialization
//!   through the other call, so a descriptor that dropped its values reads as that case's
//!   unspecialized answer;
//! - **a descriptor carrying an option** (`MTLFunctionOptionStoreFunctionInMetalPipelinesScript`,
//!   then `MTLFunctionOptionCompileToBinary`, `MTLFunctionOptionFailOnBinaryArchiveMiss`,
//!   `MTLFunctionOptionPipelineIndependent`, then all four at once): every `MTLFunctionOptions`
//!   bit a plain function may be made with, one case a bit and one case for the whole word. No
//!   other case of the suite leaves `options` anything but `MTLFunctionOptionNone`, so a side that
//!   ignored a descriptor's options reads as `None` here, and one that masked a bit it has no use
//!   for reads as a smaller set. Plain Metal takes all sixteen combinations over a descriptor that
//!   lists no binary archive and answers the word back verbatim (an M3 Max, macOS 27.0.1,
//!   `probes/argument_descriptor_types.m`).
//!
//! # Why the library getters are not an empty comparison
//!
//! Three of the four would compare equal between two sides that answered nothing: an empty name
//! list, and a nil install name. So the family's positive control is the **names themselves** —
//! `px_samplers` answers seven specific function names and `a27_ob1_threadgroup` exactly one, and
//! both are values no other case of the suite leaves at these sites. The install name is nil here
//! because an executable library has none, and the case that pins a non-nil one needs a dynamic
//! library, which this family does not make ([`crate::cases`] says what a dynamic-library case
//! would take).
//!
//! # Why this family is safe
//!
//! Each case makes one library and reads its getters, or makes one function of it and reads that
//! function's getters. Nothing is compiled into a pipeline, nothing is allocated for the device,
//! no command buffer is made and nothing is dispatched, so no case here can fault the device or
//! trip its watchdog.
//!
//! # Exactness
//!
//! Every field is a number, an enumeration, a name or a bool, and the family declares no tolerance
//! and announces nothing. An enumeration is recorded as its Metal enumerator name beside its
//! ordinal, so a difference reads `MTLLibraryTypeDynamic` against `MTLLibraryTypeExecutable`
//! rather than 1 against 0.

use crate::case::{Abort, Case, Failure, Input, MetalSource, Side};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::function_facts::ConstantValue;
use crate::library_facts::FunctionDescriptorSpec;
use crate::recording::ShaderRef;
use crate::transcript::{Digest, Node, Observed, Sites};
use metal2vulkan::metal::FunctionOptions;

/// The site the library's make sits at.
const LIBRARY: &str = "library";

/// The site the library's own answers sit at.
const FACTS: &str = "facts";

/// What a case reads.
enum Reads {
    /// The library's own getters.
    Library,
    /// A function the descriptor makes, and its own getters.
    Descriptor(FunctionDescriptorSpec),
}

/// A case over one library's own facts, or over a function a descriptor makes of it.
pub struct LibraryFactsCase {
    name: &'static str,
    /// The fixture the library is made from, and the entry point the case names of it.
    fixture: &'static str,
    entry: &'static str,
    reads: Reads,
}

/// Every case of the family.
pub fn all() -> Vec<LibraryFactsCase> {
    vec![
        // A library of one kernel: a one-name `functionNames`, an executable type and no install
        // name.
        LibraryFactsCase {
            name: "library-facts-one-function",
            fixture: "a27_ob1_threadgroup",
            entry: "ob1_threadgroup",
            reads: Reads::Library,
        },
        // A library of several functions, which is where the order and the length of
        // `functionNames` are facts rather than one fixture's accident.
        LibraryFactsCase {
            name: "library-facts-many-functions",
            fixture: "px_samplers",
            entry: "px_sample_argument",
            reads: Reads::Library,
        },
        // `newFunctionWithDescriptor:error:` of a name alone: the facts the by-name call gives,
        // through the other call, with `options` at `MTLFunctionOptionNone`.
        LibraryFactsCase {
            name: "library-facts-descriptor",
            fixture: "a27_ob1_threadgroup",
            entry: "ob1_threadgroup",
            reads: Reads::Descriptor(FunctionDescriptorSpec {
                name: "ob1_threadgroup".to_owned(),
                constants: Vec::new(),
                options: FunctionOptions::NONE,
            }),
        },
        // A descriptor carrying the same constant value `function-facts-specialized-by-index`
        // supplies through `newFunctionWithName:constantValues:error:`: `has_uv` is
        // `[[function_constant(5)]]`, which gates the `uv` attribute at index 1.
        LibraryFactsCase {
            name: "library-facts-descriptor-constants",
            fixture: "function_constant_interface",
            entry: "fc_vertex_input_to_output",
            reads: Reads::Descriptor(FunctionDescriptorSpec {
                name: "fc_vertex_input_to_output".to_owned(),
                constants: vec![ConstantValue::bool_at(5, true)],
                options: FunctionOptions::NONE,
            }),
        },
        // A descriptor carrying an option, which is the one site of the suite where `options`
        // holds anything but `MTLFunctionOptionNone`.
        LibraryFactsCase {
            name: "library-facts-descriptor-options",
            fixture: "a27_ob1_threadgroup",
            entry: "ob1_threadgroup",
            reads: Reads::Descriptor(FunctionDescriptorSpec {
                name: "ob1_threadgroup".to_owned(),
                constants: Vec::new(),
                options: FunctionOptions::STORE_FUNCTION_IN_METAL_PIPELINES_SCRIPT,
            }),
        },
        // The two option bits the runtime used to refuse. Plain Metal takes each on its own over
        // a descriptor that lists no binary archive and answers `options` back verbatim, measured
        // on an Apple M3 Max under macOS 27.0.1 (`probes/argument_descriptor_types.m`, the
        // `function` shape, every one of the sixteen combinations of the four bits, and these
        // rows' own recordings). Each is here on its own so the per-value axis reads the bit back
        // from a site rather than from a combination, and a side that masked the bit out of
        // `options` reads as `None`.
        //
        // `MTLFunctionOptionFailOnBinaryArchiveMiss` asks the creation to fail where the
        // descriptor's binary archives hold no such function; with no archive listed there is
        // nothing to miss, which is what makes the bit askable of one library on its own.
        LibraryFactsCase {
            name: "library-facts-descriptor-fail-on-binary-archive-miss",
            fixture: "a27_ob1_threadgroup",
            entry: "ob1_threadgroup",
            reads: Reads::Descriptor(FunctionDescriptorSpec {
                name: "ob1_threadgroup".to_owned(),
                constants: Vec::new(),
                options: FunctionOptions::FAIL_ON_BINARY_ARCHIVE_MISS,
            }),
        },
        // `MTLFunctionOptionPipelineIndependent`, which asks that a compiled binary function not
        // be specialized to one pipeline. What a case can ask of it without an archive is the
        // same as of the bit above: that the creation succeeds and the word comes back. This is
        // the one row of the family where Metal's **own API validation** disagrees with plain
        // Metal: it stops the creation ("MTLFunctionOptionPipelineIndependent requires
        // MTLFunctionOptionCompileToBinary") where plain Metal, the oracle, answers the call.
        LibraryFactsCase {
            name: "library-facts-descriptor-pipeline-independent",
            fixture: "a27_ob1_threadgroup",
            entry: "ob1_threadgroup",
            reads: Reads::Descriptor(FunctionDescriptorSpec {
                name: "ob1_threadgroup".to_owned(),
                constants: Vec::new(),
                options: FunctionOptions::PIPELINE_INDEPENDENT,
            }),
        },
        // All four bits at once, which no single-bit case decides: the bits are not exclusive, and
        // a side that answered `options` from one state field, or dropped a bit it has no use for
        // once another is set, reads as a smaller set here. 0xf is one of the sixteen combinations
        // the probe put to Metal, and it is also the combination Metal's own validation accepts
        // where it refuses `PIPELINE_INDEPENDENT` alone.
        LibraryFactsCase {
            name: "library-facts-descriptor-every-option",
            fixture: "a27_ob1_threadgroup",
            entry: "ob1_threadgroup",
            reads: Reads::Descriptor(FunctionDescriptorSpec {
                name: "ob1_threadgroup".to_owned(),
                constants: Vec::new(),
                options: FunctionOptions::COMPILE_TO_BINARY
                    | FunctionOptions::STORE_FUNCTION_IN_METAL_PIPELINES_SCRIPT
                    | FunctionOptions::FAIL_ON_BINARY_ARCHIVE_MISS
                    | FunctionOptions::PIPELINE_INDEPENDENT,
            }),
        },
        // `MTLFunctionOptionCompileToBinary`, with **no** binary archive listed on the descriptor.
        // The bit is what makes a function eligible for a `MTLBinaryArchive`, and what a case can
        // ask of it on its own is whether the creation still succeeds and still answers the word
        // back: plain Metal takes every combination of the four bits here and answers `options`
        // verbatim, measured on an Apple M3 Max under macOS 27.0.1
        // (`probes/argument_descriptor_types.m`, the `function` shape). What the bit does *with*
        // an archive is `MTLBinaryArchive`'s question, which `cases/binary_archives` is about.
        LibraryFactsCase {
            name: "library-facts-descriptor-compile-to-binary",
            fixture: "a27_ob1_threadgroup",
            entry: "ob1_threadgroup",
            reads: Reads::Descriptor(FunctionDescriptorSpec {
                name: "ob1_threadgroup".to_owned(),
                constants: Vec::new(),
                options: FunctionOptions::COMPILE_TO_BINARY,
            }),
        },
    ]
}

impl LibraryFactsCase {
    fn loaded(&self) -> Fixture {
        Fixture {
            name: self.fixture,
            entry: self.entry,
        }
    }
}

impl Case for LibraryFactsCase {
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
        // The library's own facts are facts about the whole library, so a library case feeds the
        // fixture and nothing else; a descriptor case feeds the descriptor it sets, which is the
        // whole of what decides the function it makes.
        let read = match &self.reads {
            Reads::Library => b"library".to_vec(),
            Reads::Descriptor(descriptor) => {
                let mut out = b"descriptor".to_vec();
                out.extend(descriptor.words());
                out
            }
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
            Input::bytes("entry", self.entry.as_bytes()),
            Input::bytes("reads", &read),
        ])
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let stop = |at: &str, why: String| Abort {
            at: at.to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        };
        let loaded = self.loaded().loaded().map_err(|why| stop(LIBRARY, why))?;
        let library = crate::case::made(sites, LIBRARY, side.library(&loaded.source()))?;
        let answer = match &self.reads {
            Reads::Library => side.library_facts(library).map(|facts| facts.node(FACTS)),
            Reads::Descriptor(descriptor) => side
                .function_facts_from_descriptor(library, descriptor)
                .map(|facts| facts.node(FACTS)),
        };
        match answer {
            Ok(node) => {
                sites.push(node);
                Ok(())
            }
            Err(failure) => {
                sites.push(Node::leaf(FACTS, Observed::value(failure.recorded())));
                Err(Abort {
                    at: FACTS.to_owned(),
                    failure,
                })
            }
        }
    }
}
