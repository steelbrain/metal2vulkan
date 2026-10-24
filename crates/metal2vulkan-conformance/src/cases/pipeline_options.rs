//! `MTLPipelineOption`, swept: what each value of the option word makes a pipeline creation
//! answer for its reflection.
//!
//! # What the suite measured before this
//!
//! The option is the argument that decides whether a creation reflects at all, and the suite
//! passed **two** of its values anywhere in its Metal leg: `MTLPipelineOptionNone`, at the six
//! call sites that want no reflection, and `MTLPipelineOptionBindingInfo`, at the two that do.
//! Nothing compared the option itself, so every claim about the rest of the word — the three bits
//! Metal defines and the ones it does not — rested on the runtime's prose
//! ([`crate::pipeline_options`] lists the four). This family is the sweep that holds them to
//! Metal.
//!
//! The option word is three defined bits, so its defined space is **eight values and the family
//! compares all eight**, on a compute creation and again on a render creation, because the two are
//! different selectors and nothing says Metal reads the word the same way in both. Beyond them the
//! family compares four undefined shapes on the compute creation and one on the render creation.
//!
//! # Which values change what is observed, and which cannot
//!
//! The honest statement of a sweep is which of its values move the answer. Here two do and the
//! rest are expected not to, each for a stated reason:
//!
//! - **0 against 1 is the family's positive control.** The same library, the same function, the
//!   same creation: at 0 the `reflection:` out-parameter stays nil and every stage's site records
//!   `None`; at 1 it comes back and the sites hold the whole binding tree — three buffers for the
//!   kernel, one of them an argument buffer whose struct members are a texture reference and a
//!   sampler, and for the render pipeline nine texture bindings across two stages at three
//!   different accesses, one of them an array of two and one never referenced. A side that wired
//!   `reflects` to the wrong bit answers the other way round at both ends of that pair.
//! - **2 against 0 moves it too**, which is the whole of the claim that `BUFFER_TYPE_INFO` alone
//!   reflects rather than needing `BINDING_INFO` beside it.
//! - **3, 5, 6 and 7 are expected to answer exactly as 1 does**, because the claim under test is
//!   that `BUFFER_TYPE_INFO` adds nothing over `BINDING_INFO` and that the archive bit adds
//!   nothing when no archive is listed. They are not indistinguishable *by construction*: a side
//!   whose reflection carried struct and pointer types under one bit and not the other, or that
//!   refused the combination, reads differently at these values and at no other.
//! - **4 is expected to answer exactly as 0 does**, and that is indistinguishable from 0 **by
//!   construction** in this family: no case lists a binary archive, so the bit has nothing to miss.
//!   What the value still buys is that the creation *succeeds* — a side that refused the bit
//!   outright, or that reflected under it, is a difference.
//! - **8, `1 << 31` and `1 << 40` are expected to answer exactly as 0 does, and 9 exactly as 1
//!   does.** Also indistinguishable by construction, and deliberately: the claim is that Metal
//!   keeps every bit of the word and acts only on the three it defines, so a side that refused an
//!   undefined bit, that let one suppress a reflection, or that read `1 << 40` as `1 << 8` by
//!   truncating the word to 32 bits, differs here. The three are the runtime's own named
//!   values, one inside the low byte, one at the top of 32 bits and one past them.
//!
//! So of twelve compute values the sweep expects **two** distinct answers, and says so rather than
//! implying twelve independent facts. The value of the ten that repeat an answer is that they are
//! the ten ways a side could get the option word wrong.
//!
//! # The two fixtures, and why each
//!
//! Both are chosen so that the reflection a case compares is **not** an empty list.
//!
//! - **the compute sweep** reads `px_samplers`' `px_sample_argument`, which
//!   [`crate::cases::function_bindings`] already compares at `BINDING_INFO`: two plain device
//!   pointers, whose binding answers a `bufferPointerType`, and an argument buffer at slot 2 whose
//!   `bufferStructType` holds a texture reference and a sampler. That is the shape where
//!   `BUFFER_TYPE_INFO`'s own name says something could differ, and it is the shape the family
//!   sweeps for exactly that reason;
//! - **the render sweep** reads the owned `reflection/texture_accesses` fixture, which
//!   [`crate::cases::texture_accesses`] makes the same pipeline out of. Its vertex function takes
//!   one `access::read` texture and its fragment function takes a read, a write, a read-write, an
//!   array of two writes, a write it never references and a sampled texture. So the two stage
//!   lists are non-empty, different from each other, and between them carry all three
//!   `MTLBindingAccess` values, an `arrayLength` above one and an `isUsed` that is false — and a
//!   side that answered the vertex stage's list where the fragment stage's belongs reads wrong
//!   rather than passing.
//!
//! **Three of a render reflection's five stage lists are left out.** A drawing pipeline's
//! `tileBindings`, `objectBindings` and `meshBindings` are empty on both sides by construction, so
//! a site for one would compare an empty list against an empty list and pin nothing. All three are
//! already reached by [`crate::cases::texture_accesses`], which asks them per slot.
//!
//! # Why this family is safe
//!
//! Each case makes one library and one pipeline state and reads what its creation answered.
//! Nothing is allocated for the device, no command buffer is made and nothing is drawn or
//! dispatched, so no case here can fault the device or trip its watchdog. The pipeline creation is
//! the one-time translation cost the project's ceilings already cover.
//!
//! # Exactness
//!
//! Every field is a number, an enumeration, a name or a bool, and the family declares no tolerance
//! and announces nothing: no case touches a texture object, a sampler object or a float.

use crate::case::{Abort, Case, Failure, Input, MetalSource, ShaderSource, Side};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::pipeline_options::OptionValue;
use crate::recording::ShaderRef;
use crate::transcript::{Digest, Node, Observed, Sites};
use metal2vulkan::metal::PixelFormat;
use metal2vulkan_devtools::{fixtures_dir, llvm_assemble, AssembleError};
use metal2vulkan_runtime::PipelineOption;

/// The site the library's make sits at.
const LIBRARY: &str = "library";

/// The site the creation's answer sits at.
const OUTCOME: &str = "outcome";

/// The compute sweep's fixture and kernel: `function_bindings`' argument-buffer kernel, so this
/// family needs no fixture of its own and its reflection is directly comparable with that
/// family's.
const COMPUTE_FIXTURE: &str = "px_samplers";
const KERNEL: &str = "px_sample_argument";

/// The render sweep's fixture, under the owned `reflection` fixtures, and its two functions:
/// `texture_accesses`' own, whose pipeline that family already creates.
const RENDER_FIXTURE: &str = "texture_accesses";
const VERTEX: &str = "texture_accesses_vertex";
const FRAGMENT: &str = "texture_accesses_fragment";

/// The colour attachment the render sweep's pipeline declares, which is
/// [`crate::cases::texture_accesses`]'.
const ATTACHMENT: PixelFormat = PixelFormat::RGBA8Unorm;

/// The three bits Metal defines and the empty word, each taken from the runtime's own value for it
/// rather than written as a literal.
///
/// Through [`PipelineOption`] and not `1 << 0`, because a raw ordinal is a value no reading of the
/// text can attribute: `docs/CONFORMANCE-COVERAGE.md`'s option-space axis read this family as
/// reaching **none** of `MTLPipelineOption`'s four values while it was sweeping all eight
/// combinations of them, for exactly that reason. Naming each value also ties the sweep to the
/// constant the runtime acts on, so a bit that moved in one place and not the other would not
/// compile.
const NONE: u64 = PipelineOption::NONE.raw();
const BINDING_INFO: u64 = PipelineOption::BINDING_INFO.raw();
const BUFFER_TYPE_INFO: u64 = PipelineOption::BUFFER_TYPE_INFO.raw();
const FAIL_ON_BINARY_ARCHIVE_MISS: u64 = PipelineOption::FAIL_ON_BINARY_ARCHIVE_MISS.raw();

/// The whole of the defined option space, every combination of the three bits in order of the
/// raw value, as the **compute** sweep's cases, then the four undefined shapes.
///
/// Eight of eight defined values. A value is here because the word can hold it, not because
/// somebody expected it to matter. The undefined three — a bit inside the low byte, the top bit of
/// 32, and a bit past 32 — are three of the five raw values the runtime's own documentation says
/// it measured Metal accepting; `1 << 10` and 16 are left out as the same question as 8 asked
/// twice more. The fourth says an undefined bit does not suppress a defined one.
const COMPUTE: [(&str, OptionValue); 12] = [
    (
        "pipeline-options-compute-none",
        OptionValue::new(NONE, "MTLPipelineOptionNone"),
    ),
    (
        "pipeline-options-compute-binding-info",
        OptionValue::new(BINDING_INFO, "MTLPipelineOptionBindingInfo"),
    ),
    (
        "pipeline-options-compute-buffer-type-info",
        OptionValue::new(BUFFER_TYPE_INFO, "MTLPipelineOptionBufferTypeInfo"),
    ),
    (
        "pipeline-options-compute-binding-and-buffer-type",
        OptionValue::new(
            BINDING_INFO | BUFFER_TYPE_INFO,
            "BindingInfo|BufferTypeInfo",
        ),
    ),
    (
        "pipeline-options-compute-archive-miss",
        OptionValue::new(
            FAIL_ON_BINARY_ARCHIVE_MISS,
            "MTLPipelineOptionFailOnBinaryArchiveMiss",
        ),
    ),
    (
        "pipeline-options-compute-binding-info-and-archive-miss",
        OptionValue::new(
            BINDING_INFO | FAIL_ON_BINARY_ARCHIVE_MISS,
            "BindingInfo|FailOnBinaryArchiveMiss",
        ),
    ),
    (
        "pipeline-options-compute-buffer-type-and-archive-miss",
        OptionValue::new(
            BUFFER_TYPE_INFO | FAIL_ON_BINARY_ARCHIVE_MISS,
            "BufferTypeInfo|FailOnBinaryArchiveMiss",
        ),
    ),
    (
        "pipeline-options-compute-every-defined-bit",
        OptionValue::new(
            BINDING_INFO | BUFFER_TYPE_INFO | FAIL_ON_BINARY_ARCHIVE_MISS,
            "every defined bit",
        ),
    ),
    (
        "pipeline-options-compute-undefined-bit-3",
        OptionValue::new(1 << 3, "an undefined bit, 1 << 3"),
    ),
    (
        "pipeline-options-compute-undefined-bit-31",
        OptionValue::new(1 << 31, "an undefined bit, 1 << 31"),
    ),
    (
        "pipeline-options-compute-undefined-bit-40",
        OptionValue::new(1 << 40, "an undefined bit past 32, 1 << 40"),
    ),
    (
        "pipeline-options-compute-binding-info-and-undefined-bit",
        OptionValue::new(BINDING_INFO | (1 << 3), "BindingInfo | an undefined bit"),
    ),
];

/// The **render** sweep: the same eight defined values, and the one undefined bit past 32.
///
/// The render creation is a different selector of Metal's, so the defined space is swept again
/// rather than inferred from the compute sweep. One undefined value rather than four: what a
/// second selector can get wrong about an undefined bit is reading the word at the wrong width,
/// and `1 << 40` is the value that says so.
const RENDER: [(&str, OptionValue); 9] = [
    (
        "pipeline-options-render-none",
        OptionValue::new(NONE, "MTLPipelineOptionNone"),
    ),
    (
        "pipeline-options-render-binding-info",
        OptionValue::new(BINDING_INFO, "MTLPipelineOptionBindingInfo"),
    ),
    (
        "pipeline-options-render-buffer-type-info",
        OptionValue::new(BUFFER_TYPE_INFO, "MTLPipelineOptionBufferTypeInfo"),
    ),
    (
        "pipeline-options-render-binding-and-buffer-type",
        OptionValue::new(
            BINDING_INFO | BUFFER_TYPE_INFO,
            "BindingInfo|BufferTypeInfo",
        ),
    ),
    (
        "pipeline-options-render-archive-miss",
        OptionValue::new(
            FAIL_ON_BINARY_ARCHIVE_MISS,
            "MTLPipelineOptionFailOnBinaryArchiveMiss",
        ),
    ),
    (
        "pipeline-options-render-binding-info-and-archive-miss",
        OptionValue::new(
            BINDING_INFO | FAIL_ON_BINARY_ARCHIVE_MISS,
            "BindingInfo|FailOnBinaryArchiveMiss",
        ),
    ),
    (
        "pipeline-options-render-buffer-type-and-archive-miss",
        OptionValue::new(
            BUFFER_TYPE_INFO | FAIL_ON_BINARY_ARCHIVE_MISS,
            "BufferTypeInfo|FailOnBinaryArchiveMiss",
        ),
    ),
    (
        "pipeline-options-render-every-defined-bit",
        OptionValue::new(
            BINDING_INFO | BUFFER_TYPE_INFO | FAIL_ON_BINARY_ARCHIVE_MISS,
            "every defined bit",
        ),
    ),
    (
        "pipeline-options-render-undefined-bit-40",
        OptionValue::new(1 << 40, "an undefined bit past 32, 1 << 40"),
    ),
];

/// Which creation a case sweeps.
enum Subject {
    /// `newComputePipelineStateWithFunction:options:reflection:error:`.
    Compute,
    /// `newRenderPipelineStateWithDescriptor:options:reflection:error:`.
    Render,
}

/// A case over one creation under one option value.
pub struct PipelineOptionsCase {
    name: &'static str,
    option: OptionValue,
    subject: Subject,
}

/// Every case of the family.
pub fn all() -> Vec<PipelineOptionsCase> {
    let compute = COMPUTE
        .into_iter()
        .map(|(name, option)| PipelineOptionsCase {
            name,
            option,
            subject: Subject::Compute,
        });
    let render = RENDER
        .into_iter()
        .map(|(name, option)| PipelineOptionsCase {
            name,
            option,
            subject: Subject::Render,
        });
    compute.chain(render).collect()
}

/// The render fixture's committed text in the dialect `extension` names.
///
/// The owned `reflection` fixtures sit beside the `engine` ones under the public fixtures and are
/// reached through the same devtools manifest, as [`crate::cases::texture_accesses`] reaches this
/// same fixture, so a host without Metal gets them from the git-indexed tree like any other.
fn render_text(extension: &str) -> Result<String, String> {
    let path = fixtures_dir()
        .join("reflection")
        .join(format!("{RENDER_FIXTURE}{extension}"));
    std::fs::read_to_string(&path).map_err(|error| format!("{}: {error}", path.display()))
}

/// The AIR bitcode the runtime parses for the render fixture.
fn render_bitcode() -> Result<Vec<u8>, String> {
    let path = fixtures_dir()
        .join("reflection")
        .join(format!("{RENDER_FIXTURE}{RUNTIME_TEXT}"));
    llvm_assemble(&path).map_err(|error| match error {
        AssembleError::Missing(why) => format!("no llvm-as: {why}"),
        AssembleError::Rejected(why) => format!("llvm-as rejected {RENDER_FIXTURE}: {why}"),
    })
}

impl PipelineOptionsCase {
    /// The compute sweep's fixture.
    fn compute_fixture(&self) -> Fixture {
        Fixture {
            name: COMPUTE_FIXTURE,
            entry: KERNEL,
        }
    }

    /// Both committed texts of whichever fixture the case reads.
    fn texts(&self) -> Result<[String; 2], String> {
        match self.subject {
            Subject::Compute => {
                let fixture = self.compute_fixture();
                Ok([fixture.text(RUNTIME_TEXT)?, fixture.text(METAL_TEXT)?])
            }
            Subject::Render => Ok([render_text(RUNTIME_TEXT)?, render_text(METAL_TEXT)?]),
        }
    }
}

impl Case for PipelineOptionsCase {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let texts = self.texts()?;
        let (prefix, entries): (&str, &[&str]) = match self.subject {
            Subject::Compute => (COMPUTE_FIXTURE, &[KERNEL]),
            Subject::Render => (RENDER_FIXTURE, &[VERTEX, FRAGMENT]),
        };
        let directory = match self.subject {
            Subject::Compute => String::new(),
            Subject::Render => "reflection/".to_owned(),
        };
        let mut shaders = Vec::new();
        for entry in entries {
            for (extension, text) in [RUNTIME_TEXT, METAL_TEXT].into_iter().zip(&texts) {
                shaders.push(ShaderRef {
                    fixture: format!("{directory}{prefix}{extension}"),
                    entry: (*entry).to_owned(),
                    source_sha256: crate::sha256_hex(text.as_bytes()),
                });
            }
        }
        Ok(shaders)
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let [runtime, metal] = self.texts()?;
        let mut inputs = vec![
            Input {
                name: "air".to_owned(),
                digest: Digest::sha256(runtime.as_bytes()),
            },
            Input {
                name: "air-typed".to_owned(),
                digest: Digest::sha256(metal.as_bytes()),
            },
            // The option word is the case's whole question: a different value is a different
            // question, and a recording of one is no answer for another.
            Input::bytes("options", &self.option.words()),
        ];
        if matches!(self.subject, Subject::Render) {
            inputs.push(Input::bytes(
                "attachment",
                &ATTACHMENT.as_raw().to_le_bytes(),
            ));
        }
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let stop = |at: &str, why: String| Abort {
            at: at.to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        };
        let answer = match self.subject {
            Subject::Compute => {
                let loaded = self
                    .compute_fixture()
                    .loaded()
                    .map_err(|why| stop(LIBRARY, why))?;
                let library = crate::case::made(sites, LIBRARY, side.library(&loaded.source()))?;
                side.compute_pipeline_under_options(library, KERNEL, self.option.raw)
            }
            Subject::Render => {
                let bitcode = render_bitcode().map_err(|why| stop(LIBRARY, why))?;
                let air_text = render_text(METAL_TEXT).map_err(|why| stop(LIBRARY, why))?;
                let library = crate::case::made(
                    sites,
                    LIBRARY,
                    side.library(&ShaderSource {
                        bitcode: &bitcode,
                        air_text: &air_text,
                        entry: VERTEX,
                    }),
                )?;
                side.render_pipeline_under_options(
                    library,
                    VERTEX,
                    FRAGMENT,
                    ATTACHMENT,
                    self.option.raw,
                )
            }
        };
        match answer {
            Ok(outcome) => {
                sites.push(outcome.node(OUTCOME));
                Ok(())
            }
            Err(failure) => {
                sites.push(Node::leaf(OUTCOME, Observed::value(failure.recorded())));
                Err(Abort {
                    at: OUTCOME.to_owned(),
                    failure,
                })
            }
        }
    }
}
