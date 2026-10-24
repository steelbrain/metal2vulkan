//! `MTLArgumentEncoder`: what it says of a layout, where it puts each argument, and what the
//! function then reads.
//!
//! The suite could reach an argument buffer before this family and never an argument **encoder**:
//! [`Side::argument_buffer`] writes the resource IDs by hand, as a Metal 3 program may, so the
//! whole of `MTLArgumentEncoder` — 23 implemented items — was compared against Metal nowhere. An
//! argument buffer is how a real Metal program hands a function its resources, and the binding
//! plan is this project's central contract (`docs/API.md` section 5), so a defect in how one is
//! laid out produces plausible-looking wrong output rather than a crash. That is the shape of
//! defect a conformance suite exists to catch.
//!
//! [`Side::argument_buffer`]: crate::case::Side::argument_buffer
//!
//! # What each case compares
//!
//! Three things, and none of them is a word the encoder wrote: a buffer's argument is its Metal
//! address plus an offset and every other kind is a resource ID, which are each host's own numbers
//! ([`crate::argument_encoders`]).
//!
//! - **The layout the encoder states**: `encodedLength`, `alignment`, the `label` it keeps, and
//!   whether `device` is the case's own.
//! - **Where every argument landed**: the case fills the argument buffer with
//!   [`crate::argument_encoders::BLANK`] first and then compares, word by eight-byte word,
//!   which words the encoder wrote. That is the offset, the stride and the alignment of every
//!   argument, with none of the host's numbers in it — and the padding between them, which must
//!   stay as it was. Pre-filling is what makes it a real check: over a buffer created zeroed, an
//!   encoder that wrote nothing and one that wrote a zero word read alike.
//! - **What the function read**, where the case dispatches: the words a kernel wrote after
//!   sampling the texture an encoder put in an argument buffer, through the sampler beside it.
//!
//! `constantDataAtIndex:` answers an address, so what a case compares is its **offset inside the
//! arguments**, which is the layout fact the address stands for.
//!
//! # The two shapes of layout, and why both
//!
//! The function-layout cases take the encoder from the fixture's own kernel
//! (`newArgumentEncoderWithBufferIndex:`), so the layout is the one the AIR declares and the
//! dispatch proves the encoder and the translated function agree. The descriptor cases describe a
//! layout themselves (`newArgumentEncoderWithArguments:`), which is the only way to reach
//! `MTLArgumentDescriptor` at all and the only way to put every argument kind the encoder can
//! write in one layout — a mixed layout of a texture, a sampler, a pointer, two constants of
//! different alignments and an array is where an offset or a stride is wrong by eight bytes.
//!
//! # Why this family is safe
//!
//! Every case writes one 4 by 4 texture, one sampler state and three small buffers, and the two
//! that dispatch run **one threadgroup of sixteen threads** of a loop-free kernel that samples one
//! texel and stores a `float4`. Nothing can fault the device or trip its watchdog.
//!
//! The cases that do **not** dispatch are the ones whose argument buffer a kernel could not read
//! safely: a layout described by descriptors matches no kernel's struct, and the array-element case
//! deliberately puts its arguments 48 bytes into the buffer, where a kernel bound at offset 0 would
//! read the pattern as a resource ID. Each states the layout facts and the word mask and stops
//! there, which is the whole of what it means.
//!
//! # The exact texels, and why they need no tolerance
//!
//! The texture's sixteen texels are the sixteen patterns of `(r, g, b, a)` over `{0, 255}`, one a
//! texel, and each thread samples its own texel's centre through a **nearest**, normalized,
//! clamp-to-edge sampler. So every component the kernel stores is `0.0` or `1.0`, which every
//! implementation converts alike: the comparison is bit-exact float and the family declares no
//! tolerance. An intermediate byte would have put the 8-bit-to-float conversion in the answer,
//! which is a different question and one `pixel_samplers` already asks with the notice it needs.

use crate::allowance::Allowance;
use crate::argument_encoders::{
    changed_node, changed_words, ArgumentDescriptorSpec, ArgumentWrite, ARGUMENT_WORD, BLANK,
};
use crate::case::{
    call, made, Abort, Bindings, Case, Failure, Handle, Input, MetalSource, SamplerSpec, Side,
    TextureRole, TextureSpec,
};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::transcript::{ByteLayout, Digest, Element, Gaps, Node, Observed, Region, Sites, Value};
use metal2vulkan::metal::{DataType, PixelFormat, SamplerAddressMode, SamplerMinMagFilter};
use metal2vulkan_runtime::Degradation;

/// The fixture whose kernel takes a texture and a sampler in an argument buffer. The same one
/// `pixel_samplers` uses for its `arg` source, so this family needs no fixture of its own.
const FIXTURE: &str = "px_samplers";

/// Its argument-buffer kernel: `constant PixelArguments &a [[buffer(2)]]`, holding
/// `texture2d<float> t [[id(0)]]` and `sampler s [[id(1)]]`.
const ENTRY: &str = "px_sample_argument";

/// The buffer index the kernel takes its argument buffer at.
const ARGUMENT_BUFFER_INDEX: u32 = 2;

/// The texture's side: sixteen texels, which is one per pattern of four channels over `{0, 255}`.
const SIDE: u32 = 4;

/// The threads the kernel runs: one a texel, in one threadgroup.
const THREADS: u32 = SIDE * SIDE;

/// Bytes of every buffer the case makes, chosen so an argument buffer of any of these layouts fits
/// at the case's offset with room over, and so the word mask reports the bytes past the arguments
/// as well as the arguments themselves.
const BUFFER_BYTES: usize = 256;

/// The words the mask reports: the whole of a case's argument buffer.
const WORDS: usize = BUFFER_BYTES / ARGUMENT_WORD;

/// The label every case sets and reads back.
const LABEL: &str = "conformance-argument-encoder";

/// The site each call and answer sits at.
mod site {
    pub const LIBRARY: &str = "library";
    pub const PIPELINE: &str = "pipeline";
    pub const TEXTURE: &str = "texture";
    pub const SAMPLER: &str = "sampler";
    pub const COORDINATES: &str = "buffer/coordinates";
    pub const OUTPUT_BUFFER: &str = "buffer/output";
    pub const ARGUMENT_BUFFER: &str = "buffer/arguments";
    pub const ENCODER: &str = "encoder";
    pub const FACTS: &str = "facts";
    pub const SET_ARGUMENT_BUFFER: &str = "setArgumentBuffer";
    pub const READ_BACK: &str = "read-back";
    pub const CHANGED: &str = "changed-words";
    pub const STATUS: &str = "status";
    pub const OUTPUT: &str = "output";
}

/// The resources a case has made, which its writes name.
///
/// A write is stated as data, with no handle in it, because the handles only exist once the run has
/// made them: a case says "the texture" and this is what that means on the side running it.
pub struct Made {
    texture: Handle,
    sampler: Handle,
    /// The coordinate buffer, which a pointer argument names where a layout has one. A case needs
    /// no buffer of its own for that: what a pointer argument records is an address, and any
    /// buffer of the case's is one.
    buffer: Handle,
    pipeline: Handle,
}

/// Where a case's encoder comes from.
enum Layout {
    /// `newArgumentEncoderWithBufferIndex:` of the fixture's kernel: the layout its AIR declares.
    Function(u32),
    /// `newArgumentEncoderWithArguments:` of these descriptors: a layout the case describes.
    Descriptors(&'static [ArgumentDescriptorSpec]),
}

/// A case over one argument encoder.
pub struct ArgumentEncoderCase {
    name: &'static str,
    layout: Layout,
    /// The offset `setArgumentBuffer:` is given, and the array element where the case uses
    /// `setArgumentBuffer:startOffset:arrayElement:` instead.
    at: u64,
    element: Option<u64>,
    /// The writes the case makes, in order.
    writes: fn(&Made) -> Vec<ArgumentWrite>,
    /// The argument indices the case reads `constantDataAtIndex:` for.
    constants: &'static [u32],
    /// Whether the case dispatches the fixture's kernel over the encoded argument buffer.
    ///
    /// Only a case whose layout **is** the kernel's, with the arguments at the offset the kernel
    /// is bound at, may: elsewhere the kernel would read the pattern as a resource ID.
    dispatches: bool,
}

/// The fixture kernel's own layout, which `newArgumentEncoderWithBufferIndex:` answers.
const FUNCTION_LAYOUT: u32 = ARGUMENT_BUFFER_INDEX;

/// A mixed layout of every argument kind that holds a resource or a constant: where an offset, a
/// stride or an alignment is wrong by a word, this is the layout it shows up in.
///
/// The two constants are of different alignments on purpose — a `uint` of four bytes and a
/// `float4` of sixteen — so the padding Metal inserts between them is part of the answer, and the
/// array of three textures states the argument-index stride.
const MIXED: &[ArgumentDescriptorSpec] = &[
    ArgumentDescriptorSpec::new(DataType::Texture, 0),
    ArgumentDescriptorSpec::new(DataType::Sampler, 1),
    ArgumentDescriptorSpec::new(DataType::Pointer, 2),
    ArgumentDescriptorSpec::new(DataType::UInt, 3),
    ArgumentDescriptorSpec::new(DataType::Float4, 4),
    ArgumentDescriptorSpec::new(DataType::Texture, 5).array(3),
];

/// A layout of the four argument kinds that hold another Metal object's resource ID.
///
/// Its own case rather than more of [`MIXED`], because these four are the kinds whose descriptor
/// Metal may not take at all: a layout it refuses would otherwise take the mixed case's answer
/// down with it, and this family's point is that each gap is readable on its own.
const STATES: &[ArgumentDescriptorSpec] = &[
    ArgumentDescriptorSpec::new(DataType::ComputePipeline, 0),
    ArgumentDescriptorSpec::new(DataType::RenderPipeline, 1),
    ArgumentDescriptorSpec::new(DataType::IndirectCommandBuffer, 2),
    ArgumentDescriptorSpec::new(DataType::VisibleFunctionTable, 3),
];

/// Every case of the family.
pub fn all() -> Vec<ArgumentEncoderCase> {
    vec![
        // The smallest case that exercises the encoder end to end: the kernel's own layout, the
        // texture and the sampler written one at a time, and the dispatch that proves the encoder
        // and the translated function agree about where each went.
        ArgumentEncoderCase {
            name: "argument-encoder-texture-sampler",
            layout: Layout::Function(FUNCTION_LAYOUT),
            at: 0,
            element: None,
            writes: |made| {
                vec![
                    ArgumentWrite::Texture {
                        texture: Some(made.texture),
                        index: 0,
                    },
                    ArgumentWrite::Sampler {
                        sampler: Some(made.sampler),
                        index: 1,
                    },
                ]
            },
            constants: &[],
            dispatches: true,
        },
        // The same layout and the same answer through the **range** setters, which is a different
        // pair of Metal calls over the same two arguments: a range of one at the argument's own
        // index must write the word the single setter writes.
        ArgumentEncoderCase {
            name: "argument-encoder-ranges",
            layout: Layout::Function(FUNCTION_LAYOUT),
            at: 0,
            element: None,
            writes: |made| {
                vec![
                    ArgumentWrite::Textures {
                        textures: vec![Some(made.texture)],
                        first: 0,
                    },
                    ArgumentWrite::Samplers {
                        samplers: vec![Some(made.sampler)],
                        first: 1,
                    },
                ]
            },
            constants: &[],
            dispatches: true,
        },
        // `setArgumentBuffer:startOffset:arrayElement:`: element 1 of an array of argument buffers
        // from offset 32, so the arguments land at 32 + 1 * encodedLength and the mask says so.
        // It does not dispatch: the kernel is bound at offset 0, where this layout's arguments are
        // not.
        ArgumentEncoderCase {
            name: "argument-encoder-array-element",
            layout: Layout::Function(FUNCTION_LAYOUT),
            at: 32,
            element: Some(1),
            writes: |made| {
                vec![
                    ArgumentWrite::Texture {
                        texture: Some(made.texture),
                        index: 0,
                    },
                    ArgumentWrite::Sampler {
                        sampler: Some(made.sampler),
                        index: 1,
                    },
                ]
            },
            constants: &[],
            dispatches: false,
        },
        // A layout the case describes itself, holding a texture, a sampler, a pointer, two
        // constants of different alignments and an array of three textures, written through both
        // the single and the range setters. A nil in the middle of the texture array is there so
        // the mask distinguishes an argument written as zero from one not written at all.
        ArgumentEncoderCase {
            name: "argument-encoder-mixed-descriptors",
            layout: Layout::Descriptors(MIXED),
            at: 0,
            element: None,
            writes: |made| {
                vec![
                    ArgumentWrite::Texture {
                        texture: Some(made.texture),
                        index: 0,
                    },
                    ArgumentWrite::Sampler {
                        sampler: Some(made.sampler),
                        index: 1,
                    },
                    ArgumentWrite::Buffer {
                        buffer: Some(made.buffer),
                        offset: 0,
                        index: 2,
                    },
                    // The same pointer argument through the range form, at another offset into the
                    // same buffer: the word must move by the eight bytes the offset moved.
                    ArgumentWrite::Buffers {
                        buffers: vec![(Some(made.buffer), 8)],
                        first: 2,
                    },
                    ArgumentWrite::Samplers {
                        samplers: vec![Some(made.sampler)],
                        first: 1,
                    },
                    ArgumentWrite::Constant {
                        index: 3,
                        bytes: 0x1234_5678u32.to_le_bytes().to_vec(),
                    },
                    ArgumentWrite::Constant {
                        index: 4,
                        bytes: [1.0f32, -2.0, 0.5, 64.0]
                            .iter()
                            .flat_map(|value| value.to_le_bytes())
                            .collect(),
                    },
                    ArgumentWrite::Textures {
                        textures: vec![Some(made.texture), None, Some(made.texture)],
                        first: 5,
                    },
                ]
            },
            constants: &[3, 4],
            dispatches: false,
        },
        // The four argument kinds that hold another Metal object's resource ID, each written by its
        // single setter and then by its range form. Only the compute state is a real object: the
        // other three are Metal's nil, which writes a zero word, and a zero word is still a word
        // the mask reports as written because the buffer held the pattern before.
        ArgumentEncoderCase {
            name: "argument-encoder-state-descriptors",
            layout: Layout::Descriptors(STATES),
            at: 0,
            element: None,
            writes: |made| {
                vec![
                    ArgumentWrite::ComputePipelineState {
                        state: Some(made.pipeline),
                        index: 0,
                    },
                    ArgumentWrite::ComputePipelineStates {
                        states: vec![Some(made.pipeline)],
                        first: 0,
                    },
                    ArgumentWrite::RenderPipelineState {
                        state: None,
                        index: 1,
                    },
                    ArgumentWrite::RenderPipelineStates {
                        states: vec![None],
                        first: 1,
                    },
                    ArgumentWrite::IndirectCommandBuffer {
                        commands: None,
                        index: 2,
                    },
                    ArgumentWrite::IndirectCommandBuffers {
                        commands: vec![None],
                        first: 2,
                    },
                    ArgumentWrite::VisibleFunctionTable {
                        table: None,
                        index: 3,
                    },
                    ArgumentWrite::VisibleFunctionTables {
                        tables: vec![None],
                        first: 3,
                    },
                ]
            },
            constants: &[],
            dispatches: false,
        },
    ]
}

/// The texture's bytes: texel `i` holds the four channels of `i`'s low four bits, each 0 or 255.
fn texels() -> Vec<u8> {
    (0..THREADS)
        .flat_map(|texel| {
            (0..4).map(move |channel| match texel & (1 << channel) {
                0 => 0u8,
                _ => 0xff,
            })
        })
        .collect()
}

/// The normalized coordinate of each thread's own texel centre.
fn coordinates() -> Vec<u8> {
    (0..THREADS)
        .flat_map(|thread| {
            let x = (thread % SIDE) as f32 + 0.5;
            let y = (thread / SIDE) as f32 + 0.5;
            [x / SIDE as f32, y / SIDE as f32]
        })
        .flat_map(f32::to_le_bytes)
        .collect()
}

impl ArgumentEncoderCase {
    /// The fixture the case runs.
    fn fixture(&self) -> Fixture {
        Fixture {
            name: FIXTURE,
            entry: ENTRY,
        }
    }

    /// The texture the case samples.
    fn texture(&self) -> TextureSpec {
        TextureSpec::new(
            PixelFormat::RGBA8Unorm,
            [SIDE, SIDE],
            SIDE * 4,
            TextureRole::Read,
        )
    }

    /// The sampler state the encoder writes: nearest, clamp to edge, normalized, and able to live
    /// in an argument buffer at all, which Metal gives a state no resource ID without.
    fn sampler(&self) -> SamplerSpec {
        SamplerSpec::filtered(SamplerMinMagFilter::Nearest)
            .addressed(SamplerAddressMode::ClampToEdge)
            .for_argument_buffer()
    }

    /// The layout of the words the kernel wrote: one row a thread, four floats each, so a diff
    /// names the texel whose sample moved and the channel inside it.
    fn output_layout(&self) -> ByteLayout {
        ByteLayout {
            region: Region::Buffer {
                offset: 0,
                length: u64::from(THREADS) * 16,
            },
            element: Element::F32,
            row_offset: 0,
            row_elements: 4,
            row_pitch: 16,
            rows: THREADS,
            image_pitch: THREADS * 16,
            images: 1,
            gaps: Gaps::Ignored,
        }
    }
}

impl Case for ArgumentEncoderCase {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let fixture = self.fixture();
        [RUNTIME_TEXT, METAL_TEXT]
            .into_iter()
            .map(|extension| {
                Ok(ShaderRef {
                    fixture: format!("{FIXTURE}{extension}"),
                    entry: ENTRY.to_owned(),
                    source_sha256: crate::sha256_hex(fixture.text(extension)?.as_bytes()),
                })
            })
            .collect()
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let fixture = self.fixture();
        let mut inputs = vec![
            Input {
                name: "air".to_owned(),
                digest: Digest::sha256(fixture.text(RUNTIME_TEXT)?.as_bytes()),
            },
            Input {
                name: "air-typed".to_owned(),
                digest: Digest::sha256(fixture.text(METAL_TEXT)?.as_bytes()),
            },
            Input::bytes("texture", &texels()),
            Input::bytes("buffer:coordinates", &coordinates()),
            Input::bytes("sampler", &self.sampler().words()),
            // Both the output buffer and the argument buffer hold the pattern before anything
            // writes them, which is what makes "this word changed" a question with an answer.
            Input::bytes("buffer:output", &vec![BLANK; BUFFER_BYTES]),
            Input::bytes("buffer:arguments", &vec![BLANK; BUFFER_BYTES]),
        ];
        let mut layout = Vec::new();
        match &self.layout {
            Layout::Function(index) => {
                layout.extend(b"function");
                layout.extend(index.to_le_bytes());
            }
            Layout::Descriptors(descriptors) => {
                layout.extend(b"descriptors");
                for descriptor in *descriptors {
                    layout.extend(descriptor.words());
                }
            }
        }
        layout.extend(self.at.to_le_bytes());
        layout.extend(self.element.unwrap_or(u64::MAX).to_le_bytes());
        inputs.push(Input::bytes("layout", &layout));
        // The writes are the case's inputs too, and they name no handle: what identifies a
        // resource is the case's own bytes for it, above.
        let plan = Made {
            texture: Handle(0),
            sampler: Handle(0),
            buffer: Handle(0),
            pipeline: Handle(0),
        };
        let mut writes = Vec::new();
        for write in (self.writes)(&plan) {
            writes.extend(write.words());
        }
        for index in self.constants {
            writes.extend(index.to_le_bytes());
        }
        inputs.push(Input::bytes("writes", &writes));
        inputs.push(Input::bytes(
            "dispatch",
            &[u32::from(self.dispatches), THREADS]
                .iter()
                .flat_map(|word| word.to_le_bytes())
                .collect::<Vec<u8>>(),
        ));
        Ok(inputs)
    }

    /// The notices a venue may announce for the sampling these cases do, each declared on its own
    /// code.
    ///
    /// Nothing of a value is loosened here: every sampled word, the encoder's own facts and the
    /// argument buffer's changed-word mask are compared exactly on every venue, which is what the
    /// texels buy. The texture holds the sixteen patterns of each component at 0 or 255, so every
    /// sampled component is exactly 0.0 or 1.0 wherever the sampling happens, and a filter, a
    /// denormal or a decode a device does its own way cannot move a word of that.
    ///
    /// The three codes were measured, not guessed: Apple Metal announces none of them for this
    /// work and llvmpipe announces all three, so a run on either venue alone would have left the
    /// other's set undeclared. Only the two cases that dispatch declare them; the three that
    /// encode and read back never sample, so a declaration there would never fire.
    fn allowances(&self) -> Vec<Allowance> {
        // Every row reads its answer back through a `Shared` buffer, so on a device without
        // unified memory the runtime announces `shared-in-system-memory`
        // (`crate::replay::shared_memory_notice` holds the whole reason). It declares the
        // notices site alone, so every value of this family still compares against Metal's
        // exactly.
        //
        // Measured on the RTX 5080 Laptop (615.71.09): it fired for
        // all 5 rows of the family, and each was divergent on that field and
        // nothing else (replay-nvidia.log). On MoltenVK, where
        // the chip's GPU shares the CPU's memory, it cannot fire at all, which is why no
        // recording of this suite declares it.
        let mut named = vec![crate::replay::shared_memory_notice()];
        if self.dispatches {
            named.extend(
                [
                    Degradation::DeviceTexelDenormals,
                    Degradation::FloatSampleFilter,
                    Degradation::SrgbTexelDecode,
                ]
                .into_iter()
                .map(|notice| crate::replay::notice_allowance(notice.code())),
            );
        }
        named
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let stop = |at: &str, why: String| Abort {
            at: at.to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        };
        let loaded = self
            .fixture()
            .loaded()
            .map_err(|why| stop(site::LIBRARY, why))?;

        let library = made(sites, site::LIBRARY, side.library(&loaded.source()))?;
        let pipeline = made(sites, site::PIPELINE, side.compute_pipeline(library, ENTRY))?;
        let texture = made(
            sites,
            site::TEXTURE,
            side.texture_2d_in_role(&self.texture(), &texels()),
        )?;
        let sampler = made(sites, site::SAMPLER, side.sampler(&self.sampler()))?;
        let coordinates = made(sites, site::COORDINATES, side.buffer(&coordinates()))?;
        let output = made(
            sites,
            site::OUTPUT_BUFFER,
            side.buffer(&vec![BLANK; BUFFER_BYTES]),
        )?;
        let arguments = made(
            sites,
            site::ARGUMENT_BUFFER,
            side.buffer(&vec![BLANK; BUFFER_BYTES]),
        )?;

        let encoder = match &self.layout {
            Layout::Function(index) => made(
                sites,
                site::ENCODER,
                side.argument_encoder(library, ENTRY, *index),
            )?,
            Layout::Descriptors(descriptors) => made(
                sites,
                site::ENCODER,
                side.argument_encoder_with_arguments(descriptors),
            )?,
        };
        match side.argument_encoder_facts(encoder, LABEL) {
            Ok(facts) => sites.push(facts.node(site::FACTS)),
            Err(failure) => {
                sites.push(Node::leaf(site::FACTS, Observed::value(failure.recorded())));
                return Err(Abort {
                    at: site::FACTS.to_owned(),
                    failure,
                });
            }
        };
        made(
            sites,
            site::SET_ARGUMENT_BUFFER,
            side.set_argument_buffer(encoder, arguments, self.at, self.element),
        )?;

        let made_resources = Made {
            texture,
            sampler,
            buffer: coordinates,
            pipeline,
        };
        for (at, write) in (self.writes)(&made_resources).iter().enumerate() {
            made(
                sites,
                &format!("write/{at}-{}", write.call()),
                side.encode_argument(encoder, write),
            )?;
        }
        for index in self.constants {
            call(
                sites,
                &format!("constant-offset/{index}"),
                side.constant_data_offset(encoder, *index),
                |offset| Value::Uint(*offset),
            )?;
        }

        // The argument buffer's own bytes never enter the transcript: they are addresses and
        // resource IDs. What the case records is which words stopped holding the pattern, which
        // is the layout and nothing of the host's.
        let encoded = call(
            sites,
            site::READ_BACK,
            side.read_buffer(arguments, 0, BUFFER_BYTES),
            |_| Value::Absent,
        )?;
        sites.push(changed_node(
            site::CHANGED,
            changed_words(&encoded, WORDS),
            0,
        ));

        if self.dispatches {
            let command_buffer = made(
                sites,
                "dispatch",
                side.dispatch(
                    pipeline,
                    &Bindings::buffers(&[coordinates, output, arguments]),
                    [1, 1, 1],
                    [THREADS, 1, 1],
                ),
            )?;
            call(sites, site::STATUS, side.wait(command_buffer), Clone::clone)?;
            let words = call(
                sites,
                "output-read-back",
                side.read_buffer(output, 0, THREADS as usize * 16),
                |_| Value::Absent,
            )?;
            sites.leaf(site::OUTPUT, Observed::bytes(words, self.output_layout()));
        }
        Ok(())
    }
}
