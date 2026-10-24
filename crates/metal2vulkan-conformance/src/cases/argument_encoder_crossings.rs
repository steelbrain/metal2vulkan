//! Argument-buffer layout where **two** things decide the answer: the crossings a per-type sweep
//! cannot see.
//!
//! [`crate::cases::argument_encoders`] compares what an encoder does with one layout at a time and
//! [`crate::cases::function_bindings`] sweeps `MTLDataType` one member at a time. Between them they
//! are a sweep over single values, and the honest limit of a sweep over single values is stated in
//! `docs/CONFORMANCE-COVERAGE.md`: "a type whose values are all compared says nothing about the
//! pairs and triples of them, which is where most of the behaviour an implementation can get wrong
//! lives". This family is three of those pairs, each one chosen because **neither** axis alone
//! moves the answer and the two together do.
//!
//! # The crossings, and the measurement that says each is a crossing
//!
//! Every number below is Metal's, taken on an Apple M3 Max under macOS 27.0.1 by
//! `probes/argument_descriptor_types.m` before a case was written, so the family states what it is
//! pointed at rather than discovering it in a recording.
//!
//! - **An argument's type crossed with its position.** A single `MTLDataTypeChar` argument is an
//!   encoder of 1 byte aligned to 1, and a single `MTLDataTypeFloat4` one of 16 aligned to 16.
//!   Neither says what a layout of two of each is, and the **order** decides it: `char, char,
//!   float4, float4` is **48** bytes and `float4, char, float4, char` — the same multiset — is
//!   **64**. A per-type sweep cannot see a difference that only a permutation makes, and an
//!   implementation that laid members out by a running sum, or that sorted them, answers one
//!   number for both.
//! - **`constantBlockAlignment` crossed with what precedes the constant.** A single
//!   `MTLDataTypeUInt` with `constantBlockAlignment` 64 is an encoder of **4** bytes aligned to
//!   **4**: the field changes nothing at all, because offset 0 is already aligned to anything. Put
//!   a texture in front of it and the same field moves the constant from offset 8 to offset
//!   **64** and the encoder from 16 bytes to **72** — while the encoder's own `alignment` stays
//!   **8** and does not rise to 64. So the field is indistinguishable by construction on its own
//!   and load-bearing one argument later, which is the definition of a crossing rather than a
//!   value.
//! - **The two offsets of `MTLArgumentEncoder`, which compose.**
//!   `setArgumentBuffer:offset:` puts the arguments at `offset`;
//!   `setArgumentBuffer:startOffset:arrayElement:` puts them at
//!   `startOffset + arrayElement * encodedLength`. Over the 48-byte layout, `(0, element 1)` is 48
//!   and `(8, no element)` is 8, and **`(8, element 1)` is 56** — a cell neither single axis
//!   reaches. Metal takes a `startOffset` that is not a multiple of the layout's alignment (8 on
//!   an alignment-16 layout) without complaint, which is itself a fact nothing had measured.
//! - **Which member of a layout is a nested argument buffer.** This is Metal's own
//!   classification of a member and not a value a case states, so no sweep of any enumeration
//!   reaches it. `newArgumentEncoderForBufferAtIndex:` answers an encoder for one index of this
//!   family's fixture and **nil** for twenty others: a plain `device float *`, a pointer to a
//!   struct of constants, an inline struct of a texture and a sampler, a texture, a sampler, a
//!   depth texture, an array of textures, an array of samplers, an array of pointers, an array of
//!   inline constants, an array of structs, an indirect command buffer, a compute pipeline state,
//!   and an index past the layout. Only a pointer whose pointee holds a **resource** answers an
//!   encoder, and that encoder's own layout — 8 bytes aligned to 8, the one texture of
//!   `ReflectionSub` — is a second layout crossed with the first.
//!
//! # What each case compares, and why none of it is a word a host owns
//!
//! The same three things [`crate::argument_encoders`] states, for the same reason: an argument's
//! word is a Metal address or a resource ID and belongs in no recording.
//!
//! - **the layout the encoder states** — `encodedLength`, `alignment`, the `label` round trip and
//!   whether `device` is the case's own;
//! - **where each constant landed** — `constantDataAtIndex:` as its **offset inside the
//!   arguments**, which is the layout fact the address stands for, and the only probe into the
//!   position of an argument whose type has no setter (a depth-stencil state, an acceleration
//!   structure, an intersection function table);
//! - **which eight-byte words the encoder wrote** — over a buffer pre-filled with
//!   [`BLANK`], so a word written as zero and a word not written
//!   read differently. That is every written argument's offset and stride with none of the host's
//!   numbers in it.
//!
//! The offsets crossing is read off the **mask** and not off the constant offsets, and the reason
//! is worth stating because it is the opposite of what a reader expects:
//! `constantDataAtIndex:` is recorded as the offset of the constant inside the *arguments*, which
//! is the layout fact the address stands for, so it answers the same number at every cell of that
//! case by construction. What moves is where the arguments start in the buffer, and the mask is
//! the only site that says so — word 0 at `(0, no element)`, word 6 at `(0, element 1)`, word 1 at
//! `(8, no element)` and word 7 at `(8, element 1)`, which is the cell the composition owns.
//!
//! # `MTLDataType`'s argument-only values
//!
//! Four values of `MTLDataType` that no shader in this tree declares are reachable through
//! `MTLArgumentDescriptor` alone, and the singles case is where they are compared:
//! `MTLDataTypeDepthStencilState`, which MSL has no type for, and
//! `MTLDataTypeIntersectionFunctionTable`, `MTLDataTypePrimitiveAccelerationStructure` and
//! `MTLDataTypeInstanceAccelerationStructure`. Each is **8 bytes aligned to 8** in Metal's own
//! answer, and reaching them costs this suite no ray-tracing work at all: a descriptor states a
//! layout, and nothing here makes an acceleration structure, links an intersection function or
//! asks a device for a ray-tracing feature. What is compared is the layout arithmetic, and the
//! mixtures case crosses them with a constant that follows them so their size shows up as that
//! constant's offset and not only as a total.
//!
//! `MTLDataTypeNone` and `MTLDataTypeTensor` are the two values a descriptor cannot carry: Metal's
//! own validation ends the process for each, which `scripts/conformance-coverage/unreachable_values.tsv`
//! records with the assertion it printed.
//!
//! # Why this family is safe
//!
//! Every case makes argument encoders, small buffers, one 4 by 4 texture and one sampler state, and
//! reads what the encoders say. **Nothing is dispatched, no command buffer is made and nothing is
//! drawn**, so no case here can fault the device or trip its watchdog. No case asks for a data
//! type, an index or an offset Metal's own validation refuses: each was measured accepted by the
//! probe first, and the two fatal data types are left out by name.
//!
//! # Exactness
//!
//! Every site is a number, a bool, a name or a word mask. The family declares no tolerance and
//! announces nothing: no case samples a texture, blends a pixel or touches a float.

use crate::argument_encoders::{
    changed_node, changed_words, ArgumentDescriptorSpec, ArgumentWrite, ARGUMENT_WORD, BLANK,
};
use crate::case::{call, made, Abort, Case, Failure, Handle, Input, MetalSource, Side};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::transcript::{Digest, Node, Observed, Sites, Value};
use metal2vulkan::metal::DataType;

/// The fixture whose kernel takes the argument buffer the nested case opens: the one
/// [`crate::cases::function_bindings`] reads a reflection off, so this family needs no fixture of
/// its own and the two compare the same layout from the two ends.
const FIXTURE: &str = "reflection/argument_buffers";

/// Its kernel, and the buffer index its argument buffer is at.
const ENTRY: &str = "reflection_argument_buffers";
const OUTER_BUFFER_INDEX: u32 = 0;

/// The one argument of that layout whose member is a nested argument buffer:
/// `constant ReflectionSub *sub [[id(15)]]`, whose pointee holds a texture.
const NESTED_INDEX: u32 = 15;

/// Bytes of every buffer a case makes, chosen so the largest composed offset of the offsets case
/// (16 + 2 * 48, with 48 bytes of arguments after it) still lies inside one, and so the mask
/// reports the bytes past the arguments as well as the arguments themselves.
const BUFFER_BYTES: usize = 256;

/// The words a mask reports: the whole of a case's argument buffer.
const WORDS: usize = BUFFER_BYTES / ARGUMENT_WORD;

/// The label every encoder of this family is given and read back.
const LABEL: &str = "conformance-argument-crossing";

/// The bytes a constant argument is filled with, chosen so no word of it is [`BLANK`]: a constant
/// written as zero and a constant not written must read differently in the mask.
const CONSTANT: [u8; 16] = [
    0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09,
];

/// One argument type's own encoder, with no second argument to interact with.
///
/// The control the mixtures are read against: each of these is one `MTLDataType`'s size and
/// alignment as Metal states it, and the point of the family is that these ten numbers do not
/// determine the nine layouts below.
const SINGLES: &[(&str, &[ArgumentDescriptorSpec])] = &[
    ("char", &[ArgumentDescriptorSpec::new(DataType::Char, 0)]),
    ("uint", &[ArgumentDescriptorSpec::new(DataType::UInt, 0)]),
    (
        "float4",
        &[ArgumentDescriptorSpec::new(DataType::Float4, 0)],
    ),
    (
        "texture",
        &[ArgumentDescriptorSpec::new(DataType::Texture, 0)],
    ),
    (
        "sampler",
        &[ArgumentDescriptorSpec::new(DataType::Sampler, 0)],
    ),
    (
        "pointer",
        &[ArgumentDescriptorSpec::new(DataType::Pointer, 0)],
    ),
    // The four values no shader of this tree declares, each reachable through a descriptor alone.
    (
        "depth-stencil-state",
        &[ArgumentDescriptorSpec::new(DataType::DepthStencilState, 0)],
    ),
    (
        "intersection-function-table",
        &[ArgumentDescriptorSpec::new(
            DataType::IntersectionFunctionTable,
            0,
        )],
    ),
    (
        "primitive-acceleration-structure",
        &[ArgumentDescriptorSpec::new(
            DataType::PrimitiveAccelerationStructure,
            0,
        )],
    ),
    (
        "instance-acceleration-structure",
        &[ArgumentDescriptorSpec::new(
            DataType::InstanceAccelerationStructure,
            0,
        )],
    ),
    // `constantBlockAlignment` with nothing in front of the constant, which is the row that makes
    // the field's own effect indistinguishable by construction: 4 and 4, exactly as the plain
    // `uint` above. The mixtures case is where the same field moves 56 bytes.
    (
        "uint-block-aligned-64",
        &[ArgumentDescriptorSpec::new(DataType::UInt, 0).block_aligned(64)],
    ),
];

/// Layouts of more than one argument, where what each argument's position and the encoder's own
/// totals come to is decided by **two** of the arguments and not one.
const MIXTURES: &[(&str, &[ArgumentDescriptorSpec])] = &[
    // The permutation pair: one multiset, two orders, and Metal answers 48 for the first and 64
    // for the second. Nothing in `SINGLES` distinguishes them.
    (
        "narrow-then-wide",
        &[
            ArgumentDescriptorSpec::new(DataType::Char, 0),
            ArgumentDescriptorSpec::new(DataType::Char, 1),
            ArgumentDescriptorSpec::new(DataType::Float4, 2),
            ArgumentDescriptorSpec::new(DataType::Float4, 3),
        ],
    ),
    (
        "wide-then-narrow",
        &[
            ArgumentDescriptorSpec::new(DataType::Float4, 0),
            ArgumentDescriptorSpec::new(DataType::Char, 1),
            ArgumentDescriptorSpec::new(DataType::Float4, 2),
            ArgumentDescriptorSpec::new(DataType::Char, 3),
        ],
    ),
    (
        "alternating",
        &[
            ArgumentDescriptorSpec::new(DataType::Char, 0),
            ArgumentDescriptorSpec::new(DataType::Float4, 1),
            ArgumentDescriptorSpec::new(DataType::Char, 2),
            ArgumentDescriptorSpec::new(DataType::Float4, 3),
        ],
    ),
    // A handle among the constants, in both orders: a texture is 8 bytes aligned to 8, so where it
    // sits decides which constant the padding lands in front of.
    (
        "constants-then-handle",
        &[
            ArgumentDescriptorSpec::new(DataType::Char, 0),
            ArgumentDescriptorSpec::new(DataType::UInt, 1),
            ArgumentDescriptorSpec::new(DataType::Float4, 2),
            ArgumentDescriptorSpec::new(DataType::Texture, 3),
        ],
    ),
    (
        "handle-then-constants",
        &[
            ArgumentDescriptorSpec::new(DataType::Texture, 0),
            ArgumentDescriptorSpec::new(DataType::Float4, 1),
            ArgumentDescriptorSpec::new(DataType::UInt, 2),
            ArgumentDescriptorSpec::new(DataType::Char, 3),
        ],
    ),
    // A depth-stencil state crossed with the constants after it: the state has no setter, so the
    // only thing that says it took 8 bytes aligned to 8 is where the `char` after it landed.
    (
        "depth-stencil-state-then-constants",
        &[
            ArgumentDescriptorSpec::new(DataType::DepthStencilState, 0),
            ArgumentDescriptorSpec::new(DataType::Char, 1),
            ArgumentDescriptorSpec::new(DataType::Float4, 2),
            ArgumentDescriptorSpec::new(DataType::Sampler, 3),
        ],
    ),
    // The three table and acceleration-structure types in one layout, with a `char` behind them
    // for the same reason: 32 bytes of handles put it at offset 32, and the encoder at 40.
    (
        "tables-then-constant",
        &[
            ArgumentDescriptorSpec::new(DataType::IntersectionFunctionTable, 0),
            ArgumentDescriptorSpec::new(DataType::PrimitiveAccelerationStructure, 1),
            ArgumentDescriptorSpec::new(DataType::InstanceAccelerationStructure, 2),
            ArgumentDescriptorSpec::new(DataType::VisibleFunctionTable, 3),
            ArgumentDescriptorSpec::new(DataType::Char, 4),
        ],
    ),
    // `constantBlockAlignment` with an argument in front of it, which is the same field as the
    // `uint-block-aligned-64` single and the row where it moves 56 bytes.
    (
        "handle-then-block-aligned-constant",
        &[
            ArgumentDescriptorSpec::new(DataType::Texture, 0),
            ArgumentDescriptorSpec::new(DataType::UInt, 1).block_aligned(64),
            ArgumentDescriptorSpec::new(DataType::Char, 2),
        ],
    ),
    // An array between two constants: the array's own stride times its length decides where the
    // second constant goes, which neither the element type nor the length says on its own.
    (
        "array-between-constants",
        &[
            ArgumentDescriptorSpec::new(DataType::Char, 0),
            ArgumentDescriptorSpec::new(DataType::Texture, 1).array(2),
            ArgumentDescriptorSpec::new(DataType::Float4, 3),
        ],
    ),
];

/// The layout the offsets case composes over: four arguments, 48 bytes, aligned to 16, with three
/// constants whose offsets inside the arguments say where the arguments start.
const COMPOSED: &[ArgumentDescriptorSpec] = &[
    ArgumentDescriptorSpec::new(DataType::Char, 0),
    ArgumentDescriptorSpec::new(DataType::UInt, 1),
    ArgumentDescriptorSpec::new(DataType::Float4, 2),
    ArgumentDescriptorSpec::new(DataType::Texture, 3),
];

/// The second layout the offsets case composes over: 72 bytes aligned to 8, so the composed base
/// is a multiple of neither the first layout's length nor its alignment, and the block-aligned
/// constant's offset is one the composition and the packing decide **together**.
const COMPOSED_BLOCK: &[ArgumentDescriptorSpec] = &[
    ArgumentDescriptorSpec::new(DataType::Texture, 0),
    ArgumentDescriptorSpec::new(DataType::UInt, 1).block_aligned(64),
    ArgumentDescriptorSpec::new(DataType::Char, 2),
];

/// Where one cell of the offsets case puts the arguments: the `setArgumentBuffer:` offset and the
/// array element, or `None` for the setter that takes no element.
///
/// The cells are the crossing written out. Two rows vary only the offset, two only the element,
/// and three vary both — and it is the last three that a side composing the two wrongly fails
/// while answering the first four exactly.
const PLACES: &[(&str, u64, Option<u64>)] = &[
    ("at-0", 0, None),
    ("at-0-element-0", 0, Some(0)),
    ("at-0-element-1", 0, Some(1)),
    ("at-8", 8, None),
    ("at-8-element-0", 8, Some(0)),
    ("at-8-element-1", 8, Some(1)),
    ("at-16-element-2", 16, Some(2)),
];

/// The indices the nested case asks `newArgumentEncoderForBufferAtIndex:` for, with what each
/// member of the fixture's layout is.
///
/// Every one but [`NESTED_INDEX`] is a member Metal answers **nil** for, and the list is the
/// crossing: the member kinds a layout can hold against Metal's own answer to "is this a nested
/// argument buffer". The last is past the layout altogether.
const NESTED_PLACES: &[(&str, u32)] = &[
    ("texture", 0),
    ("sampler", 1),
    ("device-pointer", 2),
    ("array-of-textures", 3),
    ("constant-pointer", 6),
    ("inline-constant", 7),
    ("indirect-command-buffer", 9),
    ("write-texture", 10),
    ("depth-texture", 11),
    ("pointer-to-struct-of-constants", 12),
    ("compute-pipeline-state", 13),
    ("pointer-to-struct-of-resources", NESTED_INDEX),
    ("array-of-samplers", 16),
    ("array-of-pointers", 18),
    ("array-of-inline-constants", 20),
    ("inline-struct-of-resources", 22),
    ("array-of-inline-structs", 24),
    ("past-the-layout", 99),
];

/// The offsets the nested case sets its nested argument buffer at: the composition inside the
/// **nested** buffer, which the outer encoder's own offset has nothing to do with.
const NESTED_OFFSETS: &[(&str, u64)] = &[("nested-at-0", 0), ("nested-at-40", 40)];

/// What one case of the family crosses.
enum Crossing {
    /// One encoder a layout, over the layouts given: the totals and every argument's position.
    Layouts(&'static [(&'static str, &'static [ArgumentDescriptorSpec])]),
    /// One layout over every place of [`PLACES`], through both of the two setters.
    Places(&'static [(&'static str, &'static [ArgumentDescriptorSpec])]),
    /// The fixture kernel's own layout, every member asked whether it is a nested argument buffer,
    /// and the one that is encoded into at two offsets.
    Nested,
}

/// A case over one crossing.
pub struct CrossingCase {
    name: &'static str,
    crossing: Crossing,
}

/// Every case of the family.
pub fn all() -> Vec<CrossingCase> {
    vec![
        // The control: one argument type an encoder, which is what the mixtures are read against.
        CrossingCase {
            name: "argument-encoder-cross-type-sizes",
            crossing: Crossing::Layouts(SINGLES),
        },
        // The crossing: the same types in layouts of four, where the order and the neighbours
        // decide the answer.
        CrossingCase {
            name: "argument-encoder-cross-type-mixtures",
            crossing: Crossing::Layouts(MIXTURES),
        },
        // The two offsets, composed.
        CrossingCase {
            name: "argument-encoder-cross-offsets",
            crossing: Crossing::Places(&[
                ("composed", COMPOSED),
                ("composed-block-aligned", COMPOSED_BLOCK),
            ]),
        },
        // The nested argument buffer: which member is one, and what its own encoder does.
        CrossingCase {
            name: "argument-encoder-cross-nested",
            crossing: Crossing::Nested,
        },
    ]
}

/// The writes that fill every argument of `layout` a setter can reach, so the mask says where each
/// one landed.
///
/// A type with no setter — a depth-stencil state, an acceleration structure, an intersection
/// function table — is left unwritten on purpose, and what says it took the room it did is the
/// offset of the constant behind it.
fn writes(
    layout: &[ArgumentDescriptorSpec],
    texture: Handle,
    sampler: Handle,
    buffer: Handle,
) -> Vec<ArgumentWrite> {
    let mut out = Vec::new();
    for one in layout {
        let count = one.array_length.max(1);
        for element in 0..count {
            let index = one.index + element;
            match one.data_type {
                DataType::Texture => out.push(ArgumentWrite::Texture {
                    texture: Some(texture),
                    index,
                }),
                DataType::Sampler => out.push(ArgumentWrite::Sampler {
                    sampler: Some(sampler),
                    index,
                }),
                DataType::Pointer => out.push(ArgumentWrite::Buffer {
                    buffer: Some(buffer),
                    offset: 0,
                    index,
                }),
                _ => {}
            }
        }
    }
    out
}

/// The constant arguments of `layout`, with the bytes each is filled with: as many of [`CONSTANT`]
/// as the type takes, so a constant's own bytes never hold [`BLANK`] and the mask reports it.
fn constants(layout: &[ArgumentDescriptorSpec]) -> Vec<(u32, Vec<u8>)> {
    layout
        .iter()
        .filter_map(|one| {
            let bytes = match one.data_type {
                DataType::Char => 1,
                DataType::UInt => 4,
                DataType::Float4 => 16,
                _ => return None,
            };
            Some((one.index, CONSTANT[..bytes].to_vec()))
        })
        .collect()
}

impl CrossingCase {
    /// The fixture the nested case reads; the others read none and feed its digest anyway, so one
    /// family has one input shape.
    fn fixture(&self) -> Fixture {
        Fixture {
            name: FIXTURE,
            entry: ENTRY,
        }
    }

    /// Every layout the case states, in order, as the words its input fingerprint covers.
    fn layout_words(&self) -> Vec<u8> {
        let mut out = Vec::new();
        match &self.crossing {
            Crossing::Layouts(table) | Crossing::Places(table) => {
                for (name, layout) in *table {
                    out.extend(name.as_bytes());
                    for one in *layout {
                        out.extend(one.words());
                    }
                }
            }
            Crossing::Nested => {
                out.extend(ENTRY.as_bytes());
                out.extend(OUTER_BUFFER_INDEX.to_le_bytes());
                for (name, index) in NESTED_PLACES {
                    out.extend(name.as_bytes());
                    out.extend(index.to_le_bytes());
                }
                for (name, offset) in NESTED_OFFSETS {
                    out.extend(name.as_bytes());
                    out.extend(offset.to_le_bytes());
                }
            }
        }
        if let Crossing::Places(_) = &self.crossing {
            for (name, offset, element) in PLACES {
                out.extend(name.as_bytes());
                out.extend(offset.to_le_bytes());
                out.extend(element.unwrap_or(u64::MAX).to_le_bytes());
            }
        }
        out
    }

    /// One encoder's facts, its arguments' positions and its argument buffer's changed words,
    /// under `at`.
    ///
    /// Every encoder of the family goes through this, so a cell differs from another cell by the
    /// layout and the offset it was given and by nothing of the harness.
    fn encoded(
        side: &mut dyn Side,
        sites: &mut Sites,
        at: &str,
        encoder: Handle,
        layout: &[ArgumentDescriptorSpec],
        place: (u64, Option<u64>),
        held: &Resources,
    ) -> Result<(), Abort> {
        let facts = match side.argument_encoder_facts(encoder, LABEL) {
            Ok(facts) => facts,
            Err(failure) => {
                sites.push(Node::leaf(
                    format!("{at}/facts"),
                    Observed::value(failure.recorded()),
                ));
                return Err(Abort {
                    at: at.to_owned(),
                    failure,
                });
            }
        };
        sites.push(facts.node(format!("{at}/facts")));
        let arguments = made(
            sites,
            &format!("{at}/buffer"),
            side.buffer(&vec![BLANK; BUFFER_BYTES]),
        )?;
        made(
            sites,
            &format!("{at}/setArgumentBuffer"),
            side.set_argument_buffer(encoder, arguments, place.0, place.1),
        )?;
        for (step, write) in writes(layout, held.texture, held.sampler, held.buffer)
            .iter()
            .enumerate()
        {
            made(
                sites,
                &format!("{at}/write/{step}-{}", write.call()),
                side.encode_argument(encoder, write),
            )?;
        }
        for (index, bytes) in constants(layout) {
            call(
                sites,
                &format!("{at}/constant-offset/{index}"),
                side.constant_data_offset(encoder, index),
                |offset| Value::Uint(*offset),
            )?;
            made(
                sites,
                &format!("{at}/constant-write/{index}"),
                side.encode_argument(encoder, &ArgumentWrite::Constant { index, bytes }),
            )?;
        }
        // The buffer's own bytes never enter the transcript: they are addresses and resource IDs.
        // Which words stopped holding the pattern is the layout and nothing of the host's.
        let encoded = call(
            sites,
            &format!("{at}/read-back"),
            side.read_buffer(arguments, 0, BUFFER_BYTES),
            |_| Value::Absent,
        )?;
        sites.push(changed_node(
            format!("{at}/changed-words"),
            changed_words(&encoded, WORDS),
            0,
        ));
        Ok(())
    }
}

/// The resources a case's writes name, made once and named by every cell.
struct Resources {
    texture: Handle,
    sampler: Handle,
    buffer: Handle,
}

impl Case for CrossingCase {
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
        Ok(vec![
            Input {
                name: "air".to_owned(),
                digest: Digest::sha256(fixture.text(RUNTIME_TEXT)?.as_bytes()),
            },
            Input {
                name: "air-typed".to_owned(),
                digest: Digest::sha256(fixture.text(METAL_TEXT)?.as_bytes()),
            },
            Input::bytes("layouts", &self.layout_words()),
            Input::bytes("constant", &CONSTANT),
        ])
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        // The one texture, sampler and buffer every cell's writes name. A texture argument's word
        // is a resource ID and a buffer argument's an address, so which object it is changes
        // nothing a case compares: what the mask says is that a word stopped holding the pattern.
        let texture = made(
            sites,
            "texture",
            side.texture_2d_in_role(&spec::texture(), &spec::texels()),
        )?;
        let sampler = made(sites, "sampler", side.sampler(&spec::sampler()))?;
        let buffer = made(sites, "buffer", side.buffer(&vec![0u8; BUFFER_BYTES]))?;
        let held = Resources {
            texture,
            sampler,
            buffer,
        };

        match &self.crossing {
            Crossing::Layouts(table) => {
                for (at, layout) in *table {
                    let encoder = made(
                        sites,
                        &format!("{at}/encoder"),
                        side.argument_encoder_with_arguments(layout),
                    )?;
                    Self::encoded(side, sites, at, encoder, layout, (0, None), &held)?;
                }
            }
            Crossing::Places(table) => {
                for (layout_name, layout) in *table {
                    for (place_name, offset, element) in PLACES {
                        // One encoder a cell, not one encoder reset between cells: an encoder
                        // keeps the argument buffer it was last given, so a shared one would make
                        // each cell's answer depend on the cell before it.
                        let at = format!("{layout_name}/{place_name}");
                        let encoder = made(
                            sites,
                            &format!("{at}/encoder"),
                            side.argument_encoder_with_arguments(layout),
                        )?;
                        Self::encoded(
                            side,
                            sites,
                            &at,
                            encoder,
                            layout,
                            (*offset, *element),
                            &held,
                        )?;
                    }
                }
            }
            Crossing::Nested => {
                let stop = |at: &str, why: String| Abort {
                    at: at.to_owned(),
                    failure: Failure::named("FixtureNotRead", why),
                };
                let loaded = self
                    .fixture()
                    .loaded()
                    .map_err(|why| stop("library", why))?;
                let library = made(sites, "library", side.library(&loaded.source()))?;
                let outer = made(
                    sites,
                    "outer/encoder",
                    side.argument_encoder(library, ENTRY, OUTER_BUFFER_INDEX),
                )?;
                // The outer layout's own statement, and the word it writes for the member that is
                // itself an argument buffer: a nested argument buffer is a buffer to the outer
                // encoder, so what goes in at index 15 is a buffer's address like any other, and
                // the outer mask says it landed at the member's own offset and nowhere else.
                Self::encoded(
                    side,
                    sites,
                    "outer",
                    outer,
                    &[ArgumentDescriptorSpec::new(DataType::Pointer, NESTED_INDEX)],
                    (0, None),
                    &held,
                )?;
                // Which member of the layout is a nested argument buffer, asked of every member
                // kind the layout holds and of an index past its end. Metal's answer is an encoder
                // or nil, and nil is an answer: a side that refused here would be refusing what
                // plain Metal hands a program.
                let mut nested_handle = None;
                for (at, index) in NESTED_PLACES {
                    let answer = call(
                        sites,
                        &format!("member/{at}"),
                        side.nested_argument_encoder(outer, *index),
                        |nested| Value::Bool(nested.is_some()),
                    )?;
                    if *index == NESTED_INDEX {
                        nested_handle = answer;
                    }
                }
                // The nested encoder's own layout, and where its one texture lands in its own
                // buffer at each offset: the inner composition, which the outer encoder's offset
                // has nothing to do with.
                let nested = nested_handle.ok_or_else(|| Abort {
                    at: "nested/encoder".to_owned(),
                    failure: Failure::named(
                        "NoNestedEncoder",
                        format!("no encoder for argument {NESTED_INDEX}"),
                    ),
                })?;
                for (at, offset) in NESTED_OFFSETS {
                    Self::encoded(
                        side,
                        sites,
                        at,
                        nested,
                        &[ArgumentDescriptorSpec::new(DataType::Texture, 0)],
                        (*offset, None),
                        &held,
                    )?;
                }
            }
        }
        Ok(())
    }
}

/// The one texture and sampler the family's writes name, described where a reader meets them.
mod spec {
    use crate::case::{SamplerSpec, TextureRole, TextureSpec};
    use metal2vulkan::metal::{PixelFormat, SamplerAddressMode, SamplerMinMagFilter};

    /// The texture's side: small, because nothing samples it. What a texture argument contributes
    /// is a resource ID.
    const SIDE: u32 = 4;

    /// The texture every texture argument of the family is written with.
    pub fn texture() -> TextureSpec {
        TextureSpec::new(
            PixelFormat::RGBA8Unorm,
            [SIDE, SIDE],
            SIDE * 4,
            TextureRole::Read,
        )
    }

    /// Its bytes, which nothing reads: a constant pattern, so the case's input fingerprint holds
    /// one.
    pub fn texels() -> Vec<u8> {
        vec![0x40; (SIDE * SIDE * 4) as usize]
    }

    /// The sampler state every sampler argument is written with: able to live in an argument
    /// buffer at all, which Metal gives a state no resource ID without.
    pub fn sampler() -> SamplerSpec {
        SamplerSpec::filtered(SamplerMinMagFilter::Nearest)
            .addressed(SamplerAddressMode::ClampToEdge)
            .for_argument_buffer()
    }
}
