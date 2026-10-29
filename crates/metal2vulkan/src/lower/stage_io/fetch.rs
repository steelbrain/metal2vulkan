//! Vertex fetch: the in-shader (pulled) fetch of the attributes a vertex function's projection
//! pulls, and the format encodings both fetch paths share.
//!
//! **Projection.** The one projection is `pipeline::link::vertex::vertex_fetch_projection`: it
//! takes each attribute's declared class, and the compile API and link both call it. The projection
//! is part of the function's lowering inputs and its compile key; the rest of the vertex descriptor
//! is link-time `VertexInputState`.
//!
//! **Pulled fetch**. A pulled attribute is not an `Input` variable. The function reads it once, in
//! its first block before any instruction of the body (`super::emit_entry`), from the storage
//! buffer `ResourceSlot{Vertex, Buffer, buffer_index}` (a word view, `memory::views`, or its device
//! address; see *Source buffers* below), at the Metal displacement
//!
//! ```text
//! element · VertexBufferStride(buffer) + offset
//! ```
//!
//! from the bind offset (a view adds the bind residual `SlotBindOffset(buffer)`), in wrapping
//! `u32` arithmetic, where `element` is the step's element index:
//!
//! - `PerVertex`: `VertexIndex` (it includes the base vertex in Vulkan as `[[vertex_id]]` does in
//!   Metal); in the de-indexed and private list forms, whose draws have no indices, the vertex
//!   the draw's index names ([`Step::CapturedVertex`], `super::CapturedIndex`), as
//!   `[[vertex_id]]` reads it there;
//! - `PerInstance` with rate 1: `InstanceIndex` (which includes the base instance in both APIs);
//! - `PerInstance` with rate `n > 1`: `base + (InstanceIndex - base) / n`, the element the
//!   fixed-function divisor fetches, where `base` is the `BaseInstance` builtin with
//!   `shaderDrawParameters` and the second lane of `PushField::BaseVertexInstance` without it (the
//!   `BaseVertexInstance` mechanism). This is Metal's element as MoltenVK relies on it: it maps
//!   Vulkan's divisor to Metal's step rate and the first instance to Metal's base instance
//!   unchanged while advertising `supportsNonZeroFirstInstance` (not observed on Metal here);
//! - `Constant`: no element, the attribute's offset alone. Metal fetches the first element
//!   whatever the base instance: MoltenVK emulates Vulkan's zero divisor with Metal's constant
//!   step by offsetting the bound buffer by `firstInstance · stride`.
//!
//! The bytes are the format's store size, read through the access family's leaf composition
//! (`memory::leaf`) on the byte path, because neither the stride nor the bind residual has an
//! alignment the shader can rely on: Metal's minimum alignment of a layout stride and an
//! attribute offset is 1 byte on Apple5 and later (4 bytes only on Apple2-Apple4; Metal feature
//! set tables), and the projection pulls the attributes whose offset or stride fixed-function input
//! could not align (and every attribute where the profile has no buffer robustness). They are
//! contained like any descriptor access (the normative in-shader check when the profile has no
//! buffer robustness, reading at offset 0 and yielding zero bytes out of bounds) and decoded
//! ([`Encoding`]): integers zero- or sign-extended by the format's signedness, normalized integers
//! by the device's product of the code and its maximum's reciprocal (`c · float(1 / (2^b - 1))`,
//! signed `max(c · float(1 / (2^(b-1) - 1)), -1)`, within one `f32` ulp of Metal's correctly
//! rounded quotient: `PrecisionNote::DeviceVertexConversion`), `half` bits by the device's
//! widening (`narrow::from_bits`), and the packed 10:10:10:2, 11:11:10 float, and
//! 9:9:9:5 shared-exponent formats by their layouts. Missing components read as `(0, 0, 0, 1)` and
//! extra ones are dropped. A float format then rounds to a narrow float parameter; an integer
//! format is masked to a narrow integer parameter. Across the classes ([`class_match`]): an
//! unnormalized 8- or 16-bit integer format read into a float parameter converts each extended
//! component to its value as a float (exact) and rounds it to a narrow one, as Metal does; a 32-bit
//! one is an invalid descriptor, as Metal refuses the pipeline; and a float-class format read into
//! an integer parameter is not a conversion whose Metal meaning is established, so the projection
//! refuses it (`NotImplemented`) and the pulled read refuses it again for a hand-built projection.
//!
//! **Kernel stage inputs**. A kernel's `[[stage_in]]` attribute is fetched the same way, once at
//! entry, from the kernel buffer slot `ResourceSlot{Kernel, Buffer, buffer_index}` its stage-input
//! descriptor names, at `element · stride + offset` with the layout's stride: a fixed one compiled
//! in, or for a layout of `MTLBufferLayoutStrideDynamic` the one the encoder binds the buffer with,
//! from `PushField::StageInStride(buffer_index)`, where `element` is the step's ([`Step::Grid`],
//! [`Step::Indexed`], [`Step::Constant`]; `super::stage_in` plans them).
//!
//! **Source buffers**. This family records which Metal buffer indices the fetch reads
//! ([`super::StageIoPlan::buffers`]); their slots are the access family's (`lower::access::pulled`,
//! chosen by `access::finish_slots` with the function's own slots). A source buffer whose index is
//! also an accessed buffer argument shares that argument's slot, widened to a `Whole` storage slot:
//! Metal binds one buffer at one offset per vertex-buffer index, so both reach the same memory. The
//! argument may be written or reached by atomics: the fetch precedes every access of the body, so
//! it reads what Metal's attribute holds. An argument window outside the first 2³² bytes above the
//! bind offset is refused. Any source buffer may be reached by device address (a `BufferRangeLimit`
//! variant, the per-stage descriptor-limit fallback, or a shared argument's wide pointers or typed
//! atomics): the fetch then reads the Metal displacement `element · stride + offset` through the
//! slot's root entry instead of adding the bind residual.

use crate::error::{DescriptorError, Error, Feature, LowerFamily, Unsupported};
use crate::lower::access::pulled::{self, SourceRead};
use crate::lower::arith::{narrow, ops, FloatFmt, Form, Held};
use crate::lower::dispatch;
use crate::lower::emit::EmitCx;
use crate::lower::select::SelectCx;
use crate::memory::contain::{self, Check};
use crate::memory::leaf::{k32, Leaf, Path};
use crate::metal::{VertexFormat, VertexStepFunction};
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::lower_inputs::VertexFetch;
use crate::pipeline::stage_io::{PulledInput, StageBuiltin};
use crate::pipeline::sym::{PushField, SlotClass, SlotRef, Stage};
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::error::{BResult, BuildError, Dangling};
use crate::spirv::ids::Value as SValue;
use crate::spirv::ir::{BinOp, Contraction, ConvOp, GlslOp, IoDir, MemAccess};

use super::plan::{builtin, Shape};

fn not_lowered() -> Error {
    Unsupported::not_implemented(Feature::Lowering(LowerFamily::StageIo)).into()
}

/// How the bytes of a Metal vertex format encode its components (little-endian; the packed
/// formats are named from the least significant bit).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Encoding {
    /// `count` components of `bits` bits (8, 16, or 32), in order.
    Int {
        bits: u8,
        count: u8,
        signed: bool,
        normalized: bool,
    },
    /// `count` IEEE binary16 components.
    Half { count: u8 },
    /// `count` IEEE binary32 components.
    Float { count: u8 },
    /// Four unsigned normalized bytes in memory order b, g, r, a (`UChar4Normalized_BGRA`).
    Bgra8Unorm,
    /// One word: x, y, z in 10-bit fields from bit 0, w in the top 2 bits; normalized.
    Rgb10A2 { signed: bool },
    /// One word: unsigned 11-bit floats x (bit 0) and y (bit 11), an unsigned 10-bit float z
    /// (bit 22); each has a 5-bit exponent of bias 15.
    Rg11B10Float,
    /// One word: 9-bit mantissas x, y, z from bit 0 and a shared 5-bit exponent at bit 27;
    /// component `m · 2^(e - 24)`.
    Rgb9E5Float,
}

impl Encoding {
    /// The format's store size in bytes.
    pub fn bytes(self) -> u32 {
        match self {
            Encoding::Int { bits, count, .. } => u32::from(bits / 8) * u32::from(count),
            Encoding::Half { count } => 2 * u32::from(count),
            Encoding::Float { count } => 4 * u32::from(count),
            Encoding::Bgra8Unorm
            | Encoding::Rgb10A2 { .. }
            | Encoding::Rg11B10Float
            | Encoding::Rgb9E5Float => 4,
        }
    }

    /// Whether a component is a float narrower than 32 bits, which the fetch widens to `float`:
    /// the `half` formats, and the 11- and 10-bit fields of `FloatRG11B10`, whose exponent field
    /// is all ones for a NaN. The 9:9:9:5 shared-exponent format decodes every code to a finite
    /// value, so none of its components is ever a NaN.
    pub fn widens_narrow_float(self) -> bool {
        matches!(self, Encoding::Half { .. } | Encoding::Rg11B10Float)
    }

    /// Whether each component is a code divided by its own maximum: the 8- and 16-bit
    /// `Normalized` formats of either sign, `UChar4Normalized_BGRA` and the two
    /// `1010102Normalized` words.
    pub fn is_normalized(self) -> bool {
        matches!(
            self,
            Encoding::Int {
                normalized: true,
                ..
            } | Encoding::Bgra8Unorm
                | Encoding::Rgb10A2 { .. }
        )
    }

    /// Whether the components are floats (every format except the unnormalized integers).
    pub fn is_float(self) -> bool {
        !matches!(
            self,
            Encoding::Int {
                normalized: false,
                ..
            }
        )
    }

    /// The number of components the format has.
    #[cfg(test)]
    pub fn components(self) -> u8 {
        match self {
            Encoding::Int { count, .. } | Encoding::Half { count } | Encoding::Float { count } => {
                count
            }
            Encoding::Bgra8Unorm | Encoding::Rgb10A2 { .. } => 4,
            Encoding::Rg11B10Float | Encoding::Rgb9E5Float => 3,
        }
    }

    /// The raw form the bytes are read in: `u32`-held lanes of the component width, or one word
    /// for a packed format.
    fn raw(self) -> Form {
        let (bits, lanes) = match self {
            Encoding::Int { bits, count, .. } => (bits, count),
            Encoding::Half { count } => (16, count),
            Encoding::Float { count } => (32, count),
            Encoding::Bgra8Unorm => (8, 4),
            Encoding::Rgb10A2 { .. } | Encoding::Rg11B10Float | Encoding::Rgb9E5Float => (32, 1),
        };
        Form {
            held: Held::Int {
                bits,
                container: 32,
            },
            lanes,
        }
    }

    /// The leaf the bytes are read as (the byte path: see the module docs).
    fn leaf(self) -> Leaf {
        Leaf {
            form: self.raw(),
            bytes: self.bytes(),
            path: Path::Bytes,
        }
    }
}

/// How a format's components meet the class of the attribute that reads them: the shader
/// declares every attribute `float` or `uint`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ClassMatch {
    /// The format's own class: a float-class format (float, half, normalized, packed) into a
    /// float attribute, an unnormalized integer format into an integer one.
    Same,
    /// An unnormalized 8- or 16-bit integer format into a float attribute: each component reads
    /// as its integer value, zero- or sign-extended by the format's signedness, as a float, which
    /// is exact, and the missing ones as `(0, 0, 0, 1)`; a `half` attribute rounds that value to
    /// nearest even (65520 and above to infinity). Metal's answer, measured on an Apple M4.
    IntToFloat,
}

/// How `encoding`, the format of attribute `attribute`, meets an attribute of the float class
/// (`float`) or the integer one ([`ClassMatch`]): a 32-bit integer format into a float attribute is
/// `InvalidDescriptor(VertexAttribute)`, as Metal refuses the pipeline ("cannot be read using" the
/// format); a float-class format into an integer attribute is not a conversion whose Metal meaning
/// is established, `Unsupported{Lowering(StageIo), NotImplemented}`.
pub fn class_match(encoding: Encoding, float: bool, attribute: u32) -> Result<ClassMatch, Error> {
    match encoding {
        e if e.is_float() == float => Ok(ClassMatch::Same),
        Encoding::Int {
            bits: 8 | 16,
            normalized: false,
            ..
        } => Ok(ClassMatch::IntToFloat),
        Encoding::Int {
            normalized: false, ..
        } => Err(Error::InvalidDescriptor(DescriptorError::VertexAttribute {
            attribute,
        })),
        _ => Err(not_lowered()),
    }
}

/// The encoding of `format`; `Invalid` is a placeholder, not a format.
pub fn encoding(format: VertexFormat) -> Result<Encoding, Error> {
    use VertexFormat as V;
    let int = |bits, count, signed, normalized| Encoding::Int {
        bits,
        count,
        signed,
        normalized,
    };
    Ok(match format {
        V::Invalid => {
            return Err(Error::InvalidDescriptor(
                DescriptorError::PlaceholderEnumValue {
                    ty: VertexFormat::METAL_NAME,
                    value: format.as_raw(),
                },
            ))
        }
        V::UChar => int(8, 1, false, false),
        V::UChar2 => int(8, 2, false, false),
        V::UChar3 => int(8, 3, false, false),
        V::UChar4 => int(8, 4, false, false),
        V::Char => int(8, 1, true, false),
        V::Char2 => int(8, 2, true, false),
        V::Char3 => int(8, 3, true, false),
        V::Char4 => int(8, 4, true, false),
        V::UCharNormalized => int(8, 1, false, true),
        V::UChar2Normalized => int(8, 2, false, true),
        V::UChar3Normalized => int(8, 3, false, true),
        V::UChar4Normalized => int(8, 4, false, true),
        V::CharNormalized => int(8, 1, true, true),
        V::Char2Normalized => int(8, 2, true, true),
        V::Char3Normalized => int(8, 3, true, true),
        V::Char4Normalized => int(8, 4, true, true),
        V::UChar4NormalizedBGRA => Encoding::Bgra8Unorm,
        V::UShort => int(16, 1, false, false),
        V::UShort2 => int(16, 2, false, false),
        V::UShort3 => int(16, 3, false, false),
        V::UShort4 => int(16, 4, false, false),
        V::Short => int(16, 1, true, false),
        V::Short2 => int(16, 2, true, false),
        V::Short3 => int(16, 3, true, false),
        V::Short4 => int(16, 4, true, false),
        V::UShortNormalized => int(16, 1, false, true),
        V::UShort2Normalized => int(16, 2, false, true),
        V::UShort3Normalized => int(16, 3, false, true),
        V::UShort4Normalized => int(16, 4, false, true),
        V::ShortNormalized => int(16, 1, true, true),
        V::Short2Normalized => int(16, 2, true, true),
        V::Short3Normalized => int(16, 3, true, true),
        V::Short4Normalized => int(16, 4, true, true),
        V::Half => Encoding::Half { count: 1 },
        V::Half2 => Encoding::Half { count: 2 },
        V::Half3 => Encoding::Half { count: 3 },
        V::Half4 => Encoding::Half { count: 4 },
        V::Float => Encoding::Float { count: 1 },
        V::Float2 => Encoding::Float { count: 2 },
        V::Float3 => Encoding::Float { count: 3 },
        V::Float4 => Encoding::Float { count: 4 },
        V::Int => int(32, 1, true, false),
        V::Int2 => int(32, 2, true, false),
        V::Int3 => int(32, 3, true, false),
        V::Int4 => int(32, 4, true, false),
        V::UInt => int(32, 1, false, false),
        V::UInt2 => int(32, 2, false, false),
        V::UInt3 => int(32, 3, false, false),
        V::UInt4 => int(32, 4, false, false),
        V::Int1010102Normalized => Encoding::Rgb10A2 { signed: true },
        V::UInt1010102Normalized => Encoding::Rgb10A2 { signed: false },
        V::FloatRG11B10 => Encoding::Rg11B10Float,
        V::FloatRGB9E5 => Encoding::Rgb9E5Float,
    })
}

/// The element index of a pulled attribute's step.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Step {
    /// `VertexIndex`, the stage-IO object `index`.
    PerVertex { index: usize },
    /// `InstanceIndex` (object `index`) divided by `rate` from `base` (`None` for rate 1).
    PerInstance {
        index: usize,
        rate: u32,
        base: Option<Base>,
    },
    /// The attribute's offset alone.
    Constant,
    /// A kernel stage-in step by thread position in the grid on `axis` (0 `x`, 1 `y`): the stage-in
    /// region's origin plus the position divided by `rate`, clamped to the region's last element
    /// (`RuntimeFact::StageInRegion`).
    Grid { axis: u8, rate: u32 },
    /// A kernel stage-in step through the index buffer on `axis`: the index at the thread position
    /// clamped to the region's last element, plus the region's origin.
    Indexed { axis: u8, index: IndexRead },
    /// A per-vertex step in the de-indexed or private list form: the vertex the draw's index names
    /// (`super::CapturedIndex`), read once at entry.
    CapturedVertex,
}

/// The index read of an indexed kernel stage-in step: `bytes` (2 or 4) at the clamped thread
/// position times `bytes` in the index buffer at Metal kernel buffer index `buffer`, an unsigned
/// integer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct IndexRead {
    pub buffer: u32,
    pub bytes: u32,
}

impl IndexRead {
    /// The leaf the index is read as, on the byte path (a bind offset has no alignment the
    /// shader may assume).
    fn leaf(self) -> Leaf {
        Leaf {
            form: Form {
                held: Held::Int {
                    bits: (self.bytes * 8) as u8,
                    container: 32,
                },
                lanes: 1,
            },
            bytes: self.bytes,
            path: Path::Bytes,
        }
    }
}

/// Where a pulled read's layout stride comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Stride {
    /// The `u32` push field the binder writes: a vertex layout's
    /// (`PushField::VertexBufferStride`), static or per bind, or a kernel stage-in layout's
    /// per-bind one (`PushField::StageInStride`).
    Push(PushField),
    /// A kernel stage-in layout's fixed one, from the descriptor.
    Static(u32),
}

/// Where a divided per-instance step reads the base instance.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Base {
    /// The `BaseInstance` builtin, stage-IO object `index` (`shaderDrawParameters`).
    Builtin { index: usize },
    /// The second lane of `PushField::BaseVertexInstance`.
    Push,
}

/// One planned pulled read: a parameter's value, fetched once at entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PulledRead {
    pub attribute: u32,
    pub encoding: Encoding,
    /// The Metal buffer index of the source buffer.
    pub buffer: u32,
    /// The attribute's byte offset in an element.
    pub offset: u32,
    pub step: Step,
    pub stride: Stride,
    /// The parameter's form.
    pub form: Form,
    pub check: Check,
}

impl PulledRead {
    /// An upper estimate of the SPIR-V words the fetch emits, once per attribute at entry (its uses
    /// read the held value and add nothing): the byte-path leaf, the offset and containment
    /// (through a view or the root entry of a device address, whichever the access family chooses),
    /// and the decode of four lanes at the cost of the bound on a half's decode (the dearest one)
    /// plus the conversion to the parameter's form.
    pub fn words(&self) -> u32 {
        let lane =
            64 + narrow::bits_words(Form {
                held: Held::Float(FloatFmt::Half),
                lanes: 1,
            }) + narrow::round_words(Form {
                lanes: 1,
                ..self.form
            });
        // A kernel step derives the thread position in the grid (the region arithmetic), clamps it
        // to the stage-in region, reads its index the way it reads the attribute, and adds the
        // region's origin.
        let element = match self.step {
            Step::Grid { .. } => GRID_POSITION_WORDS + REGION_AXIS_WORDS + 16,
            Step::Indexed { index, .. } => {
                GRID_POSITION_WORDS + REGION_AXIS_WORDS + index.leaf().cost() + 176
            }
            // The index read is the plan's, counted once (`super::CAPTURED_INDEX_WORDS`).
            Step::PerVertex { .. }
            | Step::PerInstance { .. }
            | Step::Constant
            | Step::CapturedVertex => 0,
        };
        self.encoding.leaf().cost() + 4 * lane + 160 + element
    }
}

/// An upper bound of the words that derive the thread position in the grid in a kernel: the
/// `DispatchRegion` read and unpacking, the threadgroup position, and the position in the
/// threadgroup, unflattened in the flattened tiers (`dispatch::KernelDecl::value`).
const GRID_POSITION_WORDS: u32 = 400;

/// An upper bound of the words of a step's clamp to the stage-in region (`region_axis`): the
/// region's push-field read, its origin and size, the size less 1, the `UMin`, and the origin's
/// add, with their constants.
const REGION_AXIS_WORDS: u32 = 64;

/// The Metal slot of source buffer `buffer` of a function of `stage`: a vertex buffer of a
/// vertex function (post-tessellation ones included), a kernel buffer of a kernel.
pub fn slot(stage: Stage, buffer: u32) -> SlotRef {
    SlotRef {
        stage,
        class: SlotClass::Buffer,
        index: buffer,
    }
}

/// Plan the pulled read of attribute `attribute` (fetched by `fetch`) into a parameter of form
/// `form`: its step and the push fields and runtime facts it needs. Its source buffer is recorded
/// in [`super::StageIoPlan::buffers`]; the access family chooses that buffer's slot.
pub(super) fn select(
    cx: &mut SelectCx<'_>,
    attribute: u32,
    fetch: VertexFetch,
    form: Form,
) -> Result<PulledRead, Error> {
    let VertexFetch::Pulled {
        format,
        buffer_index: buffer,
        offset,
        step_function,
        step_rate,
    } = fetch
    else {
        return Err(cx.invariant("a fixed-function attribute planned as pulled"));
    };
    let encoding = encoding(format)?;
    if form.is_bool() {
        return Err(not_lowered());
    }
    class_match(encoding, form.is_float(), attribute)?;
    decode_notes(cx, encoding, form);
    let s = slot(Stage::Vertex, buffer);
    let draw_parameters = cx.profile.fields().draw_parameters;
    // The de-indexed and private list forms' per-vertex element is the vertex its index names.
    let captured = super::plan::captured_index_type(cx);
    if let (VertexStepFunction::PerVertex, Some(ty)) = (step_function, captured) {
        super::plan::captured_index(cx, ty)?;
    }
    let sp = &mut cx.families.stage_io;
    let step = match step_function {
        VertexStepFunction::PerVertex if captured.is_some() => Step::CapturedVertex,
        VertexStepFunction::PerVertex => Step::PerVertex {
            index: builtin(
                sp,
                crate::spv::BuiltIn::VertexIndex,
                IoDir::Input,
                Shape::U32,
                StageBuiltin::VertexIndex,
            ),
        },
        VertexStepFunction::PerInstance => {
            if step_rate == 0 {
                return Err(Error::InvalidDescriptor(DescriptorError::VertexStepRate {
                    step_rate,
                }));
            }
            let index = builtin(
                sp,
                crate::spv::BuiltIn::InstanceIndex,
                IoDir::Input,
                Shape::U32,
                StageBuiltin::InstanceIndex,
            );
            let base = match (step_rate, draw_parameters) {
                (1, _) => None,
                (_, true) => Some(Base::Builtin {
                    index: builtin(
                        sp,
                        crate::spv::BuiltIn::BaseInstance,
                        IoDir::Input,
                        Shape::U32,
                        StageBuiltin::BaseInstance,
                    ),
                }),
                (_, false) => Some(Base::Push),
            };
            Step::PerInstance {
                index,
                rate: step_rate,
                base,
            }
        }
        VertexStepFunction::Constant => Step::Constant,
        _ => return Err(Unsupported::not_implemented(Feature::Tessellation).into()),
    };
    if let Step::PerInstance { base: Some(_), .. } = step {
        cx.facts.insert(RuntimeFact::BaseVertexInstance);
    }
    if let Step::PerInstance {
        base: Some(Base::Push),
        ..
    } = step
    {
        cx.push.insert(PushField::BaseVertexInstance);
    }
    if step != Step::Constant {
        cx.push.insert(PushField::VertexBufferStride(buffer));
        cx.facts
            .insert(RuntimeFact::VertexBufferStride { layout: buffer });
    }
    // A view's read adds the bind residual; `access::finish_slots` drops the field for a source
    // buffer it reaches by device address, as for any slot.
    cx.push.insert(PushField::SlotBindOffset(s));
    cx.facts
        .insert(RuntimeFact::VertexFetch { attr: attribute });
    // The access family chooses the source buffer's slot, its facts, and its descriptor.
    let sp = &mut cx.families.stage_io;
    sp.buffers.insert(buffer);
    // Link checks the projection it pipelines against this record.
    if let Err(at) = sp
        .interface
        .pulled
        .binary_search_by_key(&attribute, |p| p.index)
    {
        let input = PulledInput::new(attribute, super::plan::io_type(form), fetch);
        sp.interface.pulled.insert(at, input);
    }
    Ok(PulledRead {
        attribute,
        encoding,
        buffer,
        offset,
        step,
        stride: Stride::Push(PushField::VertexBufferStride(buffer)),
        form,
        check: contain::encoding(cx.profile),
    })
}

/// The element index `base + (instance - base) / rate` (wrapping `u32`; `rate > 0`).
pub fn divided(
    fb: &mut FunctionBuilder<'_>,
    instance: SValue,
    base: SValue,
    rate: u32,
) -> BResult<SValue> {
    let d = fb.binary(BinOp::ISub, instance, base)?;
    let r = k32(fb, rate)?;
    let q = fb.binary(BinOp::UDiv, d, r)?;
    fb.binary(BinOp::IAdd, base, q)
}

/// The Metal displacement `element · stride + offset` from the bind offset (wrapping `u32`); the
/// offset alone without an element.
pub fn displacement(
    fb: &mut FunctionBuilder<'_>,
    element: Option<(SValue, SValue)>,
    offset: u32,
) -> BResult<SValue> {
    let at = match element {
        Some((e, stride)) => fb.binary(BinOp::IMul, e, stride)?,
        None => return k32(fb, offset),
    };
    if offset == 0 {
        return Ok(at);
    }
    let o = k32(fb, offset)?;
    fb.binary(BinOp::IAdd, at, o)
}

/// Fetch `encoding` through `read` (a source-buffer read of the format's store size,
/// `access::pulled`) and decode it into form `form` (see the module docs).
pub fn fetch_from(
    fb: &mut FunctionBuilder<'_>,
    read: SourceRead,
    encoding: Encoding,
    form: Form,
) -> BResult<SValue> {
    let raw = read.load(fb, &encoding.leaf())?;
    decode(fb, encoding, form, raw)
}

fn u32_scalar() -> Form {
    Form {
        held: Held::Int {
            bits: 32,
            container: 32,
        },
        lanes: 1,
    }
}

fn f32_scalar() -> Form {
    Form {
        held: Held::Float(FloatFmt::F32),
        lanes: 1,
    }
}

/// Bits `[at, at + width)` of the `u32` `w`.
fn field(fb: &mut FunctionBuilder<'_>, w: SValue, at: u32, width: u32) -> BResult<SValue> {
    let x = if at == 0 {
        w
    } else {
        let s = k32(fb, at)?;
        fb.binary(BinOp::ShiftRightLogical, w, s)?
    };
    if at + width >= 32 {
        return Ok(x);
    }
    let m = k32(fb, (1u32 << width) - 1)?;
    fb.binary(BinOp::BitwiseAnd, x, m)
}

/// The float of a normalized `bits`-bit field `c` (zero-extended in a `u32`, `bits <= 16`, as in
/// every normalized vertex format): the device's product of the code and the reciprocal of its
/// maximum, `float(c) · float(1 / (2^b - 1))`, or signed `max(float(c) · float(1 / (2^(b-1) - 1)),
/// -1)` on the sign-extended code, rounded once to `float` (no contraction). Metal's fetch answers
/// the correctly rounded quotient `c / (2^b - 1)` (signed `max(c / (2^(b-1) - 1), -1)`), and an
/// `Int1010102Normalized` 10-bit field's this same product; the product is within one `f32` ulp of
/// the quotient, exact at 0, at the maximum and at -1 ([`decode_notes`] records
/// `PrecisionNote::DeviceVertexConversion` where it can differ).
fn normalized(fb: &mut FunctionBuilder<'_>, c: SValue, bits: u8, signed: bool) -> BResult<SValue> {
    let (v, d, op) = if signed {
        let field = u32_scalar().with(Held::Int {
            bits,
            container: 32,
        });
        let v = ops::sext(fb, field, c)?;
        (v, (1u32 << (bits - 1)) - 1, ConvOp::ConvertSToF)
    } else {
        (c, (1u32 << bits) - 1, ConvOp::ConvertUToF)
    };
    let ft = ops::ty(fb, f32_scalar())?;
    let f = fb.convert(op, v, ft)?;
    let r = ops::splat(fb, f32_scalar(), u64::from((1.0f32 / d as f32).to_bits()))?;
    let q = fb.binary_with(BinOp::FMul, f, r, Contraction::Forbidden)?;
    if !signed {
        return Ok(q);
    }
    let minus_one = ops::fsplat(fb, f32_scalar(), -1.0)?;
    fb.glsl(GlslOp::FMax, &[q, minus_one])
}

/// The `f32` of half bits `h`: the device's widening of them (`narrow::from_bits`), the `half`'s
/// value but for a NaN's bits and, on a device that may flush 16-bit denormals, a denormal, which
/// the module records (`PrecisionNote::NativeNanEncoding`, `NativeHalfConversion`).
fn half(fb: &mut FunctionBuilder<'_>, h: SValue) -> BResult<SValue> {
    narrow::from_bits(
        fb,
        Form {
            held: Held::Float(FloatFmt::Half),
            lanes: 1,
        },
        h,
    )
}

/// One decoded component: a `u32` (integer formats) or an `f32` (float formats).
fn components(fb: &mut FunctionBuilder<'_>, e: Encoding, raw: SValue) -> BResult<Vec<SValue>> {
    let lanes = usize::from(e.raw().lanes);
    let mut lane = Vec::with_capacity(lanes);
    for i in 0..lanes {
        lane.push(if lanes == 1 {
            raw
        } else {
            fb.composite_extract(raw, &[i as u32])?
        });
    }
    let mut out = Vec::with_capacity(4);
    match e {
        Encoding::Int {
            bits,
            signed,
            normalized: norm,
            ..
        } => {
            for c in lane {
                out.push(if norm {
                    normalized(fb, c, bits, signed)?
                } else if signed {
                    let f = u32_scalar().with(Held::Int {
                        bits,
                        container: 32,
                    });
                    ops::sext(fb, f, c)?
                } else {
                    c
                });
            }
        }
        Encoding::Half { .. } => {
            for c in lane {
                out.push(half(fb, c)?);
            }
        }
        Encoding::Float { .. } => {
            let ft = ops::ty(fb, f32_scalar())?;
            for c in lane {
                out.push(fb.convert(ConvOp::Bitcast, c, ft)?);
            }
        }
        Encoding::Bgra8Unorm => {
            for i in [2usize, 1, 0, 3] {
                let c = *lane
                    .get(i)
                    .ok_or(BuildError::Dangling(Dangling::Value(raw)))?;
                out.push(normalized(fb, c, 8, false)?);
            }
        }
        Encoding::Rgb10A2 { signed } => {
            for (at, width) in [(0, 10u8), (10, 10), (20, 10), (30, 2)] {
                let c = field(fb, raw, at, u32::from(width))?;
                out.push(normalized(fb, c, width, signed)?);
            }
        }
        Encoding::Rg11B10Float => {
            // An unsigned 11-bit (10-bit) float is the half whose bits are it shifted left by 4
            // (5): the same exponent field and bias, the mantissa's low bits zero.
            for (at, width, up) in [(0, 11, 4), (11, 11, 4), (22, 10, 5)] {
                let c = field(fb, raw, at, width)?;
                let s = k32(fb, up)?;
                let h = fb.binary(BinOp::ShiftLeftLogical, c, s)?;
                out.push(half(fb, h)?);
            }
        }
        Encoding::Rgb9E5Float => {
            // `2^(e - 24)` is the normal `f32` with biased exponent `e + 103`; the product with
            // a 9-bit mantissa is exact.
            let e = field(fb, raw, 27, 5)?;
            let k = k32(fb, 103)?;
            let e = fb.binary(BinOp::IAdd, e, k)?;
            let k = k32(fb, 23)?;
            let e = fb.binary(BinOp::ShiftLeftLogical, e, k)?;
            let ft = ops::ty(fb, f32_scalar())?;
            let scale = fb.convert(ConvOp::Bitcast, e, ft)?;
            for at in [0, 9, 18] {
                let m = field(fb, raw, at, 9)?;
                let m = fb.convert(ConvOp::ConvertUToF, m, ft)?;
                out.push(fb.binary_with(BinOp::FMul, m, scale, Contraction::Forbidden)?);
            }
        }
    }
    Ok(out)
}

/// Record the precision notes of the module's decode ([`decode`]) of an attribute of `encoding`
/// read into `form`, at every pulled read: a vertex function's ([`select`]), a kernel's
/// `[[stage_in]]` attribute's (`super::stage_in`) and a post-tessellation function's
/// (`lower::tessellation`), on every profile.
///
/// - `PrecisionNote::DeviceNanEncoding` where the encoding is a narrow float's
///   ([`Encoding::widens_narrow_float`]) read as `float` lanes: Metal's fetch answers its
///   canonical NaN, sign and payload dropped, where the decode, the device's widening of the 16-,
///   11- or 10-bit field as a half's bits, answers a NaN of the device's bits. A `half` read
///   stored unchanged keeps the field's own bits, as Metal's does.
/// - `PrecisionNote::DeviceVertexConversion` where the decode's normalized product
///   ([`normalized`]) differs from Metal's fetch at some code ([`product_differs`]): the product
///   is the module's arithmetic, not the fixed-function conversion
///   `DriverDeviation::MetalNormalizedVertexFetch` measures.
pub(crate) fn decode_notes(cx: &mut SelectCx<'_>, encoding: Encoding, form: Form) {
    use crate::pipeline::artifact::PrecisionNote;
    if encoding.widens_narrow_float() && form.float_fmt() == Some(FloatFmt::F32) {
        cx.precision.insert(PrecisionNote::DeviceNanEncoding);
    }
    if product_differs(encoding, form) {
        cx.precision.insert(PrecisionNote::DeviceVertexConversion);
    }
}

/// Whether the normalized product ([`normalized`]) of a field of `encoding`, read into `form`,
/// differs from Metal's fetch at some code. Into a `float`, every normalized encoding's but
/// `Int1010102Normalized`'s, whose 10-bit fields Metal computes as this product and whose 2-bit
/// field's product is exact: an 8-bit unsigned field's at 126 of its 256 codes, one ulp high (127
/// as `0x3EFEFF00` where Metal reads `0x3EFEFEFF`), and one ulp toward zero a signed 8-bit
/// field's at 16 codes, a 10-bit unsigned one's at 24, and a 16-bit one's at 512 unsigned and
/// 1,536 signed. Into a `half`, a 16-bit field's, its `float` rounded to `half` again, one `half`
/// ulp from Metal's quotient rounded once at codes 65455 and 65519 unsigned, 32759 and -32759
/// signed; an 8-, 10- or 2-bit field's product rounds to Metal's `half` at every code.
fn product_differs(encoding: Encoding, form: Form) -> bool {
    match form.float_fmt() {
        Some(FloatFmt::F32) => {
            encoding.is_normalized() && encoding != Encoding::Rgb10A2 { signed: true }
        }
        Some(FloatFmt::Half) => matches!(
            encoding,
            Encoding::Int {
                bits: 16,
                normalized: true,
                ..
            }
        ),
        // A `bfloat`'s product rounds to the quotient's `bfloat` at every code.
        _ => false,
    }
}

/// The value of form `form` of the raw bytes `raw` of `encoding` (see the module docs).
pub fn decode(
    fb: &mut FunctionBuilder<'_>,
    encoding: Encoding,
    form: Form,
    raw: SValue,
) -> BResult<SValue> {
    let mut comps = components(fb, encoding, raw)?;
    // An integer format read into a float parameter (`ClassMatch::IntToFloat`): each extended
    // component's value as an `f32`, exact for the 8- and 16-bit formats it admits.
    if let (Encoding::Int { signed, .. }, true) =
        (encoding, !encoding.is_float() && form.is_float())
    {
        let op = if signed {
            ConvOp::ConvertSToF
        } else {
            ConvOp::ConvertUToF
        };
        let ft = ops::ty(fb, f32_scalar())?;
        for c in &mut comps {
            *c = fb.convert(op, *c, ft)?;
        }
    }
    let scalar = Form { lanes: 1, ..form };
    let exact_half =
        matches!(encoding, Encoding::Half { .. }) && form.float_fmt() == Some(FloatFmt::Half);
    let mut lanes = Vec::with_capacity(usize::from(form.lanes));
    for i in 0..usize::from(form.lanes) {
        let v = if form.is_float() {
            let v = match comps.get(i) {
                Some(v) => *v,
                None => ops::fsplat(fb, f32_scalar(), if i == 3 { 1.0 } else { 0.0 })?,
            };
            // A half component read into a half parameter is already a half value.
            if exact_half {
                v
            } else {
                narrow::round(fb, scalar, v)?
            }
        } else {
            let v = match comps.get(i) {
                Some(v) => *v,
                None => k32(fb, u32::from(i == 3))?,
            };
            ops::mask(fb, scalar, v)?
        };
        lanes.push(v);
    }
    if form.lanes == 1 {
        return lanes
            .first()
            .copied()
            .ok_or(BuildError::Dangling(Dangling::Value(raw)));
    }
    let t = ops::ty(fb, form)?;
    fb.composite_construct(t, &lanes)
}

/// Load the `u32` builtin of stage-IO object `index`.
pub(super) fn load_builtin(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    index: usize,
) -> BResult<SValue> {
    let g = super::global(cx, index)?;
    let p = fb.module().global_value(g)?;
    fb.load(p, MemAccess::default())
}

/// Component `axis` of the kernel's thread position in the grid.
fn grid_position(cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>, axis: u8) -> BResult<SValue> {
    let p = dispatch::grid_position(cx, fb)?;
    fb.composite_extract(p, &[u32::from(axis)])
}

/// The stage-in region's origin on `axis` and its last element's position from it, its size
/// less 1: the step's position clamps to it, and a size of 0 wraps to `u32::MAX`, which clamps
/// nothing, as Metal reads such a region.
fn region_axis(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    axis: u8,
) -> BResult<(SValue, SValue)> {
    let region = cx.push_field(fb, PushField::StageInRegion)?;
    let origin = fb.composite_extract(region, &[0, u32::from(axis)])?;
    let size = fb.composite_extract(region, &[1, u32::from(axis)])?;
    let one = k32(fb, 1)?;
    Ok((origin, fb.binary(BinOp::ISub, size, one)?))
}

/// The value of pulled read `r`, fetched in the open block.
pub(super) fn emit(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    r: &PulledRead,
) -> BResult<SValue> {
    let element = match r.step {
        Step::Grid { axis, rate } => {
            let at = grid_position(cx, fb, axis)?;
            let stepped = match rate {
                1 => at,
                rate => {
                    let r = k32(fb, rate)?;
                    fb.binary(BinOp::UDiv, at, r)?
                }
            };
            let (origin, last) = region_axis(cx, fb, axis)?;
            let within = fb.glsl(GlslOp::UMin, &[stepped, last])?;
            Some(fb.binary(BinOp::IAdd, origin, within)?)
        }
        Step::Indexed { axis, index } => {
            let at = grid_position(cx, fb, axis)?;
            let (origin, last) = region_axis(cx, fb, axis)?;
            let within = fb.glsl(GlslOp::UMin, &[at, last])?;
            let n = k32(fb, index.bytes)?;
            let m = fb.binary(BinOp::IMul, within, n)?;
            let read = pulled::prepare(cx, fb, index.buffer, m, index.bytes, r.check)?;
            let element = read.load(fb, &index.leaf())?;
            Some(fb.binary(BinOp::IAdd, element, origin)?)
        }
        Step::PerVertex { index }
        | Step::PerInstance {
            index, base: None, ..
        } => Some(load_builtin(cx, fb, index)?),
        Step::PerInstance {
            index,
            rate,
            base: Some(base),
        } => {
            let instance = load_builtin(cx, fb, index)?;
            let base = match base {
                Base::Builtin { index } => load_builtin(cx, fb, index)?,
                Base::Push => {
                    let v = cx.push_field(fb, PushField::BaseVertexInstance)?;
                    fb.composite_extract(v, &[1])?
                }
            };
            Some(divided(fb, instance, base, rate)?)
        }
        Step::Constant => None,
        Step::CapturedVertex => Some(super::captured_vertex(cx, fb)?),
    };
    let element = match (element, r.stride) {
        (Some(e), Stride::Push(field)) => Some((e, cx.push_field(fb, field)?)),
        (Some(e), Stride::Static(s)) => Some((e, k32(fb, s)?)),
        (None, _) => None,
    };
    let m = displacement(fb, element, r.offset)?;
    let read = pulled::prepare(cx, fb, r.buffer, m, r.encoding.bytes(), r.check)?;
    fetch_from(fb, read, r.encoding, r.form)
}

/// A reference decoder for the tests: the component values the decode answers for `bytes` of
/// `encoding` (missing ones `(0, 0, 0, 1)`), each as an `f64` for a float format or the
/// sign-extended `u32` bits for an integer one. They are Metal's but a normalized field's, which
/// is the device's product of its code and its maximum's reciprocal ([`normalized`]), within one
/// `f32` ulp of Metal's correctly rounded quotient.
#[cfg(test)]
pub(crate) fn reference(encoding: Encoding, bytes: &[u8]) -> [f64; 4] {
    use crate::lower::arith::legal::f16_to_f64;
    let byte = |i: usize| u32::from(bytes.get(i).copied().unwrap_or(0));
    let word = |i: usize| byte(i) | byte(i + 1) << 8 | byte(i + 2) << 16 | byte(i + 3) << 24;
    let sx = |v: u32, bits: u32| -> i64 {
        if bits >= 32 {
            i64::from(v as i32)
        } else {
            let s = 32 - bits;
            i64::from(((v << s) as i32) >> s)
        }
    };
    let norm = |v: u32, bits: u32, signed: bool| -> f64 {
        if signed {
            let d = ((1u32 << (bits - 1)) - 1) as f32;
            f64::from(((sx(v, bits) as f32) * (1.0 / d)).max(-1.0))
        } else {
            let d = ((1u64 << bits) - 1) as f32;
            f64::from((v as f32) * (1.0 / d))
        }
    };
    let small_float = |v: u32, mbits: u32| -> f64 {
        let e = (v >> mbits) & 0x1F;
        let m = f64::from(v & ((1 << mbits) - 1));
        let scale = f64::from(1u32 << mbits);
        match e {
            0 => m / scale * 2f64.powi(-14),
            31 if m == 0.0 => f64::INFINITY,
            31 => f64::NAN,
            _ => (1.0 + m / scale) * 2f64.powi(e as i32 - 15),
        }
    };
    let mut out = [0.0, 0.0, 0.0, 1.0];
    match encoding {
        Encoding::Int {
            bits,
            count,
            signed,
            normalized: n,
        } => {
            let w = u32::from(bits / 8);
            for (i, o) in out.iter_mut().enumerate().take(usize::from(count)) {
                let mut v = 0u32;
                for b in 0..w {
                    v |= byte(i * w as usize + b as usize) << (8 * b);
                }
                *o = if n {
                    norm(v, u32::from(bits), signed)
                } else if signed {
                    f64::from(sx(v, u32::from(bits)) as u32)
                } else {
                    f64::from(v)
                };
            }
        }
        Encoding::Half { count } => {
            for (i, o) in out.iter_mut().enumerate().take(usize::from(count)) {
                *o = f16_to_f64((byte(2 * i) | byte(2 * i + 1) << 8) as u16);
            }
        }
        Encoding::Float { count } => {
            for (i, o) in out.iter_mut().enumerate().take(usize::from(count)) {
                *o = f64::from(f32::from_bits(word(4 * i)));
            }
        }
        Encoding::Bgra8Unorm => {
            for (o, i) in out.iter_mut().zip([2usize, 1, 0, 3]) {
                *o = norm(byte(i), 8, false);
            }
        }
        Encoding::Rgb10A2 { signed } => {
            let w = word(0);
            for (o, (at, width)) in out.iter_mut().zip([(0, 10), (10, 10), (20, 10), (30, 2)]) {
                *o = norm((w >> at) & ((1 << width) - 1), width, signed);
            }
        }
        Encoding::Rg11B10Float => {
            let w = word(0);
            out[0] = small_float(w & 0x7FF, 6);
            out[1] = small_float((w >> 11) & 0x7FF, 6);
            out[2] = small_float(w >> 22, 5);
        }
        Encoding::Rgb9E5Float => {
            let w = word(0);
            let e = (w >> 27) as i32;
            for (i, o) in out.iter_mut().take(3).enumerate() {
                *o = f64::from((w >> (9 * i)) & 0x1FF) * 2f64.powi(e - 24);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests;
