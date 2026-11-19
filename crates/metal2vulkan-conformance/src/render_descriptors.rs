//! What a **render pipeline descriptor** answers of itself: `MTLRenderPipelineDescriptor`'s own
//! surface, and what `reset` puts back.
//!
//! [`crate::pipeline_facts`] is the pipeline **state**'s side of this: a state is immutable and
//! answers what its descriptor decided. This is the descriptor, before any state exists. The two
//! are different questions, and the second is where Metal's **defaults** live: a program that sets
//! six fields of thirty-four gets Metal's value for the other twenty-eight, and every one of them
//! reaches a pipeline we compile.
//!
//! The runtime has no descriptor object. Its
//! [`RenderPipelineDescriptor`](metal2vulkan::pipeline::RenderPipelineDescriptor) is a plain
//! struct a consumer fills, so its defaults are the ones
//! [`RenderPipelineDescriptor::new`](metal2vulkan::pipeline::RenderPipelineDescriptor::new) writes
//! and nothing in the tree ties any of them to a Metal run: the factor of 16, the pow-2
//! partitioning, the clockwise winding, the call-stack depth of 1, the unspecified topology, the
//! `One`/`Zero`/`Add` blend of an attachment nobody blended. They were read off Apple's
//! documentation. A case over this family is what holds them to Metal.
//!
//! `reset` is the same claim from the other end. Metal's descriptor is mutable and `reset` puts
//! every field back; the runtime's counterpart is `RenderPipelineDescriptor::new` again (the
//! coverage map says so), because a fresh struct **is** the reset. So a case that sets every field
//! away from its default, resets, and reads everything back says "the defaults a program gets after
//! a reset are the defaults it gets at the start", on both sides at once.
//!
//! # What the site observes, and what it cannot
//!
//! Four of Metal's getters answer **objects a program supplies**, and the runtime takes each one
//! somewhere other than the descriptor:
//!
//! - `binaryArchives` is `PipelineOptions.binary_archives`;
//! - `vertexPreloadedLibraries` and `fragmentPreloadedLibraries` are each stage's
//!   `FunctionDescriptor.libraries`;
//! - `vertexLinkedFunctions` and `fragmentLinkedFunctions` are each stage's
//!   `FunctionDescriptor.linked`.
//!
//! For those the site records **how many** of each the descriptor holds, never the objects. The
//! four per-stage ones are also the four sites [`Shape::Reset`] leaves out, for the measured reason
//! [`SuppliedLists`] states. Metal
//! answers nil for a list nothing set and the runtime answers an empty `LinkedFunctions` or
//! `DynamicLibraries`, which mean the same thing — nothing linked, nothing preloaded — and only one
//! of the two shapes can be a number both sides state. A count is that shape, and `nil` is counted
//! as none. **This family therefore compares those five items at their default alone**: it says the
//! lists start empty and that a reset empties them, and says nothing about a descriptor carrying a
//! real archive or dynamic library. That is a narrower case, not a loosened one — no value is
//! widened anywhere — and the wider case waits on cases for `MTLBinaryArchive` and
//! `MTLDynamicLibrary`, which no case of the suite reaches at all.
//!
//! `vertexDescriptor` is recorded **slot by slot** rather than as an object that is there or not,
//! for the same reason and on the same evidence: Metal's `init` leaves a non-nil, empty vertex
//! descriptor (measured here, at the oracle), the runtime's field is `None` until a program sets
//! one, and both mean no attribute is fetched.
//!
//! `shaderValidation` is deliberately absent: Apple's enumeration has no counterpart in the
//! runtime, and a getter whose answer nothing of ours can state is not a comparison.
//!
//! # Why every case of this family is safe
//!
//! Nothing here is compiled, allocated or submitted. A case makes one library, one or two
//! functions of it, and a descriptor it never hands to `newRenderPipelineStateWithDescriptor:`. No
//! command buffer is made, so no case can fault the device, trip its watchdog or reach a queue.
//!
//! The one hazard a descriptor's own surface carries is a **setter** Metal's validation stops, and
//! there is exactly one in this family's reach: `setLabel:` with nil ends Metal's process (the
//! compute and render pipeline sides both say so where they set it). No shape here sets a nil
//! label — the label is set only where the case states one — and the default shape reads the nil
//! Metal's own `init` left.

use crate::case::field;
use crate::transcript::{Node, NodeName, Observed, Value};
use metal2vulkan::metal;

/// The colour attachments every shape reads back.
///
/// Three, and the middle one is the point: the set shape states attachments 0 and 2, so index 1 is
/// a slot nothing touched and its answer is Metal's default for an attachment, read beside two
/// that were set. Metal's array holds eight and `objectAtIndexedSubscript:` past the last asserts,
/// so the bound is a constant of the family rather than a case's choice.
pub const READ_COLORS: u32 = 3;

/// The buffer-mutability slots every shape reads back, per stage.
///
/// Three, for the same reason: the set shape states vertex buffers 0 and 2 and fragment buffer 1,
/// so each stage has both a stated slot and an untouched one in the record.
pub const READ_BUFFERS: u32 = 3;

/// The vertex-descriptor attribute and layout slots a shape with a vertex descriptor reads back.
pub const READ_VERTEX_SLOTS: u32 = 2;

/// The name a site gives an ordinal neither side's enumerations have a name for.
const UNNAMED: &str = "unnamed";

/// Which of the three shapes a case states.
///
/// The three other pipeline descriptors' families state the same three
/// ([`crate::compute_descriptors`], [`crate::tile_descriptors`], [`crate::mesh_descriptors`]), and
/// share this one enumeration rather than declaring four of their own: what a shape means — nothing
/// set, everything set, everything set and reset — is the same question of every descriptor Metal
/// has, and a family that spelled it again would be a second vocabulary for one contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shape {
    /// The stage functions and nothing else: Metal's default for every other field.
    Defaults,
    /// Every field [`Away`] states, each away from Metal's default, read straight back.
    Set,
    /// Every field set, then `reset`, then the stage functions set again — which must read as
    /// [`Shape::Defaults`] does.
    Reset,
}

impl Shape {
    /// The ordinal a case's input fingerprint carries.
    pub const fn word(self) -> u8 {
        match self {
            Shape::Defaults => 0,
            Shape::Set => 1,
            Shape::Reset => 2,
        }
    }

    /// Whether the shape sets [`Away`]'s fields before reading.
    pub const fn sets(self) -> bool {
        matches!(self, Shape::Set | Shape::Reset)
    }

    /// Whether `reset` is sent after the setting and before the read.
    pub const fn resets(self) -> bool {
        matches!(self, Shape::Reset)
    }
}

/// One colour attachment the set shape states, as both sides set it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ColorAway {
    /// The attachment's index in the descriptor's array, below [`READ_COLORS`].
    pub index: u32,
    pub pixel_format: metal::PixelFormat,
    pub blending_enabled: bool,
    pub source_rgb: metal::BlendFactor,
    pub destination_rgb: metal::BlendFactor,
    pub rgb_operation: metal::BlendOperation,
    pub source_alpha: metal::BlendFactor,
    pub destination_alpha: metal::BlendFactor,
    pub alpha_operation: metal::BlendOperation,
    pub write_mask: metal::ColorWriteMask,
}

/// One vertex attribute the set shape states.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttributeAway {
    pub index: u32,
    pub format: metal::VertexFormat,
    pub offset: u32,
    pub buffer_index: u32,
}

/// One vertex buffer layout the set shape states.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LayoutAway {
    pub index: u32,
    pub stride: u32,
    pub step_function: metal::VertexStepFunction,
    pub step_rate: u32,
}

/// Every field the set shape moves away from Metal's default, each to a value no other field of
/// the record holds, so a side that wired one getter to its neighbour answers a different number
/// rather than passing.
///
/// There is one of these ([`AWAY`]) and the two setting shapes share it: what a case means is "the
/// descriptor with every field stated", and two tables of stated values would make the reset shape
/// compare against something the default shape never saw.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Away {
    pub label: &'static str,
    /// `setSampleCount:`, Metal's **deprecated** spelling of the raster sample count. The set shape
    /// states this one and not `setRasterSampleCount:`, and reads **both** getters back: that the
    /// two are one field is a claim the runtime makes by holding a single
    /// `raster_sample_count`, and this is what checks it.
    pub sample_count: u32,
    pub alpha_to_coverage: bool,
    pub alpha_to_one: bool,
    pub rasterization_enabled: bool,
    pub max_vertex_amplification_count: u32,
    pub colors: &'static [ColorAway],
    pub depth_format: metal::PixelFormat,
    pub stencil_format: metal::PixelFormat,
    pub input_primitive_topology: metal::PrimitiveTopologyClass,
    pub tessellation_partition_mode: metal::TessellationPartitionMode,
    pub max_tessellation_factor: u32,
    pub tessellation_factor_scale_enabled: bool,
    pub tessellation_factor_format: metal::TessellationFactorFormat,
    pub tessellation_control_point_index_type: metal::TessellationControlPointIndexType,
    pub tessellation_factor_step_function: metal::TessellationFactorStepFunction,
    pub tessellation_output_winding_order: metal::Winding,
    /// `vertexBuffers[i].mutability`, by index.
    pub vertex_buffers: &'static [(u32, metal::Mutability)],
    /// `fragmentBuffers[i].mutability`, by index.
    pub fragment_buffers: &'static [(u32, metal::Mutability)],
    pub support_indirect_command_buffers: bool,
    pub support_adding_vertex_binary_functions: bool,
    pub support_adding_fragment_binary_functions: bool,
    pub max_vertex_call_stack_depth: u32,
    pub max_fragment_call_stack_depth: u32,
    pub attributes: &'static [AttributeAway],
    pub layouts: &'static [LayoutAway],
}

/// The one table of stated values, each away from Metal's default.
///
/// The numbers are chosen apart: a sample count of 4, an amplification count of 2, a tessellation
/// factor of 12, a vertex call-stack depth of 3 and a fragment one of 5, an attribute offset of 8
/// in buffer 1, a layout stride of 32 at a step rate of 2. No two of them are equal, so a side
/// that answered one getter from another's field reads wrong.
///
/// `tessellation_factor_format` is `Half` although that is also Metal's default: `MTLTessellationFactorFormat`
/// has exactly one value in the macOS 27 SDK, so there is no second value to state. The getter is
/// still read, and what it says is that the one value is what both sides hold.
///
/// Rasterization is **disabled** here and a fragment function is set beside it, which Metal's
/// pipeline creation refuses. No shape of this family creates a pipeline, so the descriptor is
/// never handed anything that would check it; what the field means to the suite is that a program's
/// `setRasterizationEnabled:NO` is what the descriptor then answers.
pub const AWAY: Away = Away {
    label: "conformance-render-descriptor",
    sample_count: 4,
    alpha_to_coverage: true,
    alpha_to_one: true,
    rasterization_enabled: false,
    max_vertex_amplification_count: 2,
    colors: &[
        ColorAway {
            index: 0,
            pixel_format: metal::PixelFormat::RGBA8Unorm,
            blending_enabled: true,
            source_rgb: metal::BlendFactor::SourceAlpha,
            destination_rgb: metal::BlendFactor::OneMinusSourceAlpha,
            rgb_operation: metal::BlendOperation::ReverseSubtract,
            source_alpha: metal::BlendFactor::DestinationColor,
            destination_alpha: metal::BlendFactor::BlendColor,
            alpha_operation: metal::BlendOperation::Max,
            write_mask: metal::ColorWriteMask::RED,
        },
        ColorAway {
            index: 2,
            pixel_format: metal::PixelFormat::RGBA16Float,
            blending_enabled: false,
            source_rgb: metal::BlendFactor::One,
            destination_rgb: metal::BlendFactor::Zero,
            rgb_operation: metal::BlendOperation::Add,
            source_alpha: metal::BlendFactor::One,
            destination_alpha: metal::BlendFactor::Zero,
            alpha_operation: metal::BlendOperation::Add,
            write_mask: metal::ColorWriteMask::BLUE,
        },
    ],
    depth_format: metal::PixelFormat::Depth32Float,
    stencil_format: metal::PixelFormat::Stencil8,
    input_primitive_topology: metal::PrimitiveTopologyClass::Triangle,
    tessellation_partition_mode: metal::TessellationPartitionMode::FractionalOdd,
    max_tessellation_factor: 12,
    tessellation_factor_scale_enabled: true,
    tessellation_factor_format: metal::TessellationFactorFormat::Half,
    tessellation_control_point_index_type: metal::TessellationControlPointIndexType::UInt32,
    tessellation_factor_step_function:
        metal::TessellationFactorStepFunction::PerPatchAndPerInstance,
    tessellation_output_winding_order: metal::Winding::CounterClockwise,
    vertex_buffers: &[
        (0, metal::Mutability::Immutable),
        (2, metal::Mutability::Mutable),
    ],
    fragment_buffers: &[(1, metal::Mutability::Immutable)],
    support_indirect_command_buffers: true,
    support_adding_vertex_binary_functions: true,
    support_adding_fragment_binary_functions: true,
    max_vertex_call_stack_depth: 3,
    max_fragment_call_stack_depth: 5,
    attributes: &[AttributeAway {
        index: 0,
        format: metal::VertexFormat::Float2,
        offset: 8,
        buffer_index: 1,
    }],
    layouts: &[LayoutAway {
        index: 1,
        stride: 32,
        step_function: metal::VertexStepFunction::PerInstance,
        step_rate: 2,
    }],
};

impl Away {
    /// The words a case's input fingerprint covers: every stated field, so a table that moves is a
    /// change to every case of the family.
    pub fn words(&self) -> Vec<u8> {
        let mut out = self.label.as_bytes().to_vec();
        for word in [
            self.sample_count,
            self.max_vertex_amplification_count,
            self.max_tessellation_factor,
            self.max_vertex_call_stack_depth,
            self.max_fragment_call_stack_depth,
            self.depth_format.as_raw(),
            self.stencil_format.as_raw(),
            self.input_primitive_topology.as_raw(),
            self.tessellation_partition_mode.as_raw(),
            self.tessellation_factor_format.as_raw(),
            self.tessellation_control_point_index_type.as_raw(),
            self.tessellation_factor_step_function.as_raw(),
            self.tessellation_output_winding_order.as_raw(),
        ] {
            out.extend(word.to_le_bytes());
        }
        for flag in [
            self.alpha_to_coverage,
            self.alpha_to_one,
            self.rasterization_enabled,
            self.tessellation_factor_scale_enabled,
            self.support_indirect_command_buffers,
            self.support_adding_vertex_binary_functions,
            self.support_adding_fragment_binary_functions,
        ] {
            out.push(u8::from(flag));
        }
        for color in self.colors {
            for word in [
                color.index,
                color.pixel_format.as_raw(),
                color.source_rgb.as_raw(),
                color.destination_rgb.as_raw(),
                color.rgb_operation.as_raw(),
                color.source_alpha.as_raw(),
                color.destination_alpha.as_raw(),
                color.alpha_operation.as_raw(),
                color.write_mask.as_raw(),
            ] {
                out.extend(word.to_le_bytes());
            }
            out.push(u8::from(color.blending_enabled));
        }
        for (index, mutability) in self.vertex_buffers.iter().chain(self.fragment_buffers) {
            out.extend(index.to_le_bytes());
            out.extend(mutability.as_raw().to_le_bytes());
        }
        for attribute in self.attributes {
            for word in [
                attribute.index,
                attribute.format.as_raw(),
                attribute.offset,
                attribute.buffer_index,
            ] {
                out.extend(word.to_le_bytes());
            }
        }
        for layout in self.layouts {
            for word in [
                layout.index,
                layout.stride,
                layout.step_function.as_raw(),
                layout.step_rate,
            ] {
                out.extend(word.to_le_bytes());
            }
        }
        out
    }
}

/// What one case makes and reads: the stage functions, and which of the three shapes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DescriptorSpec<'a> {
    /// `vertexFunction`'s entry in the case's library.
    pub vertex: &'a str,
    /// `fragmentFunction`'s entry, where the shape sets one.
    pub fragment: Option<&'a str>,
    pub shape: Shape,
}

impl<'a> DescriptorSpec<'a> {
    /// The descriptor of `vertex` alone, in `shape`.
    pub const fn new(vertex: &'a str, shape: Shape) -> Self {
        Self {
            vertex,
            fragment: None,
            shape,
        }
    }

    /// The descriptor with `fragment` set too.
    #[must_use]
    pub const fn with_fragment(mut self, fragment: &'a str) -> Self {
        self.fragment = Some(fragment);
        self
    }

    /// The words a case's input fingerprint covers: the functions, the shape, and — where the shape
    /// sets anything — the whole stated table.
    pub fn words(&self) -> Vec<u8> {
        let mut out = self.vertex.as_bytes().to_vec();
        out.push(0);
        out.extend(self.fragment.unwrap_or("").as_bytes());
        out.push(self.shape.word());
        if self.shape.sets() {
            out.extend(AWAY.words());
        }
        out
    }
}

/// What one colour attachment of the descriptor answers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ColorFacts {
    pub pixel_format: u32,
    pub blending_enabled: bool,
    pub source_rgb: u32,
    pub destination_rgb: u32,
    pub rgb_operation: u32,
    pub source_alpha: u32,
    pub destination_alpha: u32,
    pub alpha_operation: u32,
    pub write_mask: u32,
}

/// What one vertex attribute slot of the descriptor's vertex descriptor answers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttributeFacts {
    pub format: u32,
    pub offset: u64,
    pub buffer_index: u64,
}

/// What one buffer layout slot of the descriptor's vertex descriptor answers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LayoutFacts {
    pub stride: u64,
    pub step_function: u32,
    pub step_rate: u64,
}

/// What the descriptor's vertex descriptor answers, slot by slot.
///
/// [`VertexDescriptorFacts::empty`] is what a side with no vertex descriptor at all answers, which
/// is Metal's own empty one slot for slot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VertexDescriptorFacts {
    /// Slots 0 up to [`READ_VERTEX_SLOTS`].
    pub attributes: Vec<AttributeFacts>,
    pub layouts: Vec<LayoutFacts>,
}

impl VertexDescriptorFacts {
    /// Every slot at Metal's default: format `Invalid` at offset 0 in buffer 0, and a layout of
    /// stride 0 stepping per vertex at a rate of 1. What a nil or absent vertex descriptor is.
    pub fn empty(slots: u32) -> Self {
        Self {
            attributes: (0..slots)
                .map(|_| AttributeFacts {
                    format: metal::VertexFormat::Invalid.as_raw(),
                    offset: 0,
                    buffer_index: 0,
                })
                .collect(),
            layouts: (0..slots)
                .map(|_| LayoutFacts {
                    stride: 0,
                    step_function: metal::VertexStepFunction::PerVertex.as_raw(),
                    step_rate: 1,
                })
                .collect(),
        }
    }
}

/// The per-stage lists of objects a program supplies, by how many each holds.
///
/// [`Shape::Reset`] reads none of them, and the reason is Metal's own behaviour plus a third-party
/// defect: **`reset` leaves `vertexPreloadedLibraries` nil** where `init` leaves a non-nil empty
/// array (measured on this family's oracle), and `objc2-metal` 0.3.2 declares the property
/// non-null, so its generated accessor panics on the nil rather than answering it. Nothing of the
/// runtime's answers that question anyway — it has no descriptor object, so a program's own `Vec`
/// is the counterpart and there is no reset to observe — so the reset shape leaves the four sites
/// out rather than reading a nil it cannot record and our side cannot have.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SuppliedLists {
    /// `vertexPreloadedLibraries`' length; a nil list is 0.
    pub vertex_preloaded_libraries: u64,
    /// `fragmentPreloadedLibraries`' length.
    pub fragment_preloaded_libraries: u64,
    /// `vertexLinkedFunctions`' four lengths; a list it answers nil for is 0.
    pub vertex_linked_functions: LinkedCounts,
    /// `fragmentLinkedFunctions`' four lengths.
    pub fragment_linked_functions: LinkedCounts,
}

/// How many objects one of the descriptor's supplied-object lists holds. A nil list is none, for the
/// reason the module's prose gives.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LinkedCounts {
    pub functions: u64,
    pub private_functions: u64,
    pub binary_functions: u64,
    pub groups: u64,
}

/// What a render pipeline descriptor answers of itself. Each field is one Metal getter's answer,
/// an enumeration by its raw ordinal so an ordinal neither side names still compares.
#[derive(Clone, Debug, PartialEq)]
pub struct RenderDescriptorFacts {
    /// `label`; `None` is Metal's nil.
    pub label: crate::case::Field<Option<String>>,
    /// `vertexFunction`'s own `name`; `None` where the descriptor holds none.
    pub vertex_function: Option<String>,
    /// `fragmentFunction`'s own `name`.
    pub fragment_function: Option<String>,
    /// `vertexDescriptor`, slot by slot.
    ///
    /// The **slots** and not the object's presence. Metal's own `init` leaves a non-nil, empty
    /// vertex descriptor on the descriptor — measured on this family's oracle, not assumed — while
    /// the runtime's `vertex_descriptor` is `None` until a program sets one, and the two mean the
    /// same thing: no attribute is fetched by the fixed-function stage. A nil or absent descriptor
    /// is therefore recorded as every slot at its default, which is what both sides can state.
    pub vertex_descriptor: VertexDescriptorFacts,
    /// `sampleCount`, Metal's deprecated spelling.
    pub sample_count: u64,
    /// `rasterSampleCount`.
    pub raster_sample_count: u64,
    pub alpha_to_coverage_enabled: bool,
    pub alpha_to_one_enabled: bool,
    pub rasterization_enabled: bool,
    pub max_vertex_amplification_count: u64,
    /// `colorAttachments[i]` for `i` below [`READ_COLORS`].
    pub colors: Vec<ColorFacts>,
    pub depth_attachment_pixel_format: u32,
    pub stencil_attachment_pixel_format: u32,
    pub input_primitive_topology: u32,
    pub tessellation_partition_mode: u32,
    pub max_tessellation_factor: u64,
    pub tessellation_factor_scale_enabled: bool,
    pub tessellation_factor_format: u32,
    pub tessellation_control_point_index_type: u32,
    pub tessellation_factor_step_function: u32,
    pub tessellation_output_winding_order: u32,
    /// `vertexBuffers[i].mutability` for `i` below [`READ_BUFFERS`].
    pub vertex_buffers: Vec<u32>,
    /// `fragmentBuffers[i].mutability` for `i` below [`READ_BUFFERS`].
    pub fragment_buffers: Vec<u32>,
    pub support_indirect_command_buffers: bool,
    /// `binaryArchives`' length; a nil list is 0.
    pub binary_archives: crate::case::Field<u64>,
    /// The four per-stage lists of supplied objects, which [`Shape::Reset`] does not read
    /// ([`SuppliedLists`] says why).
    pub supplied: Option<SuppliedLists>,
    pub support_adding_vertex_binary_functions: bool,
    pub support_adding_fragment_binary_functions: bool,
    pub max_vertex_call_stack_depth: u64,
    pub max_fragment_call_stack_depth: u64,
}

/// An enumeration as a site records it: the Metal enumerator name the translator's own enumeration
/// gives the ordinal, beside the ordinal itself, in the shape [`crate::function_facts`] records
/// one. An ordinal no enumeration of ours names is recorded all the same, under [`UNNAMED`], so a
/// value Apple added in a release we have not seen compares rather than refusing.
pub(crate) fn ordinal<T: TryFrom<u32>>(raw: u32, name: impl Fn(T) -> &'static str) -> Value {
    let named = T::try_from(raw).ok().map_or(UNNAMED, name);
    Value::fields([
        ("name", Value::Lines(vec![named.to_owned()])),
        ("raw", Value::Uint(u64::from(raw))),
    ])
}

/// A bit set as a site records it: the raw bits, which is the whole of what a mask means.
fn bits(raw: u32) -> Value {
    Value::Uint(u64::from(raw))
}

/// `Option<String>` as a site records it.
pub(crate) fn optional(held: &Option<String>) -> Value {
    match held {
        Some(held) => Value::Variant {
            name: "Some".to_owned(),
            payload: Box::new(Value::Lines(vec![held.clone()])),
        },
        None => Value::unit_variant("None"),
    }
}

impl LinkedCounts {
    /// The four lengths as a site records them.
    pub(crate) fn value(&self) -> Value {
        Value::fields([
            ("functions", Value::Uint(self.functions)),
            ("private_functions", Value::Uint(self.private_functions)),
            ("binary_functions", Value::Uint(self.binary_functions)),
            ("groups", Value::Uint(self.groups)),
        ])
    }
}

impl ColorFacts {
    /// One attachment's answers as a site records them.
    pub(crate) fn value(&self) -> Value {
        Value::fields([
            (
                "pixel_format",
                ordinal(self.pixel_format, metal::PixelFormat::name),
            ),
            ("blending_enabled", Value::Bool(self.blending_enabled)),
            (
                "source_rgb_blend_factor",
                ordinal(self.source_rgb, metal::BlendFactor::name),
            ),
            (
                "destination_rgb_blend_factor",
                ordinal(self.destination_rgb, metal::BlendFactor::name),
            ),
            (
                "rgb_blend_operation",
                ordinal(self.rgb_operation, metal::BlendOperation::name),
            ),
            (
                "source_alpha_blend_factor",
                ordinal(self.source_alpha, metal::BlendFactor::name),
            ),
            (
                "destination_alpha_blend_factor",
                ordinal(self.destination_alpha, metal::BlendFactor::name),
            ),
            (
                "alpha_blend_operation",
                ordinal(self.alpha_operation, metal::BlendOperation::name),
            ),
            ("write_mask", bits(self.write_mask)),
        ])
    }
}

impl VertexDescriptorFacts {
    /// The vertex descriptor's slots as a site records them.
    fn value(&self) -> Value {
        let attributes = self
            .attributes
            .iter()
            .map(|slot| {
                Value::fields([
                    ("format", ordinal(slot.format, metal::VertexFormat::name)),
                    ("offset", Value::Uint(slot.offset)),
                    ("buffer_index", Value::Uint(slot.buffer_index)),
                ])
            })
            .collect();
        let layouts = self
            .layouts
            .iter()
            .map(|slot| {
                Value::fields([
                    ("stride", Value::Uint(slot.stride)),
                    (
                        "step_function",
                        ordinal(slot.step_function, metal::VertexStepFunction::name),
                    ),
                    ("step_rate", Value::Uint(slot.step_rate)),
                ])
            })
            .collect();
        Value::fields([
            ("attributes", Value::List(attributes)),
            ("layouts", Value::List(layouts)),
        ])
    }
}

impl RenderDescriptorFacts {
    /// The facts as the sites a case emits under one node, so both sides emit one shape.
    pub fn node(&self, name: impl Into<NodeName>) -> Node {
        let value = |value: Value| Observed::value(value);
        let mut node = Node::branch(name, Vec::new())
            .with(Node::leaf("label", field(&self.label, optional)))
            .with(Node::leaf(
                "vertex_function",
                value(optional(&self.vertex_function)),
            ))
            .with(Node::leaf(
                "fragment_function",
                value(optional(&self.fragment_function)),
            ))
            .with(Node::leaf(
                "vertex_descriptor",
                value(self.vertex_descriptor.value()),
            ))
            .with(Node::leaf(
                "sample_count",
                value(Value::Uint(self.sample_count)),
            ))
            .with(Node::leaf(
                "raster_sample_count",
                value(Value::Uint(self.raster_sample_count)),
            ))
            .with(Node::leaf(
                "alpha_to_coverage_enabled",
                value(Value::Bool(self.alpha_to_coverage_enabled)),
            ))
            .with(Node::leaf(
                "alpha_to_one_enabled",
                value(Value::Bool(self.alpha_to_one_enabled)),
            ))
            .with(Node::leaf(
                "rasterization_enabled",
                value(Value::Bool(self.rasterization_enabled)),
            ))
            .with(Node::leaf(
                "max_vertex_amplification_count",
                value(Value::Uint(self.max_vertex_amplification_count)),
            ))
            .with(Node::leaf(
                "color_attachments",
                value(Value::List(
                    self.colors.iter().map(ColorFacts::value).collect(),
                )),
            ))
            .with(Node::leaf(
                "depth_attachment_pixel_format",
                value(ordinal(
                    self.depth_attachment_pixel_format,
                    metal::PixelFormat::name,
                )),
            ))
            .with(Node::leaf(
                "stencil_attachment_pixel_format",
                value(ordinal(
                    self.stencil_attachment_pixel_format,
                    metal::PixelFormat::name,
                )),
            ))
            .with(Node::leaf(
                "input_primitive_topology",
                value(ordinal(
                    self.input_primitive_topology,
                    metal::PrimitiveTopologyClass::name,
                )),
            ));
        node = node
            .with(Node::leaf(
                "tessellation_partition_mode",
                value(ordinal(
                    self.tessellation_partition_mode,
                    metal::TessellationPartitionMode::name,
                )),
            ))
            .with(Node::leaf(
                "max_tessellation_factor",
                value(Value::Uint(self.max_tessellation_factor)),
            ))
            .with(Node::leaf(
                "tessellation_factor_scale_enabled",
                value(Value::Bool(self.tessellation_factor_scale_enabled)),
            ))
            .with(Node::leaf(
                "tessellation_factor_format",
                value(ordinal(
                    self.tessellation_factor_format,
                    metal::TessellationFactorFormat::name,
                )),
            ))
            .with(Node::leaf(
                "tessellation_control_point_index_type",
                value(ordinal(
                    self.tessellation_control_point_index_type,
                    metal::TessellationControlPointIndexType::name,
                )),
            ))
            .with(Node::leaf(
                "tessellation_factor_step_function",
                value(ordinal(
                    self.tessellation_factor_step_function,
                    metal::TessellationFactorStepFunction::name,
                )),
            ))
            .with(Node::leaf(
                "tessellation_output_winding_order",
                value(ordinal(
                    self.tessellation_output_winding_order,
                    metal::Winding::name,
                )),
            ))
            .with(Node::leaf(
                "vertex_buffers",
                value(Value::List(
                    self.vertex_buffers
                        .iter()
                        .map(|raw| ordinal(*raw, metal::Mutability::name))
                        .collect(),
                )),
            ))
            .with(Node::leaf(
                "fragment_buffers",
                value(Value::List(
                    self.fragment_buffers
                        .iter()
                        .map(|raw| ordinal(*raw, metal::Mutability::name))
                        .collect(),
                )),
            ))
            .with(Node::leaf(
                "support_indirect_command_buffers",
                value(Value::Bool(self.support_indirect_command_buffers)),
            ))
            .with(Node::leaf(
                "binary_archives",
                field(&self.binary_archives, |count| Value::Uint(*count)),
            ))
            .with(Node::leaf(
                "support_adding_vertex_binary_functions",
                value(Value::Bool(self.support_adding_vertex_binary_functions)),
            ))
            .with(Node::leaf(
                "support_adding_fragment_binary_functions",
                value(Value::Bool(self.support_adding_fragment_binary_functions)),
            ))
            .with(Node::leaf(
                "max_vertex_call_stack_depth",
                value(Value::Uint(self.max_vertex_call_stack_depth)),
            ))
            .with(Node::leaf(
                "max_fragment_call_stack_depth",
                value(Value::Uint(self.max_fragment_call_stack_depth)),
            ));
        if let Some(supplied) = &self.supplied {
            node = node
                .with(Node::leaf(
                    "vertex_preloaded_libraries",
                    value(Value::Uint(supplied.vertex_preloaded_libraries)),
                ))
                .with(Node::leaf(
                    "fragment_preloaded_libraries",
                    value(Value::Uint(supplied.fragment_preloaded_libraries)),
                ))
                .with(Node::leaf(
                    "vertex_linked_functions",
                    value(supplied.vertex_linked_functions.value()),
                ))
                .with(Node::leaf(
                    "fragment_linked_functions",
                    value(supplied.fragment_linked_functions.value()),
                ));
        }
        node
    }
}
