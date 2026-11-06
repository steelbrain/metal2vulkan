//! What a **mesh render pipeline descriptor** answers of itself:
//! `MTLMeshRenderPipelineDescriptor`'s own surface, and what `reset` puts back.
//!
//! [`crate::render_descriptors`] is the same question of `MTLRenderPipelineDescriptor`, and this
//! family shares that module's [`Shape`], its [`ColorAway`] and its [`ColorFacts`] rather than
//! declaring a second vocabulary for the colour attachments: Metal's mesh descriptor holds the same
//! `MTLRenderPipelineColorAttachmentDescriptorArray` a render descriptor does, with the same nine
//! getters per attachment.
//!
//! `mesh_draws`, `mesh_barriers` and `mesh_commands` reached five of this descriptor's eighteen
//! items by **making pipelines** with it — the three stage functions, the colour-attachment array
//! and `supportIndirectCommandBuffers` — and nothing read a getter back. So the label, the raster
//! sample count, the two coverage flags, the rasterization switch, the two attachment formats, the
//! binary archives, the three per-stage linked-function objects and `reset` were reached by nothing
//! at all, and the runtime's
//! [`MeshPipelineDescriptor::new`](metal2vulkan::pipeline::MeshPipelineDescriptor::new) holds each
//! of their defaults as a literal read off Apple's documentation.
//!
//! `reset` is the same claim from the other end: Metal's descriptor is mutable and `reset` puts
//! every field back, while the runtime's counterpart is `MeshPipelineDescriptor::new` again (the
//! coverage map says so), because a fresh struct **is** the reset.
//!
//! # What the site observes, and what it cannot
//!
//! Three items of Metal's mesh descriptor are **not implemented** by the runtime at all — the
//! three per-stage buffer-mutability arrays — because `MeshPipelineDescriptor` has no field for
//! them. The coverage map records each as missing, and a getter whose answer nothing of ours can
//! state is not a comparison, so no site here reads one. They are the runtime's gap, not this
//! family's. The object and mesh thread limits, the two execution-width hints, the payload length,
//! the mesh-grid limit, the vertex amplification count and the two required threadgroup sizes are
//! fields of `MeshPipelineDescriptor` with Metal's defaults, and no site here reads them yet: this
//! family's recordings hold no answer of theirs.
//!
//! Four getters answer **objects a program supplies**, and the runtime takes each one somewhere
//! other than the descriptor: `binaryArchives` is `PipelineOptions.binary_archives`, and
//! `objectLinkedFunctions`, `meshLinkedFunctions` and `fragmentLinkedFunctions` are each stage's
//! `FunctionDescriptor.linked`. For those the site records **how many** objects each list holds and
//! a nil list is none, so **this family compares those four items at their default alone** — the
//! lists start empty and a reset empties them, and nothing is said about a descriptor carrying a
//! real archive. That is a narrower case, not a loosened one;
//! [`crate::render_descriptors`] gives the reasoning in full.
//!
//! [`Shape::Reset`] leaves the three linked-function sites out, for the measured reason
//! [`SuppliedLists`] states.
//!
//! `shaderValidation` is absent: Apple's enumeration has no counterpart in the runtime.
//!
//! # Why every case of this family is safe
//!
//! Nothing here is compiled, allocated or submitted. A case makes one library, two or three
//! functions of it, and a descriptor it never hands to
//! `newRenderPipelineStateWithMeshDescriptor:`. No command buffer is made, so no case can fault the
//! device, trip its watchdog or reach a queue.
//!
//! The one hazard in reach is `setLabel:` with nil, which ends Metal's process. No shape sends it:
//! the label is set only where the shape states one, and the default shape reads the nil Metal's
//! own `init` left. The three stage functions are all **nullable** in `objc2-metal` 0.3.2, unlike
//! the tile descriptor's, so a nil object function is a value the site records rather than a panic.

use crate::render_descriptors::{optional, ordinal, ColorAway, ColorFacts, LinkedCounts, Shape};
use crate::transcript::{Node, NodeName, Observed, Value};
use metal2vulkan::metal;

/// Every field the set shape moves away from Metal's default, each to a value no other field of
/// the record holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Away {
    pub label: &'static str,
    pub raster_sample_count: u32,
    pub alpha_to_coverage: bool,
    pub alpha_to_one: bool,
    pub rasterization_enabled: bool,
    /// `colorAttachments[i]`, by index below [`crate::render_descriptors::READ_COLORS`].
    pub colors: &'static [ColorAway],
    pub depth_format: metal::PixelFormat,
    pub stencil_format: metal::PixelFormat,
    pub support_indirect_command_buffers: bool,
}

/// The one table of stated values, each away from Metal's default.
///
/// The raster sample count is the only number of the surface, so distinctness is a question about
/// the **colour attachments**, and the table states slots 1 and 2 with no two of their nine getters
/// agreeing across the two: a side answering a source factor out of a destination's field, or one
/// attachment's blend operation out of its neighbour's, reads a different ordinal rather than
/// passing. Slot 0 is stated by nothing, so its answer is Metal's own default for an attachment,
/// recorded beside two that were set — and it is slot 0 deliberately, which is where a side that
/// ignored the index would have written.
///
/// The two attachments are **not** [`crate::render_descriptors::AWAY`]'s: those state slots 0 and
/// 2 with their own factors, and the two readers are two functions over the same selector names, so
/// a line copied from one into the other reads a value this family never stated.
///
/// Rasterization is **disabled** here and a fragment function is set beside it, which Metal's
/// pipeline creation refuses. No shape of this family creates a pipeline, so the descriptor is
/// never handed anything that would check it; what the field means to the suite is that a program's
/// `setRasterizationEnabled:NO` is what the descriptor then answers.
pub const AWAY: Away = Away {
    label: "conformance-mesh-descriptor",
    raster_sample_count: 4,
    alpha_to_coverage: true,
    alpha_to_one: true,
    rasterization_enabled: false,
    colors: &[
        ColorAway {
            index: 1,
            pixel_format: metal::PixelFormat::RGBA16Float,
            blending_enabled: true,
            source_rgb: metal::BlendFactor::SourceColor,
            destination_rgb: metal::BlendFactor::OneMinusSourceColor,
            rgb_operation: metal::BlendOperation::Subtract,
            source_alpha: metal::BlendFactor::DestinationAlpha,
            destination_alpha: metal::BlendFactor::BlendAlpha,
            alpha_operation: metal::BlendOperation::Min,
            write_mask: metal::ColorWriteMask::GREEN,
        },
        ColorAway {
            index: 2,
            pixel_format: metal::PixelFormat::RG16Unorm,
            blending_enabled: false,
            source_rgb: metal::BlendFactor::One,
            destination_rgb: metal::BlendFactor::Zero,
            rgb_operation: metal::BlendOperation::Add,
            source_alpha: metal::BlendFactor::One,
            destination_alpha: metal::BlendFactor::Zero,
            alpha_operation: metal::BlendOperation::Add,
            write_mask: metal::ColorWriteMask::ALPHA,
        },
    ],
    depth_format: metal::PixelFormat::Depth16Unorm,
    stencil_format: metal::PixelFormat::Stencil8,
    support_indirect_command_buffers: true,
};

impl Away {
    /// The words a case's input fingerprint covers: every stated field.
    pub fn words(&self) -> Vec<u8> {
        let mut out = self.label.as_bytes().to_vec();
        for word in [
            self.raster_sample_count,
            self.depth_format.as_raw(),
            self.stencil_format.as_raw(),
        ] {
            out.extend(word.to_le_bytes());
        }
        for flag in [
            self.alpha_to_coverage,
            self.alpha_to_one,
            self.rasterization_enabled,
            self.support_indirect_command_buffers,
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
        out
    }
}

/// What one case makes and reads: the stage functions, and which of the three shapes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DescriptorSpec<'a> {
    /// `meshFunction`'s entry in the case's library.
    pub mesh: &'a str,
    /// `fragmentFunction`'s entry.
    pub fragment: &'a str,
    /// `objectFunction`'s entry, where the shape sets one.
    pub object: Option<&'a str>,
    pub shape: Shape,
}

impl<'a> DescriptorSpec<'a> {
    /// The descriptor of `mesh` and `fragment` alone, in `shape`.
    pub const fn new(mesh: &'a str, fragment: &'a str, shape: Shape) -> Self {
        Self {
            mesh,
            fragment,
            object: None,
            shape,
        }
    }

    /// The descriptor with `object` set too.
    #[must_use]
    pub const fn with_object(mut self, object: &'a str) -> Self {
        self.object = Some(object);
        self
    }

    /// The words a case's input fingerprint covers: the functions, the shape, and — where the shape
    /// sets anything — the whole stated table.
    pub fn words(&self) -> Vec<u8> {
        let mut out = self.mesh.as_bytes().to_vec();
        out.push(0);
        out.extend(self.fragment.as_bytes());
        out.push(0);
        out.extend(self.object.unwrap_or("").as_bytes());
        out.push(self.shape.word());
        if self.shape.sets() {
            out.extend(AWAY.words());
        }
        out
    }
}

/// The per-stage lists of linked functions, by how many objects each holds.
///
/// [`Shape::Reset`] reads none of them, for the reason
/// [`crate::render_descriptors::SuppliedLists`] measured on this family's oracle: `reset` leaves
/// such a list nil where `init` leaves a non-nil object, and `objc2-metal` 0.3.2 declares all three
/// of `MTLMeshRenderPipelineDescriptor`'s linked-function properties non-null, so each accessor
/// panics on the nil rather than answering it. Nothing of the runtime's answers that question
/// anyway — it has no descriptor object, so a program's own value is the counterpart and there is
/// no reset to observe.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SuppliedLists {
    /// `objectLinkedFunctions`' four lengths; a list it answers nil for is 0.
    pub object_linked_functions: LinkedCounts,
    /// `meshLinkedFunctions`' four lengths.
    pub mesh_linked_functions: LinkedCounts,
    /// `fragmentLinkedFunctions`' four lengths.
    pub fragment_linked_functions: LinkedCounts,
}

/// What a mesh render pipeline descriptor answers of itself. Each field is one Metal getter's
/// answer, an enumeration by its raw ordinal so an ordinal neither side names still compares.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MeshDescriptorFacts {
    /// `label`; `None` is Metal's nil.
    pub label: Option<String>,
    /// `objectFunction`'s own `name`; `None` where the descriptor holds none.
    pub object_function: Option<String>,
    /// `meshFunction`'s own `name`.
    pub mesh_function: Option<String>,
    /// `fragmentFunction`'s own `name`.
    pub fragment_function: Option<String>,
    pub raster_sample_count: u64,
    pub alpha_to_coverage_enabled: bool,
    pub alpha_to_one_enabled: bool,
    pub rasterization_enabled: bool,
    /// `colorAttachments[i]` for `i` below [`crate::render_descriptors::READ_COLORS`].
    pub colors: Vec<ColorFacts>,
    pub depth_attachment_pixel_format: u32,
    pub stencil_attachment_pixel_format: u32,
    pub support_indirect_command_buffers: bool,
    /// `binaryArchives`' length; a nil list is 0.
    pub binary_archives: u64,
    /// The three per-stage linked-function lists, which [`Shape::Reset`] does not read.
    pub supplied: Option<SuppliedLists>,
}

impl MeshDescriptorFacts {
    /// The facts as the sites a case emits under one node, so both sides emit one shape.
    pub fn node(&self, name: impl Into<NodeName>) -> Node {
        let value = Observed::value;
        let mut node = Node::branch(name, Vec::new())
            .with(Node::leaf("label", value(optional(&self.label))))
            .with(Node::leaf(
                "object_function",
                value(optional(&self.object_function)),
            ))
            .with(Node::leaf(
                "mesh_function",
                value(optional(&self.mesh_function)),
            ))
            .with(Node::leaf(
                "fragment_function",
                value(optional(&self.fragment_function)),
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
                "support_indirect_command_buffers",
                value(Value::Bool(self.support_indirect_command_buffers)),
            ))
            .with(Node::leaf(
                "binary_archives",
                value(Value::Uint(self.binary_archives)),
            ));
        if let Some(supplied) = &self.supplied {
            node = node
                .with(Node::leaf(
                    "object_linked_functions",
                    value(supplied.object_linked_functions.value()),
                ))
                .with(Node::leaf(
                    "mesh_linked_functions",
                    value(supplied.mesh_linked_functions.value()),
                ))
                .with(Node::leaf(
                    "fragment_linked_functions",
                    value(supplied.fragment_linked_functions.value()),
                ));
        }
        node
    }
}
