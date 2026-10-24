//! The render vocabulary: a render target, a vertex-and-fragment pipeline, and the work one
//! command buffer holds.
//!
//! The mesh families reach the device through [`crate::case::MeshStep`], which says nothing about a
//! load action, a clear color, a blend or a resolve, because a mesh case needs none of them. The
//! families where Metal's float rules live need all four: what a target stores turns on the pass's
//! load action and clear value, on the pipeline's blend, and on whether the samples are resolved.
//! So this is a second vocabulary beside that one rather than a widening of it, and it keeps the
//! mesh cases' shape untouched.
//!
//! Four things a case states, and nothing else:
//!
//! - **A target** ([`TargetSpec`]) is private by default, because that is what a render target of
//!   one of Apple's extended-range or bfloat formats is made with, and because Metal's clear route
//!   reads the texture's usage and extent rather than where it lives. A private target is read back
//!   with [`RenderWork::ReadTexture`], the blit every host can run.
//! - **A pipeline** ([`RenderPipelineSpec`]) names its two functions, the format, **write mask**
//!   and blend of each color attachment, the depth and stencil formats, and its sample count.
//! - **A depth and stencil state** ([`crate::case::Side::depth_stencil_state`]) is Metal's
//!   `MTLDepthStencilDescriptor` as the translator already names it, so there is no second
//!   spelling of a compare function or a stencil operation either.
//! - **The work** ([`RenderWork`]) is a list, and the whole list goes into **one** command buffer,
//!   committed and waited for once. That is deliberate: the clear sweeps ask Metal hundreds of
//!   one-texel questions, and one command buffer a question would cost hundreds of round trips on
//!   both sides. A case that wants several command buffers calls [`crate::case::Side::render`] more
//!   than once, so the batching is the case's to state rather than the vocabulary's to assume.
//!
//! A pass ([`RenderPass`]) holds its color attachments, its **depth** and **stencil** attachments
//! and its **visibility-result buffer**, and its steps ([`RenderStep`]) are the encoder's own
//! calls, in the order the case lists them:
//!
//! - the **pipeline** and the **depth-stencil state**;
//! - the **fixed-function state** a draw is decided by: the viewport and the viewports, the
//!   scissor rectangle and the scissor rectangles, the cull mode, the front-facing winding, the
//!   triangle fill mode, the depth clip mode, the depth bias with its slope scale and clamp, the
//!   blend color, and the stencil reference, both faces together or apart;
//! - the **bindings** each stage reads: a buffer, its offset alone, a range of buffers, inline
//!   bytes, a texture, a range of textures, a sampler state with or without its lod clamps, and a
//!   range of either;
//! - the **visibility-result mode**;
//! - the **barriers** between two draws of one pass, by scope or by resource;
//! - the **residency** a draw's functions may reach a resource through, by resource or by heap,
//!   with the stages it names or in Metal's deprecated form that names them all;
//! - and the **draws**: direct, indexed and indirect.
//!
//! **A multisampled attachment is read one sample at a time through a kernel**, not through a
//! call of its own: neither side can `getBytes` or blit a multisample texture, so a case makes the
//! target `shader_read` as well ([`TargetSpec::multisampled`] beside
//! [`TargetSpec::read`]), binds it to a [`Dispatch`] in the same command buffer, and the kernel
//! writes each sample's texels into a buffer. A resolve is the other half of that question and is
//! the descriptor's ([`PassColor::resolving`]).
//!
//! Every enumeration here is Metal's own as the translator and the runtime name it, so a value a
//! case writes reaches `MTLBlendFactor`, `MTLBlendOperation`, `MTLColorWriteMask`, `MTLLoadAction`,
//! `MTLStoreAction`, `MTLPrimitiveType`, `MTLIndexType`, `MTLVisibilityResultMode` and the
//! multisample resolve filters by its raw ordinal on the Metal side, and reaches the runtime's
//! descriptor unchanged. There is no second spelling to keep in step.

use crate::case::Failure;
use crate::case::Handle;
use metal2vulkan::metal::{
    BarrierScope, BlendFactor, BlendOperation, ColorWriteMask, CullMode, DepthClipMode, IndexType,
    PixelFormat, PrimitiveType, RenderStages, ResourceUsage, TessellationControlPointIndexType,
    TessellationFactorFormat, TessellationFactorStepFunction, TessellationPartitionMode,
    TriangleFillMode, VertexFormat, VertexStepFunction, Winding,
};
use metal2vulkan_runtime::{
    LoadAction, MultisampleDepthResolveFilter, MultisampleStencilResolveFilter, SamplePosition,
    StoreAction, StoreActionOptions, VisibilityResultMode,
};

/// A render target a case makes: `MTLTextureDescriptor`'s facts a render case states.
///
/// The usage is part of what the case means, not a detail: Metal clears a `BGR10_XR` target
/// straight from the `float` for a plain private target and through `half` for one of
/// `ShaderWrite` usage, so a case that let the side choose its usage would be asking a different
/// question on each side.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TargetSpec {
    pub format: PixelFormat,
    pub size: [u32; 2],
    /// `sampleCount`: 1, or the samples of a multisampled target.
    pub samples: u32,
    /// `MTLTextureUsageShaderRead` beside `MTLTextureUsageRenderTarget`.
    pub shader_read: bool,
    /// `MTLTextureUsageShaderWrite` beside `MTLTextureUsageRenderTarget`.
    pub shader_write: bool,
    /// Whether the target is of shared storage rather than private.
    ///
    /// Private is the default and what an extended-range target is normally made with. A case says
    /// shared when the host has to put **arbitrary raw codes** into the target before the pass:
    /// a blend's destination texels are random codes, and neither a clear nor a kernel write can
    /// set an arbitrary code, so they go in through `replaceRegion:`, which no private texture
    /// takes.
    pub shared: bool,
    /// A **framebuffer-only texture over one plane of an IOSurface**, rather than a texture of its
    /// own memory ([`crate::surface::PlaneSpec`]).
    ///
    /// This is the one target shape that is not a plain descriptor: `framebufferOnly` and a
    /// rotation are Apple SPI on the descriptor, and Metal makes such a texture over an IOSurface
    /// alone. A side that cannot make one answers its own typed refusal.
    pub surface: Option<crate::surface::PlaneSpec>,
    /// A **texture over a buffer the case made**, rather than a texture of its own memory
    /// ([`OverBuffer`]).
    ///
    /// Such a target's texels are a buffer's bytes, so the case reads the buffer back and sees
    /// both what the pass wrote and what it left between the rows. The buffer is the case's own
    /// object, from [`Side::buffer`](crate::case::Side::buffer), so the case states its contents
    /// and reads them with [`Side::read_buffer`](crate::case::Side::read_buffer).
    pub over_buffer: Option<OverBuffer>,
}

/// Where a target's texels lie in a buffer the case made:
/// `newTextureWithDescriptor:offset:bytesPerRow:`.
///
/// The row in bytes is **stated** rather than derived from the pixel format, for the reason the
/// buffer-backed texture cases state theirs: `MTLTexture` answers no bytes-per-texel, and a format
/// table of ours would be a second spelling of Metal's to keep in step. It is here because a
/// pitch below the row is the one shape of this call that ends Metal's process rather than
/// refusing ([`OverBuffer::unsafe_stride`]), and the number the guard compares against has to come
/// from the case.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OverBuffer {
    /// The buffer the target lies over. A handle never reaches a fingerprint, so this field is not
    /// in [`OverBuffer::words`].
    pub buffer: Handle,
    /// `offset`: the target's first row's first byte in the buffer.
    pub offset: u64,
    /// `bytesPerRow`: how far apart the target's rows lie in the buffer.
    pub bytes_per_row: u64,
    /// One row of the target in bytes, which the case states.
    pub row_bytes: u64,
}

impl OverBuffer {
    /// The target over `buffer` at `offset`, its rows `bytes_per_row` apart and `row_bytes` wide.
    pub const fn new(buffer: Handle, offset: u64, bytes_per_row: u64, row_bytes: u64) -> Self {
        Self {
            buffer,
            offset,
            bytes_per_row,
            row_bytes,
        }
    }

    /// Why handing this backing to Metal would end the process, where it would.
    ///
    /// A `bytesPerRow` below the texture's own row fails AGX's assertion
    /// `bytes_per_row >= used_bytes_per_row`, which ends the process rather than answering nil, so
    /// no side may make the call and no case may ask for it. It is decided from the backing's own
    /// numbers, so a side reaches the refusal without a device.
    pub fn unsafe_stride(&self) -> Option<String> {
        (self.bytes_per_row < self.row_bytes).then(|| {
            format!(
                "bytesPerRow {} is below the target's own row of {} bytes, which fails AGX's \
                 assertion `bytes_per_row >= used_bytes_per_row` and ends Metal's process",
                self.bytes_per_row, self.row_bytes
            )
        })
    }

    /// The words a case's input fingerprint covers: the call's three numbers and the stated row.
    /// The buffer's handle is not among them.
    pub fn words(&self) -> Vec<u8> {
        let mut bytes = b"newTextureWithDescriptor:offset:bytesPerRow:".to_vec();
        bytes.extend(self.offset.to_le_bytes());
        bytes.extend(self.bytes_per_row.to_le_bytes());
        bytes.extend(self.row_bytes.to_le_bytes());
        bytes
    }
}

impl TargetSpec {
    /// A single-sampled target of `format` and `size` a shader also reads.
    pub const fn read(format: PixelFormat, size: [u32; 2]) -> Self {
        Self {
            format,
            size,
            samples: 1,
            shader_read: true,
            shader_write: false,
            shared: false,
            surface: None,
            over_buffer: None,
        }
    }

    /// A single-sampled target of `format` and `size` of `MTLTextureUsageRenderTarget` **alone**,
    /// which is how the lanes of the tile-store and visibility families make theirs.
    ///
    /// The usage is part of the question, not a detail: Metal refuses a depth texture of
    /// shader-write usage, and a target a shader may read is a different texture to the driver, so
    /// a case that asks what a pass leaves in a plain render target says so here.
    pub const fn plain(format: PixelFormat, size: [u32; 2]) -> Self {
        Self {
            format,
            size,
            samples: 1,
            shader_read: false,
            shader_write: false,
            shared: false,
            surface: None,
            over_buffer: None,
        }
    }

    /// The target as a framebuffer-only texture over one plane of an IOSurface, of `plane`'s
    /// rotation and stride. Such a texture is of shared storage, as the surface's bytes are.
    #[must_use]
    pub const fn over_plane(mut self, plane: crate::surface::PlaneSpec) -> Self {
        self.surface = Some(plane);
        self.shared = true;
        self
    }

    /// The target as a texture over `backing`'s buffer. Such a texture holds the buffer's own
    /// bytes, so it is of shared storage as the buffer is, and its contents are the case's to
    /// state and to read back.
    #[must_use]
    pub const fn over_buffer(mut self, backing: OverBuffer) -> Self {
        self.over_buffer = Some(backing);
        self.shared = true;
        self
    }

    /// The target of shared storage, so the host may write its texels with `replaceRegion:`.
    #[must_use]
    pub const fn shared(mut self) -> Self {
        self.shared = true;
        self
    }

    /// The target at `samples` samples.
    #[must_use]
    pub const fn multisampled(mut self, samples: u32) -> Self {
        self.samples = samples;
        self
    }

    /// The target with `MTLTextureUsageShaderWrite` as well.
    #[must_use]
    pub const fn written(mut self) -> Self {
        self.shader_write = true;
        self
    }

    /// The words a case's input fingerprint covers: every field, so a target that moves is a
    /// change to the case's inputs rather than a quiet pass against another target's answer.
    pub fn words(&self) -> Vec<u8> {
        let mut bytes = self.format.as_raw().to_le_bytes().to_vec();
        for axis in self.size {
            bytes.extend(axis.to_le_bytes());
        }
        bytes.extend(self.samples.to_le_bytes());
        bytes.push(u8::from(self.shader_read));
        bytes.push(u8::from(self.shader_write));
        bytes.push(u8::from(self.shared));
        // Written **only where the target has one**, as a pass writes its optional attachments:
        // a target of its own memory then fingerprints to exactly the bytes it always has, so
        // adding this field left every existing render case's recording valid. Writing a 0 here
        // for `None` did not, and the whole-set replay caught it as 62 cases of other streams'
        // going unrecorded.
        if let Some(plane) = &self.surface {
            bytes.push(1);
            bytes.extend(plane.words());
        }
        // Written **only where the target has one**, for the reason the surface plane above is:
        // a target of its own memory fingerprints to exactly the bytes it always has, so adding
        // this field leaves every existing render case's recording valid. A `0` for `None` would
        // not, and that mistake cost 62 of other families' cases a verdict once already.
        if let Some(over) = &self.over_buffer {
            bytes.push(1);
            bytes.extend(over.words());
        }
        bytes
    }
}

/// A color attachment's blend: `MTLRenderPipelineColorAttachmentDescriptor`'s six fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Blend {
    pub source_rgb: BlendFactor,
    pub destination_rgb: BlendFactor,
    pub rgb: BlendOperation,
    pub source_alpha: BlendFactor,
    pub destination_alpha: BlendFactor,
    pub alpha: BlendOperation,
}

impl Blend {
    /// The blend that adds the source times `source` to the destination times `destination`, on
    /// the color channels and the alpha channel alike.
    pub const fn added(source: BlendFactor, destination: BlendFactor) -> Self {
        Self {
            source_rgb: source,
            destination_rgb: destination,
            rgb: BlendOperation::Add,
            source_alpha: source,
            destination_alpha: destination,
            alpha: BlendOperation::Add,
        }
    }

    /// The color channels' blend and the alpha channel's, stated apart.
    pub const fn separate(
        rgb: (BlendFactor, BlendFactor, BlendOperation),
        alpha: (BlendFactor, BlendFactor, BlendOperation),
    ) -> Self {
        Self {
            source_rgb: rgb.0,
            destination_rgb: rgb.1,
            rgb: rgb.2,
            source_alpha: alpha.0,
            destination_alpha: alpha.1,
            alpha: alpha.2,
        }
    }

    /// The words a case's input fingerprint covers.
    pub fn words(&self) -> Vec<u8> {
        [
            self.source_rgb.as_raw(),
            self.destination_rgb.as_raw(),
            self.rgb.as_raw(),
            self.source_alpha.as_raw(),
            self.destination_alpha.as_raw(),
            self.alpha.as_raw(),
        ]
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect()
    }
}

/// One color attachment of a render pipeline: its format, and its blend where blending is on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ColorState {
    pub format: PixelFormat,
    /// `writeMask`: which of the attachment's channels a fragment writes.
    ///
    /// Metal's default is every channel, and a mask of **none** is a question of its own: a draw
    /// that writes no texel still makes the pass a drawing pass, which is how the tile-store
    /// family asks what Metal's store leaves behind in the texels nothing wrote.
    pub write_mask: ColorWriteMask,
    /// `blendingEnabled` and the six factors, `None` for blending off.
    pub blend: Option<Blend>,
}

impl ColorState {
    /// An attachment of `format` with Metal's defaults: every channel written, blending off.
    pub const fn new(format: PixelFormat) -> Self {
        Self {
            format,
            write_mask: ColorWriteMask::ALL,
            blend: None,
        }
    }

    /// The attachment with `writeMask` set to `mask`.
    #[must_use]
    pub const fn masked(mut self, mask: ColorWriteMask) -> Self {
        self.write_mask = mask;
        self
    }

    /// The attachment with `blend` enabled.
    #[must_use]
    pub const fn blended(mut self, blend: Blend) -> Self {
        self.blend = Some(blend);
        self
    }

    /// The words a case's input fingerprint covers.
    ///
    /// The write mask is written **only where it is not Metal's default**, so an attachment that
    /// writes every channel fingerprints to the bytes it always has. Every widening here keeps
    /// that property: a recording is invalidated by a case's own inputs moving, never by a field
    /// another family needed being added beside them.
    pub fn words(&self) -> Vec<u8> {
        let mut bytes = self.format.as_raw().to_le_bytes().to_vec();
        if self.write_mask != ColorWriteMask::ALL {
            bytes.extend(self.write_mask.as_raw().to_le_bytes());
        }
        match &self.blend {
            Some(blend) => {
                bytes.push(1);
                bytes.extend(blend.words());
            }
            None => bytes.push(0),
        }
        bytes
    }
}

/// One stage of a render pipeline: the function's entry name, and the library it comes from when
/// the pipeline's two stages do not share one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StageFunction<'a> {
    /// The library the function comes from, when that is not the one the call to
    /// [`crate::case::Side::render_pipeline`] names.
    pub library: Option<Handle>,
    pub entry: &'a str,
}

impl<'a> StageFunction<'a> {
    /// A function of the library the call names.
    #[must_use]
    pub const fn named(entry: &'a str) -> Self {
        Self {
            library: None,
            entry,
        }
    }

    /// A function of `library`, whichever library the call names.
    #[must_use]
    pub const fn of(library: Handle, entry: &'a str) -> Self {
        Self {
            library: Some(library),
            entry,
        }
    }

    /// The words a case's input fingerprint covers: the entry, and the library only where the
    /// stage names one of its own, so a pipeline of one library fingerprints as it always has.
    fn words(&self) -> Vec<u8> {
        let mut bytes = self.entry.as_bytes().to_vec();
        bytes.push(0);
        if let Some(library) = self.library {
            bytes.extend(library.0.to_le_bytes());
        }
        bytes
    }
}

/// One entry of `MTLVertexDescriptor`'s `attributes`:
/// `MTLVertexAttributeDescriptor`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VertexAttribute {
    /// The `[[attribute(n)]]` the vertex function declares.
    pub index: u32,
    pub format: VertexFormat,
    /// `offset`, in bytes from the start of the buffer layout's element.
    pub offset: u32,
    /// `bufferIndex`: which vertex buffer the attribute is fetched from.
    pub buffer: u32,
}

impl VertexAttribute {
    /// The attribute `index` of `format` at `offset` of buffer `buffer`.
    pub const fn new(index: u32, format: VertexFormat, offset: u32, buffer: u32) -> Self {
        Self {
            index,
            format,
            offset,
            buffer,
        }
    }

    /// The words a case's input fingerprint covers.
    fn words(&self) -> Vec<u8> {
        let mut bytes = self.index.to_le_bytes().to_vec();
        bytes.extend(self.format.as_raw().to_le_bytes());
        bytes.extend(self.offset.to_le_bytes());
        bytes.extend(self.buffer.to_le_bytes());
        bytes
    }
}

/// One entry of `MTLVertexDescriptor`'s `layouts`: `MTLVertexBufferLayoutDescriptor`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VertexBufferLayout {
    /// Which vertex buffer the layout describes.
    pub index: u32,
    /// `stride`, in bytes between elements.
    pub stride: u32,
    /// `stepFunction`.
    pub step: VertexStepFunction,
    /// `stepRate`.
    pub rate: u32,
}

impl VertexBufferLayout {
    /// Buffer `index` of `stride` bytes, stepped once per vertex.
    pub const fn per_vertex(index: u32, stride: u32) -> Self {
        Self {
            index,
            stride,
            step: VertexStepFunction::PerVertex,
            rate: 1,
        }
    }

    /// Buffer `index` of `stride` bytes, fetched from the same element for every vertex: Metal's
    /// constant step, whose rate is 0.
    pub const fn constant(index: u32, stride: u32) -> Self {
        Self {
            index,
            stride,
            step: VertexStepFunction::Constant,
            rate: 0,
        }
    }

    /// The words a case's input fingerprint covers.
    fn words(&self) -> Vec<u8> {
        let mut bytes = self.index.to_le_bytes().to_vec();
        bytes.extend(self.stride.to_le_bytes());
        bytes.extend(self.step.as_raw().to_le_bytes());
        bytes.extend(self.rate.to_le_bytes());
        bytes
    }
}

/// A pipeline's `vertexDescriptor`: how the fixed-function stage fetches each attribute the vertex
/// function declares.
///
/// Borrowed rather than owned, so a case states one in a `const` beside its other tables and the
/// spec stays [`Copy`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VertexLayout<'a> {
    pub attributes: &'a [VertexAttribute],
    pub layouts: &'a [VertexBufferLayout],
}

impl<'a> VertexLayout<'a> {
    pub const fn new(attributes: &'a [VertexAttribute], layouts: &'a [VertexBufferLayout]) -> Self {
        Self {
            attributes,
            layouts,
        }
    }

    /// The words a case's input fingerprint covers: every attribute and every layout, in the order
    /// the case states them, since that order is the case's own statement.
    fn words(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        for attribute in self.attributes {
            bytes.extend(attribute.words());
        }
        for layout in self.layouts {
            bytes.extend(layout.words());
        }
        bytes
    }
}

/// The tessellation fields of `MTLRenderPipelineDescriptor`, which a pipeline whose vertex stage is
/// a post-tessellation vertex function states: what the synthesized control stage does with a
/// patch's factors, and how the tessellator places the points it generates from them.
///
/// Every field is Metal's own, and so is every default: `maxTessellationFactor` 16, `pow2`
/// partitioning, `half` factors, the constant factor step, clockwise output and no control-point
/// index type. The struct carries the defaults rather than `Option`s so a case states the one axis
/// it sweeps and says nothing about the rest, and so a field this suite forgets to forward is a
/// difference in an answer rather than an absence nobody looks at.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TessellationState {
    /// `maxTessellationFactor`: the ceiling every edge and inside factor is clamped to.
    pub max_factor: u32,
    /// `tessellationPartitionMode`.
    pub partition: TessellationPartitionMode,
    /// `tessellationFactorFormat`.
    pub factor_format: TessellationFactorFormat,
    /// `tessellationFactorStepFunction`: which patch's and which instance's factors a patch reads.
    pub factor_step: TessellationFactorStepFunction,
    /// `tessellationOutputWindingOrder`.
    pub winding: Winding,
    /// `tessellationControlPointIndexType`, which a `drawIndexedPatches:…` draw reads its
    /// control-point indices as.
    pub control_point_index_type: TessellationControlPointIndexType,
    /// `tessellationFactorScaleEnabled`.
    pub factor_scale_enabled: bool,
}

impl Default for TessellationState {
    fn default() -> Self {
        Self {
            max_factor: 16,
            partition: TessellationPartitionMode::Pow2,
            factor_format: TessellationFactorFormat::Half,
            factor_step: TessellationFactorStepFunction::Constant,
            winding: Winding::Clockwise,
            control_point_index_type: TessellationControlPointIndexType::None,
            factor_scale_enabled: false,
        }
    }
}

impl TessellationState {
    /// Metal's own defaults, every field as a descriptor that states none of them answers.
    pub fn new() -> Self {
        Self::default()
    }

    /// The state under `partition`.
    #[must_use]
    pub const fn partitioned(mut self, partition: TessellationPartitionMode) -> Self {
        self.partition = partition;
        self
    }

    /// The state whose factors step by `step`.
    #[must_use]
    pub const fn stepping(mut self, step: TessellationFactorStepFunction) -> Self {
        self.factor_step = step;
        self
    }

    /// The state whose ceiling is `max_factor`.
    #[must_use]
    pub const fn clamped_to(mut self, max_factor: u32) -> Self {
        self.max_factor = max_factor;
        self
    }

    /// The state whose control-point indices are read as `index_type`.
    #[must_use]
    pub const fn indexed_by(mut self, index_type: TessellationControlPointIndexType) -> Self {
        self.control_point_index_type = index_type;
        self
    }

    /// The state whose tessellator winds its output `winding`.
    #[must_use]
    pub const fn wound(mut self, winding: Winding) -> Self {
        self.winding = winding;
        self
    }

    /// The state with `tessellationFactorScaleEnabled`, whose factors the encoder's
    /// [`RenderStep::TessellationFactorScale`] multiplies.
    #[must_use]
    pub const fn scaled(mut self) -> Self {
        self.factor_scale_enabled = true;
        self
    }

    /// The words a case's input fingerprint covers: every field, so any one of them moving moves
    /// the case's key.
    pub fn words(&self) -> Vec<u8> {
        let mut bytes = self.max_factor.to_le_bytes().to_vec();
        bytes.extend(self.partition.as_raw().to_le_bytes());
        bytes.extend(self.factor_format.as_raw().to_le_bytes());
        bytes.extend(self.factor_step.as_raw().to_le_bytes());
        bytes.extend(self.winding.as_raw().to_le_bytes());
        bytes.extend(self.control_point_index_type.as_raw().to_le_bytes());
        bytes.push(u8::from(self.factor_scale_enabled));
        bytes
    }
}

/// A vertex-and-fragment render pipeline a case makes:
/// `newRenderPipelineStateWithDescriptor:error:`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RenderPipelineSpec<'a> {
    pub vertex: StageFunction<'a>,
    /// The fragment stage, or `None` for a pipeline that rasterizes without one.
    pub fragment: Option<StageFunction<'a>>,
    /// The color attachments by index; `None` at an index of `MTLPixelFormatInvalid`, which is how
    /// a pipeline that writes attachment 1 alone is stated.
    pub colors: &'a [Option<ColorState>],
    /// `depthAttachmentPixelFormat`; `None` is `MTLPixelFormatInvalid`, a pipeline without one.
    pub depth_format: Option<PixelFormat>,
    /// `stencilAttachmentPixelFormat`; `None` is `MTLPixelFormatInvalid`.
    pub stencil_format: Option<PixelFormat>,
    /// `rasterSampleCount`.
    pub samples: u32,
    /// `vertexDescriptor`, where the vertex function takes a stage input the fixed-function stage
    /// fetches; `None` for a function that reads its buffers itself.
    pub vertex_layout: Option<VertexLayout<'a>>,
    /// `rasterizationEnabled`. A `vertex void` function has no position to rasterize, so its
    /// pipeline must disable it and must carry no fragment stage.
    pub rasterization: bool,
    /// `label`, which the made state answers back as its own `label`; `None` is Metal's nil.
    pub label: Option<&'a str>,
    /// `supportIndirectCommandBuffers`: an indirect command buffer's render commands may set the
    /// state, which the made state answers back.
    pub support_indirect_command_buffers: bool,
    /// The tessellation fields, where the pipeline's vertex stage is a post-tessellation vertex
    /// function; `None` leaves every one of them at Metal's default, which is what a pipeline of an
    /// ordinary vertex function states.
    pub tessellation: Option<TessellationState>,
}

impl<'a> RenderPipelineSpec<'a> {
    /// A single-sampled pipeline of `vertex` and `fragment`, both of the library the call names,
    /// over `colors`.
    pub const fn new(vertex: &'a str, fragment: &'a str, colors: &'a [Option<ColorState>]) -> Self {
        Self::of(
            StageFunction::named(vertex),
            Some(StageFunction::named(fragment)),
            colors,
        )
    }

    /// A single-sampled pipeline of stages that may each name their own library, and whose fragment
    /// stage may be absent.
    pub const fn of(
        vertex: StageFunction<'a>,
        fragment: Option<StageFunction<'a>>,
        colors: &'a [Option<ColorState>],
    ) -> Self {
        Self {
            vertex,
            fragment,
            colors,
            depth_format: None,
            stencil_format: None,
            samples: 1,
            vertex_layout: None,
            rasterization: true,
            label: None,
            support_indirect_command_buffers: false,
            tessellation: None,
        }
    }

    /// The pipeline whose descriptor carries `label`, which the made state answers back.
    #[must_use]
    pub const fn labelled(mut self, label: &'a str) -> Self {
        self.label = Some(label);
        self
    }

    /// The pipeline whose descriptor sets `supportIndirectCommandBuffers`.
    #[must_use]
    pub const fn for_indirect_commands(mut self) -> Self {
        self.support_indirect_command_buffers = true;
        self
    }

    /// The pipeline of a `vertex void` function that writes its answer through a buffer: its stage
    /// input fetched under `layout`, rasterization disabled and no fragment stage, which is the
    /// one shape Metal accepts for such a function.
    #[must_use]
    pub const fn converting(vertex: StageFunction<'a>, layout: VertexLayout<'a>) -> Self {
        let mut spec = Self::of(vertex, None, &[]);
        spec.vertex_layout = Some(layout);
        spec.rasterization = false;
        spec
    }

    /// The pipeline of a `vertex void` **post-tessellation** vertex function that writes its answer
    /// through a buffer: [`RenderPipelineSpec::converting`] with the tessellation fields `state`
    /// names, which is the shape a patch draw of such a function needs.
    #[must_use]
    pub const fn tessellating(
        vertex: StageFunction<'a>,
        layout: VertexLayout<'a>,
        state: TessellationState,
    ) -> Self {
        let mut spec = Self::converting(vertex, layout);
        spec.tessellation = Some(state);
        spec
    }

    /// The pipeline at `samples` samples.
    #[must_use]
    pub const fn multisampled(mut self, samples: u32) -> Self {
        self.samples = samples;
        self
    }

    /// The pipeline whose depth and stencil attachment formats are `depth` and `stencil`. A
    /// pipeline that tests or writes either must name its format, as Metal's descriptor must.
    #[must_use]
    pub const fn depth_stencil(
        mut self,
        depth: Option<PixelFormat>,
        stencil: Option<PixelFormat>,
    ) -> Self {
        self.depth_format = depth;
        self.stencil_format = stencil;
        self
    }

    /// The words a case's input fingerprint covers: the functions, every attachment, the depth and
    /// stencil formats where the pipeline names them, the sample count, and the label and the
    /// indirect-command support where it states them.
    pub fn words(&self) -> Vec<u8> {
        let mut bytes = self.vertex.words();
        match &self.fragment {
            Some(fragment) => bytes.extend(fragment.words()),
            None => bytes.push(0xff),
        }
        for color in self.colors {
            match color {
                Some(state) => bytes.extend(state.words()),
                None => bytes.push(0xff),
            }
        }
        for format in [self.depth_format, self.stencil_format]
            .into_iter()
            .flatten()
        {
            bytes.extend(format.as_raw().to_le_bytes());
        }
        bytes.extend(self.samples.to_le_bytes());
        // Appended, and only where the pipeline states them, so a pipeline that names neither
        // fingerprints to the bytes it always has and every recording already committed still
        // reads.
        if !self.rasterization {
            bytes.push(0xfe);
        }
        if let Some(layout) = &self.vertex_layout {
            bytes.extend(layout.words());
        }
        if let Some(label) = self.label {
            bytes.push(0xfd);
            bytes.extend(label.as_bytes());
        }
        if self.support_indirect_command_buffers {
            bytes.push(0xfc);
        }
        if let Some(state) = &self.tessellation {
            bytes.push(0xfb);
            bytes.extend(state.words());
        }
        bytes
    }
}

/// The depth attachment of a render pass: `MTLRenderPassDepthAttachmentDescriptor`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PassDepth {
    pub texture: Handle,
    pub load: LoadAction,
    pub store: StoreAction,
    /// `clearDepth`, as `f64`, since Metal's is a double.
    pub clear: f64,
    /// `resolveTexture`.
    pub resolve: Option<Handle>,
    /// `depthResolveFilter`.
    pub filter: MultisampleDepthResolveFilter,
    /// `storeActionOptions`: Metal's default is none, and
    /// `MTLStoreActionOptionCustomSamplePositions` is the one option it defines.
    pub options: StoreActionOptions,
}

impl PassDepth {
    /// `texture` cleared to `clear` and stored.
    pub const fn cleared(texture: Handle, clear: f64) -> Self {
        Self {
            texture,
            load: LoadAction::Clear,
            store: StoreAction::Store,
            clear,
            resolve: None,
            filter: MultisampleDepthResolveFilter::Sample0,
            options: StoreActionOptions::NONE,
        }
    }

    /// The attachment with `storeAction` set to `store`.
    #[must_use]
    pub const fn stored(mut self, store: StoreAction) -> Self {
        self.store = store;
        self
    }

    /// The attachment resolved into `resolve` through `filter`, stored and resolved.
    #[must_use]
    pub const fn resolving(
        mut self,
        resolve: Handle,
        filter: MultisampleDepthResolveFilter,
    ) -> Self {
        self.resolve = Some(resolve);
        self.filter = filter;
        self.store = StoreAction::StoreAndMultisampleResolve;
        self
    }

    /// The attachment with `storeActionOptions` set to `options`.
    #[must_use]
    pub const fn with_options(mut self, options: StoreActionOptions) -> Self {
        self.options = options;
        self
    }

    /// The words a case's input fingerprint covers. Nothing of a handle is fingerprinted: a
    /// side's handle numbers are its own.
    ///
    /// The store-action options are written **only where they are not Metal's default**, so an
    /// attachment that states none fingerprints to the bytes it always has.
    pub fn words(&self) -> Vec<u8> {
        let mut bytes = vec![u8::from(self.resolve.is_some())];
        bytes.extend(self.load.raw().to_le_bytes());
        bytes.extend(self.store.raw().to_le_bytes());
        bytes.extend(self.clear.to_bits().to_le_bytes());
        bytes.extend(self.filter.raw().to_le_bytes());
        if self.options != StoreActionOptions::NONE {
            bytes.extend(self.options.raw().to_le_bytes());
        }
        bytes
    }
}

/// The stencil attachment of a render pass: `MTLRenderPassStencilAttachmentDescriptor`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PassStencil {
    pub texture: Handle,
    pub load: LoadAction,
    pub store: StoreAction,
    /// `clearStencil`; Metal clears its low 8 bits.
    pub clear: u32,
    /// `resolveTexture`.
    pub resolve: Option<Handle>,
    /// `stencilResolveFilter`.
    pub filter: MultisampleStencilResolveFilter,
    /// `storeActionOptions`. **The stencil attachment is the one Metal checks**: its validation
    /// stops `CustomSamplePositions` here without positions, or under a store action that keeps no
    /// samples, where a color or a depth attachment's options change nothing.
    pub options: StoreActionOptions,
}

impl PassStencil {
    /// `texture` cleared to `clear` and stored.
    pub const fn cleared(texture: Handle, clear: u32) -> Self {
        Self {
            texture,
            load: LoadAction::Clear,
            store: StoreAction::Store,
            clear,
            resolve: None,
            filter: MultisampleStencilResolveFilter::Sample0,
            options: StoreActionOptions::NONE,
        }
    }

    /// The attachment with `storeAction` set to `store`.
    #[must_use]
    pub const fn stored(mut self, store: StoreAction) -> Self {
        self.store = store;
        self
    }

    /// The attachment resolved into `resolve` through `filter`, stored and resolved.
    #[must_use]
    pub const fn resolving(
        mut self,
        resolve: Handle,
        filter: MultisampleStencilResolveFilter,
    ) -> Self {
        self.resolve = Some(resolve);
        self.filter = filter;
        self.store = StoreAction::StoreAndMultisampleResolve;
        self
    }

    /// The attachment with `storeActionOptions` set to `options`.
    #[must_use]
    pub const fn with_options(mut self, options: StoreActionOptions) -> Self {
        self.options = options;
        self
    }

    /// The words a case's input fingerprint covers; the options only where they are not Metal's
    /// default.
    pub fn words(&self) -> Vec<u8> {
        let mut bytes = vec![u8::from(self.resolve.is_some())];
        bytes.extend(self.load.raw().to_le_bytes());
        bytes.extend(self.store.raw().to_le_bytes());
        bytes.extend(self.clear.to_le_bytes());
        bytes.extend(self.filter.raw().to_le_bytes());
        if self.options != StoreActionOptions::NONE {
            bytes.extend(self.options.raw().to_le_bytes());
        }
        bytes
    }
}

/// One color attachment of a render pass: `MTLRenderPassColorAttachmentDescriptor`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PassColor {
    /// The texture the attachment renders into; `None` leaves the index unattached, which is how a
    /// pass that attaches color 1 alone is stated.
    pub texture: Option<Handle>,
    pub load: LoadAction,
    pub store: StoreAction,
    /// `clearColor`'s four channels, as `f64`, since Metal's clear color is a double and the
    /// double a case names is part of what the clear route rounds.
    pub clear: [f64; 4],
    /// `resolveTexture`: the single-sampled texture a resolving store action writes.
    pub resolve: Option<Handle>,
    /// `storeActionOptions`: Metal's default is none, and
    /// `MTLStoreActionOptionCustomSamplePositions` is the one option it defines.
    pub options: StoreActionOptions,
}

impl PassColor {
    /// An attachment that is not there.
    pub const fn none() -> Self {
        Self {
            texture: None,
            load: LoadAction::DontCare,
            store: StoreAction::Store,
            clear: [0.0, 0.0, 0.0, 1.0],
            resolve: None,
            options: StoreActionOptions::NONE,
        }
    }

    /// `texture` cleared to `clear` and stored.
    pub const fn cleared(texture: Handle, clear: [f64; 4]) -> Self {
        Self {
            texture: Some(texture),
            load: LoadAction::Clear,
            store: StoreAction::Store,
            clear,
            resolve: None,
            options: StoreActionOptions::NONE,
        }
    }

    /// `texture` loaded and stored, the shape a blend draws over.
    pub const fn loaded(texture: Handle) -> Self {
        Self {
            texture: Some(texture),
            load: LoadAction::Load,
            store: StoreAction::Store,
            clear: [0.0, 0.0, 0.0, 1.0],
            resolve: None,
            options: StoreActionOptions::NONE,
        }
    }

    /// `texture` whose texels the pass does not read and does not keep: the attachment a routes
    /// case draws into beside the one it is asking about.
    pub const fn discarded(texture: Handle) -> Self {
        Self {
            texture: Some(texture),
            load: LoadAction::DontCare,
            store: StoreAction::Store,
            clear: [0.0, 0.0, 0.0, 1.0],
            resolve: None,
            options: StoreActionOptions::NONE,
        }
    }

    /// The attachment with `storeAction` set to `store`: a multisampled attachment a pass does
    /// not keep says `DontCare` here, since Metal stores no multisample texture a host reads.
    #[must_use]
    pub const fn stored(mut self, store: StoreAction) -> Self {
        self.store = store;
        self
    }

    /// The attachment with `resolve` as its resolve texture, stored and resolved.
    #[must_use]
    pub const fn resolving(mut self, resolve: Handle) -> Self {
        self.resolve = Some(resolve);
        self.store = StoreAction::StoreAndMultisampleResolve;
        self
    }

    /// The attachment with `storeActionOptions` set to `options`.
    #[must_use]
    pub const fn with_options(mut self, options: StoreActionOptions) -> Self {
        self.options = options;
        self
    }

    /// The words a case's input fingerprint covers: the actions and the clear color's bits, so a
    /// clear value that moves by one `f64` step is a change to the case's inputs. Nothing of a
    /// handle is fingerprinted: a side's handle numbers are its own.
    pub fn words(&self) -> Vec<u8> {
        let mut bytes = vec![
            u8::from(self.texture.is_some()),
            u8::from(self.resolve.is_some()),
        ];
        bytes.extend(self.load.raw().to_le_bytes());
        bytes.extend(self.store.raw().to_le_bytes());
        for channel in self.clear {
            bytes.extend(channel.to_bits().to_le_bytes());
        }
        // Only where they are not Metal's default, so an attachment that states none fingerprints
        // to the bytes it always has.
        if self.options != StoreActionOptions::NONE {
            bytes.extend(self.options.raw().to_le_bytes());
        }
        bytes
    }
}

/// One viewport: `MTLViewport`'s six numbers.
///
/// The depths are kept beside the rectangle because `znear` and `zfar` are part of the same Metal
/// call, and a case that sets a viewport without saying what it does to depth would be two
/// questions in one.
///
/// Two viewports are the same viewport when their six numbers hold the same bits, which is how the
/// input fingerprint reads them too ([`ViewportRect::words`]). Comparing the bits rather than the
/// numbers keeps the equality reflexive, so the type is honestly [`Eq`] despite holding `f64`s, and
/// it is what a case wants: a viewport edge one `f64` step away is a different viewport.
#[derive(Clone, Copy, Debug)]
pub struct ViewportRect {
    /// `originX` and `originY`, in pixels.
    pub origin: [f64; 2],
    /// `width` and `height`, in pixels; a negative extent mirrors the draw, as Metal's does.
    pub size: [f64; 2],
    /// `znear` and `zfar`.
    pub depth: [f64; 2],
}

impl PartialEq for ViewportRect {
    fn eq(&self, other: &Self) -> bool {
        self.words() == other.words()
    }
}

impl Eq for ViewportRect {}

impl ViewportRect {
    /// A viewport of `origin` and `size` over the whole depth range.
    pub const fn new(origin: [f64; 2], size: [f64; 2]) -> Self {
        Self {
            origin,
            size,
            depth: [0.0, 1.0],
        }
    }

    /// The words a case's input fingerprint covers: every number by its bits, so a viewport edge
    /// that moves by one `f64` step is a change to the case's inputs.
    pub fn words(&self) -> Vec<u8> {
        self.origin
            .iter()
            .chain(&self.size)
            .chain(&self.depth)
            .flat_map(|number| number.to_bits().to_le_bytes())
            .collect()
    }
}

/// `setBlendColor:green:blue:alpha:`'s four channels, which a `BlendColor` or `BlendAlpha` blend
/// factor reads.
///
/// Its own type for the same reason [`ViewportRect`] is: two blend colours are the same blend
/// colour when their four numbers hold the same bits, which is how the input fingerprint reads them
/// too ([`BlendColorChannels::words`]). Comparing the bits rather than the numbers keeps the
/// equality reflexive, so the type is honestly [`Eq`] despite holding `f32`s -- and [`RenderStep`]
/// keeps its `Eq` derive, which a bare `[f32; 4]` in a variant would have cost it.
#[derive(Clone, Copy, Debug)]
pub struct BlendColorChannels(pub [f32; 4]);

impl PartialEq for BlendColorChannels {
    fn eq(&self, other: &Self) -> bool {
        self.words() == other.words()
    }
}

impl Eq for BlendColorChannels {}

impl BlendColorChannels {
    /// The four channels, red first.
    pub const fn new(channels: [f32; 4]) -> Self {
        Self(channels)
    }

    /// The words a case's input fingerprint covers: every channel by its bits, so a blend colour
    /// that moves by one `f32` step is a change to the case's inputs.
    pub fn words(&self) -> Vec<u8> {
        self.0
            .iter()
            .flat_map(|channel| channel.to_bits().to_le_bytes())
            .collect()
    }
}

/// `MTLScissorRect`: the pixels of the target a draw may write, origin at the top left.
///
/// Four whole pixels, so the type is honestly [`Eq`] and the fingerprint reads exactly what the
/// case wrote ([`ScissorBox::words`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScissorBox {
    /// `x`, in pixels.
    pub x: u64,
    /// `y`, in pixels.
    pub y: u64,
    /// `width`, in pixels.
    pub width: u64,
    /// `height`, in pixels.
    pub height: u64,
}

impl ScissorBox {
    /// The rectangle of these four numbers.
    pub const fn new(x: u64, y: u64, width: u64, height: u64) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// Whether the rectangle reaches past a target of `size` texels, which is the one shape of
    /// `setScissorRect:` Metal's own validation calls a programming error.
    ///
    /// Decided from the case's own numbers, so a side reaches the answer without a device. The
    /// suite never hands Metal such a rectangle: what plain Metal's driver programs for one is
    /// unmeasured and it reaches the rasterizer's own clip, which is not a thing a committed case
    /// finds out by trying.
    pub fn past(&self, size: [u32; 2]) -> bool {
        self.x + self.width > u64::from(size[0]) || self.y + self.height > u64::from(size[1])
    }

    /// The words a case's input fingerprint covers.
    pub fn words(&self) -> Vec<u8> {
        [self.x, self.y, self.width, self.height]
            .iter()
            .flat_map(|number| number.to_le_bytes())
            .collect()
    }
}

/// `setDepthBias:slopeScale:clamp:`'s three numbers.
///
/// Its own type for the reason [`BlendColorChannels`] is: two biases are the same bias when their
/// three numbers hold the same bits, which is how the input fingerprint reads them too, so the
/// type is honestly [`Eq`] despite holding `f32`s and [`RenderStep`] keeps its `Eq` derive.
#[derive(Clone, Copy, Debug)]
pub struct DepthBiasValues {
    /// `depthBias`: the constant offset, in the depth format's own minimum resolvable difference.
    pub bias: f32,
    /// `slopeScale`: what the primitive's largest depth slope is multiplied by.
    pub slope_scale: f32,
    /// `clamp`: the bound the whole offset is held to, or 0 for no bound.
    pub clamp: f32,
}

impl PartialEq for DepthBiasValues {
    fn eq(&self, other: &Self) -> bool {
        self.words() == other.words()
    }
}

impl Eq for DepthBiasValues {}

impl DepthBiasValues {
    /// The bias of these three numbers.
    pub const fn new(bias: f32, slope_scale: f32, clamp: f32) -> Self {
        Self {
            bias,
            slope_scale,
            clamp,
        }
    }

    /// The words a case's input fingerprint covers: every number by its bits, so a bias one `f32`
    /// step away is a different input.
    pub fn words(&self) -> Vec<u8> {
        [self.bias, self.slope_scale, self.clamp]
            .iter()
            .flat_map(|number| number.to_bits().to_le_bytes())
            .collect()
    }
}

/// The `lodMinClamp` and `lodMaxClamp` a sampler binding carries at an encoder slot.
///
/// Its own type for the reason [`DepthBiasValues`] is: the pair compares by its bits, so a step
/// that carries one is honestly [`Eq`].
#[derive(Clone, Copy, Debug)]
pub struct LodClamp {
    pub min: f32,
    pub max: f32,
}

impl PartialEq for LodClamp {
    fn eq(&self, other: &Self) -> bool {
        self.words() == other.words()
    }
}

impl Eq for LodClamp {}

impl LodClamp {
    /// The clamps `min` to `max`.
    pub const fn new(min: f32, max: f32) -> Self {
        Self { min, max }
    }

    /// The words a case's input fingerprint covers, each by its bits.
    pub fn words(&self) -> Vec<u8> {
        [self.min, self.max]
            .iter()
            .flat_map(|number| number.to_bits().to_le_bytes())
            .collect()
    }
}

/// A resource a residency or a barrier step names: which pool the handle belongs to.
///
/// `useResource:usage:stages:` and `memoryBarrierWithResources:count:afterStages:beforeStages:`
/// take an `MTLResource`, which a buffer and a texture both are, and a handle alone cannot say
/// which pool it came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepResource {
    Buffer(Handle),
    Texture(Handle),
}

impl StepResource {
    /// The word the fingerprint carries for the pool, so a buffer handle and a texture handle of
    /// the same number are two different inputs.
    pub fn word(&self) -> u8 {
        match self {
            StepResource::Buffer(_) => 0,
            StepResource::Texture(_) => 1,
        }
    }
}

/// One command a case encodes into a render pass, in the order it lists them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RenderStep {
    /// `setRenderPipelineState:`.
    Pipeline(Handle),
    /// `setDepthStencilState:` of a state from
    /// [`Side::depth_stencil_state`](crate::case::Side::depth_stencil_state).
    DepthStencilState(Handle),
    /// `setStencilReferenceValue:`.
    StencilReference(u32),
    /// `setVertexBuffer:offset:atIndex:`.
    VertexBuffer {
        buffer: Handle,
        offset: u64,
        index: u32,
    },
    /// `setFragmentBuffer:offset:atIndex:`.
    FragmentBuffer {
        buffer: Handle,
        offset: u64,
        index: u32,
    },
    /// `setFragmentTexture:atIndex:`.
    FragmentTexture { texture: Handle, index: u32 },
    /// `setFragmentSamplerState:atIndex:`.
    FragmentSampler { sampler: Handle, index: u32 },
    /// `setVertexBytes:length:atIndex:`: a vertex binding whose bytes the call copies.
    ///
    /// A case uses this where the binding is a handful of bytes that change every draw — a point's
    /// two coordinates — because a buffer a draw would be hundreds of allocations saying the same
    /// thing, and because the inline binding is its own Metal path.
    VertexBytes { bytes: Vec<u8>, index: u32 },
    /// `setViewport:`.
    Viewport(ViewportRect),
    /// `setBlendColor:green:blue:alpha:`, the four channels a `BlendColor` or `BlendAlpha` factor
    /// reads. An encoder call rather than a field of the pass, so a case states it where it makes
    /// it.
    BlendColor(BlendColorChannels),
    /// `setVisibilityResultMode:offset:`. A pass that sets one needs a visibility-result buffer
    /// ([`RenderPass::counting_into`]); setting it without one is a refusal a case may be asking
    /// about.
    VisibilityResultMode {
        mode: VisibilityResultMode,
        offset: u64,
    },
    /// `drawPrimitives:vertexStart:vertexCount:instanceCount:baseInstance:`, whose base instance
    /// is 0.
    Draw {
        primitive: PrimitiveType,
        first: u32,
        count: u32,
        instances: u32,
    },
    /// `drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:
    /// instanceCount:baseVertex:baseInstance:`.
    DrawIndexed {
        primitive: PrimitiveType,
        count: u32,
        index_type: IndexType,
        indices: Handle,
        offset: u64,
        instances: u32,
        base_vertex: i32,
        base_instance: u32,
    },
    /// `drawPrimitives:indirectBuffer:indirectBufferOffset:`, whose buffer holds the draw's four
    /// 32-bit arguments.
    DrawIndirect {
        primitive: PrimitiveType,
        arguments: Handle,
        offset: u64,
    },
    /// `drawPrimitives:vertexStart:vertexCount:`, the form that names no instance count at all.
    ///
    /// Metal's shortest draw. It is a call of its own rather than [`RenderStep::Draw`] with one
    /// instance: the two selectors are two entry points of the driver, and a case that asks what
    /// the short form does may not ask it through the long one.
    DrawPlain {
        primitive: PrimitiveType,
        first: u32,
        count: u32,
    },
    /// `drawPrimitives:vertexStart:vertexCount:instanceCount:`, which names the instances and
    /// leaves the base instance to Metal's own 0, where [`RenderStep::Draw`] passes it.
    DrawInstanced {
        primitive: PrimitiveType,
        first: u32,
        count: u32,
        instances: u32,
    },
    /// `drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:`, the shortest
    /// indexed draw: no instance count, no base vertex and no base instance.
    DrawIndexedPlain {
        primitive: PrimitiveType,
        count: u32,
        index_type: IndexType,
        indices: Handle,
        offset: u64,
    },
    /// `drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:instanceCount:`,
    /// which names the instances and leaves the base vertex and base instance at Metal's 0.
    DrawIndexedInstanced {
        primitive: PrimitiveType,
        count: u32,
        index_type: IndexType,
        indices: Handle,
        offset: u64,
        instances: u32,
    },
    /// `drawIndexedPrimitives:indexType:indexBuffer:indexBufferOffset:indirectBuffer:
    /// indirectBufferOffset:`, whose argument buffer holds the draw's five 32-bit arguments:
    /// `indexCount`, `instanceCount`, `indexStart`, `baseVertex` (signed) and `baseInstance`.
    DrawIndexedIndirect {
        primitive: PrimitiveType,
        index_type: IndexType,
        indices: Handle,
        /// `indexBufferOffset`: where the draw's indices start, which is the encoder's parameter
        /// rather than the argument buffer's.
        index_offset: u64,
        arguments: Handle,
        offset: u64,
    },
    /// `useResource:usage:stages:` of a buffer, which is how a resource an **indirect** command's
    /// functions reach is made resident: a command's buffer bindings are addresses, so Metal
    /// tracks nothing for them and the pass must say what they are.
    UseBuffer {
        buffer: Handle,
        /// `MTLResourceUsageWrite` beside `MTLResourceUsageRead`.
        write: bool,
        /// The raw `MTLRenderStages` bits ([`crate::case::STAGE_VERTEX`] and the rest).
        stages: u64,
    },
    /// `executeCommandsInBuffer:withRange:` of `count` commands from `first`.
    ExecuteCommands {
        commands: Handle,
        first: u64,
        count: u64,
    },
    /// `setScissorRect:`.
    Scissor(ScissorBox),
    /// `setScissorRects:count:`. A draw reads the rectangle its `[[viewport_array_index]]` names,
    /// and 0 from a vertex function that writes none, which is the fact a case with one such
    /// function can state.
    Scissors(Vec<ScissorBox>),
    /// `setViewports:count:`, read as [`RenderStep::Scissors`] is.
    Viewports(Vec<ViewportRect>),
    /// `setCullMode:`.
    Cull(CullMode),
    /// `setFrontFacingWinding:`.
    FrontFacing(Winding),
    /// `setTriangleFillMode:`.
    FillMode(TriangleFillMode),
    /// `setDepthClipMode:`.
    DepthClip(DepthClipMode),
    /// `setDepthBias:slopeScale:clamp:`.
    DepthBias(DepthBiasValues),
    /// `setStencilFrontReferenceValue:backReferenceValue:`, the two faces' references apart,
    /// which [`RenderStep::StencilReference`] sets together.
    StencilReferences { front: u32, back: u32 },
    /// `setVertexBufferOffset:atIndex:`: the offset of whatever is bound at the index, with the
    /// binding itself left alone.
    VertexBufferOffset { offset: u64, index: u32 },
    /// `setFragmentBufferOffset:atIndex:`.
    FragmentBufferOffset { offset: u64, index: u32 },
    /// `setFragmentBytes:length:atIndex:`: a fragment binding whose bytes the call copies.
    FragmentBytes { bytes: Vec<u8>, index: u32 },
    /// `setVertexBuffers:offsets:withRange:`: each buffer and its offset, from `first`.
    VertexBuffers {
        buffers: Vec<(Handle, u64)>,
        first: u32,
    },
    /// `setFragmentBuffers:offsets:withRange:`.
    FragmentBuffers {
        buffers: Vec<(Handle, u64)>,
        first: u32,
    },
    /// `setVertexTexture:atIndex:`.
    VertexTexture { texture: Handle, index: u32 },
    /// `setVertexTextures:withRange:`.
    VertexTextures { textures: Vec<Handle>, first: u32 },
    /// `setFragmentTextures:withRange:`.
    FragmentTextures { textures: Vec<Handle>, first: u32 },
    /// `setVertexSamplerState:atIndex:`.
    VertexSampler { sampler: Handle, index: u32 },
    /// `setVertexSamplerStates:withRange:`.
    VertexSamplers { samplers: Vec<Handle>, first: u32 },
    /// `setFragmentSamplerStates:withRange:`.
    FragmentSamplers { samplers: Vec<Handle>, first: u32 },
    /// `setVertexSamplerState:lodMinClamp:lodMaxClamp:atIndex:`.
    VertexSamplerClamped {
        sampler: Handle,
        clamp: LodClamp,
        index: u32,
    },
    /// `setFragmentSamplerState:lodMinClamp:lodMaxClamp:atIndex:`.
    FragmentSamplerClamped {
        sampler: Handle,
        clamp: LodClamp,
        index: u32,
    },
    /// `setVertexSamplerStates:lodMinClamps:lodMaxClamps:withRange:`.
    VertexSamplersClamped {
        samplers: Vec<(Handle, LodClamp)>,
        first: u32,
    },
    /// `setFragmentSamplerStates:lodMinClamps:lodMaxClamps:withRange:`.
    FragmentSamplersClamped {
        samplers: Vec<(Handle, LodClamp)>,
        first: u32,
    },
    /// `memoryBarrierWithScope:afterStages:beforeStages:`.
    BarrierScope {
        scope: BarrierScope,
        after: RenderStages,
        before: RenderStages,
    },
    /// `memoryBarrierWithResources:count:afterStages:beforeStages:`.
    BarrierResources {
        resources: Vec<StepResource>,
        after: RenderStages,
        before: RenderStages,
    },
    /// `useResource:usage:`, Metal's deprecated form, which names every render stage.
    UseResource {
        resource: StepResource,
        usage: ResourceUsage,
    },
    /// `useResource:usage:stages:`.
    UseResourceInStages {
        resource: StepResource,
        usage: ResourceUsage,
        stages: RenderStages,
    },
    /// `useResources:count:usage:`, Metal's deprecated form.
    UseResources {
        resources: Vec<StepResource>,
        usage: ResourceUsage,
    },
    /// `useResources:count:usage:stages:`.
    UseResourcesInStages {
        resources: Vec<StepResource>,
        usage: ResourceUsage,
        stages: RenderStages,
    },
    /// `useHeap:`, Metal's deprecated form.
    UseHeap(Handle),
    /// `useHeap:stages:`.
    UseHeapInStages { heap: Handle, stages: RenderStages },
    /// `useHeaps:count:`, Metal's deprecated form.
    UseHeaps(Vec<Handle>),
    /// `useHeaps:count:stages:`.
    UseHeapsInStages {
        heaps: Vec<Handle>,
        stages: RenderStages,
    },
    /// `setTessellationFactorBuffer:offset:instanceStride:`, the buffer a patch draw of a
    /// tessellation pipeline reads its edge and inside factors from. `None` is Metal's nil.
    TessellationFactorBuffer {
        buffer: Option<Handle>,
        offset: u64,
        instance_stride: u64,
    },
    /// `setTessellationFactorScale:`, which every factor of a pipeline with
    /// `tessellationFactorScaleEnabled` is multiplied by. An encoder that never makes the call
    /// holds no scale, which is its own shape: Metal culls every patch of such a draw.
    ///
    /// The scale is held as its `f32` bits, so a case can state one NaN payload rather than
    /// whichever `f32::NAN` is, and two steps compare and fingerprint by those bits.
    TessellationFactorScale { bits: u32 },
    /// `drawPatches:patchStart:patchCount:patchIndexBuffer:patchIndexBufferOffset:instanceCount:
    /// baseInstance:`: `count` patches of `control_points` control points from `first`, `instances`
    /// instances from `base_instance`, each patch's index taken from `patch_indices` where one is
    /// given and from its own position where it is nil.
    DrawPatches {
        control_points: u32,
        first: u32,
        count: u32,
        patch_indices: Option<Handle>,
        patch_index_offset: u64,
        instances: u32,
        base_instance: u32,
    },
    /// `drawIndexedPatches:patchStart:patchCount:patchIndexBuffer:patchIndexBufferOffset:
    /// controlPointIndexBuffer:controlPointIndexBufferOffset:instanceCount:baseInstance:`, which
    /// reads control point `k` of patch `p` at position `p · control_points + k` of the
    /// control-point index buffer, as the index type the pipeline states.
    DrawIndexedPatches {
        control_points: u32,
        first: u32,
        count: u32,
        patch_indices: Option<Handle>,
        patch_index_offset: u64,
        control_point_indices: Handle,
        control_point_index_offset: u64,
        instances: u32,
        base_instance: u32,
    },
}

impl RenderStep {
    /// The fullscreen triangle every fill fixture here draws: three vertices of one instance.
    pub const fn triangle() -> Self {
        Self::Draw {
            primitive: PrimitiveType::Triangle,
            first: 0,
            count: 3,
            instances: 1,
        }
    }

    /// The same triangle from vertex 3, which covers no pixel: how a visibility case asks what a
    /// draw that passes no sample writes.
    pub const fn no_triangle() -> Self {
        Self::Draw {
            primitive: PrimitiveType::Triangle,
            first: 3,
            count: 3,
            instances: 1,
        }
    }

    /// A vertex buffer bound at `index`, from its start.
    pub const fn vertex_buffer(buffer: Handle, index: u32) -> Self {
        Self::VertexBuffer {
            buffer,
            offset: 0,
            index,
        }
    }

    /// A fragment buffer bound at `index`, from its start.
    pub const fn fragment_buffer(buffer: Handle, index: u32) -> Self {
        Self::FragmentBuffer {
            buffer,
            offset: 0,
            index,
        }
    }

    /// The site name the step's answer is emitted at: the call it makes, so a refusal reads as the
    /// Metal call we refused.
    pub fn call(&self) -> &'static str {
        match self {
            RenderStep::Pipeline(_) => "setRenderPipelineState",
            RenderStep::DepthStencilState(_) => "setDepthStencilState",
            RenderStep::StencilReference(_) => "setStencilReferenceValue",
            RenderStep::VertexBuffer { .. } => "setVertexBuffer",
            RenderStep::FragmentBuffer { .. } => "setFragmentBuffer",
            RenderStep::FragmentTexture { .. } => "setFragmentTexture",
            RenderStep::FragmentSampler { .. } => "setFragmentSamplerState",
            RenderStep::VertexBytes { .. } => "setVertexBytes",
            RenderStep::Viewport(_) => "setViewport",
            RenderStep::BlendColor(_) => "setBlendColor",
            RenderStep::VisibilityResultMode { .. } => "setVisibilityResultMode",
            RenderStep::Draw { .. } => "drawPrimitives",
            RenderStep::DrawIndexed { .. } => "drawIndexedPrimitives",
            RenderStep::DrawIndirect { .. } => "drawPrimitivesWithIndirectBuffer",
            RenderStep::DrawPlain { .. } => "drawPrimitivesPlain",
            RenderStep::DrawInstanced { .. } => "drawPrimitivesInstanced",
            RenderStep::DrawIndexedPlain { .. } => "drawIndexedPrimitivesPlain",
            RenderStep::DrawIndexedInstanced { .. } => "drawIndexedPrimitivesInstanced",
            RenderStep::DrawIndexedIndirect { .. } => "drawIndexedPrimitivesWithIndirectBuffer",
            // `UseBuffer` is an indirect command's own residency call and names the same site
            // as `RenderStep::UseResource` below. No case holds both, and a case that comes to
            // hold both has to rename one and re-record it: the site a step is emitted at is
            // this name, and two siblings sharing one compares no bytes of either.
            RenderStep::UseBuffer { .. } => "useResource",
            RenderStep::ExecuteCommands { .. } => "executeCommandsInBuffer",
            RenderStep::Scissor(_) => "setScissorRect",
            RenderStep::Scissors(_) => "setScissorRects",
            RenderStep::Viewports(_) => "setViewports",
            RenderStep::Cull(_) => "setCullMode",
            RenderStep::FrontFacing(_) => "setFrontFacingWinding",
            RenderStep::FillMode(_) => "setTriangleFillMode",
            RenderStep::DepthClip(_) => "setDepthClipMode",
            RenderStep::DepthBias(_) => "setDepthBias",
            RenderStep::StencilReferences { .. } => "setStencilFrontReferenceValue",
            RenderStep::VertexBufferOffset { .. } => "setVertexBufferOffset",
            RenderStep::FragmentBufferOffset { .. } => "setFragmentBufferOffset",
            RenderStep::FragmentBytes { .. } => "setFragmentBytes",
            RenderStep::VertexBuffers { .. } => "setVertexBuffers",
            RenderStep::FragmentBuffers { .. } => "setFragmentBuffers",
            RenderStep::VertexTexture { .. } => "setVertexTexture",
            RenderStep::VertexTextures { .. } => "setVertexTextures",
            RenderStep::FragmentTextures { .. } => "setFragmentTextures",
            RenderStep::VertexSampler { .. } => "setVertexSamplerState",
            RenderStep::VertexSamplers { .. } => "setVertexSamplerStates",
            RenderStep::FragmentSamplers { .. } => "setFragmentSamplerStates",
            RenderStep::VertexSamplerClamped { .. } => "setVertexSamplerStateWithLodClamps",
            RenderStep::FragmentSamplerClamped { .. } => "setFragmentSamplerStateWithLodClamps",
            RenderStep::VertexSamplersClamped { .. } => "setVertexSamplerStatesWithLodClamps",
            RenderStep::FragmentSamplersClamped { .. } => "setFragmentSamplerStatesWithLodClamps",
            RenderStep::BarrierScope { .. } => "memoryBarrierWithScope",
            RenderStep::BarrierResources { .. } => "memoryBarrierWithResources",
            RenderStep::UseResource { .. } => "useResource",
            RenderStep::UseResourceInStages { .. } => "useResourceWithStages",
            RenderStep::UseResources { .. } => "useResources",
            RenderStep::UseResourcesInStages { .. } => "useResourcesWithStages",
            RenderStep::UseHeap(_) => "useHeap",
            RenderStep::UseHeapInStages { .. } => "useHeapWithStages",
            RenderStep::UseHeaps(_) => "useHeaps",
            RenderStep::UseHeapsInStages { .. } => "useHeapsWithStages",
            RenderStep::TessellationFactorBuffer { .. } => "setTessellationFactorBuffer",
            RenderStep::TessellationFactorScale { .. } => "setTessellationFactorScale",
            RenderStep::DrawPatches { .. } => "drawPatches",
            RenderStep::DrawIndexedPatches { .. } => "drawIndexedPatches",
        }
    }

    /// The parameters the step's own call carries, for a case's input fingerprint: everything but
    /// the handles, whose numbers are a side's own.
    pub fn parameters(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        match self {
            RenderStep::Pipeline(_) | RenderStep::DepthStencilState(_) => {}
            RenderStep::StencilReference(value) => bytes.extend(value.to_le_bytes()),
            RenderStep::VertexBuffer { offset, index, .. }
            | RenderStep::FragmentBuffer { offset, index, .. } => {
                bytes.extend(offset.to_le_bytes());
                bytes.extend(index.to_le_bytes());
            }
            // A texture or a sampler a step names is fingerprinted by the case through its own
            // texels or its own descriptor words, as a buffer is, so only the binding index is the
            // call's own parameter here.
            RenderStep::FragmentTexture { index, .. }
            | RenderStep::FragmentSampler { index, .. } => bytes.extend(index.to_le_bytes()),
            // The inline bytes and the viewport are the case's own values rather than a handle, so
            // they are part of what the case feeds its run and the fingerprint covers them whole.
            RenderStep::VertexBytes {
                bytes: inline,
                index,
            } => {
                bytes.extend(inline);
                bytes.extend(index.to_le_bytes());
            }
            RenderStep::Viewport(rect) => bytes.extend(rect.words()),
            RenderStep::BlendColor(channels) => bytes.extend(channels.words()),
            RenderStep::VisibilityResultMode { mode, offset } => {
                bytes.extend(mode.raw().to_le_bytes());
                bytes.extend(offset.to_le_bytes());
            }
            RenderStep::Draw {
                primitive,
                first,
                count,
                instances,
            } => {
                bytes.extend(primitive.as_raw().to_le_bytes());
                bytes.extend(first.to_le_bytes());
                bytes.extend(count.to_le_bytes());
                bytes.extend(instances.to_le_bytes());
            }
            RenderStep::DrawIndexed {
                primitive,
                count,
                index_type,
                offset,
                instances,
                base_vertex,
                base_instance,
                ..
            } => {
                bytes.extend(primitive.as_raw().to_le_bytes());
                bytes.extend(count.to_le_bytes());
                bytes.extend(index_type.as_raw().to_le_bytes());
                bytes.extend(offset.to_le_bytes());
                bytes.extend(instances.to_le_bytes());
                bytes.extend(base_vertex.to_le_bytes());
                bytes.extend(base_instance.to_le_bytes());
            }
            RenderStep::DrawIndirect {
                primitive, offset, ..
            } => {
                bytes.extend(primitive.as_raw().to_le_bytes());
                bytes.extend(offset.to_le_bytes());
            }
            RenderStep::DrawPlain {
                primitive,
                first,
                count,
            } => {
                bytes.extend(primitive.as_raw().to_le_bytes());
                bytes.extend(first.to_le_bytes());
                bytes.extend(count.to_le_bytes());
            }
            RenderStep::DrawInstanced {
                primitive,
                first,
                count,
                instances,
            } => {
                bytes.extend(primitive.as_raw().to_le_bytes());
                bytes.extend(first.to_le_bytes());
                bytes.extend(count.to_le_bytes());
                bytes.extend(instances.to_le_bytes());
            }
            RenderStep::DrawIndexedPlain {
                primitive,
                count,
                index_type,
                offset,
                ..
            } => {
                bytes.extend(primitive.as_raw().to_le_bytes());
                bytes.extend(count.to_le_bytes());
                bytes.extend(index_type.as_raw().to_le_bytes());
                bytes.extend(offset.to_le_bytes());
            }
            RenderStep::DrawIndexedInstanced {
                primitive,
                count,
                index_type,
                offset,
                instances,
                ..
            } => {
                bytes.extend(primitive.as_raw().to_le_bytes());
                bytes.extend(count.to_le_bytes());
                bytes.extend(index_type.as_raw().to_le_bytes());
                bytes.extend(offset.to_le_bytes());
                bytes.extend(instances.to_le_bytes());
            }
            // The five arguments of an indexed indirect draw are the **argument buffer's** bytes,
            // which the case fingerprints as its own buffer contents, so the call's own parameters
            // are the primitive, the index type and the two offsets.
            RenderStep::DrawIndexedIndirect {
                primitive,
                index_type,
                index_offset,
                offset,
                ..
            } => {
                bytes.extend(primitive.as_raw().to_le_bytes());
                bytes.extend(index_type.as_raw().to_le_bytes());
                bytes.extend(index_offset.to_le_bytes());
                bytes.extend(offset.to_le_bytes());
            }
            RenderStep::UseBuffer { write, stages, .. } => {
                bytes.push(u8::from(*write));
                bytes.extend(stages.to_le_bytes());
            }
            // The commands a range executes are fingerprinted by the case through the buffer's
            // own descriptor and every command's steps, so only the range is the call's own.
            RenderStep::ExecuteCommands { first, count, .. } => {
                bytes.extend(first.to_le_bytes());
                bytes.extend(count.to_le_bytes());
            }
            // The state setters below carry no handle, so the whole call is the case's own value
            // and the fingerprint covers all of it.
            RenderStep::Scissor(rect) => bytes.extend(rect.words()),
            RenderStep::Scissors(rects) => {
                for rect in rects {
                    bytes.extend(rect.words());
                }
            }
            RenderStep::Viewports(rects) => {
                for rect in rects {
                    bytes.extend(rect.words());
                }
            }
            RenderStep::Cull(mode) => bytes.extend(mode.as_raw().to_le_bytes()),
            RenderStep::FrontFacing(winding) => bytes.extend(winding.as_raw().to_le_bytes()),
            RenderStep::FillMode(mode) => bytes.extend(mode.as_raw().to_le_bytes()),
            RenderStep::DepthClip(mode) => bytes.extend(mode.as_raw().to_le_bytes()),
            RenderStep::DepthBias(values) => bytes.extend(values.words()),
            RenderStep::StencilReferences { front, back } => {
                bytes.extend(front.to_le_bytes());
                bytes.extend(back.to_le_bytes());
            }
            RenderStep::VertexBufferOffset { offset, index }
            | RenderStep::FragmentBufferOffset { offset, index } => {
                bytes.extend(offset.to_le_bytes());
                bytes.extend(index.to_le_bytes());
            }
            RenderStep::FragmentBytes {
                bytes: inline,
                index,
            } => {
                bytes.extend(inline);
                bytes.extend(index.to_le_bytes());
            }
            // A buffer, texture or sampler a range names is fingerprinted by the case through its
            // own bytes or its own descriptor words, as a single binding's is, so the call's own
            // parameters are the offsets, the clamps, the count and where the range starts. The
            // count is written because a range of two bindings and a range of three are two
            // different calls.
            RenderStep::VertexBuffers { buffers, first }
            | RenderStep::FragmentBuffers { buffers, first } => {
                bytes.extend((buffers.len() as u32).to_le_bytes());
                for (_, offset) in buffers {
                    bytes.extend(offset.to_le_bytes());
                }
                bytes.extend(first.to_le_bytes());
            }
            RenderStep::VertexTexture { index, .. } | RenderStep::VertexSampler { index, .. } => {
                bytes.extend(index.to_le_bytes());
            }
            RenderStep::VertexTextures { textures, first }
            | RenderStep::FragmentTextures { textures, first } => {
                bytes.extend((textures.len() as u32).to_le_bytes());
                bytes.extend(first.to_le_bytes());
            }
            RenderStep::VertexSamplers { samplers, first }
            | RenderStep::FragmentSamplers { samplers, first } => {
                bytes.extend((samplers.len() as u32).to_le_bytes());
                bytes.extend(first.to_le_bytes());
            }
            RenderStep::VertexSamplerClamped { clamp, index, .. }
            | RenderStep::FragmentSamplerClamped { clamp, index, .. } => {
                bytes.extend(clamp.words());
                bytes.extend(index.to_le_bytes());
            }
            RenderStep::VertexSamplersClamped { samplers, first }
            | RenderStep::FragmentSamplersClamped { samplers, first } => {
                bytes.extend((samplers.len() as u32).to_le_bytes());
                for (_, clamp) in samplers {
                    bytes.extend(clamp.words());
                }
                bytes.extend(first.to_le_bytes());
            }
            RenderStep::BarrierScope {
                scope,
                after,
                before,
            } => {
                bytes.extend(scope.as_raw().to_le_bytes());
                bytes.extend(after.as_raw().to_le_bytes());
                bytes.extend(before.as_raw().to_le_bytes());
            }
            // Which pool each resource came from is the call's own parameter: the handle is a
            // side's own number, and a buffer and a texture of the same number are two calls.
            RenderStep::BarrierResources {
                resources,
                after,
                before,
            } => {
                bytes.extend((resources.len() as u32).to_le_bytes());
                bytes.extend(resources.iter().map(StepResource::word));
                bytes.extend(after.as_raw().to_le_bytes());
                bytes.extend(before.as_raw().to_le_bytes());
            }
            RenderStep::UseResource { resource, usage } => {
                bytes.push(resource.word());
                bytes.extend(usage.as_raw().to_le_bytes());
            }
            RenderStep::UseResourceInStages {
                resource,
                usage,
                stages,
            } => {
                bytes.push(resource.word());
                bytes.extend(usage.as_raw().to_le_bytes());
                bytes.extend(stages.as_raw().to_le_bytes());
            }
            RenderStep::UseResources { resources, usage } => {
                bytes.extend((resources.len() as u32).to_le_bytes());
                bytes.extend(resources.iter().map(StepResource::word));
                bytes.extend(usage.as_raw().to_le_bytes());
            }
            RenderStep::UseResourcesInStages {
                resources,
                usage,
                stages,
            } => {
                bytes.extend((resources.len() as u32).to_le_bytes());
                bytes.extend(resources.iter().map(StepResource::word));
                bytes.extend(usage.as_raw().to_le_bytes());
                bytes.extend(stages.as_raw().to_le_bytes());
            }
            RenderStep::UseHeap(_) => {}
            RenderStep::UseHeapInStages { stages, .. } => {
                bytes.extend(stages.as_raw().to_le_bytes());
            }
            RenderStep::UseHeaps(heaps) => bytes.extend((heaps.len() as u32).to_le_bytes()),
            RenderStep::UseHeapsInStages { heaps, stages } => {
                bytes.extend((heaps.len() as u32).to_le_bytes());
                bytes.extend(stages.as_raw().to_le_bytes());
            }
            RenderStep::TessellationFactorBuffer {
                buffer,
                offset,
                instance_stride,
            } => {
                // Whether the call names a buffer at all is the call's own parameter and not a
                // handle: a nil factor buffer is a shape of its own.
                bytes.push(u8::from(buffer.is_some()));
                bytes.extend(offset.to_le_bytes());
                bytes.extend(instance_stride.to_le_bytes());
            }
            // The scale's bits and not its value: a case states NaNs whose payloads differ, and a
            // fingerprint that read them through `f32` equality would key two of them alike.
            RenderStep::TessellationFactorScale { bits } => bytes.extend(bits.to_le_bytes()),
            RenderStep::DrawPatches {
                control_points,
                first,
                count,
                patch_indices,
                patch_index_offset,
                instances,
                base_instance,
            } => {
                bytes.extend(control_points.to_le_bytes());
                bytes.extend(first.to_le_bytes());
                bytes.extend(count.to_le_bytes());
                bytes.push(u8::from(patch_indices.is_some()));
                bytes.extend(patch_index_offset.to_le_bytes());
                bytes.extend(instances.to_le_bytes());
                bytes.extend(base_instance.to_le_bytes());
            }
            RenderStep::DrawIndexedPatches {
                control_points,
                first,
                count,
                patch_indices,
                patch_index_offset,
                control_point_index_offset,
                instances,
                base_instance,
                ..
            } => {
                bytes.extend(control_points.to_le_bytes());
                bytes.extend(first.to_le_bytes());
                bytes.extend(count.to_le_bytes());
                bytes.push(u8::from(patch_indices.is_some()));
                bytes.extend(patch_index_offset.to_le_bytes());
                bytes.extend(control_point_index_offset.to_le_bytes());
                bytes.extend(instances.to_le_bytes());
                bytes.extend(base_instance.to_le_bytes());
            }
        }
        bytes
    }
}

/// The Metal call a pass's custom sample positions are keyed by in a case's input fingerprint:
/// `MTLRenderPassDescriptor`'s own selector, so the positions key by their call's name rather than
/// by a place in the words.
pub const SAMPLE_POSITIONS_CALL: &str = "setSamplePositions";

/// The two sample-position shapes the **Metal leg** must never encode, refused before anything is
/// made, decided from the positions themselves and the pass's sample count alone.
///
/// Both come from a measurement rather than from caution, and both are shapes a case is forbidden
/// to hand Metal:
///
/// - **More positions than the pass's samples.** Measured on an Apple M3 Max: plain Metal ends the
///   process past the pass's samples
///   (`metal2vulkan_runtime::RenderPassProblem::SamplePositionsSamples`). There is no safe way to
///   ask Metal this, so there is no answer to record and the shape may not be built.
/// - **A coordinate outside `[0, 1)`.** Metal documents the range and its API validation asserts on
///   another coordinate; what plain Metal's driver programs for one is unmeasured, and the value
///   reaches the rasterizer's own sample grid. The suite does not find that out by trying it.
///
/// It refuses **nothing** where the limit does not apply: no positions at all is Metal's default,
/// and a count at or below the pass's samples with every coordinate inside the pixel is a shape the
/// probes behind this family already ran on plain Metal. Fewer positions than samples is one of
/// those: Metal puts the samples past them at the pixel's corner, which is a recorded answer rather
/// than a hazard.
///
/// Our own leg is not guarded here and must not be: the runtime refuses both shapes typed, and that
/// typed refusal is what a case compares.
pub fn forbidden_positions(positions: &[SamplePosition], samples: u32) -> Option<Failure> {
    if positions.len() as u64 > u64::from(samples) {
        return Some(Failure::named(
            "MorePositionsThanSamples",
            format!(
                "{} custom sample positions in a pass of {samples} sample(s), past which plain \
                 Metal ends the process",
                positions.len()
            ),
        ));
    }
    if let Some((at, position)) = positions
        .iter()
        .enumerate()
        .find(|(_, p)| !(0.0..1.0).contains(&p.x) || !(0.0..1.0).contains(&p.y))
    {
        return Some(Failure::named(
            "SamplePositionOutsideThePixel",
            format!(
                "sample position {at} at ({}, {}), outside the [0, 1) Metal documents and asserts \
                 on",
                position.x, position.y
            ),
        ));
    }
    None
}

/// One render pass: its color attachments by index, and the commands encoded into it.
#[derive(Clone, Debug, PartialEq)]
pub struct RenderPass {
    pub colors: Vec<PassColor>,
    /// `depthAttachment`, where the pass has one.
    pub depth: Option<PassDepth>,
    /// `stencilAttachment`, where the pass has one: the same texture as the depth attachment's,
    /// or another one.
    pub stencil: Option<PassStencil>,
    /// `visibilityResultBuffer`: the buffer the pass's visibility results are written into.
    pub visibility: Option<Handle>,
    /// `renderTargetWidth` and `renderTargetHeight`, where the pass states them. A pass with no
    /// attachment at all must, because Metal takes the extent from the attachments otherwise and a
    /// pass without one has none to take it from.
    pub size: Option<[u32; 2]>,
    /// The commands, in order. **A pass of none is a pass with no draw call**, which is one of the
    /// facts Metal's clear route reads, so the empty list is a case rather than an oversight.
    pub steps: Vec<RenderStep>,
    /// `setSamplePositions:count:`: where each sample lies in its pixel, or **empty** for the
    /// device's own positions, which is Metal's default.
    ///
    /// It is the pass descriptor's field and not an encoder step, because that is where Metal puts
    /// it: `MTLRenderPassDescriptor` takes the positions and `MTLRenderCommandEncoder` has no such
    /// call, so a pass cannot move its samples once it is open.
    pub positions: Vec<SamplePosition>,
}

impl RenderPass {
    /// A pass over `colors` that encodes nothing: a clear, stored, with no draw call.
    pub fn clearing(colors: Vec<PassColor>) -> Self {
        Self {
            colors,
            depth: None,
            stencil: None,
            visibility: None,
            size: None,
            steps: Vec::new(),
            positions: Vec::new(),
        }
    }

    /// A pass of `size` with no attachment of any kind: how a case runs a vertex stage that writes
    /// its answer through a buffer rather than into a target.
    pub fn sized(size: [u32; 2]) -> Self {
        let mut pass = Self::clearing(Vec::new());
        pass.size = Some(size);
        pass
    }

    /// The pass with `steps` encoded into it.
    #[must_use]
    pub fn drawing(mut self, steps: Vec<RenderStep>) -> Self {
        self.steps = steps;
        self
    }

    /// The pass with `depth` as its depth attachment.
    #[must_use]
    pub fn with_depth(mut self, depth: PassDepth) -> Self {
        self.depth = Some(depth);
        self
    }

    /// The pass with `stencil` as its stencil attachment.
    #[must_use]
    pub fn with_stencil(mut self, stencil: PassStencil) -> Self {
        self.stencil = Some(stencil);
        self
    }

    /// The pass whose visibility results are written into `buffer`.
    #[must_use]
    pub fn counting_into(mut self, buffer: Handle) -> Self {
        self.visibility = Some(buffer);
        self
    }

    /// The pass that states `positions` as its samples' places in the pixel
    /// (`setSamplePositions:count:`).
    ///
    /// A case may state fewer positions than the pass has samples, which is a question of its own:
    /// Metal puts the samples past them at the pixel's corner. It may state **more** only through
    /// the field, never through this method, because the guard that keeps the suite off Metal's
    /// fatal shapes ([`forbidden_positions`]) is the one place that decides it.
    #[must_use]
    pub fn at(mut self, positions: Vec<SamplePosition>) -> Self {
        self.positions = positions;
        self
    }

    /// The words a case's input fingerprint covers: every attachment's actions and clear color,
    /// the depth, stencil and visibility attachments **where the pass has them**, and the shape of
    /// every step. A pipeline or a buffer a step names is fingerprinted by the case through the
    /// pipeline's own words and the buffer's own bytes.
    ///
    /// A pass that attaches no depth, no stencil and no visibility buffer writes none of those
    /// three, so a color-only pass fingerprints to the bytes it always has.
    pub fn words(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        for color in &self.colors {
            bytes.extend(color.words());
        }
        if let Some(depth) = &self.depth {
            bytes.push(1);
            bytes.extend(depth.words());
        }
        if let Some(stencil) = &self.stencil {
            bytes.push(2);
            bytes.extend(stencil.words());
        }
        if self.visibility.is_some() {
            bytes.push(3);
        }
        if let Some(size) = self.size {
            bytes.push(4);
            bytes.extend(size[0].to_le_bytes());
            bytes.extend(size[1].to_le_bytes());
        }
        for step in &self.steps {
            bytes.extend(step.call().as_bytes());
            bytes.extend(step.parameters());
        }
        // Appended, keyed by the Metal call's own name and written **only where the pass states
        // positions**, so a pass at the device's own positions fingerprints to the bytes it always
        // has and every recording already committed still reads. Each coordinate by its bits: a
        // position one `f32` step away is a different input, and Metal rounds it to a sixteenth of
        // a pixel, so two positions that round alike are still two inputs.
        if !self.positions.is_empty() {
            bytes.extend(SAMPLE_POSITIONS_CALL.as_bytes());
            for position in &self.positions {
                bytes.extend(position.x.to_bits().to_le_bytes());
                bytes.extend(position.y.to_bits().to_le_bytes());
            }
        }
        bytes
    }
}

/// One compute dispatch inside a render program's command buffer, so the kernel that reads a
/// target reads it in the command buffer that drew it and one wait covers both.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Dispatch {
    pub pipeline: Handle,
    /// The buffers bound at Metal buffer index 0 up: each buffer and the offset its binding starts
    /// at, because a sweep writes each answer at its own place in one buffer.
    pub buffers: Vec<(Handle, u64)>,
    /// The textures bound at Metal texture index 0 up.
    pub textures: Vec<Handle>,
    /// `dispatchThreads:threadsPerThreadgroup:`.
    pub grid: [u32; 3],
    pub threads: [u32; 3],
}

/// One piece of the work a command buffer holds.
#[derive(Clone, Debug, PartialEq)]
pub enum RenderWork {
    /// One render pass, opened with its descriptor and ended.
    Pass(RenderPass),
    /// `copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toBuffer:
    /// destinationOffset:destinationBytesPerRow:destinationBytesPerImage:` of level 0's whole
    /// extent, which is how a private target's texels reach the host.
    ReadTexture {
        texture: Handle,
        into: Handle,
        offset: u64,
        row_pitch: u32,
        size: [u32; 2],
    },
    /// One compute dispatch.
    Dispatch(Dispatch),
    /// `copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:`.
    ///
    /// It is here so that a case can ask what a **read of a buffer taken on the GPU** sees, as
    /// against what the CPU sees through the buffer's own contents. For memory whose GPU writes
    /// are dropped those two differ, and they differ again between a copy taken in the same
    /// command buffer as the write and one taken in a later command buffer: one Metal program's
    /// own measured behaviour. A whole [`RenderWork`] list is one command buffer, so a case states
    /// the same-command-buffer read by putting this after the write in one list, and the
    /// later-command-buffer read by calling [`Side::render`](crate::case::Side::render) again.
    CopyBuffer {
        from: Handle,
        from_offset: u64,
        into: Handle,
        into_offset: u64,
        length: u64,
    },
    /// One compute encoder that makes every buffer of `used` resident (`useResource:usage:`) and
    /// then runs `count` of an indirect command buffer's commands from `first`
    /// (`executeCommandsInBuffer:withRange:`).
    ///
    /// The render half of the same question is two steps of a pass ([`RenderStep::UseBuffer`] and
    /// [`RenderStep::ExecuteCommands`]), because a render pass has attachments and a load action
    /// to state and a compute encoder has nothing. Each buffer says whether the commands write
    /// it.
    ComputeCommands {
        commands: Handle,
        first: u64,
        count: u64,
        /// The buffers the commands' functions reach, and whether they are written.
        used: Vec<(Handle, bool)>,
    },
    /// One blit encoder holding `resetCommandsInBuffer:withRange:` over `count` of an indirect
    /// command buffer's commands from `first`, which puts them back as a fresh buffer's **as the
    /// command buffer runs**.
    ///
    /// It is the GPU-timeline twin of `resetWithRange:`
    /// ([`Side::reset_indirect_commands`](crate::case::Side::reset_indirect_commands)), which the
    /// host runs at once. A case tells them apart by where it puts this: a reset encoded before an
    /// execution of the same command buffer resets commands the execution then runs, so the words
    /// a reset command's function would have written stay as the case made them.
    ResetCommands {
        commands: Handle,
        first: u64,
        count: u64,
    },
    /// One blit encoder holding `copyIndirectCommandBuffer:sourceRange:destination:
    /// destinationIndex:`: `count` commands from `first` of one buffer become the commands from
    /// `into_index` of another, as the command buffer runs.
    CopyCommands {
        from: Handle,
        first: u64,
        count: u64,
        into: Handle,
        into_index: u64,
    },
    /// One blit encoder holding `optimizeIndirectCommandBuffer:withRange:` over `count` of a
    /// buffer's commands from `first`.
    ///
    /// Stated plainly: **nothing Metal's optimize changes is observable**, so a case over it asks
    /// whether encoding it leaves every command's own answer alone, which is what a consumer that
    /// optimizes its buffers needs to hold.
    OptimizeCommands {
        commands: Handle,
        first: u64,
        count: u64,
    },
}

impl RenderWork {
    /// The site name the work's answer is emitted at: the call it makes.
    pub fn call(&self) -> &'static str {
        match self {
            RenderWork::Pass(_) => "renderCommandEncoder",
            RenderWork::ReadTexture { .. } => "copyFromTextureToBuffer",
            RenderWork::Dispatch(_) => "dispatchThreads",
            RenderWork::CopyBuffer { .. } => "copyFromBufferToBuffer",
            RenderWork::ComputeCommands { .. } => "executeCommandsInBuffer",
            RenderWork::ResetCommands { .. } => "resetCommandsInBuffer",
            RenderWork::CopyCommands { .. } => "copyIndirectCommandBuffer",
            RenderWork::OptimizeCommands { .. } => "optimizeIndirectCommandBuffer",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::fmt::Debug;
    use std::mem::{discriminant, Discriminant};

    /// A target of its own memory fingerprints to exactly the bytes it did before a backing could
    /// be stated, and a target over a buffer appends its backing's words after them.
    ///
    /// This is the invariant the surface plane beside it learned the hard way: a `None` that wrote
    /// a discriminant moved every render case's recording, and only a whole-set replay caught it.
    /// Stated as the whole target's words rather than the field's, so the test fails if the field
    /// is ever written anywhere but at the end, or when the target has no backing at all.
    #[test]
    fn a_target_over_no_buffer_fingerprints_as_it_always_did() {
        let plain = TargetSpec::plain(PixelFormat::RGBA8Unorm, [4, 4]);
        // The format, the two axes, the samples and the three flags: what a target always has.
        assert_eq!(plain.words().len(), 4 + 2 * 4 + 4 + 3);
        assert_eq!(
            plain.words(),
            TargetSpec::plain(PixelFormat::RGBA8Unorm, [4, 4]).words()
        );
        let over = OverBuffer::new(handle(), 256, 32, 16);
        // A backing is of shared storage, so the storage flag moves with it; everything the
        // backing itself adds comes after every byte a plain target has.
        let mut expected = plain.shared().words();
        expected.push(1);
        expected.extend(over.words());
        assert_eq!(plain.over_buffer(over).words(), expected);
    }

    /// A backing whose rows lie closer together than a row is wide is refused, and one at or above
    /// its row is not.
    ///
    /// Such a call ends Metal's process instead of answering nil, so the refusal has to be decided
    /// from the backing's own numbers and reached without a device. Its failure mode was exercised
    /// while writing it: comparing with `<=` makes the second half fail, naming the pitch.
    #[test]
    fn a_backing_below_its_own_row_is_refused_and_nothing_at_or_above_it() {
        let row = 16;
        for pitch in 0..row {
            let over = OverBuffer::new(handle(), 256, pitch, row);
            assert!(
                over.unsafe_stride().is_some(),
                "a pitch of {pitch} under a row of {row} is not refused, so a case could hand \
                 Metal the call that ends its process"
            );
        }
        for pitch in [row, row + 1, row * 2, 4096] {
            let over = OverBuffer::new(handle(), 256, pitch, row);
            assert_eq!(
                over.unsafe_stride(),
                None,
                "a pitch of {pitch} at or above a row of {row} is refused, which would refuse a \
                 target Metal makes"
            );
        }
    }

    /// A handle to name in a step. Which handle a step holds never reaches the fingerprint, so any
    /// one will do here.
    fn handle() -> Handle {
        Handle(0)
    }

    /// A resource to name in a residency or a barrier step. Which pool it came from is the call's
    /// own parameter, and a buffer is as good as a texture for a guard that reads only the name.
    fn resource() -> StepResource {
        StepResource::Buffer(handle())
    }

    /// The lod clamps a sampler binding carries at a slot.
    fn clamps() -> LodClamp {
        LodClamp::new(0.0, 1.0)
    }

    /// The render stages a residency or a barrier step names.
    fn stages() -> RenderStages {
        RenderStages::VERTEX
    }

    /// Every kind of one vocabulary, from `first` through the successor each kind names.
    ///
    /// This is where the exhaustiveness comes from: `after` is a `match` over every variant of its
    /// enum with **no wildcard arm**, so a variant added to that enum does not compile until it is
    /// given its own arm, and the arm of the kind it is placed after is what hands it to this walk.
    /// A guard then reads the walk rather than an array written out beside the enum, which is what
    /// a new variant can be added without touching.
    ///
    /// The walk is checked to be a list and not a cycle, so an arm naming a kind already walked
    /// fails here, by name, instead of running forever.
    fn every_kind<T: Debug>(first: T, after: impl Fn(&T) -> Option<T>) -> Vec<T> {
        let mut kinds = vec![first];
        let mut walked = vec![discriminant(&kinds[0])];
        while let Some(next) = after(kinds.last().expect("the walk starts at its first kind")) {
            assert!(
                !walked.contains(&discriminant(&next)),
                "the walk of {} returns to {next:?}, so it is a cycle rather than each kind once",
                std::any::type_name::<T>()
            );
            walked.push(discriminant(&next));
            kinds.push(next);
        }
        kinds
    }

    /// The site names more than one of `kinds` emits, each with the kinds that emit it.
    ///
    /// Empty is what every one of these vocabularies wants. The site a kind's answer is emitted at
    /// **is** its name, so two siblings sharing one name is the failure that made a comparison
    /// report six cases as differing while comparing no bytes of either side.
    fn sharing_a_site<'a, T>(
        kinds: &'a [T],
        call: impl Fn(&'a T) -> &'static str,
    ) -> BTreeMap<&'static str, Vec<&'a T>> {
        let mut by_name: BTreeMap<&'static str, Vec<&'a T>> = BTreeMap::new();
        for kind in kinds {
            by_name.entry(call(kind)).or_default().push(kind);
        }
        by_name.retain(|_, with| with.len() > 1);
        by_name
    }

    /// Every piece of work a command buffer can hold, one of each kind, in declaration order.
    fn every_piece() -> Vec<RenderWork> {
        every_kind(RenderWork::Pass(RenderPass::sized([4, 4])), piece_after)
    }

    /// The piece of work that follows `piece` in declaration order, or `None` after the last.
    ///
    /// One arm per variant and no wildcard: see [`every_kind`] for why that is the whole point.
    fn piece_after(piece: &RenderWork) -> Option<RenderWork> {
        Some(match piece {
            RenderWork::Pass(_) => RenderWork::ReadTexture {
                texture: handle(),
                into: handle(),
                offset: 0,
                row_pitch: 16,
                size: [4, 4],
            },
            RenderWork::ReadTexture { .. } => RenderWork::Dispatch(Dispatch {
                pipeline: handle(),
                buffers: Vec::new(),
                textures: Vec::new(),
                grid: [1, 1, 1],
                threads: [1, 1, 1],
            }),
            RenderWork::Dispatch(_) => RenderWork::CopyBuffer {
                from: handle(),
                from_offset: 0,
                into: handle(),
                into_offset: 0,
                length: 16,
            },
            RenderWork::CopyBuffer { .. } => RenderWork::ComputeCommands {
                commands: handle(),
                first: 0,
                count: 1,
                used: Vec::new(),
            },
            RenderWork::ComputeCommands { .. } => RenderWork::ResetCommands {
                commands: handle(),
                first: 0,
                count: 1,
            },
            RenderWork::ResetCommands { .. } => RenderWork::CopyCommands {
                from: handle(),
                first: 0,
                count: 1,
                into: handle(),
                into_index: 0,
            },
            RenderWork::CopyCommands { .. } => RenderWork::OptimizeCommands {
                commands: handle(),
                first: 0,
                count: 1,
            },
            RenderWork::OptimizeCommands { .. } => return None,
        })
    }

    /// Every step a pass can hold, one of each kind, in declaration order.
    fn every_step() -> Vec<RenderStep> {
        every_kind(RenderStep::Pipeline(handle()), step_after)
    }

    /// The step that follows `step` in declaration order, or `None` after the last.
    ///
    /// One arm per variant and no wildcard: see [`every_kind`] for why that is the whole point.
    fn step_after(step: &RenderStep) -> Option<RenderStep> {
        Some(match step {
            RenderStep::Pipeline(_) => RenderStep::DepthStencilState(handle()),
            RenderStep::DepthStencilState(_) => RenderStep::StencilReference(0),
            RenderStep::StencilReference(_) => RenderStep::vertex_buffer(handle(), 0),
            RenderStep::VertexBuffer { .. } => RenderStep::fragment_buffer(handle(), 0),
            RenderStep::FragmentBuffer { .. } => RenderStep::FragmentTexture {
                texture: handle(),
                index: 0,
            },
            RenderStep::FragmentTexture { .. } => RenderStep::FragmentSampler {
                sampler: handle(),
                index: 0,
            },
            RenderStep::FragmentSampler { .. } => RenderStep::VertexBytes {
                bytes: Vec::new(),
                index: 0,
            },
            RenderStep::VertexBytes { .. } => {
                RenderStep::Viewport(ViewportRect::new([0.0; 2], [1.0; 2]))
            }
            RenderStep::Viewport(_) => RenderStep::BlendColor(BlendColorChannels::new([0.0; 4])),
            RenderStep::BlendColor(_) => RenderStep::VisibilityResultMode {
                mode: VisibilityResultMode::Counting,
                offset: 0,
            },
            RenderStep::VisibilityResultMode { .. } => RenderStep::triangle(),
            RenderStep::Draw { .. } => RenderStep::DrawIndexed {
                primitive: PrimitiveType::Triangle,
                count: 3,
                index_type: IndexType::UInt16,
                indices: handle(),
                offset: 0,
                instances: 1,
                base_vertex: 0,
                base_instance: 0,
            },
            RenderStep::DrawIndexed { .. } => RenderStep::DrawIndirect {
                primitive: PrimitiveType::Triangle,
                arguments: handle(),
                offset: 0,
            },
            RenderStep::DrawIndirect { .. } => RenderStep::DrawPlain {
                primitive: PrimitiveType::Triangle,
                first: 0,
                count: 3,
            },
            RenderStep::DrawPlain { .. } => RenderStep::DrawInstanced {
                primitive: PrimitiveType::Triangle,
                first: 0,
                count: 3,
                instances: 1,
            },
            RenderStep::DrawInstanced { .. } => RenderStep::DrawIndexedPlain {
                primitive: PrimitiveType::Triangle,
                count: 3,
                index_type: IndexType::UInt16,
                indices: handle(),
                offset: 0,
            },
            RenderStep::DrawIndexedPlain { .. } => RenderStep::DrawIndexedInstanced {
                primitive: PrimitiveType::Triangle,
                count: 3,
                index_type: IndexType::UInt16,
                indices: handle(),
                offset: 0,
                instances: 1,
            },
            RenderStep::DrawIndexedInstanced { .. } => RenderStep::DrawIndexedIndirect {
                primitive: PrimitiveType::Triangle,
                index_type: IndexType::UInt16,
                indices: handle(),
                index_offset: 0,
                arguments: handle(),
                offset: 0,
            },
            RenderStep::DrawIndexedIndirect { .. } => RenderStep::UseBuffer {
                buffer: handle(),
                write: false,
                stages: crate::case::STAGE_VERTEX,
            },
            RenderStep::UseBuffer { .. } => RenderStep::ExecuteCommands {
                commands: handle(),
                first: 0,
                count: 1,
            },
            RenderStep::ExecuteCommands { .. } => RenderStep::Scissor(ScissorBox::new(0, 0, 1, 1)),
            RenderStep::Scissor(_) => RenderStep::Scissors(vec![ScissorBox::new(0, 0, 1, 1)]),
            RenderStep::Scissors(_) => {
                RenderStep::Viewports(vec![ViewportRect::new([0.0; 2], [1.0; 2])])
            }
            RenderStep::Viewports(_) => RenderStep::Cull(CullMode::None),
            RenderStep::Cull(_) => RenderStep::FrontFacing(Winding::Clockwise),
            RenderStep::FrontFacing(_) => RenderStep::FillMode(TriangleFillMode::Fill),
            RenderStep::FillMode(_) => RenderStep::DepthClip(DepthClipMode::Clip),
            RenderStep::DepthClip(_) => RenderStep::DepthBias(DepthBiasValues::new(0.0, 0.0, 0.0)),
            RenderStep::DepthBias(_) => RenderStep::StencilReferences { front: 0, back: 0 },
            RenderStep::StencilReferences { .. } => RenderStep::VertexBufferOffset {
                offset: 0,
                index: 0,
            },
            RenderStep::VertexBufferOffset { .. } => RenderStep::FragmentBufferOffset {
                offset: 0,
                index: 0,
            },
            RenderStep::FragmentBufferOffset { .. } => RenderStep::FragmentBytes {
                bytes: Vec::new(),
                index: 0,
            },
            RenderStep::FragmentBytes { .. } => RenderStep::VertexBuffers {
                buffers: vec![(handle(), 0)],
                first: 0,
            },
            RenderStep::VertexBuffers { .. } => RenderStep::FragmentBuffers {
                buffers: vec![(handle(), 0)],
                first: 0,
            },
            RenderStep::FragmentBuffers { .. } => RenderStep::VertexTexture {
                texture: handle(),
                index: 0,
            },
            RenderStep::VertexTexture { .. } => RenderStep::VertexTextures {
                textures: vec![handle()],
                first: 0,
            },
            RenderStep::VertexTextures { .. } => RenderStep::FragmentTextures {
                textures: vec![handle()],
                first: 0,
            },
            RenderStep::FragmentTextures { .. } => RenderStep::VertexSampler {
                sampler: handle(),
                index: 0,
            },
            RenderStep::VertexSampler { .. } => RenderStep::VertexSamplers {
                samplers: vec![handle()],
                first: 0,
            },
            RenderStep::VertexSamplers { .. } => RenderStep::FragmentSamplers {
                samplers: vec![handle()],
                first: 0,
            },
            RenderStep::FragmentSamplers { .. } => RenderStep::VertexSamplerClamped {
                sampler: handle(),
                clamp: clamps(),
                index: 0,
            },
            RenderStep::VertexSamplerClamped { .. } => RenderStep::FragmentSamplerClamped {
                sampler: handle(),
                clamp: clamps(),
                index: 0,
            },
            RenderStep::FragmentSamplerClamped { .. } => RenderStep::VertexSamplersClamped {
                samplers: vec![(handle(), clamps())],
                first: 0,
            },
            RenderStep::VertexSamplersClamped { .. } => RenderStep::FragmentSamplersClamped {
                samplers: vec![(handle(), clamps())],
                first: 0,
            },
            RenderStep::FragmentSamplersClamped { .. } => RenderStep::BarrierScope {
                scope: BarrierScope::TEXTURES,
                after: stages(),
                before: stages(),
            },
            RenderStep::BarrierScope { .. } => RenderStep::BarrierResources {
                resources: vec![resource()],
                after: stages(),
                before: stages(),
            },
            RenderStep::BarrierResources { .. } => RenderStep::UseResource {
                resource: resource(),
                usage: ResourceUsage::READ,
            },
            RenderStep::UseResource { .. } => RenderStep::UseResourceInStages {
                resource: resource(),
                usage: ResourceUsage::READ,
                stages: stages(),
            },
            RenderStep::UseResourceInStages { .. } => RenderStep::UseResources {
                resources: vec![resource()],
                usage: ResourceUsage::READ,
            },
            RenderStep::UseResources { .. } => RenderStep::UseResourcesInStages {
                resources: vec![resource()],
                usage: ResourceUsage::READ,
                stages: stages(),
            },
            RenderStep::UseResourcesInStages { .. } => RenderStep::UseHeap(handle()),
            RenderStep::UseHeap(_) => RenderStep::UseHeapInStages {
                heap: handle(),
                stages: stages(),
            },
            RenderStep::UseHeapInStages { .. } => RenderStep::UseHeaps(vec![handle()]),
            RenderStep::UseHeaps(_) => RenderStep::UseHeapsInStages {
                heaps: vec![handle()],
                stages: stages(),
            },
            RenderStep::UseHeapsInStages { .. } => RenderStep::TessellationFactorBuffer {
                buffer: Some(handle()),
                offset: 0,
                instance_stride: 0,
            },
            RenderStep::TessellationFactorBuffer { .. } => RenderStep::TessellationFactorScale {
                bits: 1.0f32.to_bits(),
            },
            RenderStep::TessellationFactorScale { .. } => RenderStep::DrawPatches {
                control_points: 1,
                first: 0,
                count: 1,
                patch_indices: None,
                patch_index_offset: 0,
                instances: 1,
                base_instance: 0,
            },
            RenderStep::DrawPatches { .. } => RenderStep::DrawIndexedPatches {
                control_points: 1,
                first: 0,
                count: 1,
                patch_indices: None,
                patch_index_offset: 0,
                control_point_indices: handle(),
                control_point_index_offset: 0,
                instances: 1,
                base_instance: 0,
            },
            RenderStep::DrawIndexedPatches { .. } => return None,
        })
    }

    /// Every piece of work a command buffer can hold names a Metal call no other piece names.
    ///
    /// The site a piece's answer is emitted at is its call, and two siblings sharing a name is the
    /// failure that made one comparison report six cases as differing while comparing no bytes of
    /// either side. So this holds the property rather than trusting the next author to notice: it
    /// reads [`every_piece`], the walk the compiler keeps complete, and not a list written out
    /// here, which is exactly how a variant added this round went unchecked until a reviewer
    /// happened to see it.
    #[test]
    fn every_piece_of_work_names_a_call_no_other_piece_names() {
        let pieces = every_piece();
        let shared = sharing_a_site(&pieces, RenderWork::call);
        assert!(
            shared.is_empty(),
            "pieces of work that hold one site between them: {shared:?}"
        );
    }

    /// Every step a pass can hold names a Metal call no other step names, so a refusal reads as the
    /// call we refused and the fingerprint keys the step by its own name.
    ///
    /// Read over [`every_step`], the walk the compiler keeps complete, so a step added to the
    /// vocabulary is in this guard whether or not its author remembers it.
    ///
    /// One pair is permitted, named here and nowhere else, and only as this pair: a third step
    /// joining them fails, as does either of them leaving. It is the pair documented at
    /// [`RenderStep::call`] -- `UseBuffer` is an indirect command's own residency call and emits
    /// the name `UseResource` emits. It is sound only while no case holds both.
    ///
    /// Renaming is not a way out of it, because the vocabulary holds **two variants for one Metal
    /// selector**: `UseBuffer` sends `useResource:usage:stages:`, which is what
    /// `UseResourceInStages` sends, so the name this module's convention gives `UseBuffer` is the
    /// one `UseResourceInStages` already holds. A rename moves which pair shares a site; it does
    /// not leave none sharing one. What a rename does **not** do is move a recording: giving
    /// `UseBuffer` a name of its own leaves all 785 committed recordings reading and every case's
    /// input fingerprint where it was, with this guard the only thing in the suite that objects.
    #[test]
    fn every_step_names_a_call_no_other_step_names() {
        let permitted = [
            RenderStep::UseBuffer {
                buffer: handle(),
                write: false,
                stages: crate::case::STAGE_VERTEX,
            },
            RenderStep::UseResource {
                resource: resource(),
                usage: ResourceUsage::READ,
            },
        ];
        let steps = every_step();
        let shared = sharing_a_site(&steps, RenderStep::call);
        let kinds: Vec<Vec<Discriminant<RenderStep>>> = shared
            .values()
            .map(|with| with.iter().map(|step| discriminant(*step)).collect())
            .collect();
        assert_eq!(
            kinds,
            vec![permitted.iter().map(discriminant).collect::<Vec<_>>()],
            "the steps that hold one site between them are {shared:?}; the only pair this permits \
             is {permitted:?}"
        );
    }

    /// A viewport's six numbers all reach its words, and two viewports are equal exactly when
    /// those words are: `-0.0` is not `0.0` here, which is the bit-exact equality the type claims.
    #[test]
    fn a_viewport_compares_and_fingerprints_by_its_bits() {
        let rect = ViewportRect::new([4.5, 0.0], [1.0, 8.0]);
        assert_eq!(rect.words().len(), 6 * 8);
        assert_eq!(rect, rect);
        for moved in [
            ViewportRect::new([4.5f64.next_up(), 0.0], [1.0, 8.0]),
            ViewportRect::new([4.5, -0.0], [1.0, 8.0]),
            ViewportRect::new([4.5, 0.0], [-1.0, 8.0]),
            ViewportRect {
                depth: [0.0, 0.5],
                ..rect
            },
        ] {
            assert_ne!(rect, moved, "{moved:?}");
            assert_ne!(rect.words(), moved.words(), "{moved:?}");
        }
    }

    /// A viewport step's every number reaches its parameters, so a case whose viewport moved reads
    /// as `Unrecorded` rather than comparing against the old viewport's answer.
    #[test]
    fn a_viewport_step_s_numbers_are_its_parameters() {
        let rect = ViewportRect::new([4.25, 0.0], [1.5, 8.0]);
        assert_eq!(RenderStep::Viewport(rect).parameters(), rect.words());
        let moved = ViewportRect {
            depth: [0.25, 1.0],
            ..rect
        };
        assert_ne!(
            RenderStep::Viewport(rect).parameters(),
            RenderStep::Viewport(moved).parameters()
        );
    }

    /// An inline vertex binding's bytes and its index are both its parameters: the bytes are the
    /// case's own values, not a handle, so a draw whose position moved is a different input.
    #[test]
    fn inline_vertex_bytes_and_their_index_are_its_parameters() {
        let step =
            |bytes: Vec<u8>, index: u32| RenderStep::VertexBytes { bytes, index }.parameters();
        assert_eq!(step(vec![1, 2, 3, 4], 0), vec![1, 2, 3, 4, 0, 0, 0, 0]);
        assert_ne!(step(vec![1, 2, 3, 4], 0), step(vec![1, 2, 3, 5], 0));
        assert_ne!(step(vec![1, 2, 3, 4], 0), step(vec![1, 2, 3, 4], 1));
    }

    /// A fragment texture's and a fragment sampler's binding index is its one parameter, and the
    /// handle is not: the case fingerprints the texels and the sampler's descriptor itself. The two
    /// are told apart by their call names, as every step is.
    #[test]
    fn a_fragment_texture_and_sampler_fingerprint_by_their_index() {
        let texture = |index| {
            RenderStep::FragmentTexture {
                texture: handle(),
                index,
            }
            .parameters()
        };
        let sampler = |index| {
            RenderStep::FragmentSampler {
                sampler: handle(),
                index,
            }
            .parameters()
        };
        assert_eq!(texture(0), 0u32.to_le_bytes());
        assert_ne!(texture(0), texture(1));
        assert_ne!(sampler(0), sampler(1));
        assert_eq!(texture(3), sampler(3));
    }

    /// The variant name of the guard's refusal, so a test names the shape rather than the words.
    fn refused_as(failure: &Failure) -> String {
        match &failure.recorded() {
            crate::transcript::Value::Variant { name, payload } if name == "Err" => {
                match payload.as_ref() {
                    crate::transcript::Value::Variant { name, .. } => name.clone(),
                    other => panic!("not an error variant: {other:?}"),
                }
            }
            other => panic!("not an error: {other:?}"),
        }
    }

    /// A position at `(x, y)`, as a case states one.
    fn at(x: f32, y: f32) -> SamplePosition {
        SamplePosition { x, y }
    }

    /// Four positions at `(x, 0.5)`, the shape the coverage family states.
    fn four(x: f32) -> Vec<SamplePosition> {
        vec![at(x, 0.5); 4]
    }

    /// **Invariant 1 for the pass's positions**: a pass at the device's own positions writes none
    /// of them, so every recording of a pass already committed still reads.
    ///
    /// Stated as the whole pass's words rather than one field's: the positions are appended after
    /// the steps, and a pass that states none must end at its last step's parameters.
    #[test]
    fn a_pass_at_the_devices_own_positions_fingerprints_as_it_always_did() {
        let steps = vec![
            RenderStep::fragment_buffer(handle(), 1),
            RenderStep::triangle(),
        ];
        let mut expected = PassColor::cleared(handle(), [0.0; 4]).words();
        for step in &steps {
            expected.extend(step.call().as_bytes());
            expected.extend(step.parameters());
        }
        let pass = RenderPass::clearing(vec![PassColor::cleared(handle(), [0.0; 4])])
            .drawing(steps.clone());
        assert!(pass.positions.is_empty());
        assert_eq!(pass.words(), expected);
        // And stating the empty list is the same pass, not a pass with a sentinel in it.
        assert_eq!(pass.clone().at(Vec::new()).words(), expected);
        // While stating positions appends, and appends only at the end.
        let moved = pass.at(four(0.25));
        assert!(moved.words().starts_with(&expected));
        assert!(moved.words().len() > expected.len());
    }

    /// The positions are keyed by their Metal call's own name and then their bits, so no place in
    /// the words is load-bearing and a coordinate one `f32` step away is a different input.
    #[test]
    fn positions_are_keyed_by_their_call_name_and_their_bits() {
        let words =
            |positions: Vec<SamplePosition>| RenderPass::clearing(Vec::new()).at(positions).words();
        let four_at_a_quarter = words(four(0.25));
        assert!(four_at_a_quarter.starts_with(SAMPLE_POSITIONS_CALL.as_bytes()));
        assert_eq!(
            four_at_a_quarter.len(),
            SAMPLE_POSITIONS_CALL.len() + 4 * 2 * 4
        );
        // Two coordinates Metal rounds to the same sixteenth are still two inputs.
        assert_ne!(words(four(0.53)), words(four(0.5625)));
        assert_ne!(words(four(0.25)), words(four(0.25f32.next_up())));
        // The order the case states them in is the case's own statement.
        assert_ne!(
            words(vec![at(0.25, 0.5), at(0.75, 0.5)]),
            words(vec![at(0.75, 0.5), at(0.25, 0.5)])
        );
        // And a count that moved is a different input, not the same positions.
        assert_ne!(words(four(0.25)), words(vec![at(0.25, 0.5); 2]));
    }

    /// **Invariant 1 for the store-action options**: an attachment that states Metal's default
    /// writes nothing of them, and one that states the option appends exactly its word.
    #[test]
    fn store_action_options_are_written_only_where_an_attachment_states_them() {
        let custom = StoreActionOptions::CUSTOM_SAMPLE_POSITIONS;
        let color = PassColor::cleared(handle(), [0.0; 4]);
        let depth = PassDepth::cleared(handle(), 1.0);
        let stencil = PassStencil::cleared(handle(), 0x11);
        for (plain, optioned) in [
            (color.words(), color.with_options(custom).words()),
            (depth.words(), depth.with_options(custom).words()),
            (stencil.words(), stencil.with_options(custom).words()),
        ] {
            assert_eq!(optioned.len(), plain.len() + 8, "the option's one word");
            assert!(optioned.starts_with(&plain), "appended, nothing shifted");
        }
        assert_eq!(
            color.with_options(StoreActionOptions::NONE).words(),
            color.words()
        );
    }

    /// The Metal leg's guard refuses exactly the two shapes it names, and refuses nothing where the
    /// limit does not apply.
    ///
    /// Each refusal is reached with **no device**: the decision is the positions' own shape and the
    /// pass's sample count, so the shape that ends Metal's process is never built to watch the
    /// guard hold.
    #[test]
    fn the_guard_refuses_more_positions_than_samples_and_a_coordinate_outside_the_pixel() {
        // Nothing to refuse: the device's own positions, and a count at or below the samples with
        // every coordinate inside the pixel.
        assert!(forbidden_positions(&[], 1).is_none());
        assert!(forbidden_positions(&[], 4).is_none());
        assert!(forbidden_positions(&four(0.25), 4).is_none());
        assert!(forbidden_positions(&four(0.9375), 4).is_none());
        // Fewer positions than samples is a recorded answer, not a hazard.
        assert!(forbidden_positions(&[at(0.75, 0.5); 2], 4).is_none());
        assert!(forbidden_positions(&[at(0.5, 0.5)], 4).is_none());
        // One past the pass's samples, which is where plain Metal ends the process.
        let refusal = forbidden_positions(&[at(0.25, 0.5); 5], 4)
            .expect("five positions in a pass of four samples");
        assert_eq!(refused_as(&refusal), "MorePositionsThanSamples");
        assert!(forbidden_positions(&four(0.25), 2).is_some());
        assert!(forbidden_positions(&[at(0.5, 0.5)], 0).is_some());
        // A coordinate outside [0, 1), on either axis and at either end.
        for outside in [
            at(1.0, 0.5),
            at(-0.1, 0.5),
            at(0.5, 1.0),
            at(0.5, 0.0f32.next_down()),
            at(f32::NAN, 0.5),
        ] {
            let refusal = forbidden_positions(&[outside; 4], 4)
                .unwrap_or_else(|| panic!("({}, {}) is outside the pixel", outside.x, outside.y));
            assert_eq!(
                refused_as(&refusal),
                "SamplePositionOutsideThePixel",
                "({}, {})",
                outside.x,
                outside.y
            );
        }
        // The count is read first, because a shape that is both is the fatal one.
        let refusal = forbidden_positions(&[at(1.0, 0.5); 5], 4).expect("both");
        assert_eq!(refused_as(&refusal), "MorePositionsThanSamples");
    }

    /// A pass keys each step by its call name and then its parameters, so adding a variant
    /// renumbers nothing: a pass of the steps the other cases use fingerprints to the same bytes
    /// whether or not this module's four variants exist.
    #[test]
    fn a_step_contributes_its_name_and_its_parameters_and_no_discriminant() {
        let steps = vec![
            RenderStep::fragment_buffer(handle(), 1),
            RenderStep::triangle(),
        ];
        let mut expected = Vec::new();
        for step in &steps {
            expected.extend(step.call().as_bytes());
            expected.extend(step.parameters());
        }
        let pass = RenderPass::clearing(Vec::new()).drawing(steps);
        assert!(pass.words().ends_with(&expected));
    }
}
