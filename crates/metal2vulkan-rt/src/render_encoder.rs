//! Render command encoders: one dynamic rendering a Metal render encoder, begun as the encoder is
//! made and ended as it ends, and the draws recorded into it.
//!
//! The pass orders itself as a blit's copy does. Its first barrier runs it after every earlier
//! command, makes their writes visible to it, and moves each attachment whose home is
//! shader-read-only to the attachment layout; an attachment whose home is general or the
//! attachment layout renders in it. It pays the barrier an earlier encoder owed, and the hazard
//! sets start again. A pass that opens its segment's command buffer with no attachment to move
//! records none: every earlier command is in a command buffer whose end barrier orders it before
//! every later command (`record::end`). Its second, after `vkCmdEndRendering`, moves those
//! attachments home and makes the pass's writes visible to every later command. Where none moves,
//! that barrier is owed instead (`Building::owe_pass`): the segment's next command records it
//! first, reaching every later command, and where none follows, the segment's end barrier is it.
//! So a command buffer of one pass whose attachments render at home records one barrier, its
//! end's.
//!
//! Draws record into the rendering with no barrier between them, as Metal orders the memory
//! accesses of no two draws of one encoder: what a draw writes, a later draw of the encoder sees
//! after a memory barrier. A memory barrier after a draw that wrote memory ends the rendering,
//! records one barrier and begins the rendering again, loading what it stored; so does a draw that
//! needs a copy first (a texture over a buffer in an image of its own, or one over a Managed
//! buffer's device copy, `crate::over`), recording the copy between. A memory barrier after no
//! such draw records nothing: everything before the pass is ordered before it.
//!
//! A draw may read an attachment of its own pass through a texture, as Metal reads a render target
//! (`floor::Capabilities::attachment_reads`): through the attachment feedback loop where the
//! device has it, else through a copy. Its image is a shader-read colour render target with no
//! storage view, whose home is shader-read-only.
//!
//! Through the feedback loop: at the pass's first draw that reads the image, the rendering ends,
//! one barrier moves the whole image to the feedback layout (its rendered level and layer from the
//! attachment layout, the rest from home), and the rendering begins again with the attachment in
//! that layout; the pass's second barrier moves the whole image home. While it is there, every
//! draw of the pass reads the image's textures through their elements in that layout
//! (`crate::arrays`). Every pipeline takes the feedback loop's enable as dynamic state, which the
//! encoder sets at its first draw and where it changes: colour for a draw that reads an
//! attachment, none otherwise.
//!
//! What such a draw reads that an earlier draw of the pass rendered, Metal orders by a memory
//! barrier between the two. A memory barrier after draws that wrote no memory, where the pass
//! moved an image, is owed to the next draw that reads an attachment: the rendering ends, one
//! global barrier is recorded and it begins again, which orders every pixel, the draw's own and
//! any other it reads. A move orders everything before it, so a barrier before the pass's first
//! move records nothing.
//!
//! Through a copy, on a device without the feedback loop (`crate::shadow`): a texture that reaches
//! the attachment's level and layer reads the image's shadow through its second element, and one
//! that reaches neither reads the image at home. Where a draw reads what the shadow does not hold,
//! the rendering ends, one barrier orders every earlier command before the copies and moves what
//! they copy to the transfer layouts (the image's rendered level and layer from the attachment
//! layout, the rest of it from home, the shadow from home), the copies are recorded, one barrier
//! moves both back, and the rendering begins again. At the pass's first draw, what it reads of the
//! level and layer the pass cleared, inside the render area, is a clear of the shadow's to the
//! pass's clear colour instead, between the same barriers. A memory barrier after draws leaves
//! every shadow holding nothing, so the next draw that reads an attachment copies what it reads
//! again, after what the earlier draws rendered; it records nothing itself.
//!
//! A draw sets Metal's state as dynamic state: every state at the encoder's first draw, Metal's
//! defaults where the encoder set none, and after that what changed. A viewport is Vulkan's with
//! its height negated (from `y + height`), so framebuffer rows, the fragment position and
//! derivatives follow Metal's clip space, whose y points up; Metal's front-facing winding is then
//! Vulkan's front face of the same name. Metal clips a draw to the pixels its viewport covers, so
//! the scissor is Metal's rectangle cut to those pixels.

use crate::buffer::Buffer;
use crate::command::{Building, CommandBuffer};
use crate::counters::Metric;
use crate::depth_stencil::{DepthStencilState, Tests};
use crate::device::Core;
use crate::encoder::{check_alignment, BUFFER_SLOTS, SAMPLER_SLOTS, TEXTURE_SLOTS};
use crate::error::{DrawKind, Error, Gap, Limit, Refusal};
use crate::fence::Fence;
use crate::hazard::Access;
use crate::kept::{Handle, Lent, Uncounted};
use crate::over;
use crate::pass::{layout_barrier, Pass, RenderPassDescriptor};
use crate::pipeline::{FieldValue, PUSH_STAGES};
use crate::record::{memory_barrier, Reach};
use crate::render::{Draws, RenderPipelineState, FRAGMENT, VERTEX};
use crate::ring::{Placed, CHUNK};
use crate::sampler::SamplerState;
use crate::shadow::{self, Area, Held};
use crate::texture::Texture;
use ash::vk;
use metal2vulkan::metal::format::integer_channels_word;
use metal2vulkan::metal::state::{input_assembly, DepthStencilDescriptor};
use metal2vulkan::metal::{CullMode, IndexType, PrimitiveType, Winding};
use std::sync::Arc;

/// `MTLViewport`: an origin and size in pixels, and a depth range.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Viewport {
    /// `originX`.
    pub origin_x: f64,
    /// `originY`.
    pub origin_y: f64,
    /// `width`.
    pub width: f64,
    /// `height`: negative flips the draw.
    pub height: f64,
    /// `znear`.
    pub znear: f64,
    /// `zfar`.
    pub zfar: f64,
}

/// `MTLScissorRect`: a rectangle of pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScissorRect {
    /// `x`.
    pub x: u64,
    /// `y`.
    pub y: u64,
    /// `width`.
    pub width: u64,
    /// `height`.
    pub height: u64,
}

/// The index buffer of an indexed draw
/// ([`RenderCommandEncoder::draw_indexed_primitives`]): `indexBuffer`, `indexBufferOffset` and
/// `indexType`.
#[derive(Clone, Copy)]
pub struct IndexBuffer<'a> {
    /// `indexBuffer`.
    pub buffer: &'a Buffer,
    /// `indexBufferOffset`.
    pub offset: u64,
    /// `indexType`.
    pub index_type: IndexType,
}

/// The dynamic state a draw records before it: the viewport and scissor, the cull mode, the
/// front face, the depth and stencil tests, the stencil references, the blend constants, the
/// depth bias the encoder does not set, and the feedback loop's enable where the device reads
/// attachments.
const VIEWPORT: u32 = 1;
const CULL: u32 = 2;
const FRONT: u32 = 4;
const TESTS: u32 = 8;
const REFERENCE: u32 = 16;
const BLEND: u32 = 32;
const BIAS: u32 = 64;
const LOOP: u32 = 128;
const ALL: u32 = 255;

/// An image the pass moved whole to the feedback layout, which its second barrier moves home.
struct Moved {
    /// Its textures' hazard id.
    hazard: u32,
    image: vk::Image,
    /// Every level and layer of it.
    whole: vk::ImageSubresourceRange,
    home: vk::ImageLayout,
}

/// An image a draw moves to the feedback layout: its attachment's level and layer from the layout
/// the pass renders it in, the rest of it from home.
struct Move {
    moved: Moved,
    /// The attachment's level and layer.
    rendered: vk::ImageSubresourceRange,
    from: vk::ImageLayout,
}

/// An image the pass's draws read as an attachment through its shadow, on a device without the
/// feedback loop (`crate::shadow`), and what the shadow holds of it.
struct Shadowed {
    /// Its textures' hazard id.
    hazard: u32,
    image: vk::Image,
    shadow: vk::Image,
    aspects: vk::ImageAspectFlags,
    /// The level and layer its attachment renders, in `layout`; the rest of it is at `home`.
    rendered: (u32, u32),
    layout: vk::ImageLayout,
    home: vk::ImageLayout,
    /// The bytes of a texel.
    texel: u64,
    held: Held,
    /// The colour the pass cleared the rendered level and layer to, where the shadow is of the
    /// attachment's format, so that a clear of the shadow with it writes the texels the pass's
    /// clear wrote (`crate::shadow`); `None` elsewhere.
    cleared: Option<vk::ClearColorValue>,
}

/// What a draw of the encoder's pipeline state reaches of its bound resources, worked out at the
/// first draw after either changes and kept until the next change: the vertex buffers it binds,
/// and whether it reads an attachment of its pass through a texture, reaches a Managed buffer's
/// device copy or a texture over one, or writes memory the host may read. A draw after no change
/// runs none of the scans; one after a change checks its vertex buffers' offsets and its buffers'
/// alignments again.
///
/// A change is the encoder's own call: a pipeline state set, or a buffer, an offset, bytes or a
/// texture bound. Nothing outside the encoder changes what a fact holds: each is of the objects
/// bound (a buffer's device copy, storage and length, a texture's image, storage and pixel
/// format) and of the pass, which do not change. What can change outside it between two draws, a
/// Managed buffer's textures over its device copy and the textures over a buffer in images of
/// their own that live, each draw looks at again where a fact says it may matter.
#[derive(Default)]
struct Facts {
    /// The facts are the bound pipeline state's and resources'.
    held: bool,
    /// The fixed-function vertex buffers the vertex descriptor reads: binding, buffer, offset.
    vertex: Vec<(u32, vk::Buffer, u64)>,
    /// A texture the draw reads is of an image the pass renders.
    attachments: bool,
    /// A buffer the draw reaches has a device copy, or a texture it writes is over one.
    managed: bool,
    /// A write the host may read once the command buffer completes: to a Shared buffer or a
    /// texture over one, or through an address outside the draw's resources.
    host: bool,
    /// The nil buffer's address where a buffer argument the draw reaches is bound to nothing
    /// (`Ring::nil`), else 0.
    nil: u64,
}

impl Facts {
    /// The facts of a draw of `draws` with `tables` bound in `pass`, but the alignments, which the
    /// draw checks after its attachment reads.
    ///
    /// Err: [`Refusal::VertexBufferOffset`]; [`Refusal::NoBufferAtIndex`] for a vertex buffer the
    /// vertex descriptor reads that is not bound.
    fn find(&mut self, draws: &Draws, tables: &[Tables; 2], pass: &Pass) -> Result<(), Error> {
        self.vertex.clear();
        for &(binding, index) in &draws.bindings {
            let (buffer, offset) = match tables[VERTEX].buffers.get(index as usize) {
                Some(Bound::Buffer { buffer, offset }) => {
                    if *offset >= buffer.length() {
                        return Err(Error::Refused(Refusal::VertexBufferOffset {
                            index,
                            offset: *offset,
                            length: buffer.length(),
                        }));
                    }
                    let (reached, at) = buffer.device_range();
                    (reached, at + offset)
                }
                Some(Bound::Bytes(placed)) => (placed.buffer, placed.offset),
                Some(Bound::Empty) | None => {
                    return Err(Error::Refused(Refusal::NoBufferAtIndex { index }))
                }
            };
            self.vertex.push((binding, buffer, offset));
        }
        self.attachments = (draws.textures.iter()).any(|&(table, slot, _)| {
            tables[table].texture(slot).is_some_and(|texture| {
                let hazard = texture.hazard_id();
                pass.targets().any(|t| t.texture.hazard_id() == hazard)
            })
        });
        let copied = |buffer: &Buffer| buffer.held().1.is_some();
        self.managed = (draws.buffers.iter())
            .any(|&(table, slot, _)| tables[table].buffer(slot).is_some_and(copied))
            || (draws.bindings.iter())
                .any(|&(_, index)| tables[VERTEX].buffer(index).is_some_and(copied))
            || (draws.textures.iter()).any(|&(table, slot, access)| {
                access.writes()
                    && (tables[table].texture(slot)).is_some_and(|t| t.managed_over().is_some())
            });
        self.host = draws.writes_unbound
            || (draws.buffers.iter()).any(|&(table, slot, access)| {
                access.writes() && tables[table].buffer(slot).is_some_and(Buffer::is_shared)
            })
            || (draws.textures.iter()).any(|&(table, slot, access)| {
                access.writes() && tables[table].texture(slot).is_some_and(Texture::host_reads)
            });
        Ok(())
    }
}

/// What a draw reads of its pass's attachments through textures
/// ([`RenderCommandEncoder::attachment_reads`]).
struct AttachmentReads {
    /// Through the feedback loop: a texture reaches an attachment's level and layer.
    reads: bool,
    /// The images the draw moves to the feedback layout first.
    moves: Vec<Move>,
    /// Without the feedback loop: each texture that reads a shadow, and what the draw reads
    /// through it.
    shadow_reads: Vec<(Texture, Vec<Area>)>,
    /// Their slots: function and slot.
    shadow_slots: Vec<(usize, u32)>,
}

/// What a buffer slot holds.
#[derive(Clone, Default)]
enum Bound {
    #[default]
    Empty,
    Buffer {
        buffer: Lent<Buffer>,
        offset: u64,
    },
    /// A `set_*_bytes` copy in the parameter ring.
    Bytes(Placed),
}

/// One function's argument tables: their objects borrowed from what the command buffer keeps
/// (`crate::kept`).
struct Tables {
    buffers: Vec<Bound>,
    textures: Vec<Option<Lent<Texture>>>,
    samplers: Vec<Option<Lent<SamplerState>>>,
}

impl Tables {
    fn new() -> Tables {
        Tables {
            buffers: vec![Bound::Empty; BUFFER_SLOTS as usize],
            textures: vec![None; TEXTURE_SLOTS as usize],
            samplers: vec![None; SAMPLER_SLOTS as usize],
        }
    }

    /// The buffer bound at `slot`, where a buffer is.
    fn buffer(&self, slot: u32) -> Option<&Buffer> {
        match self.buffers.get(slot as usize) {
            Some(Bound::Buffer { buffer, .. }) => Some(buffer),
            _ => None,
        }
    }

    /// The texture bound at `slot`, where one is.
    fn texture(&self, slot: u32) -> Option<&Texture> {
        self.textures.get(slot as usize)?.as_deref()
    }

    /// The address of buffer slot `slot`: the buffer's plus its offset, or the bytes' copy; `nil`
    /// where none is bound.
    fn address(&self, slot: u32, nil: u64) -> u64 {
        match self.buffers.get(slot as usize) {
            Some(Bound::Buffer { buffer, offset }) => buffer.gpu_address().wrapping_add(*offset),
            Some(Bound::Bytes(placed)) => placed.address,
            Some(Bound::Empty) | None => nil,
        }
    }
}

/// What a draw draws.
#[derive(Clone, Copy)]
enum Call<'a> {
    Direct {
        vertex_start: u32,
        vertex_count: u32,
    },
    Indexed {
        index_count: u32,
        indices: IndexBuffer<'a>,
        base_vertex: i32,
    },
}

/// `MTLRenderCommandEncoder`: one dynamic rendering into its pass's attachments, begun as the
/// encoder is made, with each attachment's load action, and ended by
/// [`end_encoding`](Self::end_encoding), with each one's store action; its draws record into it.
///
/// It orders itself against every command before and after it with a barrier of its own on each
/// side, as a blit's copy does: it needs no hazard tracking of its own and leaves none for the
/// encoders after it. Its draws take no barrier between them, as Metal orders none: a draw sees
/// what an earlier draw of the encoder wrote after a
/// [`memory_barrier`](Self::memory_barrier).
///
/// Built: a render pipeline state, the vertex and fragment functions' buffers, bytes, textures
/// and sampler states, one viewport and scissor rectangle, the cull mode, the front-facing
/// winding, a depth-stencil state that tests the stencil, the stencil references, the blend color,
/// memory barriers, fences, and direct draws of lines and triangles, indexed or not. Not built,
/// and so not methods: indirect draws, several viewports or scissor rectangles, the triangle fill
/// mode, the depth clip mode, depth bias, visibility results, `useResource:`, the store action
/// setters, tile functions and the `setVertexBuffer:offset:attributeStride:atIndex:` family. A
/// draw of what the runtime does not build is [`Gap::Draw`], naming its [`DrawKind`].
///
/// Owned: one thread encodes at a time. Dropping it ends it, as
/// [`end_encoding`](Self::end_encoding) does.
pub struct RenderCommandEncoder {
    command_buffer: CommandBuffer,
    /// The device, borrowed from the recording's own handle in `building` with no count of its
    /// own: every use is while the encoder is open, and so while `building` holds that handle.
    core: Uncounted<Core>,
    /// What the command buffer records, held while the encoder is open.
    building: Option<Building>,
    /// The checked pass, for its second barrier and to begin its rendering again.
    pass: Pass,
    /// The command buffer the pass records into.
    command: vk::CommandBuffer,
    /// The vertex function's argument tables, then the fragment function's.
    tables: [Tables; 2],
    /// The state set, borrowed from what the command buffer keeps (`crate::kept`).
    pipeline: Option<Lent<RenderPipelineState>>,
    /// Whether the pipeline state renders into the pass's attachments' formats.
    renders_into: bool,
    /// The depth-stencil state's tests, Metal's default state's where none is set.
    tests: Tests,
    viewport: Viewport,
    /// Metal's scissor rectangle cut to the render area, per axis `[first, end)`.
    scissor: [[u32; 2]; 2],
    cull_mode: CullMode,
    winding: Winding,
    /// The front and back stencil references, saturated to 8 bits.
    stencil_reference: [u32; 2],
    blend_color: [f32; 4],
    /// The dynamic state to record before the next draw.
    dirty: u32,
    /// The topology and primitive restart the last draw set.
    topology: Option<(vk::PrimitiveTopology, bool)>,
    /// A draw since the rendering began wrote memory: a memory barrier ends the rendering.
    wrote: bool,
    /// A draw since the encoder last ordered its draws rendered.
    drew: bool,
    /// A draw of the pass recorded: what the pass cleared may hold more than its clear colour.
    pass_drew: bool,
    /// A memory barrier after draws that wrote no memory, which the next draw that reads an
    /// attachment pays.
    owed: bool,
    /// The images the pass moved to the feedback layout.
    moved: Vec<Moved>,
    /// The feedback loop's aspects the last draw set: colour for a draw that reads an attachment.
    loop_aspects: vk::ImageAspectFlags,
    /// The images the pass's draws read through their shadows, without the feedback loop.
    shadowed: Vec<Shadowed>,
    /// The texture slots of the draw being recorded that read a shadow: function and slot.
    shadow_slots: Vec<(usize, u32)>,
    /// What a draw of the pipeline state reaches of the bound resources.
    facts: Facts,
    /// The push bytes of the last draw.
    push: Vec<u8>,
    /// The push bytes are those of the bound pipeline state and resources, and of the images the
    /// pass moved and the shadows the last draw read: the next draw pushes them as they are.
    push_filled: bool,
}

impl CommandBuffer {
    /// `renderCommandEncoderWithDescriptor:`: begins `descriptor`'s pass, one
    /// `vkCmdBeginRendering` between the pass's first barrier and its commands. The command
    /// buffer keeps each attachment's texture until it retires.
    ///
    /// Err: [`Refusal::Committed`];
    /// [`Refusal::EncoderOpen`] while another encoder of the command
    /// buffer is open; [`Refusal::OtherDevice`] for another device's
    /// texture; [`Refusal::AttachmentUsage`];
    /// [`Refusal::AttachmentPlace`];
    /// [`Refusal::AttachmentFormat`];
    /// [`Gap::RenderPass`];
    /// [`Limit::ColorAttachments`]; [`Error::Vulkan`] where an
    /// attachment's view cannot be made or the command buffer cannot record. A refused
    /// descriptor leaves the command buffer as it was, open for another encoder.
    pub fn render_command_encoder(
        &self,
        descriptor: &RenderPassDescriptor,
    ) -> Result<RenderCommandEncoder, Error> {
        let mut building = self.open_encoder()?;
        // SAFETY: the encoder holds `building`, whose device handle keeps the core, until it ends;
        // nothing uses the borrow after that, and a refused descriptor drops it with no use.
        let core = unsafe { Uncounted::of(building.device.core()) };
        match begin(&mut building, &core, descriptor) {
            Ok((pass, command)) => {
                let [width, height] = pass.area;
                Ok(RenderCommandEncoder {
                    command_buffer: self.clone(),
                    core,
                    building: Some(building),
                    command,
                    tables: [Tables::new(), Tables::new()],
                    pipeline: None,
                    renders_into: false,
                    tests: Tests::of(&DepthStencilDescriptor::default()),
                    viewport: Viewport {
                        origin_x: 0.0,
                        origin_y: 0.0,
                        width: f64::from(width),
                        height: f64::from(height),
                        znear: 0.0,
                        zfar: 1.0,
                    },
                    scissor: [[0, width], [0, height]],
                    cull_mode: CullMode::None,
                    winding: Winding::Clockwise,
                    stencil_reference: [0; 2],
                    blend_color: [0.0; 4],
                    dirty: ALL,
                    topology: None,
                    wrote: false,
                    drew: false,
                    pass_drew: false,
                    owed: false,
                    moved: Vec::new(),
                    loop_aspects: vk::ImageAspectFlags::NONE,
                    shadowed: Vec::new(),
                    shadow_slots: Vec::new(),
                    facts: Facts::default(),
                    push: Vec::new(),
                    push_filled: false,
                    pass,
                })
            }
            Err(error) => {
                self.close_encoder(building);
                Err(error)
            }
        }
    }
}

/// Checks `descriptor` and records its pass's first barrier, where it has one, and its
/// `vkCmdBeginRendering`.
fn begin(
    building: &mut Building,
    core: &Arc<Core>,
    descriptor: &RenderPassDescriptor,
) -> Result<(Pass, vk::CommandBuffer), Error> {
    let pass = Pass::check(core, descriptor)?;
    let overs: Vec<(u32, Access)> = (pass.targets())
        .map(|t| (t.texture.hazard_id(), Access::ReadWrite))
        .collect();
    for t in pass.targets() {
        building.keep_texture(&t.texture);
        building.wrote_texture_managed(&t.texture);
    }
    building.overs_before(&overs, false)?;
    let opens = building.opens_command_buffer();
    let command = building.ordered_command()?;
    // Marked after `ordered_command`, as a blit's copy marks it: the host reads a texture over a
    // Shared buffer's bytes in the buffer.
    if pass.targets().any(|t| t.texture.host_reads()) {
        building.wrote_host();
    }
    building.overs_after(&overs, false);
    building.count(Metric::Passes);
    let earlier = [memory_barrier(Reach::Commands)];
    let away: Vec<vk::ImageMemoryBarrier2> = (pass.targets())
        .filter(|t| t.moves())
        .map(|t| t.barrier(t.attachment.home, t.layout))
        .collect();
    let device = &core.device;
    // A pass that opens its command buffer and moves no attachment needs no first barrier: each
    // earlier command buffer on the queue ended with one that orders it before every later
    // command.
    if !(opens && away.is_empty()) {
        // SAFETY: the command buffer records outside any render pass and keeps each attachment's
        // texture until it retires; each attachment is in its home layout between commands,
        // which the barrier moves to the layout the pass names.
        unsafe {
            device.cmd_pipeline_barrier2(
                command,
                &vk::DependencyInfo::default()
                    .memory_barriers(&earlier)
                    .image_memory_barriers(&away),
            );
        }
        building.count(Metric::Barriers);
    }
    // SAFETY: the command buffer records outside any render pass and keeps each attachment's
    // texture until it retires; each attachment is in the layout the pass names.
    unsafe { pass.begin(device, command, false) };
    Ok((pass, command))
}

impl RenderCommandEncoder {
    fn building(&mut self) -> &mut Building {
        self.building
            .as_mut()
            .expect("an encoder holds its command buffer's recording until it ends")
    }

    /// The pipeline state or a binding changed: the next draw finds its facts and fills its push
    /// bytes again.
    fn changed(&mut self) {
        self.facts.held = false;
        self.push_filled = false;
    }

    /// `setRenderPipelineState:`. The command buffer keeps the state until it retires.
    ///
    /// Err: [`Refusal::OtherDevice`] for another device's state.
    pub fn set_render_pipeline_state(&mut self, state: &RenderPipelineState) -> Result<(), Error> {
        if !state.is_of(self.building().device.core()) {
            return Err(Error::Refused(Refusal::OtherDevice));
        }
        if self.pipeline.as_ref().is_some_and(|set| set.is(state)) {
            return Ok(());
        }
        self.renders_into = state.inner.compiled.renders_into(&self.pass);
        // SAFETY: the encoder holds the borrow, and it ends before its command buffer commits.
        self.pipeline = Some(unsafe { self.building().lend_render_pipeline(state) });
        self.changed();
        Ok(())
    }

    /// `setVertexBuffer:offset:atIndex:` (`setVertexBuffers:offsets:withRange:` is a loop of it).
    /// `None` unbinds the slot. The slot serves the vertex function's buffer argument of its index
    /// and the vertex descriptor's layout of its index alike. The command buffer keeps the buffer,
    /// and so its block, until it retires.
    ///
    /// Err: [`Refusal::BufferIndex`] for an index of 31 or more; [`Refusal::OtherDevice`] for
    /// another device's buffer.
    pub fn set_vertex_buffer(
        &mut self,
        buffer: Option<&Buffer>,
        offset: u64,
        index: u32,
    ) -> Result<(), Error> {
        self.set_buffer(VERTEX, buffer, offset, index)
    }

    /// `setFragmentBuffer:offset:atIndex:` (`setFragmentBuffers:offsets:withRange:` is a loop of
    /// it), as [`set_vertex_buffer`](Self::set_vertex_buffer) for the fragment function.
    ///
    /// Err: as [`set_vertex_buffer`](Self::set_vertex_buffer).
    pub fn set_fragment_buffer(
        &mut self,
        buffer: Option<&Buffer>,
        offset: u64,
        index: u32,
    ) -> Result<(), Error> {
        self.set_buffer(FRAGMENT, buffer, offset, index)
    }

    fn set_buffer(
        &mut self,
        table: usize,
        buffer: Option<&Buffer>,
        offset: u64,
        index: u32,
    ) -> Result<(), Error> {
        let slot = slot(index)?;
        let bound = match buffer {
            Some(buffer) => {
                if let Bound::Buffer {
                    buffer: held,
                    offset: at,
                } = &self.tables[table].buffers[slot]
                {
                    if held.address() == buffer.address() && *at == offset {
                        return Ok(());
                    }
                }
                let building = self.building();
                if !buffer.is_of(building.device.core()) {
                    return Err(Error::Refused(Refusal::OtherDevice));
                }
                Bound::Buffer {
                    // SAFETY: the encoder's table holds the borrow, and it ends before its command
                    // buffer commits.
                    buffer: unsafe { building.lend_buffer(buffer) },
                    offset,
                }
            }
            None => Bound::Empty,
        };
        self.tables[table].buffers[slot] = bound;
        self.changed();
        Ok(())
    }

    /// `setVertexBufferOffset:atIndex:`.
    ///
    /// Err: [`Refusal::BufferIndex`] for an index of 31 or more; [`Refusal::NoBufferAtIndex`]
    /// where no buffer is bound there.
    pub fn set_vertex_buffer_offset(&mut self, offset: u64, index: u32) -> Result<(), Error> {
        self.set_buffer_offset(VERTEX, offset, index)
    }

    /// `setFragmentBufferOffset:atIndex:`.
    ///
    /// Err: as [`set_vertex_buffer_offset`](Self::set_vertex_buffer_offset).
    pub fn set_fragment_buffer_offset(&mut self, offset: u64, index: u32) -> Result<(), Error> {
        self.set_buffer_offset(FRAGMENT, offset, index)
    }

    fn set_buffer_offset(&mut self, table: usize, offset: u64, index: u32) -> Result<(), Error> {
        match &mut self.tables[table].buffers[slot(index)?] {
            Bound::Buffer { offset: bound, .. } => {
                *bound = offset;
                self.changed();
                Ok(())
            }
            _ => Err(Error::Refused(Refusal::NoBufferAtIndex { index })),
        }
    }

    /// `setVertexBytes:length:atIndex:`: the bytes are copied into the parameter ring now, as
    /// Metal copies them, and the slot is the copy, for the vertex function's argument and the
    /// vertex descriptor's layout alike.
    ///
    /// Err: [`Refusal::BufferIndex`] for an index of 31 or more; [`Limit::BytesLength`] past one
    /// ring chunk (64 KiB; Metal documents the call for data under 4 KB);
    /// [`Limit::MemoryAllocations`] where the ring's next block would take the device past its
    /// allocation count; [`Error::Vulkan`] where the ring cannot grow.
    pub fn set_vertex_bytes(&mut self, bytes: &[u8], index: u32) -> Result<(), Error> {
        self.set_bytes(VERTEX, bytes, index)
    }

    /// `setFragmentBytes:length:atIndex:`, as [`set_vertex_bytes`](Self::set_vertex_bytes) for
    /// the fragment function.
    ///
    /// Err: as [`set_vertex_bytes`](Self::set_vertex_bytes).
    pub fn set_fragment_bytes(&mut self, bytes: &[u8], index: u32) -> Result<(), Error> {
        self.set_bytes(FRAGMENT, bytes, index)
    }

    fn set_bytes(&mut self, table: usize, bytes: &[u8], index: u32) -> Result<(), Error> {
        let slot = slot(index)?;
        if bytes.len() as u64 > CHUNK {
            return Err(Error::Limit(Limit::BytesLength {
                len: bytes.len() as u64,
                max: CHUNK,
            }));
        }
        let placed = self.building().write_ring_placed(bytes)?;
        self.tables[table].buffers[slot] = Bound::Bytes(placed);
        self.changed();
        Ok(())
    }

    /// `setVertexTexture:atIndex:` (`setVertexTextures:withRange:` is a loop of it). `None`
    /// unbinds the slot. A draw pushes the texture's `gpuResourceID` for each texture slot its
    /// vertex function reads, and the channel widths an integer write saturates to where the
    /// function takes them. The command buffer keeps the texture until it retires.
    ///
    /// Err: [`Refusal::TextureArgumentIndex`] for an index of 128 or more;
    /// [`Refusal::OtherDevice`] for another device's texture.
    pub fn set_vertex_texture(
        &mut self,
        texture: Option<&Texture>,
        index: u32,
    ) -> Result<(), Error> {
        self.set_texture(VERTEX, texture, index)
    }

    /// `setFragmentTexture:atIndex:` (`setFragmentTextures:withRange:` is a loop of it), as
    /// [`set_vertex_texture`](Self::set_vertex_texture) for the fragment function.
    ///
    /// Err: as [`set_vertex_texture`](Self::set_vertex_texture).
    pub fn set_fragment_texture(
        &mut self,
        texture: Option<&Texture>,
        index: u32,
    ) -> Result<(), Error> {
        self.set_texture(FRAGMENT, texture, index)
    }

    fn set_texture(
        &mut self,
        table: usize,
        texture: Option<&Texture>,
        index: u32,
    ) -> Result<(), Error> {
        if index >= TEXTURE_SLOTS {
            return Err(Error::Refused(Refusal::TextureArgumentIndex { index }));
        }
        let held = &self.tables[table].textures[index as usize];
        if held.as_deref().map(Handle::address) == texture.map(Handle::address) {
            return Ok(());
        }
        let lent = match texture {
            Some(texture) => {
                let building = self.building();
                if !texture.is_of(building.device.core()) {
                    return Err(Error::Refused(Refusal::OtherDevice));
                }
                // SAFETY: the encoder's table holds the borrow, and it ends before its command
                // buffer commits.
                Some(unsafe { building.lend_texture(texture) })
            }
            None => None,
        };
        self.tables[table].textures[index as usize] = lent;
        self.changed();
        Ok(())
    }

    /// `setVertexSamplerState:atIndex:` (`setVertexSamplerStates:withRange:` is a loop of it).
    /// `None` unbinds the slot. A draw pushes the state's `gpuResourceID` for each sampler slot its
    /// vertex function samples through. The command buffer keeps the state until it retires.
    ///
    /// Err: [`Refusal::SamplerArgumentIndex`] for an index of 16 or more;
    /// [`Refusal::OtherDevice`] for another device's state.
    pub fn set_vertex_sampler_state(
        &mut self,
        sampler: Option<&SamplerState>,
        index: u32,
    ) -> Result<(), Error> {
        self.set_sampler_state(VERTEX, sampler, index)
    }

    /// `setFragmentSamplerState:atIndex:` (`setFragmentSamplerStates:withRange:` is a loop of
    /// it), as [`set_vertex_sampler_state`](Self::set_vertex_sampler_state) for the fragment
    /// function.
    ///
    /// Err: as [`set_vertex_sampler_state`](Self::set_vertex_sampler_state).
    pub fn set_fragment_sampler_state(
        &mut self,
        sampler: Option<&SamplerState>,
        index: u32,
    ) -> Result<(), Error> {
        self.set_sampler_state(FRAGMENT, sampler, index)
    }

    fn set_sampler_state(
        &mut self,
        table: usize,
        sampler: Option<&SamplerState>,
        index: u32,
    ) -> Result<(), Error> {
        if index >= SAMPLER_SLOTS {
            return Err(Error::Refused(Refusal::SamplerArgumentIndex { index }));
        }
        let held = &self.tables[table].samplers[index as usize];
        if held.as_deref().map(Handle::address) == sampler.map(Handle::address) {
            return Ok(());
        }
        let lent = match sampler {
            Some(sampler) => {
                let building = self.building();
                if !sampler.is_of(building.device.core()) {
                    return Err(Error::Refused(Refusal::OtherDevice));
                }
                // SAFETY: the encoder's table holds the borrow, and it ends before its command
                // buffer commits.
                Some(unsafe { building.lend_sampler(sampler) })
            }
            None => None,
        };
        self.tables[table].samplers[index as usize] = lent;
        self.push_filled = false;
        Ok(())
    }

    /// `setViewport:`. Metal's default covers the render area with the depth range `[0, 1]`. A
    /// viewport of no width or height draws nothing, as Metal's; so does one whose origin's x is
    /// NaN, as plain Metal draws nothing through it. A negative height flips the draw, as Metal's.
    ///
    /// Err: [`Refusal::ViewportNotFinite`] for any other field NaN or infinite; [`Gap::Draw`] of
    /// [`DrawKind::Viewport`] for a negative width, a depth range past `[0, 1]`, or bounds past
    /// the device's `maxViewportDimensions` and `viewportBoundsRange`.
    pub fn set_viewport(&mut self, viewport: Viewport) -> Result<(), Error> {
        let v = match viewport.origin_x.is_nan() {
            true => Viewport {
                origin_x: 0.0,
                width: 0.0,
                ..viewport
            },
            false => viewport,
        };
        let fields = [v.origin_x, v.origin_y, v.width, v.height, v.znear, v.zfar];
        if fields.iter().any(|f| !f.is_finite()) {
            return Err(Error::Refused(Refusal::ViewportNotFinite));
        }
        let limits = &self.core.limits;
        let [low, high] = limits.viewport_bounds.map(f64::from);
        let vk = vk_viewport(&v);
        let [x, y, width, height] = [vk.x, vk.y, vk.width, vk.height].map(f64::from);
        let unit = 0.0..=1.0;
        let fits = v.width >= 0.0
            && unit.contains(&v.znear)
            && unit.contains(&v.zfar)
            && width <= f64::from(limits.max_viewport[0])
            && height.abs() <= f64::from(limits.max_viewport[1])
            && x >= low
            && x + width <= high
            && [y, y + height].iter().all(|y| (low..=high).contains(y));
        if !fits {
            return Err(Error::Unsupported(Gap::Draw {
                kind: DrawKind::Viewport,
            }));
        }
        self.viewport = v;
        self.dirty |= VIEWPORT;
        Ok(())
    }

    /// `setScissorRect:`. Metal's default is the render area. A rectangle past the render area,
    /// which Metal's validation stops, is cut to it, as plain Metal cuts it.
    pub fn set_scissor_rect(&mut self, rect: ScissorRect) {
        let [width, height] = self.pass.area;
        let cut = |origin: u64, size: u64, limit: u32| {
            let limit = u64::from(limit);
            [origin.min(limit), origin.saturating_add(size).min(limit)].map(|at| at as u32)
        };
        self.scissor = [
            cut(rect.x, rect.width, width),
            cut(rect.y, rect.height, height),
        ];
        self.dirty |= VIEWPORT;
    }

    /// `setCullMode:`. Metal's default culls nothing.
    pub fn set_cull_mode(&mut self, mode: CullMode) {
        self.cull_mode = mode;
        self.dirty |= CULL;
    }

    /// `setFrontFacingWinding:`. Metal's default is clockwise.
    pub fn set_front_facing_winding(&mut self, winding: Winding) {
        self.winding = winding;
        self.dirty |= FRONT;
    }

    /// `setDepthStencilState:`; `None` sets Metal's default state again, which tests nothing.
    ///
    /// Err: [`Refusal::OtherDevice`] for another device's state.
    pub fn set_depth_stencil_state(
        &mut self,
        state: Option<&DepthStencilState>,
    ) -> Result<(), Error> {
        self.tests = match state {
            Some(state) => {
                if !state.is_of(self.building().device.core()) {
                    return Err(Error::Refused(Refusal::OtherDevice));
                }
                state.tests()
            }
            None => Tests::of(&DepthStencilDescriptor::default()),
        };
        self.dirty |= TESTS;
        Ok(())
    }

    /// `setStencilReferenceValue:`: both faces' reference, saturated to 255 as Metal stores it.
    pub fn set_stencil_reference_value(&mut self, value: u32) {
        self.set_stencil_front_back_reference_value(value, value);
    }

    /// `setStencilFrontReferenceValue:backReferenceValue:`, each saturated to 255.
    pub fn set_stencil_front_back_reference_value(&mut self, front: u32, back: u32) {
        self.stencil_reference = [front.min(255), back.min(255)];
        self.dirty |= REFERENCE;
    }

    /// `setBlendColorRed:green:blue:alpha:`. Metal's default is zero.
    pub fn set_blend_color(&mut self, red: f32, green: f32, blue: f32, alpha: f32) {
        self.blend_color = [red, green, blue, alpha];
        self.dirty |= BLEND;
    }

    /// `memoryBarrierWithScope:afterStages:beforeStages:` and
    /// `memoryBarrierWithResources:count:afterStages:beforeStages:`: after a draw of the encoder
    /// that wrote memory, the rendering ends, one global memory barrier is recorded, and the
    /// rendering begins again, loading what it stored. After draws that only rendered, where the
    /// pass moved an image to the feedback layout, the barrier is owed to the next draw that reads
    /// an attachment through a texture, which records it first (the module's documentation);
    /// otherwise nothing is recorded, as everything before the pass is ordered before it and no
    /// draw reads an attachment before a move that orders everything before it. On a device
    /// without the feedback loop, after draws, every shadow holds nothing from then, so the next
    /// draw that reads an attachment copies what it reads again (`crate::shadow`).
    ///
    /// Err: [`Error::Vulkan`] where the command buffer cannot record.
    pub fn memory_barrier(&mut self) -> Result<(), Error> {
        if self.drew {
            for shadowed in &mut self.shadowed {
                shadowed.held.clear();
            }
        }
        if self.wrote {
            return self.ordered();
        }
        self.owed |= self.drew && !self.moved.is_empty();
        Ok(())
    }

    /// Ends the rendering, records one global memory barrier and begins the rendering again: the
    /// encoder's draws so far are ordered before its later ones.
    fn ordered(&mut self) -> Result<(), Error> {
        (self.drew, self.owed) = (false, false);
        self.interrupt(|building, _| {
            let command = building.ordered_command()?;
            building.barrier(command, Reach::Commands);
            Ok(())
        })
    }

    /// Moves each image of `moves` whole to the feedback layout: the rendering ends, one barrier
    /// orders every earlier command before every later one and moves them, and the rendering
    /// begins again with their attachments in that layout.
    fn feedback_moves(&mut self, moves: Vec<Move>) -> Result<(), Error> {
        let looped = vk::ImageLayout::ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT;
        let mut images = Vec::new();
        for m in &moves {
            let Moved {
                image, whole, home, ..
            } = m.moved;
            images.push(layout_barrier(image, m.rendered, m.from, looped));
            for range in around(whole, m.rendered) {
                images.push(layout_barrier(image, range, home, looped));
            }
        }
        for t in self.pass.colors.iter_mut().flatten() {
            if (moves.iter()).any(|m| m.moved.hazard == t.texture.hazard_id()) {
                t.layout = looped;
            }
        }
        self.moved.extend(moves.into_iter().map(|m| m.moved));
        self.push_filled = false;
        (self.drew, self.owed) = (false, false);
        self.interrupt(|building, core| {
            let command = building.ordered_command()?;
            let memory = [memory_barrier(Reach::Commands)];
            // SAFETY: the command buffer records outside any rendering; each range is in the
            // layout it moves from: the attachment's in the layout the pass rendered it in, the
            // rest of the image at home.
            unsafe {
                core.device.cmd_pipeline_barrier2(
                    command,
                    &vk::DependencyInfo::default()
                        .memory_barriers(&memory)
                        .image_memory_barriers(&images),
                )
            };
            building.count(Metric::Barriers);
            Ok(())
        })
    }

    /// Copies into each image's shadow what the draw's `reads`, each a texture and what the draw
    /// reads through it, read that the shadow does not hold (`crate::shadow`), each shadow and each
    /// texture's view of it made where none is yet: the rendering ends, one barrier orders every
    /// earlier command before the copies and moves what they copy to the transfer layouts, the
    /// copies are recorded, one barrier moves it all back, and the rendering begins again. Where
    /// the shadows hold every read, nothing is recorded.
    ///
    /// Err: [`Gap::Draw`] of [`DrawKind::AttachmentRead`] for a texture with no view of a shadow;
    /// [`Error::Vulkan`] where a shadow or a view cannot be made or the command buffer cannot
    /// record; [`Error::Quiesced`].
    fn shadow_copies(&mut self, reads: &[(Texture, Vec<Area>)]) -> Result<(), Error> {
        // Every shadow first: a failure leaves what each holds as it was.
        let mut places = Vec::with_capacity(reads.len());
        for (texture, _) in reads {
            let Some(shadow) = texture.shadow()? else {
                return Err(Error::Unsupported(Gap::Draw {
                    kind: DrawKind::AttachmentRead,
                }));
            };
            let hazard = texture.hazard_id();
            let known = (self.shadowed.iter()).position(|s| s.hazard == hazard);
            let place = match known {
                Some(place) => place,
                None => {
                    let target = (self.pass.targets())
                        .find(|t| t.texture.hazard_id() == hazard)
                        .expect("an attachment a draw reads is one of its pass's");
                    let a = &target.attachment;
                    let same = texture.shadow_format() == Some(a.format);
                    self.shadowed.push(Shadowed {
                        hazard,
                        image: a.image,
                        shadow,
                        aspects: a.aspects,
                        rendered: (a.level, a.layer),
                        layout: target.layout,
                        home: a.home,
                        texel: target.texture.texel_bytes(),
                        held: Held::default(),
                        cleared: target.clear_color().filter(|_| same),
                    });
                    self.shadowed.len() - 1
                }
            };
            places.push(place);
        }
        let mut planned: Vec<Vec<Area>> = vec![Vec::new(); self.shadowed.len()];
        for ((_, areas), &place) in reads.iter().zip(&places) {
            let copies = self.shadowed[place].held.plan(areas);
            planned[place].extend(copies);
        }
        if planned.iter().all(Vec::is_empty) {
            return Ok(());
        }
        let (source, destination) = (
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        );
        let read_only = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        let (mut away, mut back, mut copies) = (Vec::new(), Vec::new(), Vec::new());
        let mut clears = Vec::new();
        let (mut count, mut bytes) = (0, 0);
        let (untouched, render_area) = (!self.pass_drew, self.pass.area);
        for (s, areas) in self.shadowed.iter().zip(&planned) {
            if areas.is_empty() {
                continue;
            }
            let range = |(level, [first, end]): (u32, [u32; 2])| vk::ImageSubresourceRange {
                aspect_mask: s.aspects,
                base_mip_level: level,
                level_count: 1,
                base_array_layer: first,
                layer_count: end - first,
            };
            let (level, layer) = s.rendered;
            // At the pass's first draw, the level and layer it cleared holds only its clear
            // colour inside the render area: the shadow's is cleared to it in place of a copy.
            let clear =
                (s.cleared).filter(|_| untouched && shadow::clears(areas, s.rendered, render_area));
            let copied: Vec<Area> = match clear {
                Some(color) => {
                    clears.push((s.shadow, color, range((level, [layer, layer + 1]))));
                    (areas.iter())
                        .filter(|area| !area.reaches(s.rendered))
                        .copied()
                        .collect()
                }
                None => areas.clone(),
            };
            let sources = shadow::ranges(&copied);
            let (rendered, rest) = shadow::parted(&sources, s.rendered);
            if rendered {
                let at = range((level, [layer, layer + 1]));
                away.push(layout_barrier(s.image, at, s.layout, source));
                back.push(layout_barrier(s.image, at, source, s.layout));
            }
            for at in rest.into_iter().map(range) {
                away.push(layout_barrier(s.image, at, s.home, source));
                back.push(layout_barrier(s.image, at, source, s.home));
            }
            for at in shadow::ranges(areas).into_iter().map(range) {
                away.push(layout_barrier(s.shadow, at, read_only, destination));
                back.push(layout_barrier(s.shadow, at, destination, read_only));
            }
            if copied.is_empty() {
                continue;
            }
            let regions: Vec<vk::ImageCopy> = (copied.iter())
                .map(|area| region(area, s.aspects))
                .collect();
            count += copied.len() as u64;
            bytes += (copied.iter())
                .map(|area| area.texels() * s.texel)
                .sum::<u64>();
            copies.push((s.image, s.shadow, regions));
        }
        self.interrupt(|building, core| {
            let command = building.ordered_command()?;
            let memory = [memory_barrier(Reach::Commands)];
            let device = &core.device;
            // SAFETY: the command buffer records outside any rendering and keeps each image until
            // it retires (its pass's attachment texture holds it, and it its shadow); each range
            // is in the layout it moves from: the rendered level and layer in the one the pass
            // renders it in, the rest of the image and the shadow at home; each region lies in
            // both images, in their transfer layouts between the barriers; each cleared range is
            // a colour level and layer of its shadow, in the transfer destination layout, which
            // no region of the copies reaches.
            unsafe {
                device.cmd_pipeline_barrier2(
                    command,
                    &vk::DependencyInfo::default()
                        .memory_barriers(&memory)
                        .image_memory_barriers(&away),
                );
                for (shadow, color, range) in &clears {
                    device.cmd_clear_color_image(command, *shadow, destination, color, &[*range]);
                }
                for (image, shadow, regions) in &copies {
                    device.cmd_copy_image(command, *image, source, *shadow, destination, regions);
                }
                device.cmd_pipeline_barrier2(
                    command,
                    &vk::DependencyInfo::default().image_memory_barriers(&back),
                );
            }
            building.count(Metric::Barriers);
            building.count(Metric::Barriers);
            core.counters.attachment_read_copied(count, bytes);
            Ok(())
        })
    }

    /// `updateFence:afterStages:`: encoders that wait on `fence` after this one see what this
    /// encoder and the ones before it wrote. It records nothing ([`Fence`]): the pass's second
    /// barrier, recorded or owed, already reaches them.
    ///
    /// Err: [`Refusal::OtherDevice`] for another device's fence; [`Gap::FenceOtherQueue`] for a
    /// fence another queue used first.
    pub fn update_fence(&mut self, fence: &Fence) -> Result<(), Error> {
        fence.used_on(&self.core, self.command_buffer.queue())
    }

    /// `waitForFence:beforeStages:`: this encoder sees what the encoders up to `fence`'s last
    /// update wrote. It records nothing ([`Fence`]): everything before the pass is ordered before
    /// it, by its first barrier or the one that ends each earlier command buffer.
    ///
    /// Err: as [`update_fence`](Self::update_fence).
    pub fn wait_for_fence(&mut self, fence: &Fence) -> Result<(), Error> {
        fence.used_on(&self.core, self.command_buffer.queue())
    }

    /// `drawPrimitives:vertexStart:vertexCount:instanceCount:baseInstance:` (the shorter forms are
    /// this call with one instance and a base instance of 0): one `vkCmdDraw`. A draw of no
    /// vertices or no instances records nothing.
    ///
    /// The draw pushes its functions' fields (the addresses of their buffer slots plus their
    /// offsets, the resource IDs of their textures and sampler states, and the indices of the
    /// state's constexpr samplers), binds the vertex buffers the pipeline state's vertex
    /// descriptor reads, and sets the dynamic state that changed. A draw of lines first makes
    /// its content's line `VkPipeline`, where no draw made it before.
    ///
    /// Err: [`Refusal::NoPipeline`]; [`Refusal::NoBufferAtIndex`] for a vertex buffer the vertex
    /// descriptor reads that is not bound; [`Refusal::VertexBufferOffset`];
    /// [`Refusal::BufferOffsetAlignment`] for a buffer a function reaches bound off its argument's
    /// alignment, checked before anything is recorded; [`Refusal::StencilAttachment`];
    /// [`Gap::Draw`] for what the runtime does not build, naming
    /// it; [`Error::Vulkan`] where a pipeline cannot be made or the command buffer cannot record.
    pub fn draw_primitives(
        &mut self,
        primitive_type: PrimitiveType,
        vertex_start: u32,
        vertex_count: u32,
        instance_count: u32,
        base_instance: u32,
    ) -> Result<(), Error> {
        let call = Call::Direct {
            vertex_start,
            vertex_count,
        };
        self.draw(primitive_type, call, instance_count, base_instance)
    }

    /// `drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:instanceCount:baseVertex:baseInstance:`
    /// (the shorter forms are this call with one instance and bases of 0), its index buffer
    /// arguments in `indices`: one `vkCmdDrawIndexed`. The index buffer counts as a read, and the
    /// command buffer keeps it until it retires. A strip restarts at the all-ones index, as
    /// Metal's. A draw of no indices or no instances records nothing.
    ///
    /// Err: as [`draw_primitives`](Self::draw_primitives); [`Refusal::OtherDevice`] for another
    /// device's index buffer; [`Refusal::IndexBufferOffset`]; [`Refusal::ArgumentPlace`] for
    /// indices at a byte of the buffer's block that is not a multiple of the index size, as a
    /// buffer placed at such a byte has.
    pub fn draw_indexed_primitives(
        &mut self,
        primitive_type: PrimitiveType,
        index_count: u32,
        indices: IndexBuffer<'_>,
        instance_count: u32,
        base_vertex: i32,
        base_instance: u32,
    ) -> Result<(), Error> {
        let call = Call::Indexed {
            index_count,
            indices,
            base_vertex,
        };
        self.draw(primitive_type, call, instance_count, base_instance)
    }

    fn draw(
        &mut self,
        primitive_type: PrimitiveType,
        call: Call<'_>,
        instance_count: u32,
        base_instance: u32,
    ) -> Result<(), Error> {
        let Some(state) = &self.pipeline else {
            return Err(Error::Refused(Refusal::NoPipeline));
        };
        let compiled = &state.inner.compiled;
        let draws = &compiled.draws;
        let gap = |kind| Err(Error::Unsupported(Gap::Draw { kind }));
        if !self.renders_into {
            return gap(DrawKind::AttachmentFormats);
        }
        let (topology, restart) = input_assembly(primitive_type);
        let topology = vk::PrimitiveTopology::from_raw(topology.as_raw());
        if topology == vk::PrimitiveTopology::POINT_LIST {
            return gap(DrawKind::Points);
        }
        if self.tests.depth {
            return gap(DrawKind::DepthTest);
        }
        if self.tests.stencil && self.pass.stencil.is_none() {
            return Err(Error::Refused(Refusal::StencilAttachment));
        }
        // The index buffer: its `VkBuffer`, the offset there, and the index type.
        let index = match call {
            Call::Direct { .. } => None,
            Call::Indexed {
                index_count,
                indices:
                    IndexBuffer {
                        buffer,
                        offset,
                        index_type,
                    },
                ..
            } => {
                if !buffer.is_of(&self.core) {
                    return Err(Error::Refused(Refusal::OtherDevice));
                }
                let size = match index_type {
                    IndexType::UInt16 => 2,
                    IndexType::UInt32 => 4,
                };
                let end = u64::from(index_count)
                    .checked_mul(size)
                    .and_then(|bytes| offset.checked_add(bytes));
                if offset % size != 0 || end.is_none_or(|end| end > buffer.length()) {
                    return Err(Error::Refused(Refusal::IndexBufferOffset {
                        offset,
                        length: buffer.length(),
                    }));
                }
                let (reached, at) = buffer.device_range();
                let place = at + offset;
                if !place.is_multiple_of(size) {
                    return Err(Error::Refused(Refusal::ArgumentPlace {
                        place,
                        alignment: size,
                    }));
                }
                let ty = metal2vulkan::metal::state::index_type(index_type);
                Some((buffer, reached, place, vk::IndexType::from_raw(ty.as_raw())))
            }
        };
        if !self.facts.held {
            self.facts.find(draws, &self.tables, &self.pass)?;
        }
        let reads = match self.facts.attachments {
            true => Some(self.attachment_reads(draws)?),
            false => None,
        };
        if !self.facts.held {
            // A buffer argument bound to nothing reads the nil buffer (`Ring::nil`).
            let mut nil = 0;
            for &(_, table, value) in &draws.fields {
                if let FieldValue::BufferAddress { slot, mask } = value {
                    match self.tables[table].buffers.get(slot as usize) {
                        Some(Bound::Buffer { buffer, offset }) => {
                            check_alignment(slot, mask, buffer, *offset)?
                        }
                        Some(Bound::Bytes(_)) => {}
                        Some(Bound::Empty) | None => nil = self.core.ring.nil(&self.core)?,
                    }
                }
            }
            self.facts.nil = nil;
            self.facts.held = true;
        }
        let count = match call {
            Call::Direct { vertex_count, .. } => vertex_count,
            Call::Indexed { index_count, .. } => index_count,
        };
        if count == 0 || instance_count == 0 {
            return Ok(());
        }
        if draws.first_instance_zero && base_instance != 0 {
            return gap(DrawKind::DividedBaseInstance);
        }
        let pipeline = compiled.pipeline(topology)?;
        let writes_memory = draws.writes_memory;
        let building = (self.building.as_ref())
            .expect("an encoder holds its command buffer's recording until it ends");
        // What a draw reaches that each draw looks at again: an attachment of its pass through a
        // texture, a Managed buffer's device copy or a texture over one, or textures over a buffer
        // in an image of their own while any lives (`crate::over`).
        let index_buffer = index.map(|(buffer, ..)| buffer);
        let each = reads.is_some()
            || self.facts.managed
            || index_buffer.is_some_and(|buffer| buffer.managed_overs().is_some())
            || building.copies_over();
        match each {
            true => self.draw_reaches(reads, index_buffer)?,
            false => {
                if !self.shadow_slots.is_empty() {
                    self.shadow_slots.clear();
                    self.push_filled = false;
                }
                self.loop_reads(false);
                (self.building.as_mut())
                    .expect("an encoder holds its command buffer's recording until it ends")
                    .overs_untracked();
            }
        }
        let building = (self.building.as_mut())
            .expect("an encoder holds its command buffer's recording until it ends");
        if self.facts.host {
            building.wrote_host();
        }
        self.wrote |= writes_memory;
        (self.drew, self.pass_drew) = (true, true);
        let (bind, first) = building.bind_graphics(pipeline);
        if !self.push_filled {
            if let Some(state) = &self.pipeline {
                fill_push(
                    &mut self.push,
                    (&self.tables, self.facts.nil),
                    &self.moved,
                    &self.shadow_slots,
                    state,
                );
            }
            self.push_filled = true;
        }
        let push = !self.push.is_empty() && building.push_needed(&self.push);
        let core = &*self.core;
        let command = self.command;
        // SAFETY: the command buffer is inside the pass's rendering, which the pipeline renders
        // into (its formats are the attachments'), and keeps the pipeline state, every bound
        // resource and set 1 until it retires; the layout and set 1 are the device's, set 1 of
        // that layout and update-after-bind.
        unsafe {
            if first {
                core.device.cmd_bind_descriptor_sets(
                    command,
                    vk::PipelineBindPoint::GRAPHICS,
                    core.layout,
                    1,
                    &[core.arrays.set()],
                    &[],
                );
            }
            if bind {
                core.device
                    .cmd_bind_pipeline(command, vk::PipelineBindPoint::GRAPHICS, pipeline);
            }
            self.record_state(topology, restart);
        }
        let core = &*self.core;
        let device = &core.device;
        let building = (self.building.as_mut())
            .expect("an encoder holds its command buffer's recording until it ends");
        // SAFETY: as above; the push bytes lie in the layout's one push range, of `PUSH_STAGES`;
        // every vertex binding the pipeline reads is bound inside its buffer, the ones the
        // command buffer holds already as they are.
        unsafe {
            for &(binding, buffer, offset) in &self.facts.vertex {
                if building.vertex_needed(binding, buffer, offset) {
                    device.cmd_bind_vertex_buffers(command, binding, &[buffer], &[offset]);
                }
            }
            if push {
                device.cmd_push_constants(command, core.layout, PUSH_STAGES, 0, &self.push);
            }
            match (call, index) {
                (
                    Call::Indexed {
                        index_count,
                        base_vertex,
                        ..
                    },
                    Some((_, buffer, offset, ty)),
                ) => {
                    device.cmd_bind_index_buffer(command, buffer, offset, ty);
                    device.cmd_draw_indexed(
                        command,
                        index_count,
                        instance_count,
                        0,
                        base_vertex,
                        base_instance,
                    );
                }
                (
                    Call::Direct {
                        vertex_start,
                        vertex_count,
                    },
                    _,
                ) => device.cmd_draw(
                    command,
                    vertex_count,
                    instance_count,
                    vertex_start,
                    base_instance,
                ),
                (Call::Indexed { .. }, None) => {}
            }
        }
        Ok(())
    }

    /// What a draw of `draws` reads of its pass's attachments through textures. A texture whose
    /// image the pass renders and that reaches the attachment's level and layer reads the
    /// attachment: from the fragment function, never written, of an image a draw may read so that
    /// one attachment of the pass renders. Through the feedback loop, one that reaches neither
    /// reads its image at home, or, once the pass moved the image, in the feedback layout, never
    /// written, and the pass's first draw that reads an attachment moves its image. Without the
    /// loop, one that reaches the attachment reads the image's shadow, into which the draw copies
    /// what it reads first, and one that reaches neither reads its image at home.
    ///
    /// Err: [`Gap::Draw`] of [`DrawKind::AttachmentRead`] for a read the device cannot make.
    fn attachment_reads(&self, draws: &Draws) -> Result<AttachmentReads, Error> {
        let core = &*self.core;
        let drawn = self.drawn();
        let tables = &self.tables;
        let mut found = AttachmentReads {
            reads: false,
            moves: Vec::new(),
            shadow_reads: Vec::new(),
            shadow_slots: Vec::new(),
        };
        for &(table, slot, access) in &draws.textures {
            let Some(texture) = tables[table].texture(slot) else {
                continue;
            };
            let hazard = texture.hazard_id();
            let mut rendered = (self.pass.targets()).filter(|t| t.texture.hazard_id() == hazard);
            let Some(target) = rendered.next() else {
                continue;
            };
            let a = &target.attachment;
            let reaches = texture.reaches(a.level, a.layer);
            let moved = (self.moved.iter()).any(|m| m.hazard == hazard)
                || (found.moves.iter()).any(|m| m.moved.hazard == hazard);
            if !reaches && !moved {
                continue;
            }
            let readable = core.attachment_reads
                && texture.attachment_reads()
                && !access.writes()
                && (table == FRAGMENT || !reaches)
                && rendered.next().is_none();
            if !readable {
                return Err(Error::Unsupported(Gap::Draw {
                    kind: DrawKind::AttachmentRead,
                }));
            }
            if core.feedback.is_none() {
                // What the slot reads: at the fragment's own pixel, or all its texture reaches.
                let own = (table == FRAGMENT && draws.own_pixel.binary_search(&slot).is_ok())
                    .then_some(drawn);
                found
                    .shadow_reads
                    .push((texture.clone(), texture.read_areas(own)));
                found.shadow_slots.push((table, slot));
                continue;
            }
            found.reads |= reaches;
            if !moved {
                found.moves.push(Move {
                    moved: Moved {
                        hazard,
                        image: a.image,
                        whole: texture.whole(),
                        home: a.home,
                    },
                    rendered: target.range(),
                    from: target.layout,
                });
            }
        }
        Ok(found)
    }

    /// The part of a draw that each draw that reaches an attachment of its pass through a texture,
    /// a Managed buffer's device copy or a texture over one, or records while a texture over a
    /// buffer in an image of its own lives, works out again: the attachments it reads (`reads`,
    /// [`attachment_reads`](Self::attachment_reads)), their images moved, which orders everything
    /// before, or the barrier owed paid, and without the feedback loop what their shadows do not
    /// hold copied; then the copies it needs first, its textures over a buffer in an image of
    /// their own (`crate::over`) and the textures over a Managed buffer's device copy it reaches
    /// (`crate::over::ManagedOvers`), each recorded between the rendering's end and its new
    /// beginning. `index` is its index buffer.
    fn draw_reaches(
        &mut self,
        reads: Option<AttachmentReads>,
        index: Option<&Buffer>,
    ) -> Result<(), Error> {
        let reads_attachment = match reads {
            Some(found) => {
                if !found.moves.is_empty() {
                    self.feedback_moves(found.moves)?;
                } else if found.reads && self.owed {
                    self.ordered()?;
                }
                if !found.shadow_reads.is_empty() {
                    self.shadow_copies(&found.shadow_reads)?;
                }
                self.shadow_slots = found.shadow_slots;
                self.push_filled = false;
                found.reads
            }
            None => {
                if !self.shadow_slots.is_empty() {
                    self.shadow_slots.clear();
                    self.push_filled = false;
                }
                false
            }
        };
        self.loop_reads(reads_attachment);
        let Some(state) = &self.pipeline else {
            return Err(Error::Refused(Refusal::NoPipeline));
        };
        let draws = &state.inner.compiled.draws;
        let tables = &self.tables;
        let building = (self.building.as_mut())
            .expect("an encoder holds its command buffer's recording until it ends");
        let vertex_reads = (draws.bindings.iter())
            .filter_map(|&(_, index)| tables[VERTEX].buffer(index))
            .chain(index);
        let accesses: Vec<(u32, Access)> = match building.copies_over() {
            false => Vec::new(),
            true => (draws.buffers.iter())
                .filter_map(|&(table, slot, access)| {
                    Some((tables[table].buffer(slot)?.id(), access))
                })
                .chain(
                    (draws.textures.iter()).filter_map(|&(table, slot, access)| {
                        Some((tables[table].texture(slot)?.hazard_id(), access))
                    }),
                )
                .chain(
                    vertex_reads
                        .clone()
                        .map(|buffer| (buffer.id(), Access::Read)),
                )
                .collect(),
        };
        let unbound = draws.reads_unbound || draws.writes_unbound;
        let writes_unbound = draws.writes_unbound;
        let copies = building.overs_owed(&accesses, unbound);
        let mut backs = Vec::new();
        for &(table, slot, access) in &draws.buffers {
            if let Some(buffer) = tables[table].buffer(slot) {
                if buffer.managed_overs().is_some() {
                    backs.extend(over::backs(buffer, None, true));
                    if access.writes() {
                        building.wrote_managed(buffer);
                    }
                }
            }
        }
        for buffer in vertex_reads {
            if buffer.managed_overs().is_some() {
                backs.extend(over::backs(buffer, None, true));
            }
        }
        for &(table, slot, access) in &draws.textures {
            if let (true, Some(texture)) = (access.writes(), tables[table].texture(slot)) {
                building.wrote_texture_managed(texture);
            }
        }
        if !copies.is_empty() || !backs.is_empty() {
            self.interrupt(|building, _| {
                building.record_overs(copies)?;
                building.managed_backs(backs)
            })?;
        }
        (self.building.as_mut())
            .expect("an encoder holds its command buffer's recording until it ends")
            .overs_after(&accesses, writes_unbound);
        Ok(())
    }

    /// Sets the feedback loop's aspects for a draw that reads an attachment (`reads`) or none,
    /// where the device reads attachments through it: dirty where they change.
    fn loop_reads(&mut self, reads: bool) {
        if self.core.feedback.is_some() {
            let aspects = match reads {
                true => vk::ImageAspectFlags::COLOR,
                false => vk::ImageAspectFlags::NONE,
            };
            if aspects != self.loop_aspects {
                self.loop_aspects = aspects;
                self.dirty |= LOOP;
            }
        }
    }

    /// Records the dynamic state that changed since the last draw, and the draw's topology and
    /// primitive restart where they changed.
    ///
    /// # Safety
    ///
    /// The encoder's command buffer is inside its pass's rendering.
    unsafe fn record_state(&mut self, topology: vk::PrimitiveTopology, restart: bool) {
        let device = &self.core.device;
        let feedback = self.core.feedback.as_ref();
        let command = self.command;
        let dirty = self.dirty;
        self.dirty = 0;
        let both = [vk::StencilFaceFlags::FRONT, vk::StencilFaceFlags::BACK];
        // SAFETY: the command buffer records; each state is dynamic in every render pipeline.
        unsafe {
            if dirty & VIEWPORT != 0 {
                device.cmd_set_viewport(command, 0, &[vk_viewport(&self.viewport)]);
                device.cmd_set_scissor(command, 0, &[self.vk_scissor()]);
            }
            if dirty & CULL != 0 {
                let mode = match self.cull_mode {
                    CullMode::None => vk::CullModeFlags::NONE,
                    CullMode::Front => vk::CullModeFlags::FRONT,
                    CullMode::Back => vk::CullModeFlags::BACK,
                };
                device.cmd_set_cull_mode(command, mode);
            }
            if dirty & FRONT != 0 {
                let face = match self.winding {
                    Winding::Clockwise => vk::FrontFace::CLOCKWISE,
                    Winding::CounterClockwise => vk::FrontFace::COUNTER_CLOCKWISE,
                };
                device.cmd_set_front_face(command, face);
            }
            if dirty & TESTS != 0 {
                // No pass has a depth attachment: a state that tests depth draws nothing
                // (`DrawKind::DepthTest`).
                device.cmd_set_depth_test_enable(command, false);
                device.cmd_set_depth_write_enable(command, false);
                device.cmd_set_depth_compare_op(command, vk::CompareOp::ALWAYS);
                device.cmd_set_stencil_test_enable(command, self.tests.stencil);
                for (face, f) in both.iter().zip(&self.tests.faces) {
                    device.cmd_set_stencil_op(
                        command,
                        *face,
                        f.fail,
                        f.pass,
                        f.depth_fail,
                        f.compare,
                    );
                    device.cmd_set_stencil_compare_mask(command, *face, f.compare_mask);
                    device.cmd_set_stencil_write_mask(command, *face, f.write_mask);
                }
            }
            if dirty & REFERENCE != 0 {
                for (face, reference) in both.iter().zip(self.stencil_reference) {
                    device.cmd_set_stencil_reference(command, *face, reference);
                }
            }
            if dirty & BLEND != 0 {
                device.cmd_set_blend_constants(command, &self.blend_color);
            }
            if dirty & BIAS != 0 {
                device.cmd_set_depth_bias(command, 0.0, 0.0, 0.0);
                device.cmd_set_depth_bias_enable(command, false);
            }
            // Every pipeline takes the enable as dynamic state where the device reads attachments.
            if let (true, Some(feedback)) = (dirty & LOOP != 0, feedback) {
                (feedback.fp().cmd_set_attachment_feedback_loop_enable_ext)(
                    command,
                    self.loop_aspects,
                );
            }
            if self.topology != Some((topology, restart)) {
                device.cmd_set_primitive_topology(command, topology);
                device.cmd_set_primitive_restart_enable(command, restart);
                self.topology = Some((topology, restart));
            }
        }
    }

    /// The pixels a draw covers, per axis `[first, end)`: Metal's scissor rectangle cut to the
    /// pixels its viewport covers.
    fn drawn(&self) -> [[u32; 2]; 2] {
        let pixels = viewport_pixels(&self.viewport, self.pass.area);
        [0, 1].map(|axis| {
            let first = self.scissor[axis][0].max(pixels[axis][0]);
            let end = self.scissor[axis][1].min(pixels[axis][1]);
            [first, end.max(first)]
        })
    }

    /// The draw's scissor: the pixels it covers ([`drawn`](Self::drawn)).
    fn vk_scissor(&self) -> vk::Rect2D {
        let [x, y] = self.drawn();
        // Each bound lies in the render area, which fits an `i32`.
        vk::Rect2D {
            offset: vk::Offset2D {
                x: x[0] as i32,
                y: y[0] as i32,
            },
            extent: vk::Extent2D {
                width: x[1] - x[0],
                height: y[1] - y[0],
            },
        }
    }

    /// Ends the pass's rendering, has `record` record into the command buffer, and begins the
    /// rendering again, loading what it stored; the rendering's draws since it began are
    /// unwritten again.
    fn interrupt(
        &mut self,
        record: impl FnOnce(&mut Building, &Core) -> Result<(), Error>,
    ) -> Result<(), Error> {
        let building = (self.building.as_mut())
            .expect("an encoder holds its command buffer's recording until it ends");
        let core = &*self.core;
        // SAFETY: the command buffer is inside the pass's rendering, which nothing else records
        // into while the encoder holds it.
        unsafe { core.device.cmd_end_rendering(self.command) };
        let recorded = record(building, core);
        // SAFETY: the command buffer records outside any rendering and keeps each attachment's
        // texture until it retires; each attachment stays in the layout the pass renders it in,
        // and what `record` recorded orders itself after the rendering and before the next by
        // barriers of its own.
        unsafe { self.pass.begin(&core.device, self.command, true) };
        self.wrote = false;
        recorded
    }

    /// `endEncoding`: ends the pass, one `vkCmdEndRendering` and the pass's second barrier, owed
    /// where no attachment moves home.
    pub fn end_encoding(mut self) {
        self.end();
    }

    fn end(&mut self) {
        let Some(mut building) = self.building.take() else {
            return;
        };
        let core = &*self.core;
        let later = [memory_barrier(Reach::Commands)];
        let looped = vk::ImageLayout::ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT;
        let moved = &self.moved;
        let back: Vec<vk::ImageMemoryBarrier2> = (self.pass.targets())
            .filter(|t| t.moves() && !moved.iter().any(|m| m.hazard == t.texture.hazard_id()))
            .map(|t| t.barrier(t.layout, t.attachment.home))
            .chain(
                moved
                    .iter()
                    .map(|m| layout_barrier(m.image, m.whole, looped, m.home)),
            )
            .collect();
        let device = &core.device;
        // SAFETY: the command buffer is inside the pass's rendering, which nothing else records
        // into while the encoder holds it.
        unsafe { device.cmd_end_rendering(self.command) };
        if back.is_empty() {
            // Nothing moves home: the barrier that orders the pass before later commands is the
            // segment's next command's, or its end's.
            building.owe_pass();
        } else {
            // SAFETY: the command buffer records outside any rendering; each attachment the
            // barrier moves is in the layout the pass rendered it in.
            unsafe {
                device.cmd_pipeline_barrier2(
                    self.command,
                    &vk::DependencyInfo::default()
                        .memory_barriers(&later)
                        .image_memory_barriers(&back),
                );
            }
            building.count(Metric::Barriers);
        }
        self.command_buffer.close_encoder(building);
    }
}

impl Drop for RenderCommandEncoder {
    fn drop(&mut self) {
        self.end();
    }
}

/// Fills `push`, the push bytes of a draw of `state`, from the `tables`, a buffer argument bound to
/// nothing taking `nil`, the nil buffer's address: a texture whose image the pass `moved` to the
/// feedback layout reads through its second element in that layout, and one in `shadow_slots`,
/// which reads a shadow, through its second element, the shadow's.
fn fill_push(
    push: &mut Vec<u8>,
    (tables, nil): (&[Tables; 2], u64),
    moved: &[Moved],
    shadow_slots: &[(usize, u32)],
    state: &RenderPipelineState,
) {
    let draws = &state.inner.compiled.draws;
    let ids = &state.inner.numbered.ids;
    push.clear();
    push.resize(draws.push_len.next_multiple_of(4) as usize, 0);
    for &(offset, table, value) in &draws.fields {
        let tables = &tables[table];
        let at = offset as usize;
        match value {
            // At a multiple of the argument's alignment, which the draw checked first.
            FieldValue::BufferAddress { slot, .. } => {
                let address = tables.address(slot, nil);
                push[at..at + 8].copy_from_slice(&address.to_le_bytes());
            }
            FieldValue::TextureId(slot) => {
                let id = (tables.texture(slot)).map_or(0, |t| {
                    let second = (moved.iter()).any(|m| m.hazard == t.hazard_id())
                        || shadow_slots.contains(&(table, slot));
                    match second {
                        true => t.second_resource_id().to_bits(),
                        false => t.gpu_resource_id().to_bits(),
                    }
                });
                push[at..at + 8].copy_from_slice(&id.to_le_bytes());
            }
            FieldValue::SamplerId(slot) => {
                let id = match tables.samplers.get(slot as usize) {
                    Some(Some(sampler)) => sampler.gpu_resource_id().to_bits(),
                    _ => 0,
                };
                push[at..at + 8].copy_from_slice(&id.to_le_bytes());
            }
            FieldValue::IntegerChannels(slot) => {
                let word = (tables.texture(slot))
                    .map_or(0, |t| integer_channels_word(t.descriptor().pixel_format));
                push[at..at + 4].copy_from_slice(&word.to_le_bytes());
            }
            FieldValue::ConstexprSampler(place) => {
                let index = ids.get(place).copied().unwrap_or(0);
                push[at..at + 4].copy_from_slice(&index.to_le_bytes());
            }
            // A render content has no dispatch region (`render::draws`).
            FieldValue::DispatchRegion => {}
        }
    }
}

/// The copy of `area` from an image into its shadow, of `aspects`, at the same place in both.
fn region(area: &Area, aspects: vk::ImageAspectFlags) -> vk::ImageCopy {
    let layers = vk::ImageSubresourceLayers {
        aspect_mask: aspects,
        mip_level: area.level,
        base_array_layer: area.layers[0],
        layer_count: area.layers[1] - area.layers[0],
    };
    // Each bound lies in the level, which fits an `i32`.
    let offset = vk::Offset3D {
        x: area.x[0] as i32,
        y: area.y[0] as i32,
        z: 0,
    };
    vk::ImageCopy {
        src_subresource: layers,
        src_offset: offset,
        dst_subresource: layers,
        dst_offset: offset,
        extent: vk::Extent3D {
            width: area.x[1] - area.x[0],
            height: area.y[1] - area.y[0],
            depth: 1,
        },
    }
}

/// `whole` but `rendered`, one level and layer of it: at most four ranges.
fn around(
    whole: vk::ImageSubresourceRange,
    rendered: vk::ImageSubresourceRange,
) -> Vec<vk::ImageSubresourceRange> {
    let (level, layer) = (rendered.base_mip_level, rendered.base_array_layer);
    let levels = |first: u32, end: u32| {
        (first < end).then_some(vk::ImageSubresourceRange {
            base_mip_level: first,
            level_count: end - first,
            ..whole
        })
    };
    let layers = |first: u32, end: u32| {
        (first < end).then_some(vk::ImageSubresourceRange {
            base_mip_level: level,
            level_count: 1,
            base_array_layer: first,
            layer_count: end - first,
            ..whole
        })
    };
    [
        levels(whole.base_mip_level, level),
        levels(level + 1, whole.base_mip_level + whole.level_count),
        layers(whole.base_array_layer, layer),
        layers(layer + 1, whole.base_array_layer + whole.layer_count),
    ]
    .into_iter()
    .flatten()
    .collect()
}

/// The slot of buffer index `index`.
///
/// Err: [`Refusal::BufferIndex`] for an index of 31 or more.
fn slot(index: u32) -> Result<usize, Error> {
    match index < BUFFER_SLOTS {
        true => Ok(index as usize),
        false => Err(Error::Refused(Refusal::BufferIndex { index })),
    }
}

/// The Vulkan viewport of `v`: its height negated, from `y + height`, so the draw follows Metal's
/// clip space; a viewport of no width or height takes one pixel, behind an empty scissor.
pub(crate) fn vk_viewport(v: &Viewport) -> vk::Viewport {
    let width = if v.width == 0.0 { 1.0 } else { v.width };
    let height = if v.height == 0.0 { 1.0 } else { v.height };
    vk::Viewport {
        x: v.origin_x as f32,
        y: (v.origin_y + height) as f32,
        width: width as f32,
        height: (-height) as f32,
        min_depth: v.znear as f32,
        max_depth: v.zfar as f32,
    }
}

/// The pixels `v` covers in a render area of `extent`, per axis `[first, end)`: none for a
/// viewport of no width or height.
pub(crate) fn viewport_pixels(v: &Viewport, extent: [u32; 2]) -> [[u32; 2]; 2] {
    let empty = v.width == 0.0 || v.height == 0.0;
    let axis = |origin: f64, size: f64, limit: u32| {
        let (low, high) = match size < 0.0 {
            true => (origin + size, origin),
            false => (origin, origin + size),
        };
        let limit = f64::from(limit);
        let first = low.floor().clamp(0.0, limit) as u32;
        let end = high.ceil().clamp(0.0, limit) as u32;
        match empty {
            true => [first, first],
            false => [first, end.max(first)],
        }
    };
    [
        axis(v.origin_x, v.width, extent[0]),
        axis(v.origin_y, v.height, extent[1]),
    ]
}
