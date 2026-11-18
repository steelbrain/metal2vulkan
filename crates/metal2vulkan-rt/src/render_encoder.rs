//! Render command encoders: one dynamic rendering a Metal render encoder, begun as the encoder is
//! made and ended as it ends, and the draws recorded into it.
//!
//! The pass orders itself as a blit's copy does. Its first barrier runs it after every earlier
//! command, makes their writes visible to it, and moves each attachment whose home is
//! shader-read-only to the attachment layout; an attachment whose home is general or the
//! attachment layout renders in it. It pays the barrier an earlier encoder owed, and the hazard
//! sets start again. Its second, after `vkCmdEndRendering`, moves those attachments home and makes
//! the pass's writes visible to every later command. A pass is two barriers, whatever it renders.
//!
//! Draws record into the rendering with no barrier between them, as Metal orders the memory
//! accesses of no two draws of one encoder: what a draw writes, a later draw of the encoder sees
//! after a memory barrier. A memory barrier after a draw that wrote memory ends the rendering,
//! records one barrier and begins the rendering again, loading what it stored; so does a draw that
//! needs a copy first (a texture over a buffer in an image of its own, or one over a Managed
//! buffer's device copy, `crate::over`), recording the copy between. A memory barrier after no
//! such draw records nothing: the pass's first barrier ordered everything before it.
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
use crate::encoder::{check_alignment, BUFFER_SLOTS, SAMPLER_SLOTS, TEXTURE_SLOTS};
use crate::error::{DrawKind, Error, Gap, Limit, Refusal};
use crate::fence::Fence;
use crate::hazard::Access;
use crate::over;
use crate::pass::{Pass, RenderPassDescriptor};
use crate::pipeline::{FieldValue, PUSH_STAGES};
use crate::record::{memory_barrier, Reach};
use crate::render::{RenderPipelineState, FRAGMENT, VERTEX};
use crate::ring::{Placed, CHUNK};
use crate::sampler::SamplerState;
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
/// front face, the depth and stencil tests, the stencil references, the blend constants, and the
/// depth bias the encoder does not set.
const VIEWPORT: u32 = 1;
const CULL: u32 = 2;
const FRONT: u32 = 4;
const TESTS: u32 = 8;
const REFERENCE: u32 = 16;
const BLEND: u32 = 32;
const BIAS: u32 = 64;
const ALL: u32 = 127;

/// What a buffer slot holds.
#[derive(Clone, Default)]
enum Bound {
    #[default]
    Empty,
    Buffer {
        buffer: Buffer,
        offset: u64,
    },
    /// A `set_*_bytes` copy in the parameter ring.
    Bytes(Placed),
}

/// One function's argument tables.
struct Tables {
    buffers: Vec<Bound>,
    textures: Vec<Option<Texture>>,
    samplers: Vec<Option<SamplerState>>,
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
        self.textures.get(slot as usize)?.as_ref()
    }

    /// The address of buffer slot `slot`: the buffer's plus its offset, or the bytes' copy; 0
    /// where none is bound.
    fn address(&self, slot: u32) -> u64 {
        match self.buffers.get(slot as usize) {
            Some(Bound::Buffer { buffer, offset }) => buffer.gpu_address().wrapping_add(*offset),
            Some(Bound::Bytes(placed)) => placed.address,
            Some(Bound::Empty) | None => 0,
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
    /// What the command buffer records, held while the encoder is open.
    building: Option<Building>,
    /// The checked pass, for its second barrier and to begin its rendering again.
    pass: Pass,
    /// The command buffer the pass records into.
    command: vk::CommandBuffer,
    /// The vertex function's argument tables, then the fragment function's.
    tables: [Tables; 2],
    pipeline: Option<RenderPipelineState>,
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
    /// The fixed-function vertex buffers of the draw being recorded: binding, buffer and offset.
    vertex_buffers: Vec<(u32, vk::Buffer, u64)>,
    /// The push bytes of the draw being recorded.
    push: Vec<u8>,
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
        match begin(&mut building, descriptor) {
            Ok((pass, command)) => {
                let [width, height] = pass.area;
                Ok(RenderCommandEncoder {
                    command_buffer: self.clone(),
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
                    vertex_buffers: Vec::new(),
                    push: Vec::new(),
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

/// Checks `descriptor` and records its pass's first barrier and its `vkCmdBeginRendering`.
fn begin(
    building: &mut Building,
    descriptor: &RenderPassDescriptor,
) -> Result<(Pass, vk::CommandBuffer), Error> {
    let core = Arc::clone(building.device.core());
    let pass = Pass::check(&core, descriptor)?;
    let overs: Vec<(u32, Access)> = (pass.targets())
        .map(|t| (t.texture.hazard_id(), Access::ReadWrite))
        .collect();
    for t in pass.targets() {
        building.keep_texture(&t.texture);
        building.wrote_texture_managed(&t.texture);
    }
    building.overs_before(&overs, false)?;
    let command = building.ordered_command()?;
    // Marked after `ordered_command`, as a blit's copy marks it: the host reads a texture over a
    // Shared buffer's bytes in the buffer.
    if pass.targets().any(|t| t.texture.host_reads()) {
        building.wrote_host();
    }
    building.overs_after(&overs, false);
    core.counters.count(Metric::Passes);
    let earlier = [memory_barrier(Reach::Commands)];
    let away: Vec<vk::ImageMemoryBarrier2> = (pass.targets())
        .filter(|t| t.moves())
        .map(|t| t.barrier(t.attachment.home, t.layout))
        .collect();
    let device = &core.device;
    // SAFETY: the command buffer records outside any render pass and keeps each attachment's
    // texture until it retires; each attachment is in its home layout between commands, which
    // the barrier moves to the layout the pass names.
    unsafe {
        device.cmd_pipeline_barrier2(
            command,
            &vk::DependencyInfo::default()
                .memory_barriers(&earlier)
                .image_memory_barriers(&away),
        );
        pass.begin(device, command, false);
    }
    core.counters.count(Metric::Barriers);
    Ok((pass, command))
}

impl RenderCommandEncoder {
    fn building(&mut self) -> &mut Building {
        self.building
            .as_mut()
            .expect("an encoder holds its command buffer's recording until it ends")
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
        self.building().keep_render_pipeline(state);
        self.renders_into = state.inner.compiled.renders_into(&self.pass);
        self.pipeline = Some(state.clone());
        Ok(())
    }

    /// `setVertexBuffer:offset:atIndex:` (`setVertexBuffers:offsets:withRange:` is a loop of it).
    /// `None` unbinds the slot. The slot serves the vertex function's buffer argument of its index
    /// and the vertex descriptor's layout of its index alike. The command buffer keeps the buffer's
    /// block until it retires.
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
                let building = self.building();
                if !buffer.is_of(building.device.core()) {
                    return Err(Error::Refused(Refusal::OtherDevice));
                }
                building.keep(buffer);
                Bound::Buffer {
                    buffer: buffer.clone(),
                    offset,
                }
            }
            None => Bound::Empty,
        };
        self.tables[table].buffers[slot] = bound;
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
        if let Some(texture) = texture {
            let building = self.building();
            if !texture.is_of(building.device.core()) {
                return Err(Error::Refused(Refusal::OtherDevice));
            }
            building.keep_texture(texture);
        }
        self.tables[table].textures[index as usize] = texture.cloned();
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
        if let Some(sampler) = sampler {
            let building = self.building();
            if !sampler.is_of(building.device.core()) {
                return Err(Error::Refused(Refusal::OtherDevice));
            }
            building.keep_sampler(sampler);
        }
        self.tables[table].samplers[index as usize] = sampler.cloned();
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
        let core = Arc::clone(self.building().device.core());
        let limits = &core.limits;
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
    /// rendering begins again, loading what it stored; after none, nothing is recorded, as the
    /// pass's first barrier ordered everything before it.
    ///
    /// Err: [`Error::Vulkan`] where the command buffer cannot record.
    pub fn memory_barrier(&mut self) -> Result<(), Error> {
        if !self.wrote {
            return Ok(());
        }
        self.interrupt(|building| {
            let command = building.ordered_command()?;
            building.barrier(command, Reach::Commands);
            Ok(())
        })
    }

    /// `updateFence:afterStages:`: encoders that wait on `fence` after this one see what this
    /// encoder and the ones before it wrote. It records nothing ([`Fence`]): the pass's second
    /// barrier already reaches them.
    ///
    /// Err: [`Refusal::OtherDevice`] for another device's fence; [`Gap::FenceOtherQueue`] for a
    /// fence another queue used first.
    pub fn update_fence(&mut self, fence: &Fence) -> Result<(), Error> {
        let core = Arc::clone(self.building().device.core());
        fence.used_on(&core, self.command_buffer.queue())
    }

    /// `waitForFence:beforeStages:`: this encoder sees what the encoders up to `fence`'s last
    /// update wrote. It records nothing ([`Fence`]): the pass's first barrier already orders them.
    ///
    /// Err: as [`update_fence`](Self::update_fence).
    pub fn wait_for_fence(&mut self, fence: &Fence) -> Result<(), Error> {
        let core = Arc::clone(self.building().device.core());
        fence.used_on(&core, self.command_buffer.queue())
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
    /// device's index buffer; [`Refusal::IndexBufferOffset`].
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
        let state = (self.pipeline.clone()).ok_or(Error::Refused(Refusal::NoPipeline))?;
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
        let core = Arc::clone(self.building().device.core());
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
                if !buffer.is_of(&core) {
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
                let ty = metal2vulkan::metal::state::index_type(index_type);
                Some((
                    buffer,
                    reached,
                    at + offset,
                    vk::IndexType::from_raw(ty.as_raw()),
                ))
            }
        };
        // The vertex buffers the vertex descriptor reads.
        self.vertex_buffers.clear();
        for &(binding, index) in &draws.bindings {
            let (buffer, offset) = match self.tables[VERTEX].buffers.get(index as usize) {
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
            self.vertex_buffers.push((binding, buffer, offset));
        }
        // A texture the pass renders into, read or written as the pass renders it, is a feedback
        // loop the runtime does not build.
        let tables = &self.tables;
        let rendered = (draws.textures.iter()).any(|&(table, slot, _)| {
            tables[table].texture(slot).is_some_and(|texture| {
                (self.pass.targets()).any(|t| t.texture.hazard_id() == texture.hazard_id())
            })
        });
        if rendered {
            return gap(DrawKind::AttachmentRead);
        }
        for &(_, table, value) in &draws.fields {
            if let FieldValue::BufferAddress { slot, mask } = value {
                if let Some(Bound::Buffer { buffer, offset }) =
                    tables[table].buffers.get(slot as usize)
                {
                    check_alignment(slot, mask, buffer, *offset)?;
                }
            }
        }
        let count = match call {
            Call::Direct { vertex_count, .. } => vertex_count,
            Call::Indexed { index_count, .. } => index_count,
        };
        if count == 0 || instance_count == 0 {
            return Ok(());
        }
        let pipeline = compiled.pipeline(topology)?;
        // The copies the draw needs first: its textures over a buffer in an image of their own
        // (`crate::over`), and the textures over a Managed buffer's device copy it reaches
        // (`crate::over::ManagedOvers`), each recorded between the rendering's end and its new
        // beginning.
        let building = (self.building.as_mut())
            .expect("an encoder holds its command buffer's recording until it ends");
        let vertex_reads = (draws.bindings.iter())
            .filter_map(|&(_, index)| tables[VERTEX].buffer(index))
            .chain(index.map(|(buffer, ..)| buffer));
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
        // A write the host may read once the command buffer completes: to a Shared buffer or a
        // texture over one, or through an address outside the draw's resources.
        let host = draws.writes_unbound
            || (draws.buffers.iter()).any(|&(table, slot, access)| {
                access.writes() && tables[table].buffer(slot).is_some_and(Buffer::is_shared)
            })
            || (draws.textures.iter()).any(|&(table, slot, access)| {
                access.writes() && tables[table].texture(slot).is_some_and(Texture::host_reads)
            });
        if !copies.is_empty() || !backs.is_empty() {
            self.interrupt(|building| {
                building.record_overs(copies)?;
                building.managed_backs(backs)
            })?;
        }
        let building = (self.building.as_mut())
            .expect("an encoder holds its command buffer's recording until it ends");
        building.overs_after(&accesses, draws.writes_unbound);
        if host {
            building.wrote_host();
        }
        self.wrote |= draws.writes_memory;
        let (bind, first) = building.bind_graphics(pipeline);
        self.fill_push(&state);
        let device = &core.device;
        let command = self.command;
        // SAFETY: the command buffer is inside the pass's rendering, which the pipeline renders
        // into (its formats are the attachments'), and keeps the pipeline state, every bound
        // resource and set 1 until it retires; the layout and set 1 are the device's, set 1 of
        // that layout and update-after-bind; the push bytes lie in the layout's one push range,
        // of `PUSH_STAGES`; every vertex binding the pipeline reads is bound inside its buffer.
        unsafe {
            if first {
                device.cmd_bind_descriptor_sets(
                    command,
                    vk::PipelineBindPoint::GRAPHICS,
                    core.layout,
                    1,
                    &[core.arrays.set()],
                    &[],
                );
            }
            if bind {
                device.cmd_bind_pipeline(command, vk::PipelineBindPoint::GRAPHICS, pipeline);
            }
            self.record_state(device, topology, restart);
            for &(binding, buffer, offset) in &self.vertex_buffers {
                device.cmd_bind_vertex_buffers(command, binding, &[buffer], &[offset]);
            }
            if !self.push.is_empty() {
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

    /// Fills the push bytes of a draw of `state` from the tables.
    fn fill_push(&mut self, state: &RenderPipelineState) {
        let draws = &state.inner.compiled.draws;
        let ids = &state.inner.numbered.ids;
        let push = &mut self.push;
        push.clear();
        push.resize(draws.push_len.next_multiple_of(4) as usize, 0);
        for &(offset, table, value) in &draws.fields {
            let tables = &self.tables[table];
            let at = offset as usize;
            match value {
                // At a multiple of the argument's alignment, which the draw checked first.
                FieldValue::BufferAddress { slot, .. } => {
                    let address = tables.address(slot);
                    push[at..at + 8].copy_from_slice(&address.to_le_bytes());
                }
                FieldValue::TextureId(slot) => {
                    let id = (tables.texture(slot)).map_or(0, |t| t.gpu_resource_id().to_bits());
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

    /// Records the dynamic state that changed since the last draw, and the draw's topology and
    /// primitive restart where they changed.
    ///
    /// # Safety
    ///
    /// The encoder's command buffer is inside its pass's rendering.
    unsafe fn record_state(
        &mut self,
        device: &ash::Device,
        topology: vk::PrimitiveTopology,
        restart: bool,
    ) {
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
            if self.topology != Some((topology, restart)) {
                device.cmd_set_primitive_topology(command, topology);
                device.cmd_set_primitive_restart_enable(command, restart);
                self.topology = Some((topology, restart));
            }
        }
    }

    /// The draw's scissor: Metal's rectangle cut to the pixels its viewport covers.
    fn vk_scissor(&self) -> vk::Rect2D {
        let pixels = viewport_pixels(&self.viewport, self.pass.area);
        let [x, y] = [0, 1].map(|axis| {
            let first = self.scissor[axis][0].max(pixels[axis][0]);
            let end = self.scissor[axis][1].min(pixels[axis][1]);
            [first, end.max(first)]
        });
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
        record: impl FnOnce(&mut Building) -> Result<(), Error>,
    ) -> Result<(), Error> {
        let building = (self.building.as_mut())
            .expect("an encoder holds its command buffer's recording until it ends");
        let core = Arc::clone(building.device.core());
        // SAFETY: the command buffer is inside the pass's rendering, which nothing else records
        // into while the encoder holds it.
        unsafe { core.device.cmd_end_rendering(self.command) };
        let recorded = record(building);
        // SAFETY: the command buffer records outside any rendering and keeps each attachment's
        // texture until it retires; each attachment stays in the layout the pass renders it in,
        // and what `record` recorded orders itself after the rendering and before the next by
        // barriers of its own.
        unsafe { self.pass.begin(&core.device, self.command, true) };
        self.wrote = false;
        recorded
    }

    /// `endEncoding`: ends the pass, one `vkCmdEndRendering` and the pass's second barrier.
    pub fn end_encoding(mut self) {
        self.end();
    }

    fn end(&mut self) {
        let Some(building) = self.building.take() else {
            return;
        };
        let core = Arc::clone(building.device.core());
        let later = [memory_barrier(Reach::Commands)];
        let back: Vec<vk::ImageMemoryBarrier2> = (self.pass.targets())
            .filter(|t| t.moves())
            .map(|t| t.barrier(t.layout, t.attachment.home))
            .collect();
        let device = &core.device;
        // SAFETY: the command buffer is inside the pass's rendering, which nothing else records
        // into while the encoder holds it; each attachment the barrier moves is in the layout the
        // pass rendered it in.
        unsafe {
            device.cmd_end_rendering(self.command);
            device.cmd_pipeline_barrier2(
                self.command,
                &vk::DependencyInfo::default()
                    .memory_barriers(&later)
                    .image_memory_barriers(&back),
            );
        }
        core.counters.count(Metric::Barriers);
        self.command_buffer.close_encoder(building);
    }
}

impl Drop for RenderCommandEncoder {
    fn drop(&mut self) {
        self.end();
    }
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
