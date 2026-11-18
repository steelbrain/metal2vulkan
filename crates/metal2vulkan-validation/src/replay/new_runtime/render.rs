//! Render pipeline states, render pass descriptors and render encoders on the new runtime.
//!
//! What the runtime does not build yet is skipped and counted: a render pipeline state it answers
//! with a typed gap or the translator refuses (`render_states`); a render encoder whose pass the
//! runtime's descriptor cannot hold (a resolve texture, a depth plane, store action options, a
//! visibility result buffer, layered or tile rendering, a skipped texture) or that it answers with
//! a typed gap (`encoders`); a draw whose pipeline state, or a buffer or texture bound to either
//! stage, is a skipped object (`draws`); and a render command the runtime answers with a typed
//! gap, a draw it does not build among them (`render_commands`, by the gap, with the first frame
//! and call of each).

use super::super::args::{arg, data, index, list, metal, refuse, uint, Fields};
use super::super::translator::{self, attachment_keys, LINKED_KEYS, PASS_KEYS};
use super::encoders::first;
use super::{failed, unaddressed, NewRuntime, Object};
use crate::replay::player::{Made, PlayError};
use crate::replay::stream::Call;
use crate::replay::value::Value;
use metal2vulkan_rt as rt;
use std::collections::HashSet;

/// Where a binding is: the stage (fragment or not) and whether it is a texture.
type Slot = (bool, bool, u32);

/// A render encoder, its stages' buffer and texture slots that hold a skipped object, and whether
/// its pipeline state is a skipped one.
pub(super) struct Render {
    encoder: rt::RenderCommandEncoder,
    skipped: HashSet<Slot>,
    skipped_pipeline: bool,
}

/// The attachment index of the pass's own properties in [`Pass::unheld`].
const OWN: usize = usize::MAX;

/// A render pass descriptor, and the properties it holds that the runtime's descriptor does not:
/// each by its attachment (the colors' indices, then the depth and stencil attachments') or
/// [`OWN`], and its key. The runtime makes an encoder of it only where there are none.
#[derive(Default)]
pub(super) struct Pass {
    descriptor: rt::RenderPassDescriptor,
    unheld: HashSet<(usize, &'static str)>,
}

impl Pass {
    /// Records that property `key` of attachment `at` is set to a value the runtime's descriptor
    /// holds (`held`) or one it does not.
    fn hold(&mut self, at: usize, key: &'static str, held: bool) {
        match held {
            true => self.unheld.remove(&(at, key)),
            false => self.unheld.insert((at, key)),
        };
    }
}

/// `MTLLoadAction`.
fn load_action(call: &Call, raw: u64) -> Result<rt::LoadAction, PlayError> {
    Ok(match raw {
        0 => rt::LoadAction::DontCare,
        1 => rt::LoadAction::Load,
        2 => rt::LoadAction::Clear,
        other => return Err(refuse(call, format!("load action {other}"))),
    })
}

/// `MTLStoreAction`.
fn store_action(call: &Call, raw: u64) -> Result<rt::StoreAction, PlayError> {
    Ok(match raw {
        0 => rt::StoreAction::DontCare,
        1 => rt::StoreAction::Store,
        2 => rt::StoreAction::MultisampleResolve,
        3 => rt::StoreAction::StoreAndMultisampleResolve,
        4 => rt::StoreAction::Unknown,
        5 => rt::StoreAction::CustomSampleDepthStore,
        other => return Err(refuse(call, format!("store action {other}"))),
    })
}

/// What an attachment's setter record sets of the properties every attachment has that the
/// runtime's descriptor holds: an object property `Some(None)` where it is set to nil.
struct AttachmentSet {
    texture: Option<Option<rt::Texture>>,
    level: Option<u32>,
    slice: Option<u32>,
    load_action: Option<rt::LoadAction>,
    store_action: Option<rt::StoreAction>,
}

/// `set`'s properties set on the attachment descriptor `a`, of any of the three kinds.
macro_rules! set_attachment {
    ($set:expr, $a:expr) => {{
        let (set, a) = ($set, $a);
        if let Some(texture) = set.texture {
            a.texture = texture;
        }
        if let Some(level) = set.level {
            a.level = level;
        }
        if let Some(slice) = set.slice {
            a.slice = slice;
        }
        if let Some(action) = set.load_action {
            a.load_action = action;
        }
        if let Some(action) = set.store_action {
            a.store_action = action;
        }
    }};
}

impl NewRuntime {
    /// `newRenderPipelineStateWithDescriptor:error:`: its constexpr samplers numbered as a compute
    /// pipeline state's are; skipped and counted where the runtime answers a typed gap or the
    /// translator refuses it. The runtime's states take no label, and every index is the
    /// backend's (`resourceIndex` is not played).
    pub(super) fn render_pipeline_state(
        &mut self,
        call: &Call,
    ) -> Result<Option<(Object, Made)>, PlayError> {
        let (descriptor, _) = translator::render_pipeline(call, |f, key, linked| {
            let Some(id) = f.object(key)? else {
                return Ok(None);
            };
            let linked = self.linked(call, f.fields(linked, &LINKED_KEYS)?)?;
            let function = self.function_of(call, id)?;
            Ok(Some(rt::StageSource::Function(
                function.descriptor.with_linked(linked),
            )))
        })?;
        self.unbound.render(call.result, &descriptor);
        let made = self.numbered(|device, ids| {
            device.new_render_pipeline_state(&descriptor, |samplers| ids(samplers))
        });
        match made {
            Ok(state) => unaddressed(Object::RenderPipeline(state)),
            Err(error @ (rt::Error::Unsupported(_) | rt::Error::Translation(_))) => {
                self.skips.render_states += 1;
                (self.skips.first_render_state).get_or_insert((call.result, error.to_string()));
                unaddressed(Object::Skipped)
            }
            Err(error) => Err(failed(call)(error)),
        }
    }

    /// `renderCommandEncoderWithDescriptor:` of `command_buffer`: skipped and counted where the
    /// pass holds what the runtime's descriptor does not, or the runtime answers a typed gap.
    pub(super) fn render_encoder(
        &mut self,
        call: &Call,
        command_buffer: &rt::CommandBuffer,
    ) -> Result<Option<(Object, Made)>, PlayError> {
        let descriptor = match self.get(call, Self::id(call, 0)?)? {
            Object::Pass(pass) => (pass.unheld.is_empty()).then(|| pass.descriptor.clone()),
            _ => return Err(refuse(call, "argument 0 is no render pass descriptor")),
        };
        let Some(descriptor) = descriptor else {
            return self.skip_encoder();
        };
        match command_buffer.render_command_encoder(&descriptor) {
            Ok(encoder) => unaddressed(Object::Render(Box::new(Render {
                encoder,
                skipped: HashSet::new(),
                skipped_pipeline: false,
            }))),
            Err(rt::Error::Unsupported(_)) => self.skip_encoder(),
            Err(error) => Err(failed(call)(error)),
        }
    }

    /// A render encoder the runtime does not build, counted.
    pub(super) fn skip_encoder(&mut self) -> Result<Option<(Object, Made)>, PlayError> {
        self.skips.encoders += 1;
        unaddressed(Object::Skipped)
    }

    fn pass(&mut self, id: u64) -> &mut Pass {
        match self.objects.get_mut(&id) {
            Some(Object::Pass(pass)) => pass,
            _ => unreachable!("a render pass descriptor's call"),
        }
    }

    /// What an attachment's map `f` sets of the properties every attachment has, and records in
    /// the pass what of them the runtime's descriptor does not hold.
    fn attachment_set(
        &mut self,
        call: &Call,
        f: &Fields,
        at: usize,
    ) -> Result<AttachmentSet, PlayError> {
        // A skipped texture is held as none, and the pass marked as one the runtime does not
        // build while the attachment names it.
        let (texture, skipped) = match f.set_object("texture")? {
            Some(Some(id)) if self.skipped(id) => (Some(None), Some(true)),
            Some(Some(id)) => (Some(Some(self.texture_of(call, id)?)), Some(false)),
            Some(None) => (Some(None), Some(false)),
            None => (None, None),
        };
        let resolve = f.set_object("resolveTexture")?;
        let depth_plane = f.uint("depthPlane")?;
        let options = f.uint("storeActionOptions")?;
        let narrow = |key: &str| -> Result<Option<u32>, PlayError> {
            (f.uint(key)?)
                .map(|raw| super::super::args::narrow(call, raw, key))
                .transpose()
        };
        let set = AttachmentSet {
            level: narrow("level")?,
            slice: narrow("slice")?,
            load_action: (f.uint("loadAction")?)
                .map(|raw| load_action(call, raw))
                .transpose()?,
            store_action: (f.uint("storeAction")?)
                .map(|raw| store_action(call, raw))
                .transpose()?,
            texture,
        };
        let pass = self.pass(call.receiver);
        if let Some(skipped) = skipped {
            pass.hold(at, "texture", !skipped);
        }
        if let Some(resolve) = resolve {
            pass.hold(at, "resolveTexture", resolve.is_none());
        }
        if let Some(plane) = depth_plane {
            pass.hold(at, "depthPlane", plane == 0);
        }
        if let Some(options) = options {
            pass.hold(at, "storeActionOptions", options == 0);
        }
        Ok(set)
    }

    /// A setter record of the render pass descriptor `call` is sent to: each property its map
    /// sets. `defaultColorSampleCount` and `defaultRasterSampleCount`, which change nothing a pass
    /// with an attachment shows (the runtime refuses one without), are not played; a rate map and
    /// accumulated visibility results are refused, as the old runtime's backend refuses them.
    pub(super) fn pass_call(&mut self, call: &Call) -> Result<(), PlayError> {
        match &*call.selector {
            "m2v.setPass" => {
                let f = Fields::arg(call, 0, &PASS_KEYS)?;
                if let Some(Some(_)) = f.set_object("rasterizationRateMap")? {
                    return Err(refuse(call, "a rasterization rate map"));
                }
                f.only("visibilityResultType", Fields::uint, 0)?;
                let visibility = f.set_object("visibilityResultBuffer")?;
                let layers = f.uint("renderTargetArrayLength")?;
                let tile = [
                    "imageblockSampleLength",
                    "threadgroupMemoryLength",
                    "tileWidth",
                    "tileHeight",
                ]
                .map(|key| f.uint(key).map(|value| (key, value)));
                let width = f.count("renderTargetWidth")?;
                let height = f.count("renderTargetHeight")?;
                let pass = self.pass(call.receiver);
                if let Some(visibility) = visibility {
                    pass.hold(OWN, "visibilityResultBuffer", visibility.is_none());
                }
                if let Some(layers) = layers {
                    pass.hold(OWN, "renderTargetArrayLength", layers <= 1);
                }
                for set in tile {
                    if let (key, Some(value)) = set? {
                        pass.hold(OWN, key, value == 0);
                    }
                }
                if let Some(width) = width {
                    pass.descriptor.render_target_width = width;
                }
                if let Some(height) = height {
                    pass.descriptor.render_target_height = height;
                }
            }
            "m2v.setColorAttachment" => {
                let at = usize::try_from(uint(call, 0)?)
                    .ok()
                    .filter(|&at| at < rt::MAX_COLOR_ATTACHMENTS)
                    .ok_or_else(|| refuse(call, "a color attachment past the last"))?;
                let f = Fields::arg(call, 1, &attachment_keys(&["clearColor"]))?;
                let set = self.attachment_set(call, &f, at)?;
                let clear = match f.fields("clearColor", &["red", "green", "blue", "alpha"])? {
                    Some(c) => Some(rt::ClearColor {
                        red: c.required("red", Fields::float)?,
                        green: c.required("green", Fields::float)?,
                        blue: c.required("blue", Fields::float)?,
                        alpha: c.required("alpha", Fields::float)?,
                    }),
                    None => None,
                };
                let a = &mut self.pass(call.receiver).descriptor.color_attachments[at];
                set_attachment!(set, &mut *a);
                if let Some(clear) = clear {
                    a.clear_color = clear;
                }
            }
            "m2v.setDepthAttachment" => {
                let own = ["clearDepth", "depthResolveFilter"];
                let f = Fields::arg(call, 0, &attachment_keys(&own))?;
                let set = self.attachment_set(call, &f, rt::MAX_COLOR_ATTACHMENTS)?;
                let clear = f.float("clearDepth")?;
                let a = &mut self.pass(call.receiver).descriptor.depth_attachment;
                set_attachment!(set, &mut *a);
                if let Some(clear) = clear {
                    a.clear_depth = clear;
                }
            }
            "m2v.setStencilAttachment" => {
                let own = ["clearStencil", "stencilResolveFilter"];
                let f = Fields::arg(call, 0, &attachment_keys(&own))?;
                let set = self.attachment_set(call, &f, rt::MAX_COLOR_ATTACHMENTS + 1)?;
                let clear = f.count("clearStencil")?;
                let a = &mut self.pass(call.receiver).descriptor.stencil_attachment;
                set_attachment!(set, &mut *a);
                if let Some(clear) = clear {
                    a.clear_stencil = clear;
                }
            }
            _ => return Err(refuse(call, "not played on a render pass descriptor")),
        }
        Ok(())
    }

    fn render(&mut self, id: u64) -> &mut Render {
        match self.objects.get_mut(&id) {
            Some(Object::Render(render)) => render,
            _ => unreachable!("a render encoder's call"),
        }
    }

    /// A buffer or texture argument `at` bound at `slot`: none where it is nil, and the slot
    /// marked skipped where it is a skipped object.
    fn bound<T>(
        &mut self,
        call: &Call,
        at: usize,
        slot: Slot,
        of: impl Fn(&Self, &Call, u64) -> Result<T, PlayError>,
    ) -> Result<Option<Option<T>>, PlayError> {
        let id = match arg(call, at)? {
            Value::Nil => None,
            Value::Object(id) => Some(*id),
            other => return Err(refuse(call, format!("{other} is no buffer or texture"))),
        };
        let held = match id {
            Some(id) if self.skipped(id) => {
                self.render(call.receiver).skipped.insert(slot);
                return Ok(None);
            }
            Some(id) => Some(of(self, call, id)?),
            None => None,
        };
        self.render(call.receiver).skipped.remove(&slot);
        Ok(Some(held))
    }

    pub(super) fn render_call(&mut self, call: &Call) -> Result<(), PlayError> {
        let id = call.receiver;
        let fragment = call.selector.starts_with("setFragment");
        let result = match &*call.selector {
            "endEncoding" => {
                let Some(Object::Render(render)) = self.objects.remove(&id) else {
                    unreachable!("a render encoder's call");
                };
                render.encoder.end_encoding();
                Ok(())
            }
            "setRenderPipelineState:" => {
                let state = match self.get(call, Self::id(call, 0)?)? {
                    Object::RenderPipeline(state) => Some(state.clone()),
                    Object::Skipped => None,
                    _ => return Err(refuse(call, "argument 0 is no render pipeline state")),
                };
                let render = self.render(id);
                render.skipped_pipeline = state.is_none();
                match state {
                    Some(state) => render.encoder.set_render_pipeline_state(&state),
                    None => Ok(()),
                }
            }
            "setDepthStencilState:" => {
                let state = match arg(call, 0)? {
                    Value::Nil => None,
                    _ => match self.get(call, Self::id(call, 0)?)? {
                        Object::DepthStencil(state) => Some(state.clone()),
                        _ => return Err(refuse(call, "argument 0 is no depth-stencil state")),
                    },
                };
                self.render(id)
                    .encoder
                    .set_depth_stencil_state(state.as_ref())
            }
            "setVertexBuffer:offset:atIndex:" | "setFragmentBuffer:offset:atIndex:" => {
                let (offset, at) = (uint(call, 1)?, index(call, 2)?);
                let Some(buffer) = self.bound(call, 0, (fragment, false, at), Self::buffer_of)?
                else {
                    return Ok(());
                };
                let encoder = &mut self.render(id).encoder;
                match fragment {
                    false => encoder.set_vertex_buffer(buffer.as_ref(), offset, at),
                    true => encoder.set_fragment_buffer(buffer.as_ref(), offset, at),
                }
            }
            "setVertexBufferOffset:atIndex:" | "setFragmentBufferOffset:atIndex:" => {
                let (offset, at) = (uint(call, 0)?, index(call, 1)?);
                let encoder = &mut self.render(id).encoder;
                match fragment {
                    false => encoder.set_vertex_buffer_offset(offset, at),
                    true => encoder.set_fragment_buffer_offset(offset, at),
                }
            }
            "setVertexBytes:length:atIndex:" | "setFragmentBytes:length:atIndex:" => {
                let bytes = data(call, 0)?.to_vec();
                if bytes.len() as u64 != uint(call, 1)? {
                    return Err(refuse(call, "data of another length than its length"));
                }
                let at = index(call, 2)?;
                let render = self.render(id);
                render.skipped.remove(&(fragment, false, at));
                match fragment {
                    false => render.encoder.set_vertex_bytes(&bytes, at),
                    true => render.encoder.set_fragment_bytes(&bytes, at),
                }
            }
            "setVertexTexture:atIndex:" | "setFragmentTexture:atIndex:" => {
                let at = index(call, 1)?;
                let Some(texture) = self.bound(call, 0, (fragment, true, at), Self::texture_of)?
                else {
                    return Ok(());
                };
                let encoder = &mut self.render(id).encoder;
                match fragment {
                    false => encoder.set_vertex_texture(texture.as_ref(), at),
                    true => encoder.set_fragment_texture(texture.as_ref(), at),
                }
            }
            "setVertexSamplerState:atIndex:" | "setFragmentSamplerState:atIndex:" => {
                let sampler = self.sampler(call, arg(call, 0)?)?;
                let at = index(call, 1)?;
                let encoder = &mut self.render(id).encoder;
                match fragment {
                    false => encoder.set_vertex_sampler_state(sampler.as_ref(), at),
                    true => encoder.set_fragment_sampler_state(sampler.as_ref(), at),
                }
            }
            "setVertexSamplerStates:withRange:" | "setFragmentSamplerStates:withRange:" => {
                let samplers = list(call, 0)?.iter();
                let samplers = samplers.map(|value| self.sampler(call, value));
                let samplers = samplers.collect::<Result<Vec<_>, _>>()?;
                let at = first(call, 1, samplers.len())?;
                let encoder = &mut self.render(id).encoder;
                let mut result = Ok(());
                for (slot, sampler) in (at..).zip(&samplers) {
                    result = result.and(match fragment {
                        false => encoder.set_vertex_sampler_state(sampler.as_ref(), slot),
                        true => encoder.set_fragment_sampler_state(sampler.as_ref(), slot),
                    });
                }
                result
            }
            "setViewport:" => {
                let known = ["originX", "originY", "width", "height", "znear", "zfar"];
                let f = Fields::arg(call, 0, &known)?;
                let [x, y, width, height, znear, zfar] =
                    known.map(|key| f.required(key, Fields::float));
                let viewport = rt::Viewport {
                    origin_x: x?,
                    origin_y: y?,
                    width: width?,
                    height: height?,
                    znear: znear?,
                    zfar: zfar?,
                };
                self.render(id).encoder.set_viewport(viewport)
            }
            "setScissorRect:" => {
                let known = ["x", "y", "width", "height"];
                let f = Fields::arg(call, 0, &known)?;
                let [x, y, width, height] = known.map(|key| f.required(key, Fields::uint));
                let rect = rt::ScissorRect {
                    x: x?,
                    y: y?,
                    width: width?,
                    height: height?,
                };
                self.render(id).encoder.set_scissor_rect(rect);
                Ok(())
            }
            "setCullMode:" => {
                let mode = metal(call, uint(call, 0)?)?;
                self.render(id).encoder.set_cull_mode(mode);
                Ok(())
            }
            "setFrontFacingWinding:" => {
                let winding = metal(call, uint(call, 0)?)?;
                self.render(id).encoder.set_front_facing_winding(winding);
                Ok(())
            }
            "setBlendColorRed:green:blue:alpha:" => {
                let channel = |at: usize| match arg(call, at)?.as_f64() {
                    Some(value) => Ok(value as f32),
                    None => Err(refuse(call, format!("argument {at} is no float"))),
                };
                let [red, green, blue, alpha] =
                    [channel(0)?, channel(1)?, channel(2)?, channel(3)?];
                self.render(id)
                    .encoder
                    .set_blend_color(red, green, blue, alpha);
                Ok(())
            }
            "setStencilReferenceValue:" => {
                let value = index(call, 0)?;
                self.render(id).encoder.set_stencil_reference_value(value);
                Ok(())
            }
            "setStencilFrontReferenceValue:backReferenceValue:" => {
                let (front, back) = (index(call, 0)?, index(call, 1)?);
                let encoder = &mut self.render(id).encoder;
                encoder.set_stencil_front_back_reference_value(front, back);
                Ok(())
            }
            "drawPrimitives:vertexStart:vertexCount:"
            | "drawPrimitives:vertexStart:vertexCount:instanceCount:"
            | "drawPrimitives:vertexStart:vertexCount:instanceCount:baseInstance:" => {
                let kind = metal(call, uint(call, 0)?)?;
                let (start, count) = (index(call, 1)?, index(call, 2)?);
                let instances = match call.args.len() {
                    3 => 1,
                    _ => index(call, 3)?,
                };
                let base_instance = match call.args.len() {
                    5 => index(call, 4)?,
                    _ => 0,
                };
                if self.draw_skipped(id) {
                    return Ok(());
                }
                (self.render(id).encoder).draw_primitives(
                    kind,
                    start,
                    count,
                    instances,
                    base_instance,
                )
            }
            "drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:"
            | "drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:\
               instanceCount:"
            | "drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:\
               instanceCount:baseVertex:baseInstance:" => {
                let kind = metal(call, uint(call, 0)?)?;
                let count = index(call, 1)?;
                let index_type = metal(call, uint(call, 2)?)?;
                let buffer = Self::id(call, 3)?;
                let offset = uint(call, 4)?;
                let instances = match call.args.len() {
                    5 => 1,
                    _ => index(call, 5)?,
                };
                let (base_vertex, base_instance) = match call.args.len() {
                    8 => {
                        // `NSInteger`: signed where the recorder saw it negative.
                        let base = match arg(call, 6)? {
                            Value::Sint(base) => i32::try_from(*base).ok(),
                            Value::Uint(base) => i32::try_from(*base).ok(),
                            _ => None,
                        };
                        let base = base.ok_or_else(|| refuse(call, "argument 6 is no i32"))?;
                        (base, index(call, 7)?)
                    }
                    _ => (0, 0),
                };
                if self.skipped(buffer) {
                    self.skips.draws += 1;
                    return Ok(());
                }
                let buffer = self.buffer_of(call, buffer)?;
                if self.draw_skipped(id) {
                    return Ok(());
                }
                let indices = rt::IndexBuffer {
                    buffer: &buffer,
                    offset,
                    index_type,
                };
                (self.render(id).encoder).draw_indexed_primitives(
                    kind,
                    count,
                    indices,
                    instances,
                    base_vertex,
                    base_instance,
                )
            }
            "updateFence:afterStages:" => {
                let fence = self.fence_of(call, 0)?;
                self.render(id).encoder.update_fence(&fence)
            }
            "waitForFence:beforeStages:" => {
                let fence = self.fence_of(call, 0)?;
                self.render(id).encoder.wait_for_fence(&fence)
            }
            "memoryBarrierWithScope:afterStages:beforeStages:" => {
                self.render(id).encoder.memory_barrier()
            }
            _ => return Err(refuse(call, "not played on a render encoder")),
        };
        match result {
            // A command the runtime does not build: skipped and counted, as a render state it
            // does not build is.
            Err(rt::Error::Unsupported(gap)) => {
                let (frame, at) = self.at;
                let kind = self.skips.render_commands.entry(format!("{gap:?}"));
                kind.or_insert((0, frame, at)).0 += 1;
                Ok(())
            }
            result => result.map_err(failed(call)),
        }
    }

    /// Whether the render encoder `id`'s draw is skipped, counted: its pipeline state or a bound
    /// buffer or texture is a skipped object.
    fn draw_skipped(&mut self, id: u64) -> bool {
        let render = self.render(id);
        let skipped = render.skipped_pipeline || !render.skipped.is_empty();
        if skipped {
            self.skips.draws += 1;
        }
        skipped
    }

    /// The sampler state `value` names, or none where it is nil.
    fn sampler(&self, call: &Call, value: &Value) -> Result<Option<rt::SamplerState>, PlayError> {
        match value {
            Value::Nil => Ok(None),
            Value::Object(object) => match self.get(call, *object)? {
                Object::Sampler(sampler) => Ok(Some(sampler.clone())),
                _ => Err(refuse(call, format!("#{object} is no sampler state"))),
            },
            other => Err(refuse(call, format!("{other} is no sampler state"))),
        }
    }
}
