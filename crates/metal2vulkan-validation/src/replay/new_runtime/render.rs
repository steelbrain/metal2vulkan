//! Render pipeline states, render pass descriptors and render encoders on the new runtime.
//!
//! What the runtime does not build yet is skipped and counted: a render pipeline state it answers
//! with a typed gap or the translator refuses (`render_states`, by the gap, with the first state
//! and the runtime's answer of each); a render encoder whose pass the
//! runtime's descriptor cannot hold (a depth plane, store action options, a visibility result
//! buffer, layered or tile rendering, a skipped texture or resolve texture) or that it answers with
//! a typed gap (`encoders`); a draw whose pipeline state, or a buffer or texture bound to either
//! stage, is a skipped object (`draws`); and a render command the runtime answers with a typed
//! gap, a draw it does not build among them (`render_commands`, by the gap, with the first frame
//! and call of each).

use super::super::args::{arg, data, index, list, metal, refuse, uint, Fields};
use super::super::translator::{self, attachment_keys, LINKED_KEYS, PASS_KEYS};
use super::encoders::first;
use super::{failed, unaddressed, NewRuntime, Object, Skips};
use crate::replay::player::{Made, PlayError};
use crate::replay::stream::Call;
use crate::replay::value::Value;
use metal2vulkan_runtime as rt;
use std::collections::{HashMap, HashSet};

/// Where a binding is: the stage (fragment or not) and whether it is a texture.
type Slot = (bool, bool, u32);

/// The tessellation factor buffer's place among a render encoder's slots that hold a skipped
/// object.
const FACTORS: Slot = (false, false, u32::MAX);

/// A render encoder, its stages' buffer and texture slots that hold a skipped object, its stages'
/// sampler slots whose bind it skipped (`clamped`), and whether its pipeline state is a skipped
/// one.
pub(super) struct Render {
    encoder: rt::RenderCommandEncoder,
    skipped: HashSet<Slot>,
    skipped_samplers: HashSet<(bool, u32)>,
    skipped_pipeline: bool,
    /// The bound pipeline state's record lacks the `tessellation` key.
    unrecorded_tessellation: bool,
}

/// The render pipeline states whose record lacks the `tessellation` key, and the patch draws
/// played through one. Such a record comes from a recorder that did not carry the key, so the
/// program's tessellation state is unknown. The player draws these patches with Metal's defaults,
/// its stand-in for that state.
#[derive(Default)]
pub(super) struct UnrecordedTessellation {
    states: HashSet<u64>,
    /// The patch draws played through such a state.
    pub(super) patch_draws: u64,
}

/// Whether a render pipeline record's arguments carry the `tessellation` key.
pub(super) fn records_tessellation(args: &[Value]) -> bool {
    match args.first() {
        Some(Value::Map(map)) => map.iter().any(|(key, _)| &**key == "tessellation"),
        _ => false,
    }
}

impl UnrecordedTessellation {
    /// A released object: a state's mark goes with it.
    pub(super) fn released(&mut self, object: u64) {
        self.states.remove(&object);
    }

    /// The report's line: the patch draws played with the player's stand-in.
    pub(super) fn line(&self) -> String {
        format!(
            "patch draws through a render pipeline state whose record lacks tessellation {}: \
             the program's tessellation state is unknown, and the player drew them with Metal's \
             defaults",
            self.patch_draws
        )
    }
}

/// The attachment index of the pass's own properties in [`Pass::unheld`].
const OWN: usize = usize::MAX;

/// A render pass descriptor, and the properties it holds that the runtime's descriptor does not:
/// each by its attachment (the colors' indices, then the depth and stencil attachments') or
/// [`OWN`], and its key. The runtime makes an encoder of it only where there are none. Each
/// attachment's texture as the stream names it, by attachment, for the surfaces' shapes a pass
/// writes (`surfaces`).
#[derive(Default)]
pub(super) struct Pass {
    descriptor: rt::RenderPassDescriptor,
    unheld: HashSet<(usize, &'static str)>,
    textures: HashMap<usize, u64>,
}

impl Pass {
    /// Its attachments' textures, as the stream names them.
    pub(super) fn textures(&self) -> impl Iterator<Item = u64> + '_ {
        self.textures.values().copied()
    }

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

/// The viewport of `value`, an `MTLViewport`'s fields.
fn viewport(call: &Call, value: &Value) -> Result<rt::Viewport, PlayError> {
    let known = ["originX", "originY", "width", "height", "znear", "zfar"];
    let f = Fields::of(call, value, &known)?;
    let [x, y, width, height, znear, zfar] = known.map(|key| f.required(key, Fields::float));
    Ok(rt::Viewport {
        origin_x: x?,
        origin_y: y?,
        width: width?,
        height: height?,
        znear: znear?,
        zfar: zfar?,
    })
}

/// The scissor rectangle of `value`, an `MTLScissorRect`'s fields.
fn scissor(call: &Call, value: &Value) -> Result<rt::ScissorRect, PlayError> {
    let known = ["x", "y", "width", "height"];
    let f = Fields::of(call, value, &known)?;
    let [x, y, width, height] = known.map(|key| f.required(key, Fields::uint));
    Ok(rt::ScissorRect {
        x: x?,
        y: y?,
        width: width?,
        height: height?,
    })
}

/// `MTLMultisampleDepthResolveFilter`.
fn depth_resolve_filter(
    call: &Call,
    raw: u64,
) -> Result<rt::MultisampleDepthResolveFilter, PlayError> {
    Ok(match raw {
        0 => rt::MultisampleDepthResolveFilter::Sample0,
        1 => rt::MultisampleDepthResolveFilter::Min,
        2 => rt::MultisampleDepthResolveFilter::Max,
        other => return Err(refuse(call, format!("depth resolve filter {other}"))),
    })
}

/// `MTLMultisampleStencilResolveFilter`.
fn stencil_resolve_filter(
    call: &Call,
    raw: u64,
) -> Result<rt::MultisampleStencilResolveFilter, PlayError> {
    Ok(match raw {
        0 => rt::MultisampleStencilResolveFilter::Sample0,
        1 => rt::MultisampleStencilResolveFilter::DepthResolvedSample,
        other => return Err(refuse(call, format!("stencil resolve filter {other}"))),
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
    resolve_texture: Option<Option<rt::Texture>>,
    resolve_level: Option<u32>,
    resolve_slice: Option<u32>,
    resolve_depth_plane: Option<u32>,
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
        if let Some(texture) = set.resolve_texture {
            a.resolve_texture = texture;
        }
        if let Some(level) = set.resolve_level {
            a.resolve_level = level;
        }
        if let Some(slice) = set.resolve_slice {
            a.resolve_slice = slice;
        }
        if let Some(plane) = set.resolve_depth_plane {
            a.resolve_depth_plane = plane;
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
        let made = self.numbered(|device, ids| {
            device.new_render_pipeline_state(&descriptor, |samplers| ids(samplers))
        });
        match made {
            Ok(state) => {
                if !records_tessellation(&call.args) {
                    self.unrecorded_tessellation.states.insert(call.result);
                }
                unaddressed(Object::RenderPipeline(state))
            }
            Err(error) if Skips::skipped_state(&mut self.skips.render_states, call, &error) => {
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
            Object::Pass(pass) => match pass.unheld.is_empty() {
                true => Ok(pass.descriptor.clone()),
                false => {
                    let mut keys: Vec<&str> = pass.unheld.iter().map(|&(_, key)| key).collect();
                    keys.sort_unstable();
                    keys.dedup();
                    Err(format!("Unheld({})", keys.join(", ")))
                }
            },
            _ => return Err(refuse(call, "argument 0 is no render pass descriptor")),
        };
        let descriptor = match descriptor {
            Ok(descriptor) => descriptor,
            Err(unheld) => return self.skip_encoder(unheld),
        };
        match command_buffer.render_command_encoder(&descriptor) {
            Ok(encoder) => unaddressed(Object::Render(Box::new(Render {
                encoder,
                skipped: HashSet::new(),
                skipped_samplers: HashSet::new(),
                skipped_pipeline: false,
                unrecorded_tessellation: false,
            }))),
            Err(rt::Error::Unsupported(gap)) => self.skip_encoder(format!("{gap:?}")),
            Err(error) => Err(failed(call)(error)),
        }
    }

    /// A render encoder the runtime does not build, counted, by `kind`.
    pub(super) fn skip_encoder(
        &mut self,
        kind: String,
    ) -> Result<Option<(Object, Made)>, PlayError> {
        self.skips.encoders += 1;
        let (frame, at) = self.at;
        let kinds = self.skips.encoder_kinds.entry(kind);
        kinds.or_insert((0, frame, at)).0 += 1;
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
        let recorded = f.set_object("texture")?;
        let (texture, skipped) = match recorded {
            Some(Some(id)) if self.skipped(id) => (Some(None), Some(true)),
            Some(Some(id)) => (Some(Some(self.texture_of(call, id)?)), Some(false)),
            Some(None) => (Some(None), Some(false)),
            None => (None, None),
        };
        // A skipped resolve texture is held as none, and the pass marked as one the runtime does
        // not build while the attachment names it.
        let (resolve_texture, resolve_skipped) = match f.set_object("resolveTexture")? {
            Some(Some(id)) if self.skipped(id) => (Some(None), Some(true)),
            Some(Some(id)) => (Some(Some(self.texture_of(call, id)?)), Some(false)),
            Some(None) => (Some(None), Some(false)),
            None => (None, None),
        };
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
            resolve_texture,
            resolve_level: narrow("resolveLevel")?,
            resolve_slice: narrow("resolveSlice")?,
            resolve_depth_plane: narrow("resolveDepthPlane")?,
        };
        let pass = self.pass(call.receiver);
        match recorded {
            Some(Some(id)) => pass.textures.insert(at, id),
            Some(None) => pass.textures.remove(&at),
            None => None,
        };
        if let Some(skipped) = skipped {
            pass.hold(at, "texture", !skipped);
        }
        if let Some(skipped) = resolve_skipped {
            pass.hold(at, "resolveTexture", !skipped);
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
    /// accumulated visibility results are refused.
    pub(super) fn pass_call(&mut self, call: &Call) -> Result<(), PlayError> {
        match &*call.selector {
            "m2v.setPass" => {
                let f = Fields::arg(call, 0, &PASS_KEYS)?;
                if let Some(Some(_)) = f.set_object("rasterizationRateMap")? {
                    return Err(refuse(call, "a rasterization rate map"));
                }
                f.only("visibilityResultType", Fields::uint, 0)?;
                // A skipped buffer is held as none, and the pass marked as one the runtime does
                // not build while the descriptor names it.
                let (visibility, skipped) = match f.set_object("visibilityResultBuffer")? {
                    Some(Some(id)) if self.skipped(id) => (Some(None), true),
                    Some(Some(id)) => (Some(Some(self.buffer_of(call, id)?)), false),
                    Some(None) => (Some(None), false),
                    None => (None, false),
                };
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
                    pass.hold(OWN, "visibilityResultBuffer", !skipped);
                    pass.descriptor.visibility_result_buffer = visibility;
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
                let filter = (f.uint("depthResolveFilter")?)
                    .map(|raw| depth_resolve_filter(call, raw))
                    .transpose()?;
                let a = &mut self.pass(call.receiver).descriptor.depth_attachment;
                set_attachment!(set, &mut *a);
                if let Some(clear) = clear {
                    a.clear_depth = clear;
                }
                if let Some(filter) = filter {
                    a.depth_resolve_filter = filter;
                }
            }
            "m2v.setStencilAttachment" => {
                let own = ["clearStencil", "stencilResolveFilter"];
                let f = Fields::arg(call, 0, &attachment_keys(&own))?;
                let set = self.attachment_set(call, &f, rt::MAX_COLOR_ATTACHMENTS + 1)?;
                let clear = f.count("clearStencil")?;
                let filter = (f.uint("stencilResolveFilter")?)
                    .map(|raw| stencil_resolve_filter(call, raw))
                    .transpose()?;
                let a = &mut self.pass(call.receiver).descriptor.stencil_attachment;
                set_attachment!(set, &mut *a);
                if let Some(clear) = clear {
                    a.clear_stencil = clear;
                }
                if let Some(filter) = filter {
                    a.stencil_resolve_filter = filter;
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

    /// A buffer or texture `value` of `call` bound at `slot`: none where it is nil, and the slot
    /// marked skipped where it is a skipped object.
    fn bound<T>(
        &mut self,
        call: &Call,
        value: &Value,
        slot: Slot,
        of: impl Fn(&Self, &Call, u64) -> Result<T, PlayError>,
    ) -> Result<Option<Option<T>>, PlayError> {
        let id = match value {
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

    /// Binds the buffer `value` names at `slot` of the stage `fragment` names, at `offset`:
    /// nothing where it is a skipped object, whose slot [`bound`](Self::bound) marks.
    fn bind_render_buffer(
        &mut self,
        call: &Call,
        value: &Value,
        fragment: bool,
        offset: u64,
        slot: u32,
    ) -> Result<Result<(), rt::Error>, PlayError> {
        let at = (fragment, false, slot);
        let Some(buffer) = self.bound(call, value, at, Self::buffer_of)? else {
            return Ok(Ok(()));
        };
        let encoder = &mut self.render(call.receiver).encoder;
        Ok(match fragment {
            false => encoder.set_vertex_buffer(buffer.as_ref(), offset, slot),
            true => encoder.set_fragment_buffer(buffer.as_ref(), offset, slot),
        })
    }

    /// Binds the texture `value` names at `slot` of the stage `fragment` names: nothing where it
    /// is a skipped object, whose slot [`bound`](Self::bound) marks.
    fn bind_render_texture(
        &mut self,
        call: &Call,
        value: &Value,
        fragment: bool,
        slot: u32,
    ) -> Result<Result<(), rt::Error>, PlayError> {
        let at = (fragment, true, slot);
        let Some(texture) = self.bound(call, value, at, Self::texture_of)? else {
            return Ok(Ok(()));
        };
        let encoder = &mut self.render(call.receiver).encoder;
        Ok(match fragment {
            false => encoder.set_vertex_texture(texture.as_ref(), slot),
            true => encoder.set_fragment_texture(texture.as_ref(), slot),
        })
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
                let state_id = Self::id(call, 0)?;
                let state = match self.get(call, state_id)? {
                    Object::RenderPipeline(state) => Some(state.clone()),
                    Object::Skipped => None,
                    _ => return Err(refuse(call, "argument 0 is no render pipeline state")),
                };
                let unrecorded = self.unrecorded_tessellation.states.contains(&state_id);
                let render = self.render(id);
                render.skipped_pipeline = state.is_none();
                render.unrecorded_tessellation = unrecorded;
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
                self.bind_render_buffer(call, arg(call, 0)?, fragment, offset, at)?
            }
            "setVertexBuffers:offsets:withRange:" | "setFragmentBuffers:offsets:withRange:" => {
                let buffers = list(call, 0)?;
                let offsets = arg(call, 1)?.as_uints();
                let offsets = offsets.ok_or_else(|| refuse(call, "argument 1 is no offsets"))?;
                if offsets.len() != buffers.len() {
                    return Err(refuse(call, "as many offsets as buffers"));
                }
                let at = first(call, 2, buffers.len())?;
                let mut result = Ok(());
                for ((slot, buffer), offset) in (at..).zip(buffers).zip(offsets) {
                    let bound = self.bind_render_buffer(call, buffer, fragment, offset, slot)?;
                    result = result.and(bound);
                }
                result
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
                self.bind_render_texture(call, arg(call, 0)?, fragment, at)?
            }
            "setVertexTextures:withRange:" | "setFragmentTextures:withRange:" => {
                let textures = list(call, 0)?;
                let at = first(call, 1, textures.len())?;
                let mut result = Ok(());
                for (slot, texture) in (at..).zip(textures) {
                    result = result.and(self.bind_render_texture(call, texture, fragment, slot)?);
                }
                result
            }
            "setVertexSamplerState:atIndex:" | "setFragmentSamplerState:atIndex:" => {
                let sampler = self.sampler(call, arg(call, 0)?)?;
                let at = index(call, 1)?;
                self.bind_sampler(id, fragment, at, sampler, None)
            }
            "setVertexSamplerStates:withRange:" | "setFragmentSamplerStates:withRange:" => {
                let samplers = list(call, 0)?.iter();
                let samplers = samplers.map(|value| self.sampler(call, value));
                let samplers = samplers.collect::<Result<Vec<_>, _>>()?;
                let at = first(call, 1, samplers.len())?;
                let mut result = Ok(());
                for (slot, sampler) in (at..).zip(samplers) {
                    result = result.and(self.bind_sampler(id, fragment, slot, sampler, None));
                }
                result
            }
            "setVertexSamplerState:lodMinClamp:lodMaxClamp:atIndex:"
            | "setFragmentSamplerState:lodMinClamp:lodMaxClamp:atIndex:" => {
                let sampler = self.sampler(call, arg(call, 0)?)?;
                let clamps = [clamp(call, arg(call, 1)?)?, clamp(call, arg(call, 2)?)?];
                let at = index(call, 3)?;
                self.bind_sampler(id, fragment, at, sampler, Some(clamps))
            }
            "setVertexSamplerStates:lodMinClamps:lodMaxClamps:withRange:"
            | "setFragmentSamplerStates:lodMinClamps:lodMaxClamps:withRange:" => {
                let samplers = list(call, 0)?.iter();
                let samplers = samplers.map(|value| self.sampler(call, value));
                let samplers = samplers.collect::<Result<Vec<_>, _>>()?;
                let (mins, maxes) = (list(call, 1)?, list(call, 2)?);
                if mins.len() != samplers.len() || maxes.len() != samplers.len() {
                    return Err(refuse(call, "clamps of another count than the samplers"));
                }
                let at = first(call, 3, samplers.len())?;
                let mut result = Ok(());
                for (k, (slot, sampler)) in (at..).zip(samplers).enumerate() {
                    let clamps = [clamp(call, &mins[k])?, clamp(call, &maxes[k])?];
                    result =
                        result.and(self.bind_sampler(id, fragment, slot, sampler, Some(clamps)));
                }
                result
            }
            "setViewport:" => {
                let viewport = viewport(call, arg(call, 0)?)?;
                self.render(id).encoder.set_viewport(viewport)
            }
            "setViewports:count:" => {
                let viewports = (list(call, 0)?.iter())
                    .map(|value| viewport(call, value))
                    .collect::<Result<Vec<_>, _>>()?;
                self.render(id).encoder.set_viewports(&viewports)
            }
            "setScissorRect:" => {
                let rect = scissor(call, arg(call, 0)?)?;
                self.render(id).encoder.set_scissor_rect(rect);
                Ok(())
            }
            "setScissorRects:count:" => {
                let rects = (list(call, 0)?.iter())
                    .map(|value| scissor(call, value))
                    .collect::<Result<Vec<_>, _>>()?;
                self.render(id).encoder.set_scissor_rects(&rects)
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
            "setTessellationFactorBuffer:offset:instanceStride:" => {
                let (offset, stride) = (uint(call, 1)?, uint(call, 2)?);
                let Some(buffer) = self.bound(call, arg(call, 0)?, FACTORS, Self::buffer_of)?
                else {
                    return Ok(());
                };
                (self.render(id).encoder).set_tessellation_factor_buffer(
                    buffer.as_ref(),
                    offset,
                    stride,
                )
            }
            "drawPatches:patchStart:patchCount:patchIndexBuffer:patchIndexBufferOffset:\
             instanceCount:baseInstance:" => {
                let (points, start, count) = (index(call, 0)?, index(call, 1)?, index(call, 2)?);
                let indices = match arg(call, 3)? {
                    Value::Nil => None,
                    _ => Some((Self::id(call, 3)?, uint(call, 4)?)),
                };
                let (instances, base_instance) = (index(call, 5)?, index(call, 6)?);
                if indices.is_some_and(|(buffer, _)| self.skipped(buffer)) {
                    self.skips.draws += 1;
                    return Ok(());
                }
                let indices = match indices {
                    Some((buffer, offset)) => Some((self.buffer_of(call, buffer)?, offset)),
                    None => None,
                };
                if self.draw_skipped(id) {
                    return Ok(());
                }
                if self.render(id).unrecorded_tessellation {
                    self.unrecorded_tessellation.patch_draws += 1;
                }
                (self.render(id).encoder).draw_patches(
                    points,
                    start,
                    count,
                    indices.as_ref().map(|(buffer, offset)| (buffer, *offset)),
                    instances,
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
            "setDepthBias:slopeScale:clamp:" => {
                let float = |at: usize| match arg(call, at)?.as_f64() {
                    Some(value) => Ok(value as f32),
                    None => Err(refuse(call, format!("argument {at} is no float"))),
                };
                let [bias, slope, clamp] = [float(0)?, float(1)?, float(2)?];
                self.render(id).encoder.set_depth_bias(bias, slope, clamp)
            }
            "setColorStoreAction:atIndex:" => {
                let (action, at) = (store_action(call, uint(call, 0)?)?, uint(call, 1)?);
                let at = usize::try_from(at).map_err(|_| refuse(call, "an index past usize"))?;
                self.render(id).encoder.set_color_store_action(action, at)
            }
            "setDepthStoreAction:" => {
                let action = store_action(call, uint(call, 0)?)?;
                self.render(id).encoder.set_depth_store_action(action)
            }
            "setStencilStoreAction:" => {
                let action = store_action(call, uint(call, 0)?)?;
                self.render(id).encoder.set_stencil_store_action(action)
            }
            "setTriangleFillMode:" => {
                let mode = metal(call, uint(call, 0)?)?;
                self.render(id).encoder.set_triangle_fill_mode(mode)
            }
            "setDepthClipMode:" | "setDepthClipModeSPI:" => {
                let mode = metal(call, uint(call, 0)?)?;
                self.render(id).encoder.set_depth_clip_mode(mode)
            }
            // Apple's private line width: kept by the encoder, whose draw of lines at another
            // width than 1 is the runtime's gap, skipped and counted.
            "setLineWidth:" => {
                let width = match arg(call, 0)?.as_f64() {
                    Some(value) => value as f32,
                    None => return Err(refuse(call, "argument 0 is no float")),
                };
                self.render(id).encoder.set_line_width(width);
                Ok(())
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
        let skipped = render.skipped_pipeline
            || !render.skipped.is_empty()
            || !render.skipped_samplers.is_empty();
        if skipped {
            self.skips.draws += 1;
        }
        skipped
    }

    /// Binds `sampler` at `slot` of the stage `fragment` names, with the LOD clamps `clamps` where
    /// the call states them. A bind the runtime answers with [`rt::Gap::LodClamps`] (clamps whose
    /// samples read otherwise than the sampler's own) binds nothing: it is skipped and counted
    /// (`clamped_samplers`), and draws are skipped while the slot holds it.
    fn bind_sampler(
        &mut self,
        id: u64,
        fragment: bool,
        slot: u32,
        sampler: Option<rt::SamplerState>,
        clamps: Option<[f32; 2]>,
    ) -> Result<(), rt::Error> {
        let encoder = &mut self.render(id).encoder;
        let sampler = sampler.as_ref();
        let bound = match (clamps, fragment) {
            (None, false) => encoder.set_vertex_sampler_state(sampler, slot),
            (None, true) => encoder.set_fragment_sampler_state(sampler, slot),
            (Some([min, max]), false) => {
                encoder.set_vertex_sampler_state_with_lod_clamps(sampler, min, max, slot)
            }
            (Some([min, max]), true) => {
                encoder.set_fragment_sampler_state_with_lod_clamps(sampler, min, max, slot)
            }
        };
        match bound {
            Err(rt::Error::Unsupported(rt::Gap::LodClamps)) => {
                self.skips.clamped_samplers += 1;
                self.render(id).skipped_samplers.insert((fragment, slot));
                Ok(())
            }
            Ok(()) => {
                self.render(id).skipped_samplers.remove(&(fragment, slot));
                Ok(())
            }
            failed => failed,
        }
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

/// A LOD clamp argument, a number.
fn clamp(call: &Call, value: &Value) -> Result<f32, PlayError> {
    let clamp = value
        .as_f64()
        .ok_or_else(|| refuse(call, format!("{value} is no clamp")))?;
    Ok(clamp as f32)
}
