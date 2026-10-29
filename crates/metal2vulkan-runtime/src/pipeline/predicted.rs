//! The `VkPipeline` a render state's first draw is predicted to take, begun inside its creation,
//! where Metal compiles.
//!
//! Metal pays a render state's compile inside `newRenderPipelineState`, outside any command
//! buffer. The runtime made a state's `VkPipeline`s at its draws, so on MoltenVK a cold making of
//! seconds sat inside a command buffer's execution, past a guest's GPU watchdog. A state's
//! creation now begins the making of the `VkPipeline` of the key a first draw takes under Metal's
//! defaults, built from the state alone (where it runs: the end of these docs):
//!
//! - **The variant** the base, where the plan of a draw binding the vertex buffers alone (below)
//!   runs on it (a variant only a draw needs is made by that draw, as before); **the
//!   specialization** that plan's: each constant at the value an unbound slot gives it, and each
//!   buffer slot read through a uniform view read at its residual (its
//!   `SpecKey::RangeAtBindOffset` at 0, the key's residual form, `render/residual.rs`), as an
//!   unbound one is. A draw that leaves those slots unbound takes this key itself; the draws that
//!   bind them do so at offsets the creation cannot know, and the residual form reads the same
//!   bytes at every one: a draw whose own key is not made takes it at once while its own is made.
//!   On a device that reads every view at its residual (MoltenVK) it is their own key. The key
//!   made is the general form of that (`render/residual.rs`): each texture slot's
//!   `SpecKey::PlainHalfTexels` at 0 too, the class routine that converts every bound class, and
//!   its `SpecKey::UnmeasuredTexels` at 0, so a draw of any texture takes it at once while its own
//!   key is made; the key with those slots' `PlainHalfTexels` at 1, as unbound ones take, which a
//!   draw binding a plain, measured class takes, is handed to the pool right after, at the rank
//!   nothing waits for, and nothing waits for it.
//! - **The tests**: for a fragment function that stores a color through the interlocked storage
//!   form before its depth and stencil tests (`RuntimeFact::TestedStorageWrites`) in a pipeline
//!   with a depth or stencil attachment, tests that drop fragments but write neither attachment and
//!   count nothing (`FragmentTests::Drop`), whose plan takes the variant that declares early tests:
//!   that variant is translated and its pipeline made here, as its first draw made them before.
//!   Unstated tests, the default, refuse such a function's plan, so the creation made nothing for
//!   it; a draw stating other tests makes its own, as before.
//! - **The vertex buffers**: the buffer of each layout of the vertex descriptor that
//!   fixed-function input fetches. Every draw that runs binds each of them (a draw without one is
//!   refused, `Bind(UnboundVertexBuffer)`), so the draw predicted binds them, at an offset every
//!   attribute format's alignment divides (offset 0, or inline bytes, which the runtime places so):
//!   a layout of a fixed stride at that stride, which keys nothing; a layout of
//!   `MTLBufferLayoutStrideDynamic`, whose stride only the draw states, at the stride that packs
//!   its attributes (the end of the furthest, rounded up to `MAX_VERTEX_ALIGNMENT`). Where the
//!   pipeline makes strides dynamic state, any stride fixed-function input fetches takes that
//!   key; elsewhere the key holds the stride, and a draw at another makes its own. A draw whose
//!   bind offset or stride fixed-function input cannot fetch runs a variant that pulls the
//!   layout, which that draw makes, as before.
//! - **The textures**: none bound. Nothing a creation sees names a texture a draw binds, and an
//!   unbound slot is Metal's default (no format is chosen from what earlier sessions bound).
//! - **The samplers**: at each runtime sampler slot that does not compare, Metal's default sampler
//!   (`MTLSamplerDescriptor`'s defaults). Where the device has no Metal sampler, the function's
//!   specialization reads the bound sampler's word.
//!
//!   Where that shape's plan needs another variant or refuses it, the plan of the draw binding
//!   the vertex buffers alone is predicted, as before.
//! - **The pass** of the descriptor's attachments: each color's and the depth and stencil's
//!   format as an attachment of its pixel format, its raster sample count, no resolve, and the
//!   colors the pipeline reads through storage left out of the subpass (`framebuffer_fetch.rs`).
//! - **The input assembly** of the descriptor's topology class (a list; a triangle list when
//!   unspecified), and **the fixed state** of Metal's default encoder state.
//!
//! A state the prediction cannot describe from itself alone (tessellated, a mesh pipeline,
//! separate depth and stencil formats, a fragment function that writes depth) makes nothing here.
//! The key is the draw's own key type: a draw that takes the same key takes the pipeline made, and
//! so does a draw whose key differs only in constants the general form sets (a uniform-view slot
//! read at the bind offset alone, a texture slot's plain `half` arm or its unmeasured class), until
//! its own key's pipeline, made on the pool meanwhile, is made (`render/residual.rs`); any other
//! draw makes its own, as before. Nothing an answer depends on changes, only when the
//! driver is called. A failed making is not kept and refuses nothing: the draw that needs the key
//! meets it as before.
//!
//! The pipeline's specialized notices wait for the first draw that takes it, which announces
//! them naming what it binds (`RenderVariant::first_use`): a pipeline made here and never drawn
//! announces nothing.
//!
//! **Nothing kept from earlier processes** (the owner's ruling of 2026-10-10). A creation makes the
//! predicted key alone, from the request in hand. The keys its pipeline's earlier draws took, kept
//! on disk, the variants they needed, a record's completeness and a first boot's alternative
//! samplers are removed: each chose what to make from what earlier sessions did, not from the
//! state. A draw whose key the creation did not make makes its own, as before's records of the
//! driver's data still serve that making where the same pipeline was made before.
//!
//! Under `PipelineWaitPolicy::Exact` the making of the predicted key's general form runs on the
//! creating thread, joined before the creation returns (`RenderVariant::make_at_creation`), at the
//! rank of a making the program waits for. Under `PipelineWaitPolicy::Bounded` (the default) it is
//! handed to the device's pool, kept by the state as a draw's making is, and the creation returns:
//! a host that creates a state inside a command buffer's encoding (a guest's state looked up there
//! first) is not held by a driver's cold compile, measured at 11.6 s on MoltenVK with a cold
//! Metal compiler service. The first draw takes the making in flight, and its command buffer
//! waits a bound for it (`render/pending.rs`). A key predicted on the early-tests variant (above)
//! is made there, on that variant.

use super::graphics::{FixedState, GraphicsKey};
use super::render::{GraphicsState, RenderVariant, StateDescriptor};
use super::BufferForms;
use crate::color_list::ColorList;
use crate::fetch_reach::format_bytes;
use crate::framebuffer_fetch;
use crate::render_state::EncoderState;
use crate::render_target::{self, PassShape};
use ash::vk;
use metal2vulkan::bind::{
    BindShape, BindingPlan, BoundSampler, BufferShape, FragmentTests, SamplerDescriptor,
};
use metal2vulkan::device::FormatUsage;
use metal2vulkan::metal::vertex::MAX_VERTEX_ALIGNMENT;
use metal2vulkan::metal::{PixelFormat, PrimitiveTopologyClass, PrimitiveType};
use metal2vulkan::pipeline::{
    RenderPipeline, RenderPipelineDescriptor, RuntimeFact, SlotInterface, Stage, VertexStride,
};
use std::sync::Arc;

impl GraphicsState {
    /// Make the `VkPipeline` of the key the state's first draw is predicted to take (the module
    /// docs), now, in its general form, which a draw of every binding can take
    /// (`render/residual.rs`); where the key differs from that form, hand the key itself to the
    /// device's pool right after, at the rank nothing waits for, for the draws that bind what the
    /// prediction names. A key whose pass is refused is left to its draw.
    pub(crate) fn make_at_creation(&self) {
        let Some((variant, key)) = self.predicted() else {
            return;
        };
        let device = self.device();
        let Ok(pass) = render_target::compatible_pass(device, &key.pass) else {
            return;
        };
        let (made, ahead) = match variant.general_key(&key) {
            None => (key, Vec::new()),
            Some(general) => match render_target::compatible_pass(device, &key.pass) {
                Ok(own) => (general, vec![(Arc::clone(&variant), key, own)]),
                Err(_) => (general, Vec::new()),
            },
        };
        RenderVariant::make_at_creation(vec![(variant, made, pass)], ahead, self);
    }

    /// The key of the predicted first draw (the module docs), where the state alone describes
    /// it, with the variant it runs on.
    fn predicted(&self) -> Option<(Arc<RenderVariant>, GraphicsKey)> {
        let StateDescriptor::Vertex(descriptor) = self.descriptor() else {
            return None;
        };
        let (variant, plan) = self.predicted_plan(descriptor)?;
        let key = self.predicted_key(descriptor, &variant, &plan)?;
        Some((variant, key))
    }

    /// The plan of the predicted first draw: [`with_default_samplers`]'s, where it plans; else
    /// that of the draw binding the vertex layouts alone ([`bound_layouts`]).
    fn predicted_plan(
        &self,
        descriptor: &RenderPipelineDescriptor,
    ) -> Option<(Arc<RenderVariant>, Arc<BindingPlan>)> {
        // A post-tessellation vertex function's pipeline tessellates where the descriptor's
        // tessellation state is `None` (Metal's defaults) too.
        if crate::tessellation::patches(&self.base().graphics).is_some() {
            return None;
        }
        let predicting = self.predicting_variant()?;
        let tests = predicted_tests(predicting.pipeline());
        let planned = |mut shape: BindShape| {
            shape.fragment_tests = tests;
            match tests {
                FragmentTests::Drop => self.settle(&shape, BufferForms::Bound).ok(),
                _ => self.base_plan(&shape),
            }
        };
        let layouts = bound_layouts(predicting.pipeline(), descriptor);
        let bare = planned(layouts.clone())?;
        Some(
            with_default_samplers(bare.0.pipeline(), &layouts)
                .and_then(planned)
                .unwrap_or(bare),
        )
    }

    /// The key of a predicted draw planned as `plan` on `variant` (the module docs), where the
    /// state alone describes it.
    fn predicted_key(
        &self,
        descriptor: &RenderPipelineDescriptor,
        variant: &Arc<RenderVariant>,
        plan: &BindingPlan,
    ) -> Option<GraphicsKey> {
        let pipeline = variant.pipeline();
        let graphics = &pipeline.graphics;
        if graphics.writes_depth {
            return None;
        }
        let profile = self.device().compile_device().profile();
        let attachment = |pf: PixelFormat, usage: FormatUsage| {
            metal2vulkan::metal::format::vk_format(pf, &[usage].into(), profile)
                .ok()
                .map(|mapping| mapping.format)
        };
        let mut colors = Vec::with_capacity(descriptor.color_attachments.len());
        for color in &descriptor.color_attachments {
            colors.push(match color {
                Some(c) if c.pixel_format != PixelFormat::Invalid => {
                    Some(attachment(c.pixel_format, FormatUsage::ColorAttachment)?)
                }
                _ => None,
            });
        }
        while colors.last().is_some_and(Option::is_none) {
            colors.pop();
        }
        let given = |pf: Option<PixelFormat>| pf.filter(|&pf| pf != PixelFormat::Invalid);
        let depth_stencil = match (
            given(descriptor.depth_format),
            given(descriptor.stencil_format),
        ) {
            (None, None) => None,
            (Some(pf), None) | (None, Some(pf)) => Some(pf),
            (Some(depth), Some(stencil)) if depth == stencil => Some(depth),
            (Some(_), Some(_)) => return None,
        };
        let depth_stencil = match depth_stencil {
            Some(pf) => Some(attachment(pf, FormatUsage::DepthStencilAttachment)?),
            None => None,
        };
        let fetch_inputs =
            colors.iter().any(Option::is_some) && framebuffer_fetch::through_inputs(profile);
        let mut storage = Vec::new();
        if !fetch_inputs {
            for s in &graphics.storage_attachments {
                let present = colors.get(s.color_index as usize).copied().flatten();
                if present != Some(s.format) {
                    return None;
                }
                storage.push(s.color_index);
            }
            storage.sort_unstable();
            storage.dedup();
        }
        let pass = PassShape {
            color_resolves: ColorList::of(colors.iter().map(|_| None)).ok()?,
            colors: ColorList::of(colors).ok()?,
            depth_stencil,
            samples: descriptor.raster_sample_count,
            depth_stencil_resolve: None,
            fetch_inputs,
            no_inputs: ColorList::new(),
            storage: ColorList::of(storage).ok()?,
            view_mask: 0,
            locations: ColorList::new(),
        };
        let primitive = match descriptor.input_primitive_topology {
            PrimitiveTopologyClass::Point => PrimitiveType::Point,
            PrimitiveTopologyClass::Line => PrimitiveType::Line,
            // Unspecified is Metal's default and lets a draw take any primitive, so the class
            // says nothing and this guess decides whether the prediction is used at all: a draw
            // of another topology makes its own `VkPipeline`, and the one made here is never
            // taken. A state that **discards rasterization** runs its vertex stage for its side
            // effects alone -- Metal refuses such a state whose vertex function returns a value
            // (`unrasterized.rs`), so it writes through buffers and nothing else -- and a draw
            // like that takes a point list, which shades every vertex of any count and assembles
            // none of them away. Measured over this suite: of the five case families that make an
            // unrasterized state, the four that draw at all draw points, and every triangle draw
            // belongs to a rasterizing state. A rasterizing state keeps the triangle guess, which
            // is what its draws take.
            PrimitiveTopologyClass::Unspecified if graphics.rasterizer_discard => {
                PrimitiveType::Point
            }
            PrimitiveTopologyClass::Triangle | PrimitiveTopologyClass::Unspecified => {
                PrimitiveType::Triangle
            }
            // A class this build does not know: no prediction.
            _ => return None,
        };
        let (topology, primitive_restart) = metal2vulkan::metal::state::input_assembly(primitive);
        let dynamic: Vec<vk::DynamicState> = graphics
            .dynamic_states
            .iter()
            .map(|&state| state.into())
            .collect();
        // A layout of a fixed stride keys nothing; one of `MTLBufferLayoutStrideDynamic` keys the
        // stride its draw binds, the packed one predicted ([`bound_layouts`]), unless the
        // pipeline makes strides dynamic state; in binding order, as a draw lists them.
        let strides = match dynamic.contains(&vk::DynamicState::VERTEX_INPUT_BINDING_STRIDE) {
            true => Vec::new(),
            false => pipeline
                .vertex_input
                .bindings
                .iter()
                .filter(|b| b.stride == VertexStride::PerBind)
                .map(|b| (b.binding, packed_stride(descriptor, b.buffer_index)))
                .collect(),
        };
        let state = EncoderState::metal_defaults([1, 1]);
        let mut fixed = state
            .fixed(
                state.viewport_count(graphics.writes_viewport_index),
                profile,
            )
            .ok()?;
        // A shader-side stencil's draws test the stencil in the function (`draw_fixed`).
        if graphics.shadow_stencil.is_some() {
            fixed.depth_stencil.stencil_test_enable = false;
        }
        // Each uniform-view slot read at its residual: the residual form, which a draw binding the
        // slot at any offset can take (`render/residual.rs`).
        let specialization = variant
            .residual_form(&plan.specialization)
            .unwrap_or_else(|| plan.specialization.clone());
        let key = GraphicsKey {
            specialization,
            topology,
            primitive_restart,
            pass,
            strides,
            fixed: FixedState::without_dynamic(fixed, &graphics.dynamic_states),
            sample_positions: Vec::new(),
        };
        Some(key)
    }
}

/// The depth and stencil tests a first draw of `pipeline` is predicted to state
/// (`BindShape::fragment_tests`): where its fragment function stores a color through the
/// interlocked storage form before the tests (`RuntimeFact::TestedStorageWrites`) in a pipeline
/// with a depth or stencil attachment, tests that drop fragments but write neither attachment
/// and count nothing ([`FragmentTests::Drop`]), whose plan asks for the variant that declares
/// early tests; otherwise the default, which such a pipeline's plan refuses and every other's
/// does not read. A draw of such a pipeline states its tests (Metal's default, unstated, is
/// refused). On the consumer's boots, the states whose creation made no pipeline were such
/// states, and their draws stated `Drop` (166 of 174 states on one boot, the other 8 `Pass`; 98
/// of 98 on another): predicting the default made nothing for them, so each first draw made its
/// variant and its pipeline.
///
/// A function whose stores before the tests have no early-tests variant (it writes other memory,
/// or returns depth, stencil, or a sample mask) runs only draws whose tests drop no fragment
/// ([`FragmentTests::Pass`]), so that is predicted for it.
fn predicted_tests(pipeline: &RenderPipeline) -> FragmentTests {
    let early_variant = pipeline
        .dependencies
        .iter()
        .find_map(|dependency| match dependency.fact {
            RuntimeFact::TestedStorageWrites { early_variant, .. } => Some(early_variant),
            _ => None,
        });
    let graphics = &pipeline.graphics;
    let tested = graphics.depth_format.is_some() || graphics.stencil_format.is_some();
    match early_variant {
        Some(true) if tested => FragmentTests::Drop,
        Some(false) if tested => FragmentTests::Pass,
        _ => FragmentTests::default(),
    }
}

/// The bind shape of a first draw of `pipeline` a creation predicts: `layouts` (the vertex
/// layouts, [`bound_layouts`]) with Metal's default sampler (`MTLSamplerDescriptor`'s defaults,
/// the word 0) at each runtime sampler slot that does not compare, and nothing else bound. `None`
/// where there is no such slot (the draw binding `layouts` alone is then the prediction).
fn with_default_samplers(pipeline: &RenderPipeline, layouts: &BindShape) -> Option<BindShape> {
    let default = BoundSampler::new(SamplerDescriptor::new()).shape();
    let mut shape = layouts.clone();
    for (slot, used) in &pipeline.interface.slots {
        if let SlotInterface::Sampler(sampler) = used {
            if !sampler.compare {
                let stage = shape.stages.entry(slot.stage).or_default();
                stage.samplers.insert(slot.index, default);
            }
        }
    }
    (shape != *layouts).then_some(shape)
}

/// The bind shape of a draw of `pipeline` (of `descriptor`) that binds what every draw of it binds
/// and nothing else: the buffer of each layout fixed-function vertex input fetches, as the module
/// docs bind it. A draw that binds none of them is refused (`Bind(UnboundVertexBuffer)`), so no
/// plan of a draw binding less is any draw's; a layout the pipeline pulls is a buffer slot of its
/// vertex function, unbound here as every other is.
fn bound_layouts(pipeline: &RenderPipeline, descriptor: &RenderPipelineDescriptor) -> BindShape {
    let mut shape = BindShape::default();
    for binding in &pipeline.vertex_input.bindings {
        let attribute_stride = match binding.stride {
            VertexStride::Static(_) => None,
            VertexStride::PerBind => Some(packed_stride(descriptor, binding.buffer_index)),
        };
        shape
            .stages
            .entry(Stage::Vertex)
            .or_default()
            .buffers
            .insert(
                binding.buffer_index,
                BufferShape::Bytes { attribute_stride },
            );
    }
    shape
}

/// The stride a predicted draw binds the buffer of `descriptor`'s layout at Metal buffer index
/// `buffer` at, a layout of `MTLBufferLayoutStrideDynamic`: the one that packs the layout's
/// attributes, the end of the furthest (its offset and its format's size), rounded up to
/// `MAX_VERTEX_ALIGNMENT`, which every attribute format's alignment divides.
fn packed_stride(descriptor: &RenderPipelineDescriptor, buffer: u32) -> u32 {
    let end = descriptor
        .vertex_descriptor
        .iter()
        .flat_map(|vertex| vertex.attributes.values())
        .filter(|attribute| attribute.buffer_index == buffer)
        .map(|attribute| {
            attribute
                .offset
                .saturating_add(format_bytes(attribute.format))
        })
        .max()
        .unwrap_or(0);
    end.checked_next_multiple_of(MAX_VERTEX_ALIGNMENT)
        .unwrap_or(end)
}
