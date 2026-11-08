//! A render encoder's kept draw after calls that set the encoder's render state
//! (`RenderLowering::log_restating`: its viewports, scissor rectangles, cull mode, winding,
//! depth/stencil state, stencil references, depth bias, depth clip mode, fill mode or blend
//! color), which keep it, restated (`render/draw_again.rs`): its next draw derives again
//! everything that state decides of it, and goes on rebound (`rebound.rs`) only where none of it
//! changes what the kept draw's every other step worked out.
//!
//! - **Its settings.** Derived again as every step derives them (`RenderLowering::routed_draw`,
//!   through the encoder's shape memo): they must be the kept draw's (by value), with no hidden
//!   depth or stencil given, no strip listed, a provoking vertex the kept draw's, and the fields
//!   the settle takes of the route beside them (primitive IDs' indices, private and provoking
//!   forms) as its bind state holds them, so its settle would find the kept shape: every field of
//!   the bind state but the values below is a function of those and of its tables. A draw whose
//!   depth/stencil state fails every fragment (`DrawSettings::failing_state`) takes every step.
//! - **The bind state's values.** Its viewport count, blend color, alpha test reference, depth
//!   clip mode and stencil references, and its depth/stencil state (`refresh_bind_state`,
//!   `settle_draw`), which no shape reads: where one changed, the draw takes every step if its
//!   variant pushes a field that reads one, else the bind state takes them. Its fragment tests,
//!   which the shape reads, must be the kept draw's.
//! - **Its draw state.** Derived again as every step derives it (`ShapeInputs::draw_state`, the
//!   memo's where it holds one), through the encoder's scissor rectangles: its static part must
//!   key the kept draw's pipeline (`FixedState::without_dynamic`), so the kept `VkPipeline` draws
//!   it; the draw records its dynamic states against what its record holds, comparing each
//!   (`Recording`'s `settled` none).
//! - **What it leaves.** Its depth writes, as its depth/stencil state has them, and, once
//!   recorded, the pixels it may write added to the drawn areas and coverage
//!   (`RenderLowering::add_drawn`), as every step's draw adds them.
//! - **Checked in builds with debug assertions.** As a draw rebound is (`rebound.rs`): its every
//!   step must record with the same draw state too.

use metal2vulkan::bind::DrawCall;
use metal2vulkan::pipeline::PushField;
use std::sync::Arc;

use super::RenderLowering;
use crate::degradation::EachWatch;
use crate::live_bind::LiveBind;
use crate::pipeline::{FixedState, RenderStateKind};
use crate::render::{line_width, provoking, Assembly};

impl RenderLowering {
    /// Derive again what the encoder's render state decides of the kept draw, drawn now as `call`
    /// of `assembly`, `draw`, with the encoder's bind state `live` (the module docs): whether the
    /// draw may go on rebound, its kept recording's draw state then the new one.
    pub(super) fn restate_kept(
        &mut self,
        live: &mut LiveBind,
        call: &'static str,
        assembly: Assembly,
        draw: &DrawCall,
    ) -> bool {
        let Some(RenderStateKind::Graphics(pipeline)) = self.pipeline.as_ref().map(|p| p.kind())
        else {
            return false;
        };
        let pipeline = pipeline.clone();
        if !live.settles_kept(&pipeline, self.forms) {
            return false;
        }
        let Some(variant) = (live.settled.as_ref()).map(|here| Arc::clone(&here.settled().variant))
        else {
            return false;
        };
        let primitives = line_width::draws_primitives(assembly, draw);
        // A route refused is refused again by every step.
        let Ok(routed) = self.routed_draw(call, &pipeline, (assembly, primitives), draw) else {
            return false;
        };
        let Some(kept) = self.again.kept.as_ref() else {
            return false;
        };
        let settings = &routed.settings;
        let state = &live.state;
        let provoking = match routed.provoking {
            provoking::Provoking::AsIs => false,
            provoking::Provoking::Native => true,
            _ => return false,
        };
        let routed_as_kept = (Arc::ptr_eq(settings, &kept.settings)
            || **settings == *kept.settings)
            && settings.failing_state.is_none()
            && routed.hidden.is_none()
            && routed.restarted.is_none()
            && routed.assembly == kept.assembly
            && provoking == kept.last_vertex
            && state.primitive_id_indices == routed.primitive_id_indices
            && state.quad_primitive_ids == routed.private.quad_primitive_ids
            && state.private_list == routed.private.private_list
            && state.rotated_weights == routed.provoking_forms.rotated_weights
            && state.mesh_last_first == routed.provoking_forms.mesh_last_first;
        if !routed_as_kept {
            return false;
        }
        // The values no shape reads (`refresh_bind_state`), and the depth/stencil state, read
        // through the fragment tests alone (`settle_draw`): no extended shadow stores, as the
        // kept draw's bound none.
        let viewport_count = self.state.viewport_count(true);
        let blend_color = self.state.blend_color();
        let alpha_test_reference = self.state.alpha_test_reference();
        let depth_clip_mode_clamp = self.state.depth_clip_mode_clamp();
        let stencil_references = self.state.stencil_reference();
        let depth_stencil = self.state.depth_stencil_descriptor();
        let tests = self.fragment_tests(&pipeline, &depth_stencil, false);
        let state = &mut live.state;
        if tests != state.fragment_tests {
            return false;
        }
        let changed = state.viewport_count != viewport_count
            || state.blend_color != blend_color
            || state.alpha_test_reference != alpha_test_reference
            || state.depth_clip_mode_clamp != depth_clip_mode_clamp
            || state.stencil_references != stencil_references
            || state.depth_stencil != depth_stencil;
        let compiled = variant.pipeline();
        let pushes_values = (compiled.interface.push.iter())
            .flat_map(|placement| &placement.fields)
            .any(|placed| {
                matches!(
                    placed.field,
                    PushField::ViewportCount
                        | PushField::HalfBlendColor
                        | PushField::AlphaTestReference
                        | PushField::ShadowStencilState
                        | PushField::ShaderTestsState
                        | PushField::ShaderStencilState
                )
            });
        if changed && pushes_values {
            return false;
        }
        state.viewport_count = viewport_count;
        state.blend_color = blend_color;
        state.alpha_test_reference = alpha_test_reference;
        state.depth_clip_mode_clamp = depth_clip_mode_clamp;
        state.stencil_references = stencil_references;
        state.depth_stencil = depth_stencil;
        // Its draw state: the memo's for the settings' inputs, through the scissor rectangles,
        // else derived and kept, as every step finds it.
        let graphics = &compiled.graphics;
        let failed = (None, false, false);
        let token = routed.shape_token;
        let shaped = match token.and_then(|token| self.shape_memo.state(token, &variant)) {
            Some(mut kept) => {
                let selects = graphics.writes_viewport_index;
                kept.state.scissors = self.state.vk_scissors(&self.scissors, selects);
                kept
            }
            None => {
                let watch = EachWatch::start();
                let inputs = self.shape_inputs();
                let Ok(shaped) = inputs.draw_state(graphics, settings, failed, &self.scissors)
                else {
                    return false;
                };
                if let (Some(token), false) = (token, watch.met()) {
                    self.shape_memo.keep_state(token, &variant, &shaped);
                }
                shaped
            }
        };
        let Some(recording) = live.kept.recording_mut() else {
            return false;
        };
        let dynamic = &graphics.dynamic_states;
        let keyed = |fixed: FixedState| FixedState::without_dynamic(fixed, dynamic);
        if keyed(shaped.state.fixed) != keyed(recording.state.fixed) {
            return false;
        }
        recording.state = shaped.state;
        let depth_writes = self.state.depth_writes();
        if let Some(kept) = self.again.kept.as_mut() {
            kept.settings = routed.settings;
            kept.depth_writes = depth_writes;
        }
        true
    }
}
