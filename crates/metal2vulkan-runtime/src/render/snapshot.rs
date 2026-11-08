//! A draw that reads a texture its own render pass instance renders, after a memory barrier,
//! read through a snapshot inside the instance: two draws of one pipeline in the same subpass,
//! in place of a split of the pass (`texture/attached_reads.rs`).
//!
//! - **What it replaces.** After a barrier inside an instance, a draw that samples a color
//!   attachment's texture records into a new instance, before which the texture's copy is taken
//!   (`WorkCounts::attachment_read_splits`). The split ends the render pass instance, copies the
//!   texture and begins another, on every device. Here the draw stays in its instance.
//! - **The two draws.** The draw runs the variant whose fragment function reads each such slot
//!   through its snapshot (`BindState::snapshot_textures`, `VariantKey::snapshot_textures`), a
//!   pipeline made with `VK_PIPELINE_CREATE_COLOR_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT` and the color
//!   write masks as dynamic state (`GraphicsState::snapshot_reads`). Each of its Vulkan draws
//!   records twice: first the snapshot draw, every color write mask 0 and the phase 1
//!   (`PushHole::SnapshotPhase`), whose fragments copy the texture's texel at their pixel into the
//!   snapshot inside the pixel interlock and return; then the drawing draw, the pipeline's own
//!   masks and the phase 0, whose fragments read the snapshot where the function read the
//!   texture. Every snapshot draw of the Metal draw records before every drawing draw of it, so a
//!   draw split into several reads what the texture held before it, as Metal's one draw does.
//!   One pipeline, not two: "Changes to the following state values have no side effects (the use
//!   of any other state value is not affected by the change)", among them "Write masks (color,
//!   depth, stencil)", and "Fragment generation is invariant with respect to the state values
//!   listed in Rule 2" (Invariance, Rule 2 and Corollary 1): the drawing draw's fragments are the
//!   snapshot draw's, and "Otherwise, at least one fragment shader must be executed" (Fragment
//!   Shading) for each, so every pixel the drawing draw reads was taken. The snapshot draw's
//!   fragments write no color ("Otherwise, the value in memory is unmodified", the color write
//!   mask), and the drawing draw's fetch of the texture never runs: "Data races can only occur
//!   due to instructions that are actually executed" (the memory model). The snapshot draw keeps
//!   the draw's depth/stencil state, so a draw that writes depth or stencil would write them in
//!   the snapshot draw first: a pass with a depth or stencil attachment takes the split, as the
//!   binder makes no snapshot variant for it (`metal2vulkan::bind`'s snapshot reads; lane case
//!   `snapshot_reads::a_reader_in_a_pass_with_depth_or_stencil_takes_the_split`).
//! - **The ordering, in the subpass.** The texture's color attachment is in
//!   `VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT` for the instance
//!   (`render_target::Feedback`), so "A vkCmdPipelineBarrier inside a subpass can synchronize prior
//!   attachment writes in the subpass with subsequent non-attachment reads if feedback loop is
//!   enabled for the attachment", and then "the overlap is defined to occur at a per-pixel
//!   granularity" (the Render Pass chapter). Before the snapshot draws, `B1`: from the color
//!   writes of the draws before (and the load, as "Load operations for attachments with a color
//!   format execute in the VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT pipeline stage") and the
//!   fragment shader's writes and reads of the snapshot by an earlier pair, to the fragment
//!   shader's reads and writes: the texture's sampled read and the snapshot's write. Between the
//!   snapshot draws and the drawing draws, `B2`: from the fragment shader's writes of the snapshot
//!   to its reads and writes of it (the drawing draw's function holds the write, never run), and
//!   to the color output, whose writes of the texture come after the
//!   snapshot draw's reads of it ("Write-after-read hazards can be solved with just an execution
//!   dependency", Execution and Memory Dependencies). Both are by region, as every barrier inside
//!   a subpass whose source has framebuffer-space stages must be
//!   (`VUID-vkCmdPipelineBarrier-dependencyFlags-07891`); "each region in that dependency covers a
//!   single (x,y,layer) position" (Framebuffer Region Dependencies), and each draw reads and writes
//!   the snapshot and the texture at its fragment's own pixel alone. Each lies within the pass's
//!   dependency of its subpass on itself (`render_target::DRAWS_SELF_DEPENDENCY`), as
//!   `VUID-vkCmdPipelineBarrier-None-07889` requires, and names framebuffer-space stages alone
//!   (`-07890`).
//! - **Pair per reader.** Every such draw takes its own pair, its `B1` ordering every draw
//!   before it: Metal's barrier over the render targets orders the draws after it behind those
//!   before it on a Mac2 GPU (`fence.rs`), which the copy after a split gave, and a reader after
//!   another reader sees what that one wrote. (An M3 Max reads, at a draw's own pixel, what the
//!   texture held before the pass with a barrier or without: probe `readers.m`, two readers of a
//!   texture a draw wrote, barriers before each or not, 64 to 2048 pixels wide, 0 to 500 draws
//!   between. Apple GPUs do not support that scope, so they are no answer for it.)
//! - **The snapshot** is a texture's storage image of its level-0 extent in the form the plan
//!   names for the slot (`VariantKey::snapshot_textures`, `SnapshotForm::of` the bound view: the
//!   texel's codes, whatever the device's decode, or the fetched lanes' bits), made at the first
//!   draw that takes it in that form and kept with the texture (`Texture::snapshot_texels`),
//!   bound in `GENERAL` (`WriteSource::SnapshotTexels`): 4 bytes a pixel for an 8-bit
//!   normalized or integer format's codes (about 8.3 MB for a 1920x1080 texture), 8 for a 16-bit
//!   one's or a half-float one's, 16 for the bits (`SnapshotForm::format`). The draw's use of it
//!   is a fragment shader's storage read and write, so draws of other instances and encoders
//!   that take it are ordered as any image's.
//! - **Where it runs** (each check else the split, as before): the device enabled
//!   `attachmentFeedbackLoopLayout`, `extendedDynamicState3ColorWriteMask` and
//!   `fragmentShaderPixelInterlock` (`dynamic_state::ColorWriteMasks`); the draw follows a barrier
//!   inside its instance, records no execution's command or reroute, and runs in a pass
//!   with no visibility query active, no input attachment fetch, rasterization-rate map, views or
//!   more than one sample or layer; every slot of the draw that reads a color attachment's texture
//!   is a fragment slot its function can read through a snapshot (`StageModule::snapshot_textures`)
//!   of a texture the attachment renders in place, at the binding's own first level and slice,
//!   with feedback loop usage (`Texture::snapshot_color`, `texture::feedback_candidate`), not
//!   through a view of another format, and which no later fetch can render through a stand-in
//!   (`RenderCommandEncoder::keeps_own_attachment`: its image has `STORAGE`, which a texture of
//!   the runtime's own without it, its device laying it out otherwise with it, takes at a later
//!   commit where it may, noted at its reader's draw, `texture/storage_form.rs`) or which the draw
//!   pins (below), no two of the colors of one image (whose
//!   snapshot is one); the draw costs no more so than the split would
//!   (`RenderCommandEncoder::snapshot_cheaper`: at any area where every such color is a texture
//!   whose image lies over its buffer's bytes, `RenderCommandEncoder::over_bytes`; else its
//!   scissors' area times [`SNAPSHOT_PIXEL_COST`] for each color against each color's level
//!   times its texel bytes, whatever the snapshot's form, so a 4-byte color's reader,
//!   `RGBA8Unorm`'s or `BGRA8Unorm`'s, takes the snapshot up to a fifth of the render area, an
//!   `RGBA16Float` one's up to two fifths and an `RGBA32Float` one's up to four fifths); the binder keeps every such slot (in a pass with no depth or stencil attachment, on a device that also enabled
//!   `fragmentStoresAndAtomics`) and the variant is made; no slot takes the snapshot slot as
//!   storage. The draw's instance holds those colors in the feedback loop layout from its render
//!   pass instance's begin, which records in a secondary: where the current instance
//!   cannot (`RenderCommandEncoder::snapshot_apart`: it records inline, `render/inline_first.rs`
//!   and `render/inline_splits.rs`, its render pass instance begun already; it ran an execution
//!   of render commands; a draw of it bound such a color's image as a storage image; or it
//!   leaves the color out), the draw records into a new instance, with no copy
//!   (`WorkCounts::snapshot_inline_splits` counts the first cause). After it, an execution of
//!   render commands, and a draw that binds such a color's image as a storage image, record into
//!   another instance: the feedback loop layout serves neither.
//! - **Pins.** A fetch of a color in the pixel interlock reads and writes its image as a storage
//!   image, so a later fetch of a color whose image lacks `STORAGE` renders it through a stand-in
//!   (`attachment_stand_in.rs`), which the pass's end puts in place for the whole pass, copied in
//!   before it: a snapshot draw before the fetch would have read the texture's own image, which
//!   the pass no longer rendered. A draw reads such a color through a snapshot all the same where
//!   it can pin it (`RenderCommandEncoder::pins_own_attachment`): where nothing the pass's end
//!   decides from the whole pass changes what its instances so far render, and nothing arrives
//!   with the draw (the pass's first instance recorded inline, which ran its load actions itself,
//!   `render/inline_first.rs`; no stand-in, memoryless color or depth/stencil attachment, every
//!   color stored `Store`, no tile dispatch, execution, color groups or extended-range shadow; a
//!   draw that fetches nothing and needs no hidden depth or stencil). Its instance and those
//!   before it render each color's own image (`RenderLowering::pinned_through`), and a stand-in
//!   arriving after them takes its attachment's texels after the last of them, as after an
//!   instance recorded inline: the pass's end renders those instances through the colors' own
//!   images, records each stand-in's copy in after them, the keep of one merged back into a
//!   buffer's bytes after that (`linear_merge.rs`), and the reloads after a barrier after them
//!   alone, and no run of passes takes the pass (`pass_runs.rs`). Once a stand-in or the hidden
//!   depth and stencil arrived after a pin, no later draw pins (`RenderLowering::pins_frozen`):
//!   its reader splits.
//! - **Counted**: `WorkCounts::reading_draws_snapshotted`, and of them
//!   `WorkCounts::snapshot_inline_splits` and `WorkCounts::reading_draws_pinned`; the passes whose
//!   stand-ins arrived after a pin, `WorkCounts::stand_ins_after_pins`. With `METAL2VULKAN_LOG_PIPELINES` set,
//!   the first draw of each reason a draw took the split says so, once per device
//!   (`metal2vulkan-snapshot-fallback`), and the draws the area rule weighed are counted by their
//!   share of their colors' pixels and texel bytes, a line at 64 of them, each power of two
//!   after, and as the device ends (`metal2vulkan-snapshot-readers`, `reader_sizes.rs`).

use super::{inline_first, PlanFacts, Reformat, RenderLowering};
use crate::copy_plan::block_bytes;
use crate::dynamic_state::ColorWriteMasks;
use crate::pipeline::{GraphicsState as DrawingState, Settled};
use crate::render_state::MAX_VIEWPORTS;
use crate::render_target::{self, Feedback};
use crate::render_values::StoreAction;
use crate::texture::Texture;
use ash::vk;
use metal2vulkan::bind::{BindingPlan, ImageSource, PushHole, WriteSource};
use metal2vulkan::pipeline::{SlotClass, SlotRef, SnapshotForm, Stage};
use std::collections::BTreeSet;

/// What a draw asks to read through snapshots, before its plan
/// ([`RenderLowering::snapshot_wanted`]): the fragment slots, and the colors they read.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct Wanted {
    pub(super) slots: BTreeSet<u32>,
    pub(super) feedback: Feedback,
    /// The colors among them that no later fetch would keep in their own images, which the
    /// draw pins to them ([`RenderLowering::pins_own_attachment`]).
    pub(super) pins: u32,
}

/// The form `plan` keeps fragment texture slot `slot`'s snapshot in
/// (`VariantKey::snapshot_textures`), where it reads the slot through one.
pub(super) fn form(plan: &BindingPlan, slot: u32) -> Option<SnapshotForm> {
    let slot = SlotRef {
        stage: Stage::Fragment,
        class: SlotClass::Texture,
        index: slot,
    };
    plan.variant.snapshot_textures.get(&slot).copied()
}

/// What the snapshot draw costs at each pixel a draw's scissors bound, for each color it
/// snapshots, in bytes of the split's copy (`RenderCommandEncoder::snapshot_cheaper`): one more
/// fragment invocation at the pixel, inside the pixel interlock, that reads the texel and stores
/// its snapshot, and the drawing draw's read of it. The invocation's cost, not the snapshot's
/// bytes: a snapshot of a color's codes, 4 bytes a pixel, costs about what one of the fetched
/// lanes' bits, 16, does.
const SNAPSHOT_PIXEL_COST: u64 = 20;

/// Which of a draw's two Vulkan draws one recording is (the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Phase {
    /// The snapshot draw: every color write mask 0, the phase 1.
    Snapshot,
    /// The drawing draw: the pipeline's own masks, the phase 0.
    Drawing,
}

/// One recording's part in a draw read through snapshots (the module docs).
#[derive(Clone)]
pub(super) struct Pair {
    pub(super) phase: Phase,
    /// The first recording of its phase, which records the barrier before it: `B1` before the
    /// snapshot draws, `B2` before the drawing draws.
    pub(super) first: bool,
    /// The phase's push constant: its stages and offset (`PushHole::SnapshotPhase`).
    pub(super) hole: (vk::ShaderStageFlags, u32),
    pub(super) masks: ColorWriteMasks,
    /// The pipeline's own write mask of each color blend element
    /// (`pipeline::color_write_masks`), which the drawing draw sets.
    pub(super) own: Vec<vk::ColorComponentFlags>,
}

impl Pair {
    /// Record, before the draw of a recording whose pipeline `layout` is bound with its push
    /// constants and dynamic states (all but the write masks), the pair's steps: the barrier where
    /// it is its phase's first, the write masks, and the phase.
    ///
    /// # Safety
    ///
    /// `command_buffer` is recording inside a render pass instance of a pass `render_target`
    /// made, of no views, whose subpass the draw's pipeline was made for; the pipeline, made with
    /// the write masks dynamic, is bound; only this thread records it.
    pub(super) unsafe fn record(
        &self,
        raw: &ash::Device,
        command_buffer: vk::CommandBuffer,
        layout: vk::PipelineLayout,
    ) {
        let flags = render_target::self_dependency_flags(0);
        let fragment = vk::PipelineStageFlags::FRAGMENT_SHADER;
        let (src, dst, masks, phase): (_, _, Vec<vk::ColorComponentFlags>, u32) = match self.phase {
            // `B1`: the color writes before (the earlier draws' and the load's) and an earlier
            // pair's writes of the snapshot, before the snapshot draw's sampled read of the
            // texture and its write of the snapshot; an earlier pair's reads of the snapshot
            // before that write by the execution dependency alone.
            Phase::Snapshot => (
                (
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT | fragment,
                    vk::AccessFlags::COLOR_ATTACHMENT_WRITE | vk::AccessFlags::SHADER_WRITE,
                ),
                (
                    fragment,
                    vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                ),
                vec![vk::ColorComponentFlags::empty(); self.own.len()],
                1,
            ),
            // `B2`: the snapshot's writes before the drawing draw's reads of it, and the snapshot
            // draw's reads of the texture before the drawing draw's color writes of it by the
            // execution dependency into the color output. Its writes too: the drawing draw's
            // function holds the snapshot's write, which it never runs, and a write the pipeline
            // holds is ordered as if it ran (the validation layer's synchronization checks count
            // it so).
            Phase::Drawing => (
                (fragment, vk::AccessFlags::SHADER_WRITE),
                (
                    fragment | vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                ),
                self.own.clone(),
                0,
            ),
        };
        let (hole_stages, offset) = self.hole;
        // SAFETY: the caller's contract; the barrier's scopes lie within the pass's dependency
        // of its subpass on itself, by region as it is (the module docs), and the masks are one
        // per color blend element of the bound pipeline, which makes them dynamic; the push
        // constant lies in the range of the hole's stages.
        unsafe {
            if self.first {
                let barrier = vk::MemoryBarrier::default()
                    .src_access_mask(src.1)
                    .dst_access_mask(dst.1);
                raw.cmd_pipeline_barrier(command_buffer, src.0, dst.0, flags, &[barrier], &[], &[]);
            }
            (self.masks.0)(command_buffer, 0, masks.len() as u32, masks.as_ptr());
            raw.cmd_push_constants(
                command_buffer,
                layout,
                hole_stages,
                offset,
                &phase.to_ne_bytes(),
            );
        }
    }
}

/// The snapshot phase's hole of a draw's push holes, where its plan has one.
pub(super) fn phase_hole(holes: &[PushHole]) -> Option<(vk::ShaderStageFlags, u32)> {
    holes.iter().find_map(|hole| match *hole {
        PushHole::SnapshotPhase { stages, offset } => Some((stages.into(), offset)),
        _ => None,
    })
}

impl RenderLowering {
    /// What a draw of `pipeline` (its state's base) through `reformats` asks to read through
    /// snapshots, where it can ask (the module docs' checks that come before the plan); `None`
    /// where it reads no attachment so or takes the split.
    pub(super) fn snapshot_wanted(
        &self,
        pipeline: &DrawingState,
        reformats: &[Reformat],
    ) -> Option<Wanted> {
        // After a barrier before the pass's first draw alone (`loads_ordered`), a slot reads
        // through a snapshot only where its texture's color is one the pass clears: one the pass
        // loads holds the texels from before the pass, which its copy holds
        // (`texture/attached_reads.rs`).
        let ordered = self.current.barriers || self.barrier;
        if self.replaying || self.rerouting || !(ordered || self.loads_ordered) {
            return None;
        }
        let base = pipeline.base();
        let fragment = base
            .stages
            .iter()
            .find(|m| m.stage == metal2vulkan::vk::ShaderStage::FRAGMENT)?;
        let mut wanted = Wanted::default();
        for &slot in &fragment.snapshot_textures {
            let Some(texture) = self.fragment.textures.get(&slot) else {
                continue;
            };
            if !texture.attached_in(&self.layout) || !(ordered || texture.cleared_in(&self.layout))
            {
                continue;
            }
            let Some(index) = texture.snapshot_color(&self.layout) else {
                return self.snapshot_fallback("the texture is not read where its attachment renders it in place, or its image lacks feedback loop usage");
            };
            wanted.slots.insert(slot);
            wanted.feedback.0 |= 1 << index;
        }
        if wanted.slots.is_empty() {
            return None;
        }
        let device = self.core.device();
        if device.shared().color_write_masks.is_none() {
            return self.snapshot_fallback("the device lacks attachmentFeedbackLoopLayout, extendedDynamicState3ColorWriteMask or fragmentShaderPixelInterlock");
        }
        if self.visibility.active() {
            return self.snapshot_fallback("a visibility query is active");
        }
        if self.layout.fetch_inputs
            || self.rate_map.is_some()
            || self.layout.samples != 1
            || self.layout.layers != 1
        {
            return self.snapshot_fallback(
                "the pass fetches input attachments, or has a rasterization-rate map, samples or layers",
            );
        }
        if reformats.iter().any(|r| wanted.feedback.has(r.index)) {
            return self.snapshot_fallback(
                "the draw renders the attachment through a view of another format",
            );
        }
        let colors: Vec<usize> = (0..self.layout.colors.len())
            .filter(|&i| wanted.feedback.has(i))
            .collect();
        for &index in &colors {
            if self.keeps_own_attachment(index) {
                continue;
            }
            // A texture of the runtime's own whose image lacks `STORAGE` (the device lays it
            // out otherwise with it) takes its storage form at a later commit, where it may
            // (`texture/storage_form.rs`): its later passes' readers need no pin.
            if let Some(color) = self.layout.colors.get(index).and_then(Option::as_ref) {
                color.target.texture.want_storage_form();
            }
            if !self.pins_own_attachment(pipeline, index) {
                return self.snapshot_fallback(
                    "a framebuffer fetch of the attachment would render it through a stand-in",
                );
            }
            wanted.pins |= 1 << index;
        }
        // A texture's snapshot is its image's (`Texture::snapshot_texels`): two colors of one
        // image would copy their texels into one snapshot.
        let image = |index: usize| {
            let color = self.layout.colors.get(index)?.as_ref()?;
            color.target.texture.raw_image().ok()
        };
        let images: Vec<_> = colors.iter().map(|&index| image(index)).collect();
        let shared = images
            .iter()
            .enumerate()
            .any(|(i, a)| a.is_none() || images[i + 1..].iter().any(|b| b == a));
        if shared {
            return self.snapshot_fallback("the draw reads two attachments of one image");
        }
        if !self.snapshot_cheaper(pipeline, &colors) {
            return self.snapshot_fallback(
                "area: the snapshot draw over the draw's scissors costs more than the split's copy",
            );
        }
        Some(wanted)
    }

    /// Whether the draw of `pipeline` reads the pass's color `index`, whose own image a later
    /// fetch would not keep (`keeps_own_attachment`), through a snapshot of that image all the
    /// same, pinning it (the module docs): where nothing the pass's end decides from the whole
    /// pass changes what its instances through the draw's render, and nothing arrives with the
    /// draw. So: the pass's first instance recorded inline, which ran the pass's load actions
    /// itself (`render/inline_first.rs`); nothing arrived since an earlier pin
    /// (`pins_frozen`); no stand-in, memoryless color or depth/stencil attachment in the pass,
    /// hidden or not, every color stored `Store` (which no later call changes: so no texture over
    /// a buffer renders through an image it keeps, `kept_bytes.rs`), and no tile dispatch,
    /// execution of render commands, color groups or extended-range shadow that records an
    /// instance's draws again; a draw that fetches no color and whose state needs no hidden
    /// depth or stencil; and the color a 2D one.
    pub(super) fn pins_own_attachment(&self, pipeline: &DrawingState, index: usize) -> bool {
        let Some(color) = self.layout.colors.get(index).and_then(Option::as_ref) else {
            return false;
        };
        let graphics = &pipeline.base().graphics;
        let (depth, stencil) = self.layout.aspects();
        let plain = |c: &crate::render_pass::ColorTarget| {
            c.stand_in.is_none()
                && !c.memoryless
                && !super::memoryless(&c.target)
                && c.store == StoreAction::Store
        };
        matches!(self.inline, inline_first::Inline::Begun(_))
            && !self.pins_frozen
            && self.grouped.is_none()
            && self.dispatches.is_empty()
            && self.executions.is_empty()
            && !self.extended.keeps_draws()
            && self.layout.depth_stencil.is_none()
            && self.layout.colors.iter().flatten().all(plain)
            && graphics.storage_attachments.is_empty()
            && graphics.input_attachments.is_empty()
            && !self.state.needs_hidden(depth, stencil)
            && !crate::render_pass::not_2d(&color.target)
    }

    /// Whether a draw of `pipeline` that reads the pass's `colors` through snapshots costs no
    /// more than the split would: at any area over a buffer's bytes
    /// ([`over_bytes`](Self::over_bytes)), else by [`snapshot_weights`](Self::snapshot_weights);
    /// counted by its size where `METAL2VULKAN_LOG_PIPELINES` is set (`reader_sizes.rs`).
    fn snapshot_cheaper(&self, pipeline: &DrawingState, colors: &[usize]) -> bool {
        let (covered, pixels, bytes) = self.snapshot_weights(pipeline, colors);
        let within = self.over_bytes(colors) || covered * SNAPSHOT_PIXEL_COST <= bytes;
        if crate::pipeline::creation_log_enabled() {
            let device = self.core.device();
            let line = crate::device::lock(&device.shared().reader_sizes)
                .add(covered, pixels, bytes, within);
            if let Some(line) = line {
                device.inform("metal2vulkan-snapshot-readers", line);
            }
        }
        within
    }

    /// Whether every one of the pass's `colors` renders a texture over its buffer's bytes
    /// (`Texture::renders_bytes`: its image lies over them, or it is their shadow, as a surface
    /// over host pages that renders is, the consumer's guest's framebuffer), whose reader takes
    /// the snapshot whatever its area: the split's copy reads the bytes into the copy the reader
    /// reads (through a shadow, after the instances before wrote theirs back through a keep and a
    /// merge), where the snapshot draw reads only the pixels the draw covers. On the Iris Xe
    /// (`runtime_lane::attachment_read_costs::snapshot_crossover`, a 1920x1080 surface) the
    /// snapshot took 0.56 to 0.99 times the split's GPU time for readers through scissors of
    /// 960x540 and up, covering 1.5% to 100% of them, at most 1.08 times through the whole
    /// target, and at most 1.2 times through a 320x180 scissor it covers half of or more, about
    /// 10 microseconds a reader.
    fn over_bytes(&self, colors: &[usize]) -> bool {
        colors.iter().all(|&index| {
            let color = self.layout.colors.get(index).and_then(Option::as_ref);
            color.is_some_and(|color| color.target.texture.renders_bytes())
        })
    }

    /// What weighs whether a draw of `pipeline` that reads the pass's `colors` through snapshots
    /// costs no more than the split would: both forms are exact, so the draw takes the cheaper. The
    /// snapshot draw costs [`SNAPSHOT_PIXEL_COST`] for each color at each pixel the draw covers,
    /// which its Vulkan scissors bound (`EncoderState::vk_scissors`: each viewport's pixels within
    /// its scissor rectangle, inside the render area); the split's copy is weighed as reading and
    /// writing each color's every texel of the rendered level, its format's bytes each, an upper
    /// bound: a copy of a one-level, one-slice texture's image moves only the texels around the
    /// draw's scissors, or those the instances before changed (`ShadowCopies::cuts`), which errs
    /// toward the snapshot. The split's pass break (each attachment stored and loaded again) is
    /// left out, which errs toward the split. The
    /// draw's pixels for every color (the scissors' pixels times the colors), the colors' pixels,
    /// and their bytes; it takes the snapshot where the first times [`SNAPSHOT_PIXEL_COST`] is at
    /// most the last.
    fn snapshot_weights(&self, pipeline: &DrawingState, colors: &[usize]) -> (u64, u64, u64) {
        let selects = pipeline.base().graphics.writes_viewport_index
            || self.amplification.offsets_viewports();
        let mut rects = [vk::Rect2D::default(); MAX_VIEWPORTS];
        let bound: u64 = self
            .state
            .vk_scissors_in(&self.scissors, selects, &mut rects)
            .iter()
            .map(|r| u64::from(r.extent.width) * u64::from(r.extent.height))
            .sum();
        let (pixels, copied) = colors
            .iter()
            .filter_map(|&index| self.layout.colors.get(index)?.as_ref())
            .map(|color| {
                let [width, height] = color.target.extent;
                let bytes = block_bytes(color.target.texture.pixel_format()).unwrap_or(0);
                let pixels = u64::from(width) * u64::from(height);
                (pixels, pixels * u64::from(bytes))
            })
            .fold((0, 0), |(p, c), (pixels, bytes)| (p + pixels, c + bytes));
        (bound * colors.len() as u64, pixels, copied)
    }

    /// Whether the plan `settled` for `wanted` reads through them (the module docs' checks that
    /// come after the plan): the binder kept every slot in a variant that reads so, the draw reads
    /// no other attachment's texture at a slot, and binds none of the slots as a storage image.
    /// Else the reason it takes the split.
    pub(super) fn snapshot_holds(
        &self,
        settled: &Settled,
        wanted: &Wanted,
    ) -> Result<(), &'static str> {
        let plan = &settled.plan;
        let kept: BTreeSet<u32> = plan
            .variant
            .snapshot_textures
            .keys()
            .filter(|slot| slot.stage == Stage::Fragment && slot.class == SlotClass::Texture)
            .map(|slot| slot.index)
            .collect();
        if kept != wanted.slots || !settled.variant.pipeline().graphics.snapshot_reads {
            return Err("the binder left a slot out");
        }
        if plan.views.view_mask != 0 {
            return Err("the draw renders views");
        }
        for write in &plan.descriptors {
            let slot = match &write.source {
                WriteSource::Texture { slot, .. }
                | WriteSource::CombinedImageSampler {
                    image: ImageSource::Texture { slot, .. },
                    ..
                } => slot,
                _ => continue,
            };
            let snapshot = is_snapshot_slot(&wanted.slots, slot);
            if snapshot && write.ty == metal2vulkan::vk::DescriptorType::STORAGE_IMAGE {
                return Err("the draw binds the texture as a storage image");
            }
            if !snapshot && self.slot_attached(slot) {
                return Err("the draw reads another attachment's texture");
            }
        }
        Ok(())
    }

    /// The texture bound at `slot`, where one is.
    fn slot_texture(&self, slot: &SlotRef) -> Option<&Texture> {
        let table = match slot.stage {
            Stage::Vertex => &self.vertex,
            Stage::Fragment => &self.fragment,
            Stage::Object => &self.object,
            Stage::Mesh => &self.mesh,
            _ => return None,
        };
        table.textures.get(&slot.index)
    }

    /// Whether the texture bound at `slot` presents a color attachment of the pass.
    fn slot_attached(&self, slot: &SlotRef) -> bool {
        self.slot_texture(slot)
            .is_some_and(|texture| texture.attached_in(&self.layout))
    }

    /// The pass's colors (a mask of their indices) whose images a draw of a plan of `facts` binds
    /// as a storage image: the feedback loop layout serves no storage access ("This layout must only
    /// be used as either a color attachment or depth/stencil attachment and/or read-only access
    /// in a shader as a sampled image, combined image/sampler, or input attachment", the
    /// `VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT` layout), so such a draw records
    /// apart from an instance that holds one of them so (the module docs).
    pub(super) fn storage_colors(&self, facts: &PlanFacts) -> u32 {
        let mut colors = 0;
        for slot in facts.stored() {
            let Some(image) = self.slot_texture(slot).and_then(|t| t.raw_image().ok()) else {
                continue;
            };
            for (index, color) in self.layout.colors.iter().enumerate() {
                let same = color
                    .as_ref()
                    .is_some_and(|c| c.target.texture.raw_image().ok() == Some(image));
                if same {
                    colors |= 1u32.checked_shl(index as u32).unwrap_or(0);
                }
            }
        }
        colors
    }

    /// Whether a draw that reads through `wanted` records into a new instance, as the current
    /// one cannot hold its colors in the feedback loop layout: it records inline, its render
    /// pass instance begun already (`render/inline_first.rs`, `render/inline_splits.rs`); it
    /// runs an execution of render commands, whose runs begin render pass instances of their
    /// own (`render/executions.rs`); a draw recorded into it binds such a color's image as a
    /// storage image (`storage_colors`); or it leaves such a color out of its subpass (the
    /// storage form).
    pub(super) fn snapshot_apart(&self, wanted: &Wanted) -> bool {
        let current = &self.current;
        current.inline
            || current
                .pieces
                .iter()
                .any(|piece| matches!(piece, super::Piece::Execution(_)))
            || current.storage_colors & wanted.feedback.0 != 0
            || current
                .shape
                .storage
                .iter()
                .any(|&c| wanted.feedback.has(c as usize))
    }

    /// Say, once per device and reason, where `METAL2VULKAN_LOG_PIPELINES` is set, why a draw
    /// that reads its pass's texture took the split; `None`, for the caller to return.
    pub(super) fn snapshot_fallback(&self, reason: &'static str) -> Option<Wanted> {
        snapshot_fallback(self.core.device(), reason);
        None
    }
}

/// Whether `slot` is a fragment texture slot among `slots`.
pub(super) fn is_snapshot_slot(slots: &BTreeSet<u32>, slot: &SlotRef) -> bool {
    slot.stage == Stage::Fragment && slot.class == SlotClass::Texture && slots.contains(&slot.index)
}

/// [`RenderLowering::snapshot_fallback`]'s message, from anywhere the device is known.
pub(super) fn snapshot_fallback(device: &crate::Device, reason: &'static str) {
    if !crate::pipeline::creation_log_enabled() {
        return;
    }
    let said = &device.shared().snapshot_fallbacks;
    let first = {
        let mut said = crate::device::lock(said);
        let first = !said.contains(&reason);
        if first {
            said.push(reason);
        }
        first
    };
    if first {
        device.inform(
            "metal2vulkan-snapshot-fallback",
            format!(
                "a draw that reads its pass's texture after a barrier takes the split: {reason}"
            ),
        );
    }
}
