//! An execution of render commands the CPU encoded, recorded chunk by chunk so a later one runs
//! the chunks no write reached again (`kept_commands.rs`).
//!
//! - **Which executions.** One `commit` records from the CPU's records (no host step, no GPU
//!   form), of a range in the call, of a buffer whose command types hold a draw of primitives
//!   (`DRAW` or `DRAW_INDEXED`: a mesh or patch draw is never plain, so a buffer of those alone
//!   has no chunk to keep), with no check of the caller's that may see something among its
//!   commands (one whose needs rule out what the buffer's records may hold calls nothing at
//!   `commit` either, `RenderExecution::check_sees`), no visibility result counted, no shadowed
//!   texture, no tile threadgroup memory argument set or reached, no rasterization rate map, no
//!   memoryless attachment or stand-in, no record barrier the buffer may hold, no write before
//!   the fragments and no indirect read of the encoder's draws before the call, no fragment write
//!   of the pass noted before it, no inherited table that binds anything, in a command buffer
//!   with no executed-commands handler that is not encoded again after its commit;
//!   no chunk of its range last recorded otherwise than plainly; and a chunk of its range run
//!   before, no write reaching it since (`kept_commands.rs`, **From the second execution**): a
//!   range's first execution records as one that keeps nothing.
//! - **The key.** A chunk is recorded from its records and from the inputs [`RenderKey`] holds:
//!   the inherited pipeline state, the instance's compatible pass and shape, the pass's extent,
//!   layers, samples and sample positions, the attachments that are among the resources made
//!   resident, the encoder's state, scissor rectangles, render target array length and
//!   amplification at the call, whether the pass holds tile imageblock data, and the resources
//!   made resident (`Residency::kept_key`). A kept chunk runs again only under an equal key.
//! - **Each chunk** runs its kept recording again, or records apart: its draws into a secondary of
//!   its own, begun against the instance's compatible pass, through an encoder of the call's
//!   state over a core of its own (`EncoderCore::capture`), which the pass's core then takes
//!   (`EncoderCore::absorb`); from a context of its own where an execution ran it before, no
//!   write reaching it since, else from its command buffer's own, kept by none. Its draws are
//!   plain where each one is (`RenderLowering::replay_plain`) and they left the pass's storage
//!   and layout as they found them in one run: such a chunk's secondary joins the others in the
//!   execution's first run, and is kept where it was recorded from a context of its own, its
//!   recording did nothing a later command buffer cannot do again (`Captured::plain`) and it
//!   reached nothing its command buffer alone holds (`CommandBuffer::own_reached`). Plain draws
//!   write no memory, so no draw of a later chunk depends on them but through the attachments,
//!   which the chunks' order in their subpass keeps.
//! - **A chunk that is not plain** is no more apart: its encoder records the rest of the range
//!   too, as the execution's would, its runs after the plain chunks' (its first sharing their
//!   run, as an execution's first run shares its instance with the encoder's draws before the
//!   call, which its encoder's state takes for granted), and the chunk is noted, so later
//!   executions of it record as before until a write reaches it.

use super::{
    Asked, Command, Draw, ExecutionRange, Found, PassStorage, PointIndices, Recorded,
    RenderExecution, Rendered, Run, Visibility,
};
use crate::amplification::Amplification;
use crate::command_buffer::CommandBuffer;
use crate::encoder::EncoderCore;
use crate::error::{internal, vulkan, Error};
use crate::indirect_command_buffer::Reader;
use crate::kept_commands::{self, Kept, KeptChunk, KeptContext, KeptEffects, Resolution};
use crate::pipeline::{BufferForms, RenderPipelineState};
use crate::render::{memoryless, RenderLowering};
use crate::render_pass::PassLayout;
use crate::render_state::{EncoderState, ScissorRects};
use crate::render_target::{PassShape, RenderPassObject};
use crate::residency::Residency;
use crate::resource::ResourceKey;
use crate::sample_positions::Sixteenths;
use metal2vulkan::bind::indirect_commands::{BufferStage, CommandRecord};
use metal2vulkan::bind::{ReplayFaults, UsedResource};
use metal2vulkan::metal::IndirectCommandType;
use std::collections::BTreeSet;
use std::ops::Range;
use std::sync::atomic::Ordering;
use std::sync::Arc;

/// What a chunk of an execution's draws is recorded from beside its records (the module docs).
#[derive(Clone)]
pub(super) struct RenderKey {
    pipeline: Option<RenderPipelineState>,
    /// Held, so no other pass takes its `VkRenderPass`'s handle while the key lives: a key of the
    /// same handle names the same pass, which the device keeps for the shape (each encoder holds
    /// a pass object of its own over it).
    compatible: Arc<RenderPassObject>,
    shape: PassShape,
    extent: [u32; 2],
    layers: u32,
    samples: u32,
    sample_positions: Vec<Sixteenths>,
    attached: Vec<ResourceKey>,
    state: EncoderState,
    scissors: ScissorRects,
    render_target_array_length: u32,
    amplification: Amplification,
    tile_imageblock: bool,
    resident: (Vec<UsedResource>, Vec<ResourceKey>),
}

impl PartialEq for RenderKey {
    fn eq(&self, other: &RenderKey) -> bool {
        self.pipeline == other.pipeline
            && self.compatible.raw == other.compatible.raw
            && self.shape == other.shape
            && self.extent == other.extent
            && self.layers == other.layers
            && self.samples == other.samples
            && self.sample_positions == other.sample_positions
            && self.attached == other.attached
            && self.state.same_values(&other.state)
            && self.scissors == other.scissors
            && self.render_target_array_length == other.render_target_array_length
            && self.amplification == other.amplification
            && self.tile_imageblock == other.tile_imageblock
            && self.resident == other.resident
    }
}

/// An execution that keeps its commands (the module docs): its key, what is kept of its range's
/// chunks in order, and the CPU's writes of the buffer as they were looked up.
pub(super) struct Keeping {
    key: RenderKey,
    kept: Vec<Kept>,
    read_at: u64,
}

/// A chunk recorded apart: its encoder after its commands, the addresses they named and what
/// they resolved to, the no-copy buffers' moves before it read its records, and whether a command
/// of it named a released state.
struct Apart {
    encoder: RenderLowering,
    resolved: Vec<Resolution>,
    moved: u64,
    released: bool,
}

impl RenderExecution {
    /// Whether the execution keeps its commands (the module docs), recorded by `command_buffer`
    /// against `residency` in a pass `rendered` lays out: its key and its chunks.
    pub(super) fn keeping(
        &self,
        command_buffer: &CommandBuffer,
        residency: &Residency,
        rendered: &Rendered,
    ) -> Option<Keeping> {
        let snapshot = &self.snapshot;
        let layout = &snapshot.layout;
        let ExecutionRange::Fixed(range) = &self.range else {
            return None;
        };
        let inherited = self.icb.descriptor().inherit_buffers;
        let tables = [
            &snapshot.vertex,
            &snapshot.fragment,
            &snapshot.object,
            &snapshot.mesh,
        ];
        let binds_nothing = tables.iter().all(|table| table.binds_nothing());
        let colors = layout.colors.iter().flatten();
        let plain_colors = colors
            .clone()
            .all(|color| color.stand_in.is_none() && color.planes.is_none());
        let memoryless_target = colors.clone().any(|color| memoryless(&color.target))
            || (layout.depth_stencil.as_ref())
                .is_some_and(|ds| ds.stand_in.is_some() || memoryless(&ds.target));
        // A mesh or patch draw is never plain (`RenderLowering::replay_plain`), so a buffer whose
        // commands can be no draw of primitives keeps no chunk: recording one apart would only
        // find it not plain.
        let primitives = IndirectCommandType::DRAW.union(IndirectCommandType::DRAW_INDEXED);
        let keeps = !range.is_empty()
            && self.icb.descriptor().command_types.intersects(primitives)
            && !self.icb.gpu_written()
            && !self.check_sees_now()
            && self.lent.is_none()
            && layout.visibility.is_none()
            && self.shadows.is_empty()
            && !self.icb.may_hold_barriers()
            && !snapshot.tile_memory.set()
            && !snapshot.tile_memory_reached
            && snapshot.rate_map.is_none()
            && snapshot.geometry_writes.is_empty()
            && snapshot.indirect_reads.is_empty()
            && !snapshot.pass_start.fragment_wrote()
            && (!inherited || binds_nothing)
            && rendered.backed.is_empty()
            && plain_colors
            && !memoryless_target
            && !crate::replay::is_replaying()
            && !command_buffer.reports_executed();
        if !keeps {
            return None;
        }
        let (kept, read_at) = self.icb.kept_chunks(range.clone())?;
        let resident = residency.kept_key();
        // An attachment is among them as a texture made resident, or in a heap or over a buffer
        // made resident.
        let keys: BTreeSet<ResourceKey> = resident.1.iter().copied().collect();
        let attached = colors
            .map(|color| &color.target.texture)
            .chain(layout.depth_stencil.as_ref().map(|ds| &ds.target.texture))
            .filter(|texture| {
                keys.contains(&texture.use_key())
                    || texture.bytes_key().is_some_and(|key| keys.contains(&key))
            })
            .map(|texture| texture.use_key())
            .collect();
        let key = RenderKey {
            pipeline: self.pipeline.clone(),
            compatible: Arc::clone(&snapshot.compatible),
            shape: snapshot.shape.clone(),
            extent: layout.extent,
            layers: layout.layers,
            samples: layout.samples,
            sample_positions: layout.sample_positions.clone(),
            attached,
            state: snapshot.state.clone(),
            scissors: snapshot.scissors.clone(),
            render_target_array_length: snapshot.render_target_array_length,
            amplification: snapshot.amplification.clone(),
            tile_imageblock: snapshot.tile_imageblock,
            resident,
        };
        Some(Keeping { key, kept, read_at })
    }

    /// Record the commands chunk by chunk through `core`, the pass's (the module docs), into
    /// secondaries begun against the execution's instance, over the pass's `storage`, in a pass
    /// `rendered` lays out, with `residency`, `command_buffer`'s: `record`'s, where the execution
    /// keeps its commands (`keeping`).
    pub(super) fn record_kept(
        &self,
        mut core: EncoderCore,
        (command_buffer, residency): (&CommandBuffer, &Residency),
        keeping: Keeping,
        found: &mut Found,
        (storage, rendered): (PassStorage, &mut Rendered),
    ) -> Result<Recorded, Error> {
        let device = command_buffer.device().clone();
        let (range, faults) = self.range.commands(self.icb.size())?;
        found.faults = ReplayFaults(found.faults.0 | faults.0);
        let Keeping { key, kept, read_at } = keeping;
        // Counted before the records are read: a write meanwhile records it again.
        self.read_at.store(read_at, Ordering::Release);
        let layout = rendered.backing(&self.snapshot.layout)?;
        let mut draws = Vec::new();
        let mut rest = None;
        for ((chunk, covered), kept) in kept_commands::chunks(range.clone()).zip(kept) {
            let (kept, keeps) = match kept {
                Kept::Recording(kept) => (Some(kept), true),
                Kept::Again => (None, true),
                Kept::First => (None, false),
            };
            let again = kept.and_then(|kept| {
                let held = kept.runs_again(command_buffer, (&key, &covered), residency)?;
                Some((kept, held))
            });
            if let Some((kept, held)) = again {
                core.run_kept(&kept.effects, held)?;
                found.misses = found.misses.wrapping_add(kept.misses);
                found.faults = ReplayFaults(found.faults.0 | kept.faults.0);
                draws.push(kept.raw);
                device
                    .shared()
                    .counts
                    .indirect_chunk_kept(u64::from(covered.end - covered.start));
                core.retain_shared(kept);
                continue;
            }
            // A chunk's first execution records from its command buffer's own.
            let context = match keeps {
                true => Some(KeptContext::new(command_buffer)?),
                false => None,
            };
            if let Some(context) = &context {
                core.retain_shared(Arc::clone(context) as _);
                command_buffer.lend_kept(Some(context));
            }
            let reached = command_buffer.own_reached();
            let mut chunk_found = Found::default();
            let apart = self.record_apart(
                (command_buffer, residency),
                covered.clone(),
                (&storage, &layout),
                &mut chunk_found,
            );
            if context.is_some() {
                command_buffer.lend_kept(None);
            }
            let Apart {
                encoder,
                resolved,
                moved,
                released,
            } = apart?;
            let plain = encoder.replay_plain
                && encoder.earlier.is_empty()
                && same_storage(&storage, &encoder)
                && same_layout(&layout, &encoder.layout);
            if !plain {
                self.icb.keep_chunk(chunk, None, read_at);
                rest = Some((encoder, covered.end, chunk_found));
                break;
            }
            let raw = encoder.current.draws;
            let capture = self.end_apart(encoder, &device)?;
            let captured = capture.finish_capture()?;
            let held = captured.retained.all_shared();
            let keeps = captured.plain() && !released && command_buffer.own_reached() == reached;
            if let (true, Some(held), Some(context)) = (keeps, held, context) {
                let effects = KeptEffects::new(
                    captured.summary.clone(),
                    (captured.keys.clone(), captured.taken.named().clone()),
                    &held,
                    captured.executions.clone(),
                );
                let kept = KeptChunk::new(
                    (context, raw),
                    covered,
                    Box::new(key.clone()),
                    (resolved, moved),
                    effects,
                    (chunk_found.misses, chunk_found.faults),
                );
                self.icb.keep_chunk(chunk, Some(Arc::new(kept)), read_at);
            }
            core.absorb(captured)?;
            found.misses = found.misses.wrapping_add(chunk_found.misses);
            found.faults = ReplayFaults(found.faults.0 | chunk_found.faults.0);
            draws.push(raw);
        }
        let Some((mut encoder, from, chunk_found)) = rest else {
            let run = Run {
                draws,
                other: None,
                shadow: false,
                steps: None,
                framed: Vec::new(),
                before: Vec::new(),
            };
            return Ok(Recorded {
                core,
                runs: vec![run],
                storage,
                layout,
            });
        };
        // The rest of the range, from the chunk that is not plain on, through its encoder, as the
        // execution's own records them.
        found.misses = found.misses.wrapping_add(chunk_found.misses);
        found.faults = ReplayFaults(found.faults.0 | chunk_found.faults.0);
        if from < range.end {
            let (commands, released) = self.icb.render_records(from..range.end, Reader::Commit)?;
            self.icb.degrade_released(&released);
            self.replay(&mut encoder, commands, residency, found)?;
        }
        let tail = self.finish(encoder, found, rendered, &device)?;
        let Recorded {
            core: capture,
            mut runs,
            storage,
            layout,
        } = tail;
        core.absorb(capture.finish_capture()?)?;
        let first = runs
            .first_mut()
            .ok_or_else(|| internal("an execution's rest of no runs"))?;
        draws.append(&mut first.draws);
        first.draws = draws;
        Ok(Recorded {
            core,
            runs,
            storage,
            layout,
        })
    }

    /// Record commands `covered` apart (the module docs): through an encoder of the call's state
    /// over a core of its own into a secondary of its own, begun against the instance's
    /// compatible pass, over the pass's `storage`, in `layout`, the call's; what their draws found
    /// noted into `found`.
    fn record_apart(
        &self,
        (command_buffer, residency): (&CommandBuffer, &Residency),
        covered: Range<u32>,
        (storage, layout): (&PassStorage, &PassLayout),
        found: &mut Found,
    ) -> Result<Apart, Error> {
        let snapshot = &self.snapshot;
        let moved = crate::buffer::host_cells_moved();
        let (commands, released) = self.icb.render_records(covered, Reader::Commit)?;
        self.icb.degrade_released(&released);
        let resolved = self.resolutions(&commands, residency);
        let draws = command_buffer.begin_draws(&snapshot.compatible)?;
        let raw = command_buffer.device().shared().vk();
        let visibility = Visibility::borrowing(None, None, raw, draws);
        let mut encoder = self.replay_encoder(
            EncoderCore::capture(command_buffer),
            (draws, visibility),
            (storage.clone(), layout.clone(), residency),
            BufferForms::Bound,
        );
        self.replay(&mut encoder, commands, residency, found)?;
        Ok(Apart {
            encoder,
            resolved,
            moved,
            released: !released.is_empty(),
        })
    }

    /// End a plain chunk's encoder: its secondary ended. Its core back.
    fn end_apart(
        &self,
        encoder: RenderLowering,
        device: &crate::Device,
    ) -> Result<EncoderCore, Error> {
        let raw = device.shared().vk();
        let RenderLowering {
            core,
            current,
            mut visibility,
            ..
        } = encoder;
        visibility.end(raw, current.draws);
        visibility.close_spares(raw, current.draws);
        // SAFETY: only this recording records the secondary command buffer.
        unsafe { core.command_buffer().end_raw(current.draws) }.map_err(vulkan)?;
        Ok(core)
    }

    /// The addresses `commands` name (their buffers', where the buffer does not inherit them,
    /// and their draws'), resolved against `residency` (`kept_commands::resolutions`).
    fn resolutions(&self, commands: &[Command], residency: &Residency) -> Vec<Resolution> {
        let descriptor = self.icb.descriptor();
        let stages = [
            (BufferStage::Vertex, descriptor.max_vertex_buffer_bind_count),
            (
                BufferStage::Fragment,
                descriptor.max_fragment_buffer_bind_count,
            ),
            (BufferStage::Object, descriptor.max_object_buffer_bind_count),
            (BufferStage::Mesh, descriptor.max_mesh_buffer_bind_count),
        ];
        let addresses = commands.iter().flat_map(|(record, _)| {
            let bound = stages
                .iter()
                .filter(|_| !descriptor.inherit_buffers)
                .flat_map(|&(stage, count)| {
                    let bindings = record.bindings.buffers(stage);
                    (0..count).zip(bindings).filter_map(|(_, binding)| *binding)
                })
                .filter(|binding| binding.names_buffer())
                .map(|binding| binding.address);
            bound
                .chain(addresses(record, &self.icb))
                .collect::<Vec<_>>()
        });
        kept_commands::resolutions(addresses, residency)
    }
}

/// The addresses a command's draw names: its index buffer's, and a patch draw's buffers'.
fn addresses(
    record: &CommandRecord,
    icb: &crate::indirect_command_buffer::IndirectCommandBuffer,
) -> Vec<u64> {
    let Asked::Draw(draw) = Asked::of(&record.action, icb) else {
        return Vec::new();
    };
    match draw {
        Draw::Primitives(..) | Draw::Mesh(..) => Vec::new(),
        Draw::Indexed(_, address) => vec![address],
        Draw::Patches {
            patch_indices,
            control_point_indices,
            factors,
            ..
        } => {
            let points = match control_point_indices {
                PointIndices::Buffer(address) => Some(address),
                PointIndices::Own | PointIndices::Zero => None,
            };
            patch_indices
                .into_iter()
                .chain(points)
                .chain([factors.buffer])
                .collect()
        }
    }
}

/// Whether a chunk's `encoder` left the pass's storage as `storage` held it: it made none. (No
/// draw of an execution that keeps its commands reaches the tile threadgroup memory.)
fn same_storage(storage: &PassStorage, encoder: &RenderLowering) -> bool {
    storage.imageblock.is_some() == encoder.imageblock.is_some()
        && storage.shadow.is_some() == encoder.shadow.is_some()
}

/// Whether a chunk's draws left the pass's layout as `before` held it: no stand-in or sample
/// planes made.
fn same_layout(before: &PassLayout, after: &PassLayout) -> bool {
    let colors = |layout: &PassLayout| {
        (layout.colors.iter())
            .map(|color| {
                color
                    .as_ref()
                    .map(|c| (c.stand_in.is_some(), c.planes.is_some()))
            })
            .collect::<Vec<_>>()
    };
    let depth =
        |layout: &PassLayout| (layout.depth_stencil.as_ref()).map(|ds| ds.stand_in.is_some());
    colors(before) == colors(after) && depth(before) == depth(after)
}
