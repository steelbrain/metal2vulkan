//! The encoder's own draws of a fanning pass the GPU routes (`render/fan_out.rs`, **Routes the
//! GPU picks**): each drawn by its fan-out variant where its probe flagged none of its triangles,
//! else by its region-copy variant into every region, as the GPU decides between the two, with no
//! host step and no wait.
//!
//! - **The draws.** On a device that draws region copies (`fan_out::copies_regions`) and has the
//!   indirect-count draws (`DeviceShared::count_draws`: `drawIndirectCount` and
//!   `multiDrawIndirect`), a direct draw of the encoder's own in a rate-mapped pass that would
//!   draw through its region-copy variant into the whole table, of one recording a row, whose
//!   pipeline writes no memory, recorded with its pipeline made, into an instance whose draws
//!   since it began wrote no memory and ran no execution of render commands
//!   ([`InstanceRoutes::stopped`]), records three things, each through its own variant of the
//!   draw's pipeline, made where a fanning pass meets the pipeline as its region-copy variant is:
//!   its region-copy route, its rows' recordings as they record today, each an indirect-count
//!   draw of as many of the draw's arguments as its row's regions; its fan-out route, one
//!   indirect-count draw of at most one; and its probe (`RegionFanOut::Probe`), into the
//!   instance's probes' secondary ([`Probes`]). Each route's count is a word of the draw's routes
//!   ([`ROUTE_WORDS`]), written as the draw records to the region-copy route's alone: 0 for the
//!   fan-out route and [`MAX_SEGMENTS`] for the other. Once all three recorded in the instance,
//!   with their pipelines made, the draw's mark ([`ROUTE_ROUTED`]) is written, and the routes'
//!   kernel decides its counts (`bind::fan_out::route_counts_module`). Any other draw records as
//!   before, and a draw whose fan-out route or probe did not record keeps the region-copy route:
//!   every draw is drawn once, by one route, in its place.
//! - **Before the instance** ([`ProbeSteps`]): after a barrier from everything before, the
//!   probes' secondary runs in a render pass instance of its own over the instance's framebuffer,
//!   which loads and stores each attachment and whose draws emit nothing; then after a barrier
//!   the kernel runs over each chunk of the probes' words, and a barrier takes its writes to the
//!   instance's indirect reads, to everything after, and to the host. So the probe reads what the
//!   draw reads: no draw of the instance before it wrote memory.
//! - **The kernel, made ahead.** A pass's first routed draw begins the kernel on the device's
//!   pool (`Device::begin_route_counts_kernel`), once per device; the encoder takes it as it
//!   ends only where it is made (`Device::route_counts_kernel`), so neither its making nor a wait
//!   for its turn lies on the encoding thread. An instance whose kernel is not made by then runs
//!   no steps: its draws' counts keep what the CPU wrote, so each draws its region-copy route, as
//!   a draw whose other routes did not record does.
//! - **Counted** (`WorkCounts::rate_map_gpu_routed_draws`, `rate_map_gpu_fanned_draws`,
//!   `rate_map_gpu_copied_draws`): an instance's marked draws as its steps are planned, and as
//!   the command buffer completes, the routed draws the kernel decided to fan out and those it
//!   decided to draw by region copies, read from their marks and counts ([`RouteCounts`]).
//!
//! Every route the GPU picks between is the pair a host step's routes mix in one pass
//! (`bind::fan_out`, **One route per pass**): a draw that tests `Equal` the depth another stored
//! meets it whichever each took, and a pixel takes a draw's fragments in the draw's order under
//! either (`render/fan_out.rs`, **Boxed draws**).

use super::bound::Bound;
use crate::device::functions::Vk;
use crate::error::Error;
use crate::exit_guard;
use crate::owned_kernel::{OwnedBinds, OwnedKernel, OwnedSet};
use crate::recorder::CountDraws;
use crate::render_target::RenderPassObject;
use crate::tracker::Barrier;
use crate::upload::{Upload, Uploads};
use crate::Device;
use ash::vk;
use metal2vulkan::bind::fan_out::{
    RouteCountsModule, MAX_SEGMENTS, PROBE_DRAW_WORDS, ROUTE_DECIDED, ROUTE_ROUTED, ROUTE_WORDS,
};
use metal2vulkan::pipeline::RegionFanOut;

/// The draws one chunk of an instance's probes' words holds: a dispatch of the kernel each.
const CHUNK_DRAWS: u32 = 16;

/// The routes of an instance's draws the GPU routes (the module docs).
#[derive(Default)]
pub(super) struct InstanceRoutes {
    /// A draw that writes memory, or an execution of render commands, recorded into the instance
    /// since it began: the probe, before the instance, would not see what a later draw may read,
    /// so no later draw of the instance is routed.
    pub(super) stopped: bool,
    /// The probes of its routed draws, from its first.
    pub(super) probes: Option<Probes>,
}

/// The probes of an instance's routed draws: the secondary they record into, begun against the
/// instance's pass, what they bound there, and their words, a chunk at a time.
pub(super) struct Probes {
    pub(super) draws: vk::CommandBuffer,
    pub(super) bound: Bound,
    chunks: Vec<Chunk>,
    /// The draws marked routed ([`mark`](Self::mark)).
    marked: u64,
}

/// One chunk of an instance's probes' words: [`CHUNK_DRAWS`] draws' probe words, zero, and their
/// routes, each written to the region-copy route's alone; and the draws that took a place in it.
struct Chunk {
    flags: Upload,
    routes: Upload,
    used: u32,
}

/// A routed draw's place among its instance's probes' words: the probes' secondary, which names
/// the instance's probes, its chunk, and its index there.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct RouteSlot {
    draws: vk::CommandBuffer,
    chunk: usize,
    pub(super) index: u32,
}

/// What a draw the GPU routes records at each of its draw's three recordings (the module docs):
/// its region-copy route, which takes its place; its fan-out route; and its probe. Or, where its
/// region-copy route's pipeline is being made, the pipeline of the other two's variant begun,
/// nothing recorded: made where a fanning pass meets the pipeline state, so a pass without a map
/// makes none.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum RouteCall {
    Copies,
    Fan(RouteSlot),
    Probe(RouteSlot),
    Begin(RegionFanOut),
}

/// One recording of a route of a draw the GPU routes: an indirect-count draw, indexed or not, of
/// at most `max` of the draw's arguments `stride` bytes apart from `args`, as many as the word at
/// `count` says.
#[derive(Clone, Copy, Debug)]
pub(super) struct RoutedDraw {
    pub(super) args: (vk::Buffer, u64),
    pub(super) stride: u32,
    pub(super) count: (vk::Buffer, u64),
    pub(super) max: u32,
    pub(super) indexed: bool,
}

/// What the recordings of a draw the GPU routes record (the module docs).
pub(super) enum GpuForm {
    /// Its region-copy route: each recording an indirect-count draw of as many of the draw's
    /// arguments as its row's regions.
    Copies(CopyRoute),
    /// Its fan-out route.
    Fan(RoutedDraw),
    /// Its probe, into the probes' secondary.
    Probe(vk::CommandBuffer),
}

/// A draw's region-copy route: its arguments, as many times in a row as its widest row's regions,
/// `stride` bytes apart, and its count.
#[derive(Clone, Copy, Debug)]
pub(super) struct CopyRoute {
    pub(super) args: (vk::Buffer, u64),
    pub(super) stride: u32,
    pub(super) count: (vk::Buffer, u64),
    pub(super) indexed: bool,
}

impl CopyRoute {
    /// The routed draw of a recording of a row of `regions` regions.
    pub(super) fn of(&self, regions: u32) -> RoutedDraw {
        RoutedDraw {
            args: self.args,
            stride: self.stride,
            count: self.count,
            max: regions,
            indexed: self.indexed,
        }
    }
}

impl RoutedDraw {
    /// Record it into `command_buffer` through `draws`.
    ///
    /// # Safety
    ///
    /// `command_buffer` is recording inside a pass compatible with the bound pipeline's, with
    /// every state the pipeline takes bound or set, and the arguments and the count are live
    /// until it completes, inside their buffers, which carry indirect usage.
    pub(super) unsafe fn record(&self, draws: &CountDraws, command_buffer: vk::CommandBuffer) {
        // SAFETY: the caller's contract.
        unsafe {
            draws.record_draw(
                command_buffer,
                self.indexed,
                self.args,
                self.count,
                (self.max, self.stride),
            )
        }
    }
}

/// Whether `device` routes a fanning pass's draws on the GPU (the module docs): it draws region
/// copies, and it has the indirect-count draws.
pub(super) fn routes_on_gpu(device: &Device) -> bool {
    super::fan_out::copies_regions(device) && device.shared().count_draws.is_some()
}

/// The bytes of a direct draw's arguments as an indirect draw reads them, `copies` times in a
/// row, and their stride: a `VkDrawIndexedIndirectCommand` of an indexed draw, else a
/// `VkDrawIndirectCommand`. `None` for any other draw.
pub(super) fn arguments(
    draw: &metal2vulkan::vk::DrawCommand,
    copies: u32,
) -> Option<(Vec<u8>, u32)> {
    let words: Vec<u32> = match *draw {
        metal2vulkan::vk::DrawCommand::Draw {
            vertex_count,
            instance_count,
            first_vertex,
            first_instance,
        } => vec![vertex_count, instance_count, first_vertex, first_instance],
        metal2vulkan::vk::DrawCommand::DrawIndexed {
            index_count,
            instance_count,
            first_index,
            vertex_offset,
            first_instance,
            ..
        } => vec![
            index_count,
            instance_count,
            first_index,
            vertex_offset as u32,
            first_instance,
        ],
        _ => return None,
    };
    let stride = 4 * words.len() as u32;
    let bytes = (0..copies)
        .flat_map(|_| words.iter().flat_map(|w| w.to_le_bytes()))
        .collect();
    Some((bytes, stride))
}

impl Probes {
    /// The probes of an instance, recording into `draws`, with no draw yet.
    pub(super) fn new(draws: vk::CommandBuffer) -> Probes {
        Probes {
            draws,
            bound: Bound::default(),
            chunks: Vec::new(),
            marked: 0,
        }
    }

    /// A place for the next routed draw, in a chunk of `uploads` where the last is full.
    pub(super) fn slot(
        &mut self,
        device: &Device,
        uploads: &mut Uploads,
    ) -> Result<RouteSlot, Error> {
        if self
            .chunks
            .last()
            .is_none_or(|chunk| chunk.used == CHUNK_DRAWS)
        {
            let flags_bytes = 4 * u64::from(PROBE_DRAW_WORDS * CHUNK_DRAWS);
            let flags = uploads.upload(device, &[], flags_bytes)?;
            let routes: Vec<u8> = (0..CHUNK_DRAWS)
                .flat_map(|_| [0, 0, MAX_SEGMENTS])
                .flat_map(u32::to_le_bytes)
                .collect();
            let routes = uploads.upload(device, &routes, routes.len() as u64)?;
            self.chunks.push(Chunk {
                flags,
                routes,
                used: 0,
            });
        }
        let chunk = self.chunks.len() - 1;
        let last = &mut self.chunks[chunk];
        let index = last.used;
        last.used += 1;
        Ok(RouteSlot {
            draws: self.draws,
            chunk,
            index,
        })
    }

    /// Whether `slot` is a place among these probes' words.
    pub(super) fn holds(&self, slot: RouteSlot) -> bool {
        slot.draws == self.draws
            && (self.chunks.get(slot.chunk)).is_some_and(|chunk| slot.index < chunk.used)
    }

    /// The probe words of `slot`'s chunk, which its probe binds whole, writing its own at its
    /// index.
    pub(super) fn flags(&self, slot: RouteSlot) -> Option<vk::DescriptorBufferInfo> {
        let chunk = self.chunks.get(slot.chunk).filter(|_| self.holds(slot))?;
        Some(
            vk::DescriptorBufferInfo::default()
                .buffer(chunk.flags.raw())
                .offset(chunk.flags.offset())
                .range(chunk.flags.len()),
        )
    }

    /// The word `slot`'s draw's route of `word` reads its count from: 1 the fan-out route's, 2
    /// the region-copy route's.
    pub(super) fn count(&self, slot: RouteSlot, word: u32) -> Option<(vk::Buffer, u64)> {
        let chunk = self.chunks.get(slot.chunk).filter(|_| self.holds(slot))?;
        let at = 4 * u64::from(slot.index * ROUTE_WORDS + word);
        Some((chunk.routes.raw(), chunk.routes.offset() + at))
    }

    /// Mark `slot`'s draw routed: its probe and both its routes recorded, the kernel decides its
    /// counts.
    pub(super) fn mark(&mut self, slot: RouteSlot) {
        if let Some(chunk) = self.chunks.get(slot.chunk).filter(|_| self.holds(slot)) {
            let at = 4 * u64::from(slot.index * ROUTE_WORDS);
            chunk.routes.write_at(at, &ROUTE_ROUTED.to_le_bytes());
            self.marked += 1;
        }
    }
}

/// The steps before an instance whose draws the GPU routes (the module docs), planned as the
/// encoder ends: the probes' pass and secondary, the kernel and its binds over each chunk, and the
/// counts read as the command buffer completes. They hold what they record until it completes.
pub(super) struct ProbeSteps {
    pass: RenderPassObject,
    draws: vk::CommandBuffer,
    kernel: OwnedKernel,
    binds: OwnedBinds,
    /// Each chunk's draws, its dispatch's push.
    chunks: Vec<u32>,
    _counts: RouteCounts,
}

impl ProbeSteps {
    /// The steps of `probes`, their secondary ended, run in `pass` on `device` by `kernel`, the
    /// routes' kernel made ahead (the module docs); their marked draws counted routed.
    pub(super) fn plan(
        device: &Device,
        probes: Probes,
        kernel: OwnedKernel,
        pass: RenderPassObject,
    ) -> Result<ProbeSteps, Error> {
        let sets: Vec<OwnedSet> = (probes.chunks.iter())
            .map(|chunk| OwnedSet {
                images: Vec::new(),
                storage_images: Vec::new(),
                buffers: vec![
                    (chunk.flags.raw(), chunk.flags.offset(), chunk.flags.len()),
                    (
                        chunk.routes.raw(),
                        chunk.routes.offset(),
                        chunk.routes.len(),
                    ),
                ],
            })
            .collect();
        let binds = OwnedBinds::new(device, &kernel, &sets)?;
        let chunks = probes.chunks.iter().map(|chunk| chunk.used).collect();
        (device.shared().counts).rate_map_gpu_routed_draws(probes.marked);
        let counts = RouteCounts {
            device: device.clone(),
            routes: (probes.chunks.into_iter())
                .map(|chunk| (chunk.routes, chunk.used))
                .collect(),
        };
        Ok(ProbeSteps {
            pass,
            draws: probes.draws,
            kernel,
            binds,
            chunks,
            _counts: counts,
        })
    }

    /// Record the steps into `command_buffer`, before the instance over `framebuffer` at
    /// `render_area` (the module docs).
    ///
    /// # Safety
    ///
    /// `command_buffer` is recording outside a render pass, before the instance's steps begin
    /// it; `framebuffer` is the instance's and compatible with the steps' pass, every image rests
    /// in the layout that pass holds it in, and the probes' secondary was recorded against a pass
    /// compatible with it and ended.
    pub(super) unsafe fn record(
        &self,
        raw: &Vk,
        command_buffer: vk::CommandBuffer,
        framebuffer: vk::Framebuffer,
        render_area: vk::Rect2D,
    ) {
        let begin = vk::RenderPassBeginInfo::default()
            .render_pass(self.pass.raw)
            .framebuffer(framebuffer)
            .render_area(render_area);
        // SAFETY: the caller's contract; the kernel, its binds and the chunks' words live until
        // the command buffer completes (held here, and the uploads with the command buffer).
        unsafe {
            Barrier::everything().record(raw, command_buffer);
            raw.cmd_begin_render_pass(
                command_buffer,
                &begin,
                vk::SubpassContents::SECONDARY_COMMAND_BUFFERS,
            );
            raw.cmd_execute_commands(command_buffer, &[self.draws]);
            raw.cmd_end_render_pass(command_buffer);
            Barrier::everything().record(raw, command_buffer);
            raw.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.kernel.pipeline,
            );
            for (index, &draws) in self.chunks.iter().enumerate() {
                self.binds.bind(raw, command_buffer, &self.kernel, index);
                raw.cmd_push_constants(
                    command_buffer,
                    self.kernel.layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    &draws.to_le_bytes(),
                );
                raw.cmd_dispatch(command_buffer, RouteCountsModule::groups(draws), 1, 1);
            }
            Barrier::to_all().record(raw, command_buffer);
        }
    }
}

/// The routes of an instance's routed draws, read as the command buffer completes, when its steps
/// go: each chunk's routes and its draws.
struct RouteCounts {
    device: Device,
    routes: Vec<(Upload, u32)>,
}

impl Drop for RouteCounts {
    fn drop(&mut self) {
        let Some(_busy) = exit_guard::enter() else {
            return;
        };
        let (mut fanned, mut copied) = (0u64, 0u64);
        for (routes, used) in &self.routes {
            let mut words = vec![0u32; (used * ROUTE_WORDS) as usize];
            let len = 4 * words.len() as u64;
            // SAFETY: the command buffer that wrote the routes completed, or never ran, and its
            // last barrier made the kernel's writes visible to the host; the words are writable
            // and apart from the region.
            unsafe { routes.copy_out(0, words.as_mut_ptr().cast(), len) };
            for draw in words.as_chunks::<3>().0 {
                if draw[0] == ROUTE_DECIDED {
                    match draw[1] {
                        1 => fanned += 1,
                        _ => copied += 1,
                    }
                }
            }
        }
        self.device
            .shared()
            .counts
            .rate_map_gpu_decided(fanned, copied);
    }
}
