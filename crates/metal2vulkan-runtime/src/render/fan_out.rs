//! A rate-mapped pass's triangles that a host step records, each drawn once and fanned out to the
//! map's regions (`rate_mapped_pass.rs`, **Fan-out**): whether the device fans them out, what the
//! next command's draw records, the probe's and the routes' runs of an execution's commands, and
//! the region table and viewport a fanned-out draw records through.

use super::executions::Command;
use crate::render_values::{ScissorRect, Viewport};
use crate::Device;
use ash::vk;
use metal2vulkan::bind::ReplayFaults;
use metal2vulkan::device::{RateMapRegionPolicy, VkFeature};
use metal2vulkan::pipeline::RegionFanOut;
use std::sync::Arc;

/// Whether `device` fans a rate-mapped pass's triangles out where a host step records them: its
/// policy asks it (`RateMapRegionPolicy::FanOut`), and it enabled the features the fan-out and its
/// probe take.
pub(super) fn fans_out(device: &Device) -> bool {
    let compile = device.compile_device();
    let features = &compile.profile().enabled().features;
    compile.policy().rate_map_regions == RateMapRegionPolicy::FanOut
        && [
            VkFeature::GeometryShader,
            VkFeature::ShaderClipDistance,
            VkFeature::VertexPipelineStoresAndAtomics,
        ]
        .iter()
        .all(|feature| features.contains(feature))
}

/// What the draw of an execution's next command records where its host step fans the pass's
/// triangles out (the module docs): its probe, or its draw fanned out; a probe at the index of its
/// pair of words in the probe's flags (`BindState::region_fan_out`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct FanOutDraw {
    pub(super) mode: RegionFanOut,
    pub(super) index: u32,
    /// A draw's: the copies of its triangles the probe counted, which its work counts
    /// (`rate_mapped_pass::fan_out_work`).
    pub(super) copies: u32,
}

/// An encoder's run of an execution's commands where its host step fans the pass's triangles
/// out: probing their draws, or drawing them by the probe's words.
pub(super) enum FanOutRun {
    Probe(Probe),
    Route(Arc<FanOutRoutes>),
}

/// The probe's run of an execution's commands.
pub(super) struct Probe {
    /// The probe's flags, two words a probed draw, bound whole.
    flags: vk::DescriptorBufferInfo,
    /// The index the next probed draw takes.
    next: u32,
    /// The index of each command's draw, by its place among the commands, where it was probed.
    slots: Vec<Option<u32>>,
    /// A command's barrier was met, or a draw that writes memory: the probe, recorded before the
    /// pass, cannot see what a later command's vertex function may read, so no later command is
    /// probed.
    stopped: bool,
}

impl Probe {
    /// A probe whose first draw takes index `next` of `flags`.
    pub(super) fn new(flags: vk::DescriptorBufferInfo, next: u32) -> Probe {
        Probe {
            flags,
            next,
            slots: Vec::new(),
            stopped: false,
        }
    }

    /// The index its next draw would take, and each command's.
    pub(super) fn finish(self) -> (u32, Vec<Option<u32>>) {
        (self.next, self.slots)
    }
}

impl FanOutRun {
    /// The fan-out of the draw of the command at `place` among the commands: the probe's, where
    /// it probes on; the route's draw where the probe probed it and flagged none of its
    /// triangles; `None` for a draw recorded region by region.
    pub(super) fn begin(&self, place: usize) -> Option<FanOutDraw> {
        match self {
            FanOutRun::Probe(probe) => (!probe.stopped).then_some(FanOutDraw {
                mode: RegionFanOut::Probe,
                index: probe.next,
                copies: 0,
            }),
            FanOutRun::Route(routes) => {
                let index = routes.slots.get(place).copied().flatten()?;
                let word = |w: u32| routes.words.get(w as usize).copied();
                match word(2 * index)? {
                    0 => Some(FanOutDraw {
                        mode: RegionFanOut::Draw,
                        index,
                        copies: word(2 * index + 1)?,
                    }),
                    _ => None,
                }
            }
        }
    }

    /// After the draw of the command at `place`: whether it ran its fan-out variant (`fanned`),
    /// and whether its pipeline writes memory (`writes`).
    pub(super) fn end(&mut self, place: usize, fanned: bool, writes: bool) {
        if let FanOutRun::Probe(probe) = self {
            if fanned && !probe.stopped {
                if probe.slots.len() <= place {
                    probe.slots.resize(place + 1, None);
                }
                probe.slots[place] = Some(probe.next);
                probe.next += 1;
            }
            probe.stopped |= writes;
        }
    }

    /// A command's barrier stands: under the probe, no command from it on is probed.
    pub(super) fn barrier(&mut self) {
        if let FanOutRun::Probe(probe) = self {
            probe.stopped = true;
        }
    }

    /// Whether it probes.
    pub(super) fn probes(&self) -> bool {
        matches!(self, FanOutRun::Probe(_))
    }

    /// The probe's flags, where it probes.
    pub(super) fn flags(&self) -> Option<vk::DescriptorBufferInfo> {
        match self {
            FanOutRun::Probe(probe) => Some(probe.flags),
            FanOutRun::Route(_) => None,
        }
    }
}

/// The routes of an execution's commands at its host step (the module docs): the commands as the
/// step read them, once, for the probe and the routes alike, from command `start`; the faults the
/// read found; the index of each command's draw among the probe's, by its place among the
/// commands that run, where it was probed; and the probe's words.
pub(super) struct FanOutRoutes {
    pub(super) commands: Vec<Option<Command>>,
    pub(super) start: u32,
    pub(super) faults: ReplayFaults,
    pub(super) slots: Vec<Option<u32>>,
    pub(super) words: Arc<[u32]>,
}

/// What a region table was made for: the encoder's first viewport and scissor, and the render
/// area (`RenderLowering::fan_table`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct FanTableKey {
    viewport: [u64; 4],
    scissor: [u64; 4],
    extent: [u32; 2],
}

impl FanTableKey {
    /// The key of `viewport` and `scissor` over a render area of `extent`.
    pub(super) fn new(viewport: &Viewport, scissor: &ScissorRect, extent: [u32; 2]) -> FanTableKey {
        let v = viewport;
        FanTableKey {
            viewport: [v.origin_x, v.origin_y, v.width, v.height].map(f64::to_bits),
            scissor: [scissor.x, scissor.y, scissor.width, scissor.height],
            extent,
        }
    }
}

/// The target's viewport `[x, y, width, height]` a fan-out draw records through: the whole
/// render area `extent`, y flipped as each region's is (`rate_mapped_pass::on_device`).
fn target_rect([width, height]: [u32; 2]) -> [f32; 4] {
    let (width, height) = (width as f32, height as f32);
    [0.0, height, width, -height]
}

/// The region table of a fan-out draw over `regions`, each a region's viewport and scissor as the
/// device draws it (`RateMappedPass::draws`), through one viewport over the render area `extent`
/// (`bind::fan_out::table`): `None` where they are not a grid of segments or hold no pixel.
pub(super) fn table(regions: &[(vk::Viewport, vk::Rect2D)], extent: [u32; 2]) -> Option<Vec<u8>> {
    if !(regions.iter()).any(|(_, s)| s.extent.width > 0 && s.extent.height > 0) {
        return None;
    }
    let regions: Vec<([f32; 4], [u32; 4])> = (regions.iter())
        .map(|(v, s)| {
            let scissor = [
                u32::try_from(s.offset.x).ok()?,
                u32::try_from(s.offset.y).ok()?,
                s.extent.width,
                s.extent.height,
            ];
            Some(([v.x, v.y, v.width, v.height], scissor))
        })
        .collect::<Option<_>>()?;
    metal2vulkan::bind::fan_out::table(target_rect(extent), &regions)
}

/// The viewport and scissor a fan-out draw records through: one over the whole render area
/// `extent`, of the depth range `[min_depth, max_depth]`.
pub(super) fn target(
    extent: [u32; 2],
    [min_depth, max_depth]: [f32; 2],
) -> (vk::Viewport, vk::Rect2D) {
    let [x, y, width, height] = target_rect(extent);
    (
        vk::Viewport {
            x,
            y,
            width,
            height,
            min_depth,
            max_depth,
        },
        vk::Rect2D {
            offset: vk::Offset2D::default(),
            extent: vk::Extent2D {
                width: extent[0],
                height: extent[1],
            },
        },
    )
}
