//! Color attachments rendered into linear images in place.
//!
//! A texture over a buffer of the runtime's memory is a linear image over the buffer's bytes where
//! the device makes one that lies on Metal's layout there (`Device::alias_image`; one over a
//! no-copy buffer's host pages is an image in place where the device's check passed for an image
//! that renders, `in_place.rs`, else a shadow), and a heap's texture may be a linear image
//! (`texture/heap.rs`). On a device that enabled `VK_NV_linear_color_attachment`'s
//! `linearColorAttachment` (`enables.rs`), such an image of a format whose linear tiling has
//! `LINEAR_COLOR_ATTACHMENT_NV` takes color attachment usage (`ImageFacts::linear`), and a pass
//! renders into it in place: no shadow, no copy in, no copy back. Vulkan renders it there only
//! where every color attachment of the render pass instance is linear (the extension's rule): such
//! an attachment is *bound* ([`bound`]; `ImageFacts::linear_attachment_rule`). A multisampled image
//! is never linear, so:
//!
//! - **A pass beside a color attachment that is not linear** (a private texture's image, an image
//!   of the runtime's standing in for an attachment), **or with a depth or stencil attachment**
//!   (never linear; NVIDIA's hardware renders a linear color only with none: an RTX 3090 lost the
//!   device on one, and Mesa's NVK renders in place only without one), renders each bound
//!   attachment through the image its texture keeps for the queue
//!   (`attachment_stand_in::kept_backing`), copied in where the pass loads the attachment over the
//!   box its draws write in ([`Coverage`]), and, where it stores it, the words the pass changed
//!   merged back into the texture's bytes (`linear_merge.rs`), once for a run of such passes one
//!   after another (`pass_runs.rs`): never a copy back of the box, which would write the pixels
//!   no draw wrote with what the copy in read, where Metal on a Mac family GPU writes none of them
//!   (another writer's bytes there survive the pass). Decided as the pass
//!   begins, from its descriptor ([`apart`]), so every use a draw makes of the attachment (a
//!   framebuffer fetch's view, a view in another format) finds the image the pass renders; and
//!   again as it ends ([`settle`]), where a draw made an attachment of the runtime's (a fetched
//!   texture's scratch, a stand-in of another format): a bound attachment a draw took a view of is
//!   refused there (`NotImplemented(Render)`), as the pass cannot render it apart any more.
//! - **A multisampled pass resolving into a bound texture** resolves into a scratch image copied
//!   there (`resolve.rs`).
//!
//! Each announces `linear-attachment-copies` (`Degradation::LinearAttachmentCopies`), naming the
//! rule. A device whose linear tiling has the color attachment feature renders a linear image
//! beside any other, so no attachment is bound there; a format whose linear tiling has neither
//! keeps its texture's shadow (`shadowed-texture`). On every device, a pass that does not store
//! such an attachment renders it through the image its texture keeps, copied nothing back, so
//! that its buffer keeps its bytes (`kept_bytes.rs`, settled before these as the pass ends): an
//! attachment that is not linear beside the others.
//!
//! **Where the rule allows it, the pass renders in place** ([`choose`], as it ends), whatever it
//! loads: the copies above are a fallback for an attachment the device cannot render in place,
//! never a choice for speed. A copy of a loaded attachment into the image its texture keeps would
//! make a draw's read of the texture's bytes (a sampled texture over them, a buffer, a pointer:
//! the consumer's x86 guest samples the surface it renders into) miss the pass's earlier draws,
//! and no structure proves a pass free of such reads in general; a cleared one renders in place
//! too, though clearing an image of the device's and copying it back costs less (below).
//!
//! Measured (on an RTX 3090 and the consumer's run of it on an RTX 5080 Laptop
//! GPU; 1080p `BGRA8` over host pages, in place against copies of the draws' box, medians of 31
//! interleaved): on the 5080 a blending draw reads and writes its pixels across the bus once each
//! in place, so one layer costs 0.55 to 0.87 of the copies at every area and two or more 1.5 to 7.4
//! times them; an opaque layer costs 0.63, two 1.23; a pass that clears the linear image in place
//! costs 10 to 17 times clearing an image of the device's on chip and copying it back (1.9 to 4.5
//! on the 3090). A draw's pixels are bounded by its Vulkan scissors (its scissor rectangles within
//! its viewports' pixels, `EncoderState::vk_scissors`), outside which Vulkan writes none; an
//! execution of render commands, a tile dispatch, and a pass through a rasterization-rate map
//! bound none, and count as writing the whole render area.

use super::attachment_stand_in::{self, Cause, StandIn};
use super::color_reformat::Routing;
use super::command_buffer::CommandBuffer;
use super::degradation::Degradation;
use super::error::{Error, RuntimeGap};
use super::render_pass::{ColorTarget, Target, MAX_COLOR_ATTACHMENTS};
use super::render_values::LoadAction;
use super::Device;
use ash::vk;
use metal2vulkan::pipeline::ColorAttachmentState;

/// What a pass's draws write of each color attachment, as far as the host bounds it: the box an
/// attachment rendered apart is copied in and merged back over ([`choose`]).
#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct Coverage {
    /// By color index: the box of every pixel a draw may write, `[x0, y0, x1, y1]`, half-open.
    colors: [Option<[u32; 4]>; MAX_COLOR_ATTACHMENTS],
    /// A draw's pixels were not bounded (an execution of render commands, a tile dispatch, a
    /// draw through a rasterization-rate map or of a state the host does not read): each
    /// attachment's draws write over the whole render area, in no box.
    unbounded: bool,
}

impl Coverage {
    /// A coverage no draw's pixels bound, an encoder's that records an execution's commands
    /// (`render/executions.rs`), whose pass the executing encoder ends.
    pub(super) fn unbounded() -> Coverage {
        Coverage {
            colors: [None; MAX_COLOR_ATTACHMENTS],
            unbounded: true,
        }
    }

    /// Add a draw that writes the colors `writes` names (by color index: `Some(_)` where it
    /// writes the color; [`writes`]), in `rects` (its Vulkan scissors, which bound every pixel it
    /// writes).
    pub(super) fn add(&mut self, writes: &[Option<bool>], rects: &[vk::Rect2D]) {
        let bounds = rects
            .iter()
            .filter(|r| r.extent.width > 0 && r.extent.height > 0)
            .map(|r| {
                let (x, y) = (r.offset.x.max(0) as u32, r.offset.y.max(0) as u32);
                [x, y, x + r.extent.width, y + r.extent.height]
            })
            .reduce(union);
        let Some(bounds) = bounds else {
            return;
        };
        for (color, write) in self.colors.iter_mut().zip(writes) {
            if write.is_some() {
                *color = Some(color.map_or(bounds, |b| union(b, bounds)));
            }
        }
    }

    /// A draw whose pixels the host does not bound.
    pub(super) fn everywhere(&mut self) {
        self.unbounded = true;
    }

    /// The box the draws write color `index` in: `None` where no draw writes it, or the draws
    /// are not bounded, which write over the whole render area.
    pub(super) fn of(&self, index: usize) -> Option<[u32; 4]> {
        match self.unbounded {
            true => None,
            false => self.colors.get(index).copied().flatten(),
        }
    }
}

/// What a draw writes of each color, by color index: `Some(reads)` where it writes the color,
/// reading it where `reads`; `None` where it writes none, past its colors too.
pub(super) type ColorWrites = [Option<bool>; MAX_COLOR_ATTACHMENTS];

/// What a draw of a state with color attachments `colors` (by color index) writes of each:
/// `Some(reads)` where it writes the color (a present attachment with a write mask), reading it as
/// it does where `reads` (a blend, or a write mask that keeps some of its channels).
pub(super) fn writes(colors: &[Option<ColorAttachmentState>]) -> ColorWrites {
    let all = metal2vulkan::vk::ColorComponentFlags::known_bits();
    let mut writes = [None; MAX_COLOR_ATTACHMENTS];
    for (write, state) in writes.iter_mut().zip(colors) {
        *write = state
            .as_ref()
            .filter(|s| !s.write_mask.is_empty())
            .map(|state| state.blend.is_some() || !state.write_mask.contains(all));
    }
    writes
}

/// What a draw writes of each of the pass's colors, by pass color index: each state color's
/// [`writes`] (`state`, by state color index) in the pass color its draws store it into
/// (`routing.placed`), none for a dropped one; and, where its variant zeroes the pass colors at no
/// place of its record (`zeroed`), each of those (`routing.uncovered`), written without a read. A
/// pass color two of them reach reads where either does.
pub(super) fn pass_writes(state: &[Option<bool>], routing: &Routing, zeroed: bool) -> ColorWrites {
    let mut out = [None; MAX_COLOR_ATTACHMENTS];
    let mut put = |index: u32, reads: bool| {
        // A pass color's index: less than its colors'.
        if let Some(write) = out.get_mut(index as usize) {
            *write = Some(write.unwrap_or(false) || reads);
        }
    };
    for (color, &placed) in &routing.placed {
        if let Some(Some(reads)) = state.get(*color as usize) {
            put(placed, *reads);
        }
    }
    if zeroed {
        for &index in routing.uncovered.values() {
            put(index, false);
        }
    }
    out
}

/// The box holding boxes `a` and `b`.
fn union(a: [u32; 4], b: [u32; 4]) -> [u32; 4] {
    [
        a[0].min(b[0]),
        a[1].min(b[1]),
        a[2].max(b[2]),
        a[3].max(b[3]),
    ]
}

/// Whether `target` is bound by `VK_NV_linear_color_attachment`'s rule: its image is linear now,
/// and the device renders a linear image viewed in the target's format only where every color
/// attachment of the render pass instance is linear.
pub(super) fn bound(device: &Device, target: &Target) -> Result<bool, Error> {
    if !target.texture.linear_image()? {
        return Ok(false);
    }
    #[cfg(test)]
    if (device.shared().linear_rule_forced).load(std::sync::atomic::Ordering::Relaxed) {
        return Ok(true);
    }
    let profile = device.compile_device().profile();
    let facts = device.shared().textures.facts(profile);
    Ok(facts.linear_attachment_rule(target.format))
}

/// Why [`apart`] renders a pass's bound attachments apart as it begins.
#[derive(Clone, Copy, Debug)]
enum Apart {
    /// The pass's color attachment at this index is not linear: the extension's rule.
    Color(usize),
    /// The pass has a depth or stencil attachment (named), which is never linear.
    DepthStencil(&'static str),
}

/// Render the bound attachments among `colors`, a pass's as it begins, apart where another is
/// not linear, or where the pass has a depth or stencil attachment (`depth_stencil` names it):
/// each through the image its texture keeps for `maker`'s queue, copied in and back (the module
/// docs), announced.
pub(super) fn apart(
    maker: &CommandBuffer,
    colors: &mut [Option<ColorTarget>],
    depth_stencil: Option<&'static str>,
) -> Result<(), Error> {
    let device = maker.device();
    let mut bound_at = Vec::new();
    let mut other = None;
    for (index, color) in colors.iter().enumerate() {
        let Some(color) = color else {
            continue;
        };
        if color.stand_in.is_none() && bound(device, &color.target)? {
            bound_at.push(index);
        } else if !color.target.texture.linear_image()? {
            other.get_or_insert(index);
        }
    }
    let why = match (other, depth_stencil) {
        (Some(other), _) => Apart::Color(other),
        (None, Some(attachment)) => Apart::DepthStencil(attachment),
        (None, None) => return Ok(()),
    };
    for index in bound_at {
        if let Some(Some(color)) = colors.get_mut(index) {
            through_backing(maker, index, color, || match why {
                Apart::Color(other) => {
                    rule_broken(format!("the pass's color attachment {other} is not linear"))
                }
                Apart::DepthStencil(attachment) => format!(
                    "the device renders a linear image of the format as a color attachment \
                     only in a render pass instance with no depth or stencil attachment, as \
                     a depth or stencil image is never linear (NVIDIA: an RTX 3090 lost the \
                     device on one), and the pass has {attachment}"
                ),
            })?;
        }
    }
    Ok(())
}

/// Render the bound attachments among `layout`'s colors, a pass's as it ends, apart where a
/// render pass instance of it holds a color attachment that is not linear: `other`, the first
/// color index an instance renders through an image of the runtime's (a stand-in of another
/// format), of the first instance with one. Refused (`NotImplemented(Render)`) where a draw took a view of a bound one
/// (`taken`): the pass renders it in place.
pub(super) fn settle(
    maker: &CommandBuffer,
    colors: &mut [Option<ColorTarget>],
    mut other: Option<usize>,
    taken: impl Fn(usize) -> bool,
) -> Result<(), Error> {
    let device = maker.device();
    let mut bound_at = Vec::new();
    for (index, color) in colors.iter().enumerate() {
        let Some(color) = color else {
            continue;
        };
        if color.stand_in.is_none() && bound(device, &color.target)? {
            bound_at.push(index);
        } else if !color.target.texture.linear_image()? {
            other.get_or_insert(index);
        }
    }
    let Some(other) = other else {
        return Ok(());
    };
    if bound_at.iter().any(|&index| taken(index)) {
        return Err(Error::NotImplemented(RuntimeGap::Render));
    }
    for index in bound_at {
        if let Some(Some(color)) = colors.get_mut(index) {
            through_backing(maker, index, color, || {
                rule_broken(format!(
                    "an instance of the pass renders its color {other} through an image of \
                     the runtime's"
                ))
            })?;
        }
    }
    Ok(())
}

/// Render `color`, a pass's bound color attachment `index`, through the image its texture keeps
/// for `maker`'s queue, copied in where the pass loads it and back where it stores it (over the
/// render area, until [`choose`] bounds a loaded one's copies by its draws' box), announced
/// `linear-attachment-copies`, `reason` saying why.
fn through_backing(
    maker: &CommandBuffer,
    index: usize,
    color: &mut ColorTarget,
    reason: impl FnOnce() -> String,
) -> Result<(), Error> {
    attachment_stand_in::check_copies(&color.target)?;
    let queue = maker.queue_place().id();
    let own = color.target.clone();
    color.target = attachment_stand_in::kept_backing(maker, &own, queue, true)?;
    let format = own.texture.pixel_format();
    let site = || {
        format!("a {format:?} texture's linear image, color attachment {index} of a render pass")
    };
    let kind = Degradation::LinearAttachmentCopies;
    maker.device().degrade(kind, || (site(), reason()));
    color.stand_in = Some(StandIn {
        own,
        cause: Cause::Linear,
        area: None,
    });
    Ok(())
}

/// The reason of [`apart`] and [`settle`]'s notices: the extension's rule, broken as `beside`
/// says.
fn rule_broken(beside: String) -> String {
    format!(
        "the device renders a linear image of the format as a color attachment only where every \
         color attachment of the render pass instance is linear (VK_NV_linear_color_attachment), \
         and {beside}"
    )
}

/// Settle each bound attachment among `colors`, a pass's as it ends (the module docs): one
/// rendered apart that the pass loads, over the box its draws write in (`coverage`); every other
/// one (no stand-in) in place, whatever its load. Counted as the pass renders them
/// (`WorkCounts::linear_passes_in_place`, `linear_passes_through_copies`), but for one a draw
/// took a view of (`taken`).
pub(super) fn choose(
    maker: &CommandBuffer,
    colors: &mut [Option<ColorTarget>],
    coverage: &Coverage,
    taken: impl Fn(usize) -> bool,
) -> Result<(), Error> {
    let device = maker.device();
    let (mut any, mut through) = (false, false);
    for (index, color) in colors.iter_mut().enumerate() {
        let Some(color) = color else {
            continue;
        };
        if let Some(stand_in) = color.stand_in.as_mut() {
            if stand_in.cause == Cause::Linear {
                (any, through) = (true, true);
                if color.load == LoadAction::Load {
                    stand_in.area = coverage.of(index);
                }
            }
            continue;
        }
        any |= !taken(index) && bound(device, &color.target)?;
    }
    if any {
        device.shared().counts.linear_pass(through);
    }
    Ok(())
}
