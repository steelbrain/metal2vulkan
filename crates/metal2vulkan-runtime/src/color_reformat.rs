//! A render pipeline's color drawn into a pass attachment of another pixel format, as plain Metal
//! stores it.
//!
//! Metal's validation stops such a state at `setRenderPipelineState:`; plain Metal draws it
//! (Metal probes, an M4 under macOS 27.0.1, each case in its own process). Each sample holds one
//! record of the state's colors, packed as `imageblockSampleLength` packs them ([`color_record`]);
//! the pass loads and stores each of its attachments at its own place in its own record. A color
//! written in the state's pixel format is that format's encoding, in red-first order (`BGRA8Unorm`
//! as `RGBA8Unorm`, `BGR10A2Unorm` as `RGB10A2Unorm`), and the pass stores the bytes there into its
//! attachment unchanged, except that a `BGRA8Unorm` or `BGRA8Unorm_sRGB` attachment exchanges bytes
//! 0 and 2 and a `BGR10A2Unorm` attachment its red and blue fields. A blend reads the attachment's
//! bytes as the state's format, and a write mask names the state's channels.
//!
//! Where the attachment's bytes then are the state's encoding in a channel order a Vulkan format
//! names, a draw through a view of that format ([`view_format`]) stores, blends and masks as
//! Metal does: the state's format in the attachment's channel order (`RGBA8Unorm` in a
//! `BGRA8Unorm` pass through `B8G8R8A8_UNORM`, its own; `RGBA8Unorm_sRGB` in a `BGRA8Unorm` pass
//! through `B8G8R8A8_SRGB`). The runtime draws such a color where the view's format is the
//! attachment's own; a view of another format stays refused `AttachmentFormat`.
//!
//! Each state color is drawn into the pass color at its place in the record ([`routing`]): the
//! draws' subpass references that pass color at the state color's location. A pass color at no
//! state color's place is zeroed or kept as Metal's compiled code writes the record
//! ([`uncovered`]), and a state color at no pass color's place is dropped. A color whose bytes
//! Metal stores into part of an attachment, or into more than one, stays refused.

use super::clear_value::has_channels;
use metal2vulkan::device::{DeviceProfile, FormatUsage};
use metal2vulkan::metal::format::{color_record, vk_format, FormatEmulation};
use metal2vulkan::metal::{BlendFactor, BlendOperation, ColorWriteMask, PixelFormat};
use metal2vulkan::pipeline::ColorAttachmentDescriptor;
use metal2vulkan::vk::{Format, NumericFormat};
use std::collections::{BTreeMap, BTreeSet};

/// The fragment output locations a draw can use.
const LOCATIONS: u32 = 8;

/// The Vulkan format a color attachment of `pf` holds Metal's texels in on `profile`, where it
/// holds them in a format of exactly Metal's layout and numeric format; `None` for a format held
/// otherwise (an extended-range format's planes) or not renderable.
pub(crate) fn held(pf: PixelFormat, profile: &DeviceProfile) -> Option<Format> {
    let usages: BTreeSet<FormatUsage> = [FormatUsage::ColorAttachment].into();
    let mapping = vk_format(pf, &usages, profile).ok()?;
    (mapping.emulation == FormatEmulation::Direct).then_some(mapping.format)
}

/// The lowest color index whose place in Metal's per-sample record of a state whose colors are
/// `state` is not its place in the record of the pass whose colors are `pass`
/// ([`color_record`]): a color both have that starts at another byte, or a color of either that
/// overlaps a color of another index of the other. Plain Metal stores the state's bytes at each
/// place into the pass's attachment there (Metal probes: a state of colors 0 and 1 in a pass of
/// color 1 alone stores the state's color 0 into the pass's color 1, and a state of color 1 alone
/// in a pass of colors 0 and 1 its color 1 into the pass's color 0). `None` where every color keeps
/// its place, or where a format has no place in the record.
pub(crate) fn misplaced(
    state: &[Option<PixelFormat>],
    pass: &[Option<PixelFormat>],
) -> Option<u32> {
    let (state, pass) = (spans(state)?, spans(pass)?);
    let mut lowest = None;
    for &(index, offset, size) in &state {
        for &(other, at, length) in &pass {
            let overlaps = offset < at + length && at < offset + size;
            let moved = match index == other {
                true => offset != at,
                false => overlaps,
            };
            if moved {
                let first = index.min(other);
                lowest = Some(lowest.map_or(first, |l: u32| l.min(first)));
            }
        }
    }
    lowest
}

/// How a draw stores a state's per-sample record into its pass's attachments as plain Metal
/// does: each state color into the pass color at its place in the record ([`color_record`]), a
/// pass color at no state color's place as [`uncovered`] says, a state color at no pass color's
/// place nowhere (Metal probes).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Routing {
    /// The pass color each fragment output location stores into, by location: the draws'
    /// subpass's color references. Empty where each location stores into the pass color of its
    /// own index.
    pub(crate) locations: Vec<Option<u32>>,
    /// The state colors at a pass color's place, by state color index: that pass color.
    pub(crate) placed: BTreeMap<u32, u32>,
    /// The pass colors no state color lies on, by the location the draws reference each at: its
    /// own index where the state has no color there, else the highest location the state has no
    /// color at and no other color takes (a fragment function rarely writes there).
    pub(crate) uncovered: BTreeMap<u32, u32>,
    /// The state colors at no pass color's place, whose output is dropped.
    pub(crate) dropped: BTreeSet<u32>,
}

impl Routing {
    /// Each color at its own index: a color both have placed there, one the pass alone has
    /// uncovered at its index, one the state alone has dropped. The draws of a state that writes
    /// no color, where the record decides nothing, and of a record of formats without a place.
    pub(crate) fn by_index(state: &[Option<PixelFormat>], pass: &[Option<PixelFormat>]) -> Routing {
        let has = |formats: &[Option<PixelFormat>], index: u32| {
            formats
                .get(index as usize)
                .copied()
                .flatten()
                .is_some_and(|pf| pf != PixelFormat::Invalid)
        };
        let mut routing = Routing::default();
        for index in 0..state.len().max(pass.len()) as u32 {
            match (has(state, index), has(pass, index)) {
                (true, true) => {
                    routing.placed.insert(index, index);
                }
                (false, true) => {
                    routing.uncovered.insert(index, index);
                }
                (true, false) => {
                    routing.dropped.insert(index);
                }
                (false, false) => {}
            }
        }
        routing
    }

    /// The lowest location whose color index the routing changes, where it changes one.
    pub(crate) fn moved(&self) -> Option<u32> {
        (0u32..)
            .zip(&self.locations)
            .find(|&(location, index)| *index != Some(location))
            .map(|(location, _)| location)
    }
}

/// How a draw of a state whose colors are `state` stores its record into a pass whose colors are
/// `pass` ([`Routing`]). `None` where Metal's answer is not a color's bytes stored whole into one
/// attachment: a state color overlapping a pass color it does not start at, or more than one; a
/// pass color a state color overlaps but does not lie on; or no free location for an uncovered
/// pass color. A state color starting at a pass color of its own index is placed there whatever
/// their sizes (the format check decides). `None` too where a format has no place in the record.
pub(crate) fn routing(
    state: &[Option<PixelFormat>],
    pass: &[Option<PixelFormat>],
) -> Option<Routing> {
    let (spans_of_state, spans_of_pass) = (spans(state)?, spans(pass)?);
    let overlap = |(_, a, x): (u32, u32, u32), (_, b, y): (u32, u32, u32)| a < b + y && b < a + x;
    let mut placed = BTreeMap::new();
    let mut dropped = BTreeSet::new();
    for &color in &spans_of_state {
        let (index, offset, size) = color;
        let over: Vec<(u32, u32, u32)> = spans_of_pass
            .iter()
            .copied()
            .filter(|&other| overlap(color, other))
            .collect();
        match over.as_slice() {
            [] => {
                dropped.insert(index);
            }
            [(other, at, length)] if *at == offset && (*length == size || *other == index) => {
                placed.insert(index, *other);
            }
            _ => return None,
        }
    }
    let mut at: BTreeMap<u32, u32> = placed.clone();
    let mut uncovered = BTreeMap::new();
    let taken = |location: u32| {
        state
            .get(location as usize)
            .copied()
            .flatten()
            .is_some_and(|pf| pf != PixelFormat::Invalid)
    };
    for &color in &spans_of_pass {
        let index = color.0;
        if placed.values().any(|&k| k == index) {
            continue;
        }
        if spans_of_state.iter().any(|&other| overlap(color, other)) {
            return None;
        }
        let free = |location: u32, at: &BTreeMap<u32, u32>| {
            !taken(location) && !at.contains_key(&location)
        };
        let location = match free(index, &at) {
            true => index,
            false => (0..LOCATIONS).rev().find(|&l| free(l, &at))?,
        };
        at.insert(location, index);
        uncovered.insert(location, index);
    }
    let locations = match at.iter().all(|(location, index)| location == index) {
        true => Vec::new(),
        false => {
            let length = at.keys().max().map_or(0, |last| last + 1);
            (0..length)
                .map(|location| at.get(&location).copied())
                .collect()
        }
    };
    Some(Routing {
        locations,
        placed,
        uncovered,
        dropped,
    })
}

/// What plain Metal stores into a pass color at no state color's place where a draw covers it
/// (Metal probes, an M4): zero where the draw's compiled code writes its whole per-sample record
/// without reading the attachments, their contents otherwise.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Uncovered {
    /// Zero: each state color writes every channel of its format (its write mask) and reads no
    /// attachment (blending off, or no factor reading the destination and destination factors
    /// of zero).
    Zeroed,
    /// Its contents: a state color writes some of its channels alone, or blends reading the
    /// destination whatever the fragment's output (a destination factor of one, of the
    /// destination or of the blend color, a source factor of the destination, `Min` or `Max`).
    Kept,
    /// Its contents, where a state color's only reads of the destination are through a
    /// destination factor of the fragment's output (`SourceAlpha`, `OneMinusSourceAlpha`, ...):
    /// Metal's compiler folds an output it proves constant, and where the factor folds to zero
    /// it stores zero (an alpha of the literal 1 under `OneMinusSourceAlpha`); the same 1 read
    /// from a buffer kept them. A near-equivalent.
    KeptUnlessFolded,
}

/// What a pass color at no place of a state whose colors are `colors` holds where a draw covers
/// it ([`Uncovered`]).
pub(crate) fn uncovered<'a>(
    colors: impl IntoIterator<Item = &'a ColorAttachmentDescriptor>,
) -> Uncovered {
    let mut out = Uncovered::Zeroed;
    for color in colors {
        if color.pixel_format == PixelFormat::Invalid {
            continue;
        }
        let rule = color_reads(color);
        if rule == Uncovered::Kept {
            return Uncovered::Kept;
        }
        if rule == Uncovered::KeptUnlessFolded {
            out = rule;
        }
    }
    out
}

/// Whether state color `color` writes its record place without reading the attachment
/// ([`Uncovered::Zeroed`]), reads it ([`Uncovered::Kept`]), or reads it unless its output folds
/// ([`Uncovered::KeptUnlessFolded`]).
fn color_reads(color: &ColorAttachmentDescriptor) -> Uncovered {
    // Metal 4's blend placeholders resolve to their slot's own default, which is the factor and
    // the operation the draw blends with, so what the attachment reads is decided from the
    // resolved blend rather than from the placeholder ([`ColorAttachmentDescriptor::specialized`]).
    let color = &color.specialized();
    let Some(channels) = has_channels(color.pixel_format) else {
        return Uncovered::Kept;
    };
    let masks = [
        ColorWriteMask::RED,
        ColorWriteMask::GREEN,
        ColorWriteMask::BLUE,
        ColorWriteMask::ALPHA,
    ];
    let whole = channels
        .iter()
        .zip(masks)
        .all(|(&has, mask)| !has || color.write_mask.contains(mask));
    if !whole || color.write_mask.contains(ColorWriteMask::UNSPECIALIZED) {
        return Uncovered::Kept;
    }
    if !color.blending_enabled {
        return Uncovered::Zeroed;
    }
    use BlendFactor as F;
    let reads_destination = |factor: BlendFactor| {
        matches!(
            factor,
            F::DestinationColor
                | F::OneMinusDestinationColor
                | F::DestinationAlpha
                | F::OneMinusDestinationAlpha
                | F::SourceAlphaSaturated
        )
    };
    let operations = [color.rgb_blend_operation, color.alpha_blend_operation];
    let sources = [
        color.source_rgb_blend_factor,
        color.source_alpha_blend_factor,
    ];
    let destinations = [
        color.destination_rgb_blend_factor,
        color.destination_alpha_blend_factor,
    ];
    let reading_operation = operations
        .iter()
        .any(|op| matches!(op, BlendOperation::Min | BlendOperation::Max));
    if reading_operation || sources.into_iter().any(reads_destination) {
        return Uncovered::Kept;
    }
    let mut out = Uncovered::Zeroed;
    for factor in destinations {
        match factor {
            F::Zero => {}
            F::SourceColor
            | F::OneMinusSourceColor
            | F::SourceAlpha
            | F::OneMinusSourceAlpha
            | F::Source1Color
            | F::OneMinusSource1Color
            | F::Source1Alpha
            | F::OneMinusSource1Alpha => out = Uncovered::KeptUnlessFolded,
            _ => return Uncovered::Kept,
        }
    }
    out
}

/// The placed colors of `formats`' record: each one's index, byte offset and size.
fn spans(formats: &[Option<PixelFormat>]) -> Option<Vec<(u32, u32, u32)>> {
    let placed = (0u32..).zip(color_record(formats)?);
    Some(
        placed
            .filter_map(|(index, span)| span.map(|(offset, size)| (index, offset, size)))
            .collect(),
    )
}

/// The format of the view through which a draw stores a state's color held in `state` into a
/// pass attachment held in `pass` as plain Metal stores it, both of one texel size (the module
/// docs): the state's encoding in red-first order, in the attachment's channel order. `None`
/// where no Vulkan format stores it so: the attachment's channel order is not red-first and the
/// state's encoding has no format in that order (a `BGRA8Unorm` pass of anything but four 8-bit
/// channels, a `BGR10A2Unorm` pass of anything but `RGB10A2Unorm`), the attachment stores bytes
/// other than the record's (a signed normalized attachment stores -128 as -127, a probe
/// `n_rgba8_in_rgba8snorm`), a packed 16-bit format, which Metal's record holds otherwise
/// (probe 2), or texels of different sizes (unbuilt).
pub(crate) fn view_format(state: Format, pass: Format) -> Option<Format> {
    let block = |format: Format| format.block();
    let (state_block, pass_block) = (block(state)?, block(pass)?);
    let packed_16 = |b: metal2vulkan::vk::FormatBlock| b.packed_bits == Some(16);
    if state_block.size != pass_block.size
        || packed_16(state_block)
        || packed_16(pass_block)
        || pass_block.color == Some(NumericFormat::Snorm)
    {
        return None;
    }
    let encoding = red_first(state);
    match pass {
        Format::B8G8R8A8_UNORM | Format::B8G8R8A8_SRGB => match encoding {
            Format::R8G8B8A8_UNORM => Some(Format::B8G8R8A8_UNORM),
            Format::R8G8B8A8_SRGB => Some(Format::B8G8R8A8_SRGB),
            Format::R8G8B8A8_SNORM => Some(Format::B8G8R8A8_SNORM),
            Format::R8G8B8A8_UINT => Some(Format::B8G8R8A8_UINT),
            Format::R8G8B8A8_SINT => Some(Format::B8G8R8A8_SINT),
            _ => None,
        },
        Format::A2R10G10B10_UNORM_PACK32 => {
            (encoding == Format::A2B10G10R10_UNORM_PACK32).then_some(pass)
        }
        _ => Some(encoding),
    }
}

/// Metal's encoding of a color held in `format`, in red-first order: the format of the same
/// layout with red and blue in their red-first places.
fn red_first(format: Format) -> Format {
    match format {
        Format::B8G8R8A8_UNORM => Format::R8G8B8A8_UNORM,
        Format::B8G8R8A8_SRGB => Format::R8G8B8A8_SRGB,
        Format::A2R10G10B10_UNORM_PACK32 => Format::A2B10G10R10_UNORM_PACK32,
        other => other,
    }
}
