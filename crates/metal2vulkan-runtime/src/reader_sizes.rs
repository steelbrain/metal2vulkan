//! The sizes of the draws that read their pass's texture through a snapshot or take the split by
//! the area rule (`render/snapshot.rs`'s `RenderCommandEncoder::snapshot_cheaper`), counted for a
//! diagnostic line where `METAL2VULKAN_LOG_PIPELINES` is set: how many such draws weigh which share
//! of their colors' pixels, at which texel bytes, on which side of the rule. A device counts them
//! only while the log is on, writes the line after 64 of them and after each power of two past
//! it, and once more as it ends where it counted one since; the line changes nothing it records.
//!
//! A draw's share is its scissors' pixels for each color it reads so, over the pixels of those
//! colors' levels, in steps of 0.5% (one step at 100% and above); its texel bytes are those
//! colors' bytes over their pixels, rounded down: a color's own format's bytes for a draw that
//! reads one color, or colors of one width. The rule's bound on the share
//! (`render/snapshot.rs`'s area rule) turns on the texel bytes and on the colors' size, so the
//! line shows how many draws lie near it for each width.

use std::collections::BTreeMap;
use std::fmt::Write;

/// The steps of a share: 0.5% each, the last one 100% and above.
const STEPS: u64 = 200;

/// The fewest weighed draws a line is written after, and each power of two from it.
const FIRST_LINE: u64 = 64;

/// The draws a device weighed by the area rule, by texel bytes, side and share.
#[derive(Debug, Default)]
pub(crate) struct ReaderSizes {
    weighed: u64,
    /// The count of each texel bytes, side of the rule (`true` within, so the snapshot) and
    /// share step.
    counts: BTreeMap<(u64, bool, u64), u64>,
}

impl ReaderSizes {
    /// Count a draw whose scissors bound `covered` pixels over all the colors it reads so
    /// (the scissors' pixels times the colors), of colors of `pixels` pixels and `bytes` bytes in
    /// all, `within` the rule or not; the diagnostic line, where the count of weighed draws
    /// reaches [`FIRST_LINE`] or a power of two past it.
    pub(crate) fn add(
        &mut self,
        covered: u64,
        pixels: u64,
        bytes: u64,
        within: bool,
    ) -> Option<String> {
        let step = match pixels {
            0 => STEPS,
            p => (covered.saturating_mul(STEPS) / p).min(STEPS),
        };
        let texel = bytes.checked_div(pixels).unwrap_or(0);
        *self.counts.entry((texel, within, step)).or_default() += 1;
        self.weighed += 1;
        (self.weighed >= FIRST_LINE && self.weighed.is_power_of_two()).then(|| self.line())
    }

    /// The line a device writes as it ends, where it weighed a draw since its last line.
    pub(crate) fn last_line(&self) -> Option<String> {
        let written = self.weighed >= FIRST_LINE && self.weighed.is_power_of_two();
        (self.weighed > 0 && !written).then(|| self.line())
    }

    /// The diagnostic line: the draws weighed, then for each texel bytes and side, each share
    /// step's count.
    pub(crate) fn line(&self) -> String {
        let mut text = format!(
            "{} draws that read their pass's texture weighed by the area rule (texel bytes, side, \
             then share of the colors' pixels the scissors bound, from, in 0.5% steps: count):",
            self.weighed
        );
        let mut group = None;
        for (&(texel, within, step), &count) in &self.counts {
            if group != Some((texel, within)) {
                let side = if within { "snapshot" } else { "split" };
                let gap = if group.is_some() { ";" } else { "" };
                let _ = write!(text, "{gap} {texel} B {side}");
                group = Some((texel, within));
            }
            let label = match step {
                STEPS => "100%+".to_string(),
                s => format!("{}.{}%", s / 2, (s % 2) * 5),
            };
            let _ = write!(text, " {label}:{count}");
        }
        text
    }
}
