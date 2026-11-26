//! The report's `over copies by frame` line (`docs/REPLAY-STREAM.md`, section 10): the bytes the
//! runtime copied between textures over buffers and their images of their own
//! (`rt::Metrics::over_copy_bytes`), a frame at a time as the play locates its calls
//! (`Backend::locate`), split by the memory class of the buffer's bytes and by the runtime's
//! cause; their mean a frame, and the largest frame's with the number of textures in an image of
//! their own that frame's calls named. Each count is one the runtime already keeps; the textures
//! are those whose making the runtime's `textures_copied_over` gauge rose across, and the views
//! of them.

use super::{objects_named, NewRuntime, Object};
use crate::replay::stream::Call;
use crate::replay::value::Value;
use metal2vulkan_runtime as rt;
use std::collections::{HashMap, HashSet};

/// The parts of a frame's bytes, in the line's order: every over copy's; those over Private and
/// over Shared buffers (the rest are over Managed buffers' device copies); and by cause, a row
/// pitch the linear image does not take, a dma-buf's import, host memory's import, and every
/// other cause's (`own_image`: the device's linear image does not take the texture for another
/// reason, `Metrics::over_copy_bytes_linear` and the rest).
const PARTS: [&str; 7] = [
    "bytes",
    "private",
    "shared",
    "pitch",
    "dma_buf",
    "host_memory",
    "own_image",
];

type Parts = [u64; 7];

/// The cumulative parts of `m`.
fn parts(m: &rt::Metrics) -> Parts {
    let own_image = m.over_copy_bytes_dma_bufs
        + m.over_copy_bytes_dedicated
        + m.over_copy_bytes_surface
        + m.over_copy_bytes_spilled
        + m.over_copy_bytes_linear
        + m.over_copy_bytes_linear_features
        + m.over_copy_bytes_render_reach
        + m.over_copy_bytes_alignment
        + m.over_copy_bytes_image_memory;
    [
        m.over_copy_bytes,
        m.over_copy_bytes_private,
        m.over_copy_bytes_shared,
        m.over_copy_bytes_pitch,
        m.over_copy_bytes_dma_buf,
        m.over_copy_bytes_host_memory,
        own_image,
    ]
}

/// The frames ended: how many, their parts summed, and the largest by bytes (the first of a
/// tie): its number, its parts and the textures its calls named.
#[derive(Clone, Copy, Default)]
struct Summary {
    frames: u64,
    sums: Parts,
    largest: Option<(u64, Parts, u64)>,
}

impl Summary {
    fn add(&mut self, parts: Parts, textures: u64) {
        let frame = self.frames;
        self.frames += 1;
        for (sum, part) in self.sums.iter_mut().zip(parts) {
            *sum += part;
        }
        if self
            .largest
            .is_none_or(|(_, largest, _)| parts[0] > largest[0])
        {
            self.largest = Some((frame, parts, textures));
        }
    }
}

/// The over copies by frame.
#[derive(Default)]
pub(super) struct OverFrames {
    /// Each live texture in an image of its own, and each view of one, by stream id: the texture
    /// whose copies it takes.
    copied: HashMap<u64, u64>,
    /// The textures in an image of their own the open frame's calls named.
    named: HashSet<u64>,
    /// The parts as the open frame began.
    began: Parts,
    /// Whether a call of the open frame was located.
    called: bool,
    summary: Summary,
}

impl OverFrames {
    /// `texture` is made in an image of its own.
    pub(super) fn made(&mut self, texture: u64) {
        self.copied.insert(texture, texture);
    }

    /// `view` is made of `texture`: where that is in an image of its own, the view takes its
    /// copies.
    pub(super) fn viewed(&mut self, texture: u64, view: u64) {
        if let Some(&over) = self.copied.get(&texture) {
            self.copied.insert(view, over);
        }
    }

    /// A call of the open frame named `objects`.
    pub(super) fn named(&mut self, objects: &[u64]) {
        let copied = &self.copied;
        let textures = objects.iter().filter_map(|id| copied.get(id).copied());
        self.named.extend(textures);
    }

    /// `object` is released.
    pub(super) fn released(&mut self, object: u64) {
        self.copied.remove(&object);
    }

    /// A call of the open frame is located.
    pub(super) fn located(&mut self) {
        self.called = true;
    }

    /// The open frame ended, the runtime's metrics `now`.
    pub(super) fn ended(&mut self, now: &rt::Metrics) {
        let now = parts(now);
        let textures = self.named.len() as u64;
        (self.summary).add(grown(now, self.began), textures);
        self.named.clear();
        self.began = now;
        self.called = false;
    }

    /// The line, the runtime's metrics `now`: the open frame counts where a call of it was
    /// located.
    pub(super) fn line(&self, now: &rt::Metrics) -> String {
        let mut summary = self.summary;
        if self.called {
            summary.add(grown(parts(now), self.began), self.named.len() as u64);
        }
        let Some((frame, largest, textures)) = summary.largest else {
            return "over copies by frame: no frame".to_owned();
        };
        let named = |parts: Parts| -> String {
            (PARTS.iter().zip(parts))
                .map(|(name, part)| format!(" {name}={part}"))
                .collect()
        };
        let mean = summary.sums.map(|sum| sum / summary.frames);
        format!(
            "over copies by frame, {} frames as the calls played: mean{}; largest, frame {frame}{}, \
             {textures} textures in an image of their own its calls named",
            summary.frames,
            named(mean),
            named(largest),
        )
    }
}

/// `now`'s growth past `then`, part by part.
fn grown(now: Parts, then: Parts) -> Parts {
    std::array::from_fn(|at| now[at].saturating_sub(then[at]))
}

impl NewRuntime {
    /// Before `call` plays: where it is an encoder's, the textures in an image of their own its
    /// arguments name; where it makes a render encoder, its pass's attachments; where it makes a
    /// view of such a texture, the view, which takes its texture's copies (`OverFrames`).
    pub(super) fn over_named(&mut self, call: &Call) {
        if self.over_frames.copied.is_empty() {
            return;
        }
        let mut named = Vec::new();
        if self.owners.contains_key(&call.receiver) {
            objects_named(&call.args, &mut named);
        } else if &*call.selector == "renderCommandEncoderWithDescriptor:" {
            let pass = call.args.first().and_then(Value::as_object);
            if let Some(Object::Pass(pass)) = pass.and_then(|id| self.objects.get(&id)) {
                named.extend(pass.textures());
            }
        } else if call.selector.starts_with("newTextureView") {
            self.over_frames.viewed(call.receiver, call.result);
        }
        self.over_frames.named(&named);
    }
}
