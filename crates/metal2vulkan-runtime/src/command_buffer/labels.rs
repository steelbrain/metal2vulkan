//! The labels a program gives a command buffer and its encoders (`MTLCommandBuffer.label`,
//! `MTLCommandEncoder.label`), and the command buffer as a notice names it with them: the
//! recovery from a lost device names the command buffer it failed as the hang so
//! (`device/line.rs`). An encoder's label is its command buffer's open encoder's until it ends:
//! a command buffer has one encoder open at a time, and the record of an ended one keeps it
//! ([`CommandBuffer::ended_encoders`]).

use super::{CommandBuffer, State};
use crate::EncoderKind;
use std::fmt::Write;

/// The ended encoders a command buffer's naming lists, at most: it counts the rest.
const NAMED_ENCODERS: usize = 32;

impl CommandBuffer {
    /// `label`: the label [`set_label`](Self::set_label) gave it, or `None`, as Metal's is `nil`
    /// until set.
    pub fn label(&self) -> Option<String> {
        self.state().label.clone()
    }

    /// `setLabel:`: label it `label`, at any time, committed or not, as Metal's. The runtime's
    /// notices that name the command buffer state it: the recovery from a lost device names the
    /// command buffer it failed as the hang by its label, with its encoders' kinds and labels
    /// (`Degradation::DeviceRecovered`).
    pub fn set_label(&self, label: &str) {
        self.state().label = Some(label.to_owned());
    }

    /// Label its open encoder `label` (an encoder's `set_label`).
    pub(crate) fn label_open_encoder(&self, label: &str) {
        self.state().open_label = Some(label.to_owned());
    }

    /// The label of its open encoder (an encoder's `label`).
    pub(crate) fn open_encoder_label(&self) -> Option<String> {
        self.state().open_label.clone()
    }
}

impl State {
    /// The command buffer as a notice names it, of the queue labeled `queue` where it has a label:
    /// its label and its queue's, then each of its first [`NAMED_ENCODERS`] ended encoders by its
    /// place, its kind and its label, and the count of the rest.
    pub(crate) fn named(&self, queue: Option<&str>) -> String {
        let mut named = match &self.label {
            Some(label) => format!("command buffer `{label}`"),
            None => "a command buffer".to_owned(),
        };
        if let Some(queue) = queue {
            let _ = write!(named, " of queue `{queue}`");
        }
        let count = self.ended.len();
        if count == 0 {
            named.push_str(" (no encoder ended)");
            return named;
        }
        let plural = if count == 1 { "" } else { "s" };
        let _ = write!(named, " (its {count} encoder{plural}:");
        for (place, record) in self.ended.iter().enumerate().take(NAMED_ENCODERS) {
            let kind = match record.kind {
                EncoderKind::Blit => "blit",
                EncoderKind::Compute => "compute",
                EncoderKind::Render => "render",
                EncoderKind::ResourceState => "resource state",
            };
            let comma = if place == 0 { "" } else { "," };
            let _ = write!(named, "{comma} {place} {kind}");
            if let Some(label) = &record.label {
                let _ = write!(named, " `{label}`");
            }
        }
        if count > NAMED_ENCODERS {
            let _ = write!(named, ", and {} more", count - NAMED_ENCODERS);
        }
        named.push(')');
        named
    }
}
