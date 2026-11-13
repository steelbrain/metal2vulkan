//! The encoder trail: a default-off diagnostic, `METAL2VULKAN_ENCODER_TRAIL`
//! (`metal2vulkan::env_vars`), or `DeviceOptions::encoder_trail` for one device, that names the
//! encoder a device's GPU stopped in, and the last one that completed before it.
//!
//! - **Each encoder its own submission.** On a device with the trail on, each encoder's
//!   recording, and each part of one that a host step splits off, ends its batch
//!   (`command_buffer/batches.rs`, `Batches::alone`), and no batch waits in a group
//!   (`command_buffer/group.rs`): each goes in a `vkQueueSubmit` of its own, which signals its
//!   queue's timeline, so the value the timeline reached names the last encoder that completed.
//!   That changes when work goes to the driver, never what it computes; its timing is not the
//!   default's.
//! - **What an encoder was** ([`Notes`]): its kind, its command buffer's object ID and its place
//!   among the command buffer's ended encoders; a render pass's area, its attachments' formats and
//!   sizes, its Vulkan render pass instances and whether it is over a rasterization-rate map; its
//!   executions of indirect commands; and the label hashes of the pipeline states it bound, each
//!   once, in the order it first did, at most [`KEPT`]: the first 8 bytes of the SHA-256 of a
//!   state's label and its functions' names ([`label_hash`]). No name and no shader text.
//! - **Host steps** ([`Step`]). A host step that records an ended encoder's segment
//!   (`CommandBuffer::record_host_step`) notes the states its indirect commands name, as the GPU
//!   and the CPU left the records, and its batch names them after the encoder's own. A submission
//!   the step makes and waits for itself (a rasterization-rate map's fan-out probe,
//!   `render/fan_out.rs`, or a predicated loop's check run again, `command_buffer/rerun.rs`)
//!   signals no timeline: once it ran for [`AFTER`], one line names it ([`waited_line`]).
//! - **The line** (`command_buffer/completion.rs`'s `stalled`). Once no work of the device
//!   advanced for [`AFTER`] while some was pending, one line for each `VkQueue` whose oldest batch
//!   is pending, through the device's message sink at `Info` (id name [`ID_NAME`]), else on
//!   stderr: what that batch's encoders were, and what the last batch the queue completed held,
//!   and how long before. Once a stall: a timeline that advances ends it.
//! - **Off, it costs nothing**: the switch is read once a device is made; an encoder notes
//!   nothing and no line is formatted.

use crate::pipeline::{ComputePipelineState, RenderPipelineState};
use crate::render_pass::PassLayout;
use crate::used_resources::EncoderKind;
use sha2::{Digest, Sha256};
use std::fmt::Write as _;
use std::sync::Arc;
use std::time::Duration;

/// How long no work of a device advances, with some pending, before its trail's lines; and how
/// long a submission the host waits for runs before its line.
pub(crate) const AFTER: Duration = Duration::from_secs(2);

/// The most pipeline states a line names of one encoder.
const KEPT: usize = 16;

/// The id name of the trail's lines.
pub(crate) const ID_NAME: &str = "metal2vulkan-encoder-trail";

/// What an encoder did, noted while the trail is on (the module docs).
#[derive(Default)]
pub(crate) struct Notes {
    /// The label hash of each pipeline state it bound, once, in the order it first did.
    states: Vec<String>,
    /// The states it bound past [`KEPT`].
    more: usize,
    /// Its executions of indirect commands.
    executions: u64,
    /// The commands of those executions' ranges given in the call.
    commands: u64,
    /// The executions whose range a buffer holds.
    indirect: u64,
    /// A render pass's shape, noted at its end ([`pass_named`]).
    pass: Option<String>,
}

impl Notes {
    /// It bound a pipeline state labeled `label`, of `functions`.
    pub(crate) fn state(&mut self, label: Option<&str>, functions: &[&str]) {
        self.hashed(label_hash(label, functions));
    }

    /// It bound `state`, a render pipeline state.
    pub(crate) fn render(&mut self, state: &RenderPipelineState) {
        let names = state.function_names();
        let names: Vec<&str> = names.iter().map(String::as_str).collect();
        self.state(state.label(), &names);
    }

    /// It bound `state`, a compute pipeline state.
    pub(crate) fn compute(&mut self, state: &ComputePipelineState) {
        self.state(state.label(), &[state.function_name()]);
    }

    /// It bound a state of label hash `hash`.
    fn hashed(&mut self, hash: String) {
        if self.states.contains(&hash) {
            return;
        }
        match self.states.len() < KEPT {
            true => self.states.push(hash),
            false => self.more += 1,
        }
    }

    /// It executed indirect commands: `commands` of them, where the call gave the range.
    pub(crate) fn execution(&mut self, commands: Option<u64>) {
        self.executions += 1;
        match commands {
            Some(commands) => self.commands += commands,
            None => self.indirect += 1,
        }
    }

    /// Its render pass, as [`pass_named`] names it.
    pub(crate) fn pass(&mut self, pass: String) {
        self.pass = Some(pass);
    }

    /// The encoder of `kind`, at place `encoder` among the ended encoders of the command buffer
    /// of object ID `command_buffer`, as a line names it.
    pub(crate) fn described(
        &self,
        kind: EncoderKind,
        command_buffer: u64,
        encoder: usize,
    ) -> String {
        let mut text = format!("{kind:?} encoder {encoder} of command buffer {command_buffer}");
        if let Some(pass) = &self.pass {
            let _ = write!(text, ", pass {pass}");
        }
        if self.executions > 0 {
            let _ = write!(
                text,
                ", {} execution(s) of indirect commands ({} commands in ranges given, {} ranges \
                 a buffer holds)",
                self.executions, self.commands, self.indirect
            );
        }
        let more = match self.more {
            0 => String::new(),
            more => format!(" and {more} more"),
        };
        let _ = write!(text, ", states [{}]{more}", self.states.join(" "));
        text
    }
}

/// What a host step noted as it recorded an ended encoder's segment (the module docs, **Host
/// steps**): the encoder as it ended, and the states its commands named.
pub(crate) struct Step {
    encoder: Option<Arc<str>>,
    states: Notes,
}

impl Step {
    /// A step of the encoder `encoder` names, as it ended.
    pub(crate) fn new(encoder: Option<Arc<str>>) -> Step {
        Step {
            encoder,
            states: Notes::default(),
        }
    }

    /// Its commands named the states of `notes`.
    pub(crate) fn noted(&mut self, notes: Notes) {
        for hash in notes.states {
            self.states.hashed(hash);
        }
        self.states.more += notes.more;
    }

    /// Whether its commands named any state.
    fn named(&self) -> bool {
        !self.states.states.is_empty() || self.states.more > 0
    }

    /// The encoder, and the states its commands named where they named any, as a line names it.
    pub(crate) fn described(&self) -> String {
        let encoder = self
            .encoder
            .as_deref()
            .unwrap_or("an encoder noted nothing of");
        if !self.named() {
            return encoder.to_owned();
        }
        let more = match self.states.more {
            0 => String::new(),
            more => format!(" and {more} more"),
        };
        let states = self.states.states.join(" ");
        format!("{encoder}, its host step's commands' states [{states}]{more}")
    }

    /// The encoder's name after the step: [`described`](Self::described) where its commands
    /// named a state, else `None` (it stays as it ended).
    pub(crate) fn stepped(&self) -> Option<Arc<str>> {
        self.named().then(|| Arc::from(self.described()))
    }
}

/// A pipeline state's label hash: the first 8 bytes, in hex, of the SHA-256 of its label and
/// its functions' names, each after its length.
pub(crate) fn label_hash(label: Option<&str>, functions: &[&str]) -> String {
    let mut hash = Sha256::new();
    hash.update(b"metal2vulkan encoder trail state\0");
    match label {
        Some(label) => {
            hash.update([1]);
            hash.update((label.len() as u64).to_le_bytes());
            hash.update(label.as_bytes());
        }
        None => hash.update([0]),
    }
    for function in functions {
        hash.update((function.len() as u64).to_le_bytes());
        hash.update(function.as_bytes());
    }
    let digest = hash.finalize();
    digest[..8].iter().fold(String::new(), |mut text, byte| {
        let _ = write!(text, "{byte:02x}");
        text
    })
}

/// A render pass over `layout`, recorded in `instances` Vulkan render pass instances, over a
/// rasterization-rate map where `rate_mapped`, as a line names it: its area, each color
/// attachment's format and size, and the depth/stencil attachment's.
pub(crate) fn pass_named(layout: &PassLayout, instances: usize, rate_mapped: bool) -> String {
    let [width, height] = layout.extent;
    let colors: Vec<String> = (layout.colors.iter().enumerate())
        .filter_map(|(index, color)| {
            let texture = &color.as_ref()?.target.texture;
            Some(format!(
                "{index}: {:?} {}x{}",
                texture.pixel_format(),
                texture.width(),
                texture.height()
            ))
        })
        .collect();
    let depth_stencil = match layout.depth_stencil.as_ref().filter(|ds| !ds.hidden) {
        Some(ds) => format!(
            "{:?} {}x{}",
            ds.target.texture.pixel_format(),
            ds.target.texture.width(),
            ds.target.texture.height()
        ),
        None => "none".to_owned(),
    };
    let map = match rate_mapped {
        true => ", over a rasterization-rate map",
        false => "",
    };
    format!(
        "{width}x{height}, {} sample(s), colors [{}], depth/stencil {depth_stencil}, \
         {instances} Vulkan render pass instance(s){map}",
        layout.samples,
        colors.join(", ")
    )
}

/// The line of `queue`'s stall of `stalled` (the module docs): `front`, its oldest pending batch's
/// encoders as noted, and `last`, the last batch it completed, with how long before.
pub(crate) fn line(
    queue: usize,
    stalled: Duration,
    front: Option<&str>,
    last: Option<(&str, Duration)>,
) -> String {
    let last = match last {
        Some((last, ago)) => format!("{last} ({ago:?} before)"),
        None => "none since the device was made".to_owned(),
    };
    format!(
        "metal2vulkan: encoder trail: no work of the device advanced for {stalled:?} with some \
         pending; VkQueue {queue}'s oldest pending batch: {}; the last batch it completed: {last}",
        front.unwrap_or("no encoder noted"),
    )
}

/// The line of `what`, a submission on `queue` the host waits for itself (the module docs, **Host
/// steps**), which has run for `ran`; `last`, the last batch the queue completed, with how long
/// before.
pub(crate) fn waited_line(
    queue: usize,
    what: &str,
    ran: Duration,
    last: Option<(&str, Duration)>,
) -> String {
    let last = match last {
        Some((last, ago)) => format!("{last} ({ago:?} before)"),
        None => "none since the device was made".to_owned(),
    };
    format!(
        "metal2vulkan: encoder trail: a submission the host waits for has run for {ran:?} on \
         VkQueue {queue}: {what}; the last batch it completed: {last}"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A state's hash covers its label and each function's name apart, and an encoder names each
    /// state once, in the order bound, past the kept ones as a count.
    #[test]
    fn an_encoder_names_each_state_once_by_its_label_hash() {
        let a = label_hash(Some("a"), &["v", "f"]);
        // The hash a line names a state by, which a reader of a run's pipeline creation log
        // computes again from the state's label and functions' names.
        assert_eq!(a, "feece2ca5cb278a8");
        assert_ne!(a, label_hash(Some("a"), &["vf"]));
        assert_ne!(a, label_hash(None, &["v", "f"]));
        assert_eq!(a, label_hash(Some("a"), &["v", "f"]));
        let mut notes = Notes::default();
        notes.state(Some("a"), &["v", "f"]);
        notes.state(Some("a"), &["v", "f"]);
        for i in 0..KEPT + 2 {
            notes.state(None, &[&i.to_string()]);
        }
        notes.execution(Some(8162));
        notes.execution(None);
        let text = notes.described(EncoderKind::Render, 7, 3);
        assert!(
            text.starts_with("Render encoder 3 of command buffer 7"),
            "{text}"
        );
        assert!(text.contains(&format!("states [{a} ")), "{text}");
        assert!(text.contains(" and 3 more"), "{text}");
        assert!(
            text.contains("2 execution(s) of indirect commands (8162 commands in ranges given, 1"),
            "{text}"
        );
        assert_eq!(text.matches(&a).count(), 1, "{text}");
        // A host step's commands' states follow the encoder's, each once.
        let mut step = Step::new(Some(Arc::from("Render encoder 3 of command buffer 7")));
        assert!(step.stepped().is_none());
        let mut named = Notes::default();
        named.state(Some("b"), &["v"]);
        step.noted(named);
        let mut again = Notes::default();
        again.state(Some("b"), &["v"]);
        step.noted(again);
        let b = label_hash(Some("b"), &["v"]);
        let stepped = step.stepped().expect("a state named");
        assert_eq!(
            &*stepped,
            format!("Render encoder 3 of command buffer 7, its host step's commands' states [{b}]")
        );
    }
}
