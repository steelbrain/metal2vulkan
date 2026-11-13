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
//!   and the CPU left the records, and what it drew of them ([`Drawn`]): each range it read from a
//!   buffer, the draws it recorded, their vertices or indices times their instances, the draw of
//!   the most of those, the route each draw took ([`Route`]: plain, a rate-mapped pass's probe,
//!   fanned out or region by region), the executions it recorded in their GPU form, and the kept
//!   chunks it ran again (`render/executions/kept.rs`), whose draws it did not read. Its batch
//!   names them after the encoder's own; the line names the device's pipeline pool's makings
//!   waiting and under way.
//! - **GPU forms** ([`GpuForm`]). An execution recorded in its GPU form, at `commit` or by a host
//!   step (`render/executions/gpu.rs`), whose records the host never reads, is named by its
//!   shape: merged into one draw or a draw per command, indexed or not, its primitive type, the
//!   commands its preprocess runs over and where its range comes from. Where its batch is the
//!   oldest pending one, the line reads the scratch the GPU writes, as it stands while the work
//!   is pending: the range the GPU copied, the merged draw's two `VkDrawIndirectCommand`s, the
//!   commands with arguments and their vertices or indices times instances, the most of them,
//!   and the command buffer's execution status words. It reads host-visible, host-coherent
//!   memory the pending command buffer holds (`upload.rs`), already mapped, and only reads it: a
//!   word the GPU writes meanwhile may read either value. It takes no lock and waits for
//!   nothing. An encoder encoded again (`command_buffer/reencode.rs`) names none of its own.
//! - A submission
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
use crate::upload::Upload;
use crate::used_resources::EncoderKind;
use metal2vulkan::bind::DrawReplayLayout;
use metal2vulkan::metal::PrimitiveType;
use sha2::{Digest, Sha256};
use std::fmt::Write as _;
use std::sync::Arc;
use std::time::Duration;

/// How long no work of a device advances, with some pending, before its trail's lines; and how
/// long a submission the host waits for runs before its line.
pub(crate) const AFTER: Duration = Duration::from_secs(2);

/// The most pipeline states a line names of one encoder.
const KEPT: usize = 16;

/// The most commands of a GPU form's arguments the line reads.
const READ_COMMANDS: u32 = 1 << 20;

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
    /// What a host step drew of its commands.
    drawn: Drawn,
    /// The executions recorded in their GPU form.
    forms: Vec<Arc<GpuForm>>,
}

/// What a host step drew of an execution's commands (the module docs, **Host steps**).
#[derive(Default)]
pub(crate) struct Drawn {
    /// Each range it read from a buffer, at most [`KEPT`]: its location and length as the bytes
    /// held them, and the commands it ran of them.
    ranges: Vec<([u32; 2], std::ops::Range<u32>)>,
    /// The ranges read past [`KEPT`].
    more_ranges: usize,
    /// The draws it recorded.
    draws: u64,
    /// Their vertices or indices times their instances, summed.
    work: u64,
    /// The draw of the most of those: whether indexed, its primitive type's raw value, its
    /// vertices or indices, and its instances.
    largest: Option<(bool, u32, u32, u32)>,
    /// The kept chunks it ran again, and their commands.
    kept: u64,
    kept_commands: u64,
    /// Its draws by the route they took, in [`Route`]'s order.
    routes: [u64; 4],
    /// Its executions recorded in their GPU form (`render/executions/gpu.rs`), whose records the
    /// host did not read.
    gpu_alone: u64,
}

/// The route a host step's draw took: its pipeline's variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Route {
    /// Outside a rate-mapped pass: the state's own variant.
    Plain,
    /// A rate-mapped pass's fan-out probe (`VariantKey::region_fan_out`, `Probe`).
    Probe,
    /// A rate-mapped pass's draw fanned out (`region_fan_out`, `Draw`).
    FannedOut,
    /// A rate-mapped pass's draw recorded region by region.
    RegionByRegion,
}

impl Drawn {
    fn join(&mut self, other: Drawn) {
        for range in other.ranges {
            match self.ranges.len() < KEPT {
                true => self.ranges.push(range),
                false => self.more_ranges += 1,
            }
        }
        self.more_ranges += other.more_ranges;
        self.draws += other.draws;
        self.work = self.work.saturating_add(other.work);
        if let Some(largest) = other.largest {
            self.largest(largest);
        }
        self.kept += other.kept;
        self.kept_commands += other.kept_commands;
        for (route, more) in self.routes.iter_mut().zip(other.routes) {
            *route += more;
        }
        self.gpu_alone += other.gpu_alone;
    }

    fn largest(&mut self, draw: (bool, u32, u32, u32)) {
        let work = |(_, _, count, instances): (bool, u32, u32, u32)| {
            u64::from(count) * u64::from(instances)
        };
        if self
            .largest
            .is_none_or(|largest| work(largest) < work(draw))
        {
            self.largest = Some(draw);
        }
    }

    fn any(&self) -> bool {
        !self.ranges.is_empty()
            || self.draws > 0
            || self.kept > 0
            || self.routes.iter().any(|&route| route > 0)
            || self.gpu_alone > 0
    }

    /// As a line names it.
    fn described(&self) -> String {
        let ranges: Vec<String> = (self.ranges.iter())
            .map(|([location, length], ran)| {
                format!("{location}+{length} ran {}..{}", ran.start, ran.end)
            })
            .collect();
        let more = match self.more_ranges {
            0 => String::new(),
            more => format!(" and {more} more"),
        };
        let largest = match self.largest {
            Some((indexed, primitive, count, instances)) => format!(
                ", the largest {} of primitive type {primitive}: {count} {} times {instances} \
                 instance(s)",
                if indexed { "an indexed draw" } else { "a draw" },
                if indexed { "indices" } else { "vertices" },
            ),
            None => String::new(),
        };
        let [plain, probe, fanned, regions] = self.routes;
        format!(
            "its host step read ranges [{}]{more}, drew {} draw(s) of {} vertices or indices \
             times instances{largest}, and ran {} kept chunk(s) of {} command(s) again; routes: \
             {plain} plain, {probe} probe, {fanned} fanned out, {regions} region by region, {} \
             execution(s) on the GPU alone",
            ranges.join(", "),
            self.draws,
            self.work,
            self.kept,
            self.kept_commands,
            self.gpu_alone
        )
    }
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

    /// A host step read the range `raw` (location, then length) from a buffer, and ran the
    /// commands `ran` of it.
    pub(crate) fn range_read(&mut self, raw: [u32; 2], ran: std::ops::Range<u32>) {
        match self.drawn.ranges.len() < KEPT {
            true => self.drawn.ranges.push((raw, ran)),
            false => self.drawn.more_ranges += 1,
        }
    }

    /// A host step recorded a draw, `indexed` or not, of primitive type `primitive` (its raw
    /// value), of `count` vertices or indices, times `instances`.
    pub(crate) fn drew(&mut self, indexed: bool, primitive: u32, count: u32, instances: u32) {
        let drawn = &mut self.drawn;
        drawn.draws += 1;
        drawn.work = (drawn.work).saturating_add(u64::from(count) * u64::from(instances));
        drawn.largest((indexed, primitive, count, instances));
    }

    /// A host step's draw took `route`.
    pub(crate) fn routed(&mut self, route: Route) {
        self.drawn.routes[route as usize] += 1;
    }

    /// An execution was recorded in its GPU form `form`.
    pub(crate) fn gpu_form(&mut self, form: GpuForm) {
        self.drawn.gpu_alone += 1;
        self.forms.push(Arc::new(form));
    }

    /// A host step ran a kept chunk of `commands` again.
    pub(crate) fn kept_again(&mut self, commands: u64) {
        self.drawn.kept += 1;
        self.drawn.kept_commands += commands;
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

/// What a batch's encoders (or parts of them) were, as the trail names them: the text, and the
/// executions of theirs recorded in their GPU form (the module docs, **GPU forms**), whose
/// scratch the line reads where the batch is the oldest pending one.
#[derive(Clone)]
pub(crate) struct Named {
    text: Arc<str>,
    forms: Vec<Arc<GpuForm>>,
}

impl Named {
    /// An encoder (or a part of one) named `text`, of no GPU form yet.
    pub(crate) fn new(text: impl Into<Arc<str>>) -> Named {
        Named {
            text: text.into(),
            forms: Vec::new(),
        }
    }

    /// Its text.
    pub(crate) fn text(&self) -> &Arc<str> {
        &self.text
    }

    /// Its executions took the GPU forms `forms` too.
    pub(crate) fn join_forms(&mut self, forms: Vec<Arc<GpuForm>>) {
        self.forms.extend(forms);
    }

    /// `parts`, each noted one of a batch's in order, as one: their texts joined by `; `, and
    /// their forms; `None` where none was noted.
    pub(crate) fn joined<'a>(mut parts: impl Iterator<Item = &'a Named>) -> Option<Named> {
        let first = parts.next()?;
        let Some(second) = parts.next() else {
            return Some(first.clone());
        };
        let mut text = format!("{}; {}", first.text, second.text);
        let mut forms = first.forms.clone();
        forms.extend(second.forms.iter().cloned());
        for more in parts {
            text.push_str("; ");
            text.push_str(&more.text);
            forms.extend(more.forms.iter().cloned());
        }
        Some(Named {
            text: Arc::from(text),
            forms,
        })
    }

    /// Its text, then each GPU form's shape and its scratch as it stands now, its work pending
    /// (the module docs, **GPU forms**).
    pub(crate) fn pending(&self) -> String {
        let mut text = self.text.to_string();
        for form in &self.forms {
            let _ = write!(text, "; {}", form.described());
        }
        text
    }

    /// The text of each of its GPU forms ([`GpuForm::described`]): a device test's view.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn forms(&self) -> Vec<String> {
        self.forms.iter().map(|form| form.described()).collect()
    }
}

/// An execution recorded in its GPU form (`render/executions/gpu.rs`), as the trail names it (the
/// module docs, **GPU forms**): its shape, and where its scratch lies.
pub(crate) struct GpuForm {
    /// Recorded at `commit`, else by a host step.
    at_commit: bool,
    /// Indexed draws among its commands, merged into one draw alone.
    indexed: bool,
    primitive_type: PrimitiveType,
    /// The first command its preprocess runs over (the range given; 0 for one read from a
    /// buffer, whose every command it runs over).
    first: u32,
    /// The offset in its buffer of the range read from one; `None` for a range given.
    range_from: Option<u64>,
    /// Its scratch (`bind::DrawReplayLayout`), and the command buffer's execution status words.
    scratch: Upload,
    layout: DrawReplayLayout,
    status: Upload,
}

impl GpuForm {
    /// The GPU form of an execution of commands from `first`, drawing `primitive_type`, `indexed`
    /// or not, its range read at `range_from` of a buffer or given, recorded `at_commit` or by a
    /// host step, into `scratch` of `layout`, with the command buffer's status words `status`.
    pub(crate) fn new(
        (at_commit, indexed, primitive_type): (bool, bool, PrimitiveType),
        (first, range_from): (u32, Option<u64>),
        (scratch, layout, status): (Upload, DrawReplayLayout, Upload),
    ) -> GpuForm {
        GpuForm {
            at_commit,
            indexed,
            primitive_type,
            first,
            range_from,
            scratch,
            layout,
            status,
        }
    }

    /// Its shape, then its scratch's words as they stand now: the range the GPU copied, the
    /// merged draw's two `VkDrawIndirectCommand`s, the commands whose arguments draw and their
    /// vertices or indices times instances, the most of them (with its index entry's whole indices
    /// and index size where indexed), and the command buffer's status words. Each word read once,
    /// as it stood (`Upload::glimpse`).
    pub(crate) fn described(&self) -> String {
        let layout = &self.layout;
        let when = match self.at_commit {
            true => "at commit",
            false => "by its host step",
        };
        let merged = match layout.merged {
            Some(_) => "merged into one draw",
            None => "a draw per command",
        };
        let indexed = match self.indexed {
            true => ", indexed",
            false => "",
        };
        let range = match self.range_from {
            Some(offset) => format!("its range read from a buffer at offset {offset}"),
            None => "its range given".to_owned(),
        };
        let mut text = format!(
            "an execution's GPU form, recorded {when}: {merged}{indexed}, of primitive type {:?},              over {} command(s) from {}, {range}; read while the work is pending:",
            self.primitive_type, layout.commands, self.first
        );
        if self.range_from.is_some() {
            let [location, length] = word_pair(&self.scratch.glimpse(layout.range_offset(), 2));
            let _ = write!(text, " the range copied {location}+{length},");
        }
        if let Some(merged) = layout.merged {
            let draws = self.scratch.glimpse(merged.draw_offset(0), 8);
            let draw = |words: &[u32]| match words {
                [vertices, instances, first, base] => {
                    format!(
                        "{vertices} vertices x {instances} from vertex {first}, instance {base}"
                    )
                }
                _ => "unread".to_owned(),
            };
            let _ = write!(
                text,
                " the merged draws [{}; {}],",
                draw(draws.get(..4).unwrap_or_default()),
                draw(draws.get(4..).unwrap_or_default())
            );
        }
        let read = layout.commands.min(READ_COMMANDS);
        let words = (read as usize) * (layout.types as usize) * 4;
        let arguments = self.scratch.glimpse(layout.argument_offset(0, 0), words);
        let (mut drawing, mut work) = (0u64, 0u64);
        let mut most: Option<(u64, u32, [u32; 4])> = None;
        for (at, draw) in arguments.as_chunks::<4>().0.iter().enumerate() {
            let [vertices, instances, first, base] = *draw;
            let draws = u64::from(vertices) * u64::from(instances);
            if draws == 0 {
                continue;
            }
            drawing += 1;
            work = work.saturating_add(draws);
            if most.is_none_or(|(most, ..)| most < draws) {
                let command = at as u32 / layout.types.max(1);
                most = Some((draws, command, [vertices, instances, first, base]));
            }
        }
        let past = match layout.commands > read {
            true => format!(" (of the first {read})"),
            false => String::new(),
        };
        let _ = write!(
            text,
            " {drawing} argument(s) that draw{past}, {work} vertices or indices times instances"
        );
        if let Some((_, command, [vertices, instances, first, base])) = most {
            let _ = write!(
                text,
                ", the most command {command}'s: {vertices} x {instances} from {first}, instance                  {base}"
            );
            if let Some(merged) = layout.merged.filter(|_| self.indexed) {
                let entry = merged.indices_offset + u64::from(command) * 16;
                let entry = self.scratch.glimpse(entry, 4);
                if let [_, _, whole, size] = entry[..] {
                    let _ = write!(text, " ({whole} whole indices of {size} byte(s))");
                }
            }
        }
        let status = self.status.glimpse(0, 3);
        if let [bits, misses, faults] = status[..] {
            let _ = write!(
                text,
                "; status bits {bits:#x}, resident misses {misses}, replay faults {faults:#x}"
            );
        }
        text
    }
}

/// The two words of `words`, zero where they were not read.
fn word_pair(words: &[u32]) -> [u32; 2] {
    match words {
        [a, b, ..] => [*a, *b],
        _ => [0, 0],
    }
}

/// What a host step noted as it recorded an ended encoder's segment (the module docs, **Host
/// steps**), or what `commit` noted as it recorded one (its GPU forms alone): the encoder as it
/// ended, and the states its commands named.
pub(crate) struct Step {
    encoder: Option<Named>,
    states: Notes,
}

impl Step {
    /// A step of the encoder `encoder` names, as it ended.
    pub(crate) fn new(encoder: Option<Named>) -> Step {
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
        self.states.drawn.join(notes.drawn);
        self.states.forms.extend(notes.forms);
    }

    /// The GPU forms its recording took: what `commit` keeps of its notes.
    pub(crate) fn into_forms(self) -> Vec<Arc<GpuForm>> {
        self.states.forms
    }

    /// Whether its commands named any state, or it drew any.
    fn named(&self) -> bool {
        !self.states.states.is_empty() || self.states.more > 0 || self.states.drawn.any()
    }

    /// The encoder, and the states its commands named where they named any, as a line names it.
    pub(crate) fn described(&self) -> String {
        let encoder =
            (self.encoder.as_ref()).map_or("an encoder noted nothing of", |named| &*named.text);
        if !self.named() {
            return encoder.to_owned();
        }
        let more = match self.states.more {
            0 => String::new(),
            more => format!(" and {more} more"),
        };
        let states = self.states.states.join(" ");
        let drawn = match self.states.drawn.any() {
            true => format!("; {}", self.states.drawn.described()),
            false => String::new(),
        };
        format!("{encoder}, its host step's commands' states [{states}]{more}{drawn}")
    }

    /// The encoder's name after the step: [`described`](Self::described) where its commands
    /// named a state, else `None` (it stays as it ended), with the GPU forms it ended with and
    /// those the step's recording took.
    pub(crate) fn stepped(&self) -> Option<Named> {
        self.named().then(|| {
            let mut forms = (self.encoder.as_ref())
                .map(|named| named.forms.clone())
                .unwrap_or_default();
            forms.extend(self.states.forms.iter().cloned());
            Named {
                text: Arc::from(self.described()),
                forms,
            }
        })
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
/// encoders as noted, `last`, the last batch it completed, with how long before, and the device's
/// pipeline pool's makings waiting and under way (`pipeline/pool.rs`).
pub(crate) fn line(
    queue: usize,
    stalled: Duration,
    front: Option<&str>,
    last: Option<(&str, Duration)>,
    (waiting, making): (usize, usize),
) -> String {
    let last = match last {
        Some((last, ago)) => format!("{last} ({ago:?} before)"),
        None => "none since the device was made".to_owned(),
    };
    format!(
        "metal2vulkan: encoder trail: no work of the device advanced for {stalled:?} with some \
         pending; VkQueue {queue}'s oldest pending batch: {}; the last batch it completed: \
         {last}; the device's pipeline pool: {waiting} making(s) waiting, {making} under way",
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
        let mut step = Step::new(Some(Named::new("Render encoder 3 of command buffer 7")));
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
            &**stepped.text(),
            format!("Render encoder 3 of command buffer 7, its host step's commands' states [{b}]")
        );
        // What the step drew follows: the ranges it read, its draws and their work, the draw of
        // the most, and the kept chunks it ran again.
        let mut drawn = Notes::default();
        drawn.range_read([4, 9], 4..10);
        drawn.drew(true, 3, 36, 2);
        drawn.drew(false, 3, 6, u32::MAX);
        drawn.drew(true, 3, 300, 1);
        drawn.kept_again(64);
        drawn.routed(Route::Plain);
        drawn.routed(Route::FannedOut);
        drawn.routed(Route::FannedOut);
        drawn.drawn.gpu_alone += 1;
        step.noted(drawn);
        let stepped = step.stepped().expect("a state named");
        let work = 72 + 6 * u64::from(u32::MAX) + 300;
        assert_eq!(
            &**stepped.text(),
            format!(
                "Render encoder 3 of command buffer 7, its host step's commands' states [{b}]; \
                 its host step read ranges [4+9 ran 4..10], drew 3 draw(s) of {work} vertices \
                 or indices times instances, the largest a draw of primitive type 3: 6 vertices \
                 times {} instance(s), and ran 1 kept chunk(s) of 64 command(s) again; routes: \
                 1 plain, 0 probe, 2 fanned out, 0 region by region, 1 execution(s) on the GPU \
                 alone",
                u32::MAX
            )
        );
        // A step that drew and named no state is named too.
        let mut unnamed = Step::new(None);
        let mut drawn = Notes::default();
        drawn.drew(true, 3, 3, 1);
        unnamed.noted(drawn);
        let stepped = unnamed.stepped().expect("a draw noted");
        let text = stepped.pending();
        assert!(text.contains("drew 1 draw(s) of 3 vertices"), "{text}");
    }
}
