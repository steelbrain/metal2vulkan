//! The line a failed `vkQueueSubmit` logs: what the submission held, so a failure the driver
//! answers with no more than a `VkResult` (`VK_ERROR_UNKNOWN`, which "is not expected to ever be
//! returned if the application behavior is valid, and if the implementation is bug-free", the
//! Vulkan specification, VkResult Return Codes) can be told apart from the runtime's own misuse.
//!
//! - **When.** Each `vkQueueSubmit` of the runtime's that fails, whatever its `VkResult`: a
//!   batch's, alone or after a group it sent (`command_buffer/group.rs`), a group's, a failed
//!   group's end signalled alone, a predicated loop's run again (`command_buffer/rerun.rs`), a
//!   fence's after a queue's work, an internal submission (`device/submit.rs`), and the staging
//!   and fill queues' (`device/staging.rs`). One line per failed call, at `Warning` severity,
//!   id name [`ID_NAME`], through the device's message sink, else on stderr. It is read and
//!   formatted only once the call failed: a submission that is made does no work for it.
//! - **What it names.** The call's `VkResult`, its `VkQueue` and what submitted; each
//!   `VkSubmitInfo`'s waits, command buffers and signals. Each semaphore by its role (a
//!   `VkQueue`'s timeline or its pieces' timeline, the device's own timelines, an event's, the
//!   caller's binary one), and each timeline's value with the value it holds as the line is made
//!   (`vkGetSemaphoreCounterValue`, never of a binary one:
//!   `VUID-vkGetSemaphoreCounterValue-semaphore-03255`), flagged where the value breaks
//!   `VUID-VkSubmitInfo-pSignalSemaphores-03242` (a signal not above it) or lies further from it
//!   than the least `maxTimelineSemaphoreValueDifference` Vulkan requires, `2^31 - 1`
//!   (`VUID-VkSubmitInfo-pWaitSemaphores-03243`, `VUID-VkSubmitInfo-pSignalSemaphores-03244`).
//!   Each `VkCommandBuffer` by its role where its submitter knows it (an encoder's, a command
//!   buffer of barriers, a kept one submitted again, the first work's, the end's). The Metal
//!   command buffers it carried, as the runtime's notices name them, and the resources their
//!   encoders' uses name, each as what owns its memory (`used_resources.rs`): its kind, size,
//!   storage mode and memory (its type, a no-copy buffer's import, the caller's memory), its age
//!   in the device's submissions since it was made, and whether it is gone, detached or its import
//!   released.
//! - **Bounded.** At most [`NAMED_INFOS`] `VkSubmitInfo`s, [`NAMED_SEMAPHORES`] waits and as
//!   many signals of each, [`NAMED`] `VkCommandBuffer`s, [`NAMED_BUFFERS`] Metal command buffers
//!   and [`NAMED`] resources are named; each list counts the rest.
//! - **Without waiting.** It runs where the submission failed, under the locks its submitter
//!   holds (the scheduler's, a command buffer's state), so it waits for no lock another thread may
//!   hold while it waits for those: a command buffer's state, a no-copy buffer's pages, an
//!   import's residency, an image's detach and the device's memory are read only where free
//!   ([`try_lock`]), the line saying so where one was not; only the registries of the program's
//!   resources and heaps are locked, as a batch's submission locks them. A resource is held only
//!   while it is stated, and a command buffer that names it holds it (`retained.rs`), so its last
//!   hold never goes here.
//! - **A recording's failure too.** A command buffer whose recording failed fails with a raw
//!   `VkResult` too (`VK_ERROR_UNKNOWN` for the runtime's own error), as one whose submission
//!   failed does: its warning (`command_buffer::recording_error`, id name
//!   `metal2vulkan-recording-failed`) names the error it was, which the command buffer's error
//!   does not keep.

use crate::device::DeviceShared;
use crate::exit_guard::{self, Call};
use crate::resource::ObjectId;
use crate::{DebugMessage, Device, MessageKind, MessageSeverity};
use ash::vk;
use ash::vk::Handle;
use std::collections::BTreeSet;
use std::fmt::Write;
use std::sync::{Mutex, MutexGuard, TryLockError};

/// The id name of the line's messages.
pub(crate) const ID_NAME: &str = "metal2vulkan-submit-failed";

/// The most `VkCommandBuffer`s, and the most resources, a line names.
pub(crate) const NAMED: usize = 32;

/// The most `VkSubmitInfo`s a line names.
pub(crate) const NAMED_INFOS: usize = 8;

/// The most waits, and the most signals, a line names of one `VkSubmitInfo`.
pub(crate) const NAMED_SEMAPHORES: usize = 8;

/// The most Metal command buffers a line names.
pub(crate) const NAMED_BUFFERS: usize = 8;

/// The least `maxTimelineSemaphoreValueDifference` Vulkan requires of a device (its required
/// limits): a value further than this from the timeline's breaks the submission's valid usage on
/// every device.
const LEAST_DIFFERENCE: u64 = (1 << 31) - 1;

/// A failed `vkQueueSubmit`, as its caller made it.
pub(crate) struct Failed<'a> {
    /// What it submitted, in words ("a batch of a command buffer").
    pub(crate) what: &'a str,
    /// Its `VkQueue`'s place among the device's queues for command buffers; `None` for the
    /// staging or the fill queue (`device/staging.rs`), which `what` names.
    pub(crate) queue: Option<usize>,
    pub(crate) result: vk::Result,
    pub(crate) infos: &'a [vk::SubmitInfo<'a>],
    pub(crate) fence: Option<vk::Fence>,
}

/// What the submitter knows of a failed submission's semaphores, command buffers and resources.
#[derive(Default)]
pub(crate) struct Carried {
    /// The binary semaphores among the infos' (the caller's): they hold no value to read.
    pub(crate) binary: Vec<vk::Semaphore>,
    /// The events' semaphores among the infos'.
    pub(crate) events: Vec<vk::Semaphore>,
    /// The Metal command buffers whose work it carried, each as the runtime's notices name it.
    pub(crate) buffers: Vec<String>,
    /// The role of each `VkCommandBuffer` the submitter knows it of.
    pub(crate) roles: Vec<(vk::CommandBuffer, String)>,
    pub(crate) resources: Resources,
    /// The identities of the resources stated so far: each is stated once.
    pub(crate) stated: BTreeSet<ObjectId>,
}

/// The resources a failed submission's command buffers name, as its line states them.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct Resources {
    /// The first [`NAMED`], each stated.
    pub(crate) named: Vec<String>,
    /// Every one stated, those named among them.
    pub(crate) stated: usize,
    /// The keys of the runtime's own resources (scratch, uploads, stand-ins), which no program
    /// resource owns: counted, not stated.
    pub(crate) own: usize,
    /// Those gone: no hold keeps them, though a command buffer names them.
    pub(crate) gone: usize,
    /// Those whose memory a detach let go of, or whose import is released now.
    pub(crate) released: usize,
}

/// Where a resource a failed submission names stands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Standing {
    /// It holds its memory.
    Held,
    /// Its bytes are detached, or its import is released now (`host_residency.rs`).
    Released,
    /// Nothing holds it.
    Gone,
}

impl Resources {
    /// State one more, `text`, standing so.
    pub(crate) fn state(&mut self, text: String, standing: Standing) {
        self.stated += 1;
        match standing {
            Standing::Held => {}
            Standing::Released => self.released += 1,
            Standing::Gone => self.gone += 1,
        }
        if self.named.len() < NAMED {
            self.named.push(text);
        }
    }
}

/// `mutex` locked where no other thread holds it (a poisoned one too): a failed submission's line
/// reads what it states without waiting for any lock but the registries' (`used_resources.rs`),
/// whatever order other threads take them in.
pub(crate) fn try_lock<T>(mutex: &Mutex<T>) -> Option<MutexGuard<'_, T>> {
    match mutex.try_lock() {
        Ok(guard) => Some(guard),
        Err(TryLockError::Poisoned(poisoned)) => Some(poisoned.into_inner()),
        Err(TryLockError::WouldBlock) => None,
    }
}

/// The device's memory types' property flags, as a failed submission's line words them: read
/// once per line, where the device's memory lock is free.
pub(crate) struct TypeFlags(Option<Vec<vk::MemoryPropertyFlags>>);

impl TypeFlags {
    /// The flags of `shared`'s memory types, where its memory lock is free now.
    pub(crate) fn read(shared: &DeviceShared) -> TypeFlags {
        let types = try_lock(&shared.memory).map(|memory| {
            (0..vk::MAX_MEMORY_TYPES as u32)
                .map_while(|index| memory.type_flags(index))
                .collect()
        });
        TypeFlags(types)
    }

    /// Memory type `index` with its flags, where known: "type 2 (DEVICE_LOCAL)".
    pub(crate) fn of(&self, index: u32) -> String {
        match (self.0.as_ref()).and_then(|types| types.get(index as usize)) {
            Some(&flags) => format!("type {index} ({})", flag_names(flags)),
            None => format!("type {index}"),
        }
    }
}

/// `flags` by their names, `|` between them, any bit without one by its value; "none" for none.
pub(crate) fn flag_names(flags: vk::MemoryPropertyFlags) -> String {
    use vk::MemoryPropertyFlags as F;
    let named = [
        (F::DEVICE_LOCAL, "DEVICE_LOCAL"),
        (F::HOST_VISIBLE, "HOST_VISIBLE"),
        (F::HOST_COHERENT, "HOST_COHERENT"),
        (F::HOST_CACHED, "HOST_CACHED"),
        (F::LAZILY_ALLOCATED, "LAZILY_ALLOCATED"),
        (F::PROTECTED, "PROTECTED"),
    ];
    let mut names: Vec<String> = (named.iter())
        .filter(|(flag, _)| flags.contains(*flag))
        .map(|(_, name)| (*name).to_owned())
        .collect();
    let known = named.iter().fold(F::empty(), |all, (flag, _)| all | *flag);
    let other = flags & !known;
    if !other.is_empty() {
        names.push(format!("{:#x}", other.as_raw()));
    }
    match names.is_empty() {
        true => "none".to_owned(),
        false => names.join("|"),
    }
}

/// What a semaphore of a failed submission is to the runtime.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Role {
    /// The timeline of the `VkQueue` at this place.
    Queue(usize),
    /// The timeline the pieces of a batch's work signal on the `VkQueue` at this place.
    Pieces(usize),
    /// The device's own timeline at this place: the internal submissions', the staging queue's,
    /// the fill queue's (`Device::own_completed`).
    Own(usize),
    /// An event's.
    Event,
    /// The caller's binary semaphore.
    Binary,
    /// Another timeline.
    Timeline,
}

/// A wait or a signal of a failed submission.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SemaphoreUse {
    pub(crate) handle: u64,
    pub(crate) role: Role,
    /// The value waited for or signalled; `None` for a binary semaphore.
    pub(crate) value: Option<u64>,
    /// The timeline's value as the line was made; `None` for a binary semaphore, or where the
    /// read failed.
    pub(crate) now: Option<u64>,
}

/// One `VkSubmitInfo` of a failed submission, as its line states it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct InfoReport {
    /// The first [`NAMED_SEMAPHORES`] waits, and how many it has.
    pub(crate) waits: (Vec<SemaphoreUse>, usize),
    /// Its command buffers, each with its role where known, within the line's [`NAMED`]; and
    /// how many it has.
    pub(crate) buffers: (Vec<(u64, Option<String>)>, usize),
    /// The first [`NAMED_SEMAPHORES`] signals, and how many it has.
    pub(crate) signals: (Vec<SemaphoreUse>, usize),
}

/// A failed submission, as its line states it.
pub(crate) struct Report {
    pub(crate) what: String,
    pub(crate) queue: Option<usize>,
    pub(crate) result: vk::Result,
    pub(crate) fence: Option<u64>,
    /// The first [`NAMED_INFOS`] infos, and how many it has.
    pub(crate) infos: (Vec<InfoReport>, usize),
    /// The command buffers its infos hold, in all.
    pub(crate) command_buffers: usize,
    pub(crate) carried: Carried,
}

/// One `VkSubmitInfo`'s lists, read from it ([`lists`]).
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct InfoLists<'a> {
    pub(crate) waits: &'a [vk::Semaphore],
    /// Its `VkTimelineSemaphoreSubmitInfo`'s wait values, where its chain holds one.
    pub(crate) wait_values: Option<&'a [u64]>,
    pub(crate) buffers: &'a [vk::CommandBuffer],
    pub(crate) signals: &'a [vk::Semaphore],
    /// Its `VkTimelineSemaphoreSubmitInfo`'s signal values, where its chain holds one.
    pub(crate) signal_values: Option<&'a [u64]>,
}

/// The `count` elements at `pointer`, none where it is null.
///
/// # Safety
/// A non-null `pointer` points at `count` initialized elements that outlive `'a`.
unsafe fn elements<'a, T>(pointer: *const T, count: u32) -> &'a [T] {
    if pointer.is_null() || count == 0 {
        return &[];
    }
    // SAFETY: the caller's.
    unsafe { std::slice::from_raw_parts(pointer, count as usize) }
}

/// `info`'s lists, its timeline values from the `VkTimelineSemaphoreSubmitInfo` its chain holds.
///
/// # Safety
/// `info` is a `VkSubmitInfo` as a `vkQueueSubmit` was handed it: its arrays hold their counts'
/// elements, and its chain valid structures, alive for `'a`.
pub(crate) unsafe fn lists<'a>(info: &'a vk::SubmitInfo<'a>) -> InfoLists<'a> {
    let mut timeline: Option<&vk::TimelineSemaphoreSubmitInfo<'_>> = None;
    let mut next = info.p_next.cast::<vk::BaseInStructure<'_>>();
    // SAFETY: the chain's structures are valid, each starting with its type and next pointer.
    while let Some(base) = unsafe { next.as_ref() } {
        if base.s_type == vk::StructureType::TIMELINE_SEMAPHORE_SUBMIT_INFO {
            // SAFETY: a structure of that type is one.
            timeline = Some(unsafe { &*next.cast::<vk::TimelineSemaphoreSubmitInfo<'_>>() });
            break;
        }
        next = base.p_next;
    }
    // SAFETY: the caller's.
    unsafe {
        InfoLists {
            waits: elements(info.p_wait_semaphores, info.wait_semaphore_count),
            wait_values: timeline.map(|timeline| {
                elements(
                    timeline.p_wait_semaphore_values,
                    timeline.wait_semaphore_value_count,
                )
            }),
            buffers: elements(info.p_command_buffers, info.command_buffer_count),
            signals: elements(info.p_signal_semaphores, info.signal_semaphore_count),
            signal_values: timeline.map(|timeline| {
                elements(
                    timeline.p_signal_semaphore_values,
                    timeline.signal_semaphore_value_count,
                )
            }),
        }
    }
}

/// Log the line of `failed`, a `vkQueueSubmit` of `device`'s that failed, with what `carry` adds
/// of its semaphores, command buffers and resources (`types` the device's memory types' flags):
/// read now, on the failure's path alone.
pub(crate) fn submit_failed(
    device: &Device,
    failed: &Failed<'_>,
    carry: impl FnOnce(&mut Carried, &TypeFlags),
) {
    let mut carried = Carried::default();
    carry(&mut carried, &TypeFlags::read(device.shared()));
    let report = Report::read(device, failed, carried);
    warn(device, ID_NAME, report.text());
}

/// Hand `text`, a warning of id name `id_name`, to `device`'s sink, else write it on stderr.
pub(crate) fn warn(device: &Device, id_name: &str, text: String) {
    let message = DebugMessage {
        severity: MessageSeverity::Warning,
        kind: MessageKind {
            general: true,
            ..MessageKind::default()
        },
        id_name: id_name.to_owned(),
        id: 0,
        // The text ends in its id name, as a degrade notice's does, so a consumer that writes the
        // text alone still lets a reader find the line by id.
        text: format!("{text} [{id_name}]"),
        degradation: None,
    };
    match device.shared().instance.sink() {
        Some(sink) => sink.deliver(&message),
        None => eprintln!("{}", message.text),
    }
}

impl Report {
    /// The report of `failed`, a submission of `device`'s, its semaphores' values read now.
    fn read(device: &Device, failed: &Failed<'_>, carried: Carried) -> Report {
        let shared = device.shared();
        let own = device.own_semaphores();
        let role = |semaphore: vk::Semaphore| {
            if carried.binary.contains(&semaphore) {
                return Role::Binary;
            }
            if let Some(at) = (shared.queues.iter()).position(|slot| slot.timeline == semaphore) {
                return Role::Queue(at);
            }
            if let Some(at) = (shared.queues.iter()).position(|slot| slot.pieces == semaphore) {
                return Role::Pieces(at);
            }
            if let Some(at) = own.iter().position(|own| *own == Some(semaphore)) {
                return Role::Own(at);
            }
            match carried.events.contains(&semaphore) {
                true => Role::Event,
                false => Role::Timeline,
            }
        };
        let raw = shared.vk();
        let read = |semaphore: vk::Semaphore, value: Option<u64>| {
            let role = role(semaphore);
            let binary = role == Role::Binary;
            let now = (!binary).then(|| {
                // SAFETY: a timeline semaphore of the device's, live while a submission names it.
                let now = unsafe { raw.get_semaphore_counter_value(semaphore) };
                exit_guard::called(Call::GetSemaphoreCounterValue);
                now.ok()
            });
            SemaphoreUse {
                handle: semaphore.as_raw(),
                role,
                value: value.filter(|_| !binary),
                now: now.flatten(),
            }
        };
        let uses = |semaphores: &[vk::Semaphore], values: Option<&[u64]>| {
            let named = (semaphores.iter().enumerate().take(NAMED_SEMAPHORES))
                .map(|(at, &semaphore)| read(semaphore, values.and_then(|v| v.get(at).copied())))
                .collect();
            (named, semaphores.len())
        };
        let mut command_buffers = 0;
        let infos = (failed.infos.iter().take(NAMED_INFOS))
            .map(|info| {
                // SAFETY: the infos are the failed call's, as it was handed them.
                let lists = unsafe { lists(info) };
                let room = NAMED.saturating_sub(command_buffers);
                command_buffers += lists.buffers.len().min(room);
                let buffers = (lists.buffers.iter().take(room))
                    .map(|&buffer| {
                        let role = (carried.roles.iter())
                            .find(|(raw, _)| *raw == buffer)
                            .map(|(_, role)| role.clone());
                        (buffer.as_raw(), role)
                    })
                    .collect();
                InfoReport {
                    waits: uses(lists.waits, lists.wait_values),
                    buffers: (buffers, lists.buffers.len()),
                    signals: uses(lists.signals, lists.signal_values),
                }
            })
            .collect();
        let command_buffers = (failed.infos.iter())
            .map(|info| info.command_buffer_count as usize)
            .sum();
        Report {
            what: failed.what.to_owned(),
            queue: failed.queue,
            result: failed.result,
            fence: failed.fence.map(|fence| fence.as_raw()),
            infos: (infos, failed.infos.len()),
            command_buffers,
            carried,
        }
    }

    /// The line.
    pub(crate) fn text(&self) -> String {
        let (infos, info_count) = &self.infos;
        let queue = match self.queue {
            Some(at) => format!("VkQueue {at}"),
            None => "its own VkQueue".to_owned(),
        };
        let mut line = format!(
            "metal2vulkan: vkQueueSubmit failed with VkResult {} ({:?}) on {queue}, submitting \
             {}: {info_count} VkSubmitInfo{}, {} VkCommandBuffer{}",
            self.result.as_raw(),
            self.result,
            self.what,
            plural(*info_count),
            self.command_buffers,
            plural(self.command_buffers),
        );
        match self.fence {
            Some(fence) => {
                let _ = write!(line, ", fence {fence:#x}");
            }
            None => line.push_str(", no fence"),
        }
        let mut buffers_named = 0;
        for (at, info) in infos.iter().enumerate() {
            let _ = write!(line, "; info {at}: waits ");
            semaphores(&mut line, &info.waits, false);
            line.push_str(", runs ");
            let (named, count) = &info.buffers;
            buffers_named += named.len();
            match count {
                0 => line.push_str("nothing"),
                _ => {
                    line.push('[');
                    for (place, (handle, role)) in named.iter().enumerate() {
                        let comma = if place == 0 { "" } else { ", " };
                        let _ = write!(line, "{comma}{handle:#x}");
                        if let Some(role) = role {
                            let _ = write!(line, " {role}");
                        }
                    }
                    rest(&mut line, named.len(), *count);
                    line.push(']');
                }
            }
            line.push_str(", signals ");
            semaphores(&mut line, &info.signals, true);
        }
        if *info_count > infos.len() {
            let _ = write!(line, "; {} more VkSubmitInfos", info_count - infos.len());
        }
        if buffers_named < self.command_buffers {
            let _ = write!(
                line,
                "; {} VkCommandBuffers not named",
                self.command_buffers - buffers_named
            );
        }
        let buffers = &self.carried.buffers;
        if !buffers.is_empty() {
            let _ = write!(
                line,
                "; of {} command buffer{}: ",
                buffers.len(),
                plural(buffers.len())
            );
            for (place, named) in buffers.iter().take(NAMED_BUFFERS).enumerate() {
                let comma = if place == 0 { "" } else { "; " };
                let _ = write!(line, "{comma}{named}");
            }
            if buffers.len() > NAMED_BUFFERS {
                let _ = write!(line, "; and {} more", buffers.len() - NAMED_BUFFERS);
            }
        }
        let resources = &self.carried.resources;
        let _ = write!(
            line,
            "; resources: {} stated, {} gone, {} detached or released, and {} key{} of the \
             runtime's own not stated",
            resources.stated,
            resources.gone,
            resources.released,
            resources.own,
            plural(resources.own),
        );
        if !resources.named.is_empty() {
            line.push_str("; stated: ");
            line.push_str(&resources.named.join("; "));
            if resources.stated > resources.named.len() {
                let _ = write!(
                    line,
                    "; and {} more",
                    resources.stated - resources.named.len()
                );
            }
        }
        line
    }
}

/// "s" for a count other than one.
fn plural(count: usize) -> &'static str {
    if count == 1 {
        ""
    } else {
        "s"
    }
}

/// Note the `count - named` elements of a list not named.
fn rest(line: &mut String, named: usize, count: usize) {
    if count > named {
        let _ = write!(line, ", and {} more", count - named);
    }
}

/// Write the waits or signals (`signals`) `uses` names, with the count of all.
fn semaphores(line: &mut String, (named, count): &(Vec<SemaphoreUse>, usize), signals: bool) {
    if *count == 0 {
        line.push_str("none");
        return;
    }
    line.push('[');
    for (place, used) in named.iter().enumerate() {
        let comma = if place == 0 { "" } else { ", " };
        let _ = write!(line, "{comma}{:#x} ", used.handle);
        match used.role {
            Role::Queue(at) => {
                let _ = write!(line, "VkQueue {at}'s timeline");
            }
            Role::Pieces(at) => {
                let _ = write!(line, "VkQueue {at}'s pieces");
            }
            Role::Own(at) => {
                let own = ["internal", "staging", "fill"];
                let _ = write!(
                    line,
                    "the device's {} timeline",
                    own.get(at).unwrap_or(&"own")
                );
            }
            Role::Event => line.push_str("an event's timeline"),
            Role::Binary => line.push_str("binary, the caller's"),
            Role::Timeline => line.push_str("timeline"),
        }
        if let Some(value) = used.value {
            let _ = write!(line, " to {value}");
        }
        match (used.value, used.now) {
            (Some(value), Some(now)) => {
                let _ = write!(line, " (at {now})");
                if signals && value <= now {
                    line.push_str(
                        " NOT ABOVE ITS VALUE (VUID-VkSubmitInfo-pSignalSemaphores-03242)",
                    );
                }
                if value.abs_diff(now) > LEAST_DIFFERENCE {
                    let vuid = match signals {
                        true => "VUID-VkSubmitInfo-pSignalSemaphores-03244",
                        false => "VUID-VkSubmitInfo-pWaitSemaphores-03243",
                    };
                    let _ = write!(line, " PAST 2^31 - 1 FROM ITS VALUE ({vuid})");
                }
            }
            (Some(_), None) => line.push_str(" (its value unread)"),
            (None, _) => {}
        }
    }
    rest(line, named.len(), *count);
    line.push(']');
}
