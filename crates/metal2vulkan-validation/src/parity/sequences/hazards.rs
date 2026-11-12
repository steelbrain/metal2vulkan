//! The synchronization validation's hazards on a sequence's run, read against the sequence:
//! which of them no order can matter for, and which may be a barrier the runtime left out.
//!
//! With `METAL2VULKAN_LANE_VALIDATION=sync` the layer reports a hazard between two commands when
//! no barrier, wait or submission it sees orders them and their ranges of one memory meet, one of
//! them writing. Its ranges are wider than a sequence's accesses: a shader's buffer counts as
//! accessed over its whole descriptor, which the runtime binds over the bytes the translation
//! proves the shader reaches, from the device's offset alignment down and in 16-byte granules
//! (`seq_copy`'s and `seq_mix`'s one word: 16 bytes or more), and from the bound offset to the
//! buffer's end where the shader indexes it (a draw's rectangles and colours, a texel kernel's
//! rows); an image a shader uses counts whole. Neither Metal nor the runtime orders by those
//! ranges:
//!
//! - Metal leaves untracked memory between encoders, and a concurrent compute encoder's
//!   dispatches, to the program (its fences, events and memory barriers), and the runtime records
//!   no barrier there either.
//! - Where Metal orders (a tracked resource between encoders of a queue, a serial compute
//!   encoder's dispatches), the runtime orders the bytes a command reaches, and records no barrier
//!   between commands whose bytes are apart: no order of them can change a byte.
//!
//! The generator orders every two accesses of a sequence that share a byte (`order`), so a
//! hazard between commands that share none changes no answer, whatever Metal orders between
//! them. A hazard is **explained** when no two commands it can name share a byte of its memory:
//! an earlier command accessing it as the hazard's earlier access does (writing or reading), and a
//! later one accessing it as its later access does, after the earlier (the runtime submits the
//! batches of every queue in commit order, as a sequence's events allow, and records a command
//! buffer's commands in encoded order); each recorded with the Vulkan command the hazard names
//! ([`vulkan_commands`]); the later one in a command buffer whose batches the layer may have
//! checked since the lane last took its messages (the lane takes them after each commit, once the
//! command buffer is scheduled or a bound passed, and a command buffer seen scheduled at a take
//! sent no batch after it); and both in one command buffer when the layer found no barrier at all
//! after an earlier write, or reported the hazard as a command was recorded ([`Hazard`]). Its
//! memory is the buffer's or the heap's whose `VkBuffer` the hazard names,
//! and any texture for a `VkImage` (the lane does not ask the runtime for its images: exporting
//! one changes what the runtime may do with it). It is logged with what Metal orders between the
//! commands ([`MetalOrder`]) and fails nothing. Every other message fails the run as before: a
//! hazard between two commands that share a byte (the generator ordered them, so the runtime
//! should have), a hazard on a `VkBuffer` no command of the sequence bound (a staging buffer, the
//! runtime's own), a hazard no two commands of the sequence can make (an access of the
//! runtime's own), and any message of another shape.

use super::order::{Access, Order, Position};
use super::program::{At, Command, Kind, Program, Resource, Step};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
#[cfg(any(m2v_gpu, m2v_cpu_icd))]
use {super::order::accesses, metal2vulkan_runtime::DebugMessage};

/// A call of the runner that binds buffers, as the runtime's side of the lane notes it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Call {
    Fill,
    Copy,
    ToTexture,
    FromTexture,
    Dispatch,
    TexelDispatch,
    Draw,
}

/// One encoded call of a run: the call, the raw `VkBuffer` of each buffer it passed, in the
/// call's order, and how many command buffers were committed before it (the runner's setup is
/// the first, so command buffer `k` of the sequence is encoded after `k + 1`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Encoded {
    pub(crate) committed: usize,
    pub(crate) call: Call,
    pub(crate) buffers: Vec<u64>,
}

/// The call the runner makes for `command`, with the buffers it passes, in order; `None` for a
/// command that passes none (a memory barrier).
fn call_of(command: &Command) -> Option<(Call, Vec<At>)> {
    let with = |mut ats: Vec<At>, indirect: Option<At>| {
        ats.extend(indirect);
        ats
    };
    Some(match *command {
        Command::Fill { buffer, start, .. } => (
            Call::Fill,
            vec![At {
                buffer,
                offset: start,
            }],
        ),
        Command::Copy { from, to, .. } => (Call::Copy, vec![from, to]),
        Command::ToTexture { from, .. } => (Call::ToTexture, vec![from]),
        Command::FromTexture { to, .. } => (Call::FromTexture, vec![to]),
        Command::Dispatch { a, b, indirect, .. } => (Call::Dispatch, with(vec![a, b], indirect)),
        Command::TexelRead { to, .. } => (Call::TexelDispatch, vec![to]),
        Command::TexelWrite { from, .. } => (Call::TexelDispatch, vec![from]),
        Command::Barrier { .. } => return None,
        Command::Draw {
            rects,
            colors,
            indirect,
            ..
        } => (Call::Draw, with(vec![rects, colors], indirect)),
    })
}

/// The memory each `VkBuffer` the sequence's commands bound is: a buffer's own, or a heap's (every
/// buffer of a heap is in its `VkBuffer`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Handles(BTreeMap<u64, BTreeSet<Resource>>);

impl Handles {
    /// The handles of a run of `program` whose calls were `encoded`: its sequence's calls in
    /// encoded order, matched with the program's commands. A run a refusal ended has a prefix of
    /// them. Calls that do not match the program's are an error, which leaves every hazard
    /// unexplained.
    pub(crate) fn of(program: &Program, encoded: &[Encoded]) -> Result<Handles, String> {
        let mut handles = Handles::default();
        for (index, spec) in program.command_buffers.iter().enumerate() {
            let mut calls = encoded
                .iter()
                .filter(|encoded| encoded.committed == index + 1);
            let commands = spec.steps.iter().filter_map(|step| match step {
                Step::Encoder(encoder) => Some(&encoder.commands),
                _ => None,
            });
            for command in commands.flatten() {
                let Some((call, ats)) = call_of(command) else {
                    continue;
                };
                let Some(encoded) = calls.next() else {
                    break;
                };
                if encoded.call != call || encoded.buffers.len() != ats.len() {
                    return Err(format!(
                        "cb{index}: the run encoded {:?} of {} buffer(s) where the sequence has \
                         {command}",
                        encoded.call,
                        encoded.buffers.len()
                    ));
                }
                for (at, &raw) in ats.iter().zip(&encoded.buffers) {
                    let (resource, _, _) = program.memory(*at);
                    handles.0.entry(raw).or_default().insert(resource);
                }
            }
            if let Some(extra) = calls.next() {
                return Err(format!(
                    "cb{index}: the run encoded {:?} past the sequence's commands",
                    extra.call
                ));
            }
        }
        Ok(handles)
    }
}

/// The memory a hazard names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Named {
    Buffer(u64),
    Image(u64),
}

/// A hazard as the layer words it: whether its later and earlier accesses write (else read), the
/// commands it names, the memory, and whether its two commands are in one command buffer.
///
/// They are when the layer reported it as a command was recorded (two commands of one
/// `VkCommandBuffer`), and when it found no barrier at all after an earlier write ("No
/// sufficient synchronization is present"): every batch of the runtime's ends with a barrier
/// from every write to the host (`Barrier::to_host`), so a write of an earlier batch always has
/// one, and the layer words its hazards "The current synchronization allows ..."; and a batch is
/// one command buffer's.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Hazard {
    pub(crate) later_writes: bool,
    pub(crate) earlier_writes: bool,
    pub(crate) later: String,
    pub(crate) earlier: String,
    pub(crate) memory: Named,
    pub(crate) one_cb: bool,
}

/// The accesses of a hazard of id `id_name`: whether the later one writes, and the earlier one.
fn kinds(id_name: &str) -> Option<(bool, bool)> {
    Some(match id_name {
        "SYNC-HAZARD-READ-AFTER-WRITE" | "SYNC-HAZARD-READ-RACING-WRITE" => (false, true),
        "SYNC-HAZARD-WRITE-AFTER-READ" | "SYNC-HAZARD-WRITE-RACING-READ" => (true, false),
        "SYNC-HAZARD-WRITE-AFTER-WRITE" | "SYNC-HAZARD-WRITE-RACING-WRITE" => (true, true),
        _ => return None,
    })
}

/// The command name at the start of `text`: up to a space, a parenthesis or a comma.
fn command(text: &str) -> &str {
    let end = text.find([' ', '(', ',', '.']).unwrap_or(text.len());
    &text[..end]
}

/// The hexadecimal handle at the start of `text` (`0x...`).
fn handle(text: &str) -> Option<u64> {
    let hex = text.strip_prefix("0x")?;
    let end = hex
        .find(|c: char| !c.is_ascii_hexdigit())
        .unwrap_or(hex.len());
    u64::from_str_radix(&hex[..end], 16).ok()
}

/// The hazard a message of id name `id_name` and text `text` reports, when it is one the lane
/// reads: "`<call>`(): `<KIND>` hazard detected. `<initiator>` [...] reads `<VkBuffer|VkImage>
/// 0x...`[, ...] which was previously written by [another ]`<command>`", as the layer words a
/// hazard at recording and at a submission (`sync_reporting.cpp`, `sync_error_messages.cpp`).
/// The initiator is the later command, or from 1.4.363 for a shader's access through a
/// descriptor "Shader stage `<stage>` in `<command>`" at a submission and "Shader stage
/// `<stage>`" at recording, where the later command is the call.
pub(crate) fn parse(id_name: &str, text: &str) -> Option<Hazard> {
    let (later_writes, earlier_writes) = kinds(id_name)?;
    let (_, rest) = text.split_once(" hazard detected. ")?;
    let (later, rest) = match rest.strip_prefix("Shader stage ") {
        Some(staged) => {
            let after = &staged[staged.find(' ')?..];
            match after.strip_prefix(" in ") {
                Some(initiated) => (command(initiated), initiated),
                None => (command(text), after),
            }
        }
        None => (command(rest), rest),
    };
    let later = later.to_owned();
    let verb = if later_writes {
        " writes to "
    } else {
        " reads "
    };
    let (_, rest) = rest.split_once(verb)?;
    let memory = if let Some(raw) = rest.strip_prefix("VkBuffer ") {
        Named::Buffer(handle(raw)?)
    } else {
        Named::Image(handle(rest.strip_prefix("VkImage ")?)?)
    };
    let previously = if earlier_writes {
        "which was previously written by "
    } else {
        "which was previously read by "
    };
    let (_, rest) = rest.split_once(previously)?;
    let rest = rest.strip_prefix("another ").unwrap_or(rest);
    let recorded = !text.starts_with("vkQueueSubmit");
    let unbarriered = earlier_writes && text.contains("No sufficient synchronization is present");
    Some(Hazard {
        later_writes,
        earlier_writes,
        later,
        earlier: command(rest).to_owned(),
        memory,
        one_cb: recorded || unbarriered,
    })
}

/// What Metal orders between two commands of a hazard's that share no byte, as far as the
/// generator's order (`order`) states it or Metal's hazard tracking adds to it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum MetalOrder {
    /// The generator's order holds between them (a fence, an event, a serial encoder's tracked
    /// object, a memory barrier): Metal orders them, and the runtime needs no barrier, since they
    /// share no byte.
    Ordered,
    /// One tracked object of one queue: Metal's tracking orders whole resources, the runtime the
    /// bytes, which they do not share.
    Tracked,
    /// Untracked memory between encoders: its order is the program's.
    Untracked,
    /// Two dispatches of a concurrent compute encoder with no memory barrier between.
    Concurrent,
    /// Two commands of one blit encoder, or two of one render pass, whose order the generator
    /// does not rely on.
    SameEncoder,
    /// A texture and a view of it, or two views: objects of their own to Metal's tracking.
    OtherObjects,
    /// Two queues, with no event between them.
    OtherQueues,
}

impl MetalOrder {
    fn of(
        program: &Program,
        order: &Order,
        earlier: (Position, &Access),
        later: (Position, &Access),
    ) -> MetalOrder {
        let ((p, a), (q, b)) = (earlier, later);
        if order.ordered((p, a), (q, b)) {
            return MetalOrder::Ordered;
        }
        let queue = |cb: usize| program.command_buffers[cb].queue;
        let tracked = |access: &Access| program.tracked(access.object);
        if (p.cb, p.step) == (q.cb, q.step) {
            let Step::Encoder(encoder) = &program.command_buffers[p.cb].steps[p.step] else {
                return MetalOrder::SameEncoder;
            };
            return match encoder.kind {
                Kind::Compute { concurrent: true } => MetalOrder::Concurrent,
                _ => MetalOrder::SameEncoder,
            };
        }
        if !tracked(a) || !tracked(b) {
            MetalOrder::Untracked
        } else if a.object != b.object {
            MetalOrder::OtherObjects
        } else if queue(p.cb) != queue(q.cb) {
            MetalOrder::OtherQueues
        } else {
            MetalOrder::Tracked
        }
    }
}

/// How the lane reads one hazard of a run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Verdict {
    /// No two commands the hazard can name share a byte: the pairs it can name, and what Metal
    /// orders between them.
    Explained {
        pairs: usize,
        metal: BTreeSet<MetalOrder>,
    },
    /// Two commands it can name share a byte, which the sequence orders: the earlier and the
    /// later access.
    Shared {
        earlier: (Position, Access),
        later: (Position, Access),
    },
    /// The lane cannot name the memory, or no two commands of the sequence access it as the
    /// hazard says: why.
    Unnamed(String),
}

/// The Vulkan commands the runtime records a Metal dispatch with, as the layer names them: a
/// dispatch, its base or an indirect one (an indirect dispatch's flat path dispatches the kernel
/// indirectly from arguments a dispatch of its own wrote).
pub(crate) const DISPATCHES: &[&str] = &[
    "vkCmdDispatch",
    "vkCmdDispatchBase",
    "vkCmdDispatchIndirect",
];

/// The Vulkan commands the runtime records `command` with, as the layer names them. A render
/// pass's own accesses (`command` `None`: its target, loaded, cleared, drawn and stored) are its
/// beginning's and end's, a clear's, a layout transition's and its draws'. A fill is a fill, an
/// update or a copy of a pattern (`word_fill.rs`); a dispatch of either kind and a texel kernel's,
/// [`DISPATCHES`]; a draw, a direct or an indirect draw as Metal's is.
pub(crate) fn vulkan_commands(command: Option<&Command>) -> &'static [&'static str] {
    const PASS: &[&str] = &[
        "vkCmdBeginRenderPass",
        "vkCmdBeginRenderPass2",
        "vkCmdBeginRenderPass2KHR",
        "vkCmdNextSubpass",
        "vkCmdNextSubpass2",
        "vkCmdNextSubpass2KHR",
        "vkCmdEndRenderPass",
        "vkCmdEndRenderPass2",
        "vkCmdEndRenderPass2KHR",
        "vkCmdBeginRendering",
        "vkCmdBeginRenderingKHR",
        "vkCmdEndRendering",
        "vkCmdEndRenderingKHR",
        "vkCmdClearAttachments",
        "vkCmdClearColorImage",
        "vkCmdPipelineBarrier",
        "vkCmdPipelineBarrier2",
        "vkCmdPipelineBarrier2KHR",
        "vkCmdDraw",
        "vkCmdDrawIndirect",
    ];
    let Some(command) = command else {
        return PASS;
    };
    match command {
        Command::Fill { .. } => &["vkCmdFillBuffer", "vkCmdUpdateBuffer", "vkCmdCopyBuffer"],
        Command::Copy { .. } => &["vkCmdCopyBuffer"],
        Command::ToTexture { .. } => &["vkCmdCopyBufferToImage"],
        Command::FromTexture { .. } => &["vkCmdCopyImageToBuffer"],
        Command::Dispatch { .. } | Command::TexelRead { .. } | Command::TexelWrite { .. } => {
            DISPATCHES
        }
        Command::Barrier { .. } => &[],
        Command::Draw { indirect: None, .. } => &["vkCmdDraw"],
        Command::Draw {
            indirect: Some(_), ..
        } => &["vkCmdDrawIndirect"],
    }
}

/// Whether the access at `q` may come after the one at `p` as the layer checks them: a later
/// command, or a render pass's own access (its load, its store) and a command of it.
fn after(p: Position, q: Position) -> bool {
    p < q
        || p != q
            && (p.cb, p.step) == (q.cb, q.step)
            && (p.command.is_none() || q.command.is_none())
}

/// Whether the command at `at` of `program` may be recorded with the Vulkan command `named`.
fn may_be(program: &Program, at: Position, named: &str) -> bool {
    let command = at.command.map(
        |index| match &program.command_buffers[at.cb].steps[at.step] {
            Step::Encoder(encoder) => &encoder.commands[index],
            other => panic!("cb{} s{} is no encoder: {other:?}", at.cb, at.step),
        },
    );
    vulkan_commands(command).contains(&named)
}

/// The verdict on `hazard` of a run of `program` whose `VkBuffer`s are `handles`, its later
/// command in one of the command buffers `later_cbs`.
pub(crate) fn verdict(
    program: &Program,
    order: &Order,
    all: &[(Position, Access)],
    handles: &Result<Handles, String>,
    hazard: &Hazard,
    later_cbs: &BTreeSet<usize>,
) -> Verdict {
    let memories: BTreeSet<Resource> = match hazard.memory {
        Named::Image(_) => (0..program.textures.len()).map(Resource::Texture).collect(),
        Named::Buffer(raw) => match handles {
            Err(why) => return Verdict::Unnamed(why.clone()),
            Ok(handles) => match handles.0.get(&raw) {
                Some(memories) => memories.clone(),
                None => {
                    return Verdict::Unnamed(format!(
                        "VkBuffer {raw:#x} is no buffer a command of the sequence bound"
                    ))
                }
            },
        },
    };
    let memories = &memories;
    let of = |writes: bool| {
        all.iter()
            .filter(move |(_, access)| {
                memories.contains(&access.resource) && access.write == writes
            })
            .copied()
    };
    let earlier = of(hazard.earlier_writes).filter(|(p, _)| may_be(program, *p, &hazard.earlier));
    let later: Vec<(Position, Access)> = of(hazard.later_writes)
        .filter(|(q, _)| later_cbs.contains(&q.cb) && may_be(program, *q, &hazard.later))
        .collect();
    let mut pairs = 0;
    let mut metal = BTreeSet::new();
    for (p, a) in earlier {
        for &(q, b) in &later {
            if a.resource != b.resource || !after(p, q) || (hazard.one_cb && p.cb != q.cb) {
                continue;
            }
            if a.start < b.end && b.start < a.end {
                return Verdict::Shared {
                    earlier: (p, a),
                    later: (q, b),
                };
            }
            pairs += 1;
            metal.insert(MetalOrder::of(program, order, (p, &a), (q, &b)));
        }
    }
    if pairs == 0 {
        let cbs: Vec<String> = later_cbs.iter().map(|cb| format!("cb{cb}")).collect();
        return Verdict::Unnamed(format!(
            "no two commands of the sequence access its memory as it says, the later in {}",
            if cbs.is_empty() {
                "none of its command buffers".to_owned()
            } else {
                cbs.join(" or ")
            }
        ));
    }
    Verdict::Explained { pairs, metal }
}

/// A run's messages, sorted: the hazards explained, each with its hazard and verdict, and the
/// messages that fail the run, a hazard among them carrying the lane's verdict in its text.
#[cfg(any(m2v_gpu, m2v_cpu_icd))]
pub(crate) struct Sorted {
    pub(crate) explained: Vec<(Hazard, Verdict)>,
    pub(crate) failing: Vec<DebugMessage>,
}

/// Sort the messages of a run of `program` whose calls were `encoded`, each with the command
/// buffers of the sequence whose batches the layer may have checked as it reported it.
#[cfg(any(m2v_gpu, m2v_cpu_icd))]
pub(crate) fn sort(
    program: &Program,
    encoded: &[Encoded],
    messages: Vec<(BTreeSet<usize>, DebugMessage)>,
) -> Sorted {
    let handles = Handles::of(program, encoded);
    let order = Order::new(program);
    let all = accesses(program);
    let mut sorted = Sorted {
        explained: Vec::new(),
        failing: Vec::new(),
    };
    for (later_cbs, mut message) in messages {
        let Some(hazard) = parse(&message.id_name, &message.text) else {
            sorted.failing.push(message);
            continue;
        };
        let verdict = verdict(program, &order, &all, &handles, &hazard, &later_cbs);
        match verdict {
            Verdict::Explained { .. } => sorted.explained.push((hazard, verdict)),
            _ => {
                message.text = format!("{}\nthe lane: {verdict}", message.text);
                sorted.failing.push(message);
            }
        }
    }
    sorted
}

impl fmt::Display for MetalOrder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            MetalOrder::Ordered => "Metal orders them (the sequence's order)",
            MetalOrder::Tracked => "Metal's tracking orders them (one tracked object of one queue)",
            MetalOrder::Untracked => "untracked memory: Metal leaves their order to the program",
            MetalOrder::Concurrent => {
                "a concurrent encoder's dispatches with no memory barrier between: Metal leaves \
                 their order to the program"
            }
            MetalOrder::SameEncoder => "one blit encoder or render pass: not relied on",
            MetalOrder::OtherObjects => {
                "a texture and a view of it, or two views: not tracked as one"
            }
            MetalOrder::OtherQueues => {
                "two queues with no event between: Metal leaves their order to the program"
            }
        })
    }
}

/// An access as a verdict names it: its command, what it does, and the bytes.
struct Placed<'a>(&'a (Position, Access));

impl fmt::Display for Placed<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (position, access) = self.0;
        let at = match position.command {
            Some(command) => format!("cb{} s{} c{command}", position.cb, position.step),
            None => format!("cb{} s{} (the pass)", position.cb, position.step),
        };
        let memory = match access.resource {
            Resource::Buffer(index) => format!("buffer{index}"),
            Resource::Heap(index) => format!("heap{index}"),
            Resource::Texture(index) => format!("texture{index}"),
        };
        let does = if access.write { "writes" } else { "reads" };
        write!(
            f,
            "{at} {does} {memory} bytes {}..{}",
            access.start, access.end
        )
    }
}

impl fmt::Display for Verdict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Verdict::Explained { pairs, metal } => {
                let metal: Vec<String> = metal.iter().map(ToString::to_string).collect();
                write!(
                    f,
                    "explained: no two of the {pairs} pair(s) of commands it can name share a \
                     byte; {}",
                    metal.join("; ")
                )
            }
            Verdict::Shared { earlier, later } => write!(
                f,
                "{} and {} share bytes, which the sequence orders: a barrier the runtime left out?",
                Placed(earlier),
                Placed(later)
            ),
            Verdict::Unnamed(why) => write!(f, "unexplained: {why}"),
        }
    }
}

impl fmt::Display for Hazard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let does = |writes| if writes { "writes" } else { "reads" };
        let memory = match self.memory {
            Named::Buffer(raw) => format!("VkBuffer {raw:#x}"),
            Named::Image(raw) => format!("VkImage {raw:#x}"),
        };
        write!(
            f,
            "{} {} {memory} after {} {} it",
            self.later,
            does(self.later_writes),
            self.earlier,
            does(self.earlier_writes)
        )
    }
}
