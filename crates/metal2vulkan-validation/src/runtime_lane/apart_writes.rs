//! The validation layer's write-after-write hazards that leave no two writes of one byte: between
//! two of a test's own writes whose bytes the test placed apart, between the two draws of one
//! index bounds pass, at most one of which draws, and between the chunk draws of an emulated patch
//! draw, at most one of which draws. A lane claim that lets those pass, and nothing wider.
//!
//! **Apart writes.** The layer's synchronization validation takes a shader's storage buffer as
//! written over the whole range its descriptor binds. The runtime binds a storage slot of a buffer
//! within the view's maximum range from the buffer's start to its padded end, the function reading
//! at the slot's residual (`bind::planned_range`), and its hazard tracking orders a serial
//! encoder's dispatches by the bytes the translation proves the function reaches alone (the slot's
//! window, `bind::plan`'s `buffer_use`). So two dispatches that write words 4 bytes apart in one 16-byte
//! block get no barrier between them, and the layer reports a `WRITE_AFTER_WRITE`; so do a fill
//! and a dispatch beside it in one block, and two draws of one render encoder, which Metal leaves
//! unordered (docs/API.md's accepted differences), writing their own bytes of one buffer bound
//! wider. No order of two such writes can change a byte: Vulkan's memory model has a data race
//! only between "operations that access overlapping sets of memory locations", and "a memory
//! operation must not affect memory at a memory location not within its set of memory locations"
//! (the Memory Model appendix). The lane's positive control
//! `sync_coverage::disjoint_words_of_one_block_are_a_hazard_to_the_layer_alone` shows the report
//! beside the bytes, word by word.
//!
//! **The index bounds pass's two draws.** Where Metal has one indexed indirect draw, the runtime's
//! index bounds pass (the runtime's `index_bounds.rs`, announced as `indirect-index-bounds-pass`)
//! records two `vkCmdDrawIndexedIndirect` in the pass, exactly one of which draws the primitives
//! (neither, where the index pool cannot hold them). The layer reads no indirect draw's counts,
//! so it takes both as writing their storage buffers' bound ranges, and reports a
//! `WRITE_AFTER_WRITE` between them as the second is recorded. A draw of no index or no instance
//! makes no write: "primitives are assembled using ... indexCount vertices", "drawn instanceCount
//! times" (`vkCmdDrawIndexed`), so no shader runs. Of the pair at most one accesses memory, and a
//! data race takes two.
//!
//! **An emulated patch draw's chunk draws.** On a device without `tessellationShader` the runtime
//! draws a Metal patch draw as `K` chunk draws (the runtime's `render/tessellation_scan.rs`,
//! announced as `emulated-tessellation-draws`), each a `vkCmdDrawIndirect` of `2^30` of the draw's
//! triangles, whose command the draw's scan writes; `K` covers every triangle the draw's records
//! can hold, every record of the pool for an indirect draw. A chunk past the draw's triangles
//! draws no vertex: the scan writes its `vertexCount` 0, and "primitives are assembled using ...
//! vertexCount consecutive vertex indices" (`vkCmdDraw`, whose parameters a `vkCmdDrawIndirect`
//! command holds). A test keeps each patch draw's triangles below `2^30`, so only chunk 0
//! draws: of the chunks at most one accesses memory. The layer reads no indirect draw's counts and
//! reports a `WRITE_AFTER_WRITE` between the chunks as each is recorded.
//!
//! A test notes each write of a buffer it notes as it places it ([`NotedWrites`]: every write of
//! such a buffer, by its kind, [`Writer`]), and states its claim with
//! [`layer_reported_nothing_but_apart_writes`] once the work ran. A message passes only when it is
//! a `WRITE_AFTER_WRITE` (not the racing form) on a `VkBuffer` the test noted, and either:
//!
//! - **apart**: the test noted two writes or more of the buffer, each of a kind one Vulkan command
//!   records ([`Writer::split`]), none empty and no two sharing a byte, and each of the two
//!   commands is one a kind the test noted on the buffer records ([`Writer::commands`]). It passes
//!   with a `LAYER-APART-WRITES` line.
//! - **idle**: both commands are `vkCmdDrawIndexedIndirect`, reported as the later one was
//!   recorded (two commands of one `VkCommandBuffer`, which holds one command buffer's work: the
//!   command buffer records into pools of its own), every write the test noted of the buffer is a
//!   bounded draw's ([`Writer::BoundedDraw`]), no command buffer it noted holds two, and a device
//!   reporting to the list announced the pass. It passes with a `LAYER-IDLE-DRAW` line.
//! - **idle chunk**: both commands are `vkCmdDrawIndirect`, reported as the later one was
//!   recorded, every write the test noted of the buffer is a patch draw's ([`Writer::PatchDraw`]),
//!   those of each command buffer none empty and no two sharing a byte (one each, or each into a
//!   buffer of its own placed in one `VkBuffer`: two chunks of one draw then, at most one of which
//!   draws, or chunks of two draws whose bytes share nothing), and a device reporting to the list
//!   announced the emulation. It passes with a `LAYER-IDLE-CHUNK` line.
//!
//! Everything else fails as [`layer_reported_nothing`] fails it: a read after a write, a write
//! after a read, a command of a kind the test did not note on the buffer (a copy's among them), an
//! image, a buffer the test did not note or noted one write of, two noted writes that share a byte
//! (the runtime orders those, so the layer's report of them is a missing barrier), bounded draws
//! reported at a submission, between command buffers, or of a command buffer that noted two, and
//! patch draws' chunks reported so, of a command buffer whose noted patch draws share a byte, or
//! where the emulation was not announced.
//!
//! [`layer_reported_nothing`]: super::layer_reported_nothing

use crate::parity::sequences::hazards::{parse, Named, DISPATCHES};
use std::ops::Range;
#[cfg(any(m2v_gpu, m2v_cpu_icd))]
use {
    super::{LayerClaim, Messages},
    ash::vk::Handle as _,
    metal2vulkan_runtime::{Buffer, DebugMessage, Degradation},
};

/// The id name of the one hazard that may pass.
const WRITE_AFTER_WRITE: &str = "SYNC-HAZARD-WRITE-AFTER-WRITE";

/// The command the index bounds pass records each of its two draws with.
const BOUNDED_DRAW: &str = "vkCmdDrawIndexedIndirect";

/// The command the tessellation emulation records each chunk draw of a patch draw with.
const CHUNK_DRAW: &str = "vkCmdDrawIndirect";

/// The Vulkan commands the runtime records a Metal draw with, as the layer names them.
const DRAWS: &[&str] = &[
    "vkCmdDraw",
    "vkCmdDrawIndexed",
    "vkCmdDrawIndirect",
    BOUNDED_DRAW,
];

/// The Vulkan commands the runtime records a Metal buffer fill of whole words with
/// (`word_fill.rs`), as the layer names them; a fill recorded as a copy is not among them.
const FILLS: &[&str] = &["vkCmdFillBuffer", "vkCmdUpdateBuffer"];

/// The kind of a write a test noted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Writer {
    /// A dispatch's kernel.
    Dispatch,
    /// A draw's shaders.
    Draw,
    /// A blit encoder's fill.
    Fill,
    /// The shaders of an indexed indirect draw the index bounds pass bounds: two draws, at most
    /// one of which draws.
    BoundedDraw,
    /// The post-tessellation vertex function of a patch draw of fewer than `2^30` triangles:
    /// where the runtime emulates tessellation, its chunk draws, at most one of which draws.
    PatchDraw,
}

impl Writer {
    /// The Vulkan commands the runtime records such a write with, as the layer names them.
    fn commands(self) -> &'static [&'static str] {
        match self {
            Writer::Dispatch => DISPATCHES,
            Writer::Draw => DRAWS,
            Writer::Fill => FILLS,
            Writer::BoundedDraw => &[BOUNDED_DRAW],
            Writer::PatchDraw => &[CHUNK_DRAW],
        }
    }

    /// Whether the runtime may record one such write as several Vulkan commands: then two of them
    /// may be one write's, and the apart form does not hold.
    fn split(self) -> bool {
        matches!(self, Writer::BoundedDraw | Writer::PatchDraw)
    }
}

/// Which of the runtime's draws of one Metal draw as several a device reporting to the list
/// announced: the index bounds pass (`indirect-index-bounds-pass`) and the tessellation
/// emulation (`emulated-tessellation-draws`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Announced {
    bounds_pass: bool,
    emulated_tessellation: bool,
}

/// One noted write in its `VkBuffer`: its kind, the raw handle, the bytes there, and the command
/// buffer it was noted in (counted from 0 in the order the test noted them).
#[derive(Clone, Debug, PartialEq, Eq)]
struct Noted {
    writer: Writer,
    buffer: u64,
    bytes: Range<u64>,
    command_buffer: usize,
}

/// Why a message passed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Passed {
    /// Its two writes are noted apart.
    Apart,
    /// Its two draws are one bounded draw's.
    Idle,
    /// Its two draws are chunk draws of patch draws noted apart, or of one.
    IdleChunk,
}

/// Why the layer's message of id name `id_name` and text `text` passes against the writes `noted`,
/// where `announced` says which of the runtime's splits a device announced; `None` where it fails
/// (the module docs).
fn passes(noted: &[Noted], announced: Announced, id_name: &str, text: &str) -> Option<Passed> {
    if id_name != WRITE_AFTER_WRITE {
        return None;
    }
    let hazard = parse(id_name, text)?;
    if !(hazard.later_writes && hazard.earlier_writes) {
        return None;
    }
    let Named::Buffer(buffer) = hazard.memory else {
        return None;
    };
    let on: Vec<&Noted> = noted.iter().filter(|w| w.buffer == buffer).collect();
    if apart(&on, &hazard.later, &hazard.earlier) {
        return Some(Passed::Apart);
    }
    // Two commands of `command`, reported as the later one was recorded, on a buffer every noted
    // write of which is `writer`'s, those of each command buffer one, or where `apart_within`,
    // none empty and no two sharing a byte.
    let idle = |command: &str, writer: Writer, apart_within: bool| {
        text.starts_with(&format!("{command}(): "))
            && hazard.later == command
            && hazard.earlier == command
            && !on.is_empty()
            && on.iter().all(|w| w.writer == writer)
            && (on.iter()).all(|w| {
                let mates: Vec<&Noted> = (on.iter().copied())
                    .filter(|v| v.command_buffer == w.command_buffer)
                    .collect();
                match apart_within {
                    true => disjoint(&mates),
                    false => mates.len() == 1,
                }
            })
    };
    if announced.bounds_pass && idle(BOUNDED_DRAW, Writer::BoundedDraw, false) {
        return Some(Passed::Idle);
    }
    let chunks = announced.emulated_tessellation && idle(CHUNK_DRAW, Writer::PatchDraw, true);
    chunks.then_some(Passed::IdleChunk)
}

/// Whether the writes `on` of one buffer are two or more, each of a kind one Vulkan command
/// records, none empty and no two sharing a byte, and the commands `later` and `earlier` each one
/// that a kind noted among them records.
fn apart(on: &[&Noted], later: &str, earlier: &str) -> bool {
    let noted = |command: &str| on.iter().any(|w| w.writer.commands().contains(&command));
    noted(later)
        && noted(earlier)
        && on.len() >= 2
        && on.iter().all(|w| !w.writer.split())
        && disjoint(on)
}

/// Whether the writes `on` are none empty and no two sharing a byte.
fn disjoint(on: &[&Noted]) -> bool {
    let mut bytes: Vec<&Range<u64>> = on.iter().map(|w| &w.bytes).collect();
    bytes.sort_by_key(|bytes| bytes.start);
    // Sorted by their starts, the writes share no byte when each ends at or before the next starts.
    bytes.iter().all(|bytes| bytes.start < bytes.end)
        && bytes.windows(2).all(|pair| pair[0].end <= pair[1].start)
}

/// The writes a test's commands made of the buffers it notes, each as the test placed it: its
/// kind, the buffer, the bytes from its byte 0, and the command buffer. Made, it holds the
/// layer's hazard reports for its claim ([`Messages::hold_hazards`]), which reads every one.
#[cfg(any(m2v_gpu, m2v_cpu_icd))]
pub(super) struct NotedWrites {
    writes: Vec<(Writer, Buffer, Range<u64>, usize)>,
    command_buffer: usize,
}

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
impl NotedWrites {
    /// No write noted yet, the layer's hazard reports to `messages` held from here on.
    pub(super) fn new(messages: &Messages) -> NotedWrites {
        messages.hold_hazards();
        NotedWrites {
            writes: Vec::new(),
            command_buffer: 0,
        }
    }

    /// Note that a dispatch writes `bytes` of `buffer`.
    pub(super) fn dispatched(&mut self, buffer: &Buffer, bytes: Range<u64>) {
        self.wrote(Writer::Dispatch, buffer, bytes);
    }

    /// Note that a draw's shaders write `bytes` of `buffer`.
    pub(super) fn drew(&mut self, buffer: &Buffer, bytes: Range<u64>) {
        self.wrote(Writer::Draw, buffer, bytes);
    }

    /// Note that a fill writes `bytes` of `buffer`.
    pub(super) fn filled(&mut self, buffer: &Buffer, bytes: Range<u64>) {
        self.wrote(Writer::Fill, buffer, bytes);
    }

    /// Note that an indexed indirect draw the index bounds pass bounds writes `bytes` of `buffer`.
    pub(super) fn drew_bounded(&mut self, buffer: &Buffer, bytes: Range<u64>) {
        self.wrote(Writer::BoundedDraw, buffer, bytes);
    }

    /// Note that a patch draw of fewer than `2^30` triangles writes `bytes` of `buffer` from its
    /// post-tessellation vertex function.
    pub(super) fn drew_patches(&mut self, buffer: &Buffer, bytes: Range<u64>) {
        self.wrote(Writer::PatchDraw, buffer, bytes);
    }

    /// The writes noted from here on are another command buffer's.
    pub(super) fn next_command_buffer(&mut self) {
        self.command_buffer += 1;
    }

    /// Note that a write of `writer`'s kind writes `bytes` of `buffer`. A test notes every write
    /// of a buffer it notes.
    fn wrote(&mut self, writer: Writer, buffer: &Buffer, bytes: Range<u64>) {
        (self.writes).push((writer, buffer.clone(), bytes, self.command_buffer));
    }

    /// Each noted write in its `VkBuffer`, as the layer names it. Asked only once the work ran,
    /// so handing the buffers out changes nothing of it.
    fn in_vk_buffers(&self) -> Vec<Noted> {
        (self.writes.iter())
            .map(|(writer, buffer, bytes, command_buffer)| {
                // SAFETY: the handle is only compared with the layer's reports, never used.
                let (raw, at) = unsafe { buffer.vk_buffer() };
                Noted {
                    writer: *writer,
                    buffer: raw.as_raw(),
                    bytes: at + bytes.start..at + bytes.end,
                    command_buffer: *command_buffer,
                }
            })
            .collect()
    }
}

/// [`layer_reported_nothing`](super::layer_reported_nothing) of `claim`, but for the layer's write
/// after write between two writes at bytes `writes` notes apart, each of which passes with a
/// `LAYER-APART-WRITES` line, between the two draws of one bounded draw `writes` notes alone in
/// its command buffer, each of which passes with a `LAYER-IDLE-DRAW` line, and between chunk draws
/// of patch draws `writes` notes apart in their command buffer, each of which passes with a
/// `LAYER-IDLE-CHUNK` line (the module docs).
/// Each distinct report is judged once and passes or fails with the times it came. A list the
/// sink filled fails: what it dropped past [`Messages::MAX`] is unread.
///
/// # Panics
///
/// Where the layer reported anything else, or filled a list.
#[cfg(any(m2v_gpu, m2v_cpu_icd))]
pub(super) fn layer_reported_nothing_but_apart_writes(
    messages: &Messages,
    claim: &str,
    writes: &NotedWrites,
) -> LayerClaim {
    let held = messages.take_held_hazards();
    let reported = messages.take();
    assert!(
        held.len() < Messages::MAX && reported.len() < Messages::MAX,
        "{claim}: a message list filled, so what it dropped is unread\n{held:#?}\n{reported:#?}"
    );
    let noted = match held.is_empty() && reported.is_empty() {
        true => Vec::new(),
        false => writes.in_vk_buffers(),
    };
    let announced = Announced {
        bounds_pass: messages.announced(Degradation::IndirectIndexBoundsPass),
        emulated_tessellation: messages.announced(Degradation::EmulatedTessellationDraws),
    };
    let test = std::thread::current().name().unwrap_or_default().to_owned();
    let all = (held.into_iter()).chain(reported.into_iter().map(|message| (message, 1)));
    let rest: Vec<(DebugMessage, usize)> = all
        .filter(|(message, times)| {
            let (tag, what) = match passes(&noted, announced, &message.id_name, &message.text) {
                None => return true,
                Some(Passed::Apart) => (
                    "LAYER-APART-WRITES",
                    "a write after write between writes whose bytes the test placed apart",
                ),
                Some(Passed::Idle) => (
                    "LAYER-IDLE-DRAW",
                    "a write after write between the two draws of one bounded draw, at most one \
                     of which draws",
                ),
                Some(Passed::IdleChunk) => (
                    "LAYER-IDLE-CHUNK",
                    "a write after write between chunk draws of patch draws the test placed \
                     apart, at most one chunk of each drawing",
                ),
            };
            eprintln!(
                "{tag} {test}: passed, {what} ({times} time(s)): {} ({}): {}",
                message.id_name,
                message.id,
                message.text.replace('\n', " ")
            );
            false
        })
        .collect();
    let stated = messages.state_layer_claim(claim);
    assert!(
        rest.is_empty(),
        "{claim}: the validation layer reported (each with the times it came)\n{rest:#?}"
    );
    stated
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The buffer the hazards below name.
    const BUFFER: u64 = 0x7400_0000_0074;

    /// Another buffer.
    const OTHER: u64 = 0x5900_0000_0059;

    /// The hazard as the layer 1.4.357 reported it at a submission, between the dispatches of two
    /// command buffers of one batch (`argument_buffers` on lavapipe), naming no shader stage.
    const AT_SUBMIT_357: &str = "vkQueueSubmit(): WRITE_AFTER_WRITE hazard detected. \
        vkCmdDispatch (from VkCommandBuffer 0x7f51840071c0 submitted on the current VkQueue \
        0x7f51b41363f0) writes to VkBuffer 0x740000000074, which was previously written by \
        another vkCmdDispatch command (from VkCommandBuffer 0x7f5184002e10 submitted on VkQueue \
        0x7f51b41363f0). \nNo sufficient synchronization is present to ensure that a write \
        (VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT) at VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT does not \
        conflict with a prior write of the same type at the same stage.";

    /// The same as the layer 1.4.363 reported it, with the shader stage and the command
    /// buffer's place in the submission.
    const AT_SUBMIT: &str = "vkQueueSubmit(): pSubmits[0].pCommandBuffers[3] WRITE_AFTER_WRITE \
        hazard detected. Shader stage VK_SHADER_STAGE_COMPUTE_BIT in vkCmdDispatch (from \
        VkCommandBuffer 0x7f5e8f3637b0 submitted on the current VkQueue 0x7f5e8c131d50) writes \
        to VkBuffer 0x740000000074, which was previously written by another vkCmdDispatch \
        command (from VkCommandBuffer 0x7f5e8f367170 submitted on VkQueue 0x7f5e8c131d50). \n\
        The buffer is referenced by descriptor binding 1 (VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) \
        from descriptor set 0 (VkDescriptorSet 0x0), VkPipeline 0x760000000076.\nNo sufficient \
        synchronization is present to ensure that a write (VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT) \
        at VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT does not conflict with a prior write of the \
        same type at the same stage.";

    /// The hazard as the layer reported it as the later dispatch was recorded.
    const AT_RECORD: &str = "vkCmdDispatch(): WRITE_AFTER_WRITE hazard detected. Shader stage \
        VK_SHADER_STAGE_COMPUTE_BIT writes to VkBuffer 0x740000000074, which was previously \
        written by another vkCmdDispatch command. \nThe buffer is referenced by descriptor \
        binding 1 (VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) from descriptor set 0 (VkDescriptorSet \
        0x0), VkPipeline 0x760000000076.\nNo sufficient synchronization is present to ensure \
        that a write (VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT) at \
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT does not conflict with a prior write of the same \
        type at the same stage.";

    /// Every wording above.
    const FORMS: [&str; 3] = [AT_SUBMIT_357, AT_SUBMIT, AT_RECORD];

    /// A fill's write after a dispatch's, as the layer 1.4.357 reported it at a submission
    /// (`memoryless_uses` on lavapipe).
    const FILL_AFTER_DISPATCH: &str = "vkQueueSubmit(): WRITE_AFTER_WRITE hazard detected. \
        vkCmdFillBuffer (from VkCommandBuffer 0x7f3f49c76530 submitted on the current VkQueue \
        0x7f3f48135d40) writes to VkBuffer 0x740000000074, which was previously written by \
        vkCmdDispatch (from VkCommandBuffer 0x7f3f49c73410 submitted on VkQueue 0x7f3f48135d40). \
        \nNo sufficient synchronization is present to ensure that a write \
        (VK_ACCESS_2_TRANSFER_WRITE_BIT) at VK_PIPELINE_STAGE_2_CLEAR_BIT does not conflict with \
        a prior write (VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT) at \
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT.";

    /// A fill's write after a dispatch's, as the layer reported it as the fill was recorded in the
    /// dispatch's `VkCommandBuffer`, naming the fill's parameter (`memoryless_uses` on lavapipe).
    const FILL_AT_RECORD: &str = "vkCmdFillBuffer(): WRITE_AFTER_WRITE hazard detected. \
        vkCmdFillBuffer writes to dstBuffer VkBuffer 0x740000000074, which was previously written \
        by vkCmdDispatch. \nNo sufficient synchronization is present to ensure that a write \
        (VK_ACCESS_2_TRANSFER_WRITE_BIT) at VK_PIPELINE_STAGE_2_CLEAR_BIT does not conflict with \
        a prior write (VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT) at \
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT.\nBuffer access region: {\n  offset = 12\n  size \
        = 4\n}\n";

    /// A dispatch's write after a fill's, at a submission with a barrier between them that orders
    /// no shader write (`memoryless_uses` on lavapipe).
    const DISPATCH_AFTER_FILL: &str = "vkQueueSubmit(): WRITE_AFTER_WRITE hazard detected. \
        vkCmdDispatch (from VkCommandBuffer 0x7fa965d6d350 submitted on the current VkQueue \
        0x7fa964135d10) writes to VkBuffer 0x740000000074, which was previously written by \
        vkCmdFillBuffer (from VkCommandBuffer 0x7fa965ce7890 submitted on VkQueue \
        0x7fa964135d10).  The current synchronization allows 0, but to prevent this hazard, it \
        must allow VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT accesses at \
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT.";

    /// Two draws of one render encoder, as the layer 1.4.357 reported the later one's recording
    /// (`int_attributes` on lavapipe).
    const DRAW_AT_RECORD: &str = "vkCmdDraw(): WRITE_AFTER_WRITE hazard detected. Shader stage \
        VK_SHADER_STAGE_VERTEX_BIT writes to VkBuffer 0x740000000074, which was previously \
        written by another vkCmdDraw command.  The buffer is referenced by descriptor binding 0 \
        (VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) from descriptor set 0 (VkDescriptorSet 0x0), \
        VkPipeline 0x420000000042. No sufficient synchronization is present to ensure that a \
        write (VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT) at VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT \
        does not conflict with a prior write of the same type at the same stage.";

    /// A draw of an indirect command buffer's execution after the encoder's own, as the layer
    /// 1.4.357 reported it at `vkCmdExecuteCommands` (`render_executions` on lavapipe).
    const DRAW_AT_EXECUTE: &str = "vkCmdExecuteCommands(): WRITE_AFTER_WRITE hazard detected. \
        vkCmdDraw (from the secondary VkCommandBuffer 0x7f8f71dd19a0) writes to VkBuffer \
        0x740000000074, which was previously written by another vkCmdDraw command (from the \
        primary VkCommandBuffer 0x7f8f71ce1090).  No sufficient synchronization is present to \
        ensure that a write (VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT) at \
        VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT does not conflict with a prior write of the same \
        type at the same stage.";

    /// The index bounds pass's two draws, as the layer 1.4.357 reported the second's recording
    /// (`draw_bases` on lavapipe).
    const BOUNDED_AT_RECORD: &str = "vkCmdDrawIndexedIndirect(): WRITE_AFTER_WRITE hazard \
        detected. Shader stage VK_SHADER_STAGE_VERTEX_BIT writes to VkBuffer 0x740000000074, \
        which was previously written by another vkCmdDrawIndexedIndirect command.  The buffer is \
        referenced by descriptor binding 0 (VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) from descriptor \
        set 0 (VkDescriptorSet 0x0), VkPipeline 0x3c000000003c. No sufficient synchronization is \
        present to ensure that a write (VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT) at \
        VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT does not conflict with a prior write of the same \
        type at the same stage.";

    /// A noted write of `writer` at `bytes` of `buffer`, in command buffer `command_buffer`.
    fn noted(writer: Writer, buffer: u64, bytes: Range<u64>, command_buffer: usize) -> Noted {
        Noted {
            writer,
            buffer,
            bytes,
            command_buffer,
        }
    }

    /// `count` words of [`BUFFER`] side by side, each written by `writer`: the four ID words of
    /// `argument_buffers` for dispatches.
    fn words(writer: Writer, count: u64) -> Vec<Noted> {
        (0..count)
            .map(|word| noted(writer, BUFFER, 4 * word..4 * word + 4, 0))
            .collect()
    }

    /// Both splits announced.
    const BOTH: Announced = Announced {
        bounds_pass: true,
        emulated_tessellation: true,
    };

    /// The tessellation emulation announced alone.
    const CHUNKS_ONLY: Announced = Announced {
        bounds_pass: false,
        emulated_tessellation: true,
    };

    /// The index bounds pass announced alone.
    const BOUNDS_ONLY: Announced = Announced {
        bounds_pass: true,
        emulated_tessellation: false,
    };

    /// Whether `text` passes against `noted` with both splits announced, and why.
    fn judged(noted: &[Noted], text: &str) -> Option<Passed> {
        passes(noted, BOTH, WRITE_AFTER_WRITE, text)
    }

    #[test]
    fn dispatches_writing_words_apart_pass_in_either_form() {
        for text in FORMS {
            let two = words(Writer::Dispatch, 2);
            assert_eq!(judged(&two, text), Some(Passed::Apart), "{text}");
            let four = words(Writer::Dispatch, 4);
            assert_eq!(judged(&four, text), Some(Passed::Apart), "{text}");
            // The noted writes of other buffers change nothing.
            let mut more = words(Writer::Dispatch, 2);
            more.push(noted(Writer::Dispatch, OTHER, 0..4, 0));
            assert_eq!(judged(&more, text), Some(Passed::Apart), "{text}");
            // A later dispatch of either other form.
            for command in ["vkCmdDispatchBase", "vkCmdDispatchIndirect"] {
                let other = text.replacen("vkCmdDispatch", command, 1);
                assert_eq!(judged(&two, &other), Some(Passed::Apart), "{other}");
            }
        }
    }

    #[test]
    fn a_fill_and_a_dispatch_apart_pass_either_way_round() {
        let mut beside = vec![noted(Writer::Fill, BUFFER, 0..4, 0)];
        beside.push(noted(Writer::Dispatch, BUFFER, 8..12, 0));
        for text in [FILL_AFTER_DISPATCH, FILL_AT_RECORD, DISPATCH_AFTER_FILL] {
            assert_eq!(judged(&beside, text), Some(Passed::Apart), "{text}");
            // A fill recorded as an update.
            let update = text.replace("vkCmdFillBuffer", "vkCmdUpdateBuffer");
            assert_eq!(judged(&beside, &update), Some(Passed::Apart), "{update}");
            // Sharing a byte.
            let mut over = beside.clone();
            over.push(noted(Writer::Fill, BUFFER, 10..14, 0));
            assert_eq!(judged(&over, text), None, "{text}");
            // A fill the test did not note: two dispatches' writes alone.
            let dispatches = words(Writer::Dispatch, 2);
            assert_eq!(judged(&dispatches, text), None, "{text}");
            // A copy where a fill was noted.
            let copy = text.replace("vkCmdFillBuffer", "vkCmdCopyBuffer");
            assert_eq!(judged(&beside, &copy), None, "{copy}");
        }
    }

    #[test]
    fn draws_writing_their_own_bytes_pass_and_a_shared_byte_fails() {
        for text in [DRAW_AT_RECORD, DRAW_AT_EXECUTE] {
            let rows = words(Writer::Draw, 8);
            assert_eq!(judged(&rows, text), Some(Passed::Apart), "{text}");
            // Each of the runtime's draw commands, at either end.
            for command in ["vkCmdDrawIndexed", "vkCmdDrawIndirect", BOUNDED_DRAW] {
                let later = (text.replacen("vkCmdDraw(): ", &format!("{command}(): "), 1))
                    .replacen("detected. vkCmdDraw ", &format!("detected. {command} "), 1);
                assert_eq!(judged(&rows, &later), Some(Passed::Apart), "{later}");
                let earlier = text.replace("another vkCmdDraw ", &format!("another {command} "));
                assert_eq!(judged(&rows, &earlier), Some(Passed::Apart), "{earlier}");
            }
            // Two draws of one row.
            let mut shared = words(Writer::Draw, 8);
            shared.push(noted(Writer::Draw, BUFFER, 12..16, 0));
            assert_eq!(judged(&shared, text), None, "{text}");
            // Draws noted where the layer names a dispatch, and dispatches where it names draws.
            let dispatch = text.replacen("vkCmdDraw", "vkCmdDispatch", 1);
            assert_eq!(judged(&rows, &dispatch), None, "{dispatch}");
            assert_eq!(judged(&words(Writer::Dispatch, 8), text), None, "{text}");
        }
    }

    #[test]
    fn writes_that_share_a_byte_fail() {
        for text in FORMS {
            let one_word = [words(Writer::Dispatch, 1), words(Writer::Dispatch, 1)].concat();
            assert_eq!(judged(&one_word, text), None, "{text}");
            let mut overlapping = words(Writer::Dispatch, 1);
            overlapping.push(noted(Writer::Dispatch, BUFFER, 8..12, 0));
            overlapping.push(noted(Writer::Dispatch, BUFFER, 2..6, 0));
            assert_eq!(judged(&overlapping, text), None, "{text}");
            let mut empty = words(Writer::Dispatch, 1);
            empty.push(noted(Writer::Dispatch, BUFFER, 4..4, 0));
            assert_eq!(judged(&empty, text), None, "{text}");
        }
    }

    #[test]
    fn a_buffer_the_test_did_not_note_twice_fails() {
        for text in FORMS {
            assert_eq!(judged(&[], text), None, "{text}");
            assert_eq!(judged(&words(Writer::Dispatch, 1), text), None, "{text}");
            let elsewhere: Vec<Noted> = (words(Writer::Dispatch, 2).into_iter())
                .map(|w| Noted { buffer: OTHER, ..w })
                .collect();
            assert_eq!(judged(&elsewhere, text), None, "{text}");
        }
    }

    #[test]
    fn reads_and_racing_writes_fail() {
        let two = words(Writer::Dispatch, 2);
        let read_after_write = AT_RECORD
            .replace("WRITE_AFTER_WRITE", "READ_AFTER_WRITE")
            .replace(" writes to ", " reads ");
        for id_name in ["SYNC-HAZARD-READ-AFTER-WRITE", WRITE_AFTER_WRITE] {
            assert_eq!(passes(&two, BOTH, id_name, &read_after_write), None);
        }
        let write_after_read = AT_RECORD
            .replace("WRITE_AFTER_WRITE", "WRITE_AFTER_READ")
            .replace("previously written by", "previously read by");
        for id_name in ["SYNC-HAZARD-WRITE-AFTER-READ", WRITE_AFTER_WRITE] {
            assert_eq!(passes(&two, BOTH, id_name, &write_after_read), None);
        }
        let racing = AT_RECORD.replace("WRITE_AFTER_WRITE", "WRITE_RACING_WRITE");
        let id_name = "SYNC-HAZARD-WRITE-RACING-WRITE";
        assert_eq!(passes(&two, BOTH, id_name, &racing), None);
        // A bounded draw's pair, read rather than written.
        let bounded = [noted(Writer::BoundedDraw, BUFFER, 0..32, 0)];
        let read = BOUNDED_AT_RECORD
            .replace("WRITE_AFTER_WRITE", "READ_AFTER_WRITE")
            .replace(" writes to ", " reads ");
        let id_name = "SYNC-HAZARD-READ-AFTER-WRITE";
        assert_eq!(passes(&bounded, BOTH, id_name, &read), None);
    }

    #[test]
    fn a_draw_or_a_copy_where_dispatches_were_noted_fails() {
        let two = words(Writer::Dispatch, 2);
        for command in [
            "vkCmdDraw",
            "vkCmdDrawIndexed",
            "vkCmdCopyBuffer",
            "vkCmdFillBuffer",
        ] {
            // The later command.
            let later = AT_RECORD.replacen("vkCmdDispatch", command, 1);
            assert_eq!(judged(&two, &later), None, "{later}");
            let later = AT_SUBMIT.replacen("in vkCmdDispatch", &format!("in {command}"), 1);
            assert_eq!(judged(&two, &later), None, "{later}");
            let later = AT_SUBMIT_357.replacen(
                "detected. vkCmdDispatch",
                &format!("detected. {command}"),
                1,
            );
            assert_eq!(judged(&two, &later), None, "{later}");
            // The earlier one.
            for text in FORMS {
                let earlier = text.replace("another vkCmdDispatch", &format!("another {command}"));
                assert_eq!(judged(&two, &earlier), None, "{earlier}");
            }
        }
        // A copy's own wording, with no shader stage.
        let copy = "vkCmdCopyBuffer(): WRITE_AFTER_WRITE hazard detected. vkCmdCopyBuffer writes \
            to VkBuffer 0x740000000074, which was previously written by vkCmdDispatch command.";
        assert_eq!(judged(&two, copy), None);
    }

    #[test]
    fn an_image_fails() {
        let image = AT_RECORD.replace("VkBuffer 0x740000000074", "VkImage 0x740000000074");
        assert_eq!(judged(&words(Writer::Dispatch, 2), &image), None);
        let image = BOUNDED_AT_RECORD.replace("VkBuffer 0x740000000074", "VkImage 0x740000000074");
        let bounded = [noted(Writer::BoundedDraw, BUFFER, 0..32, 0)];
        assert_eq!(judged(&bounded, &image), None);
    }

    /// One bounded draw's whole output, in each of `command_buffers` command buffers.
    fn bounded(command_buffers: usize) -> Vec<Noted> {
        (0..command_buffers)
            .map(|at| noted(Writer::BoundedDraw, BUFFER, 0..32, at))
            .collect()
    }

    #[test]
    fn the_two_draws_of_a_bounded_draw_alone_in_its_command_buffer_pass() {
        assert_eq!(judged(&bounded(1), BOUNDED_AT_RECORD), Some(Passed::Idle));
        // One such draw in each of several command buffers.
        assert_eq!(judged(&bounded(3), BOUNDED_AT_RECORD), Some(Passed::Idle));
        // The noted writes of other buffers change nothing.
        let mut more = bounded(1);
        more.push(noted(Writer::Draw, OTHER, 0..4, 0));
        more.push(noted(Writer::Draw, OTHER, 0..4, 0));
        assert_eq!(judged(&more, BOUNDED_AT_RECORD), Some(Passed::Idle));
    }

    #[test]
    fn bounded_draws_that_share_a_command_buffer_or_a_byte_with_another_write_fail() {
        // Two bounded draws of one command buffer writing the same bytes: a real overlap.
        let mut two = bounded(1);
        two.push(noted(Writer::BoundedDraw, BUFFER, 0..32, 0));
        assert_eq!(judged(&two, BOUNDED_AT_RECORD), None);
        // And beside another command buffer's one.
        two.push(noted(Writer::BoundedDraw, BUFFER, 0..32, 1));
        assert_eq!(judged(&two, BOUNDED_AT_RECORD), None);
        // A plain draw of the same bytes beside the bounded one.
        let mut beside = bounded(1);
        beside.push(noted(Writer::Draw, BUFFER, 0..4, 0));
        assert_eq!(judged(&beside, BOUNDED_AT_RECORD), None);
        // Beside it in another command buffer: the buffer's writes are not all bounded draws'.
        let mut other = bounded(1);
        other.push(noted(Writer::Draw, BUFFER, 0..4, 1));
        assert_eq!(judged(&other, BOUNDED_AT_RECORD), None);
        // No bounded draw noted on the buffer.
        assert_eq!(judged(&[], BOUNDED_AT_RECORD), None);
        let elsewhere = [noted(Writer::BoundedDraw, OTHER, 0..32, 0)];
        assert_eq!(judged(&elsewhere, BOUNDED_AT_RECORD), None);
    }

    #[test]
    fn bounded_draws_pass_only_as_recorded_both_bounded_and_announced() {
        let one = bounded(1);
        // Not announced.
        let unannounced = passes(&one, CHUNKS_ONLY, WRITE_AFTER_WRITE, BOUNDED_AT_RECORD);
        assert_eq!(unannounced, None);
        // At a submission, or at an execution of secondary command buffers: two command buffers.
        let submitted = format!(
            "vkQueueSubmit(): WRITE_AFTER_WRITE hazard detected. {BOUNDED_DRAW} (from \
             VkCommandBuffer 0x7f00000001 submitted on the current VkQueue 0x7f00000002) writes \
             to VkBuffer 0x740000000074, which was previously written by another {BOUNDED_DRAW} \
             command (from VkCommandBuffer 0x7f00000003 submitted on VkQueue 0x7f00000002)."
        );
        assert_eq!(judged(&one, &submitted), None, "{submitted}");
        let executed = DRAW_AT_EXECUTE.replace("vkCmdDraw ", &format!("{BOUNDED_DRAW} "));
        assert_eq!(judged(&one, &executed), None, "{executed}");
        // Either command another draw.
        for command in ["vkCmdDraw", "vkCmdDrawIndexed", "vkCmdDrawIndirect"] {
            let later = BOUNDED_AT_RECORD.replacen(BOUNDED_DRAW, command, 1);
            assert_eq!(judged(&one, &later), None, "{later}");
            let earlier = BOUNDED_AT_RECORD.replace(
                &format!("another {BOUNDED_DRAW}"),
                &format!("another {command}"),
            );
            assert_eq!(judged(&one, &earlier), None, "{earlier}");
        }
    }

    /// Two chunk draws of an emulated patch draw, as the layer reported the later one's recording
    /// (`patches::emulated_patch_draws_read_their_factors_as_metal` on lavapipe).
    const CHUNK_AT_RECORD: &str = "vkCmdDrawIndirect(): WRITE_AFTER_WRITE hazard detected. Shader \
        stage VK_SHADER_STAGE_VERTEX_BIT writes to VkBuffer 0x740000000074, which was previously \
        written by another vkCmdDrawIndirect command.  The buffer is referenced by descriptor \
        binding 0 (VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) from descriptor set 0 (VkDescriptorSet \
        0x0), VkPipeline 0x470000000047. No sufficient synchronization is present to ensure that a \
        write (VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT) at VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT does \
        not conflict with a prior write of the same type at the same stage.";

    /// A patch draw's marks, of `MARKS` bytes, at mark buffer `at` of one `VkBuffer`, in command
    /// buffer `command_buffer`.
    fn patch_draw(at: u64, command_buffer: usize) -> Noted {
        const MARKS: u64 = 4225 * 16;
        noted(
            Writer::PatchDraw,
            BUFFER,
            at * MARKS..(at + 1) * MARKS,
            command_buffer,
        )
    }

    #[test]
    fn chunks_of_patch_draws_apart_in_their_command_buffer_pass() {
        // One patch draw in each of several command buffers, of the same bytes.
        let alone = [patch_draw(0, 0), patch_draw(0, 1), patch_draw(0, 2)];
        assert_eq!(judged(&alone, CHUNK_AT_RECORD), Some(Passed::IdleChunk));
        // Several in one command buffer, each into a buffer of its own in the `VkBuffer`.
        let own = [patch_draw(0, 0), patch_draw(1, 0), patch_draw(2, 0)];
        assert_eq!(judged(&own, CHUNK_AT_RECORD), Some(Passed::IdleChunk));
        // The noted writes of other buffers change nothing.
        let mut more = vec![patch_draw(0, 0)];
        more.push(noted(Writer::Draw, OTHER, 0..4, 0));
        more.push(noted(Writer::Draw, OTHER, 0..4, 0));
        assert_eq!(judged(&more, CHUNK_AT_RECORD), Some(Passed::IdleChunk));
    }

    #[test]
    fn patch_draws_that_share_a_byte_in_a_command_buffer_or_with_another_write_fail() {
        // Two patch draws of one command buffer writing the same marks: a real overlap.
        let shared = [patch_draw(0, 0), patch_draw(0, 0)];
        assert_eq!(judged(&shared, CHUNK_AT_RECORD), None);
        // A plain draw beside the patch draw, in its command buffer or another.
        for command_buffer in [0, 1] {
            let beside = [
                patch_draw(0, 0),
                noted(Writer::Draw, BUFFER, 0..4, command_buffer),
            ];
            assert_eq!(judged(&beside, CHUNK_AT_RECORD), None);
        }
        // Patch draws noted apart, but more than one Vulkan command each: not the apart form.
        let apart = [patch_draw(0, 0), patch_draw(1, 0)];
        let draws = CHUNK_AT_RECORD.replace("vkCmdDrawIndirect", "vkCmdDraw");
        assert_eq!(judged(&apart, &draws), None, "{draws}");
        // No patch draw noted on the buffer.
        assert_eq!(judged(&[], CHUNK_AT_RECORD), None);
        let elsewhere = [Noted {
            buffer: OTHER,
            ..patch_draw(0, 0)
        }];
        assert_eq!(judged(&elsewhere, CHUNK_AT_RECORD), None);
    }

    #[test]
    fn chunks_pass_only_as_recorded_both_chunks_and_announced() {
        let one = [patch_draw(0, 0)];
        // Not announced.
        let unannounced = passes(&one, BOUNDS_ONLY, WRITE_AFTER_WRITE, CHUNK_AT_RECORD);
        assert_eq!(unannounced, None);
        // At a submission, or at an execution of secondary command buffers: two command buffers.
        let submitted = format!(
            "vkQueueSubmit(): WRITE_AFTER_WRITE hazard detected. {CHUNK_DRAW} (from \
             VkCommandBuffer 0x7f00000001 submitted on the current VkQueue 0x7f00000002) writes \
             to VkBuffer 0x740000000074, which was previously written by another {CHUNK_DRAW} \
             command (from VkCommandBuffer 0x7f00000003 submitted on VkQueue 0x7f00000002)."
        );
        assert_eq!(judged(&one, &submitted), None, "{submitted}");
        let executed = DRAW_AT_EXECUTE.replace("vkCmdDraw ", &format!("{CHUNK_DRAW} "));
        assert_eq!(judged(&one, &executed), None, "{executed}");
        // Either command another draw.
        for command in ["vkCmdDraw", "vkCmdDrawIndexed", BOUNDED_DRAW] {
            let later = CHUNK_AT_RECORD.replacen(CHUNK_DRAW, command, 1);
            assert_eq!(judged(&one, &later), None, "{later}");
            let earlier = CHUNK_AT_RECORD.replace(
                &format!("another {CHUNK_DRAW}"),
                &format!("another {command}"),
            );
            assert_eq!(judged(&one, &earlier), None, "{earlier}");
        }
    }
}
