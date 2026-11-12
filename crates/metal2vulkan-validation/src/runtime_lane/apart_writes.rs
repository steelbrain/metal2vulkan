//! The validation layer's write-after-write hazards between two of a test's own dispatches whose
//! bytes the test placed apart: a lane claim that lets those pass, and nothing wider.
//!
//! The layer's synchronization validation takes a dispatch's storage buffer as written over the
//! whole range its descriptor binds. The runtime binds a buffer slot over the bytes the translation
//! proves the kernel reaches, from the device's offset alignment down and in 16-byte granules, 16
//! bytes at least (`bind::descriptor_range`), and its hazard tracking orders a serial encoder's
//! dispatches by those proven bytes alone (the slot's window, `bind::plan`'s `buffer_use`). So two
//! dispatches that write words 4 bytes apart in one 16-byte block get no barrier between them,
//! and the layer reports a `WRITE_AFTER_WRITE`. No order of the two can change a byte: Vulkan's
//! memory model has a data race only between "operations that access overlapping sets of memory
//! locations", and "a memory operation must not affect memory at a memory location not within its
//! set of memory locations" (the Memory Model appendix). The lane's positive control
//! `sync_coverage::disjoint_words_of_one_block_are_a_hazard_to_the_layer_alone` shows the
//! report beside the bytes, word by word.
//!
//! A test whose dispatches write such words notes each write as it places it ([`DispatchWrites`]:
//! every dispatch write of a buffer it notes), and states its claim with
//! [`layer_reported_nothing_but_apart_writes`] once the work ran. A message passes only when it is
//! a `WRITE_AFTER_WRITE` (not the racing form) between two dispatches ([`DISPATCHES`]) on a
//! `VkBuffer` the test noted two writes or more of, no two of which share a byte; it passes with a
//! `LAYER-APART-WRITES` line. Everything else fails as [`layer_reported_nothing`] fails it: a read
//! after a write, a write after a read, a draw's or a copy's access, a buffer the test did not note
//! or noted one write of, and two noted writes that share a byte (the runtime orders those, so the
//! layer's report of them is a missing barrier).
//!
//! [`layer_reported_nothing`]: super::layer_reported_nothing

use crate::parity::sequences::hazards::{parse, Named, DISPATCHES};
use std::ops::Range;
#[cfg(any(m2v_gpu, m2v_cpu_icd))]
use {
    super::{LayerClaim, Messages},
    ash::vk::Handle as _,
    metal2vulkan_runtime::{Buffer, DebugMessage},
};

/// The id name of the one hazard that may pass.
const WRITE_AFTER_WRITE: &str = "SYNC-HAZARD-WRITE-AFTER-WRITE";

/// Whether the layer's message of id name `id_name` and text `text` is a write after write between
/// two dispatches on a `VkBuffer` of which `writes` (each a raw `VkBuffer` handle and the bytes
/// written in it) holds two writes or more, none empty and no two sharing a byte.
fn apart(writes: &[(u64, Range<u64>)], id_name: &str, text: &str) -> bool {
    if id_name != WRITE_AFTER_WRITE {
        return false;
    }
    let Some(hazard) = parse(id_name, text) else {
        return false;
    };
    let dispatch = |command: &str| DISPATCHES.contains(&command);
    if !(hazard.later_writes
        && hazard.earlier_writes
        && dispatch(&hazard.later)
        && dispatch(&hazard.earlier))
    {
        return false;
    }
    let Named::Buffer(buffer) = hazard.memory else {
        return false;
    };
    let mut bytes: Vec<&Range<u64>> = (writes.iter())
        .filter(|(noted, _)| *noted == buffer)
        .map(|(_, bytes)| bytes)
        .collect();
    bytes.sort_by_key(|bytes| bytes.start);
    // Sorted by their starts, the writes share no byte when each ends at or before the next starts.
    bytes.len() >= 2
        && bytes.iter().all(|bytes| bytes.start < bytes.end)
        && bytes.windows(2).all(|pair| pair[0].end <= pair[1].start)
}

/// The writes a test's dispatches made of the buffers it notes, each as the test placed it: the
/// buffer, and the bytes from its byte 0.
#[cfg(any(m2v_gpu, m2v_cpu_icd))]
#[derive(Default)]
pub(super) struct DispatchWrites(Vec<(Buffer, Range<u64>)>);

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
impl DispatchWrites {
    /// Note that a dispatch writes `bytes` of `buffer`. A test notes every dispatch write of a
    /// buffer it notes.
    pub(super) fn wrote(&mut self, buffer: &Buffer, bytes: Range<u64>) {
        self.0.push((buffer.clone(), bytes));
    }

    /// Each noted write in its `VkBuffer`, as the layer names it: the raw handle, and the bytes
    /// there. Asked only once the work ran, so handing the buffers out changes nothing of it.
    fn in_vk_buffers(&self) -> Vec<(u64, Range<u64>)> {
        (self.0.iter())
            .map(|(buffer, bytes)| {
                // SAFETY: the handle is only compared with the layer's reports, never used.
                let (raw, at) = unsafe { buffer.vk_buffer() };
                (raw.as_raw(), at + bytes.start..at + bytes.end)
            })
            .collect()
    }
}

/// [`layer_reported_nothing`](super::layer_reported_nothing) of `claim`, but for the layer's write
/// after write between two dispatches on a buffer at bytes `writes` notes apart, each of which
/// passes with a `LAYER-APART-WRITES` line. A list the sink filled fails: what it dropped past
/// [`Messages::MAX`] is unread.
///
/// # Panics
///
/// Where the layer reported anything else, or filled the list.
#[cfg(any(m2v_gpu, m2v_cpu_icd))]
pub(super) fn layer_reported_nothing_but_apart_writes(
    messages: &Messages,
    claim: &str,
    writes: &DispatchWrites,
) -> LayerClaim {
    let reported = messages.take();
    assert!(
        reported.len() < Messages::MAX,
        "{claim}: the message list filled, so what it dropped is unread\n{reported:#?}"
    );
    let noted = match reported.is_empty() {
        true => Vec::new(),
        false => writes.in_vk_buffers(),
    };
    let test = std::thread::current().name().unwrap_or_default().to_owned();
    let rest: Vec<DebugMessage> = (reported.into_iter())
        .filter(|message| {
            let passes = apart(&noted, &message.id_name, &message.text);
            if passes {
                eprintln!(
                    "LAYER-APART-WRITES {test}: passed, a write after write between dispatches \
                     whose bytes the test placed apart: {} ({}): {}",
                    message.id_name,
                    message.id,
                    message.text.replace('\n', " ")
                );
            }
            !passes
        })
        .collect();
    let stated = messages.state_layer_claim(claim);
    assert!(
        rest.is_empty(),
        "{claim}: the validation layer reported\n{rest:#?}"
    );
    stated
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The buffer the hazards below name.
    const BUFFER: u64 = 0x7400_0000_0074;

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

    /// Two words 4 bytes apart, and two more after them: the four ID words of `argument_buffers`.
    fn words(count: u64) -> Vec<(u64, Range<u64>)> {
        (0..count)
            .map(|word| (BUFFER, 4 * word..4 * word + 4))
            .collect()
    }

    #[test]
    fn dispatches_writing_words_apart_pass_in_either_form() {
        for text in FORMS {
            assert!(apart(&words(2), WRITE_AFTER_WRITE, text), "{text}");
            assert!(apart(&words(4), WRITE_AFTER_WRITE, text), "{text}");
            // The noted writes of other buffers change nothing.
            let mut more = words(2);
            more.push((0x5900_0000_0059, 0..4));
            assert!(apart(&more, WRITE_AFTER_WRITE, text), "{text}");
            // A later dispatch of either other form.
            for command in ["vkCmdDispatchBase", "vkCmdDispatchIndirect"] {
                let other = text.replacen("vkCmdDispatch", command, 1);
                assert!(apart(&words(2), WRITE_AFTER_WRITE, &other), "{other}");
            }
        }
    }

    #[test]
    fn writes_that_share_a_byte_fail() {
        for text in FORMS {
            let one_word = vec![(BUFFER, 0..4), (BUFFER, 0..4)];
            assert!(!apart(&one_word, WRITE_AFTER_WRITE, text), "{text}");
            let overlapping = vec![(BUFFER, 0..4), (BUFFER, 8..12), (BUFFER, 2..6)];
            assert!(!apart(&overlapping, WRITE_AFTER_WRITE, text), "{text}");
            let empty = vec![(BUFFER, 0..4), (BUFFER, 4..4)];
            assert!(!apart(&empty, WRITE_AFTER_WRITE, text), "{text}");
        }
    }

    #[test]
    fn a_buffer_the_test_did_not_note_twice_fails() {
        for text in FORMS {
            assert!(!apart(&[], WRITE_AFTER_WRITE, text), "{text}");
            assert!(!apart(&words(1), WRITE_AFTER_WRITE, text), "{text}");
            let elsewhere: Vec<_> = (words(2).into_iter())
                .map(|(_, bytes)| (0x5900_0000_0059, bytes))
                .collect();
            assert!(!apart(&elsewhere, WRITE_AFTER_WRITE, text), "{text}");
        }
    }

    #[test]
    fn reads_and_racing_writes_fail() {
        let read_after_write = AT_RECORD
            .replace("WRITE_AFTER_WRITE", "READ_AFTER_WRITE")
            .replace(" writes to ", " reads ");
        for id_name in ["SYNC-HAZARD-READ-AFTER-WRITE", WRITE_AFTER_WRITE] {
            assert!(!apart(&words(2), id_name, &read_after_write), "{id_name}");
        }
        let write_after_read = AT_RECORD
            .replace("WRITE_AFTER_WRITE", "WRITE_AFTER_READ")
            .replace("previously written by", "previously read by");
        for id_name in ["SYNC-HAZARD-WRITE-AFTER-READ", WRITE_AFTER_WRITE] {
            assert!(!apart(&words(2), id_name, &write_after_read), "{id_name}");
        }
        let racing = AT_RECORD.replace("WRITE_AFTER_WRITE", "WRITE_RACING_WRITE");
        assert!(!apart(&words(2), "SYNC-HAZARD-WRITE-RACING-WRITE", &racing));
    }

    #[test]
    fn a_draw_or_a_copy_at_either_end_fails() {
        for command in [
            "vkCmdDraw",
            "vkCmdDrawIndexed",
            "vkCmdCopyBuffer",
            "vkCmdFillBuffer",
        ] {
            // The later command.
            let later = AT_RECORD.replacen("vkCmdDispatch", command, 1);
            assert!(!apart(&words(2), WRITE_AFTER_WRITE, &later), "{later}");
            let later = AT_SUBMIT.replacen("in vkCmdDispatch", &format!("in {command}"), 1);
            assert!(!apart(&words(2), WRITE_AFTER_WRITE, &later), "{later}");
            let later = AT_SUBMIT_357.replacen(
                "detected. vkCmdDispatch",
                &format!("detected. {command}"),
                1,
            );
            assert!(!apart(&words(2), WRITE_AFTER_WRITE, &later), "{later}");
            // The earlier one.
            for text in FORMS {
                let earlier = text.replace("another vkCmdDispatch", &format!("another {command}"));
                assert!(!apart(&words(2), WRITE_AFTER_WRITE, &earlier), "{earlier}");
            }
        }
        // A copy's own wording, with no shader stage.
        let copy = "vkCmdCopyBuffer(): WRITE_AFTER_WRITE hazard detected. vkCmdCopyBuffer writes \
            to VkBuffer 0x740000000074, which was previously written by vkCmdDispatch command.";
        assert!(!apart(&words(2), WRITE_AFTER_WRITE, copy));
    }

    #[test]
    fn an_image_fails() {
        let image = AT_RECORD.replace("VkBuffer 0x740000000074", "VkImage 0x740000000074");
        assert!(!apart(&words(2), WRITE_AFTER_WRITE, &image));
    }
}
