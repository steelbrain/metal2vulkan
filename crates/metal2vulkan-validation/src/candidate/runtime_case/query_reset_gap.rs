//! A validation-layer gap a [`Messages`](super::Messages) sink lets pass: the layer's check of
//! `VUID-vkResetQueryPool-firstQuery-02741` reports a whole query pool in use where the spec
//! speaks of the range reset.
//!
//! The valid usage reads: "Submitted commands that refer to the range specified by `firstQuery`
//! and `queryCount` in `queryPool` must have completed execution" (`vkResetQueryPool`). The
//! layer checks it so: for each query of the range, "if (query_state != QUERYSTATE_AVAILABLE &&
//! query_state != QUERYSTATE_UNKNOWN)", it calls `ValidateObjectNotInUse(query_pool_state.get(),
//! ...)`, under "Same idea as VUID-vkDestroyQueryPool-queryPool-00793" (Vulkan-ValidationLayers
//! v1.4.363, tag commit 2ea16de8, and vulkan-sdk-1.4.363.1, 317cff06,
//! `layers/core_checks/cc_query.cpp`, `CoreChecks::PreCallValidateResetQueryPool`). That reports
//! the pool in use by any command buffer in flight that refers to any of its queries. A query
//! whose command buffer completed stays out of the two states it accepts, so a reset of it while
//! another range of the same pool is in flight is reported. Main later checks only a query still
//! `QUERYSTATE_ENDED`, "only ENDED between the submission that wrote it and that submission
//! completing (or its results being read)" (main of 2026-11-07); v1.4.357 has no check.
//!
//! The runtime's one host reset is a timestamp slot's (`runtime/src/gpu_time.rs`, `slot`): a
//! slot's queries are reset as a command buffer takes it, and a slot is taken only once given
//! back, after its last holding's command buffer completed and its timestamps were read, or was
//! never submitted. Its pool holds other slots, which other command buffers in flight hold, so
//! every lane that keeps two timed command buffers in flight at once meets the report: on
//! lavapipe and on RADV (a Radeon 780M) with the layer 1.4.363,
//! `runtime_lane::staging_batches::batches_after_one_that_waited_for_a_staging_copy_follow_its_wait`
//! failed every run, and both `runtime_lane::replayed_frames` tests that keep a frame in flight
//! failed some runs.
//!
//! [`shape`] matches that message for a `vkResetQueryPool` call and nothing wider; [`read`] keys
//! the layer releases whose source was read with the whole-pool check in place. The message
//! cannot tell the range's own use from another range's, so where it passes the reset is checked
//! as by v1.4.357, which has no check. On any other release it fails, marked "re-read
//! vkResetQueryPool in layer X.Y.Z", so a fixed or changed layer is checked again rather than
//! trusted.

use ash::vk;

/// `messageIdNumber` of `VUID-vkResetQueryPool-firstQuery-02741`, as the layer hashes its name.
const ID: i32 = 150880399;

/// The id name of the valid usage.
const ID_NAME: &str = "VUID-vkResetQueryPool-firstQuery-02741";

/// The start of the message: the call, and the pool (`ValidateObjectNotInUse`).
const CALL: &str = "vkResetQueryPool(): can't be called on VkQueryPool 0x";

/// What follows the pool: the command buffer in flight that holds it.
const IN_USE: &str = " that is currently in use by VkCommandBuffer 0x";

/// The end of the message: the valid usage's text, as the layer 1.4.363 quotes it.
const STATES: &str = "\nThe Vulkan spec states: Submitted commands that refer to the range \
    specified by firstQuery and queryCount in queryPool must have completed execution \
    (https://docs.vulkan.org/spec/latest/chapters/queries.html#VUID-vkResetQueryPool-firstQuery-02741)";

/// The note the layer puts before a message it reports for the last time
/// (`duplicate_message_limit`).
const LAST_TIME: &str = "(Warning - This VUID has now been reported ";

/// The end of that note.
const LAST_TIME_END: &str = ", this will be the last time reporting it).\n";

/// The layer releases (`VkLayerProperties::specVersion` of `VK_LAYER_KHRONOS_validation`) whose
/// `PreCallValidateResetQueryPool` was read with the whole-pool check in place: v1.4.363 (tag
/// commits 2ea16de8 and, for vulkan-sdk-1.4.363.1, 317cff06).
pub(super) const READ: &[u32] = &[vk::make_api_version(0, 1, 4, 363)];

/// Where the gap is, for the line each message let pass prints.
pub(super) const SOURCE: &str = "Vulkan-ValidationLayers v1.4.363 (2ea16de8; \
    vulkan-sdk-1.4.363.1, 317cff06) layers/core_checks/cc_query.cpp, \
    CoreChecks::PreCallValidateResetQueryPool: a query of the range neither AVAILABLE nor UNKNOWN \
    calls ValidateObjectNotInUse on the whole pool (\"Same idea as \
    VUID-vkDestroyQueryPool-queryPool-00793\")";

/// Whether a message with this id, id name and text is the gap's: `vkResetQueryPool` reported
/// for a pool in use by a command buffer, with the valid usage's text, and nothing else.
pub(super) fn shape(id: i32, id_name: &str, text: &str) -> bool {
    if id != ID || id_name != ID_NAME {
        return false;
    }
    let text = match text.strip_prefix(LAST_TIME) {
        Some(note) => match note.find(LAST_TIME_END) {
            Some(end) if !note[..end].contains('\n') => &note[end + LAST_TIME_END.len()..],
            _ => return false,
        },
        None => text,
    };
    let Some(rest) = text.strip_prefix(CALL) else {
        return false;
    };
    let Some(rest) = after_handle(rest).and_then(|rest| rest.strip_prefix(IN_USE)) else {
        return false;
    };
    after_handle(rest) == Some(STATES)
}

/// What follows the hexadecimal digits a handle begins `text` with, and the object's debug name
/// in brackets where the layer prints one; `None` without any digit.
fn after_handle(text: &str) -> Option<&str> {
    let end = text
        .find(|c: char| !c.is_ascii_hexdigit())
        .unwrap_or(text.len());
    if end == 0 {
        return None;
    }
    let rest = &text[end..];
    match rest.strip_prefix('[') {
        Some(named) => {
            let close = named.find(']')?;
            (!named[..close].contains('\n')).then(|| &named[close + 1..])
        }
        None => Some(rest),
    }
}

/// Whether the layer release `version` is one whose source was read with the gap in place.
pub(super) fn read(version: u32) -> bool {
    READ.contains(&version)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The message as the lanes saw it on lavapipe (Mesa 26.2.6, LLVM 23.1.4) and RADV with the
    /// layer 1.4.363.
    fn seen() -> String {
        format!(
            "vkResetQueryPool(): can't be called on VkQueryPool 0x340000000034 that is currently \
             in use by VkCommandBuffer 0x7f28f5e85630{STATES}"
        )
    }

    #[test]
    fn the_layer_s_report_of_a_pool_in_use_at_a_host_reset_is_the_gap() {
        assert!(shape(ID, ID_NAME, &seen()));
        let last = format!(
            "{LAST_TIME}10 times, which is the duplicate_message_limit value{LAST_TIME_END}{}",
            seen()
        );
        assert!(shape(ID, ID_NAME, &last));
        // A command buffer, or the pool, with a debug name.
        let named = seen().replace("0x7f28f5e85630", "0x7f28f5e85630[a label]");
        assert!(shape(ID, ID_NAME, &named), "a named command buffer");
        let pool = seen().replace("0x340000000034", "0x340000000034[timestamps]");
        assert!(shape(ID, ID_NAME, &pool), "a named pool");
    }

    #[test]
    fn other_messages_are_not_the_gap() {
        let text = seen();
        assert!(!shape(0, ID_NAME, &text));
        assert!(!shape(ID, "VUID-vkResetQueryPool-firstQuery-09436", &text));
        // Another call, another object, another holder.
        let destroy = text.replacen("vkResetQueryPool(): ", "vkDestroyQueryPool(): ", 1);
        assert!(!shape(ID, ID_NAME, &destroy), "another call");
        let buffer = text.replace("VkQueryPool 0x", "VkBuffer 0x");
        assert!(!shape(ID, ID_NAME, &buffer), "another object");
        let queue = text.replace("VkCommandBuffer 0x", "VkQueue 0x");
        assert!(!shape(ID, ID_NAME, &queue), "another holder");
        // No handle, another valid usage's text, or more after it.
        let unhandled = text.replace("0x340000000034", "0x");
        assert!(!shape(ID, ID_NAME, &unhandled), "no handle");
        let other = text.replace("must have completed execution", "must not be in use");
        assert!(!shape(ID, ID_NAME, &other), "another text");
        let more = format!("{text}\nSomething else.");
        assert!(!shape(ID, ID_NAME, &more), "more after it");
        // Anything else before the call.
        let before = format!("Something else.\n{text}");
        assert!(!shape(ID, ID_NAME, &before), "another preface");
        let broken = format!("{LAST_TIME}10 times\nand more{LAST_TIME_END}{text}");
        assert!(!shape(ID, ID_NAME, &broken), "a note over two lines");
    }

    #[test]
    fn only_the_releases_read_pass() {
        assert!(read(vk::make_api_version(0, 1, 4, 363)));
        assert!(!read(vk::make_api_version(0, 1, 4, 357)));
        assert!(!read(vk::make_api_version(0, 1, 4, 364)));
        assert!(!read(vk::make_api_version(0, 1, 5, 0)));
    }
}
