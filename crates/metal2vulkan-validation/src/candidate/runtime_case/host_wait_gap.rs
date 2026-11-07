//! The one validation-layer message a [`Messages`](super::Messages) sink lets pass: the layer's
//! synchronization validation forgets a semaphore chain once the host waited on a timeline.
//!
//! The layer credits a host wait for a timeline semaphore (`vkGetSemaphoreCounterValue`, or a
//! wait-all `vkWaitSemaphores`) only to the accesses of the queue that signalled it:
//! `SyncValidator::WaitForSemaphore` calls `ApplyTaggedWait` with no queue sync tags, under
//! "TODO: specify queue sync tags argument similar to WaitForFence" (Vulkan-ValidationLayers
//! v1.4.357, commit e4786f7c, `layers/sync/sync_validation.cpp` line 280; v1.4.363, commit
//! 2ea16de8, line 282, `WaitForSemaphore` and `ApplyTaggedWait` unchanged; still there in main
//! 4f8ac9ce of 2026-10-05, line 282). That wait also drops the waited queue's writes from the
//! batches a semaphore chain ran through. So when a batch on `VkQueue` A was ordered before a
//! batch on B by a semaphore, the host read B's timeline past it, and A's next batch waits for
//! B's, the layer finds A's earlier write ordered only by the barrier that ends A's batch for
//! the host, and reports a `WRITE_AFTER_WRITE` between A's two batches. It prints that barrier
//! as "allows 0", since the host's accesses are no queue's. Vulkan orders them through the
//! chain, and the host's read orders them again.
//!
//! A standalone program shows it with no runtime code: two queues, three fills of one buffer,
//! each command buffer ending with the barrier to the host the runtime's batches end with, the
//! host waiting for the middle one (the RTX 5080 and the RTX 3090). Its control, the same chain
//! without the host wait, gets no message.
//!
//! The runtime's queue waiters read the timelines while its scheduler submits, so on a
//! device with several `VkQueue`s a lane test hits it at random: on the RTX 5080,
//! `runtime_lane::exits::an_exit_while_submitting_is_clean` failed 4 to 12 runs in 30. A single
//! `VkQueue` cannot hit it.
//!
//! [`shape`] matches that message and nothing wider. [`read`] keys the layer releases whose
//! source was read with the TODO in place. On any other release, the message fails as before,
//! marked "re-read WaitForSemaphore in layer X.Y.Z", so a fixed or changed layer is checked
//! again rather than trusted.

use ash::vk;

/// `messageIdNumber` of `SYNC-HAZARD-WRITE-AFTER-WRITE`, as the layer hashes its name.
const ID: i32 = 1544472022;

/// The id name of the hazard.
const ID_NAME: &str = "SYNC-HAZARD-WRITE-AFTER-WRITE";

/// The start of the hazard as the layer reports it at a `vkQueueSubmit`: the hazard against a
/// command buffer submitted earlier (`sync_error_messages.cpp`, `AddReplayInfo`).
const SUBMIT_START: &str = "vkQueueSubmit(): WRITE_AFTER_WRITE hazard detected. ";

/// What names the queue of the command buffer being submitted.
const CURRENT_QUEUE: &str = " submitted on the current ";

/// What starts the earlier write.
const PRIOR: &str = "which was previously written by another ";

/// What names the queue of the earlier write's command buffer.
const PRIOR_QUEUE: &str = " submitted on ";

/// The earlier write's barriers: none a queue's access is in (`sync_reporting.cpp`,
/// `FormatErrorMessage` and `FormatSyncAccesses`): the barrier for the host that ends a batch
/// orders nothing later on a queue. A write with no barrier at all reads "No sufficient
/// synchronization is present" instead, which no batch of the runtime's leaves.
const NO_BARRIER: &str = "\nThe current synchronization allows 0, but to prevent this hazard";

/// The layer releases (`VkLayerProperties::specVersion` of `VK_LAYER_KHRONOS_validation`) whose
/// `WaitForSemaphore` was read with the TODO in place: v1.4.357 (tag commit e4786f7c) and
/// v1.4.363 (tag commit 2ea16de8), whose `WaitForSemaphore`, `ApplyTaggedWait` and the text the
/// hazard's message is made of (`sync_error_messages.cpp`, `sync_reporting.cpp`,
/// `FormatErrorMessage`) are those of v1.4.357.
pub(super) const READ: &[u32] = &[
    vk::make_api_version(0, 1, 4, 357),
    vk::make_api_version(0, 1, 4, 363),
];

/// Where the TODO is, for the line each message let pass prints.
pub(super) const SOURCE: &str = "Vulkan-ValidationLayers v1.4.357 (e4786f7c) \
    layers/sync/sync_validation.cpp:280 and v1.4.363 (2ea16de8) :282, \
    SyncValidator::WaitForSemaphore: \"TODO: specify queue \
    sync tags argument similar to WaitForFence\"";

/// Whether a message with this id, id name and text is the gap's: a `WRITE_AFTER_WRITE` found at
/// a `vkQueueSubmit` against a write of a command buffer submitted earlier on the same
/// `VkQueue`, which no barrier for a queue's later access ordered.
pub(super) fn shape(id: i32, id_name: &str, text: &str) -> bool {
    if id != ID || id_name != ID_NAME {
        return false;
    }
    let Some(rest) = text.strip_prefix(SUBMIT_START) else {
        return false;
    };
    let Some(prior_at) = rest.find(PRIOR) else {
        return false;
    };
    let (current, prior) = rest.split_at(prior_at);
    let queue_after = |text: &str, marker: &str| -> Option<String> {
        let start = text.find(marker)? + marker.len();
        let end = start + text[start..].find(')')?;
        let queue = &text[start..end];
        queue.starts_with("VkQueue ").then(|| queue.to_owned())
    };
    let (Some(current_queue), Some(prior_queue)) = (
        queue_after(current, CURRENT_QUEUE),
        queue_after(prior, PRIOR_QUEUE),
    ) else {
        return false;
    };
    current_queue == prior_queue && prior.contains(NO_BARRIER)
}

/// Whether the layer release `version` is one whose source was read with the gap in place.
pub(super) fn read(version: u32) -> bool {
    READ.contains(&version)
}

/// `version` as `X.Y.Z`.
pub(super) fn release(version: u32) -> String {
    format!(
        "{}.{}.{}",
        vk::api_version_major(version),
        vk::api_version_minor(version),
        vk::api_version_patch(version)
    )
}

/// The `specVersion` of the validation layer the loader `loader` (the system's default when
/// `None`) offers; `None` when it cannot be read.
pub(super) fn layer_version(loader: Option<&std::path::Path>) -> Option<u32> {
    // SAFETY: opening the loader library runs only its initializers.
    let entry = unsafe {
        match loader {
            Some(path) => ash::Entry::load_from(path),
            None => ash::Entry::load(),
        }
    }
    .ok()?;
    // SAFETY: a global command of a loaded entry, called with no instance.
    let layers = unsafe { entry.enumerate_instance_layer_properties() }.ok()?;
    layers
        .iter()
        .find(|layer| {
            layer
                .layer_name_as_c_str()
                .is_ok_and(|name| name == super::VALIDATION_LAYER)
        })
        .map(|layer| layer.spec_version)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The message the RTX 5080 reported (`runtime_lane::exits`, run 18).
    const SEEN: &str = "vkQueueSubmit(): WRITE_AFTER_WRITE hazard detected. vkCmdFillBuffer (from \
        VkCommandBuffer 0x7faa33f276b0 submitted on the current VkQueue 0x7faa30e9aab0) writes to \
        VkBuffer 0x1c000000001c, which was previously written by another vkCmdFillBuffer command \
        (from VkCommandBuffer 0x7faa33ebf730 submitted on VkQueue 0x7faa30e9aab0). \nThe current \
        synchronization allows 0, but to prevent this hazard, it must allow \
        VK_ACCESS_2_TRANSFER_WRITE_BIT accesses at VK_PIPELINE_STAGE_2_CLEAR_BIT.";

    #[test]
    fn the_layer_s_message_is_the_gap() {
        assert!(shape(ID, ID_NAME, SEEN));
    }

    #[test]
    fn another_hazard_is_not_the_gap() {
        let read_after_write = SEEN
            .replace("WRITE_AFTER_WRITE", "READ_AFTER_WRITE")
            .replace("writes to", "reads");
        assert!(!shape(
            -1,
            "SYNC-HAZARD-READ-AFTER-WRITE",
            &read_after_write
        ));
        assert!(!shape(ID, "SYNC-HAZARD-READ-AFTER-WRITE", SEEN));
        assert!(!shape(-1, ID_NAME, SEEN));
        let write_after_read = "vkQueueSubmit(): WRITE_AFTER_READ hazard detected. \
            vkCmdFillBuffer (from VkCommandBuffer 0x2 submitted on the current VkQueue 0x1) \
            writes to VkBuffer 0x3, which was previously read by vkCmdCopyBuffer command (from \
            VkCommandBuffer 0x4 submitted on VkQueue 0x1). \nThe current synchronization defines \
            the destination stage mask as 0, but to prevent this hazard, it must include \
            VK_PIPELINE_STAGE_2_CLEAR_BIT.";
        assert!(!shape(ID, ID_NAME, write_after_read));
    }

    #[test]
    fn a_hazard_inside_one_command_buffer_is_not_the_gap() {
        let recorded = "vkCmdFillBuffer(): WRITE_AFTER_WRITE hazard detected. vkCmdFillBuffer \
            writes to VkBuffer 0x1c000000001c, which was previously written by another \
            vkCmdFillBuffer command. \nThe current synchronization allows 0, but to prevent this \
            hazard, it must allow VK_ACCESS_2_TRANSFER_WRITE_BIT accesses at \
            VK_PIPELINE_STAGE_2_CLEAR_BIT.";
        assert!(!shape(ID, ID_NAME, recorded));
    }

    #[test]
    fn an_earlier_write_on_another_queue_is_not_the_gap() {
        let other = SEEN.replace(
            "submitted on VkQueue 0x7faa30e9aab0",
            "submitted on VkQueue 0x7faa30e9a000",
        );
        assert!(!shape(ID, ID_NAME, &other));
    }

    #[test]
    fn an_earlier_write_with_no_barrier_at_all_is_not_the_gap() {
        let start = SEEN.find("\nThe current").unwrap();
        let none = format!(
            "{}\nNo sufficient synchronization is present to ensure that a write \
             (VK_ACCESS_2_TRANSFER_WRITE_BIT) at VK_PIPELINE_STAGE_2_CLEAR_BIT does not conflict \
             with a prior write of the same type at the same stage.",
            &SEEN[..start]
        );
        assert!(!shape(ID, ID_NAME, &none));
    }

    #[test]
    fn an_earlier_write_with_a_barrier_is_not_the_gap() {
        let some = SEEN.replace(
            "allows 0,",
            "allows VK_ACCESS_2_TRANSFER_WRITE_BIT accesses at VK_PIPELINE_STAGE_2_COPY_BIT,",
        );
        assert!(!shape(ID, ID_NAME, &some));
    }

    #[test]
    fn only_the_releases_read_let_it_pass() {
        assert!(read(vk::make_api_version(0, 1, 4, 357)));
        assert!(read(vk::make_api_version(0, 1, 4, 363)));
        assert!(!read(vk::make_api_version(0, 1, 4, 358)));
        assert!(!read(vk::make_api_version(0, 1, 4, 364)));
        assert!(!read(vk::make_api_version(0, 1, 4, 341)));
        assert_eq!(release(vk::make_api_version(0, 1, 4, 358)), "1.4.358");
    }
}
