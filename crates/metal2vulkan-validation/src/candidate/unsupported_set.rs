//! The validation layer's message about a descriptor set layout a driver refuses through
//! `vkGetDescriptorSetLayoutSupport` though it makes and runs it.
//!
//! MoltenVK answers the query `sum of descriptorCount < maxPerSetDescriptors` (1212 on Apple's
//! GPUs), though it encodes an update-after-bind set in a Metal argument buffer and runs one of
//! 65536 elements and more as Metal runs the argument buffer. The executor's context makes a heap's
//! set 1 there past that answer (`platform/device.rs`). The layer asks the query for
//! every layout past `maxPerSetDescriptors` and reports one it refuses, as Vulkan says it must
//! (`VUID-vkCreateDescriptorSetLayout-support-09582`, "If the descriptor layout exceeds the limits
//! reported through the physical device limits, then vkGetDescriptorSetLayoutSupport must have
//! returned VkDescriptorSetLayoutSupport with support equal to VK_TRUE for pCreateInfo"). That is
//! no layer defect: the use is outside what the driver's query admits, and the measured defect is
//! what licenses it.
//!
//! [`shape`] matches exactly that message: the layer's format, "`pCreateInfo` was passed to
//! vkGetDescriptorSetLayoutSupport which reported this layout is not supported", read in
//! Vulkan-ValidationLayers 1.4.363 (`CoreChecks::ValidateDescriptorSetLayoutSupport`), at
//! `vkCreateDescriptorSetLayout`, as the lanes saw it on MoltenVK 1.4.2, alone or after the
//! layer's note on its last report (`duplicate_message_limit`). The executor's context's sink
//! passes it only on a device whose driver runs such a layout (`platform/device.rs`). Every other
//! message, and this one anywhere else, still fails.

/// The message's id name.
pub(crate) const ID_NAME: &str = "VUID-vkCreateDescriptorSetLayout-support-09582";

/// What the message says of a layout the driver's query refused.
const REFUSED: &str = "vkCreateDescriptorSetLayout(): pCreateInfo was passed to \
                       vkGetDescriptorSetLayoutSupport which reported this layout is not \
                       supported";

/// The note the layer puts on a line of its own before a message it reports for the last time
/// (`duplicate_message_limit`).
const LAST_TIME: &str = "(Warning - This VUID has now been reported ";

/// Whether a message with this id name and text is the layer's report of a layout the driver's
/// support query refused: the report, alone or after the layer's last-time note.
pub(crate) fn shape(id_name: &str, text: &str) -> bool {
    let report = match text.strip_prefix(LAST_TIME) {
        Some(noted) => noted.split_once('\n').map_or("", |(_, report)| report),
        None => text,
    };
    id_name == ID_NAME && report.starts_with(REFUSED)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The message as the layer gave it for a heap's set 1 of `total` descriptors on MoltenVK
    /// 1.4.2 (layer 1.4.363).
    fn seen(total: u32) -> String {
        format!(
            "vkCreateDescriptorSetLayout(): pCreateInfo was passed to \
             vkGetDescriptorSetLayoutSupport which reported this layout is not supported.\nThe \
             layout has {total} total descriptors which is more than maxPerSetDescriptors \
             (1212).\nThe Vulkan spec states: If the descriptor layout exceeds the limits \
             reported through the physical device limits, then vkGetDescriptorSetLayoutSupport \
             must have returned VkDescriptorSetLayoutSupport with support equal to VK_TRUE for \
             pCreateInfo (https://docs.vulkan.org/spec/latest/chapters/descriptorsets.html\
             #VUID-vkCreateDescriptorSetLayout-support-09582)"
        )
    }

    #[test]
    fn a_refused_layout_s_message_is_the_shape() {
        for total in [65536, 131072, 393216] {
            assert!(shape(ID_NAME, &seen(total)), "{total}");
            // Reported for the last time, after the layer's note.
            let last = format!(
                "{LAST_TIME}10 times, which is the duplicate_message_limit value, this will be \
                 the last time reporting it).\n{}",
                seen(total)
            );
            assert!(shape(ID_NAME, &last), "{total}, the last time");
        }
    }

    #[test]
    fn another_message_is_not() {
        let text = seen(65536);
        assert!(!shape(
            "VUID-vkCreateDescriptorSetLayout-flags-00281",
            &text
        ));
        assert!(!shape("", &text));
        // The note before another message.
        let noted = format!(
            "{LAST_TIME}10 times, which is the duplicate_message_limit value, this will be the \
             last time reporting it).\nvkCreatePipelineLayout(): {text}"
        );
        assert!(!shape(ID_NAME, &noted));
        assert!(!shape(
            ID_NAME,
            "vkCreatePipelineLayout(): pCreateInfo was passed to vkGetDescriptorSetLayoutSupport"
        ));
    }
}
