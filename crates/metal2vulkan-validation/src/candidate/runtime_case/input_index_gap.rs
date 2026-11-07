//! A validation-layer gap a [`Messages`](super::Messages) sink lets pass: the layer's link-time
//! check of a fragment shader's input attachments reads a pipeline made against a `VkRenderPass`
//! as if it were made for dynamic rendering, and reports every input attachment it reads.
//!
//! `VUID-VkGraphicsPipelineCreateInfo-renderPass-09652` holds "If the pipeline is being created
//! with fragment shader state and fragment output state, the value of renderPass is
//! `VK_NULL_HANDLE`, and `VkRenderingInputAttachmentIndexInfo` is not included": it is a rule of
//! dynamic rendering, where input attachment `i` is colour attachment `i`. In a subpass, input
//! attachment `i` is `pInputAttachments[i]`, and the render pass is part of both states, so a
//! pipeline linked from libraries made against one has it.
//!
//! In layer v1.4.357 (tag commit e4786f7c) the check runs only for dynamic rendering:
//! `CoreChecks::ValidateShaderInputAttachment` (`layers/core_checks/cc_spirv.cpp` line 87) tests
//! `rp_state->UsesDynamicRendering()` first. v1.4.363 (tag commit 2ea16de8) adds a second pass
//! at link time, in `CoreChecks::ValidateInterfaceFragmentOutput`
//! (`layers/core_checks/cc_shader_interface.cpp` line 221), for a pipeline that takes its
//! fragment shader or fragment output state from a library, and that pass tests no such thing:
//! it calls `ValidateShaderInputAttachmentDynamicRendering` with the libraries' render pass,
//! whose `dynamic_pipeline_rendering_create_info` a `VkRenderPass` never sets
//! (`layers/state_tracker/render_pass_state.cpp` lines 340 and 357), so it compares each index
//! with a colour attachment count of 0. The layer's main branch adds the missing test in
//! 65fdd118 of 2026-10-03, "layers: Fix 09652 false positive", after the v1.4.363 tag.
//!
//! The runtime makes every graphics pipeline against a `VkRenderPass`, and links one from
//! libraries where the device enables `VK_EXT_graphics_pipeline_library` and links them fast
//! (`runtime/src/pipeline/libraries.rs`), so on layer v1.4.363 every linked pipeline that fetches
//! through an input attachment reports each one it reads (lavapipe links them). [`shape`]
//! matches only the second pass's message, word for word but for its numbers: at the create
//! info's location (the per-stage check reports at `pStages[i]`), against a count of 0. [`read`]
//! keys the layer releases whose source was read with the gap in place. On any other release
//! such a message fails, marked "re-read the link-time input attachment check in layer X.Y.Z",
//! so a fixed or changed layer is checked again rather than trusted.

use ash::vk;

/// `messageIdNumber` of the valid usage, as the layer hashes its name.
const ID: i32 = 454563899;

/// The id name of the valid usage.
const ID_NAME: &str = "VUID-VkGraphicsPipelineCreateInfo-renderPass-09652";

/// The start of the message: the create info's location, the second pass's (`create_info_loc`).
const CALL: &str = "vkCreateGraphicsPipelines(): pCreateInfos[";

/// What follows the create info's index: the fragment shader's input attachment
/// (`DescribeInputAttachmentIndex`), its set and binding.
const SHADER: &str = "] shader [VK_SHADER_STAGE_FRAGMENT_BIT] [Set ";

/// The end of the message's first line: compared with the colour attachment count a
/// `VkRenderPass`'s state holds, 0.
const COUNT: &str =
    " which is not less than VkPipelineRenderingCreateInfo::colorAttachmentCount (0)";

/// The rest of the message, after its first line, as layer v1.4.363 words it.
const REST: &str = "If VkRenderingInputAttachmentIndexInfo is provided, the index can be set, \
    but without it, it uses the default index values.\nHint: If using Depth/Stencil attachments, \
    you should be omitting the input attachment index in the shader to access the depth/stencil \
    attachments.\nThe Vulkan spec states: If the pipeline is being created with fragment shader \
    state and fragment output state, the value of renderPass is VK_NULL_HANDLE, and \
    VkRenderingInputAttachmentIndexInfo is not included, the fragment shader must not contain any \
    input attachments with a InputAttachmentIndex greater than or equal to \
    VkPipelineRenderingCreateInfo::colorAttachmentCount \
    (https://docs.vulkan.org/spec/latest/chapters/pipelines.html\
    #VUID-VkGraphicsPipelineCreateInfo-renderPass-09652)";

/// The note the layer puts before a message it reports for the last time
/// (`duplicate_message_limit`), around the count.
const LAST_TIME: (&str, &str) = (
    "(Warning - This VUID has now been reported ",
    " times, which is the duplicate_message_limit value, this will be the last time reporting \
     it).\n",
);

/// The layer releases (`VkLayerProperties::specVersion` of `VK_LAYER_KHRONOS_validation`) whose
/// source was read with the gap in place: v1.4.363 (tag commit 2ea16de8).
pub(super) const READ: &[u32] = &[vk::make_api_version(0, 1, 4, 363)];

/// Where the gap is, for the line each message let pass prints.
pub(super) const SOURCE: &str = "Vulkan-ValidationLayers v1.4.363 (2ea16de8) \
    layers/core_checks/cc_shader_interface.cpp:221, CoreChecks::ValidateInterfaceFragmentOutput: \
    the link-time input attachment pass does not test rp_state->UsesDynamicRendering() \
    (added in main 65fdd118, \"layers: Fix 09652 false positive\")";

/// Whether a message with this id, id name and text is the gap's: the link-time pass's report of
/// a fragment shader's input attachment at the create info's location, against a colour
/// attachment count of 0, word for word as layer v1.4.363 reports it but for its numbers.
pub(super) fn shape(id: i32, id_name: &str, text: &str) -> bool {
    if id != ID || id_name != ID_NAME {
        return false;
    }
    // Before the message, nothing or the layer's last-time note.
    let text = match text.strip_prefix(LAST_TIME.0) {
        Some(note) => match number(note).and_then(|rest| rest.strip_prefix(LAST_TIME.1)) {
            Some(rest) => rest,
            None => return false,
        },
        None => text,
    };
    let Some((first, rest)) = text.split_once('\n') else {
        return false;
    };
    let Some(index) = first.strip_prefix(CALL).and_then(number) else {
        return false;
    };
    let Some(binding) = index.strip_prefix(SHADER).and_then(number) else {
        return false;
    };
    let Some(read) = binding
        .strip_prefix(", Binding ")
        .and_then(number)
        .and_then(|rest| rest.strip_prefix("] has an "))
    else {
        return false;
    };
    // The index the shader reads, or an array's element of one.
    let end = match read.strip_prefix("InputAttachmentIndex of ") {
        Some(rest) => number(rest),
        None => read
            .strip_prefix("effective InputAttachmentIndex of ")
            .and_then(number)
            .and_then(|rest| rest.strip_prefix(" (started at InputAttachmentIndex "))
            .and_then(number)
            .and_then(|rest| rest.strip_prefix(" plus index "))
            .and_then(number)
            .and_then(|rest| rest.strip_prefix(" into the array)")),
    };
    end == Some(COUNT) && rest == REST
}

/// What follows the decimal digits `text` starts with; `None` without any.
fn number(text: &str) -> Option<&str> {
    let digits = text.bytes().take_while(u8::is_ascii_digit).count();
    (digits > 0).then(|| &text[digits..])
}

/// Whether the layer release `version` is one whose source was read with the gap in place.
pub(super) fn read(version: u32) -> bool {
    READ.contains(&version)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};

    /// The message as the lanes saw it on lavapipe (Mesa, LLVM 23.1.4) with the layer 1.4.363,
    /// word for word.
    fn seen() -> String {
        "vkCreateGraphicsPipelines(): pCreateInfos[0] shader [VK_SHADER_STAGE_FRAGMENT_BIT] \
         [Set 0, Binding 2] has an InputAttachmentIndex of 0 which is not less than \
         VkPipelineRenderingCreateInfo::colorAttachmentCount (0)\nIf \
         VkRenderingInputAttachmentIndexInfo is provided, the index can be set, but without it, \
         it uses the default index values.\nHint: If using Depth/Stencil attachments, you should \
         be omitting the input attachment index in the shader to access the depth/stencil \
         attachments.\nThe Vulkan spec states: If the pipeline is being created with fragment \
         shader state and fragment output state, the value of renderPass is VK_NULL_HANDLE, and \
         VkRenderingInputAttachmentIndexInfo is not included, the fragment shader must not \
         contain any input attachments with a InputAttachmentIndex greater than or equal to \
         VkPipelineRenderingCreateInfo::colorAttachmentCount \
         (https://docs.vulkan.org/spec/latest/chapters/pipelines.html\
         #VUID-VkGraphicsPipelineCreateInfo-renderPass-09652)"
            .to_owned()
    }

    #[test]
    fn the_link_time_report_of_an_input_attachment_is_the_gap() {
        assert!(shape(ID, ID_NAME, &seen()));
        let last = format!(
            "(Warning - This VUID has now been reported 10 times, which is the \
             duplicate_message_limit value, this will be the last time reporting it).\n{}",
            seen()
        );
        assert!(shape(ID, ID_NAME, &last));
        // Another create info, binding or index (an index the layer cannot resolve reads the
        // same), or an array's element.
        let other = seen()
            .replacen("pCreateInfos[0]", "pCreateInfos[12]", 1)
            .replacen(
                "Binding 2] has an InputAttachmentIndex of 0",
                "Binding 5] has an InputAttachmentIndex of 3",
                1,
            );
        assert!(shape(ID, ID_NAME, &other));
        let element = seen().replacen(
            "has an InputAttachmentIndex of 0",
            "has an effective InputAttachmentIndex of 1 (started at InputAttachmentIndex 0 plus \
             index 1 into the array)",
            1,
        );
        assert!(shape(ID, ID_NAME, &element));
    }

    #[test]
    fn the_per_stage_check_and_a_real_colour_count_are_not_the_gap() {
        let text = seen();
        // The per-stage check, which runs only for dynamic rendering, reports at the stage.
        let stage = text.replacen(
            "pCreateInfos[0] shader",
            "pCreateInfos[0].pStages[1] shader",
            1,
        );
        assert!(!shape(ID, ID_NAME, &stage), "at a stage");
        // A count of colour attachments a dynamic-rendering pipeline states.
        let counted = text.replacen(
            "colorAttachmentCount (0)\n",
            "colorAttachmentCount (2)\n",
            1,
        );
        assert!(!shape(ID, ID_NAME, &counted), "a stated count");
        // Another call, stage, valid usage or message.
        let call = text.replacen("vkCreateGraphicsPipelines()", "vkCreateShadersEXT()", 1);
        assert!(!shape(ID, ID_NAME, &call), "another call");
        let vertex = text.replacen(
            "VK_SHADER_STAGE_FRAGMENT_BIT",
            "VK_SHADER_STAGE_VERTEX_BIT",
            1,
        );
        assert!(!shape(ID, ID_NAME, &vertex), "another stage");
        assert!(!shape(
            ID,
            "VUID-VkGraphicsPipelineCreateInfo-renderPass-09531",
            &text
        ));
        assert!(!shape(0, ID_NAME, &text));
        let unindexed = text.replacen("pCreateInfos[0]", "pCreateInfos[]", 1);
        assert!(!shape(ID, ID_NAME, &unindexed), "no index");
        let before = format!("Something else.\n{text}");
        assert!(!shape(ID, ID_NAME, &before), "another preface");
        let first = text.split_once('\n').unwrap().0.to_owned();
        assert!(!shape(ID, ID_NAME, &first), "the first line alone");
        let after = text.replacen("If VkRenderingInputAttachmentIndexInfo", "If something", 1);
        assert!(!shape(ID, ID_NAME, &after), "another second line");
        let longer = format!("{text} More.");
        assert!(!shape(ID, ID_NAME, &longer), "more after the message");
        let named = text.replacen("Binding 2]", "Binding 2, variable \"x\"]", 1);
        assert!(
            !shape(ID, ID_NAME, &named),
            "another descriptor's description"
        );
        let note = format!("{}{}{}", LAST_TIME.0, LAST_TIME.1, text);
        assert!(!shape(ID, ID_NAME, &note), "a note without its count");
    }

    #[test]
    fn only_the_releases_read_pass() {
        assert!(read(vk::make_api_version(0, 1, 4, 363)));
        assert!(!read(vk::make_api_version(0, 1, 4, 357)));
        assert!(!read(vk::make_api_version(0, 1, 4, 364)));
    }

    fn rust_sources(dir: &Path, out: &mut Vec<PathBuf>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                rust_sources(&path, out);
            } else if path.extension().is_some_and(|e| e == "rs") {
                out.push(path);
            }
        }
    }

    /// The gap passes the message only because no pipeline the lanes validate is made for
    /// dynamic rendering, where the valid usage holds: neither the runtime nor this crate's
    /// executors name the structure such a pipeline is made with, or begin dynamic rendering.
    #[test]
    fn no_pipeline_here_is_made_for_dynamic_rendering() {
        let crates = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        let mut files = Vec::new();
        rust_sources(&crates.join("metal2vulkan-runtime/src"), &mut files);
        rust_sources(&crates.join("metal2vulkan-validation/src"), &mut files);
        assert!(files.len() > 100, "{} sources", files.len());
        // Spelled in pieces so this file does not match itself.
        let names = [
            concat!("PipelineRendering", "CreateInfo"),
            concat!("cmd_begin", "_rendering("),
            concat!("RenderingInfo", "::default()"),
        ];
        let found: Vec<String> = files
            .iter()
            .flat_map(|file| {
                let text = std::fs::read_to_string(file).unwrap();
                names
                    .iter()
                    .filter(|name| text.contains(**name))
                    .map(|name| format!("{} names {name}", file.display()))
                    .collect::<Vec<_>>()
            })
            .collect();
        assert_eq!(found, Vec::<String>::new());
    }
}
