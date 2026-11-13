//! A failed `vkQueueSubmit` logs one line naming what it held (`submit_failure.rs`): a blit's
//! batch whose call the device's test hook fails as `VK_ERROR_UNKNOWN`
//! (`Device::fail_next_submissions_with`, the call never made) fails its command buffer with that
//! raw result, and the device's sink hears one warning that names the result, the command buffer
//! by its label, its encoder's `VkCommandBuffer`, the queue's timeline it signals with that
//! timeline's value, and the buffer the fill names with its size, memory and age. A command buffer
//! submitted after it runs, and no line comes for it.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. Printed (`SUBMIT-FAILED`): the line.

use crate::device::lock;
use crate::submit_failure::ID_NAME;
use crate::{
    CommandBufferError, CommandBufferStatus, DebugMessage, Device, DeviceKind, DeviceOptions,
    MessageSeverity, MessageSink, ResourceOptions, VkResultCode,
};
use ash::vk;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// How long a command buffer is waited for.
const WAIT: Duration = Duration::from_secs(60);

/// The lines the device's sink heard of failed submissions, with their severities.
type Heard = Arc<Mutex<Vec<(MessageSeverity, String)>>>;

/// The device the cfg selects, with a sink that keeps every failed submission's line.
fn listening() -> (Device, Heard) {
    let heard: Heard = Arc::default();
    let kept = Arc::clone(&heard);
    let sink = MessageSink::new(move |message: &DebugMessage| {
        if message.id_name == ID_NAME {
            lock(&kept).push((message.severity, message.text.clone()));
        }
    });
    let options = DeviceOptions {
        kinds: match cfg!(m2v_cpu_icd) {
            true => vec![DeviceKind::Cpu],
            false => vec![
                DeviceKind::Discrete,
                DeviceKind::Integrated,
                DeviceKind::Virtual,
            ],
        },
        messages: Some(sink),
        ..DeviceOptions::default()
    };
    (Device::system_default_with(&options).unwrap(), heard)
}

#[test]
fn a_failed_submission_logs_one_line_naming_what_it_held() {
    let (device, heard) = listening();
    let queue = device.new_command_queue().unwrap();
    let buffer = device.new_buffer(4096, ResourceOptions::shared()).unwrap();
    let fill = |label: &str| {
        let command_buffer = queue.command_buffer().unwrap();
        command_buffer.set_label(label);
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.fill_buffer(&buffer, 0..4096, 7).unwrap();
        blit.end_encoding().unwrap();
        command_buffer
    };
    let failing = fill("submit-failure-probe");
    // The batch's `vkQueueSubmit`, the next one of a command buffer's work, fails.
    device.fail_next_submissions_with(1, vk::Result::ERROR_UNKNOWN);
    failing.commit().unwrap();
    failing.wait_until_completed_for(WAIT).unwrap();
    let (status, error) = (failing.status(), failing.error());
    let left = (device.shared().failing_submissions).load(std::sync::atomic::Ordering::Relaxed);
    assert_eq!(left, 0, "the batch's submission failed");
    assert_eq!(status, CommandBufferStatus::Error, "{error:?}");
    assert_eq!(error, Some(CommandBufferError::Internal(VkResultCode(-13))));
    // The next one runs, and logs nothing.
    let next = fill("submit-failure-next");
    next.commit().unwrap();
    next.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(
        next.status(),
        CommandBufferStatus::Completed,
        "{:?}",
        next.error()
    );
    let heard = lock(&heard).clone();
    for (_, line) in &heard {
        eprintln!("SUBMIT-FAILED {line}");
    }
    assert_eq!(
        heard.len(),
        1,
        "one line for the one failed call: {heard:?}"
    );
    let (severity, line) = &heard[0];
    assert_eq!(*severity, MessageSeverity::Warning);
    let named = format!("buffer #{} of 4096 bytes, Shared", buffer.object_id().get());
    for expected in [
        "vkQueueSubmit failed with VkResult -13 (ERROR_UNKNOWN) on VkQueue",
        "command buffer `submit-failure-probe`",
        "encoder 0 Blit of #",
        "'s timeline to ",
        named.as_str(),
        "submissions before",
        "resources: ",
    ] {
        assert!(line.contains(expected), "{expected:?} missing from {line}");
    }
    assert!(!line.contains("submit-failure-next"), "{line}");
    assert!(line.ends_with(" [metal2vulkan-submit-failed]"), "{line}");
}
