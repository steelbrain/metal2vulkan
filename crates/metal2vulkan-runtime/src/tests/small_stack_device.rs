//! A device made on a thread with a small stack: the making builds no large value on the calling
//! thread's stack (a device's command count shards lie on the heap, each made there,
//! `shards.rs`). The device made on a thread of [`STACK`] bytes fills a buffer, read back, and goes
//! on that thread.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use crate::device::DeviceShared;
use crate::work_counts::Counters;
use crate::{Device, DeviceKind, DeviceOptions, MessageSink, ResourceOptions};
use std::time::Duration;

/// The stack of the thread that makes the device: half of the 512 KiB of a macOS program's
/// secondary thread, on which Metal makes a device.
const STACK: usize = 256 << 10;

/// The longest the command buffer may take.
const WAIT: Duration = Duration::from_secs(60);

/// The bytes the buffer holds.
const LENGTH: u64 = 4096;

/// Options of the kind the cfg selects, with a message sink of their own: a group of devices no
/// other test makes (a sink is compared by its identity), so the device is made on the thread.
fn options() -> DeviceOptions {
    DeviceOptions {
        kinds: match cfg!(m2v_cpu_icd) {
            true => vec![DeviceKind::Cpu],
            false => vec![
                DeviceKind::Discrete,
                DeviceKind::Integrated,
                DeviceKind::Virtual,
            ],
        },
        messages: Some(MessageSink::new(|_| {})),
        ..DeviceOptions::default()
    }
}

#[test]
fn a_device_is_made_used_and_released_on_a_thread_with_a_small_stack() {
    println!(
        "SMALL-STACK sizes: DeviceShared {} bytes, Counters {} bytes",
        size_of::<DeviceShared>(),
        size_of::<Counters>()
    );
    let made = std::thread::Builder::new()
        .name("small stack".into())
        .stack_size(STACK)
        .spawn(|| {
            let device = Device::system_default_with(&options()).unwrap();
            let buffer = device
                .new_buffer(LENGTH, ResourceOptions::shared())
                .unwrap();
            let queue = device.new_command_queue().unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            blit.fill_buffer(&buffer, 0..LENGTH, 0x5a).unwrap();
            blit.end_encoding().unwrap();
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(command_buffer.error(), None);
            let contents = buffer.contents().unwrap();
            // SAFETY: the buffer's `LENGTH` bytes are host-visible, and its command buffer
            // completed.
            let bytes = unsafe { std::slice::from_raw_parts(contents.as_ptr(), LENGTH as usize) };
            assert!(
                bytes.iter().all(|&byte| byte == 0x5a),
                "bytes the fill missed"
            );
            device.name().to_owned()
        })
        .unwrap()
        .join()
        .unwrap();
    println!("SMALL-STACK: {made} made, filled a buffer and went on a {STACK}-byte stack");
}
