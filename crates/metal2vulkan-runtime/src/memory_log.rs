//! The memory log: a default-off diagnostic, `METAL2VULKAN_LOG_MEMORY` (`metal2vulkan::env_vars`),
//! that places the address of a GPU fault a driver reports (NVIDIA's `Xid 31`, a write to an
//! address with no page) against the device memory the runtime held.
//!
//! - **What it prints.** One line per `VkDeviceMemory` a device allocates, imports or frees,
//!   through the device's message sink at `Info` severity (id name `metal2vulkan-memory`), else on
//!   stderr: what it is (a block of the runtime's buffers or images, pooled or dedicated, an import
//!   of the caller's host memory with its host address, an import of the caller's dma-buf), its
//!   handle, its memory type, its bytes, and its device address range. A free names the handle,
//!   bytes and range again. Memory that waits for the batches in flight before it is freed
//!   (`freed_memory.rs`) is named as it starts to wait. The memory device creation makes and frees
//!   to prime the driver (`device/prime.rs`) is not logged.
//! - **Large placements.** One line per placement of at least `LOGGED_PLACEMENT_BYTES` (4 MiB)
//!   in a pooled block, as it is placed and as it is returned ([`placement_line`]): whether it is
//!   an image or a buffer, its memory class where placed, its bytes, its offset and its block's
//!   handle and memory type, which the block's own lines name. A few such placements fill a block,
//!   so their comings and goings show what makes and frees blocks. A placement that takes a kept
//!   empty dedicated block (`memory.rs`'s `Pool::place_dedicated`), which makes no block, says so
//!   too, with the block's handle and memory type.
//! - **The device address.** A block with a buffer of addresses (`memory.rs`) gives that buffer's.
//!   Any other memory's line says the address is unknown: the log makes and binds no Vulkan object
//!   to learn one, as a diagnostic changes no GPU work. A buffer of `SHADER_DEVICE_ADDRESS` usage
//!   made over the memory to ask would be, on MoltenVK, a Metal buffer of the whole block, made
//!   resident as it is asked (about 25 ms a GiB on an M3 Max), whose destroy fails a submission
//!   encoding meanwhile (`DestroyedAddressBufferFailsSubmission`). On NVIDIA's drivers a buffer's
//!   device address is the GPU virtual address the driver's fault reports name.
//! - **Off, it costs nothing.** The switch is read once; no line is formatted.

use super::driver_facts::inform_through;
use super::memory::MemoryClass;
use super::MessageSink;
use ash::vk;
use std::sync::OnceLock;

/// The id name of the log's messages.
const ID_NAME: &str = "metal2vulkan-memory";

/// Whether the log is on: read once, at the first device made.
fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(metal2vulkan::env_vars::log_memory)
}

/// A device's memory log, where the switch is on: where its lines go.
pub(super) struct MemoryLog {
    sink: Option<MessageSink>,
}

impl MemoryLog {
    /// The log of a device whose instance reports to `sink`, where the switch is on.
    pub(super) fn new(sink: Option<&MessageSink>) -> Option<MemoryLog> {
        enabled().then(|| MemoryLog {
            sink: sink.cloned(),
        })
    }

    /// Say `what` happened to `memory` (`bytes` bytes of type `type_index`, its device address
    /// `address` where known).
    pub(super) fn say(
        &self,
        what: &str,
        memory: vk::DeviceMemory,
        type_index: u32,
        bytes: u64,
        address: Option<u64>,
    ) {
        use ash::vk::Handle;
        let range = match address {
            Some(at) => format!(
                "device address 0x{at:x} to 0x{:x}",
                at.saturating_add(bytes)
            ),
            None => "device address unknown".to_owned(),
        };
        self.line(format!(
            "metal2vulkan: memory: {what}, memory 0x{:x} of type {type_index}: {bytes} bytes, \
             {range}",
            memory.as_raw()
        ));
    }

    /// Deliver `text`, a line of the log.
    pub(super) fn line(&self, text: String) {
        inform_through(self.sink.as_ref(), ID_NAME, text);
    }
}

/// The line saying `what` happened to a placement of `bytes` bytes, an image's (`image`) or a
/// buffer's, of `class` where given (as it is placed), at `offset` in `memory`, of memory type
/// `type_index` (the module docs).
pub(super) fn placement_line(
    what: &str,
    bytes: u64,
    class: Option<MemoryClass>,
    image: bool,
    (memory, offset, type_index): (vk::DeviceMemory, u64, u32),
) -> String {
    use ash::vk::Handle;
    let kind = match (class, image) {
        // Every class's name starts with a consonant.
        (Some(class), true) => format!("a {class:?} image"),
        (Some(class), false) => format!("a {class:?} buffer"),
        (None, true) => "an image".to_owned(),
        (None, false) => "a buffer".to_owned(),
    };
    format!(
        "metal2vulkan: memory: {what} {kind} of {bytes} bytes at offset 0x{offset:x} of memory \
         0x{:x} of type {type_index}",
        memory.as_raw()
    )
}
