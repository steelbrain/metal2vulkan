//! No-copy buffers on a device that imports no host pointer: one made without
//! `VK_EXT_external_memory_host` (`DeviceOptions::withheld_extensions`), as a device whose driver
//! lists none is. Its importer takes dma-bufs of the pages alone (`host_memory.rs`'s `Importer`):
//! - where the device lists the dma-buf route (`VK_EXT_external_memory_dma_buf` and
//!   `VK_KHR_external_memory_fd`), a no-copy buffer whose memory has a source of dma-bufs of its
//!   pages (`HostMemory::with_dma_buf`), over a sealed `memfd` mapped shared as QEMU's guest RAM
//!   is, is made over imports of the source's dma-bufs (`HostBacking::ImportedAsDmaBuf`,
//!   `WorkCounts::dma_buf_imports`) with no pointer imported (`host_pointer_imports_made`); the GPU
//!   reads the CPU's bytes, its writes land in the file's pages in place, and the CPU's later
//!   writes reach its next read;
//! - on a device that lists no dma-buf route, memory with a source is refused, naming the
//!   extension.
//!
//! Memory with no source, or with one that fails, is a copy of the pages there
//! (`tests/copied_device.rs`'s
//! `copied_buffer_and_heap_are_made_on_a_device_that_imports_no_pointer`).
//! The guest RAM's side is `tests/host_ram_dma_buf.rs`'s
//! `without_pointer_imports_a_ram_s_chunks_are_its_dma_bufs`.
//!
//! A device test: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`), on Linux.
//! Where `/dev/udmabuf` cannot be opened, the test of a source says so and passes without running.

use super::host_residency::{counting, gpu_fill, host_page, read, udmabuf_openable, Ram, Scratch};
use crate::{
    Device, DeviceKind, DeviceOptions, Error, HostBacking, HostMemory, MessageSink,
    ResourceOptions, ShaderCacheOptions,
};
use metal2vulkan::device::VkExtension;
use metal2vulkan::error::{Capability, Feature, Unsupported};
use std::collections::BTreeSet;
use std::ffi::{CStr, CString};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// A device of the kind the cfg selects (a CPU device under `m2v_cpu_icd`, else a GPU) made
/// without `VK_EXT_external_memory_host`, its shader cache in `cache` and its messages given to
/// `messages`, its submission checks answered.
pub(super) fn device(cache: &Path, messages: Option<MessageSink>) -> Device {
    let options = DeviceOptions {
        kinds: match cfg!(m2v_cpu_icd) {
            true => vec![DeviceKind::Cpu],
            false => vec![
                DeviceKind::Discrete,
                DeviceKind::Integrated,
                DeviceKind::Virtual,
            ],
        },
        shader_cache: ShaderCacheOptions {
            directory: Some(cache.to_owned()),
            ..ShaderCacheOptions::default()
        },
        messages,
        withheld_extensions: [VkExtension::ExtExternalMemoryHost].into_iter().collect(),
        ..DeviceOptions::default()
    };
    let device = Device::system_default_with(&options).unwrap();
    assert!(
        !(device.enabled().extensions).contains(&VkExtension::ExtExternalMemoryHost),
        "{}: VK_EXT_external_memory_host enabled though withheld",
        device.name()
    );
    assert!(
        device.settle_submit_checks(crate::tests::SETTLE),
        "the submission checks never answered"
    );
    device
}

/// The refusal of host memory on a device that imports no host pointer and no dma-buf, and of a
/// guest RAM with no source of dma-bufs on one that imports no host pointer: `Unsupported`, naming
/// `VK_EXT_external_memory_host`.
pub(super) fn refusal() -> Error {
    Error::Unsupported(Unsupported::needs(
        Feature::HostMemory,
        Capability::Extension(VkExtension::ExtExternalMemoryHost),
    ))
}

/// Whether the physical device under `device` lists the dma-buf route's extensions
/// (`create.rs`'s `dma_buf_names`), which the runtime then enables: asked of the driver, not of
/// the device's importer.
pub(super) fn lists_dma_buf_route(device: &Device) -> bool {
    // SAFETY: the handles are read while the device lives.
    let handles = unsafe { device.raw() };
    // SAFETY: the physical device is the instance's.
    let listed = unsafe {
        handles
            .instance
            .enumerate_device_extension_properties(handles.physical_device)
    }
    .unwrap();
    let names: BTreeSet<CString> = (listed.iter())
        .filter_map(|extension| extension.extension_name_as_c_str().ok().map(CStr::to_owned))
        .collect();
    crate::device::create::dma_buf_names(&names).is_some()
}

/// A no-copy buffer over 3 pages of a `memfd` mapped shared, 256 bytes into its second page, whose
/// memory has a source of dma-bufs of them: on a device that lists the dma-buf route, made over
/// imports of the source's dma-bufs, no pointer imported, its source called once; the GPU reads
/// the file's bytes through it, fills it in the file's pages (the bytes around it as they were),
/// and reads the CPU's next writes. On a device that lists no such route, refused as one that
/// imports no host memory.
#[test]
fn without_pointer_imports_a_sourced_no_copy_buffer_lies_in_its_dma_bufs() {
    const TEST: &str = "without_pointer_imports_a_sourced_no_copy_buffer_lies_in_its_dma_bufs";
    if !udmabuf_openable(TEST) {
        return;
    }
    let scratch = Scratch::new();
    let device = device(&scratch.0, None);
    let page = host_page();
    let ram = Ram::new(8 * page, 0, 8 * page);
    let bytes = counting(8 * page, 0);
    ram.write_file(0, &bytes);
    let (offset, length) = (page + 256, 3 * page);
    let calls = Arc::new(AtomicUsize::new(0));
    let before = device.work_counts();
    // SAFETY: the view outlives the buffer, and the source makes dma-bufs of its pages.
    let made = unsafe {
        device.new_buffer_with_bytes_no_copy(
            HostMemory::with_dma_buf(ram.pointer(offset), ram.source(&calls)),
            length as u64,
            ResourceOptions::default(),
            None,
        )
    };
    let after = device.work_counts();
    let route = lists_dma_buf_route(&device);
    println!(
        "{TEST}: on {} (the dma-buf route listed: {route}): {}; {} dma-buf import(s), {} pointer \
         import(s), the source called {} time(s)",
        device.name(),
        match &made {
            Ok(buffer) => format!("made, {:?}", buffer.host_backing()),
            Err(error) => format!("refused: {error}"),
        },
        after.dma_buf_imports - before.dma_buf_imports,
        after.host_pointer_imports_made - before.host_pointer_imports_made,
        calls.load(Ordering::Acquire),
    );
    let buffer = match (route, made) {
        (true, made) => made.unwrap_or_else(|error| panic!("{TEST}: refused: {error}")),
        (false, made) => {
            assert_eq!(made.err(), Some(refusal()), "{TEST}");
            assert_eq!(
                calls.load(Ordering::Acquire),
                0,
                "{TEST}: the source called"
            );
            return;
        }
    };
    assert_eq!(
        buffer.host_backing(),
        Some(HostBacking::ImportedAsDmaBuf),
        "{TEST}"
    );
    assert!(
        after.dma_buf_imports > before.dma_buf_imports,
        "{TEST}: no dma-buf imported"
    );
    assert_eq!(
        after.host_pointer_imports_made, before.host_pointer_imports_made,
        "{TEST}: a pointer imported"
    );
    assert_eq!(
        calls.load(Ordering::Acquire),
        1,
        "{TEST}: the source's calls"
    );
    assert_eq!(buffer.contents(), Some(ram.pointer(offset)), "{TEST}");
    let queue = device.new_command_queue().unwrap();
    assert!(
        read(&queue, &buffer, length) == bytes[offset..offset + length],
        "{TEST}: the GPU's read of the CPU's bytes"
    );
    gpu_fill(&queue, &buffer, 0..length as u64, 0x5a);
    assert!(
        ram.read_file(offset, length).iter().all(|&b| b == 0x5a),
        "{TEST}: the GPU's fill in the file's pages"
    );
    assert!(
        ram.read_file(0, offset) == bytes[..offset]
            && ram.read_file(offset + length, page) == bytes[offset + length..][..page],
        "{TEST}: the bytes around the buffer"
    );
    let later = counting(length, 7);
    ram.write_file(offset, &later);
    assert!(
        read(&queue, &buffer, length) == later,
        "{TEST}: the GPU's read of the CPU's later bytes"
    );
    assert_eq!(
        calls.load(Ordering::Acquire),
        1,
        "{TEST}: the source called again"
    );
    drop(buffer);
}
