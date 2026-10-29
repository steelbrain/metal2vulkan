//! The device's functions, the only door to them: `DeviceShared::vk` hands them out through
//! [`Functions::get`], which checks the caller (`exit_guard::check`) and, inside one of the
//! runtime's sections, hands out the exit guard's table alone ([`Call`]): the same device, loaded
//! with only the table's entry points, every other one ash's stub, which panics where unwinding
//! cannot pass and so ends the process. In builds with debug assertions (every test run), a section
//! that calls an entry point outside the table therefore fails the lane that reaches it, so the
//! table holds every entry point a section calls, and priming, which makes each before the exit
//! hook is registered (`prime.rs`), covers them all. Other builds hand out every function.
//!
//! Each is a [`Vk`]: the `ash::Device`, which it dereferences to, the reads the commands of the
//! device's queue family for encoders make, which the runtime's global memory barriers name in
//! their destination (`tracker::Barrier::record`, `image_barrier::MemoryReads`), the push
//! command where the device has it (`binding::push_descriptor_command`), and
//! `vkSetDebugUtilsObjectNameEXT` where its instance reports messages (`pipeline/graphics.rs`).
//! Its copy commands count each place in the runtime that records them ([`CopySites`]).
//!
//! Each `vkCmdCopyBuffer` region over [`SPLIT_ABOVE`] bytes whose size is no multiple of
//! [`SPLIT_UNIT`] is recorded as two regions of the same command: its bytes up to the last whole
//! multiple of [`SPLIT_UNIT`], and the rest. The same bytes are copied, by the same command, so
//! every caller's stages and accesses hold as they are (`WorkCounts::copy_regions_split`). A
//! driver's recording CPU for one region may grow with its size at some sizes and not at others:
//! on ANV, Mesa 26.2.4, on the UHD 770 (`row_copies::span_copy_cost` in the validation crate's
//! runtime lane, the recording thread's CPU time a frame of one blit of one region) 8,294,411 bytes
//! (a prime) took 28-40 ms and 6.0 ms of GPU time, 8,294,576 (16 times a prime) 1.5-2.4 ms and
//! 3,110,608 (16 times a prime) 0.74-0.91 ms, against 0.006-0.035 ms for 8,294,400 (2025 times
//! 4 KiB); split so, 8,294,411 bytes took 0.018-0.060 ms and 0.64-0.76 ms of GPU time. RADV
//! (Mesa 26.0.8) recorded the split in the same CPU time; its GPU time was 5 to 9 us (0.3-1.1%)
//! more on the RX 7900 XTX at 3.1 and 8.3 MB, and on the Radeon 780M up to 9% more at some sizes
//! and 25% less at others (8,294,411 bytes: 242-243 us split, 320-328 us whole). NVIDIA's
//! 615.71.09 on the RTX 3090 likewise: within 3 us of the GPU time, and 8,294,411 bytes 1.52 ms
//! split against 6.05-6.06 ms whole.

use crate::exit_guard::{self, Call};
use crate::image_barrier::MemoryReads;
use crate::recorder::Slot;
use crate::work_counts::{CopySite, CopySites, COPY_SITES};
use ash::vk;
use std::ffi::{c_void, CStr};
use std::ops::Deref;
use std::panic::Location;
use std::sync::Arc;

/// The bytes past which a `vkCmdCopyBuffer` region whose size is no multiple of [`SPLIT_UNIT`] is
/// two (the module docs). Measured no slower up to it: on ANV, Mesa 26.2.4, on the UHD 770 one
/// region of 16,381 bytes (a prime) took 0.016-0.036 ms of the recording CPU, as 4,093 and 4,095
/// bytes did.
const SPLIT_ABOVE: u64 = 16 * 1024;

/// What the longer of a region's two parts is a whole multiple of (the module docs): 4 KiB, whose
/// multiples ANV recorded as quickly as 8,294,400 bytes (8,302,592 and 16,764,928 bytes, 4 KiB
/// times a prime, in 0.020-0.077 ms), and under which the rest's time stays short.
const SPLIT_UNIT: u64 = 4096;

/// `region` as its two parts, where it is split (the module docs).
fn split(region: &vk::BufferCopy) -> Option<[vk::BufferCopy; 2]> {
    let rest = region.size % SPLIT_UNIT;
    if region.size <= SPLIT_ABOVE || rest == 0 {
        return None;
    }
    let head = region.size - rest;
    Some([
        vk::BufferCopy {
            size: head,
            ..*region
        },
        vk::BufferCopy {
            src_offset: region.src_offset + head,
            dst_offset: region.dst_offset + head,
            size: rest,
        },
    ])
}

/// A device's functions as the runtime records commands with them: the `ash::Device`, the reads
/// a global memory barrier recorded into a command buffer of the device's queue family for
/// encoders names in place of `MEMORY_READ`, the push command and the naming of an object, and
/// the places that recorded its copy commands (the module docs).
#[derive(Clone)]
pub(crate) struct Vk {
    device: ash::Device,
    reads: MemoryReads,
    push_descriptor: Option<vk::PFN_vkCmdPushDescriptorSetKHR>,
    object_name: Option<vk::PFN_vkSetDebugUtilsObjectNameEXT>,
    copies: Arc<CopySites>,
}

impl Vk {
    /// `device`'s functions, whose global memory barriers name `reads`, with `push_descriptor`,
    /// its push command where it has one, and `object_name`, its naming of an object where its
    /// instance reports messages, whose copy commands count their places in `copies`: for the
    /// work the runtime records before the device's functions are made (`prime.rs`).
    pub(super) fn new(
        device: ash::Device,
        reads: MemoryReads,
        push_descriptor: Option<vk::PFN_vkCmdPushDescriptorSetKHR>,
        object_name: Option<vk::PFN_vkSetDebugUtilsObjectNameEXT>,
        copies: Arc<CopySites>,
    ) -> Vk {
        Vk {
            device,
            reads,
            push_descriptor,
            object_name,
            copies,
        }
    }

    /// The reads a global memory barrier's destination names.
    pub(crate) fn reads(&self) -> MemoryReads {
        self.reads
    }

    /// `vkCmdPushDescriptorSet`, where the device has it: what a dispatch and a tile dispatch
    /// that push their set 0 record it with (`binding::push_set`).
    pub(crate) fn push_descriptor(&self) -> Option<vk::PFN_vkCmdPushDescriptorSetKHR> {
        self.push_descriptor
    }

    /// `vkSetDebugUtilsObjectNameEXT`, where the device's instance reports messages: what names a
    /// graphics pipeline for the layers' messages (`pipeline/graphics.rs`).
    pub(crate) fn object_name(&self) -> Option<vk::PFN_vkSetDebugUtilsObjectNameEXT> {
        self.object_name
    }

    /// The places that recorded its copy commands so far (`WorkCounts::copy_sites`).
    pub(crate) fn copy_sites(&self) -> [CopySite; COPY_SITES] {
        self.copies.read()
    }

    /// The `vkCmdCopyBuffer` regions it split so far (`WorkCounts::copy_regions_split`).
    pub(crate) fn copy_regions_split(&self) -> u64 {
        self.copies.regions_split()
    }

    /// `vkCmdCopyBuffer`, counted at the place in the runtime that calls it ([`CopySites`]), its
    /// long regions split (the module docs).
    ///
    /// # Safety
    ///
    /// As `ash::Device::cmd_copy_buffer`'s.
    #[track_caller]
    pub(crate) unsafe fn cmd_copy_buffer(
        &self,
        command_buffer: vk::CommandBuffer,
        src: vk::Buffer,
        dst: vk::Buffer,
        regions: &[vk::BufferCopy],
    ) {
        (self.copies).count("vkCmdCopyBuffer", Location::caller(), regions.len());
        let splits = regions
            .iter()
            .filter(|region| split(region).is_some())
            .count();
        if splits > 0 {
            self.copies.split(splits);
        }
        // One region split needs no list of its own.
        let one = match (splits, regions) {
            (1, [region]) => split(region),
            _ => None,
        };
        // SAFETY: the caller's contract; a region's two parts are its bytes, in the same buffers.
        unsafe {
            match (splits, one) {
                (0, _) => (self.device).cmd_copy_buffer(command_buffer, src, dst, regions),
                (_, Some(parts)) => (self.device).cmd_copy_buffer(command_buffer, src, dst, &parts),
                (_, None) => {
                    let mut parts = Vec::with_capacity(regions.len() + splits);
                    for region in regions {
                        match split(region) {
                            Some(two) => parts.extend(two),
                            None => parts.push(*region),
                        }
                    }
                    (self.device).cmd_copy_buffer(command_buffer, src, dst, &parts);
                }
            }
        }
    }

    /// `vkCmdCopyBufferToImage`, counted at the place in the runtime that calls it
    /// ([`CopySites`]).
    ///
    /// # Safety
    ///
    /// As `ash::Device::cmd_copy_buffer_to_image`'s.
    #[track_caller]
    pub(crate) unsafe fn cmd_copy_buffer_to_image(
        &self,
        command_buffer: vk::CommandBuffer,
        src: vk::Buffer,
        dst: vk::Image,
        layout: vk::ImageLayout,
        regions: &[vk::BufferImageCopy],
    ) {
        (self.copies).count("vkCmdCopyBufferToImage", Location::caller(), regions.len());
        // SAFETY: the caller's contract.
        unsafe {
            (self.device).cmd_copy_buffer_to_image(command_buffer, src, dst, layout, regions);
        }
    }

    /// `vkCmdCopyImageToBuffer`, counted at the place in the runtime that calls it
    /// ([`CopySites`]).
    ///
    /// # Safety
    ///
    /// As `ash::Device::cmd_copy_image_to_buffer`'s.
    #[track_caller]
    pub(crate) unsafe fn cmd_copy_image_to_buffer(
        &self,
        command_buffer: vk::CommandBuffer,
        src: vk::Image,
        layout: vk::ImageLayout,
        dst: vk::Buffer,
        regions: &[vk::BufferImageCopy],
    ) {
        (self.copies).count("vkCmdCopyImageToBuffer", Location::caller(), regions.len());
        // SAFETY: the caller's contract.
        unsafe {
            (self.device).cmd_copy_image_to_buffer(command_buffer, src, layout, dst, regions);
        }
    }

    /// `vkCmdCopyImage`, counted at the place in the runtime that calls it ([`CopySites`]).
    ///
    /// # Safety
    ///
    /// As `ash::Device::cmd_copy_image`'s.
    #[track_caller]
    pub(crate) unsafe fn cmd_copy_image(
        &self,
        command_buffer: vk::CommandBuffer,
        src: vk::Image,
        src_layout: vk::ImageLayout,
        dst: vk::Image,
        dst_layout: vk::ImageLayout,
        regions: &[vk::ImageCopy],
    ) {
        (self.copies).count("vkCmdCopyImage", Location::caller(), regions.len());
        // SAFETY: the caller's contract.
        unsafe {
            (self.device).cmd_copy_image(command_buffer, src, src_layout, dst, dst_layout, regions);
        }
    }
}

impl Deref for Vk {
    type Target = ash::Device;

    fn deref(&self) -> &ash::Device {
        &self.device
    }
}

/// A device's functions: all of them, and the exit guard's table alone.
#[derive(Clone)]
pub(crate) struct Functions {
    all: Vk,
    sectioned: Vk,
    /// The address of each device-level entry point of the table, in its order (0 for an
    /// instance-level one, or one the device has not): the topmost layer's where layers run,
    /// else the driver's.
    entry_points: Vec<u64>,
}

impl Functions {
    /// The functions of `all`'s device, made from `instance`, whose global memory barriers name
    /// `reads`, with `push_descriptor`, its push command where it has one, and `object_name`, its
    /// naming of an object where its instance reports messages, whose commands go through
    /// `recording`, its slot where it has one (`recorder`).
    ///
    /// # Safety
    ///
    /// `all` is a live device of `instance`, routed (`compiles::route`, and through `recording`
    /// where it is set, `recorder::route`), and the loader serves `vkGetDeviceProcAddr` for it
    /// (its loader word is in place).
    pub(super) unsafe fn new(
        instance: &ash::Instance,
        all: ash::Device,
        reads: MemoryReads,
        push_descriptor: Option<vk::PFN_vkCmdPushDescriptorSetKHR>,
        object_name: Option<vk::PFN_vkSetDebugUtilsObjectNameEXT>,
        recording: Option<&Slot>,
    ) -> Functions {
        let handle = all.handle();
        let get = instance.fp_v1_0().get_device_proc_addr;
        // SAFETY: the caller's contract; a missing function is null, as ash loads it.
        let address = |name: &CStr| unsafe {
            std::mem::transmute::<vk::PFN_vkVoidFunction, *const c_void>(get(handle, name.as_ptr()))
        };
        // Only the device-level functions ash asks for are looked up: asking the device for an
        // instance-level one (the table's instance and messenger releases) is an error the
        // validation layer reports.
        let mut entry_points = vec![0; Call::ALL.len()];
        let load = |name: &CStr| -> *const c_void {
            match Call::ALL.iter().position(|call| call.name() == name) {
                Some(at) => {
                    let pointer = address(name);
                    entry_points[at] = pointer as usize as u64;
                    // The pipeline makings come through `compiles.rs`, and the commands through
                    // the device's slot, as `all`'s do.
                    super::compiles::entry(name)
                        .or_else(|| recording.and_then(|slot| slot.entry(name)))
                        .unwrap_or(pointer)
                }
                None => std::ptr::null(),
            }
        };
        // SAFETY: as above: each pointer is the device's function of that name, or null.
        let sectioned = unsafe { ash::Device::load_with(load, handle) };
        // The naming is no `ash::Device` function: its entry point, where the device has it.
        if let (Some(at), Some(name)) = (
            Call::ALL
                .iter()
                .position(|&call| call == Call::SetDebugUtilsObjectName),
            object_name,
        ) {
            entry_points[at] = name as usize as u64;
        }
        // Both count their copies' places together.
        let copies = Arc::new(CopySites::default());
        Functions {
            all: Vk::new(
                all,
                reads,
                push_descriptor,
                object_name,
                Arc::clone(&copies),
            ),
            sectioned: Vk::new(sectioned, reads, push_descriptor, object_name, copies),
            entry_points,
        }
    }

    /// The address of each entry point of the exit guard's table, in its order.
    pub(super) fn entry_points(&self) -> &[u64] {
        &self.entry_points
    }

    /// The device's functions for this thread now (see the module).
    pub(crate) fn get(&self) -> &Vk {
        exit_guard::check();
        if cfg!(debug_assertions) && exit_guard::in_section() {
            &self.sectioned
        } else {
            &self.all
        }
    }
}
