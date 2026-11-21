//! A lost device's fault report (`VK_EXT_device_fault`): read once, at the device's first loss of
//! the runtime's own, where device creation enabled the extension. Nothing is read beside the
//! work.

use crate::command::DeviceLoss;
use crate::device::Core;
use ash::vk;
use std::fmt;

/// What the driver reported of a lost device's fault ([`DeviceLoss::fault`]), as it gave it:
/// `vkGetDeviceFaultInfoEXT`'s description, addresses and vendor records. The runtime places no
/// address: the consumer, which is told of every allocation's mapping and unmapping
/// ([`BlockMappings`](crate::BlockMappings)) and holds its own blocks and buffers, places them.
///
/// Metal's counterpart is a command buffer's `MTLCommandBufferErrorPageFault`, which names no
/// address.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceFault {
    /// The driver's description, empty where it gave none.
    pub description: String,
    /// The memory accesses it reported (each address of a kind that is no instruction pointer),
    /// in its order.
    pub accesses: Vec<FaultAddress>,
    /// The instruction pointers it reported (`INSTRUCTION_POINTER_UNKNOWN`, `_INVALID` and
    /// `_FAULT`), in its order.
    pub instructions: Vec<FaultAddress>,
    /// The vendor's records (`VkDeviceFaultVendorInfoEXT`): each one's description, code and data.
    pub vendor: Vec<(String, u64, u64)>,
}

/// One address the driver reported (`VkDeviceFaultAddressInfoEXT`).
///
/// No Metal counterpart: Metal reports no fault address.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct FaultAddress {
    /// `addressType`.
    pub kind: vk::DeviceFaultAddressTypeEXT,
    /// `reportedAddress`, a GPU virtual address.
    pub address: u64,
    /// `addressPrecision`, a power of two: the address lies in the `precision` bytes from
    /// `address` rounded down to a multiple of it.
    pub precision: u64,
}

/// The kind by its Vulkan name, then the address and the precision in hex: ash's
/// `DeviceFaultAddressTypeEXT` has no `Debug` of its own without ash's `debug` feature.
impl fmt::Debug for FaultAddress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        const KINDS: [&str; 7] = [
            "NONE",
            "READ_INVALID",
            "WRITE_INVALID",
            "EXECUTE_INVALID",
            "INSTRUCTION_POINTER_UNKNOWN",
            "INSTRUCTION_POINTER_INVALID",
            "INSTRUCTION_POINTER_FAULT",
        ];
        let raw = self.kind.as_raw();
        match usize::try_from(raw).ok().and_then(|at| KINDS.get(at)) {
            Some(name) => f.write_str(name)?,
            None => write!(f, "kind {raw}")?,
        }
        write!(f, " {:#x} precision {:#x}", self.address, self.precision)
    }
}

/// `loss`, a loss the runtime met itself, with its [`fault`](DeviceLoss::fault) set: the
/// device's report where `loss` is the device's first and its result is `ERROR_DEVICE_LOST`, None
/// otherwise. Only then is the device lost, as `vkGetDeviceFaultInfoEXT` requires
/// (`VUID-vkGetDeviceFaultInfoEXT-device-07336`), so a loss the consumer reports
/// ([`Device::report_loss`](crate::Device::report_loss)), which may name a device that is not
/// lost, never comes here.
pub(crate) fn report(core: &Core, mut loss: DeviceLoss) -> DeviceLoss {
    let first = !core.work.lock().lost();
    loss.fault = (first && loss.result == vk::Result::ERROR_DEVICE_LOST)
        .then(|| read(core))
        .flatten();
    loss
}

/// The lost device's report: its counts, then its records. None where device creation did not
/// enable the extension or a read failed.
fn read(core: &Core) -> Option<DeviceFault> {
    let get = core.fault.as_ref()?.fp().get_device_fault_info_ext;
    let device = core.device.handle();
    let mut counts = vk::DeviceFaultCountsEXT::default();
    // SAFETY: the device is lost (`report`), and `counts` lives through the call.
    unsafe { get(device, &mut counts, std::ptr::null_mut()) }
        .result()
        .ok()?;
    let mut addresses =
        vec![vk::DeviceFaultAddressInfoEXT::default(); counts.address_info_count as usize];
    let mut vendor =
        vec![vk::DeviceFaultVendorInfoEXT::default(); counts.vendor_info_count as usize];
    counts.vendor_binary_size = 0;
    let mut info = vk::DeviceFaultInfoEXT {
        p_address_infos: addresses.as_mut_ptr(),
        p_vendor_infos: vendor.as_mut_ptr(),
        ..Default::default()
    };
    // SAFETY: as above; the arrays hold the counts the first call answered, and no vendor binary
    // is asked for.
    match unsafe { get(device, &mut counts, &mut info) } {
        vk::Result::SUCCESS | vk::Result::INCOMPLETE => {}
        _ => return None,
    }
    addresses.truncate(counts.address_info_count as usize);
    vendor.truncate(counts.vendor_info_count as usize);
    let text = |chars: &[std::ffi::c_char]| {
        let bytes: Vec<u8> = (chars.iter().map(|&c| c as u8))
            .take_while(|&b| b != 0)
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    };
    let (accesses, instructions) = split(&addresses);
    Some(DeviceFault {
        description: text(&info.description),
        accesses,
        instructions,
        vendor: (vendor.iter())
            .map(|v| {
                (
                    text(&v.description),
                    v.vendor_fault_code,
                    v.vendor_fault_data,
                )
            })
            .collect(),
    })
}

/// The driver's addresses, as reported, apart: the accesses, then the instruction pointers.
pub(crate) fn split(
    addresses: &[vk::DeviceFaultAddressInfoEXT],
) -> (Vec<FaultAddress>, Vec<FaultAddress>) {
    let all = addresses.iter().map(|info| FaultAddress {
        kind: info.address_type,
        address: info.reported_address,
        precision: info.address_precision,
    });
    all.partition(|at| {
        !matches!(
            at.kind,
            vk::DeviceFaultAddressTypeEXT::INSTRUCTION_POINTER_UNKNOWN
                | vk::DeviceFaultAddressTypeEXT::INSTRUCTION_POINTER_INVALID
                | vk::DeviceFaultAddressTypeEXT::INSTRUCTION_POINTER_FAULT
        )
    })
}
