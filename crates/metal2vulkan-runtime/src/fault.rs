//! A lost device's fault report (`VK_EXT_device_fault`): read once, at the device's first loss of
//! the runtime's own, where device creation enabled the extension, each of its addresses placed
//! among the runtime's own ranges a shader reaches by device address ([`Spans`]): the live ones
//! and the last [`FREED`] freed, noted as each is made and freed. Nothing is read beside the work.

use crate::command::DeviceLoss;
use crate::device::{lock, Core};
use crate::memory::{AllocationId, BlockId, MemoryClass};
use ash::vk;
use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::sync::Mutex;

/// The freed ranges a device keeps for its fault report ([`Spans`]), the oldest dropped first.
pub(crate) const FREED: usize = 1024;

/// What the driver reported of a lost device's fault ([`DeviceLoss::fault`]), as it gave it:
/// `vkGetDeviceFaultInfoEXT`'s description, addresses and vendor records, with each address placed
/// among the runtime's own ranges ([`places`](Self::places)). The consumer, which is told of every
/// allocation's mapping and unmapping ([`BlockMappings`](crate::BlockMappings)) and holds its own
/// blocks and buffers, places them among those.
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
    /// Each address of [`accesses`](Self::accesses), then of
    /// [`instructions`](Self::instructions), in their order, placed among the runtime's own ranges
    /// as the report was read. Empty in a fault the consumer reports.
    pub places: Vec<FaultPlace>,
}

/// Where one of the driver's addresses lies among the runtime's own ranges a shader reaches by
/// device address: the pool's allocations, the parameter ring's blocks, imports, exports,
/// surfaces' buffers, imports over several dma-bufs and indirect command buffers' preprocess
/// regions, live, or among the last 1,024 freed. Its `Display` is one line. Images, which Vulkan
/// gives no device address, and the zero source and staging memory, which only copies reach, are
/// not among them.
///
/// No Metal counterpart: Metal reports no fault address.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FaultPlace {
    /// The address, as reported.
    pub address: u64,
    /// Its precision, at least 1: the bytes placed are the address rounded down to a multiple of
    /// it, this many.
    pub precision: u64,
    /// The host time of the read ([`host_time`](crate::host_time)).
    pub at: u64,
    /// The ranges that meet those bytes: the live ones by address, then the freed ones, newest
    /// first.
    pub holders: Vec<FaultRange>,
    /// The nearest range, live or freed, wholly below those bytes.
    pub below: Option<FaultRange>,
    /// The nearest range, live or freed, wholly above them.
    pub above: Option<FaultRange>,
}

/// One of the runtime's own ranges with a device address ([`FaultPlace`]).
///
/// No Metal counterpart.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FaultRange {
    /// What it is.
    pub kind: RangeKind,
    /// The device address of its first byte.
    pub address: u64,
    /// Its bytes.
    pub len: u64,
    /// The host time it was made ([`host_time`](crate::host_time)).
    pub made: u64,
    /// Its free, where it was freed.
    pub freed: Option<RangeFreed>,
}

/// A [`FaultRange`]'s free.
///
/// No Metal counterpart.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RangeFreed {
    /// The host time of the free ([`host_time`](crate::host_time)).
    pub at: u64,
    /// The command buffers committed and not retired at the free.
    pub in_flight: u64,
    /// For an allocation of the pool, the progress value of the submission its free waited for,
    /// the last of those committed when it went idle; None where none was in flight then, and for
    /// every other kind.
    pub waited_for: Option<u64>,
}

/// What a [`FaultRange`] is.
///
/// No Metal counterpart.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RangeKind {
    /// An allocation of the pool, of its class ([`AllocationInfo`](crate::AllocationInfo)).
    Allocation(AllocationId, MemoryClass),
    /// An imported block's allocation of its own.
    Import,
    /// An exported block's allocation of its own, by the block's number
    /// ([`BlockInfo::id`](crate::BlockInfo::id)).
    Export(BlockId),
    /// An exported surface's buffer.
    Surface,
    /// An import over several dma-bufs: its sparse buffer, on Unix.
    Sparse,
    /// A block of the parameter ring.
    RingBlock,
    /// A region of an indirect command buffer's execution's preprocess memory.
    Preprocess,
}

impl fmt::Display for RangeKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RangeKind::Allocation(id, class) => write!(f, "{class:?} pool allocation {}", id.get()),
            RangeKind::Import => f.write_str("import"),
            RangeKind::Export(block) => write!(f, "export block {}", block.get()),
            RangeKind::Surface => f.write_str("surface buffer"),
            RangeKind::Sparse => f.write_str("import over dma-bufs"),
            RangeKind::RingBlock => f.write_str("parameter ring block"),
            RangeKind::Preprocess => f.write_str("preprocess region"),
        }
    }
}

impl FaultPlace {
    /// The bytes placed: `low..high`.
    fn bytes(&self) -> (u64, u64) {
        let low = self.address - self.address % self.precision;
        (low, low.saturating_add(self.precision))
    }

    /// `span`, its ages counted back from the read.
    fn name(&self, f: &mut fmt::Formatter<'_>, span: &FaultRange) -> fmt::Result {
        let ago = |at: u64| self.at.saturating_sub(at) as f64 / 1e9;
        write!(
            f,
            "{} [{:#x}, +{:#x}) made {:.3} s before",
            span.kind,
            span.address,
            span.len,
            ago(span.made)
        )?;
        let Some(freed) = span.freed else {
            return f.write_str(", live");
        };
        write!(
            f,
            ", freed {:.3} s before with {} command buffers in flight",
            ago(freed.at),
            freed.in_flight
        )?;
        match freed.waited_for {
            Some(progress) => write!(f, ", after progress {progress}"),
            None => Ok(()),
        }
    }
}

impl fmt::Display for FaultPlace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (low, high) = self.bytes();
        write!(f, "{:#x}", self.address)?;
        if self.holders.is_empty() {
            f.write_str(": in none of the runtime's ranges")?;
        }
        for (n, span) in self.holders.iter().enumerate() {
            f.write_str(if n == 0 { ": in " } else { "; in " })?;
            self.name(f, span)?;
        }
        if let Some(span) = &self.below {
            f.write_str("; nearest below: ")?;
            self.name(f, span)?;
            let end = span.address + span.len;
            write!(f, ", its end {:#x} bytes below", low - end)?;
        }
        if let Some(span) = &self.above {
            f.write_str("; nearest above: ")?;
            self.name(f, span)?;
            write!(f, ", {:#x} bytes above", span.address - high)?;
        }
        Ok(())
    }
}

/// The device's own ranges a shader reaches by device address, by the `VkBuffer` that spans each,
/// and the last [`FREED`] freed, oldest first: noted as each is made and freed, and read only at a
/// loss ([`report`]).
#[derive(Default)]
pub(crate) struct Spans(Mutex<Book>);

#[derive(Default)]
struct Book {
    live: HashMap<vk::Buffer, FaultRange>,
    freed: VecDeque<FaultRange>,
}

impl Spans {
    /// `buffer`'s range, `len` bytes at `address`, was made. A block of no buffer (an opaque
    /// handle's surface allocation, which holds only the surface's plane images) has no device
    /// address, and is not noted.
    pub(crate) fn made(&self, buffer: vk::Buffer, address: u64, len: u64, kind: RangeKind) {
        if buffer == vk::Buffer::null() {
            return;
        }
        let made = crate::host_time();
        let span = FaultRange {
            kind,
            address,
            len,
            made,
            freed: None,
        };
        lock(&self.0).live.insert(buffer, span);
    }

    /// `buffer`'s range is freed now, with `in_flight` command buffers committed and not retired,
    /// after the submission of progress `waited_for`. A buffer not noted is skipped.
    pub(crate) fn freed(&self, buffer: vk::Buffer, in_flight: u64, waited_for: Option<u64>) {
        let at = crate::host_time();
        let mut book = lock(&self.0);
        let Some(mut span) = book.live.remove(&buffer) else {
            return;
        };
        span.freed = Some(RangeFreed {
            at,
            in_flight,
            waited_for,
        });
        if book.freed.len() == FREED {
            book.freed.pop_front();
        }
        book.freed.push_back(span);
    }

    /// `address` placed, at the host time `at`.
    pub(crate) fn place(&self, address: &FaultAddress, at: u64) -> FaultPlace {
        let mut place = FaultPlace {
            address: address.address,
            precision: address.precision.max(1),
            at,
            holders: Vec::new(),
            below: None,
            above: None,
        };
        let (low, high) = place.bytes();
        let book = lock(&self.0);
        let all = || book.live.values().chain(book.freed.iter().rev());
        let meets = |span: &&FaultRange| span.address < high && low < span.address + span.len;
        let mut live: Vec<FaultRange> = book.live.values().filter(meets).copied().collect();
        live.sort_unstable_by_key(|span| span.address);
        place.holders = (live.into_iter())
            .chain(book.freed.iter().rev().filter(meets).copied())
            .collect();
        place.below = (all().filter(|span| span.address + span.len <= low))
            .min_by_key(|span| low - (span.address + span.len))
            .copied();
        place.above = (all().filter(|span| span.address >= high))
            .min_by_key(|span| span.address - high)
            .copied();
        place
    }

    /// For the tests: the freed ranges kept, oldest first.
    #[cfg(test)]
    pub(crate) fn freed_kept(&self) -> Vec<FaultRange> {
        lock(&self.0).freed.iter().copied().collect()
    }
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
        .flatten()
        .map(Box::new);
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
    let at = crate::host_time();
    let places = (accesses.iter().chain(&instructions))
        .map(|address| core.spans.place(address, at))
        .collect();
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
        places,
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
