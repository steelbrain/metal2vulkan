//! A copy of rows of bytes from one buffer into another: one `vkCmdCopyBuffer` of a region per
//! row, or, where the rows are equal and the device records the regions slower, one dispatch of
//! `bind::rows_copy_module` that writes the same bytes ([`RowCopies`]). These copies go through
//! here: a blit between a buffer and a texture over a buffer, or between two textures over buffers
//! (`blit_texture.rs`, `RowsCopy`, through a scratch buffer too where its rows share bytes); a
//! buffer-texture copy's rows through its staging buffer; and a shadow's copy back out of its
//! staging buffer (`shadow.rs`, `ShadowCopy::staged_back`): a texture over a buffer's bytes, a
//! texture buffer, a rotated texture (`rotated.rs`), a heap's two-plane or tiled texture
//! (`texture/heap_two_planes.rs`, `texture/heap_tiled.rs`), as an encoder records it or as a
//! render pass's steps record it between its instances (`render/pass_steps.rs`).
//!
//! - **Why.** A driver may record each region of a copy as an operation of its own: ANV records
//!   one blit operation per region, a full 3D-pipeline draw with its own state. On the Iris Xe
//!   the 1080 rows of a 1920-texel `BGRA8Unorm` texture took 3.1-3.3 ms of CPU, and one
//!   `vkCmdCopyBuffer` a frame took about 15 of the recording thread's 19 ms a frame of a
//!   program's frames. One dispatch records the same rows as one command.
//! - **When.** On a device measured to run the dispatch faster
//!   (`DriverFact::RowCopiesDispatchFaster`: ANV, and NVIDIA's on the RTX 5080 Laptop GPU, from 2
//!   rows on, on the CPU and the GPU alike; RADV runs it no faster), where the copy has at least
//!   [`DISPATCH_ROWS`] regions, each the same number of bytes, a whole number of 4-byte words,
//!   their offsets in both buffers 4-byte aligned and each row the same distance after the one
//!   before it in each buffer (the destination's rows apart by a row's length at least), within the
//!   device's storage buffer range and workgroup counts. Any other copy, any other device, and a
//!   device that does not make the kernel or its set, copies as before.
//! - **Exact.** The dispatch stores each row's words from the source row's place at the
//!   destination row's (invocation `(x, y)`: word `x` of row `y`), and no byte between the rows
//!   or past them: the bytes `vkCmdCopyBuffer` writes. The callers' copies keep `vkCmdCopyBuffer`'s
//!   rules either way: the destination rows are disjoint, and in one buffer no source row meets a
//!   destination row, so the dispatch's invocations write disjoint words that none reads.
//! - **Synchronization.** The dispatch reads the source as `SHADER_STORAGE_READ` and writes the
//!   destination as `SHADER_STORAGE_WRITE`, in `COMPUTE_SHADER`, where the copy reads and writes
//!   them as `TRANSFER_READ` and `TRANSFER_WRITE` in `TRANSFER`: the caller names its uses by
//!   [`RowCopies::read`] and [`RowCopies::write`] in [`RowCopies::stages`], so the encoder's
//!   tracker orders it after the buffers' earlier accesses and before their next ones, as any
//!   command.
//! - **Bound.** On a device with `VK_KHR_push_descriptor` the dispatch pushes its two buffers'
//!   descriptors (the kernel's set 0 is a push descriptor set's), so a copy makes no object on
//!   the encoding thread. Elsewhere its set comes from a pool of its own, which lives until the
//!   command buffer completes: the caller retains [`RowCopies::retained`] with its command (a
//!   pass's steps hold it, `render/pass_steps.rs`'s `StagedRows`).
//! - **Counted.** `WorkCounts::row_copies`, `row_copy_regions`, `row_copy_regions_most` and
//!   `row_copies_dispatched`.

use super::owned_kernel::{OwnedBinds, OwnedKernel, OwnedModule, OwnedSet, OwnedSets};
use super::tracker::Access;
use super::Device;
use crate::device::functions::Vk;
use ash::vk;
use metal2vulkan::bind::RowsCopyModule;
use metal2vulkan::device::DriverFact;

/// The fewest regions a copy of rows takes as a dispatch: on ANV and on NVIDIA's RTX 5080 Laptop
/// GPU even 2 rows ran no slower on the CPU and faster on the GPU as one
/// (`DriverFact::RowCopiesDispatchFaster`).
pub(super) const DISPATCH_ROWS: usize = 2;

/// A copy of rows between two buffers, as one of its two forms (the module docs).
pub(super) struct RowCopies {
    form: Form,
}

enum Form {
    /// `vkCmdCopyBuffer` of the regions.
    Copy {
        src: vk::Buffer,
        dst: vk::Buffer,
        regions: Vec<vk::BufferCopy>,
    },
    /// One dispatch of the rows-copy kernel.
    Dispatch(RowsDispatch),
}

/// The dispatch that copies the rows: the kernel, its two buffers' bytes from their first rows'
/// words (as aligned as the device binds them), the rows' shape and the workgroups.
struct RowsDispatch {
    kernel: OwnedKernel,
    /// Its two buffers' bytes: pushed where the device pushes, else a set of a pool of its own.
    binds: OwnedBinds,
    shape: [u8; 16],
    groups: [u32; 2],
}

impl RowCopies {
    /// The copy on `device` of `regions` from `src` into `dst`: a dispatch where the rows take
    /// one (the module docs), else the regions.
    pub(super) fn new(
        device: &Device,
        src: vk::Buffer,
        dst: vk::Buffer,
        regions: Vec<vk::BufferCopy>,
    ) -> RowCopies {
        let dispatch = RowsDispatch::new(device, src, dst, &regions);
        device
            .shared()
            .counts
            .row_copy(regions.len(), dispatch.is_some());
        let form = match dispatch {
            Some(dispatch) => Form::Dispatch(dispatch),
            None => Form::Copy { src, dst, regions },
        };
        RowCopies { form }
    }

    /// Whether the copy is a dispatch.
    pub(super) fn dispatched(&self) -> bool {
        matches!(self.form, Form::Dispatch(_))
    }

    /// The stages the copy runs in.
    pub(super) fn stages(&self) -> vk::PipelineStageFlags2 {
        match self.form {
            Form::Copy { .. } => vk::PipelineStageFlags2::TRANSFER,
            Form::Dispatch(_) => vk::PipelineStageFlags2::COMPUTE_SHADER,
        }
    }

    /// The copy's read of the source.
    pub(super) fn read(&self) -> Access {
        match self.form {
            Form::Copy { .. } => Access::new(
                vk::PipelineStageFlags2::TRANSFER,
                vk::AccessFlags2::TRANSFER_READ,
            ),
            Form::Dispatch(_) => Access::new(
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_STORAGE_READ,
            ),
        }
    }

    /// The copy's write of the destination.
    pub(super) fn write(&self) -> Access {
        match self.form {
            Form::Copy { .. } => Access::new(
                vk::PipelineStageFlags2::TRANSFER,
                vk::AccessFlags2::TRANSFER_WRITE,
            ),
            Form::Dispatch(_) => Access::new(
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
            ),
        }
    }

    /// Record the copy into `command_buffer`.
    ///
    /// # Safety
    ///
    /// `command_buffer` is recording, outside a render pass; both buffers are live until it
    /// completes, the regions lie inside them, the destination's are disjoint, and in one buffer
    /// no source region meets a destination region.
    pub(super) unsafe fn record(&self, device: &Vk, command_buffer: vk::CommandBuffer) {
        match &self.form {
            // SAFETY: the caller's contract.
            Form::Copy { src, dst, regions } => unsafe {
                device.cmd_copy_buffer(command_buffer, *src, *dst, regions);
            },
            // SAFETY: the caller's contract; the descriptors bind both buffers' bytes the rows
            // lie in, and the shape and workgroups stay inside them (`RowsDispatch::new`).
            Form::Dispatch(dispatch) => unsafe { dispatch.record(device, command_buffer) },
        }
    }

    /// What the command buffer holds until it completes: the dispatch's set, where the copy is a
    /// dispatch that binds one.
    pub(super) fn retained(self) -> Option<OwnedSets> {
        match self.form {
            Form::Dispatch(RowsDispatch {
                binds: OwnedBinds::Sets(sets),
                ..
            }) => Some(sets),
            Form::Copy { .. } | Form::Dispatch(_) => None,
        }
    }
}

/// A storage binding of `rows` rows `size` bytes long, the first at byte `first` of a buffer and
/// each `apart` bytes after the one before: its offset (`first` aligned down to `align`), its
/// range, and the first row's word in it; `None` past the device's range or the shape's first
/// word.
fn binding(first: u64, apart: u64, size: u64, rows: u64, align: u64, most: u64) -> Option<Binding> {
    let offset = first - first % align;
    let end = first
        .checked_add(apart.checked_mul(rows - 1)?)?
        .checked_add(size)?;
    let range = end - offset;
    let first_word = (first - offset) / 4;
    (range <= most && first_word <= u64::from(RowsCopyModule::MOST_FIRST_WORD)).then_some(Binding {
        offset,
        range,
        first_word: first_word as u32,
    })
}

struct Binding {
    offset: u64,
    range: u64,
    first_word: u32,
}

impl RowsDispatch {
    /// The dispatch on `device` that copies `regions` from `src` into `dst`; `None` where the copy
    /// takes them (the module docs).
    fn new(
        device: &Device,
        src: vk::Buffer,
        dst: vk::Buffer,
        regions: &[vk::BufferCopy],
    ) -> Option<RowsDispatch> {
        let (first, rest) = regions.split_first()?;
        let faster = device
            .properties()
            .driver_facts
            .contains(&DriverFact::RowCopiesDispatchFaster);
        if !faster
            || regions.len() < DISPATCH_ROWS
            || src == vk::Buffer::null()
            || dst == vk::Buffer::null()
        {
            return None;
        }
        let size = first.size;
        let second = rest.first()?;
        let src_apart = second.src_offset.checked_sub(first.src_offset)?;
        let dst_apart = second.dst_offset.checked_sub(first.dst_offset)?;
        let aligned = |v: u64| v.is_multiple_of(4);
        let equal = regions.iter().enumerate().all(|(i, region)| {
            let i = i as u64;
            region.size == size
                && Some(region.src_offset) == src_apart.checked_mul(i).map(|a| a + first.src_offset)
                && Some(region.dst_offset) == dst_apart.checked_mul(i).map(|a| a + first.dst_offset)
        });
        let words = size / 4;
        if !equal
            || size == 0
            || !aligned(size)
            || !aligned(first.src_offset)
            || !aligned(first.dst_offset)
            || !aligned(src_apart)
            || !aligned(dst_apart)
            || dst_apart < size
        {
            return None;
        }
        let limits = &device.properties().description.limits;
        let align = limits.min_storage_buffer_offset_alignment.max(4);
        let most = u64::from(limits.max_storage_buffer_range);
        let rows = regions.len() as u64;
        let source = binding(first.src_offset, src_apart, size, rows, align, most)?;
        let destination = binding(first.dst_offset, dst_apart, size, rows, align, most)?;
        let width = u32::try_from(words).ok()?;
        let rows = u32::try_from(rows).ok()?;
        let groups = RowsCopyModule::groups(width, rows);
        if groups[0] > limits.max_compute_work_group_count[0]
            || groups[1] > limits.max_compute_work_group_count[1]
        {
            return None;
        }
        // Each below `maxStorageBufferRange`, a `u32`, as both bindings' ranges are.
        let row_words = [(src_apart / 4) as u32, (dst_apart / 4) as u32];
        let kernel = device.owned_kernel(OwnedModule::RowsCopy).ok()?;
        let buffers = [
            (src, source.offset, source.range),
            (dst, destination.offset, destination.range),
        ];
        // The kernel's set 0 is a push descriptor set's where the device pushes (`OwnedKernel`).
        let set = OwnedSet {
            images: Vec::new(),
            storage_images: Vec::new(),
            buffers: buffers.to_vec(),
        };
        let binds = OwnedBinds::new(device, &kernel, &[set]).ok()?;
        Some(RowsDispatch {
            kernel,
            binds,
            shape: RowsCopyModule::shape(
                [source.first_word, destination.first_word],
                width,
                row_words,
            ),
            groups,
        })
    }

    /// Record the dispatch into `command_buffer`.
    ///
    /// # Safety
    ///
    /// As [`RowCopies::record`].
    unsafe fn record(&self, device: &Vk, command_buffer: vk::CommandBuffer) {
        let point = vk::PipelineBindPoint::COMPUTE;
        // SAFETY: the caller's contract; the descriptors bind both buffers' bytes the rows lie
        // in, and the shape and workgroups stay inside them (checked in `new`).
        unsafe {
            device.cmd_bind_pipeline(command_buffer, point, self.kernel.pipeline);
            self.binds.bind(device, command_buffer, &self.kernel, 0);
            device.cmd_push_constants(
                command_buffer,
                self.kernel.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &self.shape,
            );
            device.cmd_dispatch(command_buffer, self.groups[0], self.groups[1], 1);
        }
    }
}
