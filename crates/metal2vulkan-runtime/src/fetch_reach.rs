//! A pulled fetch past 4 GiB of its buffer: the interim refusal of a silent wrong answer, a vertex
//! attribute's (`render/vertex_reach.rs`) and a kernel `[[stage_in]]` attribute's (here).
//!
//! - **What differs.** Metal fetches an attribute at the displacement `element · stride +
//!   offset` from its buffer's bind offset in 64 bits. The compile layer fetches an attribute it
//!   pulls (a vertex attribute or a kernel stage-in one) whose bytes lie past `2^32` as zero bytes
//!   (`lower::stage_io::fetch`). So where the buffer holds more than `2^32` bytes past its bind
//!   offset, a byte past `2^32` reads zero where Metal reads the buffer, silently.
//! - **The refusal.** Until the fetch takes 64-bit displacements, a fetch that may read past `2^32`
//!   bytes of its buffer's bind offset, inside a buffer that runs past them, is refused, naming the
//!   buffer index and the furthest byte it may read; one that may not runs as before, and so does
//!   one from a buffer of at most `2^32` bytes past its bind offset, which Metal reads outside the
//!   buffer. The reach is the furthest element times the stride, plus the attribute's offset and
//!   its format's size: exact where the call states the elements it fetches, and the largest the
//!   call allows where GPU memory states them.
//! - **A kernel's stage-in attribute** (`check_stage_in_reach`). A dispatch (the encoder's,
//!   an indirect one, an indirect command buffer's command, recorded at the commit or by the
//!   host step) whose kernel may fetch such an attribute is refused
//!   `NotImplemented(StageInFetchPast4GiB)`. The furthest element of a layout stepping by thread
//!   position at rate `n` on an axis is the region's origin plus the grid's last position on the
//!   axis divided by `n`, clamped to the region's last element (its size less 1; a size of 0
//!   clamps nothing), as the fetch computes it (`lower::stage_io::stage_in`); an indexed step's
//!   is the region's origin plus its index type's largest index, clamped so; a constant step
//!   fetches its offset alone. A dispatch whose grid GPU memory states (an indirect dispatch on
//!   the GPU path) may run any 32-bit position. An attribute is fetched where the kernel reads
//!   its layout's buffer slot. An execution of commands a kernel writes, whose records' buffers
//!   and grids are GPU memory, runs through the host step, which checks each command, wherever a
//!   state its records may name fetches a stage-in attribute (`fetches_stage_in`), never on the
//!   GPU alone (`gpu_execute.rs`).

use crate::argument_table::ArgumentTable;
use crate::error::{Error, RuntimeGap};
use crate::pipeline::ComputePipelineState;
use metal2vulkan::bind::{Dispatch, StageInRegion};
use metal2vulkan::metal::vertex::vk_vertex_format;
use metal2vulkan::metal::{IndexType, StepFunction, VertexFormat};
use metal2vulkan::pipeline::{
    PipelineInterface, SlotClass, Stage, StageInputDescriptor, StageInputLayoutDescriptor,
};
use metal2vulkan::vk;

/// The bytes past a bind offset the fetch's 32-bit displacement reaches.
pub(crate) const REACHED: u64 = 1 << 32;

/// The byte past the last that an attribute of `size` bytes at `offset` reads at `element` of
/// `stride`, from the bind offset, saturating: `element · stride + offset + size`.
pub(crate) fn reach(element: u64, stride: u32, offset: u32, size: u32) -> u64 {
    element
        .saturating_mul(u64::from(stride))
        .saturating_add(u64::from(offset))
        .saturating_add(u64::from(size))
}

/// The store size of `format`: its Vulkan format's block, or 16, the largest, for a format with
/// none.
pub(crate) fn format_bytes(format: VertexFormat) -> u32 {
    vk_vertex_format(format)
        .ok()
        .and_then(vk::Format::block)
        .map_or(16, |block| block.size)
}

/// The grid of threads `dispatch` runs, by axis, from its own counts; `None` where GPU memory
/// states them.
pub(crate) fn grid(dispatch: &Dispatch) -> Option<[u64; 3]> {
    let product =
        |a: [u32; 3], b: [u32; 3]| std::array::from_fn(|i| u64::from(a[i]) * u64::from(b[i]));
    match *dispatch {
        Dispatch::Threadgroups {
            groups,
            threads_per_group,
        }
        | Dispatch::ThreadgroupRange {
            groups,
            threads_per_group,
            ..
        } => Some(product(groups, threads_per_group)),
        Dispatch::Threads { threads, .. } => Some(threads.map(u64::from)),
        _ => None,
    }
}

/// Refuse a dispatch of `pipeline` (`NotImplemented(StageInFetchPast4GiB)`, the module docs)
/// over `grid` (`None`: stated by GPU memory) and the stage-in `region`, with `table`'s
/// buffers, where it may fetch a stage-in attribute past `2^32` bytes of the bind offset of a
/// buffer that runs past them; `interface` is the slots its variant reads.
pub(crate) fn check_stage_in_reach(
    pipeline: &ComputePipelineState,
    interface: &PipelineInterface,
    table: &ArgumentTable,
    region: Option<StageInRegion>,
    grid: Option<[u64; 3]>,
) -> Result<(), Error> {
    let Some(input) = pipeline.stage_input() else {
        return Ok(());
    };
    for attribute in input.attributes.values() {
        let buffer_index = attribute.buffer_index;
        let Some(layout) = input.layouts.get(&buffer_index) else {
            continue;
        };
        let Some((buffer, offset)) = table.buffers.get(&buffer_index) else {
            continue;
        };
        let held = buffer.length().saturating_sub(*offset);
        if held <= REACHED || !reads_buffer(interface, buffer_index) {
            continue;
        }
        // A layout of `MTLBufferLayoutStrideDynamic` reads the bound stride, 0 for a buffer bound
        // without one (`MTLAttributeStrideStatic`'s is `2^32 - 1`).
        let stride = layout.stride.unwrap_or_else(|| {
            table
                .attribute_strides
                .get(&buffer_index)
                .copied()
                .unwrap_or(0)
        });
        let Some(element) = furthest_element(layout, input, region, grid) else {
            continue;
        };
        // A format the pipeline could not fetch is no attribute the kernel reads.
        let size = VertexFormat::try_from(attribute.format.as_raw()).map_or(16, format_bytes);
        let reach = reach(element, stride, attribute.offset, size);
        if reach.min(held) > REACHED {
            return Err(Error::NotImplemented(RuntimeGap::StageInFetchPast4GiB {
                buffer_index,
                reach,
            }));
        }
    }
    Ok(())
}

/// Whether `pipeline`'s kernel fetches a `[[stage_in]]` attribute: its stage-input descriptor
/// names one whose layout's buffer the kernel reads. An execution of commands a kernel writes
/// runs through the host step where such a state may run (`gpu_execute.rs`), which checks
/// each command's reach.
pub(crate) fn fetches_stage_in(pipeline: &ComputePipelineState) -> bool {
    pipeline.stage_input().is_some_and(|input| {
        input.attributes.values().any(|attribute| {
            input.layouts.contains_key(&attribute.buffer_index)
                && reads_buffer(pipeline.interface(), attribute.buffer_index)
        })
    })
}

/// Whether the kernel reads the buffer at `index` of its table: a stage-in attribute's fetch
/// reads its layout's buffer slot.
fn reads_buffer(interface: &PipelineInterface, index: u32) -> bool {
    interface.slots.keys().any(|slot| {
        slot.stage == Stage::Kernel && slot.class == SlotClass::Buffer && slot.index == index
    })
}

/// The furthest element a dispatch over `grid` and `region` may fetch of `layout` of `input`
/// (the module docs): `None` where it runs no thread, `u64::MAX` for a step no kernel fetches.
fn furthest_element(
    layout: &StageInputLayoutDescriptor,
    input: &StageInputDescriptor,
    region: Option<StageInRegion>,
    grid: Option<[u64; 3]>,
) -> Option<u64> {
    use StepFunction as S;
    let (axis, indexed) = match layout.step_function {
        S::Constant => return Some(0),
        S::ThreadPositionInGridX => (0, false),
        S::ThreadPositionInGridY => (1, false),
        S::ThreadPositionInGridXIndexed => (0, true),
        S::ThreadPositionInGridYIndexed => (1, true),
        _ => return Some(u64::MAX),
    };
    let position = match grid {
        Some(grid) if grid.contains(&0) => return None,
        Some(grid) => grid[axis] - 1,
        None => u64::from(u32::MAX),
    };
    let element = match (indexed, input.index_type) {
        (true, IndexType::UInt16) => u64::from(u16::MAX),
        (true, _) => u64::from(u32::MAX),
        (false, _) => position / u64::from(layout.step_rate.max(1)),
    };
    let (origin, size) = region.map_or((0, 0), |region| (region.origin[axis], region.size[axis]));
    // The region's last element, a 32-bit difference: a size of 0 clamps nothing.
    let last = u64::from(size.wrapping_sub(1));
    Some(u64::from(origin) + element.min(last))
}

#[cfg(test)]
mod tests {
    use super::{furthest_element, grid, reach, REACHED};
    use metal2vulkan::bind::{Dispatch, StageInRegion};
    use metal2vulkan::metal::{IndexType, StepFunction};
    use metal2vulkan::pipeline::{StageInputDescriptor, StageInputLayoutDescriptor};

    /// Bytes past `2^32` are reached only by a displacement there, and a stride of 0 reads the
    /// offset alone, whatever the element.
    #[test]
    fn the_reach_is_the_last_byte_read() {
        assert_eq!(reach(3, 32, 8, 16), 3 * 32 + 8 + 16);
        assert_eq!(reach(u64::MAX, 0, 8, 16), 8 + 16);
        assert!(reach(u64::from(u32::MAX), 2, 0, 4) > REACHED);
        assert_eq!(reach(u64::from(u32::MAX), 1, 0, 1), REACHED);
    }

    /// A thread-position step's furthest element is the region's origin plus the grid's last
    /// position over the rate, clamped to the region's last element; an indexed step's the
    /// index type's largest index clamped so; a grid GPU memory states runs any 32-bit position.
    #[test]
    fn a_dispatch_fetches_its_region_s_furthest_element() {
        let layout = |step_function, step_rate| {
            let mut layout = StageInputLayoutDescriptor::new(Some(4));
            layout.step_function = step_function;
            layout.step_rate = step_rate;
            layout
        };
        let input = StageInputDescriptor::new();
        let threads = grid(&Dispatch::Threads {
            threads: [10, 3, 1],
            threads_per_group: [4, 1, 1],
        });
        assert_eq!(threads, Some([10, 3, 1]));
        let region = |origin, size| Some(StageInRegion::new([origin, 7, 0], [size, 0, 1]));
        let x = layout(StepFunction::ThreadPositionInGridX, 1);
        assert_eq!(
            furthest_element(&x, &input, region(5, 100), threads),
            Some(14)
        );
        assert_eq!(furthest_element(&x, &input, region(5, 4), threads), Some(8));
        assert_eq!(
            furthest_element(&x, &input, region(5, 0), threads),
            Some(14)
        );
        let halved = layout(StepFunction::ThreadPositionInGridX, 2);
        assert_eq!(
            furthest_element(&halved, &input, region(0, 0), threads),
            Some(4)
        );
        let y = layout(StepFunction::ThreadPositionInGridY, 1);
        assert_eq!(furthest_element(&y, &input, region(0, 0), threads), Some(9));
        let groups = grid(&Dispatch::Threadgroups {
            groups: [3, 1, 1],
            threads_per_group: [8, 1, 1],
        });
        assert_eq!(furthest_element(&x, &input, region(0, 0), groups), Some(23));
        assert_eq!(
            furthest_element(&x, &input, region(1, 0), None),
            Some(u64::from(u32::MAX) + 1)
        );
        assert_eq!(furthest_element(&x, &input, region(1, 6), None), Some(6));
        let empty = Some([0, 3, 1]);
        assert_eq!(furthest_element(&x, &input, region(0, 0), empty), None);
        let indexed = layout(StepFunction::ThreadPositionInGridXIndexed, 1);
        let mut short = StageInputDescriptor::new();
        short.index_type = IndexType::UInt16;
        assert_eq!(
            furthest_element(&indexed, &short, region(2, 0), threads),
            Some(2 + u64::from(u16::MAX))
        );
        let mut long = StageInputDescriptor::new();
        long.index_type = IndexType::UInt32;
        assert_eq!(
            furthest_element(&indexed, &long, region(2, 50), threads),
            Some(51)
        );
        let constant = layout(StepFunction::Constant, 0);
        assert_eq!(furthest_element(&constant, &input, None, None), Some(0));
    }
}
