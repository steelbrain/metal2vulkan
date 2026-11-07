//! A draw's vertex fetch past 4 GiB of its buffer: an interim refusal of a silent wrong answer
//! (`fetch_reach.rs`).
//!
//! - **What differs.** The compile layer fetches an attribute it pulls
//!   (`VertexInputState::pulled`: a format or step fixed-function vertex input does not take, or
//!   every attribute where the device bounds no fixed-function fetch) whose bytes lie past `2^32`
//!   from the buffer's bind offset as zero bytes, where Metal's displacement is 64 bits (in a
//!   buffer of 4 GiB + 1 MiB, element 1 at stride `2^32 - 1` read the bytes at `2^32 - 1` and
//!   `2^32 + 3`).
//! - **The refusal.** A draw (the encoder's, or an indirect command buffer's command,
//!   `render/executions.rs`) that may read a pulled attribute past `2^32` bytes of its buffer's
//!   bind offset, inside a buffer that runs past them, is refused
//!   `NotImplemented(VertexFetchPast4GiB)`, naming the buffer index and the furthest byte it may
//!   read. A draw that may not is drawn as before; so is one past a buffer of at most `2^32`
//!   bytes past its bind offset, which Metal reads outside the buffer.
//! - **The furthest element** a draw may fetch of each step: a direct draw's own (its last
//!   vertex, its last instance divided by the step rate, its last patch or control point); an
//!   indexed draw's index type's largest index plus its base vertex; `2^32 - 1` for a patch's
//!   index or a control point's read from a buffer; and no bound for a draw whose counts and
//!   starts are read from an indirect buffer. A constant step fetches its first element alone,
//!   and a stride of 0 reads every element at the attribute's offset. So an indexed draw of
//!   32-bit indices, or an indirect draw, that pulls an attribute at a stride from such a buffer
//!   is refused whatever its indices or arguments hold: they are GPU memory.

use super::RenderLowering;
use crate::error::{Error, RuntimeGap};
use crate::fetch_reach::{format_bytes, reach, REACHED};
use metal2vulkan::bind::DrawCall;
use metal2vulkan::metal::VertexStepFunction;
use metal2vulkan::pipeline::{PulledAttribute, VertexStride};
use metal2vulkan::vk;

impl RenderLowering {
    /// Refuse `draw` (`NotImplemented(VertexFetchPast4GiB)`, the module docs) where it may
    /// read an attribute of `pulled`, its pipeline variant's pulled attributes, past `2^32` bytes
    /// of the bind offset of a buffer of the encoder's vertex table that runs past them.
    pub(super) fn check_vertex_reach(
        &self,
        pulled: &[PulledAttribute],
        draw: &DrawCall,
    ) -> Result<(), Error> {
        for attribute in pulled {
            let Some((buffer, offset)) = self.vertex.buffers.get(&attribute.buffer_index) else {
                continue;
            };
            let held = buffer.length().saturating_sub(*offset);
            if held <= REACHED {
                continue;
            }
            let stride = match attribute.stride {
                VertexStride::Static(stride) => stride,
                // A layout of `MTLBufferLayoutStrideDynamic` reads a buffer bound without a stride
                // at stride 0 (`render/vertex_strides.rs`).
                VertexStride::PerBind => self
                    .vertex
                    .attribute_strides
                    .get(&attribute.buffer_index)
                    .copied()
                    .unwrap_or(0),
            };
            let Some(element) =
                furthest_element(attribute.step_function, attribute.step_rate, draw)
            else {
                continue;
            };
            let reach = reach(
                element,
                stride,
                attribute.offset,
                format_bytes(attribute.format),
            );
            if reach.min(held) > REACHED {
                return Err(Error::NotImplemented(RuntimeGap::VertexFetchPast4GiB {
                    buffer_index: attribute.buffer_index,
                    reach,
                }));
            }
        }
        Ok(())
    }
}

/// The furthest element `draw` may fetch of an attribute of `step` and `rate` (the module docs):
/// `None` where it fetches none (a count of 0), `u64::MAX` where nothing bounds it.
fn furthest_element(step: VertexStepFunction, rate: u32, draw: &DrawCall) -> Option<u64> {
    const UNBOUNDED: u64 = u64::MAX;
    let last = |start: u32, count: u32| {
        count
            .checked_sub(1)
            .map(|last| u64::from(start) + u64::from(last))
    };
    let instance = |base: u32, count: u32| {
        count
            .checked_sub(1)
            .map(|last| u64::from(base) + u64::from(last / rate.max(1)))
    };
    let largest = u64::from(u32::MAX);
    match (step, *draw) {
        (VertexStepFunction::Constant, _) => Some(0),
        (
            VertexStepFunction::PerVertex,
            DrawCall::Primitives {
                vertex_start,
                vertex_count,
                ..
            },
        ) => last(vertex_start, vertex_count),
        (
            VertexStepFunction::PerVertex,
            DrawCall::Indexed {
                index_type,
                index_count,
                base_vertex,
                ..
            },
        ) => {
            let index = if index_type == vk::IndexType::UINT16 {
                u64::from(u16::MAX)
            } else {
                largest
            };
            let element = index.saturating_add_signed(i64::from(base_vertex));
            (index_count != 0).then_some(element)
        }
        (
            VertexStepFunction::PerInstance,
            DrawCall::Primitives {
                instance_count,
                base_instance,
                ..
            }
            | DrawCall::Indexed {
                instance_count,
                base_instance,
                ..
            }
            | DrawCall::Patches {
                instance_count,
                base_instance,
                ..
            },
        ) => instance(base_instance, instance_count),
        (
            VertexStepFunction::PerPatch,
            DrawCall::Patches {
                patch_start,
                patch_count,
                patch_index,
                ..
            },
        ) => match patch_index {
            None => last(patch_start, patch_count),
            Some(_) => (patch_count != 0).then_some(largest),
        },
        (
            VertexStepFunction::PerPatchControlPoint,
            DrawCall::Patches {
                patch_start,
                patch_count,
                patch_index,
                control_points,
                control_point_index,
                ..
            },
        ) => {
            let patch = match patch_index {
                None => last(patch_start, patch_count)?,
                Some(_) => (patch_count != 0).then_some(largest)?,
            };
            match control_point_index {
                None => Some(
                    (patch + 1)
                        .saturating_mul(u64::from(control_points))
                        .saturating_sub(1),
                ),
                Some(_) => Some(largest),
            }
        }
        // The indirect forms read their counts and starts from GPU memory, and no other draw
        // fetches the remaining steps.
        _ => Some(UNBOUNDED),
    }
}

#[cfg(test)]
mod tests {
    use super::furthest_element;
    use metal2vulkan::bind::{DrawCall, IndexBufferRef};
    use metal2vulkan::metal::VertexStepFunction;
    use metal2vulkan::vk;

    fn primitives(vertex_start: u32, vertex_count: u32) -> DrawCall {
        DrawCall::Primitives {
            vertex_start,
            vertex_count,
            instance_count: 3,
            base_instance: 5,
        }
    }

    /// The furthest element of each step a direct draw fetches is its own, and of an indexed
    /// draw the index type's largest plus the base vertex (the module docs).
    #[test]
    fn a_draw_fetches_its_own_furthest_element() {
        use VertexStepFunction as S;
        let draw = primitives(10, 4);
        assert_eq!(furthest_element(S::PerVertex, 1, &draw), Some(13));
        assert_eq!(furthest_element(S::PerInstance, 1, &draw), Some(7));
        assert_eq!(furthest_element(S::PerInstance, 2, &draw), Some(6));
        assert_eq!(furthest_element(S::Constant, 1, &draw), Some(0));
        assert_eq!(furthest_element(S::PerVertex, 1, &primitives(10, 0)), None);
    }

    /// An indexed draw's furthest vertex is its index type's largest plus the base vertex
    /// (the reach of an element, `fetch_reach.rs`).
    #[test]
    fn an_indexed_draw_fetches_up_to_its_largest_index() {
        let indexed = |index_type, base_vertex| DrawCall::Indexed {
            index_type,
            index_buffer: IndexBufferRef::new(0, 64),
            index_count: 3,
            index_offset: 0,
            instance_count: 1,
            base_vertex,
            base_instance: 0,
        };
        let vertex = VertexStepFunction::PerVertex;
        let short = indexed(vk::IndexType::UINT16, -5);
        assert_eq!(furthest_element(vertex, 1, &short), Some(65530));
        let long = indexed(vk::IndexType::UINT32, 7);
        assert_eq!(
            furthest_element(vertex, 1, &long),
            Some(u64::from(u32::MAX) + 7)
        );
    }
}
