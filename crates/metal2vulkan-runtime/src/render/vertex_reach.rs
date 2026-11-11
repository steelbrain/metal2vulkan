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
//!
//! A draw's fixed-function fetch past 4 GiB of a small buffer, on a driver whose fixed-function
//! vertex input wraps there (`RuntimeDriverDefect::VertexFetchWrapsPast4GiB`: lavapipe):
//!
//! - **What differs.** Metal reads an element past `2^32` bytes of the bind offset of a buffer
//!   that ends before them as zero; such a driver takes the element's displacement modulo `2^32`
//!   and reads the bytes there, though it bounds the fetch by the bound range
//!   (`robustBufferAccess2`) and Vulkan places the attribute at the whole displacement.
//! - **The pull** ([`far_layouts`]). A fixed-function layout of the pipeline (its binding's
//!   stride, a per-bind one the encoder's, and the end of its furthest attribute) that a draw may
//!   fetch past `2^32` bytes of its bind offset is named in the draw's bind state
//!   (`BindState::far_vertex_buffers`), and the plan pulls it (the pulled-vertex-buffer variant),
//!   whose fetch reads the bytes past `2^32` as zero (the refusal above covers a buffer that runs
//!   past them). The furthest element is the one above, from the draw's own values: its last
//!   vertex without indices, `65535` plus its base vertex with 16-bit indices, its base instance
//!   plus its last instance over the step rate. A per-vertex layout of an indexed draw of 32-bit
//!   indices, and any layout of an indirect draw, has its reach stated by GPU memory: pulled under
//!   `VertexReachPolicy::Exact`, else left in fixed function and announced
//!   (`vertex-fetch-may-wrap`, once per device). A draw that pulls every attribute anyway (the
//!   private list and de-indexed forms) names none. On every other driver no draw names one, and
//!   the check is one lookup of the device's defects.

use super::RenderLowering;
use crate::error::{Error, RuntimeGap};
use crate::fetch_reach::{format_bytes, reach, REACHED};
use metal2vulkan::bind::DrawCall;
use metal2vulkan::device::VertexReachPolicy;
use metal2vulkan::error::RuntimeDriverDefect;
use metal2vulkan::metal::VertexStepFunction;
use metal2vulkan::pipeline::{PulledAttribute, RenderPipeline, VertexInputState, VertexStride};
use metal2vulkan::vk;
use std::collections::{BTreeMap, BTreeSet};

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

    /// The fixed-function layouts of `pipeline` (its base: every fixed-function layout of its
    /// descriptor) that `draw` pulls on this device ([`far_layouts`], the module docs): none on a
    /// driver without `RuntimeDriverDefect::VertexFetchWrapsPast4GiB`.
    pub(super) fn far_vertex_buffers(
        &self,
        pipeline: &RenderPipeline,
        draw: &DrawCall,
    ) -> FarLayouts {
        let device = self.core.device();
        let defect = RuntimeDriverDefect::VertexFetchWrapsPast4GiB;
        if !device.properties().driver_defects.contains(&defect) {
            return FarLayouts::default();
        }
        let policy = device.compile_device().policy().vertex_reach;
        let layouts = fixed_layouts(&pipeline.vertex_input, &self.vertex.attribute_strides);
        far_layouts(layouts, draw, policy)
    }
}

/// What a draw's fixed-function layouts take on a driver whose fixed-function vertex input wraps
/// past `2^32` bytes ([`far_layouts`]).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct FarLayouts {
    /// The Metal vertex buffer indices of the layouts the draw pulls
    /// (`BindState::far_vertex_buffers`).
    pub(super) pulled: BTreeSet<u32>,
    /// A layout left in fixed function whose reach GPU memory states, under
    /// `VertexReachPolicy::Bounded`: announced `vertex-fetch-may-wrap`.
    pub(super) unchecked: Option<u32>,
}

/// One fixed-function layout as a draw's reach reads it.
#[derive(Clone, Copy, Debug)]
pub(super) struct FixedLayout {
    /// The Metal vertex buffer index.
    pub(super) buffer_index: u32,
    pub(super) step: VertexStepFunction,
    pub(super) rate: u32,
    /// The binding's stride: a static one, or the encoder's per-bind one (0 for a buffer bound
    /// without one, `render/vertex_strides.rs`).
    pub(super) stride: u32,
    /// The end of its furthest attribute: its offset plus its format's size.
    pub(super) end: u32,
}

/// The fixed-function layouts of `input`, a per-bind stride read from `strides` (the encoder's
/// vertex table's attribute strides).
pub(super) fn fixed_layouts<'a>(
    input: &'a VertexInputState,
    strides: &'a BTreeMap<u32, u32>,
) -> impl Iterator<Item = FixedLayout> + 'a {
    input.bindings.iter().filter_map(move |binding| {
        let end = (input.attributes.iter())
            .filter(|a| a.binding == binding.binding)
            .map(|a| {
                let size = a.format.block().map_or(16, |b| b.size);
                a.offset.saturating_add(size)
            })
            .max()?;
        let (step, rate) = match (binding.input_rate, binding.divisor) {
            (vk::VertexInputRate::VERTEX, _) => (VertexStepFunction::PerVertex, 1),
            (_, 0) => (VertexStepFunction::Constant, 1),
            (_, divisor) => (VertexStepFunction::PerInstance, divisor),
        };
        let stride = match binding.stride {
            VertexStride::Static(stride) => stride,
            VertexStride::PerBind => strides.get(&binding.buffer_index).copied().unwrap_or(0),
        };
        Some(FixedLayout {
            buffer_index: binding.buffer_index,
            step,
            rate,
            stride,
            end,
        })
    })
}

/// The layouts of `layouts` that `draw` pulls under `policy` on a driver whose fixed-function
/// vertex input wraps past `2^32` bytes (the module docs): each whose furthest element's reach
/// passes `2^32` bytes of its bind offset, where the draw's own values state it, or under
/// [`VertexReachPolicy::Exact`] where GPU memory does; under [`VertexReachPolicy::Bounded`] the
/// first left so is [`FarLayouts::unchecked`].
pub(super) fn far_layouts(
    layouts: impl Iterator<Item = FixedLayout>,
    draw: &DrawCall,
    policy: VertexReachPolicy,
) -> FarLayouts {
    let mut far = FarLayouts::default();
    for layout in layouts {
        let Some(element) = furthest_element(layout.step, layout.rate, draw) else {
            continue;
        };
        if reach(element, layout.stride, 0, layout.end) <= REACHED {
            continue;
        }
        if stated_by_gpu(layout.step, draw) && policy == VertexReachPolicy::Bounded {
            far.unchecked.get_or_insert(layout.buffer_index);
            continue;
        }
        far.pulled.insert(layout.buffer_index);
    }
    far
}

/// Whether GPU memory states how far `draw` fetches a layout of `step`: per vertex through 32-bit
/// indices, or any step but a constant one through an indirect draw's arguments.
fn stated_by_gpu(step: VertexStepFunction, draw: &DrawCall) -> bool {
    match (step, draw) {
        (VertexStepFunction::Constant, _) => false,
        (VertexStepFunction::PerVertex, DrawCall::Indexed { index_type, .. }) => {
            *index_type != vk::IndexType::UINT16
        }
        (
            _,
            DrawCall::Indirect { .. }
            | DrawCall::PatchesIndirect { .. }
            | DrawCall::MeshThreadgroupsIndirect { .. },
        ) => true,
        _ => false,
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
    use super::{far_layouts, furthest_element, FarLayouts, FixedLayout};
    use metal2vulkan::bind::{DrawCall, IndexBufferRef, IndirectRef};
    use metal2vulkan::device::VertexReachPolicy;
    use metal2vulkan::metal::VertexStepFunction;
    use metal2vulkan::vk;
    use std::collections::BTreeSet;

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

    /// A `uint` layout at buffer `buffer_index` of `step` at `stride`.
    fn layout(buffer_index: u32, step: VertexStepFunction, stride: u32) -> FixedLayout {
        FixedLayout {
            buffer_index,
            step,
            rate: 1,
            stride,
            end: 4,
        }
    }

    /// An indexed draw of `index_type` from `base_vertex`, of base instance `base_instance`.
    fn indexed(index_type: vk::IndexType, base_vertex: i32, base_instance: u32) -> DrawCall {
        DrawCall::Indexed {
            index_type,
            index_buffer: IndexBufferRef::new(0, 64),
            index_count: 4,
            index_offset: 0,
            instance_count: 4,
            base_vertex,
            base_instance,
        }
    }

    /// The pulled layouts and the unchecked one of `layouts` for `draw` under both policies.
    fn both(layouts: &[FixedLayout], draw: &DrawCall) -> [FarLayouts; 2] {
        [VertexReachPolicy::Bounded, VertexReachPolicy::Exact]
            .map(|policy| far_layouts(layouts.iter().copied(), draw, policy))
    }

    fn pulled(buffers: &[u32]) -> FarLayouts {
        FarLayouts {
            pulled: buffers.iter().copied().collect(),
            unchecked: None,
        }
    }

    /// Under either policy a layout is pulled where the draw's own values reach past `2^32`
    /// bytes of its bind offset: a vertex start, a 16-bit draw's base vertex, a base instance;
    /// the element before the first past it, a stride of 0, and a constant step are not.
    #[test]
    fn a_draw_s_own_values_pull_a_layout_past_4_gib() {
        use VertexStepFunction as S;
        let layouts = [
            layout(1, S::PerVertex, 4),
            layout(2, S::PerVertex, 0),
            layout(3, S::PerInstance, 12),
            layout(4, S::Constant, 4),
        ];
        let primitives = |vertex_start, base_instance| DrawCall::Primitives {
            vertex_start,
            vertex_count: 4,
            instance_count: 4,
            base_instance,
        };
        // The last vertex at 2^30 - 1 ends at 2^32 exactly; at 2^30 it passes it.
        let far = both(&layouts, &primitives((1 << 30) - 4, 0));
        assert_eq!(far, [pulled(&[]), pulled(&[])]);
        let far = both(&layouts, &primitives((1 << 30) - 3, 0));
        assert_eq!(far, [pulled(&[1]), pulled(&[1])]);
        // The last instance at 357,913,942 lies at 2^32 + 8 at a stride of 12.
        let far = both(&layouts, &primitives(0, 357_913_939));
        assert_eq!(far, [pulled(&[3]), pulled(&[3])]);
        let short = indexed(vk::IndexType::UINT16, 1 << 30, 0);
        assert_eq!(both(&layouts, &short), [pulled(&[1]), pulled(&[1])]);
        let short = indexed(vk::IndexType::UINT16, 0, 0);
        assert_eq!(both(&layouts, &short), [pulled(&[]), pulled(&[])]);
    }

    /// A per-vertex layout of a draw of 32-bit indices, and any layout of an indirect draw, reach
    /// as GPU memory states: pulled under `Exact`, left unchecked under `Bounded`, whose first is
    /// named. A per-instance layout of an indexed draw keeps its own bound.
    #[test]
    fn gpu_memory_s_reach_is_pulled_under_the_exact_policy_alone() {
        use VertexStepFunction as S;
        let layouts = [layout(1, S::PerVertex, 4), layout(3, S::PerInstance, 12)];
        let long = indexed(vk::IndexType::UINT32, 0, 0);
        let [bounded, exact] = both(&layouts, &long);
        assert_eq!(
            bounded,
            FarLayouts {
                pulled: BTreeSet::new(),
                unchecked: Some(1),
            }
        );
        assert_eq!(exact, pulled(&[1]));
        let indirect = DrawCall::Indirect {
            indirect: IndirectRef::new(0, 0),
            indexed: None,
        };
        let [bounded, exact] = both(&layouts, &indirect);
        assert_eq!((bounded.pulled.len(), bounded.unchecked), (0, Some(1)));
        assert_eq!(exact, pulled(&[1, 3]));
        // A base instance past 2^30 is the draw's own: pulled under either policy.
        let far = indexed(vk::IndexType::UINT32, 0, 357_913_942);
        let [bounded, exact] = both(&layouts, &far);
        assert_eq!(bounded.pulled, BTreeSet::from([3]));
        assert_eq!(bounded.unchecked, Some(1));
        assert_eq!(exact, pulled(&[1, 3]));
    }
}
