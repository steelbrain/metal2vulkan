//! A Metal draw call in Metal's terms ([`DrawCall`], with its index buffer, [`IndexBufferRef`]),
//! and the code a pipeline's private alpha test compares with ([`alpha_test_code`]). How a draw
//! becomes Vulkan draws, after the pipeline's interface and push fields, is the consumer's.

use crate::vk;

/// A consumer-owned index buffer.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct IndexBufferRef {
    /// The consumer's id for the buffer.
    pub token: u32,
    /// The end of the buffer's bytes (`MTLBuffer.length`), in the offsets the draw binds the
    /// `VkBuffer` at (the length itself where the consumer binds the buffer from its start).
    pub length: u64,
}

impl IndexBufferRef {
    /// The consumer's index buffer `token`, `length` bytes long.
    pub fn new(token: u32, length: u64) -> IndexBufferRef {
        IndexBufferRef { token, length }
    }
}

/// A Metal draw call.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum DrawCall {
    /// `drawPrimitives:vertexStart:vertexCount:instanceCount:baseInstance:`.
    Primitives {
        vertex_start: u32,
        vertex_count: u32,
        instance_count: u32,
        base_instance: u32,
    },
    /// `drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:…`.
    Indexed {
        index_type: vk::IndexType,
        index_buffer: IndexBufferRef,
        index_count: u32,
        index_offset: u64,
        instance_count: u32,
        base_vertex: i32,
        base_instance: u32,
    },
    /// `drawPatches:patchStart:patchCount:patchIndexBuffer:patchIndexBufferOffset:instanceCount:
    /// baseInstance:` of a tessellation pipeline. `control_points` is `numberOfPatchControlPoints`;
    /// `patch_index` the patch index buffer and its offset, a multiple of 4 (`None` for `nil`): the
    /// `uint` at the offset plus `4 · (patchStart + i)` is patch `i`'s index, whose control points
    /// and per-patch attributes it reads, while its factors stay at `patchStart + i`.
    /// `control_point_index` is `drawIndexedPatches:...`'s control-point index buffer and its
    /// offset, a multiple of the pipeline's index size, for a pipeline with a control-point index
    /// type (`None` for `drawPatches`): control point `k` of the patch whose index is `p` is the
    /// index at position `p · numberOfPatchControlPoints + k` from the offset.
    Patches {
        control_points: u32,
        patch_start: u32,
        patch_count: u32,
        patch_index: Option<(IndexBufferRef, u64)>,
        instance_count: u32,
        base_instance: u32,
        control_point_index: Option<(IndexBufferRef, u64)>,
    },
    /// `drawMeshThreadgroups:threadsPerObjectThreadgroup:threadsPerMeshThreadgroup:` of a mesh
    /// render pipeline: `groups` object threadgroups of `threads_per_object_group` threads, each
    /// launching the mesh grid it sets, or, without an object function, `groups` mesh
    /// threadgroups of `threads_per_mesh_group` threads.
    MeshThreadgroups {
        groups: [u32; 3],
        threads_per_object_group: [u32; 3],
        threads_per_mesh_group: [u32; 3],
    },
    /// `drawMeshThreads:threadsPerObjectThreadgroup:threadsPerMeshThreadgroup:`: without an object
    /// function, `threads / threads_per_mesh_group` mesh threadgroups along each axis, rounded
    /// down (Metal's measured answer).
    MeshThreads {
        threads: [u32; 3],
        threads_per_object_group: [u32; 3],
        threads_per_mesh_group: [u32; 3],
    },
}

/// The 8-bit unorm code Metal's private alpha test compares the reference `reference` as
/// (`setAlphaTestReferenceValue:`, Apple SPI; `PushField::AlphaTestReference`, probes on an Apple
/// M3 Max): NaN is 0, the value is clamped to `[0, 1]`, and the code is the exact product with
/// 255 rounded to nearest, as the fragment function converts its alpha
/// (`lower::arith::color_write::unorm8_code`). The product of an `f32` and 255 is exact as an
/// `f64`, and its only tie, at 0.5, rounds to 128 either way. A runtime fills the push field with
/// it.
pub fn alpha_test_code(reference: f32) -> u32 {
    let clamped = match reference.is_nan() {
        true => 0.0,
        false => f64::from(reference.clamp(0.0, 1.0)),
    };
    (clamped * 255.0).round_ties_even() as u32
}
