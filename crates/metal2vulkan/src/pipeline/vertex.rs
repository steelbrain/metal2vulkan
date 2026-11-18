//! `VertexInputState`: a complete, device-valid Vulkan vertex input description.
//!
//! Every format substitution is chosen with the shader's real read width in hand, so the consumer
//! never guards against `(x, y, z, 1)` versus `(x, y, z, garbage)` reads.

use std::collections::BTreeSet;

use crate::metal;
use crate::vk;

/// Vertex input of a render pipeline.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VertexInputState {
    /// Fixed-function bindings, in binding order.
    pub bindings: Vec<VertexBinding>,
    /// Fixed-function attributes, in location order.
    pub attributes: Vec<VertexAttribute>,
    /// Attributes fetched in the shader; the plan binds their source buffers.
    pub pulled: Vec<PulledAttribute>,
    /// The Metal vertex-buffer indices whose layouts fixed-function vertex input fetches in the
    /// descriptor's own projection but this pipeline pulls, because it was compiled under a
    /// `VariantKey` naming them (`pulled_vertex_buffers`). Its plans keep listing them.
    pub variant_pulled: BTreeSet<u32>,
}

impl VertexInputState {
    /// Whether a fixed-function attribute can end past its binding's stride, which the
    /// portability subset gates on `vertexAttributeAccessBeyondStride`: a static stride below the
    /// attribute's end, or a per-bind one, which the bind sets. Where the subset withholds the
    /// feature, link pulls the first and the binder a bound stride below the end, so a pipeline
    /// names the feature only where the device has it.
    pub(crate) fn may_access_beyond_stride(&self) -> bool {
        self.attributes.iter().any(|a| {
            let end = a
                .format
                .block()
                .map_or(u32::MAX, |b| a.offset.saturating_add(b.size));
            self.bindings
                .iter()
                .filter(|b| b.binding == a.binding)
                .any(|b| match b.stride {
                    VertexStride::Static(s) => end > s,
                    VertexStride::PerBind => true,
                })
        })
    }
}

/// A Vulkan vertex binding fed by one Metal vertex-buffer layout.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VertexBinding {
    pub binding: u32,
    /// The Metal buffer index (the layout index of the vertex descriptor).
    pub buffer_index: u32,
    pub input_rate: vk::VertexInputRate,
    /// Instance divisor (`VK_EXT_vertex_attribute_divisor`); 1 when not divided, 0 for a constant
    /// step at instance rate with zero-divisor support. Under the identity domains a constant step
    /// is a vertex-rate binding of divisor 1 and stride 0.
    pub divisor: u32,
    pub stride: VertexStride,
}

/// Where a vertex binding's stride comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum VertexStride {
    /// Fixed in the pipeline.
    Static(u32),
    /// Set per bind (`setVertexBuffer:offset:attributeStride:`): dynamic state, or static pipeline
    /// state per stride on profiles without dynamic strides.
    PerBind,
}

/// A fixed-function vertex attribute.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VertexAttribute {
    pub location: u32,
    pub binding: u32,
    pub format: vk::Format,
    pub offset: u32,
}

/// An attribute fetched in the shader: its format is not a supported vertex format, or its step
/// function has no fixed-function equivalent on the profile.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PulledAttribute {
    /// The Metal attribute index.
    pub attribute: u32,
    /// The Metal buffer index the attribute reads.
    pub buffer_index: u32,
    pub format: metal::VertexFormat,
    pub offset: u32,
    pub step_function: metal::VertexStepFunction,
    pub step_rate: u32,
    /// The layout's stride: the binder writes a static one, or the bind's `attributeStride` for
    /// a per-bind one, into `PushField::VertexBufferStride`. A constant step reads none.
    pub stride: VertexStride,
}
