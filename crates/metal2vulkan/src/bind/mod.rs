//! What a consumer computes from what the guest bound, as plain data, beside the pipelines.
//!
//! The translator binds nothing: a consumer (the runtime) binds each resource itself, under the
//! identity domains, as the pipeline's interface and push fields state. This module holds the
//! pieces of that work the translator's own semantics decide:
//!
//! - An [`ArgumentBufferLayout`] says where an argument encoder (`MTLArgumentEncoder`) writes each
//!   argument of an argument buffer: a function's (`Function::argument_buffer_layout`), or one of
//!   [`ArgumentDescriptor`]s ([`ArgumentBufferLayout::from_descriptors`]).
//! - An ASTC texture on a device without its block is read through a twin of Metal's decode of its
//!   blocks ([`AstcDecode`], in the texels [`AstcTexels`] names); a PVRTC texture on a device
//!   without its format through a twin of its texels' codes ([`PvrtcDecode`]); an ETC2 or EAC
//!   texture on a device without the formats through a twin of its codes or an 11-bit format's
//!   values ([`EtcDecode`], in the texels [`EtcTexels`] names).
//! - [`clamped_depth`] is a 32-bit float depth clamped into `[0, 1]`, as a copy into a depth image
//!   of a device without `VK_EXT_depth_range_unrestricted` needs it.
//! - [`alpha_test_code`] is the code a pipeline's private alpha test compares with
//!   (`PushField::AlphaTestReference`).
//! - A [`RasterizationRateMap`] is the consumer's `MTLRasterizationRateMap`: it compiles a
//!   [`RateMapDescriptor`], answers the map's queries, and gives the parameter data a
//!   `rasterization_rate_map_decoder` reads.
//! - [`SamplerDescriptor`] is `MTLSamplerDescriptor`.
//! - [`Dispatch`], [`DrawCall`], [`IndexBufferRef`] and [`StageInRegion`] state a dispatch, a
//!   draw, its index buffer and the compute encoder's stage-in region in Metal's terms;
//!   [`indirect_commands`] lays out an indirect command buffer's records.
//! - [`function_table_header`] and [`function_table_entry_offset`] lay out the words of a
//!   visible-function table's buffer, which a `visible_function_table` argument reads.
//!
//! The frozen surface is the items named at this root; the submodules are crate-private.

pub(crate) mod argument_buffer;
pub(crate) mod astc_decode;
pub(crate) mod depth_clamp;
pub(crate) mod dispatch;
pub(crate) mod etc_decode;
pub(crate) mod graphics;
pub(crate) mod heap;
pub mod indirect_commands;
pub(crate) mod mesh_arguments;
pub(crate) mod neutral;
pub(crate) mod pvrtc_decode;
pub(crate) mod rate_map;
pub(crate) mod replay;
pub(crate) mod state;
pub(crate) mod status;
pub(crate) mod table;
pub(crate) mod texel_parts;
pub(crate) mod ycbcr_convert;

pub use argument_buffer::{
    ArgumentBufferLayout, ArgumentDescriptor, ArgumentLayoutError, ArgumentMember, ArgumentType,
    ARGUMENT_HANDLE_BYTES,
};
pub use astc_decode::{AstcDecode, AstcTexels};
pub use depth_clamp::clamped_depth;
pub use dispatch::Dispatch;
pub use etc_decode::{EtcDecode, EtcTexels};
pub use graphics::{alpha_test_code, DrawCall, IndexBufferRef};
pub use pvrtc_decode::PvrtcDecode;
#[cfg(test)]
pub(crate) use rate_map::RateMapAxis;
pub use rate_map::{
    RasterizationRateMap, RateMapDescriptor, RateMapError, RateMapLayerDescriptor, UnpinnedCompile,
    MAX_SCREEN_EXTENT,
};
pub(crate) use replay::{merged_fragment_table, merged_vertex, MergedVertex};
pub use state::{SamplerDescriptor, StageInRegion};
#[cfg(test)]
pub(crate) use table::FunctionTableWords;
pub use table::{function_table_entry_offset, function_table_header};
pub(crate) use texel_parts::part_texels;
pub(crate) use ycbcr_convert::{YCbCrCodes, YCbCrOutput};
