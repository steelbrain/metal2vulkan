//! The binder's size of a stage-in attribute against the fetch's own.
//!
//! The binder decides a dispatch's `SpecKey::StageInReach` from each grid fetch's element end,
//! its offset plus its format's store size (`bind::dispatch`'s `element_end`), and takes that size
//! from the vertex format table (`metal::vertex::vertex_bytes`, the Vulkan format's block), so
//! that it reaches no lowering. The shader's reach check takes the same size from the fetch's
//! encoding (`lower::stage_io::fetch::Encoding::bytes`). Where the two disagreed, a key of false
//! would drop a check the fetch needs, or a key of true keep one it does not: this holds them
//! equal for every format.
