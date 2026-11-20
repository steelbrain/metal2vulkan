//! `ModuleBuilder` / `FunctionBuilder`.
//!
//! Every constructor checks its typing rules and derives result types, so a module built here is
//! well-typed by construction; an ill-formed request is a typed [`BuildError`](super::error::BuildError),
//! never an invalid module.

pub mod constructs;
mod function;
mod module;
pub mod ops;

pub use function::FunctionBuilder;
pub(crate) use module::{geometry_per_vertex, mesh_arrayed, mesh_per_primitive};
pub use module::{ModuleBuilder, NarrowFmt, ProductForm, QuantizeForm};

#[cfg(test)]
mod tests;
