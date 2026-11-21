#![doc = include_str!("api_reference.md")]
// `too_many_arguments` and `type_complexity` are threshold heuristics that fire pervasively and
// benignly across this translator: lowering and canonicalization functions legitimately thread many
// typed parameters, and their return and table shapes are genuinely nested domain types. Factoring
// every such signature behind a wrapper struct or a one-off `type` alias would add indirection
// without improving clarity, so both are accepted crate-wide rather than scattered per-site.
#![allow(clippy::too_many_arguments, clippy::type_complexity)]

// The re-architected core. Its public modules are the API: `error`, `device`, `library`,
// `reflection`, `pipeline`, `bind`, `cache`, `metal`, `vk`, and `ash_interop` under feature `ash`,
// with `Budget` and `CancelToken` at the root. The AIR front end is the `metal2vulkan-air` crate, a
// documented library of its own that validation depends on directly; this crate names it `air`, and
// its `budget` and `graph` modules at its root, as when they were modules here, all crate-private.
// The translator core (`lower`, `memory`, `spirv`, `structurize`) and every submodule of a public
// module are crate-private: nothing outside the crate reaches them, this workspace's validation and
// dev tools included.
//
// `tests/api_surface.rs` pins which crate-root items are which.

pub(crate) use metal2vulkan_air as air;
#[cfg(feature = "ash")]
pub mod ash_interop;
pub mod bind;
pub(crate) use metal2vulkan_air::budget;
// The compile budget every entry point takes is public at the crate root; `budget` itself is
// internal.
pub use metal2vulkan_air::budget::{Budget, CancelToken};
pub mod cache;
pub mod device;
pub mod error;
pub(crate) use metal2vulkan_air::graph;
pub mod library;
pub(crate) mod lower;
pub(crate) mod memory;
pub mod metal;
pub mod pipeline;
pub mod reflection;
pub(crate) mod spirv;
// The SPIR-V grammar's opcodes and enumerants the translator names, generated from Khronos's grammar.
pub(crate) mod spv;
pub(crate) mod structurize;
pub mod vk;

// Kept from before the re-architecture (X34): the host command-record format (`indirect_commands`,
// which lowering and the binder's replay share, and which consumers and validation's oracle and
// case model reach as `bind::indirect_commands`), the half-precision conversions, the
// `METAL2VULKAN_*` registry (which the `metal2vulkan` binary reads, so it is `pub` and hidden in
// every build), and, for the crate's tests and test support only, the SPIR-V binary parser. The
// external-tool runner is the dev-tools crate's (`devtools/`).
#[doc(hidden)]
pub mod env_vars;
pub(crate) mod float16;
pub(crate) mod indirect_commands;
// The FALLBACK repro bundles the `metal2vulkan` binary writes (under
// feature `cli`): `pub` and hidden, as `env_vars` is, not API.
#[cfg(feature = "cli")]
#[doc(hidden)]
pub mod repro;
#[cfg(test)]
mod spirv_binary;

// The crate's whole-translator tests, which reach internal items directly.
#[cfg(test)]
mod tests;
