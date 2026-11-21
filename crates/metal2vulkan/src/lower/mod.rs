//! Strategy selection and per-family lowering.
//!
//! Every family is declared and routed up front, so a family's lowering edits none of `select.rs`,
//! `strategy.rs`, `emit.rs`, `push.rs`, or this file:
//!
//! - [`select`]: the total routing of every instruction, intrinsic family, and parameter, and the
//!   per-function [`select::Plan`] (push placement, size estimate).
//! - [`strategy`]: [`strategy::Strategy`], one variant per family wrapping its own enum.
//! - [`emit`]: [`emit::Emitter`], the `CodeEmitter` that routes each strategy to its family's
//!   emit function, and the module driver.
//! - [`push`]: the complete placement algorithm; [`estimate`]: `SizeEstimate`;
//!   [`view`]: the structurizer's `CfgView`.
//! - Families: [`arith`], [`access`], [`texture`], [`sampler`], [`simd`], [`sync`] (`memory_model`,
//!   `barrier`), [`fragment`], [`stage_io`], [`dispatch`], [`raytracing`], [`mesh`]. A family
//!   refuses what it does not lower as `Unsupported{Lowering(family), NotImplemented}`; the
//!   engine fixture inventory (`tests/public_fixture_expectations_engine.tsv`) records what
//!   lowers on each profile.
//!
//! Every family-owned decision is a hook in the family's own module (parameter values, returns,
//! family plans and `finish`, module-scope `declare`, slot forms, descriptor counts, simd class,
//! tangled instructions); `select` and `emit` only combine what the families contribute.

// Panic-lint policy for the new core: product code in this tree may not unwrap or index unchecked;
// tests are exempt.
#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::indexing_slicing))]

pub mod access;
pub mod arith;
pub mod dispatch;
pub mod emit;
pub mod estimate;
pub mod fragment;
pub mod mesh;
pub mod push;
pub mod raytracing;
pub mod sampler;
pub mod select;
pub mod simd;
pub mod stage_io;
pub mod strategy;
pub mod sync;
pub mod tessellation;
pub mod texture;
pub mod users;
pub mod view;
