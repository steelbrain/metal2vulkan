//! Typed SPIR-V 1.5 IR, builder, verifier, requirement derivation, serializer.

// Panic-lint policy for the new core: product code in this tree may not unwrap or index unchecked;
// tests are exempt.
#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::indexing_slicing))]

pub mod builder;
pub mod dead;
pub mod error;
pub mod half_hold;
pub mod hoist;
pub mod ids;
// The CPU interpreter: test support for the crate's own tests.
#[cfg(test)]
pub mod interp;
pub mod ir;
pub mod requirements;
pub mod roll;
pub mod serialize;
pub mod spec_bits;
// Test support for the crate's own tests.
#[cfg(test)]
pub mod testing;
pub mod verify;
