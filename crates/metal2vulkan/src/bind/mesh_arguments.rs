//! What Metal draws of a mesh grid. The object function's lowering holds the mesh grid an object
//! threadgroup sets to it (`lower::mesh`): past these limits it launches nothing.
//!
//! Measured by our probe on an M3 Max (macOS 27.0.1), direct and indirect draws alike, with and
//! without API validation, which stops none of them: a grid with an axis past
//! [`METAL_MESH_GRID_AXIS`] (65,535) draws nothing; so does one of exactly 2^20 threadgroups (1024
//! by 1024, 128 by 128 by 64, 4096 by 256) and every one measured of 2^21 or more. Grids of up to
//! [`METAL_MESH_GRID_TOTAL`] (1,048,575, Apple's limit for the family) draw whole. Past that Metal
//! is erratic (17 by 61,681 and 65,535 by 24 drew, 1025 by 1025 and 33 by 32,768 did not); nothing
//! is Metal's answer at 2^20 and above 2^21. The mesh grid an object threadgroup sets follows the
//! same rule (65,536 along `x` or `y`, 1024 by 1024 and 2048 by 1024 launch nothing from it; 1023
//! by 1025 and 65,535 by 16 launch whole).

/// The most mesh threadgroups along an axis of a grid Metal draws without an object function
/// (the module docs).
pub const METAL_MESH_GRID_AXIS: u32 = 65_535;

/// The most mesh threadgroups of a grid Metal draws whole without an object function (the module
/// docs).
pub const METAL_MESH_GRID_TOTAL: u32 = 1_048_575;
