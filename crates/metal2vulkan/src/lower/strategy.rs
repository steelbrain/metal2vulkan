//! `Strategy`: the closed enum of per-family strategy enums.
//!
//! One variant per lowering family that instructions route to, each wrapping that family's own
//! enum, defined in the family module. A family whose lowering has not landed has an empty
//! (uninhabited) enum: its `select` refuses, so no plan ever holds one, and every match over it
//! is total by construction. Each variant of a family enum has exactly one emit function (the
//! family's `emit`), and a stable name and a word cost for the `StrategyCensus`.
//!
//! Interface-level lowering (stage IO, dispatch builtins, push fields) is planned per entry, not
//! per instruction, so it has no variant here.

use std::borrow::Cow;

use crate::lower::access::AccessStrategy;
use crate::lower::arith::ArithStrategy;
use crate::lower::fragment::FragmentStrategy;
use crate::lower::mesh::MeshStrategy;
use crate::lower::raytracing::RayTracingStrategy;
use crate::lower::sampler::SamplerStrategy;
use crate::lower::select::FamilyPlans;
use crate::lower::simd::SimdStrategy;
use crate::lower::sync::barrier::BarrierStrategy;
use crate::lower::sync::memory_model::MemoryModelStrategy;
use crate::lower::tessellation::TessellationStrategy;
use crate::lower::texture::TextureStrategy;

/// What every family's strategy enum states about each of its variants.
pub trait FamilyStrategy {
    /// The stable census name of one use of the variant in a finished plan whose family plans
    /// are `plans` (`family.variant`): read from the variant's payload and from the family plans'
    /// decisions that choose what the variant emits after it was selected, which are the access
    /// family's slot forms (a static-root access reaches a slot through its descriptor view or by
    /// its device address, `access::BufferReach`).
    fn name(&self, plans: &FamilyPlans) -> Cow<'static, str>;
    /// An upper estimate of the SPIR-V words the variant emits per use.
    fn words(&self) -> u32;
    /// Whether emission defines a value for the instruction that another instruction may read.
    /// A variant whose result no emitted instruction reads (the read sampler) defines none, so its
    /// value is never carried from one block to another.
    fn yields_value(&self) -> bool {
        true
    }
}

/// The lowering of one AIR instruction.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Strategy {
    Arith(ArithStrategy),
    Access(AccessStrategy),
    Texture(TextureStrategy),
    Sampler(SamplerStrategy),
    Simd(SimdStrategy),
    Barrier(BarrierStrategy),
    MemoryModel(MemoryModelStrategy),
    Fragment(FragmentStrategy),
    Tessellation(TessellationStrategy),
    RayTracing(RayTracingStrategy),
    Mesh(MeshStrategy),
}

impl Strategy {
    fn family(&self) -> &dyn FamilyStrategy {
        match self {
            Strategy::Arith(s) => s,
            Strategy::Access(s) => s,
            Strategy::Texture(s) => s,
            Strategy::Sampler(s) => s,
            Strategy::Simd(s) => s,
            Strategy::Barrier(s) => s,
            Strategy::MemoryModel(s) => s,
            Strategy::Fragment(s) => s,
            Strategy::Tessellation(s) => s,
            Strategy::RayTracing(s) => s,
            Strategy::Mesh(s) => s,
        }
    }
}

impl FamilyStrategy for Strategy {
    fn name(&self, plans: &FamilyPlans) -> Cow<'static, str> {
        self.family().name(plans)
    }

    fn words(&self) -> u32 {
        self.family().words()
    }

    fn yields_value(&self) -> bool {
        self.family().yields_value()
    }
}
