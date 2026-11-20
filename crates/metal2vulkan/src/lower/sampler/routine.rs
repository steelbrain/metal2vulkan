//! The sampling routines: the emulation a combined binding's sample shape takes, defined once in
//! the module and called by every sample of that shape with its plain data.
//!
//! **Why.** An emulation written at every sample site is compiled again at every site: the
//! consumer's CoreAnimation fragment `Xgc`, a linear filter emulated at each of its 256 sample
//! sites, was 246,000 instructions from 15,800 AIR instructions, about 574 instructions a sample
//! site; NVIDIA's driver compiled every copy, 13 to 26 s a pipeline on the RTX 5080. The same
//! emulation as one routine that 256 sites call, declared `DontInline`, made in 0.39 s on the RTX
//! 3090 where the 256 inlined copies took 10.2 s.
//!
//! **What a routine holds.** One per shape ([`Shape`]):
//! - [`Filter`]: Metal's anisotropic filter emulated (`anisotropic`): the level-0 size and level
//!   count, the sampler's degree and mip filter, the footprint, and the taps' samples at explicit
//!   levels and their sum.
//! - [`Extended`]: an extended-range texture's sample (`extended`): the levels its mip filter
//!   reads, the filter's footprint and addressing, the fetches and their decodes, the sums' calls,
//!   the border and the swizzle.
//!
//! The routine reads its binding's image and sampler variables and its slots' specialization
//! constants (the runtime sampler's word, the offset tests) itself, so no image handle crosses a
//! call, and a device's specialization folds its arithmetic as it folded each site's. The site
//! keeps the word's tests, the device's sample with its level, bias or gradients, and the texel's
//! conversion: no implicit level crosses a call, so every answer is the one the inline code gave.
//!
//! An anisotropic sample's call passes its coordinate (a cube's direction, its ties broken), its
//! gradients (a cube's on its face) and a cube's face steps, the array layer and the
//! `min_lod_clamp`, and its texel offset, constant or dynamic, where the profile admits the
//! `Offset` operand on a sample (`maintenance8` and `shaderImageGatherExtended`), which the taps'
//! samples take, so the samples at every offset call one routine; elsewhere a constant offset is
//! the shape's, as the taps' `ConstOffset`. The site keeps the gradients (no implicit derivative
//! crosses a call), the word's test, the plain arm and the texel's conversion of the sum. An
//! extended-range sample's call passes its coordinate, the array layer and the level its mip
//! filter reads at (an explicit level, or the level of detail its site takes from derivatives or
//! gradients), and its constant texel offset, which the routine places on each lane, so the
//! samples at every offset call one routine; the site keeps the result's status word.
//!
//! **Not routines.** A heap image's sample (an argument buffer's), whose `NonUniform` handle a
//! call would carry (a later step's check); a switch-at-use's shared emulation
//! (`sampler::shared_value`), which runs once already; a gather's fetches; an anisotropic sample
//! at a dynamic texel offset on a profile without the `Offset` operand on a sample, whose arms its
//! site's access reads; and the bicubic filter's site code.
//!
//! **The call's control.** A sampling routine is a kept routine (`ModuleBuilder::kept_routine`),
//! declared `DontInline` on every device: a driver that honours the control (NVIDIA's) compiles
//! it once, and Mesa's drivers inline every function whatever it states, so there the module is
//! smaller and the pipeline the same.
//!
//! **Census**. An extended-range sample's routine is counted once per shape the plan notes
//! ([`note_extended`], `sampler::routine_words`), under `sampler.extended_range.routine`
//! (`extended::Sample::words` its site's). The census counts no anisotropic emulation, at its
//! sites or in a routine.

use std::collections::HashMap;

use crate::metal::format::ExtendedRange;
use crate::spirv::ids::{FuncId, TypeId};

use super::anisotropic::Anisotropic;
use super::bicubic::Bicubic;
use super::extended;
use super::{BindingKey, SamplerPlan};

/// The estimated words a routine adds besides its body: `OpFunction`, its parameters, the entry
/// label, the return, and `OpFunctionEnd`.
pub const ROUTINE_WORDS: u32 = 48;

/// The estimated words a sample spends at its site on a routine's call: the call with its
/// arguments, and the arm's label and branch.
pub const CALL_WORDS: u32 = 32;

/// The shape of a sampling routine (module docs): what its body depends on, besides its call's
/// argument types.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Shape {
    /// Metal's anisotropic filter emulated (`anisotropic`).
    Anisotropic(Filter),
    /// An extended-range texture's sample (`extended`).
    Extended(Box<Extended>),
}

impl Shape {
    /// The census name of the shape's routines.
    pub fn census_name(&self) -> &'static str {
        match self {
            Shape::Anisotropic(_) => "sampler.anisotropic.routine",
            Shape::Extended(_) => "sampler.extended_range.routine",
        }
    }

    /// An upper estimate of the routine's words: its body's, which the site counted previously, and
    /// the function's own.
    pub fn words(&self) -> u32 {
        match self {
            Shape::Anisotropic(_) => FILTER_WORDS + ROUTINE_WORDS,
            Shape::Extended(e) => e.words + ROUTINE_WORDS,
        }
    }
}

/// The estimated words of an anisotropic emulation's body (`anisotropic`), its taps past the first
/// in one loop: about 2,100 measured on a runtime slot's.
pub const FILTER_WORDS: u32 = 2_400;

/// Metal's anisotropic filter emulated (`anisotropic`), as a sample of a texture slot through its
/// combined binding takes it.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Filter {
    /// The sample's combined binding.
    pub binding: BindingKey,
    /// The sampler's degree, mip filter and the device's anisotropy.
    pub filter: Anisotropic,
    /// The coordinate's plane lanes (2; a cube direction's 3).
    pub lanes: u8,
    /// The sample reads a cube: its call passes the face's steps.
    pub cube: bool,
    /// The sample reads an array's layer.
    pub layer: bool,
    /// The sample states a `min_lod_clamp`.
    pub min_lod: bool,
    /// The constant texel offset the taps' samples take as `ConstOffset`, where the sample has one
    /// and the profile admits no `Offset` operand on a sample.
    pub offset: Option<Vec<i32>>,
    /// The call passes the sample's texel offset, constant or dynamic, which the taps' samples
    /// take as their `Offset` operand: on a profile that admits it on a sample (`maintenance8`
    /// and `shaderImageGatherExtended`), so the samples at every offset call one routine.
    pub passed_offset: bool,
}

/// An extended-range texture's sample (`extended::Sample`) without the operands its call passes:
/// its coordinate, layer and the level its mip filter reads at.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Extended {
    /// The texture slot, whose variant binds [`form`](Self::form).
    pub texture: u32,
    /// The extended-range form the slot's texels are decoded from.
    pub form: ExtendedRange,
    /// The sampler's state the sample follows.
    pub state: extended::State,
    /// Metal's bicubic filter of the decoded texels, where the sampler's state is cubic.
    pub cubic: Option<Bicubic>,
    /// A gather of this component instead of a sample.
    pub gather: Option<u32>,
    /// The sample swizzles its decoded texel under the slot's `TextureSwizzle` specialization.
    pub swizzle: bool,
    /// The lanes of the sample's constant texel offset, which the call passes, where it has one.
    pub offset: Option<u8>,
    /// The offset is placed by the slot's `SpecKey::OffsetBaseAxes` tests.
    pub offset_base: bool,
    /// A 3D texture: the sample reads three plane axes.
    pub volume: bool,
    /// The sample reads an array's layer, which the call passes.
    pub layer: bool,
    /// The mip filter the level is read by; a level the call passes but at the base level.
    pub level: extended::Mip,
    /// The result is `half`, else `float`.
    pub half: bool,
    /// The census words of the routine's body (`extended::Sample::body_words`).
    pub words: u32,
}

impl Extended {
    /// The routine extended-range sample `e` calls.
    pub(super) fn of(e: &extended::Sample) -> Extended {
        Extended {
            texture: e.texture,
            form: e.form,
            state: e.state.clone(),
            cubic: e.cubic.clone(),
            gather: e.gather,
            swizzle: e.swizzle,
            offset: e.offset.as_ref().map(|o| o.len() as u8),
            offset_base: e.offset_base,
            volume: e.volume,
            layer: e.layer.is_some(),
            level: e.level.mip(),
            half: e.half,
            words: e.body_words(),
        }
    }

    /// The plane axes the sample reads: a 2D texture's two, a 3D texture's three.
    pub(super) fn axes(&self) -> usize {
        match self.volume {
            true => 3,
            false => extended::AXES,
        }
    }
}

/// Note the routine of extended-range sample `e` in `plan`, its shape once, for the census.
pub(super) fn note_extended(plan: &mut SamplerPlan, e: &extended::Sample) {
    let shape = Shape::Extended(Box::new(Extended::of(e)));
    if !plan.routines.contains(&shape) {
        plan.routines.push(shape);
    }
}

/// The sampling routines one function's emission has defined, by shape and by the types of their
/// call's arguments: each defined at its first call.
#[derive(Debug, Default)]
pub(crate) struct Routines {
    defined: HashMap<(Shape, Vec<TypeId>), FuncId>,
}

impl Routines {
    /// The routine of `shape` taking arguments of `args`' types, where defined.
    pub(super) fn get(&self, shape: &Shape, args: &[TypeId]) -> Option<FuncId> {
        self.defined.get(&(shape.clone(), args.to_vec())).copied()
    }

    /// Record `f` as the routine of `shape` taking arguments of `args`' types.
    pub(super) fn insert(&mut self, shape: Shape, args: Vec<TypeId>, f: FuncId) {
        self.defined.insert((shape, args), f);
    }
}
