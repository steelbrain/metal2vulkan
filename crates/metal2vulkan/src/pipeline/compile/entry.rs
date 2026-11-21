//! Translating one entry of a bare AIR module, without a library: [`DeviceProfile::compile_entry`]
//! and the queries beside it.
//!
//! A tool that holds bitcode and an entry name (the CLI, a census or regression harness, a
//! refusal triage, a cache warmer) compiles here exactly as [`Device::compile_function`] compiles
//! a library's function: the same decode, specialization, lowering, size cap and
//! `catch_unwind`. A translation depends on the profile alone, so these are the profile's.
//!
//! [`Device::compile_function`]: crate::device::Device::compile_function

use std::collections::BTreeMap;

use crate::air::Entry;
use crate::budget::{Budget, Meter, SPIRV_WORDS};
use crate::cache::VariantKey;
use crate::device::DeviceProfile;
use crate::error::{DescriptorError, Error};
use crate::library::{CompiledFunction, FunctionConstantValues};
use crate::pipeline::artifact::StrategyUse;
use crate::pipeline::compile::api::CompileReport;
use crate::pipeline::compile::{
    census, compile_one, guarded, lower_inputs, ordered_entry, plan_one, specialize_one, EntryRef,
    InputsFrom, LinkedSource, Request,
};
use crate::pipeline::descriptor::VertexDescriptor;
use crate::pipeline::layout::PipelineLayout;
use crate::pipeline::link::{link, LinkInputs};
use crate::pipeline::lower_inputs::LoweringInputs;
use crate::pipeline::relocate::relocate;
use crate::pipeline::sym::Stage;

/// One entry of a bare AIR module (a bitcode wrapper or a raw module), with the function constants
/// and linked functions it compiles with: what [`DeviceProfile::compile_entry`] takes where
/// [`Device::compile_function`](crate::device::Device::compile_function) takes a
/// `FunctionDescriptor`.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct EntryDescriptor<'a> {
    /// The module: a bitcode wrapper (`0x0b17c0de`) or raw bitcode.
    pub bitcode: &'a [u8],
    /// Which of the module's entries.
    pub entry: EntryRef<'a>,
    /// `MTLFunctionConstantValues`: a value for a constant the module does not declare is
    /// ignored, as Metal ignores it.
    pub constants: FunctionConstantValues,
    /// `MTLLinkedFunctions.functions`: the functions that resolve the entry's visible-function
    /// references.
    pub linked: Vec<LinkedSource<'a>>,
}

impl<'a> EntryDescriptor<'a> {
    /// Entry `entry` of `bitcode`, with no constant values and no linked functions.
    pub fn new(bitcode: &'a [u8], entry: EntryRef<'a>) -> EntryDescriptor<'a> {
        EntryDescriptor {
            bitcode,
            entry,
            constants: FunctionConstantValues::defaults(),
            linked: Vec::new(),
        }
    }

    /// The compile request of the entry under `inputs`. A bare module has no library to state
    /// its AIR hash, so the request hashes it here, as the compile did before.
    fn request(&self, inputs: InputsFrom<'a>) -> Request<'_> {
        Request {
            bitcode: self.bitcode,
            air_hash: crate::air::container::sha256(self.bitcode),
            entry: self.entry,
            container: None,
            consts: &self.constants,
            linked: &self.linked,
            dynamic: super::DynamicOrder::default(),
            inputs,
            shared: super::Shared::Nothing,
        }
    }
}

/// A function linked alone ([`DeviceProfile::link_standalone`]): what a validator, a SPIR-V
/// compiler or a disassembler reads.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StandaloneModule {
    /// The relocated module: every link-assigned literal numbered as link numbers it for the
    /// function alone.
    pub words: Vec<u32>,
    /// The layout link gave the function alone.
    pub layout: PipelineLayout,
}

/// What a compile would do before emitting ([`DeviceProfile::plan_entry`]): the plan reported,
/// not handed out.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EntryPlan {
    /// The stage the entry compiles as.
    pub stage: Stage,
    /// Each lowering strategy the plan selected: its uses, and the words it is estimated to emit.
    pub strategies: BTreeMap<String, StrategyUse>,
    /// The size check's estimate of the module's words: a compile refuses above
    /// `budget::SPIRV_WORDS` before it emits.
    pub estimated_words: u64,
}

impl DeviceProfile {
    /// Compile entry `d` under `inputs` for this profile, and what the compile cost against
    /// `budget`, whether it compiled or not: `Device::compile_function_with_report` of a bare
    /// module's entry. The result is as cacheable, and a pipeline takes it as
    /// `StageSource::Compiled`.
    pub fn compile_entry(
        &self,
        d: &EntryDescriptor<'_>,
        inputs: &LoweringInputs,
        budget: &Budget,
    ) -> (Result<CompiledFunction, Error>, CompileReport) {
        let mut m = Meter::new(budget);
        let compiled = guarded(|| {
            compile_one(
                &d.request(InputsFrom::Given(inputs)),
                self,
                &mut m,
                SPIRV_WORDS,
            )
            .map(CompiledFunction::new)
        });
        (compiled, CompileReport::of(&m))
    }

    /// `f` linked alone and relocated: a kernel's module is its compute pipeline's; a vertex or
    /// fragment function's is a valid module with no partner stage, though no Vulkan pipeline on
    /// its own. A function compiled for another profile is
    /// `InvalidDescriptor(ForeignCompiledFunction)`.
    pub fn link_standalone(&self, f: &CompiledFunction) -> Result<StandaloneModule, Error> {
        let artifact = f.artifact();
        if artifact.profile != self.fingerprint() {
            return Err(Error::InvalidDescriptor(
                DescriptorError::ForeignCompiledFunction,
            ));
        }
        guarded(|| {
            let linked = link(&[artifact], &LinkInputs::standalone(), self)?;
            let words = relocate(artifact, &linked.table)?;
            Ok(StandaloneModule {
                words,
                layout: linked.layout,
            })
        })
    }

    /// The lowering inputs a render pipeline's vertex stage compiles vertex function `d` under
    /// with vertex descriptor `vertex`, the empty variant and Metal's default call depth: the
    /// vertex-fetch projection over the attributes the specialized entry reads, or a
    /// post-tessellation function's patch fetches. It decodes and specializes the entry, and lowers
    /// nothing.
    pub fn vertex_entry_inputs(
        &self,
        d: &EntryDescriptor<'_>,
        vertex: Option<&VertexDescriptor>,
        budget: &Budget,
    ) -> Result<LoweringInputs, Error> {
        let mut m = Meter::new(budget);
        let variant = VariantKey::new();
        guarded(|| {
            let r = d.request(InputsFrom::Vertex(vertex, &variant, 1, false, false));
            let spec = specialize_one(&r, &mut m)?.spec;
            lower_inputs(&spec, r.inputs, self, &mut m)
        })
    }

    /// Everything [`DeviceProfile::compile_entry`] does before emission (decode, specialize,
    /// analyze, select), and what it cost against `budget`: its first plan, which a kernel whose
    /// emitted module passes the bound every stage of a profile `Device::from_queries` built is
    /// held under replaces (`device::profile::StageBound`).
    pub fn plan_entry(
        &self,
        d: &EntryDescriptor<'_>,
        inputs: &LoweringInputs,
        budget: &Budget,
    ) -> (Result<EntryPlan, Error>, CompileReport) {
        let mut m = Meter::new(budget);
        let planned = guarded(|| {
            let r = d.request(InputsFrom::Given(inputs));
            let widest = crate::lower::simd::WidestPin::Weighed;
            plan_one(&r, self, widest, &mut m).map(|planned| EntryPlan {
                stage: planned.plan.stage,
                strategies: census(&planned.plan).entries,
                estimated_words: planned.plan.estimate.words,
            })
        });
        (planned, CompileReport::of(&m))
    }

    /// Entry `d` as this profile's compile under `inputs` specializes it: its constants, its
    /// linked functions, and the order Apple's simdgroup lockstep gives its threadgroup
    /// accesses, made explicit for this profile. It is the compile's own entry, built by the same
    /// code, so a refusal's location (function, block, instruction) names an instruction of it.
    pub fn specialized_entry(
        &self,
        d: &EntryDescriptor<'_>,
        inputs: &LoweringInputs,
        budget: &Budget,
    ) -> Result<Entry, Error> {
        let mut m = Meter::new(budget);
        guarded(|| {
            ordered_entry(&d.request(InputsFrom::Given(inputs)), self, &mut m)
                .map(|specialized| specialized.spec)
        })
    }
}
