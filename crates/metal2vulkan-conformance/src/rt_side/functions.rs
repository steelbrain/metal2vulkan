//! The rt side's function and library facts: what an `MTLFunction` and an `MTLLibrary` answer of
//! themselves.
//!
//! The runtime's library is the translator's ([`rt::Library`]), and a function is the
//! translator's function of it, so this side answers each getter as a consumer holding them reads
//! it: the library's functions and type, and a function's reflected properties and constants
//! (`Function::properties`, or `Function::specialized_properties` for a function made with
//! constant values, whose dictionary the translator states is empty, as Metal answers it). The
//! runtime keeps no device on either, so the device relation is a refused field (the consumer's
//! own), and a function keeps no `MTLFunctionOptions`, so its options are a refused field too.

use super::RtSide;
use crate::case::{Answer, Failure, Field, Handle};
use crate::function_facts::{
    attribute, constant_values, ConstantFacts, ConstantValue, FunctionFacts,
};
use crate::library_facts::{FunctionDescriptorSpec, LibraryFacts};
use metal2vulkan::library::FunctionStage;
use metal2vulkan::metal::FunctionOptions;
use metal2vulkan::Budget;
use metal2vulkan_rt as rt;
use std::collections::BTreeMap;

/// `device`: neither the runtime's library nor a function of it keeps a device.
fn device() -> Field<bool> {
    Err(Failure::unavailable("device"))
}

/// `options`: a function of the runtime's library keeps no `MTLFunctionOptions`.
fn options() -> Field<FunctionOptions> {
    Err(Failure::unavailable("options"))
}

impl RtSide<'_> {
    /// [`crate::case::Side::library_facts`]: the library's functions a program makes by name (a
    /// Core Image kernel library's extern functions, which the translator lists with their own
    /// stage, are not among them, as Metal's `functionNames` leaves them out), its type and its
    /// install name.
    pub(super) fn library_facts_of(&self, library: Handle) -> Answer<LibraryFacts> {
        let held: &rt::Library = self.held_library(library)?;
        Ok(LibraryFacts {
            function_names: (held.functions())
                .filter(|info| info.stage != FunctionStage::Extern)
                .map(|info| info.name.to_owned())
                .collect(),
            library_type: held.library_type(),
            install_name: held.install_name().map(str::to_owned),
            device_is_the_cases: device(),
        })
    }

    /// [`crate::case::Side::function_facts`]: the library's `entry`, made with `constants` where
    /// the case states any.
    pub(super) fn function_facts_of(
        &self,
        library: Handle,
        entry: &str,
        constants: &[ConstantValue],
    ) -> Answer<FunctionFacts> {
        let function = (self.held_library(library)?.function(entry))
            .map_err(|error| self.on.translation_failure(&error))?;
        let budget = Budget::standard();
        let (properties, dictionary) = if constants.is_empty() {
            (
                function.properties(&budget),
                function.function_constants_dictionary(),
            )
        } else {
            (
                function.specialized_properties(&constant_values(constants), &budget),
                Ok(BTreeMap::new()),
            )
        };
        let properties = properties.map_err(|error| self.on.translation_failure(&error))?;
        let dictionary = dictionary.map_err(|error| self.on.translation_failure(&error))?;
        Ok(FunctionFacts {
            name: function.info().name.to_owned(),
            function_type: properties.function_type(),
            patch_type: properties.patch_type(),
            patch_control_point_count: properties.patch_control_point_count(),
            vertex_attributes: (properties.vertex_attributes())
                .map(|held| held.iter().map(attribute).collect()),
            stage_input_attributes: (properties.stage_input_attributes())
                .map(|held| held.iter().map(attribute).collect()),
            // A `BTreeMap`, so the names are in the order the Metal side sorts its own into.
            constants: (dictionary.iter())
                .map(|(name, constant)| ConstantFacts {
                    name: name.clone(),
                    data_type: constant.data_type(),
                    index: constant.index(),
                    required: constant.is_required(),
                })
                .collect(),
            options: options(),
            device_is_the_cases: device(),
        })
    }

    /// [`crate::case::Side::function_facts_from_descriptor`]: the function the descriptor names,
    /// made with its constant values, as [`RtSide::function_facts_of`] makes one by name. The
    /// descriptor's options are kept nowhere, so they reach no answer.
    pub(super) fn described_function_facts(
        &self,
        library: Handle,
        descriptor: &FunctionDescriptorSpec,
    ) -> Answer<FunctionFacts> {
        self.function_facts_of(library, &descriptor.name, &descriptor.constants)
    }
}
