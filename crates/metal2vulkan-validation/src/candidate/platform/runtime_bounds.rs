//! The out-of-bounds facts of a kernel case's run through the runtime ([`super::runtime`]): how
//! the modules the runtime's state may dispatch contain each of the case's targets, decided before
//! anything is submitted.
//!
//! The runtime keeps its modules to itself, so the run lowers each variant the translator offers
//! of the case's kernel on the device's own translator (`rt::Device::translator`), under
//! `LoweringInputs::kernel_variant` of the inputs `kernel_inputs_read` gives the descriptor's: the
//! inputs the state lowers each of its modules under, so each is a module a dispatch may run (the
//! runtime's test `every_device_s_kernel_modules_are_the_translator_s_lowerings_of_their_variants`
//! holds it). The run does not decide which of them the dispatch takes. A variant the translator
//! refuses is one no dispatch runs. Each lowered variant's facts give the run's out-of-bounds
//! facts as the executor's own context reads a module's (`engine::out_of_bounds_run`); where two
//! variants give different ones, which module a dispatch runs would decide the case, so it is
//! refused before it runs (`NotSupported::Uncontained`, naming the first variant whose facts
//! differ from the first lowered one's).

use crate::candidate::engine::{out_of_bounds_run, product_refusal, ComputeCase};
use crate::check::admission::{ExecutorError, NotSupported};
use crate::coverage::ExecutionClass;
use crate::executor_contract::out_of_bounds::{self, OutOfBoundsRun};
use metal2vulkan::error::Error as TranslationError;
use metal2vulkan::pipeline::{ComputePipelineDescriptor, KernelVariant, StageEntries, StageSource};
use metal2vulkan::Budget;
use metal2vulkan_runtime as rt;

/// The out-of-bounds facts of `case`'s run of `descriptor` on `device`, executing as `class`, as
/// the module documentation says: `None` for a case without targets.
///
/// Err: the translator's refusal of every variant, as the product's (`product_refusal`), the
/// first one's named; its other errors the same way; `NotSupported::Uncontained` where the
/// variants disagree or `class` leaves a form uncontained; a target no variant's module reaches
/// as validation knows fails the run.
pub(super) fn contained(
    device: &rt::Device,
    case: &ComputeCase,
    descriptor: &ComputePipelineDescriptor,
    class: Option<ExecutionClass>,
) -> Result<Option<OutOfBoundsRun>, ExecutorError> {
    if case.out_of_bounds.is_empty() {
        return Ok(None);
    }
    let StageSource::Function(function) = &descriptor.kernel else {
        return Err(ExecutorError::Failed(
            "a run through the runtime makes its kernel from a function".into(),
        ));
    };
    let translator = device.translator();
    let budget = Budget::standard();
    let entries = StageEntries::new();
    let plain = translator
        .kernel_inputs_read(function, &descriptor.lowering_inputs(), &budget, &entries)
        .map_err(|error| product_refusal(&error))?;
    let mut refused = None;
    let mut runs = Vec::new();
    for variant in KernelVariant::ALL {
        let inputs = plain.kernel_variant(variant);
        let lowered =
            match translator.compile_function_from_entries(function, &inputs, &budget, &entries) {
                Ok(lowered) => lowered,
                Err(error @ TranslationError::Unsupported(_)) => {
                    refused.get_or_insert(error);
                    continue;
                }
                Err(error) => return Err(product_refusal(&error)),
            };
        let attributes = out_of_bounds::Fetch::of_function(&lowered);
        let module = out_of_bounds::ModuleFacts {
            stage: lowered.stage(),
            slots: lowered.slots(),
            attributes: &attributes,
            resident_roots: lowered.reads_resident_roots(),
            handle_table: lowered.resolves_handles(),
            strategies: lowered.strategies(),
        };
        let run = out_of_bounds_run(class, translator.profile(), &module, &case.out_of_bounds);
        runs.push((variant, run));
    }
    let Some(((_, first), rest)) = runs.split_first() else {
        return Err(match refused {
            Some(error) => product_refusal(&error),
            None => ExecutorError::Failed("the translator lowered no variant".into()),
        });
    };
    if let Some(&(variant, _)) = rest.iter().find(|(_, run)| run != first) {
        return Err(NotSupported::Uncontained {
            forms: Default::default(),
            variant: Some(variant),
        }
        .into());
    }
    first.clone()
}
