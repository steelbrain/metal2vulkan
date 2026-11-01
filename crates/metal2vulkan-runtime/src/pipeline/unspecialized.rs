//! Metal's refusal of a pipeline state of a stage function made by `newFunctionWithName:` whose
//! function declares function constants.
//!
//! - **What Metal answers** (an Apple M3 Max under macOS 27.0.1, `MTL_DEBUG_LAYER` unset; the
//!   conformance crate's probe `undefined_function_constant_values.m`). A function made by
//!   `newFunctionWithName:` whose `functionConstantsDictionary` is not empty ends the process at
//!   every pipeline creator's descriptor validation, whether its constant is required, or optional
//!   and read only where `is_function_constant_defined` says, or only gates an argument: a compute
//!   state's ("Compute Pipeline Descriptor Validation / function ko cannot be used to build a
//!   pipeline state. Use newFunctionWithName:constantValues:... to get the specialized
//!   function"), a render state's vertex and fragment functions' (`vertexFunction`,
//!   `fragmentFunction`), a tile state's ("function tc cannot be used to build a pipeline state
//!   without setting constant values first.") and a mesh render state's object and mesh
//!   functions' (`objectFunction`, `meshFunction`). One made by name that declares none is made,
//!   and so is every function made with constant values, none among them too, or by
//!   `newFunctionWithDescriptor:` without them, its constants without a value undefined.
//! - **The answer.** [`Error::Invalid`] with [`Problem::UnspecializedFunction`], before anything is
//!   compiled. Whether a stage is specialized is its descriptor's
//!   (`FunctionDescriptor::specialized`, which [`Function::function_descriptor`] sets; one with
//!   constant values is specialized whatever it says), and whether it declares constants is the
//!   compile layer's answer (`FunctionDescriptor::needs_specialization`), read through the
//!   device's stage entries, whose module decode the stage's compile then copies. What that answer
//!   refuses refuses the creation. A linked function keeps its own rule
//!   (`LinkProblem::UnresolvedConstant`, `Function::linked_function`).
//!
//! [`Function::function_descriptor`]: crate::Function::function_descriptor

use super::Device;
use crate::device::device_error;
use crate::error::{invalid, Error, Problem};
use metal2vulkan::pipeline::{Stage, StageSource};
use metal2vulkan::Budget;

/// Refused in `call` (the module docs): `source`, the function of `stage` (`Kernel` for a compute
/// or tile state's), where it is not specialized and declares function constants. A stage the
/// compile layer already compiled is no function, and is not refused here.
pub(super) fn check(
    device: &Device,
    call: &'static str,
    stage: Stage,
    source: &StageSource,
    budget: &Budget,
) -> Result<(), Error> {
    let StageSource::Function(f) = source else {
        return Ok(());
    };
    let entries = device.shared().shader_cache.memo.stage_entries();
    match f.needs_specialization(budget, entries) {
        Ok(false) => Ok(()),
        Ok(true) => Err(invalid(
            call,
            Problem::UnspecializedFunction {
                stage,
                name: f.function.info().name.to_owned(),
            },
        )),
        Err(e) => Err(device_error(e)),
    }
}
