//! A mesh render pipeline's link check: its mesh function's payload holds its object function's.

use crate::error::{Error, RuntimeInput};
use crate::pipeline::interface::StageInterface;
use crate::pipeline::sym::Stage;

/// The payload check of the mesh render pipeline of stages `ifaces`: its mesh function's payload
/// arena holds at least its object function's (`StageInterface::payload`,
/// `LoweringInputs::object_payload`), else `NeedsInput(ObjectPayload)`.
pub fn payload(ifaces: &[&StageInterface]) -> Result<(), Error> {
    let of = |stage: Stage| {
        ifaces
            .iter()
            .find(|i| i.stage == stage)
            .and_then(|i| i.payload)
    };
    match (of(Stage::Object), of(Stage::Mesh)) {
        (Some(object), Some(mesh)) if mesh < object => {
            Err(Error::NeedsInput(RuntimeInput::ObjectPayload))
        }
        _ => Ok(()),
    }
}
