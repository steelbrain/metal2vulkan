//! What a need of a state's variant reports (`pipeline::Lowered::kernel`), by what the
//! variant's lowering gave, with no device: the translator's refusal as the variant's own gap
//! (`Gap::PartialSimdgroups` for the emulated-SIMD-group variant), and every other failure as
//! itself, so a translator bug, a limit or an input the compute encoder does not supply never
//! reads as that gap.

use crate::pipeline::Lowered;
use crate::{Error, Gap};
use metal2vulkan::error::{Error as TranslationError, InternalError, InternalKind};

/// The emulated variant's gap, as the compute encoder's need of it states it.
fn partial_simdgroups() -> Error {
    Error::Unsupported(Gap::PartialSimdgroups {
        threads_per_threadgroup: [30, 1, 1],
        width: 32,
    })
}

/// What a need of the variant reports, in words.
fn reported(lowered: &Lowered) -> String {
    match lowered.kernel(partial_simdgroups) {
        Ok(_) => "a module".to_owned(),
        Err(error) => format!("{error:?}"),
    }
}

/// The builder's rejection of the variant, a translator bug, reaches the need as the translator's
/// own error, and the variant's read of an input the encoder does not supply as that gap; only
/// the translator's refusal is the partial-SIMD-group gap.
#[test]
fn a_lowering_error_is_not_reported_as_the_variant_s_gap() {
    let internal = TranslationError::Internal(InternalError::new(
        InternalKind::Build,
        "a constant of a type the builder takes no null of",
    ));
    assert!(
        matches!(
            Lowered::Translation(internal.clone()).kernel(partial_simdgroups),
            Err(Error::Translation(ref error)) if *error == internal
        ),
        "a builder error: {}",
        reported(&Lowered::Translation(internal))
    );
    assert!(
        matches!(
            Lowered::Unsupplied.kernel(partial_simdgroups),
            Err(Error::Unsupported(Gap::KernelInput))
        ),
        "an input the encoder does not supply: {}",
        reported(&Lowered::Unsupplied)
    );
    assert!(
        matches!(
            Lowered::Refused.kernel(partial_simdgroups),
            Err(Error::Unsupported(Gap::PartialSimdgroups { .. }))
        ),
        "the translator's refusal: {}",
        reported(&Lowered::Refused)
    );
}
