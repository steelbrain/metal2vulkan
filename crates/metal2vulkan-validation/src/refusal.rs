//! The engine's typed refusals as committed evidence.
//!
//! A case the engine executor refuses writes no row. When it refused the case after opening
//! the device (the product's refusal at compile, the binder's, a pipeline requirement the device
//! did not enable, a containment form the device leaves open), the executor stores a
//! [`RefusalObservation`] in the slot the row would have taken: the case, the build, the opened
//! device's facts (the `environment` a row records, its identity and profile among them), and the
//! typed [`Refusal`]. A slot holds a row or a refusal, never both: the store's write of either
//! removes the other. A refusal before any device opened (a tooling requirement, a case the
//! executor cannot map) is not a fact about a device and is not stored: the case was not run.
//!
//! C3 reads these records: a current refusal is `refused`, not `missing`, and a device gap C1's
//! reviewed list holds for the venue's profile is `device_gap` (`c3`). A refusal is current by the
//! rows' rule: the build (the executor ABI and the translator fingerprint of the recorded
//! profile), the case's inputs, and the venue's declared device and driver.

use crate::c1::reviewed::GapClass;
use crate::candidate::engine::artifact::ExecutedArtifact;
use crate::check::admission::NotSupported;
use crate::observation::{validate_hash, Backend, CaseTiming};
use serde::{Deserialize, Serialize};

/// What a refusal says is missing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RefusalClass {
    /// The device lacks what the case needs: a capability the product's translation needs
    /// (`device_capability_missing`), a feature the pipeline requires that the device did not
    /// enable, a containment form the device leaves open, or a limit or address the binder needs.
    Device,
    /// The product does not translate the case on any device: a feature not implemented or not
    /// representable, an input it reads as malformed, a structural limit.
    Product,
    /// This executor does not run the case: a family it does not implement.
    Executor,
}

impl RefusalClass {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Device => "device",
            Self::Product => "product",
            Self::Executor => "executor",
        }
    }

    /// The class of the product's typed outcome `label` and `key` (`engine::Outcome`): a device
    /// gap exactly when C1's vocabulary calls it one ([`GapClass::DeviceCapability`]).
    pub fn of_product(label: &str, key: &str) -> Self {
        match GapClass::of(label, key) {
            Some(GapClass::DeviceCapability) => Self::Device,
            _ => Self::Product,
        }
    }
}

/// The product's typed outcome of a product refusal (`engine::Outcome::label` and `key`): the key
/// C1's reviewed list holds device gaps by.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProductOutcome {
    pub label: String,
    pub key: String,
    /// Whether the binder refused what the case binds (its resources, its dispatch) rather than
    /// its translation or the device's description: `None` in a record of a build before the
    /// executor recorded it, whose phase is unknown.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub at_bind: Option<bool>,
}

/// One typed refusal.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Refusal {
    pub class: RefusalClass,
    /// The stable label the executor prints (`not_supported{...}`, [`NotSupported::outcome`]).
    pub outcome: String,
    /// The product's outcome, for a refusal by the product.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub product: Option<ProductOutcome>,
    /// The refusal as the executor reports it.
    pub detail: String,
}

impl Refusal {
    /// The refusal `not_supported` is.
    pub fn of(not_supported: &NotSupported) -> Refusal {
        let (class, product) = match not_supported {
            NotSupported::Product {
                outcome, at_bind, ..
            } => {
                let (label, key) = (outcome.label(), outcome.key());
                (
                    RefusalClass::of_product(label, &key),
                    Some(ProductOutcome {
                        label: label.into(),
                        key,
                        at_bind: Some(*at_bind),
                    }),
                )
            }
            NotSupported::Bind(_)
            | NotSupported::NotEnabled(_)
            | NotSupported::Uncontained { .. }
            | NotSupported::Restriction { .. } => (RefusalClass::Device, None),
            NotSupported::NotImplemented(_) | NotSupported::Requirements(_) => {
                (RefusalClass::Executor, None)
            }
        };
        Refusal {
            class,
            outcome: not_supported.outcome(),
            product,
            detail: not_supported.to_string(),
        }
    }

    /// The class agrees with the product's outcome (a product refusal names one, no other does),
    /// and the label and the detail are present.
    fn validate(&self) -> Result<(), String> {
        if self.outcome.trim().is_empty() || self.detail.trim().is_empty() {
            return Err("a refusal needs its outcome label and its detail".into());
        }
        match (&self.product, self.class) {
            (Some(product), class)
                if class != RefusalClass::of_product(&product.label, &product.key) =>
            {
                Err(format!(
                    "a refusal of class {} names the product outcome {} {}, which is {}",
                    class.as_str(),
                    product.label,
                    product.key,
                    RefusalClass::of_product(&product.label, &product.key).as_str()
                ))
            }
            (None, RefusalClass::Product) => {
                Err("a product refusal needs the product's outcome".into())
            }
            _ => Ok(()),
        }
    }
}

/// A refusal of one case in one engine slot, after the device opened.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RefusalObservation {
    pub case_id: String,
    pub air_sha256: String,
    pub input_sha256: String,
    /// The engine's fingerprint of the opened device's profile under this product, as a row
    /// carries it.
    pub translator_fingerprint: String,
    pub backend: Backend,
    /// The engine slot (`ENV@engine`, `ENV@engine@P` restricted to profile `P`).
    pub environment_id: String,
    /// The opened device's facts, as a row records them: the device and driver, the layers, the
    /// robustness features, the profile and its fingerprint.
    pub environment: serde_json::Value,
    pub executor_abi: String,
    pub refusal: Refusal,
    /// What the run had compiled when it refused, by content, and under which product runtime, as a
    /// row records it ([`crate::coverage::ExecutedCensus::artifact`]). `None` for a refusal of an
    /// executor before ABI v129.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifact: Option<ExecutedArtifact>,
    /// What the refused attempt cost in time ([`CaseTiming`]): the attempt prepared the case and
    /// compiled before it refused, which is a case the suite ran, so the row records what that
    /// cost. The figure is the device worker's reading around the attempt, as a run's is. Every
    /// refusal this build records carries one ([`crate::store::CorpusStore::upsert_refusal`]
    /// refuses one that does not); a refusal recorded before the figure existed holds none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timing: Option<CaseTiming>,
}

impl RefusalObservation {
    /// Well-formed hashes, an engine slot and an executor ABI, an environment object, and a
    /// well-formed refusal.
    pub fn validate_content(&self) -> Result<(), String> {
        for (field, value) in [
            ("case_id", &self.case_id),
            ("air_sha256", &self.air_sha256),
            ("input_sha256", &self.input_sha256),
            ("translator_fingerprint", &self.translator_fingerprint),
        ] {
            validate_hash(field, value)?;
        }
        if !crate::candidate::engine::is_engine_slot(&self.environment_id) {
            return Err(format!(
                "refusal slot {:?} is not an engine slot",
                self.environment_id
            ));
        }
        if self.executor_abi.trim().is_empty() {
            return Err("a refusal needs its executor ABI".into());
        }
        if !self.environment.is_object() {
            return Err("refusal environment must be a JSON object".into());
        }
        if let Some(artifact) = &self.artifact {
            artifact.validate()?;
        }
        self.refusal.validate()
    }
}

#[cfg(test)]
mod tests;
