//! What an engine run executed, by content.
//!
//! The pipeline a run compiles before it reads device output (the case's own, the compile
//! [`super::case_compile`] plans) contributes one compile digest: its subject (`case`), the
//! pipeline's content digest (`ComputePipeline::content_digest`, every field but the cache
//! identity), the content digest of the binding plan the run makes of it for its bind shape
//! (`BindingPlan::content_digest`, or the refusal of that plan). The case's artifact is the
//! SHA-256 over the compile digests, by subject.
//!
//! The executor records it ([`ExecutedArtifact`]) with the product's
//! `cache::RUNTIME_VERSION`; the authored compile census computes it the same way, device-free,
//! for each committed profile; C3's currency compares the two.

use std::collections::BTreeMap;

use metal2vulkan::bind::{BindShape, BindingPlan};
use metal2vulkan::cache::Hash256;
use metal2vulkan::device::Device;
use metal2vulkan::error::BindError;
use metal2vulkan::pipeline::{ComputePipeline, RenderPipeline};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::check::admission::ExecutorError;

/// What a run executed, by content: recorded in an engine row's census and in a refusal the run
/// made after it opened its device.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutedArtifact {
    /// The product's `cache::RUNTIME_VERSION`: what its runtime (instantiation, status decoding)
    /// does with a compiled pipeline.
    pub runtime_version: u32,
    /// Lowercase hex SHA-256 over the compile digests of every pipeline the run compiled before
    /// it read device output ([`CaseArtifact`]); `None` when it compiled none (a refusal at the
    /// first compile).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
}

impl ExecutedArtifact {
    /// A well-formed digest, when there is one.
    pub fn validate(&self) -> Result<(), String> {
        match &self.sha256 {
            Some(sha256) => crate::observation::validate_hash("artifact.sha256", sha256),
            None => Ok(()),
        }
    }

    /// Whether this build would execute what the run did: the run recorded a digest, under this
    /// product's `cache::RUNTIME_VERSION`, equal to `current`, this build's digest of the case on
    /// the run's profile (`authored_compile::CurrentArtifacts`).
    pub fn is_current(&self, current: Option<&str>) -> bool {
        self.runtime_version == metal2vulkan::cache::RUNTIME_VERSION
            && self.sha256.is_some()
            && self.sha256.as_deref() == current
    }
}

/// A pipeline whose content and binding plan a compile digest covers.
pub trait Planned {
    /// The pipeline's content digest.
    fn content_digest(&self) -> Hash256;
    /// The pipeline's binding plan for `shape` on `device`.
    fn plan_for(&self, device: &Device, shape: &BindShape) -> Result<BindingPlan, BindError>;
}

impl Planned for ComputePipeline {
    fn content_digest(&self) -> Hash256 {
        ComputePipeline::content_digest(self)
    }
    fn plan_for(&self, device: &Device, shape: &BindShape) -> Result<BindingPlan, BindError> {
        self.plan(device, shape)
    }
}

impl Planned for RenderPipeline {
    fn content_digest(&self) -> Hash256 {
        RenderPipeline::content_digest(self)
    }
    fn plan_for(&self, device: &Device, shape: &BindShape) -> Result<BindingPlan, BindError> {
        self.plan(device, shape)
    }
}

/// The subject of the case's own compile, as the artifact names it.
const CASE: &str = "case";

/// The digests of what one run compiled before it read device output, by subject.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CaseArtifact {
    parts: BTreeMap<String, Hash256>,
}

impl CaseArtifact {
    /// Record the case's compile: `pipeline` planned for `shape` on `device`. A plan the binder
    /// refuses is part of the digest as its refusal. Compiled twice, it must be the same compile.
    pub fn record(
        &mut self,
        pipeline: &dyn Planned,
        device: &Device,
        shape: &BindShape,
    ) -> Result<(), ExecutorError> {
        let label = CASE.to_owned();
        let mut h = Sha256::new();
        h.update(b"metal2vulkan C3 compile digest v1\0");
        put(&mut h, label.as_bytes());
        h.update(pipeline.content_digest());
        match pipeline.plan_for(device, shape) {
            Ok(plan) => {
                h.update([1]);
                h.update(plan.content_digest().map_err(|error| {
                    ExecutorError::Failed(format!("the binding plan has no digest: {error}"))
                })?);
            }
            Err(refusal) => {
                h.update([0]);
                put(&mut h, format!("{refusal:?}").as_bytes());
            }
        }
        self.insert(label, h.finalize().into())
    }

    fn insert(&mut self, label: String, digest: Hash256) -> Result<(), ExecutorError> {
        match self.parts.insert(label.clone(), digest) {
            Some(earlier) if earlier != digest => Err(ExecutorError::Failed(format!(
                "the run compiled {label} twice, to different artifacts"
            ))),
            _ => Ok(()),
        }
    }

    /// The case's artifact digest, lowercase hex: `None` when nothing was recorded.
    pub fn sha256(&self) -> Option<String> {
        if self.parts.is_empty() {
            return None;
        }
        let mut h = Sha256::new();
        h.update(b"metal2vulkan C3 case artifact v1\0");
        h.update((self.parts.len() as u64).to_le_bytes());
        for (label, digest) in &self.parts {
            put(&mut h, label.as_bytes());
            h.update(digest);
        }
        Some(
            h.finalize()
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect(),
        )
    }

    /// What a run with this artifact executed under this product's runtime.
    pub fn executed(&self) -> ExecutedArtifact {
        ExecutedArtifact {
            runtime_version: metal2vulkan::cache::RUNTIME_VERSION,
            sha256: self.sha256(),
        }
    }
}

/// A length-prefixed byte string.
fn put(h: &mut Sha256, bytes: &[u8]) {
    h.update((bytes.len() as u64).to_le_bytes());
    h.update(bytes);
}

#[cfg(test)]
mod tests;
