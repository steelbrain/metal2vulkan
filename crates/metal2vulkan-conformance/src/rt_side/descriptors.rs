//! The rt side's pipeline descriptor families: a render and a compute pipeline descriptor read
//! back before any state is made.
//!
//! The runtime takes both descriptors as the translator's ([`rt::RenderPipelineDescriptor`],
//! [`rt::ComputePipelineDescriptor`]), so this side states and reads them through the walk both
//! sides share ([`crate::descriptor_reads`]), each default as the translator holds it. The
//! runtime keeps no label and takes no binary archive, so those two getters answer refused
//! fields.

use super::RtSide;
use crate::case::{Answer, Failure, Field, Handle};
use crate::compute_descriptors::{ComputeDescriptorFacts, DescriptorSpec as ComputeDescriptorSpec};
use crate::descriptor_reads;
use crate::render_descriptors::{DescriptorSpec, RenderDescriptorFacts};
use metal2vulkan_runtime as rt;

/// `label`: the runtime keeps none, on a descriptor or on what it makes of one.
fn label() -> Field<Option<String>> {
    Err(Failure::unavailable("label"))
}

/// `binaryArchives`: the runtime takes no binary archive.
fn binary_archives() -> Field<u64> {
    Err(Failure::unavailable("binaryArchives"))
}

impl RtSide<'_> {
    /// [`crate::case::Side::render_descriptor_facts`]: the translator's descriptor of the
    /// library's stages, stated and reset as the spec's shape says, then read.
    ///
    /// The runtime has no mutable descriptor object, so Metal's `init` is
    /// `RenderPipelineDescriptor::new` and its `reset` is the same call again, after the stating
    /// ran.
    pub(super) fn render_descriptor(
        &self,
        library: Handle,
        spec: &DescriptorSpec<'_>,
    ) -> Answer<RenderDescriptorFacts> {
        let fresh = || -> Answer<rt::RenderPipelineDescriptor> {
            let mut descriptor =
                rt::RenderPipelineDescriptor::new(self.kernel(library, spec.vertex)?);
            if let Some(fragment) = spec.fragment {
                descriptor.fragment = Some(self.kernel(library, fragment)?);
            }
            Ok(descriptor)
        };
        let mut descriptor = fresh()?;
        if spec.shape.sets() {
            descriptor_reads::state_render(&mut descriptor);
        }
        if spec.shape.resets() {
            descriptor = fresh()?;
        }
        Ok(descriptor_reads::read_render(
            &descriptor,
            label(),
            binary_archives(),
            !spec.shape.resets(),
        ))
    }

    /// [`crate::case::Side::compute_descriptor_facts`], as [`RtSide::render_descriptor`] reads a
    /// render one.
    pub(super) fn compute_descriptor(
        &self,
        library: Handle,
        spec: &ComputeDescriptorSpec<'_>,
    ) -> Answer<ComputeDescriptorFacts> {
        let fresh = || -> Answer<rt::ComputePipelineDescriptor> {
            Ok(rt::ComputePipelineDescriptor::new(
                self.kernel(library, spec.kernel)?,
            ))
        };
        let mut descriptor = fresh()?;
        if spec.shape.sets() {
            descriptor_reads::state_compute(&mut descriptor);
        }
        if spec.shape.resets() {
            descriptor = fresh()?;
        }
        Ok(descriptor_reads::read_compute(
            &descriptor,
            label(),
            binary_archives(),
            !spec.shape.resets(),
        ))
    }
}
