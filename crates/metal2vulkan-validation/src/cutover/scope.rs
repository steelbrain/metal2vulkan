//! What the rewrite's launch leaves out of scope, and how the cutover criteria account for it
//! (the user's decision of 2026-10-03).
//!
//! The user decided that ray tracing is out of scope for the rewrite's launch: the existing
//! ray-tracing lowering stays as it is, no new ray-tracing work is done, and ray tracing is
//! reported apart everywhere, so that it neither hides another gap nor counts as a defect. C1's
//! report and C3's device records account such a row or case in its own bucket, counted per
//! profile and per venue: it does not keep the criterion from holding, and it is not a signed-off
//! device gap either (C1's reviewed list refuses an entry for it).
//!
//! A row or case is out of scope by its **typed refusal**, never by a name or a hash: the product's
//! [`Unsupported`] refusal names a Metal [`Feature`] of the ray-tracing family
//! ([`RAY_TRACING_FEATURES`]: acceleration structures and ray intersection, intersection-function
//! tables, and the ray-tracing lowering family), or every way to translate it needs a ray-tracing
//! device capability ([`ray_tracing_capability`]: ray queries, acceleration structures and their
//! descriptor limits). That covers the device-capability refusals (`RayTracing needs
//! Feature(RayQuery)+...`), the motion refusals (`not_representable RayTracing`), the
//! intersection-function refusals (`not_implemented IntersectionFunctionTable`), and the
//! acceleration-structure queries not implemented yet (`not_implemented RayTracing`). A row a
//! ray-tracing shader refuses for another reason first (a pointer it cannot dereference, say)
//! keeps that refusal's class: its gap is not ray tracing's.
//!
//! One question, three forms of the same typed refusal: the product's [`Error`]
//! ([`OutOfScope::of_error`]), the validation outcome the census and the executors carry
//! ([`OutOfScope::of_outcome`]), and that outcome as recorded, its label and key in an index row,
//! a reviewed list's entry, or a refusal record ([`OutOfScope::of_recorded`]). The recorded forms
//! are compared with the outcome vocabulary's own rendering of the typed values
//! ([`crate::engine::feature_key`], [`crate::engine::capability_key`]), never with a spelling of
//! their own, so all three agree (tested).

use crate::engine::{capabilities_complete, capability_key, feature_key, Outcome};
use metal2vulkan::device::{VkExtension, VkFeature, VkLimit};
use metal2vulkan::error::{Capability, Error, Feature, LowerFamily, Needs, Unsupported};
use serde::{Deserialize, Serialize};

/// A family of Metal the rewrite's launch leaves out of scope.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutOfScope {
    /// Ray tracing (the user's decision of 2026-10-03): ray intersection, acceleration
    /// structures, intersection functions, and their device capabilities.
    RayTracing,
}

/// The Metal features of the ray-tracing family: a refusal that names one is ray tracing's,
/// whatever it needs.
pub const RAY_TRACING_FEATURES: [Feature; 3] = [
    Feature::RayTracing,
    Feature::IntersectionFunctionTable,
    Feature::Lowering(LowerFamily::RayTracing),
];

/// The device features only ray tracing needs.
const RAY_TRACING_DEVICE_FEATURES: [VkFeature; 2] =
    [VkFeature::RayQuery, VkFeature::AccelerationStructure];

/// The device extensions only ray tracing needs (`VK_KHR_deferred_host_operations` is required
/// by `VK_KHR_acceleration_structure`).
const RAY_TRACING_EXTENSIONS: [VkExtension; 3] = [
    VkExtension::KhrRayQuery,
    VkExtension::KhrAccelerationStructure,
    VkExtension::KhrDeferredHostOperations,
];

/// The device limits only ray tracing needs: acceleration-structure descriptors.
const RAY_TRACING_LIMITS: [VkLimit; 2] = [
    VkLimit::MaxPerStageDescriptorAccelerationStructures,
    VkLimit::MaxDescriptorSetAccelerationStructures,
];

/// Whether `capability` is one only ray tracing needs.
pub fn ray_tracing_capability(capability: &Capability) -> bool {
    match capability {
        Capability::Feature(feature) => RAY_TRACING_DEVICE_FEATURES.contains(feature),
        Capability::Extension(extension) => RAY_TRACING_EXTENSIONS.contains(extension),
        Capability::Limit { limit, .. } => RAY_TRACING_LIMITS.contains(limit),
        _ => false,
    }
}

/// Whether a recorded capability token ([`capability_key`]) is one only ray tracing needs: it is
/// the rendering of a ray-tracing feature or extension, or of a ray-tracing limit whatever its
/// counts.
fn ray_tracing_capability_key(token: &str) -> bool {
    RAY_TRACING_DEVICE_FEATURES
        .iter()
        .map(|feature| Capability::Feature(*feature))
        .chain(
            RAY_TRACING_EXTENSIONS
                .iter()
                .map(|extension| Capability::Extension(*extension)),
        )
        .any(|capability| capability_key(&capability) == token)
        || RAY_TRACING_LIMITS.iter().any(|limit| {
            // A limit renders with its counts; the part before them names the limit.
            let rendered = capability_key(&Capability::Limit {
                limit: *limit,
                required: 0,
                available: 0,
            });
            let named = |key: &str| {
                key.split_once(", required:")
                    .map(|(named, _)| named.to_string())
            };
            named(&rendered).is_some_and(|named_limit| named(token) == Some(named_limit))
        })
}

impl OutOfScope {
    pub const ALL: [OutOfScope; 1] = [OutOfScope::RayTracing];

    pub fn as_str(self) -> &'static str {
        match self {
            OutOfScope::RayTracing => "ray_tracing",
        }
    }

    /// The family a typed refusal is out of scope by: a ray-tracing feature, or device
    /// capabilities of which every alternative needs one only ray tracing needs.
    pub fn of_unsupported(unsupported: &Unsupported) -> Option<OutOfScope> {
        if RAY_TRACING_FEATURES.contains(&unsupported.feature) {
            return Some(OutOfScope::RayTracing);
        }
        match &unsupported.needs {
            Needs::AnyOf(alternatives)
                if !alternatives.is_empty()
                    && alternatives
                        .iter()
                        .all(|all| all.iter().any(ray_tracing_capability)) =>
            {
                Some(OutOfScope::RayTracing)
            }
            _ => None,
        }
    }

    /// [`OutOfScope::of_unsupported`] of a product error; `None` for every error that is not a
    /// typed refusal.
    pub fn of_error(error: &Error) -> Option<OutOfScope> {
        match error {
            Error::Unsupported(unsupported) => OutOfScope::of_unsupported(unsupported),
            _ => None,
        }
    }

    /// The family a validation outcome is out of scope by: [`OutOfScope::of_unsupported`] of the
    /// refusal it records. A device-capability key that may have lost alternatives at its end
    /// ([`capabilities_complete`]) is out of scope only by its feature.
    pub fn of_outcome(outcome: &Outcome) -> Option<OutOfScope> {
        let ray_tracing_feature = |feature: &str| {
            RAY_TRACING_FEATURES
                .iter()
                .any(|family| feature_key(family) == feature)
        };
        let in_family = match outcome {
            Outcome::Unsupported { feature, .. } => ray_tracing_feature(feature),
            Outcome::DeviceCapabilityMissing {
                feature,
                capability,
            } => {
                ray_tracing_feature(feature)
                    || (capabilities_complete(capability)
                        && !capability.is_empty()
                        && capability.split(" | ").all(|alternative| {
                            alternative.split('+').any(ray_tracing_capability_key)
                        }))
            }
            _ => false,
        };
        in_family.then_some(OutOfScope::RayTracing)
    }

    /// [`OutOfScope::of_outcome`] of an outcome recorded as its label and key
    /// ([`Outcome::recorded_refusal`]): a census row, a C1 gap bucket or list entry, a refusal
    /// record's product outcome.
    pub fn of_recorded(label: &str, key: &str) -> Option<OutOfScope> {
        Outcome::recorded_refusal(label, key).and_then(|outcome| OutOfScope::of_outcome(&outcome))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use metal2vulkan::device::VkProperty;
    use metal2vulkan::error::{DiagName, RuntimeInput};

    fn recorded(error: &Error) -> Option<OutOfScope> {
        let outcome = Outcome::from(error);
        OutOfScope::of_recorded(outcome.label(), &outcome.key())
    }

    /// The refusals the census records for ray tracing today, each in the three forms, are out of
    /// scope; the refusals beside them are not.
    #[test]
    fn ray_tracing_refusals_are_out_of_scope_in_every_form() {
        let ray_query = Capability::Feature(VkFeature::RayQuery);
        let structure = Capability::Feature(VkFeature::AccelerationStructure);
        let device_address = Capability::Feature(VkFeature::BufferDeviceAddress);
        let out_of_scope: Vec<Error> = vec![
            // moltenvk-m3's device-capability refusal, and vk12-floor's with device addresses.
            Unsupported::needs_any_of(
                Feature::RayTracing,
                vec![vec![ray_query.clone(), structure.clone()]],
            )
            .into(),
            Unsupported::needs_any_of(
                Feature::RayTracing,
                vec![vec![
                    ray_query.clone(),
                    structure.clone(),
                    device_address.clone(),
                ]],
            )
            .into(),
            // Motion, intersection functions, instance-structure queries.
            Unsupported::not_representable(Feature::RayTracing).into(),
            Unsupported::not_implemented(Feature::IntersectionFunctionTable).into(),
            Unsupported::not_implemented(Feature::RayTracing)
                .naming(DiagName::new("air.intersect.triangle_data"))
                .into(),
            Unsupported::not_implemented(Feature::Lowering(LowerFamily::RayTracing)).into(),
            // A refusal whose every alternative needs a ray-tracing capability.
            Unsupported::needs(
                Feature::DescriptorCount,
                Capability::Limit {
                    limit: VkLimit::MaxPerStageDescriptorAccelerationStructures,
                    required: 20,
                    available: 16,
                },
            )
            .into(),
            Unsupported::needs_any_of(
                Feature::DescriptorCount,
                vec![
                    vec![Capability::Extension(VkExtension::KhrRayQuery)],
                    vec![structure.clone(), device_address.clone()],
                ],
            )
            .into(),
        ];
        for error in &out_of_scope {
            assert_eq!(
                OutOfScope::of_error(error),
                Some(OutOfScope::RayTracing),
                "{error}"
            );
            assert_eq!(recorded(error), Some(OutOfScope::RayTracing), "{error}");
        }
        let in_scope: Vec<Error> = vec![
            Unsupported::not_implemented(Feature::ArgumentBufferObjects).into(),
            Unsupported::not_representable(Feature::DereferenceLoadedPointer).into(),
            Unsupported::not_implemented(Feature::TileShading).into(),
            Unsupported::not_implemented(Feature::Lowering(LowerFamily::Sampler)).into(),
            Unsupported::needs(
                Feature::DereferenceLoadedPointer,
                Capability::Feature(VkFeature::BufferDeviceAddress),
            )
            .into(),
            Unsupported::needs(
                Feature::DescriptorCount,
                Capability::Limit {
                    limit: VkLimit::MaxPerStageDescriptorSamplers,
                    required: 32,
                    available: 16,
                },
            )
            .into(),
            Unsupported::needs(
                Feature::SubgroupOperations,
                Capability::Property(VkProperty::SubgroupQuadOperationsInAllStages),
            )
            .into(),
            // One alternative without ray tracing: a device without ray queries can translate it.
            Unsupported::needs_any_of(
                Feature::DescriptorCount,
                vec![vec![ray_query.clone()], vec![device_address.clone()]],
            )
            .into(),
            Error::NeedsInput(RuntimeInput::VertexDescriptor { attribute: 0 }),
            Error::Cancelled,
        ];
        for error in &in_scope {
            assert_eq!(OutOfScope::of_error(error), None, "{error}");
            assert_eq!(recorded(error), None, "{error}");
        }
    }

    /// The recorded forms the sweeps print are read as the typed refusals they render.
    #[test]
    fn recorded_census_keys_are_read_by_their_typed_rendering() {
        for (label, key, scope) in [
            (
                "device_capability_missing",
                "RayTracing needs Feature(RayQuery)+Feature(AccelerationStructure)",
                Some(OutOfScope::RayTracing),
            ),
            (
                "unsupported",
                "not_representable RayTracing",
                Some(OutOfScope::RayTracing),
            ),
            (
                "unsupported",
                "not_implemented IntersectionFunctionTable",
                Some(OutOfScope::RayTracing),
            ),
            (
                "device_capability_missing",
                "DescriptorCount needs Limit { limit: MaxPerStageDescriptorAccelerationStructures, required: 20, available: 16 }",
                Some(OutOfScope::RayTracing),
            ),
            (
                "device_capability_missing",
                "DescriptorCount needs Limit { limit: MaxPerStageDescriptorSamplers, required: 32, available: 16 }",
                None,
            ),
            ("unsupported", "not_implemented ArgumentBufferObjects", None),
            ("unsupported", "not_implemented Linking", None),
            ("needs_input", "function_constant", None),
            ("translated", "", None),
            // Not a key the outcome vocabulary writes.
            ("unsupported", "RayTracing", None),
            ("device_capability_missing", "RayTracing", None),
        ] {
            assert_eq!(OutOfScope::of_recorded(label, key), scope, "{label} {key}");
        }
    }

    /// A device-capability key cut at the bound may have lost an alternative without ray
    /// tracing, so only its feature can put it out of scope.
    #[test]
    fn a_cut_capability_key_is_out_of_scope_only_by_its_feature() {
        let many = (0..40)
            .map(|_| vec![Capability::Feature(VkFeature::RayQuery)])
            .chain([vec![Capability::Feature(VkFeature::ShaderInt64)]])
            .collect::<Vec<_>>();
        let error: Error = Unsupported::needs_any_of(Feature::DescriptorCount, many.clone()).into();
        assert_eq!(OutOfScope::of_error(&error), None);
        assert!(!capabilities_complete(&crate::engine::capabilities_key(
            &many
        )));
        assert_eq!(recorded(&error), None);
        let ray: Error = Unsupported::needs_any_of(Feature::RayTracing, many).into();
        assert_eq!(recorded(&ray), Some(OutOfScope::RayTracing));
    }
}
