//! `DevicePolicy`: the consumer's product choices that, together with a `DeviceDescription`,
//! decide a `DeviceProfile`.

use crate::device::profile::{HandleDomain, PointerDomain};

/// Policy inputs of `Device::from_queries`. Every field can change emitted code, so the profile
/// carries it and its fingerprint covers it.
///
/// [`DevicePolicy::default`] is the one constructor: the runtime compiles under it, and the
/// command line, the compile worker and every validation tool take their policy from it, so a
/// tool compiles what the runtime does. [`DevicePolicy::RESIDENT_TABLES`] is the validation
/// executor's alone.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DevicePolicy {
    /// How pointer values resolve to resources: `Identity` with `handle_domain` `Identity` (the
    /// default), or `ResidentTable` with `Table` (the validation executor's).
    pub pointer_domain: PointerDomain,
    /// How resource handles resolve: `Identity` with `pointer_domain` `Identity` (the default), or
    /// `Table` with `ResidentTable` (the validation executor's).
    pub handle_domain: HandleDomain,
}

impl DevicePolicy {
    /// A resident table of pointers and a handle table, for the validation executor alone, which
    /// builds the resident root and the handle table itself.
    pub const RESIDENT_TABLES: DevicePolicy = DevicePolicy {
        pointer_domain: PointerDomain::ResidentTable,
        handle_domain: HandleDomain::Table,
    };
}

impl Default for DevicePolicy {
    /// The identity domains, the runtime's: a pointer is the device address it holds and a
    /// texture's or sampler's handle is its `gpuResourceID` ([`PointerDomain::Identity`],
    /// [`HandleDomain::Identity`]). `Device::from_queries` refuses them on a device without
    /// `bufferDeviceAddress` and `shaderInt64` (`Feature::IdentityDomains`).
    fn default() -> DevicePolicy {
        DevicePolicy {
            pointer_domain: PointerDomain::Identity,
            handle_domain: HandleDomain::Identity,
        }
    }
}
