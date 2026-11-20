//! `DevicePolicy`: the consumer's product choices that, together with a `DeviceDescription`,
//! decide a `DeviceProfile`.

use crate::device::profile::{HandleDomain, PointerDomain};

/// Policy inputs of `Device::from_queries`. Every field can change emitted code, so the profile
/// carries it and its fingerprint covers it.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DevicePolicy {
    /// How pointer values resolve to resources: `ResidentTable`, or `Identity` with
    /// `handle_domain` `Identity`.
    pub pointer_domain: PointerDomain,
    /// How resource handles resolve: `Table`, or `Identity` with `pointer_domain` `Identity`.
    pub handle_domain: HandleDomain,
}

impl DevicePolicy {
    /// The default domains: a resident table of pointers and a handle table.
    pub const UNTRUSTED: DevicePolicy = DevicePolicy {
        pointer_domain: PointerDomain::ResidentTable,
        handle_domain: HandleDomain::Table,
    };
}

impl Default for DevicePolicy {
    /// [`DevicePolicy::UNTRUSTED`].
    fn default() -> DevicePolicy {
        DevicePolicy::UNTRUSTED
    }
}
