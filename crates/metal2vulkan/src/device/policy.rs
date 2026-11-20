//! `DevicePolicy`: the consumer's product choices that, together with a `DeviceDescription`,
//! decide a `DeviceProfile`.

use crate::device::profile::{GuestIsolation, HandleDomain, PointerDomain};

/// Policy inputs of `Device::from_queries`. Every field can change emitted code, so the profile
/// carries it (containment as the encoding it selects) and its fingerprint covers it.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DevicePolicy {
    /// How pointer values resolve to resources: `ResidentTable`, or `Identity` with
    /// `handle_domain` `Identity`.
    pub pointer_domain: PointerDomain,
    /// How resource handles resolve: `Table`, or `Identity` with `pointer_domain` `Identity`.
    pub handle_domain: HandleDomain,
    /// Whether accesses are contained to the resources the binder supplied.
    pub containment: ContainmentPolicy,
    /// Whether guest shaders are isolated from each other's resources.
    pub isolation: GuestIsolation,
}

impl DevicePolicy {
    /// The untrusted-guest production policy: the default domains (a resident table of pointers
    /// and a handle table), containment and isolation on.
    pub const UNTRUSTED: DevicePolicy = DevicePolicy {
        pointer_domain: PointerDomain::ResidentTable,
        handle_domain: HandleDomain::Table,
        containment: ContainmentPolicy::Contained,
        isolation: GuestIsolation::On,
    };
}

impl Default for DevicePolicy {
    /// [`DevicePolicy::UNTRUSTED`].
    fn default() -> DevicePolicy {
        DevicePolicy::UNTRUSTED
    }
}

/// The containment choice. The encoding (robustness or in-shader checks) is selected by
/// `Device::from_queries` from the enabled robustness features; the policy only says whether to
/// contain.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ContainmentPolicy {
    /// Every access stays inside the resources the binder supplied (the default).
    Contained,
    /// No containment. Only valid with `GuestIsolation::Off`; recorded in the fingerprint. No
    /// access is checked: one past its buffer or its object, and a dereference of a null thread
    /// pointer (undefined in Metal, which reads or writes the private arena's first word, or past
    /// it at a field's offset), reach whatever memory the device puts there.
    UnsafeOff,
}
