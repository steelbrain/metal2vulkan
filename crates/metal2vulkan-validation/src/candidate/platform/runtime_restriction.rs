//! A run through the runtime on a device `VK_LAYER_KHRONOS_profiles` restricts to this process's
//! committed profile P (`--profiles-layer P`, [`profiles::restrict_devices_to`]).
//!
//! The runtime makes its own instance, so the layer is stacked below the validation layer through
//! the loader's variable (`VK_INSTANCE_LAYERS`), and configured through the variables its settings
//! library reads ([`ProfilesSettings::environment`]): the settings the executor's own context
//! passes as `VkLayerSettingEXT`. The variables are set while an instance under them lives and
//! removed after it, so no other instance of the process reads them. Before the run's device is
//! made, as that context checks:
//!
//! - the layer is installed, and the device as it is, read through an instance whose layer has its
//!   emulation off, has every capability P states ([`profiles::check_within`]); where it lacks one
//!   the run is a typed refusal naming the members, recorded with that device's identity;
//! - an instance whose layer presents P and hides nothing lists the extensions the layer presents,
//!   of which the run's instance hides the ones P does not list ([`ProfileRestriction::hidden`]);
//! - through the run's instance the layer is active, the members it keeps from the device are the
//!   device's, and every other member is P's ([`profiles::check_restricted`]).
//!
//! Each is read at the runtime's instance version (`rt::floor::VERSION`), the version the runtime
//! reads the device it opens at: the runtime translates for P as the layer presents it.

use super::profiles::{
    self, ActiveTool, ProfileRestriction, ProfilesLayer, ProfilesSettings, RestrictionError,
    PROFILES_LAYER, PROFILES_VARIABLE_PREFIX,
};
use super::runtime::{environment, select, VALIDATION_LAYER};
use crate::candidate::engine::{runtime_fingerprint, RefusedOn, RunAttempt};
use crate::check::admission::NotSupported;
use crate::gpu::DeviceType;
use metal2vulkan::device::{ApiVersion, DeviceDescription};
use metal2vulkan_runtime as rt;
use std::collections::BTreeSet;

#[cfg(test)]
mod tests;

/// How the instances of a run are made: under the validation layer alone, or with the profiles
/// layer below it, restricting the device to P.
pub(super) struct Layered {
    restricted: Option<Restricted>,
}

/// A restriction the device takes, read before the run.
struct Restricted {
    restriction: &'static ProfileRestriction,
    /// The device as it is.
    native: DeviceDescription,
    /// The extensions the run's instance hides.
    hidden: BTreeSet<String>,
}

impl Layered {
    /// The committed profile the run is restricted to, if any.
    pub(super) fn profile(&self) -> Option<&'static str> {
        let restricted = self.restricted.as_ref()?;
        Some(restricted.restriction.stem.as_str())
    }

    /// `body` on the runtime's instance, made under the layers: restricted, the profiles layer's
    /// variables are set while `body` runs.
    pub(super) fn instance<T>(
        &self,
        body: impl FnOnce(&rt::Instance) -> Result<T, String>,
    ) -> Result<T, String> {
        let Some(restricted) = &self.restricted else {
            return body(&made()?);
        };
        let layer = ProfilesLayer::Restrict(restricted.restriction, &restricted.hidden);
        under(&ProfilesSettings::new(layer)?, || body(&made()?))
    }

    /// The last check before the run's device is made, of the device `physical` the run's
    /// instance selected, with `tools` active on it: restricted, the layer is active and presents
    /// P ([`profiles::check_restricted`]); on the device as it is, none.
    pub(super) fn checked(
        &self,
        instance: &rt::Instance,
        physical: &rt::PhysicalDevice,
        tools: &[ActiveTool],
    ) -> Result<(), String> {
        let Some(restricted) = &self.restricted else {
            return Ok(());
        };
        let queried = description(instance, physical)?;
        let (restriction, native) = (restricted.restriction, &restricted.native);
        profiles::check_restricted(restriction, native, tools, &queried, version())
            .map(drop)
            .map_err(|error| error.to_string())
    }

    /// What a row's environment records of the profiles layer: `null` unrestricted, else the
    /// layer's tool among `tools`, P and the extensions the run hid, as the own context records it.
    pub(super) fn record(&self, tools: &[ActiveTool]) -> serde_json::Value {
        let Some(Restricted {
            restriction,
            hidden,
            ..
        }) = &self.restricted
        else {
            return serde_json::Value::Null;
        };
        serde_json::json!({
            "tool": profiles::profiles_tool(tools),
            "profile": restriction.stem,
            "profile_name": restriction.profile_name,
            "profile_sha256": restriction.sha256,
            "hidden_extensions": hidden,
        })
    }
}

/// The layers of a run of `executor` on the device of `device_type` under this process's
/// restriction, read before the run: `Err` with the restriction the device refuses where it lacks
/// a capability P states; a failure of a probe is the outer `Err`.
pub(super) fn layered(
    device_type: DeviceType,
    executor: &'static str,
) -> Result<Result<Layered, Refused>, String> {
    let Some(restriction) = profiles::current_restriction() else {
        return Ok(Ok(Layered { restricted: None }));
    };
    let pass_through = ProfilesSettings::new(ProfilesLayer::PassThrough)?;
    let (native, identity, tools) = under(&pass_through, || {
        let instance = made()?;
        let (entry, _) = instance.raw();
        // SAFETY: the loader is loaded, and the call takes no layer.
        let layers = unsafe { entry.enumerate_instance_layer_properties() }
            .map_err(|error| format!("enumerate the Vulkan layers: {error}"))?;
        let installed = layers.iter().any(|layer| {
            layer
                .layer_name_as_c_str()
                .is_ok_and(|name| name == PROFILES_LAYER)
        });
        if !installed {
            return Err(RestrictionError::LayerMissing.to_string());
        }
        let (physical, identity) = select(&instance, device_type, None)?;
        let tools = tools_on(&instance, &physical)?;
        Ok((description(&instance, &physical)?, identity, tools))
    })?;
    let gaps = match profiles::check_within(restriction, &native, version()) {
        Ok(()) => Vec::new(),
        Err(RestrictionError::Exceeds { gaps, .. }) => gaps,
        Err(error) => return Err(error.to_string()),
    };
    if !gaps.is_empty() {
        let device = environment(&identity, &tools, executor, serde_json::Value::Null);
        return Ok(Err(Refused {
            restriction,
            gaps,
            device,
        }));
    }
    let none = BTreeSet::new();
    let presenting = ProfilesSettings::new(ProfilesLayer::Restrict(restriction, &none))?;
    let presented = under(&presenting, || -> Result<Vec<String>, String> {
        let instance = made()?;
        let (physical, _) = select(&instance, device_type, None)?;
        let (_, vk_instance) = instance.raw();
        // SAFETY: `physical` is one of the instance's physical devices, and the instance is live.
        let extensions =
            unsafe { vk_instance.enumerate_device_extension_properties(physical.raw()) }
                .map_err(|error| format!("enumerate the presented device extensions: {error}"))?;
        Ok(extensions
            .iter()
            .filter_map(|extension| extension.extension_name_as_c_str().ok())
            .map(|name| name.to_string_lossy().into_owned())
            .collect())
    })?;
    let hidden = restriction.hidden(presented.iter().map(String::as_str));
    let restricted = Restricted {
        restriction,
        native,
        hidden,
    };
    Ok(Ok(Layered {
        restricted: Some(restricted),
    }))
}

/// A restriction the device cannot take: P states `gaps`, capabilities the device as it is lacks.
pub(super) struct Refused {
    restriction: &'static ProfileRestriction,
    gaps: Vec<String>,
    /// The environment a row on the device as it is records.
    device: serde_json::Value,
}

impl Refused {
    /// The run's typed refusal, recorded with the device as it is and the restriction it refused.
    pub(super) fn attempt(self) -> RunAttempt {
        let Refused {
            restriction,
            gaps,
            mut device,
        } = self;
        if let Some(environment) = device.as_object_mut() {
            let layer = serde_json::json!({
                "profile": restriction.stem,
                "profile_name": restriction.profile_name,
                "profile_sha256": restriction.sha256,
                "refused": gaps,
            });
            environment.insert("profiles_layer".into(), layer);
        }
        RunAttempt {
            result: Err(NotSupported::Restriction {
                profile: restriction.stem.clone(),
                gaps,
            }
            .into()),
            refused_on: Some(RefusedOn {
                environment: device,
                translator_fingerprint: runtime_fingerprint(),
                artifact: None,
            }),
        }
    }
}

impl std::fmt::Display for Refused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let exceeds = RestrictionError::Exceeds {
            profile: self.restriction.stem.clone(),
            gaps: self.gaps.clone(),
        };
        write!(f, "{exceeds}")
    }
}

/// The version the runtime's instance runs at.
fn version() -> ApiVersion {
    ApiVersion::from_packed(rt::floor::VERSION)
}

/// The runtime's instance.
fn made() -> Result<rt::Instance, String> {
    rt::Instance::new(&[]).map_err(|error| format!("the runtime's instance: {error}"))
}

/// The tools active on `physical`.
pub(super) fn tools_on(
    instance: &rt::Instance,
    physical: &rt::PhysicalDevice,
) -> Result<Vec<ActiveTool>, String> {
    let (entry, vk_instance) = instance.raw();
    // SAFETY: `physical` is one of the instance's physical devices, and the instance is live.
    unsafe { profiles::active_tools(entry, vk_instance, physical.raw()) }
}

/// The description of `physical` as the runtime reads it.
fn description(
    instance: &rt::Instance,
    physical: &rt::PhysicalDevice,
) -> Result<DeviceDescription, String> {
    let (_, vk_instance) = instance.raw();
    // SAFETY: `physical` is one of the instance's physical devices, and the instance was made at
    // the runtime's version.
    unsafe { DeviceDescription::query(vk_instance, physical.raw(), version()) }
        .map_err(|error| format!("query the device: {error}"))
}

/// `body` with the profiles layer below the validation layer, configured by `settings`, for every
/// instance made while it runs: the loader's layer variable and the layer's own set, every other
/// variable of the layer's removed, and all of them restored after it. A device worker makes one
/// instance at a time, on this thread, and no other thread reads the environment.
fn under<T>(settings: &ProfilesSettings, body: impl FnOnce() -> T) -> T {
    let clear = || {
        for (name, _) in std::env::vars_os() {
            if name.to_string_lossy().starts_with(PROFILES_VARIABLE_PREFIX) {
                std::env::remove_var(name);
            }
        }
    };
    clear();
    let layers = format!("{VALIDATION_LAYER}:{}", PROFILES_LAYER.to_string_lossy());
    std::env::set_var("VK_INSTANCE_LAYERS", layers);
    for (name, value) in settings.environment() {
        std::env::set_var(name, value);
    }
    let value = body();
    clear();
    std::env::set_var("VK_INSTANCE_LAYERS", VALIDATION_LAYER);
    value
}
