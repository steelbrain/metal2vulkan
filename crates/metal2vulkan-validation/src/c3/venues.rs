//! The devices C3 judges the engine on: a committed list of venues, each a typed kind of
//! device, the operator environment its rows are recorded under, the cases it must cover, and the
//! device and driver its rows must have run on.
//!
//! The list is data (`validation/cutover/c3-venues.json`): another device is one more entry, never
//! new code. The report treats every venue alike; a venue's kind decides only which store backend
//! holds its rows and which device type its lane runs.
//!
//! **Driver drift.** Every engine row records the device it ran on (`environment`: the device
//! name, vendor and device IDs, and the driver's ID, version, and info string). A venue declares
//! the identities it allows ([`Venue::devices`], [`DeviceIdentity`]), and a row that recorded
//! another one is not the venue's evidence: C3 judges it stale. So when a driver is updated, rows
//! recorded under the new driver read stale until the venue's declared identities change, and once
//! they change, every row of the older driver reads stale until it is recorded again. Neither is
//! ever silently current. A venue allows several devices only when translation cannot tell them
//! apart: one driver build, and captures that derive one translation key, so its slots can be
//! recorded on any of those machines. The Iris Xe of RPL-P and the UHD Graphics 770 of RPL-S are
//! not such a pair: their captures differ in `maxMemoryAllocationSize`, and the Intel venue
//! allows the RPL-P alone.
//!
//! **Profile drift**. A venue's committed profile is a `vulkaninfo` capture of its own device
//! ([`VenueProfile`]), held to it in two halves. The list holds the profile to the venue's declared
//! identity ([`Venues::check_profiles`]), so a driver update recaptures the profile in the commit
//! that re-declares the identity. Each run recorded under a venue holds the live device to the
//! profile before it records anything ([`VenueCheck`]): its identity, and the translation key its
//! description derives, which covers every member translation reads at the device's API level. The
//! record keeps the profile's key; the one currency rule reads a record whose key is not its venue
//! profile's as stale ([`VenueKeys`]).
//!
//! **Build drift**. A venue's record of another build stays current while what it executed is this
//! build's artifact of its case on the slot's profile: the digest the committed authored
//! compile census records at this build ([`VenueKeys::artifact_is_current`]).

use super::AnswerGap;
use crate::authored_compile::{CurrentArtifacts, ProfileStem};
use crate::candidate::engine::artifact::ExecutedArtifact;
use crate::cutover::list_path;
use crate::observation::Backend;
use metal2vulkan::device::ApiVersion;
use metal2vulkan::device::DeviceDescription;
use metal2vulkan::vk;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

/// What kind of device a venue is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VenueKind {
    /// A CPU Vulkan implementation (`VK_PHYSICAL_DEVICE_TYPE_CPU`, lavapipe), run with
    /// `--device-type cpu`.
    CpuIcd,
    /// A GPU with its own Vulkan driver (RADV, ANV), run natively with `--device-type gpu`, with no
    /// profiles layer.
    NativeGpu,
}

impl VenueKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::CpuIcd => "cpu_icd",
            Self::NativeGpu => "native_gpu",
        }
    }

    /// The store backend that holds the venue's rows.
    pub fn backend(self) -> Backend {
        match self {
            Self::CpuIcd | Self::NativeGpu => Backend::Vulkan,
        }
    }
}

/// The cases a venue's slot must cover.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Population {
    /// Every case over a public fixture.
    Public,
    /// Every case with a Metal reference row (`c3::reference`).
    MetalQualified,
}

impl Population {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Public => "public",
            Self::MetalQualified => "metal_qualified",
        }
    }
}

/// The device and driver a row ran on, as every engine row records them in its `environment`
/// (`VkPhysicalDeviceProperties` and `VkPhysicalDeviceDriverProperties`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeviceIdentity {
    /// `deviceName`.
    pub device: String,
    /// `vendorID`.
    pub vendor_id: u32,
    /// `deviceID`.
    pub device_id: u32,
    /// `driverID`.
    pub driver_id: i32,
    /// `driverVersion`, as the driver packs it.
    pub driver_version: u32,
    /// `driverInfo`: for Mesa, the release and its build.
    pub driver_info: String,
}

impl DeviceIdentity {
    /// The identity a row's `environment` records, or `None` when a field is missing or malformed.
    pub fn recorded(environment: &serde_json::Value) -> Option<DeviceIdentity> {
        let text = |key: &str| environment.get(key)?.as_str().map(str::to_string);
        let number = |key: &str| u32::try_from(environment.get(key)?.as_u64()?).ok();
        Some(DeviceIdentity {
            device: text("device")?,
            vendor_id: number("vendor_id")?,
            device_id: number("device_id")?,
            driver_id: i32::try_from(environment.get("driver_id")?.as_i64()?).ok()?,
            driver_version: number("driver_version")?,
            driver_info: text("driver_info")?,
        })
    }

    /// The identity a committed profile document states: `VkPhysicalDeviceProperties`' device
    /// name, vendor and device IDs, and driver version, and the driver properties' driver ID
    /// (`VkPhysicalDeviceVulkan12Properties`, or `VkPhysicalDeviceDriverProperties`) and driver
    /// info, from whichever of its capability blocks states them.
    pub fn stated(document: &Value) -> Result<DeviceIdentity, String> {
        let structure = |name: &str| {
            document
                .get("capabilities")
                .and_then(Value::as_object)
                .into_iter()
                .flat_map(|blocks| blocks.values())
                .find_map(|block| block.get("properties")?.get(name))
        };
        let core = structure("VkPhysicalDeviceProperties")
            .ok_or("the profile states no VkPhysicalDeviceProperties")?;
        let driver = structure("VkPhysicalDeviceVulkan12Properties")
            .filter(|properties| properties.get("driverID").is_some())
            .or_else(|| structure("VkPhysicalDeviceDriverProperties"))
            .ok_or("the profile states no driver properties")?;
        let text = |from: &Value, key: &str| {
            from.get(key)
                .and_then(Value::as_str)
                .map(str::to_string)
                .ok_or_else(|| format!("the profile states no {key}"))
        };
        let number = |key: &str| {
            core.get(key)
                .and_then(Value::as_u64)
                .and_then(|value| u32::try_from(value).ok())
                .ok_or_else(|| format!("the profile states no {key}"))
        };
        let driver_id = text(driver, "driverID")?;
        Ok(DeviceIdentity {
            device: text(core, "deviceName")?,
            vendor_id: number("vendorID")?,
            device_id: number("deviceID")?,
            driver_id: vk::DriverId::from_name(&driver_id)
                .ok_or_else(|| format!("the profile's driverID {driver_id} is not a VkDriverId"))?
                .as_raw(),
            driver_version: number("driverVersion")?,
            driver_info: text(driver, "driverInfo")?,
        })
    }

    /// Each field of `self` that `other` does not equal, as `field: self's (other's)`.
    pub fn differences(&self, other: &DeviceIdentity) -> Vec<String> {
        let mut out = Vec::new();
        let mut field = |name: &str, a: String, b: String| {
            if a != b {
                out.push(format!("{name}: {a} ({b})"));
            }
        };
        field("device", self.device.clone(), other.device.clone());
        field(
            "vendor_id",
            self.vendor_id.to_string(),
            other.vendor_id.to_string(),
        );
        field(
            "device_id",
            self.device_id.to_string(),
            other.device_id.to_string(),
        );
        field(
            "driver_id",
            self.driver_id.to_string(),
            other.driver_id.to_string(),
        );
        field(
            "driver_version",
            self.driver_version.to_string(),
            other.driver_version.to_string(),
        );
        field(
            "driver_info",
            self.driver_info.clone(),
            other.driver_info.clone(),
        );
        out
    }

    /// The identity's fields as the report prints them.
    pub fn label(&self) -> String {
        format!(
            "{} ({:04x}:{:04x}), driver {} {} ({})",
            self.device,
            self.vendor_id,
            self.device_id,
            self.driver_id,
            self.driver_info,
            self.driver_version
        )
    }
}

/// One device C3 judges: one translation key under one driver, on any of the devices it allows.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Venue {
    pub kind: VenueKind,
    /// The operator environment ID its rows are recorded under; its engine slot is `ID@engine`.
    /// Stable across driver updates: the driver is the declared [`DeviceIdentity`].
    pub environment_id: String,
    /// The committed profile (`validation/profiles/<profile>.json`) that describes the device, and
    /// whose translation key the venue's records hold ([`VenueProfile::fingerprint`]): the profile
    /// of one of [`Venue::devices`]. C1's reviewed list keys its device gaps by it.
    pub profile: String,
    /// The cases the slot covers: every row it holds is judged, and every case of the population
    /// a Metal row judges reads missing until the slot records it.
    pub population: Population,
    /// The devices and driver every judged row must have run on: each a device identity with the
    /// committed capture of that device, which states the identity and derives the venue's
    /// translation key ([`Venues::check_profiles`]). Several devices make one venue only when
    /// translation cannot tell them apart: one driver build, one key; a row records the device it
    /// ran on, and a row of any device the list does not allow is not the venue's.
    pub devices: Vec<VenueDevice>,
}

/// A device a venue allows ([`Venue::devices`]): its identity and the committed `vulkaninfo`
/// capture of it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VenueDevice {
    /// The committed profile captured from the device (`validation/profiles/<profile>.json`).
    pub profile: String,
    /// The device and driver, as a row records them.
    pub identity: DeviceIdentity,
}

impl Venue {
    pub fn backend(&self) -> Backend {
        self.kind.backend()
    }

    /// Whether a record of `identity` ran on a device the venue allows.
    pub fn allows(&self, identity: &DeviceIdentity) -> bool {
        self.devices
            .iter()
            .any(|device| &device.identity == identity)
    }
}

/// A venue's committed profile as the venue list, the executor's per-run check, and the currency
/// rule read it: a `vulkaninfo` capture of the venue's own device, never a census profile of
/// another machine.
#[derive(Clone, Debug)]
pub struct VenueProfile {
    /// The file stem under `validation/profiles`.
    pub stem: String,
    /// SHA-256 of the file's bytes: provenance only, which a venue's record keeps beside the key.
    pub sha256: String,
    /// The device identity the profile states ([`DeviceIdentity::stated`]).
    pub identity: DeviceIdentity,
    /// The profile as a venue run's native query reads its device: read at API 1.2, the
    /// executor's instance version (`crate::acceptance::read_at`).
    pub description: DeviceDescription,
    /// The key of the venue's records: [`VenueProfile::key_of`] `description`.
    pub fingerprint: String,
}

impl VenueProfile {
    /// The committed profile `stem` (`validation/profiles/<stem>.json`).
    pub fn committed(stem: &str) -> Result<VenueProfile, String> {
        let file = crate::engine::profiles_dir().join(format!("{stem}.json"));
        let bytes = std::fs::read(&file).map_err(|error| format!("{}: {error}", file.display()))?;
        Self::parse(stem, &bytes).map_err(|error| format!("{}: {error}", file.display()))
    }

    /// The profile document `bytes` of stem `stem`: a `vulkaninfo` capture (its
    /// `provenance.source`).
    pub fn parse(stem: &str, bytes: &[u8]) -> Result<VenueProfile, String> {
        let text = std::str::from_utf8(bytes).map_err(|error| error.to_string())?;
        let document: Value = serde_json::from_str(text).map_err(|error| error.to_string())?;
        let source = document
            .get("provenance")
            .and_then(|provenance| provenance.get("source"))
            .and_then(Value::as_str);
        if source != Some("vulkaninfo") {
            return Err(format!(
                "a venue's profile is a vulkaninfo capture of its own device; this one's source \
                 is {}",
                source.unwrap_or("unstated")
            ));
        }
        let identity = DeviceIdentity::stated(&document)?;
        let description = crate::acceptance::read_at(
            &DeviceDescription::from_khronos_json(text, None).map_err(|error| error.to_string())?,
            ApiVersion::V1_2,
        );
        Ok(VenueProfile {
            stem: stem.into(),
            sha256: crate::hash::sha256_bytes(bytes),
            identity,
            fingerprint: Self::key_of(&description)?,
            description,
        })
    }

    /// The key a venue's records hold their device to: the fingerprint, hex, of the device profile
    /// `description` derives under the default policy. That profile is the only device input
    /// translation reads (the Metal-facing properties derive from it), and its fingerprint hashes
    /// every field, its feature and extension sets and every format bit the product reads among
    /// them, so a description that differs in anything the translator reads has another key. A
    /// run's own translation key (its `profile_fingerprint`) is that profile restricted to what its
    /// context enabled, which its executor decides, a key of the record too.
    pub fn key_of(description: &DeviceDescription) -> Result<String, String> {
        crate::acceptance::derived_profile(description.clone())
            .map(|profile| crate::hash::hex_encode(profile.fingerprint()))
    }

    /// What a venue's row and refusal record of the profile its run was checked against
    /// ([`VENUE_PROFILE`]): the stem, the file's SHA-256 (provenance), and
    /// [`VenueProfile::fingerprint`] (the key).
    pub fn record(&self) -> Value {
        serde_json::json!({
            "profile": self.stem,
            "sha256": self.sha256,
            "fingerprint": self.fingerprint,
        })
    }

    /// The stem a row or refusal `environment` recorded ([`VenueProfile::record`]), if any.
    pub fn recorded_stem(environment: &Value) -> Option<&str> {
        environment.get(VENUE_PROFILE)?.get("profile")?.as_str()
    }

    /// The [`VenueProfile::fingerprint`] a row or refusal `environment` recorded
    /// ([`VenueProfile::record`]), if it recorded one.
    pub fn recorded_fingerprint(environment: &Value) -> Option<&str> {
        environment.get(VENUE_PROFILE)?.get("fingerprint")?.as_str()
    }
}

/// What a venue run's check of its device against the venue's profile found.
///
/// `stops` are the differences that refuse the run: the identity (device name, vendor and device
/// IDs, driver ID, version, and info), and the translation key ([`VenueProfile::key_of`]) of the
/// device's description read at API 1.2, with every description member that differs beside it
/// as its explanation. `unread` are the description members that differ while the key holds:
/// members no translation reads (a `VK_EXT_device_generated_commands` stage mask the capture
/// cannot state, say), which leave every artifact and record as they are, so the run proceeds and
/// reports them.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VenueCheck {
    pub stops: Vec<String>,
    pub unread: Vec<String>,
}

/// The member of a venue's row and refusal `environment` that records its [`VenueProfile`].
pub const VENUE_PROFILE: &str = "venue_profile";

/// A venue key as a record holds it ([`VenueProfile::record`]): the committed profile file's
/// SHA-256 and its translation key.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecordedKey {
    pub sha256: String,
    pub fingerprint: String,
}

/// A rekey of a venue's records: every row and refusal that recorded committed profile `stem` at
/// exactly `from` records `to` instead ([`crate::store::CorpusStore::rekey_venue_records`]). It is
/// for a change that moves a venue's key, or its profile file, without changing anything a
/// translation reads or a run executes for it, which the authored-compile census proves first:
/// every artifact digest byte-identical across the change. The records then stay the venue's
/// evidence; nothing runs. Records at any other key are left as they are (stale records stay
/// stale).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VenueRekey {
    pub stem: String,
    pub from: RecordedKey,
    pub to: RecordedKey,
}

impl VenueRekey {
    /// Load a rekey list (a JSON array of [`VenueRekey`]) and [`VenueRekey::check`] it.
    pub fn load_checked(path: &Path) -> Result<Vec<VenueRekey>, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|error| format!("{}: {error}", path.display()))?;
        let rekeys: Vec<VenueRekey> =
            serde_json::from_str(&text).map_err(|error| format!("{}: {error}", path.display()))?;
        Self::check(&rekeys).map_err(|error| format!("{}: {error}", path.display()))?;
        Ok(rekeys)
    }

    /// A rekey list moves records only to this build's keys: each `to` is the committed profile
    /// `stem` as this build reads it (its file's SHA-256 and [`VenueProfile::fingerprint`]),
    /// each `from` differs from its `to`, and no `stem` and `from` are listed twice.
    pub fn check(rekeys: &[VenueRekey]) -> Result<(), String> {
        let mut seen = BTreeSet::new();
        for rekey in rekeys {
            let at = |what: String| {
                format!(
                    "rekey of {} from {}: {what}",
                    rekey.stem, rekey.from.fingerprint
                )
            };
            let profile = VenueProfile::committed(&rekey.stem).map_err(at)?;
            let now = RecordedKey {
                sha256: profile.sha256,
                fingerprint: profile.fingerprint,
            };
            if rekey.to != now {
                return Err(at(format!(
                    "its target is not this build's committed profile (sha256 {}, key {})",
                    now.sha256, now.fingerprint
                )));
            }
            if rekey.from == rekey.to {
                return Err(at("from and to are the same key".into()));
            }
            if !seen.insert((rekey.stem.as_str(), &rekey.from)) {
                return Err(at("listed twice".into()));
            }
        }
        Ok(())
    }

    /// Whether a record whose `environment` this is recorded `stem` at exactly `from`.
    pub fn matches(&self, environment: &Value) -> bool {
        let Some(recorded) = environment.get(VENUE_PROFILE) else {
            return false;
        };
        recorded.get("profile").and_then(Value::as_str) == Some(self.stem.as_str())
            && recorded.get("sha256").and_then(Value::as_str) == Some(self.from.sha256.as_str())
            && recorded.get("fingerprint").and_then(Value::as_str)
                == Some(self.from.fingerprint.as_str())
    }

    /// Make a record that [`VenueRekey::matches`] record `to`.
    pub fn apply(&self, environment: &mut Value) {
        if let Some(recorded) = environment.get_mut(VENUE_PROFILE) {
            recorded["sha256"] = Value::String(self.to.sha256.clone());
            recorded["fingerprint"] = Value::String(self.to.fingerprint.clone());
        }
    }
}

impl Venues {
    /// The key a venue's record holds as this build reads each venue's committed profile
    /// (`corpus-status --venue-keys`): the stem, the file's SHA-256 and the translation key, one
    /// per venue. The `to` sides of a [`VenueRekey`] list.
    pub fn committed_keys(&self) -> Result<Vec<(String, RecordedKey)>, String> {
        self.venues
            .iter()
            .map(|venue| {
                let profile = VenueProfile::committed(&venue.profile)?;
                Ok((
                    profile.stem,
                    RecordedKey {
                        sha256: profile.sha256,
                        fingerprint: profile.fingerprint,
                    },
                ))
            })
            .collect()
    }
}

/// What each venue's records must hold. The key: by engine slot (each venue's native slot), the
/// venue's backend and its committed profile's [`VenueProfile::fingerprint`]. The
/// one currency rule (`index::candidate_slot_state`, `index::refusal_is_current`) reads a record in
/// such a slot of that backend stale unless it recorded that fingerprint; a record anywhere else is
/// no venue's, and the rule asks nothing of it.
///
/// The artifacts: this build's digest of each case on each venue's profile
/// ([`CurrentArtifacts`]). A venue's record of another build is current for its product while
/// what it executed is that digest ([`VenueKeys::artifact_is_current`]).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VenueKeys {
    by_slot: BTreeMap<String, VenueSlot>,
    artifacts: CurrentArtifacts,
    /// The answer gaps of each slot's venue's device.
    gaps: BTreeMap<String, BTreeSet<AnswerGap>>,
}

/// A venue's native engine slot.
#[derive(Clone, Debug, PartialEq, Eq)]
struct VenueSlot {
    backend: Backend,
    /// The committed profile's fingerprint.
    fingerprint: String,
    /// The committed profile.
    profile: ProfileStem,
}

impl VenueKeys {
    /// Hold `environment_id`'s native slot in `backend` to `fingerprint`, the key of the
    /// committed profile `profile` (its stem).
    pub fn insert(
        &mut self,
        environment_id: &str,
        backend: Backend,
        fingerprint: &str,
        profile: &str,
    ) {
        self.by_slot.insert(
            crate::candidate::engine::engine_environment_id(environment_id),
            VenueSlot {
                backend,
                fingerprint: fingerprint.into(),
                profile: ProfileStem {
                    stem: profile.into(),
                },
            },
        );
    }

    /// Hold the venues' records of another build to `artifacts`, this build's digests.
    pub fn with_artifacts(self, artifacts: CurrentArtifacts) -> VenueKeys {
        VenueKeys { artifacts, ..self }
    }

    /// Record that `environment_id`'s device has the answer gaps `gaps`, in its slot
    /// ([`AnswerGap::of_device`] of its committed profile).
    pub fn insert_gaps(&mut self, environment_id: &str, gaps: &BTreeSet<AnswerGap>) {
        self.gaps.insert(
            crate::candidate::engine::engine_environment_id(environment_id),
            gaps.clone(),
        );
    }

    /// Whether the device of the venue whose slot is `slot` has the answer gap `gap`: never for
    /// a slot no venue keys.
    pub fn has_gap(&self, slot: &str, gap: AnswerGap) -> bool {
        self.gaps.get(slot).is_some_and(|gaps| gaps.contains(&gap))
    }

    /// The venue slot `slot` of `backend` is, when it is one.
    fn slot(&self, backend: Backend, slot: &str) -> Option<&VenueSlot> {
        self.by_slot
            .get(slot)
            .filter(|venue_slot| venue_slot.backend == backend)
    }

    /// The fingerprint a record in `slot` of `backend` must have recorded, when it is a venue's.
    pub fn expected(&self, backend: Backend, slot: &str) -> Option<&str> {
        self.slot(backend, slot)
            .map(|venue_slot| venue_slot.fingerprint.as_str())
    }

    /// Whether a record of `backend` in `slot` whose environment is `environment` holds the key its
    /// venue requires: always for a record no venue keys.
    pub fn holds(&self, backend: Backend, slot: &str, environment: &Value) -> bool {
        self.expected(backend, slot).is_none_or(|expected| {
            VenueProfile::recorded_fingerprint(environment) == Some(expected)
        })
    }

    /// Whether a venue's record of `case_id` in `slot` of `backend` executed `executed`, this
    /// build's artifact of the case on the slot's profile (`ExecutedArtifact::is_current`).
    /// Never for a record no venue keys, one that recorded no digest, or a case and profile this
    /// build's file holds no digest of.
    pub fn artifact_is_current(
        &self,
        backend: Backend,
        slot: &str,
        case_id: &str,
        executed: Option<&ExecutedArtifact>,
    ) -> bool {
        self.slot(backend, slot).is_some_and(|venue_slot| {
            executed.is_some_and(|executed| {
                executed.is_current(self.artifacts.get(case_id, &venue_slot.profile))
            })
        })
    }
}

/// The committed venue list.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Venues {
    pub venues: Vec<Venue>,
}

/// The committed venue list's file under `validation/cutover/`.
pub const VENUES_FILE: &str = "c3-venues.json";

impl Venues {
    /// `validation/cutover/c3-venues.json`.
    pub fn committed_path() -> PathBuf {
        list_path(VENUES_FILE)
    }

    /// Parse and check a venue list ([`Venues::check`], [`Venues::check_profiles`]).
    pub fn parse(json: &str) -> Result<Venues, String> {
        let venues: Venues =
            serde_json::from_str(json).map_err(|error| format!("C3 venue list: {error}"))?;
        venues.check()?;
        venues.check_profiles()?;
        Ok(venues)
    }

    /// Each device a venue allows has a committed capture of its own: a `vulkaninfo` profile
    /// ([`VenueProfile::committed`]) whose identity is the device's declared identity. A census
    /// profile of another machine, or one left behind by a driver update, is refused. And every
    /// allowed device's capture derives the venue's translation key, the key of the venue's
    /// profile: devices one venue allows are ones translation cannot tell apart, so a device whose
    /// capture derives another key needs a venue of its own.
    pub fn check_profiles(&self) -> Result<(), String> {
        for venue in &self.venues {
            let at = |what: String| format!("C3 venue {:?}: {what}", venue.environment_id);
            let key = VenueProfile::committed(&venue.profile)
                .map_err(at)?
                .fingerprint;
            for device in &venue.devices {
                let profile = VenueProfile::committed(&device.profile).map_err(at)?;
                let differ = device.identity.differences(&profile.identity);
                if !differ.is_empty() {
                    return Err(at(format!(
                        "its profile {} is of another device than the venue declares (declared \
                         (profile)): {}; capture the profile from the venue's device in the commit \
                         that declares its identity",
                        device.profile,
                        differ.join("; ")
                    )));
                }
                if profile.fingerprint != key {
                    return Err(at(format!(
                        "its device {} (profile {}) derives translation key {}, not the venue's \
                         {} (profile {}): translation tells the two devices apart, so each needs a \
                         venue of its own",
                        device.identity.label(),
                        device.profile,
                        profile.fingerprint,
                        key,
                        venue.profile
                    )));
                }
            }
        }
        Ok(())
    }

    /// What every venue's records must hold ([`VenueKeys`]): each venue's committed profile's
    /// fingerprint, in its backend, and `artifacts`, this build's digests
    /// (`CurrentArtifacts::load` of the corpus whose records are judged).
    pub fn keys(&self, artifacts: CurrentArtifacts) -> Result<VenueKeys, String> {
        let mut keys = VenueKeys::default().with_artifacts(artifacts);
        for venue in &self.venues {
            let profile = VenueProfile::committed(&venue.profile)
                .map_err(|error| format!("C3 venue {:?}: {error}", venue.environment_id))?;
            keys.insert(
                &venue.environment_id,
                venue.backend(),
                &profile.fingerprint,
                &venue.profile,
            );
            let derived = crate::acceptance::derived_profile(profile.description.clone())
                .map_err(|error| format!("C3 venue {:?}: {error}", venue.environment_id))?;
            keys.insert_gaps(
                &venue.environment_id,
                &AnswerGap::of_device(&profile.description, &derived),
            );
        }
        Ok(keys)
    }

    /// The venue whose rows of `backend` are recorded under the operator environment
    /// `environment_id`, if one is.
    pub fn venue_of(&self, environment_id: &str, backend: Backend) -> Option<&Venue> {
        self.venues
            .iter()
            .find(|venue| venue.environment_id == environment_id && venue.backend() == backend)
    }

    /// Read and check the venue list at `path`; an error names the file.
    pub fn load(path: &Path) -> Result<Venues, String> {
        let json = std::fs::read_to_string(path)
            .map_err(|error| format!("read {}: {error}", path.display()))?;
        Self::parse(&json).map_err(|error| format!("{}: {error}", path.display()))
    }

    /// At least one venue; each with an operator environment ID (not an engine slot) no other
    /// venue has, a committed profile, and at least one allowed device, its
    /// profile among the devices' captures; each device listed once, with a committed capture and
    /// a declared identity with a device name and a driver info string.
    pub fn check(&self) -> Result<(), String> {
        if self.venues.is_empty() {
            return Err("C3 venue list names no venue".into());
        }
        let profiles = crate::engine::committed_profiles()?;
        let mut environments = BTreeSet::new();
        for venue in &self.venues {
            let at = |what: String| format!("C3 venue {:?}: {what}", venue.environment_id);
            if venue.environment_id.trim().is_empty()
                || venue.environment_id.contains(['@', '+'])
                || venue.environment_id.contains(char::is_whitespace)
            {
                return Err(at(
                    "an operator environment ID has no whitespace, '@', or '+' (an engine \
                     slot's suffixes)"
                        .into(),
                ));
            }
            if !environments.insert(venue.environment_id.as_str()) {
                return Err(at("listed twice".into()));
            }
            if !profiles.contains(&venue.profile) {
                return Err(at(format!(
                    "{:?} is not a committed profile",
                    venue.profile
                )));
            }
            if venue.devices.is_empty() {
                return Err(at("allows no device".into()));
            }
            if !venue
                .devices
                .iter()
                .any(|device| device.profile == venue.profile)
            {
                return Err(at(format!(
                    "its profile {:?} is none of its devices' captures",
                    venue.profile
                )));
            }
            for (i, device) in venue.devices.iter().enumerate() {
                let identity = &device.identity;
                if identity.device.trim().is_empty() || identity.driver_info.trim().is_empty() {
                    return Err(at(
                        "the declared identity needs a device name and a driver info string".into(),
                    ));
                }
                if !profiles.contains(&device.profile) {
                    return Err(at(format!(
                        "{:?} is not a committed profile",
                        device.profile
                    )));
                }
                if venue.devices[..i].iter().any(|earlier| {
                    earlier.identity == *identity || earlier.profile == device.profile
                }) {
                    return Err(at(format!(
                        "allows {} (profile {}) twice",
                        identity.label(),
                        device.profile
                    )));
                }
            }
        }
        Ok(())
    }
}
