//! Device tests over the committed profiles (`validation/profiles/`, read-only) and synthetic
//! descriptions.

use std::path::PathBuf;

use super::description::required_features;
use super::*;
use crate::device::requirements::{ApiVersion, FormatUsage, VkExtension, VkFeature};
use crate::error::{DescriptorError, Error};
use crate::vk;

mod description;
mod driver_facts;
mod enables;
mod from_queries;
mod host_identity;
mod mechanism;
mod profile;
mod render_vocabulary;
mod required_formats;
mod subgroup;

/// Every committed profile: (file stem, JSON text), sorted by stem.
pub(super) fn committed_profiles() -> Vec<(String, String)> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../metal2vulkan-validation/profiles");
    let mut out: Vec<(String, String)> = std::fs::read_dir(&dir)
        .expect("validation/profiles is readable")
        .map(|e| e.expect("directory entry").path())
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .map(|p| {
            let stem = p.file_stem().unwrap().to_string_lossy().into_owned();
            let text = std::fs::read_to_string(&p).expect("profile is readable");
            (stem, text)
        })
        .collect();
    out.sort();
    out
}

/// The description of one committed profile.
pub(super) fn profile(stem: &str) -> DeviceDescription {
    let (_, text) = committed_profiles()
        .into_iter()
        .find(|(s, _)| s == stem)
        .unwrap_or_else(|| panic!("no committed profile {stem}"));
    DeviceDescription::from_khronos_json(&text, None).expect("committed profile parses")
}
