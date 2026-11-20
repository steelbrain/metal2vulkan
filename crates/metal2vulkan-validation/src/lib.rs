//! Authored-case validation infrastructure for metal2vulkan.
//!
//! Harvesting, manifests, observations, and translation A/B are deliberately separate. Nothing in
//! this crate infers execution inputs or repairs a manifest.

pub mod acceptance;
pub mod air;
pub mod air_writer;
pub mod authored_compile;
pub mod bitcode_audit;
pub mod budgets;
pub mod c1;
pub mod c3;
pub mod call_targets;
pub mod candidate;
pub mod case;
pub mod census;
pub mod check;
pub mod companion;
pub mod coverage;
pub mod cutover;
pub mod engine;
pub mod executor_contract;
pub mod gpu;
pub mod hash;
pub mod index;
pub mod jsonl;
pub mod l3;
pub mod library_module;
pub mod literal;
#[cfg(test)]
mod mesh_cases;
pub mod metal;
pub mod msl_portability;
pub mod no_effect;
pub mod observation;
pub mod observation_contract;
pub mod oracle_fuel;
pub mod pool;
pub mod refusal;
pub mod replay;
pub mod requirement;
pub mod source;
pub mod specialization;
pub mod spirv_tools;
pub mod store;
pub mod translation_audit;
pub mod triage;
pub mod vulkan_batch;
pub mod worker;

// The build's own fingerprint functions (this crate's `build.rs`), so tests can show which files
// feed each fingerprint.
#[cfg(test)]
#[path = "../build_support.rs"]
mod build_support;
#[cfg(test)]
mod build_support_tests;
#[cfg(test)]
#[path = "../product_fingerprint.rs"]
mod product_fingerprint;

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static SCRATCH_COUNTER: AtomicU64 = AtomicU64::new(0);

pub struct ScratchDir {
    path: PathBuf,
}

impl ScratchDir {
    pub fn new(purpose: &str) -> Result<Self, String> {
        let safe: String = purpose
            .chars()
            .map(|ch| {
                if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_') {
                    ch
                } else {
                    '_'
                }
            })
            .collect();
        loop {
            let serial = SCRATCH_COUNTER.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "metal2vulkan-validation-{}-{serial}-{safe}",
                std::process::id()
            ));
            match fs::create_dir(&path) {
                Ok(()) => return Ok(Self { path }),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(format!("create {}: {error}", path.display())),
            }
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for ScratchDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scratch_directory_is_removed_on_drop() {
        let path = {
            let scratch = ScratchDir::new("cleanup-test").unwrap();
            let path = scratch.path().to_path_buf();
            fs::write(path.join("file"), b"temporary").unwrap();
            path
        };
        assert!(!path.exists());
    }

    #[test]
    fn scratch_directory_skips_a_stale_process_serial() {
        let serial = SCRATCH_COUNTER.load(Ordering::Relaxed);
        let stale = std::env::temp_dir().join(format!(
            "metal2vulkan-validation-{}-{serial}-stale-serial-test",
            std::process::id()
        ));
        fs::create_dir(&stale).unwrap();

        let scratch = ScratchDir::new("stale-serial-test").unwrap();
        assert_ne!(scratch.path(), stale);

        fs::remove_dir(&stale).unwrap();
    }
}
