//! Authored-case validation infrastructure for metal2vulkan.
//!
//! Harvesting, manifests, observations, and translation A/B are deliberately separate. Nothing in
//! this crate infers execution inputs or repairs a manifest.
//!
//! On Windows the crate is the replay player (`command-replay`) alone: [`replay`], and [`gpu`]
//! without its compile deadline. Every other module (the cases, the corpus and its worker, the
//! audits and the Vulkan batch) is absent there.

#[cfg(not(windows))]
pub mod acceptance;
#[cfg(not(windows))]
pub mod air;
#[cfg(not(windows))]
pub mod air_writer;
#[cfg(not(windows))]
pub mod authored_compile;
#[cfg(not(windows))]
pub mod bitcode_audit;
#[cfg(not(windows))]
pub mod budgets;
#[cfg(not(windows))]
pub mod c1;
#[cfg(not(windows))]
pub mod c3;
#[cfg(not(windows))]
pub mod call_targets;
#[cfg(not(windows))]
pub mod candidate;
#[cfg(not(windows))]
pub mod case;
#[cfg(not(windows))]
pub mod census;
#[cfg(not(windows))]
pub mod check;
#[cfg(not(windows))]
pub mod companion;
#[cfg(not(windows))]
pub mod coverage;
#[cfg(not(windows))]
pub mod cutover;
#[cfg(not(windows))]
pub mod engine;
#[cfg(not(windows))]
pub mod executor_contract;
pub mod gpu;
#[cfg(not(windows))]
pub mod hash;
#[cfg(not(windows))]
pub mod index;
#[cfg(not(windows))]
pub mod jsonl;
#[cfg(not(windows))]
pub mod l3;
#[cfg(not(windows))]
pub mod library_module;
#[cfg(not(windows))]
pub mod literal;
#[cfg(all(test, target_os = "macos", m2v_gpu))]
mod mesh_cases;
#[cfg(not(windows))]
pub mod metal;
#[cfg(not(windows))]
pub mod no_effect;
#[cfg(not(windows))]
pub mod observation;
#[cfg(not(windows))]
pub mod observation_contract;
#[cfg(not(windows))]
pub mod oracle_fuel;
#[cfg(not(windows))]
pub mod pool;
#[cfg(not(windows))]
pub mod refusal;
pub mod replay;
#[cfg(not(windows))]
pub mod requirement;
#[cfg(not(windows))]
pub mod source;
#[cfg(not(windows))]
pub mod specialization;
#[cfg(not(windows))]
pub mod spirv_tools;
#[cfg(not(windows))]
pub mod store;
#[cfg(not(windows))]
pub mod translation_audit;
#[cfg(not(windows))]
pub mod triage;
#[cfg(not(windows))]
pub mod vulkan_batch;
#[cfg(not(windows))]
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
