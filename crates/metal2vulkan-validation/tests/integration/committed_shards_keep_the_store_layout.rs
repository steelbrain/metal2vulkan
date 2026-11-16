//! The committed case and observation shards are in the layout the store writes.
//!
//! The store sorts every shard it writes through one key per row type (`ShardRow::order_key`):
//! cases by AIR then case name, Metal and candidate rows by case then environment. The store's
//! writers (`put_case`, `delete_named_case`, the upserts) are the only ones; no script or other
//! tool writes these shards. Each key is one slot, so a shard so written has
//! strictly increasing keys, and every row sits in the shard its AIR's hash selects. The readers
//! check slots and placement but not order, so a tool that appends rows instead of installing them
//! through the store leaves a shard the store would never write: 23b3bbbc did that to three
//! MoltenVK shards. Byte-level canonical form is not part of this check: rows of executor v34 still
//! carry the legacy `containment.device_addresses` field that a rewrite drops.

use metal2vulkan_validation::observation::Backend;
use metal2vulkan_validation::store::CorpusStore;

#[test]
fn committed_shards_keep_the_store_layout() {
    let root = metal2vulkan_devtools::validation_dir().join("corpus");
    let store = CorpusStore::new(&root);
    let errors = store.layout_errors().expect("read the committed shards");
    assert!(
        errors.is_empty(),
        "{} committed shard row(s) leave the layout the store writes; install rows through \
         CorpusStore instead of appending them:\n{}",
        errors.len(),
        errors.join("\n")
    );
    // The readers' own checks over the whole store: every case's identity is its computed one and
    // unique, and every observation names a committed case with its inputs, in a unique slot.
    let cases = store.read_all_cases().expect("committed cases");
    let metal = store.read_metal().expect("committed Metal observations");
    let moltenvk = store
        .read_candidates(Backend::Moltenvk)
        .expect("committed MoltenVK observations");
    let vulkan = store
        .read_candidates(Backend::Vulkan)
        .expect("committed Vulkan observations");
    assert!(
        !cases.is_empty() && !metal.is_empty() && !moltenvk.is_empty() && !vulkan.is_empty(),
        "the committed store is empty: {} cases, {} Metal, {} MoltenVK, {} Vulkan rows",
        cases.len(),
        metal.len(),
        moltenvk.len(),
        vulkan.len()
    );
}

/// Every committed observation and every committed refusal record names a committed case, with
/// that case's AIR. The store's readers check this when they read a backend's rows or refusals, so
/// this reads every one of them from the committed store. A case re-keyed or deleted on one branch
/// while another committed rows or refusals of its old id leaves records that name no case:
/// `corpus-status --c3` then refuses the whole store (a case was once re-keyed while six refusals
/// of it were recorded on another branch).
#[test]
fn every_committed_observation_and_refusal_names_a_committed_case() {
    let root = metal2vulkan_devtools::validation_dir().join("corpus");
    let store = CorpusStore::new(&root);
    let mut errors = Vec::new();
    if let Err(error) = store.read_metal() {
        errors.push(format!("Metal observations: {error}"));
    }
    let mut refusals = 0;
    for backend in [Backend::Moltenvk, Backend::Vulkan] {
        if let Err(error) = store.read_candidates(backend) {
            errors.push(format!("{} observations: {error}", backend.directory()));
        }
        match store.read_refusals(backend) {
            Ok(rows) => refusals += rows.len(),
            Err(error) => errors.push(format!("{} refusals: {error}", backend.directory())),
        }
    }
    assert!(
        errors.is_empty(),
        "committed records that name no committed case (delete or re-key them with it):\n{}",
        errors.join("\n")
    );
    assert!(refusals > 0, "the committed store holds no refusal record");
}
