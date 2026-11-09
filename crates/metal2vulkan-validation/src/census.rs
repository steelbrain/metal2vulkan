//! The authoring-capability census run (`corpus-triage --audit authoring-capabilities`).
//!
//! A run first makes the index's public rows and case rows this checkout's
//! ([`reconcile_checkout_rows`]), so an index another checkout synchronized never hands it a row it
//! cannot read. It then selects the rows to classify from SQLite alone (the uncached ones, or every
//! one with `reclassify_all`), reads exactly their indexed byte ranges (public rows from the
//! checkout's fixtures), classifies them on a bounded pool, caches the facts in fixed-size batches,
//! and summarizes from the index.

use crate::index::{reconcile_checkout_rows, CaseReconciliation, PublicReconciliation};
use crate::pool::parallel;
use crate::source::{for_each_indexed_source_analysis_in, IndexedSourceReadStats, PublicSources};
use crate::triage::{
    authoring_capability_summary, classify_summary, select_all, select_uncached, write_cached,
    AuthoringCapabilitySummary, StructuralTriage,
};
use std::path::Path;
use std::time::{Duration, Instant};

/// Facts written to the index per transaction.
const CACHE_BATCH: usize = 1024;

/// What one census run did and found.
#[derive(Clone, Debug)]
pub struct AuthoringCensusRun {
    /// How the index's public rows were made the checkout's before selection.
    pub public: PublicReconciliation,
    /// How the index's case rows were made the checkout's before selection.
    pub cases: CaseReconciliation,
    /// Rows selected for classification.
    pub selected: usize,
    pub selection_elapsed: Duration,
    pub classify_elapsed: Duration,
    /// What reading the selected rows opened and read (a warm pass reads nothing).
    pub read: IndexedSourceReadStats,
    /// The census over every indexed row, from the index alone.
    pub summary: AuthoringCapabilitySummary,
}

/// Run the census over the index at `index` of the corpus at `root`, with `public` as this
/// checkout's public fixtures ([`crate::source::public_source_set`]): at most `limit` rows, on
/// `jobs` workers.
pub fn run_authoring_capability_census(
    root: &Path,
    index: &Path,
    public: &PublicSources,
    limit: usize,
    jobs: usize,
    reclassify_all: bool,
) -> Result<AuthoringCensusRun, String> {
    let started = Instant::now();
    let reconciliation = reconcile_checkout_rows(root, index, public)?;
    let hashes = if reclassify_all {
        select_all(index, limit)?
    } else {
        select_uncached(index, limit)?
    };
    let selection_elapsed = started.elapsed();
    let classify_started = Instant::now();
    let read = classify_and_cache_parallel(root, index, public, &hashes, jobs)?;
    let classify_elapsed = classify_started.elapsed();
    Ok(AuthoringCensusRun {
        public: reconciliation.public,
        cases: reconciliation.cases,
        selected: hashes.len(),
        selection_elapsed,
        classify_elapsed,
        read,
        summary: authoring_capability_summary(index)?,
    })
}

fn classify_and_cache_parallel(
    root: &Path,
    index: &Path,
    public: &PublicSources,
    hashes: &[String],
    jobs: usize,
) -> Result<IndexedSourceReadStats, String> {
    let worker_count = jobs.min(hashes.len().max(1));
    let mut batch = Vec::with_capacity(CACHE_BATCH);
    // A source can be tens of MiB. Keep only one parsed row waiting outside the workers; a
    // worker-count-sized queue would turn the channel itself into an unbounded-in-practice
    // corpus cache and can exceed the process memory budget on adjacent large rows.
    let read = parallel(
        worker_count,
        1,
        |send| for_each_indexed_source_analysis_in(root, index, public, hashes, send),
        |source: crate::source::SourceRow| {
            // Facts are cached under the row's key.
            classify_summary(&source).map(|result| (source.key().into_owned(), result))
        },
        |result| {
            batch.push(result?);
            if batch.len() == CACHE_BATCH {
                write_triage_batch(index, &batch)?;
                batch.clear();
            }
            Ok(())
        },
    )?;
    write_triage_batch(index, &batch)?;
    Ok(read)
}

fn write_triage_batch(index: &Path, batch: &[(String, StructuralTriage)]) -> Result<(), String> {
    if batch.is_empty() {
        return Ok(());
    }
    write_cached(
        index,
        batch.iter().map(|(hash, result)| (hash.as_str(), result)),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::sha256_bytes;
    use crate::index::{default_index_path, rebuild_index_with, sync_index_with, PublicIdentity};
    use crate::requirement::ToolingRequirement;
    use crate::source::{
        merge_source_shards, public_source_set, shard_index_for_hash, write_source_shards,
        PublicSources, SourceRow,
    };
    use crate::ScratchDir;
    use rusqlite::Connection;
    use std::collections::{BTreeMap, BTreeSet};

    /// A row of a committed kernel fixture's text with `note` appended (a new content identity),
    /// labelled `label`.
    fn variant(base: &SourceRow, note: &str, label: &str) -> SourceRow {
        let mut row = base.clone();
        row.air_ll = format!("{}\n; {note}\n", base.air_ll);
        row.air_sha256 = sha256_bytes(row.air_ll.as_bytes());
        row.label = label.into();
        row.lib_sha256s = vec!["owned-synthetic".into()];
        row
    }

    fn identity(row: &SourceRow) -> PublicIdentity {
        PublicIdentity {
            air_sha256: row.air_sha256.clone(),
            label: row.label.clone(),
        }
    }

    /// Two private rows in their shards, and the public fixtures of two checkouts: `x.ll`, whose
    /// text the later checkout re-keyed, and `y.ll`, which both share.
    struct Corpus {
        scratch: ScratchDir,
        private: [SourceRow; 2],
        before: PublicSources,
        after: PublicSources,
    }

    impl Corpus {
        fn new(purpose: &str) -> Self {
            let base = public_source_set()
                .unwrap()
                .rows
                .into_iter()
                .find(|row| row.stage == "Kernel")
                .unwrap();
            let private = [
                variant(&base, "private source a", "private/a"),
                variant(&base, "private source b", "private/b"),
            ];
            let shared = variant(&base, "fixture y", "public/y.ll");
            let set = |rows: Vec<SourceRow>| PublicSources { rows };
            let before = set(vec![
                variant(&base, "fixture x before its re-key", "public/x.ll"),
                shared.clone(),
            ]);
            let after = set(vec![
                variant(&base, "fixture x after its re-key", "public/x.ll"),
                shared,
            ]);
            let scratch = ScratchDir::new(purpose).unwrap();
            write_source_shards(scratch.path(), private.clone()).unwrap();
            Self {
                scratch,
                private,
                before,
                after,
            }
        }

        fn root(&self) -> &Path {
            self.scratch.path()
        }

        fn private_shards(&self) -> usize {
            self.private
                .iter()
                .map(|row| shard_index_for_hash(&row.air_sha256).unwrap())
                .collect::<BTreeSet<_>>()
                .len()
        }

        /// The census as `corpus-triage --audit authoring-capabilities` runs it.
        fn census(&self, index: &Path, public: &PublicSources, all: bool) -> AuthoringCensusRun {
            run_authoring_capability_census(self.root(), index, public, usize::MAX, 2, all).unwrap()
        }
    }

    fn sources(index: &Path) -> Vec<(String, String, Option<i64>)> {
        Connection::open(index)
            .unwrap()
            .prepare("SELECT air_sha256, label, shard FROM sources ORDER BY air_sha256")
            .unwrap()
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap()
    }

    fn memberships(index: &Path) -> Vec<(String, String)> {
        Connection::open(index)
            .unwrap()
            .prepare("SELECT air_sha256, lib_sha256 FROM source_library_memberships ORDER BY 1, 2")
            .unwrap()
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap()
    }

    /// Whether the index `index` holds the source rows and parent libraries a rebuild of `root`
    /// for the checkout `public` makes.
    fn rebuilt_alike(root: &Path, index: &Path, public: &PublicSources) {
        let rebuilt = root.join("rebuilt.sqlite");
        if rebuilt.exists() {
            std::fs::remove_file(&rebuilt).unwrap();
        }
        rebuild_index_with(root, &rebuilt, public).unwrap();
        assert_eq!(sources(index), sources(&rebuilt));
        assert_eq!(memberships(index), memberships(&rebuilt));
    }

    /// An index synchronized by a checkout whose public fixture was re-keyed after it, in either
    /// direction (an older checkout's index read by a newer one, and the reverse), runs the census
    /// without error: the run retires the other checkout's identity and adds its own, without
    /// scanning a private shard. Each warm pass selects nothing and reads nothing, and a checkout
    /// that returns to the index reuses the facts of its identity, keyed by content.
    #[test]
    fn a_census_over_an_index_another_checkout_synchronized_runs_in_both_directions() {
        let corpus = Corpus::new("census-public-rekey");
        let index = corpus.root().join("index.sqlite");
        let x = |public: &PublicSources| identity(&public.rows[0]);
        for (synchronized, checkout) in [
            (&corpus.before, &corpus.after),
            (&corpus.after, &corpus.before),
        ] {
            if index.exists() {
                std::fs::remove_file(&index).unwrap();
            }
            rebuild_index_with(corpus.root(), &index, synchronized).unwrap();

            let cold = corpus.census(&index, checkout, true);
            assert_eq!(cold.public.retired, vec![x(synchronized)]);
            assert_eq!(cold.public.added, vec![x(checkout)]);
            assert_eq!(cold.public.kept, 1, "the shared fixture is kept");
            assert_eq!(cold.selected, 4);
            assert_eq!(cold.summary.total_sources, 4);
            assert_eq!(cold.summary.classified_sources, 4);
            assert_eq!(cold.summary.remaining_sources(), 0);
            assert_eq!(cold.read.rows, 4);
            assert_eq!(cold.read.source_shards_opened, corpus.private_shards());
            assert_eq!(cold.read.repair_shards_scanned, 0);
            assert_eq!(cold.read.repair_bytes_scanned, 0);

            let warm = corpus.census(&index, checkout, false);
            assert!(warm.public.is_unchanged());
            assert_eq!(warm.public.kept, 2);
            assert_eq!(warm.selected, 0);
            assert_eq!(warm.read, IndexedSourceReadStats::default());
            assert_eq!(warm.summary, cold.summary);

            // The other checkout on the same index: only its own identity of `x.ll` is new to the
            // index, and it is read from that checkout's fixture, not from a private shard.
            let back = corpus.census(&index, synchronized, false);
            assert_eq!(back.public.retired, vec![x(checkout)]);
            assert_eq!(back.public.added, vec![x(synchronized)]);
            assert_eq!(back.selected, 1);
            assert_eq!(back.read.rows, 1);
            assert_eq!(back.read.source_shards_opened, 0);
            assert_eq!(back.read.source_bytes_read, 0);
            assert_eq!(back.read.repair_shards_scanned, 0);
            assert_eq!(back.summary.total_sources, 4);
            assert_eq!(back.summary.remaining_sources(), 0);
            let warm = corpus.census(&index, synchronized, false);
            assert!(warm.public.is_unchanged());
            assert_eq!(warm.selected, 0);
            assert_eq!(warm.read, IndexedSourceReadStats::default());

            // Returning reuses the facts cached under its identity: nothing to select or read.
            let again = corpus.census(&index, checkout, false);
            assert_eq!(again.public.added, vec![x(checkout)]);
            assert_eq!(again.selected, 0);
            assert_eq!(again.read, IndexedSourceReadStats::default());
            assert_eq!(again.summary.classified_sources, 4);
        }
    }

    /// A sync reconciles the public rows the same way and opens no unchanged private shard: the
    /// index it leaves is the one a rebuild by the same checkout makes.
    #[test]
    fn a_sync_retires_a_re_keyed_identity_without_opening_a_private_shard() {
        let corpus = Corpus::new("sync-public-rekey");
        let synced = corpus.root().join("synced.sqlite");
        let rebuilt = corpus.root().join("rebuilt.sqlite");
        rebuild_index_with(corpus.root(), &synced, &corpus.before).unwrap();
        let stats = sync_index_with(corpus.root(), &synced, &corpus.after).unwrap();
        assert_eq!(stats.source_shards_scanned, 0);
        assert_eq!(stats.source_bytes_scanned, 0);
        assert_eq!((stats.public_rows_retired, stats.public_rows_added), (1, 1));
        rebuild_index_with(corpus.root(), &rebuilt, &corpus.after).unwrap();
        assert_eq!(sources(&synced), sources(&rebuilt));
        let warm = sync_index_with(corpus.root(), &synced, &corpus.after).unwrap();
        assert_eq!((warm.public_rows_retired, warm.public_rows_added), (0, 0));
        assert_eq!(warm.source_shards_scanned, 0);
    }

    /// A module that is both a public fixture and a private row is the private row, with its own
    /// libraries, whichever checkout reads the index: a checkout with the fixture and one without
    /// it each rebuild and sync it alike, and neither's reconciliation retires it or adds a public
    /// row over it, so no checkout leaves the index without the private source.
    #[test]
    fn a_module_a_private_shard_holds_is_its_private_row_in_every_checkout() {
        let corpus = Corpus::new("public-and-private");
        let private = &corpus.private[0];
        let mut fixture = private.clone();
        fixture.label = "public/z.ll".into();
        fixture.lib_sha256s = vec!["fixture-library".into()];
        let with = PublicSources {
            rows: [corpus.before.rows.clone(), vec![fixture]].concat(),
        };
        let without = &corpus.before;
        let shard = shard_index_for_hash(&private.air_sha256).unwrap() as i64;
        let private_row = (
            private.air_sha256.clone(),
            private.label.clone(),
            Some(shard),
        );
        let private_libraries = (private.air_sha256.clone(), "owned-synthetic".to_string());
        let holds_the_private_row = |index: &Path| {
            assert!(
                sources(index).contains(&private_row),
                "{:?}",
                sources(index)
            );
            let libraries = memberships(index)
                .into_iter()
                .filter(|(hash, _)| *hash == private.air_sha256)
                .collect::<Vec<_>>();
            assert_eq!(libraries, std::slice::from_ref(&private_libraries));
        };

        // The checkout with the fixture rebuilds; the one without it, then the first again, run
        // the census on the same index.
        let index = corpus.root().join("index.sqlite");
        rebuild_index_with(corpus.root(), &index, &with).unwrap();
        holds_the_private_row(&index);
        let run = corpus.census(&index, without, true);
        assert!(run.public.is_unchanged(), "{:?}", run.public);
        assert_eq!(run.summary.total_sources, 4);
        assert_eq!(run.summary.remaining_sources(), 0);
        holds_the_private_row(&index);
        let run = corpus.census(&index, &with, false);
        assert!(run.public.is_unchanged(), "{:?}", run.public);
        assert_eq!(run.public.kept, 3, "x, y, and z's private row");
        assert_eq!(run.summary.total_sources, 4);
        holds_the_private_row(&index);

        // A sync with no index rebuilds it, and counts only the public rows it added: x and y,
        // not z, whose module the private row holds.
        std::fs::remove_file(&index).unwrap();
        let stats = sync_index_with(corpus.root(), &index, &with).unwrap();
        assert!(stats.rebuilt);
        assert_eq!(stats.public_rows_added, with.rows.len() - 1);
        holds_the_private_row(&index);

        // A sync by either checkout of either one's rebuild is that checkout's rebuild.
        for (rebuilt_by, synced_by) in [(&with, without), (without, &with)] {
            std::fs::remove_file(&index).unwrap();
            rebuild_index_with(corpus.root(), &index, rebuilt_by).unwrap();
            sync_index_with(corpus.root(), &index, synced_by).unwrap();
            holds_the_private_row(&index);
            rebuilt_alike(corpus.root(), &index, synced_by);
        }
    }

    /// A private row replaces the public row of its module, with the private row's libraries,
    /// both when the shard writer publishes it and when a sync reads the changed shard. When its
    /// shard drops it again, the same sync makes the module the checkout's public row. Each state
    /// is the one a rebuild makes.
    #[test]
    fn a_private_row_replaces_a_public_row_and_a_dropped_one_restores_it() {
        let corpus = Corpus::new("private-replaces-public");
        let public = &corpus.before;
        let fixture = &public.rows[1];
        let mut private = fixture.clone();
        private.label = "private/y".into();
        let harvested_library = "ab".repeat(32);
        private.lib_sha256s = vec![harvested_library.clone()];
        let shard = shard_index_for_hash(&private.air_sha256).unwrap() as i64;
        // The writer publishes to the default index; `stale` is synchronized afterwards.
        let published = default_index_path(corpus.root());
        let stale = corpus.root().join("stale.sqlite");
        rebuild_index_with(corpus.root(), &published, public).unwrap();
        rebuild_index_with(corpus.root(), &stale, public).unwrap();
        let public_row = (fixture.air_sha256.clone(), fixture.label.clone(), None);
        assert!(sources(&published).contains(&public_row));

        merge_source_shards(corpus.root(), [private.clone()]).unwrap();
        let private_row = (
            private.air_sha256.clone(),
            private.label.clone(),
            Some(shard),
        );
        let private_libraries = (private.air_sha256.clone(), harvested_library.clone());
        for index in [&published, &stale] {
            let stats = sync_index_with(corpus.root(), index, public).unwrap();
            assert_eq!((stats.public_rows_retired, stats.public_rows_added), (0, 0));
            assert!(
                sources(index).contains(&private_row),
                "{:?}",
                sources(index)
            );
            assert!(memberships(index).contains(&private_libraries));
            assert!(!memberships(index).iter().any(
                |(hash, library)| *hash == private.air_sha256 && *library != harvested_library
            ));
            rebuilt_alike(corpus.root(), index, public);
        }

        write_source_shards(corpus.root(), corpus.private.clone()).unwrap();
        let stats = sync_index_with(corpus.root(), &published, public).unwrap();
        assert_eq!((stats.public_rows_retired, stats.public_rows_added), (0, 1));
        assert!(sources(&published).contains(&public_row));
        rebuilt_alike(corpus.root(), &published, public);
    }

    /// A selected row that no repair can give a source fails, naming the row, its shard, and why:
    /// a public identity of another checkout (refused before any shard is scanned), or a private
    /// row its shard does not hold.
    #[test]
    fn a_row_without_a_source_is_named_with_its_shard_and_why() {
        let corpus = Corpus::new("census-missing-source");
        let index = corpus.root().join("index.sqlite");
        rebuild_index_with(corpus.root(), &index, &corpus.before).unwrap();
        let stale = &corpus.before.rows[0];
        let shard = shard_index_for_hash(&stale.air_sha256).unwrap();
        let error = for_each_indexed_source_analysis_in(
            corpus.root(),
            &index,
            &corpus.after,
            std::slice::from_ref(&stale.air_sha256),
            |_| Ok(()),
        )
        .unwrap_err();
        assert!(error.contains(&stale.air_sha256), "{error}");
        assert!(error.contains(&format!("(shard {shard})")), "{error}");
        assert!(error.contains("public/x.ll"), "{error}");
        assert!(error.contains("stale public identity"), "{error}");

        // A private row in the first private row's shard that the shard does not hold.
        let present = &corpus.private[0];
        let shard = shard_index_for_hash(&present.air_sha256).unwrap();
        let missing = (0u32..)
            .map(|nonce| sha256_bytes(format!("absent {nonce}").as_bytes()))
            .find(|hash| shard_index_for_hash(hash).unwrap() == shard)
            .unwrap();
        Connection::open(&index)
            .unwrap()
            .execute(
                "INSERT INTO sources (air_sha256, stage, entry, label, shard) \
                 VALUES (?1, 'Kernel', 'k', 'private/absent', ?2)",
                rusqlite::params![missing, shard],
            )
            .unwrap();
        let error = for_each_indexed_source_analysis_in(
            corpus.root(),
            &index,
            &corpus.before,
            std::slice::from_ref(&missing),
            |_| Ok(()),
        )
        .unwrap_err();
        assert!(error.contains(&missing), "{error}");
        assert!(error.contains(&format!("(shard {shard})")), "{error}");
        assert!(error.contains("missing private source"), "{error}");
    }

    /// Two kernels of one module, each storing to its buffer; `only` keeps the first alone.
    fn kernels(only: bool) -> String {
        let buffer = "!3 = !{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, \
                      !\"air.read_write\", !\"air.address_space\", i32 1, \
                      !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, \
                      !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"data\"}";
        let first = "define void @first(ptr addrspace(1) %data) {\n  \
                     store i32 1, ptr addrspace(1) %data\n  ret void\n}\n";
        if only {
            format!("{first}!air.kernel = !{{!0}}\n!0 = !{{ptr @first, !1, !2}}\n!1 = !{{}}\n!2 = !{{!3}}\n{buffer}\n")
        } else {
            format!(
                "{first}define void @second(ptr addrspace(1) %data) {{\n  \
                 store i32 2, ptr addrspace(1) %data\n  ret void\n}}\n\
                 !air.kernel = !{{!0, !4}}\n!0 = !{{ptr @first, !1, !2}}\n!1 = !{{}}\n!2 = !{{!3}}\n\
                 {buffer}\n!4 = !{{ptr @second, !1, !2}}\n"
            )
        }
    }

    /// A module declaring an object function (`!air.object`) and a mesh function (`!air.mesh`),
    /// or only the mesh function, in Apple's ABI shape: the payload in address space 6, the grid
    /// handle in `threadgroup`, the mesh output in address space 7.
    fn mesh_module(with_object: bool) -> String {
        let mesh = "define void @msh(ptr addrspace(7) %m) {\n  \
                    call void @air.set_primitive_count_mesh(ptr addrspace(7) %m, i32 1)\n  \
                    call void @air.set_primitive_id_mesh(ptr addrspace(7) %m, i32 0, i32 0)\n  \
                    ret void\n}\n\
                    declare void @air.set_primitive_count_mesh(ptr addrspace(7), i32)\n\
                    declare void @air.set_primitive_id_mesh(ptr addrspace(7), i32, i32)\n";
        let mesh_md = "!5 = !{ptr @msh, !1, !6}\n!6 = !{!7}\n\
                       !7 = !{i32 0, !\"air.mesh\", !8, !\"air.arg_type_name\", \
                       !\"mesh<V, void, 3, 1, triangle>\", !\"air.arg_name\", !\"m\"}\n\
                       !8 = !{!\"air.mesh_type_info\", !9, !1, i32 3, i32 1, !\"air.triangle\"}\n\
                       !9 = !{!10}\n\
                       !10 = !{!\"air.position\", !\"air.arg_type_name\", !\"float4\", \
                       !\"air.arg_name\", !\"position\"}\n!1 = !{}\n";
        if !with_object {
            return format!("{mesh}!air.mesh = !{{!5}}\n{mesh_md}");
        }
        format!(
            "define void @obj(ptr addrspace(6) %p, ptr addrspace(3) %g) {{\n  \
             call void @air.set_threadgroups_per_grid_mesh_properties(ptr addrspace(3) %g, \
             <3 x i32> <i32 1, i32 1, i32 1>)\n  ret void\n}}\n\
             declare void @air.set_threadgroups_per_grid_mesh_properties(ptr addrspace(3), \
             <3 x i32>)\n{mesh}\
             !air.object = !{{!0}}\n!air.mesh = !{{!5}}\n\
             !0 = !{{ptr @obj, !1, !2}}\n!2 = !{{!3, !4}}\n\
             !3 = !{{i32 0, !\"air.payload\", !\"air.arg_type_size\", i32 16, \
             !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"Pay\", \
             !\"air.arg_name\", !\"p\"}}\n\
             !4 = !{{i32 1, !\"air.mesh_grid_properties\", !\"air.arg_type_name\", \
             !\"mesh_grid_properties\", !\"air.arg_name\", !\"g\"}}\n{mesh_md}"
        )
    }

    /// Object and mesh functions are rows like any other, read through the one stage vocabulary
    /// (`case::Stage`): a public module declaring one of each is two entry rows of those stages,
    /// and a private module declaring a mesh function is its shard's row. A cold census reflects
    /// and classifies all three: the object function draws with the generated launch observer of
    /// its payload, and each mesh function, which sets its primitive's ID but no position and
    /// writes no device memory, carries `mesh_companion` (a fragment stage no executor generates),
    /// so it is classified and unresolved, never skipped; a warm pass selects and reads nothing.
    #[test]
    fn object_and_mesh_rows_are_indexed_and_classified() {
        let scratch = ScratchDir::new("census-mesh-rows").unwrap();
        let root = scratch.path();
        let fixtures = root.join("public");
        std::fs::create_dir_all(fixtures.join("engine")).unwrap();
        let pair = mesh_module(true);
        std::fs::write(fixtures.join("engine/mesh_pair.ll"), &pair).unwrap();
        let public = crate::source::public_sources_under(&fixtures).unwrap();
        let pair_hash = sha256_bytes(pair.as_bytes());
        assert_eq!(
            public
                .rows
                .iter()
                .map(|row| (
                    row.stage.as_str(),
                    row.entry.as_str(),
                    row.key().into_owned()
                ))
                .collect::<Vec<_>>(),
            [
                (
                    "Object",
                    "obj",
                    crate::source::entry_row_key(&pair_hash, "obj")
                ),
                (
                    "Mesh",
                    "msh",
                    crate::source::entry_row_key(&pair_hash, "msh")
                ),
            ]
        );

        let text = mesh_module(false);
        let private = SourceRow {
            air_sha256: sha256_bytes(text.as_bytes()),
            stage: crate::case::Stage::Mesh.metadata_label().into(),
            entry: "msh".into(),
            air_ll: text,
            blob_b64: None,
            lib_sha256s: vec!["owned-synthetic".into()],
            label: "private/mesh".into(),
        };
        private.validate().unwrap();
        write_source_shards(root, vec![private.clone()]).unwrap();

        let index = root.join("index.sqlite");
        rebuild_index_with(root, &index, &public).unwrap();
        let stages = Connection::open(&index)
            .unwrap()
            .prepare("SELECT stage, entry FROM sources ORDER BY stage, entry")
            .unwrap()
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
            .unwrap()
            .collect::<Result<Vec<(String, String)>, _>>()
            .unwrap();
        assert_eq!(
            stages,
            [
                ("Mesh".to_owned(), "msh".to_owned()),
                ("Mesh".to_owned(), "msh".to_owned()),
                ("Object".to_owned(), "obj".to_owned()),
            ]
        );

        let census = |all| {
            run_authoring_capability_census(root, &index, &public, usize::MAX, 2, all).unwrap()
        };
        let cold = census(true);
        assert_eq!(cold.selected, 3);
        assert_eq!(cold.summary.total_sources, 3);
        assert_eq!(cold.summary.classified_sources, 3);
        assert_eq!(cold.summary.remaining_sources(), 0);
        assert_eq!(cold.summary.unresolved_sources, 2, "{:?}", cold.summary);
        assert_eq!(
            cold.summary.requirements,
            BTreeMap::from([(ToolingRequirement::MeshCompanion, 2)]),
            "{:?}",
            cold.summary
        );
        assert!(
            cold.summary.unrecognized_air_intrinsics.is_empty(),
            "{:?}",
            cold.summary
        );
        assert_eq!(cold.read.source_shards_opened, 1);

        let warm = census(false);
        assert_eq!(warm.selected, 0);
        assert_eq!(warm.read, IndexedSourceReadStats::default());
        assert_eq!(warm.summary, cold.summary);
    }

    /// A census over a synthetic index whose public fixtures include a module with two entries:
    /// each entry is its own row under its own key, the module itself is no row, and a complete
    /// cold census classifies every row and closes with nothing excluded. A warm pass selects and
    /// reads nothing; an exact entry row is read from the checkout without opening a shard and a
    /// private row from its own shard only; a case over one entry is indexed under that entry's
    /// row.
    #[test]
    fn a_census_over_entry_rows_closes_and_reads_only_what_it_selects() {
        let scratch = ScratchDir::new("census-entry-rows").unwrap();
        let root = scratch.path();
        let fixtures = root.join("public");
        std::fs::create_dir_all(fixtures.join("engine")).unwrap();
        std::fs::write(fixtures.join("engine/pair.ll"), kernels(false)).unwrap();
        std::fs::write(fixtures.join("single.ll"), kernels(true)).unwrap();
        let public = crate::source::public_sources_under(&fixtures).unwrap();
        let pair = sha256_bytes(kernels(false).as_bytes());
        let single = sha256_bytes(kernels(true).as_bytes());
        let entry_key = |entry: &str| crate::source::entry_row_key(&pair, entry);
        assert_eq!(
            public
                .rows
                .iter()
                .map(|row| row.key().into_owned())
                .collect::<Vec<_>>(),
            [entry_key("first"), entry_key("second"), single.clone()]
        );

        // Two private rows of the single-entry module's text, in different shards.
        let base = SourceRow {
            lib_sha256s: vec!["owned-synthetic".into()],
            ..public.rows[2].clone()
        };
        let mut private = Vec::new();
        for note in 0.. {
            let row = variant(
                &base,
                &format!("private {note}"),
                &format!("private/{note}"),
            );
            let shard = shard_index_for_hash(&row.air_sha256).unwrap();
            if private
                .iter()
                .all(|other: &SourceRow| shard_index_for_hash(&other.air_sha256).unwrap() != shard)
            {
                private.push(row);
            }
            if private.len() == 2 {
                break;
            }
        }
        write_source_shards(root, private.clone()).unwrap();

        // A case over the module's second entry.
        let mut case =
            crate::case::attachmentless_fragment_test_case(pair.clone(), "second".into());
        case.stage = crate::case::Stage::Kernel;
        case.draw = None;
        case.dispatch = Some(crate::case::Dispatch {
            grid: [1; 3],
            threads_per_threadgroup: [1; 3],
            stage_in_region: None,
        });
        case.case_id = case.computed_case_id().unwrap();
        crate::store::CorpusStore::new(root)
            .put_case(case.clone())
            .unwrap();

        let index = root.join("index.sqlite");
        rebuild_index_with(root, &index, &public).unwrap();
        let rows = sources(&index)
            .into_iter()
            .map(|(key, _, _)| key)
            .collect::<BTreeSet<_>>();
        let mut expected = BTreeSet::from([entry_key("first"), entry_key("second"), single]);
        expected.extend(private.iter().map(|row| row.air_sha256.clone()));
        assert_eq!(rows, expected);
        assert!(!rows.contains(&pair), "the module itself is no row");
        let case_rows = Connection::open(&index)
            .unwrap()
            .prepare("SELECT air_sha256 FROM cases")
            .unwrap()
            .query_map([], |row| row.get::<_, String>(0))
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(case_rows, [entry_key("second")]);

        let census = |all| {
            run_authoring_capability_census(root, &index, &public, usize::MAX, 2, all).unwrap()
        };
        let cold = census(true);
        assert!(cold.public.is_unchanged());
        assert_eq!(cold.selected, 5);
        assert_eq!(cold.summary.total_sources, 5);
        assert_eq!(cold.summary.classified_sources, 5);
        assert_eq!(cold.summary.remaining_sources(), 0);
        assert_eq!(cold.summary.unresolved_sources, 0, "{:?}", cold.summary);
        assert_eq!(cold.summary.closure_gap(), None);
        assert_eq!(cold.read.rows, 5);
        assert_eq!(cold.read.source_shards_opened, 2);
        assert_eq!(cold.read.repair_shards_scanned, 0);

        let warm = census(false);
        assert!(warm.public.is_unchanged());
        assert_eq!(warm.selected, 0);
        assert_eq!(warm.read, IndexedSourceReadStats::default());
        assert_eq!(warm.summary, cold.summary);

        let read = |keys: &[String]| {
            let mut rows = Vec::new();
            let stats = for_each_indexed_source_analysis_in(root, &index, &public, keys, |row| {
                rows.push(row);
                Ok(())
            })
            .unwrap();
            (rows, stats)
        };
        let (rows, stats) = read(&[entry_key("second")]);
        assert_eq!(
            rows.iter()
                .map(|row| (row.entry.as_str(), row.key().into_owned()))
                .collect::<Vec<_>>(),
            [("second", entry_key("second"))]
        );
        assert_eq!(
            (stats.source_shards_opened, stats.source_bytes_read),
            (0, 0)
        );
        let (rows, stats) = read(std::slice::from_ref(&private[0].air_sha256));
        assert_eq!(rows, [private[0].clone()]);
        assert_eq!(stats.source_shards_opened, 1);
        assert_eq!(stats.repair_shards_scanned, 0);
    }
}
